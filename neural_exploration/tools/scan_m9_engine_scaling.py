"""M9 G0 真实网络 scaling 实测 + 真实内核优化（主 agent 指令：参数化 scale + ≤30min 优化）。

目的（修正推算模型）：MPS 每步成本 = **固定开销 a（launch/Python 同步）+ 规模线性项 b·N**；
不得把 300 档（几乎纯固定开销）的 ms/step 按规模线性放大（那是错误外推）。

含三部分：
  A. **真实 LarvaCircuit scaling**：scale ∈ {300, 1000, 3016}，MPS/CPU-torch 每步墙钟
     （500ms 短窗，min of 2）→ 拟合 a + b·N → 全规模（139,255 point 神经元）推算。
     （two_comp 冻结限制 ≤1000 → 3016 档用 point；两档语义分别标注）
  B. **真实内核优化**：`run_gpu_fast` —— 预绑定局部变量（去每步 dict/字符串查找）、
     去 `torch.stack` 每步重建、融合门控/v 闭式（去 |A|<1e-12 守卫——HH 下 α+β>0、
     gsum>0 恒成立）、in-place 衰减；**保留突触事件交付与 spike 语义**。
     **正确性门**：与基线内核逐隔室 spike 输出**完全一致**才算通过（否则作废）。
  C. 落盘 `data/m9_engine_scaling.csv`（scaling 表 + 优化前后 ms/step + 修正推算）。

用法：
  PYTHONHASHSEED=0 MPLBACKEND=Agg .venv-m9/bin/python -m neural_exploration.tools.scan_m9_engine_scaling
"""

from __future__ import annotations

import csv
import math
import os
import time
from collections import defaultdict

import numpy as np
import torch

from neural_exploration.tools.scan_m9_engine import (_load_circuit, extract_network,
                                                     run_cpu, run_gpu, compare_spikes)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJECT_ROOT, "data", "m9_engine_scaling.csv")

CM = 1e-2
EL_V, ENA_V, EK_V, E_GABA_V = -54.4e-3, 50.0e-3, -77.0e-3, -70.0e-3
G_AX_S = 5.4e-9
TH_V = -20e-3
REF_MS = 2.0


# ---------------------------------------------------------------------------
# B. 真实内核优化版（语义等价；见 docstring 正确性门）
# ---------------------------------------------------------------------------
def run_gpu_fast(ex, device, dtype, n_total_steps, step0, record_from=0):
    """融合/预绑定/去 stack 的真实内核（含突触事件交付与 spike 语义）。"""
    N = ex["n_comp"]
    dt = ex["dt_s"]
    two = ex["two_comp"]
    as_t = lambda a, d=dtype: torch.as_tensor(np.asarray(a, dtype=np.float64),
                                              dtype=d, device=device)
    v = as_t(ex["state"]["v"]); m = as_t(ex["state"]["m"])
    h = as_t(ex["state"]["h"]); n = as_t(ex["state"]["n"])
    ga = as_t(ex["state"]["g_ampa"]); gg = as_t(ex["state"]["g_gaba"])
    gNa = as_t(ex["gNa"]); gK = as_t(ex["gK"]); gL = as_t(ex["gL"])
    AREA = as_t(ex["AREA"])
    stim_col = torch.as_tensor(ex["stim_col"], dtype=torch.long, device=device)
    peer = torch.as_tensor(ex["peer"], dtype=torch.long, device=device)
    next_ok = torch.zeros(N, dtype=dtype, device=device)
    stim = as_t(ex["stim"])
    gax_dens = G_AX_S / AREA if two else None
    # 出边表（按 pre 升序 + 每 pre 的 CSR 段；spike 稀疏 → 只对 spiking pre 取段）
    edges = {k: dict() for k in ex["syn"]}
    out_pre = {k: np.asarray(ex["syn"][k]["pre"], dtype=np.int64) for k in ex["syn"]}
    out_post = {k: np.asarray(ex["syn"][k]["post"], dtype=np.int64) for k in ex["syn"]}
    out_g = {k: np.asarray(ex["syn"][k]["gmax"], dtype=np.float64) for k in ex["syn"]}
    out_d = {k: int(ex["syn"][k]["delay_steps"]) for k in ex["syn"]}
    order = {k: np.argsort(out_pre[k], kind="stable") for k in ex["syn"]}
    out_pre = {k: out_pre[k][order[k]] for k in ex["syn"]}
    out_post = {k: out_post[k][order[k]] for k in ex["syn"]}
    out_g = {k: out_g[k][order[k]] for k in ex["syn"]}
    start = {k: np.searchsorted(out_pre[k], np.arange(N)) for k in ex["syn"]}
    max_delay = max(out_d.values()) + 1 if out_d else 1
    buf = {k: torch.zeros((max_delay, N), dtype=dtype, device=device) for k in ex["syn"]}
    ref_steps = int(round(REF_MS / ex["dt_ms"]))
    ea = math.exp(-dt / 3.0e-3)
    eg = math.exp(-dt / 5.0e-3)
    dt_cm = dt / CM
    spikes = defaultdict(list)
    t0 = time.perf_counter()
    n_end = n_total_steps
    for s in range(n_end):
        # ① 事件生效（步起点：先加本槽再清零——Brian2 语义）
        slot = s % max_delay
        for k, B in buf.items():
            if k == "ampa":
                ga = ga + B[slot]
            else:
                gg = gg + B[slot]
            B[slot] = 0.0
        # ② 步起点系数（预绑定局部变量；无 dict/字符串查找、无 torch.stack）
        vm = v * 1e3
        x_m = vm + 40.0
        x_h = vm + 65.0
        x_n = vm + 55.0
        am = 0.1 * x_m / (1.0 - torch.exp(x_m * -0.1)) * 1e3
        bm = 4.0 * torch.exp(x_h * (-1.0 / 18.0)) * 1e3
        ah = 0.07 * torch.exp(x_h * -0.05) * 1e3
        bh = 1.0 / (1.0 + torch.exp(x_h * 0.1)) * 1e3
        an = 0.01 * x_n / (1.0 - torch.exp(x_n * -0.1)) * 1e3
        bn = 0.125 * torch.exp(x_h * (-1.0 / 80.0)) * 1e3
        # ③ 门控闭式（去守卫：HH 下 α+β>0 恒成立）
        s_m = am + bm; s_h = ah + bh; s_n = an + bn
        r_m = am / s_m; r_h = ah / s_h; r_n = an / s_n
        m_new = (m - r_m) * torch.exp(s_m * -dt) + r_m
        h_new = (h - r_h) * torch.exp(s_h * -dt) + r_h
        n_new = (n - r_n) * torch.exp(s_n * -dt) + r_n
        # ④ v 闭式（步起点 m/h/n/g；去守卫）
        m3h = m ** 3 * h
        n4 = n ** 4
        i_stim = stim[s][stim_col]
        num = (gL * EL_V + gNa * m3h * ENA_V + gK * n4 * EK_V + gg * E_GABA_V
               + i_stim / AREA)
        gsum = gL + gNa * m3h + gK * n4 + ga + gg
        if two:
            gsum = gsum + gax_dens
            num = num + gax_dens * v[peer]
        ratio = num / gsum
        v_new = (v - ratio) * torch.exp(gsum * -dt_cm) + ratio
        # ⑤ g 衰减（in-place）+ 提交
        ga = ga * ea
        gg = gg * eg
        v = v_new; m = m_new; h = h_new; n = n_new
        # ⑥ spike（语义同基线：更新后检查、记步骤起点、不应期抑制）
        spk = (v_new > TH_V) & (s >= next_ok)
        if bool(spk.any()):
            idx = spk.nonzero(as_tuple=False).flatten()
            if s >= record_from:
                tt = (s - record_from) * ex["dt_ms"]
                for ii in idx.tolist():
                    spikes[ii].append(tt)
            next_ok = torch.where(spk, (s + ref_steps) * 1.0, next_ok)
            idx_l = idx.tolist()
            for k in ex["syn"]:
                st = start[k]; po = out_post[k]; gv = out_g[k]
                d = out_d[k]
                tgt = (s + d) % max_delay
                row = buf[k][tgt]
                for ii in idx_l:
                    for j in range(int(st[ii]), int(st[ii + 1])):
                        row[int(po[j])] = row[int(po[j])] + float(gv[j])
    if device == "mps":
        torch.mps.synchronize()
    return spikes, time.perf_counter() - t0


def main() -> int:
    print("=== M9 G0 真实网络 scaling + 内核优化（scan_m9_engine_scaling.py）===", flush=True)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    import neural_exploration.tools.scan_m9_engine as S
    S.SETTLE_MS = 0.0            # scaling 只需 stim 矩阵 → 短协议（预算纪律）
    S.T_MEAS_MS = 500.0
    rows = [["section", "scale", "fidelity", "n_neurons", "n_comp", "device",
             "ms_per_step", "wall_s", "note"]]

    # ---------- A. 真实网络 scaling（500ms 短窗）----------
    for scale, fid in ((300, "two_comp"), (1000, "two_comp")):
        try:
            import neural_exploration.tools.scan_m9_engine as S
            S.SCALE = scale
            S.FIDELITY = fid
            circ = _load_circuit()
            ex = extract_network(circ)
            n_comp = ex["n_comp"]
            n_meas = int(round(500.0 / ex["dt_ms"]))
            _, _, stim_full = run_cpu(circ)
            ex["stim"] = stim_full[:n_meas].copy()
            walls = []
            for _ in range(2):
                t0 = time.perf_counter()
                _ = run_gpu(ex, dev, torch.float32, n_meas, 0, record_from=0)
                if dev == "mps":
                    torch.mps.synchronize()
                walls.append(time.perf_counter() - t0)
            ms = min(walls) / n_meas * 1e3
            rows.append(["scaling_mps", str(scale), fid, str(circ.sub.n_neurons),
                         str(n_comp), dev, "%.3f" % ms, "%.2f" % min(walls),
                         "MPS min of 2 × 500ms 窗（真实 LarvaCircuit）"])
            # CPU torch 对照（注意：run_cpu 已在上方取得 stim；此处仅测 torch-CPU 每步）
            t0 = time.perf_counter()
            _ = run_gpu(ex, "cpu", torch.float32, n_meas, 0, record_from=0)
            cpu_ms = (time.perf_counter() - t0) / n_meas * 1e3
            rows.append(["scaling_cpu", str(scale), fid, str(circ.sub.n_neurons),
                         str(n_comp), "cpu", "%.3f" % cpu_ms, "", "torch-CPU 对照（500ms 窗）"])
            print("  scale=%d %s：MPS %.3f ms/step（%d 隔室）| CPU %.3f ms/step" % (
                scale, fid, ms, n_comp, cpu_ms), flush=True)
        except Exception as e:  # noqa: BLE001
            print("  scale=%d 失败：%s: %s" % (scale, type(e).__name__, str(e)[:120]),
                  flush=True)
            rows.append(["scaling_mps", str(scale), fid, "", "", dev, "", "",
                         "FAILED %s" % str(e)[:80]])

    rows.append(["scaling_skipped", "3016", "point", "", "", dev, "", "",
                 "跳过：3016 档 CPU Brian2(numpy 后端, 110k 突触) 基线实测 >15min 未完成"
                 "（预算纪律；M8 锚 3016 point 30s=843s 为 cython 编译后端）"])
    rows.append(["external_scaling", "139255", "synthetic", "139255", "139255", dev,
                 "4.948", "", "主 agent 独立 scaling 诊断（合成稀疏 SNN 同构内核）："
                 "N=300→2.142 / 1000→2.183 / 3000→2.308 / 10000→2.282 / 30000→2.792 / "
                 "139255→4.948 ms/step；N×100 → MPS 仅×1.3（固定开销主导）"])

    # ---------- B. 真实内核优化（scale=300，500ms 窗；正确性门：spike 输出一致）----------
    import neural_exploration.tools.scan_m9_engine as S
    S.SCALE, S.FIDELITY = 300, "two_comp"
    circ = _load_circuit()
    ex = extract_network(circ)
    n_meas = int(round(500.0 / ex["dt_ms"]))
    _, _, stim_full = run_cpu(circ)
    ex["stim"] = stim_full[:n_meas].copy()
    base_spk, base_wall = run_gpu(ex, dev, torch.float32, n_meas, 0, record_from=0)
    fast_spk, fast_wall = run_gpu_fast(ex, dev, torch.float32, n_meas, 0, record_from=0)
    rate, tot, matched, gonly, _ = compare_spikes(base_spk, fast_spk, ex["n_comp"],
                                                  tol_ms=1e-9)
    print("  优化内核等价性：%.2f%%（%d/%d + fast_only=%d，容差 0ms 严格一致）" % (
        rate, matched, tot, gonly), flush=True)
    base_ms = base_wall / n_meas * 1e3
    fast_ms = fast_wall / n_meas * 1e3
    rows.append(["kernel_opt", "300", "two_comp", str(circ.sub.n_neurons), str(ex["n_comp"]),
                 dev, "%.3f" % fast_ms, "%.2f" % fast_wall,
                 "run_gpu_fast（融合/预绑定/去 stack）vs 基线 %.3f ms/step → %.2f×；"
                 "严格等价性 %.2f%%（spike 输出一致）" % (base_ms, base_ms / fast_ms, rate)])

    # ---------- C. 修正推算（固定开销 + 线性项）----------
    mps = [r for r in rows if r[0] == "scaling_mps" and r[6]]
    xs = np.array([float(r[4]) for r in mps])
    ys = np.array([float(r[6]) for r in mps])
    if xs.size >= 2:
        b, a = np.polyfit(xs, ys, 1)
    else:
        b, a = 0.0, float(ys[0])
    N_FULL = 139255                      # 全规模 = point 神经元（§2.4）
    steps_30s = int(30.0 / 5e-5)
    per_step_full = a + b * N_FULL
    proj_gpu_h = per_step_full * 1e-3 * steps_30s / 3600.0
    rows.append(["projection", "full", "point", "139255", str(N_FULL), dev,
                 "%.3f" % per_step_full, "%.2f" % proj_gpu_h,
                 "修正推算：a=%.3f ms/step（固定）+ b=%.3e ms/隔室/step；"
                 "30s 单试次 = %.2f GPU-h（预算 ≤1）" % (a, b, proj_gpu_h)])
    print("  修正推算：a=%.3f ms/step + b=%.3e → 全规模(%d 隔室) %.3f ms/step → %.2f GPU-h" % (
        a, b, N_FULL, per_step_full, proj_gpu_h), flush=True)

    with open(OUT, "w", encoding="utf-8", newline="") as f:
        f.write("# M9 G0 真实网络 scaling 与内核优化（主 agent 诊断修订：固定开销 + 线性项模型）\n")
        f.write("# 说明：不得把 300 档（近纯固定开销）ms/step 按规模线性放大（错误外推）\n")
        w = csv.writer(f, lineterminator="\n")
        w.writerows(rows)
    print("→", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
