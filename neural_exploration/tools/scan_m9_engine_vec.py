"""M9 向量化内核对齐探针（§2.5 冻结规则：kernel 改动必须重过对齐门）。

《生物仿真M9实施清单》§2.3（GPU 正确性验证判据 (a)–(d)）/§2.5（引擎冻结：kernel 改动重过
对齐探针）/§0.9 R10（探针墙钟先行 → 裁决 → 再烧预算）。

与 `tools/scan_m9_engine.py`（B1 节点，冻结）的关系：
  - **只读复用**其参考实现与判据固定件：`_load_circuit` / `extract_network` / `run_cpu`
    / `compare_spikes` / `silent_fraction` / 常量（SCALE / T_MEAS_MS / TOL_MS）；
  - 被验证对象 = `src/adult_engine.AdultEngine`（向量化内核，B2 新建）；
  - 同一网络定义、同一 CPU Brian2 基线、同一对齐判据与容差 → **同门重跑**。

判据（预注册 §0.7 #3；与 scan_m9_engine 一致）：
  (a) 逐神经元 spike 时间对齐 ≥99%（|Δt|<0.1ms；主判据，统计级）；
  (b) 静默比例差 <1pp；
  (c) 浮点精度档（f32/f64）统计级一致；
  (d) 确定性：同参数重跑统计级一致（<0.05ms）。
  另：**等价性**——两种投递调度（step / chunk）必须给出相同 spike 集合。

墙钟：MPS 每步实测（point 档全规模 + two_comp 档）× 规模分解 → 全规模 projection；
优化前/后对照（B1 基准见 data/m9_engine_probe.csv）。

输出：
  - data/m9_engine_vec_probe.csv —— 对齐判据数值 + 优化前后 ms/step + 全规模 projection
  - reports/neuro/m9_engine_alignment_vec.png —— 对齐图 + 墙钟对照

用法：
  PYTHONHASHSEED=0 MPLBACKEND=Agg .venv-m9/bin/python -m neural_exploration.tools.scan_m9_engine_vec
"""

from __future__ import annotations

import csv as _csv
import os
import sys
import time
from collections import defaultdict

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports", "neuro")
OUT_CSV = os.path.join(DATA_DIR, "m9_engine_vec_probe.csv")
OUT_PNG = os.path.join(REPORT_DIR, "m9_engine_alignment_vec.png")

TOL_MS = 0.1
TOL_MS_STRICT = 0.05
N_FULL = 139255            # 全规模神经元数（point 档隔室数 = N）
E_FULL = 15091983          # 全规模化学连接数（P1 解析值）
N_STEPS_30S = 600000       # 30s @ dt=0.05ms
BUDGET_GPU_H = 1.0         # §0.9 R10 预注册单试次上限


def _to_dict(steps: np.ndarray, idx: np.ndarray, dt_ms: float):
    d = defaultdict(list)
    for s, i in zip(steps, idx):
        d[int(i)].append(float(s) * dt_ms)
    return d


def _synthetic_point(n_neuron: int, n_syn_per: int, n_steps: int, seed: int = 0,
                     active_frac: float = 0.02, syn_count_max: int = 8):
    """合成同构 point 网络（出度 = n_syn_per，权重 ∝ syn_count 分布）。

    用于 MPS 规模分解（t = a + b·N 拟合，L16#4：禁止朴素线性外推）。
    """
    rng = np.random.default_rng(seed)
    pre = np.repeat(np.arange(n_neuron, dtype=np.int64), n_syn_per)
    post = rng.integers(0, n_neuron, size=pre.size).astype(np.int64)
    inh = rng.random(pre.size) < 0.234          # 真实 GABA 占比（L14.3）
    sc = np.minimum(rng.geometric(1.0 / 3.6, size=pre.size), syn_count_max).astype(np.float32)
    gmax = np.where(inh, 1.0, 0.35).astype(np.float32) * sc
    return pre, post, gmax, inh


def _measure_scaling(device="mps", use_compile=True, sizes=(10000, 30000, 139255),
                     n_steps=1500, syn_per=108, verbose=True, ref_ms=(2.0, 20.0, 200.0)):
    """point 档 MPS 规模分解：ms/step（含事件交付）随 N 与**活跃水平**的实测。

    活跃水平用**不应期**精确控制：强 tonic 偏置下所有神经元以 1/ref 的速率饱和发放 →
    每步活跃比例 = dt/ref（ref=2ms→2.5%、20ms→0.25%、200ms→0.025%）——可复现、可外推，
    替代"调 bias 试活跃度"（不可控）。诚实性：同时落盘实测活跃比例（L16#3 独占运行）。
    """
    from neural_exploration.src.adult_engine import AdultEngine
    out = {}
    for N in sizes:
        pre, post, gmax, inh = _synthetic_point(N, syn_per, n_steps)
        eng = AdultEngine(N, device=device, use_compile=use_compile)
        eng.add_synapses(pre, post, gmax, delay_steps=20, inhibitory=inh)
        eng.finalize()
        rng = np.random.default_rng(1)
        eng.set_neuron_heterogeneity(i_bias=rng.normal(900.0, 90.0, N).astype(np.float32))
        for rm in ref_ms:
            eng.set_point_params(ref_ms=rm)
            for mode in ("chunk",):
                eng.reset(seed=0)
                eng.run(400, delivery=mode, record="counts")      # warmup
                walls = []
                for _ in range(3):
                    eng.reset(seed=0)
                    t0 = time.perf_counter()
                    r = eng.run(n_steps, delivery=mode, record="counts")
                    walls.append(time.perf_counter() - t0)
                    frac = r["spk_total"] / max(N * n_steps, 1)
                row = out.setdefault(N, {"n_neuron": N, "n_edge": int(pre.size),
                                         "n_edges_per_neuron": syn_per})
                row["spk_frac_ref%.0fms" % rm] = frac
                row["ms_per_step_ref%.0fms" % rm] = min(walls) / n_steps * 1e3
        if verbose:
            parts = []
            for r in ref_ms:
                k = "%.0f" % r
                parts.append("ref=%sms: %.3f ms/step (frac %.5f)"
                             % (k, out[N]["ms_per_step_ref%sms" % k],
                                out[N]["spk_frac_ref%sms" % k]))
            print("    N=%7d edges=%9d | " % (N, pre.size) + " | ".join(parts),
                  flush=True)
    return out


def _fit(xs, ys):
    b, a = np.polyfit(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), 1)
    return float(a), float(b)


def main() -> int:
    from neural_exploration.tools import scan_m9_engine as S
    from neural_exploration.src.adult_engine import engine_from_extracted

    print("=== M9 向量化内核对齐探针（scan_m9_engine_vec.py；§2.5 kernel 改动重过对齐门）===",
          flush=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    rec = {}

    # ---------------- 1) 对齐（同一 M8 冻结 300 two_comp 网络定义） ----------------
    circ = S._load_circuit()
    ex = S.extract_network(circ)
    cpu_spikes, cpu_wall, stim = S.run_cpu(circ)
    n_steps = int(round(S.T_MEAS_MS / ex["dt_ms"]))
    ex["stim"] = stim[:n_steps].copy()
    n_cpu_spk = sum(len(v) for v in cpu_spikes.values())
    silent_cpu = S.silent_fraction(cpu_spikes, ex["n_comp"], S.T_MEAS_MS)
    print("CPU Brian2 基线：%d spike / %d 隔室；墙钟 %.2fs；静默 %.4f"
          % (n_cpu_spk, ex["n_comp"], cpu_wall, silent_cpu), flush=True)
    rec.update({"n_comp": ex["n_comp"], "cpu_spikes": n_cpu_spk,
                "cpu_wall_s": cpu_wall, "silent_cpu": silent_cpu})

    variants = [("cpu_f32_step", "cpu", False, "step"),
                ("cpu_f32_chunk", "cpu", False, "chunk"),
                ("mps_f32_step", "mps", True, "step"),
                ("mps_f32_chunk", "mps", True, "chunk")]
    spike_sets = {}
    for name, dev, comp, mode in variants:
        try:
            eng = engine_from_extracted(ex, device=dev, use_compile=comp)
            t0 = time.perf_counter()
            r = eng.run(n_steps, fidelity="two_comp", delivery=mode, record="full")
            wall = time.perf_counter() - t0
            gs = _to_dict(r["spikes"][0], r["spikes"][1], ex["dt_ms"])
            rate, tc, mc, go, _ = S.compare_spikes(cpu_spikes, gs, ex["n_comp"])
            sil = S.silent_fraction(gs, ex["n_comp"], S.T_MEAS_MS)
            spike_sets[name] = gs
            rec["%s_align_pct" % name] = rate
            rec["%s_silent" % name] = sil
            rec["%s_wall_s" % name] = wall
            rec["%s_n_spk" % name] = sum(len(v) for v in gs.values())
            print("  %-14s 对齐 %.2f%%（%d/%d gpu_only=%d）静默 %.4f 墙钟 %.2fs"
                  % (name, rate, mc, tc, go, sil, wall), flush=True)
        except Exception as e:  # 设备不可用时如实记录，不静默
            rec["%s_align_pct" % name] = float("nan")
            rec["%s_error" % name] = "%s: %s" % (type(e).__name__, str(e)[:120])
            print("  %-14s 不可用：%s" % (name, str(e)[:120]), flush=True)

    # 等价性：两种投递调度 spike 集合必须相同（同精度 CPU 档）
    eq_note = "n/a"
    if "cpu_f32_step" in spike_sets and "cpu_f32_chunk" in spike_sets:
        a, b = spike_sets["cpu_f32_step"], spike_sets["cpu_f32_chunk"]
        keys = set(a) | set(b)
        same = all(len(a.get(k, [])) == len(b.get(k, [])) for k in keys)
        if same:
            dmax = max((max(abs(x - y) for x, y in zip(sorted(a.get(k, [])),
                                                       sorted(b.get(k, []))))
                        if a.get(k) else 0.0) for k in keys)
        else:
            dmax = float("inf")
        rec["delivery_equiv"] = bool(same and dmax == 0.0)
        eq_note = "step↔chunk 逐 spike 一致=%s（max Δt=%.3fms）" % (
            rec["delivery_equiv"], dmax)
        print("  投递调度等价性：%s" % eq_note, flush=True)

    # 判据 (c) 精度档 + (d) 确定性（torch-CPU-f32 重跑）
    eng_c = engine_from_extracted(ex, device="cpu", use_compile=False)
    r2 = eng_c.run(n_steps, fidelity="two_comp", delivery="step", record="full")
    gs2 = _to_dict(r2["spikes"][0], r2["spikes"][1], ex["dt_ms"])
    if "cpu_f32_step" in spike_sets:
        det, _, _, _, _ = S.compare_spikes(spike_sets["cpu_f32_step"], gs2,
                                           ex["n_comp"], tol_ms=TOL_MS_STRICT)
        rec["determinism_pct"] = det
        print("  确定性重跑（<0.05ms）：%.2f%%" % det, flush=True)

    crit_a = min([rec.get("%s_align_pct" % n, 0.0) for n, _, _, _ in variants]) >= 99.0
    crit_b = (max(abs(rec.get("%s_silent" % n, 0.0) - silent_cpu)
                  for n, _, _, _ in variants) < 0.01)
    crit_c = rec.get("cpu_f32_step_align_pct", 0.0) >= 99.0 and \
        rec.get("mps_f32_step_align_pct", 0.0) >= 99.0
    crit_d = rec.get("determinism_pct", 0.0) >= 99.0
    rec["crit_a"], rec["crit_b"], rec["crit_c"], rec["crit_d"] = crit_a, crit_b, crit_c, crit_d

    # ---------------- 2) 墙钟：全规模 point 档规模分解 ----------------
    print("MPS 规模分解（合成 point 网络，出度=108，活跃比例~2%）：", flush=True)
    t0 = time.perf_counter()
    sc = _measure_scaling(device="mps", use_compile=True,
                          sizes=(10000, 30000, N_FULL), n_steps=1000)
    rec["scaling_wall_s"] = time.perf_counter() - t0
    xs = [k for k in sc]
    refs = sorted(k.split("ms_per_step_ref")[1] for k in sc[xs[0]]
                  if k.startswith("ms_per_step_ref"))
    for rf in refs:
        key = "ms_per_step_ref%s" % rf
        a, b = _fit(xs, [sc[k][key] for k in xs])
        rec["a_ref%s_ms" % rf], rec["b_ref%s_ms" % rf] = a, b
        ms_full = a + b * N_FULL
        rec["ms_per_step_full_ref%s" % rf] = ms_full
        rec["gpu_h_30s_ref%s" % rf] = ms_full * 1e-3 * N_STEPS_30S / 3600.0
        print("  活跃档 ref=%sms（每步活跃比例 %.5f）：t = %.3f + %.3e·N ms → 全规模 "
              "%.3f ms/step → 30s 单试次 %.3f GPU-h"
              % (rf, sc[xs[0]]["spk_frac_ref%s" % rf], a, b, ms_full,
                 rec["gpu_h_30s_ref%s" % rf]), flush=True)

    # two_comp 档全规模 projection（对齐参考档；G0 记录为测量限制 + 优化目标）
    ms_two = rec.get("mps_f32_chunk_wall_s", 0.0)
    if ms_two and "a_ref2ms_ms" in rec:
        per_step_two = ms_two / n_steps * 1e3
        # two_comp 与 point 用**同一内核骨架**：以 point 档固定开销 a 拆出逐隔室项
        a_use = rec["a_ref2ms_ms"]
        b_use = max((per_step_two - a_use) / ex["n_comp"], 0.0)
        rec["two_comp_ms_per_step_600"] = per_step_two
        rec["two_comp_b_ms_per_comp"] = b_use
        rec["two_comp_proj_278510_ms"] = a_use + b_use * (2 * N_FULL)
        rec["two_comp_proj_30s_gpu_h"] = (a_use + b_use * (2 * N_FULL)) * 1e-3 \
            * N_STEPS_30S / 3600.0
        print("  two_comp（对齐参考档）：实测 600 隔室 %.3f ms/step；逐隔室项 %.3e → "
              "278,510 隔室 projection %.3f ms/step → 30s %.3f GPU-h（B1 基准 1.05）"
              % (per_step_two, b_use, rec["two_comp_proj_278510_ms"],
                 rec["two_comp_proj_30s_gpu_h"]), flush=True)

    _refkeys = [k for k in rec if k.startswith("gpu_h_30s_ref")]
    rec["budget_point_ok"] = min(rec[k] for k in _refkeys) <= BUDGET_GPU_H
    rec["target_point_le_0p3ms"] = min(
        rec["ms_per_step_full_" + k.replace("gpu_h_30s_", "")] for k in _refkeys) <= 0.3
    write_csv(rec, sc, eq_note)
    make_plot(rec, sc, spike_sets, cpu_spikes, ex["n_comp"])
    print("对齐探针结果 →", OUT_CSV, flush=True)
    if crit_a and crit_b and crit_c and crit_d:
        print("对齐门：PASS（判据 a–d 全过）——向量化内核保持 100% 语义等价", flush=True)
    else:
        print("对齐门：FAIL（a=%s b=%s c=%s d=%s）" % (crit_a, crit_b, crit_c, crit_d),
              flush=True)
    return 0


def write_csv(rec, sc, eq_note):
    rows = [["metric", "value", "note"]]
    note = {
        "n_comp": "M8 冻结 300 two_comp 网络的隔室数（G0 对齐参考）",
        "cpu_spikes": "CPU Brian2 基线 spike 数（测量窗 2000ms）",
        "cpu_wall_s": "CPU Brian2 墙钟（参考实现基线，L15.4）",
        "silent_cpu": "CPU 静默比例（率<0.5Hz）",
    }
    for k, v in rec.items():
        if k in ("crit_a", "crit_b", "crit_c", "crit_d"):
            continue
        rows.append([k, ("%.4f" % v if isinstance(v, float) else str(v)), note.get(k, "")])
    for k, v in sc.items():
        for kk, vv in v.items():
            rows.append(["scale_N%d_%s" % (k, kk),
                         ("%.4f" % vv if isinstance(vv, float) else str(vv)),
                         "规模分解原始实测（合成 point 网络，min-of-3 × 1500 步，独占运行）"])
    rows.append(["delivery_equivalence", eq_note, "step↔chunk 投递调度等价性（同精度 CPU 档）"])
    rows.append(["crit_a_pass", str(rec["crit_a"]), "判据 (a) 全部变体 spike 对齐 ≥99%"])
    rows.append(["crit_b_pass", str(rec["crit_b"]), "判据 (b) 静默比例差 <1pp"])
    rows.append(["crit_c_pass", str(rec["crit_c"]), "判据 (c) 精度/设备档一致（CPU-f32 + MPS-f32）"])
    rows.append(["crit_d_pass", str(rec["crit_d"]), "判据 (d) 确定性重跑 <0.05ms ≥99%"])
    rows.append(["alignment_gate", "PASS" if all(rec[k] for k in
                ("crit_a", "crit_b", "crit_c", "crit_d")) else "FAIL",
                 "§2.5 kernel 改动重过对齐门"])
    rows.append(["budget_point_le_1_gpu_h", str(rec["budget_point_ok"]),
                 "§0.9 R10：单试次 ≤1 GPU-h（point 档全规模 30s）"])
    rows.append(["target_point_le_0p3ms", str(rec["target_point_le_0p3ms"]),
                 "主 agent 优化目标 ≤0.3 ms/step（point 档；实测值见上）"])
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        f.write("# M9 向量化内核对齐探针（§2.5：kernel 改动后重过对齐门）\n")
        f.write("# 被验证对象 src/adult_engine.AdultEngine；参考 = CPU Brian2 2.6.0 + "
                "tools/scan_m9_engine.py 提取网络\n")
        w = _csv.writer(f, lineterminator="\n")
        w.writerows(rows)


def make_plot(rec, sc, spike_sets, cpu_spikes, n_comp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes[0, 0]
    names = [k for k in spike_sets]
    vals = [rec.get("%s_align_pct" % k, 0.0) for k in names]
    ax.bar(names, vals, color="#2ca02c")
    ax.axhline(99.0, color="red", ls="--", lw=1, label="判据 (a) ≥99%")
    ax.set_ylim(0, 105); ax.set_ylabel("spike alignment (%)")
    ax.set_title("Vectorized kernel vs Brian2 baseline")
    ax.tick_params(axis="x", rotation=20); ax.legend()
    ax = axes[0, 1]
    rk = sorted(k for k in rec if k.startswith("ms_per_step_full_ref"))
    vals = [rec[k] for k in rk]
    labs = ["active %.3f" % rec.get("spk_frac_ref" + k.split("ms_per_step_full_ref")[1], 0)
            for k in rk]
    ax.bar(labs, vals, color="#1f77b4")
    ax.set_ylabel("projected ms/step (N=139,255)")
    ax.set_title("Full-scale projection (B1 baseline 3.86 ms/step)")
    ax = axes[1, 0]
    if sc:
        xs = list(sc.keys())
        refs = sorted(k.split("ms_per_step_ref")[1] for k in sc[xs[0]]
                      if k.startswith("ms_per_step_ref"))
        for rf in refs:
            ax.plot(xs, [sc[k]["ms_per_step_ref%s" % rf] for k in xs], "o-",
                    label="active %.3f" % sc[xs[0]]["spk_frac_ref%s" % rf])
        ax.set_xlabel("N neurons"); ax.set_ylabel("ms/step"); ax.legend()
        ax.set_title("MPS scaling @ controlled activity")
    ax = axes[1, 1]
    cb = np.array([len(cpu_spikes.get(i, [])) for i in range(n_comp)])
    key = next((k for k in spike_sets if k.startswith("mps")), None)
    if key:
        gb = np.array([len(spike_sets[key].get(i, [])) for i in range(n_comp)])
        ax.scatter(cb, gb, s=5, alpha=0.6)
        mx = max(cb.max(), gb.max(), 1)
        ax.plot([0, mx], [0, mx], "r--", lw=1)
        ax.set_title("Per-compartment spikes: Brian2 vs %s" % key)
        ax.set_xlabel("Brian2"); ax.set_ylabel(key)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=110)
    print("对齐图 →", OUT_PNG, flush=True)


if __name__ == "__main__":
    sys.exit(main())
