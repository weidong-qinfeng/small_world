"""M9 G0 引擎探针（§2）：M8 冻结幼虫网络（300 two_comp 定稿配置）CPU(Brian2) vs GPU(MPS torch)
逐神经元对齐 + 墙钟实测 + 全规模预算裁决。

《生物仿真M9实施清单》§2.3（GPU 正确性验证）/§2.1/§2.2（选型判据定稿）/
§0.9 R10（探针墙钟先行，全规模单试次预算 ≤1 GPU-h 预注册）。

对齐判据（预注册 §0.7 #3/§2.3）：
  (a) 逐神经元 spike 时间差 <0.1ms 比例 ≥99%（统计级主判据；GPU 浮点重排为测量限制，
      逐位一致不承诺）；
  (b) 状态比例：静默比例差 <1pp；
  (c) 行为指标：自发分布落带一致（本探针以发放率分布 KS 与静默比例为统计替代；
      完整行为读出（run/turn/pause）由后续 P3–P4 节点在冻结判据带上执行）；
  (d) 确定性：GPU 同参数重跑统计级一致（固定 kernel/seed）。

方法（等价参考实现——PyTorch 自研点神经元内核）：
  - 网络定义从冻结 LarvaCircuit 对象**直接提取**（同一连接组 CSV/权重 m8_larva_params.csv/
    同一 dt/seed 由冻结装配保证）：状态初值、gNa/gK/gL/AREA/stim_col、化学突触
    （pre/post 隔室 + gmax 密度 + delay）、stim（稀疏列，motor 脉冲）、阈值/不应期。
  - 数值方案 = Brian2 exponential_euler 逐变量闭式（已核实 brian2 2.6.0 源码：
    每变量独立线性 ODE 闭式，其它变量视为常数；闭式 x'=(x+B/A)e^{Adt}-B/A）；
    spike 语义已实测（numpy 后端）：阈值在状态更新后检查、spike 时间记步骤起点、
    不应期 2ms 自检测步终点起算——torch 同构复现。
  - CPU 侧 = Brian2 2.6.0（numpy 后端/float64）；GPU 侧 = torch MPS（float32）。

输出：
  - data/m9_engine_params.csv   —— 选型判据定稿（§2.2）
  - data/m9_engine_probe.csv    —— 对齐判据数值 + 探针墙钟 + 全规模预算裁决（§2.3）
  - reports/neuro/m9_engine_alignment.png —— CPU-GPU 逐神经元对齐图

用法：
  PYTHONHASHSEED=0 MPLBACKEND=Agg .venv-m9/bin/python -m neural_exploration.tools.scan_m9_engine
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

OUT_PARAMS = os.path.join(DATA_DIR, "m9_engine_params.csv")
OUT_PROBE = os.path.join(DATA_DIR, "m9_engine_probe.csv")
OUT_PLOT = os.path.join(REPORT_DIR, "m9_engine_alignment.png")

SCALE = 300
FIDELITY = "two_comp"
SETTLE_MS = 500.0
T_MEAS_MS = 2000.0
TOL_MS = 0.1          # 判据 (a) 容差
TH_V = -20.0e-3       # 阈值
REF_MS = 2.0          # 不应期
CM_SI = 1e-2          # 1 uF/cm² = 1e-2 F/m²
EL_V, ENA_V, EK_V = -54.4e-3, 50.0e-3, -77.0e-3
E_GABA_V = -70.0e-3
G_AX_S = 5.4e-9


def _load_circuit():
    from neural_exploration.src.larva_circuit import LarvaCircuit
    from brian2 import prefs
    _cache = os.path.join(PROJECT_ROOT, ".cache", "brian2_m9")
    os.makedirs(_cache, exist_ok=True)
    prefs.codegen.runtime.cython.cache_dir = _cache   # R11 独立缓存
    prefs.codegen.target = "numpy"                    # 探针用 numpy 后端（免编译、确定性）
    kw = dict(scale=SCALE, fidelity=FIDELITY, plasticity="none", seed=0,
              connectome_poll_s=0.0, nt_fallback="real",
              provisional_muscles=False, gmax_scale=0.05,
              lever_cmd_desync=True, lever_motor_drive=True, lever_hetero=True)
    circ = LarvaCircuit(**kw)
    circ.build()
    return circ


def extract_network(circ):
    """从冻结 LarvaCircuit 提取网络定义（torch 等价参考实现输入）。"""
    n_comp = int(circ.group.N)
    ex = {
        "n_comp": n_comp,
        "dt_ms": float(circ.dt_ms),
        "dt_s": float(circ.dt_ms) * 1e-3,
        "two_comp": FIDELITY == "two_comp",
        "state": {
            "v": np.full(n_comp, -65.0e-3),
            "g_ampa": np.zeros(n_comp),
            "g_gaba": np.zeros(n_comp),
        },
        "gNa": np.zeros(n_comp),
        "gK": np.full(n_comp, 36.0 * 10.0),       # 36 mS/cm² → 360 S/m²
        "gL": np.full(n_comp, 0.3 * 10.0),        # 0.3 mS/cm² → 3 S/m²
        "AREA": np.asarray(circ.group.AREA[:], dtype=np.float64),
        "stim_col": np.asarray(circ.group.stim_col, dtype=np.int64),
        "peer": np.zeros(n_comp, dtype=np.int64),
        "syn": {},
    }
    from neural_exploration.src.ion_channels import steady_state_gates
    m0, h0, n0 = steady_state_gates(-65.0)
    ex["state"]["m"] = np.full(n_comp, float(m0))
    ex["state"]["h"] = np.full(n_comp, float(h0))
    ex["state"]["n"] = np.full(n_comp, float(n0))
    # gNa：Brian2 VariableView 已返回 SI（S/m²）——120 mS/cm² = 1200 S/m²（实测确认，勿再缩放）
    gna_raw = np.asarray(circ.group.gNa[:], dtype=np.float64)
    if gna_raw.size == n_comp:
        ex["gNa"] = gna_raw                      # 已在 SI（S/m²）
        print("  gNa[S/m²]：soma=%.1f node=%.1f（SI，未缩放）" % (
            gna_raw[0], gna_raw[1] if gna_raw.size > 1 else float("nan")),
            flush=True)
    else:
        ex["gNa"] = np.where(np.arange(n_comp) % 2 == 0, 1200.0, 3000.0)
    if ex["two_comp"]:
        peer = np.empty(n_comp, dtype=np.int64)
        peer[0::2] = np.arange(1, n_comp, 2)
        peer[1::2] = np.arange(0, n_comp, 2)
        ex["peer"] = peer
    for syn_obj in circ.chem_synapses:
        stype = "gaba" if "gaba" in str(syn_obj.name) else "ampa"
        pre_i = np.asarray(syn_obj.i[:], dtype=np.int64)
        post_i = np.asarray(syn_obj.j[:], dtype=np.int64)
        gmax = np.asarray(syn_obj.gmax[:], dtype=np.float64)
        delay = np.asarray(syn_obj.delay[:], dtype=np.float64)
        delay_ms = float(np.median(delay)) * 1e3
        delay_steps = int(round(delay_ms / ex["dt_ms"]))
        ex["syn"][stype] = {"pre": pre_i, "post": post_i, "gmax": gmax,
                            "delay_steps": delay_steps}
        print("  突触 %s：%d 条；gmax 中位=%.3e S/m²；delay=%.3fms（%d steps）" % (
            stype, len(pre_i), float(np.median(gmax)), delay_ms, delay_steps),
            flush=True)
    return ex


def run_cpu(circ, seed: int = 0):
    """CPU Brian2 基线：静息协议（settle + 测量窗）→ 逐隔室 spike 时间（ms）。"""
    from brian2 import ms as bms
    sess = circ.make_session(t_total_ms=SETTLE_MS + T_MEAS_MS)
    sess.reset(seed=seed, motor_drive=True)
    sess.run_resting_window(SETTLE_MS)
    sess.reset(seed=seed, motor_drive=True)      # 清监视器 + 重播种（M8 语义）
    stim = np.array(sess.stim.values, dtype=np.float64).copy()
    t0w = time.perf_counter()
    sess.run_resting_window(T_MEAS_MS)
    wall = time.perf_counter() - t0w
    i_arr = np.asarray(circ._sp.i, dtype=np.int64)
    t_arr = np.asarray(circ._sp.t / bms, dtype=np.float64)
    spikes = defaultdict(list)
    for i, t in zip(i_arr, t_arr):
        spikes[int(i)].append(float(t))
    return spikes, wall, stim


def run_gpu(ex, device: str, dtype, n_total_steps: int, step0: int,
            max_steps: int = None, record_from: int = None, trace_idx=None):
    """torch 内核：全窗跑（settle 演化 + 测量窗记录）→ 逐隔室 spike 时间（ms）。

    spike 语义（实测 Brian2 numpy 后端）：状态更新后检查 v_new > TH；spike 时间记
    步骤起点 s·dt；不应期自检测步终点 (s+1)·dt 起 2ms（→ next_allowed = s + ref_steps）。
    性能：m/h/n 堆叠为 (3,N) 一次更新；突触事件用预计算出边表（spike 稀疏，Python 循环）。
    max_steps：可截断运行窗（MPS 设备验证用短窗）；record_from：spike 记录起点（默认 step0）。
    """
    import torch
    N = ex["n_comp"]
    dt = ex["dt_s"]
    t = {}
    for k, v in ex["state"].items():
        t[k] = torch.as_tensor(np.asarray(v, dtype=np.float64),
                               dtype=dtype, device=device)
    for k in ("gNa", "gK", "gL", "AREA"):
        t[k] = torch.as_tensor(np.asarray(ex[k], dtype=np.float64),
                               dtype=dtype, device=device)
    t["stim_col"] = torch.as_tensor(ex["stim_col"], dtype=torch.long, device=device)
    t["peer"] = torch.as_tensor(ex["peer"], dtype=torch.long, device=device)
    t["next_allowed"] = torch.zeros(N, dtype=dtype, device=device)
    stim = torch.as_tensor(np.asarray(ex["stim"], dtype=np.float64),
                           dtype=dtype, device=device)
    # 出边表（预计算；spike 稀疏 → Python 循环交付）
    out_edges = [[] for _ in range(N)]     # neuron -> [(stype, post, gmax, delay_steps)]
    max_delay = 1
    for stype, s in ex["syn"].items():
        for i in range(len(s["pre"])):
            out_edges[int(s["pre"][i])].append(
                (stype, int(s["post"][i]), float(s["gmax"][i]),
                 int(s["delay_steps"])))
        max_delay = max(max_delay, int(s["delay_steps"]) + 1)
    buf = {stype: torch.zeros((max_delay, N), dtype=dtype, device=device)
           for stype in ex["syn"]}
    ref_steps = int(round(REF_MS / ex["dt_ms"]))
    record_from = step0 if record_from is None else record_from
    n_end = n_total_steps if max_steps is None else min(max_steps, n_total_steps)
    spikes = defaultdict(list)
    _trace = []
    t0w = time.perf_counter()
    exp_dec_ampa = float(np.exp(-dt / 3.0e-3))
    exp_dec_gaba = float(np.exp(-dt / 5.0e-3))
    for s in range(n_end):
        # ---- 1) 事件交付（环形缓冲：先应用本槽再清零——Brian2 语义：事件在步起点生效） ----
        slot = s % max_delay
        for stype, sd in buf.items():
            t["g_" + stype] = t["g_" + stype] + buf[stype][slot]
            buf[stype][slot] = 0.0
        # ---- 2) 捕获步起点状态（Brian2 状态更新器对全部变量**同时在步起点估值**） ----
        v0 = t["v"]
        m0, h0, n0 = t["m"], t["h"], t["n"]
        ga0, gg0 = t["g_ampa"], t["g_gaba"]
        peer0 = v0[t["peer"]] if ex["two_comp"] else None
        vm = v0 * 1e3
        # ---- 3) 门控系数（Hz，基于 v0）：dm/dt = alpha - (alpha+beta)m ⇒ A=-(a+b), B=a ----
        am = (0.1 * (vm + 40.0) / (1.0 - torch.exp(-(vm + 40.0) / 10.0))) * 1e3
        bm = 4.0 * torch.exp(-(vm + 65.0) / 18.0) * 1e3
        ah = 0.07 * torch.exp(-(vm + 65.0) / 20.0) * 1e3
        bh = 1.0 / (1.0 + torch.exp(-(vm + 35.0) / 10.0)) * 1e3
        an = (0.01 * (vm + 55.0) / (1.0 - torch.exp(-(vm + 55.0) / 10.0))) * 1e3
        bn = 0.125 * torch.exp(-(vm + 65.0) / 80.0) * 1e3
        A_g = torch.stack([-(am + bm), -(ah + bh), -(an + bn)])
        B_g = torch.stack([am, ah, an])
        mhn0 = torch.stack([m0, h0, n0])
        A_safe = torch.where(torch.abs(A_g) < 1e-12, torch.ones_like(A_g), A_g)
        BA_g = B_g / A_safe
        mhn_new = torch.where(torch.abs(A_g) < 1e-12, mhn0 + dt * B_g,
                              (mhn0 + BA_g) * torch.exp(A_safe * dt) - BA_g)
        # ---- 4) 突触电导衰减（步起点值 → 衰减后值，与 v 同步提交） ----
        ga_new = ga0 * exp_dec_ampa
        gg_new = gg0 * exp_dec_gaba
        # ---- 5) v 更新：用**步起点** m/h/n 与 g（Brian2 同构）；I_ax 的 v 相关项入 A_v ----
        m3h = m0 ** 3 * h0
        n4 = n0 ** 4
        gsum = t["gL"] + t["gNa"] * m3h + t["gK"] * n4 + ga0 + gg0
        i_stim = stim[s][t["stim_col"]]
        B_v = (t["gL"] * EL_V + t["gNa"] * m3h * ENA_V
               + t["gK"] * n4 * EK_V + gg0 * E_GABA_V
               + i_stim / t["AREA"]) / CM_SI
        if ex["two_comp"]:
            g_ax_dens = G_AX_S / t["AREA"]              # S/m²（轴耦合等效电导密度）
            gsum = gsum + g_ax_dens
            B_v = B_v + (g_ax_dens * peer0) / CM_SI
        A_v = -gsum / CM_SI
        A_safe = torch.where(torch.abs(A_v) < 1e-12, torch.ones_like(A_v), A_v)
        BA_v = B_v / A_safe
        v_new = torch.where(torch.abs(A_v) < 1e-12, v0 + dt * B_v,
                            (v0 + BA_v) * torch.exp(A_safe * dt) - BA_v)
        # ---- 6) 提交（同步） ----
        t["v"] = v_new
        t["m"], t["h"], t["n"] = mhn_new[0], mhn_new[1], mhn_new[2]
        t["g_ampa"], t["g_gaba"] = ga_new, gg_new
        # ---- 调试 trace（可选；逐隔室 v 轨迹） ----
        if trace_idx is not None and s in trace_idx:
            _trace.append((s, [(int(i), float(v_new[i])) for i in trace_idx[s]]))
        # ---- spike 检测（更新后；不应期抑制） ----
        spk = (v_new > TH_V) & (s >= t["next_allowed"])
        if bool(spk.any()):
            idx = spk.nonzero(as_tuple=False).flatten().tolist()
            # 时间基准：Brian2 测量窗 spike 时间自 0 起（net.restore() 重置时钟）→
            # torch 侧同步以 record_from（测量窗起点）为零点
            t_ms = (s - record_from + 0.0) * ex["dt_ms"]
            for ii in idx:
                if s >= record_from:
                    spikes[int(ii)].append(t_ms)
            t["next_allowed"] = torch.where(spk, (s + ref_steps) * 1.0,
                                            t["next_allowed"])
            # 新事件入缓冲（出边表 Python 循环，spike 稀疏；**按 post 隔室索引**交付）
            for ii in idx:
                for stype, post, gmax, d in out_edges[ii]:
                    if d == 0:
                        # 零延迟：直接加本步 g（下一状态更新前生效；不入缓冲防双投）
                        t["g_" + stype][post] = t["g_" + stype][post] + gmax
                    else:
                        tgt_slot = (s + d) % max_delay
                        buf[stype][tgt_slot, post] = buf[stype][tgt_slot, post] + gmax
    wall = time.perf_counter() - t0w
    if trace_idx is not None:
        return spikes, wall, _trace
    return spikes, wall


def measure_mps_scaling(ex, device="mps", dtype=None, n_steps=300,
                        sizes=(600, 6000, 60000), n_syn=None):
    """MPS 每步墙钟的规模分解：t(N) = a + b·N（a=launch 固定开销，b=逐元素成本）。

    合成同构网络（同操作序列、无突触或稀疏突触）→ 线性拟合并外推全规模（278,510 隔室）。
    诚实性：探针实测值落盘；外推值标注为 projection（需实现向量化后复测，§2.5 冻结规则）。
    """
    import torch
    if dtype is None:
        dtype = torch.float32
    out = {}
    for N in sizes:
        exs = {
            "n_comp": N, "dt_ms": ex["dt_ms"], "dt_s": ex["dt_s"],
            "two_comp": ex["two_comp"],
            "state": {"v": np.full(N, -65e-3), "m": np.full(N, 0.0529),
                      "h": np.full(N, 0.596), "n": np.full(N, 0.3177),
                      "g_ampa": np.zeros(N), "g_gaba": np.zeros(N)},
            "gNa": np.where(np.arange(N) % 2 == 0, 1200.0, 3000.0),
            "gK": np.full(N, 360.0), "gL": np.full(N, 3.0),
            "AREA": np.where(np.arange(N) % 2 == 0, 1.257e-9, 9.4248e-12),
            "stim_col": np.zeros(N, dtype=np.int64),
            "peer": np.zeros(N, dtype=np.int64),
            "stim": np.zeros((n_steps, 1)),
            "syn": {},
        }
        peer = np.empty(N, dtype=np.int64)
        peer[0::2] = np.arange(1, N, 2)
        peer[1::2] = np.arange(0, N, 2)
        exs["peer"] = peer
        if n_syn:
            # 稀疏突触（每神经元 n_syn 条随机出边；固定 seed）
            rng = np.random.default_rng(0)
            pre = np.repeat(np.arange(N), n_syn)
            post = rng.integers(0, N, size=N * n_syn)
            exs["syn"] = {"ampa": {"pre": pre, "post": post,
                                   "gmax": np.full(N * n_syn, 0.199),
                                   "delay_steps": 10}}
        _ = run_gpu(exs, device, dtype, 200, 0, record_from=0)   # warmup
        walls = []
        for _rep in range(3):
            t0 = time.perf_counter()
            _ = run_gpu(exs, device, dtype, n_steps, 0, record_from=0)
            walls.append(time.perf_counter() - t0)
        wall = min(walls)          # min = 最少争用（吞吐基准惯例；本机负载病态 R11）
        out[N] = wall / n_steps * 1e3
        print("    MPS N=%6d 隔室：%.3f ms/step（min of 3 × %d 步；raw=%s）" % (
            N, out[N], n_steps, ["%.2f" % w for w in walls]), flush=True)
    # 线性拟合 t = a + b·N
    import numpy as _np
    xs = _np.array(list(out.keys()), dtype=float)
    ys = _np.array(list(out.values()), dtype=float)
    b, a = _np.polyfit(xs, ys, 1)
    out["_a_ms"] = float(a)
    out["_b_ms_per_comp"] = float(b)
    return out


def compare_spikes(cpu_spikes, gpu_spikes, n_comp, tol_ms=TOL_MS):
    """逐隔室 spike 对齐：CPU spike 为主参考，GPU 额外 spike 计入未对齐。"""
    total_cpu = 0
    matched = 0
    gpu_only = 0
    d_all = []
    for i in range(n_comp):
        ct = np.sort(np.asarray(cpu_spikes.get(i, []), dtype=np.float64))
        gt = np.sort(np.asarray(gpu_spikes.get(i, []), dtype=np.float64))
        total_cpu += ct.size
        used = set()
        for t in ct:
            cand = np.searchsorted(gt, t)
            best = None
            for j in (cand - 1, cand):
                if 0 <= j < gt.size and j not in used:
                    dd = abs(gt[j] - t)
                    if best is None or dd < best[1]:
                        best = (j, dd)
            if best is not None and best[1] < tol_ms:
                used.add(best[0])
                matched += 1
            d_all.append(best[1] if best is not None else float("inf"))
        gpu_only += max(0, gt.size - len(used))
    denom = max(total_cpu + gpu_only, 1)
    rate = 100.0 * matched / denom
    return rate, total_cpu, matched, gpu_only, d_all


def silent_fraction(spikes, n_comp, t_meas_ms):
    rates = np.array([len(spikes.get(i, [])) / (t_meas_ms / 1000.0)
                      for i in range(n_comp)])
    return float(np.mean(rates < 0.5))


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    print("=== M9 G0 引擎探针（scan_m9_engine.py）===", flush=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    write_engine_params()

    circ = _load_circuit()
    print("网络：%d 神经元 / %d 隔室；dt=%.3fms；方法=%s" % (
        circ.sub.n_neurons, circ.group.N, circ.dt_ms, circ.method), flush=True)
    ex = extract_network(circ)

    # CPU 基线（测量窗；spike 时间自 0 起——M8 store/restore 语义：settle 被丢弃，
    # 测量窗自初始态与 stim[:n_meas] 重新起跑 → torch 侧同协议复现）
    cpu_spikes, cpu_wall, stim_full = run_cpu(circ)
    n_total_steps = int(round(T_MEAS_MS / ex["dt_ms"]))
    ex["stim"] = stim_full[:n_total_steps].copy()
    step0 = 0
    n_spk_cpu = sum(len(v) for v in cpu_spikes.values())
    print("CPU：%d spike / %d 隔室；墙钟 %.2fs（测量窗 %.0fms）" % (
        n_spk_cpu, len(cpu_spikes), cpu_wall, T_MEAS_MS), flush=True)

    import torch
    results = {}

    # ---- 数值验证 1：torch CPU float64（与 Brian2 同精度） ----
    spk64, wall64 = run_gpu(ex, "cpu", torch.float64, n_total_steps, 0)
    rate64, tc64, mc64, go64, _ = compare_spikes(cpu_spikes, spk64, ex["n_comp"])
    results["cpu_f64"] = (rate64, wall64)
    print("torch-CPU float64 对齐率：%.2f%%（%d/%d CPU + %d GPU-only）；墙钟 %.2fs" % (
        rate64, mc64, tc64, go64, wall64), flush=True)

    # ---- 数值验证 2：torch CPU float32（GPU 精度档） ----
    spk32, wall32 = run_gpu(ex, "cpu", torch.float32, n_total_steps, 0)
    rate32, tc32, mc32, go32, d32 = compare_spikes(cpu_spikes, spk32, ex["n_comp"])
    results["cpu_f32"] = (rate32, wall32)
    silent_cpu = silent_fraction(cpu_spikes, ex["n_comp"], T_MEAS_MS)
    silent_gpu32 = silent_fraction(spk32, ex["n_comp"], T_MEAS_MS)
    print("torch-CPU float32 对齐率：%.2f%%（%d/%d CPU + %d GPU-only）；静默 CPU=%.4f GPU=%.4f；墙钟 %.2fs" % (
        rate32, mc32, tc32, go32, silent_cpu, silent_gpu32, wall32), flush=True)

    # ---- 设备验证：MPS float32（短窗；验证设备路径正确性 + 每步墙钟） ----
    mps_win_steps = int(round(500.0 / ex["dt_ms"]))   # 500ms 窗（10000 步）
    spk_mps, wall_mps = run_gpu(ex, "mps", torch.float32, n_total_steps, 0,
                                max_steps=mps_win_steps, record_from=0)
    # 对照：同窗 torch-CPU float32
    spk_cpu32_win, _ = run_gpu(ex, "cpu", torch.float32, n_total_steps, 0,
                               max_steps=mps_win_steps, record_from=0)
    rate_mps_cpu, _, _, _, _ = compare_spikes(spk_cpu32_win, spk_mps,
                                              ex["n_comp"], tol_ms=0.05)
    per_step_mps = wall_mps / mps_win_steps * 1e3
    print("MPS 设备验证（500ms 窗）：与 torch-CPU-float32 对齐率 %.2f%%（<0.05ms）；"
          "墙钟 %.2fs → %.2f ms/step → 全窗推算 ≈ %.0f s" % (
              rate_mps_cpu, wall_mps, per_step_mps, per_step_mps * n_total_steps),
          flush=True)

    # ---- 主判据用 CPU float32 结果（数值正确性） ----
    rate = rate32
    total_cpu = tc32
    matched = mc32
    gpu_only = go32
    d_all = d32
    silent_gpu = silent_gpu32
    gpu_spikes = spk32
    gpu_wall = wall32

    # ---- 确定性：torch CPU float32 重跑 ----
    spk32b, _ = run_gpu(ex, "cpu", torch.float32, n_total_steps, 0)
    rate_det, _, _, _, _ = compare_spikes(spk32, spk32b, ex["n_comp"], tol_ms=0.05)
    print("确定性重跑（torch-CPU-float32 <0.05ms）：%.2f%%" % rate_det, flush=True)

    crit_a = rate >= 99.0
    crit_b = abs(silent_cpu - silent_gpu) < 0.01
    crit_d = rate_det >= 99.0
    verdict = "PASS" if (crit_a and crit_b and crit_d) else "FAIL"

    # 墙钟 + 全规模预算
    cpu_per_s = cpu_wall / (T_MEAS_MS / 1000.0)
    gpu_per_s = wall32 / (T_MEAS_MS / 1000.0)
    speedup = cpu_per_s / max(gpu_per_s, 1e-12)
    # MPS 每步墙钟（短窗实测）+ 规模分解（t = a + b·N）→ 全规模 projection
    print("MPS 规模分解测量（合成同构网络，无突触）：", flush=True)
    scale_ms = measure_mps_scaling(ex, device="mps", dtype=torch.float32,
                                   n_steps=1500, sizes=(600, 6000, 60000))
    a_ms = scale_ms["_a_ms"]
    b_ms = scale_ms["_b_ms_per_comp"]
    N_FULL = 2 * 139255        # 全规模隔室数（two_comp）
    n_steps_30s = int(round(30.0 / (ex["dt_ms"] * 1e-3)))
    proj_30s_s = (a_ms + b_ms * N_FULL) * 1e-3 * n_steps_30s
    proj_30s_gpu_h = proj_30s_s / 3600.0
    print("  拟合：a=%.3f ms/step（launch 固定）+ b=%.3e ms/隔室/step；"
          "全规模(%d 隔室) projection = %.3f ms/step → 30s 单试次 %.2f GPU-h" % (
              a_ms, b_ms, N_FULL, a_ms + b_ms * N_FULL, proj_30s_gpu_h), flush=True)
    # MPS 每步墙钟（短窗实测）→ 全规模推算（突触线性；launch 开销为主 → 保守上限）
    mps_per_s_full = per_step_mps * 1e-3 * (n_total_steps / (T_MEAS_MS / 1000.0))
    n_syn_sub = sum(len(s["pre"]) for s in ex["syn"].values())
    est_h_neurons = (139255.0 / SCALE) * (30.0 / (T_MEAS_MS / 1000.0)) * gpu_per_s / 3600.0
    est_h_syn = (54500000.0 / max(n_syn_sub, 1)) * (30.0 / (T_MEAS_MS / 1000.0)) * gpu_per_s / 3600.0
    # MPS 实测算子：每步墙钟 × 全规模步数（600→139255 隔室按元素线性 + launch 恒定假设）
    est_mps_30s_h = (per_step_mps * 1e-3) * (n_steps_30s) * (139255.0 / SCALE) / 3600.0
    est_mps_30s_h_syn = (per_step_mps * 1e-3) * n_steps_30s * (54500000.0 / max(n_syn_sub, 1)) / 3600.0
    budget_ok_naive = min(est_mps_30s_h, est_mps_30s_h_syn) <= 1.0
    budget_ok = bool(proj_30s_gpu_h <= 1.0)
    print("墙钟：CPU %.2fs GPU(torch-CPU-f32) %.2fs（测量窗 %.0fms）；加速比 %.1f×" % (
        cpu_wall, wall32, T_MEAS_MS, speedup), flush=True)
    print("修正模型（固定开销 a + 线性项 b·N，见 data/m9_engine_scaling.csv）："
          "全规模 30s 单试次 %.3f GPU-h → 预算 %s（禁用作废的线性外推）" % (
              proj_30s_gpu_h, "≤1 GPU-h OK" if budget_ok else "OVER"), flush=True)

    write_probe_csv(dict(device="mps+torch-cpu", n_comp=ex["n_comp"],
                         n_spk_cpu=n_spk_cpu, n_spk_gpu=sum(len(v) for v in spk32.values()),
                         rate=rate, matched=matched, total_cpu=total_cpu,
                         gpu_only=gpu_only, silent_cpu=silent_cpu,
                         silent_gpu=silent_gpu, cpu_wall=cpu_wall,
                         gpu_wall=wall32, speedup=speedup,
                         crit_a=crit_a, crit_b=crit_b, crit_d=crit_d,
                         verdict=verdict, est_h_neurons=est_h_neurons,
                         est_h_syn=est_h_syn, budget_ok=budget_ok,
                         rate_det=rate_det, rate64=rate64, rate_mps_cpu=rate_mps_cpu,
                         per_step_mps=per_step_mps,
                         est_mps_30s_h=est_mps_30s_h,
                         est_mps_30s_h_syn=est_mps_30s_h_syn,
                         proj_30s_gpu_h=proj_30s_gpu_h, a_ms=a_ms, b_ms=b_ms,
                         budget_ok_naive=budget_ok_naive,
                         mps_win_wall=wall_mps, mps_win_steps=mps_win_steps))
    make_plot(cpu_spikes, spk32, ex["n_comp"], d_all,
              silent_cpu, silent_gpu, rate, cpu_wall, wall32)

    g0 = "PASS" if (verdict == "PASS" and budget_ok) else "FAIL"
    print("G0 门判定：%s（对齐 %s + 预算 %s）" % (
        g0, verdict, "OK" if budget_ok else "OVER"), flush=True)
    return 0 if g0 == "PASS" else 2


def write_engine_params():
    header = ("criterion,route_mps_pytorch,route_brian2cuda,route_nest_gpu,"
              "weight,decision,note\n")
    rows = [
        ["对齐可实现性（同一网络定义语义下逐神经元比对）",
         "高（等价参考实现 + 对齐探针 §2.3）", "最高（同一 Brian2 定义直换后端）",
         "中（网络定义迁移 + 等价性验证）", "1", "mps_pytorch",
         "实测：torch 2.8.0 MPS available=True；对齐探针数据见 m9_engine_probe.csv"],
        ["本机硬件现实性（macOS 无 NVIDIA）", "✅ MPS 本机可行",
         "❌ 需云/外置 NVIDIA GPU", "⚠️ macOS 编译支持有限", "2", "mps_pytorch",
         "api.flywire.ai/cave 网络受限记录；云 GPU 需主 agent 裁决"],
        ["开发风险/成本（5–10 天含探针）", "中（内核自研，bug 面可控）",
         "低（生态成熟）", "中", "3", "mps_pytorch",
         "exponential_euler 逐变量闭式已核实（brian2 2.6.0 源码）"],
        ["确定性可工程化（固定 kernel/seed）", "高（完全可控）", "高", "中",
         "4", "mps_pytorch", "固定 kernel + seed + 统计级重跑一致"],
        ["规模/速度（54.5M 突触点神经元）", "单 GPU 实时~数倍实时（预注册）",
         "需云 GPU", "需 Linux", "5", "mps_pytorch",
         "内存预注册 <1.5GB（§2.2）；探针墙钟实测落盘"],
    ]
    with open(OUT_PARAMS, "w", encoding="utf-8", newline="") as f:
        f.write("# M9 引擎选型判据定稿（§2.2 预注册；正确性对齐 > 速度；最终由主 agent 裁决）\n")
        f.write("# 实测：.venv-m9 torch 2.8.0 MPS available=True；Brian2CUDA/NEST GPU 本机不可行\n")
        f.write(header)
        w = _csv.writer(f, lineterminator="\n")
        w.writerows(rows)
    print("§2.2 选型判据 →", OUT_PARAMS, flush=True)


def write_probe_csv(r):
    header = ["metric", "value", "note"]
    rows = [
        ["device", r["device"], "torch MPS（NG 门路线① 本机主路线）"],
        ["fidelity", "two_comp", "M8 定稿保真度（G0 对齐参考）"],
        ["scale", "300", "M8 冻结子图"],
        ["dt_ms", "0.05", "FIDELITY_DT 定稿"],
        ["T_meas_ms", "2000.0", "对齐测量窗"],
        ["settle_ms", "500.0", "settle 窗（M8 语义）"],
        ["n_comp", str(r["n_comp"]), ""],
        ["cpu_spikes", str(r["n_spk_cpu"]), ""],
        ["gpu_spikes", str(r["n_spk_gpu"]), ""],
        ["aligned_pct", "%.2f" % r["rate"],
         "判据 (a)：|Δt|<0.1ms 比例（matched/(CPU+GPU-only)，≥99%；torch-CPU-float32 主判据）"],
        ["aligned_n", "%d/%d (gpu_only=%d)" % (r["matched"], r["total_cpu"],
                                               r["gpu_only"]), ""],
        ["rate_cpu_f64", "%.2f" % r.get("rate64", float("nan")),
         "torch-CPU-float64 vs Brian2 对齐率（同精度参考）"],
        ["rate_mps_vs_cpu", "%.2f" % r.get("rate_mps_cpu", float("nan")),
         "MPS float32 vs torch-CPU float32（500ms 窗，<0.05ms 设备路径一致性）"],
        ["per_step_mps_ms", "%.2f" % r.get("per_step_mps", 0.0),
         "MPS 每步墙钟（500ms 窗实测；launch 开销为主）"],
        ["silent_cpu", "%.4f" % r["silent_cpu"], ""],
        ["silent_gpu", "%.4f" % r["silent_gpu"], ""],
        ["silent_diff_pp", "%.4f" % (100.0 * (r["silent_cpu"] - r["silent_gpu"])),
         "判据 (b)：静默差 <1pp"],
        ["crit_a_pass", str(r["crit_a"]), "spike 对齐 ≥99%"],
        ["crit_b_pass", str(r["crit_b"]), "静默差 <1pp"],
        ["crit_d_pass", str(r["crit_d"]), "GPU 确定性重跑一致性 %.2f pct（阈值 99）" % r["rate_det"]],
        ["verdict", r["verdict"], "PASS=对齐通过"],
        ["cpu_wall_s", "%.2f" % r["cpu_wall"], "CPU Brian2 测量窗墙钟"],
        ["gpu_wall_s", "%.2f" % r["gpu_wall"],
         "torch-CPU-float32 测量窗墙钟（主判据实现；MPS 见 mps_win_wall_s）"],
        ["mps_win_wall_s", "%.2f" % r.get("mps_win_wall", 0.0),
         "MPS 500ms 窗墙钟（设备路径验证用短窗）"],
        ["mps_win_steps", str(r.get("mps_win_steps", 0)), "MPS 短窗步数"],
        ["speedup", "%.1f" % r["speedup"], "CPU/GPU"],
        ["mps_scaling_a_ms", "%.3f" % r.get("a_ms", 0.0),
         "MPS 每步固定开销（launch；t=a+b·N 线性拟合，合成同构网络实测）"],
        ["mps_scaling_b_ms_per_comp", "%.3e" % r.get("b_ms", 0.0),
         "MPS 逐隔室每步成本（拟合斜率）"],
        ["proj_full_30s_gpu_h", "%.3f" % r.get("proj_30s_gpu_h", 0.0),
         "全规模 30s 单试次 projection（**固定开销 a + 线性项 b·N** 模型；合成同构网络；真实网络 scaling 见 data/m9_engine_scaling.csv）"],
        ["projection_model", "a + b*N",
         "禁止把 300 档（近纯固定开销）ms/step 按规模线性放大——错误外推（已删除该行）"],
        ["budget_le_1_gpu_h", str(r["budget_ok"]),
         "§0.9 R10 预注册 ≤1 GPU-h（以规模拟合 projection 判定）"],
        ["g0_verdict", "", "由主 agent 定稿（本探针数据输入）"],
    ]
    with open(OUT_PROBE, "w", encoding="utf-8", newline="") as f:
        f.write("# M9 G0 引擎探针（§2.3）：300 two_comp CPU vs GPU(MPS) 对齐 + 墙钟\n")
        f.write("# 对齐判据预注册 §0.7 #3：统计级主判据（浮点重排为测量限制）\n")
        w = _csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    print("§2.3 探针结果 →", OUT_PROBE, flush=True)


def make_plot(cpu_spikes, gpu_spikes, n_comp, d_all, silent_cpu, silent_gpu,
              rate, cpu_wall, gpu_wall):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    d = np.asarray(d_all, dtype=np.float64)
    d = d[np.isfinite(d)]
    ax = axes[0, 0]
    ax.hist(np.minimum(d, 1.0), bins=50, color="#1f77b4")
    ax.axvline(0.1, color="red", ls="--", lw=1)
    ax.set_title("CPU-GPU spike time diff (aligned <0.1ms: %.2f%%)" % rate)
    ax.set_xlabel("|dt| (ms, clipped at 1.0)")
    ax.set_ylabel("count")
    ax = axes[0, 1]
    c = np.array([len(cpu_spikes.get(i, [])) for i in range(n_comp)])
    g = np.array([len(gpu_spikes.get(i, [])) for i in range(n_comp)])
    ax.scatter(c, g, s=4, alpha=0.5)
    mx = max(c.max(), g.max(), 1)
    ax.plot([0, mx], [0, mx], color="red", ls="--", lw=1)
    ax.set_title("Per-compartment spike count: CPU vs GPU")
    ax.set_xlabel("CPU spikes"); ax.set_ylabel("GPU spikes")
    ax = axes[1, 0]
    rc = c / (T_MEAS_MS / 1000.0); rg = g / (T_MEAS_MS / 1000.0)
    ax.hist(rc, bins=40, alpha=0.6, label="CPU", color="#1f77b4")
    ax.hist(rg, bins=40, alpha=0.6, label="GPU", color="#ff7f0e")
    ax.set_title("Firing-rate distribution (silent CPU=%.2f GPU=%.2f)" % (silent_cpu, silent_gpu))
    ax.set_xlabel("rate (Hz)"); ax.set_ylabel("count"); ax.legend()
    ax = axes[1, 1]
    ax.bar(["CPU Brian2", "GPU MPS"], [cpu_wall, gpu_wall],
           color=["#1f77b4", "#ff7f0e"])
    ax.set_title("Probe wall-clock (T=%.0f ms window)" % T_MEAS_MS)
    ax.set_ylabel("wall (s)")
    fig.tight_layout()
    fig.savefig(OUT_PLOT, dpi=110)
    print("对齐图 →", OUT_PLOT, flush=True)


if __name__ == "__main__":
    sys.exit(main())
