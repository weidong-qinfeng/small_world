"""M6 P1 验证：STDP 组件（成对时序可塑性）vs 文献理论曲线 + STP 回归 + 网络接口就绪。

清单 §0 P1 / §2.3（G0 组件门验证对象）：
  1. 2 神经元对（PointNeuron）+ 1 条 `StdpSynapse`（ampa，w0=5.0nS 语义）；
  2. 成对脉冲协议 Δt ∈ {−60,−40,−20,−10,−5,+5,+10,+20,+40,+60} ms × 各 50 对
     （确定性 seed=0，p=1/n=1）→ 每点 Δw/w0 vs 理论曲线
     ΔW ∝ exp(−|Δt|/τ)，τ=20ms；LTP 窗 Δt>0 / LTD 窗 Δt<0；
  3. 判据（预注册 §0 P1）：每点 |ΔW_rel 差| ≤ 0.2；幅值比 A₋/A₊=0.9 核对；
     权重有界 [0, w_max]；确定性重跑逐位一致；
  4. STP 回归：M2 P3 协议（50Hz×10 易化/抑制）重跑 vs data/m2_stp.csv（冻结）不回归；
  5. 三因子冒烟（informational，P4 机制前提）：M=1 配对 → 0<Δw<w_max−w0；
     M=0 → Δw=0（P4(d) 消融前提，证明调质门控必需）；
  6. 网络级接口就绪（§0 预注册 #1）：默认不启用（no-op），enabled=True 在
     合成 mini-circuit 上可构建冒烟（非 302 集成——G1 门后由 learning.py 组装）。

输出（清单 §2.3）：
  data/m6_stdp_reference.csv（理论标准表，预生成）+ data/m6_p1_stdp.csv +
  reports/neuro/m6_p1_stdp.png + data/m6_learning_params.csv（stdp 段母版）。

运行：python neural_exploration/tools/validate_p1_stdp.py
（确定性 p=1/n=1；同参数重跑逐位一致；运行前检查无并发。）
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DATA_DIR = os.path.join(ROOT, "neural_exploration", "data")
REPORTS_DIR = os.path.join(ROOT, "neural_exploration", "reports", "neuro")

M6_PARAMS_CSV = os.path.join(DATA_DIR, "m6_learning_params.csv")
M6_REF_CSV = os.path.join(DATA_DIR, "m6_stdp_reference.csv")
M6_P1_CSV = os.path.join(DATA_DIR, "m6_p1_stdp.csv")
M6_P1_PNG = os.path.join(REPORTS_DIR, "m6_p1_stdp.png")
M2_STP_CSV = os.path.join(DATA_DIR, "m2_stp.csv")

#: 成对脉冲协议（清单 §2.3 预注册集合）
DT_LIST = [-60.0, -40.0, -20.0, -10.0, -5.0, 5.0, 10.0, 20.0, 40.0, 60.0]
#: 每个 Δt 的脉冲对数（预注册 50 对）
N_PAIRS = 50
#: 相邻对锚点间隔（ms）——≥10τ 保证痕迹隔离（交叉项 ~exp(−17)）
PAIR_SPACING_MS = 400.0
#: 饱和验证：对数（超出 w_max/w0 下界）
SAT_N_PAIRS = 200
SAT_SPACING_MS = 120.0
#: 刺激脉冲时长（ms）；实际发放时刻由 SpikeMonitor 测量（realized Δt）
PULSE_MS = 1.0
PULSE_AMP_UA_CM2 = 20.0
DT_MS = 0.1


# --------------------------------------------------------------------- #
# 协议运行
# --------------------------------------------------------------------- #
def _stim_array(pulse_starts_ms, dt_ms, n_steps, amp_nA):
    """单神经元刺激数组（1 列，单位 amp；脉冲 [t, t+PULSE_MS]）。"""
    from brian2 import amp
    arr = np.zeros((n_steps, 1))
    n_pulse = int(round(PULSE_MS / dt_ms))
    for t0 in pulse_starts_ms:
        i0 = int(round(t0 / dt_ms))
        arr[i0:i0 + n_pulse, 0] = amp_nA
    return arr * amp


def _run_stdp_protocol(params, dt_nominal, n_pairs=N_PAIRS,
                       spacing=PAIR_SPACING_MS, t_total=None, idx=0,
                       record_w=True):
    """一次 STDP 协议运行（2 PointNeuron + 1 StdpSynapse，确定性）。

    返回 realized_dt（全部对一致的平均值）、dW_total、w_final、w_min/w_max
    （w 轨迹，StateMonitor）、n_pre/n_post（发放计数断言）。
    """
    from brian2 import (Network, SpikeMonitor, StateMonitor, TimedArray, ms,
                        nA, seed as bseed, start_scope, defaultclock)
    from neural_exploration.src.brian_env import configure_brian2
    from neural_exploration.src.point_neuron import PointNeuron
    from neural_exploration.src.plasticity import StdpSynapse
    from neural_exploration.src.synapse_model import (SynapseParams,
                                                      chemical_im_terms,
                                                      chemical_post_eqs)

    configure_brian2()
    start_scope()
    defaultclock.dt = DT_MS * ms
    bseed(0)

    sp = SynapseParams(synapse_type="ampa", g_max_ns=params.g_max_ns,
                       tau_ms=params.tau_syn_ms, e_rev_mv=params.e_rev_mv,
                       p_release=params.p_release, n_vesicles=params.n_vesicles)
    pre = PointNeuron(name=f"p{idx}_pre", dt_ms=DT_MS, stim_var="stim_pre")
    post = PointNeuron(name=f"p{idx}_post", dt_ms=DT_MS, stim_var="stim_post",
                       extra_eqs=chemical_post_eqs({"ampa": sp}),
                       extra_im_terms=chemical_im_terms({"ampa": sp}))
    pre.build()
    post.build()
    syn = StdpSynapse(pre, post, params=params, name=f"p{idx}_syn").build()

    anchors = [100.0 + k * spacing for k in range(n_pairs)]
    # 统一锚点：Δt≥0 → pre 在锚点、post 在锚点+Δt；Δt<0 → post 在锚点、pre 在锚点+|Δt|
    pre_stims = [a + max(0.0, -dt_nominal) for a in anchors]
    post_stims = [a + max(0.0, dt_nominal) for a in anchors]
    t_tot = t_total or (anchors[-1] + 300.0)
    n_steps = int(round(t_tot / DT_MS))
    amp_nA = pre.density_to_nA(PULSE_AMP_UA_CM2)
    ta_pre = TimedArray(_stim_array(pre_stims, DT_MS, n_steps, amp_nA),
                        dt=DT_MS * ms, name="stim_pre")
    ta_post = TimedArray(_stim_array(post_stims, DT_MS, n_steps, amp_nA),
                         dt=DT_MS * ms, name="stim_post")
    sp_pre = SpikeMonitor(pre.neuron, "v", name="sp_pre")
    sp_post = SpikeMonitor(post.neuron, "v", name="sp_post")
    net = Network(pre.neuron, post.neuron, syn.synapses, sp_pre, sp_post)
    if record_w:
        mon_w = StateMonitor(syn.synapses, "w", record=[0], dt=DT_MS * ms,
                             name="mon_w")
        net.add(mon_w)
    net.run(t_tot * ms, namespace={"stim_pre": ta_pre, "stim_post": ta_post})

    t_pre = np.array(sp_pre.t / ms)
    t_post = np.array(sp_post.t / ms)
    w_final = float(syn.synapses.w[0])
    if record_w:
        w_traj = np.array(mon_w.w[0])
        w_min, w_max = float(np.min(w_traj)), float(np.max(w_traj))
    else:
        w_min = w_max = w_final
    if len(t_pre) != n_pairs or len(t_post) != n_pairs:
        raise AssertionError(
            f"dt={dt_nominal}: 发放计数异常 pre={len(t_pre)} post={len(t_post)}"
            f"（期望 {n_pairs}；确定性协议发放数必须精确）")
    # 按序配对（两组时间序列均严格递增，第 k 对 = 第 k 次发放）
    dt_k = t_post - t_pre
    if dt_k.max() - dt_k.min() > 1e-6:
        raise AssertionError(
            f"dt={dt_nominal}: realized Δt 不一致 {dt_k.min():.4f}..{dt_k.max():.4f}")
    realized_dt = float(dt_k[0])
    return dict(dt_nominal=float(dt_nominal), realized_dt=realized_dt,
                dW_total=float(w_final - params.w0), w_final=w_final,
                w_min=w_min, w_max=w_max, n_pre=int(len(t_pre)),
                n_post=int(len(t_post)), t_total_ms=t_tot)


def _theory_at(dt_ms, params):
    """理论 ΔW_rel(Δt) = A₊·exp(−Δt/τ₊)（Δt>0）/ −A₋·exp(Δt/τ₋)（Δt<0）。"""
    dt = np.asarray(dt_ms, dtype=float)
    out = np.where(dt > 0,
                   params.a_plus * np.exp(-dt / params.tau_plus_ms),
                   -params.a_minus * np.exp(dt / params.tau_minus_ms))
    return out


def _run_sweep(params, idx_base=0):
    """协议全点扫描 → [{dt_nominal, realized_dt, dW_total, ...}]。"""
    out = []
    for k, dt in enumerate(DT_LIST):
        out.append(_run_stdp_protocol(params, dt, idx=idx_base + k))
    return out


def _saturation_runs(params):
    """权重有界硬验证：超出 w_max/w0 的连续 LTP/LTD → clip 至 2.0/0.0 精确。"""
    ltp = _run_stdp_protocol(params, +5.0, n_pairs=SAT_N_PAIRS,
                             spacing=SAT_SPACING_MS, idx=100)
    ltd = _run_stdp_protocol(params, -5.0, n_pairs=SAT_N_PAIRS,
                             spacing=SAT_SPACING_MS, idx=101)
    return dict(ltp_w=ltp["w_final"], ltd_w=ltd["w_final"],
                ltp_wmax=ltp["w_max"], ltd_wmin=ltd["w_min"])


def _three_factor_smoke(params):
    """三因子冒烟（informational）：M=1 配对 → 0<Δw<w_max−w0；M=0 → Δw=0。"""
    from brian2 import (Network, SpikeMonitor, TimedArray, amp, ms, nA,
                        start_scope, defaultclock, seed as bseed)
    from neural_exploration.src.brian_env import configure_brian2
    from neural_exploration.src.point_neuron import PointNeuron
    from neural_exploration.src.plasticity import ThreeFactorSynapse
    from neural_exploration.src.synapse_model import (SynapseParams,
                                                      chemical_im_terms,
                                                      chemical_post_eqs)

    def run_once(m_profile, idx):
        configure_brian2()
        start_scope()
        defaultclock.dt = DT_MS * ms
        bseed(0)
        sp = SynapseParams(synapse_type="ampa", g_max_ns=params.g_max_ns,
                           tau_ms=params.tau_syn_ms, e_rev_mv=params.e_rev_mv,
                           p_release=1.0, n_vesicles=1)
        pre = PointNeuron(name=f"tf{idx}_pre", dt_ms=DT_MS, stim_var="stim_pre")
        post = PointNeuron(name=f"tf{idx}_post", dt_ms=DT_MS, stim_var="stim_post",
                           extra_eqs=chemical_post_eqs({"ampa": sp}),
                           extra_im_terms=chemical_im_terms({"ampa": sp}))
        pre.build()
        post.build()
        syn = ThreeFactorSynapse(pre, post, params=params,
                                 name=f"tf{idx}_syn").build(
                                     modulation_timedarray=_mod_timedarray(
                                         m_profile))
        t_tot = 2000.0
        n_steps = int(round(t_tot / DT_MS))
        amp_nA = pre.density_to_nA(PULSE_AMP_UA_CM2)
        pre_stims = [100.0 + k * 100.0 for k in range(10)]  # 10Hz×10（100–1000ms）
        ta_pre = TimedArray(_stim_array(pre_stims, DT_MS, n_steps, amp_nA),
                            dt=DT_MS * ms, name="stim_pre")
        ta_post = TimedArray(np.zeros((n_steps, 1)) * amp,
                             dt=DT_MS * ms, name="stim_post")
        sp_pre = SpikeMonitor(pre.neuron, "v", name="sp_pre")
        net = Network(pre.neuron, post.neuron, syn.synapses, sp_pre)
        net.run(t_tot * ms, namespace={"stim_pre": ta_pre, "stim_post": ta_post})
        return float(syn.synapses.w[0]) - params.w0, len(sp_pre.t)

    def _mod_timedarray(profile):
        from brian2 import TimedArray, ms
        n_steps = int(round(2000.0 / DT_MS))
        arr = np.zeros(n_steps)
        if profile is not None:
            i0, i1 = int(round(profile[0] / DT_MS)), int(round(profile[1] / DT_MS))
            arr[i0:i1] = 1.0
        return TimedArray(arr, dt=DT_MS * ms, name="M_t")

    dw_m1, n_spikes = run_once((500.0, 1500.0), idx=200)
    dw_m0, _ = run_once(None, idx=201)
    return dict(dW_M1=dw_m1, dW_M0=dw_m0, n_pre_spikes=n_spikes,
                w_max_minus_w0=params.w_max - params.w0)


def _stp_regression():
    """STP 回归：M2 P3 协议重跑 vs 冻结 data/m2_stp.csv（逐脉冲 |Δ| ≤ 1e-3 mV）。"""
    from neural_exploration.src.neuron_pair import NeuronPair, pulse_train
    from neural_exploration.tools.synapse_metrics import psp_amplitudes

    freq_hz, n_pulses, t_total = 50.0, 10, 320.0

    def run_case(stp):
        pair = NeuronPair(t_total_ms=t_total, seed=0)
        pair.add_chemical("ampa", g_max_ns=0.5, p_release=1.0, n_vesicles=1,
                          stp=stp)
        r = pair.run(pre_pulses=pulse_train(50.0, freq_hz, n_pulses, 1.0, 20.0),
                     record=["pre_node3", "post_soma"])
        return psp_amplitudes(r.t_ms, r.v_mv["post_soma"],
                              r.spike_times_ms["pre_node3"])

    fac = run_case((0.03, 120.0, 40.0))
    dep = run_case((0.6, 10.0, 400.0))

    ref = {}
    with open(M2_STP_CSV, newline="", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("case"):
                continue
            parts = [x.strip() for x in line.split(",")]
            ref.setdefault(parts[0], []).append(float(parts[2]))
    fac_ref, dep_ref = np.asarray(ref["facilitation"]), np.asarray(ref["depression"])
    d_fac = np.abs(fac - fac_ref)
    d_dep = np.abs(dep - dep_ref)
    max_diff = float(max(d_fac.max(), d_dep.max()))
    tol = 1e-3  # mV（同机同版本应逐位一致；跨环境容差）
    within = max_diff <= tol
    fac_ok = bool(len(fac) == n_pulses and fac[-1] >= fac[0] * 1.5)
    dep_ok = bool(len(dep) == n_pulses and dep[-1] <= dep[0] * 0.6)
    return dict(stp_regression_pass=bool(within and fac_ok and dep_ok),
                max_abs_diff_mv=max_diff, within_tol=bool(within),
                fac_ok=fac_ok, dep_ok=dep_ok,
                fac_meas=fac.tolist(), dep_meas=dep.tolist(),
                fac_ref=fac_ref.tolist(), dep_ref=dep_ref.tolist())


def _network_interface_check(params):
    """网络级接口就绪（§0 预注册 #1）：默认不启用；启用路径 mini-circuit 冒烟。"""
    from types import SimpleNamespace

    from neural_exploration.src.plasticity import (attach_subgraph_stdp,
                                                   stdp_network_connections)

    # 1) 母版 CSV 无 stdp_connections 段 → 默认不启用任何连接
    conns = stdp_network_connections(M6_PARAMS_CSV)
    assert conns == [], f"默认不应启用网络级 STDP 连接：{conns}"
    # 2) CSV 解析路径（临时文件含候选行）
    tmp_csv = os.path.join(DATA_DIR, "_m6_tmp_conn.csv")
    try:
        with open(tmp_csv, "w", encoding="utf-8") as f:
            f.write("section,pre,post,syn_type,note\n"
                    "stdp_connections,ASEL,AIYL,ampa,候选(G1后定稿)\n"
                    "stdp_connections,ALML,AVAL,ampa,候选(G1后定稿)\n")
        parsed = stdp_network_connections(tmp_csv)
        assert ("ASEL", "AIYL", "ampa") in parsed and \
            ("ALML", "AVAL", "ampa") in parsed, parsed
    finally:
        if os.path.exists(tmp_csv):
            os.remove(tmp_csv)
    # 3) enabled=False → no-op（不触碰 circuit）
    fake = SimpleNamespace()
    out = attach_subgraph_stdp(fake, connections=[("A", "B", "ampa")],
                               params=params, enabled=False)
    assert out is None and not hasattr(fake, "_m6_stdp"), "enabled=False 必须 no-op"
    # 4) enabled=True 冒烟：合成 2 神经元 mini-circuit（IF + g_ampa + 连接组事实）
    from brian2 import (NeuronGroup, Network, SpikeMonitor, TimedArray, amp,
                        defaultclock, ms, mV, nA, pF, start_scope)
    from neural_exploration.src.brian_env import configure_brian2
    from neural_exploration.src.worm_circuit import ChemRow, ConnectomeSpec

    configure_brian2()
    start_scope()
    defaultclock.dt = DT_MS * ms
    # 点神经元膜电容（1µF/cm² × 1.257e-5 cm² ≈ 12.57 pF，PointNeuron 同源）
    eqs = ("dv/dt = (-10*mV - v)/(10*ms) + (stim_i(t,i)/C_MEM) : volt\n"
           "dg_ampa/dt = -g_ampa/(3*ms) : siemens/meter**2")
    g = NeuronGroup(2, eqs, threshold="v > 0*mV", reset="v = -10*mV",
                    method="euler", refractory=1.0 * ms,
                    namespace={"C_MEM": 12.57 * pF})
    g.v = -10 * mV
    mini = SimpleNamespace(
        group=g, role_index={"n0": 0, "n1": 1}, names=["n0", "n1"],
        sub=ConnectomeSpec(neurons={"n0": "x", "n1": "x"},
                           chem=[ChemRow("n0", "n1", "ampa", 0.3, 0.1)]))
    syns = attach_subgraph_stdp(mini, [("n0", "n1", "ampa")], params=params,
                                enabled=True)
    assert len(syns) == 1 and len(syns[0]) == 1, "应构建 1 条 STDP 连接"
    n_steps = int(round(200.0 / DT_MS))
    arr = np.zeros((n_steps, 2)) * amp
    arr[500:510, 0] = 1.0 * nA
    arr[550:560, 1] = 1.0 * nA
    stim = TimedArray(arr, dt=DT_MS * ms, name="stim_i")
    sp = SpikeMonitor(g, "v", name="sp_mini")
    net = Network(g, syns[0], sp)
    net.run(200 * ms, namespace={"stim_i": stim})
    w_after = float(syns[0].w[0])
    assert int(len(sp.t)) >= 2, f"mini 至少 pre+post 各 1 次发放，实测 {len(sp.t)}"
    assert w_after > params.w0 + 0.005, \
        f"mini STDP 应产生 LTP（Δt≈5ms）：w={w_after}"
    return dict(network_connections_default=[], csv_parse_ok=True,
                enabled_false_noop=True, mini_circuit_w=w_after,
                mini_circuit_ok=True)


# --------------------------------------------------------------------- #
# 主验证
# --------------------------------------------------------------------- #
def run_p1_stdp(save_plot: bool = True, verbose: bool = True) -> dict:
    from neural_exploration.src.plasticity import (load_stdp_params,
                                                   write_stdp_params_csv)

    # ---- 0. 参数母版 CSV 定稿 + 回读闭环 ----
    params = load_stdp_params()          # 定稿默认（与 CSV 同源）
    write_stdp_params_csv(M6_PARAMS_CSV, params)
    params_loaded = load_stdp_params(M6_PARAMS_CSV)  # CSV 唯一定稿源
    for f in ("tau_plus_ms", "tau_minus_ms", "a_plus", "a_minus", "w0",
              "w_max", "eta", "tau_e_ms"):
        assert getattr(params_loaded, f) == getattr(params, f), f
    if verbose:
        print(f"[P1] STDP 参数定稿（m6_learning_params.csv stdp 段）："
              f"τ₊={params.tau_plus_ms} τ₋={params.tau_minus_ms} "
              f"A₊={params.a_plus} A₋={params.a_minus} "
              f"A₋/A₊={round(params.a_minus_over_a_plus, 4)} w_max={params.w_max}")

    # ---- 1. 理论标准表（预生成）----
    grid = np.concatenate([np.arange(-60, 0, 1.0), np.arange(1, 61, 1.0)])
    grid = np.round(grid, 6)
    theory_grid = _theory_at(grid, params)
    with open(M6_REF_CSV, "w", encoding="utf-8") as f:
        f.write("# M6 P1 STDP 理论标准表（文献理论曲线；B1b 预生成，"
                "validate_p1_stdp.py）\n")
        f.write("# ΔW_rel(Δt) = A₊·exp(−Δt/τ₊)（Δt>0，LTP 窗）/ "
                "−A₋·exp(Δt/τ₋)（Δt<0，LTD 窗）\n")
        f.write(f"# 参数：τ₊={params.tau_plus_ms}ms τ₋={params.tau_minus_ms}ms "
                f"A₊={params.a_plus}·w0 A₋={params.a_minus}·w0 "
                f"(A₋/A₊={round(params.a_minus_over_a_plus, 4)}) w0={params.w0} "
                f"w_max={params.w_max}（定稿：m6_learning_params.csv stdp 段）\n")
        f.write("dt_ms,dW_rel_theory,window\n")
        for dt, val in zip(grid, theory_grid):
            win = "LTP" if dt > 0 else "LTD"
            f.write(f"{dt:.1f},{val:.10f},{win}\n")

    # ---- 2. STDP 实测（协议全点 × 50 对）----
    sweep1 = _run_sweep(params, idx_base=0)
    # ---- 3. 确定性重跑（逐位一致）----
    sweep2 = _run_sweep(params, idx_base=10)
    det_keys = ("realized_dt", "dW_total", "w_final", "w_min", "w_max",
                "n_pre", "n_post")
    det_ok = all(s1[k] == s2[k] for s1, s2 in zip(sweep1, sweep2)
                 for k in det_keys)
    if verbose:
        print(f"[P1] 确定性重跑（2×10 协议点，逐位比较 {len(det_keys)} 项）"
              f"→ {'一致' if det_ok else '不一致！'}")

    # ---- 4. 判据：ΔW vs 理论（每点 |ΔW_rel 差| ≤ 0.2，理论取 realized Δt）----
    rows = []
    for r in sweep1:
        dW_meas = r["dW_total"] / N_PAIRS          # 每对 ΔW（w0=1 → ΔW_rel）
        theory = float(_theory_at([r["realized_dt"]], params)[0])
        diff = abs(dW_meas - theory)
        rows.append(dict(dt_ms=r["dt_nominal"], dt_ms_realized=r["realized_dt"],
                         dW_rel_meas=dW_meas, dW_rel_theory=theory,
                         dW_rel_diff=diff, ok=bool(diff <= 0.2),
                         w_final=r["w_final"], w_min=r["w_min"],
                         w_max=r["w_max"]))
    per_point_ok = all(r["ok"] for r in rows)
    max_diff = max(r["dW_rel_diff"] for r in rows)

    # ---- 5. 幅值比 A₋/A₊（对称 ±Δt 点）----
    ratios = []
    for a in (5.0, 10.0, 20.0, 40.0, 60.0):
        m_neg = next(r for r in rows if r["dt_ms"] == -a)
        m_pos = next(r for r in rows if r["dt_ms"] == +a)
        ratios.append(abs(m_neg["dW_rel_meas"]) / m_pos["dW_rel_meas"])
    ratio_mean = float(np.mean(ratios))
    ratio_ok = bool(abs(ratio_mean - params.a_minus_over_a_plus) <= 0.05)
    if verbose:
        print(f"[P1] 幅值比实测 |ΔW(−a)|/ΔW(+a) = {ratio_mean:.4f} "
              f"（预注册 A₋/A₊={round(params.a_minus_over_a_plus, 4)}，容差 0.05）"
              f"→ {'OK' if ratio_ok else 'FAIL'}")

    # ---- 6. 权重有界 [0, w_max] ----
    bounds_ok = all(0.0 <= r["w_min"] and r["w_max"] <= params.w_max
                    for r in rows)
    sat = _saturation_runs(params)
    sat_ok = bool(sat["ltp_w"] == params.w_max and sat["ltd_w"] == 0.0)
    if verbose:
        print(f"[P1] 权重有界：协议内 w∈[{min(r['w_min'] for r in rows):.4f},"
              f"{max(r['w_max'] for r in rows):.4f}]（[0,{params.w_max}]）✓；"
              f"饱和验证 LTP→{sat['ltp_w']}（=w_max）、LTD→{sat['ltd_w']}（=0）"
              f"→ {'OK' if sat_ok else 'FAIL'}")

    # ---- 7. 三因子冒烟（informational）----
    tf = _three_factor_smoke(params)
    tf_ok = bool(0.0 < tf["dW_M1"] < tf["w_max_minus_w0"] and
                 tf["dW_M0"] == 0.0)
    if verbose:
        print(f"[P1] 三因子冒烟（informational）：M=1 → Δw={tf['dW_M1']:.4f}"
              f"（∈(0, {tf['w_max_minus_w0']})）；M=0 → Δw={tf['dW_M0']}"
              f"（消融前提）→ {'OK' if tf_ok else 'FAIL'}")

    # ---- 8. STP 回归（M2 P3 协议 vs 冻结 m2_stp.csv）----
    stp = _stp_regression()
    if verbose:
        print(f"[P1] STP 回归：max|ΔEPSP|={stp['max_abs_diff_mv']:.2e} mV"
              f"（≤1e-3）；易化末/首={stp['fac_meas'][-1]/stp['fac_meas'][0]:.3f}"
              f"（≥1.5）、抑制末/首={stp['dep_meas'][-1]/stp['dep_meas'][0]:.4f}"
              f"（≤0.6）→ {'PASS' if stp['stp_regression_pass'] else 'FAIL'}")

    # ---- 9. 网络级接口就绪 ----
    net_ok = True
    try:
        netc = _network_interface_check(params)
        if verbose:
            print(f"[P1] 网络级接口：默认不启用 ✓；CSV 解析 ✓；enabled=False "
                  f"no-op ✓；mini-circuit enabled=True → w={netc['mini_circuit_w']:.4f}"
                  f"（LTP>w0）✓")
    except Exception as exc:  # noqa: BLE001 —— 接口冒烟失败记入 summary
        net_ok = False
        netc = dict(error=str(exc))
        if verbose:
            print(f"[P1] 网络级接口冒烟失败：{exc}")

    pass_ = bool(per_point_ok and ratio_ok and bounds_ok and sat_ok and
                 det_ok and stp["stp_regression_pass"] and net_ok)

    # ---- 10. 落盘 m6_p1_stdp.csv ----
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(M6_P1_CSV, "w", encoding="utf-8") as f:
        f.write("# M6 P1 STDP 实测 vs 理论（成对协议 Δt∈{−60..+60}ms × 50 对；"
                "PointNeuron + StdpSynapse ampa；确定性 seed=0）\n")
        f.write("# 判据（预注册 §0 P1）：每点 |ΔW_rel 差| ≤ 0.2；幅值比 "
                "A₋/A₊=0.9；权重有界 [0,w_max]；确定性重跑逐位一致\n")
        f.write("dt_ms,dt_ms_realized,dW_rel_meas,dW_rel_theory,"
                "dW_rel_diff,ok,w_final,w_min,w_max\n")
        for r in rows:
            f.write(f"{r['dt_ms']:.1f},{r['dt_ms_realized']:.4f},"
                    f"{r['dW_rel_meas']:.10f},{r['dW_rel_theory']:.10f},"
                    f"{r['dW_rel_diff']:.2e},{r['ok']},"
                    f"{r['w_final']:.6f},{r['w_min']:.6f},{r['w_max']:.6f}\n")
        f.write("# summary\n")
        f.write(f"# stdp_pass={pass_}\n")
        f.write(f"# per_point_ok={per_point_ok} (max|diff|={max_diff:.2e}, "
                f"tol=0.2)\n")
        f.write(f"# amplitude_ratio_meas={ratio_mean:.4f} "
                f"(pre-registered {round(params.a_minus_over_a_plus, 4)}, tol=0.05, "
                f"ok={ratio_ok})\n")
        f.write(f"# bounds_protocol_ok={bounds_ok} "
                f"(w_min={min(r['w_min'] for r in rows):.4f}, "
                f"w_max={max(r['w_max'] for r in rows):.4f})\n")
        f.write(f"# bounds_saturation_ok={sat_ok} "
                f"(ltp_w={sat['ltp_w']}, ltd_w={sat['ltd_w']}, "
                f"w_max={params.w_max})\n")
        f.write(f"# deterministic_ok={det_ok}\n")
        f.write(f"# stp_regression_pass={stp['stp_regression_pass']} "
                f"(max_abs_diff_mv={stp['max_abs_diff_mv']:.2e})\n")
        f.write(f"# three_factor_smoke_ok={tf_ok} "
                f"(dW_M1={tf['dW_M1']:.4f}, dW_M0={tf['dW_M0']})\n")
        f.write(f"# network_interface_ok={net_ok}\n")

    # ---- 11. 出图 ----
    if save_plot:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
        ax = axes[0]
        ax.axhline(0, color="k", lw=0.6)
        ax.axvline(0, color="k", lw=0.6)
        ax.fill_between([0, 61], -0.012, 0.012, color="tab:blue", alpha=0.06)
        ax.fill_between([-61, 0], -0.012, 0.012, color="tab:red", alpha=0.06)
        ax.plot(grid, theory_grid, lw=1.6, color="tab:blue",
                label=f"theory: A₊·exp(−|Δt|/τ), τ={params.tau_plus_ms}ms")
        m_dt = [r["dt_ms"] for r in rows]
        m_dw = [r["dW_rel_meas"] for r in rows]
        ax.plot(m_dt, m_dw, "o", ms=5, color="k", mec="w", zorder=5,
                label="measured (50 pairs)")
        ax.text(0.02, 0.94,
                f"$A_+={params.a_plus}$, $A_-={params.a_minus}$\n"
                f"$A_-/A_+={round(params.a_minus_over_a_plus, 4)}$（实测 "
                f"{ratio_mean:.3f}）",
                transform=ax.transAxes, fontsize=8, va="top")
        ax.set_xlabel("Δt = t_post − t_pre (ms)")
        ax.set_ylabel("ΔW_rel (Δw/w0)")
        ax.set_title(f"P1: pairwise STDP vs theory\n"
                     f"LTP Δt>0 / LTD Δt<0 — pass={pass_}")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.3)

        ax = axes[1]
        diffs = [r["dW_rel_diff"] for r in rows]
        ax.bar(range(len(rows)), np.maximum(diffs, 1e-18), color="tab:orange")
        ax.axhline(0.2, color="r", ls="--", lw=1.2,
                   label="pre-registered tol 0.2")
        ax.set_yscale("log")
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([f"{r['dt_ms']:.0f}" for r in rows], rotation=45,
                           fontsize=7)
        ax.set_ylabel("|ΔW_meas − ΔW_theory|")
        ax.set_title(f"per-point residual — max={max_diff:.1e}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[2]
        x = np.arange(1, 11)
        ax.plot(x, stp["fac_meas"], "o-", color="tab:green", ms=4,
                label="facilitation (re-run)")
        ax.plot(x, stp["fac_ref"], "s--", color="tab:green", alpha=0.5,
                label="M2 frozen")
        ax.plot(x, stp["dep_meas"], "o-", color="tab:red", ms=4,
                label="depression (re-run)")
        ax.plot(x, stp["dep_ref"], "s--", color="tab:red", alpha=0.5,
                label="M2 frozen")
        ax.set_xlabel("pulse #")
        ax.set_ylabel("EPSP (mV)")
        ax.set_title(f"STP regression (50Hz×10) — "
                     f"max|Δ|={stp['max_abs_diff_mv']:.1e} mV")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        fig.suptitle("M6 P1: STDP component vs literature curve"
                     " (Bi & Poo 1998 exp windows)", y=1.02)
        fig.tight_layout()
        fig.savefig(M6_P1_PNG, dpi=150, bbox_inches="tight")
        plt.close(fig)

    summary = dict(
        pass_=pass_, stdp_pass=pass_,
        per_point_ok=per_point_ok, max_abs_diff=max_diff,
        amplitude_ratio_meas=ratio_mean,
        amplitude_ratio_reg=round(params.a_minus_over_a_plus, 4),
        amplitude_ratio_ok=ratio_ok,
        bounds_ok=bounds_ok, saturation=sat, saturation_ok=sat_ok,
        deterministic_ok=det_ok,
        three_factor=tf, three_factor_ok=tf_ok,
        stp_regression=stp,
        network_interface=netc,
        params=dict(tau_plus_ms=params.tau_plus_ms,
                    tau_minus_ms=params.tau_minus_ms,
                    a_plus=params.a_plus, a_minus=params.a_minus,
                    a_minus_over_a_plus=round(params.a_minus_over_a_plus, 4),
                    w0=params.w0, w_max=params.w_max, eta=params.eta,
                    tau_e_ms=params.tau_e_ms),
        points=rows,
        report_png=M6_P1_PNG, report_csv=M6_P1_CSV,
        reference_csv=M6_REF_CSV, params_csv=M6_PARAMS_CSV,
    )
    return summary


if __name__ == "__main__":
    res = run_p1_stdp()
    slim = {k: v for k, v in res.items() if k not in ("points",)}
    print(json.dumps(slim, indent=2, ensure_ascii=False, default=str))
    print()
    print("=" * 72)
    print("M6 P1 STDP 实测 vs 理论（ΔW_rel，每对）")
    print(f"{'Δt(ms)':>8} {'realized':>10} {'meas':>12} {'theory':>12} "
          f"{'|diff|':>10}  ok")
    for r in res["points"]:
        print(f"{r['dt_ms']:>8.1f} {r['dt_ms_realized']:>10.4f} "
              f"{r['dW_rel_meas']:>12.8f} {r['dW_rel_theory']:>12.8f} "
              f"{r['dW_rel_diff']:>10.2e}  {r['ok']}")
    print("=" * 72)
    print(f"幅值比 A₋/A₊ 实测={res['amplitude_ratio_meas']:.4f} "
          f"(预注册 {res['amplitude_ratio_reg']})")
    print(f"权重有界 [0,{res['params']['w_max']}]："
          f"协议内 w_max={max(r['w_max'] for r in res['points']):.4f}；"
          f"饱和 LTP→{res['saturation']['ltp_w']} / LTD→{res['saturation']['ltd_w']}")
    print(f"确定性重跑：{'逐位一致' if res['deterministic_ok'] else '不一致！'}")
    print(f"STP 回归：max|ΔEPSP|={res['stp_regression']['max_abs_diff_mv']:.2e} mV "
          f"({'PASS' if res['stp_regression']['stp_regression_pass'] else 'FAIL'})")
    print(f"P1 {'PASS' if res['pass_'] else 'FAIL'}"
          f"（G0 组件门：{'通过' if res['pass_'] else '不通过'}）")
