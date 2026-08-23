"""M2 参考解：NEURON 9.0.1 cvode 高精度（化学突触）+ scipy 独立数值解（缝隙连接）。

清单《生物仿真M2实施清单》§3（步骤 2）：两个 NEURON 多隔室神经元（复用 M1 形态学）
+ ExpSyn(AMPA 类) / NMDASyn(自编译 mod，Mg²⁺ 阻断) + 多试次释放失败统计；
输出 `data/m2_synapse_ref.npz`：

  - t_ms                    均匀网格（dt=0.01ms）
  - epsp_ampa_post_mv       单刺激 AMPA EPSP（post soma）
  - ipsp_gaba_post_mv       单刺激 GABA_A IPSP
  - epsp_train_ampa_mv      50Hz×10 脉冲 AMPA EPSP 序列（P3 参考）
  - nmda_post_mv            NMDA EPSP（Mg²⁺ 阻断下、轻度去极化 hold，P5 参考）
  - nmda_g_peak_vs_v        NMDA 释放增量峰值 g 对 hold 电位（B(V) 电压依赖实测）
  - gap_ref_*               scipy 高精度耦合解（缝隙连接参考，见下）
  - failure_rate_neuron     多试次释放失败率（二项量子模型，P2 参考）

缝隙连接参考的说明（m2_env_notes §L2）：本机 pip 安装的 NEURON 9.0.1 运行时不导出
经典 gap.mod 所需的 EXTERNAL vother 符号（dlopen 报 symbol not found），故缝隙连接
参考改用 **scipy solve_ivp 独立高精度解**（两等势 HH 胞体 + 欧姆耦合，同方程同参数）：
耦合 PSP 形状/量级/双向性与 Brian2 逐项可比，P4 判据为定性特征（近即时/双向/衰减），
不依赖 NEURON 形态学复刻。

用法：
  .venv-neuro/bin/python -m neural_exploration.tools.build_synapse_ref
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.src.synapse_model import load_synapse_params  # noqa: E402
from neural_exploration.tools.load_morphology import (  # noqa: E402
    V0, load_morphology,
)
from neural_exploration.tools.build_neuron_ref import (  # noqa: E402
    SOMA_AREA_CM2, build_neuron,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
REF_NPZ = os.path.join(DATA_DIR, "m2_synapse_ref.npz")
NMODL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nmodl")

# 与 Brian2 侧一致的刺激/记录约定（清单 §2/§5）
DT_OUT = 0.01          # ms
T_TOTAL = 120.0        # ms（单脉冲实验）
T_TRAIN = 320.0        # ms（50Hz×10）
T_TRIAL = 70.0         # ms（失败统计单试次）
PULSE_START = 50.0     # ms（等静息瞬态漂移衰减后再刺激，M1 env_notes §L3）
PULSE_DUR = 1.0        # ms
PULSE_UA_CM2 = 20.0    # µA/cm²（从静息态可靠发放的最小稳定幅度，实测）
FREQ_HZ = 50.0
N_PULSES = 10
N_TRIALS = 100         # 释放失败统计试次数

# NMDA 电压依赖扫描的 hold 电位（mV）
NMDA_HOLDS_MV = (-80.0, -60.0, -40.0, -20.0, 0.0)


def _load_nmda_mechanism():
    """加载自编译 NMDASyn 机制（tools/nmodl/arm64）。"""
    from neuron import h

    if hasattr(h, "NMDASyn"):
        return h
    arch = "arm64" if os.path.exists(os.path.join(NMODL_DIR, "arm64")) else "x86_64"
    dll = os.path.join(NMODL_DIR, arch, "libnrnmech.dylib")
    if not os.path.exists(dll):
        raise FileNotFoundError(f"NMDASyn 机制未编译：{dll}（在 tools/nmodl 运行 nrnivmodl）")
    h.nrn_load_dll(dll)
    return h


def _run_chemical(params, g_ns: float, tau_ms: float, e_rev_mv: float,
                  n_pulses: int = 1, t_total: float = T_TOTAL,
                  mg_mm: float = 1.2, nmda: bool = False) -> dict:
    """NEURON 化学突触单次运行 → {t, v_pre_node3, v_post_soma, g}（均匀网格）。

    nmda=True 时用自编译 NMDASyn（Mg²⁺ 阻断方程与 Brian2 一致）。
    """
    from neuron import h

    h = _load_nmda_mechanism()
    spec = load_morphology()
    pre = build_neuron(spec, clear=True, name_prefix="pre_")
    post = build_neuron(spec, clear=False, name_prefix="post_")

    h.load_file("stdrun.hoc")
    h.celsius = 6.3
    h.v_init = V0

    # pre 刺激（脉冲串）——注意保留 IClamp 引用（NEURON Python 会被 GC 回收，
    # 否则只剩最后一个脉冲生效，M2 实测踩坑）
    clamps = []
    if n_pulses > 1:
        for k in range(n_pulses):
            cl = h.IClamp(pre["soma"](0.5))
            cl.delay = PULSE_START + k * 1000.0 / FREQ_HZ
            cl.dur = PULSE_DUR
            cl.amp = PULSE_UA_CM2 * 1e-6 * SOMA_AREA_CM2 * 1e9
            clamps.append(cl)
    else:
        cl = h.IClamp(pre["soma"](0.5))
        cl.delay = PULSE_START
        cl.dur = PULSE_DUR
        cl.amp = PULSE_UA_CM2 * 1e-6 * SOMA_AREA_CM2 * 1e9
        clamps.append(cl)

    # 突触（ExpSyn / NMDASyn）
    if nmda:
        syn = h.NMDASyn(post["soma"](0.5))
        syn.tau = tau_ms
        syn.e = e_rev_mv
        syn.gmax = g_ns * 1e-3      # µS（机制内部以 µS/nA 为单位，与 ExpSyn 一致）
        syn.mg = mg_mm
        w = g_ns * 1e-3             # NMDASyn NET_RECEIVE weight 单位 µS
    else:
        syn = h.ExpSyn(post["soma"](0.5))
        syn.tau = tau_ms
        syn.e = e_rev_mv
        w = g_ns * 1e-3              # ExpSyn weight 单位 µS

    nc = h.NetCon(pre["node3"](0.5)._ref_v, syn, sec=pre["node3"])
    nc.threshold = -20.0
    nc.delay = 0.1
    nc.weight[0] = w

    tvec = h.Vector(); tvec.record(h._ref_t)
    vpre = h.Vector(); vpre.record(pre["node3"](0.5)._ref_v)
    vpost = h.Vector(); vpost.record(post["soma"](0.5)._ref_v)
    grec = h.Vector(); grec.record(syn._ref_g)

    cvode = h.CVode()
    cvode.active(1)
    cvode.atol(1e-8)
    cvode.rtol(1e-8)
    h.tstop = t_total
    h.run()

    t_unif = np.arange(0.0, t_total + DT_OUT / 2, DT_OUT)
    t_irr = np.array(tvec)
    order = np.argsort(t_irr)
    return dict(
        t_ms=t_unif,
        v_pre_node3=np.interp(t_unif, t_irr[order], np.array(vpre)[order]),
        v_post_soma=np.interp(t_unif, t_irr[order], np.array(vpost)[order]),
        g=np.interp(t_unif, t_irr[order], np.array(grec)[order]),
    )


def _run_failure_trials(params, n_trials: int = N_TRIALS) -> dict:
    """多试次释放失败：每试次 k ~ Binomial(n_vesicles, p_release) 个量子。

    单次事件 → NetCon weight = k·g_quantum（µS）；k=0 为释放失败。
    返回失败率 + 量子数分布（P2 参考）。
    """
    from neuron import h

    h = _load_nmda_mechanism()
    spec = load_morphology()
    p = params["ampa"]
    g_quantum_ns = p.g_max_ns
    rng = np.random.default_rng(2026)

    h.load_file("stdrun.hoc")
    h.celsius = 6.3
    h.v_init = V0

    failures = 0
    measured_failures = 0
    quanta_list = []
    for _ in range(n_trials):
        pre = build_neuron(spec, clear=True, name_prefix="pre_")
        post = build_neuron(spec, clear=False, name_prefix="post_")
        exp = h.ExpSyn(post["soma"](0.5))
        exp.tau = p.tau_ms
        exp.e = p.e_rev_mv
        nc = h.NetCon(pre["node3"](0.5)._ref_v, exp, sec=pre["node3"])
        nc.threshold = -20.0
        nc.delay = 0.1
        k = int(rng.binomial(p.n_vesicles, p.p_release))
        nc.weight[0] = k * g_quantum_ns * 1e-3
        cl = h.IClamp(pre["soma"](0.5))
        cl.delay = PULSE_START
        cl.dur = PULSE_DUR
        cl.amp = PULSE_UA_CM2 * 1e-6 * SOMA_AREA_CM2 * 1e9
        tvec = h.Vector(); tvec.record(h._ref_t)
        vrec = h.Vector(); vrec.record(post["soma"](0.5)._ref_v)
        cvode = h.CVode(); cvode.active(1); cvode.atol(1e-8); cvode.rtol(1e-8)
        h.tstop = T_TRIAL
        h.run()
        t = np.array(tvec)
        v = np.array(vrec)
        # 局部基线 + 发放后窗（node3 发放 ≈ PULSE_START+2.9ms）
        base_win = (t > PULSE_START - 3.0) & (t < PULSE_START - 0.5)
        base = float(np.median(v[base_win])) if base_win.sum() else float(np.median(v))
        psp_win = (t > PULSE_START + 3.2) & (t < PULSE_START + 18.0)
        peak = float(v[psp_win].max() - base) if psp_win.sum() else float(v.max() - base)
        if k == 0:
            failures += 1
        if peak < 0.05:
            measured_failures += 1
        quanta_list.append(k)
    return dict(
        n_trials=n_trials,
        failure_rate=float(failures / n_trials),
        measured_failure_rate=float(measured_failures / n_trials),
        quanta=np.asarray(quanta_list),
        p_release=p.p_release,
        n_vesicles=p.n_vesicles,
        expected_failure=(1 - p.p_release) ** p.n_vesicles,
    )


def _run_gap_reference(g_gap_ns: float, t_total: float = T_TOTAL) -> dict:
    """scipy solve_ivp 高精度缝隙连接参考（两等势 HH 胞体 + 欧姆耦合）。

    与 Brian2 同方程同参数（M1 hh_spec 参数、胞体面积 π·d²、g_gap 耦合），
    LSODA rtol=1e-10 作为独立高精度解。说明见模块 docstring 与 m2_env_notes §L2。

    单位（沿用 M1 约定）：v 为 mV、电流 µA/cm²、Cm µF/cm²、时间 ms；
    I_gap[µA/cm²] = g_gap[nS]·ΔV[mV]·1e-6 / area[cm²]（1 nS·1 mV = 1 pA = 1e-6 µA）。
    """
    from scipy.integrate import solve_ivp

    from neural_exploration.tools.hh_spec import CM, EK, EL, ENA, GK, GL, GNA, steady_state

    d_um = 20.0
    area_cm2 = np.pi * (d_um * 1e-4) ** 2
    g_gap = g_gap_ns * 1e-9                     # S

    def rates(v):
        a_m = 0.1 * (v + 40.0) / (1.0 - np.exp(-(v + 40.0) / 10.0))
        b_m = 4.0 * np.exp(-(v + 65.0) / 18.0)
        a_h = 0.07 * np.exp(-(v + 65.0) / 20.0)
        b_h = 1.0 / (1.0 + np.exp(-(v + 35.0) / 10.0))
        a_n = 0.01 * (v + 55.0) / (1.0 - np.exp(-(v + 55.0) / 10.0))
        b_n = 0.125 * np.exp(-(v + 65.0) / 80.0)
        return a_m, b_m, a_h, b_h, a_n, b_n

    def f(t, y):
        v1, m1, h1, n1, v2, m2, h2, n2 = y
        a_m1, b_m1, a_h1, b_h1, a_n1, b_n1 = rates(v1)
        a_m2, b_m2, a_h2, b_h2, a_n2, b_n2 = rates(v2)
        stim = PULSE_UA_CM2 if PULSE_START <= t < PULSE_START + PULSE_DUR else 0.0  # µA/cm²
        i_na1 = GNA * m1 ** 3 * h1 * (v1 - ENA)   # µA/cm²（外向正；mS·mV=µA）
        i_k1 = GK * n1 ** 4 * (v1 - EK)
        i_l1 = GL * (v1 - EL)
        i_na2 = GNA * m2 ** 3 * h2 * (v2 - ENA)
        i_k2 = GK * n2 ** 4 * (v2 - EK)
        i_l2 = GL * (v2 - EL)
        i_gap = g_gap * (v1 - v2) * 1e3 / area_cm2   # µA/cm²：g[S]·ΔV[mV]·1e-3[A]·1e6[µA/A]/area
        dm1 = a_m1 * (1 - m1) - b_m1 * m1
        dh1 = a_h1 * (1 - h1) - b_h1 * h1
        dn1 = a_n1 * (1 - n1) - b_n1 * n1
        dm2 = a_m2 * (1 - m2) - b_m2 * m2
        dh2 = a_h2 * (1 - h2) - b_h2 * h2
        dn2 = a_n2 * (1 - n2) - b_n2 * n2
        return [
            (stim - i_na1 - i_k1 - i_l1 - i_gap) / CM, dm1, dh1, dn1,
            (-i_na2 - i_k2 - i_l2 + i_gap) / CM, dm2, dh2, dn2,
        ]

    m0, h0, n0 = steady_state(V0)
    y0 = [V0, m0, h0, n0, V0, m0, h0, n0]
    sol = solve_ivp(f, (0, t_total), y0, method="LSODA",
                    rtol=1e-10, atol=1e-12, dense_output=True)
    t_unif = np.arange(0.0, t_total + DT_OUT / 2, DT_OUT)
    ys = sol.sol(t_unif)
    return dict(t_ms=t_unif, v1_mv=ys[0], v2_mv=ys[4])


def _run_nmda_hold_scan(params, holds_mv=NMDA_HOLDS_MV) -> list:
    """NMDA g_peak 对 hold 电位扫描（passive 胞体，全范围 -80..0mV）。

    post 用等势被动胞体（pas, g=0.3mS/cm², e=-54.4），DC IClamp 精确设定 hold；
    突触事件由 pre（完整 HH 神经元）node3 发放驱动（与 P1 同路径）。
    g 峰值 = gmax·B(v_hold)，直接实证 Mg²⁺ 去阻断的电压依赖（P5 参考）。
    """
    from neuron import h

    h = _load_nmda_mechanism()
    spec = load_morphology()
    p = params["nmda"]
    out = []
    for vh in holds_mv:
        pre = build_neuron(spec, clear=True, name_prefix="pre_")
        # post：等势被动胞体
        post = h.Section(name="post_soma")
        post.L = 20.0
        post.diam = 20.0
        post.nseg = 1
        post.Ra = 0.001
        post.insert("pas")
        for seg in post:
            seg.pas.g = 0.0003     # S/cm²
            seg.pas.e = -54.4
        h.load_file("stdrun.hoc")
        h.celsius = 6.3
        h.v_init = V0
        area_cm2 = np.pi * (20.0 * 1e-4) ** 2
        dc = h.IClamp(post(0.5))
        dc.delay = 0.0
        dc.dur = T_TOTAL
        dc.amp = (vh - (-54.4)) * 1e-3 * 0.0003 * area_cm2 * 1e9   # nA（V·S=A）
        cl = h.IClamp(pre["soma"](0.5))
        cl.delay = PULSE_START
        cl.dur = PULSE_DUR
        cl.amp = PULSE_UA_CM2 * 1e-6 * SOMA_AREA_CM2 * 1e9
        nmd = h.NMDASyn(post(0.5))
        nmd.tau = p.tau_ms
        nmd.e = p.e_rev_mv
        nmd.gmax = p.g_max_ns * 1e-3   # µS
        nmd.mg = p.mg_mm
        nc = h.NetCon(pre["node3"](0.5)._ref_v, nmd, sec=pre["node3"])
        nc.threshold = -20.0
        nc.delay = 0.1
        nc.weight[0] = p.g_max_ns * 1e-3   # µS
        tvec = h.Vector(); tvec.record(h._ref_t)
        grec = h.Vector(); grec.record(nmd._ref_g)
        vrec = h.Vector(); vrec.record(post(0.5)._ref_v)
        cvode = h.CVode(); cvode.active(1); cvode.atol(1e-8); cvode.rtol(1e-8)
        h.tstop = T_TOTAL
        h.run()
        t = np.array(tvec)
        v = np.array(vrec)
        g = np.array(grec)
        # 事件前实测 hold（t = PULSE_START-0.5ms 附近）
        t_event = PULSE_START + 3.0   # 实际发放约在此附近；取事件前窗
        win = (t > PULSE_START + 2.0) & (t < PULSE_START + 2.9)
        v_at = float(np.median(v[win])) if win.sum() else float(np.median(v))
        b_theory = 1.0 / (1.0 + p.mg_mm * np.exp(-0.062 * v_at) / 3.57)
        out.append(dict(
            hold_requested_mv=vh, v_actual_mv=v_at,
            g_peak_ns=float(g.max()) * 1e3,        # µS → nS
            g_peak_theory_ns=p.g_max_ns * b_theory,
            b_theory=b_theory,
        ))
    return out


def run_reference(out_npz: str = REF_NPZ) -> str:
    """生成全部参考数据并落盘 npz。"""
    os.makedirs(DATA_DIR, exist_ok=True)
    params = load_synapse_params()

    out = {}
    # P1：AMPA EPSP / GABA IPSP 单刺激
    r = _run_chemical(params, params["ampa"].g_max_ns, params["ampa"].tau_ms,
                      params["ampa"].e_rev_mv, n_pulses=1, t_total=T_TOTAL)
    out["t_ms"] = r["t_ms"]
    out["epsp_ampa_post_mv"] = r["v_post_soma"]
    out["epsp_ampa_pre_node3_mv"] = r["v_pre_node3"]
    r = _run_chemical(params, params["gaba"].g_max_ns, params["gaba"].tau_ms,
                      params["gaba"].e_rev_mv, n_pulses=1, t_total=T_TOTAL)
    out["ipsp_gaba_post_mv"] = r["v_post_soma"]

    # P3：50Hz×10 脉冲串（AMPA）
    r = _run_chemical(params, params["ampa"].g_max_ns, params["ampa"].tau_ms,
                      params["ampa"].e_rev_mv, n_pulses=N_PULSES, t_total=T_TRAIN)
    out["t_train_ms"] = r["t_ms"]
    out["epsp_train_ampa_post_mv"] = r["v_post_soma"]
    out["epsp_train_pre_node3_mv"] = r["v_pre_node3"]

    # P2：释放失败统计
    out["failure"] = _run_failure_trials(params)

    # P5：NMDA EPSP（mg=0 形状对比用，另存）+ Mg²⁺ 阻断（mg=1.2）+ 电压依赖扫描
    r = _run_chemical(params, params["nmda"].g_max_ns, params["nmda"].tau_ms,
                      params["nmda"].e_rev_mv, n_pulses=1, t_total=T_TOTAL,
                      nmda=True, mg_mm=0.0)
    out["t_nmda_ms"] = r["t_ms"]
    out["nmda_post_mv_nomg"] = r["v_post_soma"]
    r = _run_chemical(params, params["nmda"].g_max_ns, params["nmda"].tau_ms,
                      params["nmda"].e_rev_mv, n_pulses=1, t_total=T_TOTAL,
                      nmda=True, mg_mm=params["nmda"].mg_mm)
    out["nmda_post_mv"] = r["v_post_soma"]
    out["nmda_g_vs_v"] = _run_nmda_hold_scan(params)

    # P4：缝隙连接参考（scipy 高精度解）
    gap = _run_gap_reference(params["gap"].g_max_ns)
    out["gap_ref_t_ms"] = gap["t_ms"]
    out["gap_ref_v1_mv"] = gap["v1_mv"]
    out["gap_ref_v2_mv"] = gap["v2_mv"]

    meta = dict(
        engine="NEURON 9.0.1 cvode (chemical/NMDA) + scipy solve_ivp LSODA (gap)",
        celsius=6.3,
        dt_out_ms=DT_OUT,
        pulse_start_ms=PULSE_START, pulse_dur_ms=PULSE_DUR,
        pulse_uA_cm2=PULSE_UA_CM2, freq_hz=FREQ_HZ, n_pulses=N_PULSES,
        n_trials=N_TRIALS,
        gap_ref_note="scipy 独立高精度解（等势 HH 胞体对 + 欧姆耦合；见 m2_env_notes §L2）",
    )
    np.savez(out_npz, **out, meta=np.array(meta, dtype=object))
    return out_npz


if __name__ == "__main__":
    out = run_reference()
    d = np.load(out, allow_pickle=True)
    print(f"参考解已写入: {out}")
    print("键:", [k for k in d.files if k != "meta"])
    print("failure:", d["failure"].item())
    for row in d["nmda_g_vs_v"].item():
        print("  nmda hold:", {k: round(v, 4) if isinstance(v, float) else v
                               for k, v in row.items()})
    print("meta:", d["meta"].item())
