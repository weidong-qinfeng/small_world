"""P5 验证：受体亚型区分——AMPA（快、短）vs NMDA（慢、Mg²⁺ 阻断、电压依赖）。

清单 §0 P5 / §5.5，至少各复现一个特征：
  1. AMPA 快：EPSP 衰减 τ≈3ms（τ_ampa）；
  2. NMDA 慢：EPSP 衰减 τ≈100ms（τ_nmda）；
  3. Mg²⁺ 阻断：静息下 mg=1.2 的 NMDA EPSP ≪ mg=0（幅度比 ≈ B(V_rest)≈0.06）；
  4. 电压依赖：B(V) 曲线（Jahr–Stevens 方程）与 NEURON 参考的实测
     g_peak/gmax 逐点一致（相对误差 <5%）。
输出：reports/neuro/m2_receptor_subtypes.png + data/m2_receptor_subtypes.csv
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.tools.build_synapse_ref import (  # noqa: E402
    PULSE_START, REF_NPZ,
)
from neural_exploration.tools.synapse_metrics import psp_amplitudes  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
REPORTS_DIR = os.path.join(ROOT, "neural_exploration", "reports", "neuro")
REPORT_PNG = os.path.join(REPORTS_DIR, "m2_receptor_subtypes.png")
REPORT_CSV = os.path.join(DATA_DIR, "m2_receptor_subtypes.csv")

MG_BLOCK_A = 0.062
MG_BLOCK_B = 3.57


def _b_theory(v_mv, mg_mm=1.2):
    return 1.0 / (1.0 + mg_mm * np.exp(-MG_BLOCK_A * v_mv) / MG_BLOCK_B)


def _conductance_tau(g_trace, t_ms):
    """受体电导指数衰减 τ（ms）：释放跳变 → 衰减至峰值 10%。"""
    g = np.asarray(g_trace, dtype=float)
    t = np.asarray(t_ms, dtype=float)
    i = int(np.argmax(g))
    if g[i] <= 1e-12:
        return None
    y = np.clip(g, 1e-12, None)
    floor = 0.10 * g[i]
    m = (np.arange(len(g)) > i + 2) & (g > floor)
    if m.sum() < 5:
        return None
    k, _ = np.polyfit(t[m] - t[i], np.log(y[m]), 1)
    return -1.0 / k if k < 0 else None


def run_p5(save_plot: bool = True) -> dict:
    from neural_exploration.src.neuron_pair import NeuronPair

    ref = np.load(REF_NPZ, allow_pickle=True)
    t_ref = ref["t_ms"]

    # ---- AMPA EPSP（快）----
    pair = NeuronPair(t_total_ms=160.0, seed=0)
    pair.add_chemical("ampa", g_max_ns=0.5, p_release=1.0, n_vesicles=1)
    r_a = pair.run(pre_pulses=[(PULSE_START, 1.0, 20.0, "soma")],
                   record=["pre_node3", "post_soma"], record_g=["g_ampa"])
    amps_a = psp_amplitudes(r_a.t_ms, r_a.v_mv["post_soma"], r_a.spike_times_ms["pre_node3"],
                            tail_window_ms=60.0)
    # 受体动力学 τ 直接对电导拟合（膜电位有 HH K⁺ 激活的 undershoot，非纯指数）
    tau_a = _conductance_tau(r_a.g["g_ampa"], r_a.t_ms)

    # ---- NMDA EPSP（慢；mg=0 以获得干净形状）----
    pair = NeuronPair(t_total_ms=300.0, seed=0)
    pair.add_chemical("nmda", g_max_ns=1.0, p_release=1.0, n_vesicles=1, mg_mm=0.0)
    r_n = pair.run(pre_pulses=[(PULSE_START, 1.0, 20.0, "soma")],
                   record=["pre_node3", "post_soma"], record_g=["g_nmda"])
    amps_n = psp_amplitudes(r_n.t_ms, r_n.v_mv["post_soma"], r_n.spike_times_ms["pre_node3"],
                            tail_window_ms=200.0)
    tau_n = _conductance_tau(r_n.g["g_nmda"], r_n.t_ms)

    # ---- Mg²⁺ 阻断：mg=1.2 vs mg=0 的 NMDA EPSP 幅度比 ----
    pair = NeuronPair(t_total_ms=300.0, seed=0)
    pair.add_chemical("nmda", g_max_ns=1.0, p_release=1.0, n_vesicles=1, mg_mm=1.2)
    r_nm = pair.run(pre_pulses=[(PULSE_START, 1.0, 20.0, "soma")],
                    record=["pre_node3", "post_soma"])
    amps_nm = psp_amplitudes(r_nm.t_ms, r_nm.v_mv["post_soma"], r_nm.spike_times_ms["pre_node3"],
                             tail_window_ms=200.0)
    mg_ratio = float(amps_nm[0] / amps_n[0]) if amps_n[0] > 0 else float("nan")
    # 理论比：B(V_rest)（post 静息 ≈ -63.1mV）
    b_rest = float(_b_theory(-63.1, 1.2))

    # 形状图用的相对时间轴
    t_rel_a = r_a.t_ms - r_a.spike_times_ms["pre_node3"][0]
    t_rel_n = r_n.t_ms - r_n.spike_times_ms["pre_node3"][0]

    # ---- 电压依赖：NEURON 参考实测 g_peak/gmax vs B(V) ----
    rows = list(ref["nmda_g_vs_v"])
    errs = [abs(r["g_peak_ns"] / r["g_peak_theory_ns"] - 1.0) for r in rows]
    vdep_ok = bool(max(errs) < 0.05)

    fast_ok = bool(tau_a is not None and tau_a < 8.0)
    slow_ok = bool(tau_n is not None and tau_n > 60.0)
    mg_ok = bool(mg_ratio < 0.2)   # Mg²⁺ 阻断使 EPSP 缩小 >5 倍
    pass_ = fast_ok and slow_ok and mg_ok and vdep_ok

    if save_plot:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
        # 1) AMPA vs NMDA 形状（相对各自释放前基线）
        def base_of(v, t_rel):
            return float(np.median(v[t_rel < -0.5]))

        m_a = t_rel_a > -0.5
        m_n = t_rel_n > -0.5
        b_a = base_of(r_a.v_mv["post_soma"], t_rel_a)
        b_n = base_of(r_n.v_mv["post_soma"], t_rel_n)
        axes[0].plot(t_rel_a[m_a], r_a.v_mv["post_soma"][m_a] - b_a,
                     lw=1.2, label=f"AMPA (τ={tau_a:.1f}ms)")
        axes[0].plot(t_rel_n[m_n], r_n.v_mv["post_soma"][m_n] - b_n,
                     lw=1.2, label=f"NMDA mg=0 (τ={tau_n:.1f}ms)")
        axes[0].set_title(f"Fast vs slow: AMPA τ={tau_a:.1f}ms, NMDA τ={tau_n:.1f}ms")
        axes[0].legend(fontsize=8); axes[0].set_xlabel("t - release (ms)")
        axes[0].set_ylabel("ΔV (mV)"); axes[0].set_xlim(0, 220); axes[0].grid(alpha=0.3)
        # 2) Mg²⁺ 阻断
        axes[1].bar(["NMDA mg=0", "NMDA mg=1.2"], [amps_n[0], amps_nm[0]],
                    width=0.5, color=["#1f77b4", "#d62728"])
        axes[1].set_title(f"Mg²⁺ block: ratio {mg_ratio:.3f} vs B(V_rest)={b_rest:.3f}")
        axes[1].set_ylabel("EPSP (mV)"); axes[1].grid(alpha=0.3)
        # 3) 电压依赖 B(V)
        vv = np.linspace(-80, 0, 200)
        axes[2].plot(vv, _b_theory(vv, 1.2), "k-", lw=1.4, label="B(V) theory (Jahr–Stevens)")
        vmeas = [r["v_actual_mv"] for r in rows]
        gmeas = [r["g_peak_ns"] for r in rows]
        axes[2].plot(vmeas, gmeas, "ro", ms=5, label="NEURON NMDASyn measured g/gmax")
        axes[2].set_title(f"Voltage-dependent Mg²⁺ unblock (max err {max(errs)*100:.2f}%)")
        axes[2].set_xlabel("post V (mV)"); axes[2].set_ylabel("B(V)")
        axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)
        fig.suptitle("P5: Receptor subtypes — AMPA fast vs NMDA slow + Mg²⁺ block", y=1.0)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(REPORT_PNG, dpi=150)
        plt.close(fig)

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(REPORT_CSV, "w") as f:
        f.write("metric,value\n")
        f.write(f"ampa_tau_ms,{tau_a if tau_a else ''}\n")
        f.write(f"nmda_tau_ms,{tau_n if tau_n else ''}\n")
        f.write(f"nmda_epsp_mg0_mv,{amps_n[0]:.4f}\n")
        f.write(f"nmda_epsp_mg12_mv,{amps_nm[0]:.4f}\n")
        f.write(f"mg_ratio,{mg_ratio:.4f}\n")
        f.write(f"b_theory_rest,{b_rest:.4f}\n")
        f.write(f"vdep_max_err,{max(errs):.5f}\n")
        f.write(f"pass,{pass_}\n")

    return dict(
        pass_=pass_, fast_ok=fast_ok, slow_ok=slow_ok, mg_ok=mg_ok, vdep_ok=vdep_ok,
        ampa_tau_ms=tau_a, nmda_tau_ms=tau_n,
        nmda_epsp_mg0_mv=float(amps_n[0]), nmda_epsp_mg12_mv=float(amps_nm[0]),
        mg_ratio=mg_ratio, b_theory_rest=b_rest,
        vdep_max_err=float(max(errs)), vdep_points=rows,
        report_png=REPORT_PNG, report_csv=REPORT_CSV,
    )


if __name__ == "__main__":
    import json
    res = run_p5()
    print(json.dumps({k: v for k, v in res.items() if k != "vdep_points"},
                     indent=2, ensure_ascii=False, default=str))
    print("P5 PASS" if res["pass_"] else "P5 FAIL")
