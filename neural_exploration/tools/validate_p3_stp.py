"""P3 验证：短期可塑性（Tsodyks–Markram）——50Hz×10 脉冲下易化/抑制曲线。

清单 §0 P3 / §5.3：复现 facilitation（幅度递增）或 depression（幅度递减）至少一种，
且与文献趋势一致（Tsodyks & Markram 1998）。本验证复现两种：
  - facilitation：小 U（0.03）、快 τ_fac（120ms）、快 τ_rec（40ms）→ EPSP 递增；
  - depression：大 U（0.6）、慢 τ_rec（400ms）→ EPSP 递减。
判定：10 个 EPSP 的单调节点
  - facilitation: 末次 ≥ 首次 × 1.5 且单调不减（容差 1 次）；
  - depression: 末次 ≤ 首次 × 0.6 且单调不增（容差 1 次）。
输出：reports/neuro/m2_stp.png + data/m2_stp.csv
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.tools.synapse_metrics import psp_amplitudes  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
REPORTS_DIR = os.path.join(ROOT, "neural_exploration", "reports", "neuro")
REPORT_PNG = os.path.join(REPORTS_DIR, "m2_stp.png")
REPORT_CSV = os.path.join(DATA_DIR, "m2_stp.csv")

FREQ_HZ = 50.0
N_PULSES = 10
T_TOTAL = 320.0
PULSE_START = 50.0


def _monotonic(amps, direction):
    """direction: +1 要求非降（容差 1 次违反），-1 非增。"""
    d = np.diff(amps) * direction
    return int(np.sum(d < -1e-6)) <= 1


def run_p3(save_plot: bool = True) -> dict:
    from neural_exploration.src.neuron_pair import NeuronPair, pulse_train

    # ---- facilitation ----
    pair = NeuronPair(t_total_ms=T_TOTAL, seed=0)
    pair.add_chemical("ampa", g_max_ns=0.5, p_release=1.0, n_vesicles=1,
                      stp=(0.03, 120.0, 40.0))
    r = pair.run(pre_pulses=pulse_train(PULSE_START, FREQ_HZ, N_PULSES, 1.0, 20.0),
                 record=["pre_node3", "post_soma"])
    fac = psp_amplitudes(r.t_ms, r.v_mv["post_soma"], r.spike_times_ms["pre_node3"])

    # ---- depression ----
    pair = NeuronPair(t_total_ms=T_TOTAL, seed=0)
    pair.add_chemical("ampa", g_max_ns=0.5, p_release=1.0, n_vesicles=1,
                      stp=(0.6, 10.0, 400.0))
    r = pair.run(pre_pulses=pulse_train(PULSE_START, FREQ_HZ, N_PULSES, 1.0, 20.0),
                 record=["pre_node3", "post_soma"])
    dep = psp_amplitudes(r.t_ms, r.v_mv["post_soma"], r.spike_times_ms["pre_node3"])

    fac_ok = bool(len(fac) == N_PULSES and fac[-1] >= fac[0] * 1.5 and _monotonic(fac, +1))
    dep_ok = bool(len(dep) == N_PULSES and dep[-1] <= dep[0] * 0.6 and _monotonic(dep, -1))
    pass_ = fac_ok and dep_ok

    if save_plot:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, amps, title, ok in (
                (axes[0], fac, "Facilitation (U=0.03, τfac=120ms, τrec=40ms)", fac_ok),
                (axes[1], dep, "Depression (U=0.6, τfac=10ms, τrec=400ms)", dep_ok)):
            x = np.arange(1, len(amps) + 1)
            ax.plot(x, amps, "o-", lw=1.4)
            ax.set_xlabel("pulse #"); ax.set_ylabel("EPSP (mV)")
            ax.set_title(f"{title}\npass={ok}  ({amps[0]:.3f} → {amps[-1]:.3f} mV)")
            ax.grid(alpha=0.3)
        fig.suptitle("P3: Short-term plasticity at 50 Hz × 10 pulses (Tsodyks–Markram)", y=1.0)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(REPORT_PNG, dpi=150)
        plt.close(fig)

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(REPORT_CSV, "w") as f:
        f.write("case,pulse,epsps_mv\n")
        for k, a in enumerate(fac, 1):
            f.write(f"facilitation,{k},{a:.5f}\n")
        for k, a in enumerate(dep, 1):
            f.write(f"depression,{k},{a:.5f}\n")

    return dict(
        pass_=pass_, facilitation_ok=fac_ok, depression_ok=dep_ok,
        facilitation=fac.tolist(), depression=dep.tolist(),
        freq_hz=FREQ_HZ, n_pulses=N_PULSES,
        report_png=REPORT_PNG, report_csv=REPORT_CSV,
    )


if __name__ == "__main__":
    import json
    res = run_p3()
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("facilitation", "depression")},
                     indent=2, ensure_ascii=False))
    print("  facilitation:", np.round(res["facilitation"], 4))
    print("  depression:", np.round(res["depression"], 4))
    print("P3 PASS" if res["pass_"] else "P3 FAIL")
