"""P2 验证：动作电位波形误差 <5%（Brian2 vs NEURON cvode 参考解）。

判定：`waveform_rmse`（复用 tools/metrics.py）归一化到参考轨迹峰-峰幅度后 < 5%。
输出：reports/neuro/m1_p2_waveform.png（叠加 + 残差）+ 数值表 data/m1_p2_waveform.csv
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.tools.metrics import waveform_rmse  # noqa: E402
from neural_exploration.tools.build_neuron_ref import (  # noqa: E402
    DT_OUT, REF_NPZ, STIM_END, STIM_START, STIM_UA_CM2, T_TOTAL, run_reference,
)
from neural_exploration.tools.load_morphology import load_morphology  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
REPORTS_DIR = os.path.join(ROOT, "neural_exploration", "reports", "neuro")
REPORT_PNG = os.path.join(REPORTS_DIR, "m1_p2_waveform.png")
REPORT_CSV = os.path.join(DATA_DIR, "m1_p2_waveform.csv")

# 判定阈值（清单 §0 P2）：归一化 RMSE < 5%
NORM_RMSE_THRESHOLD = 0.05


def ensure_reference():
    """参考解存在则复用，否则现场生成。"""
    if not os.path.exists(REF_NPZ):
        run_reference(load_morphology())
    return np.load(REF_NPZ, allow_pickle=True)


def run_p2(save_plot: bool = True) -> dict:
    from neural_exploration.src.neuron_model import MultiCompartmentNeuron

    ref = ensure_reference()
    t_ref = ref["t_ms"]

    n = MultiCompartmentNeuron(t_total_ms=T_TOTAL, dt_ms=DT_OUT)
    # 注意：Brian2 输出网格与参考一致（dt=0.01ms）
    r = n.run_stimulus(
        amplitude_uA_cm2=STIM_UA_CM2,
        stim_start_ms=STIM_START,
        stim_end_ms=STIM_END,
        record=["soma", "dend2#1", "node3"],
    )

    pairs = {
        "soma": ("v_soma_mv", r.v_mv["soma"]),
        "dend_end": ("v_dend_end_mv", r.v_mv["dend2#1"]),
        "axon_end_node3": ("v_node3_mv", r.v_mv["node3"]),
    }
    results = {}
    for label, (ref_key, v_sim) in pairs.items():
        v_ref = ref[ref_key]
        n_common = min(len(v_sim), len(v_ref))
        v_sim_c, v_ref_c = v_sim[:n_common], v_ref[:n_common]
        rmse = waveform_rmse(v_sim_c, v_ref_c, DT_OUT)
        span = float(v_ref_c.max() - v_ref_c.min())
        norm = rmse / span if span > 0 else float("nan")
        results[label] = dict(
            rmse_mv=rmse, norm_rmse=norm, v_ref_span_mv=span,
            pass_=bool(norm < NORM_RMSE_THRESHOLD),
        )

    all_pass = all(v["pass_"] for v in results.values())

    if save_plot:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(len(pairs), 2, figsize=(12, 2.6 * len(pairs)),
                                 gridspec_kw={"width_ratios": [2.2, 1]})
        for row, (label, (ref_key, v_sim)) in enumerate(pairs.items()):
            v_ref = ref[ref_key]
            n_common = min(len(v_sim), len(v_ref))
            axes[row][0].plot(t_ref[:n_common], v_sim[:n_common], lw=1.2, label="Brian2 (rk4)")
            axes[row][0].plot(t_ref[:n_common], v_ref[:n_common], lw=1.0, ls="--", color="k",
                              label="NEURON cvode")
            axes[row][0].set_title(f"{label}: norm_RMSE={results[label]['norm_rmse']*100:.2f}%")
            axes[row][0].legend(fontsize=8)
            axes[row][0].set_ylabel("V (mV)")
            axes[row][0].grid(alpha=0.3)
            axes[row][1].plot(t_ref[:n_common], v_sim[:n_common] - v_ref[:n_common],
                              lw=0.8, color="#d62728")
            axes[row][1].set_title("residual (mV)")
            axes[row][1].grid(alpha=0.3)
        axes[-1][0].set_xlabel("t (ms)")
        axes[-1][1].set_xlabel("t (ms)")
        fig.suptitle("P2: Brian2 vs NEURON reference (multi-compartment HH)", y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(REPORT_PNG, dpi=150)
        plt.close(fig)

    # 数值表
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(REPORT_CSV, "w") as f:
        f.write("location,rmse_mv,norm_rmse,v_ref_span_mv,pass\n")
        for label, v in results.items():
            f.write(f"{label},{v['rmse_mv']:.6f},{v['norm_rmse']:.6f},"
                    f"{v['v_ref_span_mv']:.3f},{v['pass_']}\n")

    return dict(
        pass_=all_pass,
        threshold=NORM_RMSE_THRESHOLD,
        per_location=results,
        report_png=REPORT_PNG,
        report_csv=REPORT_CSV,
    )


if __name__ == "__main__":
    import json
    res = run_p2()
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print("P2 PASS" if res["pass_"] else "P2 FAIL")
