"""P2 验证：释放失败率符合量子释放模型（二项统计）。

清单 §0 P2 / §5.2：固定 p_release=0.3、n_vesicles=3（量子释放 k ~ Binomial(3, 0.3)），
重复 N 次单刺激 → 统计"无 PSP"比例，与模型失败率 (1-p)^n = 0.343 对比：
  - 失败率落入 Wilson 95% 置信区间；
  - 量子数分布（0/1/2/3）与二项 PMF 的卡方/比率对比（宽松判据：总量子数均值在 ±25%）；
  - 与 NEURON 参考解（同协议 100 试次）的失败率一致（±0.15）。
输出：reports/neuro/m2_p2_failure.png + data/m2_p2_failure.csv
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.tools.build_synapse_ref import PULSE_START, REF_NPZ  # noqa: E402
from neural_exploration.tools.synapse_metrics import binomial_ci  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
REPORTS_DIR = os.path.join(ROOT, "neural_exploration", "reports", "neuro")
REPORT_PNG = os.path.join(REPORTS_DIR, "m2_p2_failure.png")
REPORT_CSV = os.path.join(DATA_DIR, "m2_p2_failure.csv")

P_RELEASE = 0.3
N_VESICLES = 3
N_TRIALS = 100
QUANTUM_NS = 0.3          # 单量子电导（与 ampa CSV 一致）
T_TRIAL = 80.0            # ms


def _quanta_from_g(g_trace, q_density):
    """由 g_ampa 轨迹恢复该试次释放的量子数（k = 峰值/单量子密度）。"""
    peak = float(np.max(g_trace))
    return int(round(peak / q_density))


def run_p2(save_plot: bool = True, n_trials: int = N_TRIALS) -> dict:
    from neural_exploration.src.neuron_pair import NeuronPair

    ref = np.load(REF_NPZ, allow_pickle=True)
    ref_fail = ref["failure"].item()

    pair = NeuronPair(t_total_ms=T_TRIAL, seed=0)
    pair.add_chemical("ampa", g_max_ns=QUANTUM_NS, p_release=P_RELEASE,
                      n_vesicles=N_VESICLES)
    area = np.pi * (20e-6) ** 2
    q_density = QUANTUM_NS * 1e-9 / area

    trials = pair.run_trials(
        pre_pulses=[(PULSE_START, 1.0, 20.0, "soma")], n_trials=n_trials,
        seed_base=2026, record=["post_soma"], record_g=["g_ampa"],
    )
    quanta = np.array([_quanta_from_g(r.g["g_ampa"], q_density) for r in trials])
    failures = int(np.sum(quanta == 0))
    p_hat = failures / n_trials
    lo, hi = binomial_ci(n_trials, p_hat)
    expected = (1 - P_RELEASE) ** N_VESICLES

    # 二项拟合：总量子数均值 vs 期望 n·p
    mean_k = float(np.mean(quanta))
    expected_mean = N_VESICLES * P_RELEASE
    mean_ok = abs(mean_k - expected_mean) <= 0.25 * expected_mean

    # 与 NEURON 参考对比（失败率差 < 0.15）
    ref_rate = ref_fail["failure_rate"]
    ref_ok = abs(p_hat - ref_rate) <= 0.15

    pass_ = bool((lo <= expected <= hi) and mean_ok and ref_ok)

    if save_plot:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        # 1) 试次失败序列
        axes[0].plot(np.arange(1, n_trials + 1), quanta, "o", ms=4)
        axes[0].axhline(expected_mean, color="r", ls="--", lw=1,
                        label=f"expected n·p={expected_mean}")
        axes[0].set_xlabel("trial"); axes[0].set_ylabel("quanta k")
        axes[0].set_title("Release quanta per trial (Binomial(3, 0.3))")
        axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
        # 2) 失败率对比
        bins = np.arange(0, N_VESICLES + 2) - 0.5
        hist, _ = np.histogram(quanta, bins=bins, density=True)
        pmf = [np.math.comb(N_VESICLES, k) * P_RELEASE ** k *
               (1 - P_RELEASE) ** (N_VESICLES - k) for k in range(N_VESICLES + 1)]
        axes[1].bar(range(N_VESICLES + 1), hist, width=0.7, alpha=0.6, label="measured")
        axes[1].plot(range(N_VESICLES + 1), pmf, "ro--", label="Binomial PMF")
        axes[1].set_xlabel("quanta k"); axes[1].set_ylabel("probability")
        axes[1].set_title(f"Failure rate {p_hat:.2f} vs {expected:.2f} (CI [{lo:.2f},{hi:.2f}])")
        axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
        # 3) 与 NEURON 参考对比
        labels = ["Brian2", "NEURON ref"]
        rates = [p_hat, ref_rate]
        axes[2].bar(labels, rates, width=0.5, color=["#1f77b4", "#ff7f0e"])
        axes[2].axhline(expected, color="r", ls="--", lw=1, label=f"model (1-p)^n={expected:.2f}")
        axes[2].set_ylabel("failure rate"); axes[2].set_ylim(0, 0.6)
        axes[2].set_title("Failure rate comparison")
        axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)
        fig.suptitle(f"P2: Release failure — Binomial quantum model (N={n_trials})", y=1.0)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(REPORT_PNG, dpi=150)
        plt.close(fig)

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(REPORT_CSV, "w") as f:
        f.write("metric,value\n")
        f.write(f"n_trials,{n_trials}\n")
        f.write(f"p_release,{P_RELEASE}\n")
        f.write(f"n_vesicles,{N_VESICLES}\n")
        f.write(f"failures,{failures}\n")
        f.write(f"failure_rate,{p_hat:.4f}\n")
        f.write(f"expected_failure,{expected:.4f}\n")
        f.write(f"ci_lo,{lo:.4f}\n")
        f.write(f"ci_hi,{hi:.4f}\n")
        f.write(f"mean_quanta,{mean_k:.4f}\n")
        f.write(f"expected_mean_quanta,{expected_mean:.4f}\n")
        f.write(f"ref_failure_rate,{ref_rate:.4f}\n")
        f.write(f"pass,{pass_}\n")

    return dict(
        pass_=pass_, n_trials=n_trials, failures=failures, p_hat=p_hat,
        expected_failure=expected, ci=(lo, hi), quanta=quanta.tolist(),
        mean_quanta=mean_k, expected_mean_quanta=expected_mean,
        mean_ok=bool(mean_ok), ref_failure_rate=ref_rate, ref_ok=bool(ref_ok),
        report_png=REPORT_PNG, report_csv=REPORT_CSV,
    )


if __name__ == "__main__":
    import json
    res = run_p2()
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print("P2 PASS" if res["pass_"] else "P2 FAIL")
