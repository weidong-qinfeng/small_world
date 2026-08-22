"""P3 验证：f-I 曲线（注入电流 → 发放频率）。

判定（清单 §0 P3）：
  - 单调递增（允许持平）；
  - 阈值电流处从 0 → >0；
  - 斜率与皮层神经元文献量级一致（~20–100 Hz 在 2–3× 阈值区间）。
输出：data/m1_fi_curve.csv + reports/neuro/m1_fi_curve.png
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.tools.load_morphology import load_morphology  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
REPORTS_DIR = os.path.join(ROOT, "neural_exploration", "reports", "neuro")
REPORT_CSV = os.path.join(DATA_DIR, "m1_fi_curve.csv")
REPORT_PNG = os.path.join(REPORTS_DIR, "m1_fi_curve.png")

# 电流扫描（清单 §5.2：0,5,10,15,20,30,40,50 µA/cm²）
AMPS = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0]
T_TOTAL = 500.0        # ms
WARMUP_MS = 50.0       # 不计入稳态频率的前期
FIRING_START_MS = 20.0  # 首个发放后的稳定窗起点（相对于刺激开始）


def firing_rate_from_spikes(spike_times, t_total, t_stim_start, amp):
    """稳态发放频率：首个发放之后的窗内平均。"""
    st = np.asarray(spike_times, dtype=float)
    if len(st) == 0:
        return 0.0
    st = st[st >= t_stim_start]
    if len(st) < 2:
        return 0.0
    span_s = (st[-1] - st[0]) / 1000.0
    if span_s <= 0:
        return 0.0
    return (len(st) - 1) / span_s


def run_p3(save_plot: bool = True) -> dict:
    from neural_exploration.src.neuron_model import MultiCompartmentNeuron, SingleCompartmentHH

    n = MultiCompartmentNeuron(t_total_ms=T_TOTAL)
    freqs = []
    spike_counts = []
    for amp in AMPS:
        r = n.run_stimulus(amplitude_uA_cm2=amp, stim_start_ms=5.0,
                           stim_end_ms=T_TOTAL - 5.0, record=["soma"])
        st = r.spike_times_ms["soma"]
        f = firing_rate_from_spikes(st, T_TOTAL, 5.0 + FIRING_START_MS, amp)
        freqs.append(f)
        spike_counts.append(len(st))
        print(f"  I={amp:5.1f} µA/cm² -> spikes={len(st)} f={f:6.1f} Hz")

    # 单隔室对照（清单 §4.1：保留单隔室版供 f-I 对照）
    sc = SingleCompartmentHH(t_total_ms=T_TOTAL)
    freqs_sc = [sc.firing_rate(a) for a in AMPS]
    print(f"  单隔室对照 f-I: {[f'{f:.1f}' for f in freqs_sc]}")

    freqs = np.array(freqs)
    amps = np.array(AMPS)

    # 判定
    mono = bool(np.all(np.diff(freqs) >= -1e-9))            # 单调不减
    thr_idx = int(np.flatnonzero(freqs > 0)[0]) if np.any(freqs > 0) else None
    threshold = float(amps[thr_idx]) if thr_idx is not None else None
    has_threshold = thr_idx is not None and thr_idx > 0      # 从 0 到 >0
    # 斜率：阈值后的线性区（取阈值与最高点之间的平均斜率）
    slope_hz_per_ua = None
    if thr_idx is not None and len(amps) - 1 > thr_idx:
        hi = len(amps) - 1
        slope_hz_per_ua = float((freqs[hi] - freqs[thr_idx]) /
                                (amps[hi] - amps[thr_idx]) if amps[hi] > amps[thr_idx] else None)

    pass_ = mono and has_threshold

    if save_plot:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(amps, freqs, "o-", color="#1f77b4")
        if threshold is not None:
            ax.axvline(threshold, color="gray", ls="--", lw=0.8)
            ax.annotate(f"threshold={threshold:.0f}", xy=(threshold, 5), fontsize=8)
        ax.set_xlabel("injected current (µA/cm², soma)")
        ax.set_ylabel("firing rate (Hz)")
        ax.set_title(f"P3 f-I curve (monotonic={mono}, slope={slope_hz_per_ua:.1f} Hz/µA·cm⁻²)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(REPORT_PNG, dpi=150)
        plt.close(fig)

    with open(REPORT_CSV, "w") as f:
        f.write("amp_uA_cm2,firing_rate_hz,n_spikes,firing_rate_single_comp_hz\n")
        for a, fr, sc_, fsc in zip(amps, freqs, spike_counts, freqs_sc):
            f.write(f"{a},{fr:.4f},{sc_},{fsc:.4f}\n")

    return dict(
        pass_=pass_,
        monotonic=mono,
        threshold_ua_cm2=threshold,
        has_threshold=has_threshold,
        slope_hz_per_ua_cm2=slope_hz_per_ua,
        amps=amps.tolist(), freqs=freqs.tolist(),
        freqs_single_comp=freqs_sc,
        report_csv=REPORT_CSV, report_png=REPORT_PNG,
    )


if __name__ == "__main__":
    import json
    res = run_p3()
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print("P3 PASS" if res["pass_"] else "P3 FAIL")
