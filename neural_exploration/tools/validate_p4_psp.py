"""P4 验证：树突输入 → 胞体 PSP（衰减与时延）。

方法（清单 §5.3）：同一亚阈值短脉冲（1ms）分别注入
  (a) 树突远端（dend2 末隔室）→ 胞体记录 PSP（远端输入）
  (b) 胞体 → 胞体记录（近端对照）
判定：
  - 远端 PSP 幅度 < 近端 PSP（衰减正确）；
  - 远端 PSP 峰时延 > 近端（传导时延正确）；
  - PSP 衰减时间常数 τ 与模型膜时间常数一致（HH 1952 静息电导
    g_rest≈0.68 mS/cm² → τ≈1.5–4 ms；皮层锥体 5–20ms 需更低 gL，
    判据放宽为 1–20ms 并记录实测值——见 m1_report.md §5.3）。
输出：reports/neuro/m1_psp_propagation.png + data/m1_psp.csv
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
REPORTS_DIR = os.path.join(ROOT, "neural_exploration", "reports", "neuro")
REPORT_PNG = os.path.join(REPORTS_DIR, "m1_psp_propagation.png")
REPORT_CSV = os.path.join(DATA_DIR, "m1_psp.csv")

PULSE_UA_CM2 = 4.0      # µA/cm²（亚阈值；ΔV≈4mV，远低于 HH 阈值）
PULSE_MS = 1.0          # ms（短脉冲，模拟单突触输入）
T_START = 100.0         # ms（待静息瞬态衰减后注入）
T_TOTAL = 180.0         # ms
TAU_RANGE = (1.0, 20.0) # ms，膜时间常数合理范围（含皮层典型 5–20ms）


def _psp_metrics(t, v, t_pulse_start, t_pulse_end):
    """返回 (psp_amp_mv, t_peak_ms, tau_ms)。PSP 相对脉冲前本地基线。"""
    t = np.asarray(t)
    v = np.asarray(v)
    # 本地基线：脉冲前 3ms（消除静息漂移）
    base_win = (t >= t_pulse_start - 3.0) & (t < t_pulse_start - 0.5)
    v_baseline = float(np.median(v[base_win]))
    win = (t >= t_pulse_start - 1.0) & (t <= t_pulse_end + 60.0)
    tw, vw = t[win], v[win]
    peak_idx = int(np.argmax(vw))
    psp_amp = vw[peak_idx] - v_baseline
    t_peak = tw[peak_idx]
    # τ：峰后指数衰减拟合（相对本地基线）
    tail = (tw >= t_peak + 0.5) & (tw <= t_peak + 40.0)
    y = vw[tail] - v_baseline
    tau = None
    if len(y) > 8 and y[0] > 0.05 * max(y):
        y_log = np.log(np.clip(y, 1e-6, None))
        k, _ = np.polyfit(tw[tail] - t_peak, y_log, 1)
        if k < 0:
            tau = -1.0 / k
    return float(psp_amp), float(t_peak), (float(tau) if tau is not None else None), v_baseline


def run_p4(save_plot: bool = True) -> dict:
    from neural_exploration.src.neuron_model import MultiCompartmentNeuron

    n = MultiCompartmentNeuron(t_total_ms=T_TOTAL)

    # 远端树突注入（同电荷）
    r_dist = n.run_stimulus(
        amplitude_uA_cm2=PULSE_UA_CM2, stim_start_ms=T_START,
        stim_end_ms=T_START + PULSE_MS, t_total_ms=T_TOTAL,
        record=["soma", "dend2#1"], inject_at="dend2", inject_compartment=1,
    )
    # 近端（胞体）注入
    r_prox = n.run_stimulus(
        amplitude_uA_cm2=PULSE_UA_CM2, stim_start_ms=T_START,
        stim_end_ms=T_START + PULSE_MS, t_total_ms=T_TOTAL,
        record=["soma"], inject_at="soma",
    )

    t = r_dist.t_ms
    amp_d, t_peak_d, tau_d, base_d = _psp_metrics(
        t, r_dist.v_mv["soma"], T_START, T_START + PULSE_MS)
    amp_p, t_peak_p, _, base_p = _psp_metrics(
        t, r_prox.v_mv["soma"], T_START, T_START + PULSE_MS)
    # 远端注射位点自身幅度（验证注入确实发生了且亚阈值）
    amp_local, _, _, _ = _psp_metrics(
        t, r_dist.v_mv["dend2#1"], T_START, T_START + PULSE_MS)

    attenuated = bool(amp_d > 0.05 and amp_d < amp_p)     # 衰减（且远端确有 PSP）
    delayed = bool(t_peak_d > t_peak_p + 0.05)
    tau_ok = tau_d is not None and TAU_RANGE[0] <= tau_d <= TAU_RANGE[1]
    pass_ = attenuated and delayed and tau_ok

    if save_plot:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(t, r_prox.v_mv["soma"] - base_p, lw=1.2, label="soma injection (proximal)")
        ax.plot(t, r_dist.v_mv["soma"] - base_d, lw=1.2, label="distal dendrite injection")
        ax.plot(t, r_dist.v_mv["dend2#1"] - base_d, lw=0.8, ls="--", label="at dendrite end (local)")
        ax.axvline(T_START, color="gray", ls=":", lw=0.8)
        ax.annotate(f"proximal PSP {amp_p:.2f} mV @{t_peak_p:.1f}ms",
                    xy=(t_peak_p, amp_p), xytext=(t_peak_p + 3, amp_p + 0.3), fontsize=8)
        ax.annotate(f"distal PSP {amp_d:.2f} mV @{t_peak_d:.1f}ms, τ={tau_d:.1f}ms" if tau_d else
                    f"distal PSP {amp_d:.2f} mV @{t_peak_d:.1f}ms",
                    xy=(t_peak_d, amp_d), xytext=(t_peak_d + 3, amp_d + 0.3), fontsize=8)
        ax.set_xlabel("t (ms)")
        ax.set_ylabel("ΔV (mV)")
        ax.set_title(f"P4: PSP propagation (attenuated={attenuated}, delayed={delayed}, τ={tau_d}ms)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(REPORT_PNG, dpi=150)
        plt.close(fig)

    with open(REPORT_CSV, "w") as f:
        f.write("case,psp_amp_mv,t_peak_ms,tau_ms\n")
        f.write(f"proximal_soma,{amp_p:.4f},{t_peak_p:.4f},\n")
        f.write(f"distal_dendrite,{amp_d:.4f},{t_peak_d:.4f},{tau_d if tau_d is not None else ''}\n")
        f.write(f"local_dendrite,{amp_local:.4f},,{tau_d if tau_d is not None else ''}\n")

    return dict(
        pass_=pass_,
        proximal_psp_mv=amp_p, proximal_t_peak_ms=t_peak_p,
        distal_psp_mv=amp_d, distal_t_peak_ms=t_peak_d,
        local_psp_mv=amp_local,
        attenuation_ratio=float(amp_d / amp_p) if amp_p > 0 else None,
        delay_ms=float(t_peak_d - t_peak_p),
        tau_ms=tau_d,
        tau_range=TAU_RANGE,
        report_png=REPORT_PNG, report_csv=REPORT_CSV,
    )


if __name__ == "__main__":
    import json
    res = run_p4()
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print("P4 PASS" if res["pass_"] else "P4 FAIL")
