"""P4 验证：强度-响应曲线单调（主 agent 修订判据，清单 §5.4）。

- 协议：6 档强度 {0, 0.5, 1, 2, 4, 8}×I0 各 run(intensity=s)（确定性 p=1/n=1）；
- 判据：
  * 0 档：触觉链全静默（PLM/AVM/DA 无发放）、C_back = 0；
  * 潜伏期 vs 强度单调非增：Spearman ρ ≤ −0.9；
  * 幅度补充验证（修订判据）：8×I0 档延长刺激时程至 20ms
    → DA ≥ 2 发放 且 C_back 峰值 > 单发放值 w_back（0.6）
    （强刺激 → 更强响应；单发放档幅度恒等是模型限制，见 P4 修订记录）。
输出：reports/neuro/m3_p4_intensity.png + data/m3_p4_intensity.csv
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.src.reflex_arc import ReflexArc  # noqa: E402

DATA_DIR = os.path.join(ROOT, "neural_exploration", "data")
REPORTS_DIR = os.path.join(ROOT, "neural_exploration", "reports", "neuro")
CSV_PATH = os.path.join(DATA_DIR, "m3_reflex_params.csv")
REF_NPZ = os.path.join(DATA_DIR, "m3_reflex_ref.npz")
REPORT_PNG = os.path.join(REPORTS_DIR, "m3_p4_intensity.png")
REPORT_CSV = os.path.join(DATA_DIR, "m3_p4_intensity.csv")

RHO_THRESHOLD = -0.9
SUPPLEMENT_DUR_MS = 20.0     # 8×I0 档延长刺激时程
SUPPLEMENT_MIN_SPIKES = 2    # DA ≥ 2 发放
MULTI_AMPLITUDE_MARGIN = 1e-6


def run_p4(save_plot: bool = True) -> dict:
    ref = np.load(REF_NPZ, allow_pickle=True)
    ref_lat = ref["latency_nerve"]

    arc = ReflexArc(csv_path=CSV_PATH)   # 确定性
    touch_start = arc.params.touch.start_ms
    levels = list(arc.params.intensity_levels)
    w_back = arc.params.muscle.w_back

    rows = []
    for s in levels:
        rr = arc.run(intensity=s)
        st = rr.spikes("DA", "node3")
        lat = float(st[0] - touch_start) if len(st) else None
        silent = (len(rr.spikes("PLM", "node3")) == 0
                  and len(rr.spikes("AVM", "node3")) == 0
                  and len(rr.spikes("DA", "node3")) == 0)
        rows.append(dict(
            intensity=s, latency_ms=lat, c_back_peak=float(rr.c_back_peak),
            da_spikes=len(st), silent=bool(silent),
        ))

    # ---- 0 档：全链静默 + C_back=0 ----
    r0 = rows[0]
    zero_ok = bool(r0["silent"] and r0["c_back_peak"] == 0.0)

    # ---- 潜伏期单调（Spearman ρ ≤ −0.9，非零档）----
    lats = np.asarray([r["latency_ms"] for r in rows if r["latency_ms"] is not None])
    ints = np.asarray([r["intensity"] for r in rows if r["latency_ms"] is not None])
    from scipy.stats import spearmanr
    rho, _ = spearmanr(ints, lats)
    rho_ok = bool(rho <= RHO_THRESHOLD)

    # ---- 幅度补充：8×I0 / dur=20ms 多发放 ----
    arc_sup = ReflexArc(csv_path=CSV_PATH)
    arc_sup.set_touch(dur_ms=SUPPLEMENT_DUR_MS)
    r8 = arc_sup.run(intensity=8.0)
    da8 = r8.spikes("DA", "node3")
    sup_spikes_ok = bool(len(da8) >= SUPPLEMENT_MIN_SPIKES)
    sup_amp_ok = bool(r8.c_back_peak > w_back + MULTI_AMPLITUDE_MARGIN)
    sup_ok = bool(sup_spikes_ok and sup_amp_ok)

    pass_ = bool(zero_ok and rho_ok and sup_ok)

    if save_plot:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
        # 1) 强度-潜伏期
        ax = axes[0]
        xs = [r["intensity"] for r in rows if r["latency_ms"] is not None]
        ys = [r["latency_ms"] for r in rows if r["latency_ms"] is not None]
        ax.plot(xs, ys, "o-", color="#1f77b4", lw=1.5, label="Brian2 (rk4)")
        ax.plot(ref["intensities"][1:], ref_lat[1:], "s--", color="k", lw=1.1,
                label="NEURON ref", ms=5)
        ax.set_xlabel("intensity (×I0)"); ax.set_ylabel("neural latency (ms)")
        ax.set_title(f"intensity–latency: Spearman ρ={rho:.3f} (≤−0.9: {rho_ok})")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        # 2) 幅度：每档 C_back 峰值 + 8×I0/20ms 补充
        ax = axes[1]
        xs_all = [r["intensity"] for r in rows]
        ys_all = [r["c_back_peak"] for r in rows]
        ax.plot(xs_all, ys_all, "o-", color="#2ca02c", lw=1.5,
                label="C_back peak (5ms touch)")
        ax.axhline(w_back, color="gray", ls=":", lw=1.0, label=f"w_back={w_back} (single-spike)")
        ax.plot([8.0], [r8.c_back_peak], "D", color="#d62728", ms=9,
                label=f"8×I0/20ms: peak={r8.c_back_peak:.3f}, DA n={len(da8)}")
        ax.set_xlabel("intensity (×I0)"); ax.set_ylabel("C_back peak")
        ax.set_title(f"amplitude supplement (revised): DA≥{SUPPLEMENT_MIN_SPIKES}: "
                     f"{sup_spikes_ok}, peak>w_back: {sup_amp_ok}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.suptitle("P4: intensity–response monotonicity (revised criteria)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(REPORT_PNG, dpi=150)
        plt.close(fig)

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(REPORT_CSV, "w", encoding="utf-8") as f:
        f.write("intensity,latency_ms,c_back_peak,da_spikes,chain_silent\n")
        for r in rows:
            f.write(f"{r['intensity']},{r['latency_ms'] if r['latency_ms'] is not None else ''},"
                    f"{r['c_back_peak']:.5f},{r['da_spikes']},{int(r['silent'])}\n")
        f.write(f"8.0_supplement20ms,{da8[0] - touch_start if len(da8) else ''},"
                f"{r8.c_back_peak:.5f},{len(da8)},0\n")
        f.write(f"summary,rho,{rho:.6f},{int(rho_ok)}\n")
        f.write(f"summary,zero_silent,{int(zero_ok)},,\n")
        f.write(f"summary,supplement_ok,{int(sup_ok)},,\n")
        f.write(f"summary,pass,{int(pass_)},,\n")

    return dict(
        pass_=pass_,
        rows=rows, spearman_rho=float(rho), rho_ok=rho_ok,
        zero_silent=zero_ok, c_back_peak_at_zero=rows[0]["c_back_peak"],
        supplement=dict(dur_ms=SUPPLEMENT_DUR_MS, da_spikes=len(da8),
                        da_first_ms=float(da8[0]) if len(da8) else None,
                        c_back_peak=float(r8.c_back_peak), w_back=float(w_back),
                        spikes_ok=sup_spikes_ok, amp_ok=sup_amp_ok, ok=sup_ok),
        report_png=REPORT_PNG, report_csv=REPORT_CSV,
    )


if __name__ == "__main__":
    import json
    res = run_p4()
    print(json.dumps(res, indent=2, ensure_ascii=False, default=str))
    print("P4 PASS" if res["pass_"] else "P4 FAIL")
