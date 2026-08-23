"""P1 验证：定向反应（方向性）——后退通道激活、前进通道被抑制。

清单 §0 P1 / §5.1：
- 协议：基准强度 I0 触刺激，5 次有效重复（量子释放噪声 p=0.95/n=2，
  失败试次 = DA 无发放，剔除并计数）；
- 判据：有效试次 5/5 的 D_peak = max(C_back − C_fwd) > 0.3；
- 无刺激对照（intensity=0，确定性）：触觉链静默、C_back = 0，
  VB 张力注入维持前进基线 C_fwd ≈ 0.2（0.1–0.4）。
输出：reports/neuro/m3_p1_direction.png + data/m3_p1_direction.csv
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
REPORT_PNG = os.path.join(REPORTS_DIR, "m3_p1_direction.png")
REPORT_CSV = os.path.join(DATA_DIR, "m3_p1_direction.csv")

D_PEAK_THRESHOLD = 0.3     # 判据：D_peak > 0.3 → 后退
N_VALID_REQUIRED = 5       # 判据：5/5 有效试次
N_TRIALS = 8               # 多跑几试次以保证 5 个有效（失败剔除并计数）
SEED_BASE = 1234
P_RELEASE, N_VESICLES = 0.95, 2   # 量子释放噪声协议


def run_p1(save_plot: bool = True) -> dict:
    # ---- 5 次有效重复（量子噪声）----
    arc = ReflexArc(csv_path=CSV_PATH)
    arc.set_quantum_noise(P_RELEASE, N_VESICLES)
    trials = arc.run_trials(intensity=1.0, n_trials=N_TRIALS, seed_base=SEED_BASE)

    rows = []
    n_fail = 0
    for i, tr in enumerate(trials):
        da_n = len(tr.spikes("DA", "node3"))
        failed = da_n < 1
        if failed:
            n_fail += 1
        rows.append(dict(
            trial=i, da_spikes=da_n, d_peak=float(tr.d_peak),
            direction="failed" if failed else ("back" if tr.d_peak > D_PEAK_THRESHOLD else "not_back"),
            failed=int(failed),
        ))
    valid = [r for r in rows if not r["failed"]]
    used = valid[:N_VALID_REQUIRED]
    d_peaks = np.asarray([r["d_peak"] for r in used])
    valid_ok = bool(len(used) >= N_VALID_REQUIRED and np.all(d_peaks > D_PEAK_THRESHOLD))

    # ---- 无刺激对照（确定性）----
    arc_ctrl = ReflexArc(csv_path=CSV_PATH)   # 默认 p=1/n=1（确定性铁律）
    r0 = arc_ctrl.run(intensity=0.0)
    c_back_peak0 = float(r0.c_back_peak)
    c_fwd_steady = float(np.median(r0.c_fwd[r0.t_ms >= 0.5 * r0.t_ms[-1]]))  # 稳态段中位数
    silent = (
        len(r0.spikes("PLM", "node3")) == 0
        and len(r0.spikes("AVM", "node3")) == 0
        and len(r0.spikes("DA", "node3")) == 0
    )
    ctrl_ok = bool(silent and c_back_peak0 == 0.0 and 0.1 < c_fwd_steady < 0.4)
    touch_start = r0.meta["touch_start_ms"]

    pass_ = bool(valid_ok and ctrl_ok)

    if save_plot:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        # 1) 5 次有效试次 D_peak 分布
        ax = axes[0]
        xs = np.arange(len(used)) + 1
        colors = ["#2ca02c" if d > D_PEAK_THRESHOLD else "#d62728" for d in d_peaks]
        ax.bar(xs, d_peaks, width=0.6, color=colors, edgecolor="k", lw=0.6)
        ax.axhline(D_PEAK_THRESHOLD, color="k", ls="--", lw=1.2,
                   label=f"threshold {D_PEAK_THRESHOLD}")
        ax.set_xticks(xs)
        ax.set_xlabel("valid trial #")
        ax.set_ylabel("D_peak = max(C_back − C_fwd)")
        ax.set_ylim(0, max(0.55, d_peaks.max() * 1.15))
        ax.set_title(f"P1: direction per trial (n_valid={len(used)}, "
                     f"failures={n_fail}, all>0.3: {valid_ok})")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        for x, d in zip(xs, d_peaks):
            ax.text(x, d + 0.012, f"{d:.3f}", ha="center", fontsize=8)
        # 2) 无刺激对照：C_back / C_fwd 轨迹
        ax = axes[1]
        ax.plot(r0.t_ms, r0.c_fwd, lw=1.2, color="#8c564b", label="C_fwd (VB tonic baseline)")
        ax.plot(r0.t_ms, r0.c_back, lw=1.2, color="#2ca02c", label="C_back (must be 0)")
        ax.axhline(0.2, color="gray", ls=":", lw=0.8, label="target baseline ≈0.2")
        ax.axhline(0.1, color="gray", ls=":", lw=0.6)
        ax.axhline(0.4, color="gray", ls=":", lw=0.6)
        ax.set_xlabel("t (ms)"); ax.set_ylabel("muscle C")
        ax.set_title(f"no-stimulus control: C_back={c_back_peak0:.3f}, "
                     f"C_fwd steady={c_fwd_steady:.3f} (0.1–0.4: {ctrl_ok})")
        ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(-0.02, 0.5)
        fig.suptitle("P1: directional response (5/5 valid trials D_peak > 0.3)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(REPORT_PNG, dpi=150)
        plt.close(fig)

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(REPORT_CSV, "w", encoding="utf-8") as f:
        f.write("trial,da_spikes,d_peak,direction,failed\n")
        for r in rows:
            f.write(f"{r['trial']},{r['da_spikes']},{r['d_peak']:.5f},"
                    f"{r['direction']},{r['failed']}\n")
        f.write(f"control,c_back_peak,{c_back_peak0:.5f},silent,{int(silent)}\n")
        f.write(f"control,c_fwd_steady,{c_fwd_steady:.5f},,0\n")
        f.write(f"summary,n_valid_used,{len(used)},n_failures,{n_fail}\n")
        f.write(f"summary,valid_ok,{int(valid_ok)},ctrl_ok,{int(ctrl_ok)}\n")
        f.write(f"summary,pass,{int(pass_)},,0\n")

    return dict(
        pass_=pass_,
        n_valid_used=len(used), n_failures=n_fail,
        d_peaks=d_peaks.tolist(), d_peak_threshold=D_PEAK_THRESHOLD,
        valid_ok=valid_ok, ctrl_ok=ctrl_ok,
        c_back_peak_control=c_back_peak0, c_fwd_steady_control=c_fwd_steady,
        control_silent=silent, touch_start_ms=touch_start,
        report_png=REPORT_PNG, report_csv=REPORT_CSV,
    )


if __name__ == "__main__":
    import json
    res = run_p1()
    print(json.dumps(res, indent=2, ensure_ascii=False, default=str))
    print("P1 PASS" if res["pass_"] else "P1 FAIL")
