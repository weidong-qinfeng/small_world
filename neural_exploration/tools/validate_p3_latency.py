"""P3 验证：潜伏期对照行为学（主 agent 修订判据，清单 §5.3）。

- 神经潜伏期 = DA 首发放 − touch_start：
  * vs NEURON 参考链 latency_nerve（I0 档）误差 < 15%；
  * 落在回路生理窗 [5, 20] ms（Chalfie 1985 行为潜伏期 30–50ms 含机械转导
    与肌肉收缩动力学——本模型触电流直接注入感觉神经元 → 差异归因见报告 §4）；
  * 强度-潜伏期单调非增：Spearman ρ ≤ −0.9（6 档）；
  * 20 试次（量子释放噪声 p=0.95/n=2，失败剔除并计数）潜伏期 SD < 5 ms。
- 行为潜伏期（清单 §5.3 原定义：首个 C_back ≥ 0.3·C_back_peak − touch_start）
  如实计算并记录，与神经潜伏期对比 → 差异归因（L7：δ 驱动肌肉对单发放瞬间达峰，
  行为潜伏期 ≡ 神经潜伏期；Chalfie 30–50ms 差异登记为 M3 简化假设，不伪造窗口）。
输出：reports/neuro/m3_p3_latency.png + data/m3_p3_latency.csv
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.src.muscle import behavioral_latency_ms  # noqa: E402
from neural_exploration.src.reflex_arc import ReflexArc  # noqa: E402

DATA_DIR = os.path.join(ROOT, "neural_exploration", "data")
REPORTS_DIR = os.path.join(ROOT, "neural_exploration", "reports", "neuro")
CSV_PATH = os.path.join(DATA_DIR, "m3_reflex_params.csv")
REF_NPZ = os.path.join(DATA_DIR, "m3_reflex_ref.npz")
REPORT_PNG = os.path.join(REPORTS_DIR, "m3_p3_latency.png")
REPORT_CSV = os.path.join(DATA_DIR, "m3_p3_latency.csv")

LAT_REF_INDEX = 2          # intensities[2] = 1.0（I0 档）
LAT_WINDOW = (5.0, 20.0)   # 回路生理窗（修订判据）
RHO_THRESHOLD = -0.9       # Spearman ρ ≤ −0.9
SD_THRESHOLD_MS = 5.0      # 20 试次潜伏期 SD < 5 ms
N_TRIALS = 20
SEED_BASE_TRIALS = 5678
P_RELEASE, N_VESICLES = 0.95, 2


def run_p3(save_plot: bool = True) -> dict:
    ref = np.load(REF_NPZ, allow_pickle=True)
    ref_lat = ref["latency_nerve"]
    intens = ref["intensities"]

    arc = ReflexArc(csv_path=CSV_PATH)   # 确定性默认 p=1/n=1
    touch_start = arc.params.touch.start_ms

    # ---- 6 档强度-潜伏期（确定性）----
    levels = list(arc.params.intensity_levels)
    lat_by_level = {}
    for s in levels:
        rr = arc.run(intensity=s)
        st = rr.spikes("DA", "node3")
        lat_by_level[s] = float(st[0] - touch_start) if len(st) else None

    lats_nonzero = np.asarray([lat_by_level[s] for s in levels if lat_by_level[s] is not None])
    ints_nonzero = np.asarray([s for s in levels if lat_by_level[s] is not None])
    from scipy.stats import spearmanr
    rho, _ = spearmanr(ints_nonzero, lats_nonzero)
    rho_ok = bool(rho <= RHO_THRESHOLD)

    lat_i0 = lat_by_level[1.0]
    ref_lat_i0 = float(ref_lat[LAT_REF_INDEX])
    ref_err = abs(lat_i0 - ref_lat_i0) / ref_lat_i0
    ref_ok = bool(ref_err < 0.15)
    window_ok = bool(LAT_WINDOW[0] <= lat_i0 <= LAT_WINDOW[1])

    # ---- 20 试次量子噪声（失败剔除并计数）----
    arc_noise = ReflexArc(csv_path=CSV_PATH)
    arc_noise.set_quantum_noise(P_RELEASE, N_VESICLES)
    trials = arc_noise.run_trials(intensity=1.0, n_trials=N_TRIALS, seed_base=SEED_BASE_TRIALS)
    trial_rows, lats_trials = [], []
    for i, tr in enumerate(trials):
        st = tr.spikes("DA", "node3")
        lat = float(st[0] - touch_start) if len(st) else None
        trial_rows.append(dict(trial=i, latency_ms=lat, da_spikes=len(st)))
        if lat is not None:
            lats_trials.append(lat)
    n_fail = N_TRIALS - len(lats_trials)
    lat_sd = float(np.std(lats_trials)) if lats_trials else float("nan")
    sd_ok = bool(len(lats_trials) >= 10 and lat_sd < SD_THRESHOLD_MS)

    # ---- 行为潜伏期（清单 §5.3 原定义，如实计算）----
    r_i0 = arc.run(intensity=1.0)
    beh_lat = behavioral_latency_ms(r_i0.c_back, r_i0.t_ms, touch_start)
    beh_ref = float(ref["latency_behavior"][LAT_REF_INDEX])
    beh_equiv = bool(np.isclose(beh_lat, lat_i0, atol=2.0))   # L7：行为潜伏期 ≡ 神经潜伏期

    pass_ = bool(ref_ok and window_ok and rho_ok and sd_ok)

    if save_plot:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
        # 1) 20 试次潜伏期直方图 + 生理窗
        ax = axes[0]
        ax.hist(lats_trials, bins=14, color="#1f77b4", edgecolor="k", alpha=0.85)
        ax.axvspan(LAT_WINDOW[0], LAT_WINDOW[1], color="orange", alpha=0.18,
                   label="circuit window [5,20] ms")
        ax.axvline(lat_i0, color="#d62728", ls="--", lw=1.3,
                   label=f"I0 deterministic {lat_i0:.2f} ms")
        ax.axvline(ref_lat_i0, color="k", ls=":", lw=1.3,
                   label=f"NEURON ref {ref_lat_i0:.2f} ms")
        ax.set_xlabel("neural latency (ms)"); ax.set_ylabel("trials")
        ax.set_title(f"20 noise trials: mean={np.mean(lats_trials):.2f} ms, "
                     f"SD={lat_sd:.2f} ms (<5: {sd_ok}), failures={n_fail}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        # 2) 强度-潜伏期曲线（sim vs ref）
        ax = axes[1]
        xs = [s for s in levels if lat_by_level[s] is not None]
        ys = [lat_by_level[s] for s in xs]
        ax.plot(xs, ys, "o-", color="#1f77b4", lw=1.4, label="Brian2 (rk4)")
        ax.plot(intens[1:], ref_lat[1:], "s--", color="k", lw=1.1,
                label="NEURON ref", ms=5)
        ax.axhspan(*LAT_WINDOW, color="orange", alpha=0.12)
        ax.set_xlabel("intensity (×I0)"); ax.set_ylabel("neural latency (ms)")
        ax.set_title(f"intensity–latency: Spearman ρ={rho:.3f} (≤−0.9: {rho_ok})")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.suptitle("P3: latency vs behavioral reference (revised criteria)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(REPORT_PNG, dpi=150)
        plt.close(fig)

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(REPORT_CSV, "w", encoding="utf-8") as f:
        f.write("kind,key,value,pass\n")
        for s in levels:
            v = lat_by_level[s]
            f.write(f"level,intensity_{s},{v if v is not None else ''},,\n")
        f.write(f"spearman,rho,{rho:.6f},{rho_ok}\n")
        f.write(f"lat_i0_sim_ms,value,{lat_i0:.5f},,\n")
        f.write(f"lat_i0_ref_ms,value,{ref_lat_i0:.5f},,\n")
        f.write(f"lat_i0_ref_err,value,{ref_err:.6f},{ref_ok}\n")
        f.write(f"window_ok,value,{int(window_ok)},,\n")
        f.write(f"beh_lat_i0_sim_ms,value,{beh_lat:.5f},,\n")
        f.write(f"beh_lat_i0_ref_ms,value,{beh_ref:.5f},,\n")
        f.write(f"beh_equiv_neural,value,{int(beh_equiv)},,\n")
        for r in trial_rows:
            f.write(f"trial,{r['trial']},{r['latency_ms'] if r['latency_ms'] is not None else ''},"
                    f"{int(r['da_spikes']>0)}\n")
        f.write(f"trials,n_valid,{len(lats_trials)},,\n")
        f.write(f"trials,n_fail,{n_fail},,\n")
        f.write(f"trials,sd_ms,{lat_sd:.5f},{sd_ok}\n")
        f.write(f"summary,pass,{int(pass_)},,\n")

    return dict(
        pass_=pass_,
        lat_by_level=lat_by_level, spearman_rho=float(rho), rho_ok=rho_ok,
        lat_i0_ms=lat_i0, ref_lat_i0_ms=ref_lat_i0, ref_err_rel=ref_err, ref_ok=ref_ok,
        window_ms=list(LAT_WINDOW), window_ok=window_ok,
        n_trials_valid=len(lats_trials), n_failures=n_fail,
        latency_sd_ms=lat_sd, sd_ok=sd_ok,
        behavioral_latency_ms=beh_lat, behavioral_ref_ms=beh_ref,
        behavioral_equiv_neural=beh_equiv,
        report_png=REPORT_PNG, report_csv=REPORT_CSV,
    )


if __name__ == "__main__":
    import json
    res = run_p3()
    print(json.dumps(res, indent=2, ensure_ascii=False, default=str))
    print("P3 PASS" if res["pass_"] else "P3 FAIL")
