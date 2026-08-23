"""P5 验证：方向机制消融（主 agent 修订判据，清单 §5.5）。

- 协议：完整链 vs remove_synapse("AVM","VB") 消融链（均确定性 p=1/n=1，同参数）；
  响应窗 [touch_start, touch_start+40ms] 内 VB 发放数 + C_fwd 均值。
- 判据（主 agent 三态裁决落地修订，2026-08-23）：
  pass_ = 发放数子判据 且 响应分量符号子判据：
    * 发放数：消融链响应窗内 VB 发放数 ≥ 完整链 + 1（两引擎一致：2 → 3）；
    * 响应分量符号：完整链 C_fwd 响应分量 < 0（GABA 抑制压过张力 → C_fwd 低于无触基线）
      且 消融链 C_fwd 响应分量 ≥ −1e-6（无 GABA → 触刺激不改变 VB 发放）。
  响应分量 = 响应窗 C_fwd 均值 − 无刺激对照同窗均值。
  C_fwd 窗均值 ×1.2 幅度子判据因 VB 张力 ~60Hz 基线振荡主导窗均值而结构性不可达
  （两引擎均 ≈1.05，NEURON 参考 1.054 / Brian2 1.058）——主 agent 判据校准错误，
  非机制失败；该比值仅作 informational 输出，不再进 pass 判定。
- 实测诊断（两引擎一致）：GABA 只推迟/抑制响应窗内 1 个 VB 发放；D_peak 两情形相同
  （0.369，m3_env_notes §L8 预测复现——D_peak 时刻早于 GABA 生效时刻）。
输出：reports/neuro/m3_p5_ablation.png + data/m3_p5_ablation.csv
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
REPORT_PNG = os.path.join(REPORTS_DIR, "m3_p5_ablation.png")
REPORT_CSV = os.path.join(DATA_DIR, "m3_p5_ablation.csv")

WINDOW_MS = 40.0          # 响应窗 [touch_start, touch_start+40ms]
COUNT_DELTA = 1           # 消融链 VB 发放数 ≥ 完整链 + 1
MEAN_RATIO = 1.2          # 消融链 C_fwd 窗均值 > 完整链 × 1.2


def run_p5(save_plot: bool = True) -> dict:
    arc_full = ReflexArc(csv_path=CSV_PATH)          # 完整链（确定性）
    arc_abl = ReflexArc(csv_path=CSV_PATH)           # 消融链
    arc_abl.remove_synapse("AVM", "VB")
    arc_ctrl = ReflexArc(csv_path=CSV_PATH)          # 无刺激对照（响应分量诊断）

    rf = arc_full.run(intensity=1.0)
    ra = arc_abl.run(intensity=1.0)
    rc = arc_ctrl.run(intensity=0.0)

    touch_start = rf.meta["touch_start_ms"]
    win = (rf.t_ms >= touch_start) & (rf.t_ms <= touch_start + WINDOW_MS)

    vbf = rf.spikes("VB", "node3")
    vba = ra.spikes("VB", "node3")
    n_full = int(((vbf >= touch_start) & (vbf <= touch_start + WINDOW_MS)).sum())
    n_abl = int(((vba >= touch_start) & (vba <= touch_start + WINDOW_MS)).sum())
    count_ok = bool(n_abl >= n_full + COUNT_DELTA)

    mf = float(rf.c_fwd[win].mean())
    ma = float(ra.c_fwd[win].mean())
    mean_ratio = ma / mf if mf > 0 else float("nan")
    mean_ratio_ok_info = bool(mean_ratio > MEAN_RATIO)   # informational（裁决后不进 pass）

    # 诊断：响应分量（窗均值 − 无刺激对照同窗均值）；D_peak / C_back 对照
    mc = float(rc.c_fwd[win].mean())
    resp_full = mf - mc
    resp_abl = ma - mc
    d_full, d_abl = float(rf.d_peak), float(ra.d_peak)
    cb_full, cb_abl = float(rf.c_back_peak), float(ra.c_back_peak)

    # 主 agent 落地修订判据：发放数 + 响应分量符号（完整链被抑制、消融链无抑制）
    resp_component_sign_ok = bool(resp_full < 0.0 and resp_abl >= -1e-6)
    pass_ = bool(count_ok and resp_component_sign_ok)

    if save_plot:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
        # 1) C_fwd / C_back 对比（完整 vs 消融）
        ax = axes[0]
        ax.plot(rf.t_ms, rf.c_fwd, lw=1.2, color="#8c564b",
                label=f"C_fwd full (nVB={n_full})")
        ax.plot(ra.t_ms, ra.c_fwd, lw=1.2, ls="--", color="#d62728",
                label=f"C_fwd ablated (nVB={n_abl})")
        ax.plot(rf.t_ms, rf.c_back, lw=1.0, color="#2ca02c", alpha=0.7,
                label="C_back (both, identical)")
        ax.axvspan(touch_start, touch_start + WINDOW_MS, color="orange", alpha=0.15,
                   label=f"response window {WINDOW_MS:.0f}ms")
        ax.set_xlabel("t (ms)"); ax.set_ylabel("muscle C")
        ax.set_title(f"count {n_full}→{n_abl} (≥+1: {count_ok}); "
                     f"resp-sign full={resp_full:+.4f} (<0: {resp_full<0}) / "
                     f"abl={resp_abl:+.4f} (≥0: {resp_abl>=0}); "
                     f"mean-ratio {mean_ratio:.3f} (informational)")
        ax.legend(fontsize=7.5); ax.grid(alpha=0.3); ax.set_ylim(0, 0.75)
        # 2) VB 发放栅栏图
        ax = axes[1]
        ax.eventplot(vbf, lineoffsets=1, linewidths=2.2, colors="#8c564b",
                     label=f"VB full ({len(vbf)} spikes)")
        ax.eventplot(vba, lineoffsets=0, linewidths=2.2, colors="#d62728",
                     label=f"VB ablated ({len(vba)} spikes)")
        ax.axvspan(touch_start, touch_start + WINDOW_MS, color="orange", alpha=0.15)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["ablated", "full"])
        ax.set_xlabel("t (ms)"); ax.set_xlim(0, 150)
        ax.set_title(f"D_peak full={d_full:.3f} vs ablated={d_abl:.3f} (L8 预测复现: 相同); "
                     f"response ΔC_fwd full={resp_full:+.4f} vs abl={resp_abl:+.4f}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.suptitle("P5: directional-mechanism ablation (AVM→VB GABA removed)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(REPORT_PNG, dpi=150)
        plt.close(fig)

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(REPORT_CSV, "w", encoding="utf-8") as f:
        f.write("metric,full,ablated,pass\n")
        f.write(f"vb_spikes_in_window,{n_full},{n_abl},{int(count_ok)}\n")
        f.write(f"c_fwd_window_mean,{mf:.6f},{ma:.6f},\n")
        f.write(f"c_fwd_window_mean_ratio,1.0,{mean_ratio:.6f},{int(mean_ratio_ok_info)}\n")
        f.write(f"c_fwd_response_component,{resp_full:.6f},{resp_abl:.6f},{int(resp_component_sign_ok)}\n")
        f.write(f"d_peak,{d_full:.6f},{d_abl:.6f},,\n")
        f.write(f"c_back_peak,{cb_full:.6f},{cb_abl:.6f},,\n")
        f.write(f"vb_total_spikes,{len(vbf)},{len(vba)},,\n")
        f.write(f"window_ms,{touch_start},{touch_start + WINDOW_MS},,\n")
        f.write(f"summary,count_ok,{int(count_ok)},,\n")
        f.write(f"summary,resp_component_sign_ok,{int(resp_component_sign_ok)},,\n")
        f.write(f"summary,mean_ratio_info_ok,{int(mean_ratio_ok_info)},,\n")
        f.write(f"summary,pass,{int(pass_)},,\n")

    return dict(
        pass_=pass_,
        window_ms=(touch_start, touch_start + WINDOW_MS),
        vb_spikes_full=n_full, vb_spikes_ablated=n_abl, count_ok=count_ok,
        c_fwd_mean_full=mf, c_fwd_mean_ablated=ma, c_fwd_mean_ratio=mean_ratio,
        c_fwd_mean_ratio_info_ok=mean_ratio_ok_info,
        c_fwd_response_full=resp_full, c_fwd_response_ablated=resp_abl,
        resp_component_sign_ok=resp_component_sign_ok,
        d_peak_full=d_full, d_peak_ablated=d_abl,
        c_back_peak_full=cb_full, c_back_peak_ablated=cb_abl,
        vb_total_spikes_full=len(vbf), vb_total_spikes_ablated=len(vba),
        criterion="count_ok AND resp_component_sign_ok（主 agent 三态裁决落地修订）",
        note=("主 agent 裁决（2026-08-23）：机制已由直接证据证明——发放数子判据两引擎一致 "
              "（响应窗 [50,90]ms VB 完整 2 → 消融 3）+ 响应分量符号（full −0.011 < 0 被抑制、"
              "abl 0.000 ≥ 0 无抑制）；C_fwd 窗均值 ×1.2 子判据因 VB 张力 ~60Hz 基线振荡主导"
              "窗均值而结构性不可达（两引擎均 ≈1.05，NEURON 1.054 / Brian2 1.058）——主 agent "
              "判据校准错误，非机制失败；该比值保留为 informational 输出。D_peak 两情形相同"
              "（0.369，L8 预测复现）。"),
        report_png=REPORT_PNG, report_csv=REPORT_CSV,
    )


if __name__ == "__main__":
    import json
    res = run_p5()
    print(json.dumps(res, indent=2, ensure_ascii=False, default=str))
    print("P5 PASS" if res["pass_"] else "P5 FAIL")
