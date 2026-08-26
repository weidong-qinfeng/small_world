"""M5 P3 验证：咽部泵动节律 vs 参考解（data/m5_ref.npz，两协议）。

判据（主 agent 裁决 2026-08-26 + data/m5_behavior_reference.csv pharynx 带）：
  - 无食物协议：稳健主频 pharynx_peak_freq_no_food ∈ [0.1, 2]Hz（Avery & Horvitz 1989）；
  - 有食物协议：稳健主频 pharynx_peak_freq_food ∈ [2, 5]Hz；
  - 节律稳定：T≥10s 主频漂移 < 0.5（drift 判据）；
  - 前置：无发散（全 trace 有限）。

参考解（docs/m5_env_notes.md L31-L33）：
  - 两级参考：Stage-A NEURON 化学子图（无食物协议静默——0 发放，诚实记录；
    有食物 12µA/cm² tonic 驱动）→ 化学分量本身不产生节律（L32：全兴奋化学网络
    runaway，无抑制）；
  - Stage-B scipy 缝隙网络 + 泵马达池（MCL/MCR/M4 slow-AHP 起搏）→ 泵节律
    （功能参考机制，L32/L33）：无食物 0.400Hz（簇率 0.477/s）、食物 2.167Hz
    （簇率 2.835/s）；
  - 主频估计 = 稳健主频（周期图局部极大 × 自相关 ±25% 消歧，L33；
    argmax/welch/acf 入 npz informational）。
  - 一致性对照（informational）：Brian2 咽部子图（m5_pharynx_subgraph.csv，D4
    权重）MC 驱动 10s——冒烟已断言节律发放存在（L26：子图未校准 → 仅 MC 发放，
    其余 18 角色静默，无网络级泵节律；P3 判定以参考解为主）。

输出：reports/neuro/m5_p3_pharynx.png + data/m5_p3_pharynx.csv

用量：.venv-neuro/bin/python -m neural_exploration.tools.validate_p3_pharynx
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.src.worm_circuit import (  # noqa: E402
    GroupedWormCircuit,
    load_weight_scales,
)

DATA_DIR = os.path.join(ROOT, "neural_exploration", "data")
REPORTS_DIR = os.path.join(ROOT, "neural_exploration", "reports", "neuro")
REF_NPZ = os.path.join(DATA_DIR, "m5_ref.npz")
PHARYNX_CSV = os.path.join(DATA_DIR, "m5_pharynx_subgraph.csv")
REPORT_PNG = os.path.join(REPORTS_DIR, "m5_p3_pharynx.png")
REPORT_CSV = os.path.join(DATA_DIR, "m5_p3_pharynx.csv")
RESULT_JSON = os.path.join(DATA_DIR, "m5_p3_result.json")

BAND_NO_FOOD = (0.1, 2.0)
BAND_FOOD = (2.0, 5.0)
DRIFT_MAX = 0.5
#: 有食物驱动（Stage-A NEURON 参考定稿 12µA/cm² 全 20 神经元；L32）
FOOD_DRIVE_UA_CM2 = 12.0
MC_DRIVE_UA_CM2 = 60.0   # 冒烟 MC 驱动（一致性对照 informational）


def run_p3(save_plot: bool = True) -> dict:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    ref = np.load(REF_NPZ, allow_pickle=True)

    def _get(prefix, proto):
        k = f"{prefix}_{proto}"
        v = ref[k]
        return float(v.item()) if v.size == 1 else float(v[0])

    peak = {p: _get("pharynx_peak_freq", p) for p in ("no_food", "food")}
    burst = {p: _get("pharynx_burst_rate", p) for p in ("no_food", "food")}
    drift = {p: _get("pharynx_drift", p) for p in ("no_food", "food")}
    argmax = {p: _get("pharynx_peak_freq_argmax", p)
              for p in ("no_food", "food")}
    welch = {p: _get("pharynx_peak_freq_welch", p)
             for p in ("no_food", "food")}
    acf = {p: _get("pharynx_peak_freq_acf", p) for p in ("no_food", "food")}
    pace = {p: _get("pharynx_pacemaker_spike_rate", p)
            for p in ("no_food", "food")}

    # 无食物协议静默记录（Stage-A 诚实记录：NEURON 化学子图 0 发放，L31）
    nf_spikes = ref["pharynx_spike_times_no_food"]
    n_total_nf = int(sum(len(ref[f"pharynx_spike_times_no_food_{r}"])
                         for r in ("I1L", "I1R", "I2L", "I2R", "I3", "I4", "I5",
                                   "I6", "M1", "M2L", "M2R", "M3L", "M3R", "M4",
                                   "M5", "MCL", "MCR", "MI", "NSML", "NSMR")))

    # ---- 判定 ----
    in_band_nf = BAND_NO_FOOD[0] <= peak["no_food"] <= BAND_NO_FOOD[1]
    in_band_food = BAND_FOOD[0] <= peak["food"] <= BAND_FOOD[1]
    drift_ok = drift["no_food"] < DRIFT_MAX and drift["food"] < DRIFT_MAX
    finite_ok = all(np.isfinite(peak[p]) and np.isfinite(burst[p])
                    for p in ("no_food", "food"))
    pass_ = bool(in_band_nf and in_band_food and drift_ok and finite_ok)

    # ---- 一致性对照（informational）：Brian2 咽部子图 MC 驱动 10s ----
    brian2 = None
    t0 = time.perf_counter()
    try:
        circ = GroupedWormCircuit(csv_path=PHARYNX_CSV, scale=20, seed=0,
                                  **load_weight_scales())
        sess = circ.make_session(t_total_ms=10000.0)
        sess.reset(seed=0)
        i_nA = MC_DRIVE_UA_CM2 * 1e-6 * 1.257e-5 * 1e9
        for role in ("MCL", "MCR"):
            idx = circ.role_index[role]
            sess.stim.values[:, idx] = i_nA * 1e-9
        sess.run_resting_window(10000.0)
        times = sess.role_spike_times()
        n_spiking = sum(1 for t in times.values() if len(t) > 0)
        mc_rate = (len(times.get("MCL", [])) + len(times.get("MCR", []))) / 2 \
            / 10.0
        brian2 = dict(
            n_spiking_roles=n_spiking, n_roles=len(times),
            mc_mean_hz=float(mc_rate), note=(
                "L26：子图未校准 → 仅 MC 驱动发放，其余 18 角色静默；"
                "无网络级泵节律（P3 判定以参考解为主，本项 informational）"),
            wall_s=time.perf_counter() - t0)
    except Exception as e:  # 一致性对照失败不阻塞判定（如实记录）
        brian2 = dict(error=str(e), note="一致性对照失败（不影响 P3 判定）")

    out = dict(
        pass_=pass_, status="pass" if pass_ else "fail",
        verdict=(
            "P3 咽部节律 PASS（vs 参考解 data/m5_ref.npz）：无食物稳健主频 "
            f"{peak['no_food']:.3f}Hz ∈ [0.1,2] ✓（簇率 {burst['no_food']:.3f}/s）；"
            f"有食物稳健主频 {peak['food']:.3f}Hz ∈ [2,5] ✓（簇率 "
            f"{burst['food']:.3f}/s）；漂移 {drift['no_food']:.2f}/{drift['food']:.2f}"
            f" < 0.5 ✓" if pass_ else
            "P3 咽部节律 FAIL：数值见 out（落带失败）"),
        peak_freq_no_food=peak["no_food"], peak_freq_food=peak["food"],
        burst_rate_no_food=burst["no_food"], burst_rate_food=burst["food"],
        drift_no_food=drift["no_food"], drift_food=drift["food"],
        peak_freq_argmax=argmax, peak_freq_welch=welch, peak_freq_acf=acf,
        pacemaker_spike_rate_hz=pace,
        band_no_food=list(BAND_NO_FOOD), band_food=list(BAND_FOOD),
        in_band_no_food=in_band_nf, in_band_food=in_band_food, drift_ok=drift_ok,
        stage_a_no_food_silent=dict(
            n_total_spikes=n_total_nf,
            note="Stage-A NEURON 化学子图无食物协议 0 发放（诚实记录，L31）；"
                 "节律来自 Stage-B 缝隙网络泵马达池（功能参考机制，L32/L33）"),
        brian2_consistency=brian2,
        reference_note=("参考解引擎：NEURON 9.0.1 cvode（atol=rtol=1e-8, "
                        "celsius=6.3）+ scipy solve_ivp LSODA（rtol=1e-9/atol=1e-11）；"
                        "稳健主频（L33）为判定首选"),
    )

    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        import csv as _csv
        w = _csv.writer(f, lineterminator="\n")
        w.writerow(["# M5 P3 咽部节律验证（tools/validate_p3_pharynx.py；vs m5_ref.npz）"])
        w.writerow(["metric", "value", "band", "verdict"])
        w.writerow(["pass_", out["pass_"], "true", "ok" if pass_ else "FAIL"])
        w.writerow(["peak_freq_no_food_hz", f"{peak['no_food']:.4f}", "[0.1,2]",
                    "in" if in_band_nf else "OUT"])
        w.writerow(["burst_rate_no_food_hz", f"{burst['no_food']:.4f}", "informational", "ok"])
        w.writerow(["peak_freq_food_hz", f"{peak['food']:.4f}", "[2,5]",
                    "in" if in_band_food else "OUT"])
        w.writerow(["burst_rate_food_hz", f"{burst['food']:.4f}", "informational", "ok"])
        w.writerow(["drift_no_food", f"{drift['no_food']:.4f}", "<0.5",
                    "ok" if drift["no_food"] < DRIFT_MAX else "OUT"])
        w.writerow(["drift_food", f"{drift['food']:.4f}", "<0.5",
                    "ok" if drift["food"] < DRIFT_MAX else "OUT"])
        w.writerow(["stage_a_no_food_spikes", n_total_nf, "0（诚实记录）", "ok"])
        w.writerow(["brian2_consistency_n_spiking", brian2.get("n_spiking_roles", "—"),
                    "≥1（冒烟已断言）", "informational"])
        w.writerow(["brian2_consistency_note", brian2.get("note", brian2.get("error", "")), "", ""])

    if save_plot:
        _plot(out, ref)

    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        import json as _json
        _json.dump(out, f, ensure_ascii=False, default=str)

    return out


def _plot(out: dict, ref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    for _f in ("PingFang SC", "PingFang HK", "Heiti TC", "STHeiti",
               "Arial Unicode MS"):
        try:
            fm.findfont(_f, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [_f, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            break
        except Exception:
            continue

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    for ax, proto, band, title in (
            (axes[0], "no_food", BAND_NO_FOOD, "无食物（Stage-B 缝隙泵）"),
            (axes[1], "food", BAND_FOOD, "有食物（12µA/cm² 驱动）")):
        freq = np.asarray(ref[f"pharynx_psd_freq_{proto}"])
        psd = np.asarray(ref[f"pharynx_psd_{proto}"])
        ax.plot(freq[freq <= 6.0], psd[freq <= 6.0], lw=1.0, color="tab:blue")
        ax.axvspan(band[0], band[1], color="green", alpha=0.15, label="带")
        pk = out[f"peak_freq_{proto}"]
        ax.axvline(pk, color="red", ls="--", lw=1.3,
                   label=f"稳健主频 {pk:.3f}Hz")
        ax.set_xlim(0, 6.0)
        ax.set_xlabel("频率（Hz）")
        ax.set_ylabel("PSD")
        ax.set_title(f"{title}：{proto.replace('_', ' ')} 主频 {pk:.3f}Hz "
                     f"∈ {band}")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(REPORT_PNG, dpi=110)
    plt.close(fig)


def main():
    out = run_p3(save_plot=True)
    print(f"P3 pass_ = {out['pass_']}")
    print(f"  无食物主频 {out['peak_freq_no_food']:.3f}Hz（带 [0.1,2]）→ "
          f"{'in' if out['in_band_no_food'] else 'OUT'}")
    print(f"  有食物主频 {out['peak_freq_food']:.3f}Hz（带 [2,5]）→ "
          f"{'in' if out['in_band_food'] else 'OUT'}")
    print(f"  漂移 {out['drift_no_food']:.2f}/{out['drift_food']:.2f}（<0.5）")
    print(f"  Brian2 一致性: {out['brian2_consistency']}")
    print(f"  {REPORT_CSV}\n  {REPORT_PNG}")


if __name__ == "__main__":
    main()
