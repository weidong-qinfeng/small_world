#!/usr/bin/env python
"""P8 活动金标准验证（B1d：活动正向模型 + 统计对照冒烟；300 档 two_comp 短协议）。

《生物仿真M8实施清单》§8/D4：
- 活动正向模型（src/larva_activity.py）：发放序列 → GCaMP 类荧光卷积（τ_GCaMP 预注册
  0.5–1.5s，取 1.0s）→ 成像采样 2Hz 窗口平均降采样（§8.1）；
- 统计对照（§8.2）：
  (a) 群体发放率分布 vs 成像荧光事件率分布：KS p>0.05 或 <20% 偏差——成像数据不可得
      （Lemon 2015 全 CNS GCaMP 数据不可下载）→ 文献统计回退 = data/m8_behavior_reference.csv
      resting 带（median<1Hz / max<60Hz / silent∈[50,90]%）+ 测量限制记录（§0.7 #3）；
  (b) run↔turn 转换 ±2s 窗活动态序列（全局活动水平分位二值）：无 NaN、确定性、
      转换条件统计（转换前/后群体发放率、high 态占用）；成像参考矩阵不可得 → 限制记录；
  (c) 逐神经元对应不声称（测量限制）。
- 验收（§8）：活动正向模型冒烟（无 NaN、确定性）+ 统计对照落盘。

输出：data/m8_p8_activity.csv + reports/neuro/m8_p8_activity.png + JSON。
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.src.larva_circuit import LarvaCircuit  # noqa: E402
from neural_exploration.src.virtual_body import (  # noqa: E402
    VirtualBody,
    classify_state,
)
from neural_exploration.src.larva_activity import (  # noqa: E402
    activity_state_sequence,
    global_activity_per_bin,
    spikes_to_fluorescence,
    transition_windows,
)

DATA_DIR = os.path.join(ROOT, "neural_exploration", "data")
REPORTS_NEURO = os.path.join(ROOT, "neural_exploration", "reports", "neuro")
PARAMS_CSV = os.path.join(DATA_DIR, "m8_larva_params.csv")
BEHAV_REF_CSV = os.path.join(DATA_DIR, "m8_behavior_reference.csv")

SCALE = 300
FIDELITY = "two_comp"
SPONT_T_MS = 3000.0
SEED = 0
TAU_GCAMP_MS = 1000.0     # 预注册窗 [500,1500]ms 内取中
DT_IMG_MS = 500.0         # 成像 2Hz
ACT_BIN_MS = 100.0        # 活动率 bin（100ms）
TRANS_WIN_MS = 2000.0     # ±2s 窗（§8.2 (b)）


def load_weight_rows() -> dict:
    out = {}
    import csv as _csv
    with open(PARAMS_CSV, newline="", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            fields = next(_csv.reader([s]))
            if len(fields) < 11 or fields[0].strip().lower() != "weight":
                continue
            try:
                out[fields[1].strip()] = float(fields[9])
            except ValueError:
                continue
    return out


def load_resting_bands() -> dict:
    """m8_behavior_reference.csv resting 带（文献统计回退参考）。"""
    out = {}
    from neural_exploration.src.larva_loop import load_behavior_reference
    ref = load_behavior_reference(BEHAV_REF_CSV)
    for key in ("median_firing_hz", "max_firing_hz", "silent_fraction_band"):
        band = ref.get(("resting", key))
        if band:
            out[key] = band
    return out


def run_spont_with_spikes(circ: LarvaCircuit, t_total_ms: float, seed: int):
    """复刻 run_spontaneous 循环，同时捕获发放时刻（单会话，确定性）。"""
    from brian2 import ms as bms
    body = VirtualBody(v_fwd0=circ.v_fwd0, v_rev0=circ.v_rev0,
                       omega_max=circ.omega_max, dt_b=circ.dt_b,
                       arena_L=circ.arena_L, boundary="reflect")
    sess = circ.make_session(t_total_ms=t_total_ms)
    sess.reset(seed=seed)
    n_epochs = max(1, int(round(t_total_ms / circ.dt_b)))
    states, vs, omegas, mus_hist = [], [], [], []
    for e in range(n_epochs):
        mus = sess.run_epoch(circ.dt_b, 0.0)
        c_fwd = float(mus.get("fwd", 0.0))
        c_back = float(mus.get("back", 0.0))
        c_left = float(mus.get("left", 0.0))
        c_right = float(mus.get("right", 0.0))
        v = body.speed(c_fwd, c_back)
        omega = body.turn_rate(c_left, c_right, e * circ.dt_b)
        st = classify_state(v, omega, c_fwd, c_back,
                            v_fwd0=circ.v_fwd0, omega_max=circ.omega_max)
        body.step(c_fwd, c_back, c_left, c_right, circ.dt_b, e * circ.dt_b)
        states.append(st)
        vs.append(v)
        omegas.append(omega)
        mus_hist.append(mus)
    spike_times = sess.role_spike_times()
    t_end_ms = float(sess.net.t / bms)
    return dict(states=states, vs=np.asarray(vs), omegas=np.asarray(omegas),
                mus_hist=mus_hist, spike_times=spike_times, t_end_ms=t_end_ms,
                n_epochs=n_epochs)


def map_larva_state(st: str) -> str:
    """virtual_body 状态 → 幼虫语义（larva_loop 映射：run=fwd、turn=turn+rev）。"""
    return {"fwd": "run", "rev": "turn"}.get(st, st)


def main() -> int:
    t0 = time.perf_counter()
    os.makedirs(REPORTS_NEURO, exist_ok=True)
    w = load_weight_rows()
    gmax = float(w["gmax_scale"])
    class_scales = {}
    for k, v in w.items():
        if k.startswith("class_scale_"):
            parts = k.split("_")
            if len(parts) == 4:
                class_scales[(parts[2], parts[3])] = v
    ckw = dict(scale=SCALE, fidelity=FIDELITY, seed=SEED,
               nt_fallback="class", provisional_muscles=True,
               gmax_scale=gmax, class_scales=class_scales,
               plasticity="none")
    circ = LarvaCircuit(**ckw)

    r1 = run_spont_with_spikes(circ, SPONT_T_MS, SEED)
    r2 = run_spont_with_spikes(circ, SPONT_T_MS, SEED)   # 确定性重跑

    states1 = [map_larva_state(s) for s in r1["states"]]
    states2 = [map_larva_state(s) for s in r2["states"]]
    det_states = bool(states1 == states2)

    # ---- 发放率分布（逐角色）----
    t_s = r1["t_end_ms"] / 1000.0
    rates = {role: len(t) / t_s for role, t in r1["spike_times"].items()}
    rate_arr = np.array(list(rates.values()), dtype=float)
    median_hz = float(np.median(rate_arr)) if rate_arr.size else float("nan")
    max_hz = float(np.max(rate_arr)) if rate_arr.size else float("nan")
    silent_frac = float(np.mean(rate_arr < 0.5)) if rate_arr.size else float("nan")

    bands = load_resting_bands()
    band_checks = {}
    for key, band in bands.items():
        lo, hi = band["lo"], band["hi"]
        val = {"median_firing_hz": median_hz,
               "max_firing_hz": max_hz,
               "silent_fraction_band": silent_frac * 100.0}.get(key)
        ok = bool(lo is not None and hi is not None and lo <= val <= hi)
        band_checks[key] = dict(value=val, lo=lo, hi=hi, in_band=ok,
                                provenance=band["provenance"])

    # ---- KS（成像参考不可得 → 限制记录；无参考分布则返回 None）----
    ref_csv = os.path.join(DATA_DIR, "m8_imaging_stats.csv")
    ks = None
    if os.path.exists(ref_csv):
        from scipy import stats
        ref = np.loadtxt(ref_csv, delimiter=",", comments="#")
        if ref.size:
            ks = dict(statistic=float(stats.ks_2samp(rate_arr, ref).statistic),
                      p=float(stats.ks_2samp(rate_arr, ref).pvalue))
    imaging_limitation = dict(
        reference="Lemon et al. 2015 Nat Commun 6:8924（一龄幼虫全 CNS GCaMP 体积成像）",
        data_available=False,
        ref_csv=ref_csv,
        note=("成像事件率分布数据不可得（网络/数据获取受限）→ KS 对照不可执行；"
              "按 §0.7 #3 预注册回退 = m8_behavior_reference.csv resting 文献带"
              "（统计级回退）+ 测量限制记录；不伪造成像参考分布。"))

    # ---- 活动正向模型（逐角色荧光 → 帧）----
    frames_all = {}
    for role, times in r1["spike_times"].items():
        frames_all[role] = spikes_to_fluorescence(
            times, SPONT_T_MS, tau_gcamp_ms=TAU_GCAMP_MS, dt_img_ms=DT_IMG_MS)
    frame_mat = np.stack(list(frames_all.values())) if frames_all else \
        np.empty((0, 0))
    has_nan_frames = bool(np.any(~np.isfinite(frame_mat)))
    # 确定性：重跑荧光逐位一致
    frames_det = {}
    for role, times in r2["spike_times"].items():
        frames_det[role] = spikes_to_fluorescence(
            times, SPONT_T_MS, tau_gcamp_ms=TAU_GCAMP_MS, dt_img_ms=DT_IMG_MS)
    det_frames = bool(len(frames_all) == len(frames_det)
                      and all(np.array_equal(frames_all[r], frames_det[r])
                              for r in frames_all))

    # ---- (b) run↔turn 转换 ±2s 窗活动态序列 ----
    act = global_activity_per_bin(r1["spike_times"], SPONT_T_MS,
                                  bin_ms=ACT_BIN_MS)
    act_det = global_activity_per_bin(r2["spike_times"], SPONT_T_MS,
                                      bin_ms=ACT_BIN_MS)
    det_act = bool(np.array_equal(act, act_det))
    tw = transition_windows(states1, act, dt_state_ms=circ.dt_b,
                            win_ms=TRANS_WIN_MS, act_bin_ms=ACT_BIN_MS)
    tw_det = transition_windows(states2, act_det, dt_state_ms=circ.dt_b,
                                win_ms=TRANS_WIN_MS, act_bin_ms=ACT_BIN_MS)
    det_trans = bool(np.array_equal(
        np.concatenate(tw["windows"]) if tw["windows"] else np.array([]),
        np.concatenate(tw_det["windows"]) if tw_det["windows"]
        else np.array([])))
    # 活动态序列全网络（全局分位二值，预注册活动态定义）
    global_states = activity_state_sequence(act, percentile=50.0)
    has_nan_seq = bool(np.any(~np.isfinite(act))
                       or any(not np.all(np.isfinite(x))
                              for x in tw["windows"]))

    # 转换矩阵（模型侧；成像参考距离不可得 → 限制记录）
    occ = {s: states1.count(s) / len(states1) for s in set(states1)}
    n_tr = len(states1) - 1
    tm = {"run_to_turn": 0.0, "turn_to_run": 0.0}
    for i in range(n_tr):
        if states1[i] == "run" and states1[i + 1] == "turn":
            tm["run_to_turn"] += 1.0
        elif states1[i] == "turn" and states1[i + 1] == "run":
            tm["turn_to_run"] += 1.0
    tm["run_to_turn"] = tm["run_to_turn"] / max(1, states1.count("run"))
    tm["turn_to_run"] = tm["turn_to_run"] / max(1, states1.count("turn"))
    matrix_limitation = dict(
        reference="成像参考状态占用/转换矩阵（Lemon 2015）",
        data_available=False,
        note="成像参考矩阵不可得 → 距离判据不可执行；模型侧占用/转换如实落盘 "
             "（informational），测量限制记录（§8.2 (b) 预注册阈值 ≤0.15 留 B2）。")

    # ---- 判据 ----
    crit_a_bands = all(v["in_band"] for v in band_checks.values())
    crit_b = bool(tw["has_nan"] is False and has_nan_seq is False
                  and det_trans and det_act)
    crit_c = bool(not has_nan_frames and det_frames and det_states)
    pass_model = bool(crit_a_bands and crit_b and crit_c)

    summary = dict(
        meta=dict(scale=SCALE, fidelity=FIDELITY, spont_t_ms=SPONT_T_MS,
                  seed=SEED, tau_gcamp_ms=TAU_GCAMP_MS, dt_img_ms=DT_IMG_MS,
                  act_bin_ms=ACT_BIN_MS, trans_win_ms=TRANS_WIN_MS,
                  wall_s=round(time.perf_counter() - t0, 2)),
        rates=dict(median_hz=median_hz, max_hz=max_hz, silent_frac=silent_frac,
                   n_roles=len(rates),
                   distribution_csv="data/m8_p8_activity.csv"),
        band_checks=band_checks,
        ks=ks,
        imaging_limitation=imaging_limitation,
        fluorescence=dict(n_frames=int(frame_mat.shape[1]) if frame_mat.size
                          else 0, has_nan=has_nan_frames,
                          deterministic=det_frames),
        transitions=dict(n=tw["n_transitions"],
                         pre_mean=tw["pre_mean"], post_mean=tw["post_mean"],
                         occupancy_high=tw["occupancy_high"],
                         n_high=tw["n_high"], n_low=tw["n_low"],
                         has_nan=tw["has_nan"], deterministic=det_trans,
                         global_states_non_nan=has_nan_seq is False),
        transition_matrix_model=tm, occupancy_model=occ,
        matrix_limitation=matrix_limitation,
        criteria=dict(a_band_fallback=crit_a_bands,
                      b_transition_sequences=crit_b,
                      c_forward_model_smoke=crit_c,
                      pass_model=pass_model,
                      note=("统计级承诺（§0.7 #3）：成像数据不可得 → 文献带回退 + "
                            "测量限制记录；不逐神经元对应声称")),
        determinism=dict(states=det_states, activity=det_act,
                         fluorescence=det_frames, transitions=det_trans))

    # ---- 落盘 CSV（逐角色发放率 + 荧光帧样例 + 活动态序列）----
    csv_path = os.path.join(DATA_DIR, "m8_p8_activity.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        import csv as _csv
        wtr = _csv.writer(f)
        wtr.writerow(["# M8 P8 活动金标准（B1d）：逐角色发放率 + 荧光/活动态序列摘要"])
        wtr.writerow(["role", "rate_hz", "n_spikes", "fluor_frame_mean",
                      "fluor_frame_max"])
        for role in sorted(rates):
            fr = frames_all.get(role, np.array([0.0]))
            wtr.writerow([role, f"{rates[role]:.6f}", len(r1["spike_times"][role]),
                          f"{float(np.mean(fr)):.6f}",
                          f"{float(np.max(fr)):.6f}"])
        wtr.writerow([])
        wtr.writerow(["# 全局活动态序列（100ms bin；分位二值 1=high/0=low）"])
        wtr.writerow(["bin", "global_rate_hz", "state"])
        for b, (g, st) in enumerate(zip(act, global_states)):
            wtr.writerow([b, f"{g:.6f}", int(st)])

    # ---- 出图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        # (a) 发放率分布
        ax = axes[0]
        ax.hist(rate_arr, bins=20, color="#4c72b0", alpha=0.8)
        ax.axvline(0.5, color="gray", ls="--", lw=1, label="静默阈值 0.5Hz")
        ax.set_xlabel("逐角色发放率 (Hz)")
        ax.set_ylabel("角色数")
        ax.set_title(f"发放率分布：median={median_hz:.2f}Hz max={max_hz:.1f}Hz "
                     f"silent={silent_frac:.2f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        # (b) run↔turn 转换窗活动态序列热图
        ax = axes[1]
        if tw["windows"]:
            maxlen = max(len(w) for w in tw["windows"])
            mat = np.full((len(tw["windows"]), maxlen), -1.0)
            for i, w in enumerate(tw["windows"]):
                mat[i, :len(w)] = w
            ax.imshow(mat, aspect="auto", cmap="viridis", vmin=-1, vmax=1,
                      interpolation="nearest")
            ax.set_xlabel(f"±{TRANS_WIN_MS / 1000:.0f}s 窗内 bin（{ACT_BIN_MS}ms）")
            ax.set_ylabel("转换序号")
            ax.set_title(f"run↔turn 转换窗活动态（high=1/low=0）：n={tw['n_transitions']} "
                         f"high 占用={tw['occupancy_high']:.2f}")
        else:
            ax.text(0.5, 0.5, "无 run↔turn 转换", ha="center", va="center")
            ax.set_title("转换窗活动态序列（空）")
        fig.suptitle("P8 活动金标准（B1d）：发放率分布 + 转换窗活动态序列", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        png_path = os.path.join(REPORTS_NEURO, "m8_p8_activity.png")
        fig.savefig(png_path, dpi=130)
        plt.close(fig)
        summary["plot"] = png_path
    except Exception as e:  # noqa: BLE001
        summary["plot"] = f"FAILED: {e}"

    json_path = os.path.join(REPORTS_NEURO, "m8_p8_activity.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print("=== P8 活动金标准（B1d）===")
    print(f"发放率: median={median_hz:.3f}Hz max={max_hz:.3f}Hz "
          f"silent={silent_frac:.3f} n_roles={len(rates)}")
    for k, v in band_checks.items():
        print(f"  带 {k}: {v['value']:.3f} in [{v['lo']}, {v['hi']}] "
              f"→ {v['in_band']}")
    print(f"荧光帧: n_frames={summary['fluorescence']['n_frames']} "
          f"has_nan={has_nan_frames} det={det_frames}")
    print(f"run↔turn 转换: n={tw['n_transitions']} pre={tw['pre_mean']:.3f} "
          f"post={tw['post_mean']:.3f} high_occ={tw['occupancy_high']:.3f} "
          f"has_nan={tw['has_nan']}")
    print(f"确定性: states={det_states} activity={det_act} "
          f"fluorescence={det_frames} trans={det_trans}")
    print(f"criteria: a={crit_a_bands} b={crit_b} c={crit_c} "
          f"pass_model={pass_model}")
    print(f"imaging_limitation: {imaging_limitation['note'][:60]}…")
    print(f"csv={csv_path} json={json_path} wall={summary['meta']['wall_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
