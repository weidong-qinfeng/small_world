#!/usr/bin/env python
"""P4 自发行为分布全协议验证（B2；清单 §7.1，tools/validate_p8_spontaneous.py）。

《生物仿真M8实施清单》§7.1 + §0 P4：
- 协议：无刺激无梯度 300 档 two_comp（G0 定稿保真度）T≥30s × N≥10（种子 0..9，
  D5 定稿权重）；`classify_larva_state` 分类（larva_body 语义，curl 优先、
  净后退并入 turn——与 m8_larva_body_params.csv note_rev_in_turn 一致）；
- 判据（§0.7 #8）：run/turn/pause 时间比例 vs `data/m8_behavior_reference.csv`
  带（唯一定稿源，不事后调）；bout 时长量级 informational；阈值不事后调；
- 反证联动（§0.4 反证路径）：比例不落带 → 命令/运动耦合排查（§1 D6 杠杆）→
  缺失机制清单 + 三态裁决请求（B2 预算裁决：行为判据以 300 档 two_comp 全协议
  为准；3016 全规模只做结构性验证——G1 已 PASS silent=0.8477/0.8167）；
- 确定性：p=1/n=1；同参数重跑逐位一致（seed 0 复跑 frac/states 逐位比较）。

已知反证信号（预算实测，D5 权重 300 two_comp 自发）：fwd=9.9%/rev=18.9%/
turn=62.5%/pause=8.7% → run 比例 9.9% 远低于带 [60,85]%、turn 81.4%（rev 并入）
远超带 [10,30]% → 自发分布不落 P4 带 = **行为不涌现反证**（本脚本如实复测 +
N=10 全协议确认，不静默、不伪造）。

输出：data/m8_p4_spontaneous.csv + reports/neuro/m8_p4_spontaneous.png +
      reports/neuro/m8_p4_spontaneous.json。
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

from neural_exploration.src.larva_body import (  # noqa: E402
    VirtualLarvaBody,
    classify_larva_state,
    larva_state_fractions,
)
from neural_exploration.src.larva_loop import (  # noqa: E402
    band_check,
    load_behavior_reference,
)

DATA_DIR = os.path.join(ROOT, "neural_exploration", "data")
REPORTS_NEURO = os.path.join(ROOT, "neural_exploration", "reports", "neuro")
PARAMS_CSV = os.path.join(DATA_DIR, "m8_larva_params.csv")
BEHAV_REF_CSV = os.path.join(DATA_DIR, "m8_behavior_reference.csv")
BODY_PARAMS_CSV = os.path.join(DATA_DIR, "m8_larva_body_params.csv")

#: 全协议（G0 定稿保真度 two_comp；T≥30s × N≥10，B2 预算裁决）
SCALE = 300
FIDELITY = "two_comp"
T_TOTAL_MS = 30000.0
N_TRIALS = 10
SEEDS = list(range(N_TRIALS))


def load_weight_rows() -> dict:
    """读 m8_larva_params.csv 的 weight 行（D5 定稿；value 在 fields[9]）。"""
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


def make_circuit():
    """D5 定稿权重电路（300 档 two_comp，nt_fallback=class，provisional 肌肉）。"""
    from neural_exploration.src.larva_circuit import LarvaCircuit
    w = load_weight_rows()
    gmax = float(w["gmax_scale"])
    class_scales = {}
    for k, v in w.items():
        if k.startswith("class_scale_"):
            parts = k.split("_")
            if len(parts) == 4:
                class_scales[(parts[2], parts[3])] = v
    return LarvaCircuit(scale=SCALE, fidelity=FIDELITY, seed=0,
                        nt_fallback="class", provisional_muscles=True,
                        gmax_scale=gmax, class_scales=class_scales,
                        plasticity="none")


def run_spont_classified(circ, t_total_ms: float, seed: int) -> dict:
    """无刺激自发协议 + `classify_larva_state` 分类（清单 §7.1 语义）。

    逐 epoch 取肌肉通道（fwd/back/left/right/curl）→ VirtualLarvaBody
    运动学（v/ω，curl 优先）→ classify_larva_state（阈值来自
    m8_larva_body_params.csv 定稿语义）。确定性 p=1/n=1。
    """
    from neural_exploration.src.larva_body import load_larva_body_params
    bp = load_larva_body_params(BODY_PARAMS_CSV)
    st = bp["state"]
    body = VirtualLarvaBody(n_seg=int(bp["body"]["n_seg"]),
                            v_fwd0=circ.v_fwd0, v_rev0=circ.v_rev0,
                            omega_max=circ.omega_max, dt_b=circ.dt_b,
                            arena_L=circ.arena_L, boundary="reflect")
    sess = circ.make_session(t_total_ms=t_total_ms)
    sess.reset(seed=seed)
    n_epochs = max(1, int(round(t_total_ms / circ.dt_b)))
    states, vs, omegas = [], [], []
    ch_acc = dict(fwd=0.0, back=0.0, left=0.0, right=0.0, curl=0.0)
    for e in range(n_epochs):
        mus = sess.run_epoch(circ.dt_b, 0.0)
        c_fwd = float(mus.get("fwd", 0.0))
        c_back = float(mus.get("back", 0.0))
        c_left = float(mus.get("left", 0.0))
        c_right = float(mus.get("right", 0.0))
        c_curl = float(mus.get("curl", 0.0))
        for k in ch_acc:
            ch_acc[k] += float(mus.get(k, 0.0))
        v = body.speed(c_fwd, c_back)
        omega = body.turn_rate(c_left, c_right, e * circ.dt_b)
        stt = classify_larva_state(v, omega, c_curl,
                                   v_thr_frac=st["v_run_frac"],
                                   omega_thr_frac=st["omega_turn_frac"],
                                   curl_thr=st["curl_thr"],
                                   v_fwd0=circ.v_fwd0,
                                   omega_max=circ.omega_max)
        body.step(c_fwd, c_back, c_left, c_right, c_curl,
                  circ.dt_b, e * circ.dt_b)
        states.append(stt)
        vs.append(v)
        omegas.append(omega)
    frac = larva_state_fractions(states)
    ch_mean = {k: v / max(1, n_epochs) for k, v in ch_acc.items()}
    # bout 时长（run/turn 连续游程，informational）
    bouts = _run_lengths(states, "run")
    turn_bouts = _run_lengths(states, "turn")
    return dict(frac=frac, states=states, ch_mean=ch_mean, n_epochs=n_epochs,
                bout_run=bouts, bout_turn=turn_bouts)


def _run_lengths(states, target: str) -> list:
    """连续 target 状态游程长度（epoch 数 × dt_b ms）。"""
    out, cur = [], 0
    for s in states:
        if s == target:
            cur += 1
        else:
            if cur:
                out.append(cur)
            cur = 0
    if cur:
        out.append(cur)
    return out


def main() -> int:
    t0 = time.perf_counter()
    os.makedirs(REPORTS_NEURO, exist_ok=True)
    ref = load_behavior_reference(BEHAV_REF_CSV)

    circ = make_circuit()
    results = []
    progress_csv = os.path.join(DATA_DIR, "m8_p4_spontaneous_progress.csv")
    with open(progress_csv, "w", newline="", encoding="utf-8") as f:
        import csv as _csv
        wtr = _csv.writer(f)
        wtr.writerow(["seed", "run_frac", "turn_frac", "pause_frac",
                      "curl_frac", "wall_s"])
        for s in SEEDS:
            ts = time.perf_counter()
            r = run_spont_classified(circ, T_TOTAL_MS, s)
            wall = time.perf_counter() - ts
            rec = dict(seed=s, frac=r["frac"], states=r["states"],
                       ch_mean=r["ch_mean"], n_epochs=r["n_epochs"],
                       bout_run=r["bout_run"], bout_turn=r["bout_turn"],
                       wall_s=wall)
            results.append(rec)
            wtr.writerow([s, f"{r['frac'].get('run', 0.0):.9f}",
                          f"{r['frac'].get('turn', 0.0):.9f}",
                          f"{r['frac'].get('pause', 0.0):.9f}",
                          f"{r['frac'].get('curl', 0.0):.9f}",
                          f"{wall:.1f}"])
            f.flush()
            print(f"  seed {s}: run={r['frac'].get('run', 0.0):.4f} "
                  f"turn={r['frac'].get('turn', 0.0):.4f} "
                  f"pause={r['frac'].get('pause', 0.0):.4f} "
                  f"wall={wall:.1f}s", flush=True)

    # ---- 聚合（N=10）----
    states_all = []
    for s in SEEDS:
        states_all += [("seed%d" % s, x) for x in results[s]["states"]]
    mean_frac = {}
    sd_frac = {}
    for k in ("run", "turn", "pause", "curl"):
        vals = np.array([r["frac"].get(k, 0.0) for r in results])
        mean_frac[k] = float(np.mean(vals))
        sd_frac[k] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    # 带判定（m8_behavior_reference.csv 唯一定稿源；容差窗语义同 larva_loop）
    checks = {}
    for state in ("run", "turn", "pause", "curl"):
        band = ref.get(("spontaneous", f"time_fraction_{state}"))
        checks[state] = band_check(mean_frac[state] * 100.0, band)
    bout = sum(mean_frac.get(k, 0.0) for k in ("run", "turn"))
    checks["bout_activity"] = dict(in_band=bool(bout >= 0.10),
                                   value=round(bout, 4), band=None)

    # bout 时长（informational）：run/turn 游程均值/中位（ms）
    run_lens = [l for r in results for l in r["bout_run"]]
    turn_lens = [l for r in results for l in r["bout_turn"]]
    bout_stats = dict(
        run_ms_mean=float(np.mean(run_lens) * circ.dt_b) if run_lens else 0.0,
        run_ms_median=float(np.median(run_lens) * circ.dt_b) if run_lens else 0.0,
        turn_ms_mean=float(np.mean(turn_lens) * circ.dt_b) if turn_lens else 0.0,
        turn_ms_median=float(np.median(turn_lens) * circ.dt_b) if turn_lens else 0.0,
        n_run_bouts=len(run_lens), n_turn_bouts=len(turn_lens))

    # ---- 确定性：seed 0 复跑逐位一致 ----
    r_det = run_spont_classified(circ, T_TOTAL_MS, 0)
    det_frac = bool(r_det["frac"] == results[0]["frac"])
    det_states = bool(r_det["states"] == results[0]["states"])
    determinism = bool(det_frac and det_states)

    # ---- P4 主判据：run/turn/pause 落带（容差窗）----
    crit_run = bool(checks["run"]["in_band"])
    crit_turn = bool(checks["turn"]["in_band"])
    crit_pause = bool(checks["pause"]["in_band"])
    pass_spontaneous = bool(crit_run and crit_turn and crit_pause)

    # ---- 反证记录（§0.4：不静默、不伪造；已知 D5 信号复测确认）----
    counter_evidence = dict(
        run_pct=round(mean_frac["run"] * 100.0, 2),
        turn_pct=round(mean_frac["turn"] * 100.0, 2),
        pause_pct=round(mean_frac["pause"] * 100.0, 2),
        curl_pct=round(mean_frac["curl"] * 100.0, 2),
        band_run_pct=[60.0, 85.0], band_turn_pct=[10.0, 30.0],
        band_pause_pct=[3.0, 20.0],
        verdict=("自发分布不落 P4 带：run 比例远低于 [60,85]%（感觉驱动过强 → "
                 "转向过多，turn 远超 [10,30]%）→ **行为不涌现反证**（B2 全协议 "
                 "N=10 确认 B1d 预算实测信号）"),
        missing_mechanisms=[
            dict(mechanism="D5 权重 s2i6 放大（class_scale_sensory_inter=6.0）→ "
                           "感觉驱动过强 → 转向（turn）主导、run 稀缺",
                 check="降低 s2i6 或改 s2i6 分段缩放后复测自发分布；校准反证记录于 "
                       "data/m8_calibration.csv（d5_g050）"),
            dict(mechanism="缺 GABA 递质标注（B1a：inter 递质经 nt_fallback=class "
                           "hash 分配，非真实 GABA）→ 抑制平衡缺失 → 网络难维持 "
                           "稳定前进模态",
                 check="补充 GABA 标注/递质行后重测（主 agent 裁决）"),
            dict(mechanism="provisional 肌肉映射仅 fwd/back/left/right（无真实 "
                           "curl 通道）→ curl=0 固定，防御态缺失",
                 check="curl 通道留真实肌肉映射（P3 定稿后）"),
            dict(mechanism="运动层与命令层分离驱动（杠杆②）在当前 D5 权重下仍 "
                           "不足以稳定 run 模态",
                 check="杠杆组合消融 sanity（§1 D6 三杠杆）")],
        three_state_verdict_request=(
            "P4 行为判据 = FAIL（反证记录）。请求主 agent 三态裁决："
            "①接受反证路径（缺失机制清单如上，M9 必需机制清单累积）；"
            "②或裁决调整 D5 权重（降低 s2i6 / 补 GABA 标注）后 B3 复测；"
            "③或裁定 P4 在 3016 全规模结构验证（G1 已 PASS）下按 M4 P4 先例"
            "记录反证型 pass。本节点不静默判定 PASS。"))

    summary = dict(
        meta=dict(scale=SCALE, fidelity=FIDELITY, t_total_ms=T_TOTAL_MS,
                  n_trials=N_TRIALS, seeds=SEEDS,
                  wall_s=round(time.perf_counter() - t0, 2)),
        per_trial=[dict(seed=r["seed"], frac=r["frac"],
                        ch_mean=r["ch_mean"], wall_s=round(r["wall_s"], 2))
                   for r in results],
        aggregate=dict(mean_frac=mean_frac, sd_frac=sd_frac,
                       bout_stats=bout_stats),
        band_checks=checks,
        determinism=dict(frac_identical=det_frac, states_identical=det_states,
                         identical=determinism),
        criteria=dict(run_in_band=crit_run, turn_in_band=crit_turn,
                      pause_in_band=crit_pause, pass_spontaneous=pass_spontaneous),
        counter_evidence=counter_evidence,
        note=("B2 预算裁决：行为判据以 300 档 two_comp 全协议为准（G0 定稿 "
              "保真度 two_comp）；3016 全规模长协议行为判据不可行（3016 point "
              "30s=843s/试次；3016 point CI=0 行为层退化反证；3016 two_comp 组合"
              "从未构建——B1b 遗留裁决）→ 3016 只做结构性验证（G1 PASS："
              "silent=0.8477/0.8167，m8_larva_params.csv g1 行）"),
        plot="", csv="")

    # ---- 落盘 CSV（逐试次 + 聚合带判定）----
    csv_path = os.path.join(DATA_DIR, "m8_p4_spontaneous.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        import csv as _csv
        wtr = _csv.writer(f)
        wtr.writerow(["# M8 P4 自发行为分布（B2 全协议：300 two_comp T=30s N=10）"])
        wtr.writerow(["# 判据带（m8_behavior_reference.csv 定稿）：run [60,85]% "
                      "turn [10,30]% pause [3,20]% curl informational"])
        wtr.writerow(["seed", "run_frac", "turn_frac", "pause_frac", "curl_frac",
                      "run_pct", "turn_pct", "pause_pct", "curl_pct", "wall_s"])
        for r in results:
            fr = r["frac"]
            wtr.writerow([r["seed"], f"{fr.get('run', 0.0):.9f}",
                          f"{fr.get('turn', 0.0):.9f}",
                          f"{fr.get('pause', 0.0):.9f}",
                          f"{fr.get('curl', 0.0):.9f}",
                          f"{fr.get('run', 0.0) * 100:.3f}",
                          f"{fr.get('turn', 0.0) * 100:.3f}",
                          f"{fr.get('pause', 0.0) * 100:.3f}",
                          f"{fr.get('curl', 0.0) * 100:.3f}",
                          f"{r['wall_s']:.1f}"])
        wtr.writerow([])
        wtr.writerow(["AGG", f"{mean_frac['run']:.9f}", f"{mean_frac['turn']:.9f}",
                      f"{mean_frac['pause']:.9f}", f"{mean_frac['curl']:.9f}",
                      f"{mean_frac['run'] * 100:.3f}",
                      f"{mean_frac['turn'] * 100:.3f}",
                      f"{mean_frac['pause'] * 100:.3f}",
                      f"{mean_frac['curl'] * 100:.3f}", ""])
    summary["csv"] = csv_path

    # ---- 出图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        for _f in ("PingFang HK", "Heiti TC", "PingFang SC", "Arial Unicode MS"):
            try:
                fm.findfont(_f, fallback_to_default=False)
                plt.rcParams["font.sans-serif"] = [_f, "DejaVu Sans"]
                plt.rcParams["axes.unicode_minus"] = False
                break
            except Exception:
                continue
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        # (a) 逐试次状态分布（N=10）
        ax = axes[0]
        xpos = np.arange(N_TRIALS)
        run_v = [r["frac"].get("run", 0.0) * 100 for r in results]
        turn_v = [r["frac"].get("turn", 0.0) * 100 for r in results]
        pause_v = [r["frac"].get("pause", 0.0) * 100 for r in results]
        ax.bar(xpos, run_v, color="#4c72b0", label="run")
        ax.bar(xpos, turn_v, bottom=run_v, color="#55a868", label="turn")
        ax.bar(xpos, pause_v, bottom=[a + b for a, b in zip(run_v, turn_v)],
               color="#c44e52", label="pause")
        ax.axhspan(60, 85, color="#4c72b0", alpha=0.12)
        ax.axhspan(10, 30, color="#55a868", alpha=0.12)
        ax.set_xlabel("seed 试次")
        ax.set_ylabel("时间比例 (%)")
        ax.set_title(f"P4 自发分布（N={N_TRIALS}）：run={mean_frac['run'] * 100:.1f}% "
                     f"turn={mean_frac['turn'] * 100:.1f}% "
                     f"pause={mean_frac['pause'] * 100:.1f}%")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        # (b) 聚合 vs 带
        ax = axes[1]
        states = ["run", "turn", "pause"]
        means = [mean_frac[s] * 100 for s in states]
        sds = [sd_frac[s] * 100 for s in states]
        bands = {"run": (60, 85), "turn": (10, 30), "pause": (3, 20)}
        colors = ["#4c72b0", "#55a868", "#c44e52"]
        x2 = np.arange(3)
        ax.bar(x2, means, yerr=sds, color=colors, alpha=0.85, capsize=4)
        for i, s in enumerate(states):
            lo, hi = bands[s]
            ax.plot([i - 0.35, i + 0.35], [lo, lo], "k--", lw=1.2)
            ax.plot([i - 0.35, i + 0.35], [hi, hi], "k--", lw=1.2)
        ax.set_xticks(x2)
        ax.set_xticklabels(states)
        ax.set_ylabel("时间比例 (%)")
        ax.set_title(f"P4 判据带判定：PASS={pass_spontaneous}"
                     + ("（反证：run 不落带 / turn 超带）" if not pass_spontaneous
                        else ""))
        ax.grid(True, alpha=0.3)
        fig.suptitle("M8 P4 自发行为分布（B2 全协议 300 two_comp）："
                     + ("不落带 → 反证记录 + 缺失机制清单" if not pass_spontaneous
                        else "落带"),
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        png_path = os.path.join(REPORTS_NEURO, "m8_p4_spontaneous.png")
        fig.savefig(png_path, dpi=130)
        plt.close(fig)
        summary["plot"] = png_path
    except Exception as e:  # noqa: BLE001
        summary["plot"] = f"FAILED: {e}"

    json_path = os.path.join(REPORTS_NEURO, "m8_p4_spontaneous.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print("=== P4 自发行为分布（B2 全协议）===")
    print(f"N={N_TRIALS} T={T_TOTAL_MS:.0f}ms {SCALE} two_comp（D5 权重）")
    for k in ("run", "turn", "pause", "curl"):
        print(f"  {k}: {mean_frac[k] * 100:.2f}% ± {sd_frac[k] * 100:.2f} "
              f"带内={checks[k]['in_band']}")
    print(f"  bout 活动: {bout:.3f}（≥0.10 → {checks['bout_activity']['in_band']}）")
    print(f"  run bout: mean={bout_stats['run_ms_mean']:.0f}ms "
          f"median={bout_stats['run_ms_median']:.0f}ms（informational）")
    print(f"determinism: frac={det_frac} states={det_states} "
          f"→ {determinism}")
    print(f"criteria: run={crit_run} turn={crit_turn} pause={crit_pause} "
          f"→ pass_spontaneous={pass_spontaneous}")
    print("反证记录：自发分布不落 P4 带 → 行为不涌现反证；缺失机制清单 + "
          "三态裁决请求已入 JSON/报告")
    print(f"csv={csv_path} json={json_path} "
          f"wall={summary['meta']['wall_s']}s")
    return 0 if pass_spontaneous else 1


if __name__ == "__main__":
    sys.exit(main())
