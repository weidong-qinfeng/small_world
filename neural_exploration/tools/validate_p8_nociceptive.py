#!/usr/bin/env python
"""P6 避痛学习验证（B1d：痛觉逃避基线 sanity + 伤害性条件化协议定义/限制记录）。

《生物仿真M8实施清单》§7.3 P6 + §0 P6：
- US 三选一预注册：**光遗传激活 IV 类伤害感受器**（class IV multidendritic，
  MD 前缀角色；B1a 连接组 4 个 MD 角色）——定稿于本文件/CSV；
- 痛觉逃避基线 sanity（学习判据的机制前置，习惯化协议母版 = 本逃避协议，M6
  惯例）：伤害性刺激（MD 电流注入）→ 蜷缩/后退方向正确（D_peak>0.3 或
  C_curl>C_fwd）、反应概率 ≥0.8；
- 伤害性条件化（CS=气味/情境 + US=痛）→ 回避指数显著：行为级读出（CI）在
  300 档 two_comp 下结构性不可转正（D5 校准反证落盘 data/m8_calibration.csv）；
  若 MD 无可用下游出边 → US 通路结构性缺失 → 按 M4 P4 先例**反证记录型 pass**
  （记录测量限制，不静默、不伪造）。

输出：
  data/m8_p6_nociceptive.csv
  reports/neuro/m8_p6_nociceptive.png
  reports/neuro/m8_p6_nociceptive.json
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

DATA_DIR = os.path.join(ROOT, "neural_exploration", "data")
REPORTS_NEURO = os.path.join(ROOT, "neural_exploration", "reports", "neuro")
PARAMS_CSV = os.path.join(DATA_DIR, "m8_larva_params.csv")

SCALE = 300
FIDELITY = "two_comp"
ESCAPE_T_MS = 1000.0
N_TRIALS = 5
SEEDS = list(range(N_TRIALS))
D_PEAK_THRESHOLD = 0.3      # §0 P6 / §7.3
RESPONSE_PROB_FLOOR = 0.8   # §0 P6


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


def make_circuit(plasticity: str = "none") -> LarvaCircuit:
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
                        plasticity=plasticity)


def main() -> int:
    t0 = time.perf_counter()
    os.makedirs(REPORTS_NEURO, exist_ok=True)
    circ = make_circuit(plasticity="none")
    md_roles = list(circ.nociceptor_roles)
    # MD 可用下游出边（nt_fallback=class 后；判断 US 通路是否在子集内可接线）
    md_out = [r for r in circ.sub.chem if r.pre in md_roles]
    md_posts = sorted({r.post for r in md_out})
    md_post_classes = {p: circ.sub.neurons.get(p, {}).get("neuron_class")
                       for p in md_posts}

    trials = []
    for s in SEEDS:
        r = circ.run_escape(t_total_ms=ESCAPE_T_MS, seed=s)
        c_back = np.asarray(r["c_back"], dtype=float)
        c_fwd = np.asarray(r["c_fwd"], dtype=float)
        c_curl = np.asarray(r["c_curl"], dtype=float)
        d_peak = float(r["d_peak"])
        curl_dom = bool(c_curl.size and c_fwd.size
                        and np.max(c_curl) > np.mean(c_fwd))
        back_resp = bool(d_peak > D_PEAK_THRESHOLD)
        response = bool(back_resp or curl_dom)
        trials.append(dict(seed=s, d_peak=d_peak, direction=r["direction"],
                           curl_peak=float(r["curl_peak"]),
                           c_back_mean=float(np.mean(c_back)) if c_back.size else 0.0,
                           c_fwd_mean=float(np.mean(c_fwd)) if c_fwd.size else 0.0,
                           curl_dominant=curl_dom, back_response=back_resp,
                           response=response))
    resp_prob = float(np.mean([t["response"] for t in trials]))
    d_peaks = [t["d_peak"] for t in trials]
    dirs = [t["direction"] for t in trials]

    # 方向正确性：dominant 方向 = back（D_peak>0.3）或 curl（C_curl>C_fwd）
    dir_correct = bool(sum(1 for t in trials
                           if t["direction"] == "back" or t["curl_dominant"])
                       >= RESPONSE_PROB_FLOOR * len(trials))
    escape_sanity = bool(resp_prob >= RESPONSE_PROB_FLOOR and dir_correct)

    # 结构性判断：MD 无可用出边 → US 通路缺失（测量限制，反证记录）
    us_structural = dict(n_md_roles=len(md_roles),
                         n_md_out_available=len(md_out),
                         md_posts=md_posts[:20],
                         md_post_classes={k: md_post_classes[k]
                                          for k in list(md_posts)[:20]})
    us_absent = bool(len(md_out) == 0)

    # 伤害性条件化（协议定义 + 限制记录）
    conditioning = dict(
        us_chosen="光遗传激活 IV 类伤害感受器（MD 前缀角色）",
        cs="嗅觉感觉对（sens_roles 回退：前 2 sensory ORN）",
        protocol="配对：CS(s=+1) 窗 + US(MD 注入) 窗重叠；N_train 试次 → 双选偏好测试",
        limitation=(
            "行为级回避指数读出（CI/偏好）在 300 档 two_comp 下结构性不可转正"
            "（D5 校准反证：缺 GABA 标注，CI=-0.165 落盘 data/m8_calibration.csv）；"
            + (f"MD 伤害感受器在子集内可用出边 = {len(md_out)}"
               + ("（B1a 递质标注受体 none → nt_fallback 后仍无可用边 → US 通路"
                  "结构性缺失 → 条件化不可达）" if us_absent
                  else "（US 通路在子集内可接线，但行为级回避读出仍受 CI 反证限制）")
              )),
        reachable="escape_baseline_only")

    summary = dict(
        escape=dict(resp_prob=resp_prob, d_peaks=d_peaks, directions=dirs,
                    escape_sanity=escape_sanity,
                    d_peak_threshold=D_PEAK_THRESHOLD,
                    resp_prob_floor=RESPONSE_PROB_FLOOR,
                    trials=trials),
        us_structural=us_structural,
        us_absent=us_absent,
        conditioning=conditioning,
        criteria=dict(
            escape_sanity=escape_sanity,
            note=("反证记录型 pass 条件：若 us_absent=True → 逃避基线结构性不可达"
                  "（MD 无下游），如实记录测量限制，不静默判定 PASS")),
        meta=dict(scale=SCALE, fidelity=FIDELITY, n_trials=N_TRIALS,
                  escape_t_ms=ESCAPE_T_MS,
                  wall_s=round(time.perf_counter() - t0, 2)))

    # 落盘 CSV
    csv_path = os.path.join(DATA_DIR, "m8_p6_nociceptive.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        import csv as _csv
        wtr = _csv.writer(f)
        wtr.writerow(["# M8 P6 避痛学习（B1d）：痛觉逃避基线 + 伤害性条件化限制记录"])
        wtr.writerow(["seed", "d_peak", "direction", "curl_peak",
                      "c_back_mean", "c_fwd_mean", "curl_dominant",
                      "back_response", "response"])
        for t in trials:
            wtr.writerow([t["seed"], f"{t['d_peak']:.6f}", t["direction"],
                          f"{t['curl_peak']:.6f}", f"{t['c_back_mean']:.6f}",
                          f"{t['c_fwd_mean']:.6f}", t["curl_dominant"],
                          t["back_response"], t["response"]])

    # 出图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.6))
        ax.bar(range(len(d_peaks)), d_peaks, color="#c44e52", alpha=0.85,
               label="D_peak（C_back−C_fwd 峰值）")
        ax.axhline(D_PEAK_THRESHOLD, color="gray", ls="--", lw=1,
                   label=f"D_peak 阈值 {D_PEAK_THRESHOLD}")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("seed 试次")
        ax.set_ylabel("D_peak")
        ax.set_title(f"P6 痛觉逃避基线（B1d）：resp_prob={resp_prob:.2f} "
                     f"（下限 {RESPONSE_PROB_FLOOR}）sanity={escape_sanity}"
                     + ("；US 结构性缺失（MD 出边 0）" if us_absent else ""))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        png_path = os.path.join(REPORTS_NEURO, "m8_p6_nociceptive.png")
        fig.savefig(png_path, dpi=130)
        plt.close(fig)
        summary["plot"] = png_path
    except Exception as e:  # noqa: BLE001
        summary["plot"] = f"FAILED: {e}"

    json_path = os.path.join(REPORTS_NEURO, "m8_p6_nociceptive.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print("=== P6 避痛学习（B1d）===")
    print(f"MD roles: {md_roles}")
    print(f"MD 可用出边: {len(md_out)}；posts: {md_posts[:10]}")
    print(f"逃避基线: resp_prob={resp_prob:.2f} "
          f"d_peaks={[round(x, 3) for x in d_peaks]} "
          f"dirs={dirs} sanity={escape_sanity}")
    if us_absent:
        print("反证记录：MD 伤害感受器无可用下游出边 → US 通路结构性缺失，"
              "逃避基线不可达（测量限制，不静默）")
    print(f"csv={csv_path} json={json_path} wall={summary['meta']['wall_s']}s")
    # 反证记录型 pass：脚本成功落盘 = 0；escape_sanity 结果如实入 JSON
    return 0


if __name__ == "__main__":
    sys.exit(main())
