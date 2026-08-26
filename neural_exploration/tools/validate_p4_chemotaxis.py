"""M5 P4 验证：趋化全协议（T=15s × N=20 梯度 + 对照，D4 定稿权重 302 全连接组）。

判据（主 agent 裁决 2026-08-26，L39/L41 #2）：
  (a) 显著性：CĪ > 0，单样本 t 检验 p < 0.05 且 Cohen's d ≥ 0.5；
  (b) 落带且与行为参考 ΔCI ≤ 0.15 **或**方向一致（CĪ > 0）；
      参考 = 行为参考模型 T=15s（m4_calibration.csv ref-T15000：CI=0.417）；
  (c) 终态偏向（dist_end_food < dist_start_food 比例，informational→记录）；
  (d) 无梯度对照：p > 0.05（|CĪ| < 0.1 informational）；
  前置：轨迹有界/无 NaN（WormLoop 内部断言，异常抛错）。
  预算纪律（预注册 ≤200 CPU-h）：单试次墙钟 > 5min（300s）→ 报主 agent（不静默）。

协议（data/m5_worm_params.csv 定稿）：T=15000ms、N=20、start_x/y=5.0±0.3、
theta0∈[0,2π)（伪随机起点，确定性 p=1/n=1；方差来自起点扰动）；
权重 = load_weight_scales()（D4 定稿：gap_scale=0.05，类级=先验 1.0）。

输出：reports/neuro/m5_p4_chemotaxis.png + data/m5_p4_chemotaxis.csv

用量：.venv-neuro/bin/python -m neural_exploration.tools.validate_p4_chemotaxis
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
    load_weight_scales,
    make_worm_circuit,
)
from neural_exploration.src.worm_loop import WormLoop  # noqa: E402

DATA_DIR = os.path.join(ROOT, "neural_exploration", "data")
REPORTS_DIR = os.path.join(ROOT, "neural_exploration", "reports", "neuro")
REPORT_PNG = os.path.join(REPORTS_DIR, "m5_p4_chemotaxis.png")
REPORT_CSV = os.path.join(DATA_DIR, "m5_p4_chemotaxis.csv")
RESULT_JSON = os.path.join(DATA_DIR, "m5_p4_result.json")
CAL_CSV = os.path.join(DATA_DIR, "m4_calibration.csv")

#: P4 协议（m5_worm_params.csv protocol 行定稿）
T_TOTAL_MS = 15000.0
N_TRIALS = 20
CI_TOL = (0.25, 0.75)       # ci_band_tolerance（data/m5_behavior_reference.csv）
DELTA_CI_MAX = 0.15
BUDGET_TRIAL_WALL_S = 300.0  # 单试次 >5min → 报主 agent


def _load_reference_ci_15s() -> float:
    """参考模型 CI@T=15s（m4_calibration.csv ref-T15000 行；L41 #2 引用 0.417）。"""
    import csv as _csv
    with open(CAL_CSV, newline="", encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            if r.get("point_id") == "ref-T15000":
                return float(r["ci_mean"])
    return float("nan")


def _stats(cis: np.ndarray) -> dict:
    from scipy import stats as sps

    c = np.asarray(cis, dtype=float)
    n = int(c.size)
    mean = float(c.mean())
    sd = float(c.std(ddof=1)) if n > 1 else 0.0
    sem = sd / np.sqrt(n) if n > 0 else float("nan")
    if n > 1 and sd > 0:
        t_stat, p_val = sps.ttest_1samp(c, 0.0)
        d = mean / sd
    else:
        t_stat, p_val, d = float("nan"), float("nan"), float("nan")
    return dict(n=n, mean=mean, sem=sem, sd=sd, t_stat=float(t_stat),
                p_value=float(p_val), cohen_d=float(d))


def run_p4(save_plot: bool = True, n_trials: int = N_TRIALS,
           t_total_ms: float = T_TOTAL_MS) -> dict:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    ref_ci_15s = _load_reference_ci_15s()

    wc = make_worm_circuit(scale=302, seed=0, **load_weight_scales())
    wl = WormLoop(wc)

    # ---- 梯度组（N=20）----
    print(f"梯度组：T={t_total_ms:.0f}ms × N={n_trials}（302 全连接组，D4 权重）")
    t0 = time.perf_counter()
    res_grad = wl.run_trials(n_trials=n_trials, seed_base=0,
                             t_total_ms=t_total_ms)
    wall_grad = time.perf_counter() - t0
    ci_grad = np.array([r.ci for r in res_grad], dtype=float)
    wall_trials = [wall_grad / n_trials] * n_trials  # 试次墙钟 = 均值（run_trials 无逐次计时）
    dist0 = np.array([r.meta["dist_start_food"] for r in res_grad], dtype=float)
    dist1 = np.array([r.meta["dist_end_food"] for r in res_grad], dtype=float)
    # 终态偏向（判据 c，informational）：接近食物比例
    approaching = float(np.mean(dist1 < dist0)) if dist0.size else float("nan")
    s_grad = _stats(ci_grad)
    print(f"  梯度 CĪ={s_grad['mean']:.3f}±{s_grad['sem']:.3f} "
          f"(p={s_grad['p_value']:.4f}, d={s_grad['cohen_d']:.2f}) "
          f"墙钟 {wall_grad:.0f}s（{wall_grad/n_trials:.0f}s/试次）")

    # ---- 预算纪律：单试次超 5min → 报主 agent ----
    max_trial_wall = float(np.max(wall_trials)) if wall_trials else float("nan")
    budget_ok = max_trial_wall <= BUDGET_TRIAL_WALL_S
    if not budget_ok:
        print(f"⚠ 预算纪律：单试次墙钟 {max_trial_wall:.0f}s > 5min——报主 agent")

    # ---- 无梯度对照（N=20，seed_base=1000）----
    print(f"对照组：N={n_trials}（C_max=0 → C≡C_bg）")
    t0 = time.perf_counter()
    res_ctrl = wl.run_control(n_trials=n_trials, seed_base=1000,
                              t_total_ms=t_total_ms)
    wall_ctrl = time.perf_counter() - t0
    ci_ctrl = np.array([r.ci for r in res_ctrl], dtype=float)
    s_ctrl = _stats(ci_ctrl)
    print(f"  对照 CĪ={s_ctrl['mean']:.3f}±{s_ctrl['sem']:.3f} "
          f"(p={s_ctrl['p_value']:.4f}) 墙钟 {wall_ctrl:.0f}s")

    # ---- 确定性抽查（同 seed_base 重跑 trial 0：同起点+同种子 → 逐位一致）----
    det_t0 = time.perf_counter()
    res_det = wl.run_trials(n_trials=1, seed_base=0, t_total_ms=t_total_ms)
    r_det = res_det[0]
    deterministic = bool(r_det.ci == res_grad[0].ci
                         and np.array_equal(r_det.x, res_grad[0].x)
                         and np.array_equal(r_det.y, res_grad[0].y))
    det_wall = time.perf_counter() - det_t0
    print(f"  确定性抽查（seed_base=0 重跑 trial0）: {deterministic}"
          f"（CI={r_det.ci:.4f} vs {res_grad[0].ci:.4f}，{det_wall:.0f}s）")

    # ---- 判定（主 agent 最终裁决 2026-08-26：P4 = fail + 反证记录）----
    # 预注册指标：显著性 p<0.05 且 d≥0.5；ΔCI≤0.15 或方向一致；对照 p>0.05。
    # 实测（T=15s×N=20 全协议，M5 定稿 WormLoop/VirtualBody）：CĪ=-0.065, p=0.71,
    # d=-0.08，方向负 → 预注册指标不满足。主 agent 最终裁决：趋化在 302 网络中确实
    # 不涌现（CI=-0.065），**如实判 fail（pass_=False）+ 反证记录**——M5 部分达标。
    # 反证记录内容（M6 优先验证清单）：根因 = fwd/back 运动池夹带共同发放 → v≈0
    # （L39/L40，与 P2/P6 同根因：夹带极限环 + 缺失调质/异质权重 + 命令互抑缺失）；
    # M4 前向身体对照（data/m5_p4_body_comparison.csv：+0.360 vs -0.407）仅作记录，
    # 不改变判据主体（换身体规避失败 = 不诚实）。
    sig_ok = bool(s_grad["p_value"] < 0.05 and s_grad["cohen_d"] >= 0.5)
    in_band = bool(CI_TOL[0] <= s_grad["mean"] <= CI_TOL[1])
    delta_ci = abs(s_grad["mean"] - ref_ci_15s)
    dir_ok = bool(s_grad["mean"] > 0.0)
    delta_ok = bool(delta_ci <= DELTA_CI_MAX)
    ctrl_ok = bool(s_ctrl["p_value"] > 0.05)
    finite_ok = bool(np.all(np.isfinite(ci_grad)) and np.all(np.isfinite(ci_ctrl)))

    indicator_pass = bool(finite_ok and sig_ok and (delta_ok or dir_ok) and ctrl_ok)
    # 主 agent 最终裁决 2026-08-26：P2/P4/P6 编码统一 pass_=False，
    # status=counter-evidence-record（反证记录：记录本身即科学交付物）；
    # P4 判据主体 = M5 定稿闭环（VirtualBody），负 CI 是真实网络行为。
    pass_ = False
    status = "counter-evidence-record"

    verdict = (
        "P4 趋化 = 反证记录（pass_=False, status=counter-evidence-record，主 agent "
        "最终裁决 2026-08-26）：302 网络趋化不涌现——全协议（T=15s×N=20，M5 定稿 "
        "WormLoop/VirtualBody 为判据主体）实测 CĪ="
        f"{s_grad['mean']:.3f}±{s_grad['sem']:.3f}（p={s_grad['p_value']:.3f}，"
        f"d={s_grad['cohen_d']:.2f}），ΔCI vs 参考(0.417@15s)="
        f"{delta_ci:.3f}；对照 CI={s_ctrl['mean']:+.3f}（p={s_ctrl['p_value']:.3f} "
        "正常）——负 CI 是真实网络行为（fwd/back 夹带共同发放 → v≈0，L39/L40）；"
        "→ 反证记录（夹带极限环 + 缺失调质/异质权重 + 命令互抑缺失 → M6 优先验证）；"
        "M4 前向身体对照（data/m5_p4_body_comparison.csv：CĪ +0.360 vs -0.407）"
        "仅作对照记录，不改变判据主体（换身体规避失败 = 不诚实）"
    )

    out = dict(
        pass_=pass_, status=status, verdict=verdict,
        indicator_pass=indicator_pass,
        n_trials=s_grad["n"], t_total_ms=t_total_ms,
        ci_grad=ci_grad.tolist(), ci_mean=s_grad["mean"],
        ci_sem=s_grad["sem"], ci_sd=s_grad["sd"],
        p_value=s_grad["p_value"], t_stat=s_grad["t_stat"],
        cohen_d=s_grad["cohen_d"],
        ci_ctrl=ci_ctrl.tolist(), ctrl_mean=s_ctrl["mean"],
        ctrl_sem=s_ctrl["sem"], ctrl_p=s_ctrl["p_value"],
        in_band=in_band, ci_tolerance=list(CI_TOL),
        reference_ci_15s=ref_ci_15s,
        delta_ci_vs_reference=delta_ci, delta_ci_max=DELTA_CI_MAX,
        direction_ok=dir_ok, direction="positive" if dir_ok else "negative/zero",
        significance_ok=sig_ok, control_ok=ctrl_ok, finite_ok=finite_ok,
        approaching_frac=approaching,
        dist_start_mean=float(np.mean(dist0)) if dist0.size else float("nan"),
        dist_end_mean=float(np.mean(dist1)) if dist1.size else float("nan"),
        deterministic=deterministic,
        budget=dict(
            max_trial_wall_s=max_trial_wall, budget_trial_wall_s=BUDGET_TRIAL_WALL_S,
            budget_ok=budget_ok,
            wall_gradient_s=wall_grad, wall_control_s=wall_ctrl,
            wall_determinism_s=det_wall,
            total_wall_s=wall_grad + wall_ctrl + det_wall,
            note="预注册预算 ≤200 CPU-h；本协议实测 "
                 f"{wall_grad + wall_ctrl + det_wall:.0f}s ≈ "
                 f"{(wall_grad + wall_ctrl + det_wall) / 3600:.2f} CPU-h（单机），远低于上限"),
        weights="D4 定稿（load_weight_scales：gap_scale=0.05，类级=先验 1.0）",
        protocol_source="data/m5_worm_params.csv protocol 行（T=15000/N=20/start 5.0±0.3）",
        diagnosis=dict(
            root_cause=(
                "D4 权重下 fwd/back 运动池共同发放（与 P2/P6 夹带同根，L39/L40）→ "
                "M5 VirtualBody v = C_fwd − C_back ≈ 0 → 15s 位移仅 0.2-0.5 皿单位"
                "（pause/turn 主导）→ 虫停留起点圈外 → CI≈0 或负——非统计噪声，"
                "是夹带病理的直接表现"),
            protocol_semantics=dict(
                note=("B1e2 校准/G0 缩放的 CI@5s（g1=0.465、L39 +0.078）用 "
                      "WormCircuit.run_chemotaxis_trials（M4 ChemotaxisBody 前向身体，"
                      "忽略 C_back）；本全协议用 M5 定稿 WormLoop/VirtualBody（含后退 "
                      "通道，virtual_body.py §5.2 #3）"),
                m4_body_ci_mean_6="+0.360（同种子 N=6，位移 3.9-6.3）",
                m5_body_ci_mean_6="-0.407（同种子 N=6，位移 0.2-0.5）",
                evidence="tools/validate_p4_chemotaxis.run_p4_body_comparison"
                         "（data/m5_p4_body_comparison.csv，2026-08-26）",
                adjudication=(
                    "主 agent 最终裁决 2026-08-26（三态选项 ①）：M5 定稿闭环（VirtualBody）是"
                    "设计最终身体、P4 判据主体 → 负 CI 是真实网络行为；编码统一 "
                    "pass_=False + status=counter-evidence-record（反证记录：记录本身即"
                    "科学交付物，与 P2/P6 同型）；M4 前向身体语义仅作对照记录，不改变判据"
                    "主体（换身体规避失败 = 不诚实）；M6 命令互抑/调质后复核。"),
            ),
            diagnosis_evidence=(
                "/tmp/m5_b2_p4diag.py：6 试次位移 [0.36,0.22,0.45,0.19,0.30,0.35] 皿单位，"
                "state_frac fwd~20%/pause 56-71%/turn 0-84%"),
        ),
    )

    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        import csv as _csv
        w = _csv.writer(f, lineterminator="\n")
        w.writerow(["# M5 P4 趋化验证（tools/validate_p4_chemotaxis.py；T=15s×N=20 全协议）"])
        w.writerow(["metric", "value", "criterion", "verdict"])
        w.writerow(["pass_", out["pass_"],
                    "False（反证记录：记录本身即科学交付物，主 agent 最终裁决）",
                    "record"])
        w.writerow(["status", out["status"], "counter-evidence-record", "ok"])
        w.writerow(["indicator_pass", out["indicator_pass"],
                    "预注册指标（p<0.05 且 d≥0.5 且方向一致/ΔCI≤0.15）",
                    "ok" if indicator_pass else "OUT（→ 反证记录）"])
        w.writerow(["n_trials", out["n_trials"], "20", "ok"])
        w.writerow(["t_total_ms", out["t_total_ms"], "15000", "ok"])
        w.writerow(["ci_mean", f"{out['ci_mean']:.4f}", ">0", "ok" if dir_ok else "OUT"])
        w.writerow(["ci_sem", f"{out['ci_sem']:.4f}", "", ""])
        w.writerow(["p_value", f"{out['p_value']:.4f}", "<0.05", "ok" if sig_ok else "OUT"])
        w.writerow(["cohen_d", f"{out['cohen_d']:.4f}", ">=0.5", "ok" if sig_ok else "OUT"])
        w.writerow(["in_tolerance_band", out["in_band"], "[0.25,0.75]", "informational"])
        w.writerow(["delta_ci_vs_reference_15s", f"{out['delta_ci_vs_reference']:.4f}",
                    f"<=0.15 或方向一致", "ok" if (delta_ok or dir_ok) else "OUT"])
        w.writerow(["reference_ci_15s", f"{out['reference_ci_15s']:.4f}",
                    "m4_calibration ref-T15000", "ok"])
        w.writerow(["ctrl_mean", f"{out['ctrl_mean']:.4f}", "|CI|<0.1 informational", ""])
        w.writerow(["ctrl_p", f"{out['ctrl_p']:.4f}", ">0.05", "ok" if ctrl_ok else "OUT"])
        w.writerow(["approaching_frac", f"{out['approaching_frac']:.4f}", "informational", ""])
        w.writerow(["deterministic", out["deterministic"], "true", "ok"])
        w.writerow(["budget_ok", out["budget"]["budget_ok"], "<=300s/trial", "ok"])
        w.writerow(["max_trial_wall_s", f"{out['budget']['max_trial_wall_s']:.1f}",
                    "<=300", "ok" if budget_ok else "REPORT"])
        w.writerow(["total_wall_s", f"{out['budget']['total_wall_s']:.0f}", "", ""])
        w.writerow(["verdict", out["verdict"], "", ""])

    if save_plot:
        _plot(out)

    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        import json as _json
        _json.dump(out, f, ensure_ascii=False, default=str)

    return out


def _plot(out: dict):
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

    ci_g = np.asarray(out["ci_grad"])
    ci_c = np.asarray(out["ci_ctrl"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

    ax = axes[0]
    bins = np.linspace(-1.2, 1.2, 25)
    ax.hist(ci_g, bins=bins, alpha=0.65, color="tab:blue", label="梯度组")
    ax.hist(ci_c, bins=bins, alpha=0.65, color="tab:orange", label="无梯度对照")
    ax.axvline(out["ci_mean"], color="blue", ls="--", lw=1.4,
               label=f"CĪ={out['ci_mean']:.3f}")
    ax.axvline(out["reference_ci_15s"], color="red", ls=":", lw=1.4,
               label=f"参考 {out['reference_ci_15s']:.3f}")
    ax.axvspan(0.25, 0.75, color="green", alpha=0.10, label="容差带 [0.25,0.75]")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("CI")
    ax.set_ylabel("试次数")
    ax.set_title(f"P4 全协议（T=15s×N=20）：CĪ={out['ci_mean']:.3f}±"
                 f"{out['ci_sem']:.3f}，p={out['p_value']:.4f}，"
                 f"d={out['cohen_d']:.2f}")
    ax.legend(fontsize=8)

    ax = axes[1]
    trials = np.arange(out["n_trials"])
    ax.errorbar(trials, ci_g, fmt="o", ms=5, color="tab:blue",
                label="梯度试次")
    ax.axhline(out["ci_mean"], color="blue", ls="--", lw=1.2)
    ax.fill_between(trials, out["ci_mean"] - out["ci_sem"],
                    out["ci_mean"] + out["ci_sem"], color="blue", alpha=0.15,
                    label="±SEM")
    ax.axhline(out["reference_ci_15s"], color="red", ls=":", lw=1.2,
               label=f"参考 {out['reference_ci_15s']:.3f}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("试次")
    ax.set_ylabel("CI")
    ax.set_title(f"逐试次 CI（ΔCI vs 参考={out['delta_ci_vs_reference']:.3f}，"
                 f"方向 {'positive ✓' if out['direction_ok'] else '✗'}）")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(REPORT_PNG, dpi=110)
    plt.close(fig)


def run_p4_body_comparison(n_trials: int = 6, t_total_ms: float = T_TOTAL_MS,
                           save_csv: bool = True) -> dict:
    """M4 前向身体 vs M5 定稿 VirtualBody 同种子对照（主 agent 裁决关键证据）。

    - M4 语义：`WormCircuit.run_chemotaxis_trials`（M4 ChemotaxisBody，v≥0，忽略 C_back）
      ——B1e2 校准/G0 缩放的 CI@5s 测量路径；
    - M5 语义：`WormLoop.run_trials`（VirtualBody，v = v_fwd0·C_fwd − v_rev0·C_back）
      ——M5 定稿闭环（virtual_body.py §5.2 #3），P4 判据主体。
    同种子（seed_base=0）→ 逐试次 CI + 位移对照，量化"纯身体语义 vs 网络夹带"来源。
    输出：data/m5_p4_body_comparison.csv
    """
    import csv as _csv

    ws = load_weight_scales()
    wc = make_worm_circuit(scale=302, seed=0, **ws)
    wl = WormLoop(wc)

    res_a, meta_a = wc.run_chemotaxis_trials(n_trials=n_trials, seed_base=0,
                                             t_total_ms=t_total_ms)
    res_b = wl.run_trials(n_trials=n_trials, seed_base=0, t_total_ms=t_total_ms)
    ci_a = np.array([r.ci for r in res_a], dtype=float)
    ci_b = np.array([r.ci for r in res_b], dtype=float)
    disp_a = np.array([np.hypot(r.x[-1] - r.x[0], r.y[-1] - r.y[0])
                       for r in res_a], dtype=float)
    disp_b = np.array([np.hypot(r.x[-1] - r.x[0], r.y[-1] - r.y[0])
                       for r in res_b], dtype=float)
    # M4 body（ChemotaxisResult）无 state_frac；M5 body 有
    frac_b = [r.meta.get("state_frac", {}) for r in res_b]

    out = dict(
        n_trials=n_trials, t_total_ms=t_total_ms,
        m4_body=dict(ci=ci_a.tolist(), ci_mean=float(ci_a.mean()),
                     displacement=disp_a.tolist(),
                     disp_mean=float(disp_a.mean())),
        m5_body=dict(ci=ci_b.tolist(), ci_mean=float(ci_b.mean()),
                     displacement=disp_b.tolist(),
                     disp_mean=float(disp_b.mean())),
        delta_ci_mean=float(ci_a.mean() - ci_b.mean()),
        conclusion=(
            "M4 前向身体（忽略 C_back）同种子 CĪ="
            f"{ci_a.mean():+.3f}（位移 {disp_a.mean():.2f}/15s）vs M5 定稿 "
            f"VirtualBody CĪ={ci_b.mean():+.3f}（位移 {disp_b.mean():.2f}/15s）"
            "——差异来源 = 身体语义（M5 含后退通道：fwd/back 运动池夹带共同发放 → "
            "v≈0 → 无净趋化位移，L39/L40），非统计噪声；M4 语义仅作对照记录，"
            "不改变 P4 判据主体（主 agent 裁决 2026-08-26）"),
    )
    if save_csv:
        path = os.path.join(DATA_DIR, "m5_p4_body_comparison.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f, lineterminator="\n")
            w.writerow(["# M5 P4 身体语义对照（validate_p4_chemotaxis.run_p4_body_comparison；"
                        "同种子 N=6，T=15s）"])
            w.writerow(["metric", "value", "note"])
            for k, v in out["m4_body"].items():
                w.writerow([f"m4_body_{k}", v, "M4 ChemotaxisBody（前向，忽略 C_back）"])
            for k, v in out["m5_body"].items():
                w.writerow([f"m5_body_{k}", v, "M5 VirtualBody（含后退通道，判据主体）"])
            w.writerow(["delta_ci_mean", out["delta_ci_mean"], ""])
            w.writerow(["conclusion", out["conclusion"], ""])
    return out


def main():
    out = run_p4(save_plot=True)
    print(f"P4 pass_ = {out['pass_']}（{out['status']}）")
    print(f"  CĪ={out['ci_mean']:.3f}±{out['ci_sem']:.3f} "
          f"(p={out['p_value']:.4f}, d={out['cohen_d']:.2f})")
    print(f"  ΔCI vs 参考(15s={out['reference_ci_15s']:.3f}) = "
          f"{out['delta_ci_vs_reference']:.3f}（≤0.15 或方向一致）")
    print(f"  对照 CĪ={out['ctrl_mean']:.3f} (p={out['ctrl_p']:.4f})")
    print(f"  预算: max 单试次 {out['budget']['max_trial_wall_s']:.0f}s，"
          f"总 {out['budget']['total_wall_s']:.0f}s ≈ "
          f"{out['budget']['total_wall_s']/3600:.2f} CPU-h")
    print(f"  {REPORT_CSV}\n  {REPORT_PNG}")


if __name__ == "__main__":
    main()
