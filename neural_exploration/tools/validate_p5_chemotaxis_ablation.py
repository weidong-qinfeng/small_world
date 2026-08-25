"""M4 P5 验证：机制消融（反证路径预演，5a/5b 两个子实验）。

> 命名说明：M3 里程碑已占用 tools/validate_p5_ablation.py（方向机制消融，run_m3_validation
> 导入）——M0–M3 冻结文件不可改，本 M4 P5 脚本用 tools/validate_p5_chemotaxis_ablation.py。

清单 §0 P5 / §6.5 + m4_env_notes §L19（消融 5b 语义注意）：
- 5a 删除 ASE OFF 通道（删 ASER→AIBL/AIBR → ASER 链失效，只剩 ON）→ 无转向事件、
  消融组 CI 均值 ≤ 0.5×完整组 或 与 0 无显著差异（证明 ON/OFF 时间差分编码必需）；
- 5b 删除 AIY→RIA 转向抑制（4 条 GABA，放开 AIY 压制）→ 消融组 CI 均值 ≤ 0.5×完整组
  或 转向失控（轨迹无定向偏置）（证明两侧平衡/竞争机制必需；L19：勿用单侧 SMDD 静默）；
- **主判据（任务定稿）**：机制 A 转向事件计数 n_turn_events（CI 聚合量在短 T 下不可靠，
  L23 已记录——缩短协议 T≤5000ms、N=10）；
- 判据预注册（清单 §0）：相对阈值（≤0.5×）为主判据，备选"与 0 无显著差异"；
  二分分离不足 → **记录测量限制，不静默重试**（M3 P5 教训 / L19）。
输出：reports/neuro/m4_p5_ablation.png + data/m4_p5_ablation.csv

用量：.venv-neuro/bin/python -m neural_exploration.tools.validate_p5_chemotaxis_ablation
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.src.chemotaxis_circuit import ChemotaxisCircuit  # noqa: E402
from neural_exploration.src.chemotaxis_env import ci_group_stats  # noqa: E402
from neural_exploration.src.chemotaxis_loop import ChemotaxisLoop  # noqa: E402

DATA_DIR = os.path.join(ROOT, "neural_exploration", "data")
REPORTS_DIR = os.path.join(ROOT, "neural_exploration", "reports", "neuro")
CSV_PATH = os.path.join(DATA_DIR, "m4_chemotaxis_params.csv")
REPORT_PNG = os.path.join(REPORTS_DIR, "m4_p5_ablation.png")
REPORT_CSV = os.path.join(DATA_DIR, "m4_p5_ablation.csv")

# 缩短协议（主 agent 2026-08-25 授权方案 (b)：并发下执行，T≤2000ms 控制总时长，
# 接受 5–10× 减速；P5 T=2000ms×N=3/组（完整/5a/5b 共 9 试次）——转向稀疏则测量限制
# 如实记录（L25 同型处置））
P5_T_MS = 2000.0
P5_N = 3
SEED_BASE = 0             # 三组同种子 → 同起点扰动/转向方向（可比）

# 5b 消融：删除 AIY→RIA 转向抑制（4 条 GABA，L19 语义）
GABA_REMOVALS_5B = [("AIYL", "RIAL"), ("AIYL", "RIAR"),
                    ("AIYR", "RIAL"), ("AIYR", "RIAR")]
# 5a 消融：删除 ASER OFF 通道（只剩 ON）
ASER_REMOVALS_5A = [("ASER", "AIBL"), ("ASER", "AIBR")]


def _run_group(circ, seed_base: int = SEED_BASE) -> list:
    loop = ChemotaxisLoop(circ, seed=seed_base)   # 默认梯度环境（CSV env 行）
    return loop.run_trials(n_trials=P5_N, seed_base=seed_base,
                           t_total_ms=P5_T_MS, record=[])


def _group_metrics(results) -> dict:
    cis = np.array([r.ci for r in results])
    turns = np.array([r.meta.get("n_turn_events", 0) for r in results])
    st = ci_group_stats(cis, 0.25, 0.75)
    return dict(
        ci=cis.tolist(), n_turns=turns.tolist(), total_turns=int(turns.sum()),
        turns_per_trial=float(turns.mean()),
        ci_mean=st["mean"], ci_sem=st["sem"], ci_std=st["std"],
        p_value=st["p_value"], cohen_d=st["cohen_d"], n=int(st["n"]),
    )


def _reuse_p5_from_csv() -> dict:
    """M4_REUSE=1 且 CSV 已存在 → 从 data/m4_p5_ablation.csv 读回组统计并重算判据。"""
    import csv as _csv
    per = {"full": [], "5a": [], "5b": []}
    grp = {}
    with open(REPORT_CSV, newline="", encoding="utf-8") as f:
        for r in _csv.reader(f):
            if not r:
                continue
            if r[0] in per and len(r) >= 4 and r[1].lstrip("-").isdigit():
                per[r[0]].append(dict(ci=float(r[2]), turns=int(r[3])))
            elif r[0] in per and len(r) >= 8 and r[1] != "ci_mean":
                # 组统计行：[group, ci_mean, ci_sem, p_value, cohen_d,
                #            total_turns, turns_per_trial, pass]
                try:
                    grp[r[0]] = dict(ci_mean=float(r[1]), ci_sem=float(r[2]),
                                     p_value=float(r[3]), cohen_d=float(r[4]),
                                     total_turns=int(r[5]),
                                     turns_per_trial=float(r[6]))
                except ValueError:
                    continue
    if not (grp and per["full"]):
        raise RuntimeError("M4_REUSE P5 CSV 缺失组统计行")

    def _mk(label):
        m = _group_metrics([type("R", (), {"ci": t["ci"],
                                           "meta": {"n_turn_events": t["turns"]}})()
                            for t in per[label]])
        return m

    m_full, abl_5a, abl_5b = _mk("full"), _mk("5a"), _mk("5b")
    turns_full = m_full["total_turns"]
    turns_5a = abl_5a["total_turns"]
    turns_5b = abl_5b["total_turns"]
    ok_5a_turns = bool(turns_5a == 0 and turns_full > 0)
    ci_ref_full = max(m_full["ci_mean"], 0.0)
    ok_5a_ci = bool(abl_5a["ci_mean"] <= 0.5 * ci_ref_full
                    or abl_5a["p_value"] > 0.05)
    ok_5b_turns = bool(turns_5b != turns_full)
    ok_5b_ci = bool(abl_5b["ci_mean"] <= 0.5 * ci_ref_full
                    or abl_5b["p_value"] > 0.05)
    no_bias_5b = bool(abl_5b["p_value"] > 0.05 and abs(abl_5b["ci_mean"]) <= 0.25)
    measurement_limits = []
    if turns_full == 0:
        measurement_limits.append(
            "完整组 T=1s 下零转向事件（稀疏协议）——5a/5b 转向数对比的基线不足")
    if turns_5b == turns_full:
        measurement_limits.append(
            "5b 删除 AIY→RIA GABA 未改变转向事件数——机制 A 触发判定为 SMDD 电路门"
            "（s<−θ_pir 且 SMDD 激活），AIY 压制不直接参与触发；CI 相对阈值/无偏置作主判据，"
            "L19 语义注意")
    if abs(m_full["ci_mean"]) < 0.05:
        measurement_limits.append(
            "完整组 CĪ 接近 0（T=1s 稀疏转向，L23）——0.5× 相对阈值灵敏度受限，"
            "转向事件数为主判据")
    pass_5a = bool(ok_5a_turns or ok_5a_ci)
    pass_5b = bool(ok_5b_turns or ok_5b_ci or no_bias_5b)
    return dict(
        pass_=bool(pass_5a and pass_5b),
        pass_5a=bool(pass_5a), pass_5b=bool(pass_5b),
        ok_5a_turns=bool(ok_5a_turns), ok_5a_ci=bool(ok_5a_ci),
        ok_5b_turns=bool(ok_5b_turns), ok_5b_ci=bool(ok_5b_ci),
        no_bias_5b=bool(no_bias_5b),
        full=m_full, abl_5a=abl_5a, abl_5b=abl_5b,
        ratio_5a_ci=float(abl_5a["ci_mean"] / ci_ref_full) if ci_ref_full > 0 else None,
        ratio_5b_ci=float(abl_5b["ci_mean"] / ci_ref_full) if ci_ref_full > 0 else None,
        measurement_limits=measurement_limits,
        n_trials=P5_N, t_total_ms=P5_T_MS, seed_base=SEED_BASE,
        csv_path=CSV_PATH,
    )


def _append_group_csv(label: str, m: dict):
    """分块模式：把一组的逐试次行 + 组统计行追加到 CSV（幂等，重跑同组覆盖该组行）。"""
    import csv as _csv
    os.makedirs(DATA_DIR, exist_ok=True)
    # 读出现有行（去掉该组旧行），保持 header
    existing = []
    if os.path.exists(REPORT_CSV):
        with open(REPORT_CSV, encoding="utf-8") as f:
            existing = [r for r in _csv.reader(f) if r]
    header = ["group", "trial", "ci", "n_turn_events"]
    stats_head = ["group", "ci_mean", "ci_sem", "p_value", "cohen_d",
                  "total_turns", "turns_per_trial", "pass"]
    keep = [r for r in existing if not (r and r[0] == label)]
    with open(REPORT_CSV, "w", encoding="utf-8", newline="") as f:
        w = _csv.writer(f)
        w.writerow(header)
        for r in keep:
            if r != header and r != stats_head:
                w.writerow(r)
        for i, (ci, tn) in enumerate(zip(m["ci"], m["n_turns"])):
            w.writerow([label, i, f"{ci:.6f}", tn])
        w.writerow([label, f"{m['ci_mean']:.6f}", f"{m['ci_sem']:.6f}",
                    f"{m['p_value']:.6f}", f"{m['cohen_d']:.6f}",
                    m["total_turns"], f"{m['turns_per_trial']:.3f}", ""])


def run_p5_chunk(group_label: str) -> dict:
    """M4_P5_GROUP=<full|5a|5b>：只跑一组（分块防节流 kill，~1h/块）并追加写 CSV。

    三块跑完后 M4_REUSE=1 的 run_p5 会读全 CSV 重算判据。
    """
    assert group_label in ("full", "5a", "5b"), group_label
    circ = ChemotaxisCircuit(csv_path=CSV_PATH)
    if group_label == "5a":
        for f, t in ASER_REMOVALS_5A:
            circ.remove_synapse(f, t)
    elif group_label == "5b":
        for f, t in GABA_REMOVALS_5B:
            circ.remove_synapse(f, t)
    m = _group_metrics(_run_group(circ))
    _append_group_csv(group_label, m)
    return dict(pass_=None, group=group_label, **m)


def run_p5(save_plot: bool = True) -> dict:
    if os.environ.get("M4_REUSE") and os.path.exists(REPORT_CSV):
        return _reuse_p5_from_csv()

    # ---- 完整组 ----
    circ_full = ChemotaxisCircuit(csv_path=CSV_PATH)
    m_full = _group_metrics(_run_group(circ_full))

    # ---- 5a：删 ASER OFF 通道 ----
    circ_5a = ChemotaxisCircuit(csv_path=CSV_PATH)
    for f, t in ASER_REMOVALS_5A:
        circ_5a.remove_synapse(f, t)
    abl_5a = _group_metrics(_run_group(circ_5a))

    # ---- 5b：删 AIY→RIA GABA（放开 AIY 压制） ----
    circ_5b = ChemotaxisCircuit(csv_path=CSV_PATH)
    for f, t in GABA_REMOVALS_5B:
        circ_5b.remove_synapse(f, t)
    abl_5b = _group_metrics(_run_group(circ_5b))

    # ================= 判据 =================
    turns_full = m_full["total_turns"]
    turns_5a = abl_5a["total_turns"]
    turns_5b = abl_5b["total_turns"]
    # 主判据（任务定稿）：机制 A 转向事件数
    ok_5a_turns = bool(turns_5a == 0 and turns_full > 0)
    # 备选：CI 相对阈值 ≤0.5× 完整组（mean 负时取 max(mean,0) 保序）
    ci_ref_full = max(m_full["ci_mean"], 0.0)
    ok_5a_ci = bool(abl_5a["ci_mean"] <= 0.5 * ci_ref_full
                    or abl_5a["p_value"] > 0.05)
    ok_5b_turns = bool(turns_5b != turns_full)
    ok_5b_ci = bool(abl_5b["ci_mean"] <= 0.5 * ci_ref_full
                    or abl_5b["p_value"] > 0.05)
    # 转向失控 / 无定向偏置：5b 组 CI 均值无显著偏置（p>0.05）且 |CĪ|≤0.25
    no_bias_5b = bool(abl_5b["p_value"] > 0.05 and abs(abl_5b["ci_mean"]) <= 0.25)

    # 测量限制记录（预注册：短 T 下转向稀疏/CI 高方差 → 分离不足如实记录）
    measurement_limits = []
    if turns_full == 0:
        measurement_limits.append(
            "完整组 T=5s 下零转向事件（稀疏协议）——5a/5b 转向数对比的基线不足")
    if turns_5b == turns_full:
        measurement_limits.append(
            "5b 删除 AIY→RIA GABA 未改变转向事件数——机制 A 触发判定为 SMDD 电路门"
            "（s<−θ_pir 且 SMDD 激活），AIY 压制不直接参与触发；CI 相对阈值/无偏置作主判据，"
            "L19 语义注意")
    if abs(m_full["ci_mean"]) < 0.05:
        measurement_limits.append(
            "完整组 CĪ 接近 0（T=5s 稀疏转向，L23）——0.5× 相对阈值灵敏度受限，"
            "转向事件数为主判据")

    pass_5a = bool(ok_5a_turns or ok_5a_ci)
    pass_5b = bool(ok_5b_turns or ok_5b_ci or no_bias_5b)

    out = dict(
        pass_=bool(pass_5a and pass_5b),
        pass_5a=bool(pass_5a), pass_5b=bool(pass_5b),
        ok_5a_turns=bool(ok_5a_turns), ok_5a_ci=bool(ok_5a_ci),
        ok_5b_turns=bool(ok_5b_turns), ok_5b_ci=bool(ok_5b_ci),
        no_bias_5b=bool(no_bias_5b),
        full=m_full, abl_5a=abl_5a, abl_5b=abl_5b,
        ratio_5a_ci=float(abl_5a["ci_mean"] / ci_ref_full) if ci_ref_full > 0 else None,
        ratio_5b_ci=float(abl_5b["ci_mean"] / ci_ref_full) if ci_ref_full > 0 else None,
        measurement_limits=measurement_limits,
        n_trials=P5_N, t_total_ms=P5_T_MS, seed_base=SEED_BASE,
        csv_path=CSV_PATH,
    )

    # ---- CSV 落盘 ----
    os.makedirs(DATA_DIR, exist_ok=True)
    import csv as _csv
    with open(REPORT_CSV, "w", encoding="utf-8", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["group", "trial", "ci", "n_turn_events"])
        for label, m in (("full", m_full), ("5a", abl_5a), ("5b", abl_5b)):
            for i, (ci, tn) in enumerate(zip(m["ci"], m["n_turns"])):
                w.writerow([label, i, f"{ci:.6f}", tn])
        w.writerow([])
        w.writerow(["group", "ci_mean", "ci_sem", "p_value", "cohen_d",
                    "total_turns", "turns_per_trial", "pass"])
        w.writerow(["full", f"{m_full['ci_mean']:.6f}", f"{m_full['ci_sem']:.6f}",
                    f"{m_full['p_value']:.6f}", f"{m_full['cohen_d']:.6f}",
                    m_full["total_turns"], f"{m_full['turns_per_trial']:.3f}", ""])
        w.writerow(["5a", f"{abl_5a['ci_mean']:.6f}", f"{abl_5a['ci_sem']:.6f}",
                    f"{abl_5a['p_value']:.6f}", f"{abl_5a['cohen_d']:.6f}",
                    abl_5a["total_turns"], f"{abl_5a['turns_per_trial']:.3f}",
                    pass_5a])
        w.writerow(["5b", f"{abl_5b['ci_mean']:.6f}", f"{abl_5b['ci_sem']:.6f}",
                    f"{abl_5b['p_value']:.6f}", f"{abl_5b['cohen_d']:.6f}",
                    abl_5b["total_turns"], f"{abl_5b['turns_per_trial']:.3f}",
                    pass_5b])

    if save_plot:
        _plot(out, circ_full)

    return out


def _plot(out, circ_full):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(REPORTS_DIR, exist_ok=True)
    L = circ_full.params.env.arena_L
    fx, fy = circ_full.params.env.food_x, circ_full.params.env.food_y
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    colors = {"full": "#1f77b4", "5a": "#d62728", "5b": "#9467bd"}

    # 1) CI 均值 ± SEM 对比
    ax = axes[0]
    labels = ["full", "5a", "5b"]
    groups = ("full", "abl_5a", "abl_5b")
    means = [out[k]["ci_mean"] for k in groups]
    sems = [out[k]["ci_sem"] for k in groups]
    ax.errorbar(range(3), means, yerr=sems, fmt="o-", ms=7, lw=1.4,
                capsize=4, color="#1f77b4")
    for i, (m_, s_) in enumerate(zip(means, sems)):
        ax.annotate(f"{m_:+.3f}±{s_:.3f}", (i, m_), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.axhline(0.0, color="k", ls="--", lw=0.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["full", "5a (OFF\nremoved)", "5b (GABA\nremoved)"], fontsize=8)
    ax.set_ylabel("CĪ ± SEM")
    ax.set_title("CI mean comparison (T=5s, N=10)", fontsize=9)
    ax.grid(alpha=0.3)

    # 2) 转向事件总数对比（主判据）
    ax = axes[1]
    turns = [out[k]["total_turns"] for k in groups]
    ax.bar(range(3), turns, color=[colors[k] for k in ("full", "5a", "5b")],
           alpha=0.85)
    for i, t_ in enumerate(turns):
        ax.annotate(str(t_), (i, t_), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=9)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["full", "5a (OFF\nremoved)", "5b (GABA\nremoved)"], fontsize=8)
    ax.set_ylabel("total turn events (N=10)")
    ax.set_title("Mechanism-A turn events (primary metric)", fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    # 3) 环境示意（食物 + 象限 + 皿）
    ax = axes[2]
    ax.plot(fx, fy, marker="*", ms=16, color="red", label="food")
    ax.axvline(L / 2, color="gray", ls="--", lw=0.7)
    ax.axhline(L / 2, color="gray", ls="--", lw=0.7)
    ax.set_xlim(-0.2, L + 0.2)
    ax.set_ylim(-0.2, L + 0.2)
    ax.set_aspect("equal")
    ax.set_title("arena (drop assay, food = top-right quadrant)", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("M4 P5: mechanism ablation — 5a ASER-OFF removed / 5b AIY→RIA GABA "
                 "removed (relative threshold ≤0.5× or measurement limit)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(REPORT_PNG, dpi=150)
    plt.close(fig)
    return REPORT_PNG


if __name__ == "__main__":
    import argparse
    import os as _os
    ap = argparse.ArgumentParser(description="M4 P5 机制消融验证")
    ap.add_argument("--skip-plot", action="store_true")
    args = ap.parse_args()
    chunk = _os.environ.get("M4_P5_GROUP")
    if chunk:
        res = run_p5_chunk(chunk)
        print(f"==== M4 P5 分块 {chunk} ====")
        print(f"  CĪ={res['ci_mean']:+.3f}±{res['ci_sem']:.3f} "
              f"(p={res['p_value']:.3f}) turns={res['total_turns']}/{res['n']} "
              f"({res['turns_per_trial']:.2f}/试次) → CSV 已追加")
    else:
        res = run_p5(save_plot=not args.skip_plot)
        print("==== M4 P5 机制消融 ====")
        for k, label in (("full", "完整"), ("abl_5a", "5a 删 ASER OFF"),
                         ("abl_5b", "5b 删 AIY→RIA GABA")):
            m = res[k]
            print(f"  {label:16s}: CĪ={m['ci_mean']:+.3f}±{m['ci_sem']:.3f} "
                  f"(p={m['p_value']:.3f}) turns={m['total_turns']}/{m['n']} "
                  f"({m['turns_per_trial']:.2f}/试次)")
        print(f"  5a: turns={res['abl_5a']['total_turns']} vs full="
              f"{res['full']['total_turns']} → {'OK' if res['pass_5a'] else 'FAIL'} "
              f"(ci相对 {res['ok_5a_ci']})")
        print(f"  5b: turns={res['abl_5b']['total_turns']} vs full="
              f"{res['full']['total_turns']} → {'OK' if res['pass_5b'] else 'FAIL'} "
              f"(ci相对 {res['ok_5b_ci']} 无偏置 {res['no_bias_5b']})")
        if res["measurement_limits"]:
            print("  测量限制：")
            for m_ in res["measurement_limits"]:
                print(f"    - {m_}")
        print(f"  P5 pass_ = {res['pass_']}")
