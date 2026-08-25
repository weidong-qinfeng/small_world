"""M4 逐项验证总运行器（清单 §7 验收：P1–P6 顺序运行 + P4 协议限制反证记录）。

依次运行 P1/P2/P3/P5/P6 五个验证（含图/CSV 落盘），P4 按主 agent 裁决
（2026-08-24，L23）**不重跑**——用 data/m4_calibration.csv（点 0 + ref-T 缩放行）+
docs/m4_env_notes.md L23 记录组装协议限制反证记录；汇总判定写入
  reports/neuro/m4_validation_summary.json（结构沿用 run_m3_validation.py）。
P7（pytest 全绿 + 报告写盘）由全量 pytest + tools/gen_m4_report.py 独立确认。

用法：
  .venv-neuro/bin/python -m neural_exploration.tools.run_m4_validation
可选：
  --skip-p3p5  跳过重协议（P3/P5 Brian2 闭环，计算受限时快速回归其余判据）
  --skip-plot  不出图（纯判定）
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.tools import (  # noqa: E402
    validate_p1_ase_encoding,
    validate_p2_circuit,
    validate_p3_env_control,
    validate_p5_chemotaxis_ablation,
    validate_p6_reference,
)

REPORTS_DIR = os.path.join(ROOT, "neural_exploration", "reports", "neuro")
DATA_DIR = os.path.join(ROOT, "neural_exploration", "data")
SUMMARY_JSON = os.path.join(REPORTS_DIR, "m4_validation_summary.json")
CAL_CSV = os.path.join(DATA_DIR, "m4_calibration.csv")


def _load_cal_rows() -> dict:
    rows = {}
    with open(CAL_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("ci_mean") not in (None, ""):
                rows[r["point_id"]] = r
    return rows


def _load_decisive_evidence() -> dict:
    """决定性点完整证据（/tmp/m4_res/point_4/trial_*.json，N=20 全部完成）。

    θ_pir=1e-6, T=10s, v_fwd0=1.0, g=8e6（定稿参数，L23）：CĪ=0.099±0.139（p=0.482,
    d=0.16）不显著；ΔCI vs 参考(10s)=0.317 → 0.217 > 0.15——确证 L23 协议限制裁决
    （可行协议下统计功效结构性不足，非机制失败）。早期快照（3/20）"全正向"印象被
    N=20 全数据修正（试次 10–19 多负值/近零）。
    """
    import glob
    from scipy import stats as sps
    files = sorted(glob.glob("/tmp/m4_res/point_4/trial_*.json"))
    cis = []
    for fp in files:
        try:
            with open(fp, encoding="utf-8") as f:
                cis.append(float(json.load(f)["ci"]))
        except Exception:
            continue
    if not cis:
        return dict(ci_values=None, n=0, ci_str="无（扫描未完成试次）",
                    mean_str="—", sem=None, p_value=None, cohen_d=None,
                    delta_ci_10s=None)
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
    # 参考模型 CI(10s)（calibration ref-T10000 行）→ ΔCI
    delta = None
    cal = _load_cal_rows()
    t10 = cal.get("ref-T10000")
    if t10:
        delta = abs(mean - float(t10["ci_mean"]))
    return dict(
        ci_values=[round(x, 4) for x in cis],
        n=n, ci_str=", ".join(f"{x:.2f}" for x in cis),
        mean_str=f"{mean:.3f}", sem=float(sem),
        p_value=float(p_val), cohen_d=float(d),
        delta_ci_10s=float(delta) if delta is not None else None,
        note=("决定性点（T=10s, v=1.0, g=8e6, θ_pir=1e-6, N=20）全部完成：CĪ=%.3f±%.3f "
              "(p=%.3f, d=%.2f)——不显著，确证 L23 协议限制裁决；ΔCI vs 参考(10s)=0.317 "
              "→ %.3f（>0.15 记录为测量限制，L21 点 7 处置：不静默推进，反证笔记；"
              "L23 已裁决为协议限制反证）" % (mean, sem, p_val, d,
                                          float(delta) if delta is not None else float("nan"))),
    )


def _load_point5_partial() -> dict:
    """点 5（θ_pir=2e-6, T=10s, v=1.0, g=8e6）部分结果（N=9/20，扫描终止前落盘）。"""
    import glob
    files = sorted(glob.glob("/tmp/m4_res/point_5/trial_*.json"))
    cis = []
    for fp in files:
        try:
            with open(fp, encoding="utf-8") as f:
                cis.append(float(json.load(f)["ci"]))
        except Exception:
            continue
    if not cis:
        return dict(ci_values=[], n=0, mean=None, p_value=None, mean_str="—")
    c = np.asarray(cis, dtype=float)
    from scipy import stats as sps
    _, p_val = sps.ttest_1samp(c, 0.0) if c.size > 1 else (None, float("nan"))
    return dict(ci_values=[round(x, 4) for x in cis], n=int(c.size),
                mean=float(c.mean()), p_value=float(p_val),
                mean_str=f"{c.mean():.3f}")


def p4_protocol_limited_record() -> dict:
    """P4 协议限制反证记录（不重跑——主 agent 2026-08-24 裁决，L23）。

    用已记录证据组装：data/m4_calibration.csv（点 0 + ref-T* 缩放行）+ L23 记录 +
    决定性点 3/20 试次初步证据（主 agent 2026-08-25 终止扫描；N=3 不足显著，如实记录）。
    """
    cal = _load_cal_rows()
    t25 = cal.get("ref-T25000", {})
    t10 = cal.get("ref-T10000", {})
    pt0 = cal.get("0", {})
    # 决定性点初步证据（/tmp/m4_res/point_4/trial_{0,1,2}.json；扫描于 N=3 终止）
    dec = _load_decisive_evidence()
    point5 = _load_point5_partial()
    return dict(
        pass_=True,   # 记录完成（协议限制反证记录，非机制失败；清单 §0 P4 修订文本）
        status="protocol-limited-counter-evidence",
        verdict=("P4(a) 统计显著性 = 协议限制反证记录（主 agent 2026-08-24 裁决，"
                 "m4_env_notes L23）：机制 A 在可行计算协议（T≤10s/N≤20，20 神经元 "
                 "多隔室闭环）下统计功效结构性不足——参考模型自身 N=10/T=10s 通过率仅 "
                 "23–40%、N=20/T=10s 43%，稳健显著需 T≥15–25s ≈ 数千 CPU-小时（本机 "
                 "不可行）→ 计算可行性边界反证，非机制失败（P1/P2 电路正确性已验证，"
                 "冒烟/全量回归绿）。"),
        reference_ci_25s=float(t25["ci_mean"]) if t25 else None,
        reference_p_25s=float(t25["p_value"]) if t25 else None,
        reference_d_25s=float(t25["cohen_d"]) if t25 else None,
        reference_in_band=bool(t25 and 0.25 <= float(t25["ci_mean"]) <= 0.75),
        t_scaling={5000: float(cal["ref-T5000"]["ci_mean"]),
                   10000: float(cal["ref-T10000"]["ci_mean"]),
                   15000: float(cal["ref-T15000"]["ci_mean"]),
                   25000: float(cal["ref-T25000"]["ci_mean"])},
        decided_point_brian2=dec["ci_values"],
        decided_point_note=("决定性点（T=10s, v_fwd0=1.0, g=8e6, θ_pir=1e-6, N=20）"
                            "全部完成：CĪ=" + dec["mean_str"] + "±" +
                            (f"{dec['sem']:.3f}" if dec["sem"] is not None else "—") +
                            "（p=" + (f"{dec['p_value']:.3f}" if dec["p_value"] is not None else "—") +
                            ", d=" + (f"{dec['cohen_d']:.2f}" if dec["cohen_d"] is not None else "—") +
                            "）——不显著；ΔCI vs 参考(10s)=0.317 → " +
                            (f"{dec['delta_ci_10s']:.3f}" if dec["delta_ci_10s"] is not None else "—") +
                            "（>0.15 记录为测量限制）。**确证 L23 协议限制裁决**。"
                            "主 agent 2026-08-25 科学结论：Brian2 电路实现机制 A 但效果"
                            "减弱（θ_eff≈1.9e-6 电路门限 + 突触时序延迟削弱 pirouette → "
                            "ΔCI≈0.23）；可行协议（T=10s）下不显著；参考模型 T=25s 稳健落带"
                            "（CI=0.494, p=0.0002, d=1.02）→ **P4 反证记录（非机制失败："
                            "机制经参考模型验证有效；是电路实现效率 + 计算可行性边界）**。"
                            "主 agent 终止时点快照 N=15（mean=0.085~0.105, p=0.55~0.64）"
                            "与完整 N=20 结论一致；N=15 终止判据（剩余 5 试次全 +1.0 亦"
                            "无法稳健显著）在完整数据下同样成立。弹性续跑能力留档 M5 前可选"),
        decisive_n_completed=dec["n"],
        decisive_stats=dict(mean=dec["mean_str"], sem=dec["sem"],
                            p_value=dec["p_value"], cohen_d=dec["cohen_d"],
                            delta_ci_10s=dec["delta_ci_10s"]),
        decisive_n15_snapshot=dict(
            mean=0.105, p_value=0.552,
            note="N=15 原始值统计（主 agent 舍入值 mean=0.085, p=0.64）；结论一致：不显著"),
        point5_partial=dict(
            n=point5["n"], mean=point5["mean_str"],
            p_value=point5["p_value"], ci_values=point5["ci_values"],
            note="点 5（θ_pir=2e-6, T=10s）部分结果 N=9/20（扫描终止前落盘）：CĪ="
                 + point5["mean_str"] + "（p=" +
                 (f"{point5['p_value']:.3f}" if point5["p_value"] is not None else "—") +
                 "，不显著，informational 敏感性证据；θ=2e-6 在 θ_eff≈1.9e-6 门限附近）"),
        main_agent_conclusion=(
            "Brian2 电路实现机制 A 但效果减弱（θ_eff=max(θ_pir, I_thresh/g_off)≈1.9e-6 "
            "门限 + 突触时序延迟削弱 pirouette → ΔCI vs 参考 ≈0.23）；可行协议（T=10s）"
            "下不显著（p=0.48~0.64）；参考模型 T=25s 稳健落带（CI=0.494, p=0.0002, d=1.02）"
            "→ P4 反证记录（非机制失败：机制经参考模型验证有效；是电路实现效率 + "
            "计算可行性边界）"),
        point0_ci=float(pt0["ci_mean"]) if pt0 else None,
        point0_p=float(pt0["p_value"]) if pt0 else None,
        point0_note=("点 0（θ_pir=4e-6, T=5s, v_fwd0=0.5, g=8e6, N=10）：CĪ=0.043±0.199, "
                     "p=0.834, d=0.07——ΔCI vs 参考(5s)=0.177 → 0.134 ≤ 0.15 ✓ 但显著性 ✗"),
        c_endpoint_bias=None,
        c_note=("P4(c) 终态偏向以可行协议实测为准（未跑 → 记录为可选，不静默编造，L23）"),
        control_note=("无梯度对照 p=0.41–0.46（>0.05 ✓，L7 同款处置：|CĪ| 点值 "
                      "informational）"),
        evidence_ref="docs/m4_env_notes.md §L23 + data/m4_calibration.csv（点 0 + "
                     "ref-T5000/10000/15000/25000 行）+ /tmp/m4_res/point_4/"
                     "trial_*.json（决定性点 N=20 全部完成：CĪ=0.099±0.139, p=0.482, "
                     "d=0.16, ΔCI vs ref(10s)=0.217>0.15——确证 L23 协议限制裁决）",
    )


def run_all(skip_p3p5: bool = False, skip_plot: bool = False) -> dict:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    sp = not skip_plot
    results = {
        "p1_ase_encoding": validate_p1_ase_encoding.run_p1(save_plot=sp),
        "p2_circuit": validate_p2_circuit.run_p2(save_plot=sp),
        "p4_chemotaxis_protocol_limited": p4_protocol_limited_record(),
    }
    if not skip_p3p5:
        results["p3_env_control"] = validate_p3_env_control.run_p3(save_plot=sp)
        results["p5_ablation"] = validate_p5_chemotaxis_ablation.run_p5(save_plot=sp)
    results["p6_reference"] = validate_p6_reference.run_p6(save_plot=sp)

    summary = {
        "milestone": "M4",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results,
        "all_pass": all(v["pass_"] for v in results.values()),
        "note": ("P4 = 协议限制反证记录（主 agent 2026-08-24 裁决，L23；不重跑）；"
                 "P3/P5 用最短协议（B2b 落地：T=1000ms、N=2，探针实测 T=1s 单试次 "
                 "1194–1888s 墙钟 → T=5000ms×N=10 全协议不可行，见 m4_env_notes L25）；"
                 "P6b 生物带验证主体 = numpy 行为参考模型全协议（T=25s/N=20）；"
                 "P1/P2/P3/P5/P6 已落盘 CSV，M4_REUSE=1 时直接读回判定（不重跑 Brian2）"),
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    return summary


def main():
    ap = argparse.ArgumentParser(description="M4 验证总运行器（P1–P6）")
    ap.add_argument("--skip-p3p5", action="store_true")
    ap.add_argument("--skip-plot", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    summary = run_all(skip_p3p5=args.skip_p3p5, skip_plot=args.skip_plot)
    print("==== M4 验证判定汇总 ====")
    for k, v in summary["results"].items():
        st = v.get("status", "")
        print(f"  {k:36s} pass={v['pass_']}" + (f" ({st})" if st else ""))
    print(f"  all_pass = {summary['all_pass']}")
    print(f"汇总 JSON: {SUMMARY_JSON}")
    print(f"总耗时: {time.time() - t0:.1f}s")
    return summary


if __name__ == "__main__":
    main()
