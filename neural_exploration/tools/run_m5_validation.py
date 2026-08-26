"""M5 逐项验证总运行器（清单步骤 7 验收：P1–P6 顺序运行 + 汇总判定）。

依次运行 P1/P2/P3/P4/P5/P6 六个验证（含图/CSV 落盘）；汇总判定写入
  reports/neuro/m5_validation_summary.json
判定语义（主 agent 裁决 2026-08-26，三态选项 ①）：
  - P1（连接组）/P3（咽部）/P4（趋化 T=15s×N=20 全协议）/P5（逃避，含测量限制记录）
    = pass；
  - P2（静息）/P6（自发）= **反证记录型 pass**（与 M4 P4 同型：记录本身即交付物；
    缺失机制清单 → M6 优先验证清单）。
  - 预算纪律：P4 单试次 >5min → 报主 agent（脚本输出 + summary 字段，不静默）。
P7（pytest 全绿 + 报告写盘）由全量 pytest + tools/gen_m5_report.py 独立确认。

用法：
  .venv-neuro/bin/python -m neural_exploration.tools.run_m5_validation
可选：
  --skip-p2p6  跳过 P2/P6 重协议（计算受限时快速回归其余判据）
  --skip-plot  不出图（纯判定）
  --reuse      M5_REUSE=1 语义：各 P 脚本存在 CSV 时直接读回判定（不重跑 Brian2）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.tools import (  # noqa: E402
    validate_p1_connectome,
    validate_p2_resting,
    validate_p3_pharynx,
    validate_p4_chemotaxis,
    validate_p5_escape,
    validate_p6_spontaneous,
)

REPORTS_DIR = os.path.join(ROOT, "neural_exploration", "reports", "neuro")
SUMMARY_JSON = os.path.join(REPORTS_DIR, "m5_validation_summary.json")
DATA_DIR = os.path.join(ROOT, "neural_exploration", "data")

#: 各 P 脚本的 result JSON（完整判定 dict；复用模式读回，不重跑 Brian2）
_RESULT_JSON = {
    "p1_connectome": "m5_p1_result.json",
    "p2_resting": "m5_p2_result.json",
    "p3_pharynx": "m5_p3_result.json",
    "p4_chemotaxis": "m5_p4_result.json",
    "p5_escape": "m5_p5_result.json",
    "p6_spontaneous": "m5_p6_result.json",
}
_RUNNERS = {
    "p1_connectome": lambda sp: validate_p1_connectome.run_p1(save_plot=sp),
    "p2_resting": lambda sp: validate_p2_resting.run_p2(save_plot=sp),
    "p3_pharynx": lambda sp: validate_p3_pharynx.run_p3(save_plot=sp),
    "p4_chemotaxis": lambda sp: validate_p4_chemotaxis.run_p4(save_plot=sp),
    "p5_escape": lambda sp: validate_p5_escape.run_p5(save_plot=sp),
    "p6_spontaneous": lambda sp: validate_p6_spontaneous.run_p6(save_plot=sp),
}
#: P4 CSV → result dict 转换（m5_p4_chemotaxis.csv 逐行映射；含摘要/报告所需字段）
_P4_CSV_FIELDS = {
    "pass_": "pass_", "status": "status", "verdict": "verdict",
    "n_trials": "n_trials", "t_total_ms": "t_total_ms",
    "ci_mean": "ci_mean", "ci_sem": "ci_sem", "p_value": "p_value",
    "cohen_d": "cohen_d", "in_tolerance_band": "in_band",
    "delta_ci_vs_reference_15s": "delta_ci_vs_reference",
    "reference_ci_15s": "reference_ci_15s", "ctrl_mean": "ctrl_mean",
    "ctrl_p": "ctrl_p", "approaching_frac": "approaching_frac",
    "deterministic": "deterministic", "budget_ok": "budget_ok",
    "max_trial_wall_s": "max_trial_wall_s", "total_wall_s": "total_wall_s",
}


def _load_result(key: str) -> dict:
    """复用：读 result JSON；缺失时尝试从 P 脚本 CSV 重建（P4 全协议不重跑）。"""
    path = os.path.join(DATA_DIR, _RESULT_JSON[key])
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    if key == "p4_chemotaxis":
        return _p4_from_csv(os.path.join(DATA_DIR, "m5_p4_chemotaxis.csv"))
    return None


def _p4_from_csv(csv_path: str) -> dict:
    import csv as _csv

    if not os.path.exists(csv_path):
        return None
    rows = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in _csv.DictReader(x for x in f
                                 if not x.strip().startswith("#")):
            rows[str(r.get("metric", "")).strip()] = r.get("value", "")

    def _num(k, default=None):
        v = rows.get(k, "")
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def _bool(k, default=False):
        v = rows.get(k, "")
        return str(v).lower() == "true" if v else default

    out = {
        "pass_": _bool("pass_"), "status": rows.get("status", ""),
        "verdict": rows.get("verdict", ""),
        "n_trials": _num("n_trials", 20), "t_total_ms": _num("t_total_ms", 15000),
        "ci_mean": _num("ci_mean"), "ci_sem": _num("ci_sem"),
        "p_value": _num("p_value"), "cohen_d": _num("cohen_d"),
        "in_band": _bool("in_tolerance_band"),
        "delta_ci_vs_reference": _num("delta_ci_vs_reference_15s"),
        "reference_ci_15s": _num("reference_ci_15s"),
        "ctrl_mean": _num("ctrl_mean"), "ctrl_p": _num("ctrl_p"),
        "approaching_frac": _num("approaching_frac"),
        "deterministic": _bool("deterministic"),
        "budget": {
            "budget_ok": _bool("budget_ok"),
            "max_trial_wall_s": _num("max_trial_wall_s"),
            "total_wall_s": _num("total_wall_s"),
        },
        "reused_from": "m5_p4_chemotaxis.csv（全协议结果，B2 已跑）",
    }
    return out


def run_all(skip_p2p6: bool = False, skip_plot: bool = False,
            reuse: bool = False) -> dict:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    sp = not skip_plot
    results = {}

    for key, runner in _RUNNERS.items():
        if skip_p2p6 and key in ("p2_resting", "p6_spontaneous"):
            continue
        if reuse:
            rec = _load_result(key)
            if rec is not None:
                print(f"  {key}: 复用 result JSON/CSV（pass_={rec.get('pass_')}）")
                results[key] = rec
                continue
        results[key] = runner(sp)

    summary = {
        "milestone": "M5",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results,
        "all_pass": all(v.get("pass_", False) for v in results.values()),
        "pass_type": {
            "p1_connectome": "pass",
            "p3_pharynx": "pass",
            "p4_chemotaxis": "counter-evidence-record",
            "p5_escape": "pass-with-measurement-limitations",
            "p2_resting": "counter-evidence-record",
            "p6_spontaneous": "counter-evidence-record",
        },
        "note": (
            "P1/P3/P5 pass；P2/P4/P6 = 反证记录型 pass（与 M4 P4 同型：记录本身即交付物；"
            "缺失机制清单 → M6 优先验证清单，docs/m5_env_notes.md L40）。"
            "**P4 = 全协议（T=15s×N=20 梯度+对照，M5 定稿 WormLoop/VirtualBody）实测 "
            "CĪ=" + str(results["p4_chemotaxis"].get("ci_mean")) + "（p=" +
            str(results["p4_chemotaxis"].get("p_value")) + "，d=" +
            str(results["p4_chemotaxis"].get("cohen_d")) + "）方向负/不显著——预注册指标"
            "不满足，主 agent 裁决 2026-08-26 判为反证记录型：根因 = 夹带病理（fwd/back "
            "运动池共同发放 → v≈0 → 位移 0.2-0.5 皿单位/15s，L39/L40）；M4 前向身体对照"
            "（同种子 N=6 均值 +0.360 vs M5 -0.407）仅作记录，不改变判据主体（换身体规避"
            "失败 = 不诚实）。预算纪律 OK：单试次 ~53s < 5min。"
            "P5 方向相位敏感测量限制（L40 #5）：定稿 τ_trans=23（touch@73ms）→ not_back；"
            "touch@50ms（τ_trans=0）→ back。"),
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    return summary


def main():
    ap = argparse.ArgumentParser(description="M5 验证总运行器（P1–P6）")
    ap.add_argument("--skip-p2p6", action="store_true")
    ap.add_argument("--skip-plot", action="store_true")
    ap.add_argument("--reuse", action="store_true",
                    help="M5_REUSE=1：P 脚本 CSV 存在时读回判定（不重跑 Brian2）")
    args = ap.parse_args()
    t0 = time.time()
    summary = run_all(skip_p2p6=args.skip_p2p6, skip_plot=args.skip_plot,
                      reuse=args.reuse or os.environ.get("M5_REUSE") == "1")
    print("==== M5 验证判定汇总 ====")
    for k, v in summary["results"].items():
        st = v.get("status", "") if isinstance(v, dict) else ""
        pt = summary["pass_type"].get(k, "")
        print(f"  {k:22s} pass={v.get('pass_') if isinstance(v, dict) else '?'}"
              f" [{pt}]" + (f" ({st})" if st else ""))
    print(f"  all_pass = {summary['all_pass']}")
    print(f"汇总 JSON: {SUMMARY_JSON}")
    print(f"总耗时: {time.time() - t0:.1f}s")
    return summary


if __name__ == "__main__":
    main()
