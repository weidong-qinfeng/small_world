"""M3 逐项验证总运行器（清单 §5 验收：P1–P5 顺序运行）。

依次运行 P1–P5 五个验证（含图/CSV 落盘），汇总判定写入
  reports/neuro/m3_validation_summary.json
结构沿用 tools/run_m2_validation.py；P6（pytest 全绿 + 报告写盘）由
  tools/gen_m3_report.py + 全量 pytest 独立确认（见报告 §8）。

用法：
  .venv-neuro/bin/python -m neural_exploration.tools.run_m3_validation
可选：
  --skip-p5  跳过 P5（主 agent 三态裁决期间快速回归其余判据）
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
    validate_p1_direction,
    validate_p2_chain,
    validate_p3_latency,
    validate_p4_intensity,
    validate_p5_ablation,
)

REPORTS_DIR = os.path.join(ROOT, "neural_exploration", "reports", "neuro")
SUMMARY_JSON = os.path.join(REPORTS_DIR, "m3_validation_summary.json")


def run_all(skip_p5: bool = False) -> dict:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    results = {
        "p1_direction": validate_p1_direction.run_p1(save_plot=True),
        "p2_chain": validate_p2_chain.run_p2(save_plot=True),
        "p3_latency": validate_p3_latency.run_p3(save_plot=True),
        "p4_intensity": validate_p4_intensity.run_p4(save_plot=True),
    }
    if not skip_p5:
        results["p5_ablation"] = validate_p5_ablation.run_p5(save_plot=True)
    summary = {
        "milestone": "M3",
        "results": results,
        "all_pass": all(v["pass_"] for v in results.values()),
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    return summary


def main():
    ap = argparse.ArgumentParser(description="M3 验证总运行器（P1–P5）")
    ap.add_argument("--skip-p5", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    summary = run_all(skip_p5=args.skip_p5)
    print("==== M3 验证判定汇总 ====")
    for k, v in summary["results"].items():
        print(f"  {k:14s} pass={v['pass_']}")
    print(f"  all_pass = {summary['all_pass']}")
    print(f"汇总 JSON: {SUMMARY_JSON}")
    print(f"总耗时: {time.time() - t0:.1f}s")
    return summary


if __name__ == "__main__":
    main()
