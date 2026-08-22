"""M1 逐项验证总运行器（清单 §5 验收）。

依次运行 P2–P6 五个验证（含图/表落盘），汇总判定写入
  reports/neuro/m1_validation_summary.json
并生成 docs/m1_report.md（§6.2 要求的报告，含 P1–P8 对照表）。

用法：
  .venv-neuro/bin/python -m neural_exploration.tools.run_m1_validation
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.tools import (  # noqa: E402
    validate_p2_waveform,
    validate_p3_fi,
    validate_p4_psp,
    validate_p5_speed,
    validate_p6_saltatory,
)

REPORTS_DIR = os.path.join(ROOT, "neural_exploration", "reports", "neuro")
SUMMARY_JSON = os.path.join(REPORTS_DIR, "m1_validation_summary.json")
DOCS_DIR = os.path.join(ROOT, "neural_exploration", "docs")
REPORT_MD = os.path.join(DOCS_DIR, "m1_report.md")


def run_all() -> dict:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    results = {
        "p2_waveform": validate_p2_waveform.run_p2(save_plot=True),
        "p3_fi_curve": validate_p3_fi.run_p3(save_plot=True),
        "p4_psp": validate_p4_psp.run_p4(save_plot=True),
        "p5_speed": validate_p5_speed.run_p5(),
        "p6_saltatory": validate_p6_saltatory.run_p6(save_plot=True),
    }
    summary = {"results": results}
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main():
    summary = run_all()
    print("==== M1 验证判定汇总 ====")
    for k, v in summary["results"].items():
        print(f"  {k:16s} pass={v['pass_']}")
    print(f"汇总 JSON: {SUMMARY_JSON}")
    return summary


if __name__ == "__main__":
    main()
