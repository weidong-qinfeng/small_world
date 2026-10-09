#!/usr/bin/env python
"""P8 3016 point ≤5s 短窗规模对照（B2；可选——预算允许时单独跑）。

清单 §8 + B2 预算裁决：3016 只做结构性验证（3016 point 30s=843s/试次不可行）；
本脚本跑 3016 point T=5s 自发短窗 → 发放率/静默/转换窗统计 → 合并进
reports/neuro/m8_p8_activity_full.json 的 short3016 字段（不重跑 300 全协议）。

用法：PYTHONHASHSEED=0 MPLBACKEND=Agg ./.venv-neuro/bin/python \
  neural_exploration/tools/validate_p8_activity_3016.py
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.src.larva_circuit import LarvaCircuit  # noqa: E402
from neural_exploration.tools.validate_p8_activity import (  # noqa: E402
    DATA_DIR,
    REPORTS_NEURO,
    SCALE_3016,
    FIDELITY_3016,
    SHORT_T_MS,
    SEED,
    load_resting_bands,
    load_weight_rows,
    run_activity_full,
)

FULL_JSON = os.path.join(REPORTS_NEURO, "m8_p8_activity_full.json")


def main() -> int:
    if not os.path.exists(FULL_JSON):
        print("m8_p8_activity_full.json 缺失——先跑 validate_p8_activity.py")
        return 2
    with open(FULL_JSON, encoding="utf-8") as f:
        res = json.load(f)

    print(f"=== P8 3016 point 短窗（T={SHORT_T_MS}ms）===")
    short3016 = run_activity_full(SCALE_3016, FIDELITY_3016, SHORT_T_MS, SEED,
                                  "3016_point_short_5s")
    full300 = res.get("full300", {})
    res["short3016"] = short3016
    res["scale_comparison"] = dict(
        note=("3016 point 只跑 ≤5s 短窗（预算纪律：3016 point 30s=843s/试次不可行）；"
              "3016 point CI=0 行为层退化反证已记录（缩放扫描）；活动对照仅作规模侧"
              "结构性参考"),
        silent_frac_300=full300.get("rates", {}).get("silent_frac"),
        silent_frac_3016=short3016["rates"]["silent_frac"],
        median_300=full300.get("rates", {}).get("median_hz"),
        median_3016=short3016["rates"]["median_hz"])
    with open(FULL_JSON, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=str)

    print(f"3016 point: median={short3016['rates']['median_hz']:.3f}Hz "
          f"max={short3016['rates']['max_hz']:.3f}Hz "
          f"silent={short3016['rates']['silent_frac']:.3f} "
          f"trans n={short3016['transitions']['n']} "
          f"pass_model={short3016['criteria']['pass_model']}")
    print(f"300 对照: silent={full300.get('rates', {}).get('silent_frac')} "
          f"median={full300.get('rates', {}).get('median_hz')}")
    print(f"已合并入 {FULL_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
