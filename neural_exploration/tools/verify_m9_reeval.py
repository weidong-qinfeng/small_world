"""主agent 独立验收探针：复核 `m10_m9_reeval` 的 E-M9-1 声明（M9 冻结引擎绝对速率不可复现）。

**只读**：复用 `tools/reeval_m10_m9_rebuttals.py` 的装配函数与协议常量，**不写它的产物**
（`data/m10_m9_reeval.csv` 是 M10-B1 的交付物，主agent 不得污染）。

验收目标（对 B1 结算里最重的那条量化声明做第三方复现）：
  1. M9 冻结引擎同网络/同参数/同协议在**全新进程**里重跑，群率是否偏离 M9 记录值 4.6783 Hz（声称 +28%）；
  2. **同一臂重复两个全新进程**，看 M9 冻结引擎在 MPS 上的值是否**进程间可复现**
     —— 若两次不一致，即是"密集发放下越界读、值依赖分配历史"的直接证据（E-M9-1 的机制解释）；
  3. 修正内核 m10 同臂对照。

用法（每臂独立进程，M9 L40 纪律）：
    PYTHONHASHSEED=0 MPLBACKEND=Agg ./.venv-m9/bin/python \
        -m neural_exploration.tools.verify_m9_reeval --engine m9 --lam0 0.0 --rep 1
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUT_CSV = os.path.join(DATA_DIR, "_mainagent_m9_reeval_verify.csv")

# M9 定稿记录值（m9_report / m9_validation_summary）
M9_RECORDED = {"lam0_0.0": 4.6783, "lam0_0.5": 4.7482, "lam0_2.0": 5.0925}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="m9", choices=["m9", "m10"])
    ap.add_argument("--lam0", type=float, default=0.0)
    ap.add_argument("--rep", type=int, default=1, help="重复编号（进程间可复现性对照）")
    args = ap.parse_args(argv)

    # 延迟导入：装配函数只读 M9 冻结件
    from .reeval_m10_m9_rebuttals import (build_m10_engine, build_m9_engine, run_arm,
                                          T_MS, SETTLE_MS, SEGMENT, G_EXT)

    t0 = time.time()
    dev = os.environ.get("M10_REEVAL_DEVICE", "mps")
    if args.engine == "m10":
        eng, n, p = build_m10_engine()
    else:
        eng, n, p = build_m9_engine()
    r = run_arm(eng, n, args.lam0, seed=0, tag=f"{args.engine}#{args.rep} λ={args.lam0}")
    r["engine"] = args.engine
    r["lam0"] = args.lam0
    r["rep"] = args.rep
    r["device"] = dev
    r["protocol"] = dict(T_MS=T_MS, SETTLE_MS=SETTLE_MS, SEGMENT=SEGMENT, G_EXT=G_EXT)
    key = f"lam0_{args.lam0}"
    r["m9_recorded_hz"] = M9_RECORDED.get(key)
    if r["m9_recorded_hz"]:
        r["rel_dev_vs_recorded"] = round(
            (r["pop_rate_hz"] - r["m9_recorded_hz"]) / r["m9_recorded_hz"], 6)
    r["wall_s_total"] = round(time.time() - t0, 1)

    new = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        if new:
            f.write("# 主agent 独立验收探针（M10 阶段3 期间）：复核 E-M9-1\n")
            f.write("# 只读 M9 冻结件；每臂全新进程；设备默认 mps\n")
            w.writerow(["engine", "lam0", "rep", "device", "n_events", "pop_rate_hz",
                        "rate_median_hz", "silent_frac", "bout_fraction", "n_bouts",
                        "silent_bin_frac", "ms_per_step", "m9_recorded_hz",
                        "rel_dev_vs_recorded", "wall_s"])
        w.writerow([r["engine"], r["lam0"], r["rep"], r["device"], r["n_events"],
                    round(r["pop_rate_hz"], 6), round(r["rate_median_hz"], 6),
                    round(r["silent_frac"], 6), round(r["bout_fraction"], 6), r["n_bouts"],
                    round(r["silent_bin_frac"], 6), round(r["ms_per_step"], 4),
                    r["m9_recorded_hz"] or "", r.get("rel_dev_vs_recorded", ""), r["wall_s"]])
    print("MAINAGENT_VERIFY " + json.dumps(
        {k: r[k] for k in ("engine", "lam0", "rep", "device", "pop_rate_hz",
                           "rate_median_hz", "silent_frac", "bout_fraction", "n_bouts",
                           "m9_recorded_hz", "rel_dev_vs_recorded", "wall_s")},
        ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
