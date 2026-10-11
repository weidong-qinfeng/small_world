"""M10 P5 标定杠杆扫描（G1 门前置；标定顺序沿 M9 L21.4：背景率 λ > bias 中位 b0 > 类级权重 w）。

目标（**判据带见 `data/m10_calibration_targets.csv`，不事后调**）：
  1. **零驱动群率 ≈ 0**（自持比 ≤ `selfsustain_ratio_max`）——关键是**递归回路增益 < 1**；
  2. **基准驱动（λ0）下中位发放率落带**（0.3–1.5 Hz）且静默比例落带；
  3. 输出可直接喂给 `validate_p10_resting.py` 与机制阶梯的 `--g-exc/--g-inh/--bias-median/--lam0`。

**结构证据级别 = 抽象（Fish1 partial 锚定）**；权重为类级自由参数（§0.7 #7，不做逐突触拟合）。

用法：
    PYTHONHASHSEED=0 ./.venv-m9/bin/python -m neural_exploration.tools.calibrate_m10_weights \
        --n-neuron 1000 --T-ms 2000
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import time

from ..src.larval_zf_circuit import ZFNetParams, build_engine, run_protocol, DT_MS  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-neuron", type=int, default=1000)
    ap.add_argument("--indegree", type=int, default=20)
    ap.add_argument("--T-ms", type=float, default=2000.0)
    ap.add_argument("--settle-ms", type=float, default=500.0)
    ap.add_argument("--segment-steps", type=int, default=20000)
    ap.add_argument("--g-exc", default="2,4,6,9")
    ap.add_argument("--g-inh", default="10,22,40")
    ap.add_argument("--bias", default="150,250")
    ap.add_argument("--lam0", default="1.0,2.0,4.0")
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "m10_calibration_scan.csv"))
    args = ap.parse_args(argv)
    gx = [float(v) for v in args.g_exc.split(",")]
    gi = [float(v) for v in args.g_inh.split(",")]
    b0 = [float(v) for v in args.bias.split(",")]
    lams = [float(v) for v in args.lam0.split(",")]
    T = int(round(args.T_ms / DT_MS)); settle = int(round(args.settle_ms / DT_MS))
    rows = []
    t0 = time.time()
    with open(args.out, "w", newline="") as f:
        f.write("# M10 P5 标定扫描（结构证据级别 = 抽象/Fish1-partial 锚定；判据带 = 预注册）\n")
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["g_exc", "g_inh", "bias_median", "lam_hz", "pop_rate_hz",
                    "rate_median_hz", "silent_frac", "bout_fraction", "n_bouts",
                    "wall_s"])
        f.flush()
        for g_e, g_i, bm in itertools.product(gx, gi, b0):
            Pm = ZFNetParams(n_neuron=args.n_neuron, indegree=args.indegree,
                             g_exc=g_e, g_inh=g_i, bias_median_mv_s=bm)
            for lam in [0.0] + lams:
                e = build_engine(Pm, device=args.device)
                r = run_protocol(e, T, settle_steps=settle,
                                 segment_steps=args.segment_steps, lam_hz=lam, ev_seed=3)
                del e
                w.writerow([g_e, g_i, bm, lam, round(r["pop_rate_hz"], 5),
                            round(r["rate_median_hz"], 5), round(r["silent_frac"], 5),
                            round(r["bout_fraction"], 5), r["n_bouts"], r["wall_s"]])
                f.flush()
                rows.append(dict(g_exc=g_e, g_inh=g_i, bias=bm, lam=lam,
                                 pop=round(r["pop_rate_hz"], 4),
                                 med=round(r["rate_median_hz"], 4),
                                 silent=round(r["silent_frac"], 4),
                                 bout=round(r["bout_fraction"], 4),
                                 nbouts=r["n_bouts"]))
                print(f"  g_exc={g_e} g_inh={g_i} bias={bm} λ={lam}: pop={rows[-1]['pop']} "
                      f"med={rows[-1]['med']} silent={rows[-1]['silent']} "
                      f"bout={rows[-1]['bout']} ({r['wall_s']}s)", flush=True)
    # 选优：零驱动 pop 最小 且 λ 下 med 落 [0.1,1.5] 且 silent ≥0.3
    by = {}
    for r in rows:
        key = (r["g_exc"], r["g_inh"], r["bias"])
        by.setdefault(key, {})[r["lam"]] = r
    best = None
    for key, d in by.items():
        z = d.get(0.0)
        if z is None or z["pop"] > 0.5:
            continue
        for lam, r in d.items():
            if lam == 0.0:
                continue
            if 0.1 <= r["med"] <= 1.5 and r["silent"] >= 0.3:
                score = abs(r["med"] - 0.6) + 0.1 * z["pop"]
                if best is None or score < best[0]:
                    best = (score, key, lam, r, z)
    summary = dict(generated_by="tools/calibrate_m10_weights.py",
                   n_points=len(rows), elapsed_s=round(time.time() - t0, 1),
                   best=None)
    if best:
        _, key, lam, r, z = best
        summary["best"] = dict(g_exc=key[0], g_inh=key[1], bias_median=key[2], lam0=lam,
                               zero_drive_pop=z["pop"], med=r["med"], silent=r["silent"],
                               bout=r["bout"],
                               suggested_args=(f"--g-exc {key[0]} --g-inh {key[1]} "
                                               f"--bias-median {key[2]} --lam0 {lam}"))
        print("[calib] best:", json.dumps(summary["best"], ensure_ascii=False), flush=True)
    else:
        print("[calib] 未找到满足（零驱动静默 ∧ 落带 ∧ 静默≥0.3）的配置——"
              "按清单 §8.1 G1 失败路径登记（不静默）", flush=True)
    with open(os.path.join(DATA_DIR, "m10_calibration_scan.json"), "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
