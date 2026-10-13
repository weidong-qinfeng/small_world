"""**σ 杠杆受控配对探针的分析（R84 第 37 条 / R76）**：读两臂产物 ⇒ 脚本算分布指标 + 判定。

**受控配对前置检查（机器核对，不靠人眼）**：两臂必须是
①同 `net_seed/event_seed/reset_seed`、②同 `gmax_scale/bias_scale/dose_mult/T_ms/settle_ms/indelivery`、
③**仅 `bias_log_sigma` 不同**。任一不成立 ⇒ 退出 **15**（"控制了变量"这句话必须可机器验证）。

**判定（阈值**跑前声明**、写在 `neural_exploration/tools/run_m11_p5_arms.py`）**：
`D_KS ≥ max(0.10, 3×噪声底)` **且** |p90−p10 极差比 − 1| ≥ 25% ⇒ `EFFECT`；否则 `no_effect` 并退出 **9**。

用法：
    ./.venv-m11/bin/python -m neural_exploration.tools.analyze_m11_sigma_probe \
        --dir neural_exploration/data/m11_p5_l1_sigma_probe \
        --sigma-pair 0.30,0.90 [--negative-pair 0.90,0.90] [--noise-seed-pair ARM_A,ARM_B]
产物：`data/m11_l1_sigma_probe.json`
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np

from neural_exploration.tools.run_m11_p5_arms import (SIGMA_EFFECT_MIN_D,
                                                      SIGMA_EFFECT_MIN_SPREAD_REL,
                                                      SIGMA_EFFECT_NOISE_MULT, ks_statistic,
                                                      rate_dist_metrics)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "neural_exploration", "data")
PAIR_KEYS = ("net_seed", "event_seed", "reset_seed")
# ⚠️ 臂产物的字段是**嵌套**的（gmax/bias/indegree/delay_bins 在 `net` 下）⇒ 用点路径取值
ENV_KEYS = ("net.gmax_scale", "net.bias_scale", "drive_mult", "T_ms", "settle_ms", "delivery",
            "net.indegree", "net.delay_bins", "n_neuron")


def dig(a: dict, path: str):
    cur = a
    for k in path.split("."):
        cur = cur[k]
    return cur


def load_arm(d: str, arm_id: str) -> dict:
    with open(os.path.join(d, arm_id + ".json")) as f:
        a = json.load(f)
    counts = []
    with open(os.path.join(d, arm_id + "_counts.csv")) as f:
        for r in csv.DictReader(f):
            counts.append(int(r["count"]))
    a["_counts"] = np.asarray(counts, dtype=np.int64)
    a["_m"] = rate_dist_metrics(a["_counts"], a["T_ms"])
    a["_key"] = (int(a["seeds"]["net_seed"]), int(a["seeds"]["event_seed"]),
                 int(a["seeds"]["reset_seed"]))
    a["_env"] = {k: dig(a, k) for k in ENV_KEYS}
    a["_sigma"] = a["net"].get("bias_log_sigma", None)
    return a


def env_diff(a: dict, b: dict) -> list:
    return ["%s: %r vs %r" % (k, a["_env"][k], b["_env"][k])
            for k in ENV_KEYS if a["_env"][k] != b["_env"][k]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(DATA_DIR, "m11_p5_l1_sigma_probe"))
    ap.add_argument("--sigma-pair", required=True,
                    help="两臂 __substring__（如 0.30,0.90）；按 run_id 子串匹配")
    ap.add_argument("--negative-pair", default=None, help="同 σ 重跑两臂（反例自证）")
    ap.add_argument("--noise-seed-pair", default=None, help="同 σ 换 net_seed 两臂（噪声底）")
    ap.add_argument("--out-json", default=os.path.join(DATA_DIR, "m11_l1_sigma_probe.json"))
    args = ap.parse_args(argv)

    def find(sub: str) -> str:
        hits = [f[:-5] for f in os.listdir(args.dir)
                if f.endswith(".json") and f[:-5].endswith(sub)]
        if len(hits) != 1:
            raise SystemExit("**匹配不唯一/找不到**：'%s' ⇒ %r" % (sub, hits))
        return hits[0]

    sA, sB = args.sigma_pair.split(",")
    idA, idB = find(sA), find(sB)
    A, B = load_arm(args.dir, idA), load_arm(args.dir, idB)
    print("== σ 受控配对（%s vs %s）==" % (idA, idB))
    print("  配对键 (net,event,reset)：A=%r B=%r" % (A["_key"], B["_key"]))
    fails = []
    if A["_key"] != B["_key"]:
        fails.append("复刻轴三元组不同 ⇒ 不是受控配对：%r vs %r" % (A["_key"], B["_key"]))
    d = env_diff(A, B)
    if d:
        fails.append("除 σ 外还有别的量不同（非受控）：" + "；".join(d))
    if A["_sigma"] == B["_sigma"] or A["_sigma"] is None or B["_sigma"] is None:
        fails.append("σ 未真正区分（A=%r B=%r）⇒ 该对臂不能用来证明 σ 有效果"
                     % (A["_sigma"], B["_sigma"]))
    print("  σ：A=%.2f B=%.2f｜bias=%.2f gmax=%.2f dose=%s｜T=%s ms｜%s"
          % (A["_sigma"] or -1, B["_sigma"] or -1, A["_env"]["net.bias_scale"],
             A["_env"]["net.gmax_scale"],
             A["drive_mult"], A["T_ms"], A["delivery"]))
    # 量纲自检（R76/R80）：counts = rate_hz × T_s 的算例
    ex = dict(identity="rate_hz = counts / (T_ms/1e3)", counts=8, T_ms=1000.0,
              rate_hz=8 / (1000.0 / 1000.0))
    print("  [量纲自检] %s：counts=8 @ T=%g ms ⇒ %.1f Hz（silent_frac 无量纲）"
          % (ex["identity"], ex["T_ms"], ex["rate_hz"]))

    D = ks_statistic(A["_counts"], B["_counts"])
    spread_ratio = (B["_m"]["spread_p10_p90"] / A["_m"]["spread_p10_p90"]
                    if A["_m"]["spread_p10_p90"] > 0 else float("inf"))

    def pair_block(sub_pair, tag):
        if not sub_pair:
            return None
        x, y = sub_pair.split(",")
        X, Y = load_arm(args.dir, find(x)), load_arm(args.dir, find(y))
        return dict(tag=tag, arms=[X["arm_id"], Y["arm_id"]],
                    D=ks_statistic(X["_counts"], Y["_counts"]),
                    bit_identical=(X["sha256_of_trace"] == Y["sha256_of_trace"]),
                    spread_p10_p90_ratio=(Y["_m"]["spread_p10_p90"] / X["_m"]["spread_p10_p90"]
                                          if X["_m"]["spread_p10_p90"] > 0 else None),
                    key_X=list(X["_key"]), key_Y=list(Y["_key"]))

    neg = pair_block(args.negative_pair, "negative_same_sigma")
    noi = pair_block(args.noise_seed_pair, "noise_same_sigma_other_netseed")
    D_noise = (noi or {}).get("D", 0.0) or 0.0
    thr = max(SIGMA_EFFECT_MIN_D, SIGMA_EFFECT_NOISE_MULT * D_noise)
    eff = (D >= thr) and (abs(spread_ratio - 1.0) >= SIGMA_EFFECT_MIN_SPREAD_REL)
    print("  A：中位率 %.4f Hz｜静默 %.4f｜p10 %.3f｜p90 %.3f｜极差 %.3f｜spk %d"
          % (A["_m"]["rate_median"], A["_m"]["silent_frac"], A["_m"]["rate_p10"],
             A["_m"]["rate_p90"], A["_m"]["spread_p10_p90"], A["agg"]["spk_total"]))
    print("  B：中位率 %.4f Hz｜静默 %.4f｜p10 %.3f｜p90 %.3f｜极差 %.3f｜spk %d"
          % (B["_m"]["rate_median"], B["_m"]["silent_frac"], B["_m"]["rate_p10"],
             B["_m"]["rate_p90"], B["_m"]["spread_p10_p90"], B["agg"]["spk_total"]))
    print("  D_KS=%.4f ≥ max(%.2f, %.1f×噪声底 %.4f)=%.4f 且 极差比 %.4f（偏离 ≥%.0f%%）⇒ **%s**"
          % (D, SIGMA_EFFECT_MIN_D, SIGMA_EFFECT_NOISE_MULT, D_noise, thr, spread_ratio,
             100 * SIGMA_EFFECT_MIN_SPREAD_REL, "EFFECT" if eff else "no_effect"))
    if neg:
        print("  [反例自证] 同 σ 两臂：D=%.4f、逐位相同=%s ⇒ %s"
              % (neg["D"], neg["bit_identical"], "no_effect ✓（检测器不假阳性）"
                 if (neg["D"] == 0.0 and neg["bit_identical"]) else "**检测器假阳性 ✗**"))
        if not (neg["D"] == 0.0 and neg["bit_identical"]):
            fails.append("反例自证失败：同 σ 同规格两臂被判出差异")
    if noi:
        print("  [噪声底] 同 σ 换实现：D=%.4f（net_seed %s vs %s）"
              % (noi["D"], noi["key_X"][0], noi["key_Y"][0]))
    out = dict(tool="tools/analyze_m11_sigma_probe.py",
               generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
               dir=os.path.relpath(args.dir, PROJECT_ROOT),
               controlled_pair=dict(A=idA, B=idB, key_A=list(A["_key"]), key_B=list(B["_key"]),
                                    sigma_A=A["_sigma"], sigma_B=B["_sigma"],
                                    env_A=A["_env"], env_B=B["_env"],
                                    controlled=(not fails)),
               metrics_A=A["_m"], metrics_B=B["_m"],
               sha256_of_trace=dict(A=A["sha256_of_trace"], B=B["sha256_of_trace"]),
               dimension_self_check=ex,
               thresholds=dict(min_D=SIGMA_EFFECT_MIN_D, noise_mult=SIGMA_EFFECT_NOISE_MULT,
                               min_spread_rel=SIGMA_EFFECT_MIN_SPREAD_REL, D_threshold_used=thr),
               D_KS=D, spread_p10_p90_ratio=spread_ratio,
               negative_control=neg, noise_floor=noi, D_noise_floor=D_noise,
               verdict=("EFFECT（σ 真改变了逐神经元率分布的形状）" if eff and not fails
                        else ("**受控配对不成立（结论无效）**" if fails else "no_effect（形状没动）")),
               failures=fails)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("产物 -> %s" % args.out_json)
    if fails:
        print("**受控配对检查失败**：")
        for x in fails:
            print("   -", x)
        return 15
    if not eff:
        print("**σ 杠杆判定 no_effect ⇒ 退出码 9（L1 全量不建议开跑）**")
        return 9
    print("σ 杠杆判定 EFFECT（受控配对 + 阈值双条件齐备）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
