"""构建 **G1（P5）正式矩阵计划** `data/m11_p5_plan_g1.json`（★ 结构全量）。

> **不修改生产者**（R48：`tools/run_m11_p5_arms.py` 已冻结于 sha256 `81f29d528f554ca2`）——
> 本脚本只产出计划 JSON，由既有驱动器 `--plan` 消费。

**★ 结构（主agent R45/R63/R74/R78 裁决）**
| 段 | 内容 | 臂数 |
|---|---|---|
| **P1** | 5 剂量 × 1 seed @ `T_min=10 s` | 5 |
| **P2** | T{10,20,40} s × dose{0,1,4} × 1 seed | 9 |
| **B**  | 4 轮 × 5 剂量 × 1 seed @ `T_min` 单档（杠杆搜索）| 20 |
| **C**  | 5 剂量 × 3 seed × 3 T（**判据单元**：含 zero_drive≥3 seed 与 baseline≥3 seed）| 45 |
| **D**  | 2 重复 × 3 T（`replica_of` 指向 C 的 dose=1/seed=0 臂）| 6 |
| 合计 | | **85** |

**口径**：`N=8,192`（N4 档）；**生产口径 `delivery="chunk"`**（R78：E 段剂量两端孪生臂 `pop_trace`
逐位相同 ⇒ 全网格可用 chunk）；**判据口径 `step`**（评估器只读产物，不跑仿真）；`settle=500 ms`；
`drive_rate_hz=8.0`（dose=1 的事件率）；`mechanism="none"`；`delay_bins=[40,60,80]`。
**角色规则（对齐 `validate_p11_resting.py`）**：`dose=0 → zero_drive`、`dose=1 → baseline`（硬门⑤要求
`mechanism=none ∧ drive_mult=1`）、其余 → `dose`、重复臂 → `replica`。
**启动形式（R70/R71 强制）**：整条驱动器须以 `caffeinate -i` 包裹（子进程随父持有防休眠断言 ⇒ 每臂等效带 caffeinate）。

用法：
    ./.venv-m9/bin/python -m neural_exploration.tools.make_m11_p5_plan_g1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "neural_exploration", "data")
N = 8192
T_MIN = 10_000.0          # 单位 = **ms**（干跑抓到：初版误写 10.0 ⇒ 实跑 T=10 ms）
TS = (10_000.0, 20_000.0, 40_000.0)
DOSES = (0.0, 0.5, 1.0, 2.0, 4.0)
SEEDS_C = (0, 1, 2)
DRIVE_RATE_HZ = 8.0
SETTLE_MS = 500.0
DELIVERY = "chunk"                     # 生产口径（R78）
PRODUCER = os.path.join(PROJECT_ROOT, "neural_exploration", "tools", "run_m11_p5_arms.py")
PRODUCER_SHA = "81f29d528f554ca2"      # R48 冻结记录（跑前不得再改）


def role_of(dose: float) -> str:
    if dose == 0.0:
        return "zero_drive"
    if dose == 1.0:
        return "baseline"
    return "dose"


def rid(stage: str, T_ms: float, dose: float, seed: int) -> str:
    return "%s_%d_T%dms_d%s_s%d" % (stage, N, int(T_ms), dose, seed)


def arm(stage: str, cfg: str, T_ms: float, dose: float, seed: int, role=None, replica_of=None):
    a = dict(run_id=rid(stage, T_ms, dose, seed), stage=stage, config_id=cfg,
             T_ms=float(T_ms), drive_mult=float(dose), seed=int(seed),
             role=(role or role_of(dose)), mechanism="none", delivery=DELIVERY)
    if replica_of:
        a["replica_of"] = replica_of
        a["role"] = "replica"
    return a


def build_plan() -> dict:
    arms = []
    # ---- P1：5 剂量 × 1 seed @ T_min ----
    for d in DOSES:
        arms.append(arm("p1", "p5_P1", T_MIN, d, 0))
    # ---- P2：T 阶梯 × dose{0,1,4} × 1 seed ----
    for T in TS:
        for d in (0.0, 1.0, 4.0):
            arms.append(arm("p2", "p5_P2", T, d, 0))
    # ---- B：4 轮杠杆搜索 × 5 剂量 @ T_min（每轮独立 config_id，避免与判据单元混）----
    for r in (1, 2, 3, 4):
        for d in DOSES:
            arms.append(arm("b_r%d" % r, "p5_B%d" % r, T_MIN, d, 0))
    # ---- C：判据单元（5 剂量 × 3 seed × 3 T）----
    for T in TS:
        for d in DOSES:
            for s in SEEDS_C:
                arms.append(arm("c", "p5_C", T, d, s))
    # ---- D：重复臂（每 T 两个，replica_of 指向 C 的 dose=1/seed=0 臂）----
    for T in TS:
        for rep in (1, 2):
            tgt = rid("c", T, 1.0, 0)
            a = arm("d", "p5_C", T, 1.0, 0, replica_of=tgt)
            a["run_id"] = "%s_rep%d" % (rid("d", T, 1.0, 0), rep)
            arms.append(a)
    ids = [a["run_id"] for a in arms]
    assert len(ids) == len(set(ids)), "run_id 必须唯一"
    plan = dict(
        plan_id="g1_star_n8192", structure="★（P1+P2+B4+C+D，R45/R63/R74/R78）",
        n_neuron=N, settle_ms=SETTLE_MS, dt_ms=0.05, drive_rate_hz=DRIVE_RATE_HZ,
        device="mps", bin_ms=100.0, smooth_frames=5, estimator="per_neuron",
        indegree=20, gmax_scale=1.0, bias_scale=1.0, net_seed=0, delay_bins=[40, 60, 80],
        subtypes=True, slow=True,
        production_delivery=DELIVERY, criterion_delivery="step",
        launch_prefix="caffeinate -i",
        judged_unit=dict(config_id="p5_C", note="唯一判据单元：每 T 均含 zero_drive(3 seed) 与 baseline(3 seed)"),
        frozen_tool=dict(path="neural_exploration/tools/run_m11_p5_arms.py",
                         sha256_prefix=PRODUCER_SHA, sha256_full=sha256_of(PRODUCER),
                         rule="R48：跑前不得再改工具"),
        arms=arms)
    return plan


def sha256_of(p: str) -> str:
    try:
        with open(p, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:  # noqa: BLE001
        return "unavailable"


def stats(plan: dict) -> dict:
    segs = {}
    for a in plan["arms"]:
        st = a["stage"]
        segs.setdefault(st, dict(n=0, steps=0))
        segs[st]["n"] += 1
        segs[st]["steps"] += int(round((plan["settle_ms"] + a["T_ms"]) / 0.05))
    tot_steps = sum(v["steps"] for v in segs.values())
    out = dict(segments=segs, n_arms=len(plan["arms"]), total_steps=tot_steps,
               n_zero_drive=len([a for a in plan["arms"] if a["role"] == "zero_drive"]),
               n_baseline=len([a for a in plan["arms"] if a["role"] == "baseline"]),
               n_dose=len([a for a in plan["arms"] if a["role"] == "dose"]),
               n_replica=len([a for a in plan["arms"] if a["role"] == "replica"]),
               judged_unit_seeds=dict(zero_drive=sorted({a["seed"] for a in plan["arms"]
                                                         if a["config_id"] == "p5_C"
                                                         and a["role"] == "zero_drive"}),
                                      baseline=sorted({a["seed"] for a in plan["arms"]
                                                       if a["config_id"] == "p5_C"
                                                       and a["role"] == "baseline"})))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default=os.path.join(DATA_DIR, "m11_p5_plan_g1.json"))
    args = ap.parse_args(argv)
    plan = build_plan()
    with open(args.out_json, "w") as f:
        json.dump(plan, f, indent=1, ensure_ascii=False)
    st = stats(plan)
    print("plan -> %s" % args.out_json)
    print("sha256 =", sha256_of(args.out_json)[:32])
    print(json.dumps(st, ensure_ascii=False, indent=1))
    print("冻结工具 sha256 =", sha256_of(PRODUCER)[:16], "（期望 %s）" % PRODUCER_SHA)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
