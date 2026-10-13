"""构建 **L0 杠杆扫描计划** `data/m11_p5_plan_l0.json`（P5 工作点搜索的第一段）。

**输入（预注册，跑前定稿）**：`data/m11_p5_lever_prereg.json`（B3 作者）。
本脚本**不改生产者**（`tools/run_m11_p5_arms.py` 冻结 sha256 `24617d90dd4a6eb9f518244808029ac3`），
只产出计划 JSON，由既有驱动器 `--plan` 消费。

**★ 结构（本轮任务书 + B3 预注册 + L0 内部优先级调整）**

| 档（`config_id`） | 杠杆取值 | 角色 | 臂 |
|---|---|---|---|
| `p5_L0_g1.00_b1.00` | 「参考档」gmax=1.00, bias=1.00（≡ G1 `p5_C` 工况） | zero_drive 4 seed + baseline 4 seed | 8 |
| `p5_L0_g1.00_b0.70` | **L0-b 主杠杆** bias=0.70 | 同上（+ dose=4 ×1 seed 供 (d)） | 9 |
| `p5_L0_g1.00_b0.50` | **L0-b 主杠杆** bias=0.50 | 同上（+ dose=4 ×1 seed 供 (d)） | 9 |
| `p5_L0_g2.00_b1.00` | L0-a 对照 gmax=2.00 | 同上 | 8 |
| `p5_L0_g4.00_b1.00` | L0-a 对照 gmax=4.00 | 同上 | 8 |
| 合计 | | | **42** |

**为什么是这两条杠杆（优先级已被主 agent 接受）**：(c) 自持比 R 在剂量维度上几乎不动（dose 0→4 时
中位率 ×8.1、而 R 仍在 0.28），故必须先动**内在兴奋性/复发项**：`bias_scale` 下调（主）＋`gmax_scale`
上调（对照；其预测机制是"抬基准臂而不动零驱动臂"，属输入侧，若其有效则说明自持来自输入侧）。

**判据口径**：评估器 `validate_p11_resting.py` 以 `(config_id, T_ms, mechanism)` 为判据单元
⇒ **每档一个 config_id**（杠杆取值写进 `config_id`，并被本脚本的断言逐臂核对）；每档都带
`role="zero_drive"`（**4 个真正独立 seed**）与 `role="baseline"` ⇒ (c) 可**逐档**复核。
生产口径 `chunk`（R78：端点孪生臂逐位相同）；判据口径 `step`（评估器只读产物，不跑仿真）。
`T_ms` = `T_min` = 10 s 单档（不做 T 阶梯）；`settle=500 ms` ⇒ 210,000 步/臂（≥200,000 下限）。

**复刻轴（R93 第 43 条）**：`net_seed/event_seed/reset_seed` 三源分别声明，**不设计划级 `net_seed`**
（否则恒定注入、seed 无法驱动网络）。除调用生产者自身的非构造性预检
`assert_plan_replication`（(net_seed,event_seed) 对数 == 不同 seed 数）外，本脚本追加一条
**对 dose=0 臂才算数**的强断言：每个判据单元内 `net_seed` 必须随 seed 一一对应
（否则零驱动臂只有 event_seed 在变 ⇒ 恒真通过）。

用法：
    ./.venv-m11/bin/python -m neural_exploration.tools.make_m11_p5_plan_l0            # 生成
    ./.venv-m11/bin/python -m neural_exploration.tools.make_m11_p5_plan_l0 --check F  # 只读校验
    ./.venv-m11/bin/python -m neural_exploration.tools.make_m11_p5_plan_l0 --self-test-negative
    ./.venv-m11/bin/python -m neural_exploration.tools.make_m11_p5_plan_l0 --verify-runs DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time

from neural_exploration.tools.make_m11_p5_plan_g1 import (ALLOWED_T_MS, EXPECTED_STEPS,
                                                          assert_plan_dimensions)
from neural_exploration.tools.run_m11_p5_arms import (NET_DEFAULTS, assert_plan_replication,
                                                      effective_seeds)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "neural_exploration", "data")
PRODUCER = os.path.join(PROJECT_ROOT, "neural_exploration", "tools", "run_m11_p5_arms.py")
PRODUCER_SHA = "eeeb2283cbf6533531ad1a4925b39a5d6815908714587675c2eba8ca612db2b1"
# ⚠️ **工具版本变更留痕（B2-L28）**：`data/m11_p5_plan_l0.json` 写于改动前、其 `frozen_tool.sha256_full`
#    记的是 `24617d90dd4a6eb9f518244808029ac3`。此后为 L1 加入 `--bias-log-sigma`（新 sha 见上），
#    **改动对不带该键的臂逐位惰性**（`tools/verify_m11_producer_inert.py` 实证：旧/新模块同 spec
#    `sha256_of_trace` 相同、且"不带键" ≡ "显式 0.30"）。变更记录见
#    `data/m11_p5_plan_l0_amendment.json`；**L0 计划文件本身未改**（J5 正在用，其 sha 引用有效）。
PREREG = os.path.join(DATA_DIR, "m11_p5_lever_prereg.json")

N = 8192
T_MS = 10_000.0                 # = T_min（ms）
SETTLE_MS = 500.0
DT_MS = 0.05
DRIVE_RATE_HZ = 8.0
DELIVERY = "chunk"              # 生产口径
SEEDS = (0, 1, 2, 3)            # 4 个独立实现（≥3；(c) 的 CI 需要）
MIN_INDEP_SEEDS = 3
ELASTICITY_DOSE = 4.0           # (d) 需要 drive_mult ∈ {1,4}；只给两个 L0-b 主档 1 个 seed
ELASTICITY_SEED = 0

# (config_id, tag, stage, gmax_scale, bias_scale, lever, with_elasticity)
LEVELS = (
    ("p5_L0_g1.00_b1.00", "g1.00_b1.00", "l0ref", 1.00, 1.00, "reference(≡G1 p5_C 工况)", False),
    ("p5_L0_g1.00_b0.70", "g1.00_b0.70", "l0b",   1.00, 0.70, "L0-b 主杠杆 bias↓", True),
    ("p5_L0_g1.00_b0.50", "g1.00_b0.50", "l0b",   1.00, 0.50, "L0-b 主杠杆 bias↓", True),
    ("p5_L0_g2.00_b1.00", "g2.00_b1.00", "l0a",   2.00, 1.00, "L0-a 对照 gmax↑", False),
    ("p5_L0_g4.00_b1.00", "g4.00_b1.00", "l0a",   4.00, 1.00, "L0-a 对照 gmax↑", False),
)
CFG_RE = re.compile(r"^p5_L0_g(\d+\.\d\d)_b(\d+\.\d\d)$")

# ---- L1（B3 设计，主 agent 2026-10-13 派工）：阈值异质性 σ 的形状杠杆 ----
# 基线 = **L0 的 bias_scale=0.70 档**（主 agent 指定）；gmax=1.00；σ ∈ {0.30 对照, 0.45, 0.60, 0.90}
# (config_id, tag, stage, bias_scale, gmax_scale, sigma, lever, with_elasticity)
L1_LEVELS = (
    ("p5_L1_b0.70_g1.00_s0.30", "b0.70_g1.00_s0.30", "l1ctl", 0.70, 1.00, 0.30, "L1 对照档 σ=0.30（=夹具默认）", True),
    ("p5_L1_b0.70_g1.00_s0.45", "b0.70_g1.00_s0.45", "l1s45", 0.70, 1.00, 0.45, "L1 σ=1.5× 对照", True),
    ("p5_L1_b0.70_g1.00_s0.60", "b0.70_g1.00_s0.60", "l1s60", 0.70, 1.00, 0.60, "L1 σ=2.0× 对照", True),
    ("p5_L1_b0.70_g1.00_s0.90", "b0.70_g1.00_s0.90", "l1s90", 0.70, 1.00, 0.90, "L1 σ=3.0× 对照（重尾上限）", True),
)
CFG_RE_L1 = re.compile(r"^p5_L1_b(\d+\.\d\d)_g(\d+\.\d\d)_s(\d+\.\d\d)$")
# B3 预注册 `L1_design.replicate_axis_convention.per_rung_seed_count`：每档 zero_drive/baseline
# **各 3 个真正独立实现**（realization0=(net0,ev11)、1=(1,12)、2=(2,13)）⇒ 4 档 × 2 角色 × 3 实现 = **24 run**
L1_SEEDS = (0, 1, 2)
L1_SEED_PAIRS = {0: (0, 11), 1: (1, 12), 2: (2, 13)}          # (net_seed, event_seed) 预注册对


def role_of(dose: float) -> str:
    """角色规则**逐字对齐** C 段（也是 `validate_p11_resting.py` 硬门⑤的读法）。"""
    if dose == 0.0:
        return "zero_drive"
    if dose == 1.0:
        return "baseline"
    return "dose"


def sha256_of(p: str) -> str:
    try:
        with open(p, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:  # noqa: BLE001
        return "unavailable"


def _rid(tag: str, dose: float, seed: int) -> str:
    return "l0_%s_%d_T%dms_d%g_s%d" % (tag, N, int(T_MS), dose, seed)


def build_plan() -> dict:
    arms, levels = [], []
    for cfg, tag, stage, gs, bs, lever, elast in LEVELS:
        m = CFG_RE.match(cfg)
        assert m, "config_id 必须形如 p5_L0_g<gmax>_b<bias>（杠杆取值写进 config_id）：%s" % cfg
        assert abs(float(m.group(1)) - gs) < 1e-9 and abs(float(m.group(2)) - bs) < 1e-9, \
            "config_id 内的杠杆取值与档定义不一致：%s / %s %s" % (cfg, gs, bs)
        rids = []
        for dose in (0.0, 1.0):
            for s in SEEDS:
                a = dict(run_id=_rid(tag, dose, s), stage=stage, config_id=cfg,
                         T_ms=T_MS, drive_mult=dose, seed=s, role=role_of(dose),
                         mechanism="none", delivery=DELIVERY,
                         gmax_scale=gs, bias_scale=bs, lever=lever)
                arms.append(a)
                rids.append(a["run_id"])
        if elast:
            a = dict(run_id=_rid(tag, ELASTICITY_DOSE, ELASTICITY_SEED), stage="l0d4", config_id=cfg,
                     T_ms=T_MS, drive_mult=ELASTICITY_DOSE, seed=ELASTICITY_SEED,
                     role=role_of(ELASTICITY_DOSE), mechanism="none", delivery=DELIVERY,
                     gmax_scale=gs, bias_scale=bs, lever=lever + " | (d) 弹性探针")
            arms.append(a)
            rids.append(a["run_id"])
        levels.append(dict(config_id=cfg, stage=stage, gmax_scale=gs, bias_scale=bs, lever=lever,
                           n_seeds=len(SEEDS), seeds=list(SEEDS), elasticity_arm=bool(elast),
                           run_ids=rids))
    ids = [a["run_id"] for a in arms]
    assert len(ids) == len(set(ids)), "run_id 必须唯一"
    plan = dict(
        plan_id="l0_lever_n8192",
        structure="L0 杠杆扫描（5 档 × {dose 0,1} × 4 seed + 2 个 (d) 弹性探针臂 = 42 臂）",
        parent_prereg=dict(path="neural_exploration/data/m11_p5_lever_prereg.json",
                           sha256_full=sha256_of(PREREG),
                           cited=dict(levers=["L0-a 输入通路增益（gmax_scale↑）",
                                              "L0-b 内在兴奋性（bias_scale↓）"],
                                      priority_override="主 agent 已接受：**L0-b 升为主杠杆**"
                                                        "（(c) 在剂量维度上不动 ⇒ 必须动内在/复发项），"
                                                        "L0-a 降为对照臂",
                                      stop_rules=["任一档出现 seed_dimension_inert ⇒ 立即停",
                                                  "L0-a/L0-b 网格内都不能把 R 压到 ≤0.10 ⇒ L0 失败停机",
                                                  "任一档 hard_failures 非空 ⇒ 停并查口径",
                                                  "总 GPU-h 触及包络 ⇒ 停（不得静默超支）"])),
        units={"T_ms": "millisecond", "settle_ms": "millisecond", "dt_ms": "millisecond",
               "wall_s": "second", "ms_per_step": "millisecond per step",
               "drive_rate_hz": "Hz per neuron", "gpu_h": "GPU-hour (=3.6e6 ms)",
               "bias_scale": "无量纲倍数（乘在偏置张量上）",
               "gmax_scale": "无量纲倍数（乘在突触权重上）",
               "note": "R80 §⑥：跨接口传递的量必须在接口定义处写明单位；"
                       "接收方写产物前必须做量纲自检"},
        n_neuron=N, settle_ms=SETTLE_MS, dt_ms=DT_MS, drive_rate_hz=DRIVE_RATE_HZ,
        device="mps", bin_ms=100.0, smooth_frames=5, estimator="per_neuron",
        indegree=NET_DEFAULTS["indegree"], gmax_scale=1.0, bias_scale=1.0,
        delay_bins=list(NET_DEFAULTS["delay_bins"]),
        # ⚠️ 无计划级 net_seed / event_seed（R93）：恒定注入会让 seed 无法驱动网络。
        seed_policy={"net_seed": "derive(=seed)", "event_seed": "derive(=11+seed)",
                     "reset_seed": "=seed",
                     "per_arm_explicit": False,
                     "rationale": "R93 第 43 条：复刻轴三源分别声明；seed=0 派生为 (net0/ev11/reset0)"
                                  " ⇒ 与历史常量一致，参考档 seed=0 与 G1 `p5_C` 同工况臂应逐位相同"},
        subtypes=True, slow=True,
        production_delivery=DELIVERY, criterion_delivery="step",
        launch_prefix="caffeinate -i",
        judged_units=[dict(config_id=l["config_id"], T_ms=T_MS, mechanism="none",
                           zero_drive_seeds=list(SEEDS), baseline_seeds=list(SEEDS),
                           note="评估器逐 (config_id,T,mechanism) 单元复核 (a)(b)(c)(e)；(d) 需 dose 4")
                      for l in levels],
        frozen_tool=dict(path="neural_exploration/tools/run_m11_p5_arms.py",
                         sha256_prefix=PRODUCER_SHA, sha256_full=sha256_of(PRODUCER),
                         rule="R48：本计划跑前不得再改生产者；改动即作废本计划的冻结记录"),
        cross_plan_anchor=dict(
            l0_arm=_rid("g1.00_b1.00", 1.0, 0), g1_arm="c_8192_T10000ms_d1.0_s0",
            rule="同规格（N=8192/T=10 s/settle=500 ms/dose=1/seed=0/gmax1.0/bias1.0/chunk）"
                 "⇒ 两计划产物的 `sha256_of_trace` 必须**逐位相同**（跨计划可复现锚点；"
                 "不同即说明工具或口径被改过）"),
        levels=levels, arms=arms)
    return plan


def build_plan_l1(with_elasticity: bool = False, seeds=L1_SEEDS) -> dict:
    """**L1 计划**：σ 四档 × {dose 0, 1} × **3 个独立实现** = **24 臂**（B3 预注册 `L1_design` 的
    `per_rung_arms`：每档 zero_drive 与 baseline 各 3 个真正独立实现；`(net_seed, event_seed)` 对
    两两互异）。`with_elasticity=True` 时每档追加 1 个 dose=4 探针（+4 臂，供 (d) sanity，网格登记为不全）。

    与 L0 的两点结构差异：①`config_id` 形如 `p5_L1_b<bias>_g<gmax>_s<sigma>`（σ 也进 id）；
    ②每臂**必须显式带 `bias_log_sigma`**（含对照档 0.30）——由 `assert_lever_structure` 强制，
    以堵"键被接受但没进计划/没生效"的静默失效（M10 L21、D2 同族）。基线 bias_scale=0.70、gmax_scale=1.00。
    """
    arms, levels = [], []
    for cfg, tag, stage, bs, gs, sg, lever, elast in L1_LEVELS:
        elast = bool(elast and with_elasticity)
        m = CFG_RE_L1.match(cfg)
        assert m and abs(float(m.group(1)) - bs) < 1e-9 and abs(float(m.group(2)) - gs) < 1e-9 \
            and abs(float(m.group(3)) - sg) < 1e-9, "L1 config_id 与档定义不一致：%s" % cfg
        rids = []
        for dose in (0.0, 1.0):
            for sd in seeds:
                a = dict(run_id="l1_%s_%d_T%dms_d%g_s%d" % (tag, N, int(T_MS), dose, sd),
                         stage=stage, config_id=cfg, T_ms=T_MS, drive_mult=dose, seed=sd,
                         role=role_of(dose), mechanism="none", delivery=DELIVERY,
                         gmax_scale=gs, bias_scale=bs, bias_log_sigma=sg, lever=lever)
                arms.append(a)
                rids.append(a["run_id"])
        if elast:
            a = dict(run_id="l1_%s_%d_T%dms_d%g_s%d" % (tag, N, int(T_MS), ELASTICITY_DOSE,
                                                       ELASTICITY_SEED),
                     stage="l1d4", config_id=cfg, T_ms=T_MS, drive_mult=ELASTICITY_DOSE,
                     seed=ELASTICITY_SEED, role=role_of(ELASTICITY_DOSE), mechanism="none",
                     delivery=DELIVERY, gmax_scale=gs, bias_scale=bs, bias_log_sigma=sg,
                     lever=lever + " | (d) 弹性探针")
            arms.append(a)
            rids.append(a["run_id"])
        levels.append(dict(config_id=cfg, stage=stage, gmax_scale=gs, bias_scale=bs,
                           bias_log_sigma=sg, lever=lever, n_seeds=len(seeds), seeds=list(seeds),
                           net_event_pairs={str(k): list(v) for k, v in
                                            L1_SEED_PAIRS.items() if k in set(seeds)},
                           elasticity_arm=bool(elast), run_ids=rids))
    ids = [a["run_id"] for a in arms]
    assert len(ids) == len(set(ids)), "run_id 必须唯一"
    plan = dict(
        plan_id="l1_sigma_n8192",
        structure=("L1 σ 形状杠杆扫描（4 档 × {dose 0,1} × %d 独立实现%s = %d 臂；"
                   "B3 预注册 L1_design）" % (len(seeds),
                                            " + %d 个 (d) 探针" % len(L1_LEVELS) if with_elasticity else "",
                                            len(arms))),
        parent_prereg=dict(path="neural_exploration/data/m11_p5_lever_prereg.json",
                           sha256_full=sha256_of(PREREG),
                           cited=dict(lever="L1 阈值异质性（bias_log_sigma↑）",
                                      design="B3 L1 设计（已验收）；σ∈{0.30 对照,0.45,0.60,0.90}，"
                                             "基线 bias_scale=0.70 / gmax_scale=1.00（主 agent 2026-10-13 派工）",
                                      hypothesis="(a) 偏低与 (b) 偏低同因：窄分布 ⇒ 加宽阈值分布应同时把"
                                                 "中位率与静默比例往目标推；n_both_in_band ≥1 才算支撑",
                                      stop_rules=["扫描后 n_both_in_band 仍为 0 ⇒ 判张力结构性（R91）",
                                                  "任一档 seed_dimension_inert ⇒ 立即停",
                                                  "任一档 hard_failures 非空 ⇒ 停并查口径"])),
        units={"T_ms": "millisecond", "settle_ms": "millisecond", "dt_ms": "millisecond",
               "wall_s": "second", "ms_per_step": "millisecond per step",
               "drive_rate_hz": "Hz per neuron", "gpu_h": "GPU-hour (=3.6e6 ms)",
               "bias_scale": "无量纲倍数（乘在偏置张量上）",
               "gmax_scale": "无量纲倍数（乘在突触权重上）",
               "bias_log_sigma": "无量纲（对数正态宽度；乘在标准正态分量上，不改 RNG 序列）",
               "note": "R80 §⑥：跨接口传递的量必须在接口定义处写明单位；接收方写产物前必须做量纲自检"},
        n_neuron=N, settle_ms=SETTLE_MS, dt_ms=DT_MS, drive_rate_hz=DRIVE_RATE_HZ,
        device="mps", bin_ms=100.0, smooth_frames=5, estimator="per_neuron",
        indegree=NET_DEFAULTS["indegree"], gmax_scale=1.0, bias_scale=0.70,
        bias_log_sigma=0.30, delay_bins=list(NET_DEFAULTS["delay_bins"]),
        seed_policy={"net_seed": "derive(=seed)", "event_seed": "derive(=11+seed)",
                     "reset_seed": "=seed", "per_arm_explicit": True,
                     "rationale": "R93 第 43 条：复刻轴三源分别声明；L1 每臂显式带 bias_log_sigma"},
        subtypes=True, slow=True,
        production_delivery=DELIVERY, criterion_delivery="step",
        launch_prefix="caffeinate -i",
        judged_units=[dict(config_id=l["config_id"], T_ms=T_MS, mechanism="none",
                           zero_drive_seeds=list(seeds), baseline_seeds=list(seeds),
                           bias_log_sigma=l["bias_log_sigma"],
                           net_event_pairs=l["net_event_pairs"],
                           note="评估器逐 (config_id,T,mechanism) 复核 (a)(b)(c)(e)；(d) 需 dose 4")
                      for l in levels],
        prerequisites=dict(
            cli_exposure="生产者 `--bias-log-sigma`（B2 已加；不显式给 ⇒ 沿用夹具默认 0.30）",
            positive_control_required=("**跑前必须过** `--self-test-sigma`（R84 第 37 条：未证伪过的断言不构成"
                                       "证据）⇒ 产物 `neural_exploration/data/m11_lever_sigma_selftest.json`；"
                                       "C1 杠杆生效 + C2 反例自证 + C3 默认语义保持，三者齐备"),
            producer_amendment="本计划要求生产者 sha256 = %s（含 `--bias-log-sigma`）；"
                               "L0 计划的 frozen_tool 记的是改动前 sha（见 data/m11_p5_plan_l0_amendment.json）"
                               % PRODUCER_SHA[:16]),
        lever_prereg_falsifier="σ 扫描下 (中位率, 静默) 云团若不朝可行角移动 ⇒ 判张力结构性，停机并写入"
                               "缺失机制清单（需换机制而非调参数）",
        frozen_tool=dict(path="neural_exploration/tools/run_m11_p5_arms.py",
                         sha256_prefix=PRODUCER_SHA, sha256_full=sha256_of(PRODUCER),
                         rule="R48：本计划跑前不得再改生产者；改动即作废本冻结记录"),
        cross_plan_anchor=dict(
            l1_control_arm="l1_b0.70_g1.00_s0.30_8192_T10000ms_d1_s0",
            l0_arm="l0_g1.00_b0.70_8192_T10000ms_d1_s0",
            rule="σ=0.30 是夹具默认值 ⇒ L1 对照档 seed=0 的 dose=1 臂应与 L0 `p5_L0_g1.00_b0.70` "
                 "同 seed 臂**逐位相同**（跨计划可复现锚点；不同即说明 σ 键改了行为）"),
        levels=levels, arms=arms)
    return plan


def _cfg_expect(cfg: str):
    """从 `config_id` 解析该档**声明的杠杆取值**（解析不出 ⇒ None ⇒ 断言判失败）。

    L0：`p5_L0_g<gmax>_b<bias>`；L1：`p5_L1_b<bias>_g<gmax>_s<sigma>`。
    """
    m = CFG_RE.match(cfg)
    if m:
        return dict(gmax_scale=float(m.group(1)), bias_scale=float(m.group(2)))
    m = CFG_RE_L1.match(cfg)
    if m:
        return dict(bias_scale=float(m.group(1)), gmax_scale=float(m.group(2)),
                    bias_log_sigma=float(m.group(3)))
    return None


def assert_l0_structure(plan: dict) -> dict:
    """**L0 结构断言（非可选；可证伪）**：① 每档 `config_id` 内的杠杆取值 == 逐臂声明；
    ② 每档 `zero_drive` ≥3 个不同 seed 且 `baseline` 存在；③ 角色规则 == 剂量规则；
    ④ 判据单元内 `net_seed` 随 seed 一一对应（**dose=0 臂专用，堵"只有 event_seed 在变"的恒真洞**）。
    失败 ⇒ SystemExit(4)/(5)。
    """
    fails_cfg, fails_seed, rows = [], [], []
    by_unit, by_cfg = {}, {}
    for a in plan["arms"]:
        by_cfg.setdefault(a["config_id"], []).append(a)
        by_unit.setdefault((a["config_id"], float(a["T_ms"]), a.get("mechanism", "none")), []).append(a)
        if a["role"] != role_of(float(a["drive_mult"])):
            fails_cfg.append("%s: role=%s 与 dose=%s 不符（角色规则必须逐字对齐 C 段）"
                             % (a["run_id"], a["role"], a["drive_mult"]))
    for cfg, lst in sorted(by_cfg.items()):
        exp = _cfg_expect(cfg)
        if exp is None:
            fails_cfg.append("config_id 未编码杠杆取值：%s" % cfg)
            continue
        for a in lst:
            for k, v in exp.items():
                if k not in a:
                    fails_cfg.append("%s: config_id(%s) 要求 %s=%s，但该臂**没有这个键**"
                                     "（杠杆静默失效形态）" % (a["run_id"], cfg, k, v))
                elif abs(float(a[k]) - v) > 1e-9:
                    fails_cfg.append("%s: 臂声明 %s=%s ≠ config_id(%s) 的 %s=%s"
                                     % (a["run_id"], k, a[k], cfg, k, v))
        zs = {a["seed"] for a in lst if a["role"] == "zero_drive"}
        bs_seeds = {a["seed"] for a in lst if a["role"] == "baseline"}
        rows.append((cfg, exp.get("gmax_scale"), exp.get("bias_scale"),
                     len(zs), len(bs_seeds), len(lst)))
        if len(zs) < MIN_INDEP_SEEDS:
            fails_seed.append("%s: zero_drive 独立 seed 数 %d < %d（(c) 的 CI 会失效）"
                              % (cfg, len(zs), MIN_INDEP_SEEDS))
        if not bs_seeds:
            fails_seed.append("%s: 缺 role=baseline（drive_mult=1）臂 ⇒ (c) 无法逐档复核" % cfg)
        if len(bs_seeds) < MIN_INDEP_SEEDS and bs_seeds:
            fails_seed.append("%s: baseline 独立 seed 数 %d < %d" % (cfg, len(bs_seeds), MIN_INDEP_SEEDS))
    # ---- dose=0 专用强断言：net_seed 必须随 seed 一一对应（event_seed 对零驱动臂是惰性的）----
    fails_net = []
    for key, lst in sorted(by_unit.items()):
        pairs = {}
        for a in lst:
            e = effective_seeds(a)
            pairs.setdefault(int(e["net_seed"]), set()).add(int(a["seed"]))
        multi = {k: sorted(v) for k, v in pairs.items() if len(v) > 1}
        seeds = {int(a["seed"]) for a in lst}
        if len(seeds) > 1 and (len(pairs) < len(seeds) or multi):
            fails_net.append("%s T=%s dose=%s: %d 个 seed 只映射到 %d 个 net_seed %r"
                             % (key[0], key[1], key[2], len(seeds), len(pairs), multi))
    print("  %-20s %-6s %-6s %8s %10s %6s" % ("config_id", "gmax", "bias", "n_zero", "n_baseline", "n_arm"))
    for r in rows:
        print("  %-20s %-6.2f %-6.2f %8d %10d %6d" % r)
    if fails_cfg:
        print("**L0 结构断言失败（config_id ↔ 杠杆取值 / 角色规则，%d 条）**：" % len(fails_cfg))
        for x in fails_cfg[:10]:
            print("   -", x)
        raise SystemExit(4)
    if fails_seed:
        print("**L0 结构断言失败（seed 独立性，%d 条）**：" % len(fails_seed))
        for x in fails_seed[:10]:
            print("   -", x)
        raise SystemExit(5)
    if fails_net:
        print("**L0 结构断言失败（net_seed 未随 seed 变化，%d 条）**：" % len(fails_net))
        for x in fails_net[:10]:
            print("   -", x)
        raise SystemExit(5)
    print("杠杆结构断言通过：%d 档 × (zero_drive/baseline ≥%d 独立实现)；config_id 已编码杠杆取值；"
          "判据单元内 net_seed 一一对应" % (len(by_cfg), MIN_INDEP_SEEDS))
    return dict(n_configs=len(by_cfg), n_units=len(by_unit), n_arms=len(plan["arms"]))


assert_lever_structure = assert_l0_structure      # 通用别名：L0/L1 共用同一实现


def stats(plan: dict) -> dict:
    segs, zero, base, dose_arms = {}, 0, 0, 0
    for a in plan["arms"]:
        steps = int(round((plan["settle_ms"] + a["T_ms"]) / plan["dt_ms"]))
        v = segs.setdefault(a["stage"], dict(n=0, steps=0))
        v["n"] += 1
        v["steps"] += steps
        zero += (a["role"] == "zero_drive")
        base += (a["role"] == "baseline")
        dose_arms += (a["role"] == "dose")
    return dict(segments=segs, n_arms=len(plan["arms"]),
                total_steps=sum(v["steps"] for v in segs.values()),
                n_zero_drive=zero, n_baseline=base, n_dose=dose_arms,
                steps_per_arm=int(round((plan["settle_ms"] + T_MS) / plan["dt_ms"])))


def verify_runs(plan: dict, out_dir: str) -> int:
    """跑后核对：每臂产物的 `net.gmax_scale/bias_scale/net_seed` 与计划声明一致（防杠杆静默失效）。"""
    bad, seen = [], 0
    for a in plan["arms"]:
        p = os.path.join(out_dir, a["run_id"] + ".json")
        if not os.path.exists(p):
            bad.append("%s：缺产物（%s）" % (a["run_id"], p))
            continue
        with open(p) as f:
            got = json.load(f)
        seen += 1
        e = effective_seeds(a)
        checks = [("net.gmax_scale", got["net"]["gmax_scale"], float(a["gmax_scale"])),
                  ("net.bias_scale", got["net"]["bias_scale"], float(a["bias_scale"])),
                  ("net.net_seed", got["net"]["net_seed"], int(e["net_seed"])),
                  ("event_seed", got["event_seed"], int(e["event_seed"])),
                  ("role", got["role"], a["role"]),
                  ("config_id", got["config_id"], a["config_id"]),
                  ("delivery", got["delivery"], a["delivery"]),
                  ("T_ms", float(got["T_ms"]), float(a["T_ms"]))]
        for name, g, w in checks:
            if g != w:
                bad.append("%s: %s = %r ≠ 计划声明 %r" % (a["run_id"], name, g, w))
    print("跑后核对：%d/%d 臂产物存在" % (seen, len(plan["arms"])))
    if bad:
        print("**跑后核对失败（%d 条）**：" % len(bad))
        for x in bad[:10]:
            print("   -", x)
        return 6
    print("跑后核对通过：逐臂杠杆取值/复刻轴三元组/角色/口径与计划一致")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", default="l0", choices=("l0", "l1"),
                    help="l0=杠杆扫描（bias_scale/gmax_scale）；l1=σ 形状杠杆扫描（bias_log_sigma）")
    ap.add_argument("--out-json", default=None,
                    help="默认按 --section 落到 data/m11_p5_plan_<section>.json")
    ap.add_argument("--check", default=None, help="**只读**校验既有计划（不改文件；段由 plan_id 自识别）")
    ap.add_argument("--l1-seeds", default="0,1,2",
                    help="L1：独立实现号（预注册 0,1,2 ⇒ 24 臂；给 0,1,2,3 ⇒ 32 臂，CI 更稳、更贵）")
    ap.add_argument("--l1-elasticity", action="store_true",
                    help="L1：每档追加 1 个 dose=4 探针（+4 臂，供 (d) sanity；B3 预注册默认不跑）")
    ap.add_argument("--self-test-negative", action="store_true",
                    help="反例自检：故意做坏的计划必须各自触发断言（L0 4 例 + L1 2 例）")
    ap.add_argument("--verify-runs", default=None, help="跑后核对产物目录")
    ap.add_argument("--diff-against", default=None,
                    help="**只读**：把磁盘上既有计划与刚构建的计划做**逐字段 diff**（证明『只有 provenance "
                         "字段变了、臂网格没变』，用于工具/预注册变更后的留痕）")
    ap.add_argument("--amendment-out", default=None,
                    help="配合 --diff-against：把 diff + 工具变更 + 惰性证据写成 amendment JSON")
    args = ap.parse_args(argv)
    if args.out_json is None:
        args.out_json = os.path.join(DATA_DIR, "m11_p5_plan_%s.json" % args.section)
    l1_seeds = tuple(int(x) for x in str(args.l1_seeds).split(",") if x.strip() != "")
    builder = (build_plan if args.section == "l0"
               else (lambda: build_plan_l1(with_elasticity=bool(args.l1_elasticity),
                                           seeds=l1_seeds)))

    if args.diff_against:
        with open(args.diff_against) as f:
            live = json.load(f)
        fresh = builder()

        def flat(d, pre=""):
            out = {}
            if isinstance(d, dict):
                for k, v in d.items():
                    out.update(flat(v, pre + "/" + str(k)))
            elif isinstance(d, list):
                for i, v in enumerate(d):
                    out.update(flat(v, pre + "/%d" % i))
            else:
                out[pre] = d
            return out

        fa, fb = flat(fresh), flat(live)
        diffs = [(k, fa.get(k, "<缺>"), fb.get(k, "<缺>"))
                 for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)]
        print("== 计划 diff：刚构建 vs 磁盘在用（%s）==" % args.diff_against)
        print("  字段差异 %d 处｜臂数 %d vs %d｜**臂内容完全相同 = %s**"
              % (len(diffs), len(fresh["arms"]), len(live["arms"]),
                 fresh["arms"] == live["arms"]))
        for k, x, y in diffs[:12]:
            print("   - %s\n       重构建: %s\n       在用  : %s" % (k, str(x)[:90], str(y)[:90]))
        if args.amendment_out:
            am = dict(kind="计划冻结记录修订留痕（不改动原计划文件）",
                      generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                      live_plan=dict(path=os.path.relpath(args.diff_against, PROJECT_ROOT),
                                     sha256_full=sha256_of(args.diff_against)),
                      regenerated_plan=dict(sha256_full=sha256_of(args.out_json)
                                            if os.path.exists(args.out_json) else None,
                                            note="仅在 --out-json 指向真实新文件时有值"),
                      arm_grid_identical=(fresh["arms"] == live["arms"]),
                      n_arms=[len(live["arms"]), len(fresh["arms"])],
                      field_diffs=[dict(field=k, regenerated=str(x), live=str(y)) for k, x, y in diffs],
                      interpretation="**臂网格与全部判据参数逐字节相同**；差异只在 provenance "
                                     "字段（生产者 sha / 预注册 sha）⇒ 在用计划仍是 J5 的有效输入",
                      producer_change=dict(pre="24617d90dd4a6eb9f518244808029ac3",
                                           post=sha256_of(PRODUCER),
                                           what="加入 `--bias-log-sigma`（L1 阻塞项）+ `--self-test-sigma`",
                                           inertness_evidence="neural_exploration/data/m11_producer_inert_check.json",
                                           inertness_claim="不带该键的臂逐位不变；不带键 ≡ 显式 0.30"))
            with open(args.amendment_out, "w") as f:
                json.dump(am, f, indent=1, ensure_ascii=False)
            print("  修订留痕 -> %s" % args.amendment_out)
        return 0

    if args.check:
        with open(args.check) as f:
            plan = json.load(f)
        print("== 只读校验：%s（sha256 %s）==" % (args.check, sha256_of(args.check)[:32]))
        assert_l0_structure(plan)
        assert_plan_replication(plan)
        assert_plan_dimensions(plan)
        return 0

    if args.self_test_negative:
        print("== 反例自检（每个坏计划都必须被**至少一条**断言拦下）==")
        # ① T 单位 bug（R80：秒被当毫秒）
        p1 = build_plan()
        p1["arms"][0] = dict(p1["arms"][0], T_ms=10.0, run_id="NEG_CTRL_T10ms")
        p1["arms"][1] = dict(p1["arms"][1], T_ms=10.0, run_id="NEG_CTRL_T10ms_b")
        # ② 复刻轴恒定：显式给所有臂 net_seed=0（历史缺陷的"半退化"形态）
        p2 = build_plan()
        p2["arms"] = [dict(a, net_seed=0) for a in p2["arms"]]
        # ③ config_id 与杠杆取值不符（杠杆静默失效形态）
        p3 = build_plan()
        p3["arms"][5] = dict(p3["arms"][5], gmax_scale=2.0)
        # ④ zero_drive 只有 2 个独立 seed（复刻不足形态；注意 3 个 seed 是**满足**下限的）
        p4 = build_plan()
        p4["arms"] = [a for a in p4["arms"]
                      if not (a["role"] == "zero_drive" and a["seed"] in (2, 3))]
        # ⑤⑥ L1 专有反例：σ 与 config_id 不符 / 臂**缺 σ 键**（键被接受但没进计划）
        p5 = build_plan_l1()
        p5["arms"][3] = dict(p5["arms"][3], bias_log_sigma=0.99)
        p6 = build_plan_l1()
        p6["arms"][5] = {k: v for k, v in p6["arms"][5].items() if k != "bias_log_sigma"}
        cases = (("①T 单位 bug", p1, [("量纲断言", assert_plan_dimensions)]),
                 ("②恒定 net_seed", p2, [("生产者复刻轴预检", assert_plan_replication),
                                          ("L0 结构断言(net_seed 轴)", assert_l0_structure)]),
                 ("③config_id 与杠杆不符", p3, [("L0 结构断言", assert_l0_structure)]),
                 ("④zero_drive seed<3", p4, [("L0 结构断言", assert_l0_structure)]),
                 ("⑤L1 σ 与 config_id 不符", p5, [("杠杆结构断言", assert_lever_structure)]),
                 ("⑥L1 臂缺 σ 键", p6, [("杠杆结构断言", assert_lever_structure)]))
        caught, notes = [], []
        for name, pl, fns in cases:
            hit = []
            for fn_name, fn in fns:
                try:
                    fn(pl)
                    print("   · %s｜%s：未触发" % (name, fn_name))
                    if (name.startswith("②")) and fn_name.startswith("生产者"):
                        notes.append("**已记录的残余缺口**：生产者 `assert_plan_replication` 以 "
                                     "(net_seed,event_seed) **对**为键 ⇒ 恒定 net_seed + 派生 "
                                     "event_seed 可溜过；本脚本的 net_seed 轴断言补上（dose=0 时 "
                                     "event_seed 惰性，只有 net_seed 是真轴）")
                except SystemExit as ex:
                    print("   ✓ %s｜%s：触发（exit %s）" % (name, fn_name, ex.code))
                    hit.append(fn_name)
            if hit:
                caught.append(name)
            else:
                print("   ✗ %s：**全部断言未触发（不可证伪）**" % name)
        for n in notes:
            print("   ⚠ " + n)
        if len(caught) != len(cases):
            print("**反例自检失败：只有 %d/%d 坏计划被拦下**" % (len(caught), len(cases)))
            return 3
        print("反例自检通过：%d/%d 坏计划至少被一条断言拦下（断言可证伪）" % (len(caught), len(cases)))
        return 0

    plan = builder()
    if args.verify_runs:
        return verify_runs(plan, args.verify_runs)

    print("== ①杠杆结构断言（写出前；段=%s）==" % args.section)
    assert_lever_structure(plan)
    print("== ②复刻轴预检（生产者自带，非构造性）==")
    assert_plan_replication(plan)
    print("== ③量纲自检（写出前；R80）==")
    assert_plan_dimensions(plan)
    with open(args.out_json, "w") as f:
        json.dump(plan, f, indent=1, ensure_ascii=False)
    st = stats(plan)
    print("plan -> %s" % args.out_json)
    print("sha256 =", sha256_of(args.out_json))
    print(json.dumps(st, ensure_ascii=False, indent=1))
    print("冻结生产者 sha256 =", sha256_of(PRODUCER)[:16], "（期望 %s）" % PRODUCER_SHA)
    print("预注册 sha256 =", sha256_of(PREREG)[:16])
    print("写出时间 =", time.strftime("%Y-%m-%dT%H:%M:%S"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
