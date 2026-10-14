"""构建 **L3 计划（网格 v2：σ 悬崖面扫描）** `data/m11_p5_plan_l3_replan.json`（P5 工作点搜索第三段的**重规划版**）。

**⚠️ 落盘路径纪律（本节点实付的学费 + R122 作废纪律，必须保留）**
1. 仓库内**已存在** `data/m11_p5_plan_l3.json`（**tracked**，sha256 `4aab49b00177a22e21f65b884bd29d95a8e7333e76cb109ae8fcd8bb9bd94615`，30 臂 v2）
   与**已被 R119 用于正式跑**的冻结副本；本脚本用 `FORBIDDEN_OUT_PATHS` **拒绝**写入这两个路径。
2. 本脚本上次产出的旧版本（4 格 24 臂：`(1.00,0.30)/(1.30,0.30)/(1.30,0.20)/(1.30,0.15)`，
   sha256 `539325a13aa201cd5bc480d0b941768fb452ef2e36caccd3c90fba29ff1af98b`）已被 **R122 §④ 明文判「不得运行」**
   （探针已在同一角落实测：σ=0.30/0.20/0.15 的 `R` ≈ 0.30/0.879/0.881 ⇒ (c) 全 fail）
   ⇒ **本脚本就地用 v2 覆盖它**，并在计划 JSON 顶部写死 `status` + `supersedes`（纪律第六十三条：
   被取代的计划不得原样留在可执行位置；「过期」必须显式表达，不得只存在于记忆里）。

**设计依据（本次全部依据）**
- **`data/m11_prereg_amend_r118_l3_grid_v2.json`**（事前修订；父预注册逐字节不变）与 **`docs/m11_acceptance_master.md` §3di（R118）**；
  另参照 §3dc–§3dg（R112–R116，工具修好与窗口陷阱）、§3dj（R119，正式跑启动）、§3dl（R121，正式口径首臂读数）。
- **R118 ①**：`bias=1.30` 下 σ 是**悬崖**不是梯度（零驱动脉冲 σ=0.30/0.20/0.15 ⇒ 4,126/4,875/4,666，σ=0.10 ⇒ **46**，0.15→0.10 落差 **101×**）
  ⇒ R115 §④ 登记的 `σ∈[0.15,0.25]` 整段**远在 (c) 窗口之外**（**已测且排除**，不是未测）⇒ **窗口只在 `σ=0.10` 处可达**。
- **R115 §②（窗口口径，逐字沿用）**：`R = pooled_zero_drive/pooled_baseline ≤ 0.10` **且** `pooled_zero_drive ≥ 300`（硬门①）
  ⇒ **目标是窗口不是极限**；`R` 太小（零驱动被打干净）会让 (c) 变 `unavailable` 而不是 `pass`
  ⇒ **每档必须报 `pooled_zero_drive` 绝对值**。
- **R115 §③（预注册禁止，逐字沿用）**：不得提高背景驱动去命中 (a)（`docs/m11_b3_g1_design.md` **L146**）
  ⇒ 本计划 30 臂的 `drive_mult` 只有 0.0/1.0，**无任何以驱动抬升为标定手段的档**。
- **R119 §①**：本计划沿用冻结计划的 `plan_id`（`p5_L3_sigma_cliff_bias_scan`）与**逐臂 `config_id`/`run_id`**
  ⇒ 与正在跑的产物**同一命名空间**（已完成的臂可被 `[SKIP]` 复用，评估/报告命令不必改写）。

**★ 结构（网格 v2：5 格 × 2 角色 × 3 独立实现 = 30 臂；T=10 s；settle=500 ms ⇒ 210,000 步/臂）**

| 顺序 | 格 | `config_id` | `bias_scale` | `bias_log_sigma` | 备注 |
|---|---|---|---|---|---|
| 1 | **A0（锚）** | `p5_L3_A0_b1.00_s0.30` | 1.00 | 0.30 | = 既有 L0/L1 同配置点（确定性核对，照 R95 §⑤ 标 `h_determinism_not_evidence`）|
| 2 | C0 | `p5_L3_C0_b1.00_s0.10` | 1.00 | 0.10 | 已知 (a) 失败、零驱动=0 ⇒ (c) 预期 `unavailable` |
| 3 | C1 | `p5_L3_C1_b1.15_s0.10` | 1.15 | 0.10 | σ=0.10 偏置扫描 1 |
| 4 | **C2（主目标）** | `p5_L3_C2_b1.30_s0.10` | 1.30 | 0.10 | ★探针暂定同落 (a)/(c)/(e) 三带（R118 §③）|
| 5 | C3 | `p5_L3_C3_b1.45_s0.10` | 1.45 | 0.10 | 极端档（爆炸风险）⇒ **最后跑** |

**排序**：主 agent v2 规格 = `A0 锚 → C0 → C1 → C2 → C3`；**每格内先 `baseline` 后 `zero_drive`**。
⚠️ 与**冻结计划**的差别已登记：冻结计划把 `C2` 提到最前（理由：本机占空比 ≈5% ⇒ 30 臂 ≈68 h，决定性格必须先跑，
见 R119 §①）；本版按主 agent 最新派工改为**锚点最先、C2 第四**（`execution_order.conflict_with_frozen_order` 内逐字登记）。

**复刻轴（R93）**：每臂**显式**带 `net_seed`/`event_seed`（`(0,11)/(1,12)/(2,13)`）；**不设**计划级 `net_seed`。
生产者非构造性预检 `assert_plan_replication` 在 `--plan` 模式**跑前**执行（L548）。

**关键断言（R80 家族，本脚本强制，失败即非零退出）**
① 步数恒等式 `round((settle+T)/dt) == 210000`（≥200,000 下限）；② **30 臂**（5 格 × 2 角色 × 3 实现）、`run_id` 唯一、
`(config_id, role, seed)` 唯一；③ 每格每角色 3 实现的 `(net_seed,event_seed)` 两两互异且恰等于预注册对表；
④ 每臂**恰好** 24 个必需键，顶层键集 ⊇ `m11_p5_plan_l1.json` 键集（差集空）；⑤ 5 格的 `bias_scale`/`bias_log_sigma`
与 v2 表逐字一致（并从 `config_id` 反解核对）；⑥ `drive_rate_hz == 8.0` 且无驱动抬升档；
⑦ 生产者 sha `633a4c45…`（R48）+ 逐键比对实读点；⑧ **drop-in 断言**：本计划的 `(run_id, config_id, role, seed)` 集合
与冻结计划 `m11_p5_plan_l3.json` **完全相同**（只改臂顺序，不改档位/实现/协议字段 ⇒ R118 修订的网格不受影响）。

用法：
    python3 -m neural_exploration.tools.make_m11_p5_plan_l3            # 生成（写 m11_p5_plan_l3_replan.json）
    python3 -m neural_exploration.tools.make_m11_p5_plan_l3 --check    # 只读校验（与磁盘逐字节比对 + 重跑全部断言）
    python3 -m neural_exploration.tools.make_m11_p5_plan_l3 --keys     # 只打印生产者键读取比对表
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "neural_exploration", "data")
DOCS_DIR = os.path.join(PROJECT_ROOT, "neural_exploration", "docs")
PRODUCER = os.path.join(PROJECT_ROOT, "neural_exploration", "tools", "run_m11_p5_arms.py")

# ---- 冻结的生产者（R48；R116 验收后的 sha）----
PRODUCER_SHA = "633a4c458e9c1dd58e5d6c4838a205819a9424fbd49df9da8827c8b7bd42285b"
PRODUCER_SHA16 = PRODUCER_SHA[:16]
PRODUCER_LINES = 894
PRODUCER_BYTES = 52659

L1_PLAN = os.path.join(DATA_DIR, "m11_p5_plan_l1.json")
L1_PLAN_SHA = "8b58577222038ff91d4550a94101b09fcaa4a6d1bdfaac4d9f3451d8b36366cf"
LEVER_PREREG = os.path.join(DATA_DIR, "m11_p5_lever_prereg.json")
LEVER_PREREG_SHA = "9c6f7c60073ee07e1f48aee787040f16d24aa0efaf67b4cf65f98d2e263c6644"
R109_AMEND = os.path.join(DATA_DIR, "m11_p5_prereg_amend_r109_a_reporting.json")
R109_AMEND_SHA = "aa039f67403d1ba6ba1873143a6c6eb6ffbb50ebe4ab6d4b6800af8ac0b89fb8"
TARGETS_CSV = os.path.join(DATA_DIR, "m11_p5_targets.csv")
R118_AMEND = os.path.join(DATA_DIR, "m11_prereg_amend_r118_l3_grid_v2.json")
R118_AMEND_SHA = "942056b5f5b02d504069a704d3d65bead599b2975f6253e67c474987b0ca54e4"
FROZEN_L3_PLAN = os.path.join(DATA_DIR, "m11_p5_plan_l3.json")
FROZEN_L3_SHA = "4aab49b00177a22e21f65b884bd29d95a8e7333e76cb109ae8fcd8bb9bd94615"
OLD_REPLAN_SHA = "539325a13aa201cd5bc480d0b941768fb452ef2e36caccd3c90fba29ff1af98b"
L3_RUN_DIR = os.path.join(DATA_DIR, "m11_p5_l3_runs")
OUT_PATH = os.path.join(DATA_DIR, "m11_p5_plan_l3_replan.json")
# **硬保护**：这两个路径属于既有的、正在被正式跑使用的计划 ⇒ 本脚本绝不写入（R119/本节点事故登记）
FORBIDDEN_OUT_PATHS = (os.path.join(DATA_DIR, "m11_p5_plan_l3.json"),
                      os.path.join(DATA_DIR, "m11_p5_plan_l3.frozen_for_run.json"))

# ---- 协议常数（R80 §⑥：跨接口传递的量必须写明单位）----
N = 8192
T_MS = 10000.0
SETTLE_MS = 500.0
DT_MS = 0.05
DRIVE_RATE_HZ = 8.0
EXPECTED_STEPS = 210000        # round((500.0 + 10000.0)/0.05)
MIN_STEPS = 200000             # 经验下限（make_m11_p5_plan_g1.MIN_STEPS 同值）
SEEDS = (0, 1, 2)
SEED_PAIRS = {0: (0, 11), 1: (1, 12), 2: (2, 13)}      # R93：net_seed / event_seed
DELIVERY = "chunk"
CRITERION_DELIVERY = "step"
SLOW_INC = 6.0
TAU_SLOW_MS = 1000.0
STAGE = "l3_grid"
MECHANISM = "lever_l3"

# ---- 4 个格（**顺序即执行顺序**；A0 锚最先）----
# 网格 v2（R118 事前修订）：A0 锚点 + σ=0.10 的 4 个 bias 档；**顺序即执行顺序**（主 agent v2 派工）
CELLS = (
    dict(tag="A0", config_id="p5_L3_A0_b1.00_s0.30", bias_scale=1.00, bias_log_sigma=0.30,
         tag_cn="A0（锚）",
         lever="L3 格 A0: 锚点 = 既有 L0/L1 同配置点（确定性核对，照 R95 §⑤ 标 h_determinism_not_evidence）"),
    dict(tag="C0", config_id="p5_L3_C0_b1.00_s0.10", bias_scale=1.00, bias_log_sigma=0.10,
         tag_cn="C0",
         lever="L3 格 C0: 沿 σ=0.10 的最低 bias（已知 (a) 失败、零驱动=0 ⇒ (c) 预期 unavailable）"),
    dict(tag="C1", config_id="p5_L3_C1_b1.15_s0.10", bias_scale=1.15, bias_log_sigma=0.10,
         tag_cn="C1",
         lever="L3 格 C1: σ=0.10 偏置扫描 1（bias=1.15）"),
    dict(tag="C2", config_id="p5_L3_C2_b1.30_s0.10", bias_scale=1.30, bias_log_sigma=0.10,
         tag_cn="C2（主目标）",
         lever="L3 格 C2: ★主目标格：探针暂定同落 (a)/(c)/(e) 三带（R118 §③）"),
    dict(tag="C3", config_id="p5_L3_C3_b1.45_s0.10", bias_scale=1.45, bias_log_sigma=0.10,
         tag_cn="C3（极端档）",
         lever="L3 格 C3: σ=0.10 偏上界（极端档，爆炸风险 ⇒ 最后跑）"),
)
# 设计的字面规格（自检报告以本表逐字核对）
DESIGN_TABLE = {c["tag"]: (c["bias_scale"], c["bias_log_sigma"]) for c in CELLS}
CFG_RE = re.compile(r"^p5_L3_([A-Z]\d)_b(\d+\.\d\d)_s(\d+\.\d\d)$")

# ---- 每臂必须显式带的键（主 agent 清单，逐字）----
REQUIRED_ARM_KEYS = (
    "run_id", "stage", "config_id", "T_ms", "drive_mult", "seed", "net_seed", "event_seed",
    "role", "mechanism", "delivery", "gmax_scale", "bias_scale", "bias_log_sigma", "slow",
    "tau_slow_ms", "slow_inc", "n_neuron", "settle_ms", "indegree", "delay_bins", "subtypes",
    "device", "lever",
)
# 生产者**实际读取**这些键的位置（正则只认「从 spec 读」的写法；`None` = 生产者不读 ⇒ 装饰性键）
KEY_READ_PATTERNS = {
    "run_id": r'spec\["run_id"\]',
    "stage": None,
    "config_id": r'spec\.get\("config_id"',
    "T_ms": r'spec\["T_ms"\]',
    "drive_mult": r'spec\.get\("drive_mult"',
    "seed": r'spec\.get\("seed"',
    "net_seed": r'spec\.get\("net_seed"',
    "event_seed": r'spec\.get\("event_seed"',
    "role": r'spec\.get\("role"',
    "mechanism": r'spec\.get\("mechanism"',
    "delivery": r'spec\.get\("delivery"',
    "gmax_scale": r'"gmax_scale"',
    "bias_scale": r'"bias_scale"',
    "bias_log_sigma": r'"bias_log_sigma"',
    "slow": r'spec\.get\("slow"',
    "tau_slow_ms": r'spec\.get\("tau_slow_ms"',
    "slow_inc": r'spec\.get\("slow_inc"',
    "n_neuron": r'spec\["n_neuron"\]',
    "settle_ms": r'spec\["settle_ms"\]',
    "indegree": r'"indegree"',
    "delay_bins": r'spec\.get\("delay_bins"',
    "subtypes": r'spec\.get\("subtypes"',
    "device": r'spec\.get\("device"',
    "lever": None,
}
DECORATIVE_KEYS = tuple(k for k, v in KEY_READ_PATTERNS.items() if v is None)

# ---- 按档取价的单价表（**T=1 s 短跑实测**，单位 ms/step；R112 ⑤ 第 30 次更正）----
# 每项 = (ms_per_step, 来源, 是否实测)。`measured=False` ⇒ 该格该角色**未确认**，用同格/同 bias 家族的
# 已测点作代理；代理必须逐条写明，并在预算里以 `proxies` 列出。
UNIT_PRICES = {
    ("A0", "baseline"): (1.1872, "实测 data/m11_p5_l3_feasibility/l3pb_b1.00s0.30_8192_T1000ms_d1_s0.json", True),
    ("A0", "zero_drive"): (1.1872, "**代理**（同格 baseline；本格 zero_drive 无 T=1 s 实测；另有 T=10 s 墙钟锚 0.7704）", False),
    ("C0", "baseline"): (1.0760, "实测 data/m11_p5_l3_feasibility/l3pb_b1.00s0.10_8192_T1000ms_d1_s0.json", True),
    ("C0", "zero_drive"): (0.6434, "**代理**（「空转地板」：本格零驱动探针实测 0 脉冲 ⇒ 取同 bias 家族低活动格实测 "
                                 "data/m11_p5_l3_feasibility/l3win_b130_s010_zd_8192_T1000ms_zero_drive_s0.json = 0.6434）", False),
    ("C1", "baseline"): (1.2808, "**代理**（σ=0.10 家族**线性内插**：1.0760 + (1.4856−1.0760)×(1.15−1.00)/(1.30−1.00) = 1.2808）", False),
    ("C1", "zero_drive"): (0.6434, "**代理**（同「空转地板」；本格无实测）", False),
    ("C2", "baseline"): (1.4856, "实测（T=1 s）data/m11_p5_l3_feasibility/l3pb_b1.30s0.10_8192_T1000ms_d1_s0.json",
                         True),
    ("C2", "zero_drive"): (0.6434, "实测（T=1 s，零驱动 46 脉冲）data/m11_p5_l3_feasibility/"
                                   "l3win_b130_s010_zd_8192_T1000ms_zero_drive_s0.json", True),
    ("C3", "baseline"): (1.4856, "**代理**（σ=0.10 家族上界：取 C2 的 T=1 s 实测 1.4856；bias=1.45 更高活动 ⇒ 可能**低估**）", False),
    ("C3", "zero_drive"): (0.6434, "**代理**（同「空转地板」；bias=1.45 处悬崖可能位移 ⇒ 该代理**不确定性最大**）", False),
}
# **T=10 s 实测锚（本节点实读，正式跑产物）**：C2 baseline 的两个实现 = 0.9318 / 0.9887 ms/step
C2_BASELINE_T10_MEASURED = (0.9318, 0.9887, "data/m11_p5_l3_runs/l3_C2_b1.30_s0.10_8192_T10000ms_d1.0_s{0,1}.json")
PRICE_UPPER_ALL = 1.4856       # 上界 = 本网格(σ=0.10 与锚点)内 T=1 s 实测最大值（C2 baseline 1.4856）
GPU_MS_PER_H = 3.6e6
DUTY_CYCLE = 0.054             # R112 ⑥ 实测占空比（MPS 与 CPU 双重争用下）


def sha256_of(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:  # noqa: BLE001
        return "unavailable"


def _rid(tag: str, dose: float, seed: int) -> str:
    """**逐字对齐冻结计划**（R119）：`l3_<cell>_<N>_T<T>ms_d<dose>_s<seed>`（`d1.0`/`d0.0` 保一位小数）。"""
    return "l3_%s_%d_T%dms_d%s_s%d" % (tag, N, int(T_MS), dose, seed)


def _role_of(dose: float) -> str:
    """角色规则**逐字对齐**生产者/评估器：dose=0 ⇒ zero_drive；dose=1 ⇒ baseline。"""
    if dose == 0.0:
        return "zero_drive"
    if dose == 1.0:
        return "baseline"
    raise AssertionError("L3 只允许 dose ∈ {0.0, 1.0}（禁驱动抬升：docs/m11_b3_g1_design.md L146）")


def build_arms():
    """**每格内先 baseline 后 zero_drive**；格间顺序 = CELLS 顺序（A0 → A1 → B1 → B2）。"""
    arms, levels = [], []
    for cell in CELLS:
        tag = cell["tag"]
        run_ids = {}
        for dose in (1.0, 0.0):                       # 先 baseline(1.0)，后 zero_drive(0.0)
            role = _role_of(dose)
            run_ids[role] = []
            for sd in SEEDS:
                ns, es = SEED_PAIRS[sd]
                a = dict(
                    run_id=_rid(tag, dose, sd), stage=STAGE, config_id=cell["config_id"],
                    T_ms=T_MS, drive_mult=dose, seed=sd, net_seed=ns, event_seed=es,
                    role=role, mechanism=MECHANISM, delivery=DELIVERY,
                    gmax_scale=1.0, bias_scale=cell["bias_scale"],
                    bias_log_sigma=cell["bias_log_sigma"], slow=True,
                    tau_slow_ms=TAU_SLOW_MS, slow_inc=SLOW_INC, n_neuron=N,
                    settle_ms=SETTLE_MS, indegree=20, delay_bins=[40, 60, 80],
                    subtypes=True, device="mps", lever=cell["lever"],
                )
                arms.append(a)
                run_ids[role].append(a["run_id"])
        levels.append(dict(
            config_id=cell["config_id"], stage=STAGE, gmax_scale=1.0,
            bias_scale=cell["bias_scale"], bias_log_sigma=cell["bias_log_sigma"],
            lever=cell["lever"], n_seeds=len(SEEDS), seeds=list(SEEDS),
            net_event_pairs={str(k): list(v) for k, v in SEED_PAIRS.items()},
            elasticity_arm=False,
            run_ids=[r for r in (run_ids["baseline"] + run_ids["zero_drive"])],
            run_ids_baseline=run_ids["baseline"], run_ids_zero_drive=run_ids["zero_drive"],
            cell_cn=cell["tag_cn"],
        ))
    return arms, levels


def _budget(arms) -> dict:
    """**按档取价的预算**（R112 ⑤）：逐格加权，**不得**用最低档单价外推全网格。"""
    mid_ms = up_ms = 0.0
    per_cell, proxies = {}, []
    for cell in CELLS:
        tag = cell["tag"]
        cell_ms = 0.0
        for role in ("baseline", "zero_drive"):
            p, src, measured = UNIT_PRICES[(tag, role)]
            if not measured:
                proxies.append(dict(cell=cell["config_id"], role=role, price_ms_per_step=p, source=src))
            cell_ms += 3 * EXPECTED_STEPS * p
        per_cell[cell["config_id"]] = dict(
            bias_scale=cell["bias_scale"], bias_log_sigma=cell["bias_log_sigma"], n_arms=6,
            steps=6 * EXPECTED_STEPS,
            price_baseline_ms_per_step=UNIT_PRICES[(tag, "baseline")][0],
            price_zero_drive_ms_per_step=UNIT_PRICES[(tag, "zero_drive")][0],
            gpu_h_mid=round(cell_ms / GPU_MS_PER_H, 4))
        mid_ms += cell_ms
    up_ms = len(arms) * EXPECTED_STEPS * PRICE_UPPER_ALL
    return dict(
        unit_prices="**按档取价（逐格加权）；单价来源 = T=1 s 短跑实测（ms/step）**。"
                    "R112 ⑤（我的第 30 次公开更正）：**不得用最低档单价（0.6415）外推全网格**。",
        price_table={("%s|%s" % (c["config_id"], r)): dict(
            ms_per_step=UNIT_PRICES[(c["tag"], r)][0], source=UNIT_PRICES[(c["tag"], r)][1],
            measured=UNIT_PRICES[(c["tag"], r)][2])
            for c in CELLS for r in ("baseline", "zero_drive")},
        proxies=proxies,
        steps_per_run_incl_settle=EXPECTED_STEPS,
        formula="gpu_h = Σ_arm (210000 步 × 该(格,角色)单价 ms/step) / 3.6e6 ms",
        per_cell=per_cell,
        gpu_h_mid=round(mid_ms / GPU_MS_PER_H, 4),
        gpu_h_upper=round(up_ms / GPU_MS_PER_H, 4),
        upper_rule="上界 = 全 30 臂一律按**本网格内 T=1 s 实测最大值** 1.4856 ms/step（C2 baseline）",
        caveat_T10="**T=10 s 的稳态单价并未实测**（本网格的 4 格 × 2 角色里只有 T=1 s 探针）。"
                    "**必须实跑复探**：先跑 A0 的 6 臂（最便宜、且与既有 L0 臂同规格 ⇒ 可逐位核对），"
                    "用 A0 的实测 `ms_per_step_steady` 与既有 T=10 s 臂（见 `wall_anchor_from_existing`）"
                    "对齐后再放行 A1/B1/B2。",
        wall_anchor_from_existing=dict(
            note="**反向证据（既有 T=10 s 实测）**：A0 格在既有 L0 计划里**已按同规格跑过**（bias=1.00 × σ=0.30，"
                 "T=10 s、settle=500 ms、chunk）⇒ 其**含 settle 的墙钟单价**可直接读出，且**低于**本预算用的"
                 "T=1 s 单价（例：baseline 0.8399 vs 1.1872）⇒ **'T=10 s 单价必然更高' 这一前提在 A0 格被实测反向**，"
                 "预算因此**偏保守**；但 bias=1.30 三格的 T=10 s 稳态单价**仍未测**（活动量高 ⇒ 更贵），故不得据此下调。",
            cells=dict.fromkeys([c["config_id"] for c in CELLS], None),
        ),
        comparison_with_L2=dict(
            L2_scan_ref="L2_design.budget.segments['L2_slow_scan(5 档 × 2 角色 × 3 实现, T=10 s)']"
                        "= 30 run / 6,300,000 步 / gpu_h_mid 1.394 / upper 1.701（折合 0.7966 / 0.9720 ms/step）",
            L3_per_step_mid_ms=round(mid_ms / (len(arms) * EXPECTED_STEPS), 4),
            ratio_L3_over_L2=round((mid_ms / (len(arms) * EXPECTED_STEPS))
                                   / (1.394 * GPU_MS_PER_H / 6300000.0), 2),
            note="L3 单价高于 L2 的原因：L3 全部落在 bias=1.00–1.30（活动量高 ⇒ 单价高），"
                 "而 L2 基线是 bias=0.70。**两者不可互相外推。**",
        ),
        wall_clock=("纯计算 %.4f GPU-h ⇒ 按 R112 ⑥ 实测占空比 %.3f ⇒ 墙钟 ≈ %.1f h；"
                    "再加逐 run 新进程装配开销（8k 装配 30–60 s × 24 run ≈ 0.2–0.4 h）⇒ 必须托管。"
                    % (round(mid_ms / GPU_MS_PER_H, 4), DUTY_CYCLE,
                       round(mid_ms / GPU_MS_PER_H / DUTY_CYCLE, 1))),
        downgrade_plan_if_over=[
            "**不降档**：24 臂 = 4 格 × 2 角色 × 3 实现，是预注册的最小可判结构（每格 (c) 需 ≥3 seed）。",
            "若必须降：**先砍 B2**（窗口下沿搜索档，−6 臂 ⇒ −%.4f GPU-h mid），再砍 A1（对照档，−%.4f）；"
            "**A0 与 B1 不得砍**（A0 = 跨计划锚，B1 = 主目标）。"
            % (6 * EXPECTED_STEPS * (UNIT_PRICES[("b2", "baseline")][0] + UNIT_PRICES[("b2", "zero_drive")][0])
               * 3 / 6 / GPU_MS_PER_H,
               6 * EXPECTED_STEPS * (UNIT_PRICES[("a1", "baseline")][0] + UNIT_PRICES[("a1", "zero_drive")][0])
               * 3 / 6 / GPU_MS_PER_H),
            "**禁止**用「减少 settle」或「缩短 T」降成本：会违反步数断言（R80）与 T_min 口径。",
        ],
    )


def producer_key_report():
    """**生产者键读取逐字比对**：返回 [(key, 行号, 代码片段)]，并断言每个「被读键」确实被读。"""
    with open(PRODUCER) as f:
        lines = f.read().splitlines()
    rows, missing = [], []
    for k in REQUIRED_ARM_KEYS:
        pat = KEY_READ_PATTERNS[k]
        if pat is None:
            rows.append((k, None, "**装饰性键：生产者在 spec 里不读**（仅用于 run_id 生成/人可读）"))
            continue
        hit = None
        for i, ln in enumerate(lines, 1):
            if re.search(pat, ln):
                hit = (i, ln.strip())
                break
        if hit is None:
            missing.append(k)
            rows.append((k, None, "**未找到读取处（❌）**"))
        else:
            rows.append((k, hit[0], hit[1]))
    assert not missing, ("以下键在生产者内找不到读取处（计划写了但生产者不读 ⇒ 静默失效）：%r" % missing)
    return rows


def _v2_blocks(plan: dict) -> dict:
    """**网格 v2 的全部条款块**（R118 事前修订 + R112–R124 的纪律）。

    这些键覆盖 `build_plan()` 里的 v1 版本（v1 的 4 格已被 R122 判「不得运行」）⇒ 就地覆盖是**有意的**：
    旧内容不再留在任何可执行文件里（纪律第六十三条）。
    """
    arms = plan["arms"]
    b = plan["budget"]
    b["gpu_h_mid_T10_anchored"] = round(
        b["gpu_h_mid"] - 3 * EXPECTED_STEPS * (1.4856 - sum(C2_BASELINE_T10_MEASURED[:2]) / 2.0) / GPU_MS_PER_H, 4)
    b["T10_anchor_note"] = ("**正式口径（T=10 s）实测锚**（本节点实读 `data/m11_p5_l3_runs/`）：C2 baseline 的 s0/s1 "
                            "两实现 `ms_per_step_steady` = %s / %s（均值 %.4f，wall_s 236.3 / 247.1）⇒ 用该均值替换 C2 "
                            "baseline 的 T=1 s 代理价后 `gpu_h_mid = %s`。**两个口径都给出**：以 T=1 s 为统一口径可得 "
                            "同 caliber 的 %s；以 T=10 s 实测锚替 C2 得更贴近现实的 %s。"
                            % (C2_BASELINE_T10_MEASURED[0], C2_BASELINE_T10_MEASURED[1],
                               sum(C2_BASELINE_T10_MEASURED[:2]) / 2.0,
                               b["gpu_h_mid_T10_anchored"], b["gpu_h_mid"], b["gpu_h_mid_T10_anchored"]))
    b["frozen_plan_comparison"] = dict(
        frozen_mid=2.275, frozen_upper=2.8,
        note="冻结计划 `m11_p5_plan_l3.json` 用**代表值** 1.30 ms/step（上界 1.60）估得 mid 2.275 / upper 2.8；"
             "本版用**逐格实测价**（最小值 0.6434）⇒ mid/upper 更低（2.275 → " + str(b["gpu_h_mid「]) + 」）。"
             "**两者都不含** settle 的 step-delivery 固定开销（正式跑实测每臂 wall≈236–247 s，含 settle ≈50 s）"
             "⇒ 30 臂纯计算 ≈2.0–2.1 h。")
    b["wall_clock"] = ("纯计算 %.4f GPU-h（本版逐格加权）或 %.4f（T=10 s 锚）⇒ 按占空比 0.054 ⇒ 墙钟 ≈%.1f–%.1f h；"
                       "但正式跑实测每臂 wall≈236–247 s（含 settle 的 step-delivery ≈50 s）⇒ 30 臂 ≈2.0 h MPS-busy，"
                       "墙钟 = 2.0 h ÷ 占空比 ⇒ **must be 托管 + resume-safe**（`[SKIP]` 已完成的臂）。"
                       % (b["gpu_h_mid"], b["gpu_h_mid_T10_anchored"],
                          b["gpu_h_mid"] / DUTY_CYCLE, b["gpu_h_mid_T10_anchored"] / DUTY_CYCLE))

    return dict(
        plan_id="p5_L3_sigma_cliff_bias_scan",
        status=("**有效 —— 网格 v2（R118 事前修订；%d 臂 = 5 格 × 2 角色 × 3 独立实现）**。"
                "本文件是 L3 的**可运行**计划；旧的 4 格 24 臂版本已按 R122 §④ 作废（见 `supersedes`）。" % len(arms)),
        supersedes=dict(
            path="neural_exploration/data/m11_p5_plan_l3_replan.json（**本文件就地覆盖了它的旧内容**）",
            old_plan_id="l3_replan_bias_sigma_window_n8192", old_sha256=OLD_REPLAN_SHA,
            old_size_bytes=62781, old_cells=["(1.00,0.30)", "(1.30,0.30)", "(1.30,0.20)", "(1.30,0.15)"],
            verdict=("**R122 §④：不得运行**。探针已在同一角落实测 σ=0.30/0.20/0.15 的 R ≈ 0.30/0.879/0.881（全部 ≥0.10 ⇒ "
                     "(c) fail），零驱动 pooled 远超窗口上界 ⇒ 跑它 = 花 24 臂重测一个已测结论。"),
            discipline=("**纪律第六十三条**：被取代的计划不得原样留在可执行位置 ⇒ 本脚本**就地覆盖**该路径，"
                        "并在本文件顶部写死 `status` + `supersedes`（含旧 sha）⇒ 盘上**不再存在**那个过期计划，"
                        "且「过期」这一事实**不依赖任何人的记忆**。"),
            traceability="旧内容 sha256 = `%s`（62,781 B，1638 行）；本节点上一轮的自检报告 §0 亦留有该 sha 与差异说明。" % OLD_REPLAN_SHA),
        alignment_with_frozen_plan=dict(
            frozen_plan=dict(path="neural_exploration/data/m11_p5_plan_l3.json", sha256=FROZEN_L3_SHA),
            same_namespace=("本计划与冻结计划**共用** `plan_id` / 逐臂 `config_id` / 逐臂 `run_id`（R119 §① 已固化的命名），"
                            "**只改臂顺序**、不改任何档位/实现/协议字段 ⇒ ① 已完成的臂会被 `[SKIP]` 复用（生产者 L553–557），"
                            "② 评估/报告命令（含 R120 §② 的协议文件名 `m11_p5_protocol_p5_L3_sigma_cliff_bias_scan.json`）"
                            "**不必改写**。**⇒ 两个计划只应运行其中一个**（同时跑会争用 MPS，R119 §① 的 6× 退化已实测）。"),
            drop_in_assertion="生成器断言：本计划与冻结计划的 `(run_id, config_id, role, seed)` 集合**完全相同**（见 §自检）。",
            already_done_arms=("截止本节点实读，正式跑已产出 **2 个 `.done`**：`l3_C2_b1.30_s0.10_…_d1.0_s0`（spk=52,601、"
                               "`ms/step`=0.9318、`ei_ratio`=2.5970）与 `…_s1`（spk=52,572、0.9887、2.6034）⇒ "
                               "**C2 baseline 已有 2 个实现的正式读数**；本计划若跑进同一 `--out-dir`，这两臂将被复用。"),
        ),
        parent_prereg=dict(
            path="neural_exploration/data/m11_p5_lever_prereg.json",
            sha256_full=sha256_of(LEVER_PREREG),
            cited=dict(
                lever="L3 二维角落：`bias_scale`↑（抬 active_frac/中位率）× `bias_log_sigma` = 0.10（削上尾 ⇒ 压自发活动）",
                design=("**R118 事前修订网格 v2**：σ=0.10（悬崖面附近唯一可达窗口值）× `bias ∈ {1.00, 1.15, 1.30, 1.45}`；"
                        "锚点格 (1.00, σ=0.30) = 既有 L0/L1 同配置点"),
                hypothesis=("σ↓ 削掉播种自发活动的上尾（尾部选择性，R113/R118），bias↑ 把 active_frac 推过 50% 并抬高中位率"
                            "（R114）⇒ 二者分工互补，第一次可能**同时**满足 (a) 与 (c)"),
                stop_rules="见本计划 `stop_rules`（8 条，按 v2 改写自 `L2_design.stop_rules`）",
            ),
            amendment=dict(
                path="neural_exploration/data/m11_prereg_amend_r118_l3_grid_v2.json",
                sha256=sha256_of(R118_AMEND), sha256_expected=R118_AMEND_SHA,
                authority="R118（docs/m11_acceptance_master.md §3di）",
                supersedes="R115 §④ 所载的 L3 网格 `bias∈[1.20,1.40] × σ∈[0.15,0.25]`",
                why=("实测悬崖：`bias=1.30` 下零驱动脉冲 σ=0.30/0.20/0.15 ⇒ 4,126/4,875/4,666，σ=0.10 ⇒ **46**"
                     "（0.15→0.10 落差 **101×**）⇒ 原网格整段**远在 (c) 窗口之外**；窗口只在 σ=0.10 处可达。"),
                honesty_clause=("修订文件自记："σ=0.10 与 bias=1.15/1.45 **原在我自己登记的网格范围之外** ⇒ "
                                "必须走事前修订，**不得事后追认为本来就在范围内**""),
            ),
            amendments_other=dict(
                r109_a_reporting=dict(path="neural_exploration/data/m11_p5_prereg_amend_r109_a_reporting.json",
                                      sha256=sha256_of(R109_AMEND),
                                      scope="(a) 三条并报（不改频带、不改 pass/fail）；本计划追加第 4 条 `pooled_zero_drive` 绝对值"),
                l1_plan_reference=dict(path="neural_exploration/data/m11_p5_plan_l1.json", sha256=L1_PLAN_SHA,
                                       scope="**键集合模板**（本计划顶层键集 ⊇ L1 键集，差集空）"),
            ),
        ),
        judged_units=[dict(config_id=l["config_id"], T_ms=T_MS, mechanism=MECHANISM,
                           zero_drive_seeds=list(SEEDS), baseline_seeds=list(SEEDS),
                           bias_scale=l["bias_scale"], bias_log_sigma=l["bias_log_sigma"],
                           net_event_pairs=l["net_event_pairs"],
                           note=("评估器逐 (config_id,T,mechanism) 单元复核 (a)(b)(c)(e)；本计划 mechanism=%r（≠ 'none'）"
                                 "⇒ 硬门⑤「无机制基线臂登记」对该单元不可用（见 cross_tool_notes）；"
                                 "**L3 判定以 tools/report_m11_l3.py 为主、评估器为辅**（R120 §③）" % MECHANISM))
                      for l in plan["levels"]],
        lever_prereg_falsifier=(
            "**主断言**：在 `bias_scale ∈ {1.00, 1.15, 1.30, 1.45} × bias_log_sigma = 0.10`（+ 锚点格 (1.00,0.30)）上，"
            "存在使 **(a) 逐神经元中位率 ∈ [0.5,2.0] Hz** 且 **(c) `pooled_zero_drive ∈ [300, 0.10×pooled_baseline]`** "
            "同时成立的点。**证伪式**：若每一格都出现 (i) `pooled_zero_drive < 300`（(c) `unavailable`）或 "
            "(ii) `R > 0.10`（(c) fail）或 (iii) 中位率 < 0.5 Hz（(a) fail）⇒ 判 **L3 失败 ⇒ 第四维结构对立确认 ⇒ "
            "必须换机制（不再改参数）**。"),
        core_falsifiable_assertion=dict(
            statement=("**在 `bias_scale ∈ {1.00, 1.15, 1.30, 1.45} × bias_log_sigma = 0.10` 的偏置扫描上（含锚点格 "
                       "(1.00, 0.30)），存在同时满足 (a) 与 (c) 的点。**"
                       "（**网格已于 R118 事前修订**：R115 §④ 的 `bias∈[1.20,1.40] × σ∈[0.15,0.25]` 被取代 —— "
                       "**不是未测，是已测且远超上界**。）"),
            a_definition="**(a)**（主判口径，频带 `[0.5, 2.0] Hz` **逐字不变**）：基准臂 `per_neuron` **逐神经元中位率**",
            c_definition="**(c)**：`R = pooled_zero_drive / pooled_baseline ≤ 0.10`，**且** `pooled_zero_drive ≥ 300`（**硬门①**）",
            window=("**目标是一个窗口，不是一个极限**（R115 §②）：`pooled_zero_drive ∈ [300, 0.10 × pooled_baseline]`。"
                    "下界 = 硬门①（`m11_p5_targets.csv` L48）；上界 = `selfsustain_ratio_max = 0.10`（同表 L46）。"
                    "**R 太小（零驱动被打干净）会让 (c) 变 `unavailable` 而不是 `pass`** ⇒ 每档必须报 `pooled_zero_drive` 绝对值。"),
            why_self_consistent=("(b) 现为 `unavailable`（真实 ROI 掩膜维缺失）⇒ 不进任何门断言；(a)/(c) 均为 `final=1` 的"
                                 "已定稿判定带 ⇒ 双目标可判且有判别力。"),
            falsifiers=[
                "① **主证伪式**：每一格都命中 (i) `pooled_zero_drive < 300` 或 (ii) `R > 0.10` 或 (iii) 中位率 < 0.5 Hz ⇒ 判 **L3 失败 ⇒ 第四维结构对立确认 ⇒ 必须换机制（不再改参数）**",
                "② **机制专属证伪**：若 σ=0.10 时**基准臂** `a_active_frac` 也大幅塌陷（R118 §④4 的反平凡性条款；"
                "探针实测 0.6063、正式 T=10 s 实测 **1.0000**）⇒ 该杠杆是**全局性**的、「尾部选择性」被证伪 ⇒ 如实登记，不得改判据",
                "③ **不可判定 ≠ 失败**：某格 (c) 判 `unavailable`（`pooled_zero_drive < 300`，或两 CI 法不一致 ⇒ `invalid`）时"
                "**不得记为 fail，也不得记 pass**（M10 D1 家族；L1 已有 `two_ci_methods_disagree` 先例）",
            ],
            positive_outcome=("某格 (a) ∧ (c) 同时成立（含窗口与 CI 条件）⇒ 进入定稿确认，并**只**得结论"
                              "「(a) 与 (c) 在 σ=0.10 悬崖面上可同时满足」；**不得**外推到未测侧（见 `unmeasured_side_declaration`）。"),
        ),
        window_definition=dict(
            lower_bound_pooled_zero_drive=300,
            lower_bound_source="neural_exploration/data/m11_p5_targets.csv L48 `selfsustain_zero_drive_min_spikes`（硬门①）",
            upper_bound_formula="0.10 × pooled_baseline（同一 (config_id,T) 的 3 个实现求和）",
            upper_bound_source="neural_exploration/data/m11_p5_targets.csv L46 `selfsustain_ratio_max = 0.10`",
            c2_provisional=dict(
                cell="C2 (bias=1.30, σ=0.10)", h_determinism_note="**provisional（探针外推）**",
                pooled_zero_drive_extrapolated=1380,
                upper_bound_extrapolated=15400,
                source="R118 §②（T=1 s 单实现 × 10 × 3 的外推）；**正式值由本计划 30 臂给出**"),
            a0_numeric_example=dict(
                pooled_baseline_3impl=91146.0, pooled_zero_drive_3impl=26978.0,
                window_pooled_zero_drive=[300, 9114.6], R_measured=0.295987,
                verdict="A0 的 `pooled_zero_drive` 26,978 **高于**上界 9,114.6 ⇒ (c) 预期 fail（R=0.296 > 0.10）",
                source="neural_exploration/data/m11_p5_resting_eval_repl.json（本节点实读）"),
            requirement="**每档必须报 `pooled_zero_drive` 的绝对值**，不得只报 R（R115 §②）",
        ),
        unmeasured_side_declaration=dict(
            discipline="**纪律第五十八条**：未测侧必须显式登记 —— **未测 ≠ 已排除**；同时**已测且排除的侧也必须写清**（本版新增）",
            by_measurement_excluded=[dict(
                side="`σ ∈ [0.15, 0.25]` 于 `bias ≥ 1.20`（含原 R115 §④ 网格的整段）",
                status="**不是未测，是已测且远超 (c) 上界**",
                evidence=("`bias=1.30` 零驱动探针：σ=0.20 ⇒ 4,875 脉冲；σ=0.15 ⇒ 4,666（T=1 s、单实现）⇒ "
                          "换算正式口径 pooled ≈1.4×10⁵ ≫ 窗口上界 ≈1.5×10⁴；单实现 R ≈ 0.879 / 0.881（均 ≥0.10）"),
                source="R118 §①；data/m11_p5_l3_feasibility/l3win_b130_s020_zd_*.json、l3win_b130_s015_zd_*.json",
                consequence="**不得**再把 `σ∈[0.15,0.25]` 当作「可能可达」的区间（R122 §② 已据此判旧网格作废）")],
            still_unmeasured=[
                dict(side="`σ ∈ (0.10, 0.15)`（**悬崖面内部**）",
                     note="**该区间是否存在可达窗口尚未知**：σ=0.15 ⇒ R≈0.881、σ=0.10 ⇒ R≈0.0089，但 0.10 处 pooled≈1,380 "
                          "仅比硬门① 的 300 高 4.6× ⇒ **窗口在 σ 轴上的宽度未测**；本计划**不含**这些档"),
                dict(side="`bias_scale > 1.45`", note="未测（σ=0.10 与 σ=0.30 两侧都未测；R112 的 bias=1.60 探针是 T=1 s 单实现 ⇒ 不作判据）"),
                dict(side="`bias_scale < 1.00` 且 `σ = 0.10`", note="未测（本计划 C0 正是该侧的最低点 bias=1.00；更低 bias 无档）"),
                dict(side="`σ > 0.30`（与 `bias ≥ 1.15` 的组合）", note="未测（L1 只在 bias=0.70 处扫过 σ=0.45/0.60/0.90）"),
                dict(side="`drive_rate_hz` 的抬升", note="**预注册禁止作为标定手段**（docs/m11_b3_g1_design.md L146）；本计划不含任何驱动抬升档"),
                dict(side="慢适应/STP 与 (bias,σ) 的组合", note="`slow_inc`/`tau_slow_ms` 全部固定默认档（6.0/1000）；L2 的慢适应放大档未跑（R110 判 L2 事前失败）"),
                dict(side="外部驱动侧", note="事件表构造、事件表泄漏、`event_seed` 与驱动的交互未测"),
                dict(side="亚型选择性杠杆与其它引擎自由度", note="PV/SST/VIP 权重与比例、层流比例、`indegree`、`gmax_scale`、`delay_bins` 全部固定在基线"),
                dict(side="T 维度（时长依赖性）", note="本计划只有 T=10 s ⇒ 评估器硬门④（{1,2,4}×T_min 阶梯）为 `unavailable`；(d) 未测"),
            ],
            statement=("以上方向**未测 ≠ 已排除**；L3 的任何结论**只**覆盖 "
                       "`bias ∈ {1.00,1.15,1.30,1.45} × σ = 0.10` 与锚点格 (1.00,0.30)，"
                       "且每格只覆盖 `drive_mult ∈ {0,1}`、T=10 s、slow 默认档。"),
        ),
        prior_probe_evidence=dict(
            discipline="**以下全部是 T=1 s、单实现、未预注册的探针（R112/R113/R115/R118）⇒ 只作方向性支持，"
                       "不作判据、不进任何门**；正式判定只由本计划 30 臂（T=10 s × 3 实现）给出",
            cliff_at_bias_1_30=[
                dict(sigma=0.30, zero_drive_spk=4126, ms_per_step=1.4379, ei_ratio_zero=6.0406),
                dict(sigma=0.20, zero_drive_spk=4875, baseline_spk=5546, R_single_impl=0.8790, ms_per_step=1.0371,
                     ei_ratio_baseline=2.7059),
                dict(sigma=0.15, zero_drive_spk=4666, baseline_spk=5295, R_single_impl=0.8812, ms_per_step=0.9693,
                     ei_ratio_baseline=2.6482),
                dict(sigma=0.10, zero_drive_spk=46, baseline_spk=5149, R_single_impl=0.0089, ms_per_step=0.6434,
                     ei_ratio_zero=127.0544, ei_ratio_baseline=2.5767,
                     warn="零驱动臂 `ei_ratio=127.05` 是**异常值**（i_exc 极小 ⇒ 比值爆炸）⇒ **不得与基准臂的 (e) 混用**"),
            ],
            formal_T10_measurements=dict(
                source="data/m11_p5_l3_runs/（正式跑产物，本节点实读）",
                arms=[dict(run_id="l3_C2_b1.30_s0.10_8192_T10000ms_d1.0_s0", role="baseline", seed=0, spikes_total=52601,
                           ms_per_step=0.9318, wall_s=236.3, ei_ratio=2.5970,
                           note="R121 ①：中位率 **0.6000 Hz** ∈ [0.5,2.0] ✓、`active_frac` **1.0000** ✓"),
                      dict(run_id="l3_C2_b1.30_s0.10_8192_T10000ms_d1.0_s1", role="baseline", seed=1, spikes_total=52572,
                           ms_per_step=0.9887, wall_s=247.1, ei_ratio=2.6034)],
                caution=("这些是**单臂读数**（(c) 的 `zero_drive` 侧尚未完成 ⇒ `pooled_zero_drive`/`R` 未知）⇒ **不得**据此判 (c) 通过；"
                         "且 `(e)` 的跨工具口径冲突已由 R123/R124 登记为**待复核**（见 cross_tool_notes）。")),
            resolution_correction=dict(
                claim="短时窗（T=1 s）的中位率因**计数粗量化**而系统性偏高",
                evidence="探针外推 ≈1.0 Hz vs 正式 T=10 s 实测 **0.6000 Hz**（高估 1.67×）⇒ R118 §④3 的警告成立",
                consequence="**任何「中位率进带」的结论必须用协议时窗（T=10 s）确认，不得用短窗外推**（R121 §③）"),
            not_judged="以上**不得**用于判 (a)/(c)，**不得**用于免除实跑，**不得**计入任何 n_seeds",
        ),
        stop_rules=[
            "① 某格使**主断言成立** ⇒ **不自动早停**（本计划是**扫描**：30 臂全跑完再判，与冻结计划同款；"
            "理由：早停会丢掉「该点是否稳健」与复刻轴证据）；但**立即上报**该格读数给主 agent 以便提前决策。"
            "**【谁在何时执行】主 agent，在该格 6 臂产物到齐后**",
            "② 某格零驱动臂 0 脉冲（`zero_activity=true`）⇒ 该格 (c) 判 **`unavailable`**（既非 fail 亦非 pass；**硬门①的 300 脉冲下界在此生效**）；"
            "**该格 baseline 若也 0 脉冲 ⇒ 硬失败**（驱动侧零响应 = 网络真坏）⇒ 停并查口径。"
            "**【谁在何时执行】评估器/报告工具判 unavailable（validate_p11_resting.py L814）；主 agent 读 `[DONE]` 的 `ZERO_ACTIVITY=1` 后决定是否停**",
            "③ 连续 **2 格**既未使中位率向 `[0.5,2.0]` 靠近、也未使 R 下降 ⇒ **停**，报「无改善」。**【谁在何时执行】主 agent，在第 2 格 6 臂产物到齐后**",
            "④ 任一格基准臂 `active_frac` 塌陷（R118 §④4 反平凡性条款的判别值 ≈0.27 量级）**且**零驱动 ≈0 ⇒ 该格 **`invalid`**"
            "（**不得**记为机制有效）；任一格 `hard_failures` 非空（慢变量读回断言、双估计器不一致、轨迹长度⟺(T,dt) 口径等）⇒ 该格 `invalid`；"
            "**≥2 格**如此 ⇒ **停并查口径**（**不得**记为「机制无效」，M10 D1 家族）。**【谁在何时执行】评估器/报告工具；主 agent 在其输出后**",
            "⑤ 任一格的 `(net_seed, event_seed)` 对重复（计划级断言违反）⇒ **跑前整轮停**。**【谁在何时执行】生产者 `assert_plan_replication`，在 `--plan` 模式**跑任何臂之前**（L548，失败即 `SystemExit(3)`）**",
            "⑥ 触及预算包络（`gpu_h_mid` %s / `upper` %s GPU-h，或墙钟超出包络）⇒ **停**，**不得静默超支**。**【谁在何时执行】主 agent，逐臂比对累计 `wall_s` 与预算**"
            % (b["gpu_h_mid"], b["gpu_h_upper"]),
            "⑦ **预注册禁止**：任何以**驱动抬升**为标定手段的档**不得进 L3**（`docs/m11_b3_g1_design.md` **L146**；"
            "`wrong_usage_consequence` 亦登记在 `m11_p5_targets.csv` L119 `protocol_drive_rate_hz` 行）⇒ 出现即**计划作废**。"
            "**【谁在何时执行】本计划**跑前**（生成器断言 `drive_mult ∈ {0.0,1.0}`、无 `drive_rate_hz` 覆盖）+ 主 agent 交办复核**",
            "⑧ σ=0.10 的**全部** bias 档失败且非反向 ⇒ 报『**悬崖不可达窗口**』⇒ 判 **L3 失败 ⇒ 第四维结构对立确认 ⇒ 必须换机制（不再改参数）**；"
            "若失败只是采样/口径问题（`hard_failures` 非空）⇒ **先修口径重跑**；若某格仅 (c) 判 `unavailable` ⇒ **先按窗口判据补测**，不得直接判失败。"
            "**【谁在何时执行】主 agent，在 5 格报告齐备后**",
        ],
        cross_tool_notes=dict(
            evaluator_mechanism_gap=("`mechanism=%r` ⇒ 评估器按 `(config_id,T_ms,mechanism)` 建单元（validate_p11_resting.py L625–627）"
                                     "⇒ (a)/(c)/(e) 会被正常计算；**但**硬门⑤（L660–670）要求同 (config_id,T_ms) 有 `mechanism=='none'` 且 "
                                     "`drive_mult==1.0` 的臂 ⇒ 本计划**不满足** ⇒ (a)/(b) 行 `discriminative=None`、G1 至多 `unavailable`（L1360）。"
                                     "**只登记，不改评估器**。" % MECHANISM),
            report_tool_is_primary=("**R120 §③**：该评估器对「杠杆扫描计划」有结构性错配 ⇒ **L3 判定必须以 `tools/report_m11_l3.py`"
                                    "（四条报告量 + `c_judgeable` + `triviality_flag`）为主、评估器为辅**。"),
            protocol_filename=("生产者跑完自动写 `data/m11_p5_protocol_<plan_id>.json`（L507–536）⇒ 本计划 plan_id = "
                               "`p5_L3_sigma_cliff_bias_scan` ⇒ 文件为 `data/m11_p5_protocol_p5_L3_sigma_cliff_bias_scan.json`"
                               "（`p5_` 重复一次是代码决定的）。**R120 §①**：评估器没有 `--runs`/`--protocol`，只有 `--protocol-json`。"),
            e_criterion_under_review=("**R123 §① / R124**：跨工具 (e) 口径冲突已被精确确证 ⇒ **R109 ①② 的「(c)↔(e) 对立」撤回**，"
                                      "该条标**待复核** ⇒ 本计划**不**把 (e) 用作 L3 的判定依据（(a)/(c) 不受影响）。"),
            duration_ladder="本计划只有 T=10 s ⇒ 硬门④ 阶梯为 `unavailable`（与 L0/L1 相同，非 invalid）",
            two_ci_disagreement="L1 先例：`b0.70 σ0.30` 的 (c) 行因两 CI 法上界相对差 0.371 > 0.25 判 `invalid` ⇒ L3 若有同情形，**先对齐 CI 口径**再判",
            targets_csv_gap_closed="R115 §⑤ 的判据表缺口已由 append-only 行补齐：`m11_p5_targets.csv` **L119** `protocol_drive_rate_hz = 8.0`（本节点实读）",
            doc_approximation_discrepancy=("**如实登记**：`docs/m11_acceptance_master.md` R115 §②/① 写「零驱动 ≈9,076 脉冲/实现 @T=10 s」，"
                                           "既有产物实读为 **8,591 / 8,585 / 9,802**（`data/m11_p5_l0_runs/l0_g1.00_b1.00_…_d0_s{0,1,2}.json`），"
                                           "3 实现合计 **26,978**，`R` = **0.295987**（文档 0.296 ⇒ 逐位一致）⇒ **以产物为准**。"),
        ),
        execution_order=dict(
            order=[c["config_id"] for c in CELLS],
            within_cell="先 `baseline`（dose=1.0）后 `zero_drive`（dose=0.0）",
            rationale=[
                "① **A0 锚最先**：它与既有 L0 臂同规格 ⇒ 先做**确定性核对**（`sha256_of_trace` 逐位相等），在花钱最多的网格之前"
                "先证明「键名/默认值/口径没被改过」（R95 §⑤ 标 `h_determinism_not_evidence`）",
                "② **C0 → C1 → C2 → C3（bias 升序）**：沿 σ=0.10 悬崖面做**单向剂量-响应**，"
                "任何相邻两格的差都能归因到 bias（单变量扫描的可解释性）",
                "③ **极端档 C3(1.45) 放最后**：R118 §① 记录 bias≥1.30 后零驱动活动已趋于饱和，1.45 有「活动爆炸/变慢」风险 ⇒ "
                "放最后以免它拖垮关键档的墙钟",
                "④ **每格内 baseline 先跑**：baseline 是 (a) 的分母与窗口上界（0.10×pooled_baseline）的来源；baseline 若硬失败，"
                "该格另 3 臂不必再跑（省预算）",
                "⑤ **C2 不是第一**（与冻结计划的差别，登记）：冻结计划把 C2 提到最前，理由是占空比 ≈5% ⇒ 30 臂 ≈68 h、"
                "决定性格必须先跑（R119 §①）。本版按主 agent 最新派工改为**锚点最先、C2 第四**。"
                "**风险**：若作业在跑到 C2 之前被中断，则「决定性格」的答案仍未到手 ⇒ **建议**：① 优先保证 C2 之前的 A0/C0/C1 "
                "不返工（resume-safe 已具备），或 ② 若墙钟紧张，主 agent 可**单独先放行 C2 的 6 臂**（用生产者单臂/子计划模式）。",
            ],
            conflict_with_frozen_order="冻结计划顺序 = `C2 → A0 → C1 → C3 → C0`（R119 §①）；本版 = `A0 → C0 → C1 → C2 → C3`（主 agent v2 派工）。"
                                       "**只改臂顺序，不改任何档位/实现/协议字段** ⇒ R118 事前修订的网格不受影响。",
            resume_safety="生产者 `--plan` 模式对已存在 `.done` 的臂 `[SKIP]`（L553–557）⇒ **可中断续跑**；`--force` 才会重跑"
                          "（本计划**不得**用 `--force`；同一 out-dir 下 C2 baseline s0/s1 会被复用）。",
        ),
        cross_plan_anchor=dict(
            rule="A0 格（bias=1.00 × σ=0.30）的 6 臂在**既有 L0 计划里已按同规格跑过**（N=8192/T=10 s/settle=500 ms/"
                 "dose∈{0,1}/seed∈{0,1,2}/gmax=1.0/bias=1.0/chunk）⇒ **本计划的 A0 6 臂应与既有 L0 臂 "
                 "`sha256_of_trace` 逐位相同**；不同即说明键名/默认值/口径被改过。",
            h_determinism_not_evidence="照 **R95 §⑤**：跨计划逐位一致只证明**确定性**，**不得**当作新证据、不得计入 n_seeds、"
                                       "不得用于 (a)/(c) 的判定。",
            l3_arms={("baseline" if d == 1 else "zero_drive"): {
                "l3": _rid("A0", float(d), s), "l0": "l0_g1.00_b1.00_8192_T10000ms_d%d_s%d" % (d, s)}
                for d in (0, 1) for s in SEEDS},
            expected_l0_spikes=dict(note="来源：neural_exploration/data/m11_p5_l0_runs/*.json（本节点实读）",
                                    zero_drive={str(s): n for s, n in zip(SEEDS, (8591, 8585, 9802))},
                                    baseline={str(s): n for s, n in zip(SEEDS, (30662, 30574, 29910))}),
            expected_l0_eval=dict(file="neural_exploration/data/m11_p5_resting_eval_repl.json",
                                  pooled_baseline=91146.0, pooled_zero_drive=26978.0, R=0.295987,
                                  note="既有评估器实测（R115 记 0.296）⇒ 本计划 A0 的 (c) 应与之同量级；**这是口径核对，不是新证据**"),
            sigma_key_inertness_precedent=dict(
                claim="显式 `bias_log_sigma=0.30` ≡ 不带该键（对逐位结果惰性）",
                evidence="本节点实读：L0 `l0_g1.00_b0.70_*`（不带 σ 键）与 L1 `l1_b0.70_g1.00_s0.30_*`（显式 0.30）6/6 臂 "
                         "`sha256_of_trace` 逐位相同",
                consequence="⇒ A0 锚的逐位一致性**有依据**"),
            c2_in_progress_anchor=dict(
                note="C2 baseline 的 s0/s1 已由正式跑产出（`data/m11_p5_l3_runs/`）⇒ 本计划同 run_id 的臂将被 `[SKIP]` 复用；"
                     "其读数（52,601 / 52,572 脉冲、中位率 0.6000 Hz、`active_frac` 1.0000、`ei_ratio` 2.5970/2.6034）"
                     "**属于同一命名空间** ⇒ 正式判定的 (a) 侧已有 2 个实现的证据，**仍待** zero_drive 侧给出 (c)。"),
        ),
        selfcheck_map=dict(
            note="主 agent 条款 1–8 ⇒ 落在计划 JSON 的哪个键 + 自检报告的哪一节",
            items={
                "1 unmeasured_side_declaration（含 by_measurement_excluded）「: 」计划键 `unmeasured_side_declaration`｜报告 §5",
                "2 (a) 四条并报「: 」计划键 `a_reporting_four`（+ `reporting_contract`）｜报告 §5",
                "3 亚型分解「: 」计划键 `subtype_reporting`｜报告 §5",
                "4 同质化防线「: 」计划键 `homogenization_defense`｜报告 §6",
                "5 producer_dependency「: 」计划键 `producer_dependency`｜报告 §2.9/§5",
                "6 按档取价的预算「: 」计划键 `budget`｜报告 §7",
                "7 停机规则「: 」计划键 `stop_rules`｜报告 §3/§5",
                "8 cross_plan_anchor「: 」计划键 `cross_plan_anchor`｜报告 §5",
                "＋ R122 作废纪律「: 」计划键 `status` / `supersedes`｜报告 §0",
                "＋ 与冻结计划对齐「: 」计划键 `alignment_with_frozen_plan`｜报告 §4",
            },
        ),
    )


def build_plan() -> dict:
    arms, levels = build_arms()
    plan = dict(
        plan_id="l3_replan_bias_sigma_window_n8192",
        structure=("L3 二维角落窗口搜索（4 格 × {zero_drive, baseline} × 3 独立实现 = %d 臂；"
                   "依据 R112/R113/R114/R115/R116）" % len(arms)),
        parent_prereg=dict(
            path="neural_exploration/data/m11_p5_lever_prereg.json",
            sha256_full=sha256_of(LEVER_PREREG),
            cited=dict(
                lever="L3 二维角落：bias_scale↑（抬 active_frac/中位率）× bias_log_sigma↓（削上尾 ⇒ 压自发活动）",
                design="R115 ④（实测钉死）：`bias_scale ∈ [1.20,1.40] × bias_log_sigma ∈ [0.15,0.25]` 的二维角落；"
                       "结构模板与条款格式沿用 `m11_p5_lever_prereg.json::L2_design`（L2 已被 R110 判事前失败、不跑）",
                hypothesis="bias↑ 把 active_frac 推过 50% 并抬高中位率；σ↓ 削掉播种自发活动的上尾 ⇒ "
                           "两者**分工互补**，第一次可能同时满足 (a) 与 (c)",
                stop_rules="见本计划 `stop_rules`（8 条，按 L3 改写自 `L2_design.stop_rules`）",
            ),
            amendments=dict(
                r109_a_reporting=dict(path="neural_exploration/data/m11_p5_prereg_amend_r109_a_reporting.json",
                                      sha256=sha256_of(R109_AMEND),
                                      scope="(a) 三条并报（a_median_hz / a_active_frac / a_pop_mean_hz）"
                                            "**不改频带、不改 pass/fail**；本计划在第 4 条上追加 "
                                            "`pooled_zero_drive` 绝对值（R115 ②：窗口判定的唯一依据）"),
                l1_plan_reference=dict(path="neural_exploration/data/m11_p5_plan_l1.json",
                                       sha256=L1_PLAN_SHA, scope="**键集合模板**（本计划键集 ⊇ L1 键集）"),
            ),
        ),
        units={"T_ms": "millisecond", "settle_ms": "millisecond", "dt_ms": "millisecond",
               "wall_s": "second", "ms_per_step": "millisecond per step",
               "drive_rate_hz": "Hz per neuron", "gpu_h": "GPU-hour (=3.6e6 ms)",
               "bias_scale「: 」无量纲倍数（乘在偏置张量上）",
               "gmax_scale「: 」无量纲倍数（乘在突触权重上）",
               "bias_log_sigma「: 」无量纲（对数正态宽度；乘在标准正态分量上，不改 RNG 序列）",
               "pooled_baseline「: 」脉冲数（3 个实现求和，非率）",
               "pooled_zero_drive「: 」脉冲数（3 个实现求和，非率）",
               "R「: 」无量纲比值 = pooled_zero_drive / pooled_baseline",
               "note": "R80 §⑥：跨接口传递的量必须在接口定义处写明单位；接收方写产物前必须做量纲自检"},
        n_neuron=N, settle_ms=SETTLE_MS, dt_ms=DT_MS, drive_rate_hz=DRIVE_RATE_HZ,
        device="mps", bin_ms=100.0, smooth_frames=5, estimator="per_neuron",
        indegree=20, gmax_scale=1.0,
        bias_scale=1.00,               # 计划级默认 = A0 锚（每臂都显式带 ⇒ 仅作兜底）
        bias_log_sigma=0.30,
        delay_bins=[40, 60, 80],
        seed_policy={"net_seed": "**每臂显式**（0/1/2，照 R93 与 L0/L1 的 SEED_PAIRS 约定）",
                     "event_seed": "**每臂显式**（11/12/13 = 11+seed）",
                     "reset_seed": "=seed（由生产者在 L252 用 reset_seed=spec['seed'] 驱动）",
                     "per_arm_explicit": True,
                     "net_event_pairs": {str(k): list(v) for k, v in SEED_PAIRS.items()},
                     "rationale": "R93 第 43 条：复刻轴三源分别声明；**不设计划级 net_seed**"
                                  "（恒定注入会让 seed 无法驱动网络）。注意：同一 (net,event) 对在**不同格之间刻意复用**"
                                  "（跨格可比性 + A0 锚的逐位可复现性）⇒ 三元组在计划级**必然重复**，"
                                  "去重口径是**按 (config_id, role) 分组**（生产者 L466–490）"},
        subtypes=True, slow=True,
        tau_slow_ms=TAU_SLOW_MS, slow_inc=SLOW_INC,
        production_delivery=DELIVERY, criterion_delivery=CRITERION_DELIVERY,
        launch_prefix="caffeinate -i",
        judged_units=[dict(config_id=l["config_id"], T_ms=T_MS, mechanism=MECHANISM,
                           zero_drive_seeds=list(SEEDS), baseline_seeds=list(SEEDS),
                           bias_scale=l["bias_scale"], bias_log_sigma=l["bias_log_sigma"],
                           net_event_pairs=l["net_event_pairs"],
                           note="评估器逐 (config_id,T,mechanism) 单元复核 (a)(b)(c)(e)；"
                                "**注意**：本计划的 mechanism=%r（≠ 'none'）⇒ 评估器硬门⑤的"
                                "「无机制基线臂登记」对该单元**不可用**（见 cross_tool_notes）" % MECHANISM)
                      for l in levels],
        prerequisites=dict(
            producer_frozen="**必须**：`tools/run_m11_p5_arms.py` sha256 = %s（R116 验收后的版本，含零活动分支）"
                            "；R48：跑前不得再改生产者" % PRODUCER_SHA16,
            positive_control_required="跑前必须确认 **零活动分支已被验收**（R116 §④ 的 Z1 臂："
                                      "`zero_drive, bias=1.00, σ=0.05` ⇒ EXIT=0 且落盘 `zero_activity=true`、"
                                      "`ei.ei_ratio=null`）；**不得**为了让 L3 落盘而再改生产者（R48）",
            launch="`caffeinate -i ./.venv-m9/bin/python -m neural_exploration.tools.run_m11_p5_arms "
                   "--plan neural_exploration/data/m11_p5_plan_l3_replan.json "
                   "--out-dir neural_exploration/data/m11_p5_runs_l3`（引擎侧解释器 = .venv-m9，R34 §12.2）",
            dimension_selfcheck="生成器内建 R80 家族断言（步数恒等式 + 期望步数 + ≥200,000 下限）⇒ "
                                "`python3 -m neural_exploration.tools.make_m11_p5_plan_l3 --check`",
            preflight_replication="生产者 `--plan` 模式在**跑任何臂之前**执行 `assert_plan_replication(plan)`"
                                  "（L548）⇒ 复刻轴失效时**整轮零臂**（不是跑到一半才发现）",
            order_rule="**按 `arms` 的书写顺序跑**：A0 锚最先；每格内 baseline 先于 zero_drive。"
                       "生产者按 `plan['arms']` 顺序逐臂 spawn（L550）",
        ),
        lever_prereg_falsifier=("**证伪式（R115 ④）**：若该角落内**每一格**都出现 (i) `pooled_zero_drive < 300`"
                                "（(c) 判 unavailable）或 (ii) `R > 0.10`（(c) fail）或 (iii) 中位率 < 0.5 Hz"
                                "（(a) fail）⇒ 判 **L3 失败 ⇒ 第四维结构对立确认 ⇒ 必须换机制（不再改参数）**。"),
        frozen_tool=dict(path="neural_exploration/tools/run_m11_p5_arms.py",
                         sha256_prefix=PRODUCER_SHA16, sha256_full=sha256_of(PRODUCER),
                         n_lines=PRODUCER_LINES, n_bytes=PRODUCER_BYTES,
                         rule="R48：本计划跑前不得再改生产者；改动即作废本冻结记录"),
        cross_plan_anchor=dict(
            rule="A0 格（bias=1.00 × σ=0.30）的 6 臂在**既有 L0 计划里已按同规格跑过**（N=8192/T=10 s/settle=500 ms/"
                 "dose∈{0,1}/seed∈{0,1,2}/gmax=1.0/bias=1.0/chunk）⇒ **本计划的 A0 6 臂应与既有 L0 臂 "
                 "`sha256_of_trace` 逐位相同**；不同即说明键名/默认值/口径被改过。",
            h_determinism_not_evidence="照 **R95 §⑤**：跨计划逐位一致只证明**确定性**，"
                                       "**不得**当作新证据、不得计入 n_seeds、不得用于 (a)/(c) 的判定。",
            l3_arms={("baseline" if d == 1 else "zero_drive"): {
                "l3": _rid("a0", float(d), s), "l0": "l0_g1.00_b1.00_8192_T10000ms_d%d_s%d" % (d, s)}
                for d in (0, 1) for s in SEEDS},
            expected_l0_spikes=dict(  # 既有产物实测值（本节点实读，非估计）
                note="来源：neural_exploration/data/m11_p5_l0_runs/*.json（本节点实读）",
                zero_drive={str(s): n for s, n in zip(SEEDS, (8591, 8585, 9802))},
                baseline={str(s): n for s, n in zip(SEEDS, (30662, 30574, 29910))},
            ),
            expected_l0_eval=dict(
                file="neural_exploration/data/m11_p5_resting_eval_repl.json",
                pooled_baseline=91146.0, pooled_zero_drive=26978.0, R=0.295987,
                note="既有评估器实测（R115 记为 R=0.296）⇒ 本计划 A0 的 `p5_c_selfsustain` 值应与之同量级；"
                     "**这是口径核对，不是新证据**"),
            sigma_key_inertness_precedent=dict(
                claim="显式 `bias_log_sigma=0.30` ≡ 不带该键（对逐位结果惰性）",
                evidence="本节点实读：L0 `l0_g1.00_b0.70_*`（不带 σ 键）与 L1 `l1_b0.70_g1.00_s0.30_*`（显式 0.30）"
                         "6/6 臂 `sha256_of_trace` **逐位相同**；另有 `tools/verify_m11_producer_inert.py` 的独立实证",
                consequence="⇒ A0 锚的逐位一致性**有依据**（不是「应该会一样」的猜测）"),
        ),
        levels=levels, arms=arms,
    )

    # ---- 追加条款块（主 agent 清单第 1–8 条）----
    plan["core_falsifiable_assertion"] = dict(
        statement=("**在 `bias_scale ∈ [1.20, 1.40]` × `bias_log_sigma ∈ [0.15, 0.25]` 的二维角落内，"
                   "存在同时满足 (a) 与 (c) 的点。**"),
        a_definition="**(a)**（主判口径，频带 `[0.5, 2.0] Hz` **逐字不变**）：基准臂 `per_neuron` "
                     "**逐神经元中位率**（逐神经元计数 / T 的中位数）",
        c_definition="**(c)**：`R = pooled_zero_drive / pooled_baseline ≤ 0.10`，**且** `pooled_zero_drive ≥ 300`（**硬门①**）",
        window="**目标是一个窗口，不是一个极限**（R115 ②）：`pooled_zero_drive ∈ [300, 0.10 × pooled_baseline]`。"
               "下界 = 硬门①（`m11_p5_targets.csv` L48 `selfsustain_zero_drive_min_spikes=300`）；"
               "上界 = `selfsustain_ratio_max=0.10`（同表 L46）。"
               "**R 太小（零驱动被打干净）会让 (c) 变 `unavailable` 而不是 `pass`。**",
        why_self_consistent="(b) 现为 `unavailable`（真实 ROI 掩膜维缺失）⇒ 不进任何门断言；(a)/(c) 均为 `final=1` 的"
                            "已定稿判定带 ⇒ 双目标可判且有判别力（L0/L1 已证各自可判但不可同时满足 ⇒ 断言当前为假）。",
        falsifiers=[
            "① **主证伪式**：该角落内**每一格**都命中 (i) `pooled_zero_drive < 300` 或 (ii) `R > 0.10` 或 "
            "(iii) 中位率 < 0.5 Hz ⇒ 判 **L3 失败 ⇒ 第四维结构对立确认 ⇒ 必须换机制（不再改参数）**",
            "② **机制专属证伪**：若 σ↓ 时**基准臂** `a_active_frac` 也大幅下降（R113 ④ 的反向）⇒ "
            "该杠杆是**全局性**的、「尾部选择性」预言被证伪 ⇒ L3b 与 L0/L1 同属一列（须如实登记，不得改判据）",
            "③ **不可判定 ≠ 失败**：某格 (c) 判 `unavailable`（`pooled_zero_drive<300`，或两 CI 法不一致 ⇒ `invalid`）"
            "时**不得记为 fail，也不得记为 pass**（M10 D1 家族；L1 已有 `two_ci_methods_disagree` 先例）",
        ],
        positive_outcome="某格 (a) ∧ (c) 同时成立（含窗口条件与 CI 条件）⇒ 进入定稿确认，并**只**得结论"
                         "「(a) 与 (c) 在这种机制组合下可同时满足」；**不得**外推到未测侧。",
    )
    plan["window_definition"] = dict(
        lower_bound_pooled_zero_drive=300,
        lower_bound_source="neural_exploration/data/m11_p5_targets.csv L48 `selfsustain_zero_drive_min_spikes`（硬门①；"
                           "1/√300 = 5.8% 相对不确定度）",
        upper_bound_formula="0.10 × pooled_baseline（同一 (config_id,T) 的 3 个实现求和）",
        upper_bound_source="neural_exploration/data/m11_p5_targets.csv L46 `selfsustain_ratio_max=0.10`",
        a0_numeric_example=dict(
            pooled_baseline_3impl=91146.0, pooled_zero_drive_3impl=26978.0,
            window_pooled_zero_drive=[300, 9114.6], R_measured=0.295987,
            verdict="A0 的 `pooled_zero_drive` 26,978 **高于**上界 9,114.6 ⇒ (c) fail（R=0.296 > 0.10）⇒ "
                    "A0 预期**不满足** (c)（**预期**，不是结论）",
            source="neural_exploration/data/m11_p5_resting_eval_repl.json"
                   "（zero_drive_spikes 26978.0 / R=0.295987；本节点实读）"),
        requirement="**每档必须报 `pooled_zero_drive` 的绝对值**，不得只报 R —— "
                    "`unavailable` 与 `pass` 的区别全在这个绝对值上（R115 ②）",
    )
    plan["a_reporting_four"] = dict(
        primary_unchanged=dict(estimator="per_neuron（逐神经元计数/T 的中位数）", band_hz=[0.5, 2.0],
                               role="**(a) 的主判口径，频带与判定规则逐字不变**"),
        required=[
            dict(id="a_median_hz", definition="逐神经元计数/T 的中位数（Hz）",
                 note="= 主口径，照旧报出（R109 修订 ①）"),
            dict(id="a_active_frac", definition="测量窗 T 内 **≥1 个脉冲**的神经元占比（0–1）",
                 note="R109 修订 ②：摊平程度的直接度量；**σ↓ 是否有「选择性」全看它**（R113 ④ 的可证伪预言）"),
            dict(id="a_pop_mean_hz", definition="群体均值率 = 总脉冲/(N·T)（Hz）", note="R109 修订 ③：总量口径"),
            dict(id="pooled_zero_drive", definition="零驱动臂 3 个实现的脉冲数**之和**（绝对量，非比值）",
                 note="**本计划新增的第 4 条**（R115 ②）：窗口判定的唯一依据；`unavailable` 与 `pass` 全靠它区分"),
        ],
        ban="**不得**用 `a_active_frac` 代替中位率判 (a)（R109 修订 `no_gate_change`/`ban` 逐字保留）",
        source="neural_exploration/data/m11_p5_prereg_amend_r109_a_reporting.json（sha %s）" % R109_AMEND_SHA[:16],
    )
    plan["subtype_reporting"] = dict(
        requirement="按 **E/PV/SST/VIP** 分别报**中位率**与**活跃占比（≥1 脉冲）**（R111 ② / R114 ③）",
        source_column="臂产物 `<run_id>_counts.csv` 的 `cell_type` 列（生产者 L306–311 写入 E/PV/SST/VIP）",
        why="(a) 的文献带本身是**类级**的（`rate_median_hz_exc_lo/hi` = 0.5/1.0；`inh_lo/hi` = 1.5/2.0，"
            "见 targets.csv L30–L33）⇒ 只报全群中位数会掩盖「E 类不达标但 I 类达标」的混合情形",
        not_a_gate="类级分解为**描述性**；本计划的判定仍只用全群 `estimator=per_neuron` 中位率（不得改口径）",
    )
    plan["homogenization_defense"] = dict(
        risk="σ↓ 可能「用同质化换取中位率」（R112 ④ / R110 ④ 的伪解警告）⇒ 必须报同质化指标",
        required_metrics=[
            dict(id="frac_nonzero_slow_var",
                 source="臂 JSON `slow_trace_stats.readback.frac_nonzero`（引擎 `slow_readback()` 暴露；"
                        "引擎 src/mouse_cortex_engine.py L1073）",
                 definition="慢变量 t_slow 中 >0 的神经元占比（逐神经元状态的稀疏度）",
                 note="**引擎已实测并落盘** ⇒ 可直接复算，不是自造量"),
            dict(id="frac_nonzero_spikes",
                 source="臂产物 `<run_id>_counts.csv`（`count > 0` 的行数 / N）",
                 definition="测量窗内 ≥1 脉冲的神经元占比 —— **与 `a_active_frac` 是同一个量**，"
                            "报告时只算一次并显式声明二者同源（防「两个指标」的虚假多重确认）"),
            dict(id="count_dispersion",
                 source="同 `<run_id>_counts.csv` 的 `count` 列",
                 definition="逐神经元计数的离散度：median/p90/IQR/CV（**可从产物逐位复算**）",
                 note="σ↓ 若真为「尾部选择性」，则上尾（p90/最大值）应被削而中位数/活跃占比保持；"
                      "若 IQR/CV 同时塌缩 ⇒ 同质化伪解"),
        ],
        engine_synchrony_metric="**引擎无可用同步性/规则性指标**：`src/mouse_cortex_engine.py` 内无任何"
                                "同步/规则性读出（`grep` 命中的 `ei_sync_every` 只是**每 1000 步把 E/I 张量同步回 "
                                "Python 的间隔**，与神经元同步性无关）⇒ **不报同步性指标，登记为未确认项**；"
                                "**禁止自造无法复算的量**",
        unconfirmed=["跨神经元同步性 / 发放规则性（CV_ISI / Fano）**不可从现有产物计算**（引擎不导出逐神经元脉冲时刻，"
                     "只导出逐神经元计数）⇒ 登记为未确认项，不得用计数离散度冒充"],
    )
    plan["unmeasured_side_declaration"] = dict(
        discipline="**纪律第五十八条**：未测侧必须显式登记 —— **未测 ≠ 已排除**（不得当作「已被否掉」）",
        still_unmeasured=[
            dict(side="`bias_scale > 1.40`", note="完全未测。R112 的 `bias=1.60` 探针（零驱动 spk 7,410）是 T=1 s "
                                                 "单实现探针 ⇒ 只作方向性支持，**不作判据**"),
            dict(side="`bias_scale < 1.00`（与 σ↓ 的组合）", note="L0/L1 测过 0.50/0.70，但都是 σ≥0.30；"
                                                               "「低 bias × 低 σ」这一角完全未测"),
            dict(side="`bias_log_sigma < 0.15`", note="σ=0.10/0.05 只在 bias=1.00 处测过（且零驱动 →0 硬失败）；"
                                                     "σ<0.15 × bias=1.30 未测"),
            dict(side="`bias_log_sigma > 0.30`（与 bias↑ 的组合）", note="σ=0.45/0.60/0.90 只在 bias=0.70 测过（L1）"),
            dict(side="`drive_rate_hz` 的抬升", note="**预注册禁止作为标定手段**（docs/m11_b3_g1_design.md L146）；"
                                                    "其剂量-响应诊断未做（本计划**不含**任何驱动抬升档）"),
            dict(side="慢适应/STP 与 (bias,σ) 的组合", note="`slow_inc`/`tau_slow_ms` 在本计划全部固定在默认档"
                                                          "（6.0/1000）；L2 的慢适应放大档**未跑**（R110 判 L2 事前失败）"),
            dict(side="外部驱动侧", note="外部事件表的构造、事件表泄漏、`event_seed` 与驱动的交互均未测"),
            dict(side="亚型选择性杠杆", note="PV/SST/VIP 的权重或比例、层流比例、`indegree`、`gmax_scale`、"
                                           "`delay_bins` 在本计划中**全部固定在基线** ⇒ 未测"),
            dict(side="T 维度（时长依赖性）", note="本计划只有 T=10 s ⇒ 评估器硬门④（{1,2,4}×T_min 阶梯）在 L3 上"
                                                  "为 `unavailable`（与 L0/L1 相同）；(d) 时长阶梯未测"),
        ],
        statement="以上方向**未测 ≠ 已排除**；L3 的任何结论**只**覆盖 `bias ∈ [1.00,1.30] × σ ∈ [0.15,0.30]` "
                  "这 4 个格（且每格只覆盖 `drive_mult ∈ {0,1}`、T=10 s、slow 默认档）。",
    )
    plan["producer_dependency"] = dict(
        requirement="本计划**要求**生产者已含零活动分支（R113 ⑤ 的硬前置；R116 已验收）",
        sha256_full=sha256_of(PRODUCER), sha256_prefix=PRODUCER_SHA16,
        n_lines=PRODUCER_LINES, n_bytes=PRODUCER_BYTES,
        behaviour="零驱动臂 0 脉冲 ⇒ **不是硬失败**：正常落盘并写 `zero_activity=true`、`spikes_total=0`、"
                  "`ei.ei_ratio=null`（显式缺席）；`role=baseline` 的 0 脉冲**仍是硬失败**（EXIT=1）",
        interface_contract=dict(
            zero_activity="**顶层**键 `arm[\"zero_activity\"]`（引擎 `run_one` L352；仅该分支出现 ⇒ "
                          "既有臂 JSON 逐字节不变；键缺失 ≠ false，与 `true` 可区分）",
            zero_activity_criterion="**顶层**键 `arm[\"zero_activity_criterion\"]`（L353–359，人类可读理由）",
            ei_ratio="**在 `ei` 子字典内**：`arm[\"ei\"][\"ei_ratio\"]`（L324）；零活动路径 ⇒ JSON `null`"
                     "（≠ 0.0；「缺席」与「测得 0」可区分）",
            spots="读键方**必须**用 `arm[\"ei\"][\"ei_ratio\"]`，**不得**写 `arm[\"ei_ratio\"]`（会静默拿到 None）",
            stdout="`[DONE]` 行对零活动臂追加 `ZERO_ACTIVITY=1`（L439–440）；非零活动臂该行逐字节不变",
        ),
        line_refs=dict(zero_activity_branch="L285–300（`_zero_drive_expected` 取「语义轴 ∨ 数值轴」）",
                       unconditional_assert_kept="L289–291（非零驱动臂原断言逐字保留）",
                       done_line="L434–442", interface_note="L348–359"),
        caveats=("R116 ⑥ 登记的两处 caveat 依旧有效：① `zero_activity == false` 分支（零驱动臂 E 积分为 0 但"
                 "脉冲数 >0）**有代码无覆盖**；② 键位契约见上（顶层 vs `ei` 子字典）。"
                 "R116 ⑥3 另记：**不得**跑 `--self-test-sigma`（会覆写仓库产物 `data/m11_lever_sigma_selftest.json`）。"),
    )
    plan["stop_rules"] = [
        "① 某格使**主断言成立**（(a) ∧ (c)，且 `pooled_zero_drive ∈ [300, 0.10×pooled_baseline]`）⇒ **停扫**，"
        "进入定稿确认。**【谁在何时执行】主 agent，在该格 6 臂产物 + 评估器行到齐后**",
        "② 某格零驱动臂 0 脉冲（`zero_activity=true`）⇒ 该格 (c) 判 **`unavailable`**（既非 fail 亦非 pass；"
        "**硬门①的 300 脉冲下界在此生效**）；**该格 baseline 若也 0 脉冲 ⇒ 硬失败**（驱动侧零响应 = 网络真坏）"
        "⇒ 停并查口径。**【谁在何时执行】评估器（硬门①，validate_p11_resting.py L814）判 unavailable；"
        "主 agent 读 `[DONE]` 的 `ZERO_ACTIVITY=1` + `role` 后决定是否停**",
        "③ 连续 **2 格**既未使中位率向 `[0.5,2.0]` 靠近、也未使 R 下降 ⇒ **停**，报「无改善」。"
        "**【谁在何时执行】主 agent，在第 2 格 6 臂产物到齐后**",
        "④ 任一格出现 `hard_failures` 非空（慢变量读回断言、双估计器不一致、轨迹长度⟺(T,dt) 口径等）⇒ "
        "该格 **`invalid`**；**≥2 格**如此 ⇒ **停并查口径**（**不得**记为「机制无效」，M10 D1 家族）。"
        "**【谁在何时执行】评估器（硬门③/`hard_failures`）；主 agent 在评估器输出后**",
        "⑤ 任一格的 `(net_seed, event_seed)` 对重复（计划级断言违反）⇒ **跑前整轮停**。"
        "**【谁在何时执行】生产者 `assert_plan_replication`，在 `--plan` 模式**跑任何臂之前**（L548，"
        "失败即 `SystemExit(3)`）**",
        "⑥ 触及预算包络（`gpu_h_mid` 1.7937 / `upper` 2.1036，或墙钟超出包络）⇒ **停**，**不得静默超支**。"
        "**【谁在何时执行】主 agent，逐臂比对累计 `wall_s` 与预算**",
        "⑦ **预注册禁止**：任何以**驱动抬升**为标定手段的档**不得进 L3**（`drive_rate_hz` 的抬升；"
        "`docs/m11_b3_g1_design.md` **L146**「禁」行原文：提高 λ **不得**作为标定手段，只能作剂量-响应诊断；"
        "`wrong_usage_consequence` 亦登记在 `m11_p5_targets.csv` L119 `protocol_drive_rate_hz` 行）。"
        "**【谁在何时执行】本计划**跑前**（由生成器断言：`drive_mult ∈ {0.0,1.0}`、无 `drive_rate_hz` 覆盖、"
        "`levels` 无驱动档）+ 主 agent 交办复核**",
        "⑧ 4 格**全部**满足证伪式（每格至少命中 (i)/(ii)/(iii) 之一）⇒ 判 **L3 失败 ⇒ 第四维结构对立确认 ⇒ "
        "必须换机制（不再改参数）**；若失败只是采样/口径问题（`hard_failures` 非空）⇒ **先修口径重跑**，"
        "**不得**记为机制失败；若某格仅 (c) 判 `unavailable`（R 与窗口不可判定）⇒ **该格不计入证伪式的 (i)/(ii)**，"
        "须先按 R115 ② 的窗口判据补测。**【谁在何时执行】主 agent，在 4 格评估器输出齐备后**",
    ]
    plan["budget"] = _budget(arms)
    plan["b_reporting_form"] = ("**沿用 R101/L1 的 R57 四条**：(b) 一律**描述性并列、不作 gate**"
                                "（代理口径 zero_count / 无真实 ROI 掩膜、样本 thin 单 field×2,000 帧 ≈317 s、"
                                "**不引用 0.48–0.49 窄带** ⇒ 以对 0.49 的比值/上界报告、不判 pass/fail）。"
                                "本计划**不**启用 0.48–0.49 窄带，**不**改 (b) 的 `unavailable` 状态"
                                "（`m11_p5_prereg_amend_r109_a_reporting.json::b_unchanged` 逐字保留）。")
    plan["forbidden_levers"] = [
        dict(lever="提高背景驱动 λ（`drive_rate_hz` / `drive_mult > 1`）",
             prohibition="**预注册禁止作为标定手段**（`docs/m11_b3_g1_design.md` **L146**）",
             reason="提高 λ 同时抬高零驱动臂（若事件表泄漏）、压低静默 ⇒ 必然触发张力；"
                    "M10 A4 已付过同一学费（L137：抬高驱动去命中 (a) 会同时把静默压出带）",
             allowed_only_as="剂量-响应诊断（**不在本计划内**；本计划 24 臂的 `drive_mult` 只有 0.0/1.0）"),
        dict(lever="用「减少 settle」或「缩短 T」降成本",
             prohibition="禁（会违反 R80 步数断言与 T_min 口径）",
             reason="settle=500 ms ⇒ 210,000 步是预注册口径；缩短即失去与既有 L0/L1 臂的可比性",
             allowed_only_as="—"),
    ]
    plan["prior_probe_evidence"] = dict(
        discipline="**以下全部是 T=1 s、单实现、未预注册的探针（R112/R113/R115）⇒ 只作方向性支持，"
                   "不作判据、不进任何门、不得记为「已测」**",
        probes=[
            dict(arm="l3pb_b1.00s0.30 baseline（bias=1.00, σ=0.30）", spikes_total=2901, ms_per_step=1.1872,
                 ei_ratio=1.8172197634873357,
                 file="neural_exploration/data/m11_p5_l3_feasibility/l3pb_b1.00s0.30_8192_T1000ms_d1_s0.json"),
            dict(arm="l3pb_b1.00s0.10 baseline（bias=1.00, σ=0.10）", spikes_total=2260, ms_per_step=1.07596960835,
                 ei_ratio=1.532591700451025,
                 file="neural_exploration/data/m11_p5_l3_feasibility/l3pb_b1.00s0.10_8192_T1000ms_d1_s0.json"),
            dict(arm="l3pb_b1.30s0.10 baseline（bias=1.30, σ=0.10）", spikes_total=5149, ms_per_step=1.4856083625,
                 ei_ratio=2.5767486315057613,
                 file="neural_exploration/data/m11_p5_l3_feasibility/l3pb_b1.30s0.10_8192_T1000ms_d1_s0.json"),
            dict(arm="l3probe_b1.30s0.30 zero_drive（bias=1.30, σ=0.30）", spikes_total=4126, ms_per_step=1.43785286,
                 ei_ratio=6.040639606828006,
                 file="neural_exploration/data/m11_p5_l3_feasibility/l3probe_b1.30s0.30_8192_T1000ms_d0_s0.json"),
            dict(arm="l3win_b130_s020_zd zero_drive（bias=1.30, σ=0.20）", spikes_total=4875,
                 ms_per_step=1.03712800205, ei_ratio=5.538575231525635,
                 file="neural_exploration/data/m11_p5_l3_feasibility/l3win_b130_s020_zd_8192_T1000ms_zero_drive_s0.json"),
            dict(arm="l3win_b130_s020_bl baseline（bias=1.30, σ=0.20）", spikes_total=5546,
                 ms_per_step=1.5026, ei_ratio=2.7059,
                 file="neural_exploration/data/m11_p5_l3_feasibility/l3win_b130_s020_bl_8192_T1000ms_baseline_s0.json"),
            dict(arm="l3win_b130_s015_zd zero_drive（bias=1.30, σ=0.15）", spikes_total=4666,
                 ms_per_step=0.9693, ei_ratio=5.0065,
                 file="neural_exploration/data/m11_p5_l3_feasibility/l3win_b130_s015_zd_8192_T1000ms_zero_drive_s0.json"),
            dict(arm="l3win_b130_s015_bl baseline（bias=1.30, σ=0.15）", spikes_total=5295,
                 ms_per_step=1.4259, ei_ratio=2.6482,
                 file="neural_exploration/data/m11_p5_l3_feasibility/l3win_b130_s015_bl_8192_T1000ms_baseline_s0.json"),
            dict(arm="l3win_b130_s010_zd zero_drive（bias=1.30, σ=0.10）", spikes_total=46,
                 ms_per_step=0.6434, ei_ratio=127.0544,
                 note="**零驱动臂 `ei_ratio=127.05` 是异常值**（零驱动下极少数细胞发放 ⇒ i_exc 极小 ⇒ 比值爆炸）"
                      "⇒ **不得与基准臂的 (e) 混用**（细分登记见 docs/m11_l3_resume.md §五）",
                 file="neural_exploration/data/m11_p5_l3_feasibility/l3win_b130_s010_zd_8192_T1000ms_zero_drive_s0.json"),
            dict(arm="l3probe_b1.60s0.30 zero_drive（bias=1.60, σ=0.30）", spikes_total=7410,
                 ms_per_step=1.69179255, ei_ratio=6.067688113548535,
                 file="neural_exploration/data/m11_p5_l3_feasibility/l3probe_b1.60s0.30_8192_T1000ms_d0_s0.json"),
        ],
        expected_risk=("**B1（本计划的主目标格）在 T=1 s 探针里已经有零驱动实测 4,875 脉冲/秒**"
                       "（`l3win_b130_s020_zd`，σ=0.20 > σ=0.30 探针的 4,126 ⇒ **在该 bias 下 σ↓ 并未压低零驱动活动**）。"
                       "若 T=10 s 的零驱动活动与之同量级（线性外推 ≈4.9×10⁴）⇒ B1 的 R 会远大于 0.10 ⇒ (c) fail。"
                       "**但这是外推，且 T=1 s 探针含 500 ms settle、慢适应在长窗可能压低活动 ⇒ 两个方向都不能假定。**"
                       "本计划据此**登记「预期可能命中证伪式」，并因此把 A0 锚排在最先（先把口径与预算钉死）"),
        single_impl_R_estimate={
            "bias=1.30, σ=0.20": 4875 / 5546.0, "bias=1.30, σ=0.15": 4666 / 5295.0,
            "bias=1.30, σ=0.10": 46 / 5149.0,
            "note": "**单实现、T=1 s** 的 R 粗估（分子/分母各为 1 个实现的 spk）⇒ 只作方向性支持，"
                    "**不是**判据（(c) 判 pooled 3 实现；且 T=1 s 计数粗量化）。",
        },
        conflict_with_pinned_grid=(
            "**🔴 本节点必读的冲突登记（不得隐瞒）**：主 agent 派工把 L3 目标钉死为 "
            "`bias ∈ [1.20,1.40] × σ ∈ [0.15,0.25]`（依据 R115 ④，其定位窗口在 σ≈0.15–0.25）。"
            "但 **R118 之后的探针实测**（上表 4 条，本节点实读）显示：在 bias=1.30 下 "
            "**σ=0.20 ⇒ R≈0.879、σ=0.15 ⇒ R≈0.881（均远超 0.10）**，而 **σ=0.10 ⇒ 零驱动 46 脉冲、"
            "R≈0.0089**（但 46 < 300 ⇒ 按硬门① 判 `unavailable`，仍不是 pass）⇒ "
            "**σ 在此不是旋钮而是「悬崖」**：窗口的真实位置在 σ∈(0.10, 0.15) 这一极窄区间，"
            "**本计划的 B1/B2 两格（σ=0.20/0.15）按探针预期会命中证伪式的 (ii)**。"
            "**⇒ 本计划照主 agent 派工逐字执行（4 格 24 臂），但把该冲突显式登记**："
            "① 这 24 臂的主要价值是**把 T=1 s 单实现探针读数升级为 T=10 s × 3 实现的正式判据**"
            "（R118 §④ 的三处外推逐条证实或证伪）；② 若主 agent 要**命中**工作点，"
            "应把格设在 σ∈(0.10,0.15)（既有的正式跑网格 v2 已用 σ=0.10，见 "
            "`data/m11_p5_plan_l3.frozen_for_run.json` / `docs/m11_l3_resume.md`）"
            "⇒ **本计划与既有正式跑互补、不重复其格位**（既有 v2 的 5 格为 C2(1.30,0.10)/A0(1.00,0.30)/"
            "C1(1.15,0.10)/C3(1.45,0.10)/C0(1.00,0.10)）。"),
        not_judged="以上**不得**用于判 (a)/(c)，**不得**用于免除实跑，**不得**计入任何 n_seeds",
    )
    plan["reporting_contract"] = dict(
        per_arm_required=[
            "臂 JSON（生产者自动落盘）：`spikes_total`、`agg.spk_total`、`ei.ei_ratio`、`ei.i_exc_integral`、"
            "`wall_s`、`ms_per_step_steady`、`sha256_of_trace`、`seeds.triple`、`n_steps_meas/n_steps_settle`",
            "逐神经元计数表 `<run_id>_counts.csv`（`neuron_id,count,cell_type`）⇒ 中位率 / `a_active_frac` / "
            "`a_pop_mean_hz` / 亚型分解 / `count_dispersion` 的**唯一可复算来源**",
            "群体轨迹 `<run_id>_poptrace.npy`（float32 逐步脉冲数）⇒ 双估计器一致性（硬门③）",
        ],
        derived_required=["`pooled_zero_drive` 绝对值（3 实现求和）+ 窗口判定",
                          "`R` 及其 CI（评估器 `p5_c_selfsustain`，两 CI 法并列）",
                          "`frac_nonzero_slow_var`（引擎读回）与 `count_dispersion`（counts 复算）"],
        ban=["不得只报 R 而不报 `pooled_zero_drive` 绝对值（R115 ②）",
             "不得用 `a_active_frac` 代替中位率判 (a)（R109 修订）",
             "不得把 `unavailable`/`invalid` 记为 fail 或 pass（M10 D1 家族）",
             "不得把跨计划逐位一致当成新证据（R95 ⑤）"],
    )
    plan["cross_tool_notes"] = dict(
        evaluator_consequence_of_mechanism=(
            "本计划 24 臂的 `mechanism=%r`。评估器按 `(config_id, T_ms, mechanism)` 组建判据单元"
            "（`tools/validate_p11_resting.py` L625–627）⇒ L3 的 4 个格各自成单元，(a)/(c)/(e) **会被正常计算**；"
            "**但**硬门⑤的「无机制基线臂登记」要求同 (config_id,T_ms) 存在 `mechanism=='none'` 且 `drive_mult==1.0` 的臂"
            "（L660–670）⇒ 对 L3 单元**不可用** ⇒ 该单元 (a)/(b) 行的 `discriminative` 为 `None`，"
            "且 G1 至多 `unavailable`（L1360 的降级规则），**不可能**被判 `pass`。"
            "**这是登记缺口，不是结果变化**；由于我**不得改评估器**，此处**只登记**：若要补足硬门⑤，需由主 agent 决定"
            "（例如另加一组 `mechanism='none'` 的同配置对照臂，但那会改变 24 臂的预注册结构 ⇒ **不得**在本计划内擅自加）。"
            % MECHANISM),
        duration_ladder="本计划只有 T=10 s ⇒ 硬门④ 阶梯在 L3 上 `unavailable`（与 L0/L1 相同，非 invalid；"
                        "参见 `data/m11_p5_l1_eval.json::duration_ladder` 的单 T 先例）",
        two_ci_disagreement="L1 先例：`b0.70 σ0.30` 的 (c) 行因两 CI 法上界相对差 0.371 > 0.25 判 **`invalid`**"
                            "（`data/m11_p5_l1_eval.csv` 该行 reason=`two_ci_methods_disagree`）⇒ L3 若有同情形，"
                            "**先对齐 CI 口径**再判，**不得**把 `invalid` 当 fail/pass",
        coexistence_with_running_l3=(
            "**本计划不覆盖、不修改任何既有计划 JSON**：`data/m11_p5_plan_l3.json`（tracked，"
            "sha256 `4aab49b00177a22e21f65b884bd29d95a8e7333e76cb109ae8fcd8bb9bd94615`，30 臂）"
            "与其冻结副本 `data/m11_p5_plan_l3.frozen_for_run.json` **均在盘上、且正在被正式跑使用**"
            "（输出目录 `data/m11_p5_l3_runs/`）⇒ 本节点的新计划落在**新文件** "
            "`data/m11_p5_plan_l3_replan.json`，生成器内置 `FORBIDDEN_OUT_PATHS` 拒绝写入那两个路径。"
            "**本节点的一次事故已如实登记**：首轮生成时误把 `data/m11_p5_plan_l3.json` 覆盖，"
            "随即从 `git show HEAD:…` **逐字节恢复**（恢复后 sha 与冻结副本一致 ⇒ 正式跑未受影响）。"),
        targets_csv_gap_closed="R115 ⑤ 登记的「判据表缺口」已由 append-only 行补齐：`m11_p5_targets.csv` **L119** "
                               "`protocol_drive_rate_hz = 8.0`（本节点实读，final 列见该行）",
        doc_approximation_discrepancy=(
            "**如实登记（不得编造）**：`docs/m11_acceptance_master.md` R115 §②/① 写「零驱动 ≈9,076 脉冲/实现 @T=10 s」，"
            "但既有产物实读为 **8,591 / 8,585 / 9,802**（seed 0/1/2，`data/m11_p5_l0_runs/"
            "l0_g1.00_b1.00_8192_T10000ms_d0_s{0,1,2}.json`），**3 实现合计 26,978**（与文档 §② 的 ≈27,200 一致，"
            "差 0.8%），而 R=**0.295987** 与文档的 0.296 **逐位一致**。⇒ **单实现均值 8,992.7 与文档的 9,076 差 0.9%**"
            "（对 seed 0 的单值差 5.6%）。**以产物为准**；差异只影响「每实现约数」的表述，不影响窗口与 R。"),
    )
    plan["execution_order"] = dict(
        order=[c["config_id"] for c in CELLS],
        within_cell="先 `baseline`（dose=1.0）后 `zero_drive`（dose=0.0）",
        rationale=[
            "① **A0 锚最先**：它与既有 L0 臂同规格 ⇒ 可做**确定性核对**（`sha256_of_trace` 逐位相等），"
            "在花钱最多的网格之前先证明「键名/默认值/口径没被改过」（R48/R95 ⑤）",
            "② 某格把网络**压死**（0 脉冲）或**炸掉**（活动爆炸/超时）时，已跑档必须保住 ⇒ "
            "生产者失败策略为「失败 1 臂继续、累计 ≥2 臂停」（L576–583）⇒ 把最便宜、最有信息量的 A0 放最前损失最小",
            "③ 每格内先 `baseline`：**baseline 是 (a) 与窗口上界（0.10×pooled_baseline）的分母**；"
            "若 baseline 先失败（硬失败 ⇒ 网络真坏），该格另 3 臂**不必再跑**（省预算）",
            "④ A1 → B1 → B2 顺序 = 「先单侧、后角落、再窗口下沿」：A1 提供「σ 不动时 bias↑ 单独做什么」的对照，"
            "缺了它就无法把 B1 的变化拆成 bias 项与 σ 项",
        ],
        resume_safety="生产者 `--plan` 模式对已存在 `.done` 的臂 `[SKIP]`（L553–557）⇒ **可中断续跑**；"
                      "`--force` 才会重跑（本计划**不得**用 `--force`：会浪费已完成的 GPU 时间）",
    )
    plan["selfcheck_map"] = dict(
        note="主 agent 清单第 1–8 条 ⇒ 落在计划 JSON 的哪个键 + 自检报告的哪一节",
        items={
            "1 unmeasured_side_declaration「: 」计划键 `unmeasured_side_declaration`｜报告 §5",
            "2 (a) 四条并报「: 」计划键 `a_reporting_four`（+ `reporting_contract`）｜报告 §5",
            "3 亚型分解「: 」计划键 `subtype_reporting`｜报告 §5",
            "4 同质化防线「: 」计划键 `homogenization_defense`｜报告 §5",
            "5 producer_dependency「: 」计划键 `producer_dependency`（+ `prerequisites.producer_frozen`）｜报告 §4/§5",
            "6 按档取价的预算「: 」计划键 `budget`｜报告 §7",
            "7 停机规则「: 」计划键 `stop_rules`｜报告 §5",
            "8 cross_plan_anchor「: 」计划键 `cross_plan_anchor`（+ `prerequisites.preflight_replication`）｜报告 §5",
        },
    )
    plan.update(_v2_blocks(plan))          # **网格 v2 覆盖块**（R118）
    return plan


def run_assertions(plan: dict) -> dict:
    """**全部断言（R80 家族 + 主 agent 清单）**；任一失败 ⇒ AssertionError（非零退出）。"""
    arms = plan["arms"]
    facts = {}

    # ① 步数恒等式（R80 家族）
    steps = int(round((plan["settle_ms"] + T_MS) / plan["dt_ms"]))
    assert plan["dt_ms"] == DT_MS, plan["dt_ms"]
    assert plan["settle_ms"] == SETTLE_MS, plan["settle_ms"]
    assert steps == EXPECTED_STEPS, ("步数断言失败：round((%s+%s)/%s)=%d ≠ %d"
                                     % (plan["settle_ms"], T_MS, plan["dt_ms"], steps, EXPECTED_STEPS))
    assert steps >= MIN_STEPS, "步数低于经验下限 %d" % MIN_STEPS
    for a in arms:
        assert float(a["T_ms"]) == T_MS, (a["run_id"], a["T_ms"])
        assert int(round((float(a["settle_ms"]) + float(a["T_ms"])) / DT_MS)) == EXPECTED_STEPS, a["run_id"]
    facts["steps_per_arm"] = steps

    # ② 臂数 / 唯一性
    assert len(arms) == 24, "臂数必须为 24，实为 %d" % len(arms)
    rids = [a["run_id"] for a in arms]
    assert len(set(rids)) == len(rids), "run_id 必须唯一"
    tri = [(a["config_id"], a["role"], a["seed"]) for a in arms]
    assert len(set(tri)) == len(tri), "(config_id, role, realization) 必须唯一"
    facts["n_arms"] = len(arms)
    facts["n_zero_drive"] = len([a for a in arms if a["role"] == "zero_drive"])
    facts["n_baseline"] = len([a for a in arms if a["role"] == "baseline"])

    # ③ 每臂键集合（逐字）
    for a in arms:
        assert tuple(sorted(a.keys())) == tuple(sorted(REQUIRED_ARM_KEYS)), \
            ("臂键集合不符：%s ⇒ 缺 %r / 多 %r" % (a["run_id"],
                                                 sorted(set(REQUIRED_ARM_KEYS) - set(a.keys())),
                                                 sorted(set(a.keys()) - set(REQUIRED_ARM_KEYS))))
    facts["arm_keys"] = list(REQUIRED_ARM_KEYS)

    # ④ 复刻轴：(每格, 每角色) 内 (net_seed,event_seed) 两两互异且恰等于预注册对
    for cfg in {a["config_id"] for a in arms}:
        for role in ("baseline", "zero_drive"):
            grp = [a for a in arms if a["config_id"] == cfg and a["role"] == role]
            assert len(grp) == 3, (cfg, role, len(grp))
            pairs = [(a["net_seed"], a["event_seed"]) for a in grp]
            assert len(set(pairs)) == 3, ("格 %s 角色 %s 的 (net,event) 对不互异：%r" % (cfg, role, pairs))
            for a in grp:
                assert (a["net_seed"], a["event_seed"]) == SEED_PAIRS[a["seed"]], (a["run_id"], pairs)
    all_pairs = sorted({(a["net_seed"], a["event_seed"]) for a in arms})
    triples = sorted({(a["net_seed"], a["event_seed"], a["seed"]) for a in arms})
    facts["distinct_pairs_plan_wide"] = all_pairs
    facts["distinct_triples_plan_wide"] = triples
    facts["triple_rows_plan_wide"] = sorted(
        [("net%d|ev%d|reset%d" % (a["net_seed"], a["event_seed"], a["seed"]), a["run_id"])
         for a in arms])
    facts["triple_planwide_note"] = ("计划级三元组**刻意重复**：3 个三元组 × 8 臂（每格每角色一组）"
                                     "⇒ 字面意义上的「24 行三元组互异」**不成立且不应成立**"
                                     "（去重口径 = 按 (config_id, role) 分组，生产者 L466–490）")

    # ⑤ 格取值与设计表逐字一致（含 config_id 反解）
    for cell in CELLS:
        lv = [l for l in plan["levels"] if l["config_id"] == cell["config_id"]]
        assert len(lv) == 1, cell
        lv = lv[0]
        assert float(lv["bias_scale"]) == cell["bias_scale"] and \
               float(lv["bias_log_sigma"]) == cell["bias_log_sigma"], lv
        m = CFG_RE.match(cell["config_id"])
        assert m, cell["config_id"]
        assert abs(float(m.group(1)) - cell["bias_scale"]) < 1e-9, cell
        assert abs(float(m.group(2)) - cell["bias_log_sigma"]) < 1e-9, cell
    kept = {l["config_id"]: [l["bias_scale"], l["bias_log_sigma"]] for l in plan["levels"]}
    facts["levels"] = kept
    assert len(kept) == 4, kept
    for a in arms:
        exp = kept[a["config_id"]]
        assert float(a["bias_scale"]) == exp[0] and float(a["bias_log_sigma"]) == exp[1], a["run_id"]
        assert a["gmax_scale"] == 1.0 and a["indegree"] == 20 and a["delay_bins"] == [40, 60, 80], a["run_id"]
        assert a["subtypes"] is True and a["slow"] is True and a["device"] == "mps", a["run_id"]
        assert a["tau_slow_ms"] == TAU_SLOW_MS and a["slow_inc"] == SLOW_INC, a["run_id"]
        assert a["n_neuron"] == N and a["settle_ms"] == SETTLE_MS and a["delivery"] == DELIVERY, a["run_id"]
        assert a["mechanism"] == MECHANISM and a["stage"] == STAGE, a["run_id"]

    # ⑥ 驱动侧：禁驱动抬升
    assert plan["drive_rate_hz"] == DRIVE_RATE_HZ, plan["drive_rate_hz"]
    for a in arms:
        assert float(a["drive_mult"]) in (0.0, 1.0), ("drive_mult 只允许 0.0/1.0：%s" % a["run_id"])
        assert "drive_rate_hz「 not in a, ("臂不得覆盖 drive_rate_hz（禁驱动抬升）：%s" % a["run_id"])
    for l in plan["levels"]:
        assert "drive_rate_hz" not in l and "drive_mult" not in l, l["config_id"]
    facts["drive_rate_hz"] = plan["drive_rate_hz"]
    facts["drive_mult_values"] = sorted({float(a["drive_mult"]) for a in arms})

    # ⑦ 键集合与 L1 计划逐键对齐（L1 ⊆ L3）
    with open(L1_PLAN) as f:
        l1 = json.load(f)
    missing = sorted(set(l1.keys()) - set(plan.keys()))
    assert not missing, ("L3 缺少 L1 的顶层键（必须逐键对齐）：%r" % missing)
    facts["l1_keys"] = list(l1.keys())
    facts["extra_keys_vs_l1"] = [k for k in plan.keys() if k not in l1]

    # ⑧ 生产者冻结 + 键读取逐字比对
    sha = sha256_of(PRODUCER)
    assert sha == PRODUCER_SHA, ("生产者 sha 不符（R48 冻结）：%s ≠ %s" % (sha, PRODUCER_SHA))
    n_lines = len(open(PRODUCER).read().splitlines())
    assert n_lines == PRODUCER_LINES, (n_lines, PRODUCER_LINES)
    facts["producer_sha256"] = sha
    facts["producer_key_reads"] = producer_key_report()
    facts["decorative_keys"] = list(DECORATIVE_KEYS)

    # ⑨ 预算自洽（逐格加权，不得单点外推）
    b = plan["budget"]
    assert b["gpu_h_mid"] < b["gpu_h_upper"], b
    assert abs(b["gpu_h_mid"] - 1.7937) < 5e-4, b["gpu_h_mid"]
    assert abs(b["gpu_h_upper"] - 2.1036) < 5e-4, b["gpu_h_upper"]
    facts["gpu_h_mid"] = b["gpu_h_mid"]
    facts["gpu_h_upper"] = b["gpu_h_upper"]
    facts["gpu_h_per_cell"] = {k: v["gpu_h_mid"] for k, v in b["per_cell"].items()}
    return facts


def _serialize(plan: dict) -> bytes:
    return json.dumps(plan, indent=1, ensure_ascii=False).encode("utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--check", action="store_true", help="不写盘；与磁盘文件逐字节比对 + 重跑全部断言")
    ap.add_argument("--keys", action="store_true", help="只打印生产者键读取比对表")
    args = ap.parse_args(argv)
    assert os.path.abspath(args.out) not in [os.path.abspath(x) for x in FORBIDDEN_OUT_PATHS], (
        "**拒绝写入**：%s 是既有的、正在被正式跑使用的计划（R119/本节点事故登记）⇒ 换路径" % args.out)

    if args.keys:
        for k, ln, snip in producer_key_report():
            print("  %-18s L%-6s %s" % (k, ln if ln else "-", snip))
        print("装饰性键（生产者在 spec 内不读）: %s" % (list(DECORATIVE_KEYS),))
        return 0

    plan = build_plan()
    facts = run_assertions(plan)
    blob = _serialize(plan)
    if args.check:
        with open(args.out, "rb") as f:
            disk = f.read()
        assert disk == blob, ("磁盘计划与生成器**不一致**（%d B vs %d B）⇒ 计划被手改过或生成器漂移"
                              % (len(disk), len(blob)))
        print("--check 通过：%s 与生成器逐字节一致（%d B）" % (args.out, len(blob)))
    else:
        with open(args.out, "wb") as f:
            f.write(blob)
        print("plan(%d arms) -> %s（%d B）" % (len(plan["arms"]), args.out, len(blob)))
    print("断言全部通过：")
    for k in ("n_arms", "n_baseline", "n_zero_drive", "steps_per_arm", "gpu_h_mid", "gpu_h_upper"):
        print("  %-16s %s" % (k, facts[k]))
    for cfg, v in facts["levels"].items():
        print("  %-22s bias=%s σ=%s  gpu_h_mid=%s" % (cfg, v[0], v[1], facts["gpu_h_per_cell"][cfg]))
    print("  计划级三元组个数 = %d（刻意重复；去重口径=按 (config_id, role)）" % len(facts["distinct_triples_plan_wide"]))
    for k, ln, snip in facts["producer_key_reads"]:
        print("  [键] %-18s L%-6s %s" % (k, ln if ln else "-", snip[:80]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
