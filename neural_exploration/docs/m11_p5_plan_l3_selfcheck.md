# M11 · P5 **L3 计划（重规划节点版）** 自检报告

> **权威依据**：`docs/m11_acceptance_master.md` **§3dc(R112) / §3dd(R113) / §3de(R114) / §3df(R115) / §3dg(R116)**（逐节读完）；
> `data/m11_p5_lever_prereg.json::L2_design`（结构与条款格式模板；L2 本身已被 R110 判**事前失败、不跑**）；
> `data/m11_p5_plan_l1.json`（键集合模板）；`data/m11_p5_prereg_amend_r109_a_reporting.json`（(a) 三条并报）；
> `data/m11_p5_targets.csv`（判据表，L46 `selfsustain_ratio_max=0.10`、L48 `selfsustain_zero_drive_min_spikes=300`、
> L119 `protocol_drive_rate_hz=8.0`）；`docs/m11_b3_g1_design.md` L146（驱动抬升禁令）。
>
> **一句话结论**：计划已生成并通过**全部静态断言**（24 臂、步数 210,000、复刻轴互异、键名与生产者逐字一致），
> 但**有两项必须由主 agent 裁决的事**：**① 本节点发生过一次"覆盖既有计划 JSON"的事故（已逐字节恢复，见 §0）；
> ② 主 agent 派工钉死的 4 格（B1/B2 = σ 0.20/0.15）与 R118 之后的探针实测（σ 是"悬崖"）相冲突（见 §8）**。

---

## §0 🔴 本节点事故登记与恢复（**必读，不得省略**）

| 项 | 内容 |
|---|---|
| 事故 | 本节点**首轮**把计划写到 `data/m11_p5_plan_l3.json`，**覆盖了一个 tracked 的既有文件**（任务书硬约束③要求"只新增三个文件、不改任何既有计划 JSON"）|
| 发现 | `git status --porcelain` 显示 ` M neural_exploration/data/m11_p5_plan_l3.json`（**M = 已跟踪且被改**）+ 目录内出现 `data/m11_p5_l3_runs/`（**正在被正式跑写入**）|
| 影响面 | 既有文件 = `git` HEAD 版本 sha256 **`4aab49b00177a22e21f65b884bd29d95a8e7333e76cb109ae8fcd8bb9bd94615`**（30 臂"网格 v2"）；正式跑实际使用的是**冻结副本** `data/m11_p5_plan_l3.frozen_for_run.json`（同 sha）+ `--out-dir` 绝对路径（见 `docs/m11_l3_resume.md` L11–L15）⇒ **正式跑未受影响** |
| 恢复 | `git show HEAD:neural_exploration/data/m11_p5_plan_l3.json > <该文件>` ⇒ 回读 sha256 = `4aab49b0…`（**与冻结副本一致**），`git status --porcelain <该文件>` **输出为空**（= 与 HEAD 逐字节相同）|
| 防复发 | 本节点新计划改写到**新文件名** `data/m11_p5_plan_l3_replan.json`；生成器内建 `FORBIDDEN_OUT_PATHS`（含上述两个路径）并在 `main()` 里 `assert` 拒绝写入 |
| 纪律对应 | **第五十九条**（"已写入"必须能回读：本报告 §1 给 `wc -l -c` + `shasum -a 256`）；**第五十六条**（静态检查必须同时验路径与外部命令——本次正是"路径"未验）|

**⇒ 给主 agent 的一句话**：请以 `data/m11_p5_plan_l3.frozen_for_run.json`（sha `4aab49b0…`）为**正在跑的**正式计划，
本节点产物是**另一份、不冲突的**预注册计划（新路径），二者格位互补（见 §8）。

---

## §1 三件套指纹与臂数（**回读值**）

| 文件 | 行数 | 字节 | sha256 |
|---|---|---|---|
| `neural_exploration/data/m11_p5_plan_l3_replan.json` | 1638 | 62,781 | `539325a13aa201cd5bc480d0b941768fb452ef2e36caccd3c90fba29ff1af98b` |
| `neural_exploration/tools/make_m11_p5_plan_l3.py` | 923 | 69,055 | `de97ebaf9122ffdf8669ceab43dae69a2109eb01fc9c99af6a87f3d8e1f7cebc` |
| `neural_exploration/docs/m11_p5_plan_l3_selfcheck.md`（本文件） | 见文末 §10「本报告指纹」 | 同左 | 同左 |

- **臂总数 = 24**（= 4 格 × 2 角色 × 3 独立实现）⇒ **pass**（生成器断言 `len(arms) == 24`）。
- 角色分布：`baseline` 12 / `zero_drive` 12；`dose` 0 / `replica` 0。
- 计划顶层键 **48**（= `m11_p5_plan_l1.json` 的 30 键 **全部保留** + 18 个新增条款键，见 §2 第 6 条）。
- 生产者（冻结）：`tools/run_m11_p5_arms.py`，**894 行 / 52,659 B**，
  sha256 = `633a4c458e9c1dd58e5d6c4838a205819a9424fbd49df9da8827c8b7bd42285b`（= 任务书给定值 ✓，= R116 验收后的版本）。

---

## §2 逐条断言（**每条：实测值 + pass/fail**）

### 2.1 臂数 = 24

| 断言 | 实测值 | 判定 |
|---|---|---|
| `len(plan["arms"]) == 24` | 24（baseline 12 + zero_drive 12） | **pass** |
| `len(plan["levels"]) == 4` | 4 | **pass** |

### 2.2 `(config_id, role, realization)` 无重复

| 断言 | 实测值 | 判定 |
|---|---|---|
| `(config_id, role, seed)` 三元组互异 | 24 行 → 24 个**互异**三元组（去重后 24） | **pass** |
| `run_id` 互异 | 24 行 → 24 个互异 `run_id` | **pass** |
| 生成器断言 `len(set(tri)) == len(tri)` | 通过 | **pass** |

### 2.3 每格内 3 个实现的 `(net_seed, event_seed)` 两两互异

| 格（`config_id`） | 角色 | 3 个实现的 `(net_seed, event_seed)` | 判定 |
|---|---|---|---|
| `p5_L3_b1.00_s0.30` | baseline | (0,11) / (1,12) / (2,13) | **pass** |
| `p5_L3_b1.00_s0.30` | zero_drive | (0,11) / (1,12) / (2,13) | **pass** |
| `p5_L3_b1.30_s0.30` | baseline | (0,11) / (1,12) / (2,13) | **pass** |
| `p5_L3_b1.30_s0.30` | zero_drive | (0,11) / (1,12) / (2,13) | **pass** |
| `p5_L3_b1.30_s0.20` | baseline | (0,11) / (1,12) / (2,13) | **pass** |
| `p5_L3_b1.30_s0.20` | zero_drive | (0,11) / (1,12) / (2,13) | **pass** |
| `p5_L3_b1.30_s0.15` | baseline | (0,11) / (1,12) / (2,13) | **pass** |
| `p5_L3_b1.30_s0.15` | zero_drive | (0,11) / (1,12) / (2,13) | **pass** |

**独立第三方复核（最强的证据，不是自证）**：用**生产者自己的**非构造性预检跑本计划：

```
./.venv-m11/bin/python -c "from neural_exploration.tools.run_m11_p5_arms import assert_plan_replication; \
  plan=json.load(open('neural_exploration/data/m11_p5_plan_l3_replan.json')); print(assert_plan_replication(plan))"
⇒ 复刻轴预检通过：8 组 (config,T,dose) 内不同 seed 的三元组互不相同
⇒ PREFLIGHT_OK {'n_groups': 8, 'n_arms': 24}                                    ← **pass**
```
（该预检就是 `--plan` 模式**跑任何臂之前**执行的那一个，见 §3 的 L548。）

### 2.4 全计划 24 行三元组无重复（**列全**）

| # | run_id | config_id | role | seed | net_seed | event_seed | 三元组 |
|---|---|---|---|---|---|---|---|
| 1 | `l3_a0_8192_T10000ms_d1_s0` | `p5_L3_b1.00_s0.30` | baseline | 0 | 0 | 11 | `net0\|ev11\|reset0` |
| 2 | `l3_a0_8192_T10000ms_d1_s1` | `p5_L3_b1.00_s0.30` | baseline | 1 | 1 | 12 | `net1\|ev12\|reset1` |
| 3 | `l3_a0_8192_T10000ms_d1_s2` | `p5_L3_b1.00_s0.30` | baseline | 2 | 2 | 13 | `net2\|ev13\|reset2` |
| 4 | `l3_a0_8192_T10000ms_d0_s0` | `p5_L3_b1.00_s0.30` | zero_drive | 0 | 0 | 11 | `net0\|ev11\|reset0` |
| 5 | `l3_a0_8192_T10000ms_d0_s1` | `p5_L3_b1.00_s0.30` | zero_drive | 1 | 1 | 12 | `net1\|ev12\|reset1` |
| 6 | `l3_a0_8192_T10000ms_d0_s2` | `p5_L3_b1.00_s0.30` | zero_drive | 2 | 2 | 13 | `net2\|ev13\|reset2` |
| 7 | `l3_a1_8192_T10000ms_d1_s0` | `p5_L3_b1.30_s0.30` | baseline | 0 | 0 | 11 | `net0\|ev11\|reset0` |
| 8 | `l3_a1_8192_T10000ms_d1_s1` | `p5_L3_b1.30_s0.30` | baseline | 1 | 1 | 12 | `net1\|ev12\|reset1` |
| 9 | `l3_a1_8192_T10000ms_d1_s2` | `p5_L3_b1.30_s0.30` | baseline | 2 | 2 | 13 | `net2\|ev13\|reset2` |
| 10 | `l3_a1_8192_T10000ms_d0_s0` | `p5_L3_b1.30_s0.30` | zero_drive | 0 | 0 | 11 | `net0\|ev11\|reset0` |
| 11 | `l3_a1_8192_T10000ms_d0_s1` | `p5_L3_b1.30_s0.30` | zero_drive | 1 | 1 | 12 | `net1\|ev12\|reset1` |
| 12 | `l3_a1_8192_T10000ms_d0_s2` | `p5_L3_b1.30_s0.30` | zero_drive | 2 | 2 | 13 | `net2\|ev13\|reset2` |
| 13 | `l3_b1_8192_T10000ms_d1_s0` | `p5_L3_b1.30_s0.20` | baseline | 0 | 0 | 11 | `net0\|ev11\|reset0` |
| 14 | `l3_b1_8192_T10000ms_d1_s1` | `p5_L3_b1.30_s0.20` | baseline | 1 | 1 | 12 | `net1\|ev12\|reset1` |
| 15 | `l3_b1_8192_T10000ms_d1_s2` | `p5_L3_b1.30_s0.20` | baseline | 2 | 2 | 13 | `net2\|ev13\|reset2` |
| 16 | `l3_b1_8192_T10000ms_d0_s0` | `p5_L3_b1.30_s0.20` | zero_drive | 0 | 0 | 11 | `net0\|ev11\|reset0` |
| 17 | `l3_b1_8192_T10000ms_d0_s1` | `p5_L3_b1.30_s0.20` | zero_drive | 1 | 1 | 12 | `net1\|ev12\|reset1` |
| 18 | `l3_b1_8192_T10000ms_d0_s2` | `p5_L3_b1.30_s0.20` | zero_drive | 2 | 2 | 13 | `net2\|ev13\|reset2` |
| 19 | `l3_b2_8192_T10000ms_d1_s0` | `p5_L3_b1.30_s0.15` | baseline | 0 | 0 | 11 | `net0\|ev11\|reset0` |
| 20 | `l3_b2_8192_T10000ms_d1_s1` | `p5_L3_b1.30_s0.15` | baseline | 1 | 1 | 12 | `net1\|ev12\|reset1` |
| 21 | `l3_b2_8192_T10000ms_d1_s2` | `p5_L3_b1.30_s0.15` | baseline | 2 | 2 | 13 | `net2\|ev13\|reset2` |
| 22 | `l3_b2_8192_T10000ms_d0_s0` | `p5_L3_b1.30_s0.15` | zero_drive | 0 | 0 | 11 | `net0\|ev11\|reset0` |
| 23 | `l3_b2_8192_T10000ms_d0_s1` | `p5_L3_b1.30_s0.15` | zero_drive | 1 | 1 | 12 | `net1\|ev12\|reset1` |
| 24 | `l3_b2_8192_T10000ms_d0_s2` | `p5_L3_b1.30_s0.15` | zero_drive | 2 | 2 | 13 | `net2\|ev13\|reset2` |

| 断言 | 实测值 | 判定 |
|---|---|---|
| **字面**：全计划 24 行三元组互异 | **不成立**：只有 **3 个互异三元组**（`net0\|ev11\|reset0` ×8、`net1\|ev12\|reset1` ×8、`net2\|ev13\|reset2` ×8） | **fail（按字面）** |
| **判据本意**：同一 `(config_id, role)` 组内三元组互异 | 8 组，每组 3 个**互异** | **pass** |

**⇒ 该"字面 fail"是设计使然、且不应消除（登记为 by-design，不是缺陷）**，理由三条：
1. **生产者的判据本身就是分组去重**：`assert_plan_replication`（`run_m11_p5_arms.py` L466–467）按
   `(config_id, T_ms, drive_mult, mechanism)` 分组，主判据是组内 `(net_seed,event_seed)` **对数 == 不同 seed 数**
   （L483–490；注释 L470–473 明说"只比三元组会构造性恒真"）。**全计划级互异从未是判据**。
2. **A0 锚要求跨格复用 seed**：A0 的 6 臂必须与既有 L0 臂 `l0_g1.00_b1.00_…_s{0,1,2}` **逐位相同**
   （`cross_plan_anchor`）；若给每格不同的 `net_seed`（例如 `100*格号+seed`）来满足"24 行互异"，
   **锚点立即失效**（这正是 R95 §⑤ 要求的确定性核对的前提）。
3. **R93 预注册只定义了 3 组 `(net,event)` 对**（= L0/L1 的 `SEED_PAIRS = {0:(0,11), 1:(1,12), 2:(2,13)}`）；
   新增第 4–24 组对属于**改预注册约定**，不得由本节点擅自做。

### 2.5 `T_ms == 10000.0`、`settle_ms == 500.0`、步数 = 210,000（**写算式**）

**算式（R80 家族，生成器内建断言）**：

```
steps = round((settle_ms + T_ms) / dt_ms) = round((500.0 + 10000.0) / 0.05) = round(210000.0) = 210000
另：210000 ≥ MIN_STEPS(200000) ✓      T_ms ∈ {10000.0} ✓
```
逐臂复核：24 臂**每一臂**都满足 `round((arm.settle_ms + arm.T_ms)/0.05) == 210000`（生成器逐臂断言，全部通过）。

| 断言 | 实测值 | 判定 |
|---|---|---|
| 计划级 `dt_ms` | 0.05（= 生产者 `DT_MS`，L62） | **pass** |
| 计划级 `settle_ms` | 500.0 | **pass** |
| 臂级 `T_ms` 全部 = 10000.0 | 24/24 | **pass** |
| 臂级步数全部 = 210,000 | 24/24 | **pass** |
| 步数 ≥ 200,000 | 210,000 ≥ 200,000 | **pass** |

### 2.6 键集合与 `m11_p5_plan_l1.json` 逐键对齐（差集应为空）

| 断言 | 实测值 | 判定 |
|---|---|---|
| `set(L1 顶层键) − set(L3 顶层键)` | **`[]`（空）** —— 30 键**全部保留** | **pass** |
| `set(L3) − set(L1)`（新增条款键） | 18 个（逐字列出，见下） | 登记（**非**失败） |
| 臂级必需键（任务书清单 24 键）逐臂齐全 | 24/24 臂**恰好**等于 24 键（无缺、无多） | **pass** |

- **L1 的 30 键**：`plan_id, structure, parent_prereg, units, n_neuron, settle_ms, dt_ms, drive_rate_hz, device,
  bin_ms, smooth_frames, estimator, indegree, gmax_scale, bias_scale, bias_log_sigma, delay_bins, seed_policy,
  subtypes, slow, production_delivery, criterion_delivery, launch_prefix, judged_units, prerequisites,
  lever_prereg_falsifier, frozen_tool, cross_plan_anchor, levels, arms`。
- **L3 新增的 18 键**：`tau_slow_ms, slow_inc, core_falsifiable_assertion, window_definition, a_reporting_four,
  subtype_reporting, homogenization_defense, unmeasured_side_declaration, producer_dependency, stop_rules, budget,
  b_reporting_form, forbidden_levers, prior_probe_evidence, reporting_contract, cross_tool_notes, execution_order,
  selfcheck_map`（其中 `tau_slow_ms/slow_inc` 是生产者**会**读的计划级兜底键，见 §2.8；其余是条款/登记块）。
- **臂级键差异（L3 ⊃ L1）**：L3 每臂比 L1 每臂多 11 键 —— `net_seed, event_seed, n_neuron, settle_ms, device,
  indegree, delay_bins, subtypes, slow, tau_slow_ms, slow_inc`（= 任务书"每臂必须显式带"清单的落点）。

### 2.7 4 个格的 `bias_scale`/`bias_log_sigma` 与设计表**逐字一致**（对照表）

| 格（任务书） | `bias_scale` | `bias_log_sigma` | 计划 `levels` 实测 | 由 `config_id` 反解 | 判定 |
|---|---|---|---|---|---|
| A0（锚） | 1.00 | 0.30 | 1.0 / 0.3 | `p5_L3_b1.00_s0.30` → 1.00 / 0.30 | **pass** |
| A1 | 1.30 | 0.30 | 1.3 / 0.3 | `p5_L3_b1.30_s0.30` → 1.30 / 0.30 | **pass** |
| B1（角落） | 1.30 | 0.20 | 1.3 / 0.2 | `p5_L3_b1.30_s0.20` → 1.30 / 0.20 | **pass** |
| B2 | 1.30 | 0.15 | 1.3 / 0.15 | `p5_L3_b1.30_s0.15` → 1.30 / 0.15 | **pass** |

逐臂复核：24/24 臂的 `bias_scale`/`bias_log_sigma` 与该格 `levels` 值一致；且 `gmax_scale=1.0`、`indegree=20`、
`delay_bins=[40,60,80]`、`subtypes=true`、`slow=true`、`device="mps"`、`tau_slow_ms=1000.0`、`slow_inc=6.0`、
`n_neuron=8192`、`settle_ms=500.0`、`delivery="chunk"`、`mechanism="lever_l3"`、`stage="l3_grid"` 全部逐臂一致（**pass**）。

### 2.8 `drive_rate_hz == 8.0` 且**无任何档以驱动抬升为标定**（禁止条款核对）

| 断言 | 实测值 | 判定 |
|---|---|---|
| 计划级 `drive_rate_hz` | **8.0** | **pass** |
| 24 臂的 `drive_mult` 取值集合 | **{0.0, 1.0}**（0.0 = zero_drive，1.0 = baseline） | **pass** |
| 24 臂**均不**带 `drive_rate_hz` 覆盖键 | 0/24 带 | **pass** |
| `levels` 内**均无** `drive_rate_hz` / `drive_mult` 键 | 4/4 干净 | **pass** |
| 驱动抬升档（`drive_mult>1` 或 `drive_rate_hz>8.0`） | **不存在**（生成器断言 + 逐档核对） | **pass** |
| 禁令出处 | `docs/m11_b3_g1_design.md` **L146**"**禁**：提高背景驱动 λ …… 只能作为**剂量-响应诊断**，**不得**作为标定手段"；`data/m11_p5_targets.csv` **L119** `protocol_drive_rate_hz=8.0`（R115 ⑤ 的 append-only 补登行，**本节点实读确认已存在**） | 登记 ✓ |

### 2.9 生产者键读取处 ⇔ 计划键名**逐字比对**（**给行号 + 代码片段**）

生产者 = `neural_exploration/tools/run_m11_p5_arms.py`（sha `633a4c45…`，894 行；行号即该冻结版本的行号）。
下表由生成器 `producer_key_report()` 实读源码生成（`python3 -m …make_m11_p5_plan_l3 --keys` 可重放）：

| 计划键 | 生产者行号 | 代码片段（实读） |
|---|---|---|
| `run_id` | 302 | `run_id = str(spec["run_id"])` |
| `config_id` | 314 | `arm_id=run_id, config_id=str(spec.get("config_id", "p5")),` |
| `T_ms` | 237 | `T_ms = float(spec["T_ms"])` |
| `drive_mult` | 242 | `dose = float(spec.get("drive_mult", 1.0))` |
| `seed` | 159 | `seed = int(spec.get("seed", 0))`（`effective_seeds`） |
| `net_seed` | 134 | `net = build_laminar_probe_column(n, seed=int(spec.get("net_seed", 0)), delay_bins=bins, **net_kw)` |
| `event_seed` | 161 | `es = spec.get("event_seed", None)`（→ L163 `event_seed = (EVENT_SEED_BASE + seed) if es is None else int(es)`）|
| `role` | 286 | `_zero_drive_expected = (str(spec.get("role", "dose")) == "zero_drive") or (dose == 0.0)` |
| `mechanism` | 316 | `mechanism=str(spec.get("mechanism", "none")), drive_mult=dose,` |
| `delivery` | 257 | `delivery = str(spec.get("delivery", "step"))` |
| `gmax_scale` | 124 | `if k in ("indegree", "gmax_scale", "bias_scale", "bias_log_mean", "bias_log_sigma")`（→ `build_laminar_probe_column(**net_kw)`）|
| `bias_scale` | 124 | 同上 |
| `bias_log_sigma` | 124 | 同上（另：L178–179 `if "bias_log_sigma" in spec: d["bias_log_sigma"] = float(...)` 回显进产物）|
| `indegree` | 124 | 同上 |
| `delay_bins` | 125 | `bins = tuple(spec.get("delay_bins") or NET_DEFAULTS["delay_bins"])` |
| `subtypes` | 127 | `eng.configure_subtypes(bool(spec.get("subtypes", True)))` |
| `slow` | 128 | `if spec.get("slow", True):` |
| `tau_slow_ms` | 129 | `eng.configure_slow(True, tau_slow_ms=float(spec.get("tau_slow_ms", 1000.0)),` |
| `slow_inc` | 130 | `inc_per_spike=float(spec.get("slow_inc", 6.0)))` |
| `n_neuron` | 236 | `n = int(spec["n_neuron"])` |
| `settle_ms` | 238 | `settle_ms = float(spec["settle_ms"])` |
| `device` | 246 | `device = str(spec.get("device", "mps"))` |
| `stage` | **—（生产者不读）** | **装饰性键**：只用于 `run_id` 生成与人可读；**不得**期待它有任何物理效果 |
| `lever` | **—（生产者不读）** | **装饰性键**：同上（人可读档位说明） |

**⇒ 24 个必需键中 22 个有实读点（逐字一致），2 个（`stage`/`lever`）显式登记为装饰性（pass + 登记）**。
另核对**计划级兜底键**：`run_plan_main` L562–566 的兜底白名单 =
`indegree, gmax_scale, bias_scale, bias_log_sigma, bias_log_mean, net_seed, delay_bins, subtypes, slow, tau_slow_ms,
slow_inc` + `n_neuron` ⇒ 本计划这 12 个计划级键**全部命中该白名单**（`bias_log_mean` 未写 ⇒ 沿用引擎默认 **230.0**；
本节点实读 `src/mouse_cortex_circuit.py` **L44** `bias_log_mean: float = 230.0`、**L118**
`bias = np.exp(rng.normal(np.log(...mean...), sigma, n_total)) * bias_scale`）。

### 2.10 **接口级**验证（不是只比字符串）：用本计划的一臂**真跑**过生产者（CPU 极小冒烟）

把计划里 A0 的 baseline 臂**原样**取下来，仅改 `n_neuron=1000`、`T_ms=1000.0`、`device="cpu"`（**未用 MPS**），
其余键（含 `bias_scale/bias_log_sigma/slow/slow_inc/tau_slow_ms/indegree/delay_bins/subtypes/delivery/mechanism`）
**逐字保留**，交给生产者执行：

```
./.venv-m9/bin/python -m neural_exploration.tools.run_m11_p5_arms \
    --arm /tmp/m11_l3_smoke/arm.json --out-dir /tmp/m11_l3_smoke/out
⇒ [DONE] l3_a0_smoke1000_T1000ms_d1_s0  spk=355  ei_ratio=1.8497  ms/step=0.1645  wall=4.54s  RSS=199MB
⇒ EXIT=0；产物 JSON 含 net={bias_scale:1.0, bias_log_sigma:0.3, gmax_scale:1.0, indegree:20, delay_bins:[40,60,80], net_seed:0}
```
**⇒ 判定 pass**，且与 R116 §④ 的节点冒烟臂 **B1（`baseline, σ=0.30`）逐位同量级**（该节点记 `spk=355`、`ei_ratio=1.84973`）
⇒ 本计划的键名谱**确实**被生产者按预期消费（不是"键写了但没生效"）。产物**只写进 `/tmp`，未污染仓库**。

---

## §3 驱动器失败策略：**继续**还是**中止**（**读码确认，给行号 + 片段**）

命令：`read tools/run_m11_p5_arms.py`（行号属 sha `633a4c45…` 的 894 行版本）。

```python
549     arms, n_fail = [], 0
550     for spec in plan["arms"]:
551         rid = str(spec["run_id"])
552         done = os.path.join(out_dir, f"{rid}.done")
553         if os.path.exists(done) and not args.force:        # ← 续跑：已完成的臂直接跳过
554             with open(os.path.join(out_dir, f"{rid}.json")) as f:
555                 arms.append(json.load(f))
556             print("[SKIP] %s（已有 .done）" % rid, flush=True)
557             continue
...
576         if p.returncode != 0:
577             n_fail += 1
578             sys.stderr.write(p.stderr[-800:])
579             print("[FAIL] %s（%.1fs）" % (rid, time.time() - t0), flush=True)
580             if n_fail >= 2:
581                 print("**停点**：连续 ≥2 臂失败 ⇒ 停并汇报（不自行重跑）", flush=True)
582                 break
583             continue
...
588     print("完成：%d 臂（失败 %d）｜out_dir=%s" % (len(arms), n_fail, out_dir), flush=True)
589     return 0 if not n_fail else 1
```

**结论（逐字）**：
- **第 1 臂失败 ⇒ 继续**（L576–579 记失败 + L583 `continue` 到下一臂，**不中止**）。
- **累计第 2 臂失败 ⇒ 中止整个 plan**（L580–582 `break`）。
  ⚠️ **代码与提示语有细微不符（登记）**：提示语写"**连续** ≥2 臂失败"，但 `n_fail`（L549 初始化）**只在失败时自增、从不重置**
  ⇒ 实际语义是"**累计** ≥2 臂失败即停"。**⇒ 主 agent 若期望"连续"语义，需知这一点**（一处 1 行的小缺陷，本节点**未改**代码，
  遵守"不得改生产者"）。
- 单臂内部失败原因：`run_arm_main` L425–430 捕获异常、写 `<run_id>.fail`、`return 1`；`--plan` 侧据此累加 `n_fail`。

**排序如何保住已跑档 + 续跑是否 resume-safe**：
- **保住已跑档**：① 失败**不删除**任何已落盘产物（L420–422 只删本臂自己的 `.done/.fail`）；② 一旦 `break`，
  已完成臂的 `.json/.done/_counts.csv/_poptrace.npy` **全部留在盘上**；③ 本计划把**最便宜且最有信息量**的 A0 锚放第 1 位
  （`execution_order.rationale` ②），故"某格把网络压死/炸掉"时损失最小；④ 每格内 **baseline 先跑**（它是 (a) 与窗口上界
  `0.10×pooled_baseline` 的分母），baseline 若硬失败则该格另 3 臂不必再跑。
- **resume-safe**：**是**。L553–557 对已有 `.done` 的臂打印 `[SKIP]` 并**复用其 JSON**（`--force` 才会重跑）
  ⇒ **直接重跑同一命令即可续跑**；本计划明确**不得**用 `--force`（`execution_order.resume_safety`）。
  另：`assert_plan_replication(plan)` 在 **L548 = 循环之前**执行 ⇒ 复刻轴失效时**整轮零臂启动**（不是跑一半才发现），
  且该检查**已在本节点对 L3 计划实跑通过**（§2.3）。

---

## §4 排序表 + 理由

| 序 | 阶段 | `config_id` | 角色顺序 | 臂（6/格） | 预算 mid（GPU-h） |
|---|---|---|---|---|---|
| 1 | `l3_grid` | `p5_L3_b1.00_s0.30`（A0，**锚**） | baseline s0,s1,s2 → zero_drive s0,s1,s2 | 6 | 0.4155 |
| 2 | `l3_grid` | `p5_L3_b1.30_s0.30`（A1） | 同上 | 6 | 0.5146 |
| 3 | `l3_grid` | `p5_L3_b1.30_s0.20`（B1，**主目标**） | 同上 | 6 | 0.4444 |
| 4 | `l3_grid` | `p5_L3_b1.30_s0.15`（B2） | 同上 | 6 | 0.4192 |

**理由（写进计划 `execution_order.rationale`，逐条）**：
1. **A0 锚最先**：与既有 L0 臂同规格 ⇒ 先做**确定性核对**（A0 6 臂的 `sha256_of_trace` 应与既有
   `l0_g1.00_b1.00_8192_T10000ms_d{0,1}_s{0,1,2}` **逐位相同**），在花钱最多的网格之前先证明"键名/默认值/口径没被改过"。
2. **保住已跑档**：生产者失败策略是"失败 1 臂继续、**累计** ≥2 臂停"（§3）⇒ 把最便宜、最可判的档放最前，损失最小。
3. **每格先 baseline**：baseline 是 (a) 的分母与窗口上界的来源；baseline 硬失败 ⇒ 该格另 3 臂不必再跑。
4. **A1 → B1 → B2**：A1 提供"σ 不动、只抬 bias"的对照；缺它就无法把 B1 的变化拆成 `bias` 项与 `σ` 项。

---

## §5 任务书条款 1–8 的落点

| 条款 | 落在计划 JSON 的哪个键 | 本报告哪一节 |
|---|---|---|
| ① `unmeasured_side_declaration`（纪律第五十八条） | `unmeasured_side_declaration`（9 条未测侧 + 范围句） | §2.6 / 本节下方 |
| ② (a) **四条**并报（R109 修订 + R115 ②） | `a_reporting_four`（`a_median_hz` 主判 / `a_active_frac` / `a_pop_mean_hz` / **`pooled_zero_drive` 绝对值**）+ `reporting_contract` | 本节 |
| ③ 亚型分解 E/PV/SST/VIP | `subtype_reporting`（来源 = `<run_id>_counts.csv` 的 `cell_type` 列） | 本节 |
| ④ 同质化防线 | `homogenization_defense`（`frac_nonzero_slow_var` / `frac_nonzero_spikes` / `count_dispersion` + "引擎无同步指标"登记） | §6 |
| ⑤ `producer_dependency` | `producer_dependency`（sha/行数/键位契约/行号引用）+ `prerequisites.producer_frozen` | §2.9 / §3 |
| ⑥ 按档取价的预算 | `budget`（`unit_prices` / `proxies` / `per_cell` / `comparison_with_L2` / `caveat_T10` / `wall_anchor_from_existing` / `wall_clock` / `downgrade_plan_if_over`） | **§7** |
| ⑦ 停机规则（抄 L2 8 条并按 L3 改写） | `stop_rules`（8 条，每条带【谁在何时执行】；硬门① 的 300 脉冲下界写进 ②） | §3 / §7 |
| ⑧ `cross_plan_anchor`（含 R95 §⑤ `h_determinism_not_evidence`） | `cross_plan_anchor`（`l3_arms` / `expected_l0_spikes` / `expected_l0_eval` / `sigma_key_inertness_precedent` / `h_determinism_not_evidence`） | 本节 |

**条款要点摘录（计划内逐字存在）**
- **窗口不是极限**：`core_falsifiable_assertion.window` = `pooled_zero_drive ∈ [300, 0.10 × pooled_baseline]`；
  下界 = 硬门①（`targets.csv` **L48**），上界 = `selfsustain_ratio_max=0.10`（**L46**）；
  **R 太小（零驱动被打干净）⇒ (c) `unavailable` 而不是 `pass`** ⇒ **每档必须报 `pooled_zero_drive` 绝对值**。
- **(a) 频带逐字不变**：`[0.5, 2.0] Hz`，`estimator=per_neuron`；**不得**用 `a_active_frac` 代替中位率。
- **亚型分解**：E/PV/SST/VIP 各自**中位率**与**活跃占比**；文献带本身是类级（`targets.csv` L30–L33）⇒ 描述性，不改判据。
- **同质化防线**：`frac_nonzero_slow_var`（引擎读回，`src/mouse_cortex_engine.py` **L1073** 落盘
  `slow_trace_stats.readback.frac_nonzero`）+ `frac_nonzero_spikes`（≡ `a_active_frac`，**同源须声明**）
  + `count_dispersion`（counts 复算的 median/p90/IQR/CV）；**并显式登记"引擎无可用同步性/规则性指标"**。
- **`cross_plan_anchor` 的既有实测值（本节点实读，非估计）**：

| A0 臂（L3） | 对应既有 L0 臂 | 既有 `spikes_total` | 既有 `sha256_of_trace`（前 16 位） |
|---|---|---|---|
| `l3_a0_8192_T10000ms_d0_s0` | `l0_g1.00_b1.00_8192_T10000ms_d0_s0` | 8,591 | `c3b0d166b46db310` |
| `l3_a0_8192_T10000ms_d0_s1` | `l0_g1.00_b1.00_8192_T10000ms_d0_s1` | 8,585 | `ec7f18493ac8b71b` |
| `l3_a0_8192_T10000ms_d0_s2` | `l0_g1.00_b1.00_8192_T10000ms_d0_s2` | 9,802 | `73998b4938c5feb` |
| `l3_a0_8192_T10000ms_d1_s0` | `l0_g1.00_b1.00_8192_T10000ms_d1_s0` | 30,662 | `0096eefa8d19f67f` |
| `l3_a0_8192_T10000ms_d1_s1` | `l0_g1.00_b1.00_8192_T10000ms_d1_s1` | 30,574 | `1fbbee3c5727d76f` |
| `l3_a0_8192_T10000ms_d1_s2` | `l0_g1.00_b1.00_8192_T10000ms_d1_s2` | 29,910 | `d03518c45993113d` |

  - 既有 (c) 实测（`data/m11_p5_resting_eval_repl.json`，本节点实读）：`pooled_baseline = 91,146`、
    `pooled_zero_drive = 26,978`、`R = 0.295987`（文档 R115 记 0.296 ⇒ 逐位一致）⇒ A0 的窗口上界 =
    `0.10 × 91,146 = 9,114.6` ⇒ **26,978 > 9,114.6 ⇒ A0 的 (c) 预期 fail**（**预期**，不是结论）。
  - **σ 键惰性先例（让锚点断言有依据）**：L0 的 `b0.70` 臂（**不带** σ 键）与 L1 的 `b0.70 σ=0.30` 臂（**显式 0.30**）
    **6/6 臂 `sha256_of_trace` 逐位相同**（本节点实读）⇒ "显式 σ=0.30 ≡ 不带键"成立 ⇒ A0 应与 L0 逐位相同。
  - **R95 §⑤**：以上只证明**确定性**，`h_determinism_not_evidence` ⇒ **不作新证据**、不计入 `n_seeds`、不用于判 (a)/(c)。

**未测侧逐条（`unmeasured_side_declaration.still_unmeasured`，9 条）**：
`bias_scale > 1.40`；`bias_scale < 1.00`（与 σ↓ 的组合）；`σ < 0.15`；`σ > 0.30`（与 bias↑ 的组合）；
`drive_rate_hz` 的抬升（预注册禁止为标定手段，剂量-响应诊断未做）；慢适应/STP 与 (bias,σ) 的组合（L2 档未跑）；
外部驱动侧（事件表构造/泄漏/`event_seed` 交互）；亚型选择性杠杆与 `indegree`/`gmax_scale`/`delay_bins`/层流比例；
T 维度（只有 T=10 s ⇒ 硬门④ 阶梯 `unavailable`）。
⇒ **范围句**：以上**未测 ≠ 已排除**；L3 的任何结论**只**覆盖 `bias ∈ [1.00,1.30] × σ ∈ [0.15,0.30]` 这 4 格、
`drive_mult ∈ {0,1}`、T=10 s、slow 默认档。

---

## §6 未确认项清单（**逐条写明卡在哪一行/哪个文件**）

| # | 未确认项 | 卡在哪（具体位置） | 本计划如何处置 |
|---|---|---|---|
| 1 | **A0 的 `zero_drive` T=1 s 单价**未实测 | `ls data/m11_p5_l3_feasibility/` 内**没有** (bias=1.00, σ=0.30, `zero_drive`, T=1 s) 的产物（现有 10 个产物见 §7 表） | 用**同格 baseline 实测 1.1872** 作代理（`budget.proxies` 登记）；另有 T=10 s 墙钟锚 0.7704 |
| 2 | **A1 的 `baseline`（bias=1.30, σ=0.30）单价**未实测 | 同上：该 (bias=1.30, σ=0.30, `baseline`) 组合无产物（现有的是 σ=0.30 的 **zero_drive**） | 用 bias=1.30 家族**实测最大值 1.5026**（σ=0.20 baseline）作代理 |
| 3 | **T=10 s 的稳态单价**（除 A0 格）未实测 | A1/B1/B2 三格在 `data/m11_p5_l3_runs/`（正在跑的 v2 网格）与 `data/m11_p5_l3_feasibility/` 里都**还没有 T=10 s 产物**（v2 跑到的是 C2 格） | `budget.caveat_T10` 明写"**须实跑复探**"；并给出 A0 格的 T=10 s 墙钟锚作对照（§7） |
| 4 | **跨神经元同步性/发放规则性指标**不可得 | `src/mouse_cortex_engine.py` 内**无**任何同步/规则性读出（grep 只命中 `ei_sync_every`，那是**每 1000 步把 E/I 张量同步回 Python 的间隔**，与神经元同步无关）；产物只导出**逐神经元计数**，不导出脉冲时刻 ⇒ `CV_ISI`/`Fano` **无法复算** | `homogenization_defense.engine_synchrony_metric` 明写"**引擎无可用同步指标**"+ `unconfirmed` 登记；**不自造** |
| 5 | **硬门⑤（判别性登记）对 L3 单元不可用** | `tools/validate_p11_resting.py` **L660–670**（要求同 `(config_id,T_ms)` 有 `mechanism=='none'` 且 `drive_mult==1.0` 的臂）+ **L1334–1360**（缺失 ⇒ 只降级 G1 至 `unavailable`）；而本计划 `mechanism="lever_l3"`（任务书指定），且**不得改评估器** | `cross_tool_notes.evaluator_consequence_of_mechanism` **只登记**：L3 单元的 (a)/(b) 行 `discriminative=None`，G1 至多 `unavailable`（**不是结果变化**）；补足需主 agent 决定是否另加 `mechanism='none'` 对照臂（会改动 24 臂结构 ⇒ 本节点**不擅自加**） |
| 6 | **引擎 `ms_per_step` 的计时口径**与"含 settle 的墙钟单价"不一致 | `src/mouse_cortex_engine.py` **L666** `t0 = time.perf_counter()` … **L802** `wall = perf_counter() - t0`、**L819** `"ms_per_step": wall / max(n_steps,1) * 1e3` ⇒ 只覆盖**测量窗的步循环**（不含 settle、不含事件表构造）；实测例：A0 zero_drive s0 的引擎值 0.6487 vs 含 settle 墙钟值 `161.79 s/210,000 步 = 0.7704` | `budget` 明写"单位 = ms/step（引擎口径）"，并把**两个口径都给出**（§7 表）；**未把差异藏起来** |
| 7 | **文档 R115 的"≈9,076 脉冲/实现"不可复现** | `docs/m11_acceptance_master.md` **L3156/L3161/L3238** 写 ≈9,076；产物实读为 **8,591 / 8,585 / 9,802**（`data/m11_p5_l0_runs/l0_g1.00_b1.00_8192_T10000ms_d0_s{0,1,2}.json`）⇒ 3 实现均值 **8,992.7**，与 9,076 差 **0.9%**（对 seed 0 的单值差 5.6%） | `cross_tool_notes.doc_approximation_discrepancy` 如实登记；**以产物为准**（pooled 26,978 与 R=0.295987 均可复现） |
| 8 | **σ∈(0.10,0.15) 是否存在可判窗口点**：未测 | R118 之后的探针只到 σ=0.10/0.15/0.20 三档（§8 表）；σ=0.12/0.13 等**无任何产物** | 登记为未测（不属本计划 4 格）；**不得**由 0.15 与 0.10 的两点插值当结论 |
| 9 | `data/m11_p5_targets.csv` 的 `protocol_drive_rate_hz` 行 | **已确认存在**：**L119**（append-only 补登，`final` 列见该行）⇒ R115 ⑤ 的缺口**已关闭**（本节点实读） | 不属于未确认项，登记以正视听 |

---

## §7 预算：**按档取价、逐格加权**（不得用最低档单价外推全网格）

**单价来源（全部为本节点实读的 T=1 s 短跑实测 `ms_per_step_steady`，单位 ms/step）**

| 格（计划内） | 角色 | 单价 | 来源（实测 / 代理） |
|---|---|---|---|
| A0 `p5_L3_b1.00_s0.30` | baseline | **1.1872** | 实测 `data/m11_p5_l3_feasibility/l3pb_b1.00s0.30_8192_T1000ms_d1_s0.json` |
| A0 | zero_drive | 1.1872 | **代理**（同格 baseline；本格 zero_drive 无 T=1 s 实测） |
| A1 `p5_L3_b1.30_s0.30` | zero_drive | **1.4379** | 实测 `…/l3probe_b1.30s0.30_8192_T1000ms_d0_s0.json` |
| A1 | baseline | 1.5026 | **代理**（bias=1.30 家族实测最大值 `…/l3win_b130_s020_bl_…json`） |
| B1 `p5_L3_b1.30_s0.20` | zero_drive | **1.0371** | 实测 `…/l3win_b130_s020_zd_8192_T1000ms_zero_drive_s0.json` |
| B1 | baseline | **1.5026** | 实测 `…/l3win_b130_s020_bl_8192_T1000ms_baseline_s0.json` |
| B2 `p5_L3_b1.30_s0.15` | zero_drive | **0.9693** | 实测 `…/l3win_b130_s015_zd_8192_T1000ms_zero_drive_s0.json` |
| B2 | baseline | **1.4259** | 实测 `…/l3win_b130_s015_bl_8192_T1000ms_baseline_s0.json` |

（**8 个格-角色单价中 6 个为实测**；2 个未测者已按上文显式代理。参考：`bias=0.70` 的 0.6415 与
`bias=1.00, σ=0.10` 的 1.0760、`bias=1.30, σ=0.10` 的 1.4856 亦均为实测，但**不属**本计划的格。
**⚠️ 我**没有**用最低档 0.6415 外推**。）

**逐格加权（`gpu_h = Σ_臂 210,000 步 × 该(格,角色)单价 / 3.6e6 ms`）**

| 格 | 6 臂的单价和（ms/step） | 步数 | GPU-h（mid） |
|---|---|---|---|
| A0 | 3 × (1.1872 + 1.1872) = 7.1232 | 1,260,000 | **0.4155** |
| A1 | 3 × (1.5026 + 1.4379) = 8.8215 | 1,260,000 | **0.5146** |
| B1 | 3 × (1.5026 + 1.0371) = 7.6191 | 1,260,000 | **0.4444** |
| B2 | 3 × (1.4259 + 0.9693) = 7.1856 | 1,260,000 | **0.4192** |
| **合计** | **30.7494** | **5,040,000** | **1.7937** |

- **`gpu_h_mid = 1.7937`**（逐格加权；折合 1.2812 ms/step/臂）
- **`gpu_h_upper = 2.1036`**（上界规则：全 24 臂一律按**全网格实测最大值 1.5026 ms/step**）
  = `24 × 210,000 × 1.5026 / 3.6e6`
- **与 `L2_design.budget` 的对照**：L2 的 `L2_slow_scan`（30 run / 6,300,000 步）mid **1.394** / upper **1.701**
  ⇒ 折合 **0.7966 / 0.9720 ms/step**；**L3 的 1.2812 ms/step = L2 的 ×1.61**（因为 L3 全部落在 bias 1.00–1.30，
  L2 基线是 bias=0.70 ⇒ **两者不可互相外推**）。L2 的**全计划** total mid 6.629 / upper 7.982 —— L3（1.79/2.10）
  在 M11 的 ≤10 GPU-h 包络内。
- **墙钟**：纯计算 1.7937 GPU-h ÷ 实测占空比 **0.054**（R112 ⑥）⇒ **≈33.2 h**，再加逐 run 装配/结算开销
  （见下）≈0.2–0.4 h ⇒ **必须托管**；占空比是唯一大杠杆（关掉 GUI 浏览器标签可 ≈5–10×）。
- **⚠️ "T=10 s 稳态单价可能高于 1 s 短跑"这一前提：在 A0 格被**既有 T=10 s 实测反向****（本节点实读）：
  A0 格在既有 L0 计划里已按**同规格**跑过（bias=1.00 × σ=0.30，T=10 s，settle=500 ms，chunk）⇒
  含 settle 的墙钟单价 = **zero_drive 0.7704 / 0.7708 / 0.7533**、**baseline 0.8399 / 0.8453 / 0.8447** ms/step，
  **低于**本预算用的 1.1872（T=1 s 探针值，含**长进程启动 + MPS/CPU 争用**，见 R112 ⑥）
  ⇒ A0 的预算**偏保守**（A0 按墙钟锚只有 **0.2818** GPU-h，是 mid 估计 0.4155 的 68%）。
  **但 bias=1.30 三格的 T=10 s 稳态单价仍未测**（活动量高 ⇒ 更贵）⇒ **不得**据此统一下调；
  必须实跑复探（`budget.caveat_T10`：**先跑 A0 的 6 臂**，用其实测值对齐后再放行 A1/B1/B2）。
- **另一条实读的口径发现（登记，用于预算修正）**：生产者对 **settle 用 `delivery="step"`**（L255）、
  对**测量窗用计划里的 `delivery="chunk"`**（L257–258）⇒ 每臂有一次**固定的慢步开销**：
  A0 zero_drive s0 实读 `wall_s=161.79`（含 settle）而引擎测量窗单价 0.6487 ⇒ 测量窗 ≈129.7 s，
  **settle 10,000 步 ≈ 32.1 s（≈3.2 ms/step，比 chunk 慢 ≈4.9×）**。24 臂 ⇒ **≈0.2 h 固定开销**，
  已在 `budget.wall_clock` 的"逐 run 新进程开销 0.2–0.4 h"内计入。

---

## §8 🔴 与"钉死网格"的冲突登记（**必读：影响这 4 格能否命中**）

主 agent 派工把 L3 目标钉死为 `bias ∈ [1.20,1.40] × σ ∈ [0.15,0.25]`（依据 **R115 ④**）。但**R118 之后的探针**（本节点实读
`data/m11_p5_l3_feasibility/` 全部 10 个产物）给出：

| 格（bias=1.30） | 零驱动 spk | 基准 spk | **R（单实现、T=1 s 粗估）** | 是否落在窗口 `[300, 0.10×base]` | 本计划对应 |
|---|---|---|---|---|---|
| σ=0.30 | 4,126 | —（无实测） | — | 不可判 | A1 |
| **σ=0.20** | 4,875 | 5,546 | **0.879** | **否**（R ≫ 0.10） | **B1（主目标格）** |
| **σ=0.15** | 4,666 | 5,295 | **0.881** | **否**（R ≫ 0.10） | **B2** |
| σ=0.10 | **46** | 5,149 | **0.0089** | R 达标，但 **46 < 300 ⇒ 硬门① ⇒ `unavailable`** | **不在本计划**（在正在跑的 v2 网格 C2 格） |

**⇒ 结论（如实登记，不粉饰）**：
1. **σ 在此是"悬崖"不是旋钮**：0.15 → 0.10 使 R 崖降 ~100×，而基准臂只掉 2.8%（= R118 §① 的"尾部选择性"）。
   可判窗口的真实位置在 **σ ∈ (0.10, 0.15)** 这一极窄区间，**与本计划 B1/B2 的 σ 不重合**。
2. ⇒ **按探针预期，B1/B2 会命中证伪式 (ii) `R > 0.10`**；（A0 预期命中 (ii)：R=0.296；A1 零驱动 4,126/s ⇒ 也远超窗口）。
   **即：这 4 格很可能整片命中"L3 失败 ⇒ 第四维结构对立确认"**。
3. **仍然值得跑的唯一理由（也是本计划价值的定位）**：把 R118 §④ 登记的**三处外推**（T=1 s⇒10 s 的计数线性、
   单实现⇒3 实现、活跃占比⇒中位率）**逐条证实或证伪**——探针读数一律 `provisional`（`docs/m11_l3_resume.md` §四）。
   本计划把这 4 格与既有 v2 网格（C2/A0/C1/C3/C0）**互补**而非重复。
4. **若主 agent 的目标是"命中工作点"**：应把格设在 σ∈(0.10,0.15)（既有正式跑已用 σ=0.10）⇒ 本计划**不覆盖**该需求，
   且**不得**把本计划当作"L3 尚未设计"的证据（既有 `m11_p5_plan_l3.frozen_for_run.json` 已在跑，见 §0）。

---

## §9 本节点**未使用 MPS** 的声明 + 实际执行命令清单

**声明**：本节点**从未**使用 `--device mps`，**从未**跑任何 MPS 作业，**未**启动任何新服务器，**未**修改
`tools/run_m11_p5_arms.py` / 评估器 / 任何既有计划 JSON（唯一一次误覆盖已逐字节恢复，见 §0）。
**未**使用 `2>/dev/null` 掩盖任何命令（纪律第五十九条）。唯一一次真跑仿真使用 `device="cpu", n_neuron=1000`（§2.10），
写在 `/tmp`；正在跑的 MPS 正式作业（`data/m11_p5_l3_runs/`）**未被我触碰**。

实际执行过的命令（按类别，全部可重放）：

```bash
# ① 只读：读权威文档/产物（read 工具 + python3 标准库 + grep/sed/wc/shasum）
grep -n "§3dc\|R112\|R113\|R114\|R115\|R116" neural_exploration/docs/m11_acceptance_master.md
python3 -c "import json; …"                      # 读 L1 计划键集/L2_design/targets.csv/既有臂产物
grep -n "提高背景驱动" neural_exploration/docs/m11_b3_g1_design.md      # ⇒ L146（禁）
grep -n "selfsustain_zero_drive_min_spikes" neural_exploration/data/m11_p5_targets.csv   # ⇒ L48 = 300
grep -n "drive_rate" neural_exploration/data/m11_p5_targets.csv          # ⇒ L119（已补登）
sed -n '250,320p;405,460p;537,600p' neural_exploration/tools/run_m11_p5_arms.py   # 读码（零活动分支/键读取/失败策略）

# ② 生成 + 自检（纯 python3 标准库；无 torch/无 MPS）
python3 -m neural_exploration.tools.make_m11_p5_plan_l3             # 写 data/m11_p5_plan_l3_replan.json
python3 -m neural_exploration.tools.make_m11_p5_plan_l3 --check      # 与磁盘逐字节比对 + 重跑全部断言
python3 -m neural_exploration.tools.make_m11_p5_plan_l3 --keys       # 打印生产者键读取比对表（行号+片段）

# ③ 接口验证（生产者自己的预检；纯 Python，无仿真）
./.venv-m11/bin/python -c "… assert_plan_replication(plan) …"        # ⇒ PREFLIGHT_OK {'n_groups':8,'n_arms':24}

# ④ 极小冒烟（**device=cpu / n=1000 / T=1000 ms**；任务书允许，非必需）
./.venv-m9/bin/python -m neural_exploration.tools.run_m11_p5_arms \
    --arm /tmp/m11_l3_smoke/arm.json --out-dir /tmp/m11_l3_smoke/out
# ⇒ [DONE] l3_a0_smoke1000_T1000ms_d1_s0 spk=355 ei_ratio=1.8497 ms/step=0.1645 wall=4.54s EXIT=0

# ⑤ 事故发现与恢复（§0）
git status --porcelain
git show HEAD:neural_exploration/data/m11_p5_plan_l3.json > /tmp/plan_l3_HEAD.json && cp … && git status --porcelain <该文件>
```

---

## §10 本报告指纹（**回读**，纪律第五十九条）

- 本报告在追加本段**之前**的实际回读值（`wc -l -c` + `shasum -a 256`）见下（由追加命令产出）：

| 项 | 值 |
|---|---|
| 行数 | **498** |
| 字节 | **40912** |
| sha256 | **`e23d4c48b99f2f61529c476e45653ce4475834e4f76f5e6f8164aa118d5562a3`** |
| 口径说明 | 上述指纹取自**本段被追加之前**的文件字节（自指 sha 不可能存在）；追加本段后文件增大，请以 `shasum -a 256` 现读为准 |

**三件套回读（同一时刻）**

```
$ wc -l -c data/m11_p5_plan_l3_replan.json tools/make_m11_p5_plan_l3.py docs/m11_p5_plan_l3_selfcheck.md
   1638   62781 data/m11_p5_plan_l3_replan.json
    923   69055 tools/make_m11_p5_plan_l3.py
$ shasum -a 256 data/m11_p5_plan_l3_replan.json tools/make_m11_p5_plan_l3.py
539325a13aa201cd5bc480d0b941768fb452ef2e36caccd3c90fba29ff1af98b  data/m11_p5_plan_l3_replan.json
de97ebaf9122ffdf8669ceab43dae69a2109eb01fc9c99af6a87f3d8e1f7cebc  tools/make_m11_p5_plan_l3.py
4aab49b00177a22e21f65b884bd29d95a8e7333e76cb109ae8fcd8bb9bd94615  data/m11_p5_plan_l3.json   ← 既有（已恢复为 HEAD 版本）
```

**未确认项汇总**：§6 的 9 条（其中 #1/#2 是预算代理、#3 是 T=10 s 单价、#4 是同步指标不可得、
#5 是评估器硬门⑤ 的登记缺口、#6 是引擎 `ms_per_step` 计时口径、#7 是文档 R115 的 ≈9,076 近似值不可复现、
#8 是 σ∈(0.10,0.15) 未测、#9 已确认为"缺口已关闭"）。

**fail 项汇总**：**仅 1 条** —— §2.4 的"全计划 24 行三元组互异（字面）"= **fail（by-design，且不应消除，
理由见 §2.4 三条：生产者判据是分组去重 / A0 锚要求跨格复用 seed / R93 只定义 3 组对）**。
**其余全部 pass 或"登记（非失败）"**。
