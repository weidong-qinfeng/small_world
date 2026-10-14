# M11 · P5 **L3 冻结计划独立复核报告**

> **复核对象**：`neural_exploration/data/m11_p5_plan_l3.json`（**正式跑正在使用**的冻结计划，**只读**）
> **复核方式**：全部断言由本节点**独立重算**（`python3` 标准库 + 只读读盘），**不采信**计划内的任何自报值；
> 计划内/文档内自报的数字逐条对账（§3）。**本次复核对 `data/` 下任何文件均无写入**（详见 §0/§6）。
> **复核工具（可重放）**：`python3 -m neural_exploration.tools.make_m11_p5_plan_l3 --verify-frozen`（**只读模式**：
> sha 校验 → 30 条断言 → 预算对账 → 从产物独立重算已完成臂；事实落盘到 `/tmp`；`refuse_data_write()` 硬闸门拒绝写 `data/`）。
> **本报告替换了上一版**（"L3 网格 v2 计划自检"）的同一路径文件；旧内容已备份至 `/tmp/m11_p5_plan_l3_selfcheck_v2_prev.md`，
> sha256 `8543d1f013a7b1aaac3c4c6195aa7db54605fd83b4f0039459563795a95d192a`（可追溯）。

---

## §0 结论摘要（先看这四条）

1. **计划文件 sha256 三方一致，未被改动**：`data/m11_p5_plan_l3.json` = `data/m11_p5_plan_l3.frozen_for_run.json`
   = 运行脚本日志所载 = **`4aab49b00177a22e21f65b884bd29d95a8e7333e76cb109ae8fcd8bb9bd94615`**（本节点实读，§2 A0）。
   **⇒ 规则变更已落实**：我的生成器 `tools/make_m11_p5_plan_l3.py` 已改为 ① 默认只写**临时文件**；
   ② 新增 `refuse_data_write()` **硬闸门**：任何落在 `neural_exploration/data/` 下的写入一律 `AssertionError` 拒绝；
   ③ 新增 `--verify-frozen` **只读模式**。
   （**诚实登记**：**早前轮次我曾误覆盖过 `data/m11_p5_plan_l3.json`**，随即用 `git show HEAD:…` 逐字节恢复；
   本轮 `shasum -a 256` 复核确认它**仍等于** `4aab49b0…`、与冻结副本逐字节相同 ⇒ **审计链完好**。）
2. **计划结构断言 30/30 全部 PASS**（臂数/格位/顺序/步数/复刻轴/键集合/生产者键名/驱动禁令/臂键集合/角色一致性/A0 锚/条款块）—— §2。
3. **与主 agent 自报数字对账：结构类逐条一致；3 项口径差异 + 1 项文档近似值**—— §3。
4. **🔴 一条重大发现（超出本轮要求，但直接影响该计划的目标）**：**决策格 C2 的 (c) 已可判 `fail`**，
   且其原因是 **R118 §② 的"×30"窗口外推被实测否证约 56×** —— §4。**这不是"计划写错了"，而是"计划所依据的那一步外推被正式数据推翻"。**

---

## §1 复核对象与产物指纹

| 项 | 值（本节点回读） |
|---|---|
| 计划（只读） | `neural_exploration/data/m11_p5_plan_l3.json`，1056 行 / 28,005 B，sha256 `4aab49b0…`，mtime `10-14 07:55` |
| 冻结副本 | `neural_exploration/data/m11_p5_plan_l3.frozen_for_run.json`，1056 行 / 28,005 B，sha256 **同上** |
| 运行日志 | `/tmp/ma_l3.log` 第 1 行 `L3_PLAN_SHA=4aab49b0…`（**三方一致**）|
| 运行脚本 | `/tmp/l3_run.sh`（`case "$OUT" in /*)` 绝对路径断言；`--plan "$NE/data/m11_p5_plan_l3.json"`）|
| 生产者 | `tools/run_m11_p5_arms.py`，894 行，sha256 `633a4c458e9c1dd58e5d6c4838a205819a9424fbd49df9da8827c8b7bd42285b`（= 计划 L46/L62 所记 ✓）|
| 本报告 | `neural_exploration/docs/m11_p5_plan_l3_selfcheck.md`（sha 见 §7）|
| 复核事实档 | `$TMPDIR/m11_p5_plan_l3_verify_facts.json`（**不在仓库内**）|

---

## §2 逐条断言（**独立重算**：实测值 + pass/fail）

（下表由 `--verify-frozen` 实跑输出；**先校验冻结计划 sha，再执行全部只读断言**。）

| # | 断言 | 实测值 | 判定 |
|---|---|---|---|
| A0 | 冻结计划 sha == `4aab49b0…` | `4aab49b00177a22e21f65b884bd29d95a8e7333e76cb109ae8fcd8bb9bd94615` | **PASS** |
| A1 | 臂数 = **30** | `n_arms=30` | **PASS** |
| A1b | 格数 = 5 | `p5_L3_{C2,A0,C1,C3,C0}_…` | **PASS** |
| A1c | 每格 6 臂（2 角色 × 3 实现） | 5 格全部 = 6 | **PASS** |
| A2 | 格位 `(bias,σ)` == R118 修订 | 差集 `[]`；实测 = `{(C2,1.30,0.10),(A0,1.00,0.30),(C1,1.15,0.10),(C3,1.45,0.10),(C0,1.00,0.10)}` | **PASS** |
| A2b | **运行顺序 == `C2→A0→C1→C3→C0`** | `['C2','A0','C1','C3','C0']`（由 `arms[]` 中 `config_id` **首次出现顺序**推得） | **PASS** |
| A2c | `ordering_and_its_reason` 与臂顺序一致 | 文本以 `C2(1.30,0.10) → A0 锚点(1.00,0.30) → C1…` 开头（计划 **L1056**） | **PASS** |
| A3 | `T_ms == 10000.0`（逐臂 30/30） | 取值集合 `[10000.0]` | **PASS** |
| A3b | `settle_ms == 500.0`（计划级） | `500.0` | **PASS** |
| A3c | `settle_ms` 逐臂覆盖 30/30 | 缺键数 = 0 | **PASS** |
| A3d | **步数**：`round((500.0+10000.0)/0.05)=round(210000.0)=210000 ≥ 200000` | `steps=210000` | **PASS** |
| A3e | `dt_ms == 0.05` | `0.05` | **PASS** |
| A4 | 每格每角色 **3 个 `(net_seed,event_seed)` 两两互异**且 == 预注册对 | 10 组、组内均 3 个互异；逐臂 = `(seed, 11+seed)` ⇒ `(0,11)/(1,12)/(2,13)` | **PASS** |
| A5 | `(config_id, role, seed)` 三元组唯一 | `n_unique=30` | **PASS** |
| A5b | `run_id` 唯一 | `n_unique=30` | **PASS** |
| A6 | 键集合与 `data/m11_p5_plan_l1.json` **逐键对齐（差集空）** | `set(L1)−set(冻结) = []`（L1 的 30 键全保留；冻结 39 键，多 9 键）；**子结构差异见 §5.1** | **PASS** |
| A7 | 生产者**键读取行**与计划键名逐字比对 | 24 键中 **22 键有实读点**（行号+片段见 §2.1）；`stage`/`lever` **生产者不读**（装饰性，显式登记） | **PASS** |
| A8 | 无"以驱动抬升作标定"的档 | `drive_mult` 取值集合 = `{0.0, 1.0}`，越界臂 `[]` | **PASS** |
| A8b | `drive_rate_hz == 8.0` | `8.0` | **PASS** |
| A8c | 无臂覆盖 `drive_rate_hz` | 覆盖臂 `[]` | **PASS** |
| A8d | `levels` 内无 `drive_rate_hz`/`drive_mult` 键 | 越界格 `[]` | **PASS** |
| A9 | 每臂**恰好 24 必需键**（全计划同一键集合） | `key_set 数 = 1`，大小 24 | **PASS** |
| A10 | `role ↔ drive_mult` 一致（zero_drive↔0.0 / baseline↔1.0） | 异常 `[]` | **PASS** |
| A10b | `config_id ↔ (bias,σ)` 一致（如 `p5_L3_C2_b1.30_s0.10`） | 异常 `[]` | **PASS** |
| A11 | A0 锚的既有 L0 产物可读（确定性核对前提） | 读到 6/6：零驱动 8,591/8,585/9,802、基准 30,662/30,574/29,910 | **PASS** |
| A12 | `stop_rules` 8 条 | `n=8` | **PASS** |
| A12b | `a_reporting` 含四条（`a_median_hz`/`a_active_frac`/`a_pop_mean_hz`/**`pooled_zero_drive`**） | 主判口径 + 三个附加键齐备 | **PASS** |
| A12c | `unmeasured_side_declaration` 含 `by_measurement_excluded`（"σ∈[0.15,0.25] 于 bias≥1.20 —— **不是未测，是已测且远超上界**"） | 命中 | **PASS** |
| A12d | 窗口口径（300 / 0.10 / R）在计划内 | `lever_prereg_falsifier.statement` 内含 `pooled_zero_drive∈[300, 0.10×pooled_baseline]` | **PASS** |
| A13 | 我的生成器内存版与冻结计划**臂集合完全相同** | 只多 `[]` / 只少 `[]` | **PASS** |
| A13b | 两者**顺序不同**（我的版本 A0 先；冻结版 C2 先） | `mine=[A0…]` / `frozen=[C2,A0,C1,C3,C0]` | **PASS** |

### 2.1 生产者键读取行（**行号 + 代码片段**，实读 `tools/run_m11_p5_arms.py` sha `633a4c45…`）

| 计划键 | 行号 | 代码片段 |
|---|---|---|
| `run_id` | 302 | `run_id = str(spec["run_id"])` |
| `config_id` | 314 | `arm_id=run_id, config_id=str(spec.get("config_id", "p5")),` |
| `T_ms` | 237 | `T_ms = float(spec["T_ms"])` |
| `drive_mult` | 242 | `dose = float(spec.get("drive_mult", 1.0))` |
| `seed` | 159 | `seed = int(spec.get("seed", 0))` |
| `net_seed` | 134 | `net = build_laminar_probe_column(n, seed=int(spec.get("net_seed", 0)), delay_bins=bins, **net_kw)` |
| `event_seed` | 161 | `es = spec.get("event_seed", None)`（→ L163 未给则派生 `EVENT_SEED_BASE+seed = 11+seed`）|
| `role` | 286 | `_zero_drive_expected = (str(spec.get("role", "dose")) == "zero_drive") or (dose == 0.0)` |
| `mechanism` | 316 | `mechanism=str(spec.get("mechanism", "none")), drive_mult=dose,` |
| `delivery` | 257 | `delivery = str(spec.get("delivery", "step"))` |
| `gmax_scale` / `bias_scale` / `bias_log_sigma` / `indegree` | 124 | `if k in ("indegree", "gmax_scale", "bias_scale", "bias_log_mean", "bias_log_sigma")`（→ `**net_kw`）|
| `delay_bins` | 125 | `bins = tuple(spec.get("delay_bins") or NET_DEFAULTS["delay_bins"])` |
| `subtypes` | 127 | `eng.configure_subtypes(bool(spec.get("subtypes", True)))` |
| `slow` | 128 | `if spec.get("slow", True):` |
| `tau_slow_ms` | 129 | `eng.configure_slow(True, tau_slow_ms=float(spec.get("tau_slow_ms", 1000.0)),` |
| `slow_inc` | 130 | `inc_per_spike=float(spec.get("slow_inc", 6.0)))` |
| `n_neuron` | 236 | `n = int(spec["n_neuron"])` |
| `settle_ms` | 238 | `settle_ms = float(spec["settle_ms"])` |
| `device` | 246 | `device = str(spec.get("device", "mps"))` |
| `stage` / `lever` | **—** | **生产者不读**（装饰性键；已在计划与本节显式登记，**不得**期待其有物理效果）|

**⇒ 24 键中 22 键逐字命中实读点；2 键登记为装饰性**（计划里的 `judged_units`（字符串）与 `levels`（5×`{cell,bias_scale,bias_log_sigma,note}`）
生产者都不读 ⇒ 不影响运行；但见 §5.1 的接口一致性登记）。

---

## §3 与主 agent 自报数字逐条对账

### 3.1 **一致（逐条）**

| 项 | 主 agent 自报（出处） | 我的独立重算 | 判定 |
|---|---|---|---|
| 计划 sha256 | `4aab49b0…`（本次消息 + R119 §③ + `/tmp/ma_l3.log` L1） | `4aab49b0…` | **一致** |
| 臂数 | 30（R119 §①/交接件 §一） | 30 | **一致** |
| 格位 | `(1.30,0.10)/(1.00,0.30)/(1.15,0.10)/(1.45,0.10)/(1.00,0.10)`（本次消息 + R118 修订文件） | 同（差集空） | **一致** |
| 顺序 | `C2→A0→C1→C3→C0`（本次消息 + 计划 L1056） | 同 | **一致** |
| `T_ms`/`settle_ms`/步数 | 10000.0 / 500.0 / 210,000（R119 §①） | 10000.0 / 500.0 / 210,000 | **一致** |
| 复刻轴 | 每格 `(0,11)/(1,12)/(2,13)` 两两互异各 3（R119 §①） | 10 组 × 3 互异 | **一致** |
| 三元组唯一 | `(config_id,role,seed)` 唯一（R119 §①） | `n_unique=30` | **一致** |
| 键集合对齐 | "与 `m11_p5_plan_l1.json` 完全对齐（差集空）"（R119 §①） | `L1−冻结 = []`（顶层） | **一致**（子结构差异见 §5.1）|
| 驱动禁令 | 无驱动抬升档、`drive_rate_hz=8.0`（R115 §③/R117/targets L119） | `drive_mult∈{0,1}`、8.0、无覆盖键、levels 无驱动键 | **一致** |
| C2 基准臂 (a)/(e) 等六项 | 中位率 **0.6000 Hz**、`ei_ratio` **2.5970**、spk **52,601**、`active_frac` **1.0000**、`ms/step` **0.9318**、亚型中位率 4×0.6、亚型活跃占比 4×1.0（**R121 §①②**） | 从 `*_counts.csv` 独立重算：**0.6000 / 2.59702 / 52,601 / 1.0000 / 0.93176 / 4×(0.6, 1.0)** | **一致（逐位级）** |
| C2 基准臂 3/3 完成且高度一致 | HEAD=`1885532`（R128）"3/3 实现全部完成且高度一致" | 3/3：52,601 / 52,572 / 52,349 脉冲；中位率 3×0.6000；`ei` 2.5970 / 2.6034 / 2.5839（极差 **0.19%**） | **一致** |
| R118 §① σ 悬崖 | 4,126 / 4,875 / 4,666 / **46**（σ=0.30/0.20/0.15/0.10，bias=1.30 零驱动，T=1 s） | 与 `data/m11_p5_l3_feasibility/*.json` **逐值一致** | **一致** |
| A0 锚 (c) | `R=0.296`（R115） | 既有评估器 `data/m11_p5_resting_eval_repl.json`：`R=0.295987`、pooled 26,978 / 91,146 | **一致**（R 逐位）|

### 3.2 **不一致 / 需判定的项（逐条：项、我的值、他们的值、判谁错、依据行号）**

**① 预算 `gpu_h_mid` / `upper` —— 判：口径差异，他们**偏保守**，不是错误（但"mid"标签名不符实）**
- 项：计划 `budget.gpu_h_mid` / `gpu_h_upper`（计划 **L104–119**；其 `unit_price_note` 用"1.30 / 1.60 ms/step 作为 1.00–1.45 bias 段的**代表值与上界**"）。
- 他们的值：**mid 2.275 / upper 2.8**（runs=30、steps=6,300,000）。
- 我的值（**逐格加权**，单价取 T=1 s 探针实测）：per-cell mid = A0 `0.4155` + C0 `0.3009` + C1 `0.3367` + C2 `0.3726` + C3 `0.3726`
  ⇒ **mid 1.7983**；upper（全臂按本网格 T=1 s 实测最大值 1.4856）**2.5998**；
  若把 C2 baseline 换成**正式 T=10 s 实测均值 0.9603** ⇒ **mid ≈ 1.7064**。
- **判谁对**：**逐格加权为准**。依据 `docs/m11_acceptance_master.md` **R112 §⑤**（"预算外推**必须按档取价**，不得用单一价外推全网格"——R112 的第 30 次公开更正）；计划用**单一代表值 1.30** 覆盖全网格，方向与该条相反（取值偏保守）。
  **净效应**：其 mid 比逐格加权高 **26%**（2.275 vs 1.7983）、比按正式 T=10 s 实测折算高 **33%**（2.275 vs 1.7064）⇒ **风险侧安全（不会超支），但估时偏悲观**。

**② 墙钟"每臂约 1.75 h / 整轮约 68 h" —— 判：占空比口径，非计算错误；但严重偏悲观**
- 项：本次消息"每臂约 1.75 h 墙钟，整轮约 68 h"；计划 **L1056** 亦写"占空比 ≈5% ⇒ 30 臂 ≈68 h"；交接件 §一写"占空比 ≈18.8% ⇒ 单臂 ≈28 min ⇒ 30 臂 ≈14 h"。
- 我的值（**从正式跑产物独立重算**，5 臂已完成）：`wall_s` = baseline **236.3 / 247.1 / 254.2 s**、zero_drive **195.4 / 202.8 s**
  ⇒ 平均 **227 s/臂 ≈ 3.8 min** ⇒ **30 臂纯计算（GPU-busy）≈ 1.89 h**。
- 换算：墙钟 = 纯计算 ÷ 占空比 ⇒ **@5.4% ≈ 35.1 h**、**@18.8% ≈ 10.1 h**、**@2.9% ≈ 65.3 h（≈ 他们说的 68 h）**。
- **判谁对**：**"1.75 h/臂"是占空比折算值，不是纯计算值** —— 它隐含的占空比 ≈2.9%，**低于你们自测的 5.4%–18.8%**
  ⇒ 数字本身**不算错，但口径必须写明**，否则会误导"要不要为抢关键档而改建制"的决策：按交接件自测的 18.8%，整轮只有 **≈10 h**（而不是 68 h）。
- 依据：`data/m11_p5_l3_runs/*.json` 的 `wall_s`/`ms_per_step_steady`；`/tmp/ma_l3.log`；`docs/m11_l3_resume.md` **L28–31**；计划 **L1056**。

**③ A0 零驱动"≈9,076 脉冲/实现" —— 判：**文档近似值不可复现**，以产物为准**
- 他们的值：`docs/m11_acceptance_master.md` **L3156 / L3161 / L3238** 写 `≈9,076`（@T=10 s，`bias=1.00, σ=0.30` 零驱动）。
- 我的值（实读产物）：`data/m11_p5_l0_runs/l0_g1.00_b1.00_8192_T10000ms_d0_s{0,1,2}.json` = **8,591 / 8,585 / 9,802**
  （3 实现合计 **26,978**；均值 **8,992.7**）。
- **判谁错**：**文档的"≈9,076"不可复现**（对均值差 **0.9%**、对 seed 0 单值差 **5.6%**）；而 **pooled 26,978 与 `R=0.295987` 可逐位复现**（既有评估器 `data/m11_p5_resting_eval_repl.json`）⇒ **以产物为准**。影响：仅"每实现约数"的表述，**不影响窗口与 R**。

**④ 🔴 R118 §② 的窗口外推 —— 判：**已由正式数据否证（≈56×）** ⇒ 单独见 §4。**

### 3.3 未能对账的项（**如实登记**）
- 计划 `budget.measured_prices` 里的 `bias=0.70, σ=0.30 = 0.6415`：对应产物为**冒烟臂**，本节点**未逐步核到该臂 JSON** ⇒ 登记为**未确认**
  （不影响本网格：`bias=0.70` 不在 v2 网格内）。
- **R121 §③**"短窗中位率系统性偏高 1.67×"：其"探针 ≈1.0 Hz"是 T=1 s **粗量化**（计数 0/1/2）结果 ⇒ 本节点**无法独立复算该探针的"真实值"**
  （只能确认正式 T=10 s = **0.6000 Hz** ✓）⇒ 该倍数属**推断**，登记。
- **实时性说明**：运行中的作业**在我复核期间继续产出**（本节点看到 5 臂完成；`data/m11_p5_l3_runs/` 仍在增长）⇒ §4 的 pooled 值为**某一时刻的快照**，其中"第 3 个 zero_drive 实现"当时未完成（已按最保守假设处理，见 §4②）。

---

## §4 🔴 重大发现：决策格 C2 的 **(c) 已可判 `fail`**，且 **R118 §② 的"×30"窗口外推被否证 ≈56×**

> 这不是"计划写错了"，而是"**计划所依据的那一步外推**（R118 §④1 自己登记的"T=1 s ⇒ T=10 s 计数线性"）**被正式数据推翻**"。

**① 实测（本节点从产物独立重算，不采信任何自报值）**

| C2（`bias=1.30, σ=0.10`） | 实现 0 | 实现 1 | 实现 2 | pooled |
|---|---|---|---|---|
| `baseline` 脉冲（counts 求和 / 臂 JSON 双估计器一致） | 52,601 | 52,572 | **52,349** | **157,522** |
| `zero_drive` 脉冲 | **38,583** | **38,426** | 未完成（预计同量级） | **≥77,009** |
| `zero_drive` 中位率 / 活跃占比 | 0.5000 Hz / 0.9907 | 0.5000 Hz / 0.9899 | — | — |
| `zero_drive` `ei_ratio` | 5.2614 | 5.2557 | — | — |

（双估计器核对：每一臂 `spikes_total` 与 `counts.csv` 求和**逐臂相等** ⇒ 硬门③口径在这一侧无异常。）

**② 判定（**即使第 3 个零驱动实现贡献 0** 也已成立）**
```
窗口上界 = 0.10 × pooled_baseline = 0.10 × 157,522 = 15,752.2
pooled_zero_drive（已完成 2/3） = 77,009      ⇒ R = 77,009 / 157,522 = 0.4889
若第 3 实现 ≈38,500              ⇒ pooled ≈ 115,509 ⇒ R ≈ 0.7333
硬门①（pooled_zero_drive ≥ 300）  = 满足（77,009 ≫ 300）⇒ **不是** unavailable 支
⇒ **(c) 判 `fail`**（走的是 `R > 0.10` 这一支）
```
**⇒ C2 的 (c) 已可判 `fail`（不需等第 3 臂，也不需等评估器）**。这比 R121 §④ 保留的谨慎表述（"(c) 未定……**仍可能触发** R115 §② 的陷阱"）**更强**：现在是实测结论。

**③ 与 R118 §② 的差值（判谁错：**外推错**，以实测为准）**
- R118 §②（`docs/m11_acceptance_master.md` **L3370–3377**）假设 `pooled ≈ 探针读数 × 10(T) × 3(实现) = 46 × 30 =` **≈1,380**，
  并据此判"**✓ 在窗口内；`R ≈ 0.0089` ⇒ 可判且 pass**"（**L3376**）。
- 实测（**仅 2/3 实现**）pooled = **77,009** ⇒ **×55.8**；按单实现比：线性外推应得 460，实测 **38,583** ⇒ **×83.9**。
- **⇒ R118 §④1"计数随 T 线性"在 σ=0.10 的**零驱动侧被证伪**：自发活动在 10 s 内**持续累积**（T=1 s 46 个脉冲 → T=10 s 3.86×10⁴）。
- **⇒ 连带影响（必须让主 agent 知道）**：R118 §①② 的"**窗口只在 σ=0.10 处可达**"这一结论，其**可达性**部分建立在同一个 ×30 外推上
  ⇒ **至少对 `bias=1.30` 已被实测推翻**；本计划其余 4 格（A0/C0/C1/C3）的窗口可达性**同样只有 T=1 s 探针/无量测支撑**
  ⇒ **不得**再用"窗口可达"作为设格前提。
- **反平凡性（正面确认）**：C2 基准臂 `active_frac = 1.0000`（8,192/8,192 全部发放）⇒ **不是"把网络弄死换取 R→0"**；
  同时零驱动侧 7.7×10⁴ 个脉冲 ⇒ **也不是"零驱动被打干净导致 `unavailable`"**
  ⇒ 这是**真实的失败**：被驱动响应完好，而**自发（零驱动）活动未被消除**。

**④ (a) 侧正面确认**：C2 基准臂中位率 **0.6000 Hz ∈ [0.5, 2.0]**（3/3 实现一致；亚型 E/PV/SST/VIP 各自 0.6 Hz、活跃占比 1.0）
⇒ **本计划的 (a) 目标是达成的**，失败**只在 (c)**。

**⑤ 对 falsifier 的落点**：计划 `lever_prereg_falsifier.falsifiers①` 的 (ii) 支（`R > 0.10`）**已在 C2 命中**；
按计划 `stop_rules ①`（本计划为扫描、**不自动早停**）与 `⑧`（"σ=0.10 的全部 bias 档失败 ⇒ 报『悬崖不可达窗口』⇒ 换机制"），
**建议**：C2 的 `zero_drive` 第 3 臂跑完后即可**提前给出 (c) 的正式判定**，无需等整轮 68 h（另见 §3.2② 的墙钟口径更正）。

---

## §5 观察项与未确认项（**非判定错误，但登记**）

### 5.1 与 L1 键集合模板的**子结构**差异（顶层差集已空 ✓，但接口形状不同）
| 键 | L1 的形状 | 冻结计划的形状（行号） | 风险判断 |
|---|---|---|---|
| `parent_prereg` | `dict{path, sha256_full, cited{…}}` | **字符串** `"data/m11_prereg_amend_r118_l3_grid_v2.json"`（**L26**）；父哈希另写在 `parent_prereg_note`（含 `9c6f7c60…`）| **低**：父哈希仍可查，但**机器不可直接读** ⇒ 下游按 L1 形状取 `plan["parent_prereg"]["sha256_full"]` 会 `TypeError` |
| `levels` | `list[dict{config_id,stage,gmax_scale,bias_scale,bias_log_sigma,lever,n_seeds,seeds,net_event_pairs,elasticity_arm,run_ids}]` | `list[dict{cell,bias_scale,bias_log_sigma,note}]`（**L120–151**） | **中**：**无 `config_id`/`run_ids`** ⇒ 下游若"按 levels 枚举判据单元"会缺键；本计划臂清单实际在 `arms[]`（**齐全 ✓**）|
| `judged_units` | `list[dict{config_id,T_ms,mechanism,zero_drive_seeds,baseline_seeds,note}]` | **字符串**（**L44**） | **中**：评估器**不从计划读**（它读协议 JSON，R120 §②）⇒ 实际影响低，但接口不一致须登记 |
| `units` | 含 `bias_scale`/`gmax_scale`/`bias_log_sigma` 的单位声明 | 只声明 `T_ms/settle_ms/dt_ms/wall_s/ms_per_step/drive_rate_hz`（**L28–43**） | **低但违纪律**：**R80 §⑥** 要求"跨接口传递的量必须在接口定义处写明单位"，本计划**未登记** `bias_scale`/`gmax_scale`/`bias_log_sigma`（虽无量纲）⇒ **建议 append-only 补记**（**不得**改冻结件）|

### 5.2 引擎/评估器侧的既有缺口（沿用登记，非本次引入）
- **同步性 / 规则性指标不可得**：产物只导出逐神经元**计数**（`*_counts.csv`），不导出脉冲时刻 ⇒ `CV_ISI`/`Fano` **无法复算**；
  `src/mouse_cortex_engine.py` 无同步性读出（`ei_sync_every` 只是"每 1000 步把 E/I 张量同步回 Python 的间隔"）。
- **评估器硬门⑤**：`mechanism="lever_l3"` ⇒ 同 `(config_id,T_ms)` 无 `mechanism=='none'` 臂 ⇒ `discriminative=None`、G1 至多 `unavailable`
  （`tools/validate_p11_resting.py` **L660–670 / L1334–1360**）⇒ 按 **R120 §③** 判定应以 `tools/report_m11_l3.py` 为主。
- **`ms_per_step` 只覆盖测量窗**（引擎 **L666/L802/L819**）⇒ 与"含 settle 的墙钟单价"差 ≈16%（A0 zero_drive s0：0.6487 vs 161.79 s / 210,000 步 = 0.7704）；
  生产者对 settle 用 `delivery="step"`（L255）、测量窗用 `chunk`（L257–258）⇒ 每臂固定 ≈32 s。
- **`(e)` 口径**：R123/R124 已确证评估器 (e) 口径缺陷（R109 ①② 撤回）⇒ 本次复核**不用 (e)** 做任何判定。

### 5.3 我此前生成的 v2 顺序版计划（**建议作废，未擅自处置**）
- `data/m11_p5_plan_l3_grid_v2_replan.json`（我上一轮的 30 臂 **v2 顺序版**，`A0 先`）**仍在盘上**：
  其**臂集合与冻结计划完全相同、只是顺序不同**（§2 A13/A13b 已验证）。
- 正式跑**正使用冻结计划** ⇒ 该文件应按 **R122 §④ / 纪律第六十三条** 就地标注"不得运行"或删除。
  **我没有擅自改它**（它是上一轮交付物，可能已入库）。**一行命令（由主 agent 裁决是否执行）**：
  `printf '%s\n' '本文件不得运行：正式跑使用 data/m11_p5_plan_l3.json（sha 4aab49b0…）。' > data/m11_p5_plan_l3_grid_v2_replan.json.SUPERSEDED`

---

## §6 本节点执行过的命令清单 + **未使用 MPS** 声明

**声明**：本节点**从未**使用 `--device mps`、**从未**跑任何 MPS 作业、**未**修改生产者/评估器；
**对 `data/` 下任何文件均无写入**（生成器新增 `refuse_data_write()` 硬闸门 + `--verify-frozen` 只读模式；事实档写 `$TMPDIR`）。
**未**使用 `2>/dev/null` 掩盖任何命令。

```bash
# ① sha 三方核对（只读）
shasum -a 256 neural_exploration/data/m11_p5_plan_l3.json \
                neural_exploration/data/m11_p5_plan_l3.frozen_for_run.json     # ⇒ 两者同为 4aab49b0…
cat /tmp/ma_l3.log ; head -20 /tmp/l3_run.sh                                  # ⇒ 日志 L3_PLAN_SHA=4aab49b0…
wc -l -c neural_exploration/data/m11_p5_plan_l3.json                          # ⇒ 1056 28005
ls neural_exploration/data/m11_p5_l3_runs/*.done | wc -l                      # ⇒ 4（复核期间增长到 5）

# ② 只读复核（本轮新增的 verify 模式；**硬闸门拒绝写 data/**）
python3 -m neural_exploration.tools.make_m11_p5_plan_l3 --verify-frozen
#   ⇒ sha 校验一致；A0–A13 共 30 条断言全部 PASS；预算对账；从产物独立重算已完成臂

# ③ 逐臂独立重算（标准库读 counts.csv + 臂 JSON；不跑仿真、不用 numpy）
python3 - <<'EOF'
# 中位率 = median(count)/T_s；活跃占比 = mean(count>0)；亚型分解；双估计器（spikes_total vs Σcount）
EOF

# ④ 引用核对（只读）
grep -n '"budget"\|"levels"\|"judged_units"\|"ordering_and_its_reason"\|"parent_prereg"\|"units"' \
     neural_exploration/data/m11_p5_plan_l3.json
grep -n "≈1,380\|pooled 零驱动（外推）" neural_exploration/docs/m11_acceptance_master.md   # ⇒ R118 §② L3373/3376/3383
sed -n '3357,3398p;3459,3489p' neural_exploration/docs/m11_acceptance_master.md            # ⇒ R118 / R121 原文
```

---

## §7 本报告指纹（**回读**，纪律第五十九条）

| 项 | 值 |
|---|---|
| 行数（追加本段**之前**的回读） | **274** |
| 字节 | **25115** |
| sha256 | **`c31385920f5734d3d5891d7a62a1e1264e1d592885bf4f269cc28f8060ae3729`** |
| 口径 | 自指 sha 不可能存在 ⇒ 上值取自**本段被追加之前**的字节；追加后请以现读 `shasum -a 256` 为准 |

**同一时刻的回读（含复核对象与被复核文件的完整性证明）**

```
$ wc -l -c docs/m11_p5_plan_l3_selfcheck.md tools/make_m11_p5_plan_l3.py data/m11_p5_plan_l3.json
       274  25115  docs/m11_p5_plan_l3_selfcheck.md        ← 本报告
      1583 123302  tools/make_m11_p5_plan_l3.py           ← 生成器（已加 refuse_data_write 硬闸门 + --verify-frozen）
      1056  28005  data/m11_p5_plan_l3.json               ← **复核对象（只读，未改动）**
$ shasum -a 256 <上述三件 + 冻结副本 + 生产者>
4aab49b00177a22e21f65b884bd29d95a8e7333e76cb109ae8fcd8bb9bd94615  data/m11_p5_plan_l3.json                ← **== 期望 4aab49b0… ✓（未被改动）**
4aab49b00177a22e21f65b884bd29d95a8e7333e76cb109ae8fcd8bb9bd94615  data/m11_p5_plan_l3.frozen_for_run.json  ← 冻结副本，同 sha ✓
c31385920f5734d3d5891d7a62a1e1264e1d592885bf4f269cc28f8060ae3729  docs/m11_p5_plan_l3_selfcheck.md         ← 本报告
7fe556651d20f151305eef328c767ce126187d667b8d590151cbc0a561256010  tools/make_m11_p5_plan_l3.py             ← 生成器
633a4c458e9c1dd58e5d6c4838a205819a9424fbd49df9da8827c8b7bd42285b  tools/run_m11_p5_arms.py                 ← 生产者（R48 冻结；未触碰）
```

**结论汇总**：结构断言 **30/30 PASS**；**fail 0 条**；与主 agent 自报数字：**结构类逐条一致**，
**3 项口径差异**（预算 mid/upper、墙钟 1.75 h/臂、文档 ≈9,076 近似值）+ **1 项外推被实测否证（R118 §②，≈56×）**；
**未确认 3 项**（`bias=0.70` 冒烟价未逐步核、R121 的 1.67× 属推断、pooled 值为运行中快照）+ **接口形状差异 4 项**（§5.1）。
