# M12 · B-6：**17 项「清单未注册的口径常数」登记清单**（规划节点 A 交付物）

> **本文件的性质**：**开发清单（规划件）**，不做实现、不改任何已登记 sha 的件。
> **交付对象**：主 agent（裁决）＋ 执行节点 node B（实施）。
> **工作包**：`data/m12_list_review_master.md` **R150-补登**（L109–L129，B-6 的登记内容与性质）；
> **并行安全声明**：`docs/m11_g1_resume_v2.md` **L106** 明文「（B-6）与 L5 并行安全（**纯文档、不涉 MPS**）」✓
> **路径约定**：本文所有相对路径均以 `/Users/weidong/ai/small_world/` 为根。凡未实读的，一律写「未读/未核实」。

## §0 一句话结论＋给主 agent 的六条要点

**17 项逐条对上，没有多也没有少**；但**「未登记」的实际形态与派发描述不同**：
**16/17 项在 `data/m12_targets.csv` 里已有 `final=0` 的「缺口占位行」**（A-5 已建），
**只有第 16 项（P11 锚级 `n_min`）连行都没有**。⇒ **B-6 的工作不是"追加 17 行"，而是"把占位行变成值登记（+1 个新 key）"。**

| # | 要点 | 证据 |
|---|---|---|
| **①** | **17 项核对一致**（逐条与 `docs/m12_criteria_table_skeleton.md` §5.2 L168–186 的 1–17 对上，无合并、无凑数） | §2.1 |
| **②** | **0 项有「M12 口径下的文献/数据依据」**；**10 项是仅工程约定**；**🔴 7 项「查无依据」= 真缺口**（名单见 §2.3） | §2.2/§2.3 |
| **③** | **值的分布**：**11 项**在别处已有硬编码值（M10/M11 注册表或代码），其中 **3 项存在两/三方冲突**（T、剂量网格、点数档）；**6 项全库无任何数值** | §2.2 |
| **④** | **「现在就能登记」4 项**（settle / 驱动率 / bin 宽 / 平滑窗，后两项附口径声明义务）；**只需一次本地裁决即可登记 3 项**；**欠证据、暂不可登记 10 项**。**直接依赖 M11 G1 的 = 0 项**（这正印证 R150「B-6 不依赖 M11」） | §3 |
| **⑤** | 🔴 **一处必须先裁决的机制冲突**：骨架 §7.1 组 B 的解封动作（「每个常数新增一行 `final=1`，不改本行」）与**同表 G0 闸门的"重复 key 即失败"**互斥；本节点**已实测**同 key 追加 ⇒ **G0 FAIL、退出码 1**。可走的路只有**原位填占位行**（M11 第四十四条/第八十条），而它需主 agent 对清单 L971「全表禁止就地改值」作一次解释性裁决 | §2.4-①、§4.2 |
| **⑥** | **另发现 1 个第 18 项候选**：骨架 §5.1 L158/L389 声称 `protocol_T_dwell_ratio_min` 是 `final=1` 行（「唯一有值的 T 类常数」），**实测该 key 在全表 124 行中不存在**（只在骨架文档里） ⇒ 依同一条判据（「凡决定判据取值的常数必须有行」）它也是缺口。**本节点不把它并入 17**（不凑数），单列待裁决 | §2.4-③ |

---

## §1 问题的精确定义

### 1.1 为什么「口径未登记」⇒「协议不可定稿」（五步链，逐步给行号）

| 步 | 命题 | 依据（行号） |
|---|---|---|
| **1** | **凡决定某个判据取值的协议常数必须在本表有行，且 `wrong_usage_consequence` 须写明"若被改动会怎样误导"** | `docs/m12_criteria_table_skeleton.md` **L143–146**（§5.1 机制，直接承 M11 R117） |
| **2** | **`final=0` = 该带未定稿/口径未确认 ⇒ 依赖它的判据记 `unavailable`**（不是 fail，也不是 pass） | `data/m12_targets.csv` **L6**（表内注解）＋骨架 **L131**（§4.2 明文条款）＋闸门 G3 逐行输出（`tools/check_m12_targets_gates.py` **L198–205**） |
| **3** | **判据带的定稿时间必须早于对应协议运行时间**（逐行核对 `mtime`）；**禁止事后调带** | `docs/生物仿真M12实施清单.md` **L971–972**（§5.6 #5/#6）＋ **L1081**（§8.3 第 5 条）＋表内 `rule_final1_frozen_after_run`（`data/m12_targets.csv` **L52**） |
| **4** | **"值测到了 ≠ 条件满足了"**（「未测 ≠ 未命中」的对偶：「**已测 ≠ 可判定**」）；占位行只有在**其声明的全部验收条件已满足**时才可填入 | M11 **第四十四条**：`docs/m11_acceptance_master.md` **L2449**；**第八十条**（事后填入的四个条件）：同文 **L4943–4946** |
| **5** | **清单自己已把"口径未登记"的判定后果写成条文**：`acc_min` **无 provenance ⇒ invalid** | `docs/生物仿真M12实施清单.md` **L370**（§1.2 F3） |

**⇒ 因果收束**：常数未登记 ⇒ 其值不能进判定 ⇒ 依赖它的判据只能落 `unavailable`；
若在跑臂**之后**再补登记，则违反步 3/4（事后调带）⇒ 该轮判定作废（`invalid`）。
**⇒ 二者只能二选一，没有第三条路 ⇒ P5a/P5b/P6a/P9/P10b 永久不可定稿。**
这正是 R150 §② 的原文结论（`data/m12_list_review_master.md` **L117**：「即使 M11 解封，`P5a`/`P5b`/`P6a`/`P9`/`P10b` 仍不可定稿」）。

**🔴 与 M11 最新状态（R183/R184）的关系（本节点实读所得，必须纳入）**：
M11 的 G1 **目前是「不可判 pass」**，且**根因已从"缺带"变成"缺口径桥"**——
`(b)` 的带 `[0.101852, 0.229885]` 是**钙事件检测口径**，而仿真侧统计量是**脉冲计数口径**，**两者之间的桥不存在**（`docs/m11_g1_resume_v2.md` **L8/L13/L27–L35**）。
⇒ **对本清单的两条直接后果**：
1. **M12 侧的行更不能抄 M11 的钙域带**（抄了就是同一个跨口径错误）；
2. **`protocol_silent_frac_band`（`data/m12_targets.csv` **L17**）连"自己属于哪个口径域"都没写** ⇒ 它天然会重演 R183 ⇒ 这是本清单把它判为 🔴③「查无依据」的关键理由之一（§2.3-①）。

### 1.2 B-6 与 A-5 残留验收的关系

- **A-5 的交付物①** = `data/m12_targets.csv`（124 记录/26 `final=0`）＋骨架文档＋闸门工具；
  A-5 在骨架 **§5.2（L164–189）** 把「清单自己没注册、但决定判据取值」的常数**实测清点为 17 项**，并把 **1–16 落成 `final=0` 行、第 17 项登记为"清单内部冲突"**（骨架 **L188–189**）。
- **A-5 的 §10 自检只把"反向缺口 17 项已登记"记为 pass**（骨架 **L432**：「反向缺口 17 项已登记（§5.2）」；同文 **L420–422**：「**它们全部已在本表落成 `final=0` 行**」）。
  ⇒ **那次 pass 证明的是"缺口被登记"，不是"常数被注册"** ⇒ **A-5 的 pass 不覆盖 B-6**。
- **⇒ B-6 = A-5 交付物① 的自认缺口的收口**：A-5 把**缺口**写进了表（占位行），B-6 必须把**值＋口径＋出处＋唯一未满足前提**写进表。
  两者的关系是**同一张表的两个阶段**（缺口登记 → 值登记），不是两份独立工作。
- **A-5 已如实登记的自身未确认项**中与本清单直接相关的两条须沿用：
  **U6**（`protocol_T_main_s = 30` 是否算已注册，骨架 **L339**）与 **U5**（点数档清单内部冲突，骨架 **L338**）。

### 1.3 B-6 与 B-5 的边界差异

| 维度 | **B-5**（真实 ROI 掩膜） | **B-6**（17 项口径常数） |
|---|---|---|
| 缺的是什么 | **数据**（真实活动区掩膜） | **登记**（值＋口径＋出处） |
| 是否要取数/联网 | **要**（已取 field2 掩膜 8/8 COMPLETE） | **不要**（`docs/m11_g1_resume_v2.md` **L106**：纯文档） |
| 是否要 MPS/GPU | 不要 | 不要 |
| 阻塞谁 | **M11 的 G1**（tension 合取项；`m12_list_review_master.md` **L94**） | **M12 自己的 5 个判据定稿**（P5a/P5b/P6a/P9/P10b） |
| 产物形态 | 独立目录 `data/m12_b5_roi_mask/` ＋ `proposed_p5b_criteria_rows.csv`（3 行提案）＋ 审计 JSON | 本文档 ＋ `data/m12_b6_unregistered_constants/proposed_m12_criteria_rows.csv`（3 行提案） |
| **落点** | `data/m11_p5_targets.csv` **L115/L116**（**既有占位行**，已被 R169 依第四十四条**原位填入**） | `data/m12_targets.csv` **L11/L14**（既有占位行，**原位填入**）＋ **1 个新 key**（追加） |
| 共同纪律 | ① **绝不修改注册表本体**（`data/m11_g1_resume_v2.md` **L107** 对 B-6 的明文要求）② 产出独立 CSV 供主 agent 裁决 ③ 数值必须带出处 | 同左 |
| **本清单对 B-5 形态的一处改进** | B-5 的提案 CSV 未显式声明"填"还是"追加" | 本清单的提案 CSV 增 **`mode` 列**（`fill`/`append`）＋ **`precondition_unsatisfied` 列** |

> **B-5 的先例落地形态（可复核）**：`data/m12_b5_roi_mask/proposed_p5b_criteria_rows.csv`（5 物理行，sha256 `f917b7e2…`）；
> 三个提案行 `B5-P1/P2/P3`，其中 `target_row` 直接写 `data/m11_p5_targets.csv L115`/`L116` ⇒ **"填既有行"的语义**；
> B-5 文档 **L274–275** 明文「🔴 **本节点未修改 `data/m11_p5_targets.csv`**…以下为**判据**与**建议值**，须由**主 agent 裁决**后另行写入」。

### 1.4 B-6 的产物形态（本清单主张）

1. **本清单文档**（＝§2 枚举、§3 方案、§4 实施步骤、§5 红线）；
2. **一份独立提案 CSV**（沿 B-5 形态）：`data/m12_b6_unregistered_constants/proposed_m12_criteria_rows.csv`，
   只放"**现在就能写入注册表**"的行（2 行原位填入 ＋ 1 行新增 key），**其余 14 项不进 CSV**（其值/前提尚不具备，写入即造伪登记）；
3. **明确不做**：不改 `data/m12_targets.csv`、不改 `data/m11_p5_targets.csv`、不改任何已登记 sha 的件、不跑仿真、不联网。

---

## §2 那 17 个常数的完整枚举（核心）

### 2.1 核对结论：与「17」是否一致

**✅ 一致：17 项逐条对上**（编号、身份、所指判据三者均与骨架 §5.2 一致），**未发现"少一项/多一项"**。
逐条映射（骨架 §5.2 表格行号 → M12 表 key → 表内物理行）：

| 骨架 §5.2 编号（行） | M12 表 key | 物理行 | 表内 value / `final` |
|---|---|---|---|
| 1（L170） | `protocol_T_main_s` | **L10** | `30` / **0** |
| 2（L171） | `protocol_settle_ms` | **L11** | `unavailable` / **0** |
| 3（L172） | `protocol_bin_wide_ms` | **L12** | `unavailable` / **0** |
| 4（L173） | `protocol_smooth_window_frames` | **L13** | `unavailable` / **0** |
| 5（L174） | `protocol_drive_rate_hz` | **L14** | `unavailable` / **0** |
| 6（L175） | `protocol_drive_dose_grid` | **L15** | `unavailable` / **0** |
| 7（L176） | `protocol_self_sustain_ratio_max` | **L16** | `unavailable` / **0** |
| 8（L177） | `protocol_silent_frac_band` | **L17** | `unavailable` / **0** |
| 9（L178） | `protocol_drive_elasticity_min` | **L18** | `unavailable` / **0** |
| 10（L179） | `target_p5b_ratio_window` | **L19** | `unavailable` / **0** |
| 11（L180） | `protocol_p6a_rho_min` | **L20** | `unavailable` / **0** |
| 12（L181） | `gate_align_mech_indicator_band` | **L93** | `unavailable` / **0** |
| 13（L182） | `protocol_acc_min` | **L22** | `unavailable` / **0** |
| 14（L183） | `protocol_p9_consolidation_drop_min` | **L123** | `unavailable` / **0** |
| 15（L184） | `protocol_pharma_dose_caliber` | **L124** | `unavailable` / **0** |
| 16（L185） | **（无专属 key）** | — | — |
| 17（L186） | `gate_points_per_region_ladder` | **L102** | `500\|1000\|2000\|5000` / **0** |

**🔴 与派发描述的两处偏差（如实报告，不合并、不凑数）**

**偏差 A（形态，不影响计数）**：派发件写「17 个常数…**没有进入**正式的口径登记表 `m12_targets.csv`」；
**实测：16/17 项在该表中已有 `final=0` 占位行**（上表逐行给出物理行号；表共 124 记录 / 26 `final=0`，与闸门实测一致）。
⇒ 准确表述应为「**17 项的『值』未登记**」而非「17 项未进表」。
**证据**：`PYTHONHASHSEED=0 python3` 读表 ⇒ `records=124 / final=1:98 / final=0:26`，26 行 `final=0` 名单中含上表 16 个 key（§7 命令 3）；
骨架自己也这么写（**L422**：「它们**全部已在本表落成 `final=0` 行**」）。

**偏差 B（身份，1 项）**：第 16 项（P11 锚级 `n_min`）**在表内没有专属行**。
最接近的两行都不是它：`target_p11_anchor_set`（**L26**，值 `partial`）是**锚集状态**；`gate_family_p11_perturbation_size`（**L74**，值 `N_anchor`）是**family 大小**。
**证据**：全表 124 key 的精确匹配 ⇒ `含 'T_' 的 key` 仅 `protocol_T_main_s`；`n_min` 仅命中 P8 的 `ref_p8_fit_set_vs_n_min`（L29）与 `gate_n_min_*` 四行（L62–L65）（§7 命令 4）。
⇒ **第 16 项是 17 项里唯一"连行都没有"的** ⇒ 它是本清单唯一的 **`append`** 类提案。

**偏差 C（引用，1 处，纯文档层）**：骨架 §5.2 第 16 项写「卡在哪里」=「**L218** 写作『**如** `n_min=50`』」。
**实测：`docs/生物仿真M12实施清单.md` 的 L218 不是这个内容**（L218 = 「> **路径判定（预注册）**：G2-M ∧ G0-M ∧ G1-M ∧ C-M 为前置门链…」）；
`grep -n "n_min=50"` 在该清单**零命中**。
**真正的 L218 在另一个文件**：`docs/m12_b1_gate_closure.md` **L218** 逐字 =「**F3/F4 的门**：F4 分母下限按锚数预注册（如 `n_min=50` ⇒ 目标 ≥35 命中才可能 CI95 下界 ≥0.70）；」（§7 命令 5）。
⇒ **结论：引用本身成立，但缺文件名** ⇒ 接手者若照字面去 M12 清单找 L218 会**找不到**（这是"引用不完整"，不是"数据错误"）。

### 2.2 主表：17 项的身份、位置、值、作用、量纲、依据状态

**依据状态的三态判据（本清单的定义，先声明再使用）**：
- **① 有文献/数据依据** = 存在**就 M12 口径而言**可引用的文献（PMID/DOI）或**实测数据**；
- **② 仅工程约定** = 值存在且有出处链，但出处是**内部教训/实现默认值/预注册草案**，**无文献值**（其登记件自述"设计选择，非文献值"者直接归此类）；
- **③ 查无依据** = **🔴 全库无值**，或**只有"不同口径/不同量"的值**（后者不得当作 M12 的值 ⇒ 与"合并近似量"严格区分）。

| # | 常数（骨架 §5.2 写法） | 表内现值/`final` | **值的硬编码/落盘位置（实读 `file:line`）** | 作用（参与哪个判据） | 量纲 | **依据状态** |
|---|---|---|---|---|---|---|
| **1** | 协议分析窗 **T** | `30` / 0 | **清单（唯一数值出处）**：`docs/生物仿真M12实施清单.md` **L712**（D5 降档式「T=30 s → 10 s」左值）、**L408**、**L848**（§5.1 列定义注释里的命名例）；**代码实跑值 = 10 s**：`tools/make_m11_p5_plan_l3.py:105`（`T_MS = 10000.0`）、`tools/make_m11_p5_plan_l0.py:66`、`tools/make_m11_p5_plan_g1.py:37–38`（`T_MIN = 10_000.0` / `TS=(10k,20k,40k)`）与 **:147**（`ALLOWED_T_MS`）、`tools/run_m11_p5_arms.py:1454`（`--T-ms default=10000.0`）；**M11 注册行**：`data/m11_p5_targets.csv:16`（`protocol_T_min_ms=10000`） | **P5a/P5c**：率分辨率 `1/T`、稳态性与"短窗不得判长时标" | `s` | **②**（工程约定：预算表与降档菜单反推；无 PMID/DOI）**＋🔴 3× 冲突**（表内 30 s vs 代码 10 s；M11 交接红线 `docs/m11_g1_resume_v2.md:84`「**只用协议时窗 `T=10 s`** 判率/比值/阈值」） |
| **2** | **settle** 时长 | `unavailable` / 0 | **M11 注册行（同名）**：`data/m11_p5_targets.csv:15`（`500` ms、`final=1`）；**代码**：`tools/make_m11_p5_plan_g1.py:42`、`tools/make_m11_p5_plan_l0.py:67`、`tools/make_m11_p5_plan_l3.py:106`（三处 `SETTLE_MS = 500.0`）、`tools/run_m11_p5_arms.py:1455`（`--settle-ms default=500.0`）；**落盘计划**：`data/m11_p5_plan_g1.json:5`；**M10 溯源行**：`data/m10_calibration_targets.csv:12`（`settle_window_ms,500`） | **G0-M/P3、G1-M/P5a**：t=0 初始化瞬态是否入统计 | `ms` | **②**（出处链 = M11 清单 §4.5 协议行 → M10 §4.2 预注册（M5 L37#2 教训）；**无文献**）。⚠️ 同库另有 300 ms（`tools/reeval_m10_m9_rebuttals.py:45` 等）与 1000 ms（`tools/validate_p9_resting.py:60`） |
| **3** | **bin 宽** | `unavailable` / 0 | **同名 key 在 M11 不存在**，但**同义值遍布**：`tools/make_m11_p5_plan_g1.py:120`（`bin_ms=100.0, smooth_frames=5`）、`tools/make_m11_p5_plan_l0.py:173` 与 `:265`、`tools/make_m11_p5_plan_l3.py:728`、`tools/run_m11_p5_arms.py:1122`（`.get("bin_ms", 100.0)`）与 **:1427**、`tools/validate_p11_resting.py:524` 与 **:66**、`data/m11_p5_plan_g1.json:9`、`data/m11_p5_targets.csv:26`（`bin_ms_binwise=100`）、`tools/validate_p9_resting.py:62`（`BIN_MS = 100.0`）、实现默认 `src/larval_zf_circuit.py:124`（`bout_stats(bin_ms=100.0)`） | **P5a/P10a/P10c**：区域率分布与任务依赖动力学（**对 bin 宽直接敏感**） | `ms` | **②**（工程约定：同族引擎的率估计器口径；无文献）。⚠️ M12 的 scope 是**区域级**分布而 M11 是 **`estimator="per_neuron"`** ⇒ 需一条口径对齐声明 |
| **4** | **平滑窗（帧数）** | `unavailable` / 0 | 同上族：`tools/build_m10_imaging_reference.py:329`（`k_win = 5`）、`tools/build_m10_imaging_dwell_reference.py:64`（`SMOOTH_BINS_REF = 5`）、`tools/validate_p10_state_structure.py:36`、`tools/verify_m11_microns_series.py:69`、`tools/verify_m11_same_source_band.py:46`、`tools/run_m11_p5_arms.py:1123` 与 **:1427**、`data/m11_p5_plan_g1.json:10`、`data/m11_p5_targets.csv:27`（`smooth_frames_state=5`、`final=1`） | **P10a/P10c**：率的时间分辨率与量纲 | `frames` | **②**（工程约定：M10 成像冻结管线「0.914 s 箱 → 5 帧平滑 → z → AR(1)」，`docs/生物仿真M11实施清单.md:584`；无文献）。⚠️ 该平滑窗**本身已被登记为口径风险**（`docs/m10_acceptance_master.md:243`：慢时标"不能宣称独立于 5 帧平滑"） |
| **5** | **驱动率** | `unavailable` / 0 | **M11 注册行（同名）**：`data/m11_p5_targets.csv:119`（`8.0` Hz/神经元、`final=1`，note 自述「**由 R115 §⑤ 补登**」）；**代码 6 处**：`tools/make_m11_p5_plan_g1.py:41`、`tools/make_m11_p5_plan_l0.py:69`、`tools/make_m11_p5_plan_l3.py:108`（`DRIVE_RATE_HZ = 8.0`）、`tools/run_m11_p5_arms.py:737`/`1126`/`1170`/`1458`；`data/m11_p5_plan_g1.json:7` | **P5b 驱动弹性 / P5a 率带**（M11 的 `(a)` 判定直接依赖它） | `Hz`（per neuron） | **②**（工程约定：出处 = plan JSON 既存值；**无文献**；且 `docs/m11_b3_g1_design.md:146` 明文**禁止**把它当标定杠杆） |
| **6** | **驱动剂量网格** | `unavailable` / 0 | **M11 注册行**：`data/m11_p5_targets.csv:19`（`protocol_drive_mult_grid = 0\|0.5\|1\|2\|4\|8`、`final=1`）；**代码实为 5 档**：`tools/make_m11_p5_plan_g1.py:39`（`DOSES = (0.0, 0.5, 1.0, 2.0, 4.0)`）、`tools/price_m11_p5_l1.py:32`；**CLI 默认仅 3 档**：`tools/run_m11_p5_arms.py:1456`（`--doses default="0,1,4"`）；M10 另有相异网格 `tools/audit_m10_point_defect.py:35`（`LAM_GRID=(0.0,1.0,4.0,8.0)`） | **P5b**：弹性的剂量点与"单调"的判定层 | —（档位表） | **②**（工程约定）**＋🔴 三方冲突**（表 6 档 / 代码 5 档 / CLI 3 档）；且 M11 注册行的 provenance（清单 §4.5(d)）**经查不含该网格**（§2.4-②） |
| **7** | 自持比**上限** | `unavailable` / 0 | **M11 注册行**：`data/m11_p5_targets.csv:46`（`selfsustain_ratio_max=0.10`、`final=1`，note 逐字「**该值是设计选择（草案），不是文献值**」）；**代码/落盘**：`tools/report_m11_l3.py:50`（`R_MAX = 0.10`）、`tools/build_m10_reference_bands.py:78`、`data/m10_calibration_targets.csv:8`（`selfsustain_ratio_max,0.10`）与同表 `:18`/`:19`（`p5_selfsustain_ratio_max,0.10`、`p5_selfsustain_ci_upper_max,0.12`） | **P5b**（G1 门条件之一） | 无量纲比 | **②**（其登记件**自述非文献值**） |
| **8** | 静默比例**带** | `unavailable` / 0 | **尖峰/电生理域：无值**——`data/m11_p5_targets.csv:36/37`（`silent_frac_lo/hi` = `unavailable`/`final=0`）、`:38/39`（`silent_frac_ephys_lo/hi` = `unavailable`/`final=0`）；**钙域：有值但口径不同**——`data/m11_p5_targets.csv:115/116`（`0.101852`/`0.229885`、`measured`、`final=1`，由 B-5/R169 原位填入）；**钙域文献锚**：`:40`（`silent_frac_ca_active_per_wave_pct = 12 %`、`literature`、**PMID 19706480 / DOI 10.1073/pnas.0907660106**，摘要原文 "reaching 12% of active neurons per wave"）、`:42`（由 12% 反推 `0.88`） | **P5b**（"静默落带，带运行前定稿"） | 无量纲比 | **🔴③ 查无依据（且 M12 行未声明口径域）**：M12 的 `protocol_silent_frac_band` **没有写它是 spike 域还是钙域**；而 M11 的钙域带已被 R183/R184 判为**跨口径不可比**（`docs/m11_g1_resume_v2.md:13/L29`，并引表内 `calcium_vs_spike_caliber_gap` 为禁令）⇒ **不得抄 `0.101852/0.229885`**。**最近邻的 ① 级证据（钙域 12% 活跃）存在，但不能跨域使用** |
| **9** | 驱动弹性**下限** | `unavailable` / 0 | **M11 注册行**：`data/m11_p5_targets.csv:53`（`drive_elasticity_min=0.30`、`final=1`，note 逐字「**设计选择，非文献值**」）；**落盘**：`data/m10_calibration_targets.csv:9`（`drive_elasticity_min,0.30`）、`tools/build_m10_reference_bands.py:80`、`tools/build_m10_imaging_reference.py:451` | **P5b**（4× 增幅 ≥ 下限） | 相对增幅 | **②**（其登记件自述非文献值）。⚠️ M12 行 note 已自述「不得直接移植」 |
| **10** | 比值**可判下界** | `unavailable` / 0 | **M11 注册行**：`data/m11_p5_targets.csv:48`（`selfsustain_zero_drive_min_spikes=300`、`final=1`，note：`1/√300 = 5.8%` ⇒ 相对不确定度 ≤6%）；**代码硬编码副本**：`tools/report_m11_l3.py:49`（`WINDOW_LO = 300.0`）、`tools/validate_p11_resting.py:814`（`… or 300`）；**M11 判定实例**：`docs/m11_acceptance_master.md:3811`（`pooled_zero_drive = 77,009 ≥ 300 ⇒ c_judgeable = True`）、同文 `:88` 交接红线「`pooled_zero < 300` ⇒ `unavailable`（**不是 pass**）」 | **P5b**（比值/自持比是否可判） | 计数（脉冲数） | **②**（工程约定；有**定量理由**：Poisson 相对不确定度 1/√n；无文献） |
| **11** | P6a 的 **ρ 下限** | `unavailable` / 0 | **全库无数值**：清单只有口径行 `docs/生物仿真M12实施清单.md:947`（「Spearman **ρ ≥ 下限**（下限由边界可达性自检 + 文献定）」）；`tools/` 内的 `rho`/`spearman` 命中**全是别的量**（`tools/validate_p9_generalization.py:202` 泛化单调 ≥0.9；`tools/validate_p9_resting.py:215` 确定性 ≥0.99；`tools/validate_p3_latency.py:65` M3 强度-潜伏期）；M10 另有 `slow_var_corr_min = 0.30`（`tools/validate_p10_slow_variable.py:362`、`tools/build_m10_reference_bands.py:117`）——**不同量** | **C-M/P6a**（区域间耦合 ↔ 图谱权重） | Spearman ρ | **🔴③ 查无依据** |
| **12** | G0-M (c) 的**"带"** | `unavailable` / 0 | **全库无数值**：清单只有 `docs/生物仿真M12实施清单.md:667`（「调质/慢变量级电导积分与 PSTH 形状一致（**Δ < 带**）」）。**最近邻但是不同量**：M11 的 `double_estimator_rel_tol = 0.02`（`data/m11_p5_targets.csv:22`，note「硬门③：counts 求和 vs pop_trace 求和相对差 ≤2%」；代码 `tools/scan_m11_engine.py:58` `DUAL_TOL = 0.02`）——那是**双估计器一致性**，不是"电导积分 vs PSTH 形状" | **G0-M/P3**（`(c)` 对齐门） | 相对差 | **🔴③ 查无依据**（**明确： 0.02 不得挪用**） |
| **13** | P10b 的 **`acc_min`** | `unavailable` / 0 | **全库无数值**：清单只要求给值并规定后果（`docs/生物仿真M12实施清单.md:370`：「`acc_min` 无 provenance ⇒ **invalid**」；`:953` P10b 行）；**最近邻但不同量**：`tools/validate_p10_altdecoding.py:126–127`（`--margin-min default=0.10`，「cross − drift_floor ≥ 0.10 才判该对比对可分」）——那是**相对裕度**不是绝对准确率下限；`data/m12_reference.csv` 无该行 | **F/P10b**（优越性/信息类） | 正确率 | **🔴③ 查无依据**（且按 F3 的条文，当前应判 `invalid`，**不是** `unavailable`——见 §5-②） |
| **14** | P9 巩固下降的**"带"** | `unavailable` / 0 | **全库无数值**：清单只有 `docs/生物仿真M12实施清单.md:951`（「**replay 关闭 ⇒ 巩固下降 ≥ 带**」）与 `:254`/`:790`/`:987`（三层断言）；文献锚**只有方向**（PMID 8036517 / DOI 10.1126/science.8036517，`docs/生物仿真M12实施清单.md:188`：重放的机制依据）；**代码侧**：`tools/` 与 `src/` 内 `consolidation`/`巩固` **零命中** | **F/P9**（F5 第三层：动态范围） | 无量纲（下降幅度） | **🔴③ 查无依据**（文献只到"方向"级别，`docs/生物仿真M12实施清单.md:969` 明文「`literature` ≠ 参数已核实；本轮表中所有文献的 `value_verified` 均为 **`no`**」） |
| **15** | 药理口径**三要素** | `unavailable` / 0 | **全库无剂量/途径/给药时窗数值**：清单只有模板行 `docs/生物仿真M12实施清单.md:912`（「[药理口径]（若扰动）剂量/途径/给药时窗/效应方向的定义」）与 `:269`；药理锚**只有题录**（`：190`/`:531`；PMID 37644210 / 19416740 / 23073640 / 9054347 / 8036517）；**`data/m12_literature_priors.csv` 不存在**（§7 命令 6）；`data/m11_literature_priors.csv`（39 行）**无药理行** | **F/P11**（能否声明"与文献同口径"） | 剂量/途径/时窗 | **🔴③ 查无依据** |
| **16** | P11 **锚级 `n_min`** | **（无行）** | **唯一数值出处是"示例"**：`docs/m12_b1_gate_closure.md:218`（「如 `n_min=50` ⇒ 目标 ≥35 命中才可能 CI95 下界 ≥0.70」）；M12 清单侧只有 `docs/生物仿真M12实施清单.md:371`（「或 ≥预注册锚级下限，如『有锚 **≥20/50**』」）与 `:528`（「预注册锚 **≤50 篇**」）；M11/M10 清单同类表述见 `docs/生物仿真M11实施清单.md:161/1024/1058`、`docs/生物仿真M10实施清单.md:130/189/382/711/803` | **F/P11**（F4 分母下限） | 计数（锚/命中数） | **🔴③ 查无依据**（**只有示例值，不得当注册值**；骨架 §3.2 B4 同判：用未注册常数作判据 ⇒ 必须落 `unavailable`） |
| **17** | 每区域点数档（**清单自相矛盾**） | `500\|1000\|2000\|5000` / 0 | **两处原文不一致**：`docs/生物仿真M12实施清单.md:655`（E5-M：`{500, 1,000, 2,000, 5,000}`，**无 10000**）**vs** `:689`（§3.4 探针：`{500, 1000, 2000, 5000, 10000}`，**含 10000**）；降档菜单 `:709`（「5,000 → 1,000 → 500」）；**代码使用方 = 无**（`gate_points_per_region_ladder` 在 `.py` 中零引用；`gate_scale_N_ladder` 同） | **G0-M/P3 定档、P12** | 点/区域 | **②**（工程约定）**＋🔴 冲突待裁决**（两候选档都在清单里，代码两档都还没用） |

### 2.3 依据状态统计 ＋ 「查无依据」7 项名单（最关键输出）

| 状态 | 个数 | 名单 |
|---|---|---|
| **① 有文献/数据依据（M12 口径）** | **0** | —— |
| **② 仅工程约定** | **10** | T(#1)、settle(#2)、bin(#3)、平滑窗(#4)、驱动率(#5)、剂量网格(#6)、自持比上限(#7)、弹性下限(#9)、比值下界(#10)、点数档(#17) |
| **🔴③ 查无依据** | **7** | **静默比例带(#8)**、**P6a ρ 下限(#11)**、**G0-M (c) 的带(#12)**、**`acc_min`(#13)**、**P9 巩固下降带(#14)**、**药理口径三要素(#15)**、**P11 锚级 `n_min`(#16)** |

**🔴 「查无依据」7 项的缺口分级（便于排优先级）**

| 级别 | 项 | 缺到什么程度 | 为什么是真缺口（不是"惯例"） |
|---|---|---|---|
| **A：值全库不存在** | #11 ρ 下限、#12 (c) 的带、#13 `acc_min`、#14 巩固下降带、#15 药理三要素、#16 P11 `n_min` | **6 项**：连一个字面数值都没有（#15、#16 只有模板/"如"字示例） | 三态里**没有任何一态**能承接它们：既无文献/数据（不是①）、又无出处链可引（不是②）。**写"惯例/经验"就是无出处断言** ⇒ 违 `docs/生物仿真M12实施清单.md:966–969`（每条判据行必须有 provenance，文献必须给 PMID/DOI+查询式+hitCount+检索日期） |
| **B：只有"不同口径/不同量"的值** | #8 静默比例带 | **1 项**：钙域有 `0.101852/0.229885`（`final=1`，B-5 实测）；**但 M12 行未声明自己是 spike 域还是钙域**，而 M11 已因"钙域带套到 spike 统计量"被 R183/R184 判为不可判 | 若直接填入 ⇒ 重演 R183 的**跨口径错误**；表内已有禁令行 `calcium_vs_spike_caliber_gap`（`data/m11_p5_targets.csv:100`，被 `docs/m11_g1_resume_v2.md:29` 引用） |

**⇒ 结论：17 项里 0 项"依据充分"；能立刻登记的只有 4 项（且其中 2 项还要附一条口径声明义务）。**
这一结论**支持**而非削弱 R150 的判断：B-6 的价值恰恰在于**把"缺登记"与"缺依据"分开**——**前者可立刻清（4 项），后者必须回到数据/文献/自检（13 项）**，
且**不能再用"等上游"把后者一起盖住**（`data/m12_list_review_master.md:129`：缺数据 ⇒ B-5 型；缺登记 ⇒ B-6 型，**不得挂在"等上游"下面**）。

### 2.4 与既有裁决/文档的冲突与漂移（三处，逐条给证据）

**① 🔴 机制冲突：「同 key 追加」与 G0 闸门互斥（最高优先级，必须先裁决）**

| 主张 | 原文出处 |
|---|---|
| 解封动作 = **每个常数新增一行 `final=1`（值 + 口径），不改本行** | `docs/m12_criteria_table_skeleton.md` **L273**（§7.1 组 B） |
| **闸门 G0：重复 key 即报错** | `tools/check_m12_targets_gates.py` **L107–110**（`dup = sorted({k for k in keys if keys.count(k) > 1})` ⇒ `problems.append("重复 key…")`） |
| **本节点实测**：把同 key 的 `final=1` 行追加到副本 ⇒ **G0 FAIL / 退出码 1** | 见 §4.2 实测输出（`ERROR: 重复 key：['protocol_settle_ms']`） |
| 而 M11 的 R117「追加且仅追加一行」**之所以能成立，是因为那条 key 原本不存在**（`data/m11_p5_targets.csv:119` note：「此前 103 行中无任何行登记该常数」） | 同左 |
| **M11 对"占位行"的机制另有明文**：**原位填入**（第四十四条；第八十条四条件：表内已有允许填写的规则、原 `final=0` 的唯一理由是可被独立否证且已被否证的前提、填入值是数据的属性与本次运行无关、执行时做"前态断言+记录数+key 差集+新旧哈希"四项核对） | `docs/m11_acceptance_master.md` **L2449**、**L4913–4931**、**L4943–4946**；落地实例 = `data/m11_p5_targets.csv:115/116`（R169） |
| **但 M12 清单的字面写「全表禁止就地改值」**，唯一例外是"未决冲突的裁决"，且必须在 `status_history` 留痕 | `docs/生物仿真M12实施清单.md` **L970–971** |
| → **而 10 列表里没有 `status_history` 列**（列 = `key,value,value_kind,unit,evidence_level,provenance,note,wrong_usage_consequence,final,scope`），且 U1 已登记该表与清单 §5.1 的 11 列差异 | `data/m12_targets.csv` **L3**（列声明）；骨架 **L66–81**（U1）、**L334** |

**⇒ 三条路都不完全干净，必须由主 agent 裁决其一**：
**(甲)** 认定为"**非修订**"（占位行从未载值，`value=unavailable`）⇒ 依 **M11 第四十四条/第八十条原位填入**（本节点**推荐**，理由：与表内 `rule_final1_frozen_after_run` 的实际措辞一致——该行管的是「**`final=1` 的行**在协议跑臂后不得就地修改」，`data/m12_targets.csv` **L52**）；
**(乙)** 认定为 L971 的"**未决冲突的裁决**"例外，并在 `note` 内留痕（10 列表无 `status_history`，需同时决定留痕载体）；
**(丙)** 给新 key（如 `protocol_settle_ms_registered`）—— 本节点**不推荐**（制造同名双 key，且 G1 的 token 扫描会把两个 key 都当独立量 ⇒ 后续任何引用都可能指错）。

**② 🟡 值↔出处漂移 3 处（不改变"未登记"的结论，但影响谁有权填值）**
- **T**：表内 `30`（来自 D5 降档式左值）vs 代码实跑 `10000.0` ms（§2.2 #1）。⇒ 填值前必须裁决**哪个是 M12 的协议 T**。
- **剂量网格**：M11 注册行 `0|0.5|1|2|4|8`（`data/m11_p5_targets.csv:19`）vs 代码 `DOSES=(0,0.5,1,2,4)`（`tools/make_m11_p5_plan_g1.py:39`）vs CLI `"0,1,4"`（`tools/run_m11_p5_arms.py:1456`）；
  且该注册行的 provenance 指 `docs/生物仿真M11实施清单.md §4.5(d)`，而 §4.5 的 (d) 行（**L862**）**不含该网格**（只有「4× 增幅 ≥ 下限（草案 ≥0.30）」）⇒ **注册值的清单依据查无实据**。
- **点数档**：清单 **L655** vs **L689**（§2.2 #17；A-5 已在骨架 **L338/U5** 登记，未裁决）。

**③ 🟡 文档↔表漂移 2 处（骨架文档声称的行与表内实际不符）**
- **`protocol_T_dwell_ratio_min` 不存在**：骨架 §5.1 表格 **L158** 称其 `final=1`（「唯一有值的 T 类常数」）、§9.1 **L389** 亦登记它；
  **实测：`data/m12_targets.csv` 124 行中无此 key**（`grep "T_dwell\|dwell_ratio"` 零命中；`含 'dwell' 的 key` 仅 `target_p10c_dwell_band`）⇒ 若依 §5.1 自己的判据（「凡决定判据取值的协议常数必须在本表有行」），
  `docs/生物仿真M12实施清单.md:712` 的「**T ≥ 3× 目标 dwell**」（**决定降档是否合法**，直接管 P5a/P5c）**也是未注册常数** ⇒ **真实缺口可能是 18 项而不是 17 项**。**本节点不自行并入 17**，单列待裁决。
- **骨架 §5.1 的行数描述与表不符**：§5.1（**L148**）称「本表据此登记了 **11 行** `protocol_*`，其中 **9 行**是缺口行」；
  **实测：表内 `protocol_*` 共 21 行**（**13 行 `final=0`** / 8 行 `final=1`）（§7 命令 4）。⇒ 文档描述需要一次核对（**不是数值结论错误，但会让接手者按错误的行数去核对**）。
- **`m12_targets.csv` L10 的 provenance 自称**「§2–§5（L415–L986）内出现 T 数值的仅 **L408/L712** 两处」——**实测该区间内 L848 也含「30s 窗」**（§7 命令 5）⇒ **引用不完整**（结论不变）。

---

## §3 每个常数的建议登记方案（17 项）

### 3.1 分类判据（先声明，再用）

- **【现-1】现在就能登记（无需任何前提）**：值唯一、出处可引、**不依赖任何未测量/未裁决的量**。
- **【现-2】现在就能登记（附一条声明义务）**：值是唯一且可引的，但**必须在 `note` 内逐字声明 M12 口径与 M11 口径的关系**（EQ-1）；**该声明由登记人当场写，故不构成外部依赖**；若登记人不写该声明 ⇒ 该项**必须退回 `final=0`**。
- **【裁】只需一次本地裁决即可登记**：值有两个以上候选、或清单自相矛盾 ⇒ **不需新数据、不需 M11**，只需主 agent 一次裁决。
- **【等】欠证据，暂不可登记**：缺文献值/实测/自检 ⇒ 必须回到数据或文献，**且不得用"等上游"概括**（R150 §⑤）。

**🔴 一个必须写明的答案**：**17 项中，直接依赖 M11 G1 的 = 0 项**（这正是 R150 说"不依赖 M11"的实测印证）。
但**这不等于它们都能立刻登记**：**【现-1】+【现-2】= 4 项**立刻可办，**【裁】3 项**一次裁决即可，**【等】10 项**必须补证据。
（**登记常数** ≠ **判据可通过**：即便 17 项全部登记完，P5a/P5b 的**判定**仍受 M11 G1 未解封与 `list_review_master` L108 的"皮层协议不得定稿"约束。）

### 3.2 建议表（17 行；`final=0` 者**必须写明唯一一条未满足前提**）

> 列语义沿 `data/m12_targets.csv`（`key,value,value_kind,unit,evidence_level,provenance,note,wrong_usage_consequence,final,scope`）。
> **所有行沿用原 key**（原位填入，`scope` 保持原值不动）；**唯 #16 为 `append` 新 key**。

| # | key | 建议 `value` | `value_kind` / `unit` | `evidence_level` | 建议 `final` | 类别 | **唯一一条未满足前提**（`final=0` 时必填；不得写两条，不得写"未知"） |
|---|---|---|---|---|---|---|---|
| 1 | `protocol_T_main_s` | **待裁决**：`30`（清单 D5 左值）**或** `10`（代码实跑值） | `point` / `s` | `derived` | **0 → 1（裁决后）** | 【裁】 | 主 agent 对「M12 协议 T = 30 s（沿用 IBL 30 s 窗）还是 10 s（沿用 M11 实跑与红线 `m11_g1_resume_v2.md:84`）」的**一次裁决** |
| 2 | `protocol_settle_ms` | **`500`** | `point` / `ms` | `derived` | **1** | 【现-1】 | ——（已满足）**注**：须在 `note` 内写明"沿用 M10/M11 口径（EQ-1）" |
| 3 | `protocol_bin_wide_ms` | **`100`** | `point` / `ms` | `derived` | **1（附声明义务）** | 【现-2】 | ——，但**前提是** `note` 内逐字声明：区域级率估计器与 M11 `estimator="per_neuron"` 口径的关系（`tools/run_m11_p5_arms.py:1427`） |
| 4 | `protocol_smooth_window_frames` | **`5`** | `point` / `frames` | `derived` | **1（附声明义务）** | 【现-2】 | 同 #3（并须登记"平滑窗本身是口径风险"，`docs/m10_acceptance_master.md:243`） |
| 5 | `protocol_drive_rate_hz` | **`8.0`** | `point` / `Hz` | `derived` | **1** | 【现-1】 | ——（已满足）**注**：`note` 须写"沿用 M11 的 8.0（EQ-1 跨里程碑声明）"，并保留 M11 R110 ④ 禁令 |
| 6 | `protocol_drive_dose_grid` | **待裁决**：`0\|0.5\|1\|2\|4`（代码实跑 5 档）**或** `0\|0.5\|1\|2\|4\|8`（M11 表 6 档） | `list` / —— | `derived` | **0 → 1（裁决后）** | 【裁】 | 主 agent 对「剂量网格取"实际跑过的 5 档"还是"注册的 6 档"」的裁决（并同时决定是否把 `8` 标为"未测档"） |
| 7 | `protocol_self_sustain_ratio_max` | **不建议现在填值**（M11 的 `0.10` 不得直接移植：`data/m12_targets.csv:16` note 已明写） | `point` / —— | `derived` | **0** | 【等】 | 边界可达性自检（`docs/生物仿真M12实施清单.md:382`）在 **M12 自己的基准臂/零驱动臂**上跑通并给出 margin 的那一次结果 |
| 8 | `protocol_silent_frac_band` | **不建议现在填值**；**即便填也必须先声明口径域** | `list`（`lo\|hi`）/ —— | `derived` | **0** | 【等】 | **口径域声明**（spike 域 / 钙域）+ 与该域同口径的数据或文献带（当前钙域带因跨口径被 R183/R184 判不可用） |
| 9 | `protocol_drive_elasticity_min` | **不建议现在填值**（M11 的 `0.30` 不得直接移植） | `point` / —— | `derived` | **0** | 【等】 | 边界可达性自检在 M12 剂量级聚合值（`dose_level_aggregate`）上的实测结果 |
| 10 | `target_p5b_ratio_window` | **不建议现在填值**（M11 的 `300` 是 M11 口径） | `point` / —— | `derived` | **0** | 【等】 | 在 M12 的 `arm=zero_drive` 上完成一次"零驱动臂脉冲数 vs 比值可判性"的自检并给出下界 |
| 11 | `protocol_p6a_rho_min` | 无候选值 | `point` / —— | `unavailable` | **0** | 【等】 | 清单 **L947** 指定的两条定值路径之一被走完：**边界可达性自检 + 文献**（当前两条都未开始） |
| 12 | `gate_align_mech_indicator_band` | 无候选值（**不得用 M11 的 `0.02`**） | `point` / —— | `unavailable` | **0** | 【等】 | M12 引擎（B3）能产出"电导积分 vs PSTH"两路量之后，该带才可定稿（属 M12 自有前置，**不依赖 M11**） |
| 13 | `protocol_acc_min` | 无候选值 | `point` / —— | `unavailable` | **0** | 【等】 | M12 的解码管线与置换臂就绪 ⇒ 才能定 `acc_min`（**在此之前按 F3 判 `invalid`**，见 §5-②） |
| 14 | `protocol_p9_consolidation_drop_min` | 无候选值 | `point` / —— | `unavailable` | **0** | 【等】 | P9 的巩固曲线量在 M12 上可测（B4/B6 就绪）后，由"边界可达性自检 + 文献方向"定带；**文献当前只到方向级**（`value_verified=no`） |
| 15 | `protocol_pharma_dose_caliber` | 无候选值（**逐篇核实**后才可填） | `list`（剂量\|途径\|时窗）/ —— | `unavailable` | **0** | 【等】 | S8 药理锚由"题录级"升级为"参数级"（逐篇给 `value_verified=yes`），当前 3 条锚全是 `no`（`docs/生物仿真M12实施清单.md:969`） |
| 16 | **`gate_p11_anchor_n_min`（新 key）** | **`unavailable`**（占位） | `unavailable` / `个` | `unavailable` | **0** | 【等】+**append** | `data/m12_targets.csv:26` `target_p11_anchor_set` 由 `partial` 定稿（**＝锚集定稿这一件事**；`n_min` 是它的直接函数） |
| 17 | `gate_points_per_region_ladder` | **待裁决**：`500\|1000\|2000\|5000`（E5-M）**或** 含 `10000` 的 5 档（§3.4 探针） | `list` / 点/区域 | `derived` | **0 → 1（裁决后）** | 【裁】 | 主 agent 对清单 **L655 vs L689** 内部冲突的裁决（A-5 已在骨架 **L338/U5** 登记待裁） |

**计数**：【现-1】2 项（#2、#5）＋【现-2】2 项（#3、#4）＝ **现在就能登记 4 项**；【裁】3 项（#1、#6、#17）；【等】10 项（#7–#16）。

### 3.3 「唯一未满足前提」的写法纪律（继承本项目的硬规矩）

1. **恰好一条**：写两条以上即等于**掩盖**（R150 §⑤ 的同族：把可清的清掉、剩下的才是真缺口）；
2. **必须可被独立否证**：前提必须指向**一个可机检的对象**（某表的某字段、某门的某条件、某次自检的某个输出），**不得**写"依据不足/尚无文献/未知"；
3. **不得写"等上游"**：若前提在 M11 侧，必须写成"等 **M11 的哪条**"（本项目 0 项属此列）；
4. **`final=0` 不是 `fail`**：占位行表达的是"**不可判定**"，`docs/生物仿真M12实施清单.md:896` 的四态语义与 **L897** 硬条款①（未测 ≠ 未命中）必须保持。

---

## §4 实施清单（给执行节点 node B 用）

> **每步给出：动作 / 输入 / 输出（绝对路径）/ **Pass 标准（可机械复核）** / **伪证条件（什么实测结果会证明这步做错了）**。**
> **全流程零网络、零 MPS、零既有件修改**（`docs/m11_g1_resume_v2.md:106` 的并行安全约束）。

### 4.0 前置提醒：两条已实测的陷阱（**不要重犯**）

| 陷阱 | 实测输出 | 正确做法 |
|---|---|---|
| **T1：同 key 追加新行** | `=== G0 结构闸门 === FAIL / ERROR: 重复 key：['protocol_settle_ms'] / [RESULT] GATE FAILURE`，**退出码 1** | **原位填入**占位行（或先取得主 agent 对 §2.4-① 的裁决） |
| **T2：正文里出现"半注册 token"** | 我第一版提案正文写了 `gate_n_min_* 族` ⇒ `=== G1 附录缺失检查 === FAIL / ERROR: key=gate_p11_anchor_n_min 的文本里出现判据式 token gate_n_min_ ⇒ 本表无此 key（附录缺失）`，**退出码 1**（`*` 不是 token 的终止符，`gate_n_min_` 本身被当成 key） | 正文中引用注册行时**写全 key**（如 `gate_n_min_coef_power`），**不要用 `xxx_*` 通配写法**；提交前用 `tools/check_m12_targets_gates.py` 实测 |

### 4.1 步骤

**B6-0 开工前硬约束核对（只读）**
- 动作：记录注册表 sha256/行数；跑闸门；确认 `git status` 对该文件为空。
- 输入：`neural_exploration/data/m12_targets.csv`、`neural_exploration/tools/check_m12_targets_gates.py`。
- 输出：无新文件（把实测贴进本次交付报告）。
- **Pass**：`shasum -a 256 data/m12_targets.csv` = `182847d8c2bbf661e5a43e19e9935c9a17154d9a7c66b15047e82984b86ccd8b`（登记值见 `docs/m12_criteria_table_skeleton.md:517`）**且** `python3 neural_exploration/tools/check_m12_targets_gates.py --quiet` 退出码 = **0**、打印 `records=124 final=1:98 final=0:26` **且** `git status --porcelain <该文件>` 为空。
- **伪证**：sha 不等于登记值（⇒ 表已被改，**立即停止并报主 agent**）；或 `final=0` ≠ 26；或闸门退出码非 0。

**B6-1 裁决与口径确认（不写盘）**
- 动作：把 §2.4-①（填入机制）、§2.4-③（第 18 项候选）、以及 §3.2 的三个【裁】项（T / 剂量网格 / 点数档）整理成**一页裁决请求**报主 agent；**未获裁决前不得执行 B6-2/B6-3 之外的填值**。
- 输入：本清单 §2.4/§3.2；`docs/m11_acceptance_master.md:2449`（第四十四条）、`:4943–4946`（第八十条）；`docs/生物仿真M12实施清单.md:970–972`。
- 输出：`neural_exploration/docs/m12_b6_adjudication_request.md`（新建；**由 node B 写**）。
- **Pass**：文档内**每个**裁决项都给出（i）候选值 ×（ii）`file:line` 出处 ×（iii）"若裁决 A 则填 X、若裁决 B 则填 Y"的可执行分支；且**不含任何未给行号的断言**。
- **伪证**：出现"建议取 X"却没有出处行号；或把【等】类的 10 项也塞进裁决请求（那会把"缺证据"伪装成"缺裁决"）。

**B6-2 填 `protocol_settle_ms`（原位填入，M11 第四十四条/第八十条四条件）**
- 动作：对 **`data/m12_targets.csv` L11** 做**前态断言**（`value=unavailable`、`final=0`，否则**拒改并退出非 0**）⇒ 按提案行 `B6-P1` 写入（`500 / point / ms / derived / final=1`，`note` 含 EQ-1 声明与"非文献值"），`scope` **逐字不动**；随后做四项核对（前态断言 / 记录数 / key 差集 / 新旧 sha256）。
- 输入：本清单提案 CSV `B6-P1` 行；`data/m11_p5_targets.csv:15`；`tools/make_m11_p5_plan_l3.py:106`。
- 输出：`neural_exploration/data/m12_targets.csv`（**唯一被修改的既有件**；改动前必须已取得 B6-1 的裁决）；核对记录写入 `neural_exploration/docs/m12_b6_edit_ledger.md`（新建）。
- **Pass**（**可写成一条断言或一条命令**）：
  `python3 -c "…读 L11 的 10 字段…assert 与提案 CSV B6-P1 逐字相等"` **且** 记录数 **124 → 124 不变** **且** `set(keys) 差集 == set()` **且** `check_m12_targets_gates.py --quiet` **退出码 0 且 `final=0` 由 26 变 25**。
- **伪证**：记录数变 125（⇒ 追加而非原位填入，且会触发 T1）；key 差集非空（⇒ 有行被删/新增）；`final=0` 计数没变（⇒ 没写进去）；闸门 G0 报重复 key。

**B6-3 填 `protocol_drive_rate_hz`**
- 动作/输入/输出/Pass/伪证：同 B6-2，对象换成 **L14**、提案行 **`B6-P2`**（`8.0 Hz / point / derived / final=1`），`note` 必须保留"**不得提高该值去命中判据**"的禁令（依据 `docs/m11_b3_g1_design.md:146`）。
- **Pass 增量**：`final=0` 由 25 → **24**；且该行 `value` 与 `data/m11_p5_targets.csv:119` 的 `8.0` **逐字节相同**。
- **伪证**：写入的值写成 `16`（M11 文档 `m11_acceptance_master.md:3280` 里的对照推算值）或其他非 8.0 的值 ⇒ 那是"调参迁就结论"的形态，**该轮作废**。

**B6-4 追加新 key `gate_p11_anchor_n_min`（`append`）**
- 动作：把提案行 `B6-P3` **追加**到注册表末尾（不得连表头追加；不得插到中间）。
- 输入：提案 CSV `B6-P3`；`docs/m12_b1_gate_closure.md:218`；`docs/生物仿真M12实施清单.md:371`、`:528`。
- 输出：`neural_exploration/data/m12_targets.csv`（追加 1 行）；`docs/m12_b6_edit_ledger.md`。
- **Pass**：记录数 **124 → 125**；`key=gate_p11_anchor_n_min` **恰出现 1 次**；该行 `final=0` 且 `note` 内**恰好一条**未满足前提（指向 `target_p11_anchor_set`）；闸门 **退出码 0**、`unique_keys == 125`。
- **伪证**：该 key 出现 0 次或 ≥2 次；`final=1`（⇒ 用了示例值 `50`，违 `docs/生物仿真M12实施清单.md:969` 与骨架 §3.2 B4）；闸门 G1 报"未注册 token"。

**B6-5 三个【裁】项的登记（裁决后执行）**
- 动作：按 B6-1 的裁决结果，对 #1（T）、#6（剂量网格）、#17（点数档）各自给出**唯一的**填入值或维持 `final=0` 并写明唯一前提。
- **Pass**：三项各自"值 ∈ 候选集合"**且** `note` 内**逐字**记录裁决编号（如"依主 agent Rxxx"）与**未选中的候选值及其出处**（防止"事后改口"）。
- **伪证**：写入的值**不在**候选集合内（⇒ 凭空造值）；或 `note` 未记录未选候选（⇒ 无法复核是否事后调整）。

**B6-6 为 10 项【等】各写**一条**"唯一未满足前提"**
- 动作：不改 `value`/`final`，只在 `note` 内补写那一句前提（若采用原位修改的限制使 `note` 不可改，则把 10 条前提**集中登记**到 `docs/m12_b6_unregistered_constants_checklist.md` 的附录或独立台账，并在台账内引用行号）。
- 输出：`neural_exploration/docs/m12_b6_edit_ledger.md`（或本文档的追加节）。
- **Pass**：10 项**每项恰好 1 条**前提；每条前提都指向一个可在盘上检出的对象（某表字段 / 某门条件 / 某次自检输出）；**0 条**写成"未知/依据不足/等上游"。
- **伪证**：出现 ≥2 条前提的项；或出现不可否证的前提措辞；或把某【等】项直接写成 `final=1`。

**B6-7 收尾核对（四项核对 + 闸门 + 哈希）**
- 动作：按 M11 纪律第六十一条（`docs/m11_acceptance_master.md` §3dh L3347–3354，表内亦登记为 `rule_append_only_no_header`）做"**记录数 + 唯一性 + 与上版差集 + 前缀逐字节**"四项核对；跑闸门正/负向；给出新旧 sha256 与 `wc -l -c`。
- **Pass**：① 闸门退出码 **0**；② **负向测试**退出码 **1**（`--negative-test drop-provenance`、`break-quad`、`dangling-ref` 三者各返回 1 ⇒ 守卫仍有判别性，`docs/m12_negative_control_scaffold.md:56` NC-19）；③ 与改动前的**前缀逐字节**一致（除被填/被追加的行）；④ 台账内四项核对**逐项有实测值**。
- **伪证**：负向测试返回 0（⇒ 闸门失去判别性）；或前缀出现非预期差异（⇒ 有其他行被静默改写）；或只做了"计数"一项核对（M11 学费：计数曾骗过 105≠104 的定位，**差集才是定位手段**）。

### 4.2 本节点已预跑的机械验证（node B 可直接复用，**不是提议而是实测**）

**验证 1：基线闸门（未改任何件）**
```
$ PYTHONHASHSEED=0 MPLBACKEND=Agg python3 neural_exploration/tools/check_m12_targets_gates.py --quiet
[TABLE] .../data/m12_targets.csv  records=124  final=1:98  final=0:26
[RESULT] ALL GATES PASS      # 退出码 0
```

**验证 2：同 key 追加 ⇒ G0 拒绝（陷阱 T1；在 `/tmp` 副本上做，**未动仓库内注册表**）**
```
$ python3 …  # 在 /tmp/b6_probe/append_dup.csv 追加一行 key=protocol_settle_ms, final=1
$ python3 neural_exploration/tools/check_m12_targets_gates.py --table /tmp/b6_probe/append_dup.csv
=== G0 结构闸门 === FAIL
    ERROR: 重复 key：['protocol_settle_ms']
[RESULT] GATE FAILURE        # 退出码 1
```

**验证 3：把本清单的 3 行提案**（2 填 ＋ 1 追加）**灌进 `/tmp` 候选副本 ⇒ 全闸门通过（= 目标态可达）**
```
$ python3 neural_exploration/tools/check_m12_targets_gates.py --table /tmp/b6_probe/filled_candidate.csv --quiet
[TABLE] /tmp/b6_probe/filled_candidate.csv  records=125  final=1:100  final=0:25
[RESULT] ALL GATES PASS      # 退出码 0
（G0: unique_keys=125 / G1: registered_keys=125 / G2: rows_with_quad=125）
```
⇒ **这给出 node B 一个可机械复核的目标态**：`records=125`、`final=1:100`、`final=0:25`、退出码 0。
**注**：验证 3 的第一版因陷阱 T2（正文写了 `gate_n_min_*`）被判 G1 FAIL；改正后才通过 ⇒ **T2 是真实发生过的失败，不是设想**。

---

## §5 红线与风险（违反即本次交付作废）

1. **🔴 不得为凑结论修改已登记 sha 的文件。**
   `data/m12_targets.csv` 的 sha256 已**逐字登记在三处**：`docs/m12_criteria_table_skeleton.md:517` 与 `:546`（`182847d8c2bbf661e5a43e19e9935c9a17154d9a7c66b15047e82984b86ccd8b`）、`docs/m11_acceptance_master.md:4393`、`docs/m11_g1_resume_v2.md:107`。**改动即失效**（L107 对该条的原文：「**绝不修改 `data/m12_targets.csv` 本体**…改动即失效」）。
   **本节点的执行**：只新建 `docs/m12_b6_unregistered_constants_checklist.md` 与 `data/m12_b6_unregistered_constants/proposed_m12_criteria_rows.csv`；所有灌表实验都在 `/tmp` 副本上做；`git status --porcelain` 对 6 个已登记件**全部为空**（§7 命令 1）。
2. **🔴 不得把「未测」写成「未命中」；也不得把「已测」写成「可判定」。**
   四态 = `pass` / `fail` / `invalid` / `unavailable`（`docs/生物仿真M12实施清单.md:892–900`，表内 `gate_four_state_enum`）；
   两条明文：**硬条款①「未测 ≠ 未命中」**（同清单 L897）与 **第四十四条「值测到了 ≠ 条件满足了 ⇒ 已测 ≠ 可判定」**（`docs/m11_acceptance_master.md:2449`）。
   **具体落到 B-6**：
   - `protocol_acc_min` 未登记的**当前后果按 F3 是 `invalid`**（清单 **L370** 明文「`acc_min` 无 provenance ⇒ `invalid`」），**不是** `unavailable`、**更不是** `fail` ⇒ 报判据状态时**不得**一律写"unavailable"；
   - `protocol_p9_consolidation_drop_min` 未登记 ⇒ P9 的 F5 **第三层**（动态范围）不可判，但**前两层（接线/replay 计数 >0、作用域）仍可判** ⇒ **不得**把整条 P9 写成"未通过"；
   - 静默带/弹性/自持比的下限缺登记 ⇒ 相关子判据 `unavailable`，**但其"网络死了比自持强更容易 pass"的退化风险必须单独登记**（`data/m12_targets.csv:19` 的 `wrong_usage_consequence` 已写明）。
3. **🔴 不得事后调参迁就结论（本项目已有两次学费）。**
   - 判据带定稿时间必须**早于**协议运行时间（清单 **L971–972**、**L1081**；表内 `rule_final1_frozen_after_run` **L52**）；
   - 填占位行的**四条件**（第八十条）：其中最关键的是 **(c) 填入值是"数据的属性"，与本次运行的产出无关**；**缺 (c) 时它就是调带**（`docs/m11_acceptance_master.md:4946`）；
   - **口径桥类工作**（若 M12 也要建 spike→钙 之类正演）必须**在看结果之前冻结全部自由参数并预注册**，且**凡"为了让判据落进带"而试出来的参数一律不得用于判定**（`docs/m11_g1_resume_v2.md:142`，新纪律第八十四条的精神）。
4. **🔴 不得把"不同量"合并成一个数（本清单已挡下 4 处危险近似）。**
   | 危险近似 | 正确关系 |
   |---|---|
   | M11 `double_estimator_rel_tol = 0.02`（`data/m11_p5_targets.csv:22`）⇒ 当 M12 (c) 的带 | **不同量**（双估计器一致性 ≠ 电导积分 vs PSTH 形状）⇒ 不得挪用 |
   | M10 `slow_var_corr_min = 0.30`（`tools/validate_p10_slow_variable.py:362`）⇒ 当 P6a 的 ρ 下限 | **不同量**（慢变量 vs 群率相关 ≠ 区域对 ↔ 图谱权重 ρ） |
   | `tools/validate_p10_altdecoding.py:126` 的 `--margin-min 0.10` ⇒ 当 `acc_min` | **不同量**（相对裕度 ≠ 绝对准确率下限） |
   | M11 钙域静默带 `[0.101852, 0.229885]`（`data/m11_p5_targets.csv:115/116`）⇒ 当 M12 的静默带 | **不同口径域**（R183/R184 已判跨口径不可比；表内 `calcium_vs_spike_caliber_gap` **L100** 是禁令行） |
5. **🔴 不得用同 key 追加"填"占位行**（陷阱 T1，已实测退出码 1）；也**不得**用 `key` 后缀造"孪生 key"（会让 G1 的 token 扫描指向两个量，见 §2.4-①(丙)）。
6. **🔴 不得制造"过期同 sha 副本"**（B-5 节点自己删过两件同 sha 副本，理由"过期副本即隐患"，见 `docs/m12_b5_roi_mask.md:4909`）⇒ 本项目内**不复制**注册表；实验一律在 `/tmp`。
7. **🟡 正文里的 token 写法有陷阱**（陷阱 T2）：`gate_n_min_*` 会被当成"未注册的 key"而报 G1 失败（**实测**）⇒ 引用注册行一律写全 key。
8. **🟡 EQ-2 的适用边界必须显式裁决**：EQ-2 的原文禁的是「把 M9/M10/M11（及本清单内 M11 目标带）的任何**绝对速度/相关/dwell/τ 作 M12 判据带**」（清单 **L271**）。
   本清单把 #2/#3/#4/#5（口径常数）判为"可沿用（EQ-1：口径一致才可比）"，把 #7–#10（**阈值/带**）判为"**不得移植**"。**该切分需主 agent 确认**（它是 B6-P1/P2 生效的前提）。
9. **🟡 文献锚"题录级 ≠ 参数级"**：清单 **L969** 明文「参数值维度用 `value_verified=yes/no` 单列；本轮表中所有文献的 `value_verified` 均为 **`no`**」⇒ 任何"照文献填带"的动作**在 `value_verified=no` 时都不成立**（#14/#15 的直接卡点）。
10. **🟡 未测侧不得并入"已排除"**：杠杆扫描必须双侧并显式登记 `unmeasured`（表内 `rule_lever_scan_both_sides` **L49**；M11 学费：一个错误的"结构性对立"活了 4 条裁决）⇒ #7/#9 的带若只由单侧数据定出，**必须同时登记未测侧**。

---

## §6 未确认项（逐条；每条指出卡在哪，并说明本节点为何不自行裁决）

| # | 未确认项 | 卡在哪（`file:line`） | 本节点为何不裁决 |
|---|---|---|---|
| **U1** | 占位行的填入机制：**原位填**（M11 第四十四条/第八十条）**vs** 清单 L970–971 的「全表禁止就地改值」 | `docs/m12_criteria_table_skeleton.md:273`（组 B 的"新增一行"）↔ `tools/check_m12_targets_gates.py:107–110`（重复 key 即 FAIL，**已实测**）↔ `docs/生物仿真M12实施清单.md:970–971` ↔ `docs/m11_acceptance_master.md:2449/4943–4946` | 两条纪律分别来自 M11 与 M12 两套裁决，**本节点不是裁决者**；且它决定 B6-2/B6-3 能否执行 |
| **U2** | 留痕载体：清单 L971 要求"必须在 `status_history` 留痕"，**但 10 列表无该列** | `docs/生物仿真M12实施清单.md:971`；`data/m12_targets.csv:3`（列声明）；骨架 **L334/U1**（列结构差异） | 属表结构变更 ⇒ 需主 agent 裁决（与 U1 同批） |
| **U3** | **第 18 项候选**：`protocol_T_dwell_ratio_min`（清单 L712「T ≥ 3× 目标 dwell」）**无行** | 骨架 **L158/L389** 声称其存在且 `final=1`；实测表中无此 key（§7 命令 4） | 本节点**不把它并入 17**（不凑数）；是否补登记属主 agent 裁决 |
| **U4** | 骨架 §5.1 的"11 行 `protocol_*` / 9 行缺口"与实测"21 行 / 13 行缺口"不符 | 骨架 **L148** vs 实测（§7 命令 4） | 文档描述层差异，**不改文档**（骨架非本节点所有件；其 sha 未被登记，但仍是他人交付物） |
| **U5** | `m12_targets.csv:10` 的 provenance 称"仅 L408/L712"（漏 L848） | 同行 vs `docs/生物仿真M12实施清单.md:848`（**同区间 L415–L986 内**） | 属引用完整性 ⇒ 报请更正（不得由本节点改他人行） |
| **U6** | `m12_targets.csv:13` 的 note 称「**M11 表亦无该行** ⇒ 两表同缺」 | 实测 M11 表有**同义行** `smooth_frames_state = 5`（`data/m11_p5_targets.csv:27`，`final=1`），且 M11 的 P5 率估计器**也用** `smooth_frames=5`（`tools/run_m11_p5_arms.py:1427`、`tools/make_m11_p5_plan_g1.py:120`） | 措辞过强（就 key 名而言成立、就**量**而言不成立）；**须裁决是否把 M11 的 5 帧视为同量**，因为它直接决定 #4 是【现-2】还是【等】 |
| **U7** | `protocol_T_main_s` 的表内值 `30` 与代码实跑 `10000.0` ms 谁优先 | `data/m12_targets.csv:10` vs `tools/make_m11_p5_plan_l3.py:105`、`tools/run_m11_p5_arms.py:1454`、`docs/m11_g1_resume_v2.md:84`（M11 红线"只用 T=10 s"） | 属协议口径裁决；**3× 差异会直接改变率分辨率与稳态性结论** |
| **U8** | 剂量网格的三方冲突（表 6 档 / 代码 5 档 / CLI 3 档） | `data/m11_p5_targets.csv:19`、`tools/make_m11_p5_plan_g1.py:39`、`tools/run_m11_p5_arms.py:1456` | 同上；且 M11 注册值的清单出处（§4.5(d) L862）**不含该网格** ⇒ 需一并裁决 |
| **U9** | 本清单把 #2/#3/#4/#5 判为"可沿用 M11 口径（EQ-1）"是否成立 | 清单 **L271**（EQ-2 原文）；骨架 **L273**（组 B 动作） | 这是**对既有条文的解释** ⇒ 需主 agent 确认（§5-8） |

---

## §7 本节点实际执行过的命令与实测输出（可审计；无 `2>/dev/null`）

```bash
# 0) 工作目录与硬约束（只读）
cd /Users/weidong/ai/small_world
pwd                                        # /Users/weidong/ai/small_world

# 1) 确认 6 个已登记件相对 HEAD 未被修改（输出应为空）
git status --porcelain neural_exploration/data/m12_targets.csv \
    neural_exploration/docs/m12_list_review_master.md \
    neural_exploration/docs/m12_criteria_table_skeleton.md \
    neural_exploration/docs/m12_negative_control_scaffold.md \
    neural_exploration/tools/check_m12_targets_gates.py \
    neural_exploration/docs/生物仿真M12实施清单.md          # ⇒ 无输出（全部未改）

# 2) 注册表身份（与登记值比对）
shasum -a 256 neural_exploration/data/m12_targets.csv
# ⇒ 182847d8c2bbf661e5a43e19e9935c9a17154d9a7c66b15047e82984b86ccd8b   （== 登记值 182847d8…）
wc -l neural_exploration/data/m12_targets.csv                # ⇒ 133

# 3) 基线闸门
PYTHONHASHSEED=0 MPLBACKEND=Agg python3 neural_exploration/tools/check_m12_targets_gates.py --quiet
# ⇒ [TABLE] … records=124  final=1:98  final=0:26 / [RESULT] ALL GATES PASS（退出码 0）

# 4) 键空间精确核对（26 行 final=0 名单 + protocol_* 行数 + n_min/dwell 分布）
python3 - <<'EOF'
import csv
lines=[l for l in open('neural_exploration/data/m12_targets.csv',encoding='utf-8') if not l.startswith('#')]
rows=list(csv.DictReader(lines)); keys=[r['key'] for r in rows]
print(len(keys))                                  # ⇒ 124
print([r['key'] for r in rows if r['final']=='0']) # ⇒ 26 行（含本清单 16 个目标 key）
print([k for k in keys if k.startswith('protocol_')])            # ⇒ 21 行
print([k for k in keys if 'dwell' in k.lower()])   # ⇒ ['target_p10c_dwell_band']（**无 protocol_T_dwell_ratio_min**）
print([k for k in keys if 'n_min' in k])           # ⇒ 仅 P8 的 ref_p8_fit_set_vs_n_min + gate_n_min_* 四行
EOF

# 5) 30 s 与 `n_min=50` 的引用定位
grep -n "30s 窗\|30 s 窗" neural_exploration/docs/生物仿真M12实施清单.md     # ⇒ L408 / L848（**L848 是 L10 行 provenance 漏掉的那处**）
grep -rn "n_min=50" neural_exploration/docs/*.md
# ⇒ 仅 docs/m12_b1_gate_closure.md:218（+ 骨架自引 L185）⇒ 骨架 §5.2 #16 的「L218」**指的不是 M12 清单**

# 6) 药理/平滑/驱动等值的落盘位置（节选）
sed -n '526,532p' neural_exploration/docs/生物仿真M12实施清单.md   # S8 锚：≤50 篇 / 题录级
ls neural_exploration/data/m12_literature_priors.csv              # ⇒ No such file or directory
grep -n "^protocol_settle_ms,\|^selfsustain_ratio_max,\|^drive_elasticity_min,\|^selfsustain_zero_drive_min_spikes,\|^silent_frac_calcium_band_lo,\|^silent_frac_calcium_band_hi,\|^protocol_drive_rate_hz," neural_exploration/data/m11_p5_targets.csv
# ⇒ 15 / 46 / 53 / 48 / 115 / 116 / 119（值依次 500 / 0.10 / 0.30 / 300 / 0.101852 / 0.229885 / 8.0）

# 7) 两条陷阱的实测（在 /tmp 副本上，未动仓库内注册表）
python3 …            # 生成 /tmp/b6_probe/append_dup.csv（同 key 追加）与 /tmp/b6_probe/filled_candidate.csv（本清单 3 行提案）
PYTHONHASHSEED=0 MPLBACKEND=Agg python3 neural_exploration/tools/check_m12_targets_gates.py --table /tmp/b6_probe/append_dup.csv
# ⇒ G0 FAIL / ERROR: 重复 key：['protocol_settle_ms'] / 退出码 1
PYTHONHASHSEED=0 MPLBACKEND=Agg python3 neural_exploration/tools/check_m12_targets_gates.py --table /tmp/b6_probe/filled_candidate.csv --quiet
# ⇒ records=125  final=1:100  final=0:25 / [RESULT] ALL GATES PASS（退出码 0）
```

**本文件与提案 CSV 的哈希见 §8。**

---

## §8 附录：产物清单、声明

### 8.1 本节点新建的文件（**两件，均在 git status 中为未跟踪状态**）

| 文件（相对根） | 行数/字节 | sha256 |
|---|---|---|
| `neural_exploration/docs/m12_b6_unregistered_constants_checklist.md`（本文件） | **492 行** / **67030 B**（`wc -l -c` 实测） | **自指哈希**（把全文所有 64 位十六进制串替换为 64 个 `0` 后计算 ⇒ 本单元格可稳定复算）= `17ebaf9957f9dafd495d7f6f7592f3d20520a403fcdee286181640eb0379b92e` |
| `neural_exploration/data/m12_b6_unregistered_constants/proposed_m12_criteria_rows.csv` | **7 物理行**（3 行注释 ＋ 1 行表头 ＋ **3 行提案**） / 5,340 B | `48b80f0d3266c4cfc10d393bcc8ee8974392a1c23cf8c5c2e15211c49a3942b6` |

**提案行的三行内容（一句话）**：`B6-P1` = 填 `data/m12_targets.csv L11`（`protocol_settle_ms = 500`、`final=1`）；`B6-P2` = 填 `L14`（`protocol_drive_rate_hz = 8.0`、`final=1`）；`B6-P3` = **追加**新 key `gate_p11_anchor_n_min`（`unavailable`、`final=0`，唯一前提 = `target_p11_anchor_set` 由 `partial` 定稿）。

### 8.2 本节点**没有**做的事（逐条声明）

1. **未运行任何仿真**；**未触碰 MPS/GPU**（本机另一作业独占 MPS；本节点全程只用 `grep`/`sed`/`wc`/`ls`/`shasum`/`python3` 标准库读文件）；
2. **未联网**（零下载、零探针）；
3. **未修改 `neural_exploration/data/m12_targets.csv`**（sha256 实测仍为登记值 `182847d8c2bbf661e5a43e19e9935c9a17154d9a7c66b15047e82984b86ccd8b`）；
4. **未修改任何已登记 sha 的件**，包括 `docs/m12_list_review_master.md`（裁决史，只读）、`data/m11_p5_targets.csv`、骨架文档、负对照脚手架、闸门工具（`git status --porcelain` 对上述 6 件**输出为空**）；
5. **未 `git add` / 未 `git commit`**（提交由主 agent 负责）；
6. **未制造注册表的同 sha 副本**（§5-6）；两处灌表实验的对象是 `/tmp/b6_probe/` 下的**临时副本**，**不在仓库内**；
7. **未对 17 项中的任何一项自行裁决取值**（含 `protocol_T_main_s` 的 30/10 之争、剂量网格的 5/6 档之争、点数档之争）——**它们全部作为裁决请求列出**（§3.2 的【裁】项与 §6 的 U7/U8/U3）。

### 8.3 待主 agent 裁决的最小集合（一页）

1. **U1/U2**：占位行"原位填入"是否被允许（M11 第四十四条/第八十条 vs 清单 L970–971），以及留痕载体（10 列表无 `status_history`）；
2. **U3**：第 18 项候选 `protocol_T_dwell_ratio_min` 是否补登记；
3. **U7/U8 + 点数档**：T（30 vs 10）、剂量网格（5 档 vs 6 档）、点数档（含/不含 10000）三处冲突；
4. **U9**：EQ-2 是否适用于"口径常数"（决定 `B6-P1`/`B6-P2` 能否登记）；
5. **U6**：M11 的 `smooth_frames_state = 5` 是否与 M12 的 `protocol_smooth_window_frames` 同量（决定 #4 的类别）。
