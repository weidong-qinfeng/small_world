# M11-B3 · 环境 / 口径 / 冲突登记（工作包 = G1/P5 判据·协议·工具骨架）

> **节点**：M11 执行节点 B3 · **日期**：2026-10-11 · **解释器**：`./.venv-m11/bin/python`（3.9.6 / numpy 2.0.2）
> **纪律**：全程 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0`；**零 GPU / 零 MPS**；**未启动任何仿真或长作业**
> **编号**：本文件独立编号 L1 起（与 `m11_env_notes.md`（B1，L1–L27）分属不同工作包，引用时须带文件前缀）

---

## L1 — 环境与复现

| 项 | 值 |
|---|---|
| 命令前缀 | `cd /Users/weidong/ai/small_world` + `PYTHONPATH=/Users/weidong/ai/small_world` + `./.venv-m11/bin/python`（**未用 `sys.executable`**） |
| 依赖 | **仅 numpy**（工具纯 numpy + stdlib；`neural_exploration/__init__.py` 为空 ⇒ 无重型依赖，符合清单 §0.10 L5「判据脚手架 import 不引入重型依赖」） |
| 工具启动耗时 | 直接调用 `python neural_exploration/tools/validate_p11_resting.py` **0.086 s**（实测） |
| 墙钟 / CPU / GPU | 本工作包：**墙钟 ≈2.5 h**（读文档 + 文献核读 + 编码 + 6 次冒烟）；**GPU = 0**；CPU ≈0.2 CPU-h（单线程，远低于 §10.1 的 G1 段 ≤10 CPU-h 上限）；**下载 0 B** |
| 冻结件 | **零修改**（见 L10） |

---

## L2 — 口径差①：自持比 CI 的 z 分位（本工具 vs M10）

- M10 的记录（`data/m10_p5_gate1_ci.json`）用 **z = 1.96**；本工具用**精确分位 z₀.₉₇₅ = 1.959963984540054**。
- 同一输入下的实测对照（冒烟 S2）：

| 量 | 工具 | M10 记录 | 一致性 |
|---|---|---|---|
| `ratio_pooled` | 0.1015384615 | 0.1015384615 | **逐位一致** |
| `se_log` | 0.0365403631 | 0.0365403631 | **逐位一致** |
| `per_seed_ratio` | 0.0987078248384781 / 0.10560675883256529 / 0.10054844606946983 | 同 | **逐位一致** |
| `ci95` 下界 | 0.0945207949 | 0.0945206705 | 相对差 **1.3e-6** |
| `ci95` 上界 | 0.1090771526 | 0.1090772961 | 相对差 **1.3e-6** |

⇒ **结论**：CI 差异**只**来自 z 分位口径（相对 1.3e-6，不影响任何判定）；**R 与 se 逐位一致**⇒ 本工具的 CI 实现已被 M10 的独立实测件校验。
**纪律**：跨节点比较自持比 CI 时必须声明 z 口径。

## L3 — 口径差②：静默比例的定义（本工具 vs M10）

| 口径 | 定义 | 出处 |
|---|---|---|
| **M11 主口径** | `silent_frac_zero_count = mean(count_i == 0)` | 清单 §4.5(b)（本节点定稿，写入判据表 `silent_def_primary`） |
| M10 兼容口径 | `mean(rate_i < 0.5 Hz)` | `src/larval_zf_circuit.py::run_protocol::silent_frac` |

- 工具**同时**输出二者（前者 `p5_b_silent_frac` 主行，后者在 `arm_metrics.silent_frac_lt_thr` 与原因串中）。
- **量化后果**：同一网络两口径可差一个量级——M10 修正档 λ×1 臂实测 `silent(rate<0.5Hz)=0.920`，而其 `<0.5 Hz` 阈值等价于「20 s 内 10 个脉冲」；`zero_count` 口径会**更高**（近静默臂可达 0.95+）。混用会让 (b) 判定随风向改变。
- 与**跨制备代理**的区别：Rochefort 2009 的 12% 是「**每次自发波**招募比例」，Allen 2p 的 23% 是「**对刺激**不可靠响应比例」⇒ 两者都**不是**本口径（判据表已逐行给 `wrong_usage_consequence`）。

## L4 — 工具正确性在 M10 冻结数据上的独立核对

| 项 | 工具读数 | M10 公开记录 | 出处 |
|---|---|---|---|
| M0 基线 T=120 s 自持比 | **0.1719836**（1154/6710） | **0.1720** | `docs/m10_acceptance_master.md` §4.3 |
| M-6 T=120 s 自持比 | **0.1719091**（1175/6835） | 0.1720（同配置族） | 同上 |
| gate1 3 seed CI | 见 L2 | `data/m10_p5_gate1_ci.json` | 同 |

⇒ 工具能在 M10 冻结轨迹上**复现 M10 的公开记录值**。**M10 数值一律不作 M11 判据**（RF-M11-1）。

## L5 — 🔴 冲突①：清单 §1.5 C3 的 up/down dwell 依据系**误引**（物种/制备/数值三重不符）

- 清单原文（§1.5 C3）：「小鼠皮层 up/down 态 dwell **0.3–3 s** 量级（**PMID 11017176**）」。
- **本轮 Europe PMC 实测核读**（`PMID 11017176`，DOI 10.1038/79848）摘要原文：
  *"slices of ferret neocortex maintained in vitro generate this slow (<1 Hz) rhythm"*、*"the slow (0.1-0.5 Hz) oscillation during slow-wave sleep"*。
  ⇒ 该文是**雪貂皮层切片 in vitro**；摘要给 **0.1–0.5 Hz**；**摘要中不存在 0.3–3 s**。
- **替代锚（本轮核读）**：`PMID 8340806`（DOI 10.1523/JNEUROSCI.13-08-03252.1993）摘要：*"slow depolarizing envelopes (lasting for **0.8-1.5 sec**)"*、*"recurred rhythmically at less than 1 Hz, with a prevailing oscillation between 0.3 and 0.4 Hz in 67% of urethane-anesthetized animals"* ⇒ **猫在体（含 V17/18）、氨基甲酸乙酯麻醉**。
- **处置**：按纪律「**先改清单再改代码**」，本节点**不擅自改清单**；已在判据表新增 `proxy_steriade1993_upstate_duration_s`（0.8–1.5 s，跨物种+跨麻醉代理）并在 `m11_p5_targets.json::checklist_conflicts.L5` 提出改法。**注**：R10 已用 MICrONS 同源数据自建 8–9 s 目标带 ⇒ 该误引对 S 门的实际影响已被 R10 覆盖，但**清单文字仍需更正**（否则后续读者会继续沿用错误 provenance）。

## L6 — 冲突②：MICrONS 功能神经元数的精确值**未核实**

- 清单 §0.9 记 N_full = **75,909**，标注出处「Nature 2025 摘要」。
- 本轮 Europe PMC 核读 `PMID 40205214`（DOI 10.1038/s41586-025-08790-w）摘要原文：*"dense calcium imaging of **around 75,000** neurons in primary visual cortex (VISp) and higher visual areas ... in an awake mouse"*。
- ⇒ 精确值 **75,909 在摘要中不存在** ⇒ 判据表标 `literature_unverified`（`final=1` 但**不得**声称 measured/literature）；另立 `n_neuron_full_abstract_value = 75000`（literature）。
- **处置**：请主agent 补全出处（正文/表格/SI）或把清单改为 75,000。

## L7 — 冲突③：`m11_acceptance_master.md` 内部新旧段落自相矛盾（R9 未同步 §4）

- 同文档 §3d（R9）已判定「MICrONS 功能体素序列**可解码**（measured）」，而 §4「对 P2 判定的影响」仍写「功能侧：体素序列 `format_unverified`（R2）」「活动通道改由 Allen NWB 承担」。
- 本节点**不修改他节点/主agent 文件**；仅登记，请主agent 在 §4 加「（已被 R9 反转，见 §3d）」标注。

## L8 — ★ 时长依赖性的**分解实测**：截断 vs 独立重跑（本工作包的新事实）

用 M10 冻结轨迹构造**对称截断**（zero 与 base 各截到 T=2 s / 60 s，与 T=120 s 原档对照），把「同一轨迹的不同观察窗」与「不同协议时长的独立运行」分开：

| 来源 | T=2 s | T=60 s | T=120 s | 相对变化 |
|---|---|---|---|---|
| **同一轨迹截断**（本节点实测；冒烟 S5） | **0.16964** | **0.17198** | **0.17198** | **+1.4%** |
| **M10 独立重跑**（主agent A12 记录） | 0.0980 | 0.1082 | **0.1720** | **+76%** |

⇒ **结论（重要）**：M10 A12 观测到的「自持比随 T 漂移 +76%」**不是**脉冲列车内部时间结构的窗口效应（截断下仅 +1.4%，且 T=2 s 就已给出 0.17），而是**协议效应**——不同 T 的**独立运行**有不同的**初始化/初值语义（`v0=v_rest+2·N(0,1)`, rng(1)）/ settle 窗 / 事件表 RNG（`make_events(seed=3)` 的抽样随 T 变化）**。M10 记录中 T=2 s 的 0.0980 出自 `validate_p10_resting.py` 的 P5 校准档（另一套网络参数与 λ0），而 T=120 s 出自 pop-trace 作业 ⇒ **两者可能根本不是同一配置**（配置注释为 N=250/bias 150/g_exc 1.0/g_inh 22/λ0=8，但 `settle` 与事件表种子未检定稿）。
- **对判据设计的三条强制推论**：
  1. **时长阶梯必须逐档独立重跑**（含 settle 与初值语义），**禁止**用同一轨迹的截断构造阶梯；
  2. 工具已加**截断守卫**（`ladder_arms_look_truncated`：短轨迹是长轨迹的逐位前缀 ⇒ 阶梯 `invalid`，冒烟 S5 已触发）；
  3. 阶梯的每一档必须**同时登记 `settle_ms` 与事件表 seed**（本工具已把 `protocol_settle_ms`/`protocol_seed` 作为逐行字段输出）。
- **待主agent 裁决的具体问题**：M10 A12 的 0.0980/0.1082/0.1720 是否**同配置同 seed 仅改 T**？若否，则「判据时长依赖性 +76%」这一论据本身需要重测（本节点**无权限跑仿真**，故只登记）。**这不影响本工作包的结论**（无论漂移来源是窗口还是协议，判据都必须带协议时长做阶梯），但影响主agent 对 M10 A12 的引用强度。

## L9 — 跨制备代理的量化后果（判据表 `wrong_usage_consequence` 的数字来源）

| 代理 | 实测/核读值 | 若误用为 (a)/(b) 带的后果 |
|---|---|---|
| Allen Cell Types `ef__avg_firing_rate`（**in vitro 电流钳**） | 全体 n=1808 中位 **17.4130 Hz**（均值 31.7180、p5 6.879、p95 107.33）；VISp n=1490 中位 **17.6338 Hz**（L2/3 19.8220 / L4 16.7383 / L5 19.3125）；spiny 12.9067 vs aspiny 34.1246；**frac<0.5 Hz = 0.0000**、frac<5 Hz = 0.0195 | 中位率带被判 **高 8.7–34.8×**（相对 0.5–2.0 Hz）；**静默带被压到 ≈[0,0.02]** ⇒ 与 (a) 结构性冲突（M10 A4 标定权衡的正体）。**这正是 M10 自持/静默两难的口径根源** |
| Rochefort 2009（小鼠 V1 L2/3 **2p、每次自发波**） | 成年 **12%** 活跃/波、波频 ≈**0.5 Hz**（摘要已核） | 「12% 活跃」被读成「静默 0.12」⇒ 与长窗零计数口径差一个量级 |
| Allen 2p Visual Coding（**对刺激可靠响应**口径） | **77%** 响应 ⇒ 23% 未响应（OA 全文已核） | 静默带被压到 ≈[0,0.25]，与「存在大量近静默细胞」冲突 |
| Steriade 1993（猫在体、氨基甲酸乙酯） | up 态 **0.8–1.5 s**、0.3–0.4 Hz | 用麻醉猫的 up 态时长判清醒小鼠 dwell（且 P5 不判 dwell ⇒ 越界） |
| E/I 四篇（猫清醒/自然睡眠、大鼠听皮层、麻醉猫 V1、小鼠 V1 ISN） | 方向：抑制不弱于兴奋、E/I 同量级、个体高度多样 | 压窄到 [1.0,1.5] ⇒ 合规小鼠 V1 配置被误判失败（假阴性）；放宽到 [0.1,10] ⇒ 判据失去约束力 |

## L10 — 冻结件零修改核查

- 本节点**只新建**：`data/m11_p5_targets.{csv,json}`、`data/m11_p5_smoke_*`（协议/夹具/输出，**28 件**）、`data/m11_b3_managed_jobs.json`、`tools/validate_p11_resting.py`、`docs/m11_b3_{g1_design,notes}.md`。
- **未改**：`src/mouse_cortex_*`（B2）、`data/m11_data_*` / `tools/fetch_m11_*` / `tools/build_m11_*`（B1）、`docs/m11_acceptance_master.md` / `docs/m11_data_gate_report.md` / `docs/m11_env_notes.md` / `docs/m11_list_review_master.md`（主agent）、以及全部 M0–M10 冻结件（只读消费：`m10_p6_poptraces/*.npy`、`m10_p5_gate1_ci.json`、`m10_calibration_targets.csv`、`src/larval_zf_circuit.py`）。
- ⚠️ 另有**非本节点**的改动正在并发发生（B2/主agent）。以 **mtime 交叉证明**（本节点写入窗口 21:45:23–21:48:17）：

| 文件 | mtime | 归属 |
|---|---|---|
| `data/m11_p5_targets.csv` / `.json` | 21:46:56 / 21:47:09 | **本节点** |
| `tools/validate_p11_resting.py` | 21:45:23 | **本节点** |
| `docs/m11_b3_g1_design.md` / `m11_b3_notes.md` / `data/m11_b3_managed_jobs.json` | 21:47:24 / 21:48:01 / 21:48:17 | **本节点** |
| `src/mouse_cortex_engine.py` | 21:46:01 | **B2**（非本节点） |
| `tools/scan_m11_engine.py` / `data/m11_engine_alignment.json` | 21:48:12 / 21:48:21 | **B2**（非本节点） |
| `data/m10_p6_judgement_table.csv`、`tools/normalize_m10_p6_table.py`、`reports/neuro/m11_g0_engine_summary.json` | 同窗 | **B2/主agent**（非本节点） |

⇒ 请**勿**把上述并发改动归因于本工作包；本节点全程未向这些路径发起任何写操作。

## L11 — 🔴 跨节点并发事件：我的首版表述按 R2 写、R9 到达后更正

- 时间线（本节点实测）：
  - 本节点约 **20:2x** 读 `docs/m11_acceptance_master.md` ⇒ 当时是 **95 行**的版本（含 R2「体素序列 format_unverified」与 R6「活动通道改由 Allen 承担」）；
  - 主agent 于 **21:23 / 21:46** 把该文件更新为 **185 行**（新增 **R9 反转**、**R10 同源目标带**、**R11 G0=fail**）；
  - 本节点的 `data/m11_p5_targets.csv` 首版据此写了 `microns_activity_channel = unavailable`（理由引 R2）；
  - **21:5x** 本节点重读该文件 ⇒ 发现 R9 ⇒ **已更正**为 `microns_activity_channel_voxel_series = measured`（引 `data/m11_reference.csv::R-10..R-14`）+ 新增 `microns_activity_channel_percell = unavailable`（R1 保持）。
- **纪律沉淀**：① 长工作包在**收口前必须重读主agent 的验收/裁决文件**（其可能在我开工后被重写）；② 「不可解码/不可用」结论的举证标准（R9 新增）本节点已遵守——所有 `unavailable` 行均注明**未测/未交付/永久限制**三类理由，且不把「只测了一种解码」写成不可用；③ 本节点**未**因为并发而覆盖或回退他节点的改动。
- 另：`tools/validate_p11_resting.py` 在本会话中被主agent **于中途 commit 入库**（`5cf62d5`/`a2e59cf`），故 `git diff` 相对 HEAD 只显示我此后的增量（54+/23−）。**最终版本以工作区为准**，sha256 见 L12。

## L12 — 跨节点 sha256（**必须标注基准**，纪律 §6-3）

| 文件 | 字节 | `file_bytes` sha256（前 32） | 备注 |
|---|---|---|---|
| `data/m11_p5_targets.csv` | **32,921** | `600a515df55550e6c661c826522c2c65` | **判据带唯一定稿表（80 行）**；`payload` 基准（行 canonical JSON）sha256 = `6839c710705b33abbf12fdc2e5b29d46` ⇒ **同一文件两基准不同是正常现象**（主agent 纪律 §6-3 的成因） |
| `data/m11_p5_targets.json` | 102,391 | `fc742811d5aa0fcf2f05d06d7d8e6bde` | 内含 `hash_basis.csv_sha256_file_bytes` 与 `hash_basis.rows_payload_sha256` 双基准（自描述） |
| `tools/validate_p11_resting.py` | **81,191** | `4250ded5d9d9f88196482f4db760782f` | 该文件被主agent **中途 commit**（`5cf62d5`/`a2e59cf`）⇒ 跨节点比对该哈希时必须带时间戳（本值为**最终工作区版**；相对 HEAD 的增量 = 54+/23−，见 L11） |
| `data/m10_p5_gate1_ci.json`（只读） | 670 | `347165ab37aa17cc0af84adf7a064046` | 工具 CI 校验的参照 |
| `data/m10_calibration_targets.csv`（只读） | 2,737 | `1ae636d62d3904b653df1ab3477cf9cb` | M10 判据带（只读引用） |
| `data/m11_allen_celltypes_metadata.csv`（只读） | 1,171,753 | `8cbf7d186dd15c7c46b94cabaf44582f` | L9 的复算来源 |
| `data/m11_microns_protocol.json`（只读） | 231,518 | `29366f6f5aa6a8f796e399191c0cabdb` | 帧率 6.300006 Hz 来源 |
| `data/m11_reference.csv`（只读，B1） | 7,084 | 见文件 `data/m11_data_inventory.csv` | R-01/R-10..R-14/R-21 的引用来源 |

> **基准定义**：`file_bytes` = 文件原始字节；`payload` = 按 `key` 排序后的 canonical JSON（`sort_keys=True, separators=(',',':')`）；`semantic` = 行集/字段语义（本工作包只用前两者）。

## L13 — 冒烟测试结果（S1–S6，全部 <10 s，只用 M10 冻结件 + SYNTHETIC 夹具）

| 用例 | 配置 | 判定 | exit | 证明的能力 |
|---|---|---|---|---|
| **S1** | 单臂、**只有群体轨迹**（M0 λ×0） | `unavailable` | 3 | 四态如实；缺 counts ⇒ (a)(b)(f) 逐神经元口径 `unavailable`；单一估计器 ⇒ 硬门③ `unavailable` 且**阻断 pass** |
| **S2** | M10 `gate1` **3 seed 聚合档**（275/275/275 vs 2786/2604/2735） | **`fail`** | 1 | 硬门①满足（3+3 seed、零驱动 825 脉冲 ≥300）；硬门②CI 与 M10 **逐位一致**（L2）；判定「点估计 0.10154 > 0.10 ⇒ 未过」并标注「统计未分辨（CI 上界 0.109 ≤ 0.12）」⇒ **复现并固化 M10 的临界语义** |
| **S3** | 2 配置 × 3 剂量（M10 轨迹 + **SYNTHETIC** counts） | `unavailable` | 3 | **22 行输出齐（(a)–(i)）**；双估计器 `rel_diff = 0.0`（6/6 臂）；自持比 0.17198/0.17191（复现 M10 0.1720，L4）；**硬门①触发 ⇒ (c) `unavailable`（每臂仅 1 seed）**、点估计保留为描述性；(d) `pass` 但标**非判别性 sanity**；张力块给出 `ρ(驱动,率)=+1.0`、`ρ(驱动,静默)=−1.0`、`qualitative_monotone_opposition=true`（(b) 带未定稿 ⇒ `structural_tension` 明示"只能定性"，**防构造性结论**） |
| **S4** | 同 S3，把基准臂 counts **人为放大 1.15×** | **`invalid`** | **2** | 硬门③硬失败：`rel_diff = 0.1304 > 0.02` ⇒ 整轮 `invalid`（**不是 fail**）⇒ M10 四次口径事故的机械化堵漏 |
| **S5** | T=2/60/120 s **截断**阶梯 | **`invalid`** | **2** | 硬门④截断守卫触发（`ladder_arms_look_truncated`）⇒ 截断不能冒充协议时长阶梯；附带 L8 的分解实测 |
| **S6** | 单臂 CLI（`--pop-trace` + `--counts`） | `unavailable` | 3 | 单臂模式可用；`rel_diff = 0.0` |

## L14 — 本工作包的诚实边界（**未做 ≠ 未命中**）

| 项 | 状态 | 原因 |
|---|---|---|
| (a) 的**实测**带（小鼠 V1 在体率） | `unavailable` | 逐细胞轨迹 only-in-DB（R1）；体素序列虽已可解码（R9）但**逐 ROI 率参考量未重建**（`m11_reference.csv::R-21` 仍 unverified；无 ROI 提取管线）；Allen in vivo NWB 未下载（仅目录 569 对象）⇒ **registered job J1/J2** |
| (b) 钙域 / 电生理域两个带 | `unavailable`（`final=0`） | 口径未定稿（清单 §0.7 #9；R4 的 10× 口径差仍 NEEDS_VERIFY） |
| (e) 的实测量 | `unavailable` | 本工作包不做仿真；`(e)` 只能由仿真侧注入电流积分 |
| (g) 墙钟上限 | `unavailable`（`final=0`） | 待 G0 探针实测（R11 下 G0 = fail ⇒ 更远） |
| 全部 P5 判定 | **不得发布** | **G0 = fail（R11）⇒ G1 不得启动**；判据表与工具处于**待命态** |
| 仿真 / 长作业 | **零** | 纪律；>10 min 一律交主agent 托管（清单见 `data/m11_b3_managed_jobs.json`） |

## L15 — 教训沉淀（可复用到 M11 后续）

1. **判据带未定稿必须机械阻断判定**：把「带是否定稿」做成表字段（`final`）并由工具强制 ⇒ 未定稿带只能产出 `unavailable`；这比"写在文档里提醒"可靠（本工作包有 3 条带因此被拦住，避免伪通过）。
2. **"能被截断冒充"的量必须加守卫**：时长阶梯可被同一轨迹切片伪造 ⇒ 工具加逐位前缀检测（S5 触发）。凡"成比例/成倍数"的协议维度都应做同类守卫（如「剂量扫描用同一事件表的缩放副本」）。
3. **张力识别要与判据带状态绑定**：`n_both_in_band == 0` 在带未定稿时**恒真** ⇒ 若据此报"结构性张力"就是构造性结论（M10 D2/D3 同族）。修法：`structural_tension` 仅在两条带都定稿时可为 True，否则只给 `qualitative_monotone_opposition`。
4. **"单一估计器"是 `unavailable` 而不是 `invalid`**：只有 1 个估计器时硬门③**不可校验**（未测），与"两个估计器不一致（口径缺陷）"是两回事 ⇒ 分别映射到 `unavailable` / `invalid`。（首次实现把前者记成硬失败，已修正。）
5. **收口前重读主agent 裁决文件**（L11）：长工作包期间上游可能被重写（本次 95 行 → 185 行，含一次**裁定反转**）。

## L16 — 初审反馈落实：**只允许新增行**的口径分列升级（72 → 80 → 94 行）

主agent 初审（对 72 行版）要求：① 升级一律**新增行 + 保留原行**；② 钙口径与电生理口径**分列**；③ 样本范围写进 note；④ 哨兵掩膜前置并登记；⑤ Allen 2p 也要分列。

| 动作 | 行 | 说明 |
|---|---|---|
| **恢复原行**（逐字取自 commit `5cf62d5`） | `microns_activity_channel` | 我上一轮曾**就地替换**该行为两个新键 ⇒ 本轮**恢复原行文本**（value/value_kind/evidence_level/provenance 逐字保留），仅 `final` 置 0 并追加「R9 后状态」指针。**同类的就地修改还有 `rate_median_hz_measured_in_vivo_band`（note 被改）**，本轮同样恢复原 note 文本后仅追加指针。⇒ **纪律自纠**：我自己在 R9 到达时违反了「只允许新增行」，已在本条登记并回退。 |
| **新增（口径分列）** | `rate_median_hz_literature_spike_band`（literature，0.5–2.0，电生理口径命名行）、`rate_median_hz_measured_calcium_band_lo/hi`（**unavailable**）、`calcium_caliber_nyquist_ceiling_hz`(3.150003 Hz)、`calcium_caliber_resolvable_floor_hz`(0.00315 Hz)、`calcium_vs_spike_caliber_gap` | 钙口径带**当前仍不可得**：R9 只使数据「可解码」，**逐 ROI 的事件率管线尚未建立** ⇒ 已测的钙域「率」只有**群体状态转换率**（0.104–0.438 Hz），不是逐细胞率。 |
| **新增（样本范围 / 哨兵）** | `microns_functional_calcium_sample_scope`（**2 field / 2,000 帧 ≈317.46 s**；单 field 完整记录 40,000 帧/6,349.2 s 未全取）、`microns_functional_sentinel_count`(10,383)、`microns_functional_sentinel_fraction`(0.01298%)、`microns_calcium_measured_population_anchors` | 「未处理 ≠ 无缺失」已写入 note。 |
| **新增（Allen 参照）** | `allen_2p_role_in_m11`（cross_preparation_reference）、`allen_bo_neuropixels_spike_rate_band`（unavailable，仅目录 569 对象） | 按 R6/R9 只作参照。 |
| 行数 | 72 → **80**（R9/R10/R11）→ **94**（本轮） | 证据级别：`derived 52 / literature 18 / measured 11 / unavailable 12 / literature_unverified 1`；`final=0` **13 行** |

### L16.1 🔴 未决冲突：哨兵掩膜的处置（**影响上界 ≤0.0130 个百分点 ⇒ 不阻塞 G1**）

- **位置 A**（主agent 本轮指令）：`uint16 读法 > 65000 ⇔ int16 < 0` 必须**前置为缺失**并登记；
- **位置 B**（`docs/m11_acceptance_master.md` §3f-3 + `data/m11_reference.csv::R-14`，**B1 推翻主agent 且已被采信**）：**无掩膜，保留并登记**；理由是**空间成簇证据**（3×3 邻域负值比例 p50 **0.111** vs 随机 ≈0.000129 ⇒ **~860× 富集**、孤立比例 **0.000**、位于暗区、邻域均值 ≈395 vs 全局中位 1,953）⇒ 按 A 处理会**删掉真实暗区数据**。
- **两侧一致的部分**：计数 **10,383 / 80,000,000 = 0.01298%** 与「必须登记」。
- **本节点的量化界定**：即使把这 0.01298% 全部当成缺失或全部当成有效，任何**比例/中位数**类统计的变化 **≤0.0130 个百分点**，**低于任何合理带宽**（判据表带宽以 0.05 量级计）⇒ **该处置之争不阻塞 P5/G1**；但对**空间成簇**类判据（P14 的像素/邻域统计、暗区偏置）可能更重要，**仍需裁决**。
- 本节点**不擅自裁决**：判据表以 `microns_sentinel_mask_policy = unresolved_conflict` + **`final=0`** 登记，并列出三条裁决路径（两侧各自复算给差异 / 采纳 B1 的成簇证据 / 主agent 明确指定并注明与 §3f-3 的关系）。

## L17 — 硬门⑥ 的**真断言**落地（并由断言抓到一处真缺口）

- 新增 `Evaluator._assert_output_invariants()`（**raise AssertionError**，非"软提示"），三条不变式：
  ① 每行必须带 7 个口径字段（`protocol_id/T_ms/settle_ms/seed/dt_ms/bin_ms/smooth_frames`）+ `estimator` + 非空 `protocol_T_ms`；
  ② **`pass` 的带类判据（a/b/c/e）必须有非空 `band_lo`（a/b/e 还需 `band_hi`）与非空 `value`** ⇒ 把「未定稿带不得判 pass」从"约定"变成"不可绕过"；
  ③ 每个评估单元（config×T×mechanism）必须出现**全部 9 条子判据** ⇒ 防「判据静默丢失」（与 M10 编码路径静默失效同族）。
- **断言在首跑即抓到一处真缺口**：无臂配置（S1 之类）下 `(d)` 行由 `arm=None` 生成 ⇒ `protocol_T_ms` 为 `None` ⇒ **硬门⑥ 违反**。修法：`_proto_fields(arm, T_ms)` 在无臂时**回退到该评估单元的 T**（settle 回退到协议值，seed 记 `None`）。
- 全部冒烟输出 `output_invariants = {'rows': N, 'units': M, 'criteria_per_unit': [9], 'caliber_fields_checked': 8}` ⇒ 硬门⑥ 已**可机械核验**。

## L18 — 硬门③ 用 `hard_failures + verdict` 而非 `raise` 的取舍（有意为之）

- M10 的内核用 `assert` 直接抛异常（`larval_zf_circuit.py::run_protocol` 的双估计器断言）：优点是"当场炸"，缺点是**整轮输出（含 JSON/CSV）全部丢失**，事后无法审计是哪一段口径错了。
- 本工具的选择：**记录到 `hard_failures` + 总判定置 `invalid` + exit=2**，输出照常落盘（S4 实测：`rel_diff=0.1304` ⇒ `invalid`、CSV/JSON 完整可用）。
- 代价：脚本本身**不会非零退出以外的"硬停止"**（若调用方忽略 exit code 就可能漏掉）⇒ 规避方式：① exit code 语义固定并写入 docstring；② `hard_failures` 在 `print_summary` 与 JSON 顶层各出现一次；③ 只有**输出不变式**（硬门⑥/带纪律/子判据完整性）用 `raise`（那三类一旦违反，输出本身不可信）。

## L19 — 🔴 清单 §10.1 的 GPU-h 公式常数偏大 **12.96×**（预算级发现）

- 清单 §10.1 原文：`GPU-h = Σ_条件 Σ_试次 [ N_steps(T, dt) × ms_per_step(实测) × 3.6e-6 × (1+消融臂数) ]`。
- **物理正确式**：`GPU-h = Σ (steps × ms_per_step) / 3.6e6`（1 h = 3.6e6 ms；即常数应为 **2.7778e-7**）。
- **自相矛盾证据（同节内）**：该节同时写「单试次（T=30 s, dt=0.05 ms ⇒ 600,000 步）：N_mid **0.05–0.15 GPU-h**」。
  用 0.3 ms/step 代入**清单的常数** ⇒ 600,000×0.3×3.6e-6 = **0.648 h**（是 0.05 的 12.96×）；
  用**正确常数** ⇒ 600,000×0.3/3.6e6 = **0.05 h** ✅ 与该节的单试次行一致。
- **影响**：若按清单常数做预算 ⇒ G1 段（本节点估 ≈8.9 GPU-h）会被算成 **115 GPU-h**，直接触发不必要的 NG 降档/云 GPU 裁决。
- **处置**：本节点在 `m11_b3_g1_design.md` §9 明确采用 **1/3.6e6** 并逐条给算式；**建议主agent 把清单 §10.1 的常数更正为 `2.7778e-7` 或 `/3.6e6`**（按「先改清单再改代码」由您落）。**未擅自改清单。**

## L20 — 🔴 R10 目标带的**样本范围实测更正**：落盘数据来自**单一 field**（不是 2 field）

- 主agent 本轮表述：「2 field、317.5 s/**field**」。三处**独立实测证据**表明并非如此：

| 证据 | 实测内容 |
|---|---|
| `data/m11_functional_subset_manifest.csv` | **200/200** 对象，url 前缀**唯一 = `session4_scan7_field4`**，`t_chunk` 0–199，verdict 全 `REACHABLE` |
| 磁盘 `data/m11_raw/microns/functional_subset/` | **200 个文件**，前缀全部为 `session4_scan7_field4_*` |
| `data/_mainagent_m11_same_source_band.json` | `n_frames = 2000`、`duration_s = 317.46001511744595`、`frame_hz = 6.300006`（**单一序列**） |
| `data/m11_microns_protocol.json` | **声明** 2 个 series（`session4_scan7_field4` / `session5_scan3_field1`），**第 2 个未取数** |

- ⇒ **正确的样本范围 = 1 field × 200 chunk × 10 帧 = 2,000 帧 = 317.46 s**；R10 的「两个」指的是**两条分析管线**
  （`frame_mean` / `high_pixel_frac`），**不是两个 field**。
- **连带后果（重要）**：R10 自己列的「**节点须做：① 两个 field 都建带 ② 给跨 field 离散度**」**仍然 open**
  ⇒ 该带**跨 field 通用性未测**、**跨 field 离散度 unavailable** ⇒ **RF-M11-2 的「同阶窗」目前只能取跨条件离散度**
  （Q/H 双法 2.6–6×、跨平滑 3.29×、跨箱宽 4.2×）作依据，**不得**声称来自跨 field。
- **处置**：新增 `microns_functional_calcium_sample_scope_perfield_measured`（measured）；旧行
  `microns_functional_calcium_sample_scope` **value 逐字保留**、`final` 置 0、note 追加实测更正指针（append-only）。
  已同步写入 `m11_p5_targets.json::sample_scope_correction`。
- **同时补入**（主agent 本轮要求）：`microns_same_source_tau_band_smoothed_ref`（5 帧平滑 τ 29.68/30.48 s，放大 3.29×）
  与 `microns_same_source_dwell_anchors`（Q：中位 1.11–1.27 s、p90 7.0–7.6 s、n=69/63；H：3.17–6.67 s、p90 32.6–34.2 s、n=17/16）。
  后者的 H 法 **n=17/16 接近 RF-M11-4「bout 数 ≥10」下限** ⇒ 再少即须判 `unavailable`。

## L21 — 收尾状态（G0 阻塞下的"待命"定义）

| 项 | 状态 |
|---|---|
| 设计与工具 | **验收通过**（主agent 代码级核对）；`--check-targets` 自检 OK（97/97 行，CSV⟺JSON 一致） |
| 冒烟 | S1–S6 全过（见 L13 / §8）；M10 数值与 SYNTHETIC 计数**一律不作 M11 判据** |
| G1 执行 | **不得启动**（G0 = fail，R11）；协议矩阵与预算已在 `m11_b3_g1_design.md` §9 待命 |
| 判据带待定稿 | `silent_frac_lo/hi`（钙/电生理两域）、`rate_median_hz_measured_calcium_band_lo/hi`、`ei_ratio_measured`、`single_trial_wall_s_max`、`microns_sentinel_mask_policy` 等 **14 行 final=0** |
| 未做（≠未命中） | 钙口径带的逐 ROI 事件率管线（J2）、Allen in vivo 率带（J1）、G1 全矩阵（J3–J5） |

## L22 — 教训沉淀（补充）

6. **"声明" ≠ "落盘"**：`protocol.json` 声明 2 个 functional_series，但 manifest/磁盘只有 1 个 ⇒ **凡"数据来自 N 个 X"的表述必须回溯到 manifest/磁盘逐件核**（本次差 2× 的样本范围与「跨 field」结论）。
7. **公式常数也要做量纲自检**：清单 §10.1 的 `3.6e-6` 与同节单试次行相差 12.96× ⇒ **凡预算公式必须与同文档的"示例单值"交叉验算**（本工作包的"双估计器"律在**预算文本**上的又一次实例化）。
8. **"两个口径"与"两个样本"是两件事**：R10 的 2 条管线被读成 2 个 field ⇒ 把**方法学口径差**误当**生物学变异**（与既有口径纪律同源）。

## L23 — 主agent 六项裁决落盘（收口轮）

| 裁决 | 内容 | 本节点的落盘动作 |
|---|---|---|
| **A12 更正**（主agent **自纠**） | 三值实际配置不同：T=2 s/**N=1000**/**settle=300 ms**；T=60 s/N=1000/settle=1000 ms；T=120 s/**N=250**/settle=1000 ms ⇒ 「+76% 时长漂移」**是把配置差异说成时长依赖**；受控截断实验仅 **+1.4%** | 本节点 **L8** 的「先登记、不擅改」被确认为正确处置；**判据设计不变**（协议元数据 + 时长阶梯仍必需 —— **配置敏感性反而被证实更强**）。已同步入 `m11_p5_targets.json::mainagent_rulings_2026_10_11` |
| ② (a) 的带 | **保留 literature 档**（0.5/2.0 Hz，PMID 30419013 全文已核）；**J1 判 `unavailable（吞吐限制）`**（S3 实测 **10–17 KB/s** ⇒ ETA **1.6–7 h**） | `allen_bo_neuropixels_spike_rate_band` note 追加吞吐裁决 + **续取计划**（换网络/时段 + Range 续传 + 先探吞吐）+ 「原写 20–60 min 已废」；`rate_median_hz_literature_spike_band` note 追加**性质限制**（建模论文对实测的**转述、非原测、只覆盖 L4**） |
| ③ J2 | **批准**；**执行归 B1** | 新增 `j2_reserved_keys_and_acceptance`：预留 **5 个 key**（钙率带 2 + 静默带 2 + **新增跨 field 离散度**）+ **5 条验收条件**（口径逐项 / 跨 field 离散度 / 与电生理域并列 / 哨兵处置先声明 / 不可得时须写明已覆盖维度）；`m11_b3_managed_jobs.json` 的 J2 标 approved + handover_to=B1（占位命令作废，**不得新增 tools/*.py**） |
| ④ 清单文字 | 批准（§1.5 C3 → PMID 8340806 + 跨物种/麻醉；§0.9 N_full → `literature_unverified`）；**由主agent 落盘** | 本节点**不改清单**（先改清单再改代码）；判据表已按新依据给行 |
| ⑤ 文档矛盾 | `m11_acceptance_master.md` §4 与 §3d 的矛盾**已修** | L7 可标为**已闭环** |
| ⑥ **G1 门条件加强** | **G1 = (c) ∧ (e) ∧ 张力已定量识别**；**(d) 降为 sanity** | 判据表新增 `gate_g1_conditions_v2`（旧行保留 + final=0 + 指针）；工具 `GATE_CRITERIA` 改为 (c, e, `p5_g1_tension_quantified`)，新增该子判据的发射与判定；**新增 `_assert_gate_composition()` 断言**（表↔代码一致性，负向测试实测 raise「门组成漂移」）；`CRIT_EXPECTED` 由 9 → **10 条**（不变式随之更新） |
| G0 现状 | **partial**（(a)(e)(f) pass、`off_state_equivalence` 由 fail 转 pass；**(b)(c)(d)(g)(h) 与 4 项 extensions 仍 unavailable**） | 门链不变：**G0 通过条件 = 全项 measured 且全过 ⇒ G1 暂不得启动**；`m11_b3_managed_jobs.json` 的 policy 已同步该状态 |

### L23.1 冒烟复跑（新门组成后，全部通过）

`S1 unavailable/3`（12 行）· `S2 fail/1`（12 行）· `S3 unavailable/3`（**24 行**）· `S4 invalid/2` · `S5 invalid/2`；
全部 `criteria_per_unit=[10]`（9 + 张力门）、`gate_criteria=['p5_c_selfsustain','p5_e_ei_balance','p5_g1_tension_quantified']`、`sanity=['p5_d_drive_elasticity']`。
**张力门在 (b) 带未定稿时的正确行为已实测**：S3 给出 `unavailable` 并显式说明「**n_both_in_band 在带未定稿时恒为 0，不构成张力证据**」（防构造性结论）。

### L23.2 新增教训

9. **门组成本身也要防漂移**：清单/判据表声明与代码实现是**两份可独立漂移的文本** ⇒ 本工作包用 `_assert_gate_composition()` 把「表怎么写就怎么判」做成断言（与"双估计器一致"同族：**同一语义的两个载体必须一致**）。负向测试（把表改成 `c_and_d_and_e`）实测触发 `AssertionError: **门组成漂移**`。
10. **"张力被定量" ≠ "没有张力"**：门条件写作「张力已**定量识别**」而不是「无张力」——否则会把清单 §4.5(i) 明确要登记的标定权衡变成门失败（判据语义的**方向性**错误，比数值错误更难发现）。

## L24 — 掩膜裁决定稿 + J2 key 命名对称化（主agent 2026-10-11 §一/§二）

### L24.1 掩膜：**定稿 = 无掩膜（`no_mask(retain_and_log)`）**
- 主agent 裁决：`int16<0` 的样本**保留并登记**；**其旧指令**（`uint16>65000 ⇔ int16<0` 前置为缺失，写在 B1 取证**之前**）**已撤回**（M11 验收记录 §3f-3 与 R9 反转已载明）。
- 依据（B1 空间成簇证据）：3×3 邻域负值比例 p50 **0.111** vs 随机 ≈0.000129 ⇒ **~860× 富集**、**孤立比例 0.000**、位于**暗区**（邻域均值 ≈395 vs 全局中位 1,953）、仅比局部基线低 ~381 counts；若按 uint16 读，其真值将是该对象自身最大值的 ~2.3× 而邻居仅 395 ⇒ **物理荒诞解**。
- **本节点动作（含审计）**：`microns_sentinel_mask_policy` 的 **value** `unresolved_conflict` → `no_mask(retain_and_log)`、**evidence_level** → `measured`、**final** 0 → 1；**原 provenance / note / wrong_usage_consequence 逐字保留于该行 note 的「修订记录（append-only 审计）」段**。
  ⇒ 这是本表**唯一**一处按裁决指令就地改 value 的行（非判据带、而是**未决冲突的裁定**），已在此登记；其余行一律「新增行 + 指针」。
- **量化界定保留**：两侧计数一致（10,383 / 80,000,000 = **0.01298%**）⇒ 比例/中位数类统计影响 **≤0.0130 个百分点**；**但 P14 的「空间成簇」类判据（像素/邻域统计、暗区偏置）可能更敏感** ⇒ 该处置在 P14 必须显式声明。
- **「哨兵」一词的口径**：仅指 **32767 饱和**（实测计数 **0**），**不是** below-baseline 样本（R9 §3f-3 已更正）。

### L24.2 J2 预留 key 命名对称化
- 新增 **`silent_frac_calcium_band_lo/hi`**（`unavailable`，`final=0`）⇒ 与 rate 侧的 `rate_median_hz_measured_calcium_band_*` **命名对称**；原 `silent_frac_lo/hi` 保留 + 加命名分列指针。
- **J2 执行归 B1**（本节点**不动 J2、不新增 `tools/*.py`**）；`j2_reserved_keys_and_acceptance` 的 key 清单与 note 已同步（5 个 key + 5 条验收条件）。
- 钙域静默带的两个口径候选（`zero_activity` / `low_activity`）已写入预留行 note，**阈值必须在 B1 落口径时定稿**。

### L24.3 G0 现状（R13 后）
节点报 `pass`，但 **(b) 出现 `fail→pass` 的极端翻转**（0.5 Hz 档：静默比差 **14.0 pp / KS p=6.8e-4** → **0.0 pp / p=1.0 / 逐位相同**）⇒ **主agent 暂缓采信**，要求交「翻转解释（引擎修复？参照替换？口径变更？）+ `root_cause`/`fix` 字段 + `registered` 四态归一」。
⇒ **未解释前 G0 不算通过 ⇒ G1 继续停在设计/工具态**（与 L23 的门链一致）。**这正是「极端翻转 = 口径变更或参照替换」的典型信号**（M10 的 RF-5~RF-8 家族），在解释落盘前**不得**把 G0 记为 pass。

### L24.4 教训补充
11. **"未决冲突"行被裁决后，就地改值也要留原文**：本表对 `microns_sentinel_mask_policy` 按裁决改 value，但把**原 provenance/note/wuc 逐字保留在 note 的修订记录段** ⇒ 审计链不丢（若只改值不留原文，则"曾经的冲突两侧证据"会随改值消失）。
12. **未决项的"影响上界"比"哪一方对"更有用**：本节点在冲突未决时先算出「**≤0.0130 个百分点 → 不阻塞 G1**」，使主agent 可以**在不阻塞关键路径的前提下**从容裁决（并把唯一真正敏感的场合（P14 空间类）单独点名）。

## L25 — 用**实测** ms/step 重定价（主agent 派工；**纯台面算术，未跑仿真**）

### L25.1 换算因子结案（E-M11-2）
- 错式 `3.6e-6` 的**位置** = **`data/m11_engine_params.csv` L21**（`gpu_h_per_trial_formula`）—— **不在清单里** ⇒ 与主agent R18「定位前不改清单」一致；**由 B2 append-only 修正**，本节点不动该 CSV。
- 量纲自检（我复算，与主agent 一致）：`600,000 步 × 1.1362 ms/step` = **2.4542（错式）vs 0.1894（对式）**，比 **12.960**；恒等式 **1 GPU-h = 3.6e6 ms** ⇒ 正确因子 **2.7778e-7**。
- **本工作包 §9/§10/§11 一律用正确因子**（未受 ×13 污染）。

### L25.2 实测基准与口径（三处一致）
`ms/step`：**N=8,192 ⇒ 0.6615**、**N=31,000 ⇒ 1.1362**（B2 交付，稳态、MPS、禁用段 0 预热）。
`dt=0.05 ms`（`m11_p5_targets.csv::protocol_dt_ms` ← `src/larval_zf_circuit.py::DT_MS` + `point_neuron.py:239`，三处一致）⇒ T=10 s = **200,000 步**、T=30 s = **600,000 步**（与 §10.1 的 0.6M=3 run 合计一致 ✓）。

### L25.3 🔴 本节点自纠：§10.1 的 **P2 行算术错误**
P2 = 9 run = 3 剂量 × 3 档 T = **4.2M 步** ⇒ 按 0.6 ms/step 应为 **0.70 GPU-h**，而 §10.1 写了 **0.30**（漏乘三档 T 的合成），连带把"探针合计"写成 **0.60**（正确 **1.00**，低估 40%）。**旧数字在 §10.1 逐字保留**，§11.2 给出更正版。
- 主agent 的 **1.894 GPU-h 与本节点逐臂表逐位一致** ✅。
- 合计 1.894 相对旧估 0.60 = **3.16×**；分解：**ms/step 因子 1.894×（1.1362/0.6） × P2 漏乘 1.667×（0.70/0.42）** ⇒ 1.894×1.667 = 3.16 ✓。

### L25.4 各段重算（逐臂表见 design §11.2/§11.3）
| 段 | 步数 | @N | GPU-h |
|---|---|---|---|
| A 探针（§10 分档 P0+P1+P2，6 剂量） | 6.0M | 31,000 | **1.8937**（@8,192 = 1.1025） |
| A′（§9.2 的"A 探针"变体：全 6 剂量，18 run） | 8.4M | 31,000 | 2.6511 |
| B 杠杆（4 轮 × 6 dose × 3 T × 1 seed） | 33.6M | 8,192 | **6.1740** |
| C 正式矩阵（6 dose × 3 seed × 3 T） | 25.2M | 31,000 | **7.9534**（@8,192 = 4.6305） |
| C′（N1：5 dose） | 21.0M | 31,000 | 6.6278（@8,192 = 3.8587） |
| D 确定性（2 重复 × 3 T） | 2.8M | 31,000 | **0.8837** |

### L25.5 🔴 **结论翻转**：§9.2 的「≈8.87 GPU-h，落在包络内」**不成立**
- 实测合计 = **1.8937 + 6.1740 + 7.9534 + 0.8837 = 16.905 GPU-h ⇒ 超 ≤10 包络 69%**。
- §9.2 的分段算术本身**无误**，**错在假设**（ms/step=0.6、以及 B 段 0.3）⇒ 该行结论必须作废（已在 design §11.4 显式作废并保留旧值）。
- **对主agent 校核的回复（按派工要求"以我的表为准并指出错在哪"）**：
  1. **探针 1.894 ✅ 正确**（与我逐位一致）；
  2. **C = 8.4 GPU-h ❌ 应为 7.9534**（差 +5.6%）；可能来源 = 我 §9.2 的 **A 行步数 "8.4M 步"** 被当作 C 段 GPU-h（或 4.2×2 缩放）；
  3. **漏项**：B 段 **6.1740** 与 D 段 **0.8837** 未计入 ⇒ 故 "≈10.3" 应为 **≈16.90**。**不是"略超"，是 +69%。**

### L25.6 菜单代入与建议（N2/N3 不采用）
| 方案 | GPU-h | 包络 |
|---|---|---|
| 原方案 | 16.905 | ❌ +69% |
| N1 单用（6→5 档） | **14.487** | ❌ +45%（**R21 记的 ≈11.8 与本表不符**：本表按"全段 5 档 + 三段齐全"算；若其 11.8 只算探针+C 而漏 B/D，则同 L25.5 的漏项） |
| N4 单用（C 改 8,192） | 13.213 | ❌ +32% |
| N4（探针亦改 8,192） | 12.421 | ❌ +24% |
| N4 + N1 | 10.584 | ❌ **+6%（仍超）** |
| N4 + N1 + B 轮数 4→3 | 9.298 | ✅ |
| **★N4 + N1 + B 只做 `T_min` 单档搜索（4 轮）** | **6.174** | ✅ **余量 38%** |

**建议采用 ★**：① 单项与 N4+N1 都不足以回到包络内；② **B 段（杠杆搜索）本不需要 T 阶梯**——§5 的钳制是"每次改动重核 (c)"，而 (c) 的**定稿阶梯**属 C 段责任 ⇒ 搜索只在 `T_min` 单档做、C 段对**最终配置**做三档确认，**方法学上更正确**；③ 余量可吸收占空比损耗、`T_min` 上抬（B 4.0M→8.0M 步 ⇒ +0.735 ⇒ ≈6.9）、或一次额外 B 轮。
**兜底（若 8,192 档 (a)∧(b) 不可达而 31k 可达）**：31k 下 N1 + B 单档 4 轮 = **10.605（+6%）**；3 轮 = **10.289（+3%）**；**2 轮 = 9.973 ✅**；**3 轮 + D 并入 A = 9.405 ✅**。

### L25.7 改档前提（已预注册，见 design §11.6）
1. **探针段必须与 C 段同档**（探针职责含"定 `T_min`"与"(a)(b)(c) 首读"，二者都依赖 N；跨档会使 `T_min` 决策不可传递）；
2. **档位是 P5 协议的一部分 ⇒ 显式登记**（8,192 属 RF-M11-6 默认档内 ⇒ 合法；N_full 仍只在 E 行进）；
3. **任何档上都要重验 (a)(b) 是否同时可达**（工作点随 N 变）——本方案下首读即在 8,192，无口径断层。

### L25.8 教训补充
13. **"预注册假设"必须标注为假设并给替换点**：§9.2 用 0.6 ms/step 是**清单预注册值**，但报告里写成"落在包络内"的结论 ⇒ 一旦实测到来结论翻转。**正确写法**应是"在预注册假设下 ≈8.87；**实测后以 §11 为准**"（本节点补做的正是后者）。
14. **分段预算必须逐段互检总数**：本次两处漏项（我的 P2 漏乘、主agent 漏 B/D）都能被"**逐段列出步数并求和**"一步抓住 ⇒ 与"双估计器一致"同族：**总步数（Σ段）与总 GPU-h（Σ/3.6e6）必须双向核对**。
15. **换算因子的自检方式**：用**同一个算例同时代两种因子**并看比值是否等于已知的单位比值（12.96）——比逐项推导更快发现量纲错（本次 30 秒定位）。

## L26 — R34 更正：**解释器口径**（我此前把引擎侧命令写成 `.venv-m11`，是错的）

### L26.1 事实（主agent 实测 + **本节点独立复验**，逐项一致）
| venv | python | torch | pytest | h5py | numpy |
|---|---|---|---|---|---|
| **`.venv-m9`** | 3.9.6 | **2.8.0（MPS）** | **8.4.2** | MISSING | **1.26.4** |
| `.venv-m10` | 3.9.6 | MISSING | MISSING | 3.14.0 | 2.0.2 |
| **`.venv-m11`** | 3.9.6 | **MISSING** | MISSING | 3.14.0 | 2.0.2 |

复验命令：`./<venv>/bin/python -c "import sys,numpy,torch,h5py,pytest"`（逐项打印版本/MISSING）⇒ `.venv-m11` 的 torch **MISSING**、`.venv-m9` 的 torch **2.8.0** ✅。

### L26.2 我的错误与更正
- **错在哪**：本文档 **L1「命令前缀」** 与 `m11_b3_g1_managed_jobs.json::common_env.python` 把**单一解释器**写成 `.venv-m11`；
  J3 的「每个 (config,mechanism,dose,seed,T) 新进程」一行是**引擎侧**（生产臂）⇒ 用 `.venv-m11` 会**立即 `No module named torch`**。
  **根因**：我把"本节点一直在用的解释器"（评估器/数据侧确实用 m11）**外推**到引擎侧 —— 属**跨口径外推**，与第四律同一家族（口径维度 = 解释器）。
- **更正**：
  - **引擎侧（任何 `import torch`／跑仿真，含 P5 跑臂生产者）⇒ `./.venv-m9/bin/python`**（依据 `data/m11_engine_params.csv::cpu_interp` + 清单 §0.10 L5）；
  - **评估器侧（`validate_p11_resting.py`）⇒ 可留在 `./.venv-m11`**（numpy-only），但**必须显式标注"评估器与生产者使用不同解释器"**；
  - 数据侧 ⇒ `./.venv-m11`（h5py 3.14.0；`.venv-m10` 等价）。
- **落盘**：`jobs JSON` 的 `common_env` 改为**按角色三分**（`python_engine_side`/`python_evaluator_side`/`python_data_side`）+ 新增 `interpreter_caliber_R34` 块；
  **旧值逐字保留**在 `r34_correction_note.old_values_verbatim`（含旧 `common_env` 与三条被更正的命令行 + 旧注释）；
  design 文档 §8 的旧 `PY=./.venv-m11/bin/python` 行**逐字保留**并就地追加【R34】标记；**§12** 给出更正后的逐档命令（P0/P1/P2，含 `ENG`/`EVA` 两个变量）与 §12.3–§12.5。

### L26.3 ⚠️ numpy 偏斜 ⇒ 「逐位/BIT」结论的**解释器维度**
- `numpy 1.26.4`（.venv-m9）vs `2.0.2`（.venv-m11/.venv-m10）：`pop_trace.npy`/`counts.csv` **格式兼容可读**；
- **但「逐位一致 / BIT」类断言只在"同一解释器 + 同一 numpy 版本"内才可比** ⇒ **评估器侧的数值结果不得用于支持或推翻任何「逐位」结论**（第四律「口径一致才可比」的**解释器维度**）；
- **受影响场合**：工具的 `(h)` 确定性判据、任何 `bit_identical`、**G0 的 (e) 分段可组合性 / 跨进程逐位** ⇒ 一律由**引擎侧生产者**给出，评估器只做统计级核对（与判据表 `determinism_mode=statistical` 一致）。
- **留痕（不改工具）**：评估器把 protocol-json 原样回写进输出 `protocol` 段 ⇒ 在 `m11_p5_protocol.json` 写 `engine.python/numpy/torch` 即自动留痕。
- **每臂 JSON 追加**：`host/device` 之外必须含 `python`/`numpy`/`torch`（R34 §四）。

### L26.4 教训
16. **"我一直在用"不是口径依据**：解释器/依赖版本与"物理档、T/settle、估计器实现"同级，**必须按角色逐项声明并实测**（本次用 30 秒的三 venv 复验即定案）。
17. **BIT 类断言的适用域必须显式收窄**：跨解释器/跨 numpy 版本**不是"差不多"，而是"不可比"** ⇒ 这类断言应在设计与 NOTES 里写明**只由哪一侧给出**（本次：引擎侧）。

## L27 — R43 双口径重定价（**判据口径 = step**；§11 的 ★ 方案作废）

### L27.1 作废原因（口径不互引）
§11/R21/R36 用的 **`1.1362 ms/step`（N=31k）** = **`chunk` 投递 + 1/4 事件率** ⇒ **与判据口径（`step`）不可互引**。
B2 孪生臂（同种子/同网/同事件表, N=31,000, dose=1）：**`step` = 5.4163**（wall 180.7 s）vs **`chunk` = 2.0955**（111.7 s），该点 **`ndiff=0`、`pop_trace` 逐位相同**。
三个比值：**step/chunk = 2.585**；step/旧 = **4.767**；**chunk/旧 = 1.844**（**这 1.844 来自事件率 1/4，不是投递口径** ⇒ 旧值同时叠了两种差异）。

### L27.2 🔴 我的第二条自纠：**§11 的步数表漏了 settle**
B2 生产者实测单 run **210,000 步 = 10,000 settle + 200,000 测量** ⇒ **§11.2/§11.3 只算测量窗**：T=10 s 少 **5.0%**、T=20 s 少 **2.5%**、T=40 s 少 **1.25%**。
§11 表格逐字保留；§13 全部含 settle（T=10 s → 210,000；20 s → 410,000；40 s → 810,000）。
**P0 单段核对**：我算 **0.9479 GPU-h = 56.9 min**，与主agent 的 **≈0.95 / ≈57 min 一致 ✅**。

### L27.3 逐段重算（★方案：5 档、B 单档、D 独立）
| 段 | run | 步数 | **step** | **chunk** |
|---|---|---|---|---|
| P0（T=10 s, dose{0,1,4}, seed0） | 3 | 630,000 | 0.9479 | 0.3667 |
| P1（T_min, 5 档, seed0） | 5 | 1,050,000 | 1.5798 | 0.6112 |
| P2（T{10,20,40}, dose{0,1,4}, seed0） | 9 | 4,290,000 | 6.4544 | 2.4971 |
| B（★单档, 4 轮×5 档） | 20 | 4,200,000 | 6.3190 | 2.4447 |
| C（5 档×3 seed×3 T） | 45 | 21,450,000 | **32.2721** | 12.4857 |
| D（2 重复×3 T） | 6 | 2,860,000 | 4.3029 | 1.6648 |
| E **孪生臂校核（R43 硬前提）** | 4 | 840,000 | （全 step 方案不需） | **0.8764** |
| **合计** | **86** | **34,480,000** | **51.8761** | **20.0702（含 E）** |

### L27.4 **≤10 GPU-h 包络：不成立（两种口径都不成立）**
- **step 口径 @31k：51.876 GPU-h ⇒ 超包络 419%（5.19×）**；
- **chunk 口径 @31k（含 E 段 0.8764）：20.947 ⇒ 仍超 109%（2.09×）**。
⇒ **§11 的 ★ 方案（6.174）彻底作废**；单靠换口径或换档都回不去。

### L27.5 N=8,192：**step 口径无实测 ⇒ `unverified`（不得默认外推）**
两个候选档因子差 **2.20×**：0.25（B2 估"约 1/4"）vs 0.5822（旧表**同口径**比值）——**后者只在 chunk 口径内可比，不能搬到 step 口径**。

| 档因子 | 依据 | step 合计 | chunk 合计 |
|---|---|---|---|
| 0.25 | B2 估 | **12.969** ❌ +30% | **5.018** ✅ |
| 0.2643 | 线性 N 比 | **13.709** ❌ +37% | **5.304** ✅ |
| 0.5822 | 旧表同口径比值 | 30.202 ❌ | 11.685 ❌ |

**替换点**：先跑 **P0′（N=8,192、step 口径、3 run，估 0.24–1.0 GPU-h）**，用实测替换后再给 C 段定稿价；**拿到之前 C 段不得启动**。

### L27.6 R43 两条约束已并入
1. **判据口径 = step（默认）**；**chunk 仅作生产口径**，前提 = **剂量网格两端各一对 step/chunk 孪生臂且 `pop_trace` 逐位相同**
   （**E-M10-1 的失效恰是口径相关的 ⇒ 单点等价不足以外推整网格**）；**任一对不逐位 ⇒ 全网格退回 step 并重定价，不得只换那一格** ⇒
   故 **E 段 0.8764 GPU-h 是 chunk 路线的前置成本**（已计入 §13.2）。
2. **T_min 定档规则不变**（`S0 ≥ 300 ⇒ 10 s`，否则 20/40 s）；**T_min 上抬须重算**：阶梯 {20,40,80} ⇒ 每 (dose,seed) = 410k+810k+1,610k = **2,830,000 步**，比 {10,20,40} 的 1,430,000 步 **×1.98**。

### L27.7 建议（二选一，均需先测 8k 的 step ms/step）
- **A 提高 G1 包络**：保留完整计划 + N=8,192 ⇒ **≈12.97–30.20**（取决于实测因子）⇒ 须把 §10.1 的 **≤10 上限正式改写**（NG 三态 + 新增行）；
- **B chunk 生产 + step 判据**：先跑 E 段（0.8764）⇒ 通过后 8k 下 chunk ⇒ **≈5.02–11.69** ⇒ **唯一有余量的路**，但依赖两端点逐位一致；
- C（P0+B 1 轮+C，D 并入 P0）⇒ 8k×0.25 = **8.70** ✅ / ×0.2643 = **9.20** ✅ / ×0.5822 = 20.26 ❌ ⇒ **零余量、B 仅 1 轮 ⇒ 不推荐**。
**本节点建议**：**先跑 P0′（8k, step, 3 run）+ E 段（两端点孪生臂）**⇒ 两个实测值同时钉住"档因子"与"口径可比性"，合计 <2 GPU-h，避免把 30–52 GPU-h 押在未实测因子上。

### L27.8 占位符落实
`<B2 的 P5 跑臂生产者>` ⇒ **`tools/run_m11_p5_arms.py`**（B2 已交付并端到端自检：三估计器 `rel_diff=0.0000`、`dim_check=0`、`hard_fail=[]`）。
**§12.5 的占位行逐字保留**并就地加【R43】标记；**可运行版命令见 §13.7**（含 `P0′` 与 `E 段` 两条新命令）。

### L27.9 教训
18. **"ms/step" 不是单一量**：它随**投递口径（step/chunk）× 事件率 × 档位 N** 变化 ⇒ 每次引用必须写全三元组；本工作包连续两次（R34 解释器、R43 口径）都在"未写全口径的单一数字"上出错。
19. **步数口径要把 settle 算进去**：`steps = settle/dt + T/dt`（本次漏 5%；T 越长占比越小，故**短 T 的定价误差最大**）。
20. **档因子与口径因子不得相乘外推**：8k 的 step 值需**独立实测**（两个候选因子差 2.20× ⇒ 任何默认比例外推都可能把预算差 2 倍）。

## L28 — G1 终判后的两项口径核查（任务 A）+ 工具修复 + 任务 B 预注册

### L28.1 任务 A 结论（(c) 的措辞问题）——**行状态与门条件并非同一行，且 p5_C 不是那条消息的来源**
- **引用的"1 seed"消息来自 single-seed 配置**（`p5_B1/B2/B3/B4/P1/P2`，各 5–9 臂、只有 seed 0），**不是 p5_C**。
  证据：`m11_p5_resting_eval_g1_v2.json` 的 11 条 `p5_c_selfsustain` 行里，6 条来自上述配置（`n_seeds=1`、`unavailable`），
  **p5_C 的 3 条（T=10/20/40 s）是 `n_seeds=3`、`zero_spike=25773/52506/110334`**（3 seed 池化）**且状态 = `fail`**（R=0.2802/0.2857/0.2998；CI 上界 0.2841/0.2885/0.3018 > 0.12）。
- **代码定位**：
  - `units = sorted({(a.config_id, a.T_ms, a.mechanism) ...})`（**L610**）⇒ 评估单元 = config × **精确 T_ms** × mechanism；
  - `_unit_arms(cfg, T_ms, mech, role)`（**L638–641**）⇒ 按 `config_id` + `abs(T_ms-unit_T)<1e-9` + `mechanism` + `role` 过滤；
  - `nz = len({a.seed for a in zero_arms if a.seed is not None})`（**L771**）、`nb`（**L772**）⇒ **`n_seed_zero` = 该单元内 `role=="zero_drive"` 臂的"声明 seed"去重个数**；
  - 样本量门（**L781–790**）⇒ 不满足 ⇒ `status="unavailable"`；`_verdict` 的判据级聚合（**L1178–1186**）：任意行 `invalid` ⇒ invalid，否则任意行 `fail` ⇒ **fail** ⇒ 门读判据级。
- ⇒ **"行状态 vs 门条件不同"是跨行聚合，不是同一行换规则**：p5_C 三行 = `fail`，B/P 配置六行 = `unavailable` ⇒ 判据级 = `fail`。**该规则已写入工具**（`_verdict` 注释 + 输出新增 `criterion_row_status_histogram` 与 `aggregation_rule`）。
- **(3) 修正后 (c) 的正确状态 = `invalid`（不是 fail）** —— 见 L28.2 的复刻轴发现：`R=0.2802` 远超上限**但 3 个 seed 是同一实现** ⇒ RF-M11-3 的"≥3 独立 seed"**实质上未满足** ⇒ 按四态语义（维度/编码路径未生效）判 **`invalid`**，点估计/CI 仅作描述性。**若主agent 选择按"声明 seed"读协议，则为 `fail`，但措辞必须带"3 seed 名义、有效 1"。**

### L28.2 🔴 复刻轴失效（矩阵级，第 5 次"字段存在 ≠ 生效"）
- **实测**：同一 `(config,N,T,dose)` 下 `seed∈{0,1,2}` 的臂 **`pop_trace` 逐位相同（15/15 组）且 `counts.csv` sha256 全同（15/15）**；抽样：`c_8192_T10000ms_d0.0` 三个 seed 的 sum 均为 **8591**，T=20 s 均为 **17502**，T=40 s 均为 **36778**。
- **根因（生产者码级）**：`run_m11_p5_arms.py` **L124** `build_laminar_probe_column(n, seed=int(spec["net_seed"]), ...)`（默认 0）、**L151/155** `event_seed`（默认 11）、**L157** `eng.reset(seed=spec["seed"])` ⇒ **计划只改了 `seed`，而 `net_seed`/`event_seed` 全程常量 0/11** ⇒ 网络与事件表完全相同 ⇒ 有效重复数 = 1。
- **后果**：(c) 的 CI（[0.2763,0.2841]，宽 ±1.4%）由**计数噪声**撑出、统计上无效；(h) determinism **平凡通过**；≥3 seed 门名义满足。**这条同时解释了为什么 T 越大 R 越大（0.2802→0.2857→0.2998）却每个 seed 都一样。**
- **工具加固（已实现）**：新增 `_effective_seed_check()` —— 同一单元内**声明 seed 不同**的臂若**强指纹**（完整轨迹/逐神经元计数数组）**逐位相同** ⇒ 记 `seed_dimension_inert` ⇒ 该子判据判 **`invalid`**（不判 fail）。**聚合标量（sum）不作指纹**（两个独立实现的脉冲总数可偶然相同；冒烟 S2 的 275/275/275 即此类）⇒ 证据不足时记 `insufficient_data` 而不误报。

### L28.3 工具修复第二项：截断守卫**降级**（假阳性，曾把整轮判成 invalid）
- **旧行为**：短档轨迹是长档轨迹的逐位前缀 ⇒ 阶梯 `invalid` ⇒ 整轮 `invalid`。
- **实测为什么错**：p5_C 的 T=10/20/40 s 是三组**独立进程**（pid 55458 vs 55710、finished_at 不同），但**在确定性同 seed/同事件表下，短档轨迹必然**是长档轨迹的前缀（且因 L28.2 的复刻轴失效，连不同 seed 也互为前缀）⇒ 27 对"前缀"**全部是预期现象**。
- **新行为**：前缀同一性只记 `prefix_identity_observed`（informational，附说明），**不改变阶梯状态**；**截断只在"轨迹长度 ≠ round(T_ms/dt_ms)"时才可证**（该检查已是 `trajectory_len_mismatch` 硬失败）。⇒ 修复后 p5_C 阶梯 = **`pass`**，`hard_failures=[]`，整轮 `invalid` 的**唯一原因**变成"判据级 invalid = p5_C_selfsustain"（理由准确）。
- **回归**：S1 `unavailable`/3 · **S2 `fail`/1（恢复）** · S3 `unavailable`/3 · S4 `invalid`/2（硬失败 1 条：×1.15 双估计器）· S5 `unavailable`/3（不再是 invalid，语义变更已登记）· S6 `unavailable`/3；全部 `criteria_per_unit=[10]`。
- **sha256（冻结纪律）**：修复前 `7b3ed467aaea56cc2f8d2760285fa8a600efdd761baadf8082ff30784504757a`（89,482 B / 1,495 行）
  ⇒ 修复后 **`5eb76edf58e3c0ec`…（完整 sha256 见下）**（95,375 B / 1,570 行）；备份 `/tmp/validate_p11_resting_pre_R_fix.py`。
  中间态（加了两项修复但未含最后两处微调）`be7eaf0e8f6d78d8dcd6b57a6e44dc99ab8c9fea3f18518b03b442bea7e87f4e`（94,950 B / 1,568 行）。
  **未改写任何已判定产物**：全量复评输出到新文件 `m11_p5_resting_eval_g1_v3.{csv,json}`（v2 原样保留）。

### L28.4 🔴 判据表 CSV 与 JSON **不同步**（非本节点改动；`--check-targets` 已捕获）
CSV 101 行（mtime 10-13 04:41）/ JSON 114 行（mtime 10-12 07:21）；**值不一致 2 行**：
`silent_frac_calcium_band_lo/hi` = CSV `0.4800/0.4900`（**final=1**）vs JSON `unavailable`（**final=0**）。
JSON 另含 13 行把同一件事标得更严：`silent_frac_calcium_zero_count_proxy_field4_measured=0.4900`、
`silent_frac_calcium_fullfov_frame_unit_field4_measured=0.8915`、`rate_median_hz_measured_calcium_band_status=per_neuron_unavailable(proxy_not_transferable)`、
`silent_frac_calcium_two_caliber_scope_note=calcium_not_comparable_to_ephys`、`silent_frac_calcium_coverage_register=covered=7; missing=1(real_ROI_mask)`。
**影响**：评估器读 **CSV** ⇒ G1 的 (b) 用 0.48–0.49 判 `fail`；但按 JSON 的严格语义（proxy、不可迁移、**缺真实 ROI 掩膜**）**不满足本工作包 J2 预留行的验收条件 e)**（"ROI 管线不可得 ⇒ 判 unavailable 并写明已覆盖维度"）⇒ **(b) 应为 `unavailable`，不是 `fail`**。⇒ 请主agent 裁决以哪份为准。

### L28.5 任务 B：预注册已落盘 `data/m11_p5_lever_prereg.json`
- **前置阻塞项**：复刻轴修正（L28.2）——**不修则一切杠杆实验的 CI 与 (c)/(h) 都无效**。
- **杠杆顺序（按 §5 钳制）**：**L0-a 输入通路增益 `gmax_scale`↑**（零驱动臂无外部事件 ⇒ 只抬分母 ⇒ R↓）→ **L0-b 内在兴奋性 `bias_scale`↓** → **L1 阈值异质性 `bias_log_sigma`↑（形状杠杆：同时抬中位率与静默 ⇒ n_both_in_band≥1 是本轮可证伪断言）**；**禁**提高 λ/dose 上限（G1 已测 ρ=±1.000、n_both_in_band=0）。
- **判据**：(c) 每档改动后必须重核（R≤0.10 ∧ 更宽 CI 上界 ≤0.12）；(a) ∈[0.5,2.0]；(b) 见 L28.4 的裁决；统计必须用 **net_seed 复刻**的 ≥3 独立实现；生产 chunk / 判据 step。
- **停机规则 6 条**（含 L0 全失败即停机、L1 后仍 n_both_in_band=0 ⇒ 判张力结构性并入缺失机制清单）。
- **GPU-h 预估（用实测单价，见 `m11_p5_g1_pricing.csv`）**：R1 L0 扫描 36 run = **7.016（step）/ 1.802（chunk）**；R2 L1 扫描 27 run = **5.262 / 1.352**；R3 定稿确认 45 run = **19.803 / 5.179**；R4 新工况孪生臂 4 run = **0.789 / 0.190** ⇒ **合计 32.87（step）/ 8.52（chunk）** ⇒ **step 口径超 ≤10 包络 3.3×；chunk 生产 + step 判据（端点孪生臂）8.52 ⇒ 可行（包络内）**。
- **探针（诚实登记）**：B3 自跑的 2 臂探针（net_seed 1/2）**被 QoS 饥饿回收**（墙钟 32:59、累计 CPU 1:52.9 ⇒ **占空比 5.7%**，零产物）⇒ 已终止并转为托管作业 **J7**；**复刻轴修正尚未被验证**（不得写成已验证）。

### L28.6 教训
21. **"声明了 N 个 seed" ≠ "有 N 个实现"**：工具的样本量门必须做**实质检查**（本次加"强指纹逐位相同 ⇒ 维度未生效 ⇒ invalid"），否则失效维度会以**过窄 CI**的形式伪装成"统计可判"——这比"样本量不足"更危险。
22. **前缀同一性不是截断证据**：确定性同 seed 重跑下**必然**出现前缀；只有"长度 ≠ 声明窗"才可证。**凡"结构性异常检测"都必须先排除"确定性重跑"这一平凡解释**（否则假阳性会掩盖真结论——本次它把 fail 伪装成了 invalid）。
23. **自跑的探针必须先验占空比**：本次 33 min 墙钟仅 1m53s CPU（5.7%）⇒ 若按 `%CPU` 判断会误以为"在正常跑"；**任何自跑作业超过 10 min 一律终止并转托管**。

### L28.7 工具最终 sha256（供主agent 冻结核对）
`neural_exploration/tools/validate_p11_resting.py` = **`5eb76edf58e3c0ec65b3369e426e1c326479ce9205ebd30ce666478eb28581de`**（95,375 B / 1,570 行；现场 sha256sum 重算）。
> **勘误（同会话内自纠）**：本节初稿曾把 16 位前缀**人为补全**成一个完整 sha256（凭印象，违反「不得凭印象」纪律）⇒ 已用现场真值替换。**纪律（教训 24）**：哈希只能现场重算后整值誊写，**绝不可**由前缀补全。
> 冻结前值 = `7b3ed467aaea56cc2f8d2760285fa8a600efdd761baadf8082ff30784504757a`（89,482 B / 1,495 行，= HEAD `119477d` 的版本）。
24. **哈希不得由前缀补全**：本次我登记「最终 sha256」时把 16 位前缀补成了 64 位（凭印象）⇒ 发现后立即用现场 sha256sum 替换并自纠。哈希是取证物，只能整值现场重算（与「不得凭印象写文献值」同级）。

## L29 — (h) 确定性守卫（纪律第四十五条：「通过」必须能回答「它本来可能失败吗」）

### L29.1 问题
R93 给 (c) 加了 `_effective_seed_check()`，但 **(h) `p5_h_determinism` 仍是 `pass` —— 且是平凡通过**：
复刻臂与其源臂的 **(net_seed, event_seed, reset_seed) = (0, 11, 0) 完全相同** ⇒ 在确定性引擎下**必然**同轨迹
⇒ 该判据在此配置下**不可能失败** ⇒ `pass` **不构成证据**。与 R93 同源、同型遗漏。

**先例对照（要点）**：R78 的孪生臂"逐位相同"**有效**，是因为 **step/chunk 两个口径都可能不同**；
而 (h) 的"逐位相同"是**必然发生**的 ⇒ **无效**。

### L29.2 实现（最小侵入）
- `Arm` 增加 `net_seed / event_seed / reset_seed / delivery / sha256_of_trace / net_meta`（`net_seed` 优先取臂顶层，其次取臂内 `net.net_seed`——生产者把它落在 `net`）。
- `_eval_engineering` 的 (h) 分支新增**守卫**：对每个 `replica_of` 配对计算三元组；**三元组相同**或**无法判定（字段缺失 ⇒ 不能证明二者不同）** ⇒
  该行判 **`invalid`**（`insufficient` 语义）并写入 `self.guard_failures`；detail 里逐条记录两侧三元组、`triple_complete`、`sha256_of_trace`、`delivery`。
- `_verdict` 的 invalid 条件加入 `guard_failures` ⇒ **非零退出（exit 2）**；输出新增 `guard_failures` 与
  **`discipline_45_rule`**（与 `aggregation_rule` / `criterion_row_status_histogram` 同处声明）。
- **未改动**其余已冻结判据逻辑（样本量门 / CI / 双估计器 / 时长阶梯 / 张力 / 带纪律）。

### L29.3 反例自证（R84 第 37 条：未证伪过的断言不构成证据）——**三连**
| 夹具 | 三元组 | 数值 | 期望 | **实测** |
|---|---|---|---|---|
| **S7（反例/负对照）** | 相同 `(net_seed=0,event_seed=11,seed=0)` | 相同 | (h) 不得判 pass | **`invalid`**、`guard_failures` 1 条、**exit=2** ✅ |
| **S7b（正对照1）** | **不同**（复刻臂 `net_seed=1`） | 相同 | 守卫**不**命中 ⇒ 可判 pass | **`pass`**、guards=0 ✅（证明守卫**有判别力**，不是一刀切禁止） |
| **S7c（正对照2）** | 不同（`net_seed=2`） | **不同** | (h) 应判 fail | **`fail`**（"同参数重跑脉冲数不一致：7716 vs 6710"）✅（证明该判据**能**失败） |
夹具：`data/m11_p5_smoke_protocol_S7{,b,c}.json`（+ 输出 `m11_p5_smoke_out_S7{,b,c}.*`）。S7c 同时复现了硬门③（×1.15 双估计器 ⇒ hard_fail）。

### L29.4 修正后 v4 系列的 (h) 状态（**输出到新文件 v5，未覆盖 `g1_v4.*`**）
`m11_p5_resting_eval_g1_v5.{csv,json}`：
- **`(h) = invalid`**（全部 11 个评估单元；`guard_failures` 11 条：`d_8192_*_rep1 vs c_8192_*` 三元组相同）
- `(c) = invalid`（R93 守卫）｜`(b) = unavailable`（R93 §③ 的带裁定已生效）｜`张力 = unavailable`（(b) 不可判 ⇒ 随之）
- `gate = {c: invalid, e: pass, tension: unavailable}` ⇒ **verdict = `invalid`、exit=2**（未通过，但措辞为"口径缺陷"而非反证）
- `hard_failures = []`（阶梯已恢复 pass）；**回归 S1–S5 与修复前一致**（无 replica 的用例 (h)=`unavailable`、guards=0）⇒ 守卫无副作用。

### L29.5 哈希（现场重算；纪律：不得由前缀补全）
| 时点 | sha256 | 大小 |
|---|---|---|
| 本次修改前（= R93 后） | `5eb76edf58e3c0ec65b3369e426e1c326479ce9205ebd30ce666478eb28581de` | 95,375 B / 1,570 行 |
| **本次修改后** | `f99d5c6cc71beddd3ec7a74d988ceb2d8863a017399feb0a7e4ff5e94c2cf7ae` | 99,312 B / 1,615 行 |

### L29.6 教训
25. **"必然发生"的现象不能当证据**：与"可复现 ≠ 正确"（L34/M10）并列——**"必然复现"更不是证据**。
   判据设计时应对每条断言追问：**在这个配置下，它本来可能失败吗？** 不能 ⇒ 该判据只能标 `invalid/insufficient`。
26. **同型遗漏要成对扫描**：我为 (c) 加了复刻轴守卫后，(h) **同源同型**却被漏掉 ⇒
   以后加一条"不可能失败"类守卫时，**必须把同一根因的所有判据一并扫一遍**（(c)/(h)/孪生臂/determinism 家族）。

## L30 — R96 §① (d) 单调口径修正 + R97 §② (h) 登记语义

### L30.1 (d) `p5_d_drive_elasticity`：**聚合层次错位**（确为口径问题，非数据问题）
- **定位**：旧实现 `pts = sorted({(drive_mult, rate) ...}, key=lambda t: t[0])` —— 集合内**同一剂量下的多个 seed 顺序任意**，
  随后 `monotone = all(b >= a)` 在**这 15 个池化点**上判 ⇒ **seed 间离散本身就能破坏单调性**。
- **实测违反对（p5_C, T=10 s）全部是"同剂量内"**：`(0.0, 0.119653) → (0.0, 0.104871)`、`(0.5,…)→(0.5,…)` 等 **5 处**（每剂量 1 处）；
  而**剂量级中位数严格递增**（见下）。
- **修正**：判定序列 = **剂量级聚合值**（主口径 **中位数**，并列均值），dose 排序后判非减；旧口径保留为 `monotone_pooled_15pt`（**仅审计，不参与判定**）。
  **注释已写入工具**（`_eval_elasticity` 内，"聚合层次错位"说明）。新行字段：`monotone_dose_median / monotone_pooled_15pt / dose_level_median_rates / dose_level_mean_rates / dose_grid`。
- **修前 / 修后（同一 `m11_p5_protocol_g1_repl_cd.json`）**：

| T | value | 修前 status | 修后 status | 剂量级中位序列（修后） | 旧口径 |
|---|---|---|---|---|---|
| 10 s | 1.2819 | **fail**（唯因单调=False） | **pass** | 0.104871 / 0.260559 / **0.373218** / 0.558240 / 0.854102 | False |
| 20 s | 1.3109 | fail | **pass** | 0.106989 / 0.260974 / 0.373877 / 0.558801 / 0.850781 | False |
| 40 s | 1.2988 | fail | **pass** | 0.107785 / 0.262830 / 0.374066 / 0.558575 / 0.848215 | False |

  ⇒ 序列**严格递增**（五档、每档 3 seed，剂量网格 `[0,0.5,1,2,4]`）；value 均 ≥0.3 下限。输出文件 **`m11_p5_resting_eval_repl_v6.{csv,json}`**（**未覆盖** `m11_p5_resting_eval_repl.*`）。

### L30.2 (h) 登记语义（R97 §②1/§②2/§②3）
- **(h) = 登记项**：判据表新增行 **`p5_h_determinism_semantics = not_applicable_to_p5`**（append-only；CSV 103 行 / JSON 115 行）；
  工具输出把 (h) 行标 `evidence_class=registration_only(not_applicable_to_p5)`、`p5_role=registered_not_judged`，reason 首句注明"属 G0 性质、P5 只登记"。
- **D 段复刻臂 seed 约定**（写入 `m11_p5_lever_prereg.json::replica_axis_convention`）：**复刻臂 `seed` 必须与源臂相同**；
  并区分**两种"复刻"**：① 独立实现轴（**net_seed 变**）用于 (c) 的 ≥3 seed 统计门；② 位一致轴（**seed 同**）用于 (h)/G0 的跨进程确定性登记。
  本轮 repl 的 D 段用了 `seed=7`（源臂 0）⇒ **不同 seed ⇒ 必然不同** ⇒ (h) 判 fail（登记如实）；下一批计划恢复同 seed 后才有意义。
- **守卫保留**（不改 `(net,event)` 对形式，理由 R95 §⑤：B2 的计划级断言管计划缺陷，本守卫管"该行不构成 P5 证据"，**各司其职**）；
  **作用域界定**：守卫命中仍写 `guard_failures`（审计）+ 该行 `invalid`，但因 (h) 已是**登记项**，**不再驱动整轮 verdict**
  （连带语义变化：冒烟 **S7 由 `invalid`/exit2 变为 `unavailable`/exit3**，而 **(h)=invalid、guards=1 照旧** —— 已登记）。
- **repl v6 终态**：`gate = {c: fail, e: pass, tension: unavailable}` ⇒ **verdict = `fail` / exit=1**，`hard_failures=[]`，`guard_failures=0`
  （D 段 seed=7 ⇒ 三元组**不同** ⇒ "不构成证据"守卫不适用；该行 fail 是**如实登记**的位不一致）⇒ **与主agent R97 §③ 的 G1 终态 `fail ∧ pass ∧ unavailable` 一致**。

### L30.3 哈希（现场重算）
| 时点 | 工具 sha256 | 大小 |
|---|---|---|
| 本项修改前（= (h) 守卫版） | `f99d5c6cc71beddd3ec7a74d988ceb2d8863a017399feb0a7e4ff5e94c2cf7ae` | 99,312 B |
| **本次修改后** | 见 L30.4 | — |

### L30.4 工具最终 sha256
`neural_exploration/tools/validate_p11_resting.py` = **0d3c1d990dfe707720ec316b57ae2bb699181dae0d87042c920638ed9a4c48f3**（102,697 B / 1,649 行；现场 sha256sum 重算）。
判据表：`data/m11_p5_targets.csv` = **eb5695bda10888bb610e72c1dc402b793cb1468950ed907e212f58b979df2a17**（60,211 B / 103 行）；`m11_p5_targets.json`（115 行；**既有 CSV↔JSON 漂移仍在**，见 L28.4/B-1，未由本次变更引入）。

### L30.5 教训
27. **判"序关系"必须先把同一工况聚合到同一层再判序**：本次 (d) 把"同剂量多 seed"摊平成并列点 ⇒ 噪声被误读成"非单调"；
   **凡"单调/排序/凹性"类判据，先检查聚合层次**，并**并列报告两种口径**（主判定用聚合层，池化层仅审计）。
28. **同一根因的判据家族要成对扫描**（承 L29.6）：(c) 的复刻轴 ↔ (h) 的位一致轴是**两种不同的"复刻"**，
   **混用会造出"必然失败"或"必然通过"两种伪结论**（R95 改 seed ⇒ 必然 fail；R93 前 seed 惰性 ⇒ 必然 pass/过窄 CI）。

## L31 — L1（形状杠杆）预注册已落盘（R100）

- **基线改到 L0 最优点**：`bias_scale=0.70, gmax_scale=1.00`（R 0.2896→**0.0293** ≪0.1 ⇒ (c) 可满足）；
  **L0 的代价**：基准活动 −72%，而 (a) 在参考档已 fail ⇒ **同一个 bias 反向控制 (c) 与 (a)** = 结构性对立的具体形态。
  `bias_scale ≤ 0.50` **不再进 L1**（压过头：零驱动 19 spk、`i_exc_integral==0` ⇒ 近全静默、(e) 不可测）。
- **L1 档位**：`bias_log_sigma` ∈ {0.30(对照), 0.45, 0.60, 0.90}，**在 bias=0.70 基础上扫**；每档 `zero_drive` 与 `baseline` **各 3 个真正独立实现** ⇒ 4×2×3 = **24 run**。
- **🔴 核心可证伪断言（已按当前判据状态改写）**：
  **「存在某 σ 档使 (a) ∈ [0.5,2.0] Hz **且** R ≤ 0.10（点估计 ∧ 更宽 CI 上界 ≤0.12）同时成立」**。
  **为何自洽**：(b) 现为 `unavailable`（真实 ROI 掩膜维缺失）⇒ 把 (b) 写进门断言会造出**恒不可判**的断言（`n_both_in_band` 已算不出）；
  而 (a)/(c) 都是 `final=1` 的**已定稿判定带** ⇒ 双目标**可判且有判别力**（G1 已证二者各自可判但**不可同时满足** ⇒ 断言**当前为假**，可被证伪）。
  **证伪式**：σ 网格上 (a) 与 R **仍单调对立**（同号 ρ 且 |ρ|≥0.8，或双目标零命中）⇒ 判 **L1 失败 / 结构性对立**（R91 第 1 条升级为"形状杠杆亦不足"）。
- **(b) 改为描述性并列（不作 gate）**，按 R57 四条：代理口径（`zero_count`、无真实 ROI 掩膜）／样本 `thin`（单 field×2,000 帧≈317 s）／**不引用 0.48–0.49 窄带**（已被裁定 unavailable）⇒ 对 **0.49** 以**比值/上界**形式报告／**不判 pass/fail**。
  **并在预注册里明写**：**「L1 即便成功，也只能让 (a) 与 (c) 同时可行；(b) 的判定仍待真实 ROI 掩膜这一维」**，**不得**读成「L1 成功 = (b) 通过 = G1 可过」。
- **复刻轴改为非构造性形式**：**`(net_seed, event_seed)` 对**（realization0/1/2 = (0,11)/(1,12)/(2,13)），要求**同一档内 3 个实现的对两两互异**。
  旧三元组形式在 `seed` 惰性时**恒真**（只变 seed 就"不同"）⇒ **构造性断言，已弃用**（与 B2 的计划级判据同形；R95 §⑤ 分工不变）。
- **停机规则 8 条**（主断言成立即停扫／σ↑ 使 R 单升即停／连续 2 档无改善即停／`hard_failures` 或对数断言违反即停／`i_exc_integral==0` 单臂 invalid、≥2 实现则整档 invalid／触包络即停／bias≤0.50 不进／σ=0.90 仍不达标 ⇒ 升级到"换机制"）。
- **预算（实测单价）**：扫描 24 run **1.115**（上界 1.361）＋定稿确认 45 run **4.745**（上界 5.791）＋新工况孪生臂 4 run **0.490**＋第二轮 σ 精扫 12 run **0.557** ⇒ **合计 ≈6.907 GPU-h（上界 8.322）⇒ 在 ≤10 包络内**（余量 1.7–3.1）；**墙钟 ≈8.0 h**（纯计算 6.9 h＋逐 run 新进程开销 ≈1.1 h），**必须托管**，进度按累计 CPU/墙钟比判。
  **降档（若超包络）**：④ **推荐** 定稿确认只跑 3 剂量 ⇒ 总 **≈5.01**（代价：(d) 网格不全，因 (d) 为 sanity 非 gate 可接受，须登记）；① 去掉 3.0× 档 −0.279；②/③ 违反硬门④/RF-M11-3 ⇒ **不推荐**。
- **B2 阻塞前置**：`bias_log_sigma` **未在 CLI 暴露**且计划模板的臂 key 组（`run_m11_p5_arms.py` **L456/L498**）不含它 ⇒ 需 B2 加 `--bias-log-sigma` 并并入 `arm_spec_from_cli`（**L226–237**）与计划模板；`net_kw` 过滤器（**L120**）已支持该键 ⇒ 临时可用 `--arm <spec.json>` 逐臂传（无需改码）。
- **边界遵守**：工具 sha256 仍 `0d3c1d990dfe7077…`、判据表 CSV 仍 `eb5695bda10888bb…`（均未改）；未改写已判定产物；**未启动任何 MPS 作业**。

## L32 — L2（活动历史依赖机制）设计与最小阳性对照（R105 派工）

### L32.1 候选与选择（含「为何不点燃自持」的机制论证）
| 候选 | 引擎通路 | 为何**不**点燃自持 | 优先级 |
|---|---|---|---|
| **C1 慢适应/慢负反馈变量（E2b）** | `mouse_cortex_engine.configure_slow(tau_slow_ms, inc_per_spike, e_slow)`（**L334–353**，含 `inc_per_spike>0` 构造性守卫 + 读回断言）；轨迹汇报 `res["slow_trace"]`（**L858–866**） | s_i **只在自身发放时增加**、按 τ_slow 衰减、**以超极化注入**（e_slow=−77 mV）⇒ **纯负反馈**：零驱动下不发放的单元 s_i 恒为 0 ⇒ **不产生额外活动**（与 σ↑ 的**静态**低阈值本质不同）。**实测方向确认**：slow off spk=598 → on(inc=180) spk=101（**−83%**） | **首选** |
| **C2 STP depress（突触使用依赖）** | `larval_zf_engine.configure_stp(...)`（**L69–79**）+ `_apply_stp_and_deliver`（**L936–945**）；`stp_enabled` 默认 False ⇒ **生产者从未调用** ⇒ 当前 **OFF** | 资源 `x` 随**该突触使用**耗竭 ⇒ **使用依赖负反馈**；无发放时 x→1 ⇒ 不制造活动。**只允许 depress**；**facilitate 是正反馈** ⇒ 会点燃自持 ⇒ 作**负对照臂** | 次选（需 B2 暴露） |
| C3 阈值随活动漂移 | 引擎未暴露（慢变量不进阈值） | 与 C1 同族 | 低 |

**重要口径**：`slow=True` **本来就是默认开启**（生产者 **L124–126**：τ=1000 ms、inc=6.0）⇒ **既有 G1/L0/L1 全部结果都在 slow 默认档下取得** ⇒ L2 是**放大**该机制（inc/τ），不是首次启用。
**禁用混用**：M10「STP 抑制型不能产生成像量级慢结构」是 **P6（慢结构）** 的判据，与 L2 的工作点目标 (a)∧(c) **不同** ⇒ 不得互相否证。

### L32.2 核心可证伪断言 + 证伪式
**主断言**：存在某档使 **(a) ∈ [0.5,2.0] Hz ∧ R ≤ 0.10**（点估计 ∧ 更宽 CI 上界 ≤0.12）同时成立（(b) 现为 `unavailable` ⇒ 不进任何门断言；(a)/(c) 均为 `final=1` 已定稿带 ⇒ 可判且有判别力）。
**证伪式**：① 全网格双目标零命中 **且** (a) 与 R **同号单调**（|ρ|≥0.8）⇒ **L2 失败/第三维结构性对立** ⇒ 升级换机制；② **机制专属**：R 随 `slow_inc` ↑ **反向升高** ⇒ 机制点燃自持 ⇒ 立即停；③ `slow_trace.alive=False`/`std≤1e-6`/读回不一致 ⇒ 该档 **`invalid`**（**不得**记『机制无效』）。

### L32.3 档位表（基线 = L0 最优点 bias=0.70 **且 σ=0.30 未启用 L1**）
S0 `inc=6`(默认对照) → S1 `inc=18`(×3) → S2 `inc=60`(×10) → S3 `inc=180`(×30，阳性对照已用) → **S4（条件档）`inc=180, τ=3000`**（仅当 S1–S3 方向正确时进入）。每档 `zero_drive`/`baseline` **各 3 个真正独立实现**（`(net_seed,event_seed)` 对 = (0,11)/(1,12)/(2,13)，**两两互异**、非构造性）⇒ 5×2×3 = **30 run**。

### L32.4 停机规则 8 条
① 主断言成立 ⇒ 停扫进定稿｜② `slow_inc`↑ 使 R **反升** ⇒ 立即停｜③ 连续 2 档无改善 ⇒ 停｜④ `slow_trace` 非活/读回不一致 ⇒ 该档 `invalid`，**≥2 档则停并查引擎**｜⑤ 对数断言违反或 `hard_failures` ⇒ 停｜⑥ 触包络 ⇒ 停｜⑦ `bias≤0.50` 与 `inc>180` 不进｜⑧ S1–S3 全败且非反向 ⇒ 启动 **C2 作为 L2b**。

### L32.5 预算（chunk 实测单价：bias=0.70 干跑 **0.7964**，上界 0.9720）
扫描 30 run **1.394**（上界 1.701）＋定稿确认 45 run **4.745**（5.791）＋新工况孪生臂 4 run **0.490** ⇒ **合计 6.629 GPU-h（上界 7.982）⇒ 在 ≤10 包络内**（余量 2.0–3.4）；**墙钟 ≈7.6 h**，**必须托管**。
**降档**：④ **推荐** 定稿确认只跑 3 剂量 ⇒ 总 **≈4.73**（(d) 网格不全，因 (d) 为 sanity 可接受，须登记）。

### L32.6 最小阳性对照（R84 第 37 条）—— **✅ 已完成，CPU，不占 MPS**
- **设计**：N=**1000**、T=**1 s** + settle 500 ms、`device=cpu`、`OMP_NUM_THREADS=1`、bias=0.70、σ=0.30、dose=1（有活动才有历史）；三臂 = 阳性(inc=180,τ=3000) / 默认(inc=6,τ=1000) / **反例(slow off)**；断言 = `slow_trace.n>0 ∧ alive ∧ mean>0 ∧ std>1e-6`。
- **实测**：阳性 `mean=22.267, std=6.347, alive=True`；默认 `mean=0.5686, alive=True`；**反例 `slow_trace=None` ⇒ 断言失败** ⇒ **非零退出语义成立**（`verdict=positive_control_pass`）。
- **参数是活的**：`slow_inc` 6→180 ⇒ `slow_trace.mean` **×39.2**（M10 D1 家族的正面对照）。
- **机制方向**：slow off spk=**598** → on(inc=180) spk=**101**（**−83%**）⇒ **负反馈方向被实测确认**（「不点燃自持」论证的直接支持）。
- **口径教训（新）**：**事件表必须覆盖 settle+T**（绝对步号语义）——首跑只覆盖 T ⇒ `IndexError: index 20040 out of bounds for size 20001` ⇒ 已修正并登记。
- 产物：`data/m11_p5_l2_selftest_slowvar.json`。**不声称**：该对照只证明接线生效与方向，**不**证明 8192 档上能把 (a) 抬进带。

### L32.7 🔴 B2 硬前置（两项）
1. **必须**：把 `slow_trace` 统计**写进臂 JSON**（当前只在引擎 `res` 里）⇒ 否则正式档无法做机制读回断言（M10 D1 家族的防线）。
2. 建议：CLI 暴露 `--tau-slow-ms/--slow-inc`（计划模板键组 **L436** 已含二者，`arm_spec_from_cli`（**L226–237**）与 CLI 未透传；**臂 spec JSON 路径已可用**）；C2 路线另需在 `build_engine`（**L113–128**）内调用 `configure_stp` 并**显式禁 facilitate**。
