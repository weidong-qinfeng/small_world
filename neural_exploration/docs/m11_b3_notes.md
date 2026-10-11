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
