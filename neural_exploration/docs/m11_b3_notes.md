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

- 本节点**只新建**：`data/m11_p5_targets.{csv,json}`、`data/m11_p5_smoke_*`（协议/夹具/输出，14 件）、`data/m11_b3_managed_jobs.json`、`tools/validate_p11_resting.py`、`docs/m11_b3_{g1_design,notes}.md`。
- **未改**：`src/mouse_cortex_*`（B2）、`data/m11_data_*` / `tools/fetch_m11_*` / `tools/build_m11_*`（B1）、`docs/m11_acceptance_master.md` / `docs/m11_data_gate_report.md` / `docs/m11_env_notes.md` / `docs/m11_list_review_master.md`（主agent）、以及全部 M0–M10 冻结件（只读消费：`m10_p6_poptraces/*.npy`、`m10_p5_gate1_ci.json`、`m10_calibration_targets.csv`、`src/larval_zf_circuit.py`）。
- ⚠️ `git status` 另有 **9 个非本节点的改动**（`src/mouse_cortex_engine.py`、`m11_engine_alignment.*`、`scan_m11_engine.py`、`normalize_m10_p6_table.py`、`m10_p6_judgement_table.csv`、`m10_stage3_report.md`、`demo_m11_microns_decode.py` 等）⇒ **B2/主agent 正在并发编辑**，与本工作包无关，勿归因于本节点。

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
