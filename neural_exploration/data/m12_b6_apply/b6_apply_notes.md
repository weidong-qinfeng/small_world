# B-6 登记执行说明（node B 交付物 2）—— M12 `data/m12_targets.csv` 的 4 处原位填入 + 1 处 append

> **性质**：本文件是**执行留痕与逐条对照**（不是裁决件、不是数据件）。**本节点未修改任何已存在的件**；
> 全部产物位于新目录 `neural_exploration/data/m12_b6_apply/`，`data/m12_targets.csv` 的 sha256 全程未变（见 §5）。

## 0. 授权范围（严格照 R188 ⑦，未扩大）

| 类别 | 项 | 本节点做了什么 |
|---|---|---|
| 无条件可登记 | `protocol_settle_ms = 500`、`protocol_drive_rate_hz = 8.0` | 原位填入 + `final=1`（第 1、4 项） |
| 附口径声明义务 | `protocol_bin_wide_ms = 100`、`protocol_smooth_window_frames = 5` | 原位填入 + `final=1`，note 内当场写口径并经第八十四条检查（第 2、3 项） |
| 仅可提案 | `B6-P3` 的 `append`（新 key `gate_p11_anchor_n_min`，`unavailable`/`final=0`） | 追加 1 行（第 5 项） |
| **暂不授权（本节点一律未动）** | 需裁决的 3 项（`protocol_T_main_s`／`protocol_drive_dose_grid`／`gate_points_per_region_ladder`）与欠证据的 10 项 | **value/final/note 一律未改**（候选表与现表在这 13 行上逐字节相同，见 §5） |

## 1. 候选表身份与闸门实测（Pass 标准）

| 项 | 值 |
|---|---|
| 候选表 | `/Users/weidong/ai/small_world/neural_exploration/data/m12_b6_apply/m12_targets.candidate.csv` |
| 现表（未改，sha 仍为登记值） | `/Users/weidong/ai/small_world/neural_exploration/data/m12_targets.csv` |
| 候选表 sha256 | `043882ce44bb06f8911376a42f08d7a571bfe60ab32c4305bc0c0d4b2a364b34` |
| 现表 sha256 | `182847d8c2bbf661e5a43e19e9935c9a17154d9a7c66b15047e82984b86ccd8b` |
| 字节数 | 现表 80871 → 候选表 101971 |
| 记录数/`final` | 现表 `records=124 final=1:98 final=0:26` → 候选表 `records=125 final=1:102 final=0:23` |

```bash
$ cd /Users/weidong/ai/small_world && PYTHONHASHSEED=0 MPLBACKEND=Agg .venv-m11/bin/python \
    neural_exploration/tools/check_m12_targets_gates.py \
    --table neural_exploration/data/m12_b6_apply/m12_targets.candidate.csv
=== G0 结构闸门 === PASS
=== G1 附录缺失检查 === PASS
=== G2 四元组完整性 === PASS
=== G3 final=0 计数 === INFO
=== G4 清单列覆盖（清单 §5.1 L887） === INFO
[TABLE] /Users/weidong/ai/small_world/neural_exploration/data/m12_b6_apply/m12_targets.candidate.csv  records=125  final=1:102  final=0:23
[RESULT] ALL GATES PASS
$ echo $?
0
```

> 注：候选表的 `final=0` 为 **23**（不是 B-6 清单 §4.2「验证 3」的 25）—— 因为 R188 ⑦ 额外授权了 `protocol_bin_wide_ms` 与 `protocol_smooth_window_frames` 两处填入（B-6 的预跑只灌了 2 填 + 1 追加）。

## 2. 逐项说明（5 处改动）

### 2.1 `protocol_settle_ms`（现表 L11（占位行）→ 候选表 L11（同一物理行））

| 字段 | 现表 | 候选表 |
|---|---|---|
| `value` | `unavailable` | `500` |
| `value_kind` | `unavailable` | `point` |
| `evidence_level` | `unavailable` | `derived` |
| `final` | `0` | `1` |
| `unit` | `ms` | `ms`（**逐字不变**） |
| `provenance` / `wrong_usage_consequence` / `scope` | —— | **逐字节不变**（§5 的 diff 证明） |

**写入 `note` 的完整留痕字符串**（首句即 R188 ② 的强制格式；下块与候选表内字节逐字相同）：

```
【事后填入：R188（docs/m11_acceptance_master.md §3fz ①/②/⑦）；原值 value=unavailable、final=0；唯一未满足前提=「M12 清单 §2–§5 全文无 settle 数值（本行原 note 自述的唯一缺口）⇒ 该常数在 M12 口径下无可引出处、故本行原 final=0」；依据=该前提已被独立否证并被 R188 ⑦ 明示为「无条件可登记」项（该条同时给出机械证据：原位填 protocol_settle_ms 与 protocol_drive_rate_hz ⇒ ALL GATES PASS）：同口径值有正式注册行与实现默认值四处 —— [other_node] data/m11_p5_targets.csv L15（protocol_settle_ms=500、value_kind=point、unit=ms、derived、final=1；provenance 逐字指向 M11 清单 §4.5 协议行 + data/m10_calibration_targets.csv::settle_window_ms）+ data/m10_calibration_targets.csv L12（settle_window_ms=500、口径注逐字「t=0 初始化瞬态剔除窗；判据只看 settle 之后」）+ [self] tools/make_m11_p5_plan_l3.py L106（SETTLE_MS = 500.0）、tools/make_m11_p5_plan_l0.py L67、tools/make_m11_p5_plan_g1.py L42（三处同值）、tools/run_m11_p5_arms.py L1455（--settle-ms 默认 500.0）】【口径声明 + 第八十四条口径桥检查】本行的量 = settle 时长；口径域 = **脉冲（spike）引擎**、单位 ms、作用量 = 「t=0 初始化瞬态剔除窗」，判据只统计 settle 之后的窗。被判定的量 = G0-M/P3（对齐门）与 G1-M/P5a（区域率/静默比）—— 它们统计的是同一脉冲引擎在同一初始化瞬态下的发放 ⇒ 与 M10/M11 的该常数**同口径**（同为脉冲域、同为 t=0 瞬态剔除、同为 ms 尺度）。跨里程碑声明沿 §0.8 EQ-1（口径一致才可比）：EQ-2 所禁的是把 M9/M10/M11 的**判据带**（绝对速度/相关/dwell/τ）作 M12 的带；口径常数属「必须一致」类（R188 ⑦ 采此切分）。【非文献值】出处链全部是内部教训（M5 L37#2）与实现默认值、无 PMID/DOI ⇒ evidence_level=derived ⇒ 不得声称有生理文献依据。（第八十四条规定：**不同口径**才必须同时登记口径桥或判 unavailable；本项判定为同口径 ⇒ 不触发建桥分支。另按 R188 ②，留痕载体 = 本表的既有自由文本列 note、不新造 status_history 列。）【以下为本行 final=0 期间的原 note 原文（逐字保留供审计）】口径要求存在（§5.3 模板强制填写）但**数值缺失** ⇒ R117 同型缺口（"填得出模板、给不出值"）。本行即 skeleton 的"口径常数不得缺席"机制要抓的对象之一【第八十条四项检查（缺一即不得填）】(a) 表内已有允许填入的规则：data/m12_targets.csv L52 的 rule_final1_frozen_after_run 明文只禁 **final=1 的行在跑臂后**就地修改，本行原为 value=unavailable/final=0 的占位行 ⇒ 没有「已登记的值」被改动（该机制另有 M11 第四十四条 docs/m11_acceptance_master.md L2449 与 R169 先例 data/m11_p5_targets.csv L115/L116）；(b) 原 final=0 的**唯一**理由=上列「唯一未满足前提」这一条可被独立否证的前提、且该前提已按上列「依据」被否证（本 note 未写第二条前提）；(c) 填入值是**数据/实现属性的值**：值取自 M10/M11 的注册行与同族脉冲引擎的实现默认值，**不是本轮的产出、也不是为命中任何判据而调** —— M12 至今未跑任何仿真臂 ⇒ 「看结果调带」在结构上不可能发生；(d) 执行时四项核对（本节点在候选表上实测；写入注册表时须由主 agent 重做一次）：前态断言 value=unavailable 且 final=0 ✓；记录数 124→125（4 处原位填入不改计数、1 处 append 加 1）；key 差集 set(候选)−set(原)={gate_p11_anchor_n_min} 且反向差集=∅；原表 sha256=182847d8c2bbf661e5a43e19e9935c9a17154d9a7c66b15047e82984b86ccd8b、候选表 sha256 见 b6_apply_notes.md。
```

**第八十条四项检查（逐条对照）**

- **(a) 表内已有允许填入的规则**：**有既有规则**：`data/m12_targets.csv` L52 的 `rule_final1_frozen_after_run` 明文只管「**`final=1` 的行**在对应协议跑臂后不得就地修改」⇒ 本行原为 `value=unavailable`/`final=0` 的占位行、**没有任何已登记的值**可被改动（R188 ② 的保护对象不存在论证）；另有明文条文 **M11 第四十四条**（`docs/m11_acceptance_master.md` L2449：「占位行可填入的前提是该行声明的全部验收条件已满足」）与 **R169 先例**（`data/m11_p5_targets.csv` L115/L116 的钙域静默带原位填入）。
- **(b) 原 `final=0` 的唯一理由＝一条可被独立否证的前提、且已被否证**：**原 `final=0` 的唯一理由**（本行原 note 自述，恰一条）＝「M12 清单 §2–§5 全文无 settle 数值」（清单 L910 只要求写明 settle 时长而不给值）⇒ 该理由**已被独立否证**：同口径值有**正式注册行** `data/m11_p5_targets.csv` **L15**（`protocol_settle_ms=500`、`point`、`ms`、`derived`、`final=1`、provenance 指向 M11 清单 §4.5 协议行 + `data/m10_calibration_targets.csv::settle_window_ms`）＋ 溯源行 `data/m10_calibration_targets.csv` **L12**（`settle_window_ms=500`，口径注「t=0 初始化瞬态剔除窗；判据只看 settle 之后」）＋ 四处实现默认值（`tools/make_m11_p5_plan_l3.py` L106、`make_m11_p5_plan_l0.py` L67、`make_m11_p5_plan_g1.py` L42、`run_m11_p5_arms.py` L1455）⇒ 缺口是「M12 未登记」而**不是**「值不存在」。**未写第二条前提**（B-6 §3.3 纪律①）。
- **(c) 填入值是数据的属性、与本次运行的产出无关**：**填入值是数据/实现属性的值、与本次运行产出无关**：500 ms 是 M10 §4.2 预注册（M5 L37#2 教训）的**口径常数**（t=0 初始化瞬态剔除窗），其取值不依赖任何 M12 观测；**且 M12 至今未跑任何仿真臂**（`data/` 下 M12 产物全为数据门/掩膜件，见 `b6_deferred_evidence.md` §1.4）⇒ 「为命中判据而调」在结构上不可能发生（这一条正是第八十条判据句「缺少 (c) 时它就是调带」的排除形态）。
- **(d) 执行时的四项机械核对**：**四项核对**（在候选表上实测）：① 前态断言 —— 4 个目标行填入前 `value=unavailable` 且 `final=0`（脚本内 `assert`，不满足即不产出候选表）；② 记录数 —— 124 → **125**（4 处原位填入**不改计数**、1 处 append 加 1）；③ key 差集 —— `set(候选) − set(原) = {gate_p11_anchor_n_min}`、反向差集 = ∅；④ 哈希 —— 原表 `182847d8c2bbf661e5a43e19e9935c9a17154d9a7c66b15047e82984b86ccd8b`、候选表 `043882ce44bb06f8911376a42f08d7a571bfe60ab32c4305bc0c0d4b2a364b34`（**写入注册表后须由主 agent 重算并登记新 sha**）。

**第八十四条口径桥检查（逐条对照）**

- **口径（本值所属）**：脉冲（spike）引擎的「settle 时长」，单位 ms，定义 = t=0 初始化瞬态剔除窗。<br>**被判定的量**：G0-M/P3（对齐门，逐神经元 spike 时间对齐/率分布）与 G1-M/P5a（区域率/静默比）——它们统计的是**同一脉冲引擎在同一初始化瞬态之后**的发放。<br>**判定**：同口径（同为脉冲域、同为 t=0 瞬态剔除、同为 ms 尺度）⇒ **不需要建口径桥**（第八十四条的「不同口径才需建桥 / 判 unavailable」分支不触发）。<br>**跨里程碑**：沿 §0.8 EQ-1（口径一致才可比）沿用 500 ms；EQ-2 禁的是把 M9/M10/M11 的**判据带**（绝对速度/相关/dwell/τ）作 M12 的带，**口径常数属「必须一致」类**（R188 ⑦ 采此切分）。<br>**证据等级**：出处链全为内部教训与实现默认值、无 PMID/DOI ⇒ `evidence_level=derived`，**不得**声称有生理文献依据。

### 2.2 `protocol_bin_wide_ms`（现表 L12（占位行）→ 候选表 L12）

| 字段 | 现表 | 候选表 |
|---|---|---|
| `value` | `unavailable` | `100` |
| `value_kind` | `unavailable` | `point` |
| `evidence_level` | `unavailable` | `derived` |
| `final` | `0` | `1` |
| `unit` | `ms` | `ms`（**逐字不变**） |
| `provenance` / `wrong_usage_consequence` / `scope` | —— | **逐字节不变**（§5 的 diff 证明） |

**写入 `note` 的完整留痕字符串**（首句即 R188 ② 的强制格式；下块与候选表内字节逐字相同）：

```
【事后填入：R188（docs/m11_acceptance_master.md §3fz ①/②/⑦）；原值 value=unavailable、final=0；唯一未满足前提=「M12 清单 §2–§5 全文无 bin 宽数值（本行原 note 自述的唯一缺口）⇒ 该常数在 M12 口径下无可引出处、故本行原 final=0」；依据=该前提已被独立否证并被 R188 ⑦ 列为「附口径声明义务」的可登记项（本 note 即该义务的当场履行）：同口径值 100 ms 在同族脉冲引擎的率估计器口径上有五类落点 —— [self] tools/make_m11_p5_plan_l0.py L173 与 L265、tools/make_m11_p5_plan_g1.py L120、tools/make_m11_p5_plan_l3.py L728（协议 dict 内 bin_ms=100.0 与 smooth_frames=5 与 estimator=per_neuron 三者并列）+ tools/run_m11_p5_arms.py L1122（bin_ms 取 plan 值、缺省 100.0）与 L1427 + tools/validate_p11_resting.py L66（口径登记行 bin_ms: 100.0）与 L524（proto 缺省回退到 bin_ms_binwise）+ data/m11_p5_targets.csv L26（bin_ms_binwise=100、point、ms、derived、final=1；provenance 指 neural_exploration/src/larval_zf_circuit.py::bout_stats(bin_ms=100.0)）+ data/m11_p5_plan_g1.json L9（bin_ms: 100.0）。】【口径声明 + 第八十四条口径桥检查】bin 宽的定义 = **区域率的时间分箱宽度**（把脉冲计数除以多长的窗）；单位 ms；口径域 = **脉冲（spike）**；作用量 = 决定区域率序列的时间分辨率与率估计的方差。被判定的量 = F/P10a（区域率分布）与 F/P10c（任务依赖动力学）—— 本行原 note 自述它们「对 bin 宽直接敏感」⇒ 本常数正是它们的口径输入 ⇒ 同口径。★ 口径关系声明（B-6 清单 §3.2 #3 要求当场写明）：M11 的同一常数用在 estimator=per_neuron（逐神经元中位率）上、M12 用在**区域级**读出上 ⇒ 两者**不是同一个统计量的聚合层次**（M11 逐神经元、M12 逐区域）。但**bin 宽本身不是聚合层次**：它只定义「脉冲计数除以多长的窗」这一时间分箱、与聚合层次（per_neuron / per_region）正交 ⇒ 就「bin 宽」这一个量而言 M12 与 M11 **同口径**（同为脉冲域计数窗）⇒ 无需口径桥。需口径桥的是**率分布的聚合层次**那一维、而它已由本表 rule_a_per_impl_not_pooled（data/m12_targets.csv L47）与本行 scope 的 quad 段（region|all_arms|region_rate_distribution|bin_dependent）显式承载 ⇒ 不属本行。【非文献值】无 PMID/DOI ⇒ evidence_level=derived。（第八十四条规定：**不同口径**才必须同时登记口径桥或判 unavailable；本项判定为同口径 ⇒ 不触发建桥分支。另按 R188 ②，留痕载体 = 本表的既有自由文本列 note、不新造 status_history 列。）【以下为本行 final=0 期间的原 note 原文（逐字保留供审计）】口径要求存在但数值缺失。★ 注意与 M11 的差别：M11 的 (a) 是逐神经元中位率（对 bin 宽不敏感）而 M12 的 P10a/P10c 是区域率分布与任务依赖动力学（**对 bin 宽直接敏感**）⇒ 该缺口的代价比 M11 更大【第八十条四项检查（缺一即不得填）】(a) 表内已有允许填入的规则：data/m12_targets.csv L52 的 rule_final1_frozen_after_run 明文只禁 **final=1 的行在跑臂后**就地修改，本行原为 value=unavailable/final=0 的占位行 ⇒ 没有「已登记的值」被改动（该机制另有 M11 第四十四条 docs/m11_acceptance_master.md L2449 与 R169 先例 data/m11_p5_targets.csv L115/L116）；(b) 原 final=0 的**唯一**理由=上列「唯一未满足前提」这一条可被独立否证的前提、且该前提已按上列「依据」被否证（本 note 未写第二条前提）；(c) 填入值是**数据/实现属性的值**：值取自 M10/M11 的注册行与同族脉冲引擎的实现默认值，**不是本轮的产出、也不是为命中任何判据而调** —— M12 至今未跑任何仿真臂 ⇒ 「看结果调带」在结构上不可能发生；(d) 执行时四项核对（本节点在候选表上实测；写入注册表时须由主 agent 重做一次）：前态断言 value=unavailable 且 final=0 ✓；记录数 124→125（4 处原位填入不改计数、1 处 append 加 1）；key 差集 set(候选)−set(原)={gate_p11_anchor_n_min} 且反向差集=∅；原表 sha256=182847d8c2bbf661e5a43e19e9935c9a17154d9a7c66b15047e82984b86ccd8b、候选表 sha256 见 b6_apply_notes.md。
```

**第八十条四项检查（逐条对照）**

- **(a) 表内已有允许填入的规则**：**有既有规则**：同 `rule_final1_frozen_after_run`（`data/m12_targets.csv` L52，只管 `final=1` 行跑臂后不得就地改）＋ M11 第四十四条（`docs/m11_acceptance_master.md` L2449）＋ R169 先例；本行原为占位行，无已登记值被改动。
- **(b) 原 `final=0` 的唯一理由＝一条可被独立否证的前提、且已被否证**：**原 `final=0` 的唯一理由**（本行原 note 自述，恰一条）＝「M12 清单 §2–§5 全文无 bin 宽数值」（清单 L908 只要求写明 bin 宽而无数值）⇒ **已被独立否证**：同族脉冲引擎的率估计器口径里 `bin_ms=100.0` 有五类落点 —— `tools/make_m11_p5_plan_l0.py` L173/L265、`make_m11_p5_plan_g1.py` L120、`make_m11_p5_plan_l3.py` L728（协议 dict 内 `bin_ms=100.0` 与 `smooth_frames=5` 与 `estimator=per_neuron` 三者并列）＋ `tools/run_m11_p5_arms.py` L1122（缺省 100.0）与 L1427 ＋ `tools/validate_p11_resting.py` L66、L524 ＋ `data/m11_p5_targets.csv` **L26**（`bin_ms_binwise=100`、`final=1`）＋ `data/m11_p5_plan_g1.json` L9。**未写第二条前提**。
- **(c) 填入值是数据的属性、与本次运行的产出无关**：**填入值是数据/实现属性的值**：100 ms 是「群体/区域率序列的分箱宽度」这一实现口径的既存默认值（先于 M12 存在、由 M10/M11 的估计器管线固定），与本轮任何 M12 观测无关；M12 未跑任何仿真 ⇒ 非调带。
- **(d) 执行时的四项机械核对**：**四项核对**：① 前态断言 ✓（`value=unavailable`、`final=0`）；② 记录数 124 → 125；③ key 差集 = `{gate_p11_anchor_n_min}`、反向差集 = ∅；④ 原表/候选表 sha256 见 `protocol_settle_ms` 条（同一份候选表，sha 相同）。

**第八十四条口径桥检查（逐条对照）**

- **口径（本值所属）**：脉冲（spike）域**区域率的时间分箱宽度**（把脉冲计数除以多长的窗），单位 ms。<br>**被判定的量**：F/P10a（区域率分布）与 F/P10c（任务依赖动力学）——本行原 note 自述「对 bin 宽直接敏感」⇒ 本常数是它们的口径输入。<br>**★ 附口径声明义务（R188 ⑦）的履行**：M11 的同一常数用在 `estimator=per_neuron`（逐神经元中位率）上、M12 用在**区域级**读出上，两者**不是同一个统计量的聚合层次**；但 **bin 宽本身不是聚合层次**（它只定义时间分箱，与 per_neuron / per_region 正交）⇒ 就「bin 宽」这一个量而言**同口径**（同为脉冲域计数窗）⇒ 无需建桥。<br>需建桥的是「率分布的**聚合层次**」那一维 ⇒ 它已由本表 `rule_a_per_impl_not_pooled`（L47）与本行 `scope` 的 `quad=region|all_arms|region_rate_distribution|bin_dependent` 显式承载，**不得**记在本常数名下。<br>**证据等级**：无 PMID/DOI ⇒ `derived`。

### 2.3 `protocol_smooth_window_frames`（现表 L13（占位行）→ 候选表 L13）

| 字段 | 现表 | 候选表 |
|---|---|---|
| `value` | `unavailable` | `5` |
| `value_kind` | `unavailable` | `point` |
| `evidence_level` | `unavailable` | `derived` |
| `final` | `0` | `1` |
| `unit` | `frames` | `frames`（**逐字不变**） |
| `provenance` / `wrong_usage_consequence` / `scope` | —— | **逐字节不变**（§5 的 diff 证明） |

**写入 `note` 的完整留痕字符串**（首句即 R188 ② 的强制格式；下块与候选表内字节逐字相同）：

```
【事后填入：R188（docs/m11_acceptance_master.md §3fz ①/②/⑦）；原值 value=unavailable、final=0；唯一未满足前提=「M12 清单 §2–§5 全文无平滑窗数值（本行原 note 自述的唯一缺口）⇒ 该常数在 M12 口径下无可引出处、故本行原 final=0」；依据=该前提已被独立否证并被 R188 ⑦ 列为「附口径声明义务」的可登记项：**同口径**（脉冲域 bin 序列）的平滑窗实现默认值 5 帧在五类落点存在 —— [self] tools/make_m11_p5_plan_l0.py L173 与 L265、tools/make_m11_p5_plan_g1.py L120、tools/make_m11_p5_plan_l3.py L728（与 bin_ms=100.0 同一协议 dict 并列）+ tools/run_m11_p5_arms.py L1123（plan 缺省 smooth_frames=5）与 L1427 + tools/validate_p11_resting.py L66（口径登记行 smooth_frames: 5）与 L525（proto 缺省回退到 smooth_frames_state）+ data/m11_p5_plan_g1.json L10（smooth_frames: 5）。】【口径声明 + 第八十四条口径桥检查（R188 ⑦ 强制项）】单位 = **帧**；**本行的「帧」特指「活动 bin」而不是钙成像帧**：作用量 = 对由 protocol_bin_wide_ms（100 ms）分箱后的区域率序列做 5 帧滑动平均 ⇒ 等效平滑窗 = 5 × 100 ms = 500 ms。口径域 = 脉冲（spike）。被判定的量 = F/P10a 的区域率分布与 F/P10c 的任务依赖动力学，且按本行 scope 的 quad 段（region|all_arms|smoothed_rate|window_undefined）其主口径量是 smoothed_rate ⇒ 平滑窗是该量的**口径输入** ⇒ 同口径。★ 本声明的作用正是把「帧长」钉在 bin 宽上：单位名 frames 在两条口径链上帧长不同（见下）⇒ 不声明就会重演「同一单位名、不同量」的合并错误。【同量判断（R188 ⑤/⑦ 强制写入）：M12 的 protocol_smooth_window_frames 与 M11 的 smooth_frames_state 是否同量 —— 判断 = 就 key 名成立、就「量」不成立 ⇒ 两者不是同一个量】依据（两侧均实读给行号）：① M11 侧：data/m11_p5_targets.csv L27（smooth_frames_state=5、point、frames、derived、final=1）的 provenance 逐字为「docs/m11_acceptance_master.md R6/R7（0.914 s 箱 + 5 帧平滑 + z 分数…）」⇒ 它的「帧」是**钙成像帧**（帧长 0.914 s ⇒ 5 帧 ≈ 4.57 s、见 docs/生物仿真M11实施清单.md L584「B4 冻结管线：0.914 s 箱 → 5 帧平滑 → z → AR(1)」）；其 note 逐字「状态/慢变量分析口径；本门（P5）不判慢状态、仅登记以保证与 S 门口径一致」⇒ 它承载的是**慢状态/钙域分析**、明文不用于 P5 的率判定；② M12 侧：本行承载的是**区域率估计器**（脉冲域 bin 序列）⇒ 5 帧 = 5 × 100 ms = 500 ms。两者若都被当「5 帧」使用 ⇒ 作用量相差 4.57 s / 0.5 s ≈ 9.1× ⇒ **不是同一个量**（「就量不成立」的量化形态）。③ 「同量」方向也必须如实登记：M11 **另有一处同口径的值** —— M11 的 P5 协议里 smooth_frames=5 与 bin_ms=100.0 并列（tools/make_m11_p5_plan_l0.py L173、tools/make_m11_p5_plan_g1.py L120、tools/make_m11_p5_plan_l3.py L728、tools/run_m11_p5_arms.py L1427）⇒ M12 的该项与**这一处**同口径（脉冲域 bin 序列）。⇒ 本行取 5 的依据是**同口径先例**、不是把钙域的帧数（smooth_frames_state）移植过来；两者数值相同（均为 5）但量不同 ⇒ 数值巧合。④ 与本行原 note 的关系：原 note 称「（M11 表亦无该行 ⇒ 两表同缺 ⇒ 这是跨里程碑的缺口而非 M12 引入）」——**就 key 名成立、就量不成立**（M11 表确实无同名 key、但有同义量行 smooth_frames_state=5 位于 L27）；该处文档缺陷的更正建议见 data/m12_b6_apply/b6_doc_defects.md（依 R188 ⑤：登记、不就地改）。【已知口径风险（B-6 清单 §3.2 #4 要求登记）】平滑窗本身已被登记为口径风险：docs/m10_acceptance_master.md L243 明文慢时标「不能宣称独立于 5 帧平滑」⇒ M12 报告若引用 τ 类慢时标量必须同时报平滑窗。【非文献值】无 PMID/DOI ⇒ evidence_level=derived。（第八十四条规定：**不同口径**才必须同时登记口径桥或判 unavailable；本项判定为同口径 ⇒ 不触发建桥分支。另按 R188 ②，留痕载体 = 本表的既有自由文本列 note、不新造 status_history 列。）【以下为本行 final=0 期间的原 note 原文（逐字保留供审计）】口径要求存在但数值缺失（M11 表亦无该行 ⇒ 两表同缺 ⇒ 这是跨里程碑的缺口而非 M12 引入）【第八十条四项检查（缺一即不得填）】(a) 表内已有允许填入的规则：data/m12_targets.csv L52 的 rule_final1_frozen_after_run 明文只禁 **final=1 的行在跑臂后**就地修改，本行原为 value=unavailable/final=0 的占位行 ⇒ 没有「已登记的值」被改动（该机制另有 M11 第四十四条 docs/m11_acceptance_master.md L2449 与 R169 先例 data/m11_p5_targets.csv L115/L116）；(b) 原 final=0 的**唯一**理由=上列「唯一未满足前提」这一条可被独立否证的前提、且该前提已按上列「依据」被否证（本 note 未写第二条前提）；(c) 填入值是**数据/实现属性的值**：值取自 M10/M11 的注册行与同族脉冲引擎的实现默认值，**不是本轮的产出、也不是为命中任何判据而调** —— M12 至今未跑任何仿真臂 ⇒ 「看结果调带」在结构上不可能发生；(d) 执行时四项核对（本节点在候选表上实测；写入注册表时须由主 agent 重做一次）：前态断言 value=unavailable 且 final=0 ✓；记录数 124→125（4 处原位填入不改计数、1 处 append 加 1）；key 差集 set(候选)−set(原)={gate_p11_anchor_n_min} 且反向差集=∅；原表 sha256=182847d8c2bbf661e5a43e19e9935c9a17154d9a7c66b15047e82984b86ccd8b、候选表 sha256 见 b6_apply_notes.md。
```

**第八十条四项检查（逐条对照）**

- **(a) 表内已有允许填入的规则**：**有既有规则**：同 `rule_final1_frozen_after_run`（L52）＋ M11 第四十四条（L2449）＋ R169 先例；本行原为占位行，无已登记值被改动。
- **(b) 原 `final=0` 的唯一理由＝一条可被独立否证的前提、且已被否证**：**原 `final=0` 的唯一理由**（原 note 自述，恰一条）＝「M12 清单 §2–§5 全文无平滑窗数值」⇒ **已被独立否证**：**同口径**（脉冲域 bin 序列）的平滑窗实现默认值 5 帧有四处落点 —— `tools/make_m11_p5_plan_l0.py` L173/L265、`make_m11_p5_plan_g1.py` L120、`make_m11_p5_plan_l3.py` L728（与 `bin_ms=100.0` 同一 dict 并列）＋ `tools/run_m11_p5_arms.py` L1123（plan 缺省 5）与 L1427 ＋ `tools/validate_p11_resting.py` L66、L525 ＋ `data/m11_p5_plan_g1.json` L10。**未写第二条前提**。
- **(c) 填入值是数据的属性、与本次运行的产出无关**：**填入值是数据/实现属性的值**：5 帧是 M10/M11 冻结成像与 P5 估计器管线的既存默认平滑窗长度（`docs/生物仿真M11实施清单.md` L584 的「0.914 s 箱 → 5 帧平滑 → z → AR(1)」管线），与本轮 M12 观测无关；M12 未跑任何仿真 ⇒ 非调带。
- **(d) 执行时的四项机械核对**：**四项核对**：① 前态断言 ✓；② 记录数 124 → 125；③ key 差集 = `{gate_p11_anchor_n_min}`、反向差集 = ∅；④ sha 见上。

**第八十四条口径桥检查（逐条对照）**

- **口径（本值所属）**：单位 = 帧，且本行的「帧」**特指活动 bin**（**不是钙成像帧**）：作用量 = 对由 `protocol_bin_wide_ms`（100 ms）分箱后的区域率序列做 5 帧滑动平均 ⇒ 等效平滑窗 = 5 × 100 ms = 500 ms；口径域 = 脉冲（spike）。<br>**被判定的量**：F/P10a 的区域率分布与 F/P10c 的任务依赖动力学；按本行 `scope` 的 `quad=region|all_arms|smoothed_rate|window_undefined`，其主口径量是 `smoothed_rate` ⇒ 平滑窗是该量的口径输入 ⇒ 同口径 ⇒ 无需建桥。<br>**★ 同量判断（R188 ⑤/⑦ 强制写入，也是本行 note 的一部分）**：M12 的 `protocol_smooth_window_frames` 与 M11 的 `smooth_frames_state` **就 key 名成立、就「量」不成立** ⇒ **不是同一个量**：① M11 侧 `data/m11_p5_targets.csv` **L27** 的 `smooth_frames_state=5`（`point`/`frames`/`derived`/`final=1`）provenance 逐字为「docs/m11_acceptance_master.md R6/R7（**0.914 s 箱 + 5 帧平滑 + z 分数**…）」⇒ 它的帧是**钙成像帧**（帧长 0.914 s ⇒ 5 帧 ≈ 4.57 s），note 逐字「状态/慢变量分析口径；本门（P5）不判慢状态、仅登记以保证与 S 门口径一致」⇒ 承载**慢状态/钙域分析**、明文不用于 P5 的率判定；② M12 侧本行承载**区域率估计器**（脉冲域 bin 序列）⇒ 5 帧 = 500 ms；两者同称「5 帧」时作用量相差 ≈ **9.1×**（4.57 s / 0.5 s）⇒ 不同量（数值相同属巧合）；③ **同量的否证方向也如实登记**：M11 **另有一处同口径的值** —— M11 的 P5 协议里 `smooth_frames=5` 与 `bin_ms=100.0` 并列（`tools/make_m11_p5_plan_l0.py` L173、`make_m11_p5_plan_g1.py` L120、`make_m11_p5_plan_l3.py` L728、`run_m11_p5_arms.py` L1427）⇒ 本行取 5 的依据是**同口径先例**，不是移植钙域帧数。<br>**已知口径风险（B-6 §3.2 #4 要求登记）**：`docs/m10_acceptance_master.md` L243 明文慢时标「不能宣称独立于 5 帧平滑」⇒ M12 报告若引用 τ 类慢时标量必须同时报平滑窗。<br>**证据等级**：无 PMID/DOI ⇒ `derived`。

### 2.4 `protocol_drive_rate_hz`（现表 L14（占位行）→ 候选表 L14）

| 字段 | 现表 | 候选表 |
|---|---|---|
| `value` | `unavailable` | `8.0` |
| `value_kind` | `unavailable` | `point` |
| `evidence_level` | `unavailable` | `derived` |
| `final` | `0` | `1` |
| `unit` | `Hz` | `Hz`（**逐字不变**） |
| `provenance` / `wrong_usage_consequence` / `scope` | —— | **逐字节不变**（§5 的 diff 证明） |

**写入 `note` 的完整留痕字符串**（首句即 R188 ② 的强制格式；下块与候选表内字节逐字相同）：

```
【事后填入：R188（docs/m11_acceptance_master.md §3fz ①/②/⑦）；原值 value=unavailable、final=0；唯一未满足前提=「M12 清单 §2–§5 全文无 M12 驱动率注册行（本行原 note 自述：本行是「缺口登记」而不是「值登记」）⇒ 该常数在 M12 口径下无可引出处、故本行原 final=0」；依据=该前提已被独立否证并被 R188 ⑦ 明示为「无条件可登记」项：同口径值有 M11 的**正式注册行** —— [other_node] data/m11_p5_targets.csv L119（protocol_drive_rate_hz=8.0、value_kind=point、unit=Hz、derived、final=1；note 自述「本行由 R115 §⑤ 补登 —— 此前 103 行中无任何行登记该常数」，口径逐字「口径=**逐神经元**外部泊松事件率（Hz per neuron）、用于 P5 基准臂（role=baseline、drive_mult=1.0）」）+ [self] 实现默认值四处：tools/make_m11_p5_plan_l3.py L108、tools/make_m11_p5_plan_l0.py L69、tools/make_m11_p5_plan_g1.py L41（三处 DRIVE_RATE_HZ = 8.0）、tools/run_m11_p5_arms.py L1458（--drive-rate-hz 默认 8.0）+ data/m11_p5_plan_g1.json L7（drive_rate_hz: 8.0）。】【口径声明 + 第八十四条口径桥检查】口径 = **逐神经元**外部泊松事件率（Hz per neuron）、且是 **dose=1 档**的基准量（剂量档本身由本表 protocol_drive_dose_grid 承载、不属本行）。被判定的量 = G1-M/P5b（驱动弹性/自持比）与 G1-M/P5a（率带）—— 它们统计的是脉冲域逐神经元/逐区域的发放，与「外部泊松驱动事件率」同处**脉冲域**（同一引擎、同一事件表口径）⇒ **同口径**。跨里程碑声明沿 §0.8 EQ-1（口径一致才可比）沿用 M11 的 8.0 Hz：EQ-2 所禁的是把 M9/M10/M11 的**判据带**（绝对速度/相关/dwell/τ）作 M12 的带、而口径常数属「必须一致」类（R188 ⑦ 采此切分、本行即该切分的落地）。【禁令（随行持久化）】🔴 不得把该值当「可调旋钮」去命中率带：提高背景驱动以「制造」静息态判据已由 docs/m11_b3_g1_design.md L146 明文禁止（该行逐字「若确需 λ 扫描、只能作为**剂量-响应诊断**…**不得**作为标定手段」）、同旨见 M11 R110 ④；此类命中不得记为工作点命中。【非文献值】无 PMID/DOI ⇒ evidence_level=derived（不得声称有生理文献依据）。（第八十四条规定：**不同口径**才必须同时登记口径桥或判 unavailable；本项判定为同口径 ⇒ 不触发建桥分支。另按 R188 ②，留痕载体 = 本表的既有自由文本列 note、不新造 status_history 列。）【以下为本行 final=0 期间的原 note 原文（逐字保留供审计）】★**这就是 R117 的复现**：M11 的 (a) 判定依赖驱动率常数而它在 103 行中从未登记；M12 的 P5b（驱动弹性）/P5a（率带）同样依赖它，而本骨架登记时**清单里查不到该常数** ⇒ 本行是"缺口登记"而不是"值登记"。补值路径：B3 在跑 P5b 前必须新增 final=1 的 protocol_drive_rate_hz 行（值 + 口径：逐神经元 还是 逐区域）若沿用 M11 的 8.0 ⇒ 必须另写一行说明"跨里程碑移植"的理由（§0.8 EQ-2 禁移植绝对数值）【第八十条四项检查（缺一即不得填）】(a) 表内已有允许填入的规则：data/m12_targets.csv L52 的 rule_final1_frozen_after_run 明文只禁 **final=1 的行在跑臂后**就地修改，本行原为 value=unavailable/final=0 的占位行 ⇒ 没有「已登记的值」被改动（该机制另有 M11 第四十四条 docs/m11_acceptance_master.md L2449 与 R169 先例 data/m11_p5_targets.csv L115/L116）；(b) 原 final=0 的**唯一**理由=上列「唯一未满足前提」这一条可被独立否证的前提、且该前提已按上列「依据」被否证（本 note 未写第二条前提）；(c) 填入值是**数据/实现属性的值**：值取自 M10/M11 的注册行与同族脉冲引擎的实现默认值，**不是本轮的产出、也不是为命中任何判据而调** —— M12 至今未跑任何仿真臂 ⇒ 「看结果调带」在结构上不可能发生；(d) 执行时四项核对（本节点在候选表上实测；写入注册表时须由主 agent 重做一次）：前态断言 value=unavailable 且 final=0 ✓；记录数 124→125（4 处原位填入不改计数、1 处 append 加 1）；key 差集 set(候选)−set(原)={gate_p11_anchor_n_min} 且反向差集=∅；原表 sha256=182847d8c2bbf661e5a43e19e9935c9a17154d9a7c66b15047e82984b86ccd8b、候选表 sha256 见 b6_apply_notes.md。
```

**第八十条四项检查（逐条对照）**

- **(a) 表内已有允许填入的规则**：**有既有规则**：同 `rule_final1_frozen_after_run`（L52）＋ M11 第四十四条（L2449）＋ R169 先例；本行原为占位行，无已登记值被改动。
- **(b) 原 `final=0` 的唯一理由＝一条可被独立否证的前提、且已被否证**：**原 `final=0` 的唯一理由**（原 note 自述，恰一条）＝「M12 清单 §2–§5 全文无 M12 驱动率注册行」（本行原 note 自述「本行是『缺口登记』而不是『值登记』」）⇒ **已被独立否证**：同口径值有 M11 的**正式注册行** `data/m11_p5_targets.csv` **L119**（`protocol_drive_rate_hz=8.0`、`point`、`Hz`、`derived`、`final=1`；note 自述「本行由 R115 §⑤ 补登 —— 此前 103 行中无任何行登记该常数」，口径逐字「**逐神经元**外部泊松事件率（Hz per neuron）、用于 P5 基准臂（role=baseline、drive_mult=1.0）」）＋ 四处实现默认值（`tools/make_m11_p5_plan_l3.py` L108、`make_m11_p5_plan_l0.py` L69、`make_m11_p5_plan_g1.py` L41 三处 `DRIVE_RATE_HZ = 8.0`、`run_m11_p5_arms.py` L1458）＋ `data/m11_p5_plan_g1.json` L7。**未写第二条前提**。
- **(c) 填入值是数据的属性、与本次运行的产出无关**：**填入值是数据/实现属性的值**：8.0 Hz 是 M11 已登记的**口径常数**（逐神经元外部泊松事件率、dose=1 档），先于 M12 存在且与 M12 的任何观测无关；M12 未跑任何仿真 ⇒ 非调带。**特别地**：本值**不是**为提高命中率而选的旋钮 —— R188 ⑦ 无条件授权、B-6 §4.1 B6-3 的伪证条件（写入非 8.0 的值即「调参迁就结论」）在本行不成立。
- **(d) 执行时的四项机械核对**：**四项核对**：① 前态断言 ✓；② 记录数 124 → 125；③ key 差集 = `{gate_p11_anchor_n_min}`、反向差集 = ∅；④ sha 见上。

**第八十四条口径桥检查（逐条对照）**

- **口径（本值所属）**：**逐神经元**外部泊松事件率（Hz per neuron），且是 **dose=1 档**的基准量（剂量档本身由本表 `protocol_drive_dose_grid` 承载，不属本行）。<br>**被判定的量**：G1-M/P5b（驱动弹性/自持比）与 G1-M/P5a（率带）——它们统计脉冲域逐神经元/逐区域的发放，与「外部泊松驱动事件率」同处**脉冲域**（同一引擎、同一事件表口径）⇒ **同口径** ⇒ 无需建桥。<br>**跨里程碑**：沿 §0.8 EQ-1 沿用 M11 的 8.0 Hz；EQ-2 所禁的是把 M9/M10/M11 的**判据带**作 M12 的带，口径常数属「必须一致」类（R188 ⑦）。<br>**随行禁令**：🔴 不得把该值当「可调旋钮」去命中率带 —— `docs/m11_b3_g1_design.md` **L146** 明文「若确需 λ 扫描、只能作为**剂量-响应诊断**…**不得**作为标定手段」（同旨 M11 R110 ④）⇒ 此类命中不得记为工作点命中。<br>**证据等级**：无 PMID/DOI ⇒ `derived`。

### 2.5 `gate_p11_anchor_n_min`（**新增 key**，append-only；候选表 L134）

B-6-P3 是 17 项里唯一「既无值也无行」的一项（骨架 §5.2 第 16 项）。R188 ⑦ 把它列为「**仅可提案**」类 ⇒ 本节点只**追加**一行、值照旧 `unavailable`/`final=0`。

| 字段 | 现表 | 候选表（新增行） |
|---|---|---|
| 记录 | **不存在**（原 124 key 中无此 key；`grep` 仅命中 P8 的 `ref_p8_fit_set_vs_n_min` 与 `gate_n_min_coef_power`/`gate_n_total_min_coef`/`gate_n_min_ci_coef` 四行） | `gate_p11_anchor_n_min`（候选表 L134） |
| `value` / `value_kind` / `unit` / `evidence_level` / `final` | —— | `unavailable` / `unavailable` / `个` / `unavailable` / `0` |
| `scope` | —— | `applies=F/P11;arm=lesion_arms|pharma_arms;quad=anchor|lesion_arms|hit_rate|per_anchor_fraction` |

**写入的完整行（`provenance` / `note` / `wrong_usage_consequence` 三列全文）**

`provenance`：

```
[checklist] docs/生物仿真M12实施清单.md L371（P11 行「或 ≥预注册锚级下限，如『有锚 ≥20/50』」）与 L528（「预注册锚 ≤50 篇」）+ [other_node] docs/m12_b1_gate_closure.md L218 逐字「F4 分母下限按锚数预注册（如 n_min=50 ⇒ 目标 ≥35 命中才可能 CI95 下界 ≥0.70）」——**该处是示例不是注册**+ [self] 本节点实测：原表 124 key 中无 P11 锚级 n_min 专属行（最接近的两行都不是它）
```

`note`：

```
【新增 key 登记（append-only）：R188 ⑦（docs/m11_acceptance_master.md §3fz）授权「仅可提案」的 B6-P3 以 append 方式新增本行、走表首 L1 的 append-only 规则；本 key 原不存在（原 124 key 中无 P11 锚级 n_min 专属行）⇒ 本行无「原值」可保留、故不适用 R188 ② 的【事后填入】留痕格式】【口径】F/P11（F4）锚级命中率的**分母下限**：分母 < 该值 ⇒ invalid（不是 pass、也不是 fail）。它是 docs/生物仿真M12实施清单.md §5.2 第 16 项（17 项里唯一「既无值也无行」的一项）的显式登记 ⇒ 登记后该项从「查无此常数」变为「已登记但值未定稿」。【本条 value=unavailable 的唯一前提】data/m12_targets.csv L26 的 target_p11_anchor_set 由 partial 定稿（＝ M12 §2.2 S8 药理/损伤锚逐条 provenance 化并给出 value_verified；当前 6 区域中 2 有 literature 锚、文献幅度值一律 value_verified=no）；n_min 是该锚集的直接函数 ⇒ 锚集定稿前不得填值（**恰好一条前提、无第二条**）。【为什么必须有这一行】不登记 ⇒ 「P11 分母下限」是一个没有 key 的量 ⇒ 既不能引用也不能机器核对（任何 key 式 token 都会被 G1 闸门按「附录缺失」拦下）；若直接把示例值 50 当注册值 ⇒ 用未注册常数作判据（骨架 §3.2 B4 明文：必须落 unavailable、不得用示例值凑）。【第八十四条口径桥检查】本行只登记「分母下限」这一个维度、不携带任何跨口径的带值 ⇒ 判定时分母与分子同属 P11 锚级命中率口径（本表 gate_f4_denominator_exclusion 已把 unavailable/injection_ineffective/no-experiment 排除出分母）⇒ 同口径、无需建桥。【第八十条说明】本行是**新增 key** 而非「事后填入占位行」⇒ 第八十条 (a)–(d) 的适用对象不成立（它管的是把已有 final=0 占位行改成值登记）；但仍须满足 append-only 的四项核对：记录数 124→125、本 key 恰出现 1 次、与上版差集={gate_p11_anchor_n_min}、前缀逐字节一致（见 data/m12_b6_apply/b6_apply_notes.md 的 diff 证明）。【禁令】不得把示例值 50 当注册值（docs/生物仿真M12实施清单.md L969 与 docs/m12_criteria_table_skeleton.md §3.2 B4 明文）。
```

`wrong_usage_consequence`：

```
不登记该行 ⇒ 「P11 分母下限」是一个没有 key 的量 ⇒ 既不能引用也不能机器核对；若直接把示例值 50 当注册值 ⇒ 用了未注册常数作判据（骨架 §3.2 B4 明文：必须落 unavailable、不得用示例值凑）；若把 unavailable/no-experiment 计入分母 ⇒ 该判定作废（docs/生物仿真M12实施清单.md §1.2 F4 明文：判 invalid、不是 fail）。
```

**第八十条是否适用**：**不适用** —— 第八十条管的是「**把已有 `final=0` 占位行改成值登记**」；本项是**新增 key**（append-only），不存在「原值」可留痕，故 note 内不套用 R188 ② 的 `【事后填入…】` 格式，改以 `【新增 key 登记…】` 起首并在 note 内写明**唯一前提**。

**仍然适用的四项核对（M11 第六十一条 / 本表 `rule_append_only_no_header`）**：① 记录数 124 → **125**；② 唯一性 —— `gate_p11_anchor_n_min` 恰出现 **1** 次；③ 与上版差集 —— `set(候选) − set(原) = {gate_p11_anchor_n_min}`，反向差集 = ∅；④ 前缀逐字节 —— 现表 L1–L133 与候选表 L1–L133 中**除 4 处填入行外逐字节相同**（§5 证明），且**未连表头一起追加**。

**唯一前提（恰一条，可独立否证）**：`data/m12_targets.csv` L26 的 `target_p11_anchor_set` 由 `partial` 定稿（＝ M12 §2.2 S8 药理/损伤锚逐条 provenance 化并给出 `value_verified`；当前 6 区域中 2 有 literature 锚、文献幅度值一律 `value_verified=no`）⇒ `n_min` 是该锚集的直接函数。

**为什么没有取示例值 50**：`docs/m12_b1_gate_closure.md` L218 的「如 `n_min=50`」与清单 L371 的「如『有锚 ≥20/50』」都是**示例**；清单 L969 与骨架 §3.2 B4 明文：用未注册常数作判据 ⇒ 必须落 `unavailable`、**不得用示例值凑**。

## 3. 与授权范围的一致性自检

| 检查 | 结果 |
|---|---|
| 是否只改了 R188 ⑦ 授权的 4 项 + 1 条 append？ | 是：候选表与现表的**全部差异**＝ L11/L12/L13/L14 四行 + 新增 L134（§5 给出机器证明） |
| 是否为「暂不授权」的 13 项选过值或填过行？ | **否** —— `protocol_T_main_s`(L10)、`protocol_drive_dose_grid`(L15)、`gate_points_per_region_ladder`(L102) 与其余 10 项在候选表中**逐字节等于现表** |
| 是否新造了 `status_history` 列？ | **否** —— 10 列 schema 未动（G0 的列名/列序检查 PASS；`--m11-table` 前 9 列逐列同名同义检查亦 PASS） |
| 是否改了表头/注释行/列序？ | **否** —— 现表 L1–L9（注释 + 表头）逐字节相同 |
| 是否直接改了 `data/m12_targets.csv`？ | **否** —— 全程只读该文件；sha256 未变（§5） |

## 4. 复核命令（可逐条重跑）

```bash
# 4.1 现表身份（应为登记值 182847d8…）
shasum -a 256 /Users/weidong/ai/small_world/neural_exploration/data/m12_targets.csv

# 4.2 候选表闸门（Pass 标准）
cd /Users/weidong/ai/small_world && PYTHONHASHSEED=0 MPLBACKEND=Agg .venv-m11/bin/python \
  neural_exploration/tools/check_m12_targets_gates.py \
  --table neural_exploration/data/m12_b6_apply/m12_targets.candidate.csv

# 4.3 差异清单（应只有 hunk 11,4 与 133,0）
diff -U0 neural_exploration/data/m12_targets.csv \
     neural_exploration/data/m12_b6_apply/m12_targets.candidate.csv | grep '^@@'

# 4.4 除 5 处外逐字节等同（见 §5 的脚本）
```

## 5. 「除 5 处外逐字节等同现表」的证明

命令与实测输出：

```text
$ cd /Users/weidong/ai/small_world/neural_exploration && python3 <按行号对齐比较>
原表 split len = 134 候选 split len = 135
前 134 行中内容不同的物理行号 = [11, 12, 13, 14, 134]
非差异行（原表）拼接 sha256 = 6f93d88a99ab4a12de032fe22dd7c71e1770ef01f96096ff1b1421c2070e215f
非差异行（候选）拼接 sha256 = 6f93d88a99ab4a12de032fe22dd7c71e1770ef01f96096ff1b1421c2070e215f
逐行相等的行数 = 129 | 全部逐字节相同 = True
原表 sha256 = 182847d8c2bbf661e5a43e19e9935c9a17154d9a7c66b15047e82984b86ccd8b
候选 sha256 = 043882ce44bb06f8911376a42f08d7a571bfe60ab32c4305bc0c0d4b2a364b34
原表 bytes = 80871 | 候选 bytes = 101971

$ diff -U0 m12_targets.csv m12_b6_apply/m12_targets.candidate.csv | grep '^@@'
@@ -11,4 +11,4 @@      ⇒ 4 处原位填入（L11–L14）
@@ -133,0 +134 @@     ⇒ 1 处 append（新行 L134）
```

⇒ **129 行逐字节相同、差异集合恰为 `{L11, L12, L13, L14, L134}`**（L134 为新增行；`diff` 未出现任何其他 hunk）。

## 6. 本节点**没有**做的事

1. **未运行任何仿真**、**未触碰 MPS/GPU**（本机另一作业独占 MPS）；
2. **未联网**；
3. **未修改任何已存在的文件** —— 尤其 `data/m12_targets.csv`、`docs/m11_acceptance_master.md`、`docs/m12_criteria_table_skeleton.md`、`docs/生物仿真M12实施清单.md`、`docs/m12_b6_unregistered_constants_checklist.md`、任何 `tools/*.py`、任何 `src/*.py`（本节点只**新建** `data/m12_b6_apply/` 下 4 件）；
4. **未 `git add` / 未 `git commit`**；
5. **未为「暂不授权」的 13 项选任何值或填任何行**、未扩大授权范围；
6. **未修改 `data/m12_targets.csv`**（sha256 全程 = 登记值）。

> 环境前缀：所有 python 调用均带 `PYTHONHASHSEED=0 MPLBACKEND=Agg`；解释器用 `/Users/weidong/ai/small_world/.venv-m11/bin/python`（仅标准库 + 无需 GPU）。

