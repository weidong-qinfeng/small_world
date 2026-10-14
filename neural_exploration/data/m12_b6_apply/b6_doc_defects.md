# B-6 文档缺陷更正建议（node B 交付物 4，依 R188 ⑤）

> **性质**：**只写建议 + 证据，不就地改任何已登记文件**。R188 ⑤ 逐字：「属**文档缺陷**，**登记、不就地改**；该『是否同量』的判断**并入 ⑥ 的登记义务**」、
> 「处置**不得**就地改已登记 sha 的骨架/清单 ⇒ **由 node B 在独立件里给出更正建议，我裁决后另立条目**」。
> 因此本文件**不含任何对既有文件的写入**；下列「建议更正文本」均为**草案字符串**，须由主 agent 裁决后执行。
>
> **方法**：每条断言附 `file:line` 或命令与输出。**本节点未修改任何已存在的文件**（全部产物在新目录 `data/m12_b6_apply/`）。

---

## 1. `protocol_T_dwell_ratio_min` **名存实亡**（骨架称其为 `final=1` 行，实测全表无此 key）

### 1.1 缺陷陈述

| 项 | 内容 |
|---|---|
| **声称方** | `docs/m12_criteria_table_skeleton.md` **L158**（§5.1 表的一行）：<br>`| `protocol_T_dwell_ratio_min` | L712（`T ≥ 3× 目标 dwell`） | **final=1**（唯一有值的 T 类常数） | 短窗不得判长时标 |`<br>同文 **L389**（§7 的「清单行 → 本表 key」映射表）：<br>`| L712 | `T ≥ 3× dwell` | `protocol_T_dwell_ratio_min` |` |
| **实测** | `data/m12_targets.csv` **124 key 中不存在该 key**（含 `dwell` 的 key **仅** `target_p10c_dwell_band`；含 `T_` 的 key **仅** `protocol_T_main_s`） |
| **⇒ 判定** | 骨架 §5.1 对它「**final=1（唯一有值的 T 类常数）**」的描述**与表内实际不符** —— 该常数**连行都没有**（不是 `final=0`，是不存在）⇒ **名存实亡** |
| **但它非惰性** | 该名字**被引用**（骨架 L158、L389）**且**它对应的规则**决定降档是否合法**：`docs/生物仿真M12实施清单.md` **L712** 逐字含「**不得**用短窗判长时标（**T ≥ 3× 目标 dwell** 纪律）」⇒ 依 **第八十三条**（被引用的名字必须处置）⇒ **必须处置**，不得默默留着 |

### 1.2 实测命令与输出（可复跑）

```text
$ cd /Users/weidong/ai/small_world/neural_exploration && python3 <读表并精确匹配>
keys= 124
[含 'dwell' 的 key] = ['target_p10c_dwell_band']
[含 'T_'    的 key] = ['protocol_T_main_s']
[含 'n_min' 的 key] = ['ref_p8_fit_set_vs_n_min', 'gate_n_min_coef_power', 'gate_n_min_ci_coef', 'gate_bootstrap_n_min', 'gate_perm_n_min']
⇒ 'protocol_T_dwell_ratio_min' 零命中
```

### 1.3 建议处置（**不选值**；三种处置方向，交主 agent 裁决）

| 选项 | 内容 | 代价 / 注意 |
|---|---|---|
| **(A) 文档更正（最小动作）** | 把骨架 **L158** 的「状态」列与 **L389** 的映射改为「**本表无此 key**（原『final=1』的说法为误）」；并把该名字登记为 **R188 ⑥ 的「B-6 范围边界待裁项」** | 不改表、不改判据；但清单 L712 的「T ≥ 3× 目标 dwell」仍是未注册常数（**缺口继续存在**，只是被如实标注） |
| **(B) 补登记为 append 新 key** | 走 **append-only** 新增 1 行（**绝不可**同 key 追加：G0 会以「重复 key」拒绝，R188 ① 已实测退出码 1） | 需要另一次裁决（R188 ⑥ 明示「**另立**为待裁项」）；且**其值当前无法定稿**：清单 L712 原文只有「T ≥ 3× 目标 dwell」这一**比值形式**，而分母「目标 dwell」依赖 `target_p10c_dwell_band`（`data/m12_targets.csv` **L24**，`value=unavailable`、`final=0`）⇒ 无论如何填，`final` 只能是 0 |
| **(C) 维持现状，仅在本文件登记** | 不改任何件 | 风险：接手者照骨架 L158 去找「final=1 的 T 类常数」会**找不到**（这正是第八十三条要防的形态） |

**⇒ 本节点的建议（不是裁决）**：先做 **(A)**（把错误描述改对，成本最低、不引入新值），把 **(B)** 作为独立裁决项；**本节点不选任何值、不改骨架/清单/表**。
**⇒ 与其值的可判性有关的一条事实**（供裁决参考，非建议取值）：清单里**唯一**的字面形式是 `docs/生物仿真M12实施清单.md` **L712** 的「**T ≥ 3× 目标 dwell**」；该形式的右端「目标 dwell」在 M12 侧**未定稿**（`target_p10c_dwell_band` L24 `final=0`，其 note 自述「M11 的 dwell 带自身即双口径（H 法 3.17–6.67 s / Q 法 1.11–1.27 s 相差 2.6–6×）」）⇒ **任何登记都必须先解决 dwell 的口径**。

---

## 2. `m12_targets.csv` L10 的 provenance：30 s 引用**不完整且区间端点写错**（漏 L848）

### 2.1 缺陷所在（**准确位置**：不是"注释行"，是**数据行 L10 的 `provenance` 字段**）

> 说明（本节点核实）：该说法位于 `data/m12_targets.csv` **第 10 行**（`protocol_T_main_s` 那一行）的 **`provenance` 字段**内，**不是**表首的注释行（L1–L8）。逐字原文如下：

```text
[checklist] docs/生物仿真M12实施清单.md L712（降档菜单 D5 行原文「T=30 s → 10 s」）+ L408（§1.4 合规命名例「30 s 窗」）；本节点实测 grep：§2–§5（L415–L986）内出现 T 数值的仅 L408/L712 两处
```

### 2.2 实测：三处不成立（逐条给证据）

| # | 缺陷 | 证据（实读） |
|---|---|---|
| **①** | **区间端点写错**：provenance 写「§2–§5（**L415–L986**）」，但 §5 止于 **L975**、**L976 起已是 §6** | 节边界实测（逐行匹配 `^## §`）：`SECTION L415: ## §2 数据门…`、`SECTION L827: ## §5 判据表…`、**`SECTION L976: ## §6 反证设计…`**；清单共 1203 行（`wc`） |
| **②** | **L408 根本不在该区间内**，却被列为「区间内」的一处 | `docs/生物仿真M12实施清单.md` **L408** 属 **§1.4**（§1 起于 L352、§2 起于 L415）⇒ **L408 < L415** |
| **③** | **漏列 L848**（真正的区间内漏项） | **L848** 逐字：`criterion_name,          # ★ 必须含实测范围，见 §1.4：'P10a-[IBL bw 可达 S_session/500 区域汇总,30s 窗] 区域率分布 TOST'`（含「30s 窗」）⇒ 在 L415–L975 内 |

**区间内逐行扫描的完整结果**（正则 `T *=|T≥|T ≥|T 取|30 *s|30s`，范围 L415–L975）：

```text
L712: | **D5** | 降协议（T=30 s → 10 s；条件数/试次数下调） | … （唯一「T=…」式左值）
L848: criterion_name,  # ★ 必须含实测范围，见 §1.4：'…,30s 窗] 区域率分布 TOST' （区间内、原 provenance 漏列）
L946: | **P5c** | 量纲/有限性 | 断言 | `率×T=计数`；… （量纲式里的符号 T，非数值）
```

### 2.3 建议更正文本（**草案，未就地改**；由主 agent 裁决后执行）

```text
[checklist] docs/生物仿真M12实施清单.md L712（降档菜单 D5 行原文「T=30 s → 10 s」）；本节点实测 grep：§2–§5（L415–L975）内出现 T 数值的仅 L712/L848 两处（L848 = §5.1 列定义注释的命名例「30s 窗」；L946 只含符号 T 的量纲式）；另 L408（§1.4 命名例「30 s 窗」）**不在 §2–§5 区间内**，仅作跨节旁证
```

> ⚠️ 注意：**L10 不在 R188 ⑦ 授权的 4 项之内**（授权项是 `protocol_settle_ms`／`protocol_drive_rate_hz`／`protocol_bin_wide_ms`／`protocol_smooth_window_frames` 与 `B6-P3` 的 append）
> ⇒ **本节点未改 L10**，候选表中该行**逐字节等于现表**（见 `b6_apply_notes.md` §5 的 diff 证明）。

---

## 3. `m12_targets.csv` **L13** 的 note：「M11 表亦无该行（两表同缺）」——**就 key 名成立、就「量」不成立**

### 3.1 缺陷所在（准确位置）

`data/m12_targets.csv` **L13**（`protocol_smooth_window_frames`）的 `note` 字段逐字含：

```text
口径要求存在但数值缺失（M11 表亦无该行 ⇒ 两表同缺 ⇒ 这是跨里程碑的缺口而非 M12 引入）
```

### 3.2 ★ 判断：**M12 的 `protocol_smooth_window_frames` 与 M11 的 `smooth_frames_state` 是否同量**

| 判断 | 结论 |
|---|---|
| 就 **key 名** 而言 | **成立**：M11 表（`data/m11_p5_targets.csv`，104 记录）中**确实没有** `protocol_smooth_window_frames` 这个 key（同名 key 缺失） |
| 就 **「量」** 而言 | **不成立**：M11 表**有**一个**同义但不同量**的行 `smooth_frames_state = 5`；且 M11 的 P5 协议侧另有一个**同口径**的 `smooth_frames = 5` ⇒ 「两表同缺」这一表述**过强** |
| **两者是否同一个量** | **不是同一个量**（数值相同（均为 5）纯属巧合；作用量相差 ≈ **9.1×**） |

**依据（两侧实读，逐条给行号）**

| 侧 | file:line | 逐字/关键内容 | 推出的量 |
|---|---|---|---|
| M11 表行 | `data/m11_p5_targets.csv` **L27** | `smooth_frames_state,5,point,frames,derived,docs/m11_acceptance_master.md R6/R7（**0.914 s 箱 + 5 帧平滑 + z 分数**；未平滑 φ=−0.41…−0.34）,状态/慢变量分析口径；本门（P5）不判慢状态，仅登记以保证与 S 门口径一致,把它当 P5 判据 ⇒ 越界（慢状态归 P6/S 门）,1` | **钙成像帧域**：帧长 = **0.914 s**（provenance 逐字「0.914 s 箱」）⇒ 5 帧 ≈ **4.57 s**；note 明文「**本门（P5）不判慢状态**」 |
| 该钙成像帧长的第二处锚 | `docs/生物仿真M11实施清单.md` **L584** | 「…（B4 冻结管线：**0.914 s 箱 → 5 帧平滑 → z → AR(1)）**…」 | 同上（帧 = 0.914 s 的成像采样帧） |
| M11 协议侧（**同口径**的先例） | `tools/make_m11_p5_plan_l0.py` **L173**、`tools/make_m11_p5_plan_g1.py` **L120**、`tools/make_m11_p5_plan_l3.py` **L728** | 三处协议 dict 内**并列**：`bin_ms=100.0, smooth_frames=5, estimator="per_neuron"` | **脉冲域 bin 序列**：帧 = **100 ms** ⇒ 5 帧 = **500 ms** |
| M11 运行/评估器的缺省 | `tools/run_m11_p5_arms.py` **L1123**（`smooth_frames=plan.get("smooth_frames", 5)`）、**L1427**；`data/m11_p5_plan_g1.json` **L10**（`"smooth_frames": 5`） | 缺省 5 | 同上 |
| **一条必须登记的口径风险** | `tools/validate_p11_resting.py` **L525** | `self.smooth_frames = proto.get("smooth_frames", tt.num("smooth_frames_state"))` | 评估器把**脉冲域**协议字段的缺省值**回退到钙域表行** `smooth_frames_state` ⇒ 这是"同名单位跨域继承"的实际发生处；**它正是本判断必须写进 note 的理由** |
| M12 侧 | `data/m12_targets.csv` **L13** | `unit=frames`；`scope` 的 `quad=region|all_arms|smoothed_rate|window_undefined`；note 自述「平滑帧数与 bin 宽混用 ⇒ 率的时间分辨率不可复现」 | **脉冲域区域率估计器**：帧 = 活动 bin ⇒ 5 帧 = 5 × `protocol_bin_wide_ms`（100 ms）= **500 ms** |

**⇒ 量化对比**：同称「5 帧」时，`0.914 s × 5 = 4.57 s` vs `0.1 s × 5 = 0.5 s` ⇒ **≈ 9.1×** ⇒ 两者**不同量**（若被当作同一个量使用，就是把「钙成像慢状态平滑」与「脉冲率平滑」混用）。
**⇒ 同量的否证方向（如实登记）**：M11 **另有**同口径的行外先例（协议侧 `smooth_frames=5` 与 `bin_ms=100.0` 并列，见上表第 3 行）⇒ M12 的该项与之**同口径**；这使「两表同缺」的说法在**量**的层面更不成立。

### 3.3 该判断已落地处

本判断**已写入候选表 L13 的 `note`**（R188 ⑤ 明示「该『是否同量』的判断**并入 ⑥ 的登记义务**」，R188 ⑦ 亦要求该判断写进该项的 note）——逐字见 `b6_apply_notes.md` §2.3 的「同量判断」段。

### 3.4 建议更正文本（**草案，未就地改**）

```text
口径要求存在但数值缺失。★ 更正（原表述「M11 表亦无该行 ⇒ 两表同缺」**就 key 名成立、就量不成立**）：M11 表 data/m11_p5_targets.csv L27 有 smooth_frames_state=5（final=1），但其「帧」是钙成像帧（provenance 逐字「0.914 s 箱」⇒ 5 帧 ≈ 4.57 s）且明文「本门（P5）不判慢状态」；M11 的 P5 协议侧另有一处同口径值 smooth_frames=5 与 bin_ms=100.0 并列（tools/make_m11_p5_plan_l0.py L173 等）⇒ 本行与后者同口径（脉冲域活动 bin ⇒ 5 帧 = 500 ms）、与前者不同量（相差约 9.1×）
```

> 该更正**同时是 M12 该项登记（`final=1`）的先决条件之一**：R188 ⑦ 把它列为「附口径声明义务」项，而口径声明的核心正是把「帧」钉到 `protocol_bin_wide_ms` 上（候选表 L13 的 note 已当场写明）。

---

## 4. 额外发现（本节点在取证过程中实测到的**其他**文档缺陷；均**未就地改**）

### 4.1 B-6 清单引用 M11 红线时**行号错**（`:84` → 真为 **`:109`**）

| 项 | 内容 |
|---|---|
| 缺陷所在 | `docs/m12_b6_unregistered_constants_checklist.md` **L142**（§2.2 #1 表内）与 **L400**（§6 U7）均写 `docs/m11_g1_resume_v2.md:84`，并把该行描述为「**只用协议时窗 `T=10 s`** 判率/比值/阈值」 |
| 实测 | 该文件 **L84** 逐字 = `> **要的是「~77–90% 神经元清楚发放（中位率 ≥0.5 Hz）∧ 一个 10–23% 的**完全静默亚群**」**`（讲的是目标形态，**不是**时窗口径）；该红线的**真行号 = L109**：`1. **只用协议时窗 `T=10 s`** 判率/比值/阈值；**短窗探针只能**推单价与稳定性（R130 第六十七条）；`（文件共 **205** 行；另 **L119** 为 R187 §② 的「率类判据须同时报告窗长」） |
| 影响 | **结论不受影响**（30 s vs 10 s 之争仍然成立），但引用行号须更正 ⇒ 属"引用不完整/错误"类 |
| 处置建议 | **不就地改**（B-6 清单是 node A 的交付物）⇒ 由主 agent 在裁决时顺带更正；本节点已在 `b6_deferred_evidence.md` §1.2 末登记该勘误 |

### 4.2 骨架 §5.1 的**行数描述**与表内实测不符（复核确认 B-6 §2.4-③ 的 U4）

| 项 | 内容 |
|---|---|
| 骨架声称 | `docs/m12_criteria_table_skeleton.md` **L148**：「本表据此登记了 **11 行**『协议/口径常数』（`protocol_*`），其中 **9 行是缺口行**（`value=unavailable`，`final=0`）：」 |
| 实测 | `data/m12_targets.csv` 中 `protocol_*` 共 **21 行**（`final=1` **8** 行 / `final=0` **13** 行）；13 行缺口名单 = `protocol_T_main_s`、`protocol_settle_ms`、`protocol_bin_wide_ms`、`protocol_smooth_window_frames`、`protocol_drive_rate_hz`、`protocol_drive_dose_grid`、`protocol_self_sustain_ratio_max`、`protocol_silent_frac_band`、`protocol_drive_elasticity_min`、`protocol_p6a_rho_min`、`protocol_acc_min`、`protocol_p9_consolidation_drop_min`、`protocol_pharma_dose_caliber` |
| 影响 | 不是数值结论错误，但**接手者按错误的行数去核对会得出错误结论** ⇒ 建议随骨架的更正件一并修正 |
| 处置建议 | **不就地改**（骨架非本节点所有件） |

### 4.3 表内 118 个字段含**半角双引号**，与表首 L5 的格式约定不符（不影响解析）

| 项 | 内容 |
|---|---|
| 约定 | `data/m12_targets.csv` **L5** 逐字：「字段内不使用半角逗号/半角双引号/换行 ⇒ 物理行数 == 记录数」 |
| 实测 | 124 行中，**含半角双引号的字段 = 118 个**（全部在既有文本内，例如 L11/L13/L14 的 `note`、L12 的 `wrong_usage_consequence`）；**含半角逗号的字段 = 0 个** |
| 影响 | **不影响解析**（`csv.DictReader` 对"字段中部"的 `"` 按字面处理 ⇒ 每行仍是 10 字段，G0 从未因此报错；本节点实测候选表同样 10 字段 × 125 行）。但该约定**字面上已被既有内容违反** ⇒ 属**文档↔内容漂移**；若未来出现"字段以 `"` 开头"的内容，约定将失效 |
| 处置建议 | 由主 agent 决定是（i）放宽 L5 的措辞，还是（ii）统一清理既有引号（后者会改已登记 sha 的行，须走独立裁决）⇒ **本节点不就地改**；本节点**自己的新增文本**遵守了「无半角逗号」的约定（候选表实测：含半角逗号字段 = 0） |

### 4.4 M11 注册行 `protocol_drive_mult_grid` 的 provenance **行号错**（写 `§4.5(d)`、实为 **L808**）

- 摘要：`data/m11_p5_targets.csv` **L19** 的 provenance 指 `docs/生物仿真M11实施清单.md §4.5(d)`；实测 §4.5(d) = **L862**、**不含该网格**；该网格的真实出处是**同清单 L808**（§4.2 机制自检门第 4 项，逐字 `{0, 0.5, 1, 2, 4, 8}`）。
- 影响：属**「值 ↔ 出处行号漂移」**（不是"值无依据"）⇒ 裁决剂量网格时**依据是存在的（L808）**。
- 完整证据与命令见 `b6_deferred_evidence.md` §2.3（本文件不重复）。

---

## 5. 本节点**没有**做的事（逐条）

1. **未就地修改任何文件** —— 尤其 `docs/m12_criteria_table_skeleton.md`、`docs/生物仿真M12实施清单.md`、`data/m12_targets.csv`、`docs/m12_b6_unregistered_constants_checklist.md`、`data/m11_p5_targets.csv`（本文件全部内容为**建议草案**）；
2. **未处置 `protocol_T_dwell_ratio_min`**（R188 ⑥：另立为「B-6 范围边界待裁项」，须待本更正件到位后**单独裁**）⇒ 本节点**未补登记该 key、未选任何值**；
3. **未运行任何仿真、未触碰 MPS/GPU、未联网**；
4. **未 `git add` / 未 `git commit`**；
5. 本文件所有行号均为本节点**实读**所得；凡未实读者未列入（无凭印象补的行号）。
