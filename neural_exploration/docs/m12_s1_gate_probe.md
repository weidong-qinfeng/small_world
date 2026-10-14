# M12 · S1 数据门当场裁决探针（工作包说明 + 实测报告）

> **节点**：M12 执行节点 **S1-GATE**（与 M11 G1 的 L3 跑臂**并行**运行；**全程未使用 MPS/GPU**）。
> **本文件只增不改**；所有数字均为**本轮当场实测**，凡未亲测的一律标 `未测` / `估计` / `unverified`。
> **口径**：每条断言 → 实测值 → pass/fail 见 §4（由 `data/m12_s1_gate_assertions.json` **机验**，18/18）；
> 报告引用计数一律引该 JSON，不引记忆（M11-R58/R63）。
> **§0.2.3 回读义务（清单第 122–123 行）已执行**：本轮依据 = **世界 B**（M11 G1/P5 仍在 L3 跑臂中、
> `docs/m11_report.md` 仍 **MISSING** ⇒ 现状即主假设；见 §7）。本工作包**与慢状态/调质层判据无关**，
> 故**不随世界 A/B 分支变化**。

---

## 1 我选的是哪个工作包（一句话）＋ 清单行号 ＋ 为什么满足 ①–⑤

**工作包：S1（Allen Mouse Brain Connectivity Atlas 投射矩阵）数据门的 T1–T6 当场实测与四态裁决
——把它从"按继承吞吐判的 `unavailable（吞吐限制）` 主假设"变成"按当场探针判的四态"。**

| 清单/评审件位置 | 内容（原文要点） | 本工作包如何落地 |
|---|---|---|
| `生物仿真M12实施清单.md` **第 86 行**（§0.2.2 **A-1**） | 「§2 数据门（G2-M）**全部**：Allen 图谱 / CCFv3 / 调质投射图谱 / IBL / 药理文献」列为**可并行项**；理由明写「数据门是**纯数据 + 元数据 + 实测吞吐**……数据门不读 M11 的 P5/P6 判定」 | 本工作包 = A-1 中**唯一**尚未被 B1 节点覆盖的高价值子项（B1 全节点的边界见其 L12/L15：只覆盖 S2/S7/S5） |
| 同上 **第 453–461 行**（§2.2 **S1**） | S1 的**预注册四态判据**；**第 460 行**给出数值条件；**第 461 行**给出四级降级路径 | §5 的四态裁决直接按第 460 行的**两个分支**逐条数值化（**不是我自定的判据**） |
| 同上 **第 429–436 行**（§2.3 **T1–T6**） | T1 计数 / T2 对象级 / T3 吞吐 3 轮中位 / T4 ETA 门 / T5 内容级 / **T6 装配可行性** | 六个步骤全部走完（Phase A/B/C/D/E 一一对应；T6 = §5.4） |
| `m12_b1_gate_closure.md` **§3 表首行（第 238 行）** | S1 仍缺**两维**：「**服务端白名单过滤后的行数实测**」与「**实测吞吐下的 ETA**」 | 这两维就是本工作包的**验收对象**（§3 断言 Z3/Z4/Z11/Z14） |
| `m12_list_review_master.md` **R41-2 条件 1（第 19–21 行）** | 「**S1/S3/S4/S6 的 `unavailable` 主假设必须在 M12 执行时以"当场 30 s 探针"确认后才生效**；探针结果与继承值不符时，**以探针为准并新增更正行**」 | 本轮**执行**该条件（对 S1）；结果见 §6（主假设被否证）＋ 跨节点更正行 `X1` |
| `m12_env_notes.md` **L25（第 558–562 行）** | 实测更正：「`S1-01` 计数查询在 90 s 窗内返回 200 ⇒ 该源**计数可达**，只是服务端生成慢；**S1 的 pass/fail 仍须由覆盖它的节点以"白名单过滤后行数 + 当场吞吐"两维实测裁决**（本节点只给两条证据，不越界下结论）」 | **B1 明确把 S1 的裁决留给"覆盖它的节点"** —— 本节点就是那个节点（这是选择本工作包的最直接依据） |

**为什么满足 ①–⑤**

- **① 不依赖 M11 的 G1 判定**：S1 的判据对象是 Allen RMA 的 `ProjectionStructureUnionize` 端点（行数/过滤/吞吐/对象语义）；
  M11 G1 的判据对象是**皮层引擎的生理工作点带**（阻塞项 = 钙域静默口径 J2）。**两者无交集**，本轮**未读**也未引用
  `m11_acceptance_master.md` 的任何门链结论（§7）。**不需等任何 G1 结果**。
- **② 不占 MPS**：全程纯 HTTP + 标准库；**无 torch、无 GPU、无仿真**。已**机验**（断言 **Z15**：
  AST 级检查 imports 无 `torch`、且零 `sys.executable` 用法）。
- **③ 有明确、可判定的验收标准**：判据不是我自定——是清单**第 460 行**的预注册数值条件
  （`pass` ⇐ 白名单行数 ≤ 10⁵ **或** 吞吐 ≥ 200 KB/s；`unavailable` ⇐ 行数 ≥ 10⁶ **且** 吞吐 ≤ 16 KB/s）
  ＋ **第 434 行** T4 门 ＋ **第 436 行** T6 门。可判定 = 产物（4 表 + 4 JSON + 1 CSV）+ 18 条机验断言 + 一条复算命令（§3）。
- **④ 能消掉一个「未覆盖维度」**：`b1_gate_closure.md` §3 表首行点名的**两维**本轮都测了
  （行数：A3–A6 + A7 交叉校验；吞吐/ETA：C + F）；并把三处「未测」变成「已测」
  （见 §5.5：`$in[...]` 静默陷阱、Range 被忽略、服务端 60 s 生成上限 + 分页可用）。
- **⑤ §0.2 矩阵标注为「可并行/无等待」**：A-1（第 86 行）明列。且**不触碰** B 组任何一项
  （B-1…B-7 均与本工作包的对象无关）。

> **未选其他工作包的理由（一句话）**：A-2（§3 引擎门 ms/step 探针）**占 MPS**（违反硬约束 1）；
> A-3/A-6（装配）依赖 G2-M 的区域表且属 `src/` 引擎侧（本轮不能改/起仿真）；
> A-4/A-5 是行为闭环与判据脚手架，**B1 已就 S5 给出了行为侧的可计算性证据**，边际价值低于 S1（S1 是 P6/C-M 的**唯一**结构输入，清单第 947 行）。

---

## 2 判据（预注册，事中不得改）

**源**：`https://api.brain-map.org/api/v2/data/query.csv|query.json?criteria=model::ProjectionStructureUnionize,rma::criteria[...],rma::options[...]`

**预注册区域白名单**（`data/m12_s1_region_whitelist.csv`；规则 = 只依 S2 已落盘件
`data/m12_raw/s2/model_Structure_all.json` 的 `graph_id=1` + CCFv3 `depth` 阈值；**出数前定死**，重跑确定性已机验 **Z16**）：

| 白名单 | 规则 | 区域数 | 定位 |
|---|---|---|---|
| `W_DEPTH_LE2` | `depth ≤ 2` | **24** | §2.2 第 461 行降级路径①「~50 个区域族」的下界 |
| `W_DEPTH_LE3` | `depth ≤ 3` | **85** | 族级上界候选 |
| `W_DEPTH_LE5` | `depth ≤ 5` | **324** | 500 区量级的下界 |
| `W_DEPTH_LE6` | `depth ≤ 6` | **586** | 清单「~500 区域」的最近档（**主白名单候选**） |

> ⚠️ **本白名单不是 Oh 2014 论文的区域集**——它是我按 CCFv3 层级做的**可复现预注册**；
> 两者是否一致**未测**（§8 U7）。

**四态判据（原文转载，逐条数值化）**

- **第 460 行** 🔹 `pass` ⇐ **服务端白名单过滤后行数 ≤ 10⁵**（≈10 MB）**或** 实测吞吐 ≥ 200 KB/s（≤15 min）
- **第 460 行** 🔸 `unavailable（吞吐限制）` ⇐ 行数 ≥ 10⁶ **且** 实测吞吐 ≤ 16 KB/s（ETA 17–35 h）
- **第 434 行** T4 门：`ETA_对象 > 30 min` 或 `ETA_总计 > 4 h` ⇒ `unavailable（吞吐限制）`；`30 min < ETA ≤ 4 h` ⇒ 需主 agent 托管作业 + 断点续传
- **第 436 行** T6 门（M12 新增第 4 层）：取得到但**装不进模型** ⇒ `unavailable（装配不可行）`
- **两分支均不满足 ⇒ 必须判 `NEEDS_VERIFY` 并上报，不得自选**（本工具已如此固化）

---

## 3 验收标准（可判定：产物 + 断言 + 复算命令）

### 3.1 产物

| 产物 | 内容 |
|---|---|
| `data/m12_s1_gate_reachability.csv`（+ `.history.csv`） | 逐 spec 实测（35 行）：T1 计数 / 对象级 / 分页 / 跨节点更正；每行带 `utc`/`provenance`/`recheck_policy`/`wrong_usage_consequence`/`missing_dimension`/`total_rows` |
| `data/m12_s1_gate_throughput.csv`（+ `.history.csv`） | 吞吐实测（有效页多轮 + 中位）＋ `probe_or_fetch` 口径列 |
| `data/m12_s1_gate_integrity.csv`（+ `.history.csv`） | 完整性门（三级）逐件：`size==CL` / ETag md5 复算 / sha256 / **语义锚** |
| `data/m12_s1_gate_syntax.csv`（+ `.history.csv`） | RMA **查询语法**实测（9 行）：`success` 字段 + 错误文本（HTTP 200 ≠ 成功） |
| `data/m12_s1_gate_counts.json` | 机器可读计数锚（供报告引用，防"引陈旧数字"） |
| `data/m12_s1_gate_assembly.json` | **四态裁决** + T4/T6 门逐条 + 逐白名单分页 ETA + 降级路径可行性 |
| `data/m12_s1_gate_assertions.json` | **18 条机验断言**（断言/实测值/`pass_`） |
| `data/m12_s1_region_whitelist.csv` | 预注册区域白名单（确定性可重生成） |
| `data/m12_s1_raw/**` | 落盘对象与原始响应（含 E1/E2/E3 整件 CSV、各页 CSV、响应头、`A*/B*/F*_body.json`） |
| `tools/probe_m12_s1_gate.py` | 本工作包的全部探针（**只用标准库**；硬回收 + TTFB + 多轮中位 + `success` 检查） |

### 3.2 断言（18 条，机验；见 §4）

### 3.3 复算命令（单一入口，可整段复制）

```bash
cd /Users/weidong/ai/small_world
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# ① 断言表（**不需要网络**；只读已落盘产物 ⇒ 主 agent 可独立复核）
python3 -c "import json;d=json.load(open('neural_exploration/data/m12_s1_gate_assertions.json'));\
print(d['n_pass'],'/',d['n'],'pass');\
[print(' ',a['id'],'PASS' if a['pass_'] else 'FAIL',a['claim'][:64]) for a in d['assertions']]"

# ② 四态裁决（不需要网络）
python3 -c "import json;d=json.load(open('neural_exploration/data/m12_s1_gate_assembly.json'));\
print(d['four_state']);print(d['clause_460']);print(d['clause_460_unavailable']);print(d['paged_ETA'])"

# ③ 全程重跑（**需要网络**；会重新探测。注意：含两次 ~60 s 的 504 等待）
python3 -u neural_exploration/tools/probe_m12_s1_gate.py --all
```

**复算口径声明**：`②`/`③` 中**唯一**会变的是 `utc`/`ttfb`/`seconds`/`bytes_per_s` 等**时间相关列**
（吞吐按 M11-R56/R63 本就不可外推，每行已带 `recheck_policy = 须以当场探针复核（≤24 h）`）；
**计数类与结构类字段（`total_rows`/`success`/行数/对象字节/判定）应逐值相同**。

---

## 4 实测结果（逐条断言 → 实测值 → pass/fail）

### 4.1 断言表（**18/18 pass**，机读 `data/m12_s1_gate_assertions.json`）

| id | 断言（要测什么） | 实测值 | 判定 |
|---|---|---|---|
| **Z1** | S2 结构表计数（本轮）== S2 已落盘件 `total_rows`（跨节点一致性） | A1 = **13,709** == 磁盘 **13,709** | ✅ pass |
| **Z2** | PSU 全表行数 ≥ 10⁶（第 460 行 unavailable 条件的**第一半**） | A2 = **16,357,801** | ✅ pass |
| **Z3** | **白名单过滤语义成立**：Σ(逐 id 单独计数) == `count($in<同一白名单>)` | Σ = **396,424** vs `$in` = **396,424**（24 ids，逐值相等） | ✅ pass |
| **Z4** | 白名单层级嵌套 ⇒ 计数单调不减 | 396,424 < 1,009,176 < 4,185,500 < 8,162,198 | ✅ pass |
| **Z5** | `$in[...]`（带方括号）语法错误、`$in...`（逗号列表）成功；**HTTP 均 200，只有 `success` 能区分** | A14 `success=false`（323 B 错误信封，`no viable alternative at input "2"`）／A15 `success=true`，total_rows=38,626 | ✅ pass |
| **Z6** | 复现 B1 表 `S1-03` 同 URL：bytes == B1 记录值 **且** `total_rows == 0` | 本轮 **85 B** == B1 表 **85 B**；`total_rows=0`；`success=true` | ✅ pass |
| **Z7** | T5 语义锚：单实验整件 CSV 解析行数 == 计数查询行数 | 解析 **2,574** == 计数 **2,574** | ✅ pass |
| **Z8** | 重跑确定性（清单 §2.5 条件④）：两次取数 sha256 一致 | `3b882148a1d47a3a…` == `3b882148a1d47a3a…` | ✅ pass |
| **Z9** | 该端点**忽略 Range**：`-r 0-1023` 实际返回**完整对象** | B1 落盘 **354,941 B** == E1 无 Range 整件 **354,941 B**，sha256 逐字节一致 | ✅ pass |
| **Z10** | 大结果集**整件**请求 ⇒ 504（服务端生成上限）；B3/B4 登记为**未测**（不重试，理由见 §6.4） | B2 = **HTTP 504**，TTFB **60.60 s**，132 B | ✅ pass |
| **Z11** | **分页语法** `rma::options[num_rows$eqN][start_row$eqM]` 可用；末尾页锚 `start_row = 总行数−500` **恰好 500 行** | 最大可取页 = **10,000 行**；末尾页 parsed = **500**（期望 500）；中段页（start_row=200,000）= **2,000** | ✅ pass |
| **Z12** | `is_injection=true` 抽样行同属一个 sds，但其 structure_id **不构成单一祖先链** ⇒ 「按 sds 取 `is_injection=true` 反推单一注射源区」**被证伪** | sds = `170505811`（10/10 行同一实验）；structures = {8,28,60,95,111,131,163,234,247,284}；**单链 = False**（本地按 `structure_id_path` 核） | ✅ pass |
| **Z13** | 抽样实验的 `is_injection=true` 行数 == 0 且 `success=true`（**真 0** 而非语法错） | 同实验计数 = **0**（`success=true`）；全表 = **342,475** | ✅ pass |
| **Z14** | 实测吞吐（**有效页**上的多轮中位）可得；`bytes/row` 由整件实测得出 | 中位 **275,148 B/s**（逐轮 340,954 / 275,148 / 251,157）；`bytes/row` = **137.895** | ✅ pass |
| **Z15** | 本工具**只用标准库**（无 torch / 无 MPS / 无 GPU）且**零** `sys.executable`（AST 级检查） | imports = csv/hashlib/json/os/subprocess/sys/time/datetime/argparse；`sys.executable` 出现 **0** 次 | ✅ pass |
| **Z16** | 白名单预注册件**确定性可重生成** | 重生成 sha256 == 已落盘 sha256 | ✅ pass |
| **Z17** | 产物表非空且 spec 无重复 | reach 35 / syntax 9 / thru 4 / integ 13 行 | ✅ pass |
| **Z18** | 四态裁决由**预注册条件**推出（非自选），触发条件可回读 | `pass` ← 「第 460 行 pass 条件至少一支满足：实测吞吐 ≥ 200 KB/s（实测 275,148.0）」 | ✅ pass |

### 4.2 关键计数（可直接引用的单一来源：`data/m12_s1_gate_counts.json`）

| 量 | 实测值 | 说明 |
|---|---|---|
| `ProjectionStructureUnionize` 全表行数 | **16,357,801** | T1；清单第 458 行原估"可能达 10⁶ 量级" ⇒ 实测为 **1.64×10⁷** |
| 白名单过滤后行数（**目标结构维**） | **396,424**（24 区）／1,009,176（85 区）／4,185,500（324 区）／8,162,198（586 区） | 第 460 行第一支判据的输入 |
| 单实验行数（`section_data_set_id$eq166218353`） | **2,574** | = **859 个不同目标结构 × 3 半球**（E4 实测 859、hemi={1,2,3}） |
| `is_injection=true` 行数（源侧） | **342,475** | 源维锚；语义未确认（§8 U1） |
| `is_injection=false` ∧ 586 区白名单 | **7,952,265** | A11；与 A6 之差 = 209,933 = 586 区内的源侧行 |
| `is_injection=true` ∧ 85 族白名单 | **86,299** | A12 |
| `SectionDataSet` 全表 | **170,036** | B1 表 S1-04 记的是"19313"（= 响应字节数，非行数）⇒ 见 §6.5 |
| 实验数（**推算，非实测**） | ≈ **6,355** = 16,357,801 ÷ 2,574 | 标注为推算；逐实验分布未测（§8 U3） |
| CSV `bytes/row`（实测） | **137.895 B** = 354,941 ÷ 2,574 | |
| 实测吞吐（有效页 3 轮中位） | **275,148 B/s** ≈ 268.7 KiB/s | ≥ 200 KB/s ⇒ 第 460 行 pass 支成立 |
| 估计对象体量（**估计值，不得当实测**） | 54.66 MB（24 区）／1.126 GB（586 区）／**2.256 GB（全表）** | 单价 × 行数 |

### 4.3 T4 门的**档位依赖**（这是本工作包最可操作的一张表）

| 白名单 | 行数 | 分页数（10,000 行/页） | 分页 ETA（实测单页墙钟 3.929 s） | 第 434 行判定 |
|---|---|---|---|---|
| `W_DEPTH_LE2`（24 区） | 396,424 | 40 | **2.62 min** | ✅ pass（≤30 min） |
| `W_DEPTH_LE3`（85 区） | 1,009,176 | 101 | **6.61 min** | ✅ pass |
| `W_DEPTH_LE5`（324 区） | 4,185,500 | 419 | **27.44 min** | ✅ pass（**贴边**，余量 2.6 min） |
| `W_DEPTH_LE6`（**586 区**） | 8,162,198 | 817 | **53.50 min** | ⚠️ **>30 min ⇒ 需主 agent 托管作业 + 断点续传**（**不是** unavailable） |

> **⇒ 对主 agent 的具体裁决输入**：M12 若想一次装配**586 区**的投射矩阵，必须走**托管作业**；
> 若走 **≤324 区**（`W_DEPTH_LE5`）则可在节点内完成（27.4 min，余量很薄，建议仍托管）。
> **该结论依赖"单页墙钟 3.929 s"这一当场值**，按 M11-R56/R63 须 ≤24 h 复核。

---

## 5 本轮的五个"未测 → 已测"（含两处**会造出假结论**的陷阱）

### 5.1 ★ `$in[...]` 带方括号是**静默陷阱**（清单第 457 行模板的自然写法）

**实测**：`rma::criteria[structure_id$in[16997,1,2]]` ⇒ **HTTP 200** + `success: false` +
`total_rows: 0` + 323 B 错误信封（`line 1, column 73: no viable alternative at input "2"`）；
`rma::criteria[structure_id$in16997,1,2]` ⇒ `success: true`、`total_rows: 38,626`。

**后果**：只读 `total_rows` 的实现会把**查询写错**读成**「该过滤下没有数据」**（静默失效家族第 6 例）。
**固化**：本工具强制同时解析 `success` 与错误文本（`data/m12_s1_gate_syntax.csv` 逐行留痕）。

### 5.2 ★ 多 facet 过滤必须写成**多个 `rma::criteria[...]` 组**

**实测**：`rma::criteria[is_injection$eqfalse],rma::criteria[structure_id$in<586 区>]` ⇒ `success=true`、7,952,265；
同 facet 逗号连接写法 ⇒ **语法错误**（`no viable alternative at input "6144542…"`）；
`$eq1/$ne1` 写法 ⇒ 同样失败。（`data/m12_s1_gate_syntax.csv::A11/A11b/A11c/A12/A12b/A12c`）

### 5.3 ★★ 该端点**忽略 Range**，且**大结果集整件请求 HTTP 504**（服务端 60 s 生成上限）

- **忽略 Range**（断言 Z9）：「`-r 0-1023`」返回 **200**（非 206）、无 `accept-ranges`，**体就是整件**。
  ⇒ 清单第 579–582 行的 T3 模板（`-r 0-1048575` 取 3 轮中位）**在 S1 上口径不适用**。
- **60 s 生成上限**：`is_injection$eqtrue`（342,475 行）整件请求 ⇒ **HTTP 504，TTFB 60.60 s**、132 B。
- **本机冻结型挂死（实测）**：上一轮对 396,424 行对象的整件请求 ⇒ **936.2 s / 0 字节 / curl exit 28**，
  且**父进程 `communicate(timeout=190)` 的硬回收也未按时触发**（`hard_killed=false`）
  ⇒ 与 B1 **L19（第 439–448 行）/ L24** 记录的"整机冻结"**同型**。按 **M12-M-01/L24** 判
  `probe_window_too_short`／冻结，**排除出统计**，**不得**读成"该对象不可得"。
  > 该行完整留在 `data/m12_s1_gate_reachability.csv.history.csv`，键名 `_obj_B3_prior_freeze`。
- **⇒ 正确的 ETA 口径 = 页数 × 实测单页墙钟**，**不是**"对象字节 / 链路吞吐"。本工具的
  `data/m12_s1_gate_assembly.json::paged_ETA_basis` 明确写了这一点。

### 5.4 ★★ **分页可用**（把"C 不可行"变回"可行"的决定性一维）

**实测**：`rma::options[num_rows$eqN][start_row$eqM]`（**每项一个方括号组**；逗号连接会解析失败）：

| 页参数 | 结果 |
|---|---|
| `num_rows$eq2000][start_row$eq0` | **2,000 行**解析成功，290,162 B，墙钟 2.543 s |
| `num_rows$eq10000][start_row$eq0` | **10,000 行**，1,443,421 B，墙钟 **3.929 s** |
| `num_rows$eq50000` | ❌ 非 PSU CSV（错误页；首行列数 = 2）⇒ 判 `invalid` |
| `num_rows$eq200000` | ❌ 同上 |
| `num_rows$eq2000][start_row$eq395924`（= 396,424 − 500） | ✅ **恰好 500 行**（末尾页锚：`start_row` = 有序结果的**行偏移**，且与计数自洽） |
| `num_rows$eq2000][start_row$eq200000` | ✅ **2,000 行**（**随机定位**成立，非只有首页可取） |

### 5.5 ★ 自查拦下的第 7 类静默失效（**我自己的首版探针犯的**，留档）

首版 `page_fetch` 用朴素 `csv.DictReader` 统计行数 ⇒ RMA 的**语法错误页**（HTTP 200 + 两行文本：
错误描述 + 回显查询串）被读成 **2 条"数据行"**，于是得到 `parsed=2`、`bytes≈495`，
工具**一度据此推出"分页返回 2 行、页大小参数生效"的错误结论**（并把它写进了 `F_max_page_rows`）。
根因与 B1 L6/L25（错误体落盘成假件、`--fail` 吞掉 403 正文）**同族**：**HTTP 200 与"拿到数据"是两件事**。
**固化**：新增 `read_psu_csv()`——任何 PSU CSV 必须**先核列名 == 19 字段 schema**，否则判 `invalid` 且不入统计（第 13 条纪律）。
被取代的键已从 `counts.json` 显式清除，历史行仍在 `.history.csv`。

---

## 6 四态裁决（结果）

### 6.1 裁决

```
四态 = pass
触发 = 第 460 行 pass 条件至少一支满足：实测吞吐 ≥ 200 KB/s（实测 275,148.0 B/s）
```

| 判据 | 实测 | 判定 |
|---|---|---|
| **第 460 行 pass**：白名单过滤后行数 ≤ 10⁵ | **396,424**（最小族白名单 24 区） | ❌ 不满足 |
| **第 460 行 pass**：实测吞吐 ≥ 200 KB/s | **275,148 B/s** | ✅ **满足** ⇒ 判定成立 |
| **第 460 行 unavailable**：行数 ≥ 10⁶ **且** 吞吐 ≤ 16 KB/s | 行数 16,357,801 ✅／吞吐 275,148 B/s ❌ | ❌ **不成立**（吞吐比阈值高 **16.8×**） |
| **第 434 行 T4**：ETA_对象 > 30 min | 24 区对象 = **2.62 min**；586 区 = 53.50 min | 24/85/324 区 pass；**586 区需托管**（非 unavailable） |
| **第 434 行 T4**：ETA_总计 > 4 h | 全表（估计 2.256 GB）≈ **2.28 h** | ✅ pass |
| **第 436 行 T6**：装配可行性 | R=586 的 R×R float64 = **2.747 MB**；单实验 2,574 行；`projection_volume>0` 占 99.0% | ✅ 装得进（**全脑真实装配未做**，§8 U6） |

### 6.2 ⇒ 对清单 §2.2 S1 主假设的处置（R41-2 条件 1 要求的"新增更正行"）

**S1 的 🔸 `unavailable（吞吐限制）`（主假设）当场被否证**：其前提是继承吞吐「`api.brain-map.org` 1.6–16 KB/s」
（清单第 459 行，`unverified（继承）`）⇒ 本轮当场实测 **275,148 B/s**，高 **16.8–168×**，
与 B1 在同通道（S2/S7/S5）的实测更正（197 KB/s / 602–730 KB/s / 322 KB/s，见 B1 L9 表）**同向**。
更正行已落 `data/m12_s1_gate_reachability.csv::D1`（`kind=verdict`）与 `::X1`（跨节点）。

### 6.3 ⇒ 对降级路径的影响（清单第 461 行）

| 降级路径 | 本轮实测结论 |
|---|---|
| ① **区域族汇总** | **可行且已量化**：24 族（depth≤2）= 396,424 行 ⇒ 40 页 / **2.62 min**；85 族 = 1,009,176 行 ⇒ 6.61 min。**但** 24 族的最小值仍 **>10⁵** ⇒ 路径① **不能**把行数压到第 460 行 `pass` 第一支的 10⁵ 以下（**该支在 S1 上不可达**，这本身是一条结论） |
| ② **种子实验子集** | **可行且已量化**：单实验 = **2,574 行 / 354,941 B**（实测整件，含 sha256）⇒ 50 个实验 ≈ 17.7 MB / 128,700 行 |
| ③ **文献档（Oh 2014 补充材料）** | **未测**（§8 U8） |
| ④ **距离核抽象** | 非数据依赖（清单第 481 行 S3 已给质心距离降级） |
| **⑤（本轮新增）服务端分页** | **实测可用**（§5.4）⇒ **优先级建议**：①/② 之上再加"分页整取白名单对象"，使"取不到"不再成立 |

### 6.4 ⇒ **源（行）维未建立**（本轮最重要的"负面但确定"的结论）

R×R 投射矩阵需要**每个实验的注射源区**。实测发现该表**不足以**给出它：

1. `A0_body.json` 的 **19 个字段中没有** `injection_structure_id`（字段实测为：
   `id, section_data_set_id, structure_id, hemisphere_id, is_injection, sum_pixels,
   sum_projection_pixels, sum_pixel_intensity, sum_projection_pixel_intensity,
   projection_density, projection_intensity, projection_energy, volume, projection_volume,
   normalized_projection_volume, max_voxel_density, max_voxel_x/y/z`）⇒ 源维只能从 `is_injection` **布尔列**反推。
2. **抽样 10 个 `is_injection=true` 行同属一个 sds**（`170505811`）但其 `structure_id`
   = {8, 28, 60, 95, 111, 131, 163, 234, 247, 284} **不构成单一祖先链**（本地按 `structure_id_path` 核对：
   它们共享前缀 `/997/8/567/688/` 后**分叉**到 isocortex / piriform / amygdala / entorhinal 等**不相关分支**）
   ⇒ 断言 Z12：**「按 sds 取 `is_injection=true` 反推单一注射源区」被证伪**。
3. 抽样实验 `166218353` 的 `is_injection=true` 计数 = **0**（`success=true`，**真 0**）⇒
   该布尔列**不是逐实验常量**（全表却有 342,475 行为 true）⇒ 无法用"每实验恰有一行"的规则建源维。
   （E3 = 0 行与 F5 计数 0 相互独立地一致 ⇒ 不是单点偶然。）

**⇒ 结论**：清单第 947 行 **P6a（区域间耦合 ↔ 图谱权重）所需的 `m12_connectivity_matrix.csv`**，
其**行（源）维**在本轮**未建立**；须另走 `SectionDataSet` 的注射元数据（`SectionDataSet` 全表
= **170,036** 行，已实测可达；但**该 join 本节点未测**）。**这不影响 §6.1 的四态 `pass`**
（四态判的是"该源作为 M12 数据源是否可取"，不是"矩阵是否已建好"），但**必须**作为前置条件写进
P6/C-M 的开工条件。

### 6.5 跨节点更正（`X1`）：B1 表 `S1-03` 行是**空结果信封**

| 项 | B1 表 `m12_source_reachability.csv::S1-03` | 本轮同 URL 复现 |
|---|---|---|
| URL | `…structure_id$eq16997],rma::options[count$eqtrue]` | 同 |
| `bytes` | **85** | **85**（逐字节相同） |
| `verdict` | `REACHABLE` | —（本节点判 `NEEDS_VERIFY`） |
| `missing_dimension` | **无**（`verdict_scope` 写"对象级/内容级均已实测"） | `total_rows` **在本表里没有列** |
| `total_rows` | **该表无此列** | **0**（`success=true` ⇒ 空结果信封，非语法错） |
| B1 自注 | 「若返回 4xx/**空** ⇒ 记 `REACHABLE_NO_DATA` 并换 `StructureUnionize` 变体」 | 与其 `verdict=REACHABLE` **不一致** |

**处置（不越权改写他人产物）**：按 append-only 纪律，本节点**只**在自有产物里新增更正行 `X1`，
并在此登记：① `structure_id=16997` 在 PSU 里**确实 0 行**（B1 的"白名单口径试算"**没有**给出任何行数信息）；
② **`m12_source_reachability.csv` 没有 `total_rows` 列** ⇒ B1 文档 §3 表首行点名要补的
「服务端白名单过滤后行数」这一维**在原表里连落点都没有** —— 这是该缺口的**结构性**原因，
建议主 agent 在收口时给该表**新增列**（本节点的 `total_rows` 列可作模板）。
③ 附带更正：B1 表 `S1-04`（label 写"`SectionDataSet` 行数（**实验数锚**）"）的 `bytes` = **19,013**
（那是**响应字节数**；其 `speed_Bps` = 19,313 亦非行数），且其 `note` 只写"计数查询成功（成本极小）"、
**没有**记录 `total_rows`（**该表无此列**）⇒ 该行**未给出**实验数锚。
本轮实测 `SectionDataSet` 行数 = **170,036**（`A9`，`success=true`）。

---

## 7 与 M11 的关系（独立性论证 + 隐含依赖点名）

**一句话**：本工作包的判据对象是 **Allen RMA 的 `ProjectionStructureUnionize` 端点**（行数 / 过滤语义 /
吞吐 / 对象字节 / 分页），与 M11 G1 的判据对象（**皮层引擎的生理工作点带**，阻塞项 = 钙域静默口径 J2）
**无任何交集**；本轮**未读**、**未引用** `m11_acceptance_master.md` 的任何门链结论，
**不需要 G1 的任何输出**，因此**可以且已经与 M11 的 L3 跑臂并行完成**。

**§0.2.3 回读义务（清单第 122–123 行）的执行记录**：
- `docs/m11_report.md` ⇒ **仍不存在**（`ls` 实测：`No such file or directory`）⇒ 印证清单第 68 行的"上游未交付"；
- `docs/m11_acceptance_master.md` ⇒ 存在（471,548 B，mtime 10-14 09:21，**仍在被更新**），
  其最新内容显示 **G1/P5 仍在 L3 跑臂**（`docs/m11_l3_resume.md` §八：第 2 步等 C2 格三判据）
  ⇒ **G1 结果未定**；按清单第 120 行，现状 = **世界 B（主假设）**。
- **本工作包不随世界 A/B 变化**：世界分支只影响 M12 的**慢状态/调质层**语义（清单第 117–120 行），
  与本工作包（结构投射矩阵的取数可行性）**无关**。

**隐含依赖（点名，逐条说明降低了什么风险）**

| 隐含依赖 | 是否影响本轮判据 | 说明 |
|---|---|---|
| 清单 **第 97 行 B-1**：M12 的**皮层区域率带**须用 M11 G1 同口径先验 | **不影响** | 本轮**不产出**任何判据带、不写 `data/m12_targets.csv`，只做"取数可行性"四态 ⇒ 不触碰 B-1 |
| 清单 **第 98 行 B-2**：慢变量/调质层机制与参数 | **不影响** | 本轮不涉及慢变量 |
| `src/mouse_cortex_engine.py`（M11 G0 pass 定稿档） | **不依赖** | 本轮**未 import 任何 `src/` 模块**（断言 Z15：imports 仅标准库） |
| M11 的**数据门**结论（非 G1） | **弱依赖，已降低风险** | 本轮唯一复用的 M11 侧文件是 `data/m12_raw/s2/model_Structure_all.json`（**B1 节点**的 S2 落盘件，非 M11 件），用于生成白名单；并做了**跨节点一致性断言 Z1**（13,709 == 13,709）⇒ 该依赖已自证 |
| M11 的**吞吐**继承值 | **已解耦** | 本轮**不继承**任何吞吐（M11-R56/R63 + R41-2 条件 1）⇒ 全部吞吐/ETA 为当场实测，每行带 `recheck_policy` |

> **风险降低量（本次实际消除的）**：R41-2 条件 1 对 S1 一直**未执行**（B1 按 M11-R40"一判据一作用域"
> 明确留给"覆盖它的节点"）。本节点执行后，S1 从"**按未实测的继承吞吐判的 unavailable**"
> （会把 P6/C-M 连带判 `unavailable`、并使 P7/P10/P11 全部禁止声称"图谱权重约束下"，清单第 600 行）
> 变为"**当场实测的 `pass`**"⇒ **解除了一条可能连锁废掉整个里程碑区域级判据的假阻塞**
> （与 B1 L3 发现的 S2 假阻塞**同型**，这是该风险的**第二个实例**）。

---

## 8 未覆盖 / 未确认项（逐条；含"卡在哪一行/哪个字段"）

| # | 未确认项 | 卡在哪里（精确到行/字段） | 处置 |
|---|---|---|---|
| **U1** | **`is_injection` 的粒度语义 ⇒ R×R 的源（行）维未建立** | ① `data/m12_s1_raw/A0_body.json` 的 19 字段**无** `injection_structure_id`；② `counts.json::_F4_is_single_chain = false`（10 个 structure_id 非单链）；③ `counts.json::_F5_same_exp_injection = {total_rows: 0, success: true}` | **不改判据、不臆造规则**；登记为 P6/C-M 的**前置条件**：须实测 `SectionDataSet` 的注射元数据 join（**未测**）。**这不改变 §6.1 的 `pass`**（四态判的是取数可行性） |
| **U2** | 最大可取页的**精确临界**未定（10,000 可取；50,000/200,000 ⇒ 非 PSU CSV 错误页） | `counts.json::_F1_sweep` 的 `F1_n50000`/`F1_n200000`（`rej` = 非 19 字段；`bytes` 272/274） | 已用**实测最大值 10,000** 做 ETA（**保守**：临界可能更高 ⇒ ETA 只会更短，不会更长） |
| **U3** | `is_injection=true` 的**逐实验分布**未测 | 缺 group-by 能力；`A10 = 342,475` 与 `A8 = 2,574/实验` 的比值**不能**定出"每实验行数"（`342,475 / 6,355 ≈ 53.9` 但 6,355 本身是**推算**） | 登记为未测；**不得**据此下任何"源维已建立"的结论 |
| **U4** | 586 区对象与**全表**对象的整件总长**未实测** | `counts.json::_obj_B3_prior_freeze`（936.2 s / 0 字节 / exit 28 = 冻结型挂死）⇒ 按 M12-M-01/L24 **排除出统计** | 只给**估计**（54.66 MB / 2.256 GB，`counts.json::B_estimate`，已标注"不得当实测"）；替代口径 = 分页实测 |
| **U5** | 全表 16,357,801 行的**分页可穷尽性**未验（只验了 3 个页位置 + 末尾锚） | `counts.json::_F1_sweep`/`_F2_tail500`/`_F3_mid` 仅 3 个位置 | 末尾页锚（恰好 500 行）是该维的**最强可得证据**；穷尽性属"取数时才知道"，登记为未测 |
| **U6** | **全脑 586×586 的真实装配未做**（只做了单实验透视） | `counts.json::_E4_assembly`（`R_whitelist=586`、单实验 2,574 行、`ok=true`） | 内存 2.747 MB 为**实测输入**下的**推算**；T6 门的"装配"结论为**条件性 pass** |
| **U7** | 本预注册白名单（CCFv3 depth 阈值）**是否等于 Oh 2014 论文的区域集**未测 | 清单第 458 行"CCF 结构在其查询层级为**数百**"一句**无出处**；`A7` 的 24 个 id 中 **6 个为 0 行**（`124/624/1024/1032/1040/304325711`）⇒ PSU 覆盖的 `structure_id` 集合**与 CCFv3 graph1 不等** | 登记为未测；提醒：**0 行 = 该结构不在 PSU 里**，**不是**"该结构无投射"（"未测 ≠ 未命中"的**计数版**） |
| **U8** | 降级路径③（Oh 2014 补充材料矩阵，`literature`）**未测** | 清单第 461 行③；本轮未探 `nature.com` 补充材料对象 | 登记为未测（非"不存在"）；B1 表 `S1-07` 只做了 `HEAD`（`clen=0`） |
| **U9** | 吞吐的**时效性** | `counts.json::_obj_B2.ttfb` 与 `C_speeds_Bps` 均为**当场值** | 每行落 `recheck_policy = 须以当场探针复核（≤24 h）`（M11-R56/R63） |

---

## 9 产物清单（含 `wc -l -c` + `shasum -a 256`）

见本文件末尾由 `shasum -a 256` 现场生成的回读表（§9.1）。**报告"已写入"以该表为准**（纪律第五十九条）。

### 9.1 回读表

<!-- SHASUM_TABLE_BEGIN -->
| 文件 | wc -l | wc -c | shasum -a 256 |
|---|---|---|---|
| `tools/probe_m12_s1_gate.py` | 1529 | 91849 | `55dedaf3ed6ec60323e4727aa9cb08c63b8f146246506f1d065a5d8bf301ee42` |
| `data/m12_s1_gate_reachability.csv` | 38 | 43261 | `67cc4e3529416369b45fef1d3f12f844e679eeba3846a37a38380274869bfbea` |
| `data/m12_s1_gate_reachability.csv.history.csv` | 55 | 61146 | `d4f24d567a16c420f561995d6faa4b3a06365e3ccaf5761683128889bcee76b0` |
| `data/m12_s1_gate_throughput.csv` | 7 | 2344 | `180d5f5ac18edd9237f914e7e89f8bf6c0837929c217d7dd71ee9bfae7a1c5f3` |
| `data/m12_s1_gate_throughput.csv.history.csv` | 10 | 3451 | `6a67abe4b9b23987207518f918e71d70a0c56c4d8d9c54071200c41e40f99025` |
| `data/m12_s1_gate_integrity.csv` | 16 | 6613 | `c3df2919ec3ba3d01b4672daef629d18e348f379230d461f7afe09e70971e2b4` |
| `data/m12_s1_gate_integrity.csv.history.csv` | 19 | 8093 | `1cb81b11818562fd8aba24cd21bb62c509fc71f1a0d1836f8a3d9d00c3bab1f0` |
| `data/m12_s1_gate_syntax.csv` | 23 | 14047 | `297289daa361d35bf49f0c359ab6ffd551472bde0952088e7662908e2d005797` |
| `data/m12_s1_gate_syntax.csv.history.csv` | 22 | 13779 | `0ea79afca54f71910ca151d667be3415dceb9c04d48a735b54d61abf649e60b6` |
| `data/m12_s1_gate_counts.json` | 1503 | 26023 | `1acbf90c47e548e4c7993b45823ba58e7f0cfdae36fc1d97a2e7adc60f2b5516` |
| `data/m12_s1_gate_assembly.json` | 127 | 3793 | `7b49be3d604d43a170b093a5057cbb44d31de6f2ca003271e8439849caccde5b` |
| `data/m12_s1_gate_assertions.json` | 132 | 6079 | `03128aa784f446c91b312431d5d3e579f06e55faec29703716e7af3806a3228e` |
| `data/m12_s1_region_whitelist.csv` | 1021 | 81536 | `3ae89b1d09c156e761f51235017641214760df594802a4f784e234d6e74720be` |
| `data/m12_s1_raw/A0_body.json` | 0 | 25184 | `0590f9d42620176e9a6adcfcfc99a505c5db69c04e30189f3d85e1b07ca006a3` |
| `data/m12_s1_raw/A10_body.json` | 0 | 25280 | `65cb1ea9a44ab454f54d1e99f201e8416dcc146202d2a7eebeda0ad4093c942c` |
| `data/m12_s1_raw/A11_body.json` | 0 | 25213 | `497fe74402dab8f5cac6be65f156bb165b24cb2445a131e15baaa1e9c23fca01` |
| `data/m12_s1_raw/A11b_body.json` | 0 | 5458 | `9bd731018efe5ae3d08171bc624bbfc18b941934617b6e60e5b2f5583b3cd77d` |
| `data/m12_s1_raw/A11c_body.json` | 0 | 5450 | `3a6c7a64fedfc35fca1a14507e65e0b24684857d29bbc6dd3b162c3cc3906db3` |
| `data/m12_s1_raw/A12_body.json` | 0 | 28125 | `15de5c0e1379e33f1e8a1c3dd3fdf9dbdc89f06e4eaaca660da502b2e613a80a` |
| `data/m12_s1_raw/A12b_body.json` | 0 | 1045 | `44649b23fee2a0bec369401bb0c2b69294f2bd4a7322bf92f269d1ad3d350772` |
| `data/m12_s1_raw/A12c_body.json` | 0 | 1039 | `6b343422f8c430c1d7e44e7436a5ea8ea4d55618b809dbdf8a4b4911aed418b1` |
| `data/m12_s1_raw/A14_body.json` | 0 | 323 | `701e713025864c1fc9d3cc9e76717b46dd756bc75dc472b92c6c0e697863e3ac` |
| `data/m12_s1_raw/A15_body.json` | 0 | 26134 | `d9e2c5eec0fa5e7f8b2a750bc31305fa7d2fb98d3582128fadf3ecc2dc892744` |
| `data/m12_s1_raw/A16_body.json` | 0 | 85 | `3370431813a2de3afb606ef7f234ea7b0345af95df7b9816963e18389258a2b2` |
| `data/m12_s1_raw/A1_body.json` | 0 | 26727 | `924ec195b28764312383cdb615e78e380f3b2ba1b6cdc6531de15a8d476d0108` |
| `data/m12_s1_raw/A2_body.json` | 0 | 25184 | `0590f9d42620176e9a6adcfcfc99a505c5db69c04e30189f3d85e1b07ca006a3` |
| `data/m12_s1_raw/A3_body.json` | 0 | 25629 | `c37ded02cc5db8506e7bf0b6742b93f707dbf6ea7dfa76cff34f1611d0b1e0af` |
| `data/m12_s1_raw/A4_body.json` | 0 | 25605 | `6b8976083e972630cf5d8c63421667815c6317702460bade1d57cd1000b8da2c` |
| `data/m12_s1_raw/A5_body.json` | 0 | 25326 | `83a2e98336285ec17b0746fefd740f04e57e635dea0a7f8f0aefa45e30282919` |
| `data/m12_s1_raw/A6_body.json` | 0 | 25213 | `c1e4c15109f758d32673a2928201e3fa6a6f7d32cde34ec57a49ea507a675128` |
| `data/m12_s1_raw/A7_1000_body.json` | 0 | 28069 | `302c4eff7ab6e482a32f53e6bd0002481b18c9f3810da18bfdcd2f567f02a419` |
| `data/m12_s1_raw/A7_1009_body.json` | 0 | 28308 | `b3bdd7ae2cb7d536d98173435c8d8819d6f9e44d92563c2da91c52d7238ab875` |
| `data/m12_s1_raw/A7_1024_body.json` | 0 | 85 | `3370431813a2de3afb606ef7f234ea7b0345af95df7b9816963e18389258a2b2` |
| `data/m12_s1_raw/A7_1032_body.json` | 0 | 85 | `3370431813a2de3afb606ef7f234ea7b0345af95df7b9816963e18389258a2b2` |
| `data/m12_s1_raw/A7_1040_body.json` | 0 | 85 | `3370431813a2de3afb606ef7f234ea7b0345af95df7b9816963e18389258a2b2` |
| `data/m12_s1_raw/A7_124_body.json` | 0 | 85 | `3370431813a2de3afb606ef7f234ea7b0345af95df7b9816963e18389258a2b2` |
| `data/m12_s1_raw/A7_129_body.json` | 0 | 28229 | `75faf1701edc8db80b4d4655725e1a44a3b8f38a1ce9506268c181685f353ea2` |
| `data/m12_s1_raw/A7_140_body.json` | 0 | 26858 | `309820a101ca3b6ec72ad0af55b4b0514274bc86f8fb62b69d2f9aabad30f162` |
| `data/m12_s1_raw/A7_145_body.json` | 0 | 28274 | `c043c744054492d8d596f33bea881d52e2803857d72bdc69219c68be1dd18134` |
| `data/m12_s1_raw/A7_164_body.json` | 0 | 23533 | `129521285b6d8d4c19098f9294214d7de18045b8ee3df2f1d1bf42e7937017fe` |
| `data/m12_s1_raw/A7_304325711_body.json` | 0 | 85 | `3370431813a2de3afb606ef7f234ea7b0345af95df7b9816963e18389258a2b2` |
| `data/m12_s1_raw/A7_343_body.json` | 0 | 28812 | `386fe02283a7c86c84c0499172bb3ba9b99f8698b26b6a24aa2ad30625db474f` |
| `data/m12_s1_raw/A7_484682512_body.json` | 0 | 28114 | `a29ce5bd4470e5379a40e36b7a8a3080ae07c76f845f9fd6abc3f03aa90a4c62` |
| `data/m12_s1_raw/A7_512_body.json` | 0 | 28481 | `cc6b723c6d08992249c3c9b97170e806273dccc92ddccc0c83782e1bcdf884f2` |
| `data/m12_s1_raw/A7_567_body.json` | 0 | 28847 | `cd5bed9f8e49d503f6a79cc474b58e1c6720be68840cbbe520029302a5e09f50` |
| `data/m12_s1_raw/A7_624_body.json` | 0 | 85 | `3370431813a2de3afb606ef7f234ea7b0345af95df7b9816963e18389258a2b2` |
| `data/m12_s1_raw/A7_73_body.json` | 0 | 29023 | `af338d88a87f02fd0895276953bb75de3cd7b7b0e605419c1ce5f1f2b22c53b0` |
| `data/m12_s1_raw/A7_81_body.json` | 0 | 28281 | `2aa441d84be781a5af3e56bf175d0590f376ee7c4d9a3aa3423bf48a5968642a` |
| `data/m12_s1_raw/A7_8_body.json` | 0 | 28782 | `bcf9b836af03cd22c16f801e8ae2917d18c6a075c6c1f33d75349738f0657f31` |
| `data/m12_s1_raw/A7_960_body.json` | 0 | 28872 | `3f15f5958363a31f664dad93368f1030762dae289097c763cb1c0be69e376883` |
| `data/m12_s1_raw/A7_967_body.json` | 0 | 29071 | `955cd7b57e4c71ec69a33a789759b6950117b35600ad41a5306b07881bf762b0` |
| `data/m12_s1_raw/A7_983_body.json` | 0 | 28797 | `fdf4e7f9b27cca5d5bba8cb29d26282f5c62240f34b889b8dd58c40863c34b09` |
| `data/m12_s1_raw/A7_991_body.json` | 0 | 28902 | `ac6b48a8009879d014f9771f8498b93734c65954e080552b73c935f3aa216fac` |
| `data/m12_s1_raw/A7_997_body.json` | 0 | 28127 | `0584e173c8b5588a5e2f8e5cb1b38b0ffdf2c9ce3556cae0fa1c9c3f15d3f1dd` |
| `data/m12_s1_raw/A8_body.json` | 0 | 25323 | `7d135e4d87abef926d464eaec55a791930d28fb4c96511a6fa11a65c6fec3543` |
| `data/m12_s1_raw/A9_body.json` | 0 | 19013 | `9b8c98d60a2509f45a27d12e4b65da32296fb211b6a7d70d0b1113fb48b13a44` |
| `data/m12_s1_raw/B1_head.bin` | 2575 | 354941 | `3b882148a1d47a3ae773bf27334b50430df3a412297ba0fb1e59fea9486e7571` |
| `data/m12_s1_raw/B2_head.bin` | 6 | 132 | `ac3d2ee82ed9f1e9c9550e5de29c9e0b059296ffd1c5e1b28e2c1ac7d9a7ca66` |
| `data/m12_s1_raw/B3_head.bin` | 6 | 132 | `ac3d2ee82ed9f1e9c9550e5de29c9e0b059296ffd1c5e1b28e2c1ac7d9a7ca66` |
| `data/m12_s1_raw/C_round1.bin` | 10001 | 1443421 | `6dbbf5dcda168f22a5284e9bbf70aac2a1ad5d011e5c5acc358cc3b4f64b0998` |
| `data/m12_s1_raw/C_round2.bin` | 10001 | 1387949 | `2080fe1ea8c75c13c523ffa2c83590f289074f3484f09abd383c698f32a42f5c` |
| `data/m12_s1_raw/C_round3.bin` | 10001 | 1422364 | `a00bfa8e0129624e0fef2d1b9540ecc3c0b6f9a0be0263277f9bed0921eba530` |
| `data/m12_s1_raw/E1_experiment_166218353.csv` | 2575 | 354941 | `3b882148a1d47a3ae773bf27334b50430df3a412297ba0fb1e59fea9486e7571` |
| `data/m12_s1_raw/E2_experiment_166218353_rerun.csv` | 2575 | 354941 | `3b882148a1d47a3ae773bf27334b50430df3a412297ba0fb1e59fea9486e7571` |
| `data/m12_s1_raw/E3_experiment_166218353_injection.csv` | 1 | 313 | `35bad45c2342b7c96d4d5f71c799d4528af41d185a38fd917666c208a2b4216c` |
| `data/m12_s1_raw/F1_n10000_page.csv` | 10001 | 1443421 | `6dbbf5dcda168f22a5284e9bbf70aac2a1ad5d011e5c5acc358cc3b4f64b0998` |
| `data/m12_s1_raw/F1_n200000_page.csv` | 0 | 274 | `6a2981551903d194f7388ea81b7b2031c7b9748a020337a8ac35766916566dc4` |
| `data/m12_s1_raw/F1_n20000_page.csv` | 2 | 495 | `8d2596e6cbb901339d210d44a121fafddbf818150f2dff5396d57c8aefa8b4d1` |
| `data/m12_s1_raw/F1_n2000_page.csv` | 2001 | 290162 | `1f395943c628a76a01da17605bf8a318be323b46cb52bde40668abfa08bcc979` |
| `data/m12_s1_raw/F1_n40000_page.csv` | 2 | 495 | `4cc964a7e31e97750d858561eecedf43d3cb693092f8c42ea96b26963bd9b7f8` |
| `data/m12_s1_raw/F1_n50000_page.csv` | 0 | 272 | `13a806b01385af3707aa8a60f0422a2f2c0c2cae1f1319324fe9471da7731a98` |
| `data/m12_s1_raw/F1_n5000_page.csv` | 2 | 493 | `08415ff0deb485cca0ce33dcbd9b918777c6e86f84c33a195c951a9a4c4e0dc9` |
| `data/m12_s1_raw/F2_lastpage_page.csv` | 2 | 498 | `bf44f1fbac493a8f7a1bb1a8f30befd9202bb62ea08b145432c437f0c22e272f` |
| `data/m12_s1_raw/F2_tail500_page.csv` | 501 | 101629 | `aa1930c810fbe577f53e1cc4e51749248db4ff19700f0c2cab119cd629aab659` |
| `data/m12_s1_raw/F3_mid_page.csv` | 2001 | 405682 | `c28c2da864cad4ab253f03a4fdd54399404103338db5f1fb4327913541e99ec5` |
| `data/m12_s1_raw/F3_midpage_page.csv` | 2 | 498 | `7cfda42d34b26fc46423d9ff1bf6965ac140f336d59f85ca9aff019a7fbf5271` |
| `data/m12_s1_raw/F4_injection_rows.csv` | 11 | 1679 | `59fe590df2243350e0fdd9a7279b6a369e1bab2d8ec7950748aa87c208e583d3` |
| `data/m12_s1_raw/F5_body.json` | 0 | 85 | `3370431813a2de3afb606ef7f234ea7b0345af95df7b9816963e18389258a2b2` |
| `data/m12_s1_raw/_hdr_A0.txt` | 18 | 916 | `8449f8f6fa85275480e0603ba3858933ff346c21942957e30f04d2f6daf3dc30` |
| `data/m12_s1_raw/_hdr_A1.txt` | 18 | 915 | `855c21f194928ea2e4e36d149363116b78b70e9ff28d69348c84990238909590` |
| `data/m12_s1_raw/_hdr_A10.txt` | 18 | 915 | `1bc68be377fc3c2559e986884919f908749217c7b7882f61cb59ff99413dc393` |
| `data/m12_s1_raw/_hdr_A11.txt` | 18 | 916 | `6691b11d056ff6e7bd93a6e630ef9595291dfd679f624bb482157ef902ae5755` |
| `data/m12_s1_raw/_hdr_A11b.txt` | 18 | 915 | `9cd006b2e5812806e3d545169b51d60c356cd446b6e48832217c8f840868a4d4` |
| `data/m12_s1_raw/_hdr_A11c.txt` | 18 | 915 | `9e85b8a8d5c486053621ab762976367e352c83536e04d204db7c7cfdaf58c473` |
| `data/m12_s1_raw/_hdr_A12.txt` | 18 | 915 | `3fa1c93c934e8b4b58281fcaa410d71836a187663eb941ee40b6131c216e32f1` |
| `data/m12_s1_raw/_hdr_A12b.txt` | 18 | 915 | `e3205d0a95f52e6067d2dc90c3c107b48188e60b3687a53b3a3c70636490ba45` |
| `data/m12_s1_raw/_hdr_A12c.txt` | 18 | 915 | `993dc2a6c8ed9c0199c6aae773dfb80770e00d57aef43005df8ab75ddac253c4` |
| `data/m12_s1_raw/_hdr_A14.txt` | 18 | 915 | `58a16ccb564002c680b9e82f79738286d48f8a3796c04db6fa2f049ca0d364fe` |
| `data/m12_s1_raw/_hdr_A15.txt` | 18 | 915 | `a43d5310657c076185079404930d3051f63f409cf509ac3a1275dbade52f361d` |
| `data/m12_s1_raw/_hdr_A16.txt` | 18 | 915 | `e71d7b1340cac8faa90b365e2aa92568757315dd1555f0b4e5a86d8a6fe78ea9` |
| `data/m12_s1_raw/_hdr_A2.txt` | 18 | 916 | `8f7300afd81e778abd26347a5ffd32bc5332acd11886da6f62016ece1bdf3b2c` |
| `data/m12_s1_raw/_hdr_A3.txt` | 18 | 915 | `70e6335b4c15ce1186c8666fdc87f2b8f86cf5c258c9cb494590c6070727f9d4` |
| `data/m12_s1_raw/_hdr_A4.txt` | 18 | 915 | `3babf41828dd24ce71cd8087d43557b5e55aeb858f2596f85b932da19fe7e4de` |
| `data/m12_s1_raw/_hdr_A5.txt` | 18 | 915 | `105c0298b15571a8f439a965cdcc88e1e14cdc22f3741edc3fdcb99aff9dee61` |
| `data/m12_s1_raw/_hdr_A6.txt` | 18 | 916 | `7f8893f4bfa2fc4c46c2025d00dccfe6ce048c3dd14195d574facb1d3c534f2e` |
| `data/m12_s1_raw/_hdr_A7_1000.txt` | 18 | 915 | `5622285518ee4717098aff66554cb09b329b20c1319e11d99b55371907ddc627` |
| `data/m12_s1_raw/_hdr_A7_1009.txt` | 18 | 915 | `af497dbbf22b4575766996895fa5363c3a06415da98146f29a572ef2e9b04449` |
| `data/m12_s1_raw/_hdr_A7_1024.txt` | 18 | 915 | `114556f588f00a95f3160bce291c7d9a4c48ef5b63b033d6595d1f974d4bbff5` |
| `data/m12_s1_raw/_hdr_A7_1032.txt` | 18 | 915 | `45ccbf100afa5878dfe13c930a5cf9e4159f0e325aa0089ef540c6ccb1771558` |
| `data/m12_s1_raw/_hdr_A7_1040.txt` | 18 | 915 | `42935fbf15d30037a3f9703fe9b1602520025ef83d294d373c37a444f3a232a7` |
| `data/m12_s1_raw/_hdr_A7_124.txt` | 18 | 915 | `90ea81b64e964229f44cbdc8ad618e0369576a876c6242e4d507546cd501095c` |
| `data/m12_s1_raw/_hdr_A7_129.txt` | 18 | 915 | `89b0ef4eef8adac698a4087c763e587d3e64f68e54d019f7a9f43913710ba44b` |
| `data/m12_s1_raw/_hdr_A7_140.txt` | 18 | 915 | `c96514a94675e6e4dd62080b7025398b446a59da75ae24a19308a39e51f40a3b` |
| `data/m12_s1_raw/_hdr_A7_145.txt` | 18 | 915 | `bc0c5e256eb2be370071d767920872644efc2af828535cdf336ddb95462a551f` |
| `data/m12_s1_raw/_hdr_A7_164.txt` | 18 | 915 | `0ed0b5a4562b07156a49ac30a55d71ff056df739d050227a949952f255c8bd8b` |
| `data/m12_s1_raw/_hdr_A7_304325711.txt` | 18 | 915 | `d463a6ce36091c8d79d4bbb3db32d40e96cbfaa8aabde45c13bf7a7b7db598bb` |
| `data/m12_s1_raw/_hdr_A7_343.txt` | 18 | 915 | `8cf666f4d1d4bc1fb1e68345b132e59499e989bf8b0f01fa341c692717629d05` |
| `data/m12_s1_raw/_hdr_A7_484682512.txt` | 18 | 915 | `ed7ab4f2575d423e0651c46bfde97decfae8ff37124c8d43abc52638b1eb381c` |
| `data/m12_s1_raw/_hdr_A7_512.txt` | 18 | 915 | `b6884ecc1a05a7ca698c276f50c93b7488d305aa0d7ac4a8a802c16da9427a42` |
| `data/m12_s1_raw/_hdr_A7_567.txt` | 18 | 915 | `f1b181cb783ef2d875f75ef47ec918ab0ca88256afee2bed67913b27ad1981cb` |
| `data/m12_s1_raw/_hdr_A7_624.txt` | 18 | 915 | `0a108a521890f371c7c59574bf9eadc42b4e2afb6386c2e74e605a28b1777559` |
| `data/m12_s1_raw/_hdr_A7_73.txt` | 18 | 915 | `b690ed3b80c6e4dd62cf756fde6ca5b5d3a17d07f5fa75ae1623bf0ed9027fb5` |
| `data/m12_s1_raw/_hdr_A7_8.txt` | 18 | 915 | `dd23748a7cba6e34f0dbb090c50fcbc962f78459d451603143c78d9c52798b11` |
| `data/m12_s1_raw/_hdr_A7_81.txt` | 18 | 915 | `20458ba5b5a3949475a3f2d6ec4bfdd6c641ac2c8fd6d605d1b404375135404b` |
| `data/m12_s1_raw/_hdr_A7_960.txt` | 18 | 915 | `9581e6af5bcc25164d1b98b15b09afdaf35a17d8fb8d509e5904f9de9d6dcd3e` |
| `data/m12_s1_raw/_hdr_A7_967.txt` | 18 | 915 | `02c0eb7e4f899f8e1803a847ab6e6026d93a52e2a9f0340e2790895019e8cf25` |
| `data/m12_s1_raw/_hdr_A7_983.txt` | 18 | 915 | `94411b5bde2bdd8ee37ad46b83db608bbebc50c2dea1232ebd0eb6eb6a74d3bf` |
| `data/m12_s1_raw/_hdr_A7_991.txt` | 18 | 915 | `eabc70c996e15e030a45ba57812a919521da4f04aebb5b7b348e01504719527a` |
| `data/m12_s1_raw/_hdr_A7_997.txt` | 18 | 915 | `4d4061bd77cd332e96a194267cc918322cf5f402e626adb42d2c853d84752c2f` |
| `data/m12_s1_raw/_hdr_A8.txt` | 18 | 915 | `dab3e28397c272d3f0a65581e88492b0e7bd77e7035edd4e5b6f8d8fbaa91892` |
| `data/m12_s1_raw/_hdr_A9.txt` | 18 | 915 | `8b68acef2d037dea69674409f19d0309f8e0933651b99a39a7fa90a870f0a963` |
| `data/m12_s1_raw/_hdr_B1.txt` | 18 | 907 | `3540c6e0964d7fd96b8e459582d59ca82a9df218d9f5b708db4a92feffc14920` |
| `data/m12_s1_raw/_hdr_B2.txt` | 8 | 529 | `7bd49568e47d7a9de79838a23dab8829b68dea00d0e98614d4e45c1b1b19fc7a` |
| `data/m12_s1_raw/_hdr_B3.txt` | 0 | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `data/m12_s1_raw/_hdr_E1_experiment_166218353_csv.txt` | 18 | 907 | `432740e894253c207ff236156a9b73a25bc23f0f8904ece8213b4f91b103c6b6` |
| `data/m12_s1_raw/_hdr_E2_experiment_166218353_rerun_csv.txt` | 18 | 907 | `14af4b8a6943da1c11e1036c1a28d8dd0684096ec2df3623504a87af3e860e6c` |
| `data/m12_s1_raw/_hdr_E3_experiment_166218353_injection_csv.txt` | 18 | 907 | `d0aad601cc7b6c9da29793bbbb41bcf6276024c74f6ab16261b240d3acb0a009` |
| `data/m12_s1_raw/_hdr_F1_n10000_page_csv.txt` | 18 | 907 | `8615b6598c2cb04ab8037526f84ec701ef7cb1d116d308f4664a691e79a6ba04` |
| `data/m12_s1_raw/_hdr_F1_n200000_page_csv.txt` | 18 | 907 | `4ea12eb4ddcd5881b535b29cf8b7b45d44976bc734258d781a217bc8d25de697` |
| `data/m12_s1_raw/_hdr_F1_n20000_page_csv.txt` | 18 | 907 | `603b95938cab3cf81168f135a79fd433a70a27dd7a70e311cffdc5d784b6c1ff` |
| `data/m12_s1_raw/_hdr_F1_n2000_page_csv.txt` | 18 | 907 | `1d43b28f1cbb9fb5761c3f723a163c607f330da1e74ecad56771b9123870e804` |
| `data/m12_s1_raw/_hdr_F1_n40000_page_csv.txt` | 18 | 907 | `4c2d7c348a81451eeb2575ab0d953ad3a299ae29678658fa2a886f5d84ff24fe` |
| `data/m12_s1_raw/_hdr_F1_n50000_page_csv.txt` | 18 | 907 | `912a995d6393f7855c301b5650bd6547f9cf85c21141ccb35c5a149c59248d65` |
| `data/m12_s1_raw/_hdr_F1_n5000_page_csv.txt` | 18 | 907 | `56e6c8db07a8398ebadc907dcdff6685dc47d8bd89c659c62f15940c7d4e44fb` |
| `data/m12_s1_raw/_hdr_F2_lastpage_page_csv.txt` | 18 | 907 | `2d5060e0bd01751a64c251bac95b3f0f1c0bb6a9d53b3ee9283794254c7977e1` |
| `data/m12_s1_raw/_hdr_F2_tail500_page_csv.txt` | 18 | 907 | `0d4572607a5795122d7c64fabd990eea6f2a869510f2688c24236f5725fe4687` |
| `data/m12_s1_raw/_hdr_F3_mid_page_csv.txt` | 18 | 907 | `1a92f786f1eb9ddddf6f242284f069ce3f876e219398c2226e94fe911f19b169` |
| `data/m12_s1_raw/_hdr_F3_midpage_page_csv.txt` | 18 | 907 | `680b9af8d1ba4670ca14366a7b4000d5b6fdbe24d924cf170506bc3fa6ce0d7b` |
| `data/m12_s1_raw/_hdr_F4_injection_rows_csv.txt` | 18 | 907 | `271c058eb6394bcb58c71ea7583de12333a2f6cab23e1f756b1cbce086f1e0da` |
| `data/m12_s1_raw/_hdr_F5.txt` | 18 | 915 | `f96cbd365d33fd15c2e4257967b5c5c7c4541d408f1a3da25d0fd43a7b4e671b` |
| `data/m12_s1_raw/_hdr_X_V1.txt` | 18 | 916 | `acdf4c44863ff927aa0f040d21ba844be0c5b176fe0472d2d415a474aa597e36` |
| `data/m12_s1_raw/_hdr_X_V2.txt` | 18 | 916 | `0435580180a5eeb4e591837d0eed12ff1c6fc33e4b84200deeb81255f437462b` |
| `data/m12_s1_raw/_hdr_X_V3.txt` | 18 | 916 | `ffd69b5d2b2328accc3da43f7907e05544c7c9c58ef11824e80788c068a2630e` |
| `data/m12_s1_raw/_hdr_X_V4.txt` | 18 | 916 | `a9d9dbc259836ce33c652dd390d94bcf8ffc74564881aa40121de59586e07b17` |
| `data/m12_s1_raw/_hdr_X_V5.txt` | 18 | 916 | `3613b747f57a52bac34dea771f2929c1f180e4f91901f94b1a3cebf50344c002` |
| `data/m12_s1_raw/_wl_regen.csv` | 1021 | 81536 | `3ae89b1d09c156e761f51235017641214760df594802a4f784e234d6e74720be` |
| `docs/m12_s1_gate_probe.md`（**本文件自身**） | 见 `data/m12_s1_gate_manifest.txt` | 同左 | 自引用**不可内嵌** ⇒ 权威回读表见 `data/m12_s1_gate_manifest.txt` |

**合计（不含本文件自身）**：146 个文件，58967 行，8917498 字节。
<!-- SHASUM_TABLE_END -->

---

## 10 本轮实际执行过的命令（可审计；**未使用 MPS**）

```bash
# 预检：确认 M11 的 MPS 作业在跑（**只读，不 kill、不起第二个**）
pgrep -f run_m11_p5_arms

# 探针各阶段（全部 python3 = 系统解释器 /usr/bin/python3 3.9.6；纯标准库）
cd /Users/weidong/ai/small_world && export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 -u neural_exploration/tools/probe_m12_s1_gate.py --phase A   # T1 计数 + 交叉校验
python3 -u neural_exploration/tools/probe_m12_s1_gate.py --phase B   # T2 对象级
python3 -u neural_exploration/tools/probe_m12_s1_gate.py --phase C   # T3 吞吐（有效页 3 轮）
python3 -u neural_exploration/tools/probe_m12_s1_gate.py --phase D   # T4 ETA + 四态
python3 -u neural_exploration/tools/probe_m12_s1_gate.py --phase E   # T5/T6 真实取数 + 解析 + 装配
python3 -u neural_exploration/tools/probe_m12_s1_gate.py --phase F   # 分页可行性 + 页上限 + 源维语义
python3 -u neural_exploration/tools/probe_m12_s1_gate.py --phase X   # 跨节点更正行
python3 -u neural_exploration/tools/probe_m12_s1_gate.py --phase Z   # 18 条机验断言
python3 -u neural_exploration/tools/probe_m12_s1_gate.py --all       # 一键重跑（需网络）

# 本地核对（零网络）
python3 -c "…读 data/m12_raw/s2/model_Structure_all.json 数祖先链、leaf 数、depth 分布…"
```

**未做**：未 `kill` 任何进程；未起第二个 MPS 作业；未改任何 `m11_*` 文件、未改
`data/m11_p5_plan_l3.json`、`data/m11_p5_l3_runs/`、`tools/run_m11_p5_arms.py`、`tools/validate_p11_resting.py`、
`tools/probe_m12_sources.py`（B1 件）及其产物；**未用 `2>/dev/null`**；**未 import torch**（断言 Z15）。
