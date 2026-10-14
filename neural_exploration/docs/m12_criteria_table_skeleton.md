# M12 判据表骨架（§5.1/§5.3/§5.6 的落地）+ 三条闸门（M12 执行节点 A-5 交付物 ①）

> **工作包**：清单 §0.2.2 **A-5** 行（`docs/生物仿真M12实施清单.md` **L90**）原文：
> 「**§5 判据表骨架 + §6 负对照脚手架** —— 判据带的 provenance 化（每条带给出处）与负对照设计是**文档 + 脚手架**工作，
> 可在任何门之前完成；`final=0` 行先落盘，待门后用 `measured` 行追加（**append-only**）。」
>
> **配套产物**：`data/m12_targets.csv`（判据带骨架，**124 条记录**）、`tools/check_m12_targets_gates.py`（三条闸门检查器，纯 stdlib）、
> 以及交付物 ② `docs/m12_negative_control_scaffold.md`。
>
> **本节点硬约束遵守声明（逐条）**：① **全程未运行 `--device mps`**、未起任何 MPS 作业（开工前 `pgrep -f run_m11_p5_arms` ⇒ 空，
> 见 §11 命令清单）；**未 kill 任何进程**；② **未修改任何既有文件**（只新增本节点的 3 个文件：本文件、`data/m12_targets.csv`、`tools/check_m12_targets_gates.py`，
> 见 §11 的 `git status` 实测）；③ 本文档所有断言都有实测输出支撑，无法确认的一律显式写「未确认」并指出卡在哪一行/哪个字段（§8）；
> ④ **未使用 `2>/dev/null` 掩盖任何命令失败**（命令清单里所有命令均保留 stdout/stderr 与退出码）；
> ⑤ **未降规模、未用云 GPU、未取新数据**（本轮零网络：所有引用均为磁盘上已存在件 + 文档行号）。

---

## §0 一句话结论

**本骨架可以定稿「口径常数 + 门标准 + 已实测锚 + 判据作用域」这四类共 98 行（`final=1`）；
但判据带本体（区域率带、自维持比上限、ρ 下限、dwell 带、目标带）有 26 行只能落 `final=0`** ——
其中**皮层侧（G1-M/P5a）整块卡在 M11 G1 未解封**（清单 L106–L108 / GAP-01），
而另有一批常数是**清单自己没注册**（R117 型缺口，见 §5.2）⇒ 即便 M11 解封，P5a/P5b/P6a/P9/P10b 的判据仍不可定稿。
**详见 §9（卡点清单）。**

---

## §1 交付物与身份

| 项 | 值 |
|---|---|
| 工作包 | A-5（清单 L90） |
| 产物 ① | `data/m12_targets.csv` —— **124 条记录**；133 物理行；80,871 B；`final=1`: **98**；`final=0`: **26** |
| 产物 ② | `docs/m12_criteria_table_skeleton.md`（本文件） |
| 产物 ③ | `tools/check_m12_targets_gates.py` —— 314 行；13,547 B（三条闸门 + 负向测试模式） |
| 产物 ④ | `docs/m12_negative_control_scaffold.md`（§6 负对照脚手架） |
| 本轮特性 | **零网络、零 GPU、零 MPS、零既有件修改**（§11） |

**表格形态的取样对象**：`data/m11_p5_targets.csv`（M11 已被验收的判据表）。
本节点实测该表：**104 条记录**（119 物理行，含 3 行 `#` 注释与**字段内换行** ⇒ 物理行数 ≠ 记录数）、**104 个唯一 key**、
`final=1`: 88 / `final=0`: 16、`evidence_level` 实测域 = {`derived`:57, `literature`:18, `measured`:16, `unavailable`:12, `literature_unverified`:1}。
> ⚠️ 派发件写「105 行」，本节点实测 **104 条记录**（与 `m11_acceptance_master.md` §3dh L3340 的「104 = 103 + 1」一致；105 应为 R117 §③ 那次**把表头也追加进去**时的错误计数）。
> 这是一处**口径差异**，本节点按实测值记录，供主 agent 核对（不影响任何结论）。

---

## §2 列结构与两表对照

### 2.1 逐列对照（`data/m12_targets.csv` 实测列名）

| # | M11 `m11_p5_targets.csv` | M12 `m12_targets.csv` | 关系 |
|---|---|---|---|
| 1 | `key` | `key` | 同名同义 |
| 2 | `value` | `value` | 同名同义 |
| 3 | `value_kind` | `value_kind` | 同名同义（取值域沿 M11 实测：`point`/`list`/`enum`/`unavailable`/`bool`/`range_lo`/`range_hi`） |
| 4 | `unit` | `unit` | 同名同义（空 unit 是允许值：M11 有 69 行为空） |
| 5 | `evidence_level` | `evidence_level` | 同名同义（域沿 M11 实测：{measured, derived, literature, literature_unverified, unavailable}） |
| 6 | `provenance` | `provenance` | 同名同义；**本表额外规定**：每行 provenance 以标签开头（§4.1） |
| 7 | `note` | `note` | 同名同义（口径/依据写在这里） |
| 8 | `wrong_usage_consequence` | `wrong_usage_consequence` | 同名同义（本表**不允许** `N/A`：124/124 行均有实质内容） |
| 9 | `final` | `final` | 同名同义（域 {0,1}） |
| 10 | —（M11 无此列） | **`scope`** | **本表新增**：承载「判据作用域 + 身份四元组」（§3）——正是 M11 缺的那一维 |

**实测核对**：`len(m12 列)=10`；`len(m11 列)=9`；`m11_cols + 1 == 10` ✔；`m12 列[:9] == m11 列` **逐列相同** ✔（`tools/check_m12_targets_gates.py` 的 G0 闸门每次运行都断言这一条）。

### 2.2 ★ 与清单 §5.1 的差异（**未确认项 U1，须主 agent 裁决**）

清单 **§5.1 L887** 为同名文件给定的列是 **11 列**：
`key,value,value_kind,unit,evidence_level,provenance,provenance_verified_by,note,wrong_usage_consequence,calibration_note,final`。

**本骨架按派发要求用「M11 的 9 列 + `scope`」= 10 列** ⇒ 与清单相比**缺 `provenance_verified_by` 与 `calibration_note` 两列**（G4 闸门每次运行都会打印该差集）。
本节点**不自行改判**，处置建议（二选一，均已在表内保持可迁移）：

| 方案 | 内容 | 迁移代价 |
|---|---|---|
| **A（骨架扩为 12 列）** | 追加 `provenance_verified_by`（124 行全部可从 provenance 标签机械映射，见 §4.1）与 `calibration_note`（相关信息已在 `note` 内以「【口径】…」段存在） | 低：一次机械映射 |
| **B（维持 10 列）** | 在 `m12_criteria.csv`（§5.1 L844–884 的另一张表，列更全）承载这两列，`m12_targets.csv` 保持"带与常数"的窄表 | 零 |

**本骨架已做的兼容动作**：① provenance 首标签即 `provenance_verified_by` 的等价物（映射表见 §4.1）；
② 口径文字一律写在 `note` 的「【口径】」段（等价于 `calibration_note`）；
③ 新增行 `protocol_provenance_type_domain` 把清单 L872 的 `provenance_type` 取值域与本表的标签词表**显式对齐**。

### 2.3 字段格式约定（为让 append-only 核对可机械执行）

**每行字段内不使用半角逗号 / 半角双引号 / 换行** ⇒ **物理行数 = 记录数 + 9**（本表实测：**133 物理行** = 8 行 `#` 注释 + 1 行表头 + **124 条记录**，文件以换行结尾）。
> 依据：M11 §3dh 纪律第六十一条 —— 追加时"不得连表头一起追加"，且事后必须做**记录数 + 唯一性 + 与上版差集 + 前缀逐字节**四项核对；
> M11 的 119 物理行 / 104 记录的形态使"行数"与"记录数"分离，本表刻意避免该二义性。

---

## §3 `scope` 列语义（R124 教训的落地：**判据作用域**）

`scope` 是 **3 段**、以半角分号分隔、每段必须存在且非空：

```
applies=<门/判据 ID>[|<门/判据 ID>…];arm=<该行适用的臂域>;quad=<config_id>|<role>|<estimator>|<aggregation>
```

| 段 | 语义 | 例 |
|---|---|---|
| `applies=` | 该行**适用的判据**（清单 §5.5 的 P 编号 + 门名 G2-M/G0-M/G1-M/C-M/F/EXT） | `applies=G1-M/P5a\|G1-M/P5c` |
| `arm=` | **在哪条臂上适用** —— 这正是 R124/R118 的教训所在 | `arm=baseline_only`（E/I 类只在基准臂上判） |
| `quad=` | **身份四元组** `(config_id, role, estimator, aggregation)`（M11 §3dr L3688–3690 纪律第六十五条）；**不适用写 `N/A`**（四段都要写，如 `quad=N/A\|N/A\|N/A\|N/A`） | `quad=per_config\|baseline+zero_drive\|pooled_ratio\|n_num=n_den_asserted` |

**为什么必须新增这一列（三条 M11 实测学费）**：

1. **R124（§3do L3554–3577）**：评估器 (e) 的判据口径缺陷被确证 ⇒ `(c)↔(e)` 对立被撤回。
   本表由此把「(e) 类只在基准臂上判」写成 `rule_e_baseline_arm_only`（`arm=baseline_only`），
   并把 M12 的同类量（P5b 的自持比/静默/弹性）逐个绑定到 `arm=zero_drive|baseline` 域。
2. **R127（§3dr）**：同一个量（`a_active_frac`）在三处用了三套口径（单实现 / 对实现取均值 / 对实现取中位数）⇒ 身份**四元组**。
3. **R131（§3dv L3804–3810）**：比值 `R` 的分子分母各自汇聚时必须打印 `n_num/n_den`，不等则**拒绝出值** ⇒ 四元组之外还需要 `(n_num, n_den)`，
   本表用 `rule_ratio_requires_equal_n` + `target_p5b_ratio_window` 两行承载。

---

## §4 provenance 标签与 `final` 规则

### 4.1 provenance 标签词表（每行 provenance 的首段标签）

| 标签 | 含义 | 映射到清单 `provenance_verified_by`（L874） |
|---|---|---|
| `[self]` | **本节点当场实测/复算**（命令 + 输出见 §11 与各行 note） | `self` |
| `[self_readback]` | 本节点**读回某已落盘件的字段**（证明"该值在盘上"；**不证明**上游测量的正确性） | `self`（但证据强度弱于 `[self]`，本表在 note 中注明） |
| `[other_node]` | 另一执行节点的落盘结论（读其文档，本节点未复算） | `other_node` |
| `[checklist]` | 清单行号（设计输入/协议条款） | `assumption`（清单条款属假设性输入，见 `protocol_provenance_type_domain`） |
| `[literature]` | 文献（本表当前 **0 行**：M12 的文献锚由 B1/G2 逐条 provenance 化，本表只引用其结论行） | `literature` |
| `[inherited_unverified]` | 继承未复算（§0.8 #12 要求显式标注） | `measured_inherited` |

### 4.2 `final` 规则（**明文条款**）

- `final=0` = 该带**未定稿 / 口径未确认** ⇒ 依赖它的判据记 **`unavailable`**（沿 M11 经验：`final=0` 行导致对应对不可判）。
- `final=1` = **可用于判定**。
- 🔴 **`final=1` 的行不得在跑臂后修改**。依据：M11 **R93**（`m11_acceptance_master.md` §3cj L2417）与 **R117**（§3dh L3331「追加且仅追加一行」）+
  清单 §5.6 #5（L970–972「任何修订 = 新增行 + 旧行 `final=0` + 更正指针」）+ §8.3 第 5 条（L1081：判据带定稿时间必须早于对应协议运行时间）。
  本表用 `rule_final1_frozen_after_run` 行把该条款**登记在表内**（而不是只写在文档里）。

---

## §5 口径常数登记册（R117 的直接产物）

### 5.1 机制（写进表里的判据）

> **判据**：凡**决定某个判据取值**的协议常数（驱动率 / 步长 / 静默窗 / T / settle / bin / 平滑帧数 / 分辨率 …）**必须在本表有行**，
> 且其 `wrong_usage_consequence` 必须写明"**若被改动会怎样误导**"。
> 依据：M11 **R117**（§3dh L3329–3336）—— `drive_rate_hz = 8.0` 这个决定 (a) 判定的常数**在 103 行里从未登记**，
> 补登后才使 (a) 的判定可从注册常数复现。

本表据此登记了 **11 行**「协议/口径常数」（`protocol_*`），其中 **9 行是缺口行**（`value=unavailable`，`final=0`）：

| key | 清单出处 | 状态 | 它会决定什么 |
|---|---|---|---|
| `protocol_T_main_s` | L408/L712（仅降档式左值） | **final=0**（无独立注册行） | P5a/P5c 的率分辨率（1/T）与稳态性 |
| `protocol_settle_ms` | §5.3 L910 要求填，**无值** | **final=0** | t=0 瞬态是否入统计 |
| `protocol_bin_wide_ms` | §5.3 L908 / §0.8 #1 L268 要求填，**无值** | **final=0** | P10a/P10c 的区域率与动力学 |
| `protocol_smooth_window_frames` | §5.3 L908 要求填，**无值** | **final=0** | 率的时间分辨率与量纲 |
| `protocol_drive_rate_hz` | 清单**无注册行**（对照 M11 的 R117 补登行） | **final=0** | P5b 驱动弹性 / P5a 率带 |
| `protocol_drive_dose_grid` | 清单**无注册行**（M11 有 `0\|0.5\|1\|2\|4\|8`） | **final=0** | 弹性的剂量点与"单调"判定层 |
| `protocol_T_dwell_ratio_min` | L712（`T ≥ 3× 目标 dwell`） | **final=1**（唯一有值的 T 类常数） | 短窗不得判长时标 |
| `protocol_ccfv3_voxel_resolution_options_um` | L477（25/100 µm） | final=1（候选档；文件名未注册） | 空间尺度口径（GAP-14 同型陷阱） |
| `protocol_p9_consolidation_drop_min` | L951（「≥ 带」无值） | **final=0** | P9 的 F5 第三层（动态范围） |
| `protocol_pharma_dose_caliber` | §5.3 L912 要求填，**无值** | **final=0** | P11 是否与文献同口径 |
| `protocol_provenance_type_domain` | L872 | final=1 | 证据等级词表 |

### 5.2 M12 实测的「口径常数缺口清单」（= R117 的复现）

**清单 §2–§5 全文里找不到数值、但决定某判据取值**的常数（本节点实测 grep 逐条核对）：

| # | 常数 | 卡在哪里（精确到行/字段） | 影响的判据 |
|---|---|---|---|
| 1 | 协议分析窗 **T** | 清单**无**"协议 T = 值"的注册行；只有 L712 降档式「T=30 s → 10 s」的左值与 L408 的命名例「30 s 窗」 | G1-M/P5a、P5c |
| 2 | **settle** 时长 | §5.3 L910 列为必填字段，§2–§5 **无任何数值** | G0-M/P3、G1-M/P5a |
| 3 | **bin 宽** | §5.3 L908 + §0.8 #1 L268 要求，**无值** | P10a、P10c |
| 4 | **平滑窗（帧数）** | §5.3 L908 要求，**无值**（M11 表亦无 ⇒ 跨里程碑缺口） | P10a、P10c |
| 5 | **驱动率**（逐神经元/逐区域） | 清单**无注册行**（M11 正是被 R117 补登的那个常数） | P5b 弹性、P5a 率带 |
| 6 | **驱动剂量网格** | 清单**无注册行** | P5b（含"单调"的判定层） |
| 7 | 自持比**上限** | L945 只写「≤ 上限」 | P5b |
| 8 | 静默比例**带** | L945 只写「带运行前定稿」 | P5b |
| 9 | 驱动弹性**下限** | L945 只写「≥ 下限」 | P5b |
| 10 | 比值**可判下界**（M11 的 `pooled_zero_drive ≥ 300` 那一维） | 清单**无对应注册** | P5b（会重演 M11 R115 §②的 unavailable 陷阱） |
| 11 | P6a 的 **ρ 下限** | L947 只写「下限由边界可达性自检 + 文献定」 | C-M/P6a |
| 12 | **G0-M (c) 的"带"** | L667 只写「Δ < 带」 | G0-M/P3 |
| 13 | P10b 的 **`acc_min`** | L953 + §1.2 F3（L370 明文：无 provenance ⇒ invalid），**无值** | P10b |
| 14 | **P9 巩固下降的"带"** | L951 只写「≥ 带」 | P9 |
| 15 | **药理口径三要素**（剂量/途径/给药时窗） | §5.3 L912 要求填，**无值** | P11 |
| 16 | P11 锚级 `n_min` | L218 写作「**如** `n_min=50`」= 示例而非注册 | P11 |
| 17 | 每区域点数档（清单**自相矛盾**） | L655 `{500,1000,2000,5000}` vs L689 `{500,1000,2000,5000,10000}` | G0-M/P3 定档、P12 |

> **本骨架的处置**：1–16 全部落成 `final=0` 行（值列写 `unavailable`，note 写明"清单未注册"）；
> 17 登记为**清单内部冲突**（`gate_points_per_region_ladder`，note 同时列两处原文）⇒ `final=0`，待主 agent 或 B2 裁决。

---

## §6 三条闸门（条款 + 检查器 + 实测输出）

检查器：`tools/check_m12_targets_gates.py`（纯 stdlib；**只读**：变异只在内存中做，从不以写模式打开任何文件）。
命令（仓库根）：`python3 neural_exploration/tools/check_m12_targets_gates.py`

### ⑥.① 闸门 ①：附录缺失检查（G1）

**条款**：任何被 `note`/`provenance`/`wrong_usage_consequence` 引用为"判据/常数"的量
（形式一：`key=<K>` 显式引用；形式二：`protocol_*`/`gate_*`/`rule_*`/`target_*`/`ref_*`/`nc_*` 形态的 token，**排除 `文件::key` 形态的跨表引用**）
若不在本表 ⇒ **报错**（退出码 1）。

**条款②（防伪装）**：本闸门必须报告"检查了多少次引用"，否则通过可能只是因为"没扫到东西"。

### ⑥.② 闸门 ②：四元组完整性（G2）

**条款**：每行的 `scope` 必须含 `applies=`/`arm=`/`quad=` 三段，且 `quad` 恰为 4 段、每段非空；
不适用必须**明文写 `N/A`**（不允许空段、不允许省略）。

### ⑥.③ 闸门 ③：`final=0` 计数（G3）

**条款**：输出「当前有多少行 `final=0`」，并逐行给出**它会让哪个判据 `unavailable`**（由 `scope.applies` 直接得出）。
依据（M11 经验）：`final=0` 的行会导致对应对判据 `unavailable`。

### ⑥.④ 闸门 ④（本节点附加）：清单列覆盖（G4）

**条款**：打印清单 §5.1 L887 指定但本骨架缺失的列（当前 = `provenance_verified_by`、`calibration_note`，见 §2.2 U1）。

### 6.5 正向实测（全部通过）

```
=== G0 结构闸门 === PASS
    final0 = 26
    final1 = 98
    records = 124
    self_diff_empty = True
    unique_keys = 124
=== G1 附录缺失检查 === PASS
    bare_keylike_distinct = 30
    bare_keylike_refs_checked = 46
    explicit_refs_checked = 0
    registered_keys = 124
=== G2 四元组完整性 === PASS
    rows_with_quad = 124
=== G3 final=0 计数 === INFO
    count = 26          ← 明细见 §9（逐行 + 它会让哪个判据 unavailable）
    total = 124
=== G4 清单列覆盖（清单 §5.1 L887） === INFO
    missing = ['provenance_verified_by', 'calibration_note']
[TABLE] .../data/m12_targets.csv  records=124  final=1:98  final=0:26
[RESULT] ALL GATES PASS
```
**退出码 = 0**（实测）。

### 6.6 负向测试（三个闸门分别用**故意构造的坏输入**实测必须报错）

| 变异 | 期望 | **实测输出（摘录）** | 退出码 |
|---|---|---|---|
| `--negative-test drop-provenance`（把首行 `provenance` 置空） | G0 报错 | `=== G0 结构闸门 === FAIL` / `ERROR: key=protocol_T_main_s 的 provenance 列为空` | **1** |
| `--negative-test break-quad`（把首行 quad 改成 2 段） | G2 报错 | `=== G2 四元组完整性 === FAIL` / `ERROR: key=protocol_T_main_s：quad 应为 (config_id\|role\|estimator\|aggregation) 四段，实测 2 段：'only_two\|parts'` | **1** |
| `--negative-test dangling-ref`（在首行 note 里引用未注册常数） | G1 报错 | `=== G1 附录缺失检查 === FAIL` / `ERROR: key=protocol_T_main_s 显式引用 key=protocol_undeclared_constant_hz ⇒ 本表无此 key（附录缺失）` / `ERROR: key=protocol_T_main_s 的文本里出现判据式 token protocol_undeclared_constant_hz ⇒ 本表无此 key（附录缺失）` | **1** |
| 正向（未变异） | 全过 | `[RESULT] ALL GATES PASS` | **0** |

> 注：闸门 ③（计数）本身是 INFO 型，不产生 pass/fail —— 它的"负向测试"形态是**缺口清单本身**（§9：26 行 final=0），
> 由闸门 ①/② 保证这些行不是"忘记填"，而是"确实未注册"。

---

## §7 卡点：`final=0` 的 26 行分别卡在什么门后（**对 ⚠️ 提示的直接回答**）

> **⚠️ 直答**：**A-5 的骨架无法在缺少 M11 G1 结论的情况下填出"可判定的判据带"。**
> 具体地：(i) **G1-M/P5a 整块**（皮层与非皮层的区域率带）无同口径生理靶 ⇒ 只能占位；
> (ii) 另有一批**清单自己未注册**的口径常数（§5.2 的 1–16）⇒ **即使 M11 解封**，P5a/P5b/P6a/P9/P10b 仍不可定稿；
> (iii) 已取回数据的目标带（P8 心理测量曲线）**未经本节点独立复算** ⇒ 按 §0.8 #12 不得作 `final=1` 的带。
> 因此本骨架**没有**用 `literature`/`derived` 值去"凑"一个看似完整的带 —— 结构完整但过期的内容会被执行，残缺的才会被发现（M11 教训）。

### 7.1 按阻塞源分组

| 组 | 阻塞条件 | 行（key） | 解封动作 |
|---|---|---|---|
| **A. M11 G1 未解封**（清单 §11 GAP-01 / §0.2.2 处置 1 L106–L108） | M11 P5 门链终态（当前 (b) 因钙域静默口径而 `unavailable`） | `target_p5a_cortex_rate_band_hz`、`target_p5a_noncortex_rate_band_hz` | M11 解封后**新增 `measured` 行**（并把本行 `final` 保持 0 + 加更正指针），**皮层协议在解封前不得定稿**（L108 明文） |
| **B. 清单未注册的口径常数**（R117 型） | 需主 agent / B2 / B3 用**一行显式注册** | `protocol_T_main_s`、`protocol_settle_ms`、`protocol_bin_wide_ms`、`protocol_smooth_window_frames`、`protocol_drive_rate_hz`、`protocol_drive_dose_grid`、`protocol_pharma_dose_caliber`、`gate_points_per_region_ladder`（清单 L655/L689 自相矛盾）、`gate_align_mech_indicator_band`、`protocol_p9_consolidation_drop_min` | 每个常数新增一行 `final=1`（值 + 口径），**不改本行** |
| **C. 门条件阈值未定稿** | 需边界可达性自检（清单 §1.2 附加条款 4 L382）与文献定值 | `protocol_self_sustain_ratio_max`、`protocol_silent_frac_band`、`protocol_drive_elasticity_min`、`target_p5b_ratio_window`、`protocol_p6a_rho_min`、`protocol_acc_min`、`gate_family_p11_perturbation_size` | 先跑**边界可达性自检**（目标数据劈半）⇒ 定稿 margin/阈值 ⇒ 新增 `final=1` 行 |
| **D. 目标数据/落盘件未就绪** | 见各行 note | `target_p10a_region_rate_band`、`target_p10c_dwell_band`（S6 缺"区域级活动时间序列"这一维）、`target_p7_region_function_map`（**表未落盘**：`ls` 实测 No such file）、`target_p11_anchor_set`（锚集 partial） | 数据门/装配门按清单 §2.4 降级路径推进；P7 需先落 `m12_region_function_map.csv` |
| **E. 待独立复算或裁决** | §0.8 #12 / 口径裁决 | `target_p8_psychometric_band`（B1 已算，本节点未复算）、`target_p8_denominator_rule`（`bwm_include` 分母两档相差 48.1%）、`ref_m11_engine_probe_anchors`（M9–M11 继承锚，清单自己标 `unverified` 且 L319 明文禁用于放行协议） | 任一节点在落盘件上独立复算 ⇒ 新增 `measured` 行；分母口径由主 agent 裁决 |

### 7.2 闸门 ③ 的逐行输出（26 行，`final=0` ⇒ 对应判据 `unavailable`）

```
count = 26 / total = 124
protocol_T_main_s                    ⇒ 会让 applies=G1-M/P5a|G1-M/P5c 的判据 unavailable
protocol_settle_ms                   ⇒ 会让 applies=G0-M/P3|G1-M/P5a 的判据 unavailable
protocol_bin_wide_ms                 ⇒ 会让 applies=G1-M/P5a|F/P10a|F/P10c 的判据 unavailable
protocol_smooth_window_frames        ⇒ 会让 applies=F/P10a|F/P10c 的判据 unavailable
protocol_drive_rate_hz               ⇒ 会让 applies=G1-M/P5b 的判据 unavailable
protocol_drive_dose_grid             ⇒ 会让 applies=G1-M/P5b 的判据 unavailable
protocol_self_sustain_ratio_max      ⇒ 会让 applies=G1-M/P5b 的判据 unavailable
protocol_silent_frac_band            ⇒ 会让 applies=G1-M/P5b 的判据 unavailable
protocol_drive_elasticity_min        ⇒ 会让 applies=G1-M/P5b 的判据 unavailable
target_p5b_ratio_window              ⇒ 会让 applies=G1-M/P5b 的判据 unavailable
protocol_p6a_rho_min                 ⇒ 会让 applies=C-M/P6a 的判据 unavailable
protocol_acc_min                     ⇒ 会让 applies=F/P10b 的判据 unavailable
target_p10a_region_rate_band         ⇒ 会让 applies=F/P10a 的判据 unavailable
target_p10c_dwell_band               ⇒ 会让 applies=F/P10c 的判据 unavailable
target_p7_region_function_map        ⇒ 会让 applies=F/P7 的判据 unavailable
target_p11_anchor_set                ⇒ 会让 applies=F/P11 的判据 unavailable
target_p8_psychometric_band          ⇒ 会让 applies=F/P8 的判据 unavailable
target_p8_denominator_rule           ⇒ 会让 applies=F/P8|F/P9 的判据 unavailable
ref_m11_engine_probe_anchors         ⇒ 会让 applies=G0-M/P3 的判据 unavailable
gate_family_p11_perturbation_size    ⇒ 会让 applies=F/P11 的判据 unavailable
gate_align_mech_indicator_band       ⇒ 会让 applies=G0-M/P3 的判据 unavailable
gate_points_per_region_ladder        ⇒ 会让 applies=G0-M/P3|F/P12 的判据 unavailable
target_p5a_cortex_rate_band_hz       ⇒ 会让 applies=G1-M/P5a 的判据 unavailable
target_p5a_noncortex_rate_band_hz    ⇒ 会让 applies=G1-M/P5a 的判据 unavailable
protocol_p9_consolidation_drop_min   ⇒ 会让 applies=F/P9 的判据 unavailable
protocol_pharma_dose_caliber         ⇒ 会让 applies=F/P11 的判据 unavailable
```
（以上为检查器 G3 段的**逐行原文**，仅把行首的 `final=0: ` 前缀省略；要机器可读的完整输出请运行 `python3 neural_exploration/tools/check_m12_targets_gates.py`。）

**⇒ 当前可判 vs 不可判（按清单 §5.5 的 P 编号）**：

| P | 状态 | 依据 |
|---|---|---|
| P1/P2（数据门） | 部分可判 | 计数锚与吞吐门已登记（`ref_*`、`gate_s1_*`、`gate_s6_*`）；S1 的**源维未建立**另由 `ref_s1_source_dim_established` 登记 |
| P3（G0-M） | 部分可判 | 对齐/守卫/探针门已登记；**(c) 的带**与**逐包预算**缺失 |
| P5a（工作点） | ❌ 不可判 | 组 A + 组 B/1、2、3 |
| P5b | ❌ 不可判 | 组 B/5、6 + 组 C（上限/带/下限/下界四缺） |
| P5c | ❌ 不可判 | 组 B/1 |
| P6a | ❌ 不可判 | 组 C（ρ 下限）+ `ref_s1_source_dim_established=False` |
| P7 | ❌ 不可判 | 方向表未落盘 |
| P8/P9 | 部分可判 | 样本量结论**已可判**（`ref_p8_fit_set_vs_n_min`：A=13 < 35 ⇒ 必须扩样本或改白名单档）；P8 的带待独立复算；P9 的带缺失 |
| P10a/P10c | ❌ 不可判 | S6 缺"区域级"这一维 |
| P10b | ❌ 不可判 | `acc_min` 未注册（按 §1.2 F3 应判 `invalid`） |
| P11 | ❌ 不可判 | 锚集 partial + 药理口径三要素未注册 |
| P12 | 可判（门已登记） | 档数与缩放门齐备（`gate_scale_N_ladder`/`gate_p12_scale_steps_min`） |

---

## §8 未确认项（逐条；每条指出卡在哪一行/哪个字段）

| # | 未确认项 | 卡在哪 | 本节点为何不自行裁决 |
|---|---|---|---|
| **U1** | `data/m12_targets.csv` 的列结构：本骨架 **10 列**（9+scope）vs 清单 §5.1 **L887** 的 **11 列**（多 `provenance_verified_by`、`calibration_note`） | `docs/生物仿真M12实施清单.md` **L887** 与派发件「列结构照 M11 的 9 列 + 1 列」直接冲突 | 派发件是**硬约束**、清单是设计件；两者都有效 ⇒ 需主 agent 裁决（两种迁移方案见 §2.2，代价已量化） |
| **U2** | `evidence_level` 的 5 值域**缺少"继承未复算"**的位置 | 清单 L871 与 M11 实测域都是 {measured, derived, literature, literature_unverified, unavailable}；而 M11 §0.10 的继承锚（清单自己标 `unverified（继承）`）无合适取值 | 本骨架用 `derived` + 标签 `[inherited_unverified]` + `final=0` 表达（`ref_m11_engine_probe_anchors`），并在此显式登记该域缺口 |
| **U3** | KB/MB 的进制（10³ vs 2²⁰） | 清单 L460（≥200 KB/s）、L510（≤100 MB / ≥100 KB/s）、L518（2 GB 上限）**均未写明进制** | 本骨架登记**原值**并在 note 标注歧义；已核：S1 的 pass 判定在两种读法下**均成立**（实测 275,148 B/s）⇒ 该歧义不改变已有结论，但会改变 ETA 判定 |
| **U4** | M11 表的记录数：**104**（本节点实测）vs 派发件写「105 行」 | `data/m11_p5_targets.csv` 实测：`csv.DictReader` 得 104 条；`m11_acceptance_master.md` §3dh L3340 记「104 = 103 + 1」 | 只报实测值；105 疑为该次"表头被当数据"的错误计数（R117 §③ 形态） |
| **U5** | 清单内部不一致：每区域点数档 | **L655** `{500,1000,2000,5000}` vs **L689** `{500,1000,2000,5000,10000}`（本节点逐字 grep 确认两处原文不同） | 两候选都登记（`gate_points_per_region_ladder`，value 取 L655 档并在 note 列 L689）⇒ `final=0` |
| **U6** | `protocol_T_main_s = 30` 是否算"已注册" | 清单**无**独立注册行；只有 L712 降档式「T=30 s → 10 s」左值与 L408 命名例 | 按 R117 的严格读法判 `final=0`（"决定判据取值的常数不得散落在非注册位置"）；若主 agent 认为 D5 的左值即注册，本行可新增一行 `final=1`（**不改本行**） |
| **U7** | 清单 §5.5 里若干**声明性/格式性**条款未逐条登记（非数值，故非"常数缺席"） | 例：§2.3 的 9 个 verdict 词表（L591–594）；§3.4 的记录列清单（L695）；§2.5 的 7 条硬断言（L616–618）；§10.1 的**逐包** CPU/GPU 上界（L1116–1127，本表只有合计行） | 属"判据/清单的列清单"，宜由 `m12_criteria.csv` 与报告承载；**但逐包上界是数值** ⇒ 若要做"逐包超包络判定"，须新增 9 行（本节点未加，避免把预算表复制成骨架噪声） |

---

## §9 §2–§5 协议常数覆盖对照（自检 §四.3 的原始表）

**逐条核对方式**：本节点通读清单 §2–§5（**L415–L986**）与 §1.2/§1.3，把每个**数值型/取值域型**常数与下表对照；
每次出现一处"清单有、本表无"即登记。

### 9.1 已覆盖（清单行 → 本表 key）

| 清单行 | 常数 | 本表 key |
|---|---|---|
| L368 | `δ_rel = 0.20×ref` | `gate_margin_delta_rel` |
| L368 | `max(p_low,p_high) < 0.05` | `gate_alpha_tost` |
| L369 | bootstrap ≥10,000 | `gate_bootstrap_n_min` |
| L370 | `n_perm ≥ 1,000` | `gate_perm_n_min` |
| L370 | `acc_min`（无 provenance ⇒ invalid） | `protocol_acc_min`（final=0） |
| L371 | Clopper–Pearson 下界 ≥0.70 | `gate_f4_ci_lower_min` |
| L371 | 分母排除规则 | `gate_f4_denominator_exclusion` |
| L373 | 方向一致率 ≥0.70 | `gate_f6_direction_agreement_min` |
| L380–381 | Holm–Bonferroni + family 预注册 | `gate_mult_comp_method` + `gate_family_*`（4 行） |
| L382 | 边界可达性自检 | `gate_boundary_reachability_selfcheck` |
| L384–385 | `final=1` 才入判定；`p>0.05` 不得单独作 pass | `rule_final1_frozen_after_run`、`gate_p005_branch_form` |
| L284–285 | 量纲自检表达式 | `gate_dimension_check_rule` |
| L286–290 | 双估计器三对 | `gate_two_independent_estimators` |
| L434 | ETA >30 min / >4 h | `gate_eta_object_max_s`、`gate_eta_total_max_s` |
| L433 / L588 | 吞吐 3 轮取中位 / 两探测+多数裁决 | `gate_throughput_probe_rounds` |
| L435 / L1023 | 完整性门 5 步 | `rule_integrity_gate_5_steps` |
| L460 | S1 门（≥200 KB/s / ≤10⁵ 行 / ≥10⁶ 行） | `gate_s1_pass_throughput_Bps`、`gate_s1_pass_rows_max`、`gate_s1_unavailable_rows_min` |
| L470 | S2 门（≤5 MB / ≤20 MB） | `gate_s2_bytes_max` |
| L477 | 25 µm / 100 µm | `protocol_ccfv3_voxel_resolution_options_um` |
| L481 / L653 | 延迟 2–4 档打包 | `gate_delay_group_options` |
| L500 | 白名单 ≤40 session | `gate_ibl_whitelist_max_sessions` |
| L510 | S6 门（≤100 MB / ≥100 KB/s / >220 MB） | `gate_s6_pass_bytes_max`、`gate_s6_pass_throughput_Bps`、`gate_s6_unavailable_bytes_min` |
| L518 | 2 GB 合规硬上限 | `gate_download_size_hard_max_gb` |
| L565 | S12 ≤10 MB | `gate_s12_bytes_max` |
| L639–641 | 定档裁决门（ms/step 8/16/40；RSS 24/30 GB） | `gate_engine_perf_gates` |
| L655 / L689 | N 档、R 档、E/N 档、每区域点数档 | `gate_scale_N_ladder`、`gate_region_count_options`、`gate_region_interareal_indegree_cap`、`gate_points_per_region_ladder`(final=0) |
| L665–672 | (a)(b)(d)(e)(h) 对齐门 | `gate_align_gates` |
| L667 | (c) 的"带" | `gate_align_mech_indicator_band`（final=0） |
| L668 | (d) 确定性层级 | `rule_determinism_level_must_be_declared` |
| L670 | (f) 同步发放判别 | `gate_engine_sync_discriminator` |
| L673–674 | (i)(j) 槽守卫 | `gate_slot_guard_required` |
| L676–679 | 四扩展关闭态等价 | `gate_extension_off_equivalence` |
| L680 / L681 | R-6 边计数 / R-4 参数审计 | `gate_edge_count_identity`、`gate_param_audit_required` |
| L691–697 | 状态档 / 分段形状 / 120 段 / 6.7 GPU-h / MPS 串行 | `gate_probe_state_arms`、`gate_probe_shape`、`gate_budget_constants`、`rule_mps_strictly_serial` |
| L694 | 探针生理标定 1–5 Hz | `gate_probe_physio_rate_band_hz` |
| L701 | held-out 偏差 ≤15% | `gate_probe_holdout_rel_dev_max` |
| L712 | `T ≥ 3× dwell` | `protocol_T_dwell_ratio_min` |
| L716–717 | 降档必须重过门 | `rule_downgrade_explicit_registration` |
| L790 | 无证据臂 ≈0.5 | `gate_no_evidence_arm_chance_level` |
| L806 | P10 组合门（≥2/3 且含①或②） | `gate_p10_pass_composition` |
| L814 | P11 四项前置门 | `gate_p11_preconditions` |
| L872 | `provenance_type` 域 | `protocol_provenance_type_domain` |
| L876 / L892–900 | 四态取值域 | `gate_four_state_enum` |
| L906–915 | `calibration_note` 8 必填字段 | `protocol_calibration_note_required_fields` |
| L919–934 | NC-1..12 负对照必须逐判据有行 | `gate_negative_control_required`（臂清单见交付物 ②） |
| L951 / L945 / L947 / L953 / L954 / L956 | P5b/P6a/P7/P8/P9/P10c/P12 的阈值与带 | 见 §5.1、§7.1（多数为 final=0 缺口行） |
| L966–969 | 文献溯源纪律 | `rule_citation_discipline` |
| L970–972 | append-only | `rule_append_only_no_header`、`rule_final1_frozen_after_run` |
| L1119–1134 | 预算/下载/单试次上限 | `gate_budget_constants` |
| L1155 / L1163 | GAP-01 / GAP-09 | `target_p5a_*`（final=0）、`gate_align_gates`（"关闭态"命名） |

### 9.2 ★「清单有、本表没有」清单（**必报项**）

| # | 清单出处 | 内容 | 为什么本表没有（类别） |
|---|---|---|---|
| 1 | L1116–L1127（§10.1） | **逐包** CPU/GPU 上界 9 行（B0 ≤4 / B1 ≤50 / B2 ≤30+15 / B3 ≤20+25 / B4 ≤20+35 / B5 ≤15+25 / B6 ≤15+30 / B7 ≤12+35 / B8 ≤10+5） | **数值型但属清单复制**：本表只登记合计（`gate_budget_constants` 的 `budget_upper=176CPU-h\|195GPU-h`）。若要做**逐包**超包络判定须补 9 行 |
| 2 | L591–594（§2.3） | 判定词表 9 词（REACHABLE / REACHABLE_NO_DATA / NEEDS_TOKEN / NEEDS_VERIFY / BLOCKED / BLOCKED_DNS / NOT_EXIST / TOO_SLOW / UNLOADABLE） | **取值域**（非阈值）：宜由 `m12_source_reachability.csv` 承载；本表只登记 S1 的阈值 |
| 3 | L695（§3.4） | 探针记录列 10 项（ms/step、墙钟、RSS、driver_allocated、对象数、段间残差、spk/neuron、edges、delay_groups） | **列清单**（非数值常数）⇒ 建议由 `m12_engine_probe.csv` 承载 |
| 4 | L616–618（§2.5） | 数据门 7 条硬断言 | **断言清单**（非数值）⇒ 由 P1/P2 的验收条件承载 |
| 5 | L369、L868 | F1 必须报告的 7 项（`δ`/`Δ̂`/`CI95`/`p_low`/`p_high`/`n_sim`/`n_data`）与 `data/m12_criteria.csv` 的 37 列 | **报告项/列清单**（非数值）⇒ 由 `m12_criteria.csv` 承载 |
| 6 | L1132–1135 | 禁止用 M9/M10/M11 锚外推（外推禁令）+ 预算锚对照（M9 9.7/12 等） | **禁令 + 对照值**：禁令已由 `ref_m11_engine_probe_anchors`（final=0）承载；预算锚对照值属报告项 |
| 7 | L442–446、L459、L479、L499、L509、L519 | 全部**继承吞吐值**（S3 10–17 KB/s、DANDI 7,237 B/s、GCS 2.8–4.0 MB/s、api.brain-map 1.6–16 KB/s 等） | **有意不登记**：M12-N9/§2.1 要求当场实测取代继承值（`ref_throughput_corrections_Bps` 与 `ref_s1_throughput_Bps` 已登记当场实测；`M12-C-04` 明文禁止外推） |
| 8 | L537、L539 | S9 复用 M11 件（零新增下载）／MICrONS 逐细胞轨迹 103.8 GB 永久 unavailable | **声明性条款**（非数值门）⇒ 列入 `data/m12_m11_gap.csv`/报告 |
| 9 | L713（D6） | 云 GPU 档 | **禁令/裁决项**（本里程碑用户明令禁用云 GPU；本节点未用）⇒ 非判据常数 |
| 10 | L833 | `status_history` 半角分号分隔格式约定 | **格式约定**（非数值）⇒ 由 `m12_criteria.csv` 承载 |
| 11 | L897 硬条款③ | 判定表必须四态齐全无空白 | **一致性断言** ⇒ 由 `m12_criteria.csv` 的四态列承载 |

> **⇒ 结论（自检 §四.3 的答案）**：「清单有、本表没有」的**数值型**常数只有 1 项（**逐包预算上界 9 行**）——
> 它是**清单内的复制件**且本表已登记合计行；其余 10 项均为**非数值**（词表/列清单/断言清单/禁令/格式约定/有意不登记的继承吞吐值）。
> **⚠️ 但反向的缺口更严重**：§5.2 列出的 **17 项"决定判据取值却未在清单注册"** 的常数才是 M12 版的 R117 风险，它们全部已在本表落成 `final=0` 行。

---

## §10 自检（逐条 pass/fail）

| # | 自检项 | 实测 | 判定 |
|---|---|---|---|
| 1 | 行数 / 唯一 key / `set(key)` 与自身差集 / `final=0` 计数 / 无重复 key / 列数与 M11 一致+1 | 记录 **124**；唯一 key **124**；差集 **空**；`final=0` **26**；重复 key **0**；列 **10 = 9+1**；`前9列==M11列` **True** | **pass** |
| 2 | 值域检查 `evidence_level ∈ {measured, literature, derived, literature_unverified, unavailable}`、`final ∈ {0,1}` | 实测分布：`derived 86 / unavailable 19 / measured 18 / literature_unverified 1 / literature 0`；`final ∈ {0,1}`（`0`:26、`1`:98）。★ `literature` **0 行**的说明：M12 的文献锚由 B1/G2 逐条 provenance 化，本骨架只引用其结论（`target_p11_anchor_set` 取 `literature_unverified`），**未**自行给任何文献值 ⇒ 该取值在本表为空是预期形态，不是遗漏 | **pass** |
| 3 | 口径常数覆盖对照（清单 §2–§5） | §9.1（覆盖）/ §9.2（缺失：数值型仅 1 项=逐包上界）；**反向缺口 17 项已登记**（§5.2） | **pass**（含显式声明） |
| 4 | 三条闸门的负向测试 | 三个变异各返回 **退出码 1** 并打印具体 ERROR（§6.6） | **pass** |
| 5 | 每个写过的文件给 `wc -l -c` + `shasum -a 256` | 见 §11.3 | **pass** |

---

## §11 附录

### 11.1 本节点实际执行过的命令（可审计；**无 `2>/dev/null`**）

```bash
# 1) 开工前硬约束核对（只读；未 kill 任何进程）
cd /Users/weidong/ai/small_world && pwd && pgrep -f run_m11_p5_arms ; echo "[pgrep exit: $?]"   # ⇒ 无输出、exit=1（无 MPS 作业）
ls                                                                                            # ⇒ 仓库根清单

# 2) 必读材料的实测读回（全部只读）
wc -l -c neural_exploration/data/m11_p5_targets.csv                                            # ⇒ 119 行 / 61,355 B
python3 -c "import csv,hashlib; ..."   # 解析 M11 表：104 记录 / 9 列 / 104 唯一 key / final=1:88 final=0:16
                                             # sha256 = 8ed7f0e1aeee1a29a3b6516efc796a2a3bbffc7babc0491488f616a8ae7a9ba8
grep -n "A-5" neural_exploration/docs/生物仿真M12实施清单.md                                     # ⇒ L90（原文见本文首）
grep -n "^#" neural_exploration/docs/生物仿真M12实施清单.md                                      # 章节定位
# 逐段 read：清单 §1.2/§1.3/§1.4、§2.1–§2.5、§3.1–§3.6、§4.1–§4.10、§5.1–§5.6、§6、§7、§8、§10、§11、附录 A
# 逐段 read：docs/m12_b1_gate_closure.md（全 253 行）、docs/m12_env_notes.md §L11/L12、
#            docs/m11_acceptance_master.md §3cj(R93) §3cn(R97) §3da(R110) §3dh(R117) §3di(R118) §3dr(R127) §3du(R130) §3dv(R131) §3do(R124)

# 3) 本节点自带证据（zero-network；均为只读）
python3 - <<'PY'   # 仅含 stdlib csv/json/hashlib：IBL 聚合表与参考锚读回
# ⇒ IBL_AGG rows=459  sum(n_trials)=295920  max=1525  min=401 ; JSON n_rows=295920 n_sessions=459 sum_equals=True
# ⇒ M12-G-01..09 / M12-C-01..04 逐行读回（值见 data/m12_reference.csv）
PY
ls neural_exploration/docs/m12_region_function_map.csv neural_exploration/data/m12_region_function_map.csv
# ⇒ 两个路径均 "No such file or directory"; exit=1（P7 方向表未落盘，已据此判 target_p7_region_function_map final=0）

# 4) 骨架生成与闸门（把表写盘后逐轮核对）
awk -F, '!/^#/ && NF!=10 && NF!=0 {print NR": NF="NF}' neural_exploration/data/m12_targets.csv   # 字段数核对（终态无输出）
python3 neural_exploration/tools/check_m12_targets_gates.py                                       # ⇒ ALL GATES PASS / EXIT=0
python3 neural_exploration/tools/check_m12_targets_gates.py --negative-test drop-provenance       # ⇒ EXIT=1 + G0 ERROR
python3 neural_exploration/tools/check_m12_targets_gates.py --negative-test break-quad            # ⇒ EXIT=1 + G2 ERROR
python3 neural_exploration/tools/check_m12_targets_gates.py --negative-test dangling-ref          # ⇒ EXIT=1 + G1 ERROR ×2

# 5) 收尾核对
wc -l -c neural_exploration/data/m12_targets.csv neural_exploration/tools/check_m12_targets_gates.py
shasum -a 256 neural_exploration/data/m12_targets.csv neural_exploration/tools/check_m12_targets_gates.py
git status --porcelain    # ⇒ 只出现本节点新增的 3 个文件（见 §11.4）
```

### 11.2 本节点自造的**格式错误与自查拦下的过程**（诚实记录）

| 现象 | 根因 | 发现方式 | 处置 |
|---|---|---|---|
| 一批行的字段数不是 10 | 我在行尾把 `final` 值前的字段分隔**误写成半角分号**（`…;1,applies=…`） | `awk -F, 'NF!=10'` 逐行核对 | 全表替换 `;0,applies=`/`;1,applies=` → `,0,applies=`/`,1,applies=`；**并保留字段数核对作为每轮收尾动作** |
| 另一批行 NF>10 | 字段文本里用了 ASCII 千分位逗号（如 `295,920`）与 ASCII 双引号 | 同上 | 千分位逗号改为无分隔写法；引号改全角/「」；**在追加脚本里加断言**：任何字段不得含 `,`/换行/以 `"` 开头 |
| 一行出现占位残留（`269?`、`1? no → 0.0?`） | 我写作时的半成品未清理 | 提交前的 `grep -n "?"` 自查 | 用实测值替换（`sum(n_trials)=295920 == json::n_rows`）与补全 `final`/`scope` 字段 ⇒ 这正是「**不得编造**」条款要拦的情形，本节点如实登记 |
| 一次数字内部的千分位清理**误吃**了一个字段分隔（`…M12-N10,1,applies=…` → `M12-N101,applies=`） | 正则 `(?<=\d),(?=\d)` 的过度匹配 | 该行 NF 从 10 变 9 | 精确回改；**此后所有批量替换都在逐行 NF 核对之后进行** |

> 这四条都不是"无关紧要的格式问题"：M11 的 R117 §③ 正是一次"追加时的格式/结构错误"（表头被追加）造成的假记录，
> 而**唯一能定位"多的是什么"的检查是差集**（M11 纪律第六十一条）。本节点因此把字段数核对固化为每轮收尾动作。

### 11.3 回读表（`wc -l -c` + `shasum -a 256`）

```
     133   80871  neural_exploration/data/m12_targets.csv
      566  （本文件；其字节数/哈希随每次写入变化 ⇒ 终态值见最终回复，不在此自嵌）
     201   23093  neural_exploration/docs/m12_negative_control_scaffold.md
     314   13547  neural_exploration/tools/check_m12_targets_gates.py
```
（四个文件均已实测；`sha256` 见 §11.5 的终态回读。**本文件自身的终态 sha256 由最终回复给出** —— 任何再写入都会改变它，故不能自嵌。）

### 11.4 声明：未使用 MPS / 未取新数据 / 未改既有件

1. **未使用 MPS**：`pgrep -f run_m11_p5_arms` 开工前实测为空（exit=1）；本轮**从未**执行 `--device mps`，未启动任何 MPS 作业，未 kill 任何进程。
2. **未取新数据**：全部命令零网络（`grep`/`read`/`wc`/`shasum`/`python3`（stdlib `csv`/`json`/`hashlib`/`re`））。
3. **未改既有件**：只**新增** `docs/m12_criteria_table_skeleton.md`、`data/m12_targets.csv`、`docs/m12_negative_control_scaffold.md`、`tools/check_m12_targets_gates.py`；
   `git status --porcelain` 的实测输出见 §11.5。**未触碰** `data/m11_*`、`data/m12_s1_*`、`data/m12_source_*`、`tools/probe_m12_*`、`tools/run_m11_p5_arms.py`、`tools/validate_p11_resting.py`。
4. **未降规模、未用云 GPU**。

### 11.5 终态回读（四个交付物写盘后实测）

```
      133    80871  neural_exploration/data/m12_targets.csv
      （本文件的 wc：见最终回复；自嵌会失效）
      201    23093  neural_exploration/docs/m12_negative_control_scaffold.md
      314    13547  neural_exploration/tools/check_m12_targets_gates.py
```
```
182847d8c2bbf661e5a43e19e9935c9a17154d9a7c66b15047e82984b86ccd8b  data/m12_targets.csv
745cc55751804923fd101383e905d3784134948b699b1e6281034a0effae7843  tools/check_m12_targets_gates.py
869c2e0e13da490f8b5257ae2ebccd1e0c590f93cd99ed9dee3e6682e329429d  docs/m12_negative_control_scaffold.md
（本文件 docs/m12_criteria_table_skeleton.md 的终态 sha256 见最终回复）
```
> 🔴 **本文件的 sha256 不可内嵌**（写入本节即改变哈希）：**本文件的终态 sha256 由本节点的最终回复给出**（与回复同批实测）。
> 其余三个文件不受本节写入影响 —— 上式中的值即终态值（收尾时又各实测一次，见最终回复）。

#### 11.5.1 `git status` 实测（含一处必须如实报告的并发现象）

```
 M neural_exploration/data/m12_targets.csv                     ← 本节点产物（见下）
?? neural_exploration/docs/m12_criteria_table_skeleton.md      ← 本节点新增
?? neural_exploration/docs/m12_negative_control_scaffold.md    ← 本节点新增
?? neural_exploration/tools/check_m12_targets_gates.py         ← 本节点新增
 M neural_exploration/data/m11_p5_prereg_amend_l4.json         ← **非本节点**（另一个节点的产物，mtime 23:22:53）
 M neural_exploration/docs/m11_l4_mechanism_plan.md            ← **非本节点**（同上，mtime 23:23:15）
?? neural_exploration/data/m11_p5_l3_runs/*（一批）             ← **非本节点**（M11 跑臂产物）
```

**① `data/m12_targets.csv` 显示为 ` M` 的原因（本节点实测，不是"改了既有件"）**：

本节点开工时的 `ls -la data/ | grep -i m12` **没有** `m12_targets.csv`（该文件当时不在盘上），`write` 工具亦报告 `Created file`。
`git status` 的 ` M` 来自**并发提交**：本节点工作期间，某个并发进程把本节点**写到一半**的文件提交进了 HEAD
（M11 `docs/m11_env_notes.md` §L18 记录过同一现象："本节点的在写文件被并发提交卷入（R41-3 的复发）"）。
**证据（本节点实测）**：

```
HEAD 版本 = 124 行 / 74,749 B / sha256 775d9b6acdcbeb9e20b236dac45620165269a95b00e059e0b063477166b511d5
当前文件  = 133 行 / 80,871 B / sha256 182847d8c2bbf661e5a43e19e9935c9a17154d9a7c66b15047e82984b86ccd8b
HEAD 版本是当前文件的逐字节前缀 = True      ⇒  本节点之后只做了「追加 9 行 / +6,122 B」
```
⇒ 即：**HEAD 里的那份就是本节点自己的文件**（不是别人的既有件），而本节点对它的后续动作是**纯 append**
（正好演示了 append-only 的"前缀逐字节相同"这一项核对，M11 纪律第六十一条）。

**② MPS 作业状态（如实报告，本节点未启动也未 kill）**：
开工时 `pgrep -f run_m11_p5_arms` ⇒ **空（exit=1）**（符合派发前提）；
**收尾时**同一 `pgrep` ⇒ **4 个 PID（71893 / 71897 / 71916 / 71917）** ⇒ 说明**在这段工作期间有另一节点启动了 M11 的 MPS 作业**。
本节点的命令清单里**没有**任何启动/终止进程的命令；本轮**从未**执行 `--device mps`。
（⇒ 交付物 ② 的 NC 相关条款与本文的 `rule_mps_strictly_serial` 因此也包含一条"运行期可见性"的现实证据。）

**③ 两个被改的 `m11_*` 文件的 mtime = 23:22:53 / 23:23:15**，落在本节点会话窗口内，但**本节点的命令清单中没有任何写它们的命令**
（全部命令为 `grep`/`read`/`wc`/`shasum`/`git`/`python3`（只读）与对本节点 4 个新文件的写）。
⇒ 本节点**无法从 mtime 断定**是谁改的；只能声明**不是本节点**（命令级可审计），并如实登记该现象。


---

*本骨架为 M12 的**判据带骨架**，不是判据带定稿：`final=0` 的 26 行按 A-5 行的 append-only 规则，
待对应门/注册动作完成后由**新增 `measured` 行**替换（旧行 `final` 保持 0 并加更正指针）。*
