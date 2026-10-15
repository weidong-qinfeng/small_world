# M11 评估器 `validate_p11_resting.py`：R200 §③ 两处缺陷（A/B）修复与回归对照

**节点**：M11 执行节点（本文件由该节点交付）
**日期**：2026-10-15
**依据**：`docs/m11_acceptance_master.md` **§3gl = R200 §③**（两处成因定位）＋ **§3do = R124**（`(e)` 口径：只在基准臂上判）
＋ `data/m11_p5_targets.csv::ei_ratio_lo/hi`（`[0.5,3.0]`，均 `final=1`）／`selfsustain_ratio_max=0.10`（L46）
＋ `data/m12_targets.csv::rule_e_baseline_arm_only`（L48，M12 侧同名规则）
**被修文件（全项目唯一被改的已存在文件）**：`neural_exploration/tools/validate_p11_resting.py`
**未触碰**：`tools/report_m11_l3.py`（主判；sha 仍 `69a8a12732cd2ab2459c3c2ff170ff123b553e335dafa89ffceafdad1b9e5dd6`）、
`data/m11_p5_targets.csv`（sha 仍 `0991808242fd85481e8ea907e453ef3b7c8d7c0dedbfda288910ea9b644f3871` = R200 登记 `TGT_SHA` ✓）、
`data/m12_targets.csv`、`docs/m11_acceptance_master.md`、任何 `src/*.py`、任何已登记产物。

---

## §0 结论摘要（五句）

1. **缺陷 A 已修**：`(e)` 的取值改为 **`baseline_arm_only`**（只取 `role=="baseline"` 且 `drive_mult==1.0` 的臂）⇒
   L5 的 4 个实格 `(e)` = **1.2309759 / 1.5161500 / 1.3797731 / 1.3782987 ⇒ 全部 `pass`**（带 `[0.5,3.0]`），
   与主判 reporter 登记的基准臂口径 **1.230976 / 1.51615 / 1.379773 / 1.378299 逐位一致** ✓
2. **缺陷 B 已修**：门条件只由**格条件齐备**（同时提供 `baseline` 与 `zero_drive` 的臂）的格贡献 ⇒
   L5 的 `gate_conditions` = **`c=pass, e=pass, tension=pass`**（旧：`c=unavailable, e=fail, tension=unavailable`）；
   **被排除的格与原因显式打印且登记进输出**（`p5_L5_work_d2.0@10000ms`，`missing_roles=['baseline','zero_drive']`）✓
3. **改后总判仍为 `unavailable`（exit 3）**，但 `reason` 已换为 **`硬门⑤：缺「无机制基线臂」的同一判据值 ⇒ …不得判 pass`**
   —— 这是 **R120/§③ 早已登记的语义**（本链 `mechanism="lever_*"` ⇒ 无 `mechanism=="none"` 基线臂），**不是**缺臂格污染，
   **本次未动、也不应动硬门⑤**（R203 §④ 已由主 agent 独立核验并登记同一结论）✓
4. **回归对照（10 个协议 × 旧/新版本）**：所有差异**逐条归因到 A 或 B**——
   `(e)` 行（`value`/`status`/`reason`）、`(e)` 的判据级聚合、门条件、以及 3 个**新增键**（`ei_scope` /
   `gate_aggregation_scope` / `gate_criterion_status`）与 `aggregation_rule` 的**追加式**补写；
   **CSV 列集合与注释行逐字节不变**，`arms`/`arm_metrics`/`output_invariants`/`gate_composition`/`duration_ladder`/
   `tension`/`hard_gates`/`non_discriminative_criteria` 等**逐字不变**（§5）。
5. **一句话回归风险**：本修改**只可能**改变"(e) 的取值与判定"与"门条件的聚合作用域"两类结果；
   由于 `(e)` 的新口径正是 R124 登记的口径，**所有按基准臂口径登记的 `(e)` 数值（R124 §④ 的 0.6748 / 2.5970、
   L3 报告的 1.5769–2.9400、R124 §⑤ 引用的 1.8635 / 1.8499·1.8545·1.8530）在改后逐位成立**（§7）；
   唯一的**新**影响面是"**格不齐备**"的产物（`e_twin`、`p0_*`）其门条件由"混合口径的 pass/fail"变为
   `unavailable`（= 未测，**未测 ≠ 未命中**），该变化在本文件 §8 逐条登记，**未发现**任何已登记结论由此被推翻。

---

## §1 缺陷 A：`(e)` 的口径伪影（R124／R200 §③1）

### 1.1 行号级定位（**旧版 = `_archive/validate_p11_resting.0d3c1d990dfe.py`，行号即原文行号**）

| 位置 | 旧行号 | 旧代码（逐字） | 问题 |
|---|---|---|---|
| 调用点 | **L751** | `self._eval_ei(cfg, T_ms, mech, base_arms + zero_arms + dose_arms, n_neuron)` | 把**该格全部臂**（baseline+zero_drive+dose）交给 `(e)` |
| 取值处 | **L954–958** | `def _eval_ei(...)`；`vals = [self.arm_metrics[a.arm_id].get("ei_ratio") for a in arms]`；`val = float(np.median(vals)) if vals else None` | **`arms` = 全部角色** ⇒ 中位数被零驱动臂拉高 |
| 已有的正确路径（未走） | **L660–681** | `_baseline_arm_value(...)`：只取「`mechanism=="none"` 且 `role∈(baseline,dose,zero_drive)` 且 `drive_mult==1.0`」 | 两条路径口径不同，而 `(e)` 走的是前者（R124 §② 原文） |

**实测复现（未修改工具、既有 L5 产物，输出写 `/tmp`）**：

| 格 | 基准臂 `ei_ratio`（3 臂） | 零驱动臂 `ei_ratio`（3 臂） | 旧 `(e)`（全臂中位数） | 旧判定 |
|---|---|---|---|---|
| `p5_L5_fs_g0.30` | 1.230976 / 1.233570 / 1.219910 ⇒ 中位 **1.230976** | 7.932217 / 7.997702 / 7.783317 ⇒ 中位 **7.932217** | **4.5084438** | `fail` |
| `p5_L5_fs_g0.40` | 1.516150 / 1.522803 / 1.503865 ⇒ **1.516150** | 7.726655 / 7.773319 / 7.487700 ⇒ **7.726655** | **4.5052516** | `fail` |
| `p5_L5_work_T20000` | 1.380173 / 1.379773 / 1.371155 ⇒ **1.379773** | 7.790080 / 7.894118 / 7.668265 ⇒ **7.790080** | **4.5242192** | `fail` |
| `p5_L5_work_T40000` | 1.378299 / 1.379523 / 1.370585 ⇒ **1.378299** | 7.779920 / 7.897108 / 7.664121 ⇒ **7.779920** | **4.5218220** | `fail` |

（零驱动臂 7.49–8.00 与基准臂 1.22–1.52 混池 ⇒ 中位数 ≈4.51，与 R200 §③1 的定位逐位一致；单臂中位数取自
`data/m11_p5_l5_resting_eval.json::arm_metrics`，该键在改前/改后**逐字相同**。）

### 1.2 最小改动（新版行号）

| 新版位置 | 改动 |
|---|---|
| **L698–716**（新增静态方法 `_ei_scope_arms`） | 只保留 `role=="baseline"` **且** `drive_mult==1.0` 的臂；其余臂进 `excluded`（带 `arm_id/role/drive_mult/reason`）⇒ **不静默丢弃** |
| **L1052–1092**（`_eval_ei` 重写取值段） | `scope_arms, scope_excluded = self._ei_scope_arms(arms)`；`vals` 只由 `scope_arms` 取；`reason` 追加 `口径=baseline_arm_only…；所用臂=[…]；排除臂=[[arm_id,role,drive_mult],…]` |
| **L1064–1074**（新增登记） | 逐格写入输出 JSON 的 `ei_scope`（`arms_used` / `arms_excluded` / `ei_ratio_median` / `band` / `basis`） |
| **L1092**（`arm=` 兜底） | `arm=(scope_arms[0] if scope_arms else (arms[0] if arms else None))` ⇒ **4 个实格的 `protocol_seed` 等口径字段逐字不变**（仍取首个基准臂） |
| **L1786–1787**（`main` 输出） | 新增顶层键 `ei_scope` |
| **L1052 起源的 `unavailable` 分支** | 无 `baseline` 臂的格 ⇒ `baseline_arm_missing(…口径 baseline_arm_only 不可判…)`（**不从其它角色代判**） |

### 1.3 修复后 `(e)` 逐格值（L5，`--out-* /tmp/m11_p5_eval_fixAB/fixed/`）

| 格 | 新 `(e)` | 新判定 | 口径内所用臂 |
|---|---|---|---|
| `p5_L5_fs_g0.30`@10000ms | **1.2309758817580405** | `pass` | `l5_fs_g0.30_8192_T10000ms_d1.0_s{0,1,2}` |
| `p5_L5_fs_g0.40`@10000ms | **1.5161499563526044** | `pass` | `l5_fs_g0.40_8192_T10000ms_d1.0_s{0,1,2}` |
| `p5_L5_work_T20000`@20000ms | **1.3797730971230793** | `pass` | `l5_T20000_d1.0_s{0,1,2}` |
| `p5_L5_work_T40000`@40000ms | **1.3782986986188162** | `pass` | `l5_T40000_d1.0_s{0,1,2}` |
| `p5_L5_work_d2.0`@10000ms | `None` | `unavailable` | 无（该格只有 `dose` 臂 ⇒ **依口径不可判**；旧值为混合口径 0.9630646 `pass`） |

**与主判 reporter 对照**：`1.230976 / 1.51615 / 1.379773 / 1.378299`（R200 §① 登记值）⇒ 新评估器**四位有效数字以上逐位一致** ✓

---

## §2 缺陷 B：门条件被"缺臂的辅助格"污染（R200 §③2）

### 2.1 行号级定位

| 位置 | 旧行号 | 旧代码（逐字） | 问题 |
|---|---|---|---|
| 判据级聚合 | **L1303–1314** | `crit_status[c]` = 该判据**全部行**的最坏优先聚合（`invalid > fail > pass > unavailable`） | 行来自**全部 5 个格**，含只有 `dose` 臂的第 5 格 |
| 门条件取值 | **L1328–1330** | `gate = {}` / `for c in GATE_CRITERIA: gate[c] = crit_status.get(c, "unavailable")` | **直接把全量判据级状态当门条件** ⇒ `p5_L5_work_d2.0` 的 `(c)=unavailable` 拉低门条件 |

**实测复现（旧版）**：
`verdict_G1 = fail`，`reason = G1 门条件未过：['p5_c_selfsustain=unavailable', 'p5_e_ei_balance=fail', 'p5_g1_tension_quantified=unavailable']`；
而同一 run 的 4 个实格 `(c)` = `0.057771 / 0.0757352 / 0.0682339 / 0.0702513` **全部 `pass`**（`c_window_ok=True, in_window=True`）；
第 5 格 `p5_L5_work_d2.0`（`roles_present=['dose']`）的 `(a)(b)(c)` 全 `unavailable`；
其张力行也只有 1 个剂量点 ⇒ `p5_g1_tension_quantified=unavailable`（**同一成因**）。

### 2.2 最小改动（新版行号）

| 新版位置 | 改动 |
|---|---|
| **L718–757**（新增 `_gate_aggregation_scope`） | 定义**格齐备** = 该 `(config_id, T_ms, mechanism)` 格同时提供 `role=baseline` 与 `role=zero_drive` 的臂；把每个格登记为 `units_used` 或 `units_excluded`（`roles_present` / `missing_roles` / `reason`），**并给出规则文本** |
| **L666–667**（`run()`） | 在 `_verdict` 之前算出 `self.gate_aggregation_scope` |
| **L1449–1472**（`_verdict`） | 门聚合改为**只遍历齐备格的行**；聚合规则（最坏优先）**与旧代码逐字相同**，只是行集合受限 ⇒ 差异可逐条归因；`gate = {c: gate_status.get(c, "unavailable") for c in GATE_CRITERIA}` |
| **L1507–1512**（返回） | 新增 `gate_criterion_status`（只含齐备格）与 `gate_aggregation_scope`；`aggregation_rule` **只追加**一段说明，**原句逐字保留** |
| **L1568–1577**（`print_summary`） | **显式打印**齐备格清单与**被排除的格及其原因** |
| **未改** | `crit_status`（⇒ 输出键 `criterion_status`）**仍按全量行**计算 ⇒ **任何格都不被静默丢弃**；`GATE_CRITERIA`／`gate_g1_conditions_v2` 声明／`_assert_gate_composition` **未动**（自检仍通过） |

### 2.3 修复后 `verdict_G1` 与 `reason`（L5）

```
M11 P5 / G1 评估：**unavailable**  exit=3
  reason: 硬门⑤：缺「无机制基线臂」的同一判据值 ⇒ 判别性未登记，整体不得判 pass
  门条件: {'p5_c_selfsustain': 'pass', 'p5_e_ei_balance': 'pass', 'p5_g1_tension_quantified': 'pass'}
  门条件聚合作用域（R200-B）：齐备格 4 个 → ['p5_L5_fs_g0.30@10000ms|mech=lever_l4',
    'p5_L5_fs_g0.40@10000ms|mech=lever_l4', 'p5_L5_work_T20000@20000ms|mech=lever_l4',
    'p5_L5_work_T40000@40000ms|mech=lever_l4']
  ⚠ 被排除的格 1 个（其行仍原样在 CSV/rows 中；仅不参与门条件）：
    - p5_L5_work_d2.0@10000ms|mech=lever_l4  roles_present=['dose'] missing_roles=['baseline', 'zero_drive']
      ⇒ unit_incomplete(缺角色=['baseline', 'zero_drive'])：该格的行照常输出，但门条件的作用域前提
        （同时提供 baseline 与 zero_drive）不满足 ⇒ 不得贡献门条件（R200 §③2）
```

- **`reason` 里不再有** `p5_c_selfsustain=unavailable`（旧的缺臂格污染）✓
- 新的 `reason` 来自**硬门⑤**（评估器要求存在 `mechanism=="none"` 的同判据基线臂；本链为 `mechanism="lever_l4"`）
  ⇒ 这正是 R120/§③「评估器对本链判不了 G1 ⇒ 以 reporter 为主判」的**已登记语义**，**残余 `unavailable` 不是缺陷**，
  主 agent 已在 **R203 §④** 独立核验并登记同一结论。（旧 `verdict_G1=fail` 属**伪影**：门条件被污染 + `(e)` 伪影。）
- **被排除的格与原因已显式打印**（stdout）+ 结构化登记（JSON `gate_aggregation_scope.units_excluded`）✓

---

## §3 基线复现（先证明"我的对照是可信的"）

用**未修改**的评估器（改前真实文件，`PROJECT_ROOT` = 项目根）对既有 L5 产物复算，输出写
`/tmp/m11_p5_eval_fixAB/baseline/`，与**已登记产物** `data/m11_p5_l5_resting_eval.{csv,json}` 对比：

| 对比 | 结果 |
|---|---|
| `data/m11_p5_l5_resting_eval.json` vs 复算 JSON（排除 `generated_at`） | **差异条数 = 0**（逐字段/逐行全同）|
| `verdict_G1` / `reason` / `gate_conditions` | 与 R200 §③ 所载逐字一致（`fail`；含 `p5_c_selfsustain=unavailable`）✓ |
| 未覆盖已登记产物 | 旧、新两组输出**全部写在新路径**（`/tmp/…`）；`data/m11_p5_l5_resting_eval.csv/json` 的 sha 改前改后均为 `b443f4ee…` / `df625fe3…`（未变）✓ |

**⇒ 结论**：改前基线**可信且可复现（0 差异）**，故 §4/§5 的"前后对照"具有逐字段效力。
（附带事实：评估器在同一输入下**完全确定**：新版本连跑两次，CSV 逐字节相同、JSON 除 `generated_at` 全同。）

---

## §3b 主 agent 复核要求的三条判据：逐条复核与证据（2026-10-15 补完）

> **背景澄清（重要）**：主 agent 第一次复核所用的 sha **`12cec98096f68df1…`** 是本节点**顺序改动的中间态**
> （A 已落盘、B 尚未落盘）⇒ 它观察到的
> `G1 门条件中存在未测项：['p5_c_selfsustain=unavailable', 'p5_e_ei_balance=unavailable', 'p5_g1_tension_quantified=unavailable'] ⇒ 不得判 pass（未测 ≠ 未命中）`
> （`verdict_G1=unavailable`，exit 3）正是"**门条件仍读全量行**"的旧行为（在中间态下 `(e)` 已按 A 改为基准臂口径 ⇒
> 全量聚合为 `4 pass + 1 unavailable` ⇒ `unavailable`；`(c)`/张力仍被第 5 格拉成 `unavailable`）。
> （**说明**：上述"中间态成因"是按代码逻辑（门条件读 `crit_status` 全量行、最坏优先）**推得的分析结论**；
> 本节点**没有**重建中间态文件再跑——该态的直接证据即主 agent 自己那次冒烟的输出。）
> **B 落盘后**（= 当前 sha `a9c2756ecf63c61f…`，主 agent 已在 **R203 §③** 独立复核并登记为 ✅）门条件已被那 4 个实格接管。
> 本节按主 agent 的三条判据**重新独立复核**，输出写 `/tmp/m11_p5_eval_fixAB/recheck/**`（**未覆盖**任何已登记产物）。

### 3b.1 判据①：门条件由「格齐备」的 4 个实格贡献（**达成**）

复跑命令（与主 agent 的命令同参，仅换输出路径）：

```
PYTHONHASHSEED=0 MPLBACKEND=Agg .venv-m11/bin/python neural_exploration/tools/validate_p11_resting.py \
  --protocol-json .../data/m11_p5_protocol_p5_L5_confirm_working_point.json \
  --targets-csv   .../data/m11_p5_targets.csv \
  --out-csv /tmp/m11_p5_eval_fixAB/recheck/eval.csv --out-json /tmp/m11_p5_eval_fixAB/recheck/eval.json
```

**改后的 `verdict_G1` 与 `reason`（逐字）**：

```
M11 P5 / G1 评估：**unavailable**  exit=3
  reason: 硬门⑤：缺「无机制基线臂」的同一判据值 ⇒ 判别性未登记，整体不得判 pass
  门条件: {'p5_c_selfsustain': 'pass', 'p5_e_ei_balance': 'pass', 'p5_g1_tension_quantified': 'pass'}
```

**由哪些格贡献门条件（逐门条件逐行，`primary=1` 行）**：

| 门条件 | ✅ 贡献（齐备格 4 个） | ⛔ 被排除 |
|---|---|---|
| `p5_c_selfsustain` | `g0.30@10000ms` `n_seeds=3` zero_spikes=8159 `pass` 0.0577710；`g0.40@10000ms` 3 / 10765 `pass` 0.0757352；`work_T20000@20000ms` 3 / 19331 `pass` 0.0682339；`work_T40000@40000ms` 3 / 39809 `pass` 0.0702513 | `p5_L5_work_d2.0@10000ms`（`unavailable`）|
| `p5_e_ei_balance` | 同上 4 格：`1.2309759 / 1.5161500 / 1.3797731 / 1.3782987` 全 `pass` | 同上一格（`unavailable`）|
| `p5_g1_tension_quantified` | 同上 4 格：ρ(驱动,静默) = `-0.891133 / -0.933257 / -0.933257 / -0.933257` 全 `pass` | 同上一格（`unavailable`）|

⇒ **3 条门条件各自 4 行全部 `pass`、均无 `fail`/`unavailable` 行 ⇒ 判据级 = `pass`**；
`criterion_status`（**全量**，仍原语义）与 `gate_criterion_status`（**只含齐备格**）的并列即为可核对的定位手段：
`{c: unavailable→pass, e: unavailable→pass, tension: unavailable→pass}`。

### 3b.2 判据②：被排除的格及其原因**显式打印**（**达成**）

```
门条件聚合作用域（R200-B）：齐备格 4 个 → ['p5_L5_fs_g0.30@10000ms|mech=lever_l4',
  'p5_L5_fs_g0.40@10000ms|mech=lever_l4', 'p5_L5_work_T20000@20000ms|mech=lever_l4',
  'p5_L5_work_T40000@40000ms|mech=lever_l4']
⚠ 被排除的格 1 个（其行仍原样在 CSV/rows 中；仅不参与门条件）：
  - p5_L5_work_d2.0@10000ms|mech=lever_l4  roles_present=['dose'] missing_roles=['baseline', 'zero_drive']
    ⇒ unit_incomplete(缺角色=['baseline','zero_drive'])：该格的行照常输出，但门条件的作用域前提
      （同时提供 baseline 与 zero_drive）不满足 ⇒ 不得贡献门条件（R200 §③2）
```

结构化登记（JSON）：`gate_aggregation_scope.units_used`（4 项）/ `.units_excluded`（1 项，含
`roles_present`、`missing_roles`、`reason`）+ `gate_criterion_status`（只含齐备格）。
**该格的行未被删改**：其 `(a)(b)(c)` 仍为 `unavailable`、`(e)` 仍为 `unavailable`（`baseline_arm_missing`），
CSV/`rows` 中原样保留 ✓

### 3b.3 判据③：「格齐备」是否含 **seed 数达标** —— 本 run 两种定义**逐格等价**，且**未放宽任何语义**

1. **本 run 的齐备格同时满足全部样本量前提**（数据来源：输出行的 `n_seeds` / `zero_drive_spikes` 列 +
   臂清单角色计数）：

| 齐备格 | 臂角色计数 | 基准臂 seed 数（需 ≥ `protocol_seeds_min_per_arm=3`）| 零驱动臂 seed 数（需 ≥3）| 零驱动总脉冲（需 ≥ `selfsustain_zero_drive_min_spikes=300`）|
|---|---|---|---|---|
| `p5_L5_fs_g0.30@10000ms` | baseline 3 + zero_drive 3 | **3 ✓** | **3 ✓** | **8159 ✓** |
| `p5_L5_fs_g0.40@10000ms` | baseline 3 + zero_drive 3 | **3 ✓** | **3 ✓** | **10765 ✓** |
| `p5_L5_work_T20000@20000ms` | baseline 3 + zero_drive 3 | **3 ✓** | **3 ✓** | **19331 ✓** |
| `p5_L5_work_T40000@40000ms` | baseline 3 + zero_drive 3 | **3 ✓** | **3 ✓** | **39809 ✓** |

   ⇒ **"角色齐备"与"角色齐备 ∧ seed 达标 ∧ 脉冲达标"在本 run 上是同一个 4 格集合**（逐格相同）✓
2. **本实现的"齐备"= 角色齐备**（依据 **R200 §③2 的字面**："缺 `baseline`/`zero_drive` 的格不得把门条件拉成 `unavailable`"），
   **不把"样本量不足"作为排除理由** ⇒ 因为若把"seed 不达标"也算作"不齐备"，就会把**真正未测**的格**排除掉**、
   从而让门条件**更容易判 pass** —— 那是**放宽**。为证这一点，本节点做了一个**反证 probe**（只写 `/tmp`）：

   **probe 设计**：取 L5 协议副本，**删掉 `p5_L5_fs_g0.30` 的 2 个基准臂**（`d1.0_s1`、`d1.0_s2`）⇒ 该格
   **角色仍齐备**（baseline 1 + zero_drive 3）但**基准臂 seed 数 1 < 3**。协议副本写
   `/tmp/m11_p5_eval_fixAB/recheck/probe_seed_inadequate.json`（**未触碰任何登记文件**，仅 25 臂的 `/tmp` 副本）。

   **probe 结果**：

   ```
   probe verdict_G1: unavailable
   probe gate_conditions: {'p5_c_selfsustain': 'unavailable', 'p5_e_ei_balance': 'pass', 'p5_g1_tension_quantified': 'pass'}
   门条件聚合作用域：4 个齐备格（含上述 seed 不达标的 g0.30）| 排除: ['p5_L5_work_d2.0@10000ms|mech=lever_l4']
   (c) 行 g0.30: n_seeds=3 status=unavailable reason=sample_size_gate_unmet: 基准臂 seed 数 1 < 3；
                点估计仅作描述性（点估计 R=0.173084…）
   ```

   ⇒ **该格"角色齐备但判据本身 `unavailable`"时，门条件 `(c)` 仍为 `unavailable`（总判 `unavailable`）** ——
   即主 agent 判据③要求的"**若某格确实齐备但判据本身 `unavailable`，门条件仍应是 `unavailable`**"**成立** ✓
   ⇒ **本次修复没有放宽任何判据、阈值或"未测 ≠ 未命中"语义**：唯一被排除的是**角色缺失**（构造上无法支撑 `(c)/(e)`）的格。

---

## §4 前后对照表（L5，同一命令、仅换输出路径）

命令（两次唯一差别是 `--out-*`）：

```
PYTHONHASHSEED=0 MPLBACKEND=Agg .venv-m11/bin/python neural_exploration/tools/validate_p11_resting.py \
  --protocol-json .../data/m11_p5_protocol_p5_L5_confirm_working_point.json \
  --targets-csv   .../data/m11_p5_targets.csv \
  --out-csv <新路径> --out-json <新路径>
```

| 项目 | 旧版（改前） | 新版（改后） | 归因 |
|---|---|---|---|
| `(e)` `p5_L5_fs_g0.30` | `4.5084438` `fail` | **`1.2309759` `pass`** | **A** |
| `(e)` `p5_L5_fs_g0.40` | `4.5052516` `fail` | **`1.5161500` `pass`** | **A** |
| `(e)` `p5_L5_work_T20000` | `4.5242192` `fail` | **`1.3797731` `pass`** | **A** |
| `(e)` `p5_L5_work_T40000` | `4.5218220` `fail` | **`1.3782987` `pass`** | **A** |
| `(e)` `p5_L5_work_d2.0`（只有 dose 臂） | `0.9630646` `pass`（混合口径） | `None` `unavailable`（口径不可判） | **A**（副作用） |
| `criterion_status['p5_e_ei_balance']` | `fail` | `unavailable` | **A**（4 pass + 1 unavailable 的全量聚合） |
| `state_counts` | `{pass 22, fail 16, invalid 0, unavailable 22}` | `{pass 25, fail 12, invalid 0, unavailable 23}` | **A**（+4 `(e)` fail→pass；1 `(e)` pass→unavailable，账目完全对得上） |
| `gate_conditions` | `c=unavailable, e=fail, tension=unavailable` | **`c=pass, e=pass, tension=pass`** | **A+B** |
| `verdict_G1` / `exit_code` | `fail` / `1` | `unavailable` / `3` | **A+B**（旧 `fail` 为伪影） |
| `verdict_reason` | `G1 门条件未过：['p5_c_selfsustain=unavailable', 'p5_e_ei_balance=fail', 'p5_g1_tension_quantified=unavailable']` | `硬门⑤：缺「无机制基线臂」的同一判据值 ⇒ 判别性未登记，整体不得判 pass` | **A+B**（残余为硬门⑤，已登记语义） |
| 被排除格打印 | 无 | **齐备格 4 个 + 排除格 1 个（含 `missing_roles` 与原因）** | **B**（新增，R200 §③ 要求） |
| `baseline_arm_registered` | `false` | `false` | **不变**（硬门⑤ 的输入，未动） |

---

## §5 "变 / 不变"清单（逐字段，L5）

### 5.1 CSV（`eval.csv`：60 行 × 41 列 + 2 行 `#` 注释）

| 项 | 结论 |
|---|---|
| 注释行、表头（41 列，含 11 个 `(d)` 特有列） | **逐字节不变**（CSV 列集合**未增未减**）✓ |
| 60 行中变更的行 | **仅 5 行**，且**全部是 `criterion=p5_e_ei_balance`**（第 7/18/29/40/51 行） |
| 这 5 行变更的列 | **仅 `value` / `status` / `reason` 三列**；`arm`/`protocol_*`/`band_*`/`estimator`/`note`/`discriminative`/`baseline_*` 等**逐字不变** |
| 其余 55 行（含 `(a)(b)(c)(d)(f)(g)(h)(i)` 与 5 条张力行） | **逐字节不变** ✓ |

### 5.2 JSON（顶层 31 键 + 3 新键）

| 顶层键 | 变/不变 | 说明 |
|---|---|---|
| `arms`（27 臂清单） | **逐字不变** | 未增删臂、未改角色 |
| `arm_metrics`（逐臂三估计器等） | **逐字不变** | ⇒ **`estimator` 语义未变**；`(e)` 的臂级读数也逐位未变（变的是**聚合口径**） |
| `output_invariants` | **逐字不变** | `{rows 60, units 5, criteria_per_unit [10], …}` |
| `gate_composition` | **逐字不变** | 门组成仍 = `c ∧ e ∧ tension`（`gate_g1_conditions_v2`）|
| `hard_gates`（六条硬门文本） | **逐字不变** | 硬门⑤ 语义未动 |
| `non_discriminative_criteria` | **逐字不变** | `['p5_d_drive_elasticity']` |
| `duration_ladder` | **逐字不变** | 5 个格的阶梯状态与漂移全同 |
| `tension` | **逐字不变** | 5 个格的剂量点/两条 Spearman/`n_both_in_band` 全同 |
| `hard_failures` / `guard_failures` / `estimator_gate_unmet` | **逐字不变** | 均 `[]` |
| `protocol` / `protocol_path` / `targets_csv` / `n_arms` / `role` / `generated_by` / `discipline_45_rule` / `elapsed_s` | **逐字不变** | — |
| `baseline_arm_registered` | **不变** | `false` |
| `rows` | 变 | **仅 5 个 `(e)` 行的 `value`/`status`/`reason`**（与 CSV 同源）|
| `criterion_status` | 变 | **仅 `p5_e_ei_balance` 一项**（`fail`→`unavailable`）；其余 9 项**逐字不变** |
| `criterion_row_status_histogram` | 变 | **仅 `p5_e_ei_balance` 的子计数**（`{fail 4, pass 1}` → `{pass 4, unavailable 1}`）|
| `state_counts` | 变 | 见 §4（账目闭合）|
| `gate_conditions` | 变 | A+B |
| `verdict_G1` / `verdict_reason` / `exit_code` | 变 | A+B |
| `aggregation_rule` | 变（**追加**） | 原句**逐字保留**，其后追加一段 `【v2/R200-B 补写（只追加，原句保留）】…` |
| `ei_scope` | **新增** | A 的可核对登记（逐格 `arms_used` / `arms_excluded` / `ei_ratio_median`）|
| `gate_aggregation_scope` | **新增** | B 的可核对登记（`units_used` / `units_excluded`+原因 / `rule`）|
| `gate_criterion_status` | **新增** | 只含齐备格的判据级状态（与 `criterion_status` 并列，便于定位差异）|

**⇒ "变"的部分 100% 可归因到 A（`(e)` 口径）或 B（门聚合作用域）**，其余**逐字不变**。

---

## §6 跨数据集回归对照（旧版 vs 新版，10 个协议）

方法：**冻结原版**取自归档件（`_archive/validate_p11_resting.0d3c1d990dfe.py`，逐字节原文；用
`/tmp/m11_p5_eval_fixAB/run_frozen.py` 以"原登记 `__file__`"语义执行 ⇒ `PROJECT_ROOT` 与登记时一致；
该夹具已在 L5 上**证明与登记产物 0 差异**）。两组输出均写 `/tmp`。

| 协议（`data/m11_p5_protocol_*.json`） | 格 / 齐备格 | `verdict_G1` 旧→新 | 门条件变化 | `(e)` 行变化 | 非 `(e)` 字段变化 |
|---|---|---|---|---|---|
| **`p5_L5_confirm_working_point`**（本任务目标） | 5 / **4** | `fail` → `unavailable` | `c: unavail→pass`、`e: fail→pass`、`tension: unavail→pass` | 5 行（4 值变、1 值→空） | 无 |
| `l1_sigma_n8192` | 4 / 4 | `invalid` → `invalid`（同因 `(c)`） | `e: fail→pass` | 4 行值 | 无 |
| `g1_star_n8192` | 11 / 11 | `invalid` → `invalid` | 无 | **0 行值变**（仅 `reason` 追加口径标注） | 无 |
| `g1_repl_cd` | 3 / 3 | `invalid` → `invalid` | 无 | **0 行值变**（仅 `reason`） | 无 |
| `p5_L3_sigma_cliff_bias_scan` | 5 / 5 | `invalid` → `invalid` | `e: fail→pass` | 5 行值 | 无 |
| `p5_L4_0_gmax_recurrent_gain` | 3 / 3 | `fail` → `fail`（同因 `(c)`） | `e: fail→pass` | 3 行值 | 仅 `verdict_reason` 的 `(e)` 项 |
| `e_twin` | 1 / **0** | `fail` → `unavailable` | `e: fail→unavail`、`tension: pass→unavail` | 1 行值→空 | 无 |
| `p0_n1000_T1000_s0` | 1 / **0** | `fail` → `unavailable` | `e: fail→unavail` | 1 行值→空 | 无 |
| `p0_n31000_T10000_s0` | 1 / **0** | `unavailable` → `unavailable` | `e: pass→unavail`、`tension: pass→unavail` | 1 行值→空 | 仅 `verdict_reason` 文本 |
| `p0_n8192_T10000_s0` | 1 / **0** | `unavailable` → `unavailable` | `e: pass→unavail`、`tension: pass→unavail` | 1 行值→空 | 仅 `verdict_reason` 文本 |

**读法**：① 10 个协议中，**除 `(e)` 与门条件以外的任何字段都没有变**（CSV 变更列的统计一律只有
`(e)` 的 `value`/`status`/`reason`）；② "格齐备"的协议里 `(e)` 一律由 `fail`（或混合口径值）变为**基准臂口径值**；
③ "格不齐备"（无 `baseline` 臂）的协议里，门条件由"混合口径的 pass/fail"退回 **`unavailable`（= 未测）**，
**总判的种类在 2 个协议上由 `fail` 变为 `unavailable`**（§8 逐条登记）。

---

## §7 与已登记数值的交叉核对（改后逐位命中）

| 登记出处（逐字） | 登记值 | 改后评估器（本文件实测） | 命中 |
|---|---|---|---|
| R124 §④「`bias=0.70` 基准臂 `(e)`=**0.6748**」 | 0.674811 | `p5_L1_b0.70_g1.00_s0.30` = **0.6748113272421348** | **✓ 逐位** |
| R124 §④「`bias=1.30, σ=0.10` 基准臂 `(e)`=**2.5970**」 | 2.597024 | `p5_L3_C2_b1.30_s0.10` = **2.5970241288261766** | **✓ 逐位** |
| **§3ec = R139 ⑥（L3 定案，`docs/m11_acceptance_master.md` L4034）**「`(e)` 在**所有** 5 格都 `pass`（**1.5769–2.9400**）」 | 1.5769 / 2.9400 | 改后 5 格 = **1.5768572 / 1.8499679 / 2.1617790 / 2.5970241 / 2.9399795** | **✓ 区间逐位吻合**（旧版为 1.5809 + 3.9728 / 3.6183 / 3.9295 / 4.1122 ⇒ 4 格 `fail`，**与登记矛盾**）|
| R124 §⑤「`g1_v5` 的 `p5_B1..B4` `(e)`=**1.8635** `pass`；`repl_v6` 的 `p5_C` = **1.8499/1.8545/1.8530**」 | 1.8635 / 1.8499·1.8545·1.8530 | 改后 `g1_star` = **1.8635211**；`g1_repl_cd` = **1.8499679 / 1.8545163 / 1.8530114** | **✓ 改前=改后（数值不变）** ⇒ 这些登记值在**新口径下依然成立** |
| R200 §①「reporter 基准臂口径 `1.230976 / 1.51615 / 1.379773 / 1.378299`」 | 见左 | L5 改后 = **1.2309759 / 1.5161500 / 1.3797731 / 1.3782987** | **✓** |

**⇒ R124 §⑤ 的待办（"G1 的 `(e)` 结论需按新口径重算"）在"格齐备"的产物上已闭合**：
新口径值 = 报告中已登记的基准臂值。（注意：`g1_v5`/`repl_v6` 的**产物文件本身**由**更早的工具/判据表修订版**生成，
用冻结版复算有 479 / 123 处**既有**差异（`(b)` 带、`(g)`、阶梯等，与本任务无关）⇒ **这些产物的"整体可复现性"是既存问题，不是本次引入**；
我的上述比对是"旧版 vs 新版同一条命令"的对照，以及"改后值 vs 登记值"的核对。）

---

## §8 回归风险声明（哪些既往结论**可能**受影响）

**结论**：本修改**只影响两类输出** —— ① `(e)` 的取值/判定与其判据级聚合；② 门条件（`(c)/(e)/张力`）的**聚合作用域**。
除此之外**无影响**（§5/§6 的逐字段对照：`arm_metrics`/`estimator` 语义/阈值/带/其余 8 条判据逐字不变）。

逐项评估已登记结论的影响面：

| 既往结论类型 | 是否受影响 | 证据 |
|---|---|---|
| R124 §④（基准臂 `(e)` 数值） | **不受影响** | 改后逐位命中（§7）|
| §3ec = R139 ⑥（L3 定案：`(e)` 5 格 `pass`，1.5769–2.9400） | **不受影响；且由"矛盾"变为"一致"** | 旧版 4 格 `fail` 与登记矛盾；改后逐位吻合 ✓ |
| R124 §⑤（`g1_v5`/`repl_v6` 的 `(e)`） | **不受影响（数值不变）** | `g1_star`/`g1_repl_cd` 旧=新（仅 `reason` 追加口径标注）✓ |
| R200 §① 的 4 条门裁决（除 `(e)`） | **不受影响** | `(a)`/`(b)`/`(c)`/`tension` 的**行状态**在 L5 全同（CSV 差异仅 5 个 `(e)` 行）✓ |
| R200 §② 反平凡性（`frac_nonzero`/`triviality`） | **不受影响** | 该结论来自 reporter，且本工具未动 `_eval_engineering` 与逐臂度量的任何字段 ✓ |
| R199/R201/R202/R204（钙域口径锚、重跑自证） | **不受影响** | 与评估器 `(e)`/门聚合无交集；未运行仿真、未用 MPS ✓ |
| **`(e)` 旧值被记为 `fail` 的产物**（如 R200 §③1 引用的 4.50844 等） | **受影响（这是修复目标）** | 旧值为**混合口径伪影**（R124 已登记）⇒ 改后为基准臂口径；**R200 §① 早已裁定该 `fail` 不可用作裁决** ⇒ 无结论被推翻 ✓ |
| **门条件曾被缺臂格拉成 `unavailable` 的产物**（L5） | **受影响（这是修复目标）** | 改后 `c/e/tension = pass`；主判仍为 reporter ✓ |
| **"格不齐备"的早期冒烟产物**（`e_twin`、`p0_n1000`、`p0_n31000`、`p0_n8192`） | ⚠ **受影响，需登记** | 这些协议**无 `role=baseline` 臂** ⇒ 齐备格 = 0 ⇒ 门条件由"混合口径的 `pass`/`fail`"变为 **`unavailable`**。其中 `e_twin`、`p0_n1000` 的 `verdict_G1` 由 **`fail` → `unavailable`**；`p0_n31000`/`p0_n8192` 的总判种类**不变**（`unavailable`），仅 `reason` 文本变。**核查**：`docs/` 中**没有**任何结论引用这 4 个协议的评估器总判／门条件（`grep -rn "e_twin" docs/` 无命中；`p0_*` 仅出现在 `m11_acceptance_master.md` L1122/L1126 关于 `engine_interpreters` 的**生产者**事实、`m11_b2_notes.md` L734 的命令示例，均与评估器裁决无关）⇒ **未发现任何已登记结论被推翻**；`data/m11_p5_resting_eval.json`（默认产物，协议=`p0_n31000`）的 `gate_conditions['p5_e_ei_balance']` 由 `pass` 变 `unavailable`（总判仍 `unavailable`）—— 该产物用冻结版复算本就有 41 处**既有**差异（工具/判据表修订漂移，与本任务无关），且 `docs/` 仅把它当用法示例引用（`m11_b3_g1_design.md` L487–488/L611）⇒ **登记为"门条件下调为未测"，不构成结论推翻**。 |

**⇒ 最终口径**：**除 A、B 两处及其直接后果之外，其余输出逐字不变**（10 个协议 × 逐字段对照为证）；
A 的后果是"`(e)` 由混合口径伪影值 → 登记的基准臂口径值"（§7 逐位命中登记值），
B 的后果是"门条件不再被缺臂格污染"（并已显式打印排除项）。
**唯一需要主 agent 留意的"新"变化**是：**格全部不齐备的产物其门条件会由 pass/fail 退为 `unavailable`**（未测 ≠ 未命中），已在 §6/§8 逐条列出。

---

## §9 版本、sha、备份与变更说明位置

| 项 | 值 |
|---|---|
| **新文件** | `/Users/weidong/ai/small_world/neural_exploration/tools/validate_p11_resting.py`，**1810 行** |
| **新 sha256** | **`a9c2756ecf63c61f9e4c14822a591c6d6fdc012bae0eeb3600ce4a2ab44ed223`** |
| **原文备份** | `/Users/weidong/ai/small_world/neural_exploration/tools/_archive/validate_p11_resting.0d3c1d990dfe.py`，**1649 行** |
| **原文 sha256** | `0d3c1d990dfe707720ec316b57ae2bb699181dae0d87042c920638ed9a4c48f3`（与 `cmp` 逐字节相同 ✓）|
| **变更说明位置** | 新文件**头部 L6–L31**（`**文件版本与变更说明（v2；本文件原为冻结登记版）**` 块）：写明"原版 sha / 备份路径 / 依据（R200 §③、R124、`m12_targets.csv::rule_e_baseline_arm_only`）/ 缺陷 A、B 的修法与副作用 / **未改动项清单**" |
| **本文件** | `neural_exploration/docs/m11_evaluator_fix_A_B.md` |
| diff 规模 | `diff -u` 共 **10 个 hunk**、**+171 / −10 行**（其中 26 行是头部变更说明，正文改动集中在 §1.2/§2.2 列出的 6 处）|

### 9.1 最小改动 diff（归档原文 → 新版，全文）

```diff
--- _archive/validate_p11_resting.0d3c1d990dfe.py	2026-10-13 12:49:45
+++ validate_p11_resting.py	2026-10-15 08:43:42
@@ -2,6 +2,33 @@
 
 《生物仿真M11实施清单》§4.5（P5 判据 (a)–(i)）+ §0.7（判据可测性预注册）+ §7 P5 行
 + 《docs/m11_list_review_master.md》RF-M11-3 + 《docs/m11_acceptance_master.md》A4/A8/A12。
+
+================================================================================
+**文件版本与变更说明（v2；本文件原为冻结登记版）**
+================================================================================
+  原版 sha256：`0d3c1d990dfe707720ec316b57ae2bb699181dae0d87042c920638ed9a4c48f3`
+  原文备份（逐字节保留）：`neural_exploration/tools/_archive/validate_p11_resting.0d3c1d990dfe.py`
+  变更依据：`docs/m11_acceptance_master.md` **§3gl = R200 §③**（工具级两处成因定位）＋ **§3do = R124**
+            （(e) 口径缺陷：(e) 必须只在基准臂上判）＋ `data/m12_targets.csv::rule_e_baseline_arm_only`。
+  **本版只修下面两处缺陷，其余一律不动**（前后逐字段对照见 `docs/m11_evaluator_fix_A_B.md`）：
+
+  - **缺陷 A —— `(e)` 的口径伪影（R124／R200 §③1）**：原 `(e)` 取「**该格全部臂**（baseline+zero_drive
+    +dose）的 `ei_ratio` 中位数」⇒ 零驱动臂（`i_exc→0` ⇒ `ei_ratio`≈7.49–8.00）混入后把中位数抬到
+    ≈4.51 ⇒ 判 `fail`（**口径伪影，非生理结论**）。
+    **改为**：`(e)` 只取 **`role=="baseline"` 且 `drive_mult==1.0`** 的臂（口径 `baseline_arm_only`）；
+    所用的臂与被排除的臂（含原因）写入该行 `reason` 及输出 JSON 的 `ei_scope` 字段（**可核对、不静默丢弃**）。
+    ⇒ 副作用（应当如实登记）：只有 `dose` 臂的格不再有 `(e)` 值 ⇒ 该行 `unavailable`（此前是混合口径值）。
+  - **缺陷 B —— 门条件被「缺臂的辅助格」污染（R200 §③2）**：门条件原读「**全部格**的判据级聚合状态」
+    ⇒ 只有 `dose` 臂的辅助格（缺 `baseline`/`zero_drive`）把 `p5_c_selfsustain` 拉成 `unavailable`
+    （而同一 run 的 4 个实格 `(c)` 全部 `pass`）。
+    **改为**：门条件只由「**格条件齐备**（同时提供 `baseline` 与 `zero_drive` 的臂）的格」贡献；
+    被排除的格与其原因登记进输出 JSON 的 `gate_aggregation_scope` 并由 `print_summary` **显式打印**。
+    原 `criterion_status`（**全量**行的判据级状态）**保持原语义不变** ⇒ 任何格都不会被静默丢弃，
+    门聚合另给 `gate_criterion_status`（新键）。
+  - **未改动项（硬约束）**：判据 id／阈值／带来源（仍只读 `data/m11_p5_targets.csv`）、`estimator` 语义、
+    其余判据（a)(b)(c)(d)(f)(g)(h)(i) 与时长阶梯/张力/不变量）的口径、CSV 列集合、
+    硬门①②③④⑤⑥ 的判定规则（含硬门⑤「`mechanism=="none"` 的基线臂」这一既有语义）**均未改动**。
+================================================================================
 
 **为什么需要这个工具（主agent 对 M10 的裁定，本工具的三类堵漏对象）**
   1. **门 1（自持比）不是统计可判**：M10 同配置 3 seed = 0.0987/0.1056/0.1005、合并 R=0.1015、
@@ -540,6 +567,10 @@
         self.gate_composition: Dict[str, Any] = {}
         self.seed_effectiveness_check: str = "not_run"
         self.guard_failures: List[str] = []
+        # 【v2/R200-A】(e) 口径作用域登记（逐格：所用基准臂 + 被排除臂及其原因）
+        self.ei_scope: Dict[str, Any] = {}
+        # 【v2/R200-B】门条件聚合作用域登记（齐备格/被排除格及其原因）
+        self.gate_aggregation_scope: Dict[str, Any] = {}
 
     # ---------------- 协议前缀（口径字段，逐行携带） ----------------
     def _proto_fields(self, arm: Optional[Arm], T_ms: Optional[float] = None) -> Dict[str, Any]:
@@ -632,6 +663,8 @@
         self._apply_duration_flip()
         # 门组成与判据表声明一致性（防"代码与清单漂移"）
         self.gate_composition = self._assert_gate_composition()
+        # 【v2/R200-B】门条件的聚合作用域（齐备格/被排除格，含原因）——必须在 _verdict 之前算
+        self.gate_aggregation_scope = self._gate_aggregation_scope()
         # 硬断言（违反即 raise）：口径标注 / 未定稿带不得判 pass / 子判据不得静默丢失
         self.invariants = self._assert_output_invariants(self.rows)
         return self._verdict()
@@ -656,6 +689,71 @@
             out = [a for a in out if a.role == role]
         return out
 
+    # ---- 评估单元键（与 _tension 的键格式一致）与 (e) 的口径作用域（R200-A） ----
+    @staticmethod
+    def _unit_key_str(u: Tuple[str, float, str]) -> str:
+        return f"{u[0]}@{u[1]:g}ms|mech={u[2]}"
+
+    @staticmethod
+    def _ei_scope_arms(arms: Sequence[Arm]) -> Tuple[List[Arm], List[Dict[str, Any]]]:
+        """**(e) 的口径 = `baseline_arm_only`（R124 / R200 §③1 / 缺陷 A）**。
+
+        只取 `role=="baseline"` **且** `drive_mult==1.0` 的臂（既有的基准臂约定）；
+        返回 `(所用的臂, 被排除的臂[{arm_id, role, drive_mult, reason}])` —— **被排除者必须可核对**，
+        不得静默丢弃。**禁止**混入 `zero_drive`（`i_exc→0` ⇒ 比值爆炸）与 `dose` 臂。
+        """
+        used: List[Arm] = []
+        excluded: List[Dict[str, Any]] = []
+        for a in arms:
+            if a.role != "baseline":
+                excluded.append({"arm_id": a.arm_id, "role": a.role, "drive_mult": a.drive_mult,
+                                 "reason": f"role={a.role}：非基准臂 ⇒ (e) 不得混入（R124/R200-A）"})
+            elif abs(a.drive_mult - 1.0) >= 1e-9:
+                excluded.append({"arm_id": a.arm_id, "role": a.role, "drive_mult": a.drive_mult,
+                                 "reason": f"drive_mult={a.drive_mult} ≠ 1.0：非基准臂约定 ⇒ (e) 不得混入"})
+            else:
+                used.append(a)
+        return used, excluded
+
+    def _gate_aggregation_scope(self) -> Dict[str, Any]:
+        """**门条件的聚合作用域（R200 §③2 / 缺陷 B）**。
+
+        门条件（(c) ∧ (e) ∧ 张力已定量）只能由「**格条件齐备**」的格贡献；
+        **齐备 = 该 `(config_id, T_ms, mechanism)` 格同时提供 `role=="baseline"` 与 `role=="zero_drive"`
+        的臂**（这正是 (c)（零驱动/基准群率比）与 (e)（基准臂口径）的**作用域前提**）。
+        缺角色的格（例如只有 `dose` 臂的辅助格）：其行**照常输出**（不静默丢弃，`criterion_status` 仍按全量算），
+        但**不得**把门条件拉成 `unavailable`/`fail`；被排除的格与其原因在此显式登记并由 `print_summary` 打印。
+        """
+        inv: Dict[Tuple[str, float, str], set] = {}
+        for a in self.arms:
+            inv.setdefault((a.config_id, a.T_ms, a.mechanism), set()).add(a.role)
+        keys = sorted({(r["config_id"], r["T_ms"], r["mechanism"]) for r in self.rows},
+                      key=lambda t: (t[0], t[1], t[2]))
+        used, excluded = [], []
+        for k in keys:
+            roles = inv.get(k, set())
+            missing = [r for r in ("baseline", "zero_drive") if r not in roles]
+            rec = {"unit": self._unit_key_str(k), "config_id": k[0], "T_ms": k[1], "mechanism": k[2],
+                   "roles_present": sorted(roles), "missing_roles": missing}
+            if not roles:
+                rec["reason"] = ("no_arms_with_this_unit_key（该格在臂清单中不存在 ⇒ 其角色齐备性不可判）"
+                                 "⇒ 不参与门条件")
+                excluded.append(rec)
+            elif missing:
+                rec["reason"] = (f"unit_incomplete(缺角色={missing})：该格的行照常输出，但门条件的作用域前提"
+                                 f"（同时提供 baseline 与 zero_drive）不满足 ⇒ 不得贡献门条件（R200 §③2）")
+                excluded.append(rec)
+            else:
+                used.append(self._unit_key_str(k))
+        return {"rule": ("门条件只由**格条件齐备**的格贡献：齐备 = 该 (config_id,T_ms,mechanism) 格"
+                         "同时提供 role=baseline 与 role=zero_drive 的臂；缺角色的格其行照常输出"
+                         "（criterion_status 仍按全量算）但不参与门条件的判据级聚合（R200 §③2 / 缺陷 B）"),
+                "unit_key_format": "config_id@T_ms|mech=<mechanism>",
+                "units_used": used, "units_excluded": excluded,
+                "n_units_used": len(used), "n_units_excluded": len(excluded),
+                "note": ("被排除的格**不静默丢弃**：其逐行四态仍在 rows/CSV 中原样保留，"
+                         "只是不参与 gate_conditions；对照 `gate_criterion_status` 与 `criterion_status` 可定位差异。")}
+
     # ---- 基线臂（无机制）解析（硬门⑤）----
     def _baseline_arm_value(self, cfg: str, T_ms: float, criterion: str, estimator: str) -> Dict[str, Any]:
         """找「无机制基线臂」在同一 (config,T) 下的同一判据值；找不到即 unavailable。"""
@@ -953,18 +1051,40 @@
 
     def _eval_ei(self, cfg: str, T_ms: float, mech: str, arms: List[Arm], n_neuron: Optional[int]) -> None:
         lo, hi, bstat, breasons = self.tt.band("ei_ratio_lo", "ei_ratio_hi")
-        vals = [self.arm_metrics[a.arm_id].get("ei_ratio") for a in arms]
+        # 【v2/R200-A（= R124）】**(e) 只在基准臂上判**：口径 = baseline_arm_only。
+        #   原实现取「该格全部臂的中位数」⇒ 零驱动臂（i_exc→0 ⇒ ei_ratio≈7.5–8.0）混入 ⇒ 伪影 fail。
+        scope_arms, scope_excluded = self._ei_scope_arms(arms)
+        scope_txt = ("；口径=baseline_arm_only（只取 role=baseline 且 drive_mult=1.0 的臂；R124/R200 §③1）"
+                     f"；所用臂={[a.arm_id for a in scope_arms]}"
+                     + (f"；排除臂={[[x['arm_id'], x['role'], x['drive_mult']] for x in scope_excluded]}"
+                        if scope_excluded else "；排除臂=[]"))
+        vals = [self.arm_metrics[a.arm_id].get("ei_ratio") for a in scope_arms]
         vals = [v for v in vals if v is not None]
         val = float(np.median(vals)) if vals else None
+        self.ei_scope[self._unit_key_str((cfg, T_ms, mech))] = {
+            "scope": "baseline_arm_only", "definition": "role==baseline 且 drive_mult==1.0",
+            "basis": "docs/m11_acceptance_master.md R124（§3do）+ R200 §③1（§3gl）；"
+                     "data/m12_targets.csv::rule_e_baseline_arm_only（M12 侧同名规则）",
+            "arms_used": [a.arm_id for a in scope_arms],
+            "arms_excluded": scope_excluded,
+            "ei_ratio_median": val, "band": [lo, hi], "band_status": bstat,
+            "estimator": "synaptic_current_integral_ratio",
+        }
         bl = self._baseline_arm_value(cfg, T_ms, CRIT["e"], "per_neuron")
-        if not arms:
-            status, reason = "unavailable", "arm_missing"
+        if not scope_arms:
+            if arms:
+                status, reason = "unavailable", ("baseline_arm_missing(该格未提供 role=baseline 且 "
+                                                 "drive_mult=1.0 的臂 ⇒ 口径 baseline_arm_only 不可判"
+                                                 "；本工具不从其它角色代判而只能判 unavailable)" + scope_txt)
+            else:
+                status, reason = "unavailable", ("arm_missing（该格无任何臂）" + scope_txt)
         elif not vals:
             status, reason = "unavailable", ("ei_input_unavailable: 臂内未注入 i_exc_integral/i_inh_integral"
-                                            "（本工具不从发放率反推电流；(e) 的唯一来源是仿真侧注入）")
+                                             "（本工具不从发放率反推电流；(e) 的唯一来源是仿真侧注入）" + scope_txt)
         else:
             status, reason = self._judge(val, lo, hi, bstat, breasons, True,
-                                        extra="定义=i_inh_integral/i_exc_integral（判据表 ei_ratio_definition）")
+                                         extra="定义=i_inh_integral/i_exc_integral（判据表 ei_ratio_definition）"
+                                               + scope_txt)
         self._emit(config_id=cfg, T_ms=T_ms, mechanism=mech, criterion=CRIT["e"],
                    estimator="synaptic_current_integral_ratio",
                    value=val, unit="", status=status, reason=reason, band_lo=lo, band_hi=hi,
@@ -972,7 +1092,7 @@
                     discriminative=(None if not bl["available"] else (not _meets(bl["value"], lo, hi))),
                     baseline_value=bl.get("value"), baseline_meets=(None if not bl["available"] else _meets(bl["value"], lo, hi)),
                     note="数值端点为预注册设计选择（derived）；四篇文献仅支撑方向与量级（判据表 ei_ratio_* 行）",
-                   arm=(arms[0] if arms else None))
+                   arm=(scope_arms[0] if scope_arms else (arms[0] if arms else None)))
 
     def _eval_engineering(self, cfg: str, T_ms: float, mech: str, arms: List[Arm],
                           n_neuron: Optional[int]) -> None:
@@ -1326,8 +1446,30 @@
                 h[r["status"]] = h.get(r["status"], 0) + 1
             crit_rows[c] = h
         gate = {}
-        for c in GATE_CRITERIA:
-            gate[c] = crit_status.get(c, "unavailable")
+        # 【v2/R200-B（缺陷 B 修复）】门条件**只由「格条件齐备」的格贡献**：缺 role=baseline/zero_drive 的
+        #   辅助格（例如只有 dose 臂的格）不再把门条件拉成 unavailable。**聚合规则与上面逐字相同**
+        #   （最坏优先），只是行集合限定为齐备格 ⇒ 差异可逐条归因。
+        #   注意：上面的 `crit_status`（⇒ 输出键 `criterion_status`）**仍按全量行**计算，语义不变
+        #   ⇒ 被排除的格**不静默丢弃**（其四态仍在 criterion_status / rows / CSV 里）。
+        scope = self.gate_aggregation_scope or self._gate_aggregation_scope()
+        gate_units = set(scope["units_used"])
+        gate_by_crit: Dict[str, List[Dict[str, Any]]] = {}
+        for r in self.rows:
+            if self._unit_key_str((r["config_id"], r["T_ms"], r["mechanism"])) in gate_units:
+                gate_by_crit.setdefault(r["criterion"], []).append(r)
+        gate_status: Dict[str, str] = {}
+        for c, rs in gate_by_crit.items():
+            prim = [r for r in rs if r["is_primary_estimator"] == 1] or rs
+            sts = {r["status"] for r in prim}
+            if "invalid" in sts:
+                gate_status[c] = "invalid"
+            elif sts == {"pass"}:
+                gate_status[c] = "pass"
+            elif "fail" in sts:
+                gate_status[c] = "fail"
+            else:
+                gate_status[c] = "unavailable"
+        gate = {c: gate_status.get(c, "unavailable") for c in GATE_CRITERIA}
         # 判别性摘要（硬门⑤）
         non_disc = sorted({r["criterion"] for r in self.rows
                            if r["discriminative"] is False or r["evidence_class"].startswith("sanity_precondition")})
@@ -1363,10 +1505,17 @@
                                        f"（{len(self.estimator_gate_unmet)} 个臂）⇒ 不得判 pass")
         counts = {s: sum(1 for r in self.rows if r["status"] == s) for s in STATES}
         return {"verdict_G1": g1, "verdict_reason": why, "gate_conditions": gate,
+                # 【v2/R200-B 新增键】门聚合的判据级状态（只含齐备格）与作用域登记（含被排除格及原因）
+                "gate_criterion_status": gate_status,
+                "gate_aggregation_scope": scope,
                 "criterion_status": crit_status, "state_counts": counts,
                 "criterion_row_status_histogram": crit_rows,
                 "aggregation_rule": ("判据级状态 = 该判据**所有行**的最坏优先聚合（invalid > fail > pass > unavailable）；"
-                                     "门条件读判据级状态 ⇒ 行状态与门条件可以不同（见 verdict_G1 处的规则声明）"),
+                                     "门条件读判据级状态 ⇒ 行状态与门条件可以不同（见 verdict_G1 处的规则声明）"
+                                     "【v2/R200-B 补写（只追加，原句保留）】门条件**不读**全量判据级状态，"
+                                     "而只读「**格条件齐备**（同时提供 baseline 与 zero_drive 臂）的格」的行聚合"
+                                     "（见 `gate_aggregation_scope` / `gate_criterion_status`）；缺角色的格其行照常输出"
+                                     "（`criterion_status` 仍按全量算）但不参与门条件。"),
                 "hard_failures": self.hard_failures,
                 "guard_failures": self.guard_failures,
                 "discipline_45_rule": ("「通过」必须能回答「它本来可能失败吗」：若某判据在该配置下**不可能失败**"
@@ -1416,6 +1565,16 @@
     print(f"M11 P5 / G1 评估：**{out['verdict_G1']}**  exit={out['exit_code']}")
     print(f"  reason: {out['verdict_reason']}")
     print(f"  门条件: {out['gate_conditions']}")
+    # 【v2/R200-B】**不得静默丢弃**：门条件的作用域 + 被排除的格及其原因必须打印出来（可核对）
+    _scope = out.get("gate_aggregation_scope") or {}
+    if _scope:
+        print(f"  门条件聚合作用域（R200-B）：齐备格 {_scope.get('n_units_used')} 个 → {_scope.get('units_used')}")
+        _ex = _scope.get("units_excluded") or []
+        if _ex:
+            print(f"  ⚠ 被排除的格 {_scope.get('n_units_excluded')} 个（其行仍原样在 CSV/rows 中；仅不参与门条件）：")
+            for _r in _ex:
+                print(f"    - {_r['unit']}  roles_present={_r['roles_present']} "
+                      f"missing_roles={_r['missing_roles']} ⇒ {_r['reason']}")
     print(f"  四态计数: {out['state_counts']}")
     if out["hard_failures"]:
         print("  硬失败：")
@@ -1625,6 +1784,8 @@
                   "n_neuron": a.n_neuron, "pop_trace": a.pop_trace_path, "counts": a.counts_path,
                   "n_steps_present": (None if a.trace is None else int(a.trace.size))} for a in ev.arms],
         "arm_metrics": ev.arm_metrics,
+        # 【v2/R200-A 新增键】(e) 的口径作用域登记：逐格的所用基准臂 ID 与被排除臂（含原因）——可核对
+        "ei_scope": ev.ei_scope,
         "output_invariants": ev.invariants,
         "gate_composition": ev.gate_composition,
         "seed_effectiveness_check": ev.seed_effectiveness_check,
```

---

## §10 与既有裁决/登记可能冲突之处（登记缺口）

1. **R124 §⑦ 的建议尚未落地**：`data/m11_p5_targets.csv` 中**没有** `ei_ratio_definition` 行的
   `scope=baseline_arm_only` 补写（`grep -rn "baseline_arm_only"` 在 M11 判据表 **无命中**；该表的
   `ei_ratio_definition` 行 L58 的 `wrong_usage_consequence` 只写了"权重比 ≠ 电流积分比"，
   **没有**写"把零驱动臂并入 ⇒ 比值爆炸 ⇒ 判 fail 是伪影"这层）。
   ⇒ **本次未改该表**（任务禁令），故**新口径的依据落在**：`docs/m11_acceptance_master.md`（R124 §④/R200 §③1）
   ＋ `data/m12_targets.csv::rule_e_baseline_arm_only`（L48，M12 侧同名规则）＋ 工具头部 L6–L31 的显式登记。
   **建议主 agent 追加该行**（append-only）以消除"M11 判据表未登记作用域"的缺口。
   ⇒ **未发现与既有裁决冲突**：R124 §④ 的"`(e)` 只在基准臂上判"是本次实现的口径；R200 §③ 要求的两处修正
   已逐条落实。
2. **R120/§③ 的"评估器判不了 `mechanism="lever_*"`"**（硬门⑤：无 `mechanism=="none"` 基线臂）
   **仍然成立**，故改后总判仍 `unavailable`；本任务**不去动它**（R203 §④ 已登记同一判断：不该为让总判变 `pass` 而放宽硬门⑤）。
3. **既有"产物不可复现"现象（非本次引入）**：`data/m11_p5_resting_eval*.json`（含默认产物与 `g1_v5`/`repl_v6`）
   用冻结原版按**当前**协议/判据表复算即有 41 / 479 / 123 处差异（`discipline_45_rule`、`guard_failures`、
   `(b)` 的带、阶梯、张力结构等）⇒ 这些产物由**更早的工具/判据表修订**生成。**本次未改它们**，也**未**据其复现结论。
4. **R203（主 agent 独立复核）**：已用**改后**版本在 `/tmp` 冒烟并逐条确认 A ✅ / B ✅ / 残余 `unavailable` = 硬门⑤；
   本文件的 §1.3/§2.3 数值与其登记值一致（`1.23098 / 1.51615 / 1.37977 / 1.3783`；门条件三项 `pass`）。
   ⇒ **未发现冲突**；本文件补上了 R203 §⑤ 要求的"逐字段变/不变清单 + 回归风险声明 + 交付文档"。

---

## §11 本节点**没有**做的事（明确声明）

- **未运行任何仿真**（评估器是**只读分析**工具；所有 `--out-*` 写入 `/tmp/m11_p5_eval_fixAB/**`）；
- **未使用 MPS/GPU**（未启动任何生产/重跑作业，未调用 `run_m11_p5_arms.py` 等任何生产者；未 `pgrep`／未查看 GPU）；
- **未联网**（无 `web_search`、无 `curl`/`pip`/下载）；
- **除 `tools/validate_p11_resting.py` 外未修改任何已存在文件**（`git status --porcelain` 的已跟踪项**仅 1 条**
  `M neural_exploration/tools/validate_p11_resting.py`；`tools/_archive/` 为**新建未跟踪**目录）；
  `report_m11_l3.py` sha 仍 `69a8a127…`、`data/m11_p5_targets.csv` sha 仍 `0991808242fd…`、`data/m12_targets.csv`／
  `docs/m11_acceptance_master.md`／任何 `src/*.py` **未改**；
- **未覆盖任何已登记产物**：`data/m11_p5_l5_resting_eval.csv`/`.json` 的 sha 仍为
  `b443f4ee5bb1457d3bb3439d9004155036a6788d36dfa16ef0157351ff7ed177` /
  `df625fe320fed7615cef516f081010d0e4bb07a829ebeb86b06d52e88e0192ed`（与改前一致）✓
- **未 `git add` / 未 `git commit`**；
- **未改判据表、未改阈值、未放宽任何判据**（含**未**放宽硬门⑤ 以"消除"残余 `unavailable`）。
