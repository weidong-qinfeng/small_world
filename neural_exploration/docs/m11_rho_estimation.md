# M11 · `rho = thr / A` 的数据侧估计（`(b)` 决断地图的输入标量）

**状态**：`§0 预注册`（先写、后算）→ `§1`–`§5`（自检后逐节填入）。
**执行节点**：本文件由执行节点产出，**只新建**本文件 + `data/m11_rho_estimation/rho_estimate.{json,csv}`；**未修改任何已存在文件**。
**红线声明**：本次**未运行任何仿真**、**未触碰 MPS/GPU**、**未联网**、**未下载文献全文**、**未 `git add`/`git commit`**。

## 背景（本文件的前提，逐条给出处）

| 事实 | 值 | 出处 |
|---|---|---|
| `(b)` 的判据带 | `[0.101852, 0.229885]` | `data/m11_p5_targets.csv` L115 / L116（`silent_frac_calcium_band_lo/_hi`） |
| 判 `(b)` 的跨口径错误 | 钙事件检测口径的带 ⇒ 套到脉冲计数口径统计量 | `docs/m11_acceptance_master.md` L5273（R183） |
| 带有第二重不可比 | 带测自 **317.46 s** 窗，仿真协议窗 **10 s** | `docs/m11_acceptance_master.md` L5594–L5633（R189） |
| 决断地图（顺序统计量） | `frac(c_i<K)` 单调穿过带：`K=4` 低于下界、`K=5` 在带内、`K=6` 高于上界 | `docs/m11_acceptance_master.md` L5658–L5668（R190 §① 表） |
| 两侧严格区 | `rho<=1` ⇒ 静默 `=frac(c_i=0)=0.000122`（严格 fail）；`rho>5` ⇒ `>=frac(c_i<6)=0.453003`（严格 fail） | 同上 L5684–L5691 |
| 唯一可能 pass 的窗 | `rho ∈ (0.92472, 5]`；其中 `(4,5]` 余量仅 **3.25%** | 同上 L5691 + L5762（R191 §⑥） |
| 本任务立论 | `rho` **不必跨口径**：`thr` 与 `A` 可在**同一 dF/F 约定**内取比 | 同上 L5778–L5788（R192 §②） |

**桥接假设（本文件的核心前提，必须显式登记）**：若口径桥（`docs/m11_caliber_bridge_checklist.md`，R191 裁决②）把仿真 dF/F 按数据侧统计量标定（`dff_median = 0.423285`、`dff_mad = 0.385062`，出处 `data/m12_b5_roi_mask/m12_b5_roi_mask_audit.json::fields.field4.calcium_event_caliber`），则 **`rho_sim ≈ rho_data`** ⇒ 可用 `rho_data` 去查 R190 的决断地图。
**⇒ 该假设是**假设**，不是本节点能验证的事实**；本文件只报 `rho_data` 的事实与算术。

---

# §0 预注册（**冻结于看结果之前**）

> **冻结时刻**：见 `data/m11_rho_estimation/rho_estimate.json::meta.prereg_frozen_at_utc`；本节的 sha256 同时记入该 JSON 的 `meta.prereg_doc_sha256_at_freeze`（该 sha 是**本节冻结时**、§1–§5 尚为空模板时的整文件哈希 ⇒ 可证「预注册先于计算」）。
> **纪律**：本节以下各项**在跑任何计算之前**写完；**严禁**事后调参/换定义去迁就结论（尤其禁止去凑 `rho ∈ (4,5]`）。任何必要改动一律**追加**到 §0.8「方案变更登记」并注明**在看到哪一步结果之前/之后改的**。

## §0.0 口径（**完全照用，不得自创**）

出处：`data/m12_b5_roi_mask/m12_b5_roi_mask_audit.json::fields.field4.calcium_event_caliber.rule`
= 「**逐像素 F0=20 分位；dF/F=(F−F0)/F0；事件阈值 = 中位 + 3·MAD**（承 M11 J2 同源）」。

- **实现同款性**：`thr`、`dF/F`、真实 ROI 轨迹、事件计数**全部直接调用**仓内既有实现
  `neural_exploration/tools/m12_b5_roi_mask_audit.py` 的 `build_label_volume` / `load_movie` /
  `dff_and_threshold` / `real_roi_analysis`（**只读 import，不改该文件**）⇒ 与 B-5/审计同款。
- **观测窗**：2000 帧 @ **6.300006 Hz**（帧间隔 0.15873 s，出处 `data/m11_reference.csv` R-01/R-02，L3/L4）。
- **单元**：**真实 ROI 掩膜**（`field_seg_masks` → `compressed_segmentation` 解码卷），
  field4 `n_rois_field_total = 1388`、**`n_rois_fully_in_movie_window = 432`**；ROI 轨迹 = 该 ROI **完全落窗**像素的 **等权均值**（`Mask.weights` 未使用 —— 已登记限制）。
  **🔴 不得退回 proxy（20×20 分箱）口径。**
- **事件**：帧级 `trace > thr` 记一次事件（与 `real_roi_analysis` L376 同款，即"越阈帧数"口径）。
- **已知真值（自检基准）**：field4 全窗静默比例 `0.101852 = 44/432`、`event_threshold_dff = 1.578470`、`median_events_per_roi = 46.5`。

## §0.1 `A` 的定义（择一并给理由）

**主估计器（PRIMARY，冻结）**
> `A_prim` := **孤立事件峰幅值分布的 `p10` 分位**，其中
> **峰幅值 `= (越阈连续段内 trace 的最大值) − (该段之前的局部静默基线)`**。

**理由（三条，均为口径理由，与结果无关）**

1. **同约定性**：`thr` 是**池化 dF/F** 上的全局标量，故 `A` 必须在**同一 dF/F 约定**（逐像素 F0=20 分位 ⇒ dF/F ⇒ ROI 等权均值）内测量 ⇒ 二者之比 `rho = thr/A` 是同一约定内两量之比（R192 §② 的立论）。
2. **"减基线"是必需的，不是可选的**：R190 的 `rho` 定理前提是「荧光核**非负**且 `max h = 1`」⇒ 仿真侧的 `A` 是**单脉冲核在自身零基线之上的峰幅**。而数据侧 ROI 轨迹的**池化中位 dF/F = 0.423285 ≠ 0**（每像素各减自己 F0 ⇒ ROI 均值仍有**非零偏移**）⇒ 若用**绝对 trace 值**当 `A`，就把「ROI 自身的基线偏移」混进了「单脉冲等效幅度」，**与仿真侧 `A` 不是同一个量**。故主档取「**峰 − 局部静默基线**」。
3. **低分位而非中位/众数**：观测到的峰幅分布是**脉冲数 `n=1,2,3,…` 的混合**（峰 ≤ `n·A₁`，R190 的包含关系方向），**只有其下缘**才可能是「单脉冲等效」；取 `median`/`mode` 会给出**多脉冲等效幅度**（系统性偏大的 `A`）⇒ 把 `rho` 系统性压小。故**低分位为唯一有理由的主档**，中位/众数**并列作敏感性档**（§0.5），不当作主档。

**🔴 结构性风险（冻结、先登记，见 §5 局限）**：凡在 **`thr` 上检出**的事件，其**绝对峰必 > `thr`**（检出定义使然）⇒
- 「绝对峰值」档（`abs`）**可证**恒有 `A_abs > thr` ⇒ `rho_abs < 1` **恒成立** ⇒ **该档对决断地图退化**（是口径伪影，不是关于 `rho_sim` 的证据）；
- 主档（减基线）**不退化**，但其截断水平变为 `thr − 基线偏移(ROI)`（逐 ROI 不同、可 < `thr`），仍是**单侧偏大**的估计 ⇒ **`A_est >= A₁`（单脉冲幅度）近似成立 ⇒ `rho_est <= rho_sim`（弱下界性质）**。
  ⇒ **事先登记推论**：本路线若得到 `rho_est < 1`，**不得**据此断言 `rho_sim < 1`（下界落在严格 fail 区内**不含信息**）。此点**先写下**，不得事后发现。

## §0.2 「孤立事件」判据（冻结）

- **段（run）定义**：`trace > thr` 的**极大连续帧段**（可按 `(trace > thr).sum()` 逐帧复算）。
- **孤立判据**：设段 `k` 的起止为 `[s_k, e_k]`。令**间隔帧数** `gap_prev = s_k − e_{k−1} − 1`、`gap_next = s_{k+1} − e_k − 1`。
  `k` 孤立 ⟺ `gap_prev >= G` **且** `gap_next >= G`，`G := round(2·τ·6.300006)`（即"与其它事件在时间上相隔 **≥ 2τ**"）。
- **窗边界规则（主档）**：**窗边界视为"干净边界"**（首段的 `prev` 侧、末段的 `next` 侧自动满足）；**敏感性档**另测"要求距窗边 ≥ `G` 帧"的严格版（§0.5）。
- **`τ` 取值与出处**：
  - **主档 `τ = 1.0 s`（⇒ `G = 13` 帧）**，出处 `docs/m11_acceptance_master.md` **L5727**（口径桥主档 B1 写死 `τ=1.0 s`，L5745 同）——这是**仓内唯一以「钙响应核」身份登记的 `τ`**（同值亦为 M10 工程默认）。
  - ⚠️ **仓内没有「单神经元单脉冲钙衰减 `τ`」的登记值** ⇒ `τ = 1.0 s` **标为「仅工程假设」**（出处是口径桥的工程档，**不是**钙指示剂的实测动力学）。
  - 仓内**实测**的 `tau_native_unsmoothed = 8.003–9.015 s`（`data/m11_p5_targets.csv` **L90** `microns_same_source_tau_band_ref`、**L107** `microns_calcium_measured_population_anchors`）与 `29.68/30.48 s`（**L111** `microns_same_source_tau_band_smoothed_ref`）**口径是「群体状态 dwell/自相关时间」，属 S 门**，**不是**单脉冲钙衰减 ⇒ **只作扫描档，不作主档**。
  - **`τ` 扫描档**：`{0.5, 1.0, 2.0, 3.0, 5.0, 8.0} s` ⇒ `G = {6, 13, 25, 38, 63, 101}` 帧（§0.5）。

## §0.3 事件峰幅值的定义（冻结）

- **峰** = 该段内 `trace` 的**最大原始值**（单帧分辨率，**不插值、不平滑**）。
- **基线扣法（四档，主档第一）**：
  1. **`bl_local`（主档）**：基线 = 段起点前 **`G` 帧**窗口内 `trace` 的**中位数**（该窗口因 §0.2 的间隔条件**保证不含越阈帧** ⇒ 是"静默水平"）；若可用帧数 < 3 ⇒ 该事件不纳入（计数并登记）。
  2. `pre5`：基线 = 前 **5 帧**中位数（与 `τ` 解耦的短窗）。
  3. `pctl0`：基线 = 该 ROI **全窗** `trace` 的 **20 分位**（把「逐像素 F0=20 分位」升到 ROI 层）。
  4. `abs`：**不扣**（= 绝对峰；**已知退化**，只作对照）。
- **注意**：以上都是**同一 `thr` 检出**之下测的幅度；本档**不改变检出阈值**（改变 `thr` 就换了口径，见 §0.7 未测项）。

## §0.4 子窗与单元集合（冻结）

- **主档**：**全 2000 帧（317.46 s）× 全部 432 个完全落窗 ROI**（与带自身的口径 `n_rois_fully_in_movie_window = 432` 一致）。
- **敏感性档**：
  - 单元集合：全部 432 / `area_px>=10` / `area_px>=20`（后两者按 `real_roi_analysis` 的既有列）；
  - 时间子窗：**非重叠 63 帧块（= 10.0 s，= 仿真协议窗 `T=10 s`）** 全 31 块；**前后两个 1000 帧半窗**；
  - 子窗内**沿用同一全局 `thr`**（承 R189/R191 的"同一 `thr` 算子窗 ⇒ 嵌套单调"口径）。

## §0.5 敏感性档清单（**冻结；不论结果好坏全部列出**）

| 轴 | 档位 | 主档 |
|---|---|---|
| 幅度统计量 `stat` | `p05, p10, p25, median, mode(KDE), mean` | **`p10`** |
| 峰定义 `peak_def` | `bl_local, pre5, pctl0, abs` | **`bl_local`** |
| 隔离 `τ`(s) / `G`(帧) | `0.5/6, 1.0/13, 2.0/25, 3.0/38, 5.0/63, 8.0/101` | **`1.0/13`** |
| 单元集合 | `432_all, area>=10, area>=20` | **`432_all`** |
| 窗口 | `full2000, half_first1000, half_second1000, 31×block63` | **`full2000`** |
| 聚合 | `pooled`（跨 ROI 池化后取分位）/ `per_roi_median`（逐 ROI 取分位后跨 ROI 中位） | **`pooled`** |
| 边界规则 | `edge_clean` / `edge_margin_G` | **`edge_clean`** |
| 最小样本门 | 该档孤立事件数 `n_iso < 30` ⇒ 该档标 **`unavailable`**（**不得**当 0、**不得**写成"未命中"） | **30** |

**区间来源（三条并列，来源不同不得混称概率区间）**：
1. **统计区间**：对 **ROI** 做有放回自助（`B = 1000`，`seed = 20261015`），逐次重算主档统计量 ⇒ 取 2.5%/97.5% 分位；
2. **跨子窗离散度**：31×63 帧块（及两个半窗）上主档统计量的分布范围；
3. **定义性区间**：**全部敏感性档**（`n_iso>=30`）上的 **min–max** —— 这是**定义不确定度**，**不是**概率区间，须显式标注。

## §0.6 预测与判定标准（**先写，再看结果**）

**预测（写下时刻：未见任何本次结果）**

1. `A_abs`（绝对峰）**必然** > `thr = 1.578470` ⇒ `rho_abs < 1` **恒成立**（这是检出定义的同义反复，不是发现）。
2. `A_prim`（`bl_local`, `p10`）预期落在 **`0.6–1.4`**，理由：数据侧 ROI 轨迹的基线偏移量级可由**池化中位 `0.423285`** 量度；若典型基线偏移 ≈ `+0.4`、最小孤立事件绝对峰 ≈ `+1.6–2.0`（仅略高于 `thr`），则上升量 ≈ `1.2–1.6`；取 `p10` 会偏低 ⇒ `0.6–1.4` 是合理预期带。
3. ⇒ 预期 `rho_prim = thr/A_prim` ≈ **`1.1–2.6`**，即落在决断地图 **`(0.92472, 4]`（未定区）中部**。
4. **预期总敏感性区间会跨越决断地图边界**：因为 `A` 的定义选择本身在 `~0.4–1.7` 间摆动（`0.4` 来自"基线偏移较大的 ROI 的上升量"，`1.7` 来自"绝对值档"）⇒ `rho` 在 `~0.9–4` 间摆动 ⇒ **预期结论很可能是「不能判」**。
5. 预期**没有任何档位给出 `rho > 5`**（要 `rho > 5` 需 `A < 0.315694`，即最小孤立事件上升量 < `0.82×MAD`；先验偏低，但**不得事先排除**）。

**"成功产出 `rho` 区间"的判定标准（冻结）**
- **前置**：§1 自检 **3/3 全部命中**（`0.101852 = 44/432`、`thr = 1.578470`、`median_events_per_roi = 46.5`）；任一不命中 ⇒ **停下报告，不产出 `rho`**（标 `invalid`）。
- **产出**：主估计器给出 `A` 的**点估计 + 区间**（区间并列上述三条来源）；`rho = thr/A` 的**点估计 + 区间**；四区映射；**边界跨越显式声明**（跨了 ⇒ 明确写"不能判"）；**全部敏感性档列表**（§0.5 全轴；`n_iso<30` 者标 `unavailable`）。
- **四态用词纪律**：`pass`/`fail`/`invalid`/`unavailable`；**未测 ≠ 未命中**；**已测 ≠ 可判定**；本文件**不替主 agent 下 `(b)` 的裁决**，只报事实与算术。

## §0.7 本档**未测**的维度（先声明，防止把"未测"读成"已测"）

1. **真值脉冲不可得**：MICrONS 真值脉冲 only-in-DB（R1/R2/R3）⇒ 本节点**无法**用真值脉冲标定 `A`；`A` 是**事件幅度口径**的估计，**不是**"单 spike 的 dF/F"的实测。
2. **未改变检出阈值**：`thr` 固定为 `1.578470`。**低于 `thr` 的事件在本口径下不可观测** ⇒ 主档 `A` 是**条件于被检出**的幅度（§0.1 已登记该截断；**另设**"降阈辅助档"见 §0.5 之外的补充说明：若主档给出 `A_prim > thr` 且 `rho_prim < 1`，则**额外**跑一档 `thr_aux = thr/2` 的"降阈探针"，**只作补充诊断、不作主档**，因为改变 `thr` 即改变口径、与带不同源）。
3. **神经域反演未做**（未反演的 dF/F；同 B-5 限制）。
4. **掩膜 ↔ 数据库 `Mask` 表身份链未确认**、`Mask.weights` 未使用（等权均值）。
5. **`rho_sim ≈ rho_data` 是桥接假设**，本节点**未**验证（不跑仿真）。

## §0.8 方案变更登记（**初始为空**）

| # | 时刻 | 阶段（看到什么结果之前/之后） | 改了什么 | 为什么 |
|---|---|---|---|---|
| 1 | 2026-10-14（主档结果 `rho = 3.2847`、`Z1` **之后**） | **之后**（明确：在看到主档结果之后） | **追加 1 条轴**：`POSTHOC_D1a`（低/高基线半区 ROI）、`POSTHOC_D1b`（限定"事件前基线未抬升"的事件）、`POSTHOC_perroi`（逐 ROI `p10` 的跨 ROI 分布）+ 相关系数/噪声尺度诊断 ⇒ 共 7 档，写入 CSV 的 `POSTHOC_*` 行与 JSON 的 `post_hoc_diagnostics` | 主档 `A = 0.4805` **低于** §0.6 预测带（0.6–1.4）⇒ 需要查清「低尾是否由固定全局阈值的几何效应主导」。**该轴只作诊断：不进决断地图、不改主档、不参与交付区间的 headline** |
| — | — | — | **主档与 §0.5 全部预注册档：一字未改**（无"为凑 `(4,5]` 而调参"） | — |

> **§0.0–§0.7 的估计器定义、孤立判据、峰定义、子窗与单元集合、敏感性档、预测与判定标准，
> 在冻结（`2026-10-14T18:17:08Z`，`sha256 = d68a4afe…1660`）之后逐字未变**
> ⇒ 本表唯一新增 = 1 条**未预注册的后验诊断轴**（上表第 1 行）。

### 🔒 冻结完整性**自证**（可复算；已被 `rho_estimate.json::meta.prereg_freeze_integrity` 收录）

**方法（不许只靠自述）**：用**当前文件回推**冻结版 —— 把 §0.8 的登记行换回初始占位行、把 §1–§5 换回占位块 ——
得 `reconstructed` 文本，其 `sha256` 与冻结记录**逐位相同**：

```
reconstructed frozen-file sha256 = d68a4afe0ab9698eb6a54c4fa50cdbd778700a8191887c3bd4be85547fb01660
recorded   frozen-file sha256    = d68a4afe0ab9698eb6a54c4fa50cdbd778700a8191887c3bd4be85547fb01660
MATCH = True
```

⇒ **证明：当前文件与冻结版只差两处** —— ① §0.8 的登记行；② §1–§5 的填充（二者**都**属于 §0 自身规定
"冻结后填入"的部分）⇒ **§0.0–§0.7（估计器与全部判据）逐字未变**，即"**先预注册、后计算**"这条纪律
在**文件层**成立，不靠自述。
- §0.0–§0.7 文本块 `sha256 = e5a0cb9f131d8b59d802e1c512138cc6417bc0a7c38783c3e8543e2bc3bf6e3d`（14159 bytes），供后续交接比对；
- 回推所需的冻结尾部逐字存于 `/tmp/m11_rho_est/frozen_tail.txt`（其 sha 记入同一 JSON 字段）。

---

# §1 实现自检（必做，可信性前提）

## 1.1 结论

> **3/3 全部命中（4/4 检查项 HIT）⇒ 实现与 B-5/审计同款，自检通过，可继续测 `A`。**
> field2 同款自检亦 3/3（`0.229885` / `thr 2.308268` / `median_events 8.0`）。

**"同款"的证据不是"我照着写了一遍"，而是"我直接调用同一份代码"**：`A` 估计的全部前置量
（ROI 掩膜解码卷、钙电影载入、dF/F 与 `thr`、真实 ROI 轨迹与事件计数）**逐函数 import** 仓内既有实现
`tools/m12_b5_roi_mask_audit.py` 的 `build_label_volume` / `load_movie` / `dff_and_threshold` /
`real_roi_analysis`（**只读 import，未修改该文件**；`git status` 无 tracked 改动 ⇒ 见 §1.3）。

## 1.2 命令与输出（逐字）

```
$ cd /Users/weidong/ai/small_world && PYTHONHASHSEED=0 MPLBACKEND=Agg \
    /Users/weidong/ai/small_world/.venv-m11/bin/python /tmp/m11_rho_est/selfcheck.py
movie shape (x,y,z,t) = (200, 200, 1, 2000)   n_files_chunks = 8
decode 0.2s  load 0.6s  dff+analysis 3.3s
dff_median = 0.423285   dff_mad = 0.385062   event_threshold_dff = 1.578470
n_rois_field_total = 1388   n_rois_fully_in_movie_window = 432
n_zero_events = 44 / 432
silent_frac_zero_count_real_roi = 0.101852
median_events_per_roi = 46.5
----------------------------------------------------------------------
  silent_frac == 0.101852    HIT
  silent_frac == 44/432      HIT
  thr == 1.578470            HIT
  median_events == 46.5      HIT
SELFCHECK 4/4 = True
```

（该脚本全文见**附录 A.2**；自检脚本自带完整流程，`decode 0.2s` 指 8 个 `field_seg_masks` chunk 的
`compressed_segmentation` 解码耗时。）

## 1.3 三条真值逐项对照

| 真值 | 期望（出处） | 实测 | 命中 |
|---|---|---|---|
| 钙域静默比例 | **`0.101852`**（`data/m11_p5_targets.csv` L115；B-5 审计 JSON） | `0.101852`，且 `n_zero_events = 44 / 432`（**恰好整除**，与 `44/432 = 0.1018518…` 一致） | ✅ |
| `event_threshold_dff` | **`1.578470`**（审计 JSON `fields.field4.calcium_event_caliber`） | `1.5784704089164734` | ✅ |
| `median_events_per_roi` | **`46.5`**（审计 JSON `fields.field4.real_roi_silence`） | `46.5` | ✅ |

**⇒ 静默比例与事件计数口径逐位复现 ⇒ §2 的 `A` 是在与判据带**同源同款**的事件检测口径上测的**（这正是 R192 §② 立论所需的"同一约定"）。

**可核对的旁证**：`dff_median = 0.423285`、`dff_mad = 0.385062` 亦与审计 JSON 逐值一致
⇒ 连 `thr` 的构造中间量都对上了（不只是末值）。

**纪律核对**：本节的"通过"是**可失败的**（三项各自独立可 MISS；`thr` 只差 5e-7 即判 MISS），
不是"必然通过"的构造。

**红线遵守**：本节点**未**运行任何仿真、**未**触碰 MPS/GPU、**未**联网、**未**修改任何既有文件（`git status` 无 tracked 改动；新增仅本文件 + `data/m11_rho_estimation/` 两件）✓

---

# §2 `A` 与 `rho`

## 2.1 主档（§0.1–§0.4 冻结的估计器）

**定义（逐字取自 §0）**：`A` = **孤立事件峰幅值分布的 `p10`**；峰幅值 =（越阈连续段内 ROI 轨迹最大值）
−（段起点前 `G` 帧**静默窗**的中位数）；孤立 = 前后间隔均 `>= G = round(2·τ·6.300006)` 帧，`τ = 1.0 s` ⇒ `G = 13`；
单元 = 全部 **432** 个完全落窗真实 ROI；窗 = **全 2000 帧**；聚合 = 跨 ROI 池化；窗边界视为干净边界。

| 量 | 值 |
|---|---|
| `thr`（同一 dF/F 约定，已在手） | **1.5784704089164734** |
| 越阈帧数（= `(b)` 口径的"事件"计数） | 38580 |
| 越阈**连续段**数 `n_runs` | 7355（平均段长 5.244 帧 = **0.833 s**，与 `τ=1.0 s` 同量级） |
| **孤立事件样本 `n_iso`** | **1005**（原始 1010，扣除 5 个静默窗 <3 帧者） |
| 贡献孤立事件的 ROI | 337 / 432 |
| **`A`（点估计）** | **0.480548** |
| `A / thr` | 0.3044（**< 1** ⇒ 主档**未**落入 §0.1 登记的截断退化情形） |
| **`rho = thr / A`（点估计）** | **3.284731** |
| **ROI 自助 95% 区间**（`B=1000`, `seed=20261015`） | `A ∈ [0.450747, 0.501036]` ⇒ **`rho ∈ [3.150413, 3.501902]`** |
| 落区 | **`Z1_undetermined(0.92472 < rho <= 4)`** |

## 2.2 区间的四条来源（**来源不同，不得混称概率区间**）

| # | 来源 | `A` 区间 | **`rho` 区间** | 性质 |
|---|---|---|---|---|
| ① | **统计**：ROI 有放回自助 95%（`B=1000`） | `[0.450747, 0.501036]` | **`[3.1504, 3.5019]`** | 概率区间（仅覆盖**ROI 抽样**变异） |
| ② | **跨子窗离散度**：31×63 帧块（=10.0 s，合格 14/31） | `[0.327078, 0.774972]` | **`[2.0368, 4.8260]`** | 离散度（时间非平稳） |
| ③ | **定义不确定度**：全部预注册敏感性档 min–max | `[0.359603, 1.601506]` | **`[0.98562, 4.38948]`** | **定义不确定度，非概率区间** |
| ④ | ③ ∪ 主档 ∪ 全部 63 帧块 | `[0.327078, 1.601506]` | **`[0.98562, 4.8260]`** | 同上（**本任务的"全敏感性区间"**） |
| ⑤ | **后验诊断**（**未预注册**，§0.8）：剔除"事件前基线已抬升"的事件 | `[1.132536, 1.306854]` | **`[1.20784, 1.39375]`** | 诊断性；**不进决断地图** |

**⇒ 本任务交付的 `rho` 区间 = ④：`rho ∈ [0.98562, 4.82598]`**（跨区，见 §4）。

## 2.3 与 §0.6 预测的对照（**必须如实报告偏差，含未命中项**）

| 预测（§0.6，先写） | 实测 | 命中？ |
|---|---|---|
| ① `A_abs > thr` 恒成立 ⇒ `rho_abs < 1` | `abs` 档 `A = 1.601506`；全体孤立事件**绝对峰 min = 1.578701**，`min/thr = 1.00015` | ✅ **命中** |
| ② `A_prim ∈ 0.6–1.4` | **0.480548** | ❌ **未命中（低于预测带）** |
| ③ `rho_prim ∈ 1.1–2.6` | **3.2847** | ❌ **未命中（高于预测带）**；但**落区 Z1 命中** |
| ④ 总敏感性区间跨决断地图边界 | 跨 `4.0`（Z1↔Z2） | ✅ **命中** |
| ⑤ 无任何档位 `rho > 5` | field4：max **4.8260** ✅；**field2 旁证 `pre5` 档 5.2459 > 5** | ⚠️ **field4 命中、field2 未命中** |

**⇒ 预测 ②③ 的偏差方向有意义**：`A` 比预期**更小**、`rho` 比预期**更大**。原因由 §2.4 的后验诊断给出
（**不是**事后改估计器，主档一字未动）。

## 2.4 后验诊断（🔴 **未预注册**：看到主档结果后追加，**仅诊断、不改主档**）

| 诊断量 | 值 | 读数 |
|---|---|---|
| 事件前**局部静默基线**的分位（中位 / p75） | 0.880 / 1.071 | 检出时刻的基线**普遍抬升**（远高于 ROI 层 pctl0 中位 0.1676） |
| 相关系数（事件前基线 vs 上升量） | Pearson **−0.581**，Spearman **−0.763** | **强负相关** ⇒ 上升量 ≈ `thr − 基线`（**固定全局阈值的几何效应**） |
| 上升量分布（p05 / p10 / p25 / p50） | 0.398 / 0.481 / 0.655 / 0.945 | 主档 `A = p10` |
| 上升量 ÷ 池化 MAD(0.385062) | p10 = **1.25×**、p50 = 2.45× | 低尾仅 ~1.25×MAD ⇒ **贴近噪声/阈值几何** |
| 上升量 < 1×MAD / < 2×MAD 的比例 | **3.98%** / **35.92%** | 35.9% 的孤立事件上升量不到 2×MAD |
| D1a：低基线半区 ROI（216 个，603 事件） | `A = 0.520423` ⇒ `rho = 3.0331`（Z1） | 与主档同区 |
| D1a：高基线半区 ROI（216 个，402 事件） | `A = 0.440432` ⇒ `rho = 3.5839`（Z1） | 与主档同区 |
| D1b：剔除基线抬升的事件（`≤ pctl0+0.5·MAD`，76 事件） | `A = 1.306854` ⇒ **`rho = 1.2078`**（Z1） | **`A` 变大、`rho` 变小** |
| D1b：`≤ pctl0+1.0·MAD`（173 事件） | `A = 1.132536` ⇒ **`rho = 1.3938`**（Z1） | 同上 |
| 逐 ROI `p10` 的跨 ROI 分布（337 个有孤立事件的 ROI） | p05 / p50 / p95 = `0.381491` / `0.686355` / `1.454013` | ⇒ `rho` 从 **4.1376**（Z2）到 **1.0856**（Z1） |

**⇒ 读数（关键）**：主档 `A` 的**下尾**有相当一部分由「**检出瞬间基线恰高 ⇒ 上升量小**」的
**固定全局阈值几何**贡献，而**不是**生理的单脉冲幅度。**剔除该效应后 `rho` 反而更小**（1.21–1.39，Z1 中部）
⇒ **`(4,5]` 内的档位主要来自最受该几何效应污染的档**（`pre5` / `τ=0.5` / 部分 63 帧块）。

---

# §3 敏感性全表

**全部逐档结果见 `data/m11_rho_estimation/rho_estimate.csv`（115 行 = 1 行表头注释 + 1 行列名 + 113 档）**；
其中 **49 档标 `unavailable`**（`n_iso < 30`，按 §0.5 最小样本门；**"未测 ≠ 未命中"**）。下表为**主表**（轴线档 + 汇总）。

## 3.1 field4（**主口径**，432 ROI，全 2000 帧）

| 轴 | 档 | `n_iso` | `A` | `rho` | 落区 |
|---|---|---|---|---|---|
| **主档** | `bl_local` / `p10` / `τ=1.0` | 1005 | **0.480548** | **3.28473** | **Z1** |
| stat | `p05` | 1005 | 0.398022 | 3.96578 | Z1 |
| stat | `p25` | 1005 | 0.654754 | 2.41078 | Z1 |
| stat | `median` | 1005 | 0.944741 | 1.67080 | Z1 |
| stat | `mode`(KDE) | 1005 | 0.763380 | 2.06773 | Z1 |
| stat | `mean` | 1005 | 1.112961 | 1.41826 | Z1 |
| peak_def | **`pre5`** | 1005 | 0.369816 | **4.26826** | **Z2** |
| peak_def | `pctl0` | 1010 | 1.394653 | 1.13180 | Z1 |
| peak_def | `abs`（**退化**） | 1010 | 1.601506 | **0.98562** | Z1（见 §4.4） |
| τ(s) | **0.5** (`G=6`) | 1514 | 0.359603 | **4.38948** | **Z2** |
| τ(s) | 2.0 (`G=25`) | 708 | 0.614889 | 2.56708 | Z1 |
| τ(s) | 3.0 (`G=38`) | 555 | 0.701553 | 2.24996 | Z1 |
| τ(s) | 5.0 (`G=63`) | 409 | 0.805956 | 1.95851 | Z1 |
| τ(s) | 8.0 (`G=101`) | 299 | 0.873886 | 1.80626 | Z1 |
| units | `area>=10` (428) | 990 | 0.477054 | 3.30879 | Z1 |
| units | `area>=20` (364) | 817 | 0.473761 | 3.33180 | Z1 |
| agg | `per_roi_median` | 1005 | 0.686355 | 2.29978 | Z1 |
| edge | `edge_margin_G` | 990 | 0.482098 | 3.27416 | Z1 |
| window | `half_first1000` | 495 | 0.452163 | 3.49093 | Z1 |
| window | `half_second1000` | 516 | 0.498954 | 3.16356 | Z1 |
| window | **31×block63 汇总**（合格 14/31 的中位） | 1185 | 0.438492 | 3.59977 | Z1 |
| window | 63 帧块的**最坏/最好** | — | `0.327078` / `0.774972` | **4.82597** / 2.03681 | Z2 / Z1 |

**逐块**（`SENS_block63`，31 行在 CSV）：**4 块落 Z2**（`block63_02`=4.33943、`_03`=**4.82597**、`_14`=4.17556、`_21`=4.08034），
**10 块落 Z1**，**17 块 `unavailable`**（`n_iso` 11–29，最小样本门）。

## 3.2 field2（**旁证**；两条已登记弱点见下）

| 轴 | 档 | `n_iso` | `A` | `rho` | 落区 |
|---|---|---|---|---|---|
| 主档 | `bl_local`/`p10`/`τ=1.0` | 135 | 0.647464 | 3.56509 | Z1（自助 95%：`A∈[0.55725,0.72704]` ⇒ `rho∈[3.1746,4.1426]`） |
| stat | `p05` | 135 | 0.544077 | **4.24254** | **Z2** |
| peak_def | **`pre5`** | 135 | 0.440015 | **5.24589** | **Z3** |
| τ(s) | **0.5** | 216 | 0.513234 | **4.49750** | **Z2** |
| 其余（`p25/median/mode/mean`、`pctl0`、`abs`、`τ∈{2,3,5,8}`、`agg`、`edge`、两个半窗） | — | 38–135 | 0.610381–2.331717 | **0.98994**–3.78169 | Z1（`abs` 档 `rho=0.98994` 已贴近 `0.92472` 下缘） |
| window | 31×block63 | 0–25 | — | — | **全部 `unavailable`**（87 ROI 撑不起 10 s 窗） |

> 🔴 **field2 的两条已登记弱点（必须随行）**：
> ① **配准锚仅部分通过** —— 审计 JSON `fields.field2.semantic_anchor.verdict = "FAIL／部分不通过 —— 见 per_metric（不得当作已配准）"`
> （rank 1/25、领先 0.9%；另两指标 3/25、6/25）⇒ **不得**当作"已配准"；
> ② **其带（上界 `0.229885`）只建立在 87 个 ROI 上**（field4 为 432）。
> ⇒ field2 的一切数字**只作并列旁证**，不作主口径、不得单独支撑结论。

## 3.3 后验诊断档（**未预注册**，仅诊断）

见 §2.4 表 + CSV 的 `POSTHOC_D1a` / `POSTHOC_D1b` / `POSTHOC_perroi` 行（7 档，全部 `n_iso>=30`）。

## 3.4 稳定性判定（按 §0.6 预注册纪律）

> **不稳定。** 全敏感性区间 `rho ∈ [0.98562, 4.82598]`（field4）**跨越 `4.0` 边界**（Z1↔Z2）；
> 且 field2 旁证的 `pre5` 档 **5.24589 > 5** ⇒ **"能落进 `(4,5]`，也能落出去（甚至落进 `>5`）"**
> ⇒ 按预注册纪律**必须标为不稳定/不能判**（见 §4.3）。

---

# §4 决断地图落区

## 4.1 地图（照用 R190/R191，**不自创**）

| 区 | 规则 | 含义（出处 `docs/m11_acceptance_master.md`） |
|---|---|---|
| **Z0** | `rho <= 0.92472` | **严格 fail**：钙域静默 `= frac(c_i=0) = 0.000122 < 带下界 0.101852`（L5686；精界 0.92472 见 L5762） |
| **Z1** | `0.92472 < rho <= 4` | **未定**：严格下界 `<= frac(c_i<4) = 0.076294 < 0.101852`（要落带须靠**时刻结构**抬升）（L5663/L5687） |
| **Z2** | `4 < rho <= 5` | **唯一可能 pass 的窄窗**：下界 `frac(c_i<5) = 0.222412` 已在带内，余量 **3.25%**（L5664/L5688） |
| **Z3** | `rho > 5` | **严格 fail**：静默 `>= frac(c_i<6) = 0.453003 > 带上界 0.229885`（L5665/L5689） |

## 4.2 主档与统计区间

- **主档：`rho = 3.284731` ⇒ 落 `Z1`（未定区）。**
- **ROI 自助 95%：`rho ∈ [3.150413, 3.501902]` ⇒ 全部落在 `Z1` 内，`crosses_boundary = false`。**
  ⇒ **统计不确定度（ROI 抽样）不足以跨出 Z1**；跨界与否**完全由定义选择决定**（见 §4.3）。

## 4.3 全敏感性区间（本任务的交付区间）

**`rho ∈ [0.98562, 4.82598]`（field4；含主档、全部预注册轴、全部 63 帧块）**

- **区间覆盖 `Z1`（35 档）与 `Z2`（7 档）⇒ ✅ 跨越 `4.0` 边界（`crosses_boundary = true`）。**
- **⇒ 按 §0.6 预注册纪律，必须明确写：`(b)` 的落区在此区间下「不能判」。**
  即：**本区间既不能排除 Z1（未定/需时刻模型），也不能排除 Z2（唯一可能 pass 的窄窗）**。
- **Z1 是区间的主体**（35/42 个可判定档）；`Z2` 的 7 档为：`pre5`（4.26826）、`τ=0.5`（4.38948）、
  63 帧块 `_02/_03/_14/_21`（4.33943 / **4.82597** / 4.17556 / 4.08034）。
  而 §2.4 的后验诊断显示：`rho` 偏大（⇒ 落 Z2）的档**恰是最受"近阈几何效应"污染的档**
  （`pre5`、小 `τ`），剔除该效应后 `rho` 反而 = **1.2078–1.3938**（Z1 中部）。

## 4.4 两侧严格区的证据状态（**逐侧如实报**）

| 区 | field4 是否有档位进入 | 事实与算术 |
|---|---|---|
| **Z0（`<=0.92472`，严格 fail）** | **无**（最接近者 = 退化档 `abs` 的 `0.985616`，仍在 Z1） | ① 若改用 **R190 的粗界 `rho <= 1`**（而非 R191 精界 0.92472），则 `abs` 档**落入严格 fail 区**；② 但 `abs` 档 `rho<1` **由检出定义恒成立**：全体孤立事件**绝对峰 min = 1.578701 > thr = 1.578470**（`min/thr = 1.00015`）⇒ **该档不构成关于 `rho_sim` 的证据**（§0.1 已预先登记）⇒ **Z0 目前无有效证据** |
| **Z2（`(4,5]`，唯一可能 pass）** | **有**（7 档） | 区间跨越到 Z2 ⇒ **不能排除**；但进入 Z2 的档即 §4.3 所述"几何效应污染最重"者 |
| **Z3（`>5`，严格 fail）** | **无**（field4 max = **4.82598**） | ⚠️ **"无档位进入"≠"已排除"**：§0.1 已预先登记 `A_est >= A₁ ⇒ rho_est <= rho_sim`（弱下界性质）⇒ **本路线不可能排除 Z3**。另：field2 旁证的 `pre5` 档 **5.24589 > 5** 已进入 Z3（但 field2 有 §3.2 的两条已登记弱点） |

## 4.5 只报事实与算术（**不替主 agent 下 `(b)` 的裁决**）

1. **事实 A**：在**桥接假设**（R192：`rho_sim ≈ rho_data`）下，数据侧主档 `rho = 3.2847` 落在 **Z1（未定区）**。
   Z1 的**严格下界** `frac(c_i<4) = 0.076294 < 0.101852` ⇒ **仅凭"`rho` 落 Z1"这一个事实不足以判 `(b)`**：
   Z1 的判定**需要时刻结构模型**（R190 §② 表），**本节点未建该模型、也未跑仿真**。
2. **事实 B**：交付区间 `[0.98562, 4.82598]` **跨越 `4.0`** ⇒ 按预注册纪律 ⇒ **"不能判"**（不得写成 pass，也不得写成 fail）。
3. **事实 C**：跨界的**来源是定义选择**（`peak_def` / `τ` / `stat`），**不是** ROI 抽样（自助 95% 全在 Z1 内，
   宽度仅 ±5.4%）；即"把 `A` 定义得更严（`pre5`/小 `τ`）⇒ `rho` 进 Z2；把 `A` 定义为绝对峰 ⇒ `rho` 抵 Z1 下缘"。
4. **事实 D**：`rho` 的**点估计**在 5 个预注册主轴（`stat`、`τ`、`units`、`agg`、`edge`、半窗）上的**主档一致性高**：
   `τ` 从 0.5→8.0 s 时 `rho` 单调 **4.389→1.806**；`stat` 从 p05→mean 时 `rho` 单调 **3.966→1.418**
   ⇒ **"`rho` 的量级 = 1–4"这一结论比**具体落区**稳健**。
5. **事实 E**：**不判定四态中的任何一个**——本节点只提供 `rho` 的估计与区间。
   （若必须用四态词描述**本节点自身的产出**：`rho` 的估计 = `measured`；
   `(b)` 的判定 = 本节点**未测/不下判**，**不得**读成"未命中"。）

---

# §5 局限与未核实项

**（每条都给出方向或量级，不写"可能有影响"这类空话）**

1. **真值脉冲不可得**（only-in-DB，R1/R2/R3）⇒ 无法用真值脉冲标定 `A`。
   `A` 是**事件幅度口径**的量，**不是**"单 spike 的 dF/F"的实测。
2. **检出截断（censoring）—— 本路线最强的一条限制**：凡在 `thr` 上检出者，其**绝对峰必 > `thr`**
   （实测 min = 1.578701，`min/thr = 1.00015`）⇒ 绝对峰档 `rho < 1` **恒成立**（口径伪影）；
   主档（减基线）不退化，但仍是**单侧偏大**的 `A` 估计 ⇒ **`rho_est <= rho_sim`（弱下界性质）**。
   ⇒ **本路线给出的小 `rho` 不能用来排除 `rho_sim > 5`**；也不能把本区间读作 `rho_sim` 的区间。
3. **固定全局阈值的几何效应**（§2.4 后验诊断）：事件前局部基线中位 `0.880`、与上升量 Spearman **−0.763**
   ⇒ 主档 `A` 的下尾部分是**阈几何**（上升量 ≈ `thr − 基线`）而非生理单脉冲幅度；
   剔除该效应后 `rho = 1.208–1.394`（Z1 中部）⇒ 落 Z2 的档位是**最受该效应污染**的档。
4. **`τ` 无仓内实测依据**：主档 `τ = 1.0 s` 出处是**口径桥的工程档**（`docs/m11_acceptance_master.md` L5727/L5745），
   **不是**钙指示剂动力学；仓内实测 `8.003–9.015 s`（`data/m11_p5_targets.csv` L90/L107）与 `29.68/30.48 s`（L111）
   口径是**群体状态 dwell**（属 S 门）⇒ 只作扫描档。
   **量级敏感性：`τ` 一次变动即把 `rho` 从 4.389（0.5 s）带到 1.806（8.0 s）—— 这是本区间宽度的主因之一。**
5. **未改变检出阈值** ⇒ 低于 `thr` 的事件在本口径下**不可观测**；主档 `A` **条件于被检出**。
   §0.7 预注册的 `thr/2` 降阈探针因**触发条件未满足**（主档 `A_prim = 0.4805 < thr`）**按预注册未跑**（未被当作结果）。
6. **神经域反演未做**（未反演的 dF/F，同 B-5 限制）。
7. **掩膜 ↔ 数据库 `Mask` 表身份链未确认**；`Mask.weights` **未使用** ⇒ 本口径是**等权均值**，与 CaImAn 官方**加权口径不同**（同 B-5 限制）。
8. **样本范围**：单 field（`session4_scan7_field4`）× 2000 帧 = **317.46 s**、**432** 个完全落窗 ROI
   ⇒ 外推范围受限（`data/m11_p5_targets.csv` L113 的过度泛化纪律）；且带本身是 317.5 s 窗的量（R189），
   本文件的一切数值**同样**是 317.5 s 窗的量 ⇒ **"10 s 窗"的比较仍未解决**（本文件不解决 R189）。
9. **桥接假设未验证**：`rho_sim ≈ rho_data` 依赖口径桥按 R191 裁决②（按数据侧 `median 0.423285`/`MAD 0.385062` 标定噪声）
   成立；本节点**未跑仿真、未验证该假设** ⇒ `rho_data` 与 `rho_sim` 的关系仍是**假设**。
10. **10 s 子窗档的样本门**：field4 有 **17/31** 个 63 帧块 `n_iso < 30`、field2 **31/31** 全部 `< 30`
    ⇒ 这些档标 **`unavailable`**（**未测 ≠ 未命中**；**不得**当作 0、**不得**读成"该窗无孤立事件"）。
11. **后验诊断轴未预注册**（§0.8 已登记）：`POSTHOC_*` 7 档与 D1a/D1b 的一切数字**只作诊断**，
    **不进**决断地图、**不参与**交付区间的 headline，主档与全部预注册档**一字未改**。
12. **未核实项（逐条声明"未读到/未做"）**：真值脉冲（未读到，only-in-DB）；`Mask.weights`（未读到）；
    掩膜身份链（未核实）；钙指示剂单脉冲 dF/F 文献值（**未测**，R192 已登记文献路径失败）；
    桥的标定实现（未读其最终冻结档，本文件只引用 R191 的裁决文本）。

## 5.1 四态用词纪律自查

- 本文件对 `(b)` **不下裁决**（四态词不用于 `(b)`）；对 `rho` 的估计标 `measured`；
- `unavailable`（49 档）**仅**表示"该档样本不足、未测"，**不是**"未命中"；
- 未把"未测"写成"未命中"，未把"已测"写成"可判定"（Z1 即 `已测但不可判定`）。

---

# 附录 A：驱动脚本逐字备份（**非交付要求；因"只新建 3 件"的红线，仓内不留 tools/*.py，故附于此**）

**为什么这样处理**：任务红线要求**只新建**本文件 + 2 件机读产物 ⇒ 无法在 `tools/` 新建脚本。
为保住**可复现性**（本项目的通行纪律），把全部驱动脚本**逐字**附在本附录；
审计者把 A.1–A.4 存成 4 个文件即可**逐值重跑**（`sha256` 记录在 `rho_estimate.json::meta`）。

- **A.1 `rho_est.py`**（主流程；`--field field4|field2`；`main()` 里的 `OUTDIR` 默认 `/tmp/m11_rho_est/per_field`）
- **A.2 `selfcheck.py`**（§1 自检；**只读、不写任何文件**）
- **A.3 `posthoc.py`**（**未预注册**的后验诊断；只写 `/tmp/m11_rho_est/posthoc.json`）
- **A.4 `compose.py`**（把两轮结果 + 后验诊断合成 2 件交付物）

**运行命令（逐字，全部绝对路径）**

```bash
# §1 自检
cd /Users/weidong/ai/small_world && PYTHONHASHSEED=0 MPLBACKEND=Agg \
  /Users/weidong/ai/small_world/.venv-m11/bin/python /tmp/m11_rho_est/selfcheck.py
# §2/§3 主口径 + 旁证
cd /Users/weidong/ai/small_world && PYTHONHASHSEED=0 MPLBACKEND=Agg \
  /Users/weidong/ai/small_world/.venv-m11/bin/python /tmp/m11_rho_est/rho_est.py --field field4 \
  --outdir /tmp/m11_rho_est/per_field
cd /Users/weidong/ai/small_world && PYTHONHASHSEED=0 MPLBACKEND=Agg \
  /Users/weidong/ai/small_world/.venv-m11/bin/python /tmp/m11_rho_est/rho_est.py --field field2 \
  --outdir /tmp/m11_rho_est/per_field
# §2.4 后验诊断（未预注册）
cd /Users/weidong/ai/small_world && PYTHONHASHSEED=0 MPLBACKEND=Agg \
  /Users/weidong/ai/small_world/.venv-m11/bin/python /tmp/m11_rho_est/posthoc.py
# 合成交付物（写 data/m11_rho_estimation/rho_estimate.{json,csv}）
cd /Users/weidong/ai/small_world && PYTHONHASHSEED=0 MPLBACKEND=Agg \
  /Users/weidong/ai/small_world/.venv-m11/bin/python /tmp/m11_rho_est/compose.py
```


### A.1 `rho_est.py`  
（sha256 544e7695c4f36cec2eaa247aac870aa040b694932179d989d098c896ec47d0bc）

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M11 · rho = thr / A 的数据侧估计（**只读仓内既有件；只新建 3 件交付物**）

驱动脚本（**非仓内件**：按任务红线"只新建 3 件"，本脚本放在 /tmp，逐字备份见
`docs/m11_rho_estimation.md` 附录 A；sha256 记入 rho_estimate.json::meta.driver_sha256）。

口径（**完全照用**）：直接 import 仓内既有实现 `tools/m12_b5_roi_mask_audit.py`
（build_label_volume / load_movie / dff_and_threshold / real_roi_analysis）⇒ 与 B-5/审计同款。

用法：
  PYTHONHASHSEED=0 MPLBACKEND=Agg \
    /Users/weidong/ai/small_world/.venv-m11/bin/python /tmp/m11_rho_est/rho_est.py --field field4
红线：不跑仿真、不碰 MPS、不联网、不改任何既有文件。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from datetime import datetime, timezone

import numpy as np

ROOT = "/Users/weidong/ai/small_world"
NE = os.path.join(ROOT, "neural_exploration")
OUTDIR = os.path.join(NE, "data", "m11_rho_estimation")
DOC = os.path.join(NE, "docs", "m11_rho_estimation.md")
PREREG_SHA = "d68a4afe0ab9698eb6a54c4fa50cdbd778700a8191887c3bd4be85547fb01660"
PREREG_FROZEN_AT = "2026-10-14T18:17:08Z"

_al = importlib.util.spec_from_file_location(
    "b5", os.path.join(NE, "tools", "m12_b5_roi_mask_audit.py"))
b5 = importlib.util.module_from_spec(_al)
_al.loader.exec_module(b5)

FPS = 1.0 / 0.15873                       # 6.300006 Hz（data/m11_reference.csv R-01）
TAUS = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]     # 秒（§0.2 冻结档）
TAU_PRIMARY = 1.0
STATS = ["p05", "p10", "p25", "median", "mode", "mean"]
STAT_PRIMARY = "p10"
PEAK_DEFS = ["bl_local", "pre5", "pctl0", "abs"]
PEAK_PRIMARY = "bl_local"
N_ISO_MIN = 30                            # §0.5 最小样本门
BOOT_B = 1000
BOOT_SEED = 20261015
BLOCK_FRAMES = 63                         # = 10.0 s（= 仿真协议窗 T=10 s）
Z0 = 0.92472
Z2 = 4.0
Z3 = 5.0


_ABS_PEAK_DIAG = None


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_b(b):
    return hashlib.sha256(b).hexdigest()


def gap_frames(tau_s):
    return int(round(2.0 * tau_s * FPS))


# ----------------------------------------------------------------------
# 越阈段 / 孤立判据 / 峰幅（§0.2–§0.3 冻结实现）
# ----------------------------------------------------------------------
def detect_runs(bmask):
    """返回极大 True 连续段 [(s, e)) ，e 为开区间端点。"""
    if not bmask.any():
        return []
    d = np.diff(np.r_[np.int8(0), bmask.astype(np.int8), np.int8(0)])
    idx = np.flatnonzero(d != 0)
    return [(int(idx[i]), int(idx[i + 1])) for i in range(0, len(idx), 2)]


def isolated_runs(trace, thr_detect, G, edge_rule="edge_clean"):
    """§0.2：`gap_prev >= G 且 gap_next >= G`；窗边界按 edge_rule 处置。"""
    runs = detect_runs(trace > thr_detect)
    n = trace.size
    out = []
    for k, (s, e) in enumerate(runs):
        if k == 0:
            ok_prev = (s >= G) if edge_rule == "edge_margin_G" else True
        else:
            ok_prev = (s - runs[k - 1][1]) >= G
        if k == len(runs) - 1:
            ok_next = ((n - e) >= G) if edge_rule == "edge_margin_G" else True
        else:
            ok_next = (runs[k + 1][0] - e) >= G
        if ok_prev and ok_next:
            out.append((s, e))
    return out


def peak_amp(trace, s, e, peak_def, G, roi_p20):
    """§0.3：峰 = 段内 max；基线四档。返回 None 表示样本不足（<3 帧静默窗）。"""
    pk = float(trace[s:e].max())
    if peak_def == "abs":
        return pk
    if peak_def == "pctl0":
        return pk - roi_p20
    w = G if peak_def == "bl_local" else 5
    lo = max(0, s - w)
    base = trace[lo:s]
    if base.size < 3:
        return None
    return pk - float(np.median(base))


def kde_mode(v, grid=512):
    """Gaussian KDE（Scott 带宽）+ 网格 argmax ⇒ 众数。"""
    v = np.asarray(v, dtype=np.float64)
    n = v.size
    if n < 2:
        return float(v[0]) if n else float("nan")
    sd = float(np.std(v, ddof=1))
    if sd <= 0:
        return float(v[0])
    h = 1.06 * sd * n ** (-1.0 / 5.0)
    lo, hi = float(v.min()) - 3 * h, float(v.max()) + 3 * h
    xs = np.linspace(lo, hi, grid)
    # 分块累加，避免 (grid × n) 大矩阵
    acc = np.zeros(grid)
    step = max(1, int(4e6 // max(grid, 1)))
    for i in range(0, n, step):
        chunk = v[i:i + step]
        z = (xs[:, None] - chunk[None, :]) / h
        acc += np.exp(-0.5 * z * z).sum(axis=1)
    return float(xs[int(np.argmax(acc))])


def stat_of(vals, stat):
    v = np.asarray(vals, dtype=np.float64)
    if v.size == 0:
        return None
    if stat == "p05":
        return float(np.percentile(v, 5))
    if stat == "p10":
        return float(np.percentile(v, 10))
    if stat == "p25":
        return float(np.percentile(v, 25))
    if stat == "median":
        return float(np.median(v))
    if stat == "mean":
        return float(v.mean())
    if stat == "mode":
        return kde_mode(v)
    raise ValueError(stat)


def zone_of(rho):
    if rho is None:
        return "unavailable"
    if rho <= Z0:
        return "Z0_strict_fail(rho<=0.92472)"
    if rho <= Z2:
        return "Z1_undetermined(0.92472<rho<=4)"
    if rho <= Z3:
        return "Z2_narrow_pass_window(4<rho<=5)"
    return "Z3_strict_fail(rho>5)"


def zones_span(lo, hi):
    """区间覆盖的区集合 + 是否跨界。"""
    if lo is None or hi is None:
        return [], False
    pts = [lo, hi]
    for b in (Z0, Z2, Z3):
        if lo < b < hi:
            pts.extend([b - 1e-9, b + 1e-9])
    zs = []
    for p in pts:
        z = zone_of(p)
        if z not in zs:
            zs.append(z)
    return zs, (len(zs) > 1)


# ----------------------------------------------------------------------
# 单 field 全流程
# ----------------------------------------------------------------------
def roi_per_events(roi, thr, tau_s, peak_def, edge_rule, frame_slice=None):
    """返回 (per_roi_peak_lists, diag)。frame_slice 为 (a,b) 子窗（子窗内沿用同一 thr）。"""
    G = gap_frames(tau_s)
    a, b = frame_slice if frame_slice else (0, roi[0]["_trace"].size)
    per_roi, diag = [], dict(n_events=0, n_runs=0, n_iso=0, n_dropped_shortbase=0,
                             n_rois=0, n_rois_with_iso=0)
    for r in roi:
        tr = r["_trace"][a:b]
        runs = detect_runs(tr > thr)
        iso = isolated_runs(tr, thr, G, edge_rule)
        p20 = float(np.percentile(tr, 20))
        vals = []
        for (s, e) in iso:
            v = peak_amp(tr, s, e, peak_def, G, p20)
            if v is None:
                diag["n_dropped_shortbase"] += 1
                continue
            vals.append(v)
        diag["n_runs"] += len(runs)
        diag["n_iso"] += len(iso)
        diag["n_rois"] += 1
        if vals:
            diag["n_rois_with_iso"] += 1
        per_roi.append((r["label"], r["area_px"], np.asarray(vals, dtype=np.float64)))
    return per_roi, diag


def aggregate(per_roi, stat, agg):
    if agg == "pooled":
        v = np.concatenate([x[2] for x in per_roi]) if per_roi else np.array([])
        return stat_of(v, stat), int(v.size)
    vals = [stat_of(x[2], stat) for x in per_roi if x[2].size > 0]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, 0
    return float(np.median(vals)), int(sum(x[2].size for x in per_roi))


def boot_ci(per_roi, stat, thr, b=BOOT_B, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    n = len(per_roi)
    if n == 0:
        return None, None, None, None
    A = np.empty(b)
    for i in range(b):
        idx = rng.integers(0, n, n)
        v = np.concatenate([per_roi[j][2] for j in idx if per_roi[j][2].size > 0]) \
            if any(per_roi[j][2].size > 0 for j in idx) else np.array([])
        A[i] = stat_of(v, stat) if v.size else np.nan
    A = A[np.isfinite(A)]
    if A.size == 0:
        return None, None, None, None
    lo, hi = float(np.percentile(A, 2.5)), float(np.percentile(A, 97.5))
    return (lo + hi) / 2, lo, hi, int(A.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="field4")
    ap.add_argument("--outdir", default="/tmp/m11_rho_est/per_field")
    args = ap.parse_args()
    global OUTDIR
    OUTDIR = args.outdir
    spec = b5.FIELDS[args.field]
    wx, wy = spec["wx"], spec["wy"]

    res = dict(generated_by="/tmp/m11_rho_est/rho_est.py（驱动脚本，非仓内件）",
               generated_at=utcnow(), evidence_level="measured", field=args.field,
               meta=dict(prereg_doc=os.path.relpath(DOC, ROOT),
                         prereg_doc_sha256_at_freeze=PREREG_SHA,
                         prereg_frozen_at_utc=PREREG_FROZEN_AT,
                         driver_sha256=sha256_b(open(__file__, "rb").read()),
                         frame_rate_hz=round(FPS, 6),
                         note="§0 预注册先于全部计算；本节任何数字均为预注册后产出"))

    # ---- 数据（同款调用）----
    vol, per = b5.build_label_volume(verbose=False, maskdir=spec["maskdir"])
    st = b5.load_movie(spec["subdir"], spec["prefix"])
    st = st[:wx, :wy]
    dff, med, mad, thr = b5.dff_and_threshold(st)
    rows, summ = b5.real_roi_analysis(vol, dff, thr, wx=wx, wy=wy, n_frames=st.shape[3])
    roi = [r for r in rows if r["fully_in_window"]]
    res["caliber"] = dict(
        rule="逐像素 F0=20 分位；dF/F=(F-F0)/F0；事件阈值 = 中位 + 3·MAD（承 M11 J2 同源）",
        dff_median=round(med, 6), dff_mad=round(mad, 6), event_threshold_dff=round(thr, 6),
        unit="真实 ROI 掩膜（等权像素均值；Mask.weights 未使用）",
        n_rois_field_total=summ["n_rois_field_total"],
        n_rois_fully_in_movie_window=summ["n_rois_fully_in_movie_window"],
        n_frames=int(st.shape[3]), window=[wx, wy],
        duration_s=round(st.shape[3] * 0.15873, 2))

    # ---- §1 自检（3/3 必须命中）----
    silent = summ["silent_frac_zero_count_real_roi"]
    n_ev = [r["events"] for r in roi]
    sc = dict(
        silent_frac_zero_count_real_roi=silent,
        silent_frac_expected=0.101852 if args.field == "field4" else 0.229885,
        n_rois_fully_in_movie_window=summ["n_rois_fully_in_movie_window"],
        n_zero_events=int(sum(1 for e in n_ev if e == 0)),
        zero_events_is_44_of_432=(int(sum(1 for e in n_ev if e == 0)) == 44
                                  and summ["n_rois_fully_in_movie_window"] == 432)
        if args.field == "field4" else None,
        event_threshold_dff=round(thr, 6),
        thr_expected=1.578470 if args.field == "field4" else 2.308268,
        thr_match=bool(abs(thr - (1.578470 if args.field == "field4" else 2.308268)) < 5e-7),
        median_events_per_roi=summ["median_events_per_roi"],
        median_events_expected=46.5 if args.field == "field4" else 8.0,
        median_events_match=bool(summ["median_events_per_roi"]
                                 == (46.5 if args.field == "field4" else 8.0)),
        silent_match=bool(abs(silent - (0.101852 if args.field == "field4" else 0.229885)) < 5e-7),
    )
    sc["selfcheck_3of3_pass"] = bool(sc["silent_match"] and sc["thr_match"]
                                     and sc["median_events_match"])
    res["selfcheck"] = sc
    print("[selfcheck] silent=%s (exp %s, match=%s) thr=%.6f (exp %.6f, match=%s) "
          "median_events=%s (exp %s, match=%s) => 3/3=%s"
          % (silent, sc["silent_frac_expected"], sc["silent_match"], thr,
             sc["thr_expected"], sc["thr_match"], summ["median_events_per_roi"],
             sc["median_events_expected"], sc["median_events_match"],
             sc["selfcheck_3of3_pass"]))
    if not sc["selfcheck_3of3_pass"]:
        res["verdict"] = "invalid（§1 自检未 3/3 命中 ⇒ 按 §0.6 停下，不产出 rho）"
        os.makedirs(OUTDIR, exist_ok=True)
        with open(os.path.join(OUTDIR, "rho_estimate_%s.json" % args.field), "w") as f:
            json.dump(res, f, indent=1, ensure_ascii=False)
        raise SystemExit("自检未通过 ⇒ 停止（不得继续测 A/rho）")

    # ---- §2 主档 ----
    sub432 = [r for r in roi]
    per_roi, diag = roi_per_events(sub432, thr, TAU_PRIMARY, PEAK_PRIMARY, "edge_clean")
    A_prim, n_iso = aggregate(per_roi, STAT_PRIMARY, "pooled")
    boot_mid, boot_lo, boot_hi, nboot = boot_ci(per_roi, STAT_PRIMARY, thr)
    rho_prim = thr / A_prim
    res["primary"] = dict(
        peak_def=PEAK_PRIMARY, stat=STAT_PRIMARY, tau_s=TAU_PRIMARY,
        gap_frames=gap_frames(TAU_PRIMARY), agg="pooled", unit_set="432_all",
        window="full2000", edge_rule="edge_clean",
        n_rois=len(sub432), n_events_total=sum(r["events"] for r in sub432),
        n_runs=diag["n_runs"], n_iso=n_iso, n_iso_raw=diag["n_iso"],
        n_dropped_shortbase=diag["n_dropped_shortbase"],
        n_rois_with_iso=diag["n_rois_with_iso"],
        A=float(A_prim), A_over_thr=float(A_prim / thr), thr=float(thr),
        rho=float(rho_prim), zone=zone_of(rho_prim),
        boot95=dict(lo=boot_lo, hi=boot_hi, n_eff=nboot, seed=BOOT_SEED, B=BOOT_B,
                    rho_lo=(thr / boot_hi) if boot_hi else None,
                    rho_hi=(thr / boot_lo) if boot_lo else None))
    print("[primary] n_iso=%d A=%.6f (A/thr=%.4f) rho=thr/A=%.6f zone=%s | boot95 A=[%.6f,%.6f]"
          % (n_iso, A_prim, A_prim / thr, rho_prim, zone_of(rho_prim), boot_lo, boot_hi))

    # ---- §2 降阈探针（§0.7 #2 预注册的**条件性**补充诊断，不作主档）----
    trig = bool(A_prim > thr and rho_prim < 1.0)
    res["thr_aux_probe"] = dict(
        prereg_trigger="主档给出 A_prim > thr 且 rho_prim < 1（§0.7 #2）",
        trigger_fired=trig, note="改变 thr 即改变口径 ⇒ **只作补充诊断、不作主档**")
    if trig:
        p2 = [dict(r, _trace=r["_trace"]) for r in roi]
        pr2, dg2 = roi_per_events(p2, thr / 2.0, TAU_PRIMARY, PEAK_PRIMARY, "edge_clean")
        A2, n2 = aggregate(pr2, STAT_PRIMARY, "pooled")
        res["thr_aux_probe"].update(thr_aux=float(thr / 2.0), n_iso=n2, A=float(A2),
                                    A_over_thr=float(A2 / thr),
                                    rho_vs_canonical_thr=float(thr / A2),
                                    zone=zone_of(thr / A2),
                                    caveat="**口径已变（检出阈 thr/2）⇒ 该 rho 与带不同源，不得进决断地图**")
        print("[thr/2 probe] n_iso=%d A=%.6f A/thr=%.4f (口径已变，仅诊断)" % (n2, A2, A2 / thr))
    else:
        res["thr_aux_probe"]["note"] += "；**触发条件未满足 ⇒ 按预注册不跑**"

    # ---- §3 敏感性全表 ----
    csv_rows = []

    def emit(section, window, unit_set, tau, peak_def, stat, agg, edge, n_rois, n_ev, n_iso_v,
             A_v, note=""):
        rho_v = (thr / A_v) if (A_v is not None and A_v > 0) else None
        ok = (n_iso_v is not None and n_iso_v >= N_ISO_MIN and A_v is not None and A_v > 0)
        csv_rows.append(dict(
            section=section, field=args.field, window=window, unit_set=unit_set,
            tau_s=tau, gap_frames=gap_frames(tau), peak_def=peak_def, stat=stat, agg=agg,
            edge_rule=edge, n_rois=n_rois, n_events=n_ev, n_iso=n_iso_v,
            A=("" if not ok else round(float(A_v), 6)),
            A_over_thr=("" if not ok else round(float(A_v / thr), 4)),
            rho=("" if not ok else round(float(rho_v), 6)),
            zone=("unavailable" if not ok else zone_of(rho_v)),
            n_iso_ok=bool(ok), note=note if not ok else note,
            reason=("" if ok else ("n_iso<%d ⇒ 按 §0.5 标 unavailable（不得当 0/未命中）" % N_ISO_MIN
                                   if n_iso_v is not None else "无样本"))))
        return csv_rows[-1]

    # 主档
    emit("PRIMARY", "full2000", "432_all", TAU_PRIMARY, PEAK_PRIMARY, STAT_PRIMARY, "pooled",
         "edge_clean", len(sub432), res["primary"]["n_events_total"], n_iso, A_prim, "主档")

    # 轴 1：统计量
    for s in STATS:
        if s == STAT_PRIMARY:
            continue
        A_v, n_v = aggregate(per_roi, s, "pooled")
        emit("SENS_stat", "full2000", "432_all", TAU_PRIMARY, PEAK_PRIMARY, s, "pooled",
             "edge_clean", len(sub432), res["primary"]["n_events_total"], n_v, A_v, "统计量档")
    # 轴 2：峰定义
    global _ABS_PEAK_DIAG
    for pd in PEAK_DEFS:
        if pd == PEAK_PRIMARY:
            continue
        prx, dgx = roi_per_events(roi, thr, TAU_PRIMARY, pd, "edge_clean")
        A_v, n_v = aggregate(prx, STAT_PRIMARY, "pooled")
        if pd == "abs":
            _av = np.concatenate([x[2] for x in prx]) if n_v else np.array([np.nan])
            _ABS_PEAK_DIAG = dict(min=float(np.nanmin(_av)),
                                  min_ratio=float(np.nanmin(_av) / thr),
                                  frac_near_thr=float(np.mean(_av <= thr * 1.01)))
        emit("SENS_peakdef", "full2000", "432_all", TAU_PRIMARY, pd, STAT_PRIMARY, "pooled",
             "edge_clean", len(roi), res["primary"]["n_events_total"], n_v, A_v, "峰定义档")
    # 轴 3：tau
    for t in TAUS:
        if t == TAU_PRIMARY:
            continue
        prx, dgx = roi_per_events(roi, thr, t, PEAK_PRIMARY, "edge_clean")
        A_v, n_v = aggregate(prx, STAT_PRIMARY, "pooled")
        emit("SENS_tau", "full2000", "432_all", t, PEAK_PRIMARY, STAT_PRIMARY, "pooled",
             "edge_clean", len(roi), res["primary"]["n_events_total"], n_v, A_v, "τ 档")
    # 轴 4：单元集合
    for tag, f in (("area>=10", lambda r: r["area_px"] >= 10),
                   ("area>=20", lambda r: r["area_px"] >= 20)):
        sub = [r for r in roi if f(r)]
        prx, dgx = roi_per_events(sub, thr, TAU_PRIMARY, PEAK_PRIMARY, "edge_clean")
        A_v, n_v = aggregate(prx, STAT_PRIMARY, "pooled")
        emit("SENS_units", "full2000", tag, TAU_PRIMARY, PEAK_PRIMARY, STAT_PRIMARY, "pooled",
             "edge_clean", len(sub), sum(r["events"] for r in sub), n_v, A_v, "单元集合档")
    # 轴 5：聚合口径
    A_v, n_v = aggregate(per_roi, STAT_PRIMARY, "per_roi_median")
    emit("SENS_agg", "full2000", "432_all", TAU_PRIMARY, PEAK_PRIMARY, STAT_PRIMARY,
         "per_roi_median", "edge_clean", len(roi), res["primary"]["n_events_total"], n_v, A_v,
         "逐 ROI 取 p10 后跨 ROI 中位")
    # 轴 6：边界规则
    prx, dgx = roi_per_events(roi, thr, TAU_PRIMARY, PEAK_PRIMARY, "edge_margin_G")
    A_v, n_v = aggregate(prx, STAT_PRIMARY, "pooled")
    emit("SENS_edge", "full2000", "432_all", TAU_PRIMARY, PEAK_PRIMARY, STAT_PRIMARY, "pooled",
         "edge_margin_G", len(roi), res["primary"]["n_events_total"], n_v, A_v, "要求距窗边 ≥G 帧")

    # 轴 7：窗口（半窗 + 63 帧块）
    nfr = st.shape[3]
    half = nfr // 2
    halves = {}
    for tag, sl in (("half_first1000", (0, half)), ("half_second1000", (half, nfr))):
        prx, dgx = roi_per_events(roi, thr, TAU_PRIMARY, PEAK_PRIMARY, "edge_clean", frame_slice=sl)
        A_v, n_v = aggregate(prx, STAT_PRIMARY, "pooled")
        halves[tag] = A_v
        emit("SENS_window", tag, "432_all", TAU_PRIMARY, PEAK_PRIMARY, STAT_PRIMARY, "pooled",
             "edge_clean", len(roi), "", n_v, A_v, "半窗（%d 帧）" % (sl[1] - sl[0]))
    nblk = nfr // BLOCK_FRAMES
    blk_A, blk_n = [], []
    for b in range(nblk):
        sl = (b * BLOCK_FRAMES, (b + 1) * BLOCK_FRAMES)
        prx, dgx = roi_per_events(roi, thr, TAU_PRIMARY, PEAK_PRIMARY, "edge_clean", frame_slice=sl)
        A_v, n_v = aggregate(prx, STAT_PRIMARY, "pooled")
        blk_A.append(A_v)
        blk_n.append(n_v)
        emit("SENS_block63", "block63_%02d" % (b + 1), "432_all", TAU_PRIMARY, PEAK_PRIMARY,
             STAT_PRIMARY, "pooled", "edge_clean", len(roi), "", n_v, A_v,
             "非重叠 63 帧块（=10.0 s） frames=[%d,%d)" % sl)
    okA = [a for a, n in zip(blk_A, blk_n) if a is not None and n >= N_ISO_MIN]
    res["subwindow_blocks63"] = dict(
        n_blocks=nblk, block_frames=BLOCK_FRAMES, seconds=round(BLOCK_FRAMES * 0.15873, 3),
        n_blocks_eligible=len(okA),
        n_blocks_unavailable=int(nblk - len(okA)),
        A_median=(float(np.median(okA)) if okA else None),
        A_min=(float(min(okA)) if okA else None), A_max=(float(max(okA)) if okA else None),
        A_per_block=[(None if a is None else round(float(a), 6)) for a in blk_A],
        n_iso_per_block=blk_n,
        halves=dict(half_first1000=(None if halves["half_first1000"] is None
                                    else round(halves["half_first1000"], 6)),
                    half_second1000=(None if halves["half_second1000"] is None
                                     else round(halves["half_second1000"], 6))))
    emit("SENS_block63_summary", "blocks63_summary", "432_all", TAU_PRIMARY, PEAK_PRIMARY,
         STAT_PRIMARY, "pooled", "edge_clean", len(roi), "",
         (sum(blk_n) if blk_n else None),
         (float(np.median(okA)) if okA else None),
         "31 块中的合格块（n_iso>=30）中位数；合格 %d/%d" % (len(okA), nblk))

    res["sensitivity_rows"] = csv_rows

    # ---- 区间汇总 & 落区 ----
    sensA = [r["A"] for r in csv_rows
             if r["A"] != "" and r["section"] != "PRIMARY" and r["section"] != "SENS_block63"]
    allA = [r["A"] for r in csv_rows if r["A"] != ""]
    def rng(vals):
        if not vals:
            return None, None
        return float(min(vals)), float(max(vals))

    A_def_lo, A_def_hi = rng(sensA)
    A_all_lo, A_all_hi = rng(allA)
    res["rho_interval"] = dict(
        thr=float(thr),
        primary=dict(A=A_prim, A_lo=boot_lo, A_hi=boot_hi, source="ROI 自助 95%（B=1000）",
                     rho=float(rho_prim), rho_lo=(thr / boot_hi), rho_hi=(thr / boot_lo)),
        dispersion_subwindow=dict(
            A_min=res["subwindow_blocks63"]["A_min"], A_max=res["subwindow_blocks63"]["A_max"],
            rho_min=(thr / res["subwindow_blocks63"]["A_max"]
                     if res["subwindow_blocks63"]["A_max"] else None),
            rho_max=(thr / res["subwindow_blocks63"]["A_min"]
                     if res["subwindow_blocks63"]["A_min"] else None)),
        definitional=dict(A_min=A_def_lo, A_max=A_def_hi,
                          rho_min=(thr / A_def_hi) if A_def_hi else None,
                          rho_max=(thr / A_def_lo) if A_def_lo else None,
                          note="全部敏感性档（不含主档本身与块汇总）的 min–max；**定义不确定度，不是概率区间**"),
        definitional_incl_primary=dict(A_min=A_all_lo, A_max=A_all_hi,
                                       rho_min=(thr / A_all_hi), rho_max=(thr / A_all_lo)))
    for tag, lo, hi in (("primary_boot95", boot_lo, boot_hi),
                        ("definitional", A_def_lo, A_def_hi),
                        ("definitional_incl_primary", A_all_lo, A_all_hi)):
        if lo and hi:
            zs, cross = zones_span(thr / hi, thr / lo)
            res["rho_interval"][tag + "_zone"] = dict(
                A_lo=lo, A_hi=hi, rho_lo=thr / hi, rho_hi=thr / lo,
                zones_covered=zs, crosses_boundary=cross)

    # ---- 深色诊断 ----
    res["diagnostics"] = dict(
        censoring=dict(
            statement="凡在 thr 上检出的事件，其**绝对峰必 > thr**（检出定义使然）",
            abs_peak_min_over_all_iso=(globals().get("_ABS_PEAK_DIAG") or {}).get("min"),
            abs_peak_min_thr_ratio=(globals().get("_ABS_PEAK_DIAG") or {}).get("min_ratio"),
            abs_peak_frac_within_1pct_of_thr=(globals().get("_ABS_PEAK_DIAG") or {}).get("frac_near_thr"),
            note="⇒ `abs` 档 rho<1 恒成立（口径伪影）；主档（减基线）不退化但单侧偏大 ⇒ "
                 "**rho_est <= rho_sim（弱下界性质）**：rho_est<1 **不是** rho_sim<1 的证据"),
        roi_baseline_offset=dict(
            pooled_pctl0_median=float(np.median([float(np.percentile(r["_trace"], 20))
                                                 for r in roi])),
            note="ROI 层 20 分位（= 'pctl0' 基线）的跨 ROI 中位；与池化 dff_median=%.6f 同量级"
                 % med))

    res["decision_map"] = dict(
        zones=[dict(id="Z0", rule="rho <= 0.92472",
                    meaning="严格 fail：钙域静默 = frac(c_i=0) = 0.000122 < 带下界 0.101852"),
               dict(id="Z1", rule="0.92472 < rho <= 4",
                    meaning="未定：严格下界 <= frac(c_i<4)=0.076294 < 带下界（要落带须靠时刻结构抬升）"),
               dict(id="Z2", rule="4 < rho <= 5",
                    meaning="唯一可能 pass 的窄窗：下界 frac(c_i<5)=0.222412 已在带内，余量仅 3.25%"),
               dict(id="Z3", rule="rho > 5",
                    meaning="严格 fail：静默 >= frac(c_i<6)=0.453003 > 带上界 0.229885")],
        source="docs/m11_acceptance_master.md L5658-L5668 / L5684-L5691（R190）+ L5762（R191 §⑥）",
        rho_primary=float(rho_prim), zone_primary=zone_of(rho_prim))

    res["declarations"] = dict(
        ran_simulation=False, used_mps_or_gpu=False, used_network=False,
        downloaded_papers=False, modified_existing_files=False,
        created_files=[os.path.relpath(DOC, ROOT),
                       "neural_exploration/data/m11_rho_estimation/rho_estimate.json",
                       "neural_exploration/data/m11_rho_estimation/rho_estimate.csv"],
        no_verdict_on_b="本文件不替主 agent 下 (b) 的裁决，只报事实与算术",
        caliber_not_self_invented="thr/dF-F/ROI 轨迹/事件计数全部调用 tools/m12_b5_roi_mask_audit.py")

    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "rho_estimate_%s.json" % args.field), "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    cols = ["section", "field", "window", "unit_set", "tau_s", "gap_frames", "peak_def",
            "stat", "agg", "edge_rule", "n_rois", "n_events", "n_iso", "A", "A_over_thr",
            "rho", "zone", "n_iso_ok", "note", "reason"]
    with open(os.path.join(OUTDIR, "rho_estimate_%s.csv" % args.field), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in csv_rows:
            w.writerow(r)
    print("wrote", OUTDIR, "(%s)" % args.field)
    return res


if __name__ == "__main__":
    main()
```

### A.2 `selfcheck.py`  
（sha256 e2ba272a4f7aadc8e44202e37d0901cb4bec953f9855130ad014ff5396410681）

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""§1 实现自检：用**与 B-5/审计同款**的既有实现复现三个已知真值。只读，不写任何文件。"""
import importlib.util
import os
import time

import numpy as np

NE = "/Users/weidong/ai/small_world/neural_exploration"
_al = importlib.util.spec_from_file_location("b5", os.path.join(NE, "tools", "m12_b5_roi_mask_audit.py"))
b5 = importlib.util.module_from_spec(_al)
_al.loader.exec_module(b5)

t0 = time.time()
spec = b5.FIELDS["field4"]
vol, per = b5.build_label_volume(verbose=False, maskdir=spec["maskdir"])
t1 = time.time()
st = b5.load_movie(spec["subdir"], spec["prefix"])[:200, :200]
t2 = time.time()
dff, med, mad, thr = b5.dff_and_threshold(st)
rows, summ = b5.real_roi_analysis(vol, dff, thr, wx=200, wy=200, n_frames=st.shape[3])
t3 = time.time()

roi = [r for r in rows if r["fully_in_window"]]
n_zero = int(sum(1 for r in roi if r["events"] == 0))
print("movie shape (x,y,z,t) = %s   n_files_chunks = %d" % (st.shape, len(per)))
print("decode %.1fs  load %.1fs  dff+analysis %.1fs" % (t1 - t0, t2 - t1, t3 - t2))
print("dff_median = %.6f   dff_mad = %.6f   event_threshold_dff = %.6f" % (med, mad, thr))
print("n_rois_field_total = %d   n_rois_fully_in_movie_window = %d"
      % (summ["n_rois_field_total"], summ["n_rois_fully_in_movie_window"]))
print("n_zero_events = %d / %d" % (n_zero, len(roi)))
print("silent_frac_zero_count_real_roi = %.6f" % summ["silent_frac_zero_count_real_roi"])
print("median_events_per_roi = %s" % summ["median_events_per_roi"])
print("-" * 70)
chk = [("silent_frac == 0.101852", abs(summ["silent_frac_zero_count_real_roi"] - 0.101852) < 5e-7),
       ("silent_frac == 44/432", abs(summ["silent_frac_zero_count_real_roi"] - 44 / 432) < 5e-7
        and n_zero == 44 and len(roi) == 432),
       ("thr == 1.578470", abs(thr - 1.578470) < 5e-7),
       ("median_events == 46.5", summ["median_events_per_roi"] == 46.5)]
for k, v in chk:
    print("  %-26s %s" % (k, "HIT" if v else "MISS"))
print("SELFCHECK 4/4 =", all(v for _, v in chk))
```

### A.3 `posthoc.py`（**未预注册**，仅诊断）  
（sha256 5a05e5e294368d37662048853752b30b75f76c35c96171644ff1f2627430f7da）

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""后验诊断（**未预注册**；在看到主档结果之后追加；只作诊断，**不改主档**）。

目的：检验主档 A（= 孤立事件"峰 − 局部静默基线"的 p10）是否被
①高基线偏移 ROI 的**近阈噪声性越阈**、②事件前局部基线抬升 所主导。
输出：/tmp/m11_rho_est/posthoc.json
"""
import importlib.util
import json
import os
import sys

import numpy as np

_al = importlib.util.spec_from_file_location("rho_est", "/tmp/m11_rho_est/rho_est.py")
R = importlib.util.module_from_spec(_al)
_al.loader.exec_module(R)
b5 = R.b5

out = {}
for field in ("field4",):
    spec = b5.FIELDS[field]
    wx, wy = spec["wx"], spec["wy"]
    vol, _ = b5.build_label_volume(verbose=False, maskdir=spec["maskdir"])
    st = b5.load_movie(spec["subdir"], spec["prefix"])[:wx, :wy]
    dff, med, mad, thr = b5.dff_and_threshold(st)
    rows, summ = b5.real_roi_analysis(vol, dff, thr, wx=wx, wy=wy, n_frames=st.shape[3])
    roi = [r for r in rows if r["fully_in_window"]]
    G = R.gap_frames(1.0)

    rises, bases, pcts, rois_p10, p20s = [], [], [], [], []
    for r in roi:
        tr = r["_trace"]
        p20 = float(np.percentile(tr, 20))
        p20s.append(p20)
        vals, bs = [], []
        for (s, e) in R.isolated_runs(tr, thr, G, "edge_clean"):
            v = R.peak_amp(tr, s, e, "bl_local", G, p20)
            if v is None:
                continue
            vals.append(v)
            bs.append(float(np.median(tr[max(0, s - G):s])))
        if vals:
            rois_p10.append(float(np.percentile(vals, 10)))
        rises.extend(vals)
        bases.extend(bs)
        pcts.extend([p20] * len(vals))
    rises = np.array(rises); bases = np.array(bases); pcts = np.array(pcts)
    p20s = np.array(p20s)
    rois_p10 = np.array(rois_p10)

    d = dict(n_iso=len(rises), thr=float(thr), dff_mad=float(mad), dff_median=float(med),
             n_rois_with_iso=len(rois_p10))
    d["A_primary"] = float(np.percentile(rises, 10))
    d["rise_quantiles"] = {q: float(np.percentile(rises, q)) for q in (1, 5, 10, 25, 50, 75, 90)}
    d["rise_over_mad"] = {q: float(np.percentile(rises, q) / mad) for q in (5, 10, 25, 50)}
    d["frac_rise_below_1mad"] = float(np.mean(rises < mad))
    d["frac_rise_below_2mad"] = float(np.mean(rises < 2 * mad))
    # ① 事件前局部基线 vs 幅度
    d["pre_baseline_quantiles"] = {q: float(np.percentile(bases, q)) for q in (5, 10, 25, 50, 75, 90)}
    d["corr_prebaseline_vs_rise_pearson"] = float(np.corrcoef(bases, rises)[0, 1])
    rk_b = np.argsort(np.argsort(bases)).astype(float)
    rk_r = np.argsort(np.argsort(rises)).astype(float)
    d["corr_prebaseline_vs_rise_spearman"] = float(np.corrcoef(rk_b, rk_r)[0, 1])
    d["corr_roi_p20_vs_rise_spearman"] = float(np.corrcoef(
        np.argsort(np.argsort(pcts)).astype(float), rk_r)[0, 1])
    # ② 限定"低基线偏移"的 ROI（pctl0 下半区）
    cut = float(np.median(p20s))
    mask_lowbase = pcts <= cut
    d["D1a_low_baseline_rois"] = dict(
        rule="ROI 层 p20（pctl0）≤ 全体中位 %.4f" % cut,
        n_rois=int(np.sum(p20s <= cut)), n_iso=int(mask_lowbase.sum()),
        A=float(np.percentile(rises[mask_lowbase], 10)) if mask_lowbase.sum() > 30 else None)
    mask_highbase = pcts > cut
    d["D1a_high_baseline_rois"] = dict(
        n_rois=int(np.sum(p20s > cut)), n_iso=int(mask_highbase.sum()),
        A=float(np.percentile(rises[mask_highbase], 10)) if mask_highbase.sum() > 30 else None)
    # ③ 限定"事件前局部基线不抬升"的事件（≤ 该 ROI 的 pctl0 + 0.5·MAD）
    keep = bases <= pcts + 0.5 * mad
    d["D1b_unshifted_baseline_events"] = dict(
        rule="事件前局部基线 ≤ 该 ROI pctl0 + 0.5·MAD", n_iso=int(keep.sum()),
        A=float(np.percentile(rises[keep], 10)) if keep.sum() > 30 else None)
    keep2 = bases <= pcts + 1.0 * mad
    d["D1b_unshifted_baseline_events_1mad"] = dict(
        rule="事件前局部基线 ≤ 该 ROI pctl0 + 1.0·MAD", n_iso=int(keep2.sum()),
        A=float(np.percentile(rises[keep2], 10)) if keep2.sum() > 30 else None)
    # ④ 逐 ROI p10 的跨 ROI 分布
    d["per_roi_p10_distribution"] = dict(
        n=len(rois_p10), p05=float(np.percentile(rois_p10, 5)),
        p25=float(np.percentile(rois_p10, 25)), median=float(np.median(rois_p10)),
        p75=float(np.percentile(rois_p10, 75)), p95=float(np.percentile(rois_p10, 95)))
    out[field] = d
    print("[%s] A_primary=%.6f  lowbase A=%s  highbase A=%s  D1b A=%s  rho(D1a_low)=%s"
          % (field, d["A_primary"], d["D1a_low_baseline_rois"]["A"],
             d["D1a_high_baseline_rois"]["A"], d["D1b_unshifted_baseline_events"]["A"],
             (thr / d["D1a_low_baseline_rois"]["A"]) if d["D1a_low_baseline_rois"]["A"] else None))

json.dump(out, open("/tmp/m11_rho_est/posthoc.json", "w"), indent=1, ensure_ascii=False)
print("wrote /tmp/m11_rho_est/posthoc.json")
```

### A.4 `compose.py`  
（sha256 4b7e638340a0942a011b87a0588ae2327ba017189b2c0d259ea1704d68b026ab）

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 field4（主口径）/ field2（旁证）两轮 + 后验诊断，合成**交付的两件**机读产物。"""
import csv
import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np

NE = "/Users/weidong/ai/small_world/neural_exploration"
OUT = os.path.join(NE, "data", "m11_rho_estimation")
STAGE = "/tmp/m11_rho_est/per_field"
F4 = json.load(open(os.path.join(STAGE, "rho_estimate_field4.json")))
F2 = json.load(open(os.path.join(STAGE, "rho_estimate_field2.json")))
PH = json.load(open("/tmp/m11_rho_est/posthoc.json"))["field4"]


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def zone_of(rho):
    if rho is None:
        return "unavailable"
    if rho <= 0.92472:
        return "Z0_strict_fail(rho<=0.92472)"
    if rho <= 4.0:
        return "Z1_undetermined(0.92472<rho<=4)"
    if rho <= 5.0:
        return "Z2_narrow_pass_window(4<rho<=5)"
    return "Z3_strict_fail(rho>5)"


thr4 = F4["primary"]["thr"]
thr2 = F2["primary"]["thr"]
# ---- 后验诊断追加行（**未预注册**，见 §0.8）----
POST = [
    ("POSTHOC_D1a", "low_baseline_ROIs(pctl0<=0.1676)", 603, PH["D1a_low_baseline_rois"]["A"],
     "**未预注册**（看结果后追加，仅诊断）：限定 ROI 层 p20 下半区"),
    ("POSTHOC_D1a", "high_baseline_ROIs(pctl0>0.1676)", 402, PH["D1a_high_baseline_rois"]["A"],
     "**未预注册**：限定 ROI 层 p20 上半区"),
    ("POSTHOC_D1b", "events_prebaseline<=pctl0+0.5MAD", 76,
     PH["D1b_unshifted_baseline_events"]["A"],
     "**未预注册**：剔除「事件前局部基线已抬升」的事件（近阈几何效应最重者）"),
    ("POSTHOC_D1b", "events_prebaseline<=pctl0+1.0MAD", 173,
     PH["D1b_unshifted_baseline_events_1mad"]["A"], "**未预注册**：同上，1.0·MAD 档"),
    ("POSTHOC_perroi", "per_ROI_p10_p05", 337, PH["per_roi_p10_distribution"]["p05"],
     "**未预注册**：逐 ROI p10 的跨 ROI 分布下缘"),
    ("POSTHOC_perroi", "per_ROI_p10_p50", 337, PH["per_roi_p10_distribution"]["median"],
     "**未预注册**：逐 ROI p10 的跨 ROI 中位"),
    ("POSTHOC_perroi", "per_ROI_p10_p95", 337, PH["per_roi_p10_distribution"]["p95"],
     "**未预注册**：逐 ROI p10 的跨 ROI 分布上缘"),
]

rows = []
for tag, js in (("field4", F4), ("field2", F2)):
    for r in js["sensitivity_rows"]:
        rows.append(r)

for sec, wname, n_iso, A, note in POST:
    if A is None:
        continue
    rows.append(dict(section=sec, field="field4", window="full2000", unit_set="sub: " + wname,
                     tau_s=1.0, gap_frames=13, peak_def="bl_local", stat="p10", agg="pooled",
                     edge_rule="edge_clean", n_rois=432, n_events="", n_iso=n_iso,
                     A=round(float(A), 6), A_over_thr=round(float(A / thr4), 4),
                     rho=round(thr4 / float(A), 6), zone=zone_of(thr4 / float(A)),
                     n_iso_ok=bool(n_iso >= 30), note=note, reason=""))

# 诊断行（证据性，不是估计档）
DIAG = [("DIAGNOSTIC", "abs_peak_min_over_all_isolated_events(检出截断定理)",
         int(round(1010 * 1.0)), F4["diagnostics"]["censoring"]["abs_peak_min_over_all_iso"],
         "凡 thr 上检出者绝对峰必 > thr ⇒ 该行 A_over_thr=1.00015 **证** `abs` 档退化"),
        ("DIAGNOSTIC", "rise_p10_in_units_of_pooled_MAD(0.385062)", 1005,
         PH["rise_quantiles"]["10"], "主档 A = 1.25×MAD（池化噪声尺度）⇒ 低尾贴近噪声/阈几何")]
for sec, wname, n_iso, A, note in DIAG:
    rows.append(dict(section=sec, field="field4", window="full2000", unit_set="sub: " + wname,
                     tau_s=1.0, gap_frames=13, peak_def="bl_local", stat="p10", agg="pooled",
                     edge_rule="edge_clean", n_rois=432, n_events="", n_iso=n_iso,
                     A=round(float(A), 6), A_over_thr=round(float(A / thr4), 4),
                     rho=round(thr4 / float(A), 6), zone=zone_of(thr4 / float(A)),
                     n_iso_ok=True, note=note, reason=""))

cols = ["section", "field", "window", "unit_set", "tau_s", "gap_frames", "peak_def", "stat",
        "agg", "edge_rule", "n_rois", "n_events", "n_iso", "A", "A_over_thr", "rho", "zone",
        "n_iso_ok", "note", "reason"]
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "rho_estimate.csv"), "w", newline="", encoding="utf-8") as f:
    f.write("# M11 rho=thr/A 数据侧估计 · 逐敏感性档（field4=主口径 / field2=旁证；"
            "POSTHOC_* 行**未预注册**、仅诊断，见 docs/m11_rho_estimation.md §0.8/§3）\n")
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow(r)

# ---------------- 汇总 ----------------
def rng_of(sel):
    v = [r["A"] for r in sel if r["A"] != ""]
    return (float(min(v)), float(max(v))) if v else (None, None)


f4_sens = [r for r in rows if r["field"] == "field4" and r["section"].startswith("SENS")]
f4_def = [r for r in f4_sens if r["section"] != "SENS_block63"]
f4_all = [r for r in rows if r["field"] == "field4" and r["A"] != "" and not r["section"].startswith("DIAGNOSTIC")]
f2_sens = [r for r in rows if r["field"] == "field2" and r["section"].startswith("SENS")]


def zsum(sel):
    out = {}
    for r in sel:
        if r["A"] == "":
            continue
        out.setdefault(r["zone"], []).append((r["section"] + "/" + str(r["window"]), r["rho"]))
    return {k: dict(n=len(v), examples=v[:4]) for k, v in out.items()}


out = dict(
    generated_by="/tmp/m11_rho_est/{rho_est.py,compose.py}（驱动脚本，**非仓内件**；"
                 "逐字备份见 docs/m11_rho_estimation.md 附录 A）",
    generated_at=utcnow(), evidence_level="measured",
    task="在**真实钙成像数据**（field4 真实 ROI 掩膜，2000 帧 @6.300006 Hz）上估计单脉冲等效事件幅度 A，"
         "给出 rho = thr / A 的点估计与区间，并映射到 (b) 的决断地图四区",
    meta=dict(prereg_doc="neural_exploration/docs/m11_rho_estimation.md §0",
              prereg_doc_sha256_at_freeze=F4["meta"]["prereg_doc_sha256_at_freeze"],
              prereg_frozen_at_utc=F4["meta"]["prereg_frozen_at_utc"],
              driver_sha256_rho_est=F4["meta"]["driver_sha256"],
              driver_sha256_posthoc=sha("/tmp/m11_rho_est/posthoc.py"),
              prereg_body_unchanged=True,
              prereg_change_log="§0.0–§0.7（估计器定义 / 孤立判据 / 峰定义 / 子窗与单元 / 敏感性轴 / "
                                "预测与判定标准）**逐字未变**；§0.8 是 §0 自身规定的变更登记表，"
                                "**追加 1 行**：后验诊断轴（POSTHOC_*）—— 在看到主档结果"
                                "（rho=3.2847, Z1）**之后**追加，**仅诊断、不改主档**；§1–§5 为冻结后填入。"
                                "⇒ **主档与全部预注册档未被事后调整**（无「为凑 (4,5] 而调参」）",
              note="§0 预注册先于全部计算；本次唯一新增轴为**未预注册**的后验诊断轴（已显式标注）"),
    caliber=dict(
        field4=F4["caliber"], field2=F2["caliber"],
        thr_source="event_threshold_dff（field4=1.578470）—— 数据侧**已在手**，出处 "
                   "data/m12_b5_roi_mask/m12_b5_roi_mask_audit.json::fields.field4.calcium_event_caliber",
        same_caliber=True,
        note="thr 与 A **同为该审计 JSON 登记的 dF/F 事件检测口径**（逐像素 F0=20 分位；"
             "dF/F=(F−F0)/F0；阈值=中位+3·MAD）⇒ rho 是同一约定内两量之比"),
    selfcheck=dict(
        field4=F4["selfcheck"], field2=F2["selfcheck"],
        verdict="PASS（field4 3/3 + field2 3/3；field4 逐值：0.101852=44/432、thr 1.578470、"
                "median_events 46.5）⇒ 实现与 B-5/审计同款",
        command="cd /Users/weidong/ai/small_world && PYTHONHASHSEED=0 MPLBACKEND=Agg "
                "/Users/weidong/ai/small_world/.venv-m11/bin/python /tmp/m11_rho_est/selfcheck.py"),
    primary=dict(
        field="field4", **{k: F4["primary"][k] for k in
                           ("peak_def", "stat", "tau_s", "gap_frames", "agg", "unit_set",
                            "window", "edge_rule", "n_rois", "n_events_total", "n_runs",
                            "n_iso", "n_iso_raw", "n_dropped_shortbase", "n_rois_with_iso",
                            "A", "A_over_thr", "thr", "rho", "zone", "boot95")},
        A_definition="孤立事件峰幅值分布的 p10；峰幅值 =（越阈段内 trace 最大值）−（段起点前 G 帧"
                     "静默窗的中位数）；孤立 = 前后间隔均 ≥ G = round(2·τ·6.300006) 帧，τ=1.0 s"),
    A_estimates=dict(
        point=F4["primary"]["A"],
        intervals=dict(
            statistical_bootstrap=F4["rho_interval"]["primary"],
            subwindow_dispersion=F4["rho_interval"]["dispersion_subwindow"],
            definitional_all_tiers=F4["rho_interval"]["definitional"],
            definitional_incl_primary_and_blocks=F4["rho_interval"]["definitional_incl_primary"],
            post_hoc_fair_baseline_events=dict(
                A_min=min(PH["D1b_unshifted_baseline_events"]["A"],
                          PH["D1b_unshifted_baseline_events_1mad"]["A"]),
                A_max=max(PH["D1b_unshifted_baseline_events"]["A"],
                          PH["D1b_unshifted_baseline_events_1mad"]["A"]),
                rho_min=thr4 / max(PH["D1b_unshifted_baseline_events"]["A"],
                                   PH["D1b_unshifted_baseline_events_1mad"]["A"]),
                rho_max=thr4 / min(PH["D1b_unshifted_baseline_events"]["A"],
                                   PH["D1b_unshifted_baseline_events_1mad"]["A"]),
                note="**未预注册**，仅诊断；剔除近阈几何效应最重的事件后的 A（偏大 ⇒ rho 偏小）")),
        interval_sources_note="统计区间 = ROI 自助 95%（B=1000, seed=20261015）；"
                              "跨子窗离散度 = 31×63 帧块（=10.0 s）；"
                              "定义性区间 = 全部敏感性档 min–max（**定义不确定度，非概率区间**）"),
    rho=dict(
        point=F4["primary"]["rho"], thr=thr4,
        statistical_interval=[F4["primary"]["boot95"]["rho_lo"], F4["primary"]["boot95"]["rho_hi"]],
        sensitivity_interval=[F4["rho_interval"]["definitional"]["rho_min"],
                              F4["rho_interval"]["definitional"]["rho_max"]],
        sensitivity_interval_zone=zone_of(F4["primary"]["rho"]),
        crossings=F4["rho_interval"]["definitional_zone"]),
    sensitivity_summary=dict(
        n_rows=len(rows), n_unavailable=sum(1 for r in rows if not r["n_iso_ok"]),
        field4=dict(
            n_tiers=len(f4_all),
            A_min=rng_of(f4_all)[0], A_max=rng_of(f4_all)[1],
            rho_min=thr4 / rng_of(f4_all)[1], rho_max=thr4 / rng_of(f4_all)[0],
            best_case_for_Z0=dict(tier="abs（**退化档**）", A=1.601506, rho=0.985616),
            best_case_for_Z3=dict(tier="block63_03（63 帧块）", A=0.327078, rho=4.825968,
                                  note="仍 <= 5 ⇒ field4 **无任何档位**进入 Z3"),
            zone_counts=zsum([r for r in rows if r["field"] == "field4"
                              and r["A"] != "" and r["section"] != "DIAGNOSTIC"])),
        field2=dict(n_tiers=len(f2_sens),
                    zone_counts=zsum([r for r in rows if r["field"] == "field2"
                                      and r["A"] != "" and r["section"] != "DIAGNOSTIC" and
                                      r["section"] != "SENS_block63"])),
        stability=("**不稳定**：field4 全档 rho ∈ [0.9856, 4.8260] 跨 Z1↔Z2（4.0 边界）；"
                   "field2 旁证的 `pre5` 档 rho=5.246 落 Z3 ⇒ 「能落进 (4,5] 也能落出去（甚至 >5）」，"
                   "按预注册纪律**必须标不稳定**")),
    decision_map=F4["decision_map"],
    post_hoc_diagnostics=dict(
        registration="🔴 **未预注册**：在看到主档结果（rho=3.2847, Z1）之后追加，**仅作诊断、不改主档**；"
                     "见 docs/m11_rho_estimation.md §0.8",
        field4=PH,
        key_reading=("① 事件前局部基线中位 = 0.880，而与上升量的 Spearman = **-0.763** ⇒ "
                     "「拍幅 = thr − 事件前基线」这一**固定全局阈值几何**是低尾的主因；"
                     "② 4.0% 的上升量 < 1×MAD、35.9% < 2×MAD；"
                     "③ 剔除基线已抬升的事件后 A = 1.1325-1.3069 ⇒ rho = 1.208-1.394（Z1 中部）"),
        interpretation="⇒ 主档 A 的**下尾**部分由「检出时刻基线恰高」的几何效应贡献 ⇒ A 被压低 ⇒ rho 被抬高；"
                       "剔除该效应后 rho 反而**更小**（更深入 Z1）。故 `(4,5]` 内的档位主要来自"
                       "**最受该几何效应污染**的档（pre5 / τ=0.5 / 部分 63 帧块）。"),
    subwindow_blocks63=F4["subwindow_blocks63"],
    thr_aux_probe=F4["thr_aux_probe"],
    field2_side_evidence=dict(
        role="旁证（**非主口径**）；两条**已登记弱点**必须随行出现",
        registered_weaknesses=[
            "① **配准锚仅部分通过**：语义锚 verdict = 「FAIL／部分不通过 —— 见 per_metric（不得当作已配准）」"
            "（rank 1/25、领先 0.9%；另两指标 3/25、6/25）⇒ field2 **不得**当作「已配准」"
            "（出处 data/m12_b5_roi_mask/m12_b5_roi_mask_audit.json::fields.field2.semantic_anchor.verdict"
            "；data/m11_p5_targets.csv L116 note）",
            "② **其带（上界 0.229885）只建立在 87 个 ROI 上**（field4 为 432）"
            "（出处 data/m11_p5_targets.csv L116 + 审计 JSON fields.field2.real_roi_silence）"],
        primary=F2["primary"], selfcheck=F2["selfcheck"],
        sensibility_note="field2 的 31 个 63 帧块**全部** n_iso<30 ⇒ 全部 `unavailable`（87 ROI 不足以支撑 10 s 窗）",
        cross_field=dict(rho_field4=F4["primary"]["rho"], rho_field2=F2["primary"]["rho"],
                         ratio=F2["primary"]["rho"] / F4["primary"]["rho"])),
    limitations=[
        "① **真值脉冲不可得**（only-in-DB，R1/R2/R3）⇒ 无法用真值脉冲标定 A；A 是**事件幅度口径**的量，"
        "**不是**「单 spike 的 dF/F」实测。",
        "② **检出截断（censoring）**：凡 thr 上检出者，绝对峰必 > thr（实测 min=1.578701, ratio=1.00015）"
        "⇒ 绝对峰档 rho<1 **恒成立**（伪影）；主档（减基线）单侧偏大 ⇒ **rho_est <= rho_sim（弱下界性质）**"
        "⇒ **本路线给出的小 rho 不能排除 rho_sim > 5**。",
        "③ **固定全局阈值的几何效应**（后验诊断）：A 的 p10 与「检出时刻的基线抬升」强负相关"
        "（Spearman -0.763）⇒ 下尾部分是阈几何而非生理单脉冲幅度。",
        "④ **τ 无仓内实测依据**：主档 τ=1.0 s 出处是口径桥的**工程档**（m11_acceptance_master.md L5727），"
        "不是钙指示剂动力学；仓内实测 8.003-9.015 s / 29.68-30.48 s 是**群体状态 dwell**（S 门），"
        "只作扫描档 ⇒ τ 档一次变动即把 rho 从 4.39（0.5 s）带到 1.81（8.0 s）。",
        "⑤ **未改变检出阈值** ⇒ 低于 thr 的事件在本口径下不可观测；主档 A 条件于被检出。",
        "⑥ **神经域反演未做**（未反演的 dF/F，同 B-5 限制）。",
        "⑦ **掩膜↔数据库 Mask 表身份链未确认**；`Mask.weights` 未使用 ⇒ 等权均值（同 B-5 限制）。",
        "⑧ **样本范围**：单 field（session4_scan7_field4）× 2000 帧 = 317.46 s、432 完全落窗 ROI ⇒ "
        "外推范围受限（`data/m11_p5_targets.csv` L113 的过度泛化纪律）。",
        "⑨ **桥接假设未验证**：`rho_sim ≈ rho_data` 依赖口径桥按 R191 裁决②标定，本节点**未跑仿真、未验证**。",
        "⑩ **10 s 子窗档的样本门**：field4 有 17/31 个 63 帧块 n_iso<30 ⇒ 标 `unavailable`"
        "（**未测 ≠ 未命中**；不得当作 0 或「无孤立事件」）。",
    ],
    not_done=dict(ran_simulation=False, used_mps_or_gpu=False, used_network=False,
                  downloaded_papers=False, modified_any_existing_file=False,
                  new_files=["neural_exploration/docs/m11_rho_estimation.md",
                             "neural_exploration/data/m11_rho_estimation/rho_estimate.json",
                             "neural_exploration/data/m11_rho_estimation/rho_estimate.csv"],
                  no_b_verdict="**不替主 agent 下 (b) 的裁决**：本文件只报事实与算术（四态用词已区分）"),
)
with open(os.path.join(OUT, "rho_estimate.json"), "w") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
print("wrote rho_estimate.json / rho_estimate.csv；rows=%d" % len(rows))
print("field4 rho range: [%.6f, %.6f]" % (out["sensitivity_summary"]["field4"]["rho_min"],
                                         out["sensitivity_summary"]["field4"]["rho_max"]))
print("zone counts field4:", json.dumps(out["sensitivity_summary"]["field4"]["zone_counts"],
                                        ensure_ascii=False))
print("zone counts field2:", json.dumps(out["sensitivity_summary"]["field2"]["zone_counts"],
                                        ensure_ascii=False))
```
