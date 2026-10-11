# M11-B3 · G1（P5 生理工作点门）判据 / 协议 / 工具 设计说明书

> **节点**：M11 执行节点 **B3**（工作包 = **G1（生理工作点门，P5）的判据/协议设计与工具骨架**）
> **依据**：`docs/生物仿真M11实施清单.md` §0.5 / §0.7 / §0.9 / §4.5 / §7 P5 行 / §10.1 ·
> `docs/m11_list_review_master.md`（RF-M11-1..6）· `docs/m11_acceptance_master.md`（G2 判定 + R1–R7）·
> `docs/m11_data_gate_report.md`（B1 数据门）· **M10 教训**：`docs/m10_env_notes.md` L17/L21/L25/L26/L29/L36/L39/L40/L46/L51 ·
> `docs/m10_acceptance_master.md` A4/A8/A10/A12
> **日期**：2026-10-11 · **本工作包不做任何仿真**（离线评估器 + 判据表 + 设计）

---

## §1 交付物

| # | 文件 | 说明 |
|---|---|---|
| 1 | `data/m11_p5_targets.csv`（**94 行**） | **判据带 / 协议常数唯一定稿表**，逐行 provenance（PMID/DOI/数据文件::字段/清单行）；**只允许新增行**，`unavailable→measured` 升级一律走新增行 |
| 2 | `data/m11_p5_targets.json` | 同一内容的机器可读版 + provenance ledger（本轮文献核读原文摘录、实测复算、冲突登记、未定稿带清单） |
| 3 | `tools/validate_p11_resting.py` | **P5/G1 离线评估器**（只做分析，不做仿真）：接受注入的群体轨迹 / 逐神经元计数 / 聚合读数，输出 (a)–(i) + **四态** + 六项硬门 |
| 4 | `docs/m11_b3_g1_design.md`（本文） | 判据表、协议、标定杠杆顺序、统计方案、张力识别、时长依赖处置、失败/反证路径 |
| 5 | `docs/m11_b3_notes.md` | 环境/口径/冲突登记（L1 起）+ **跨节点 sha256（标注基准）** |
| 6 | `data/m11_b3_managed_jobs.json` | 需主agent 托管的作业（确切命令行 + 产物路径 + 估计墙钟） |
| 7 | `data/m11_p5_smoke_*`（**28 件**：5 协议 + 7 SYNTHETIC 计数 + 4 截断阶梯 + 12 输出） | 冒烟测试夹具与输出（S1–S6；M10 冻结件只读 + **SYNTHETIC** 计数夹具，文件内已标注） |

**未做（边界）**：不启动任何仿真/长作业；不碰 `src/mouse_cortex_*`（B2）、`data/m11_data_*` / `tools/fetch_m11_*` / `tools/build_m11_*`（B1）与 M0–M10 全部冻结件；不写判据带（工具只读带）。

---

## §2 判据表（`data/m11_p5_targets.csv`）——覆盖与可用性

**逐行 provenance 覆盖：94/94 行非空**（工具 `--check-targets` 机械校验：缺 provenance 即报错）。

> **修订轨迹（只允许新增行）**：72 行（首版，主agent 初审）→ **80 行**（R9/R10/R11 到位后）→ **94 行**（初审反馈后：口径分列 + 样本范围 + 哨兵 + 掩膜冲突 + Allen 参照）。
> 旧行的 `value / value_kind / evidence_level / provenance` **一律逐字保留**；仅两行的 `final` 由 1 置 0 并追加「R9 后状态」指针（`microns_activity_channel`、`rate_median_hz_measured_in_vivo_band`），已在 notes L11/L16 登记。

| 证据级别 | 行数 | 代表键 | 来源 |
|---|---|---|---|
| `literature` | **18** | `rate_median_hz_lo/hi`、**`rate_median_hz_literature_spike_band`**（电生理口径命名行）、`rate_median_hz_exc_*`、`rate_median_hz_inh_*`、`silent_frac_ca_active_per_wave_pct`(12%)、`unresponsive_neurons_pct_all_areas`(23%)、`proxy_steriade1993_upstate_duration_s`(0.8–1.5 s)、`proxy_sanzeni2020_isn`、`proxy_wehr2003_balanced_inhibition`、`proxy_monier2008_ei_diversity`、`proxy_buzsaki2014_lognormal_rates` 等 | PMID 30419013/40205214/19706480/31844315/8340806/11017176/32598278/14647382/17507551/18215425/24569488（**11 篇本轮逐条核读**） |
| `derived` | **52** | 全部门限/网格/口径常数 + **`calcium_caliber_nyquist_ceiling_hz`(3.150003 Hz)** / **`calcium_caliber_resolvable_floor_hz`(0.00315 Hz)** / **`calcium_vs_spike_caliber_gap`** + R9/R10/R11 门链与纪律登记 | 清单 §4.5/§0.9 草案行 + RF-M11-3 + R9/R10/R11 + `m10_calibration_targets.csv::字段` + **由 measured 帧率/样本范围推出的结构界** |
| `measured` | **11** | `frame_rate_hz_microns_functional`(6.300006 Hz)、`microns_activity_channel_voxel_series`、`microns_same_source_tau_band_ref`(8.003–9.015 s，**归属 S 门**)、**`microns_functional_calcium_sample_scope`**(2 field/2,000 帧/317.46 s)、**`microns_functional_sentinel_count`**(10,383) / **`_fraction`**(0.01298%)、**`microns_sentinel_mask_policy`**(未决冲突)、**`microns_calcium_measured_population_anchors`**、3 条 Allen in vitro `proxy_*` | `m11_microns_protocol.json::frame_rate_hz`；`m11_reference.csv::R-01/R-10..R-14/R-18`；`_mainagent_m11_same_source_band.csv`；`m11_allen_celltypes_metadata.csv::ef__avg_firing_rate`（本节点独立复算） |
| `literature_unverified` | **1** | `n_neuron_gear_full`=75909 | 清单 §0.9（摘要只给 'around 75,000'，冲突 L6） |
| `unavailable` | **12** | **(a) 电生理实测带**、**(a) 钙口径带 lo/hi**、`silent_frac_lo/hi`、`silent_frac_ephys_lo/hi`、`ei_ratio_measured`、`single_trial_wall_s_max`、`microns_activity_channel`（原行）、`microns_activity_channel_percell`、`allen_bo_neuropixels_spike_rate_band` | B1 未交付项 + R1（逐细胞 only-in-DB）+ `R-21` unverified + G0 探针未回 + Allen NWP 仅目录 |

**`final=0`（未定稿 ⇒ 依赖它的子判据强制 `unavailable`）：8 行** ——
`(a)` 的**实测**带、`(b)` 的钙域/电生理域两个带、`(e)` 的实测量、`(g)` 的墙钟上限。
⇒ **当前状态下 (b) 与 (g) 必判 `unavailable`；G1 因此必然不能判 `pass`**（这是本工作包的**诚实结论**，不是缺陷）。

### §2.1 跨制备 / 跨口径代理 与 越界警示（明确登记 + 量化后果）

| 代理行 / 警示行 | 制备 | 量化后果（`wrong_usage_consequence`） |
|---|---|---|
| `proxy_allen_celltypes_avg_firing_rate_median_all/visp` | Allen Cell Types **in vitro 电流钳**（非在体、非自发） | 中位 **17.41 / 17.63 Hz** vs 本门文献带 0.5–2.0 Hz ⇒ **高 8.7–34.8×**；且其 `frac<0.5 Hz = 0.0000` ⇒ 静默带被压到 ≈[0,0.02]，**必然与 (a) 冲突**（M10 A4 标定权衡的正体） |
| `proxy_sanchezvives2000_ferret_slice` | **雪貂皮层切片 in vitro**，0.1–0.5 Hz | 清单 §1.5 C3 把它当「小鼠皮层 in vivo dwell 0.3–3 s」= **物种/制备/数值三重误引**（冲突 L5） |
| `proxy_steriade1993_upstate_duration_s` | 猫**在体**、氨基甲酸乙酯麻醉 | up 态 0.8–1.5 s；跨物种 + 跨麻醉 ⇒ 仅量级代理；且 **P5 不判 dwell（越界）** |
| `proxy_sanzeni2020_isn` / `proxy_wehr2003_balanced_inhibition` / `proxy_monier2008_ei_diversity` | 小鼠 V1 清醒/轻麻醉（ISN）· **大鼠听皮层**在体 · 麻醉猫 V1 | 仅支撑 E/I 的**方向与量级**（抑制不弱于兴奋、E/I 同量级、个体多样 ⇒ 带不得窄）；**不能**支撑任何数值端点 |
| `silent_frac_ca_active_per_wave_pct` / `unresponsive_neurons_pct_all_areas` | 小鼠 V1 2p（**每次自发波**招募口径）· Allen 2p（**对刺激可靠响应**口径） | 都不是「T 窗内零计数比例」；直接搬用 ⇒ 静默带口径错（12% 活跃被读成静默 0.12；23% 未响应被读成静默 0.23） |
| `rate_median_hz_m10_fish_band` / `silent_frac_m10_fish_band` | 斑马鱼幼鱼（M10） | RF-M11-1 禁移植；若移植，M10 的「中位率 0.000 Hz + 静默 0.920」会被判成 (b) 通过 |
| ⚠️ **越界警示**：`microns_same_source_tau_band_ref`（8.003–9.015 s） | MICrONS **同源** 2 field × 2,000 帧 @6.300006 Hz（**measured**，非代理） | **归属 S 门（P6）/RF-M11-2**；在 P5/G1 内引用它判任何子判据 ⇒ **越界**；且 Q 法低态 dwell 1.11–1.27 s vs H 法 3.17–6.67 s（**2.6–6×**）⇒ 口径差必须并列 |

### §2.3 ★ 口径分列（主agent 初审反馈落地）：**钙口径 ≠ 电生理口径**

| 口径 | 行 | 状态 | 值 |
|---|---|---|---|
| **电生理（spike rate）** | `rate_median_hz_literature_spike_band`（命名行）+ `rate_median_hz_lo/hi` + 类级 exc/inh 4 行 | `literature` / **已定稿** | 0.5–2.0 Hz（E 0.5–1.0 / I 1.5–2.0） |
| **钙（MICrONS 同源）** | `rate_median_hz_measured_calcium_band_lo` / `_hi` | **`unavailable`**（`final=0`） | 待「ROI 掩膜 × 帧序列 → 逐 ROI ΔF/F → 事件检测 → 事件率中位」管线（**尚未建立**；J2） |
| **钙的结构界**（由 measured 帧率/样本范围推出） | `calcium_caliber_nyquist_ceiling_hz` = **3.150003 Hz**；`calcium_caliber_resolvable_floor_hz` = **0.00315 Hz** | `derived` / 已定稿 | 上界 = 帧率/2；下界 = 1/317.46 s |
| **跨口径混用的后果** | `calcium_vs_spike_caliber_gap` | `derived` / 已定稿 | 方向：**中位率低估**（上限压缩到 3.15 Hz，spike 带上界 2.0 Hz 已达其 63.5%）+ **静默比例高估**（下限比 0.5 Hz 低 **159×**）；**幅度未测 ⇒ 不给倍数** |
| **样本范围** | `microns_functional_calcium_sample_scope` | `measured` / 已定稿 | **2 个 field（144 中）+ 每 field 2,000 帧 ≈ 317.46 s**；单 field 完整记录 40,000 帧/6,349.2 s（未全取） |
| **哨兵/缺失** | `microns_functional_sentinel_count`=**10,383**、`_fraction`=**0.01298%** | `measured` / 已定稿 | 两侧计数一致；**未处理 ≠ 无缺失** |
| **哨兵处置** | `microns_sentinel_mask_policy` | **`unresolved_conflict`（`final=0`）** | 位置 A（主agent 指令）= 前置为缺失；位置 B（§3f-3 / `R-14`，B1 证据）= **无掩膜、保留并登记**。**影响上界 ≤0.0130 个百分点 ⇒ 不阻塞 G1**（notes L16） |
| **Allen 2p（参照）** | `allen_2p_role_in_m11`=cross_preparation_reference + `unresponsive_neurons_pct_all_areas`(23%) + `allen_bo_neuropixels_spike_rate_band`(`unavailable`) | `derived`/`literature`/`unavailable` | 按 R6/R9 **只作跨制备参照**，不是目标带来源 |

**诚实结论**：R9 让 MICrONS 钙序列**可解码（measured）**，但 **(a) 的钙口径带仍不可得**——「可解码」≠「已重建出率口径」。已测的钙域「率」只有**群体状态转换率**（0.104–0.438 Hz，`microns_calcium_measured_population_anchors`），**不是逐细胞率**，且**归属 S 门**（P5 内引用即越界）。

### §2.2 R9 / R10 / R11 到位后的更正（跨节点并发，如实登记）

| 项 | 内容 | 对本表的影响 |
|---|---|---|
| **R9**（反转） | MICrONS 功能**体素序列可解码**（大端 int16；200/200 对象通过；掩膜规则=无掩膜）；R2 的 `format_unverified` 被推翻；新增纪律「判不可解码前必须覆盖 dtype×字节序×排列×差分编码」 | 本表**首版按 R2 写 `microns_activity_channel = unavailable` ⇒ 已更正**为 `microns_activity_channel_voxel_series = measured` + `microns_activity_channel_percell = unavailable`（R1 保持）。本节点首读的 `m11_acceptance_master.md` 是 95 行的 R2 版本（21:46 被主agent 更新为 185 行的 R9/R10/R11 版）⇒ 见 notes **L11** |
| **R10** | 主agent 独立建出**同源目标带**：原生帧未平滑 τ ≈ **8–9 s**（frame_mean 9.015 s / high_pixel_frac 8.003 s，n=2000）；S 门目标尺度定为 8–9 s；**Allen 2p 降为跨制备参照** | 新增 `microns_same_source_tau_band_ref`（measured，**归属 S 门**）+ `allen_2p_role_downgraded`。(a) 的当前 literature 锚**不改**（它来自小鼠 V1 胞外电生理文献，不是 Allen 图像） |
| **R11** | **G0 引擎门验收裁定 = fail**（(e) 分段可组合性 fail、(f) 同步发放判别 fail；均为 M11 扩展引入的**对 M10 的回归**）⇒ **G1/S 不得启动；M11-B3 的 G1 工作停在设计/工具态** | 新增 `g0_engine_gate_status_fail` + `g1_start_precondition_all`。**本工作包的全部判定在 G0 全项复测通过之前不得发布**（工具与判据表处于待命态） |

---

## §3 协议（预注册）

| 项 | 值 | 依据 |
|---|---|---|
| dt | 0.05 ms | G0 定稿（`DT_MS`） |
| settle 窗 | 500 ms（测量窗起点必须清零并断言） | 清单 §4.5 + M5 L37#2 + M10 L39.1 |
| **T** | `T_min = 10 s` 起；**必须给出 {T_min, 2×T_min, 4×T_min} 三档** | RF-M11-3 + M10 A12（时长依赖实测） |
| seed | 零驱动臂与基准臂**各 ≥3**，且**零驱动臂总脉冲 ≥300** | RF-M11-3（1/√300 = 5.8% ≤ 6%） |
| 剂量网格 | `{0, 0.5, 1, 2, 4, 8}×`（0 档 = 零驱动臂） | 清单 §4.5(d) |
| 主口径（estimator） | **逐神经元**（`rate_median_hz`、`silent_frac = 计数==0 的比例`） | 清单 §4.5(a)(b) + M10 定义 |
| 并列口径 | 群体轨迹逐箱（箱宽 100 ms；`silent_bin_frac`）—— **只作描述性交叉核对** | M10 `bout_stats(bin_ms=100)` |
| 阶梯硬约束 | **必须逐档独立重跑**（含 settle/初值/RNG）；**禁止用同一轨迹截断构造阶梯**（工具含截断守卫） | 本工作包实测（notes L8） |
| 分箱 / 平滑 | 100 ms / 5 帧（与 S 门口径对齐；P5 内不判慢状态） | `m11_acceptance_master.md` R6/R7 |
| 每档 N | 8k / 31k / 75,909（N_full 精确值待定稿，冲突 L6） | 清单 §0.9 |

**输出口径要求（⑥）**：每条评估行都带
`protocol_id / protocol_T_ms / protocol_settle_ms / protocol_seed / protocol_dt_ms / protocol_bin_ms / protocol_smooth_frames`
与 `estimator`，并带 `is_primary_estimator` 以区分主判定与描述性交叉核对。

---

## §4 六项硬门如何堵死 M10 的三类判据缺陷

> **G1 门组成（主agent 2026-10-11 §二.6 裁决）**：**G1 = (c) ∧ (e) ∧ 张力已定量识别**；**(d) 降为 sanity**。
> 落地：判据表新增 `gate_g1_conditions_v2`（旧行 `gate_g1_conditions` 保留 + final=0 + 指针）；工具 `GATE_CRITERIA = (c, e, p5_g1_tension_quantified)`，
> 并加 **`_assert_gate_composition()` 断言**：表声明与代码不一致即 `raise`（负向测试实测触发：「门组成漂移」）。
> 新子判据 `p5_g1_tension_quantified`：剂量-响应 ≥3 点 **且** (a)(b) 两条带均已定稿 ⇒ `pass`（给 `n_both_in_band` + 两条 Spearman）；否则 `unavailable`。
> **`pass` 只表示「张力已被定量」，不表示「张力不存在」**——张力存在时按清单 §4.5(i) 登记为 P6 的输入约束（不得据此判 G1 fail，也不得放宽带）。

| M10 缺陷 | 本工作包的机械堵法（工具内已实现并有冒烟证据） |
|---|---|
| **① 门 1 不是统计可判**（3 seed = 0.0987/0.1056/0.1005、R=0.1015、CI95 [0.0945,0.1091] 跨门限；零驱动臂仅 60 脉冲 ⇒ 相对不确定度 13%） | **硬门①样本量门**：零驱动/基准臂各 ≥3 seed（去重）+ 零驱动总脉冲 ≥300；不满足 ⇒ `unavailable`（**不得判 pass，也不得判 fail**），点估计退为描述性。**硬门②CI 门**：主法泊松对数正态（`se(log R)=√(1/S₀+1/S₁)`，与 M10 实现**逐位一致**）+ 次法 seed 重采样×泊松自助，两法上界相对差 >0.25 ⇒ `invalid`；判定 =「点估计 ≤ 上限 **且** 更宽的 CI 上界 ≤ 上限×1.2」。<br>**冒烟证据（S2）**：注入 M10 `m10_p5_gate1_ci.json` 的逐 seed 脉冲数 ⇒ 工具给出 R=0.1015384615、se_log=0.0365403631、逐 seed 比值**与 M10 逐位一致**，判 **fail** 并标注「统计未分辨（点估计未过，但 CI 上界未越过 ×1.2 带）」。 |
| **② 门 1/2 对机制阶梯无判别性**（基线自身即过：+579.6%） | **硬门⑤判别性断言**：每轮必须给出「**无机制基线臂**（`mechanism=none`、`drive_mult=1`）」的同一判据值；缺 ⇒ 整体 `unavailable`（不得 pass）。基线自身达标 ⇒ 该判据标 `discriminative=false`。并且判据表登记 `criterion_d_as_mechanism_evidence=0`（`(d)` **不得**作为机制有效性证据）⇒ 工具的 (d) 行永远带 `evidence_class=sanity_precondition(non_discriminative)`，且 `non_discriminative_criteria` 列表在总判定里单独输出。<br>**新增判别性来源**：`(a)` 改为**类级带**（E 0.5–1.0 Hz / I 1.5–2.0 Hz，**I > E 的方向**），`(e)` 给带 —— 二者都是基线**不必然**通过的判据。 |
| **③ 判据的时长依赖性**（T=2 s→0.0980、T=60 s→0.1082、T=120 s→**0.1720**，+76%；λ×4 中位率随 T 由 0→1.0 Hz） | **硬门④时长阶梯**：同一配置必须给 {T_min, 2T_min, 4T_min} 三档的**同一判据值**，并把漂移做成**被测量的量**（`rel_change_first_to_last`、`spearman_vs_T`、`status_flip`）。跨 T 判定翻转 ⇒ 该子判据 `invalid`（协议时长绑定失败）。**另加截断守卫**：若短时长臂的轨迹是长时长臂的逐位前缀 ⇒ 阶梯判 `invalid`（`ladder_arms_look_truncated`），因为截断测的是窗口长度而非协议时长。缺阶梯 ⇒ 阶梯 `unavailable` 且**该判据不得定稿**。<br>判定结果永远携带其 `protocol_T_ms`（M10 的「T=2 s 窗下通过」限定被机械化）。 |
| 另：**四次口径事故**（×1.3004 / ×1.15 / 43× / 无断言） | **硬门③双估计器一致性**：`counts` 求和 vs `pop_trace` 求和 vs 外部聚合读数，两两相对差 >2% ⇒ **整轮 `invalid`**（不是 fail）；只有单一估计器 ⇒ `unavailable` 且阻断 pass；并加**非平凡量纲自检**：轨迹长度 ⟺ `round(T_ms/dt_ms)`（**该断言正是 ×1.3004 的同族缺陷探测器**）、计数长度 ⟺ N。<br>**冒烟证据（S3/S4）**：正常臂 `rel_diff = 0.0`；把基准臂计数人为放大 1.15× ⇒ `rel_diff = 0.1304 > 0.02` ⇒ 整轮 `invalid`、exit=2。 |
| 另：**判据带未定稿却照判** | **带定稿机械强制**：带值全部读自 `m11_p5_targets.csv`；`final=0` 或 `evidence_level=unavailable` ⇒ 该子判据 `unavailable`（**拒绝用未定稿带判 pass/fail**）。 |
| 另：**"未测" 被写成 pass/fail** | 四态 + 退出码：`pass`=0 / `fail`=1 / `invalid`=2 / `unavailable`=3；`invalid` 与 `unavailable` **永不计入** pass/fail（清单 §0.7 #6）。 |

### §4.1 六条硬门的**实现与断言状态**（主agent 要求逐条确认）

| 硬门 | 实现位置 | 断言形式 | 冒烟证据 |
|---|---|---|---|
| **① 样本量门** | `Evaluator._eval_selfsustain`（seed 去重 + `zero_spikes` 汇总） | **门条件判定**（不 raise）：`gate_fail` 非空 ⇒ `status=unavailable`，点估计保留为描述性 | S3：每臂 1 seed ⇒ `sample_size_gate_unmet: 零驱动臂 seed 数 1 < 3; …`；S5：T=2 s 臂 `零驱动脉冲 19 < 300` |
| **② CI 门** | `selfsustain_ci()` + `_eval_selfsustain` | 主/次两法并列；`ci_upper_rel_diff > tol(0.25)` ⇒ `status=invalid`；判定式 = `R ≤ max 且 max(两法上界) ≤ max×1.2` | **S2**：R=0.1015384615、se_log=0.0365403631、逐 seed 比值与 `m10_p5_gate1_ci.json` **逐位一致**；两法上界相对差 0.0054 ⇒ 一致；判 `fail` |
| **③ 双估计器断言** | `arm_metrics()`（三估计器两两相对差 + 量纲自检） | **硬失败**：`rel_diff > 0.02` ⇒ 进 `hard_failures` ⇒ 总判定 `invalid`（**不 raise**，以保证输出落盘——M10 用 `assert` 直接抛异常反而丢了整轮证据）；单一估计器 ⇒ `unavailable` 且**阻断 pass**；另断言 `trajectory_len == round(T_ms/dt_ms)`、`counts_len == n_neuron` | **S3**：6/6 臂 `rel_diff = 0.0`；**S4**：`rel_diff = 0.1304 > 0.02` ⇒ `invalid`、exit=2 |
| **④ 时长三档** | `_duration_ladder()` + `_apply_duration_flip()` | 缺档 ⇒ `unavailable`；**截断守卫**（短轨迹是长轨迹逐位前缀）⇒ `invalid` 且并入 `hard_failures`；跨 T `status_flip` ⇒ 该子判据 `invalid` | **S5**：`ladder_arms_look_truncated` 触发 ⇒ `invalid`、exit=2；S1/S3/S4 给出 `T_levels_present_ms`/`ladder_required_ms` 与 `unavailable` 原因 |
| **⑤ 判别性断言** | `_baseline_arm_value()` + `_verdict()` + 判据表 `criterion_d_as_mechanism_evidence` | 缺「无机制基线臂」⇒ 整体 `unavailable`（不得 pass）；基线达标 ⇒ `discriminative=false`；(d) 恒标 `sanity_precondition(non_discriminative)` 并进 `non_discriminative_criteria` | S1：`baseline_arm_registered=False` ⇒ 阻断 pass；S3/S4：`baseline_arm_registered=True` 且 `non_discriminative_criteria=['p5_d_drive_elasticity']` |
| **⑥ 口径标注** | `Evaluator._proto_fields()`（每行 7 个 protocol 字段 + estimator）**+ `_assert_output_invariants()` 真断言** | **raise AssertionError**（三条）：(i) 每行必须带 7 个口径字段 + `estimator` + `protocol_T_ms`；(ii) `pass` 的带类判据必须有非空 `band_lo`（a/b/e 还需 `band_hi`）与 `value`（**未定稿带不得判 pass**）；(iii) 每个评估单元必须出现**全部 9 条子判据**（防「判据静默丢失」，与 M10 编码路径静默失效同族） | 断言在实现首跑即**抓到一处真缺口**（无臂行缺 `protocol_T_ms` ⇒ 修 `_proto_fields` 回退到单元 T）——这正是断言价值的证据；最终 S1–S5 全部输出 `output_invariants = {'rows': …, 'units': …, 'criteria_per_unit': [9], 'caliber_fields_checked': 8}` |

---

## §5 标定杠杆顺序（**含「顺序受自持判据钳制」**）

M10 A4 的正体是「(a) 中位率 ∧ (b) 静默 同时成立」未达成，而抬高驱动去命中 (a) 会同时把静默压出带（M10 修正档：λ×8 才有中位率 4.0 Hz，其静默 0.131 **低于**带下界 0.5）。
⇒ M11 **不得**用「提高 λ」当第一杠杆，且**顺序受 (c) 自持判据钳制**：递归增益/驱动谱的每一次上抬都必须重新过 (c)（新增递归增益会在零驱动臂里同样放大自持），因此顺序必须是「**先压自持 → 再调工作点 → 最后换杠杆**」：

| 序 | 杠杆 | 作用点 | 为什么在这个位置 | 钳制条件（不过则回退） |
|---|---|---|---|---|
| **L0** | **压自持**：抑制权重/亚型比例、阈值分布、E/I 相对强度 | 递归环路增益 | **必须最先做**：任何提高率的手段都会同时抬高零驱动自持比（M10 indegree=5 稀疏臂自持比 0.3078 即反例） | (c) 带 CI 通过（点估计 ≤0.10 **且** 更宽 CI 上界 ≤0.12）；否则不得进入 L1 |
| **L1** | **异质阈值 / 异质适应**（细胞间 σ、适应电流的多时间尺度） | 单细胞增益分布 | 在不改递归增益的前提下把**中位率**推入带，且因按细胞异质而**抬高静默比例的尾部**（同时服务 (b)） | 每次改动后重跑 (c)（零驱动臂）；(c) 一经跨线即回退上一档 |
| **L2** | **E/I 比例与真实亚型分布**（PV/SST/VIP 三类槽的权重与连接概率） | 抑制/兴奋电流比 | 直接对准 (e)；同时改变 (a)/(b) 的联合可达域（抑制主导会把率压到低段、把静默抬到高段） | (c) ∧ (e) 同时成立；若 (a)∧(b) 仍不可同时成立 ⇒ 进 L3 |
| **L3** | **慢变量提供分级响应**（而非提高 λ） | 增益/阈值的慢状态调制（C-7 类；**必须三联动**：增益+抑制权重+阈值） | 只在 L0–L2 已把静默与自持锁住时使用；**它本身是 P6/S 门对象，P5 内只作为「分级响应」手段** | (c) 必须仍过；且必须登记「机制作用域 + 最小阳性对照」（M10 L51 协议：仅缩放输入幅度 = 未实施） |
| **禁** | **提高背景驱动 λ** | — | 提高 λ 同时抬高零驱动臂（若事件表泄漏）、压低静默 ⇒ 必然触发张力 | 若确需 λ 扫描，只能作为**剂量-响应诊断**（本工具的张力块），**不得**作为标定手段 |

**张力识别（工具已实现）**：对每个 (config, T) 的剂量-响应点集 (驱动 × 中位率 × 静默)，输出
`n_both_in_band`、`spearman_drive_vs_rate`、`spearman_drive_vs_silent`、`qualitative_monotone_opposition`、
`structural_tension`（**只在 (a)(b) 两条带都已定稿时才可判 True**，否则明示「张力只能定性」——防构造性结论）。
冒烟证据（S3）：3 个剂量点上 `ρ(驱动,中位率)=+1.0`、`ρ(驱动,静默)=−1.0`、`n_both_in_band=0`（因 (b) 带未定稿，故仅给 `qualitative_monotone_opposition=true`）。

---

## §6 统计方案

1. **主统计量为中位数**（清单 §4.5(a)「中位率」）：发放率为重尾/对数正态（PMID 24569488）⇒ 均值受尾部支配（Allen in vitro 实测：均值 32.48 vs 中位 17.63 Hz，**1.84×**）。
2. **(c) 自持比**：`R = Σ 零驱动脉冲 / Σ 基准脉冲`（同 T 下等价于率比）。
   - 主法 CI：`se(log R) = √(1/S₀ + 1/S₁)`，`CI95 = R·exp(±z₀.₉₇₅·se)`（**z 取精确分位 1.959963984540054**）。
     > **口径注（notes L2）**：M10 的实现用 z=1.96，故两者 CI 末端相对差 **≈1.3e-6**（R 与 se_log 逐位一致）。
   - 次法 CI：seed 重采样 × 泊松扰动的参数自助（B≥10⁴）。
   - **取两者中更宽的 CI 上界**作为判定依据（保守）；两法上界相对差 >0.25 ⇒ `invalid`。
3. **判定式（预注册）**：`点估计 ≤ selfsustain_ratio_max` **且** `更宽CI上界 ≤ selfsustain_ratio_max × selfsustain_ci_upper_factor`。
4. **类级判据**：若 counts 带 `cell_type` 列，逐类输出 `rate_median_hz` 与 `silent_frac_zero_count`（服务 P8 的类间方向：I > E）。
5. **(e) E/I**：`i_inh_integral / i_exc_integral`（测量窗内逐槽积分后求和）——**只接受仿真侧注入**；工具**不从发放率反推电流**（防口径臆造）。
6. **不做的统计**：不做逐突触拟合、不做多重比较校正后的"扫到即宣布"（清单 §0.7 #11 与 §4.1 候选纪律）。

---

## §7 失败 / 反证路径（本门的收口方式）

| 情形 | 工具输出 | 后续处置（交主agent） |
|---|---|---|
| (a) 落带而 (b) 不落带（或反之） | `(i)=fail` + 张力块给出「无一剂量同时落带」 | 按 §5 换杠杆（L1→L2→L3）；**不得**静默放宽任一带 |
| (b)/(g) 的带未定稿 | `unavailable`（带行 `final=0`） | 先由**托管作业 J1/J2** 产出实测带并**新增行**（不改原行）后重评 |
| 零驱动臂脉冲 <300 或 seed <3 | `(c)=unavailable`（点估计保留为描述性） | 延长 T 或补 seed（>10 min 作业交主agent 托管） |
| 跨 T 判定翻转 | `(c)=invalid` + 漂移数字 | 判据与其协议时长绑定；若必须选一档，按**最长 T** 定稿并登记限定 |
| 阶梯由截断构造 | 阶梯 `invalid`（`ladder_arms_look_truncated`） | 逐档独立重跑 |
| 双估计器不一致（>2%） | **整轮 `invalid`**、exit=2 | 先查口径（测量窗/时长/N/估计器实现），**不得**在 invalid 轮上做任何科学结论（M10 四次事故规程） |
| (e) 无注入 | `(e)=unavailable` | 由 B2 的 G0 槽级断言 + P5 运行注入电流积分 |
| **全部门条件不可达** | `G1=unavailable`/`fail` | **P6 不得启动**（清单 §0.5 门链）；登记为 P6 的输入约束（清单 §4.5(i)） |
| **G0 = fail（R11，当前状态）** | 工具与判据表**待命**（不发布任何 P5 判定） | 门链：G0 → G1 → S；**G0 未全项过 ⇒ G1 不得启动**（R11 明示本工作包「停在设计/工具态」）。修复 (e)(f) + 补跑 (a)–(d)(g)(h) 与全部 extensions 后，本工具可直接消费其注入量 |

---

## §8 复现命令

```bash
cd /Users/weidong/ai/small_world
export PYTHONPATH=/Users/weidong/ai/small_world OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
PY=./.venv-m11/bin/python

# 判据表自检（CSV⟺JSON 互查 + provenance 完整性 + 代理行后果齐全）
$PY -m neural_exploration.tools.validate_p11_resting --check-targets

# 冒烟 S1–S6（全部 <10 s；只用 M10 冻结轨迹 + SYNTHETIC 计数夹具）
for s in S1 S2 S3 S4 S5; do
  $PY -m neural_exploration.tools.validate_p11_resting \
      --protocol-json neural_exploration/data/m11_p5_smoke_protocol_$s.json \
      --out-csv neural_exploration/data/m11_p5_smoke_out_$s.csv \
      --out-json neural_exploration/data/m11_p5_smoke_out_$s.json --n-boot 8000
done
# 单臂 CLI 模式
$PY -m neural_exploration.tools.validate_p11_resting \
    --pop-trace neural_exploration/data/m10_p6_poptraces/poptrace_M0_lamx1.0_n250.npy \
    --counts neural_exploration/data/m11_p5_smoke_counts_M0_lamx1.0_base.csv \
    --dt-ms 0.05 --T-ms 120000 --n-neuron 250 --role baseline --config-id CLI_M0

# 夹具生成（SYNTHETIC，仅用于测硬门③）
$PY -m neural_exploration.tools.validate_p11_resting --smoke-emit-counts OUT.csv \
    --pop-trace <trace.npy> --T-ms 120000 --dt-ms 0.05 --n-neuron 250 [--smoke-scale 1.15]
```

> **冒烟结论（S1–S6）**：S1 `unavailable`/3 · S2 `fail`/1（CI 与 M10 逐位一致）· S3 `unavailable`/3（22 行、四态齐、张力与判别性均输出）· S4 **`invalid`/2**（口径事故被抓）· S5 **`invalid`/2**（截断阶梯被抓）· S6 单臂 CLI `unavailable`/3。**M10 数值与合成计数一律不得作为 M11 判据。**

---

## §9 G0 修好后的 G1 执行计划（协议矩阵 + 成本估算；**先探针 → 裁决 → 再烧**）

> **前置（硬约束）**：G0 全项 measured 且全过（R11 当前 = fail）⇒ 门链 §0.5 要求 G1 不得启动。本节是**待命方案**，
> 所有数字以 **G0 探针实测的 ms/step** 替换后才能烧预算（清单 §0.9 纪律 7）。
> **成本公式（本节点的修正版）**：`GPU-h = Σ_runs (steps × ms_per_step) / 3.6e6`（1 h = 3.6e6 ms）。
> ⚠️ 清单 §10.1 写的常数 `3.6e-6` **偏大 12.96×**（与同节「T=30 s/600,000 步 ⇒ N_mid 0.05–0.15 GPU-h」自相矛盾；
> 用 0.3 ms/step 代入得 0.648 h，而物理正确值 0.05 h）⇒ 见 notes L19。**预算一律用 1/3.6e6。**

### §9.1 协议矩阵（预注册）

| 阶段 | N | T 档 | 剂量网格 | seed | run 数 | 目的 / 交付 |
|---|---|---|---|---|---|---|
| **A 探针** | N_mid=31k | {10, 20, 40} s | {0,0.5,1,2,4,8} | 1 | **18** | 实测稳态 ms/step 与 RSS；**实测零驱动臂脉冲数 ⇒ 定 T_min**；首读 (a)(b)(c)、(f) |
| **B 杠杆** | N_small=8k | {T_min,2T_min,4T_min} | 同 | 1 | **18/轮** | 按 §5 顺序 L0→L1→L2→L3 **逐杠杆定向**（便宜档）；每次改动**重跑零驱动臂重核 (c)** |
| **C 定稿** | N_mid=31k | 同 | 同 | **3** | **54** | (a)(b)(c)(d)(e) 正式判定 + CI + 判别性 + 张力 + 时长阶梯 |
| **D 确定性** | N_mid | 3 档 | 选定配置 | 1（×2 重复） | **6** | (h) 统计级一致 |
| **E 条件** | N_full=75,909 | S 门关键条件 | — | — | ≤6 条件×3 试次 | **仅当** C 阶段证明「状态结构随 N 显著变化」才进（RF-M11-6 默认档 = 8k/31k） |

- **T_min 判定规则（预注册）**：`T_min` = 使**零驱动臂总脉冲 ≥300** 的最小档（RF-M11-3）；A 阶段实测后定稿，阶梯仍为 `1|2|4`。
  **T=120 s 不进 G1**（清单 §0.9：该类长协议只允许在 S 门 ≤6 个关键条件上跑）。
- **每个 run 必须落**：`pop_trace`（逐步群体脉冲数 .npy）、`counts`（逐神经元计数 .csv，**含 `cell_type`** 以支持类级带）、
  `ei.{i_exc_integral,i_inh_integral}`（(e) 的唯一来源）、`wall_s`/`ms_per_step`/`device`、`settle_ms` 与**事件表 seed**。
- **纪律**：逐 run **新进程**（显式 venv）；`OMP_NUM_THREADS=1`；MPS **串行**；段间记 RSS/ms-step；判进度看**累计 CPU/墙钟比**。

### §9.2 成本估算（N_mid 取 0.6 ms/step 中位；括号为 0.3 / 0.9 上下界）

单 run：T=10 s→200k 步 **0.033 (0.017–0.050) h**；T=20 s→400k 步 **0.067 (0.033–0.100) h**；T=40 s→800k 步 **0.133 (0.067–0.200) h**。

| 阶段 | 计算量 | GPU-h（中位） | 上下界 |
|---|---|---|---|
| A 探针（18 run） | 8.4M 步 | **1.4** | 0.7 – 2.1 |
| B 杠杆（4 轮 × 18 run，N_small 0.3 ms/step） | 33.6M 步 | **2.8** | 1.4 – 4.2 |
| C 定稿（54 run） | 25.2M 步 | **4.2** | 2.1 – 6.3 |
| D 确定性（6 run） | 2.8M 步 | **0.47** | 0.23 – 0.70 |
| **合计** | 70M 步 | **≈8.9 GPU-h** | **4.5 – 13.4** |

⇒ **落在 §10.1 的 G1 包络（≤10 GPU-h）内**（中位 8.9）；**上界 13.4 超包络** ⇒ 触发 **NG 门三态裁决**（降 N / 降试次 / 降 T 档），
**不得静默超支**。若强取 N_full：C 阶段单段即 4.2→4.2–12.6 h ⇒ **整体出包络** ⇒ 只能按 §9.1 E 行做「关键条件对照」。
- **墙钟**：MPS 串行 ⇒ 纯计算 ≈8.9 h；＋逐 run 新进程开销（31k 网络装配，估 30–60 s × 78 run ≈ 0.7–1.3 h）⇒ **≈10–14 h**（对接 §10.1「5–8 天」含分析与裁决，合理）。**若占空比异常（M10 实测曾低至 10%）以累计 CPU/墙钟比判进度、不误杀**。
- **磁盘**：pop_trace 78 run × ≤3.2 MB ≈ 250 MB；counts 31k 行/run × 78 ≈ 2.4M 行 ≈ 60 MB；合计 **<0.5 GB**。
- **内存**：M10 锚 100k 神经元/2M 边 = 2.2 GB ⇒ N_mid 线性 ≈0.68 GB，× M11 机制开销 1.5–2.5 ⇒ **≈1.0–1.7 GB/进程**（须在 A 阶段实测）。
- **CPU 兜底**：仅限对齐与探针（清单 §0.5 NG 门），不承担全规模 G1。

### §9.3 先探针 → 裁决 → 再烧 的检查点（每个都要落盘）

1. **A→B 检查点**：`ms/step` 与 RSS 落盘；零驱动臂脉冲数决定 T_min；若 (c) 在 A 阶段已超门限 ⇒ **先做 L0（压自持）**，不得进 B 的其他杠杆（§5 的钳制顺序）。
2. **B→C 检查点**：(a)∧(b) 是否在同一杠杆档同时可达 —— 若 B 的四轮都不可达 ⇒ **登记"标定权衡"并交主agent 裁决**（不得静默放宽带，清单 §4.5(i)）。
3. **C→D 检查点**：跨 T 判定是否翻转（翻转 ⇒ `invalid`，需重定 T 档）；张力块的 `n_both_in_band` 是否为 0。
4. **任意阶段**：`hard_failures` 非空 ⇒ 立即停（整轮 `invalid`），先查口径再续跑。

---

## §10 **探针级**算力方案（先探针 → 裁决 → 再烧；本节点不真跑）

> 目的：用**最小花费**换取「把预注册假设换成实测值」的三件事——`ms/step`、零驱动臂脉冲数（定 `T_min`）、(a)(b)(c) 首读。
> **成本公式**：`GPU-h = Σ steps × ms_per_step / 3.6e6`（清单 §10.1 的 `3.6e-6` 偏大 12.96×，见 notes L19）。
> 预注册锚（清单 §0.9，**待探针替换**）：N_small 0.15–0.45、N_mid 0.30–0.90、N_full 0.60–1.80 ms/step。

### §10.1 三档探针（可**逐档裁决**，每档都便宜到能"错了就停"）

| 档 | 内容 | run 数 | 步数 | GPU-h（0.6 ms/step @31k） | 墙钟（MPS 串行） | **裁决产出** |
|---|---|---|---|---|---|---|
| **P0 最小探针** | N=31k，T=10 s，dose {0,1,4}，seed 0 | **3** | 0.6M | **0.10** | ≈5–10 min | `ms/step` + RSS + **零驱动臂 10 s 内脉冲数** ⇒ **定 `T_min`** |
| **P1 剂量全扫（单 T）** | N=31k，T=T_min，dose {0,0.5,1,2,4,8}，seed 0 | **6** | (6×T_min×20k 步) | T_min=10 s ⇒ **0.20** | ≈12–20 min | (a)(b)(c)(d) 首读 + **张力三点**（若 T_min≥20 s ⇒ 0.40） |
| **P2 时长阶梯** | N=31k，T={T_min,2T_min,4T_min}，dose {0,1,4}，seed 0 | **9** | — | T_min=10 s ⇒ **0.30** | ≈18–30 min | 硬门④机器校验 + 漂移量（对照 A12 更正后的 +1.4% 受控口径） |
| **探针合计** | — | **18** | 8.4M | **≈0.60 GPU-h** | **≈35–60 min** | 三档全过 ⇒ 才允许提交 §9 的正式矩阵 |

- **P0 的裁决规则（预注册）**：零驱动臂 10 s 内脉冲数 `S0`：若 `S0 ≥ 300` ⇒ `T_min = 10 s`（阶梯 {10,20,40} s）；
  若 `300 > S0 ≥ 75` ⇒ `T_min = 20 s`（阶梯 {20,40,80} s）；若 `S0 < 75` ⇒ `T_min = 40 s` 并要求先查「网络是否近全静默」（属工作点问题而非预算问题）。
- **P0→P1→P2 逐档放行**：任一档出现 `hard_failures`（双估计器/量纲）⇒ **立即停**，先查口径再续（不得"跑完再说"）。
- **P1 的额外产出**：张力块的 `n_both_in_band` 与两条 Spearman ⇒ 若在 P1 就出现"无一剂量同时落带"且单调对立 ⇒ **提前触发 §5 换杠杆**，可省掉 P2 之后的一切。

### §10.2 若 `T_min` 上抬到 20 s：**预注册的降档菜单**（请主agent 预先裁决）

阶梯 {20,40,80} s 使 §9 的 C 阶段升到 **8.4 GPU-h**、总估 **≈13.2–13.6 GPU-h ⇒ 超 §10.1 的 ≤10 包络**。三个选项（**任选其一，不静默选择**）：

| 选项 | 做法 | 代价 | 风险 |
|---|---|---|---|
| **N1（推荐）** | 剂量网格由 6 档减到 **5 档**（去掉 8×；张力分析的端点由 4× 提供） | C 段 ≈7.0 GPU-h；总估 ≈11.8 | 张力端点上移到 4×（P1 已在 4× 看过，风险低） |
| **N2** | C 阶段 T 用 **{20,40}** + **复用 P2/P1 已跑的 10 s 档**（若 P0 已证 10 s 的 `S0<300`，则 10 s 档只能作**描述性**，不进阶梯） | 取决于复用量 | 阶梯只有 2 档 ⇒ 硬门④ 未满足（仅作描述性） |
| **N3** | 请主agent 批准 G1 包络临时提高到 **14 GPU-h** | 0 | 与 §10.1 的预注册上限冲突，须显式记录 |

### §10.3 **增量日志 / mtime 进度规范**（主agent 以 mtime 判进度 ⇒ 本工作包按此落盘）

```
data/m11_p5_runs/
  p0_31k_T10000ms_d0_s0.json      # 每 run **完成即写**（含 status/ms_per_step_steady/n_spikes/rate_median/silent_frac/wall_s/pid/host/device/sha256_of_trace）
  p0_31k_T10000ms_d0_s0.done      # 完成标记（**mtime = 该 run 的完成时刻** ⇒ `ls -t *.done | head -1` 即进度）
  p0_31k_T10000ms_d0_s0.fail      # 失败标记（含 stderr 尾部；**不计入分母**，四态标 `invalid`；沿 M10 RF-22 纪律）
  p0_31k_T10000ms_d0_s0_poptrace.npy   # 群体轨迹（逐步脉冲数，float32）
  p0_31k_T10000ms_d0_s0_counts.csv     # 逐神经元计数（neuron_id,count,cell_type）
```
- **run_id 规范**：`{stage}_{N}_{T}ms_d{dose}_s{seed}`（如 `p1_31k_T20000ms_d4.0_s2`）。
- **进度口径（给主agent 一句话）**：`ls -t neural_exploration/data/m11_p5_runs/*.done | head -1` 的 mtime = 最近完成时刻；
  `ls neural_exploration/data/m11_p5_runs/*.done | wc -l` = 已完成 run 数；**.fail 数 + .done 数 = 已启动 run 数**。
- **判算力是否被饥饿**：仍以 **累计 CPU/墙钟比** 为准（M10 L46/L50：`ps %CPU` 不可信，实测占空比曾低至 10%）。
- **停点**：`.fail` 连续 ≥2 或 `hard_failures` 非空 ⇒ 停并汇报（不自行重跑第三轮）。

---

## §11 用**实测** ms/step 重定价（主agent 派工；**纯台面算术，未跑任何仿真**）

> **本节为新增小节：§9/§10 的全部旧数字逐字保留**（其假设为清单 §0.9 的**预注册** ms/step），
> 本节用 **B2 交付的实测基准**重算，并**明确回答包络是否仍成立**。
> **正确换算因子**：`1 GPU-h = 3,600 s = 3.6e6 ms` ⇒ `GPU-h = steps × ms_per_step / 3.6e6`（= **2.7778e-7**）。
> 错式因子 `3.6e-6` 已被主agent 定位在 **`data/m11_engine_params.csv` L21**（`gpu_h_per_trial_formula`），
> 量纲自检：`600,000 步 × 1.1362 ms/step` = **2.4542（错式） vs 0.1894（对式）**，比 **12.960**；登记为 **E-M11-2**（由 B2 append-only 修正）。
> **§9/§10 用的就是正确因子**（未受 ×13 污染）。

### §11.1 输入（实测 + 口径）

| 项 | 值 | 出处 |
|---|---|---|
| `ms/step` @ N=8,192 | **0.6615** | B2 交付 `data/m11_engine_params.csv`（稳态、MPS、禁用段 0 预热） |
| `ms/step` @ N=31,000 | **1.1362** | 同上 |
| dt | **0.05 ms** | `m11_p5_targets.csv::protocol_dt_ms`（provenance `src/larval_zf_circuit.py::DT_MS`，`final=1`）+ `point_neuron.py:239 dt_ms=0.05` ⇒ **三处一致** |
| 步数 | T=10 s → **200,000**；T=20 s → **400,000**；T=30 s → **600,000**；T=40 s → **800,000**；T=80 s → 1,600,000 | `steps = T_s / dt_s` |
| 每 (dose, seed) 走三档 T | **1,400,000 步** | 200k+400k+800k |

### §11.2 探针段 P0/P1/P2 逐档重算（§10.1 的旧值并列保留）

| 档 | run | 步数 | 旧估（§10.1，0.6 ms/step） | **实测 @31k（1.1362）** | **实测 @8,192（0.6615）** |
|---|---|---|---|---|---|
| **P0**（T=10 s，dose{0,1,4}） | 3 | 600,000 | 0.10 | **0.1894** | **0.1102** |
| **P1**（T=T_min，6 档） | 6 | 1,200,000 | 0.20 | **0.3787** | **0.2205** |
| **P2**（T 三档 × dose{0,1,4}） | 9 | 4,200,000 | ~~0.30~~ → **本节点算术误差，正确值 0.70** | **1.3256** | **0.7717** |
| **探针合计** | **18** | **6,000,000** | ~~0.60~~ → 随 P2 更正后 **1.00** | **1.8937** | **1.1025** |

- **❗本节点自纠（§10.1 的 P2 行算术错误）**：P2 = 9 run = 4.2M 步，按 0.6 ms/step 应为 **0.70** GPU-h，而 §10.1 写了 **0.30**（漏乘 3 档 T 的合成），
  连带把"探针合计"由 **1.00** 写成 **0.60**（**低估 40%**）。**旧数字保留在 §10.1 不删**，本节为准。
- **主agent 的 1.894 GPU-h 与本节逐位一致** ✅（`6.0M × 1.1362 ms = 6,817.2 s = 1.8937 h`）。
- 合计相对我设计值 **3.16×**（1.8937/0.60）；其中 **1.72×** 来自 ms/step（1.1362/0.6 = 1.894），**1.84×** 来自 P2 漏乘（3 档 T）
  ——（两者相乘 1.894×1.667 = 3.16，逐项分解见 §11.5）。
- ⚠️ **§9.2 的"A 探针"是另一个（更重的）变体**：18 run × **全 6 剂量** × 3 T = **8.4M 步**（实测算如下），
  与 §10 的分档探针（P2 只 3 剂量 ⇒ 6.0M 步）**不是同一物**；**本节点采纳 §10 的分档探针为"探针段"**，§9.2 的 A 行按该变体读：@31k **2.6511**、@8,192 **1.5435** GPU-h。

### §11.3 **C 段逐臂步数表**（正式矩阵；臂 × T × seed）

预注册矩阵：**dose ∈ {0, 0.5, 1, 2, 4, 8}（6 档）× seed ∈ {0,1,2}（3）× T ∈ {10, 20, 40} s（三档）**。

| 臂（dose × seed 组合） | T=10 s | T=20 s | T=40 s | 该臂步数 |
|---|---|---|---|---|
| 每 1 个 (dose, seed) | 200,000 | 400,000 | 800,000 | **1,400,000** |
| **6 dose × 3 seed = 18 臂** | 3,600,000 | 7,200,000 | 14,400,000 | **25,200,000（25.2M）** |
| （若 N1：5 dose × 3 seed = 15 臂） | 3,000,000 | 6,000,000 | 12,000,000 | **21,000,000（21.0M）** |

| 版本 | 步数 | **实测 @31k** | **实测 @8,192** |
|---|---|---|---|
| C（6 档） | 25.2M | **7.9534** | **4.6305** |
| C（5 档，N1） | 21.0M | **6.6278** | **3.8587** |

⇒ 主agent 用的 **C = 8.4 GPU-h** 与逐臂表不符：**正确为 7.95**（差 +5.6%）。可能来源：§9.2 的 **A 行步数 "8.4M 步"** 被当作 C 段 GPU-h（或按 4.2×2 缩放）。

### §11.4 三段合计与包络对照（**结论翻转**）

| 段 | 内容 | 步数 | ***实测* GPU-h** |
|---|---|---|---|
| **A（探针）** | §10 分档 P0+P1+P2（18 run，31k） | 6.0M | **1.8937** |
| **B（杠杆搜索）** | 4 轮 × (6 dose × 3 T × 1 seed) @N=8,192 | 33.6M | **6.1740** |
| **C（正式矩阵）** | 6 dose × 3 seed × 3 T @N=31k | 25.2M | **7.9534** |
| **D（确定性）** | 2 重复 × 3 T @N=31k | 2.8M | **0.8837** |
| **合计** | — | 67.6M | **16.905 GPU-h** |

> **🔴 明确回答：§9.2 的「≈8.87 GPU-h，落在 ≤10 包络内」**不成立**。**
> 用实测 ms/step 重算为 **≈16.90 GPU-h ⇒ 超包络 69%**（旧值 8.87 是 0.6 ms/step 假设 + 我 §11.2 那处未出现在 §9.2 的算术错误之外的**低估**共同结果；
> §9.2 的分段数字本身算术无误，**错在假设** ms/step=0.6 而非实测 1.1362/0.6615）。
> **主agent 的"探针 1.894 + C 8.4 ≈ 10.3 ⇒ 触顶略超"同样过于乐观**：**漏掉 B 段（6.174）与 D 段（0.884）**，且 C 应为 7.95。
> ⇒ **R21 的三项超包络裁决依然有效，且比原判更严重**（不是"略超"，是 **+69%**）。

### §11.5 菜单逐项代入（R21 预注册项；**N2/N3 不采用**）

| 方案 | 组成与步数 | 实测 GPU-h | 对照 ≤10 |
|---|---|---|---|
| 原方案（§9.2） | 探针 31k + B 6档×3T×4轮@8k + C 6档@31k + D@31k = 67.6M | **16.905** | ❌ +69% |
| **N1 单用**（剂量 6→5；其余同原） | 探针 5.8M + B 28.0M + C 21.0M + D 2.8M | **14.487** | ❌ +45% |
| **N4 单用**（C 段改 8,192；探针仍 31k） | C 步数不变，仅换算变 | **13.213** | ❌ +32% |
| N4（探针亦改 8,192） | 探针 6.0M@8k | **12.421** | ❌ +24% |
| **N4 + N1**（全段 8,192，5 档） | 探针 5.8M + B 28.0M + C 21.0M + D 2.8M，全部 @8,192 | **10.584** | ❌ **+6%（仍超）** |
| N4 + N1 + **B 轮数 4→3** | B 21.0M | **9.298** | ✅ |
| **★N4 + N1 + B 只做 `T_min` 单档搜索（4 轮）** | B 4.0M（5 dose × 1 T × 1 seed × 4 轮） | **6.174** | ✅ **余量 38%** |

**本节点建议采用 ★（N4 + N1 + B 段单档）**，理由三条：
1. **单项都不够**：N1 单用 +45%、N4 单用 +24–32%，**N4+N1 也仍 +6%** ⇒ 必须再有一项削减；
2. **B 段（杠杆搜索）本不需要 T 阶梯**：§5 的钳制是"每次改动后重核 (c)"，而 (c) 的**定稿**阶梯属 C 段责任 ⇒ 搜索阶段只在 `T_min` 单档做、由 C 段对该**最终配置**做三档确认，**方法学上更正确而非妥协**（避免在未定稿配置上烧长 T）；
3. 余量 38% 可吸收 (i) 占空比损耗、(ii) 若 `T_min` 上抬到 20 s 的升档（B 段 4.0M→8.0M 步 ⇒ +0.735 GPU-h ⇒ 仍 ≈6.9）、(iii) 一次额外 B 轮。

### §11.6 改档前提提醒（主agent 已点名，此处预注册）

1. **档位是 P5 协议的一部分 ⇒ 必须显式登记**：N=8,192 属 RF-M11-6 的**默认档（8k/31k）**之内 ⇒ 合法，但须在 `m11_p5_protocol.json` 与 P5 报告里写明"**本档 = N_small=8,192**"，且 **N_full 仍只在 E 行（关键条件对照）进**。
2. **任何档上都要重验 (a)(b) 是否同时可达**（工作点随 N 变）：本方案下 **P0/P1 的 (a)(b) 首读即在 8,192** ⇒ 探针与 C 段**同档**，不存在"探针在 31k、定稿在 8k"的口径断层。
3. **若 8,192 档 (a)∧(b) 不可达而 31k 可达**（回退分支，预算对照）：31k 下 **N1 + B 单档 4 轮 = 10.605（+6%）**；**N1 + B 单档 3 轮 = 10.289（+3%）**；**N1 + B 单档 2 轮 = 9.973 ✅**；**N1 + B 单档 3 轮 + D 并入 A（重复臂复用探针） = 9.405 ✅** ⇒ 兜底可达包络，但需再削一轮 B 或复用 D。
4. **探针段是否要改档**：**要**（见 2）——探针的职责含"定 `T_min`"与"(a)(b)(c) 首读"，两者都依赖 N；探针在 31k 而 C 在 8,192 会使 `T_min` 决策（零驱动臂 ≥300 脉冲）**不可传递**。
