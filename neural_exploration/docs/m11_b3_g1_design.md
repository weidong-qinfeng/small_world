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
| 1 | `data/m11_p5_targets.csv`（72 行） | **判据带 / 协议常数唯一定稿表**，逐行 provenance（PMID/DOI/数据文件::字段/清单行） |
| 2 | `data/m11_p5_targets.json` | 同一内容的机器可读版 + provenance ledger（本轮文献核读原文摘录、实测复算、冲突登记、未定稿带清单） |
| 3 | `tools/validate_p11_resting.py` | **P5/G1 离线评估器**（只做分析，不做仿真）：接受注入的群体轨迹 / 逐神经元计数 / 聚合读数，输出 (a)–(i) + **四态** + 六项硬门 |
| 4 | `docs/m11_b3_g1_design.md`（本文） | 判据表、协议、标定杠杆顺序、统计方案、张力识别、时长依赖处置、失败/反证路径 |
| 5 | `docs/m11_b3_notes.md` | 环境/口径/冲突登记（L1 起）+ **跨节点 sha256（标注基准）** |
| 6 | `data/m11_b3_managed_jobs.json` | 需主agent 托管的作业（确切命令行 + 产物路径 + 估计墙钟） |
| 7 | `data/m11_p5_smoke_*`（14 件） | 冒烟测试夹具与输出（S1–S6；M10 冻结件只读 + **SYNTHETIC** 计数夹具，文件内已标注） |

**未做（边界）**：不启动任何仿真/长作业；不碰 `src/mouse_cortex_*`（B2）、`data/m11_data_*` / `tools/fetch_m11_*` / `tools/build_m11_*`（B1）与 M0–M10 全部冻结件；不写判据带（工具只读带）。

---

## §2 判据表（`data/m11_p5_targets.csv`）——覆盖与可用性

**逐行 provenance 覆盖：72/72 行非空**（工具 `--check-targets` 机械校验：缺 provenance 即报错）。

| 证据级别 | 行数 | 代表键 | 来源 |
|---|---|---|---|
| `literature` | **17** | `rate_median_hz_lo/hi`（0.5 / 2.0 Hz）、`rate_median_hz_exc_*`、`rate_median_hz_inh_*`、`silent_frac_ca_active_per_wave_pct`(12%)、`unresponsive_neurons_pct_all_areas`(23%)、`proxy_steriade1993_upstate_duration_s`(0.8–1.5 s)、`proxy_sanzeni2020_isn`、`proxy_wehr2003_balanced_inhibition`、`proxy_monier2008_ei_diversity`、`proxy_buzsaki2014_lognormal_rates` 等 | PMID 30419013/40205214/19706480/31844315/8340806/11017176/32598278/14647382/17507551/18215425/24569488（**11 篇本轮逐条核读**：数值在摘要或 OA 全文中可见者才标 `literature`） |
| `derived` | **42** | 全部门限/网格/口径常数（`selfsustain_ratio_max`=0.10、`selfsustain_ci_upper_factor`=1.2、`selfsustain_zero_drive_min_spikes`=300、`protocol_T_ladder_ratios`、`ei_ratio_lo/hi`=0.5/3.0、`calibration_tradeoff_requires_a_and_b`、`tension_detect_required` 等） | 清单 §4.5/§0.9 草案行 + RF-M11-3 裁决行 + `data/m10_calibration_targets.csv::字段` + `src/larval_zf_circuit.py::字段`（**均为设计选择/实测派生，不是文献值**） |
| `measured` | **4** | `frame_rate_hz_microns_functional`=6.300006 Hz、`proxy_allen_celltypes_avg_firing_rate_median_all`=17.4130、`..._visp`=17.6338、`proxy_allen_celltypes_rate_lt_0p5_frac_visp`=0.0000 | `data/m11_microns_protocol.json::protocol.functional_series[0].frame_rate_hz`；`data/m11_allen_celltypes_metadata.csv::ef__avg_firing_rate`（n=1808 / VISp n=1490，本节点独立复算） |
| `literature_unverified` | **1** | `n_neuron_gear_full`=75909 | 清单 §0.9 标注「Nature 2025 摘要」⇒ **实测该摘要只给 'around 75,000'**（冲突 L6） |
| `unavailable` | **8** | `rate_median_hz_measured_in_vivo_band`、`silent_frac_lo/hi`、`silent_frac_ephys_lo/hi`、`ei_ratio_measured`、`single_trial_wall_s_max`、`microns_activity_channel` | B1 未交付项 + R1/R2（MICrONS 活动永久不可用）+ G0 探针未回 |

**`final=0`（未定稿 ⇒ 依赖它的子判据强制 `unavailable`）：8 行** ——
`(a)` 的**实测**带、`(b)` 的钙域/电生理域两个带、`(e)` 的实测量、`(g)` 的墙钟上限。
⇒ **当前状态下 (b) 与 (g) 必判 `unavailable`；G1 因此必然不能判 `pass`**（这是本工作包的**诚实结论**，不是缺陷）。

### §2.1 跨制备 / 跨口径代理（明确登记 + 量化后果）

| 代理行 | 制备 | 量化后果（`wrong_usage_consequence`） |
|---|---|---|
| `proxy_allen_celltypes_avg_firing_rate_median_all/visp` | Allen Cell Types **in vitro 电流钳**（非在体、非自发） | 中位 **17.41 / 17.63 Hz** vs 本门文献带 0.5–2.0 Hz ⇒ **高 8.7–34.8×**；且其 `frac<0.5 Hz = 0.0000` ⇒ 静默带被压到 ≈[0,0.02]，**必然与 (a) 冲突**（M10 A4 标定权衡的正体） |
| `proxy_sanchezvives2000_ferret_slice` | **雪貂皮层切片 in vitro**，0.1–0.5 Hz | 清单 §1.5 C3 把它当「小鼠皮层 in vivo dwell 0.3–3 s」= **物种/制备/数值三重误引**（冲突 L5） |
| `proxy_steriade1993_upstate_duration_s` | 猫**在体**、氨基甲酸乙酯麻醉 | up 态 0.8–1.5 s；跨物种 + 跨麻醉 ⇒ 仅量级代理；且 **P5 不判 dwell（越界）** |
| `proxy_sanzeni2020_isn` / `proxy_wehr2003_balanced_inhibition` / `proxy_monier2008_ei_diversity` | 小鼠 V1 清醒/轻麻醉（ISN）· **大鼠听皮层**在体 · 麻醉猫 V1 | 仅支撑 E/I 的**方向与量级**（抑制不弱于兴奋、E/I 同量级、个体多样 ⇒ 带不得窄）；**不能**支撑任何数值端点 |
| `silent_frac_ca_active_per_wave_pct` / `unresponsive_neurons_pct_all_areas` | 小鼠 V1 2p（**每次自发波**招募口径）· Allen 2p（**对刺激可靠响应**口径） | 都不是「T 窗内零计数比例」；直接搬用 ⇒ 静默带口径错（12% 活跃被读成静默 0.12；23% 未响应被读成静默 0.23） |
| `rate_median_hz_m10_fish_band` / `silent_frac_m10_fish_band` | 斑马鱼幼鱼（M10） | RF-M11-1 禁移植；若移植，M10 的「中位率 0.000 Hz + 静默 0.920」会被判成 (b) 通过 |

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

| M10 缺陷 | 本工作包的机械堵法（工具内已实现并有冒烟证据） |
|---|---|
| **① 门 1 不是统计可判**（3 seed = 0.0987/0.1056/0.1005、R=0.1015、CI95 [0.0945,0.1091] 跨门限；零驱动臂仅 60 脉冲 ⇒ 相对不确定度 13%） | **硬门①样本量门**：零驱动/基准臂各 ≥3 seed（去重）+ 零驱动总脉冲 ≥300；不满足 ⇒ `unavailable`（**不得判 pass，也不得判 fail**），点估计退为描述性。**硬门②CI 门**：主法泊松对数正态（`se(log R)=√(1/S₀+1/S₁)`，与 M10 实现**逐位一致**）+ 次法 seed 重采样×泊松自助，两法上界相对差 >0.25 ⇒ `invalid`；判定 =「点估计 ≤ 上限 **且** 更宽的 CI 上界 ≤ 上限×1.2」。<br>**冒烟证据（S2）**：注入 M10 `m10_p5_gate1_ci.json` 的逐 seed 脉冲数 ⇒ 工具给出 R=0.1015384615、se_log=0.0365403631、逐 seed 比值**与 M10 逐位一致**，判 **fail** 并标注「统计未分辨（点估计未过，但 CI 上界未越过 ×1.2 带）」。 |
| **② 门 1/2 对机制阶梯无判别性**（基线自身即过：+579.6%） | **硬门⑤判别性断言**：每轮必须给出「**无机制基线臂**（`mechanism=none`、`drive_mult=1`）」的同一判据值；缺 ⇒ 整体 `unavailable`（不得 pass）。基线自身达标 ⇒ 该判据标 `discriminative=false`。并且判据表登记 `criterion_d_as_mechanism_evidence=0`（`(d)` **不得**作为机制有效性证据）⇒ 工具的 (d) 行永远带 `evidence_class=sanity_precondition(non_discriminative)`，且 `non_discriminative_criteria` 列表在总判定里单独输出。<br>**新增判别性来源**：`(a)` 改为**类级带**（E 0.5–1.0 Hz / I 1.5–2.0 Hz，**I > E 的方向**），`(e)` 给带 —— 二者都是基线**不必然**通过的判据。 |
| **③ 判据的时长依赖性**（T=2 s→0.0980、T=60 s→0.1082、T=120 s→**0.1720**，+76%；λ×4 中位率随 T 由 0→1.0 Hz） | **硬门④时长阶梯**：同一配置必须给 {T_min, 2T_min, 4T_min} 三档的**同一判据值**，并把漂移做成**被测量的量**（`rel_change_first_to_last`、`spearman_vs_T`、`status_flip`）。跨 T 判定翻转 ⇒ 该子判据 `invalid`（协议时长绑定失败）。**另加截断守卫**：若短时长臂的轨迹是长时长臂的逐位前缀 ⇒ 阶梯判 `invalid`（`ladder_arms_look_truncated`），因为截断测的是窗口长度而非协议时长。缺阶梯 ⇒ 阶梯 `unavailable` 且**该判据不得定稿**。<br>判定结果永远携带其 `protocol_T_ms`（M10 的「T=2 s 窗下通过」限定被机械化）。 |
| 另：**四次口径事故**（×1.3004 / ×1.15 / 43× / 无断言） | **硬门③双估计器一致性**：`counts` 求和 vs `pop_trace` 求和 vs 外部聚合读数，两两相对差 >2% ⇒ **整轮 `invalid`**（不是 fail）；只有单一估计器 ⇒ `unavailable` 且阻断 pass；并加**非平凡量纲自检**：轨迹长度 ⟺ `round(T_ms/dt_ms)`（**该断言正是 ×1.3004 的同族缺陷探测器**）、计数长度 ⟺ N。<br>**冒烟证据（S3/S4）**：正常臂 `rel_diff = 0.0`；把基准臂计数人为放大 1.15× ⇒ `rel_diff = 0.1304 > 0.02` ⇒ 整轮 `invalid`、exit=2。 |
| 另：**判据带未定稿却照判** | **带定稿机械强制**：带值全部读自 `m11_p5_targets.csv`；`final=0` 或 `evidence_level=unavailable` ⇒ 该子判据 `unavailable`（**拒绝用未定稿带判 pass/fail**）。 |
| 另：**"未测" 被写成 pass/fail** | 四态 + 退出码：`pass`=0 / `fail`=1 / `invalid`=2 / `unavailable`=3；`invalid` 与 `unavailable` **永不计入** pass/fail（清单 §0.7 #6）。 |

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
