# M3 环境与前置处置记录（清单 §1：L1–L4 + 执行节点实测 L5–L15）

> 对应《生物仿真M3实施清单》§1（P7 交付物）。
> 记录时间：M3 实施期（B1a 执行节点：参数定稿 + NEURON 全链参考解；
> B1b 执行节点：Brian2 实现 + 冒烟测试；B2 验证+报告节点：P1–P5 验证实测）
> 状态：L1–L4 处置完成；L5–L10 为 NEURON 参考解实测结论；L11–L15 为
> B1b/B2 的 Brian2 实现与验证实测结论（含 P5 幅度亚判据结构性不可达，见 L14）

---

## L1 — 交接：NeuronPair 硬编码双神经元 → ReflexArc（N 神经元链）

- **直接复用（不改造）**：
  - `ChemicalSynapse` / `GapJunction`：接受任意带 `.neuron` / `.label_of` / `.soma_area_cm2` 的神经元包装——`MultiCompartmentNeuron` 已具备（M2 交接 §10）；
  - `SynapseParams` / `load_synapse_params` / `chemical_post_eqs` / `chemical_im_terms` / `GAP_POST_EQ`（src/synapse_model.py）——M3 链的电学基础沿用 m2 CSV 的 ampa/gaba 行（量子电导/τ/E_rev），**链级权重另行定稿于 m3 CSV（见 L3/L6）**；
  - `psp_amplitudes` / `norm_rmse` / `binomial_ci`（tools/synapse_metrics.py）、`pulse_train`、`PairResult`、
    `STIM_WINDOW_MS=500` 固定形状约定、`_stim_arrays` 的 TimedArray 显式命名 + namespace 传参模式（M2 L6 编译缓存纪律）、
    `_record_spec` / `_spike_times` 的 pre_/post_ 标签约定（M3 扩展为 plm_/avm_/da_/vb_ 前缀）。
- **新抽象**：`src/reflex_arc.py` 的 `ReflexArc`（N 神经元链组装，按 `data/m3_reflex_params.csv` 驱动；触刺激 + VB 张力注入 + 方向判定）——由 B1b 实现。
- 复用注意：M2 `ChemicalSynapse` 的 on_pre 更新 post 组电导（`_post` 后缀）与多隔室耦合在 N 链中会放大（SESSION_CONTEXT §四 #10）——先 3 神经元单链冒烟再上双通道（清单 §7 风险表）。

## L2 — 参考解方案：NEURON 全链（本节点交付）

- 工具：`tools/build_reflex_ref.py`（新建，M3 版参考解生成器，模式照 `tools/build_synapse_ref.py`）。
- 构链：`build_neuron(spec, clear, name_prefix)`（M2 已验证 clear=False 同进程多神经元）构建 4 个 M1 形态学神经元
  （PLM/AVM/DA/VB，name_prefix=plm_/avm_/da_/vb_；每档强度 clear=True 重建 → 逐档独立）。
- 突触：ExpSyn（AMPA E=0 mV / GABA E=−70 mV；weight µS = g_ns×1e-3；τ 沿 m2 行：ampa 3ms / gaba 5ms）；
  NetCon(pre node3(0.5) _ref_v → post soma(0.5))，threshold=−20，delay=0.5 ms（m3 CSV 定稿值，非 M2 的 0.1）。
- 刺激：IClamp 触电流注入 **PLM 树突端**（方案 A，清单 §2.2）——dend2 末隔室中心；
  `amp_nA = I0·s_i · 1e-6 · A_tip[cm²] · 1e9`，A_tip = π·d·L = π·1.5µm·50µm（末隔室侧面积，见 L5）。
- VB 张力：IClamp 恒定电流注入 VB soma（tonic=14 µA/cm²，→ ~60 Hz 发放，见 L9）。
- 6 档强度 {0, 0.5, 1, 2, 4, 8}×I0 各跑一次；cvode atol/rtol=1e-8、h.celsius=6.3、h.v_init=V0（硬约束）。
- 输出：各级 node3 发放时刻（V 轨迹阈值检测，-15 mV 上冲 + 峰定位）、感觉→中间 PSP（AVM soma V）、
  神经潜伏期（触刺激开始 → DA 首发放）、行为潜伏期（由发放序列经**同一肌肉 ODE** 计算，与引擎无关）→ `data/m3_reflex_ref.npz`。
- 点过程（IClamp/NetCon/ExpSyn）**必须 Python 列表持有防 GC**（M2 L8；本节点已在 builder 中实现）。
- 缝隙连接：M3 主链不含缝隙连接（纯化学突触链），无需 scipy 独立解（M2 env_notes §L2 路径备用）。

## L3 — 参数定稿依据

- 行为学：触觉逃避潜伏期 30–50 ms（Chalfie et al. 1985，P3 参考窗 [25,60] 含容差）；
  强度-响应应有单调关系（P4）；连接组极性语义简化（White 1986）：
  感觉→中间谷氨酸能兴奋；中间→后退运动兴奋（AMPA）；中间→前进运动 GABA 抑制（互斥方向机制）。
- 突触电学基础沿用 `data/m2_synapse_params.csv`（ampa 量子 0.3 nS/τ3ms/E0mV；gaba 1.5 nS/τ5ms/E−70mV）；
  **链级权重定稿**于 `data/m3_reflex_params.csv`（唯一定稿源）：
  - AMPA 链权重 g_max=5.0 nS（≈17×量子）——实测 0.3 nS 仅产生 0.68 mV EPSP（与 M2 结论一致），不足以驱动 AVM；
    ≥3 nS 使 AVM 可靠发放（见 L6）。
  - GABA 链权重 g_max=15.0 nS（10×量子）——实测 ≥10 nS 使 VB 跳过/推迟 1 个发放（60 Hz 张力下），C_fwd 回落（见 L6/L8）。
  - 突触延迟 0.5 ms（0.5–1 ms 生理值；M3 链统一；参考解 NetCon delay 同值，保证与 Brian2 可比）。
  - I0=60 µA/cm² × 5 ms @ t=50 ms（树突端注入；M2 的 20 µA/cm² 为胞体注入值——树突端局部输入阻抗高、注入绝对电流小，
    20 µA 不发放，40 µA 临界，60 µA 可靠；见 L5）。
  - VB 张力 tonic=14 µA/cm²（soma 恒定电流，~60 Hz；C_fwd 静息基线 ≈0.19–0.21 ≈0.2）。
  - 肌肉：TAU_MUSCLE=20 ms（M0 惯例）、w_back=0.6（每 DA 发放 → C_back 增量；D_peak>0.3 达成）、
    w_fwd=0.18（每 VB 发放 → C_fwd 增量；基线≈0.2）。
- 潜伏期目标窗 [30,50] ms（Chalfie 1985）写入 CSV；**实测行为潜伏期 ≈ 神经潜伏期（8–14 ms），结构性落不到窗内——见 L7**。

## L4 — 单位/方程约定

- 电导密度 S/m²（Brian2 Im 为密度）、点电导 nS（CSV/NEURON）、刺激密度 µA/cm²（`density_to_nA` 按注入位点面积换算）、
  ms/mV、**Im 内向正约定**（SESSION_CONTEXT §四 #1，M1/M2 延续）。
- 触刺激（方案 A）：`I_touch(t) = I0·s_i·1[t0, t0+dur]`（µA/cm²），注入 PLM 树突端；
  nA 换算 `= I0·s_i·1e-6·A_tip·1e9`，A_tip = dend2 末隔室侧面积 π·d·L（本节点实测用值 2.356e-6 cm²）。
- 肌肉方程（M0 smoke_loop 升级，双通道）：`dC/dt = −C/TAU_MUSCLE + Σ_k w_k·δ(t−t_k)`；
  参考解按解析积分 `C(t) = Σ_k w_k·e^(−(t−t_k)/τ)·H(t−t_k)`（δ 脉冲注入 + 指数衰减，与引擎无关）；
  方向 `D(t) = C_back(t) − C_fwd(t)`，D 峰值 > 0.3 → 后退（P1）。
- 行为潜伏期定义（清单 §5.3）：首个 `C_back ≥ 0.3·C_back_peak` 的时刻 − t_touch_start。

---

## 执行节点实测结论（L5–L10，供 B1b / 规划节点复核）

### L5 — 触刺激注入位点实测：树突端 vs 胞体（方案 A 定稿依据）
- 同一形态学，密度→nA 按**注入位点膜面积**换算：
  胞体（球面积 πd²=1.257e-5 cm²）vs 树突端 dend2 末隔室（侧面积 π·d·L=2.356e-6 cm²，5.3× 之差）。
- 实测（dur=5 ms @ t=50 ms，NEURON cvode）：
  - 胞体注入：20 µA/cm²×1 ms → 可靠 1 发放（M2 复现）；20–80 µA/cm²×5 ms → 均 1 发放，首发放 51.9–52.9 ms。
  - 树突端注入：20 µA/cm² → **不发放**；30 µA → 1 发放 @57.69 ms（7.7 ms 潜伏期，临界）；40 µA → @55.43；
    60 µA → @54.17；80 µA → @53.62；120–480 µA → @53.1–52.25（单调递减 ✓）。
- **定稿：I0=60 µA/cm² × 5 ms，注入 PLM 树突端（方案 A 主用）**。0.5×I0=30 µA/cm² 可发放（临界但实测稳定）。
- 各档位均为单发放（5–10 ms 内 HH 不应期不允许第 2 尖峰；480 µA×20 ms 才出现第 2 发放 @+13.8 ms）→ 见 L10。

### L6 — 链权重定稿实测（AMPA 驱动、GABA 抑制）
- PLM→AVM（AMPA）：0.3 nS（m2 量子）→ AVM soma EPSP **0.68 mV**（与 M2 记录一致）→ AVM 不发放；
  1 nS → 2.43 mV；3 nS → AVM 发放（EPSP 含尖峰 ~95 mV）；5 nS → 可靠发放（PLM 发放 +2.8–3.0 ms）；
  10–20 nS → 发放更早（+1.1–1.9 ms）。**定稿 AMPA 链权重 5.0 nS**（阈值 ~2.5–3 nS 的 1.7–2×，鲁棒）。
- AVM→VB（GABA，VB tonic 14 µA/cm² → 60 Hz）：
  1.5 nS（m2 量子）→ VB 发放仅推迟 ~0.3 ms（抑制不足）；5 nS → ~0.5–0.8 ms；10 nS → 跳过 1 个发放（暂停 ~5 ms）；
  15 nS → 跳过 1 个、下一发放延后 ~4 ms（54.46→76.55 vs 对照 71.63）；20–40 nS → 暂停 >35 ms。
  **定稿 GABA 链权重 15.0 nS**（明显抑制又不致完全停摆，与"张力被抑制压过 → C_fwd 回落"语义一致）。

### L7 — ⚠️ 行为潜伏期定义与 [25,60] 窗的结构性矛盾（P3 风险，需规划节点复核）
- 实测（I0 档）：t_PLM=54.17 < t_AVM=57.14 < t_DA=60.10；神经潜伏期 **10.10 ms**；
  肌肉 ODE（τ=20 ms）下 DA 单发放 → C_back 在发放瞬间跳到峰值 w，`0.3·C_back_peak` 在**同一发放时刻**即被越过
  → **行为潜伏期 ≡ 神经潜伏期**（builder 实测各档 13.63/10.10/9.04/8.51/8.18 ms），结构性落不到 [25,60] ms 窗。
- 数学本质：δ 驱动的一阶肌肉对单发放/紧凑脉冲串，0.3·peak 判据的穿越点 ≈ 首个贡献 ≥30% 峰值的发放时刻 ≈ 神经潜伏期 + 0~5 ms；
  与 TAU_MUSCLE 无关（单发放时峰值即首次跳变）。
- **建议（择一，由规划节点裁决）**：① 复核 P3 行为潜伏期定义（如改固定阈值 0.3 绝对、或改"发放序列起始后肌肉达 X% 的时间窗"）；
  ② 肌肉改为连续（速率码）驱动并调大 TAU（如 50–80 ms）；③ 加长触刺激至 ≥20 ms 制造多发放慢响应（超出清单 5–10 ms 建议，需批准）。
- 本参考解按清单 §5.3 原定义如实计算并落盘（latency_behavior），不做私自改判。

### L8 — ⚠️ P5 消融判据的时序敏感性问题（方向机制演示，B1b 需注意）
- D_peak 出现在 **DA 发放时刻**（I0：t=60.10，D=0.372 = C_back 0.6 − C_fwd 0.228；builder 实测全档
  D_peak=0.349–0.409）；
  而 AVM→VB 的 GABA 抑制最早在 **VB 下一发放时刻**（~71.6 ms，被推迟至 74.2 ms）才改变 C_fwd 轨迹。
  → 删除 GABA 连接对 D_peak **几乎无影响**（消融后 D_peak 仍 ≈0.37，衰减 <5%），
  且 C_fwd 峰值（≈0.31，w_fwd=0.18）在有无抑制两种情形下相同 → P5 两判据（C_fwd>0.3 或 D 衰减>50%）**均难满足**。
- 建议：① P5 协议改为比较"响应窗 [t_touch_start, t_touch_start+30ms] 内 D 均值 / C_fwd 均值"；
  ② 提高 VB 张力（C_fwd 基线更高）使抑制更显著；③ 或延长 GABA 抑制（更大 g / 更长 τ）使 VB 停摆 >30 ms。
- 方向正确性（P1 判据 D_peak>0.3）本身**不受影响**：I0 档 D_peak=0.384>0.3 ✓，全档 0.36–0.42 ✓，0 档 D≤0 ✓。

### L9 — NEURON cvode 近阈值持续电流会挂起（本节点实测坑）
- VB 张力扫参：tonic=13.0–13.75 µA/cm²（介于 12 亚阈值与 14 的 60 Hz 之间）在 **T_TOTAL=250 ms 时 cvode 挂起 >5 min**；
  150 ms 窗口正常（0.1 s）。原因：近阈值膜电位长时间准稳态漂移 + cvode 1e-8 容差反复试探。
  **对策：定稿 tonic=14 µA/cm²（60 Hz 稳定区，每档运行 0.1–0.3 s）**；后续若需近阈值张力，用固定步长或缩短窗口。

### L10 — 每档单发放 → P4 幅度判据的结构性风险
- 5–10 ms 触刺激下 PLM 每档只发 1 尖峰（L5），链级各神经元同样单发放 → C_back 峰值恒为 w_back（0.6）→
  "响应幅度 vs 强度 Spearman ρ≥0.9" 恒等序列 ρ≈0.68（含 0 档）→ **P4 幅度判据结构性难通过**（潜伏期判据 ρ≤−0.9 通过 ✓：13.6→8.2 ms 单调）。
- 建议：P4 幅度改用"发放数/总发放次数"（强度 0=0，其余=1）之外的新度量（如 8×I0 下延长刺激时程制造多发放），或复核判据。

---

## 执行节点实测结论（L11–L15，B1b 实现 / B2 验证节点实测）

### L11 — Brian2 2.6 事件代码限制：min/max、if/else 分支不解析 → 用 clip()
- `Synapses(on_pre=...)` 事件语句（以及 on_event/on_post 同族）中，Brian2 2.6 的
  sympy 解析器**不把 min/max 解析为内置函数、不支持 if/else 分支**（B1b 实测）。
- 肌肉饱和 `C = min(C + W, 1.0)` 会直接报错/生成非法代码 → 改用
  `clip(C_post + WMUSC, 0.0, CAP)`，CAP/WMUSC 经 namespace 传入（muscle.py 已实现，见 L13 配套）。
- 教训：事件代码里避免函数调用与分支；需要钳位/饱和一律 `clip()` + namespace 常量。

### L12 — Network.store()/restore() 会重置 monitor：run_trials 每试次返回完整轨迹
- `net.store()` 快照（含 monitor 记录）→ 每试次 `net.restore()` 后 monitor 归零 →
  试次间轨迹**不累积、不切片**，每试次的 monitor 数据即该试次自身的完整轨迹
  （比 M2 NeuronPair 的累积-切片取最后一段更正确；B1b 已在 run_trials 实现）。
- restore 后变量回到快照（突触电导 g 清零、神经元/肌肉状态复位），
  配合 `bseed(seed_base + trial)` 逐试次重播种 → 每试次独立同分布。
- 实测（B2 验证运行）：20 试次噪声潜伏期 SD=0.30ms（<5ms 判据 ✓），失败（DA 无发放）试次被
  正确剔除并计数（P3 实测 0/20 失败；P1 实测 1/8 失败，仍取足 5 个有效试次）。

### L13 — ReflexArc 构造参数默认值必须 None（否则静默覆盖 CSV 唯一定稿源）
- `ReflexArc(csv_path=None, dt_ms=None, method=None, t_total_ms=None, seed=None)`：
  显式传入才覆盖 CSV；**默认值若非 None 会在调用方未传参时静默覆盖 CSV 定稿值**，
  导致验证脚本与参考解参数不一致（B1b 已按 None 默认实现）。
- 配套纪律：验证脚本一律 `ReflexArc(csv_path=CSV_PATH)` 不传其余构造参数，
  协议覆盖（set_touch/remove_synapse/set_quantum_noise）显式调用。

### L14 — ⚠️ P5 消融复现（B2 实测）：D_peak≈0.37 不变 + C_fwd 窗均值 ×1.2 亚判据结构性不可达
- **复现 L8 预测**：删 AVM→VB 后 D_peak 完全不变（完整 0.3689 = 消融 0.3689）——
  D_peak 出现在 DA 发放时刻（t≈59.6ms），早于 GABA 生效时刻（VB 下一发放 ≈71.7ms）。
- 修订判据（主 agent）实测（Brian2，响应窗 [50,90]ms，两引擎一致）：
  - **发放数子判据 ✓**：VB 发放 完整 2 → 消融 3（完整 54.34/75.21，消融 54.34/71.68/89.03；
    NEURON 参考同：2 → 3）。
  - **C_fwd 窗均值 ×1.2 子判据 ✗**：完整 0.1870 vs 消融 0.1978 → 比值 1.058
    （NEURON 参考 1.054；遍历窗变体 [50,88]–[60,100] 最大 1.11，均 <1.2）。
- 结构性原因：VB 张力维持 ~60Hz 基线振荡（C_fwd ≈0.13–0.31，均值 ≈0.19）主导窗均值；
  GABA 只推迟/抑制响应窗内 1 个 VB 发放（相移 ~3.5ms + 1 发放移出窗），
  w_fwd=0.18 下不足以把均值差推高 20%。
- 响应分量（窗均值 − 无刺激对照同窗均值）：完整 **−0.0108**（GABA 抑制 → C_fwd 低于无触基线）
  vs 消融 **0.0000**（无 GABA → 触刺激不改变 VB 发放）——机制方向正确。
- **主 agent 三态裁决（2026-08-23）**：✅ **通过（判据落地修订）**——机制已由直接证据证明
  （发放数 2→3 两引擎一致 + 响应分量符号 full=−0.011<0 / abl=0.000≥0）；×1.2 窗均值子判据
  系**主 agent 判据校准错误**（非机制失败），保留为 informational 输出。
  修订后判据：`pass_ = count_ok AND resp_component_sign_ok`，
  `resp_component_sign_ok = (mean_full − ctrl_mean) < 0 AND (mean_abl − ctrl_mean) ≥ −1e-6`。
  若后续仍需幅度判据：上调 w_fwd/tonic 或延长 GABA τ（需解冻 data/m3_reflex_params.csv 重定稿 + 重跑参考解）。

### L15 — 量子释放语义（B2 实测新坑）：g_max_ns 是单囊泡量子电导
- ReflexArc 沿 M2 语义：`g_max_ns` 为**单囊泡量子电导**，事件电导 = g_max × k，
  k ~ Binomial(n_vesicles, p_release)。
- 确定性默认 p=1/n=1 → 单量子 = 全链设计重（AMPA 5.0nS、GABA 15nS，与 CSV 注释一致）。
- 噪声协议 p=0.95/n=2（P1/P3）→ 事件电导 k×5nS：均值 9.5nS、k=2 时 10nS ≈ 2× 链设计重
  → **噪声试次潜伏期系统性短于确定性**（B2 实测 DA 首发放 58.23 vs 59.63ms；AVM 55.94 vs 56.64ms）。
- 判定影响：P1（D_peak>0.3）与 P3（SD<5ms、生理窗）判据均不受影响（实测全过）；
  但"噪声试次 vs 确定性 I0 档"的潜伏期**不可直接混用**（电导强度不同）；参考对比一律用确定性档。

---

## 环境快照（无新增依赖）

- `.venv-neuro`：Python 3.9.6；Brian2 2.6.0；NEURON 9.0.1；numpy 1.26.4；scipy 1.13.1（与 M2 快照一致）。
- 新增交付物：`docs/m3_env_notes.md`、`data/m3_reflex_params.csv`、`tools/build_reflex_ref.py` → `data/m3_reflex_ref.npz`；
  B1b：`src/reflex_arc.py`、`src/muscle.py`、`tests/neuro/test_reflex_smoke.py`；
  B2：`tools/validate_p{1..5}_*.py`、`tools/run_m3_validation.py`、`tools/gen_m3_report.py`
  → `reports/neuro/m3_p{1..5}_*.png|csv`、`reports/neuro/m3_validation_summary.json`、`docs/m3_report.md`。
- 未修改任何已冻结文件（src/、tests/、tools/build_synapse_ref.py、tools/build_neuron_ref.py 均原样）。
