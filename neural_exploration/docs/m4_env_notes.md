# M4 环境与前置处置记录（清单 §1：L1–L4 + 执行节点实测 L5+）

> 对应《生物仿真M4实施清单》§1（P8 交付物）。
> 记录时间：M4 实施期（B1a 执行节点：参数定稿 + NEURON 子链参考解 + 行为参考模型；
> B1b 执行节点：Brian2 实现 + 冒烟测试，并行推进）。
> 状态：L1–L4 处置完成 + L5–L8 B1a 实测结论（参考解已生成，M3 惯例）。

---

## L1 — 交接：ReflexArc/多神经元链 → ChemotaxisCircuit（环境→感觉接口，~20 神经元）

- **组合不修改纪律（冻结清单）**：`src/reflex_arc.py`、`src/muscle.py`、`src/neuron_model.py`、
  `src/synapse_model.py`（M3 冻结基线）一律不改——回归保护（M0–M3 测试必须保持绿）。
  M4 所有新代码在**新建文件** `src/chemotaxis_circuit.py`（+ `chemotaxis_env.py` /
  `chemotaxis_body.py` / `chemotaxis_loop.py`，由 B1b 实现）中完成。
- **直接复用清单（不改造）**：
  - `MultiCompartmentNeuron`（src/neuron_model.py）：`label_of` / `density_to_nA` / `soma_area_cm2` /
    `extra_eqs` / `extra_im_terms` / `stim_var` 钩子——ASE 转导电流经 `stim_var` 注入（M3 触刺激同款）；
  - `ChemicalSynapse`（src/synapse_model.py）：`pre_site="node3"` → `post_site="soma"`（与 M3 链一致）；
  - `chemical_post_eqs` / `chemical_im_terms` / `load_synapse_params`（ampa/gaba 电学基础沿用
    `data/m2_synapse_params.csv`：ampa 0.3nS/τ3ms/E0mV、gaba 1.5nS/τ5ms/E−70mV）；
  - `Muscle`（src/muscle.py）的**模式**：`connect_driver` / `monitor` / `groups` / `drivers`、
    on_pre 增量 + `clip()` 饱和（M3 L11 事件代码限制）、node3 触发位点、delay 0.1ms；
  - `STIM_WINDOW_MS=500` 固定形状 + TimedArray 显式命名 + namespace 传参纪律（M2 L6 编译缓存）——
    M4 闭环 epoch 帧缓存命中前提（清单 §8 风险表第一条）；
  - store/restore + 重播种多试次模式（M3 L12）：每试次 `net.restore()` 后 monitor 即该试次完整轨迹；
  - `psp_amplitudes` / `norm_rmse` / `binomial_ci`（tools/synapse_metrics.py）——P2 参考对照；
  - `_record_spec` / `_spike_times` 的 role_ 前缀约定（M4 扩展为 asel_/aser_/aiyl_/...）。
- **Muscle 双通道 → 三通道（本节点定稿建议：方案 ① Muscle3）**：
  - 需求：M4 需三通道（C_fwd 前进 + C_left/C_right 左右转向，清单 §2.4
    `v = v_fwd0·clip(C_fwd,0,1)`、`ω = ω_max·(C_left − C_right)`）。
  - `Muscle.connect_driver` 校验 `channel ∈ {back, fwd}`（muscle.py L170）→ 冻结文件不可改。
  - 方案 ①：在 `src/chemotaxis_circuit.py` 模块内新建 `Muscle3`（参数化通道数，照 Muscle 模式：
    build/connect_driver/monitor/get，每通道单变量 NeuronGroup，on_pre 增量 + clip 饱和，node3 触发）；
  - 方案 ②：复用两个 `Muscle` 实例（back/fwd 硬映射为 left/right + 单独前向）——通道命名语义错位
    （C_back/C_fwd 变量名、monitor 固定两通道）→ 混淆与 bug 风险高。
  - **定稿：方案 ① Muscle3**（B1b 在 chemotaxis 模块内实现，不改冻结文件）。
- 复用注意：多隔室 + Synapses 事件耦合在 20 神经元链中会放大（SESSION_CONTEXT §四 #10）——
  先两条子链（链 A 3 神经元、链 B 4 神经元）冒烟再合并全链 + 闭环（清单 §2.1 最小链冒烟基线）。

## L2 — 参考解方案：NEURON 核心子链 + 行为参考模型（本节点交付）

- **两级参考（清单 §3）**：
  1. **神经级（NEURON 核心子链，不复制全 20 神经元）**：
     - 链 A（前进）：ASEL → AIYL → AVBL（3 神经元，AMPA 链）；
     - 链 B（转向）：ASER → AIBL → RIAL → SMDDL（4 神经元，AMPA 链）；
     - 模式照 `tools/build_reflex_ref.py`：`build_neuron(spec, clear, name_prefix)` 构建多隔室
       神经元（clear=True 首神经元、clear=False 续建）；ExpSyn（AMPA E=0mV/τ=3ms，weight µS =
       g_nS×1e-3）；NetCon(node3→soma, threshold=−20, delay=CSV 0.5ms)；IClamp 阶跃电流注入
       ASE **soma**（编码 ON/OFF 阶跃：上升阶跃→ASEL、下降阶跃→ASER，密度→nA 按 SOMA_AREA_CM2 换算）；
       **点过程（IClamp/ExpSyn/NetCon）列表持有防 GC**（M2 L8）；cvode atol/rtol=1e-8、
       celsius=6.3、v_init=V0（硬约束）。
     - 输出：各级 node3 发放时刻 + ASE→AIY PSP（AIYL soma，链 A）/ ASE→AIB PSP（AIBL soma，链 B）
       + 链传导时间 → `data/m4_ref.npz`。
     - 全链（20 神经元）正确性由 Brian2 拓扑断言（P2）+ 确定性 + 子链参考覆盖（参考成本控制）。
  2. **行为级（行为参考模型，纯 numpy，引擎无关）**：
     - pirouette 机制算法化（清单 §2.4）：每行为 tick 采样 C(x,y) → 时间差分 s →
       `s < −θ_pir` 触发随机转向（ω = ±ω_pir 持续 T_pir，方向随机）→ 积分位置（反射边界）→ 逐试次 CI；
     - 参数（θ_pir/ω_pir/T_pir/v_fwd/σ/τ_win）以 CI ∈ [0.3, 0.7] 为目标粗校准（本节点执行，
         步骤 4 全量 6 维扫描由验证节点联动）；
     - 与 Brian2 虫**共用同一运动学/CI 统计代码**（`src/chemotaxis_body.py`/`chemotaxis_env.py`
       的引擎无关部分由 B1b 建立；本参考工具的 CI 函数与 §2.4 定义逐条对应，B1b 移植时保持同一函数）。
- 缝隙连接（RME）：M4 默认关闭（gap.mod 不可用，M2 L2）；如需开启走 scipy solve_ivp 或 Brian2 (summed) 写回。
- 行为参考模型不设肌肉/神经细节（点身体 + 简化转向），与 M3 肌肉 ODE 哲学一致（引擎无关）。

## L3 — 参数定稿依据（文献 + 实测定稿）

- **行为学**：趋化指数 CI ≈ 0.3–0.7（Ward 1973 *PNAS* 70:817–821；Pierce-Shimomura et al. 1999
  *J Neurosci* 19:9557–9569）——P4 生物带，容差窗 [0.25, 0.75]；趋化经浓度**时间差分**（病态运动/转向，
  pirouette）实现（Pierce-Shimomura 1999）。
- **ASE 功能不对称**：ASEL = ON 细胞（浓度上升激活）、ASER = OFF 细胞（浓度下降激活），
  计算角色为时间差分编码（Suzuki et al. 2008 *Nature* 454:114–117）。
- **ASE→AIY/AIB 极性**：谷氨酸能（AMPA）；AIY 促进前进/抑制转向、AIB 促进转向
  （Chalasani et al. 2007 *Nature* 450:63–70；White 1986 连接组语义简化，同 M3 惯例）。
- **突触电学基础**：沿用 `data/m2_synapse_params.csv`（ampa/gaba 行）。
- **回路级参数定稿**于 `data/m4_chemotaxis_params.csv`（唯一定稿源）：
  - AMPA 链权重 5.0 nS / 0.5ms、GABA 链权重 15.0 nS / 0.5ms（**M3 量级**——m3_env_notes §L6 实测：
    5nS 使下游可靠发放、15nS 使 60Hz 张力神经元跳过 1 发放；M4 沿用为初值，步骤 4 校准）；
  - 转导 g_ON = g_OFF = 4.0e5 µA/cm² per (ΔC/ms)：P1 阶跃（ΔC=0.5/τ_win=5e-3/ms）→
    I = 2000 µA/cm² 可靠发放；闭环最大 |s|≈1e-4/ms → I≈40 µA/cm²（r≲2.5 内可触发，见 L4 校准注意）；
  - tau_win = 100 ms（ASE 感受时间常数量级）、Δt_b = 25 ms（行为 tick）、σ = L/8 = 1.25、
    CI 带 [0.3, 0.7]（容差 [0.25, 0.75]）——清单 §1 L3 初值；
  - 肌肉：TAU_MUSCLE=20ms（M0 惯例）、w_fwd=0.18（M3 值，AVB 张力 14µA/cm² → C_fwd 基线≈0.216
    → 有效爬行速度 ≈ v_fwd0·0.216 ≈ 0.2 u/s，与行为参考 v_fwd=0.2 一致）、w_turn=0.50（转向增量初值）；
  - 环境：arena_L=10（相对单位）、food=(7.5,7.5)（右上象限中心，drop assay 惯例，Ward 1973）、
    boundary=reflect（P3 轨迹有界要求）。

## L4 — 单位/方程/统计约定（沿用 M1–M3 + M4 新增）

- **沿用**：电导密度 S/m²、点电导 nS、刺激密度 µA/cm²（`density_to_nA` 按注入位点面积换算）、
  ms/mV、**Im 内向正**（SESSION_CONTEXT §四 #1）。
- **环境浓度（M4 新增）**：相对浓度 **0..1（无量纲）**；`C(x,y) = C_max·exp(−r²/2σ²) + C_bg`，
  r²=(x−x_f)²+(y−y_f)²；无梯度对照 `C ≡ C_bg`（C_max 段置 0）。
- **时间差分**：`s(t) = (C_sensed(t) − C_sensed(t−τ_win)) / τ_win` **[ΔC/ms]**；
  `I_ASEL = g_ON·max(s,0)`、`I_ASER = g_OFF·max(−s,0)` **[µA/cm² 密度]**；注入位点 `ase_site=soma`
  （CSV 定稿）；`nA = density_to_nA(密度, soma)`。max/min 在刺激数组（numpy 侧）完成，不进 on_pre
  （M3 L11）。
- **双时钟**：行为 tick Δt_b=25ms（闭环 epoch）vs 神经 dt=0.01ms（Brian2 rk4）；
  每 epoch：采样 C → 算 s → 组帧 epoch TimedArray（**固定 STIM_WINDOW_MS=500 形状、显式命名、pad 零**，
  M3 L6 编译缓存纪律）→ run ΔT → 运动学积分（引擎无关）更新位姿。
- **CI 定义（Pierce-Shimomura 1999 象限式，两引擎共用同一统计代码）**：
  `CI_i = (T_in − T_out)/T_total ∈ [−1,1]`（T_in=食物象限 [L/2,L]² 时间、T_out=对侧象限时间）；
  组统计：CĪ ± SEM；单样本 t 检验（H0: μ=0）；效应量 Cohen's d；无梯度对照 CĪ_ctrl（|CĪ_ctrl|<0.1 且
  p>0.05 才认可度量，清单 §8 风险表）；生物带 [0.3, 0.7] 容差 [0.25, 0.75]；吸引圈式双指标
  （ci_radius=2σ，informational）。
- **试次协议**：起点=皿中心 + 伪随机扰动（试次间方差来源；神经网络确定性）、食物=右上象限中心、
  T_total=25000ms、N=20 试次/组；无梯度对照组同协议。
- **P1 阶跃协议与 τ_win 记忆（操作化注意）**：τ_win=100ms 时，阶跃后 s≠0 持续约 τ_win（
  s(t)=(C(t)−C(t−τ_win))/τ_win 在阶跃后 100ms 内仍见阶跃）→ **「静止段两者均静默」判据在
  无阶跃对照试次（C ≡ 基线）上评估**（或阶跃段间插入 ≥τ_win 静置），避免静态段被窗口记忆污染。
- **两侧平衡/竞争设计注意（供 B1b）**：§2.4 身体方程 `ω = ω_max·(C_left − C_right)` 在完全对称
  电路下恒为 0 → 闭环净转向需**零均值 ω 涨落源 + s 调制转向频率**（pirouette 机制：下降时转向
  概率↑、上升时直行被保持——偏置来自直行/转向时长不对称，非转向方向本身，Pierce-Shimomura 1999），
  或引入**机制 B**（weathervane 头摆空间采样，§2.4 扩展）——步骤 4 校准不到带时的反证扩展路径
  （清单 §8 风险表）；P5 消融 5b（静默一侧 SMDD → 系统转向偏置/轨迹失控）依赖该设计落地。

---

## 执行节点实测结论（L5–L8，供 B1b / 规划节点复核）

### L5 — NEURON 核心子链实测（B1a）
- 链 A（ASEL→AIYL→AVBL，AMPA 5.0nS/0.5ms）：首发放 **52.01 / 54.98 / 57.94 ms**，严格递增 ✓；
  链传导（ASEL→AVBL）**5.93 ms**。
- 链 B（ASER→AIBL→RIAL→SMDDL）：**52.01 / 54.98 / 57.95 / 60.91 ms**，严格递增 ✓；
  ASER→RIA **5.94 ms**、全链（→SMDDL）**8.90 ms**。
- 各级单发放（ASE soma 阶跃 60 µA/cm²×5ms@50ms，M3 量级）；PSP（AIYL/AIBL soma）峰值
  **≈33.2 mV**（相对 −65mV 基线 ≈ +98 mV，与 M3 L6「5nS EPSP 含尖峰 ~95 mV」量级一致）。
- 两链阶段间隔 ≈3ms（5nS AMPA 驱动，复现 M3 L6 的 +2.8–3.0ms）；60 µA/cm² 远离近阈值区，
  **无 cvode 挂起**（M3 L9 对策落实）。
- 两链输出对称（同形态、同权重 → AIYL/AIBL PSP 峰值相同），符合双侧推-拉对称设计。

### L6 — 行为参考模型校准结果（pirouette，B1a 粗校准；步骤 4 全量 6 维扫描由验证节点联动）
- 校准网格：θ_pir ∈ {4e-6, 5e-6, 6e-6, 8e-6, 1e-5, 1.5e-5, 2e-5} [ΔC/ms] ×
  转向角 {45°,60°,75°,90°,120°} × v {0.15, 0.20} u/s（70 组合 × N=20 试次）。
- **定稿组合：θ_pir=4e-6 [ΔC/ms]、转向角 90°（ω_pir=1 rad/s × T_pir=1571ms）、v_fwd=0.20 u/s**
  （= CSV 有效爬行速度 v_fwd0·C_fwd_baseline≈0.216 的锚点，协议一致性）。
- 结果：**CĪ=0.449 ± 0.098 (SEM)**，单样本 t=4.59，**p=0.0002，Cohen's d=1.03** →
  **落带 [0.3,0.7] ✓（近带中心 0.5）**，p<0.05 且 d≥0.5（P4 统计判据同型可满足）。
- 敏感性：v=0.20 时带内组合集中在 θ_pir≈4e-6、45–60° 附近（CI≈0.35–0.45）；v=0.15×120° 大转角
  亦落带（CI≈0.47）——粗校准对 (θ_pir, 转角, v) 敏感 → **步骤 4 需与转导增益 g_ON/g_OFF、
  ω_max、T_total 联合扫描**（B2 节点执行，判据可达性预注册见清单 §0）。

### L7 — ⚠️ 无梯度对照 CI 判据的有限样本测量限制（B1a 实测，供规划节点裁决）
- 对照（C≡C_bg）下 s≡0 → 永无转向 → 直行 + 反射边界 → **每试次 CI 呈 {−1,0,+1} 三峰分布**
  （直行虫被反射边界困在某角区整试次，实测 sd≈0.76–0.85）；
- N=20 → SEM≈0.17–0.19 → 实测 **CĪ_ctrl=0.117（p=0.503）**——**统计检验（与 0 无显著差异，
  p>0.05）通过；但 |CĪ|<0.1 点判据在有限样本下不可靠**（jitter 对称 + θ₀ 均匀 → E[CI]=0，
  偏差纯属抽样波动；本协议+种子约 40% 概率超阈）。
- 结构性原因：象限式 CI 对角区驻留敏感 + 直行对照无空间混合 → 高方差；真实虫无食物时仍有
  自发转向（随机游走，Pierce-Shimomura 1999）——本模型对照与确定性 Brian2 虫一致
  （无梯度→无 ASE 发放→直行，P3 联动时将同分布）。
- **建议（择一，由规划节点裁决，M3 L14 同款处置）**：① P3 对照主判据用统计显著性（p>0.05，
  本参考已通过）+ |CĪ| 点值 informational；② 对照协议引入自发转向率（需 B1b 在闭环加入
  自发 SMDD 驱动——模型变更，需批准）；③ 增大 N（40–60）压低 SEM。
- 影响：行为参考对照 CĪ=0.117 如实落盘（`behavior_ci_ctrl*` 键），不私自改判。

### L8 — 运行耗时与环境快照
- 全链参考解（NEURON 2 条子链 × 150ms cvode 1e-8 + 行为校准 70 组合 × 20 试次）总耗时
  **≈2.1 s（real）**——无长时程风险，闭环 epoch 预算（清单 §8 风险表）另计。
- 无新增依赖（Python 3.9.6 / Brian2 2.6.0 / NEURON 9.0.1 / numpy 1.26.4 / scipy 1.13.1，与 M3 快照一致）。
- 新增交付物（B1a）：`docs/m4_env_notes.md`、`data/m4_chemotaxis_params.csv`、
  `tools/build_chemotaxis_ref.py` → `data/m4_ref.npz`（28 键）。未修改任何冻结文件
  （src/、tests/、tools/build_{synapse,neuron,reflex}_ref.py 均原样）。

---

## 执行节点实测结论（L9–L15，B1b Brian2 实现 + 冒烟测试实测）

### L9 — ⚠️ CSV value/note 列分离（解析器必须"value 优先、note 兜底"，不可拼接）
- B1a 惯例：**数值写在 value 列、说明文字写在 note 列**（与 M3 的"值可能写在 note 列"不同——
  M3 是 value 空、note 放值）。若把 value+note 拼接成一个字符串再转 float
  （M3 load_reflex_params 的拼接逻辑），M4 CSV 会得到 `"4.0e5,I_ASEL = ..."` → 解析失败。
- B1b 实现：`load_chemotaxis_params` 值提取改为 **value 列优先、空则 note 列、再退到 to 列/
  表头外扩展字段**（M3 兼容），单列取值不拼接。

### L10 — 滑窗差分左侧填充（τ_win > 段长时的阶跃响应记忆）
- `s(t) = (C(t)−C(t−τ_win))/τ_win` 在 τ_win=100ms > P1 段长 50ms 时，浓度阶跃后的差分响应
  **持续 τ_win**（上升段 + 静止段都 s>0，实测 ASEL 注入 [40,140)ms）——这是滑窗差分的固有记忆，
  不是 bug（B1a L4 已预注册：静止段判据在无阶跃对照试次上评估）。
- 实现注意：`time_diff_trace` 必须**左侧以 C[0] 填充**（窗口未满时 s=(C[i]−C[0])/(k·dt)），
  否则阶跃响应整体滞后 τ_win，P1 上升窗 [40,90] 内看不到 ASEL 发放（预验证实测确认）。
- 测试落地：上升/下降窗断言用 meta['protocol'] 的 rise/fall 窗；下降段 ASEL 静默断言留
  ≥10ms 在途发放边界（ASEL 注入到下降起点才停）。

### L11 — 张力角色（AVB 14µA/cm²）与链次序断言语义
- AVBL/AVBR 有 tonic 14µA/cm² → ~60Hz 自发发放（C_fwd 基线≈0.216，与 B1a L3 一致）；
  链 A 的 AVBL **首发放 ≈2.5ms < ASEL ≈40.6ms**（张力，非链传播违反）。
- P2"严格递增"断言必须用 **"上游发放后的首个下游发放"（first_after）** 语义：
  t_ASEL=40.57 < t_AIYL=43.56 < t_AVB(首 after AIYL)≈47.8 < t_VB(首 after AVB)≈49.5 ✓；
  链 B 无张力：ASER 140.57 < AIBL 143.56 < RIAL 146.55 < SMDDL 149.54（严格递增 ✓）。
- 实测 P1 强注入：g_ON=4e5 → P1 阶跃 I≈2000µA/cm²（≈25nA soma）仍数值稳定（HH rk4），
  单发放 @40.57ms；闭环 |s|≈1e-4 → I≈40µA/cm² 为正常触发区（B1a L3 设计）。

### L12 — TimedArray 原位改写 + 越界钳位（闭环 epoch 缓存命中的机制验证）
- Brian2 2.6 实测：`TimedArray.values` 返回**裸 ndarray（SI 单位）**，epoch 间原位改写
  （`ta.values[i0:i1, idx] = nA*1e-9`）→ 同一 Network 连续 `net.run(ΔT)` 读最新数值，
  **不触发重编译**（形状/名字不变，M2 L6 纪律成立）——闭环 epoch 迭代可行性的直接证据。
- 越界索引钳位实测：`TimedArray` 对 `t/dt` 越界索引返回末行 → 未受激角色用 **(1, n_comp)
  零数组** 恒 0，内存从"20 角色 × 全试次"（25s 试次 ≈ 2.5M×18×8B×20 ≈ 7GB）降到
  "ASEL/ASER/张力角色 全试次 + 其余 (1,n)"（≈2×550MB @25s）。
- store/restore 不作用于 TimedArray（非 Network 对象）→ 每试次 reset() 先清零再
  `_fill_tonic()` 重填张力（张力是常数，restore 不恢复 stim 数组）。

### L13 — 象限式 CI 边界实现细节（B1b 实测修正）
- 象限映射：x==L/2 或 y==L/2（中心线）→ 0（不计入任何象限）；上/下再左/右：
  右上=1、左上=2、左下=3（对侧）、右下=4——**初版"先定上下再 ±1"写法在左下象限
  给出 2 而非 3（会把 T_out 低估成 0 → CI 系统性偏置），已修正并穷举断言**。
- 闭环对照实测：θ₀=0 直行虫 y 恒 =L/2（中心线）→ 全试次贡献 0 → CI=0.0（无梯度对照
  |CĪ| 判据的测量限制见 B1a L7，本冒烟只要求"CI 可计算 ∈[−1,1]"）。

### L14 — 冒烟实测时序与性能预算
- 全链 20 神经元 + 18 化学突触 + 6 肌肉驱动 + 3 通道肌肉 → **~193 个 cython kernel
  冷编译 ≈ 68 min**（一次性，缓存落盘 .cache/brian2）；预热后冒烟 8 测试 ≈ 6.8 min、
  全量 40 测试 ≈ 10.4 min——清单 §8 风险表"冷缓存后预热 1 epoch"已落实（预验证先行）。
- P1 开环单次 190ms ≈ 17–30s（20 神经元 rk4 dt=0.01）；闭环 400ms（16 epoch）≈ 37s/试次。
- 25s 全量试次预算（B2/B1c 执行时）：25s/0.01 = 2.5M 步 × 20 神经元，单试次预计 ~20–40 min
  （或经 ΔT/校准缩短）——风险表"计算量失控"条目需 B2 实测记录。

### L15 — B1b 交付核对
- 新增：`src/chemotaxis_env.py`、`src/chemotaxis_body.py`、`src/chemotaxis_circuit.py`
  （含 Muscle3）、`src/chemotaxis_loop.py`、`tests/neuro/test_chemotaxis_smoke.py`、
  `reports/neuro/m4_smoke.png`。
- 未修改任何冻结文件（src/reflex_arc.py、src/muscle.py、src/neuron_model.py、
  src/synapse_model.py、tests/ 既有文件均原样）。
- 冒烟 8/8 绿（拓扑/P1 上升/P1 下降/P1 静止/链次序/无梯度轨迹/闭环确定性/出图）；
  全量 40/40 绿（M0 4 + M1 10 + M2 11 + M3 7 + M4 8）。

---

## 执行节点实测结论（L16–L20，B1c 机制 A 落地 + 参数定稿与 CI 校准）

### L16 — 机制 A 落地修订（pirouette 转向事件，主 agent 裁决 2026-08-23 落地）
- **实现位置（最小侵入）**：body 层 `ChemotaxisBody` 加转向事件接口
  （`trigger_turn(direction, omega_pir, t_pir_ms)` / `is_turning()` / 转向中
  `turn_rate() = turn_dir·ω_pir`，事件计时在 step/step_exact 内消耗）；
  loop 层（`chemotaxis_loop.py::_session_trial`）触发：每 epoch 检查
  **`s < −θ_pir` 且 `ChemoSession.any_spikes_in_window(("SMDDL","SMDDR"), t_e, t_e+dt_b)`
  （电路耦合：ASER→AIB→RIA→SMDD 本 epoch 内激活）** → `trigger_turn`；
  方向 = `np.random.default_rng(trial_seed)` 抽签（±1）——试次种子固定 → 可复现，
  试次间随机（真实虫 pirouette 方向随机）；s>0 → ASEL→AIY 无触发 + AIY→RIA GABA
  抑制（电路内建，见 L17）——偏置来自直行/转向时长不对称（Pierce-Shimomura 1999）。
- **CSV 定稿行**：`role=mechanism_a`（theta_pir/omega_pir/t_pir_ms/enabled）；
  冒烟测试新增 `test_mechanism_a_turn_events`（梯度+背向食物起点 → SMDD 发放 +
  转向事件 ≥1 + 同种子重跑逐位一致）——冒烟 9/9。
- **⚠️ 实测坑（闭环 s 量级 vs ASE 发放阈值）**：闭环梯度下 |s| 仅 ~1e-5–1e-4
  [ΔC/ms]（τ_win=100ms、v≈0.4 u/s、σ=1.25 实测；r≳3 处 |s|≲3e-5）。
  原 g_ON=g_OFF=4e5 → ASER 需 |s|≳1e-4（I≈40µA/cm²）才发放 → **闭环中 ASE 几乎
  不发放 → SMDD 不激活 → 机制 A 零转向 → CI=0**（probe 实测 ASEL=0/ASER=0/SMDD=0）。
  → g_off 需上修到 ~1e6–1e7（θ_eff = I_thresh/g_off 降至 ~5e-6，与参考 θ_pir 同量级）。
- **⚠️ 实测坑（AIY→RIA GABA 抑制随 g 增强）**：g=8e6 近食物试次 SMDD 发放反而少
  （g=4e5: 160 spikes vs g=8e6: 2 spikes）——高 g 下 ASEL（s>0）也强发放 → AIY→RIA
  GABA（15nS，E=−70mV）持续压制 RIA → SMDD 被抑制（**电路内建的"AIY 抑制转向"**）。
  → 有效转向区 = {s<−θ_pir} ∩ {ASER 链激活} ∩ {AIY 未压制}，校准以实测 CI 为准。
- **⚠️ 计算预算实测**：闭环单试次 ≈ **156s 每 1000ms 仿真**（20 神经元 HH rk4
  dt=0.01ms，B1b L14 的 0.0925s/ms 低估 1.7×）→ 25s 试次 ≈ 65 min。
  dt=0.02 扫描加速**不可行**（Brian2 对 dt 变更重新编译且实测更慢 865s vs 318s）→
  扫描固定 dt=0.01；网格点数按预算收缩（阶段 1 numpy 快扫 + 阶段 2 Brian2 少点并行）。
- **对照复用判据（记录在案）**：无梯度（C≡C_bg）→ s≡0 → 零 ASE 发放 → 零转向事件
  → 对照轨迹 = 直行+反射（纯几何，与 g/θ_pir 无关，仅依赖 v_fwd0/协议）→ 扫描期
  对照由 numpy 直行运动学一次算定（每 v_fwd0），最终点用真实 Brian2 对照复核。

### L17 — 行为参考模型（pirouette）同步校准（步骤 4 联动）
- `tools/build_chemotaxis_ref.py` 增读 `role=mechanism_a` 行，pirouette 校准以
  CSV 机制 A 参数为锚点（θ_pir 锚入网格、ω_pir 用 CSV 值、v 锚 = body.v_fwd0×
  C_FWD_BASELINE=0.41——**B1c 实测 AVB 张力基线 C_fwd≈0.41**，非 B1a 估的 0.216，
  故 v_fwd0=1.0 → 有效 v≈0.41 u/s；v_fwd0=0.5 → v≈0.205 ≈ B1a 锚 0.20）。
- **T 敏感性实测（N=30 参考扫描）**：θ_pir=4e-6、α=90°、v≈0.2 下
  CĪ(T=10s)=0.219 → 0.255(15s) → 0.313(25s)——**带内组合需要 T≥25s**；
  θ_pir=1e-5 → CI≈0.05、θ_pir=2e-5 → CI≈−0.05（阈值越大转向越少 → 趋化越弱，
  与 pirouette 机制一致：小阈值 = 强偏置）→ **机制 A 参数定稿 θ_pir≈4e-6 为下限**。
- 参考模型无梯度对照：CĪ_ctrl≈−0.09~−0.10，p=0.43~0.50（N=30，p>0.05 ✓；
  |CĪ| 点值 informational，B1a L7 同款处置）。

### L18 — 校准协议执行（阶段 1 numpy + 阶段 2 Brian2，B1c）
- 阶段 1（快）：numpy 行为参考扫描（θ_pir×α×v_fwd0×σ×τ_win×T，共享 body/env/CI 代码）
  定候选带 → 阶段 2（慢）：Brian2 闭环候选点扫描（每点 N=10、确定性、无梯度对照并行，
  多进程并行）→ 选最小充分组合（D4：T_total 最短且落带）→ 最终点全协议复核
  （含真实 Brian2 对照）→ 写 CSV 定稿 + `data/m4_calibration.csv` +
  `reports/neuro/m4_calibration.png`。
- 产出入口：`tools/calibrate_m4_chemotaxis.py`（ref-scan / probe / scan / finalize）。

### L19 — 消融 5b 语义注意（供步骤 5 节点）
- 机制 A 转向触发判定 `any_spikes_in_window(("SMDDL","SMDDR"), ...)` 双侧任一激活即触发
  → **静默单侧 SMDD 不改变转向触发**（只影响 C_left/C_right 通道）；5b 消融应
  "静默双侧 SMDD" 或 "删除 AIY→RIA GABA"（后者放开 AIY 压制 → s>0 时 RIA 也激活 →
  转向失控/无定向偏置，符合 5b 语义）——步骤 5 判定时注意，勿用单侧静默。

### L20 — B1c 交付核对
- 修改（M4 进行中文件）：`src/chemotaxis_body.py`（转向事件）、
  `src/chemotaxis_circuit.py`（ChemoMechASpec/解析/hook/会话窗口查询）、
  `src/chemotaxis_loop.py`（机制 A 触发）、`data/m4_chemotaxis_params.csv`
  （mechanism_a 行）、`tests/neuro/test_chemotaxis_smoke.py`（机制 A 测试）、
  `tools/build_chemotaxis_ref.py`（mech_a 锚点）、新增 `tools/calibrate_m4_chemotaxis.py`。
- 未修改任何冻结文件（M0–M3 src/ 与 tests/ 零改动）；未 git commit。
- 冒烟 9/9 绿 + 全量回归（以实测为准）；`data/m4_calibration.csv` +
  `reports/neuro/m4_calibration.png` 落盘。

### L21 — 主 agent 预算裁决（2026-08-24）：闭环协议缩减 + 判据主体调整
- **背景**：B1c 实测 25s 全协议闭环单试次 ≈ 1–3.5h（20 神经元 HH rk4 dt=0.01ms，
  record=[] 后 ~84s/1000ms（死电路）~590s/1000ms（活电路；多进程并发时锁竞争恶化））
  → 全量 6 维 × N=10 的 Brian2 扫描计算不可行（预估 >100 CPU-h）。
- **裁决（记入清单 §0 P4/P6 判据落地修订）**：
  1. **Brian2 虫闭环协议缩减**（本节点按探针数据定稿）：T_total=5000ms、dt=0.01ms 保持
    （单试次 ~13–30min，N=10 用 2–3 并行 worker ~1–2h/组合）；写入 CSV 协议行并注释
    "主 agent 预算裁决"。
  2. **生物带 [0.3,0.7]（P4b）验证主体改为 numpy 行为参考模型**（全协议 T=25s、N=20；
    参考模型 CI=0.449 落带 ✓）。Brian2 虫不再要求自身 CI 落带，改为
    **ΔCI vs 同协议 numpy 参考 ≤ 0.15**（P6 同协议对比语义）。
  3. **P4(a) 显著性保持**：缩减协议 N=10、t 检验 p<0.05 且 d≥0.5；若 CI 幅度小导致
    d 不足 → 记录 T 缩放关系（参考模型 CI(T)=0.219@10s→0.313@25s，Brian2 同比例，
    不伪造，如实记录）。
  4. **P4(c) 终态偏向**：入圈比例 ≥0.5 或 终点-食物距离显著小于起点，二选一记录。
  5. **校准扫描两段**：numpy 快扫（全协议带内定稿参考模型参数）→ Brian2 少点确认
    （2–4 组合 × N=10 缩减协议，验证 ΔCI≤0.15 与显著性），不再全网格 Brian2。
  6. 产出不变：CSV 定稿 + `reports/neuro/m4_calibration.png` + `data/m4_calibration.csv`
    （含 T 缩放关系记录）+ 冒烟绿 + 全量回归绿。
  7. 若缩减协议下 ΔCI>0.15 或显著性不达标 → 不静默推进：记录反证笔记（缺失机制）+
    三态裁决请求（含机制 B weathervane 评估）。

### L22a — 缩减协议执行定稿（B1c）：方案② T_total=5000ms、dt=0.01ms
- 探针数据：dt=0.05 需全量重编译（实测 ~8min/内核、~193 内核 → ~10h+，方案①放弃）；
  dt=0.01 + record=[] 下 T=5s 单试次 ~5–18min（死电路~5min、活电路~18min，3 并发
  锁竞争 ~11–23min/试次）→ **定稿方案②**：T_total=5000ms、dt=0.01ms（写入 CSV 协议行，
  注释主 agent 预算裁决）。
- 参考模型 T 缩放（θ_pir=4e-6, α=90°, v_fwd0=0.5, N=20）：CI(T)=0.177@5s →
  0.250@10s → 0.292@15s → **0.361@25s（落带 [0.3,0.7] ✓，p=0.007，d=0.67）**；
  对照 T=5s：CĪ=−0.073，p=0.618（>0.05 ✓）。
- θ_pir 敏感性（N=20, T=25s）：2e-6→0.375、3e-6→0.367、4e-6→0.361（均落带 ✓）、
  6e-6→0.215（不落带）→ **定稿 θ_pir=4e-6**（且 T=5s 下 ΔCI 最小 0.134）。
- **Brian2 确认点（T=5s, g=8e6, θ_pir=4e-6, v_fwd0=0.5, N=10）**：CĪ=0.043±0.199
  （p=0.834，d≈0.08），turns=[0,0,0,0,0,0,1,0,2,1]（10 试次仅 4 次转向——缩减协议
  下转向稀疏，与 L22 门探针一致）；**ΔCI vs 参考(5s)=0.177 → 0.134 ≤ 0.15 ✓**；
  显著性未达（p=0.834）→ 按 L21 点 3 记录 T 缩放关系（参考 CI(T) 曲线 + 本点）。

### L22 — B1c 实测：ASE 链激活阈值与机制 A 触发条件（供缩减协议定稿）
- 开环恒定 s 注入 200ms 实测（B1c）：**ASER 持续发放区 I = g_off·|s| ≈ 15–120 µA/cm²**
  （g=8e6, s=−5e-6 → I=40µA/cm² → 19 spikes；g=4e6, s=−5e-6 → I=20µA/cm² → 14 spikes）；
  高 I（>150µA/cm²）发放反而减少（去极化区，实测 2–3 spikes）→ 机制 A 有效触发区
  = I∈[15,120]µA/cm² 即 |s|∈[15/g_off, 120/g_off]。
- **θ_eff（有效转向阈值）= max(θ_pir, I_thresh/g_off)**，I_thresh≈15µA/cm²：
  目标 θ_eff≈4e-6（参考模型带内区）→ g_off ≥ ~4e6–8e6；g_off=8e6 → θ_eff≈2–4e-6 ✓。
- 闭环门探针（g=8e6, T=3s, 中心起点）：SMDD 5/120 epochs 激活、s_min=−2.65e-6
  （< θ_pir=4e-6 未达）→ **T=5s 缩减协议下转向事件预期稀少**（s 需达 −θ_pir）；
  此即 L21 点 3「T 缩放关系」的物理来源——记录时注明。

---

### L23 — 主 agent 最终裁决：P4 行为显著性协议限制反证（2026-08-24）

> 接续：L22a 点 0（θ_pir=4e-6/T=5s/g=8e6/v=0.5 → CĪ=0.043±0.199, p=0.834 无趋化）；
> B1c2 定向调优（θ_pir 降阈值 × T=10s × v 杠杆）后主 agent 裁决：
> **M4 P4/P6(b) 行为统计显著性 = 协议限制反证记录**（记入清单 §0 P4，非机制失败）。
> 定稿参数：θ_pir=1e-6、T_total=10000ms、v_fwd0=1.0、g=8e6（`data/m4_chemotaxis_params.csv`，机制 A 最佳投注组合）。

- **背景：点 0 实测 + 决定性点终止**：
  - 点 0（θ_pir=4e-6, T=5s, v=0.5, g=8e6, N=10，L22a）：**CĪ=0.043±0.199, p=0.834,
    d=0.07, turns=0.4/试次**（10 试次仅 4 次转向）——ΔCI vs 参考(5s)=0.177 → 0.134 ≤ 0.15 ✓
    但显著性 ✗（缩减协议下转向稀疏，L22 门探针已预期）。
  - 决定性点（T=10s, v=1.0, g=8e6, **N=20**，主 agent 指定）：**因计算预算终止，无完成试次**
    ——T=10s 单试次实测 ~25–60min 墙钟（v=1.0 活电路；多进程锁竞争 + 周期停摆恶化），
    N=20 需 ~8–24h 且期间驱动多次被杀 → 结果随内存丢失（点 1 9/10 试次即如此丢失）。
- **科学结论（三组证据一致）**：
  1. **机制生物有效（参考模型全协议）**：θ_pir=1e-6/v=1.0 参考 T=25s：**CI=0.489（p=0.00）
     稳健趋化且落带 [0.3,0.7] ✓**；θ_pir=4e-6 定稿组合 N=20：CI=0.449（p=0.0002, d=1.03）
     同样落带——pirouette 机制本身能产生真实虫量级趋化。
  2. **Brian2 电路正确实现机制 A**：P1（ASE ON/OFF 编码）/P2（链传播）已验证，
     冒烟 8/8 绿、全量 40/40 绿——电路实现无缺陷，缺口在统计功效。
  3. **可行协议下显著性结构性不足（计算可行性边界）**：参考模型自身
     N=10/T=10s 通过率仅 23–40%（bootstrap 30 seeds）、N=20/T=10s 43%；稳健显著需
     **T≥15–25s**；20 神经元多隔室闭环 T≥15s×N≥20 ≈ **数千 CPU-小时**
     （本机内存压力 + Brian2 锁竞争，T=10s 单试次 ~35–60min 墙钟）→ 计算不可行。
- **T 缩放关系（参考模型，θ_pir=定稿 1e-6，N=20）**——P4(a) 记录路径（L21 点 3）：
  | T_total | 5s | 10s | 15s | 25s |
  |---|---|---|---|---|
  | CI (v_fwd0=1.0) | 0.175 | 0.317 | 0.417 | **0.494（落带）** |
  | CI (v_fwd0=0.5) | 0.177 | 0.219–0.250 | 0.292 | 0.313–0.361 |
  - 实测（seed0 定稿，落盘 `data/m4_calibration.csv` ref-T* 行）：CI=0.175@5s →
    0.317@10s (p=0.021, d=0.56) → 0.417@15s (p=0.002, d=0.79) → **0.494@25s
    (p=0.000, d=1.02)**；对照 p=0.41–0.46（>0.05 ✓，无梯度判据 L7 同款处置）。
  - 主 agent 裁决范围（bootstrap）：0.244–0.317@10s → 0.489@25s（v=1.0）——
    **显著区从 T=10s 起才有（p<0.05/d≥0.5 临界），T≥15s 稳健，T=25s 全协议落带**。
- **v 杠杆 vs σ/τ_win**：v_fwd0=0.5→1.0 转向率 ~2×（1.75–1.95 vs 0.7–1.0 次/试次，
  参考 CI(10s) 0.24–0.28 → 0.29–0.32）——**T≤10s 内唯一有效杠杆**（定稿 v_fwd0=1.0）；
  σ（1.25→2.5）、τ_win（50→200ms）几乎无影响。θ_pir 4e-6→1e-6 仅把转向率
  ~0.3→~0.5 次/试次（受转向事件时长 1571ms + 再入下降朝向动力学限制）；
  g=8e6 → θ_eff=max(θ_pir, I_thresh/g_off≈1.9e-6) 封顶 → 1e-6 与 2e-6 电路门层面等价。
- **裁决落地**：
  - **P4(b) 生物带验证主体 = numpy 参考模型全协议**（T=25s, N=20, CI=0.449/0.489 ∈ [0.3,0.7],
    p<0.001）——Brian2 虫不再要求自身落带（清单 §0 P4 修订文本）；
  - **P4(a) 显著性 = T 缩放关系 + 实测点记录**（上表 + 点 0：p=0.834/d=0.07 如实记录，不伪造）；
  - P4(c) 终态偏向以可行协议实测为准（未跑 → 记录为可选，不静默编造）；
  - **M5 教训：行为层必须采用降阶模型**（设计文档已预见）——全 20 神经元多隔室闭环
    与统计功效要求（N≥20×T≥15s）在单机计算预算内结构性不可调和。
- **决定性点续跑说明（留作 M5 前可选升级）**：`/tmp/m4_b1c2_scan_resilient.py`
  **可断点续跑**（每试次落盘、驱动重启不丢结果）；若 M5 前预算允许或换机并行，
  可补跑决定性点（T=10s, v=1.0, N=20）验证 Brian2 实测 CI 是否落参考
  ΔCI≤0.15 窗——非 M4 交付必需，不阻塞定稿。
- 交付物：`data/m4_chemotaxis_params.csv`（定稿）、`data/m4_calibration.csv`
  （点 0 + ref-T* 缩放行）、`reports/neuro/m4_calibration.png`（重绘：点 0 散点 +
  T 缩放曲线 + 生物带 [0.25,0.75] 标注 + 协议限制文字）。

---

## 执行节点实测结论（L24，M4-B2 验证+报告节点）

### L24 — P1/P2/P3/P5/P6 验证执行记录（缩短协议落地，主 agent 预算裁决 + 并发纪律）

- **并发纪律执行**：本节点开始时 `/tmp/m4_b1c2_scan_resilient.py`（决定性点 T=10s/v=1.0/
  g=8e6/N=20 断点续跑，3 worker）仍在运行 → 先在无 Brian2 条件下完成全部验证脚本编写
  （py_compile 通过）与 numpy 直行对照/ref-T 表/P4 记录的自检，再等无并发后统一跑
  P1–P6 与全量回归（L21"多进程并发时锁竞争恶化"记录支持该纪律）。
- **验证协议（任务定稿）**：P1（ASE 编码）开环阶跃 T≈190ms 正常跑；P2（拓扑/链传播）
  短协议 T≈150ms 脉冲（EPSP 对照）+ 默认阶跃（链次序）；P3/P5 用缩短协议（T≤5000ms、
  N=10、record=[]）；P4 不重跑（引用 calibration 点 0 + ref-T* 行 + L23 记录）；
  P6a 复用 P2 判定数据独立核验，P6b 引用 ref-T 表。
- **⚠️ 新坑（P2 EPSP 对照的脉冲协议）**：EPSP vs NEURON 参考需**干净单发放**——默认
  P1 阶跃协议 s>0 持续 τ_win=100ms（滑窗记忆）→ ASEL 多发放，EPSP 对齐窗受后续发放
  污染；改用短脉冲 s_trace（±I_ref/g=±60/8e6=±7.5e-6，5ms@50ms，与 NEURON 参考
  IClamp 同注入密度 60µA/cm²）→ 单发放，链 A/B 传导时间与参考误差 <2%（52→55→58 /
  52→55→58→61ms 量级，与 L5 NEURON 参考 5.93/8.90ms 一致）。
- **⚠️ 新坑（f-string 字面花括号）**：gen_m4_report.py 踩坑表含字面 `{−1,0,+1}`
  （CI 三峰分布）→ f-string 未转义花括号被当表达式 → U+2212 报"invalid character"
  （CPython 3.9 报错位置误导为 line 1）→ 转义为 `{{−1,0,+1}}`。
- **交付核对（本节点）**：新增 tools/validate_p1_ase_encoding.py、validate_p2_circuit.py、
  validate_p3_env_control.py、**validate_p5_chemotaxis_ablation.py（命名偏差：M3 已占用
  validate_p5_ablation.py，冻结不可改——见脚本头部说明）**、validate_p6_reference.py、
  run_m4_validation.py、gen_m4_report.py → reports/neuro/m4_p{1,2,3,5,6}_*.png/csv +
  m4_validation_summary.json + docs/m4_report.md。未修改任何冻结文件（M0–M3 src/ 与
  tests/ 与 tools/ 既有文件零改动）；未 git commit。

### L24b — 决定性点 N=20 完整证据 + 扫描重启冲突（主 agent 2026-08-25 介入后实测）

- **决定性点（θ_pir=1e-6, T=10s, v_fwd0=1.0, g=8e6, N=20）实际全部完成**（扫描在
  主 agent kill 前后已写满 20 试次，`/tmp/m4_res/point_4/trial_*.json`）：
  **CĪ=0.099 ± 0.139（SEM），单样本 t=0.717，p=0.482，Cohen's d=0.16——不显著**；
  ΔCI vs 参考模型(10s)=0.317 → **0.217 > 0.15**（L21 点 7 处置：记录测量限制，不静默
  推进，反证笔记）。**确证 L23 协议限制裁决**：可行协议下统计功效结构性不足
  （θ_eff=max(θ_pir, I_thresh/g_off)≈1.9e-6 > 1e-6 电路门层面等价，L22）。
  早期快照（3/20：CI=0.945/1.0/0.28"全正向"印象）被 N=20 全数据修正（试次 10–19
  多负值/近零）——教训：**N 不足时点值不可作信号解读**（L7 同型）。
- **⚠️ 扫描重启冲突**：主 agent 于 04:44 kill 扫描后，另一执行节点（node 会话
  PID 53993）经双层看门狗（watchdog3.sh ×2）于 05:01:56 **重启扫描并进入新批次
  point_5（θ_pir=2e-6, T=10s, g=8e6, N=20）**——与主 agent"不再续跑扫描"裁决冲突；
  M4-B2 验证因此再次等待并发清空（并发纪律 L21）。处置待主 agent 裁决。

---

## 执行节点实测结论（L25，M4-B2b 验证收尾 + 报告节点）

### L25 — P2 链 A 传导测量坑（相位碰撞）+ P3/P5 最短协议执行 + 全量回归

- **⚠️ 新坑（P2 链 A 传导时间 first_after=inf 的根因）**：B2 遗留的
  `data/m4_p2_circuit.csv` 中 `chain_time_sim_ms_a=inf / chain_sane_a=False /
  chain_ok=False → pass_=False`。B2b 复现定位：**AVBL 张力自发发放（~60Hz，
  2.54/19.77/37.02/54.34ms）恰在 AIYL 驱动发放（54.47ms）前 0.13ms 发放 →
  AIYL→AVBL 的 EPSP 落入 AVBL 不应期/复极期 → 无驱动发放 → `first_after(AVBL,
  AIYL)` 返回 inf**（V 轨迹证实：AVBL soma 尖峰 53.65–54.97ms，EPSP @~55ms 落
  不应期；此后 AVBL 恢复缓慢，~70ms 仍在 −57mV）。链 B 无张力角色不受影响
  （8.97 vs 8.90ms，0.79% ✓）。**对策（validate_p2_circuit.py 落地）**：
  ① 链 A 脉冲相位避让 `PULSE_START_A_MS=110`（AVBL 张力 106.38ms 后 EPSP @~115ms
  驱动发放 117.67ms → 链 A 传导 6.19ms vs 参考 5.93ms，4.4% ✓）；② 链传导改用
  **因果窗驱动尖峰搜索**（`_first_driven`：上游首发放后 [1.5,14]ms 窗口内的首个
  发放，L11 first_after 语义的加固）；③ sanity 守卫 [3,12]ms 保留。链 B 脉冲
  保持参考时序 @50ms。
- **⚠️ 僵尸进程清理**：B2 驱动（`/tmp/m4_b2_driver.sh`）遗留的
  `run_m4_validation`（PID 1982，02:06 起，P3 T=5s×12 ≈ 1194min 不可行）在
  02:25 写完 P1/P2 CSV 后**一直空转到 04:44（CPU 21.6h）**，被 B2b kill——
  教训：长任务必须有完成标记/超时看门狗（m4_b2_driver.sh 的 `touch /tmp/m4_b2_DONE`
  机制未被执行即被 kill，无人发现）。
- **P3/P5 最短协议执行（任务定稿，探针实测依据）**：探针（B2，01:15–02:06）实测
  单进程 record=[]：**无梯度对照 T=1s = 1194s 墙钟、梯度活电路 T=1s = 1888s 墙钟**
  → P3 全协议 T=5s×12 ≈ 1194min、P5 T=5s×30 ≈ 4721min 均不可行。B2b 落地：
  P3 = T=1000ms×N=2（+1 次重跑验确定性，共 3 试次 ≈ 60min）；P5 = T=1000ms×
  N=2/组（完整/5a/5b ×2 = 6 试次 ≈ 3.2h；**N=3/组需 ~4.7h 超 5h 预算 → 明确记录
  缩减**）。numpy 直行对照统计（T=5s/N=10：CĪ=−0.196 p=0.425；T=25s/N=20：
  CĪ=−0.125 p=0.456，均 p>0.05 ✓）为主判据；Brian2 部分只验有界/无 NaN/重跑一致/
  CI 有限（轨迹有界检查不需长时间，任务定稿）。
- **⚠️ 新坑（TimedArray 越界索引再确认）**：P2 步进协议运行（t_total=240ms）时
  `_n_steps=max(500, t_total)/dt` 保证 stim 数组 ≥500ms；闭环 epoch 运行无越界
  （L12 已实测）——本节点未复现越界，仅记录。
- **复用机制（计算纪律）**：validate_p1/p2/p3/p5 增加 `M4_REUSE=1` 环境变量——
  CSV 已落盘时直接读回判定（不重跑 Brian2）；P6 经 P2 复用。run_m4_validation.py
  汇总时设 M4_REUSE=1 秒级完成（P4 记录从 calibration CSV + 决定性点 JSON 组装）。
- **决定性点全数据确证（L24b 同型）**：N=20 完整：CĪ=0.099±0.139（p=0.482,
  d=0.16）不显著，ΔCI vs 参考(10s)=0.317 → 0.217 > 0.15——确证 L23 协议限制裁决
  （计算可行性边界反证，非机制失败；P1/P2 电路正确性已验证 + 冒烟/全量回归绿）。
- **交付核对（本节点）**：修改（M4 进行中文件）：tools/validate_p2_circuit.py
  （相位避让 + 因果窗 + M4_REUSE）、validate_p3_env_control.py / validate_p5_
  chemotaxis_ablation.py（最短协议 + M4_REUSE）、validate_p1_ase_encoding.py
  （M4_REUSE）、run_m4_validation.py（note 更新）、gen_m4_report.py（L25/决定性点/
  最短协议文本）→ 产出 reports/neuro/m4_p{1,2,3,5,6}_*.png/csv +
  m4_validation_summary.json + docs/m4_report.md。未修改任何冻结文件（M0–M3
  src/ 与 tests/ 与 tools/ 既有文件零改动）；未 git commit。
- **⚠️ 遗留观察（并发纪律）**：L24b 记录的 point_5 扫描（3 worker）在 B2b 执行
  P2/P3/P5 期间仍在运行（10 核机器负载 ~7.9，B2b 任务有独立核可用，未受 CPU
  饥饿；Brian2 确定性不受并发影响，仅墙钟拉伸）——是否终止待主 agent 裁决。

### L24c — 主 agent 最终科学结论与扫描彻底终止（2026-08-25 核实）
- **扫描彻底终止**：主 agent 确认点 5（θ=2e-6）扫描一并终止，`pgrep m4_b1c2` = 0——并发纪律解除。
- **决定性点 N 口径核对**：`/tmp/m4_res/point_4/` 实际含 20 试次完整数据（trial_0–19；
  主 agent 终止时点快照为 N=15）。两组统计（原始值）：
  - N=20（完整）：CĪ=0.099±0.139（sd=0.621, t=0.717, **p=0.482**, d=0.160, median=−0.025），ΔCI vs 参考(10s)=0.217；
  - N=15（快照）：CĪ=0.105±0.172（**p=0.552**；主 agent 舍入值 mean=0.085, p=0.64），ΔCI=0.211；
  - 结论一致：T=10s 可行协议下不显著（N=15 终止判据"剩余 5 试次全 +1.0 亦无法稳健显著"
    在完整数据下同样成立——实际 p=0.48~0.64 均 >0.05）。
- **主 agent 科学结论（写入报告 §3 P4）**：Brian2 电路实现机制 A 但效果减弱
  （θ_eff≈1.9e-6 电路门限 + 突触时序延迟削弱 pirouette → ΔCI vs 参考 ≈0.23）；参考模型
  T=25s 稳健落带（CI=0.494, p=0.0002, d=1.02）→ **P4 反证记录（非机制失败：机制经参考
  模型验证有效；是电路实现效率 + 计算可行性边界）**。
- 教训（复申）：N 不足时点值/早期快照不可作信号解读（3/20 "全正向" vs N=20 p=0.482）。

### L24d — ⚠️ 子 agent 长时 Brian2 进程被节流 SIGTERM（约 2h 阈值）+ 分块对策
- **实测**：主 agent 授权方案 (b) 后，P3（T=2s/N=3，wall 3828s ≈ 64min）顺利完成；
  P5（T=2s/N=3，3 组连续，wall 2h27m）在 **2.5h 处被 SIGTERM（exit=143）** 终止
  （CSV 未写入；与任务预告"子 agent 的 Brian2 计算在其长回合会被节流"一致）。
- **对策**：P5 改为**分块执行**（`M4_P5_GROUP=full|5a|5b` 单组运行 + CSV 幂等追加，
  每块 ≈1h < 节流阈值；三块后 `M4_REUSE=1` 汇总）。教训：**单进程 Brian2 长任务必须
  分块 + 每块独立落盘**（与 B2b L25"长任务须有完成标记/超时看门狗"同型教训）。
- P3 新数据（T=2s/N=3）：逐试次 CI=[−0.9625, 1.0, −0.9]，Brian2 对照 CĪ=−0.2875
  （numpy 主判据 T=5s/N=10：CĪ=−0.196, p=0.425>0.05 ✓）；有界/无 NaN/量程/重跑一致 ✓。
