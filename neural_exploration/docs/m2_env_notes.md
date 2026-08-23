# M2 环境与前置处置记录（清单 §1：L1–L4）

> 对应《生物仿真M2实施清单》§1。
> 记录时间：2026-08-22（M2 实施期）
> 状态：L1–L4 处置完成；另有 M2 实测结论（L5–L8）供复现/报告引用

---

## L1 — node3 轴突末梢作为突触前位点

- M1 `MultiCompartmentNeuron` 已暴露逐隔室索引（`label_of` / `index_map`），
  node3 即轴突末梢释放位点（M1 报告 §10 交接）。
- M2 实现：Brian2 `Synapses(pre.neuron, post.neuron, ...)` 以 SpatialNeuron 为源，
  逐隔室发放事件；`connect(i=node3_idx, j=soma_idx)` 使 `on_pre` 仅在 node3
  跨阈值（-20mV）时触发。**实测可用**（与 M1 SpikeMonitor 同源机制）。

## L2 — NEURON 突触参考解

- 化学突触：NEURON 9.0.1 **ExpSyn**（AMPA/GABA，weight 单位 **µS**，
  `weight_µS = g_nS × 1e-3`）+ 自编译 **NMDASyn**（Mg²⁺ 阻断，见下）。
- 缝隙连接：**本机 pip 安装的 NEURON 9.0.1 运行时不导出经典 gap.mod 所需的
  EXTERNAL `vother` 符号**（nrnivmodl 编译通过，但 dlopen 报
  `symbol not found in flat namespace '_vother'`，`nm -g libnrniv.dylib` 亦无该符号）。
  → 缝隙连接参考改用 **scipy solve_ivp（LSODA, rtol=1e-10）独立高精度解**
  （两等势 HH 胞体 + 欧姆耦合，同方程同参数）。P4 判据为定性特征
  （近即时/双向/衰减快），不依赖 NEURON 形态学复刻；差异记录于 m2_report.md。
- NMDASyn 机制（tools/nmodl/nmda.mod）：**i 必须声明为 (nA)、g 为 (µS)**
  ——NEURON 点过程电流按 nA 注入，若按 pA 声明（nS 电导），生成的 C 代码
  `_current += i` 无单位换算，电流被放大 1000×（M2 实测踩坑：NMDA EPSP 直接触发
  后膜发放）。修正后 g_peak = weight·B(V) 与 Jahr–Stevens 理论一致到 9 位小数。

## L3 — 释放模型参数定稿（文献支撑）

- 量子电导 q：AMPA 0.3 nS/量子 → 胞体 EPSP ≈ 0.68 mV（M1 胞体输入电阻 ~100–265 MΩ）。
  文献：量子大小 0.5–1 mV（清单 §1 L3），本模型取 0.68 mV（胞体 20µm 更小，合理）。
- 二项模型：k ~ Binomial(n=3, p=0.3)，失败率 (1-p)^n = 0.343 ≈ 清单 §5.2 的 ≈0.3；
  `data/m2_synapse_params.csv` ampa 行即定稿值；确定性实验（P1/P3/P5）运行时
  显式覆盖 p=1/n=1。
- NMDA：[Mg²⁺]=1.2 mM，B(V)=1/(1+[Mg]·exp(-0.062V)/3.57)（Jahr & Stevens 1990）。
- STP：Tsodyks & Markram 1998；易化 U=0.03/τfac=120ms/τrec=40ms，
  抑制 U=0.6/τfac=10ms/τrec=400ms（参数以"能干净复现单调趋势"定稿）。

## L4 — 单位/方程约定

- 突触后电导在 Brian2 侧为**密度 S/m²**（Im 是 amp/m² 密度，M1 约定延续）；
  CSV 用生理点电导 nS，构建时按胞体面积换算：`g_density = g_nS·1e-9 / A_soma`。
- 与 NEURON 可比性：ExpSyn 点电导 g_abs（µS）→ Brian2 密度 g_abs/A_soma；
  等势胞体下总电流一致（M1 P2 已实证两引擎离散化可比）。
- Im 内向正约定延续 M1（`g*(E-v)` 加入 Im 求和）；GABA E=-70mV（Cl⁻）→ 超极化。
- 缝隙连接：`I_gap = g_gap·(V_pre - V_post)`，点电流注入（amp），双向。

---

## M2 实测结论（L5–L8，供复现）

### L5 — Brian2 `Synapses` 多隔室实测要点
1. `on_pre` 中更新 post 组电导用 `_post` 后缀：`g_ampa_post = g_ampa_post + X`
   （Brian2 2.6 不允许在 Synapses model 里声明 `_post` 变量名——保留后缀）。
2. **同一条 on_pre 语句内至多一次 `rand()`**：多次调用报
   “more than one call of rand” NotImplementedError → 多囊泡量子释放
   拆成多条单 rand 语句累加；p=1 时生成无 rand 的确定性语句（免 abstract-code 警告）。
3. `rand()` 与数值混合会触发 sympy 解析失败（Relational in Mul）→ 用
   `int(rand() < p)` 显式转换后参与算术。
4. STP 状态变量 u/x 用 `(clock-driven)` ODE（`du/dt=(U0-u)/(TAUFAC*ms)`）；
   初值 `syn.u = U0; syn.x = 1.0` 在 connect 后赋值。
5. 缝隙连接：Synapses model 内 `I_couple = g*(v_pre - v_post) : amp`，
   `(summed)` 变量 `I_gap_post/I_gap_pre` 写回两侧神经元的
   `I_gap : amp (point current)` 参数（每时间步连续、双向、精确互反）。
6. 多试次：`Network.store()/restore()` 复用网络（免重建），monitor 每次
   restore 后重置 → 每试次取全数组即可；`seed(base+trial)` 逐试次重播种。

### L6 — 编译缓存（性能关键）
- Brian2 cython 缓存默认不落盘（每进程重编译 80–120s，M1 记录 10–30s 偏乐观）。
- **TimedArray 的行数（时间窗）与对象名进入生成代码串**：变化即触发整组重编译。
  → M2 统一 `STIM_WINDOW_MS=500` 的固定形状刺激数组 + 显式命名
  （`stim_pre`/`stim_post`、`mon_pre_v`/`mon_post_v`、`sp_pre`/`sp_post`）。
- **Synapses 参数不要格式化进字符串**：U/tau/g_max/mg 等经 Synapses `namespace`
  传入（标识符 GMAXD/PREL/MGMM/U0/TAUFAC/TAUREC），值不进入代码串，
  不同参数组合共享同一编译产物。
- 缓存目录改为项目内 `.cache/brian2`（gitignore），`delete_source_files=False`
  （保留 .pyx 避免缓存校验失败重编译；/tmp 下偶发 macOS Spotlight 索引卡顿）。
- 冷缓存首次全量编译约 10–15 分钟；预热后各实验 1–5s/次。

### L7 — 刺激参数
- 单脉冲 20µA/cm² × 1ms @ t=50ms：从静息（-63mV，等 M1 瞬态漂移衰减后）可靠发放。
  - 15µA 在无突触时也能发放，但缝隙连接的分流（0.5nS ≈ 输入电导 13%）会抑制
    边缘发放 → 统一 20µA。
  - 首脉冲必须 ≥ t=40ms：HH 静息瞬态漂移（-65→-63/胞体、-57.5/node3）未衰减时
    发放阈值不可复现（M1 env_notes §L3 同结论）。

### L8 — NEURON Python 点过程引用
- **IClamp/NetCon 等点过程必须保留 Python 引用**：循环里复用变量名会被 GC 回收，
  只剩最后一个生效（M2 实测：50Hz×10 训练只剩第 10 个脉冲触发）。

## 环境快照（无新增依赖）

- `.venv-neuro`：Python 3.9.6；Brian2 2.6.0；NEURON 9.0.1；numpy 1.26.4；
  scipy 1.13.1；matplotlib 3.9.4；pytest 8.4.2
- 新增：`tools/nmodl/nmda.mod` 自编译机制（nrnivmodl → tools/nmodl/arm64/，已 gitignore）
- 版本锁定：`docs/m0_requirements.lock`（未新增 pip 包）
