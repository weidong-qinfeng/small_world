# M3 验证报告：最小行为回路——触觉反射弧（感觉 → 中间 → 运动 → 肌肉）

> 里程碑：M3（4 神经元 + 2 肌肉双方向通道，White 1986 连接组极性语义简化）
> 清单：《生物仿真M3实施清单》
> 主线引擎：Brian2 2.6.0（rk4, dt=0.01ms）；参考解：NEURON 9.0.1（cvode atol/rtol=1e-8）
> 完成日期：2026-08-23（B2 验证+报告节点）
> 状态：**P1–P7 判定如下（P1–P5 生理验证 + P6 回归/报告 + P7 交接；P3/P4/P5 按主 agent 修订判据）**

---

## 1. 前置确认与 M2 交接（清单 §1，P7）

| # | 项 | 处置 |
|---|---|---|
| L1 | NeuronPair 硬编码双神经元 → ReflexArc（N 神经元链） | `src/reflex_arc.py` 新抽象（CSV 驱动链组装）；直接复用 `ChemicalSynapse`/`GapJunction`/`psp_amplitudes`/`norm_rmse`/`STIM_WINDOW_MS` 固定形状约定 |
| L2 | 参考解方案 | NEURON 全链参考（`tools/build_reflex_ref.py`）→ `data/m3_reflex_ref.npz`（61 键）；纯化学突触链，无需 scipy 缝隙独立解 |
| L3 | 参数定稿依据 | 行为学（Chalfie 1985）+ 连接组极性 + m2 突触电学基础 → `data/m3_reflex_params.csv`（唯一定稿源，实测依据 m3_env_notes §L5–L10） |
| L4 | 单位/方程约定 | 电导密度 S/m²、点电导 nS、刺激 µA/cm²（按位点面积换算）、ms/mV、Im 内向正；肌肉双通道 `dC/dt=−C/τ+Σw·δ` |

> 处置记录：`docs/m3_env_notes.md`（L1–L4 + 实测 L5–L15）。

## 2. 链拓扑/递质极性 + 参考解（清单 §2/§3）

| 连接 | 递质 | g_max (nS) | delay (ms) | 语义 |
|---|---|---|---|---|
| PLM → AVM | AMPA（谷氨酸兴奋） | 5.0 | 0.5 | 感觉→中间 |
| AVM → DA | AMPA（谷氨酸兴奋） | 5.0 | 0.5 | 中间→后退运动 |
| AVM → VB | GABA（抑制，E=−70mV） | 15.0 | 0.5 | 中间→前进运动（互斥方向机制） |
| DA → C_back | muscle（δ 增量 w） | 0.60 | 0.1 | 后退收缩 |
| VB → C_fwd | muscle（δ 增量 w） | 0.18 | 0.1 | 前进收缩（VB 张力 14µA/cm² 维持基线≈0.2） |

NEURON 9.0.1（cvode atol/rtol=1e-8，celsius=6.3，v_init=−65mV）全链 4 神经元（PLM→AVM→DA/VB）+ ExpSyn（AMPA E=0mV / GABA E=−70mV，weight µS = g_nS×1e-3，τ 3/5ms）+ NetCon （node3→soma，threshold=−20，delay=0.5ms）+ IClamp 触刺激（6 档）+ VB 张力 14µA/cm²；行为潜伏期由发放序列经同一肌肉 ODE（引擎无关）计算 → `data/m3_reflex_ref.npz`（61 键）。两引擎同参数同肌肉方程，链传导/潜伏期可比。

## 3. P1–P7 Pass 对照表（修订判据数值）

| Pass | 判据 | 结果 |
|---|---|---|
| P1 定向反应 | 有效试次 5/5 D_peak>0.3；无刺激 C_back=0、C_fwd 基线≈0.2 | ✅ 有效 5/5，失败 1；D_peak 0.352/0.352/0.352/0.352/0.352；对照 C_back=0.00、C_fwd=0.206 |
| P2 链传播正确性 | 发放严格递增；EPSP norm_rmse<10%；链传导误差<15% 或 <2ms；拓扑断言 | ✅ t: 53.65<56.64<59.63ms；norm_rmse=0.04%（I0 档补 0.06%）；链传导 5.98 vs ref 5.94ms；3 化学突触+2 肌肉驱动 ✓ |
| P3 潜伏期（修订判据） | 神经潜伏期 vs 参考 <15%；[5,20]ms 生理窗；强度-潜伏期 Spearman ρ≤−0.9；20 试次 SD<5ms | ✅ I0 9.63 vs ref 10.10ms（err 4.7%）；ρ=-1.000；SD=0.30ms（有效 20/20，失败 0）；行为潜伏期 9.74ms（差异归因见 §4） |
| P4 强度-响应（修订判据） | 0 档全链静默 C_back=0；Spearman ρ≤−0.9；8×I0/20ms → DA≥2 发放且 C_back 峰>w_back | ✅ 0 档 C_back=0.00 静默 ✓；ρ=-1.000；补充：DA n=2，C_back 峰=0.890>w_back=0.6 |
| P5 方向机制消融（修订判据，主 agent 裁决落地） | 发放数子判据 + 响应分量符号子判据（×1.2 窗均值记录为测量限制，informational） | ✅ 发放数 2→3（+1 ✓，两引擎一致）；响应分量 full=-0.0108<0 / abl=+0.0000≥0（符号 ✓）；D_peak 0.369=0.369（L8 预测复现）；窗均值比 1.058（informational，测量限制） |
| P6 | pytest 全绿 + m3_report.md 写盘 + summary JSON 存在 | ✅ pytest 32/32 绿 + 本报告 + `reports/neuro/m3_validation_summary.json` |
| P7 | M2 交接处置完成（L1–L4 + 实测 L5+） | ✅ 见 §1 与 m3_env_notes.md（L5–L15） |

> **P5 反证笔记（三态裁决请求）**：修订判据的**发放数子判据通过**（响应窗 [50,90]ms 内 VB 2→3，两引擎一致），机制结论（互斥方向由抑制连接实现）成立；但 **C_fwd 窗均值 ×1.2 幅度子判据结构性不可达**：两引擎均仅 ≈1.05（NEURON 参考 1.054、Brian2 1.058；本脚本遍历窗变体最大 1.11）。原因：VB 张力维持 ~60Hz 基线振荡主导窗均值（基线 ≈0.19–0.31 振荡），GABA 抑制只推迟/抑制响应窗内 1 个 VB 发放（75.2→71.7、89.0→93.1ms 相移），不足以使均值差 20%。D_peak 两情形相同（0.369，B1a L8 预测复现——D_peak 时刻早于 GABA 生效时刻）。响应分量（窗均值−无刺激对照）full=-0.0108 vs abl=+0.0000（消融消除抑制）。**裁决选项**：① 判据改为单一「响应窗内 VB 发放数 +1」（两引擎均通过）；② 判据改为响应分量 Δ 对比；③ 上调 w_fwd/tonic 或延长 GABA τ（需解冻 data/m3_reflex_params.csv 重定稿）。在裁决前 P5 判为**部分满足（机制成立，幅度亚判据不可达）**，不静默重试。

> **主 agent 裁决记录（2026-08-23）**：✅ **通过（判据落地修订）**。裁决理由：机制已由直接证据证明——发放数子判据两引擎一致（响应窗 [50,90]ms 内 VB 2→3）+ 响应分量符号（full=-0.0108<0 被抑制、abl=+0.0000≥0 无抑制）；×1.2 窗均值子判据因 VB 张力 ~60Hz 基线振荡主导窗均值而结构性不可达（两引擎均 ≈1.05，NEURON 1.054 / Brian2 1.058）——**系主 agent 判据校准错误，非机制失败**。修订后判据：`pass_ = count_ok AND resp_component_sign_ok`，其中 `resp_component_sign_ok = (mean_full − ctrl_mean) < 0 AND (mean_abl − ctrl_mean) ≥ −1e-6`（完整链被抑制、消融链无抑制）；×1.2 均值保留为 informational 输出。**P5 判定 ✅**。

## 4. 参数定稿理由（含潜伏期差异归因）

- **唯一定稿源**：`data/m3_reflex_params.csv`（B1a 实测定稿，依据 m3_env_notes §L5–L10）：
  AMPA 链权重 5.0nS（0.3nS 仅 0.68mV 不足驱动；阈值 ~2.5–3nS 的 1.7–2×）、GABA 15nS（60Hz 张力下 VB 跳过 1 发放）、
  突触延迟 0.5ms（与 NEURON NetCon 同值可比）、I0=60µA/cm²×5ms @ t=50ms（树突端注入）、
  VB 张力 14µA/cm²（~60Hz；近阈值 13–13.75µA 使 NEURON cvode 挂起）、肌肉 TAU=20ms、w_back=0.60/w_fwd=0.18。
- 神经潜伏期：I0 档 9.63ms（NEURON 参考 10.10ms，err 4.7% < 15%）；
  6 档单调 13.3→7.7ms（Spearman ρ=-1.000 ≤ −0.9）。
- 行为潜伏期：9.74ms（I0，清单 §5.3 原定义如实计算；NEURON 参考 10.10ms）。
  **差异归因（L7 定稿）**：本模型行为潜伏期 ≡ 神经潜伏期（I0 档 9.74 vs 9.63 ms；NEURON 参考同值 10.10 ms）。原因：① 触电流直接注入感觉神经元（无机械转导延迟）；② 肌肉为 δ 驱动一阶积分，单发放瞬间达峰 → 0.3·C_peak 判据在首个 DA 发放时刻即越过，与 TAU_MUSCLE 无关。Chalfie 1985 行为潜伏期 30–50 ms 含机械转导 + 肌肉收缩动力学，本模型两者均缺失 → 差异登记为 **M3 简化假设**（转导/肌肉动力学缺失 → M5 行为层补齐），不伪造窗口。

## 5. 踩坑记录（实测 L5–L15，详见 docs/m3_env_notes.md）

| # | 主题 | 实测结论与对策 |
|---|---|---|
| L5 | 触刺激注入位点 | 树突端注入密度→nA 按位点面积换算（胞体 1.257e-5 vs 树突端 2.356e-6 cm²，5.3× 差）；I0=60µA/cm² 树突端可靠发放 |
| L6 | 链权重定稿 | AMPA 0.3nS→0.68mV 不足驱动；5.0nS 可靠发放（阈值 ~2.5–3nS 的 1.7–2×）；GABA 15nS 使 60Hz 张力 VB 跳过 1 发放 |
| L7 | 行为潜伏期结构性矛盾 | δ 驱动肌肉单发放瞬间达峰 → 行为潜伏期≡神经潜伏期（8–14ms），落不到 [25,60]ms 窗；P3 修订判据为回路生理窗 [5,20]ms + 强度单调 |
| L8 | P5 判据时序敏感 | D_peak 出现在 DA 发放时刻（t≈60ms），早于 GABA 生效（VB 下一发放 ≈71.7ms）→ 原峰值类判据全失效；修订为发放数/窗均值，其中窗均值×1.2 仍不可达（见 §3 反证笔记） |
| L9 | NEURON cvode 近阈值挂起 | VB 张力 13–13.75µA/cm² 近阈值持续电流 cvode 挂起 >5min → 定稿 14µA/cm²（60Hz 稳定区） |
| L10 | 每档单发放→幅度判据失效 | 5–10ms 触刺激每档单发放 → C_back 峰恒为 w_back → 原 P4 幅度 ρ≥0.9 结构性失效；修订为潜伏期单调主判据 + 8×I0/20ms 多发放补充 |
| L11 | Brian2 事件代码限制 | Brian2 2.6 事件代码（on_pre）不解析 min/max、if/else 分支 → 肌肉饱和用 clip()（muscle.py 已实现） |
| L12 | store/restore 语义 | Network.store()/restore() 会重置 monitor → run_trials 每试次返回完整轨迹（比 NeuronPair 累积-切片更正确）；restore 后变量（g 清零/神经元/肌肉复位）回到快照 |
| L13 | 构造参数默认值 | ReflexArc 构造参数默认值必须 None（否则静默覆盖 CSV 唯一定稿源）——dt_ms/method/t_total_ms/seed 均需 None 默认 |
| L14 | P5 消融复现（B2） | 删 AVM→VB 后 D_peak≈0.37 完全不变（GABA 生效晚于 D_peak 时刻，L8 预测复现）；修订判据中发放数子判据通过（2→3）、C_fwd 窗均值 ×1.2 子判据不可达（两引擎 1.05）——见 §3 反证笔记 |
| L15 | 量子释放语义（B2 实测新坑） | g_max_ns 为单囊泡量子电导：确定性 p=1/n=1 → 单量子=全链重（AMPA 5nS）；噪声 p=0.95/n=2 → 事件电导 k×5nS（均值 9.5nS，k=2 时 10nS ≈ 2× 链设计重）→ 噪声试次潜伏期系统性短于确定性（DA 58.23 vs 59.63ms）。非 bug（M2 量子语义延续），但 P1/P3 噪声试次与确定性 I0 档不在同一电导强度上；判据（D_peak>0.3、SD<5ms）不受影响 |

## 6. M4 交接（清单 §9 入口）

- **接口复用**：`ReflexArc` 多神经元角色/位点命名（plm_/avm_/da_/vb_ 前缀）与刺激协议扩展为「环境→感觉」接口（M3 触刺激 TimedArray 注入模式 → M4 食物梯度场输入）；肌肉双通道/方向判定升级为趋化运动（正弦爬行雏形）；多神经元链组装模式（CSV 驱动 + 确定性铁律 + 编译缓存纪律）与验证报告结构沿用。
- **新建**：ASE 感觉神经元（ON/OFF 细胞对编码）→ AIY/AIB 中间 → 运动，~20 神经元链；虚拟二维环境 + 食物梯度（L5 环境雏形）。
- **验证目标**：趋化指数 CI 显著 >0 且与真实虫一致（Ward 1973; Pierce-Shimomura et al. 1999，CI≈0.3–0.7）。
- **反证路径（M4）**：若无法趋化 → 检查 ASE 编码机制（ON/OFF 细胞对）→ 或检查运动神经元两侧激活的平衡/竞争机制。
- **复现入口**：`python -m neural_exploration.tools.run_m3_validation`（P1–P5 全跑，P5 裁决后可加 --skip-p5）+ `pytest neural_exploration/tests`（P6）。
