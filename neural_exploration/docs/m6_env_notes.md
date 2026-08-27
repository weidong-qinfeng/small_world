# M6 环境与前置处置记录（清单 §1：L1–L6 + 执行节点实测 L7+）

> 对应《生物仿真M6实施清单》§0 P2/G1 门（神经调质系统 + M5 反证清单落地）。
> 执行节点：M6-B1a（`src/neuromod.py` + 组装层集成 + M5 P2/P4/P6/P5 复核 → G1 判定）。
> 冻结文件零修改（M0–M5 全部 src/tests/tools/data 不动；m5_connectome.csv 内容不变；
> 未 git commit）。M6 新建：`src/neuromod.py`、`data/m6_learning_params.csv`、
> `tools/validate_p6_modulation.py`、`tests/neuro/test_m6_neuromod.py`（+ 本文件）。

---

## L1 — M5 冻结基线 → M6 组装方式（组合不修改）

- **组合复用**：`make_modulated_circuit(scale=302, **load_weight_scales())` 包装 M5 冻结
  `GroupedWormCircuit`（`src/neuromod.py`：`ModulatedCircuit`/`ModulatedGroupedSession`），
  `WormLoop` **不经修改**直接消费（run_trial/run_trials/run_escape/run_spontaneous/
  run_resting 全部复用——仅 `ModulatedCircuit.run_resting` 覆盖为调质会话路径，
  因 `WormLoop.run_resting` 委托 `circuit.run_resting`）。
- 挂接点（清单 §3.2）：`neuromod.py` 在 `GroupedWormCircuit.make_session` 之后包装
  session（`apply_modulation(sess)` 接口幂等）；m5_connectome.csv 内容零修改。
- **M6 新建文件清单**：`src/neuromod.py`（ModulatorPool ODE + 四项机制 + 组装层）、
  `data/m6_learning_params.csv`（mod 参数 + mod_gating 作用域，唯一定稿源）、
  `tools/validate_p6_modulation.py`（G1 复核）、`tests/neuro/test_m6_neuromod.py`。

## L2 — M5 反证清单四项落地（实现 + 定稿参数 O2）

四项机制全部在 `neuromod.py` 组装层实现（连接组是事实不动；每项 enabled 开关可消融）：

| # | 机制 | 实现（定稿参数 O2，data/m6_learning_params.csv） | 消融 sanity |
|---|---|---|---|
| ① | RIM 酪胺 | AVA/AVD 发放率 → C_tyr ODE（τ=500ms，基线 tyr_baseline=1.0 → fwd_gate=1−0.6·C_tyr→floor 0.4）→ 门控 AVB/PVC 前进驱动 | 删酪胺 → gate≡1，夹带回潮（entrainment 复现） |
| ② | 命令互抑（功能边） | AVA/AVD ↔ AVB/PVC 组装层互抑电流（**非对称**：后退→前进 0.80nA 强 / 前进→后退 0.05nA 弱——RIM/酪胺语义）；`mod_dt_ms=5.0` 细粒度（逃避窗内反应延迟 ≤5ms） | 删互抑 → 方向相位敏感复现（touch@73ms 掉回 not_back） |
| ③ | AVA→DD/VD GABA 功能链 | AVA/AVD 归一化率 → DD/VD 驱动电流 0.80nA（真实连接组 0 条该边，功能补充）→ DD/VD 既有 gaba→fwd 运动池突触抑制 fwd 池 | 删链 → 后退 bout 混入 fwd 驱动（rev 比例回落） |
| ④ | 自发/调质输入 | 确定性伪随机脉冲（seed=20260826，p=1/n=1）**输出级运动池**（后退 DA/VA + 头 SMDD，2Hz×0.10nA×3ms）——命令池**不**注入（实测坑 L9） | 删自发 → bout 消失（pause→1） |

## L3 — 调质池模型（§3.1 规格落地）

- 浓度 ODE：`dC/dt = (R − C)/τ`（exponential_euler，与点档同款；τ 预注册 500ms）；
  C_da/C_5ht/C_tyr ∈ [0,1]；数值稳定（2000 步无 NaN/无发散，测试断言）。
- 门控单调有界：C_tyr↑ → fwd_gate↓（下限 clamp 0.3~0.4）；C_5ht → 前进增益↑
  （capped 1.2）；C_da → 运动层增益 1/(1+K·C)（Hill 型）——T2 横切层理念。
- 作用域只落有真实递质语义通路（§0 #10；mod_gating 行记录）：酪胺→AVB/PVC、
  血清素（ADF/NSM/RIH）→前进增益、多巴胺（ADE/CEP/PDE）→运动层增益。

---

## 执行节点实测结论（L7+，供规划节点复核 / 三态裁决）

## L7 — ⚠ 夹带极限环对调质机制的鲁棒性（G1 门核心实测）

**结论：M5 反证清单四项机制全部落地且功能正常（消融 sanity 可证），但 302 冻结权重
网络呈"双稳态"——调质层可把网络从 14Hz 夹带推到 100% 静默（tyr 基线→floor 时），
但**无法同时满足"网络安静（P2 静默 60-80%）+ 行为活跃（P6 fwd 60-80%）"**：**

1. **tyr 基线连续扫描**（mod_dt=25ms，无自发脉冲）：tyr_baseline 0→0.55→1.0：
   夹带 14Hz → 2Hz 全局同步（silent 10.6%）→ **100% 静默**（AVB/AVA ~1Hz）。
   调质门控确实打破夹带（14Hz→2Hz→静默），但过渡区无"AVB ~3Hz 且网络静默"的
   稳定中间态（M5 L38 观察在调质参数下复现：分岔陡峭）。
2. **任何命令池/运动池自发脉冲都会重新点燃夹带**：脉冲注入命令池（AVB/PVC/AVA/AVD，
   0.04-0.18nA 均试过）→ 网络回到 10-25Hz 全局同步（静默 ~10%）。注入**输出级
   运动池**（DA/VA+SMDD）→ 自发行为出现方向分离结构（P6 rev 落带、fwd/turn 近带），
   但静息仍夹带（silent ~10%）。
3. **根因**：夹带由全互兴奋命令回路（AVA↔AVB/PVC ampa，L40#1）+ 运动池互驱
   （motor→motor ampa 474 条）承载；任何"持续或准持续"驱动都点燃该共振环
   （~14-25Hz），调质门控只能整体关/开，不能制造"低活动 + 行为"中间态。

**三态裁决请求（G1 部分通过路径，不静默）**：
- ①（推荐）接受 G1=部分通过：**方向相位已修复**（P5 touch@73ms→back，G1 关键前置）
  → P3 习惯化协议可进入（R(n)=D_peak>0.3 的 sanity 可满足）；P2/P6 带内达标记为
  结构性限制（剩余缺失机制清单见 L8），M6 后续（P4 联想学习）在此配置上继续；
- ② 调整冻结权重/连接组（如引入真实 AVA→DD 化学边或运动池抑制权重）——超出本节点
  权限（冻结文件），需规划节点裁决；
- ③ P2/P6 验证主体改为行为参考模型（M4 P4(b) 同款处置，numpy 参考为验证主体）。

## L7b — 消融归属实测（方向相位修复的机制承载，与预注册假设的差异）

**预注册假设**（清单 §3.2）：方向相位修复主要由 ② 命令互抑承载（AVA→AVB 抑制）。
**实测（O2 配置消融）**：删 ① 酪胺门控 / ② 命令互抑 → touch@73ms 仍 back
（D_peak 均 +0.355）——方向相位修复**不是**由 ① 或 ② 单独承载，而是由
③ AVA→DD/VD GABA 链（AVA 激活→DD/VD→阻尼 fwd 池）+ ④ 自发 bout 驱动（DA/VA
后退池持续有驱动）联合承载（聚合 4 通道模型下 DD/VD gaba 输出同时覆盖 fwd/back
运动池——DD1→DA2、VD1→VA1 等真实边——"隔离 fwd 池"语义变为"双向阻尼"）。
**如实记录**：四项机制全部落地且可消融（每项 enabled 开关 + 行为可测变化），
但机制承载与预注册归属不同（③④为主、①②为辅）——方向相位修复是**多机制联合**
结果，不归因单一机制（消融结果入 data/m6_g1_result.json ablation 段）。

## L8 — M5 P2/P4/P6/P5 复核数值（同协议同带；判据带 = data/m5_behavior_reference.csv）

定稿配置 O2（tyr_baseline=1.0 / 非对称互抑 0.80/0.05 / GABA 链 0.80 / 运动池自发
2Hz×0.10nA×3ms / mod_dt=5ms（P5）或 25ms（P2/P6/P4））：

| 复核项 | 带（预注册） | M5 冻结值（反证起点） | O2 实测 | 判定 |
|---|---|---|---|---|
| P2 静默（post-settle 500ms） | [60,80]%（<0.1Hz） | 10.6%（中位 13.8Hz） | 10.3%（中位 22.5Hz，max 33.2<60 ✓） | **✗ 未缓解**（静息协议下运动池脉冲仍点燃共振环） |
| P6 自发 fwd/rev/turn | 60-80 / 10-25 / 5-20% | 25.5/3.0/0.5% | 36-43 / **10.9 ✓** / 3.4-4.3% | **△ 部分缓解**（rev 落带；fwd/turn 近带；方向分离结构涌现） |
| P4 趋化 CI（T=15s×N=20 全协议挂起，5s×N=5 探针） | 显著>0（p<0.05, d≥0.5）+ 对照 p>0.05 | −0.065（p=0.71） | 探针 CĪ=−0.263@5s（对照 +0.384） | **✗ 未缓解**（与 M5 同号反证：夹带网络无净趋化位移） |
| P5 方向相位 touch@73ms（τ_trans=23） | back（D_peak>0.3） | not_back（~72ms 节律污染，L40#5） | **back（D_peak +0.355，逐 seed 同值）** | **✓ 修复**（③+④ 联合承载，L7b；①/② 消融后仍 back） |

**G1 判定（§0 G1）**：**部分通过**——方向相位修复（G1 关键前置满足，P3 习惯化协议
可进入：R(n)=D_peak>0.3 的 sanity 可满足）；P2/P4 未缓解（M5 反证升级为结构性限制）、
P6 部分缓解（rev 落带/fwd·turn 近带）。**复核数值见 data/m6_g1_result.json**。

## L11 — 复核协议覆盖说明（探针 vs 全协议）

- P2/P6/P4 的**全协议复核**（P2 T=10s×N=5、P6 T=30s×N=10、P4 T=15s×N=20+对照）
  已实现在 `tools/validate_p6_modulation.py`，但本节点实测**进程内逐 net.run 开销
  ~0.3-0.5s（多会话后 brian2 对象注册表增长）** → 全协议 P4 预估 2-4h，超出本节点
  单次会话合理预算 → **全协议复核挂起**（脚本可复跑，M6_REUSE 语义），本节点用
  探针（P2 T=2-3s、P6 T=8-10s、P4 T=5s×N=5）定案 G1 判定——确定性网络（p=1/n=1，
  逐 seed 同值）使探针值即点估计，全协议复核为带内达标判定（不影响 G1 判定方向）。
- 后续节点复跑：`.venv-neuro/bin/python -m neural_exploration.tools.validate_p6_modulation`
  （建议分阶段后台跑 + gc.collect 纪律 L9 #6）。

## L9 — 实测坑（新，M6-B1a）

1. **命令池自发注入重新点燃夹带**（本轮最大坑）：初始设计自发输入注入命令池
   （AVB/PVC/AVA/AVD/DD/VD）→ 每次脉冲都点燃全互兴奋共振环（10-25Hz 全局同步），
   P2/P6 全无改善反而更差（静默 10.3%、中位 20-25Hz）。**改输出级运动池（DA/VA/SMDD）
   后自发行为才出现方向分离结构**——登记为 M6 结构性发现：夹带的点火点在命令层，
   行为驱动应作用于运动输出层（与真实蠕虫"自发/调质输入 → 运动程序"语义一致）。
2. **互抑滞后破坏逃避方向**：初始互抑按 25ms epoch 率更新 → 触刺激后首个 25ms 内
   fwd/back 共同发放无法被抑制（D_peak≈0）。**mod_dt_ms=5.0 细粒度子步修复**
   （互抑/酪胺反应延迟 ≤5ms）→ 方向相位修复的关键参数。
3. **对称互抑反而有害**：AVA↔AVB 对称互抑（0.15-0.35nA）在夹带网络下抑制 AVA/AVD
   （fwd 池率更高 → AVA 收到强抑制）→ 逃避方向丢失。**非对称**（后退→前进 0.80nA 强 /
   前进→后退 0.05nA 弱，RIM/酪胺语义）修复。
4. **tyr 基线=floor 时逃避可测但静息夹带**：floor 门控（tonic 5.6µA/cm²）使逃避方向
   干净（touch→AVA 主导），但静息时运动池脉冲仍点燃网络 → P2 静默不达标。P2 与
   行为在冻结权重 + 四项机制下仍不可兼得（M5 L40#3 的调质版复现——升级为结构性限制）。
5. **CSV 数值型参数加载为 float**：`spont_seed` 经位置解析变 float → np.random.default_rng
   报 TypeError（SeedSequence expects int）。处置：使用处 int() 强制（neuromod.py）。
6. **长任务并发纪律（M5 L21 复发风险）**：多个 302 进程同时跑会争用 brian2 cython
   缓存锁（实测单任务 ~2s/T1s，并发时单任务数分钟~数十分钟）——验证前并发清空，
   单任务串行（本节点已遵守；后续节点沿用）。
7. **AVA→DD/VD 链的网络级效应不可观测（O2 夹带态）**：DD/VD 池在静息协议下以全局
   夹带率发放（~23Hz），链的额外驱动（0.8nA×归一化后退率）被夹带网络淹没 → DD/VD
   发放/自发 rev 在删链前后无可测差异（实测 on=off）。链的方向贡献隐含于多机制联合
   （L7b）。消融 sanity 降级为组件级（链驱动电流单调 + 组装层写入门控列，测试断言）+ 
   网络级不可测性如实记录（不伪造）。
8. **进程内逐 net.run 开销随会话数增长**（~0.09s 隔离 → ~0.3-0.5s 多会话后）：302
   会话的 725MB stim 数组 + brian2 对象注册表累积 → 全协议复核（P6 T=30s×10、P4
   T=15s×40）单次会话预估 2-4h。处置：会话间 `del sess + gc.collect()`（已加入
   validate_p6_modulation.py）；本节点用探针定案 G1（确定性网络点估计），全协议
   复核挂起（L11）。

## L10 — 交付与复现

- `src/neuromod.py`：ModParams/load_m6_mod_params/ModulatorPool/ModulatedGroupedSession/
  ModulatedCircuit/make_modulated_circuit/apply_modulation；四项机制各 enabled 开关。
- `data/m6_learning_params.csv`：mod 行（O2 定稿）+ mod_gating 行（作用域预注册）。
- `tools/validate_p6_modulation.py`：P2/P6/P4/P5 复核 + 四项消融 sanity + G1 判定 →
  `data/m6_g1_result.json`。
- `tests/neuro/test_m6_neuromod.py`：ODE 稳定/门控单调/参数定稿/确定性/四项消融。
- 复现：`.venv-neuro/bin/python -m neural_exploration.tools.validate_p6_modulation`
  + `pytest neural_exploration/tests/neuro/test_m6_neuromod.py`（无网络依赖）。
- 预算：本节点实测（302/point/0.1ms，缓存热）约 1-2 CPU-h（P4 全协议占大头）。

*本文件为 M6-B1a 交付物（G1 机制门复核）；G1=部分通过（方向相位修复）+ 三态裁决请求
（L7）提交规划节点复核（WORKFLOW 流程，不静默推进）。*

---

## M6-B1c2 学习协议实现（src/learning.py + 冒烟测试）——实测结论 L12+

> 执行节点：M6-B1c2（承接前序 B1c 停滞的 learning.py/测试，落盘 + 冒烟绿）。
> 冻结文件零修改；m6_learning_params.csv 的 learning 段为唯一定稿源（mod/stdp 段不动）。

## L12 — 习惯化协议的可测性实测（B1c2 复核 B1c 记录，数值确认）

1. **302 O2 全网上 R(n) 非触诱发（确认 B1c L23）**：单会话重复触刺激 6 次
   R(n)=[0.355, 0.069, 0.397, 0.182, 0.061, 0.707]，无触对照 no_touch D_peak=0.675
   ——touch 与 no-touch 同量级（自发 bout + AVA→DD 链主导，G1 部分通过结构性限制）。
   → 302 底物冒烟只断言 R(n) 有限/确定性/首刺激 back sanity + touch≈no-touch 限制
   如实入档（不静默）；习惯化的**机制演示与消融在 M3 反射子图底物**（干净触诱发链）。
2. **反射子图 STP 机制（H1 主机制）**：isi=0（最短协议）STP 开
   R(n)=[0.353, 0.375, −0.168, −0.183, −0.183, −0.183]（指数拟合 τ_hab≈2、R²≈0.79，
   二值坍缩——命令阈值语义，B1c L23 记录）；STP 关 R(n)=[0.35, 0.31, 0.277, 0.248,
   0.41, 0.323]（无系统衰减）→ 消融对照成立（H1 机制必需）。
   ⚠ isi=100ms 时 STP 恢复 → R(n)≈0.41 常数（无习惯化）——**Rankin 10s-ISI 主协议
   受模型时程限制不可复现**（τ_rec=1000ms 在 ISI 内完全恢复；§0 #4 预注册相对判据），
   主协议 ISI=500ms（302）与 isi=0/50ms（反射，机制演示）为最短协议，10s 档 informational。
3. **联想学习底物实测**：20 规模趋化子图 CI@5s N=10 均值 0.055、std 0.68（起点抖动
   方差主导）——**CI 显著性需配对种子设计**（pre/post 同种子起点，去抖动噪声）；
   G0 的 CI=0.403@5s 为单点参考，冒烟用配对方向性判据（mean_ci_post > mean_ci_pre）。

## L13 — 联想学习实现修复（B1c 停滞根因，三因子不更新）

- **根因 1（已修）**：`AssociativeLearningLoop.env = circuit.params.env` 是 `EnvSpec`
  （无 sample/ci 方法）→ 联想学习全流程 AttributeError（前序 B1c 未跑通即停滞）。
  修复：构建 `ChemotaxisEnv`（WormLoop 同款，M4 冻结语义）。
- **根因 2（已修，关键）**：相位 store/restore 后**网络时钟 ≠ 0**——基线试次把时钟推到
  t=1000ms，`net.store()` 快照 t=1000；训练试次 restore 后运行 [1000, 4000]，而 US
  协议注入 M(t) 按**试次相对**时间写索引 [0, 3000] → M=0 全程 → dw/dt=η·M·elig≡0 →
  **无 LTP（w 恒 1.0）**。修复：US 写**绝对网络时间**索引（t_net0 + t_e）。修复后
  实测：训练 1s M=1 → w 1.0→1.14（elig≈29），周期 US（400ms 周期/200ms 窗）→ w 1.0→1.057。
- **根因 3（已修）**：`load_learning_params` 的 `fit_tau_band` 行 "3.0..15.0" 解析为
  单元素元组 → `fit_exponential` 的 `tau_band[1]` IndexError（习惯化拟合必崩）。
  修复：数值元组解析（".." 带格式 + int/float 强转）。

## L14 — 交付与复现

- `src/learning.py`：LearningParams/load_learning_params/fit_exponential/
  HabSessionReflex（M3 反射子图 + M2 STP）/HabSessionNetwork（302 O2）/
  HabituationLoop（reflex/network 双底物）/AssociativeLearningLoop（20 子图三因子，
  配对种子 CI）；B1c2 修复 L13 三根因。
- `data/m6_learning_params.csv`：learning 段（B1c 追加整理，唯一定稿源；mod/stdp 段不动）。
- `tests/neuro/test_m6_learning_smoke.py`：≥6 断言（R(n) 可测/衰减/STP 消融/302 限制/
  联想获得/η=0 无获得/消退可逆/确定性/出图）。
- 复现：`.venv-neuro/bin/python -m pytest neural_exploration/tests/neuro/test_m6_learning_smoke.py -q`。
- 预算：冒烟单跑 ~4-6 min（reflex 秒级；302 底物 ~15s；联想学习 20 档 ~3.7s 墙钟/s 模拟）。

---

## M6-B1c 学习协议运行器补充实测（L15+，追加于 B1c2 L14 之后）

> 执行节点：M6-B1c（与 B1c2 并行实现 learning.py/冒烟；本段只记录 B1c 独立实测
> 且 B1c2 L12–L14 未覆盖的结论，避免重复）。

## L15 — ⚠ Brian2 `Synapses.gmax[bool_mask] = 0.0` 静默 no-op（实现级关键坑）

- **现象**：对 grouped `Synapses` 的 `gmax`（Quantity VariableView）做布尔掩码
  赋值 `syn.gmax[mask] = 0.0`（或带单位 `0.0*siemens/meter**2`）**静默不生效**
  （逐位不变，无报错无警告）——必须整体重建数组再赋值：
  `_g = np.array(np.asarray(syn.gmax, float)); _g[mask] = 0.0;
   syn.gmax = _g * siemens / meter**2` 才生效。
- **影响**：学习层"子图替换"装配（三因子/STP 取代原连接）此前全部失效——原边
  gmax 未置 0，新建突触成为**叠加**而非**替换** → 三因子权重变化被原边淹没
  （联想学习 CI 读出失灵的根因之一；B1c2 L13 未覆盖此条）。修复已落
  `learning.py`（`_build` 三因子装配；`docs/m6_env_notes.md` L15 记录）。
- **教训**：Brian2 Quantity 数组赋值语义与 numpy 不同——掩码赋值前先
  `np.asarray` 验证；新增装配代码必须断言 gmax 实际为 0。

## L16 — 联想学习 CI_salt 读出灵敏度限制（结构性，G1 P4 未缓解的延续）

- **实测**：20-role 趋化子图上，ASE→AIY/AIB 三因子权重 w∈[1,2]（gmax 0.2–5nS
  全档扫描）对 AIY/AIB 发放与 CI_salt **不可见**（AIY/AIB 发放率由命令中间簇
  自持振荡主导：s=0 时 RIAL/RIAR/AVAL/VB1/SMDDL 仍 ~55 尖峰；感觉通路权重被
  淹没——与 G1 P4 未缓解（夹带网络无净趋化位移）同根）。
- **处置**：冒烟断言**机制级**获得/消融/消退（Δw_train>0.1、η=0 → Δw=0、
  消退 Δw<0）+ CI_salt 方向性如实入档（实测 ΔCI 为正但幅度小 ~+0.004——
  夹带限制记录，不静默）；全协议 P4 显著性判据（p<0.05 且 d≥0.5）在 20-role
  子图**不可达**（测量限制，三态裁决请求项：① 接受反证记录（网络级学习行为
  反证，§0 预注册 #1c 语义）② 减弱命令簇自持振荡（冻结权重调整，超本节点
  权限）③ 学习读出改为通路级（ASE→AIY/AIB 突触功能）而非行为级）。
- **三因子机制本身已验证**：elig 迹 + M(t) 门控 + 确定性重跑逐位一致（w 更新
  1.0→1.43、η=0 无更新、US 反号回落 1.43→0.79）。

## L17 — 并发写者（CSV learning 段 race + τ_fac=0 破坏 STP）

- 并行执行节点（B1c2）与 B1c 同写 `m6_learning_params.csv` learning 段 → 段
  内容在测试期间被来回重写（键名/值交替）；**并发执行纪律（L9 #6）同样适用
  于数据文件**。实测一次重写把 `stp_tau_fac_ms` 置 0（"0=纯抑制"注释）——
  M2 `ChemicalSynapse.stp_enabled` 判据为 τ_fac>0 且 τ_rec>0 → τ_fac=0 **静默
  禁用 STP**（习惯化消融失效假象）。处置：`HabSessionReflex` 防御性回退
  （τ_fac≤0 → 10ms，注释记录）；CSV 最终值已核定为 τ_fac=10ms。

---

## M6-B2 验证与报告节点实测结论（L18+，供主 agent 引用）

> 执行节点：M6-B2（验证+报告）。P1–P4 逐项验证脚本（validate_p2_modulation /
> validate_p3_habituation / validate_p4_associative / run_m6_validation /
> gen_m6_report）+ 全量 pytest + 本文件 L18–L27 记录。冻结文件零修改；
> B1a/B1b/B1c 落盘文件只读（m6_learning_params.csv 复核后从快照恢复字节，MD5 一致）；
> 未 git commit。

## L18 — P1 复核（B2 重跑确认，B1b 已过无新坑）

- `tools/validate_p1_stdp.py` 重跑：ΔW vs 理论逐点 max|diff|=4.05e-10（预注册 tol 0.2；
  实测 ~1e-17~4e-10 级逐位吻合）；幅值比 A₋/A₊ 实测 0.9000；权重有界 [0,2.0]
  （饱和 LTP→2.0 / LTD→0.0）；确定性重跑逐位一致；STP 回归 max|ΔEPSP|=4.89e-06 mV
  （≤1e-3 不回归）；三因子冒烟 M=1→Δw=1e-4 / M=0→Δw=0；网络级接口默认不启用 ✓。
- ⚠ 复核注意：该脚本重跑会重写 m6_learning_params.csv 的 stdp 段（内容逐位不变，
  仅段位置前移）→ 复核后本节点从快照恢复 B1 字节（MD5 016aed67… 一致），
  验证脚本对 B1 落盘文件保持只读。

## L19 — P2 复核探针（302 确定性复现 G1 数值）

- G1 json 的 `d_peak=0.355` 为 **3 位舍入**（B1a 存储）；实测精确值 **0.35526**
  （B1a 运行顺序 tau=0→23 复验同值，确定性网络单 seed 点估计即真值）→
  本节点容差 1e-3 判定可复现 ✓；判定方向/带不变（back、D_peak>0.3）。
- P2 静默探针（T=2s, seed=0, settle 500ms）：silent=10.3%（带 [60,80] 未达）、
  median=22.7Hz、max=32.7Hz（<60 ✓）——G1 数值复现（10.3/22.5/33.2）。

## L20 — P3 反射子图短 ISI（0ms）机制演示

- R(n) = [0.353, 0.375, −0.168, −0.183, −0.183, −0.183]（n=6；B1c 冒烟同款）；
  指数拟合（确定性 lstsq）τ_hab=2.0 次、R²=0.787（**≥0.5 ✓**）、A=1.18>0；
  衰减判据（后半均值 <0.5×前半）✓；首刺激方向 sanity（D_peak>0.3）✓；
  确定性重跑逐位一致 ✓。
- ⚠ **τ_hab≈2 出预注册带 [3,15]**（Rankin 10s-ISI 语义带在短 ISI 不可比——衰减更快；
  形态如实记录，不事后调带）。

## L21 — P3 消融（H1 机制必需）+ 自发恢复

- STP 关（消融）：R(n) = [0.35, 0.31, 0.277, 0.248, 0.41, 0.323] → 无系统衰减
  （decay=0.028 < 0.05，后半 ≥0.7×前半 ✓）——与 STP 开对照成立（H1 机制必需）。
- 恢复（6×短 ISI + rest 2s）：R(1)=0.353 → R(N)=−0.183 → **R_rest=0.411**
  （=1.17×R(1)，≥0.3×R(1) 预注册相对判据 ✓；R_rest > R(N) 恢复 ✓）；
  绝对恢复时程（真实分钟~小时）记录为测量限制（§0 #4 预注册，不伪造）。

## L22 — P3 ISI 扩展（判据可达性证据）

- **3s-ISI（≫τ_rec=1s）**：R(n)=[0.353, 0.312, 0.28, 0.253, 0.413, 0.322]
  （首末差 0.031，拟合 R²=0.03 无指数衰减）→ ISI≫τ_rec 无习惯化的量化对照。
- **10s-ISI（n=2，30s 会话窗内可注触上限）**：R=[0.353, 0.312]（|Δ|=0.041 < 0.05，
  反射子图无 STP 时本征抖动 ±0.08 带内 → 常数判定成立）——详见 L25。

## L23 — 302 O2 网络 D_peak 非触诱发（验证级确认；对应 B1c docstring "L23" 引用）

- 302 底物 R(n)=[0.355, 0.069, 0.397]，no-touch D_peak=0.182 →
  |touch−no_touch|=0.173 < 0.2 → **touch≈no-touch**（自发 bout + AVA→DD 链主导，
  G1 部分通过结构性限制）→ 网络级触诱发反应不可干净测量，如实入档（不静默）。

## L24 — P4 联想学习全协议数值（机制级全过）

- 全协议（20-role 子图，η=1e-2，n_test=4 配对种子，t_train=8s，t_ext=8s）：
  **Δw_train=+0.4325**（>0.1 ✓，w_pre 1.0 → w_tr 1.43）；CI_pre→CI_post
  **ΔCI=+0.0042**（方向 ✓，幅度小——见 L26）；消退 **Δw_ext=−0.1075**（<−0.01 ✓，
  w 1.43 → 1.32）且 CI_ext < CI_post ✓；η=0：**Δw=0.000**（<1e-9 ✓）、|ΔCI|=0.004
  （<0.05 ✓，无获得）；确定性重跑逐位一致 ✓（CI_pre/CI_post/w_tr/w_ext 全等）。
- 三因子机制本身（B1c L16 末条）与全协议网络级数值一致：elig 迹 + M(t) 门控 +
  确定性重跑。

## L25 — ⏱ 10s-ISI 主协议判据不可达（正式编号，供主 agent 引用）

- **τ_rec=1000ms ≪ ISI=10s** → STP x 在 ISI 内完全恢复 → R(n) 常数（无习惯化）
  ——Rankin 1990 主协议判据 (a) R²≥0.5 / (b) τ_hab∈[3,15] **不可达**；
- 且 **30s 会话窗（PROTOCOL_WINDOW_MS）内 10s-ISI 仅 2 刺激可注触**（≥3 刺激超出
  窗口，协议分段 §4.1 预注册受会话窗限制；TimedArray 30s 后回绕 + 触注入钳位）；
- **机制在短 ISI 演示**（isi=0：R²=0.787 ≥0.5 ✓、τ_hab≈2 出带 [3,15]——衰减更快
  的短 ISI 形态，如实记录）；→ **判据可达性如实记录 + 三态裁决**（主 agent 采纳
  ① 机制级 pass-with-measurement-limitations；② 延长协议窗超本节点权限；
  ③ 参考模型对照未启用——本节点按 ① 交付，记录不静默不伪造）。

## L26 — P4 CI_salt 读出测量限制（L16 验证级确认，正式编号）

- 配对 t：**p=0.391、Cohen d=0.50**——预注册显著性（p<0.05 且 d≥0.5）在网络级
  CI 读出**不可达**（命令中间簇自持振荡主导，ASE→AIY/AIB 权重 w∈[1,2] 对行为
  读出不可见，G1 P4 未缓解同根）→ ΔCI≈+0.004 幅度小如实记录（不伪造显著性）；
  机制级获得/消融/消退可验证（L24）→ **pass-with-measurement-limitations**。

## L27 — 运行纪律与交付（B2 实测坑/确认）

1. **brian2 cython 缓存锁竞争（L9 #6 复发风险确认）**：本节点一次启动后序验证时
   前序进程尚未完全退出（重叠 ~分钟级）→ 后序进程冷编译期显著变长（20-role 联想
   子图构建 ~40+ min 墙钟，其中含与 302 进程的缓存锁争用）。处置：验证前
   `job_list` + `pgrep -fl python` 确认无并发，单任务串行；缓存锁文件（.lock）
   为 brian2 正常标记（0 字节，缓存命中不阻塞）。
2. **全量 pytest（68 tests）**：M0–M5 零回归 + M6 冒烟 17/17（test_m6_neuromod 9/9
   + test_m6_learning_smoke 8/8）→ 判定写 reports/neuro/m6_pytest_status.json；
   本机实测单跑 M6 冒烟 ~20.5 min（联想学习全协议 t_train=8s/t_ext=12s 冒烟显式
   覆盖为耗时主体；全协议验证 t_ext=8s 为 CSV 定稿）。
3. **交付**：reports/neuro/m6_validation_summary.json（P1 pass / P2 反证记录型 /
   P3·P4 pass-with-measurement-limitations，milestone_complete=True）+
   docs/m6_report.md（8 节，gen_m6_report.py 生成）+ 本文件 L18–L27；
   验证脚本只读 B1 落盘文件；未 git commit。
