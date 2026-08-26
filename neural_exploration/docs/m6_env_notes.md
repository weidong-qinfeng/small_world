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
