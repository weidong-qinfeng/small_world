# M8 报告：果蝇幼虫全脑（3,016 神经元）阶段二首里程碑——三通道验证

节点：B2 验证节点（行为 P4/P5/P6 全协议 + 活动 P8 + 扰动 P7 top-50 + P9 报告）；日期：2026-08-30；工作目录：/Users/weidong/ai/small_world

运行纪律：全部 `PYTHONHASHSEED=0 MPLBACKEND=Agg ./.venv-neuro/bin/python <script>`（跨进程可复现）；冻结组件（larva_circuit/loop/body、calibrate）零修改；行为判据带定稿于 `data/m8_behavior_reference.csv` 不事后调。

## 1. 概述

- **目标**：3,016 神经元幼虫全脑仿真的行为 + 活动 + 扰动三通道验证（设计文档 §五 M8）。

- **主 agent 预算裁决**：行为判据（P4/P5/P6）以 **300 档 two_comp 全协议**为准（G0 定稿保真度 two_comp；300 two_comp 30s=118s/试次，N≥10 可行）；3016 全规模长协议行为判据不可行（3016 point 30s=843s/试次，且缩放扫描已记录**3016 point CI=0 行为层退化反证**；3016 two_comp 组合从未构建——B1b 遗留裁决）→ 3016 只做结构性验证（G1 PASS：silent=0.8477/0.8167）。

- **总判定**：`all_pass = false`——P4 自发分布不落带（反证）、P6 条件化回避结构性不可转正（反证）、P7 命中率有锚子集不足（限制记录）；P5 机制级 LI PASS、P8 活动正向模型 PASS、P1/P2/P3 结构 PASS。按 §0.4 反证路径记录缺失机制清单 + 三态裁决请求，不静默推进。

## 2. 数据

- `data/m8_larva_connectome.csv`（B1a 定稿；官方发布解析 roster 2,956，神经元间化学突触 352,611，四区室比例与论文 Fig.2C 逐位一致，递质覆盖 100%，SHA-256 逐位确定）；

- `data/m8_larva_params.csv`（G0/G1/D5 权重定稿：gmax_scale=0.05 + class_scale_sensory_inter=6.0 / inter_inter=3.0 / inter_motor=3.0，stdp_eta=12.0）；

- `data/m8_behavior_reference.csv`（行为判据带唯一定稿源：run [60,85]% / turn [10,30]% / pause [3,20]% / LI [0.1,0.8] / 逃避 D_peak>0.3、resp≥0.8）；

- `data/m8_scaling.csv`（铁律 C 三组缩放扫描）；`data/m8_calibration.csv`（D5 校准反证：CI=-0.165 落盘）。

## 3. 方法

- **P4**：无刺激无梯度 300 two_comp T=30s×N=10（种子 0–9），`classify_larva_state` 分类 → 时间比例 vs 判据带（`tools/validate_p8_spontaneous.py`）；

- **P5**：CS=触角嗅觉 ORN 对（sens→PN 出边 top 2，预注册确定性规则）注入，KC→MBON 成对 STDP 机制级 LI；配对训练 N_train=3 × 双选测试（trained vs control CS，LI_pref）；未配对对照；η=0/H1 消融；确定性（`validate_p8_olfactory.py`）；

- **P6**：US=光遗传激活 IV 类伤害感受器（MD）；痛觉逃避基线 sanity（D_peak/resp）+ 条件化回避探针（`validate_p8_nociceptive.py`）；

- **P7**：`data/m8_perturbation_plan.csv`（top-50 预注册）+ 逐神经元沉默（出边 gmax→0）/激活（tonic 0.5nA）→ 后果类（预注册类集+阈值）→ 有锚命中率（`validate_p8_perturbation.py`）；

- **P8**：发放→GCaMP 荧光正向模型（τ=1.0s、2Hz 窗口平均）→ 发放率带判定 + run↔turn 转换 ±2s 窗活动态序列（`validate_p8_activity.py`）；

- **确定性**：p=1/n=1；同参数重跑逐位一致；跨进程统一 PYTHONHASHSEED=0。

## 4. 结果（P1–P10 pass_ 判定）

| 判据 | 判定 | 关键数值 |
|---|---|---|
| P1 连接组 | ✅ | roster 2,956（官方解析）/论文 3,016；化学突触 352,611（神经元间硬断言）；唯一有向对 110,677 落带；缝隙 0；递质覆盖 100% |
| P2 缩放+G0/G1 | ✅ | G0 PASS（two_comp 定稿）；G1 PASS 3016（静默 0.8152、bout 0.895）；3016 point CI=0 行为层退化反证记录 |
| P3 身体 | ✅ | 五运动模式冒烟全过 + 状态阈值 CSV 定稿 |
| P4 自发分布 | ❌ | run=9.92% turn=81.42% pause=8.67% （N=10 T=30000.0ms；带 run[60,85]/turn[10,30]/pause[3,20]）→ **不落带反证** |
| P5 气味联想 | ✅ | LI_paired=0.8953 vs unpaired=0.1032 （p=0.0000 d=inf）；全协议 LI 学习曲线 [0.8952702682319773, 0.8952702682319773, 0.8952702682319773]、LI_pref=0.123；η=0/H1 消融→0；确定性 ✓；**全协议如实限制**：未配对背景 STDP 漂移随训练窗累积 （LI_unpaired_full=0.229 ≥ 0.05 → b 绝对读法不成立，相对读法成立），消退 (d) 结构性不可达（paired-STDP 无权重衰减项）→ 见 §6 限制 |
| P6 避痛 | 逃避基线 ✅ / 条件化 ❌ | resp_prob=1.000 D_peak≈0.783（sanity PASS）；条件化回避结构性不可转正（反证） |
| P7 扰动 | 机制 ✅ / 命中率 ❌ | top-50 全测机制可运行（sham+确定性 ✓）；有锚命中率 0.333（1/3；有锚 3/50 < 20 下限 → 限制记录） |
| P8 活动 | ✅ | median=0.500Hz silent=0.500；转换窗 n=179 无 NaN；成像不可得 → 文献带回退（限制记录） |
| P9 回归+报告 | ✅（本报告） | pytest 全量见 §8 |
| P10 交接 | ✅（§9） | 反证/测量限制逐条入档 + M9 交接 |

## 5. 反证记录（缺失机制清单 + 三态裁决请求）

1. **P4 自发分布**：D5 权重 s2i6 放大（class_scale_sensory_inter=6.0）→ 感觉驱动过强 → 转向（turn）主导、run 稀缺
   - 排查/处置：降低 s2i6 或改 s2i6 分段缩放后复测自发分布；校准反证记录于 data/m8_calibration.csv（d5_g050）

2. **P4 自发分布**：缺 GABA 递质标注（B1a：inter 递质经 nt_fallback=class hash 分配，非真实 GABA）→ 抑制平衡缺失 → 网络难维持 稳定前进模态
   - 排查/处置：补充 GABA 标注/递质行后重测（主 agent 裁决）

3. **P4 自发分布**：provisional 肌肉映射仅 fwd/back/left/right（无真实 curl 通道）→ curl=0 固定，防御态缺失
   - 排查/处置：curl 通道留真实肌肉映射（P3 定稿后）

4. **P4 自发分布**：运动层与命令层分离驱动（杠杆②）在当前 D5 权重下仍 不足以稳定 run 模态
   - 排查/处置：杠杆组合消融 sanity（§1 D6 三杠杆）

5. **P4 自发分布**：P4 行为判据 FAIL → 三态裁决请求
   - 排查/处置：P4 行为判据 = FAIL（反证记录）。请求主 agent 三态裁决：①接受反证路径（缺失机制清单如上，M9 必需机制清单累积）；②或裁决调整 D5 权重（降低 s2i6 / 补 GABA 标注）后 B3 复测；③或裁定 P4 在 3016 全规模结构验证（G1 已 PASS）下按 M4 P4 先例记录反证型 pass。本节点不静默判定 PASS。

6. **P6 条件化回避**：MD→DALD→back 通道无联想可塑性（无 STDP/三因子门控于伤害性通路）→ CS-US 配对无法产生回避获得
   - 排查/处置：US=痛觉在伤害性通路开启三因子门控可塑性（H2 语义，留 M9）

7. **P6 条件化回避**：行为级回避读出（CI）在 300 档 two_comp 下不可转正：缺 GABA 标注（inter 递质 hash 分配）→ 抑制平衡缺失
   - 排查/处置：补 GABA 标注/递质行后重测（主 agent 裁决）

8. **P6 条件化回避**：curl 通道结构性缺失（provisional 肌肉映射仅 fwd/back/left/right）→ 蜷缩防御读出不完整
   - 排查/处置：真实肌肉映射（P3 定稿后）

9. **P6 条件化回避**：P6 行为级条件化 FAIL → 三态裁决请求
   - 排查/处置：P6 行为级条件化 = FAIL（反证记录）；逃避基线 sanity = PASS。请求主 agent 三态裁决：①接受反证路径（缺失机制清单入 M9 必需机制清单）；②或裁决补 GABA 标注/伤害性通路可塑性后 B3 复测。本节点不静默判定 PASS。

10. **P7 扰动预测**：有锚子集 3/50 < 20 预注册下限（无逐神经元驱动线锚下载）
   - 排查/处置：B1d/B2 网络受限（无逐神经元驱动线锚下载）→ 有锚 = MD 3/50 < 20 下限 → 按 §0.7 #4 缩小有锚判据分母 + 测量限制记录；命中率仅 informational，≥70% 判据在有锚子集达标前不静默判定

11. **P8 活动金标准**：成像统计参考数据不可得（Lemon 2015 数值不可下载）
   - 排查/处置：文献带回退 + 测量限制记录；只承诺统计级

12. **P5/P6 US 通路**：DA 递质输出受体 none（B1a 不臆造受体作用域）→ 奖赏/回避 US 通路无功能
   - 排查/处置：三因子门控/US 门控机制留 M9（H2）


## 6. 限制（测量限制，不伪造）

1. **P4 行为判据 3016 全规模不可行**（预算：3016 point 30s=843s/试次；3016 two_comp 组合从未构建——B1b 遗留裁决）→ 行为判据以 300 two_comp 为准，3016 只做结构性验证；

2. **P5 US=DA 奖赏无功能**（B1a 递质标注：DA 输出受体 none，§3.3 不臆造受体作用域）→ 全协议三因子门控（H2）留 M9；消退判据 (d) 结构性不可达（paired-STDP 无权重衰减项）；**未配对背景 STDP 漂移**（全协议 N_train=3 训练窗累积 → LI_unpaired_full=0.229 ≥ LI_APPEAR_THRESHOLD 0.05 → b 绝对读法不成立；相对读法（未配对 = 协议固有背景，无 CS 驱动额外获得）成立——双读数如实记录）；

3. **P6 条件化回避行为级判据不可转正**（D5 反证：缺 GABA 标注，CI=-0.165 落盘；plasticity=none 无联想机制）；

4. **P7 有锚子集 3/50 < 20 预注册下限**（无逐神经元驱动线锚下载）→ 命中率仅 informational，≥70% 判据在有锚子集达标前不静默判定；另有结构发现：top-50 命令样神经元**沉默全部无行为后果**（0/50 有变化——运动层自发驱动主导）；

5. **P8 成像统计参考不可得**（Lemon 2015 全 CNS GCaMP 数值不可下载）→ 文献带回退 + 测量限制记录（§0.7 #3），只承诺统计级；

6. **P8 3016 point 短窗未跑（预算限制）**：3016 会话构建在本机负载下 >40 min 未完成（B1b 冷构建 754s 亦远超预算）→ 3016 结构性验证由 G1（silent=0.8477/0.8167）与缩放扫描（3016 point CI=0 行为层退化反证）覆盖，不重复烧预算；

7. **冻结代码跨进程 hash 非确定性**（`_apply_nt_fallback` 用 Python `hash()`）→ B1d 发现 + 统一 PYTHONHASHSEED=0 缓解；建议冻结代码改确定性哈希（zlib.crc32）留主 agent 裁决；

8. **curl 通道结构性缺失**（provisional 肌肉映射仅 fwd/back/left/right）→ 蜷缩判据留真实肌肉映射（P3 定稿后）；

9. **roster 2,956 ≠ 论文 3,016**（60 缺口 = CATMAID 注解成员不在官方发布）→ P1 裁决如实记录，不近似补齐。


## 7. 结论

- **结构层**（P1/P2/P3 + G0/G1）：连接组/降阶/身体/双状态全部 PASS——3,016 全脑在 D5 权重下保持 G1 双状态（静默 0.8477/0.8167 + bout 活动）。

- **行为层**（P4/P5/P6）：P5 机制级 LI PASS（CS 驱动 KC→MBON 获得显著 > 未配对）；**P4 自发分布不落带**（run=9.9% vs 带 [60,85]%、turn=81.4% vs [10,30]%——感觉驱动过强 → 转向过多）+ **P6 条件化回避结构性不可转正** → 行为不涌现反证（缺失机制清单 §5）。

- **活动层**（P8）：活动正向模型冒烟 PASS（无 NaN + 确定性 + 转换窗序列）；成像统计对照受数据可得性限制。

- **扰动层**（P7）：top-50 全测机制可运行（sham + 确定性）；有锚命中率受实验锚缺失限制（informational）。

- **总判定**：`all_pass=false`（如实）。M8 三通道验证在**结构层成立、行为层反证**——按 §0.4 反证路径记录缺失机制清单 + 三态裁决请求（P4/P6 行为判据、P7 锚缺口），由主 agent 裁决是否接受反证路径或安排 B3 复测。

## 8. 回归（pytest）

- 全量 pytest：passed=76 failed=0 skipped=0 （M8 冒烟 ≥8 + M1–M7 零回归，详见 m8_neuro_pytest_status.json）；

- 三通道图：/Users/weidong/ai/small_world/neural_exploration/reports/neuro/m8_three_channel.png

## 9. 交接（M9 入口）

- **必需机制清单累积**（M14 交付物②组成部分）：D5 权重 s2i6 放大 → 转向过多（P4）；缺 GABA 标注 → 抑制平衡缺失（P4/P6）；MD→DALD→back 无联想可塑性（P6）；US=DA 奖赏无功能（P5 H2）；curl 通道缺失（P6/P7）；成像统计参考缺失（P8）；扰动锚缺口（P7）；

- **引擎升级依据**：铁律 C 三组缩放曲线（m8_scaling.csv）→ M9 神经元模型/GPU 迁移决策；CPU 基线供 GPU 对齐；

- **三通道验证管线**（行为/活动/扰动 + 活动正向模型 + 扰动 top-N 预注册）为 M9 模板；

- **冻结代码建议**：`_apply_nt_fallback` 改确定性哈希（zlib.crc32）——留主 agent 裁决。
