# M8 B1d 阶段验证摘要（P5/P6/P7/P8；300 档 two_comp 短协议 T≤5s）

节点：B1d 执行节点；日期：2026-08-30；工作目录：/Users/weidong/ai/small_world
运行：全部 `PYTHONHASHSEED=0 MPLBACKEND=Agg ./.venv-neuro/bin/python <script>`

## 交付物清单

| 项 | 脚本 | 结果落盘 |
|---|---|---|
| P5 气味联想学习 | `tools/validate_p8_olfactory.py` | `data/m8_p5_olfactory.csv` + `reports/neuro/m8_p5_olfactory.{png,json}` |
| P6 避痛学习 | `tools/validate_p8_nociceptive.py` | `data/m8_p6_nociceptive.csv` + `reports/neuro/m8_p6_nociceptive.{png,json}` |
| P7 扰动预测 | `tools/gen_m8_perturbation_plan.py` + `tools/validate_p8_perturbation.py` | `data/m8_perturbation_plan.csv` + `data/m8_p7_perturbation.csv` + `reports/neuro/m8_perturbation_hitrate.png` + `reports/neuro/m8_p7_perturbation.json` |
| P8 活动金标准 | `src/larva_activity.py` + `tools/validate_p8_activity.py` | `data/m8_p8_activity.csv` + `reports/neuro/m8_p8_activity.{png,json}` |

新增组件：`src/larva_perturb.py`（PerturbLarvaCircuit 扩展 stim 列 / silence gmax→0 / tonic 激活 /
自发协议 / 后果类分类）、`src/larva_activity.py`（发放→GCaMP 荧光正向模型 + 活动态序列）。
冻结文件（larva_circuit/loop/body、calibrate_m8_weights）**零修改**（git status 仅新增未跟踪文件）。

## 关键数值

- **P5**：LI_paired = **+0.8953**（±0，5 seed）vs LI_unpaired = **+0.1032**（= 冻结探针背景 LI）；
  配对 t p<0.001、Cohen d=inf；KC 测试窗发放率 **666 spikes/s**（CS 通路确认）；η=0 → 0；H1 消融
  （KC→MBON 关）→ 0；确定性逐位一致。判据 a/b_rel/c/d/e 全过（**pass_all=True**）；b 绝对阈值读法
  （|LI_unpaired|<0.05）不满足——未配对组 0.1032 = 同网络协议固有背景 STDP 漂移，**双读数如实记录**
  （`criteria_b_readings`），不静默放宽。
- **P6**：痛觉逃避基线 sanity **PASS**——resp_prob=**1.00**（≥0.8）、D_peak=**0.783**（>0.3）全部 5 seed
  方向 back（MD 伤害感受器→DALD 下行→back 通道，nt_fallback 后可用出边 6）。伤害性条件化行为级
  回避指数（CI）结构性不可转正（D5 校准反证落盘）→ 限制记录，习惯化协议母版 = 本逃避协议。
- **P7**：扰动计划 `m8_perturbation_plan.csv`（确定性 top-50：命令样/下行 50、度分位枢纽、实验可及性；
  有锚 3 = MD class IV 伤害感受器，文献锚 蜷缩↑/后退↑）。冒烟 6 神经元（top-5 + rank-28 锚 MD）：
  沉默全部 无变化；激活 PRE-GORO3→转弯↑、PRE PMN→转弯↑、PRE-DORSAL-MAP→后退↑、**MDNB_RIGHT→后退↑
  = HIT（与文献锚一致）**；sham 逐位一致 ✓、确定性 ✓。有锚子集 3/50 < 20 预注册下限 → 命中率
  （1.0/1 有锚模拟）仅 informational + 测量限制记录（§0.7 #4 回退，不静默判定 ≥70%）。
- **P8**：发放率 median=0.667Hz（带 [0,1] ✓）、max=24.3Hz（带 [0,60] ✓）、静默比例 48.7%
  （带 [50,90] 下沿外 1.3pt——**D5 已知工作区贴边脆弱性**，如实记录不静默）；荧光正向模型
  （τ=1.0s、2Hz 降采样）6 帧无 NaN、确定性 ✓；run↔turn 转换 n=15，±2s 窗活动态序列无 NaN、
  pre=4.36/post=3.33、high 占用 0.51、确定性 ✓。成像数据不可得（Lemon 2015 全 CNS GCaMP 数值不可
  下载）→ 文献带回退 + 测量限制记录（§0.7 #3），只承诺统计级。

## 实测发现（需主 agent 知悉/裁决）

1. **跨进程非确定性（冻结代码）**：`larva_circuit._apply_nt_fallback` 用 Python `hash(r.pre)` 分配
   inter 递质（larva 命名无 `_<digits>` 后缀 → **3136/3136 inter 行走 hash 路径**）→ 逐进程随机 hash
   种子 → 跨进程网络不一致（实测同协议冻结探针 LI：0.1372 / 0.1032 / 0.0675）。B1d 处置：验证统一
   `PYTHONHASHSEED=0`（两独立进程冻结探针均 0.1032 → 可复现）；建议冻结代码改确定性哈希（zlib.crc32）。
   影响面：冒烟测试 `test_learning_probe_computable` 的 LI≥0.05 断言在部分 hash 种子下可能贴阈/失败。
2. **冻结 sens_roles CS 对无 KC 通路**（B1a 数据事实）：RH6PR/22C ORN 出边 1 条、10 跳不达 KC →
   冻结 `run_learning_probe` 的 LI 为背景相关获得（非气味联想）。B1d 改用触角嗅觉 ORN 对
   （AN-L-SENS-B1-ACA-01/12，sens→PN 出边 top）经扩展 stim 列注入（预注册规则），CS 驱动生效。
3. **冻结转导公式 s=1 → ≈100µA** 会过驱动活 ORN（冻结探针 CS 对是死对未暴露此问题）；B1d CS 注入
   预注册 1.0nA（与 P6 伤害感受器注入 0.75nA 同量级）。
4. **行为判据带缺口**：`m8_behavior_reference.csv` 无逃避潜伏期窗行（§0.7 #8 要求定稿于 CSV）；
   B1d 用清单内联判据（D_peak>0.3 / C_curl>C_fwd / resp≥0.8），建议 B2 补带。
5. **curl 通道结构性缺失**：provisional 肌肉映射仅 fwd/back/left/right → P6 的 C_curl>C_fwd 与 P7 的
   蜷缩↑ 结构性不可达（已记录）；蜷缩判据留真实肌肉映射（P3 定稿后）。
6. P7 有锚子集 3/50 < 20 下限 → 命中率 ≥70% 判据 B2 再测（网络恢复后可补逐神经元驱动线锚）。

## 状态

- P5 ✅（机制级 LI，pass_all=True；DA/US 通路缺失限制记录）
- P6 ✅（逃避基线 sanity PASS；条件化行为级判据反证记录）
- P7 ✅（计划+机制可运行，锚 MD 命中；命中率限制记录）
- P8 ✅（正向模型+状态序列冒烟通过；成像统计限制记录）
