# M9 环境与前置处置记录（清单 §0.10 L1–L7 预注册 + 执行节点实测 L8+）

> 对应《生物仿真M9实施清单》§0.10（L1–L7 前置确认）/§1 数据门（G2，P1 验证对象）/
> §2 引擎升级（G0，P2 验证对象）/§0.5 前置门。
> 执行节点：M9-B1（本节点）：G2 数据门（FlyWire 连接组 + 递质完整性 + hemibrain 核对 +
> 扰动锚库）+ G0 引擎门（torch/MPS 安装 + 引擎选型 + CPU-GPU 对齐探针 + 墙钟预算）。
> 冻结文件零修改（M0–M8 全部 src/tests/tools/data 不动）；本节点只新建
> （`src/adult_engine.py`、`tools/build_m9_connectome.py`、`tools/fetch_m9_flywire.py`、
> `tools/scan_m9_engine.py`、`data/m9_*.csv`、`docs/m9_env_notes.md` 本文件）。
> 未 git commit。运行纪律：`PYTHONHASHSEED=0 MPLBACKEND=Agg <venv>/bin/python` 固化
> （M8 R8）；引擎环境 = `.venv-m9`（L5 预注册，不触碰 .venv-neuro/.venv-db 基线）。

---

## L1 — 交接：M8 冻结基线 → M9 组装方式（组合不修改）

- **组合不修改纪律（§0.3 #8）**：M0–M8 全部冻结件零修改；本节点只新建。清单 §0.10 L1
  列出的 M9 新建文件清单为准（`adult_engine.py`/`adult_connectome.py`/`adult_circuit.py`/
  `adult_body.py`/`adult_env.py`/`adult_loop.py`/`adult_plasticity.py`/`adult_activity.py`/
  `adult_perturb.py` + `tools/build_m9_*`/`scan_m9_engine.py`/`validate_p9_*`/
  `run_m9_validation.py`/`gen_m9_report.py` + `tests/neuro/test_adult_smoke.py`）。
- **直接复用（只读）**：`larva_circuit.py`（300 two_comp CPU 基线 + G0 对齐参考）、
  `m8_larva_params.csv`/`m8_behavior_reference.csv`（对齐判据带）、
  `build_m8_connectome.py`（解析/校验/递质惯例）、`scan_m8_scaling.py`（G0/G1 决策模式）、
  M6 `neuromod.py`/`plasticity.py`（调质/可塑性语义）、`virtual_body.py`（状态分类脚手架）。

## L2 — 数据源与许可（G2 门前置；实测可达性 = L8，以下为预注册）

| 数据 | 权威源 | 版本 | 许可 | 用途 |
|---|---|---|---|---|
| FlyWire 连接组连通性 | Zenodo 10676866（= codex.flywire.ai v783 静态导出） | 783.0（2024-06-02） | CC-BY-4.0 | G2 P1 主数据 |
| FlyWire 神经递质预测 | 同上（flywire_synapses 含逐突触 gaba/ach/glut/oct/ser/da 概率；proofread_connections 含逐连接平均） | 783.0 | CC-BY-4.0 | §1.2 递质完整性 |
| 细胞类型标注 | Schlegel et al. 2024（Nature），Zenodo 10877326 | 2024 | 见记录 | 神经元 cell_type 列 |
| hemibrain | Scheffer et al. 2020（eLife），neuprint（需凭据）/ Eckstein 2024 补充 | v1.2.1 | 见记录 | §1.3 交叉核对 |
| 扰动锚库 | Aso 2014（eLife）/Robie 2017（Nat Methods）/Claudi 2024（bioRxiv/Nature） | — | 文献 | §1.4 扰动锚 |

## L3 — M8 反证记录最终状态（交接语义）

M8 §0.2 表 R1–R12 为 **M9 设计依据**，不进入 M9 交付判据的"已完成"假设：
- R1（自发 run 稀缺）→ §1 D4 权重策略；R2（缺 GABA 标注）→ §1.2 递质完整性检查（本节点）；
- R5（扰动有锚不足 3/50）→ §1.4 锚获取协议（本节点）；R6（DA 受体 none）→ §1.2.2 受体映射检查；
- R8（hash 非确定性）→ 本节点所有新建管线用 zlib.crc32 类确定性哈希 + PYTHONHASHSEED=0；
- R10/R11/R12（预算/缓存纪律）→ §0.9 + 本节点 G0 探针先行。

## L4 — M8 冻结组件作 GPU 对齐参考模型

- G0 对齐（P2）参考 = M8 冻结 300 two_comp 定稿配置（CPU Brian2 基线；`m8_larva_params.csv`/
  `m8_larva_connectome.csv`/`m8_scaling.csv`；`larva_circuit.py` 冻结文件只读调用）。
- 实测锚：300 two_comp 30s=118s/试次（CPU，M8 报告）；3016 point 30s=843s/试次。

## L5 — 环境与依赖（引擎环境预注册，实测 = L9）

- **预注册**：GPU 引擎依赖入独立环境 `.venv-m9`（不触碰 .venv-neuro/.venv-db 基线）；
  判据脚手架 import 不引入重型依赖（纯 stdlib + numpy 验证，M8 L5 惯例）。
- **M8 纪律**：运行前缀 `PYTHONHASHSEED=0 MPLBACKEND=Agg`；Brian2 缓存独立目录/
  严格串行（R11）；pytest `-p no:cacheprovider` + `PYTHONDONTWRITEBYTECODE=1`（M7 L24）。

## L6 — 预算与确定性纪律

- §0.9/§10.1：数据门 ≤20 CPU-h；引擎门 ≤40 CPU-h + ≤16 GPU-h；超限 → 记录 + 三态裁决。
- 判据带定稿于 CSV 不事后调（§0.7 #8）；CSV 唯一定稿
  （`m9_behavior_reference.csv`/`m9_imaging_reference.csv`/`m9_perturbation_plan.csv`/
  `m9_engine_params.csv`）。

## L7 — 数据隔离原则落地（§0.8）

- 拟合集 A（自发分布/静息发放率——权重校准）vs held-out B/C/D（学习/活动/扰动）；
  拆分表见本文件 L10；校准脚本只读数据集 A（纪律）。

---

## L8 — 数据源可达性实测（2026-08-31，照 m8_env_notes L16 格式）

- **可用**：Zenodo API（`zenodo.org/api/records/...`，1s 内 200，下载走
  `/api/records/{id}/files/{key}/content` 302→对象存储，实测 852MB ~2min ≈ 7MB/s）、
  codex.flywire.ai（静态页/健康检查可达，交互应用需 Google 登录）、
  neuprint.janelia.org（可达但 **API 需凭据 401**）、nature.com（2024 论文全文可达，
  含 Data availability）、storage.googleapis.com/flywire-data（bucket 可列，无连接组数据）。
- **不可用/受限**：github.com（30s 超时 000）、api.flywire.ai（000，CAVE 程序化访问不可达）、
  cave.flywire.ai（000）、biorxiv.org（429 限流）、science.org（M8 已知 403）。
- **Zenodo 10676866（v783）文件清单（已实测）**：
  `flywire_synapses_783.feather`（9,492,998,242 B ≈ 9.49GB，~130M 行逐突触）、
  `proofread_connections_783.feather`（852,022,274 B，16,847,997 行 逐连接×neuropil）、
  `per_neuron_neuropil_count_post_783.feather`（233,843,050 B）、
  `per_neuron_neuropil_count_pre_783.feather`（16,853,770 B）、
  `proofread_root_ids_783.npy`（1,114,168 B）。
- **权威计数锚（实测，L11 详）**：神经元 139,255（proofread_root_ids）；化学突触
  **54,492,922**（proofread_connections.syn_count 全和，≈ 论文 54.5M，±10% 带内）；
  唯一有向连接 **3,732,460**（codex 标注页确认 "139,255 neurons 3,732,460 connections"）；
  **缝隙连接：官方 v783 发布不含**（Nature 论文原文 "Our connectome includes only chemical
  synapses; the identification of electrical synapses awaits a future EM dataset with higher
  resolution"）→ 清单 §1.1 "~8.5k 缝隙" 锚在官方发布中**不可得**，按"或权威解析"= 0/不可用
  记录为测量限制（M8 幼虫缝隙 0±0 同哲学），诊断 OUT 如实登记 + 请求三态裁决（L13）。
- **Schlegel 2024（Zenodo 10877326）**：`nblast_flywire_hemibrain_min_comp.feather`
  （212,095,362 B，FlyWire↔hemibrain NBLAST 匹配，§1.3 神经元映射率直接数据）；
  `sk_lod1_783_healed_ds2.parquet`（5.35GB，全骨架+细胞类型标注，备选）。
- **Eckstein 2024（Zenodo 10593546）**：`hemibrain-v1.2-tbar-neurotransmitters.feather.bz2`
  （442,710,628 B，hemibrain 突触级递质预测，§1.3 递质一致性）；`synister_fw_..._synapses`
  （7.7GB，FAFB 突触级递质预测——v783 已内嵌同源预测，无需另下）。

## L9 — 引擎环境实测（G0 前置）

- `.venv-m9`（Python 3.9.6，新建，不触碰 .venv-neuro）：**torch 2.8.0**（MPS
  `is_available()=True`、`is_built()=True`，实测 MPS matmul 正常）、pyarrow 21.0.0、
  pandas（后补）、brian2 2.6.0（CPU 基线同版本）、numpy 1.26.4。
- **NG 门裁决（预注册路线 ①）**：本机 Apple Silicon + MPS 自研内核 = 本机主路线，实测可行；
  无需云 GPU；Brian2CUDA（NVIDIA only）/NEST GPU（Linux）在本机不可行——如实记录。

## L10 — 数据隔离拆分表（§0.8 落地）

| 数据集 | 用途 | 内容 | 来源 |
|---|---|---|---|
| A（拟合） | 权重校准 ≤40 参数 | 自发分布/静息发放率（成年行为学文献带） | 文献统计（Aso 2014 等行为学） |
| B（held-out） | 涌现功能 | 学习/泛化、航向、决策 | 文献判据带 |
| C（held-out） | 活动金标准 | 中枢脑钙成像统计 | 文献统计回退（R7） |
| D（held-out） | 扰动预测 | optogenetic 激活/沉默实验库 | Aso 2014/Robie 2017/Claudi 2024 |

## L11 — G2 数据管线实测（详 §1 结算；见 `tools/build_m9_connectome.py`）

- proofread_connections 解析：16,847,997 行（pre/post root_id + neuropil + syn_count +
  gaba/ach/glut/oct/ser/da 平均概率）；逐 neuropil 聚合 → 逐连接；逐 pre 神经元 dominant NT。
- 神经元表：proofread_root_ids（139,255）+ neuropil 计数文件 → 主脑区（dominant neuropil）。
- 递质覆盖率：proofread_connections 覆盖全部 3,732,460 连接（NT 列齐全）；逐神经元覆盖 =
  有出边神经元占比（待 build 输出）；flywire_synapses（~130M 行）提供逐突触 NT 覆盖
  （已下载，待解析）。

## L12 — 本机负载/缓存纪律（R11 落地）

- 本机负载病态（M8 L23：10–30× 减速）：全部重任务串行（长下载/解析/brian2 编译不并发）；
  Brian2 缓存独立目录（G0 探针时设 `prefs.codegen.runtime.cython.cache_dir`）。

## L13 — 需规划节点三态裁决的问题（汇总）

1. **缝隙连接锚（"~8.5k"）与官方发布不符**：Dorkenwald 2024 Nature 明确连接组仅含化学突触，
   Zenodo v783 无缝隙表；hemibrain 侧缝隙亦需 neuprint 凭据（401）。
   → 裁决：① 缝隙=0/不可用记测量限制（推荐，M8 幼虫先例）② 等 CAVE/网络恢复 ③ 反证笔记。
2. **hemibrain 连通性数据不可直接下载**（neuprint API 401 需注册凭据）：若凭据不可得，
   §1.3 核对以 NBLAST 映射率 + 递质一致性（Eckstein hemibrain 表）完成 informational 级；
   突触数一致性 ±10% 需 neuprint 凭据 → 记录测量限制 + 裁决。
3. **G0 引擎选型确认**（预注册推荐 MPS 自研内核）：对齐探针实测数据（L14+）后回主 agent 定稿。

---

*（执行节点继续追加 L14+：G0 探针实测、数据管线 SHA、扰动锚统计等。）*
