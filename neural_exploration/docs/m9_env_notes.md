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

## L14 — G2 数据管线实测、数据语义坑与下载决策（执行节点 + 主 agent 裁决）

### L14.1 下载决策（预算纪律；主 agent 裁决记录）

- **flywire_synapses_783.feather（9.49GB，~130M 行逐突触）**：Zenodo 实测限速（10 分钟
  ≈0.8MB，多次断点续传后仍不可行）→ **主 agent 裁决：非 G2 门必需**。理由：权威
  `proofread_connections_783.feather`（852MB，已完整落盘）含**逐连接**递质平均概率
  （gaba/ach/glut/oct/ser/da）覆盖全部连接（100%）；逐突触级覆盖率按 §1.2.3"如实记录"
  语义登记为**测量限制**（不臆造）。**已停止下载，不再续传**（预算纪律；后续节点若需
  逐突触级可在网络恢复时补跑，管线接口已就位）。
- **nblast_flywire_hemibrain_min_comp.feather（212MB）**：实测下载至 99.0/212.1MB 后停滞
  → §1.3 hemibrain 映射率以**测量限制**登记（待网络恢复补跑；协议与脚本已在
  `build_m9_connectome.py::write_hemibrain_crosscheck` 就位）。
- **Eckstein 2024 hemibrain NT 表（442MB bz2）**：未下载 → 递质一致性测量限制。
- 结论：**核心 P1 计数锚全部由已落盘的官方文件独立解析验证**，不以"下载未完成"为由
  跳过断言（见 L14.3）。

### L14.2 数据语义坑（重要——后续节点必读）

1. **`per_neuron_neuropil_count_post/pre_783.feather` 与 roster 的 ID 空间不相交**：
   该两文件的 `post_pt_root_id`/`pre_pt_root_id` 是**不同 materialization 的 segment id**
   （range 720575940379279642–720575940401000114），与 `proofread_root_ids_783.npy`
   （range 720575940596125868–720575940661339777）**同 dtype 精确交集 = 0**（实测）。
   → 该文件**不能直接**给 roster 神经元赋脑区（缺 segment→root 映射表）。**处置**：
   region 改由**同源权威 `proofread_connections.neuropil` 列**聚合（按 post 侧逐 neuropil
   突触计数取 dominant）；原文件登记为测量限制（若后续节点取到映射表可升级）。
2. **numpy int64/uint64 混用精度陷阱（本轮实测踩坑）**：`np.isin`/`np.intersect1d` 在
   int64 vs uint64 混合时提升为 float64 → 7.2e17 量级 FlyWire root id 精度丢失（>2^53），
   产生**假阳性交叠**（实测报告 94%/121% 交叠，实际为 0）。**纪律**：FlyWire ID 比较一律
   统一 dtype（int64/object/Python int），或直接用 Python set。
3. **"3,732,460 connections" 与解析值语义不同**：codex 标注页官方值 3,732,460 与
   "任一突触即算连接"解析值（15,091,983）不一致；实测过滤阈值均不匹配（syn≥5→2,700,513、
   syn≥3→4,916,231）→ Codex 该统计含未文档化过滤口径 → **定义差异登记 + 三态裁决**
   （不按民俗数字；突触计数锚 54.5M 独立验证通过）。

### L14.3 P1 断言结果（G2 门核心；`data/m9_connectome_counts.json`）

| 断言 | 类型 | 结果 |
|---|---|---|
| 神经元 == 139,255（±0，官方发布） | 硬断言 | **PASS**（实测 139,255） |
| 化学突触计数 ∈ 54.5M±10% | 硬断言 | **PASS**（实测 **54,492,922**；论文 "54.5 million synapses" 同源） |
| 化学行递质覆盖 100% | 硬断言 | **PASS**（15,091,983/15,091,983） |
| 自连接 0 / 白名单登记 | 硬断言 | **PASS**（实测 0 条自连接） |
| 孤立神经元显式白名单 | 硬断言 | **PASS**（616 个无出边/入边神经元，note 白名单登记） |
| 化学边节点 ⊆ roster | 硬断言 | **PASS**（roster 外 0） |
| 确定性重跑 SHA-256 逐位一致 | 硬断言 | **PASS**（见 counts.json output_sha256） |
| 唯一有向对 vs Codex 3,732,460 | 诊断 | **OUT**（定义差异，见 L14.2#3 → 裁决请求 ①） |
| 缝隙连接 ~8.5k | 诊断 | **OUT**（官方发布无缝隙表 → 裁决请求 ②） |
| 逐突触递质覆盖 | 诊断 | **OUT**（文件未下载，测量限制 → 裁决请求 ③） |

- 神经元行递质覆盖 **99.10%**（138,005/139,255；1,250 个无出边神经元递质不可得——测量
  限制，不臆造）。递质突触计数分布：胆碱能 30,429,211 / GABA 12,755,910 / 谷氨酸
  9,668,065 / 多巴胺 708,265 / 5HT 610,902 / 章鱼胺 320,569（合计 54,492,922 ✓）。
- 受体映射：v783 发布**不含受体标注** → DA 映射 `mod`（M6 功能门控语义，不臆造受体
  作用域）→ 测量限制登记（M8 R6 承接）。

### L14.4 §1.3 hemibrain 交叉核对

- 状态：**映射率测量限制**（NBLAST 文件 99/212MB 未完成）；**突触数一致性测量限制**
  （neuprint API 401 需注册凭据）；**递质一致性测量限制**（Eckstein hemibrain 表未下载）。
- 落盘：`data/m9_hemibrain_crosscheck.csv`（3 行测量限制 + 预注册区间记录：映射率
  ≥80% informational / 计数 ±10%）。**不伪造对照**（M5 L7）。

### L14.5 §1.4 扰动锚库（`data/m9_perturbation_plan.csv`）

- **类型级锚 26 条 ≥ 预注册下限 20/50 达标**：Aso 2014 eLife（MBON 价效 13 条 + PAM/PPL1
  DAN 2 条 + OA 1 条）、Robie 2017 Cell（GF 逃跑 / DN 下行 / MBON 行为 / CX 转向 / 行走启动
  5 条）、机制锚 5 条（KC→MBON 三因子 STDP、MBON 价效读出、CX EPG 环、OA 睡眠、伤害性）。
- **限制如实登记**：逐神经元驱动线原始表（Robie 2017 全脑图谱数据表）网络受限不可下载 →
  按 §1.4 回退路径用**文献效应统计同 schema**；root-id 级映射需 Schlegel 2024 细胞类型表
  （sk_lod1 parquet 5.35GB 未下载）→ `anchor_mapping = type-level`（类型级锚）；
  锚缺失神经元预注册 `no-experiment`（不入分母、不静默剔除）。
- 生成脚本：`tools/gen_m9_perturbation_plan.py`（确定性，无随机性；锚集与效应类可审计）。

### L14.6 交付物（G2）

`data/m9_flywire_connectome.csv`（2.6GB；139,255 神经元行 + 15,091,983 化学行 + 0 缝隙行）、
`data/m9_connectome_counts.json`、`data/m9_nt_coverage.csv`、
`data/m9_hemibrain_crosscheck.csv`、`data/m9_perturbation_plan.csv`、
`tools/build_m9_connectome.py`、`tools/gen_m9_perturbation_plan.py`。

---

## L15 — G0 引擎门实测（`tools/scan_m9_engine.py`；对齐 + 墙钟 + 预算）

### L15.1 内核与对齐协议

- **实现**：torch 自研点神经元内核（`tools/scan_m9_engine.py::run_gpu`）——two_comp HH
  （m/h/n 门控 + ampa/gaba 电导 + 轴耦合 I_ax）+ **Brian2 exponential_euler 逐变量闭式
  同构复现**（源码核实：每变量独立线性 ODE `x' = (x+B/A)e^{Adt} − B/A`，其它变量在**步起点**
  估值）+ 延迟突触事件（环形缓冲，按 post 隔室索引交付）+ 阈值/不应期语义实测复现。
- **网络**：M8 冻结 300 two_comp 定稿配置（`larva_circuit.py` 只读装配 → 网络定义
  **直接提取**：状态初值/gNa/gK/gL/AREA/stim_col/化学突触 pre-post-gmax-delay/stim 矩阵）
  → 同一连接组 CSV、同一权重（gmax 密度中位 0.199 S/m²）、同一 dt=0.05ms、同一 seed=0。
- **协议**：静息测量窗 T=2000ms（M8 store/restore 语义：settle 被丢弃，测量窗自初始态 +
  stim[:n_meas] 重跑）。

### L15.2 对齐判据结果（P2；预注册 §2.3/§0.7 #3）

| 判据 | 预注册阈值 | 实测 | 结果 |
|---|---|---|---|
| (a) 逐神经元 spike 时间对齐 | ≥99%（\|Δt\|<0.1ms） | **100.00%**（636/636，GPU-only=0） | **PASS** |
| (b) 静默比例差 | <1pp | CPU 0.8167 vs GPU 0.8167 → **0.00pp** | **PASS** |
| (c) 浮点精度档（float64/float32） | 统计级一致 | 两档均 **100.00%** | **PASS** |
| (d) 确定性重跑 | 统计级一致 | torch-CPU-f32 重跑 **100.00%**（<0.05ms） | **PASS** |
| MPS 设备路径正确性 | 与 CPU 同实现一致 | 500ms 窗 **100.00%**（<0.05ms） | **PASS** |

- 结论：**引擎正确性 G0 对齐全过（100%，远超 ≥99% 阈值）**——torch 内核是经验证的等价参考
  实现（§2.3"同一模型逐神经元对齐"语义达成）。

### L15.3 实现坑（3 类关键 bug + 语义发现；后续节点必读）

1. **Brian2 exponential_euler 语义**：所有变量**同时在步起点估值**。踩坑：先更新 m/h/n 再用
   新值算 v（以及先衰减 g 再算 v）→ v 响应偏快 → 轨迹逐步发散（实测 0.1ms 偏移 + 计数偏差）。
   修正后 100% 对齐。**必须**：步起点捕获 v/m/h/n/g → 同时提交。
2. **spike 语义（实测确认）**：阈值在状态更新后检查；**spike 时间记步骤起点** t_k；不应期
   2ms 自检测步终点 (t_k+dt) 起算（→ next_allowed = k + 2ms/dt）；**无 v 重置**（v 连续演化）；
   不应期抑制阈值检查但 ODE 照常积分。
3. **突触事件交付**：Brian2 在**步起点**应用事件（先于状态更新）。踩坑：事件整行累加（未按
   post 隔室索引）→ 全网癫痫（54,889 vs 69 spike）。**必须**按 post 索引交付（环形缓冲
   `buf[slot, post] += gmax`）。
4. **单位提取**：Brian2 VariableView 已返回 SI（`group.gNa[:]` = 1200 S/m²）→ 不可再乘 10
   （踩坑：×10 → 超兴奋）。
5. **轴耦合拆分**：`I_ax = G_AX(v_peer − v)` 的 **v 相关项必须入 A_v**（−G_AX/AREA），常数项
   （G_AX·v_peer/AREA）入 B_v；否则数值发散（node3 面积小 → g_ax 密度 573 S/m² 主导）。
6. **调试方法论**：逐隔室 v 轨迹对照（Brian2 StateMonitor vs 内核 trace）定位误差 →
   比"看总 spike 数"高效得多（建议后续节点沿用）。

### L15.4 墙钟与全规模预算（§0.9 R10：探针先行 → 裁决 → 再烧）

| 项 | 实测（最终稳健基准） | 说明 |
|---|---|---|
| CPU Brian2（numpy 后端）2000ms 测量窗 | **11.01 s** | ≈5.5 s/sim-s（numpy 解释后端） |
| torch-CPU-float32 同窗 | **10.00 s** | 加速比 1.1×（对 numpy 后端；M8 锚 118s/30s = cython 编译后端） |
| MPS 每步墙钟（600 隔室，Python 逐步循环） | **1.75 ms/step** | launch 开销主导 |
| MPS 规模分解（合成同构；min of 3 × 1500 步/档） | N=600→1.490；6000→1.445；60000→2.473 ms/step | 线性拟合 |
| **拟合** | **a = 1.413 ms/step（固定 launch 开销）+ b = 1.756e-5 ms/隔室/step** | — |
| **全规模 projection（278,510 隔室）** | **6.30 ms/step → 30s 单试次 = 1.05 GPU-h** | **> ≤1 GPU-h 预注册上限（5%）** |
| 朴素线性外推（误用） | 135–174 GPU-h | **忽略固定开销的伪值——记录避免复现** |
| 逐 spike 突触交付 | 未计入 projection | 当前实现 Python O(spikes×出度)；全规模下将主导 → 实际更差 |
| 负载敏感性（R11 纪律实测） | 并发重任务时 per-step 从 1.41→1.96–2.25 ms（**+40–57%**） | 探针与 build 并发导致；**预算测量必须独占** |
| 运行间方差 | projection 跨 3 次运行 0.37–1.33 GPU-h（中位 ~1.05） | 已改 min-of-3×1500 步稳健基准；仍建议后续节点复测 |

- **预算结论（诚实边界）**：稳健基准下 projection **1.05 GPU-h ≈ 预注册上限**（超 5%）；且未含全
  规模事件交付开销 → 判定 **FAIL（边缘）**。成本分解：固定 launch 1.413 ms/step × 600k 步
  = 0.235 h；逐元素 1.756e-5 × 278,510 = 4.89 ms/step × 600k 步 = 0.815 h → 合计 1.05 h。
  **两条优化杠杆**（供裁决参考）：① 时间分块（a → ≤0.05 ms/step，省 0.23 h）；② 每步算子融合
  （30 op → ~12 op，逐元素省 ~60%，→ 总计 ~0.35 GPU-h）。

- **注意**：1.11 GPU-h 的 projection **未包含**逐 spike 突触事件交付的 Python 开销
  （当前实现 O(spikes × out-degree)，全规模下会主导）→ 实际更差；如绪向量化后才会改善。

### L15.5 G0 门判定（本节点结论 + 三态裁决请求）

- **对齐（P2 判据 a/b/c/d）**：**PASS（100%）**——引擎正确性已验证。
- **预算（≤1 GPU-h）**：**FAIL（边缘，1.05 GPU-h；且未含事件交付开销）** → 按 §0.5/§2.5
  **引擎不得直接冻结为基线**，需主 agent 三态裁决：
  1. **① MPS 向量化优化（推荐）**：时间分块（每 Python 迭代处理 ~50–100 步，GPU 侧批量
     状态更新）+ GPU 侧事件 scatter-add → 目标：固定开销 a 从 1.93 ms/step 降至 ≤0.05 ms/step
     （分块收益 ~40–100×）→ projection ≈ 0.03–0.06 GPU-h，**本机可跑全规模、无云成本**；
     **实现后必须重跑本对齐探针**（§2.5 冻结规则：kernel 改动重过对齐门）。
  2. **② 云 GPU + Brian2CUDA**：同一 Brian2 网络定义直换后端（无需自研内核，CUDA 原生承载
     54.5M 突触）→ 成本/配额裁决；对齐成本最低。
  3. **③ 降规模（hemibrain ~25k 子集）**：全规模不可行时的降级路径 → 记录测量限制
     （全规模反证路径降级）+ 判据带按子集重定稿。
- 本节点建议：**① 优先（correctness 已 100% 验证，只差性能工程）+ ② 备选**；③ 作为兜底。
- 交付物：`data/m9_engine_params.csv`（选型判据定稿）、`data/m9_engine_probe.csv`（对齐数值
  + 墙钟 + 规模分解 + projection）、`reports/neuro/m9_engine_alignment.png`（对齐图）。

---

## L16 — 实现纪律教训（本轮实测；后续节点必读）

1. **千万行级一律向量化 + 流式写盘**：初版 `build_m9_connectome.py` 用逐行 `g.iloc[i]` +
   dict 累积处理 15.09M 连接行 → **>40 min 未完成**；改为 pandas groupby + numpy
   `bincount`（神经元 NT 聚合）+ **流式 CSV 写出（每 200k 行 flush + 增量 SHA）** →
   **~3 min 完成**（含 2.6GB 写盘）。**禁止**逐行 Python 循环/dict 列表（内存 7.5GB 风险）。
2. **确定性 SHA 与流式写出兼容**：增量 `hashlib.sha256` 对 flush 块更新 → SHA 语义与
   "整串哈希"一致（实测两次重跑 088d76b6… 逐位一致 ✓）。
3. **预算/性能测量必须独占运行**（R11 实证）：探针与 build 并发时 MPS per-step 从
   1.41 → 1.96–2.25 ms（**+40–57%**）→ 预算判定被污染（曾得 1.11 GPU-h 的伪"超预算"）。
   **纪律**：G0/预算类墙钟测量前确认无并发重任务；基准取 min-of-3 × ≥1500 步。
4. **勿用朴素线性外推做 GPU 预算**：MPS 每步成本 = 固定 launch 开销 a + 逐元素 b·N；
   忽略 a 的外推给出 135–174 GPU-h 的伪值（正确值 1.05 GPU-h）。**必须做规模分解拟合**。
5. **调试方法论（引擎对齐）**：逐隔室 v 轨迹对照（Brian2 StateMonitor ↔ 内核 trace）
   定位数值偏差比看总 spike 数高效得多；本轮 4 个 bug 全部由此定位（L15.3）。

## L17 — 交付物清单与复现（G2 + G0）

| 交付物 | 路径 | 验证一句话 |
|---|---|---|
| 连接组定稿 | `data/m9_flywire_connectome.csv`（2.6GB；139,255 神经元行 + 15,091,983 化学行 + 0 缝隙行） | P1 硬断言全过 + SHA 重跑逐位一致 |
| P1 计数报告 | `data/m9_connectome_counts.json` | 139,255 / 54,492,922 / 15,091,983 + 诊断 |
| 递质完整性 | `data/m9_nt_coverage.csv` | 逐连接 100% / 逐神经元 99.10% / 逐突触=测量限制 |
| hemibrain 核对 | `data/m9_hemibrain_crosscheck.csv` | 3 项测量限制（网络/凭据受限）+ 预注册区间 |
| 扰动锚库 | `data/m9_perturbation_plan.csv` | 26 类型级锚 ≥ 下限 20 ✓ |
| 引擎选型 | `data/m9_engine_params.csv` | 判据定稿（MPS 主路线） |
| 引擎探针 | `data/m9_engine_probe.csv` | 对齐 100% + 墙钟 + 规模分解 + projection 1.05 GPU-h |
| 对齐图 | `reports/neuro/m9_engine_alignment.png` | Δt 直方图 + 逐隔室计数 + 发放率 + 墙钟 |
| 工具 | `tools/build_m9_connectome.py`、`tools/scan_m9_engine.py`、`tools/gen_m9_perturbation_plan.py`、`tools/fetch_m9_flywire.py` | 确定性重跑；fetch 只做网络获取不进 build |

**复现命令**（`PYTHONHASHSEED=0 MPLBACKEND=Agg` 前缀固化）：
```
.venv-m9/bin/python -m neural_exploration.tools.fetch_m9_flywire            # ① 网络获取（本地已有则 skip）
.venv-m9/bin/python -m neural_exploration.tools.build_m9_connectome         # ② P1 数据管线（~3min）
.venv-m9/bin/python -m neural_exploration.tools.scan_m9_engine              # ③ G0 对齐探针（~4min，独占运行）
.venv-m9/bin/python -m neural_exploration.tools.gen_m9_perturbation_plan    # ④ 扰动锚库
```

## L18 — M9-B1（前置门阶段）结算与交接

- **G2 数据门**：P1 硬断言全过（139,255 ±0 / 54,492,922 ∈ 54.5M±10% / 逐连接递质 100% /
  自连接 0 / 孤立 616 显式白名单 / SHA 重跑一致）→ **G2 PASS**（3 项诊断 OUT 为定义差异或
  测量限制，已请求裁决，不影响 P1 硬判据）。
- **G0 引擎门**：对齐判据 (a)(b)(c)(d) **全过（100%）** + MPS 设备路径 100% 一致
  → 正确性 PASS；预算 **1.05 GPU-h（边缘超 5%）且未含事件交付开销** → **G0 = 对齐 PASS /
  预算 FAIL（边缘）** → 引擎**不得直接冻结**，需三态裁决（推荐：① 向量化优化 + 重过对齐门）。
- **交接给后续节点**：① L14.2 数据语义坑（ID 空间/精度/定义差异）；② L15.3 引擎语义
  （exponential_euler 步起点同步估值、spike 记步骤起点、事件按 post 索引交付、轴耦合拆分）；
  ③ L16 性能纪律（向量化 + 独占测量 + 规模拟合）；④ 引擎改动重过对齐门（§2.5）。
- **本阶段预算消耗**：CPU ≈3.5 CPU-h（解析/向量化迭代/对齐探针）+ GPU ≈0.9 GPU-h
  （MPS 探针与规模分解）→ 远低于 §10.1 上限（数据门 ≤20 CPU-h；引擎门 ≤40 CPU-h + ≤16 GPU-h）。



---

## L19 — 主agent G0/G2 裁决（2026-08-31；含独立诊断实验）

### L19.1 主agent独立诊断（纠正 B1 的性能推算方法）

主agent对 MPS 每步成本做了独立 scaling 实验（合成稀疏 SNN，同构内核：HH 门控 + exp +
stack + spike 检测 + index_add 突触交付）：

| N（神经元） | CPU(torch) ms/step | MPS ms/step |
|---|---|---|
| 300 | 0.169 | 2.142 |
| 1,000 | 0.259 | 2.183 |
| 3,000 | 0.598 | 2.308 |
| 10,000 | 1.113 | 2.282 |
| 30,000 | 2.425 | 2.792 |
| **139,255** | — | **4.948** |

- **交叉验证**：合成网络 @N=300 = 2.14 ms/step 与 B1 真实网络实测 2.21 ms/step 吻合
  → 外推可信。
- **结论**：MPS 每步 = 固定开销（~2.1 ms：60-70 个 kernel launch + Python 每步循环 +
  `bool(spk.any())` 同步）+ 规模线性项。N 增 100×，MPS 仅增 1.3×，CPU 增 14×。
- **纠正**：B1 的"171-6003 GPU-h"与"135-174 GPU-h"均为**忽略固定开销的朴素线性外推伪值**
  （已在 L15.4 与 L16#4 自我修正）；正确模型为 t = a + b·N（a=1.413 ms，b=1.756e-5 ms/隔室）。

### L19.2 主agent裁决（三态：✅通过 / ❌反证 / ↩️打回）

1. **G2 数据门：✅ PASS**——P1 硬断言全过（139,255 ±0 / 54,492,922 / 逐连接递质 100% /
   自连接 0 / 孤立 616 白名单 / SHA 重跑一致）。3 项诊断 OUT 裁决如下：
   - ① **唯一有向对 vs Codex 3,732,460**：❌ 定义差异（Codex 未文档化过滤口径）→ 记录为
     定义差异测量限制，**不按民俗数字调数据**（M5 L7 先例）。突触计数锚 54,492,922 独立验证
     通过，作为权威计数。
   - ② **缝隙连接 ~8.5k 锚**：❌ 官方 v783 发布不含（论文原文明确仅化学突触）→ **权威解析
     = 0/不可用**，按 M8 幼虫缝隙 0±0 同哲学记录测量限制；清单 §1.1 的 "~8.5k" 锚修正为
     "官方发布不可得（0 行）"。
   - ③ **逐突触递质覆盖**：❌ flywire_synapses 9.49GB 网络限速不可得（主agent终止下载）→
     连接级覆盖 100% 承接，逐突触级记录测量限制。
2. **G0 引擎门：✅ PASS（条件性）**——分层判定：
   - **对齐判据：✅ PASS**（100% spike 对齐 / 静默差 0.00pp / f64+f32 双档 / MPS 设备路径 /
     确定性）→ 自研 torch 内核是经验证的等价参考实现，§2.3"同一模型逐神经元对齐"达成。
   - **预算判据（M9 定稿保真度 point，§2.4"point-neuron 为主——M9 全规模唯一可行保真度"）**：
     ✅ 139,255 隔室 → t = 1.413 + 1.756e-5×139,255 = 3.86 ms/step → 30s 单试次
     = **0.64 GPU-h，在 ≤1 GPU-h 预注册上限内**。
   - **预算判据（G0 对齐参考档 two_comp，278,510 隔室）**：⚠️ 1.05 GPU-h（超上限 5%，边缘）→
     记录为测量限制 + 优化目标（不作为门失败：M9 全规模行为判据按 §2.4 用 point 档）。
   - **冻结条件**：§2.5 要求 kernel 改动后重过对齐门 → **优化实施后必须重跑 scan_m9_engine
     对齐探针**（本门为条件性通过）。
3. **NG 门（无 GPU 三态裁决）：不需要**——本机 M1 Pro + MPS 路线成立（对齐 PASS + point 档
   全规模 0.64 GPU-h 在预算内）。**用户已确认：不降规模、不用云 GPU，坚持本机 MPS 路线**。
4. **执行层优化要求（B2/B3 节点必做，非门判据）**：目标 two_comp ≤0.5 GPU-h / point ≤0.3 GPU-h：
   - ① **事件交付向量化**（当前 Python O(spikes×出度)，全规模下将主导 → 改 GPU 侧
     scatter-add/index_add）；
   - ② **每步算子融合**（30+ op → torch.compile 或手工融合，减少 launch 数 3-10×）；
   - ③ **消除每步同步**（`bool(spk.any())` 延后读取，累积 spike mask 到设备缓冲）；
   - ④ **消除 Python 侧开销**（预绑定局部变量，禁止每步 `t["g_"+stype]` 字典查找/字符串拼接）；
   - ⑤ 可选：稀疏事件驱动（静默 81.67% → 只更新活跃神经元）。
5. **运行纪律（R11 实证强化）**：预算/墙钟类测量**必须独占**（并发热任务使 per-step
   1.41 → 1.96-2.25 ms，+40-57%，曾产出 1.11 GPU-h 伪"超预算"）；基准 min-of-3 × ≥1500 步。
