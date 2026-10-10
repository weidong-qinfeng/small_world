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

### L15.4 墙钟、真实网络 scaling 与全规模预算（**修正模型：固定开销 + 线性项**）

**主 agent 诊断修订**：不得把 300 档（近纯固定开销）的 ms/step 按规模线性放大（该错误外推给出
135–174 GPU-h 的伪值，已从 `m9_engine_probe.csv` 删除相关行）。正确模型 `t(N) = a + b·N`。

**本节点实测（`data/m9_engine_scaling.csv`；`tools/scan_m9_engine_scaling.py`，500ms 短窗）**：

| 类型 | N（隔室） | MPS ms/step | 说明 |
|---|---|---|---|
| 真实 LarvaCircuit | 600 | **2.385** | scale=300 two_comp（min of 2） |
| 真实 LarvaCircuit | 2000 | **2.662** | scale=1000 two_comp |
| 合成同构 | 600 | 2.147 | min of 3 × 300 步 |
| 合成同构 | 3000 / 10000 / 30000 / 60000 | 2.410 / 1.764 / 2.066 / 2.606 | 噪声 ±20%（本机负载） |
| **合成同构（全规模档）** | **139,256** | **3.341** | **直接实测全规模量级（非外推）** |
| torch-CPU 对照 | 600 / 2000 | 0.245 / 0.411 | CPU 每步随 N 明显增长（真实网络） |

- **拟合（本机 6 点）**：a = 2.019 ms/step（固定开销）+ b = 9.14e-6 ms/隔室/step
  → 全规模 **3.29 ms/step**；**b 在本尺寸范围内噪声大**，故给出区间：取主 agent 独立宽范围表
  斜率 b=2.0e-5（N=300→30000 拟合）作上界 → 全规模 4.80 ms/step。
- **全规模 30s 单试次（600k 步）projection = 0.55–0.80 GPU-h ≤ 1 GPU-h ✓**
  （与主 agent 独立推算 0.82 GPU-h 一致）；CPU 全规模外推 ≈13h/试次（不可行）→
  **MPS 相对 CPU 约 16× 优势**。
- 结构结论：**MPS 每步成本 = 固定开销（~2.0 ms：60–70 次 kernel launch + Python 循环 +
  每步 `bool(spk.any())` 同步）+ 弱规模线性项**；N×100 → MPS 仅 ×1.3–1.6，CPU ×14。
- 交叉核对：真实网络（600→2.385 / 2000→2.662）与合成（600→2.147）同量级 ✓ → 外推可信。

### L15.4b 真实内核优化（**已通过严格等价性门**）

| 尝试 | 结果 |
|---|---|
| ② 真实内核融合版 `run_gpu_fast`（预绑定局部变量、去 `torch.stack`、去 `\|A\|<1e-12` 守卫、in-place 衰减、CSR 出边表） | **1.694 ms/step vs 基线 2.459 ms/step = 1.45×**，**严格等价性 100.00%（162/162 spike 完全一致，容差 0ms）** |
| 合成版融合（`_probe_m9_mps_fast.FastKernel`） | 5.06–6.36×（合成无突触；含 4 个 `torch.where` 守卫的开销被消除） |
| ① `torch.compile`(inductor) MPS | **35.8×**（0.042 ms/step @N=600；首次编译 0.8s）——数据依赖控制流（spike 分支）会 graph-break，需按段编译 |
| ③ launch 地板（1 op/step） | 0.0074 ms/step（融合理论下限量级） |
| 修正记录 | 首次实现漏掉 h 门控 `bh` 的 `+30` 偏移（`-(vm+35)/10 ≠ (vm+65)/10`）→ 等价性检查 **0.00% 暴露该 bug**，修正后 100%——**等价性门有效**（已写入 `data/m9_engine_optimization.csv` 与本节） |

### L15.5 G0 门判定（**PASS —— 无需 NG 门三态裁决**）

- **对齐（P2 判据 a/b/c/d）**：**PASS 100%**——spike |Δt|<0.1ms 100.00%（636/636，GPU-only=0）、
  静默差 0.00pp、float64/float32 均 100%、确定性重跑 100%、MPS 设备路径 100%。
- **性能/预算**：按修正模型（固定开销 + 线性项）+ **全规模档直接实测（N=139,256 隔室 →
  3.341 ms/step）** → 全规模 30s 单试次 **0.55–0.80 GPU-h ≤ 1 GPU-h ✓**（CPU 外推 ≈13h/试次，
  MPS 约 16× 优势）→ **PASS**。
  - 自我修正记录：本节点早前给出的 135–174 GPU-h"超预算"结论源于**错误的线性外推**
    （把 300 档近纯固定开销 2.2 ms/step 按规模放大）→ 已删除该推算行；正确模型为
    `t = a + b·N`（a≈2.0 ms 固定 + b≈9e-6–2e-5 ms/隔室）。
- **结论：G0 门 = 对齐 PASS + 性能 PASS → 本机 MPS 路线成立，不需要 NG 门三态裁决**
  （云 GPU 不再必需）。
- **冻结前必做的两件事（§2.5 纪律，写清给后续节点）**：
  1. 融合/编译内核（`run_gpu_fast` 已验证 1.45× 且 spike 严格一致；compile 档 35.8×）
     **接入 GPU 侧向量化突触事件交付**后，**必须重跑 G0 对齐探针**（现有 100% 对齐对基线
     与融合版均成立，但接入后的最终内核须复测）；
  2. `torch.compile`/MPS 与 torch 版本绑定（本机 2.8.0）→ 换版本须重测；数据依赖控制流
     （spike 分支）会 graph-break，需按"神经更新段 / 事件交付段"分段编译。
- **备用方案（仅当接入后复测超预算时启用；当前不需要）**：① 云 GPU + Brian2CUDA（同定义直换
  后端）；② 降规模（hemibrain 25k / MB+CX+感觉运动子集，记录测量限制）；③ 极短协议 T≤1–2s
  （统计功效受限）。各选项工程代价见 L15.5 旧表（本节点保留记录）。

### L15.5b 早前优化记录（合成版；保留可审计）

- 合成融合版（无突触）：5.06–6.36×；`torch.compile`：35.8×（0.042 ms/step @N=600）；
  launch 地板（1 op/step）0.0074 ms/step；真实内核融合版 1.45× + 严格等价 100%（L15.4b）。
- 结论修正：**"MPS 对稀疏 SNN 无优势"不成立**——瓶颈是 Python 每步 launch 开销与
  `bool(spk.any())` 同步（结构性可优化），非内存带宽/架构限制。

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
   把 300 档（近纯固定开销）ms/step 按规模放大会给出 135–174 GPU-h 的伪值。正确模型
   `t=a+b·N`：本机实测 **a≈2.02 ms/step，b≈9.1e-6–2.0e-5 ms/隔室/step** →
   **全规模（139,256 隔室）直接实测 3.341 ms/step → 30s 单试次 0.55–0.80 GPU-h ✓**。
   必须做规模分解拟合 + 至少覆盖一个全规模量级档（本节点已补）。
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
  → 正确性 PASS；基线实现预算 **中位 ≈1.1 GPU-h（0.37–1.30，超 ≤1 上限）** → **G0 = 对齐 PASS /
  性能 FAIL（基线实现）**；**但有限优化实测 5.06–6.36×（compile 档 35.8×）→ 优化后全规模
  0.28 GPU-h（≤1 ✓）** → 引擎**不得以基线实现冻结**；需三态裁决（推荐：① 融合内核 +
  GPU 侧事件交付 + **重过对齐门**；② 云 GPU Brian2CUDA；③ 降规模）。
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

## L19 — 抑制/递质清单（真实抑制边承接 M8 R2；H6 消融接口）

`tools/gen_m9_inhibition_inventory.py` → `data/m9_inhibition_inventory.csv`（43 行；从
`m9_flywire_connectome.csv` 15.09M 化学行流式聚合，pyarrow CSV 分块）：

| 维度 | 实测 |
|---|---|
| 递质构成（突触占比） | 胆碱能 30,429,211（**55.84%**）/ **GABA 12,755,910（23.41%）** / 谷氨酸 9,668,065（17.74%）/ DA 708,265（1.30%）/ 5HT 610,902（1.12%）/ 章鱼胺 320,569（0.59%） |
| 连接数构成 | 胆碱能 8,075,700 / **GABA 3,233,022** / 谷氨酸 2,966,878 / DA 413,938 / 5HT 225,156 / 章鱼胺 177,289 |
| **受体映射（引擎边类型）** | **gaba（抑制性离子通道）3,233,022 连接 = 21.42% / 12.76M 突触 = 23.41%**；ampa（兴奋性占位）11,042,578 连接 = 73.17%；mod（调质门控）816,383 连接 = 5.41% |
| 神经元级 dominant 递质（有出边 138,005 个） | 胆碱能 90,478（65.56%）/ 谷氨酸 24,987（18.11%）/ **GABA 19,182（13.90%）** / 5HT 2,213 / DA 896 / 章鱼胺 249 |
| GABA 边端点 | GABA 递触前神经元 **104,634** 个；**接收 GABA 边神经元 129,857 / 139,255 = 93.3%** |
| GABA 突触脑区 top | LOP_R 389,612 / AVLP_L 337,284 / LOP_L 295,095 / SAD 224,076 / EB 196,812 / PVLP_L 190,106… |

- **科学含义（M8 R2 承接）**：M8 的抑制边由 class hash 回退分配（非真实）；M9 v783 提供
  **Eckstein 2024 逐连接真实递质预测** → **真实抑制边可得且规模明确**（21.4% 连接 / 23.4%
  突触，覆盖 93.3% 神经元）→ H6（真实 GABA 抑制平衡）检验的条件**在数据侧已满足**，
  P4 双状态与 §3.5.5 消融（GABA 边 g→0）有明确的边集接口与规模。
- **注意（诚实性）**：dominant 递质为**连接级**（六递质概率 syn 加权 argmax），非逐突触
  分类；逐突触级（flywire_synapses 9.49GB）为测量限制（L14.1）。共递质（如胆碱能+谷氨酸）
  神经元按 dominant 归类，其受体占位按 dominant 映射（apa/mod）——下游若要严格共释放建模，
  需逐突触级数据（测量限制登记）。

## L20 — 状态核对（对应主 agent 待办 1–4）

1. **G0 引擎探针**：**已完成**（本节点早前执行）——`data/m9_engine_probe.csv` +
   `reports/neuro/m9_engine_alignment.png`；对齐 **100.00%**（636/636，\|Δt\|<0.1ms，0 多余）、
   静默差 **0.00pp**、确定性 100%、MPS 设备路径 100%；墙钟与全规模 projection 见 L15.4。
2. **扰动锚库**：**已完成**——`data/m9_perturbation_plan.csv`，类型级锚 **26 条 ≥ 20/50** 达标。
3. **env notes L14+**：**已完成**（L14 数据管线/下载裁决、L15 G0 实测、L16 实现纪律、
   L17 交付物与复现、L18 结算、L19 抑制清单、本 L20）。
4. **结算**：已按四段格式提交（G2 PASS / G0 对齐 PASS + 预算 FAIL 边缘 + 裁决请求）。
- **勘误（对照主 agent 状态摘要）**：唯一有向对的**解析值 = 15,091,983**（"任一突触即算连接"），
  Codex 官方 **3,732,460** 属另一统计口径（过滤阈值未文档化；实测 syn≥5→2,700,513、
  ≥3→4,916,231 均不匹配）→ 已作为**定义差异诊断 OUT** 入档并请求裁决（**不以 3,732,460
  作为 P1 硬断言通过项**，避免按民俗数字；突触计数锚 54,492,922 独立验证通过）。

---

## L20 — 主agent裁决更新（2026-08-31 晚；B1 优化实测后）

### L20.1 新证据：MPS 性能瓶颈是 Python launch 开销（可消除），非架构瓶颈

B1 优化尝试实测（`data/m9_engine_optimization.csv`；`tools/_probe_m9_mps_fast.py`）：

| 实现 | N=600 | N=6000 | N=60000 | 加速比 |
|---|---|---|---|---|
| 基线（Python 每步循环 + 小张量 + 守卫） | 1.503 | 1.886 | 2.983 ms/step | 1× |
| **② 手写融合 + in-place + 预分配 + 去守卫** | **0.298** | **0.296** | **0.586** | **5.05-6.36×** |
| **① torch.compile(inductor) MPS** | **0.042** | — | — | **35.8×**（首次 ~0.8s 编译） |
| ③ launch 地板（1 op/step，理论下限） | 0.0074 | — | — | — |

- 融合手段（未改核心逻辑）：门控闭式 `(x−α/(α+β))e^{−(α+β)dt}+α/(α+β)`；v 闭式
  `(v−B_v/gsum)e^{−gsum·dt/Cm}+B_v/gsum`；**删除 `|A|<1e-12` 守卫**（HH 下 α+β>0、
  gsum>0 恒成立 → 守卫是纯开销）；`mul_` in-place 衰减；预分配缓冲。
- 规模拟合：基线 a=1.612 ms + b=2.305e-5 → **1.34 GPU-h**；**优化② a=0.281 ms + b=5.065e-6
  → 0.28 GPU-h ✓ 在 ≤1 GPU-h 预算内**。
- 健全性：float32/64 轨迹 finite、稳态 −65.8mV、无自发发放。

### L20.2 裁决更新（覆盖 L19.2 第 2 条的性能部分）

1. **G0 引擎门：✅ PASS（本机 MPS 路线确立）**——对齐 100%（基线内核）+ 优化实测
   **two_comp 全规模 0.28 GPU-h / point 档更低，均在 ≤1 GPU-h 预算内**。L19.2 中
   "two_comp 1.05 GPU-h 边缘超 5%" 的保守判定**被优化证据取代**：MPS 路线成立，
   **NG 门三态裁决不需要**（用户已确认本机路线）。
2. **🔴 强制前置（B2 必做顺序）**：优化内核必须 ① **接入向量化突触事件交付**
  （GPU 侧 `scatter_add_`；基线为 Python O(spikes×出度)，全规模将主导）→ ② **重跑 G0
   对齐门**（§2.5：kernel 改动重过对齐门；现有 100% 对齐只对基线内核成立）→ ③ 通过后才
   可用于全规模协议。**对齐不达 100% 的优化一律弃用**（记录）。
3. **技术参考**：B1 的融合公式、去守卫、in-place、torch.compile 手段直接可用
   （`tools/_probe_m9_mps_fast.py` + `data/m9_engine_optimization.csv`）。
4. **判据带/消融接口（新交付）**：`data/m9_inhibition_inventory.csv` 给出 H6 消融的可执行
   边集——**gaba 抑制边 3,233,022 连接（21.42%），受 GABA 神经元 129,857/139,255 = 93.3%**
   → §3.5.3 双状态与 §3.5.5 抑制平衡消融判据具备执行条件（边集 g→0）。
   **M8 R2（缺 GABA 标注 → 抑制平衡缺失）在数据侧解除**——这是 M9 相较 M8 的核心结构优势，
   P4 自发分布（M8 反证 run=9.9%）是否因真实抑制边而改善，是 M9 首要科学检验点。
5. **勘误确认（主agent记录）**：唯一有向对解析值 = **15,091,983**（"任一突触即算连接"）；
   **3,732,460 是 Codex 官方统计口径**（过滤阈值未文档化，实测 syn≥5→2,700,513、
   syn≥3→4,916,231 均不匹配）。主agent此前摘要中以 3,732,460 作"唯一有向对"表述**有误**，
   正确表述以本行为准；P1 硬断言通过项不含该项（定义差异诊断 OUT）。

## L21 — 内核版本区分（重要，防混淆；主agent 2026-10-10 记录）

M9 存在**两个内核版本**，其规模拟合斜率 b 差异巨大，引用时必须指明版本：

| 内核版本 | a（固定）ms/step | b（每隔室）ms/step | 全规模 point projection | 依据 |
|---|---|---|---|---|
| **基线内核**（`scan_m9_engine.py::run_gpu`，Python 每步循环 + 60+ 小操作） | 1.886 | **3.371e-04** | **48.8 ms/step → 8.14 GPU-h**（超预算） | `data/m9_engine_scaling.csv`（真实网络 300/1000 档拟合） |
| **融合+向量化内核**（`src/adult_engine.py`，torch.compile 融合 + GPU 侧事件交付） | 0.786（chunk）/ 1.939（step） | **≈0（未饱和）** | **3.00 ms/step → 0.5007 GPU-h ✓**（chunk）/ 5.83 → 0.9716 GPU-h（step） | `data/m9_engine_vec_probe.csv`（合成 point 网络 10k/30k/139,255 三点规模分解，min-of-3 × 1500 步） |

- **为何 b≈0**：融合后每步只有少数几个大 kernel，139,255 神经元的向量操作在 M1 Pro GPU 上
  远未饱和并行度 → 每步时间由 kernel launch + 带宽决定而非规模。**独立佐证**：主agent 合成网络
  基线实测 4.948 ms/step（N=139,255），融合 5-6× → ≈0.8-1.0 ms/step，与 0.786 量级吻合。
- **G0 判定以融合内核为准**（§2.5：融合内核已重过对齐门，四档全 100%）→ **0.5007 GPU-h ✓**。
- **纪律**：引用"MPS 全规模 projection"必须标注内核版本 + 投递调度（chunk/step）；不得混用。
  基线内核的 8.14 GPU-h 是**优化前**状态，记录为改进依据而非当前结论。
- **待验证（B2 进行中）**：真实网络（非合成）全规模静息实测的 ms/step（P4 协议），作为
  projection 的最终确认；若真实活跃比例远高于合成的 2%，b 可能回升 → 以实测为准。

## L22 — B1 最终结算（2026-10-10；G0 = PASS 定稿）

1. **全规模量级直接实测（决定性，非外推）**：合成同构网络 **N=139,256 隔室 → 3.341 ms/step**
   → 30s 单试次（600k 步）= **0.557 GPU-h ✓ ≤1 GPU-h**。与主agent独立推算 0.82 一致。
   → **G0 性能判据 PASS**（不再依赖任何外推）。
2. **真实网络 scaling 复核**：真实 LarvaCircuit 300 two_comp（600 隔室）2.385 ms/step、
   1000 two_comp（2000 隔室）2.662 ms/step → N×3.3 而 MPS 仅 ×1.12（固定开销主导再次确认）；
   torch-CPU 对照 0.245→0.411（CPU 随 N 显著增长）。scale=3016 档跳过并记录（CPU numpy 基线
   >15min 未完成，预算纪律）。
3. **融合内核（真实网络版 `run_gpu_fast`）**：2.459 → **1.694 ms/step（1.45×）**，**严格等价性
   100.00%（162/162 spike，容差 0ms）**。**等价性门发挥作用**：首版漏了 h 门控 `bh` 的 +30 偏移
   （`-(vm+35)/10 ≠ (vm+65)/10`）→ 门报 0.00% 暴露 bug → 修正后 100%。**这条经验必须进 M10 惯例**：
   kernel 优化必须配严格逐 spike 等价门（容差 0ms），否则隐性语义偏差会被性能收益掩盖。
4. **`torch.compile` 实测 35.8×**（0.042 ms/step @N=600；编译 0.8s）；spike 分支会 graph-break
   → 按"神经更新段/事件交付段"分段编译。
5. **G0 定稿**：对齐 PASS（基线 + 向量化内核四档全 100%）+ 性能 PASS（0.557-0.80 GPU-h）
   → **本机 MPS 路线成立，NG 门三态裁决不需要**。冻结前必做：融合/编译内核接入 GPU 侧向量化
   突触交付后**重跑对齐门**；torch/MPS 版本绑定 2.8.0（换版本需重测）。
6. **`data/m9_engine_probe.csv` 清理**：已删除全部朴素外推行（`est_*`、`budget_*_naive`），
   新增 `projection_model,a+b*N` 行；当前 `proj_full_30s_gpu_h = 0.718`、`budget_le_1_gpu_h = True`
   （注：该 0.718 为 a+b·N 拟合值；全规模直接实测为 0.557 —— 两者均在预算内，以直接实测为准）。

---

## L20 — M9-B2 节点：向量化内核 + 全规模装配 + P4 静息（§2.5 重过对齐门 / §3.5）

- 执行节点：**M9-B2**（本节点）。新建文件：`src/adult_engine.py`、`src/adult_circuit.py`、
  `tools/scan_m9_engine_vec.py`、`tools/build_m9_network.py`、`tools/calibrate_m9_weights.py`、
  `tools/validate_p9_resting.py`、`data/m9_behavior_reference.csv`、`data/m9_network*.npz`、本节
  L20–L22。**冻结件零修改**：`tools/scan_m9_engine.py`（B1 件）只读 import（`_load_circuit` /
  `extract_network` / `run_cpu` / `compare_spikes` / `silent_fraction` + 常量）。
- 运行前缀固化：`PYTHONHASHSEED=0 MPLBACKEND=Agg ./.venv-m9/bin/python`。

### L20.1 五项向量化优化的实测结论（L19.2 执行层要求①–⑤）

| 优化 | 实现 | 实测效果 |
|---|---|---|
| ① 事件交付向量化 | CSR-by-pre + 活跃集压缩 + GPU scatter-add（替代 Python O(spikes×出边) 循环） | 交付不再是瓶颈；见 L20.2 的调度选择 |
| ② 算子融合 | `torch.compile`（inductor/MPS）：8 op 链 0.564 → 0.094 ms（**隔离基准**） | ⚠️ **在全规模真实循环中反而更慢**（见 L20.2#5）→ **默认关闭** |
| ③ 消除每步同步 | 不应期改**倒计时** `cool`（与 `s >= next_allowed` 严格等价，推演见 `_point_core_factory`）；spike 计数全程设备侧累积 | 每步零 sync（仅 chunk 末一次 `.item()`） |
| ④ 消除 Python 侧开销 | 环形槽/记录行**视图预计算**（消除每步切片对象创建）、张量全部预绑定 | 内循环 Python 语句数降到 ~8；实测 Python/launch 开销与 GPU 执行同量级（见 L20.2#5） |
| ⑤ 稀疏事件驱动 | 只处理 spiking 神经元的出边 + **chunk 批量投递（W 步一次）** | 关键杠杆：固定开销摊薄 W 倍，0.5ms 延迟档 1.19 → 2.0ms 延迟档 0.58 ms/step |

### L20.2 MPS 性能坑（**后续节点必读**；均已实测）

1. **`index_add_` 的成本随「目标张量元素数」线性，与索引数无关**：目标 5.85M 元 ≈0.20ms
   （索引 14 个与 14,000 个同价）；目标 278k 元 ≈0.04ms。→ **逐步把少量事件投进大环形缓冲是
   灾难**（曾每步 0.2ms，实测把内核从 0.6 拖到 2.9 ms/step）。
   **对策**：外部/背景事件按 **chunk 批量 + `index_put_(accumulate=True)`**（0.007ms/次）。
2. **`nonzero` / `repeat_interleave` 是固定开销**：`nonzero`(139k) ≈1.0ms、`repeat_interleave`
   ≈0.73ms（与长度无关）→ 必须靠 chunk 摊薄；`nonzero` 动态形状还会触发内部同步。
3. **环形槽数必须 ≥ W + max_delay**（chunk 批量写在 chunk 末步之后）：参考实现 R=max_delay+1
   **不支持** chunk 投递；本实现 R = max_delay + SLOT_EXTRA + 1。放宽槽数**不改变语义**
   （`d<R` 时写槽 `(s+d)%R` 首次被读即为步 s+d）——已由 step↔chunk 逐 spike 一致验证。
4. **外部事件的延迟必须与网络事件一致**（同目标槽语义）：否则会被本 chunk 已读并清零的槽吞掉。
5. **`torch.compile` 在 MPS 上对本负载是负优化**（反直觉，实测）：
   - 隔离基准：融合后 0.094 vs 0.564 ms（6× 加速）；
   - **真实逐步依赖循环**（每步输出喂下一步）中：compiled 1.82–2.35 ms/step vs **eager
     1.19–1.21 ms/step**（+50–100%）。原因：编译产物每次调用的 dynamo guard/dispatch
     开销 + 逐步依赖下无法用吞吐基准外推。
   → **决策：M9 全规模默认 `use_compile=False`（eager）**；记录为「L19.2 优化② 在本机
     MPS 上不适用」的方法学修正（**禁止用隔离吞吐基准推断逐步依赖内核的墙钟**，L16#4 同源）。
6. **同步会隐藏「Python 发射 vs GPU 执行」的真实归属**：chunk 末的 `.item()` 会把前面 W 步
   的 GPU 排队时间记到"chunk 相位"上 → 相位计时只能用于**相对**定位，不能直接当作算子耗时。

### L20.3 对齐门重跑（§2.5 冻结规则：kernel 改动必须重过对齐门）

- 探针：`tools/scan_m9_engine_vec.py`（只读复用 B1 的网络提取与判据固定件）。
- 结果：**判据 (a)–(d) 全过**——四种变体（torch-CPU-f32 step / chunk、MPS-f32 step / chunk）
  **逐神经元 spike 对齐均 100.00%**（636/636，GPU-only=0）；静默比例 0.8167（与 CPU Brian2
  差 0.00pp）；确定性重跑 100%（<0.05ms）。
- **投递调度等价性**：`step` 与 `chunk` 两种批量调度**逐 spike 完全一致**（max Δt = 0.000ms）
  → chunk 优化不改变语义（L20.2#3 的槽数放宽亦被此验证覆盖）。
- 交付：`data/m9_engine_vec_probe.csv` + `reports/neuro/m9_engine_alignment_vec.png`。

### L20.4 全规模装配实测（§3.5.1）

- `tools/build_m9_network.py`：2.4GB CSV → npz（pyarrow 流式列选择 + int64 统一 dtype 的
  searchsorted 映射，**10.2s 解析 / 31.9s 全量落盘**；L16#1 纪律：向量化，无逐行循环）。
- 网络：139,255 神经元 + **15,091,983 化学连接**（54,492,922 突触）+ **0 缝隙**（L19.2 裁决②）
  - 兴奋性 11,042,578 边 / **抑制性 3,233,022 边（真实 GABA，12,755,910 突触 = 23.4%）** /
    调质类 816,383 边（DA/5HT/OA，抽象 T2：P4 以弱兴奋电导承载）。
- 度分布：出度均值 108.4 / 中位 79 / p99 598 / **最大 9,783**（重尾枢纽）；
  `syn_count` 均值 3.61 / 中位 2 / 最大 2,405。
- `src/adult_circuit.py` 装配实测：**分段装配 0.8–2.2s**；**MPS 显存 1.05–1.33 GB**
  （§2.2 预注册 <1.5GB ✓）；环形槽 21–81（随延迟档）。
- 抽象登记（§0.3.5，逐条替代对象/误差/回归条件）：point-neuron 替代 HH / **统一轴突延迟
  1.0ms**（FlyWire v783 不含延迟列）/ 调质→弱兴奋电导 / 递质类级权重 / 背景 Poisson 驱动。

### L20.5 全规模 point 档每步墙钟（实测，独占运行；`data/m9_engine_vec_probe.csv`）

| 活跃水平（每步活跃神经元占比） | ms/step（N=139,255） | 30s 单试次 | 说明 |
|---|---|---|---|
| 0.10%（ref=200ms） | **0.539** | **0.090 GPU-h** | 极稀疏 |
| 0.20%（ref=20ms） | **0.418** | **0.069 GPU-h** | 稀疏 |
| 0.98%（ref=2ms，饱和档） | **1.017** | **0.169 GPU-h** | 全部神经元以不应期上限发放 |
| 活动≈0（无 spike） | 0.20 | 0.03 GPU-h | 内核下界（core+ring+chunk 固定开销） |

- **规模分解**（合成 point 网络、出度 108、活跃比例由不应期精确控制）：t = a + b·N，
  a = 0.29 ms（launch/Python 固定）–0.07 ms，b = 5.2e-6 – 3.3e-6 ms/神经元/步
  → 全规模外推与实测吻合（1.016 vs 实测 1.017 ms/step @0.98% 活跃）。
- **P4 实测对照**：定稿静息档实测 **0.898 ms/step**（试次 1/2；试次 0 含 MPS 分配器预热
  4.09 ms/step——**预算判定必须用稳态值**）→ 30s 单试次 **0.15 GPU-h**。
- **two_comp（对齐参考档）**：实测 600 隔室 0.238 ms/step；按同一内核骨架外推
  278,510 隔室 ≈ 0.287 ms/step → 30s 0.048 GPU-h（**B1 基准 1.05 GPU-h → 22× 改善**，
  L15.4/L19.1 的预算问题定量解决）。
- **优化前后对照（B1 基准 `data/m9_engine_probe.csv`）**：B1 point 投影 3.86 ms/step
  （未含事件交付）→ **B2 实测 0.90 ms/step（含全量 15.1M 边事件交付 + 背景驱动）**；
  30s 单试次 0.64 → **0.15 GPU-h**（§0.9 R10 ≤1 GPU-h ✓，6.6× 余量）。
- **主 agent ≤0.3 ms/step 目标：未达**（实测 0.42–1.02 ms/step 视活跃度；低活动下界 0.20）。
  瓶颈为**逐步依赖下的 GPU 执行与 Python 发射同量级**（L20.2#5/#6），非算法层。
  **预算判据（§0.9 R10）通过** → 建议按"目标修正"裁决（L23 ④）。

### L20.6 M9-B2 交付物清单（本批次）

| 交付物 | 路径 | 验证一句话 |
|---|---|---|
| 向量化内核 | `src/adult_engine.py` | 对齐门 4 变体 **100%**（§2.5 重过）+ step↔chunk 逐 spike 一致 |
| 全规模回路 | `src/adult_circuit.py` | 139,255 神经元 / 15,091,983 边 / 构建 0.79s / 显存 1.047GB |
| 网络装配 | `tools/build_m9_network.py` → `data/m9_network.npz` | 54,492,922 突触 / 真实 GABA 3,233,022 边（21.4%） |
| 对齐探针 | `tools/scan_m9_engine_vec.py` → `data/m9_engine_vec_probe.csv` | 对齐 gate PASS + 规模分解 + 全规模投影 |
| 标定 | `tools/calibrate_m9_weights.py` → `data/m9_weight_calibration.csv` | 6 配置实测 + 落带判定（拟合集 A） |
| P4 验证 | `tools/validate_p9_resting.py` → `data/m9_p4_resting.csv` + `m9_p4_resting.png` | 6 判据：4 过 2 不达（反证记录） |
| 判据带 | `data/m9_behavior_reference.csv` | resting 段 8 行 + protocol_change 1 行（运行前定稿） |
| 回路参数 | `data/m9_circuit_params.csv` | 定稿参数 + 抽象登记 |

### L21.4 标定实测记录与 T 敏感性（测量限制）

- `data/m9_weight_calibration.csv`（4 配置，T=1s，拟合集 A）：全部 `in_band=False`——因
  **中位率恒为 0**（(a) 判据结构性不达，见 L21.3#1）；静默比例在 0.834–0.919 之间，
  仅 b0=290/320 档落在 [0.5,0.9] 内。
- **静默比例具 T 敏感性**（测量限制，须入档）：同一配置 T=1s → silent 0.892，
  T=2s → **0.747**（P4 实测），T 延长会继续下降（低率神经元的 spike 累积使率估计上移）。
  → **P4 判据 (b) 的通过判定只在 T=2s 协议下成立**；T=30s 下 silent 可能出带（<0.5）。
  建议后续节点在 T=30s 复测该项（本批次墙钟不允许）。
- 标定主杠杆排序（实测）：**背景率 λ**（近阈值神经元的发放率 ≈ λ）> **bias 中位 b0**
  （决定越阈值神经元占比 → 静默比例）> AHP（发放率上限）> 类级权重 w_exc/w_inh
  （弱复发档必要；强复发档 → 自持饱和出带）。

---

## L24 — M9-B1 融合手段的接入与实测复核（B2 补充；§2.5 强制顺序）

### L24.1 采用的 B1 融合手段（`tools/_probe_m9_mps_fast.py` / `data/m9_engine_optimization.csv`）

按 B1 给定手段接入 `src/adult_engine.py`（**语义不变，只改实现**）：

1. **门控闭式**：`x ← (x − α/(α+β))·e^{−(α+β)dt} + α/(α+β)`（替代 `stack` + `where` 分支）；
2. **v 闭式**：`v ← (v − B_v·Cm/gsum)·e^{−gsum·dt/Cm} + B_v·Cm/gsum`；
3. **删除 `|A|<1e-12` 守卫**（HH：α+β>0、gsum ≥ gL = 3 S/m² > 0 恒成立 → 守卫纯 launch 开销）；
4. **m/h/n 用独立张量**（免 `stack`/解包）；5. 电导衰减/状态更新用 `mul_`/`add_` in-place。
- **踩坑（本轮实测）**：v 闭式**必须带 Cm 因子**——首版写成 `B_v/gsum`（漏 `·Cm`）→ 网络全静默
  （探针 spike=0）。参考实现里 `B/A = B_v/A_v = −B_v·Cm/gsum`。修正后对齐门 PASS。
- **已评估-不采用**：把闭式 v 也搬到 point 档（指数积分）→ 算子数反而 11 → 13，
  且会改变已定稿的 P4 静息标定 → 弃用（记录）。

### L24.2 对齐门重跑（§2.5 强制：① 融合内核 → ② 事件交付向量化 → ③ 重过对齐门 → ④ 才用于全规模）

`tools/scan_m9_engine_vec.py`（300 two_comp 冻结网络，同 CPU Brian2 基线）：

| 变体 | spike 对齐 | 静默比例 |
|---|---|---|
| torch-CPU-f32 `step` | **100.00%**（636/636，gpu_only=0） | 0.8167 |
| torch-CPU-f32 `chunk` | **100.00%** | 0.8167 |
| MPS-f32 `step` | **100.00%** | 0.8167 |
| MPS-f32 `chunk` | **100.00%** | 0.8167 |

- 静默差 **0.00pp**；`step↔chunk` **逐 spike 一致**（max Δt = 0.000ms）；确定性重跑 **100%**。
- **`alignment_gate = PASS`** → 融合内核可用于全规模协议（本批次后续 P4/H6 均用该内核）。

### L24.3 全规模 point 档墙钟（融合后实测；受控活跃度）

| 活跃比例/步 | ms/step（N=139,255） | 30s 单试次 |
|---|---|---|
| 0.10% | 0.62 | 0.104 GPU-h |
| 0.20% | 0.64 | 0.107 GPU-h |
| 0.91% | 1.34 | 0.223 GPU-h |

- two_comp（对齐参考档）：600 隔室 0.326 ms/step → 278,510 隔室外推 **0.065 GPU-h**
  （**B1 基线 1.05 GPU-h → 16× 改善**；B1 融合档自身投影 0.28 GPU-h）。
- **对 B1 两处结论的实测修正（诚实记录，附证据）**：
  1. **B1 的 ①`torch.compile`（0.042 ms/step @N=600，35.8×）在真实逐步依赖循环中不复现**：
     B2 在本机同一 MPS 上实测（`M9_ENGINE_PROF=1` 相位计时，full-scale point 档）
     **compiled 1.31–2.35 ms/step vs eager 0.58–1.19 ms/step**（compile 慢 2.3–2.9×）。
     原因：B1 的 0.042 是**隔离吞吐基准**（同张量重复调用、无逐步数据依赖）——逐步依赖下
     每步 GPU 执行与 Python 发射同量级，编译产物的 guard/dispatch 与 kernel 延迟反而占优。
     → **决策：M9 全规模默认 `use_compile=False`**（同时保留 `compile_mode` 可配，含
     `reduce-overhead`，供后续节点复核）。**方法学纪律：禁止用隔离吞吐基准推断逐步依赖内核墙钟**
     （L16#4 的同类陷阱）。
  2. **融合手段本身有效（B1 结论成立）**：融合后的 point 全规模 0.90 ms/step（P4 实测）
     vs 未融合的 B1 point 投影 3.86 ms/step → 4.3×；`index_add_`/chunk 摊薄等事件交付优化
     是另一半（B1 优化未接入事件交付，B2 已接入）。

---

## L25 — H6 消融（真实 GABA 边 g→0）：**方向反转 + 机制定位**（M9 首要科学检验点）

### L25.1 实测（`tools/validate_p9_gaba_ablation.py`；T=2s，settle=1s，N=3 固定 seed）

| 指标 | 基线（真实 GABA 边） | H6 消融（GABA 边 g→0） | Δ（消融−基线） | §3.5.5 期望方向 |
|---|---|---|---|---|
| 发放率均值 | **0.1265 Hz** | **0.1053 Hz** | **−0.0212** | ↑（**实测相反**） |
| 静默比例 | **0.7474** | **0.7898** | **+0.0425** | ↓（**实测相反**） |
| 群体发放率 | 0.1265 Hz | 0.1053 Hz | −0.0212 | ↑（**实测相反**） |
| 中位率 | 0.0000 | 0.0000 | 0 | — |

- 边集口径：**真实 GABA 连接 3,233,022 条（21.42%）/ 12,755,910 突触（23.4%）**，与
  `data/m9_inhibition_inventory.csv` 逐项一致；消融 = 该边集 `g→0`（**拓扑不变**，只置权重）。
- 三试次统计量完全相同（确定性网络）→ **Δ 不是试次噪声**，是模型层的系统性效应。
- **判定：H6 方向断言 FAIL（反证）**——真实抑制边在本定稿静息态下**没有**起预期的抑制平衡作用。

### L25.2 机制假说 → 检验 → **假说被证伪**（L25.2 修正版）

**假说 H-a（首版）**：`ahp_inc = 2500 mV/s` 把 v 驱动到 GABA 反转电位（−75mV）之下
（平衡点 `v_rest − a/gl = −102 mV`）→ `g_gaba·(E_GABA − v) > 0` 变去极化 → 移除 GABA 边
= 移除一份兴奋电导 → 活动下降。解析上自洽（v=−75 处泄漏恢复力仅 1150 mV/s < 2500）。

**检验（配对 4 臂，`tools/validate_p9_gaba_ablation.py`，T=1s、settle=0.5s、N=1 seed）**：
把 `v_floor` 由 −80 mV 抬到 **−70 mV（严格高于 E_GABA = −75 mV）→ 抑制驱动项在数学上恒 ≤0**：

| v_floor | 基线（真实 GABA）均值/静默 | H6 消融均值/静默 | Δ 率均值 / Δ 静默 |
|---|---|---|---|
| −80 mV（定稿档） | 0.0632 Hz / 0.9368 | 0.0248 Hz / 0.9752 | **−0.0384 / +0.0384** |
| −70 mV（假说修复档） | 0.0634 Hz / 0.9366 | 0.0244 Hz / 0.9756 | **−0.0390 / +0.0390** |

→ **假说 H-a 被证伪**：把 v 强制钳在 E_GABA 之上后，H6 反转方向**完全不变**（Δ 差异 < 2%）。
因此 **GABA 电流的瞬时符号不是该反转的原因**。

**剩余候选机制（未定论，交接下一批次，含检验方案）**：
- **H-b（同步/爆发 + AHP 限幅）**：移除抑制 → 放电更同步/成簇 → 每个 spike 带来的
  AHP（`ahp_inc`，强 rate-limiter）更集中 → 单神经元稳态发放率**下降**。
  *旁证*：GABA-off 臂每步墙钟显著更高（1.91–5.87 vs 1.02–2.33 ms/step）→ 与"更爆发"一致。
  *检验*：两臂都开 `pop_trace`，比较群体 spike 计数的 **Fano 因子 / 爆发指数 / 自相关**；
  再关 AHP（`ahp_inc=0`）复测 H6 方向——若方向随之转正即确证 H-b。
- **H-c（近临界态的集体态切换）**：本档网络对微扰高度敏感（早前实测：`g_ext` 0.04→0.08
  即使 spike 时刻集合完全相同也能把网络从"有活动"翻到"全静默"）→ 消融可能触发集体态切换
  而非系统性方向。*检验*：扫 `w_inh ∈ {0, 0.25, 0.5, 1.0, 2.0}` 看 Δ 是否单调。
- 结论纪律：**在机制定论前，H6 消融不得用于"抑制平衡必需性"断言**；P4 静息态亦须视为
  "未确认真实抑制边起正确作用的态"（登记为测量限制）。

### L25.3 与 M8 R2 的关系（数据侧 vs 模型侧）

- **数据侧已解除**（M9 相较 M8 的最大优势）：真实 GABA 边 21.42% / 23.4% 突触，逐连接
  递质覆盖 100%——**不再需要 hash 分配**（本批次全流程使用真实标注，`data/m9_network_stats.json`
  可复核）。
- **模型侧未解除**：真实抑制边要产生正确的抑制平衡作用，还要求**膜电位不越过 GABA 反转电位**
  （L25.2）。→ M8 R2（"缺 GABA 标注"）在 M9 转化为 **R2'（"有 GABA 标注但驱动符号被适应电流反转"）**，
  两者都不允许把"抑制平衡缺失"当作已解决。

---

## L26 — P4 第一批执行结算（墙钟 / 负载限制 / 预算）

### L26.1 交付协议与结果（`data/m9_p4_resting.csv`，判据带未改动）

- 协议：全规模 point 档 139,255 神经元 / 15,091,983 边；**settle=1s、T=2s（§3.5.4 最短协议）、
  N=3 固定 seed**；无刺激无梯度；背景 Poisson 驱动（0.5 Hz / 2 mV EPSP）。
- 结果（三试次统计量完全相同——确定性网络）：

| 判据 | 实测 | 带 | 判定 |
|---|---|---|---|
| (a) rate_median_hz | **0.0000** | [0.1, 10] | ❌ |
| (b) silent_fraction | **0.7474** | [0.5, 0.9] | ✅ |
| (c) rate_p95_hz | 0.500 | ≤50 | ✅ |
| pop_rate_hz | 0.1265 | ≤20 | ✅ |
| (d) bout_active_fraction / bout_count_per_min | **0.0000 / 0.000** | [0.05,0.9] / [1,600] | ❌ |
| (e) determinism | Spearman **1.000**；**逐位一致 True** | ≥0.99 | ✅ |
| (f) ms_per_step | **0.898**（≤6） | ≤6 | ✅ |

→ **P4 判定 FAIL（4/6）**，按 §8 反证路径记录（见 L21.3）。

### L26.2 🔴 墙钟与**本机共享负载**限制（重要，后续节点必读）

- **独占稳态实测 0.898 ms/step** → 30s 单试次 **0.15 GPU-h**（§0.9 R10 上限 1 GPU-h，6.6× 余量）。
- **但本机在本批次执行窗口内为共享高负载环境**（`uptime` load ≈3.2–3.5；并发重进程来自
  其他会话，含 Chrome 100%/67% CPU）→ 实测退化到 **2.4–12 ms/step（2.7–13×）**：
  - T=30s 档：试次 0 在 **26 分钟**内未完成 100k/600k 步（预期 9 分钟）→ **主动中止**；
  - T=5s 档：试次 0 在 **20 分钟**内未完成（预期 4 分钟）→ **主动中止**；
  - → 按 §3.5.4「最短协议 + 测量限制 + 三态裁决」交付 **T=2s 档**（判据带为率判据、T 无关）。
- **额外发现并修复的性能 bug（长试次退化根因之一）**：`_arange` 初版以**每次 chunk 的活跃边数
  `tot`** 为缓存键 → 缓存条目无界增长（30s 试次 3 万 chunk → GB 级滞留张量）+ 每 chunk 新分配。
  实测使长试次从 0.9 退化到 >15 ms/step、RSS 7.5→14 GB。**修正**：改为**单一可增长缓冲**。
  → 纪律：**任何以"每次调用的动态规模"为键的缓存都是 bug**（本次教训）。
- **复现 30s 档**：`M9_P4_T_MS=30000 PYTHONHASHSEED=0 MPLBACKEND=Agg .venv-m9/bin/python -m
  neural_exploration.tools.validate_p9_resting`（**必须在空闲机器上**；工具已支持环境变量）。
- 落盘：`data/m9_p4_wallclock.csv`（独占/争用墙钟 + 30s 推算 + 中止记录）。

### L26.3 预算（本批次增量）

- CPU ≈0.6 CPU-h（内核融合迭代 + 中止的 T=30s/T=5s 尝试 + 探针）；
  GPU ≈1.1 GPU-h（对齐门重跑 ~0.5h；T=30s 中止尝试 ~0.5h；T=5s 中止尝试 ~0.1h）。
- 累计 M9（B1+B2）：CPU ≈7.6、GPU ≈6.2 —— 远低于 §10.1 上限（§4 全规模装配+静息 ≤8 CPU-h +
  ≤24 GPU-h）。**中止的长试次按"测量限制"登记，不计入协议预算**。

---

## L27 — 主agent接管：H-b/H-c 判别实验（2026-10-10；**H-b 成立**）

### L27.1 E1 决定性结果（`ahp_inc` 消融下的 H6 方向；T=1s、bg 0.5Hz/g_ext 0.04、seed 0）

| 臂 | rate_mean (Hz) | silent_frac | n_spikes |
|---|---|---|---|
| ahp_inc=2500（当前定稿）+ 真实 GABA | 0.00085 | **0.99915** | 119 |
| ahp_inc=2500 + GABA off（H6） | 0.00055 | 0.99945 | 76 |
| **ahp_inc=0 + 真实 GABA** | **5.69177** | 0.55090 | 792,608 |
| **ahp_inc=0 + GABA off（H6）** | **69.15103** | 0.10451 | 9,629,627 |

- **H6 方向对比**：
  - ahp_inc=2500：Δrate = 0.00055 − 0.00085 = **−0.0003（反转 ✗）**
  - ahp_inc=0：Δrate = 69.15 − 5.69 = **+63.46（正确 ✓）**；Δsilent = 0.105 − 0.551 = **−0.446（正确 ✓）**
- **结论：H-b 成立**——**AHP 适应电流是 H6 方向反转的中介**。移除 AHP 后，真实 GABA 边恢复
  正常的抑制平衡作用（消融 → 活动↑/静默↓）。
- **附带关键发现（更直接）**：**当前定稿 `ahp_inc=2500 mV/s` 把网络压到近乎完全静默**
  （rate_mean 0.00085 Hz、静默 **99.9%**、全网络 1s 内仅 119 个 spike）——这是 P4 判据 (a)
  「中位率 0」与 (d)「无 bout」的**直接原因**，而非"缺机制"。ahp_inc=0 时活动回到 5.69 Hz
  （静默 55.1%，**落入 (b) 带 [0.5,0.9] ✓**）。
- **机制陈述（H-b）**：AHP 是发放率负反馈限幅器；在强 AHP 下网络被压至"几乎所有神经元长期
  低于阈值"，此时移除 GABA 边（占突触 23.4% 的抑制性电导）**移走的是唯一把部分神经元拉近
  阈值的电导来源**（GABA 在 v < E_GABA 时的去极化分量 + shunting 改变输入阻抗），故活动反而
  下降 → 符号反转。**AHP 强度与抑制边符号是耦合的**，AHP 未标定好时 H6 不能作为抑制平衡断言。
- **修复方向（主agent交办）**：**下调 `ahp_inc` 到活动落带且 H6 方向正确的区间**（扫描中，
  见 `data/m9_ahp_scan.csv`），然后重跑 P4 + H6 + 标定。这与 L25.2 的"物理修复"（AHP 电导化）
  不冲突但**优先级更高**（更直接、更少改动、可立即验证）。

### L27.2 判别实验设计（E2/E3/E4，作为 H-b 的补充验证）

- E2 `w_inh` 单调扫（0→4.0）：预期在强 AHP 下非单调/反向，在弱 AHP 下单调下降；
- E3 `pop_trace` Fano + 爆发指数（GABA on/off）：H-b 预期 GABA-off 爆发/成簇度上升；
- E4 bias 微扰扫（±2%）：检验 H-c（近临界）——若宏观状态翻转 → 该工作点不适合机制断言。
- 注：主agent首版判别脚本在 E1 后因键名 bug 中断（`rate_mean` vs `rate_mean_hz`），**E1 结果
  完整有效**；E2/E3/E4 由 `ahp_inc` 扫描（`/tmp/m9_ahp_scan.py` → `data/m9_ahp_scan.csv`）替代
  并聚焦修复参数定位。
