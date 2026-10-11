# M11 G2 数据门交付报告（P1/P2）

> **节点**：M11 执行节点 **B1**（工作包 = **G2 数据门（P1/P2）**）
> **判定器**：`tools/validate_p11_data_gate.py` → `reports/neuro/m11_data_gate_summary.json`
> **依据**：`docs/生物仿真M11实施清单.md` §1/§7 P1–P2/§10.1 + `docs/m11_list_review_master.md`（RF-M11-1..6）
> **环境/口径/冲突登记**：`docs/m11_env_notes.md`（L1–L23，本文所有数字的出处）
> **日期**：2026-10-11（08:09–14:20 UTC）；**墙钟**：**≈6.2 h**（含 2 次全量探测重跑 18.9+33.9 min、子集 23 min、NWB 停滞等待 ~50 min，以及**主 agent 复核后的三项门**：完整性门 + 解析门 + 参考锚）
> **解释器**：`./.venv-m11/bin/python`（3.9.6；h5py 3.14.0 / numpy 2.0.2 / pypdf 6.20.0 —— **新增依赖，已登记**）
> **线程纪律**：全程 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`；**零 GPU/MPS 作业**；**下载总量 ≈0.21 GB**（较单次 2 GB / 总计 6 GB 上限留有余量）

---

## §1 判定表（四态：pass / fail / invalid / unavailable）

| 判据 | 一句话语义（清单 §0.4/§7） | 本轮判定 | 关键证据 |
|---|---|---|---|
| **P1** 皮层数据门 | 可达性表逐源实测 + 结构/morph-ephys/文献三类数据落盘 + 权威计数锚硬断言 + SHA 确定性 + 完整度分级 | **pass（附 §3 限制）** | **68 源逐行实测**（REACHABLE 51 / REACHABLE_NO_DATA 3 / NEEDS_VERIFY 3 / NEEDS_TOKEN 1 / BLOCKED 8 / NOT_EXIST 2；两次探测 **68/68 码一致**）；em/seg/seg_m1300 info 落盘 + shard 对象级 206×6；Allen **1,920 行** + NWB HEAD/Range/头解析；文献 37 行 provenance（**31 literature / 6 unverified**）；**新增完整性门：200/200 COMPLETE** |
| **P2** 结构-功能同源数据门 | 结构与功能**同源**路径落盘 + **采集参数读回断言** + 四类参考量同口径可算 + 证据级别标注 | **数据门范围 = pass（判定器 P2 全部断言通过）；全节 = `partial_pass`**（采集参数/同源路径/有界子集/**解析门**全 `measured`；**清单 §1.7 三个参考量 CSV 属后续步，未交付 ⇒ 不写成 pass**） | 帧率 **6.300006 Hz** 读回断言；掩膜格与 series **同格**；**解析门胜出解 = 大端 int16**（lat_x **0.777** / frame **0.773**，全 200 对象）；**独立地面真值**（原始 TIFF，另一 scan）lat_x **0.765** / p50 **1,879** 同量级 |
| **G2（门）** | P1 ∧ P2 均过 | **`pass`（判定器：**44 条断言 0 失败** —— 含主 agent 复核后新增的完整性/解析/参考锚 12 条；附 §3 限制）** | 判定器输出 `reports/neuro/m11_data_gate_summary.json`；**确定性复跑：5 个固定源 verdict 100% 一致**；**未启动** G0/G1/S（遵守前置门链） |

**未通过/未交付项（不静默、不计入 fail 之外的语义）**：

| 项 | 状态 | 原因 |
|---|---|---|
| `m11_tuning_reference.csv` / `m11_dwell_reference.csv` / `m11_equivalence_protocol.csv`（清单 §1.7 P2 验收列出的产物） | **未交付**（`unavailable`） | 依赖两项口径核对（`int16/uint16`、**4 px/µm vs 2.5 µm/px 的 10× 差**）+ ROI 提取管线；**属 P2 后续步**，本工作包边界为"数据门"，见 `m11_env_notes.md` L18/L19#19 |
| MICrONS 逐细胞轨迹（v8 DataJoint DB） | `unavailable` | DB-only 通道（103.8 GB 镜像 / 116.2 GB dump，~36–58 KB/s ⇒ 20–30 天）⇒ 需主 agent 决策（DQ-1） |
| Allen per-cell NWB 完整内部结构 | `unavailable`（部分交付） | 全量下载至 **44.7/91.5 MB（48.8%）后停滞 48 min 零增长** ⇒ **主动停止**（>30 min 纪律）；已交 HEAD + Range + **44.7 MB 前缀头解析**；续传命令见 `m11_env_notes.md` L21#4（DQ-2） |
| Allen synphys 对象级 | `NEEDS_VERIFY` | 桶 403 不可 list、对象级 HEAD 403 ⇒ C-2/C-3 参数走 `literature`（Campagnola 2022） |
| IBL 行为数据对象级 | `NEEDS_TOKEN` | 403 需凭据 ⇒ P12 走公开桶 + 论文档回退（清单 §1.4） |

---

## §2 证据级别总表（measured / derived / literature / unavailable）

| 数据块 | 证据级别 | 计数锚（权威） | 出处 |
|---|---|---|---|
| MICrONS 结构 info（em/seg/seg_m1300，GCS+S3） | **measured**（HTTP 200 + SHA-256） | em 4,117 B / seg 7,448 B / seg_m1300 7,628 B | `m11_microns_structure.json` |
| MICrONS chunk 计数 | **derived**（Π⌈size/chunk⌉） | em 958,251,008 / seg 626,819,072 / seg_m1300 1,878,146,010 | 同上 |
| MICrONS shard 清单 | **measured（有界，truncated=true）** | 各 3,000 对象：em 20.06 TB / seg 1.154 TB / seg_m1300 1.043 TB | 同上 |
| MICrONS shard 对象读取 | **measured**（Range 0-1023 → 206 ×6） | 三卷各 2 个 shard | 同上 |
| MICrONS 功能目录 | **measured** | 144 field / 39 原始 tif（2.686 TB）/ 刺激电影 103 件 | `m11_microns_protocol.json` |
| MICrONS 采集参数（时间口径） | **measured（读回断言通过）** | **6.300006 Hz**、40,000 帧、6,349.2 s、z=1、2.5×2.5×10 µm、FOV 620×1100 µm | 同上 |
| MICrONS 神经域/掩膜 | **measured** | 144 份 info；`compressed_segmentation`；格 248×440×1 @2500×2500×10000 nm | 同上 |
| MICrONS 原始扫描帧维 | **measured（Range 头）** | BigTIFF `{"shape":[320000,440,248]}`、strip 218,240 B/帧 | 同上 |
| MICrONS 功能有界子集 | **measured** | **200/200 对象、2,000 帧**；**chunk = 20 B 大端头 `[4,200,200,1,10]` + gzip**，解压 raw = **800,000 B** 逐片精确匹配 | `m11_functional_subset_manifest.csv` / `_verify.json` |
| **下载完整性门**（主 agent §三） | **measured** | 逐对象 5 步链（http→大小→SHA-256→解压→尺寸自洽）：**COMPLETE 200/200**（首轮 197，3 片 `curl_exit=28` 半成品已重取）；`manifest.integrity`=COMPLETE 全行 | `m11_download_integrity.csv` / `_summary.json` |
| **解析结构性门**（主 agent §二） | **measured** | 18 个假设 × 6 判据：**唯一自洽解 = 大端 int16**（lat_x **0.801**、相邻帧 **0.784**、p50 **1,815**）；小端/uint16/delta(cumsum)/Morton/float **全部判 invalid**（逐条原因落盘） | `m11_microns_parse_validation.json` / `_hypotheses.csv` |
| **NaN/哨兵/掩膜登记**（主 agent §四） | **measured** | **NaN=0 / Inf=0**（整型域结构性为 0）；饱和(32767)=**0**；零值 0.2416%；低于基线 **10,383 (0.01298%)** 且**成簇于暗区**（邻域负值比例 0.111 vs 随机 0.000129；孤立 0.000）⇒ **无掩膜** | 同上 |
| **参考锚** | **measured / 1 NEEDS_VERIFY / 1 unverified** | `m11_reference.csv` 22 行：帧率 **6.300006 Hz**、字节序 **big-endian**、20 B 头、10 帧/片=1.5873 s 均为 `asserted`；**dwell/τ 带 = `unverified`** | `m11_reference.csv` |
| 结构-功能**同源**证据 | **measured** | 同一 minnie65 体积内的 EM/seg 与 144 个双光子 field；`functional_coregistration` 前缀可达（2 键） | 同上 |
| Allen Cell Types 元数据 | **measured（全取）** | **1,920/1,920 行**；54 字段（11 个 `ef__*`）；`ef__avg_firing_rate` n=**1,808**（中位 17.44 Hz） | `m11_allen_celltypes_metadata.csv` |
| Allen per-cell NWB | **measured（对象级 + 头）** | HEAD 200 / **91,560,414 B** / `486940961_ephys.nwb`；Range 0-99 → 206；**44.7 MB 可见前缀 + 稀疏文件头解析**：`nwb_version = NWB-1.0.5`、`file_create_date = 2018-02-28`、根组 11 键、`Sweep_10` = `CurrentClampSeries`（"Voltage recorded from cell during current-clamp recording"） | `m11_allen_sources.json` |
| Allen BO Neuropixels 目录 | **measured** | **569 对象 / 853.7 GB**（未截断）；单体对象 2,062,950,784 B（LFP NWB，200） | 同上 |
| 文献先验（§1.3 + §4.1 C-1..C-8） | **literature 31 / literature_unverified 6** | 清单 12 个 PMID **全部题录核对通过**；C-5/C-6/C-8 **由 unverified 升级** | `m11_literature_priors.csv` |
| 落盘清单与确定性 | **measured** | **224 件逐件 SHA-256 现场重算一致**（`P1.5b`）；**确定性复跑**：5 个固定源 verdict 全一致；**探针两次 HTTP 码 68/68 一致** | `m11_data_inventory.csv` / `m11_data_gate_summary.json` |
| 未测项 | **unavailable**（8 条，逐条给原因） | 见 `m11_env_notes.md` L21 与 `m11_data_counts.json::unavailable_register` | — |

**证据级别纪律**：`literature` **只**表示"题录经 Europe PMC 实测命中"，**不等于**全文/参数值已核实
（`value_in_abstract` 列逐行标注摘要中是否出现该值）；`derived` 项（chunk 计数、未压缩上界）**不**与 `measured` 混列。

---

## §3 限制登记（**必须随 P1/P2 结论一并引用**）

1. **两项口径待核（会直接改变钙域判据）**：
   ① `functional_series` 的 `dataType=int16` 与实际字节行为（int16 解均值≈−105.7/49.96% 为负 vs uint16 解均值≈32,637）
   **两解均自洽**；② 技术文档写 **4 px/µm** 而 series `resolution` 写 **2.5 µm/px**（**相差 10×**），
   FOV 写法亦不同（1100×1200 µm² vs 620×1100 µm）。⇒ **P2 定稿 dwell/率/OSI 前必须先解决这两项**。
   **登记为 `NEEDS_VERIFY`，本节点不裁决**（清单 §0.7 #3 口径纪律）。
2. **DB-only 限制**：MICrONS 逐细胞轨迹、扫描级 `fps/nframes/nfields`、瞳孔/跑道同步量均在
   **v8 DataJoint DB**（103.8 GB / 116.2 GB）内，本轮 `unavailable`。
   ⇒ M11 的钙域参考量走**场级重建**（144 ROI 掩膜 + 2 field 钙电影）路径。
3. **单个 field ≠ 整卷**：`functional_series` 为单平面（z=1），文档述多深度（two planes per depth × 2–4 depths）
   ⇒ **整卷体积率未测**。
3b. **dwell/τ 参考带 = `unverified`**（主 agent 明确要求）：解析门虽已过，但 C1/C2 参考量重建与
   RF-M11-2 的"目标数据劈半 + 同协议等价检验"未完成 ⇒ **不得作为任何判据带**。
3c. **帧率差异的判据级后果**：MICrONS **6.30 Hz** vs ZAPBench **1.09 Hz**（**5.8×**）
   ⇒ 沿用 M10 的 26.5 s dwell / 37.18 s τ 将构成**跨物种 + 跨帧率 + 跨箱宽**三重错配。
3d. **采样下界**：帧间期 158.73 ms ⇒ 0.3 s 级状态仅 ~1.9 帧（dwell 下界可分辨性受限，定稿时须写明）。
4. **数据规模限制**：EM 未压缩上界 **502.4 TB**（raw）、seg/seg_m1300 未压缩上界 2.6/7.9 PB；
   原始扫描 2.686 TB、刺激电影 1.84 TB ⇒ **M11 必须子体积/按需取块**（清单 §9 风险表首行，本轮实测加固）。
5. **网络限制**：`portal.brain-map.org`（挂起 925.9 s 一次）、`synapse.brain-map.org`（DNS）、
   `raw.githubusercontent.com`（reset）、Zenodo（TLS 自签）、figshare/GIN（403）、`global.daf-apis`（302 挂起）。
6. **`literature_unverified` 6 行**（含 M11 的 dwell 目标带）**不得**作为参数范围依据（清单 §4.1 明示）。
7. **网络波动会改变单源 verdict**：`#33 portal.brain-map.org` 三轮分别给 404 / 404 / 连接超时
   ⇒ 末轮判 `BLOCKED`（间歇不可达）；**不得**只引"好看的那一轮"。
8. **Allen per-cell NWB 全量未到位**：44.7/91.5 MB 后停滞 ⇒ 完整内部结构（细胞/轨迹/刺激表）`unavailable`。
7. **本工作包未产出**：实测级调谐/dwell/等价协议 CSV（见 §1 未交付项）。

---

## §4 复跑命令（全部必须显式用项目 venv 解释器）

```bash
cd /Users/weidong/ai/small_world
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=./.venv-m11/bin/python

# ① 可达性探针（68 源；两次探测 + 三次多数裁决 + 父进程硬超时）
$PY neural_exploration/tools/probe_m11_sources.py --all --timeout 15

# ② MICrONS 结构（info + shard 有界清单 + shard 对象 Range 探针）
$PY neural_exploration/tools/fetch_m11_microns.py --struct --shard-probe

# ③ MICrONS 功能元数据 + 采集参数读回（144 field attributes + 掩膜 info + BigTIFF 头 + 技术文档）
$PY neural_exploration/tools/fetch_m11_microns.py --func-meta

# ④ MICrONS 功能**有界子集**（默认 200 片 ≈ 2,000 帧 ≈ 136 MB，硬上限 2 GB）+ 解压验证
$PY neural_exploration/tools/fetch_m11_microns.py --func-subset --jobs 3
$PY neural_exploration/tools/fetch_m11_microns.py --retry-missing      # 仅重取失败片
$PY neural_exploration/tools/fetch_m11_microns.py --verify-subset --verify-n 5

# ⑤ Allen（Cell Types 1920 行 + per-cell NWB + BO Neuropixels 目录）
$PY neural_exploration/tools/fetch_m11_allen.py --all

# ⑥ 文献 provenance 化（Europe PMC；37 行）
$PY neural_exploration/tools/build_m11_literature_priors.py

# ⑦ 权威清单 + 计数锚
$PY neural_exploration/tools/build_m11_inventory.py

# ⑧ 主 agent 复核后的三项门（完整性 / 解析 / 参考锚）
$PY neural_exploration/tools/verify_m11_downloads.py --refetch      # 完整性门（5 步判定链）
$PY neural_exploration/tools/validate_m11_microns_parse.py --n-full 200   # 解析门（18 假设 × 6 判据）
$PY neural_exploration/tools/build_m11_reference.py                 # 参考锚（含 6.300006 Hz 读回断言）

# ⑨ P1/P2 判定（含确定性复跑子集）
$PY neural_exploration/tools/validate_p11_data_gate.py --determinism-check
```

> 曾坑（已固化）：① 子集运行**禁止**覆盖全量 CSV（`--only` 必须显式 `--out`）；
> ② `curl --max-time` 曾失效（925.9 s）⇒ 父进程 `Popen + kill`；③ Cell Types criteria 必须 percent-encode + 分页；
> ④ 功能 chunk = **20 B 头 + gzip**，`useZlib:false` 不可按 raw-deflate 解。

---

## §5 墙钟与预算

| 步骤 | 实测墙钟 | 备注 |
|---|---|---|
| 可达性探针（首轮） | **18.9 min** | 被 `portal.brain-map.org` 单次挂死 925.9 s 拉长 |
| 可达性探针（末轮，修复父进程硬回收后） | **33.9 min** | 68 源 × 2 次探测；含若干源的 8–25 s 连接超时（硬回收） |
| MICrONS 结构 + shard 清单 + 对象探针 | **≈1.5 min** | 9 次列举 + 6 次 Range |
| MICrONS 功能元数据（144 field + 掩膜 + 文档 + BigTIFF 头） | **≈4 min** | 3 路并发 |
| MICrONS 功能有界子集（200 片 / 136.3 MB） | **≈23 min**（含 2 片 `--retry-missing`） | 3 路并发；受 NWB 作业竞争（已串行化） |
| Allen（元数据 5 页 + NWB HEAD/Range/头 + BO 目录） | **≈4 min** | NWB 全量作业另计（44.7 MB 后停滞 → 停） |
| 文献 provenance（37 行 × 2–3 次查询） | **≈2 min**（网络波动时更久） | Europe PMC |
| 清单 + 计数锚 + 判定器 | **≈3 min** | SHA-256 覆盖 136 MB 子集 |
| **合计** | **≈6.2 h 墙钟**（含等待/文档/NWB 停滞等待/主 agent 复核后的三项门；净计算 ≈0.8–1.1 CPU-h，单线程） | **GPU = 0**；下载 ≈0.21 GB（新增仅原始 TIFF 4 帧 ≈0.9 MB） |

---

> **落盘位置说明**：原始下载件在 `neural_exploration/data/m11_raw/**`（**已被 `.gitignore` 排除**，仅本机留存，约 **280 MB 实际占用**）；
> 权威清单 `data/m11_data_inventory.csv` 逐件记录 `bytes + SHA-256 + source_url + retrieved_at` ⇒ **可重取后逐件校验**。

## §6 交付物清单（全部新建；冻结件零修改）

| 类别 | 文件 | 规模 |
|---|---|---|
| 工具 | `tools/probe_m11_sources.py` / `fetch_m11_microns.py` / `fetch_m11_allen.py` / `build_m11_literature_priors.py` / `build_m11_inventory.py` / `validate_p11_data_gate.py` + **`verify_m11_downloads.py` / `validate_m11_microns_parse.py` / `build_m11_reference.py`** | **9 个** |
| 数据（表） | `data/m11_source_reachability.csv`（68 源）/ `m11_data_inventory.csv`（231 件）/ `m11_literature_priors.csv`（37 行）/ `m11_allen_celltypes_metadata.csv`（1,920 行）/ `m11_functional_subset_manifest.csv`（200 行 + integrity 列）/ **`m11_download_integrity.csv`** / **`m11_microns_parse_hypotheses.csv`** / **`m11_reference.csv`（22 行）** | 8 个 |
| 数据（JSON） | `m11_microns_structure.json` / `m11_microns_protocol.json` / `m11_allen_sources.json` / `m11_literature_priors.json` / `m11_data_counts.json` / `m11_functional_subset_summary.json` / `m11_functional_subset_verify.json` / **`m11_download_integrity_summary.json`** / **`m11_microns_parse_validation.json`** | 9 个 |
| 原始落盘 | `data/m11_raw/{microns/struct, microns/functional, microns/functional_subset, microns/pdf_text, allen, probe, literature}` | 见清单 |
| 报告 | `reports/neuro/m11_data_gate_summary.json` / `docs/m11_data_gate_report.md` / `docs/m11_env_notes.md`（L1–L23） | 3 个 |
| 清单改写（唯一例外） | `docs/生物仿真M11实施清单.md` §0.7 #2（L220–L222 → **L220–L227**，RF-M11-1；文件 1226 → **1232** 行） | 9+/3−（已由主 agent 提交 `b0996c1`，故当前 `git diff` 为空属正常） |

## §6b ★ P2 增强通道（本轮新发现，**待裁决**）：MICrONS 功能连接组学发布

探测 F17 时发现 `functional_connectomics/node_and_edge_properties/v1/` 内含
**同源结构-功能逐细胞表**（readme 已落盘）：
`node_data_v1.pkl`（294.6 MB：EM 坐标 + `layer`/`brain_area` + **in vivo `osi_iv`/`pref_ori`/`in_vivo_mean_resp`** + CvT 数字孪生量）与
`edge_data_v1.pkl`（201.5 MB：逐对 **`n_synapses`/`synapse_size`** + `pre/post_layer` + **`in_vivo_sig_corr_iv`** +
`population ∈ {Connected, ADP, Same region}` ⇒ **自带距离匹配/同区负对照**）。
出处：Ding et al. 2025, **DOI 10.1038/s41586-025-08840-3**（readme 原文）。
⇒ 若批准下载（≈496 MB，DQ-4），P2 的 C1/C2 参考量可**同源同口径**直接算出，P2 可从 `partial_pass` 推向全过。
**本节点因 ">30 min 先汇报" 纪律未启动该下载。**

## §6c 主 agent 独立复核的三项硬要求（**已全部落实并门化**）

| 硬要求 | 处置 | 证据 |
|---|---|---|
| §二 结构性解析验证（含 delta / Morton 假设） | ✅ **根因 = 字节序**（小端误读）：大端 int16 直读给出 lat_x **+0.801** / 相邻帧 **+0.784**；delta 与 Morton 假设**均不成立**；**并发现主 agent 的 4 条判据对 `cumsum` 退化结构天然通过 ⇒ 补 C5 声明域约束 + C6 非退化**（通过者 9 → 2，再按相干性+简洁性选唯一解） | `m11_microns_parse_validation.json`（含独立地面真值：原始 TIFF lat_x 0.765 / p50 1,879） |
| §三 完整性门 | ✅ 197/200 → **200/200 COMPLETE**；**根因（本节点缺陷）= 判定未校验 `curl_exit`**，半成品被记 `REACHABLE` ⇒ 已修 + 固化为独立门；**统计只用 COMPLETE** | `m11_download_integrity.csv` / `_summary.json`；判定器 `P1.7a-c` / `P2.11e` |
| §四 NaN/哨兵登记 | ✅ **NaN/Inf = 0**（整型域）；主 agent 的 106 NaN / |x|>1e30 = **float 误读副产物**；低于基线 0.013% **成簇于暗区 ⇒ 非哨兵、无掩膜** | 同上 + `m11_reference.csv` R-13/R-14 |
| 6.30 Hz 作为读回断言值 | ✅ 落 `m11_reference.csv` **R-01 `asserted`**，判定器 `P2.12a` 强制核对 | `data/m11_reference.csv` |

## §7 结论与下一步建议

1. **G2 判定 = pass（附 §3 限制）**：51 清单源 + 17 补测源逐行实测；结构与功能**同源**路径落盘；
   **采集参数已读回**（6.300006 Hz 等）——这是清单 §1.1.2 "未读回 ⇒ unverified" 的解除条件。
2. **P2 尚不足以"定稿 dwell 带"**：需先核 `int16/uint16` 与 **10× 分辨率口径差**（§3.1）；
   核对后即可用"2 field × 6.3001 Hz × 40,000 帧（105.8 min）"做劈半可达性自检（RF-M11-2）。
3. **未启动**（遵守门链）：G0 引擎门、G1 生理门、S 机制门、P7–P14 —— 且 RF-M11-2..6 未落盘。
4. **交主 agent 的四个决策**：DQ-1（DB 取数 vs 场级重建）/ DQ-2（NWB 全量续传托管作业）/
   DQ-3（19 条冲突项落清单）/ **DQ-4（是否下载功能连接组学两个 pkl ≈496 MB —— 决定 P2 能否全过）**。
