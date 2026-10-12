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
| **参考锚** | **measured / 1 NEEDS_VERIFY / 1 unverified** | `m11_reference.csv` **26 行**：帧率 **6.300006 Hz**、字节序 **big-endian**、20 B 头、10 帧/片=1.5873 s、**Allen ephys 采样率 30 kHz / LFP 2.5 kHz** 均为 `asserted`；**dwell/τ 带 = `unverified`**（R-21） | `m11_reference.csv` |
| **体素解码证据**（RF-M11-A 反证） | **measured** | `demo_m11_microns_decode.py` 一条命令复核：LE lat_x **−0.001** / 帧轴 lag1 **0.000** / p50 **0** ⇔ **BE lat_x +0.814** / 帧轴 lag1 **+0.803** / p50 **1953**；帧轴 lag1..6 = 0.803→0.709 平滑衰减 | `m11_decode_proof.json` + `m11_microns_decode_proof.png` |
| **活动回退源**（RF-M11-A §二.3.a） | **measured**（自描述 CSV） | Allen BO ecephys：`probes.csv` **完整** ⇒ **spike 30,000.3 Hz / LFP 2,500.03 Hz**（332 probe，均值 29,909.6 / 2,492.5）；`sessions.csv` 完整（58 session）；`units.csv` 全量 51.4 MB **未取**（DQ-6），1 MB 前缀读回 27 列 schema | `m11_allen_activity_fallback.json` / `m11_allen_activity_objects.csv` |
| 结构-功能**同源**证据 | **measured** | 同一 minnie65 体积内的 EM/seg 与 144 个双光子 field；`functional_coregistration` 前缀可达（2 键） | 同上 |
| Allen Cell Types 元数据 | **measured（全取）** | **1,920/1,920 行**；54 字段（11 个 `ef__*`）；`ef__avg_firing_rate` n=**1,808**（中位 **17.4130 Hz**；min 1.984915 / max 261.210274） | `m11_allen_celltypes_metadata.csv` |
| Allen per-cell NWB | **measured（对象级 + 头）** | HEAD 200 / **91,560,414 B** / `486940961_ephys.nwb`；Range 0-99 → 206；**44.7 MB 可见前缀 + 稀疏文件头解析**：`nwb_version = NWB-1.0.5`、`file_create_date = 2018-02-28`、根组 11 键、`Sweep_10` = `CurrentClampSeries`（"Voltage recorded from cell during current-clamp recording"） | `m11_allen_sources.json` |
| Allen BO Neuropixels 目录 | **measured** | **569 对象 / 853.7 GB**（未截断）；单体对象 2,062,950,784 B（LFP NWB，200） | 同上 |
| 文献先验（§1.3 + §4.1 C-1..C-8） | **literature 31 / literature_unverified 6** | 清单 12 个 PMID **全部题录核对通过**；C-5/C-6/C-8 **由 unverified 升级** | `m11_literature_priors.csv` |
| 落盘清单与确定性 | **measured** | **224 件逐件 SHA-256 现场重算一致**（`P1.5b`）；**确定性复跑**：5 个固定源 verdict 全一致；**探针两次 HTTP 码 68/68 一致** | `m11_data_inventory.csv` / `m11_data_gate_summary.json` |
| 未测项 | **unavailable**（8 条，逐条给原因） | 见 `m11_env_notes.md` L21 与 `m11_data_counts.json::unavailable_register` | — |

**证据级别纪律**：`literature` **只**表示"题录经 Europe PMC 实测命中"，**不等于**全文/参数值已核实
（`value_in_abstract` 列逐行标注摘要中是否出现该值）；`derived` 项（chunk 计数、未压缩上界）**不**与 `measured` 混列。

---

## §3 限制登记（**必须随 P1/P2 结论一并引用**）

### §3.0 P2 功能侧的**两条核心限制**（主 agent 要求写入正文，逐条给判定词）

| # | 限制 | 判定词 | 后果 |
|---|---|---|---|
| **F-1** | **MICrONS 逐细胞 ΔF/F 与 ROI 轨迹只存在于 v8 DataJoint MySQL**（103,798,314,496 B docker tar / 116,199,864,342 B SQL dump；无 REST/API 通道，CAVE 表 API 实测 404/503） | **`unavailable`**（**不是** 0、**不是** 未命中） | **逐细胞口径的钙域参考量（率分布 / 相关 / dwell / OSI）本轮不可用**；替代路径 = 144 field ROI 掩膜自建 ROI 提取（`field_seg_masks` 已实测可达且与 series 同格） |
| **F-2** | **144 个 field 中只有 2 个有 `functional_series` 体素序列**（`session4_scan7_field4` / `session5_scan3_field1`；6.300006 Hz / 40,000 时间点 / 6,349.2 s / FOV 620×1100 µm / voxel 2.5×2.5×10 µm），其余 142 个 field 仅有网格属性与掩膜 info | **`partial`**（元数据 measured；体素覆盖 2/144） | ⇒ **"全脑活动金标准"在本里程碑**退化为"**2 个 field 的体素序列（可解码）+ 144 个 field 的元数据/掩膜**"；**不**声称"全 FOV/全脑活动统计" |
| F-3 | v8 DB 内还有**全部 field 的扫描级 `fps/nframes/nfields`、瞳孔与跑道同步量** | `unavailable` | 整卷体积率与行为同步口径未测（清单 §1.7 回退） |

> 说明（避免误读）：**体素序列本身是可解码的**（大端 int16；见 §2「体素解码证据」），
> 因此 F-2 的性质是"**只有 2 个 field 可算**"，而非"数据不可用"；**逐细胞**层面的缺口是 F-1。

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
9. **Allen `units.csv`（51.4 MB）/ `channels.csv`（6.6 MB）全量未取**：网络退化到 **16 KB/s** ⇒ 预计 >30 min
   ⇒ 按纪律未自启（**DQ-6**）；已用 1 MB Range 前缀读回 schema。
10. **Allen LFP NWB 全量 2,062,950,784 B > 2 GB** 单文件上限 ⇒ 合规地不下载，仅 Range 头解析。
11. **网络吞吐剧烈退化**：本轮实测 S3 `bossdb-open-data` **194–349 KB/s → 16 KB/s**（同一时段）
   ⇒ 一切取数决策必须按**当时的实测吞吐**做，不得沿用早前数字（本轮据此放弃了 ≥50 MB 的下载）。
7. **本工作包未产出**：实测级调谐/dwell/等价协议 CSV（见 §1 未交付项）。

---

## §3.5 优先级①（Allen NWB 抽取与读回）的实测状态与吞吐

| 通道 | 实测吞吐 | 结论 |
|---|---|---|
| `api.brain-map.org`（per-cell NWB 91.56 MB） | **1.6–16 KB/s** | 剩余 46.9 MB 需 **50 min – 8 h** ⇒ **不可自取**；已到 **44,695,552 / 91,560,414 B**（**INCOMPLETE**，可 `curl -C -` 续传） |
| DANDI 000021（最小 asset **71,781,159 B**，`sub-703279277_..._ecephys.nwb`，URL 已登记） | **7,237 B/s** | 71.8 MB ≈ **2.8 h** ⇒ **不可自取**（DQ-7 托管候选） |
| `bossdb-open-data`（MICrONS） | 194–349 KB/s → **16 KB/s** | 功能子集已取满；后续 ≥50 MB 未自启 |
| Allen BO `ecephys-cache`（表） | 完整件均已取 | `probes.csv/sessions.csv` **完整**；`units/channels` 前缀 |

**NWB 读回到底拿到了什么（据 44.7 MB **真实**前缀；注意 `head_sparse.nwb` 的 91,560,414 B 是**稀疏表观大小**，仅前 44,695,552 B 为真实数据）**：
- ✅ 可读：`nwb_version = NWB-1.0.5`、`file_create_date`、根组 **11 键**（`acquisition/analysis/epochs/general/processing/stimulus/…`）、
  **`acquisition/timeseries/Sweep_10`**：`ancestry=[TimeSeries, PatchClampSeries, **CurrentClampSeries**]`、
  `help="Voltage recorded from cell during current-clamp recording"` ⇒ **traces 结构成立**；
- ❌ 不可读（落在前缀之外的洞）：`/general/extracellular_ephys`（电极表）、`/acquisition` 的**采样率**、`session_id/subject`
  ⇒ 判定词 **`unavailable`（原因：前缀外字节未取；网络 1.6–16 KB/s）**；
- ✅ **但§四.1 的实质要求（采样率 + units 结构）已由自描述表满足**：`probes.csv`（**完整**）⇒
  **spike 30,000.3 Hz / LFP 2,500.03 Hz**（332 probe、均值 29,909.6 / 2,492.5、325 有 LFP）；
  `units.csv` 27 列 schema（`firing_rate/snr/isi_violations/presence_ratio/quality/…`）；
  units 行数**仅派生外推** ≈ **136,720**（`derived`：1 MB 前缀 2,787 行 × 51,439,189/1,048,576 ⇒ **不得作计数带**，精确值需 DQ-6）。

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

# ⑦ 主 agent 复核后的门（完整性 / 解析 / 空间口径 / 参考锚）——**必须在 ⑧ 之前**
$PY neural_exploration/tools/verify_m11_downloads.py --refetch            # 完整性门（5 步判定链）
$PY neural_exploration/tools/validate_m11_microns_parse.py --n-full 200   # 解析门（18 假设 × 6 判据）
$PY neural_exploration/tools/verify_m11_grid_alignment.py                 # 空间口径 10× 与同格性（实像素）
$PY neural_exploration/tools/build_m11_reference.py                       # 参考锚 R-01..R-29

# ⑧ 权威清单 + 计数锚（**必须最后重建**：它对 ⑦ 的产物逐件重算 SHA ⇒ 顺序错会触发 P1.5b）
$PY neural_exploration/tools/build_m11_inventory.py

# ⑨ P1/P2 判定（含确定性复跑子集）
$PY neural_exploration/tools/validate_p11_data_gate.py --determinism-check
```

> 曾坑（已固化）：⓪ **产物顺序纪律**：`build_m11_inventory.py` 对其余产物逐件重算 SHA-256，
> 因此**必须最后运行**；本轮曾把 reference 生成放在 inventory 之后 ⇒ 被自家断言 `P1.5b` 抓到（已修）；
> ① 子集运行**禁止**覆盖全量 CSV（`--only` 必须显式 `--out`）；
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

## §6b-2 RF-M11-A（体素格式裁定）的处置：**接受回退，不接受 `format_unverified`**

| 项 | 处置 |
|---|---|
| "不再攻关格式" | ✅ 服从（破解已在上一轮完成，成本已沉没；本轮只付"一条命令复核"的成本） |
| "功能通道回退到自描述源" | ✅ **已落地**：Allen BO Neuropixels 自描述表（采样率 **30 kHz / 2.5 kHz** 读回）+ NWB(HDF5) 前缀；见 §2「活动回退源」 |
| 登记体素序列 `format_unverified` | ❌ **不接受**：用**主 agent 自己的检验项**做 LE/BE 对照，BE 直读给出帧轴 lag1..6 = **0.803/0.783/0.761/0.743/0.725/0.709**、lat_x **0.814**、p50 **1953**；并有**独立地面真值**（原始未压缩 TIFF lat_x **0.767**、p50 **1,879**）。主 agent 三轮检验均为 `<i2` 小端读法 —— 其"无结构"是**字节序假象**。⇒ 登记 **`decoded(big-endian int16)`** 并保留裁定留痕（`m11_reference.csv` R-26） |
| 风险说明 | 若判死功能体素序列，M11 将失去"**同源**结构-功能"这一相对 M10/M9 的唯一方法学升级（H4/P14 不可检验）⇒ 故建议保留 `decoded` 状态 |
| **dwell/τ 参考带** | 仍 **`unverified`**（与格式无关：取决于 C1/C2 参考量重建 + RF-M11-2 劈半自检）——与 RF-M11-A §三.3 一致 |

## §6c 主 agent 独立复核的三项硬要求（**已全部落实并门化**）

| 硬要求 | 处置 | 证据 |
|---|---|---|
| §二 结构性解析验证（含 delta / Morton 假设） | ✅ **根因 = 字节序**（小端误读）：大端 int16 直读给出 lat_x **+0.801** / 相邻帧 **+0.784**；delta 与 Morton 假设**均不成立**；**并发现主 agent 的 4 条判据对 `cumsum` 退化结构天然通过 ⇒ 补 C5 声明域约束 + C6 非退化**（通过者 9 → 2，再按相干性+简洁性选唯一解） | `m11_microns_parse_validation.json`（含独立地面真值：原始 TIFF lat_x 0.765 / p50 1,879） |
| §三 完整性门 | ✅ 197/200 → **200/200 COMPLETE**；**根因（本节点缺陷）= 判定未校验 `curl_exit`**，半成品被记 `REACHABLE` ⇒ 已修 + 固化为独立门；**统计只用 COMPLETE** | `m11_download_integrity.csv` / `_summary.json`；判定器 `P1.7a-c` / `P2.11e` |
| §四 NaN/哨兵登记 | ✅ **NaN/Inf = 0**（整型域）；主 agent 的 106 NaN / |x|>1e30 = **float 误读副产物**；低于基线 0.013% **成簇于暗区 ⇒ 非哨兵、无掩膜** | 同上 + `m11_reference.csv` R-13/R-14 |
| 6.30 Hz 作为读回断言值 | ✅ 落 `m11_reference.csv` **R-01 `asserted`**，判定器 `P2.12a` 强制核对 | `data/m11_reference.csv` |

## §6d P2 判定**草案**（**待主 agent 确认，未定稿**）

| 通道 | 判定草案 | 依据 |
|---|---|---|
| **结构** | **`measured` ✅** | em/seg/seg_m1300 自描述 info（11 scales、8×8×40 nm、EM 体积 **1.2861 mm³**、bbox 1.70×1.44×0.52 mm、chunk 128×128×32 raw）+ shard 有界实测 + 对象级 Range 206 |
| **功能（体素序列）** | **`decoded(big-endian int16)`**（**非** `format_unverified`） | 结构门 pass：lat_x **0.814** / 帧轴 lag1 **0.803** / p50 **1953**；独立地面真值（原始 TIFF lat_x 0.767）；**B3b 独立复现同一解码** |
| **功能（覆盖度）** | **`partial`**：仅 **2/144 field** 有体素序列 | 见 §3.0 F-2 |
| **功能（逐细胞）** | **`unavailable`**：ΔF/F/ROI 轨迹仅存于 v8 DataJoint DB | 见 §3.0 F-1 |
| **活动金标准（P14 口径）** | 现实可为：**"2 个 field 的钙域统计 + 144 个 field 的元数据/掩膜 + Allen BO 电生理带（30 kHz/2.5 kHz + units schema）"**；**不**声称全脑/全 FOV 统计 | 上述三者合并 |

### M12 交接缺口清单（**必含主 agent 指定的两条**）

| # | 缺口 | 判定词 | 交接动作 |
|---|---|---|---|
| ~~**G-1**（R8 修订版，**已作废，见 §6m 的 G-1′**）~~ | ~~时间序列级非同源 ❌（体素序列 `format_unverified`）~~ **⬅ 本行按"只增不改"保留原文，其"体素 `format_unverified`"表述已于 R9 被推翻**；正确表述见 **§6m G-1′**（同源在**时间序列级成立**、在**逐细胞级与跨 field 级**的现状见该节） | **作废** | 以 §6m G-1′ 为准 |
| **G-1**（R8 修订版） | **同源是分级的**：结构同源 ✅ + **属性级同源 ✅**（functional_connectomics 逐细胞/逐对表，DOI 可核）+ **时间序列级非同源 ❌**（体素序列 `format_unverified`；逐细胞 ΔF/F 仅 v8 DB `unavailable`；dwell/τ 走 Allen 2p 代理；144 field 仅 2 个有 series） | `measured`（结构/属性）+ `derived/cross_source`（时间序列）+ `unavailable`（逐细胞轨迹） | M12 若需**同源时间序列**，须取得官方体素规范或 v8 DB 工具链；属性级同源可直接用于 P7/P8/P14 |
| **G-2** | **四类参考量口径**——**口径层已核实**：时间 **6.300006 Hz**、空间 **2.5 µm/px**（与 summaries 620×1100 µm 实像素同格、位移 (0,0)）、raw↔处理后 **10×** 层级差（R-27）；**属性级同源表口径** = node/edge 表（26/19 列，逐细胞/逐对） | 口径 = **`measured`**；参考量 = **部分可算**（C1 调谐、C2 的成对相关可直接由属性表算；C3 dwell 需 Allen 2p） | 重建时**必须**写明层级口径 + 跨对象 FOV 交叉核对（`P2.11f/g`） |
| G-3 | v8 DataJoint DB 未取 ⇒ 逐细胞 ΔF/F / ROI 轨迹 / 扫描级 fps / 瞳孔-跑道同步量 | `unavailable` | 需 20–30 天量级带宽 ⇒ 需外部条件变化 |
| G-4 | `units.csv` 51.4 MB / `channels.csv` 6.6 MB 全量未取（精确 units 计数仅有 `derived` 外推 ≈136,720） | `unavailable`（DQ-6） | 托管取数后替换为 `measured` 计数 |
| G-5 | Allen per-cell NWB 头：`/general/extracellular_ephys`、`/acquisition` 采样率、`/units` spike times **不可读**（本地仅 44,695,552 B 真实前缀，其余为稀疏洞；网络 1.6–16 KB/s、DANDI 7.2 KB/s） | `unavailable` | 已由自描述表替代（30 kHz / 2.5 kHz + units schema）；如需 NWB 逐字对齐再续传 |
| G-6 | 体素序列覆盖度：144 field 中仅 **2** 个有 `functional_series` | `partial` | 若需全 FOV 统计须回到 v8 DB |
| **G-7**（R8/DQ-4 新增，"应做而未做"） | **属性级同源表未取**：`node_data_v1.pkl`（294,635,032 B）/ `edge_data_v1.pkl`（201,511,086 B）受吞吐限制（10–16 KB/s ⇒ 5–8 h；GCS 404）⇒ **P7/P8/P14 的同源属性对照与其匹配负对照材料（`Connected`/`ADP`/`Same region`）本轮均未得** | **`unavailable（吞吐限制）`**——**不得**写成"已具备"或"已排除" | 未来轮次：换网络/换时段复用 `parts/` 续传；到位后由 `tools/parse_m11_functional_connectomics.py`（只解析不下载）解析并落 parquet（schema 已由 readme 核：node 26 / edge 19 列） |

> **合规声明（RF-M11-B §五.3）**：本轮**未**用 MICrONS 体素序列计算任何速率/dwell/τ/活动相关统计量。
> `m11_grid_alignment.json` 中的相关仅用于**空间同格性几何判定**（判据是"峰值因子=2.5、位移 0,0"，
> 不含任何活性统计）；若主 agent 认为该几何用途亦在禁令内，可依裁撤除该节（并记得同格性判定将退回元数据声明）。
> **本节点保留反证证据**（大端解码，`tools/demo_m11_microns_decode.py` 一条命令可复核 + B3b 独立复现），
> 并**建议**改判 `decoded(big-endian int16)`；**在裁定变更前，一切产出按 `format_unverified` 执行。**
> ★ **上句为历史引文，按"只增不改"保留原文**：`format_unverified` **已于 R9 被主 agent 推翻、R22 上调为 `pass`** ⇒ **该执行条款自即日起失效**。一切产出按 **`decoded(big-endian int16)`** 执行，**几何用途的禁令随之解除**（本节点 §6b-2 的反证已被采纳，见 §6e/§6m）。

## §6e ★★ RF-M11-D：**R2/R6 反转后的"同源"最终表述**（主 agent 独立复核确认本节点解码结论）

> **① 结构同源 ✅ `measured`**：MICrONS EM/seg/seg_m1300（em 体积 **1.2861 mm³**，主 agent 独立复算一致）。
> **② 功能时间序列同源 ✅ `measured`（R2 反转）**：MICrONS **功能体素序列**由 `format_unverified` 改为
> **`measured`（可解码：大端 int16；200/200 对象完整性通过；lat_x 0.777 / 帧间 0.773）**，
> **允许并鼓励**用于统计（**须应用哨兵掩膜** `int16 < 0 ⇔ uint16 > 65000`，本轮实测 **10,383 / 80,000,000 = 0.01298%**）。
> ★ **上句为历史引文，按"只增不改"保留原文；术语已定稿更正（主 agent §三，2026-10-12）**：
> **"哨兵"一词今后仅指 `int16 = 32767` 饱和（实测计数 `0`）**；`int16 < 0` 的样本**不是哨兵**，
> 而是**低于局部基线的噪声下探**（空间成簇、位于暗区、仅低 ~381 counts）⇒ **定稿 = 无掩膜、保留并登记**
> （10,383/80,000,000 = 0.01298%）。⇒ **本文一切新产出按定稿术语**；旧段落的"哨兵掩膜"字样一律读作
> "低于基线掩膜（已作废）"。B3 的 `unresolved_conflict` 亦已改为 `no_mask`。
> **③ 属性级同源 = 设计可达、本轮 `unavailable（吞吐限制）`**：`functional_connectomics` node/edge pkl（HEAD 200、
> readme schema 实证 node 26 列 / edge 19 列、DOI 10.1038/s41586-025-08840-3），但 294.6 + 201.5 MB 受吞吐限制（10–16 KB/s ⇒ 5–8 h）。
> **④ 逐细胞级仍非同源 ❌**：ΔF/F 轨迹仅在 v8 DataJoint DB（R1 保持，DQ-1 裁定不下载）。

⇒ **最终表述**：**"结构（EM/seg）+ 功能时间序列（2 field 体素序列，`measured`）+ 属性级（connectomics 表，本轮 `unavailable`）"**；
**只有"逐细胞"级仍非同源**（R1）。**S 门/P14 的 dwell/τ 目标带优先用同源自建**（本报告 §6i），
**Allen 2p 降为跨制备参照**（外部效度/稳健性），不再是 S 门主目标。

> **举证标准（新增纪律，RF-M11-D §四）**：把一个格式判为"不可解/不可用"**之前**，必须证明测试集覆盖
> **dtype × 字节序(LE/BE) × 排列(行主/列主/Morton/分块) × 是否差分编码** 全部维度；
> **任一维未覆盖 ⇒ 只能写 `unverified`，不得写 `unavailable/不可解码`**。
> 本条由主 agent 本轮**自身错误**得来（其测试集漏测 `>i2` ⇒ 误判"不可解码"并据此发出降级裁定；
> 本节点以 18 假设 × 6 判据的结构门纠正，主 agent 独立复算后确认并反转裁定）。**该过程已如实入档。**

## §6f-2 ★ 活动通道 = Allen **2p 钙成像**（RF-M11-C）：实测与自建目标带

| 项 | 结果 |
|---|---|
| 探针（新增 8 源 G1–G8，`--extras-append` 追加进 `m11_source_reachability.csv`） | `visual-coding-2p/manifest.json` **200 / 1,340 B**、`stimulus_mappings.json` **200 / 17,907 B**（**刺激条件表已落盘**）、`ophys_experiments.json` 206、`ophys_experiment_data/*.nwb` 206、`ophys_experiment_analysis/*.h5` 206、`ophys_experiment_events/*_events.npz` 206 |
| **★ 实测发现** | `events.npz` **不压缩**（npz=store）⇒ **`n_cells = bytes / (8 × n_frames)`**（`n_frames≈113,835`）：913 KB→**1 cell**、3.64 MB→**4**、16.36 MB→**18**、111.88 MB→**123** ⇒ 可在**不下载**的前提下选群体规模（据此列托管候选） |
| 已落盘活动数据 | `593887846_events.npz`（1×114,130 帧）、`577885923_events.npz`（4×113,835 帧）+ `stimulus_mappings.json` |
| **自建 dwell/τ 目标带** | `data/m11_dwell_reference.{csv,json}`（工具 `tools/build_m11_dwell_band.py`，**只读复用 B4 冻结 `analyse()`**：0.914 s 箱 → 5 帧平滑 → z → AR(1)；并列 H 法/Q 法/**未平滑 φ**） |
| **口径警示（本轮最重要）** | `φ_smoothed ≈ 0.880` vs **`φ_unsmoothed ≈ 0.179`** ⇒ 平滑把 φ 抬高 **≈4.9×**（τ 7.1 s → ~1 s 量级）⇒ **τ 只能作粗粒度指标**，比值口径必须写明用哪一支（与 B4 `honest_limit` 同源） |
| 样本量门实测触发 | low 态完整段仅 **1–4 段** ⇒ **KS vs 指数不可算** ⇒ 形状判据标 **`unavailable`（不是 fail）**；群体规模薄（1–4 cells）已列 caveats |
| 帧率 | **30.9 Hz 为主口径但 `NEEDS_VERIFY`**（NWB `imaging_rate` 因网络未读回；帧数÷30.9=61.4 min 合理 vs ÷11=2.87 h 不合理）；**11 Hz 作敏感性并列** |
| 托管作业 | `data/m11_b1_managed_jobs.json`：**J1**（本节点后台自建带，含日志/产物/完成标记）、**J2**（16.4 MB/18-cell 与 111.9 MB/123-cell npz，**REQUESTED**）、**J3**（NWB 采样率读回，**REQUESTED**） |

## §6f-3 ★ S 门判据**预注册**（RF-M11-C §四；`data/m11_s_gate_criteria_prereg.csv`，12 行）

主判据 = **形状 + 存在性 + 相对尺度**（非绝对 dwell 对齐）：
① 占用非退化（两态各 ≥5% 帧）→ **S-1**；② 转换存在（≥1 且每态 bout ≥10，否则 `unavailable`）→ **S-2**；
③ 低态 dwell **重尾**（p90/中位 ≥2 且 KS vs 指数 p<0.05，需 runs ≥5）→ **S-3**；
④ 慢变量去趋势相关 ≥0.30 **且 lead-lag 峰值 lag>0** → **S-4**；⑤ 绝对尺度按**比值**报告 → **S-5**；
⑥ 容差来源预先写清（B4：口径差 3.87× / 跨条件 6.9× / τ 跨条件 3.94× + 本轮自建带实测）→ **S-6**；
⑦ 样本量硬门（<65 invalid / ≥130 形状 / ≥500 BIC）→ **S-7**；⑧ **双估计器一致性断言**（率 ≤2%、H vs Q、φ_smoothed vs φ_unsmoothed 并列）→ **S-8**；
⑨ M10 失效级 no-go 对照仍为 0 → **S-9**；⑩ 基线不达标断言 → **S-10**；⑪ 消融致其消失 → **S-11**；⑫ dwell 分辨率地板登记 → **S-12**。

## §6f 🔴 优先级 4（四类参考量同口径重建）当前**不可执行** —— 显式登记 `blocked` + 待裁决

主 agent 优先级 4 指定"只用 ① Allen NWB 活动 ② MICrONS 结构 ③ 2 field 元数据；禁用体素序列"。
逐项核对后，**现有裁定与网络条件下三条输入实际都取不到**：

| 指定输入 | 实际状态 | 阻塞原因 |
|---|---|---|
| ① Allen NWB **活动** | **不可用** | 本地 `head_sparse.nwb` 仅 44,695,552 B **真实**前缀；`/units`（spike times）、`/acquisition`（采样率）、`/general/extracellular_ephys` 均在稀疏洞内 ⇒ `unavailable`；续传被禁且网络 1.6–16 KB/s（DANDI 7.2 KB/s） |
| ② MICrONS **结构** | ✅ 可用 | 但结构通道**不含活动**，无法给出率分布/相关/状态结构 |
| ③ 2 field **元数据** | 只有元数据（6.300006 Hz / 40,000 帧 / FOV / 掩膜格） | **无像素** ⇒ 给不出任何分布/相关/dwell |
| （备选）MICrONS 体素序列 | **R2 禁用** | 若启用，2 个 field 的钙域参考量本可同口径重建（每 field 40,000 帧 × 105.8 min） |
| （备选）`units.csv` 全量 51,439,189 B | 未取 | 网络退化 ⇒ 预计 >30 min（**DQ-6 待批**）；含逐 unit `firing_rate`/质量指标 ⇒ 可支撑"率分布"一项 |

⇒ **判定：`blocked`（前置输入不可得），不是"未测"、不是"未命中"**；已同步写入
`m11_data_counts.json::unavailable_register`。**请裁决三选一**：
**(a) 批 DQ-6**（取全量 `units.csv`，可算"率分布"一项，其余仍需 NWB/DB）；
**(b) 批 NWB 续传/托管**（补齐 `/units`、`/acquisition` ⇒ 可算率分布 + units 计数）；
**(c) 改判 R2 为 `decoded`**（则 2 个 field 的钙域四类参考量可在**同源**口径下重建，含 RF-M11-2 劈半自检）。

## §6f-4 待裁决（新增 DQ-8）

**DQ-8**：是否托管 **J2**（2p `events.npz`：16.4 MB/18-cell 与 111.9 MB/123-cell）与 **J3**（2p NWB 采样率读回）？
影响：① 目标带的群体规模由 1–4 cells 提升到 18/123 cells（形状判据的样本量门才可能过）；② 帧率由"推断"升为"实测"。
（清单与命令见 `data/m11_b1_managed_jobs.json`；本节点按 >10 min 纪律未自启。）

## §6h DQ-4 实测结论 = **`unavailable（吞吐限制）`**（reachable but infeasible；不阻塞、不轮询）

| 方式 | 实测吞吐 | 294 MB 推算 |
|---|---|---|
| 单连接（主 agent） | ≈17 KB/s | ≈5 h |
| 8 路并发 Range（17 min 持续） | 聚合 **10–16 KB/s**（各分片 ~1.2 KB/s） | **5–8 h** |
| GCS 镜像 | **HTTP 404**（功能数据**未镜像**） | — |
| 本节点补充实测（用于决策口径） | 同一小时内 **1.6 KB/s ↔ 32 KB/s（20× 波动）**；2p NWB 3.3 KB/s；DANDI 7.2 KB/s；brain-map 1.6 KB/s | — |

**处置（三项均已完成）**：
1. **不阻塞**：DQ-4 按 `unavailable（吞吐限制）` 登记（`m11_functional_connectomics_counts.json` + R-31 + M12 G-7），
   与 R1（v8 DB）、M10（ZENODO）同类：**"可达 ≠ 可行"**；
2. **不轮询**：本节点已**主动停掉**等待器（`waiter.log` 记 `stopped by M11-B1 per DQ-4 ruling`），此后**不再对该对象发任何请求**；
3. **readme 足以规划**：`readme_v1.md`（5,594 B）已落盘且 schema 已机器化解析（node 26 / edge 19 列）⇒
   作为**未来轮次的取数计划**（换网络/换时段；或待 `parts/` 续传完成后由
   `tools/parse_m11_functional_connectomics.py`（**只解析不下载**）自动完成解析）。
**实测吞吐口径（本轮新立）**：`data/m11_throughput_measured.csv`（11 行）——**取数决策必须有实测吞吐支撑，不得凭"应该很快"**。

## §6i 四类参考量的**四态收敛**（`data/m11_activity_reference.csv`；只用现阶段可得数据）

| 参考量 | 可算性 | n / 样本量门 | 值 | 状态 |
|---|---|---|---|---|
| **C1 调谐类**（OSI/DSI/半宽） | ✖ 两条路都不可算（属性表未取 + 2p events 无刺激标签） | 0 | — | **`unavailable`** |
| **C2 率分布** | △ 逐细胞事件率可算 | **n=5 细胞**（门 ≥130） | median **0.1096** | **`thin`**（不得作判据带） |
| **C2 成对相关** | △ 4-cell 件可算 | **6 对** | median **0.0538** | **`thin`** |
| **C3 dwell/τ 目标带（同源主带，RF-M11-D）** | ✅（B4 冻结管线，**MICrONS 自体素序列**） | **333 bins**（门 ≥130 ✅；**BIC 门 ≥500 ❌ ⇒ K 选择 `unavailable`**）；low 态完整段 **6**（门 ≥5 ✅ / ≥10 ❌） | dwell 中位 **12.86 s**（物理；模块口径 12.34 s）、p90/中位 **5.78**、转换 **0.046 Hz**、**τ 32.5 s**；占用 0.631/0.369 | **`measured`（但 S-2 `unavailable`、S-3 KS 腿不过）** |
| C3-b 跨制备参照（Allen 2p） | ✅ 保留为参照 | 4,070 bins | dwell **2.74 s**、τ **6.46 s** | `measured`（**跨制备**；与同源带差 **dwell 4.69× / τ 5.04×**） |
| **C4 行为类** | ✖ 需凭据 | 0 | — | **`unavailable（NEEDS_TOKEN）`** |

⇒ **净结果**：**C3 现有同源自建主带**（`measured`，附窗口/样本量限制）**+ 跨制备参照**；C2 为 `thin`；C1/C4 为 `unavailable`。
（有效维度 / 相位随机化代理地板**未算**：需更多细胞与同管线地板。）

### ★ 与主 agent 独立实现的**交叉核对**（RF-M11-E §一；`data/m11_same_source_crosscheck.{csv,json}`）

主 agent 独立工具 `verify_m11_same_source_band.py` 建出同源带；本节点在同一数据上穷举 **箱宽 × 平滑 × 代理** 网格逐格比对：

| 格（箱宽/平滑/代理） | 本节点 φ | 主 agent φ（差） | 本节点 τ | 主 agent τ（差） | 归因 |
|---|---|---|---|---|---|
| **原生/未平滑/帧均值** | 0.9825 | 0.9825 (**0.005%**) | **9.015 s** | **9.015 s (0.003%)** | **完全一致** |
| 原生/未平滑/高像素占比 | 0.9780 | 0.9804 (0.25%) | 7.134 | 8.003 (**10.9%**) | **τ 对数放大**（τ=−dt/lnφ ⇒ φ 0.25% ⇒ τ 10.9%）+ 代理阈值 |
| 箱/未平滑/帧均值 | 0.8875 | 0.8875 (**0.001%**) | 7.658 | 7.657 (**0.007%**) | **完全一致** |
| 箱/未平滑/高像素占比 | 0.8847 | 0.8873 (0.29%) | 7.461 | 7.642 (2.4%) | 同上 |
| 原生/5帧平滑/帧均值 | 0.9939 | 0.9947 (0.08%) | 26.08 | 29.68 (12.1%) | **平滑边界约定**（本节点 `mode='same'` n=2000；主 agent 为 `'valid'` n=1996）+ τ 对数放大 |
| 原生/5帧平滑/高像素占比 | 0.9946 | 0.9948 (0.02%) | 29.51 | 30.48 (3.2%) | 同上 |
| 箱/5帧平滑/帧均值 | 0.9629 | 0.9712 (0.86%) | 24.15 | 31.24 (22.7%) | 同上 + **箱宽 4.2%** |
| 箱/5帧平滑/高像素占比 | 0.9714 | 0.9721 (0.07%) | 31.47 | 32.27 (2.5%) | 同上 |

**维度效应（实测，作为"同阶窗"来源）**：箱宽（原生↔0.9138 s 箱）**1.18×**；平滑（无↔5帧）**2.9×**；
代理（帧均值↔高像素占比）**1.30×**；**状态估计分辨率（原生↔0.914 s 箱）2.16×**；模块/物理箱宽 **1.0422**。

**结论（三条，与主 agent 一致或已定位）**：
1. **未平滑原生口径下双方独立一致**（φ 差 **0.003–0.005%**、τ 差 **0.003%**）⇒ 主 agent 的
   **"同源目标尺度 ≈ τ 8–9 s（未平滑、原生帧率）"被本节点独立复现** ✓ ⇒ **S 门目标应取同源 8–9 s 量级**，**不使用 ZAPBench 37.18 s**；
2. **τ 的差异几乎全部来自"φ → τ"的对数放大**（dτ/τ = dφ/(φ·ln φ)；φ≈0.97 时 1% φ 误 ⇒ ~19% τ 误）
   ⇒ **凡报 τ 必须同时给 φ、时间基与平滑约定**（否则同一数据的 τ 可差 20%+）；
3. **dwell 差异由"状态估计分辨率"主导**（原生 5.71 s vs 0.914 s 箱 12.34 s = **2.16×**），
   其次为代理定义（对方两口径 3.17/6.67 s，本节点 5.71 s 落在其间 ✓ 自洽）。

### S 门判据体检（同源主带；如实四态）
| 判据 | 结果 |
|---|---|
| **S-1 占用非退化** | ✅ **pass**（低/高态 **0.631 / 0.369**，均 ≥5%） |
| **S-2 转换存在（每态 bout ≥10）** | **`unavailable`**（低态完整段仅 **6**；转换率 0.046 Hz） |
| **S-3 低态重尾 ∧ 拒绝指数** | **`fail/thin`**：p90/中位 **5.78** ✅ 重尾，但 **KS vs 指数 p = 0.558（未拒指数）**，且完整段仅 6 |
| **S-7 样本量门** | valid ✅（333 ≥65）/ shape ✅（≥130）/ **BIC ❌（333 < 500）⇒ K 选择 `unavailable`** |
| **窗口长度限制** | **317.46 s 窗 vs dwell 中位 12.86 s ⇒ 约 25 个低态周期**；时间半劈离散度 **3.23×**（13.7 vs 44.3 s）⇒ 估计不稳，**须随带登记** |
| 口径差（自测） | H vs Q dwell **1.69×**；掩膜开/关 **1.00×**；空间半劈 **1.29×**；φ_平滑 vs 未平滑差 **0.084**（同源序列本身已含 6 帧箱均值 ⇒ 平滑影响远小于 Allen 2p 的 5.6×） |
| 箱宽口径差 | 冻结模块以 **0.9138 s**/箱，实际 **6 × 0.15873 = 0.9524 s** ⇒ **4.2% 系统偏差**；已并列 `*_module` / `*_physical` 两种时间基 |

## §6j J2：**钙域静默比例口径**（P5 (b) 输入；RF-M11-F §一；`data/m11_calcium_silence_reference.{csv,json}`）

**静默定义（三种，全部显式）**：① **zero_activity** = 观测窗内从未出现 > (ΔF/F 中位 + 3·MAD) 的事件；
② **below_baseline** = 单元时间均值 < 0（低于自身 F0=20 分位）；③ **low_activity** = 单元活动 p95 低于
**0.2 × 全单元 p95 中位**（**固定阈值**，非自指分位）。
**单元口径（两种并列）**：**per-ROI 代理** = 20×20 px 空间分箱（**50×50 µm** @2.5 µm/px；**真实 ROI 掩膜未自证解码** ⇒ 标 `proxy`）
｜**full-FOV** = 全视野逐帧（200×200 px = 500×500 µm）。

| 口径 | **field4**（200×200 px，100 个单元代理） | **field2**（100×100 px，25 个单元代理） | 跨 field |
|---|---|---|---|
|---|---|
| **静默比例 · zero_activity** | **0.4900** | **0.4800** | **±2%**（极稳 ✓） |
| 静默比例 · below_baseline | 0.0000 | 0.0000 | 一致 |
| 静默比例 · low_activity（弱单元） | 0.0000 | 0.0000 | 一致 |
| full-FOV · **帧级无事件占比** | **0.8915** | 0.1461 | 差异大 ⇒ **依赖 FOV 大小**（口径警告，须随带登记） |
| full-FOV · 弱帧占比 | 0.0465 | 0.0010 | 同上 |
| 像素级零事件占比 | 0.0052 | 0.0000 | 同上 |
| 单元数（20×20 px = 50×50 µm 代理） | 100 | 25 | — |
| 事件阈值（ΔF/F 中位+3×MAD） | 1.578 | 0.807 | — |

⇒ **J2 结论**：钙域**单元级零事件静默比例 ≈ 0.48–0.49**（两独立 field 差 **±2%**）——可作为 P5 (b) 的钙域口径；
**full-FOV 口径对 FOV 大小敏感**（0.89 vs 0.15）⇒ 报告时必须写明 FOV 与单元数，**不得跨 FOV 直接比较**。

**四条必附登记**：① **钙事件口径 ≠ spike rate** —— 以 in vitro **17.4/17.6 Hz** 作 (a) 带会**高 8.7–34.8×**（B3 代理行量化）；
② 单元 = **空间分箱代理**（非真实 ROI；掩膜解码待官方客户端）；③ 与**电生理静默比例口径不同**（不得混用）；
④ **两处早先的构造性退化口径已修正并留档**（自指 p20 ⇒ 恒 20%；FOV 均值 vs 全局阈值 ⇒ 恒 1.0）——
这正是"**判据必须显式排除退化结构**"纪律的当场应用（本节点自查发现）。
**关联判据带**：新增 **R-33**（J2 行）与 `P5b-silent-ca-*` 两行（**判据带原行不动**）。

## §6k 同源目标带**基线**与"**同阶窗**"实测来源（RF-M11-F §二/§三）

- **基线（新增 R-32；R-21 原行保留不动）**：**未平滑、原生帧率 τ ≈ 8–9 s**
  （本节点 **9.015 s** ⇔ 主 agent **8.003 s**，差 **±6% = 代理定义差**）；箱基（平滑 + 0.914 s 箱）**τ 32.5 s**；
  低态 dwell 中位 **12.86 s**、p90/中位 **5.78**；占用 **0.631/0.369**。
- **同阶窗（实测，全部"同 field 内口径离散"，**非**跨 field 生物学离散）**：
  代理定义 **±6%**（9.015 vs 8.003）｜Q vs H（dwell 比 **1.69×**）｜**时间半劈 3.23×**（13.7 vs 44.3 s）｜
  空间半劈 **1.29×**｜箱宽 **1.18×**｜平滑 **2.9×**｜状态估计分辨率 **2.16×**｜模块/物理箱宽 **1.0422**。
  ⇒ **建议"同阶窗"下限取 ≥3×**（由时间半劈主导），**不得**窄于目标数据自身离散度（RF-M11-2 教训）。
- **第二 field（J5 已完成）**：**吞吐实测 96–121 KB/s（前 3 片）→ 持续 ~8 KB/s（节流）** ⇒ 200 片 = **31.5 MB / ≈50 min**；
  完整性门：首轮 **197/200**（t=75/104 缺失、t=128 截断 69,158 B）⇒ **重取后 200/200 COMPLETE** ✓。
  **跨 field 实测**：field4 dwell 中位 **12.34 s** / τ **31.23 s** / φ_sm 0.9712 ⇔ field2 **10.97 s** / **31.56 s** / 0.9715
  ⇒ **跨 field 离散 dwell 1.125× / τ 1.01×**（**比同 field 内口径离散更小**：时间半劈 3.23×）⇒
  **同源慢结构跨 field 可复现**（这是本轮最强的稳健性证据）。

## §7 结论与下一步建议

1. **G2 判定 = pass（附 §3 限制）**：51 清单源 + 17 补测源逐行实测；结构与功能**同源**路径落盘；
   **采集参数已读回**（6.300006 Hz 等）——这是清单 §1.1.2 "未读回 ⇒ unverified" 的解除条件。
2. **P2 尚不足以"定稿 dwell 带"**：需先核 `int16/uint16` 与 **10× 分辨率口径差**（§3.1）；
   核对后即可用"2 field × 6.3001 Hz × 40,000 帧（105.8 min）"做劈半可达性自检（RF-M11-2）。
3. **未启动**（遵守门链）：G0 引擎门、G1 生理门、S 机制门、P7–P14 —— 且 RF-M11-2..6 未落盘。
4. **交主 agent 的六个决策**：DQ-1（DB 取数 vs 场级重建）/ DQ-2（NWB 全量续传托管作业）/
   DQ-3（21 条冲突项落清单）/ **DQ-4**（功能连接组学两表 ≈496 MB —— 决定 P2 能否全过）/
   **DQ-5**（把大端读法/完整性门/6.30 Hz 写入清单读取约定）/ **DQ-6**（`units.csv` 51.4 MB + `channels.csv` 6.6 MB 是否托管取）/
   **DQ-7**（NWB 读回：`brain-map` 剩 46.9 MB 或 DANDI 最小 asset 71.8 MB —— 现网络 1.6–16 KB/s 与 7.2 KB/s ⇒ 需托管，或改由 `probes.csv` 自描述表满足采样率需求）。

---

## §6l 【DQ-6 完成】Allen Neuropixels 逐单元表的**三件交付**（(a) 带上界候选 / 层·区登记 / P14 隔离）

> **产物**：`tools/parse_m11_allen_units.py`（可复现）→ `data/m11_allen_units_reference.json`、
> `data/m11_allen_units_by_session.csv`（58 行）、`data/m11_allen_units_by_structure.csv`（58 行）、
> `data/m11_allen_rate_band_by_area_group.csv`（3 行）；参考锚新增 **R-34…R-39**（R-25 原行保留）。

### (1) 完整性门：**新增一条强于 sha256 的判据**
`units.csv` = **51,439,189 B / 135,275 单元 / 27 列**，`size == Content-Length`，
**S3 多段 ETag 复算通过**（`md5(concat(md5(part_i))) == "5555828a…-7"`，part=8 MiB × 7 段）
⇒ **本地文件逐字节等同服务器对象**（判据锚在**服务器侧**，不依赖外部发布的哈希）；末字节 `\n`、非 27 字段行 **0**。
连接链 `units→channels→probes→sessions` **135,275/135,275 = 100%**（未解析 0）。
另取 `channels.csv`（6,613,329 B）作连接键——★ 实测**推翻了"1 MB 前缀够用"的假设**（前缀覆盖 channel id 849.706–849.801 M，
而 units 引用 **849.991–850.264 M** ⇒ 覆盖率 **0%**）。
**吞吐**：25 日 22:33 的托管作业**已死**（无进程、日志 0 字节、3.35% 停滞 8.5 h）；今晨当场实测 **5.9 MB/s**
（昨夜 1.6 KB/s，**差 ~3,600×**）⇒ `curl -C -` 8.42 s 续传完成。

### (2) ★ (a) 带 = **仅得上界候选**（口径硬约束）
- `epoch_name_quality_metrics` 在 **全量 135,275 行一律 `complete_session`** ⇒ 该率**含视觉刺激呈现**，
  **不是自发率** ⇒ **(a) 自发率带仍 `unavailable（口径不含）`**（需 NWB `spontaneous` epoch，单件 >2 GB / 稀疏洞）。
- **皮层限定**（M11 = 皮层微环路；MICrONS minnie65 = **VISp**）：

| 口径 | n 单元 | 占比 | good 中位 | good p90 | 逐 session 中位带 | 跨 session 极差 |
|---|---|---|---|---|---|---|
| **VISp**（与 minnie65 同名区） | 10,750 | 8.0% | **2.485 Hz** | 10.30 Hz | 2.579 Hz（p10 1.969 / p90 3.144） | 3.29× |
| **VIS\***（视觉皮层全） | 49,995 | 37.0% | **2.657 Hz** | 10.35 Hz | 2.760 Hz（p10 2.234 / p90 3.107） | 1.97× |
| 海马（**非同源**） | 44,659 | 33.0% | 2.701 Hz | 11.16 Hz | 2.805 Hz | 3.81× |
| 全单元聚合（**禁止当皮层先验**） | 135,275 | 100% | 3.175 Hz | 12.41 Hz | 2.798 Hz | 2.90× |

  ⇒ **聚合值被非皮层结构抬高**（LGd 6.86 / VPM 6.64 / LP 4.95 Hz）⇒ **带必须区级限定**。

### (3) 跨口径量化：**实测替换 B3 的估计值**
VISp good 中位 = **0.794 × 钙事件率 Nyquist（3.150003 Hz）**，且 **41.29% 的 good VISp 单元率 > Nyquist**
⇒ 钙成像在**事件率口径**下**结构上无法表征约四成**兴奋性群体率。
（B3 原估计 = "spike 带上界 2.0 Hz 已达 Nyquist 的 63.5%" ⇒ 方向一致、**实测更强**。）
⇒ **钙域与电生理域必须分列**；任何"钙 vs spike 同分布"的表述本轮**被实测否证**。

### (4) 层/区登记（主 agent §四②）
区级 **58 结构 / 58 session 全覆盖**；CCF 背腹深度**全量可用**（n=135,275，-1250→4890 µm）。
★ **acronym 不含层号**（带数字者仅 CA1/CA3/CA2 = 海马亚区）⇒ **区级 `measured`；层（L1–L6）`derived`**
（须 CCF 深度 + 皮层表面/厚度模型）⇒ **不得表述为"层已登记"**。

### (5) P14 隔离材料（主 agent §四③）+ 退化登记
`brain_observatory_1.1`（32 session / 74,072 单元，good 中位 **3.390 Hz**）vs
`functional_connectivity`（26 session / 61,203 单元，**2.941 Hz**）⇒ 差 **1.15×** ⇒ **不因刺激协议分档**；
**独立性单位 = session**。质量构成 `good` 99,180 / `noise` 36,095。
**退化登记**：**5.84%（7,894/135,275）** 命中 `isolation_distance ≥1e4`（最大 2.3e15）或
`isi_violations >20`（最大 4,061.65），归因于**近静默单元（<0.01 Hz，n=4,343）的指标退化**
（**不是奇异值**）⇒ 单独登记、从带中剔除；空字段最多三列 `silhouette_score` 19,524 / `nn_hit_rate` 10,875 /
`recovery_slope` 7,920 ⇒ **空 ≠ 0**。

> **判定器**：`tools/validate_p11_data_gate.py` 新增 **P1.21a–j**（完整性/连接链/口径强制/退化登记/新增行/区级限定/
> 跨口径/层 derived/清单入账/(a) 仍 unavailable）⇒ **101 断言 0 失败，P1=pass P2=pass G2=pass**。

---

## §6m 【终裁回执】`format_unverified` 文本更正 + **G-1′/G-2′** + 优先级 4 钙域四类参考量

### 6m.1 主 agent §一 要求的文本更正（**已落**）
- **§6e 早已是更正后的表述**（"② 功能时间序列同源 ✅ `measured`（R2 反转）"）——主 agent 所指"仍按旧判写"的，
  实为 **§6b-2 的 RF-M11-B 合规声明末句**（历史引文）与 **M12 缺口 G-1 行**；两处**均已加更正标记且保留原文**
  （报告 L252 的 G-1 行、L266 的旧执行句）。
- **旧执行条款作废**：`format_unverified` 已被 R9 推翻、R22 上调 `pass` ⇒ 一切产出按
  **`decoded(big-endian int16)`** 执行；**几何用途禁令解除**。

### 6m.2 **G-1′**（同源的正确表述 —— 采用主 agent 措辞，**并按其未掌握的两项事实更正**）
> **结构-功能同源**：结构侧 `measured`（em/seg，独立复算 1.2861 mm³）；功能侧体素序列
> **`decoded(big-endian int16)`**（200/200 完整性通过、几何同格经**实像素**验证、
> **同源目标带已建**：τ 未平滑 **8.0–9.0 s** / 平滑 **29.6–32.6 s**、H 法 dwell 低态 **3.17–6.67 s**）。
> **"同源"成立的范围**：① **同源 ✅（结构 + 功能时间序列，同一动物/制备）**；② **逐细胞 ΔF/F ✗ `unavailable`**（仅 v8 DB）；
> ③ **跨 field 生物学离散 ✗ `unavailable`**。
>
> ★ **两处必须更正主 agent 草稿的事实**（否则会写进 M12 成为**假限制**）：
> **(i) 覆盖不是"仅 1 个 field × 2,000 帧"** —— **第二个 field 已在盘并已入库**
> （`data/m11_raw/microns/functional_subset_f2/`，`session5_scan3_field1`，**200/200 片 / 31,472,642 B / 2,000 帧**）
> ⇒ 覆盖 = **2 个 field × 2,000 帧 = 634.9 s**；**(ii) ③ 因此不是 `unavailable`，而是 `measured（但仅 2 field）`**：
> **跨 field 离散已实测** = dwell **1.125×** / τ **1.01×** / 静默比例 **1.0208×**
> ⇒ 写成 `unavailable` 属**"未测 ≠ 未命中"的反向错误（把已测写成未测）**。
> ⇒ **正确表述**：**"同源在时间序列级成立（2 field / 634.9 s）；在逐细胞级不成立；跨 field 级已测（离散 ≤1.125×，但仅 2 field，不足以代表 144 field）"**。

### 6m.3 **G-2′ / G-4′ / G-8（DQ-6 状态更正）**
- **G-2′**：四类参考量的**钙域口径本轮已交付**（见 §6m.4）⇒ 由"部分可算"上调为 **`measured`（代理口径）**；
- **G-4′（DQ-6）**：**`units.csv` 全量已在盘且已解析**（51,439,189 B，**多段 ETag 复算逐位命中**，
  135,275 单元 / 58 session / 连接链 100%；`channels.csv` 6,613,329 B 亦已入库）
  ⇒ 主 agent 判 `unavailable（吞吐限制）` 的**依据（3.3% 停滞）已被 07:03 的续传完成取代**
  （实测 **5.9 MB/s**；昨夜 1.6 KB/s 与今晨差 ~3,600×）。**精确 units 计数已由 `derived` 外推改为 `measured`**
  = **135,275**（此前外推 ≈136,720，偏差 **1.06%**）。**注意**：该计数是**数据集计数**，仍**不得作活动口径的计数带**。
- **G-8（新增）**：**代理单元 ≠ 神经元** —— 50×50 µm 分箱内神经元数未知（真实 ROI 掩膜未解码）
  ⇒ 分箱事件率**不可**与单神经元 spike 率并列；跨口径只允许在**全视野聚合**层面陈述。

### 6m.4 优先级 4：**钙域四类参考量**（`tools/build_m11_calcium_reference_quantities.py`，6.2 s，无新下载）
**同管线**由断言 **P1.23a 以数值一致（|Δzero|=0.00e+00）证明**（非声称）；箱 = 6 帧 = **0.9524 s**
（模块称"0.914 s 箱"，命名偏差 4.2% 已留痕）；`B4.sample_size_gate` = valid+shape ✅（2,000 帧；**333 箱 < 500 ⇒ BIC 结论 `unavailable`**）。

**① 率分布（钙事件口径，与 spike 分列）**

| 口径 | field4 | field2 | 跨 field |
|---|---|---|---|
| per-ROI 代理（50×50 µm）事件率中位 | **0.0032 Hz** | **0.0032 Hz** | 1.00× |
| 同上 p90 / max | 0.1008 / 0.3276 Hz | 0.6584 / 5.3771 Hz | 6.5× / 16.4× |
| **全视野聚合**事件率 | **3.2445 Hz** | **7.7711 Hz** | 2.40× |
| 每代理单元事件中位 | 1 | 1 | — |

★ **触分辨率地板**：中位单元在 317 s 内仅 **1 个事件** ⇒ 率的相对不确定度 ≳60% ⇒ **只能表述为
"多数代理单元在窗内几乎无事件"，不得把中位率当神经元发放率**。（同族纪律：B4 `dwell_resolution_floor`。）
★ **双估计器**：原生帧计数 vs 分箱计数（**同一帧范围**，剔除 2 帧尾差）⇒ rel_diff **0.00e+00** ✅。

**② 群体相关（★ 本轮最重要的"防假结论"）**

| field | 原始口径 平均 r | PR / λ1/Σλ | **去全局分量后** r | PR / λ1/Σλ |
|---|---|---|---|---|
| field4 | 0.79489 | **1.558** / 0.7986 | **−0.00825** | **9.033** / 0.2647 |
| field2 | 0.95943 | **1.082** / 0.9611 | **+0.02915** | **4.187** / 0.4344 |

★ **退化体检结论**：**原始平均 r 由单一共同模式（PR≈1.1–1.6）主导** ⇒ **不得**作为群体结构/同步证据；
**去掉逐箱跨单元均值后，平均成对相关塌到 ≈0**（−0.008 / +0.029，代理地板 ≈0 ± 0.008）
⇒ 在这 2,000 帧窗内**未检出全局模式以外的成对群体结构**。**若把 0.79/0.96 写进 P14 将是测量假象。**

**③ 状态结构**：复用 `m11_dwell_reference.json`（Q 法 + H 法 K=2；平滑与未平滑并列）；
**内部一致性** `|dwell − 1/(2λ)|/dwell = 0.1547 ≤ 1` ✅（`B4._consistency`，与产物内存储值一致）。
**④ 静息静默比例**：J2 正解（见 §6l / L38；验收 5/5）。
**断言**：**P1.23a–g 全部 OK**（同管线/双估计器/样本量门/内部一致性/两口径/跨 field 实测/代理地板）。

---

## §6n 【R38 ③ 回执】两侧分开写 + **一处"保留句"本身已过期**（2026-10-12）

### 6n.1 按主 agent R38 ③ 要求落盘的**两侧分开**表述（`format_unverified` 陈旧标注全部撤除）
> - **结构侧：同源成立 ✅** —— MICrONS EM/seg（`measured`；em 体积 **1.2861 mm³**，主 agent 独立复算一致）；
> - **活动侧：跨源代理** —— MICrONS **逐细胞轨迹 `unavailable`**（only-in-DB，R1 保持）；
>   但**体素级序列 = `decoded(big-endian int16)`（`pass`）**（**200/200** 片；BigTIFF 地面真值锚 0.767/1,879 ⇔ 0.777/1,863；**两个独立实现互证**；本轮新增**解码自证**：F 序 lat_x **0.7769**、C 序负对照 **0.3787**）
>   ⇒ **不得再写"体素 `format_unverified`"**（该判定已于 **R9 推翻 / R22 上调 `pass`**）；
> - **活动带来源（分层，勿混）**：**同源 MICrONS field 级带**（`m11_dwell_reference.json`，**R52/R57 起为 S 门主带**）
>   **＋** **Allen 2p 跨制备参照**（`m11_dwell_reference_allen2p.json`，证据级别 **`derived/cross_source`**）。
>
> **同步更正**：报告 §6e（已含 R2 反转表述）＋ **M12 缺口 G-1′（§6m.2）** 与 **G-2′（§6m.3）** 均按上表口径；
> 本节点**一切新产出**不再出现 `format_unverified`（判定器 **P1.24a–d** 强制）。

### 6n.2 ⚠️ **一处必须提请复议：主 agent 让"保留"的半句本身已过期**
R38 ③ 要求保留我原句中的「**不得声称'用同源成像约束'**」这半句。**该半句在当时成立、现在已不成立**：
- 当时（R38 时点）活动带确实**只**来自 Allen 2p ⇒ 对"用同源成像约束活动带"的设防是正确的；
- **但自 J5/R52/R57 起**，**同源 MICrONS field 级带已建成**（2 field × 2,000 帧 / 634.9 s；τ 未平滑 8–9 s、
  dwell 12.34/10.97 s、跨 field 离散 1.125×/1.01×），并已被主 agent 定为 **S 门主带**。
⇒ **若照原样保留该半句，将与 R52/R57 直接冲突**（等于否认已建成的同源带）。
**本节点处置**：按"**不得写假陈述**"纪律，把该半句改为**分层保留**——
> **✅ 可声称**：同源**体素级**时间序列可用于统计（`decoded`，200/200，几何同格经实像素验证）；
> **❌ 不可声称**：同源**逐细胞**活动（`unavailable`，only-in-DB）；也不得把**体素级**同源偷换为**逐细胞**同源。
**请主 agent 复核**：若坚持要求保留旧半句，请显式说明其与 R52/R57 的关系（我按"只增不改"保留此复议记录）。

### 6n.3 J3（帧率实测）进展：**已在预算内定位到值的位置**，但**取值落在未取区间**
- **零下载路径先试**（纪律：先找零成本判据）：对**已在盘**的 6 MB 前缀用**稀疏文件法**打开 ⇒
  `h5py` 拒绝截断文件，故按**稀疏三处标注**（表观 176,019,299 / 真实前缀 / 洞区间）重建后可用。
- **吞吐当场复测（带门）**：30 s 探针 = **1,116,049 B/s（1.09 MB/s）**；后续 Range = **5,624,077 B/s（5.6 MB/s）**
  ⇒ **远高于 50 KB/s 门**；⇒ 你此前"该 NWB 仅 3.3 KB/s"的依据**已过期**（你自己的纪律：吞吐依据只有小时级有效期）。
- **有界取数（<50 MB，未越阈值）**：head 25 MB + tail 20 MB（**共 45.1 MB 真实字节**，中间 130.9 MB 为洞）。
- **★ 定位成功**：`/general/optophysiology/imaging_plane_1/imaging_rate`
  （dtype `|S4`、shape `()`、storage **4 B**、object-header addr **176,018,211**，落在尾部已取区间内）；
  同组另有 `imaging depth`(已可读: `275 microns`)、`pixel_size`(|S9, addr 176,011,803)、`fov`(|S34, addr 176,006,723)。
- **但取值仍不可读**：这些标量字符串的**堆数据不在已取的 head/tail 区间内**（落在中间 130.9 MB 洞里）⇒
  `h5py` 读回空串。**继续需要 >50 MB 取数（全件 176,019,299 B）⇒ 按纪律「≥50 MB 一律先问」，
  本节点不自取**。⇒ **J3 现判 `unavailable（待预算裁决）`**，并登记：**路径与 dtype 已实证**（|S4 ⇒ 值形如 `11.0`/`30.9`），
  一旦获批全件或获批更大窗口，取值可在秒级完成（5.6 MB/s）。
