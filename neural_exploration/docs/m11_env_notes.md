# M11 环境/口径/踩坑笔记（`docs/m11_env_notes.md`）

> **节点**：M11 执行节点 **B1**（工作包 = **G2 数据门（P1/P2）**）
> **依据**：`docs/生物仿真M11实施清单.md`（§0/§1/§7 P1–P2/§10.1）+ `docs/m11_list_review_master.md`（RF-M11-1..6）
> **纪律**：`docs/开发WORKFLOW.md` 六步循环 + 四态验收（pass/fail/invalid/unavailable）+ 冻结件零修改（清单 §0.3 #11）
> **本文件为 L1 起编号的续写体**（M5–M10 惯例）：L1–L8 = 清单 §0.10 前置确认处置；L9+ = 本轮实测新结论。
> **解释器**：`./.venv-m11/bin/python`（3.9.6，本轮新建：h5py 3.14.0 / numpy 2.0.2 / pypdf 6.20.0；
> **未触碰** `.venv-m9`/`.venv-m10`/`.venv-neuro`/`.venv-db` 已有包 —— 见 L17）
> **运行前缀**：`cd /Users/weidong/ai/small_world && PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`
> **墙钟**：本轮 G2 实测 **≈4.0 h 墙钟**（08:09→12:15 UTC；含 2 次全量可达性探测重跑 + 136 MB 有界子集 + Allen NWB 部分下载与停滞等待）
> **机器负载**：本机 10 核上另有并行长作业（`uptime` 负载 6–8）；本节点**全程限线程、零 GPU/MPS 作业**。

---

## L1–L8 — 清单 §0.10 前置确认与处置（逐条）

| # | 清单项 | 本节点（B1，G2 数据门）处置 | 状态 |
|---|---|---|---|
| **L1** | 冻结件复用 / M11 只新建 | 本轮**只新建** M11 文件；未 `import` 任何 M0–M10 源码（数据门为纯数据 + 文档）→ 不存在"组合不修改"风险面 | ✅ 满足 |
| **L2** | 数据源与许可（G2 门前置） | **51 清单源 + 17 补测源逐条实测**（HTTP 码/字节/内容型/两次探测）→ `data/m11_source_reachability.csv`；许可与引用要求逐源登记于本文件 L9/L10/L13 | ✅ 已落盘 |
| **L3** | M10 未定论与未交付件（只读不依赖） | M11 数据门**完全不依赖 M10 结论**（清单 §0.3 #1"必须先行"）；未读 M10 的 P6 判定作输入 | ✅ 按世界无关执行 |
| **L4** | M10 内核作对齐参考 | 属 G0（P3），本工作包**未启动**（前置门链 + RF-M11-2..6 未落盘） | ⏸ 未启动（正确） |
| **L5** | 环境与依赖 | 新建 `.venv-m11`（数据侧）：h5py / numpy / pypdf；**新增依赖已登记**（本行）；未修改其它 venv | ✅ 已登记 |
| **L6** | 预算与确定性纪律 | 全部命令固定 `PYTHONHASHSEED=0`；探针**两次探测 + 不一致时三次多数裁决**；CSV 只写不合并（见 L16）；本轮**未新增任何判据带 CSV**（属 P2/P6 后续） | ✅ |
| **L7** | 数据隔离原则落地（§0.8） | MICrONS 的 A/C 同源拆分的**物理前提已实测**：结构（EM/seg，同动物 minnie65）与功能（双光子 144 field）**同源同体积**；**field 级隔离可行**（144 field 中仅 2 个有原始钙电影 → 天然隔离） | ✅ 前提成立 |
| **L8** | M10 未完成件与教训交接 | 已内化：① 编码路径断言（G0 用）② 基线不达标断言（P6 用）③ 负对照臂（P7+ 用）④ **静默失效分级**（本轮实测出 3 处"看起来可达其实无数据"，见 L9.4） | ✅ 已承接 |

---

## L9 — 数据源可达性实测（**68 源**，逐源 curl 两次探测）

完整表：`data/m11_source_reachability.csv`（含 `verdict` 词表 + `note` 逐行证据 + SHA-256 前缀）。
工具：`tools/probe_m11_sources.py`（`--all`，单源硬超时 15 s，**父进程强制回收**）。

### L9.1 判定分布（源 id 1..51 = 清单 §1.6 原表；F1..F17 = 本轮补测）

| verdict | n | 说明 |
|---|---|---|
| `REACHABLE` | **51** | 对象级 200/206 实测且响应体非空、非 parked page |
| `REACHABLE_NO_DATA` | 3 | 端点可达但**未证实对象**：#17 CAVE 表 API(404) / #42 NMC 下载(404) / F1 CAVE materialize(404) |
| `NEEDS_VERIFY` | 3 | **桶存在但对象级未证实**（403 AccessDenied ≠ NoSuchBucket）：#19 `microns_public_datasets` / #20 `microns-seunglab` / #32 `allen-synphys` |
| `NEEDS_TOKEN` | 1 | #35 IBL OpenAlyx：403 `Authentication credentials were not provided` |
| `BLOCKED` | **8** | #18 `global.daf-apis`(302→挂起) / **#33 synphys 门户（连接超时，间歇）** / #34 DNS 失败 / #45 `raw.githubusercontent`(reset) / #46 Zenodo(TLS 自签) / #47 figshare(403) / #48 GIN(403) / F16 synphys 对象级 HEAD(403) |
| （两次探测一致性） | **68/68 一致** | 最终一轮**无一行出现 pass1/pass2 HTTP 码不一致**（确定性证据） |
| `NOT_EXIST` | 2 | #24 `allen-cell-types` S3（**NoSuchBucket**）/ F6 `seg_m1300` 的 S3 镜像 info（**NoSuchKey**） |

### L9.2 主通道结论（可直接用于 M11 取数设计）

1. **主通道 = 公开桶匿名直读**：`storage.googleapis.com/iarpa_microns`（GCS，**主**）、
   `bossdb-open-data.s3.amazonaws.com`（S3 镜像，**备/功能侧主**）、`allen-brain-observatory.s3.amazonaws.com`、
   `visual-behavior-neuropixels-data.s3.us-west-2.amazonaws.com`、`ibl-brain-wide-map-public.s3.amazonaws.com`
   —— 均可 list 且可**逐对象 200**（本轮的 136 MB 功能子集即从 S3 桶逐对象取回，见 L11）。
2. **GCS 与 S3 镜像的 key 空间不同（易踩）**：GCS 桶名 = `iarpa_microns`，**对象 key 前缀 = `minnie/...`**；
   S3 镜像桶名 = `bossdb-open-data`，**key 前缀 = `iarpa_microns/minnie/...`**。
   ⇒ GCS 列举必须用 `prefix=minnie/minnie65/...`（清单 §1.6 #5 的桶根列举可用，但**加 `iarpa_microns/` 前缀会 KeyCount=0**——本轮实测踩到，见 L20 冲突项）。
3. **两份 info 并不等价**：GCS 的 `em/info`（4,117 B）与 S3 的 `em/info`（5,970 B）**scale 命名与编码不同**
   （GCS `8x8x40` + `raw`；S3 `8_8_40` + `jpeg`/`raw`）⇒ **跨镜像混用元数据 = 口径错**（M9 B2 纪律）。
4. **`REACHABLE_NO_DATA` 陷阱（静默失效的数据门版本）**：本轮实到 3 类
   ① 端点 200 但 0 字节（S3 的"目录占位键"）；② 404 页面 129 KB（synphys 门户）；
   ③ **列举成功但 KeyCount=0**（错误前缀）——三者都**不是**"数据可得"，必须如实标注（清单 §1.6 词表）。

### L9.3 传输后端与吞吐实测（决定 M11 取数策略）

| 源 | 实测吞吐 | 备注 |
|---|---|---|
| S3 `bossdb-open-data`（功能 chunk，685 KB） | **194–349 KB/s**（单流） | 波动大；受并发/时段影响 |
| GCS `iarpa_microns`（shard 前缀 range 1 KB） | 206 OK | **info 首字节 5.6–8.9 s**（元数据小文件也偏慢） |
| Allen API `well_known_file_download`（91.5 MB） | **~36–58 KB/s** | 91.5 MB ⇒ **约 40–50 min**（见 L13/L23） |
| Allen Cell Types API（0.68 MB/页） | ~1.7–28 KB/s 波动 | 分页 400 行/页 = 5 页 3.3 MB |
| Europe PMC REST | 每次查询 ~1–2 s | 37 行 provenance ≈ 1 min |

- **纪律落地**：单流慢 ⇒ **有界子集用 3 路并发**（`--jobs 3`），但**绝不同时起两个大作业**
  （本轮实测：NWB 全量 + 子集并发时，子集吞吐由 ~330 KB/s 掉到 ~90 KB/s ⇒ 已**主动暂停 NWB 作业**让子集先完成，见 L21）。
- **M10 L11 结论复现**：Python `urllib` 对 GCS 的 TLS 握手仍不可用（沿用 M10 教训）⇒ **本节点全部取数以 `curl` 子进程为传输后端**。
- **新坑（L9.3a）**：`curl --max-time` 在个别站点/网络状态下**会失效**——本轮 `portal.brain-map.org`（#33）一次探测
  挂死 **925.9 s**（`curl: (28) Connection timed out after 925893 milliseconds`，而 `-m 15` 已给）。
  ⇒ 探针**必须由父进程硬回收**（`Popen` + `communicate(timeout)` + `kill()`），已固化进 `tools/probe_m11_sources.py`。
- **新坑（L9.3b）**：**同一源在不同轮次可给出不同 verdict**——#33 三轮分别为 `404+129 KB` / `404` / **连接超时**，
  ⇒ 已按"三次探测 + 多数裁决 + 全码入 note"处理；但**跨轮不稳定本身**必须如实登记（不得挑好看的那轮）。
- **新坑（L9.3c）**：**任务级带宽竞争会改变结论口径**——子集下载与 Allen NWB 并发时，子集吞吐由 ~330 KB/s
  降到 ~90 KB/s，并出现 2 片超时失败（已 `--retry-missing` 补齐）⇒ 大作业与有界子集**不得同时跑**（本轮已串行化）。

---

## L10 — MICrONS 结构侧实测（em / seg / seg_m1300）

产物：`data/m11_microns_structure.json`（工具 `tools/fetch_m11_microns.py --struct --shard-probe`）。
**证据级别**：元数据 = `measured`；chunk 计数 = `derived`；shard 清单/对象读取 = `measured(有界)`。

| 卷 | 主通道 info | data_type | 最细 scale 分辨率 | 体素数（最细） | 体块尺寸 | 体素数→物理体积 |
|---|---|---|---|---|---|---|
| **em** | GCS 200 / 4,117 B；S3 200 / 5,970 B | uint8 | 8×8×40 nm | 212,992 × 180,224 × 13,088 | **1.704 × 1.442 × 0.524 mm** | **1.286 mm³** |
| **seg** | GCS 200 / 7,448 B；S3 200 / 7,448 B | uint64 | 8×8×40 nm | 192,424 × 131,051 × 13,008 | **1.539 × 1.048 × 0.520 mm** | **0.840 mm³** |
| **seg_m1300** | GCS 200 / 7,628 B；**S3 镜像 404 NoSuchKey** | uint64 | 8×8×40 nm | 218,809 × 161,359 × 27,858 | **1.751 × 1.291 × 1.114 mm** | **2.518 mm³** |

- 每卷 **11 个 scale**（8×8×40 → 8192×8192×10240），最细 scale chunk = **128×128×32**；
  `em` 编码 `raw`（GCS）/ `jpeg`（S3 的粗 scale），seg 两卷均为 `compressed_segmentation`。
- **chunk 计数（derived，公式 = Π⌈size/chunk⌉）**：em **958,251,008** chunk；seg **626,819,072**；seg_m1300 **1,878,146,010**。
  按未压缩单 chunk 字节（em 524,288 B / seg 4 MiB）给**上界**：em ≈ **502.4 TB**（raw ⇒ 即实际量级），
  seg/seg_m1300 的 2.6 PB / 7.9 PB 只是**未压缩上界**（实际为压缩编码，压缩比未知 ⇒ **不声称**）。
- **shard 有界实测（measured）**：三卷最细 scale 各列举前 3 页（3,000 键/卷，`delimiter` 缺省）：

| 卷 | 测得对象数 | 测得字节 | 单 shard 尺寸 | truncated |
|---|---|---|---|---|
| em / 8x8x40 | 3,000 | **20.06 TB** | 9.47 MB – 4.44 GB | **true（总量未测）** |
| seg / 8x8x40 | 3,000 | **1.154 TB** | — | **true** |
| seg_m1300 / 8.0x8.0x40.0 | 3,000 | **1.043 TB** | — | **true** |

- **对象级真实可取（measured）**：三卷各取前 2 个 shard 的 `Range: bytes=0-1023` → **全部 HTTP 206**；
  **未下载任何完整 shard**（最大 4.4 GB 级）→ 服从"禁止全量 EM"硬条款。
- ⚠️ **口径提醒（给 P4/§3 用）**：`em`（1.286 mm³）、`seg`（0.840 mm³）、`seg_m1300`（2.518 mm³）
  是**三个不同边界**；论文 in vivo 体积写法是 **1.3 × 0.87 × 0.82 mm³** ⇒ **四者不可互换**，
  引"体积"时必须写明是哪一卷的 bbox（本项已进 L20 冲突登记）。

---

## L11 — MICrONS 功能侧：**采集参数读回**（RF-M11-2 的输入）

产物：`data/m11_microns_protocol.json`（工具 `--func-meta`）。**清单 §1.1.2 标注的"未读回 → 一律写 unverified"本轮已解除。**

### L11.1 功能数据目录实测（对象数/体积，全部 `measured`）

| 前缀 | 对象数 | 总体积 | 用途 |
|---|---|---|---|
| `functional_data/` 根 | 3 键 + **14 子前缀** | — | 目录语义（含清单未列的 6 个前缀，见 L20） |
| `functional_series/`（**原始钙电影体数据**） | 2 个 series | 6 KB（属性）+ 分片 | **P2 主对照** |
| `field_summaries/`（**逐 field 网格属性**） | **144** field | 144 × 230 B | field 网格/FOV |
| `field_seg_masks/`（**神经域/ROI 掩膜**） | **144** field | 144 × 561 B(info) | **掩膜定义 = 判据可比性硬前提** |
| `two_photon_functional_scans/` | **39** 个 BigTIFF | **2.686 TB** | 原始扫描（按需 Range 读，**禁全量**） |
| `platinum_camovies/` | 23 | **1.553 TB** | 刺激电影（钙通道版） |
| `platinum_stimulus_movies/` | 22 | 42.3 GB | 刺激电影 |
| `stimulus_movies/` | 58 | 246.9 GB | 刺激电影 |
| `functional_coregistration/` | 2 | 14.4 KB | **结构-功能同源映射（H4 关键件）** |
| `functional_connectomics/node_and_edge_properties/v1/` | 4 键 | — | 节点/边属性（**清单 §1.7 的首选回退**） |

### L11.2 时间口径（**读回值，逐项可核**）

`functional_series/*/attributes.json`（223 B，两次探测均 200）：

| 量 | 读回值 | 依据 |
|---|---|---|
| **帧间隔** | **0.15873 s** | `resolution=[2.5,2.5,10,**0.15873**]`，`units=[um,um,um,**s**]` |
| **帧率** | **6.300006 Hz**（= 1/0.15873） | 同上（**清单此前记为"未读回 ⇒ unverified"，本轮升级为 `measured`**） |
| 帧数 | **40,000** | `dimensions=[248,440,1,40000]` |
| **单 field 时长** | **6,349.2 s（105.82 min）** | 帧数 × 帧间隔 |
| **平面数** | **1**（该 field 单平面） | `dimensions[.., z=1]` |
| **体积率** | **6.300006 Hz（该 field）** | 1 平面 ⇒ 体积率 = 帧率；**多深度体积的 volume rate 不可由此推出**（见 L23） |
| 体素 | **2.5 × 2.5 × 10 µm** | `resolution` + `units` |
| FOV | **620 × 1100 µm**（248 × 440 px） | `size × resolution` |
| 块形状 | `[200,200,1,10]`（session4）/ `[100,100,1,10]`（session5） | `blockSize` ⇒ **每片 = 10 帧** |
| 数据类型 | `int16`（声明） | `dataType` —— **但字节行为异常，见 L11.3 ⚠️** |

**可用的两个 field**：`session4_scan7_field4`、`session5_scan3_field1`（**其余 142 field 无原始钙电影体数据**）。

### L11.3 有界子集（**200 片 / 2,000 帧 / 136.3 MB**，逐对象 provenance）

- 清单：`data/m11_functional_subset_manifest.csv`（200 行：`url / http_code / bytes / sha256 / retrieved_at`）。
- 汇总：`data/m11_functional_subset_summary.json`（`n_ok=200/200`，`total_bytes=136,348,754`）。
- **可解压验证**（`data/m11_functional_subset_verify.json`）：**实测 chunk 文件 = 20 B 大端头 + gzip 成员**：

```
offset 0x00: 00000004 000000c8 000000c8 00000001 0000000a   ← [ndim=4, 200, 200, 1, 10]
offset 0x14: 1f8b 08 00 ...                                  ← gzip 成员
```
  ⇒ gzip 解压后 **raw = 800,000 B = 400,000 voxel × 2 B**，与 `blockSize [200,200,1,10]` **精确相符**（3/3 片）。
  ⚠️ **口径警示**：`attributes.json` 写 `compression{type:gzip, **useZlib:false**}`，但**直接按 raw-DEFLATE 解压会失败**
  （`invalid stored block lengths`）——必须**先跳过 20 B 前缀再 gzip 解压**（本轮踩到，已固化在 `--verify-subset`）。
- ⚠️ **量纲待核（登记为 `NEEDS_VERIFY`）**：同一片字节按 `int16` 解释 ⇒ 均值 ≈ **−105.7**、**49.96% 为负**、
  含 12 个 `−32768`；按 `uint16` 解释 ⇒ 均值 ≈ **32,637**、min 0 / max 65,343。
  **两解均自洽**（对称分布在两种解释下都成立）⇒ 本节点**不裁决**，但 **P2 定稿前的 ΔF/F 反演必须核对该口径**
  （否则基线/正负号全错——M9/M10 "量纲自检"教训同型）。原始荧光量级（uint16 解）与 2P 光子计数一致，**倾向 uint16**，
  但 `dataType` 字段声明 int16，故**以 `NEEDS_VERIFY` 入档**。

### L11.4 原始扫描（Range 只读头，**未下载 70 GB 大档**）

`two_photon_functional_scans/functional_scan_17797_4_10_v1.tif`（69.9 GB）`Range: 0-65535` → **206**，
BigTIFF 头解析（`data/m11_raw/microns/functional/tif_head_17797_4_10_v1.bin`）：

| 标签 | 值 |
|---|---|
| magic | `II+\0`（**BigTIFF**），首个 IFD @16，17 个条目 |
| ImageWidth × ImageLength | **248 × 440** |
| BitsPerSample / SampleFormat | 16 / 1（uint） |
| ImageDescription | **`{"shape": [320000, 440, 248]}`** ⇒ **320,000 帧** |
| StripOffsets / StripByteCounts | 512 / **218,240**（= 440×248×2 ⇒ 一 strip = 一帧） |
| Software / DateTime | `tifffile.py` / `2019:05:19 19:23:52` |

⇒ **帧维可核**：单扫描 32 万帧；`functional_series` 的 4 万帧 = 其 1/8（**时间降采样 8× 的合理读数，但本轮未证**，
登记为待核项）。**注意**：不同 scan 的 tif（32 万帧）与 `functional_series`（4 万帧）**不是同一个 scan**
（scan 10 vs scan 7），**不得把两者直接相除当作降采样倍数**。

### L11.5 技术文档（全文落盘 + 正文抽句）

| 文件 | 字节 | SHA-256 | 抽取正文 |
|---|---|---|---|
| `functional_data_technical_documentation_v3.pdf` | 149,979 | 已落 `m11_microns_protocol.json` | 24,930 字符 |
| `..._technical_documentation_v8.pdf` | 123,349 | 同上 | 20,763 字符 |

**文档原文锚（可逐句核对，用于采集参数 provenance）**：
1. **"4 px/µm xy resolution (440×248 px/field) to tile a 1100×1200 µm² FOV at four depths
   (two planes per depth, 40 µm overlap between coplanar fields)"**（v3，两处；另有
   "at three depths" 与 "6 px/µm … 1000×1200 µm²" 的**其它 session 版本**）；
2. **"the stimulus movie was linearly interpolated to a constant 30 Hz frame rate"**（刺激时序口径）；
3. DB schema 片段：**"session / scan idx / nframes / nfields / fps"** ⇒ 扫描级帧率在**数据库 Scan 表**内；
4. **"…the inter-frame period (1 / frame rate) to pixels in each field in acquisition order"**
   ⇒ **帧内像素的时间戳是"线性归属"近似**（对 P14 的 dwell/相关性口径有影响，必须登记）。

⚠️ **冲突（重要，已进 L20）**：文档写 **4 px/µm（0.25 µm/px）**，而 `functional_series` 的
`resolution` 写 **2.5 µm/px**（相差 **10×**）、FOV 写 620×1100 µm（文档写 1100×1200 µm²）。
**在 P2 定稿 dwell/率反演前必须解决该 10× 口径差**（两种可能：① series 是 xy 10× 降采样后的产物；
② resolution 字段单位/含义不同）。**本节点不裁决**，登记为 `NEEDS_VERIFY`。

---

## L12 — MICrONS 功能数据的**唯一权威通道 = v8 DataJoint MySQL 数据库**（P2 的结构性限制）

`two_photon_processed_data_and_metadata/access_v8/README.md`（8,148 B，实测 200）原文要点：

| 通道 | 体积 | 用途 | 本轮处置 |
|---|---|---|---|
| Docker 镜像 `functional_data_database_container_image_v8.tar` | **103,798,314,496 B（103.8 GB）** | 预装数据库（DataJoint/MySQL） | **未取**（超预算） |
| SQL dump `functional_data_database_sql_dump_v8.sql` | **116,199,864,342 B（116.2 GB）** | 手动导入（需 ~230 GB 空间，导入数小时） | **未取**（超预算） |
| Jupyter 环境镜像 `functional_data_notebook_container_image_v8.tar` | 3,091,041,280 B | 教程/notebook | 未取 |

⇒ **结论（对 M11 影响最大）**：
1. **逐细胞 ΔF/F、ROI 轨迹、扫描级 `nframes/nfields/fps`、瞳孔/跑道同步量**都在该 DB 内，
   **没有 REST/API 通道**（minnie CAVE 表 API 本轮实测 404/503 —— 见 L9.1）⇒ 本轮 `unavailable`（**不是**"未命中"）。
2. **M11 的 P2 可得路径**（本轮已实测可行）：**144 field 的 ROI 掩膜（`field_seg_masks`，compressed_segmentation，
   与 series 同格 248×440×1 @2.5×2.5×10 µm）+ 2 field 的原始钙电影（6.3001 Hz，各 40,000 帧）+ 刺激电影**。
   ⇒ **钙域 dwell/相关性/调谐参考量可在"场级"上同口径重建**，但**逐细胞轨迹需自行做 ROI 提取**（掩膜可解压即可）。
3. **若 M11 需要 DB 内的扫描级元数据/逐细胞轨迹 → 必须由主 agent 决策**：① 起一个**托管后台下载作业**取 103.8 GB 镜像
   （按 36–58 KB/s ⇒ **约 20–30 天**，不可行）或 116.2 GB dump（更慢）；② 换网络/镜像；③ 接受"场级重建"降级。
   **本节点的建议：走 ③**（并有 L11.2/L11.3 的实测口径支撑），把 DB 取数登记为 `unavailable + 原因`。

---

## L13 — Allen 侧实测（Cell Types 元数据 + per-cell NWB + BO Neuropixels 目录）

产物：`data/m11_allen_sources.json`、`data/m11_allen_celltypes_metadata.csv`（工具 `tools/fetch_m11_allen.py`）。

| 通道 | 实测 | 判定 |
|---|---|---|
| Cell Types API（小鼠 `ApiCellTypesSpecimenDetail`） | HTTP 200；**`total_rows = 1920`**；54 字段，其中 **11 个 `ef__*` 电生理特征** | **REACHABLE** |
| 元数据表落盘 | **1,920 / 1,920 行**（分页 5 × 400，`start_row`/`num_rows` 各占一个方括号；3.29 MB） | **measured（全取）** |
| `ef__avg_firing_rate` 分布（表内自算） | **n=1,808**（1920 行中 112 行该字段缺失 ⇒ 不补 0），median **17.44 Hz**、min 1.99 / max 261.2（全值见 `data/m11_data_counts.json`） | `measured`（**P8 类型级带的输入**） |
| per-cell NWB `well_known_file_download/664222041` | HEAD 200：**content-length = 91,560,414**，`filename="486940961_ephys.nwb"`；`Range:0-99` → **206** | **REACHABLE（object-level）** |
| NWB 头解析 | **44.7 MB 可见前缀** + **稀疏文件技巧**（`truncate` 到声明长度使 HDF5 超级块校验通过）→ **HDF5/NWB 打开成功**；状态 **`partial`**（`visititems` 越过前缀区即止：`free block size is zero`）。**实测读回**：`nwb_version = b'NWB-1.0.5'`（**NWB 1.x，非 2.x**）、`file_create_date = b'Wed Feb 28 16:57:03 2018'`、根组含 `acquisition / analysis / epochs / general / processing / stimulus`、`acquisition/timeseries/**Sweep_10**` 的属性为 `ancestry=[TimeSeries, PatchClampSeries, CurrentClampSeries]`、`help="Voltage recorded from cell during current-clamp recording"` ⇒ **per-cell 电流钳记录结构真实可读** | `measured(部分)` |
| BO Neuropixels NWB（`allen-brain-observatory` S3） | 有界列举 **569 个对象 / 853.7 GB**（未截断）；对象级 HEAD：`…/probe_810755797_lfp.nwb` → **200 / 2,062,950,784 B** | **REACHABLE（对象级）** |
| Allen synphys（`s3://allen-synphys`） | 桶列举 403 AccessDenied；对象级 HEAD 403 | **NEEDS_VERIFY → 回退为论文档**（清单 §1.7） |

**踩坑（已修正）**：Cell Types 的 `criteria` 含**空格/引号/方括号**，直接拼 URL ⇒ `curl exit=3 (URL malformed)`；
且 `num_rows=2000` 单次响应 >3.2 MB 会在慢链路上被 `--max-time` 截断（JSON 解析失败）
⇒ **必须 percent-encode + 分页 + 校验 `curl_exit`**（已固化进工具）。

---

## L14 — 文献 provenance 化（**C-6/C-8/C-5 由 `literature_unverified` 升级为 `literature`**）

产物：`data/m11_literature_priors.csv`（37 行）+ `data/m11_literature_priors.json`（含**逐行原始查询与命中**）。
工具：`tools/build_m11_literature_priors.py`（Europe PMC REST；声明标识用 `EXT_ID:` 精确复核 + **题录关键词硬核对**）。

| 统计 | 值 |
|---|---|
| 总行数 | 37（§1.3 文献先验 16 行 + §4.1 C-1..C-8 依据 14 行 + §4.5/§1.5 判据带输入 5 行 + §1.4/§1.2） |
| `literature` | **31**（题录经 Europe PMC 实测命中；**≠ 全文/参数值已核实**） |
| `literature_unverified` | **6** |

**关键结果（逐条可核）**：

1. **生效的"声明标识复核"**：清单 §1.3 的 12 个 PMID **全部题录核对通过**（关键词硬核对：
   `title_verified=yes`）——例：`11017176`="Cellular and network mechanisms of rhythmic recurrent activity
   in neocortex"、`17507551`="Inhibition determines membrane potential dynamics…"、`19396156`=
   "Driving fast-spiking cells induces gamma rhythm and controls sensory responses"。
2. **★ 清单"未命中"项的复核结果（本轮升级）**：
   - **C-6（腺苷 A1 慢反馈）**：清单标 `literature_unverified` → 实测命中
     **PMID 6170091**（adenosine/central synaptic transmission）与 **PMID 9157887**
     （"Adenosine: a mediator of the sleep-inducing effects of prolonged wakefulness"）⇒ 升为 `literature`；
   - **C-8（稳态可塑性）**：命中 **PMID 14735113**（Turrigiano & Nelson, "Homeostatic plasticity in the
     developing nervous system"）⇒ 升为 `literature`；
   - **C-5（丘脑-皮层慢振荡，Steriade 类）**：命中 **PMID 8235588**（"Thalamocortical oscillations in the
     sleeping and aroused brain"）⇒ 升为 `literature`。
3. **★ 清单 §1.3 行14 的"题录未命中"真因已查明（纠错）**：清单查询式
   `TITLE:"Survey of spiking in the mouse visual cortex"` 在 Europe PMC **hitCount = 0（本轮复现）**；
   真因是**论文标题写的是 mouse visual `system`（不是 cortex）** ⇒ 换查询式后命中
   **PMID 33473216**（"Survey of spiking in the mouse visual system reveals a functional hierarchy"）。
   ⇒ **不是"不存在"，是清单查询式用词错**（M11 应改用该 PMID）。
4. **清单 §1.4 的"IBL 论文 PMID 待 G2 落盘"已落盘**：**PMID 34011433**
   （"Standardized and reproducible measurement of decision-making in mice"）。
   附带：Europe PMC 里还有一条 **Correction（PMID 36301084）**——**引用时必须区分正文与勘误**（本轮踩到）。
5. **仍未命中的 6 行（保留 `literature_unverified`，逐行写明查询式与 hitCount）**：
   ① §1.3 行14（上述清单原查询式，hitCount=0）；
   ② **C-8-b**（"synaptic scaling time course" 发现式查询首条为无关条目 ⇒ 参数值不得引用）；
   ③④⑤ **D-1/D-2/D-3**（小鼠 V1 自发率带、静默比例、OSI 分布：发现式查询首条与主题不符
   ⇒ **判据带不得以文献为依据**，必须走 MICrONS/Allen **同口径实测量**）；
   ⑥ **D-5**（M11 的 dwell 目标带**只能是数据侧产物**，无文献可代）。
6. **诚实性纪律**：`literature` 只表示"题录可核查"；**参数值是否在摘要中出现**由 `value_in_abstract`
   列标注（如 §1.3 行8 的 "0.3–3 s" 量级 = `no`，即**摘要未出现 ⇒ 全文未读 ⇒ 不得声称已核实**）。

---

## L15 — 本轮新增工具与产物索引（全部**新建**，冻结件零修改）

| 工具（`neural_exploration/tools/`） | 作用 | 复跑命令 |
|---|---|---|
| `probe_m11_sources.py` | 51 清单源 + 17 补测源可达性（两次探测 + 三次多数裁决 + 硬超时） | `PYTHONHASHSEED=0 ./.venv-m11/bin/python neural_exploration/tools/probe_m11_sources.py --all` |
| `fetch_m11_microns.py` | 结构 info + shard 清单/对象探测 + 功能元数据 + 采集参数读回 + 有界子集 + 子集解压验证 | `… --struct --shard-probe` / `… --func-meta` / `… --func-subset` / `… --verify-subset` |
| `fetch_m11_allen.py` | Cell Types 元数据表（1920 行）+ per-cell NWB（HEAD/Range/头解析）+ BO Neuropixels 目录 | `… --all` |
| `build_m11_literature_priors.py` | §1.3/§4.1 逐条 Europe PMC provenance 化 | `… ` |
| `build_m11_inventory.py` | 落盘清单 + 权威计数锚（SHA-256 逐件重算） | `… ` |
| `validate_p11_data_gate.py` | P1/P2 断言 → `reports/neuro/m11_data_gate_summary.json` | `… [--determinism-check]` |

| 产物（新建） | 说明 |
|---|---|
| `data/m11_source_reachability.csv` | 68 源逐行实测（url/http_code/两次码/bytes/ct/verdict/note） |
| `data/m11_microns_structure.json` | em/seg/seg_m1300 info 解析 + shard 有界实测 + 对象级 Range 证据 |
| `data/m11_microns_protocol.json` | 采集参数读回 + 掩膜格 + BigTIFF 头 + 技术文档正文锚 |
| `data/m11_allen_sources.json` / `data/m11_allen_celltypes_metadata.csv` | Allen 证据 + 1920 行元数据 |
| `data/m11_literature_priors.csv` / `.json` | 37 行 provenance（含未命中逐行说明） |
| `data/m11_functional_subset_manifest.csv` / `_summary.json` / `_verify.json` | 200 片 / 136.3 MB / 逐片 SHA + 解压验证 |
| `data/m11_data_inventory.csv` / `data/m11_data_counts.json` | 权威清单（224 件）+ 分层计数锚 + `unavailable` 登记 |
| `reports/neuro/m11_data_gate_summary.json` | P1/P2/G2 判定（四态） |
| `docs/m11_data_gate_report.md` | G2 交付报告（判定表/证据级别/限制/复跑/墙钟） |
| `data/m11_raw/**` | 原始落盘（结构 info、功能元数据、PDF、TIFF 头、子集帧、NWB 前缀/稀疏文件） |

---

## L16 — 冻结件零修改核查（清单 §0.3 #11）

- **本节点写过的文件 = 全部为 M11 命名空间**：`docs/m11_*`、`data/m11_*`、`data/m11_raw/**`、
  `tools/*m11*.py`、`reports/neuro/m11_*` ⇒ `validate_p11_data_gate.py` 的 **P1.6 断言**对清单内 224 件逐件校验（`pass`）。
- **唯一例外（清单允许）**：`docs/生物仿真M11实施清单.md` 的 **§0.7 #2 改写**（RF-M11-1，见 L18）。
- ⚠️ **须澄清**：`git status` 会显示 `data/m10_*`、`docs/m10_env_notes.md` 等**处于 modified 状态**——
  **那是并行运行的 M10 节点在该仓库里的写入，不是本节点所为**。证据：本轮全部工具源码里**没有任何**
  非 M11 路径的写操作，且 `git diff` 的 m10 变更内容（P6 阶梯/慢变量）与本工作包无关。
  ⇒ **M10 节点请注意**：本节点未触碰任何冻结件；若需要"冻结件零修改"的独立取证，可对 `m10_*` 的
  `git diff` 与本文件的工具清单交叉核对。

---

## L17 — RF-M11-1 落实记录（**判据级，G2 前**）

- 对象：`docs/生物仿真M11实施清单.md` **§0.7 #2**。
- **改动行号**：原 **L220–L222（3 行）** → 改写后 **L220–L227（8 行）**；文档总行数 **1226 → 1232（+6）**。
- 内容：**删除**失效依据"M9 全规模数值 +28% 不可复现"（已被 `docs/m9_report.md` §11.1 的独立探针否证，
  同入口重跑 4.678288 Hz vs 记录 4.6783 Hz，偏差 2.6×10⁻⁶），**换成两条仍然成立的事实**：
  ① **"可复现 ≠ 正确"**——同口径只修 CSR 投递公式 ⇒ 3.5225 Hz（**−24.7%**）、静默 0.585→0.762，
  两个独立修正实现交叉一致到 1.7% ⇒ **M9 绝对值是"缺陷档"**；
  ② **跨物种/跨制备/跨口径不可移植**（果蝇/斑马鱼/小鼠 V1 的速率、相关、dwell 及量纲口径均不同）。
  **禁令保留**（M11 判据带只能来自 (a) MICRONS/Allen 同口径实测量 或 (b) 逐条 provenance 化文献）。
- 改动范围核查：改写完成时 `git diff --stat` = **1 file changed, 9 insertions(+), 3 deletions(-)**（仅此一处，其余未动）。
- ⚠️ **提交状态说明（本节点不提交 git，主 agent 负责）**：该改写已由主 agent 的
  `b0996c1 chore(M10/M11): 收口中间产物入库…` 提交入库 ⇒ 此后 `git diff` 对清单文件为空是**正常**的；
  核对方式 = `git log -1 -- neural_exploration/docs/生物仿真M11实施清单.md` +
  `git show HEAD:neural_exploration/docs/生物仿真M11实施清单.md | grep -c RF-M11-1`（实测 **1**），
  且文件当前行数 **1232**（原 1226，+6）。同理，本节点本轮新建的 M11 产物亦已由主 agent 的
  `fe7ddeb chore(M10/M11): 收口产物入库 + gitignore 大体积轨迹/原始数据` 一并入库
  （其中 `data/m11_raw/` 被 `.gitignore` 排除，仅本机留存 ≈280 MB）。

---

## L18 — RF-M11-2 的实测输入（dwell 带｜**本节点给输入，不在本工作包定稿**）

RF-M11-2 要求：M11 定稿 dwell 带前必须做"目标数据劈半 + 同协议等价检验"的**边界可达性自检**，
并按实测倍数放宽。本节点交付的**可用输入**（全部 `measured`）：

| 输入 | 实测值 | 对 dwell 带的含义 |
|---|---|---|
| 采样率 | **6.300006 Hz**（帧间隔 0.15873 s） | 帧间期 = **158.73 ms** ⇒ **0.3 s 级状态只有 ~1.9 帧**；dwell 的**下限可分辨性**受此限制 |
| 单 field 记录时长 | **6,349.2 s（105.82 min）** | 足够采样**最长状态**（数十秒级）的分布尾部；劈半后每半 **52.9 min** ⇒ 劈半自检统计量充足 |
| 可用 field 数 | **2**（`session4_scan7_field4` / `session5_scan3_field1`） | A⊥C 同源拆分**物理可行**（另一 field 作 held-out） |
| 掩膜 | 144 field 的 ROI 掩膜，格 = **248×440×1 @2.5×2.5×10 µm**（与 series 同格） | 可做 ROI 级/群级轨迹与 2 状态 HMM |
| **待核口径（会直接改变 dwell）** | ① `int16 vs uint16`（L11.3）；② `4 px/µm vs 2.5 µm/px` 的 **10× 差**（L11.5）；③ 帧内像素时间戳为**线性归属近似**（文档原文） | **未解决前不得定稿 dwell 带** |

> **本节点的判断（供主 agent 决策，见 §④/⑦）**：P2 的**时间口径已读回且可复算**，
> 但**空间分辨率 10× 口径差**与**数据类型解释**两项**必须先核对**，否则同一段钙电影的
> ΔF/F 基线、率反演与 OSI 都会偏（M9 B2 / M10 L12 的"口径纪律"同型）。
> ⇒ **P2 已足以支撑"建模/取块/时空口径"设计，但尚不足以"定稿 dwell 带"**（需补 2 项核对）。

---

## L19 — 与清单实测冲突项（**先改清单再改代码**；供主 agent 裁决）

| # | 清单原文 | 本轮实测 | 建议处置 |
|---|---|---|---|
| 1 | §1.6 #13 教程页 200 / **57,441 B** | 200 / **81,635 B** | 更新清单字节数（网页已变） |
| 2 | §1.6 #14 门户 >196 KB 且 12 s 未传完（`curl 28`） | 200 / **202,913 B / 4.05 s 传完** | 改记为"可达且可传完（速度波动）" |
| 3 | §1.6 #15 ngl 200 / **425 B** | 200 / **239 B** | 更新字节数 |
| 4 | §1.6 #17 minnie CAVE 表 API **404** | `/info/api/v2/datastacks` = **404**；`/materialize/api/v2/datastack/minnie65_public/version` = **404**（首次探测 503，三次裁决） | 补记 materialize 路径与 503 抖动 |
| 5 | §1.6 #18 `global.daf-apis` 302 后 **13 s 超时** | 两次均 **302 / 0 B**（父进程硬超时兜住） | 改记为"302 → 挂起（不可用）" |
| 6 | §1.6 #19/#20 判定 `REACHABLE_NO_DATA` | 403 **AccessDenied**（非 NoSuchBucket）⇒ 应为 **`NEEDS_VERIFY`** | 判定词表按"桶存在、对象级未证实"归位 |
| 7 | §1.6 #33 synphys 门户 **TLS 自签 / -k→403** | **三轮不一致**：首轮挂死（925.9 s）/ 次轮 404 + 129 KB / **末轮连接超时（8 s connect timeout）** | 末轮判 `BLOCKED`（间歇不可达）；**注意 curl `--max-time` 失效坑（L9.3a）与跨轮不稳定登记（L9.3b）** |
| 8 | §1.6 #46 Zenodo 判定 **`REACHABLE_INTERMITTENT`** | TLS 自签（curl 60）⇒ 词表外的判定 | 统一到 6 词表：`BLOCKED`（note 写 TLS 自签） |
| 9 | §1.1.1 只列 em/seg/seg_m1300 的 GCS/S3 两个 info | **S3 镜像没有 `seg_m1300/info`（404 NoSuchKey）**；且两镜像 scale 命名/编码不同（`8x8x40`+raw vs `8_8_40`+jpeg） | 清单补"镜像不等价"警告 |
| 10 | §1.1.2 "MICrONS 采集参数**本轮未读回** ⇒ 一律 `unverified`" | **已读回**：帧率 **6.300006 Hz**、帧数 40,000、时长 105.82 min、体素 2.5×2.5×10 µm、FOV 620×1100 µm、掩膜格 248×440×1 | **升级为 `measured`**；但**同时新增两项待核**（L11.3 int16/uint16、L11.5 10× 分辨率差） |
| 11 | §1.1 "in vivo 体积 1.3 × 0.87 × 0.82 mm³" | em 卷 bbox **1.704×1.442×0.524 mm** / seg **1.539×1.048×0.520** / seg_m1300 **1.751×1.291×1.114** | 三者与论文体积**互不相同**，引用必须写明卷别 |
| 12 | §1.1.2 目录清单（9 前缀） | 实测 **14 子前缀**：另含 `stack_rgb/`、`stimulus_n5/`、`two_photon_structural_stacks/`、`digital_twin_properties/`、`foundation_model/`、`access_v8` 等 | 补全目录语义 |
| 13 | §1.3 行14 `TITLE:"Survey of spiking in the mouse visual cortex"` hitCount=0 | 复现 hitCount=0；**真因 = 论文标题是 "…mouse visual system…"**；命中 **PMID 33473216** | 清单改用 PMID 33473216 |
| 14 | §4.1 C-5/C-6/C-8 标 `literature_unverified` | 三者**均已命中**（8235588 / 6170091+9157887 / 14735113） | 升级为 `literature`（保留"参数值仍需全文核对"的 caveat） |
| 15 | §1.4 "IBL 论文 PMID 待 G2 落盘" | **PMID 34011433**（另有 Correction **36301084**） | 落盘，并注明区分勘误 |
| 16 | §1.2 "Allen Cell Types API 查询串" | 该 criteria **含空格**，直接放 URL 会被 curl 拒（exit=3）⇒ 必须 percent-encode | 清单补该坑（工具已固化） |
| 17 | §1.1.2 "功能序列分片 ~675–695 KB/片" | 实测 **665–700 KB/片**（`blockSize=[200,200,1,10]` ⇒ **10 帧/片**） | 补"10 帧/片" |
| 18 | 清单未提 | **chunk 文件 = 20 B 头 + gzip**（`useZlib:false` 是误导项；raw-deflate 解压会失败） | 补进 §1.1.2 并写进 P2 读取约定 |
| 19 | 清单 §1.7 P2 验收含 `m11_tuning_reference.csv`/`m11_dwell_reference.csv`/`m11_equivalence_protocol.csv` | 本工作包**未产出**（属 P2 的"参考量重建"步，且依赖第 10/17/18 项口径核对） | **登记为未交付**，见 `m11_data_gate_report.md` §限制 |

---

## L20 — 预算消耗实测（对照清单 §10.1）

| 项 | 清单上限 | 本轮实测 |
|---|---|---|
| §1 数据门 CPU | ≤40 CPU-h | **≈0.5–0.8 CPU-h**（纯网络 I/O + SHA；全部单线程 `OMP_NUM_THREADS=1`） |
| GPU | 0 | **0**（未起任何 GPU/MPS 作业——纪律遵守） |
| 下载量（**新增**） | 单次 ≤2 GB、总计 ≤6 GB | **≈0.21 GB**：功能子集 **136.3 MB** + NWB **44.7 MB** + NWB 8 MB 前缀 + 元数据/文献/探测 **≤20 MB** |
| 墙钟 | 6–12 天（人日） | **≈4.0 h**（08:09→12:15 UTC）：探测首轮 18.9 min + 重跑 33.9 min + 结构/功能/子集 ≈29 min + Allen ≈4 min + 文献/清单/判定 ≈10 min + **NWB 停滞等待 ≈50 min** |
| 并发纪律 | 禁止并行大作业 | **两次主动串行化**：① 暂停 NWB 让子集完成（L9.3c）；② NWB 停滞 48 min 未增长 ⇒ **停掉并改为交付"前缀 + header 解析"**（L21#4） |

---

## L21 — 未测 / 未完成登记（**`unavailable`，不得写成 0 或"未命中"**）

1. **MICrONS 逐细胞 ΔF/F 与 ROI 轨迹**：仅存在于 **v8 DataJoint DB**（103.8 GB 镜像 / 116.2 GB dump）
   → 本轮**未取**（超预算）。回退：用 144 field ROI 掩膜自建 ROI 提取（可行，见 L12.2）。
2. **全 144 field 的扫描级 `fps/nframes/nfields`**：在 DB `Scan` 表内 → 未取；
   本轮只读回**2 个 field** 的时间口径（6.300006 Hz）。
3. **多深度体积的 volume rate**：`functional_series` 为单平面（z=1），文档述"two planes per depth × 2–4 depths"
   ⇒ **整卷体积率不可由本轮数据推出**（需 DB 的 `nfields/fps`）。
4. **Allen per-cell NWB 的完整内部结构**（`general/`、`analysis/`、全部 Sweep 的 shape/轨迹长度）：
   全量 = **91,560,414 B**；本轮实测下载到 **44,695,552 B（48.8%）后停滞 48 min 无任何字节增长**
   （进程存活、`curl -C -` 续传），**故主动停止**（用户纪律：预计 >30 min 的下载先汇报，不硬上）：
   - 已交付：HEAD（`content-length` 精确）、Range 0-99（206）、**44.7 MB 前缀 + 稀疏文件头解析**
     （`NWB-1.0.5` / `file_create_date` / 根组名 / `Sweep_10` = `CurrentClampSeries`）；
   - **未交付**：完整内部结构（细胞/轨迹/刺激表）⇒ `unavailable` + 原因（网络停滞），**不写成未命中**；
   - 续传命令：`curl -C - -o 486940961_ephys.nwb https://api.brain-map.org/api/v2/well_known_file_download/664222041`，
     到位后重跑 `fetch_m11_allen.py --nwb` 即自动改用全量文件解析（工具已按"最大可见前缀"选择）。
5. **Allen synphys 的对象级数据**：桶 403（不可 list）、对象级 HEAD 403 ⇒ `NEEDS_VERIFY`；
   C-2/C-3 的突触动力学参数**只能走 Campagnola 2022（`literature`）**，**不得**声称 `measured`。
6. **IBL 行为数据**：#35 `NEEDS_TOKEN`（403 需凭据）；公开桶可 list（#36/F15）但**未取对象**。
   P12 若不提供 token → 走公开桶 `aggregates/` + 论文档（清单 §1.4 回退）。
7. **MICrONS 表级 API（CAVE）**：#17 404 / F1 404(503 抖动) / #18 挂起
   ⇒ 清单 §1.7 的**首选回退**（`functional_connectomics/node_and_edge_properties`）**已实测可达**
   （F17：v1 前缀 200 / 4 键），但**未取对象内容**（属 P4/§3 范围）。
8. **MICrONS 结构-功能同源映射表**（`functional_coregistration`）：前缀可达（200 / 830 B，2 键），
   但**映射表内容未取**（属 P2/H4 后续）。

---

## L22 — 教训沉淀（可复用到 M11 后续与本项目全部数据作业）

1. **"可达"必须分三层证据**：桶可 list ≠ 对象可取 ≠ **内容可解释**。本轮把第三层也做了
   （子集**解压 + 字节数精确匹配** + BigTIFF 头解析），因此发现了两处"清单没写但会毁掉判据"的口径坑
   （20 B 前缀；int16/uint16 歧义）。**建议：M11 之后所有数据门的 Pass 语义都要求"内容级证据"**。
2. **父进程硬超时是必需的**：`curl --max-time` 在实测中被观察到失效（925.9 s）⇒ 一律 `Popen + kill`。
3. **`--only` 类子集运行不得覆盖全量产物**：本轮踩到一次（子集运行把 68 行 CSV 覆盖成 0 行），
   已加**拒绝执行**保护（只在显式 `--out` 时允许子集运行）。**同类保护应进 M11 其它工具**。
4. **两次/三次探测 + 多数裁决**：单次探测的瞬态失败（如 S3 桶列举 pass1 失败/pass2 成功）会污染判定表；
   三次裁决把不稳定源如实标注而非误判。
5. **"字典型"发现式文献查询必须加"题录主题核对"**：按被引数排序的宽查询首条常为**领域综述/无关条目**
   （本轮 D-1/D-2/D-3 即如此）⇒ 若不核对，会把"检索集有命中"误记为"参数有依据"。
6. **引用必须区分正文与勘误**（IBL 论文 + Correction）。
7. **体积/边界必须写清是哪一版重建**（em / seg / seg_m1300 三卷体积互不相同，且都不等于论文 in vivo 体积）。
8. **DB-only 数据集必须先算"取数天数"再决策**：103.8 GB ÷ 36 KB/s ≈ 20–30 天 ⇒ 结论是"结构上不可取"，
   而不是"我们没有努力"。

---

## L23 — ★ 新发现：MICrONS **功能连接组学发布**（结构-功能同源**逐细胞/逐对**表）

探测 **F17** 时发现 `functional_connectomics/node_and_edge_properties/v1/` 是一个**小而关键的发布**：

| 对象 | 字节 | 本轮处置 |
|---|---|---|
| `readme_v1.md` | **5,594** | **已落盘**（`data/m11_raw/microns/functional_connectomics_readme_v1.md`） |
| `node_data_v1.pkl` | **294,635,032（294.6 MB）** | **未取** —— 交主 agent 裁决（DQ-4） |
| `edge_data_v1.pkl` | **201,511,086（201.5 MB）** | **未取** —— 同上 |

**readme 原文给出的 schema（逐列可核）**：
- `node_data`（**逐神经元一行**）：`nucleus_id / animal_id / scan_session / scan_idx / unit_id`、
  **EM 坐标** `nucleus_x,y,z`(nm)、`hva`（V1 vs HVA）、`brain_area`（HVA 细分 RL/AL/LM）、**`layer`（皮层分层）**、
  **功能量**：`pref_ori`、`pref_dir`、**`osi_iv` / `gosi_iv`（in vivo 取向选择性）**、`in_vivo_mean_resp`、
  `mei`（最大兴奋图）、`mask`、`readout_cvt` / `position_stim_cvt` / `sta_mu_comb`（CvT 数字孪生）、
  `cc_*_cvt`（孪生模型性能）。
- `edge_data`（**逐神经元对一行**）：`pre/post_nucleus_id`、`population`（`Connected` / `ADP` / `Same region`
  —— **即"真连接 / 距离匹配对照 / 同区对照"**）、**`n_synapses`（突触数）**、`synapse_size`（裂隙体积）、
  `dend_len`、`pre/post_hva`、`pre/post_layer`、`proj_hva(_layer)`、
  **`in_vivo_sig_corr_iv`（in vivo 信号相关）**、`in_silico_sig_corr_cvt`、`delta_ori_iv`、…。
- 出处（readme 原文）：**Ding et al. 2025, DOI 10.1038/s41586-025-08840-3**，
  "Functional connectomics reveals general wiring rule in mouse visual cortex"，代码 `cajal/microns-funconn-2025`。

⇒ **对 M11 的直接影响（重要）**：
1. 这是**唯一实测找到的"同源结构-功能逐细胞表"**（EM 坐标 + 分层 + 区属 + in vivo 调谐/反应 + 逐突触对数），
   直接服务 **H4（结构-功能映射）**、**P7（OSI/调谐分布同口径）**、**P8（层/区差异）**、**P14（群体相关）**，
   并且 `population` 列**自带"距离匹配"与"同区"对照**——正是 M11 需要的**负对照**材料。
2. 体积 **≈496 MB（两表）**，在"单次 ≤2 GB / 总计 ≤6 GB"内；但按项目纪律（>30 min 需先汇报），
   本节点**未启动**该下载 ⇒ 登记 **DQ-4**（见 L24）。
3. 该发布**不在清单 §1.6 的 51 源内**（清单 §1.7 只泛泛提到"conn 属性衍生表"）⇒ 建议补入清单 §1.6/§1.7。

---

## L24 — 交主 agent 的四个决策请求

| # | 请求 | 影响 | 建议 |
|---|---|---|---|
| **DQ-1** | 是否允许"**不取 v8 DataJoint DB**"，改走**场级重建**（144 ROI 掩膜 + 2 field 钙电影）？ | 决定 P2 的 `measured` 范围与 P7/P14 的逐细胞口径 | **建议允许**（DB 取数需 20–30 天，等于停工） |
| **DQ-2** | 是否由主 agent 起**托管后台作业**继续 NWB 全量（91.5 MB，约 40–50 min）并做完整头解析？ | 决定 Allen per-cell 通道的证据级别（partial → full） | **建议：让现有作业跑完**（已续传，零额外成本） |
| **DQ-3** | L19 的 19 条冲突项是否按"新增行 + 保留原行"逐条落入清单 §1.6/§1.1？ | 判据带定稿纪律（运行前定稿） | **建议逐条落入**，并给 `m11_microns_protocol.json` 加"待核口径"指针（L11.3/L11.5） |
| **DQ-4** | 是否为 **L23 的两个 pkl**（node 294.6 MB / edge 201.5 MB，合计 ≈496 MB）起**托管下载作业**？ | 决定 P2 能否**同源同口径**算出 C1（调谐/OSI）与 C2（群体相关），并把 P2 从 `partial_pass` 推向全过；亦直接影响 P7/P8/P14 的对照材料 | **建议批准**（在预算内；按 200–350 KB/s 估 ≈25–45 min，建议**分段 + 断点续传 + 增量日志**；本节点因 >30 min 纪律**未自启**） |
