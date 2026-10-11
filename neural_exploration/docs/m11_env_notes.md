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
| 20 | 清单 §1.1.2 未规定 chunk 的**字节序与头前缀**（只说"分片化 precomputed 体数据"） | 实测 = **20 B 大端 uint32 头 `[4,200,200,1,10]` + gzip**，数据体 **大端 int16**；主 agent 与本节点首版均因**小端**误读得出"不像图像"的假象 | **补入清单读取约定**（已落 `m11_reference.csv` R-08/R-09/R-10） |
| 21 | （清单未涉及，本节点自身缺陷留档）| 首版子集判定未校验 `curl_exit` ⇒ 3 片半成品被误记 `REACHABLE`（主 agent 独立复核发现） | 已修复 + 固化为独立**完整性门**（L25.4）；**统计只用 COMPLETE 对象** |

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
9. **Allen `units.csv`（51,439,189 B）/ `channels.csv`（6,613,329 B）全量**：网络 16 KB/s ⇒ 预计 >30 min ⇒
   **未取**（登记 `unavailable`，DQ-6 裁决）；已取 1 MB Range 前缀读回 schema。
10. **Allen LFP NWB 全量**（2,062,950,784 B）：**超过"单次 ≤2 GB"上限** ⇒ 合规地不下载，只做 Range 头解析。

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
9. **★ 吞吐必须"当场实测"再决策（本轮血的教训）**：同源 S3 在数小时内由 **194–349 KB/s → 16 KB/s**，
   DANDI 路由 **7.2 KB/s**，`api.brain-map.org` **1.6 KB/s** ⇒ 任何"按早前吞吐推算的工期"都会失效；
   本轮据此**主动放弃**了 3 个 ≥50 MB 的下载（units.csv / channels.csv / NWB 全量）并登记原因。
10. **★ macOS `ps %CPU` 是近期采样，不是累计值（主 agent 实测教训，代为入档）**：判断作业进度应看
   **"累计 CPU 时间 / 墙钟"** 之比（主 agent 实测 4 个作业 51 min 仅 4:35 累计 CPU ⇒ 占空比 ≈10%，
   后台 QoS 饥饿）。**I/O 型作业不看 %CPU**；CPU 型作业**先算占空比再判快慢**。
   本节点的解析作业以**墙钟**计量（200 对象 / 137 MB 解析门 ≈7 s 墙钟），未受此影响。
11. **★ "稀疏文件"必须写明是稀疏**：`head_sparse.nwb` 表观 91,560,414 B，**真实数据仅前 44,695,552 B**，
   其余为洞（读作 0）⇒ 若误当成"完整 NWB"会得出错误的可读性结论。凡用 `truncate` 造稀疏文件，
   **必须在文件名/note/清单三处标注**（本轮已在 `m11_data_counts.json::disk_verified` 标注 `header_staging`）。

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

## L24 — 交主 agent 的六个决策请求

| # | 请求 | 影响 | 建议 |
|---|---|---|---|
| **DQ-1** | 是否允许"**不取 v8 DataJoint DB**"，改走**场级重建**（144 ROI 掩膜 + 2 field 钙电影）？ | 决定 P2 的 `measured` 范围与 P7/P14 的逐细胞口径 | **建议允许**（DB 取数需 20–30 天，等于停工） |
| **DQ-2** | 是否由主 agent 起**托管后台作业**继续 NWB 全量（91.5 MB，约 40–50 min）并做完整头解析？ | 决定 Allen per-cell 通道的证据级别（partial → full） | **建议：让现有作业跑完**（已续传，零额外成本） |
| **DQ-3** | L19 的 19 条冲突项是否按"新增行 + 保留原行"逐条落入清单 §1.6/§1.1？ | 判据带定稿纪律（运行前定稿） | **建议逐条落入**，并给 `m11_microns_protocol.json` 加"待核口径"指针（L11.3/L11.5） |
| **DQ-6** | 是否托管取 **Allen `units.csv`（51,439,189 B，逐 unit 质量表）** 与 **`channels.csv`（6,613,329 B）**？ | 决定 P7/P8/P14 的**单元级电生理带**能否 `measured`（现仅有 schema）；网络 16 KB/s ⇒ 预计 >30 min，本节点按纪律未自启 | **建议托管**（或换网/镜像）；不取则 P7/P8 的类型级带仍走 `literature` |
| **DQ-5** | 是否把 **L25.2 的读取约定**（20 B 大端头 + 大端 int16 + 无掩膜 + 完整性门）与 **L26 的 6.30 Hz 读回断言**按"新增行"写入清单 §1.1.2/§1.5？ | 决定后续节点是否会重复"小端误读 ⇒ 判据全错"与"半成品入统计"两类事故 | **建议写入**（本节点已把三者做成可复跑的门：`verify_m11_downloads.py` / `validate_m11_microns_parse.py` / `build_m11_reference.py`） |
| **DQ-4** | 是否为 **L23 的两个 pkl**（node 294.6 MB / edge 201.5 MB，合计 ≈496 MB）起**托管下载作业**？ | 决定 P2 能否**同源同口径**算出 C1（调谐/OSI）与 C2（群体相关），并把 P2 从 `partial_pass` 推向全过；亦直接影响 P7/P8/P14 的对照材料 | **建议批准**（在预算内；按 200–350 KB/s 估 ≈25–45 min，建议**分段 + 断点续传 + 增量日志**；本节点因 >30 min 纪律**未自启**） |

---

## L25 — ★ 主 agent 独立复核的三项发现：**全部复现，根因已定位并门化**

> 起因：主 agent 用独立脚本解析 `data/m11_raw/microns/functional_subset/*.bin`，发现
> ① 正面确认（20 B 头 + gzip + 800,000 B + 6.30 Hz）② 直读"不像图像" ③ 2 片解压失败 ④ 检出 NaN。
> 本节点逐条复核 + 定位根因 + **把三件事做成独立可复跑的门**。

### L25.1 根因 = **字节序（小端误读）**，不是 delta、也不是 Morton

| 假设（预注册检验） | 横向相邻像素相关 | 相邻帧相关 | p50 | 值域 | 判定 |
|---|---|---|---|---|---|
| **① 直读 int16 _BE_（大端）** | **+0.801** | **+0.784** | 1,815 | −38 … 26,367 | ✅ **唯一自洽解** |
| ① 直读 int16-LE（**我首版与主 agent 复核都用的读法**） | −0.001 | +0.000 | −40 | ±32,768 | ❌ C1/C2 不过 |
| ① 直读 uint16-LE / uint16-BE | +0.005 / +0.673 | +0.005 / +0.654 | — | — | ❌/部分（见 L25.2） |
| ② delta（`cumsum` 各轴/扁平；含 mod 65536） | +0.94 … +1.00 | +0.92 … +1.00 | **−2.1×10⁷ … +4.1×10⁸** | 超 10⁸ | ❌ **C5 声明域约束**否定 |
| ③ Morton/Z-order 反排列（200×200，补齐 256×256 后取有效位） | +0.588 | +0.784 | 1,815 | 同 int16-BE | ❌ **相干性低于直读**（0.588 < 0.784） |
| ③+② Morton + cumsum | +1.000 | +1.000 | 4.1×10⁸ | 超 10⁸ | ❌ C5 否定 |
| float32（双端序）/ float16 / uint8 | ≈0 或 NaN | — | — | ±10³⁸ | ❌ |

**四条独立证据链（判据化）**：
1. **相干性**：大端直读使 `min(lat_x, frame)` 最大（0.784），任何重排/累积运算**只会破坏**空间相干；
2. **独立地面真值**：同一数据集的**原始未压缩 BigTIFF**（`functional_scan_17797_4_10_v1.tif`，另一个 scan，
   Range 读 4 帧）给出 **lat_x = 0.765 / lat_y = 0.735 / p50 = 1,879**，与分片解码值（0.777/0.761/1,863）
   **几乎一致** ⇒ 判据阈值 0.2 由真实数据锚定，**不是自证**；
3. **低于零的样本不是哨兵**（见 L25.3）；
4. **声明域约束**：`cumsum` 类使 p50 达 10⁷–10⁸，远超 `attributes.json` 声明的 int16 域。

⚠️ **判据教训（记入纪律）**：主 agent 给的 4 条判据（相邻像素/相邻帧相关、值域合理、无 NaN/Inf）
**对"累计和（cumsum）"这类退化结构天然通过**（cumsum 必然给出相邻强相关）——
若不加约束，"判据"就不具备判别性（**RF-M11-5 的同型教训**）。本节点因此**补两条抗伪影判据**并写入工具：
**C5 声明 dtype 域约束**（值必须落在 metadata 声明的 dtype 域内）、
**C6 非退化**（沿存储序一阶差分非负比例 ∈ [0.35, 0.65]，真实图像 ≈0.5、cumsum ≈1.0；且动态范围比 ≤50）。
补后通过者从 9 个降到 **2 个**，再按 **"相干性最大 + 更简单者优先"** 选出 **① 直读 int16-BE**。

### L25.2 位置：为什么必须是大端（可复核）
`attributes.json` **只声明 `dataType: int16`，未声明字节序**；本数据集为 **N5/BossDB 系（Java）惯例 ⇒ 大端**。
⇒ 建议清单 §1.1.2 的读取约定补一条：**"chunk = 20 B 大端 uint32 头 [ndim,x,y,z,t] + gzip 成员；
数据体为 _大端_ int16"**（已落入 `data/m11_reference.csv` R-08/R-09/R-10）。

### L25.3 NaN / 哨兵 / 缺失登记（主 agent 硬要求 §四）
全量 **200 个 COMPLETE 对象 / 80,000,000 元素**（`data/m11_microns_parse_validation.json`）：

| 量 | 实测 | 处置 |
|---|---|---|
| **NaN / Inf** | **0 / 0**（整型域**结构性不可能**出现） | 主 agent 检出的 **106 个 NaN** 与 **|x|>1e30（约 11%）** 是 **float 误解读的副产物**（本节点 float32 双端序假设行同判 `invalid`） |
| **int16 饱和（32767）** | **0** | 无饱和伪影 |
| 零值 | 193,266（**0.2416%**） | 登记；不掩膜 |
| **低于基线样本（<0）** | **10,383（0.01298%）**，幅度 −1…−138 | **保留**（掩膜会引入正偏） |
| 其中 uint16 读法真值 | 65,398 … 65,535 | **拒斥**（见下） |

**拒斥 uint16 的判决性证据（这是本轮最硬的一条）**：
- 这 10,383 个样本**空间成簇**：其 3×3 邻域内负值比例 p50 = **0.111**（若随机独立应 ≈0.000129 ⇒ 富集 **≈860×**），
  **孤立比例 = 0.000**（没有一个是孤立像素）；
- 它们位于**暗区**：邻域均值 p50 ≈ **395**，而全局中位 = 1,953；相对局部基线仅低 ~**381** counts；
- 若按 uint16 读，它们的真值 65,398–65,535 会是**对象自身最大值（p50 = 28,066）的 2.3–2.3 倍**，
  而其邻域只有 395 ⇒ 需要"孤立超亮点被暗背景包围"的物理荒诞解。

⇒ **掩膜规则（预注册）：无掩膜**；NaN/Inf=0、饱和=0、低于基线 0.013% 与零值 0.24% **逐项登记**
（"未处理 ≠ 无缺失"）。

### L25.4 完整性门（主 agent 硬要求 §三）：197/200 → **200/200**，且根因已修
- 主 agent 检出 2 片解压失败；本节点全量重扫为 **3 片**（`t_chunk` 61 / 120 / 122，
  132 KB – 661 KB，均为 `curl_exit=28` 的**半成品**）→ 已 `--refetch` 补齐 ⇒ **200/200 COMPLETE**。
- **根因（本节点自身缺陷，如实入档）**：首版子集判定只检查 `http_code ∈ {200,206} and bytes>0`，
  **未校验 `curl_exit == 0`**，于是"超时留下的部分字节"被记成 `REACHABLE` —— 这正是项目反复强调的
  **静默失效**（M9 B3 / M10 L21 同型）。**已修复两处**：① 判定加 `curl_exit==0` + 就地 gzip/尺寸预检；
  ② 固化独立门 `tools/verify_m11_downloads.py`（http→大小→SHA-256→解压→尺寸自洽），
  **统计与参考带只允许使用 `COMPLETE` 对象**（解析门实际只用 COMPLETE 列表，断言 `P2.11e` 强制一致）。

---

## L26 — **6.30 Hz** 的判据级后果（RF-M11-2 由"应当"变"强制"）

- 读回断言值 **frame_rate = 6.300006 Hz**（帧间隔 0.15873 s；**10 帧/片 = 1.5873 s/片**），
  已作为 **`data/m11_reference.csv` R-01** 落盘（`asserted`），并被判定器 `P2.12a` 强制核对。
- 与 ZAPBench 的 **1.0943 Hz 相差 5.8×** ⇒ 沿用 M10 的 **26.5 s dwell / 37.18 s τ** 将发生
  **跨物种 + 跨帧率 + 跨箱宽** 三重错配。**故 RF-M11-2 的"自建 dwell/τ 带"是强制的。**
- **dwell/τ 参考带在 `m11_reference.csv` 中标 `unverified`**（R-21）且判定器 `P2.12c` 强制其为 `unverified`：
  解析门已过，但 ① C1/C2 参考量重建 ② "目标数据劈半 + 同协议等价检验"的边界可达性自检尚未完成
  ⇒ **未定稿前不得作为任何判据带**。
- 采样下限提醒：帧间期 158.73 ms ⇒ **0.3 s 级状态仅 ~1.9 帧**（dwell 下界可分辨性受限，须在定稿时写明）。

---

## L27 — 本轮新增工具与产物（主 agent 三项硬要求的落地件）

| 新增工具（`neural_exploration/tools/`） | 作用 | 复跑命令（`PYTHONHASHSEED=0 OMP_NUM_THREADS=1`） |
|---|---|---|
| `verify_m11_downloads.py` | **完整性门**（逐对象 5 步判定链；`--refetch` 重取） | `$PY neural_exploration/tools/verify_m11_downloads.py [--refetch]` |
| `validate_m11_microns_parse.py` | **结构性解析验证**（18 假设 × 6 判据 → 胜出解；含 NaN/哨兵/掩膜登记） | `$PY neural_exploration/tools/validate_m11_microns_parse.py --n-full 200` |
| `build_m11_reference.py` | **参考锚落盘**（`m11_reference.csv`，22 行，含 6.300006 Hz 读回断言） | `$PY neural_exploration/tools/build_m11_reference.py` |

| 新增产物 | 内容 |
|---|---|
| `data/m11_download_integrity.csv` / `_summary.json` | 200 对象逐行 5 步判定；`COMPLETE=200 INCOMPLETE=0` |
| `data/m11_microns_parse_validation.json` | 胜出假设、判据、选择规则、全量复核统计、NaN/哨兵/掩膜规则、地面真值 |
| `data/m11_microns_parse_hypotheses.csv` | 18 假设 × 6 文件 × 逐条判据/度量（**未通过者留档，不得用于统计**） |
| **`data/m11_reference.csv`** | 22 行参考锚：20 `asserted` / 1 `NEEDS_VERIFY`（10× 口径冲突）/ 1 `unverified`（dwell 带） |
| `data/m11_functional_subset_manifest.csv` | **新增 `integrity` 列**（200 行 = COMPLETE） |
| `data/m11_raw/microns/functional/tif_frames_probe.bin` | 原始 BigTIFF 4 帧前缀（独立地面真值；Range 512–2200000） |

---

---

## L28 — RF-M11-A（主 agent 裁定：体素格式不可解 ⇒ 功能通道回退）的处置

**处置结论：接受"不再攻关"与"回退源落地"两项；不接受把体素序列登记 `format_unverified`（理由与证据如下）。**

### L28.1 判定性实测：主 agent 的三批检验**全部是小端读法**，其"无结构"是**字节序假象**
本节点用**主 agent 自己的检验项**在同一文件上做 LE/BE 对照（`tools/demo_m11_microns_decode.py`，
产物 `data/m11_decode_proof.json` + `reports/neuro/m11_microns_decode_proof.png`）：

| 检验项（主 agent 采用） | `<i2`（小端，主 agent 与本节点首版） | **`>i2`（大端）** |
|---|---|---|
| 帧轴 lag1..6 自相关 | 0.0003 / −0.0009 / −0.0007 / −0.0003 / −0.0035 / 0.0028 | **0.803 / 0.783 / 0.761 / 0.743 / 0.725 / 0.709**（平滑衰减=钙成像时间结构） |
| 轴1(y) lag1..6 | ≈0（−0.0014…0.0013） | **0.801 / 0.654 / 0.572 / 0.516 / 0.484 / 0.468** |
| 轴2(x) lag1..6 | ≈0 | **0.814 / 0.676 / 0.593 / 0.531 / 0.489 / 0.464** |
| 横向相邻像素相关 | −0.001 | **+0.814 / +0.801** |
| p50 | **0** | **1,953** |
| 值域 | −32,768 … 32,584 | −41 … 29,796 |
| 唯一值 / 最高频次 | 8,971 / 401 (0.10%) | 8,971 / 401（**同一批字节 ⇒ 该项无判别力**） |

⇒ 主 agent 观测到的"逐轴自相关全 ≈0、p50=0、值域 ±32k 对称、近似均匀"**正是小端读法的指纹**：
把每个 int16 的 (hi,lo) 对调，原本的强空间-时间结构被打散为准随机序列。**这不是"格式不可解"。**

**四条独立证据（任一单独成立即可推翻 format_unverified）**：
1. **相干性**：BE 直读 lat_x=0.814 / 帧轴 lag1=0.803，判据（≥0.2）**远超**；LE 为 ≈0；
2. **独立地面真值**：同数据集**原始未压缩 BigTIFF**（另一 scan、Range 6 帧）lat_x=**0.767**、p50=**1,879**
   ⇔ 分片解码 0.777 / 1,863 ⇒ 阈值由真实数据锚定，非自证；
3. **Morton 被否**：本节点实现的是"补齐 256×256 后按 Z 序取有效位"（恰好 40,000，无越界），
   结果是 lat_x 0.588 < 0.784 ⇒ **Morton 只会降低相干性 ⇒ 数据不是 Z 序**；
4. **格式本身不需要官方客户端**：`20 B 大端头 [num_dims=4, x=200, y=200, z=1, t=10] + gzip 成员 + 大端 int16（x 最快）`
   —— 是 N5/BossDB（Java）系惯例；`pip install` 不可得**不影响**该读法自证（已用结构化判据门化）。

**风险对照（为何不能轻率登记 `format_unverified`）**：MICrONS 的"**同源**结构-功能"是 M11 相对 M10/M9 的
**唯一方法学升级**（H4 / P14 的核心）。若把功能体素序列判死，M11 的活动金标准将退化为**跨动物**的替代源，
H4 不可检验。⇒ 建议登记 **`decoded(big-endian int16)`**，并在文档保留
"主 agent 曾裁定 `format_unverified`，节点以 4 条证据否证"的**复核留痕**（已写入 `m11_reference.csv` R-26）。

### L28.2 回退源**已按 RF-M11-A §二.3.a 落地**（自描述格式，零解码风险）
`tools/build_m11_allen_activity.py` → `data/m11_allen_activity_fallback.json` + `m11_allen_activity_objects.csv`：

| 对象 | 状态 | **读回值** |
|---|---|---|
| `probes.csv`（**完整** 26,954 B，332 probe） | COMPLETE | **`sampling_rate` 均值 29,909.6 Hz / 最大 30,000.3 Hz（spike-band 30 kHz）**；**`lfp_sampling_rate` 均值 2,492.5 / 最大 2,500.03 Hz**；`lfp_temporal_subsampling_factor=2.0`；325/332 `has_lfp_data=True` |
| `sessions.csv`（**完整** 7,849 B，58 session） | COMPLETE | `session_type / specimen_id / age_in_days / sex / genotype / has_nwb` ⇒ **P14 的 session 级隔离材料** |
| `units.csv`（全量 **51,439,189 B 未取**） | RANGE_BOUNDED（1 MB 前缀） | **27 列 schema 读回**：`firing_rate / snr / isi_violations / presence_ratio / quality("good") / amplitude / d_prime / isolation_distance / silhouette_score / nn_hit_rate / nn_miss_rate / ecephys_channel_id` … ⇒ P7/P8 的电生理带与单元质量筛选表（**DQ-6**：是否托管取全量） |
| `channels.csv`（6,613,329 B） | 前缀（网络中断⇒INCOMPLETE，已登记） | 逐 channel 表（结构/层归属列在表头） |
| `probe_810755797_lfp.nwb`（**2,062,950,784 B > 2 GB 单文件上限**） | 前缀 Range（头部解析） | HDF5 自描述；**不下载全量**（合规） |

**纪律**：断言 `P1.8b` 强制"**INCOMPLETE/MISSING 对象不得进入统计**"；采样率统计**只**用本地大小 == 期望值的
`probes.csv`（`P1.8c` 强制）。⇒ 主 agent §三 的完整性门要求现已覆盖**全部新增下载对象**（不只功能子集）。

### L28.3 与 RF-M11-A §三 三条硬要求的对应
| 要求 | 状态 |
|---|---|
| 完整性门（逐对象 http→大小→sha256→解压/解析→尺寸自洽；失败者 INCOMPLETE 且不进统计） | ✅ **已覆盖全部对象**：功能子集 200/200 COMPLETE（`m11_download_integrity.csv`）+ 回退源逐对象登记（`m11_allen_activity_objects.csv`） |
| NaN/非有限值与哨兵登记 | ✅ **已完成**：NaN=0 / Inf=0（整型域）；10,383 个"低于基线"样本**成簇于暗区 ⇒ 非哨兵、无掩膜**；主 agent 的 106 NaN 与 \|x\|>1e30 = **float 误读副产物**（L25.3） |
| 读回 `frame_interval_s=0.15873 ⇒ 6.30 Hz`（与 ZAPBench 1.09 Hz 差 5.8×） | ✅ `m11_reference.csv` **R-01 `asserted`**（判定器 `P2.12a` 强制）；**dwell/τ 带仍 `unverified`**（R-21，`P2.12c` 强制） |

### L28.4 ★ 独立节点（M11-B3b）**独立**得到同一解码（交叉一致，非本节点自证）
复核期间观察到并行节点 M11-B3b 的独立复核工具 `tools/verify_m11_microns_series.py`（**非本节点创建**；
本节点只读其文件头以作归因，未读实现、未修改）在其独立性声明中写明，其**自行写就并自行验证**的解码路径为：
`20 B 大端头 >5I = (ndim=4, 200, 200, 1, 10) → gzip.decompress(raw[20:]) → >i2 → reshape (200,200,1,10)`。
⇒ **与本节点的结论逐位一致**（大端 int16、20 B 头、x 最快），且**两条实现互不参照** ⇒
这是"两个独立修正实现交叉一致"型的证据（与 M9 §11.1b 的 1.7% 交叉一致同型），
**强于本节点单方论证**；亦说明 RF-M11-A 的 `format_unverified` 登记已被其自身节点的实测取代。

**分工登记（避免重复/冲突产物）**：
- **本节点（B1）**：数据门（格式/完整性门/解析门/参考锚/可达性/文献/结构/Allen/回退源）——
  即 `m11_reference.csv` 的 R-01..R-26 与三道门；
- **B3b**：由同源体素序列**构建 dwell/τ 目标带**（复用 M10 冻结估计器，箱宽为其唯一自由度）。
⇒ 本节点 `m11_reference.csv` **R-21 仍标 `unverified`**（**不得**由本节点替代 B3b 定稿）；
B3b 落盘后应**新增行**（R-27+）而非修改 R-21（清单 §0.3 #8 只增不改纪律）。

---

## L29 — ★ 空间口径 10× 差的解决（RF-M11-B §四；**实像素判定，非元数据声明**）

### L29.1 两处口径的出处（逐字可核）
| 口径 | 出处（文件/字段） | 数值 |
|---|---|---|
| **4 px/µm** | `functional_data_technical_documentation_v3.pdf` 正文（已抽文本 `data/m11_raw/microns/pdf_text/…v3.pdf.txt`） | "**4 px/µm** xy resolution (440×248 px/field) to tile a **1100×1200 µm²** FOV at four depths (two planes per depth, 40 µm overlap between coplanar fields)" |
| **2.5 µm/px** | `functional_series/session4_scan7_field4/attributes.json` → `resolution=[2.5,2.5,10]` + `units=[um,um,um,s]` | 2.5 µm/px（x,y）／10 µm（z） |
| 1 µm/px | `field_summaries/session_4_scan_7_field_4/attributes.json` → `resolution=[1,1,10]`、`dimensions=[620,1100,1]` | 620 × 1100 µm |

### L29.2 实像素对齐（`tools/verify_m11_grid_alignment.py`，产物 `m11_grid_alignment.json` + `m11_grid_alignment.png`）
用两个**独立对象**的真实像素（`field_summaries` 1 µm/px 的 200×200 µm 区块 vs
`functional_series` 200 片/2000 帧的时间平均图）做尺度扫描与对齐搜索：

| 检验 | 结果 |
|---|---|
| **尺度扫描** r(f)（f = 降采样因子） | **峰值唯一出现在 f = 2.50（r = +0.126）**；f=1/1.25/1.5/2/3/4/5/8/10/12.5 全为 **−0.07…−0.01**，**f=25 ⇒ −0.031** |
| 朝向搜索 | identity **0.126** > flip_x 0.034 > transpose 0.002 > flip_y −0.042 ⇒ **无轴翻转** |
| 平移搜索（±8 px） | 最佳 **位移 (0,0)**、r=0.126 ⇒ **两对象在原点处精确同格** |
| **负对照**（series 随机平移后与对齐块相关） | **|r| = 0.014**（地板）⇒ 对齐值 0.126 = **地板的 9.0×** |
| 峰值 r 偏低的原因 | 对比机制不同（summary 是结构投影/归一化 0–1；series 是 GCaMP 时间均值），不是错位 |

### L29.3 物理解释（两个独立读回的 FOV 交叉核对）
- series：248 px × 2.5 µm = **620 µm**；440 px × 2.5 µm = **1100 µm** ⇒ FOV **620×1100 µm**；
- `field_summaries`（**另一个对象**）：dims **620×1100** px @ **1 µm/px** ⇒ FOV **620×1100 µm**；
- 文档的 4 px/µm（=0.25 µm/px）给出的是**单个 raw 子场** 440 px ÷ 4 = **110 µm**、248 ÷ 4 = **62 µm**，
  需 ~**10 × 20 个子场**（含所述 40 µm 重叠）才能拼出文档的 1100×1200 µm² **拼接 FOV**。

⇒ **结论（以实测为准）**：**两者不是同一层的矛盾，而是两个层级** ——
**4 px/µm = raw 采集的子场像素**；**2.5 µm/px = 处理后 functional_series / field_summaries 的体素**；
比值**恰为 10.0**（0.25 × 10 = 2.5），且在两个轴上都精确成立。**没有任何一处"错"**，
错的是"把 raw 子场的 px/µm 当成 series 的 µm/px"。

### L29.4 用错口径的后果（量化，可复用纪律）
| 用错方式 | 后果 |
|---|---|
| 把 0.25 µm/px 当 series 口径 | ① 一切**距离/尺度**物理量（感受野尺寸、空间相关长度、ROI/神经域半径、连接概率-距离关系）**系统性偏小 10×**；② FOV 被算成 **62×110 µm**（面积偏小 **100×**），与 `field_summaries` 的 620×1100 µm **直接冲突**；③ 任何"按物理距离配对/匹配"的对照（`dend_len`、突触-距离曲线）**口径不可比**；④ 空间下采样/池化窗口选错 ⇒ ROIs 归属错、ΔF/F 混入神经域 |
| 把 2.5 µm/px 当 raw 子场口径 | raw 子场被算成 1100×620 µm（偏大 100×）⇒ 拼接拓扑（子场数）算错，P4 的层流 FOV 划分错 |
| **纪律** | **凡使用空间尺度，必须写明取自哪个对象/字段，并做跨对象 FOV 交叉核对**（本轮已把该检查固化为断言 `P2.11f`）。|

---

## L30 — RF-M11-C 处置：活动通道改 Allen **2p 钙成像**（S 门目标带自建的前置）

### L30.1 探针（8 源，`--extras-append`，判定词表落盘）
| id | 源 | 实测 | 判定 |
|---|---|---|---|
| **G1** | `visual-coding-2p/manifest.json` | 200 / **1,340 B** | **REACHABLE**（自描述清单：BASEDIR + EXPERIMENTS/CONTAINERS/CELL_SPECIMENS/STIMULUS_MAPPINGS） |
| G2 | `_manifest.json` | 200 / 1,165 B | REACHABLE |
| **G3** | `stimulus_mappings.json` | 200 / **17,907 B** | **REACHABLE（刺激条件表已落盘）** |
| G4 | `ophys_experiments.json`（Range 64 KB） | 206 | REACHABLE（11.4 MB 全量；**无帧率字段**，实测） |
| **G5** | `ophys_experiment_events/593887846_events.npz`（913 KB 全量） | 200 / **913,278 B** | **REACHABLE（已落盘）** |
| G6 | `…/638871662_events.npz`（Range 1 MB） | 206 | REACHABLE（未取全：网络退化，见 J2） |
| G7 | `ophys_experiment_data/497256116.nwb`（Range 头） | 206 | REACHABLE（对象级）；**头解析受限**（见 L30.3） |
| G8 | `ophys_experiment_analysis/…_analysis.h5`（Range 头） | 206 | REACHABLE（对象级） |

### L30.2 ★ 关键实测：`events.npz` **不压缩** ⇒ 群体规模可由文件尺寸直接推出
- 最小件 913,278 B → `ev.npy` shape **(1, 114,130)**；3,642,958 B → **(4, 113,835)**；
  **元素数 × 8 B = 文件字节数**（913,278 /(8×114,130)=1.0；3,642,958/(8×113,835)=4.0）⇒ **npz=store（无 deflate）**。
- ⇒ `n_cells = bytes / (8 × n_frames)`，`n_frames ≈ 113,835`（多个实验一致）；
  例：16,358,638 B → **18 cells**；111,876,046 B → **123 cells**；331,278,574 B → **364 cells**。
- 意义：可在**不下载**的前提下选"性价比最高"的群体规模件（本节点据此把 16.4 MB/18-cell 与 111.9 MB/123-cell
  列为托管候选 J2）。

### L30.3 NWB 采样率读回：**受阻，如实登记**（RF-M11-C §二.2 的第 1 项）
- `ophys_experiment_data/497256116.nwb`（**176,019,299 B**）：Range 头 6 MB 请求**取到 5,100,116 B 后超时**（3.3 KB/s）；
  稀疏法 h5py 打开成功：根组 11 键可读（`acquisition`/`stimulus`/`processing`/`general`…），但
  **`/general`、`/processing` 落在前缀之外的洞**（`bad symbol table node signature`）⇒ **`imaging_rate` 不可读**。
- **替代推断（明确标注 `NEEDS_VERIFY`）**：帧数 ÷ 30.9 Hz = **61.4 min**（合理 session 长度）vs ÷ 11 Hz = **2.87 h**（不合理）
  ⇒ 自建带取 **30.9 Hz** 为主口径、**11 Hz 为敏感性并列**；`ophys_experiments.json` 实测**无帧率字段**。
- ⇒ 已登记托管请求 **J3**（取 6 MB 前缀或全量）以免把"推断"当"实测"。

### L30.4 自建 dwell/τ 目标带（**B4 冻结管线只读复用**）
- 工具 `tools/build_m11_dwell_band.py`：**直接 import** `tools/validate_p10_state_structure.py`（**不改冻结件**）调用
  `analyse(pop, dt_ms, …)` ⇒ **0.914 s 箱 → 5 帧滑动平均 → z → AR(1) φ/τ**，并**并列** H 法（2 状态高斯 HMM）/
  Q 法（分位）/ **未平滑 φ** / 样本量门 / 内部一致性 / dwell 分辨率地板。
- 产物：`data/m11_dwell_reference.{csv,json}` + `reports/neuro/m11_dwell_reference.png`；
  日志 `data/m11_raw/allen/vc2p/dwell_band.log`；**登记为托管作业 J1**（HMM 为 Python 逐时间步循环 ⇒ >10 min）。
- ⚠️ **本轮最重要的口径警示**：`φ_smoothed ≈ 0.880` vs **`φ_unsmoothed ≈ 0.179`** ⇒ 5 帧滑动平均把 φ 抬高 **≈4.9×**
  （τ 从 7.1 s 变到 ~1.0 s 量级）⇒ **绝对 τ 只能作粗粒度指标**，比值口径必须写明用哪一支；
  这与 B4 自己登记的 `honest_limit`（Ljung-Box 三口径全拒白噪）同源。
- ⚠️ **样本量门实测触发**：1-cell 件的 low 态**完整段仅 1–4 段** ⇒ **KS vs 指数不可算** ⇒ 形状判据
  标 `unavailable`（**不是 fail**）；群体规模薄（1–4 cells）已写入 caveats。
- **禁止**：沿用 ZAPBench 的 26.5 s / 37.18 s（跨物种+跨帧率+跨口径）；本带与文献档（0.3–3 s, PMID 11017176）
  并列但**不混用**。

### L30.5 吞吐波动与托管作业（纪律）
- 本轮实测：`bossdb-open-data` 194–349 → 16 KB/s；`api.brain-map.org` 1.6 KB/s；DANDI 7.2 KB/s；
  **`allen-brain-observatory` 1.6–32 KB/s**（同一小时内 913 KB 件 28 KB/s、16.4 MB 件掉到 1.6 KB/s）
  ⇒ 一切"≤300 MB 可整取"的判断都必须**当场测速**（教训 L22 #9 复现）。
- 已登记 `data/m11_b1_managed_jobs.json`：**J1**（本节点后台跑，含日志与产物）/ **J2**（更大群体 npz，REQUESTED）/
  **J3**（NWB 采样率读回，REQUESTED）；均含命令行、产物路径、完成标记。

---

## L31 — R8 与 DQ-4：属性级同源 + 逐细胞/逐对表解析就绪

### L31.1 R8（同源三级表述，取代"功能侧不可用"）
- **① 结构同源 ✅**（EM/seg，`measured`）；**② 属性级同源 ✅**：`functional_connectomics/node_and_edge_properties/v1`
  （Ding 2025, DOI 10.1038/s41586-025-08840-3）——node 表 **26 列**（`nucleus_id/animal_id/scan_session/scan_idx/unit_id/`
  `nucleus_x,y,z`/`hva`/`brain_area`/`layer`/**`osi_iv`/`gosi_iv`/`pref_ori`/`pref_dir`/`in_vivo_mean_resp`/`mei`**/…）、
  edge 表 **19 列**（`pre/post_nucleus_id`/**`population`**/`n_synapses`/`synapse_size`/`dend_len`/`pre,post_hva`/
  `pre,post_layer`/`proj_hva(_layer)`/**`in_vivo_sig_corr_iv`**/`in_silico_sig_corr_cvt`/…）；
  **`population ∈ {Connected, ADP(距离匹配), Same region}` = 清单 §0.7 #5 要求的匹配负对照**（本轮最有价值的发现）。
- **③ 时间序列级非同源 ❌**：该表无时间序列 ⇒ **S 门与 dwell/τ 目标带仍走 Allen 2p（R6 不变）**；
  逐细胞 ΔF/F 仍 `unavailable`（v8 DB，DQ-1 裁定不下载）。
- ⇒ 表述纪律：**不得**写"功能侧完全不可用"，**不得**写"全同源"；正确写法 =
  **"结构同源 + 属性级同源 + 时间序列级跨源代理（Allen 2p）"**。

### L31.2 DQ-4（托管下载）**实测状态：停滞** + 解析器/等待器就位
- `node_data_v1.pkl` 本地 **2,322,432 / 294,635,032 B = 3.417%**（`parts/` 8 段 10,067,968 B；末次写入 20:47；**进程已不在**）；
  `edge_data_v1.pkl` **0 / 201,511,086 B**；期望尺寸由 Range `Content-Range: bytes 0-0/294635032` **实测**确认。
- **按主 agent 指示不另起下载** ⇒ 交付**解析器**（`tools/parse_m11_functional_connectomics.py`，**只解析不下载**，
  完整性门不过即 `INCOMPLETE` + 非零退出）+ **等待器**（`/tmp/m11_fc_waiter.sh`，每 5 min 检查、达尺寸即自动解析、日志含 `ALL_DONE`）。
- `readme_v1.md` 已置于 `functional_connectomics/`（5,594 B）⇒ **schema 核对自动化**（node 26 / edge 19 列已解析出）。

### L31.3 教训（本轮新增）
- **`&` 与 `cd` 的作用域陷阱**：`cd X && cmd &` 会把 `cd` 放进子 shell ⇒ 后续 `curl -o` 落在**旧工作目录**
  （本轮 `577885923_events.npz` 曾落到仓库根；已归位）。**写文件后必须回读路径**。
- **"托管作业在跑"必须实测确认**：凭日志首行/用户告知不足；本轮以 **parts 字节数 + 时间戳 + 进程表**三重确认对方作业已停滞。

---

## L32 — DQ-4 判 **`unavailable（吞吐限制）`** 的处置与"实测吞吐"口径

- **主 agent 实测**：S3 单连接 ≈17 KB/s；8 路并发 Range（17 min）聚合 **10–16 KB/s** ⇒ 294 MB 需 **5–8 h**；
  **GCS 无该功能数据镜像（404）** ⇒ **`reachable but infeasible`**，与 R1（v8 DB）、M10（ZENODO）同类。
- **本节点处置（三项）**：① **不阻塞**（DQ-4 状态写实入 `m11_functional_connectomics_counts.json` / R-31 / M12 **G-7**）；
  ② **不轮询**（**已主动停掉等待器**，日志记 `stopped by M11-B1 per DQ-4 ruling`；此后不对该对象发任何请求）；
  ③ **readme 足以规划**（schema 已机器化解析：node 26 / edge 19 列 ⇒ 写入 M12"应做而未做"与未来取数计划）。
- **R8 表述定稿（主 agent 口径）**：属性级同源 = **通道已确认可达**（HEAD 200 + readme 实证 + DOI），
  **本轮因吞吐 `unavailable`** ⇒ **P7/P8/P14 的同源属性对照与其匹配负对照材料本轮均未得**，
  **不得**写"已具备"、**不得**写"已排除"；已写入报告 §6e（限制正文）与 M12 G-7。
- **四类参考量四态**（`data/m11_activity_reference.csv`）：**仅 C3 dwell 达 `measured`**（4,070 bins；dwell 2.74 s、
  p90/中位 3.23、转换 0.377 Hz、τ 6.46 s）；C2 率分布/成对相关为 **`thin`**（n=5 细胞 / 6 对）；
  C1 调谐与 C4 行为 **`unavailable`**（属性表未取 + 2p 无刺激标签；IBL 需 token）。**thin 不得记 pass**。
- **新口径（本轮立）**：**取数决策必须有实测吞吐支撑**（`data/m11_throughput_measured.csv`，11 行）；
  同小时吞吐波动可达 **20×**（1.6 KB/s ↔ 32 KB/s）⇒ 一切"应该很快"的推断都作废。
- **自查出的第二例实现缺陷**：`np.triu_indices` 返回 **(row,col) 元组** ⇒ `len(iu)` = 2 而非对数；
  已修为 `len(iu[0])`（对数 6）。**教训**：计数类字段必须与"值的个数"交叉核对（`n` 与 `len(values)` 一致性断言）。

---

## L33 — 🔴 RF-M11-D：主 agent 反转 R2/R6；**同源 dwell/τ 主带已落地**

### L33.1 事件链（如实入档，含主 agent 的自我纠错）
- 主 agent 独立复算确认本节点结论：`>i2` 直读 ⇒ 横向相关 **0.7812 / 0.7985 / 0.6513**、帧间 **0.7984 / 0.7772 / 0.6883**、
  p50 **1953 / 1472 / 1368**（三片抽样）；本节点 18 假设 × 6 判据的结构门 ⇒ 胜出 **① 直读 int16-BE**（全量 200/200）。
- **主 agent 的错误（其自述，已如实入档）**：此前用 `<i2`（小端）直读，**测试集漏测 `>i2`** ⇒ 得出"高熵/非图像/不可解码"的
  假象并据此发出降级裁定；其报告的 **106 个 NaN 与 |x|>1e30** 是 **float 误解读副产物**。
- **→ RF-M11-D：R2 反转为 `measured`（允许统计，须掩膜哨兵）；R6 改为"同源优先、Allen 2p 降为跨制备参照"**；
  R1（逐细胞轨迹仅在 DB）与 DQ-4（吞吐）不变。

### L33.2 同源 dwell/τ 主带（`tools/build_m11_dwell_band_microns.py`；无需新下载）
- 数据：**200 片 × 10 帧 = 2,000 帧**（`session4_scan7_field4` 的 **x∈[0,200)px × y∈[0,200)px** 子块 = 500×500 µm，
  t∈[0,2000) ⇒ **317.46 s**；该 field 全长 40,000 帧/105.8 min，本轮只用已落盘部分）。
- 管线：`>i2` → (10,200,200) → **哨兵掩膜** → 逐帧有效像素均值 → **B4 冻结 `analyse()`**（0.914 s 箱 → 5 帧平滑 → z → AR(1)；H+Q 并列）。
- **主带（333 箱）**：占用 **0.631/0.369**、转换 **0.046 Hz**、**dwell 中位 12.86 s（物理）**、p90/中位 **5.78**、**τ 32.5 s**。
- **S 门体检（如实四态）**：S-1 ✅ pass；**S-2 `unavailable`（低态完整段 6 < 10）**；
  **S-3 `fail/thin`（重尾✅ 但 KS p=0.558 未拒指数，且完整段 6<30）**；**BIC `unavailable`（333<500）**。
- **容差（自测，随带登记）**：H/Q **1.69×**、掩膜开/关 **1.00×**、时间半劈 **3.23×**、空间半劈 **1.29×**、
  φ 平滑/未平滑差 **0.084**；**箱宽口径差 4.2%**（模块 0.9138 s vs 实际 0.9524 s）⇒ 并列两种时间基。
- **跨制备对照（Allen 2p，已降为参照）**：dwell 2.74 s / τ 6.46 s ⇒ 与同源带差 **dwell 4.69× / τ 5.04×**
  （不同制备 + 不同箱基 + 不同群体规模；**不得**混用为同一带）。
- **迁移留痕**：原 Allen 带已保存为 `m11_dwell_reference_allen2p.{csv,json}`（**证据不丢**），
  `m11_dwell_reference.{csv,json}` 现为**同源主带**（RF-M11-D 要求的主/参照之分）。

### L33.3 新增纪律（RF-M11-D §四，写入本项目惯例）
> **"不可解码/不可用"结论的举证标准**：判"不可解"**前**必须证明测试集覆盖
> **dtype × 字节序(LE/BE) × 排列(行主/列主/Morton/分块) × 是否差分编码**；
> **任一维未覆盖 ⇒ 只能写 `unverified`，不得写 `unavailable/不可解码`**。
（本条由主 agent 自身错误得来；本节点 L25 的"18 假设 × 6 判据"可作为其模板实现。）

### L33.4 本轮新增自查缺陷（第二/三例）
- `np.triu_indices` 返回**元组** ⇒ 计数应取 `len(iu[0])`（曾把 6 对报成 2）——**计数必须与值个数交叉核对**。
- `&` 与 `cd` 的作用域：`cd X && cmd &` 中 `cd` 只在子 shell 生效 ⇒ 输出落在旧 cwd（曾把 npz 落到仓库根）。

---

## L34 — RF-M11-E：交叉核对完成 + 掩膜定稿更正 + DQ-5 落实 + 新增"退化结构"纪律

### L34.1 与主 agent 的**同源带交叉核对**（`data/m11_same_source_crosscheck.{csv,json}`）
- **完全一致的格**：**原生/未平滑/帧均值**（φ 0.9825 vs 0.9825 = **0.005%**；**τ 9.015 vs 9.015 s = 0.003%**）、
  **0.9138 s 箱/未平滑/帧均值**（φ 0.8875 vs 0.8875 = **0.001%**；τ 7.658 vs 7.657 = 0.007%）；
- **差异 >5% 的三格，全部定位到具体口径维度**：
  ① `native/none/frac` τ 10.9%（φ 仅差 0.25%）⇒ **"φ→τ 对数放大"**；
  ② `native/smooth5/*` τ 3.2–12.1% ⇒ **平滑边界约定**（本节点 `mode='same'` n=2000/333；主 agent `'valid'` n=1996/329）+ 对数放大；
  ③ `bin/smooth5/mean` τ 22.7%（φ 差 0.86%）⇒ 同上 + **箱宽 4.2%**；
- **维度效应（实测）**：箱宽 **1.18×**、平滑 **2.9×**、代理 **1.30×**、**状态估计分辨率 2.16×**、模块/物理箱宽 **1.0422**；
- **⇒ 主 agent 的"同源目标 ≈ τ 8–9 s（未平滑、原生帧率）"被本节点独立复现**（差 0.003%）；
  **S 门目标取同源 8–9 s 量级**，**不使用 ZAPBench 37.18 s**；
- **可复用教训**：**τ 必须与 φ、时间基、平滑约定同报**——φ 差 1%（≈0.97 处）即可放大为 ~19% 的 τ 差
  （`dτ/τ = dφ/(φ·lnφ)`）；**只报 τ 的交叉核对没有意义**。

### L34.2 掩膜**定稿更正**（主 agent 接受本节点证据）
- **定稿**：`int16-BE` 直读 + **低于基线样本保留并登记（不掩膜）**；**"哨兵"仅指 `32767` 饱和（实测计数 0）**。
- 已改：`tools/build_m11_dwell_band_microns.py` **主口径改为不掩膜**；掩膜版降为**稳健性对照**（比值 ≈**1.00×**）；
  `sentinel` 字段改写为 `n_below_baseline` + `n_saturated_32767=0` + 成簇/暗区证据。

### L34.3 DQ-5 落实（清单按"新增行 + 保留原行"）
- 已把 **读取约定**（20 B 大端头 + **大端 int16** + **无掩膜** + 完整性门）与 **6.30 Hz 读回断言**写入清单 §1.1.2/§1.5；
- 并把 **`tools/validate_m11_microns_parse.py` 的 18 假设 × 6 判据**登记为该数据集的**解析门标准动作**。

### L34.4 新增纪律（主 agent §四.2 采纳）
> **判据必须显式排除"退化结构"**：累计/单调/常量等只需极少参数即可拟合的结构，会让"弱判据"天然通过；
> **"能通过" ≠ "排他"**——必须证明**竞争假设被判据否掉**，而不是"首选假设通过"。
> （本轮实证：原 4 条判据下通过者 **9 个**；补 **C5 声明域约束** + **C6 非退化**后压到 **2 个**，再按相干性+简洁性唯一选出 BE 直读。）

### L34.5 两个 field（RF-M11-E §一.①）现状与廉价替代
- 本节点**只有 field 4** 的像素在盘（另 4 个 (x,y) 块与第二 field 均未下载）；**第二 field 建带需新下载**
  ⇒ 已登记 **J5**（约 **33.4 MB** = 200 片 × 167 KB，第二 field `session5_scan3_field1` 的 (0,0,0) 块；
  实测 20–30 KB/s ⇒ 19–28 min，**按 >10 min 纪律交主 agent 托管**），**未自启**；
- **已有的跨窗口离散度替代**：**时间半劈 3.23×**、**空间半劈（x 左右）1.29×**、代理 **1.30×**、状态分辨率 **2.16×**、
  箱宽 **1.18×**、平滑 **2.9×** ⇒ 作为"同阶窗"的**实测来源**（**跨 field 离散度仍缺**，明确登记）。

---

## L35 — J2（钙域静默口径）+ R-32/R-33 新行 + sha 基准约定 + 第二 field 吞吐实测

### L35.1 J2 定稿（`tools/build_m11_calcium_silence.py`）
- **三口径 × 两单元口径**（定义见报告 §6j）。field4（2,000 帧 / 317.46 s / 100 个 20×20 px 单元代理）：
  **zero_activity = 0.4900**、below_baseline = 0.0000、low_activity = 0.0000、
  **full-FOV 帧级无事件 = 0.8915**、弱帧 = 0.0465、像素级零事件 = 0.0052；
  单元事件数中位 **1** / 均值 **10.3**（重尾）；事件阈值 `ΔF/F 中位 0.423 + 3×MAD 0.385 = 1.578`。
- **必附限制**：**钙事件口径 ≠ spike rate**（in vitro 17.4/17.6 Hz 当 (a) 带会高 **8.7–34.8×**，B3 量化）；
  单元为**空间分箱代理**（真实 ROI 掩膜未解码）；与**电生理静默口径不同**（不得混用）。
- **自查出两处"构造性退化口径"并当场修正（留档）**：① `low_percentile` 用"低于自身分布的 p20"⇒ **恒 20%**；
  ② full-FOV 用"FOV 均值 vs 全局阈值"⇒ **恒 1.0**。⇒ 已改：① 固定阈值（0.2×全单元 p95 中位）；② 帧内"无任何像素越阈"。
  **这正是 RF-M11-E §四.2 新纪律（判据必须排除退化结构）的第一次当场应用。**

### L35.2 R-32（同源带基线）/ R-33（J2）——**只新增行**
- `m11_reference.csv` 现 **33 行**（30 asserted / 1 NEEDS_VERIFY / **2 unverified**）：**R-21 原行保留不动**，
  新增 **R-32**（同源带基线：未平滑原生 τ 8–9 s、箱基 32.5 s、容差来源八项）与 **R-33**（J2 三口径 + 两单元口径 + 四条限制）。

### L35.3 sha 基准约定（RF-M11-F §五；复用 M10 `tools/hash_report.py`）
- **三基准**：`file_bytes`（文件原始字节）/ `payload`（数组载荷，不含 header）/ `semantic`（语义等价判据）。
- **本节点全部 sha256 采用 `file_bytes`** ⇒ `m11_data_inventory.csv` 新增 **`sha_basis` 列**（逐行标 `file_bytes`）；
  跨节点核对时用 `tools/hash_report.py`（**非本节点文件**）一次打印三基准，避免"同值不同哈希"误判。

### L35.4 第二 field：**先测吞吐再决定**（实测结果与裁决）
- 实测（`session5_scan3_field1` 块 (0,0,0)）：**前 3 片 96–121 KB/s → 持续 ~8 KB/s（节流）**；单片 **~160 KB**；
  200 片 ≈ **32 MB**，**ETA ≈50–65 min < 90 min 阈值** ⇒ **自启下载（J5，未按 unavailable 处理）**，
  逐片写 manifest（`t/url/http/bytes/sha256/content_type`）+ `download.log`（含 `ALL_DONE`）。
  **完成后**：重跑 `build_m11_dwell_band_microns.py` 与 `build_m11_calcium_silence.py` ⇒ 得**跨 field** 离散度
  （现仅有"同 field 内口径离散"，已明确标注区别）。

### L35.5 关联 R4 状态（更正主 agent 的假设）
- 主 agent §四.1 仍把 **R4（10× 口径差）**列为我的待办；**该口径差已在上一轮解决**：
  实像素同格性（尺度扫描峰值唯一 **f=2.50**、位移 **(0,0)**、对齐 r = 地板 **9.0×**）、
  层级解释（**4 px/µm = raw 子场；2.5 µm/px = 处理后**，比值恰 10.0）、掩膜格层由 **8 chunk 键覆盖实测**确认；
  落盘 **R-27/R-28** + 断言 `P2.11f/g/h`。⇒ **(b) 的钙域静默口径与四类参考量不再被 R4 阻塞**。

---

## L36 — J5 完成（两 field）+ J2 两 field 结果 + 跨 field 离散度

### L36.1 第二 field 下载与完整性门
- 吞吐实测：**前 3 片 96–121 KB/s → 持续 ~8 KB/s（节流）**；200 片 = **31,472,642 B（31.5 MB）**，≈50 min。
- **完整性门：首轮 197/200**（`t=75`/`t=104` 缺失、`t=128` 截断 **69,158 B**）⇒ **逐片重取后 200/200 COMPLETE** ✓
  （重取即用即验：`gzip` 成功 + 解压字节 == `Πheader_dims × 2`）。**再次印证"完整性门必须在统计前跑"**。

### L36.2 J2（钙域静默）两 field 结果
| 量 | field4（200×200 px，100 单元） | field2（100×100 px，25 单元） | 跨 field |
|---|---|---|---|
| **zero_activity 静默比例** | **0.4900** | **0.4800** | **±2%（极稳）** |
| below_baseline / low_activity | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 一致 |
| full-FOV 帧级无事件 | 0.8915 | 0.1461 | **依赖 FOV 大小 ⇒ 口径警告** |
| 事件阈值（ΔF/F 中位+3·MAD） | 1.578 | 0.807 | — |

⇒ **P5 (b) 钙域口径可取 ≈0.48–0.49（单元级）**；**full-FOV 口径不得跨 FOV 比较**（0.89 vs 0.15）。

### L36.3 同源 dwell/τ 主带：**两 field**（`data/m11_dwell_reference.json::band.cross_field`）
- field4：dwell 中位 **12.34 s**（模块基）/ **12.86 s**（物理）、τ **31.23 s**、φ_sm 0.9712、333 箱
- field2：dwell 中位 **10.97 s** / τ **31.56 s** / φ_sm 0.9715 / 333 箱
- **跨 field 离散：dwell 1.125× / τ 1.01×** ⇒ **比"同 field 内口径离散"（时间半劈 3.23×）还小** ⇒
  **同源慢结构跨 field 可复现**（RF-M11-F §二 的"可选交叉核对"已由 J5 完成）。
- **同阶窗（分两维，须分列）**：① **同 field 内口径离散**：代理 ±6%、Q vs H 1.69×、时间半劈 3.23×、
  空间半劈 1.29×、箱宽 1.18×、平滑 2.9×、状态分辨率 2.16×、模块/物理箱宽 1.0422；
  ② **跨 field 离散**：dwell 1.125×、τ 1.01×。⇒ **建议"同阶窗"下限 ≥3×**（由口径维主导），**不得**窄于目标自身离散度。

### L36.4 新增教训
- **"& 与长命令链同用"**：把 `A && B && nohup C > log &` 整条后台化，会让**日志路径/工作目录语义漂移**
  （本轮一次 `final_validate6.log` 未生成）；⇒ **后台作业只用最简单的单命令形式**，且**每次只做一件事**。
- **断言字符串与产物措辞必须同步**（`proxy`/`代理` 不一致导致一次误报）⇒ 断言**优先匹配语义关键词**，避免措辞耦合。

---

## L37 【DQ-6 完成】Allen Neuropixels `units.csv` 全量解析 + **S3 多段 ETag 完整性判据**（2026-10-12 07:03–07:05）

### L37.1 取数与完整性（**判据强于 sha256 的一步**）
- **主 agent 的 DQ-6 作业已死于昨夜 22:33**（实测：无 `curl` 进程、`/tmp/ma_units_dl.log` **0 字节**、
  产物停在 **1,720,320 B / 51,439,189 B = 3.35%**，至 07:01 **8.5 h 无增长**）⇒ 本节点**当场复测吞吐**后自行续传。
- **吞吐是"当场量"而非"可外推量"**（本轮最强的一条教训）：昨夜 ~1.6–16 KB/s，**今晨 5.9 MB/s（差 ~3,600×）**；
  `curl -C -`（rc=206）49,718,869 B / **8.42 s** 完成。⇒ **任何 >50 MB 取数决策前必须当场复测**，不得引用历史档。
- **★ 新增完整性判据（建议纳入 §1.1.3 完整性门链）**：S3 多段对象的 ETag **可复算**——
  `ETag = md5(concat(md5(part_i))) + "-N"`。本对象 `ETag="5555828a…-7"`，用 **part=8 MiB × 7 段**复算**逐位命中**
  ⇒ **本地文件逐字节等同服务器对象**。意义：**锚定在服务器侧**，不依赖任何外部发布的哈希
  （对比：sha256 只能证明"我算了几次都一样"，ETag 能证明"与源一致"）。
  **门槛**：须先探到 `ETag` 的 `-N`（多段）与 `Content-Length`，并**试出 part_size**（本轮试 5 候选，唯一命中 8 MiB）。
- 复核：`size==Content-Length=51,439,189` ∧ ETag 复算通过 ∧ 末字节为 `\n` ∧ 非 27 字段数据行 **0**
  ⇒ **COMPLETE**（sha256 `1dffbe8dac7f222e…`，基准 `file_bytes`）。
- 另取 `channels.csv`（6,613,329 B，**单件 <50 MB**，2.7 MB/s）：它是 **units→session 连接链的必需键**
  （`ecephys_probe_id` → `probes.ecephys_session_id` → `sessions.session_type`）。
  ★ 纠正一条**被实测推翻的假设**：此前判"1 MB 前缀够用"——实测该前缀覆盖 channel id 849,705,558–849,800,796，
  而 units 引用的 channel id 区间为 **849,990,690–850,264,442** ⇒ **前缀覆盖率 0%**。**"前缀够用"必须实测，不得推断。**

### L37.2 ★★ 口径：`complete_session` **不是自发率**（**本轮最易犯错处**）
- `units.csv::firing_rate` 的 `epoch_name_quality_metrics` 在 **全量 135,275 行一律为 `complete_session`**
  ⇒ 该率**含视觉刺激呈现期间**，**不是自发率**。
- ⇒ **只能作 (a) 自发率带的"上界候选"**；**(a) 自发率带仍 `unavailable（口径不含）`**
  （需 NWB 的 `spontaneous` epoch；单件 >2 GB 或落在稀疏洞内，本轮按预算与阈值不取）。
  **不得**因"拿到了全量逐单元率表"就把 (a) 标成 `measured` —— 这是**口径错**，不是数据错（同族：见 L25 钙 vs spike）。
- **皮层限定（M11 = 皮层微环路，MICrONS minnie65 = VISp）**：
  - 全单元聚合 good 中位 **3.175 Hz** 被**非皮层结构**抬高（LGd 6.86 / VPM 6.64 / LP 4.95 Hz）；
  - **VISp**（与 minnie65 同名区）：good 中位 **2.485 Hz**、p90 10.30 Hz；
    **VIS\***：good 中位 **2.657 Hz**、p90 10.35 Hz；
  - ⇒ **带必须区级限定**，聚合值**不得**当皮层先验。（结构构成：VIS\* 37.0%、海马 33.0%、grey 10.3%。）
- **跨口径量化（实测量替换估计值）**：VISp good 中位 = **0.794 × 钙事件率 Nyquist（6.300006/2 = 3.150003 Hz）**，
  且 **41.29% 的 good VISp 单元率 > Nyquist** ⇒ 钙成像在事件率口径下**结构上无法表征约四成**兴奋性群体率。
  这**替换**了 B3 的估计值 63.5%（2.0/3.15），方向一致但更强。

### L37.3 退化指标登记（**"能通过 ≠ 排他"的同族纪律**）
- **7,894 / 135,275 = 5.84%** 的单元命中退化：`isolation_distance ≥1e4`（1,536；**最大 2.3e15**）
  或 `isi_violations >20`（6,436；**最大 4,061.65**）。
- 归因（**不是奇异值，是指标退化**）：这些行的 `firing_rate < 0.01 Hz`（近静默 n=**4,343**）
  ⇒ 度量在**样本量不足**时发散（无 spike ⇒ 距离/违规率爆炸）。
- ⇒ **单独登记并从所有带中剔除**（`quality_good_excl_degenerate` 与 `quality_good` 分列）
  —— 否则"带"会被 34 个 1e12 量级的假值污染。**同族自查**：小端误读也曾给出"合格"外观（L25）。

### L37.4 层/区登记（主 agent §四②）
- `channels.csv` 提供 `ecephys_structure_acronym`（**区级**）+ CCF 前/背腹/左右坐标。
- **58 个结构、58 session 全覆盖**；CCF 背腹深度**全量可用**（n=135,275，**-1250 → 4890 µm**）。
- ★ **acronym 不含层号**（全 58 个中带数字者仅 CA1/CA3/CA2，系**海马亚区**非皮层分层）
  ⇒ **区级 = `measured`；层（L1–L6）= `derived`**（须 CCF 深度 + 皮层表面/厚度模型，本轮未做）。
  **不得**把区级信息表述为"层已登记"。

### L37.5 session_type 隔离材料（P14，主 agent §四③）
- `brain_observatory_1.1`（32 session / 74,072 单元）good 中位 **3.390 Hz** vs
  `functional_connectivity`（26 session / 61,203 单元）**2.941 Hz** ⇒ 差 **1.15×** ⇒ **口径可比**。
- **独立性单位应为 session**（同 session 内单元共享探针/动物）：逐 session 中位带 = **2.798 Hz**
  （p10 2.332 / p90 3.379，跨 session 极差 **2.90×**）；VISp 逐 session 极差 **3.29×**。
- 质量构成：`good` **99,180** / `noise` **36,095**；空字段最多的三列 = `silhouette_score` 19,524 /
  `nn_hit_rate` 10,875 / `recovery_slope` 7,920 ⇒ **"空 ≠ 0"**（解析时按缺失处理，不得填 0）。

### L37.6 术语与裁定的**收口确认**（本节点无异议，仅登记）
- **"哨兵"一词自本轮起仅指 `int16` = `32767` 饱和**（实测计数 **0**）；**低于基线样本 ≠ 哨兵**，
  按**无掩膜 + 保留并登记**（10,383 / 80,000,000 = 0.01298%）。
  ⇒ 本节点**历史段落（L25/L33）中把"低于基线"写作"哨兵掩膜"的措辞按只增不改保留原文**，
  但**一切新产出**按上述定稿术语；`data/m11_data_gate_report.md` §6e 的旧引文同理（见该处新增更正行）。
- **R2 = `measured`（大端 int16）** 已由主 agent **早在 R9 反转**，本轮**无需再改裁定**；
  本节点 `R-26` + `data/m11_decode_proof.json` + PNG 作为留痕保留。
- **Morton 补证**（256×256 取有效位 = 恰好 40,000、无越界）已被主 agent 接受。

---

## L38 【J2 收口】钙域静默口径 5/5 验收通过 ⇒ **G1 的唯一阻塞已解除**（2026-10-12 07:09，**无新下载**）

主 agent §三 指出：**J2 是本里程碑唯一能解锁 G1 的动作**。B3 为 J2 预留 **5 条验收条件**，本节点逐条收口
（工具 `tools/complete_m11_j2_acceptance.py`，**只增不改**：仅向既有 JSON 追加 `acceptance_completion` 键，
既有测量键（2 行 × 4 band_rows）**未改动**，审计链不丢）。

| B3 验收条件 | 状态 | 证据（全部基于**已在盘**产物） |
|---|---|---|
| ① 口径逐项写明 | ✅ | **3 种静默定义**（零事件 / 低于基线 / 弱活动固定阈值）× **2 种单元口径**（per-ROI **代理** / 全视野逐帧）+ 逐像素口径 |
| ② 跨 field 离散度 | ✅ | 新增计算：**可复现（≤1.2×）** = 零事件静默、低于基线、弱活动、事件中位数；**不可复现（≥3×）** = full-FOV 无事件 **6.0853×**、弱活动、**事件密度 9.58×** |
| ③ 与电生理域**并列** | ✅ | 用电生理**实测量**（非估计）：VISp good 中位 **2.485 Hz** = **0.794×钙事件率 Nyquist**；**41.29%** 超 Nyquist |
| ④ 哨兵处置**先声明** | ✅ | 2/2 行带 `sentinel_rule`（**无掩膜**；「哨兵」仅指 32767 饱和，计数 0） |
| ⑤ 已覆盖维度 | ✅ | **已覆盖 7 项 / 未覆盖 5 项**分列；按 R9 纪律**未覆盖 ≠ 不可用** ⇒ 判 `partial`（measured），**不判 unavailable** |

### L38.1 ★ 口径选择（**直接决定 G1 能否判**）
- **per-ROI 代理零事件静默比例**：field4 **0.4900** / field2 **0.4800** ⇒ 跨 field **仅 1.0208×**
  ⇒ **可作 G1 的 (b) 静默判据口径**（带 = **[0.4800, 0.4900]**，已给 `silent_frac_calcium_band_lo/hi` 可直接落表）。
- **全视野逐帧口径**：0.8915 vs 0.1465 ⇒ **6.0853×，不得作判据**——其差异由**视野尺寸/活动水平**主导
  （field2 视场仅 100×100 px = field4 的 **1/4**，事件密度高 **9.6×**）。
  ⇒ 这条与 L36.1 的"同阶窗"教训同族：**跨 field 不可复现的量不得进判据带**。
- **必须随带引用的四条限制**：① 单元是**空间分箱代理**（真实 ROI 掩膜未解码）⇒ **代理↔ROI 偏移量未测**
  ⇒ 带宽须**叠加未量化口径余量**（不得当精确带用）；② 样本量 **thin**（100 / 25 代理单元，B4 门 65/130/500）；
  ③ 电生理并列**仅作量级**（**不同量纲**，不得替换/合并）；④ **观测窗**口径（前 2,000 帧），非全场。

### L38.2 残留（写清"已覆盖维度"而非"不可用"）
`field_seg_masks` 的 `compressed_segmentation` 本节点**未自证解码**（无官方客户端、PyPI 不可达）
⇒ 真实 ROI/神经毡掩膜这一维**未覆盖**，故本轮用**代理**并逐行标注 `proxy` ——
**按 R9 纪律写"未覆盖此维"，不写"不可用"**。若后续要真实 ROI，需先独立自证解码器（另立工作项）。
