# M12 · B-5：**真实 ROI 掩膜** —— 解码、自证与 (b) 口径重算

**节点**：M12 执行节点 **B-5**（工作目录 `/Users/weidong/ai/small_world`）
**日期**：2026-10-14
**结论一句话**：**目标 A 达成** —— `field_seg_masks` 的 `compressed_segmentation` 掩膜**已解码并自证**
（双实现逐字节互证 8/8 + 10/10 缝的标签 id 一致性 + 两个判别性负对照坍缩到 0），
并已用它按 M11 的**同一口径**重算 (b)；**卡住 M11 G1 的 `coverage = covered 7 / missing 1 (real ROI mask)`
这一维的"不可得"前提被否证**。
**但**：掩膜卷与数据库 `Mask` 表（CaImAn `mask_id`）的**身份链未确认**（§7 第 1 条），
该项**必须由主 agent 裁决**是否接受本证据链来填 L115/L116。

---

## 0. 硬约束遵守声明

| 约束 | 遵守情况 |
|---|---|
| 不得运行 `--device mps` | ✅ **全程未运行**；本节点所有计算为 **CPU 单线程（Numpy）** |
| 不得 kill 任何进程 | ✅ **未 kill 任何进程**。开工前 `pgrep -f run_m11_p5_arms` = `73234 73250 73251`（L5 作业在跑），收尾时 = `73234 73250 73998`（仍在跑，未被触碰） |
| 不得起第二个 MPS 作业 | ✅ 未起任何 MPS/GPU 作业；未起任何服务端 |
| 不得改既有文件 | ✅ `git status --porcelain \| grep -E '^( M\|M \|MM\|AM)'` = **空**（无任何既有件被修改）。本节点**只新增**文件 |
| 不得编造 | ✅ 每条结论给出**行号/字段/HTTP 码/实测吞吐**；不确定者显式写"**未确认**"（§7） |
| 禁止 `2>/dev/null` | ✅ 全程未用（可检：本节点命令清单见 §10） |
| 不得降规模、不得用云 GPU | ✅ 未降规模（field4 用与 M11 同源的 2000 帧 / 200×200；**并新增**了 field2 与跨时间窗维度）；未用云 GPU |
| 取数前必须实测吞吐 | ✅ 每一件都登记 (TTFB, 传输段, 对象大小) 三元组（§3.2、§7.3） |

---

## 1. 这件事为什么重要（依据链复核，行号已逐条自查）

**B-5 的定义**（`docs/m12_list_review_master.md` **L77–L107**，R146-补登）：
`data/m11_p5_targets.csv` **L115–116**（`silent_frac_calcium_band_lo` / `_hi`）为 `value=unavailable, final=0`，
**唯一缺失维 = 真实 ROI 掩膜**。

我**自行复核**了引用的每一处（不复述他人结论）：

| # | 位置（本轮实测） | 内容 | 复核结果 |
|---|---|---|---|
| 1 | `tools/validate_p11_resting.py` **L129** | `GATE_CRITERIA = (CRIT["c"], CRIT["e"], CRIT["i2"])` | ✅ 张力 (`i2`) 是 G1 的**合取项** |
| 2 | 同上 **L1228** | `rec = {"status": "ok" if (sa_ == "ok" and sb_ == "ok") else "unavailable", ...}` | ✅ `(a)` 与 `(b)` **两条带都已定稿**才 `"ok"` |
| 3 | 同上 **L1352** | `elif all(v == "pass" for v in gate.values()):` → `g1 = "pass"` | ✅ 因 `i2` 的 `status` 非 `"ok"` ⇒ 该分支**不可达**（永假） |
| 4 | 同上 **L1357** | `g1, why = "unavailable", (f"G1 门条件中存在未测项：...")` | ✅ 实际落点为 `unavailable` |
| 5 | `data/m11_p5_targets.csv` **L115 / L116** | `silent_frac_calcium_band_lo/_hi`，`value=unavailable`，`final=0` | ✅ `note` 写 `coverage = covered 7 / missing 1(real ROI mask)` |
| 6 | `data/m11_calcium_silent.json::four_state_verdict` | `real_roi_mask_unit = "unavailable（掩膜未自证；仅此一维缺）"`；`overall_b_caliber = "partial（三口径 measured + 一维 unavailable）"` | ✅ 与 5 一致 |

**⇒ 结论（复核成立）**：只要 `(b)` 的两行不转 `final=1`，M11 的 G1 **永远不可判 `pass`**，与工作点参数无关。

### 1.1 关键：阻塞的**行级前提**是哪一条

L115/L116 的 `note` 援引的是 **J2 验收条件 e)**，其原文在 `data/m11_p5_targets.csv` **L114**：

> **e)** 若 **ROI 提取管线不可得** ⇒ 判 `unavailable` **并写明已覆盖的维度**（R9 新纪律：不得在未举证前写「不可解码/不可用」）

⇒ 该条件的**前提**是"ROI 提取管线**不可得**"。M11 的 `tools/build_m11_calcium_silence.py` **L18** 逐字登记了这一前提：

> ⚠️ **真实 ROI 掩膜**（`field_seg_masks`，`compressed_segmentation`）**本节点未能自证解码**（无官方客户端）

**⇒ 因此 B-5 的可解性归结为一件具体的、可判定的工程事实：`compressed_segmentation` 能不能被自证解码。**
本节以下即对该前提做**判决性检验**。

---

## 2. 判决：掩膜**可得且可自证**（目标 A）

### 2.1 掩膜对象与几何（逐值实读，非转述）

`data/m11_raw/microns/mask_s4s7f4_info`（M11 落盘的原样 `info`；本节点另落 `info_session_4_scan_7_field_4.json`，sha256 `bc792e26…`）：

```json
{"data_type":"uint64","num_channels":1,
 "scales":[{"chunk_sizes":[[128,128,1]],
            "compressed_segmentation_block_size":[8,8,8],
            "encoding":"compressed_segmentation",
            "key":"2500_2500_10000","resolution":[2500,2500,10000],
            "size":[248,440,1],"voxel_offset":[0,0,0]}],
 "type":"segmentation"}
```

对照钙电影 `functional_series/session_4_scan_7_field_4/attributes.json`（**本轮实读**）：
`dimensions = [248, 440, 1, 40000]`、`resolution = [2.5, 2.5, 10, 0.15873]`、`blockSize = [200,200,1,10]`。

**⇒ 两者同格**：248×440、**2.5 µm/px**。掩膜 array 的 `(i,j)` 与电影 array 的 `(x,y)` **逐一对应**，
不需要任何重采样（M11 `tools/verify_m11_grid_alignment.py` L240–L268 已由 chunk 键覆盖实测得出"格层同格"，
但把**像素层**留作 `unresolved（形式未自证）`；本节点即闭合该像素层，见 §4.4/§4.6）。

### 2.2 掩膜 chunk 的来源与完整性（三元组实测）

主 field 的 8 个 chunk **M11 已落盘**于 `data/m11_raw/microns/field_masks/*.bin`。本节点做了
**线上复取逐字节比对**（不是转述 M11 的 sha）：

| 项 | 值 |
|---|---|
| S3 键（实取） | `https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/functional_data/field_seg_masks/session_4_scan_7_field_4/2500_2500_10000/0-128_0-128_0-1` |
| HTTP 码 / 对象大小 | **200** / **4,070 B** |
| **实测三元组 (TTFB, 传输段, 对象大小)** | **(0.9855 s, 0.4286 s, 4,070 B)** ⇒ 端到端 2,878 B/s |
| sha256（线上） | `cfbafb134850af6d3c9e8f14bcad11fc5718378ca88564a55e19107eba4bec60` |
| sha256（盘上 M11 件） | `cfbafb134850af6d3c9e8f14bcad11fc5718378ca88564a55e19107eba4bec60` |
| 完整性门 | **COMPLETE**（`size_download == Content-Length == 4070`） |

⇒ **掩膜出处与 M11 盘上件逐字节一致**；本节点没有引入新的"假件"。

**第二个 field（复现对照，本节点新增取数）**：`field_seg_masks/session_5_scan_3_field_1/2500_2500_10000/<x0>-<x1>_<y0>-<y1>_0-1`，
8/8 `COMPLETE`，共 **33,299 B / 10 对象**，TTFB ∈ [0.8429, 1.8503] s，传输段 ∈ [0.001, 0.618] s
（**逐件三元组**见 `data/m12_b5_roi_mask/field2_fetch_manifest.json`）。
**两 field 的 `info` 内容 sha256 相同**（`bc792e26…`，561 B，逐字节相同）⇒ 几何与编码参数一致 ⇒ 同一解码口径适用。
⚠️ 全部为 2–6 KB 小对象 ⇒ 按 env_notes **L5**，**这些 B/s 只反映单件端到端，不得当作链路吞吐**。

---

## 3. 解码：格式规范 + **双实现逐字节互证**

### 3.1 规范来源（权威，非猜测）

M11 的 `tools/verify_m11_grid_alignment.py` L265–L268 登记：实测头部 uint32 序列
`[1, 16777744, 512, 33554996, 532, …]` **"不符合已知布局"**，并因 **PyPI 不可达、无 `bossdb`/`intern`** 而放弃。

本节点复核：**此刻 PyPI 可达**（`https://pypi.org/simple/cloud-volume/` → **HTTP 200**，TTFB 0.94 s），
⇒ M11 的"不可达"**当时成立、现已失效**（这是一条**可复现性**层面的更正）。

我取得并阅读了**权威实现源码**：
`compressed_segmentation-2.3.3.tar.gz`（26,759 B，sha256 `4575de23627f29e0…`），
其 `include/decompress_segmentation.h` 头注释指向
`https://github.com/google/neuroglancer/tree/master/src/neuroglancer/sliceview/compressed_segmentation`，
`src/decompress_segmentation.cc::DecompressChannel` 给出**逐行可读的格式定义**。关键约定：

1. 缓冲区开头 = `num_channels` 个 LE uint32 通道偏移；**单通道时就是 1 个 `1`**
   （该约定写在 `DecompressChannels` 的注释里）⇒ **实测头部首字 = 1，与规范吻合**；
2. `grid[i] = ceil(volume_size[i]/block_size[i])`；块头表 `2·grid[0]·grid[1]·grid[2]` 个 uint32，每块两项：
   `h0`：低 24 位 = `tableoffset`（uint32 单位），**高 8 位 = `encoded_bits`**；`h1` = `encoded_value_start`；
3. `encoded_bits == 0` ⇒ 该块**全为** `table[0]`（常量块）；
4. 表项 = `table[tableoffset + bitval·table_entry_size]`，`table_entry_size = ceil(sizeof(Label)/4)`
   ⇒ uint64 时为 2 个 uint32（低字 | 高字<<32）；
5. 位流：`bitpos = block_size[0]·((z−zmin)·block_size[1] + (y−ymin))·encoded_bits`，沿 **x** 递增 `encoded_bits`。

**⇒ 用它逐字回解 M11 的实测头部**（这是"已知布局"被找到的直接证据）：

```
通道内 C[0]=16777744 = 0x01_000210 ⇒ encoded_bits=1, tableoffset=0x210=528
                  C[1]=512        ⇒ encoded_value_start = 512
grid = (248/8=…) 对本 chunk volume (128,128,1)/block (8,8,8) ⇒ 16×16×1 = 256 块
⇒ 块头表恰好占 2×256 = 512 个 uint32 ⇒ encoded_value_start = 512 ✓ 逐值自洽
```

即 M11 观测到的 `[1, 16777744, 512, …]` **正是** `compressed_segmentation` 的首块头 + 位流起点。

### 3.2 双实现互证（承 M11 `dual_estimator` 纪律）

- **实现 A**（本节点独立实现，`tools/m12_b5_roi_mask_audit.py::decode_compressed_segmentation_independent`，
  纯 Numpy，逐行等价于上面 §3.1 的 C++ 语义）；
- **实现 B**（seung-lab 参考实现 `compressed_segmentation==2.3.3`，装在**独立新建** venv `/tmp/m12_b5_venv`，
  **未触碰任何既有 venv**）。

**结果：8/8 chunk 逐字节一致（`np.array_equal` = True）**，两 field 均然（field4 8/8、field2 8/8）。

**结构门**（实现 A 内建，越界即 `raise`，**不静默截断**）：
① 首字必须为 `1`；② 每块 `tableoffset/vstart` 必须在通道缓冲内；③ 每次表项/位流读取必须在内。
**8+8 chunk 全部通过**，无一次越界。

> **自查留档（本节点缺陷与修正）**：结构门②的初版写成
> `tableoffset + 2**encoded_bits·tes ≤ n_ch`，对 `0-128_128-256_0-1` 误报越界
> （实测 `max = 10528` vs `n_ch = 10526`）。原因是**我**假定了"`2**bits` 个表项全被使用"，
> 而规范只要求被**实际引用**的表项在界内。已改为"逐次读取时验界"。
> **该误报不是数据问题，是我的判据过强** —— 记此以免被误读为掩膜异常。

---

## 4. 自证表（每条都可复算；弱项如实标弱）

| # | 检验 | 结果（field4 主 field / field2 复现 field） | 判别力 |
|---|---|---|---|
| **S1** | 双实现逐字节互证 | **8/8 & 8/8 bit-exact** | 强（第三方实现 + 独立实现） |
| **S2** | 结构门（无越界读） | 全通过（16/16 chunk） | 强（错误布局极可能越界） |
| **S3** | **跨 chunk 标签 id 一致性**（10 条真缝） | **correct 均值 0.9789 / 0.9235** | 强，见下 |
| **S3a** | 负对照①：一侧标签**双射到不相交新 id** | **0.0000 / 0.0000** | 使 S3 坍缩 ⇒ S3 有判别力 |
| **S3b** | 负对照②：**快轴假设 x⇄y 互换** | **0.0000 / 0.0000**（共有 id 数 = 0） | 使 S3 坍缩 ⇒ **轴序被检验** |
| **S4** | id 空间语义 | 1388 个 id ∈ **[1,1389]（缺 1 号）** / 1413 个 id ∈ **[1,1413]（缺 0 号）** | 中：与文档"`mask_id smallint, unique per field`"一致；垃圾解码不会如此 |
| **S5** | 形态（应为胞体尺度） | 面积 p50 = **26 px**（2.5 µm/px ⇒ 直径 ≈ **14 µm**）/ p50 = 24 px | 中 |
| **S6** | **配准 + 语义锚**（只用电影，不引用掩膜元数据） | **3/3 指标上 (0,0) 是平移网格唯一最优，`1/81`，领先次优 15.7% / 7.6% / 4.2%**；field2 仅 `1/25`（领先 0.9%），另两指标 3/25、6/25 | **field4 强；field2 弱（如实登记）** |
| **S7** | 独立细胞检出（规则先定死） | 含 ≥1 检出峰的 ROI 占比 **0.9434 / 0.9223**，平移对照 0.45–0.56 | **弱**：随机放置的**机会基线** = 0.7792 / 0.8307 ⇒ 富集仅 **1.21× / 1.11×** |
| **S8** | 与 M11 **已发布** proxy 值的口径一致性 | 本管线复算 20×20 分箱静默 = **0.49**，M11 定稿 **0.49** ⇒ 逐值吻合 | 强（跨节点数值一致性） |

**S3 的判别力证据（逐缝）** —— 8 个 chunk 是**各自独立的 S3 对象**，只有正确解码才能让**跨界细胞在两侧同 id**：

| 缝 | correct | 负对照①（不相交重标号） | 负对照②（轴序互换） | 共有 id 数 |
|---|---|---|---|---|
| x=127\|128, y=[0,128) | 1.000 | 0.000 | 0.000 | 12（对照 0） |
| y=127\|128, x=[0,128) | 0.983 | 0.000 | n/a（方形块无差异） | 11（对照 0） |
| x=127\|128, y=[128,256) | 1.000 | 0.000 | 0.000 | 8（对照 0） |
| y=255\|256, x=[0,128) | 0.966 | 0.000 | n/a | 6（对照 0） |
| x=127\|128, y=[256,384) | 0.951 | 0.000 | 0.000 | 13（对照 0） |
| y=383\|384, x=[0,128) | 1.000 | 0.000 | 0.000 | 11（对照 0） |
| x=127\|128, y=[384,440) | 1.000 | 0.000 | 0.000 | 2（对照 0） |
| y=127\|128, x=[128,248) | 1.000 | 0.000 | 0.000 | 9（对照 0） |
| y=255\|256, x=[128,248) | 0.912 | 0.000 | 0.000 | 12（对照 0） |
| y=383\|384, x=[128,248) | 0.978 | 0.000 | 0.000 | 9（对照 0） |

> **自查留档（本节点第二处缺陷与修正）**：负对照①的初版把标签**重排到同一 id 集合内**，
> 于是"两侧共有 id 集合"这一列**恒等、无判别力**。已改为**双射到不相交的新 id**（+1,000,000），
> 使该列也坍缩。**该缺陷由本节点自查发现并修正**，修正前的该列值不作证据。

### 4.6 轴序是怎么被钉死的（这是本节点最容易被悄悄搞错的一步）

掩膜的 array 轴序由**编码方**定义（位流沿 `x` 递增），而 chunk 键是 `x_y_z`。
二者若不匹配，**块头表依然能正常解析**（`grid` 的**乘积**不变，如 15×16 = 16×15），
⇒ **结构门 S2 单独不足以发现轴序错误**。真正钉死它的是：

1. **S3 用了块的不对称性**：`128-248_*` chunk 的 x 跨度是 **120**、y 跨度是 **128**。
   若快轴实为 y，则用 `volume_size=(120,128)` 解码会把体素位置系统性打乱 ⇒ 缝一致性应**坍缩**。
   实测**未坍缩（0.951–1.000）**；
2. **S3b 负对照**显式跑"快轴 x⇄y 互换"的解码 ⇒ 所有缝 **0.000**、共有 id **0** ⇒ 检验确有判别力；
3. **S6 配准锚**独立复认：掩膜与电影在 `(0,0)` 处**唯一最优** ⇒ 两卷 `(x,y)` 语义一致。

---

## 5. (b) 的**真实 ROI 口径**重算结果

**口径（逐项写清 —— 即 J2 条件 a)）**

| 口径项 | 取值 / 说明 |
|---|---|
| 帧率 | **6.300006 Hz**（`frame_interval_s = 0.15873`，取自 `attributes.json`） |
| 观测窗 | field4：x∈[0,200), y∈[0,200), t∈[0,2000)（取自 `data/m11_functional_subset_manifest.csv` 的 S3 键 `0/0/0/{0..199}`）；field2：x∈[0,100), y∈[0,100)，同 2000 帧 |
| **ROI 掩膜来源** | `field_seg_masks/<field>/2500_2500_10000/`（uint64 `compressed_segmentation`，8 chunk，248×440，2.5 µm/px）；出处与 sha256 见 §2.2；**解码自证见 §3–§4** |
| 单元定义 | **每个非零 label = 1 个 ROI**；ROI 时间轨迹 = **ROI 内像素 dF/F 的等权均值** |
| 是否加权 | **未加权**（掩膜卷为二值；CaImAn 的 `weights` 不在该卷内）⇒ 与数据库 `Mask.weights` 口径**不同**，已登记 |
| 是否平滑 | **无时间平滑**；**无空间平滑**（ROI 内均值即空间聚合） |
| 神经域反演 | **未做**（本口径不做 neuropil 反演/去混叠）⇒ 如实登记为未覆盖维度 |
| 事件检测阈值 | 逐像素 **F0 = 时间 20 分位**；`dF/F=(F−F0)/F0`；阈值 = **中位 + 3·MAD**（与 M11 `build_m11_calcium_silence.py` **同源同码**） |
| 实测阈值 | field4：`dff_median=0.423285, dff_mad=0.385062, thr=1.578470`；field2：`0.606383 / 0.567295 / 2.308268` |
| 静默口径 | **`zero_count`**：观测窗内 `count_i == 0`（判据表 `silent_def_primary`） |
| 哨兵 | int16-BE 直读；**实测饱和（=32767）占比 = 0.000000**（两 field） |

**结果**

| 量 | field4（主 field，R18 定案） | field2（复现对照） |
|---|---|---|
| 场内 ROI 总数 | **1388** | **1413** |
| **完全落窗**的 ROI 数 | **432** | **87** |
| **`silent_frac`（真实 ROI · zero_count）** | **0.101852** | **0.229885** |
| 面积下限敏感性（≥10 px / ≥20 px） | 0.102804 / 0.112637 | 0.229885 / 0.231707 |
| 中位事件数 / ROI | 46.5 | 8.0 |
| **跨时间窗离散度**（前 1000 帧 / 后 1000 帧） | 0.150463 / 0.252315（**Δ=0.101852**） | **0.689655** / 0.252874（**Δ=0.436781**） |
| 同尺寸（100×100）窗口对照 | 55 ROI，**0.000000** | 87 ROI，**0.229885** |

**跨 field 离散度**：**2.257×**（0.101852 vs 0.229885）。

### 5.1 ⚠️ 一条必须随数字出现的口径差：**proxy 口径高估静默 ~4.8×**

- M11 定稿的 **proxy（20×20 px 分箱）** field4 值 = **0.4900**；
- **本管线在同一电影、同一阈值、同一窗口下复算 proxy = 0.49（S8，逐值吻合）**；
- 同一管线下把单元换成**真实 ROI** ⇒ **0.101852**。

⇒ **口径差 = 0.49 / 0.1019 ≈ 4.81×**。机制：把 400 px（50×50 µm，含约 6–10 个细胞）的 dF/F
**先平均再比阈值**，会把瞬变平滑掉 ⇒ 分箱轨迹极少越阈 ⇒ **proxy 系统性高估"静默"**。
**这不是"哪个对"的问题，而是"proxy 不能替代真实 ROI 做精确落入判定"的定量证据**
（正对应 L115/L116 的 `wrong_usage_consequence`：「用代理口径的带去做精确落入判定 ⇒ 假 fail 或假 pass」）。
（仅对 `zero_count` 口径成立；M11 的 `m10_compat` 口径在 proxy 上**退化饱和为 1.0**，不参与本比较。）

### 5.2 其他必须随数字出现的限制

1. **跨时间窗离散度很大**（field4 Δ=0.102；field2 Δ=0.437）⇒ 本口径**强依赖观测窗与基线口径**
   （逐像素 F0 取整段 20 分位，若信号有慢漂移，前后半段的 dF/F 标度不同）。
   **⇒ 用于 (b) 判定时必须连同观测窗一起登记**，不得只报一个数。
2. **"完全落窗"是硬门槛**：场内有 956（field4）/ 1326（field2）个 ROI **部分落在窗外**，
   已**排除**（截断轨迹会低估事件数、高估静默）。这是 n 从 1388→432 的原因，**不得**把它读成"ROI 丢了"。
3. **钙事件 ≠ spike rate**（承 M11 `SILENCE_LIMITATION`）：与电生理域带**不得混用**（J2 条件 c)。
4. field2 的 `silent_frac` 建立在 **87 个 ROI** 上（小 n）⇒ 跨 field 的 2.257× **含明显抽样误差**，
   且 field2 的配准锚（S6）**只部分通过** ⇒ field2 数字的**可信度低于 field4**（如实登记，不掩盖）。

---

## 6. 这能让 (b) 的**哪一行**转 `final=1`？判据（**我不代改 CSV**）

> 🔴 **本节点未修改 `data/m11_p5_targets.csv`**（`git status --porcelain` 无既有件改动；见 §0）。
> 以下为**判据**与**建议值**，须由**主 agent 裁决**后另行写入。

`data/m11_p5_targets.csv` **L114** 给出的"缺一即不得 final=1"的验收条件 a)–e) 逐条对应：

| 条件 | 原文要求 | B-5 后的状态 |
|---|---|---|
| **a)** 口径逐项写清（帧率 / **ROI 掩膜来源** / 神经域反演 / 是否平滑 / 事件检测阈值） | — | **✅ 已满足**（§5 表逐项；ROI 掩膜来源含 S3 键 + sha256 + 解码规范 + 双实现自证）。⚠️ 其中"**神经域反演**"须明确写 **未做**（本口径为未反演的 dF/F） |
| **b)** 给**跨 field 与跨时间窗**离散度（≥2 field；只有单 field 时须写明场内外推限制） | — | **✅ 已满足**：2 field（0.1019 / 0.2299，**2.257×**）+ 跨时间窗（§5 表）。⚠️ 须并把"时间窗离散度大"写进 note（§5.2.1） |
| **c)** 与电生理域带并列且**不得混用** | — | **✅ 已满足**：口径名显式；承 `calcium_vs_spike_caliber_gap` |
| **d)** 哨兵 / below-baseline 样本处置口径**已声明** | — | **⚠️ 部分**：哨兵饱和实测 **0.000000** 且已登记；但 `microns_sentinel_mask_policy`（L114 原文称"未决"）**本节点未裁决、也非本节点职权** ⇒ 仍须主 agent 先裁决或两侧并列 |
| **e)** 若 **ROI 提取管线不可得** ⇒ 判 `unavailable` 并写明已覆盖的维度 | — | **✅ 前提被否证**：ROI 提取管线**可得且已自证**（§3–§4）⇒ 该条**不再触发** `unavailable` |

**⇒ 建议写入（提案，非本节点写入）** —— 落于
`data/m12_b5_roi_mask/proposed_p5b_criteria_rows.csv`（sha256 `f917b7e2…`）：

| 目标行 | 建议 `value` | 单位 | 依据 |
|---|---|---|---|
| `m11_p5_targets.csv` **L115** `silent_frac_calcium_band_lo` | **0.1019** | fraction of units（真实 ROI · zero_count） | 主 field（R18 定案）实测；S6 锚 pass；n = 432 |
| `m11_p5_targets.csv` **L116** `silent_frac_calcium_band_hi` | **0.2299** | 同上（复现 field） | field2 实测；n = 87；S6 锚仅部分通过 ⇒ **建议主 agent 判它是否够格进带** |

**🔴 三条必须由主 agent 裁决、本节点不代决的点**：
1. **L116 是否够格**：按 **b)** 的"≥2 field 离散度"要求，L116 是**必须**的第二点；
   但 field2 的配准锚只部分通过、n 仅 87。**"给离散度"与"该点够不够格进带"是两个判断**，后者是主 agent 的。
2. **band 的语义**：L115/116 是"带 lo / hi"。本节点的候选是**两个 field 的实测值**（0.1019, 0.2299），
   **不是**统计置信区间；是否等同于"带"须主 agent 界定。
3. **`coverage` 表述的更新**（**R9/R18 相关**）：L115/116 的 note 现写
   `coverage = covered 7 / missing 1 (real ROI mask)`。本节点把**该 missing 维补齐**了，
   但**不**主张"8/8 全覆盖"——因为 **`weights` 加权口径**与**神经域反演**仍未覆盖（§5、§7）。
   建议主 agent 将 coverage 改写为"**real ROI mask：covered；加权/反演：未覆盖**"，而**不得**写成整体通过。

---

## 7. 未确认项（逐条，含卡在哪一行/哪个字段）—— **不得把本节当噪声**

| # | 未确认项 | 卡在哪 | 影响 | 我做了什么 |
|---|---|---|---|---|
| **1** | **掩膜卷与数据库 `Mask` 表（CaImAn `mask_id`）的身份链** | 数据库权威载体 = `two_photon_processed_data_and_metadata/database_v8/functional_data_database_sql_dump_v8.sql`，其 **`Content-Length = 116,199,864,342 B`（116.2 GB）**，`ETag "…-6927"`（**多段上传 6927 部**），`x-amz-storage-class: INTELLIGENT_TIERING` | 我证明的是"**这是一套与主 field 同格、胞体尺度、跨 chunk id 一致的 segmentation**"，**未**证明"它与 `Fluorescence`/`ScanUnit` 引用的 `mask_id` 是同一套编号"。**这是本节点唯一实质性的证据缺口** | 见 §7.1：**实测吞吐 + 结构探查**后判定本轮不可达 |
| **2** | `Mask.weights`（CaImAn 加权）未使用 | 同 #1（`weights longblob` 在该 dump 内） | 轨迹为**等权均值**，与 CaImAn 官方加权轨迹**不同** | 已在 §5 口径表显式登记"未加权" |
| **3** | 神经域反演/去混叠**未做** | 本节点范围 | 不是"真实活动区"的完整口径 | 已显式登记；**不得**写成已覆盖 |
| **4** | `microns_sentinel_mask_policy` 未决 | `data/m11_p5_targets.csv`（L114 条件 d 原文） | 影响 (b) 是否满足条件 d) | **非本节点职权**，报请主 agent |
| **5** | field2 电影子集**无逐件 manifest** | 现盘只有 `data/m11_functional_subset_manifest.csv`，其内容为 **field4**（`series=session4_scan7_field4`）；**全盘无任何文件提及 `session5_scan3_field1`** | field2 窗口原点 `(0,0,0)` 只能由**工具构造**推出（`tools/fetch_m11_microns.py` L537 `base = ".../%s/0/0/0/"`）与 chunk 头 `dims=(100,100,1,10)`、M11 标签"(0,0,0)"，**无 sha 级 provenance** | 已登记；建议主 agent 要求补 manifest |
| **6** | **M11 侧既有件完整性问题（新发现，非本节点造成）** | `data/m11_raw/microns/functional_connectomics/node_data_v1.pkl` 现盘 **2,322,432 B**，而 `tools/parse_m11_functional_connectomics.py` L44 期望 **294,635,032 B**；同目录 `parts/*.pkl.p0..p7` 合计 **10,067,968 B**（亦远小于 294.6 MB）。我实测 `pandas.read_pickle` 抛 **`_pickle.UnpicklingError: pickle data was truncated`** | **该件是 INCOMPLETE 假件**（承 L6/L14 家族）；**凡引用它的下游断言都不成立**（`tools/validate_p11_data_gate.py` L538 在读该件的 `verdict`） | 只**读取**、未修改；且 `functional_connectomics/waiter.log` 记 **DQ-4 裁决"不得再起该对象的下载或 HEAD 轮询"** ⇒ 我**遵守该裁决，未取该对象** |
| **7** | field2 的配准锚只**部分**通过 | 见 §4 S6 | field2 的 (b) 值可信度低于 field4 | 如实标 **FAIL／部分不通过**，verdict 写死"**不得当作已配准**" |
| **8** | 独立细胞检出（S7）判别力**弱** | 机会基线 0.7792/0.8307，富集仅 1.21×/1.11× | **不得**把它当作主要证据 | 如实标弱；主证据仍是 S1–S3b + S6 |
| **9** | M12 侧 `m11_microns_protocol.json` 的 `n_masks_derived` 字段**名实不符** | `tools/fetch_m11_microns.py` L390–391 把 `n_masks_derived` 算成 `(size[0]/chunk[0])·(size[1]/chunk[1])`，对本 field 恒为 **3**（= **chunk 网格数**，1×3），而**真实 label 数是 1388** | 该字段**不是**掩膜计数；若被下游当掩膜数引用即错误 | 只读取、未改；登记备查（**本节点实测 1388**，与 3 相差 463×） |

### 7.1 为什么 #1 在本轮**不可达**（实测数字，不是推测）

| 步骤 | 实测 |
|---|---|
| 整件吞吐 | Range 读 4 MiB：`HTTP 206`，`size=4,194,304 B`，**TTFB = 1.0018 s，total = 111.478 s**（⇒ 传输段 ≈ 110.5 s，端到端 **37,624 B/s**） |
| 整件 ETA | 116.2 GB ÷ 37.6 KB/s ≈ **3.09×10⁶ s ≈ 35.8 天** ⇒ **远超任何预算**（且该对象为 `INTELLIGENT_TIERING`；**吞吐不外推**纪律在此反而由**同一对象**的实测值直接给出） |
| 结构探查 | 读前 4 MiB 实测：dump 为 **`mysqldump 10.13`**，库名 **`microns_phase3_nda`**；前 4 MiB 内**只有 1 张表 `activity`**（表结构标记 @1785，其后 5 条 `INSERT INTO \`activity\``）⇒ **首表就 > 4 MiB**，且表是**逐表整体倒出** |
| ⇒ 定向 range 取 `Mask` 表的可行性 | 需先在 116 GB 内**定位** `Mask` 表起点；由于窗口内可能**不含任何表名标记**（如本 4 MiB 段），不存在可用的单调谓词 ⇒ 定位所需的探测次数**无法给出上界**；再叠加定位后仍需前向扫描（37.6 KB/s）⇒ **本轮不可控** |
| 更小的替代载体 | `database_v7` 的 dump **122,787,143,696 B（122.8 GB）更大**；`functional_data_database_container_image_v8.tar` **103,798,314,496 B**；`node_data_v1.pkl` **294.6 MB** 但**受 DQ-4 裁决禁止**（且现盘件为 INCOMPLETE 假件） ⇒ **无更小权威载体** |

**⇒ 结论：#1 在"不降规模 + 不用云 GPU + 唯一权威载体 116 GB / 37.6 KB/s"的条件下，本轮不可解。**
这不是"实测为负"，而是"**预算-可及性为负**"——按 M12 纪律，**不得**写成"该数据不存在"。

---

## 8. 降级路径（若主 agent 不接受 §7 #1 的身份链）

按 **L114 条件 e)** 的原文，若判 ROI 提取管线**不可得** ⇒ 只能判 `unavailable` 并写明已覆盖维度。
但本节点已把该前提否证，故降级路径是"**口径替代**"而非"翻转结论"：

| 降级选项 | 是什么 | **为什么不能替代真实掩膜** | 会把 (b) 推到哪个四态 |
|---|---|---|---|
| **R1：维持 M11 的 proxy（20×20 分箱）** | 现 L115/116 的参照值 0.48/0.49 | 实测**高估 4.81×**（§5.1）；且 L115/L116 的 `wrong_usage_consequence` 已先验写明"用代理口径的带做精确落入判定 ⇒ 假 fail 或假 pass" | 若强行 `final=1` ⇒ **违反 L114 条件 a)**（ROI 掩膜来源写不出）；应维持 **`unavailable`** |
| **R2：接受本节点掩膜、但**只用 field4**（单 field）** | 0.1019 | 不满足 L114 条件 **b)** 的"≥2 field" | 按清单 §2.4 的样本量门 ⇒ 只能标 **`partial`**（并须写明"场内外推限制"） |
| **R3：接受掩膜 + 2 field（本节点推荐）** | 0.1019 / 0.2299 | 满足 a)–c)；**d) 仍待主 agent 裁决** | 条件 d) 未裁决前 ⇒ **`partial`**；d) 一并裁决后可达 **`measured/final=1`** |
| **R4：拒绝任何 ROI 掩膜** | — | 与 §3–§4 的实测证据冲突（须给出反证才能成立） | **`unavailable`**（维持现状；**但此时不得再写"未自证解码"为理由**，因为已自证） |

**⇒ (b) 的四态判据**：`L114` 的 a)–e) 是**硬门**（"缺一即不得 final=1"）。
本节点**独立满足 a) b) c)**、**否证 e) 的前提**、**d) 仍有 1 项待裁决** ⇒
**(b) 现状 = `partial`（3/4 条件就位，1 项待裁决）**，**不是** `measured/final=1`；
`final=1` 只在 **d) 被裁决后**可达。

---

## 9. 边界：本节点**没有**做什么

- **未改** `data/m11_p5_targets.csv`（也未改任何 `m11_*`/`m12_*`/`tools/`/`src/` 既有件）；
- **未**跑 MPS/GPU、**未** kill 进程、**未**起服务端、**未**动 L5 作业；
- **未**取 `node_data_v1.pkl` / `edge_data_v1.pkl`（DQ-4 裁决）；
- **未**取 116 GB / 122.8 GB 的 SQL dump（§7.1 实测不可达）；
- **未**做神经域反演、**未**用 CaImAn `weights` 加权、**未**做运动校正重算；
- **未**替主 agent 裁决 L114 条件 d)、band 语义、L116 是否够格、coverage 的最终措辞。

---

## 10. 复算命令（逐项；**无 `2>/dev/null`**）

```bash
cd /Users/weidong/ai/small_world

# 0) 硬约束复核（应看到 L5 作业在跑；本节点未 kill）
pgrep -f run_m11_p5_arms

# 1) 独立 venv（**新建，不触碰既有 venv**）+ 参考实现（仅用于双实现互证）
/Users/weidong/ai/small_world/.venv-m11/bin/python -m venv /tmp/m12_b5_venv
/tmp/m12_b5_venv/bin/pip install --no-input --disable-pip-version-check compressed-segmentation numpy

# 2) 取第二个 field 的掩膜（含逐件三元组 + 完整性门）
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 ./.venv-m11/bin/python \
  neural_exploration/tools/probe_m12_b5_roi_mask_sources.py

# 3) 解码 + 自证 + (b) 重算（**带**双实现互证；缺参考实现即非零退出，不静默降级）
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 /tmp/m12_b5_venv/bin/python \
  neural_exploration/tools/m12_b5_roi_mask_audit.py --crosscheck-ref

# 3') 只用盘上依赖（不装任何东西）——此时 ref_impl_bitexact = null（如实置空，不假装通过）
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 ./.venv-m11/bin/python \
  neural_exploration/tools/m12_b5_roi_mask_audit.py

# 4) 出处复核：掩膜 chunk 与线上逐字节同否
curl -sS -L --max-time 60 -o /tmp/b5_refetch.bin \
  'https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/functional_data/field_seg_masks/session_4_scan_7_field_4/2500_2500_10000/0-128_0-128_0-1'
shasum -a 256 /tmp/b5_refetch.bin \
  neural_exploration/data/m11_raw/microns/field_masks/0-128_0-128_0-1.bin

# 5) §7.1 的吞吐实证（116 GB 对象的 4 MiB Range 读；TTFB/传输段/大小三元组）
curl -sS -L --max-time 900 -r 0-4194303 -o /tmp/b5_sql_head.bin \
  -w 'http=%{http_code} size=%{size_download} ttfb=%{time_starttransfer} total=%{time_total} speed=%{speed_download}\n' \
  'https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/functional_data/two_photon_processed_data_and_metadata/database_v8/functional_data_database_sql_dump_v8.sql'

# 6) §7 #6 的假件实证（应抛 UnpicklingError: pickle data was truncated）
./.venv-m10/bin/python -c "import pandas as pd; pd.read_pickle('neural_exploration/data/m11_raw/microns/functional_connectomics/node_data_v1.pkl')"
wc -c neural_exploration/data/m11_raw/microns/functional_connectomics/node_data_v1.pkl \
      neural_exploration/data/m11_raw/microns/functional_connectomics/parts/*.pkl.p*
```

---

## 11. 产物清单（`wc -l -c` + `shasum -a 256`，本轮实测）

### 11.1 工具（新增）

| 文件 | `wc -l -c` | sha256 |
|---|---|---|
| `neural_exploration/tools/m12_b5_roi_mask_audit.py` | `760 40969` | `8fda4a5a8a697dd234a0782624c1943a0d01e8f3d5c55199c62ce70074a265d8` |
| `neural_exploration/tools/probe_m12_b5_roi_mask_sources.py` | `144 7330` | `fcade3cdb7b134443d60ba9e23f81513585a5017c24f15f936e8d878f2d0b86a` |

### 11.2 判定与数据（新增）

| 文件 | `wc -l -c` | sha256 |
|---|---|---|
| `neural_exploration/data/m12_b5_roi_mask/m12_b5_roi_mask_audit.json` | `1448 34244` | `3d7c59520085e9f5fd5bbf8020216da6596db44cddd5bf362492f91eb2407a1d` |
| `neural_exploration/data/m12_b5_roi_mask/real_roi_rows_field4.csv` | `1389 72710` | `75cc2743d26c6d2d3051850204e6be50bd38f387046d8f3b05c3365cc8659b94` |
| `neural_exploration/data/m12_b5_roi_mask/real_roi_rows_field2.csv` | `1414 64490` | `b1203e964ae7e56ba7fc36b95a470eeb9d94457bc49317fee2435850c5c1b9db` |
| `neural_exploration/data/m12_b5_roi_mask/proposed_p5b_criteria_rows.csv` | `5 1185` | `f917b7e2e15aa76fbde54b9c7d382fc4323bcb0b4c2125d5bdc1fb8c0db1b48c` |
| `neural_exploration/data/m12_b5_roi_mask/field2_fetch_manifest.json` | `185 6682` | `3f745582ad071088f5aee559d83c0ff801d27864c7bb0ddff631607509ff1153` |
| `neural_exploration/data/m12_b5_roi_mask/info_session_4_scan_7_field_4.json` | `37 561` | `bc792e267f5fb66c8d188d96b6bead8b47bb9cdbd4896f293f67cf5ff1faae22` |
| `neural_exploration/data/m12_b5_roi_mask/info_session_5_scan_3_field_1.json` | `37 561` | `bc792e267f5fb66c8d188d96b6bead8b47bb9cdbd4896f293f67cf5ff1faae22` ← **与上行逐字节相同** |
| `neural_exploration/data/m12_b5_roi_mask/roi_label_volume_248x440_field4.npy` | `177 436608` | `61c11e67d6d7cc38bdf0084b0c3c41beae10118fc47e62540316d7e61e66184e` |
| `neural_exploration/data/m12_b5_roi_mask/roi_label_volume_248x440_field2.npy` | `177 436608` | `3cc1c173ccb849426dca794856e7cddbccf3ebfd986203bb2c44ff6c112f883b` |
| `neural_exploration/data/m12_b5_roi_mask/field2_masks/*.bin`（8 件） | `32,177 B` 合计（`wc -c` 实测） | §11.5 |

### 11.3 掩膜 chunk 的 sha256（field4，M11 落盘件，本节点复核；与线上件逐字节一致）

| chunk | sha256 |
|---|---|
| `0-128_0-128_0-1.bin` | `cfbafb134850af6d3c9e8f14bcad11fc5718378ca88564a55e19107eba4bec60` ← **线上复取同值** |
| `0-128_128-256_0-1.bin` | `14f520e33d02481527c100565029f959876dd0809ab782386742504cbdab08b6` |
| `0-128_256-384_0-1.bin` | `5223588dcdc61ffed5bda4dc9f92b8d6040cb2c9753a4c80d8ea6840dddd4f40` |
| `0-128_384-440_0-1.bin` | `2858d1c3567e5dc3330192c2c6a5b755298c167a7fcf8905550d0113f6b01a5b` |
| `128-248_0-128_0-1.bin` | `5974a7c0ab6158b054873448526527fdd582d50bca10683b67dfe94cd092f7c8` |
| `128-248_128-256_0-1.bin` | `fc48ff4538e0390324d7999af80085259bcf3dc5bae5e91f41a856dda13190a6` |
| `128-248_256-384_0-1.bin` | `4af04b14ab329b682c824d7467c210e1152832413dafc00b435a29804668f781` |
| `128-248_384-440_0-1.bin` | `d3cd9885a660a56bf3cb2f1f7e768954691be5efeb0d739abb68fa15e5959012` |

### 11.4 标签体（`npy`，uint32，shape `(248,440)`）

```
61c11e67d6d7cc38bdf0084b0c3c41beae10118fc47e62540316d7e61e66184e  roi_label_volume_248x440_field4.npy
3cc1c173ccb849426dca794856e7cddbccf3ebfd986203bb2c44ff6c112f883b  roi_label_volume_248x440_field2.npy
```

### 11.6 本判定文档

| 文件 | `wc -l -c` | sha256 |
|---|---|---|
| `neural_exploration/docs/m12_b5_roi_mask.md` | 见下方复算命令 | **自指值：任何编辑都会改变它** ⇒ 本表**故意不写死**；交付时由主 agent 以复算命令当场取值 |

```bash
shasum -a 256 neural_exploration/docs/m12_b5_roi_mask.md
wc -l -c neural_exploration/docs/m12_b5_roi_mask.md
```

### 11.5 field2 掩膜 chunk（**本节点新增取数**，8/8 COMPLETE）

| chunk | sha256 |
|---|---|
| `0-128_0-128_0-1.bin` | `3e71d354a892af9bb7db8a363113de5052feb5f047cb05d394b12fa2b30d9ca9` |
| `0-128_128-256_0-1.bin` | `07f61db2fafe105841c5d416718c30e8748f6a90b856b0968cb12e3f8a919c09` |
| `0-128_256-384_0-1.bin` | `47757931cfc237d7482af04261028a6c3dffac8e0780d053bdc6d36c5385ca0a` |
| `0-128_384-440_0-1.bin` | `c79e44346a189da8d735bb8504f1ff63c355dfaebb1287b1fbb776cb6db94a4a` |
| `128-248_0-128_0-1.bin` | `a26c2d14be7fab952bbac4a812bb7877e6ca50ddf343e2276883fa8abbaca9e2` |
| `128-248_128-256_0-1.bin` | `2fbdf9fc0b4418ecc5738ab3876a0ba1e896a116d879faeac1371963f7c873c8` |
| `128-248_256-384_0-1.bin` | `61523179e635668ad24ecd84f23261e550d34cc657ed82a566a1e4a0dc411001` |
| `128-248_384-440_0-1.bin` | `fd07568b65a63357fa9c43d8d52038cd3a0262724f2189a4cfb2adeec14ae660` |

---

## 12. 一句话给主 agent

**(b) 的"唯一缺失维 = 真实 ROI 掩膜"已经落地：掩膜可解码、可自证，且其口径代入后 field4 = 0.1019
（而 M11 的 proxy 0.49 在同一管线下被复现且被证明高估 4.81×）。
请裁决：接受本证据链 + 裁决条件 d) ⇒ L115/L116 可转 `final=1`；
若对"掩膜卷 ↔ 数据库 `mask_id` 的身份链"要求更硬的证据，该证据在 116.2 GB / 37.6 KB/s 下
本轮不可达（§7.1），需要新的取数许可或维护方提供 per-field 掩膜导出 —— 该维在本轮**到此为止**。**
