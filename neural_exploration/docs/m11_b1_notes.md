# M11 · B1 节点记录（J2：(b) 钙域静默比例口径）

> 本文件为 **B1 节点**的工作记录（配合 `docs/m11_b3_notes.md`（B3）/`m11_b3b_verification.md`（B3b））。
> **纪律**：本文件**只增不改**；每条给出**可复算命令**；四态（pass/fail/invalid/unavailable）与
> 证据级别（measured/derived/literature/unavailable**分列**）**不得混同**。

---

## N1. J2 任务与状态（2026-10-12）

★ **状态澄清（重要，避免重复劳动）**：本任务曾被主 agent 两次派发，而**其自身验收记录 §3az/R57 已把 J2 记为「收口 5/5」，**
并把 5 个预留 key 定稿为 `silent_frac_calcium_band_lo=0.4800` / `_hi=0.4900`。
⇒ 本轮**不复述、不重裁定**，只做两件事：**(A) 按本次派发的 5 项验收条件逐条回执 + 补齐真缺口**；
**(B) 把口径与自证固化到可复算工具**（`tools/validate_m11_calcium_silent.py`）。

**真缺口（本轮新交付，此前确实没有）**：
1. **M10 兼容口径 `mean(rate_i < 0.5 Hz)` 从未算过**（判据表 `silent_def_m10_compat_hz=0.5` 要求**并列**）；
2. **解码自证**此前只在别处做过，本口径工具内**没有**（本次补：F 序签名 + 判别性负对照）；
3. `docs/m11_b1_notes.md`（本文件）此前**不存在**；
4. **rate 侧预留 key 能否填值**的判据此前未论证（本轮给出结论：**不能**，见 N4）。

## N2. 5 项验收条件 —— 逐条回执

| # | 条件 | 状态 | 证据 / 实现 |
|---|---|---|---|
| 1 | **两口径并列**（主 `zero_count` + M10 兼容 `rate<0.5 Hz`），定义与阈值写明 | ✅ | 主口径 **0.4900**（field4）/ **0.4800**（field2）；M10 兼容 **1.0000** / **0.8800**；事件阈值 = dF/F 中位 + **3·MAD**（**像素级统计、施加于代理单元自身轨迹**）⇒ 三口径（zero_count / M10 兼容 / 全视野）**分列**，不混用 |
| 2 | **钙域与电生理口径不可直接混比**必须写入 provenance/scope_note | ✅ | 新增行 `silent_frac_calcium_two_caliber_scope_note`（三条理由：指示剂衰减合并事件 / 阈值改变"零计数"含义 / 钙事件率 Nyquist **3.150003 Hz** 折叠）；JSON `caliber_note.ephys_incomparable` |
| 3 | **新增行、append-only**：旧行 value 逐字保留、`final=0` 不变，另加更正/新增指针 | ✅ | 判据表 `data/m11_p5_targets.json`：**101 → 114 行（新增 13 行）**，`final=0` **16 → 29**；**旧行 value 未变 = True**（脚本内逐键比对并落盘断言 `J2.6`）；旧行 `silent_frac_*` 的 `unavailable`/`final=0` **逐字保留**，新行 note 内写明指针 |
| 4 | ROI 掩膜不可得 ⇒ 写清"已覆盖维度 + 缺哪一维"；**不得**因未覆盖一维就写"不可用" | ✅ | 新增行 `silent_frac_calcium_coverage_register`：**已覆盖 7 维 / 缺 1 维（真实 ROI·神经毡掩膜）**；四态 = `partial（三口径 measured + 一维 unavailable）`，**未写整体 unavailable** |
| 5 | **不得使用未自证的数据/掩膜**；用掩膜须给来源与自证证据 | ✅ | **未使用任何掩膜**（`field_seg_masks` 的 `compressed_segmentation` 本节点**未自证解码** ⇒ 按「未自证即不使用」不用其像素值）；解码本身**自证**：F 序 `lat_x=0.7769`（命中定案 **0.777**）、C 序负对照 `lat_x=0.3787`（判别性 **0.3983**） |

## N3. 四态判定（精确，不得外推）

| 项 | 四态 | 值 / 说明 |
|---|---|---|
| (b) 静默比例 · per-ROI 代理 · **主口径 zero_count** | **`measured`** | field4 **0.4900** / field2 **0.4800**，跨 field **1.0208×**（= R57 定稿带 `[0.4800, 0.4900]`） |
| (b) 静默比例 · per-ROI 代理 · **M10 兼容 rate<0.5 Hz** | **`measured` 但不可用（退化）** | field4 **1.0000（完全饱和）** / field2 0.8800 ⇒ 跨 field 差 **12.0 pp**（主口径仅 1.0 pp）⇒ **该档在本数据上无判别力**，不得作带 |
| (b) 静默比例 · **全视野**（帧为单元） | **`measured` 但不得作判据** | field4 0.8915 / field2 0.1465 ⇒ 跨 field **6.0853×**（差异由视野尺寸/活动水平主导：field2 视场 1/4、事件密度高 9.6×） |
| (b) 静默比例 · 全视野（像素为单元） | `measured`（并列留档） | field4 **0.0052** / field2 **0.0000** |
| **(b) · 真实 ROI / 神经毡掩膜口径** | **`unavailable（掩膜未自证）`** | **仅此一维缺**；`field_seg_masks` 无官方客户端、PyPI 不可达 |
| **rate 侧预留 key**（`rate_median_hz_measured_calcium_band_lo/hi`） | **`unavailable`（**保持**，本轮**不可**填值）** | 见 N4：率是**绝对量**，随未知"每箱神经元数"缩放 ⇒ **代理口径不可转移** |
| 跨 field 离散度 | `measured` | 本口径 1.0208×（另有 dwell 1.125× / τ 1.01×） |
| 电生理域静默比例（`silent_frac_ephys_lo/hi`） | `unavailable`（**B3 表内另一行**） | 本轮**未**取得；本轮也**未**用它替代钙域带 |

> **必须随带引用的四条限制**（与 R57 一致）：① 单元为**空间分箱代理**，代理↔ROI 偏移量**未测** ⇒ 带宽须叠加未量化口径余量；
> ② 样本量 **thin**（100 / 25 代理单元，B4 门 65/130/500）；③ 电生理并列**仅作量级**（不同量纲）；
> ④ **观测窗**口径（每 field 前 2,000 帧 / 317.46 s），非全场。

## N4. ★ 本轮新增的关键判据：**静默比例可转移，率不可转移**

同一份数据、同一代理口径下：
- **静默比例** = "零事件单元占比" ⇒ **对「每箱神经元数」不敏感**（零事件是**定性事实**；把一箱再切成两箱，两边都零事件）
  ⇒ 代理口径的静默比例**可以**定稿为 (b) 带（R57 已定稿 `[0.4800, 0.4900]`）。
- **率** = 事件数 / 时间 ⇒ 代理单元 ≈ **多神经元聚合**，其率随未知的"每箱神经元数"**线性缩放**
  ⇒ 实测代理中位率 **0.0032 Hz**，但**不可**填入 `rate_median_hz_measured_calcium_band_lo/hi`；
  若强行填，与 spike 带（0.5–2.0 Hz）会得出**纯口径造成的上千倍**假差距。
- ⇒ **同一份数据，一侧可定稿、一侧不可定稿，差别在量纲（定性 vs 绝对量）而非数据质量。**
  新增行：`rate_median_hz_measured_calcium_proxy_bin_measured`（记实测值 + 明确"不可用于填 rate 带"）与
  `rate_median_hz_measured_calcium_band_status`（记录 rate 侧保持 `unavailable` 的理由）。

## N5. 复算命令（逐项）

```bash
cd /Users/weidong/ai/small_world
```

| # | 复算什么 | 命令 |
|---|---|---|
| 1 | **全部**（解码自证 + 两口径 + 两单元口径 + 追加判据行） | `PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./.venv-m11/bin/python neural_exploration/tools/validate_m11_calcium_silent.py` |
| 2 | 同上但**不追加判据行**（只测量，安全重跑） | 加 `--no-targets` |
| 3 | 主口径与 J2 定稿值的一致性（`J2.8`） | 见 1 的日志行 `断言：{... "J2.8_pipeline_consistent_with_j2": true ...}`；JSON 键 `measurements.field4_primary.pipeline_check_vs_j2` |
| 4 | 解码自证细节（F vs C 负对照） | `.venv-m11/bin/python -c "import json;print(json.load(open('neural_exploration/data/m11_calcium_silent.json'))['decode_selfcheck'])"` |
| 5 | 判据表**旧行未被改动**的独立核验 | `.venv-m11/bin/python -c "import json;t=json.load(open('neural_exploration/data/m11_p5_targets.json'));print(len(t['rows']),sum(1 for r in t['rows'] if str(r.get('final'))=='0'),[r['value'] for r in t['rows'] if r['key']=='silent_frac_calcium_band_lo'])"` （应为 `114 29 ['unavailable']`） |
| 6 | 与 B3/B3b 的实现对照（同管线） | `PYTHONHASHSEED=0 OMP_NUM_THREADS=1 ./.venv-m11/bin/python neural_exploration/tools/build_m11_calcium_silence.py`（J2 原始工具，应在 `silent_frac_zero_activity` 上与 1 一致到 1e-6） |
| 7 | 四类参考量的钙域口径（含率分布/群体相关） | `PYTHONHASHSEED=0 OMP_NUM_THREADS=1 ./.venv-m11/bin/python neural_exploration/tools/build_m11_calcium_reference_quantities.py` |
| 8 | 判定器（数据门总闸） | `OMP_NUM_THREADS=1 ./.venv-m11/bin/python neural_exploration/tools/validate_p11_data_gate.py` |

**实测预算**：命令 1 = **17.9 s**（离线 CPU、单线程、无 GPU；派发预算 5–15 min ⇒ **未触托管阈值**）。

## N6. 本轮自查拦下的 4 个缺陷（全部为"假结果/假跳过"类，值得留档）

1. **头解析错**：把 `>5I` 整体当形状（`num_dims=4` 已占首位）⇒ 与 `(4,200,200,1,10)` 比较**永不相等** ⇒ 解析 0 片。
   ⇒ 修：形状元组 = `hdr[1:]`。
2. **硬编码 200×200** ⇒ **field2（100×100）被静默跳过**且**无日志**。
   ⇒ 修：泛化到任意 (X,Y) + **显式登记跳过原因**（`field2_skip_reason`）——**"不得静默跳过"**纪律。
3. **事件定义错**：误把"像素级越阈次数之和"当单元事件数 ⇒ 每箱必然 >0 ⇒ 主口径**假性为 0.0000**。
   ⇒ 修：阈值施加于**代理单元自身的时间轨迹**（与 J2 定稿实现同源）⇒ 复现 **0.4900**。
4. **负对照不判别**：误用 `(10,1,200,200)` C 序 ⇒ 与 F 序**同布局**（两种写法指向同一线性映射）⇒ 两序给出**同一 lat_x**。
   ⇒ 修：负对照必须用**同一形状 `(200,200,1,10)` 的 C 序**读 ⇒ lat_x **0.3787** vs F 序 **0.7769**（判别性成立）。
   ★ 并且发现：**帧轴轮廓**（像素级图像相关 0.7738→0.7395）**不能**区分对错序 ⇒ 判别只能靠 lat_x（已在工具注释中写明）。

## N7. 文件清单（本轮新增/改动）

**新增**
- `neural_exploration/tools/validate_m11_calcium_silent.py`（本任务工具，含解码自证 + 两口径 + append-only 追加逻辑）
- `neural_exploration/data/m11_calcium_silent.json`（产物：口径/自证/测量/四态/覆盖/断言/复算命令）
- `neural_exploration/docs/m11_b1_notes.md`（本文件）
- `neural_exploration/data/m11_raw/calcium_silent.log`（增量日志，末行 `ALL_DONE`）

**改动（append-only，旧行逐字保留）**
- `neural_exploration/data/m11_p5_targets.json`：101 → **114 行**（新增 **13** 行，均带 `b1_j2_append` 标记）；`final=0` 16 → **29**
