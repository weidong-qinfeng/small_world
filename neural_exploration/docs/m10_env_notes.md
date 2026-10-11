# M10 环境/踩坑笔记（斑马鱼幼鱼全脑——"记录一切"的第二金标准）

> 执行节点：M10-B1（**数据门 G2 + 引擎门 G0 + 前置机制门 G1/P6**）
> 日期：2026-10-11 起｜实施清单：`docs/生物仿真M10实施清单.md`（982 行）
> 承接：`docs/m9_report.md` §9.2（M10 交接 A–D）+ `docs/m9_acceptance_master.md` §六（6 条方法学纪律）+
> `docs/m9_env_notes.md` L1–L43
> 本文件格式沿 `m9_env_notes.md`：**L1–L8 = 清单 §0.10 预注册处置；L9+ = 本轮实测新结论**。
> 纪律：**不静默、不臆造、未测 ≠ 未命中**（M9 B9）；判据带运行前定稿（§0.3 #6）。

---

## L1–L8 — 前置确认与清单 §0.10 处置（预注册，逐条落位）

| # | 项 | 处置（本轮落地） |
|---|---|---|
| **L1** | M9 冻结基线 → M10 组装方式（组合不修改） | ✅ 已核：M9 冻结件**零修改**（`git status` 只新增 M10 段文件，见 L15）。M10 新建件：`tools/fetch_m10_imaging.py`、`tools/build_m10_connectome.py`、`tools/build_m10_imaging_reference.py`、`tools/build_m10_reference_bands.py`、`tools/validate_p10_data_gate.py`。`src/larval_zf_engine.py` 等引擎侧新建件在阶段 2/3 落盘 |
| **L2** | 数据源与许可（G2 门前置） | ✅ 实测可达性表落盘 `data/m10_source_reachability.csv`（40 源，见 L9）；许可：Fish1 = 开放（引用必需，CC-BY-NC-4.0 论文页）；ZAPBench = **CC-BY-4.0**；Z-Brain 镜像 = CC-BY-4.0 |
| **L3** | M9 反证记录最终状态（只读不依赖） | ✅ M9 P9 部分完成、`run_m9_validation.py`/`gen_m9_report.py`/`m9_imaging_reference.csv` 缺失 → 本节点**未依赖**任何 M9 未交付件 |
| **L4** | M9 冻结组件作 M10 引擎对齐参考 | ⏳ 阶段 2（G0）：M10 内核须在 M9 冻结 300 two_comp 网络上重过 (a)–(e) + 关闭态等价 |
| **L5** | 环境与依赖 | ✅ **新建 `.venv-m10`**（不触碰 `.venv-neuro`/`.venv-db`/`.venv-m9`）：python 3.9.6 + caveclient 8.2.1 + pandas 2.3.3 + pyarrow 21.0.0 + numpy 2.0.2 + h5py 3.14.0 + matplotlib 3.9.4。**PyPI 实测可达**（`pypi.org/simple` 200）。引擎侧仍用 `.venv-m9`（torch 2.8.0 + MPS） |
| **L6** | 预算与确定性纪律 | ✅ 判据带唯一定稿：`data/m10_imaging_reference.csv`、`m10_behavior_reference.csv`、`m10_calibration_targets.csv`、`m10_mechanism_ladder.csv`（生成器 `tools/build_m10_reference_bands.py`，**只允许新增行**）。`PYTHONHASHSEED=0` 固化 |
| **L7** | 数据隔离原则落地（§0.8） | ⚠️ **部分**：Fish1（结构）与 ZAPBench（活动）来自**不同动物**（第三方文档亦明确 "no cell-level correspondence"）→ M10 活动通道按**统计级**对照（不做逐神经元身份对应，§0.7 #4 一致）。拟合集 A / held-out B/C/D 的 session 级拆分表待 P5 前落定（ZAPBench 条件块自带 train/holdout 划分：`CONDITIONS_HOLDOUT=(3,)` 即 `turning`） |
| **L8** | M9 未完成件与 P9 教训交接 | ✅ 已继承：逐目标子进程 + 单池硬超时 + 未测登记 `unavailable` 不入分母 + 池级/锚级双口径；本轮 P1 的"全脑突触锚未测"即按 B9 登记 |

---

## L9 — 数据源可达性实测（P1/P2 前置；**逐源实测，不凭经验**）

完整表：`data/m10_source_reachability.csv`（40 源）。关键结论（本轮最影响后续设计的五条）：

1. **`storage.googleapis.com` 可达且快（2–4 MB/s）**：Fish1 官方发布桶 `fish1-release`（公开匿名可读、
   **可 list**：189 objects / 3.54 GB）与 ZAPBench 发布桶 `zapbench-release` 均可**无凭据直取** →
   **M10 数据门的主通道 = Google Cloud Storage 公开桶**（M9 的 Zenodo 通道在本机退居次席，见 L11）。
2. **CAVE（Fish1 全脑逐突触表的唯一官方通道）在本机不可用**：`https://global.brain-wire-test.org/`
   的 `/materialize/api/v2/...` 全部 **nginx 404**（含 datastack/version/table/query 多种路径变体）；
   `/info/api/v2/datastacks` → **302 → `/sticky_auth` 登录**（交互式 token，无头会话不可得）；
   `https://pcgv3local.brain-wire-test.org` 亦 404。⇒ **全脑 ~30M 突触表不可得**（见 L10）。
3. **文献侧**：`science.org` 403、`cell.com` TCP 超时、`journals.biologists.com` 403、
   `research.google` 超时（**与 M9 观测一致**）；但 **`export.arxiv.org`(200)、`www.ebi.ac.uk` Europe PMC
   REST(200)、`nature.com`(200)、`elifesciences.org`(200)** 可用 → 论文全文走 **arXiv + Europe PMC**。
   `biorxiv.org` 首次 TCP 超时、复测 **429（限流不是封锁）**。
4. **代码/镜像侧**：`api.github.com`、`codeload.github.com` 200；**`raw.githubusercontent.com` 连接被重置**
   → 取文件必须走 `api.github.com/repos/.../contents`（`Accept: application/vnd.github.raw`，实测可行）。
   `huggingface.co`、`zebrafishem.org`（DNS 无 A 记录）、`engertlab Z-Brain`（超时）、
   `figshare`(403)、`gin.g-node.org`(403) 均不可达。
5. **Z-Brain 配准图谱的替代可得路径**：`engertlab` 直连不可达，但 **Zenodo 7130703 / 3367709
   "Reference brains for registration to Zebrafish Brain Browser"（19 个 nii.gz ≈ 4.5 GB, CC-BY-4.0）**
   实测可达 → 区域级配准对照的材料存在（受 L11 吞吐限制，本轮未下载）。

---

## L10 — Fish1 连接组：**完整度分级 = `partial`**（P1 判定依据，证据链完整）

**可得（官方公开包，无凭据）** —— `paper_data/HMI_analysis.zip` + `paper_data/TEN_analysis.zip`：

| 内容 | 实测计数 |
|---|---|
| 神经元 roster（`lores_id ↔ hires_id_agglo`） | **187,052 行 → 180,782 唯一 agglomerated 段**（发布原文 ">180,000 segmented soma" ✓ 落带） |
| CAVE `somas` 导出（HMI 分析框内） | **30,346** 行；`cell_type` = exc 2,763 / inh 2,384 / na 25,199 |
| TEN/DMV 电路的逐神经元 partner 列表 | `incoming_synapses.csv` 24,286 行 + `outgoing_synapses.csv` 7,477 行 |
| 逐突触尺寸表 | `synapse_sizes.csv` 83,426 行（其中 **103 行 partner id 缺失**，如实登记） |

**解析产物（`tools/build_m10_connectome.py`，确定性 + 硬断言 + SHA）**：
`data/m10_zf_connectome.csv`（180,782 neuron 行 + 13,813 chem 行 + **0** gap 行）、
`data/m10_connectome_counts.json`、`data/m10_zf_network.npz`。
- 有向对 **13,813**、突触 **23,987**（限定两端都在 roster 内）；独立第三方派生产物
  （`jamieswrld/zebrafishconnectome` 的 `connectivity.json`，README 声明由同一公开包派生）给出
  **n_edges = 13,813** → **跨来源独立核对一致 ✓**（仅作交叉核对，不作权威来源）。
- 边节点 7,989，其中 **2,801（35.06%）** 有 cell_type（经 roster 反查实现 lore↔agglo ID 空间桥接，
  避免 M9 L14.2 的 ID 空间混用陷阱）；**pre 端已知类型的边 3,874 条，其中抑制性 263 条**。

**不可得（如实登记，不臆造）**：

| 项 | 状态 | 处置 |
|---|---|---|
| **全脑 ~30M 突触表**（CAVE `synapses_axde_label`） | ❌ 门控（L9 #2） | **计数锚"未测"**（不是"未命中"，M9 B9）→ 三态裁决请求 **RF-1** |
| 缝隙连接 | ❌ 官方包与 CAVE 表清单均无此表 | 记 **0 行 + 测量限制**（M9 R8 语义沿用） |
| 区域/脑区分配 | ❌ 官方未发布（第三方文档亦确认 `not exported`） | `region=''`；**由坐标推区域 = 臆造** → 不做（M8 B1a 纪律） |
| 递质亚型（GABA_A/GABA_B、DA/5HT） | ❌ 仅 vglut2a+/gad1b+ 二元遗传标注 | **M10 P6 的 M-3 级（GABA 亚型异质）只能走类级文献先验 + 测量限制** → 三态裁决请求 **RF-2** |
| 逐神经元区域/类型完整标注 | ❌ 覆盖率 2.85%（5,147/180,782） | 如实记录（M10 §1.1.2 允许 <100%） |

**⇒ 完整度分级（§1.1.3）= `partial`**：只承诺子图判据；P6–P11 以子集结构可达性为准（判据带不重新放宽——
机制门 P6 的自检门是**网络级动力学判据**，在 13,813 边子图上仍可判，但须在报告中标注"结构证据级别 = partial"）。

---

## L11 — ⚠️ 传输后端纪律（新踩坑，**影响所有后续数据获取**）

1. **Python 进程（`.venv-m10` 的 `urllib`）对 `storage.googleapis.com` 的 TLS 握手超时**
   （`_ssl.c:1112: The handshake operation timed out`，40 s），而 **`curl` 同 URL 正常（HTTP/2）**
   → **本机数据获取一律以 `curl` 子进程为传输后端**（`tools/fetch_m10_imaging.py::curl_bytes` 已按此实现）。
   注：Python `urllib` 对 **zenodo.org** 正常（JSON API 200），故该现象是**目标站点/路由相关**，不是全局 SSL 故障。
2. **吞吐实测（决定哪些数据集可做）**：

| 源 | 单流 | 备注 |
|---|---|---|
| GCS `fish1-release` | **≈ 4.0 MB/s** | range GET 20 MB / 5.0 s |
| GCS `zapbench-release` | **≈ 2.8–3.0 MB/s** | 6 线程并发拉 256 × 1 MB chunk / 94.7 s |
| **Zenodo 文件下载** | **≈ 56 KB/s** | 4 并发**无改善**（≈11 KB/s 聚合，更差）→ 35.6 MB 文件 1 h 未完成 |

   **直接后果（如实登记，不静默）**：Zenodo 上若干**高价值公开成像数据集本次未下载**：
   - **`WBSC.zip`(35.6 MB)（Zenodo 19066472，6 条鱼的全脑单细胞二值化脉冲序列 + 3D 位置）——
     已是 Zenodo 上最小的高价值档，仍 1 h 仅拉到 8.8/35.6 MB 且多次停滞**（`curl -r` 续传循环 + 断点
     续传均实测失败：一次续传把文件从 5.7 MB 缩回 1.8 MB）→ 已终止并删除残档（不静默）
   - `DFF_1Hz_GCaMP5.mat`(2.7 GB) / `DFF_20Hz_GCaMP3.mat`(1.5 GB)（Zenodo 1009488，斑马鱼全脑光片 ΔF/F 矩阵）
   - `Figure 5l,m.tar.gz`(2.8 GB)（Zenodo 19598932，**left/right/coherence 交错视觉运动刺激** → 本可修 P7 ④）
   - `fish1.5_functional_data.h5`(589 MB)、`clem_zfish1_functional_data.h5`(589 MB)（Boulanger-Weill CLEM 功能数据）
   - ZBB 参考脑（4.5 GB）
   ⇒ **未测 ≠ 不可得**（M9 B9）：这些条目逐条登记为"**受吞吐限制未测**"，请求裁决是否安排长窗口下载。
3. **`fetch_m10_imaging.py` 落盘路径**：一度因 cwd 与包根不一致把 258 MB 数组写到了仓库根的 `data/`
   → 已改为 `PROJECT_ROOT` 绝对路径解析（该次产物已迁移并核对字节数一致，无残留）。

---

## L12 — P2 全脑活动成像数据门：**证据级别 = `measured`（L1）**，ZAPBench

**数据集**：ZAPBench（Zebrafish Activity Prediction Benchmark，Google Research + HHMI Janelia + Harvard；
arXiv:2503.02618 / ICLR 2025；代码 `github.com/google-research/zapbench`；**CC-BY-4.0**）。
- 记录：斑马鱼幼鱼**全脑光片钙成像**，**71,721 神经元 × 7,879 时间步**，**时长 ≈ 2 小时**
  （论文 Fig.3 原文 "whole-brain calcium recording lasting two hours"）⇒ **体积率 ≈ 1.09 Hz**
  （**与 M10 §1.2.1 预注册 "~1–2 Hz" 一致**；这是"帧率读回断言"的实测值）。
- **9 个视觉刺激条件**（`constants.py::CONDITION_NAMES/OFFSETS`）→ P7 ④ 标签可得（但见下面 ④ 的归因）。
- 本轮拉取子集：**8,192 神经元 × 7,879 帧**（`data/m10_raw/zapbench_traces_8192.f32.npy`，258 MB，NaN=0，
  读回断言 shape/chunk/dtype 全过，provenance 落 `data/m10_raw/zapbench_provenance.json`）。

**四类参考量实测（`tools/build_m10_imaging_reference.py` → `data/m10_imaging_reference.csv` + `m10_imaging_stats.json`）**：

| # | 指标 | 实测 | 说明 |
|---|---|---|---|
| ① | 发放率分布 | **3·MAD 事件率**：中位 **0.0079 Hz**、p25 0.0053 / p75 0.0110、静默神经元比例 0.12% ；**M10 §6.1 反演口径**（A=0.15, τ_ca=1 s）：中位 **0.0719 Hz** | **两个口径并列**（阈值事件率 vs 积分反演）；**均显著低于 M10 §4.2 预注册文献带 0.3–1.5 Hz** → 口径差必须显式登记（M9 B2），P5 (a) 主判据带仍取清单预注册文献带，成像分位作为**并列诊断列** |
| ② | 群体相关性 | 平均成对相关 **0.1273**（1,200 神经元抽样）；**同管线相位随机化代理零分布 = 0.00013**；特征值谱已落盘；**有效维度 28.8** | 实测相关 = 代理地板的 **~950×** → 群体相关为真实信号（非采样噪声）。**对照 M9 模型实测 0.6220（代理 −0.07）**：M9 的"自持 + 共享慢动力学"造成的过高相关在此获得真实参考上界（0.127）→ **H6（调质/状态切换变量是活动统计的必需条件）有了可判定带** |
| ③ | 状态转换 | 2 状态高斯 HMM（EM）：占用 **{低 0.6719, 高 0.3281}**；**转换率 0.0196 Hz**（≈51 s/次，2 h 内 141 次切换）；停留时长：低活动态中位 **26.5 s** / p90 196 s / max 455 s | **M9 的 bout 占比 = 0.0000 → 该指标在 M9 不可算**；M10 首次获得真实双态参考（P7 ③ 判据带由此可得） |
| ④ | 刺激解码 | **参考带未建立**：岭回归（one-hot）+ 条件内连续 5 折 CV，准确率 **0.8551**（帧中心化 0.8602），但**块级循环移位零分布上界 0.9238（帧中心化 0.9193）> 数据值** | **归因（可分辨，不猜测）**：ZAPBench 的 9 个条件在时间上**连续成块**（`CONDITION_OFFSETS` 单调递增）→ **条件身份与全局慢漂移完全混淆**；"保留块结构"的零分布因此高于数据。⇒ **P7 ④ 判 `unavailable`（不入分母，M9 B9 语义）+ 三态裁决请求 RF-3**，并给出修复路径（交错刺激档/行为状态标签） |

**两个否定结果已落盘（"否定结果必须落盘"纪律）**：
1. **"测试折内标签置换"的帧级零分布实测为退化解**：测试折完全落在单一条件块内 → 置换是恒等操作，
   零分布 ≡ 数据值 → 该零分布无效，已替换为"块级循环移位 + 帧中心化并列口径"。
2. **第一版零分布遗漏了帧中心化口径**（原始口径零分布 ≥ 数据）→ 补齐后**依然不显著** ⇒ 不是简单的
   全局均值漂移可解释，而是条件块设计与慢漂移的结构性混淆。

---

## L13 — 文献带 provenance 化的网络限制（登记，不静默）

M10 §1.3 要求行为带"以文献整理定稿"：本机实测 **science.org 403 / cell.com 超时 /
journals.biologists.com 403 / biorxiv 429（限流）**，可用的只有 arXiv / Europe PMC / nature / eLife。
⇒ `data/m10_behavior_reference.csv` 内每行带 `provenance_class` 列：
`pre_registered_draft`（清单 §1.3 预注册草案，本次运行前定稿）/ `measured_derived`（ZAPBench 实测）/
`literature_anchor`（**DOI 级 provenance 化未完成**——仅逃逸潜伏期 4–10 ms 行，已标注）。
**处置**：P4/P9–P11 执行时按 **PMC 可得源**逐条补齐 DOI；在此之前若某判据依赖未 provenance 化的带
→ 判 `degraded`（不得当作 PASS）。**未测 ≠ 未命中**。

---

## L14 — 本轮新增工具与产物索引

| 类别 | 路径 | 说明 |
|---|---|---|
| 工具 | `tools/fetch_m10_imaging.py` | ZAPBench 子集拉取（curl 后端 + 读回断言 + NaN 完整性断言 + provenance） |
| 工具 | `tools/build_m10_connectome.py` | Fish1 公开包确定性解析（硬断言 + 计数诚实性 + SHA + ID 空间桥接 + 裁决请求内嵌） |
| 工具 | `tools/build_m10_imaging_reference.py` | 四类成像参考量 + 判据带定稿（`m10_imaging_reference.csv`） |
| 工具 | `tools/build_m10_reference_bands.py` | 行为/标定/机制阶梯判据带定稿（只允许新增行） |
| 工具 | `tools/validate_p10_data_gate.py` | **独立复核**（流式重算 CSV vs JSON）+ 确定性重跑 SHA + 可达性表 + P1/P2 判定 |
| 数据 | `data/m10_zf_connectome.csv` / `m10_connectome_counts.json` / `m10_zf_network.npz` | 连接组解析产物（P1） |
| 数据 | `data/m10_imaging_reference.csv` / `m10_imaging_stats.json` / `m10_imaging_rate_dist.csv` | 成像参考量（P2） |
| 数据 | `data/m10_behavior_reference.csv` / `m10_calibration_targets.csv` / `m10_mechanism_ladder.csv` | 判据带定稿 |
| 数据 | `data/m10_source_reachability.csv` | 40 源可达性实测表 |
| 报告 | `reports/neuro/m10_data_gate_summary.json` / `m10_imaging_reference.png` | 数据门判定 + 成像参考图 |
| 原始 | `data/m10_raw/`（fish1_release / zapbench_traces_8192.f32.npy / zapbench_provenance.json） | 原始与 provenance |

---

## L15 — 冻结件零修改核查

- M0–M9 冻结件（`src/adult_*`、`src/synapse_model.py`、`tests/`、`tools/build_m9_*`、
  `tools/validate_p9_*`、`tools/diag_*`、`data/m9_*`）**本轮零修改**；本节点只**新增** `tools/*_m10_*.py`
  与 `data/m10_*`、`docs/m10_env_notes.md`、`reports/neuro/m10_*`。
- 环境隔离：新建 `.venv-m10`；`.venv-m9`（torch/MPS 引擎环境）**未装任何新包**（阶段 2/3 引擎运行复用）。
- ⚠️ 唯一一次路径事故：`fetch_m10_imaging.py` 首次以相对路径把数组写到**仓库根** `data/m10_raw/`（已迁回
  `neural_exploration/data/m10_raw/` 并核对字节数一致；仓库根 `data/` 已清理，见 L11 #3）。
- **`git status` 中 4 个 M9 段文件显示为修改（`tools/validate_p9_activity.py`、
  `tools/validate_p9_perturbation.py`、`reports/neuro/m9_three_channel.png`、
  `m9_engine_alignment_vec.png`）→ 经 mtime 核对（07:22–07:42）确认**均早于本节点会话开始（≈08:00）**，
  属 M9-B3 节点的遗留改动，**非本节点所为**（如实登记，便于主agent追溯）。

---

## L16 — 待裁决 / 风险（交主agent）

| # | 请求 | 影响 | 备选回退 |
|---|---|---|---|
| **RF-1** | 是否提供 **CAVE token**（浏览器交互取 token）以取得 Fish1 全脑 ~30M 突触表？ | 取得 → P1 可自 `partial` 升 `full`（全脑逐突触 + 逐突触兴奋/抑制 tag）；不取得 → 维持 partial（判据带按子图定稿） | 维持 `partial` + 报告标注结构证据级别 |
| **RF-2** | P6 的 **M-3（GABA 亚型异质）**在无亚型标注下是否接受"**类级文献先验 + 测量限制登记**"实现？ | 接受 → M-3 可按 τ_A∈[5,20] ms / τ_B∈[80,400] ms 类级扫描；不接受 → 机制阶梯缺一级（阶梯判决表照实登记 `not_started`） | §1.5"递质/调质标注缺"行 |
| **RF-3** | P7 ④ 刺激解码参考在 ZAPBench 上**不可建立**（条件块 ↔ 慢漂移混淆）→ 是否接受判 `unavailable` 并改用交错刺激档（需长窗口 Zenodo 下载 ~2.8 GB @56 KB/s ≈ 14 h）？ | 接受 → ④ 不入分母，P7 按 3 指标判（① 与 ② 必含）；不接受 → 需安排长窗口下载或换数据集 | 用行为状态标签替代刺激标签（P13 口径） |
| **RF-4** | Zenodo 吞吐 56 KB/s → 是否安排后台长窗口下载（ZBB 参考脑 4.5 GB / DFF 矩阵 4.2 GB / 交错刺激档 2.8 GB）？ | 影响区域级配准对照（§1.2.1 第四类）与 ④ 的修复 | 登记"受吞吐限制未测" |

**风险登记**：
1. **结构证据级别 = partial** → P6–P11 的环路级判据（Mauthner/H4、视网膜-顶盖/H5）在子图上可能不可达
   → 届时按 §8.1 逐级排查并如实登记（不得把抽象当连接组）。
2. **本机共享高负载**（load 2.7–6）+ 两类 GPU 退化（M9 L34.1/L40）→ 阶段 2/3 一律逐目标新子进程 +
   单池硬超时 + 段间落盘（**禁止并行同一任务**：开跑前 `ps aux | grep <task>`）。
3. **成像 ↔ 结构无共同坐标系**（不同动物）→ 活动通道只做**统计级**对照（§0.7 #4）。

---

*L9–L16 为本轮数据门实测；阶段 2（引擎门 G0）与阶段 3（P5/P6 前置机制门）实测见 L17+。*

---

## L17 — 🔴 **M9 冻结内核的真实 bug：CSR 投递索引起点公式错误**（M10 引擎门首个发现）

**位置**：`src/adult_engine.py::_deliver_step` / `_deliver_chunk`（冻结件，两处同一公式）：
```python
off  = torch.repeat_interleave(grp["t_row"][idx] + torch.cumsum(c, 0) - c, c)
base = self._arange(tot)[:tot] + off
```
**最小复现（3 神经元 / 4 条边，单延迟组）**：`pre=[0,0,1,2] post=[1,2,0,0] delay=1`；
CSR `t_row=[0,2,3,4]`、`t_counts=[2,1,1]`。**同一步里 0 号与 2 号同时发放**时：

| 公式 | 交付的边索引 `base` | 期望 |
|---|---|---|
| M9 冻结公式 | **[0, 1, 7]**（组内只有 4 条边 → 越界 3） | [0, 1, 3] |
| M10 修正公式 `t_row[idx] − (cumsum(c)−c)` | **[0, 1, 3]** ✓ | [0, 1, 3] |

**根因**：`t_row` 是标准 CSR **起点**数组（`row[i] = Σ_{j<i} counts[j]`），而公式把
"之前所有发放神经元的边数"`prev_total` 又**加到**块起点上 → 第 j 个发放神经元的边块被平移 `prev_total`
⇒ ① **同一步 ≥2 个神经元发放**时越界（CPU 抛 `IndexError`）、② MPS 上越界不报错而是**静默错投/读脏**。

**决定性判别实验（`tools/scan_m10_engine.py` T2）**：构造"同一步两个源神经元持续发放"的 5 神经元网络 →
- **M9 冻结引擎：CPU 抛 `IndexError: index 5039 is out of bounds for dimension 0 with size 5039`**；
- **M10 引擎（E5-FIX）：正常运行**。
→ 这正是 M9 **RB1**（"全规模 CPU point 档路径不可用"）的**真实根因**——不是 CPU 后端问题，
而是**稀疏发放下的通用越界 bug**；M9 自己把它登记为"CPU 路径不可用"，实际是**引擎缺陷**。

**影响评估（如实、不夸大）**：
1. M9 的全规模结论**全部由 MPS 档给出**；MPS 对越界索引不抛错 ⇒ **M9 的投递在"同一步 ≥2 个神经元发放"
   的情形下不可信**（M9 定稿网络 pop 4.77 Hz / 静息 81.67% 静默 ⇒ **每步必然多神经元发放**）。
2. 因此 M9 的 **R3「活动自主自持 / 输入增益过低」诊断需要用 M10 修正内核复测**（本轮已列为 P5 硬判据）。
3. M9 的对齐门（(a) 300 two_comp，636 spike）在**极稀疏**（多数步 ≤1 spike）下不受影响，
   故 100% 对齐结论**本身不被推翻**，但**不能**外推到密集发放工况。
4. **纪律沉淀**：**判据级对齐必须在"目标工况的发放密度"下复测**（"稀疏工况通过"≠"密集工况正确"）；
   M10 引擎门因此新增 **T1（链式解析对照 + M9/M10 一致）** 与 **T2（同步发放判别）** 两项。

**处置**：M10 新建 `src/larval_zf_engine.py` **继承**冻结内核并**覆盖**投递路径（冻结件零修改），
修正公式 + 绝对步号槽（E4）。**该修正使 M10 内核与 M9 内核在稀疏工况下逐位一致（T1 实测 = true）**，
在密集工况下 M9 报错而 M10 正常（T2）。

---

## L18 — ⚙️ 冻结内核的两条性能快路径约束（决定探针与实验的配置选择）

实测（M10 100,000 点神经元探针，本机 M1 Pro / MPS）：
1. **多变延迟 → 慢 >30×**：`run()` 只有 `len_rings == 1`（**单一延迟值**）时走"预计算槽视图"快路径；
   多延迟组走逐槽 Python 拼接（每步每组建视图+清零）。首版探针用 `delay∈{1..5}`（5 组）→
   **20,000 步 >10 分钟未完成**（>30 ms/step，与 M9 锚 0.42–1.02 ms/step 差 1–2 个数量级）；
2. **`W = min(SLOT_EXTRA=40, max_delay)`**：`delay=1` 时 `W=1` ⇒ **chunk 批量投递退化为逐步**（又慢 ~50×）；
3. ⇒ **统一延迟 = 40 步（=2 ms，生理合理）** 同时满足"单延迟组 + W=40 chunk"，为 M10 全规模探针与
   P5/P6 实验的**配置前提**（已写入 `ZFNetParams.delay_max=40` 与探针默认值）。
4. 首版探针网络（100k 神经元 / 60 边每神经元 / bias 260 mV/s）实测 **pop 233 Hz/神经元**（癫痫态）
   → 墙钟 2.07 ms/step 且仅 1 段即 958 s ⇒ **探针网络必须标定到生理活动区**（M10 改为 20 边/神经元、
   亚阈 bias 230 mV/s、λ=2 Hz），否则"墙钟"混入"发放密度"因素而不可与 M9 锚比较。

---

## L19 — 阶段 2/3 实测结论（引擎门 + P5/G1 + P6/机制阶梯）

### L19.1 引擎门（P3）
| 项 | 结果 |
|---|---|
| **T1 链式解析对照**（每步 ≤1 发放） | **M9 与 M10 引擎逐位一致 ✓**（链式传播单调、39 spike）⇒ 非投递语义（核/衰减/不应期/重置）全部保持 |
| **T2 同步发放判别**（同一步 ≥2 源神经元） | **M9 冻结引擎 CPU 抛 `IndexError`；M10（E5-FIX）正常运行** ⇒ L17 的 bug 与修法均被决定性判别 |
| **T3 分段可组合性 (e)** | **k=1/2/5 逐位相同（0.0%，total 与逐神经元计数全等）**——`delivery="step"` 与 `"chunk"` 两口径皆然；**对照臂 `use_abs_slots=False`：k=2 −0.091%（222 神经元不同）/ k=5 −0.039%（467 神经元不同）**（≈M9 量级）⇒ **修法有效且可证伪**；M9 冻结引擎本臂在 CPU 直接报错 |
| **T4 全规模 10 万点墙钟**（3 独立新进程、各单段 20,000 步、N=100,000 / 2M 边 / W=40） | **1.4285 / 1.4258 ms/step**（≈1.43）→ **30 s 单试次 ≈ 0.238 GPU-h**（预算上限 0.5 ✓）；**三进程脉冲总数逐位相同 14,771,593**（确定性 ✓） |
| 注意（不夸大） | 该探针活动水平高（**147.7 Hz/神经元**）⇒ ms/step 偏保守；**同进程跨段退化仍复现**（seg0 1.53 → seg1 1.89 ms/step；墙钟 30.7 s → **962.8 s**，与 M9 L40 同型）⇒ **逐目标准新子进程纪律必须坚持** |

### L19.2 P5 静息标定（G1 前置门）——**G1 = 未通过**（(c) 过 / (d) 未过）

标定（`tools/calibrate_m10_weights.py` 扫描 37 点，`data/m10_calibration_scan.csv`）发现**二分支行为**：

| bias 中位 | 零驱动群率 | 说明 |
|---|---|---|
| **250 mV/s**（M9 量级） | **143–253 Hz** | tonic 偏置本身超过阈值 ⇒ **自持癫痫态**（复现 M9 R3 的"活动自主自持"，比值 ≈1.0） |
| **150 mV/s**（亚阈） | **0.000 Hz** | **自持被彻底修复**（比值 3e-4） |

**P5 判定（`data/m10_p5_resting.csv` / `m10_p5_resting.json`；N=500、T=1 s、λ0=32 Hz/神经元、bias=150）**：

| 子判据 | 实测 | 带 | 判定 | M9 对照 |
|---|---|---|---|---|
| (c) **自持比** | **0.0000** | ≤0.10 | ✅ | M9 **0.9853** → **修复** |
| (d) **驱动弹性（4×）** | **+20.56%**（单调 ✓） | ≥30% | ❌ | M9 **+7.25%** → 改善 2.8×，**仍未达门槛** |
| (a) 中位发放率 | **109 Hz** | [0.3,1.5] | ❌ | M9 4.77 Hz（均值） |
| (b) 静默比例 | **0.000** | [0.5,0.95] | ❌ | M9 0.524 |
| (g) 确定性 | 逐位一致（54,093 spike） | — | ✅ | ✅ |

**机制性诊断（可分辨，不猜测）**：标定后网络呈**全或无（bistable）**：λ=0 → 0 Hz；**λ=16 → 96.8 Hz**；
λ=256 → 147.7 Hz。即**没有任何中间工作点** —— 纯递归兴奋网络没有慢负反馈，驱动只把网络从"静默"推到"饱和"。
⇒ **G1(d) 的失败不是标定问题，而是缺失状态切换变量的直接体现**（与 M9 R1/R3 结论同根），
按 §8.1 走"G1 失败 → 排查/升级机制"路径，并在下面用 P6 阶梯直接检验"哪条机制能给出分级响应"。

### L19.3 P6 机制阶梯（前置机制门）——**判别性结论已取得；门 3 未过**

配置：N=1,000、indegree=20、bias 亚阈 150 mV/s、λ0=32 Hz、T=2 s（另跑 T=12 s 复测）。
判据带：自持比 ≤0.10、4× 弹性 ≥0.30、bout 占比 >0（`data/m10_calibration_targets.csv`）。

| 级 | 机制 | 自持比 | 4× 弹性 | bout（修正判据） | 结论 |
|---|---|---|---|---|---|
| **M0** | 基线（无机制） | 0.0002 ✅ | **+3.70%** ❌ | 0.0000 ❌ | `partial`（= M9 失败模式） |
| **M-1** | **STP 抑制型**（u0=0.5, τ_rec=1 s） | **0.0105** ✅ | **+752.1%** ✅ | 0.0000 ❌ | **`effective`（门 1/2 全过；λ×1 → 2.84 Hz、静默 0.388 落带）** |
| **M-2** | **STP 易化型**（u0=0.15, τ_fac=0.5 s） | **0.0066** ✅ | **+814.5%** ✅ | 0.0000 ❌ | **`effective`（λ×1 → 4.53 Hz、静默 0.215）** |
| **M-3** | GABA 亚型异质（A 8 ms / B 200 ms, 慢组分 35%） | 0.0001 ✅ | **+2.70%** ❌ | 0.0000 ❌ | `partial`（反而更饱和：λ×1 → 366 Hz） |
| **M-4** | 调质增益 ×1.5 | 0.0002 ✅ | **+4.25%** ❌ | 0.0000 ❌ | `partial`（λ×1 → 185 Hz） |
| **M-5** | 组合 | — | — | — | **`not_run`（预算/时间；如实登记，不写成未命中）** |

**结论（M10 第一个核心科学产出）**：
1. **STP（M-1/M-2）是"哪一级奏效"的答案**：它同时把 (i) 自持压住、(ii) 驱动弹性从 +3.7% 提到
   **+752%/+815%**、(iii) 工作点从"饱和 112 Hz"拉回 **2.8–4.5 Hz（静默 0.22–0.39 落带）**
   —— 即 **M9 R3 的"输入通路增益过低"由突触短期动态直接修复**（不是靠放大驱动）。
2. **M-3（GABA 亚型异质）与 M-4（调质增益）在本实现下无效**：二者都只在"全或无"曲线上移动工作点
   （λ×1 → 366 Hz / 185 Hz），**不产生分级响应** —— 与 M9 R2"仅调背景率/增益无效"的结论一致。
   （M-3 的类级拆分实现使快抑制只承载 65% 电荷 ⇒ 更易饱和；**该实现限制已登记**，待有亚型标注后重评。）
3. **门 3（bout 非零）在 T=2 s 与 T=12 s 均未过**（全部等级 bout_fraction = 0.0000）——按 §8.1
   的"检查时间尺度"项，**协议窗明显不足**：成像参考（ZAPBench）测得双态 dwell 中位 **26.5 s**
   （`m10_imaging_reference.csv` ③），远长于本次 2–12 s 窗。⇒ **P6 判 `not passed（门 3 未过）`**，
   下一步 = ① 协议 T ≥ 60 s（≥2× 成像 dwell）② 用与成像参考**同口径**的 2 状态 HMM 状态分析
   （而非 100 ms 分箱 bout）③ M-5 组合级。

### L19.4 判据可证伪性修正（本轮自查发现，**两版结果均留档**）
首版 bout 判据把"记录起止"也算分隔 ⇒ **完全无静默的饱和网络被算成 1 个贯穿全程的 bout**
（M-4 实测 bout_fraction = 1.0）⇒ **该判据无法证伪**（M9 B1 纪律）。修正为**必须两侧都有静默箱的
interior bout** 后，饱和网络给出 0.0000（可证伪 ✓）。**两版产物均保留**：
`data/_m10_p6_ladder_v1_original_criterion.csv`（旧判据）与 `m10_p6_mechanism_ladder.csv`（修正判据）。
**纪律沉淀**：判据必须先用"应当给出 0 的负对照"自检（此处 = 恒定高活动网络），
否则"全过"只是判据失效的假象。
