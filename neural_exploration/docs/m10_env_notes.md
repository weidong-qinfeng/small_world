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

*L9–L16 为本轮实测；阶段 2（引擎门 G0）与阶段 3（P5/P6 前置机制门）的实测结论将追加为 L17+。*
