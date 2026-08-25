# M5 环境与前置处置记录（清单 §1：L1–L6 + 执行节点实测 L7+）

> 对应《生物仿真M5实施清单》§1（P8 交付物）。
> 执行节点：B1a（连接组数据管线：解析/校验/交叉核对/子图/行为对照带）。
> 状态：L1–L6 处置完成；L7+ 为实测新坑与裁决建议（供规划节点复核）。

---

## L1 — 交接：M4 冻结基线 → M5 全连接组组装方式（组合不修改）

- **组合不修改纪律（冻结清单）**：M0–M4 全部冻结文件（src/reflex_arc.py、src/muscle.py、
  src/neuron_model.py、src/synapse_model.py、src/chemotaxis_*.py、src/neuron_pair.py、
  src/ion_channels.py、src/morphology.py、tests/、tools/validate_*、data/m0-m4 定稿 CSV）
  一律不改。本节点**只新建**：`tools/build_m5_connectome.py` + `data/m5_connectome.csv` +
  `data/m5_behavior_reference.csv` + `data/m5_*.csv`（子图/交叉核对/计数报告）+ `data/m5_raw/`（原始数据归档）。
- **本节点交付的接口语义**（供 B1b `src/connectome.py` 的 `load_connectome` 消费）：
  - 神经元行：role=<302 神经元名>、neuron_class=<sensory|inter|motor|pharyngeal>、
    neurotransmitter=<ach|glut|gaba|dopamine|serotonin|other>（100% 覆盖）、muscle_target=<4 通道>（运动神经元）；
  - 化学突触行：synapse_from/to + synapse_type=chem + neurotransmitter（突触前递质）+ receptor +
    g_max_ns（初始占位）+ delay_ms；化学突触 = 神经元间**有向连接对**（含自连接，白名单保留）；
  - 缝隙连接行：synapse_from/to + synapse_type=gap + g_gap_ns（初始占位 0.5nS，M2 量级）+ delay_ms；
    缝隙 = 神经元间**无向唯一对**（多重连接合并，自连接白名单保留）；
  - 肌肉行：role=muscle_drive，synapse_from=<运动神经元>，synapse_to=<body_fwd|body_back|head_left|head_right>，
    synapse_type=muscle，g_max_ns=收缩增量 w（M3/M4 定稿值：fwd 0.18/back 0.60/head 0.50）；
  - 列序：role, neuron_class, neurotransmitter, receptor, synapse_from, synapse_to, synapse_type,
    g_max_ns, delay_ms, g_gap_ns, muscle_target, note（M3/M4 CSV 惯例扩展）。
- **子图接口**：`data/m5_pharynx_subgraph.csv`（P3）、`data/m5_command_subgraph.csv`（P5/P6）、
  `data/m5_chemotaxis_subgraph.csv`（P4）——同一 schema 子集（role/neuron_class/neurotransmitter/
  synapse_from/synapse_to/synapse_type/g_max_ns/delay_ms/note）。
- **复用注意（点神经元适配，B1b 二选一定稿点）**：化学突触行含自连接（真实存在 34 条化学自连接），
  B1b 组装时点神经元侧自突触需处理（on_pre 同一神经元事件）；不做静默删除（连接组是事实）。

## L2 — 参考解方案

- **神经级（NEURON 局部子图）**：M3 反射链（`data/m3_reflex_ref.npz`）、M4 核心子链（`data/m4_ref.npz`）
  直接复用（P5/P4 参考）；**新增咽部子图参考**（~20 神经元）由后续节点按 `tools/build_reflex_ref.py`
  模式构建（ExpSyn+NetCon+cvode 1e-8+celsius=6.3）——本节点已交付 P3 的结构基础
  `data/m5_pharynx_subgraph.csv`（20 神经元 + 128 化学 + 33 缝隙内部连接）。
- **行为级（纯 numpy，引擎无关）**：趋化参考（M4 pirouette 复用）；逃避参考 + 自发行为参考
  （bout 马尔可夫）参数锚定 `data/m5_behavior_reference.csv`（本节点交付，含 provenance）。
- 全 302 神经元 NEURON 复制不做（参考成本控制，M4 L2 哲学）。

## L3 — M4 教训与预算纪律

- M4 L23/L25 全部采纳（降阶模型为行为层唯一路径；后台长任务完成标记+看门狗；验证前并发清空）。
- **本节点额外预算纪律**：连接组管线为纯离线 CSV 构建（秒级），无长任务；联网下载仅一次
  （`data/m5_raw/` 归档 + 本文件记录来源），重跑不依赖网络（确定性铁律）。

## L4 — 单位/方程/统计约定

- 沿用 M1–M4：电导密度 S/m²、点电导 nS、刺激密度 µA/cm²、ms/mV、Im 内向正。
- **M5 连接组新增约定**：
  - 连接计数语义：化学突触=神经元间有向对（Weight=突触计数，多重）；缝隙连接=无向唯一对；
  - 递质→受体映射（清单 §2.3）：ach→ampa（E=0mV）、glut→ampa（M4 惯例；nmda 慢成分可选）、
    gaba→gaba（E=−70mV）、dopamine/serotonin→mod（M5 调质占位）、other→none；
  - g_max_ns 初始占位：ach/glut=5.0nS、gaba=15.0nS（M3/M4 链级定稿值）、gap=0.5nS（M2 值）、
    调质/other=0.0（无直接快 EPSP）——§6 类级缩放校准的起点；
  - delay_ms：化学 0.5ms（M3/M4）、缝隙 0.05ms（近瞬时占位）、肌肉 0.1ms（M3）；
  - 四类分类：以 Cook 2019 node_type 为权威（SENSORY NEURONS/INTERNEURONS/MOTOR NEURONS/PHARYNX），
    规范 override：AVM/DVA→inter（White 1986/WormAtlas；M3 语义）、CANL/CANR→inter（Cook 图外）。
- 行为/电生理对照带：`data/m5_behavior_reference.csv` 为唯一定稿源（P2/P3/P4/P5/P6 判定脚本对照）；
  带与容差全部预注册（§0 #3/#4/#5），不做事后调。

## L5 — M4 遗留/简化假设补齐登记

本节点补齐/登记（M5 行为层前置，取舍逐条记录）：
1. **机械转导延迟**（触→感觉电流延迟）：转导延迟 τ_trans 由 P5 协议节点定稿于
   `data/m5_worm_params.csv`（行为潜伏期 30-50ms − 神经潜伏期 5-20ms ≈ 转导+肌肉 10-30ms）；
   连接组侧无对应参数（纯协议参数）。
2. **肌肉/身体动力学**：真实连接组 95 块体壁肌肉（dBWM/vBWM/pm/vm/um）**聚合映射**到虚拟身体
   4 通道（body_fwd/body_back/head_left/head_right）——68 条 muscle_drive 行（每运动神经元一行，
   其真实化学支配体壁肌肉已逐条核对存在）；A 型（DA/VA/AS）→back、B 型（DB/VB）→fwd、
   头运动神经元（SMD/SMB/RMD/RIV/RMH）→head_L/R（M4 SMDD→C_left/right 语义沿用）；
   **排除**：DD/VD（GABA 能抑制，不适配收缩增量模型，M6 抑制/调质层补齐）、VC/HSN（性特异）、
   RME（GABA 头调节，M4 同款默认关闭）——登记为抽象取舍。
3. **正弦爬行姿态**：body 层（virtual_body.py）协议参数，连接组侧不涉及。
4. **感受器转导**（ASE 电流注入功能模型）：M4 P1 同款简化登记（M5 不伪造波形）。
5. **递质简化**：多巴胺/血清素/酪胺/章鱼胺 → M5 为占位（receptor=mod/none，g=0），
   M6 引入完整调质动力学（清单 §2.3 + §11 交接）。

## L6 — 数据定稿与复现入口

- 唯一定稿源：`data/m5_connectome.csv`（连接组）+ `data/m5_behavior_reference.csv`（行为/电生理带）；
- 复现入口：`.venv-neuro/bin/python -m neural_exploration.tools.build_m5_connectome`
  （从仓库根 /Users/weidong/ai/small_world 执行；读取 data/m5_raw/，输出逐位一致，见 L15）；
- 交叉核对清单：`data/m5_crosscheck_m3m4.csv`；计数报告：`data/m5_connectome_counts.json`；
- 原始数据归档：`data/m5_raw/`（含 SHA-256 记录见脚本注释与 L10）。

---

## 执行节点实测结论（L7+，供规划节点复核 / 三态裁决）

## L7 — ⚠ 预注册区间 [6300,7700]/[630,770] 与全部权威计数不符（**请求规划节点裁决**）

- **预注册目标**（清单 §2.2）：化学突触 ∈ [6300,7700]、缝隙 ∈ [630,770]（围绕民俗 "~7000/~700"）。
- **实测权威计数**（全部如实解析，未改任何数据）：
  | 来源 | 化学 | 缝隙 | 计数语义 |
  |---|---|---|---|
  | **Cook 2019（本管线主源，c302 edgelist，神经元-神经元）** | **20,589 突触 / 3,638 有向对** | **8,642 突触 / 1,093 唯一对** | 权重=突触计数 |
  | Cook 2019（官方 SI5 矩阵，含肌肉等全细胞） | 28,113 / 4,879 有向边 | 10,923 / 1,447 无向边 | 论文发布 4,887/1,447 边 |
  | White 1986（c302 数字化，神经元-神经元） | 7,914 / 2,380 对 | 971 / 575 对 | 同上 |
  | Varshney 2011（发表值） | 6,394 | 890 | 279 神经元（非咽部子集） |
- **结论**：预注册区间与**任何**权威计数的**任何**计数语义均不完全吻合（6,394∈化学区间但缝隙 890 出界；
  White 化学 7,914 出界 214、缝隙 971 出界 201；Cook 权重和/边数均出界）。**区间本身不可由诚实解析满足**。
- **本节点处置**（遵守 §0 #5 铁律：不得为过 Pass 改权威数据；超容差 → 排查 + 如实记录，不静默）：
  1. P1 计数断言实现为**数据完整性断言**（解析值 == 权威源自洽值，全部 PASS）；
  2. 预注册区间合规实现为**诊断项**（counts.json + 脚本输出标记 OUT，如实入档）；
  3. **请求规划节点三态裁决**：建议将区间语义改为「vs Cook 2019 发布图统计」（化学 4,887±10% 有向边
     或权重和 20,589±10%；缝隙 1,447±10% 边或 8,642±10%），或指定 Varshney/White 为计数权威
     （需接受各自缝隙计数出界/神经元数 279 的差异）。本管线已提供全部数字，改区间**不涉及改数据**。

## L8 — M3/M4 子图交叉核对：43 OK / 10 DIFF（差异清单 data/m5_crosscheck_m3m4.csv）

- **角色存在性全过**：M4 20 角色 + M3 4 角色全部在 302 roster 中（VB/DB/DA/PLM 为类级角色，
  成员 VB1-11/DB1-7/DA1-9/PLML-R 命中）。
- **M4 18 化学核对**：8 条存在且极性一致（ASEL→AIYL/AIYR、ASER→AIBL/AIBR、RIAL→SMDDL、
  RIAR→SMDDR、AIYL→RIAL、AIYR→RIAR）；**6 条 MISSING**（AIYL→AVBL、AIYR→AVBR、AIYL→RIAR、
  AIYR→RIAL、AIBL→RIAL、AIBR→RIAR——真实连接组无这些直接化学边，M4 为功能链简化：
  真实 AIY→RIA 仅同侧、AIB→RIA 经 AVB/AVA 命令中间神经元）；**2 条 TYPE_DIFF**（AIYL→RIAL、
  AIYR→RIAR 存在但**递质=ach**（兴奋）而非 M4 建模的 GABA（抑制）——见 L9）；
  **4 条类型差异**（RIAL→SMDVL/RIAR→SMDVR/AVBL→VB1/AVBR→DB1 在真实连接组为**缝隙连接**而非化学）。
- **M3 3 化学核对**：PLM→AVM **MISSING**（真实 PLM 无化学输出，触觉经**缝隙耦合**：PLM↔PVC/LUA 等）；
  AVM→DA1 存在（化学 weight=2，glut→ampa 兴奋 ✓ 与 M3 ampa 一致）；AVM→VB3 存在但
  **TYPE_DIFF**（glut→ampa 兴奋 vs M3 建模 gaba 抑制——见 L9）。
- **肌肉驱动全过**：M4 6 条（VB/DB→muscle_fwd、SMDD×4→muscle_left/right）+ M3 2 条
  （DA→muscle_back、VB→muscle_fwd）全部 present（通道别名映射：M4 muscle_* ↔ 本管线 body_*/head_*）。
- **处置**：连接组是事实，未改动任何权威连接；差异逐条入差异清单 CSV + 本文件；
  供 B1b 组装时**按真实连接组接线**（M3/M4 功能链作为子图先验仅用于权重初始化，见清单 §6）。

## L9 — 递质标注差异（真实递质为准；M3/M4 简化为功能建模）

- **AIY（AIYL/AIYR）**：owmeta/Cook 标注 = **Acetylcholine（cholinergic，Pereira 2015 一致）**；
  M4 将 AIY→RIA 建模为 GABA 抑制——**真实为胆碱能兴奋**。差异记录；P4 反证路径排查时注意：
  真实 AIY→RIA 为兴奋连接。
- **AVM**：owmeta = **Glutamate**；M3 将 AVM→VB 建模为 GABA 抑制——真实为谷氨酸能兴奋。
- **RIA**：Glutamate；**SMDD/SMB/RMD 等头运动**：ACh ✓（M4 一致）。
- **AVA/AVD**（owmeta 空）→ 文献补充 glut；**AVB**（空）→ ach（经典胆碱能命令中间神经元）；
  **PVM**（空）→ glut（触觉）；**RIM**=酪胺能（other）、**RIC**=章鱼胺能（other）。
- **owmeta 覆盖**：302 中 256 有经典递质、46 空（→ 文献补充 5 个 + 41 个神经肽类 → other）；
  多递质神经元取功能性主导（RIM: Ach+Glut+Tyramine → other/酪胺）。
- **备注**：owmeta 个别咽部神经元递质不对称（I2L=glut、I2R=other；I3/I4=other）——如实保留，
  不臆造；P3 咽部节律以电学连接为主，受此影响小（记录于 L12）。

## L10 — 数据获取实测坑（网络受限环境的爬取路径）

- **raw.githubusercontent.com 不可达**（curl 超时/HTTP 000）→ GitHub API（api.github.com）可用：
  - <1MB 文件走 `contents` API（base64）；
  - >1MB 文件（Cook 2019 SI5_adjacency.xlsx 4.17MB）走 `git blobs/{sha}` API（无大小限制）；
  - 全部经 api.github.com 下载成功（c302 仓库 MIT）。
- **git clone github.com 不可达**（443 超时）→ 不用 git，用 API 按文件抓取。
- **networks.skewed.de（ICON 镜像）**：官方 Cook 2019 synapse list + nodes.csv（node_type 分类源）直接可下载。
- **wormwiring.org** 为 JS 应用（首页/connectomes 页无静态数据）；plos/nature 页面 JS 渲染，正文经
  PMC BioC API 抓取核对发布数（4,887/1,447 与 Cook 论文一致）。
- **归档**：全部原始文件存 `data/m5_raw/`（herm_full_edgelist.csv、SI5_adjacency.xlsx、
  owmeta_cache.json、CElegansNeuronTables.xls、Bentley_et_al_2016_expression.csv、
  herm_chem_syn.csv/、herm_gap_syn.csv/、aconnectome_white_1986_*.csv、wormwiring_N2U.txt、
  herm_full_edgelist_MODIFIED.csv）。

## L11 — 命名归一化踩坑（零填充索引）

- c302 edgelist 与 Cook nodes.csv 用 **DA01/AS01/DB04** 等零填充名；规范名 **DA1/AS1/DB4**。
  管线统一 `norm_name()`（去前导零）后与 owmeta 规范 302 roster 对齐；未归一化会导致
  运动神经元全部"失联/缺分类"（实测 ValueError: no class for AS1 → 修复）。
- 附带修正：Cook nodes.csv 的 PHARYNX 类含肌肉（57 细胞）；咽部神经元须按 20 神经元规范 roster
  过滤（I1L..I6/M1-M5/MC/MI/NSM）。

## L12 — owmeta 标注质量（作为权威标注源的风险登记）

- owmeta class 为多标签且噪声大（VB1=[motor,sensory]、MCL=[motor,sensory]）→ 分类**不用** owmeta，
  用 Cook 2019 node_type（权威且与连接数据同源）。
- owmeta 递质为文献编译（Pereira 2015 等），作为递质权威源可接受；空值 → 文献补充 + other 登记。
- owmeta 302 roster 与 c302 PREFERRED_NEURON_NAMES 一致（含 CANL/CANR）→ 作为 302 神经元名单权威。

## L13 — Cook 2019 分类 vs 民俗四类（"~70/~80/~110/~20"）

- 实测（本管线，302 神经元）：**sensory 81 / inter 85 / motor 116 / pharyngeal 20**。
- Cook 原始 node_type：sensory 83（含 AVM/DVA→override 后 81）、inter 81、motor 116、pharyngeal 20。
- vs 民俗期望（±10%）：inter ✓（85 vs 80）、motor ✓（116 vs 110）、pharyngeal ✓（20 vs 20）、
  **sensory ✗（81 vs 70，+15.7%）**——Cook 将 ADE/PDE/ALN/PLN/PHC/DVA/AVM 等计入 sensory，
  经典 White 分类将其部分归 inter；差异如实记录（counts.json class_*_folklore_band），
  建议规划节点以 Cook 分类为权威（sensory 81）或指定 White 经典分类。
- 注：民俗 "~70/~80/~110/~20" 和为 280 ≠ 302——本身为近似值，四类必须合计 302。

## L14 — 自连接/孤立（P1 白名单）

- **自连接**：化学 34 + 缝隙 13（真实存在；ConnectomeToolbox 记 Cook 化学矩阵 38 节点自连接，
  本管线 c302 edgelist 得 34 有向自对 + 13 缝隙自对）→ **全部保留**，CSV note 列标注
  "自连接（真实连接组存在，白名单保留，不静默删除）"（清单 §2.2 白名单条款）。
- **孤立**：仅 CANL/CANR（Cook 2019 数据无连接；规范 roster 成员，c302 同款 include_nonconnected）
  → 白名单 + note 标注。

## L15 — 确定性验证（P1 判据）

- 管线只读本地 `data/m5_raw/`；行序全部固定排序（神经元按 roster 序、边按 (pre,post) 字典序）；
- 两次独立重跑输出 m5_connectome.csv SHA-256 一致：`1ac182c11eb24c17…`（脚本输出 + counts.json 记录）；
- 重跑不联网（数据已归档），逐位一致 ✓。

## L16 — 子图提取结果（P3/P5/P6/P4 结构基础）

- **咽部子图**（P3）：20 节点 + 128 化学 + 33 缝隙（内部连接）→ m5_pharynx_subgraph.csv；
- **命令子图**（P5/P6）：53 节点（触觉 PLM/ALM/PVM/AVM + 命令 AVA/AVD/AVB/PVC + 运动 DA/DB/VA/VB）
  + 245 化学 + 156 缝隙 + 肌肉行 → m5_command_subgraph.csv；
- **趋化子图**（P4）：240 节点（M4 20 角色 + 全连接组上下文）+ 830 化学 + 241 缝隙 → m5_chemotaxis_subgraph.csv；
- 注：趋化子图上下文节点多（240）——M4 角色在真实连接组中高度连接（ASE→AIY/AIB 双向、
  AIY→AIB、AIZ 等），B1b 降阶模型按规模轴取子集时以本子图为准。

---

*本文件为 M5 执行节点（连接组管线）交付物；L7/L8/L13 的区间/分类裁决请求提交规划节点三态裁决
（WORKFLOW 流程，不静默推进）。*
