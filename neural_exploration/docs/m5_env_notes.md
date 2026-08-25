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

---

# M5-B1b 执行节点实测结论（L17+：降阶模型组件 + 铁律 C 缩放扫描 + G0 决策）

> 执行节点：M5-B1b（src/point_neuron.py + src/worm_circuit.py + tools/scan_m5_scaling.py）。
> 冻结文件零修改（M0–M4 全部 src/tests/tools/data 不动）；未 git commit。

## L17 — 点神经元与 M2 突触组件适配（Brian2 2.6.0 实测，二选一定稿：方案一）

- **结论：M2 `ChemicalSynapse`/`GapJunction`/`Muscle3.connect_driver` 不经修改即可复用
  PointNeuron**（清单 L1 方案①）：`.neuron`（单隔室 NeuronGroup）、`.label_of(site)→0`、
  `.soma_area_cm2()`（1.257e-5 cm²，M4 ase_site 同值）、`.density_to_nA()`。
  实测：触刺激（60µA/cm²@50ms）→ pre 发放 50.7ms → post 经 AMPA 5nS @52.0ms；GapJunction
  数值有限。
- 方程要点（M1 同款处置，实测坑）：① **`mS` 不在 Brian2 DEFAULT_UNITS**（`_resolve_external`
  只查 DEFAULT_UNITS/group namespace/run namespace）——方程串只写 siemens/meter**2 等，
  密度作状态变量 Python 侧赋值；② TimedArray 以 `stim_var(t,i)` 调用（不可同时声明同名
  变量）；③ 缝隙电流显式 `(stim+I_gap)/AREA` 入 dv/dt（不依赖 SpatialNeuron 自动注入）。
- **数值方法（dt 定稿，M4 L16）**：点档 dt=0.1ms **rk4 发放后 NaN，exponential_euler 稳定**
  （vmax≈42.6mV）；双隔室 dt=0.05ms rk4 静息自发尖峰后发散（小隔室高 gNa=300 更 stiff），
  exponential_euler 稳定。定稿：point=(0.1, exp_euler)、two_comp=(0.05, exp_euler)、
  multicomp=(0.01, rk4)（M1 冻结）。

## L18 — ⚠ M2 GapJunction (summed) 在多缝隙拓扑下不可用（新坑）

- 实测：同一神经元 ≥2 缝隙伙伴 → `net.run` 报 `Multiple 'summed variables' target the
  variable 'I_gap'`（build 不报，run 时 _check_multiple_summed_updaters 抛错）。
- 真实连接组 1093 缝隙、多伙伴普遍 → **M2 组件仅单对语义**；worm 模块新建批量缝隙组件：
  `I_gap_in`/`I_gap_out` 两个 (summed) 目标 + `I_gap = I_gap_in + I_gap_out` 派生
  （一个 Synapses 一个对象，冻结文件零修改）。二选一定稿：单对/小图用 M2 组件验证，
  连接组规模用批量组件。

## L19 — 冷编译预算（M5 §8 风险表预注册，实测放大）

- **每个 Synapses 对象 ~5.2s 编译**（对象名进生成代码串；相同 on_pre 也逐个编译）；
  302 component 模式（2472+1093+68 ≈ 3633 对象）≈ **5-6h 冷编译不可接受**；
- **grouped 模式**（全部点神经元合为一个 NeuronGroup + 每类型一个 Synapses + 每通道
  一个肌肉驱动）→ 302 冷编译 ≈ **10min**，稳态快 ~15×（20 档 T=1s：grouped ≈3s vs
  component ≈50s；T=5s 实测 16.6s/trial）；
- component vs grouped 一致性：CI 逐位一致（no-mechA）、发放计数一致、发放时刻尾部
  ~0.2ms 浮点漂移（算术顺序差异，行为等价）；
- **stim 形状纪律**：`PROTOCOL_WINDOW_MS=6000` 固定窗口形状（探针/趋化/静息/自发共用
  同一编译产物）；多隔室 HH 内存约束按 t_total 取形。

## L20 — 连接组结构实测（B1a m5_connectome.csv 15:45 到达；fallback→connectome 切换）

- 解析：302 神经元、化学 3638（ampa 2343 + gaba 129 可用；mod 294 + none 872=1166 跳过，
  调质占位 M6 补齐）、缝隙 1093、肌肉 68（fwd/back/head_left/right）；
- **头部转向**：真实连接组 SMDD→head 肌肉存在，但 M4 手工链 SMDD→C_left 的中间级为
  RMD/RIV；机制 A（pirouette）由 SMDD 发放触发身体转向事件 → 20 档仍可趋化；
- **M4 VB/DB = Cook VB1..VB11/DB1..DB7** → 20 档子集必须补 VB*/DB*（否则无 C_fwd 驱动）；
- M4 20 角色 18/20 直接存在（VB/DB 除外），与 B1a crosscheck 一致。

## L21 — ⚠ 僵尸进程清理（M4 L25 教训复发）

- 本节点开始时 M4-B2 遗留 `validate_p5_chemotaxis_ablation`（12:43 起、无完成标记）空转
  2.5h+ 并竞争 CPU/编译缓存 → kill；父 driver 正常 finalize（EXEC2 DONE）。
  **长任务必须完成标记 + 验证前检查并发清空**（M4 L21/L24b/L25）。

## L22 — 降阶正确性（§3.4，G0 门；详值见 data/m5_scaling.csv）

- **逃避方向**：点神经元 M3 反射子图 D_peak=0.410 > 0.3 → **back**，与 M3（0.352）一致 ✓；
- **逃避潜伏期**：点档 4.70ms < M3 窗 [5,20]ms（M3 实测 8.23–8.93ms）——点神经元省略
  峰电位起始延迟 → 链路更快；**记录测量限制（不静默）**：方向一致、潜伏期结构性偏快，
  P5 判据需在权重定稿后复核（touch_delay 补偿或反证记录）；
- **趋化 CI**：20 档（连接组接线，点神经元）vs M4 记录见 m5_scaling.csv（20/point 行：
  方向一致；ΔCI 以 M4 点0 0.043@5s / 参考 0.175@5s 为参照记录）。
- **静息（占位权重）**：组中位数 ~65Hz、静默 ~0.2——过度兴奋（占位权重未校准，
  §6 类级缩放下调），如实记录不调参。

## G0 决策（第一关键决策步；定稿数据见 data/m5_worm_params.csv + data/m5_scaling.csv）

**G0 结论：PASS（带记录测量限制）** —— 行为验证协议按下列定稿配置执行（§3.3 决策规则）：

| 决策项 | 定稿值 | 依据（铁律 C 实测，data/m5_scaling.csv） |
|---|---|---|
| 规模 | **302（全连接组）** | 302 档点神经元 CI=0.243@5s > 0（方向一致），ΔCI vs 参考模型(5s)=0.175 → **0.068 ≤ 0.15 ✓**；行为随规模持续（20→0.403、50→0.333、100→0.398、302→0.243） |
| 保真度 | **点神经元（单隔室 HH）** | 20 档 point CI=0.403 vs two_comp CI=0.428 → **ΔCI=0.025 ≪ 0.15 收敛**（铁律 2：不为精细而精细；铁律 C：曲线决定） |
| dt/方法 | **0.1ms / exponential_euler** | L17 实测：dt=0.1 rk4 发散、exp_euler 稳定；定稿后不变（M4 L16） |
| T（P4 趋化协议） | **15000ms** | 参考模型 N=20 通过率 ≥80% 的最短 T（ref-T15000：p=0.002、d=0.79；L23：显著区 T≥15s 稳健） |
| 预算 | **≤200 CPU-小时（预注册）** | 302 点神经元探针 T=1s=3.8s、T=5s≈20s/trial（实测）→ T=15s×N=20（梯度+对照）≈ **5-10 CPU-h**，远低于上限 ✓ |

**降阶正确性（§3.4，G0 门组件）**：
- 趋化：20 档连接组接线点神经元 CI=0.403@5s——方向与 M4 记录一致（正趋化）；
  ΔCI vs M4 Brian2 点0(0.043@5s)=0.36、vs 参考(0.175@5s)=0.228 > 0.15 → **方向一致条款满足**
  （§3.4 语义：连接组真实接线比 M4 手工子图更有效，如实记录测量限制）；
  302 档 ΔCI vs 参考(5s)=0.068 ≤ 0.15 ✓；
- 逃避：方向 **back**（D_peak=0.410 vs M3 0.352，一致 ✓）；神经潜伏期 **4.7ms < M3 窗
  [5,20]ms**（M3 8.23–8.93ms）——点神经元省略峰电位起始延迟 → 结构性偏快，
  **记录测量限制（不静默）**：P5 判据在权重定稿后复核（touch_delay 补偿或反证记录）。

**部分未涌现（§3.6 部分通过路径，不阻塞 P4/P5）**：
- 静息（P2）/自发（P6）在**占位权重**（g=5.0nS 全连接）下过度兴奋/无 fwd-rev 状态
  （302 静默 8.6%、自发全 pause）→ **§6 权重校准前置**（类级缩放下调兴奋性 +
  命令子图 AVA/AVD/AVB/PVC 耦合检查），校准后复测 P2/P6——按预注册判据，不做事后调参。

**结构发现（L7）**：50/100 纯拓扑序子集无运动神经元 → 无 C_fwd（CI 为起点象限伪迹）；
已改为**类平衡子集**（含运动神经元）重测 50/100（50→0.333、100→0.398）。

---

# M5-B1d 执行节点实测结论（L23+：全虫闭环实现 + 冒烟测试，清单步骤 4 收尾）

> 执行节点：M5-B1d（src/virtual_body.py + src/worm_loop.py + tests/neuro/test_worm_smoke.py
> + reports/neuro/m5_smoke.png）。冻结文件零修改（M0–M4 不动；m5 数据 CSV 不动）；
> worm_circuit.py 有 2 处 API 兼容微调（L24/L25，未改签名/数值）；未 git commit。

## L23 — ⚠ m5_worm_params.csv 的 value 列错位（数据行 11+ 字段 vs 列头 10 列）

- 列头 10 列（value 在 fields[8]），但数据行 **11+ 字段**（fields[2..8] 共 7 个空列 +
  value 在 **fields[9]** + note 在 fields[10:]）——DictReader 会把 value 读成空、
  真值落进 note（实测 `body,v_fwd0,...,1.0,最大前进速度...` 的 value 读到 ''）。
- **处置**：`worm_loop.load_m5_worm_params` 按位置解析（value=fields[9]，len<11 行
  视为无 value——仅 model 描述行）；**下游 P4/P5/P6 验证脚本解析该 CSV 必须用同一
  位置语义或先修 CSV**（B1b 数据文件未动）。
- 附带：CSV 的 `escape_touch_delay_ms`（τ_trans）行**缺失** → worm_loop 机械刺激
  协议默认 τ_trans=0（触刺激在 t0=50ms 开始）；P5 节点需在 CSV 定稿
  （行为潜伏期 30-50 − 神经 5-20 ≈ 10-30ms）。

## L24 — 子图 CSV 列头带引号 → load_connectome 解析为 0 神经元（新坑）

- m5_pharynx/command/chemotaxis_subgraph.csv 的列头行是 `"role,...,note"`（**带引号**），
  `_parse_connectome_csv` 的 `startswith('"')` 过滤把列头丢掉 → DictReader 误把首数据行
  当列头 → 0 神经元。主 m5_connectome.csv 列头无引号不受影响（B1b 未踩到此坑）。
- **微调（API 兼容）**：`_parse_connectome_csv` 增加 `_clean_line()`（去行外层引号后
  再按 `#` 过滤）；解析结果不变（主 CSV 计数 302/2472/1093 与 B1a counts.json 一致）。

## L25 — 子图 CSV 无 receptor 列 → 化学突触全跳过（微调回退）

- 子图 CSV 只有 neurotransmitter 列（无 receptor）→ 原解析器 receptor='' → 全部按
  mod/none 跳过（咽部 128 化学 → 0 可用）。**微调（API 兼容）**：receptor 空时按 L4
  递质→受体映射回退（ach/glut→ampa、gaba→gaba、dopamine/serotonin→mod、other→none）。
  咽部子图现解析 87 ampa/gaba 可用 + 33 缝隙；命令子图 245 化学 + 156 缝隙 + 39 肌肉；
  主连接组 2472 化学 + 1093 缝隙 + 68 肌肉不变。

## L26 — 咽部子图占位权重：无自发发放；MC 驱动只激活 MCL/MCR

- 无刺激咽部子图 10s（占位 g=5nS + gap=0.5nS）：**0 发放**；MC（MCL/MCR）驱动
  60µA/cm² → MC ~115Hz 节律发放，**其余 18 角色静默**（子图未校准 → P3 需 §6
  权重校准后才有网络级泵动节律）。冒烟以 MC 驱动断言"节律发放存在（spike>0）"，
  不预注册节律窗（P3 协议节点职责）。

## L27 — ⚠ PROTOCOL_WINDOW_MS=6000 < G0 P4 T=15s（编译形状与协议冲突，请求裁决）

- B1b 定稿 `PROTOCOL_WINDOW_MS=6000`（扫描 ≤5s 时定），G0 定稿 P4 T=15000ms →
  闭环 epoch 在 t>6000ms 时 `stim.values[i0:i1]` **越界 IndexError**（固定形状纪律
  M4 L16 与 G0 协议的冲突）。冒烟短协议（≤2s）不受影响；
  **P4/P6 验证节点需：扩展窗口（一次性冷编译 ~10min/302）或规划节点裁决缩短 T**。
- worm_loop `run_escape` 的触刺激注入已做窗口越界钳位（i0/i1 min 到窗口），
  但 `GroupedWormSession.run_epoch` 的 ASE 写入（B1b 文件）未钳——长协议前必须解决 L27。

## L28 — 冒烟实测数值（tests/neuro/test_worm_smoke.py 全绿，详见回报）

- 点神经元脉冲（60µA/cm²@50-55ms）→ 发放 @50.7ms，同参数重跑逐位一致（L17 复核）；
- 20 规模 WormLoop 趋化短协议（T=2s）：CI=0.0 ∈[−1,1]、轨迹有界无 NaN、状态比例
  {fwd 0.94, turn 0.06}（M4 张力携带下前进主导，与 B1b 观察一致）；
- 闭环确定性：同参数重跑 **r1==r2 逐位一致**（ChemotaxisResult.__eq__）；
- 静息 20 规模 T=1s：中位数 65Hz/静默 20%/最大 66Hz，无 NaN（占位权重过度兴奋，
  G0 L22 已记录 → P2 校准后判带）；自发：fwd 0.9/turn 0.1（无 rev——命令子图
  耦合未校准，P6 前置 §6，G0 部分通过路径）；
- 机械逃避：M3 反射子图降阶 D_peak=0.410 → back（与 G0 L22 一致）；worm_loop
  触刺激窗 τ_trans 语义验证通过（τ_trans=10ms 时窗右移 100 步）。

## L29 — virtual_body 行波参数未定稿于 CSV

- body 行只有 v_fwd0/v_rev0/omega_max/dt_b；`gait_period_ms`/`wave_amp`/`wave_lambda`
  /`head_turn_gain` **缺失** → `VirtualBody` 默认（gait 500ms、wave_amp=0 关闭、
  head_turn_gain=0 informational——G0 未提升为验证级）。P6 节点如需验证级行波
  需在 CSV 定稿；`classify_state` 阈值已定稿（spont_v_thr_frac=0.05、
  spont_omega_thr_frac=0.2 ✓ 读取成功）。

## L30 — 新增 API（供 B2 写验证脚本）

- `virtual_body.py`：`VirtualBody(v_fwd0, v_rev0, omega_max, dt_b, arena_L, boundary,
  gait_period_ms, wave_amp, wave_lambda, body_len, head_turn_gain, turn_omega_pir,
  turn_duration_ms)`；`speed(c_fwd, c_back)`、`turn_rate(c_left, c_right, t_ms)`、
  `step(c_fwd, c_back, c_left, c_right, dt_ms, t_ms)`、`pose_y(x, t_ms)`、
  `head_sway(t_ms)`、`integrate(c_fwd, c_back, c_left, c_right, dt_ms)`、
  `reset(x, y, theta)`、`assert_trajectory(x, y)`；
  `classify_state(v, omega, c_fwd, c_back, v_thr_frac=0.05, omega_thr_frac=0.2,
  v_fwd0=1.0, omega_max=1.0)`（阈值 CSV 定稿，不做事后调）；
  `state_fractions(states)`；`StateThresholds(v_thr_frac, omega_thr_frac)`。
- `worm_loop.py`：`load_m5_worm_params(csv_path=None)`（位置解析，L23）；
  `WormLoop(circuit, env=None, body=None, seed=None, params_csv=None)`——
  `run_trial(start_x, start_y, theta0, t_total_ms, seed, s_override)`、
  `run_trials(n_trials, seed_base, t_total_ms, start_jitter, s_override)`、
  `run_control(...)`、`run_spontaneous(t_total_ms, seed)`（→ frac/states/v/omega）、
  `run_escape(t_total_ms, seed, touch_roles, backward_roles)`（→ d_peak/direction/
  c_back/c_fwd/neural_latency_ms）、`run_resting(t_total_ms, seed)`（委托 circuit）、
  `touch_window()`（→ (i0, i1, n_steps)，τ_trans 语义）。
- 冒烟断言 ≥10（清单要求 ≥8）：302 计数/交叉核对/点神经元/趋化短协议/逃避方向/
  咽部节律/静息无 NaN/闭环确定性/出图/virtual_body 后退与分类。

*本文件为 M5-B1d 交付物（清单步骤 4 收尾）；L23/L27 的 CSV schema 与窗口-协议冲突
请求规划节点裁决（WORKFLOW 流程，不静默推进）。*
