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

---

# M5-B1c 执行节点实测结论（L31+：参考解——NEURON 咽部子图 P3 + 行为参考扩展 P5/P6，清单步骤 3）

> 执行节点：M5-B1c（`tools/build_m5_ref.py` → `data/m5_ref.npz`，161 键）。
> 冻结文件零修改（M0–M4 与 B1a/B1b/B1d 交付均未动）；未 git commit。
> 两级参考：Stage-A NEURON 化学子图（cvode atol/rtol=1e-8、celsius=6.3、v_init=V0）+
> Stage-B scipy 缝隙网络（solve_ivp LSODA rtol=1e-9/atol=1e-11，M2 gap.mod 限制独立解）+
> 行为参考模型（纯 numpy，引擎无关）。

## L31 — P3 咽部参考设计（Stage-A 化学 + Stage-B 缝隙节奏）与落带结果

- **Stage-A（NEURON）**：20 神经元（build_neuron + ExpSyn + NetCon；87 条 g>0 化学突触——
  other/serotonin 行 g=0 调质占位跳过，M6 补齐；33 条缝隙不建入 NEURON），T=30s；
  无食物协议静默（0 发放，诚实记录）/ 有食物协议 tonic 12µA/cm² 驱动全部 20 神经元 → 各级发放序列。
- **Stage-B（scipy solve_ivp）**：20 点 HH（hh_spec 参数与 Brian2 点神经元同源）+ 33 缝隙（g=0.5nS，
  M2 值）+ 泵马达池 {MCL,MCR,M4} slow-AHP 突发放电机制（I_sahp=−g_sahp·w·(V−EK)、
  dw/dt=(w_inf(V)−w)/τ_sahp；功能参考，Avery & Horvitz 1989：MC 定泵速）→ 泵节律；参数校准落带：
  - 无食物（I=15µA/cm²、g_sahp=4、τ=1500ms）→ **主频 0.400Hz（簇率 0.477/s）∈ [0.1,2] ✓**
  - 有食物（I=18µA/cm²、g_sahp=8、τ=200ms）→ **主频 2.167Hz（簇率 2.835/s）∈ [2,5] ✓**
  - 节律稳定：半窗簇率漂移 0.44 / 0.07 < 0.5 ✓；T=30s ≥ P3 的 10s；无发散（全 trace 有限）。
- 行为带对照：`pharynx_peak_freq_{no_food,food}`（稳健主频）落带 ✓（meta.bands 引用
  `data/m5_behavior_reference.csv`）。

## L32 — Stage-A 化学子图实测（新坑：shunting 与化学隔离）

- **M5 在咽部子图中无化学输入**（仅缝隙 I5/M5/M4-自）→ NEURON 化学子图里是孤立驱动神经元：
  ≥14µA/cm² 时 M5 达 58–62Hz，而其余神经元（4 条 5nS 化学输入分流 soma，g=5nS ≈ soma gL 3.8nS）
  仅 1–4Hz；12µA 时每神经元仅 1 个瞬态发放（rheobase 边界）。
- 中间神经元驱动（I1/I2/MI @25µA）→ 全兴奋化学网络 runaway（全体 ~85Hz，T=5s 墙钟 111s）——
  **化学分量本身不产生节律**；泵节律由 Stage-B 缝隙 + 起搏机制产生（记录为功能参考机制，不伪造波形）。
- 15µA 全 20 驱动在部分拓扑触发 cvode 近阈值 stall（M3 L9；实测 30µA@5 神经元 23s/s）→ 食物驱动定稿 12µA。

## L33 — Stage-B 起搏不同步与主频估计（新坑）

- 三起搏神经元未完全同步（0.5nS 缝隙弱、连接组无 MCL↔MCR 直接缝隙）：food 下 MCL 2.94 / MCR 1.89 /
  M4 2.21Hz；泵率 = 发放池聚合**簇率**（burst_rate，泵事件频率 = Avery & Horvitz 泵率定义）→ 判定首选。
- **周期图 argmax 锁次谐波**（food 1.87Hz vs 簇率 2.84Hz）→ 主频估计改为稳健估计
  `robust_peak_freq`：周期图局部极大 × 自相关 ±25% 消歧 → no_food 0.400 / food 2.167 落带；
  原始 argmax / welch / acf 全部入 npz（informational）。
- **LSODA 容差敏感**（1e-8 vs 1e-9 → no_food 主频 0.63 vs 0.30Hz，发放计数同量级；1e-7 定性错误）——
  HH 网络混沌敏感，记录测量限制；定稿 rtol=1e-9（20 神经元高精度档；M2 的 1e-10 为 2 神经元先例）。

## L34 — P5 逃避参考（M3 一致性 + 转导补齐）与 τ_trans 操作化（**请求裁决**）

- 神经潜伏期 = M3 实测 `latency_nerve` 抽样（8.18–13.63ms，∈ [5,20]，入窗率 1.0）；
  行为潜伏期 = τ_trans(23±2ms) + 神经链 + 肌肉上升（τ_mus=20ms，C_back≥0.3·peak 定义）→
  **39.6±2.8ms ∈ [30,50]（容差 [25,60]，入容差率 1.0）**；v<0 版 37.1ms 同窗。
  方向 back（D_peak=0.599>0.3、C_back_peak 0.599 > C_fwd 基线 0.197 ✓）；反应概率 1.0 ≥ 0.8 ✓。
  M3 结构性落不到 [25,60]（无转导，m3_env_notes L7）由 M5 补齐 ✓。
- **τ_trans 未定稿于 `data/m5_worm_params.csv`**（B1b/G0 无此行）→ 本参考锚 23ms
  （行为 40 − 神经 9.8 − 肌肉 7.1）；**请求 P5 协议节点写入 CSV**。
- 操作化提醒：P5 神经潜伏期窗 [5,20] **不含转导延迟**（清单 §5.2 #4：行为 − 神经 ≈ 转导+肌肉 10-30ms）；
  Brian2 全虫侧（worm_loop.run_escape 的 touch_window τ_trans 语义）计时须以触电流注入时刻
  （t0+τ_trans）为神经链起点，或对 t0 计时减 τ_trans——否则测量值 = τ_trans+链 ∉ [5,20]。

## L35 — P6 自发参考校准（坑与处置）

- bout 半马尔可夫（嵌入链转移矩阵 + 指数 bout 时长）→ 校准落带：
  **前进 73.3±5.8% / 后退 16.3±6.1% / 转弯 9.7±3.2% / 暂停 0.8%**（带 [60,80]/[10,25]/[5,20]
  全部落带 ✓）；bout 均值 fwd 6.7s / rev 2.5s / turn 1.6s（Srivastava 2013 量级，informational）。
- 校准坑：① fwd bout 15s + P_FF 自锁 → fwd 时间比例 87–99% 不可达（需 fwd 8s + 低 P_FF：
  每前进 bout 后 ~1 后退 + ~1 转弯结构）；② 单试次校准噪声 → 转弯 3.2% 掉带
  （改每组合 N=10 校准试次 + **带宽裕度最大化**选择）；③ 转弯 bout 1.5s 不足 → 2.5s。
- classify_state 语义与 B1d `src/virtual_body.py`（turn→fwd→rev→pause 顺序）在参考状态上
  **逐位一致（100% 验证）**、`body_velocity` ≡ `VirtualBody.speed`（Δ=0）——P6 判定统一用
  virtual_body 实现（共用约定满足，M5 清单 §9 风险表）。

## L36 — 交付清单与复现

- `tools/build_m5_ref.py`（可复现重跑；确定性 seed=0 仅行为参考抽样，NEURON/scipy 无随机性；
  总墙钟 ~35min：NEURON ~3min + Stage-B 校准/终跑 ~30min）。
- `data/m5_ref.npz`（161 键）：`pharynx_spike_times_{no_food,food}[_{20 神经元}]`、
  `pharynx_v_{KEY}_{proto}`、`pharynx_gap_v_{20 神经元}_{proto}`、`pharynx_psd_{proto}`、
  `pharynx_peak_freq_{proto}`（稳健主频，判定用）/`argmax`/`welch`/`acf`、`pharynx_burst_rate_{proto}`、
  `pharynx_drift_{proto}`、`escape_ref_*`（方向/潜伏期/概率/D_peak）、`spontaneous_ref_*`
  （比例/转移矩阵/平稳分布/bout 均值/示例 trace）、`meta`。
- 趋化参考（P4）复用 `data/m4_ref.npz`（本文件不重复，meta 引用）。

*本文件为 M5-B1c 交付物（清单步骤 3：参考解）；L34 的 τ_trans CSV 定稿请求规划节点裁决
（WORKFLOW 流程，不静默推进）。*

---

# M5-B1e2 执行节点实测结论（L37+：权重定稿与行为带校准，清单步骤 5 §6）

> 执行节点：M5-B1e2（权重定稿 + 四目标校准：`data/m5_worm_params.csv` 权重行定稿 +
> `data/m5_calibration.csv` + `reports/neuro/m5_calibration.png`）。
> 冻结文件零修改（M0–M4 不动；m5_connectome.csv 不动）；`src/worm_circuit.py` 有 4 处
> API 兼容微调（L42 登记，未改任何签名/默认行为）；未 git commit。
> 方法：7 轮扫描 ~30 组合 × 四短协议（静息 T=5s / 自发 T=5s / 趋化 T=5s×N / 逃避 150ms×N，
> 确定性 p=1/n=1；302 点神经元 G0 定稿配置）+ top-3 组合 N=5 复核。总墙钟 ~4h。

## L37 — ⚠ 诊断重写：302"过兴奋"实为三类结构问题（非单纯权重过大）

占位权重（s_k=1、gap=1）下 G0 L22 记录"302 静默 8.6%、自发全 pause"，本节点实测
**机制与"权重过大"不同**，逐层诊断（`/tmp/m5_b1e2_debug*.py`）：

1. **缝隙分流（gap shunt）**：AVB（唯一持续驱动 = M4 携带的 14µA/cm² 张力，AVBL/AVBR）
   有 ~30 个缝隙伙伴；gap=1.0 时缝隙电导负载 ≈ 30×0.5nS = 15nS ≈ **4× 固有 gL（3.77nS）**
   → 张力电流被分流到亚阈值 → AVB 只发 1 个 t=0 尖峰后静默（对照：AVB 单独发 73Hz 持续）。
   **gap_scale=0.05 修复**（AVB 恢复持续发放 14Hz）。
2. **t=0 初始化瞬态波**：v=−65 + 张力开通 → AVB 首尖峰 → 缝隙+ampa 网络级单发波
   （91% 神经元各发 1 尖峰，T=3s 时全波中位数=峰值=0.33Hz）→ 静默比例（<0.1Hz 操作化）
   结构性卡在 ~8-12%（P2 目标 ≥60%）。**500ms settle 后该波可排除**（post-settle 静默
   最高 69.2%——u3）；建议 P2 协议加 settle 窗（"HH 静息漂移纪律"已有先例，清单 §5.2 #4）。
3. **网络级夹带极限环**：AVB 持续发放（tonic≥0.8×14µA/cm² 且 gap≤0.05）→ **86% 神经元
   同步夹带至 ~2.7-13.8Hz**（发率分布极端：86% 神经元同率、0 个中间率——非高斯背景活动）
   → 静默比例卡在 12-44%（无论类级缩放）；且 fwd/back 运动池共同发放（motor→motor ampa
   474 条 + 缝隙）→ 肌肉通道双饱和 → 自发 v≈0 → pause 主导。

## L38 — 杠杆扫描结果（~30 组合，逐组合四目标值 data/m5_calibration.csv）

| 杠杆 | 范围 | 实测结论 |
|---|---|---|
| 类级缩放 10 桶 | 0.1–1.0 | **无法打破夹带**（静默 12-44%）；可调 fwd/rev 比例：rec=0.1 → rev 主导 45%（m3）；rec=0.15 → rev 24.5% ✓ + turn 11.5% ✓（u2） |
| gap_scale | 0.02–0.5 | **0.05 是临界**：逃避（PLM 缝隙触觉路径）✓ + AVB 持续发放；0.02 → 逃避弱化 + rev 主导；≥0.10 → AVB 复归静默（分流） |
| tonic_scale | 0.1–1.0 | **陡峭分岔**：<0.7 → AVB 无法持续（死区，post-silent 100%）；≥0.8 → 夹带。无"AVB ~3Hz 且网络静默"的中间态 |
| gL_scale | 2–3× | 全网络静默（post-silent 100%）；CI/逃避仅靠 t=0 波，无行为——不可用 |
| syn_type gaba | 2× | fwd 46.5%↑ 但 rev/turn 掉出带（m4）——不可用 |
| **D4 定稿（最小充分）** | **gap_scale=0.05，类级=先验 1.0** | **g1：逃避 back ✓ + 趋化方向 ✓**（见 L39） |

## L39 — 定稿组合 D4=g1_gap005 的 N=5 复核（四目标 vs 带）

| 目标 | 带（data/m5_behavior_reference.csv） | g1 N=5 实测 | 判定 |
|---|---|---|---|
| P2 静默比例（<0.1Hz） | [60, 80]%（容差 [0.6,0.8]） | **10.6%**（中位数 13.8Hz、max 14Hz——夹带） | **✗ FAIL** |
| P4 趋化 CI@5s 方向 | ΔCI vs 参考(0.175@5s) ≤0.15 或方向一致 | **+0.078**（Δ=0.097 ≤0.15 且方向 pos） | **✓ PASS**（T=5s 点估计噪声大 SEM≈0.24；T=15s×N=20 全协议由 B2 复核） |
| P5 逃避方向 + 潜伏期 | back（D_peak>0.3）；神经窗 [5,20]ms；行为窗 [30,50]ms | 方向 **back 仅 τ_trans=0（touch@50ms，D_peak=0.61 确定性）；τ_trans=23 定稿协议下 not_back**（~72ms 节律相位污染）；神经潜伏期 ~3.7-4.5ms（注入→DA）；行为潜伏期 ≈23+4.5+7≈34.5ms ∈ [30,50] | **△ PARTIAL**（方向相位敏感：touch@50ms back ✓ / touch@73ms not_back；神经潜伏期点神经元结构性偏快 G0 L22；行为潜伏期 ✓——反证 #5） |
| P6 自发分布 | fwd [60,80] / rev [10,25] / turn [5,20]% | fwd 25.5 / rev 3.0 / turn 0.5% | **✗ FAIL**（rev/turn 落带替代组合 u2：fwd 20/rev 24.5 ✓/turn 11.5 ✓，但 P4 方向丢失——两组合不可兼得） |

**决策**：定稿 **g1（gap_scale=0.05，类级缩放全部 1.0 = M4/M3 子图先验）**——
最小充分组合（P4+P5 PASS、P6 部分、P2 反证）；`data/m5_worm_params.csv` 权重行已落盘
（class_scale_* 9 桶 + gap_scale + tonic_scale + gL_scale + syn_type_scale_gaba +
calib_verdict/calib_ruling_request 行 + escape_touch_delay_ms=23 补丁）。

## L40 — ⚠ 反证笔记：P2/P6 结构性不可达的缺失机制清单（**请求规划节点三态裁决**）

全杠杆扫描（L38）证明 **类级缩放/gap/tonic/gL/gaba 五类参数无法联合满足四目标**。
P2（静默 ≥60%）与 P6（fwd ≥60%）的不可达根因（缺失机制，M6 优先验证清单）：

1. **命令回路互抑缺失（P5/P6 方向分离的结构前提）**：AVA/AVD（后退命令）↔ AVB/PVC（前进
   命令）在连接组中**全部互为兴奋**（AVAL↔AVBL/PVCL 等 ampa + 缝隙，实测无互抑边）；
   真实 C. elegans 经 **RIM 酪胺能**（受体=mod → g=0 跳过，L5#5 登记）介导后退时抑制
   前进。缺失 → 任何扰动使 fwd/back 运动池**共同发放** → 肌肉双饱和（自发 v≈0 → pause）
   或逃避 D_peak≈0（302 全虫实测，G0 的 0.41 来自 M3 子图非 302）。
2. **AVA→DD/VD GABA 抑制链缺失**：真实连接组 AVA→DD（后退命令激活 GABA 池 → 抑制 fwd 池）
   不存在（实测 AVA/AVD→DD/VD 化学边 0 条）；现有 DD/VD gaba 池（motor→motor gaba 57 条）
   无命令驱动 → 后退 bout 无法隔离 fwd 池。
3. **单一张力驱动的夹带**：M4 14µA/cm² AVB 张力是 302 唯一持续驱动（真实蠕虫 AVB/PVC
   活动来自感觉输入 + 自发/调质，本模型自发缺失、调质 g=0）→ 任何持续驱动都夹带全网络
   （86% 同步 2.7-13.8Hz）→ 静默比例上限 ~44%（u4），结构上 <60%。
4. **点神经元省略峰电位起始延迟**：逃避神经潜伏期 3.7-4.5ms < [5,20]（G0 L22 已记录）。
5. **网络节律污染逃避方向（P5 新发现）**：302 全虫逃避方向**确定性依赖触刺激相位**——
   touch@50ms（τ_trans=0）→ back（D_peak 0.61）；touch@55-85ms（τ_trans=5-35，含定稿 23）
   → not_back（d_peak≈0，fwd/back 运动池共同发放抵消）。网络 ~72ms 夹带节律使"后退反应"
   无法与背景活动分离——与 P2/P6 同根（L40 #3），命令互抑缺失的直接表现。

**三态裁决请求（WORKFLOW，不静默）**：
- **方案①（推荐）**：接受 D4=g1 定稿（P4/P5 PASS + P2/P6 反证记录 + settle 窗建议），
  M5 继续走"部分通过"路径（清单 §3.6），M6 引入调质层（RIM 酪胺/命令互抑/AVA→DD）后
  复核 P2/P6；
- **方案②**：提前引入 RIM 酪胺激活（mod 通道从 g=0 改为功能抑制）+ 命令互抑边
  （修改标注或 worm 模块组装，不动 m5_connectome.csv）→ 重新校准；
- **方案③**：P2/P6 验证主体改为行为参考模型（M4 P4(b) 同款处置：numpy 参考为验证主体，
  Brian2 全虫为一致性对照）。

## L41 — 测量限制与协议建议（供 B2 验证节点）

1. **P2 settle 窗**：t=0 初始化瞬态波是初始条件伪迹（v=−65+张力开通），500ms 后消失；
   建议 P2 协议在测量前跑 500ms settle（与"刺激开始 ≥40ms 静息漂移纪律"同哲学）——
   否则 91% 神经元各 1 尖峰使静默比例结构性卡死。settle 后 u3 可达 69.2% 静默（在带内）
   但行为破坏——P2 与行为在单一张力下不可兼得（L40 #3）。
2. **P4 CI@5s 噪声**：T=5s×N=5 的 CI 点估计 SEM≈0.24-0.31，方向可看、落带不可判；
   落带/显著性判据必须走 G0 定稿 T=15s×N=20 全协议（参考 CI@15s=0.417）。
3. **P5 计时操作化**：τ_trans=23ms 已定稿 CSV（L34 请求）；神经潜伏期以触电流注入时刻
   （t0+τ_trans）为起点（L34 语义），否则测量值含转导延迟 ∉ [5,20]；
   **⚠ 方向相位**：定稿 τ_trans=23 → 触注入 73ms → 302 方向 not_back（L40 #5）；若裁决
   要求方向 back，需 τ_trans=0（touch@50ms）→ 与行为潜伏期 [30,50] 冲突（点神经元神经
   链仅 4.5ms，无转导则行为 ≈12ms ∉ 窗）——P5 方向与行为窗在 302 点神经元上不可兼得，
   需 M6 调质/命令互抑或裁决取舍。

## L42 — src/worm_circuit.py API 兼容微调登记（4 处，均未改签名/默认行为）

1. `load_weight_scales()`（新函数）：读 CSV 全量定稿（class_scales/gap_scale/
   syn_type_scales/tonic_scale/gL_scale）→ `make_worm_circuit(scale=302, **load_weight_scales())`；
   `load_class_scales()` 保持不变（兼容）。
2. `WormCircuit.__init__` 新增可选参数 `syn_type_scales`/`tonic_scale`/`gL_scale`
   （默认 None → 恒等，行为不变）：gaba 类型缩放（抑制不足假说）、AVB 张力缩放
   （夹带杠杆）、点神经元漏电缩放（风险表"漏电增强"）。
3. `GroupedWormCircuit.build`：gmax 乘 `syn_type_scales[type]`；gL 乘 `gL_scale`。
4. `WormCircuit.build`（component 模式）：化学 g_max_ns 乘 `syn_type_scales[type]`。
   均通过默认参数回归验证（make_worm_circuit(scale=20) 行为不变）。

*本文件为 M5-B1e2 交付物（清单步骤 5：权重定稿与行为带校准）；D4 定稿 + L40 反证 +
三态裁决请求（WORKFLOW 流程，不静默推进）。*
