# M7 报告：机制回迁数字大脑（P-A1 机制封装 + P-A2 接入 + P-A3 场景 + P5 回归）

> 生成：M7-B2 验证+报告节点；2026-08-27T14:14:00Z（UTC）

> 判定：P-A1 **pass**；P-A2 **pass**；P-A3 **pass**；P5 **pass**；P6 **pass**（主 agent 判定框架落实；G1 门通过——阶段一收官）。

## 1. 交接（M6 → M7 组装方式；组合不修改纪律）

**M6 → M7 组装方式（组合不修改纪律）**：
- M0–M6 全部冻结文件（`src/`、`tests/`、`tools/validate_*`、`data/` 定稿 CSV 含 `m5_connectome.csv`）零修改；M7 只新建（机制模块/接口/测试/参数 CSV/验证工具/报告）+
  对 `symbolic_interface.py` 做 2 处**加法式**修改（可选 `innate` kw 参数 + `self.innate` 属性，solve 语义零修改）；
- **回迁设计决策（M7 清单 §0 #2）**：机制回迁 = **行为级等价**（方向/衰减/增益/Δw 符号与冻结锚一致）——不是把 Brian2 网络搬进数字大脑（神经实时动力学与符号推理语义不同）；机制模块 `digital_brain/src/innate/innate_mechanisms.py` 纯 stdlib （无 brian2 依赖），两个 venv 均可 import；
- **参数唯一定稿源**：`neural_exploration/data/m7_innate_params.csv`（fields[9] 位置解析，M5 L23 惯例）；模块只读 CSV，不重训/校准（冻结纪律）；
- **未 git commit**（M0–M6 惯例，交规划节点验收后提交）。

## 2. G1 环境门（数字大脑环境——P-A2/P-A3 前提）

**G1 门（数字大脑环境门——P-A2/P-A3 前提）✅ 通过**（M7-B1a 实测，L5/L7）：

| 项 | 值 |
|---|---|
| 环境路径 | `small_world/.venv-db`（python 3.9.6 venv） |
| 依赖 | networkx 3.2.1 / pydantic **1.10.14**（锁 v1——代码用 `class Config:` v1 风格）/ PyYAML / pytest |
| G1 基线 | `pytest digital_brain/tests` = **117 passed**（1.01s）全绿零回归 |
| 新增回归 | + `test_m7_innate.py` 7 断言 = **124 passed** |
| 判定 | **G1 通过 → 允许写数字大脑代码**（机制封装/接口接入已按此执行） |

环境分叉根因（L5/L7）：`.venv-neuro` 无 networkx/pydantic/PyYAML → `pytest digital_brain/tests` collect 失败（7 文件 ERROR）。处置 = 独立 `.venv-db` （选项②，清单 §0 #1 标注'最安全'）；双线环境分离执行互不污染。

## 3. 机制提取与封装（P-A1）

**P-A1 判定：pass**（6/6 机制封装，等价性 21/21 PASS——详见 §4）。

**回迁结论（B1a 入档）**：M-1/M-2/M-4 **已回迁（最小集，P-A1 硬判据 ≥3 项满足）**；M-3/M-5/M-6 **已回迁（扩展集）**；M-7 **未回迁（设计依据交接阶段二）**。

| # | 机制 | 冻结验证状态 | 封装类（digital_brain/src/innate/） | 数字大脑接入点 |
|---|---|---|---|---|
| M-1 | 触觉反射弧 | M3 已验收（D_peak=0.352–0.369） | `ReflexArcMechanism` | 先天运动反应：`InnateInterface.actuate(escape)`（触刺激→定向回避硬连线） |
| M-2 | 嗅觉趋化回路 | M4 已验收（CI(25s)=0.494 方向正） | `ChemotaxisMechanism` | 环境感知：`sense(odor@pos)`（浓度/梯度）+ `actuate(approach)`（正向梯度趋利） |
| M-3 | 咽部 CPG | M5 P3 通过（0.400/2.167Hz 双带） | `CpgMechanism` | 行为节奏：`actuate(rhythm)`（时间→节律相位/频率） |
| M-4 | 习惯化 | M6 P3 通过（R²/τ/消融/恢复） | `HabituationMechanism` | 先天适应：`adapt(n)`（重复刺激→响应衰减） |
| M-5 | 联想学习 | M6 P4 通过（Δw_train>0.1/η=0→0/Δw_ext<0） | `AssociativeMechanism` | 机制层可观察：`brain.innate.mechanisms["associative"]`（不过四方法路由，L16） |
| M-6 | 神经调质层 | M6 P2 反证记录型（gate∈[0.3,1.2]） | `ModulationMechanism` | 运动增益门控：`gate(motivation)` → 注入 `actuate` 决策（增益×强度） |
| M-7 | M5/M6 反证清单 | 反证记录（非交付物，设计依据） | **不封装**（L3/§2.2） | 设计依据→阶段二 M8 正面设计（铁律 C 缩放扫描） |

**参数唯一定稿源**：`neural_exploration/data/m7_innate_params.csv`（六机制全参 + 冻结锚；全部从 M3–M6 冻结 CSV 提取，不重训/校准）。

## 4. 等价性验证（P-A1：冻结锚 vs 模块输出）

**P-A1 等价性验证：21/21 断言全过 + 6/6 确定性逐位一致**（B1a 落盘 `reports/neuro/m7_equivalence_summary.json`，wall_s=0.014s；B2 节点 fresh compute 复验独立确认 21 项断言 + 6 项确定性 → PASS ✅）。

| 机制 | 冻结锚（M3–M6 报告数值） | 模块输出（B1a + B2 fresh compute 复验） | 判据 | 结果 |
|---|---|---|---|---|
| M-1 反射 | D_peak=0.352–0.369（M3 定稿带） | D_peak(1.0)=**0.3601**，back；I=0→0 none；I=2→0.4529 单调 | ∈带 且 >0.3 | ✅ |
| M-2 趋化 | CI(25s)=0.494 / CI(15s)=0.417 方向正 | CI(10s) 四朝向 east/north/west/south=1.0/1.0/**0.69**/0.69 全正；canonical θ=π→0.690 | 符号正 + 落容差窗 [0.25,0.75] | ✅ |
| M-3 CPG | 0.400 / 2.167Hz（M5 P3 冻结） | 0.400 / 2.167Hz | 落预注册带 [0.1,2]/[2,5] | ✅ |
| M-4 习惯化 | R²=0.787 / τ=2.0 / R_rest=0.411 | R²=**1.000**（纯指数）/ τ=2.0 / decay=0.492 / 后半均值 −0.104 < 0.5×前半 0.085 / R_rest=0.353 | R²≥0.5 或后半<0.5×前半 + 消融 + 恢复 | ✅ |
| M-5 联想 | Δw_train=+0.4325 / Δw_ext=−0.108 / η=0→0 | Δw_train=**+1.545** / Δw_ext=**−1.600** / η=0→0.000000 | Δw_train>0.1 + Δw_ext<0 + η=0→0 （**绝对值不作硬判据**——清单 §2.1，L9 幅度差如实记录） | ✅ |
| M-6 调质 | gate∈[0.3,1.2]；酪胺关→gate≡1 | gate(mot 0→1)=0.398→0.473 单调；酪胺关=0.995 | ∈[floor,1.2] + 单调 + 消融 sanity | ✅ |

**汇总**：等价性 21/21 断言 + 确定性 6/6 （repr 级逐位一致，B1a + B2 双节点一致）。

## 5. 数字大脑接入（P-A2）

**P-A2 判定：pass**（`InnateInterface` 四方法 + `SymbolicInterface` 加法注入，117 零回归 + 14 新增断言；B1b 落实，B2 复验）。

**接入语义（D3 层位）**：认知层（应用题文本→推理）在上，机制层（环境刺激→行为）在下——先天机制 = 数字大脑'感知/运动底座'；本里程碑只验证**接口接通 + 行为差异可测**，不做'神经→符号'完整桥（预注册 §0 #9：认知层推理链不被机制层替换）。

**新建/修改**：
- `digital_brain/src/interfaces/innate_interface.py`（新建）：`sense/actuate/adapt/gate` 四方法 + `calls` 调用日志 + `set_enabled` 消融开关（M6 消融 sanity 惯例）；
- `digital_brain/src/interfaces/symbolic_interface.py`（**加法式**修改 2 处）：可选 `innate` kw 参数 + `self.innate` 属性——solve 语义零修改；
- `digital_brain/tests/test_m7_interface.py`（新建，14 断言）：机制存在性/参数可调/注入前后行为差异（消融）/调用日志/零回归（有/无 innate 同题同答）。

**B2 复验**：pytest `test_m7_interface.py` = 14/14 通过（exit=0）。
**数字大脑全量**：`pytest digital_brain/tests` = **144 passed**（117 原基线零回归 + 7 机制冒烟 + 14 接口 + 6 场景）。

## 6. 应用题场景（P-A3）

**P-A3 判定：pass**（4 场景 ≥3 达标——机制层断言 + 认知层完整推理链 + 对照；B1b 落实，B2 复验）。

**预注册边界（清单 §0 #5）**：认知层问题只用 base_curriculum **已验证模板**（0-20 加法 srl_parse 求和链）；场景词汇经标准 `learn_word` 接口教学（大脑初始唯一能力 = 学词，推理能力未扩展）；场景叙事（厨房/香气/火炉/钟声）由**机制层断言**承载（感知/运动底座语义，D3 层位）。

| 场景 | 机制层断言（a） | 认知层完整链（b） | 对照（c） |
|---|---|---|---|
| S1 闻香走向厨房（趋化） | 浓度随接近递增 + 梯度指向厨房（东北 (7.5,7.5)）+ 不利朝向 θ=π 转向趋利 + CI=0.690 ∈ [0.25,0.75] | 4+2 苹果 → 6（srl_parse + call_algorithm 链完整） | innate=None 对照可解（机制层与认知层解耦） |
| S2 碰到火炉缩手+习惯化（反射+习惯化） | escape → back + D_peak>0.3；R(n) 严格递减 + 后半均值<0.5×前半 | 2+1 缩手 → 3 | 同上 |
| S3 钟声节奏（CPG） | 无食物 0.400Hz ∈ [0.1,2] + 有食物 2.167Hz ∈ [2,5] 双带切换 + 相位推进 | 5+3 走步 → 8 | 同上 |
| S4 饿了跑得更快（调质） | gate(0)→gate(1) 单调↑ ∈ [0.3,1.2] + 增益注入 escape 强度（hungry>calm） | 3+4 跑步 → 7 | 同上 |

**B2 复验**：pytest `test_m7_applications.py` = 6/6 通过（exit=0）。
**能力边界如实记录（L17）**：认知层 srl_parse 求和链对数字组合敏感（'2步，又走了5步'→None）——场景认知句从实测通过集合选取（S1=4+2、S2=2+1、S3=5+3、S4=3+4）。

## 7. Pass 对照表（P-A1/A2/A3/P5/P6）

| 项 | 判定 | 关键数值 | 备注 |
|---|---|---|---|
| P-A1 机制提取与封装 | **pass** | 6/6 机制封装；等价性 21/21；确定性 6/6 逐位一致 | 反射 D_peak 0.3601∈带；趋化 CI 0.690；CPG 双带；习惯化 R²=1.0；联想 Δw+1.545；调质单调（B1a + B2 fresh compute 双确认） |
| P-A2 数字大脑接入 | **pass** | InnateInterface 四方法 + SymbolicInterface 加法注入；14/14 断言 | 117 零回归 + 14 新增断言；solve 语义零修改 |
| P-A3 应用题场景 | **pass** | 4 场景（S1–S4）≥3 达标；6/6 断言 | 机制层断言 + 认知层完整链 + innate=None 对照 |
| P5 回归与报告 | **pass** | 数字大脑 144/144；神经仿真 68/68 | m7_report.md + m7_validation_summary.json + m7_innate_map.png |
| P6 交接处置 | **pass** | L1–L19 + 本节点 L20+ + 阶段二 M8 交接 | docs/m7_env_notes.md + 本报告 §8/§9 |

## 8. 踩坑记录（L1–L19 摘要 + 本节点 L20–L26）

**B1a/B1b 记录（L1–L19，详见 docs/m7_env_notes.md）摘要**：

- L1–L6：M6 冻结基线→M7 组装（组合不修改）；可迁移机制清单确认（D2）；M5/M6 反证最终状态（交接语义）；数字大脑现状与接口（旁线恢复前置）；环境依赖分叉（.venv-db 建立）；预算与确定性纪律

- L7–L13（B1a）：G1 门实测依赖三件套缺失→独立 venv；机制封装位置定稿（D3 裁决）；联想 Δw 幅度差（行为级抽象如实记录）；趋化 CI 策略（Braitenberg 限速转向）；调质基线语义核对（tyr_baseline 归一化浓度非 Hz）；习惯化拟合纯 stdlib 网格 lstsq；交付纪律

- L14–L19（B1b）：认知层场景句必须落在已验证模板内（未知词→DAG 构建失败）；名词性叙事前缀破坏 DAG 求和链；InnateInterface 不拒收未路由机制（M-5 联想保持可观察）；SRL 求和链对数字组合敏感（选定已验证句）；P-A2/P-A3 回归 144 全绿；neural_exploration 侧零改动（长时全量回归交 P5 报告节点）


**本节点（M7-B2）实测记录（L20–L26）**：
- **L20** — B2 复验数字大脑全量：`.venv-db/bin/python -m pytest digital_brain/tests` → **144 passed in 1.44s**（117 零回归 + 27 新增）——G1 基线稳定，无新坑
- **L21** — B2 复验神经仿真快子集（M0–M4 + multicomp，8 文件）：→ **41 passed in 268.5s**（test_smoke 4 + test_reflex_smoke 7 + test_synapse_smoke 6 + test_chemotaxis_smoke 9 + test_synapse_validation 5 + test_multicomp_* 10）——冒烟测试确定性重写自身 PNG（m3/m4_smoke.png 内容逐位不变，git status 核对零改动）
- **L22** — B2 复验神经仿真长时（M5/M6 网络测试，L19 交接给 P5 节点）：test_worm_smoke（302 神经元）**10/10 in 22.7s** + test_m6_neuromod **9/9 in 35.8s** + test_m6_learning_smoke（全协议联想学习）**8/8 in 1170.5s（19m30s）**——**逐文件顺序跑**（brian2 缓存锁纪律 L9#6：每文件独立进程，单后台任务顺序执行不并发；合计神经仿真全量 68/68 与 M6 冻结基线零回归）
- **L23** — run_m7_validation.py 的 P-A2/P-A3 pytest 子进程必须用 `.venv-db/bin/python` 启动（networkx/pydantic 依赖——G1 L5 环境分叉）；`.venv-neuro` 下运行会自动探测依赖缺失并记 skip（不报错、不伪造）
- **L24** — 回归运行缓存纪律：pytest 一律 `-p no:cacheprovider` + `PYTHONDONTWRITEBYTECODE=1`——不落 .pytest_cache/__pycache__（只读纪律；脚本只写自身交付物 m7_validation_summary.json）
- **L25** — P-A1 复验方式：B2 不重写 B1a 的 m7_equivalence_summary.json（已落盘交付物），改为 **fresh compute**（digital_brain.src.innate 直连重跑 21 探针）与 B1a 数值交叉核对——双节点独立确认 21/21 + 6/6 逐位一致
- **L26** — 交付纪律：验证脚本只读 + 分块跑长时回归（快子集前台 / M5·M6 后台顺序）；**未 git commit**（M0–M6 惯例，交规划节点验收后提交）

## 9. 阶段二 M8 交接（果蝇幼虫全脑）

**M8 = 果蝇幼虫全脑（阶段二）**：3,016 神经元 / ~55 万突触（Winding 2023 连接组）；行为预测 + 活动预测 + 扰动预测三通道。

**M7 交付 → M8 设计基础**：
1. **M-1..M-6 迁移机制清单**（本报告 §3 表）是 M8 的机制库：反射（痛觉逃避）/趋化（AWC 通路嗅觉）/CPG（节律）/习惯化（STP/酪胺）/联想（三因子 elig 迹 + 调质门控）/调质（C_da/C_5ht/C_tyr → 增益门控）——行为级模块可直接在 M8 降阶模型上复验；
2. **降阶模型方案（M5 缩放定律）**：M5 的 scale 缩放曲线（302→20 神经元行为等价）为 M8 全脑降阶提供方法论——3,016 神经元全脑需先跑通行为通道降阶（活动/扰动通道可渐进升阶）；
3. **M-7 夹带双稳态反证（M5/M6）**：调质门控只能整体开/关夹带（14Hz→2Hz→静默），无『低活动+行为 bout』稳定中间态——**为 M8 降阶设计提供依据**：真实幼虫双状态需要（a）命令层去同步（AVA/AVB 真实递质抑制边）或（b）运动层与命令层分离驱动或（c）异质权重/传导——阶段二**铁律 C 缩放扫描**需正面设计此项（缩放扫描必须扫到双状态结构，不能只验证活动量）；
4. **三通道验证目标建议**：行为通道（趋化 CI + 逃避 D_peak + 节律带）、活动通道（静息低活动 + 行为 bout 双状态——M-7 正面设计）、扰动预测通道（消融/调质干预 → 行为/活动预测，M6 消融 sanity 惯例扩展）。

**复现入口（M7，阶段一收官）**：
- 数字大脑全量：`.venv-db/bin/python -m pytest digital_brain/tests`（144）
- 神经仿真全量：`.venv-neuro/bin/python -m pytest neural_exploration/tests`（68，M5/M6 长时逐文件跑）
- 机制等价性：`.venv-neuro/bin/python -m neural_exploration.tools.validate_m7_equivalence`（21/21）
- 验证汇总：`.venv-db/bin/python -m neural_exploration.tools.run_m7_validation`
- 报告：`.venv-neuro/bin/python -m neural_exploration.tools.gen_m7_report` → docs/m7_report.md
- 映射图：`.venv-neuro/bin/python -m neural_exploration.tools.gen_m7_innate_map` → reports/neuro/m7_innate_map.png
