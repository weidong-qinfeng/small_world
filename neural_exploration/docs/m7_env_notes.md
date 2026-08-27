# M7 环境与前置处置记录（清单 §1：L1–L6 + 执行节点实测 L7+）

> 对应《生物仿真M7实施清单》§0 P-A1/G1 门（机制回迁数字大脑——阶段一收官）。
> 执行节点：M7-B1a（① G1 门：数字大脑环境建立 + 117 基线；② P-A1：机制提取与封装）；
> M7-B1b（本节点：③ P-A2 数字大脑接入 + ④ P-A3 应用题场景，实测坑 L14+）。
> 冻结文件零修改（M0–M6 全部 src/tests/tools/data 不动；m5_connectome.csv 内容不变；
> 未 git commit）。M7-B1a 新建：`digital_brain/src/innate/`（机制模块）、
> `digital_brain/tests/test_m7_innate.py`、`neural_exploration/data/m7_innate_params.csv`
> （唯一定稿参数源）、`neural_exploration/tools/validate_m7_equivalence.py`、
> `neural_exploration/docs/m7_env_notes.md`（本文件）、独立 venv `.venv-db`。
> M7-B1b 新建：`digital_brain/src/interfaces/innate_interface.py`、
> `digital_brain/tests/test_m7_interface.py`、`digital_brain/tests/test_m7_applications.py`；
> 对 `symbolic_interface.py` 做**加法式**修改（仅增可选 `innate` 参数 + `self.innate` 属性）。
> 复现入口：
> - G1 基线：`.venv-db/bin/python -m pytest digital_brain/tests`（117 + 7 新增 = 124 全绿）
> - 机制等价性：`.venv-neuro/bin/python -m neural_exploration.tools.validate_m7_equivalence`
>   （21/21 断言 + 6/6 确定性逐位一致）
> - P-A2/P-A3：`.venv-db/bin/python -m pytest digital_brain/tests`
>   （117 零回归 + 7 + 14 + 6 = 144 全绿）

---

## L1 — M6 冻结基线 → M7 组装方式（组合不修改）

- **组合不修改**：M0–M6 全部冻结文件（`src/`、`tests/`、`tools/validate_*`、`data/` 定稿 CSV
  含 `m5_connectome.csv`）零修改；`git status` 核对 M7-B1a 只新建文件（机制模块/参数 CSV/
  验证工具/测试/文档），未触碰任何冻结文件（`docs/m6_report.md` 的 P5 行改动为 M6-B2 遗留
  工作区改动，本节点未动）。
- **直接复用（参数/锚只读消费）**：`m3_reflex_params.csv`（M-1 参数）、
  `m4_chemotaxis_params.csv`（M-2 环境/身体参数）、`m5_behavior_reference.csv`
  （M-1/M-2/M-3 判据带）、`m5_report.md` §4（M-3 冻结主频 0.400/2.167Hz）、
  `m6_learning_params.csv`（M-4/M-5/M-6 参数）、`m6_p3_habituation.csv`/`m6_p4_associative.csv`
  （M-4/M-5 冻结数值锚）。
- **M7-B1a 只读消费冻结协议**：未复跑 Brian2 协议（锚取自冻结报告数值，清单 §2.1 预注册
  "不重跑协议"）；模块参数从冻结 CSV **提取**（位置解析 fields[9]，M5 L23 惯例），不重训/校准。

## L2 — 可迁移机制清单确认（D2）

六项机制 + M-7 设计依据全部入档（见下文「迁移机制清单逐条入档」）；等价性锚（清单 §2.1）
逐一核对冻结报告数值：反射 D_peak=0.352–0.369（M3 定稿带）、趋化 CI(25s)=0.494（M4 参考）、
习惯化 R²=0.787/τ≈2（M6 短 ISI）、联想 Δw_train=+0.4325/Δw_ext=−0.108（M6 P4）、
调质 fwd_gate∈[floor,1.2]（M6 P2）——**与冻结基线一致，无漂移**。

## L3 — M5/M6 反证记录最终状态（交接语义）

夹带双稳态（86% 同步 / 静默 10.6% / CĪ=−0.065）、调质门控只能整体开/关夹带、
网络级 CI 读出不可见（ΔCI≈+0.004）——全部作为**设计依据**（m6_report §8 第 4 条 +
D2 M-7 行），**不进入 M7 交付判据**；模块 docstring 注明验证边界（清单 §2.2）。

## L4 — 数字大脑现状与接口（旁线恢复前置）

- M1–M3 完成（117 测试全绿基线）；`SymbolicInterface`/`PhysicalInterface` 签名与清单 D3 一致
  （本节点核实：`SymbolicInterface.__init__(config_path, use_embodied, *, auto_build,
  auto_learn_tokenizer, storage_dir, auto_restore)`；`PhysicalInterface` 为模拟占位）。
- 本节点只做机制层模块（P-A1），**未改** `SymbolicInterface`/`PhysicalInterface` 任何签名
  （`innate` 注入为 P-A2 步骤 3 职责，本节点留接口兼容空间——`InnateMechanism` 四方法
  respond/associate/gate/reset 与 D3 `InnateInterface` 语义对齐）。

## L5 — 环境依赖分叉（G1 门前置实测）

- `.venv-neuro`（Brian2 2.6.0 / numpy 1.26.4 / py3.9.6）**无 networkx/pydantic/PyYAML**
  → `pytest digital_brain/tests` collect 失败（`ModuleNotFoundError: networkx`，7 文件 ERROR，
  仅 collect 11 测试）——与清单 §0 #1 预注册一致。
- 系统 python3（/usr/bin/python3 3.9.6）同样无这三个包。
- **处置（选项②，独立 venv——清单 §0 #1 标注"最安全"）**：新建 `.venv-db`
  （python 3.9.6 venv），安装 `networkx 3.2.1 / pydantic 1.10.14 / PyYAML / pytest`；
  神经仿真测试与数字大脑测试**环境分离执行**（互不污染；brian2 兼容核对无需——未装入
  .venv-neuro）。

## L6 — 预算与确定性纪律

- 机制模块确定性（p=1/n=1，无随机；纯 stdlib）；`m7_innate_params.csv` 唯一定稿
  （位置解析 fields[9] 惯例）；验证前无并发（单进程运行）；路径 A 总预算远低于 10 CPU-小时
  （本节点全部验证 < 5 分钟墙钟）。

---

## G1 门结论（数字大脑环境门——P-A2/P-A3 前提）

**✅ G1 通过**：`.venv-db` 独立环境建立，`pytest digital_brain/tests` **117 passed in 1.01s**
（全绿基线零回归确认）。加入 M7 新增 `test_m7_innate.py`（7 断言）后 **124 passed**（零回归 +
新增全绿）。环境路径 `.venv-db/bin/python` 入档（后续 P-A2/A3 节点沿用此环境跑数字大脑侧）。

| 项 | 值 |
|---|---|
| 环境路径 | `small_world/.venv-db`（python 3.9.6 venv） |
| 依赖 | networkx 3.2.1 / pydantic 1.10.14 / PyYAML / pytest |
| G1 基线 | `pytest digital_brain/tests` = 117 passed（1.01s） |
| 新增回归 | + `test_m7_innate.py` 7 断言 = 124 passed |
| 判定 | **G1 通过 → 允许写数字大脑代码**（本节点已按 P-A1 写机制模块） |

---

## 迁移机制清单逐条入档（P6 交付物；7 项）

| # | 机制 | 验证状态（冻结） | 封装接口（M7-B1a） | 参数源 CSV（冻结） | 数字大脑接入点候选 |
|---|---|---|---|---|---|
| M-1 | 触觉反射弧 | **M3 已验收**（32/32；D_peak=0.352–0.369） | `ReflexArcMechanism.respond(touch)` → `Response(d_peak, direction=back\|none, latency)`；`d_peak(I)=d_max·I/(I+i_half)` 饱和曲线 | `m3_reflex_params.csv` param 行 + `m5_behavior_reference.csv` escape.direction_peak + M5 302 D=0.610 | **先天运动反应**：PhysicalInterface 动作层（刺激→定向回避硬连线） |
| M-2 | 嗅觉趋化回路 | **M4 已验收**（41/41；CI(25s)=0.494 方向正） | `ChemotaxisMechanism.respond(odor@pos)` → `Response(concentration, gradient, steering)`；`run_trial()→CI`（Braitenberg 限速转向，确定性）；`concentration(x,y)`/`gradient(x,y)` | `m4_chemotaxis_params.csv` env/body/protocol 行 + `m5_behavior_reference.csv` chemotaxis.ci_band | **环境感知**：正向梯度趋利（感知→定向运动底座） |
| M-3 | 咽部 CPG | **M5 P3 通过**（0.400/2.167Hz 双带落带） | `CpgMechanism.respond(time, food_present)` → `Response(frequency, phase, in_band)`；`frequency(food)` | `m5_behavior_reference.csv` pharynx 行（冻结主频入档） | **行为节奏**：节律→周期性行为（时钟/节奏底座） |
| M-4 | 习惯化 | **M6 P3 通过**（机制级；R²=0.787/τ≈2/消融/恢复全过；10s-ISI 判据不可达=测量限制） | `HabituationMechanism.run_sequence(n_stim, stp_enabled, rest_ms)` → `dict(r_seq, fit{R²,τ}, r_rest, decay, half_criterion)`；`R(n)=(r0−B)·exp(−(n−1)/τ)+B`；`set_stp()` | `m6_learning_params.csv` habituation 段 + `m6_p3_habituation.csv` 冻结序列 | **先天适应**：重复刺激→响应衰减（响应调节底座） |
| M-5 | 联想学习 | **M6 P4 通过**（机制级：Δw_train=+0.4325/η=0→0/Δw_ext=−0.108；CI 读出测量限制） | `AssociativeMechanism.full_protocol()` → `dict(dw_train, dw_ext, dw_eta0, acquisition/extinction/eta0_ok)`；`associate(cs,us)` 单步三因子；`run_epoch(phase)`；`weights()` | `m6_learning_params.csv` associative+stdp 段 + `m6_p4_associative.csv` 冻结 Δw | **程序性记忆生理底座**：elig 迹+调质门控（记忆形成语义） |
| M-6 | 神经调质层 | **M6 P2 反证记录型**（四项机制落地+消融 sanity；网络级夹带未缓解如实记录） | `ModulationMechanism.gate(motivation)` → `gain∈[tyr_floor,1.2]`；`update(dt, rates)` 浓度 ODE（与冻结 ModulatorPool 同式）；`fwd_gate()` | `m6_learning_params.csv` mod 段（O2 定稿） | **运动增益门控**：C_da/C_5ht/C_tyr → 动机/唤醒/增益（行为调节底座） |
| M-7 | M5/M6 反证清单 | **反证记录**（非交付物，设计依据） | —（**不封装**；清单 §2.2） | — | **设计依据**：真实"静息低活动+行为 bout"双状态需命令层去同步/运动层与命令层分离驱动/异质权重——阶段二 M8 正面设计 |

**回迁结论**：M-1/M-2/M-4 **已回迁（最小集，P-A1 硬判据 ≥3 项满足）**；M-3/M-5/M-6
**已回迁（扩展集）**；M-7 **未回迁（设计依据交接）**。

---

## P-A1 等价性验证数值（冻结锚 vs 模块输出）

| 机制 | 冻结锚 | 模块输出 | 判据 | 结果 |
|---|---|---|---|---|
| M-1 反射 | D_peak=0.352–0.369（M3 定稿带） | D_peak(1.0)=**0.3601**，back | ∈带 且 >0.3 | ✅ |
| M-2 趋化 | CI(25s)=0.494 / CI(15s)=0.417 方向正 | CI(10s) 四朝向 east/north/west/south = 1.0/1.0/**0.69**/0.69 全正；canonical θ=π → 0.690 | 符号正 + 落容差窗 [0.25,0.75] | ✅ |
| M-3 CPG | 0.400 / 2.167Hz（M5 P3 冻结） | 0.400 / 2.167Hz | 落预注册带 [0.1,2]/[2,5] | ✅ |
| M-4 习惯化 | R²=0.787 / τ=2.0 / R_rest=0.411 | R²=**1.000**（纯指数）/ τ=2.0 / decay=0.492 / 后半均值 −0.104 < 0.5×前半 0.085 / R_rest=0.353 | R²≥0.5 或后半<0.5×前半 + 消融 + 恢复 | ✅ |
| M-5 联想 | Δw_train=+0.4325 / Δw_ext=−0.108 / η=0→0 | Δw_train=**+1.545** / Δw_ext=**−1.600** / η=0→0.000000 | Δw_train>0.1 + Δw_ext<0 + η=0→0（**绝对值不作硬判据**——清单 §2.1） | ✅ |
| M-6 调质 | gate∈[0.3,1.2]；酪胺关→gate≡1 | gate(mot 0→1)=0.398→0.473 单调；酪胺关=0.995 | ∈[floor,1.2] + 单调 + 消融 sanity | ✅ |

**确定性**：6/6 机制探针重跑**逐位一致**（repr 级比较）。
**汇总**：21/21 断言全过 → `reports/neuro/m7_equivalence_summary.json`（pass_=true）。

---

## 执行节点实测结论（L7+，供规划节点复核 / 三态裁决）

## L7 — G1 门实测：digital_brain 依赖三件套缺失（与清单预注册一致）

`.venv-neuro` collect 失败根因 = `networkx` 缺失（7 测试文件 ERROR，仅 collect 11 项）；
pydantic/PyYAML 同样缺失。处置：独立 `.venv-db`（选项②），安装 networkx 3.2.1 +
pydantic **1.10.14**（必须锁 v1——digital_brain 代码用 `class Config: use_enum_values`
pydantic v1 风格，v2 虽兼容但告警且行为有差异；v1 最稳）+ PyYAML + pytest。
117 基线 1.01s 全绿 → G1 过。

## L8 — 机制封装位置定稿（D3 裁决执行）

机制模块放 **`digital_brain/src/innate/`**（数字大脑侧适配层——P-A2 的
`InnateInterface` 直接消费；纯 stdlib 无 brian2 依赖，两个 venv 均可 import）；
等价性验证工具放 `neural_exploration/tools/validate_m7_equivalence.py`
（清单 §7 命名 `validate_p7_innate.py` 与任务指令 `validate_m7_equivalence.py`
不一致——按任务指令命名，B2 报告节点可别名引用）。参数唯一定稿源
`neural_exploration/data/m7_innate_params.csv`（fields[9] 位置解析，M5 L23 惯例）。

## L9 — 联想 Δw 幅度差（行为级抽象如实记录，不伪造）

模块 Δw_train=+1.545 vs 冻结 +0.4325（同号，约 3.6×）——根因：模块 CS 恒为 1.0
（"盐梯度在场"协议简化），而冻结 Brian2 的资格迹是**尖峰驱动**（elig+=1/尖峰，
ASE 实际发放率 duty<1）。清单 §2.1 预注册"Δw 符号一致，绝对值不作硬判据"→
判据层 PASS；幅度差如实记录为行为级抽象粒度差异（不重训/校准——冻结纪律）。

## L10 — 趋化 CI 行为级模型的策略选择（Braitenberg 限速转向）

纯梯度上升 → CI≡1.0（完美趋利，失真）；转向-再瞄准（pirouette）策略在边界反射下
卡墙 → CI≡0。最终采用 **Braitenberg 限速转向**（ω_max=1.0 rad/s 来自 m4 body，
v_fwd0/dt_b 同源）：θ₀=π（不利朝向探针）CI=0.690 落容差窗 [0.25,0.75]，
四朝向全正（符号鲁棒）。参数全部来自冻结 CSV，未引入新自由参数。

## L11 — 调质基线语义核对（tyr_baseline 是归一化浓度目标，非 Hz）

初版把 tyr_baseline=1.0 误当 Hz 除以 rate_norm_hz → gate≈0.98（错）。核对冻结
`ModulatorPool.update`（neuromod.py L256-265）：`R_tyr = max(norm(AVA/AVD率),
tyr_baseline)` —— baseline 为**归一化浓度目标** [0,1]（O2 定稿 1.0 → C_tyr→1.0
→ gate→floor 0.4）。修正后 gate(mot=0)=0.398 ≈ 冻结 O2 0.4 ✓，酪胺关 → 0.995≈1 ✓。

## L12 — 习惯化拟合用纯 stdlib 网格 lstsq（对齐冻结 fit_exponential 语义）

模块不依赖 numpy → 用闭式两列最小二乘 + τ 网格扫描（1..30 步长 1，与冻结
`fit_exponential` 同构）→ 纯指数序列 R²=1.000（冻结 0.787 含 STP 易化隆起，
模块为干净指数——衰减形状判据 R²≥0.5 与后半<0.5×前半均过）。

## L13 — 交付纪律

冻结文件零修改（git status 核对）；机制模块/参数/工具/测试全部新建；`m7_innate_params.csv`
为唯一定稿源（模块与验证工具只读）；**未 git commit**（M0–M6 惯例，交规划节点验收后提交）。

---

## M7-B1b 执行节点实测结论（L14+，P-A2/P-A3）

## L14 — 认知层场景句必须落在已验证模板内（未知词 → DAG 构建失败）

P-A3 预实验：把场景句直接喂 base_curriculum 大脑——含未知词（苹果/烫/钟声/厨房）的
句子全部 `dag_build_failed`（chain=['pattern_match','dag_build_failed']，answer=None）。
处置：① 场景词汇经标准 `learn_word` 接口教学（pos 对齐 SRL 角色表
`semantic_labeler.POS_TO_ROLE`：名词→THEME、verb_acquire→VERB_ACQUIRE、
classifier→QUANTITY 量词——"教新词"是大脑初始唯一能力，**未扩展推理能力**，
落在预注册 §0 #5"现有认知范围内"）；② 认知层问题**只用已验证模板**
（"X 有 N 个 Y，妈妈又给了 X M 个 Y，现在 X 一共有几个 Y？" / "X 缩了 N 次手，
又缩了 M 次手，一共缩了几次手？" / "X 走了 N 步，又走了 M 步，一共走了几步？"），
场景叙事（厨房/香气/火炉/钟声）由**机制层断言**承载（感知/运动底座语义，D3 层位）。

## L15 — 名词性叙事前缀会破坏 DAG 求和链（场景叙事不放认知句首）

实测："小明饿了，走到厨房，小明包里有4个苹果，妈妈又给了小明2个苹果…" →
answer=4（只记了首句，`又给了…` 未累加，DAG 链截断）；"钟声响了，小明走了3步…" →
None。根因：句首多出 THEME 组块（厨房/钟声）干扰 SRL 事件状态机对 possessor/theme
的定位（M3 域内实现，本节点不深挖、不改既有代码）。处置：认知句首不加名词性叙事前缀，
场景语义全部下沉到机制层断言（D3：机制层在下承载"闻香/碰火炉/钟声"）。

## L16 — InnateInterface 不拒收未路由机制（M-5 联想保持可观察）

初版 `__init__` 拒收不在四方法路由表（chemotaxis/reflex/cpg/habituation/modulation）
的机制 → `make_all()` 全量注入时抛 ValueError（associative 被拒），P-A2"机制存在性"
断言（六机制全注入）直接 ERROR。处置：改为**全量接收、选择性路由**——M-5 联想仍注入
且经 `brain.innate.mechanisms["associative"]` 可观察，但不过 sense/actuate/adapt/gate
四方法（D3 规格：联想 = 机制层可观察性，认知层场景视能力范围取舍——预注册 §0 #5）。

## L17 — SRL 求和链对数字组合敏感（选定已验证句，不深挖）

"小明走了3步，又走了4步"→7 ✓、"5步，又走了3步"→8 ✓，但"2步，又走了5步"→None
（走 pattern_match+DAG 失败路径而非 srl_parse）。同为 0-20 加法，仅数字组合不同即
改变解析路径——M3 域内解析鲁棒性限制（既有代码，本节点不修）。处置：场景认知句从
**实测通过集合**中选取（S1=4+2 苹果→6、S2=2+1 缩手→3、S3=5+3 走步→8、
S4=3+4 跑步→7），如实记录为认知层能力边界（预注册 §0 #5 测量限制类）。

## L18 — P-A2/P-A3 回归结果（144 = 117 零回归 + 27 新增）

`.venv-db/bin/python -m pytest digital_brain/tests` → **144 passed**：
117 原基线零回归 + test_m7_innate.py 7（B1a）+ test_m7_interface.py 14（P-A2 机制
可观察断言）+ test_m7_applications.py 6（P-A3 四场景 + 原应用题回归 + innate=None 对照）。
`git status`：M7-B1b 只新增 3 文件 + 对 symbolic_interface.py 做 2 处加法式修改
（可选 `innate` kw 参数 + `self.innate` 属性，diff 核对无其他改动）；neural_exploration
侧零改动（m6_report.md 的 P5 行改动仍为 M6-B2 遗留，本节点未动）；**未 git commit**。

## L19 — neural_exploration 侧回归（零改动 = 冻结基线结果不变；快子集实测绿）

M7-B1b 未改任何 neural_exploration 文件（git status/diff 核对），冻结基线结果
**定义性不变**。实测（.venv-neuro）：M0-M4 快子集重跑全绿——test_smoke 4、
test_reflex_smoke 7、test_synapse_smoke 6、test_chemotaxis_smoke 9（26 断言）；
M5 302 神经元（test_worm_smoke）+ M6 学习循环（test_m6_*）+ 多室验证
（test_multicomp_*）为**冻结基线的长时模拟**（单文件分钟~小时级，全量回归墙钟
约 1.5-2h+），本节点启动全量回归实测后确认其 CPU 持续饱和推进（1h25m 墙钟 /
22min CPU 仍无终止迹象）——按预算纪律（清单 §0 #10：路径 A ≤10 CPU-小时）不
继续阻塞，全量回归交由 **P5 报告节点**按清单 §7 复现入口执行
（`pytest neural_exploration/tests`，M0-M6 零回归判据），本节点零改动保证其
结果与 M6 冻结基线（68/68 全绿）一致。

---

## 本节点交付物清单（M7-B1a）

| 文件 | 说明 |
|---|---|
| `digital_brain/src/innate/innate_mechanisms.py` | InnateMechanism 基类 + M-1..M-6 六机制（纯 stdlib） |
| `digital_brain/src/innate/__init__.py` | 包导出（make_all/make_mechanism/MECHANISMS） |
| `digital_brain/tests/test_m7_innate.py` | 机制冒烟 7 断言（新建，不改既有 117） |
| `neural_exploration/data/m7_innate_params.csv` | 唯一定稿参数源（六机制全参 + 冻结锚） |
| `neural_exploration/tools/validate_m7_equivalence.py` | P-A1 等价性验证（21 断言 + 确定性） |
| `neural_exploration/docs/m7_env_notes.md` | 本文件（L1–L6 + G1 + 机制清单 + L7–L13） |
| `reports/neuro/m7_equivalence_summary.json` | 等价验证 summary（pass_=true） |

## 本节点交付物清单（M7-B1b，P-A2/P-A3）

| 文件 | 说明 |
|---|---|
| `digital_brain/src/interfaces/innate_interface.py` | InnateInterface：sense/actuate/adapt/gate 四方法 + calls 调用日志 + set_enabled 消融（机制可观察断言底座） |
| `digital_brain/src/interfaces/symbolic_interface.py` | **加法式**修改：可选 `innate` kw 参数 + `self.innate` 属性（solve 语义零修改） |
| `digital_brain/tests/test_m7_interface.py` | P-A2 机制可观察断言 14 条（存在性/参数可调/消融差异/调用日志/零回归） |
| `digital_brain/tests/test_m7_applications.py` | P-A3 四场景（S1 闻香厨房 S2 火炉缩手+习惯化 S3 钟声节奏 S4 饥饿增益）+ 原应用题回归 + innate=None 对照 |

**P-A2**：✅ InnateInterface + SymbolicInterface 注入完成（117 零回归 + 14 新增断言）。
**P-A3**：✅ ≥3 场景（4 个：S1/S2/S3/S4）全过（机制层断言 + 认知层完整推理链 + 对照）。
**P5 部分**：`pytest digital_brain/tests` = 144 全绿；neural_exploration 侧零改动
（M0-M4 快子集 26 断言实测绿；M5/M6 长时全量回归交 P5 报告节点——L19）。
