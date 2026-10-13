# M11 · L3 正式跑交接（自足入口）

> **用途**：L3 正式跑跨会话（墙钟约 68 h，因本机占空比 ≈5%）。任何新会话读本文件即可接管。
> **上游裁决**：`docs/m11_acceptance_master.md` §3di（R118）、§3dj（R119）；事前修订 `data/m11_prereg_amend_r118_l3_grid_v2.json`。

## 一、当前状态（启动于 2026-10-13T23:29:00Z / 本地 07:29:00）

| 项 | 值 |
|---|---|
| 作业 | `caffeinate -i /tmp/l3_run.sh > /tmp/ma_l3.log 2>&1 &`（**托管后台作业**）|
| 计划（冻结）| `data/m11_p5_plan_l3.frozen_for_run.json`，sha256 **`4aab49b00177a22e21f65b884bd29d95a8e7333e76cb109ae8fcd8bb9bd94615`** |
| 生产者 | `tools/run_m11_p5_arms.py` sha256 **`633a4c458e9c1dd58e5d6c4838a205819a9424fbd49df9da8827c8b7bd42285b`**（894 行）|
| 输出目录 | **`neural_exploration/data/m11_p5_l3_runs/`**（`--out-dir` 已用**绝对路径**，见 R119 §②）|
| 臂数 | **30**（5 格 × 2 角色 × 3 独立实现），`T_ms=10000`、`settle_ms=500` ⇒ **步数 210,000** |
| 云/规模 | **不降规模、不用云 GPU、本机 MPS**（用户明令）|

### 查进度（三条命令）
```bash
cd /Users/weidong/ai/small_world/neural_exploration
ls data/m11_p5_l3_runs/*.done 2>/dev/null | wc -l      # 完成臂数（期望最终 30）
ls data/m11_p5_l3_runs/*.fail 2>/dev/null | wc -l      # 失败臂数（期望 0）
grep -E "DONE|FAIL|SKIP|完成|ALL_DONE" /tmp/ma_l3.log | tail -20
```
**判断长作业进度必须看"累计 CPU 时间 / 墙钟"之比，不能看 `ps` 的 `%CPU`**（纪律 L46，见 `docs/SESSION_CONTEXT.md`）；
本机实测占空比 ≈5%（Chrome ~145% + `node`/dsh web ~84% 争用）⇒ **把 GUI 浏览器标签关掉/最小化是唯一有效加速手段（约 10×）**。

### 实测占空比与墙钟估计（启动后 8 分钟实测，**以此为准**）
子进程 `ps -o etime,time` = **ELAPSED 8:19 / 累计 CPU 1:34 ⇒ 占空比 ≈18.8%**（好于 R112 时测到的 5.4%）。
按 `~1.5 ms/step × 210,000 步 = 315 s` 纯计算 ⇒ **单臂墙钟 ≈28 min ⇒ 30 臂 ≈14 h**（若占空比掉到 5% 则约 53 h）。
**⇒ 判断作业是否健康：看 `累计 CPU / 墙钟` 之比是否稳定在 15–20%；若 ≪1 先怀疑调度饥饿，不要 kill**（纪律 L46）。
**⇒ 关掉/最小化 GUI 浏览器标签可把占空比推向 100%（约 5×）。**

### 续跑（幂等、resume-safe）
驱动器对已有 `.done` 的臂打印 `[SKIP]` 并复用其 JSON ⇒ **直接重跑同一命令即可续跑**（不要 `--force`）：
```bash
cd /Users/weidong/ai/small_world/neural_exploration
L3_PLAN_SHA_WANT=4aab49b00177a22e21f65b884bd29d95a8e7333e76cb109ae8fcd8bb9bd94615
[ "$(shasum -a 256 data/m11_p5_plan_l3.json | cut -d' ' -f1)" = "$L3_PLAN_SHA_WANT" ] || { echo "计划已被改动，先查！"; }
caffeinate -i /tmp/l3_run.sh > /tmp/ma_l3.log 2>&1 &
```
**⚠️ 两条硬约束**：①**MPS 严格串行** —— 跑之前先 `pgrep -f run_m11_p5_arms` 确认空闲；
②**盘上不得有第二个 MPS 作业**（并发实测 `ms/step` 2.4→14.1，**6× 退化**）。

## 二、网格与运行顺序

| 序 | 格 | `bias_scale` | `bias_log_sigma` | 说明 |
|---|---|---|---|---|
| 1 | **C2** | 1.30 | **0.10** | ★**决定性格**（探针暂定三判据同落带）|
| 2 | A0 | 1.00 | 0.30 | 锚点 = 既有 L0/L1 同配置点（确定性核对，标 `h_determinism_not_evidence`）|
| 3 | C1 | 1.15 | 0.10 | σ=0.10 偏置扫描 |
| 4 | C3 | 1.45 | 0.10 | 极端档（爆炸风险 ⇒ 居中后）|
| 5 | C0 | 1.00 | 0.10 | 已知不可判格（探针实测零驱动=0 ⇒ (c) 预期 `unavailable`）⇒ 最后 |

每格内 **`baseline` 先、`zero_drive` 后**（R 需要分子分母）。**重排只改臂顺序，不改任何档位/实现/协议字段**（R119 §①）。

## 三、判定（跑完后照此做）

### 3.1 评估器（**命令已核对 CLI，2026-10-14 修正**）
🔴 **先弄清机制**：`tools/validate_p11_resting.py` **没有 `--runs` 选项** —— 它吃的是 `--protocol-json`，而**臂清单（含每臂 `pop_trace`/`counts` 路径）就在那个协议 JSON 里**。
**协议 JSON 由生产者在计划跑完后自动写出**（`tools/run_m11_p5_arms.py` **L507–536**：`path = os.path.join(DATA_DIR, 「m11_p5_protocol_%s.json」 % plan_id)`，并打印 `protocol -> …`）。
本计划 `plan_id = p5_L3_sigma_cliff_bias_scan` ⇒ 协议文件将是：
**`neural_exploration/data/m11_p5_protocol_p5_L3_sigma_cliff_bias_scan.json`**（`p5_` 会重复一次，这是代码决定的，不是笔误）。
```bash
cd /Users/weidong/ai/small_world/neural_exploration
# ① 先确认协议文件已生成（跑完才会出现；若没有 ⇒ 作业还没跑完）
ls -la data/m11_p5_protocol_p5_L3_sigma_cliff_bias_scan.json
# ② 用评估器侧解释器（.venv-m11：只有 numpy+h5py，无 torch；R34）
/Users/weidong/ai/small_world/.venv-m11/bin/python tools/validate_p11_resting.py \
    --protocol-json data/m11_p5_protocol_p5_L3_sigma_cliff_bias_scan.json \
    --targets-csv data/m11_p5_targets.csv \
    --out-csv data/m11_p5_l3_eval.csv --out-json data/m11_p5_l3_eval.json
```
**评估器 CLI 的完整选项**（实测 `--help`，供按需取用）：
`--protocol-json`｜`--pop-trace`/`--counts`/`--dt-ms`/`--T-ms`/`--settle-ms`/`--n-neuron`/`--seed`/`--role`/`--mechanism`/`--config-id`/`--protocol-id`（**单臂模式**）｜`--targets-csv`｜`--out-csv`/`--out-json`｜`--n-boot`｜`--check-targets`（判据表 CSV⟺JSON 自检后退出）｜`--smoke-emit-counts`/`--smoke-scale`｜`--quiet`｜`--save-json-dir`。
⚠️ **本工具对『杠杆扫描计划』有结构性错配**（R108 §⑤）：它按『每 config 一个值』设计，因此 `(c)` 的 `invalid`/`unavailable`、`(d)`/`(h)` 的 `unavailable` 有可能是**计划结构造成**的，**不一定是科学结论** ⇒ **判定必须以 `tools/report_m11_l3.py`（四条报告量，含 `c_judgeable` 与 `triviality_flag`）为主、评估器为辅**。

### 3.2 三条判据与带（**逐字不变**）
| 判据 | 口径 | 带 | 来源 |
|---|---|---|---|
| **(a)** | 基准臂**逐神经元中位率**（Hz）| `[0.5, 2.0]` | 判据表 `rate_median_hz_lo/hi`（literature）|
| **(c)** | `R = pooled_zero_drive / pooled_baseline` | `≤ 0.10`，**且 `pooled_zero_drive ≥ 300`** | 判据表 `selfsustain_zero_drive_min_spikes=300` |
| **(e)** | **基准臂** `i_inh/i_exc` 积分比 | `[0.5, 3.0]` | 判据表 `ei_ratio_lo/hi` |

### 3.3 🎯 **(c) 是"窗口"不是"极限"（R115 §②，最容易误读的一条）**
`pooled_zero_drive < 300` ⇒ (c) 判 **`unavailable`（不得判 pass 也不得判 fail）**，**不是 pass**。
⇒ **报 (c) 时必须同时给 `pooled_zero_drive` 的绝对值**，不能只给 `R`。
⇒ 目标窗口 = **`pooled_zero_drive ∈ [300, 0.10 × pooled_baseline]`**。

### 3.4 反平凡性条款（R118 §④4）
若 (c) 通过但**基准臂 `active_frac` 同时塌陷**（≈0.27 量级）⇒ 判"**用把网络弄死换取 R→0**" ⇒ **该点不得记为工作点命中**。
**判别依据**：靠"网络死"通过时，基准臂活跃占比也会塌；探针实测 `(1.30,0.10)` 基准臂活跃占比 **0.606 未塌** ✓

## 四、🔴 本轮**已登记但尚未被正式跑证实**的三处外推（R118 §④）
1. **T=1 s ⇒ T=10 s 的计数线性**；
2. **单实现 ⇒ 3 实现的同量级**（探针是单实现）；
3. **活跃占比 ⇒ 中位率**：T=1 s 时计数是 0/1/2 的**粗量化**，中位率只能取 0 或 1 Hz ⇒
   探针"**≈1.0 Hz**"的分辨率极低，**真值可能在 0.5–1.5 Hz 之间**。
**⇒ 因此探针读数一律 `provisional`；`(a)∧(c)∧(e)` 是否真同时成立，只由这 30 臂决定。**

## 五、探针已测出的结构事实（**不作判据**，仅供解释正式结果；存 `data/m11_p5_l3_feasibility/`）

**完整 `bias=1.30` 行（T=1 s、单实现）**
| σ | 零驱动 `spk` | 基准 `spk` | **R** | 基准 `ei_ratio` | `ms/step` |
|---|---|---|---|---|---|
| 0.30 | 4,126 | — | — | 6.0406（**零驱动臂**，异常值，勿与基准臂混用）| 1.4379 |
| 0.20 | 4,875 | 5,546 | **0.879** ✗ | 2.7059 | 1.5026 |
| 0.15 | 4,666 | 5,295 | **0.881** ✗ | 2.6482 | 1.4259 |
| **0.10** | **46** | **5,149** | **0.0089** ✓ | 2.5767 | 1.4856 |

- **σ 在此是"悬崖"不是旋钮**：0.15→0.10 使 R 崖降 **100×**，而基准臂只掉 **2.8%** ⇒ **尾部选择性**（R118 §①）。
- **`bias` 侧已饱和**：零驱动活动在 `bias≥1.30` 后基本不涨（σ=0.30 时 2,~4,126 @1.30 vs 7,410 @1.60）。
- **`(a)` 的真正自变量是 `active_frac`**：`bias=0.70` 四档活跃占比 0.447–0.500（**恰在 50% 上下**）⇒ 中位数被静默神经元钉在 0；
  `bias=1.00` ⇒ 0.838 ⇒ 中位数变 0.3 Hz（R114）。
- **⚠️ 零驱动臂 `ei_ratio=127.05`（σ=0.10）是异常值**（零驱动下仅极少数细胞发放 ⇒ `i_exc` 极小 ⇒ 比值爆炸）⇒
  **不得与基准臂的 (e) 混用**（R109 第五十七条：引用数值必须带身份三元组）。

## 六、预注册禁止（**不得违反**）
🔴 **不得提高背景驱动（`drive_rate_hz` / `drive_mult`）去命中 (a)**：
`docs/m11_b3_g1_design.md` **L146** 明文"禁"，只能作**剂量-响应诊断**、**不得作标定手段**；L137 记录 M10 已付同一学费。
⇒ 判据表已补登 `protocol_drive_rate_hz=8.0` 并把该禁令写进其 `wrong_usage_consequence`（R117）。

## 七、本轮新增纪律（**执行时必须遵守**）
| 编号 | 内容 |
|---|---|
| 第五十六条 | 启动前静态检查必须同时验**路径**与**外部命令**；**"非零退出码"≠"工具报不通过"** |
| 第五十七条 | 引用任何数值必须带**身份三元组** `(config_id, estimator, protocol_id)`（R80 管单位，本条管身份）|
| 第五十八条 | **杠杆扫描必须双侧**，未测侧显式登记 `unmeasured`；单侧单调性只能外推到被测半空间内 |
| 第五十九条 | **禁止用 `2>/dev/null` 掩盖 `git add`/`commit` 失败**；提交成功的判据是**回读**（`git show --stat` 文件数与 add 清单比对）|
| 第六十条 | "机制方向"论证必须区分**直接分量**（可逐位判定 ⇒ 可下"不可能"）与**间接分量**（回路效应 ⇒ 只能下"未测"）|
| 第六十一条 | 向 append-only 表格追加记录：①不得连表头一起追加 ②必须做"记录数 + 唯一性 + **与上版差集** + 前缀逐字节"四项核对 |
| 第六十二条 | 凡"**父进程写文件、子进程读文件**"的驱动器，**路径参数必须绝对**，并在启动脚本里加绝对路径断言 |

## 八、跑完之后做什么（下一步）
1. 用 §3.1 评估器出 `m11_p5_l3_eval.{csv,json}` ⇒ 按 §3.2/3.3/3.4 判 (a)/(c)/(e)。
2. **若 C2 格三判据同时成立** ⇒ **G1 门条件出现** ⇒ 进入 G1 定稿确认（并把 R118 §④ 的三处外推**逐条证实或证伪**）。
3. **若失败** ⇒ 按 R118 的证伪式：**第四维结构对立确认 ⇒ 必须换机制（不再改参数）**
   ⇒ 启动机制候选（C2 STP depress 须显式测"选择比"；VIP/SST 亚型侧；慢适应仅在 (a) 允许时）。
4. 无论成败，**M11 后续**：S 慢状态门（P6）→ P4/P7–P16。
