# 生产者 `record` 改动记录：**改动未发生**（R202 的授权已由 R204 撤回、并已由 R208 验收）

> **⚠️ 先说结论（本文件存在的唯一理由）**：主 agent 本轮**追加授权（R202 = `docs/m11_acceptance_master.md` §3gn）**要求
> "对生产者做最小改动（`record = str(spec.get("record","counts"))` ＋ L755 改 `record=record`）＋ 三条接受性判据 ＋ 新 sha 登记"。
> **但同一份裁决史里，R202 之后已有两条更新的条目**：
> - **R204（§3gp，L6249–6274）**：**"⇒ 裁决：撤回 R202 的生产者改动授权（未使用）"**（L6262），并明确
>   **"`tools/run_m11_p5_arms.py` 无需改动、无需重新登记（应保持 `26292f7b…`／1537 行）；若节点已改动它 ⇒ **必须回滚到原 sha** 并说明"**（L6264）；
>   同时把判据换成"机制可核验性 + 计数逐位"（L6267–6272，5 条）；
> - **R208（§3gt，L6382–6420）**：**"✅ 6 臂 `record="full"` 重跑验收通过 —— 逐位复现 6/6"**，且 L6406 明记
>   **"R204 撤回 R202 授权后，生产者始终未被改动 ✓"**。
> **⇒ 因此本轮 **未执行** R202 的生产者改动**（执行它反而会违反 R204 ②、并使 R208 的验收记录失真）。
> **⇒ 本文件按 R202 ③ 的**三条判据**给出**等价证据**（在同一目的上更强：连"改一行"都没有），并逐条回应 R202 ④ 的产出契约问题。

---

## ① 生产者当前状态（= R204 要求的值，逐项实测）

| 项 | 值 | 判定 |
|---|---|---|
| `neural_exploration/tools/run_m11_p5_arms.py` sha256 | `26292f7b0078aad275ea2d9d21324f6ad47ea5da53e3323479dbe1ad56972b27` | **= R204 L6264 要求值** ✅（**未改动**） |
| 行数 | **1537** | **= R204 L6264 要求值** ✅ |
| `grep -n 'record="counts"'` | **只有 L752（settle）与 L755（测量窗）** | = R202 §① 的实读事实（硬编码仍原样） |
| 全文 `record="full"` | **0 处** | 生产者自己从不记录脉冲时刻（这正是本任务的存在理由） |
| `git diff --stat` | **空**（无任何已跟踪文件被改） | ✅ |
| 冻结参考副本 | `neural_exploration/tools/_archive/run_m11_p5_arms.26292f7b0078.py`（**1537 行**，sha = `26292f7b…`，`cmp` **逐字节相同**） | 新建；`_archive/` 原有 1 件（评估器归档件），本件为第 2 件 |
| 其余已登记件 | 引擎 `src/mouse_cortex_engine.py` = `90c02b37…`；6 份 spec / 6 份 `counts.csv` / `poptrace.npy` 全部 = 注册 sha | ✅ 未变 |

⚠️ **关于"备份"**：R202 要求的 `_archive/run_m11_p5_arms.<旧sha12>.py` 语义是"**改动前原文**"。因为**本轮未改动**，
该副本与 live 文件**逐字节相同**（`cmp` 通过）⇒ 它现在的作用是"**冻结参考副本**"（把产出本批 6 臂的代码修订固定下来，
配合各臂 JSON 里的 `producer_sha256`），**不是**"已发生改动的证据"。请勿据此认为生产者被改过。

---

## ② 判据 ①「逻辑证明」在当前路径下的**对应论证**

R202 判据①针对的是"改代码"，其证明是"未给 `record` 键的臂走默认分支 ⇒ kwargs 逐字相同"。
**当前路径不改代码** ⇒ 该判据**不适用**，但它要防的事（"**执行路径被改动**"）**以更强的形式**在产物内自证：

1. **夹具 = 生产者自己的代码对象**（不是复制品）：`tools/run_m11_p5_spiketimes.py` **整体调用** `run_one(spec, out_dir)`；
   `--self-test` 实测 `build_engine`/`make_events`/`effective_seeds`/`_mechanism_preflight`/`run_one`/`_net_echo` 的
   `__code__.co_filename` 全 = 生产者文件、`__globals__ is vars(P)` 全 True、`__module__ = "run_m11_p5_arms"`。
2. **唯一的改动 = 一次已登记形参的取值替换**：`record` 是引擎 `run()` 的**登记形参**（`src/mouse_cortex_engine.py` **L551**
   `record: str = "counts"`）。替换判据 = **三重与**：`pop_trace is True` ∧ `ei_integrals is True` ∧ `record == "counts"`
   ⇒ 只命中**测量窗那次调用**（生产者 L755）；**L752 的 settle 调用**（`delivery="step"`、无 `pop_trace`/`ei_integrals`）**不触发**。
3. **替换发生次数逐臂留痕**（`<run_id>.json::record_override.forced_calls`，6/6 一致）：
   ```json
   [{"call": "passthrough", "n_steps": 10000,  "record": "counts",        "pop_trace": false, "ei_integrals": false, "delivery": "step"},
    {"call": "forced_full", "n_steps": 200000, "record": "full(forced)", "record_kwarg_seen": "counts", "pop_trace": true, "ei_integrals": true, "delivery": "chunk", "n_spikes": <spk_total>}]
   ```
   ⇒ **替换 = 1 次／臂，且落在测量窗**；settle 那次的 kwargs 与改前**逐字相同**（`record="counts"` 原样传入）。
4. **前向兼容**：若日后上游把 `record` 改成参数入口（L752/L755 不再是硬编码），本机制**不再触发**，
   且 `--self-test` 会因 `PRODUCER_RECORD_COUNTS_LINES=(752,755)` 不符而**先失败（退 95）**，不会静默漂移。

---

## ③ 判据 ②「2 臂逐字节比对」（arbiter 建议的两臂；结论对 **全部 6 臂** 成立）

新产物（生产者**自己写的** echo 副本，`_producer_echo/`）vs **已落盘** `data/m11_p5_l4_0_runs/`：

| 臂 | `_counts.csv` sha256 新 vs 旧 | `_poptrace.npy` sha256 新 vs 旧 | `sha256_of_trace` | `.json` 差异键 |
|---|---|---|---|---|
| `l4_g0.35_8192_T10000ms_d1.0_s0` | `ea4118c7a890ddbd…` = `ea4118c7a890ddbd…` **✅逐字节相同** | `dea800c46c90c3c5…` = `dea800c46c90c3c5…` **✅逐字节相同** | `dae6aa0a31ebd906…` **✅相同** | 9 个，**语义键 0 个** |
| `l4_g0.35_8192_T10000ms_d0.0_s0` | `303b8e5a67015626…` = **✅** | `efa675c3479028bc…` = **✅** | `a47f399bb1f23820…` **✅相同** | 9 个，**语义键 0 个** |

**差异键的完整分类（两臂相同、6 臂亦然）**：
- **易变（6）**：`finished_at`、`pid`、`rss_mb`、`wall_s`、`ms_per_step`、`ms_per_step_steady`；
- **路径（2）**：`counts`、`poptrace`（新产物写在 `_producer_echo/`，故路径字符串不同——任意"换目录重跑"都必然如此）；
- **入口口径（1）**：`interpreter` = `{…}`（旧）→ `null`（新）。**成因非动力学**：生产者 `--arm` 入口
  （`arm_spec_from_cli`，L943 起）会**先往 spec 注入 `interpreter`**，而本工具**逐字直传注册 spec 文件**（spec 内没有该键）
  ⇒ `run_one` 写入 `spec.get("interpreter")` = `null`。本次运行的解释器已单独记在每臂 JSON 的 `interpreter` 块（`.venv-m9`）。
- **⇒ 语义键 0 个**：`agg`、`ei`、`sha256_of_trace`、`net`、`seeds`、`slow_trace_stats`、`n_events`、
  `n_steps_meas`、`n_steps_settle`、`delivery`、`engine_version`、`n_ch`、`T_ms`、`settle_ms`、`drive_mult`、`role`、`n_neuron` … **全部逐字相同（6/6）**。

---

## ④ 判据 ③「官方臂 JSON 键集全同」（防 schema 漂移）

| 臂 | 顶层键集 | 键数 新 vs 旧 | 嵌套键集（`ei`/`seeds`/`net`/`slow_trace_stats`/`agg`） |
|---|---|---|---|
| `…d0.0_s0` | **✅ 全同** | 41 vs 41 | ✅ |
| `…d0.0_s1` | **✅ 全同** | 41 vs 41 | ✅ |
| `…d0.0_s2` | **✅ 全同** | 41 vs 41 | ✅ |
| `…d1.0_s0` | **✅ 全同** | 41 vs 41 | ✅ |
| `…d1.0_s1` | **✅ 全同** | 41 vs 41 | ✅ |
| `…d1.0_s2` | **✅ 全同** | 41 vs 41 | ✅ |
（`…` = `l4_g0.35_8192_T10000ms`；比较对象 = `_producer_echo/<run_id>.json` vs `m11_p5_l4_0_runs/<run_id>.json`。）

---

## ⑤ 判据 ④「spec 副本：仅增 `record` 一项」的证明

**本轮实际输入**：**不需要任何 spec 副本** —— 6 份**注册 spec 文件被逐字读入**，且以 **sha256 预注册守卫**
（`SPEC_SHA_PIN`，不符即**退 97**）⇒ 这比"仅增一项"**更强**（差异 = **0 项**）。
为满足 R202 ④ / R204 ③4 的**字面要求**，另做**只写 `/tmp`** 的副本演示（**不改任何注册文件**）：

| 臂 | 原 spec sha256 | 副本 sha256 | JSON 逐键：新增 / 删除 / 改值 | `diff` |
|---|---|---|---|---|
| `…d0.0_s0` | `efe51f9304b2553f…` | `9a50a86f44fb3b74…` | `+['record']` / `−[]` / 改 `[]` | **1 增 / 0 删**：`+ "record": "full",` |
| `…d0.0_s1` | `a21e92cae073bc49…` | `8646908ab9b6992f…` | `+['record']` / `−[]` / 改 `[]` | **1 增 / 0 删** |
| `…d0.0_s2` | `c1f11e5001e9ca0b…` | `6fb9dc73afc46505…` | `+['record']` / `−[]` / 改 `[]` | **1 增 / 0 删** |
| `…d1.0_s0` | `07f3b5c488c8d358…` | `4771b35343b31aed…` | `+['record']` / `−[]` / 改 `[]` | **1 增 / 0 删** |
| `…d1.0_s1` | `daf48ea96a7ab5ea…` | `b61545f89a958826…` | `+['record']` / `−[]` / 改 `[]` | **1 增 / 0 删** |
| `…d1.0_s2` | `2e3d38fd9eb96747…` | `34f6a87d8feac3ac…` | `+['record']` / `−[]` / 改 `[]` | **1 增 / 0 删** |

**单臂 `diff` 原文**（`…d1.0_s0`；副本插在**首个键之前**，故是**纯一行插入**，不触碰任何其它行）：
```
--- neural_exploration/data/m11_p5_l4_0_runs/l4_g0.35_8192_T10000ms_d1.0_s0.spec.json
+++ /tmp/r202_spec_copies_oneline/l4_g0.35_8192_T10000ms_d1.0_s0.spec.json
@@ -1,4 +1,5 @@
 {
+ "record": "full",
  "run_id": "l4_g0.35_8192_T10000ms_d1.0_s0",
  "stage": "l4_0",
  "config_id": "p5_L4_g0.35_b1.30_s0.10",
\ No newline at end of file
```
（原 spec 文件**无结尾换行**；副本沿用原风格 1 空格缩进。若把 `record` 追加为**最后一个**键，会因 JSON 逗号规则**连带改动前一行** ⇒ 故采用**首键插入**以保证"恰好一行差异"。）

---

## ⑥ R202 ④ 的「产出契约」：`_flush_rec` 的落点与 `delivery` 的判据（**实读 + 实证，非猜**）

**实读结论（附行号）**：
1. `_flush_rec` **定义在父类** `src/adult_engine.py` **L841–847**（`MouseCortexEngine` 未覆写：`run.__qualname__ == "MouseCortexEngine.run"`，其代码对象来自 `src/mouse_cortex_engine.py`）：
   ```
   841:    def _flush_rec(self, spk_ring, rows: int, t0_rel, spike_steps, spike_idx):
   842:        nz = spk_ring[:rows].nonzero(as_tuple=False)
   843:        if nz.numel():
   844:            r = nz[:, 0].cpu().numpy().astype(np.int64)      # 行号 = 窗内步号
   846:            spike_steps.append(t0_rel + r)
   847:            spike_idx.append(c)                              # 列号 = 神经元 id
   ```
2. **两个容器是 `run()` 的局部变量，不挂在 `eng` 上**：`src/mouse_cortex_engine.py` **L591–592**
   `spike_steps = []` / `spike_idx = []`；flush 点 = **L787–789**（chunk 路径）与 **L803–804**（非 chunk 路径）。
3. **唯一落点 = `res["spikes"]`**：同文件 **L817–823**
   `"spikes": ((np.concatenate(spike_steps), np.concatenate(spike_idx)) if (record == "full" and spike_steps) else None)`。
   ⇒ 脉冲时刻**只经 `res` 返回**，生产者 `run_one` **不落盘**（它只取 `pop`/`spk_total`/`ei_integrals`）
   ⇒ **这正是必须另写驱动件捕获 `res["spikes"]` 的原因**（R204 ① 的路径）。
4. **`record == "full"` 的缓冲分配**：**L581–582**（`delivery == "chunk"` ⇒ `rec_rows = W`）、**L584–587**（chunk 恒分配 `spk_ring`）、
   **L762**（chunk 路径**逐 step** `rec_rows_v[rec_ptr].copy_(spk)`）、**L766–768**（非 chunk 路径逐 step 写入）。

**`delivery` 该取哪一个？—— 答：取 spec 里的注册值 `"chunk"`，不动它**（6 份 spec 实测均为 `"chunk"`）：
- **chunk 口径下同样是"逐 step"**：`W = min(SLOT_EXTRA=40, max_delay=80) = 40` 步（探针实测 `max_delay=80`、`n_slot=121`），
  每个 chunk 内 `L762` **每 step 写一行**、`L786–789` 每个 chunk flush 一次 ⇒ **200000 步 / 40 = 5000 次 flush**，
  `rec_base` 逐 chunk 累加 ⇒ `spike_steps` 里的行号 = **窗内绝对步号**（`t0_rel = rec_base - record_from`，`record_from=0`）。
- **实证（三条，全部逐位）**：
  1. 逐神经元：`np.bincount(neuron_id) == 已落盘 counts.csv` **6/6 逐位相同**；
  2. **逐 step 精确性（最强的一条）**：由 `step` 重建群体轨迹 `np.bincount(step, minlength=200000)` 与生产者
     **`_producer_echo/<run_id>_poptrace.npy` 逐位相同（max|Δ| = 0.0）6/6** ⇒ 记录的**确是逐 step 时刻**
     （若是 chunk 粒度聚合，不可能逐位复现 200000 点轨迹）；
  3. 事件数 == `spk_total` 且 `t_ms = step × dt_ms` 精确、`step ∈ [0, 200000)` 升序。
- ⇒ **不存在"`record="full"` 与既有臂 `delivery` 冲突"的情形**，故**未改 `delivery`**（未破坏 R201 ④ 的自证）。

---

## ⑦ 6 臂逐位对账（R201 ④／R204 ③3／R208 ① 同一标准）

| 臂 | role | 新 `spk_total` | 已落盘 CSV | 逐位相同 | 不同神经元 | baseline 锚 |
|---|---|---|---|---|---|---|
| `…d0.0_s0` | zero_drive | 3143 | 3143 | **✅** | 0 | — |
| `…d0.0_s1` | zero_drive | 3174 | 3174 | **✅** | 0 | — |
| `…d0.0_s2` | zero_drive | 2895 | 2895 | **✅** | 0 | — |
| `…d1.0_s0` | baseline | **47203** | 47203 | **✅** | 0 | **✅ 47203** |
| `…d1.0_s1` | baseline | **47342** | 47342 | **✅** | 0 | **✅ 47342** |
| `…d1.0_s2` | baseline | **47036** | 47036 | **✅** | 0 | **✅ 47036** |

---

## ⑧ 若主 agent 仍要求**按 R202 执行**（覆盖 R204/R208）——**我没有执行，等你明确确认**

**为什么我没有执行**：R204 L6264 明确要求生产者保持 `26292f7b…`／1537 行、且"**若节点已改动它 ⇒ 必须回滚到原 sha**"；
R208 L6406 又已把"**生产者始终未被改动**"写进验收记录。**先改后报会让 R208 的这条记录失真**，
且在"**不改已登记文件优先**"的原则下（R204 L6263）不是必要动作。

**若你要覆盖，需要（我可在约 25–30 min 内完成，MPS 严格串行、`caffeinate -i -s`）**：
1. 改 2 处：`run_one` 内加 `record = str(spec.get("record", "counts"))`；**L755** `record="counts"` → `record=record`（L752 不动）；
2. **新 sha256 + 行数**（并重登记；旧值 `26292f7b…`／1537 行作废）；
3. R202 ③ 三条判据：① 逻辑证明（默认分支 kwargs 逐字相同）；② 2 臂 A/B（改前/改后生产者各跑一次，输出到**新目录**，
   `_counts.csv`／`_poptrace.npy`／`.json` 除易变字段外**逐字节相同** + `sha256_of_trace` 相同）；③ 键集全同；
4. 用 spec 副本（仅增 `record:"full"`）**重跑 6 臂**，再做逐位对账（baseline 锚不变）；
5. 已知**残余差异**：走生产者 `--arm` 入口时 JSON 的 `interpreter` 会被注入（本路径为 `null`）⇒ 该差异随之消失。
**⇒ 上述 3/4 我已有等价证据（本文件 ③④⑤⑦ 与 `SELFCHECK.md`），差别只在"是否改那一行代码"。**

---

## ⑨ 本轮我**没有**做的事

未改生产者（sha `26292f7b…`、1537 行，`git diff --stat` 空）；未改 `src/*.py`（引擎 `90c02b37…`）；
未改任何已登记产物（6 份 spec、`counts.csv`、`poptrace.npy`、`m11_acceptance_master.md` 均未被写）；
未覆盖 `data/m11_p5_l4_0_runs/`；本文件与 `_archive/run_m11_p5_arms.26292f7b0078.py` 为**新建**；
未联网；未 `git add`／`git commit`；未用 MPS（本轮只读 + 写 `/tmp`）。

**证据脚本（均在 `/tmp`，不落仓）**：`/tmp/r202_equiv_evidence.py`（判据②③④）、
`/tmp/r202_spec_one_line.py`（判据④一行差异）、`/tmp/spike_step_precision_check.py`（⑥2 的逐 step 逐位证明）、
`/tmp/audit_spiketimes.py`（6 臂 npz/json 独立审计）。
