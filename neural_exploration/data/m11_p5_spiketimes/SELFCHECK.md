# SELFCHECK：G1 工作点 6 臂 `record="full"` 重跑的**计数逐位对账**

**依据**：`neural_exploration/docs/m11_acceptance_master.md` **§3gm = R201**（④ 自证标准、
⑤ 实施方式）。**Pass 标准**：逐神经元 `count` 与已落盘 `data/m11_p5_l4_0_runs/<run_id>_counts.csv` **逐位相同** ⇒ 证明新产物记录的是**同一批臂**（同 `(net_seed,event_seed,seed)` 下确定性）。

**生成**：`run_m11_p5_spiketimes.py --reconcile`（只读；不跑仿真、不用 MPS）；生成时刻 2026-10-15T10:34:21。

## ① 逐臂对账表

| 臂 | role | 新 `spk_total` | 已落盘 `spk_total` | **逐位相同** | 不同神经元数 | `counts` sha256 同 | `poptrace` 逐字节 | 锚 | 新 `wall_s` | 已落盘 `wall_s` |
|---|---|---|---|---|---|---|---|---|---|---|
| `l4_g0.35_8192_T10000ms_d0.0_s0` | zero_drive | 3143 | 3143 | **是 ✓** | 0 | 是 ✓ | 相同 ✓ | — | 153.965 | 152.857 |
| `l4_g0.35_8192_T10000ms_d0.0_s1` | zero_drive | 3174 | 3174 | **是 ✓** | 0 | 是 ✓ | 相同 ✓ | — | 139.746 | 133.077 |
| `l4_g0.35_8192_T10000ms_d0.0_s2` | zero_drive | 2895 | 2895 | **是 ✓** | 0 | 是 ✓ | 相同 ✓ | — | 172.312 | 141.320 |
| `l4_g0.35_8192_T10000ms_d1.0_s0` | baseline | 47203 | 47203 | **是 ✓** | 0 | 是 ✓ | 相同 ✓ | 47203 ✓ | 218.130 | 199.297 |
| `l4_g0.35_8192_T10000ms_d1.0_s1` | baseline | 47342 | 47342 | **是 ✓** | 0 | 是 ✓ | 相同 ✓ | 47342 ✓ | 197.640 | 188.730 |
| `l4_g0.35_8192_T10000ms_d1.0_s2` | baseline | 47036 | 47036 | **是 ✓** | 0 | 是 ✓ | 相同 ✓ | 47036 ✓ | 198.971 | 205.379 |

**⇒ 逐位相同：6/6；`counts.csv` sha256 相同：6/6；`poptrace.npy` 逐字节相同：6/6。**

**baseline 独立锚（R201 §④）**：`l4_g0.35_8192_T10000ms_d1.0_s0` = 47203（锚 47203 ✓）、`l4_g0.35_8192_T10000ms_d1.0_s1` = 47342（锚 47342 ✓）、`l4_g0.35_8192_T10000ms_d1.0_s2` = 47036（锚 47036 ✓）。

**墙钟**：协议 `wall_s` 合计 = **1080.8 s**（登记值 1020.7 s）；各臂新 `wall_s` 见上表（`wall_s` 只计 settle+测量窗，不含建网 ⇒ 与已落盘同口径）。

## ② 脉冲时刻产物结构（真实读盘）

每臂 `<run_id>_spiketimes.npz`：`neuron_id`(int32) / `step`(int64) / `t_ms`(float64) 三数组 ＋ 标量 `dt_ms`/`n_steps`/`spk_total`/`t0_ms`/`run_id`/`record_mode`/`producer_sha256`。
**时间原点 = 测量窗起点**（settle 之后 `reset_counts()` 那一刻）⇒ 绝对仿真时刻 = `settle_ms + t_ms`；`t_ms = step × dt_ms`，`dt_ms`（实测值）见下表。

| 臂 | 键数 | `dt_ms` | `n_steps` | 事件数 | `spk_total` | 事件数==`spk_total` |
|---|---|---|---|---|---|---|
| `l4_g0.35_8192_T10000ms_d0.0_s0` | 10 | 0.05 | 200000 | 3143 | 3143 | 是 ✓ |
| `l4_g0.35_8192_T10000ms_d0.0_s1` | 10 | 0.05 | 200000 | 3174 | 3174 | 是 ✓ |
| `l4_g0.35_8192_T10000ms_d0.0_s2` | 10 | 0.05 | 200000 | 2895 | 2895 | 是 ✓ |
| `l4_g0.35_8192_T10000ms_d1.0_s0` | 10 | 0.05 | 200000 | 47203 | 47203 | 是 ✓ |
| `l4_g0.35_8192_T10000ms_d1.0_s1` | 10 | 0.05 | 200000 | 47342 | 47342 | 是 ✓ |
| `l4_g0.35_8192_T10000ms_d1.0_s2` | 10 | 0.05 | 200000 | 47036 | 47036 | 是 ✓ |

## ③ 可复跑的对账命令（只读）

```bash
env PYTHONHASHSEED=0 MPLBACKEND=Agg \
  /Users/weidong/ai/small_world/.venv-m9/bin/python \
  /Users/weidong/ai/small_world/neural_exploration/tools/run_m11_p5_spiketimes.py \
  --reconcile --out-dir /Users/weidong/ai/small_world/neural_exploration/data/m11_p5_spiketimes
```

退出码 0 = 6/6 逐位相同；93 = 有不符（本表会逐臂标出）。

## ④ 启动脚本要点（脚本本体在 `/tmp`，**不落仓**）

- 启动前 `pgrep -f "run_m11_p5"` 与 `pgrep -f "run_m11_p5_spiketimes"` **必须为空**（驱动内建 `mps_guard()`，命中即 **退 99**；自己的启动链 pid 被排除）；
- `caffeinate -i` 包住整条驱动器；`PYTHONHASHSEED=0 MPLBACKEND=Agg`；
- 解释器**逐字** `/Users/weidong/ai/small_world/.venv-m9/bin/python`（**不用 `sys.executable`**）；
- `--out-dir` **绝对路径**（非绝对 **退 96**）；逐臂**新进程**（MPS 串行）；
- 整条命令用后台作业跑（预计约 17 min；6 臂 `wall_s` 实测合计 1020.7 s）。

```bash
# /tmp/g1_6arm_spiketimes.sh
#!/bin/bash
# G1 工作点 6 臂 record="full" 重跑（依据 R201 = docs/m11_acceptance_master.md §3gm）
# 启动纪律：MPS 严格串行 / caffeinate -i / PYTHONHASHSEED=0 MPLBACKEND=Agg
#           解释器逐字 = .venv-m9/bin/python（绝不用 sys.executable）/ --out-dir 必须绝对路径
# 本脚本**不落仓**（只在 /tmp），要点写进 SELFCHECK.md ④
set -u

PY=/Users/weidong/ai/small_world/.venv-m9/bin/python
DRV=/Users/weidong/ai/small_world/neural_exploration/tools/run_m11_p5_spiketimes.py
OUT=/Users/weidong/ai/small_world/neural_exploration/data/m11_p5_spiketimes
LOG=/tmp/m11_p5_spiketimes.log

# ① 启动前：MPS 必须空闲（驱动器内还有一道 mps_guard ⇒ 命中即退 99）
if pgrep -f "run_m11_p5" >/dev/null 2>&1 || pgrep -f "run_m11_p5_spiketimes" >/dev/null 2>&1; then
  echo "REFUSE(99): 已有 MPS 作业 ⇒ 不启" | tee -a "$LOG"
  exit 99
fi
mkdir -p "$OUT"

{
  echo "=== $(date -u +%FT%TZ) pre-launch"
  echo "pgrep -f run_m11_p5            -> [$(pgrep -f 'run_m11_p5' | tr '\n' ' ')]"
  echo "pgrep -f run_m11_p5_spiketimes -> [$(pgrep -f 'run_m11_p5_spiketimes' | tr '\n' ' ')]"
  echo "python  = $PY"
  echo "driver  = $DRV"
  echo "out-dir = $OUT"
  # ② 整条以 `caffeinate -i -s` 包裹：
  #    -i = 阻止**用户空闲系统睡眠**；-s = 阻止**系统睡眠**（AC 供电下）。
  #    🔴 实测教训：只用 -i 时，本机仍会每 ~35 min 进入一次 **Maintenance Sleep 约 920 s**
  #    （pmset 记录 926/919/924/927 s）⇒ 进程被冻结、真实墙钟被拉长 ~7×，
  #    而 `wall_s`（perf_counter，macOS mach_absolute_time **不计睡眠**）仍 ~150 s
  #    ⇒ 必须加 -s 才能把"真实墙钟"与"协议 wall_s"拉齐。
  env PYTHONHASHSEED=0 MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    caffeinate -i -s "$PY" -u "$DRV" --out-dir "$OUT"
  rc=$?
  echo "EXIT=$rc"
  echo "=== $(date -u +%FT%TZ) end"
  exit $rc
} >> "$LOG" 2>&1
```

## ⑤ 未改动声明

- 生产者 `neural_exploration/tools/run_m11_p5_arms.py` sha256 = `26292f7b0078aad275ea2d9d21324f6ad47ea5da53e3323479dbe1ad56972b27`（登记值一致）。
- 引擎 `neural_exploration/src/mouse_cortex_engine.py` sha256 = `90c02b37d922d7ac9aefd78023c7db23a2fcd8a70147c1b226b5e9b7a7a9dc34`（登记值一致）。
- 6 份 spec 与 6 份 `counts.csv`/`poptrace.npy` 的 sha256 全部与登记值一致（跑前守卫，不符退 97）。
- 本任务**只新建** 4 件产物（本驱动件、本目录、本文件、`/tmp` 启动脚本）；`git add`/`git commit` 均未执行。

## ⑥ 对账结论

**全部通过**：计数逐位相同 6/6、`poptrace` 逐字节相同 6/6、脉冲事件数 == `spk_total` 6/6 ⇒ 新产物的**逐神经元脉冲时刻属于同一批臂**。

---

# 附录（⑦–⑩）：**人工追加**（`--reconcile --md-out` 只重写上面 ①–⑥；本附录不会被自动重写）

## ⑦ 交付物 sha256（跑完即冻结）

| 对象 | 绝对路径 | 字节 | sha256 |
|---|---|---|---|
| 驱动件 | `/Users/weidong/ai/small_world/neural_exploration/tools/run_m11_p5_spiketimes.py` | 1034 行 | `1cc70c98971cf0724b0363ebaa6e71954517b6a2e5e604b576c59fe0aacd796b` |
| 本文件 | `/Users/weidong/ai/small_world/neural_exploration/data/m11_p5_spiketimes/SELFCHECK.md` | — | 追加附录后由 `shasum -a 256` **复算**（附录内不索引自身） |
| 启动脚本 | `/tmp/g1_6arm_spiketimes.sh`（**不落仓**） | — | 原文见 ④ |
| 日志 | `/tmp/m11_p5_spiketimes.log`（**不落仓**） | — | 6 臂全过程 stdout/stderr |

**6 臂脉冲时刻文件**（`<out-dir>/<run_id>_spiketimes.npz`）与**每臂 JSON**：

| 臂 | npz 字节 | npz sha256 | json 字节 | json sha256 |
|---|---|---|---|---|
| `…d0.0_s0` | 65776 | `b1fa41df7fe5ed9c7c4149444117a7cec52e9eab1699e795d78f21fa18bcf7c7` | 12313 | `11e301174e527e4cc57e735398e735934c181434d2e6380b045b0aea120f8c00` |
| `…d0.0_s1` | 66396 | `57088da5b024696c7118ea63b55ecd774c317b83407f12f85a0847c9dfd1a10d` | 12282 | `dad88c15705654b8afc257e847aa36f51bf1aef1e4fbbb4daffb98ed904e338c` |
| `…d0.0_s2` | 60816 | `81ef0f10bb8860462c7694e5920cbd11a8675b135cfdaf75fb0ab6b6f73b5194` | 12284 | `5df596bf8b304bf02394a550f56a63bf7246b39a82d697e7abfaeaf346c21226` |
| `…d1.0_s0` | 946976 | `01c559e250bdd05ce5548f521d2c134b392490efc460d5c96791dc714d0d0b29` | 12307 | `f78cd3c67526d762dd5e3f58f59b04faad725dc033fb806d471586b56cd0b8e2` |
| `…d1.0_s1` | 949756 | `cab98943cea6e95072eb9a49beb147e5c85dac4592a68de50b1edbf3018a176e` | 12296 | `8333e904cc2e17e5f3c215b1d53f967a897651b233b88ff5749a211a2f05d576` |
| `…d1.0_s2` | 943636 | `645593709806df47e6d434f46cce65a959f5bc649536ab3aa5b16d195787a4d7` | 12296 | `f200e02e4d207cb502a6550eed9ab64b8d1678b9eb090e481f5ad72a84dcdcbd` |

（`…` = `l4_g0.35_8192_T10000ms`。`_producer_echo/` 为生产者自己重写的 `json`/`counts.csv`/`poptrace.npy` 副本 = 对账原始证据。）

**全部 6 臂均来自同一次不间断运行**（本机 10:18:24–10:34:02，驱动件修订 `1cc70c98…`）。此前有一次 08:49–09:58 的运行产出过 3 个 `d0.0` 臂，已被本次运行**逐字节覆盖**（日志 3 条 `note: 覆盖本任务自己的 npz`，行 44/49/54）⇒ 盘上 6 臂**同一修订、同一进程纪律**。

## ⑧ 启动纪律的**判别性验证**（正例 + 反例，两向都测）

- **正例（自己的启动链必须不误判）**：`caffeinate -i <venv python> <同名脚本> …` 下调用 `mps_guard()` ⇒ `n_busy=0`，并**留痕**被判为自身链的进程及理由
  （实测 `why="descendant(本驱动器 spawn 的子进程)"`：`caffeinate` 会留下一个**持断言的子进程**、其 argv 与启动行逐字相同 —— 这正是最初只排除"自己∪祖先"时**误判 99** 的成因）。
- **反例（外来 MPS 作业必须拒启）**：在自己**独立会话/进程组**里放一个 argv 含 `run_m11_p5` 的假作业
  `… python -c "import os,time; os.setsid(); time.sleep(40)" run_m11_p5_arms --decoy-foreign` ⇒ `mps_guard()` 报
  `REFUSE(99): 已有 MPS 作业 ⇒ 拒启`（该 pid 的 `why=None`，即不被任何豁免覆盖）**✓ 判别性保持**（假作业已 kill，非运行时无残留）。
- 驱动件内 `mps_guard` 的判据：`pgrep -f "run_m11_p5"` 与 `pgrep -f "run_m11_p5_spiketimes"` **逐字**照纪律；豁免集合 = 自己 ∪ 祖先 ∪ 后代 ∪ **同进程组**，且豁免项一律写进产物 JSON（`mps_guard.excluded`）。

## ⑨ 真实墙钟：**本机维护睡眠**（本次实测发现，必须登记）

**R201 §③ 的 "0.284 GPU-h / 17 min" 是按 `wall_s` 合计（1020.7 s）估的**，与**真实墙钟不等价**：

- 首次启动用 `caffeinate -i` ⇒ 每臂真实墙钟 ≈ **1075 s**，而同臂 `wall_s` 仅 **150 s**。
  根因（`pmset -g log` 实测）：本机每 ~35 min 进入一次 **Maintenance Sleep**，
  实测睡眠时长 **926 / 919 / 924 / 927 s**；`caffeinate -i` 只置 `PreventUserIdleSystemSleep=1`，**拦不住它**。
  进程被冻结 ⇒ 真实墙钟被拉长约 **7×**；而 `wall_s` 用 `time.perf_counter()`（macOS `mach_absolute_time`，**不含睡眠**）⇒ 仍 ~150 s
  ⇒ **"真实墙钟"与"`wall_s` 合计"在这台机器上可以差一个数量级**。
- 改用 **`caffeinate -i -s`**（`-s` 置 `PreventSystemSleep=1`）后：`pmset -g assertions` 实测
  `PreventSystemSleep 1 / PreventUserIdleSystemSleep 1`，`pmset -g log` 再无睡眠事件 ⇒ 6 臂
  真实墙钟 **1094.3 s**（≈18.2 min，与 `wall_s` 合计 1080.8 s 基本一致）✓
  **⇒ 纪律应写为 `caffeinate -i -s`（或 `-is`）；只写 `-i` 在本机会被维护睡眠拖慢 ~7×。**
- 协议 `wall_s` 合计 = **1080.8 s** vs 登记 **1020.7 s**（+5.9%）。

## ⑩ `record="full"` 有没有**超出"多记录"**的影响？（逐条给证据）

1. **对动力学：零影响（实测）**。6/6 臂**逐神经元 count 逐位相同**、`poptrace.npy` **逐字节相同**，
   且生产者 echo JSON 与已落盘 JSON 的**全部动力学键相同**：`agg` / `ei` / `sha256_of_trace` / `net` /
   `seeds` / `slow_trace_stats` / `n_events`（6/6 全 True）。差异键恒为 9 个且全属**来源/时耗**类：
   `counts`、`pop_trace`（路径指向 `_producer_echo/`）、`finished_at`、`pid`、`rss_mb`、`wall_s`、
   `ms_per_step`、`ms_per_step_steady`、`interpreter`。
   ⚠️ 其中 `interpreter` 由 `{…}` 变 `null` 是**因为本工具直接调用 `run_one` 并逐字传入 spec 文件**，
   而 spec 文件本身不含 `interpreter` 键（生产者 `--arm` 入口会先注入该键）⇒ **不是动力学差异**；
   本次运行的解释器已单独记在每臂 JSON 的 `interpreter` 块（`.venv-m9`，与纪律一致）。
2. **对时耗：存在的成本已定位，但本次数据不支持定量归因**。`record="full"` 在 chunk 口径下走
   引擎 `L787`/`L803` → `_flush_rec`（`adult_engine.py` L841–847），每 chunk 一次
   `spk_ring[:rows].nonzero()` ＋ `nz[:,…].cpu().numpy()`（**设备→主机同步**）。
   实测本协议 chunk 宽度 **W = min(SLOT_EXTRA=40, max_delay=80) = 40 步**
   （探针：`max_delay=80, n_slot=121, SLOT_EXTRA=40`）⇒ **每臂恰好 5000 次 `_flush_rec`**。
   但 `wall_s` 逐臂差值 **−6.4 … +31.0 s**（正负不一），与已落盘同组三 seed 的自然离散
   （zero_drive 133.1–152.9 s ≈ ±7%）同量级 ⇒ **不宣称"record=full 使 wall_s 增加 X%"**（未做配对 A/B）。
3. **语义（必须记住）**：`record="full"` 的 `res["spikes"]` = `(step, neuron_id)`，其中 `step` 是
   **测量窗内的相对步号**（`record_from=0`；绝对步号另存于 `res["abs_step_end"]`）
   ⇒ 本目录 `.npz` 用 `t0_ms=0.0` 显式标注：**绝对仿真时刻 = `settle_ms + t_ms`**。
4. **已知限制（显式，不静默）**：引擎 `L822–823` 只在 `spike_steps` 非空时给 `res["spikes"]`
   ⇒ 若某臂 `spk_total == 0`，本工具会**以退出码 1 fail-stop**（而不是产出空产物）。
   本 6 臂均有脉冲（3143/3174/2895/47203/47342/47036）⇒ 未触发；但对"全零活动臂"须另行处理。

## ⑪ 独立复核（不依赖产物自述）

只读审计脚本 `/tmp/audit_spiketimes.py`（**不落仓**）逐臂重算并全部通过（**ALL 6 ARMS OK**）：
`json.sha256 == sha256(npz 文件)`、`n_events == spk_total`、
`np.bincount(neuron_id) == 已落盘 counts.csv`（逐位）、`t_ms == step × dt_ms`（精确相等）、
`step` 单调不减、`step ∈ [0, n_steps)`、`neuron_id ∈ [0, 8192)`、
`json.spikes.sha256_step == sha256(step.tobytes())`、`role` 与 dose 一致。

**未改动声明（复核命令）**：`git diff --stat` ⇒ **空**（无任何已跟踪文件被改）；
`neural_exploration/tools/run_m11_p5_arms.py` = `26292f7b…`、`src/mouse_cortex_engine.py` = `90c02b37…`、
`data/m11_p5_l4_0_runs/l4_g0.35_8192_T10000ms_d1.0_s0_counts.csv` = `ea4118c7…`（与登记一致）。
本次**未联网**、未 `git add`/`git commit`、未改任何已存在文件。
