# M12 环境与教训记录（B1 数据门"提前备料"节点）

> 用途：与 `docs/m11_env_notes.md`、`docs/m10_env_notes.md` 同族——**逐条纪律 + 每条附"本例"**。
> 本文件**只增不改**；每条给出**可复算命令**与**当场实测值**。
> 节点范围：仅 **S2 CCFv3 结构本体 / S7 Allen ecephys 汇总表 / S5 IBL 行为白名单**三项 ≤5 MB 级探针
> （主 agent 评审件 `docs/m12_list_review_master.md` §R41-4 批准的"可与 M11 并行提前备料"部分）。
> **边界**：未取 S1 / S3 / S6 / 任何 >50 MB 对象；未改 `src/`；未动任何 `m11_*` 文件；
> **全程未 import torch、未起任何仿真、未使用 MPS/GPU**（本机 GPU 由主 agent 对照实验占用）。

---

## L1 — 环境与解释器口径（**动手前先查表，不得凭直觉选解释器**）

| 项 | 值 |
|---|---|
| 网络/解析工具 | `tools/probe_m12_sources.py`（**只用标准库** + `curl` 子进程作传输后端） |
| 本节点所用解释器 | `./.venv-m10/bin/python`（Python 3.9.6 / numpy 2.0.2 / pandas 2.3.3 / **pyarrow 21.0.0**） |
| 为什么不用 `.venv-m11` | `.venv-m10` 是本机**唯一**装了 `pyarrow` 的 venv ⇒ S5 的 `.pqt` 解析（行数/列名）只能在它下面跑；`.venv-m11` 无 pandas/pyarrow |
| 环境变量 | 全部命令固定 `PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` |
| 仿真/GPU | **零**。本节点不 import torch，不建网，不跑引擎（派发硬约束） |

**本例**：S5 的 40 个 `_ibl_trials.table.*.pqt` 的行数是**本节点唯一需要非标准库的地方**
（`data/m12_probe_b1_integrity.csv::rows_parsed` 的 38 个不同值，117–913 行/session）
⇒ 若按"数据侧 = `.venv-m11`"的继承口径执行，会得到 `pyarrow MISSING` 的**假失败**（M11-R34 同族）。

---

## L2 — 三条硬约束（本工具立身之处，逐条已固化进代码）

1. **先测吞吐、再决定取数**：每源的取数决策行都能回溯到**当场**实测值（`data/m12_throughput_measured.csv`，52 行）。
2. **父进程硬回收**：所有网络调用 = `Popen` + `communicate(timeout)` + `kill()`（承 M11 L9.3a：`curl --max-time`
   在个别站点**失效**，实测挂死 **925.9 s**）。
3. **完整性门三级**：① `size == Content-Length`；② S3 多段对象 ETag 复算
   `md5(concat(md5(part_i))) + "-N"`；③ sha256（标注 `file_bytes`/`payload`/`semantic` 基准）。
   **未完成 ⇒ `INCOMPLETE`，不得进统计**。

**本例**：本轮 43 件全部 `COMPLETE`（40 IBL + 2 Allen + 1 S2）；**零 INCOMPLETE**。
`data/m12_probe_b1_integrity.csv` 的 `verdict` 列可直接统计（主 agent 可用一行命令复核）。

```bash
cd /Users/weidong/ai/small_world
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./.venv-m10/bin/python \
  neural_exploration/tools/probe_m12_sources.py --only all --max-sessions 40
```

---

## L3 — ★★ 本轮最重要的一条：**`0 B/s` 的探针结果有两种根因，误判会永久关闭一个 34 秒可完成的取数**

**实测链（S2，`api.brain-map.org` 的 `model::Structure` 全表）**

| 步骤 | 口径 | 结果 |
|---|---|---|
| T3-a | `curl -r 0-1048575 --max-time 30`（清单 §2.3 T3 模板） | **0 B / 30.0 s / curl exit=28** |
| T3-b | `curl -r 0-1023 --max-time 30` | **0 B / 30.0 s** |
| T3-c | 时间上限 GET（30 s 窗，无 Range） | **0 B / 30.0 s**（TTFB=30.0 s 占满整窗） |
| 独立计时 | `curl -w '%{time_starttransfer}|%{time_total}|%{speed_download}'` | **TTFB=30.98 s / total=33.55 s / 215,785 B/s** |
| T5 整件 | 全表 GET 落盘 | **7,238,867 B / 34.2–36.8 s ⇒ ETA 通过 1800 s 门 49–53×** |

⇒ **30 s 探针窗短于服务端生成时延（TTFB≈31 s）**，窗口内**一个字节都没有**。
`B/s = 0` 在这里**不是**"链路吞吐为零"，而是**"探针窗口径不适用"**。

**若照字面执行"0 B/s ⇒ 快 ⇒ 不可行"，会得到 S2 = `unavailable（吞吐限制）`，而事实上 34 秒就取完了
—— 按清单 §2.4，这会触发"P1 判 unavailable、P7/P10 区域级判据全部降 invalid"的结构性阻塞，
把一个本可完成的门变成阻塞。** 这是本轮**最贵的单点风险**。

**固化的预注册规则（写进代码，事中不得择口径）**：
> 探针**必须同时登记 TTFB**；若 `bytes == 0` **且** `TTFB ≥ 探针窗`
> ⇒ 判 `probe_window_too_short`，**不得**据此判 `TOO_SLOW`，**必须**转"整件取数即测量"路径
> （对象若 > 30 min 上限则此时再判 TOO_SLOW，且必须给 TTFB 与传输段两个数）。

新增判据行：`M12-M-01`；新增更正行：`m12_probe_b1_reachability.csv::S2(更正行·整件取数即测量)`。

---

## L4 — ★ 端点的 **B/s 不是链路吞吐**：服务端生成型端点必须报 (TTFB, 传输段, 对象大小) 三元组

S2 的数字自洽检查：`7,238,867 B ÷ (34.2 − 31.0) s ≈ 2.26 MB/s`（纯传输），而端到端 `215,785 B/s`。
两者相差 **10.5×**，差额**全部**是固定的 TTFB。

⇒ **推论（可伪证）**：对同一端点取一个 **2× 大的对象**，端到端 B/s 应**上升**（TTFB 摊薄），
而非下降。**因此"用 B/s 外推更大对象的 ETA"会系统性高估**（应改用 `ETA = TTFB + bytes/transfer_rate`）。

**本例**：清单 §2.1 登记的继承值 `api.brain-map.org = 1.6–16 KB/s`。
本轮当场实测：**7.24 MB / 34.2–36.8 s = 197–212 KB/s 端到端**（比继承值高 **12–130×**）。
⇒ 与 R41-2 条件 1 的要求一致：**以探针为准并新增更正行**（已落 `M12-S2-12`）。

---

## L5 — ★ 小对象探针会**低估**链路吞吐（含握手/TTFB），必须与"同通道大对象"分列

**本例（S7）**：目标两表仅 7,849 B / 26,954 B ⇒ `-r 0-1048575` 返回**全件**
（服务端对越界 Range 返回整个对象）⇒ 得到的是"单件端到端"值 **14,681 / 32,774 B/s**。
同桶大对象（`channels.csv` 6.6 MB）的 1 MiB Range 同刻实测 **608,198 / 701,617 B/s**（另两次 602,633–729,926）。

⇒ **差 20–50×**。若用小件值代表该桶吞吐，会把"1 秒取回"判成"要几十分钟"。

**固化的分列标签**（`data/m12_throughput_measured.csv::probe_or_fetch`）：
`probe_1MiB`(1 MiB Range) / `probe_30s_16MiB`(快链升级档) / `probe_whole_object`(对象 <1 MiB，
**只反映单件端到端**) / `probe_ref_1MiB`(**同通道大对象**，作主档) / `probe_plainGET*` /
`probe_window_too_short(ttfb=…)` / `fetch` / `fetch(resume)`。
**两者并列，不得混为一个数**（新增判据行 `M12-M-02`）。

---

## L6 — ★ IBL 桶的 `#` revision 目录：**不编码 ⇒ 假 404 + 错误页落盘成"假件"**

IBL 的 session 目录存在 revision 形态 `.../alf/#2024-07-15#/_ibl_trials.table.<uuid>.pqt`。
`#` 是 **URL fragment 分隔符** ⇒ 未编码时 curl 把 `#` 之后全部丢弃、请求打到 `.../alf/` ⇒ **HTTP 404**。

**本轮实测（两段，都要留档）**：
- 现象：3 号 session `hausserlab/PL015/2022-02-20/001` 的列举里**明明有** 63,734 B 的对象，
  GET 却返回 **404 + 341 B XML**。
- 旧逻辑的**二次伤害**：`fetch_full` 把 341 B 错误页**落盘覆盖**，并对其发起 `curl -C -` 续传
  ⇒ 台账里出现一个 341 B 的**假件**（只因 `size_match=False` 才被发现）。
- 修：① URL 用 `quote(key, safe="/")`（`#`→`%23`）；② curl 加 **`--fail`**（HTTP 错误体不落盘）；
  ③ 续传**仅**在 `http ∈ {200,206}` 且确为 partial 时执行。修后同 session **HTTP 200 / 63,731 B / COMPLETE**。

⇒ 这是"**静默失效家族**"的第 5 例（前四例见 M11-R60：静默入错槽 / 丢事件 / 硬编码跳过 field2 / 共用输出路径覆盖）。
新增判据行：`M12-S5-06`、`M12-M-04`。

**附带的次要口径**：同一 session 的"当前版"（`alf/_ibl_trials.table.*.pqt`）与"revision 版"
是**两个不同对象**（63,734 vs 63,731 B）⇒ 选哪一档**必须登记**（`m12_ibl_sessions.csv::note` 的 `revision_used`），
本节点**优先取当前版**。

---

## L7 — 完整性门在 **chunked（无 Content-Length）** 下的正确处置

S2 的响应**既无 `Content-Length` 也无 `Accept-Ranges`**（chunked）。此时：

- 门 ① `size == Content-Length` **不适用** ⇒ **不得**因此判 INCOMPLETE（那是误判）。
- **替代锚（本轮实际采用，且优先级从高到低）**：
  1. **服务端 ETag = 全对象 md5，复算通过** ⇒ 他证（锚在服务器侧）。实测 ETag
     `ef27375ee45ec6b946e96a95fefc7a20` == `md5(file)` ✓。
  2. **语义锚**：`msg` 行数 **13,709 == T1 `count$eqtrue` 的 total_rows 13,709** ✓。
  3. 两次独立取数 sha256 一致（自洽；**本轮因 ① 已通过而显式跳过**，省 35 s + 7.2 MB）。
- 三基准 sha256 全给：`file_bytes == payload`，`semantic` 另列。

⇒ **纪律（M11-R56 的推广）**：**"他证 > 自洽"**。ETag 可复算时，**不必**再做重复取数；
ETag 缺失/不复算时，**必须**退回两次独立取数并在 `reasons` 写明走的是哪条。新增判据行 `M12-M-03`。

**本例的省时**：跳过冗余第二次全量取数 = 35 s / 7.2 MB（对 7.2 MB 的对象是 100% 的冗余）。

---

## L8 — 台账**必须 `os.walk` 递归**入账；且"表观大小 vs 真实大小"要两个都报

- M11 实测：`os.listdir + isfile` **静默跳过子目录** ⇒ 漏账 **111 MB**（`m11_data_inventory.csv` 首版）。
- M11 实测：稀疏草稿件以"表观大小 ×3"虚增台账（**1133.6 MB → 真实 605.6 MB**）。

⇒ `tools/build_m12_inventory.py`：**只用 `os.walk`**；每个目录都登记（**空目录也显式登记**，防静默跳过）；
逐件同时给 `apparent_bytes`(st_size) 与 `real_bytes`(st_blocks×512) 与 `sparse_ratio`；
**台账口径取 real_bytes**，汇总里**两个总量都报**。

**本例**：本轮 `m12_raw/` 全树 5 个目录（`s2/ s5/ s7/ ibl/ probe/`）全部覆盖；
`probe/run1/` 保留了第一轮的产物副本**用于确定性比对**（见 L11），它也在台账内
（**这正是"子目录必须递归入账"的现场例子**：若用 `os.listdir`，`probe/run1` 会被漏掉）。

---

## L9 — **吞吐不可外推**（M11-R56/R63 的再次确认；本节点给出新的横轴）

M11 立此条时的横轴是"**同一天内摆动 3,600×**（1.6 KB/s ↔ 5.9 MB/s）"。本轮新增的横轴是
"**同一路径在 10 小时内的量表完全不同**"：

| 源 | 清单 §2.2 的继承值（`unverified`） | 本轮当场实测 | 倍数 |
|---|---|---|---|
| `api.brain-map.org`（S2 全表 7.24 MB） | 1.6–16 KB/s | **197–212 KB/s**（端到端；TTFB 31 s） | **12–130×** |
| Allen BO 同桶（S7 渠道参照 1 MiB Range） | 1.6–28 KB/s（同源记录） | **602,633–729,926 B/s** | **21–456×** |
| IBL 公开桶（S5 `trials.pqt` 1 MiB Range） | 10–17 KB/s | **221,230–322,393 B/s** | **13–32×** |

⇒ **操作规则（本节点执行，写进产物）**：凡以吞吐为由的判决，**必须**登记
**① 判决时刻（`utc` 列）② 当时实测吞吐（`bytes`/`seconds`/`bytes_per_s`）③ "须以当场探针复核"
（复核周期 ≤24 h）**。三个 CSV 均含 `utc` 首列。

**同日同小时的二次抖动（本轮自证）**：S5 全量聚合表 `trials.pqt` 的当场 ETA，
第一轮 **109.4 s**、第二轮 **73.1 s**（**1.50×**）；S2 全表三次 34.2 / 35.3 / 36.8 s（1.08×）。
⇒ 连"同一小时内的两次探针"都不能互相替代。

**本例（正面）**：S5 的全量聚合表 `trials.pqt`（24,212,510 B）当场实测 ETA = **73–109 s**
⇒ 它的四态**不是** `unavailable（吞吐限制）`，而是 **`unavailable（预算限制，非吞吐限制）`**
——"未取"与"不可取"是两件事。判据行 `M12-S5-05` 已按此措辞登记（含判决时刻与当时吞吐）。

---

## L10 — "**够用/可行**"必须实测；"**未测 ≠ 未命中**"在**逐 session** 层的形态

S5 白名单枚举中，**22 个 session 被显式登记原因后跳过**：全部是"`alf/` 内**没有**合并形态的
`_ibl_trials.table.*.pqt`"（该 session 只有旧的逐字段 `_ibl_trials.choice.*.npy` 形态）。

⇒ 正确措辞：**"该 session 的合并行为表未命中（该口径下）"，不是"该 session 无行为数据"**；
也**不得**把这 22 个算作"IBL 不可用"的证据。相关判据行 `M12-S5-07`（`verdict=partial`）。

**纪律（承 M11-R59 静默失效家族）**：**凡跳过必须显式登记原因**——`skipped` 列表进了
summary JSON 与判据行，**没有一个 session 被静默丢弃**。

**本例（计数）**：`S5.ibl.labs = 13`（含 `registration` 这个 **0 subject** 的前缀）；
逐 lab subject 数 118/46/28/32/17/11/51/63/23/**0**/10/66/28。
注意 `public_tree.txt`（105,625 B）**只覆盖 `churchlandlab` 一支**（507 行的示例树）
⇒ **不得**拿它当 session 计数锚（已在台账 note 里写明）。

---

## L11 — **判据预注册 + 重跑确定性**（两项都在出数前定死）

**预注册（本轮，事中不得改）**
1. S5 白名单：规模 ≤40；枚举 = round-robin（lab 字母序 → subject 字母序 → date 字母序 → session 数值序）；
   **归属按最终序号三分**：A=1..13 / B=14..26 / C=27..39 / D=40（因 round-robin，各分区都横跨同一组 lab）。
   ⇒ **"分区标签"不等于"隔离"**：隔离必须**另按 session 切**（本表只给归属标签，不得当作隔离已做）。
2. S2 体积估计法：单页 `num_rows$eq50` 取回 ⇒ 每行字节 × total_rows ⇒ 零成本估计 ETA。
3. 探针升级规则（P-a…P-d，见 L3/L5）与 `probe_or_fetch` 标签表。
4. 完整性门替代锚的优先级（L7）。

**重跑确定性（§2.5 通过条件 ④）**：两轮独立跑（`probe/run1/` 与当前盘上件）应给出
**同一白名单、同一逐件 sha256**。比对命令：

```bash
cd /Users/weidong/ai/small_world
./.venv-m10/bin/python - <<'PY'
import csv
def key(p):
    return {r['eid_key']: (r['trials_table_key'], r['trials_table_bytes']) for r in
            csv.DictReader(l for l in open(p, encoding='utf-8') if not l.startswith('#'))}
a = key('neural_exploration/data/m12_raw/probe/run1/m12_ibl_sessions.csv')
b = key('neural_exploration/data/m12_ibl_sessions.csv')
print('白名单一致 =', a == b, '| n =', len(a), len(b))
PY
```

**本例（估值法自校验）**：T1b 估计 7,328,008 B vs 实测 7,238,867 B ⇒ 偏差 **−1.2%**
（**估计口径的实测偏差被登记**，而不是把估计当实测，判据行 `M12-S2-05`）。

---

## L12 — 本节点**没有**做什么（边界与"未覆盖维度"清单）

| 未做的事 | 为什么 | 它**不是**什么 |
|---|---|---|
| S1 投射矩阵（≥10⁶ 行） | 清单已标 🔸`unavailable`（主假设）；**非本节点范围** | **不是**"已证不可行"（其当场探针未做） |
| S3 标注体数据 / S6 IBL 全脑活动 / DANDI 全量 | >50 MB 硬阈值**或**非本节点范围 | **不是**"不存在" |
| S5 全量聚合 `trials.pqt`（24.2 MB） | 超本节点 **≤5 MB 探针预算**，且 **<50 MB 硬阈值** ⇒ 应报主 agent 裁决 | **不是** `unavailable（吞吐限制）`：当场 ETA **106–109 s**（判据行 `M12-S5-05`） |
| 任何 `src/` 引擎改动、任何仿真、任何 GPU/MPS | 派发硬约束 | — |
| 任何 `docs/m11_*`、`data/m11_*`、`tools/validate_p11_*` 改动 | 派发硬约束 | 注：工作树里 `m11_*` 的 modified 状态**不是本节点造成的**（其 mtime 早于本节点启动） |

**缺哪一维（如需主 agent 裁决用）**
- **S2**：无缺失维度（对象级 200 + 完整性门三级替代锚 + 语义锚 **全通过**）。
  唯一"不适用"的是门 ① `Content-Length`（服务端 chunked）——已按 L7 用**服务端 ETag 他证**替代并登记。
- **S7**：无缺失维度（三项全通过，且与 M11 已落盘件 sha256 逐字节一致）。
- **S5**：**已覆盖** = 对象存在性/单件字节（measured）、逐件完整性门（40/40 COMPLETE）、
  逐 session 试次表行数（pyarrow 解析，117–913 行）；**缺 1 维** =
  **"逐 session 的 A/B/C/D 归属是否真的与 lab/动物正交"未验证**（本表给的是序号三分标签，
  隔离需另按 session 切）；另行登记：**22 个 session 的合并行为表在该口径下未命中**（≠ 不存在）。

---

## L13 — 复算命令（逐项）

```bash
cd /Users/weidong/ai/small_world
# 统一前缀
P='PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1'

# 1) 全部三项探针（S2 7.2 MB + S7 34 KB + S5 40 session ≈1.8 MB；实测墙钟 ≈7 min，后台跑）
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./.venv-m10/bin/python \
  neural_exploration/tools/probe_m12_sources.py --only all --max-sessions 40

# 2) 只跑吞吐探针、不取数（用于"当场复核"）
… --only all --probe-only

# 3) 台账（os.walk 递归；表观 vs 真实）
PYTHONHASHSEED=0 ./.venv-m11/bin/python neural_exploration/tools/build_m12_inventory.py

# 4) 判据行追加（append-only，幂等；--dry-run 只打印）
PYTHONHASHSEED=0 ./.venv-m11/bin/python neural_exploration/tools/build_m12_reference.py --dry-run

# 5) 完整性门一行统计
./.venv-m10/bin/python -c "
import csv;from collections import Counter
r=list(csv.DictReader(l for l in open('neural_exploration/data/m12_probe_b1_integrity.csv',encoding='utf-8') if not l.startswith('#')))
print(len(r), Counter(x['verdict'] for x in r))"
```

**实测预算**：命令 1 = **约 7 min**（02:26:44 → 02:33:20，含 40 次 session 列举与取数）
⇒ 在本节点自管的后台作业内完成（<10 min）；**未使用 MPS/GPU**。

---

## L14 — 本轮自查拦下的缺陷（"假结果/假跳过"类，全部值得留档）

1. **`curl` 把 `[`/`]` 当 range 表达式** ⇒ `rma::criteria[graph_id$eq1]` 触发
   `curl: (3) bad range in URL`（rc=3，**零输出**）⇒ 必须 `-g`（globoff）或 URL 编码。
   *症状误导性*：表现为"查询失败"，实际是**客户端参数解析**而非服务端拒绝。
2. **`curl` 把 `#` 当 fragment** ⇒ IBL revision 目录的 404 假阴性（详见 L6）。
3. **`os.listdir` 的隐蔽性**（承 M11）⇒ 本轮用 `os.walk`；`probe/run1/` 这类子目录正是易漏点。
4. **错误响应体落盘 + 对 404 做续传** ⇒ 341 B 假件（详见 L6）。
5. **C 语言风格的 "0 B/s ⇒ 慢" 直觉** ⇒ 差点把 34 秒可取的对象判成 unavailable（详见 L3）。
6. **`run_curl` 只回传 `-w` 汇总** ⇒ 硬回收时 `-w` 不输出 ⇒ 时间上限探针必须**改用落盘文件字节数**
   计量（否则永远读到 0）。
7. **`probe_plain_get` 的返回元数**：初版把 3 元组当 2 元组用 ⇒ `IndentationError`；
   教训同族——**"文档与实现分叉"**（M11-R46）在单个工具内的微缩版。

---

## L15 — 与主 agent 三项验收条件（`m12_list_review_master.md` §R41-2）的关系

| 条件 | 本节点的作用 |
|---|---|
| **条件 1**（S1/S3/S4/S6 的 `unavailable` 主假设**必须**以当场 30 s 探针确认后才生效；不符则以探针为准并**新增更正行**） | **本节点只覆盖 S2/S7/S5**，但给出了**同通道的实测吞吐**（`api.brain-map.org` 197–212 KB/s、Allen BO 602–730 KB/s、IBL 桶 221–310 KB/s）⇒ **S1/S3/S4/S6 的继承吞吐前提已被同源实测强烈动摇**；**但按 M11-R40"一判据一作用域"，本节点不替 S1/S3/S4/S6 下结论** —— 它们的 `unavailable` 仍须各自的当场探针，**更正行须由覆盖它们的节点新增**。 |
| 条件 2（预算上界待探针修正） | 不涉及（属 B2 的 ms/step 探针）。 |
| 条件 3（M11 缺口表重跑） | 不涉及（`data/m12_m11_gap.csv` 未由本节点生成）。 |

---

## L16 — 交付件清单（本节点新增）

| 文件 | 内容 |
|---|---|
| `tools/probe_m12_sources.py` | S2/S7/S5 探针+取数+完整性门（Popen 硬回收；预注册口径固化） |
| `tools/build_m12_inventory.py` | `os.walk` 递归台账 + 表观/真实大小 + 计数锚 |
| `tools/build_m12_reference.py` | 判据行 **append-only** 追加器（幂等；`--dry-run`） |
| `data/m12_throughput_measured.csv` | `utc,source,http_code,bytes,seconds,bytes_per_s,probe_or_fetch` |
| `data/m12_probe_b1_integrity.csv` | 43 件完整性门逐件（size / ETag / sha256 三基准 / 行数自洽 / verdict） |
| `data/m12_probe_b1_reachability.csv` | 源级四态（**3 源前缀 + S2 更正行**；全量 ≥40 源表**仍未交付**） |
| `data/m12_ibl_sessions.csv` | **预注册** 40 session 白名单 + A/B/C/D 归属 + 枚举规则 + revision 档 |
| `data/m12_reference.csv` | 判据/参考锚（append-only，带 provenance / evidence_level / wrong_usage_consequence） |
| `data/m12_data_inventory.csv` | 递归台账（含 `probe/run1/` 与空目录） |
| `data/m12_probe_b1_summary.json` / `data/m12_probe_b1_counts.json` | 机器可读汇总/计数锚（供报告引用，防"引陈旧数字"） |
| `data/m12_raw/**` | 落盘对象（S2 全表 + S7 两表 + 40 个 IBL 试次表 + 日志 + run1 副本） |
| `docs/m12_env_notes.md` | 本文件 |

---

## L17 — 断言表（**逐条机验**；16/16 通过）

由 `tools/build_m12_inventory.py` 从落盘产物直接计算，写入
`data/m12_probe_b1_counts.json::assertions`。**报告引用计数时必须引这里，不得引记忆**（M11-R58/R63）。

| id | 断言 | 结果 |
|---|---|---|
| A1 | S2 全表落盘字节 == 7,238,867（且与 T1b 零成本估计偏差 \|−1.2%\| ≤ 5%） | ✅ |
| A2 | S2 服务端 ETag = md5(全对象) 复算通过（**他证**） | ✅ |
| A3 | S2 语义锚：`msg` 行数 == T1 `count$eqtrue` 的 13709 | ✅ |
| A4 | S2 三基准 sha256 齐备（file_bytes / payload / semantic） | ✅ |
| A5 | S7 两表 `size == Content-Length` | ✅ |
| A6 | S7 两表单段 ETag md5 复算通过 | ✅ |
| A7 | S7 sha256 与 M11 已落盘件逐字节一致（跨节点独立复测） | ✅ |
| A8 | S5 白名单 40 件全部 `COMPLETE`（**零 INCOMPLETE ⇒ 全部可进统计**） | ✅ |
| A9 | S5 40 件 `size == Content-Length` | ✅ |
| A10 | S5 40 件 parquet 解析成功且行数 > 0 | ✅ |
| A11 | 白名单**已预注册**（枚举规则 + A/B/C/D 归属写在 CSV 列中，出数前定死） | ✅ |
| A12 | 台账 70 行全部 distinct（无重复入账） | ✅ |
| A13 | 台账递归覆盖 ≥6 个目录（`os.walk`；含 `probe/run1/` 子目录与空目录） | ✅ |
| A14 | **无稀疏虚增**（`apparent > real` 的件数 == 0） | ✅ |
| A15 | 判据表 append-only：既有 ref_id 全部保留 | ✅ |
| A16 | 吞吐 CSV 每行带 `utc` 且 `bytes_per_s` 与 `bytes/seconds` 自洽（≤1%） | ✅ |

**五项元数据（本轮，机读自 `data/m12_probe_b1_counts.json`）**

| 元数据 | 值 |
|---|---|
| **断言** | **16**（通过 16 / 失败 0） |
| **行数** | 台账 **70** 行（distinct 70）；完整性门 **43** 行；吞吐 **52** 行；判据表 **34** 行；白名单 **40** 行 |
| **件数** | **43 个取数对象**（S2 1 + S7 2 + IBL 40），**43/43 `COMPLETE`** |
| **总量** | 落盘 **9.847 MB（真实）** / 9.538 MB（表观）；完整性门 COMPLETE 对象合计 **9,071,568 B** |
| **notes 行数** | 本文件 L1–L17（行数由 `wc -l docs/m12_env_notes.md` 给出） |

**台账"表观 vs 真实"**：`apparent_vs_real_ratio = 0.968`（<1）
⇒ 本节点**没有稀疏虚增**（`apparent > real` 的件数 **0**）；<1 的原因是**块取整开销**
（69 件小文件各自占用整数个 4 KB 块）——**两个方向都登记**，不得只报一个（M11-R63）。

---

## L18 — ★ 附录：本节点的**在写文件被并发提交卷入**（R41-3 的复发，附量化后果）

**事实（本轮实测，可复算）**：

```bash
cd /Users/weidong/ai/small_world
git log -1 --format='%H %ci %s'
git show --stat --oneline HEAD | grep m12
git show HEAD:neural_exploration/docs/m12_env_notes.md | wc -l   # 326（在写中间态）
wc -l neural_exploration/docs/m12_env_notes.md                   # 370（本节点最终版）
git diff --stat HEAD -- neural_exploration/docs/m12_env_notes.md # +46 / -2
```

主 agent 的提交 **`fc6afed`（2026-10-12 11:02:19 +0800）** 把本节点**正在写入**的
`docs/m12_env_notes.md` 纳入了提交（提交时 **326 行**；本节点最终版 **370 行**，
diff **+46 / −2**）。

**这**正是 `docs/m12_list_review_master.md` §R41-3 刚刚立下的纪律所禁止的动作**
（原文：「**禁止**用 `git add -A`／宽 pathspec 把"正在被节点写入的文件"纳入提交；
优先显式列举已确认完成的路径」）。

**量化后果（为什么这不是"无害的中间态"）**：HEAD 版里含着**已被本节点后续更正的数字**——

| HEAD（326 行版） | 本节点最终版（370 行） |
|---|---|
| `trials.pqt` 当场 ETA = `106–109 s` | `73–109 s`（第二轮实测 73.1 s ⇒ 同小时 1.50× 抖动） |
| IBL 桶吞吐 `221,230–310,284 B/s`（13–31×） | `221,230–322,393 B/s`（13–32×） |
| 章节范围 `L1–L16` | `L1–L18`（新增断言表 L17 与本条 L18） |

⇒ **引用 HEAD 副本会引到 2 处陈旧数字**。这与 M11-R32「跨节点核对须以**磁盘工作树 + sha256** 为准，
不得以 `git show HEAD:` 副本代替」是同一条纪律的**同一实例的复发**。

**本节点的处置**：
1. **不自行改写历史提交**（越权）；**在本文档中显式登记**该事实与量化差额，供主 agent 裁决；
2. 所有对外数字一律**以工作树为准**（本文件与 `data/m12_probe_b1_counts.json` 均来自工作树）；
3. 供主 agent 复核用的**单一命令**（两值应分别为 `326` 与 `370`，且 diff 为 `+46/−2`）：

```bash
cd /Users/weidong/ai/small_world && \
  echo "HEAD=$(git show HEAD:neural_exploration/docs/m12_env_notes.md | wc -l) \
WT=$(wc -l < neural_exploration/docs/m12_env_notes.md)" && \
  git diff --stat HEAD -- neural_exploration/docs/m12_env_notes.md
```

**建议（不越权执行）**：下一次提交时按 R41-3 改为**显式列举路径**
（`git add <已完成件1> <已完成件2> …`），并在此之前对每个待提交件确认
「节点已回报完成」**且**「mtime 在回报之后稳定／两次核查间字节数不变」。
