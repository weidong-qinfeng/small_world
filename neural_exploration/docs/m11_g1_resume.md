# M11 · G1 交接与恢复指令（自足：拿到本文件即可完成 G1 判定）

> 写于 **G1 全量矩阵运行中**（2026-10-12 13:00 启动，85 臂）。
> 本文件的目的：**任何人（或未来的会话）按此文件即可把 G1 走完**，无需重读全部验收记录。
> 更完整的裁定史见 `docs/m11_acceptance_master.md`（§3aa–§3cb / R31–R84）。

---

## 0. 一句话现状

M11 的 **G2 数据门 ✅ pass**、**G0 引擎门 ✅ pass 定稿**；**G1 的两项原始前置（钙域静默带 + P5 跑臂生产者）已就位**，环境阻塞（整机冻结）已定位并缓解，
预算已重算进包络（**★ + chunk = 8.2230 GPU-h ≤ 10**）；**G1 全量矩阵正在跑**，跑完即由脚本自动评估 ⇒ **G1 判定**。

---

## 1. 正在跑的作业（先查这个）

```bash
cd /Users/weidong/ai/small_world
ls neural_exploration/data/m11_p5_g1_runs/*.done | wc -l   # 已完成臂数（满 85）
ls neural_exploration/data/m11_p5_g1_runs/*.fail | wc -l   # 失败臂数（应为 0；>0 需查）
tail -5 /tmp/ma_g1_full.log                                 # 逐臂 [DONE] / 最终的 PLAN_EXIT / EVAL_EXIT / ALL_DONE
ps -eo pid,etime,time,command | grep run_m11_p5_arms | grep -v grep   # 是否还在跑
```

- **作业形态**：`nohup` + `caffeinate -i`，逐臂 **新进程**、**MPS 串行**；日志 `/tmp/ma_g1_full.log`。
- **进度口径**：`*.done` 的 **mtime** = 该臂完成时刻（**R39：判"在跑"须三重实测 —— 产物字节 + 最后写入时间戳 + 进程表**）。
- **若进程不在了但 `ALL_DONE` 没出现** ⇒ 作业被杀，重跑（**断点续跑是安全的**：驱动器见 `.done` 即 `[SKIP]`）。
  ```bash
  nohup bash /tmp/ma_g1_full.sh >/dev/null 2>&1 &   # 脚本仍在；会跳过已完成的臂
  ```

## 2. 跑完后自动做的事

作业脚本末尾会自动执行：

```bash
./.venv-m11/bin/python -m neural_exploration.tools.validate_p11_resting \
    --protocol-json neural_exploration/data/m11_p5_protocol_g1_star_n8192.json \
    --out-csv neural_exploration/data/m11_p5_resting_eval_g1.csv \
    --out-json neural_exploration/data/m11_p5_resting_eval_g1.json
```

**注意解释器口径（R34）**：**引擎侧 `.venv-m9`**（唯一有 torch）／**评估器侧 `.venv-m11`**（只有 numpy+h5py）。
**跨解释器 ⇒ 不做"逐位/BIT"断言**（numpy 1.26.4 vs 2.0.2 偏斜）。

若自动步骤因故没跑，手动补跑上面那条命令即可（**离线，不占 MPS**）。

## 3. G1 判定怎么读（**门 = (c) ∧ (e) ∧ 张力已定量**，**(d) 仅作 sanity**）

- **(c) 自持比**：零驱动/基准臂 ≥ **3 seed**（计划中 `zero_drive`/`baseline` 各 seed{0,1,2} ⇒ 应可判）。
- **(e) E/I 平衡**：`ei_ratio = |I_inh|/|I_exc|` ∈ **[0.5, 3.0]**（**电流积分由引擎实测注入**，不由发放率反推）。
- **张力**：须**已定量识别**（`n_both_in_band` ≥3 剂量点 + 两条 Spearman）。
- **(a)(b)**：与目标带比较 —— **(b) 钙域静默带 `[0.4800, 0.4900]`**（**per-ROI 代理口径**；全视野口径 **禁用**）；
  ⚠️ **带宽下界 ≥3×（比值口径）**（R52）⇒ **(b) 须以比值/上界形式使用，不得当"精确落入带内"判定**；
  且须随带引用四条限制（代理非 ROI / `thin` 样本 / 电生理仅量级 / 观测窗）见 `docs/m11_acceptance_master.md` §3az。
- 评估器若判 `unavailable`，**那是设计**（"未测 ≠ 未命中"），**不是失败** —— 须先补对应 role/seed 再判。

## 4. 不得违反的纪律（否则结果不可采信）

| # | 纪律 | 来源 |
|---|---|---|
| 1 | **跑完前不得改写** `data/m11_p5_plan_g1.json`（它是运行输入 + 审计锚）。带 `units` 的等价版本在 `/tmp/g1_plan_units.json`（sha256 `3f690d4d…`），**跑完后再落盘并重记 sha256** | R84 第 37 条 |
| 2 | **工具冻结**：`tools/run_m11_p5_arms.py` sha256 `81f29d528f554ca2`；改动后须重跑该臂集 | R48 |
| 3 | **生产口径 = `chunk`／判据口径 = `step`**；报告须登记"孪生臂校核通过（两端点 `ndiff=0/200000`、sha256 相同）" | R78 |
| 4 | **`chunk` 的孪生校核只覆盖"本工况 × 两个剂量端点"**；**若 N/dose/T/机制档变更 ⇒ 须在新工况重做** | R78 §⑤ |
| 5 | **MPS 串行 + 整机独占**：跑臂期间其它节点不得并发跑同量级网络（8k/dose4 臂 RSS 达 **2.5 GB**） | R55/R82 |
| 6 | **长作业监控须"低频、极短、只读"**；**禁止 `sleep` 轮询**（观测者效应可量化：有效占空曾低至 10%） | R82 第 35 条 |
| 7 | **判据之间禁止连带降级**：某判据输入不可得 ⇒ 单列成行判其四态，**不得**据此下调别的判据 | R40 |
| 8 | **参照的双重资格**：参照须 ①可比 ∧ ②可信；"与带缺陷参照对齐" **≠** 正确性证据 | R32 |

## 5. 若要加速（唯一有效手段）

**最小化或关闭本 GUI 的浏览器标签页** —— 实测争用者为 **Chrome（115% CPU）+ `dsh web`（47%）**，与 MPS 共用 Metal/GPU；
解除后有效占空可由 ~10% 回到接近 100% ⇒ 矩阵墙钟从 ~1.5 天降到 **~8 小时**。
（Spotlight 一侧已缓解：`neural_exploration/{,data/}` 与仓库根已放 `.metadata_never_index`，实测冻结差额 **88.3 min → 6 s**。）

## 6. 跑完之后的下一站

**G1 通过 ⇒ 启动 S 慢状态门**（判据形态已定稿：`data/m11_s_gate_criteria_prereg.csv` **S-1…S-13**；
目标带已双源互证：同源 **τ 8–9 s（未平滑原生帧率）**、H 法 low dwell **3.2–6.7 s**、占用 **0.61–0.64**；
ZAPBench 26.5 s **禁用**）。**S 门未过 ⇒ P4/P7–P16 不得定稿**（门链）。
