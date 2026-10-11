#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M10-B1 · P6 判决表 schema 规范化（自查缺陷第 6 例修复）

背景（本轮自查拦下）：
    `data/m10_p6_judgement_table.csv` 在 RF-25 → RF-29 五轮追加中出现
    **四种列数并存**（22 / 23 / 24 / 26），而表头恒为 23 列。后果：
      1) RF-29 的 τ CI 三列没有列名，落在表头之外 ⇒ 下游按名解析
         (`tau_lo95_s` / `tau_hi95_s` / `tau_gap_in_se`) 取不到值；
      2) RF-25/27/28 各行 `sha16` 列被写成字面量 "measured"，`evidence_note`
         被挤出表头之外 ⇒ 证据链字段错位；
      3) RF-29 行把 **bootstrap 中位数**写进 `tau_ar1_s`，而其余行该列是
         **点估计** ⇒ 同列混装两种口径，违反 L40 律①。

本脚本只 **重排列 + 补列名 + 回填权威值**，**不重算任何统计量**：
    tau_ar1_s                ← 同臂 **点估计**（0.914 s 箱 AR(1)，取自同臂 measured 行）
    tau_median_boot_s        ← `data/m10_p6_tau_bootstrap_CI.json`（权威来源）
    tau_lo95/hi95/gap_in_se  ← 同上 JSON

并执行三条断言（幂等，可反复重跑）：
    A. 表头 == 27 列规范 schema；
    B. 每一行列数 == 27；
    C. L40 律③：**点估计必须落在自助 CI 内**（两个独立估计器断言一致）。

用法：
    cd /Users/weidong/ai/small_world
    PYTHONHASHSEED=0 MPLBACKEND=Agg ./.venv-m9/bin/python \
        neural_exploration/tools/normalize_m10_p6_table.py
"""
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # neural_exploration/
DATA = ROOT / "data"
TABLE = DATA / "m10_p6_judgement_table.csv"
SNAP = DATA / "m10_p6_judgement_table_schema_v2_snapshot.csv"
BOOT = DATA / "m10_p6_tau_bootstrap_CI.json"

BASE_COLS = [
    "level", "lam", "n_frames", "status", "tau_ar1_s", "tau_criterion_status",
    "tau_window_lo", "tau_window_hi", "ar1_phi", "ar1_holdout_ll_per_frame",
    "ar1_resid_whiten", "dwell_Q_s", "dwell_H_K2_s", "occ_low", "occ_high",
    "trans_hz", "p90_over_med", "ks_exp_p", "internal_consistency", "c3_shape",
    "bic_K2_K3", "sha16", "evidence_note",
]
CI_COLS = ["tau_median_boot_s", "tau_lo95_s", "tau_hi95_s", "tau_gap_in_se"]
SCHEMA = BASE_COLS + CI_COLS
NCOL = len(SCHEMA)

HEADER = [
    "# M10 P6 判决表（RF-21 定稿；**RF-29 schema 规范化 v2**）——主判据 = AR(1) τ（0.914 s 箱 + 5 帧平滑 + z，B4 口径）",
    "# 成像基准 τ = 37.180 s（全记录；42.332 s 为前 80% 段值已更正）；窗 = [9.4, 146.3] s（±3.94×）；φ_gate = 0.9073604015624624",
    "# 🔴 **样本量门（RF-18/RF-21）**：n<65 ⇒ 全部状态结构 invalid；形状判据需 ≥130；",
    "#    **AR(1) τ 判据需 n_frames ≥ 300（T ≥ 275 s）** ⇒ 本轮 T=120 s 的 τ 点估计**仅描述性**；",
    "#    **RF-29 起 CI 可判**：块自助法（moving-block, block=15 帧, B=2000）CI95 **上界 < 窗下界 9.4 s ⇒ fail**；CI 跨越 ⇒ unavailable。",
    "#    BIC/K 需 ≥500 帧，否则 unavailable；内部一致性 |dwell−1/(2λ)|/dwell ≤1.0 否则 invalid（L40 律③）",
    "#    ⚠️ **口径分离（L40 律①）**：`tau_ar1_s` = 同臂**点估计**；`tau_median_boot_s` = **自助中位**。",
    "#       τ=−dt/ln φ 对 φ 凸 ⇒ median(τ) < τ(median φ)，两者**不得混用**；",
    "#       差异最大者 M-6 λ×4：点估计 1.8244 s vs 自助中位 3.0642 s（1.68×，分布强右偏）⇒ 该臂仅描述性。",
    "# c4（机制归属）**不得用零滞后相关**：须 去趋势 |r|≥0.30 且 lead-lag 峰值 lag>0（RF-15/RF-21(3)）；",
    "#    本轮各臂无慢变量轨迹落盘 ⇒ c4 = unavailable（不代填）",
    "# 诚实上限：Ljung-Box 三口径全部拒绝白噪（p≈0）⇒ AR(1) 仅一阶近似，τ 只作粗粒度指标",
    "# chunk 路径在重过引擎门 (e)+(f) 前**不得用于任何判据**（判据一律 delivery=step）",
    "# 自查缺陷第 6 例：本文件此前四种列数并存、CI 三列无列名 ⇒ 由 tools/normalize_m10_p6_table.py 规范化（仅重排，未重算）",
]


def load_table():
    rows = list(csv.reader(open(TABLE, newline="", encoding="utf-8")))
    data = [r for r in rows if r and not r[0].startswith("#")]
    return data[0], data[1:]


def repair(row):
    """按错位量修复为 27 列。返回 (new_row, tag)。错位量唯一，故可断言。"""
    n = len(row)
    if n == 23:                                    # 规范行：原样 + 4 空 CI 列
        return row + [""] * 4, "ok23"
    if n == 22:                                    # RF-27 行：idx6.. 起整体左移 1
        fixed = row[:6] + [""] + row[6:]
        assert len(fixed) == 23, len(fixed)
        return fixed + [""] * 4, "shift+1"
    if n == 24:                                    # RF-25/28 行：多写 1 列 ⇒ sha16 被挤错位
        tail = [row[22], row[23]]
        note = "；".join([t for t in tail if t])
        fixed = row[:21] + [""] + [note]
        assert len(fixed) == 23, len(fixed)
        return fixed + [""] * 4, "sha16-fix"
    if n == 26:                                    # RF-29 CI 行：idx22/23/24 = CI 三元组(lo,hi,gap)
        note = row[25]
        fixed = row[:22] + [note]                  # 补回 evidence_note
        assert len(fixed) == 23, len(fixed)
        # 中位槽原本被错放进 tau_ar1_s，此处补空槽占位，由调用方按 JSON 权威值回填 23..26
        return fixed + [""] + row[22:25], "ci26"
    raise AssertionError("unexpected column count %d for %s" % (n, row[:2]))


def verify():
    """A/B/C 三条断言（幂等）。返回 data 行。"""
    h, b = load_table()
    assert h == SCHEMA, "header != schema v2: %s" % h
    print("[OK] A. 表头 ncol=%d == 27（schema v2）" % len(h))
    print("[OK]    CI 具名列 =", [c for c in h if c in CI_COLS])
    bad = [i + 1 for i, r in enumerate(b) if len(r) != NCOL]
    print("[%s] B. 行数=%d，列数不一致行数=%d" % ("OK" if not bad else "FAIL", len(b), len(bad)))
    assert not bad, "ragged rows: %s" % bad
    ix = {c: h.index(c) for c in ("level", "lam", "tau_ar1_s", "tau_median_boot_s",
                                  "tau_lo95_s", "tau_hi95_s", "tau_gap_in_se")}
    print("\n%-6s %-22s %12s %12s %10s %10s %6s %8s" % (
        "level", "lam", "tau_point", "tau_boot_med", "ci_lo", "ci_hi", "inCI", "gap_SE"))
    n_in = n_ci = 0
    for r in b:
        if r[ix["tau_lo95_s"]]:
            pe = float(r[ix["tau_ar1_s"]])
            lo, hi = float(r[ix["tau_lo95_s"]]), float(r[ix["tau_hi95_s"]])
            ok = lo <= pe <= hi                    # C. L40 律③
            n_ci += 1
            n_in += int(ok)
            print("%-6s %-22s %12s %12s %10.3f %10.3f %6s %8.1f" % (
                r[ix["level"]], r[ix["lam"]][:22], r[ix["tau_ar1_s"]][:12],
                r[ix["tau_median_boot_s"]][:12], lo, hi,
                "OK" if ok else "**FAIL**", float(r[ix["tau_gap_in_se"]])))
    print("\n[%s] C. L40 律③ 一致性：点估计落在自助 CI 内 %d/%d 臂" % (
        "OK" if n_in == n_ci else "FAIL", n_in, n_ci))
    assert n_in == n_ci, "点估计落在 CI 外的臂数=%d ⇒ 两估计器不一致，须复核" % (n_ci - n_in)
    return b


def main():
    if not TABLE.exists():
        sys.exit("[FATAL] missing %s" % TABLE)

    hdr, body = load_table()
    if hdr == SCHEMA:
        print("[INFO] 已是 schema v2 —— 跳过重写，仅做 A/B/C 验证")
        verify()
        return
    assert hdr == BASE_COLS, "unexpected base header: %s" % hdr

    # 修复前留快照（仅在任何写操作之前执行一次）
    if not SNAP.exists():
        shutil.copy2(TABLE, SNAP)
        print("[SNAPSHOT]", SNAP.name)

    boot = json.load(open(BOOT, encoding="utf-8"))
    bmap = {(r["level"], float(r["lam"])): r for r in boot["rows"]}

    # 点估计索引：同臂 measured 行（23 列规范行）
    point = {}
    for r in body:
        lam = r[1].strip()
        if lam in ("1.0", "4.0") and len(r) == 23 and r[4] not in ("", "—"):
            try:
                point[(r[0], float(lam))] = float(r[4])
            except ValueError:
                pass
    assert len(point) >= 8, "点估计索引不足 8 条：%s" % point

    out, tags = [], {}
    for r in body:
        new, tag = repair(r)
        tags[tag] = tags.get(tag, 0) + 1
        if tag == "ci26":
            lam = 1.0 if "×1" in new[1] else 4.0
            key = (new[0], lam)
            src = bmap[key]
            pe = point.get(key)
            if pe is not None:                     # 口径分离：点估计归 tau_ar1_s
                new[4] = repr(pe)
            new[23] = repr(src["tau_med_s"])
            new[24] = repr(src["tau_lo95_s"])
            new[25] = repr(src["tau_hi95_s"])
            new[26] = repr(src["gap_in_se"])       # JSON 键名为 gap_in_se
        assert len(new) == NCOL, (len(new), new[:3])
        out.append(new)

    with open(TABLE, "w", newline="", encoding="utf-8") as f:
        for c in HEADER:
            f.write(c + "\n")
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(SCHEMA)
        w.writerows(out)

    verify()
    print("修复标签统计：", tags)


if __name__ == "__main__":
    main()
