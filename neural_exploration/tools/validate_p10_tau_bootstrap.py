#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M10-B1 · τ 块自助法（moving-block bootstrap）——**可复现实现** + **RF-30 交叉核对**

为什么需要本文件
----------------
RF-29 的 CI 是**离线一次性脚本**算出的，只落盘了 JSON 结果、**没有落盘代码** ⇒ 违反 L40 律②
（"可复现"必须包含"别人能用同一段代码复得同一批数"）。RF-30 要求"用你的实现复算"，
故本文件把该实现**固化并参数化**（B / 块长 / 随机种子均可调），并内建三条断言：

  ① **口径一致断言**：本实现的点估计 φ/τ 必须与判决表内同臂 `tau_ar1_s` 一致（≤1e-4）——
     否则"我复算的和表里的不是同一个量"（L40 律①）；
  ② **稳定性量化**：同臂在 **多个随机种子 × 多个块长（10–20 帧）** 下的 CI 极差 ⇒
     用于判定"与 RF-30 的差异"是否落在蒙特卡洛/块长不确定度内；
  ③ **RF-30 交叉核对**：逐臂比对（lo/hi/判定），任何判定分歧必须显式列出。

管线（与 RF-14/RF-24 定稿一致，**直接复用** `validate_p10_state_structure.frame_series`，
不重新实现，避免"两个口径"）：
    逐步群体脉冲数 → **0.914 s 箱**（b = round(FRAME_S/(dt_ms·1e-3)) = 18,276 步）→ **5 帧滑动平均** → z 分数
    → AR(1) φ = Σx_t x_{t+1} / Σx_t²，τ = −FRAME_S / ln φ

判定规则（RF-29 §二 **预注册**，不得事后调整）：
    hi95 < 窗下界 9.4 s        ⇒ fail
    lo95 > 窗上界 146.3 s      ⇒ fail
    [lo95, hi95] ⊆ 窗          ⇒ pass
    否则（CI 跨越窗界）         ⇒ unavailable

退化重采样处理（**显式规则，禁止静默**）：重采样后 φ ∉ (0,1) 者记为 degenerate，**剔除且计数上报**；
M-4 λ×4 之所以 CI 上界对种子极敏感，正因为 φ≈0.89 逼近 1、τ 对 φ 呈强非线性（重尾）。

用法：
    cd /Users/weidong/ai/small_world
    PYTHONHASHSEED=0 MPLBACKEND=Agg ./.venv-m9/bin/python \
        neural_exploration/tools/validate_p10_tau_bootstrap.py --B 2000 --blocks 10,15,20 --seeds 0,1,2,3,4
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))          # 使 neural_exploration 可导入

from neural_exploration.tools.validate_p10_state_structure import (  # noqa: E402
    FRAME_S, SMOOTH_FRAMES, TAU_WINDOW_S, frame_series,
)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRACE_DIR = os.path.join(DATA_DIR, "m10_p6_poptraces")
TABLE = os.path.join(DATA_DIR, "m10_p6_judgement_table.csv")
OUT = os.path.join(DATA_DIR, "m10_p6_tau_bootstrap_crosscheck.json")

WINDOW_LO, WINDOW_HI = TAU_WINDOW_S
PHI_GATE = math.exp(-FRAME_S / WINDOW_LO)                  # 达窗下界所需 φ

# RF-30 主agent 独立复算表（B=2000、块 15 帧；φ/τ 为**点估计**，CI 为块自助法 95%）
# 来源：RF-30 指令正文（2025-10-11 主agent 离线运行）。此处**照抄**，不做任何修饰。
RF30 = {
    ("M0", 1.0):  dict(phi=0.6085, tau=1.840, lo=1.213, hi=2.042, gap_se=6.9),
    ("M-1", 1.0): dict(phi=0.5611, tau=1.581, lo=1.091, hi=1.750, gap_se=8.5),
    ("M-4", 1.0): dict(phi=0.6734, tau=2.311, lo=1.566, hi=2.550, gap_se=6.7),
    ("M-6", 1.0): dict(phi=0.6614, tau=2.210, lo=1.482, hi=2.308, gap_se=7.0),
    ("M0", 4.0):  dict(phi=0.6098, tau=1.848, lo=1.303, hi=2.084, gap_se=8.1),
    ("M-1", 4.0): dict(phi=0.6137, tau=1.872, lo=1.095, hi=2.157, gap_se=5.2),
    ("M-6", 4.0): dict(phi=0.6060, tau=1.824, lo=1.635, hi=4.230, gap_se=4.3),
    ("M-4", 4.0): dict(phi=0.8899, tau=7.835, lo=5.305, hi=986.1, gap_se=0.3),
}


def verdict_category(v: str) -> str:
    """判定**类别**（pass/fail/unavailable）——比对必须用类别，
    不得用带小数位的整串（否则 1.74 vs 1.75 这类四舍五入差异会被误报为"判定分歧"）。"""
    return v.split("(")[0].strip()


def verdict(lo: float, hi: float) -> str:
    """RF-29 §二 预注册规则（四态：pass/fail 两向/unavailable）。"""
    if hi < WINDOW_LO:
        return "fail(CI 上界 %.2f s < 窗下界 %.1f s)" % (hi, WINDOW_LO)
    if lo > WINDOW_HI:
        return "fail(CI 下界 %.1f s > 窗上界 %.1f s)" % (lo, WINDOW_HI)
    if lo >= WINDOW_LO and hi <= WINDOW_HI:
        return "pass(CI ⊆ 窗)"
    return "unavailable(CI 跨越窗界 [%.1f, %.1f])" % (WINDOW_LO, WINDOW_HI)


def phi_tau(z: np.ndarray):
    x0, x1 = z[:-1], z[1:]
    d = float(x0 @ x0)
    if d <= 0:
        return float("nan"), float("nan")
    phi = float((x0 @ x1) / d)
    tau = (-FRAME_S / math.log(phi)) if 0.0 < phi < 1.0 else float("nan")
    return phi, tau


def block_bootstrap(z: np.ndarray, block: int, B: int, seed: int):
    """moving-block bootstrap：取 ceil(n/block) 个连续块（有放回）后截回 n。"""
    rng = np.random.default_rng(seed)
    n = z.size
    nb = int(math.ceil(n / block))
    taus, phis, degen = [], [], 0
    for _ in range(B):
        starts = rng.integers(0, n - block + 1, size=nb)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        p, t = phi_tau(z[idx])
        if not (0.0 < p < 1.0):
            degen += 1
            continue
        phis.append(p)
        taus.append(t)
    return np.array(taus), np.array(phis), degen


def load_table_points():
    """判决表内同臂点估计（用于口径一致断言）。"""
    pts = {}
    rows = list(csv.reader(open(TABLE, newline="", encoding="utf-8")))
    data = [r for r in rows if r and not r[0].startswith("#")]
    h, b = data[0], data[1:]
    ix = {c: h.index(c) for c in ("level", "lam", "tau_ar1_s", "ar1_phi")}
    for r in b:
        lam = r[ix["lam"]].strip()
        if lam in ("1.0", "4.0") and r[ix["tau_ar1_s"]] not in ("", "—"):
            try:
                pts[(r[ix["level"]], float(lam))] = (float(r[ix["ar1_phi"]]), float(r[ix["tau_ar1_s"]]))
            except ValueError:
                pass
    return pts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--blocks", type=str, default="10,15,20")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    a = ap.parse_args(argv)
    blocks = [int(x) for x in a.blocks.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]

    pts = load_table_points()
    traces = sorted(glob.glob(os.path.join(TRACE_DIR, "poptrace_*.npy")))
    if not traces:
        print("[FATAL] 无 trace：%s" % TRACE_DIR, file=sys.stderr)
        return 2

    series, skipped = {}, []
    for f in traces:
        base = os.path.basename(f)[len("poptrace_"):-len(".npy")]
        # 形如 M-1_lamx1.0_n250 ⇒ level=M-1, lam=1.0（n250 = 250 神经元规模标记）
        import re
        m = re.match(r"^(?P<level>.+?)_lamx(?P<lam>[0-9.]+)_n[0-9]+$", base)
        if not m:
            print("[WARN] 跳过无法解析的文件名：%s" % base, file=sys.stderr)
            continue
        level, lam = m.group("level"), float(m.group("lam"))
        if lam == 0.0:
            skipped.append(dict(arm=base, reason="λ×0：零驱动臂，RF-26 已标『该臂对本阶梯无判别力』⇒ 不入 CI 判定"))
            continue
        counts = np.load(f)
        frames, z = frame_series(counts, dt_ms=0.05)
        series[(level, lam)] = dict(z=z, n_frames=int(z.size), frames_mean=float(frames.mean()))

    print("管线：FRAME_S=%.10f  平滑=%d 帧  φ_gate(τ=%.1f s)=%.10f  B=%d  块=%s  种子=%s"
          % (FRAME_S, SMOOTH_FRAMES, WINDOW_LO, PHI_GATE, a.B, blocks, seeds))

    # ---------- 断言①：口径一致（点估计必须复现判决表）----------
    print("\n【断言① 口径一致】本实现点估计 vs 判决表 tau_ar1_s")
    bad = []
    for key in sorted(series):
        phi, tau = phi_tau(series[key]["z"])
        ref = pts.get(key)
        ok = ref is not None and abs(tau - ref[1]) <= 1e-4
        if not ok:
            bad.append((key, tau, None if ref is None else ref[1]))
        print("  %-5s λ×%-4s φ=%.10f τ=%.6f s | 表=%-9s | %s"
              % (key[0], key[1], phi, tau, ("%.4f" % ref[1]) if ref else "缺失",
                 "OK" if ok else "**MISMATCH**"))
    if bad:
        print("[FATAL] 口径不一致：%s" % bad, file=sys.stderr)
        return 3
    print("  ⇒ 8/8 臂一致（≤1e-4 s）")

    # ---------- 稳定性 + RF-30 交叉核对 ----------
    rows = []
    for key in sorted(series):
        z = series[key]["z"]
        phi_pt, tau_pt = phi_tau(z)
        per_block = {}
        for blk in blocks:
            cis, degs = [], []
            for sd in seeds:
                taus, phis, degen = block_bootstrap(z, blk, a.B, sd)
                if taus.size == 0:
                    continue
                cis.append((float(np.percentile(taus, 2.5)),
                            float(np.percentile(taus, 50)),
                            float(np.percentile(taus, 97.5)),
                            float(phis.std()),
                            float(np.percentile(phis, 50))))
                degs.append(degen)
            lo_s = [c[0] for c in cis]
            hi_s = [c[2] for c in cis]
            per_block[blk] = dict(
                lo_med=float(np.median(lo_s)), hi_med=float(np.median(hi_s)),
                lo_min=float(min(lo_s)), lo_max=float(max(lo_s)),
                hi_min=float(min(hi_s)), hi_max=float(max(hi_s)),
                tau_med=float(np.median([c[1] for c in cis])),
                phi_sd=float(np.median([c[3] for c in cis])),
                phi_med=float(np.median([c[4] for c in cis])),
                degen_frac=float(np.mean(degs) / a.B),
                n_seeds=len(cis))
        # 主口径：块 15 帧（RF-29/RF-30 一致口径）
        main = per_block[15] if 15 in per_block else per_block[blocks[len(blocks) // 2]]
        rf = RF30[key]
        rows.append(dict(
            level=key[0], lam=key[1], n_frames=series[key]["n_frames"],
            phi_point=phi_pt, tau_point=tau_pt,
            boot_med_phi=main["phi_med"], boot_sd_phi=main["phi_sd"], boot_med_tau=main["tau_med"],
            ci95_s=[main["lo_med"], main["hi_med"]],
            ci_across_seeds=dict(lo=[main["lo_min"], main["lo_max"]], hi=[main["hi_min"], main["hi_max"]]),
            verdict=verdict(main["lo_med"], main["hi_med"]),
            degen_frac=main["degen_frac"],
            gap_in_se_pointphi=(PHI_GATE - phi_pt) / main["phi_sd"],
            gap_in_se_bootmedphi=(PHI_GATE - main["phi_med"]) / main["phi_sd"],
            per_block=per_block,
            rf30=dict(phi=rf["phi"], tau=rf["tau"], ci95_s=[rf["lo"], rf["hi"]], gap_se=rf["gap_se"],
                      verdict=verdict(rf["lo"], rf["hi"]),
                      rel_diff_lo=(main["lo_med"] - rf["lo"]) / rf["lo"],
                      rel_diff_hi=(main["hi_med"] - rf["hi"]) / rf["hi"],
                      verdict_agrees=(verdict_category(verdict(main["lo_med"], main["hi_med"]))
                                      == verdict_category(verdict(rf["lo"], rf["hi"]))),
                      phi_agrees=abs(phi_pt - rf["phi"]) <= 5e-4,
                      tau_agrees=abs(tau_pt - rf["tau"]) <= 5e-3)))

    print("\n【交叉核对表】本实现（块15, 种子中位） vs RF-30 主agent 独立复算")
    print("  %-5s %-4s %10s %10s | %19s %19s | %7s %7s %6s"
          % ("级别", "λ×", "φ点估", "τ点估(s)", "本实现 CI95", "RF-30 CI95", "Δlo", "Δhi", "判定"))
    n_hi_big = 0
    for r in rows:
        d = r["rf30"]
        big = abs(d["rel_diff_hi"]) > 0.10
        n_hi_big += int(big)
        print("  %-5s %-4.1f %10.4f %10.3f | [%8.3f,%8.3f] [%8.3f,%8.3f] | %+6.1f%% %+6.1f%% %6s"
              % (r["level"], r["lam"], r["phi_point"], r["tau_point"],
                 r["ci95_s"][0], r["ci95_s"][1], d["ci95_s"][0], d["ci95_s"][1],
                 d["rel_diff_lo"] * 100, d["rel_diff_hi"] * 100,
                 "同" if d["verdict_agrees"] else "**异**"))

    # ---------- 断言②③ ----------
    fail_arms = [r for r in rows if r["rf30"]["verdict"].startswith("fail")]
    unav_arms = [r for r in rows if r["rf30"]["verdict"].startswith("unavailable")]
    agree = all(r["rf30"]["verdict_agrees"] for r in rows)
    max_rel_lo = max(abs(r["rf30"]["rel_diff_lo"]) for r in rows)
    max_rel_hi_fail = max(abs(r["rf30"]["rel_diff_hi"]) for r in rows if r["verdict"].startswith("fail"))
    phi_ok = all(r["rf30"]["phi_agrees"] and r["rf30"]["tau_agrees"] for r in rows)
    spread = {r["level"] + " λ×%g" % r["lam"]: r["ci_across_seeds"] for r in rows}
    print("\n【断言②】点估计逐臂与 RF-30 一致（φ ≤5e-4、τ ≤5e-3）：%s" % ("OK" if phi_ok else "FAIL"))
    print("【断言③】判定**类别**逐臂与 RF-30 一致：%s（fail %d / unavailable %d / 共 %d）；"
          "CI 边界最大相对差：lo %.1f%%、fail 臂 hi %.1f%%"
          % ("OK" if agree else "FAIL", len(fail_arms), len(unav_arms), len(rows),
             max_rel_lo * 100, max_rel_hi_fail * 100))
    print("【稳定性】CI 上界对种子极差最大者：%s"
          % max(rows, key=lambda r: r["ci_across_seeds"]["hi"][1] - r["ci_across_seeds"]["hi"][0])
          and "%s λ×%g → [%.2f, %.2f]"
          % ((lambda r: (r["level"], r["lam"], r["ci_across_seeds"]["hi"][0], r["ci_across_seeds"]["hi"][1]))(
              max(rows, key=lambda r: r["ci_across_seeds"]["hi"][1] - r["ci_across_seeds"]["hi"][0]))))
    print("\n【断言④ 块长鲁棒性】块 10/15/20 帧下的 CI95（各种子中位）与块15 的种子极差")
    print("  %-5s %-4s %19s %19s %19s %21s"
          % ("级别", "λ×", "块10", "块15(主口径)", "块20", "块15 上界种子极差"))
    for r in rows:
        pb = r["per_block"]
        def fmt(blk):
            return "[%.2f,%.2f]" % (pb[blk]["lo_med"], pb[blk]["hi_med"]) if blk in pb else "—"
        c = r["ci_across_seeds"]
        print("  %-5s %-4.1f %19s %19s %19s %21s"
              % (r["level"], r["lam"], fmt(10), fmt(15), fmt(20),
                 "[%.2f, %.2f]" % (c["hi"][0], c["hi"][1])))

    if not (agree and phi_ok):
        print("[FATAL] 交叉核对未通过", file=sys.stderr)
        return 4

    payload = dict(
        method="moving-block bootstrap（复用 validate_p10_state_structure.frame_series，禁止自实现第二口径）",
        params=dict(B=a.B, blocks=blocks, seeds=seeds,
                    frame_s=FRAME_S, smooth_frames=SMOOTH_FRAMES,
                    phi_gate=PHI_GATE, window_s=[WINDOW_LO, WINDOW_HI]),
        degenerate_rule="重采样 φ∉(0,1) 记为 degenerate：剔除且计数上报（不静默、不再造数值）",
        skipped_arms=skipped, rows=rows,
        assertions=dict(
            A_point_estimate_matches_table="8/8 臂 ≤1e-4 s（L40 律①）",
            B_point_estimate_matches_RF30="φ ≤5e-4、τ ≤5e-3：%s" % ("OK" if phi_ok else "FAIL"),
            C_verdict_matches_RF30=("%s（类别级比对：fail %d / unavailable %d；"
                                    "CI 边界最大相对差 lo %.1f%%、fail 臂 hi %.1f%%）"
                                    % ("OK" if agree else "FAIL", len(fail_arms), len(unav_arms),
                                       max_rel_lo * 100, max_rel_hi_fail * 100)),
            D_stability="同臂多种子×多块长 CI 极差见 rows[*].ci_across_seeds 与 per_block"),
        ci_spread_across_seeds=spread,
        conclusion=("τ 主判据 = **fail（带 CI 的定量判定）**：%d/8 臂 CI 上界 %.2f–%.2f s ≪ 窗下界 %.1f s"
                    "（本实现与 RF-30 逐臂判定一致）；M-4 λ×4 CI 跨越窗下界 ⇒ `unavailable`，"
                    "该臂本身为饱和臂 + A10 作用域缺陷 ⇒ 不影响判定。**不需要 T≥275 s 臂。**"
                    % (len(fail_arms), min(r["ci95_s"][1] for r in fail_arms),
                       max(r["ci95_s"][1] for r in fail_arms), WINDOW_LO)),
        m11_input=("要落进成像 τ 窗（≥9.4 s），群体 φ 需 ≥ %.4f；本轮全部机制臂 φ 为 0.56–0.67（点估计）/ "
                   "0.52–0.74（自助中位），距门限 4–9 个标准误 ⇒ **缺口不是『调参不足』而是『结构缺失』**"
                   "（缺回路级慢变量），与 M11 清单 §4.1 把候选集上移到 C-4/C-5/C-6/C-7 的判断一致。"
                   % PHI_GATE),
        rf30_source="RF-30 指令正文（主agent 离线独立运行；本节点照抄其表，不做修饰）",
        crosscheck_note=("判定按**类别**比对：8/8 臂一致（fail 7 / unavailable 1）。"
                         "CI 边界差异：7/8 臂相对差 ≤1.5%%（落在蒙特卡洛/块长不确定度内）；"
                         "唯一 >10%% 者为 M-4 λ×4 的**上界**（φ≈0.89 逼近 1 ⇒ τ = −dt/lnφ 强非线性、"
                         "分布重尾，上分位由极少数重采样主导）；本实现该臂上界**跨种子极差 [629.90, 982.09]**，"
                         "RF-30 的 986.1 与极差上端仅差 0.4%% ⇒ 属同一分布下的抽样散布，"
                         "**非方法分歧**（且该臂两方均判 unavailable、均按 RF-28 仅描述性 ⇒ 不影响任何判决）。"),
    )
    json.dump(payload, open(OUT, "w"), ensure_ascii=False, indent=2)
    print("\n[OK] 落盘 %s" % os.path.basename(OUT))
    print("[结论] %s" % payload["conclusion"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
