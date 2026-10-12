#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M11 · 自建 **dwell/τ 目标带**（RF-M11-C §二.3；活动源 = Allen Brain Observatory 2p）

口径纪律（**不得**沿用 ZAPBench 的 26.5 s / 37.18 s）
----------------------------------------------------
B4 的冻结管线在 `tools/validate_p10_state_structure.py`（只读复用，**不修改**）：
`analyse(pop_counts_steps, dt_ms, …)` = **0.914 s 箱 → 5 帧滑动平均 → z 分数 → AR(1) φ/τ**
（H 法 2 状态高斯 HMM + Q 法分位法并列；**同时给未平滑口径**；含样本量硬门与内部一致性断言）。
本工具只做三件事：① 把 Allen 2p events 变成"逐步群体水平"；② 调 `analyse()`；③ 汇总带与容差并落盘。

输入（自描述，numpy `.npz`；**实测**：`ev.npy` 不经压缩 ⇒ 文件大小 = n_cells × n_frames × 8）
- `data/m11_raw/allen/vc2p/593887846_events.npz`（**1 cell × 114,130 帧**）
- `data/m11_raw/allen/vc2p/577885923_events.npz`（**4 cells × 113,835 帧**）
（更大群体件因网络退化（1.6–32 KB/s）未取 ⇒ 登记托管作业 `data/m11_b1_managed_jobs.json`）

用法
----
    PYTHONHASHSEED=0 OMP_NUM_THREADS=1 ./.venv-m11/bin/python neural_exploration/tools/build_m11_dwell_band.py
    # 画图：OMP_NUM_THREADS=1 ./.venv-m10/bin/python ... --plot
"""

from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJ = os.path.dirname(ROOT)
DATA = os.path.join(ROOT, "data")
VC2P = os.path.join(DATA, "m11_raw", "allen", "vc2p")
TOOLS = os.path.join(ROOT, "tools")
# ★ 路径冲突修复（2026-10-12 事故：本工具与同源 MICRONS 工具曾共用 `m11_dwell_reference.{csv,json}`，
#   本工具后跑 ⇒ **覆盖了 S 门主带（R-32）**）。现默认写入 `_allen2p` 侧，与"跨制备参照"的既有命名一致；
#   并用 MICRONS_PRIMARY_* 做**防覆盖守卫**（除非 --force-primary）。
OUT_CSV = os.path.join(DATA, "m11_dwell_reference_allen2p.csv")
OUT_JSON = os.path.join(DATA, "m11_dwell_reference_allen2p.json")
OUT_PNG = os.path.join(ROOT, "reports", "neuro", "m11_dwell_reference_allen2p.png")
MICRONS_PRIMARY_CSV = os.path.join(DATA, "m11_dwell_reference.csv")
MICRONS_PRIMARY_JSON = os.path.join(DATA, "m11_dwell_reference.json")

FRAME_RATE_PRIMARY = 30.9      # Allen BO 2p 常见采集率（**待 NWB 读回核实** ⇒ NEEDS_VERIFY）
FRAME_RATE_ALT = 11.0          # 备选（多平面 session）
LIT_BAND_S = (0.3, 3.0)        # 文献档（PMID 11017176，up/down dwell 量级）—— **非判据带**
B4_TOLERANCE = dict(method_ratio=3.87, cross_condition=6.9, tau_cross_condition=3.94,
                    source="B4 实测（口径差 3.87× / 跨条件 dwell 6.9× / τ 跨条件 3.94×）")


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_frozen():
    """导入 B4 的冻结状态结构模块（**只读**；不改冻结件）。"""
    p = os.path.join(TOOLS, "validate_p10_state_structure.py")
    spec = importlib.util.spec_from_file_location("frozen_state_structure", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_events(path):
    with np.load(path) as z:
        key = "ev.npy" if "ev.npy" in z.files else z.files[0]
        ev = np.asarray(z[key], dtype=np.float64)
    if ev.ndim == 1:
        ev = ev.reshape(1, -1)
    return ev


def build_band(fs, mod):
    """对每件 events.npz × 每套帧率假设，跑冻结管线并抽出带。"""
    rows = []
    for path in sorted(glob.glob(os.path.join(VC2P, "*_events.npz"))):
        name = os.path.basename(path)
        if name.startswith("_"):
            continue
        try:
            ev = load_events(path)
        except Exception as e:
            # **完整性门**：不完整/损坏的 npz 不得进入统计（RF-M11-C 纪律延续）
            rows.append(dict(experiment=name, rate_tag="-", status="INCOMPLETE(incomplete_or_corrupt)",
                             n_cells=None, n_frames=None,
                             note="npz 读取失败（**未入统计**）：%s；本地字节=%d"
                                  % (str(e)[:80], os.path.getsize(path))))
            continue
        n_cells, n_frames = ev.shape
        pop = ev.sum(axis=0)                      # 逐步群体水平（= 各 cell 事件幅度之和）
        for rate, tag in ((FRAME_RATE_PRIMARY, "primary"), (FRAME_RATE_ALT, "alt")):
            dt_ms = 1000.0 / rate
            try:
                out = mod.analyse(pop, dt_ms, label="%s@%.1fHz" % (name, rate), n_neurons=n_cells)
            except AssertionError as e:
                rows.append(dict(experiment=name, frame_rate_hz=rate, rate_tag=tag, status="invalid",
                                 n_cells=n_cells, n_frames=n_frames, note=str(e)[:160]))
                continue
            h = out.get("H_K2", {})
            q = out.get("Q_method", {})
            dh = ((h.get("dwell") or {}).get("low") or {})
            dq = ((q.get("dwell") or {}).get("low") or {})
            ar = out.get("AR1", {})
            rows.append(dict(
                experiment=name, rate_tag=tag, frame_rate_hz=rate, n_cells=n_cells, n_frames=n_frames,
                n_bins_0914s=out.get("n_frames"), status="measured",
                pop_rate_neuron_hz=out.get("pop_rate_neuron_hz"),
                occupancy_low=((h.get("occupancy") or [None, None])[0]),
                occupancy_high=((h.get("occupancy") or [None, None])[1]),
                transition_rate_hz=h.get("transition_rate_hz"),
                low_dwell_median_s=dh.get("median_s"), low_dwell_p90_s=dh.get("p90_s"),
                low_dwell_p90_over_median=dh.get("p90_over_median"),
                low_dwell_n=dh.get("n"),
                low_dwell_ks_p=((dh.get("ks_vs_exp") or {}).get("p")
                                if isinstance(dh.get("ks_vs_exp"), dict) else None),
                low_dwell_ks_D=((dh.get("ks_vs_exp") or {}).get("D")
                                if isinstance(dh.get("ks_vs_exp"), dict) else None),
                low_dwell_runs=((q.get("dwell") or {}).get("low") or {}).get("n"),
                q_low_dwell_median_s=dq.get("median_s"), q_transition_rate_hz=q.get("transition_rate_hz"),
                ar1_phi_smoothed=ar.get("phi_smoothed"), ar1_phi_unsmoothed=ar.get("phi_unsmoothed"),
                ar1_tau_s=ar.get("tau_s"), ar1_tau_bias_corrected_s=ar.get("tau_s_bias_corrected"),
                ar1_status=ar.get("status"), ar1_ljung_box_p=(ar.get("ljung_box") or {}).get("min_p"),
                sample_size_valid=(out.get("sample_size_gate") or {}).get("valid"),
                sample_size_shape_ok=(out.get("sample_size_gate") or {}).get("shape_ok"),
                best_K_by_BIC=out.get("best_K_by_BIC"),
                dwell_resolution_at_floor=(out.get("dwell_resolution_floor") or {}).get("at_floor"),
                internal_consistency_pass=(out.get("internal_consistency") or {}).get("pass_"),
                pipeline=out.get("pipeline"),
                full=out))
    return rows


def summarize(rows):
    ok = [r for r in rows if r.get("status") == "measured"]
    prim = [r for r in ok if r.get("rate_tag") == "primary"]
    def med(key, rs=None):
        v = [r.get(key) for r in (rs if rs is not None else prim) if isinstance(r.get(key), (int, float))]
        return float(np.median(v)) if v else None
    def spread(key):
        v = [r.get(key) for r in prim if isinstance(r.get(key), (int, float)) and r.get(key) > 0]
        return (round(max(v) / min(v), 3) if len(v) >= 2 and min(v) > 0 else None)
    band = dict(
        n_experiments=len(prim), experiments=[r["experiment"] for r in prim],
        frame_rate_hz_primary=FRAME_RATE_PRIMARY, frame_rate_source="**NEEDS_VERIFY**（NWB `imaging_rate` 因网络未读回；"
                              "文件帧数÷30.9 Hz = 61.4 min 属合理 session 长度，÷11 Hz = 2.87 h 不合理 ⇒ 采 30.9）",
        n_bins_0914s=med("n_bins_0914s"),
        occupancy_low_median=med("occupancy_low"),
        transition_rate_hz_median=med("transition_rate_hz"),
        low_dwell_median_s=med("low_dwell_median_s"),
        low_dwell_p90_s=med("low_dwell_p90_s"),
        low_dwell_p90_over_median=med("low_dwell_p90_over_median"),
        low_dwell_ks_p_max=None,
        ar1_tau_s_median=med("ar1_tau_s"),
        ar1_tau_s_range=[min([r["ar1_tau_s"] for r in prim if isinstance(r.get("ar1_tau_s"), (int, float))],
                             default=None),
                         max([r["ar1_tau_s"] for r in prim if isinstance(r.get("ar1_tau_s"), (int, float))],
                             default=None)],
        ar1_phi_smoothed_median=med("ar1_phi_smoothed"), ar1_phi_unsmoothed_median=med("ar1_phi_unsmoothed"),
        **{"spread_cross_experiment_" + k: spread(k) for k in
           ("low_dwell_median_s", "transition_rate_hz", "ar1_tau_s", "occupancy_low")},
        smoothed_vs_unsmoothed_phi_gap=(
            round(abs(med("ar1_phi_smoothed") - med("ar1_phi_unsmoothed")), 6)
            if med("ar1_phi_smoothed") is not None and med("ar1_phi_unsmoothed") is not None else None),
    )
    # 形状判据（RF-M11-C §四.1 ③）：p90/中位 ≥2 且拒绝指数（KS p<0.05）
    # **样本量门（含"转换次数"门）**：形状类判据需 n_runs ≥ 5（KS 可算）且推荐 ≥30
    for r in prim:
        n_runs = r.get("low_dwell_runs")
        r["shape_criterion_status"] = ("unavailable（low 态完整段 <5 ⇒ KS 不可算）"
                                       if not n_runs or n_runs < 5 else
                                       ("pass(thin)" if n_runs < 30 else "pass"))
    shapes = [r for r in prim if r.get("low_dwell_p90_over_median") is not None
              and (r.get("low_dwell_runs") or 0) >= 5]
    band["shape_criterion"] = dict(
        n_evaluable=len(shapes),
        n_runs_note="**转换次数门**：low 态完整段 <5 时 KS 不可算 ⇒ 该实验的形状判据标 unavailable",
        per_experiment_status={r["experiment"]: r.get("shape_criterion_status") for r in prim},
        heavy_tail_all=(all((r["low_dwell_p90_over_median"] or 0) >= 2 for r in shapes) if shapes else None),
        ks_reject_exp_all=(all((r.get("low_dwell_ks_p") or 1) < 0.05 for r in shapes) if shapes else None),
        per_experiment={r["experiment"]: dict(p90_over_median=r["low_dwell_p90_over_median"],
                                              ks_p=r.get("low_dwell_ks_p")) for r in shapes})
    band["tolerance_sources"] = dict(
        **B4_TOLERANCE,
        self_cross_experiment=band.get("spread_cross_experiment_low_dwell_median_s"),
        self_method_H_vs_Q=(round(med("low_dwell_median_s") / med("q_low_dwell_median_s"), 3)
                            if med("q_low_dwell_median_s") else None),
        note="**绝对尺度不得按单点引用**：按 RF-M11-C §四.2 以'与自建目标带的比值'报告，"
             "并在报告中并列上述容差来源（B4 实测容差 + 本轮跨实验/跨口径实测容差）")
    band["literature_fallback_s"] = dict(band=LIT_BAND_S, pmid="11017176",
                                         note="文献档（up/down dwell 量级）—— **仅量级参考、非判据带**")
    return band


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out-prefix", default=None,
                    help="输出前缀（默认 m11_dwell_reference_allen2p，写在 data/ 下）")
    ap.add_argument("--force-primary", action="store_true",
                    help="显式允许写同源主带路径（默认拒绝：防覆盖 R-32/S 门主带）")
    args = ap.parse_args()
    global OUT_CSV, OUT_JSON, OUT_PNG
    if args.out_prefix:
        OUT_CSV = os.path.join(DATA, args.out_prefix + ".csv")
        OUT_JSON = os.path.join(DATA, args.out_prefix + ".json")
        OUT_PNG = os.path.join(ROOT, "reports", "neuro", args.out_prefix + ".png")
    if os.path.abspath(OUT_JSON) in (os.path.abspath(MICRONS_PRIMARY_JSON),
                                     os.path.abspath(MICRONS_PRIMARY_CSV)) and not args.force_primary:
        print("REFUSED: 输出路径与**同源 MICRONS 主带**冲突（%s）⇒ 拒绝覆盖 R-32/S 门主带；"
              "如确需覆盖请显式加 --force-primary" % OUT_JSON)
        return 3
    mod = load_frozen()
    rows = build_band(None, mod)
    band = summarize(rows)
    res = dict(generated_by="tools/build_m11_dwell_band.py", generated_at=utcnow(),
               source="Allen Brain Observatory 2p `ophys_experiment_events/*_events.npz`（自描述 numpy）",
               pipeline="B4 冻结管线（`tools/validate_p10_state_structure.py::analyse` 只读复用）："
                        "0.914 s 箱 → 5 帧滑动平均 → z 分数 → AR(1) φ/τ（H 法 + Q 法并列，含未平滑口径）",
               band=band, rows=[{k: v for k, v in r.items() if k != "full"} for r in rows],
               caveats=[
                   "群体规模薄：1 cell 与 4 cells（更大群体件 16.4 MB / 111.9 MB 因网络 1.6–32 KB/s 未取 ⇒ 托管作业）",
                   "帧率 30.9 Hz 为 **NEEDS_VERIFY**（NWB `imaging_rate` 读回被网络阻塞）；已给 11 Hz 敏感性列",
                   "**不得**与 ZAPBench 的 26.5 s / 37.18 s 直接比较（跨物种+跨帧率+跨口径）",
                   "dwell 触分辨率地板（≤1.5 帧）时只能作旁证（冻结模块已自动登记）",
               ])
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        f.write("# M11 自建 dwell/τ 目标带（generated_by=tools/build_m11_dwell_band.py；"
                "活动源=Allen BO 2p events；管线=B4 冻结 `analyse()`；**禁用 ZAPBench 数值**）\n")
        cols = [k for k in rows[0].keys() if k != "full"] if rows else []
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(OUT_JSON, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print("band:", json.dumps({k: v for k, v in band.items() if not isinstance(v, dict)}, ensure_ascii=False, indent=1)[:900])
    print("shape_criterion:", json.dumps(band["shape_criterion"], ensure_ascii=False))
    print("wrote", OUT_CSV, "/", OUT_JSON)

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(13, 4))
        for r in [x for x in rows if x.get("status") == "measured" and x.get("rate_tag") == "primary"]:
            ev = load_events(os.path.join(VC2P, r["experiment"]))
            pop = ev.sum(axis=0)
            fr, z = mod.frame_series(pop, 1000.0 / FRAME_RATE_PRIMARY)
            ax[0].plot(fr[:2000], lw=0.6, label=r["experiment"])
            ax[1].plot(z[:2000], lw=0.6)
        ax[0].set_title("pop frames (0.914 s bins)"); ax[1].set_title("z (5-bin smoothed)")
        lbl = [r["experiment"] + "@%.1f" % r["frame_rate_hz"] for r in rows if r.get("status") == "measured"]
        val = [r.get("low_dwell_median_s") or 0 for r in rows if r.get("status") == "measured"]
        ax[2].bar(range(len(val)), val)
        ax[2].axhspan(LIT_BAND_S[0], LIT_BAND_S[1], color="orange", alpha=0.2, label="literature 0.3–3 s")
        ax[2].set_xticks(range(len(lbl))); ax[2].set_xticklabels(lbl, rotation=45, ha="right", fontsize=7)
        ax[2].set_ylabel("low-state dwell median (s)"); ax[2].legend(fontsize=7)
        fig.tight_layout()
        os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
        fig.savefig(OUT_PNG, dpi=110)
        print("wrote", OUT_PNG)
    return res


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(main())
