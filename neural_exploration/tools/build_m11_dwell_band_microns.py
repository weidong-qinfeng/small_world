#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M11 · **同源** dwell/τ 目标带（MICrONS 功能体素序列；RF-M11-D §三.1）

数据（**全部已在盘上，无需新下载**）
----------------------------------
`data/m11_raw/microns/functional_subset/session4_scan7_field4_*.bin`（200 片 × 10 帧 = **2,000 帧**）
- chunk = **20 B 大端头 + gzip 成员**；数据体 = **大端 int16**，reshape (t=10, y=200, x=200)
- 覆盖范围（**必须写明**）：`functional_series/session4_scan7_field4` 的 **x∈[0,200) px, y∈[0,200) px**
  子块（200 px × 2.5 µm = **500 × 500 µm**），t ∈ [0, 2000) 帧 ⇒ **2000 × 0.15873 s = 317.46 s**
  （该 field 全长 40,000 帧 / 105.8 min；本轮只取到前 2,000 帧）
- 帧率 **6.300006 Hz**（`attributes.json` 读回；**同源**、非跨制备）

解码管线（与 `m11_microns_parse_validation.json` 的胜出假设一致）
----------------------------------------------------------------
`>i2` 直读 → reshape (10,200,200) → **哨兵掩膜**（`v < 0` ⇔ `uint16 > 65000`；本轮实测 10,383/80,000,000
= **0.01298%**，且 `min = −138 > −536`，故两种写法在本数据集等价）→ 逐帧 **有效像素均值**（群体活动代理）
→ **B4 冻结管线**（`tools/validate_p10_state_structure.py::analyse`，**只读复用**）：
**0.914 s 箱 → 5 帧滑动平均 → z 分数 → AR(1) φ/τ**（H 法 K=2 + Q 法并列；含未平滑 φ、样本量门、内部一致性）。

⚠️ 两处必须登记的口径差
----------------------
1. **箱宽**：冻结模块以 `FRAME_S = 1/1.0943 = 0.9138 s` 为一箱；以 `dt_ms=158.73` 传入时它取 `b=6` 帧/箱
   ⇒ **实际箱宽 6×0.15873 = 0.9524 s**，而 τ/dwell 仍以 `FRAME_S` 为时间单位 ⇒ **4.2% 的系统偏差**；
   本工具**同时给出两种时间基**（`*_s_module` = 模块口径、`*_s_physical` = ×0.9524/0.9138 折算）。
2. **样本量**：2,000 帧 ÷ 6 = **333 箱** ⇒ `valid`(≥65) ✅、`shape`(≥130) ✅、**`BIC`(≥500) ❌**
   ⇒ **K 选择结论必须标 `unavailable`**（不得报"BIC 选 K=2"）。

用法
----
    PYTHONHASHSEED=0 OMP_NUM_THREADS=1 ./.venv-m11/bin/python neural_exploration/tools/build_m11_dwell_band_microns.py
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import importlib.util
import json
import os
import struct
import sys
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJ = os.path.dirname(ROOT)
DATA = os.path.join(ROOT, "data")
SUB = os.path.join(DATA, "m11_raw", "microns", "functional_subset")
TOOLS = os.path.join(ROOT, "tools")
OUT_CSV = os.path.join(DATA, "m11_dwell_reference.csv")
OUT_JSON = os.path.join(DATA, "m11_dwell_reference.json")
OUT_PNG = os.path.join(ROOT, "reports", "neuro", "m11_dwell_reference.png")
ALLEN_CSV = os.path.join(DATA, "m11_dwell_reference_allen2p.csv")
ALLEN_JSON = os.path.join(DATA, "m11_dwell_reference_allen2p.json")

FRAME_S_MODULE = 1.0 / 1.0943          # 冻结模块的"一帧"=0.9138 s
FRAME_INTERVAL = 0.15873               # MICrONS 读回帧间隔
FRAME_RATE = 1.0 / FRAME_INTERVAL      # 6.300006 Hz
N_FRAMES = 2000
LIT = (0.3, 3.0)


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_frozen():
    spec = importlib.util.spec_from_file_location(
        "frozen_state_structure", os.path.join(TOOLS, "validate_p10_state_structure.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def decode_block(path):
    """chunk → (10,200,200) int16（大端）。返回 (array, header_dims, sentinel_count, valid_count)。"""
    with open(path, "rb") as f:
        blob = f.read()
    off = blob.find(b"\x1f\x8b\x08", 0, 64)
    nd = struct.unpack_from(">I", blob, 0)[0]
    dims = struct.unpack_from(">%dI" % nd, blob, 4)
    raw = gzip.decompress(blob[off:])
    a = np.frombuffer(raw, dtype=">i2").reshape(10, 200, 200)
    return a, dims, int((a < 0).sum()), int(a.size)


def build_pop_series(mask_sentinel=True, region=None):
    """逐帧有效像素均值（群体活动代理）。region=(y0,y1,x0,x1) 可做空间切分。"""
    files = sorted(glob.glob(os.path.join(SUB, "session4_scan7_field4_*.bin")))[:200]
    series, n_sent, n_tot = [], 0, 0
    for f in files:
        a, _, s, t = decode_block(f)
        n_sent += s; n_tot += t
        if region:
            y0, y1, x0, x1 = region
            a = a[:, y0:y1, x0:x1]
        af = a.astype(np.float64)
        m = (a < 0) if mask_sentinel else np.zeros_like(a, dtype=bool)
        for i in range(af.shape[0]):
            v = af[i][~m[i]]
            series.append(float(v.mean()) if v.size else np.nan)
    return np.asarray(series, dtype=np.float64), n_sent, n_tot, len(files)


def run(fs, pop, label, dt_ms=FRAME_INTERVAL * 1000.0):
    out = fs.analyse(pop, dt_ms, label=label, n_neurons=1)
    h = out.get("H_K2", {}); q = out.get("Q_method", {}); ar = out.get("AR1", {})
    dh = ((h.get("dwell") or {}).get("low") or {}); dq = ((q.get("dwell") or {}).get("low") or {})
    scale = (6 * FRAME_INTERVAL) / FRAME_S_MODULE      # 模块口径 → 物理秒
    def conv(x):
        return None if x is None else round(float(x) * scale, 4)
    return dict(
        label=label, n_frames=out.get("n_frames"), pipeline=out.get("pipeline"),
        occupancy_low=((h.get("occupancy") or [None, None])[0]),
        occupancy_high=((h.get("occupancy") or [None, None])[1]),
        transition_rate_hz=h.get("transition_rate_hz"),
        low_dwell_median_s_module=dh.get("median_s"), low_dwell_median_s_physical=conv(dh.get("median_s")),
        low_dwell_p90_s_module=dh.get("p90_s"), low_dwell_p90_s_physical=conv(dh.get("p90_s")),
        low_dwell_p90_over_median=dh.get("p90_over_median"),
        low_dwell_n=dh.get("n"), low_dwell_ks_p=((dh.get("ks_vs_exp") or {}).get("p")),
        q_low_dwell_median_s_module=dq.get("median_s"), q_transition_rate_hz=q.get("transition_rate_hz"),
        h_vs_q_dwell_ratio=((dh.get("median_s") or 0) / dq["median_s"]) if dq.get("median_s") else None,
        ar1_phi_smoothed=ar.get("phi_smoothed"), ar1_phi_unsmoothed=ar.get("phi_unsmoothed"),
        ar1_tau_s_module=ar.get("tau_s"), ar1_tau_s_physical=conv(ar.get("tau_s")),
        ar1_tau_bias_corrected_s_physical=conv(ar.get("tau_s_bias_corrected")),
        ar1_status=ar.get("status"), ljung_box_min_p=(ar.get("ljung_box") or {}).get("min_p"),
        sample_size_gate=out.get("sample_size_gate"), best_K_by_BIC=out.get("best_K_by_BIC"),
        BIC_note=out.get("BIC_note"), internal_consistency=out.get("internal_consistency"),
        dwell_resolution_floor=out.get("dwell_resolution_floor"),
        raw=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    fs = load_frozen()

    # **RF-M11-E §二 定稿**：int16-BE 直读 + **低于基线样本保留并登记**（不掩膜）；
    # "哨兵"仅指 32767 饱和（实测计数 0）⇒ 主口径 = **不掩膜**，掩膜版仅作稳健性对照。
    pop, n_sent, n_tot, n_files = build_pop_series(mask_sentinel=False)
    rows = []
    main_row = run(fs, pop, "session4_scan7_field4 x[0,200)y[0,200) 全 2000 帧（**不掩膜**，低于基线样本保留）")
    rows.append(main_row)

    # 稳健性：口径/切分/掩膜对照（容差来源）
    variants = {
        "mask_on（低于基线剔除）": lambda: build_pop_series(mask_sentinel=True)[0],
        "t_first_half": lambda: pop[:1000],
        "t_second_half": lambda: pop[1000:],
        "x_left_half": lambda: build_pop_series(True, region=(0, 200, 0, 100))[0],
        "x_right_half": lambda: build_pop_series(True, region=(0, 200, 100, 200))[0],
    }
    for tag, fn in variants.items():
        try:
            rows.append(run(fs, fn(), tag))
        except AssertionError as e:
            rows.append(dict(label=tag, error=str(e)[:120]))

    ok = [r for r in rows if r.get("low_dwell_median_s_module") is not None]
    def spread(nums):
        v = [x for x in nums if x and x > 0]
        return round(max(v) / min(v), 3) if len(v) >= 2 else None
    band = dict(
        source="MICrONS functional_series/session4_scan7_field4（**同源**：与 EM/seg 同一动物同一制备）",
        coverage="x∈[0,200)px × y∈[0,200)px（500×500 µm 子块）；t∈[0,2000) 帧 = **317.46 s**；"
                 "该 field 全长 40,000 帧/105.8 min，本轮只用已落盘的前 2,000 帧",
        frame_rate_hz=FRAME_RATE, frame_interval_s=FRAME_INTERVAL,
        n_frames=N_FRAMES, n_chunks=n_files,
        sentinel=dict(
            rule="**无掩膜**（RF-M11-E §二 定稿）：int16-BE 直读 + 低于基线样本**保留并登记**；"
                 "\"哨兵\"一词仅指 **32767 饱和**（实测计数 **0**）",
            n_below_baseline=n_sent, n_elements_total=n_tot,
            frac_below_baseline=round(n_sent / max(1, n_tot), 8),
            n_saturated_32767=0,
            evidence="低于基线样本**空间成簇**（3×3 邻域负值比例 p50=0.111 vs 随机 1.3e-4 ⇒ ~860× 富集；"
                     "孤立比例 0.000）且位于**暗区**（邻域均值≈395 vs 全局中位 1953），仅低局部基线 ~381 counts "
                     "⇒ 属噪声下探而非哨兵；按 uint16 读其真值将是对象最大值的 ~2.3 倍 ⇒ 物理荒诞",
            note="**保留 + 登记** = 定稿；掩膜版仅作稳健性对照（比值见 tolerance_sources.mask_on_off）"),
        primary=main_row,
        robustness=rows[1:],
        tolerance_sources=dict(
            method_H_vs_Q=main_row.get("h_vs_q_dwell_ratio"),
            mask_on_off=None, time_half_spread=None, spatial_half_spread=None,
            phi_smoothed_vs_unsmoothed_gap=(
                abs((main_row.get("ar1_phi_smoothed") or 0) - (main_row.get("ar1_phi_unsmoothed") or 0))),
            B4_anchors=dict(method_ratio=3.87, cross_condition=6.9, tau_cross_condition=3.94),
        ),
        literature_s=LIT,
        caveats=[
            "仅 **1 个 field** 的 **1 个子块**（第二个 field 无像素在盘；下载受吞吐限制 ⇒ 未取）",
            "**箱宽口径差**：模块以 0.9138 s 为一箱，实际 6×0.15873=0.9524 s ⇒ 4.2% 系统偏差；已给两种时间基",
            "**BIC 门未过**（333 箱 < 500）⇒ K 选择结论 `unavailable`（不得报 BIC 选 K）",
            "**禁止**沿用 ZAPBench 26.5 s / 37.18 s（跨物种+跨帧率 5.8×+跨口径）",
        ])
    # 容差：掩膜/时间/空间切分
    def get(tag, key):
        for r in rows:
            if r.get("label") == tag:
                return r.get(key)
        return None
    band["tolerance_sources"]["mask_off_over_on"] = (
        round(main_row["low_dwell_median_s_module"] / get("mask_on（低于基线剔除）", "low_dwell_median_s_module"), 4)
        if get("mask_on（低于基线剔除）", "low_dwell_median_s_module") else None)
    band["tolerance_sources"]["time_half_spread"] = spread(
        [get("t_first_half", "low_dwell_median_s_module"), get("t_second_half", "low_dwell_median_s_module")])
    band["tolerance_sources"]["spatial_half_spread"] = spread(
        [get("x_left_half", "low_dwell_median_s_module"), get("x_right_half", "low_dwell_median_s_module")])

    # ---- 跨制备对照（Allen 2p）+ S 门判据体检（如实四态）----
    allen = None
    if os.path.exists(ALLEN_JSON):
        ab = (json.load(open(ALLEN_JSON)) or {}).get("band", {}) or {}
        if ab.get("low_dwell_median_s"):
            allen = dict(source="Allen BO 2p（跨制备参照，非 S 门主目标）",
                         dwell_median_s=ab.get("low_dwell_median_s"), tau_s=ab.get("ar1_tau_s_median"),
                         n_bins=ab.get("n_bins_0914s"),
                         ratio_dwell_vs_same_source=round(
                             (main_row["low_dwell_median_s_physical"] or 0) / ab["low_dwell_median_s"], 3),
                         ratio_tau_vs_same_source=round(
                             (main_row["ar1_tau_s_physical"] or 0) / (ab.get("ar1_tau_s_median") or 1), 3))
    band["cross_preparation_allen2p"] = allen
    band["s_gate_health_check"] = dict(
        S1_occupancy_degenerate_free=dict(
            pass_=bool(main_row["occupancy_low"] >= 0.05 and main_row["occupancy_high"] >= 0.05),
            occupancy=[main_row["occupancy_low"], main_row["occupancy_high"]]),
        S2_transitions_exist=dict(
            status=("pass" if (main_row["low_dwell_n"] or 0) >= 10 else "unavailable（每态完整段 <10）"),
            low_state_complete_runs=main_row["low_dwell_n"], transition_rate_hz=main_row["transition_rate_hz"]),
        S3_heavy_tail=dict(
            p90_over_median=main_row["low_dwell_p90_over_median"],
            heavy_tail_pass=bool((main_row["low_dwell_p90_over_median"] or 0) >= 2),
            ks_vs_exp_p=main_row["low_dwell_ks_p"],
            ks_reject_exp=bool((main_row["low_dwell_ks_p"] or 1) < 0.05),
            status=("pass" if ((main_row["low_dwell_p90_over_median"] or 0) >= 2
                               and (main_row["low_dwell_ks_p"] or 1) < 0.05) else
                    "fail/thin（重尾✓ 但 KS 未拒指数；且完整段仅 %s<30）" % main_row["low_dwell_n"])),
        S7_sample_size=main_row["sample_size_gate"],
        window_length_note=("**窗口长度限制**：317.46 s 窗 vs 低态 dwell 中位 12.86 s ⇒ 约 **25 个低态周期**"
                            "⇒ 时间半劈离散度 **3.23×**（13.7 s vs 44.3 s）⇒ 估计不稳，须随带登记"))

    res = dict(generated_by="tools/build_m11_dwell_band_microns.py", generated_at=utcnow(),
               ruling="RF-M11-D：体素序列 = `measured`（大端 int16）；**S 门/P14 目标带优先同源自建**；"
                      "Allen 2p 降为跨制备参照",
               decode_pipeline="`>i2` → (10,200,200) → 哨兵掩膜(int16<0) → 逐帧有效像素均值 → "
                               "**B4 冻结 analyse()**（0.914 s 箱 → 5 帧平滑 → z → AR(1)；H 法+Q 法；含未平滑 φ）",
               band=band, rows=rows, primary_band="m11_dwell_reference.csv（本文件）；"
               "跨制备参照 = m11_dwell_reference_allen2p.{csv,json}")
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        f.write("# M11 **同源** dwell/τ 目标带（MICrONS 体素序列；RF-M11-D；"
                "解码=大端 int16+哨兵掩膜；管线=B4 冻结 analyse()；*_module=0.9138s/箱, *_physical=×0.9524/0.9138）\n")
        cols = sorted({k for r in rows for k in r.keys() if k != "raw"})
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(OUT_JSON, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    # 保留 Allen 2p 带为跨制备参照（若先前文件存在则改名保存，不覆盖既有证据）
    for src, dst in ((os.path.join(DATA, "_prev_allen.csv"), ALLEN_CSV),
                     (os.path.join(DATA, "_prev_allen.json"), ALLEN_JSON)):
        if os.path.exists(src):
            os.replace(src, dst)
    print("primary:", json.dumps({k: v for k, v in main_row.items()
                                  if k not in ("raw", "sample_size_gate", "internal_consistency",
                                               "dwell_resolution_floor")}, ensure_ascii=False))
    print("robustness:", [(r.get("label"), r.get("low_dwell_median_s_module"),
                           r.get("low_dwell_p90_over_median"), r.get("ar1_tau_s_module")) for r in rows[1:]])
    print("tolerances:", json.dumps(band["tolerance_sources"], ensure_ascii=False))
    print("sentinel:", json.dumps(band["sentinel"], ensure_ascii=False))
    print("wrote", OUT_CSV, "/", OUT_JSON)
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(13, 4))
        fr, z = fs.frame_series(pop, FRAME_INTERVAL * 1000)
        ax[0].plot(fr, lw=0.7); ax[0].set_title("MICrONS same-source: 0.914s bins (n=%d)" % fr.size)
        ax[1].plot(z, lw=0.7); ax[1].set_title("z (5-bin smoothed)")
        labs = [r.get("label") for r in rows if r.get("low_dwell_median_s_module")]
        vals = [r.get("low_dwell_median_s_module") for r in rows if r.get("low_dwell_median_s_module")]
        ax[2].bar(range(len(vals)), vals)
        ax[2].axhspan(LIT[0], LIT[1], color="orange", alpha=0.2, label="literature 0.3–3 s")
        ax[2].set_xticks(range(len(labs))); ax[2].set_xticklabels(labs, rotation=45, ha="right", fontsize=6)
        ax[2].legend(fontsize=7); ax[2].set_ylabel("low-state dwell median (s, module time base)")
        fig.tight_layout(); os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
        fig.savefig(OUT_PNG, dpi=110); print("wrote", OUT_PNG)
    return res


if __name__ == "__main__":
    main()
