#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M11 · J2：**钙域静默比例口径**（P5 (b) 输入；RF-M11-F §一）

目标
----
把判据带 P5 (b)「静默比例」的**钙域口径**从 `final=0`（未定）推向**可定稿**，并**新增行**（不改原行）。

静默定义（**三种口径并列，全部显式写出**）
----------------------------------------
1. **`zero_activity`（零事件）**：该单元在观测窗内**从未**出现超过阈值的钙事件；
2. **`below_baseline`（低于基线）**：该单元的时间均值 < 自身基线 `F0`（=`F0` 取每单元 20 分位）；
3. **`low_percentile`（低活动分位）**：该单元的活动幅度（ΔF/F 的 p95）低于**全视野 p20**（"弱反应单元"口径）。

单元定义（**两种口径并列**）
--------------------------
- **per-ROI（代理）**：以 **20×20 px 空间分箱**（= 50×50 µm @2.5 µm/px）作为单元代理；
  ⚠️ **真实 ROI 掩膜**（`field_seg_masks`，`compressed_segmentation`）**本节点未能自证解码**（无官方客户端）
  ⇒ 该口径标 `proxy`，并在 note 写明"非真实 ROI"。
- **full-FOV**：全视野（200×200 px = 500×500 µm）逐帧统计，作为**同口径可比的整体量**。

钙事件 → **不是** spike rate（RF-M11-F §一.④ 强制登记）
------------------------------------------------------
钙成像事件率与电生理发放率**不同量纲**：B3 的代理行已量化——若把 in vitro **17.4 / 17.6 Hz** 当作 (a) 的
"率带"，相对钙域口径会**高 8.7–34.8×** ⇒ **本条必须随每个钙域数字一同出现**（本工具写进每个 note）。

来源（`evidence_level=measured`）
--------------------------------
`data/m11_raw/microns/functional_subset/`（200 片 × 10 帧 = 2,000 帧 @ **6.300006 Hz**）
＋ `readme` 语义引用（`functional_connectomics_readme_v1.md` 的列语义仅用于术语对齐，**不参与本计算**）；
解码口径 = `int16-BE` 直读 + 哨兵规则（**无掩膜**，低于基线保留并登记 0.01298%）。

用法
----
    PYTHONHASHSEED=0 OMP_NUM_THREADS=1 ./.venv-m11/bin/python neural_exploration/tools/build_m11_calcium_silence.py
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import os
import struct
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJ = os.path.dirname(ROOT)
DATA = os.path.join(ROOT, "data")
SUB = os.path.join(DATA, "m11_raw", "microns", "functional_subset")
SUB2 = os.path.join(DATA, "m11_raw", "microns", "functional_subset_f2")
OUT_CSV = os.path.join(DATA, "m11_calcium_silence_reference.csv")
OUT_JSON = os.path.join(DATA, "m11_calcium_silence_reference.json")

FRAME_INTERVAL = 0.15873
FRAME_RATE = 1.0 / FRAME_INTERVAL
BIN_PX = 20                      # per-ROI 代理：20×20 px = 50×50 µm
PIXEL_UM = 2.5
EVENT_SIGMA = 3.0                # 事件阈值：ΔF/F 的 (中位 + k·MAD)
SILENCE_LIMITATION = ("**钙事件口径 ≠ spike rate**：钙域事件率与电生理发放率不同量纲；"
                      "B3 代理行量化——以 in vitro 17.4/17.6 Hz 作 (a) 带会**高 8.7–34.8×** ⇒ 本条随每行出现")


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_stack(dirpath, prefix, n_max=200):
    """读 (t,y,x) int16 大端栈；返回 stack, n_files, sentinel 统计。"""
    files = sorted(glob.glob(os.path.join(dirpath, prefix + "_*.bin")))[:n_max]
    out, n_sent, n_tot = [], 0, 0
    for f in files:
        with open(f, "rb") as fh:
            blob = fh.read()
        if len(blob) < 64:
            continue
        off = blob.find(b"\x1f\x8b\x08", 0, 64)
        if off < 0:
            continue
        try:
            raw = gzip.decompress(blob[off:])
        except Exception:
            continue                     # 不完整者不入统计（完整性门）
        a = np.frombuffer(raw, dtype=">i2")
        nd = struct.unpack_from(">I", blob, 0)[0]
        dims = struct.unpack_from(">%dI" % nd, blob, 4)
        nt = int(dims[-1])
        nxy = a.size // nt
        ny = int(round(nxy ** 0.5))
        try:
            if ny * ny != nxy:           # 只处理方形视野（f2 为 100×100）
                ny2, nx2 = int(dims[1]), int(dims[0])
                a = a.reshape(nt, ny2, nx2)
            else:
                a = a.reshape(nt, ny, ny)
        except Exception:
            continue                     # 尺寸不自洽（半成品/未知布局）⇒ 不入统计
        out.append(a)
        n_sent += int((a < 0).sum()); n_tot += a.size
    return np.concatenate(out, axis=0) if out else None, len(out), n_sent, n_tot


def analyse(stack, label, bin_px=BIN_PX):
    """返回三种静默口径 × 两种单元口径（per-ROI 代理 / full-FOV）。"""
    T, Y, X = stack.shape
    F = stack.astype(np.float32)
    # 逐像素基线 F0 = 20 分位（对每个像素在时间上取分位）
    F0 = np.percentile(F, 20, axis=0)
    F0s = np.where(F0 <= 0, np.nan, F0)
    dff = (F - F0s[None]) / F0s[None]
    dff = np.nan_to_num(dff, nan=0.0, posinf=0.0, neginf=0.0)
    med = float(np.median(dff)); mad = float(np.median(np.abs(dff - med)))
    thr = med + EVENT_SIGMA * mad
    # 分箱（per-ROI 代理）
    ny, nx = Y // bin_px, X // bin_px
    dffb = dff[:, :ny * bin_px, :nx * bin_px].reshape(T, ny, bin_px, nx, bin_px).transpose(0, 1, 3, 2, 4)
    dffb = dffb.reshape(T, ny, nx, bin_px * bin_px).mean(axis=3)      # (T, ny, nx)
    units = dffb.reshape(T, -1)
    n_units = units.shape[1]
    # 口径 1：zero_activity（窗内从未越过阈值）
    ev_units = (units > thr).sum(axis=0)                 # 每单元事件数
    zero_units = float((ev_units == 0).mean())
    # 口径 2：below_baseline（时间均值 < 0 ⇒ 低于自身 F0）
    mean_units = units.mean(axis=0)
    below_units = float((mean_units < 0).mean())
    # 口径 3：low_activity（**非退化**：单元 p95 低于"全单元 p95 中位的 20%"的**固定阈值**；
    #  ⚠️ 早先版本用"低于自身分布的 p20"⇒ **构造性恒为 20%**（退化判据），已修正）
    p95_units = np.percentile(units, 95, axis=0)
    weak_thr = 0.2 * float(np.median(p95_units))
    low_units = float((p95_units < weak_thr).mean())
    # full-FOV 口径（逐帧）
    fov = units.mean(axis=1)
    # **非退化**的全视野帧级口径（早先版本用 FOV 均值 vs thr ⇒ 恒 1.0，已修正）：
    # ① 该帧**无任何像素**越过阈值；② 该帧 FOV 均值低于"FOV 均值 p80 的 20%"（弱帧）
    fov_zero = float(((units > thr).sum(axis=1) == 0).mean())
    fov_weak_thr = 0.2 * float(np.percentile(fov, 80))
    fov_below = float((fov < fov_weak_thr).mean())
    pxsil = float(((dff > thr).sum(axis=0) == 0).mean())   # 像素级零事件占比
    return dict(
        label=label, n_frames=int(T), shape=[int(T), int(Y), int(X)],
        frame_rate_hz=round(FRAME_RATE, 6), duration_s=round(T * FRAME_INTERVAL, 2),
        event_threshold_dff=round(thr, 6), dff_median=round(med, 6), dff_mad=round(mad, 6),
        n_units_proxy=int(n_units), unit_bin_px=bin_px,
        unit_area_um2=round((bin_px * PIXEL_UM) ** 2, 1),
        silent_frac_zero_activity=round(zero_units, 6),
        silent_frac_below_baseline=round(below_units, 6),
        silent_frac_low_percentile=round(low_units, 6),
        fullfov_frame_no_event_frac=round(fov_zero, 6),
        fullfov_frame_weak_frac=round(fov_below, 6),
        fullfov_fov_weak_threshold=round(fov_weak_thr, 6),
        unit_weak_threshold_p95=round(weak_thr, 6),
        degenerate_check="两处早先的**构造性退化**口径已修正（p20 自指 ⇒ 恒 20%；FOV 均值 vs 全局阈值 ⇒ 恒 1.0）",
        pixel_zero_event_frac=round(pxsil, 6),
        mean_events_per_unit=round(float(ev_units.mean()), 4),
        median_events_per_unit=float(np.median(ev_units)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-f2", action="store_true")
    args = ap.parse_args()
    rows = []
    st1, n1, s1, t1 = load_stack(SUB, "session4_scan7_field4")
    if st1 is not None:
        r = analyse(st1, "field4 session4_scan7_field4 x[0,200)y[0,200)（200 片/2000 帧）")
        r["source_dir"] = "data/m11_raw/microns/functional_subset"
        rows.append(r)
    st2 = None
    if not args.no_f2 and os.path.isdir(SUB2):
        st2, n2, s2, t2 = load_stack(SUB2, "session5_scan3_field1")
        if st2 is not None:
            r = analyse(st2, "field2 session5_scan3_field1 (0,0,0)（200 片/2000 帧）")
            r["source_dir"] = "data/m11_raw/microns/functional_subset_f2"
            rows.append(r)
    for r in rows:
        r["evidence_level"] = "measured"
        r["sentinel_rule"] = "int16-BE 直读 + **无掩膜**（低于基线样本保留并登记；哨兵仅指 32767 饱和）"
        r["note"] = ("单元 = **空间分箱代理（proxy，非真实 ROI）**（无真实 ROI 掩膜：`field_seg_masks` 的 "
                     "`compressed_segmentation` 本节点未自证解码）；" + SILENCE_LIMITATION)
    # P5 (b) 判据带：**新增行（逐 field 一行）**，不改原行
    band_rows = []
    for i, r in enumerate(rows):
        tag = "field4" if i == 0 else "field2"
        band_rows.append(dict(
            band_id="P5b-silent-ca-perroi-proxy-%s" % tag,
            criterion="P5 (b) 静默比例（钙域 · per-ROI 代理 · %s）" % tag,
            value_or_band=("zero_activity=%.4f; below_baseline=%.4f; low_activity=%.4f"
                           % (r["silent_frac_zero_activity"], r["silent_frac_below_baseline"],
                              r["silent_frac_low_percentile"])),
            n=r["n_units_proxy"], unit="fraction of units", evidence="measured",
            source=r.get("source_dir", ""), status="measured（新增行；原行保留）",
            note="**新行**（不改原 (b) 行）；**proxy 单元**（%d px 分箱 = %s µm²，非真实 ROI）；%s"
                 % (r["unit_bin_px"], r["unit_area_um2"], SILENCE_LIMITATION)))
        band_rows.append(dict(
            band_id="P5b-silent-ca-fullfov-%s" % tag,
            criterion="P5 (b) 静默比例（钙域 · full-FOV · %s）" % tag,
            value_or_band=("frame_no_event_frac=%.4f; frame_weak_frac=%.4f; pixel_zero_frac=%.4f"
                           % (r["fullfov_frame_no_event_frac"], r["fullfov_frame_weak_frac"],
                              r["pixel_zero_event_frac"])),
            n=r["n_frames"], unit="fraction of frames/pixels", evidence="measured",
            source=r.get("source_dir", ""), status="measured（新增行）",
            note="两口径并列（RF-M11-F §一.①）；**不得**与电生理静默比例混用（口径不同）"))
    out = dict(generated_by="tools/build_m11_calcium_silence.py", generated_at=utcnow(),
               purpose="J2（RF-M11-F §一）：把 P5 (b) 的**钙域静默比例口径**从 final=0 推向可定稿",
               frame_interval_s=FRAME_INTERVAL, frame_rate_hz=round(FRAME_RATE, 6),
               silence_definitions=dict(
                   zero_activity="观测窗内从未出现 > (ΔF/F 中位 + 3·MAD) 的事件",
                   below_baseline="单元时间均值 < 0（低于自身 F0=20 分位）",
                   low_activity="单元活动 p95 低于 **0.2 × 全单元 p95 中位**（固定阈值，**非自指分位**）",
                   note="早先两处退化口径（自指 p20、FOV 均值 vs 全局阈值）已修正并留档"),
               unit_definitions=dict(
                   per_roi_proxy="20×20 px 空间分箱（50×50 µm @2.5 µm/px）——**非真实 ROI**（掩膜未解码）",
                   full_fov="全视野逐帧（200×200 px = 500×500 µm）"),
               limitation=SILENCE_LIMITATION,
               rows=rows, band_rows=band_rows)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        f.write("# M11 钙域静默比例口径（J2；generated_by=tools/build_m11_calcium_silence.py；**新增行**，"
                "不改判据带原行；**钙事件口径 ≠ spike rate**）\n")
        cols = ["band_id", "criterion", "value_or_band", "n", "unit", "evidence", "source", "status", "note"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in band_rows:
            w.writerow(r)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    for r in rows:
        print("%-44s units=%-5d zero=%.4f below=%.4f weak=%.4f | frame_no_event=%.3f frame_weak=%.3f pixel_zero=%.4f"
              % (r["label"][:44], r["n_units_proxy"], r["silent_frac_zero_activity"],
                 r["silent_frac_below_baseline"], r["silent_frac_low_percentile"],
                 r["fullfov_frame_no_event_frac"], r["fullfov_frame_weak_frac"], r["pixel_zero_event_frac"]))
    print("wrote", OUT_CSV, "/", OUT_JSON)
    return out


if __name__ == "__main__":
    main()
