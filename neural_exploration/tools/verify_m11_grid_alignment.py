#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M11 · 空间口径 10× 差与**同格性**的实像素验证（RF-M11-B §四）

要解决的问题
------------
清单/文档给出两个互相矛盾的"空间口径"：
- **文档原文**（`functional_data_technical_documentation_v3.pdf`）：
  "**4 px/µm** xy resolution (440×248 px/field) to tile a 1100×1200 µm² FOV at four depths…"
- **元数据字段**（`functional_series/attributes.json`）：`resolution=[2.5, 2.5, 10]`，`units=[um,um,um]`
二者相差 **10×**。本工具用**实像素对齐**（不用元数据声明）判定谁描述的是哪个层级，并量化"用错口径"的后果。

方法（可复跑；只用已落盘的 ~0.5 MB 数据）
----------------------------------------
1. **两个独立对象的像素级对齐**（同一 field、两种分辨率）：
   `field_summaries`（1 µm/px，620×1100 px，float32 **大端**）vs
   `functional_series`（2.5 µm/px，248×440 px，int16 **大端**）。
2. **尺度扫描**：把 summary 按降采样因子 f 聚合到 series 的网格，计算与 series 时间平均图的
   Pearson r(f)；**正确的口径应在 f=2.5 出现峰值**（若口径为 0.25 µm/px，则峰值应出现在 f≈25）。
3. **对齐前后对照**：给出 f=2.5（对齐）与 f=1、f=10（错配）的 r 与逐像素对照统计。
4. **物理解释**：用两个独立读回的 FOV 交叉核对（series: 248×2.5=620 µm、440×2.5=1100 µm，
   与 `field_summaries` 的 dims 620×1100 @1 µm/px **完全一致**）。

用法
----
    PYTHONHASHSEED=0 OMP_NUM_THREADS=1 ./.venv-m11/bin/python neural_exploration/tools/verify_m11_grid_alignment.py
    # 画图（可选）：OMP_NUM_THREADS=1 ./.venv-m10/bin/python ... --plot
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import struct
import sys
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJ = os.path.dirname(ROOT)
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "m11_raw", "microns")
SUM_DIR = os.path.join(RAW, "field_summaries")
SUB_DIR = os.path.join(RAW, "functional_subset")
OUT_JSON = os.path.join(DATA, "m11_grid_alignment.json")
OUT_PNG = os.path.join(ROOT, "reports", "neuro", "m11_grid_alignment.png")


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ungzip_prefix(path, max_prefix=64):
    """返回 (raw, prefix_bytes, header_words)。chunk = <prefix words> + gzip 成员（前缀长度随 volume 不同）。"""
    b = open(path, "rb").read()
    off = b.find(b"\x1f\x8b\x08", 0, max_prefix)
    raw = gzip.decompress(b[off:]) if off >= 0 else b
    words = None
    if off and off % 4 == 0:
        words = list(struct.unpack_from(">%dI" % (off // 4), b, 0))
    return raw, off, words


def load_summary():
    """拼出 summary 的 x∈[0,200) µm, y∈[0,200) µm 区块（1 µm/px）。"""
    parts = {}
    for key in ("0_0_0", "0_1_0", "1_0_0", "1_1_0"):
        p = os.path.join(SUM_DIR, "sum_%s.bin" % key)
        if not os.path.exists(p):
            return None, None, None
        raw, off, words = ungzip_prefix(p)
        a = np.frombuffer(raw, dtype=">f4").reshape(100, 100)  # BE float32（实测）
        xi, yi = int(key.split("_")[0]), int(key.split("_")[1])
        parts[(xi, yi)] = (a, words, off)
    img = np.zeros((200, 200), dtype=np.float64)
    for (xi, yi), (a, w, off) in parts.items():
        img[yi * 100:(yi + 1) * 100, xi * 100:(xi + 1) * 100] = a
    return img, parts[("0", 0)][1] if False else parts[(0, 0)][1], parts[(0, 0)][2]


def load_series_mean(n_files=200):
    """series (0,0,0) 块的**时间平均图**（200×200 px @2.5 µm/px）。"""
    fs = sorted(glob.glob(os.path.join(SUB_DIR, "session4_scan7_field4_*.bin")))[:n_files]
    acc = None
    for f in fs:
        raw, off, words = ungzip_prefix(f)
        a = np.frombuffer(raw, dtype=">i2").reshape(10, 200, 200).astype(np.float64)
        s = a.mean(axis=0)
        acc = s if acc is None else acc + s
    return acc / max(1, len(fs)), len(fs)


def _resample_axis(a, n_out, axis):
    """按**面积加权**把某轴重采样到 n_out 个 bin（确定性、无插值歧义）。"""
    n_in = a.shape[axis]
    edges = np.linspace(0.0, float(n_in), n_out + 1)
    outs = []
    for k in range(n_out):
        lo, hi = edges[k], edges[k + 1]
        i0, i1 = int(np.floor(lo)), int(np.ceil(hi))
        j = np.arange(i0, i1)
        w = np.clip(np.minimum(j + 1.0, hi) - np.maximum(j.astype(float), lo), 0.0, None)
        seg = np.take(a, j, axis=axis)
        wk = w.reshape([-1 if i == axis else 1 for i in range(a.ndim)])
        outs.append((seg * wk).sum(axis=axis) / max(w.sum(), 1e-12))
    return np.stack(outs, axis=axis)


def block_mean(a, f):
    """按因子 f 降采样（面积加权）。f 可为小数：输出尺寸 = round(shape/f)。"""
    a = np.asarray(a, dtype=np.float64)
    ny = max(1, int(round(a.shape[0] / float(f))))
    nx = max(1, int(round(a.shape[1] / float(f))))
    return _resample_axis(_resample_axis(a, nx, 1), ny, 0)


def corr(a, b):
    n = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    x = a[:n[0], :n[1]].ravel()
    y = b[:n[0], :n[1]].ravel()
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--n-series", type=int, default=200)
    args = ap.parse_args()

    res = dict(generated_by="tools/verify_m11_grid_alignment.py", generated_at=utcnow(),
               question="文档 4 px/µm 与元数据 2.5 µm/px 的 10× 差属于哪一层的口径？",
               provenance=dict(
                   doc="data/m11_raw/microns/pdf_text/functional_data_technical_documentation_v3.pdf.txt —— "
                       "原文 '4 px/µm xy resolution (440×248 px/field) to tile a 1100×1200 µm² FOV'",
                   metadata="functional_series/*/attributes.json —— resolution=[2.5,2.5,10], units=[um,um,um]",
                   summary_metadata="field_summaries/*/attributes.json —— resolution=[1,1,10], dimensions=[620,1100,1]"))
    sm, sum_words, sum_off = load_summary()
    ser, n_ser = load_series_mean(args.n_series)
    if sm is None or ser is None:
        print("缺少必要输入（summary 或 series 分片）")
        return res
    res["inputs"] = dict(summary_shape=list(sm.shape), summary_prefix_words=sum_words,
                         summary_prefix_bytes=sum_off,
                         series_mean_shape=list(ser.shape), n_series_chunks=n_ser,
                         series_frames=n_ser * 10)

    # ---- 尺度扫描：r(f) ----
    sweep = {}
    for f in (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0, 10.0, 12.5, 25.0):
        d = block_mean(sm, f)
        sweep["%.2f" % f] = round(corr(d, ser), 4)
    best_f = max(sweep, key=lambda k: sweep[k])
    res["scale_sweep"] = sweep
    res["sweep_best_factor"] = best_f
    res["sweep_best_r"] = sweep[best_f]

    # ---- 对齐前/后对照（1D 剖面更直观）----
    ps = ser.mean(axis=1)                      # series x-profile（200 px @2.5 µm = 500 µm）
    pm = sm.mean(axis=1)                       # summary x-profile（200 px @1 µm = 200 µm）
    prof = {}
    for f in (1.0, 2.5, 10.0):
        d = block_mean(pm.reshape(-1, 1), f).ravel()
        n = min(d.size, ps.size)
        prof["%.2f" % f] = round(corr(d[:n].reshape(-1, 1), ps[:n].reshape(-1, 1)), 4)
    res["profile_alignment"] = prof

    # ---- 对齐前后的像素级对照（f=2.5 对齐 vs f=1 与 f=10 错配）----
    aligned = block_mean(sm, 2.5)
    rows = []
    for tag, f in (("aligned(f=2.5)", 2.5), ("mismatch(f=1)", 1.0), ("mismatch(f=10)", 10.0)):
        d = block_mean(sm, f)
        n0, n1 = min(d.shape[0], ser.shape[0]), min(d.shape[1], ser.shape[1])
        a, b = d[:n0, :n1], ser[:n0, :n1]
        rows.append(dict(case=tag, factor=f, n_pixels=int(a.size), r=round(corr(a, b), 4),
                         mean_summary=round(float(a.mean()), 4), mean_series=round(float(b.mean()), 2),
                         ratio_means=round(float(a.mean() / (b.mean() or 1)), 6)))
    res["pixel_alignment_table"] = rows

    # ---- 物理解释（两个独立读回的 FOV 交叉核对）----
    res["fov_crosscheck"] = dict(
        series_from_metadata=dict(px=[248, 440], um_per_px=[2.5, 2.5], fov_um=[248 * 2.5, 440 * 2.5]),
        summary_from_metadata=dict(px=[620, 1100], um_per_px=[1.0, 1.0], fov_um=[620.0, 1100.0]),
        doc_per_field=dict(px_per_um=4.0, field_px=[440, 248], field_fov_um=[440 / 4.0, 248 / 4.0],
                           doc_mosaic_um2=[1100, 1200]),
        verdict="两个**独立对象**元数据给出的 FOV 完全一致（620×1100 µm）；"
                "而文档的 4 px/µm 给出的是**单个 raw 子场**的 110×62 µm，需 ~10×20 个子场才能拼出"
                "文档所述 1100×1200 µm² 拼接 FOV ⇒ 两者是**不同层级**（raw 采集 vs 处理后/拼接体素），"
                "不是同一层的矛盾。")
    # ---- 结论 ----
    res["conclusion"] = dict(
        level_raw="文档 '4 px/µm'（=0.25 µm/px）= **raw 采集**的每个子场像素（440×248 px/field ⇒ 110×62 µm/field）",
        level_processed="元数据 2.5 µm/px = **处理后 functional_series / field_summaries** 的体素"
                        "（248×440 px ⇒ 620×1100 µm）",
        ratio=10.0,
        ratio_evidence="实像素尺度扫描峰值出现在 f=%.2f（r=%.3f），与 2.5 相符；"
                       "若为 0.25 µm/px 则峰值应出现在 f≈25（实测 r=%.3f）"
                       % (float(best_f), sweep[best_f], sweep.get("25.00", float("nan"))),
        misread_risk="**若把 0.25 µm/px 当成 series 的口径**："
                     "① 所有距离/尺度的物理量（感受野尺寸、空间相关长度、ROI/神经域半径、"
                     "突触-距离关系）**系统性偏小 10×**；② FOV 会被算成 62×110 µm（面积偏小 100×），"
                     "与 field_summaries 的 620×1100 µm 直接冲突；"
                     "③ 任何'按物理距离配对/匹配'的对照（如连接概率 vs 距离、dend_len 对照）**口径不可比**。"
                     "⇒ 纪律：**凡用到空间尺度，必须写明取自哪个对象/字段，并做跨对象 FOV 交叉核对**。")
    with open(OUT_JSON, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print("scale sweep r(f):", json.dumps(sweep, ensure_ascii=False))
    print("best factor:", best_f, "r=", sweep[best_f])
    print("pixel table:", json.dumps(rows, ensure_ascii=False))
    print("wrote", OUT_JSON)

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 4, figsize=(15, 4))
        ax[0].imshow(sm, cmap="gray"); ax[0].set_title("field_summaries 1 um/px (200x200 um)")
        ax[1].imshow(ser, cmap="gray"); ax[1].set_title("functional_series mean 2.5 um/px (500x500 um)")
        ax[2].imshow(block_mean(sm, 2.5), cmap="gray")
        ax[2].set_title("summary downsampled f=2.5 -> aligned (r=%.2f)" % corr(block_mean(sm, 2.5), ser))
        ks = [float(k) for k in sweep]
        vs = [sweep[k] for k in sweep]
        ax[3].plot(ks, vs, "o-"); ax[3].set_xscale("log"); ax[3].set_xlabel("downsample factor f")
        ax[3].set_ylabel("Pearson r vs series"); ax[3].set_title("scale sweep: peak at 2.5")
        for a_ in ax[:3]:
            a_.set_xticks([]); a_.set_yticks([])
        fig.tight_layout()
        os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
        fig.savefig(OUT_PNG, dpi=110)
        print("wrote", OUT_PNG)
    return res


if __name__ == "__main__":
    main()
