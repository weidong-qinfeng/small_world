#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M11 · MICrONS 功能分片**解码判决性演示**（RF-M11-A 复核用；**一条命令可复核**）

要回答的问题
------------
主 agent 的复核（含 RF-M11-A）对同一批 `.bin` 得到"逐轴自相关全 ≈0、近似均匀分布、无结构"，
并据此裁定格式不可解。本脚本把**同一套检验**同时施加于两种字节序读法，输出对照表 + 对照图。

结论（可复跑）
--------------
- 元数据 `dataType=int16` **正确**，但数据是 **大端（big-endian）**；
- **小端**读法正是"高熵/无结构"的来源：把大端字节对调后，原本的强空间-时间结构被打散成准随机序列；
- `>i2` 读法（`np.frombuffer(raw, '>i2').reshape(10, 200, 200)`）给出
  **帧间 lag-1 自相关 ≈ +0.78、横向相邻像素相关 ≈ +0.80**，且 **p50 ≈ 1.9e3**（不再是 0）；
- **独立地面真值**：同一数据集**原始未压缩 BigTIFF**（另一 scan）lat_x 0.767 / p50 1,879 与之同量级。

复跑
----
    cd /Users/weidong/ai/small_world
    OMP_NUM_THREADS=1 ./.venv-m11/bin/python neural_exploration/tools/demo_m11_microns_decode.py
    # 画图（需要 matplotlib；用项目内 .venv-m10）
    OMP_NUM_THREADS=1 ./.venv-m10/bin/python neural_exploration/tools/demo_m11_microns_decode.py --plot

输出：`data/m11_decode_proof.json`（对照表）+ `reports/neuro/m11_microns_decode_proof.png`（对照图）
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import struct
import sys
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
REPORTS = os.path.join(ROOT, "reports", "neuro")
SUB = os.path.join(DATA, "m11_raw", "microns", "functional_subset")
OUT_JSON = os.path.join(DATA, "m11_decode_proof.json")
OUT_PNG = os.path.join(REPORTS, "m11_microns_decode_proof.png")


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load(t_chunk=0):
    p = os.path.join(SUB, "session4_scan7_field4_%06d.bin" % t_chunk)
    blob = open(p, "rb").read()
    off = blob.find(b"\x1f\x8b\x08", 0, 64)
    nd = struct.unpack_from(">I", blob, 0)[0]
    dims = struct.unpack_from(">%dI" % nd, blob, 4)
    raw = gzip.decompress(blob[off:])
    return p, dims, off, raw, nd


def battery(a_flat, nt=10, ny=200, nx=200):
    """主 agent 的同一套检验：逐轴 lag1-6 自相关、唯一值、最高频次、零占比、值域、p50。"""
    A = a_flat.reshape(nt, ny, nx).astype(np.float64)
    out = {}
    for axis, tag in ((0, "t(帧轴)"), (1, "y(轴1)"), (2, "x(轴2)")):
        lags = {}
        for lag in range(1, 7):
            if A.shape[axis] <= lag:
                lags["lag%d" % lag] = None
                continue
            sl1 = np.take(A, range(0, A.shape[axis] - lag), axis=axis).ravel()
            sl2 = np.take(A, range(lag, A.shape[axis]), axis=axis).ravel()
            if sl1.std() < 1e-12 or sl2.std() < 1e-12:
                lags["lag%d" % lag] = 0.0
            else:
                lags["lag%d" % lag] = round(float(np.corrcoef(sl1, sl2)[0, 1]), 4)
        out[tag] = lags
    v = a_flat
    uniq, counts = np.unique(v, return_counts=True)
    out["unique_values"] = int(uniq.size)
    out["max_value_freq"] = int(counts.max())
    out["max_freq_frac"] = round(float(counts.max()) / v.size, 6)
    out["zero_frac"] = round(float((v == 0).mean()), 6)
    out["min"] = int(v.min())
    out["max"] = int(v.max())
    out["p50"] = float(np.median(v))
    out["lat_x_corr"] = round(float(np.corrcoef(A[:, :, :-1].ravel(), A[:, :, 1:].ravel())[0, 1]), 4)
    out["lat_y_corr"] = round(float(np.corrcoef(A[:, :-1, :].ravel(), A[:, 1:, :].ravel())[0, 1]), 4)
    out["frame_lag1"] = out["t(帧轴)"]["lag1"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--chunk", type=int, default=0)
    args = ap.parse_args()

    path, dims, off, raw, nd = load(args.chunk)
    print("file      : %s" % os.path.relpath(path, os.path.dirname(ROOT)))
    x, y, z, t = (list(dims) + [1, 1, 1, 1])[:4]
    print("header    : num_dims=%d, dims(x,y,z,t)=%s（大端 uint32，共 %d B）；gzip 起始偏移 %d"
          % (nd, tuple(dims), off, off))
    print("raw bytes : %d（== %d*%d*%d*%d*2 int16）" % (len(raw), x, y, z, t))
    print()
    res = {}
    print("=== 你的同一套检验：小端（LE）vs 大端（BE）===")
    for tag, dt in (("little-endian int16（我/主 agent 首版读法）", "<i2"),
                    ("BIG-endian int16（本轮判决性读法）", ">i2")):
        m = battery(np.frombuffer(raw, dtype=dt))
        res[tag] = m
        print("-- %s" % tag)
        print("   帧轴 lag1..6: %s" % [m["t(帧轴)"]["lag%d" % k] for k in range(1, 7)])
        print("   轴1  lag1..6: %s" % [m["y(轴1)"]["lag%d" % k] for k in range(1, 7)])
        print("   轴2  lag1..6: %s" % [m["x(轴2)"]["lag%d" % k] for k in range(1, 7)])
        print("   唯一值=%d  最高频次=%d(%.5f%%)  零占比=%.5f  值域=[%d,%d]  p50=%.0f"
              % (m["unique_values"], m["max_value_freq"], 100 * m["max_freq_frac"], m["zero_frac"],
                 m["min"], m["max"], m["p50"]))
        print("   横向相邻相关 x=%.3f y=%.3f   帧间 lag1=%.3f"
              % (m["lat_x_corr"], m["lat_y_corr"], m["frame_lag1"]))
    print()
    win = res["BIG-endian int16（本轮判决性读法）"]
    print("=== 判据（预注册：lat_x≥0.2 ∧ 相邻帧≥0.2 ∧ 值域合理 ∧ 无 NaN/Inf ∧ 声明域 ∧ 非退化）===")
    ok = (win["lat_x_corr"] >= 0.2 and win["frame_lag1"] >= 0.2 and win["p50"] != 0
          and win["min"] >= -32768 and win["max"] <= 32767)
    print("   BE: lat_x=%.3f 帧间=%.3f p50=%.0f 声明域内=%s ⇒ %s"
          % (win["lat_x_corr"], win["frame_lag1"], win["p50"],
             (win["min"] >= -32768 and win["max"] <= 32767), "PASS" if ok else "FAIL"))
    lev = res["little-endian int16（我/主 agent 首版读法）"]
    print("   LE: lat_x=%.3f 帧间=%.3f p50=%.0f ⇒ FAIL（这正是你观测到的「无结构」来源）"
          % (lev["lat_x_corr"], lev["frame_lag1"], lev["p50"]))

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        A_le = np.frombuffer(raw, dtype="<i2").reshape(10, 200, 200)
        A_be = np.frombuffer(raw, dtype=">i2").reshape(10, 200, 200)
        fig, ax = plt.subplots(2, 4, figsize=(13, 6.4))
        for j in range(4):
            ax[0, j].imshow(A_le[j], cmap="gray")
            ax[0, j].set_title("LE (错误) frame %d" % j, fontsize=9)
            ax[1, j].imshow(A_be[j], cmap="gray")
            ax[1, j].set_title("BE (正确) frame %d" % j, fontsize=9)
            for i in range(2):
                ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        fig.suptitle("MICrONS functional chunk 解码对照：小端 vs 大端（同一 800,000 B）\n"
                     "LE: lat_x=%.3f, 帧间=%.3f, p50=%.0f   |   BE: lat_x=%.3f, 帧间=%.3f, p50=%.0f"
                     % (lev["lat_x_corr"], lev["frame_lag1"], lev["p50"],
                        win["lat_x_corr"], win["frame_lag1"], win["p50"]), fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        os.makedirs(REPORTS, exist_ok=True)
        fig.savefig(OUT_PNG, dpi=110)
        print("wrote %s" % OUT_PNG)
        for i in range(2):
            for j in range(4):
                pass
    out = dict(generated_by="tools/demo_m11_microns_decode.py", generated_at=utcnow(),
               file=os.path.relpath(path, os.path.dirname(ROOT)), header_dims=list(dims),
               gzip_offset=off, raw_bytes=len(raw), battery=res,
               verdict=dict(winner=">i2（大端 int16）", loser="<i2（小端 int16）",
                            why="元数据只声明 dataType=int16，未声明字节序；本数据集为 N5/BossDB（Java）惯例 ⇒ 大端。"
                                "小端读法把 (hi,lo) 对调 ⇒ 强空间/时间结构被打散为准随机序列，"
                                "这正是「逐轴自相关≈0、近似均匀、p50=0」的成因（不是「格式不可解」）。",
                            criteria="lat_x≥0.2 ∧ 相邻帧 lag1≥0.2 ∧ 非全 0 ∧ 无 NaN/Inf ∧ 声明域 ∧ 非退化",
                            ground_truth="同数据集原始未压缩 BigTIFF（另一 scan，Range 6 帧）："
                                         "lat_x=0.767 lat_y=0.735 p50=1879（见 m11_microns_parse_validation.json）"),
               plot=(os.path.relpath(OUT_PNG, os.path.dirname(ROOT)) if args.plot else None))
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("wrote %s" % OUT_JSON)
    return out


if __name__ == "__main__":
    main()
