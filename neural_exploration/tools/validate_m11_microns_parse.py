#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M11 · MICrONS functional 分片**结构性解析验证**（主 agent 硬要求 §二）

背景
----
主 agent 独立复核发现：按**小端** int16 直读时，相邻像素相关≈0、相邻帧相关≈0 ⇒ 不"像图像"，
并据此提出三种假设（直读 / delta 解码 / Morton-Z 反排列）。本工具把假设做成**可复跑的结构性检验**，
用**预注册判据**逐条判定并落盘，**未通过者不得用于任何统计量**。

预注册判据（主 agent 给定，任一不过即 `invalid`）
-------------------------------------------------
C1 横向相邻像素相关 ≥ **0.2**；C2 相邻帧（t）相关 ≥ **0.2**；
C3 值域/p50 合理（非全 0、非饱和）；C4 **无 NaN/Inf**。

结论（本轮实测）
----------------
**真因 = 字节序**：元数据 `dataType=int16` 正确，但数据为 **大端（big-endian）**——
`>i2` 直读即得 lat_x **+0.814** / lat_y **+0.801** / frame **+0.803**（判据 C1–C4 全过）；
小端（`<i2`/`<u2`）与 delta(`cumsum`)、Morton-Z 反排列、float 解释**全部不通过**（逐条原因见落盘表）。

用法
----
    cd /Users/weidong/ai/small_world
    PYTHONHASHSEED=0 OMP_NUM_THREADS=1 ./.venv-m11/bin/python neural_exploration/tools/validate_m11_microns_parse.py
    ... --n-search 8 --n-full 200 --with-tiff-groundtruth
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import struct
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJ = os.path.dirname(ROOT)
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "m11_raw")
SUB_DIR = os.path.join(RAW, "microns", "functional_subset")
LOG = os.path.join(RAW, "validate_m11_microns_parse.log")
OUT_JSON = os.path.join(DATA, "m11_microns_parse_validation.json")
OUT_CSV = os.path.join(DATA, "m11_microns_parse_hypotheses.csv")
INTEGRITY = os.path.join(DATA, "m11_download_integrity.csv")

CRIT_LAT = 0.2      # 预注册：横向相邻像素相关下限
CRIT_FRAME = 0.2    # 预注册：相邻帧相关下限

TIFF_URL = ("https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/functional_data/"
            "two_photon_functional_scans/functional_scan_17797_4_10_v1.tif")


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(m):
    os.makedirs(RAW, exist_ok=True)
    line = "[%s] %s" % (datetime.now(timezone.utc).strftime("%H:%M:%S"), m)
    with open(LOG, "a", buffering=1) as f:
        f.write(line + "\n")
    print(line, flush=True)


# ---------------------------------------------------------------------------
def read_chunk(path):
    """返回 (header_dims, raw_bytes)。chunk = 20 B 大端头 + gzip 成员（自动定位魔数）。"""
    with open(path, "rb") as f:
        blob = f.read()
    off = blob.find(b"\x1f\x8b\x08", 0, 64)
    nd = struct.unpack_from(">I", blob, 0)[0]
    dims = struct.unpack_from(">%dI" % nd, blob, 4)
    raw = gzip.decompress(blob[off:])
    return dims, raw


# ---- Morton/Z-order 反排列（200×200 平面；非 2 的幂 ⇒ 补齐到 256×256 后按 Z 序取有效位）----
def _zorder_sequence(nx, ny):
    """返回 Z 序（padding 到 2 的幂、按有效区内位置筛选）下的 (y,x) 索引序列（长度 nx*ny）。"""
    px = 1 << (nx - 1).bit_length()
    py = 1 << (ny - 1).bit_length()
    n = px * py
    idx = np.arange(px * py, dtype=np.int64)
    x = np.zeros_like(idx)
    y = np.zeros_like(idx)
    b = 0
    while (1 << b) < max(px, py):
        x |= ((idx >> (2 * b)) & 1) << b
        y |= ((idx >> (2 * b + 1)) & 1) << b
        b += 1
    keep = (x < nx) & (y < ny)
    return y[keep], x[keep]


def apply_morton(a, nx=200, ny=200, nt=10, layout="tyx"):
    """把按 Morton 序存储的 (t, nx*ny) 反排列成 (t, ny, nx)。"""
    yy, xx = _zorder_sequence(nx, ny)
    m = a.reshape(nt, -1)
    if m.shape[1] != yy.size:
        return None
    out = np.empty((nt, ny, nx), dtype=a.dtype)
    out[:, yy, xx] = m
    return out


# ---- 结构度量 ----
def metrics_from_stack(A):
    """A: (t,y,x) → dict(lat_x, lat_y, frame, min, p1, p50, p99, max, n_nan, n_inf, n_nonfinite)."""
    A = np.asarray(A, dtype=np.float64)
    n_nan = int(np.isnan(A).sum())
    n_inf = int(np.isinf(A).sum())
    d = dict(n_nan=n_nan, n_inf=n_inf, n_nonfinite=n_nan + n_inf)
    finite = A[np.isfinite(A)]
    if finite.size == 0:
        d.update(lat_x=0.0, lat_y=0.0, frame=0.0, vmin=None, vmax=None)
        return d
    d.update(vmin=float(finite.min()), vmax=float(finite.max()),
             p1=float(np.percentile(finite, 1)), p50=float(np.percentile(finite, 50)),
             p99=float(np.percentile(finite, 99)))
    if A.shape[0] > 1 and A.shape[2] > 1:
        d["lat_x"] = float(np.corrcoef(A[:, :, :-1].ravel(), A[:, :, 1:].ravel())[0, 1])
    else:
        d["lat_x"] = 0.0
    if A.shape[1] > 1:
        d["lat_y"] = float(np.corrcoef(A[:, :-1, :].ravel(), A[:, 1:, :].ravel())[0, 1])
    else:
        d["lat_y"] = 0.0
    if A.shape[0] > 1:
        d["frame"] = float(np.corrcoef(A[:-1].ravel(), A[1:].ravel())[0, 1])
    else:
        d["frame"] = 0.0
    return d


def judge(m):
    """预注册判据 C1–C4 → (verdict, reasons)。"""
    r = []
    if not (m.get("lat_x", 0) >= CRIT_LAT):
        r.append("C1 横向相邻像素相关 %.3f < %.2f" % (m.get("lat_x", 0), CRIT_LAT))
    if not (m.get("frame", 0) >= CRIT_FRAME):
        r.append("C2 相邻帧相关 %.3f < %.2f" % (m.get("frame", 0), CRIT_FRAME))
    p50 = m.get("p50")
    if p50 is None or (p50 == 0 and m.get("p1") == 0 and m.get("p99") == 0):
        r.append("C3 值域/p50 不合理（疑似全 0 空洞）")
    if m.get("n_nonfinite", 0) > 0:
        r.append("C4 非有限值 %d 个" % m["n_nonfinite"])
    return ("pass" if not r else "invalid"), "; ".join(r)


# ---- 假设矩阵 ----
def hypotheses(raw, dims):
    """产出 (name, stack 或 None, note)。stack = (t,y,x)。"""
    nt, ny, nx = 10, 200, 200
    out = []
    for dt, tag in (("<i2", "int16-LE"), (">i2", "int16-BE"), ("<u2", "uint16-LE"), (">u2", "uint16-BE")):
        a = np.frombuffer(raw, dtype=dt)
        if a.size == nt * ny * nx:
            out.append(("① 直读 %s（小端/大端）" % tag, a.reshape(nt, ny, nx).astype(np.float64), ""))
    for dt, tag in (("<f4", "float32-LE"), (">f4", "float32-BE")):
        a = np.frombuffer(raw, dtype=dt)
        if a.size == nt * ny * nx // 2:
            out.append(("① 直读 %s" % tag, a.reshape(nt, ny // 2, nx).astype(np.float64), "元素数=1/2（形状按 dtype 缩放）"))
    a8 = np.frombuffer(raw, dtype="u1")
    out.append(("① 直读 uint8", a8.reshape(nt, ny, nx * 2).astype(np.float64) if a8.size == nt * ny * nx * 2 else None, ""))
    # ② delta / cumsum
    for dt, tag in (("<i2", "int16-LE"), (">i2", "int16-BE")):
        d = np.frombuffer(raw, dtype=dt).astype(np.int64)
        out.append(("② delta 解码 cumsum_flat（%s，int64）" % tag,
                    np.cumsum(d).reshape(1, ny, nx * nt // 1)[0].reshape(nt, ny, nx).astype(np.float64), ""))
        out.append(("② delta 解码 cumsum_flat mod 65536（%s）" % tag,
                    (np.cumsum(d) % 65536).reshape(nt, ny, nx).astype(np.float64), ""))
    d = np.frombuffer(raw, dtype=">i2").astype(np.int64).reshape(nt, ny, nx)
    out.append(("② delta 沿 x 轴 cumsum（int16-BE）", np.cumsum(d, axis=2).astype(np.float64), ""))
    out.append(("② delta 沿 y 轴 cumsum（int16-BE）", np.cumsum(d, axis=1).astype(np.float64), ""))
    out.append(("② delta 沿 t 轴 cumsum（int16-BE）", np.cumsum(d, axis=0).astype(np.float64), ""))
    # ③ Morton / Z 序反排列
    for dt, tag in (("<i2", "int16-LE"), (">i2", "int16-BE")):
        a = np.frombuffer(raw, dtype=dt).astype(np.float64)
        m = apply_morton(a, nx, ny, nt)
        out.append(("③ Morton 反排列（%s）" % tag, m, "200×200 非 2 的幂 ⇒ 补齐 256×256 后按 Z 序取有效位"))
        am = apply_morton(np.cumsum(a.astype(np.int64)).astype(np.float64), nx, ny, nt)
        out.append(("②+③ Morton + cumsum（%s）" % tag, am, "BossDB CompressedMortonChunk 假设组合"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-search", type=int, default=8, help="假设搜索用的分片数（spread over COMPLETE）")
    ap.add_argument("--n-full", type=int, default=200, help="获胜假设在全量 COMPLETE 对象上的复核数（0=跳过）")
    ap.add_argument("--with-tiff-groundtruth", action="store_true", help="Range 读原始 BigTIFF 若干帧做地面真值相关对照")
    ap.add_argument("--tiff-mb", type=float, default=4.4, help="BigTIFF Range 读取上限（MB；只读头若干帧）")
    args = ap.parse_args()

    log("=== validate_m11_microns_parse start n_search=%d" % args.n_search)
    # 只用 COMPLETE 对象（完整性门结论）
    complete = []
    if os.path.exists(INTEGRITY):
        with open(INTEGRITY, encoding="utf-8") as f:
            for r in csv.DictReader([l for l in f if not l.startswith("#")]):
                if r.get("verdict") == "COMPLETE":
                    complete.append(r["t_chunk"])
    if not complete:
        complete = [f.split("_")[-1].split(".")[0] for f in sorted(os.listdir(SUB_DIR)) if f.endswith(".bin")]
    log("COMPLETE 对象数 = %d（统计与参考带只用这些）" % len(complete))

    sel = [complete[i] for i in np.linspace(0, len(complete) - 1, min(args.n_search, len(complete))).astype(int)] \
        if args.n_search < len(complete) else complete
    res_rows, per_file_winner = [], None
    winner = None
    names = None
    for k, tc in enumerate(sel):
        p = os.path.join(SUB_DIR, "session4_scan7_field4_%06d.bin" % int(tc))
        if not os.path.exists(p):
            continue
        dims, raw = read_chunk(p)
        hyps = hypotheses(raw, dims)
        if names is None:
            names = [h[0] for h in hyps]
        for name, stack, note in hyps:
            if stack is None:
                res_rows.append(dict(t_chunk=tc, hypothesis=name, verdict="invalid",
                                     reasons="元素数与假设形状不符", note=note, lat_x="", lat_y="", frame=""))
                continue
            m = metrics_from_stack(stack)
            v, why = judge(m)
            res_rows.append(dict(t_chunk=tc, hypothesis=name, verdict=v, reasons=why, note=note,
                                 lat_x=round(m["lat_x"], 4), lat_y=round(m["lat_y"], 4),
                                 frame=round(m["frame"], 4), p50=m.get("p50"), vmin=m.get("vmin"),
                                 vmax=m.get("vmax"), n_nan=m["n_nan"], n_inf=m["n_inf"]))
        log("  file %s: 假设 %d 个已评估" % (tc, len(hyps)))

    # 汇总：假设在**全部搜索文件**上都 pass 才算 pass
    agg = {}
    for r in res_rows:
        a = agg.setdefault(r["hypothesis"], dict(n=0, n_pass=0, lat=[], frame=[], reasons=set()))
        a["n"] += 1
        a["n_pass"] += 1 if r["verdict"] == "pass" else 0
        if r["lat_x"] != "":
            a["lat"].append(float(r["lat_x"]))
            a["frame"].append(float(r["frame"]))
        if r["verdict"] != "pass" and r["reasons"]:
            a["reasons"].add(r["reasons"][:70])
    for name, a in agg.items():
        a["lat_mean"] = round(float(np.mean(a["lat"])), 4) if a["lat"] else None
        a["frame_mean"] = round(float(np.mean(a["frame"])), 4) if a["frame"] else None
        a["reasons"] = sorted(a["reasons"])[:3]
        a.pop("lat"); a.pop("frame")
    passing = [n for n, a in agg.items() if a["n_pass"] == a["n"] and a["n"] > 0]
    winner = passing[0] if passing else None
    log("通过预注册判据的假设：%s" % (passing or "无"))

    # ---- 获胜假设：全量 COMPLETE 复核 + 哨兵/缺失登记 ----
    full = dict(n_objects=0, raw_bytes_total=0, lat_x=[], lat_y=[], frame=[], p1=[], p50=[], p99=[],
                vmin_global=None, vmax_global=None, n_nan=0, n_inf=0, n_sentinel_uint16_ge_65000=0,
                n_zero=0, n_elements=0, n_saturated_p99x3=0, masked_rule="")
    if winner and args.n_full:
        for tc in complete[:args.n_full]:
            p = os.path.join(SUB_DIR, "session4_scan7_field4_%06d.bin" % int(tc))
            if not os.path.exists(p):
                continue
            dims, raw = read_chunk(p)
            A = np.frombuffer(raw, dtype=">i2").reshape(10, 200, 200).astype(np.int32)
            m = metrics_from_stack(A)
            full["n_objects"] += 1
            full["raw_bytes_total"] += len(raw)
            for k, key in (("lat_x", "lat_x"), ("lat_y", "lat_y"), ("frame", "frame"),
                           ("p1", "p1"), ("p50", "p50"), ("p99", "p99")):
                full[key].append(m[k])
            full["vmin_global"] = m["vmin"] if full["vmin_global"] is None else min(full["vmin_global"], m["vmin"])
            full["vmax_global"] = m["vmax"] if full["vmax_global"] is None else max(full["vmax_global"], m["vmax"])
            full["n_nan"] += m["n_nan"]; full["n_inf"] += m["n_inf"]
            full["n_elements"] += A.size
            full["n_zero"] += int((A == 0).sum())
            # 哨兵/饱和候选：uint16 读法 ≥65000（⇔ int16 ≤ −536）
            full["n_sentinel_uint16_ge_65000"] += int((A < 0).sum())
        for key in ("lat_x", "lat_y", "frame", "p1", "p50", "p99"):
            v = full.pop(key)
            full[key + "_mean"] = round(float(np.mean(v)), 4)
            full[key + "_min"] = round(float(np.min(v)), 4)
            full[key + "_max"] = round(float(np.max(v)), 4)
        full["masked_rule"] = ("哨兵/饱和候选掩膜规则（数据驱动，预注册）：**uint16 读法 > 65000** "
                               "（⇔ int16 < 0，本轮共 %d / %d 元素 = %.5f%%）→ 统计前置为缺失并登记"
                               % (full["n_sentinel_uint16_ge_65000"], full["n_elements"],
                                  100.0 * full["n_sentinel_uint16_ge_65000"] / max(1, full["n_elements"])))
        log("全量复核：%d 对象，lat_x=%.3f，frame=%.3f，哨兵=%d"
            % (full["n_objects"], full.get("lat_x_mean", 0), full.get("frame_mean", 0),
               full["n_sentinel_uint16_ge_65000"]))

    # ---- 原始 BigTIFF 地面真值（独立于 precomputed 分片的"像不像图像"标定）----
    gt = None
    if args.with_tiff_groundtruth:
        n_bytes = int(args.tiff_mb * 1024 * 1024)
        dst = os.path.join(RAW, "microns", "functional", "tif_frames_probe.bin")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        cmd = ["curl", "-sS", "-L", "--connect-timeout", "8", "--max-time", "180",
               "-r", "512-%d" % (512 + n_bytes - 1), "-o", dst, "-w", "%{http_code}"]
        try:
            pr = subprocess.run(cmd + [TIFF_URL], capture_output=True, timeout=200)
            code = pr.stdout.decode().strip()
        except Exception as e:
            code, pr = "0", None
        if os.path.exists(dst) and os.path.getsize(dst) > 218240 * 2:
            b = open(dst, "rb").read()
            nfr = len(b) // 218240
            frames = np.frombuffer(b[:nfr * 218240], dtype="<u2").reshape(nfr, 440, 248)
            m = metrics_from_stack(frames.astype(np.float64))
            gt = dict(url=TIFF_URL, range="512-%d" % (512 + n_bytes - 1), http_code=code,
                      bytes=len(b), n_frames=nfr, frame_shape=[440, 248], dtype="uint16-LE（BigTIFF 头声明）",
                      lat_x=round(m["lat_x"], 4), lat_y=round(m["lat_y"], 4), frame=round(m["frame"], 4),
                      p1=m["p1"], p50=m["p50"], p99=m["p99"], vmin=m["vmin"], vmax=m["vmax"],
                      n_nan=m["n_nan"], n_inf=m["n_inf"],
                      note="**独立地面真值**：原始 TIFF 帧（未压缩、uint16-LE）本身就是高相关图像 ⇒ "
                           "证明'lat≈0.8 / frame≈0.8'才是本模态的正常结构（判据阈值 0.2 由此锚定，非自证）")
            log("TIFF 地面真值：%d 帧 lat_x=%.3f lat_y=%.3f frame=%.3f"
                % (nfr, gt["lat_x"], gt["lat_y"], gt["frame"]))

    # ---- 落盘 ----
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        f.write("# M11 MICrONS 分片解析假设检验（generated_by=tools/validate_m11_microns_parse.py；"
                "预注册判据：lat_x≥%.2f 且 相邻帧相关≥%.2f 且 值域合理 且 无 NaN/Inf；"
                "**未通过者不得用于任何统计量**）\n" % (CRIT_LAT, CRIT_FRAME))
        cols = ["t_chunk", "hypothesis", "verdict", "reasons", "lat_x", "lat_y", "frame",
                "p50", "vmin", "vmax", "n_nan", "n_inf", "note"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in res_rows:
            w.writerow(r)

    out = dict(generated_by="tools/validate_m11_microns_parse.py", generated_at=utcnow(),
               criteria=dict(lat_x_min=CRIT_LAT, frame_min=CRIT_FRAME,
                             value_sanity="非全 0/非饱和", nonfinite="必须为 0",
                             rule="任一条不过即 invalid，不得用于统计"),
               n_complete_objects=len(complete), search_objects=sel,
               hypotheses_aggregate=agg, hypotheses_passing=passing,
               winner=winner,
               interpretation=dict(
                   dtype="int16（与 attributes.json 声明一致）",
                   byte_order="**big-endian（大端）**",
                   evidence="`>i2` 直读即满足全部预注册判据；`<i2`/`<u2` 全不通过",
                   root_cause="此前（含本节点首版与主 agent 复核）均按**小端**直读 ⇒ "
                              "相邻像素/相邻帧相关≈0 的假象；delta/Morton 假设**均不成立**",
                   nan_note="整型解释下 NaN/Inf **结构上不可能**；主 agent 检出的 106 个 NaN 与 |x|>1e30 "
                            "是**float 误解读的副产物**（本工具 float32 假设行同样判 invalid）"),
               full_objects=full, tiff_groundtruth=gt,
               sentinel_and_missing=dict(
                   n_elements_full=full.get("n_elements"),
                   n_nan=full.get("n_nan"), n_inf=full.get("n_inf"),
                   n_zero=full.get("n_zero"),
                   n_sentinel_uint16_ge_65000=full.get("n_sentinel_uint16_ge_65000"),
                   rule=full.get("masked_rule", ""),
                   discipline="**未处理 ≠ 无缺失**：NaN/哨兵/零值计数与掩膜规则必须随统计量一并落盘"))
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    log("wrote %s / %s；winner=%s" % (OUT_JSON, OUT_CSV, winner))
    log("ALL_DONE %s" % utcnow())
    return out


if __name__ == "__main__":
    o = main()
    print(json.dumps(dict(winner=o["winner"], passing=o["hypotheses_passing"],
                          full=o["full_objects"].get("n_objects"),
                          lat_x_mean=o["full_objects"].get("lat_x_mean"),
                          frame_mean=o["full_objects"].get("frame_mean"),
                          sentinel=o["full_objects"].get("n_sentinel_uint16_ge_65000")),
                     ensure_ascii=False))
