#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M11-B3b · MICrONS 功能体素序列的**独立复核**：解码 + 同源成像目标带（dwell / τ）

独立性声明
----------
本工具**未参照** M11-B1 的实现（不读其代码），解码路径由本节点独立写就并**自行验证**：
    20 B 大端头 `>5I` = (ndim=4, 200, 200, 1, 10) → `gzip.decompress(raw[20:])` → `>i2` → reshape (200,200,1,10)
下游估计器**复用 M10 冻结口径**（`validate_p10_state_structure` 的 `ar1_fit` / `hmm_kstate` / `_dwells` /
`_consistency` / `sample_size_gate`），以保证"与 B4 完全相同的管线"；**本工具唯一的自由度是箱宽**，
故 RF-33 要求"明确写出箱宽"，本工具**并列两条路径**：

    (A) 原生帧箱：bin = 1 帧 ⇒ dt = 1/fps = 0.1587300 s（n=2000，帧率原生、不引入额外低通）
    (B) B4 可比箱：bin = 6 帧 ⇒ dt = 0.9523810 s（**注意不是 0.9138 s**：6.300006 Hz 下 0.9138 s
        恰为 5.7573 帧，非整数；取最近整数 6 帧 ⇒ 箱宽比 B4 宽 4.2%。**整箱是刻意的**：
        模型侧管线即为"按步数整除成箱"，整数箱避免引入插值/相位自由参数）

每条路径都同时报 **平滑（5 帧）** 与 **未平滑** 两种口径（RF-32 教训：平滑本身会放大 τ）。

必须自证的三件事（RF-33 §1）
    ① 空间/时间结构（相邻像素相关、相邻帧相关）——**报实数**；
    ② 哨兵掩膜比例——**独立复算**，并**精确化判据**（见下）；
    ③ 字节序对照——同一对象在 `>i2` 与 `<i2` 下结构指标并列（把"字节序是口径维度"做成证据）。

哨兵判据的**精确化**（本节点发现的口径点，务必保留）
    M11-B1 表述为"`uint16` 读法 > 65000 ⇔ `int16` < 0"。两式**并不恒等**：
      · `int16 < 0`          ⇔ 该 16 位字的高位为 1（int16 ∈ [−32768, −1]）
      · 同字节序 `uint16` > 65000 ⇔ int16 ∈ [−535, −1]（是上者的**子集**）
    二者**仅当全体负值都 ≥ −535 时**才等价。本工具**同时报三个计数**（neg16 / u16gt / u16gt−neg16），
    并显式给出等价性是否成立，避免把"近似等价"当"恒等"用。

用法（CPU、单线程、无 GPU；预计 <10 min）
    cd /Users/weidong/ai/small_world
    PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MPLBACKEND=Agg ./.venv-m9/bin/python \
        neural_exploration/tools/verify_m11_microns_series.py
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import hashlib
import json
import math
import os
import re
import struct
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # neural_exploration/
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))
from neural_exploration.tools.validate_p10_state_structure import (  # noqa: E402
    MIN_FRAMES_BIC, MIN_FRAMES_SHAPE, MIN_FRAMES_VALID, TAU_MIN_FRAMES,
    _consistency, _dwells, ar1_fit, hmm_kstate, sample_size_gate,
)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "m11_raw", "microns", "functional_subset")
OUT_JSON = os.path.join(DATA_DIR, "_b3_m11_series_reference.json")
OUT_CSV = os.path.join(DATA_DIR, "_b3_m11_series_reference.csv")

FPS = 6.300006                 # MICrONS 功能成像帧率（RF-33 给定）
FRAME_DT_S = 1.0 / FPS         # 原生帧间隔 = 0.1587300 s
B4_BIN_S = 1.0 / 1.0943        # B4 成像箱宽 = 0.9138261903 s（目标，非整数帧）
SMOOTH_FRAMES = 5              # B4 管线：5 帧滑动平均

# ------------------------------------------------------------------ 结构指标累加器
class CorrAcc:
    """流式累积 Pearson 相关所需的一阶/二阶矩（避免把 2000×40000 全量留在内存）。"""

    __slots__ = ("n", "sx", "sy", "sxx", "syy", "sxy")

    def __init__(self):
        self.n = 0
        self.sx = self.sy = self.sxx = self.syy = self.sxy = 0.0

    def add(self, x, y):
        x = np.asarray(x, np.float64).ravel()
        y = np.asarray(y, np.float64).ravel()
        self.n += x.size
        self.sx += x.sum(); self.sy += y.sum()
        self.sxx += (x * x).sum(); self.syy += (y * y).sum(); self.sxy += (x * y).sum()

    def r(self):
        if self.n == 0:
            return None
        mx = self.sx / self.n; my = self.sy / self.n
        vx = self.sxx / self.n - mx * mx
        vy = self.syy / self.n - my * my
        if vx <= 0 or vy <= 0:
            return None
        cov = self.sxy / self.n - mx * my
        return float(cov / math.sqrt(vx * vy))


def bin_series(values_per_frame: np.ndarray, bin_frames: int, fps: float = FPS):
    """按**整数帧**成箱（mean），返回 (binned, dt_bin_s)。

    与 M10 `frame_series` 同步骤（成箱 → 平滑 → z），仅把"箱宽"参数化；
    取 mean 还是 sum 对 z 后的 AR(1) φ 无影响（z 分数对正比例缩放不变）。
    """
    n = values_per_frame.size // bin_frames
    if n < 2:
        return np.zeros(0, float), bin_frames / fps
    b = values_per_frame[: n * bin_frames].reshape(n, bin_frames).mean(axis=1).astype(float)
    return b, bin_frames / fps


def smooth_z(binned: np.ndarray, smooth_frames: int):
    """与 M10 `frame_series` 完全相同的后两步：5 帧滑动平均（mode=valid）→ z 分数。"""
    if binned.size >= smooth_frames:
        k = np.ones(smooth_frames) / smooth_frames
        sm = np.convolve(binned, k, mode="valid")
    else:
        sm = binned
    z = (sm - sm.mean()) / (sm.std() + 1e-12)
    return z


def phi_unsmoothed(binned: np.ndarray):
    x = np.asarray(binned, float)
    if x.size < 20:
        return None
    x = (x - x.mean()) / (x.std() + 1e-12)
    x0, x1 = x[:-1], x[1:]
    d = float(x0 @ x0)
    return float((x0 @ x1) / d) if d > 0 else None


def build_band(trace_frames: np.ndarray, bin_frames: int, label: str, source: str, fps: float = FPS):
    """一条目标带：AR(1)（平滑/未平滑）+ Q 法 dwell + H 法 K=2 dwell + BIC(K) + 一致性。"""
    binned, dt = bin_series(trace_frames, bin_frames, fps)
    n = binned.size
    z = smooth_z(binned, SMOOTH_FRAMES)
    out = dict(label=label, source=source, bin_frames=int(bin_frames), dt_bin_s=float(dt),
               n_frames_binned=int(n), n_frames_smoothed=int(z.size), trace_s_total=float(n * dt),
               b4_target_bin_s=float(B4_BIN_S), bin_vs_b4_rel=float(dt / B4_BIN_S - 1.0),
               sample_size_gate=sample_size_gate(int(n)))
    a_sm = ar1_fit(z, dt, folds=4, lb_lags=10)
    a_sm.setdefault("status", "invalid")
    a_sm["pipeline"] = ("帧箱(bin=%d, dt=%.6f s) → %d 帧滑动平均 → z → AR(1)"
                        % (bin_frames, dt, SMOOTH_FRAMES))
    a_sm["phi_smoothed"] = (float((z[:-1] @ z[1:]) / (z[:-1] @ z[:-1])) if z.size > 20 else None)
    a_sm["phi_unsmoothed"] = phi_unsmoothed(binned)
    pu = a_sm["phi_unsmoothed"]
    a_sm["tau_unsmoothed_s"] = (float(-dt / math.log(pu)) if (pu is not None and 0 < pu < 1) else None)
    a_sm["min_frames_for_tau"] = int(TAU_MIN_FRAMES)
    a_sm["tau_criterion_status"] = ("unavailable（n_frames<%d）" % TAU_MIN_FRAMES) if n < TAU_MIN_FRAMES \
        else ("measured" if a_sm.get("status") == "measured" else "invalid（AR(1) 不可估）")
    out["AR1"] = a_sm
    # --- Q 法（分位/中位切分；与 M10 同口径：在**未平滑**箱序列上切）---
    if n >= 2:
        thr = float(np.quantile(binned, 0.5))
        qstate = (binned > thr).astype(np.int8)
        out["Q_method"] = dict(threshold=thr,
                               occupancy=[float((qstate == 0).mean()), float((qstate == 1).mean())],
                               transition_rate_hz=float((np.diff(qstate) != 0).sum() / (n * dt)),
                               dwell=_dwells(qstate, dt))
    else:
        out["Q_method"] = dict(status="invalid", reason="n_frames<2")
    # --- H 法（2 状态高斯 HMM，seed=0；M10 冻结口径）---
    if z.size >= 20:
        h2 = hmm_kstate(z, 2, n_iter=60, seed=0)
        st = h2["state"]
        out["H_K2"] = dict(occupancy=[float((st == 0).mean()), float((st == 1).mean())],
                           transition_rate_hz=float((np.diff(st) != 0).sum() / (z.size * dt)),
                           dwell=_dwells(st, dt), bic=float(h2["bic"]), loglik=float(h2["loglik"]))
        ks = {K: hmm_kstate(z, K, n_iter=60, seed=0) for K in (1, 2, 3, 4)}
        out["BIC_by_K"] = {K: dict(bic=float(v["bic"]), loglik=float(v["loglik"]), n_par=int(v["n_par"]))
                           for K, v in ks.items()}
        out["best_K_by_BIC"] = int(min(ks, key=lambda k: ks[k]["bic"])) if n >= MIN_FRAMES_BIC else "unavailable"
        if n < MIN_FRAMES_BIC:
            out["BIC_note"] = "n_frames=%d < %d ⇒ K 选择结论 unavailable（RF-18(4)）" % (n, MIN_FRAMES_BIC)
        out["internal_consistency"] = _consistency(
            out["H_K2"]["transition_rate_hz"],
            ((out["H_K2"]["dwell"].get("low") or {}).get("median_s")))
    else:
        out["H_K2"] = dict(status="invalid", reason="z 过短 n=%d" % z.size)
        out["internal_consistency"] = dict(applicable=False, reason="z 过短")
    out["dwell_resolution_floor"] = dict(
        at_floor=bool(((out.get("Q_method", {}).get("dwell", {}) or {}).get("low", {}) or {}).get("median_s", 0)
                      <= dt * 1.5),
        note="dwell 中位 ≤1.5 帧 ⇒ 触分辨率地板、无分辨力（仅作旁证）")
    return out


def classify(rf, wlo, whi):
    """成像侧"慢"三分类（为交叉核对预备；RF-33 未要求硬判，故仅登记）。"""
    if rf is None:
        return "unavailable（未平滑 φ 不可估）"
    if rf >= whi:
        return "slow(≫平滑窗)"
    if rf >= wlo:
        return "slow(同量级平滑窗)"
    return "fast(τ<平滑窗)"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=RAW_DIR)
    ap.add_argument("--fps", type=float, default=FPS)
    ap.add_argument("--bins", default="1,6", help="成箱帧数（1=原生帧箱；6≈B4 箱）")
    a = ap.parse_args(argv)

    t_start = time.time()
    files = sorted(glob.glob(os.path.join(a.raw_dir, "*.bin")))
    if not files:
        print("[FATAL] 无 .bin：%s" % a.raw_dir, file=sys.stderr)
        return 2
    fields = sorted({re.sub(r"_\d+\.bin$", "", os.path.basename(f)) for f in files})
    print("对象数 = %d | 字段前缀 = %s | 文件总字节 = %d"
          % (len(files), fields, sum(os.path.getsize(f) for f in files)))

    # ---------------- 单遍解码：自证三件事 + 留 trace 所需的数组 ----------------
    arrays, idxs = [], []
    sentinels = dict(neg16=0, u16gt=0, both=0, u16gt_not_neg=0, total=0)
    gmin = None
    smetrics = {">": dict(px_h=CorrAcc(), px_v=CorrAcc(), ft=CorrAcc()),
                "<": dict(px_h=CorrAcc(), px_v=CorrAcc(), ft=CorrAcc())}
    p50 = {">": [], "<": []}
    hdr_ok, shapes = 0, set()
    for k, f in enumerate(files):
        raw = open(f, "rb").read()
        hdr = struct.unpack(">5I", raw[:20])
        ndim = hdr[0]
        shape = tuple(int(x) for x in hdr[1:1 + ndim])
        shapes.add(shape)
        if shape == (200, 200, 1, 10):
            hdr_ok += 1
        dec = gzip.decompress(raw[20:])
        for tag, dt_ in ((">", ">i2"), ("<", "<i2")):
            arr = np.frombuffer(dec, dtype=dt_)
            if tag == ">":
                mat = arr.reshape(shape)[..., 0].astype(np.int16)      # (200,200,10)
                arrs = mat
                u = mat.view(">u2")                                    # **同字节序**的无符号读法
                neg = mat < 0
                gt = u > 65000
                sentinels["neg16"] += int(neg.sum())
                sentinels["u16gt"] += int(gt.sum())
                sentinels["both"] += int((neg & gt).sum())
                sentinels["u16gt_not_neg"] += int((gt & ~neg).sum())
                sentinels["total"] += int(mat.size)
                m = int(mat.min())
                gmin = m if gmin is None else min(gmin, m)
                p50[">"].append(float(np.percentile(mat, 50)))
                arrays.append(mat)
                idxs.append(k)
            else:
                arrs = arr.reshape(shape)[..., 0].astype(np.int16)
                p50["<"].append(float(np.percentile(arrs, 50)))
            # 空间结构：横向/纵向相邻像素
            smetrics[tag]["px_h"].add(arrs[:, :-1, :], arrs[:, 1:, :])
            smetrics[tag]["px_v"].add(arrs[:-1, :, :], arrs[1:, :, :])
            # 时间结构：相邻帧（同一体素）
            if arrs.shape[-1] >= 2:
                smetrics[tag]["ft"].add(arrs[..., :-1], arrs[..., 1:])
        if (k + 1) % 50 == 0:
            print("  解码进度 %d/%d（CPU %.1fs / 墙钟 %.1fs）"
                  % (k + 1, len(files), time.process_time(), time.time() - t_start))

    sentinels["ratio_of_total"] = sentinels["neg16"] / max(sentinels["total"], 1)
    sentinels["equiv_int16_neg_iff_u16gt65000"] = (sentinels["neg16"] == sentinels["u16gt"])
    print("\n[① 哨兵] int16<0 = %d / %d = %.6f%% ；同字节序 uint16>65000 = %d ；"
          "u16gt 但非 int16<0 = %d ⇒ 判据%s" %
          (sentinels["neg16"], sentinels["total"], 100.0 * sentinels["ratio_of_total"],
           sentinels["u16gt"], sentinels["u16gt_not_neg"],
           "**等价**" if sentinels["equiv_int16_neg_iff_u16gt65000"] else "**不等价**"))
    print("        全局 int16 最小值 = %d（负值全 ≥ −535 才能与 uint16>65000 等价）" % gmin)

    print("\n[③ 字节序对照] 同一批对象的**结构指标**（相邻像素/相邻帧 Pearson r）：")
    print("  %-8s %-22s %-22s %s" % ("字节序", "横向相邻像素", "纵向相邻像素", "相邻帧"))
    st = {}
    for tag, nm in ((">", ">i2（正解）"), ("<", "<i2（对照）")):
        row = dict(px_h=smetrics[tag]["px_h"].r(), px_v=smetrics[tag]["px_v"].r(), ft=smetrics[tag]["ft"].r(),
                   p50_median=float(np.median(p50[tag])), p50_min=float(min(p50[tag])),
                   p50_max=float(max(p50[tag])), n_prod=smetrics[tag]["px_h"].n)
        st[nm] = row
        print("  %-8s %-22s %-22s %s" % (tag, row["px_h"], row["px_v"], row["ft"]))
    print("  %-8s p50 中位 = %.1f（范围 %.1f–%.1f）" % (">i2", st[">i2（正解）"]["p50_median"],
                                                       st[">i2（正解）"]["p50_min"], st[">i2（正解）"]["p50_max"]))
    print("  %-8s p50 中位 = %.1f（范围 %.1f–%.1f）" % ("<i2", st["<i2（对照）"]["p50_median"],
                                                       st["<i2（对照）"]["p50_min"], st["<i2（对照）"]["p50_max"]))

    # ---------------- 群体轨迹（两个口径的"活动量"）----------------
    # 内存说明：逐对象拼接为 (200,200,2000) int16 ⇒ 200·200·2000·2 B = **160 MB**（可控）；
    #            不在此处做 float64 全量转换（会到 640 MB）。
    vol = np.concatenate(arrays, axis=-1)                      # (200,200,2000) int16
    del arrays
    n_frames = vol.shape[-1]
    ever_sent = (vol < 0).any(axis=-1)
    print("\n[② 群体轨迹] 体素数 = %d ；曾出现哨兵的体素 = %d（%.4f%%）⇒ 从轨迹中剔除"
          % (vol.shape[0] * vol.shape[1], int(ever_sent.sum()),
             100.0 * ever_sent.mean()))
    valid = ~ever_sent
    trace_mean = vol[valid].mean(axis=1).astype(np.float64)          # 空间均值 df/F
    thr_hi = float(np.percentile(vol[valid], 99.0))
    trace_frac = (vol[valid] > thr_hi).mean(axis=1).astype(np.float64)   # 高分位"活跃体素占比"
    del vol
    print("  轨迹长度 = %d 帧 = %.1f s（@%.6f Hz）" % (trace_mean.size, trace_mean.size / a.fps, a.fps))
    print("  口径A(空间均值 df/F)：mean=%.2f sd=%.2f | 口径B(>p99 活跃占比)：mean=%.5f" %
          (trace_mean.mean(), trace_mean.std(), trace_frac.mean()))

    # ---------------- 目标带 ----------------
    bands = []
    for bf in [int(x) for x in a.bins.split(",")]:
        for src, tr in (("spatial_mean_dfF", trace_mean), ("active_fraction_p99", trace_frac)):
            bands.append(build_band(tr, bf, "bin%d_%s" % (bf, src), src, a.fps))
    print("\n=== 目标带（与 B4 同管线；箱宽已显式声明）===")
    print("%-30s %8s %10s %10s %8s %10s %10s %10s"
          % ("band", "n_bin", "dt(s)", "τ_sm(s)", "φ_sm", "φ_unsm", "τ_unsm(s)", "Q_dwell低(s)"))
    for b in bands:
        ar = b["AR1"]
        ql = ((b.get("Q_method", {}).get("dwell", {}) or {}).get("low", {}) or {}).get("median_s")
        print("%-30s %8d %10.6f %10.3f %8.4f %10s %10s %10s"
              % (b["label"], b["n_frames_binned"], b["dt_bin_s"],
                 ar.get("tau_s") or float("nan"), ar.get("phi_smoothed") or float("nan"),
                 ("%.4f" % ar["phi_unsmoothed"]) if ar.get("phi_unsmoothed") is not None else "n/a",
                 ("%.3f" % ar["tau_unsmoothed_s"]) if ar.get("tau_unsmoothed_s") else "n/a",
                 ("%.3f" % ql) if ql else "n/a"))

    payload = dict(
        task="M11-B3b 独立复核：MICrONS 功能体素序列解码 + 同源成像目标带",
        independence="本工具未参照 M11-B1 实现；解码路径独立写就并自证；下游估计器复用 M10 冻结口径",
        raw=dict(dir=os.path.relpath(a.raw_dir, os.path.dirname(PROJECT_ROOT)),
                 n_objects=len(files), n_bytes=sum(os.path.getsize(f) for f in files),
                 field_prefixes=fields, header_shapes=sorted(str(s) for s in shapes),
                 header_ok_objects=hdr_ok,
                 decode_path="20B >5I 头(ndim=4 + 4 dims) → gzip@offset20 → >i2 → reshape"),
        fps=a.fps, frame_dt_s=1.0 / a.fps, b4_target_bin_s=B4_BIN_S,
        total_frames=int(n_frames), total_seconds=float(n_frames / a.fps),
        self_proof=dict(sentinels=sentinels, global_min_int16=gmin, structure=st),
        traces=dict(spatial_mean_dfF=dict(mean=float(trace_mean.mean()), sd=float(trace_mean.std())),
                    active_fraction_p99=dict(mean=float(trace_frac.mean()), thr_hi=thr_hi),
                    n_voxels_valid=int(valid.sum()), n_voxels_ever_sentinel=int(ever_sent.sum())),
        bands=bands,
        gates=dict(MIN_FRAMES_VALID=MIN_FRAMES_VALID, MIN_FRAMES_SHAPE=MIN_FRAMES_SHAPE,
                   TAU_MIN_FRAMES=TAU_MIN_FRAMES, MIN_FRAMES_BIC=MIN_FRAMES_BIC),
        wallclock_s=float(time.time() - t_start), cpu_s=float(time.process_time()),
        determinism="gzip 确定性 + HMM seed=0 + 无随机采样 ⇒ 可复跑逐位一致",
    )
    payload["artifacts_sha256_file_bytes"] = {}
    json.dump(payload, open(OUT_JSON, "w"), ensure_ascii=False, indent=2)

    # 长表 CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        f.write("# M11-B3b MICrONS 序列目标带（长表）——box 宽已显式声明；basis=file_bytes（RF-31 纪律）\n")
        w = csv.writer(f)
        w.writerow(["band", "bin_frames", "dt_bin_s", "n_binned", "n_smoothed", "metric", "value", "unit", "note"])
        for b in bands:
            ar = b["AR1"]
            rows = [
                ("tau_smoothed", ar.get("tau_s"), "s", ar.get("pipeline", "")),
                ("tau_unsmoothed", ar.get("tau_unsmoothed_s"), "s", "5 帧平滑的对照口径"),
                ("phi_smoothed", ar.get("phi_smoothed"), "-", ""),
                ("phi_unsmoothed", ar.get("phi_unsmoothed"), "-", ""),
                ("tau_criterion_status", ar.get("tau_criterion_status"), "-", "RF-21(1) n≥300 门"),
                ("Q_dwell_low_median", ((b["Q_method"].get("dwell", {}).get("low") or {}).get("median_s")), "s", "**未平滑**箱序列中位切分"),
                ("Q_transition_hz", b["Q_method"].get("transition_rate_hz"), "Hz", ""),
                ("Q_occ_low", (b["Q_method"].get("occupancy") or [None])[0], "-", ""),
                ("H_dwell_low_median", (((b.get("H_K2", {}).get("dwell", {}) or {}).get("low", {}) or {}).get("median_s")), "s", "2 状态 HMM(seed=0)"),
                ("H_transition_hz", b.get("H_K2", {}).get("transition_rate_hz"), "Hz", ""),
                ("H_occ_low", (b.get("H_K2", {}).get("occupancy") or [None])[0], "-", ""),
                ("best_K_by_BIC", b.get("best_K_by_BIC"), "-", b.get("BIC_note", "")),
                ("internal_consistency_rel", (b.get("internal_consistency") or {}).get("rel_diff"), "-",
                 "|dwell−1/(2λ)|/dwell ≤1 为 pass"),
                ("sample_gate_bic_ok", (b["sample_size_gate"] or {}).get("bic_ok"), "-", ""),
            ]
            for m, v, u, nt in rows:
                w.writerow([b["label"], b["bin_frames"], "%.6f" % b["dt_bin_s"], b["n_frames_binned"],
                            b["n_frames_smoothed"], m, v, u, nt])

    for p in (OUT_JSON, OUT_CSV):
        raw = open(p, "rb").read()
        payload["artifacts_sha256_file_bytes"][os.path.basename(p)] = hashlib.sha256(raw).hexdigest()
    json.dump(payload, open(OUT_JSON, "w"), ensure_ascii=False, indent=2)
    print("\n[OK] 落盘 %s / %s" % (os.path.basename(OUT_JSON), os.path.basename(OUT_CSV)))
    for k, v in payload["artifacts_sha256_file_bytes"].items():
        print("     %s  sha256(file_bytes)=%s" % (k, v[:16]))
    print("[耗时] CPU %.1fs / 墙钟 %.1fs（比值 %.2f ⇒ %s）"
          % (payload["cpu_s"], payload["wallclock_s"],
             payload["cpu_s"] / max(payload["wallclock_s"], 1e-9),
             "单线程满负荷" if payload["cpu_s"] / max(payload["wallclock_s"], 1e-9) > 0.8 else "**未满负荷，需查争用**"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
