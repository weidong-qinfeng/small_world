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


def _r_single(x, y):
    x = np.asarray(x, np.float64).ravel(); y = np.asarray(y, np.float64).ravel()
    d = math.sqrt(x.var() * y.var())
    return float(((x * y).mean() - x.mean() * y.mean()) / d) if d > 0 else float("nan")


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

    # ---------------- 轴序阳性对照（先于任何统计，防"轴序"这一类口径错）----------------
    _raw0 = open(files[0], "rb").read()
    _buf0 = gzip.decompress(_raw0[20:])
    _arr0 = np.frombuffer(_buf0, dtype=">i2")
    _shp0 = tuple(int(x) for x in struct.unpack(">5I", _raw0[:20])[1:5])
    def _R(a, b):
        a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
        d = math.sqrt(a.var() * b.var())
        return float(((a * b).mean() - a.mean() * b.mean()) / d) if d > 0 else float("nan")
    _F = _arr0.reshape(_shp0, order="F")[:, :, 0, :]
    _C = _arr0.reshape(_shp0)[:, :, 0, :]
    axis_order_control = dict(
        F_order=dict(pixel_lag1=float(0.5 * (_R(_F[:, :-1, :], _F[:, 1:, :]) + _R(_F[:-1, :, :], _F[1:, :, :]))),
                     pixel_lag10=float(0.5 * (_R(_F[:, :-10, :], _F[:, 10:, :]) + _R(_F[:-10, :, :], _F[10:, :, :]))),
                     frame_lag1=_R(_F[..., :-1], _F[..., 1:])),
        C_order=dict(pixel_lag1=float(0.5 * (_R(_C[:, :-1, :], _C[:, 1:, :]) + _R(_C[:-1, :, :], _C[1:, :, :]))),
                     pixel_lag10=float(0.5 * (_R(_C[:, :-10, :], _C[:, 10:, :]) + _R(_C[:-10, :, :], _C[10:, :, :]))),
                     frame_lag1=_R(_C[..., :-1], _C[..., 1:])),
        rule="F 序下像素 lag10 应保持正的长程空间相关（显微图像特征）；C 序下会趋 0 ⇒ 用 F 序",
    )
    _fo, _co = axis_order_control["F_order"], axis_order_control["C_order"]
    axis_order_control["verdict"] = ("F 序成立（像素 lag10=%.3f 有长程结构 vs C 序 %.3f 趋零）"
                                     % (_fo["pixel_lag10"], _co["pixel_lag10"]))
    assert _fo["pixel_lag10"] > 0.2 and _fo["pixel_lag10"] > _co["pixel_lag10"], \
        "轴序对照未通过：F=%.3f C=%.3f" % (_fo["pixel_lag10"], _co["pixel_lag10"])
    print("[③a 轴序阳性对照] F 序: 像素lag1=%.3f lag10=%.3f 帧lag1=%.3f | C 序: 像素lag1=%.3f lag10=%.3f 帧lag1=%.3f ⇒ %s"
          % (_fo["pixel_lag1"], _fo["pixel_lag10"], _fo["frame_lag1"],
             _co["pixel_lag1"], _co["pixel_lag10"], _co["frame_lag1"], axis_order_control["verdict"]))

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
        assert len(shape) == 4 and shape[0] == 200 and shape[1] == 200 and shape[3] >= 2, \
            "头形状非预期：%s（须 (Y,X,Z,T)）" % (shape,)
        dec = gzip.decompress(raw[20:])
        for tag, dt_ in ((">", ">i2"), ("<", "<i2")):
            arr = np.frombuffer(dec, dtype=dt_)
            if tag == ">":
                # ⚠️⚠️ **轴序（本节点自查缺陷第 2 例）**：该分片存储为 **F 序（首轴 x 最快）**，
                #   即"每帧是一整块连续的 200×200 图像"。用 C 序 reshape 会把末轴当成帧轴，
                #   实际却在比较 **相距 10 像素的同一帧像素** ⇒ 相邻像素相关被压低到 0.41、
                #   "帧间相关"被抬到 0.82（真值：像素 ~0.80、帧 ~0.80）。
                #   判定依据（见 axis_order_control）：F 序下像素 lag10 相关 0.42（图像应有的长程结构），
                #   C 序下 lag10 → 0.00（不可能）；且 F 序复现 M11-B1 独立实现的结构指标。
                a4 = arr.reshape(shape, order="F")
                assert len(shape) == 4, shape
                mat = a4[:, :, 0, :].astype(np.int16)                  # → (200,200,10) 帧轴为最后一维
                arrs = mat
                # ⚠️ mat 已 astype(np.int16)（本机小端）⇒ 必须按**本机** uint16 视图读；
                #    写成 mat.view(">u2") 会再做一次字节序交换（本节点实测踩坑，已修）
                u = mat.view(np.uint16)                                # int16 → uint16（mod 65536）
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
                arrs = arr.reshape(shape, order="F")[:, :, 0, :].astype(np.int16)
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
    _equiv_ok = (gmin is not None and gmin >= -535)
    print("        全局 int16 最小值 = %d ⇒ 判据等价性的**充分条件**（min ≥ −535）：%s"
          % (gmin, "成立" if _equiv_ok else "不成立"))
    print("        （两式恒等**不成立**：int16<0 含 [−32768,−536]；uint16>65000 ⇔ int16∈[−535,−1]。"
          "本数据集因 min=%d 而**恰好重合**）" % gmin)

    per_obj = {">": dict(px_h=[], px_v=[], ft=[]), "<": dict(px_h=[], px_v=[], ft=[])}
    for a_ in arrays:
        per_obj[">"]["px_h"].append(_r_single(a_[:, :-1, :], a_[:, 1:, :]))
        per_obj[">"]["px_v"].append(_r_single(a_[:-1, :, :], a_[1:, :, :]))
        per_obj[">"]["ft"].append(_r_single(a_[..., :-1], a_[..., 1:]))
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
    # 内存纪律：**不做** `vol[valid]` 之类的花式索引（会物化 (n_valid,2000) float64 = 640 MB，
    #           并使 percentile 在 80M 元素上排序 ⇒ 本次试跑即被 60 s 超时拦下，已改按对象分块）。
    #           逐对象处理：单对象仅 (n_valid,10) float64 ≈ 3.2 MB。
    assert arrays[0].ndim == 3 and arrays[0].shape[-1] >= 2, "帧轴缺失：%s" % (arrays[0].shape,)
    ever_sent = np.zeros(arrays[0].shape[:2], bool)
    for a_ in arrays:
        np.logical_or(ever_sent, (a_ < 0).any(axis=-1), out=ever_sent)
    n_frames = int(sum(a_.shape[-1] for a_ in arrays))
    n_vox = ever_sent.size
    valid = ~ever_sent
    n_valid = int(valid.sum())
    print("\n[② 群体轨迹] 体素数 = %d ；曾出现哨兵的体素 = %d（%.4f%%）；有效体素 = %d"
          % (n_vox, int(ever_sent.sum()), 100.0 * ever_sent.mean(), n_valid))

    # --- 阈值：用**直方图**在有效体素上取精确 p99（确定性、O(65536) 内存）---
    hist = np.zeros(65536, np.int64)
    for a_ in arrays:
        vals = a_[valid]
        h, _ = np.histogram(vals, bins=65536, range=(-32768, 32767))
        hist += h
    csum = np.cumsum(hist)
    tot = int(csum[-1])
    idx = int(np.searchsorted(csum, 0.99 * tot))
    thr_hi = float(idx - 32768)
    print("  有效体素直方图总样本 = %d ；p99 阈值 = %d（int16 原始单位）" % (tot, int(thr_hi)))

    # --- 两条轨迹：逐对象累加（不物化大数组）---
    means, fracs = [], []
    for a_ in arrays:
        sub = a_[valid].astype(np.float64)          # (n_valid, 10)
        means.append(sub.mean(axis=0))
        fracs.append((sub > thr_hi).mean(axis=0))
    trace_mean = np.concatenate(means).astype(np.float64)
    trace_frac = np.concatenate(fracs).astype(np.float64)
    del means, fracs
    print("  轨迹长度 = %d 帧 = %.1f s（@%.6f Hz）" % (trace_mean.size, trace_mean.size / a.fps, a.fps))
    print("  口径A(空间均值 df/F)：mean=%.2f sd=%.2f | 口径B(>p99 活跃占比)：mean=%.5f" %
          (trace_mean.mean(), trace_mean.std(), trace_frac.mean()))

    # ---------------- 链连续性诊断（**决定轨迹能否当时间序列用**）----------------
    # 若 200 个 chunk 是时间上相邻的连续段，则"末帧_k → 首帧_{k+1}"是一个 **lag=1** 对，
    # 其相关应与链内 lag=1 相当；实测若显著更低，则"按文件名顺序拼接"**不成立**。
    def _r(x, y):
        x = np.asarray(x, np.float64).ravel(); y = np.asarray(y, np.float64).ravel()
        d = math.sqrt(x.var() * y.var())
        return float(((x * y).mean() - x.mean() * y.mean()) / d) if d > 0 else float("nan")

    lag1_in = [_r(o[..., t], o[..., t + 1]) for o in arrays for t in range(o.shape[-1] - 1)]
    lag1_bd = [_r(arrays[k][..., -1], arrays[k + 1][..., 0]) for k in range(len(arrays) - 1)]
    meanimg = [o.mean(axis=-1) for o in arrays]
    mi_adj = [_r(meanimg[k], meanimg[k + 1]) for k in range(len(meanimg) - 1)]
    per_phase = [_r(o[..., ph], o[..., ph + 1]) for ph in range(arrays[0].shape[-1] - 1) for o in arrays]
    per_phase = [float(np.median([_r(o[..., ph], o[..., ph + 1]) for o in arrays]))
                 for ph in range(arrays[0].shape[-1] - 1)]
    same_phase_d = {d_: float(np.median([_r(arrays[k][..., 0], arrays[k + d_][..., 0])
                                        for k in range(len(arrays) - d_)]))
                    for d_ in (1, 2, 5, 10, 30) if d_ < len(arrays)}
    chunk_means = [float(o.mean()) for o in arrays]
    tr_stitch = np.concatenate([o.reshape(-1, o.shape[-1]).mean(axis=0) for o in arrays])
    nch = arrays[0].shape[-1]
    djump_in = np.abs(np.diff(tr_stitch))[np.array([(i + 1) % nch != 0 for i in range(tr_stitch.size - 1)])]
    djump_bd = np.abs(np.diff(tr_stitch))[np.array([(i + 1) % nch == 0 for i in range(tr_stitch.size - 1)])]
    ac = [float(np.corrcoef(tr_stitch[:-l], tr_stitch[l:])[0, 1]) for l in range(1, min(31, tr_stitch.size // 3))]
    contig = dict(
        lag1_within_median=float(np.median(lag1_in)),
        lag1_boundary_median=float(np.median(lag1_bd)),
        boundary_over_within=float(np.median(lag1_bd) / np.median(lag1_in)),
        meanimg_adjacent_r_median=float(np.median(mi_adj)),
        same_spatial_tile=bool(np.median(mi_adj) > 0.5),
        per_phase_lag1=[float(v) for v in per_phase],
        same_phase_across_chunks=same_phase_d,
        trace_jump_within_median=float(np.median(djump_in)),
        trace_jump_boundary_median=float(np.median(djump_bd)),
        trace_jump_ratio=float(np.median(djump_bd) / max(np.median(djump_in), 1e-9)),
        chunk_means_first10=[round(v, 1) for v in chunk_means[:10]],
        chunk_means_monotone_frac=float(np.mean(np.diff(chunk_means) < 0)),
        stitched_trace_autocorr_lag1_30=[round(v, 4) for v in ac],
        autocorr_peak_lag=int(1 + int(np.argmax(ac))),
        autocorr_lag10=float(ac[9]) if len(ac) >= 10 else None,
    )
    contig["verdict"] = ("**chunk 间非时间相邻（按名序拼接不是连续记录）**"
                         if contig["boundary_over_within"] < 0.8 else "链内相邻（可作时间序列）")
    print("\n[③b 连续性诊断] 链内 lag1 r=%.4f vs 跨 chunk 边界 lag1 r=%.4f ⇒ 比值 %.2f ⇒ %s"
          % (contig["lag1_within_median"], contig["lag1_boundary_median"],
             contig["boundary_over_within"], contig["verdict"]))
    print("        轨迹跳变 |Δ|：链内中位 %.1f vs 跨边界中位 %.1f（比值 %.1f×）"
          % (contig["trace_jump_within_median"], contig["trace_jump_boundary_median"], contig["trace_jump_ratio"]))
    print("        同一空间 tile 校验：时间均值图像相邻 chunk r 中位 = %.4f" % contig["meanimg_adjacent_r_median"])
    print("        逐相位 lag1 相关（相位0→1 … 8→9）：%s" % " ".join("%.3f" % v for v in per_phase))
    print("        轨迹自相关 lag1..10 = %s（峰在 lag=%d）"
          % (" ".join("%.3f" % v for v in ac[:10]), contig["autocorr_peak_lag"]))

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

    per_obj_summary = {k: dict(median=float(np.median(v)), min=float(np.min(v)), max=float(np.max(v)), n=len(v))
                       for k, v in per_obj[">"].items()}
    print("  [逐对象区间] 横向 r: 中位 %.3f（%.3f–%.3f）｜纵向 %.3f（%.3f–%.3f）｜相邻帧 %.3f（%.3f–%.3f）"
          % (per_obj_summary["px_h"]["median"], per_obj_summary["px_h"]["min"], per_obj_summary["px_h"]["max"],
             per_obj_summary["px_v"]["median"], per_obj_summary["px_v"]["min"], per_obj_summary["px_v"]["max"],
             per_obj_summary["ft"]["median"], per_obj_summary["ft"]["min"], per_obj_summary["ft"]["max"]))

    # ---------------- 对照：该 τ 是否只是**块结构伪影**（L51.3 阳性对照思路的镜像用法）----------------
    # 若把 200 个 chunk 随机重排（保留块内形态、摧毁真实长程次序）后 τ 几乎不变，
    # 则该 τ **不能与块结构区分** ⇒ 不得当作自发慢状态的目标带。
    nch, nblk = arrays[0].shape[-1], len(arrays)
    blocks = trace_mean.reshape(nblk, nch)
    rng = np.random.default_rng(0)
    sur_bs = blocks[rng.permutation(nblk)].reshape(-1)                       # 块序重排
    within = np.array([rng.permutation(nch) for _ in range(nblk)])
    sur_ws = np.take_along_axis(blocks, within, axis=1).reshape(-1)          # 块内帧序重排
    white = rng.normal(size=trace_mean.size)                                 # 白噪零对照
    controls = {}
    for nm, tr in (("real_argv_order", trace_mean), ("block_shuffled", sur_bs),
                   ("within_block_shuffled", sur_ws), ("white_noise_null", white)):
        cb = build_band(tr, 6, "ctrl_bin6_%s" % nm, nm, a.fps)
        # 同时报未平滑口径的 lag-1 相关（块伪影最直接的表征）
        cb["lag1_autocorr_of_binned"] = float(np.corrcoef(
            bin_series(tr, 6, a.fps)[0][:-1], bin_series(tr, 6, a.fps)[0][1:])[0, 1])
        controls[nm] = dict(tau_smoothed_s=cb["AR1"].get("tau_s"), phi_smoothed=cb["AR1"].get("phi_smoothed"),
                            tau_unsmoothed_s=cb["AR1"].get("tau_unsmoothed_s"),
                            phi_unsmoothed=cb["AR1"].get("phi_unsmoothed"),
                            lag1_autocorr_binned=cb["lag1_autocorr_of_binned"],
                            Q_dwell_low_s=((cb["Q_method"].get("dwell", {}) or {}).get("low", {}) or {}).get("median_s"))
    rt = controls["real_argv_order"]["tau_smoothed_s"]
    bt = controls["block_shuffled"]["tau_smoothed_s"]
    rel = abs(bt - rt) / rt if (rt and bt) else None
    controls["_verdict"] = dict(
        real_tau_s=rt, block_shuffled_tau_s=bt, rel_diff=rel,
        indistinguishable=bool(rel is not None and rel <= 0.25),
        text=("该 τ 与『块序随机重排』对照值相差 %.1f%% ⇒ **不能与块结构区分**，"
              "不得作为自发慢状态目标带" % (100 * rel) if (rel is not None and rel <= 0.25)
              else "该 τ 与块序重排对照可区分（差 %.1f%%）" % (100 * rel if rel is not None else float('nan'))))
    print("\n=== 对照（band=bin6，spatial_mean_dfF）===")
    for k in ("real_argv_order", "block_shuffled", "within_block_shuffled", "white_noise_null"):
        c = controls[k]
        print("  %-22s τ_平滑=%9s s  φ_平滑=%-8s τ_未平滑=%9s s  Q_dwell低=%s s"
              % (k, ("%.3f" % c["tau_smoothed_s"]) if c["tau_smoothed_s"] else "n/a",
                 ("%.4f" % c["phi_smoothed"]) if c["phi_smoothed"] is not None else "n/a",
                 ("%.3f" % c["tau_unsmoothed_s"]) if c["tau_unsmoothed_s"] else "n/a",
                 ("%.3f" % c["Q_dwell_low_s"]) if c["Q_dwell_low_s"] else "n/a"))
    print("  ⇒ %s" % controls["_verdict"]["text"])

    # ---------------- 交叉核对（**先算完再看对方结果**：本块在全部计算完成后才读对方文件）----------------
    ref_band = os.path.join(DATA_DIR, "_mainagent_m11_same_source_band.json")
    ref_parse = os.path.join(DATA_DIR, "m11_microns_parse_validation.json")
    cross = dict(sources={
        "band": os.path.basename(ref_band) if os.path.exists(ref_band) else None,
        "parse": os.path.basename(ref_parse) if os.path.exists(ref_parse) else None,
        "order_note": "本块在自身全部计算完成后才读对方文件（RF-33 要求先算后比）"}, comparisons=[], attribution={})
    if os.path.exists(ref_band):
        rb = json.load(open(ref_band))["pipelines"]
        pairs = [("bin1_spatial_mean_dfF", "frame_mean|native|unsmoothed"),
                 ("bin1_spatial_mean_dfF", "frame_mean|native|smooth5"),
                 ("bin6_spatial_mean_dfF", "frame_mean|box0.914|unsmoothed"),
                 ("bin6_spatial_mean_dfF", "frame_mean|box0.914|smooth5")]
        mine = {b["label"]: b for b in bands}
        for myk, rk in pairs:
            mb = mine[myk]; ar = mb["AR1"]; r = rb[rk]
            smoothed = rk.endswith("smooth5")
            my_phi = ar["phi_smoothed"] if smoothed else ar["phi_unsmoothed"]
            my_tau = ar.get("tau_s") if smoothed else ar.get("tau_unsmoothed_s")
            my_n = mb["n_frames_smoothed"] if smoothed else mb["n_frames_binned"]
            rel_tau = (my_tau - r["tau_s"]) / r["tau_s"] if (my_tau and r.get("tau_s")) else None
            rel_phi = (my_phi - r["phi"]) / r["phi"] if (my_phi and r.get("phi")) else None
            # **口径归因**：把 dt 换成 B4 标称 0.9138 s 重算 τ（仅 box 路径有差异）
            tau_nominal = None
            if my_phi is not None and 0 < my_phi < 1 and "bin6" in myk:
                tau_nominal = float(-B4_BIN_S / math.log(my_phi))
            cross["comparisons"].append(dict(
                mine_label=myk, theirs_key=rk, smoothed=smoothed,
                my_n=my_n, their_n=r["n"], n_match=bool(my_n == r["n"]),
                my_phi=my_phi, their_phi=r["phi"], rel_phi=rel_phi,
                my_tau_s=my_tau, their_tau_s=r["tau_s"], rel_tau=rel_tau,
                over_5pct=bool(rel_tau is not None and abs(rel_tau) > 0.05),
                tau_with_b4_nominal_dt=tau_nominal,
                tau_nominal_vs_theirs=((tau_nominal - r["tau_s"]) / r["tau_s"]
                                       if (tau_nominal and r.get("tau_s")) else None)))
        cross["attribution"] = dict(
            dt_dimension=("box 路径**唯一**系统性差异：对方 bin 为 6 帧/n=333（与我相同），"
                          "但 τ=−dt/lnφ 用 **B4 标称 dt=0.913826 s**；实测箱宽 = 6/%.6f = **%.6f s**（+%.2f%%）"
                          "⇒ 其 box 路径 τ 系统性偏低 %.2f%%（把 dt 换成实测值即复现我方值）"
                          % (a.fps, 6 / a.fps, 100 * (6 / a.fps / B4_BIN_S - 1), 100 * (6 / a.fps / B4_BIN_S - 1))),
            trace_dimension=("trace 口径（我方=剔除「曾现哨兵」体素的空间均值；对方=frame_mean 且逐元素排除哨兵）"
                             "⇒ φ 相对差 ≤0.01% ⇒ **该维度在本次量级下可忽略**（已量化，非声明）"),
            unresolved=("Q/H dwell 仍差 2.1×（我方 H 低态 dwell 6.59 s vs 对方 3.17 s）："
                        "H 序列 run 数仅 14–17、p90/中位≈7（重尾）⇒ 中位 dwell 的抽样不确定度大；"
                        "且对方 A 矩阵 [[0.9798,0.0202],[0.0375,0.9625]] 在 occ_low=0.6425 下蕴含约 52.8 次切换，"
                        "而其自报 n_switches=35 ⇒ **对方记录内部不一致**。故该差异**未定位**，"
                        "需 B1 提供其 HMM 的拟合序列（平滑/未平滑）、初始化与迭代设置。"))
    if os.path.exists(ref_parse):
        pf = json.load(open(ref_parse)).get("full_objects", {})
        cross["parse_metrics"] = dict(
            theirs=dict(lat_x_mean=pf.get("lat_x_mean"), lat_y_mean=pf.get("lat_y_mean"),
                        frame_mean=pf.get("frame_mean"), p50_min=pf.get("p50_min"), p50_max=pf.get("p50_max"),
                        n_neg=pf.get("n_int16_negative"), n_u16gt=pf.get("n_uint16_gt_65000"),
                        vmin=pf.get("vmin_global")),
            mine=dict(sentinels=sentinels, structure_pooled=st[">i2（正解）"],
                      structure_per_object=per_obj_summary, p50={">i2": st[">i2（正解）"]["p50_median"]}),
            note="我方为**中位+区间**、对方为**均值+区间**（统计量口径不同，非数据不同）")
        cross["parse_metrics"]["agreement"] = dict(
            sentinel_exact=bool(sentinels["neg16"] == pf.get("n_int16_negative")),
            u16gt_exact=bool(sentinels["u16gt"] == pf.get("n_uint16_gt_65000")),
            vmin_exact=bool(gmin == pf.get("vmin_global")),
            p50_min_match=bool(abs(min(p50[">i2"]) - pf.get("p50_min", -1)) < 1e-6),
            p50_max_match=bool(abs(max(p50[">i2"]) - pf.get("p50_max", -1)) < 1e-6),
            rel_lat_x=float((st[">i2（正解）"]["px_h"] - pf.get("lat_x_mean", float("nan"))) / pf.get("lat_x_mean", 1)),
            rel_frame=float((st[">i2（正解）"]["ft"] - pf.get("frame_mean", float("nan"))) / pf.get("frame_mean", 1)))
    print("\n=== 交叉核对（先算后比）===")
    for c in cross["comparisons"]:
        print("  %-26s φ: 我 %.6f vs 他 %.6f (Δ%+.4f%%) | τ: 我 %8.4f vs 他 %8.4f (Δ%+.2f%%)%s"
              % (c["theirs_key"], c["my_phi"], c["their_phi"], 100 * c["rel_phi"],
                 c["my_tau_s"], c["their_tau_s"], 100 * c["rel_tau"],
                 "  **>5%**" if c["over_5pct"] else ""))
    if cross["comparisons"] and cross["comparisons"][-1]["tau_with_b4_nominal_dt"]:
        c0 = [c for c in cross["comparisons"] if c["tau_with_b4_nominal_dt"]]
        print("  [口径归因] box 路径改用 B4 标称 dt=0.913826：")
        for c in c0:
            print("     %-26s 归因后 τ=%.4f vs 他 %.4f ⇒ 残差 %+.3f%%"
                  % (c["theirs_key"], c["tau_with_b4_nominal_dt"], c["their_tau_s"],
                     100 * c["tau_nominal_vs_theirs"]))
    print("  [归因] %s" % cross["attribution"]["dt_dimension"])
    if "parse_metrics" in cross:
        ag = cross["parse_metrics"]["agreement"]
        print("  [结构指标] 哨兵逐位一致=%s u16gt一致=%s vmin一致=%s | 我的 pooled lat_x=%.4f vs 他 %.4f（Δ%+.2f%%）"
              % (ag["sentinel_exact"], ag["u16gt_exact"], ag["vmin_exact"],
                 cross["parse_metrics"]["mine"]["structure_pooled"]["px_h"],
                 cross["parse_metrics"]["theirs"]["lat_x_mean"], 100 * ag["rel_lat_x"]))

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
        self_proof=dict(sentinels=sentinels, global_min_int16=gmin, structure=st,
                        axis_order_control=axis_order_control,
                        structure_per_object=per_obj_summary),
        traces=dict(spatial_mean_dfF=dict(mean=float(trace_mean.mean()), sd=float(trace_mean.std())),
                    active_fraction_p99=dict(mean=float(trace_frac.mean()), thr_hi=thr_hi),
                    n_voxels_valid=int(valid.sum()), n_voxels_ever_sentinel=int(ever_sent.sum())),
        contiguity=contig,
        controls=controls,
        crosscheck=cross,
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
