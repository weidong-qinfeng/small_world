"""主agent 独立探针：从**已解码的 MICrONS 同源成像**自建 S 门目标带（R9 反转后的关键路径）。

**背景（R9）**：MICrONS `functional_series` 分片此前被本 agent 误判为"不可解码"（测试集漏了字节序维度）；
节点 M11-B1 的系统化解析验证给出正确解：**20 B 大端头 `(4,200,200,1,10)` + gzip@offset20 + 大端 int16（`>i2`）**，
本 agent 已独立复算确认（横向相关 0.65–0.80、帧间相关 0.69–0.80；而 `<i2` 全 ≈0）。

**本探针做什么**：用**已在盘上的 200 个 chunk**（200×10 = **2,000 帧 @6.300006 Hz ≈ 317 s**），
**独立**构造群体活动序列并给出慢结构目标带：
  1. 哨兵掩膜：`uint16 读法 > 65000 ⇔ int16 < 0` ⇒ 计为缺失（M11-B1 报 10,383/80,000,000 = 0.01298%，本探针独立复算）；
  2. 群体活动代理（双口径并列）：**(a) 帧均值**（全视野钙强度均值）；**(b) 高像素占比**（阈值取全局中位 + 3·MAD）；
  3. **AR(1) φ/τ**：**原生帧率（0.15873 s 帧间隔）** 与 **降采样到 0.9138 s 箱（与 B4 同箱宽）** 并列；
     每档再给 **未平滑** 与 **5 帧平滑** 两个口径；
  4. **状态结构**：**分位法 Q** 与 **2 状态高斯 HMM** 的占用/转换率/低态 dwell 中位（原生帧分辨率）；
  5. **样本量门**：n_frames ≥300（τ/dwell）、≥130（形状）——本数据 2,000 帧 ⇒ 满足；
  6. **内部一致性断言**：`|dwell_low_med − 1/(2·转换率)| / dwell_low_med ≤ 1.0`（不满足即标 invalid）。

只读输入；产物写 `data/_mainagent_m11_same_source_band.{json,csv}`。

用法：
    PYTHONHASHSEED=0 OMP_NUM_THREADS=1 ./.venv-m9/bin/python \
        -m neural_exploration.tools.verify_m11_same_source_band
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import math
import os
import struct
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SUBSET_DIR = os.path.join(DATA_DIR, "m11_raw", "microns", "functional_subset")
OUT_JSON = os.path.join(DATA_DIR, "_mainagent_m11_same_source_band.json")
OUT_CSV = os.path.join(DATA_DIR, "_mainagent_m11_same_source_band.csv")

FRAME_HZ = 6.300006
FRAME_S = 1.0 / FRAME_HZ            # 0.15873 s
BOX_S = 0.9138261902586128          # B4/ZAPBench 箱宽（用于降采样对照）
SMOOTH = 5


def load_frames(limit_chunks: int | None = None):
    """流式读出全部帧：返回 (帧均值, 高像素占比, 全局统计, 缺失计数)。"""
    fs = sorted(glob.glob(os.path.join(SUBSET_DIR, "session4_scan7_field4_*.bin")))
    if limit_chunks:
        fs = fs[:limit_chunks]
    means, hpix = [], []
    n_tot = n_masked = n_sent = 0
    sum_all = sum_sq = 0.0
    # pass 1：全局均值/方差（用于 MAD 阈值）
    vals_sample = []
    for f in fs:
        b = open(f, "rb").read()
        hdr = struct.unpack(">5I", b[:20])
        if hdr != (4, 200, 200, 1, 10):
            raise ValueError(f"{os.path.basename(f)} header 非预期: {hdr}")
        raw = gzip.decompress(b[20:])
        a = np.frombuffer(raw, dtype=">i2")
        n_tot += a.size
        n_sent += int((a < 0).sum())          # 哨兵：int16<0 ⇔ uint16>65000
        v = a[a >= 0].astype(np.float32)
        n_masked += int(a.size - v.size)
        sum_all += float(v.sum())
        sum_sq += float((v.astype(np.float64) ** 2).sum())
        if len(vals_sample) < 4:
            vals_sample.append(v[::997].copy())
    n_valid = n_tot - n_masked
    g_mean = sum_all / n_valid
    g_std = math.sqrt(max(sum_sq / n_valid - g_mean ** 2, 0.0))
    # 以全局中位近似（用采样分位）作为基线：MAD 门
    samp = np.concatenate(vals_sample)
    med = float(np.median(samp))
    mad = float(np.median(np.abs(samp - med)))
    thr = med + 3.0 * mad
    # pass 2：逐帧均值 + 高像素占比
    for f in fs:
        b = open(f, "rb").read()
        raw = gzip.decompress(b[20:])
        a = np.frombuffer(raw, dtype=">i2").reshape(10, 200, 200)
        ok = a >= 0
        af = np.where(ok, a, np.nan).astype(np.float32)
        means.append(np.nanmean(af, axis=(1, 2)).astype(np.float64))
        with np.errstate(invalid="ignore"):
            hpix.append((np.nansum(np.where(ok, a, 0) > thr, axis=(1, 2)) /
                         np.maximum(ok.sum(axis=(1, 2)), 1)).astype(np.float64))
    return (np.concatenate(means), np.concatenate(hpix),
            dict(n_files=len(fs), n_elements=n_tot, n_sentinel=n_sent,
                 sentinel_frac=n_sent / n_tot, global_mean=g_mean, global_std=g_std,
                 baseline_median=med, mad=mad, threshold_med_plus_3mad=thr),
            dict(n_total=n_tot, n_valid=n_valid))


def ar1(z: np.ndarray, dt_s: float):
    x = np.asarray(z, float)
    if x.size < 20:
        return dict(status="invalid", reason=f"n={x.size}")
    x0, x1 = x[:-1], x[1:]
    phi = float((x0 @ x1) / (x0 @ x0)) if (x0 @ x0) > 0 else float("nan")
    tau = -dt_s / math.log(phi) if 0 < phi < 1 else float("nan")
    r = x1 - phi * x0
    rho = [float((r[:-k] @ r[k:]) / (r @ r)) for k in range(1, 11)]
    Q = len(r) * (len(r) + 2) * sum(rho[k - 1] ** 2 / (len(r) - k) for k in range(1, 11))
    return dict(status="measured", n=int(x.size), phi=phi, tau_s=tau, ljung_box_Q=Q,
                resid_lag1=rho[0], resid_whiten=bool(rho[0] ** 2 < 3.0 / r.size))


def zscore(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-12)


def box_aggregate(x: np.ndarray, box_s: float, dt_s: float):
    b = int(round(box_s / dt_s))
    nb = x.size // b
    return x[: nb * b].reshape(nb, b).mean(axis=1), b


def quantile_states(x: np.ndarray, dt_s: float):
    r = np.asarray(x, float)
    thr = float(np.quantile(r, 0.5))
    s = (r > thr).astype(np.int8)
    sw = int((np.diff(s) != 0).sum())
    dur = {}
    for v, nm in ((0, "low"), (1, "high")):
        bb = (s == v).astype(np.int8)
        d = np.diff(np.r_[0, bb, 0])
        st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        interior = (st > 0) & (en < s.size)
        ln = (en[interior] - st[interior]) * dt_s
        dur[nm] = dict(n=int(ln.size), median_s=float(np.median(ln)) if ln.size else 0.0,
                       p90_s=float(np.quantile(ln, 0.9)) if ln.size else 0.0)
    return dict(occ_low=float((s == 0).mean()), occ_high=float((s == 1).mean()),
                transition_rate_hz=float(sw / (s.size * dt_s)), n_switches=sw, dwell=dur)


def hmm2(x: np.ndarray, dt_s: float, n_iter: int = 60, seed: int = 0):
    """2 状态高斯 HMM（EM；与 M10/B4 同风格，仅用 numpy）。"""
    x = np.asarray(x, float).reshape(-1, 1)
    T = x.shape[0]
    q = np.quantile(x, [0.33, 0.67])
    mu = np.array([q[0], q[1]], float)
    sd = np.array([x.std(), x.std() * 1.5]) + 1e-6
    A = np.array([[0.99, 0.01], [0.01, 0.99]])
    pi = np.array([0.5, 0.5])

    def logemit(xx):
        return -0.5 * ((xx - mu) ** 2 / sd ** 2 + np.log(2 * np.pi * sd ** 2))

    for _ in range(n_iter):
        lb = logemit(x)
        logA = np.log(A + 1e-300)
        la = np.zeros((T, 2))
        la[0] = np.log(pi + 1e-300) + lb[0]
        for t in range(1, T):
            m = la[t - 1][:, None] + logA
            la[t] = np.logaddexp(m[0], m[1]) + lb[t]
        lbk = np.zeros((T, 2))
        for t in range(T - 2, -1, -1):
            m = logA + (lb[t + 1] + lbk[t + 1])[None, :]
            lbk[t] = np.logaddexp(m[:, 0], m[:, 1])
        lg = la + lbk
        g = np.exp(lg - np.logaddexp(lg[:, 0], lg[:, 1])[:, None])
        pi = g[0] / g[0].sum() if g[0].sum() else pi
        A = (g[:-1].T @ g[1:]) / np.maximum(g[:-1].sum(axis=0), 1e-12)[:, None]
        w = g.sum(axis=0)
        mu = (g * x).sum(axis=0) / np.maximum(w, 1e-12)
        var = (g * (x - mu) ** 2).sum(axis=0) / np.maximum(w, 1e-12)
        sd = np.sqrt(np.maximum(var, 1e-12))
    s = (g[:, 1] > g[:, 0]).astype(np.int8)
    sw = int((np.diff(s) != 0).sum())
    dur = {}
    for v, nm in ((0, "low"), (1, "high")):
        bb = (s == v).astype(np.int8)
        d = np.diff(np.r_[0, bb, 0])
        st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        interior = (st > 0) & (en < s.size)
        ln = (en[interior] - st[interior]) * dt_s
        dur[nm] = dict(n=int(ln.size), median_s=float(np.median(ln)) if ln.size else 0.0,
                       p90_s=float(np.quantile(ln, 0.9)) if ln.size else 0.0)
    return dict(mu=list(map(float, mu)), sd=list(map(float, sd)),
                A=A.tolist(), occ_low=float((s == 0).mean()), occ_high=float((s == 1).mean()),
                transition_rate_hz=float(sw / (s.size * dt_s)), n_switches=sw, dwell=dur,
                state=s)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-chunks", type=int, default=None)
    args = ap.parse_args(argv)
    t0 = time.time()
    fmean, frac, gstats, cnt = load_frames(args.limit_chunks)
    n_frames = fmean.size
    out = dict(probe="mainagent_m11_same_source_band", frame_hz=FRAME_HZ, frame_s=FRAME_S,
               n_frames=int(n_frames), duration_s=float(n_frames * FRAME_S),
               globals=gstats, counts=cnt,
               sample_gate=dict(n_frames=int(n_frames), tau_ok=bool(n_frames >= 300),
                                shape_ok=bool(n_frames >= 130), bic_ok=bool(n_frames >= 500)),
               pipelines={})

    for nm, series in (("frame_mean", fmean), ("high_pixel_frac", frac)):
        z = zscore(series)
        # A) 原生帧分辨率，未平滑 / 5 帧平滑
        sm = np.convolve(z, np.ones(SMOOTH) / SMOOTH, mode="valid")
        out["pipelines"][f"{nm}|native|unsmoothed"] = ar1(z, FRAME_S)
        out["pipelines"][f"{nm}|native|smooth5"] = ar1(sm, FRAME_S)
        # B) 降采样到 0.9138 s 箱（与 B4 同箱宽），未平滑 / 5 帧平滑
        agg, b = box_aggregate(series, BOX_S, FRAME_S)
        zb = zscore(agg)
        smb = np.convolve(zb, np.ones(SMOOTH) / SMOOTH, mode="valid")
        out["pipelines"][f"{nm}|box0.914|unsmoothed"] = ar1(zb, BOX_S)
        out["pipelines"][f"{nm}|box0.914|smooth5"] = ar1(smb, BOX_S)
        # C) 状态结构（原生帧分辨率）
        q = quantile_states(series, FRAME_S)
        h = hmm2(series, FRAME_S)
        cq = abs(q["dwell"]["low"]["median_s"] - 1.0 / (2 * q["transition_rate_hz"])) / max(
            q["dwell"]["low"]["median_s"], 1e-9)
        ch = abs(h["dwell"]["low"]["median_s"] - 1.0 / (2 * h["transition_rate_hz"])) / max(
            h["dwell"]["low"]["median_s"], 1e-9)
        out["pipelines"][f"{nm}|state_structure"] = dict(
            Q=q, H={k: v for k, v in h.items() if k != "state"},
            internal_consistency_Q_rel=float(cq), internal_consistency_H_rel=float(ch),
            consistency_pass_Q=bool(cq <= 1.0), consistency_pass_H=bool(ch <= 1.0))

    out["wall_s"] = round(time.time() - t0, 1)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False, default=float)
    rows = []
    for k, v in out["pipelines"].items():
        if "tau_s" in v:
            rows.append([k, v.get("status"), v.get("n"), v.get("phi"), v.get("tau_s"),
                         v.get("ljung_box_Q"), v.get("resid_whiten")])
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["pipeline", "status", "n", "phi", "tau_s", "ljung_box_Q", "resid_whiten"])
        for r in rows:
            w.writerow(r)
    print("MAINAGENT_M11_BAND " + json.dumps(
        {k: out[k] for k in ("n_frames", "duration_s", "sample_gate", "wall_s")},
        ensure_ascii=False), flush=True)
    for k, v in out["pipelines"].items():
        if "tau_s" in v:
            print(f"  {k:<34} phi={v['phi']:+.4f} tau={v['tau_s']:8.3f}s n={v['n']} Q={v['ljung_box_Q']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
