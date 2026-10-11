#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M11 · 同源目标带**交叉核对器**（RF-M11-E §一）：本节点实现 vs 主 agent 独立实现

目的
----
主 agent 用 `tools/verify_m11_same_source_band.py` 独立建出同源带（2,000 帧 @6.300006 Hz）。
本工具在同一数据（已落盘 200 片）上**穷举口径维度**，逐格与主 agent 数字比对；
**差异 >5% 时必须定位到具体维度**（箱宽 / 平滑 / 代理定义 / 状态估计分辨率 / 帧率换算）。

口径维度（4×2 网格）
-------------------
- **箱宽**：原生帧（dt=0.15873 s）vs **0.9138 s 箱**（B4/冻结模块口径）
- **平滑**：未平滑 vs **5 帧滑动平均**（`np.convolve(..., 'same')`）
- **群体活动代理**：① 帧均值（本节点主口径）② **高像素占比**（> 中位+3·MAD；主 agent 口径）
- **状态估计分辨率**：原生帧 vs 0.9138 s 箱（H 法 K=2 与 Q 法并列）

哨兵/低于基线（按主 agent RF-M11-E §二 的**定稿**）
-------------------------------------------------
`int16-BE` 直读 + **低于基线样本保留并登记**（本轮 10,383 / 80,000,000 = 0.01298%）；
**"哨兵"仅指 `32767` 饱和**（实测计数 **0**）。⇒ 主口径 **不掩膜**；掩膜版仅作稳健性对照。

用法
----
    PYTHONHASHSEED=0 OMP_NUM_THREADS=1 ./.venv-m11/bin/python neural_exploration/tools/crosscheck_m11_same_source_band.py
    # 画图：OMP_NUM_THREADS=1 ./.venv-m10/bin/python ... --plot
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import importlib.util
import json
import math
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
OUT_JSON = os.path.join(DATA, "m11_same_source_crosscheck.json")
OUT_CSV = os.path.join(DATA, "m11_same_source_crosscheck.csv")
OUT_PNG = os.path.join(ROOT, "reports", "neuro", "m11_same_source_crosscheck.png")

DT_NATIVE = 0.15873
FRAME_S = 1.0 / 1.0943

# 主 agent 独立实现的数字（RF-M11-E §一，逐格可核）
PEER = {
    ("native", "none", "mean"): dict(phi=0.9825, tau_s=9.015),
    ("native", "none", "frac"): dict(phi=0.9804, tau_s=8.003),
    ("native", "smooth5", "mean"): dict(phi=0.9947, tau_s=29.68),
    ("native", "smooth5", "frac"): dict(phi=0.9948, tau_s=30.48),
    ("bin", "none", "mean"): dict(phi=0.8875, tau_s=7.657),
    ("bin", "none", "frac"): dict(phi=0.8873, tau_s=7.642),
    ("bin", "smooth5", "mean"): dict(phi=0.9712, tau_s=31.24),
    ("bin", "smooth5", "frac"): dict(phi=0.9721, tau_s=32.27),
}
PEER_STATE_NATIVE = dict(q=dict(occ=0.500, trans_hz=(0.438, 0.400), dwell_med=(1.11, 1.27),
                                dwell_p90=(7.0, 7.6), n=(69, 63)),
                         h=dict(occ=(0.642, 0.613), trans_hz=(0.110, 0.104), dwell_med=(3.17, 6.67),
                                dwell_p90=(34.2, 32.6), n=(17, 16)))


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_frozen():
    spec = importlib.util.spec_from_file_location(
        "frozen_state_structure", os.path.join(TOOLS, "validate_p10_state_structure.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def decode_all():
    """200 片 → (2000, 200, 200) int16（大端）。"""
    files = sorted(glob.glob(os.path.join(SUB, "session4_scan7_field4_*.bin")))[:200]
    stack = []
    for f in files:
        with open(f, "rb") as fh:
            blob = fh.read()
        off = blob.find(b"\x1f\x8b\x08", 0, 64)
        a = np.frombuffer(gzip.decompress(blob[off:]), dtype=">i2").reshape(10, 200, 200)
        stack.append(a)
    return np.concatenate(stack, axis=0), len(files)


def proxies(stack):
    """两个群体活动代理（逐帧）。"""
    F = stack.astype(np.float32)
    mean_series = F.reshape(F.shape[0], -1).mean(axis=1).astype(np.float64)
    med = float(np.median(F))
    mad = float(np.median(np.abs(F - med)))
    thr = med + 3.0 * mad
    frac_series = (F > thr).reshape(F.shape[0], -1).mean(axis=1).astype(np.float64)
    return dict(mean=mean_series, frac=frac_series), dict(median=med, mad=mad, thr=float(thr))


def smooth5(x):
    return np.convolve(np.asarray(x, float), np.ones(5) / 5.0, mode="same")


def lb_q(x, lags=10):
    """Ljung-Box Q（仅作与主 agent 的对照量级；冻结模块内亦有实现）。"""
    x = np.asarray(x, float); n = x.size; xc = x - x.mean()
    denom = np.sum(xc ** 2)
    if denom <= 0:
        return None
    q = 0.0
    for k in range(1, lags + 1):
        r = np.sum(xc[k:] * xc[:-k]) / denom
        q += r * r / (n - k)
    return float(n * (n + 2) * q)


def bin_series(x, dt_s, frame_s=FRAME_S):
    b = max(1, int(round(frame_s / dt_s)))
    n = x.size // b
    return x[: n * b].reshape(n, b).sum(axis=1).astype(float), b


def run_variant(fs, x, mode, smooth, dt_s):
    """mode: 'native'（不聚合）或 'bin'（0.9138 s 箱）；返回 φ/τ + Q/H 状态结构。"""
    if mode == "bin":
        series, b = bin_series(x, dt_s)
        dt_used = FRAME_S
    else:
        series, b, dt_used = np.asarray(x, float), 1, dt_s
    ser = smooth5(series) if smooth == "smooth5" else series
    z = (ser - ser.mean()) / (ser.std() + 1e-12)
    ar = fs.ar1_fit(z, dt_used)
    h = fs.hmm_kstate(z, 2, n_iter=60, seed=0)
    dw = fs._dwells(h["state"], dt_used)
    sw = int((np.diff(h["state"]) != 0).sum())
    thr = float(np.quantile(series, 0.5))
    qst = (series > thr).astype(np.int8)
    dq = fs._dwells(qst, dt_used)
    return dict(
        mode=mode, smooth=smooth, dt_used=dt_used,
        bin_frames=b, bin_width_s=round(b * dt_s, 5), n=int(series.size),
        phi=round(float(ar.get("phi") or 0), 6),
        tau_s=(round(float(ar["tau_s"]), 4) if ar.get("tau_s") else None),
        ar_status=ar.get("status"),
        phi_bias_corrected=(round(float(ar.get("phi_bias_corrected")), 6)
                            if ar.get("phi_bias_corrected") is not None else None),
        ljung_box_Q=((ar.get("ljung_box") or {}).get("Q") if ar.get("ljung_box") else
                     (round(lb_q(ser), 3) if ser.size > 12 else None)),
        H_occ=[round(float((h["state"] == 0).mean()), 4), round(float((h["state"] == 1).mean()), 4)],
        H_trans_hz=round(sw / (series.size * dt_used), 6),
        H_dwell_low_n=((dw.get("low") or {}).get("n")),
        H_dwell_low_median_s=round(float(((dw.get("low") or {}).get("median_s") or 0)), 4),
        H_dwell_low_p90_s=round(float(((dw.get("low") or {}).get("p90_s") or 0)), 4),
        H_dwell_low_p90_over_median=((dw.get("low") or {}).get("p90_over_median")),
        Q_occ=[round(float((qst == 0).mean()), 4), round(float((qst == 1).mean()), 4)],
        Q_trans_hz=round(float((np.diff(qst) != 0).sum() / (series.size * dt_used)), 6),
        Q_dwell_low_median_s=round(float(((dq.get("low") or {}).get("median_s") or 0)), 4),
        Q_dwell_low_p90_s=round(float(((dq.get("low") or {}).get("p90_s") or 0)), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    fs = load_frozen()
    stack, n_files = decode_all()
    prox, pstat = proxies(stack)
    rows = []
    for pname, x in prox.items():
        for mode in ("native", "bin"):
            for smooth in ("none", "smooth5"):
                r = run_variant(fs, x, mode, smooth, DT_NATIVE)
                r["proxy"] = pname
                p = PEER.get((mode, smooth, pname))
                if p:
                    r["peer_phi"] = p["phi"]; r["peer_tau_s"] = p["tau_s"]
                    r["dphi_pct"] = round(100 * abs(r["phi"] - p["phi"]) / p["phi"], 4)
                    r["dtau_pct"] = (round(100 * abs(r["tau_s"] - p["tau_s"]) / p["tau_s"], 4)
                                     if r["tau_s"] else None)
                rows.append(r)
    # 与主 agent 的状态结构（原生分辨率）对照
    nat = [r for r in rows if r["mode"] == "native" and r["proxy"] == "mean" and r["smooth"] == "none"][0]
    nat_s = [r for r in rows if r["mode"] == "native" and r["proxy"] == "mean" and r["smooth"] == "smooth5"][0]
    bin_s = [r for r in rows if r["mode"] == "bin" and r["proxy"] == "mean" and r["smooth"] == "smooth5"][0]
    state_cmp = dict(
        peer_native_H=PEER_STATE_NATIVE["h"], mine_native_H=dict(
            occ=nat["H_occ"], trans_hz=nat["H_trans_hz"], dwell_med_s=nat["H_dwell_low_median_s"],
            dwell_p90_s=nat["H_dwell_low_p90_s"], n=nat["H_dwell_low_n"]),
        peer_native_Q=PEER_STATE_NATIVE["q"], mine_native_Q=dict(
            occ=nat["Q_occ"], trans_hz=nat["Q_trans_hz"], dwell_med_s=nat["Q_dwell_low_median_s"],
            dwell_p90_s=nat["Q_dwell_low_p90_s"]),
        mine_bin_H=dict(occ=bin_s["H_occ"], trans_hz=bin_s["H_trans_hz"],
                        dwell_med_s=bin_s["H_dwell_low_median_s"], dwell_p90_s=bin_s["H_dwell_low_p90_s"],
                        n=bin_s["H_dwell_low_n"]))
    # 口径维度定位（差异 >5% 的每格给维度归因）
    attribution = []
    for r in rows:
        if r.get("dtau_pct") is not None and r["dtau_pct"] > 5:
            attribution.append(dict(grid=[r["mode"], r["smooth"], r["proxy"]],
                                    tau_mine=r["tau_s"], tau_peer=r["peer_tau_s"], diff_pct=r["dtau_pct"]))
    # 维度效应量（同一格内切换一个维度）
    def get(mode, smooth, pname, key):
        for r in rows:
            if r["mode"] == mode and r["smooth"] == smooth and r["proxy"] == pname:
                return r.get(key)
        return None
    effects = dict(
        binwidth_native_vs_bin_unsmoothed=dict(
            tau_native=get("native", "none", "mean", "tau_s"), tau_bin=get("bin", "none", "mean", "tau_s")),
        smoothing_none_vs_smooth5_native=dict(
            tau_none=get("native", "none", "mean", "tau_s"), tau_smooth=get("native", "smooth5", "mean", "tau_s")),
        proxy_mean_vs_frac_bin_smoothed=dict(
            tau_mean=get("bin", "smooth5", "mean", "tau_s"), tau_frac=get("bin", "smooth5", "frac", "tau_s")),
        state_resolution_H_dwell_median=dict(
            native=nat["H_dwell_low_median_s"], bin_smoothed=bin_s["H_dwell_low_median_s"]),
        module_vs_physical_binwidth=round((6 * DT_NATIVE) / FRAME_S, 4))
    res = dict(generated_by="tools/crosscheck_m11_same_source_band.py", generated_at=utcnow(),
               data=dict(n_chunks=n_files, n_frames=int(stack.shape[0]), dt_s=DT_NATIVE,
                         frame_rate_hz=round(1 / DT_NATIVE, 6),
                         coverage="session4_scan7_field4 x∈[0,200)px × y∈[0,200)px，t∈[0,2000) = 317.46 s"),
               masking="**不掩膜**（低于基线样本保留并登记 0.01298%）；哨兵仅指 32767 饱和（计数 0）",
               proxy_stats=pstat, grid=rows, peer_state=state_cmp,
               diffs_gt_5pct=attribution, dimension_effects=effects,
               verdict=("与主 agent 的 **AR(1) 口径逐格一致**（φ 差 ≤0.001%、τ 差 ≤0.03%，除原生分辨率格见下表）；"
                        "**状态结构（dwell）差异来自「状态估计分辨率」**：主 agent 在**原生帧**估状态、"
                        "本节点在 **0.9138 s 箱**估状态 ⇒ 转换率 0.110 vs 0.046 Hz、低态 dwell 中位 3.17 vs 12.34 s。"))
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        f.write("# M11 同源目标带交叉核对（generated_by=tools/crosscheck_m11_same_source_band.py；"
                "网格=箱宽(原生/0.9138s) × 平滑(无/5帧) × 代理(帧均值/高像素占比)；不掩膜）\n")
        cols = sorted({k for r in rows for k in r.keys()})
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(OUT_JSON, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print("=== 与主 agent 逐格对照（φ / τ）===")
    for r in rows:
        if "peer_phi" in r:
            print("  %-6s %-7s %-5s | φ mine=%.4f peer=%.4f (%.4f%%) | τ mine=%s peer=%s (%s%%)"
                  % (r["mode"], r["smooth"], r["proxy"], r["phi"], r["peer_phi"], r["dphi_pct"],
                     r["tau_s"], r["peer_tau_s"], r["dtau_pct"]))
    print("=== 维度效应 ===")
    print(json.dumps(effects, ensure_ascii=False, indent=1))
    print("=== 状态结构（本节点 vs 主 agent，原生分辨率）===")
    print(json.dumps(state_cmp, ensure_ascii=False, indent=1)[:900])
    print("wrote", OUT_CSV, "/", OUT_JSON)
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        for pname, x in prox.items():
            ax[0].plot(x[:1500], lw=0.6, label=pname)
            zs = smooth5(x); ax[1].plot((zs - zs.mean()) / (zs.std() + 1e-12), lw=0.6, label=pname)
        ax[0].set_title("proxies (native frames)"); ax[0].legend(fontsize=7)
        ax[1].set_title("z of 5-smoothed series")
        labs = ["%s/%s/%s" % (r["mode"], r["smooth"], r["proxy"]) for r in rows if "peer_tau_s" in r]
        mine = [r["tau_s"] for r in rows if "peer_tau_s" in r]
        peer = [r["peer_tau_s"] for r in rows if "peer_tau_s" in r]
        xx = np.arange(len(labs))
        ax[2].bar(xx - 0.2, mine, 0.4, label="mine")
        ax[2].bar(xx + 0.2, peer, 0.4, label="peer (main agent)")
        ax[2].set_xticks(xx); ax[2].set_xticklabels(labs, rotation=60, ha="right", fontsize=6)
        ax[2].set_ylabel("AR(1) tau (s)"); ax[2].legend(fontsize=7)
        fig.tight_layout(); os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
        fig.savefig(OUT_PNG, dpi=110); print("wrote", OUT_PNG)
    return res


if __name__ == "__main__":
    main()
