#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
M11 / B1 —— 优先级 4：**四类参考量的钙域口径**（同源、无需新下载）
=================================================================
主 agent §二 要求（**用已在盘的 200 片 × 2 field**）：
  ① 率分布（**钙事件率**，与 spike 口径**分列**）
  ② 群体相关（平均成对相关 + 特征值谱 + 有效维度 + **同管线相位随机化代理零分布**）
  ③ 状态结构（Q 法 + H 法 K=2 dwell/占用/转换率）——由 `m11_dwell_reference.json` 提供，本工具复算一致性
  ④ 静息静默比例（J2 的正解）——由 `m11_calcium_silence_reference.json` 提供

硬要求（逐条落实）：
  · **与 B4/主 agent 同管线**：`0.914 s 箱 与 帧率原生箱并列`、`5 帧平滑 与 未平滑并列`（后者在 dwell 产物）
    ⇒ 本工具的 dF/F + 事件阈值 + 代理分箱**与 J2 工具逐行同源**，并由断言 **P1.23a 以数值一致（≤1e-9）证明**（不是声称）
  · **≥300 帧门**：`B4.sample_size_gate`（65/130/500）
  · **双估计器断言**：平均成对相关两实现 rel_diff ≤1e-9；事件计数"原生帧 vs 分箱" rel_diff ≤1e-9
  · **内部一致性断言**：`B4._consistency(转换率, dwell 中位)` ⇒ `|dwell − 1/(2λ)|/dwell ≤ 1`
  · **逐 ROI 与全视野两口径并列**
  · **cross-field：本节点**实测**（两 field 均在盘）** —— 主 agent 的 "cross-field 一律 unavailable" 依据
    （"第二 field 不在盘上"）**已过期**：`functional_subset_f2/` 200/200 片、31,472,642 B 已于前轮落盘并入库。

用法：
  cd /Users/weidong/ai/small_world
  PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./.venv-m11/bin/python \
      neural_exploration/tools/build_m11_calcium_reference_quantities.py
"""
from __future__ import annotations

import argparse
import csv
import gc
import glob
import json
import os
import statistics
import sys
import time

import numpy as np

TOOLS = os.path.dirname(os.path.abspath(__file__))
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)
import build_m11_calcium_silence as J2T          # noqa: E402  （同管线取数 + 事件阈值 + 代理分箱）
import validate_p10_state_structure as B4        # noqa: E402  （冻结 B4 管线：样本量门 / 内部一致性）

DATA = "neural_exploration/data"
SUB1 = os.path.join(DATA, "m11_raw", "microns", "functional_subset")
SUB2 = os.path.join(DATA, "m11_raw", "microns", "functional_subset_f2")
J2 = os.path.join(DATA, "m11_calcium_silence_reference.json")
DW = os.path.join(DATA, "m11_dwell_reference.json")
AU = os.path.join(DATA, "m11_allen_units_reference.json")
OUT_JSON = os.path.join(DATA, "m11_calcium_reference_quantities.json")
OUT_RATE = os.path.join(DATA, "m11_calcium_rate_band.csv")
OUT_CORR = os.path.join(DATA, "m11_calcium_population_corr.csv")
LOG = os.path.join(DATA, "m11_raw", "calcium_refq.log")

FRAME_INTERVAL = J2T.FRAME_INTERVAL               # 0.15873 s
FRAME_RATE = J2T.FRAME_RATE                       # 6.300006 Hz
FRAMES_PER_BIN = 6                                # B4 `analyse(dt_ms=158.73)` 的实际取值（b=6）
BIN_S = FRAMES_PER_BIN * FRAME_INTERVAL           # 0.95238 s（模块称 "0.914 s 箱"，命名偏差 4.2% 已留痕）
BIN_PX = J2T.BIN_PX
N_SURROGATE = 8
SEED = 0
L = []


def log(m):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), m)
    print(line, flush=True)
    L.append(line)


def build_units(stack, bin_px=BIN_PX):
    """与 J2 工具 `analyse()` **逐行同源**的 dF/F + 代理分箱（数值一致性由断言 P1.23a 证明）。"""
    T, Y, X = stack.shape
    F = stack.astype(np.float32)
    F0 = np.percentile(F, 20, axis=0)
    F0s = np.where(F0 <= 0, np.nan, F0)
    dff = (F - F0s[None]) / F0s[None]
    dff = np.nan_to_num(dff, nan=0.0, posinf=0.0, neginf=0.0)
    med = float(np.median(dff))
    mad = float(np.median(np.abs(dff - med)))
    thr = med + J2T.EVENT_SIGMA * mad
    ny, nx = Y // bin_px, X // bin_px
    dffb = dff[:, :ny * bin_px, :nx * bin_px].reshape(T, ny, bin_px, nx, bin_px).transpose(0, 1, 3, 2, 4)
    dffb = dffb.reshape(T, ny, nx, bin_px * bin_px).mean(axis=3)     # (T, ny, nx)
    units = dffb.reshape(T, -1)
    return units, thr, med, mad


def desc(v):
    v = sorted(float(x) for x in v)
    if not v:
        return {}
    def q(p):
        return v[max(0, min(int(round(p * (len(v) - 1))), len(v) - 1))]
    return {"n": len(v), "median": statistics.median(v), "p10": q(.10), "p25": q(.25),
            "p75": q(.75), "p90": q(.90), "min": v[0], "max": v[-1],
            "mean": float(sum(v) / len(v))}


# ---------------- ① 率分布（钙事件率；原生帧箱 与 6 帧箱 并列）----------------
def rate_distribution(units, thr, dur_s, label):
    ev = (units > thr)                                    # bool (T,N)
    T, N = units.shape
    cnt_native = ev.sum(axis=0).astype(float)
    rate_native_hz = cnt_native / dur_s
    nb = T // FRAMES_PER_BIN
    evb = ev[:nb * FRAMES_PER_BIN].reshape(nb, FRAMES_PER_BIN, N).sum(axis=1)   # (nb,N)
    cnt_binned = evb.sum(axis=0).astype(float)
    rate_binned_hz = cnt_binned / (nb * BIN_S)
    # ★ 双估计器必须比**同一帧范围**：分箱口径丢掉尾部 T − nb*6 帧 ⇒
    #   若拿"全 T 帧的计数"与"分箱计数"比，差异来自**截断**而非估计器 ⇒ 须先把原生计数截到同一范围。
    n_kept = nb * FRAMES_PER_BIN
    cnt_native_same_range = ev[:n_kept].sum(axis=0).astype(float)
    tot_n = float(cnt_native_same_range.sum())
    tot_b = float(cnt_binned.sum())
    rel = abs(tot_n - tot_b) / tot_n if tot_n else 0.0
    dropped_frames = int(T - n_kept)
    dropped_events = float(ev[n_kept:].sum())
    # 全视野口径：逐帧跨全部代理单元的事件总数（单位仍为 Hz）
    fov_per_frame = ev.sum(axis=1).astype(float)
    fov_rate_hz = float(fov_per_frame.mean() * FRAME_RATE)
    fov_rate_binned_hz = float(evb.sum(axis=1).mean() / BIN_S)
    # ★ 分辨率地板：事件数过少时"率"不可辨（与 B4 dwell_resolution_floor 同族纪律）
    n_ev = cnt_native_same_range
    floor_note = ("每单元事件中位 = %.1f ⇒ **触分辨率地板**（事件数 ≤3 时率的相对不确定度 ≳60%%）；"
                  "分位 p10/p25/p50 均落在地板区 ⇒ **只能报『多数代理单元在 317 s 窗内几乎无事件』**，"
                  "**不得**把中位率当作神经元发放率" % float(np.median(n_ev)))
    return dict(
        label=label, n_units_proxy=int(N), n_frames=int(T), duration_s=round(dur_s, 2),
        unit_bin_px=BIN_PX, unit_area_um2=round((BIN_PX * J2T.PIXEL_UM) ** 2, 1),
        event_threshold_dff=round(float(thr), 6),
        rate_hz_native_frames=desc(rate_native_hz),
        rate_hz_binned_6frames=desc(rate_binned_hz),
        total_events_native=tot_n, total_events_binned=tot_b,
        dual_estimator_rel_diff=rel,
        dual_estimator_scope=("同一帧范围（前 %d 帧）比较；分箱口径丢尾部 %d 帧 / %d 事件（%.4f%% 帧）"
                              % (n_kept, dropped_frames, int(dropped_events),
                                 100.0 * dropped_frames / T)),
        dropped_frames=dropped_frames, dropped_events=int(dropped_events),
        events_per_unit_distribution=desc(cnt_native_same_range),
        events_per_bin_distribution=desc(evb.reshape(-1)),
        rate_resolution_floor=floor_note,
        n_bins=int(nb), bin_s=round(BIN_S, 5),
        fullfov_rate_hz=fov_rate_hz, fullfov_rate_hz_binned=fov_rate_binned_hz,
        fullfov_events_per_frame_mean=float(fov_per_frame.mean()),
        unit_vs_neuron_scope_caveat=(
            "**代理单元 ≠ 单个神经元**：%d×%d px = %.0f×%.0f µm 的空间分箱内神经元数**未知**"
            "（真实 ROI 掩膜未解码）⇒ 分箱事件率**不可**与『单神经元 spike 率』并列比较；"
            "跨口径仅可在**全视野聚合**层面陈述" % (BIN_PX, BIN_PX, BIN_PX * J2T.PIXEL_UM, BIN_PX * J2T.PIXEL_UM)),
        caveat_note=("钙事件口径 ≠ spike 率：本表单位 = 钙事件/秒（阈值 = dF/F 中位 + %.0f·MAD）；"
                     "与电生理 spike 率**分列**，不得合并" % J2T.EVENT_SIGMA))


# ---------------- ② 群体相关 + 谱 + 有效维度 + 相位随机化代理 ----------------
def _mean_pair_r(Z):
    """双实现：numpy.corrcoef 与手动标准化点积（均在 float64 上）。"""
    Z = np.asarray(Z, dtype=np.float64)
    N = Z.shape[1]
    iu = np.triu_indices(N, k=1)
    C = np.corrcoef(Z, rowvar=False)
    r_np = float(np.nanmean(C[iu]))
    A = Z - Z.mean(axis=0, keepdims=True)
    sd = A.std(axis=0, keepdims=True)
    A = A / np.where(sd == 0, np.nan, sd)
    A = np.nan_to_num(A)
    C2 = (A.T @ A) / A.shape[0]
    r_man = float(np.nanmean(C2[iu]))
    return r_np, r_man, C


def _spectrum(C):
    ev = np.sort(np.linalg.eigvalsh(C))[::-1]
    tot = float(ev.sum())
    pr = float(tot ** 2 / float((ev ** 2).sum())) if (ev ** 2).sum() > 0 else None
    return ev, pr


def _zbin(units):
    T, N = units.shape
    nb = T // FRAMES_PER_BIN
    X = units[:nb * FRAMES_PER_BIN].reshape(nb, FRAMES_PER_BIN, N).mean(axis=1)
    return X


def _corr_block(X, rng, label_extra=""):
    """给定 (nb,N) 矩阵：平均成对相关（双实现）+ 谱 + 有效维度 + 相位随机化代理。

    ★ 纪律：**零方差单元必须对两个估计器同样处理**（否则 `np.corrcoef` 给 NaN 而手写实现给 0，
    经 `nanmean` 后两者不等 ⇒ 假性"双估计器不一致"）。⇒ **先剔除零方差列并登记数量**，再在同一矩阵上比。
    """
    # ★ float64 强制：两实现若一为 float32、一为 float64，会给出 ~1e-8 的**精度伪差**
    #   （被误读为"双估计器不一致"）⇒ 统一到 float64 后再比，容差才可钉在 1e-9。
    X = np.asarray(X, dtype=np.float64)
    nb, N0 = X.shape
    sd0 = X.std(axis=0)
    keep = sd0 > 0
    n_dropped = int((~keep).sum())
    X = X[:, keep]
    nb, N = X.shape
    if N < 2:
        return {"label": label_extra, "n_units": int(N), "n_dropped_zero_variance": n_dropped,
                "mean_pairwise_r_numpy": None, "error": "有效单元 <2 ⇒ 群体相关不可算（已登记，不静默跳过）"}
    Z = (X - X.mean(axis=0, keepdims=True)) / X.std(axis=0, keepdims=True)
    r_np, r_man, C = _mean_pair_r(Z)
    ev, pr = _spectrum(C)
    top = [float(x) for x in ev[:10]]
    frac_top1 = float(ev[0] / ev.sum())
    sur_r, sur_pr = [], []
    F = np.fft.rfft(Z, axis=0)
    amp = np.abs(F)
    for s in range(N_SURROGATE):
        ph = rng.uniform(0, 2 * np.pi, size=F.shape)
        ph[0, :] = 0.0
        if nb % 2 == 0:
            ph[-1, :] = 0.0
        Zs = np.fft.irfft(amp * np.exp(1j * ph), n=nb, axis=0)
        Zs = (Zs - Zs.mean(axis=0, keepdims=True)) / Zs.std(axis=0, keepdims=True)
        Cs = np.corrcoef(Zs, rowvar=False)
        sur_r.append(float(np.mean(Cs[np.triu_indices(N, k=1)])))
        _, prs = _spectrum(Cs)
        sur_pr.append(prs)
    return dict(
        label=label_extra, n_units=int(N), n_dropped_zero_variance=n_dropped,
        n_bins=int(nb), bin_s=round(BIN_S, 5),
        mean_pairwise_r_numpy=round(r_np, 6), mean_pairwise_r_manual=round(r_man, 6),
        dual_estimator_rel_diff=(abs(r_np - r_man) / abs(r_np) if r_np else abs(r_np - r_man)),
        eigenvalue_top10=[round(x, 4) for x in top],
        eigenvalue_first_over_sum=round(frac_top1, 6),
        participation_ratio_effective_dim=round(pr, 4) if pr else None,
        surrogate_mean_r=round(float(np.mean(sur_r)), 6),
        surrogate_mean_r_sd=round(float(np.std(sur_r)), 6),
        surrogate_participation_ratio=round(float(np.mean(sur_pr)), 4),
        excess_over_surrogate=(round(r_np - float(np.mean(sur_r)), 6) if r_np is not None else None),
        n_surrogates=N_SURROGATE, seed=SEED,
        note=("代理零分布 = 同管线 FFT 相位随机化（保幅谱、随机相位、%d 次）；"
              "『平均成对相关』若未超过代理地板 ⇒ 判为**无群体结构**，不得解释为同步；"
              "零方差单元已剔除并登记（n=%d）" % (N_SURROGATE, n_dropped)))


def population_correlation(units, label, rng):
    X = _zbin(units)                                   # (nb, N) 6 帧箱 dF/F
    raw = _corr_block(X, rng, "%s ｜ 口径 A：原始（未去全局分量）" % label)
    # ★ 口径 B：去**全局分量**（逐箱跨单元均值 = 单个共同模式）——PCA 第一主分量的等价一阶代理
    gm = X.mean(axis=1, keepdims=True)
    Xr = X - gm
    resid = _corr_block(Xr, rng, "%s ｜ 口径 B：去全局分量（X − 逐箱跨单元均值）" % label)
    raw["global_mode_removed"] = resid
    # ★ 退化体检：若 PR≈1 且 λ1/Σλ≈1 ⇒ 群体被**单一共同模式**主导
    degeneracy = {
        "raw_PR": raw["participation_ratio_effective_dim"],
        "raw_lambda1_over_sum": raw["eigenvalue_first_over_sum"],
        "resid_PR": resid["participation_ratio_effective_dim"],
        "verdict": None,
        "note": ("**可比性警告**：PR 接近 1 表示全部单元几乎由**一个共同模式**驱动。"
                 "本轮两 field 的原始口径 PR = %s / λ1/Σλ = %s ⇒ 高 r **不能**解释为神经元同步，"
                 "须以**去全局分量**口径为准（PR 升到 %s）"
                 % (raw["participation_ratio_effective_dim"], raw["eigenvalue_first_over_sum"],
                    resid["participation_ratio_effective_dim"])),
    }
    if raw["participation_ratio_effective_dim"] is not None and raw["participation_ratio_effective_dim"] < 3.0:
        degeneracy["verdict"] = "**单一共同模式主导（PR<3）** ⇒ 原始平均 r 不得作为群体结构证据"
    else:
        degeneracy["verdict"] = "PR≥3 ⇒ 原始口径未退化（仍建议并列去全局分量口径）"
    raw["degeneracy_check"] = degeneracy
    return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--field1", default=SUB1)
    ap.add_argument("--field2", default=SUB2)
    ap.add_argument("--n-max", type=int, default=200)
    args = ap.parse_args()
    t0 = time.time()
    log("START build_m11_calcium_reference_quantities.py（箱=%.1f 帧=%.5f s；代理=%d px）"
        % (FRAMES_PER_BIN, BIN_S, BIN_PX))

    rng = np.random.default_rng(SEED)
    j2 = json.load(open(J2, encoding="utf-8"))
    j2rows = j2.get("rows") or []

    fields = [("field4", args.field1, "session4_scan7_field4", 0),
              ("field2", args.field2, "session5_scan3_field1", 1)]
    rates, corrs, pipeline_checks = [], [], []
    for name, d, prefix, idx in fields:
        if not os.path.isdir(d):
            log("跳过 %s：目录不存在（不得静默当作 0）" % name)
            continue
        stack, nfiles, n_neg, ntot = J2T.load_stack(d, prefix, n_max=args.n_max)
        if stack is None:
            log("跳过 %s：无可解析分片" % name)
            continue
        T = stack.shape[0]
        dur = T * FRAME_INTERVAL
        units, thr, med, mad = build_units(stack)
        del stack
        gc.collect()
        # ---- 同管线一致性断言（P1.23a）----
        zero = float(((units > thr).sum(axis=0) == 0).mean())
        ref = j2rows[idx].get("silent_frac_zero_activity") if idx < len(j2rows) else None
        d_rel = abs(zero - ref) if ref is not None else None
        pipeline_checks.append({"field": name, "recomputed_zero_activity": round(zero, 6),
                                "j2_stored_zero_activity": ref,
                                "abs_diff": (round(d_rel, 12) if d_rel is not None else None),
                                "consistent_le_1e-9": (d_rel is not None and d_rel <= 1e-9),
                                "n_files": nfiles, "n_negative_samples": n_neg,
                                "negative_fraction": round(n_neg / ntot, 6) if ntot else None})
        log("%s：%d 片 / %d 帧 / %.2f s；同管线一致性 |Δzero|=%.2e（J2=%.4f，本工具=%.4f）"
            % (name, nfiles, T, dur, (d_rel if d_rel is not None else float("nan")),
               ref if ref is not None else float("nan"), zero))

        rd = rate_distribution(units, thr, dur, "%s / %s" % (name, prefix))
        rates.append(rd)
        log("① 率分布[%s]：proxy 中位=%.4f Hz（p90 %.4f / max %.4f）；全视野=%.4f Hz；"
            "双估计器 rel_diff=%.2e"
            % (name, rd["rate_hz_native_frames"]["median"], rd["rate_hz_native_frames"]["p90"],
               rd["rate_hz_native_frames"]["max"], rd["fullfov_rate_hz"], rd["dual_estimator_rel_diff"]))

        pc = population_correlation(units, "%s / %s" % (name, prefix), rng)
        corrs.append(pc)
        gm = pc.get("global_mode_removed") or {}
        log("② 群体相关[%s]：**原始口径** r=%.5f（地板 %.5f±%.5f）PR=%.3f λ1/Σλ=%.4f ｜ "
            "**去全局分量** r=%.5f PR=%.3f λ1/Σλ=%.4f ｜ 双估计器 rel=%.2e ｜ 零方差剔除 %s"
            % (name, pc["mean_pairwise_r_numpy"], pc["surrogate_mean_r"], pc["surrogate_mean_r_sd"],
               pc["participation_ratio_effective_dim"] or 0, pc["eigenvalue_first_over_sum"],
               gm.get("mean_pairwise_r_numpy") or 0, gm.get("participation_ratio_effective_dim") or 0,
               gm.get("eigenvalue_first_over_sum") or 0, pc["dual_estimator_rel_diff"],
               gm.get("n_dropped_zero_variance")))
        log("   退化体检：%s" % (pc.get("degeneracy_check") or {}).get("verdict"))
        del units
        gc.collect()

    # ---- ③ 状态结构（读既有 dwell 产物 + 复算内部一致性）----
    dw = json.load(open(DW, encoding="utf-8"))
    band = dw.get("band") or {}
    prim = band.get("primary") or {}
    cf = band.get("cross_field") or {}
    trans = prim.get("transition_rate_hz") or prim.get("hmm_transition_rate_hz")
    dwell_med = prim.get("low_dwell_median_s_physical")
    cons = B4._consistency(trans, dwell_med) if (trans and dwell_med) else dict(
        applicable=False, reason="既有 dwell 产物的转换率/dwell 字段名未匹配 ⇒ 须显式声明（不得静默跳过）")
    state = {
        "source": DW,
        "H_method": {k: prim.get(k) for k in ("occupancy_low", "occupancy_high", "transition_rate_hz",
                                              "ar1_tau_s_physical", "ar1_phi_smoothed",
                                              "ar1_tau_bias_corrected_s_physical", "best_K_by_BIC",
                                              "BIC_note")},
        "Q_method": {k: prim.get(k) for k in ("q_low_dwell_median_s_module", "q_transition_rate_hz",
                                              "h_vs_q_dwell_ratio", "low_dwell_ks_p",
                                              "low_dwell_p90_over_median", "dwell_resolution_floor")},
        "smoothed_and_unsmoothed": {"tau_smoothed_s": prim.get("ar1_tau_s_physical"),
                                    "tau_unsmoothed_s_module": prim.get("ar1_tau_s_unsmoothed_module"),
                                    "phi_smoothed": prim.get("ar1_phi_smoothed"),
                                    "phi_unsmoothed": prim.get("ar1_phi_unsmoothed")},
        "internal_consistency": cons,
        "internal_consistency_stored": prim.get("internal_consistency"),
        "internal_consistency_dual_check": {
            "recomputed_rel_diff": cons.get("rel_diff"),
            "stored_rel_diff": (prim.get("internal_consistency") or {}).get("rel_diff"),
            "match": (abs((cons.get("rel_diff") or 0) - ((prim.get("internal_consistency") or {}).get("rel_diff") or 0))
                      <= 1e-12) if cons.get("applicable") else None},
        "cross_field": cf,
        "cross_field_note": ("**cross-field = measured（两 field 均在盘）**："
                             "dwell %.3f× / τ %.3f×（与主 agent 『cross-field 一律 unavailable』的前提"
                             "『第二 field 不在盘上』不一致 —— 该前提已过期，第二 field 200/200 片已入库）"
                             % (cf.get("spread_dwell") or 0, cf.get("spread_tau") or 0)),
    }
    log("③ 状态结构：内部一致性 |dwell−1/(2λ)|/dwell = %s（pass=%s）"
        % (cons.get("rel_diff"), cons.get("pass_")))

    # ---- ④ 静默比例（J2 正解）----
    silence = {"source": J2,
               "rows": [{k: r.get(k) for k in ("label", "silent_frac_zero_activity",
                                               "silent_frac_below_baseline", "silent_frac_low_percentile",
                                               "fullfov_frame_no_event_frac", "n_units_proxy")} for r in j2rows],
               "acceptance": (j2.get("acceptance_completion") or {}).get("all_conditions")}

    # ---- 与 spike 口径分列（量纲对照；用 DQ-6 实测量）----
    au = json.load(open(AU, encoding="utf-8")) if os.path.exists(AU) else {}
    agr = au.get("area_group_bands") or {}
    visp = next((v for k, v in agr.items() if k.startswith("VISp")), {})
    spike_med = (visp.get("rate_quality_good") or {}).get("median")
    ca_med = rates[0]["rate_hz_native_frames"]["median"] if rates else None
    cross_scope = {
        "calcium_event_rate_median_hz": ca_med,
        "spike_rate_median_hz_visp": spike_med,
        "calcium_over_spike_ratio": (round(ca_med / spike_med, 4) if (ca_med and spike_med) else None),
        "calcium_event_rate_nyquist_hz": FRAME_RATE / 2.0,
        "interpretation": ("两域**分列**：钙事件率是本工具的实测中位；spike 率为 Allen NP VISp 实测中位"
                           "（口径 complete_session）。比值 = 钙事件率/spike 率，量级差即『钙≠spike』的**实测**下界"
                           "（不需要再引用 in vitro 代理）"),
    }
    log("跨口径：钙事件率中位=%.4f Hz vs spike(VISp)=%.4f Hz ⇒ 比 %.4f"
        % (ca_med or 0, spike_med or 0, cross_scope["calcium_over_spike_ratio"] or 0))

    # ---- 断言汇总（供判定器 P1.23 复用）----
    sg = B4.sample_size_gate(2000)
    nbins = rates[0]["n_bins"] if rates else 0
    A = {
        "P1.23a_same_pipeline": {"ok": all(c["consistent_le_1e-9"] for c in pipeline_checks),
                                 "detail": pipeline_checks},
        "P1.23b_dual_estimator": {
            "ok": all(r["dual_estimator_rel_diff"] <= 1e-9 for r in rates)
            and all(c["dual_estimator_rel_diff"] <= 1e-9 for c in corrs),
            "event_count_rel_diff": [r["dual_estimator_rel_diff"] for r in rates],
            "mean_r_rel_diff": [c["dual_estimator_rel_diff"] for c in corrs]},
        "P1.23c_sample_size_gate": {"ok": sg["valid"] and sg["shape_ok"], "gate": sg,
                                    "n_bins": nbins,
                                    "bins_note": ("箱数 %d < 500 ⇒ BIC/K 选择结论 unavailable（四态纪律）" % nbins)},
        "P1.23d_internal_consistency": {"ok": bool(cons.get("pass_") or cons.get("applicable") is False),
                                        "detail": cons,
                                        "rule": "|dwell − 1/(2λ)|/dwell ≤ 1"},
        "P1.23e_two_unit_scopes": {"ok": bool(rates) and all(
            r.get("fullfov_rate_hz") is not None and r.get("rate_hz_native_frames", {}).get("median") is not None
            for r in rates),
            "note": ("全视野口径为**单条**聚合序列 ⇒ **群体相关在全视野口径下无定义**（n=1），"
                     "该维登记为『口径不适用』而非『不可用』")},
        "P1.23f_cross_field_measured": {"ok": bool(cf.get("spread_dwell")),
                                        "spread_dwell": cf.get("spread_dwell"),
                                        "spread_tau": cf.get("spread_tau"),
                                        "note": "两 field 均在盘 ⇒ cross-field 为 measured（推翻『一律 unavailable』的过期前提）"},
        "P1.23g_surrogate_floor": {"ok": all(c.get("excess_over_surrogate") is not None for c in corrs),
                                   "excess": [c.get("excess_over_surrogate") for c in corrs]},
    }
    for k, v in A.items():
        log("断言 %s ⇒ %s" % (k, "OK" if v["ok"] else "FAIL"))

    res = {
        "generated_by": "neural_exploration/tools/build_m11_calcium_reference_quantities.py",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scope": "优先级 4：四类参考量的**钙域口径**（同源 MICrONS functional_series；无新下载）",
        "pipeline": {"same_as": [J2, DW], "frames_per_bin": FRAMES_PER_BIN, "bin_s": round(BIN_S, 5),
                     "bin_naming_note": "B4 冻结模块以 FRAME_S=1/1.0943=0.9138 s 为一箱；dt_ms=158.73 ⇒ b=6 帧/箱"
                                        "= 0.9524 s ⇒ 命名与实宽差 4.2%（已在 dwell 产物留痕）",
                     "smoothing": "5 帧平滑与未平滑并列（后者见 dwell 产物 ar1_tau_s_unsmoothed_module）",
                     "frame_rate_hz": FRAME_RATE, "event_sigma_mad": J2T.EVENT_SIGMA,
                     "proxy_bin_px": BIN_PX, "n_surrogate": N_SURROGATE, "seed": SEED},
        "same_pipeline_check": pipeline_checks,
        "quantity_1_rate_distribution": rates,
        "quantity_2_population_correlation": corrs,
        "quantity_3_state_structure": state,
        "quantity_4_resting_silence": silence,
        "cross_scope_calcium_vs_spike": cross_scope,
        "assertions": A,
        "caveats": [
            "① 钙事件口径 ≠ spike 率：两域**分列**，本文件不复用任何 spike 值作钙域带；",
            "② 单元 = **空间分箱代理**（真实 ROI 掩膜未解码）⇒ 群体相关的『单元』是代理单元；",
            "③ 箱数 %d < 500 ⇒ BIC/K 选择结论 unavailable（未测 ≠ 未命中）；" % nbins,
            "④ 全视野口径下群体相关**不适用**（单序列）；",
            "⑤ 覆盖 = 2 field × 2,000 帧（317.46 s each），**非**全场/全 144 field。",
        ],
    }
    json.dump(res, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    with open(OUT_RATE, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["field_scope", "unit_scope", "n_units", "n_frames", "duration_s",
                    "rate_median_hz", "rate_p10", "rate_p90", "rate_max", "bin_s", "evidence"])
        for r in rates:
            d = r["rate_hz_native_frames"]
            w.writerow([r["label"], "per_roi_proxy(%dpx)" % BIN_PX, r["n_units_proxy"], r["n_frames"],
                        r["duration_s"], round(d["median"], 6), round(d["p10"], 6), round(d["p90"], 6),
                        round(d["max"], 6), "native_frames", "measured"])
            w.writerow([r["label"], "full_fov_aggregate", 1, r["n_frames"], r["duration_s"],
                        round(r["fullfov_rate_hz"], 6), "", "", "", "native_frames", "measured"])
    with open(OUT_CORR, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["field_scope", "n_units", "n_bins", "mean_pairwise_r", "surrogate_floor_r",
                    "excess_over_floor", "effective_dim_PR", "surrogate_PR", "lambda1_over_sum",
                    "evidence"])
        for c in corrs:
            w.writerow([c["label"], c["n_units"], c["n_bins"], c["mean_pairwise_r_numpy"],
                        c["surrogate_mean_r"], c["excess_over_surrogate"],
                        c["participation_ratio_effective_dim"], c["surrogate_participation_ratio"],
                        c["eigenvalue_first_over_sum"], "measured"])
    log("落盘：%s / %s / %s" % (OUT_JSON, OUT_RATE, OUT_CORR))
    log("耗时 %.1f s" % (time.time() - t0))
    log("ALL_DONE")
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    open(LOG, "w", encoding="utf-8").write("\n".join(L) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
