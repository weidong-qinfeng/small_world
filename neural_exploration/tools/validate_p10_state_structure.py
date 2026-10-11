"""M10 RF-10 落实：P6 第三条自检门的**状态结构判据**（与成像侧同实现 + 同分辨率）。

主agent RF-10（B4 实测，`data/m10_imaging_dwell_reference.{csv,json}`）：
  - **口径差已标定**：同一序列 H 法/Q 法低态 dwell 中位比 **3.87×**（未平滑 1 帧序列上 **13.0×**）
    ⇒ 两侧必须**同实现**；本工具采用 B4 建议：**模型侧聚合到 1 帧 = 0.9138 s 箱**，
    再用 `gaussian_hmm_2state`（seed=0, n_iter=60，与成像侧**同一实现**）；**Q 法作并列对照**。
  - **成像侧 H 法基准**：低态 dwell 中位 **26.5008 s** / mean 68.14 / **p90 196.5** / n=71；
    转换率 **0.01958 Hz**；占用 **0.672/0.328**。
  - **状态数**：BIC 与 CV 都选 **K=3**（K1 ΔBIC 20438 / K2 5632 / K3 0）⇒ 模型侧**必须同时报 K=2/K=3**。
  - **分布形状**：低态 dwell **拒绝指数**（KS p=0.0016）、**重尾**（p90/中位 7.4×）⇒ 这是可证伪的目标形态。
  - **跨条件离散度极大**（等长控制 max/min 26.2×）⇒ 同阶窗 **[0.038, 26.2]** 极宽；
    **绝对 dwell 不作"必过"硬门**，只报告并标注"落在窗内不足以证明机制正确"。

**P6 第三条门（新定义，取代 bout>0）**：与成像**同实现 + 同分辨率**下
  ① 占用非退化（0.05 < occ_lo < 0.95）；② 转换率 > 0；③ **低态 dwell p90/中位 ≥ 2 且拒绝指数（KS p<0.05）**；
  ④ 慢变量与状态序列的相关 |r| ≥ 0.30（机制归属）。**bout 仅描述性输出**（D2）。

用法：
    OMP_NUM_THREADS=1 PYTHONHASHSEED=0 ./.venv-m9/bin/python -m neural_exploration.tools.validate_p10_state_structure \
        --level M-6 --n-neuron 250 --T-ms 60000
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FRAME_S = 1.0 / 1.0943          # 1 帧 = 0.9138 s（与成像同分辨率）
SMOOTH_FRAMES = 5               # RF-14 主判据管线：0.914 s 箱 → 5 帧滑动平均 → z → AR(1)
# **RF-16-3 定稿**：成像侧同口径 AR(1) τ 基准 = **37.180 s**（B4 追加任务 G，**全记录**；
# 此前引用的 42.332 s 是"前 80% 段"值 → 13.9% 口径差，已更正）。
# 容差：**跨条件 τ 极差 3.935×** ⇒ 窗 = **[37.180/3.94, 37.180×3.94] = [9.4, 146.3] s**（±3.94×，B4 建议 ±4× 已采纳）。
TAU_IMAGING_AR1_S = 37.180
TAU_WINDOW_S = (9.4, 146.3)
TAU_REF_SOURCE = "B4 追加任务 G：data/m10_imaging_slow_tau_reference.json（φ=0.9757213, 全记录）"
# ================== **RF-18 样本量硬门** ==================
MIN_FRAMES_VALID = 65      # n_frames < 65 ⇒ status='invalid'（最短协议 T ≥ 60 s）
MIN_FRAMES_SHAPE = 130     # 形状类判据（重尾 / KS 拒绝指数）要求 n_frames ≥ 130（T ≥ 120 s）
MIN_FRAMES_BIC = 500       # BIC/K 选择结论仅在 n_frames ≥ 500 时报告，否则 unavailable
TAU_MIN_FRAMES = 300       # RF-21(1)：n_frames < 300 时 **AR(1) τ 判据禁用**（半段口径 9.74× ⇒ 短记录不稳）
DWELL_WINDOW = (0.038, 26.2)    # B4 等长控制 max/min 定（极宽；仅描述性）
IMAGING_REF_JSON = os.path.join(DATA_DIR, "m10_imaging_dwell_reference.json")


def _ks_vs_exponential(x: np.ndarray) -> dict:
    x = np.asarray(x, float); x = x[x > 0]
    if x.size < 5:
        return dict(n=int(x.size), D=None, p=None, pass_=False)
    lam = 1.0 / x.mean()
    xs = np.sort(x)
    cdf = 1.0 - np.exp(-lam * xs)
    i = np.arange(1, xs.size + 1)
    D = float(max(np.max(i / xs.size - cdf), np.max(cdf - (i - 1) / xs.size)))
    ne = xs.size
    lamk = (math.sqrt(ne) + 0.12 + 0.11 / math.sqrt(ne)) * D
    p = float(2.0 * sum((-1) ** (k - 1) * math.exp(-2.0 * k * k * lamk * lamk)
                        for k in range(1, 101)))
    return dict(n=int(xs.size), D=D, p=p, pass_=bool(p < 0.05))


def _dwells(state: np.ndarray, dt_s: float):
    out = {}
    for s_, nm in ((0, "low"), (1, "high")):
        b = (state == s_).astype(np.int8)
        d = np.diff(np.r_[0, b, 0])
        st = np.flatnonzero(d == 1); en = np.flatnonzero(d == -1)
        interior = (st > 0) & (en < state.size)
        ln = (en[interior] - st[interior]) * dt_s
        out[nm] = dict(n=int(ln.size),
                       median_s=float(np.median(ln)) if ln.size else 0.0,
                       mean_s=float(ln.mean()) if ln.size else 0.0,
                       p90_s=float(np.quantile(ln, 0.9)) if ln.size else 0.0,
                       p90_over_median=(float(np.quantile(ln, 0.9) / np.median(ln))
                                        if ln.size and np.median(ln) > 0 else None),
                       ks_vs_exp=_ks_vs_exponential(ln))
    return out


def frame_series(pop_counts_steps: np.ndarray, dt_ms: float):
    """RF-14 统一管线：逐步群体脉冲数 → **0.914 s 箱** → **5 帧滑动平均** → z 分数。

    （成像侧对同一管线；**禁止**用 0.01 s 箱的分位法 dwell 直接与成像比：B4 实测外推差 91–228×。）"""
    b = max(1, int(round(FRAME_S / (dt_ms * 1e-3))))
    nb = pop_counts_steps.size // b
    frames = pop_counts_steps[: nb * b].reshape(nb, b).sum(axis=1).astype(float)
    if nb >= SMOOTH_FRAMES:
        k = np.ones(SMOOTH_FRAMES) / SMOOTH_FRAMES
        sm = np.convolve(frames, k, mode="valid")
    else:
        sm = frames
    z = (sm - sm.mean()) / (sm.std() + 1e-12)
    return frames, z


def _ar1_phi_both(z_smoothed, frames_raw):
    """并列报"平滑后/未平滑"两种口径的 φ（RF-24(3) 要求）。"""
    def _phi(x):
        x = np.asarray(x, float)
        if x.size < 20:
            return None
        x = (x - x.mean()) / (x.std() + 1e-12)
        x0, x1 = x[:-1], x[1:]
        d = float(x0 @ x0)
        return float((x0 @ x1) / d) if d > 0 else None
    return _phi(z_smoothed), (_phi(frames_raw) if frames_raw is not None else None)


def ar1_fit(z: np.ndarray, dt_s: float, folds: int = 4, lb_lags: int = 10):
    """AR(1)：x_t = φ x_{t-1} + ε → φ、τ = −dt/ln φ、**留出 ll/帧**、**残差白噪检验（Ljung-Box）**。"""
    x = np.asarray(z, float)
    if x.size < 20:
        return dict(status="invalid", reason=f"序列过短 n={x.size}")
    x0, x1 = x[:-1], x[1:]
    phi = float((x0 @ x1) / (x0 @ x0)) if (x0 @ x0) > 0 else float("nan")
    if not (0.0 < phi < 1.0):
        return dict(status="invalid", reason=f"φ={phi} 不在 (0,1)", phi=phi)
    tau = float(-dt_s / math.log(phi))
    resid = x1 - phi * x0
    # ---- 留出 ll/帧（4 折连续切分；用训练折的残差方差评测试折）----
    ll_te, n_te = 0.0, 0
    idx = np.array_split(np.arange(x.size), folds)
    for i in range(folds):
        te = idx[i]
        if te.size < 5:
            continue
        tr = np.concatenate([idx[j] for j in range(folds) if j != i])
        a0, a1 = x[tr][:-1], x[tr][1:]
        ph = float((a0 @ a1) / (a0 @ a0))
        rr = a1 - ph * a0
        var = float(rr.var() + 1e-12)
        tt0, tt1 = x[te][:-1], x[te][1:]
        e = tt1 - ph * tt0
        ll_te += float(-0.5 * (np.log(2 * math.pi * var) + (e ** 2) / var).sum())
        n_te += e.size
    ll_per_frame = (ll_te / n_te) if n_te else float("nan")
    # ---- Ljung-Box（残差白噪检验；χ²(df=lb_lags) 用 Wilson–Hilferty 近似）----
    r = resid - resid.mean()
    n = r.size
    rho = [float((r[:-k] @ r[k:]) / (r @ r)) for k in range(1, lb_lags + 1)]
    Q = float(n * (n + 2) * sum((rho[k - 1] ** 2) / (n - k) for k in range(1, lb_lags + 1)))
    df = lb_lags - 1
    zz = ((Q / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    p = float(1.0 - 0.5 * (1 + math.erf(zz / math.sqrt(2))))
    rho1 = rho[0]
    return dict(status="measured", phi=phi, tau_s=tau, holdout_ll_per_frame=ll_per_frame,
                ljung_box=dict(Q=Q, df=df, p=p, lags=lb_lags),
                rho1=rho1, rho1_ci=float(1.96 / math.sqrt(n)),
                resid_whiten=bool(p > 0.05), n_frames=int(x.size))


def hmm_kstate(x: np.ndarray, K: int, n_iter: int = 60, seed: int = 0):
    """K 状态对角高斯 HMM（EM）——K=2 时与成像侧 `gaussian_hmm_2state` 同实现语义。"""
    rng = np.random.default_rng(seed)
    xx = np.asarray(x, float).reshape(-1, 1)
    T = xx.size
    qs = np.quantile(xx, np.linspace(0.1, 0.9, K))
    mu = qs.astype(float); sd = np.full(K, xx.std() + 1e-6)
    A = np.full((K, K), 1.0 / K); np.fill_diagonal(A, 0.99)
    A = A / A.sum(axis=1, keepdims=True); pi = np.full(K, 1.0 / K)
    def logemit():
        return (-0.5 * ((xx - mu) ** 2 / sd ** 2 + np.log(2 * np.pi * sd ** 2)))
    ll_old = -np.inf
    for _ in range(n_iter):
        lb = logemit(); logA = np.log(A + 1e-300)
        la = np.zeros((T, K)); la[0] = np.log(pi + 1e-300) + lb[0]
        for t in range(1, T):
            m = la[t - 1][:, None] + logA
            mx = m.max(axis=0); la[t] = mx + np.log(np.exp(m - mx).sum(axis=0)) + lb[t]
        mx = la[-1].max(); ll = float(mx + np.log(np.exp(la[-1] - mx).sum()))
        lbk = np.zeros((T, K))
        for t in range(T - 2, -1, -1):
            m = logA + (lb[t + 1] + lbk[t + 1])[None, :]
            mx = m.max(axis=1, keepdims=True)
            lbk[t] = (mx + np.log(np.exp(m - mx).sum(axis=1, keepdims=True))).ravel()
        lg = la + lbk
        mx = lg.max(axis=1, keepdims=True)
        g = np.exp(lg - mx); g = g / g.sum(axis=1, keepdims=True)
        w = g.sum(axis=0) + 1e-300
        mu = (g * xx).sum(axis=0) / w
        sd = np.sqrt((g * (xx - mu) ** 2).sum(axis=0) / w) + 1e-9
        xi = np.zeros((K, K))
        for t in range(T - 1):
            m = la[t][:, None] + logA + (lb[t + 1] + lbk[t + 1])[None, :]
            m = np.exp(m - m.max()); xi += m / m.sum()
        A = xi / np.maximum(xi.sum(axis=1, keepdims=True), 1e-300)
        A = np.clip(A, 1e-6, 1 - 1e-6); A = A / A.sum(axis=1, keepdims=True)
        pi = g[0] / g[0].sum()
        if abs(ll - ll_old) < 1e-6:
            break
        ll_old = ll
    order = np.argsort(mu); mu = mu[order]; sd = sd[order]
    g = g[:, order]; A = A[np.ix_(order, order)]
    state = g.argmax(axis=1).astype(np.int8)
    n_par = K * (K - 1) + 2 * K
    bic = float(-2 * ll + n_par * math.log(T))
    return dict(K=K, mu=mu.tolist(), sigma=sd.tolist(), A=A.tolist(), loglik=ll,
                bic=bic, state=state, n_par=n_par)


def analyse(pop_counts_steps: np.ndarray, dt_ms: float, slow_trace=None, slow_dt_s=None,
            label: str = "model", n_neurons: int = 250):
    """把逐步群体脉冲数聚合到 **1 帧 = 0.9138 s** 箱，再做 H 法（同实现）+ Q 法（并列）。"""
    b = max(1, int(round(FRAME_S / (dt_ms * 1e-3))))
    nb = pop_counts_steps.size // b
    frames = pop_counts_steps[: nb * b].reshape(nb, b).sum(axis=1).astype(float)
    z = (frames - frames.mean()) / (frames.std() + 1e-12)
    # **RF-24(1) 口径修正**：`frames` 是**每帧全群脉冲数** ⇒ 逐神经元率必须 **/n_neurons**
    # （首版漏除 ⇒ 报 9.61 Hz，实测应为 0.224 Hz，差 43× 的第四例同族事故）。
    rate_trace = float(frames.mean() / FRAME_S / max(1, n_neurons))
    # **A3 硬断言（RF-24 要求）**：trace 级与帧级两个独立估计器必须一致（≤2%）
    _sum = float(frames.sum())
    _dur = nb * FRAME_S
    rate_trace_check = _sum / _dur / max(1, n_neurons)
    rel = abs(rate_trace - rate_trace_check) / max(rate_trace, 1e-12)
    assert rel <= 0.02, (f"**A3 双估计器硬断言失败**：pop_rate(均值法)={rate_trace:.6f} vs "
                         f"(总量法)={rate_trace_check:.6f} 相对差 {rel:.4f} > 0.02")
    # **RF-25(1)(2)**：双字段显式标注语义 + 断言（"双估计器必须一致"的又一次落地）
    pop_pop = float(rate_trace * n_neurons)
    _dev = abs(rate_trace * n_neurons - pop_pop) / max(pop_pop, 1e-12)
    assert _dev <= 1e-6, f"pop_rate 双字段一致性断言失败：相对偏差 {_dev}"
    out = dict(label=label, n_frames=int(nb), frame_s=FRAME_S, n_neurons=int(n_neurons),
               pop_rate_neuron_hz=rate_trace,          # ← **逐神经元**语义（与成像侧可比）
               pop_rate_population_hz=pop_pop,         # ← **群体总率**（= 逐神经元 × n）
               pop_rate_hz=rate_trace,                 # 兼容旧字段（**逐神经元**语义）
               pop_rate_hz_total_method=rate_trace_check,
               pop_rate_estimator_rel_diff=rel,
               pop_rate_field_assertion_rel_diff=_dev, total_spikes=int(_sum),
               fano=float(frames.var() / max(frames.mean(), 1e-12)))
    # --- H 法（同实现：2 状态高斯 HMM，seed=0 n_iter=60）---
    h2 = hmm_kstate(z, 2, n_iter=60, seed=0)
    dw2 = _dwells(h2["state"], FRAME_S)
    sw = int((np.diff(h2["state"]) != 0).sum())
    out["H_K2"] = dict(occupancy=[float((h2["state"] == 0).mean()),
                                  float((h2["state"] == 1).mean())],
                       transition_rate_hz=float(sw / (nb * FRAME_S)),
                       dwell=dw2, bic=h2["bic"], loglik=h2["loglik"])
    # --- 三态对照（RF-10 (c)：成像侧 BIC 选 K=3，模型侧必须并报）---
    ks = {}
    for K in (1, 2, 3, 4):
        ks[K] = hmm_kstate(z, K, n_iter=60, seed=0)
    out["BIC_by_K"] = {K: dict(bic=float(v["bic"]), loglik=float(v["loglik"]),
                               n_par=int(v["n_par"])) for K, v in ks.items()}
    _ssg = sample_size_gate(nb)
    out["sample_size_gate"] = _ssg
    if _ssg["bic_ok"]:
        out["best_K_by_BIC"] = int(min(ks, key=lambda k: ks[k]["bic"]))
    else:
        out["best_K_by_BIC"] = "unavailable"
        out["BIC_note"] = (f"n_frames={nb} < {MIN_FRAMES_BIC} ⇒ **K 选择结论标 unavailable**"
                           "（BIC 在小样本上无意义，RF-18(4)）")
    _h = out.get("H_K2", {})
    out["internal_consistency"] = _consistency(
        _h.get("transition_rate_hz"),
        ((_h.get("dwell", {}) or {}).get("low", {}) or {}).get("median_s"))
    # ================== **RF-14 主判据**：AR(1) 连续慢时间尺度 ==================
    _, _z = frame_series(pop_counts_steps, dt_ms)
    out["AR1"] = ar1_fit(_z, FRAME_S, folds=4, lb_lags=10)
    # **RF-24(2)**：无论成败，AR1 子字典必须完整落盘（M9 B3 同族：判据失败必须给出原因）
    out["AR1"].setdefault("status", "invalid")
    out["AR1"]["n_frames"] = int(nb)
    out["AR1"]["pipeline"] = ("0.914 s 箱 → 5 帧滑动平均 → z 分数 → AR(1)"
                              f"（平滑后 n={max(0, nb - (SMOOTH_FRAMES - 1))} 帧；未平滑 n={nb} 帧）")
    out["AR1"]["phi_smoothed"], out["AR1"]["phi_unsmoothed"] = _ar1_phi_both(_z, frames if nb else None)
    out["AR1"]["reference_tau_imaging_s"] = TAU_IMAGING_AR1_S
    out["AR1"]["tau_window_s"] = list(TAU_WINDOW_S)
    out["AR1"]["reference_source"] = TAU_REF_SOURCE
    out["AR1"]["tau_ratio_vs_imaging"] = (out["AR1"].get("tau_s") / TAU_IMAGING_AR1_S
                                          if out["AR1"].get("status") == "measured" else None)
    if out["AR1"].get("status") == "measured":
        out["AR1"]["in_tau_window"] = bool(TAU_WINDOW_S[0] <= out["AR1"]["tau_s"] <= TAU_WINDOW_S[1])
    out["AR1"]["honest_limit"] = ("B4：Ljung-Box 在平滑/未平滑/AR(4) 三口径下**全部拒绝白噪**（p≈0）"
                                  "⇒ '单一慢时间尺度'仅为一阶近似，τ 只作**粗粒度**指标")
    out["AR1"]["min_frames_for_tau"] = TAU_MIN_FRAMES
    out["AR1"]["n_frames"] = int(nb)
    # **RF-25(3)**：dwell 触分辨率地板（≤1.5 帧）时该量无分辨力 ⇒ 显式登记（不得当"×N 倍差距"引用）
    _dwq = ((out.get("Q_method", {}) or {}).get("dwell", {}) or {}).get("low", {}) or {}
    _dwh = ((out.get("H_K2", {}) or {}).get("dwell", {}) or {}).get("low", {}) or {}
    out["dwell_resolution_floor"] = dict(
        frame_s=FRAME_S, q_median_s=_dwq.get("median_s"), h_median_s=_dwh.get("median_s"),
        at_floor=bool((_dwq.get("median_s") or 0) <= FRAME_S * 1.5),
        note="dwell 中位 ≤1.5 帧 ⇒ **触分辨率地板、无分辨力**：只能作为'未产生慢状态'的**旁证**，"
             "**不得**作为'×N 倍差距'引用（时间尺度应以 AR(1) τ 与转换率为准）")
    if nb < TAU_MIN_FRAMES:
        out["AR1"]["tau_criterion_status"] = "unavailable（n_frames<%d ⇒ 短记录禁用 τ 判据）" % TAU_MIN_FRAMES
    elif out["AR1"].get("status") != "measured":
        out["AR1"]["tau_criterion_status"] = "invalid（AR(1) 不可估）"
    else:
        out["AR1"]["tau_criterion_status"] = ("pass" if out["AR1"].get("in_tau_window")
                                              else "fail（窗外）")
    out["pipeline"] = ("0.914 s 箱 → 5 帧滑动平均 → z 分数 → AR(1)（与成像侧**同一管线**）")
    # --- Q 法（并列对照；分位法）---
    thr = float(np.quantile(frames, 0.5))
    qstate = (frames > thr).astype(np.int8)
    out["Q_method"] = dict(occupancy=[float((qstate == 0).mean()), float((qstate == 1).mean())],
                           transition_rate_hz=float((np.diff(qstate) != 0).sum() / (nb * FRAME_S)),
                           dwell=_dwells(qstate, FRAME_S))
    # --- 慢变量 ↔ 状态序列相关（机制归属；(d) 在 0.9138 s 箱上算，解决 D4）---
    if slow_trace is not None and slow_dt_s:
        st = np.asarray(slow_trace, float)
        sb = max(1, int(round(FRAME_S / slow_dt_s)))
        m = st.size // sb
        if m >= 5:
            sv = st[: m * sb].reshape(m, sb).mean(axis=1)
            n = min(sv.size, h2["state"].size)
            a = sv[:n] - sv[:n].mean(); bb = h2["state"][:n] - h2["state"][:n].mean()
            den = a.std() * bb.std()
            out["slow_vs_state_corr"] = float(a.dot(bb) / (n * den)) if den > 0 else None
            out["slow_vs_state_corr_n_frames"] = int(n)
    return out


def _consistency(trans_hz, dwell_med_s):
    """RF-18(2) 内部一致性：2 状态稳态近似 dwell ≈ 1/(2·转换率)。

    不一致 ⇒ invalid（A3/L29「同一量的两个独立估计器必须断言一致」）。
    不适用条件（须显式写出）：状态数>2 / 转移矩阵强非对称 / 占用严重不平衡。
    """
    if not trans_hz or not dwell_med_s or dwell_med_s <= 0:
        return dict(applicable=False, reason="dwell 或转换率为 0/缺失（样本不足）")
    pred = 1.0 / (2.0 * trans_hz)
    rel = abs(dwell_med_s - pred) / dwell_med_s
    return dict(applicable=True, predicted_dwell_s=pred, observed_dwell_s=dwell_med_s,
                rel_diff=rel, pass_=bool(rel <= 1.0),
                note="2 状态稳态近似；若状态数>2 或转移矩阵强非对称则本条不适用（须显式声明）")


def sample_size_gate(n_frames: int):
    """RF-18(1)：样本量硬门（每行必须报 n_frames）。"""
    return dict(n_frames=int(n_frames),
                valid=bool(n_frames >= MIN_FRAMES_VALID),
                shape_ok=bool(n_frames >= MIN_FRAMES_SHAPE),
                bic_ok=bool(n_frames >= MIN_FRAMES_BIC),
                thresholds=dict(valid=MIN_FRAMES_VALID, shape=MIN_FRAMES_SHAPE, bic=MIN_FRAMES_BIC),
                note="n_frames<65 ⇒ invalid（最短协议 T≥60 s）；形状判据需 ≥130（T≥120 s）；BIC 需 ≥500")


def gate3(model: dict, imaging: dict | None):
    """RF-10 (a)：新版第三条自检门。"""
    h = model.get("H_K2", {})
    occ = h.get("occupancy", [0.5, 0.5])
    dw = (h.get("dwell", {}) or {}).get("low", {})
    c1 = bool(0.05 < occ[0] < 0.95)
    c2 = bool(h.get("transition_rate_hz", 0) > 0)
    p90m = dw.get("p90_over_median")
    c3 = bool(p90m is not None and p90m >= 2.0 and (dw.get("ks_vs_exp", {}) or {}).get("pass_"))
    # RF-21(3)：c4 **不得用零滞后相关**（共同趋势/跟随者伪相关）⇒ 去趋势 + lead-lag
    r_raw = model.get("slow_vs_state_corr")                       # 仅描述性
    r_dt = model.get("slow_vs_state_corr_detrended")
    lag = model.get("slow_vs_state_lead_lag_best_lag")
    corr_ll = model.get("slow_vs_state_lead_lag_best_corr")
    if r_dt is None or lag is None:
        c4 = None                                                  # unavailable（未测/无慢变量轨迹）
        c4_note = "unavailable（无慢变量轨迹或未做去趋势/lead-lag）"
    elif abs(r_dt) >= 0.30 and lag > 0:
        c4 = True
        c4_note = "pass（去趋势 |r|≥0.30 且峰值 lag>0 ⇒ 机制领先）"
    else:
        c4 = False
        c4_note = ("follower/共同趋势（零滞后 |r|=%.3f 但去趋势 |r|=%.3f、峰值 lag=%s ≤0）"
                   "⇒ **不进 pass**" % (abs(r_raw) if r_raw is not None else float('nan'),
                                        abs(r_dt), lag))
    _ss = model.get("sample_size_gate", {}) or {}
    _cons = model.get("internal_consistency", {}) or {}
    if not _ss.get("shape_ok", False):
        return dict(status="invalid",
                    reason=f"样本量不足（n_frames={_ss.get('n_frames')} < 形状判据下限 "
                           f"{MIN_FRAMES_SHAPE}）⇒ **状态结构数字不是物理量，是采样窗长的产物**；"
                           "既不计 pass 也不计 fail（RF-18）",
                    sample_size_gate=_ss, internal_consistency=_cons,
                    gate3_state_structure=None)
    if _cons.get("applicable") and not _cons.get("pass_", False):
        return dict(status="invalid",
                    reason=f"内部一致性失败：dwell 中位 vs 1/(2·转换率) 相对差 {_cons.get('rel_diff'):.3f} > 1.0"
                           "（A3/L29：两估计器不可能同时正确）",
                    sample_size_gate=_ss, internal_consistency=_cons)
    rep = dict(status="measured",
        criterion="与成像同实现(H 法)+同分辨率(0.9138 s 箱)：①占用非退化 ②有转换 ③低态 dwell 重尾(p90/中位≥2)且拒绝指数(KS p<0.05) ④慢变量↔状态相关≥0.30",
        c1_occupancy=dict(value=occ, pass_=c1),
        c2_transitions=dict(value=h.get("transition_rate_hz"), pass_=c2),
        c3_heavy_tail=dict(p90_over_median=p90m,
                           ks_vs_exp=(dw.get("ks_vs_exp", {}) or {}).get("p"), pass_=c3),
        c4_mechanism_attribution=dict(
            r_zerolag_DESCRIPTIVE_ONLY=r_raw, r_detrended=r_dt,
            lead_lag_best_lag=lag, lead_lag_best_corr=corr_ll, pass_=c4, note=c4_note),
        pass_=bool(c1 and c2 and c3 and c4 is True),
        note="**绝对 dwell 值不作必过硬门**（口径差 3.9–13×、跨条件 26×）；落窗内不足以证明机制正确",
    )
    if imaging:
        rep["model_vs_imaging_dwell_low_median_s"] = dict(
            model=dw.get("median_s"), imaging=imaging.get("dwell_low_median_s_H"),
            ratio=(dw.get("median_s") / imaging["dwell_low_median_s_H"]
                   if imaging.get("dwell_low_median_s_H") else None))
        rep["descriptive_window"] = dict(window_s=list(DWELL_WINDOW),
                                         note="B4 等长控制 max/min 定；极宽 ⇒ 仅描述性")
    return rep


def load_imaging_reference():
    """成像侧参照：**必须同时给出 AR(1) τ（主判据）与 dwell（次级）**（RF-21(1)）。"""
    tau_ref = dict(tau_s=TAU_IMAGING_AR1_S, window_s=list(TAU_WINDOW_S),
                   source=TAU_REF_SOURCE, min_frames_for_tau=TAU_MIN_FRAMES,
                   honest_limit=("B4：Ljung-Box 在平滑/未平滑/AR(4) 三口径下**全部拒绝白噪**（p≈0）"
                                 "⇒ AR(1) 仅一阶近似；τ 只作**粗粒度**指标，不得用精确值做精细判定"))
    if not os.path.exists(IMAGING_REF_JSON):
        return dict(ar1_tau=tau_ref)
    d = json.load(open(IMAGING_REF_JSON))
    out = {}
    try:
        out["dwell_low_median_s_H"] = float(d["D"]["median_s"])
        out["dwell_low_median_s_Q"] = float(d["A"]["Q"]["low"]["median_s"])
        out["transition_rate_hz_H"] = None
        out["occupancy_H"] = [float(d["D"].get("occ_low", 0.0)), float(d["D"].get("occ_high", 0.0))]
    except Exception:
        return None
    for k in ("occupancy", "transition_rate_hz"):
        if k in d:
            out[k] = d[k]
    out["source"] = os.path.basename(IMAGING_REF_JSON)
    out["ar1_tau"] = tau_ref            # ← 主判据参照（RF-21(1) 接线）
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop-trace", default=None, help="逐步群体脉冲计数 .npy（优先；避免重跑仿真）")
    ap.add_argument("--dt-ms", type=float, default=0.05)
    ap.add_argument("--out-json", default=os.path.join(DATA_DIR, "m10_p6_state_structure.json"))
    ap.add_argument("--out-csv", default=os.path.join(DATA_DIR, "m10_p6_state_structure.csv"))
    args = ap.parse_args(argv)
    t0 = time.time()
    if not args.pop_trace or not os.path.exists(args.pop_trace):
        print("[state] 需要 --pop-trace（逐步群体脉冲计数）；本工具只做分析，不做仿真", flush=True)
        return 2
    pop = np.load(args.pop_trace)
    img = load_imaging_reference()
    res = analyse(pop, args.dt_ms, label=os.path.basename(args.pop_trace))
    res["gate3_state_structure"] = gate3(res, img)
    res["imaging_reference"] = img
    with open(args.out_json, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False, default=str)
    with open(args.out_csv, "w", newline="") as f:
        f.write("# M10 RF-10 状态结构判据（H 法同实现 + 0.9138 s 箱；Q 法并列；K=1..4 BIC）\n")
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["key", "value"])
        w.writerow(["n_frames", res["n_frames"]])
        w.writerow(["pop_rate_hz", round(res["pop_rate_hz"], 6)])
        w.writerow(["fano", round(res["fano"], 4)])
        for k in ("H_K2", "Q_method", "BIC_by_K", "best_K_by_BIC", "slow_vs_state_corr",
                  "gate3_state_structure"):
            w.writerow([k, json.dumps(res.get(k), ensure_ascii=False, default=str)])
    print(json.dumps({"n_frames": res["n_frames"], "pop_rate_hz": round(res["pop_rate_hz"], 4),
                      "H_K2_occ": res["H_K2"]["occupancy"],
                      "H_K2_trans": round(res["H_K2"]["transition_rate_hz"], 5),
                      "H_K2_dwell_low_median": round(res["H_K2"]["dwell"]["low"]["median_s"], 3),
                      "H_K2_p90_over_median": res["H_K2"]["dwell"]["low"]["p90_over_median"],
                      "best_K_by_BIC": res["best_K_by_BIC"],
                      "gate3": res["gate3_state_structure"]["pass_"],
                      "elapsed_s": round(time.time() - t0, 1)}, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
