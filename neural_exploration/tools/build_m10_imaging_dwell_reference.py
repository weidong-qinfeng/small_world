"""M10-B4：成像侧（ZAPBench）慢状态 / 停留时长（dwell）基准的**同口径复核**与**口径敏感性量化**。

任务（主 agent 令）：模型侧（点神经元网络，0.01 s 箱）脚本 `tools/validate_p10_slow_variable.py`
用「该记录自身 50% 分位的二值化」（**Q 法**）算 dwell/转换率；成像侧 `data/m10_imaging_reference.csv`
③ 的 26.5 s 是用**高斯 2 状态 HMM**（**H 法**）算的。两者跨口径直接比较是方法学错误。
本工具在**同一份** ZAPBench 群体活动序列上把两法各算一遍，把「口径差」与「真实差」分离，
并给出箱宽 / 状态数 / 子采样的敏感性。

不修改任何既有文件；只读 `data/m10_raw/*`、`data/m10_imaging_reference.csv`、
`data/m10_imaging_stats.json`，并**复用**（import，不复制）参考实现里的
`event_rate` / `gaussian_hmm_2state` / `dwell_stats`，保证「同口径」是代码级同一份。

输出：
  data/m10_imaging_dwell_reference.csv   （长表：key,value,unit,method,scope,source,note）
  data/m10_imaging_dwell_reference.json  （结构化全量结果）
  reports/neuro/m10_b4_dwell_reference.png（可选图）
  reports/neuro/_m10_b4_progress.log     （增量进度日志，含 [DONE] 完成标记）

分析项：
  A 同口径 2×2 矩阵（Q 法 vs H 法，同一序列）        → 「口径差」标定量
  B 箱宽敏感性（1/2/4/8/16 帧一箱）                  → 箱宽 → dwell 定量映射
  C 状态数模型选择（1/2/3 状态高斯 HMM，BIC + 前向链式 CV 留出似然）
  D dwell 分布形态（几何/指数检验、KS、对数生存直线性）
  E 跨条件 / 段内稳定性离散度
  F 神经元子采样衰减（8192→4096/2048/1024）

用法：
    PYTHONHASHSEED=0 MPLBACKEND=Agg ./.venv-m10/bin/python -m \
        neural_exploration.tools.build_m10_imaging_dwell_reference --help
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd

from .build_m10_imaging_reference import (  # 只读复用：保证「同口径」是同一份代码
    ZAP,
    dwell_stats as ref_dwell_stats,
    event_rate,
    gaussian_hmm_2state,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW = os.path.join(DATA_DIR, "m10_raw", "zapbench_traces_8192.f32.npy")
REF_CSV = os.path.join(DATA_DIR, "m10_imaging_reference.csv")
REF_JSON = os.path.join(DATA_DIR, "m10_imaging_stats.json")
OUT_CSV = os.path.join(DATA_DIR, "m10_imaging_dwell_reference.csv")
OUT_JSON = os.path.join(DATA_DIR, "m10_imaging_dwell_reference.json")
OUT_FIG = os.path.join(PROJECT_ROOT, "reports", "neuro", "m10_b4_dwell_reference.png")
LOG_PATH = os.path.join(PROJECT_ROOT, "reports", "neuro", "_m10_b4_progress.log")

DT_S = 1.0 / 1.0943055555555556                     # 0.9138215509582435 s（帧率来源见 reference.csv）
SEED = 0
SMOOTH_BINS_REF = 5                                  # 参考实现的 k_win=5（≈4.57 s）
BIN_WIDTHS = (1, 2, 4, 8, 16)
SUB_N = (8192, 4096, 2048, 1024)
SUB_SEEDS = (0, 1, 2, 3, 4)
# 参考值（data/m10_imaging_reference.csv 已定稿；本工具做同口径复核，不事后调）
REF_VALUES = dict(dwell_median_s=26.500824977789062, transition_rate_hz=0.019583333333333335,
                  occupancy_low=0.6719126792740195, occupancy_high=0.32808732072598046)

LOG_LINES: list[str] = []
T0 = time.time()


def log(msg: str) -> None:
    """增量进度日志：既 print（便于前台核对）也追加落盘（L33/L35：长任务必须能被看到在推进）。"""
    line = f"[{time.time() - T0:8.1f}s] {msg}"
    print(line, flush=True)
    LOG_LINES.append(line)
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


# ==========================================================================
# 通用工具
# ==========================================================================
def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def _stationary(A: np.ndarray) -> np.ndarray:
    """转移矩阵的平稳分布（左特征向量；退化时回退均匀分布）。"""
    w, v = np.linalg.eig(A.T)
    i = int(np.argmin(np.abs(w - 1.0)))
    pi = np.real(v[:, i])
    pi = np.clip(pi, 0, None)
    s = pi.sum()
    return pi / s if s > 0 else np.full(A.shape[0], 1.0 / A.shape[0])


def gaussian_hmm_k_state(x: np.ndarray, n_state: int, n_iter: int = 60, seed: int = 0,
                         init: str = "quantile", n_restart: int = 1):
    """K 状态一维高斯 HMM 的 EM（对数域前向后向）。

    **与参考实现同口径的关键**：K=2 且 init='quantile'、seed=0、n_iter=60 时的初值与
    收敛判据与 `build_m10_imaging_reference.gaussian_hmm_2state` 一致
    （mu=quantile[0.33,0.67]、sd=[σ, 1.5σ]、A 对角 0.99、π 均匀、|Δll|<1e-6 早停），
    脚本内以断言核对两者参数一致 → K=3 的 BIC 才是可比口径的扩展。
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    T = x.shape[0]
    if n_state == 1:
        mu = float(x.mean())
        sd = float(x.std())
        ll = float(np.sum(-0.5 * ((x - mu) ** 2 / sd ** 2 + np.log(2 * np.pi * sd ** 2))))
        return dict(n_state=1, mu=[mu], sigma=[sd], A=[[1.0]], pi=[1.0],
                    loglik=ll, loglik_final=ll, n_iter_used=0, restarts=1,
                    state=np.zeros(T, dtype=np.int8), occupancy=[1.0])

    std_all = float(x.std())
    cands: list[dict] = []
    for r in range(max(1, n_restart)):
        rng = np.random.default_rng(seed + 1000 * r)
        if r == 0 and init == "quantile":
            qs = np.linspace(0.0, 1.0, n_state + 2)[1:-1]
            mu = np.quantile(x, qs).astype(np.float64)
            sd = np.array([std_all * (1.0 + 0.5 * i) for i in range(n_state)]) + 1e-6
            A = np.full((n_state, n_state), 0.01 / (n_state - 1))
            np.fill_diagonal(A, 0.99)
            pi = np.full(n_state, 1.0 / n_state)
        else:
            qs = np.linspace(0.0, 1.0, n_state + 2)[1:-1]
            mu = np.quantile(x, qs) + rng.normal(0, 0.2 * std_all, n_state)
            sd = np.abs(rng.normal(std_all * 0.6, 0.2 * std_all, n_state)) + 1e-3
            A = np.full((n_state, n_state), 0.02 / (n_state - 1))
            np.fill_diagonal(A, 0.98)
            pi = np.full(n_state, 1.0 / n_state)
        mu = np.sort(mu)  # 状态有序化（低→高），便于跨重启/跨 K 比较
        ll_old = -np.inf
        ll = -np.inf
        it_used = 0
        for it in range(n_iter):
            it_used = it + 1
            lb = -0.5 * ((x[:, None] - mu[None, :]) ** 2 / sd[None, :] ** 2
                         + np.log(2 * np.pi * sd[None, :] ** 2))          # (T,K)
            logA = np.log(A + 1e-300)
            logpi = np.log(pi + 1e-300)
            la = np.empty((T, n_state))
            la[0] = logpi + lb[0]
            for t in range(1, T):
                la[t] = _logsumexp(la[t - 1][:, None] + logA, axis=0) + lb[t]
            ll = float(_logsumexp(la[-1], axis=0))
            lbk = np.zeros((T, n_state))
            for t in range(T - 2, -1, -1):
                lbk[t] = _logsumexp(logA + (lb[t + 1] + lbk[t + 1])[None, :], axis=1)
            lg = la + lbk
            g = np.exp(lg - _logsumexp(lg, axis=1)[:, None])                 # gamma (T,K)
            w = g.sum(axis=0)
            mu = (g * x[:, None]).sum(axis=0) / w
            sd = np.sqrt((g * (x[:, None] - mu[None, :]) ** 2).sum(axis=0) / w) + 1e-9
            # xi 之和（对 t 向量化：形状 (T-1,K,K)，K<=3、T<=7879 → 内存 <1 MB）
            post = lb[1:] + lbk[1:]
            m = la[:-1][:, :, None] + logA[None, :, :] + post[:, None, :]
            xi = np.exp(m - _logsumexp(m.reshape(T - 1, -1), axis=1)[:, None, None]).sum(axis=0)
            A = xi / np.maximum(xi.sum(axis=1, keepdims=True), 1e-300)
            A = np.clip(A, 1e-6, 1 - 1e-6)
            A = A / A.sum(axis=1, keepdims=True)
            pi = g[0] / g[0].sum()
            if abs(ll - ll_old) < 1e-6:
                break
            ll_old = ll
        # 收敛判据停在 "参数更新前" 的那一步 → 用**最终参数**重算一次 loglik（BIC 需要同参数口径）
        lb = -0.5 * ((x[:, None] - mu[None, :]) ** 2 / sd[None, :] ** 2
                     + np.log(2 * np.pi * sd[None, :] ** 2))
        logA = np.log(A + 1e-300)
        la2 = np.empty((T, n_state))
        la2[0] = np.log(pi + 1e-300) + lb[0]
        for t in range(1, T):
            la2[t] = _logsumexp(la2[t - 1][:, None] + logA, axis=0) + lb[t]
        ll_final = float(_logsumexp(la2[-1], axis=0))
        cand = dict(n_state=n_state, mu=[float(v) for v in mu], sigma=[float(v) for v in sd],
                    A=[[float(v) for v in row] for row in A],
                    pi=[float(v) for v in pi], loglik=float(ll), loglik_final=ll_final,
                    n_iter_used=int(it_used), restarts=max(1, n_restart),
                    min_sigma=float(np.min(sd)),
                    degenerate=bool(np.min(sd) < 1e-3 * std_all))
        cand["state"] = g.argmax(axis=1).astype(np.int8)
        cands.append(cand)
    # 退化守卫：高斯混合/HMM 的自由方差使似然可发散（σ→0）；退化解不参与最优选择
    ok = [c for c in cands if not c["degenerate"]]
    n_degenerate = len(cands) - len(ok)
    pool = ok if ok else cands
    best = max(pool, key=lambda c: c["loglik_final"])
    best_state = best.pop("state")
    best["n_degenerate_restarts"] = int(n_degenerate)
    best["restart_logliks"] = [round(float(c["loglik_final"]), 4) for c in cands]
    best["note_degenerate"] = ("存在退化解（σ<1e-3·σ_data，似然发散）已剔除" if n_degenerate else "")
    order = np.argsort(best["mu"])
    best["mu"] = [best["mu"][i] for i in order]
    best["sigma"] = [best["sigma"][i] for i in order]
    best["A"] = [[best["A"][i][j] for j in order] for i in order]
    best["state"] = np.asarray([int(np.flatnonzero(order == s)[0]) for s in best_state],
                               dtype=np.int8)
    best["occupancy"] = [float((best["state"] == s).mean()) for s in range(n_state)]
    return best


def hmm_bic(loglik_final: float, n_state: int, T: int) -> float:
    """BIC = -2·ll + p·lnT；p = K 均值 + K 方差 + K(K-1) 转移 + (K-1) 初始 = K²+2K-1。"""
    p = n_state * n_state + 2 * n_state - 1
    return float(-2.0 * loglik_final + p * math.log(T))


def heldout_loglik(params: dict, x_test: np.ndarray) -> float:
    """用已拟合参数对留出段的预测对数似然（初始分布取 A 的平稳分布）。

    注：前向算法的 loglik 恰好 = Σ_t log p(x_t | x_{<t})（逐步预测似然），
    因此与 AR(p) 的一步预测似然是**同一个量**，可直接比较（同一训练/测试切分）。
    """
    x = np.asarray(x_test, dtype=np.float64).reshape(-1)
    mu = np.asarray(params["mu"], dtype=np.float64)
    sd = np.asarray(params["sigma"], dtype=np.float64)
    if mu.size == 1:
        return float(np.sum(-0.5 * ((x - mu[0]) ** 2 / sd[0] ** 2 + np.log(2 * np.pi * sd[0] ** 2))))
    A = np.asarray(params["A"], dtype=np.float64)
    logA = np.log(A + 1e-300)
    lb = -0.5 * ((x[:, None] - mu[None, :]) ** 2 / sd[None, :] ** 2
                 + np.log(2 * np.pi * sd[None, :] ** 2))
    la = np.empty_like(lb)
    la[0] = np.log(_stationary(A) + 1e-300) + lb[0]
    for t in range(1, x.size):
        la[t] = _logsumexp(la[t - 1][:, None] + logA, axis=0) + lb[t]
    return float(_logsumexp(la[-1], axis=0))


def ar_predictive_ll(x_train: np.ndarray, x_test: np.ndarray, p: int) -> dict:
    """AR(p) 连续隐变量基线的一步预测留出似然（与 HMM 的 forward loglik 同量、可直接比较）。

    目的：回答「ZAPBench 群体活动真的需要**离散状态**吗」——若连续 AR 以更少参数达到
    同等/更好的留出似然，则 2/3 状态 HMM 的「状态」只是连续慢过程的人为离散化。
    """
    xt = np.asarray(x_train, dtype=np.float64).reshape(-1)
    xe = np.asarray(x_test, dtype=np.float64).reshape(-1)
    def _design(v):
        # 设计矩阵 [1, x_{t-1}, …, x_{t-p}]，对齐到 y_t = x_t（t = p … len(v)-1）
        lag = [v[p - j: v.size - j] for j in range(1, p + 1)]
        return np.column_stack([np.ones(v.size - p)] + lag)
    Z = _design(xt)
    coef, *_ = np.linalg.lstsq(Z, xt[p:], rcond=None)
    s2 = float(np.mean((xt[p:] - Z @ coef) ** 2))
    hist = np.concatenate([xt[-p:], xe])
    Zt = _design(hist)
    resid = hist[p:] - Zt @ coef
    ll = float(np.sum(-0.5 * (resid ** 2 / s2 + np.log(2 * np.pi * s2))))
    n_param = p + 1 + 1  # p 系数 + 截距 + 噪声方差
    return dict(p=int(p), n_params=n_param, coef=[float(v) for v in coef], sigma=float(math.sqrt(s2)),
                test_loglik=ll, test_loglik_per_frame=ll / xe.size,
                BIC_test=float(-2.0 * ll + n_param * math.log(xe.size)),
                ar1_phi=float(coef[1]) if p >= 1 else None,
                ar1_tau_s=float(-DT_S / math.log(abs(coef[1]))) if p >= 1 and 0 < abs(coef[1]) < 1 else None,
                train_loglik=float(np.sum(-0.5 * ((xt[p:] - Z @ coef) ** 2 / s2 + np.log(2 * np.pi * s2)))))


# --------------------------------------------------------------------------
# 群体序列构造 / 两法
# --------------------------------------------------------------------------
def bin_sum(pop: np.ndarray, w: int) -> np.ndarray:
    """非重叠 w 帧求和（尾部落零头丢弃）。→ (T//w,)"""
    if w == 1:
        return pop.astype(np.float64, copy=True)
    n = pop.size // w
    return pop[: n * w].reshape(n, w).sum(axis=1).astype(np.float64)


def smooth_same(x: np.ndarray, k: int) -> np.ndarray:
    """与参考实现完全一致的平滑（np.convolve mode='same'，零填充边界）。"""
    if k <= 1:
        return np.asarray(x, dtype=np.float64)
    return np.convolve(np.asarray(x, dtype=np.float64), np.ones(k) / k, mode="same")


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / (x.std() + 1e-12)


def build_series(pop: np.ndarray, w: int = 1, smooth_bins: int = SMOOTH_BINS_REF) -> np.ndarray:
    """参考口径的群体活动序列：非重叠聚合 → 平滑 → z 分数。"""
    return zscore(smooth_same(bin_sum(pop, w), smooth_bins))


def _runs(state: np.ndarray, s: int):
    b = (state == s).astype(np.int8)
    d = np.diff(np.r_[0, b, 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def dwell_summary(state: np.ndarray, dt_s: float) -> dict:
    """停留时长统计：**同时**给「含首尾截断段（参考 dwell_stats 口径）」与「仅完整段」两套。

    参考 `dwell_stats` 的实现（np.r_[0,b,0]）会把记录首/尾被截断的 run 也计入，
    其 docstring 声称去掉 —— 本工具不修改它，而是把两种口径都落盘（口径敏感性的一部分）。
    """
    state = np.asarray(state)
    n = state.size
    out: dict = {}
    for s, nm in ((0, "low"), (1, "high")):
        st, en = _runs(state, s)
        dur_f = (en - st).astype(np.float64)
        interior = (st > 0) & (en < n)
        dfi = dur_f[interior]
        out[nm] = dict(
            n_bouts=int(dur_f.size), n_bouts_interior=int(interior.sum()),
            median_s=float(np.median(dur_f) * dt_s) if dur_f.size else 0.0,
            mean_s=float(dur_f.mean() * dt_s) if dur_f.size else 0.0,
            p90_s=float(np.quantile(dur_f, 0.9) * dt_s) if dur_f.size else 0.0,
            max_s=float(dur_f.max() * dt_s) if dur_f.size else 0.0,
            mean_frames=float(dur_f.mean()) if dur_f.size else 0.0,
            median_s_interior=float(np.median(dfi) * dt_s) if dfi.size else 0.0,
            p90_s_interior=float(np.quantile(dfi, 0.9) * dt_s) if dfi.size else 0.0,
            mean_s_interior=float(dfi.mean() * dt_s) if dfi.size else 0.0,
        )
    sw = int((np.diff(state) != 0).sum())
    out["n_switches"] = sw
    out["transition_rate_hz"] = float(sw / (state.size * dt_s))
    out["n_frames"] = int(state.size)
    return out


def dwell_per_state(state: np.ndarray, n_state: int, dt_s: float) -> list[dict]:
    """任意状态数下的逐状态停留时长（用于 C 项 K=3 的「低态」到底是谁）。"""
    state = np.asarray(state)
    out = []
    for s in range(n_state):
        st, en = _runs(state, s)
        dur_f = (en - st).astype(np.float64)
        out.append(dict(state=int(s), occupancy=float((state == s).mean()),
                        n_bouts=int(dur_f.size),
                        median_s=float(np.median(dur_f) * dt_s) if dur_f.size else 0.0,
                        mean_s=float(dur_f.mean() * dt_s) if dur_f.size else 0.0,
                        p90_s=float(np.quantile(dur_f, 0.9) * dt_s) if dur_f.size else 0.0,
                        max_s=float(dur_f.max() * dt_s) if dur_f.size else 0.0))
    return out


def q_method(series: np.ndarray, dt_s: float) -> dict:
    """Q 法：状态 = 序列 > 该序列自身 50% 分位（与 validate_p10_slow_variable._hmm_states 同口径）。"""
    r = np.asarray(series, dtype=np.float64)
    thr = float(np.quantile(r, 0.5))
    state = (r > thr).astype(np.int8)
    d = dwell_summary(state, dt_s)
    d["threshold"] = thr
    d["occupancy"] = [float((state == 0).mean()), float((state == 1).mean())]
    d["method"] = "Q(median binary threshold)"
    d["mu"] = [float(r[state == 0].mean()), float(r[state == 1].mean())]
    return d


def h_method(series: np.ndarray, dt_s: float, n_state: int = 2) -> dict:
    """H 法：高斯 K 状态 HMM（K=2 时 = 参考实现 gaussian_hmm_2state，seed=0、n_iter=60）。"""
    if n_state == 2:
        h = gaussian_hmm_2state(np.asarray(series, dtype=np.float64), n_iter=60, seed=SEED)
        state = h["state"]
        out = dict(mu=h["mu"], sigma=h["sigma"], A=h["A"],
                   loglik=float(h["loglik"]), ref_internal_loglik=float(h["loglik"]))
    else:
        h = gaussian_hmm_k_state(series, n_state, n_iter=60, seed=SEED)
        state = h["state"]
        out = dict(mu=h["mu"], sigma=h["sigma"], A=h["A"],
                   loglik=float(h["loglik_final"]), ref_internal_loglik=float(h["loglik"]))
    d = dwell_summary(state, dt_s)
    d.update(out)
    d["occupancy"] = [float((state == s).mean()) for s in range(n_state)]
    d["method"] = f"H(K={n_state} Gaussian HMM)"
    d["state"] = state
    d["theoretical_mean_dwell_frames"] = [1.0 / (1.0 - row[s]) for s, row in enumerate(out["A"])]
    d["self_transition_prob"] = [float(row[s]) for s, row in enumerate(out["A"])]
    return d


# --------------------------------------------------------------------------
# 统计检验（无 scipy，手写）
# --------------------------------------------------------------------------
def kolmogorov_q(lam: float) -> float:
    if not np.isfinite(lam) or lam <= 0:
        return 1.0
    if lam > 20:
        return 0.0
    k = np.arange(1, 101)
    return float(min(1.0, max(0.0, 2.0 * np.sum((-1.0) ** (k - 1) * np.exp(-2.0 * k * k * lam * lam)))))


def ks_test(dur: np.ndarray, cdf_fn) -> dict:
    """单样本 KS（手写； asymptotic Kolmogorov 分布 p 值）。"""
    d = np.sort(np.asarray(dur, dtype=np.float64))
    n = d.size
    if n < 5:
        return dict(n=int(n), D=None, p=None, note="n<5 未检验")
    cdf = np.clip(cdf_fn(d), 0.0, 1.0)
    i = np.arange(1, n + 1)
    D = float(max(np.max(np.abs(cdf - (i - 1) / n)), np.max(np.abs(cdf - i / n))))
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * D
    return dict(n=int(n), D=D, lambda_=float(lam), p=kolmogorov_q(lam))


def survival_logfit(dur_f: np.ndarray, min_surv: float = 0.05) -> dict:
    """对数生存曲线直线性：log S(d) ≈ -λ d 的 OLS R²（HMM 的几何/指数假设的直接检验）。"""
    d = np.asarray(dur_f, dtype=np.float64)
    if d.size < 10:
        return dict(n=int(d.size), note="n<10 未拟合")
    ks = np.arange(1, int(d.max()) + 1)
    S = np.array([(d >= k).mean() for k in ks])
    m = S >= min_surv
    if m.sum() < 5:
        return dict(n=int(d.size), note="可用点不足")
    x, y = ks[m], np.log(S[m])
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    fit = A @ coef
    ss_res = float(np.sum((y - fit) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return dict(n=int(d.size), n_points=int(m.sum()), slope_per_frame=float(coef[0]),
                implied_mean_frames=float(-1.0 / coef[0]) if coef[0] < 0 else None,
                r2=float(1 - ss_res / ss_tot) if ss_tot > 0 else None,
                min_survival_used=float(min_surv))


# --------------------------------------------------------------------------
# 记录行
# --------------------------------------------------------------------------
ROWS: list[dict] = []


def add(key, value, unit="", method="", scope="", source="", note=""):
    ROWS.append(dict(key=key,
                     value=(float(value) if isinstance(value, (int, float, np.floating, np.integer))
                            else value),
                     unit=unit, method=method, scope=scope, source=source, note=note))


def emit_dwell(prefix: str, d: dict, method: str, scope: str, source: str, note: str = ""):
    """把一套 dwell 结果摊平成 CSV 行。"""
    add(f"{prefix}_occupancy_low", d["occupancy"][0], "frac", method, scope, source, note)
    add(f"{prefix}_occupancy_high", d["occupancy"][-1], "frac", method, scope, source, note)
    add(f"{prefix}_transition_rate", d["transition_rate_hz"], "Hz", method, scope, source, note)
    add(f"{prefix}_n_switches", d["n_switches"], "count", method, scope, source, note)
    for nm in ("low", "high"):
        s = d[nm]
        add(f"{prefix}_dwell_{nm}_n_bouts", s["n_bouts"], "bouts", method, scope, source, note)
        add(f"{prefix}_dwell_{nm}_median", s["median_s"], "s", method, scope, source, note)
        add(f"{prefix}_dwell_{nm}_mean", s["mean_s"], "s", method, scope, source, note)
        add(f"{prefix}_dwell_{nm}_p90", s["p90_s"], "s", method, scope, source, note)
        add(f"{prefix}_dwell_{nm}_median_interior", s["median_s_interior"], "s", method, scope,
            source, note + " | 仅完整 bout（去首尾截断）")
        add(f"{prefix}_dwell_{nm}_mean_interior", s["mean_s_interior"], "s", method, scope,
            source, note + " | 仅完整 bout（去首尾截断）")


# ==========================================================================
# 主流程
# ==========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(
        description="M10-B4 成像侧 dwell 基准同口径复核（Q 法 vs H 法；箱宽/状态数/子采样敏感性）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--raw", default=RAW, help="ZAPBench 子集 .npy（(T,N) float32）")
    ap.add_argument("--out-csv", default=OUT_CSV)
    ap.add_argument("--out-json", default=OUT_JSON)
    ap.add_argument("--out-fig", default=OUT_FIG)
    ap.add_argument("--stages", default="A,B,C,D,E,F",
                    help="要跑的分析项（逗号分隔）：A/B/C/D/E/F")
    ap.add_argument("--n-iter", type=int, default=60, help="HMM EM 迭代上限（=参考实现 60）")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--restarts", type=int, default=5, help="C 项 HMM 多起点重启次数（含参考初值）")
    ap.add_argument("--sub-seeds", type=int, default=len(SUB_SEEDS), help="F 项子采样重复次数")
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()
    stages = {s.strip().upper() for s in args.stages.split(",") if s.strip()}
    t_start = time.time()

    log(f"[START] M10-B4 dwell 参考复核；stages={sorted(stages)} seed={args.seed} "
        f"PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED', 'unset')}")

    # ---------- 0 数据与参考值核对 ----------
    x = np.load(args.raw)
    T, N = x.shape
    log(f"[data] {args.raw} shape={(T, N)} dtype={x.dtype} NaN={int(np.isnan(x).sum())} "
        f"frame_rate={1.0 / DT_S:.6f} Hz dt={DT_S:.6f} s")
    ref_stats = json.load(open(REF_JSON, encoding="utf-8"))
    ref_csv = pd.read_csv(REF_CSV, comment="#")
    ref_kv = dict(zip(ref_csv["key"].astype(str), ref_csv["value"].astype(str)))
    log(f"[data] 参考档 ③ occupancy={{low:{ref_kv.get('p7_state_occupancy_low')},"
        f"high:{ref_kv.get('p7_state_occupancy_high')}}} transition="
        f"{ref_kv.get('p7_state_transition_rate_hz')} Hz；"
        f"stats.json dwell.low.median_s={ref_stats['states']['dwell']['low']['median_s']:.6f}")

    # ---------- 群体事件率（与参考实现同一函数） ----------
    rate_hz, ev = event_rate(x, DT_S, k=3.0, refractory_frames=2)
    pop = ev.sum(axis=1).astype(np.float64)
    log(f"[events] 3·MAD 事件率中位 {np.median(rate_hz):.6f} Hz/neuron；"
        f"群体事件计数 均值 {pop.mean():.3f} var {pop.var():.1f} Fano {pop.var() / pop.mean():.1f}")

    # 参考口径序列（w=1, k_win=5）→ 必须复现参考值
    ser_ref = build_series(pop, w=1, smooth_bins=SMOOTH_BINS_REF)
    results: dict = dict(meta=dict(
        generated_by="tools/build_m10_imaging_dwell_reference.py",
        raw=os.path.relpath(args.raw, PROJECT_ROOT), shape=[int(T), int(N)],
        frame_rate_hz=1.0 / DT_S, dt_s=DT_S, nan=int(np.isnan(x).sum()),
        event_rate_median_hz=float(np.median(rate_hz)),
        pop_count_mean=float(pop.mean()), pop_count_fano=float(pop.var() / pop.mean()),
        seed=int(args.seed), restarts=int(args.restarts),
        condition_names=list(ZAP["condition_names"]),
        condition_offsets=list(ZAP["condition_offsets"]),
        n_neurons_in_file=int(N), n_neurons_full_dataset=int(ZAP["n_neurons"]),
        subset_is_contiguous_column_block=True,
        subset_note="fetch_m10_imaging.py 按 chunk 顺序取 c0=0 起的前 8192 列 ⇒ 本子集是原数组"
                    "**前 8192 列（连续列块，非全脑随机抽样）**，占 71721 神经元的 11.4%；"
                    "这使「全脑 71721 神经元口径的 dwell」不可得（F 项只覆盖子集内部的再抽样）",
        provenance="data/m10_raw/zapbench_provenance.json（单次记录 prefix=volumes/20240930/traces"
                   " → 无按鱼/按重复分组信息）",
        reference_values=REF_VALUES, stages=sorted(stages),
        note="Q 法 = 该序列 50% 分位二值化（validate_p10_slow_variable._hmm_states 同口径）；"
             "H 法 = 高斯 HMM（build_m10_imaging_reference.gaussian_hmm_2state，seed=0, n_iter=60）"))

    # ================= A 同口径 2×2 =================
    if "A" in stages:
        log("[A] 同一序列（w=1, k_win=5, z-score）上的 Q 法 vs H 法 ...")
        qA = q_method(ser_ref, DT_S)
        hA = h_method(ser_ref, DT_S, 2)
        log(f"[A] Q: occ_high={qA['occupancy'][1]:.4f} trans={qA['transition_rate_hz']:.5f} Hz "
            f"low_dwell med={qA['low']['median_s']:.3f}s p90={qA['low']['p90_s']:.3f}s "
            f"n={qA['low']['n_bouts']} | high med={qA['high']['median_s']:.3f}s")
        log(f"[A] H: occ={hA['occupancy'][0]:.4f}/{hA['occupancy'][1]:.4f} "
            f"trans={hA['transition_rate_hz']:.5f} Hz low_dwell med={hA['low']['median_s']:.3f}s "
            f"p90={hA['low']['p90_s']:.3f}s n={hA['low']['n_bouts']} | "
            f"high med={hA['high']['median_s']:.3f}s")
        # 参考实现一致性断言（同口径核对，不修改被引用文件）
        check = dict(
            median_s_absdiff=abs(hA["low"]["median_s"] - REF_VALUES["dwell_median_s"]),
            transition_rel=abs(hA["transition_rate_hz"] - REF_VALUES["transition_rate_hz"])
            / REF_VALUES["transition_rate_hz"],
            occ_low_absdiff=abs(hA["occupancy"][0] - REF_VALUES["occupancy_low"]))
        log(f"[A] 复现参考值：低态 dwell 中位 |Δ|={check['median_s_absdiff']:.2e} s、"
            f"转换率相对差 {check['transition_rel']:.2e}、占用 |Δ|={check['occ_low_absdiff']:.2e} → "
            f"{'PASS（H 法口径与参考完全一致）' if check['transition_rel'] < 1e-9 and check['median_s_absdiff'] < 1e-9 else 'FAIL'}")
        results["A"] = dict(
            config="同一 ZAPBench 群体事件率序列（8192 神经元、7879 帧、3·MAD 事件、"
                   "5 帧平滑、z 分数）",
            Q=qA, H=hA, reference_reproduction=check,
            ratio=dict(
                low_dwell_median_H_over_Q=float(hA["low"]["median_s"] / qA["low"]["median_s"]),
                low_dwell_median_interior_H_over_Q=float(
                    hA["low"]["median_s_interior"] / qA["low"]["median_s_interior"]),
                low_dwell_mean_H_over_Q=float(hA["low"]["mean_s"] / qA["low"]["mean_s"]),
                low_dwell_p90_H_over_Q=float(hA["low"]["p90_s"] / qA["low"]["p90_s"]),
                transition_Q_over_H=float(qA["transition_rate_hz"] / hA["transition_rate_hz"]),
                # 口径内最保守/最宽松的一对（H 含截断段 vs Q 仅完整段）→ 给出口径差的区间
                low_dwell_median_H_over_Q_worst_case=float(max(
                    hA["low"]["median_s"] / qA["low"]["median_s_interior"],
                    hA["low"]["median_s_interior"] / qA["low"]["median_s"])),
                low_dwell_median_H_over_Q_best_case=float(min(
                    hA["low"]["median_s"] / qA["low"]["median_s_interior"],
                    hA["low"]["median_s_interior"] / qA["low"]["median_s"])),
            ),
            note="Q 法占用恒为 0.5（分位定义），H 法占用由 EM 决定；转换率/停留的比值即「口径差」标定量，"
                 "与数据无关、纯方法学。")
        emit_dwell("a_ref_Q", qA, "Q", "全记录(7879帧)", "ZAPBench 8192 神经元子集",
                   "Q 法 = 50% 分位二值化（模型侧现用法）")
        emit_dwell("a_ref_H2", hA, "H(K=2)", "全记录(7879帧)", "ZAPBench 8192 神经元子集",
                   "H 法 = 参考实现 gaussian_hmm_2state(seed=0,n_iter=60)")
        for k, v in results["A"]["ratio"].items():
            add(f"a_ratio_{k}", v, "x", "Q vs H", "全记录(7879帧)",
                "本工具 A 项", "口径差标定量（同数据、同序列）")
    else:
        qA = hA = None

    # ================= B 箱宽敏感性 =================
    if "B" in stages:
        log("[B] 箱宽敏感性：1/2/4/8/16 帧一箱（0.914→14.62 s），两种平滑口径 ...")
        results["B"] = dict(bin_widths_frames=list(BIN_WIDTHS),
                            variants=["nosm（只做非重叠聚合，不额外平滑）",
                                      "sm46（额外平滑窗≈4.57 s，即 bins=max(1,round(5/w))）"],
                            table=[])
        for w in BIN_WIDTHS:
            for vname, k in (("nosm", 1), ("sm46", max(1, int(round(SMOOTH_BINS_REF / w))))):
                srs = build_series(pop, w=w, smooth_bins=k)
                qd = q_method(srs, DT_S * w)
                hd = h_method(srs, DT_S * w, 2)
                row = dict(bin_frames=int(w), bin_s=float(DT_S * w), variant=vname,
                           smooth_bins=int(k), n_bins=int(srs.size),
                           Q=dict(occupancy=qd["occupancy"], transition_rate_hz=qd["transition_rate_hz"],
                                  n_switches=qd["n_switches"], low=qd["low"], high=qd["high"]),
                           H=dict(occupancy=hd["occupancy"], transition_rate_hz=hd["transition_rate_hz"],
                                  n_switches=hd["n_switches"], low=hd["low"], high=hd["high"],
                                  mu=hd["mu"], sigma=hd["sigma"], A=hd["A"]),
                           ratio_low_dwell_median_H_over_Q=float(
                               hd["low"]["median_s"] / qd["low"]["median_s"]))
                results["B"]["table"].append(row)
                log(f"[B] w={w:2d}帧({DT_S * w:6.2f}s) {vname:5s} k={k} "
                    f"T'={srs.size:4d} | Q: trans={qd['transition_rate_hz']:.4f}Hz "
                    f"low_med={qd['low']['median_s']:7.3f}s | H: trans={hd['transition_rate_hz']:.4f}Hz "
                    f"low_med={hd['low']['median_s']:7.3f}s occ={hd['occupancy'][0]:.3f} | "
                    f"H/Q={row['ratio_low_dwell_median_H_over_Q']:.2f}")
                emit_dwell(f"b_w{w}_{vname}_Q", qd, "Q", f"w={w}帧({DT_S * w:.2f}s)/{vname}",
                           "ZAPBench 8192 神经元子集", "B 项箱宽敏感性")
                emit_dwell(f"b_w{w}_{vname}_H2", hd, "H(K=2)", f"w={w}帧({DT_S * w:.2f}s)/{vname}",
                           "ZAPBench 8192 神经元子集", "B 项箱宽敏感性")
                add(f"b_w{w}_{vname}_H_over_Q_low_dwell_median",
                    row["ratio_low_dwell_median_H_over_Q"], "x", "Q vs H",
                    f"w={w}帧({DT_S * w:.2f}s)/{vname}", "本工具 B 项", "口径差随箱宽")
        # 箱宽依赖强度：Q 与 H 的低态 dwell 中位 从 w=1 → w=16 的倍数
        for vname in ("nosm", "sm46"):
            rows = [r for r in results["B"]["table"] if r["variant"] == vname]
            q1 = rows[0]["Q"]["low"]["median_s"]
            q16 = rows[-1]["Q"]["low"]["median_s"]
            h1 = rows[0]["H"]["low"]["median_s"]
            h16 = rows[-1]["H"]["low"]["median_s"]
            log(f"[B] {vname}: Q 低态 dwell 中位 {q1:.3f}s→{q16:.3f}s（×{q16 / q1:.2f}）；"
                f"H {h1:.3f}s→{h16:.3f}s（×{h16 / h1:.2f}）")
            add(f"b_w16_over_w1_{vname}_Q_low_median", q16 / q1, "x", "Q", f"w16/w1/{vname}",
                "本工具 B 项", "Q 法 dwell 随箱宽的放大倍数")
            add(f"b_w16_over_w1_{vname}_H_low_median", h16 / h1, "x", "H(K=2)", f"w16/w1/{vname}",
                "本工具 B 项", "H 法 dwell 随箱宽的放大倍数")
        results["B"]["dependence"] = {
            v: dict(Q_w16_over_w1=float([r for r in results["B"]["table"]
                                         if r["variant"] == v][-1]["Q"]["low"]["median_s"] /
                                        [r for r in results["B"]["table"]
                                         if r["variant"] == v][0]["Q"]["low"]["median_s"]),
                    H_w16_over_w1=float([r for r in results["B"]["table"]
                                         if r["variant"] == v][-1]["H"]["low"]["median_s"] /
                                        [r for r in results["B"]["table"]
                                         if r["variant"] == v][0]["H"]["low"]["median_s"]))
            for v in ("nosm", "sm46")}

    # ================= C 状态数模型选择 =================
    if "C" in stages:
        log(f"[C] 1/2/3 状态高斯 HMM（BIC + 前向链式 CV 留出似然）；restarts={args.restarts} ...")
        C = dict(T=int(ser_ref.size), n_restart=int(args.restarts), fits=[], cv_folds=[])
        # 同口径核对：自实现 K 状态 EM 在 K=2 + 参考初值下必须复现参考实现参数
        _ref2 = gaussian_hmm_2state(ser_ref, n_iter=args.n_iter, seed=args.seed)
        _my2 = gaussian_hmm_k_state(ser_ref, 2, n_iter=args.n_iter, seed=args.seed, n_restart=1)
        C["k2_engine_check"] = dict(
            mu_max_absdiff=float(np.max(np.abs(np.array(_my2["mu"]) - np.array(_ref2["mu"])))),
            sigma_max_absdiff=float(np.max(np.abs(np.array(_my2["sigma"]) - np.array(_ref2["sigma"])))),
            A_max_absdiff=float(np.max(np.abs(np.asarray(_my2["A"]) - np.asarray(_ref2["A"])))),
            state_mismatch_frac=float(np.mean(_my2["state"].reshape(-1) != _ref2["state"])),
            note="自实现 K 状态 EM（K=2、参考初值、seed=0、n_iter=60）vs 参考 gaussian_hmm_2state")
        log(f"[C] 引擎核对(K=2 自实现 vs 参考)：Δmu={C['k2_engine_check']['mu_max_absdiff']:.2e} "
            f"Δσ={C['k2_engine_check']['sigma_max_absdiff']:.2e} "
            f"ΔA={C['k2_engine_check']['A_max_absdiff']:.2e} "
            f"状态不一致帧比例={C['k2_engine_check']['state_mismatch_frac']:.4f}")
        for K in (1, 2, 3):
            fit = gaussian_hmm_k_state(ser_ref, K, n_iter=args.n_iter, seed=args.seed,
                                       n_restart=(1 if K == 1 else args.restarts))
            bic = hmm_bic(fit["loglik_final"], K, ser_ref.size)
            C["fits"].append(dict(n_state=K, loglik_final=fit["loglik_final"],
                                  loglik_em_internal=fit["loglik"], BIC=bic,
                                  n_params=K * K + 2 * K - 1, mu=fit["mu"], sigma=fit["sigma"],
                                  A=fit["A"], occupancy=fit["occupancy"],
                                  n_iter_used=fit["n_iter_used"],
                                  min_sigma=fit.get("min_sigma"),
                                  n_degenerate_restarts=fit.get("n_degenerate_restarts"),
                                  restart_logliks=fit.get("restart_logliks")))
            log(f"[C] K={K} ll={fit['loglik_final']:.3f} p={K * K + 2 * K - 1} BIC={bic:.2f} "
                f"occ={[round(v, 3) for v in fit['occupancy']]} mu={[round(v, 3) for v in fit['mu']]} "
                f"σ={[round(v, 3) for v in fit['sigma']]} iters={fit['n_iter_used']} "
                f"退化重启={fit.get('n_degenerate_restarts')}")
        bics = {f["n_state"]: f["BIC"] for f in C["fits"]}
        C["best_by_bic"] = int(min(bics, key=bics.get))
        C["delta_bic_vs_best"] = {str(k): float(v - min(bics.values())) for k, v in bics.items()}
        log(f"[C] BIC 最优状态数 = {C['best_by_bic']}；ΔBIC(vs best)="
            f"{ {k: round(v, 2) for k, v in C['delta_bic_vs_best'].items()} }")
        # 前向链式 CV：按时间顺序 5 折，训练前缀 / 测试紧随其后的块
        Tn = ser_ref.size
        cuts = [int(Tn * f) for f in (0.2, 0.4, 0.6, 0.8, 1.0)]
        for i in range(4):
            tr, te = ser_ref[:cuts[i]], ser_ref[cuts[i]:cuts[i + 1]]
            row = dict(fold=i + 1, n_train=int(tr.size), n_test=int(te.size),
                       test_range=[cuts[i], cuts[i + 1]])
            for K in (1, 2, 3):
                f = gaussian_hmm_k_state(tr, K, n_iter=args.n_iter, seed=args.seed,
                                         n_restart=(1 if K == 1 else min(2, args.restarts)))
                row[f"K{K}"] = dict(train_loglik=float(f["loglik_final"]),
                                    test_loglik_block=heldout_loglik(f, te),
                                    test_loglik_per_frame=heldout_loglik(f, te) / te.size,
                                    train_ll_per_frame=float(f["loglik_final"]) / tr.size)
            C["cv_folds"].append(row)
            log(f"[C] CV fold{i + 1} 训练{tr.size}帧 测试{te.size}帧 → "
                f"留出 ll/帧: K1={row['K1']['test_loglik_per_frame']:.4f} "
                f"K2={row['K2']['test_loglik_per_frame']:.4f} "
                f"K3={row['K3']['test_loglik_per_frame']:.4f}")
        # 汇总：最后 20% 为共同测试集（训练=前 80%），三 K 直接可比
        common = dict(n_train=int(cuts[3]), n_test=int(Tn - cuts[3]), table={})
        n_rs_cv = min(3, args.restarts)
        for K in (1, 2, 3):
            f = gaussian_hmm_k_state(ser_ref[:cuts[3]], K, n_iter=args.n_iter, seed=args.seed,
                                     n_restart=(1 if K == 1 else n_rs_cv))
            common["table"][f"K{K}"] = dict(
                train_loglik=float(f["loglik_final"]),
                BIC_train=hmm_bic(f["loglik_final"], K, cuts[3]),
                test_loglik=heldout_loglik(f, ser_ref[cuts[3]:]),
                test_loglik_per_frame=heldout_loglik(f, ser_ref[cuts[3]:]) / (Tn - cuts[3]))
        C["common_split"] = common
        C["best_by_cv_common"] = int(max(common["table"],
                                         key=lambda kk: common["table"][kk]["test_loglik"]).replace("K", ""))
        log(f"[C] 共同切分（训练前 {cuts[3]} 帧 / 测试后 {Tn - cuts[3]} 帧）留出 ll/帧: " +
            " ".join(f"K{k}={v['test_loglik_per_frame']:.4f}" for k, v in common["table"].items()) +
            f" → CV 最优 = {C['best_by_cv_common']}")

        # C-附加 1：连续隐变量基线 AR(p)——「真的需要离散状态吗」（同切分、同量的留出预测似然）
        C["ar_baseline"] = []
        for p in (1, 2, 4):
            ar = ar_predictive_ll(ser_ref[:cuts[3]], ser_ref[cuts[3]:], p)
            ar["note"] = ("AR(1) 时间常数 τ=-dt/ln φ；与 HMM 的留出 ll 同量（均为一步预测似然）")
            C["ar_baseline"].append(ar)
            log(f"[C] AR({p}) 留出 ll/帧={ar['test_loglik_per_frame']:.4f} "
                f"p={ar['n_params']} BIC_test={ar['BIC_test']:.1f} "
                f"φ1={ar['ar1_phi'] if ar['ar1_phi'] is None else round(ar['ar1_phi'], 4)} "
                f"τ={ar['ar1_tau_s'] if ar['ar1_tau_s'] is None else round(ar['ar1_tau_s'], 2)}s")
            add(f"c_AR{p}_test_ll_per_frame", ar["test_loglik_per_frame"], "nats/frame", f"AR({p})",
                "留出后20%", "本工具 C 项", f"BIC_test={ar['BIC_test']:.1f} p={ar['n_params']}")
            add(f"c_AR{p}_BIC_test", ar["BIC_test"], "", f"AR({p})", "留出后20%", "本工具 C 项", "")
        C["ar_vs_hmm"] = dict(
            best_ar_test_ll_per_frame=float(max(a["test_loglik_per_frame"] for a in C["ar_baseline"])),
            best_ar_BIC_test=float(min(a["BIC_test"] for a in C["ar_baseline"])),
            K2_test_ll_per_frame=common["table"]["K2"]["test_loglik_per_frame"],
            K3_test_ll_per_frame=common["table"]["K3"]["test_loglik_per_frame"],
            best_hmm_BIC_test=float(min(-2.0 * common["table"][k]["test_loglik"]
                                        + (int(k[1]) ** 2 + 2 * int(k[1]) - 1)
                                        * math.log(Tn - cuts[3]) for k in ("K2", "K3"))),
            note="AR(p) 与 HMM 的留出对数似然是同一个量（Σ_t log p(x_t|x_{<t})），可直接比较；"
                 "参数数已列入 BIC。")
        log(f"[C] AR vs HMM（留出 ll/帧）：AR 最好 "
            f"{C['ar_vs_hmm']['best_ar_test_ll_per_frame']:.4f} | "
            f"K2 {C['ar_vs_hmm']['K2_test_ll_per_frame']:.4f} | "
            f"K3 {C['ar_vs_hmm']['K3_test_ll_per_frame']:.4f}")

        # C-附加 2：K=2 vs K=3 的「低态」到底是哪个状态 + 状态对应表
        f2 = gaussian_hmm_k_state(ser_ref, 2, n_iter=args.n_iter, seed=args.seed, n_restart=1)
        f3 = gaussian_hmm_k_state(ser_ref, 3, n_iter=args.n_iter, seed=args.seed, n_restart=1)
        C["dwell_per_state"] = {
            "K1": [dict(state=0, occupancy=1.0, n_bouts=1,
                        median_s=float(Tn * DT_S), mean_s=float(Tn * DT_S),
                        p90_s=float(Tn * DT_S), max_s=float(Tn * DT_S),
                        note="1 状态模型没有停留时长概念（整段即一个 bout）→ dwell 未定义/不可比")],
            "K2": dwell_per_state(f2["state"], 2, DT_S),
            "K3": dwell_per_state(f3["state"], 3, DT_S)}
        cont = np.zeros((3, 2), dtype=int)
        for a in range(3):
            for b in range(2):
                cont[a, b] = int(np.sum((f3["state"] == a) & (f2["state"] == b)))
        C["state_correspondence_K3_rows_vs_K2_cols"] = cont.tolist()
        C["state_correspondence_note"] = ("K=3 的低态是否 = K=2 的低态？见列联表（行=K3 状态 0/1/2，"
                                          "列=K2 状态 0/1）")
        for K, key in ((2, "K2"), (3, "K3")):
            for s in C["dwell_per_state"][key]:
                add(f"c_{key}_state{s['state']}_median_dwell", s["median_s"], "s",
                    f"H(K={K}) 状态{s['state']}", "全记录(7879帧)", "本工具 C 项",
                    f"占用={s['occupancy']:.3f} bout={s['n_bouts']}")
        log(f"[C] K=3 逐状态 dwell: " + " | ".join(
            f"s{d['state']}(occ{d['occupancy']:.3f}) med={d['median_s']:.2f}s n={d['n_bouts']}"
            for d in C["dwell_per_state"]["K3"]))
        log(f"[C] K3→K2 状态对应表（行 K3 0/1/2 × 列 K2 0/1）={cont.tolist()}")
        C["note"] = ("BIC 对 HMM 常偏高估状态数（参数计数不含状态后验的不确定性）；"
                     "两项证据（BIC / 留出似然）并列报告，若不一致则如实登记。")
        results["C"] = C
        for f in C["fits"]:
            add(f"c_K{f['n_state']}_loglik", f["loglik_final"], "nats", f"H(K={f['n_state']})",
                "全记录(7879帧)", "本工具 C 项", "最终参数下的对数似然（与 BIC 同口径）")
            add(f"c_K{f['n_state']}_BIC", f["BIC"], "", f"H(K={f['n_state']})", "全记录(7879帧)",
                "本工具 C 项", f"p={f['n_params']} 自由参数")
            add(f"c_K{f['n_state']}_delta_BIC", C["delta_bic_vs_best"][str(f["n_state"])], "",
                f"H(K={f['n_state']})", "全记录(7879帧)", "本工具 C 项", "相对 BIC 最优 K 的差")
            add(f"c_K{f['n_state']}_test_ll_per_frame", common["table"][f"K{f['n_state']}"]["test_loglik_per_frame"],
                "nats/frame", f"H(K={f['n_state']})", "留出后20%", "本工具 C 项", "前 80% 训练")
        add("c_best_by_bic", C["best_by_bic"], "K", "BIC", "全记录(7879帧)", "本工具 C 项", "")
        add("c_best_by_cv", C["best_by_cv_common"], "K", "CV 留出似然", "留出后20%", "本工具 C 项", "")

    # ================= D dwell 分布形态 =================
    if "D" in stages:
        log("[D] 低态 dwell 分布形态（H 法参考序列 K=2）...")
        if hA is None:
            hA = h_method(ser_ref, DT_S, 2)
        st = hA["state"]
        st_low, en_low = _runs(st, 0)
        dur_f_all = (en_low - st_low).astype(np.float64)
        interior = (st_low > 0) & (en_low < st.size)
        dur_f_int = dur_f_all[interior]
        A00 = hA["A"][0][0]
        p_geom = 1.0 - A00
        ks_exp = ks_test(dur_f_all * DT_S, lambda d: 1 - np.exp(-d / (dur_f_all * DT_S).mean()))
        ks_geom_hmm = ks_test(dur_f_all, lambda d: 1 - (1 - p_geom) ** d)
        ks_geom_mle = ks_test(dur_f_all, lambda d: 1 - (1 - 1.0 / dur_f_all.mean()) ** d)
        surv = survival_logfit(dur_f_all)
        hist_edges = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        counts, _ = np.histogram(dur_f_all, bins=hist_edges)
        log(f"[D] 低态 bout n={dur_f_all.size}（完整段 {dur_f_int.size}）中位={np.median(dur_f_all) * DT_S:.3f}s "
            f"均值={dur_f_all.mean() * DT_S:.3f}s p90={np.quantile(dur_f_all, 0.9) * DT_S:.3f}s "
            f"max={dur_f_all.max() * DT_S:.3f}s")
        log(f"[D] HMM 隐含几何 dwell = 1/(1-A00) = {1 / p_geom:.1f} 帧 = {DT_S / p_geom:.2f} s "
            f"vs 实测均值 {dur_f_all.mean() * DT_S:.2f} s")
        log(f"[D] KS(指数, 均值匹配) D={ks_exp['D']:.4f} p={ks_exp['p']:.4g} | "
            f"KS(几何, p=1-A00) D={ks_geom_hmm['D']:.4f} p={ks_geom_hmm['p']:.4g} | "
            f"KS(几何, p=1/mean) D={ks_geom_mle['D']:.4f} p={ks_geom_mle['p']:.4g}")
        log(f"[D] 对数生存直线性 R²={surv.get('r2')} slope={surv.get('slope_per_frame')} "
            f"→ 隐含均值 {surv.get('implied_mean_frames')} 帧")
        results["D"] = dict(
            method="H 法（K=2 高斯 HMM，参考实现）低态 bout 时长",
            n_bouts=int(dur_f_all.size), n_bouts_interior=int(dur_f_int.size),
            median_s=float(np.median(dur_f_all) * DT_S), mean_s=float(dur_f_all.mean() * DT_S),
            p90_s=float(np.quantile(dur_f_all, 0.9) * DT_S), max_s=float(dur_f_all.max() * DT_S),
            min_s=float(dur_f_all.min() * DT_S),
            mean_frames=float(dur_f_all.mean()),
            implied_geometric_frames=float(1.0 / p_geom),
            implied_geometric_s=float(DT_S / p_geom),
            A00=float(A00),
            ks_vs_exponential=ks_exp, ks_vs_geometric_from_A=ks_geom_hmm,
            ks_vs_geometric_from_mean=ks_geom_mle, survival_logfit=surv,
            histogram=dict(bin_edges_frames=hist_edges, counts=[int(c) for c in counts]),
            dur_frames=[float(v) for v in dur_f_all],
            note="HMM 的固有假设 = dwell 服从几何分布（等价于指数）；KS 用渐近 Kolmogorov p 值；"
                 "离散数据上 KS 略保守。")
        add("d_low_dwell_n_bouts", dur_f_all.size, "bouts", "H(K=2)", "全记录", "本工具 D 项", "")
        add("d_low_dwell_median", np.median(dur_f_all) * DT_S, "s", "H(K=2)", "全记录", "本工具 D 项", "")
        add("d_low_dwell_mean", dur_f_all.mean() * DT_S, "s", "H(K=2)", "全记录", "本工具 D 项", "")
        add("d_low_dwell_p90", np.quantile(dur_f_all, 0.9) * DT_S, "s", "H(K=2)", "全记录", "本工具 D 项", "")
        add("d_low_dwell_implied_geometric", DT_S / p_geom, "s", "H(K=2)", "全记录", "本工具 D 项",
            "1/(1-A00)·dt：HMM 自洽性核对")
        add("d_ks_exponential_p", ks_exp["p"], "p", "H(K=2)", "全记录", "本工具 D 项",
            f"D={ks_exp['D']:.4f}")
        add("d_ks_geometric_A_p", ks_geom_hmm["p"], "p", "H(K=2)", "全记录", "本工具 D 项",
            f"D={ks_geom_hmm['D']:.4f}（p=1-A00）")
        add("d_survival_log_r2", surv.get("r2"), "", "H(K=2)", "全记录", "本工具 D 项",
            "log 生存曲线直线性 → 指数/几何假设")

    # ================= E 跨条件 / 段内稳定性 =================
    if "E" in stages:
        offs = list(ZAP["condition_offsets"])
        names = list(ZAP["condition_names"])
        log(f"[E] 按 {len(names)} 个条件分段（CONDITION_OFFSETS 单调递增 ⇒ 无条件交错段）...")
        E = dict(conditions=[], per_condition={}, equal_length={}, half_split={},
                 group_info="not_available（provenance: 单次记录 prefix=volumes/20240930/traces，"
                            "无按鱼/按重复分组标签）")
        segs = [(names[i], offs[i], offs[i + 1]) for i in range(len(names))]
        for nm, a, b in segs:
            E["conditions"].append(dict(name=nm, start=int(a), end=int(b), n_frames=int(b - a)))
        # 等长控制：最短段长（591 帧）
        min_len = min(b - a for _, a, b in segs)
        for nm, a, b in segs:
            sub = pop[a:b]
            ser = build_series(sub, 1, SMOOTH_BINS_REF)
            qe = q_method(ser, DT_S)
            he = h_method(ser, DT_S, 2)
            sers = build_series(pop[a:a + min_len], 1, SMOOTH_BINS_REF)
            qeq = q_method(sers, DT_S)
            heq = h_method(sers, DT_S, 2)
            # 段内前半/后半（漂移代理）
            half = (b - a) // 2
            hf = []
            for sl in (slice(a, a + half), slice(a + half, b)):
                sh = build_series(pop[sl], 1, SMOOTH_BINS_REF)
                hf.append(h_method(sh, DT_S, 2)["low"]["median_s"])
            cur = dict(name=nm, n_frames=int(b - a), duration_s=float((b - a) * DT_S),
                       Q=dict(occupancy=qe["occupancy"], transition_rate_hz=qe["transition_rate_hz"],
                              low=qe["low"], high=qe["high"]),
                       H=dict(occupancy=he["occupancy"], transition_rate_hz=he["transition_rate_hz"],
                              low=he["low"], high=he["high"], mu=he["mu"], A=he["A"]),
                       Q_equal_len=dict(transition_rate_hz=qeq["transition_rate_hz"],
                                        low_median_s=qeq["low"]["median_s"],
                                        low_n_bouts=qeq["low"]["n_bouts"]),
                       H_equal_len=dict(transition_rate_hz=heq["transition_rate_hz"],
                                        low_median_s=heq["low"]["median_s"],
                                        low_n_bouts=heq["low"]["n_bouts"],
                                        occupancy=heq["occupancy"]),
                       H_half_low_median_s=[float(v) for v in hf])
            E["per_condition"][nm] = cur
            log(f"[E] {nm:10s} T={b - a:4d}({(b - a) * DT_S:6.1f}s) | "
                f"Q trans={qe['transition_rate_hz']:.4f}Hz low_med={qe['low']['median_s']:6.3f}s "
                f"n={qe['low']['n_bouts']:3d} | H trans={he['transition_rate_hz']:.4f}Hz "
                f"low_med={he['low']['median_s']:7.3f}s n={he['low']['n_bouts']:3d} "
                f"occ_low={he['occupancy'][0]:.3f} | H(等长{min_len}) low_med={heq['low']['median_s']:7.3f}s")
            emit_dwell(f"e_{nm}_Q", qe, "Q", f"条件段 {nm}({b - a}帧)", "ZAPBench 条件分段",
                       "E 项跨条件离散度")
            emit_dwell(f"e_{nm}_H2", he, "H(K=2)", f"条件段 {nm}({b - a}帧)", "ZAPBench 条件分段",
                       "E 项跨条件离散度")
            add(f"e_{nm}_H2_equal_len_low_dwell_median", heq["low"]["median_s"], "s", "H(K=2)",
                f"条件段 {nm} 前{min_len}帧", "ZAPBench 条件分段", "等长控制（最短段长）")

        def _disp(vals: list[float]) -> dict:
            v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
            q1, q3 = float(np.quantile(v, 0.25)), float(np.quantile(v, 0.75))
            return dict(n=int(v.size), median=float(np.median(v)), q1=q1, q3=q3,
                        iqr=q3 - q1, iqr_over_median=float((q3 - q1) / np.median(v)) if np.median(v) else None,
                        min=float(v.min()), max=float(v.max()),
                        max_over_min=float(v.max() / v.min()) if v.min() > 0 else None)

        E["dispersion"] = dict(
            H_low_dwell_median=_disp([E["per_condition"][n]["H"]["low"]["median_s"] for n in names]),
            H_transition_rate=_disp([E["per_condition"][n]["H"]["transition_rate_hz"] for n in names]),
            H_occupancy_low=_disp([E["per_condition"][n]["H"]["occupancy"][0] for n in names]),
            Q_low_dwell_median=_disp([E["per_condition"][n]["Q"]["low"]["median_s"] for n in names]),
            Q_transition_rate=_disp([E["per_condition"][n]["Q"]["transition_rate_hz"] for n in names]),
            H_low_dwell_median_equal_len=_disp(
                [E["per_condition"][n]["H_equal_len"]["low_median_s"] for n in names]),
            H_transition_rate_equal_len=_disp(
                [E["per_condition"][n]["H_equal_len"]["transition_rate_hz"] for n in names]),
            H_half_split_low_dwell_median=_disp(
                [v for n in names for v in E["per_condition"][n]["H_half_low_median_s"]]),
        )
        for k, v in E["dispersion"].items():
            log(f"[E] {k}: 中位={v['median']:.4f} IQR=[{v['q1']:.4f},{v['q3']:.4f}] "
                f"max/min={v['max_over_min']:.2f}x n={v['n']}")
            add(f"e_disp_{k}_median", v["median"], "", k, "9 条件", "本工具 E 项", "跨条件中位")
            add(f"e_disp_{k}_iqr_rel", v["iqr_over_median"], "x", k, "9 条件", "本工具 E 项",
                f"IQR/中位；IQR=[{v['q1']:.4f},{v['q3']:.4f}]")
            add(f"e_disp_{k}_max_over_min", v["max_over_min"], "x", k, "9 条件", "本工具 E 项",
                "跨条件极差倍数")
        results["E"] = E

    # ================= F 子采样衰减 =================
    if "F" in stages:
        log(f"[F] 神经元子采样衰减：N={SUB_N}（每次 {args.sub_seeds} 个随机种子；"
            f"事件检测用同一份 8192 行的 per-neuron 阈值）...")
        F = dict(n_neurons_full=int(N), sizes=list(SUB_N), seeds=list(range(args.sub_seeds)),
                 per_run=[], summary={})
        counts_by_seed: dict = {}
        for n in SUB_N:
            for sd in range(args.sub_seeds):
                if n == N and sd > 0:
                    continue
                rng = np.random.default_rng(args.seed + 100 * sd)
                cols = (np.arange(N) if n == N else np.sort(rng.choice(N, size=n, replace=False)))
                cnt = ev[:, cols].sum(axis=1).astype(np.float64)
                counts_by_seed[(n, sd)] = cnt
                ser = build_series(cnt, 1, SMOOTH_BINS_REF)
                qd = q_method(ser, DT_S)
                hd = h_method(ser, DT_S, 2)
                r = float(np.corrcoef(ser, ser_ref)[0, 1])
                row = dict(n_neurons=int(n), seed=int(sd),
                           mean_count=float(cnt.mean()), var_count=float(cnt.var()),
                           fano=float(cnt.var() / cnt.mean()),
                           corr_with_full_series=r,
                           Q=dict(occupancy=qd["occupancy"], transition_rate_hz=qd["transition_rate_hz"],
                                  n_switches=qd["n_switches"], low=qd["low"], high=qd["high"]),
                           H=dict(occupancy=hd["occupancy"], transition_rate_hz=hd["transition_rate_hz"],
                                  n_switches=hd["n_switches"], low=hd["low"], high=hd["high"],
                                  mu=hd["mu"], A=hd["A"]))
                F["per_run"].append(row)
                log(f"[F] N={n:5d} seed={sd} Fano={row['fano']:7.2f} "
                    f"r(ser,full)={r:.4f} | Q trans={qd['transition_rate_hz']:.4f}Hz "
                    f"low_med={qd['low']['median_s']:6.3f}s | H trans={hd['transition_rate_hz']:.4f}Hz "
                    f"low_med={hd['low']['median_s']:7.3f}s n={hd['low']['n_bouts']:3d} "
                    f"occ_low={hd['occupancy'][0]:.3f}")
                emit_dwell(f"f_N{n}_{sd}_Q", qd, "Q", f"N={n} seed={sd}", "ZAPBench 行子采样",
                           "F 项子采样衰减")
                emit_dwell(f"f_N{n}_{sd}_H2", hd, "H(K=2)", f"N={n} seed={sd}", "ZAPBench 行子采样",
                           "F 项子采样衰减")
                add(f"f_N{n}_{sd}_corr_with_full", r, "r", "序列形状", f"N={n} seed={sd}",
                    "本工具 F 项", "与 N=8192 全长序列的 Pearson 相关")
                add(f"f_N{n}_{sd}_fano", row["fano"], "", "计数统计", f"N={n} seed={sd}",
                    "本工具 F 项", "群体事件计数的 var/mean")

        def _ms(vals):
            v = np.asarray([x for x in vals if np.isfinite(x)], float)
            return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if v.size > 1 else 0.0,
                        min=float(v.min()), max=float(v.max()), n=int(v.size))

        for n in SUB_N:
            rs = [r for r in F["per_run"] if r["n_neurons"] == n]
            F["summary"][str(n)] = dict(
                n_runs=len(rs),
                H_low_dwell_median=_ms([r["H"]["low"]["median_s"] for r in rs]),
                H_transition_rate=_ms([r["H"]["transition_rate_hz"] for r in rs]),
                H_occupancy_low=_ms([r["H"]["occupancy"][0] for r in rs]),
                Q_low_dwell_median=_ms([r["Q"]["low"]["median_s"] for r in rs]),
                Q_transition_rate=_ms([r["Q"]["transition_rate_hz"] for r in rs]),
                fano=_ms([r["fano"] for r in rs]),
                corr_with_full=_ms([r["corr_with_full_series"] for r in rs]),
                mean_count=_ms([r["mean_count"] for r in rs]))
        ref_h = F["summary"][str(N)]["H_low_dwell_median"]["mean"]
        for n in SUB_N:
            s = F["summary"][str(n)]
            add(f"f_N{n}_H_low_dwell_median", s["H_low_dwell_median"]["mean"], "s", "H(K=2)",
                f"N={n}", "本工具 F 项", f"sd={s['H_low_dwell_median']['sd']:.3f} n={s['n_runs']}")
            add(f"f_N{n}_H_transition_rate", s["H_transition_rate"]["mean"], "Hz", "H(K=2)",
                f"N={n}", "本工具 F 项", f"sd={s['H_transition_rate']['sd']:.5f}")
            add(f"f_N{n}_Q_low_dwell_median", s["Q_low_dwell_median"]["mean"], "s", "Q", f"N={n}",
                "本工具 F 项", f"sd={s['Q_low_dwell_median']['sd']:.3f}")
            add(f"f_N{n}_drift_vs_N8192", s["H_low_dwell_median"]["mean"] / ref_h, "x", "H(K=2)",
                f"N={n}/8192", "本工具 F 项", "子采样偏差倍数")
        F["drift_vs_full"] = {str(n): dict(
            H_low_dwell_median_ratio=float(F["summary"][str(n)]["H_low_dwell_median"]["mean"] / ref_h),
            Q_low_dwell_median_ratio=float(F["summary"][str(n)]["Q_low_dwell_median"]["mean"] /
                                           F["summary"][str(N)]["Q_low_dwell_median"]["mean"]),
            H_transition_ratio=float(F["summary"][str(n)]["H_transition_rate"]["mean"] /
                                     F["summary"][str(N)]["H_transition_rate"]["mean"]))
            for n in SUB_N}
        results["F"] = F

    # ---------- §8 可操作建议（数字全部由上面结果推出） ----------
    rec: dict = {}
    try:
        ratio_hq = results["A"]["ratio"]["low_dwell_median_H_over_Q"]
        rec["A_口径差倍数_H_over_Q_low_dwell_median"] = ratio_hq
        rec["成像侧_Q_法同口径基准_s"] = results["A"]["Q"]["low"]["median_s"]
        rec["成像侧_H_法基准_s"] = results["A"]["H"]["low"]["median_s"]
        rec["建议_模型侧聚合箱宽_frames"] = 1
        rec["建议_模型侧聚合箱宽_s"] = DT_S
        rec["建议_模型侧方法"] = "H（高斯 2/3 状态 HMM，与成像同一实现 gaussian_hmm_2state，seed=0）"
        tol = []
        if "E" in results:
            d = results["E"]["dispersion"]
            tol.append(("E_跨条件_H_low_dwell_max_over_min",
                        d["H_low_dwell_median"]["max_over_min"]))
            tol.append(("E_跨条件_H_low_dwell_IQR/中位",
                        d["H_low_dwell_median"]["iqr_over_median"]))
            tol.append(("E_等长控制_H_low_dwell_max_over_min",
                        d["H_low_dwell_median_equal_len"]["max_over_min"]))
            tol.append(("E_段内前半/后半_H_low_dwell_max_over_min",
                        d["H_half_split_low_dwell_median"]["max_over_min"]))
        if "F" in results:
            for k, v in results["F"]["drift_vs_full"].items():
                tol.append((f"F_N={k}_H_low_dwell_drift", max(v["H_low_dwell_median_ratio"],
                                                              1.0 / v["H_low_dwell_median_ratio"])))
        rec["容差来源实测"] = {k: float(v) for k, v in tol if v is not None and np.isfinite(v)}
        if rec["容差来源实测"]:
            rec["建议同阶窗口倍数"] = float(max(rec["容差来源实测"].values()))
            rec["建议同阶窗口"] = [1.0 / rec["建议同阶窗口倍数"], rec["建议同阶窗口倍数"]]
        # Q 法的「塌陷」标定：Q 低态 dwell 以**箱数**计几乎恒定 → 秒数随箱宽线性放大
        if "B" in results:
            rows = [r for r in results["B"]["table"] if r["variant"] == "sm46"]
            bins = [r["Q"]["low"]["median_s"] / r["bin_s"] for r in rows]
            rec["Q_法低态dwell_箱数"] = dict(
                per_bin_width={str(r["bin_frames"]): float(b) for r, b in zip(rows, bins)},
                min=float(min(bins)), max=float(max(bins)),
                mean=float(np.mean(bins)),
                note="实测：Q 法低态 dwell 折算成**箱数**在 1–16 帧/箱（0.91–14.6 s）范围内近似不变 "
                     "⇒ dwell(秒) ≈ (2~3)×箱宽；q 法在箱宽→小的时候 dwell 秒数按箱宽线性塌缩")
            rec["外推_模型侧0.01s箱若用Q法_预期dwell_s"] = [float(min(bins) * 0.01),
                                                            float(max(bins) * 0.01)]
            rec["外推_说明"] = ("由实测「Q 法 dwell ≈ 2~3 箱」线性外推到 0.01 s 箱——**这是实测关系的"
                                "外推，不是对模型侧的测量**；模型侧自相关结构不同则外推失效，故主建议"
                                "仍是「先聚合到 0.914 s 帧 + 用 H 法」而不是外推。")
    except (KeyError, TypeError, ZeroDivisionError) as ex:
        rec["error"] = f"{type(ex).__name__}: {ex}"
    results["recommendation_inputs"] = rec
    log(f"[rec] 口径差 H/Q = {rec.get('A_口径差倍数_H_over_Q_low_dwell_median')}；"
        f"建议同阶窗口 = {rec.get('建议同阶窗口')}")

    # ---------- 落盘 ----------
    elapsed = time.time() - t_start
    results["meta"]["elapsed_s"] = round(elapsed, 1)
    results["meta"]["not_run"] = dict(
        per_fish_or_repeat_groups="not_available：provenance 只有单次记录前缀 volumes/20240930/traces，"
                                  "无按鱼/按重复标签（已用条件分段 + 段内前后半替代）",
        interleaved_null_segments="not_applicable：ZAPBench 9 条件为连续时间块（偏移单调递增），"
                                  "无条件交错段 → 无法用交错段做漂移对照",
        full_71721_neurons="not_run：本机只落盘子集 8192 列（原始 71721 列未下载，zenodo 吞吐限制见 provenance）",
    )
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(to_py(results), fh, ensure_ascii=False, indent=1)
    log(f"[out] JSON → {args.out_json}（{os.path.getsize(args.out_json) / 1024:.1f} KB）")
    df = pd.DataFrame(ROWS, columns=["key", "value", "unit", "method", "scope", "source", "note"])
    with open(args.out_csv, "w", encoding="utf-8") as fh:
        fh.write("# M10-B4 成像侧 dwell 基准（同口径复核）—— 长表；由 "
                 "tools/build_m10_imaging_dwell_reference.py 生成；"
                 "value 一律来自真实运行，未测项在 JSON meta.not_run 中登记\n")
    df.to_csv(args.out_csv, mode="a", index=False)
    log(f"[out] CSV → {args.out_csv}（{len(df)} 行，{df['key'].nunique()} 个 key）")

    if not args.no_fig:
        try:
            make_figure(results, args.out_fig)
            log(f"[out] PNG → {args.out_fig}")
        except Exception as ex:  # noqa: BLE001
            log(f"[fig] 出图失败（不影响数字）：{type(ex).__name__}: {ex}")

    log(f"[DONE] M10-B4 全部完成 wall={elapsed:.1f}s stages={sorted(stages)}")
    return 0


def to_py(o):
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, np.ndarray):
        return to_py(o.tolist())
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def make_figure(res: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    have = {f.name for f in fm.fontManager.ttflist}
    for cand in ("Arial Unicode MS", "Heiti TC", "PingFang HK", "Songti SC", "STHeiti",
                 "Hiragino Sans GB"):
        if cand in have:
            matplotlib.rcParams["font.sans-serif"] = [cand, "DejaVu Sans"]
            break
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(2, 3, figsize=(19, 10))
    # (1) B 箱宽曲线
    if "B" in res:
        for vname, ls in (("nosm", "-o"), ("sm46", "--s")):
            rows = [r for r in res["B"]["table"] if r["variant"] == vname]
            w = [r["bin_s"] for r in rows]
            ax[0, 0].plot(w, [r["Q"]["low"]["median_s"] for r in rows], ls, label=f"Q low ({vname})")
            ax[0, 0].plot(w, [r["H"]["low"]["median_s"] for r in rows], ls, label=f"H low ({vname})")
        ax[0, 0].axhline(REF_VALUES["dwell_median_s"], color="k", ls=":", lw=1,
                         label="参考 26.5 s")
        ax[0, 0].set_xscale("log"); ax[0, 0].set_yscale("log")
        ax[0, 0].set_xlabel("bin width (s)"); ax[0, 0].set_ylabel("low-state dwell median (s)")
        ax[0, 0].set_title("B 箱宽敏感性"); ax[0, 0].legend(fontsize=7); ax[0, 0].grid(alpha=.3)
    # (2) A 条形
    if "A" in res:
        a = res["A"]
        labels = ["Q", "H(K=2)"]
        vals = [a["Q"]["low"]["median_s"], a["H"]["low"]["median_s"]]
        ax[0, 1].bar(labels, vals, color=["tab:orange", "tab:blue"])
        for i, v in enumerate(vals):
            ax[0, 1].text(i, v, f"{v:.2f}s", ha="center", va="bottom")
        ax[0, 1].set_ylabel("low-state dwell median (s)")
        ax[0, 1].set_title(f"A 同口径对比（同一序列）H/Q = {a['ratio']['low_dwell_median_H_over_Q']:.2f}×")
    # (3) C BIC / CV
    if "C" in res:
        ks = [f["n_state"] for f in res["C"]["fits"]]
        ax[0, 2].plot(ks, [f["BIC"] for f in res["C"]["fits"]], "-o", label="BIC (train)")
        ax[0, 2].set_xlabel("n states"); ax[0, 2].set_ylabel("BIC")
        ax2 = ax[0, 2].twinx()
        cvk = sorted(int(k.replace("K", "")) for k in res["C"]["common_split"]["table"])
        ax2.plot(cvk, [res["C"]["common_split"]["table"][f"K{k}"]["test_loglik_per_frame"] for k in cvk],
                 "-s", color="tab:red", label="held-out ll/frame (HMM)")
        for ar in res["C"].get("ar_baseline", []):
            ax2.plot([max(cvk) + ar["p"] * 0.15], [ar["test_loglik_per_frame"]], "x",
                     color="tab:green", ms=8)
        ax2.plot([], [], "x", color="tab:green", label="AR(p) baseline (same split)")
        ax2.axhline(0, color="gray", lw=0.6, ls=":")
        ax2.set_ylabel("held-out loglik/frame", color="tab:red")
        ax[0, 2].set_title(f"C 状态数（BIC 优 {res['C']['best_by_bic']} / CV 优 {res['C']['best_by_cv_common']}）")
        ax[0, 2].legend(fontsize=7, loc="center left"); ax2.legend(fontsize=7, loc="lower right")
        ax[0, 2].grid(alpha=.3)
    # (4) D 生存曲线
    if "D" in res:
        dur = np.asarray(res["D"]["dur_frames"], float)
        ks = np.arange(1, int(dur.max()) + 1)
        S = np.array([(dur >= k).mean() for k in ks])
        ax[1, 0].semilogy(ks * DT_S, S, "o-", ms=3, label="实测")
        lam = 1.0 / dur.mean()
        ax[1, 0].semilogy(ks * DT_S, np.exp(-lam * ks), "k--", label="指数(均值匹配)")
        ax[1, 0].set_xlabel("dwell (s)"); ax[1, 0].set_ylabel("survival")
        ax[1, 0].set_title(f"D 低态 dwell 生存曲线（R²={res['D']['survival_logfit'].get('r2')}）")
        ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=.3)
    # (5) E 跨条件
    if "E" in res:
        names = list(res["E"]["per_condition"].keys())
        ax[1, 1].bar(range(len(names)), [res["E"]["per_condition"][n]["H"]["low"]["median_s"] for n in names],
                     color="tab:blue", label="H(K=2)")
        ax[1, 1].plot(range(len(names)), [res["E"]["per_condition"][n]["H_equal_len"]["low_median_s"]
                                          for n in names], "r^", label="H 等长控制")
        ax[1, 1].axhline(REF_VALUES["dwell_median_s"], color="k", ls=":")
        ax[1, 1].set_xticks(range(len(names))); ax[1, 1].set_xticklabels(names, rotation=45, ha="right")
        ax[1, 1].set_ylabel("low-state dwell median (s)")
        ax[1, 1].set_title("E 跨条件离散度（黑点线 = 全记录 26.5 s）")
        ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=.3)
    # (6) F 子采样
    if "F" in res:
        ns = [int(k) for k in res["F"]["summary"]]
        ax[1, 2].errorbar(ns, [res["F"]["summary"][str(n)]["H_low_dwell_median"]["mean"] for n in ns],
                          yerr=[res["F"]["summary"][str(n)]["H_low_dwell_median"]["sd"] for n in ns],
                          fmt="-o", label="H 低态 dwell 中位")
        ax[1, 2].set_xscale("log", base=2); ax[1, 2].set_xlabel("neurons kept")
        ax[1, 2].set_ylabel("low-state dwell median (s)")
        ax[1, 2].axhline(REF_VALUES["dwell_median_s"], color="k", ls=":", label="参考 26.5 s")
        ax[1, 2].set_title("F 神经元子采样衰减"); ax[1, 2].legend(fontsize=8); ax[1, 2].grid(alpha=.3)
    fig.suptitle("M10-B4 ZAPBench 成像侧 dwell 基准同口径复核（ZAPBench 8192×7879, 1.0943 Hz）")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
