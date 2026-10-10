"""M9 P10：发放 → 钙成像（GCaMP）正向模型 + 统计级对照（`src/adult_activity.py`，新建）。

《生物仿真M9实施清单》§4.6（P10 活动金标准）/§0.3.5（抽象登记）/§0.8（数据隔离）/
§0.7 #8（判据带定稿于 CSV 不事后调）。

**测量限制（R9，先声明）**：本机**成虫全脑钙成像 / 电生理数据集不可得** →
  - 采用**文献参数化回退**参考分布（provenance 入档，不伪造对照、不臆造数据）；
  - **只承诺统计级结论**（分布形状 / 相关性量级 / 静态比例），**不声称逐神经元成像对应**；
  - 正向模型为**线性非饱和** GCaMP（无噪声、无运动伪影、无神经域污染）→ 登记为理想化上限。

正向模型（预注册参数见 `data/m9_behavior_reference.csv` 的 activity 段）：
    C(t) ← 单指数核 h(t) = exp(−t/τ_ca)，τ_ca = 1.0 s（GCaMP6s 量级；Chen 2013）
    ΔF/F(t) = A_spike · (h * Σ_j δ(t − t_j))(t)，A_spike = 单脉冲峰幅（~0.15）
    帧化：每 500 ms 取帧（2 Hz）
**统计正演一致的重建（反演）**：∫ΔF/F dt = A_spike·τ_ca·N_spike → 由 ΔF/F 积分**精确重建**
发放数（线性核的可逆性）→ 用于量化"成像式测量对率统计的畸变"，并作为模型自检
（ρ(重建率, 真率)）；**不是**成像数据对照。

统计量（预注册判据带 activity 段）：
  - KS 距离（模型发放率分布 vs 文献对数正态参考，median 1 Hz / σ_log 1.4）；
  - 平均成对相关（成像帧率下，2000 随机对）+ **相位随机化代理零分布**（采样噪声地板对照）；
  - Fano 因子（500 ms 窗）中位；ΔF/F 静默占比；重建率-真率 Spearman；确定性逐位。

用法：
    from neural_exploration.src.adult_activity import GcampParams, GcampForward
    g = GcampForward(GcampParams()); r = g.analyze(steps, idx, n_neurons, n_steps, dt_ms)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class GcampParams:
    tau_s: float = 1.0                 # 钙衰减时间常数（s；预注册）
    frame_rate_hz: float = 2.0         # 成像帧率（Hz；预注册 1–2）
    single_spike_dff: float = 0.15     # 单脉冲ΔF/F峰幅（~10–20%，Chen 2013 量级；预注册）
    bin_ms: float = 20.0               # 内部卷积时间栅格（ms）
    kernel_tau_mult: float = 6.0       # 核截断长度 = mult × τ


class GcampForward:
    """发放序列 → GCaMP ΔF/F → 统计量（确定性、无随机源）。"""

    def __init__(self, params: Optional[GcampParams] = None):
        self.p = params or GcampParams()

    # ---------------- 1. 发放栅格 ----------------
    def spike_matrix(self, spike_steps: np.ndarray, spike_idx: np.ndarray,
                     n_neurons: int, n_steps: int, dt_ms: float
                     ) -> Tuple[np.ndarray, float]:
        """(step, idx) → (n_bins, n_neurons) 计数矩阵（20 ms 栅格）。"""
        b = self.p.bin_ms
        steps_per_bin = max(1, int(round(b / dt_ms)))
        n_bins = int(np.ceil(n_steps / steps_per_bin))
        m = np.zeros((n_bins, n_neurons), dtype=np.float32)
        if spike_steps.size:
            bins = (spike_steps // steps_per_bin).astype(np.int64)
            np.add.at(m, (bins, spike_idx.astype(np.int64)), 1.0)
        return m, b

    # ---------------- 2. 核卷积（FFT；无 scipy 依赖） ----------------
    def dff(self, spk_mat: np.ndarray, bin_ms: float) -> np.ndarray:
        """ΔF/F(t) = A · (h * spikes)(t)（线性核，FFT 精确卷积）。"""
        tau = self.p.tau_s * 1000.0                     # ms
        L = int(round(self.p.kernel_tau_mult * tau / bin_ms))
        t = np.arange(L) * bin_ms                        # ms
        k = np.exp(-t / tau)
        # 归一化语义（预注册）：单脉冲 → ΔF/F 峰 = A_spike，且 ∫ΔF/F dt = A_spike·τ_ca
        #   Σ_i k_i = A·τ_s/bin_s（离散卷积无 Δ 因子 → 积分 = Σ_i k_i · bin_s）
        k = k / k.sum() * (self.p.single_spike_dff * self.p.tau_s / (bin_ms / 1000.0))
        n = spk_mat.shape[0]
        nfft = int(2 ** np.ceil(np.log2(n + L)))
        K = np.fft.rfft(k, nfft)
        S = np.fft.rfft(spk_mat, nfft, axis=0)
        y = np.fft.irfft(S * K[:, None], nfft, axis=0)[:n]
        return y.astype(np.float32)

    # ---------------- 3. 帧化 + 重建 ----------------
    def frame(self, sig: np.ndarray, bin_ms: float) -> Tuple[np.ndarray, float]:
        """按预注册帧率取帧（每帧取核内均值 → 成像式采样）。"""
        frame_ms = 1000.0 / self.p.frame_rate_hz
        spf = max(1, int(round(frame_ms / bin_ms)))
        n_frames = sig.shape[0] // spf
        if n_frames == 0:
            return sig.copy(), bin_ms
        return sig[:n_frames * spf].reshape(n_frames, spf, -1).mean(axis=1), frame_ms

    def reconstruct_rate(self, f_frames: np.ndarray, frame_ms: float,
                         n_neurons: int) -> np.ndarray:
        """由 ΔF/F 积分精确重建发放率（Hz）：N = ∫ΔF/F dt / (A·τ_ca)。"""
        T_s = f_frames.shape[0] * frame_ms / 1000.0
        integral = f_frames.sum(axis=0) * (frame_ms / 1000.0)
        n_spk = integral / (self.p.single_spike_dff * self.p.tau_s)
        return (n_spk / max(T_s, 1e-12)).astype(np.float64)

    def reconstruct_rate_fullsig(self, y_full: np.ndarray, bin_ms: float, T_s: float,
                                 n_neurons: int) -> np.ndarray:
        """补零全尾重建（无边界截断）：N = Σ_t y·bin_s / (A·τ_ca) / T_s。"""
        integral = y_full.sum(axis=0) * (bin_ms / 1000.0)
        n_spk = integral / (self.p.single_spike_dff * self.p.tau_s)
        return (n_spk / max(T_s, 1e-12)).astype(np.float64)

    # ---------------- 4. 统计 ----------------
    def analyze(self, spike_steps: np.ndarray, spike_idx: np.ndarray, n_neurons: int,
                n_steps: int, dt_ms: float, n_pairs: int = 2000, seed: int = 0,
                rate_ref: Optional[np.ndarray] = None) -> Dict[str, Any]:
        T_s = n_steps * dt_ms / 1000.0
        spk, bin_ms = self.spike_matrix(spike_steps, spike_idx, n_neurons, n_steps, dt_ms)
        true_rate = spk.sum(axis=0) / T_s
        # **边界处理**：核尾长 L 补零后再卷积 → 窗口内每个脉冲的ΔF/F 全尾保留（避免数组截断偏置）
        tau_ms = self.p.tau_s * 1000.0
        L = int(round(self.p.kernel_tau_mult * tau_ms / bin_ms))
        padded = np.zeros((spk.shape[0] + L, n_neurons), dtype=np.float32)
        padded[:spk.shape[0]] = spk
        y_full = self.dff(padded, bin_ms)
        y_win = y_full[:spk.shape[0]]                 # 实验窗内可观测部分（成像真实情形）
        f, frame_ms = self.frame(y_win, bin_ms)
        rec_win = self.reconstruct_rate(f, frame_ms, n_neurons)          # 窗内重建（**偏低偏置**）
        rec_full = self.reconstruct_rate_fullsig(y_full, bin_ms, T_s, n_neurons)  # 补零全尾重建
        # 静默占比：ΔF/F 峰值低于阈值（预注册 = 单脉冲峰幅 A_spike 的一半）
        peak = f.max(axis=0)
        silent_ca = float(np.mean(peak < 0.5 * self.p.single_spike_dff))
        # Fano 因子（500 ms 窗，内部栅格上的 25 箱/窗）
        spf = max(1, int(round(500.0 / bin_ms)))
        nw = spk.shape[0] // spf
        ff = np.full(n_neurons, np.nan)
        if nw >= 3:
            w = spk[:nw * spf].reshape(nw, spf, -1).sum(axis=1)
            mu = w.mean(axis=0)
            var = w.var(axis=0, ddof=1)
            m = mu > 0.5
            ff[m] = var[m] / mu[m]
        # 平均成对相关（成像帧率）：仅在"可检出"神经元间（真率 ≥ 0.5 Hz；成像常规做法）
        det = np.flatnonzero(true_rate >= 0.5)
        rng = np.random.default_rng(seed + 31337)
        corr = self._pair_corr(f, det, n_pairs, rng)
        corr_null = self._pair_corr_surrogate(f, det, n_pairs, rng)
        # 精细时间尺度（100 ms 箱）的描述性相关（登记为次级量）
        spf2 = max(1, int(round(100.0 / bin_ms)))
        nw2 = y_win.shape[0] // spf2
        fine = y_win[:nw2 * spf2].reshape(nw2, spf2, -1).mean(axis=1) if nw2 >= 3 else f
        corr_fine = self._pair_corr(fine, det, n_pairs, rng)
        return {
            "T_s": T_s, "bin_ms": bin_ms, "frame_ms": frame_ms,
            "n_frames": int(f.shape[0]), "n_detectable": int(det.size),
            "true_rate": true_rate, "rec_rate": rec_win, "rec_rate_full": rec_full,
            "dff_frames": f,
            "rate_mean_hz": float(true_rate.mean()),
            "rate_median_hz": float(np.median(true_rate)),
            "silent_frac_true": float(np.mean(true_rate < 0.5)),
            "silent_frac_ca": silent_ca,
            "fano_median": float(np.nanmedian(ff)) if np.isfinite(ff).any() else float("nan"),
            "fano_mean": float(np.nanmean(ff)) if np.isfinite(ff).any() else float("nan"),
            "mean_pair_corr": corr["mean"], "n_pairs_used": corr["n"],
            "mean_pair_corr_surrogate": corr_null["mean"],
            "mean_pair_corr_fine_100ms": corr_fine["mean"],
            "dff_peak_median": float(np.median(peak)),
            "recon_rate_median": float(np.median(rec_win)),
            "recon_rate_median_full": float(np.median(rec_full)),
            "recon_window_bias": float(np.median(rec_win) / max(np.median(rec_full), 1e-12)),
        }

    @staticmethod
    def _pair_corr(traces: np.ndarray, cand: np.ndarray, n_pairs: int,
                   rng: np.random.Generator) -> Dict[str, float]:
        """随机成对 Pearson r 均值（确定性 seed）。"""
        if cand.size < 4:
            return {"mean": float("nan"), "n": 0}
        a = rng.choice(cand, size=n_pairs, replace=True)
        b = rng.choice(cand, size=n_pairs, replace=True)
        keep = a != b
        a, b = a[keep], b[keep]
        x = traces[:, a] - traces[:, a].mean(axis=0)
        y = traces[:, b] - traces[:, b].mean(axis=0)
        sx = np.sqrt((x ** 2).sum(axis=0)); sy = np.sqrt((y ** 2).sum(axis=0))
        ok = (sx > 0) & (sy > 0)
        r = (x[:, ok] * y[:, ok]).sum(axis=0) / (sx[ok] * sy[ok])
        return {"mean": float(np.mean(r)) if r.size else float("nan"), "n": int(r.size)}

    @staticmethod
    def _pair_corr_surrogate(traces: np.ndarray, cand: np.ndarray, n_pairs: int,
                             rng: np.random.Generator) -> Dict[str, float]:
        """相位随机化代理零分布：每对把一条轨迹循环移位 → 相关性的**采样噪声地板**。"""
        if cand.size < 4:
            return {"mean": float("nan"), "n": 0}
        nT = traces.shape[0]
        a = rng.choice(cand, size=n_pairs, replace=True)
        b = rng.choice(cand, size=n_pairs, replace=True)
        keep = a != b
        a, b = a[keep], b[keep]
        shifts = rng.integers(1, nT, size=a.size)
        x = traces[:, a] - traces[:, a].mean(axis=0)
        yb = np.stack([np.roll(traces[:, b[i]], int(shifts[i])) for i in range(b.size)], axis=1)
        y = yb - yb.mean(axis=0)
        sx = np.sqrt((x ** 2).sum(axis=0)); sy = np.sqrt((y ** 2).sum(axis=0))
        ok = (sx > 0) & (sy > 0)
        r = (x[:, ok] * y[:, ok]).sum(axis=0) / (sx[ok] * sy[ok])
        return {"mean": float(np.mean(r)) if r.size else float("nan"), "n": int(r.size)}


# ---------------------------------------------------------------------------
# 文献参考分布（参数化回退；成像/电生理数据不可得，R9）+ 无 scipy 的统计检验
# ---------------------------------------------------------------------------
def lognormal_cdf(x: np.ndarray, median_hz: float = 1.0, sigma_log: float = 1.4
                  ) -> np.ndarray:
    """对数正态 CDF（中位 median_hz、对数标准差 sigma_log）——文献量级回退参考。"""
    x = np.asarray(x, dtype=np.float64)
    z = np.log(np.maximum(x, 1e-12) / median_hz) / sigma_log
    return 0.5 * (1.0 + _erf(z / np.sqrt(2.0)))


def _erf(x: np.ndarray) -> np.ndarray:
    """误差函数（Abramowitz-Stegun 7.1.26 近似，|误差| < 1.5e-7）。"""
    s = np.sign(x); ax = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * np.exp(-ax * ax)
    return s * y


def ks_1samp(x: np.ndarray, cdf_fn) -> Dict[str, float]:
    """单样本 KS：返回 D 与渐近 p（Kolmogorov 分布；n 大时准确）。"""
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    if n == 0:
        return {"D": float("nan"), "p": float("nan"), "n": 0}
    F = cdf_fn(x)
    i = np.arange(1, n + 1) / n
    D = float(max(np.max(np.abs(i - F)), np.max(np.abs(F - (i - 1.0 / n)))))
    lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * D
    p = float(min(1.0, 2.0 * sum((-1) ** (k - 1) * np.exp(-2.0 * k * k * lam * lam)
                                 for k in range(1, 101))))
    return {"D": D, "p": p, "n": int(n)}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman ρ（秩相关；无 scipy）。"""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.size < 3:
        return float("nan")
    ra = _rank(a); rb = _rank(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    if d == 0:
        return float("nan")
    return float((ra * rb).sum() / d)


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    r = np.empty(x.size, dtype=np.float64)
    r[order] = np.arange(1, x.size + 1, dtype=np.float64)
    return r
