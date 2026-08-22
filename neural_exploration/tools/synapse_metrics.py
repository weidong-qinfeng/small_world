"""M2 共享度量：EPSP/IPSP 提取、波形误差、二项检验（P1–P5 复用）。"""

from __future__ import annotations

import numpy as np

from neural_exploration.tools.metrics import waveform_rmse  # noqa: F401  # 复用 M0/M1


def psp_amplitudes(
    t_ms: np.ndarray,
    v_mv: np.ndarray,
    spike_times_ms: np.ndarray,
    pre_window_ms: float = 0.4,
    tail_window_ms: float = 15.0,
    base_lead_ms: float = 3.0,
    base_gap_ms: float = 0.1,
) -> np.ndarray:
    """每次突触前发放后的 PSP 峰值幅度（mV，相对发放前本地基线）。

    基线 = [t_spike - base_lead, t_spike - base_gap] 中位数（消除 HH 静息漂移，
    M1 env_notes §L3 结论）；峰值 = [t_spike + pre_window, t_spike + tail_window] 最大。
    EPSP 为正、IPSP 为负（由测量窗口内极值方向判定）。
    """
    t = np.asarray(t_ms, dtype=float)
    v = np.asarray(v_mv, dtype=float)
    amps = []
    for tk in np.atleast_1d(spike_times_ms):
        base_win = (t >= tk - base_lead_ms) & (t < tk - base_gap_ms)
        if base_win.sum() == 0:
            continue
        base = float(np.median(v[base_win]))
        win = (t >= tk + pre_window_ms) & (t <= tk + tail_window_ms)
        if win.sum() == 0:
            continue
        vw = v[win]
        if np.abs(vw.max() - base) >= np.abs(vw.min() - base):
            amps.append(float(vw.max() - base))
        else:
            amps.append(float(vw.min() - base))
    return np.asarray(amps)


def norm_rmse(v_sim, v_ref, dt_ms: float, mask=None) -> float:
    """归一化 RMSE（按参考轨迹峰-峰幅度），mask 用于限定比较窗口。"""
    v_sim = np.asarray(v_sim, dtype=float)
    v_ref = np.asarray(v_ref, dtype=float)
    n = min(len(v_sim), len(v_ref))
    a, b = v_sim[:n], v_ref[:n]
    if mask is not None:
        m = np.asarray(mask, dtype=bool)[:n]
        if m.sum() < 10:
            raise ValueError("比较窗口过短")
        a, b = a[m], b[m]
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    span = float(b.max() - b.min())
    return rmse / span if span > 0 else float("nan")


def failure_rate(v_trials_mv: np.ndarray, baseline_mv: float, epsp_min_mv: float = 0.05) -> float:
    """多试次后 PSP 失败率：PSP 峰值幅度 < epsp_min 计为一次失败。"""
    peaks = np.max(v_trials_mv, axis=1) - baseline_mv
    return float(np.mean(peaks < epsp_min_mv))


def binomial_ci(n_trials: int, p_hat: float, z: float = 1.96):
    """比例 p_hat 的 Wilson 置信区间（n 次伯努利）。"""
    n = max(n_trials, 1)
    denom = 1 + z ** 2 / n
    center = (p_hat + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2)) / denom
    return float(center - half), float(center + half)
