"""M8 活动正向模型（P8 金标准；纯 numpy，确定性——M5 同哲学）。

《生物仿真M8实施清单》§8.1/D4 规格公式：

    F_i(t) = Σ_k exp(−(t−t_ik)/τ_GCaMP)·1[t−t_ik>0]      # 发放 → 荧光卷积
    F_i^img(t) = F_i(t) 降采样至成像采样（1–2Hz，窗口平均）  # 与成像数据同采样对齐

- τ_GCaMP 预注册 0.5–1.5s（本文件默认 1.0s，调用方可覆盖）；
- 成像采样 1–2Hz（默认 2Hz，窗口平均降采样）；
- 活动态定义（§8.2 (b) 预注册）：全局活动水平分位（每窗总发放率中位数 →
  high/low 二值态）+ 可选降维主分量符号；
- 只承诺统计级不可区分（§0.7 #3），逐神经元对应不声称。

冻结组件零修改；本模块仅供验证脚本调用。
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def spikes_to_fluorescence(
    spike_times_ms: Sequence[float],
    t_span_ms: float,
    tau_gcamp_ms: float = 1000.0,
    dt_img_ms: float = 500.0,
    baseline: float = 1.0,
) -> np.ndarray:
    """发放序列 → GCaMP 类荧光（窗口平均降采样至成像采样）。

    F(t) = Σ_k exp(−(t−t_ik)/τ)·1[t>t_ik] + baseline，逐成像窗平均。
    返回形状 (n_frames,)；确定性（无噪声）。
    """
    t_arr = np.asarray(spike_times_ms, dtype=float)
    t_arr = t_arr[np.isfinite(t_arr) & (t_arr >= 0.0)]
    n_frames = max(1, int(np.ceil(t_span_ms / dt_img_ms)))
    frames = np.empty(n_frames, dtype=float)
    for f in range(n_frames):
        t0 = f * dt_img_ms
        t1 = min(t0 + dt_img_ms, t_span_ms)
        if t_arr.size == 0:
            frames[f] = baseline
            continue
        tm = (t0 + t1) / 2.0
        dt = tm - t_arr
        contrib = np.exp(-dt / tau_gcamp_ms) * (dt > 0.0)
        frames[f] = baseline + float(np.sum(contrib))
    return frames


def global_activity_per_bin(
    spike_times_by_role: Dict[str, np.ndarray],
    t_span_ms: float,
    bin_ms: float = 100.0,
) -> np.ndarray:
    """全局活动水平序列：每 bin 总发放数 / 角色数（Hz 量纲，群体发放率）。

    确定性；无 NaN（空发放 → 0）。
    """
    n_roles = max(1, len(spike_times_by_role))
    n_bins = max(1, int(np.ceil(t_span_ms / bin_ms)))
    counts = np.zeros(n_bins, dtype=float)
    for times in spike_times_by_role.values():
        t = np.asarray(times, dtype=float)
        t = t[np.isfinite(t) & (t >= 0.0)]
        if t.size == 0:
            continue
        idx = np.clip((t / bin_ms).astype(np.int64), 0, n_bins - 1)
        np.add.at(counts, idx, 1.0)
    hz = counts / (bin_ms / 1000.0) / n_roles
    return hz


def activity_state_sequence(
    activity_hz: np.ndarray,
    percentile: float = 50.0,
) -> np.ndarray:
    """活动态序列（预注册定义：全局活动水平分位 → 二值态）。

    >= 分位 → "high"（1），< 分位 → "low"（0）。确定性；空输入 → 空数组。
    """
    a = np.asarray(activity_hz, dtype=float)
    if a.size == 0:
        return np.array([], dtype=np.int8)
    thr = float(np.percentile(a, percentile))
    return (a >= thr).astype(np.int8)


def transition_windows(
    states: Sequence[str],
    activity_hz: np.ndarray,
    dt_state_ms: float,
    win_ms: float = 2000.0,
    act_bin_ms: float = 100.0,
) -> dict:
    """run↔turn 转换 ±win_ms 窗活动态序列（§8.2 (b)）。

    - states：逐 epoch 状态序列（run/turn/pause…，epoch 宽 = dt_state_ms）；
    - activity_hz：逐 bin 全局活动率（bin 宽 = act_bin_ms）；
    - 转换定义：相邻 epoch 状态在 {run, turn} 间切换（run→turn 或 turn→run）；
      转换时刻 = epoch 索引 × dt_state_ms；
    - 返回 dict(transitions, windows, pre_mean, post_mean, n_high, n_low,
      occupancy_high, has_nan)。

    windows: List[np.ndarray]，每个 = 转换时刻 ±win_ms 窗内活动态（分位二值）。
    """
    st = np.asarray(states)
    n_bin = activity_hz.size
    trans_idx = [i for i in range(1, len(st))
                 if {st[i - 1], st[i]} <= {"run", "turn"}
                 and st[i - 1] != st[i]]
    n_win_bins = max(1, int(round(win_ms / act_bin_ms)))
    windows = []
    pre_means, post_means = [], []
    n_high = n_low = 0
    for i in trans_idx:
        c_ms = i * dt_state_ms              # 转换时刻（ms）
        b0 = max(0, int(round((c_ms - win_ms) / act_bin_ms)))
        b1 = min(n_bin, int(round((c_ms + win_ms) / act_bin_ms)))
        seg = activity_hz[b0:b1]
        if seg.size == 0:
            continue
        seq = activity_state_sequence(seg)
        windows.append(seq)
        n_high += int(np.sum(seq == 1))
        n_low += int(np.sum(seq == 0))
        ci = int(round(c_ms / act_bin_ms))
        pre = activity_hz[b0:ci]
        post = activity_hz[ci:b1]
        pre_means.append(float(np.mean(pre)) if pre.size else 0.0)
        post_means.append(float(np.mean(post)) if post.size else 0.0)
    tot = n_high + n_low
    return dict(
        transitions=trans_idx,
        windows=windows,
        n_transitions=len(windows),
        pre_mean=float(np.mean(pre_means)) if pre_means else float("nan"),
        post_mean=float(np.mean(post_means)) if post_means else float("nan"),
        n_high=n_high, n_low=n_low,
        occupancy_high=(float(n_high / tot) if tot else float("nan")),
        has_nan=bool(any(not np.all(np.isfinite(w)) for w in windows)),
    )
