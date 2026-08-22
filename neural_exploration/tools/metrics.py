"""M0 评测指标（清单 §5.1）：M1 验证依赖的最小指标集。"""

import numpy as np


def waveform_rmse(v_sim, v_ref, dt):
    """模拟轨迹 vs 参考轨迹的 RMSE（mV），用于波形误差 <5% 判定。

    两轨迹长度可能不同（引擎步长差异），按较短长度对齐后计算。
    dt 保留在签名中：后续可扩展为按时间对齐（M1 需要时）。
    """
    v_sim = np.asarray(v_sim, dtype=float)
    v_ref = np.asarray(v_ref, dtype=float)
    n = min(len(v_sim), len(v_ref))
    if n == 0:
        raise ValueError("空轨迹无法计算 RMSE")
    return float(np.sqrt(np.mean((v_sim[:n] - v_ref[:n]) ** 2)))


def spike_count(v, threshold=-20.0):
    """跨阈值计数动作电位（上升沿计数），v 单位 mV。

    阈值 -20 mV 为清单默认；上升沿 (v[i]<th 且 v[i+1]>=th) 计一次。
    """
    v = np.asarray(v, dtype=float)
    if len(v) < 2:
        return 0
    rising = (v[:-1] < threshold) & (v[1:] >= threshold)
    return int(np.sum(rising))


def first_spike_time(v, t, threshold=-20.0):
    """首个动作电位时间（ms），无则返回 None。"""
    v = np.asarray(v, dtype=float)
    t = np.asarray(t, dtype=float)
    rising = (v[:-1] < threshold) & (v[1:] >= threshold)
    idx = np.flatnonzero(rising)
    if len(idx) == 0:
        return None
    return float(t[idx[0] + 1])
