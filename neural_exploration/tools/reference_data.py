"""M0 参考解：手工 RK4 求解 HH 1952（清单 §5.2）。

M0 阶段用 RK4 数值解作为参考解（dt=0.01ms，与基准一致），
落盘到 data/hh1952_trace.csv；M1 再对齐 HH 1952 原文数据。
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.hh_spec import (
    CM, DT, T_TOTAL, V0,
    alpha_h, alpha_m, alpha_n, beta_h, beta_m, beta_n,
    current, stimulus, steady_state,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def rk4_reference_trace():
    """RK4 积分 HH 方程组，返回 (t, v)，步长 DT、时长 T_TOTAL。"""
    n_steps = int(round(T_TOTAL / DT))
    t = np.arange(n_steps + 1) * DT
    v = np.empty(n_steps + 1)
    v[0] = V0
    m, h, n = steady_state(V0)

    def derivs(vv, mm, hh, nn, ii):
        dm = alpha_m(vv) * (1 - mm) - beta_m(vv) * mm
        dh = alpha_h(vv) * (1 - hh) - beta_h(vv) * hh
        dn = alpha_n(vv) * (1 - nn) - beta_n(vv) * nn
        dv = (ii - current(vv, mm, hh, nn)) / CM
        return dv, dm, dh, dn

    for i in range(n_steps):
        t_now = t[i]
        i0 = stimulus(t_now)
        k1 = derivs(v[i], m, h, n, i0)
        k2 = derivs(v[i] + DT / 2 * k1[0], m + DT / 2 * k1[1], h + DT / 2 * k1[2], n + DT / 2 * k1[3], stimulus(t_now + DT / 2))
        k3 = derivs(v[i] + DT / 2 * k2[0], m + DT / 2 * k2[1], h + DT / 2 * k2[2], n + DT / 2 * k2[3], stimulus(t_now + DT / 2))
        k4 = derivs(v[i] + DT * k3[0], m + DT * k3[1], h + DT * k3[2], n + DT * k3[3], stimulus(t_now + DT))
        v[i + 1] = v[i] + DT / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        m += DT / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        h += DT / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        n += DT / 6 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])

    return t, v


def generate_reference_csv(path=None):
    """生成 hh1952_trace.csv（t_ms, v_mV 两列），返回保存路径。"""
    path = path or os.path.join(DATA_DIR, "hh1952_trace.csv")
    t, v = rk4_reference_trace()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, np.column_stack([t, v]), delimiter=",", header="t_ms,v_mV", comments="", fmt="%.6f")
    return path


def load_reference_trace(name="hh1952_trace"):
    """加载参考轨迹 → (t, v)。name 不含扩展名。"""
    path = os.path.join(DATA_DIR, f"{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"参考轨迹不存在：{path}（先运行 generate_reference_csv）")
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


if __name__ == "__main__":
    out = generate_reference_csv()
    t, v = load_reference_trace()
    print(f"参考解已写入: {out}")
    print(f"轨迹点数: {len(t)}，时间跨度 {t[0]}..{t[-1]} ms")
    print(f"V 范围: {v.min():.2f} .. {v.max():.2f} mV")
