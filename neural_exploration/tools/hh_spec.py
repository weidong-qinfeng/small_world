"""M0 统一 HH 基准实验规格（清单 §3.1 固定参数，三引擎 + RK4 参考解共用）。

单位约定（标准 HH 1952 按膜面积归一）：
- 电导 mS/cm²，电容 µF/cm²，电位 mV，时间 ms
- 注入电流：清单写"阶跃 10nA"，在按面积归一的标准 HH 中换算为 10 µA/cm²
  （本机三引擎均按面积归一实现，保证可比；详见 m0_engine_benchmark.md 单位说明）
"""

import numpy as np

# --- HH 1952 标准参数（设计文档附录 / 清单 §3.1） ---
CM = 1.0          # µF/cm²
GNA = 120.0       # mS/cm²
GK = 36.0         # mS/cm²
GL = 0.3          # mS/cm²
ENA = 50.0        # mV
EK = -77.0        # mV
EL = -54.4        # mV
V0 = -65.0        # mV，初始膜电位

# --- 刺激与采样（清单 §3.1） ---
I_STIM = 10.0     # µA/cm²（阶跃幅度）
T_STIM_START = 0.0   # ms
T_STIM_END = 50.0    # ms（持续 50 ms）
DT = 0.01         # ms 步长
T_TOTAL = 100.0   # ms 输出总时长

# --- 标准 HH 门控速率函数（v 单位 mV，速率单位 1/ms） ---


def alpha_m(v):
    return 0.1 * (v + 40.0) / (1.0 - np.exp(-(v + 40.0) / 10.0))


def beta_m(v):
    return 4.0 * np.exp(-(v + 65.0) / 18.0)


def alpha_h(v):
    return 0.07 * np.exp(-(v + 65.0) / 20.0)


def beta_h(v):
    return 1.0 / (1.0 + np.exp(-(v + 35.0) / 10.0))


def alpha_n(v):
    return 0.01 * (v + 55.0) / (1.0 - np.exp(-(v + 55.0) / 10.0))


def beta_n(v):
    return 0.125 * np.exp(-(v + 65.0) / 80.0)


def steady_state(v):
    """v 处的门控稳态值 (m, h, n)。"""
    am, bm = alpha_m(v), beta_m(v)
    ah, bh = alpha_h(v), beta_h(v)
    an, bn = alpha_n(v), beta_n(v)
    return am / (am + bm), ah / (ah + bh), an / (an + bn)


def current(v, m, h, n):
    """给定状态下的总膜电流（µA/cm²，正 = 外向）。"""
    i_na = GNA * m ** 3 * h * (v - ENA)
    i_k = GK * n ** 4 * (v - EK)
    i_l = GL * (v - EL)
    return i_na + i_k + i_l


def stimulus(t):
    """阶跃电流：T_STIM_START..T_STIM_END 期间 I_STIM µA/cm²，其余为 0。"""
    return I_STIM if T_STIM_START <= t < T_STIM_END else 0.0
