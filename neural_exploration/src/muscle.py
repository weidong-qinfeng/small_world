"""M3 虚拟肌肉：多神经元驱动 + 方向双通道（M0 smoke_loop 积分器升级，清单 §2.3/§4.2）。

方程（清单 §2.3 规格，与 M0 smoke_loop 同一 ODE）：
    dC/dt = -C/TAU_MUSCLE + Σ_k w_k·δ(t - t_k)      # 每运动神经元发放 → 收缩增量 w_k
    C ∈ [0, 1]（可选饱和；默认不设上限，权重由 data/m3_reflex_params.csv 定稿保证 C<1）
    双通道：C_back（DA 驱动，后退）、C_fwd（VB 驱动，前进）
    方向判定：D(t) = C_back(t) - C_fwd(t)；D 峰值 > 0.3 → 后退（清单 P1）

两种形态（同一 ODE，引擎无关——两引擎共用同一肌肉方程，行为潜伏期才可比，清单 §3）：
  1. `integrate_muscle`（纯 numpy，事件驱动精确解）——供 NEURON 参考解使用：
     由 NEURON 发放序列经同一肌肉 ODE 计算行为潜伏期；
  2. `Muscle`（Brian2 NeuronGroup + Synapses on_pre 增量）——供 ReflexArc 主线使用
     （M0 同款：`on_pre="c_back_post += WMUSC"` 事件增量 + 指数衰减积分器）。

确定性铁律：本模块不含任何随机性 → 同参数重跑逐位一致。
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# --------------------------------------------------------------------- #
# 引擎无关：事件驱动精确解（NEURON 参考链 / scipy 共用）
# --------------------------------------------------------------------- #
def integrate_muscle(
    spike_times_ms: Sequence[float],
    weight: float,
    tau_ms: float,
    t_ms: np.ndarray,
    c0: float = 0.0,
    cap: Optional[float] = None,
) -> np.ndarray:
    """dC/dt = -C/TAU + Σ w·δ(t-t_k) 的事件驱动精确解。

    解析解：两次发放之间 C 按 exp(-t/TAU) 衰减，发放时刻 C 阶跃 +w。
    t_ms 为输出采样网格（均匀即可；spike 落在网格之间时按解析式精确演化）。

    Parameters
    ----------
    spike_times_ms : 发放时刻序列（ms，升序）
    weight : 每次发放的收缩增量 w
    tau_ms : 收缩衰减时间常数 TAU_MUSCLE（ms）
    t_ms : 输出时间网格（ms）
    c0 : 初始收缩量
    cap : 可选饱和上限（None = 不饱和）
    """
    t = np.asarray(t_ms, dtype=float)
    spikes = np.sort(np.asarray(spike_times_ms, dtype=float))
    c = np.empty_like(t)
    cur = c0
    k = 0
    t_prev = t[0] if len(t) else 0.0
    for i, ti in enumerate(t):
        while k < len(spikes) and spikes[k] <= ti:
            # 先按上一发放以来的衰减演化到 spike 时刻，再阶跃
            cur = cur * np.exp(-(spikes[k] - t_prev) / tau_ms) + weight
            if cap is not None:
                cur = min(cur, cap)
            t_prev = spikes[k]
            k += 1
        cur = cur * np.exp(-(ti - t_prev) / tau_ms)
        if cap is not None:
            cur = min(cur, cap)
        c[i] = cur
    return c


def behavioral_latency_ms(
    c_trace: np.ndarray,
    t_ms: np.ndarray,
    touch_start_ms: float,
    peak_frac: float = 0.3,
) -> float:
    """行为潜伏期：首个 C ≥ peak_frac·C_peak 的时刻 − t_touch_start（清单 §5.3 定义）。

    C_peak 取整条轨迹峰值；若峰值 ≤ 0（无收缩）返回 NaN。
    """
    c = np.asarray(c_trace, dtype=float)
    t = np.asarray(t_ms, dtype=float)
    peak = float(c.max())
    if peak <= 0:
        return float("nan")
    thr = peak_frac * peak
    idx = np.flatnonzero(c >= thr)
    if len(idx) == 0:
        return float("nan")
    return float(t[idx[0]] - touch_start_ms)


def direction_peak(c_back: np.ndarray, c_fwd: np.ndarray) -> float:
    """方向判定 D_peak = max(C_back − C_fwd)（清单 §2.3 / P1 判据 > 0.3）。"""
    return float(np.max(np.asarray(c_back, dtype=float) - np.asarray(c_fwd, dtype=float)))


def direction_trace(c_back: np.ndarray, c_fwd: np.ndarray) -> np.ndarray:
    """D(t) = C_back(t) − C_fwd(t) 全轨迹（供绘图/分析）。"""
    return np.asarray(c_back, dtype=float) - np.asarray(c_fwd, dtype=float)


# --------------------------------------------------------------------- #
# Brian2 组件：双通道肌肉（ReflexArc 主线用）
# --------------------------------------------------------------------- #
class Muscle:
    """Brian2 虚拟肌肉：双通道收缩积分器（c_back / c_fwd）。

    用法（ReflexArc 内部）：
        muscle = Muscle(tau_ms=..., name="muscle")
        muscle.build()
        muscle.connect_driver(da_neuron, "back", weight=w_back)   # DA → C_back
        muscle.connect_driver(vb_neuron, "fwd", weight=w_fwd)     # VB → C_fwd
    之后把 `muscle.groups` 与 `muscle.drivers`（Synapses 列表）加入 Network。

    每通道一个 NeuronGroup（单变量单神经元），避免同一组两个变量被不同
    Synapses 的 on_pre 写目标时的命名歧义；两通道同方程、同 TAU。

    饱和：cap 不为 None 时，on_pre 增量用 `min(c + W, CAP)`（Brian2 事件语句
    支持 min/max），衰减 ODE 本身保证 C ≥ 0。
    """

    def __init__(self, tau_ms: float = 20.0, cap: Optional[float] = None,
                 name: str = "muscle"):
        self.tau_ms = tau_ms
        self.cap = cap
        self.name = name
        self._groups: Dict[str, object] = {}
        self._drivers: Dict[str, list] = {}
        self._built = False

    # ------------------------------------------------------------------ #
    # 构建
    # ------------------------------------------------------------------ #
    def build(self):
        from brian2 import NeuronGroup, ms

        tau = self.tau_ms
        for ch, var in (("back", "c_back"), ("fwd", "c_fwd")):
            g = NeuronGroup(
                1,
                f"d{var}/dt = -{var}/({tau}*ms) : 1",
                method="euler",
                name=f"{self.name}_{ch}",
            )
            setattr(g, var, 0.0)
            self._groups[ch] = g
            self._drivers[ch] = []
        self._built = True
        return self

    def connect_driver(self, pre_neuron, channel: str, weight: float,
                       name: str = "mus_drv"):
        """运动神经元 → 肌肉通道的 Synapses（on_pre 增量 w）。

        pre_neuron : 带 `.neuron`/`.label_of` 的神经元包装（MultiCompartmentNeuron）；
        channel : 'back'（DA → C_back）或 'fwd'（VB → C_fwd）；
        触发位点：pre 的 node3（轴突末梢，与化学突触一致）。
        """
        from brian2 import Synapses

        if channel not in ("back", "fwd"):
            raise ValueError(f"通道需为 back/fwd：{channel}")
        var = "c_back" if channel == "back" else "c_fwd"
        g = self._groups[channel]
        if self.cap is not None:
            # 饱和：Brian2 2.6 事件代码支持 clip（min/max 不解析为标识符，
            # if/else 分支在 on_pre 中同样不解析——实测，见 L5 踩坑）；
            # CAP 经 namespace 传入（编译缓存纪律）
            on_pre = f"{var}_post = clip({var}_post + WMUSC, 0.0, CAP)"
            ns = {"WMUSC": weight, "CAP": self.cap}
        else:
            on_pre = f"{var}_post += WMUSC"
            ns = {"WMUSC": weight}
        syn = Synapses(
            pre_neuron.neuron, g,
            on_pre=on_pre,
            name=f"{name}_{channel}",
            namespace=ns,
        )
        from brian2 import ms

        i = pre_neuron.label_of("node3")
        syn.connect(i=i, j=0)
        syn.delay = 0.1 * ms  # 与化学突触同量级（发放→收缩的传导延迟）
        self._drivers[channel].append(syn)
        return syn

    # ------------------------------------------------------------------ #
    # 访问
    # ------------------------------------------------------------------ #
    @property
    def groups(self) -> list:
        return list(self._groups.values())

    @property
    def drivers(self) -> list:
        return [s for lst in self._drivers.values() for s in lst]

    def get(self, channel: str):
        """返回单通道 NeuronGroup（'back'/'fwd'）。"""
        return self._groups[channel]

    def monitor(self, dt_ms: float, name: str = "mon_muscle"):
        """记录 c_back/c_fwd 的两个 StateMonitor（需在 Network 中加入）。"""
        from brian2 import StateMonitor, ms

        m1 = StateMonitor(self._groups["back"], "c_back", record=True,
                          dt=dt_ms * ms, name=f"{name}_back")
        m2 = StateMonitor(self._groups["fwd"], "c_fwd", record=True,
                          dt=dt_ms * ms, name=f"{name}_fwd")
        return m1, m2
