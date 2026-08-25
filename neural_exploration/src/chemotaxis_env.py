"""M4 L5 环境模块（引擎无关 numpy）——嗅觉/味觉趋化二维食物梯度场。

清单《生物仿真M4实施清单》§2.3/§2.4（环境与 CI 统计规格）：

  静态食物梯度（相对浓度 0..1，无量纲）：
      C(x, y) = C_max·exp(−((x−x_f)² + (y−y_f)²) / (2σ²)) + C_bg
  采样器：虫位 (x, y) → 浓度 C_sensed(t) = C(x(t), y(t))
  时间差分：s(t) = (C_sensed(t) − C_sensed(t−τ_win)) / τ_win   # [ΔC/ms]
  无梯度对照：C_max 置 0 → C ≡ C_bg（P3 判据：CĪ 无显著偏置）

  CI（Pierce-Shimomura 1999 象限式）：
      T_in  = 虫处于食物象限的时间；T_out = 对侧象限时间
      CI_i  = (T_in − T_out) / T_total ∈ [−1, 1]      # 单试次
      组统计：CĪ ± SEM、单样本 t 检验（H0: μ=0）、效应量 Cohen's d、
      无梯度对照组对照（P3/P4 判据联动）。

本模块只依赖 numpy（+scipy.stats 组统计），与 NEURON 行为参考模型共用
同一套 CI 统计代码（清单 §0 P6 可比性保证，同 M3 肌肉 ODE 哲学）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------- #
# 环境规格
# --------------------------------------------------------------------- #
@dataclass
class EnvSpec:
    """食物梯度环境参数（清单 §2.3；唯一定稿源 = CSV 的 env 行）。"""

    arena_L: float = 10.0          # 方形培养皿边长（相对单位；drop assay 简化）
    sigma: float = 1.25            # 梯度长度尺度 σ（初值 L/8；CSV 定稿）
    c_max: float = 1.0             # 梯度峰值（相对浓度）
    c_bg: float = 0.0              # 背景浓度（无梯度对照时 C ≡ C_bg）
    food_x: float = 7.5            # 食物滴 x（右上象限中心，Ward 1973 惯例）
    food_y: float = 7.5
    boundary: str = "reflect"      # 边界处理：reflect（皿内反射）/ steer（软转向）


# --------------------------------------------------------------------- #
# 梯度场
# --------------------------------------------------------------------- #
class ChemotaxisEnv:
    """静态食物梯度场 + 采样器 + 时间差分 + 象限/CI 统计（引擎无关）。

    坐标约定：方形培养皿 [0, arena_L]×[0, arena_L]，中心 (L/2, L/2)；
    象限以中心线 x=L/2、y=L/2 划分（右上=1、左上=2、左下=3、右下=4）。
    """

    def __init__(
        self,
        arena_L: float = 10.0,
        sigma: float = 1.25,
        c_max: float = 1.0,
        c_bg: float = 0.0,
        food_x: float = 7.5,
        food_y: float = 7.5,
        boundary: str = "reflect",
    ):
        self.spec = EnvSpec(arena_L=arena_L, sigma=sigma, c_max=c_max, c_bg=c_bg,
                            food_x=food_x, food_y=food_y, boundary=boundary)

    # ------------------------------------------------------------------ #
    # 梯度场
    # ------------------------------------------------------------------ #
    def concentration(self, x, y):
        """C(x, y) = C_max·exp(−r²/(2σ²)) + C_bg（标量或数组）。"""
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        r2 = (xa - self.spec.food_x) ** 2 + (ya - self.spec.food_y) ** 2
        return self.spec.c_max * np.exp(-r2 / (2.0 * self.spec.sigma ** 2)) + self.spec.c_bg

    def sample(self, x, y) -> float:
        """虫位 → 当前浓度 C_sensed(t)（相对浓度 0..1）。"""
        return float(self.concentration(x, y))

    def no_gradient(self) -> "ChemotaxisEnv":
        """无梯度对照：C_max 置 0 → C ≡ C_bg（P3/P4 对照协议）。"""
        return ChemotaxisEnv(
            arena_L=self.spec.arena_L, sigma=self.spec.sigma, c_max=0.0,
            c_bg=self.spec.c_bg, food_x=self.spec.food_x, food_y=self.spec.food_y,
            boundary=self.spec.boundary,
        )

    # ------------------------------------------------------------------ #
    # 时间差分（ASE ON/OFF 输入编码，清单 §2.2）
    # ------------------------------------------------------------------ #
    @staticmethod
    def time_diff(c_now: float, c_prev: float, tau_win_ms: float) -> float:
        """s(t) = (C(t) − C(t−τ_win)) / τ_win  [ΔC/ms]。"""
        return (c_now - c_prev) / tau_win_ms

    @staticmethod
    def time_diff_trace(c_trace: np.ndarray, tau_win_ms: float, dt_ms: float) -> np.ndarray:
        """开环 C(t) 轨迹 → s(t) 滑窗差分数组（步长 dt_ms）。

        s[i] = (C[i] − C[i−k]) / (k·dt_ms)，k = round(τ_win/dt_ms)；
        窗口未满（i < k）时左侧以 C[0] 填充——浓度阶跃在 t0 的差分响应
        持续 τ_win（s>0 从 t0 起），与 ASE 时间差分的滑窗语义一致。
        """
        c = np.asarray(c_trace, dtype=float)
        k = max(1, int(round(tau_win_ms / dt_ms)))
        # 左侧填充 k 个 C[0] → cp[i] = C[i−k]（i<k 时为 C[0]）
        cp = np.concatenate([np.full(k, c[0]), c])
        s = (c - cp[:len(c)]) / (k * dt_ms)
        return s

    # ------------------------------------------------------------------ #
    # 边界与象限
    # ------------------------------------------------------------------ #
    def in_arena(self, x, y, tol: float = 1e-9) -> bool:
        return bool(-tol <= float(x) <= self.spec.arena_L + tol
                    and -tol <= float(y) <= self.spec.arena_L + tol)

    def quadrant(self, x, y) -> int:
        """象限：右上=1（食物默认在此）、左上=2、左下=3（对侧）、右下=4。
        中心线（x==L/2 或 y==L/2）不计入任何象限（返回 0）。
        """
        h = self.spec.arena_L / 2.0
        x, y = float(x), float(y)
        if x == h or y == h:
            return 0
        if y > h:
            return 2 if x < h else 1
        return 3 if x < h else 4

    def food_quadrant(self) -> int:
        """含食物滴的象限。"""
        return self.quadrant(self.spec.food_x, self.spec.food_y)

    def opposite_quadrant(self) -> int:
        """对侧象限（象限 1↔3、2↔4）。"""
        return {1: 3, 2: 4, 3: 1, 4: 2}[self.food_quadrant()]

    # ------------------------------------------------------------------ #
    # CI 统计（Pierce-Shimomura 1999 象限式；两引擎共用同一代码）
    # ------------------------------------------------------------------ #
    def ci_contributions(self, x, y) -> np.ndarray:
        """逐采样点 CI 贡献：食物象限 +1、对侧 −1、其余 0。"""
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        q = np.array([self.quadrant(xi, yi) for xi, yi in zip(xa.ravel(), ya.ravel())])
        contrib = np.zeros_like(q, dtype=float)
        contrib[q == self.food_quadrant()] = 1.0
        contrib[q == self.opposite_quadrant()] = -1.0
        return contrib.reshape(xa.shape)

    def ci_per_trial(self, x, y) -> float:
        """单试次 CI = (T_in − T_out)/T_total ∈ [−1, 1]。"""
        c = self.ci_contributions(x, y)
        return float(np.mean(c))

    # ------------------------------------------------------------------ #
    # 轨迹统计 / 有界检查
    # ------------------------------------------------------------------ #
    def trajectory_stats(self, x, y) -> Dict[str, float]:
        """轨迹汇总统计（供 P3/P4 判定与绘图标注）。"""
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        dist_food = np.hypot(xa - self.spec.food_x, ya - self.spec.food_y)
        return dict(
            x_min=float(xa.min()), x_max=float(xa.max()),
            y_min=float(ya.min()), y_max=float(ya.max()),
            n_points=int(xa.size),
            final_dist_food=float(dist_food[-1]),
            start_dist_food=float(dist_food[0]),
            has_nan=bool(np.any(~np.isfinite(xa)) or np.any(~np.isfinite(ya))),
        )

    def assert_bounded(self, x, y, tol: float = 1e-6) -> bool:
        """轨迹有界检查（P3）：全程在皿内、数值有限。越界/NaN 抛 ValueError。"""
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        if np.any(~np.isfinite(xa)) or np.any(~np.isfinite(ya)):
            raise ValueError("轨迹包含 NaN/Inf——运动学积分发散")
        lo, hi = -tol, self.spec.arena_L + tol
        if xa.min() < lo or xa.max() > hi or ya.min() < lo or ya.max() > hi:
            raise ValueError(
                f"轨迹越界：x∈[{xa.min():.3f},{xa.max():.3f}] "
                f"y∈[{ya.min():.3f},{ya.max():.3f}]（皿 [0,{self.spec.arena_L}]）")
        return True


# --------------------------------------------------------------------- #
# 组统计（P3/P4 判据；两引擎共用）
# --------------------------------------------------------------------- #
def ci_group_stats(ci_values: Sequence[float],
                   band_lo: float = 0.25, band_hi: float = 0.75) -> Dict[str, float]:
    """CI 组统计：均值/SEM/单样本 t 检验（H0: μ=0）/Cohen's d/生物带落位。"""
    from scipy import stats

    v = np.asarray(ci_values, dtype=float)
    n = int(v.size)
    mean = float(v.mean())
    std = float(v.std(ddof=1)) if n > 1 else 0.0
    sem = std / math.sqrt(n) if n > 0 else float("nan")
    if n > 1 and std > 0:
        t_stat, p_value = stats.ttest_1samp(v, 0.0)
        d = mean / std
    else:
        t_stat, p_value, d = float("nan"), float("nan"), float("nan")
    return dict(
        n=n, mean=mean, std=std, sem=sem,
        t_stat=float(t_stat), p_value=float(p_value),
        cohen_d=float(d),
        in_band=bool(band_lo <= mean <= band_hi),
        band_lo=band_lo, band_hi=band_hi,
    )


def compare_vs_control(ci_gradient: Sequence[float], ci_control: Sequence[float]) -> Dict[str, float]:
    """梯度组 vs 无梯度对照组（P4 联动复核 / P5 消融对比）。"""
    from scipy import stats

    g = np.asarray(ci_gradient, dtype=float)
    c = np.asarray(ci_control, dtype=float)
    out = dict(
        mean_gradient=float(g.mean()) if g.size else float("nan"),
        mean_control=float(c.mean()) if c.size else float("nan"),
        diff_mean=float(g.mean() - c.mean()) if g.size and c.size else float("nan"),
    )
    if g.size > 1 and c.size > 1:
        t_stat, p_value = stats.ttest_ind(g, c, equal_var=False)
        out["t_stat"] = float(t_stat)
        out["p_value"] = float(p_value)
    else:
        out["t_stat"] = out["p_value"] = float("nan")
    return out


# --------------------------------------------------------------------- #
# P1 开环阶跃协议（清单 §2.2：虫位固定，浓度阶跃）
# --------------------------------------------------------------------- #
def step_protocol(c_base: float = 0.2, delta_c: float = 0.5,
                  t_baseline_ms: float = 40.0, t_up_ms: float = 50.0,
                  t_hold_ms: float = 50.0, t_down_ms: float = 50.0,
                  dt_ms: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """P1 开环阶跃 C(t)：基线(≥40ms) → 上升 ΔC=+0.5(≥50ms) → 静止(≥50ms)
    → 下降 ΔC=−0.5(≥50ms) → 静止。返回 (t_ms, c_trace)。

    HH 静息瞬态漂移：基线期 ≥ 40ms（M1/M2/M3 结论）。
    """
    segs = [t_baseline_ms, t_up_ms, t_hold_ms, t_down_ms, t_hold_ms]
    t_total = sum(segs)
    t = np.arange(0.0, t_total, dt_ms)
    c = np.full_like(t, c_base)
    # 上升段
    i0 = int(round(t_baseline_ms / dt_ms))
    i1 = int(round((t_baseline_ms + t_up_ms) / dt_ms))
    c[i0:i1] = c_base + delta_c
    # 静止段（保持 c_base+delta_c）
    i2 = int(round((t_baseline_ms + t_up_ms + t_hold_ms) / dt_ms))
    c[i1:i2] = c_base + delta_c
    # 下降段 → 回到 c_base
    i3 = int(round((t_baseline_ms + t_up_ms + t_hold_ms + t_down_ms) / dt_ms))
    c[i2:i3] = c_base
    return t, c


def stationary_protocol(c_const: float = 0.2, t_total_ms: float = 150.0,
                        dt_ms: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """静止浓度协议（ΔC=0，P1 静止段判据）：全段恒定 C。"""
    t = np.arange(0.0, t_total_ms, dt_ms)
    return t, np.full_like(t, c_const)


# --------------------------------------------------------------------- #
# 时间差分跟踪器（闭环 epoch 迭代用）
# --------------------------------------------------------------------- #
class TimeDiffTracker:
    """滑窗时间差分跟踪（闭环）：记录 (t, C) 历史，按 τ_win 插值差分。

    s(t) = (C(t) − C(t−τ_win)) / τ_win；
    t−τ_win < 0（窗口未满）时用首样本 C(0) 填充 → 差分从 0 平滑建立。
    """

    def __init__(self, tau_win_ms: float, c0: float):
        self.tau_win_ms = float(tau_win_ms)
        self._t: List[float] = [0.0]
        self._c: List[float] = [float(c0)]

    def s_at(self, t_ms: float, c_now: float) -> float:
        """当前时刻浓度 c_now 处的时间差分（记录样本并返回 s）。"""
        c_prev = float(np.interp(t_ms - self.tau_win_ms, self._t, self._c,
                                 left=self._c[0], right=c_now))
        self._t.append(float(t_ms))
        self._c.append(float(c_now))
        return (c_now - c_prev) / self.tau_win_ms

    def reset(self, t0_ms: float = 0.0, c0: float = 0.0):
        self._t = [t0_ms]
        self._c = [c0]
