"""M4 L5 虚拟身体模块（引擎无关 numpy 运动学，与 NEURON 行为参考模型共用）。

清单《生物仿真M4实施清单》§2.4（趋化运动机制）：

    v(t) = v_fwd0 · clip(C_fwd(t), 0, 1)          # 前进速度 ∝ 前进肌肉收缩
    ω(t) = ω_max · (C_left(t) − C_right(t))       # 转向角速度 = 左右转向肌肉差
    dx/dt = v·cosθ；dy/dt = v·sinθ；dθ/dt = ω     # （点身体 + 朝向）

运动学积分器（引擎无关 numpy，同 M3 integrate_muscle 哲学）：
  - 半隐式（默认）：先更新 θ，再按新 θ 平移——简单稳定；
  - 精确（可选）：ω≠0 时按圆弧解析积分（Δx = v/ω·(sinθ' − sinθ) 等），
    与 ω=0 直线段一致，无旋转漂移。
  - 边界：皿内反射（reflect，确定性；P3 轨迹有界）或软转向（steer）。

单位约定（本模块定稿，记入 m4_env_notes L5+）：
  - 空间 = 皿边长相对单位（arena_L，CSV env 行）；v_fwd0 = 单位/s，
    ω_max = rad/s；行为 tick dt_b = ms（CSV body 行）。
  - 速度量程 [0, v_fwd0]，转向量程 [−ω_max, ω_max]（P3 "量程在 CSV 规格内"）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np


@dataclass
class BodySpec:
    """虚拟身体参数（清单 §2.4；唯一定稿源 = CSV 的 body 行）。"""

    v_fwd0: float = 0.3            # 前进速度上限（皿单位/s）
    omega_max: float = 3.0         # 转向角速度上限（rad/s）
    dt_b: float = 25.0             # 行为 tick（ms；= 闭环 epoch ΔT）
    v_osc: float = 0.0             # 爬行步态振荡幅度（informational；0 = 关）
    turn_omega_pir: float = 1.0    # 机制 A：转向事件角速度 [rad/s]（ω=±ω_pir）
    turn_duration_ms: float = 1571.0  # 机制 A：转向事件持续时长 [ms]（90° 转角）


class ChemotaxisBody:
    """点身体 + 朝向的运动学积分器（确定性；机制 A 转向事件由试次种子决定方向）。

    机制 A（pirouette，主 agent 裁决 2026-08-23，清单 §2.4 落地修订）：
      - 转向事件（turn event）：由闭环（chemotaxis_loop.py）在「s < −θ_pir 且
        ASER→AIB→RIA→SMDD 激活」时经 `trigger_turn(direction, ...)` 启动；
        事件持续 T_pir，期间 ω = direction·ω_pir（direction ∈ {−1,+1}，
        试次种子确定性伪随机——真实虫 pirouette 方向随机，确定性铁律不破）；
      - 事件未激活时回到两侧平衡/竞争：ω = ω_max·(C_left − C_right)。
    """

    def __init__(
        self,
        v_fwd0: float = 0.3,
        omega_max: float = 3.0,
        dt_b: float = 25.0,
        v_osc: float = 0.0,
        arena_L: float = 10.0,
        boundary: str = "reflect",
        gait_period_ms: float = 500.0,
        turn_omega_pir: float = 1.0,
        turn_duration_ms: float = 1571.0,
    ):
        self.v_fwd0 = float(v_fwd0)
        self.omega_max = float(omega_max)
        self.dt_b = float(dt_b)
        self.v_osc = float(v_osc)
        self.arena_L = float(arena_L)
        self.boundary = boundary
        self.gait_period_ms = float(gait_period_ms)   # v_osc 振荡周期（informational）
        self.turn_omega_pir = float(turn_omega_pir)   # 机制 A
        self.turn_duration_ms = float(turn_duration_ms)  # 机制 A
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.turn_remaining_ms = 0.0   # 机制 A：当前转向事件剩余时长（0 = 未转向）
        self.turn_dir = 1.0            # 机制 A：当前转向事件方向（±1）

    # ------------------------------------------------------------------ #
    # 机制 A：转向事件（pirouette）
    # ------------------------------------------------------------------ #
    def is_turning(self) -> bool:
        """是否处于转向事件中（机制 A；事件进行中不重叠触发）。"""
        return self.turn_remaining_ms > 0.0

    def trigger_turn(self, direction: float,
                     omega_pir: Optional[float] = None,
                     t_pir_ms: Optional[float] = None):
        """启动转向事件：持续 t_pir_ms，ω = direction·ω_pir。

        direction ∈ {−1, +1}（−1=左、+1=右；由试次种子确定性伪随机决定——
        机制 A 落地修订，主 agent 裁决 2026-08-23）。事件进行中调用为 no-op
        （Pierce-Shimomura 1999：单次转向事件不重叠）。
        """
        if self.turn_remaining_ms > 0.0:
            return
        self.turn_dir = 1.0 if float(direction) >= 0 else -1.0
        if omega_pir is not None:
            self.turn_omega_pir = float(omega_pir)
        if t_pir_ms is not None:
            self.turn_duration_ms = float(t_pir_ms)
        self.turn_remaining_ms = self.turn_duration_ms

    # ------------------------------------------------------------------ #
    # 运动学（引擎无关）
    # ------------------------------------------------------------------ #
    def speed(self, c_fwd: float) -> float:
        """v = v_fwd0·clip(C_fwd, 0, 1)（肌肉饱和用 numpy clip，事件代码外）。"""
        return self.v_fwd0 * float(np.clip(c_fwd, 0.0, 1.0))

    def turn_rate(self, c_left: float, c_right: float) -> float:
        """ω：机制 A 转向事件期间 = turn_dir·ω_pir；否则 ω_max·(C_left − C_right)。

        两侧平衡/竞争（ω_max·(C_left−C_right)）在对称电路下 ≈ 0——闭环净转向
        由机制 A 转向事件提供（直行/转向时长不对称偏置，Pierce-Shimomura 1999）。
        """
        if self.turn_remaining_ms > 0.0:
            return self.turn_dir * self.turn_omega_pir
        return self.omega_max * (float(c_left) - float(c_right))

    def _v_eff(self, v: float, t_ms: float) -> float:
        """可选爬行步态振荡（informational；v_osc=0 时恒等）。"""
        if self.v_osc <= 0:
            return v
        return v * (1.0 + self.v_osc * math.sin(2.0 * math.pi * t_ms / self.gait_period_ms))

    def _tick_turn_timer(self, dt_s: float):
        """行为 tick 消耗转向事件剩余时长（dt_s 秒 → ms）。"""
        if self.turn_remaining_ms > 0.0:
            self.turn_remaining_ms = max(0.0,
                                         self.turn_remaining_ms - dt_s * 1000.0)

    # ------------------------------------------------------------------ #
    # 单步积分
    # ------------------------------------------------------------------ #
    def step(self, c_fwd: float, c_left: float, c_right: float,
             dt_ms: Optional[float] = None, t_ms: float = 0.0) -> tuple:
        """一个行为 tick 的运动学积分（默认半隐式 Euler），并做边界处理。

        Returns
        -------
        (x, y, theta) : 更新后的位姿（self 同步更新）。
        """
        dt = (self.dt_b if dt_ms is None else dt_ms) / 1000.0  # ms → s
        v = self.speed(c_fwd)
        v = self._v_eff(v, t_ms)
        omega = self.turn_rate(c_left, c_right)
        # 半隐式：先转后行（θ 用更新后的值平移）
        self.theta = self.theta + omega * dt
        self.x = self.x + v * math.cos(self.theta) * dt
        self.y = self.y + v * math.sin(self.theta) * dt
        self._tick_turn_timer(dt)
        self._apply_boundary()
        return self.x, self.y, self.theta

    def step_exact(self, c_fwd: float, c_left: float, c_right: float,
                   dt_ms: Optional[float] = None, t_ms: float = 0.0) -> tuple:
        """精确运动学积分（圆弧解析；ω≈0 退化为直线段）。"""
        dt = (self.dt_b if dt_ms is None else dt_ms) / 1000.0
        v = self.speed(c_fwd)
        v = self._v_eff(v, t_ms)
        omega = self.turn_rate(c_left, c_right)
        th0 = self.theta
        if abs(omega) < 1e-12:
            self.theta = th0
            self.x = self.x + v * math.cos(th0) * dt
            self.y = self.y + v * math.sin(th0) * dt
        else:
            dth = omega * dt
            self.theta = th0 + dth
            # 圆弧：Δx = v/ω·(sinθ' − sinθ0)；Δy = v/ω·(cosθ0 − cosθ')
            self.x = self.x + v / omega * (math.sin(self.theta) - math.sin(th0))
            self.y = self.y + v / omega * (math.cos(th0) - math.cos(self.theta))
        self._tick_turn_timer(dt)
        self._apply_boundary()
        return self.x, self.y, self.theta

    # ------------------------------------------------------------------ #
    # 边界（P3：轨迹有界）
    # ------------------------------------------------------------------ #
    def _apply_boundary(self):
        L = self.arena_L
        if self.boundary == "reflect":
            # 皿内镜面反射：越 x 界 → θ → π−θ；越 y 界 → θ → −θ；位置折回皿内
            if self.x < 0.0:
                self.x = -self.x
                self.theta = math.pi - self.theta
            elif self.x > L:
                self.x = 2.0 * L - self.x
                self.theta = math.pi - self.theta
            if self.y < 0.0:
                self.y = -self.y
                self.theta = -self.theta
            elif self.y > L:
                self.y = 2.0 * L - self.y
                self.theta = -self.theta
        else:  # "steer"：软转向——把位置钳回皿内（deterministic 保守回退）
            self.x = float(np.clip(self.x, 0.0, L))
            self.y = float(np.clip(self.y, 0.0, L))

    # ------------------------------------------------------------------ #
    # 整条轨迹积分 / 重置
    # ------------------------------------------------------------------ #
    def integrate(self, c_fwd: Sequence[float], c_left: Sequence[float],
                  c_right: Sequence[float], dt_ms: Optional[float] = None,
                  exact: bool = False) -> Dict[str, np.ndarray]:
        """按三通道肌肉收缩序列积分整条轨迹（每元素 = 一个行为 tick 的收缩）。

        Returns
        -------
        dict(x, y, theta, v, omega, t_ms)
        """
        c_f = np.asarray(c_fwd, dtype=float)
        c_l = np.asarray(c_left, dtype=float)
        c_r = np.asarray(c_right, dtype=float)
        n = min(len(c_f), len(c_l), len(c_r))
        dt = self.dt_b if dt_ms is None else dt_ms
        xs, ys, ts_, vs, om = [], [], [], [], []
        for k in range(n):
            v = self.speed(c_f[k])
            v = self._v_eff(v, k * dt)
            om_k = self.turn_rate(c_l[k], c_r[k])
            if exact:
                self.step_exact(c_f[k], c_l[k], c_r[k], dt, k * dt)
            else:
                self.step(c_f[k], c_l[k], c_r[k], dt, k * dt)
            xs.append(self.x); ys.append(self.y); ts_.append(k * dt)
            vs.append(v); om.append(om_k)
        return dict(x=np.array(xs), y=np.array(ys), theta=np.array(ts_, dtype=float),
                    t_ms=np.array(ts_, dtype=float), v=np.array(vs), omega=np.array(om))

    def reset(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0):
        self.x, self.y, self.theta = float(x), float(y), float(theta)
        self.turn_remaining_ms = 0.0   # 机制 A：试次开始清空转向事件
        self.turn_dir = 1.0

    def assert_trajectory(self, x: np.ndarray, y: np.ndarray, tol: float = 1e-6) -> bool:
        """轨迹有界 + 无 NaN 检查（P3 判据前置）。越界/发散抛 ValueError。"""
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        if np.any(~np.isfinite(xa)) or np.any(~np.isfinite(ya)):
            raise ValueError("轨迹包含 NaN/Inf——运动学积分发散")
        lo, hi = -tol, self.arena_L + tol
        if xa.min() < lo or xa.max() > hi or ya.min() < lo or ya.max() > hi:
            raise ValueError(
                f"轨迹越界：x∈[{xa.min():.3f},{xa.max():.3f}] "
                f"y∈[{ya.min():.3f},{ya.max():.3f}]（皿 [0,{self.arena_L}]）")
        return True
