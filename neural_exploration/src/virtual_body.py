"""M5 L5 虚拟身体升级：正弦爬行姿态 + 后退 + 转向（引擎无关 numpy；P6 前提）。

清单《生物仿真M5实施清单》§5.2 #3（身体方程升级规格，P6 前提）::

    v(t) = v_fwd0·clip(C_fwd,0,1) − v_rev0·clip(C_back,0,1)     # 前进 − 后退
    ω(t) = ω_max·(C_left − C_right) + 转向事件项（机制 A 沿用）  # 头摆 → 转向
    正弦爬行：body_y(x, t) = A·sin(2π(x/λ − t/T_gait))（姿态行波；运动学仍用
        点身体+朝向，行波用于头部摆幅 → 转向耦合与可视化，informational→验证级
        由 G0 决定——G0 未提升 → 默认 informational 参数（wave_amp/head_turn_gain））

与 M4 `ChemotaxisBody`（冻结）的关系——本模块在其上扩展，未动冻结文件：
  - 速度方程增加**后退项** v_rev0·C_back（M4 只有 v ≥ 0；C_back 通道来自
    真实连接组 DA/VA/AS → body_back 肌肉驱动，B1a L5 聚合）；
  - 正弦爬行姿态行波（`pose_y`/`head_sway`）；头部摆幅 → 转向耦合
    （`head_turn_gain`，informational 默认 0——G0 定稿不提升为验证级）；
  - `classify_state`：自发行为状态分类（前进/后退/转弯/暂停），**阈值定稿于
    CSV（protocol.spont_v_thr_frac/spont_omega_thr_frac），不做事后调阈值**
    （M3 P5 ×1.2 教训）；行为参考模型共用同一函数（可比性保证，M4 P6 哲学）；
  - 运动学积分（半隐式）/反射边界/轨迹有界检查沿用 M4 语义。

单位约定（M4 body 行定稿 + M5 扩展）：
  - 空间 = 皿边长相对单位（arena_L）；v_fwd0/v_rev0 = 皿单位/s；
    ω_max = rad/s；行为 tick dt_b = ms；gait 周期 T_gait = ms；
    A/λ = 皿单位（姿态波幅/波长，informational 可视化参数）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

#: 状态集合（P6 行为带：Srivastava 2013）
STATES = ("fwd", "rev", "turn", "pause")


@dataclass
class StateThresholds:
    """状态分类阈值（唯一定稿源 = data/m5_worm_params.csv protocol 行）。

    - v_thr = v_thr_frac·v_fwd0（默认 0.05 → CSV spont_v_thr_frac=0.05）；
    - omega_thr = omega_thr_frac·omega_max（默认 0.2 → CSV spont_omega_thr_frac=0.2）。
    定稿后不做事后调阈值（M3 P5 教训）。
    """

    v_thr_frac: float = 0.05
    omega_thr_frac: float = 0.2


def classify_state(
    v: float,
    omega: float,
    c_fwd: Optional[float] = None,
    c_back: Optional[float] = None,
    v_thr_frac: float = 0.05,
    omega_thr_frac: float = 0.2,
    v_fwd0: float = 1.0,
    omega_max: float = 1.0,
) -> str:
    """自发行为状态分类：前进/后退/转弯/暂停（阈值 CSV 定稿，不做事后调）。

    判据（清单 §0 预注册 #2，操作化；v/ω 由身体方程给出）：
      - turn：|ω| > ω_thr（ω_thr = omega_thr_frac·omega_max）；
      - fwd：v > v_thr（v_thr = v_thr_frac·v_fwd0，即净前进）；
      - rev：v < −v_thr（净后退——P6 后退带的前提，需要身体负速度）；
      - pause：其余（|v| ≤ v_thr 且 |ω| ≤ ω_thr）。

    c_fwd/c_back 为可选上下文（肌肉收缩命令，用于诊断/日志；不影响分类——
    v/ω 已由身体方程含入二者）。
    """
    v_thr = v_thr_frac * v_fwd0
    w_thr = omega_thr_frac * omega_max
    if abs(omega) > w_thr:
        return "turn"
    if v > v_thr:
        return "fwd"
    if v < -v_thr:
        return "rev"
    return "pause"


def state_fractions(states: Sequence[str]) -> Dict[str, float]:
    """状态序列 → 时间比例（{fwd, rev, turn, pause}，和为 1）。"""
    n = len(states)
    if n == 0:
        return {s: float("nan") for s in STATES}
    return {s: float(states.count(s)) / n for s in STATES}


class VirtualBody:
    """点身体 + 朝向的运动学积分器（确定性；M4 ChemotaxisBody 的后退/行波扩展）。

    机制 A（pirouette，M4 主 agent 裁决 2026-08-23 沿用）：转向事件由闭环在
    「s < −θ_pir 且 SMDD 发放」时经 `trigger_turn` 启动；事件期间 ω = direction·ω_pir；
    事件外回到左右平衡/竞争：ω = ω_max·(C_left − C_right) + 头摆耦合（informational）。

    正弦爬行姿态（informational，G0 未提升为验证级）：`pose_y(x, t)` 为行进波
    形；`head_sway(t)` 为头部（x = body_len 处）横向摆幅；`head_turn_gain` > 0
    时摆幅变化率耦合进 ω（头部摆动 → 转向，机制验证级开关）。
    """

    def __init__(
        self,
        v_fwd0: float = 1.0,
        v_rev0: float = 1.0,
        omega_max: float = 1.0,
        dt_b: float = 25.0,
        arena_L: float = 10.0,
        boundary: str = "reflect",
        gait_period_ms: float = 500.0,
        wave_amp: float = 0.0,
        wave_lambda: float = 1.0,
        body_len: float = 1.0,
        head_turn_gain: float = 0.0,
        turn_omega_pir: float = 1.0,
        turn_duration_ms: float = 1571.0,
    ):
        self.v_fwd0 = float(v_fwd0)
        self.v_rev0 = float(v_rev0)
        self.omega_max = float(omega_max)
        self.dt_b = float(dt_b)
        self.arena_L = float(arena_L)
        self.boundary = boundary
        # 正弦爬行（informational；G0 未提升为验证级 → 默认 wave_amp=0 关闭）
        self.gait_period_ms = float(gait_period_ms)
        self.wave_amp = float(wave_amp)
        self.wave_lambda = float(wave_lambda)
        self.body_len = float(body_len)
        self.head_turn_gain = float(head_turn_gain)
        # 机制 A 转向事件（M4 沿用）
        self.turn_omega_pir = float(turn_omega_pir)
        self.turn_duration_ms = float(turn_duration_ms)
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.turn_remaining_ms = 0.0
        self.turn_dir = 1.0

    # ------------------------------------------------------------------ #
    # 机制 A 转向事件（M4 语义）
    # ------------------------------------------------------------------ #
    def is_turning(self) -> bool:
        return self.turn_remaining_ms > 0.0

    def trigger_turn(self, direction: float,
                     omega_pir: Optional[float] = None,
                     t_pir_ms: Optional[float] = None):
        if self.turn_remaining_ms > 0.0:
            return
        self.turn_dir = 1.0 if float(direction) >= 0 else -1.0
        if omega_pir is not None:
            self.turn_omega_pir = float(omega_pir)
        if t_pir_ms is not None:
            self.turn_duration_ms = float(t_pir_ms)
        self.turn_remaining_ms = self.turn_duration_ms

    # ------------------------------------------------------------------ #
    # 身体方程（清单 §5.2 #3）
    # ------------------------------------------------------------------ #
    def speed(self, c_fwd: float, c_back: float) -> float:
        """v = v_fwd0·clip(C_fwd,0,1) − v_rev0·clip(C_back,0,1)（后退支持，P6 前提）。"""
        return (self.v_fwd0 * float(np.clip(c_fwd, 0.0, 1.0))
                - self.v_rev0 * float(np.clip(c_back, 0.0, 1.0)))

    def pose_y(self, x: float, t_ms: float) -> float:
        """姿态行波 body_y(x,t) = A·sin(2π(x/λ − t/T_gait))（informational 可视化）。

        返回值 = 身体在体轴坐标 x 处的横向偏移（皿单位；wave_amp=0 时恒 0）。
        """
        if self.wave_amp <= 0.0 or self.gait_period_ms <= 0.0:
            return 0.0
        return (self.wave_amp
                * math.sin(2.0 * math.pi
                           * (x / self.wave_lambda - t_ms / self.gait_period_ms)))

    def head_sway(self, t_ms: float) -> float:
        """头部摆幅 = 行波在 x = body_len 处的横向位移（informational）。"""
        return self.pose_y(self.body_len, t_ms)

    def turn_rate(self, c_left: float, c_right: float, t_ms: float = 0.0) -> float:
        """ω：转向事件期间 = turn_dir·ω_pir；否则 ω_max·(C_left−C_right) + 头摆耦合。

        头摆耦合项（informational，G0 未提升验证级）：head_turn_gain>0 时
        ω += head_turn_gain·d(head_sway)/dt（头部横向摆速 → 转向；数值差分）。
        """
        if self.turn_remaining_ms > 0.0:
            return self.turn_dir * self.turn_omega_pir
        omega = self.omega_max * (float(c_left) - float(c_right))
        if self.head_turn_gain > 0.0:
            dt_s = self.dt_b / 1000.0
            d_sway = (self.head_sway(t_ms + self.dt_b) - self.head_sway(t_ms))
            omega += self.head_turn_gain * (d_sway / dt_s if dt_s > 0 else 0.0)
        return omega

    def _tick_turn_timer(self, dt_s: float):
        if self.turn_remaining_ms > 0.0:
            self.turn_remaining_ms = max(0.0,
                                         self.turn_remaining_ms - dt_s * 1000.0)

    # ------------------------------------------------------------------ #
    # 单步积分（半隐式，M4 语义）
    # ------------------------------------------------------------------ #
    def step(self, c_fwd: float, c_back: float, c_left: float, c_right: float,
             dt_ms: Optional[float] = None, t_ms: float = 0.0) -> Tuple[float, float, float]:
        """一个行为 tick 的运动学积分（半隐式 Euler：先转后行）+ 反射边界。

        Returns (x, y, theta)（self 同步更新）。
        """
        dt = (self.dt_b if dt_ms is None else dt_ms) / 1000.0  # ms → s
        v = self.speed(c_fwd, c_back)
        omega = self.turn_rate(c_left, c_right, t_ms)
        self.theta = self.theta + omega * dt
        self.x = self.x + v * math.cos(self.theta) * dt
        self.y = self.y + v * math.sin(self.theta) * dt
        self._tick_turn_timer(dt)
        self._apply_boundary()
        return self.x, self.y, self.theta

    # ------------------------------------------------------------------ #
    # 边界（P4/P5/P6：轨迹有界）
    # ------------------------------------------------------------------ #
    def _apply_boundary(self):
        L = self.arena_L
        if self.boundary == "reflect":
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
        else:  # "steer"：软转向——位置钳回皿内（确定性保守回退）
            self.x = float(np.clip(self.x, 0.0, L))
            self.y = float(np.clip(self.y, 0.0, L))

    # ------------------------------------------------------------------ #
    # 整条轨迹积分 / 重置 / 有界检查
    # ------------------------------------------------------------------ #
    def integrate(self, c_fwd: Sequence[float], c_back: Sequence[float],
                  c_left: Sequence[float], c_right: Sequence[float],
                  dt_ms: Optional[float] = None) -> Dict[str, np.ndarray]:
        """按四通道肌肉收缩序列积分整条轨迹（每元素 = 一个行为 tick 的收缩）。

        Returns dict(x, y, theta, v, omega, t_ms)
        """
        cf = np.asarray(c_fwd, dtype=float)
        cb = np.asarray(c_back, dtype=float)
        cl = np.asarray(c_left, dtype=float)
        cr = np.asarray(c_right, dtype=float)
        n = min(len(cf), len(cb), len(cl), len(cr))
        dt = self.dt_b if dt_ms is None else dt_ms
        xs, ys, ts_, vs, om = [], [], [], [], []
        for k in range(n):
            v = self.speed(cf[k], cb[k])
            om_k = self.turn_rate(cl[k], cr[k], k * dt)
            self.step(cf[k], cb[k], cl[k], cr[k], dt, k * dt)
            xs.append(self.x); ys.append(self.y)
            ts_.append(k * dt); vs.append(v); om.append(om_k)
        return dict(x=np.array(xs), y=np.array(ys),
                    theta=np.array(ts_, dtype=float), t_ms=np.array(ts_, dtype=float),
                    v=np.array(vs), omega=np.array(om))

    def reset(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0):
        self.x, self.y, self.theta = float(x), float(y), float(theta)
        self.turn_remaining_ms = 0.0
        self.turn_dir = 1.0

    def assert_trajectory(self, x: np.ndarray, y: np.ndarray,
                          tol: float = 1e-6) -> bool:
        """轨迹有界 + 无 NaN 检查（P4/P5/P6 判据前置）。越界/发散抛 ValueError。"""
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

    # ------------------------------------------------------------------ #
    # 便捷：当前位姿的即时状态分类（阈值 CSV 定稿语义）
    # ------------------------------------------------------------------ #
    def classify(self, c_fwd: float, c_back: float, c_left: float, c_right: float,
                 thresholds: Optional[StateThresholds] = None) -> str:
        """由当前肌肉命令即时分类状态（v/ω 按身体方程计算）。"""
        v = self.speed(c_fwd, c_back)
        omega = self.turn_rate(c_left, c_right, 0.0)
        th = thresholds or StateThresholds()
        return classify_state(v, omega, c_fwd, c_back,
                              v_thr_frac=th.v_thr_frac,
                              omega_thr_frac=th.omega_thr_frac,
                              v_fwd0=self.v_fwd0, omega_max=self.omega_max)
