"""M4 闭环耦合器：环境 ↔ 神经回路 ↔ 虚拟身体（epoch 迭代，清单 §4.2 #1）。

协议（每 epoch ΔT = 行为 tick Δt_b，CSV body 行定稿）：
    (i)  由当前位姿采样 C、算 s(t)（TimeDiffTracker，τ_win 滑窗差分）；
    (ii) 组帧 epoch 刺激并运行 Brian2 ΔT（`ChemoSession.run_epoch`——
         固定 STIM_WINDOW_MS 形状、显式命名、pad 零——编译缓存纪律，
         epoch 间仅数值变化）；
    (iii) 读三通道肌肉收缩 → 运动学积分（`ChemotaxisBody.step`，引擎无关
         numpy，与 NEURON 行为参考模型共用）更新位姿；
    (iv) 下一 epoch；试次间 store/restore + 重播种（M3 L12 语义）。

确定性：神经网络 p=1/n=1、环境/身体纯 numpy → 同参数重跑逐位一致
（P3 判据）；试次间方差来自伪随机起点扰动（start_jitter，非神经噪声）。

CI（Pierce-Shimomura 1999 象限式）由 `ChemotaxisEnv.ci_per_trial` 计算
（两引擎共用同一统计代码）。
"""

from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.src.chemotaxis_body import ChemotaxisBody  # noqa: E402
from neural_exploration.src.chemotaxis_circuit import (  # noqa: E402
    ChemotaxisCircuit,
    ChemotaxisResult,
    ChemoSession,
)
from neural_exploration.src.chemotaxis_env import (  # noqa: E402
    ChemotaxisEnv,
    TimeDiffTracker,
)


class ChemotaxisLoop:
    """闭环趋化迭代器：环境 ↔ 神经回路 ↔ 虚拟身体（确定性，可重跑）。"""

    def __init__(self, circuit: ChemotaxisCircuit, env: Optional[ChemotaxisEnv] = None,
                 body: Optional[ChemotaxisBody] = None, seed: Optional[int] = None):
        self.circuit = circuit
        p = circuit.params
        # 环境/身体缺省按 CSV 规格构建（env/body 行唯一定稿源）；
        # 机制 A：身体转向事件参数（θ_pir/ω_pir/T_pir）取 CSV mechanism_a 行
        self.env = env or ChemotaxisEnv(
            arena_L=p.env.arena_L, sigma=p.env.sigma, c_max=p.env.c_max,
            c_bg=p.env.c_bg, food_x=p.env.food_x, food_y=p.env.food_y,
            boundary=p.env.boundary)
        self.body = body or ChemotaxisBody(
            v_fwd0=p.body.v_fwd0, omega_max=p.body.omega_max, dt_b=p.body.dt_b,
            v_osc=p.body.v_osc, arena_L=p.env.arena_L, boundary=p.env.boundary,
            turn_omega_pir=p.mech_a.omega_pir, turn_duration_ms=p.mech_a.t_pir_ms)
        self.seed = seed if seed is not None else p.seed

    # ------------------------------------------------------------------ #
    # 单试次（epoch 迭代）
    # ------------------------------------------------------------------ #
    def _session_trial(self, sess: ChemoSession, start_x: float, start_y: float,
                       theta0: float, t_total_ms: float, seed: int,
                       record_extra: Optional[dict] = None) -> ChemotaxisResult:
        p = self.circuit.params
        dt_b = self.body.dt_b
        n_epochs = max(1, int(round(t_total_ms / dt_b)))
        tr = p.transduction
        mech = p.mech_a

        sess.reset(seed=seed)
        self.body.reset(start_x, start_y, theta0)
        tracker = TimeDiffTracker(tr.tau_win_ms, self.env.sample(start_x, start_y))
        # 机制 A：转向方向 = 试次种子确定性伪随机（每试次 RNG 固定 → 可复现；
        # 试次间方向随机——真实虫 pirouette 方向随机，Pierce-Shimomura 1999）
        turn_rng = np.random.default_rng(seed)
        n_turn_events = 0
        turn_epochs: List[int] = []

        xs, ys, thetas, c_sensed = [], [], [], []
        for e in range(n_epochs):
            t_e = e * dt_b
            c_now = self.env.sample(self.body.x, self.body.y)
            s = tracker.s_at(t_e, c_now)
            mus = sess.run_epoch(dt_b, s)
            # 机制 A（清单 §2.4 落地修订，主 agent 裁决 2026-08-23）：
            #   s < −θ_pir（s 调制转向频率）且 ASER→AIB→RIA→SMDD 激活
            #   （本 epoch 内 SMDD 发放——电路耦合）→ 转向事件（持续 T_pir，
            #   ω=±ω_pir，方向=试次种子伪随机）；s>0 → ASEL→AIY 压制（无触发）。
            #   偏置来自直行/转向时长不对称；事件进行中不重叠触发。
            if mech.enabled and not self.body.is_turning():
                if s < -mech.theta_pir and sess.any_spikes_in_window(
                        ("SMDDL", "SMDDR"), t_e, t_e + dt_b):
                    direction = 1.0 if turn_rng.random() < 0.5 else -1.0
                    self.body.trigger_turn(direction, mech.omega_pir, mech.t_pir_ms)
                    n_turn_events += 1
                    turn_epochs.append(e)
            self.body.step(mus["fwd"], mus["left"], mus["right"], dt_b, t_e)
            xs.append(self.body.x)
            ys.append(self.body.y)
            thetas.append(self.body.theta)
            c_sensed.append(c_now)

        xa = np.array(xs, dtype=float)
        ya = np.array(ys, dtype=float)
        # 轨迹有界 + 无 NaN（P3 判据前置；越界/发散立即抛错）
        self.env.assert_bounded(xa, ya)
        self.body.assert_trajectory(xa, ya)
        ci = self.env.ci_per_trial(xa, ya)

        meta_extra = dict(
            start_x=start_x, start_y=start_y, theta0=theta0,
            n_epochs=n_epochs, dt_b_ms=dt_b, ci=ci,
            ci_band_lo=p.protocol.ci_band_lo, ci_band_hi=p.protocol.ci_band_hi,
            c_sensed=np.array(c_sensed, dtype=float),
            n_turn_events=n_turn_events, turn_epochs=turn_epochs,
            turn_dir_seed=seed,
            dist_start_food=float(np.hypot(start_x - self.env.spec.food_x,
                                           start_y - self.env.spec.food_y)),
            dist_end_food=float(np.hypot(xa[-1] - self.env.spec.food_x,
                                         ya[-1] - self.env.spec.food_y)),
        )
        if record_extra:
            meta_extra.update(record_extra)
        return sess.finish(x=xa, y=ya, theta=np.array(thetas, dtype=float),
                           meta_extra=meta_extra)

    def run_trial(self, start_x: Optional[float] = None,
                  start_y: Optional[float] = None, theta0: Optional[float] = None,
                  t_total_ms: Optional[float] = None, seed: Optional[int] = None,
                  record: Optional[Sequence[str]] = None) -> ChemotaxisResult:
        """闭环单试次（自建会话；同参数重跑逐位一致）。"""
        p = self.circuit.params
        t_total = float(t_total_ms or p.protocol.t_total_ms)
        sx = p.protocol.start_x if start_x is None else float(start_x)
        sy = p.protocol.start_y if start_y is None else float(start_y)
        th0 = 0.0 if theta0 is None else float(theta0)
        sess = self.circuit.make_session(t_total_ms=t_total, record=record)
        return self._session_trial(sess, sx, sy, th0, t_total,
                                   seed if seed is not None else self.seed)

    def run_trials(self, n_trials: Optional[int] = None, seed_base: int = 0,
                   t_total_ms: Optional[float] = None,
                   record: Optional[Sequence[str]] = None,
                   start_jitter: Optional[float] = None) -> List[ChemotaxisResult]:
        """闭环多试次：同一会话 store/restore + 重播种；试次方差来自伪随机起点。

        同参数重跑（同 seed_base）→ 逐试次逐位一致（P3 判据）。
        """
        p = self.circuit.params
        n = int(n_trials or p.protocol.n_trials)
        t_total = float(t_total_ms or p.protocol.t_total_ms)
        jitter = float(start_jitter if start_jitter is not None
                       else p.protocol.start_jitter)
        sess = self.circuit.make_session(t_total_ms=t_total, record=record)
        rng = np.random.default_rng(seed_base)
        out = []
        for trial in range(n):
            if jitter > 0:
                sx = p.protocol.start_x + rng.normal(0.0, jitter)
                sy = p.protocol.start_y + rng.normal(0.0, jitter)
                th0 = rng.uniform(0.0, 2.0 * math.pi)
            else:
                sx, sy, th0 = p.protocol.start_x, p.protocol.start_y, 0.0
            out.append(self._session_trial(sess, sx, sy, th0, t_total,
                                           seed_base + trial,
                                           record_extra={"trial": trial}))
        return out

    def run_control(self, n_trials: Optional[int] = None, seed_base: int = 1000,
                    t_total_ms: Optional[float] = None,
                    record: Optional[Sequence[str]] = None,
                    start_jitter: Optional[float] = None) -> List[ChemotaxisResult]:
        """无梯度对照组（P3/P4）：同一协议、C_max 置 0（C ≡ C_bg）。"""
        ctrl_env = self.env.no_gradient()
        loop = ChemotaxisLoop(self.circuit, env=ctrl_env, body=self.body,
                              seed=self.seed)
        return loop.run_trials(n_trials=n_trials, seed_base=seed_base,
                               t_total_ms=t_total_ms, record=record,
                               start_jitter=start_jitter)
