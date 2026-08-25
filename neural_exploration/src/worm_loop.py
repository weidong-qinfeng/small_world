"""M5 全虫闭环 epoch 耦合器（WormLoop）：环境 ↔ 全连接组神经 ↔ 虚拟身体。

照 `chemotaxis_loop.py`（M4 冻结）模式：每 epoch ΔT = 行为 tick Δt_b
（CSV body 行定稿，M4 双时钟纪律）::

    (i)  由当前位姿采样 C、算 s(t)（TimeDiffTracker，τ_win 滑窗差分）；
    (ii) 组帧 epoch 刺激并运行 Brian2 ΔT（`WormSession.run_epoch`——固定形状
         PROTOCOL_WINDOW_MS + 显式命名 + 越界钳位 (1,n) 零数组，M2 L6/M4 L12）；
    (iii) 读肌肉通道（fwd/back/left/right）→ `VirtualBody` 积分（引擎无关 numpy，
         与行为参考模型共用同一运动学/状态分类代码）更新位姿；
    (iv) 下一 epoch；试次间 store/restore + 重播种（M3 L12 语义）。

确定性：神经网络 p=1/n=1、环境/身体纯 numpy → 同参数重跑逐位一致（P2/P4
判据）；试次间方差来自伪随机起点扰动（start_jitter，非神经噪声）。

协议（唯一定稿源 = data/m5_worm_params.csv；本模块解析后覆盖 circuit.params）：
  - 趋化（P4）：食物梯度闭环，CI 复用 `ChemotaxisEnv.ci_per_trial`；
  - 机械逃避（P5）：触电流 I_touch = I0·s_i·1[t0+τ_trans, t0+τ_trans+dur]
    注入 PLM/ALM（τ_trans = escape_touch_delay_ms 定稿于 CSV；缺失默认 0 并
    记录 m5_env_notes L23）；
  - 自发行为（P6）：无刺激无梯度，`classify_state` 状态比例（阈值 CSV 定稿，
    不做事后调）；
  - 静息（P2）：无刺激发放率分布（复用 circuit.run_resting 语义）。

构造参数默认 None（M3 L13）；`WormLoop` 本身无 Brian2 编译（session 由
circuit.make_session 提供）。
"""

from __future__ import annotations

import csv as _csv
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from neural_exploration.src.chemotaxis_body import ChemotaxisBody  # noqa: F401  (兼容引用)
from neural_exploration.src.chemotaxis_circuit import (  # noqa: E402
    ChemotaxisResult,
)
from neural_exploration.src.chemotaxis_env import (  # noqa: E402
    ChemotaxisEnv,
    TimeDiffTracker,
)
from neural_exploration.src.virtual_body import (  # noqa: E402
    StateThresholds,
    VirtualBody,
    classify_state,
    state_fractions,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_WORM_PARAMS_CSV = os.path.join(ROOT, "neural_exploration", "data",
                                       "m5_worm_params.csv")

#: 触刺激角色前缀（P5；connectome 中为 PLML/PLMR/ALML/ALMR）
TOUCH_ROLE_PREFIXES = ("PLM", "ALM")
#: 后退命令运动神经元前缀（P5 神经潜伏期：触刺激开始 → 首个后退命令发放）
BACKWARD_MOTOR_PREFIXES = ("DA", "VA")


def load_m5_worm_params(csv_path: Optional[str] = None) -> Dict[str, dict]:
    """解析 data/m5_worm_params.csv（唯一定稿源）→ 分组字典。

    返回 {"model": {...}, "protocol": {...}, "body": {...}, "transduction": {...},
          "mechanism_a": {...}, "env": {...}, "weight": {...}, "behavior_reference": {...},
          "g0": {...}}；每项 = {subkey: value}（value 数值型转 float，其余 str）。

    列 schema（B1b 定稿）：role, neuron_class, synapse_from, synapse_to,
    synapse_type, g_max_ns, delay_ms, tonic_uA_cm2, value, note。
    ⚠ 实测 schema 不一致（M5-B1d，L23）：列头 10 列（value 在 fields[8]），但
    数据行 **11+ 字段**（fields[2..8] 共 7 个空列 + value 在 fields[9] + note 在
    fields[10:]）——DictReader 会把 value 读成空、真值落进 note。→ 本模块按
    **位置**解析（value = fields[9]，len<11 行视为无 value——仅 model 描述行）。
    """
    path = csv_path or DEFAULT_WORM_PARAMS_CSV
    out: Dict[str, dict] = {}

    def _clean(ln: str) -> str:
        s = ln.strip()
        if s.startswith('"'):
            s = s.strip('"')
        return s

    with open(path, newline="", encoding="utf-8") as f:
        for ln in f:
            s = _clean(ln)
            if not s or s.startswith("#"):
                continue
            fields = next(_csv.reader([s]))
            role = (fields[0] if fields else "").strip().lower()
            key = (fields[1] if len(fields) > 1 else "").strip().lower()
            if role in ("", "role"):
                continue
            if not key:
                continue
            # 位置解析（见 docstring 的 schema 说明）：value=fields[9]、note=fields[10:]
            value = (fields[9] if len(fields) >= 11 else "")
            out.setdefault(role, {})[key] = _to_val(value)
    return out


def _to_val(raw) -> object:
    if raw is None:
        return ""
    s = str(raw).strip()
    try:
        return float(s)
    except ValueError:
        return s


class WormLoop:
    """全虫闭环迭代器：环境 ↔ 神经回路 ↔ VirtualBody（确定性，可重跑）。

    协议参数（body/protocol/transduction/mechanism_a）优先取
    data/m5_worm_params.csv（唯一定稿源），缺省回退 circuit.params（M4 源）。
    """

    def __init__(self, circuit, env: Optional[ChemotaxisEnv] = None,
                 body: Optional[VirtualBody] = None, seed: Optional[int] = None,
                 params_csv: Optional[str] = None):
        self.circuit = circuit
        p = circuit.params
        wp = load_m5_worm_params(params_csv)
        self.wp = wp
        body_row = wp.get("body", {})
        env_row = wp.get("env", {})
        tr_row = wp.get("transduction", {})
        mech_row = wp.get("mechanism_a", {})
        prot_row = wp.get("protocol", {})

        # 环境（M4 语义；CSV env 行优先）
        self.env = env or ChemotaxisEnv(
            arena_L=float(env_row.get("arena_L", p.env.arena_L)),
            sigma=float(env_row.get("sigma", p.env.sigma)),
            c_max=float(env_row.get("C_max", p.env.c_max)),
            c_bg=float(env_row.get("C_bg", p.env.c_bg)),
            food_x=float(env_row.get("food_x", p.env.food_x)),
            food_y=float(env_row.get("food_y", p.env.food_y)),
            boundary=str(env_row.get("boundary", p.env.boundary)))
        # 身体（M5 扩展：后退 v_rev0·C_back + 正弦行波；CSV body 行定稿）
        self.body = body or VirtualBody(
            v_fwd0=float(body_row.get("v_fwd0", p.body.v_fwd0)),
            v_rev0=float(body_row.get("v_rev0", 1.0)),
            omega_max=float(body_row.get("omega_max", p.body.omega_max)),
            dt_b=float(body_row.get("dt_b", p.body.dt_b)),
            arena_L=self.env.spec.arena_L, boundary=self.env.spec.boundary,
            gait_period_ms=float(body_row.get("gait_period_ms", 500.0)),
            wave_amp=float(body_row.get("wave_amp", 0.0)),
            wave_lambda=float(body_row.get("wave_lambda", 1.0)),
            head_turn_gain=float(body_row.get("head_turn_gain", 0.0)),
            turn_omega_pir=float(mech_row.get("omega_pir",
                                              p.mech_a.omega_pir)),
            turn_duration_ms=float(mech_row.get("t_pir_ms",
                                                p.mech_a.t_pir_ms)))
        self.seed = seed if seed is not None else int(p.seed)

        # 状态分类阈值（CSV protocol 行定稿；不做事后调——M3 P5 教训）
        self.v_thr_frac = float(prot_row.get("spont_v_thr_frac", 0.05))
        self.omega_thr_frac = float(prot_row.get("spont_omega_thr_frac", 0.2))
        self.thresholds = StateThresholds(v_thr_frac=self.v_thr_frac,
                                          omega_thr_frac=self.omega_thr_frac)

        # 机械刺激协议（P5，CSV protocol 行定稿；τ_trans = escape_touch_delay_ms
        # —— B1b 定稿 CSV 暂无此行 → 默认 0.0 并记录 m5_env_notes L23）
        self.touch = dict(
            site=str(prot_row.get("escape_touch_site", "soma")),
            i0_uA_cm2=float(prot_row.get("escape_i0_uA_cm2", 60.0)),
            start_ms=float(prot_row.get("escape_touch_start_ms", 50.0)),
            dur_ms=float(prot_row.get("escape_touch_dur_ms", 5.0)),
            tau_trans_ms=float(prot_row.get("escape_touch_delay_ms", 0.0)),
            t_total_ms=float(prot_row.get("escape_t_total_ms", 150.0)),
        )

    # ------------------------------------------------------------------ #
    # 单试次（epoch 迭代；ChemotaxisLoop._session_trial 语义 + VirtualBody）
    # ------------------------------------------------------------------ #
    def _session_trial(self, sess, start_x: float, start_y: float,
                       theta0: float, t_total_ms: float, seed: int,
                       record_extra: Optional[dict] = None,
                       s_override: Optional[float] = None) -> ChemotaxisResult:
        p = self.circuit.params
        dt_b = self.body.dt_b
        n_epochs = max(1, int(round(t_total_ms / dt_b)))
        tr = p.transduction
        mech = p.mech_a

        sess.reset(seed=seed)
        self.body.reset(start_x, start_y, theta0)
        tracker = TimeDiffTracker(tr.tau_win_ms, self.env.sample(start_x, start_y))
        turn_rng = np.random.default_rng(seed)  # 机制 A 方向 = 试次种子确定性伪随机
        n_turn_events = 0
        turn_epochs: List[int] = []

        xs, ys, thetas, vs, omegas, states, c_sensed = [], [], [], [], [], [], []
        for e in range(n_epochs):
            t_e = e * dt_b
            c_now = self.env.sample(self.body.x, self.body.y)
            s = tracker.s_at(t_e, c_now) if s_override is None else s_override
            mus = sess.run_epoch(dt_b, s)
            # 机制 A（M4 语义）：s < −θ_pir 且 SMDD 发放 → 转向事件
            if mech.enabled and not self.body.is_turning():
                if s < -mech.theta_pir and sess.any_spikes_in_window(
                        ("SMDDL", "SMDDR"), t_e, t_e + dt_b):
                    direction = 1.0 if turn_rng.random() < 0.5 else -1.0
                    self.body.trigger_turn(direction, mech.omega_pir,
                                           mech.t_pir_ms)
                    n_turn_events += 1
                    turn_epochs.append(e)
            c_fwd = float(mus.get("fwd", 0.0))
            c_back = float(mus.get("back", 0.0))
            c_left = float(mus.get("left", 0.0))
            c_right = float(mus.get("right", 0.0))
            v = self.body.speed(c_fwd, c_back)
            omega = self.body.turn_rate(c_left, c_right, t_e)
            st = classify_state(v, omega, c_fwd, c_back,
                                self.v_thr_frac, self.omega_thr_frac,
                                self.body.v_fwd0, self.body.omega_max)
            self.body.step(c_fwd, c_back, c_left, c_right, dt_b, t_e)
            xs.append(self.body.x); ys.append(self.body.y)
            thetas.append(self.body.theta)
            vs.append(v); omegas.append(omega); states.append(st)
            c_sensed.append(c_now)

        xa = np.array(xs, dtype=float)
        ya = np.array(ys, dtype=float)
        # 轨迹有界 + 无 NaN（P4/P5/P6 判据前置）
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
            v=np.array(vs, dtype=float), omega=np.array(omegas, dtype=float),
            states=states,
            state_frac=state_fractions(states),
            classify_thresholds=dict(v_thr_frac=self.v_thr_frac,
                                     omega_thr_frac=self.omega_thr_frac),
            dist_start_food=float(np.hypot(start_x - self.env.spec.food_x,
                                           start_y - self.env.spec.food_y)),
            dist_end_food=float(np.hypot(xa[-1] - self.env.spec.food_x,
                                         ya[-1] - self.env.spec.food_y)),
            scale=self.circuit.scale, fidelity=self.circuit.fidelity,
            dt_ms=self.circuit.dt_ms, method=self.circuit.method,
        )
        if record_extra:
            meta_extra.update(record_extra)
        return sess.finish(x=xa, y=ya, theta=np.array(thetas, dtype=float),
                           meta_extra=meta_extra)

    # ------------------------------------------------------------------ #
    # 趋化闭环（P4）
    # ------------------------------------------------------------------ #
    def run_trial(self, start_x: Optional[float] = None,
                  start_y: Optional[float] = None, theta0: Optional[float] = None,
                  t_total_ms: Optional[float] = None, seed: Optional[int] = None,
                  s_override: Optional[float] = None) -> ChemotaxisResult:
        """闭环单试次（自建会话；同参数重跑逐位一致）。"""
        p = self.circuit.params
        t_total = float(t_total_ms or p.protocol.t_total_ms)
        sx = p.protocol.start_x if start_x is None else float(start_x)
        sy = p.protocol.start_y if start_y is None else float(start_y)
        th0 = 0.0 if theta0 is None else float(theta0)
        sess = self.circuit.make_session(t_total_ms=t_total)
        return self._session_trial(sess, sx, sy, th0, t_total,
                                   seed if seed is not None else self.seed,
                                   s_override=s_override)

    def run_trials(self, n_trials: Optional[int] = None, seed_base: int = 0,
                   t_total_ms: Optional[float] = None,
                   start_jitter: Optional[float] = None,
                   s_override: Optional[float] = None) -> List[ChemotaxisResult]:
        """闭环多试次：同一会话 store/restore + 重播种；方差来自伪随机起点。

        同参数重跑（同 seed_base）→ 逐试次逐位一致（确定性判据）。
        """
        p = self.circuit.params
        n = int(n_trials or p.protocol.n_trials)
        t_total = float(t_total_ms or p.protocol.t_total_ms)
        jitter = float(start_jitter if start_jitter is not None
                       else p.protocol.start_jitter)
        sess = self.circuit.make_session(t_total_ms=t_total)
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
                                           record_extra={"trial": trial},
                                           s_override=s_override))
        return out

    def run_control(self, n_trials: Optional[int] = None, seed_base: int = 1000,
                    t_total_ms: Optional[float] = None,
                    start_jitter: Optional[float] = None) -> List[ChemotaxisResult]:
        """无梯度对照（C_max 置 0 → C ≡ C_bg；P4 主判据语义，M4 同款）。"""
        ctrl_env = self.env.no_gradient()
        loop = WormLoop(self.circuit, env=ctrl_env, body=self.body,
                        seed=self.seed)
        return loop.run_trials(n_trials=n_trials, seed_base=seed_base,
                               t_total_ms=t_total_ms,
                               start_jitter=start_jitter)

    # ------------------------------------------------------------------ #
    # 自发行为（P6）：无刺激无梯度，classify_state 状态比例
    # ------------------------------------------------------------------ #
    def run_spontaneous(self, t_total_ms: Optional[float] = None,
                        seed: Optional[int] = None) -> Dict:
        """无刺激无梯度 T 窗：状态比例 + 状态序列 + v/ω 序列（阈值 CSV 定稿）。

        使用无梯度环境（C ≡ C_bg → s ≡ 0 → ASE 无注入）+ s_override=0，
        仅网络自发活动驱动肌肉 → classify_state（P6 判据输入）。
        """
        t_total = float(t_total_ms or float(self.wp.get("protocol", {}).get(
            "spont_t_total_ms", 5000.0)))
        seed = self.seed if seed is None else int(seed)
        ctrl = self.env.no_gradient()
        loop = WormLoop(self.circuit, env=ctrl, body=self.body, seed=seed)
        res = loop.run_trial(t_total_ms=t_total, seed=seed, s_override=0.0)
        return dict(
            frac=res.meta["state_frac"], states=res.meta["states"],
            v=res.meta["v"], omega=res.meta["omega"],
            n_epochs=res.meta["n_epochs"],
            classify_thresholds=res.meta["classify_thresholds"],
            x=res.x, y=res.y, wall_s=None,
        )

    # ------------------------------------------------------------------ #
    # 机械逃避（P5）：触电流注入 PLM/ALM → 后退方向
    # ------------------------------------------------------------------ #
    def touch_window(self) -> Tuple[int, int, int]:
        """触刺激窗（协议步数，供 stim 注入）：[start+τ_trans, start+τ_trans+dur]。

        Returns (i0, i1, n_steps_total)——i0/i1 为 dt 步索引（含 τ_trans，
        P5 转导延迟语义）；n_steps_total 由 PROTOCOL_WINDOW_MS/dt 决定
        （越界钳位在注入侧处理）。纯 numpy/协议层，引擎无关可测。
        """
        dt = self.circuit.dt_ms
        t0 = self.touch["start_ms"] + self.touch["tau_trans_ms"]
        i0 = int(round(t0 / dt))
        i1 = int(round((t0 + self.touch["dur_ms"]) / dt))
        n_steps = int(round(max(500.0, 6000.0) / dt))  # PROTOCOL_WINDOW_MS 语义
        return i0, i1, n_steps

    def run_escape(self, t_total_ms: Optional[float] = None,
                   seed: Optional[int] = None,
                   touch_roles: Optional[Sequence[str]] = None,
                   backward_roles: Optional[Sequence[str]] = None) -> Dict:
        """机械刺激短协议（P5，全虫/命令子图路径）：触电流
        I0·s_i·1[t0+τ_trans, t0+τ_trans+dur] 注入 touch_roles
        （默认 circuit.names 中 PLM*/ALM* 角色）→ 无梯度 epoch →
        C_back/C_fwd 序列 → 方向（D_peak>0.3 → back）。

        touch_roles/backward_roles 为可选角色清单；None → 按前缀
        （TOUCH_ROLE_PREFIXES / BACKWARD_MOTOR_PREFIXES）在 circuit.names
        中自动匹配。返回 dict（d_peak/direction/c_back/c_fwd/neural_latency_ms/
        touch_roles/wall_s）。

        note：302 全虫或命令子图（真实接线）下 PLM→AVM 为缝隙耦合（L8：
        权威连接组无该直接化学边），P5 验证节点需按子图/缝隙路径核对方向；
        冒烟用 M3 反射子图（ReflexCircuit，G0 已验证方向 back）另测。
        """
        circ = self.circuit
        t_total = float(t_total_ms or self.touch["t_total_ms"])
        seed = self.seed if seed is None else int(seed)
        names = [str(n).upper() for n in circ.names]
        roles = [str(r).upper() for r in (touch_roles or ())
                 if str(r).upper() in names]
        if not roles:
            roles = [n for n in names
                     if n.startswith(TOUCH_ROLE_PREFIXES)]
        if not roles:
            raise ValueError(
                f"无触刺激角色：touch_roles={touch_roles} 且 circuit.names 中"
                f"无 {TOUCH_ROLE_PREFIXES}*（{len(names)} 角色）")
        bwd = [str(r).upper() for r in (backward_roles or ())
               if str(r).upper() in names]
        if not bwd:
            bwd = [n for n in names if n.startswith(BACKWARD_MOTOR_PREFIXES)]

        import time
        t0 = time.perf_counter()
        sess = circ.make_session(t_total_ms=t_total)
        sess.reset(seed=seed)

        # 触电流注入（τ_trans 后开始；越界钳位到固定窗口）
        i0, i1, n_steps = self.touch_window()
        i_nA = (self.touch["i0_uA_cm2"] * 1e-6 * 1.257e-5 * 1e9)  # µA/cm² → nA
        grouped = hasattr(sess, "stim")  # GroupedWormSession: 单组 stim (n_steps, N)
        stim_ta = sess.stim if grouped else None
        n_steps_actual = (stim_ta.values.shape[0] if grouped
                          else int(round(max(500.0, 6000.0) / circ.dt_ms)))
        i0 = max(0, min(i0, n_steps_actual))
        i1 = max(i0, min(i1, n_steps_actual))
        for r in roles:
            if grouped:
                idx = circ.role_index.get(r)
                stim_ta.values[i0:i1, idx] = i_nA * 1e-9
            else:
                ta = sess.stims[f"stim_{r.lower()}"]
                idx = circ.neurons[r].label_of("soma")
                ta.values[i0:i1, idx] = i_nA * 1e-9

        # 无梯度 epoch 运行 → 肌肉序列
        n_epochs = max(1, int(round(t_total / self.body.dt_b)))
        cbs, cfs, t_ms = [], [], []
        for e in range(n_epochs):
            mus = sess.run_epoch(self.body.dt_b, 0.0)
            cbs.append(float(mus.get("back", 0.0)))
            cfs.append(float(mus.get("fwd", 0.0)))
            t_ms.append(e * self.body.dt_b)
        c_back = np.asarray(cbs, dtype=float)
        c_fwd = np.asarray(cfs, dtype=float)
        d_peak = float(np.max(c_back - c_fwd)) if c_back.size else 0.0

        # 神经潜伏期：触刺激开始（t0，含转导前）→ 首个后退命令运动神经元发放
        # （role_spike_times 已返回 ms 数组）
        lat = float("nan")
        if bwd:
            t_arr = np.asarray(sess.role_spike_times().get(bwd[0], []),
                               dtype=float)
            if t_arr.size:
                lat = float(t_arr[0] - self.touch["start_ms"])
        wall = time.perf_counter() - t0
        return dict(
            d_peak=d_peak,
            direction=("back" if d_peak > 0.3 else "not_back"),
            c_back=c_back, c_fwd=c_fwd, t_ms=np.asarray(t_ms, dtype=float),
            neural_latency_ms=lat, touch_roles=roles, backward_roles=bwd,
            touch=self.touch, wall_s=wall, t_total_ms=t_total,
        )

    # ------------------------------------------------------------------ #
    # 静息（P2）：无刺激发放率分布
    # ------------------------------------------------------------------ #
    def run_resting(self, t_total_ms: Optional[float] = None,
                    seed: Optional[int] = None) -> Dict:
        """无刺激 T 窗：逐神经元发放率 + 静默比例 + 稳定性（P2 判据输入）。

        委托 circuit.run_resting（同语义；session 独立构建）。
        """
        t_total = float(t_total_ms or float(self.wp.get("protocol", {}).get(
            "resting_t_total_ms", 10000.0)))
        return self.circuit.run_resting(t_total_ms=t_total,
                                        seed=seed if seed is not None
                                        else self.seed)
