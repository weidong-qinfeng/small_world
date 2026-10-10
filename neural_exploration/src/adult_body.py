"""M9 虚拟成年果蝇身体（清单 §3.1/§3.2/§3.3；`src/adult_body.py`，新建）。

《生物仿真M9实施清单》§3.1（规格：六足 3 关节 ×6 = 18 驱动通道 + 翅 + 触角/复眼 +
**全行为通道肌肉映射（含 curl —— M8 R3 承接，非 provisional）** + 状态分类阈值 CSV 定稿）/
§3.2（VR 球上行走协议复刻：球转速度 → 视觉流 → 复眼 → 行为输出 → 球转更新闭环）/
§3.3（身体层验证：单通道冒烟 + 巨纤维逃跑 sanity resp ≥0.8 + 确定性）。

**冻结件零修改**：`src/virtual_body.py`（M4 冻结基类）与 `src/larva_body.py`（M8 冻结）只读参考，
本模块独立实现成虫身体（六足 tripod 步态 + 翅 + 感觉输入 + VR 闭环）。

设计（确定性、无随机性；所有量有界、无 NaN）：
  - **六足步态**：tripod 为主 —— 组 A{L1,R2,L3} 与组 B{R1,L2,R3} 相位差 180°（对角三足同相）；
    每足 3 关节（coxa 摆/femur 抬/tibia 屈伸）×6 = 18 通道；
    swing/stance 由占空比 duty_factor 划分（默认 0.5）；
  - **位移**：stance 足尖相对体轴后移 → 身体前移（`v_body > 0`）；C_leg_back 反相 → 后退；
    C_leg_left − C_leg_right 差动 → 转向（coxa 摆幅差动）；
  - **翅**：~200 Hz 翅拍（按 body dt 采样 → **测量限制**）；翅歌调制、逃跑/求偶翅展；
  - **感觉**：触角 ORN 一阶转导（M8 惯例升级）+ 机械感受；复眼视觉流（球转速度 → 光流）；
  - **VR 球（§3.2）**：tethered 范式闭环 —— 行为腿驱动 → 球转速度 → 视觉流 → 复眼 →
    CX EPG 输入通道（供 P6 航向闭环）；
  - **巨纤维逃跑（§3.3 / P-EXT1 机制前置）**：光/机械刺激超阈值 → 潜伏期 → escape 态
    （跳跃预备 + 翅展）；不应期内不重复触发；亚阈值刺激不触发（剂量-反应 sanity）；
  - **状态分类**：escape > curl > turn > run > pause（优先级固定于 CSV），阈值定稿于
    `data/m9_adult_body_params.csv`（不做事后调）。

用法：
    from neural_exploration.src.adult_body import AdultFlyBody, load_adult_body_params
    b = AdultFlyBody(); b.reset()
    out = b.run(drives={"C_leg_fwd": 1.0}, t_ms=1000.0)
"""

from __future__ import annotations

import csv as _csv
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
BODY_PARAMS_CSV = os.path.join(DATA_DIR, "m9_adult_body_params.csv")

#: 肌肉/行为通道（§3.1：全行为通道映射，含 curl —— M8 R3 承接）
CHANNELS = ("C_leg_fwd", "C_leg_back", "C_leg_left", "C_leg_right",
            "C_curl", "C_wing", "C_escape")

#: 六足以 tripod 分组（对角三足同相）
LEGS = ("L1", "L2", "L3", "R1", "R2", "R3")
TRIPOD_A = ("L1", "R2", "L3")
TRIPOD_B = ("R1", "L2", "R3")

STATES = ("run", "turn", "pause", "curl", "escape")


# ---------------------------------------------------------------------------
def load_adult_body_params(csv_path: Optional[str] = None) -> Dict[str, float]:
    """读 `data/m9_adult_body_params.csv`（唯一参数定稿源；value 在 fields[9]，位置解析）。"""
    path = csv_path or BODY_PARAMS_CSV
    out: Dict[str, float] = {}
    if not os.path.exists(path):
        raise FileNotFoundError("身体参数 CSV 不存在：%s" % path)
    with open(path, encoding="utf-8") as f:
        for row in _csv.reader(f):
            if not row or row[0].startswith("#") or row[0] == "role":
                continue
            if len(row) < 10:
                continue
            try:
                out[row[1]] = float(row[9])
            except ValueError:
                continue
    return out


@dataclass
class AdultBodyParams:
    """身体参数（默认值 = CSV 定稿值；CSV 为唯一权威）。"""
    n_legs: int = 6
    joints_per_leg: int = 3
    gait_freq_hz: float = 7.0
    tripod_phase_deg: float = 180.0
    swing_amp_deg: float = 25.0
    lift_amp_deg: float = 18.0
    tibia_amp_deg: float = 20.0
    coxa_bias_deg: float = 60.0
    femur_bias_deg: float = 90.0
    tibia_bias_deg: float = 90.0
    leg_len_mm: float = 2.5
    body_len_mm: float = 3.0
    duty_factor: float = 0.5
    wing_freq_hz: float = 200.0
    wing_amp_deg: float = 60.0
    wing_song_freq_hz: float = 30.0
    wing_spread_deg: float = 80.0
    orn_tau_ms: float = 50.0
    orn_gain: float = 1.0
    mech_gain: float = 1.0
    eye_flow_gain: float = 1.0
    ball_diameter_mm: float = 6.0
    ball_gain: float = 1.0
    v_run_frac: float = 0.15
    omega_turn_frac: float = 0.20
    curl_thr: float = 0.30
    escape_thr: float = 0.50
    escape_latency_ms: float = 6.0
    escape_refractory_ms: float = 80.0
    v_fwd0: float = 10.0      # 参考前进速度（mm/s；由步态推得，见 run()）
    omega_max: float = 3.0    # 参考最大角速度（rad/s）
    dt_ms: float = 1.0        # 身体积分步长（翅拍 200Hz → 1ms 采样，测量限制）

    @classmethod
    def from_csv(cls, path: Optional[str] = None) -> "AdultBodyParams":
        d = load_adult_body_params(path)
        p = cls()
        for k, v in d.items():
            if hasattr(p, k):
                cur = getattr(p, k)
                setattr(p, k, int(v) if isinstance(cur, int) else float(v))
        return p


# ---------------------------------------------------------------------------
class AdultFlyBody:
    """成虫虚拟身体：六足 tripod 步态 + 翅 + 触角/复眼 + VR 球闭环 + 巨纤维逃跑。"""

    def __init__(self, params: Optional[AdultBodyParams] = None, seed: int = 0,
                 arena_mm: float = 50.0):
        self.p = params or AdultBodyParams.from_csv()
        self.seed = int(seed)
        self.arena_mm = float(arena_mm)
        self.reset()

    # ---------------- 状态 ----------------
    def reset(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0) -> None:
        p = self.p
        self.t_ms = 0.0
        self.x, self.y, self.theta = float(x), float(y), float(theta)
        self.v = 0.0                 # mm/s（体轴方向）
        self.omega = 0.0             # rad/s
        self.curl = 0.0              # 0..1（防御蜷缩）
        self.wing_spread = 0.0       # 0..1
        self.escape_timer = 0.0      # ms 剩余逃跑时程
        self.escape_refractory = 0.0  # ms 剩余不应期
        self.n_escape = 0
        self.orn = 0.0               # ORN 转导状态
        self.phase0 = 0.0            # CPG 相位（确定性：固定初相）
        self.last_state = "pause"

    # ---------------- 步态 / 关节 ----------------
    def cpg_phase(self, leg: str, t_ms: float) -> float:
        """该足 CPG 相位（rad）。tripod：组 B 相对组 A 偏 180°。"""
        p = self.p
        phase = 2.0 * math.pi * p.gait_freq_hz * (t_ms * 1e-3) + self.phase0
        if leg in TRIPOD_B:
            phase += math.radians(p.tripod_phase_deg)
        return phase

    def joint_angles(self, t_ms: float, drives: Dict[str, float]) -> np.ndarray:
        """18 关节角（deg；行=足 L1..R3，列=coxa/femur/tibia）。

        swing/stance：以 `sin(φ)` 划分 —— `sin(φ) > 0` 为 swing（抬腿+前摆），
        否则 stance（着地+后蹬）。`C_leg_back` 使相位反相（后退步态）。
        """
        p = self.p
        fwd = float(drives.get("C_leg_fwd", 0.0))
        back = float(drives.get("C_leg_back", 0.0))
        dleft = float(drives.get("C_leg_left", 0.0))
        dright = float(drives.get("C_leg_right", 0.0))
        curl = float(drives.get("C_curl", 0.0))
        scale = max(fwd, back)                     # 步态幅度（0 → 站立）
        sgn = 1.0 if fwd >= back else -1.0         # 前进 / 后退
        turn = dleft - dright                      # 差动转向
        out = np.empty((p.n_legs, p.joints_per_leg), dtype=np.float64)
        for i, leg in enumerate(LEGS):
            ph = self.cpg_phase(leg, t_ms) * sgn
            s = math.sin(ph)
            c = math.cos(ph)
            side = 1.0 if leg.startswith("L") else -1.0
            # 转向差动：同侧加/减摆幅（L 侧正差动 → 左转）
            turn_term = 1.0 + 0.5 * turn * side
            coxa = (p.coxa_bias_deg
                    + p.swing_amp_deg * scale * c * turn_term * side)
            femur = p.femur_bias_deg + p.lift_amp_deg * scale * max(0.0, s)
            tibia = p.tibia_bias_deg - p.tibia_amp_deg * scale * max(0.0, s)
            # 蜷缩：六足向体侧内收 + 屈曲（有界饱和）
            coxa -= 0.4 * p.swing_amp_deg * curl * side
            femur -= 0.5 * p.lift_amp_deg * curl
            tibia += 0.6 * p.tibia_amp_deg * curl
            out[i, 0] = coxa
            out[i, 1] = femur
            out[i, 2] = tibia
        return out

    def leg_tip(self, leg: str, t_ms: float, drives: Dict[str, float]) -> np.ndarray:
        """足尖相对胸部的坐标（mm；x 前、y 右、z 上）——运动学正解（简化连杆）。"""
        p = self.p
        i = LEGS.index(leg)
        ang = self.joint_angles(t_ms, drives)[i]
        side = 1.0 if leg.startswith("L") else -1.0
        L = p.leg_len_mm / 3.0
        c0, f0, t0 = (math.radians(a) for a in ang)
        reach = L * (math.cos(c0) + math.cos(c0 + f0) * 0.6 + math.cos(c0 + f0 + t0) * 0.4)
        lift = L * (math.sin(f0) * 0.6 + math.sin(f0 + t0) * 0.4)
        stance = 1.0 if math.sin(self.cpg_phase(leg, t_ms)) <= 0.0 else 0.0
        return np.array([reach, side * (0.8 * L + 0.2 * L * math.cos(c0)), lift + stance * 0.0])

    def stance_legs(self, t_ms: float) -> Tuple[str, ...]:
        """当前着地（stance）足集合。"""
        return tuple(l for l in LEGS if math.sin(self.cpg_phase(l, t_ms)) <= 0.0)

    def body_speed_from_gait(self, drives: Dict[str, float]) -> float:
        """由 stance 足后蹬速率估计体速（mm/s；正 = 前向）。"""
        p = self.p
        fwd = float(drives.get("C_leg_fwd", 0.0))
        back = float(drives.get("C_leg_back", 0.0))
        scale = max(fwd, back)
        sgn = 1.0 if fwd >= back else -1.0
        # 步幅 ∝ swing_amp·leg_len；速率 = 步幅 × 步频 × 每周期 stance 足数占比
        stride = math.radians(p.swing_amp_deg) * p.leg_len_mm * 2.0
        n_stance = len(self.stance_legs(self.t_ms)) / p.n_legs
        base = stride * p.gait_freq_hz * n_stance * scale
        return sgn * base * (1.0 - float(drives.get("C_curl", 0.0)))

    # ---------------- 翅 ----------------
    def wing_signal(self, t_ms: float, drives: Dict[str, float]) -> Tuple[float, float]:
        """（翅拍角 deg，翅展 0..1）。翅拍 ~200Hz（按 dt 采样 → 测量限制）。"""
        p = self.p
        cwing = float(drives.get("C_wing", 0.0))
        cescape = float(drives.get("C_escape", 0.0))
        beat = p.wing_amp_deg * math.sin(2.0 * math.pi * p.wing_freq_hz * t_ms * 1e-3)
        song = 1.0 + 0.3 * math.sin(2.0 * math.pi * p.wing_song_freq_hz * t_ms * 1e-3)
        amp = beat * (1.0 + cwing * song)
        spread = min(1.0, max(0.0, cwing + max(cescape, float(self.escape_timer > 0.0))))
        return amp, spread

    # ---------------- 感觉输入 ----------------
    def antenna_orn(self, odor: float, mech: float, dt_ms: float) -> float:
        """触角 ORN 一阶转导（M8 幼虫转导惯例升级）+ 机械感受。"""
        p = self.p
        tau = max(p.orn_tau_ms, 1e-6)
        target = p.orn_gain * odor + p.mech_gain * mech
        self.orn += (target - self.orn) * (dt_ms / tau)
        return self.orn

    def eye_flow(self, ball_omega: float, ball_v: float) -> np.ndarray:
        """复眼视觉流（方位流, 转速流）：球转速度 → 光流。"""
        p = self.p
        circ = math.pi * p.ball_diameter_mm
        return np.array([p.eye_flow_gain * p.ball_gain * ball_v / max(circ, 1e-9),
                         p.eye_flow_gain * p.ball_gain * ball_omega])

    def cx_heading_input(self, ball_omega: float, ball_v: float) -> np.ndarray:
        """→ CX EPG 输入通道（§3.2 航向闭环接口；供 P6 使用）。"""
        f = self.eye_flow(ball_omega, ball_v)
        return np.array([f[0], f[1], self.theta])   # (translational flow, rotational, heading)

    # ---------------- 逃跑 ----------------
    def giant_fiber_stimulus(self, light: float, mech: float) -> bool:
        """光/机械刺激 → 巨纤维通路超阈值判定（确定性；不应期内不触发）。"""
        p = self.p
        if self.escape_refractory > 0.0:
            return False
        drive = max(float(light), float(mech))
        if drive >= p.escape_thr:
            self.escape_timer = p.escape_latency_ms + 40.0   # 潜伏期 + 逃跑时程
            self.escape_refractory = p.escape_refractory_ms
            self.n_escape += 1
            return True
        return False

    # ---------------- 状态分类 ----------------
    def classify(self, drives: Dict[str, float]) -> str:
        """状态分类（优先级 escape > curl > turn > run > pause；阈值 CSV 定稿不事后调）。"""
        p = self.p
        if self.escape_timer > 0.0:
            return "escape"
        if float(drives.get("C_curl", 0.0)) >= p.curl_thr:
            return "curl"
        if abs(self.omega) > p.omega_turn_frac * p.omega_max:
            return "turn"
        if self.v > p.v_run_frac * p.v_fwd0:
            return "run"
        if self.v < -p.v_run_frac * p.v_fwd0:
            return "run"          # 后退并入 run（M8 note_rev_in_turn 语义的对偶：此处后退=负向行进）
        return "pause"

    # ---------------- 积分 ----------------
    def step(self, drives: Dict[str, float], dt_ms: Optional[float] = None) -> Dict[str, Any]:
        """单步积分（确定性、有界）。"""
        p = self.p
        dt = float(dt_ms if dt_ms is not None else p.dt_ms)
        self.t_ms += dt
        # 通道限幅（有界）
        d = {k: float(np.clip(drives.get(k, 0.0), 0.0, 1.0)) for k in CHANNELS}
        # 逃跑时程 / 不应期倒计时
        if self.escape_timer > 0.0:
            self.escape_timer = max(0.0, self.escape_timer - dt)
            d["C_escape"] = 1.0
        if self.escape_refractory > 0.0:
            self.escape_refractory = max(0.0, self.escape_refractory - dt)
        # 蜷缩动力学（一阶趋近目标，有界）
        tgt_curl = d["C_curl"]
        self.curl += (tgt_curl - self.curl) * min(1.0, dt / 50.0)
        self.curl = float(np.clip(self.curl, 0.0, 1.0))
        # 速度：步态给出平移；差动给出转向
        v_tgt = self.body_speed_from_gait(d)
        self.v += (v_tgt - self.v) * min(1.0, dt / 20.0)
        turn = d["C_leg_left"] - d["C_leg_right"]
        w_tgt = p.omega_max * turn * (1.0 - self.curl)
        self.omega += (w_tgt - self.omega) * min(1.0, dt / 20.0)
        # 运动学积分
        self.theta += self.omega * dt * 1e-3
        self.x += self.v * math.cos(self.theta) * dt * 1e-3
        self.y += self.v * math.sin(self.theta) * dt * 1e-3
        # 蜷缩 → 位移抑制（防御态位移≈0；M8 curl_t 语义）
        if self.curl > p.curl_thr:
            self.v *= (1.0 - self.curl)
        # 有界（arena 反射）
        for attr in ("x", "y"):
            val = getattr(self, attr)
            if abs(val) > self.arena_mm:
                setattr(self, attr, math.copysign(self.arena_mm, val))
        # 翅
        wing_ang, wing_spread = self.wing_signal(self.t_ms, d)
        self.wing_spread = wing_spread
        st = self.classify(d)
        self.last_state = st
        return {"t_ms": self.t_ms, "x": self.x, "y": self.y, "theta": self.theta,
                "v": self.v, "omega": self.omega, "curl": self.curl,
                "wing_ang": wing_ang, "wing_spread": wing_spread, "state": st,
                "n_escape": self.n_escape}

    def run(self, drives: Optional[Dict[str, float]] = None, t_ms: float = 1000.0,
            stimuli: Optional[Dict[str, Any]] = None, dt_ms: Optional[float] = None
            ) -> Dict[str, Any]:
        """按固定通道驱动跑 t_ms，返回时序（关节角/足尖/位置/翅/状态）。

        stimuli（可选，§3.2 VR + 逃跑）：{"light": 时序或常数, "odor": ..., "mech": ...,
        "ball_omega": ..., "ball_v": ...}——VR 闭环：行为腿驱动 → 球转速度（可外部给定）。
        """
        p = self.p
        dt = float(dt_ms if dt_ms is not None else p.dt_ms)
        n = int(round(t_ms / dt))
        drives = dict(drives or {})
        stimuli = dict(stimuli or {})

        def _at(key, i, default=0.0):
            v = stimuli.get(key, default)
            if isinstance(v, (list, tuple, np.ndarray)):
                return float(v[min(i, len(v) - 1)])
            return float(v)

        joints = np.empty((n, 6, 3), dtype=np.float64)
        t = np.empty(n, dtype=np.float64)
        pos = np.empty((n, 2), dtype=np.float64)
        vel = np.empty(n, dtype=np.float64)
        omg = np.empty(n, dtype=np.float64)
        wang = np.empty(n, dtype=np.float64)
        wsp = np.empty(n, dtype=np.float64)
        curl = np.empty(n, dtype=np.float64)
        states: List[str] = []
        escapes: List[int] = []
        orn = np.empty(n, dtype=np.float64)
        flow = np.empty((n, 2), dtype=np.float64)
        for i in range(n):
            d = dict(drives)
            d["C_escape"] = max(d.get("C_escape", 0.0),
                                1.0 if self.giant_fiber_stimulus(_at("light", i),
                                                                 _at("mech", i)) else 0.0)
            rec = self.step(d, dt)
            joints[i] = self.joint_angles(rec["t_ms"], d)
            t[i] = rec["t_ms"]; pos[i] = (rec["x"], rec["y"]); vel[i] = rec["v"]
            omg[i] = rec["omega"]; wang[i] = rec["wing_ang"]; wsp[i] = rec["wing_spread"]
            curl[i] = rec["curl"]; states.append(rec["state"])
            orn[i] = self.antenna_orn(_at("odor", i), _at("mech", i), dt)
            flow[i] = self.eye_flow(_at("ball_omega", i), _at("ball_v", i))
            if rec["state"] == "escape":
                escapes.append(i)
        return {"t_ms": t, "joints_deg": joints, "x": pos[:, 0], "y": pos[:, 1],
                "v": vel, "omega": omg, "wing_ang": wang, "wing_spread": wsp,
                "curl": curl, "states": states, "escape_idx": escapes,
                "orn": orn, "eye_flow": flow, "theta": np.full(n, self.theta)}


def state_fractions(states: Sequence[str]) -> Dict[str, float]:
    """状态序列 → 时间比例（{run,turn,pause,curl,escape}，和为 1）。"""
    n = len(states)
    if n == 0:
        return {s: float("nan") for s in STATES}
    return {s: float(states.count(s)) / n for s in STATES}
