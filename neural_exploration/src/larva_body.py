"""M8 虚拟幼虫身体：分段体 + 蠕动行波 + 五运动模式（P3 验证对象）。

对应《生物仿真M8实施清单》§5（步骤 3：虚拟幼虫身体）与 §0 P3 判据：
  - **分段体**（N_seg 段，预注册 8–12），蠕动行波
    ``body_y(s, t) = A·sin(2π(s/λ − t/T_gait))``（段间相位差推进）；
  - **驱动通道**（肌肉耦合沿用 L5 验证件 Muscle3 通道语义）：
      C_fwd     → 前进行波（前→后相位推进，波速沿体轴正向）；
      C_back    → 后退行波（后→前相位推进，v_rev0·C_back 负向位移）；
      C_left/C_right → 侧转（头摆/侧弯，ω_max·(C_left−C_right) + 转向事件，
                       VirtualBody 机制 A 沿用）；
      C_curl    → 蜷缩（伤害性防御：段间曲率饱和、位移≈0、时程 T_curl 预注册）；
      （掘进 dig = informational：垂直位移/基质穿透语义，非验证级——占位接口）；
  - **状态分类**：classify_larva_state(run/turn/pause/curl)——阈值定稿于
    ``data/m8_larva_body_params.csv``（不做事后调，M3 P5 ×1.2 教训）；
  - **确定性**：p=1/n=1、重跑逐位一致；轨迹有界（arena 边界反射，M4 语义）、无 NaN。

实现简化（M8-B1c2 节点，清单 §5.1 允许）：前进/后退/侧转三模式为验证级
（单通道驱动 → 对应运动学响应正确）；蜷缩实现为防御语义（位移≈0 + 段间
曲率饱和 + T_curl 时程）；掘进为 informational 占位（dig_depth 语义），
接口可扩展（通道字典 + 可扩展 state 集合）。

复用（冻结文件零修改）：VirtualBody（M5 身体积分器/边界/机制 A 转向）、
classify_state/state_fractions（virtual_body.py 冻结语义，幼虫版封装）。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from neural_exploration.src.virtual_body import (
    StateThresholds,
    VirtualBody,
    classify_state,
    state_fractions,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: 幼虫状态集合（P3：run/turn/pause/curl；P4 带 run/turn/pause）
LARVA_STATES = ("run", "turn", "pause", "curl")

#: 身体参数/状态阈值唯一定稿源（本节点新建；value 在 fields[9]，M5-B1d L23 语义）
DEFAULT_BODY_PARAMS_CSV = os.path.join(ROOT, "neural_exploration", "data",
                                       "m8_larva_body_params.csv")


@dataclass
class LarvaStateThresholds:
    """幼虫状态分类阈值（唯一定稿源 = data/m8_larva_body_params.csv state 行）。

    - v_run_frac：run 速度阈值分数（v_thr = v_run_frac·v_fwd0，默认 0.05）；
    - omega_turn_frac：turn 转向阈值分数（ω_thr = omega_turn_frac·omega_max，
      默认 0.2）；
    - curl_thr：蜷缩通道阈值（C_curl ≥ curl_thr → curl，默认 0.3）。
    定稿后不做事后调阈值（M3 P5 教训）。
    """

    v_run_frac: float = 0.05
    omega_turn_frac: float = 0.2
    curl_thr: float = 0.3


def classify_larva_state(
    v: float,
    omega: float,
    c_curl: Optional[float] = None,
    v_thr_frac: float = 0.05,
    omega_thr_frac: float = 0.2,
    curl_thr: float = 0.3,
    v_fwd0: float = 1.0,
    omega_max: float = 1.0,
) -> str:
    """幼虫状态分类：curl/turn/run/pause（阈值 CSV 定稿，不做事后调）。

    判据（清单 §5.1，操作化；curl 防御态优先——位移≈0 覆盖运动学）：
      - curl：C_curl ≥ curl_thr（蜷缩防御态优先）；
      - turn：|ω| > ω_thr（含净后退 v < −v_thr——幼虫反转伴随转向，行为学
        带将反转并入 turn，P4 语义，见 m8_larva_body_params.csv note）；
      - run：v > v_thr（净前进）；
      - pause：其余（|v| ≤ v_thr 且 |ω| ≤ ω_thr）。
    """
    if c_curl is not None and float(c_curl) >= curl_thr:
        return "curl"
    v_thr = v_thr_frac * v_fwd0
    w_thr = omega_thr_frac * omega_max
    if abs(omega) > w_thr:
        return "turn"
    if v > v_thr:
        return "run"
    if v < -v_thr:
        return "turn"   # 净后退并入 turn（幼虫反转 = 转向行为，P4 语义）
    return "pause"


def larva_state_fractions(states: Sequence[str]) -> Dict[str, float]:
    """状态序列 → 时间比例（{run, turn, pause, curl}，和为 1）。"""
    n = len(states)
    if n == 0:
        return {s: float("nan") for s in LARVA_STATES}
    return {s: float(states.count(s)) / n for s in LARVA_STATES}


def load_larva_body_params(csv_path: Optional[str] = None) -> dict:
    """读 data/m8_larva_body_params.csv（唯一定稿源；value 在 fields[9]）。

    返回 {"body": {n_seg, gait_period_ms, wave_amp, wave_lambda, body_len,
    curl_t_ms, dig_depth}, "state": {v_run_frac, omega_turn_frac, curl_thr}}。
    文件缺失 → 返回预注册默认（CSV 为定稿源，代码默认值 = 定稿草案）。
    """
    path = csv_path or DEFAULT_BODY_PARAMS_CSV
    body = dict(n_seg=10, gait_period_ms=500.0, wave_amp=0.3, wave_lambda=1.0,
                body_len=1.0, curl_t_ms=500.0, dig_depth=0.1)
    state = dict(v_run_frac=0.05, omega_turn_frac=0.2, curl_thr=0.3)
    if not os.path.exists(path):
        return {"body": body, "state": state, "source": "default"}
    import csv as _csv
    with open(path, newline="", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            parts = [p.strip() for p in s.split(",")]
            if len(parts) < 10:
                continue
            section, key = parts[0], parts[1]
            try:
                val = float(parts[9])
            except ValueError:
                continue
            if section == "body" and key in body:
                body[key] = int(val) if key == "n_seg" else val
            elif section == "state" and key in state:
                state[key] = val
    return {"body": body, "state": state, "source": path}


class VirtualLarvaBody(VirtualBody):
    """分段幼虫身体（M5 VirtualBody 的幼虫扩展：分段行波 + curl/dig 模式）。

    引擎无关 numpy（与 VirtualBody 同哲学）；确定性 p=1/n=1。
    """

    def __init__(
        self,
        n_seg: int = 10,
        v_fwd0: float = 1.0,
        v_rev0: float = 1.0,
        omega_max: float = 1.0,
        dt_b: float = 25.0,
        arena_L: float = 10.0,
        boundary: str = "reflect",
        gait_period_ms: float = 500.0,
        wave_amp: float = 0.3,
        wave_lambda: float = 1.0,
        body_len: float = 1.0,
        curl_t_ms: float = 500.0,
        curl_thr: float = 0.3,
        dig_depth: float = 0.1,
        head_turn_gain: float = 0.0,
        turn_omega_pir: float = 1.0,
        turn_duration_ms: float = 1571.0,
        thresholds: Optional[LarvaStateThresholds] = None,
    ):
        super().__init__(
            v_fwd0=v_fwd0, v_rev0=v_rev0, omega_max=omega_max, dt_b=dt_b,
            arena_L=arena_L, boundary=boundary,
            gait_period_ms=gait_period_ms, wave_amp=wave_amp,
            wave_lambda=wave_lambda, body_len=body_len,
            head_turn_gain=head_turn_gain, turn_omega_pir=turn_omega_pir,
            turn_duration_ms=turn_duration_ms)
        self.n_seg = int(n_seg)
        if not (8 <= self.n_seg <= 12):
            raise ValueError(f"N_seg 预注册 8–12 段：{self.n_seg}")
        self.curl_t_ms = float(curl_t_ms)
        self.curl_thr = float(curl_thr)
        self.dig_depth = float(dig_depth)
        self.thresholds = thresholds or LarvaStateThresholds(
            v_run_frac=0.05, omega_turn_frac=0.2, curl_thr=curl_thr)
        self.curl_remaining_ms = 0.0
        #: 段间曲率（当前蜷缩强度，informational→防御语义：饱和到 wave_amp）
        self.segment_curvature = 0.0

    # ------------------------------------------------------------------ #
    # 分段体
    # ------------------------------------------------------------------ #
    def segment_positions(self) -> np.ndarray:
        """段位置 s_k（体轴坐标 0..body_len，k=0..n_seg−1）。"""
        return np.linspace(0.0, self.body_len, self.n_seg)

    def body_y(self, x: float, t_ms: float,
               c_fwd: float = 1.0, c_back: float = 0.0) -> float:
        """蠕动行波 body_y(s,t) = A·sin(2π(s/λ − t/T_gait))（P3 规格）。

        方向语义：C_fwd 主导 → 相位沿 +s 推进（前向后行波）；C_back 主导 →
        相位沿 −s 推进（后向前行波）。波幅按驱动强度 clip 缩放（蜷缩时
        段间曲率饱和——见 step）。
        """
        if self.wave_amp <= 0.0 or self.gait_period_ms <= 0.0:
            return 0.0
        fwd = float(np.clip(c_fwd, 0.0, 1.0))
        back = float(np.clip(c_back, 0.0, 1.0))
        amp = self.wave_amp * max(fwd, back)
        if self.curl_remaining_ms > 0.0:
            # 蜷缩：段间曲率饱和（所有段同相收缩，位移≈0 由 step 保证）
            return amp * math.sin(2.0 * math.pi * (x / self.wave_lambda))
        if fwd >= back:
            phase = x / self.wave_lambda - t_ms / self.gait_period_ms
        else:
            phase = x / self.wave_lambda + t_ms / self.gait_period_ms
        return amp * math.sin(2.0 * math.pi * phase)

    def segment_wave(self, t_ms: float,
                     c_fwd: float = 1.0, c_back: float = 0.0) -> np.ndarray:
        """全段波形（逐段 body_y，供冒烟/可视化）。"""
        return np.array([self.body_y(s, t_ms, c_fwd, c_back)
                         for s in self.segment_positions()])

    # ------------------------------------------------------------------ #
    # 蜷缩（P3：伤害性防御：位移≈0 + 段间曲率饱和 + T_curl 时程）
    # ------------------------------------------------------------------ #
    def is_curled(self) -> bool:
        return self.curl_remaining_ms > 0.0

    def trigger_curl(self, c_curl: float):
        """C_curl ≥ curl_thr → 启动蜷缩（T_curl 预注册时程）。"""
        if self.curl_remaining_ms > 0.0:
            return
        if float(c_curl) >= self.curl_thr:
            self.curl_remaining_ms = self.curl_t_ms
            self.segment_curvature = float(np.clip(c_curl, 0.0, 1.0))

    def _tick_curl(self, dt_s: float):
        if self.curl_remaining_ms > 0.0:
            self.curl_remaining_ms = max(
                0.0, self.curl_remaining_ms - dt_s * 1000.0)
            if self.curl_remaining_ms <= 0.0:
                self.segment_curvature = 0.0

    # ------------------------------------------------------------------ #
    # 运动学（幼虫版 step：curl 覆盖 + 行波驱动，接口与 VirtualBody 兼容）
    # ------------------------------------------------------------------ #
    def speed(self, c_fwd: float, c_back: float) -> float:
        """v = v_fwd0·clip(C_fwd,0,1) − v_rev0·clip(C_back,0,1)；蜷缩中恒 0。"""
        if self.curl_remaining_ms > 0.0:
            return 0.0
        return super().speed(c_fwd, c_back)

    def turn_rate(self, c_left: float, c_right: float, t_ms: float = 0.0) -> float:
        """ω：蜷缩中恒 0（防御态不转向）；否则 VirtualBody 语义（机制 A + 侧转）。"""
        if self.curl_remaining_ms > 0.0:
            return 0.0
        return super().turn_rate(c_left, c_right, t_ms)

    def step(self, c_fwd: float, c_back: float, c_left: float, c_right: float,
             c_curl: float = 0.0, dt_ms: Optional[float] = None,
             t_ms: float = 0.0) -> Tuple[float, float, float]:
        """一个行为 tick 的运动学积分（幼虫版：curl 优先，位移≈0）。

        Returns (x, y, theta)（self 同步更新）。curl 期间位置冻结（位移≈0），
        曲率饱和；curl 结束后恢复正常运动学。
        """
        dt = (self.dt_b if dt_ms is None else dt_ms) / 1000.0  # ms → s
        self.trigger_curl(float(c_curl))
        if self.curl_remaining_ms > 0.0:
            self._tick_curl(dt)
            return self.x, self.y, self.theta
        return super().step(c_fwd, c_back, c_left, c_right, dt_ms, t_ms)

    def classify(self, c_fwd: float, c_back: float, c_left: float,
                 c_right: float, c_curl: float = 0.0,
                 thresholds: Optional[LarvaStateThresholds] = None) -> str:
        """由当前肌肉命令即时分类（阈值 CSV 定稿语义）。"""
        v = self.speed(c_fwd, c_back)
        omega = self.turn_rate(c_left, c_right, 0.0)
        th = thresholds or self.thresholds
        return classify_larva_state(v, omega, c_curl,
                                    v_thr_frac=th.v_run_frac,
                                    omega_thr_frac=th.omega_turn_frac,
                                    curl_thr=th.curl_thr,
                                    v_fwd0=self.v_fwd0,
                                    omega_max=self.omega_max)

    def reset(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0):
        super().reset(x, y, theta)
        self.curl_remaining_ms = 0.0
        self.segment_curvature = 0.0

    # ------------------------------------------------------------------ #
    # 掘进（informational：垂直位移/基质穿透语义，非验证级——占位接口）
    # ------------------------------------------------------------------ #
    def dig_depth_at(self, t_ms: float, c_dig: float = 0.0) -> float:
        """掘进深度（informational；基质穿透语义占位——不进入运动学积分）。"""
        if c_dig <= 0.0:
            return 0.0
        return self.dig_depth * float(np.clip(c_dig, 0.0, 1.0))
