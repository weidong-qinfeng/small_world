"""先天机制模块——行为级可移植封装（P-A1，机制回迁 M7）

从 M3–M6 冻结神经仿真基线提取的 **行为级等价抽象**（纯 python / stdlib only，
无 brian2、无 numpy 依赖）：输入环境刺激 → 输出行为反应/增益/权重变化。
机制回迁 = 行为级等价（方向/衰减/增益/Δw 符号），**不是**把 Brian2 网络搬进
数字大脑（M7 清单 §0 #2 任务关键设计决策；神经实时动力学与符号推理语义不同）。

等价性锚（冻结报告数值，清单 §2.1；模块只读 `data/m7_innate_params.csv`，
不重新训练/校准——冻结基线纪律）：
  - M-1 反射：D_peak=0.352–0.369 > 0.3 → back（M3 定稿带；M5 302 0.610 饱和）
  - M-2 趋化：CI 符号正（M4 参考 CI(25s)=0.494 / CI(15s)=0.417；生物带 [0.3,0.7]）
  - M-3 CPG：无食物 0.400Hz∈[0.1,2] / 有食物 2.167Hz∈[2,5]（M5 P3 冻结）
  - M-4 习惯化：R(n) 指数衰减 R²≥0.5、τ_hab≈2（M6 短 ISI 冻结；预注册带 [3,15]
    出带为测量限制如实记录）、消融 STP 关 → 无衰减、恢复 R_rest≥0.3×R(1)
  - M-5 联想：Δw_train>0.1（冻结 +0.4325）、η=0 → Δw=0、消退 Δw_ext<0
    （冻结 −0.108；绝对值不作硬判据——清单 §2.1）
  - M-6 调质：fwd_gate ∈ [tyr_floor, 1.2]、酪胺关 → gate≡1（冻结消融 sanity）

验证边界（不伪造超出验证范围的语义——清单 §2.2）：
  - M-7（夹带双稳态等反证项）**不封装**（设计依据交接阶段二）；
  - M-4 网络级 10s-ISI 主协议判据不可达、M-5 网络级 CI 读出不可见等测量限制
    仅作模块 docstring/行为边界记录，不进入回迁判据。

确定性：无随机（p=1/n=1）；同参数重跑**逐位一致**。
"""

from __future__ import annotations

import csv
import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------- #
# 数据容器（机制输入/输出）
# --------------------------------------------------------------------- #


@dataclass
class Stimulus:
    """环境刺激（机制输入）。

    kind：touch（触刺激）/ odor（位置采样）/ time（CPG 时间）/
          pairing（联想配对）/ motivation（调质动机）
    """
    kind: str = "touch"
    intensity: float = 1.0               # 触刺激相对强度（1.0 = 基准 60µA/cm²）
    x: float = 5.0                       # 位置采样 x（趋化）
    y: float = 5.0                       # 位置采样 y（趋化）
    t_ms: float = 0.0                    # 时间（CPG 相位）
    food_present: bool = False           # 食物在场（CPG 频率切换）
    cs: float = 0.0                      # 条件刺激强度（联想）
    us: float = 0.0                      # 非条件刺激/调质信号（联想/调质）
    # 联想学习附加协议参数（None → 用 CSV 定稿值）
    n_pairings: Optional[int] = None     # 配对步数
    phase: str = "train"                 # train | ext | eta0


@dataclass
class Response:
    """行为反应（机制输出）。"""
    kind: str
    value: float                          # 主行为量（D_peak / CI / 频率 / R(n) / gain / phase）
    direction: str = ""                   # back | none | toward_gradient | ...
    extra: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {"kind": self.kind, "value": self.value,
                "direction": self.direction, "extra": dict(self.extra)}


@dataclass
class DeltaW:
    """联想学习权重变化。"""
    dw: float
    w: float
    phase: str = "train"


# --------------------------------------------------------------------- #
# CSV 参数读取（stdlib only；位置解析沿用 M5 L23 惯例：value 在 fields[9]）
# --------------------------------------------------------------------- #

_DEFAULT_PARAMS_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))),
    "neural_exploration", "data", "m7_innate_params.csv")


def _parse_value(raw: str):
    s = str(raw).strip()
    try:
        return float(s)
    except ValueError:
        return s


def _parse_band(raw: str) -> Tuple[float, float]:
    parts = [float(x) for x in str(raw).split("..") if x.strip()]
    return (parts[0], parts[1]) if len(parts) == 2 else (parts[0], parts[0])


def load_innate_params(csv_path: Optional[str] = None) -> Dict[str, Dict[str, object]]:
    """读 m7_innate_params.csv → {mechanism: {param: value}}。

    value 在 fields[9]（M5 L23 位置解析惯例）；'..' 带 → float 二元组。
    """
    path = csv_path or os.environ.get("M7_INNATE_PARAMS_CSV") or _DEFAULT_PARAMS_CSV
    out: Dict[str, Dict[str, object]] = {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"m7_innate_params.csv 不存在：{path}")
    with open(path, newline="", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            fields = next(csv.reader([s]))
            role = (fields[0] if fields else "").strip().lower()
            key = (fields[1] if len(fields) > 1 else "").strip().lower()
            if role not in ("reflex", "chemotaxis", "cpg", "habituation",
                            "associative", "modulation") or not key:
                continue
            value = (fields[9] if len(fields) >= 11 else "")
            parsed = _parse_value(value)
            if isinstance(parsed, str) and ".." in parsed:
                parsed = _parse_band(parsed)
            out.setdefault(role, {})[key] = parsed
    return out


# --------------------------------------------------------------------- #
# InnateMechanism 基类
# --------------------------------------------------------------------- #


class InnateMechanism(ABC):
    """机制接口规格（M7 清单 §2.1）。

    respond(stimulus) -> Response      # 环境刺激 → 行为反应
    associate(cs, us) -> DeltaW        # 联想学习（仅 M-5 实现）
    gate(motivation) -> float          # 调质门控（仅 M-6 实现）
    reset()                            # 试次开始：状态清零（确定性）
    """

    name: str = "base"

    def __init__(self, params: Optional[Dict[str, object]] = None,
                 params_csv: Optional[str] = None):
        all_p = load_innate_params(params_csv)
        self.params: Dict[str, object] = dict(all_p.get(self.name, {}))
        if params is not None:
            self.params.update(params)
        self.reset()

    @abstractmethod
    def respond(self, stimulus: Stimulus) -> Response:
        raise NotImplementedError

    def associate(self, cs: float, us: float) -> DeltaW:  # noqa: D401
        raise NotImplementedError(f"{self.name} 无 associate（联想学习专属）")

    def gate(self, motivation: float) -> float:  # noqa: D401
        raise NotImplementedError(f"{self.name} 无 gate（调质门控专属）")

    def reset(self) -> None:
        """试次开始：状态清零（确定性）。子类覆盖。"""

    # -- 参数访问便捷方法 -------------------------------------------- #
    def p(self, key: str, default: float = 0.0) -> float:
        v = self.params.get(key, default)
        return float(v) if isinstance(v, (int, float)) else default

    def p_band(self, key: str) -> Tuple[float, float]:
        v = self.params.get(key, (0.0, 0.0))
        if isinstance(v, (tuple, list)) and len(v) == 2:
            return float(v[0]), float(v[1])
        f = float(v)
        return f, f


# --------------------------------------------------------------------- #
# M-1 触觉反射弧（M3 冻结：D_peak 带 [0.352,0.369]，>0.3 → back）
# --------------------------------------------------------------------- #


class ReflexArcMechanism(InnateMechanism):
    """触觉反射弧——刺激强度 → 定向反应（D_peak 语义）。

    行为级模型（确定性）：d_peak(I) = d_max·I/(I+i_half)（饱和曲线；
    d_peak(1.0)=d_peak_base ∈ M3 定稿带；i_half=d_max/d_base−1 由锚反解）；
    方向 back ⟺ d_peak>0.3（behavior_reference escape.direction_peak）。
    """

    name = "reflex"

    def respond(self, stimulus: Stimulus) -> Response:
        i = float(stimulus.intensity)
        d_max = self.p("d_peak_max", 0.610)
        i_half = self.p("i_half", 0.694)
        if i <= 0.0 or not math.isfinite(i):
            return Response("reflex", 0.0, "none",
                            {"nerve_latency_ms": self.p("nerve_latency_ms", 10.0),
                             "behavior_latency_ms": self.p("behavior_latency_ms", 32.6)})
        d_peak = d_max * i / (i + i_half)
        direction = "back" if d_peak > self.p("d_peak_thr", 0.3) else "none"
        return Response(
            "reflex", d_peak, direction,
            {"c_back_peak": d_peak + self.p("muscle_w_fwd", 0.18),
             "c_fwd_peak": self.p("muscle_w_fwd", 0.18),
             "nerve_latency_ms": self.p("nerve_latency_ms", 10.0),
             "behavior_latency_ms": self.p("behavior_latency_ms", 32.6),
             "intensity": i})


# --------------------------------------------------------------------- #
# M-2 嗅觉趋化（M4 冻结：正向梯度趋利，CI 符号正；参考 CI(25s)=0.494）
# --------------------------------------------------------------------- #


class ChemotaxisMechanism(InnateMechanism):
    """嗅觉趋化——环境梯度 → 趋利决策（CI 语义）。

    环境 = M4 冻结梯度场（arena_L/sigma/c_max/c_bg/food_x/food_y/boundary）；
    运动策略 = Braitenberg 限速转向（ω≤ω_max，转向率来自 m4 body.omega_max；
    确定性，无随机）：heading 朝梯度方向以有限转向率收敛 → 趋利。
    CI 语义 = Pierce-Shimomura 象限式（食物象限 +1 / 对侧 −1，与冻结
    chemotaxis_env.ci_per_trial 相同）。
    """

    name = "chemotaxis"

    # -- 环境（M4 冻结梯度场） --------------------------------------- #
    def concentration(self, x: float, y: float) -> float:
        sigma = self.p("sigma", 1.25)
        c_max = self.p("c_max", 1.0)
        c_bg = self.p("c_bg", 0.0)
        r2 = (x - self.p("food_x", 7.5)) ** 2 + (y - self.p("food_y", 7.5)) ** 2
        return c_max * math.exp(-r2 / (2.0 * sigma * sigma)) + c_bg

    def gradient(self, x: float, y: float) -> Tuple[float, float]:
        """∇C（指向食物滴；化学梯度方向）。"""
        sigma = self.p("sigma", 1.25)
        c_max = self.p("c_max", 1.0)
        fx, fy = self.p("food_x", 7.5), self.p("food_y", 7.5)
        r2 = (x - fx) ** 2 + (y - fy) ** 2
        e = c_max * math.exp(-r2 / (2.0 * sigma * sigma))
        gx = -e * (x - fx) / (sigma * sigma)
        gy = -e * (y - fy) / (sigma * sigma)
        return gx, gy

    def steering(self, x: float, y: float, heading: float) -> Tuple[float, float]:
        """当前 heading 下，指向梯度的目标转向（限速 ω_max）。"""
        gx, gy = self.gradient(x, y)
        g_ang = math.atan2(gy, gx)
        d = g_ang - heading
        while d > math.pi:
            d -= 2.0 * math.pi
        while d < -math.pi:
            d += 2.0 * math.pi
        omega = self.p("omega_max", 1.0)
        turn = max(-omega, min(omega, d / (self.p("dt_b_ms", 25.0) / 1000.0)))
        return math.cos(heading), math.sin(heading), turn

    # -- 单试次确定性轨迹 → CI --------------------------------------- #
    def run_trial(self, start: Tuple[float, float] = (5.0, 5.0),
                  theta0: float = math.pi, t_total_ms: Optional[float] = None
                  ) -> Tuple[List[float], List[float]]:
        """确定性梯度趋利试次（Braitenberg 限速转向；边界反射）。

        theta0 默认 π（朝西——远离食物的不利朝向探针：机制仍转向趋利）。
        """
        arena_L = self.p("arena_L", 10.0)
        dt_b = self.p("dt_b_ms", 25.0)
        v = self.p("v_fwd0", 1.0)
        t_total = float(self.p("t_total_ms", 10000.0) if t_total_ms is None
                        else t_total_ms)
        x, y = float(start[0]), float(start[1])
        th = float(theta0)
        xs, ys = [x], [y]
        n = int(round(t_total / dt_b))
        for _ in range(n):
            _, _, turn = self.steering(x, y, th)
            th += turn * dt_b / 1000.0
            x += v * math.cos(th) * dt_b / 1000.0
            y += v * math.sin(th) * dt_b / 1000.0
            if x < 0.0:
                x = -x
            if x > arena_L:
                x = 2.0 * arena_L - x
            if y < 0.0:
                y = -y
            if y > arena_L:
                y = 2.0 * arena_L - y
            xs.append(x)
            ys.append(y)
        return xs, ys

    def ci(self, xs: Sequence[float], ys: Sequence[float]) -> float:
        """象限式 CI（与冻结 chemotaxis_env.ci_per_trial 同语义）。"""
        h = self.p("arena_L", 10.0) / 2.0
        fq = (1 if self.p("food_x", 7.5) > h and self.p("food_y", 7.5) > h else 3)
        oq = {1: 3, 2: 4, 3: 1, 4: 2}[fq]
        n, tot = 0.0, 0.0
        for x, y in zip(xs, ys):
            if x == h or y == h:
                continue
            q = 2 if (x < h and y > h) else (1 if x > h and y > h else
                 3 if x < h and y < h else 4)
            n += 1.0 if q == fq else (-1.0 if q == oq else 0.0)
            tot += 1.0
        return n / tot if tot else 0.0

    def respond(self, stimulus: Stimulus) -> Response:
        x, y = float(stimulus.x), float(stimulus.y)
        c = self.concentration(x, y)
        gx, gy = self.gradient(x, y)
        g = math.hypot(gx, gy)
        # 采样位置的趋利方向（指向食物象限的朝向）
        toward = math.atan2(self.p("food_y", 7.5) - y, self.p("food_x", 7.5) - x)
        return Response("chemotaxis", c, "toward_gradient",
                        {"gradient_x": gx, "gradient_y": gy,
                         "gradient_norm": g,
                         "steering_heading": toward,
                         "ci_band_lo": self.p("ci_band_lo", 0.3),
                         "ci_band_hi": self.p("ci_band_hi", 0.7)})


# --------------------------------------------------------------------- #
# M-3 咽部 CPG（M5 P3 冻结：0.400 / 2.167Hz 双带）
# --------------------------------------------------------------------- #


class CpgMechanism(InnateMechanism):
    """咽部 CPG——时间 → 节律相位/频率（带宽判据）。

    行为级模型：频率 = f_no_food / f_with_food（冻结 M5 P3 主频，落
    behavior_reference 预注册带）；phase(t) = (t·f) mod 1；脉冲 = 相位窗口。
    """

    name = "cpg"

    def frequency(self, food_present: bool = False) -> float:
        return (self.p("f_with_food_hz", 2.167) if food_present
                else self.p("f_no_food_hz", 0.400))

    def respond(self, stimulus: Stimulus) -> Response:
        f = self.frequency(stimulus.food_present)
        t_s = float(stimulus.t_ms) / 1000.0
        phase = (t_s * f) % 1.0
        lo, hi = self.p_band("band_with_food_hz" if stimulus.food_present
                             else "band_no_food_hz")
        return Response("cpg", f, "rhythm",
                        {"phase": phase,
                         "in_band": bool(lo <= f <= hi),
                         "food_present": stimulus.food_present,
                         "band_lo": lo, "band_hi": hi})


# --------------------------------------------------------------------- #
# M-4 习惯化（M6 P3 冻结：R(n) 指数衰减 R²≥0.5；τ_hab≈2；消融/恢复）
# --------------------------------------------------------------------- #


class HabituationMechanism(InnateMechanism):
    """习惯化——重复刺激 → 反应衰减（R(n) 序列语义）。

    行为级模型：R(n) = (r0 − B)·exp(−(n−1)/τ_hab) + B（指数衰减；
    r0/τ_hab/B 取 M6 短 ISI 冻结拟合值）；stp_enabled=False 消融 →
    R(n) ≡ r0（无衰减，H1 机制必需）；rest 恢复 → R_rest = r0（相对判据
    R_rest ≥ 0.3×R(1)）。衰减形状判据：指数拟合 R²≥0.5 或后半均值
    < 0.5×前半均值（清单 §0 P-A1）。10s-ISI 主协议不可达为测量限制
    （τ_rec=1000ms ≪ ISI=10s → R(n) 常数），模块默认短 ISI 语义。
    """

    name = "habituation"

    def __init__(self, params=None, params_csv=None):
        super().__init__(params, params_csv)
        self._n = 0
        self._stp_enabled = bool(self.p("stp_enabled", 1.0))

    def reset(self) -> None:
        self._n = 0

    def set_stp(self, enabled: bool) -> None:
        self._stp_enabled = bool(enabled)

    # -- 指数拟合（stdlib 确定性网格 lstsq；与冻结 fit_exponential 语义一致）--
    def fit_exponential(self, r_seq: Sequence[float]) -> Dict[str, float]:
        y = [float(v) for v in r_seq]
        n_idx = [float(k + 1) for k in range(len(y))]
        best = None
        tau0 = 1.0
        while tau0 <= 30.0 + 1e-9:
            ex = [math.exp(-ni / tau0) for ni in n_idx]
            # 闭式最小二乘 [exp, 1] 两列（正常方程）
            s11 = sum(e * e for e in ex)
            s12 = sum(ex)
            s22 = float(len(y))
            s1y = sum(e * v for e, v in zip(ex, y))
            s2y = sum(y)
            den = s11 * s22 - s12 * s12
            if abs(den) > 1e-12:
                A = (s1y * s22 - s12 * s2y) / den
                B = (s11 * s2y - s12 * s1y) / den
                yhat = [A * e + B for e in ex]
                ss_res = sum((v - h) ** 2 for v, h in zip(y, yhat))
                mean = sum(y) / len(y)
                ss_tot = sum((v - mean) ** 2 for v in y)
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
                if best is None or (r2 == r2 and r2 > best[2]):
                    best = (A, tau0, r2, B)
            tau0 += 1.0
        if best is None:
            return dict(A=float("nan"), tau_hab=float("nan"), B=float("nan"),
                        r2=float("nan"), r2_ok=False, fit_ok=False)
        A, tau, r2, B = best
        lo, hi = self.p_band("fit_tau_band")
        return dict(A=float(A), tau_hab=float(tau), B=float(B), r2=float(r2),
                    r2_ok=bool(r2 >= self.p("fit_r2_min", 0.5)),
                    fit_ok=bool(A > 0 and r2 == r2))

    # -- 响应序列 ----------------------------------------------------- #
    def r_sequence(self, n_stim: int, stp_enabled: Optional[bool] = None,
                   ) -> List[float]:
        stp = self._stp_enabled if stp_enabled is None else bool(stp_enabled)
        r0 = self.p("r0", 0.353)
        tau = self.p("tau_hab", 2.0)
        B = self.p("r_plateau", -0.183)
        if not stp:
            return [r0] * max(1, int(n_stim))
        return [(r0 - B) * math.exp(-(k - 1) / tau) + B
                for k in range(1, max(1, int(n_stim)) + 1)]

    def respond(self, stimulus: Stimulus) -> Response:
        """第 n 次刺激 → 当前反应幅度 R(n)（内部计数递增）。"""
        self._n += 1
        r = self.r_sequence(self._n)[-1]
        return Response("habituation", r,
                        "decay" if self._n > 1 and r < self.r_sequence(1)[0]
                        else "initial",
                        {"n": self._n,
                         "r0": self.p("r0", 0.353),
                         "tau_hab": self.p("tau_hab", 2.0),
                         "stp_enabled": self._stp_enabled})

    def run_sequence(self, n_stim: int = 6, stp_enabled: Optional[bool] = None,
                     rest_ms: float = 0.0) -> Dict[str, object]:
        """完整协议：R(n) 序列 + 指数拟合 + 恢复 + 衰减判据（确定性）。"""
        r_seq = self.r_sequence(n_stim, stp_enabled)
        fit = self.fit_exponential(r_seq)
        half = max(1, len(r_seq) // 2)
        first = sum(r_seq[:half]) / half
        last = sum(r_seq[half:]) / max(1, len(r_seq) - half)
        r_rest = self.p("r0", 0.353) if rest_ms > 0 else float("nan")
        return dict(
            r_seq=r_seq,
            fit=fit,
            r_rest=float(r_rest),
            decay=float(r_seq[0] - r_seq[-1]) if len(r_seq) > 1 else 0.0,
            first_half_mean=float(first),
            last_half_mean=float(last),
            decay_ok=bool(len(r_seq) > 1 and r_seq[0] > r_seq[-1]),
            half_criterion_ok=bool(last < 0.5 * first),
            direction_ok=bool(r_seq and r_seq[0] > self.p("d_peak_thr", 0.3)),
            recover_ok=bool(rest_ms > 0 and r_rest >=
                            self.p("recover_frac_min", 0.3) * r_seq[0]),
            stp_enabled=(self._stp_enabled if stp_enabled is None
                         else bool(stp_enabled)),
            n_stim=n_stim, rest_ms=rest_ms,
        )


# --------------------------------------------------------------------- #
# M-5 联想学习（M6 P4 冻结：Δw_train>0.1 / η=0 → 0 / 消退 Δw_ext<0）
# --------------------------------------------------------------------- #


class AssociativeMechanism(InnateMechanism):
    """联想学习——CS-US 配对 → 权重变化（Δw 语义，三因子门控）。

    行为级模型（确定性离散三因子）：资格迹 e(t) = e(t−1)·exp(−dt/τ_e) +
    cs·(1−exp(−dt/τ_e))（归一化稳态 → cs·1.0）；Δw = Σ η·us(t)·e(t)；
    w = clip(w0+Δw, 0, w_max)。US 窗协议（us_period/us_on）取自冻结
    associative 段；η=0 → Δw≡0（三因子门控必需，消融）；US 反号 → 消退。
    等价判据：Δw_train>0（冻结 +0.4325 方向锚）、η=0 → Δw=0、
    Δw_ext<0（冻结 −0.108）；**绝对值不作硬判据**（清单 §2.1）。
    """

    name = "associative"

    def __init__(self, params=None, params_csv=None):
        super().__init__(params, params_csv)
        self._w = self.p("w0", 1.0)
        self._e = 0.0

    def reset(self) -> None:
        self._w = self.p("w0", 1.0)
        self._e = 0.0

    def weights(self) -> float:
        return self._w

    # -- 训练/消退（确定性时间步积分） -------------------------------- #
    def run_epoch(self, t_epoch_ms: Optional[float] = None, cs: float = 1.0,
                  us_sign: Optional[float] = None,
                  phase: str = "train") -> DeltaW:
        dt = self.p("dt_ms", 25.0)
        tau_e = self.p("tau_e_ms", 200.0)
        eta = self.p("eta", 0.01)
        t_epoch = float(self.p("t_train_ms", 8000.0) if t_epoch_ms is None
                        else t_epoch_ms)
        if us_sign is None:
            us_sign = (self.p("us_train_signal", 1.0) if phase == "train"
                       else self.p("us_ext_signal", -1.0))
        period = self.p("us_period_ms", 400.0)
        us_on = self.p("us_on_ms", 200.0)
        w0 = self._w
        decay = math.exp(-dt / tau_e)
        n = int(round(t_epoch / dt))
        dw = 0.0
        for k in range(n):
            us = us_sign if (k * dt) % period < us_on else 0.0
            self._e = self._e * decay + cs * (1.0 - decay)
            dw += eta * us * self._e
        w_new = max(0.0, min(self.p("w_max", 2.0), w0 + dw))
        self._w = w_new
        return DeltaW(dw=float(dw), w=float(w_new), phase=phase)

    def respond(self, stimulus: Stimulus) -> Response:
        """CS-US 配对刺激 → 当前关联强度（w）。"""
        dw = self.associate(stimulus.cs, stimulus.us)
        return Response("associative", self._w, "association",
                        {"dw": dw.dw, "phase": dw.phase,
                         "eta": self.p("eta", 0.01)})

    def associate(self, cs: float, us: float) -> DeltaW:
        """单步配对（供逐对调用；内部资格迹/权重状态）。"""
        dt = self.p("dt_ms", 25.0)
        tau_e = self.p("tau_e_ms", 200.0)
        eta = self.p("eta", 0.01)
        decay = math.exp(-dt / tau_e)
        self._e = self._e * decay + cs * (1.0 - decay)
        dw = eta * us * self._e
        self._w = max(0.0, min(self.p("w_max", 2.0), self._w + dw))
        return DeltaW(dw=float(dw), w=float(self._w), phase="pairing")

    def full_protocol(self) -> Dict[str, object]:
        """完整协议：基线 → 训练 → 消退 + η=0 消融（确定性）。"""
        self.reset()
        w0 = self._w
        tr = self.run_epoch(phase="train")
        w_tr = self._w
        ex = self.run_epoch(phase="ext")
        w_ext = self._w
        # η=0 消融（独立实例，权重不变）
        eta0 = AssociativeMechanism(dict(self.params))
        eta0.params["eta"] = 0.0
        dw0 = eta0.run_epoch(phase="train").dw
        return dict(
            w0=float(w0), dw_train=float(tr.dw), w_tr=float(w_tr),
            dw_ext=float(ex.dw), w_ext=float(w_ext),
            dw_eta0=float(dw0),
            acquisition_ok=bool(tr.dw > 0.1),
            extinction_ok=bool(ex.dw < 0.0),
            eta0_ok=bool(abs(dw0) < 1e-9),
            eta=self.p("eta", 0.01),
            dw_train_ref=self.p("dw_train_ref", 0.4325),
            dw_ext_ref=self.p("dw_ext_ref", -0.108),
        )


# --------------------------------------------------------------------- #
# M-6 神经调质层（M6 P2：fwd_gate ∈ [tyr_floor,1.2]；消融 sanity）
# --------------------------------------------------------------------- #


class ModulationMechanism(InnateMechanism):
    """神经调质层——动机/唤醒 → 运动增益门控（fwd_gate 语义）。

    行为级模型 = 冻结 ModulatorPool 的纯 python 移植（浓度 ODE
    exponential_euler + 门控单调有界）：C_tyr↑ → fwd_gate↓（下限
    tyr_floor）；C_5ht → 前进增益↑（capped 1.2）；C_da → 运动层增益
    1/(1+K·C)（Hill 型）。消融：tyramine_enabled=0 → gate≡1
    （冻结 sanity：酪胺关 → gate≡1，escape 仍 back 0.355）。
    """

    name = "modulation"

    def __init__(self, params=None, params_csv=None):
        super().__init__(params, params_csv)
        self.C_da = 0.0
        self.C_5ht = 0.0
        self.C_tyr = 0.0

    def reset(self) -> None:
        self.C_da = 0.0
        self.C_5ht = 0.0
        self.C_tyr = 0.0

    def update(self, dt_ms: float, rates: Dict[str, float]) -> None:
        """浓度 ODE（与冻结 ModulatorPool 同式：C += (R−C)·dt/τ，clip [0,1]）。

        语义对齐冻结实现（neuromod.py ModulatorPool.update）：
          R_tyr = max(norm(AVA/AVD 后退命令率), tyr_baseline)  —— tyr_baseline
                  是**归一化浓度目标**（O2 定稿 1.0 → C_tyr→1.0 → gate→floor 0.4）；
          R_da  = max(norm(多巴胺源率), da_baseline)；
          R_5ht = max(norm(血清素源率), ht_baseline)。
        rates 键：池级 "da"/"5ht"/"tyr"（Hz）或角色级 AVA/AVD/ADE/CEP/PDE/
        ADF/NSM/RIH（按冻结源池聚合）。
        """
        norm = self.p("rate_norm_hz", 30.0)

        def _pool(vals: List[float]) -> float:
            return sum(vals) / len(vals) if vals else 0.0

        def _norm_rate(rate: float) -> float:
            return max(0.0, min(1.0, rate / norm)) if norm > 0 else 0.0

        r_tyr_src = _pool([rates.get(r, 0.0) for r in ("AVA", "AVD")
                           if r in rates])
        if "tyr" in rates:
            r_tyr_src = float(rates["tyr"])
        r_da_src = _pool([rates.get(r, 0.0) for r in ("ADE", "CEP", "PDE")
                          if r in rates])
        if "da" in rates:
            r_da_src = float(rates["da"])
        r_ht_src = _pool([rates.get(r, 0.0) for r in ("ADF", "NSM", "RIH")
                          if r in rates])
        if "5ht" in rates:
            r_ht_src = float(rates["5ht"])

        r_tyr = max(_norm_rate(r_tyr_src), self.p("tyr_baseline", 1.0)) \
            if bool(self.p("tyramine_enabled", 1.0)) else 0.0
        r_da = max(_norm_rate(r_da_src), self.p("da_baseline", 0.05)) \
            if bool(self.p("da_enabled", 1.0)) else 0.0
        r_ht = max(_norm_rate(r_ht_src), self.p("ht_baseline", 0.05)) \
            if bool(self.p("ht_enabled", 1.0)) else 0.0
        dt = max(float(dt_ms), 0.0)

        def _step(C: float, R: float, tau_ms: float) -> float:
            if tau_ms <= 0:
                return max(0.0, min(1.0, R))
            C = C + (R - C) * (dt / tau_ms)
            return max(0.0, min(1.0, C))

        self.C_da = _step(self.C_da, r_da, self.p("tau_da_ms", 500.0))
        self.C_5ht = _step(self.C_5ht, r_ht, self.p("tau_5ht_ms", 500.0))
        self.C_tyr = _step(self.C_tyr, r_tyr, self.p("tau_tyr_ms", 500.0))

    def fwd_gate(self) -> float:
        """门控增益（单调有界；等价冻结 ModulatorPool.fwd_gate）。"""
        floor = self.p("tyr_floor", 0.30)
        g = 1.0
        if bool(self.p("tyramine_enabled", 1.0)):
            g = 1.0 - self.p("tyr_gain", 0.60) * self.C_tyr
            g = max(floor, min(1.0, g))
        if bool(self.p("ht_enabled", 1.0)):
            g *= max(1.0, min(1.2, 1.0 + self.p("ht_gain", 0.20) * self.C_5ht))
        if bool(self.p("da_enabled", 1.0)):
            g *= 1.0 / (1.0 + self.p("da_gain", 0.30) * self.C_da)
        return max(floor, min(self.p("gate_max", 1.2), g))

    def gate(self, motivation: float) -> float:
        """动机标量（0..1，驱动血清素浓度）→ 运动增益。"""
        self.reset()
        self.update(500.0, {"5ht": float(motivation) * self.p("rate_norm_hz", 30.0)})
        return self.fwd_gate()

    def respond(self, stimulus: Stimulus) -> Response:
        g = self.fwd_gate()
        return Response("modulation", g, "gate",
                        {"C_da": self.C_da, "C_5ht": self.C_5ht,
                         "C_tyr": self.C_tyr,
                         "gate_min": self.p("gate_min", 0.30),
                         "gate_max": self.p("gate_max", 1.20)})


# --------------------------------------------------------------------- #
# 注册表
# --------------------------------------------------------------------- #

MECHANISMS: Dict[str, type] = {
    "reflex": ReflexArcMechanism,
    "chemotaxis": ChemotaxisMechanism,
    "cpg": CpgMechanism,
    "habituation": HabituationMechanism,
    "associative": AssociativeMechanism,
    "modulation": ModulationMechanism,
}


def make_mechanism(name: str, params: Optional[Dict[str, object]] = None,
                   params_csv: Optional[str] = None) -> InnateMechanism:
    cls = MECHANISMS.get(name)
    if cls is None:
        raise KeyError(f"未知机制：{name}（可选 {sorted(MECHANISMS)}）")
    return cls(params=params, params_csv=params_csv)


def make_all(params_csv: Optional[str] = None) -> Dict[str, InnateMechanism]:
    return {n: make_mechanism(n, params_csv=params_csv) for n in MECHANISMS}
