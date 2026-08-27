"""先天机制层接口（M7 P-A2）——InnateMechanism 注入数字大脑的感知/决策流。

层位（M7 清单 D3 接入语义）：认知层（应用题文本 → 推理）在上，机制层
（环境刺激 → 行为）在下——**先天机制 = 数字大脑"感知/运动底座"**。本里程碑只验证
**接口接通 + 行为差异可测**，不做"神经 → 符号"完整桥（预注册 §0 #9：认知层推理链
不被机制层替换，不把符号推理改写成神经计算）。

机制 → 接入映射（M7 清单 D2，行为级抽象——M7 清单 §0 #2）：
  - M-2 趋化   → 环境感知：sense(odor@pos) 读"在哪里/闻到了什么"、
                 actuate(approach) 定"往哪走"（正向梯度趋利）
  - M-1 反射   → 先天运动反应：actuate(escape) 触刺激 → 定向回避硬连线
  - M-3 CPG    → 行为节奏：actuate(rhythm) 时间 → 节律相位/频率（时钟/节奏底座）
  - M-4 习惯化 → 适应：adapt(n) 重复刺激 → 响应衰减（响应调节底座）
  - M-6 调质   → 运动增益门控：gate(motivation) 动机/唤醒 → 增益，注入 actuate 决策
  - M-5 联想   → 关联强度（机制层可观察；认知层场景视能力范围取舍——预注册 §0 #5）

可观察性（P-A2 断言依据）：
  - `calls` 调用日志：每次 sense/actuate/adapt/gate 落一条记录（机制层确实被调用）；
  - `set_enabled(name, False)` 消融：该机制贡献归零（注入前后行为差异可测；
    M6 消融 sanity 惯例保留——如反射关 → 无回避方向、习惯化关 → R(n) 无衰减、
    调质关 → gate≡1）。

确定性：无随机（p=1/n=1）；同输入同输出；同参数重跑逐位一致。
验证边界（不伪造超出验证范围的语义——M7 清单 §2.2）：M-7（夹带双稳态等反证项）
不接入；M-4 10s-ISI 主协议、M-5 网络级 CI 读出等测量限制仅作模块边界记录。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

from digital_brain.src.innate.innate_mechanisms import (
    InnateMechanism,
    Response,
    Stimulus,
)

# 路由到四方法（sense/actuate/adapt/gate）的机制名（M-1..M-6 子集）；
# M-5 联想仅机制层可观察（不经四方法路由，见 __init__ 注释）。
_ROUTED = ("chemotaxis", "reflex", "cpg", "habituation", "modulation")


class InnateInterface:
    """先天机制层接入接口（D3 规格：sense/actuate/adapt/gate 四方法）。

    构造：`InnateInterface(mechanisms)`——接受 `InnateMechanism` 序列或
    `{name: mechanism}` 映射（推荐 `make_all()` 全量注入）。
    """

    def __init__(self, mechanisms: "Sequence[InnateMechanism]"):
        if isinstance(mechanisms, dict):
            self.mechanisms: Dict[str, InnateMechanism] = dict(mechanisms)
        else:
            self.mechanisms = {m.name: m for m in mechanisms}
        # 未路由机制（如 M-5 联想）仍注入并可观察（brain.innate.mechanisms 存在性），
        # 只是不进 sense/actuate/adapt/gate 四方法路由（D3 规格：联想 = 机制层可观察性，
        # 认知层场景视能力范围取舍——预注册 §0 #5）。
        self._enabled: Dict[str, bool] = {n: True for n in self.mechanisms}
        self._calls: List[Dict[str, object]] = []

    # ------------------------------------------------------------------ #
    # 可观察性（P-A2 断言：机制存在性 / 被调用日志 / 消融开关）
    # ------------------------------------------------------------------ #

    @property
    def names(self) -> List[str]:
        return sorted(self.mechanisms)

    def has(self, name: str) -> bool:
        return name in self.mechanisms

    def is_enabled(self, name: str) -> bool:
        return self._enabled.get(name, False)

    def set_enabled(self, name: str, enabled: bool) -> None:
        """消融开关：False → 该机制贡献归零（行为差异消失，M6 惯例）。"""
        if name in self.mechanisms:
            self._enabled[name] = bool(enabled)

    @property
    def calls(self) -> List[Dict[str, object]]:
        """调用日志（机制层确实被调用的断言依据；确定性）。"""
        return [dict(c) for c in self._calls]

    def clear_calls(self) -> None:
        self._calls = []

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #

    def _m(self, name: str) -> InnateMechanism:
        mech = self.mechanisms.get(name)
        if mech is None:
            raise KeyError(
                f"InnateInterface 未注入机制：{name}（可用 {self.names}）")
        return mech

    def _record(self, method: str, mechanism: str, value: object,
                direction: str = "") -> None:
        self._calls.append({
            "method": method,
            "mechanism": mechanism,
            "value": value,
            "direction": direction,
        })

    # ------------------------------------------------------------------ #
    # 感知（M-2 趋化 / M-1 触觉 → 感知量）
    # ------------------------------------------------------------------ #

    def sense(self, stimulus: Stimulus) -> Response:
        """环境刺激 → 感知量（percept）。

        kind="odor"/"chemotaxis" → 趋化机制：浓度 + 梯度方向（"在哪里/往哪走"）；
        kind="touch"/"reflex" → 反射机制：触碰强度 → 回避方向/强度。
        消融（关）→ 中性感知（浓度/梯度/强度归零）。
        """
        if stimulus.kind in ("odor", "chemotaxis"):
            name = "chemotaxis"
            resp = self._m(name).respond(stimulus)
            if not self._enabled.get(name, True):
                resp = Response(
                    "chemotaxis", 0.0, "none",
                    {"gradient_x": 0.0, "gradient_y": 0.0,
                     "gradient_norm": 0.0, "ablation": True})
        elif stimulus.kind in ("touch", "reflex"):
            name = "reflex"
            resp = self._m(name).respond(stimulus)
            if not self._enabled.get(name, True):
                resp = Response("reflex", 0.0, "none",
                                {"behavior_latency_ms": 0.0, "ablation": True})
        else:
            raise ValueError(
                f"sense 不支持的刺激类型：{stimulus.kind}"
                f"（支持 odor/chemotaxis/touch/reflex）")
        self._record("sense", name, resp.value, resp.direction)
        return resp

    # ------------------------------------------------------------------ #
    # 动作选择（M-1 反射 / M-2 趋化 / M-3 CPG；M-6 调质门控注入）
    # ------------------------------------------------------------------ #

    def actuate(self, intent: Dict[str, object]) -> Dict[str, object]:
        """意图 → 动作选择（先天运动/节奏底座）。

        intent["type"]：
          "escape"   → 反射：触刺激 → 定向回避（direction=back|none）
          "approach" → 趋化：位置+朝向 → 指向梯度转向（正向梯度趋利）
          "rhythm"   → CPG：时间 → 节律脉冲（frequency/phase/in_band）
        任选 intent["motivation"] ∈ [0,1] → 调质门控：动作强度 × gain
        （动机/唤醒 → 运动增益，决策影响层）。消融 → 中性动作（方向 none/
        无转向/无节律/增益≡1）。
        """
        kind = intent.get("type")
        action: Dict[str, object] = {"type": "move", "direction": "none",
                                     "strength": 0.0, "mechanisms": []}
        if kind == "escape":
            name = "reflex"
            resp = self._m(name).respond(
                Stimulus(intensity=float(intent.get("touch", 1.0))))
            on = self._enabled.get(name, True)
            action = {
                "type": "move",
                "direction": resp.direction if on else "none",
                "strength": resp.value if on else 0.0,
                "latency_ms": (resp.extra.get("behavior_latency_ms", 0.0)
                               if on else 0.0),
                "mechanisms": [name],
            }
        elif kind == "approach":
            name = "chemotaxis"
            chemo = self._m(name)
            x, y = float(intent.get("x", 5.0)), float(intent.get("y", 5.0))
            heading = float(intent.get("heading", 0.0))
            resp = chemo.respond(Stimulus(kind="odor", x=x, y=y))
            on = self._enabled.get(name, True)
            if on:
                gx, gy = resp.extra["gradient_x"], resp.extra["gradient_y"]
                g_ang = math.atan2(gy, gx)
                d = g_ang - heading
                while d > math.pi:
                    d -= 2.0 * math.pi
                while d < -math.pi:
                    d += 2.0 * math.pi
                w = chemo.p("omega_max", 1.0)
                turn = max(-w, min(w, d / (chemo.p("dt_b_ms", 25.0) / 1000.0)))
            else:
                g_ang, turn = 0.0, 0.0
            action = {
                "type": "move",
                "direction": "toward_gradient" if on else "none",
                "heading_change": turn,
                "gradient_heading": g_ang,
                "concentration": resp.value,
                "mechanisms": [name],
            }
        elif kind == "rhythm":
            name = "cpg"
            resp = self._m(name).respond(Stimulus(
                kind="time",
                t_ms=float(intent.get("t_ms", 0.0)),
                food_present=bool(intent.get("food_present", False))))
            on = self._enabled.get(name, True)
            action = {
                "type": "pulse",
                "frequency": resp.value if on else 0.0,
                "phase": resp.extra["phase"] if on else 0.0,
                "in_band": resp.extra["in_band"] if on else False,
                "food_present": bool(intent.get("food_present", False)),
                "mechanisms": [name],
            }
        else:
            raise ValueError(
                f"actuate 不支持的意图类型：{kind!r}"
                f"（支持 escape/approach/rhythm）")
        # M-6 调质 → 运动增益门控（动机/唤醒 → 增益；决策影响层）
        if "motivation" in intent and self._enabled.get("modulation", True):
            gain = self.gate(float(intent["motivation"]))
            action["gain"] = gain
            if "strength" in action:
                action["strength"] = float(action["strength"]) * gain
            action["mechanisms"] = list(action["mechanisms"]) + ["modulation"]
        name = action["mechanisms"][0]
        direction = str(action.get("direction", ""))
        value = action.get("strength", action.get("frequency", 0.0))
        self._record("actuate", name, value, direction)
        return action

    # ------------------------------------------------------------------ #
    # 适应（M-4 习惯化 → 响应衰减）
    # ------------------------------------------------------------------ #

    def adapt(self, repeat_count: int) -> Response:
        """第 n 次重复刺激 → 响应幅度 R(n)（重复刺激 → 衰减，H1 机制）。

        消融（关）→ R(n) ≡ R(1)（无衰减，行为差异消失）。
        """
        name = "habituation"
        hab = self._m(name)
        n = max(1, int(repeat_count))
        r_seq = hab.r_sequence(n)
        r = r_seq[-1]
        if not self._enabled.get(name, True):
            r = hab.p("r0", 0.353)
            direction = "initial"
        else:
            direction = "decay" if n > 1 and r < r_seq[0] else "initial"
        resp = Response(
            "habituation", r, direction,
            {"n": n, "r0": hab.p("r0", 0.353),
             "tau_hab": hab.p("tau_hab", 2.0),
             "stp_enabled": hab.p("stp_enabled", 1.0)})
        self._record("adapt", name, r, direction)
        return resp

    # ------------------------------------------------------------------ #
    # 运动增益门控（M-6 调质 → 动机/唤醒 → 增益）
    # ------------------------------------------------------------------ #

    def gate(self, motivation: float) -> float:
        """动机标量（0..1）→ 运动增益 gain ∈ [tyr_floor, 1.2]。

        消融（关）→ gain ≡ 1.0（无门控，行为差异消失）。
        """
        name = "modulation"
        mod = self._m(name)
        gain = mod.gate(motivation) if self._enabled.get(name, True) else 1.0
        self._record("gate", name, gain)
        return gain
