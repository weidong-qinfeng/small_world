"""M7 P-A2 数字大脑接入测试——InnateInterface + SymbolicInterface innate 注入（新建文件）。

断言覆盖（M7 清单 §0 P-A2：机制可观察断言）：
  1. 机制存在性：注入后 `brain.innate` 存在且为 InnateInterface；未注入 → None；
  2. 可调参数暴露：机制 params 可读可调，调参后行为变化可测（如 d_peak_thr 调高
     → 触碰不再触发 back 回避）；
  3. 注入前后行为差异可测（消融 sanity，M6 惯例）：set_enabled(False) → 该机制
     贡献归零（反射无回避方向 / 趋化无转向 / 习惯化无衰减 / 调质 gain≡1）；
  4. 机制层确实被调用（calls 调用日志）+ 调质门控影响运动输出（gain 乘到强度）；
  5. 零回归：solve 语义零修改（有/无 innate 同题同答）。

机制 → 接入映射（D2）：趋化→环境感知（sense 浓度/梯度）、反射→先天运动反应
（actuate escape）、CPG→行为节奏（actuate rhythm）、习惯化→适应（adapt）、
调质→运动增益门控（gate 注入 actuate）。认知层推理链不被机制层替换（预注册 §0 #9）。
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pytest

from digital_brain.src.innate import Stimulus, make_all
from digital_brain.src.interfaces.innate_interface import InnateInterface
from digital_brain.src.interfaces.symbolic_interface import SymbolicInterface

FOOD_X = 7.5   # m7_innate_params.csv chemotaxis.food_x（厨房/食物滴位置）
FOOD_Y = 7.5


@pytest.fixture(scope="module")
def brain():
    """带先天机制层注入 + base_curriculum 的数字大脑（认知层+机制层组合）。"""
    b = SymbolicInterface(
        auto_build=False,
        auto_learn_tokenizer=False,
        innate=InnateInterface(make_all()),
    )
    b.learn_from_package("base_curriculum")
    return b


# ============================================================
# 1. 机制存在性（P-A2 断言①）
# ============================================================

def test_innate_optional_attribute_default_none():
    """未注入 → brain.innate is None（既有构造零变化）"""
    b = SymbolicInterface(auto_build=False, auto_learn_tokenizer=False)
    assert b.innate is None


def test_innate_injected_existence(brain):
    """注入后 → brain.innate 存在且为 InnateInterface"""
    assert isinstance(brain.innate, InnateInterface)
    assert brain.innate is not None


def test_mechanism_existence_and_routing(brain):
    """机制存在性：六机制全注入，可路由子集齐全"""
    names = set(brain.innate.names)
    assert {"reflex", "chemotaxis", "cpg", "habituation",
            "associative", "modulation"} <= names
    for n in ("reflex", "chemotaxis", "cpg", "habituation", "modulation"):
        assert brain.innate.has(n)
        assert brain.innate.is_enabled(n)


# ============================================================
# 2. 可调参数暴露（P-A2 断言②：params 可读可调 → 行为差异可测）
# ============================================================

def test_mechanism_params_exposed_and_tunable(brain):
    """参数暴露：从 m7_innate_params.csv 唯一定稿源读取；调参 → 行为变化可测"""
    reflex = brain.innate.mechanisms["reflex"]
    assert reflex.p("d_peak_thr") == 0.3          # CSV 定稿阈值
    assert reflex.p("d_peak_max") == pytest.approx(0.610, abs=1e-9)
    assert reflex.respond(Stimulus(intensity=1.0)).direction == "back"
    # 调参（可调参数暴露）：阈值调高到 0.9 → 基准触碰不再触发回避
    reflex.params["d_peak_thr"] = 0.9
    assert reflex.respond(Stimulus(intensity=1.0)).direction == "none"
    reflex.params["d_peak_thr"] = 0.3             # 还原（确定性）
    assert reflex.respond(Stimulus(intensity=1.0)).direction == "back"


# ============================================================
# 3. 机制层接入语义（趋化→环境感知 / 反射 / CPG / 习惯化 / 调质）
# ============================================================

def test_sense_chemotaxis_environment_perception(brain):
    """趋化→环境感知：sense(odor@pos) 读"在哪里"——浓度 + 梯度方向"""
    p = brain.innate.sense(Stimulus(kind="odor", x=5.0, y=5.0))
    assert p.kind == "chemotaxis"
    assert p.direction == "toward_gradient"
    assert p.value > 0.0                           # 闻得到（浓度>背景）
    # 梯度指向厨房（食物滴 (7.5,7.5) 在东北）：gx>0 且 gy>0
    assert p.extra["gradient_x"] > 0.0
    assert p.extra["gradient_y"] > 0.0
    # 向厨房方向走 → 浓度递增（正向梯度趋利）
    c_near = brain.innate.sense(Stimulus(kind="odor", x=6.0, y=6.0)).value
    assert c_near > p.value


def test_actuate_escape_reflex_and_ablation(brain):
    """反射→先天运动反应：触刺激 → 定向回避；消融 → 无回避方向"""
    act = brain.innate.actuate({"type": "escape", "touch": 1.0})
    assert act["direction"] == "back"
    assert act["strength"] > 0.3                   # D_peak > 0.3 → back（M3 判据）
    assert "reflex" in act["mechanisms"]
    brain.innate.set_enabled("reflex", False)      # 消融：行为差异消失
    act_off = brain.innate.actuate({"type": "escape", "touch": 1.0})
    assert act_off["direction"] == "none"
    assert act_off["strength"] == 0.0
    assert act_off["strength"] != act["strength"]  # 注入前后行为差异可测
    brain.innate.set_enabled("reflex", True)


def test_actuate_approach_steering_and_ablation(brain):
    """趋化→环境感知辅助决策：远离厨房朝向 → 机制转向趋利；消融 → 无转向"""
    act = brain.innate.actuate(
        {"type": "approach", "x": 5.0, "y": 5.0, "heading": 3.141592653589793})
    assert act["direction"] == "toward_gradient"
    assert act["heading_change"] != 0.0            # 从朝西(π)转向东北梯度
    assert act["concentration"] > 0.0
    brain.innate.set_enabled("chemotaxis", False)
    act_off = brain.innate.actuate(
        {"type": "approach", "x": 5.0, "y": 5.0, "heading": 3.141592653589793})
    assert act_off["direction"] == "none"
    assert act_off["heading_change"] == 0.0
    assert act_off["heading_change"] != act["heading_change"]
    brain.innate.set_enabled("chemotaxis", True)


def test_actuate_rhythm_cpg(brain):
    """CPG→行为节奏：时间 → 节律频率/相位；消融 → 无节律"""
    act = brain.innate.actuate({"type": "rhythm", "t_ms": 1000.0})
    assert act["type"] == "pulse"
    assert act["frequency"] == pytest.approx(0.400, abs=1e-3)  # 冻结无食物主频
    assert act["in_band"] is True                              # 落预注册带 [0.1,2]
    act_food = brain.innate.actuate(
        {"type": "rhythm", "t_ms": 1000.0, "food_present": True})
    assert act_food["frequency"] == pytest.approx(2.167, abs=1e-3)
    assert act_food["in_band"] is True                         # 落带 [2,5]
    brain.innate.set_enabled("cpg", False)
    act_off = brain.innate.actuate({"type": "rhythm", "t_ms": 1000.0})
    assert act_off["frequency"] == 0.0
    assert act_off["in_band"] is False
    brain.innate.set_enabled("cpg", True)


def test_adapt_habituation_decay_and_ablation(brain):
    """习惯化→适应：重复刺激 → 响应衰减；消融 → 无衰减"""
    r1 = brain.innate.adapt(1).value
    r6 = brain.innate.adapt(6).value
    assert r1 > r6                                        # R(n) 递减
    assert brain.innate.adapt(2).direction == "decay"
    # 衰减形状：后半均值 < 0.5 × 前半均值（P-A1 判据）
    seq = [brain.innate.adapt(n).value for n in range(1, 7)]
    half = len(seq) // 2
    assert sum(seq[half:]) / half < 0.5 * sum(seq[:half]) / half
    brain.innate.set_enabled("habituation", False)
    assert brain.innate.adapt(1).value == brain.innate.adapt(6).value  # 无衰减
    brain.innate.set_enabled("habituation", True)


def test_gate_modulation_gain_and_ablation(brain):
    """调质→运动增益门控：动机↑ → 增益单调↑ 且 ∈ [tyr_floor, 1.2]；消融 → ≡1"""
    gains = [brain.innate.gate(m) for m in (0.0, 0.5, 1.0)]
    assert all(0.3 <= g <= 1.2 for g in gains)
    assert gains[0] < gains[1] < gains[2]                 # 单调（动机↑ → gain↑）
    brain.innate.set_enabled("modulation", False)
    assert brain.innate.gate(1.0) == pytest.approx(1.0, abs=1e-9)  # 消融 → gain≡1
    brain.innate.set_enabled("modulation", True)


def test_actuate_gain_gating_affects_motor_output(brain):
    """调质门控注入决策：动机 → 动作强度 × gain（运动增益门控可观察）"""
    act_low = brain.innate.actuate(
        {"type": "escape", "touch": 1.0, "motivation": 0.0})
    act_hi = brain.innate.actuate(
        {"type": "escape", "touch": 1.0, "motivation": 1.0})
    assert "modulation" in act_low["mechanisms"]
    assert act_hi["strength"] > act_low["strength"] > 0.0
    assert act_low["gain"] < act_hi["gain"]


# ============================================================
# 4. 机制层确实被调用（P-A2 断言④：calls 调用日志）
# ============================================================

def test_observability_call_log(brain):
    """调用日志：每次 sense/actuate/adapt/gate 落一条记录（机制层确实被调用）"""
    brain.innate.clear_calls()
    assert brain.innate.calls == []
    brain.innate.sense(Stimulus(kind="odor", x=5.0, y=5.0))
    brain.innate.actuate({"type": "escape", "touch": 1.0})
    brain.innate.actuate({"type": "rhythm", "t_ms": 1000.0})
    brain.innate.adapt(3)
    brain.innate.gate(0.5)
    calls = brain.innate.calls
    assert len(calls) == 5
    mechs = {c["mechanism"] for c in calls}
    assert mechs == {"chemotaxis", "reflex", "cpg", "habituation", "modulation"}
    methods = [c["method"] for c in calls]
    assert methods == ["sense", "actuate", "actuate", "adapt", "gate"]
    # 确定性：同操作重跑，日志逐位一致
    brain.innate.clear_calls()
    brain.innate.sense(Stimulus(kind="odor", x=5.0, y=5.0))
    assert brain.innate.calls == [calls[0]]


# ============================================================
# 5. 零回归（solve 语义零修改：有/无 innate 同题同答）
# ============================================================

def test_solve_zero_regression_with_innate(brain):
    """注入 innate 后 solve 语义零修改——原有应用题/二元运算照常"""
    assert brain.solve("1+1=?").answer == 2
    assert brain.solve("10-3=?").answer == 7
    r = brain.solve("爸爸的年龄31，妈妈的年龄25，他们的年龄之和是多少？")
    assert r.answer == 56
    assert "dag_build" in [s["action"] for s in r.reasoning_chain]


def test_solve_same_with_and_without_innate(brain):
    """注入前后 solve 行为一致（117 零回归的组合复用纪律）"""
    b_plain = SymbolicInterface(auto_build=False, auto_learn_tokenizer=False)
    b_plain.learn_from_package("base_curriculum")
    assert b_plain.innate is None
    q = "小明包里有4本故事书,妈妈又给他了3本,现在小明总共有几本?"
    assert b_plain.solve(q).answer == brain.solve(q).answer == 7
    assert b_plain.solve("3+5=?").answer == brain.solve("3+5=?").answer == 8
