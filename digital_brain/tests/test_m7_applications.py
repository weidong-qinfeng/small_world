"""M7 P-A3 应用题场景测试——迁移机制通过数字大脑应用题（新建文件，不改既有测试）。

预注册（M7 清单 §0 #5）：场景落在认知层现有能力内（0-20 加法 + 关系推理）——
认知层问题使用 base_curriculum **已验证模板**（srl_parse 求和链，与既有 M1 应用题
同构）；场景叙事（厨房/香气/火炉/钟声）由**机制层断言**承载（感知/运动底座语义，
D3 层位：机制层在下，认知层在上）。场景词汇经标准 `learn_word` 接口教学（大脑初始
唯一能力 = 学词；推理能力未扩展——"教新词"非"教新推理"，如实记录于 L23+）。

场景（≥3，M7 清单 §5）：
  S1 闻香走向厨房   —— 趋化机制 → 环境感知辅助决策（梯度指向厨房/浓度递增/趋利轨迹）
  S2 碰到火炉缩手+习惯化 —— 反射弧 → 先天运动反应 + 习惯化 → 适应（R(n) 递减）
  S3 钟声节奏       —— CPG → 行为节奏（节拍频率 ∈ 预注册带宽）
  S4（扩展）饿了跑得更快 —— 调质 → 运动增益门控（motivation↑ → gain↑）

每场景断言：(a) 机制层响应正确（方向/衰减/节律/增益）(b) 认知层 solve 推理链完整
(c) innate=None 对照：机制断言不适用（test_innate_none_control 如实记录，不伪造）。
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pytest

from digital_brain.src.innate import Stimulus, make_all
from digital_brain.src.interfaces.innate_interface import InnateInterface
from digital_brain.src.interfaces.symbolic_interface import SymbolicInterface

# 场景词汇（SRL 角色表对齐：noun→THEME / verb_acquire→VERB_ACQUIRE /
# classifier→QUANTITY 量词；pos 见 semantic_labeler.POS_TO_ROLE）
SCENE_VOCAB = [
    ("苹果", "一种水果", "名词", "noun"),
    ("个", "通用量词", "量词", "classifier"),
    ("厨房", "做饭的房间", "名词", "noun"),
    ("缩", "收回（获取类动作）", "动词", "verb_acquire"),
    ("手", "手掌", "名词", "noun"),
    ("火炉", "取暖的炉子", "名词", "noun"),
    ("碰", "触碰（获取类动作）", "动词", "verb_acquire"),
    ("钟声", "钟的声音", "名词", "noun"),
    ("响", "发出声音（获取类动作）", "动词", "verb_acquire"),
    ("步", "步伐", "名词", "noun"),
    ("走", "行走（获取类动作）", "动词", "verb_acquire"),
    ("米", "长度单位", "名词", "noun"),
    ("跑", "奔跑（获取类动作）", "动词", "verb_acquire"),
]


def _make_brain(with_innate: bool = True) -> SymbolicInterface:
    b = SymbolicInterface(
        auto_build=False,
        auto_learn_tokenizer=False,
        innate=InnateInterface(make_all()) if with_innate else None,
    )
    b.learn_from_package("base_curriculum")
    for symbol, meaning, word_type, pos in SCENE_VOCAB:
        b.learn_word(symbol, meaning=meaning, word_type=word_type, pos=pos)
    return b


@pytest.fixture(scope="module")
def brain():
    return _make_brain(with_innate=True)


def _assert_cognitive_chain(brain, question, expected):
    """(b) 认知层：solve 返回完整推理链 + 正确答案（0-20 加法，现有能力内）。"""
    r = brain.solve(question)
    assert r.answer == expected, f"{question} 期望 {expected}，实际 {r.answer}"
    actions = [s["action"] for s in r.reasoning_chain]
    assert "srl_parse" in actions, f"{question} 未走 srl_parse 链"
    assert "call_algorithm" in actions and "return_value" in actions, \
        f"{question} 推理链不完整：{actions}"


# ============================================================
# S1 闻香走向厨房（趋化 → 环境感知辅助决策）
# ============================================================

def test_s1_smell_kitchen_chemotaxis(brain):
    """S1：闻到香味走向厨房——机制层趋化驱动"走向食物"感知-运动链 + 认知层推理链完整"""
    innate = brain.innate
    # (a) 机制层：环境感知（趋化）
    living = innate.sense(Stimulus(kind="odor", x=5.0, y=5.0))      # 客厅闻香
    assert living.direction == "toward_gradient"
    assert living.value > 0.0                                        # 闻得到
    # 正向梯度趋利：梯度指向厨房（food 在东北象限 (7.5,7.5)）
    assert living.extra["gradient_x"] > 0.0 and living.extra["gradient_y"] > 0.0
    # 沿厨房方向走 → 浓度递增（目标方向 = 正向梯度）
    mid = innate.sense(Stimulus(kind="odor", x=6.0, y=6.0)).value
    assert mid > living.value
    # 感知-运动链：从不利朝向（朝西）转向厨房
    steer = innate.actuate(
        {"type": "approach", "x": 5.0, "y": 5.0, "heading": 3.141592653589793})
    assert steer["direction"] == "toward_gradient"
    assert steer["heading_change"] != 0.0
    # 趋利轨迹：Braitenberg 限速转向试次 CI > 0（落容差窗 [0.25, 0.75]）
    xs, ys = innate.mechanisms["chemotaxis"].run_trial(
        start=(5.0, 5.0), theta0=3.141592653589793)
    ci = innate.mechanisms["chemotaxis"].ci(xs, ys)
    assert 0.25 <= ci <= 0.75
    # (b) 认知层：厨房食物数量应用题（0-20 加法，现有能力内）
    _assert_cognitive_chain(
        brain, "小明包里有4个苹果，妈妈又给了小明2个苹果，现在小明一共有几个苹果？", 6)


# ============================================================
# S2 碰到火炉缩手 + 习惯化（反射弧 → 先天运动反应 + 习惯化 → 适应）
# ============================================================

def test_s2_stove_reflex_habituation(brain):
    """S2：碰到火炉缩手/重复后反应减弱——反射方向 + 习惯化衰减"""
    innate = brain.innate
    # (a) 机制层：反射弧 → 先天运动反应（第一次触碰 → 定向回避）
    act = innate.actuate({"type": "escape", "touch": 1.0})
    assert act["direction"] == "back"                    # 缩手 = 回避方向
    assert act["strength"] > 0.3                         # D_peak > 0.3 → back
    # 习惯化 → 适应：重复警告 → 反应幅度 R(n) 递减
    seq = [innate.adapt(n).value for n in range(1, 7)]
    assert all(seq[i] > seq[i + 1] for i in range(len(seq) - 1))   # 严格递减
    assert innate.adapt(2).direction == "decay"
    half = len(seq) // 2
    assert sum(seq[half:]) / half < 0.5 * sum(seq[:half]) / half    # 衰减形状判据
    # (b) 认知层：火炉场景次数应用题（0-20 加法，现有能力内）
    _assert_cognitive_chain(
        brain, "小明碰到火炉缩了2次手，又缩了1次手，一共缩了几次手？", 3)


# ============================================================
# S3 钟声节奏（CPG → 行为节奏）
# ============================================================

def test_s3_bell_rhythm_cpg(brain):
    """S3：钟声节奏——CPG 节拍频率 ∈ 预注册带宽 + 相位推进；认知层走步计数"""
    innate = brain.innate
    # (a) 机制层：行为节奏（CPG）——上课铃（无食物态）0.400Hz ∈ [0.1,2]
    bell = innate.actuate({"type": "rhythm", "t_ms": 1000.0})
    assert bell["type"] == "pulse"
    assert bell["frequency"] == pytest.approx(0.400, abs=1e-3)
    assert bell["in_band"] is True                       # 节拍频率落预注册带宽
    assert 0.0 <= bell["phase"] < 1.0
    # 节律推进：不同时刻相位不同（确定性）
    phase_t1 = innate.actuate({"type": "rhythm", "t_ms": 2500.0})["phase"]
    assert phase_t1 != bell["phase"]
    # 进食态（有食物）2.167Hz ∈ [2,5]——双带切换（M5 P3 冻结）
    feed = innate.actuate({"type": "rhythm", "t_ms": 1000.0,
                           "food_present": True})
    assert feed["frequency"] == pytest.approx(2.167, abs=1e-3)
    assert feed["in_band"] is True
    # (b) 认知层：节奏下走步计数（0-20 加法，现有能力内）
    _assert_cognitive_chain(
        brain, "小明走了5步，又走了3步，一共走了几步？", 8)


# ============================================================
# S4（扩展集）饿了跑得更快（调质 → 运动增益门控）
# ============================================================

def test_s4_hunger_gain_modulation(brain):
    """S4：饿了跑得更快——动机↑ → 运动增益↑（增益 > 基线），且影响动作强度"""
    innate = brain.innate
    # (a) 机制层：调质门控（动机/唤醒 → 增益）
    g0, g1 = innate.gate(0.0), innate.gate(1.0)
    assert 0.3 <= g0 <= 1.2 and 0.3 <= g1 <= 1.2
    assert g1 > g0                                      # 饥饿（动机↑）→ 增益↑
    # 增益门控注入运动输出：饥饿时动作强度更大
    act_hungry = innate.actuate(
        {"type": "escape", "touch": 1.0, "motivation": 1.0})
    act_calm = innate.actuate(
        {"type": "escape", "touch": 1.0, "motivation": 0.0})
    assert act_hungry["gain"] > act_calm["gain"] > 0.0
    assert act_hungry["strength"] > act_calm["strength"]
    # (b) 认知层：跑步距离应用题（0-20 加法，现有能力内）
    _assert_cognitive_chain(
        brain, "小明跑了3米，又跑了4米，一共跑了多少米？", 7)


# ============================================================
# 回归 + 对照
# ============================================================

def test_original_application_regression(brain):
    """原有应用题回归（P-A3 判据：原有应用题测试全绿）——M1 经典题照常可解"""
    r = brain.solve("小明包里有4本故事书,妈妈又给他了3本,现在小明总共有几本?")
    assert r.answer == 7
    actions = [s["action"] for s in r.reasoning_chain]
    assert "srl_parse" in actions and "return_value" in actions


def test_innate_none_control():
    """(c) innate=None 对照：机制断言不适用（如实记录，不伪造）——solve 仍可解"""
    plain = _make_brain(with_innate=False)
    assert plain.innate is None                          # 无机制层
    # 机制层断言在 innate=None 时不适用（无 sense/actuate/adapt/gate 入口）——
    # 认知层照常可解（机制层与认知层解耦，组合复用纪律）
    r = plain.solve("小明包里有4个苹果，妈妈又给了小明2个苹果，现在小明一共有几个苹果？")
    assert r.answer == 6
    r2 = plain.solve("小明碰到火炉缩了2次手，又缩了1次手，一共缩了几次手？")
    assert r2.answer == 3
    r3 = plain.solve("小明走了5步，又走了3步，一共走了几步？")
    assert r3.answer == 8
