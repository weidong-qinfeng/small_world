"""M7 P-A1 机制模块冒烟测试（数字大脑侧，新建文件，不改既有测试）。

验证先天机制封装（digital_brain/src/innate/）行为级等价性锚（M7 清单 §2.1）：
  M-1 反射 D_peak>0.3 → back；M-2 趋化 CI 符号正；M-3 CPG 节律双带；
  M-4 习惯化衰减 + 消融；M-5 联想 Δw 符号；M-6 调质门控范围；
  全部确定性逐位一致。
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pytest

from digital_brain.src.innate import (
    Stimulus,
    make_all,
    make_mechanism,
)


@pytest.fixture(scope="module")
def mechs():
    return make_all()


def test_m1_reflex_direction_and_anchor_band(mechs):
    r = mechs["reflex"].respond(Stimulus(intensity=1.0))
    assert r.direction == "back"
    assert r.value > 0.3                      # D_peak>0.3 → back
    assert 0.352 <= r.value <= 0.369          # M3 定稿带
    r0 = mechs["reflex"].respond(Stimulus(intensity=0.0))
    assert r0.value == 0.0 and r0.direction == "none"


def test_m2_chemotaxis_ci_sign_positive(mechs):
    m = mechs["chemotaxis"]
    for th in (0.0, 1.5707963267948966, 3.141592653589793, -1.5707963267948966):
        xs, ys = m.run_trial(theta0=th)
        assert m.ci(xs, ys) > 0.0             # CI 符号正（四朝向）
    xs, ys = m.run_trial()
    assert 0.25 <= m.ci(xs, ys) <= 0.75       # 落容差窗（冻结参考 0.494@25s）


def test_m3_cpg_rhythm_bands(mechs):
    m = mechs["cpg"]
    r0 = m.respond(Stimulus(kind="time", t_ms=1000, food_present=False))
    r1 = m.respond(Stimulus(kind="time", t_ms=1000, food_present=True))
    assert r0.value == pytest.approx(0.400, abs=1e-3)   # 冻结无食物主频
    assert r1.value == pytest.approx(2.167, abs=1e-3)   # 冻结有食物主频
    assert r0.extra["in_band"] and r1.extra["in_band"]


def test_m4_habituation_decay_and_ablation(mechs):
    m = mechs["habituation"]
    res = m.run_sequence(n_stim=6)
    assert res["fit"]["r2_ok"]                # R²≥0.5
    assert res["decay_ok"] and res["decay"] > 0.05
    assert res["half_criterion_ok"]           # 后半均值 < 0.5×前半均值
    res_off = m.run_sequence(n_stim=6, stp_enabled=False)
    assert res_off["decay"] == 0.0            # STP 关 → 无衰减（消融）


def test_m5_associative_dw_signs(mechs):
    res = mechs["associative"].full_protocol()
    assert res["acquisition_ok"]              # Δw_train > 0.1
    assert res["extinction_ok"]               # Δw_ext < 0
    assert res["eta0_ok"]                     # η=0 → Δw≈0


def test_m6_modulation_gate_range_and_ablation(mechs):
    m = mechs["modulation"]
    for mot in (0.0, 0.25, 0.5, 0.75, 1.0):
        g = m.gate(mot)
        assert 0.3 <= g <= 1.2                # gate ∈ [tyr_floor, 1.2]
    m2 = make_mechanism("modulation")
    m2.params["tyramine_enabled"] = 0.0
    assert abs(m2.gate(0.0) - 1.0) < 0.02     # 酪胺关 → gate≡1（消融 sanity）


def test_determinism_bit_identical(mechs):
    a = mechs["reflex"].respond(Stimulus(intensity=1.0)).as_dict()
    b = mechs["reflex"].respond(Stimulus(intensity=1.0)).as_dict()
    assert a == b
    assert mechs["associative"].full_protocol() == \
        mechs["associative"].full_protocol()
    r1 = mechs["habituation"].run_sequence(n_stim=6, rest_ms=2000.0)
    r2 = mechs["habituation"].run_sequence(n_stim=6, rest_ms=2000.0)
    assert r1 == r2
