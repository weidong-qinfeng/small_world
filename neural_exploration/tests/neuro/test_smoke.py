"""M0 最小闭环冒烟测试（清单 §6.3 三断言）。"""

import numpy as np
import pytest

from neural_exploration.src.smoke_loop import run_smoke_loop

STIM = 10.0


def test_stimulus_produces_response():
    """注入刺激 → 运动神经元发放 → 肌肉收缩量 > 0。"""
    result = run_smoke_loop(stimulus_amp=STIM)
    assert result.motor_spikes > 0
    assert result.muscle_contraction > 0
    assert result.muscle_max > 0


def test_response_reproducible():
    """同参数重跑 → 结果一致（确定性验证，本实验铁律之一）。"""
    r1 = run_smoke_loop(stimulus_amp=STIM)
    r2 = run_smoke_loop(stimulus_amp=STIM)
    assert r1 == r2


def test_no_stimulus_no_response():
    """无刺激 → 无发放、无收缩（对照）。"""
    result = run_smoke_loop(stimulus_amp=0.0)
    assert result.motor_spikes == 0
    assert result.muscle_contraction == 0.0
    assert result.sensory_spikes == 0


def test_traces_are_finite():
    """轨迹无 NaN/Inf（数值健全性）。"""
    result = run_smoke_loop(stimulus_amp=STIM)
    assert np.isfinite(result.v_sensory).all()
    assert np.isfinite(result.v_motor).all()
    assert np.isfinite(result.contraction).all()
