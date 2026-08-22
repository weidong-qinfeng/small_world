"""P1 确定性验证（清单 §0 P1：重复运行逐位一致）。"""

import numpy as np

from neural_exploration.src.neuron_model import MultiCompartmentNeuron


def test_repeated_runs_bitwise_identical():
    """同参数重跑 → V 轨迹逐位一致（确定性铁律）。"""
    n = MultiCompartmentNeuron(t_total_ms=40.0)
    n.build()
    r1 = n.run_stimulus(amplitude_uA_cm2=10.0, stim_start_ms=5.0,
                        stim_end_ms=6.0, record=["soma", "node3"])
    r2 = n.run_stimulus(amplitude_uA_cm2=10.0, stim_start_ms=5.0,
                        stim_end_ms=6.0, record=["soma", "node3"])
    assert np.array_equal(r1.v_mv["soma"], r2.v_mv["soma"])
    assert np.array_equal(r1.v_mv["node3"], r2.v_mv["node3"])
    assert np.array_equal(r1.spike_times_ms["soma"], r2.spike_times_ms["soma"])


def test_no_stimulus_deterministic_baseline():
    """无刺激对照重跑一致（基线确定性）。"""
    n = MultiCompartmentNeuron(t_total_ms=20.0)
    n.build()
    a = n.run_stimulus(amplitude_uA_cm2=0.0, record=["soma"])
    b = n.run_stimulus(amplitude_uA_cm2=0.0, record=["soma"])
    assert np.array_equal(a.v_mv["soma"], b.v_mv["soma"])
