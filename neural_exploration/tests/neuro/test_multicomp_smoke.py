"""M1 多隔室冒烟测试（清单 §4.3 验收：3 断言绿）。"""

import numpy as np
import pytest

from neural_exploration.src.neuron_model import MultiCompartmentNeuron

STIM = 10.0          # µA/cm²
T_TOTAL = 40.0       # ms（足够传导到轴突末端）


@pytest.fixture(scope="module")
def neuron():
    n = MultiCompartmentNeuron(t_total_ms=T_TOTAL)
    n.build()
    return n


def test_soma_fires(neuron):
    """胞体注入 10 µA/cm² → 胞体发放 >0。"""
    r = neuron.run_stimulus(amplitude_uA_cm2=STIM, stim_start_ms=5.0,
                            stim_end_ms=6.0, record=["soma"])
    assert len(r.spike_times_ms["soma"]) > 0


def test_axon_end_fires(neuron):
    """动作电位传导到轴突末端（node3）→ 末端发放 >0。"""
    r = neuron.run_stimulus(amplitude_uA_cm2=STIM, stim_start_ms=5.0,
                            stim_end_ms=6.0, record=["node3"])
    assert len(r.spike_times_ms["node3"]) > 0


def test_traces_finite_and_no_spikes_without_stim(neuron):
    """轨迹有限 + 无刺激对照（不发放）。"""
    r0 = neuron.run_stimulus(amplitude_uA_cm2=0.0, record=["soma", "node3"])
    assert np.isfinite(r0.v_mv["soma"]).all()
    assert np.isfinite(r0.v_mv["node3"]).all()
    assert len(r0.spike_times_ms["soma"]) == 0

    r1 = neuron.run_stimulus(amplitude_uA_cm2=STIM, stim_start_ms=5.0,
                             stim_end_ms=6.0, record=["soma", "node3"])
    assert np.isfinite(r1.v_mv["soma"]).all()
    assert np.isfinite(r1.v_mv["node3"]).all()
    assert len(r1.spike_times_ms["soma"]) > 0
    assert len(r1.spike_times_ms["node3"]) > 0
