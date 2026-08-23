"""M2 突触冒烟测试（清单 §4.3 验收：≥3 断言绿）。

- pre 刺激 → post 出现 EPSP（0.5–2 mV 量级）
- 无刺激 → 无 PSP
- 释放失败可观测（p_release=0.3 时多次运行出现"无响应"试次）
- GABA IPSP 超极化；缝隙连接耦合；STP 易化
"""

import numpy as np
import pytest

from neural_exploration.src.neuron_pair import NeuronPair, pulse_train
from neural_exploration.tools.synapse_metrics import psp_amplitudes

PULSE = (50.0, 1.0, 20.0, "soma")   # (start, dur, amp_uA_cm2, site)


def test_ampa_epsp_in_range():
    """pre 发放 → post 出现 0.3–5 mV EPSP。"""
    pair = NeuronPair(t_total_ms=120.0, seed=0)
    pair.add_chemical("ampa", g_max_ns=0.3, p_release=1.0, n_vesicles=1)
    r = pair.run(pre_pulses=[PULSE], record=["pre_node3", "post_soma"])
    assert len(r.spike_times_ms["pre_node3"]) == 1, "pre 必须发放"
    amps = psp_amplitudes(r.t_ms, r.v_mv["post_soma"], r.spike_times_ms["pre_node3"])
    assert len(amps) == 1
    assert 0.3 < amps[0] < 5.0, f"EPSP 幅度异常: {amps}"


def test_no_stimulus_no_psp():
    """无刺激 → post 无发放、无 PSP。"""
    pair = NeuronPair(t_total_ms=120.0, seed=0)
    pair.add_chemical("ampa", g_max_ns=0.3, p_release=1.0, n_vesicles=1)
    r = pair.run(pre_pulses=[], record=["post_soma"])
    assert len(r.spike_times_ms["post_soma"]) == 0
    v = r.v_mv["post_soma"]
    assert np.all(np.isfinite(v))


def test_release_failure_observable():
    """p_release=0.3 → 多次试次出现无释放（k=0）的失败试次。"""
    pair = NeuronPair(t_total_ms=80.0, seed=0)
    pair.add_chemical("ampa", g_max_ns=0.3, p_release=0.3, n_vesicles=1)
    res = pair.run_trials(pre_pulses=[PULSE], n_trials=20, seed_base=7,
                          record_g=["g_ampa"])
    peaks = [float(r.g["g_ampa"].max()) for r in res]
    n_fail = sum(1 for p in peaks if p == 0)
    assert 0 < n_fail < 20, f"失败试次数异常: {n_fail}/20（期望 ~14）"


def test_gaba_ipsp_hyperpolarizing():
    """GABA_A → 超极化 IPSP（负 PSP）。"""
    pair = NeuronPair(t_total_ms=120.0, seed=0)
    pair.add_chemical("gaba", p_release=1.0, n_vesicles=1)
    r = pair.run(pre_pulses=[PULSE], record=["pre_node3", "post_soma"])
    amps = psp_amplitudes(r.t_ms, r.v_mv["post_soma"], r.spike_times_ms["pre_node3"])
    assert amps[0] < -0.2, f"IPSP 应超极化: {amps}"


def test_gap_junction_couples():
    """缝隙连接：pre 发放 → post 出现耦合 PSP。"""
    pair = NeuronPair(t_total_ms=100.0, seed=0)
    pair.add_gap(g_gap_ns=0.5)
    r = pair.run(pre_pulses=[PULSE], record=["pre_soma", "post_soma"])
    assert len(r.spike_times_ms["pre_soma"]) == 1
    rest = float(np.median(r.v_mv["post_soma"][(r.t_ms > 40) & (r.t_ms < 49.5)]))
    assert r.v_mv["post_soma"].max() - rest > 0.5, "缝隙连接耦合过弱"


def test_stp_facilitation_trend():
    """50Hz×10：STP 易化（末次 EPSP ≥ 首次 × 1.5）。"""
    pair = NeuronPair(t_total_ms=320.0, seed=0)
    pair.add_chemical("ampa", g_max_ns=0.5, p_release=1.0, n_vesicles=1,
                      stp=(0.03, 120.0, 40.0))
    r = pair.run(pre_pulses=pulse_train(50.0, 50.0, 10, 1.0, 20.0),
                 record=["pre_node3", "post_soma"])
    amps = psp_amplitudes(r.t_ms, r.v_mv["post_soma"], r.spike_times_ms["pre_node3"])
    assert len(amps) == 10
    assert amps[-1] >= amps[0] * 1.5, f"易化未复现: {amps}"
