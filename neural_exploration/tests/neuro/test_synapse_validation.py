"""M2 逐项生理验证测试（P1–P5 判定；复用 tools/validate_* 模块）。

与 M1 的 test_multicomp_validation.py 同构：真实运行 Brian2 模型 + NEURON 参考解，
保证 P1–P5 判定可复现。运行时间：约 3–5 分钟（编译缓存预热后）。
"""

import pytest

from neural_exploration.tools import (
    validate_p1_waveform,
    validate_p2_failure,
    validate_p3_stp,
    validate_p4_gap,
    validate_p5_receptor,
)


def test_p1_epsp_ipsp_waveform_error_below_10_percent():
    """P1：EPSP/IPSP 波形 vs NEURON 参考解，归一化 RMSE < 10%。"""
    res = validate_p1_waveform.run_p1(save_plot=False)
    assert res["pass_"], f"P1 FAIL: {res['cases']}"
    for key, j in res["cases"].items():
        assert j["norm_rmse"] < 0.10, f"{key} norm_rmse={j['norm_rmse']:.4f}"
        assert abs(j["amp_ratio"] - 1.0) < 0.10, f"{key} amp_ratio={j['amp_ratio']:.3f}"


def test_p2_failure_rate_matches_binomial():
    """P2：释放失败率落入二项模型 (1-p)^n 的 95% 置信区间。"""
    res = validate_p2_failure.run_p2(save_plot=False, n_trials=60)
    assert res["pass_"], f"P2 FAIL: p_hat={res['p_hat']:.3f} CI={res['ci']}"
    lo, hi = res["ci"]
    assert lo <= res["expected_failure"] <= hi
    assert res["mean_ok"], f"P2 FAIL: mean quanta {res['mean_quanta']:.2f}"


def test_p3_stp_facilitation_and_depression():
    """P3：短期易化与抑制都复现（50Hz×10）。"""
    res = validate_p3_stp.run_p3(save_plot=False)
    assert res["pass_"], f"P3 FAIL: fac={res['facilitation_ok']} dep={res['depression_ok']}"
    assert res["facilitation_ok"]
    assert res["depression_ok"]


def test_p4_gap_junction_instant_bidirectional():
    """P4：缝隙连接近即时、双向、衰减快、量级与参考一致。"""
    res = validate_p4_gap.run_p4(save_plot=False)
    assert res["pass_"], f"P4 FAIL: {res}"
    assert res["instant_ok"]
    assert res["bidirectional_ok"]
    assert res["decay_ok"]


def test_p5_receptor_subtypes_ampa_vs_nmda():
    """P5：AMPA 快 vs NMDA 慢 + Mg²⁺ 阻断 + 电压依赖。"""
    res = validate_p5_receptor.run_p5(save_plot=False)
    assert res["pass_"], f"P5 FAIL: {res}"
    assert res["fast_ok"] and res["slow_ok"], "快/慢区分失败"
    assert res["mg_ok"], f"Mg²⁺ 阻断失败: ratio={res['mg_ratio']:.3f}"
    assert res["vdep_ok"], f"电压依赖失败: max_err={res['vdep_max_err']:.4f}"
