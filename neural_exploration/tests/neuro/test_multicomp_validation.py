"""M1 逐项生理验证测试（P2–P6 判定；复用 tools/validate_* 模块）。

这些测试会真实运行 Brian2 模型（与验证脚本同源），保证 P2–P6 判定可复现。
运行时间：约 1–2 分钟（含 NEURON 参考解比对）。
"""

import pytest

from neural_exploration.tools import (
    validate_p2_waveform,
    validate_p3_fi,
    validate_p4_psp,
    validate_p5_speed,
    validate_p6_saltatory,
)


def test_p2_waveform_error_below_5_percent():
    """P2：Brian2 vs NEURON 参考解，归一化 RMSE < 5%（胞体/树突端/轴突端）。"""
    res = validate_p2_waveform.run_p2(save_plot=False)
    assert res["pass_"], f"P2 FAIL: {res['per_location']}"
    for loc, v in res["per_location"].items():
        assert v["norm_rmse"] < 0.05, f"{loc} norm_rmse={v['norm_rmse']:.4f}"


def test_p3_fi_curve_monotonic_with_threshold():
    """P3：f-I 单调递增 + 存在阈值（0 → >0）。"""
    res = validate_p3_fi.run_p3(save_plot=False)
    assert res["monotonic"], f"P3 FAIL: f-I 不单调 {res['freqs']}"
    assert res["has_threshold"], "P3 FAIL: 无阈值（0 → >0）"
    assert res["pass_"]


def test_p4_psp_attenuation_and_delay():
    """P4：树突→胞体 PSP 衰减、时延、τ 在 5–20ms。"""
    res = validate_p4_psp.run_p4(save_plot=False)
    assert res["pass_"], f"P4 FAIL: {res}"


def test_p5_conduction_velocity_in_range():
    """P5：轴突传导速度在生理范围（有髓鞘 1–20 m/s）。"""
    res = validate_p5_speed.run_p5()
    assert res["pass_"], f"P5 FAIL: cv={res['mean_cv_mps']}"


def test_p6_saltatory_conduction():
    """P6：郎飞结依次发放（跳跃）、髓鞘无主动通道、无全幅峰。"""
    res = validate_p6_saltatory.run_p6(save_plot=False)
    assert res["all_nodes_fire"], f"P6 FAIL: nodes={res['nodes']}"
    assert res["monotonic"], f"P6 FAIL: 节点时序不递增 {res['nodes']}"
    assert res["no_active_channels"], "P6 FAIL: 髓鞘段存在主动通道"
    assert res["no_full_amplitude"], ("P6 FAIL: 髓鞘出现全幅峰 "
                                      f"(myelin={res['myelin_peak_max_mv']:.1f} > "
                                      f"{res['full_amp_ratio']}×source={res['source_peak_mv']:.1f})")
    assert res["pass_"]
