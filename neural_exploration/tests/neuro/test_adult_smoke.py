"""M9 成年果蝇身体冒烟测试（清单 §3.3 验收：`tests/neuro/test_adult_smoke.py`；纯 CPU）。

覆盖（对应 P3 判据；判据带**只读** `data/m9_behavior_reference.csv` body 段，参数只读
`data/m9_adult_body_params.csv`——定稿不事后调）：

  1. **步态相序**：tripod 组间相位差 ∈ [150,210]°、组内 ≤30°、步频落带 [5,10] Hz；
  2. **单通道驱动冒烟**：`C_leg_fwd` 前向位移 >0 且 `C_leg_back` 反号（位移方向正确）、
     `C_leg_left/right` 转向符号相反、`C_curl` 曲率饱和且有界、`C_wing` 翅拍频率落带、
     `C_escape` 翅展落带；
  3. **状态分类语义**：run/curl/pause 驱动下对应状态占比 >0.5（阈值 CSV 定稿）；
  4. **巨纤维逃跑 sanity**：光刺激反应概率 ≥0.8；亚阈值（0.4×阈值）≤0.2（剂量-反应）；
  5. **确定性 + 有界**：同参数重跑**逐位一致**；关节角/位置/速度有限、角 ≤180°、无 NaN。

运行：`PYTHONHASHSEED=0 MPLBACKEND=Agg .venv-m9/bin/python -m pytest
neural_exploration/tests/neuro/test_adult_smoke.py -q -p no:cacheprovider`
"""

import os

import numpy as np
import pytest

from neural_exploration.src.adult_body import (
    AdultFlyBody,
    AdultBodyParams,
    LEGS,
    TRIPOD_A,
    TRIPOD_B,
    state_fractions,
)
from neural_exploration.tools.validate_p9_body import read_band, main_peak_freq, phase_diff_deg

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data")

P = AdultBodyParams.from_csv()
BAND = read_band("body")
DT = P.dt_ms


def _in(metric, val):
    b = BAND.get(metric)
    if b is None:
        pytest.skip("判据带缺失：%s" % metric)
    return b["lo"] <= val <= b["hi"]


# ---------------- 1. 步态相序 ----------------
def test_tripod_phase_and_frequency():
    b = AdultFlyBody(P); b.reset()
    r = b.run({"C_leg_fwd": 1.0}, 2000.0)
    lift = {leg: r["joints_deg"][:, i, 1] for i, leg in enumerate(LEGS)}
    f = main_peak_freq(lift["L1"], DT)
    pd_cross = phase_diff_deg(lift[TRIPOD_A[0]], lift[TRIPOD_B[0]], P.gait_freq_hz, DT)
    assert _in("gait_freq_hz", f), "步频 %.3f Hz 出带" % f
    assert _in("tripod_phase_deg", pd_cross), "组间相位差 %.1f° 出带" % pd_cross
    intra = max(min(phase_diff_deg(lift[a], lift[c], P.gait_freq_hz, DT),
                    360.0 - phase_diff_deg(lift[a], lift[c], P.gait_freq_hz, DT))
                for a, c in (("L1", "R2"), ("R2", "L3"), ("L1", "L3")))
    assert _in("tripod_intragroup_phase_deg", intra), "组内相位差 %.1f° 出带" % intra


# ---------------- 2. 单通道驱动冒烟 ----------------
def test_channel_forward_backward():
    b = AdultFlyBody(P); b.reset(); rf = b.run({"C_leg_fwd": 1.0}, 1000.0)
    b.reset(); rb = b.run({"C_leg_back": 1.0}, 1000.0)
    assert rf["x"][-1] > 0 > rb["x"][-1], "前进/后退位移方向不正确"
    assert _in("v_fwd_magnitude", abs(float(rf["x"][-1]))), "前向位移量级出带"


def test_channel_turn_direction():
    b = AdultFlyBody(P); b.reset(); tl = b.run({"C_leg_left": 1.0, "C_leg_fwd": 1.0}, 1000.0)
    b.reset(); tr = b.run({"C_leg_right": 1.0, "C_leg_fwd": 1.0}, 1000.0)
    assert tl["theta"][-1] > 0 > tr["theta"][-1], "左右差动转向符号不正确"


def test_channel_curl_saturates_and_suppresses_displacement():
    b = AdultFlyBody(P); b.reset(); rc = b.run({"C_curl": 1.0}, 1000.0)
    assert _in("curl_curvature_max", float(np.max(rc["curl"]))), "curl 峰值出带"
    assert abs(float(rc["x"][-1])) < 0.5, "防御蜷缩态位移未抑制"
    assert np.max(rc["curl"]) <= 1.0 + 1e-12, "curl 未饱和于 1.0"


def test_channel_wing_and_escape_spread():
    b = AdultFlyBody(P); b.reset(); rw = b.run({"C_wing": 1.0}, 200.0)
    assert _in("wing_freq_hz", main_peak_freq(rw["wing_ang"], DT)), "翅拍主频出带"
    b.reset(); re = b.run({"C_escape": 1.0}, 200.0)
    assert _in("wing_spread_on_escape", float(np.max(re["wing_spread"]))), "逃跑翅展出带"


# ---------------- 3. 状态分类语义 ----------------
def test_state_semantics_match_channels():
    b = AdultFlyBody(P); b.reset(); s_run = state_fractions(b.run({"C_leg_fwd": 1.0}, 1000.0)["states"])
    b.reset(); s_curl = state_fractions(b.run({"C_curl": 1.0}, 1000.0)["states"])
    b.reset(); s_pause = state_fractions(b.run({}, 1000.0)["states"])
    assert s_run["run"] > 0.5, "C_leg_fwd 驱动未产生 run 态"
    assert s_curl["curl"] > 0.5, "C_curl 驱动未产生 curl 态"
    assert s_pause["pause"] > 0.5, "无驱动未产生 pause 态"
    for s in (s_run, s_curl, s_pause):
        assert abs(sum(s.values()) - 1.0) < 1e-9, "状态比例和 ≠ 1"


def test_escape_priority_over_curl():
    b = AdultFlyBody(P); b.reset()
    r = b.run({"C_curl": 1.0}, 200.0, stimuli={"light": 1.0})
    assert "escape" in r["states"], "逃跑应优先于 curl（优先级 CSV 定稿）"


# ---------------- 4. 巨纤维逃跑 sanity ----------------
def test_giant_fiber_escape_sanity():
    n = 20
    resp = 0
    for k in range(n):
        b = AdultFlyBody(P, seed=k); b.reset()
        if len(b.run({}, 300.0, stimuli={"light": 1.0})["escape_idx"]) > 0:
            resp += 1
    prob = resp / float(n)
    assert _in("escape_resp_prob", prob), "逃跑反应概率 %.2f < 带下沿" % prob


def test_escape_subthreshold_no_response():
    n = 20
    resp = 0
    for k in range(n):
        b = AdultFlyBody(P, seed=k); b.reset()
        if len(b.run({}, 300.0, stimuli={"light": P.escape_thr * 0.4})["escape_idx"]) > 0:
            resp += 1
    prob = resp / float(n)
    assert _in("escape_resp_prob_subthr", prob), "亚阈值刺激误触发逃跑（%.2f）" % prob


# ---------------- 5. 确定性 + 有界 ----------------
def test_determinism_bitwise():
    b1 = AdultFlyBody(P, seed=0); b1.reset()
    b2 = AdultFlyBody(P, seed=0); b2.reset()
    r1 = b1.run({"C_leg_fwd": 1.0}, 500.0)
    r2 = b2.run({"C_leg_fwd": 1.0}, 500.0)
    assert np.array_equal(r1["joints_deg"], r2["joints_deg"]), "关节角时序非逐位一致"
    assert np.array_equal(r1["x"], r2["x"]), "轨迹非逐位一致"


def test_trajectory_finite_and_bounded():
    b = AdultFlyBody(P); b.reset()
    r = b.run({"C_leg_fwd": 1.0, "C_wing": 0.5}, 1000.0)
    for key in ("joints_deg", "x", "y", "v", "omega", "curl", "wing_ang"):
        assert np.all(np.isfinite(r[key])), "%s 含 NaN/Inf" % key
    assert np.max(np.abs(r["joints_deg"])) <= 180.0, "关节角超界"
    assert np.max(np.abs(r["x"])) <= b.arena_mm + 1e-9, "位置越出 arena"
    assert np.max(np.abs(r["y"])) <= b.arena_mm + 1e-9, "位置越出 arena"
    assert np.all((r["curl"] >= 0.0) & (r["curl"] <= 1.0)), "curl 未饱和于 [0,1]"


def test_vr_ball_closed_loop_interfaces():
    """§3.2 VR 球协议接口：视觉流/EPG 输入可算且有限。"""
    b = AdultFlyBody(P); b.reset()
    flow = b.eye_flow(ball_omega=1.0, ball_v=5.0)
    assert np.all(np.isfinite(flow)) and flow.shape == (2,)
    cx = b.cx_heading_input(ball_omega=1.0, ball_v=5.0)
    assert np.all(np.isfinite(cx)) and cx.shape == (3,), "CX 航向输入接口形状错误"
    r = b.run({"C_leg_fwd": 1.0}, 200.0, stimuli={"ball_omega": 1.0, "ball_v": 5.0})
    assert np.all(np.isfinite(r["eye_flow"]))
