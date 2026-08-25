"""M4 嗅觉趋化回路冒烟测试（清单 §4.3 验收：≥6 断言绿 + reports/neuro/m4_smoke.png）。

覆盖（清单 §4.3 + P1/P2/P3 前置断言）：
  1. CSV 拓扑断言：20 角色齐全、连接数与递质极性符合规格（P2 前置）；
  2. 上升阶跃 → ASEL 发放 ≥1 且 ASER 静默（P1 ON 编码冒烟）；
  3. 下降阶跃 → ASER 发放 ≥1 且 ASEL 静默（P1 OFF 编码冒烟）；
  4. 静止浓度（ΔC=0）→ 两者均静默（P1 静止段）；
  5. 核心子链发放次序严格递增（t_ASEL < t_AIY < t_AVB；t_ASER < t_AIB < t_RIA，P2 前置）；
  6. 无梯度 epoch 步进 → 轨迹有限、无 NaN、有界、CI 可计算（P3 前置）；
  7. 闭环确定性：同参数重跑逐位一致（ChemotaxisResult.__eq__ 数值比较，P3 判据）；
  8. 出图 reports/neuro/m4_smoke.png（核心链各级 V + 一条趋化轨迹）。

CSV 依赖：data/m4_chemotaxis_params.csv（B1a 定稿）——若缺失则 wait_for_csv
轮询（30s 间隔、最多 20min），期间其余测试已先跑完。
"""

import math
import os

import numpy as np
import pytest

from neural_exploration.src.chemotaxis_circuit import (
    ChemotaxisCircuit,
    ChemotaxisResult,
    EXPECTED_ROLES,
    wait_for_csv,
)
from neural_exploration.src.chemotaxis_env import ChemotaxisEnv, stationary_protocol
from neural_exploration.src.chemotaxis_loop import ChemotaxisLoop

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data")
CSV_PATH = os.path.join(DATA_DIR, "m4_chemotaxis_params.csv")


@pytest.fixture(scope="module")
def circuit():
    """CSV 依赖就绪后构建 ChemotaxisCircuit（module 级复用，确定性参数）。"""
    wait_for_csv(CSV_PATH, timeout_s=1200.0, interval_s=30.0)
    return ChemotaxisCircuit(csv_path=CSV_PATH)


@pytest.fixture(scope="module")
def step_run(circuit):
    """默认 P1 全阶跃协议单次运行（上升→静止→下降；P1 + 链传播共用）。"""
    return circuit.run()


@pytest.fixture(scope="module")
def stationary_run(circuit):
    """静止浓度协议（ΔC=0）运行；结束后恢复默认协议。"""
    t, c = stationary_protocol(t_total_ms=150.0, dt_ms=circuit.params.dt_ms)
    circuit.set_protocol(c_trace=c, dt_protocol_ms=circuit.params.dt_ms)
    try:
        return circuit.run()
    finally:
        circuit.clear_protocol()


@pytest.fixture(scope="module")
def ctrl_loop(circuit):
    """无梯度对照闭环（C_max 置 0；同一协议）。"""
    base_env = ChemotaxisEnv(**dict(circuit.params.env.__dict__))
    return ChemotaxisLoop(circuit, env=base_env.no_gradient(), seed=0)


@pytest.fixture(scope="module")
def ctrl_traj(ctrl_loop):
    """无梯度闭环短试次（冒烟；16 个 epoch）。"""
    return ctrl_loop.run_trial(t_total_ms=400.0)


@pytest.fixture(scope="module")
def grad_loop(circuit):
    """食物梯度闭环（机制 A 冒烟；同一协议）。"""
    base_env = ChemotaxisEnv(**dict(circuit.params.env.__dict__))
    return ChemotaxisLoop(circuit, env=base_env, seed=7)


# --------------------------------------------------------------------- #
# 1) CSV 拓扑/极性断言（P2 前置）
# --------------------------------------------------------------------- #
def test_csv_topology(circuit):
    """20 角色齐全、连接数与递质类型与规格一致（清单 §2.1 极性表）。"""
    s = circuit.chain_summary()
    assert len(s["roles"]) == 20, f"角色数应为 20：{len(s['roles'])}"
    assert set(s["roles"]) == set(EXPECTED_ROLES)
    assert s["n_chemical"] >= 12, f"化学突触数异常：{s['n_chemical']}"
    assert s["n_muscle_drives"] >= 3, f"肌肉驱动数异常：{s['n_muscle_drives']}"

    st = s["synapse_types"]
    pairs = [(k.split("->")[0], k.split("->")[1], v) for k, v in st.items()]
    # 极性语义（清单 §2.1 连接极性表）：
    # ASE ON → AIY 前进促进（AMPA）；ASE OFF → AIB 转向促进（AMPA）
    assert any(frm == "ASEL" and to.startswith("AIY") and t == "ampa"
               for frm, to, t in pairs), "缺少 ASEL→AIY AMPA"
    assert any(frm == "ASER" and to.startswith("AIB") and t == "ampa"
               for frm, to, t in pairs), "缺少 ASER→AIB AMPA"
    # AIY → RIA 转向抑制（GABA，互斥机制）；AIB → RIA 转向驱动（AMPA）
    assert any(frm.startswith("AIY") and to.startswith("RIA") and t == "gaba"
               for frm, to, t in pairs), "缺少 AIY→RIA GABA（转向抑制）"
    assert any(frm.startswith("AIB") and to.startswith("RIA") and t == "ampa"
               for frm, to, t in pairs), "缺少 AIB→RIA AMPA（转向驱动）"
    # RIA → SMDD 转向执行（AMPA）；AVB → VB/DB 前进命令（AMPA）
    assert any(frm.startswith("RIA") and to.startswith("SMDD") and t == "ampa"
               for frm, to, t in pairs), "缺少 RIA→SMDD AMPA"
    assert any(frm.startswith("AVB") and (to == "VB" or to == "DB") and t == "ampa"
               for frm, to, t in pairs), "缺少 AVB→VB/DB AMPA"
    # 三通道肌肉齐全（fwd/left/right）
    chans = set(s["muscle_channels"].values())
    assert {"fwd", "left", "right"} <= chans, f"肌肉通道应含 fwd/left/right：{chans}"


# --------------------------------------------------------------------- #
# 2) P1 ON 编码：上升阶跃 → ASEL 发放、ASER 静默
# --------------------------------------------------------------------- #
def test_rising_step_asel_on_aser_off(step_run):
    """上升阶跃（ΔC=+0.5）→ 上升段 ASEL ≥1 发放且 ASER 静默（P1 ON 编码）。"""
    info = step_run.meta["protocol"]
    rise_start, rise_end = info["rise_start_ms"], info["rise_end_ms"]
    fall_start = info["fall_start_ms"]
    t_asel = step_run.spikes("ASEL", "node3")
    t_aser = step_run.spikes("ASER", "node3")
    assert len(t_asel) >= 1, "上升阶跃 ASEL 必须发放"
    in_rise = t_asel[(t_asel >= rise_start - 1.0) & (t_asel <= rise_end + 1.0)]
    assert len(in_rise) >= 1, f"上升窗 [{rise_start},{rise_end}] 内 ASEL 必须发放"
    # ASER 在上升段（及整段下降前）静默——OFF 细胞只响应浓度下降
    assert len(t_aser[t_aser < fall_start]) == 0, \
        f"上升段 ASER 必须静默（首发放 {t_aser[0]:.2f} 早于下降 {fall_start}）"


# --------------------------------------------------------------------- #
# 3) P1 OFF 编码：下降阶跃 → ASER 发放、ASEL 静默
# --------------------------------------------------------------------- #
def test_falling_step_aser_on_asel_off(step_run):
    """下降阶跃（ΔC=−0.5）→ 下降段 ASER ≥1 发放且 ASEL 静默（P1 OFF 编码）。"""
    info = step_run.meta["protocol"]
    fall_start, fall_end = info["fall_start_ms"], info["fall_end_ms"]
    t_asel = step_run.spikes("ASEL", "node3")
    t_aser = step_run.spikes("ASER", "node3")
    assert len(t_aser) >= 1, "下降阶跃 ASER 必须发放"
    in_fall = t_aser[(t_aser >= fall_start - 1.0) & (t_aser <= fall_end + 1.0)]
    assert len(in_fall) >= 1, f"下降窗 [{fall_start},{fall_end}] 内 ASER 必须发放"
    # ASEL 在下降段静默（滑窗差分 τ_win 后 ASEL 注入已结束；留 10ms 边界
    # 容忍在途发放——M4 实测 τ_win=100ms 时 ASEL 注入持续到下降起点）
    late = t_asel[(t_asel >= fall_start + 10.0) & (t_asel <= fall_end)]
    assert len(late) == 0, f"下降段 ASEL 必须静默（{len(late)} 个晚发放）"
    # 全段归属：ASEL 发放只发生在上升/静止段（s>0 的滑窗响应）
    assert np.all(t_asel < fall_start + 10.0), "ASEL 发放应限于 s>0 窗"


# --------------------------------------------------------------------- #
# 4) P1 静止段：ΔC=0 → 两者均静默
# --------------------------------------------------------------------- #
def test_stationary_silent(stationary_run):
    """静止浓度（ΔC=0）→ ASEL/ASER 均静默（P1 静止段判据）。"""
    assert len(stationary_run.spikes("ASEL", "node3")) == 0, "静止段 ASEL 必须静默"
    assert len(stationary_run.spikes("ASER", "node3")) == 0, "静止段 ASER 必须静默"
    # 无输入时感觉→中间链静默（无自发活动；AVB/VB 有张力基线，不在此列）
    for role in ("AIYL", "AIBL", "RIAL", "SMDDL"):
        assert len(stationary_run.spikes(role, "node3")) == 0, \
            f"静止段 {role} 不得自发发放"


# --------------------------------------------------------------------- #
# 5) P2 链传播：核心子链发放次序严格递增
# --------------------------------------------------------------------- #
def _first_after(spike_times: np.ndarray, t_ref: float) -> float:
    """该角色在 t_ref 之后的首个发放时刻（张力角色用：取上游发放后的下一发放）。"""
    st = np.asarray(spike_times, dtype=float)
    later = st[st > t_ref + 1e-6]
    if len(later) == 0:
        return float("inf")
    return float(later[0])


def test_chain_order(circuit, step_run):
    """核心子链发放次序严格递增（t_ASEL < t_AIY < t_AVB；t_ASER < t_AIB < t_RIA）。

    张力角色（AVB 维持前进基线）本身持续发放——按"上游发放后的首个下游发放"
    判定链传播次序（P2 语义：传导方向正确且逐级触发）。
    """
    chain_a = circuit.chain_from("ASEL", ("fwd",))        # 前进链 → C_fwd
    chain_b = circuit.chain_from("ASER", ("left", "right"))  # 转向链 → C_left/C_right
    assert len(chain_a) >= 3, f"前进链过短：{chain_a}"
    assert len(chain_b) >= 3, f"转向链过短：{chain_b}"

    t_prev = step_run.spikes(chain_a[0], "node3")
    assert len(t_prev) >= 1, f"前进链起点 {chain_a[0]} 必须发放"
    t_prev = float(t_prev[0])
    for role in chain_a[1:]:
        st = step_run.spikes(role, "node3")
        t_next = _first_after(st, t_prev)
        assert t_next < float("inf"), f"前进链 {role} 必须随上游发放（链 {chain_a}）"
        assert t_next > t_prev + 1e-6, \
            f"前进链发放次序错误：{role} 首发放 {t_next:.3f} 应 > {t_prev:.3f}"
        t_prev = t_next

    t_prev = step_run.spikes(chain_b[0], "node3")
    assert len(t_prev) >= 1, f"转向链起点 {chain_b[0]} 必须发放"
    t_prev = float(t_prev[0])
    for role in chain_b[1:]:
        st = step_run.spikes(role, "node3")
        t_next = _first_after(st, t_prev)
        assert t_next < float("inf"), f"转向链 {role} 必须随上游发放（链 {chain_b}）"
        assert t_next > t_prev + 1e-6, \
            f"转向链发放次序错误：{role} 首发放 {t_next:.3f} 应 > {t_prev:.3f}"
        t_prev = t_next


# --------------------------------------------------------------------- #
# 6) P3 前置：无梯度 epoch 步进 → 轨迹有限/无 NaN/有界/CI 可计算
# --------------------------------------------------------------------- #
def test_closed_loop_no_gradient(ctrl_traj):
    """无梯度闭环：轨迹有限、无 NaN、全程在皿内、CI ∈ [−1,1] 可计算。"""
    assert ctrl_traj.has_trajectory
    assert np.all(np.isfinite(ctrl_traj.x)), "轨迹 x 含 NaN/Inf"
    assert np.all(np.isfinite(ctrl_traj.y)), "轨迹 y 含 NaN/Inf"
    assert np.all(np.isfinite(ctrl_traj.theta)), "朝向含 NaN/Inf"
    L = ctrl_traj.meta["env"]["arena_L"]
    assert ctrl_traj.x.min() >= -1e-6 and ctrl_traj.x.max() <= L + 1e-6, \
        f"轨迹越界 x∈[{ctrl_traj.x.min():.3f},{ctrl_traj.x.max():.3f}]"
    assert ctrl_traj.y.min() >= -1e-6 and ctrl_traj.y.max() <= L + 1e-6
    assert ctrl_traj.ci is not None and -1.0 <= ctrl_traj.ci <= 1.0, \
        f"CI 越界：{ctrl_traj.ci}"
    # 肌肉收缩量程（P3：量程在规格内）
    assert np.all(np.isfinite(ctrl_traj.c_fwd)) and np.all(np.isfinite(ctrl_traj.c_left))
    assert np.all(np.isfinite(ctrl_traj.c_right))


# --------------------------------------------------------------------- #
# 7) P3 判据：闭环确定性——同参数重跑逐位一致
# --------------------------------------------------------------------- #
def test_closed_loop_determinism(ctrl_loop, ctrl_traj):
    """同参数重跑闭环 → 轨迹/膜电位/发放/肌肉逐位一致（__eq__ 数值比较）。"""
    r2 = ctrl_loop.run_trial(t_total_ms=400.0)
    assert isinstance(ctrl_traj, ChemotaxisResult) and isinstance(r2, ChemotaxisResult)
    assert ctrl_traj == r2, "闭环同参数重跑必须逐位一致"
    assert ctrl_traj.ci == r2.ci


# --------------------------------------------------------------------- #
# 7b) 机制 A（清单 §2.4 落地修订）：转向事件 + 确定性伪随机方向
# --------------------------------------------------------------------- #
def test_mechanism_a_turn_events(grad_loop):
    """机制 A：梯度下 s<−θ_pir 且 ASER→AIB→RIA→SMDD 激活 → 转向事件；
    转向方向=试次种子确定性伪随机 → 同种子重跑逐位一致。"""
    assert grad_loop.circuit.params.mech_a.enabled, "机制 A 应在 CSV 中启用"
    # 起点靠近食物、朝向远离食物 → s<0 立即成立（τ_win 窗填满后 |s|≈1e-4）
    env = grad_loop.env.spec
    sx, sy = env.food_x - 0.7, env.food_y - 0.7          # 距食物 r≈0.99
    away = math.atan2(env.food_y - sy, env.food_x - sx) + math.pi  # 背向食物
    # record=[]：关闭 V 轨迹监视（冒烟提速，B1c L16 实测 ~15×）；发放/CI/转向仍可读
    r1 = grad_loop.run_trial(start_x=sx, start_y=sy, theta0=away,
                             t_total_ms=600.0, seed=7, record=[])
    r2 = grad_loop.run_trial(start_x=sx, start_y=sy, theta0=away,
                             t_total_ms=600.0, seed=7, record=[])
    # 电路耦合：ASER 链确实激活（SMDD 发放）——机制 A 触发的前置条件
    n_smdd = (len(r1.spikes("SMDDL", "node3"))
              + len(r1.spikes("SMDDR", "node3")))
    assert n_smdd >= 1, f"远离食物应激活 ASER→...→SMDD（SMDD 发放 {n_smdd}）"
    assert r1.meta.get("n_turn_events", 0) >= 1, \
        f"应触发转向事件（n_turn_events={r1.meta.get('n_turn_events')}）"
    assert r1.meta["turn_dir_seed"] == 7
    assert r1 == r2, "机制 A 同种子重跑必须逐位一致（转向方向确定性伪随机）"
    assert r1.ci == r2.ci


# --------------------------------------------------------------------- #
# 8) 出图 reports/neuro/m4_smoke.png
# --------------------------------------------------------------------- #
def test_plot_m4_smoke(circuit, step_run, ctrl_traj):
    """出图：核心链各级 V（两条子链）+ 一条趋化轨迹。"""
    from neural_exploration.src.chemotaxis_circuit import plot_chemotaxis

    chain_a = circuit.chain_from("ASEL", ("fwd",))
    chain_b = circuit.chain_from("ASER", ("left", "right"))
    reports_dir = os.path.join(os.path.dirname(CSV_PATH), "..", "reports", "neuro")
    out_png = os.path.join(os.path.abspath(reports_dir), "m4_smoke.png")
    png = plot_chemotaxis(step_run, ctrl_traj, out_png,
                          chain_a_roles=chain_a, chain_b_roles=chain_b)
    assert os.path.exists(png) and os.path.getsize(png) > 10_000, \
        f"图未生成: {png}"
