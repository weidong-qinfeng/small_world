"""M5 全虫冒烟测试（清单 §5.3：≥8 断言绿 + reports/neuro/m5_smoke.png）。

覆盖（清单 §5.3 逐项 + P1 前置）：
  1. P1 前置：load_connectome 302 神经元/四类计数/化学/缝隙计数 + 自连接/孤立白名单；
  2. M4 子图交叉核对断言（读 data/m5_crosscheck_m3m4.csv 结果 + 与解析后
     连接组 spec 的一致性检查——OK 边存在且极性一致、MISSING 边不存在）；
  3. 点档单神经元发放冒烟（脉冲注入 → 发放 ≥1；同参数重跑逐位一致）；
  4. 20 规模趋化短协议（WormLoop 闭环）：CI 可计算 ∈[−1,1]、轨迹有界、无 NaN；
  5. 机械刺激短协议：M3 反射子图（降阶，G0 已验证方向 back）后退方向正确
     （C_back > C_fwd / D_peak>0.3）+ WormLoop 触刺激窗（τ_trans 语义）检查；
  6. 咽部子图：MC 驱动下节律发放存在（spike count > 0，10s 窗）、无 NaN；
  7. 无刺激静息：无 NaN/无发散/发放率有限；
  8. 闭环确定性：同参数重跑轨迹逐位一致（ChemotaxisResult.__eq__）；
  9. 出图 reports/neuro/m5_smoke.png（全虫轨迹 + 咽部节律 + 静息发放分布）。

数据依赖：data/m5_connectome.csv、data/m5_pharynx_subgraph.csv、
data/m5_crosscheck_m3m4.csv、data/m5_connectome_counts.json（B1a/B1b 定稿源）。
"""

import json
import math
import os

import numpy as np
import pytest

from neural_exploration.src.point_neuron import PointNeuron
from neural_exploration.src.virtual_body import VirtualBody, classify_state
from neural_exploration.src.worm_circuit import (
    ReflexCircuit,
    GroupedWormCircuit,
    load_connectome,
)
from neural_exploration.src.worm_loop import WormLoop

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data")
CONNECTOME_CSV = os.path.join(DATA_DIR, "m5_connectome.csv")
PHARYNX_CSV = os.path.join(DATA_DIR, "m5_pharynx_subgraph.csv")
CROSSCHECK_CSV = os.path.join(DATA_DIR, "m5_crosscheck_m3m4.csv")
COUNTS_JSON = os.path.join(DATA_DIR, "m5_connectome_counts.json")
REPORTS_NEURO = os.path.join(os.path.dirname(DATA_DIR), "reports", "neuro")
SMOKE_PNG = os.path.join(REPORTS_NEURO, "m5_smoke.png")

#: 咽部 MC 驱动密度（µA/cm²；"有食物驱动"语义占位——P3 协议节点定稿正式协议，
#: 冒烟只断言子图可维持发放，不预注册节律窗）
PHARYNX_DRIVE_UA_CM2 = 60.0
#: 冒烟趋化短协议 T（ms；20 档 grouped 稳态 ~3s/T1s，短协议控制预算）
SMOKE_TRIAL_MS = 2000.0
#: 静息冒烟 T（ms）
SMOKE_RESTING_MS = 1000.0
#: 咽部冒烟 T（ms；10s 窗，清单 §5.3）
SMOKE_PHARYNX_MS = 10000.0


def _clean_line(ln: str) -> str:
    s = ln.strip()
    if s.startswith('"'):
        s = s.strip('"')
    return s


# --------------------------------------------------------------------- #
# 共享夹具（module 级：编译一次，多次断言复用——冷编译预算纪律 M4 L16/L25）
# --------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def spec():
    return load_connectome(CONNECTOME_CSV)


@pytest.fixture(scope="module")
def counts():
    with open(COUNTS_JSON, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def wcirc():
    """20 规模全虫电路（grouped，点档 G0 定稿；确定性 seed=0）。"""
    return GroupedWormCircuit(scale=20, seed=0)


@pytest.fixture(scope="module")
def wloop(wcirc):
    """全虫闭环耦合器（VirtualBody + m5_worm_params.csv 参数）。"""
    return WormLoop(wcirc)


@pytest.fixture(scope="module")
def chem_trial(wloop):
    """20 规模趋化短协议单试次（冒烟主体）。"""
    return wloop.run_trial(t_total_ms=SMOKE_TRIAL_MS, seed=0)


@pytest.fixture(scope="module")
def reflex():
    """M3 反射子图降阶（§3.4 降阶正确性；G0 已验证方向 back）。"""
    return ReflexCircuit(fidelity="point").run(t_total_ms=150.0)


@pytest.fixture(scope="module")
def pharynx_run():
    """咽部子图 10s：MC 驱动（有食物语义占位）→ 节律发放序列。"""
    circ = GroupedWormCircuit(csv_path=PHARYNX_CSV, scale=20, seed=0)
    sess = circ.make_session(t_total_ms=SMOKE_PHARYNX_MS)
    sess.reset(seed=0)
    i_nA = PHARYNX_DRIVE_UA_CM2 * 1e-6 * 1.257e-5 * 1e9  # µA/cm² → nA
    for role in ("MCL", "MCR"):
        idx = circ.role_index[role]
        sess.stim.values[:, idx] = i_nA * 1e-9
    sess.run_resting_window(SMOKE_PHARYNX_MS)
    return sess


# --------------------------------------------------------------------- #
# 1) P1 前置：连接组 302 神经元 + 计数（清单 §5.3 + P1）
# --------------------------------------------------------------------- #
def test_connectome_302(spec, counts):
    """302 神经元 + 四类计数 + 化学/缝隙计数 vs B1a counts.json + 白名单。"""
    assert spec.n_neurons == 302, f"神经元数应为 302：{spec.n_neurons}"
    got = {c: sum(1 for v in spec.neurons.values() if v == c)
           for c in ("sensory", "inter", "motor", "pharyngeal")}
    ref = counts["p1"]["class_counts"]
    assert got == ref, f"四类计数应一致：{got} vs {ref}"
    assert sum(got.values()) == 302
    # 化学（可用 ampa+gaba）与缝隙计数（vs B1a 权威解析）
    n_skipped = getattr(spec, "n_skipped_mod_none", 0)
    assert spec.n_chem == counts["p1"]["chem_directed_pairs"] - n_skipped, \
        f"化学可用计数异常：{spec.n_chem} + {n_skipped} skipped"
    assert spec.n_gap == counts["p1"]["gap_unique_pairs"], \
        f"缝隙计数应一致：{spec.n_gap}"
    # 自连接白名单保留（真实连接组存在，不静默删除——L14）
    self_chem = [r for r in spec.chem if r.pre == r.post]
    assert len(self_chem) > 0, "化学自连接（白名单）应保留"
    # 孤立白名单（CANL/CANR——Cook 无连接，规范 roster 成员，L14）
    for n in ("CANL", "CANR"):
        assert n in spec.neurons, f"孤立白名单神经元 {n} 应在 roster 中"


# --------------------------------------------------------------------- #
# 2) M4 子图交叉核对断言（读 m5_crosscheck_m3m4.csv + 与 spec 一致性）
# --------------------------------------------------------------------- #
def test_crosscheck_m3m4(spec, counts):
    """交叉核对：53 行/43 OK/10 DIFF 与 counts.json 一致；关键边存在性与极性
    与解析后的连接组 spec 一致（OK=存在且极性一致、MISSING=不存在）。"""
    rows = []
    import csv as _csv
    with open(CROSSCHECK_CSV, newline="", encoding="utf-8") as f:
        for r in _csv.DictReader(_clean_line(ln) for ln in f
                                 if _clean_line(ln) and not _clean_line(ln).startswith("#")):
            rows.append(r)
    assert len(rows) == counts["crosscheck"]["total"] == 53
    ok = [r for r in rows if r["verdict"] == "OK"]
    diff = [r for r in rows if r["verdict"] == "DIFF"]
    assert len(ok) == counts["crosscheck"]["ok"] == 43
    assert len(diff) == counts["crosscheck"]["diff"] == 10

    chem = {(r.pre, r.post): r.syn_type for r in spec.chem}
    gaps = {(r.a, r.b) for r in spec.gaps}

    # 关键边存在性 + 极性（vs 真实连接组解析结果）
    assert chem.get(("ASEL", "AIYL")) == "ampa", "ASEL→AIYL 应存在且 ampa（OK）"
    assert chem.get(("ASER", "AIBL")) == "ampa", "ASER→AIBL 应存在且 ampa（OK）"
    assert chem.get(("RIAL", "SMDDL")) == "ampa", "RIAL→SMDDL 应存在且 ampa（OK）"
    assert chem.get(("AVBL", "VB2")) == "ampa", "AVBL→VB2 应存在且 ampa（OK）"
    assert chem.get(("AVM", "DA1")) == "ampa", "AVM→DA1 应存在且 ampa（OK，M3）"
    # TYPE_DIFF（真实递质为准）：AIYL→RIAL 存在但为 ach→ampa（M4 建模 gaba）；
    # RIAL→SMDVL 在真实连接组为缝隙连接
    assert chem.get(("AIYL", "RIAL")) == "ampa", "AIYL→RIAL 应存在（ach→ampa，TYPE_DIFF）"
    assert ("RIAL", "SMDVL") in gaps, "RIAL→SMDVL 应为缝隙连接（TYPE_DIFF）"
    # MISSING（真实连接组无该直接化学边——M3/M4 功能链简化，L8）
    assert not any(r.pre == "PLML" and r.post == "AVM" for r in spec.chem), \
        "PLM→AVM 在真实连接组应为 MISSING（触觉经缝隙耦合）"
    assert not any(r.pre == "AIYL" and r.post == "AVBL" for r in spec.chem), \
        "AIYL→AVBL 在真实连接组应为 MISSING"
    # 肌肉驱动（M3/M4 先验通道：DA→back、VB→fwd 存在）
    mus = {(m.motor, m.channel) for m in spec.muscles}
    assert any(m.startswith("DA") and (m, "back") in mus for m in
               ["DA1", "DA2", "DA3"]), "DA*→back 肌肉驱动应存在（M3）"
    assert any((m, "fwd") in mus for m in
               ["VB1", "VB2", "DB1", "DB2"]), "VB*/DB*→fwd 肌肉驱动应存在（M4）"


# --------------------------------------------------------------------- #
# 3) 点档单神经元发放冒烟（脉冲 → 发放 ≥1；确定性重跑逐位一致）
# --------------------------------------------------------------------- #
def test_point_neuron_spike():
    from brian2 import (Network, SpikeMonitor, TimedArray, amp, ms, nA,
                        start_scope)

    def _run(name):
        start_scope()
        pn = PointNeuron(name=name, dt_ms=0.1, method="exponential_euler",
                         extra_eqs="", extra_im_terms="", stim_var="stim").build()
        n_steps = 6000
        arr = np.zeros((n_steps, 1)) * amp
        i_nA = pn.density_to_nA(60.0)          # 60 µA/cm²（M3 触刺激密度同量级）
        i0 = int(round(50.0 / 0.1))
        i1 = int(round(55.0 / 0.1))
        arr[i0:i1, 0] = i_nA * nA
        stim = TimedArray(arr, dt=0.1 * ms, name=f"{name}_stim")
        net = Network(pn.neuron)
        sp = SpikeMonitor(pn.neuron, "v", name=f"{name}_sp")
        net.add(sp)
        net.run(150.0 * ms, namespace={"stim": stim})
        return np.asarray(sp.t / ms)

    t1 = _run("smoke_pn")
    assert len(t1) >= 1, "脉冲注入必须触发发放"
    t2 = _run("smoke_pn2")   # 同参数重跑（不同对象名，数值等价）
    assert np.array_equal(t1, t2), "点神经元同参数重跑必须逐位一致"


# --------------------------------------------------------------------- #
# 4) 20 规模趋化短协议：CI 可计算/有界/无 NaN
# --------------------------------------------------------------------- #
def test_chemotaxis_short_protocol(chem_trial, wloop):
    """闭环短协议：轨迹有限、无 NaN、全程在皿内、CI ∈ [−1,1]、肌肉有限。"""
    r = chem_trial
    assert r.has_trajectory
    assert np.all(np.isfinite(r.x)), "轨迹 x 含 NaN/Inf"
    assert np.all(np.isfinite(r.y)), "轨迹 y 含 NaN/Inf"
    assert np.all(np.isfinite(r.theta)), "朝向含 NaN/Inf"
    L = wloop.env.spec.arena_L
    assert r.x.min() >= -1e-6 and r.x.max() <= L + 1e-6, \
        f"轨迹越界 x∈[{r.x.min():.3f},{r.x.max():.3f}]"
    assert r.y.min() >= -1e-6 and r.y.max() <= L + 1e-6
    assert r.ci is not None and -1.0 <= r.ci <= 1.0, f"CI 越界：{r.ci}"
    assert np.all(np.isfinite(r.c_fwd)) and np.all(np.isfinite(r.c_left)) \
        and np.all(np.isfinite(r.c_right)), "肌肉收缩含 NaN"
    # 状态分类可用（阈值 CSV 定稿语义；比例和为 1）
    frac = r.meta["state_frac"]
    assert abs(sum(frac.values()) - 1.0) < 1e-9
    assert set(frac) == {"fwd", "rev", "turn", "pause"}


# --------------------------------------------------------------------- #
# 5) 机械刺激短协议：后退方向正确（M3 反射子图降阶 + 触刺激窗 τ_trans 语义）
# --------------------------------------------------------------------- #
def test_escape_direction_backward(reflex, wcirc):
    """M3 反射子图（点档降阶，§3.4 一致性）：方向 back（C_back > C_fwd）。"""
    assert reflex.direction == "back", f"方向应为 back：{reflex.direction}"
    assert reflex.d_peak > 0.3, f"D_peak 应 > 0.3：{reflex.d_peak}"
    assert np.any(reflex.c_back > reflex.c_fwd), "应存在 C_back > C_fwd 时刻"
    assert np.all(np.isfinite(reflex.c_back)) and np.all(np.isfinite(reflex.c_fwd))

    # WormLoop 机械刺激协议窗（P5）：I0·1[t0+τ_trans, t0+τ_trans+dur]
    # τ_trans = CSV escape_touch_delay_ms（B1b 定稿 CSV 暂无 → 默认 0，L23）
    wl = WormLoop(wcirc)
    i0, i1, n_steps = wl.touch_window()
    assert i0 == int(round(wl.touch["start_ms"] / wcirc.dt_ms)), \
        "触刺激窗起点 = t0+τ_trans"
    assert i1 - i0 == int(round(wl.touch["dur_ms"] / wcirc.dt_ms)), \
        "触刺激窗长 = dur"
    assert wl.touch["tau_trans_ms"] == 0.0, "CSV 未定稿 τ_trans → 默认 0"
    # τ_trans 语义：显式设置后窗右移
    wl.touch["tau_trans_ms"] = 10.0
    i0t, i1t, _ = wl.touch_window()
    assert i0t - i0 == int(round(10.0 / wcirc.dt_ms)), "τ_trans 应右移刺激窗起点"


# --------------------------------------------------------------------- #
# 6) 咽部子图：节律发放存在（10s 窗）
# --------------------------------------------------------------------- #
def test_pharynx_rhythm(pharynx_run):
    """MC 驱动 10s：咽部角色发放存在（spike count > 0）、发放时刻有限。"""
    times = pharynx_run.role_spike_times()
    total = sum(len(t) for t in times.values())
    assert total > 0, "咽部子图 MC 驱动下必须出现发放"
    n_spiking = sum(1 for t in times.values() if len(t) > 0)
    assert n_spiking >= 1, "至少一个咽部角色发放"
    for role, t in times.items():
        assert np.all(np.isfinite(t)), f"{role} 发放时刻含 NaN/Inf"


# --------------------------------------------------------------------- #
# 7) 无刺激静息：无 NaN/无发散/发放率有限
# --------------------------------------------------------------------- #
def test_resting_no_nan(wloop):
    """无刺激静息：发放率有限、无 NaN、无发散（占位权重过度兴奋是已知状态，
    P2 判带由 §6 权重校准后执行——冒烟只断言数值稳定）。"""
    rest = wloop.run_resting(t_total_ms=SMOKE_RESTING_MS)
    assert not rest["has_nan"], "静息发放率含 NaN/发散"
    rates = np.array(list(rest["rates_hz"].values()), dtype=float)
    assert np.all(np.isfinite(rates)), "发放率必须有限"
    assert np.isfinite(rest["median_hz"]) and np.isfinite(rest["max_hz"])
    assert rest["max_hz"] < 1000.0, f"最大发放率发散：{rest['max_hz']:.1f} Hz"
    assert 0.0 <= rest["silent_frac"] <= 1.0


# --------------------------------------------------------------------- #
# 8) 闭环确定性：同参数重跑逐位一致
# --------------------------------------------------------------------- #
def test_closed_loop_determinism(wloop):
    """同参数重跑闭环 → 轨迹/肌肉/发放/CI 逐位一致（__eq__ 数值比较）。"""
    r1 = wloop.run_trial(t_total_ms=SMOKE_TRIAL_MS, seed=0)
    r2 = wloop.run_trial(t_total_ms=SMOKE_TRIAL_MS, seed=0)
    assert r1 == r2, "闭环同参数重跑必须逐位一致"
    assert np.array_equal(r1.x, r2.x) and np.array_equal(r1.y, r2.y)
    assert r1.ci == r2.ci


# --------------------------------------------------------------------- #
# 9) 出图 reports/neuro/m5_smoke.png（全虫轨迹 + 咽部节律 + 静息发放分布）
# --------------------------------------------------------------------- #
def test_plot_m5_smoke(chem_trial, pharynx_run, wloop):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    for _f in ("PingFang SC", "PingFang HK", "Heiti TC", "STHeiti",
               "Arial Unicode MS"):
        try:
            fm.findfont(_f, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [_f, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            break
        except Exception:
            continue

    os.makedirs(REPORTS_NEURO, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    # 1) 全虫轨迹（20 规模趋化短协议）
    ax = axes[0]
    r = chem_trial
    ax.plot(r.x, r.y, lw=1.3, color="tab:blue")
    ax.scatter([wloop.env.spec.food_x], [wloop.env.spec.food_y], c="red",
               marker="*", s=160, zorder=5, label="食物")
    ax.scatter([r.x[0]], [r.y[0]], c="k", marker="o", s=40, zorder=5,
               label="起点")
    ax.set_xlim(0, wloop.env.spec.arena_L)
    ax.set_ylim(0, wloop.env.spec.arena_L)
    ax.set_aspect("equal")
    ax.set_xlabel("x（皿单位）")
    ax.set_ylabel("y（皿单位）")
    ax.set_title(f"全虫轨迹（20 规模趋化短协议 {SMOKE_TRIAL_MS:.0f}ms，"
                 f"CI={r.ci:.3f}）")
    ax.legend(fontsize=8)

    # 2) 咽部节律栅栏图（MC 驱动 10s）
    ax = axes[1]
    times = pharynx_run.role_spike_times()
    roles = list(times.keys())
    for k, role in enumerate(roles):
        t = times[role]
        if len(t):
            ax.vlines(t, k - 0.4, k + 0.4, lw=0.5, color="tab:blue")
        ax.text(-0.5, k, role, fontsize=6, va="center", ha="right")
    ax.set_xlim(0, SMOKE_PHARYNX_MS)
    ax.set_ylim(-0.6, len(roles) - 0.4)
    ax.set_xlabel("t（ms）")
    ax.set_ylabel("咽部神经元")
    ax.set_title(f"咽部节律（MC 驱动 {PHARYNX_DRIVE_UA_CM2:.0f}µA/cm²，"
                 f"10s，发放角色 {sum(1 for t in times.values() if len(t))}/{len(roles)}）")

    # 3) 静息发放率分布（无刺激 1s）
    ax = axes[2]
    rest = wloop.run_resting(t_total_ms=SMOKE_RESTING_MS)
    rates = np.array(list(rest["rates_hz"].values()), dtype=float)
    ax.hist(rates, bins=min(30, max(5, len(np.unique(rates)))), color="tab:green",
            alpha=0.8)
    ax.axvline(rest["median_hz"], color="k", ls="--", lw=1.2,
               label=f"中位数 {rest['median_hz']:.1f}Hz")
    ax.set_xlabel("发放率（Hz）")
    ax.set_ylabel("神经元数")
    ax.set_title(f"静息发放率分布（静默比例 {rest['silent_frac']:.0%}，"
                 f"n={len(rates)}）")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(SMOKE_PNG, dpi=110)
    plt.close(fig)
    assert os.path.exists(SMOKE_PNG) and os.path.getsize(SMOKE_PNG) > 10_000, \
        f"图未生成：{SMOKE_PNG}"


# --------------------------------------------------------------------- #
# 10) virtual_body 单元冒烟（后退方程 + classify_state，引擎无关）
# --------------------------------------------------------------------- #
def test_virtual_body_backward_and_classify():
    """身体方程：v = v_fwd0·clip(C_fwd) − v_rev0·clip(C_back)（后退，P6 前提）；
    classify_state 阈值语义（fwd/rev/turn/pause 判据，CSV 定稿默认）。"""
    vb = VirtualBody(v_fwd0=1.0, v_rev0=1.0, omega_max=1.0, dt_b=25.0)
    assert vb.speed(0.0, 1.0) == -1.0, "C_back=1 应产生负速度（后退）"
    assert vb.speed(1.0, 0.0) == 1.0, "C_fwd=1 应前进"
    assert vb.speed(0.5, 0.5) == 0.0, "前后抵消应静止"
    assert classify_state(0.5, 0.0, 0.5, 0.0) == "fwd"
    assert classify_state(-0.5, 0.0, 0.0, 0.5) == "rev"
    assert classify_state(0.0, 0.5, 0.0, 0.0) == "turn"   # 0.5 > 0.2·1.0
    assert classify_state(0.0, 0.0, 0.0, 0.0) == "pause"
    # 阈值定稿语义：v_thr = 0.05·v_fwd0
    assert classify_state(0.03, 0.0, 0.0, 0.0) == "pause"
    assert classify_state(0.1, 0.0, 0.0, 0.0) == "fwd"
