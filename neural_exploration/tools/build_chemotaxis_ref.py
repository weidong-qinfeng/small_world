"""M4 参考解：NEURON 核心子链 + 行为参考模型（pirouette，纯 numpy，引擎无关）。

清单《生物仿真M4实施清单》§3（步骤 2）两级参考：

1) 神经级（NEURON 9.0.1 cvode 高精度核心子链，不复制全 20 神经元）：
   - 链 A（前进）：ASEL → AIYL → AVBL（3 神经元，AMPA 链）——输入=上升阶跃（ON 编码）
   - 链 B（转向）：ASER → AIBL → RIAL → SMDDL（4 神经元，AMPA 链）——输入=下降阶跃（OFF 编码）
   模式照 tools/build_reflex_ref.py：build_neuron(spec, clear, name_prefix) 构建 M1 形态学
   多隔室神经元；ExpSyn（AMPA E=0mV/τ=3ms，weight µS = g_nS×1e-3）；NetCon(node3→soma,
   threshold=−20, delay=CSV 0.5ms)；IClamp 阶跃电流注入 ASE soma（密度→nA 按 SOMA_AREA_CM2 换算，
   I=60µA/cm²×5ms@t=50ms，M3 量级可靠单发放）；点过程（IClamp/ExpSyn/NetCon）列表持有防 GC
   （M2 L8）；cvode atol/rtol=1e-8、celsius=6.3、v_init=V0（硬约束）。
   输出：各级 node3 发放时刻 + ASE→AIY/AIB PSP（AIYL/AIBL soma）+ 链传导时间。

2) 行为级（行为参考模型，纯 numpy，引擎无关）——pirouette 机制算法化（清单 §2.4）：
   每行为 tick（Δt_b=25ms）：采样 C(x,y) → 时间差分 s=(ΔC)/τ_win → s < −θ_pir 触发随机转向
   （ω=±ω_pir 持续 T_pir，方向随机）→ 积分位置（反射边界）→ 逐试次象限式 CI；
   参数（θ_pir/ω_pir/T_pir/v_fwd）以 CI ∈ [0.3,0.7] 为目标粗校准（σ=σ_csv、τ_win=τ_win_csv 定稿）；
   N=20 试次 + 无梯度对照 N=20（同一 CI 统计代码，与 Brian2 虫可比性保证）。

输出 `data/m4_ref.npz`：
   - spike_times_{asel,aiyl,avbl} / spike_times_{aser,aibl,rial,smddl}  子链各级 node3 发放时刻
   - v_aiyl_soma_mv / v_aibl_soma_mv   ASE→AIY/AIB PSP（PSP 参考，P2 对照）
   - chain_time_ms_a / chain_time_ms_b / chain_time_ms_b_to_ria   链传导时间
   - v_asel_node3_mv / v_aser_node3_mv   刺激神经元 node3 V（波形对照）
   - behavior_ci（20 试次）/ behavior_ci_mean/sem / t_stat/p_value/cohen_d
   - behavior_ci_ctrl（20 试次）/ behavior_ci_ctrl_mean / behavior_ci_ctrl_p
   - traj1_{x,y} / traj2_{x,y}   两条示例趋化轨迹
   - meta（引擎/参数/校准记录）

参数唯一定稿源：`data/m4_chemotaxis_params.csv`；处置与踩坑记录：`docs/m4_env_notes.md` §L2/L5+。

用法：
  .venv-neuro/bin/python -m neural_exploration.tools.build_chemotaxis_ref
"""

from __future__ import annotations

import csv
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.src.synapse_model import load_synapse_params  # noqa: E402
from neural_exploration.tools.load_morphology import (  # noqa: E402
    V0, load_morphology,
)
from neural_exploration.tools.build_neuron_ref import (  # noqa: E402
    SOMA_AREA_CM2, build_neuron,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
REF_NPZ = os.path.join(DATA_DIR, "m4_ref.npz")
PARAMS_CSV = os.path.join(DATA_DIR, "m4_chemotaxis_params.csv")

DT_OUT = 0.01          # ms，输出重采样步长（与 CSV dt_ms 一致）
SPIKE_THRESH_MV = -15.0   # node3 发放检测阈值（上冲）+ 峰定位
SPIKE_REF_MS = 1.5        # 发放检测去重窗口

# 子链刺激（NEURON 侧）：ASE soma 阶跃电流，编码 ON（上升阶跃）/ OFF（下降阶跃）
STIM_START_MS = 50.0      # 阶跃起点（≥40ms 基线，HH 静息瞬态漂移，M1/M2 结论）
STIM_DUR_MS = 5.0         # 阶跃时长（M3 量级 → 单发放，链传播可测）
I_ASE_UA_CM2 = 60.0       # 阶跃电流密度（soma 注入，M3 L5：20–80µA/cm²×5ms 均可靠单发放）
SUBCHAIN_T_MS = 150.0     # 子链仿真窗口

# 行为参考模型（numpy pirouette）粗校准网格
# θ_pir 阈值 + 转向角 + v；锚点 v=0.20（CSV 有效爬行速度 v_fwd0·C_fwd_baseline）
# 步骤 4（B1c）落地修订：机制 A 参数（θ_pir/ω_pir/T_pir）以 CSV mechanism_a 行为锚，
# 扫描网格沿锚点扩展（见 calibrate_behavior）。
CAL_THETA_PIR = (4.0e-6, 5.0e-6, 6.0e-6, 8.0e-6, 1.0e-5, 1.5e-5, 2.0e-5)  # [ΔC/ms]
CAL_ALPHA_DEG = (45.0, 60.0, 75.0, 90.0, 120.0)   # 单次转向角 [°]（ω_pir·T_pir）
CAL_V_FWD = (0.15, 0.20)                           # [units/s]；0.20=CSV 有效爬行锚点
V_ANCHOR = 0.20                                    # 首选速度（协议一致性）
OMEGA_PIR = 1.0                                     # [rad/s] 转向角速度
N_TRIALS = 20                                       # 试次/组
RNG_SEED = 0

# B1c 实测（m4_env_notes §L16）：AVB 张力基线 C_fwd≈0.41（v_fwd0·C_fwd = 有效爬行速度）
C_FWD_BASELINE = 0.41
# 主 agent 预算裁决 2026-08-24（m4_env_notes §L21）：CSV protocol.t_total_ms 为缩减后的
# Brian2 闭环协议（5000ms）；生物带 [0.3,0.7]（P4b）验证主体 = numpy 行为参考模型在
# 全协议 T=25s 下进行（N=20），见 calibrate_behavior 的 BAND_T_MS。
BAND_T_MS = 25000.0        # 全协议时长 [ms]：numpy 参考模型的生物带验证协议


# --------------------------------------------------------------------- #
# CSV 读取（data/m4_chemotaxis_params.csv：神经元/突触/转导/环境/身体/协议/全局）
# --------------------------------------------------------------------- #
def load_m4_params(csv_path: str = PARAMS_CSV) -> dict:
    """读入 m4_chemotaxis_params.csv → neurons/synapses/muscles/trans/mech_a/env/body/protocol/params。"""
    neurons, synapses, muscles = {}, [], {}
    trans, mech_a, env, body, protocol, params = {}, {}, {}, {}, {}, {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(row for row in f if not row.strip().startswith("#"))
        for r in reader:
            role = (r.get("role") or "").strip()
            nclass = (r.get("neuron_class") or "").strip()
            stype = (r.get("synapse_type") or "").strip()
            if stype == "muscle":
                muscles[(r["synapse_to"] or "").strip()] = dict(
                    w=float(r["g_max_ns"]), delay_ms=float(r["delay_ms"]))
            elif stype in ("ampa", "gaba", "nmda"):
                synapses.append(dict(
                    pre=(r["synapse_from"] or "").strip(),
                    post=(r["synapse_to"] or "").strip(),
                    stype=stype,
                    g_max_ns=float(r["g_max_ns"]),
                    delay_ms=float(r["delay_ms"]),
                ))
            elif role == "transduction":
                trans[nclass] = _row_value(r)
            elif role == "mechanism_a":
                mech_a[nclass] = _row_value(r)
            elif role == "env":
                env[nclass] = _row_value(r)
            elif role == "body":
                body[nclass] = _row_value(r)
            elif role == "protocol":
                protocol[nclass] = _row_value(r)
            elif role == "param":
                params[nclass] = _row_value(r)
            elif role in ("ASEL", "ASER", "AIYL", "AIYR", "AIBL", "AIBR",
                          "RIAL", "RIAR", "AVBL", "AVBR", "SMDDL", "SMDDR",
                          "SMDVL", "SMDVR", "VB", "DB", "AVAL", "AVAR",
                          "RMED", "RMEV"):
                tonic = (r.get("tonic_uA_cm2") or "").strip()
                neurons[role] = dict(
                    tonic_uA_cm2=float(tonic) if tonic else 0.0)
    return dict(neurons=neurons, synapses=synapses, muscles=muscles,
                transduction=trans, mech_a=mech_a, env=env, body=body,
                protocol=protocol, params=params)


def _row_value(r):
    """行 value（数值→float；非数值（如 ase_site/boundary）→ str）。"""
    val = (r.get("value") or "").strip()
    if not val:  # 兼容把值写在 note 列
        val = (r.get("note") or "").strip()
    try:
        return float(val)
    except ValueError:
        return val


def _param_f(cfg: dict, name: str) -> float:
    return float(cfg["params"][name])


# --------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------- #
def detect_spikes(t_ms: np.ndarray, v_mv: np.ndarray,
                  thresh: float = SPIKE_THRESH_MV,
                  refractory_ms: float = SPIKE_REF_MS) -> np.ndarray:
    """V 上冲过 thresh 的峰时刻（ms）：边沿检测 + 窗口内峰定位 + 去重（同 build_reflex_ref）。"""
    v = np.asarray(v_mv, dtype=float)
    t = np.asarray(t_ms, dtype=float)
    above = v > thresh
    edges = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    times = []
    for e in edges:
        win = (t >= t[e]) & (t <= t[e] + refractory_ms)
        times.append(float(t[win][np.argmax(v[win])]) if win.sum() else float(t[e]))
    out = []
    for x in times:
        if not out or x - out[-1] > refractory_ms:
            out.append(x)
    return np.asarray(out)


# --------------------------------------------------------------------- #
# NEURON 核心子链
# --------------------------------------------------------------------- #
def _run_subchain(spec, names, chain_syns, stim_name,
                  t_total: float = SUBCHAIN_T_MS,
                  stim_start: float = STIM_START_MS,
                  stim_dur: float = STIM_DUR_MS,
                  i_ase: float = I_ASE_UA_CM2) -> dict:
    """构建一条子链（clear=True 首神经元 + clear=False 续建）并运行 cvode 高精度。

    names      : 链内神经元名（按传播顺序，小写）
    chain_syns : [(pre, post, g_max_ns, delay_ms), ...]（AMPA 链）
    stim_name  : 刺激神经元（ASE；IClamp 注入其 soma，编码 ON/OFF 阶跃）
    """
    from neuron import h

    secs = {}
    first = True
    for name in names:
        secs[name] = build_neuron(spec, clear=first, name_prefix=f"{name}_")
        first = False

    h.load_file("stdrun.hoc")
    h.celsius = 6.3                     # 硬约束：Q10 参考温度（SESSION_CONTEXT §四 #2）
    h.v_init = V0

    # 点过程引用列表持有防 GC（M2 L8）
    clamps, syns, ncs = [], [], []

    # ASE 阶跃电流（soma；密度→nA 按 SOMA_AREA_CM2 换算）
    amp = i_ase * 1e-6 * SOMA_AREA_CM2 * 1e9
    cl = h.IClamp(secs[stim_name]["soma"](0.5))
    cl.delay = stim_start
    cl.dur = stim_dur
    cl.amp = amp
    clamps.append(cl)

    # 突触链：ExpSyn（AMPA，τ/E 沿 m2 行）+ NetCon(pre node3 → post soma)
    m2 = load_synapse_params()
    base = m2["ampa"]
    for pre, post, g, delay in chain_syns:
        syn = h.ExpSyn(secs[post]["soma"](0.5))
        syn.tau = base.tau_ms
        syn.e = base.e_rev_mv
        nc = h.NetCon(secs[pre]["node3"](0.5)._ref_v, syn, sec=secs[pre]["node3"])
        nc.threshold = -20.0
        nc.delay = delay
        nc.weight[0] = g * 1e-3          # ExpSyn weight 单位 µS（µS = nS×1e-3）
        syns.append(syn)
        ncs.append(nc)

    # 记录：各级 node3 + 第二级 soma（ASE→AIY/AIB PSP）
    tvec = h.Vector(); tvec.record(h._ref_t)
    vrec = {}
    for name in names:
        vv = h.Vector(); vv.record(secs[name]["node3"](0.5)._ref_v)
        vrec[name] = vv
    vpsp = h.Vector(); vpsp.record(secs[names[1]]["soma"](0.5)._ref_v)

    cvode = h.CVode()
    cvode.active(1)
    cvode.atol(1e-8)                     # 硬约束：参考真理容差
    cvode.rtol(1e-8)
    h.tstop = t_total
    h.run()

    # 重采样到均匀网格
    t_irr = np.array(tvec)
    order = np.argsort(t_irr)
    t_u = np.arange(0.0, t_total + DT_OUT / 2, DT_OUT)
    vgrid, spikes = {}, {}
    for name, vv in vrec.items():
        v = np.interp(t_u, t_irr[order], np.array(vv)[order])
        vgrid[name] = v
        spikes[name] = detect_spikes(t_u, v)
    psp = np.interp(t_u, t_irr[order], np.array(vpsp)[order])

    def _first(name):
        return float(spikes[name][0]) if len(spikes[name]) else float("nan")

    return dict(t_ms=t_u, vgrid=vgrid, spikes=spikes, psp=psp,
                first={n: _first(n) for n in names})


def run_subchains(spec) -> dict:
    """两条核心子链 → spike_times / PSP / 链传导时间。"""
    # 链 A：ASEL → AIYL → AVBL（前进；上升阶跃 ON 编码）
    chain_a = _run_subchain(
        spec,
        names=["asel", "aiyl", "avbl"],
        chain_syns=[("asel", "aiyl", 5.0, 0.5), ("aiyl", "avbl", 5.0, 0.5)],
        stim_name="asel",
    )
    # 链 B：ASER → AIBL → RIAL → SMDDL（转向；下降阶跃 OFF 编码）
    chain_b = _run_subchain(
        spec,
        names=["aser", "aibl", "rial", "smddl"],
        chain_syns=[("aser", "aibl", 5.0, 0.5), ("aibl", "rial", 5.0, 0.5),
                    ("rial", "smddl", 5.0, 0.5)],
        stim_name="aser",
    )

    fa, fb = chain_a["first"], chain_b["first"]
    out = {
        "t_ms": chain_a["t_ms"],
        "spike_times_asel": chain_a["spikes"]["asel"],
        "spike_times_aiyl": chain_a["spikes"]["aiyl"],
        "spike_times_avbl": chain_a["spikes"]["avbl"],
        "spike_times_aser": chain_b["spikes"]["aser"],
        "spike_times_aibl": chain_b["spikes"]["aibl"],
        "spike_times_rial": chain_b["spikes"]["rial"],
        "spike_times_smddl": chain_b["spikes"]["smddl"],
        "v_aiyl_soma_mv": chain_a["psp"],
        "v_aibl_soma_mv": chain_b["psp"],
        "v_asel_node3_mv": chain_a["vgrid"]["asel"],
        "v_aser_node3_mv": chain_b["vgrid"]["aser"],
        "chain_time_ms_a": fa["avbl"] - fa["asel"],
        "chain_time_ms_b": fb["smddl"] - fb["aser"],
        "chain_time_ms_b_to_ria": fb["rial"] - fb["aser"],
        "_subchain_a": chain_a,
        "_subchain_b": chain_b,
    }
    return out


# --------------------------------------------------------------------- #
# 行为参考模型（纯 numpy pirouette，引擎无关；CI 统计与 §2.4 逐条对应）
# --------------------------------------------------------------------- #
def gradient_c(x: float, y: float, env: dict) -> float:
    """静态食物梯度：C = C_max·exp(−r²/2σ²) + C_bg（相对浓度 0..1）。"""
    r2 = (x - env["food_x"]) ** 2 + (y - env["food_y"]) ** 2
    return env["C_bg"] + env["C_max"] * math.exp(-r2 / (2.0 * env["sigma"] ** 2))


def run_behavior_trial(cfg: dict, rng, x0: float, y0: float, theta0: float) -> dict:
    """单试次 pirouette：每 tick 采样 C → s → s<−θ_pir 触发随机转向 → 积分 → CI。

    cfg 需含：dt_b_ms/t_total_ms/tau_win_ms/arena_L/food_x/food_y/sigma/C_max/C_bg/
              v_fwd/omega_pir/t_pir_ms/theta_pir（delta C per ms 阈值）
    """
    dt_s = cfg["dt_b_ms"] / 1000.0
    n_ticks = int(round(cfg["t_total_ms"] / cfg["dt_b_ms"]))
    win_ticks = max(1, int(round(cfg["tau_win_ms"] / cfg["dt_b_ms"])))
    buf_n = win_ticks + 1
    cbuf = [cfg["C_bg"]] * buf_n        # 环形 C 缓冲（τ_win 记忆）

    xs = np.empty(n_ticks)
    ys = np.empty(n_ticks)
    quad = np.empty(n_ticks, dtype=np.int8)   # 1=食物象限 2=对侧 0=其他
    L = cfg["arena_L"]

    x, y, theta = x0, y0, theta0
    turn_remaining = 0.0
    turn_dir = 1.0
    for i in range(n_ticks):
        C = gradient_c(x, y, cfg)
        cbuf[i % buf_n] = C
        if i >= win_ticks:
            c_prev = cbuf[(i + 1) % buf_n]     # 窗口前 C
            s = (C - c_prev) / cfg["tau_win_ms"]   # [ΔC/ms]
        else:
            s = 0.0
        if s < -cfg["theta_pir"] and turn_remaining <= 0.0:
            turn_remaining = cfg["t_pir_ms"]
            turn_dir = rng.choice((-1.0, 1.0))
        if turn_remaining > 0.0:
            omega = turn_dir * cfg["omega_pir"]
            turn_remaining -= cfg["dt_b_ms"]
        else:
            omega = 0.0
        x += cfg["v_fwd"] * math.cos(theta) * dt_s
        y += cfg["v_fwd"] * math.sin(theta) * dt_s
        theta += omega * dt_s
        # 反射边界（P3：轨迹有界）
        if x < 0.0:
            x, theta = -x, math.pi - theta
        elif x > L:
            x, theta = 2.0 * L - x, math.pi - theta
        if y < 0.0:
            y, theta = -y, -theta
        elif y > L:
            y, theta = 2.0 * L - y, -theta
        xs[i], ys[i] = x, y
        if x > L / 2.0 and y > L / 2.0:
            quad[i] = 1
        elif x < L / 2.0 and y < L / 2.0:
            quad[i] = 2

    ci = float((quad == 1).sum() - (quad == 2).sum()) / n_ticks   # (T_in − T_out)/T_total
    return dict(ci=ci, xs=xs, ys=ys, quad=quad)


def run_behavior_group(cfg: dict, seed: int = RNG_SEED) -> dict:
    """N=20 试次（伪随机起点扰动 + 随机转向方向；同一 rng 顺序抽样保证可复现）。"""
    rng = np.random.default_rng(seed)
    n = int(cfg["n_trials"])
    cis, trajs = [], []
    jit = cfg["start_jitter"]
    for _ in range(n):
        x0 = cfg["start_x"] + jit * (rng.random() * 2.0 - 1.0)
        y0 = cfg["start_y"] + jit * (rng.random() * 2.0 - 1.0)
        theta0 = rng.random() * 2.0 * math.pi
        r = run_behavior_trial(cfg, rng, x0, y0, theta0)
        cis.append(r["ci"])
        trajs.append(r)
    return dict(ci=np.asarray(cis), trajs=trajs)


def _stats(x: np.ndarray):
    """CĪ、SEM、单样本 t（H0: μ=0）、p、Cohen's d（样本 SD）。"""
    from scipy import stats as sps
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    sem = sd / math.sqrt(n) if n > 1 else 0.0
    if sd > 0:
        t = mean / sem
        p = float(2.0 * (1.0 - sps.t.cdf(abs(t), n - 1)))
        d = mean / sd
    else:
        t, p, d = (0.0 if abs(mean) < 1e-12 else float("inf")), 1.0, float("inf")
    return dict(mean=mean, sem=sem, sd=sd, t=t, p=p, d=d)


def calibrate_behavior(env: dict, body: dict, protocol: dict,
                       transduction: dict = None, mech_a: dict = None) -> dict:
    """pirouette 参数校准：θ_pir × 转向角 × v 网格 → 选 CĪ∈[0.3,0.7] 的组合。

    步骤 4（B1c）机制 A 落地修订：CSV mechanism_a 行（θ_pir/ω_pir/T_pir）为
    校准锚点——网格沿锚点 θ_pir 扩展，ω_pir 用 CSV 值（转角 α = ω_pir·T_pir）；
    v 锚点 = body.v_fwd0·C_FWD_BASELINE（B1c 实测基线 0.41，与 Brian2 虫一致）。

    无梯度对照（C≡C_bg）下 s≡0 → 永不转向 → 直线 + 反射，CI 均值应≈0（度量校准）。
    """
    tau_win = float((transduction or {}).get("tau_win", 100.0))
    omega_pir = float((mech_a or {}).get("omega_pir", OMEGA_PIR))
    theta_anchor = float((mech_a or {}).get("theta_pir", 4.0e-6))
    t_pir_anchor = float((mech_a or {}).get("t_pir_ms", 1571.0))
    v_anchor = float(body["v_fwd0"]) * C_FWD_BASELINE   # 有效爬行速度（B1c 实测）
    base = dict(
        dt_b_ms=body["dt_b"], t_total_ms=BAND_T_MS,  # L21：带内验证用全协议
        tau_win_ms=tau_win,
        arena_L=env["arena_L"], food_x=env["food_x"], food_y=env["food_y"],
        sigma=env["sigma"], C_max=env["C_max"], C_bg=env["C_bg"],
        omega_pir=omega_pir, n_trials=protocol["n_trials"],
        start_x=protocol["start_x"], start_y=protocol["start_y"],
        start_jitter=protocol["start_jitter"],
    )
    grid_theta = tuple(sorted(set(CAL_THETA_PIR) | {theta_anchor}))
    grid_v = tuple(sorted(set(CAL_V_FWD) | {round(v_anchor, 3)}))
    # 锚点转角（CSV t_pir_ms 折算）优先进入网格
    alpha_anchor = math.degrees(omega_pir * t_pir_anchor / 1000.0)
    grid_alpha = tuple(sorted(set(CAL_ALPHA_DEG) | {round(alpha_anchor, 1)}))

    best = None
    scan = []
    for theta_pir in grid_theta:
        for alpha_deg in grid_alpha:
            for v in grid_v:
                cfg = dict(base)
                cfg.update(theta_pir=theta_pir, v_fwd=v,
                           t_pir_ms=math.radians(alpha_deg) / omega_pir * 1000.0)
                g = run_behavior_group(cfg, seed=RNG_SEED)
                s = _stats(g["ci"])
                row = dict(theta_pir=theta_pir, alpha_deg=alpha_deg, v_fwd=v,
                           ci_mean=s["mean"], ci_sem=s["sem"])
                scan.append(row)
                if 0.3 <= s["mean"] <= 0.7:
                    # 优先 v 接近锚点（协议一致性），次优先 θ_pir 接近 CSV 锚点，
                    # 再 CI 接近带中心 0.5
                    key = (abs(v - v_anchor), abs(theta_pir - theta_anchor),
                           abs(s["mean"] - 0.5))
                    if best is None or key < (abs(best["cfg"]["v_fwd"] - v_anchor),
                                              abs(best["cfg"]["theta_pir"] - theta_anchor),
                                              abs(best["stats"]["mean"] - 0.5)):
                        best = dict(cfg=cfg, stats=s, row=row)
    if best is None:  # 全网格无解（结构限制）——如实取最接近带内的组合，记录测量限制
        cand = min(scan, key=lambda r: min(abs(r["ci_mean"] - 0.3),
                                           abs(r["ci_mean"] - 0.7)))
        cfg = dict(base)
        cfg.update(theta_pir=cand["theta_pir"], v_fwd=cand["v_fwd"],
                   t_pir_ms=math.radians(cand["alpha_deg"]) / omega_pir * 1000.0)
        g = run_behavior_group(cfg, seed=RNG_SEED)
        best = dict(cfg=cfg, stats=_stats(g["ci"]), row=cand,
                    out_of_band=True)
    # 无梯度对照（同协议，C_max 段置 0 → C≡C_bg）
    ctrl_cfg = dict(best["cfg"]); ctrl_cfg["C_max"] = 0.0
    ctrl = run_behavior_group(ctrl_cfg, seed=RNG_SEED)
    ctrl_stats = _stats(ctrl["ci"])
    return dict(cfg=best["cfg"], stats=best["stats"], row=best["row"],
                ctrl_ci=ctrl["ci"], ctrl_stats=ctrl_stats,
                out_of_band=best.get("out_of_band", False),
                anchor=dict(theta_pir=theta_anchor, omega_pir=omega_pir,
                            t_pir_ms=t_pir_anchor, v_fwd=v_anchor),
                scan=scan)


# --------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------- #
def run_reference(out_npz: str = REF_NPZ) -> str:
    """NEURON 核心子链 + 行为参考模型校准 → 落盘 npz。"""
    os.makedirs(DATA_DIR, exist_ok=True)
    cfg_all = load_m4_params()
    spec = load_morphology()

    # ---- 神经级：NEURON 核心子链 ----
    sub = run_subchains(spec)

    # ---- 行为级：pirouette 校准（机制 A 锚点 = CSV mechanism_a 行，B1c 步骤 4 联动）----
    beh = calibrate_behavior(cfg_all["env"], cfg_all["body"],
                             cfg_all["protocol"], cfg_all["transduction"],
                             cfg_all["mech_a"])
    st, ct = beh["stats"], beh["ctrl_stats"]

    out = {k: v for k, v in sub.items() if not k.startswith("_")}

    # 用最终配置重跑 20 试次拿逐试次 CI + 示例轨迹
    final = run_behavior_group(beh["cfg"], seed=RNG_SEED)
    out["behavior_ci"] = final["ci"]
    out["behavior_ci_mean"] = st["mean"]
    out["behavior_ci_sem"] = st["sem"]
    out["behavior_t_stat"] = st["t"]
    out["behavior_p_value"] = st["p"]
    out["behavior_cohen_d"] = st["d"]
    out["behavior_ci_ctrl"] = beh["ctrl_ci"]
    out["behavior_ci_ctrl_mean"] = ct["mean"]
    out["behavior_ci_ctrl_p"] = ct["p"]
    out["traj1_x"] = final["trajs"][0]["xs"]
    out["traj1_y"] = final["trajs"][0]["ys"]
    out["traj2_x"] = final["trajs"][1]["xs"]
    out["traj2_y"] = final["trajs"][1]["ys"]

    meta = dict(
        engine=("NEURON 9.0.1 cvode（atol=rtol=1e-8, celsius=6.3, v_init=-65mV）"
                " + 行为参考模型（纯 numpy pirouette）"),
        params_csv="data/m4_chemotaxis_params.csv",
        dt_out_ms=DT_OUT,
        subchain_stim=dict(start_ms=STIM_START_MS, dur_ms=STIM_DUR_MS,
                           i_ase_uA_cm2=I_ASE_UA_CM2, site="ase_soma"),
        subchain_synapses=dict(
            chain_a=[("asel", "aiyl", "ampa", 5.0, 0.5),
                     ("aiyl", "avbl", "ampa", 5.0, 0.5)],
            chain_b=[("aser", "aibl", "ampa", 5.0, 0.5),
                     ("aibl", "rial", "ampa", 5.0, 0.5),
                     ("rial", "smddl", "ampa", 5.0, 0.5)]),
        chain_a_first_spikes={k: sub["_subchain_a"]["first"][k]
                              for k in ("asel", "aiyl", "avbl")},
        chain_b_first_spikes={k: sub["_subchain_b"]["first"][k]
                              for k in ("aser", "aibl", "rial", "smddl")},
        chain_time_ms_a=float(sub["chain_time_ms_a"]),
        chain_time_ms_b=float(sub["chain_time_ms_b"]),
        chain_time_ms_b_to_ria=float(sub["chain_time_ms_b_to_ria"]),
        behavior_pirouette=dict(
            theta_pir=beh["cfg"]["theta_pir"],        # [ΔC/ms] 转向触发阈值
            omega_pir=beh["cfg"]["omega_pir"],        # [rad/s]
            t_pir_ms=beh["cfg"]["t_pir_ms"],          # [ms]
            turn_angle_deg=math.degrees(beh["cfg"]["omega_pir"]
                                        * beh["cfg"]["t_pir_ms"] / 1000.0),
            v_fwd=beh["cfg"]["v_fwd"],                # [units/s]
            sigma=beh["cfg"]["sigma"], tau_win_ms=beh["cfg"]["tau_win_ms"],
            dt_b_ms=beh["cfg"]["dt_b_ms"],
            n_trials=int(beh["cfg"]["n_trials"]), seed=RNG_SEED,
            ci_mean=float(st["mean"]), ci_sem=float(st["sem"]),
            t_stat=float(st["t"]), p_value=float(st["p"]),
            cohen_d=float(st["d"]),
            ci_ctrl_mean=float(ct["mean"]), ci_ctrl_p=float(ct["p"]),
            ci_band=[0.3, 0.7], tolerance=[0.25, 0.75],
            out_of_band=bool(beh["out_of_band"]),
            mech_a_anchor=beh["anchor"],   # CSV mechanism_a 行锚点（B1c 定稿）
            calibration_scan=[(r["theta_pir"], r["alpha_deg"], r["v_fwd"],
                               round(r["ci_mean"], 3)) for r in beh["scan"]]),
        band_protocol_ms=float(BAND_T_MS),
        reduced_protocol_ms=float(cfg_all["protocol"].get("t_total_ms", BAND_T_MS)),  # L21
        note=("行为参考模型与 Brian2 虫须共用同一运动学/CI 统计代码（src/chemotaxis_body.py / "
              "chemotaxis_env.py 引擎无关部分，B1b 移植时保持同函数）；"
              "P1 静止段静默判据在无阶跃对照试次上评估（τ_win 记忆，见 docs/m4_env_notes.md §L4）"),
    )
    np.savez(out_npz, **out, meta=np.array(meta, dtype=object))
    return out_npz


if __name__ == "__main__":
    out = run_reference()
    d = np.load(out, allow_pickle=True)
    meta = d["meta"].item()
    print(f"参考解已写入: {out}")
    print(f"键数: {len([k for k in d.files if k != 'meta'])}")
    print("--- NEURON 核心子链 ---")
    for k in ("asel", "aiyl", "avbl"):
        sp = d[f"spike_times_{k}"]
        print(f"  链A {k:6s} 首发放: {sp[0] if len(sp) else '-':>7.2f} ms (n={len(sp)})")
    for k in ("aser", "aibl", "rial", "smddl"):
        sp = d[f"spike_times_{k}"]
        print(f"  链B {k:6s} 首发放: {sp[0] if len(sp) else '-':>7.2f} ms (n={len(sp)})")
    print(f"  链传导 A(ASEL→AVBL): {d['chain_time_ms_a']:.2f} ms | "
          f"B(ASER→SMDDL): {d['chain_time_ms_b']:.2f} ms | B→RIA: {d['chain_time_ms_b_to_ria']:.2f} ms")
    print("--- 行为参考模型（pirouette 校准）---")
    pb = meta["behavior_pirouette"]
    print(f"  θ_pir={pb['theta_pir']:.2e} /ms, 转向角={pb['turn_angle_deg']:.0f}°, "
          f"T_pir={pb['t_pir_ms']:.0f}ms, v={pb['v_fwd']} u/s")
    print(f"  CĪ={pb['ci_mean']:.3f} ± {pb['ci_sem']:.3f} (SEM), t={pb['t_stat']:.2f}, "
          f"p={pb['p_value']:.3f}, d={pb['cohen_d']:.2f} | 落带[0.3,0.7]: "
          f"{0.3 <= pb['ci_mean'] <= 0.7} | out_of_band={pb['out_of_band']}")
    print(f"  无梯度对照 CĪ={pb['ci_ctrl_mean']:.3f}, p={pb['ci_ctrl_p']:.3f} "
          f"(判据 |CĪ|<0.1 且 p>0.05: {abs(pb['ci_ctrl_mean']) < 0.1 and pb['ci_ctrl_p'] > 0.05})")
    print("--- 校准扫描（θ_pir, 转向角°, v, CĪ）---")
    for t_p, a, v, ci in pb["calibration_scan"]:
        print(f"  {t_p:.1e}  {a:5.0f}°  v={v:.2f}  CĪ={ci:+.3f}")
