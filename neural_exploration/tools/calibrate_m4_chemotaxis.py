"""M4 步骤 4：参数定稿与 CI 校准（B1c 执行节点）。

清单《生物仿真M4实施清单》§5（校准协议）+ §2.4（机制 A 落地修订）：

  - 校准协议：6 维网格（g_ON/g_OFF、τ_win、v_fwd0、ω_max、σ、T_total 或机制 A 关键参数），
    每点 N=10 试次、确定性、无梯度对照并行；选**最小充分组合**（D4 原则）：
    有梯度 CI ∈ [0.3,0.7]（容差 [0.25,0.75]）且无梯度对照 p>0.05。
  - 行为参考模型（纯 numpy pirouette）同步校准到同一带（P6 基准）——本工具用
    ChemotaxisBody/ChemotaxisEnv（引擎无关，与 Brian2 虫共用同一运动学/CI 代码）。
  - 计算预算（B1c 实测，见 m4_env_notes §L16）：闭环单试次 ≈ 156s 每 1000ms 仿真
    （20 神经元 HH rk4 dt=0.01ms）→ 网格扫描分两阶段：
      阶段 1（快）：numpy 行为参考模型扫描（分钟级）定候选带；
      阶段 2（慢）：Brian2 闭环在候选点上扫描（多进程并行，每点 10 试次）。
    无梯度对照（C≡C_bg → s≡0 → 零 ASE 发放 → 零转向事件）在固定 (v_fwd0, 协议) 下
    与电路参数无关 → 对照分布由 numpy 直行运动学一次算定（每 v_fwd0 一次），
    最终点用真实 Brian2 对照复核（等价性验证）。

用法：
  .venv-neuro/bin/python -m neural_exploration.tools.calibrate_m4_chemotaxis ref-scan
  .venv-neuro/bin/python -m neural_exploration.tools.calibrate_m4_chemotaxis probe
  .venv-neuro/bin/python -m neural_exploration.tools.calibrate_m4_chemotaxis scan --points <json>
  .venv-neuro/bin/python -m neural_exploration.tools.calibrate_m4_chemotaxis finalize
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.src.chemotaxis_body import ChemotaxisBody  # noqa: E402
from neural_exploration.src.chemotaxis_circuit import (  # noqa: E402
    ChemotaxisCircuit,
    load_chemotaxis_params,
)
from neural_exploration.src.chemotaxis_env import (  # noqa: E402
    ChemotaxisEnv,
    TimeDiffTracker,
    ci_group_stats,
)
from neural_exploration.src.chemotaxis_loop import ChemotaxisLoop  # noqa: E402

CSV = os.path.join(ROOT, "neural_exploration", "data", "m4_chemotaxis_params.csv")
CAL_CSV = os.path.join(ROOT, "neural_exploration", "data", "m4_calibration.csv")
CAL_PNG = os.path.join(ROOT, "neural_exploration", "reports", "neuro", "m4_calibration.png")

BAND_LO, BAND_HI = 0.3, 0.7          # 生物带（判据）
TOL_LO, TOL_HI = 0.25, 0.75          # 容差窗（判据）
N_SCAN_TRIALS = 10                   # 校准每点试次数（清单 §5）

# 实测（B1c probe，m4_env_notes §L16）：AVB 张力基线 C_fwd≈0.41（非 0.216）
C_FWD_BASELINE = 0.41


# --------------------------------------------------------------------- #
# 环境/身体构建（与 Brian2 闭环同一套代码）
# --------------------------------------------------------------------- #
def make_env(params) -> ChemotaxisEnv:
    e = params.env
    return ChemotaxisEnv(arena_L=e.arena_L, sigma=e.sigma, c_max=e.c_max,
                         c_bg=e.c_bg, food_x=e.food_x, food_y=e.food_y,
                         boundary=e.boundary)


def make_body(params, v_fwd0=None) -> ChemotaxisBody:
    b = params.body
    return ChemotaxisBody(
        v_fwd0=params.body.v_fwd0 if v_fwd0 is None else v_fwd0,
        omega_max=b.omega_max, dt_b=b.dt_b, v_osc=b.v_osc,
        arena_L=params.env.arena_L, boundary=params.env.boundary,
        turn_omega_pir=params.mech_a.omega_pir,
        turn_duration_ms=params.mech_a.t_pir_ms)


# --------------------------------------------------------------------- #
# 行为参考模型（纯 numpy pirouette；与 Brian2 共用 body/env/CI 代码）
# --------------------------------------------------------------------- #
def run_ref_trial(env: ChemotaxisEnv, body: ChemotaxisBody,
                  theta_pir: float, seed: int, t_total_ms: float,
                  v_fwd0: float, tau_win_ms: float,
                  start_x: float, start_y: float, theta0: float,
                  c_fwd: float = C_FWD_BASELINE) -> dict:
    """单试次参考 pirouette：s < −θ_pir → 转向事件（方向=试次种子伪随机）。

    与 Brian2 闭环机制 A 完全同构（body.trigger_turn/step + TimeDiffTracker +
    env.ci_per_trial），只是触发条件为纯 s 阈值（无 SMDD 电路门）。
    """
    dt_b = body.dt_b
    n_ticks = max(1, int(round(t_total_ms / dt_b)))
    body.reset(start_x, start_y, theta0)
    tracker = TimeDiffTracker(tau_win_ms, env.sample(start_x, start_y))
    rng = np.random.default_rng(seed)
    xs, ys = np.empty(n_ticks), np.empty(n_ticks)
    n_turn = 0
    for i in range(n_ticks):
        t_e = i * dt_b
        c_now = env.sample(body.x, body.y)
        s = tracker.s_at(t_e, c_now)
        if s < -theta_pir and not body.is_turning():
            direction = 1.0 if rng.random() < 0.5 else -1.0
            body.trigger_turn(direction)
            n_turn += 1
        body.step(c_fwd, 0.0, 0.0, dt_b, t_e)
        xs[i], ys[i] = body.x, body.y
    return dict(x=xs, y=ys, ci=env.ci_per_trial(xs, ys), n_turn=n_turn)


def surrogate_trial(env: ChemotaxisEnv, body: ChemotaxisBody,
                    theta_pir: float, g_off: float, i_thresh: float,
                    seed: int, t_total_ms: float, v_fwd0: float,
                    tau_win_ms: float, start_x: float, start_y: float,
                    theta0: float, c_fwd: float = C_FWD_BASELINE) -> dict:
    """机制 A 替代模型单试次：触发条件 = s < −θ_pir **且** g_off·|s| > I_thresh
    （ASER→AIB→RIA→SMDD 电路激活阈值，开环实测）；速度恒定 v=v_fwd0·C_FWD_BASELINE。

    与 Brian2 闭环同构（body/env/CI 共享代码），仅以电路响应函数代替神经元——
    用于 6 维网格快速扫描（候选点再由真实 Brian2 闭环验证）。
    """
    dt_b = body.dt_b
    n_ticks = max(1, int(round(t_total_ms / dt_b)))
    body.reset(start_x, start_y, theta0)
    tracker = TimeDiffTracker(tau_win_ms, env.sample(start_x, start_y))
    rng = np.random.default_rng(seed)
    xs, ys = np.empty(n_ticks), np.empty(n_ticks)
    n_turn = 0
    for i in range(n_ticks):
        t_e = i * dt_b
        c_now = env.sample(body.x, body.y)
        s = tracker.s_at(t_e, c_now)
        if s < -theta_pir and g_off * abs(s) > i_thresh and not body.is_turning():
            direction = 1.0 if rng.random() < 0.5 else -1.0
            body.trigger_turn(direction)
            n_turn += 1
        body.step(c_fwd, 0.0, 0.0, dt_b, t_e)
        xs[i], ys[i] = body.x, body.y
    return dict(x=xs, y=ys, ci=env.ci_per_trial(xs, ys), n_turn=n_turn)


def surrogate_group(env, body, theta_pir, g_off, i_thresh, v_fwd0, tau_win_ms,
                    t_total_ms, n_trials, seed_base, start_jitter,
                    start_x, start_y) -> dict:
    rng = np.random.default_rng(seed_base)
    cis, turns = [], []
    for trial in range(n_trials):
        if start_jitter > 0:
            sx = start_x + rng.normal(0.0, start_jitter)
            sy = start_y + rng.normal(0.0, start_jitter)
            th0 = rng.uniform(0.0, 2.0 * math.pi)
        else:
            sx, sy, th0 = start_x, start_y, 0.0
        r = surrogate_trial(env, body, theta_pir, g_off, i_thresh,
                            seed_base + trial, t_total_ms, v_fwd0, tau_win_ms,
                            sx, sy, th0)
        cis.append(r["ci"])
        turns.append(r["n_turn"])
    return dict(ci=np.asarray(cis), n_turns=turns)


def surrogate_scan(params, grid: dict, i_thresh: float = 25.0) -> list:
    """6 维网格扫描（替代模型，秒级）：g_off × θ_pir × v_fwd0 × σ × τ_win × T_total。"""
    proto = params.protocol
    rows = []
    for g_off in grid["g_off"]:
        for theta_pir in grid["theta_pir"]:
            for v_fwd0 in grid["v_fwd0"]:
                for sigma in grid["sigma"]:
                    for tau_win in grid["tau_win"]:
                        for t_total in grid["t_total"]:
                            env = ChemotaxisEnv(
                                arena_L=params.env.arena_L, sigma=sigma,
                                c_max=params.env.c_max, c_bg=params.env.c_bg,
                                food_x=params.env.food_x, food_y=params.env.food_y,
                                boundary=params.env.boundary)
                            body = make_body(params, v_fwd0)
                            g = surrogate_group(env, body, theta_pir, g_off,
                                                i_thresh, v_fwd0, tau_win,
                                                t_total, N_SCAN_TRIALS, 0,
                                                proto.start_jitter,
                                                proto.start_x, proto.start_y)
                            st = ci_group_stats(g["ci"], TOL_LO, TOL_HI)
                            rows.append(dict(
                                g_off=g_off, theta_pir=theta_pir, v_fwd0=v_fwd0,
                                sigma=sigma, tau_win_ms=tau_win,
                                t_total_ms=t_total, t_pir_ms=params.mech_a.t_pir_ms,
                                ci_mean=st["mean"], ci_sem=st["sem"],
                                ci_std=st["std"], p_value=st["p_value"],
                                cohen_d=st["cohen_d"], n=st["n"],
                                n_turns=int(np.mean(g["n_turns"])),
                                in_tol=TOL_LO <= st["mean"] <= TOL_HI,
                                in_band=BAND_LO <= st["mean"] <= BAND_HI,
                            ))
    return rows


def ref_group(env, body, theta_pir, v_fwd0, tau_win_ms, t_total_ms,
              n_trials, seed_base, start_jitter, start_x, start_y) -> dict:
    """N 试次参考组（伪随机起点扰动 + 随机转向方向，同 Brian2 run_trials 语义）。"""
    rng = np.random.default_rng(seed_base)
    cis = []
    for trial in range(n_trials):
        if start_jitter > 0:
            sx = start_x + rng.normal(0.0, start_jitter)
            sy = start_y + rng.normal(0.0, start_jitter)
            th0 = rng.uniform(0.0, 2.0 * math.pi)
        else:
            sx, sy, th0 = start_x, start_y, 0.0
        r = run_ref_trial(env, body, theta_pir, seed_base + trial, t_total_ms,
                          v_fwd0, tau_win_ms, sx, sy, th0)
        cis.append(r["ci"])
    return dict(ci=np.asarray(cis))


def ref_scan(params, grid: dict) -> list:
    """行为参考模型网格扫描（阶段 1，快）：θ_pir × 转角 × v_fwd0 × T_total × σ × τ_win。"""
    proto = params.protocol
    rows = []
    for theta_pir in grid["theta_pir"]:
        for alpha_deg in grid["alpha_deg"]:
            for v_fwd0 in grid["v_fwd0"]:
                for sigma in grid["sigma"]:
                    for tau_win in grid["tau_win"]:
                        for t_total in grid["t_total"]:
                            env = ChemotaxisEnv(
                                arena_L=params.env.arena_L, sigma=sigma,
                                c_max=params.env.c_max, c_bg=params.env.c_bg,
                                food_x=params.env.food_x, food_y=params.env.food_y,
                                boundary=params.env.boundary)
                            omega_pir = params.mech_a.omega_pir
                            t_pir_ms = math.degrees(alpha_deg) / 180.0 * math.pi \
                                / omega_pir * 1000.0
                            body = make_body(params, v_fwd0)
                            body.turn_duration_ms = t_pir_ms
                            g = ref_group(env, body, theta_pir, v_fwd0, tau_win,
                                          t_total, N_SCAN_TRIALS, 0,
                                          proto.start_jitter, proto.start_x,
                                          proto.start_y)
                            st = ci_group_stats(g["ci"], TOL_LO, TOL_HI)
                            rows.append(dict(
                                theta_pir=theta_pir, alpha_deg=alpha_deg,
                                v_fwd0=v_fwd0, sigma=sigma, tau_win_ms=tau_win,
                                t_total_ms=t_total, t_pir_ms=t_pir_ms,
                                ci_mean=st["mean"], ci_sem=st["sem"],
                                ci_std=st["std"], p_value=st["p_value"],
                                cohen_d=st["cohen_d"], n=st["n"],
                                in_tol=TOL_LO <= st["mean"] <= TOL_HI,
                                in_band=BAND_LO <= st["mean"] <= BAND_HI,
                            ))
    return rows


def ref_control(params, v_fwd0, sigma, tau_win, t_total, n=20) -> dict:
    """无梯度对照（C≡C_bg）：s≡0 → 永不转向 → 直行 + 反射（几何决定，电路无关）。"""
    env = ChemotaxisEnv(arena_L=params.env.arena_L, sigma=sigma, c_max=0.0,
                        c_bg=params.env.c_bg, food_x=params.env.food_x,
                        food_y=params.env.food_y, boundary=params.env.boundary)
    body = make_body(params, v_fwd0)
    return ref_group(env, body, params.mech_a.theta_pir, v_fwd0, tau_win,
                     t_total, n, 1000, params.protocol.start_jitter,
                     params.protocol.start_x, params.protocol.start_y)


# --------------------------------------------------------------------- #
# Brian2 闭环扫描（阶段 2，慢；多进程并行）
# --------------------------------------------------------------------- #
def brian2_point(params_overrides: dict, n_trials: int = N_SCAN_TRIALS,
                 t_total_ms: float = None, seed_base: int = 0,
                 verbose: bool = False) -> dict:
    """一个校准点：Brian2 闭环 N 试次（确定性；机制 A 方向=试次种子）。

    params_overrides: {g_on, g_off, tau_win_ms, v_fwd0, omega_max, sigma,
                       t_total_ms, theta_pir, omega_pir, t_pir_ms}
    """
    circ = ChemotaxisCircuit(csv_path=CSV)
    p = circ.params
    if "g_on" in params_overrides:
        circ.set_ase_gains(g_on=params_overrides["g_on"])
    if "g_off" in params_overrides:
        circ.set_ase_gains(g_off=params_overrides["g_off"])
    if "tau_win_ms" in params_overrides:
        p.transduction.tau_win_ms = float(params_overrides["tau_win_ms"])
    if "v_fwd0" in params_overrides:
        p.body.v_fwd0 = float(params_overrides["v_fwd0"])
    if "omega_max" in params_overrides:
        p.body.omega_max = float(params_overrides["omega_max"])
    if "sigma" in params_overrides:
        p.env.sigma = float(params_overrides["sigma"])
    if "theta_pir" in params_overrides:
        p.mech_a.theta_pir = float(params_overrides["theta_pir"])
    if "omega_pir" in params_overrides:
        p.mech_a.omega_pir = float(params_overrides["omega_pir"])
    if "t_pir_ms" in params_overrides:
        p.mech_a.t_pir_ms = float(params_overrides["t_pir_ms"])
    t_total = float(t_total_ms or params_overrides.get("t_total_ms")
                    or p.protocol.t_total_ms)

    env = make_env(p)
    loop = ChemotaxisLoop(circ, env=env, seed=seed_base)
    t0 = time.time()
    results = loop.run_trials(n_trials=n_trials, seed_base=seed_base,
                              t_total_ms=t_total)
    dt = time.time() - t0
    cis = np.array([r.ci for r in results])
    n_turns = [r.meta.get("n_turn_events", 0) for r in results]
    st = ci_group_stats(cis, TOL_LO, TOL_HI)
    if verbose:
        print(f"  point done in {dt:.0f}s: CĪ={st['mean']:.3f}±{st['sem']:.3f} "
              f"(n={st['n']}) turns={n_turns}")
    return dict(
        params=params_overrides, t_total_ms=t_total,
        ci_values=cis.tolist(), ci_mean=st["mean"], ci_sem=st["sem"],
        ci_std=st["std"], p_value=st["p_value"], cohen_d=st["cohen_d"],
        n_turns=n_turns, wall_s=dt,
    )


def run_scan(points: list, n_workers: int = 6, n_trials: int = N_SCAN_TRIALS) -> list:
    """并行扫描多个校准点 → 逐点结果（已含阶段 2 判据字段）。"""
    out_rows = []
    for idx, point in enumerate(points):
        res = brian2_point(point, n_trials=n_trials)
        row = dict(point_id=idx, **res["params"],
                   t_total_ms=res["t_total_ms"], ci_mean=res["ci_mean"],
                   ci_sem=res["ci_sem"], ci_std=res["ci_std"],
                   p_value=res["p_value"], cohen_d=res["cohen_d"],
                   ci_values=json.dumps(res["ci_values"]),
                   n_turns=json.dumps(res["n_turns"]), wall_s=res["wall_s"])
        out_rows.append(row)
        _append_cal_row(row)
    return out_rows


_CAL_COLS = ["point_id", "g_on", "g_off", "tau_win_ms", "v_fwd0", "omega_max",
             "sigma", "t_total_ms", "theta_pir", "omega_pir", "t_pir_ms",
             "ci_mean", "ci_sem", "ci_std", "p_value", "cohen_d",
             "ci_values", "n_turns", "wall_s", "note"]


def _append_cal_row(row: dict):
    """增量写 data/m4_calibration.csv（每点落盘，防长扫描中断丢数据）。"""
    import csv as _csv

    new = {k: row.get(k, "") for k in _CAL_COLS}
    os.makedirs(os.path.dirname(CAL_CSV), exist_ok=True)
    exists = os.path.exists(CAL_CSV)
    with open(CAL_CSV, "a", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=_CAL_COLS)
        if not exists:
            w.writeheader()
        w.writerow(new)


def write_cal_csv(rows: list):
    import csv as _csv

    os.makedirs(os.path.dirname(CAL_CSV), exist_ok=True)
    with open(CAL_CSV, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=_CAL_COLS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in _CAL_COLS})


def plot_calibration(rows: list, out_png: str = CAL_PNG):
    """CI 热力图（θ_pir × g_off 若存在）+ 扫描轨迹（CĪ vs 参数）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 左：CĪ 随扫描序号的轨迹（每点竖线 = SEM），带区标注
    ax = axes[0]
    idx = list(range(len(rows)))
    means = [r.get("ci_mean", float("nan")) for r in rows]
    sems = [r.get("ci_sem", 0.0) or 0.0 for r in rows]
    ax.errorbar(idx, means, yerr=sems, fmt="o-", ms=5, lw=1.2, color="#1f77b4",
                capsize=3)
    ax.axhspan(TOL_LO, TOL_HI, color="green", alpha=0.12)
    ax.axhspan(BAND_LO, BAND_HI, color="green", alpha=0.25)
    ax.axhline(0.0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("scan point #")
    ax.set_ylabel("CĪ ± SEM (gradient)")
    ax.set_title("CĪ scan trajectory (green = tolerance band [0.25,0.75])")
    ax.grid(alpha=0.3)
    for i, r in enumerate(rows):
        lbl = f"#{i}: g_off={r.get('g_off','-')} θ_pir={r.get('theta_pir','-'):.1e} T={r.get('t_total_ms','-')}"
        ax.annotate(lbl, (i, means[i]), fontsize=6, ha="center",
                    va="bottom", rotation=45, xytext=(0, 4),
                    textcoords="offset points")

    # 右：若扫描含 g_off × θ_pir 网格 → 热力图；否则 CĪ vs θ_pir
    ax2 = axes[1]
    g_offs = sorted({r.get("g_off") for r in rows if r.get("g_off") is not None})
    ths = sorted({r.get("theta_pir") for r in rows if r.get("theta_pir") is not None})
    if len(g_offs) >= 2 and len(ths) >= 2:
        heat = np.full((len(ths), len(g_offs)), np.nan)
        for r in rows:
            if r.get("g_off") in g_offs and r.get("theta_pir") in ths:
                heat[ths.index(r["theta_pir"]), g_offs.index(r["g_off"])] = r["ci_mean"]
        im = ax2.imshow(heat, aspect="auto", origin="lower",
                        extent=[math.log10(min(g_offs)), math.log10(max(g_offs)),
                                min(ths), max(ths)], cmap="RdYlBu", vmin=-0.2, vmax=0.8)
        ax2.set_xlabel("log10(g_off) [µA/cm² per ΔC/ms]")
        ax2.set_ylabel("θ_pir [ΔC/ms]")
        fig.colorbar(im, ax=ax2, label="CĪ")
        ax2.set_title("CĪ heatmap: g_off × θ_pir")
    else:
        th_sorted = sorted(rows, key=lambda r: (r.get("theta_pir", 0),
                                                r.get("t_total_ms", 0)))
        ax2.plot([r.get("theta_pir", 0) for r in th_sorted],
                 [r.get("ci_mean", float("nan")) for r in th_sorted],
                 "o-", color="#d62728")
        ax2.axhspan(TOL_LO, TOL_HI, color="green", alpha=0.15)
        ax2.set_xlabel("θ_pir [ΔC/ms]")
        ax2.set_ylabel("CĪ")
        ax2.set_title("CĪ vs θ_pir (scan)")
        ax2.grid(alpha=0.3)

    fig.suptitle("M4 步骤 4：CI 校准扫描（Brian2 闭环，N=10/点，确定性）",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------- #
# 子命令
# --------------------------------------------------------------------- #
def cmd_ref_scan():
    params = load_chemotaxis_params(CSV)
    grid = dict(
        theta_pir=[4e-6, 6e-6, 1e-5, 2e-5, 4e-5],
        alpha_deg=[60.0, 90.0, 120.0],
        v_fwd0=[0.5, 1.0, 1.5],
        sigma=[1.25, 1.875],
        tau_win=[100.0, 200.0],
        t_total=[10000.0, 15000.0, 25000.0],
    )
    t0 = time.time()
    rows = ref_scan(params, grid)
    print(f"ref-scan: {len(rows)} combos in {time.time()-t0:.0f}s")
    # 汇总：带内组合
    inband = [r for r in rows if r["in_tol"]]
    print(f"in-tolerance combos: {len(inband)}/{len(rows)}")
    # 选最小充分组合（D4：最小 T_total → 最小 v_fwd0 → 最大 θ_pir？）排序打印 top
    def key(r):
        return (r["t_total_ms"], r["v_fwd0"], -r["theta_pir"])
    for r in sorted(inband, key=key)[:25]:
        print(f"  T={r['t_total_ms']:.0f} θ_pir={r['theta_pir']:.1e} "
              f"α={r['alpha_deg']:.0f}° v_fwd0={r['v_fwd0']} σ={r['sigma']} "
              f"τ={r['tau_win_ms']:.0f} → CĪ={r['ci_mean']:.3f}±{r['ci_sem']:.3f} "
              f"(p={r['p_value']:.3f})")
    if not inband:
        best = min(rows, key=lambda r: min(abs(r["ci_mean"] - 0.3),
                                           abs(r["ci_mean"] - 0.7)))
        print(f"⚠ 无带内组合；最接近：CĪ={best['ci_mean']:.3f} @ "
              f"θ_pir={best['theta_pir']:.1e} α={best['alpha_deg']:.0f}° "
              f"v_fwd0={best['v_fwd0']} T={best['t_total_ms']:.0f}")


def cmd_probe():
    """ASER 链激活阈值探针：近食物起点背向运动 → SMDD 是否发放（机制 A 电路门）。"""
    params = load_chemotaxis_params(CSV)
    gs = [4.0e5, 1.0e6, 2.0e6, 4.0e6, 8.0e6]
    sx, sy = params.env.food_x - 0.7, params.env.food_y - 0.7
    away = math.atan2(params.env.food_y - sy, params.env.food_x - sx) + math.pi
    for g in gs:
        circ = ChemotaxisCircuit(csv_path=CSV)
        circ.set_ase_gains(g_on=g, g_off=g)
        env = make_env(circ.params)
        loop = ChemotaxisLoop(circ, env=env, seed=7)
        t0 = time.time()
        r = loop.run_trial(start_x=sx, start_y=sy, theta0=away,
                           t_total_ms=1500.0, seed=7)
        n_smdd = len(r.spikes("SMDDL", "node3")) + len(r.spikes("SMDDR", "node3"))
        print(f"g={g:.1e}: SMDD spikes={n_smdd} turn_events={r.meta['n_turn_events']} "
              f"CI={r.ci:.3f} ({time.time()-t0:.0f}s)")


def cmd_scan(points_json: str, n_workers: int = 1):
    points = json.loads(points_json)
    print(f"scan: {len(points)} points, n_workers={n_workers}")
    rows = run_scan(points, n_workers=n_workers)
    for r in rows:
        print(f"  #{r['point_id']}: g_off={r.get('g_off')} θ_pir={r.get('theta_pir')} "
              f"T={r['t_total_ms']} → CĪ={r['ci_mean']:.3f}±{r['ci_sem']:.3f}")
    png = plot_calibration(rows)
    print(f"calibration png: {png}")


def cmd_surrogate(i_thresh: float = 25.0):
    """阶段 2a：机制 A 替代模型 6 维网格扫描（秒级；候选再由 Brian2 验证）。"""
    params = load_chemotaxis_params(CSV)
    grid = dict(
        g_off=[2e6, 4e6, 8e6, 1.6e7],
        theta_pir=[4e-6, 8e-6, 1.5e-5],
        v_fwd0=[0.5, 1.0],
        sigma=[1.25, 1.875],
        tau_win=[100.0, 200.0],
        t_total=[10000.0, 15000.0, 25000.0],
    )
    t0 = time.time()
    rows = surrogate_scan(params, grid, i_thresh=i_thresh)
    print(f"surrogate-scan: {len(rows)} combos in {time.time()-t0:.0f}s "
          f"(I_thresh={i_thresh:.0f}µA/cm²)")
    inband = [r for r in rows if r["in_tol"]]
    print(f"in-tolerance: {len(inband)}/{len(rows)}")
    def key(r):
        return (r["t_total_ms"], r["v_fwd0"], r["g_off"], -r["theta_pir"])
    for r in sorted(inband, key=key)[:35]:
        print(f"  T={r['t_total_ms']:.0f} g_off={r['g_off']:.0e} "
              f"θ_pir={r['theta_pir']:.0e} v_fwd0={r['v_fwd0']} σ={r['sigma']} "
              f"τ={r['tau_win_ms']:.0f} → CĪ={r['ci_mean']:.3f}±{r['ci_sem']:.3f} "
              f"(p={r['p_value']:.3f} turns={r['n_turns']})")
    if not inband:
        best = min(rows, key=lambda r: min(abs(r["ci_mean"] - 0.3),
                                           abs(r["ci_mean"] - 0.7)))
        print(f"⚠ 无带内组合；最接近 CĪ={best['ci_mean']:.3f} @ "
              f"g_off={best['g_off']:.0e} θ_pir={best['theta_pir']:.0e} "
              f"v_fwd0={best['v_fwd0']} T={best['t_total_ms']:.0f}")
    return rows


def cmd_finalize(rows_path: str = CAL_CSV):
    """从 m4_calibration.csv 汇总 → 更新 CSV 定稿参数 + 出图。"""
    import csv as _csv

    rows = []
    if os.path.exists(rows_path):
        with open(rows_path, newline="", encoding="utf-8") as f:
            for r in _csv.DictReader(f):
                if r.get("ci_mean") != "":
                    r["ci_mean"] = float(r["ci_mean"])
                    r["ci_sem"] = float(r["ci_sem"])
                    r["ci_std"] = float(r["ci_std"])
                    r["p_value"] = float(r["p_value"])
                    r["cohen_d"] = float(r["cohen_d"])
                    r["t_total_ms"] = float(r["t_total_ms"])
                    rows.append(r)
    if rows:
        png = plot_calibration(rows)
        print(f"calibration png: {png} ({len(rows)} points)")
    else:
        print(f"no rows in {rows_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="M4 步骤 4 CI 校准")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ref-scan", help="阶段 1：行为参考模型 numpy 扫描")
    sub.add_parser("probe", help="ASER 链激活阈值探针")
    p_sur = sub.add_parser("surrogate", help="阶段 2a：机制 A 替代模型 6 维扫描")
    p_sur.add_argument("--i-thresh", type=float, default=25.0)
    p_scan = sub.add_parser("scan", help="阶段 2：Brian2 闭环扫描")
    p_scan.add_argument("--points", required=True, help="JSON 点列表")
    p_scan.add_argument("--workers", type=int, default=1)
    sub.add_parser("finalize", help="汇总 m4_calibration.csv → 出图")
    args = ap.parse_args()

    if args.cmd == "ref-scan":
        cmd_ref_scan()
    elif args.cmd == "probe":
        cmd_probe()
    elif args.cmd == "surrogate":
        cmd_surrogate(args.i_thresh)
    elif args.cmd == "scan":
        cmd_scan(args.points, args.workers)
    elif args.cmd == "finalize":
        cmd_finalize()
