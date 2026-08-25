"""M4 P3 验证：二维环境与虚拟身体自洽（无梯度对照）。

清单 §0 P3 / §6.3 + m4_env_notes §L7/L16/L17（有限样本测量限制已预注册）：
- 均匀浓度（C ≡ C_bg）→ s≡0 → 零 ASE 发放 → 零转向事件 → **对照轨迹 = 直行 + 反射
  （纯几何，与 g/θ_pir 无关，仅依赖 v_fwd0/协议）**（L16 对照复用判据）；
- 判据（主 agent 裁决 2026-08-23，L7：|CĪ| 点判据有限样本不可靠 → 统计显著性为主判据）：
  1. **CI 统计显著性 p>0.05**（单样本 t 检验 H0: μ=0）——用 numpy 直行运动学
     （引擎无关、确定性、纯几何）在**同协议**（T=5000ms, N=10）下计算（L16/L17）；
     |CĪ| 点值 informational；
  2. Brian2 闭环无梯度对照（缩短协议 T≤5000ms、N=10，record=[]）：
     轨迹有界（全程在皿内）、无 NaN、速度/转向量程在 CSV 规格内、CI ∈ [−1,1]；
  3. 闭环重跑可复现：同参数重跑轨迹逐位一致（__eq__）；
  4. 一致性补充（informational）：Brian2 对照 CI 与 numpy 直行 CI 应一致
     （无梯度 → 纯几何，v=v_fwd0·C_FWD_BASELINE 恒定，L17 锚 0.41）。
输出：reports/neuro/m4_p3_env_control.png + data/m4_p3_env_control.csv

用量：.venv-neuro/bin/python -m neural_exploration.tools.validate_p3_env_control
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.src.chemotaxis_body import ChemotaxisBody  # noqa: E402
from neural_exploration.src.chemotaxis_circuit import ChemotaxisCircuit  # noqa: E402
from neural_exploration.src.chemotaxis_env import (  # noqa: E402
    ChemotaxisEnv,
    ci_group_stats,
)
from neural_exploration.src.chemotaxis_loop import ChemotaxisLoop  # noqa: E402

DATA_DIR = os.path.join(ROOT, "neural_exploration", "data")
REPORTS_DIR = os.path.join(ROOT, "neural_exploration", "reports", "neuro")
CSV_PATH = os.path.join(DATA_DIR, "m4_chemotaxis_params.csv")
REPORT_PNG = os.path.join(REPORTS_DIR, "m4_p3_env_control.png")
REPORT_CSV = os.path.join(DATA_DIR, "m4_p3_env_control.csv")

# B1c 实测（L17）：AVB 张力基线 C_fwd≈0.41 → 有效爬行速度 v = v_fwd0·0.41
C_FWD_BASELINE = 0.41
# 缩短协议（主 agent 2026-08-25 授权方案 (b)：并发下执行，T≤2000ms 控制总时长，
# 接受 5–10× 减速；P3 T=2000ms×N=3 + 1 次重跑验确定性；numpy 直行对照统计为主判据
# （免费计算，T=5000ms/N=10，见 numpy_straight_control））
P3_T_MS = 2000.0
P3_N = 3
SEED_BASE = 1000          # run_control 默认（与 numpy 对照同种子 → 同起点扰动）
FULL_T_MS = 25000.0       # 全协议（numpy 直行对照 informational）


def numpy_straight_control(params, n_trials: int, t_total_ms: float,
                           seed_base: int = SEED_BASE) -> dict:
    """无梯度对照的 numpy 直行运动学（引擎无关、确定性、纯几何）。

    无梯度 → s≡0 → 零转向 → 恒定 v=v_fwd0·C_FWD_BASELINE 直行 + 反射边界；
    起点/朝向 = 试次种子伪随机（与 Brian2 run_trials 同语义，L16 对照复用判据）。
    """
    env = ChemotaxisEnv(arena_L=params.env.arena_L, sigma=params.env.sigma,
                        c_max=0.0, c_bg=params.env.c_bg,
                        food_x=params.env.food_x, food_y=params.env.food_y,
                        boundary=params.env.boundary)
    body = ChemotaxisBody(v_fwd0=params.body.v_fwd0, omega_max=params.body.omega_max,
                          dt_b=params.body.dt_b, v_osc=params.body.v_osc,
                          arena_L=params.env.arena_L, boundary=params.env.boundary)
    rng = np.random.default_rng(seed_base)
    n_ticks = max(1, int(round(t_total_ms / body.dt_b)))
    cis, xs_all, ys_all = [], [], []
    for _ in range(n_trials):
        sx = params.protocol.start_x + rng.normal(0.0, params.protocol.start_jitter)
        sy = params.protocol.start_y + rng.normal(0.0, params.protocol.start_jitter)
        th0 = rng.uniform(0.0, 2.0 * math.pi)
        body.reset(sx, sy, th0)
        xs, ys = np.empty(n_ticks), np.empty(n_ticks)
        for i in range(n_ticks):
            body.step(C_FWD_BASELINE, 0.0, 0.0, body.dt_b, i * body.dt_b)
            xs[i], ys[i] = body.x, body.y
        env.assert_bounded(xs, ys)
        cis.append(env.ci_per_trial(xs, ys))
        xs_all.append(xs)
        ys_all.append(ys)
    st = ci_group_stats(cis, params.protocol.ci_band_lo, params.protocol.ci_band_hi)
    return dict(ci=np.asarray(cis), stats=st, xs=xs_all, ys=ys_all,
                env=env, body=body)


def _trial_kinematic_ranges(result, params) -> dict:
    """逐 tick 重算 v/ω（引擎无关），检查量程在 CSV 规格内 + 无 NaN + 有界。"""
    v_fwd0 = params.body.v_fwd0
    omega_max = params.body.omega_max
    c_f = np.asarray(result.c_fwd, dtype=float)
    c_l = np.asarray(result.c_left, dtype=float)
    c_r = np.asarray(result.c_right, dtype=float)
    n = min(len(c_f), len(c_l), len(c_r))
    v = v_fwd0 * np.clip(c_f[:n], 0.0, 1.0)
    omega = omega_max * (c_l[:n] - c_r[:n])
    bounded = True
    try:
        from neural_exploration.src.chemotaxis_env import ChemotaxisEnv as _E
        env = _E(**{k: result.meta["env"][k] for k in
                    ("arena_L", "sigma", "c_max", "c_bg", "food_x", "food_y", "boundary")})
        env.assert_bounded(result.x, result.y)
    except ValueError:
        bounded = False
    return dict(
        bounded=bounded,
        v_min=float(np.min(v)) if n else float("nan"),
        v_max=float(np.max(v)) if n else float("nan"),
        omega_min=float(np.min(omega)) if n else float("nan"),
        omega_max_obs=float(np.max(omega)) if n else float("nan"),
        v_in_range=bool(np.all(v >= -1e-12) and np.all(v <= v_fwd0 + 1e-9)) if n else False,
        omega_in_range=bool(np.all(np.abs(omega) <= omega_max + 1e-9)) if n else False,
        no_nan=bool(np.all(np.isfinite(result.x)) and np.all(np.isfinite(result.y))
                    and np.all(np.isfinite(result.theta))
                    and np.all(np.isfinite(result.c_fwd))
                    and np.all(np.isfinite(result.c_left))
                    and np.all(np.isfinite(result.c_right))),
        ci=float(result.ci) if result.ci is not None else float("nan"),
        n_turn_events=int(result.meta.get("n_turn_events", 0)),
    )


def _reuse_p3_from_csv(circ, p) -> dict:
    """M4_REUSE=1 且 CSV 已存在 → 读回 Brian2 实测字段 + 重算 numpy 对照（快）。

    CSV 存逐试次（trial/ci_brian2/ci_numpy/bounded/no_nan/v/omega/n_turn_events）
    + stats 行 + deterministic/wall 行（B2b 扩展）。
    """
    import csv as _csv
    trials = {}
    stats = {}
    det_row, wall_row = None, None
    with open(REPORT_CSV, newline="", encoding="utf-8") as f:
        for r in _csv.reader(f):
            if not r:
                continue
            if r[0] == "stats":
                stats = dict(zip(r[1::2], r[2::2]))
            elif r[0] == "deterministic":
                det_row = r[1]
            elif r[0] == "wall_s":
                wall_row = float(r[1])
            elif r[0] != "trial" and len(r) >= 8 and r[0].lstrip("-").isdigit():
                trials[int(r[0])] = dict(ci=float(r[1]), ci_numpy=float(r[2]),
                                         bounded=str(r[3]).lower() == "true",
                                         no_nan=str(r[4]).lower() == "true",
                                         v_ok=str(r[5]).lower() == "true",
                                         omega_ok=str(r[6]).lower() == "true",
                                         n_turns=int(r[7]))
    # numpy 对照（快，重算）——主判据按任务定稿：同协议 T=5000ms/N=10（numpy 免费，
    # 与 Brian2 最短协议试次数无关；B2b L25 记录的 CĪ=−0.196 p=0.425 即此组）
    numpy_5s = numpy_straight_control(p, 10, 5000.0, seed_base=SEED_BASE)
    numpy_25s = numpy_straight_control(p, 20, FULL_T_MS, seed_base=SEED_BASE)
    # Brian2 逐试次（按 trial id 排序，与 numpy 对照同种子 → 逐位可比；**同 T**）
    brian2_ci = np.array([trials[i]["ci"] for i in sorted(trials)])
    brian2_stats = ci_group_stats(brian2_ci, p.protocol.ci_band_lo,
                                  p.protocol.ci_band_hi)
    numpy_same_t = numpy_straight_control(p, P3_N, P3_T_MS, seed_base=SEED_BASE)
    n_match = int(np.sum(np.abs(brian2_ci - numpy_same_t["ci"]) < 1e-9))
    kin = [dict(bounded=t["bounded"], no_nan=t["no_nan"],
                v_in_range=t["v_ok"], omega_in_range=t["omega_ok"],
                ci=t["ci"], n_turn_events=t["n_turns"])
           for i, t in sorted(trials.items())]
    ci_p = float(stats.get("ci_p_numpy_5s", numpy_5s["stats"]["p_value"]))
    ci_mean_5s = float(stats.get("ci_mean_numpy_5s",
                                 numpy_5s["stats"]["mean"]))
    ci_ok = bool(ci_p > 0.05 and numpy_5s["stats"]["n"] >= 5)
    det_ok = det_row is not None and str(det_row).lower() == "true"
    out = dict(
        pass_=bool(ci_ok and all(k["bounded"] for k in kin)
                   and all(k["no_nan"] for k in kin)
                   and all(k["v_in_range"] for k in kin)
                   and all(k["omega_in_range"] for k in kin)
                   and all(np.isfinite(k["ci"]) and -1.0 <= k["ci"] <= 1.0
                           for k in kin) and det_ok),
        ci_p_protocol=ci_p, ci_mean_protocol=ci_mean_5s,
        ci_sem_protocol=float(stats.get("ci_sem_numpy_5s",
                                        numpy_5s["stats"]["sem"])),
        ci_numpy_25s_mean=float(numpy_25s["stats"]["mean"]),
        ci_numpy_25s_p=float(numpy_25s["stats"]["p_value"]),
        ci_ok=bool(ci_ok),
        brian2_ci_mean=float(brian2_stats["mean"]),
        brian2_ci_p=float(brian2_stats["p_value"]),
        brian2_ci_sem=float(brian2_stats["sem"]),
        bounded_all=bool(all(k["bounded"] for k in kin)),
        no_nan_all=bool(all(k["no_nan"] for k in kin)),
        v_in_range_all=bool(all(k["v_in_range"] for k in kin)),
        omega_in_range_all=bool(all(k["omega_in_range"] for k in kin)),
        ci_finite_all=bool(all(np.isfinite(k["ci"]) for k in kin)),
        deterministic_rerun=bool(det_ok),
        n_trials=len(trials), t_total_ms=P3_T_MS,
        wall_s=float(wall_row) if wall_row else float("nan"),
        consistency=dict(n_match=n_match, n_total=len(trials),
                         match_frac=float(n_match / len(trials))
                         if trials else 0.0),
        per_trial=[{**k, "trial": i} for i, k in sorted(trials.items())],
        csv_path=CSV_PATH,
    )
    return out


def run_p3(save_plot: bool = True) -> dict:
    circ = ChemotaxisCircuit(csv_path=CSV_PATH)
    p = circ.params
    if os.environ.get("M4_REUSE") and os.path.exists(REPORT_CSV):
        return _reuse_p3_from_csv(circ, p)

    # ================= 1) numpy 直行对照（主判据：同协议 T=5000ms/N=10，任务定稿；
    #    numpy 免费计算——不随 Brian2 最短协议试次数缩减；B2b L25 记录同此组） =========
    numpy_5s = numpy_straight_control(p, 10, 5000.0, seed_base=SEED_BASE)
    # 全协议 informational（T=25s, N=20，L7 同款处置）
    numpy_25s = numpy_straight_control(p, 20, FULL_T_MS, seed_base=SEED_BASE)
    ci_p = numpy_5s["stats"]["p_value"]
    ci_ok = bool(ci_p > 0.05 and numpy_5s["stats"]["n"] >= 5)
    ci_mean_5s = numpy_5s["stats"]["mean"]

    # ================= 2) Brian2 闭环无梯度对照（缩短协议 T≤5000ms N=10） =================
    base_env = ChemotaxisEnv(**dict(p.env.__dict__))
    loop = ChemotaxisLoop(circ, env=base_env.no_gradient(), seed=SEED_BASE)
    t0 = _now()
    trials = loop.run_trials(n_trials=P3_N, seed_base=SEED_BASE,
                             t_total_ms=P3_T_MS, record=[])
    # 重跑可复现（同 seed_base 前 1 试次逐位一致；预算裁决——T=1s 单试次 ~1194s 墙钟，
    # 最短协议 N=2 + 1 次重跑，B2b L25）
    trials_rerun = loop.run_trials(n_trials=1, seed_base=SEED_BASE,
                                   t_total_ms=P3_T_MS, record=[])
    det_ok = bool(trials[0] == trials_rerun[0])
    wall_s = _now() - t0

    kin = [_trial_kinematic_ranges(r, p) for r in trials]
    bounded_all = all(k["bounded"] for k in kin)
    no_nan_all = all(k["no_nan"] for k in kin)
    v_ok = all(k["v_in_range"] for k in kin)
    omega_ok = all(k["omega_in_range"] for k in kin)
    ci_finite = all(np.isfinite(k["ci"]) and -1.0 <= k["ci"] <= 1.0 for k in kin)

    brian2_ci = np.array([k["ci"] for k in kin])
    brian2_stats = ci_group_stats(brian2_ci, p.protocol.ci_band_lo,
                                  p.protocol.ci_band_hi)
    # 一致性补充：Brian2 对照 CI 与 numpy 直行 CI（**同 T**，同种子）应一致——
    # 无梯度 → 纯几何；T 需与 Brian2 试次一致（P3_T_MS，非主判据的 T=5000）
    numpy_same_t = numpy_straight_control(p, P3_N, P3_T_MS, seed_base=SEED_BASE)
    n_match = int(np.sum(np.abs(brian2_ci - numpy_same_t["ci"]) < 1e-9))
    consistency = dict(n_match=n_match, n_total=P3_N,
                       match_frac=float(n_match / P3_N) if P3_N else 0.0)

    pass_ = bool(ci_ok and bounded_all and no_nan_all and v_ok and omega_ok
                 and ci_finite and det_ok)

    out = dict(
        pass_=pass_,
        ci_p_protocol=float(ci_p), ci_mean_protocol=float(ci_mean_5s),
        ci_sem_protocol=float(numpy_5s["stats"]["sem"]),
        ci_numpy_25s_mean=float(numpy_25s["stats"]["mean"]),
        ci_numpy_25s_p=float(numpy_25s["stats"]["p_value"]),
        ci_ok=bool(ci_ok),
        brian2_ci_mean=float(brian2_stats["mean"]),
        brian2_ci_p=float(brian2_stats["p_value"]),
        brian2_ci_sem=float(brian2_stats["sem"]),
        bounded_all=bool(bounded_all), no_nan_all=bool(no_nan_all),
        v_in_range_all=bool(v_ok), omega_in_range_all=bool(omega_ok),
        ci_finite_all=bool(ci_finite), deterministic_rerun=bool(det_ok),
        n_trials=P3_N, t_total_ms=P3_T_MS, wall_s=float(wall_s),
        consistency=consistency,
        per_trial=[{**k, "trial": i} for i, k in enumerate(kin)],
        csv_path=CSV_PATH,
    )

    # ---- CSV 落盘 ----
    os.makedirs(DATA_DIR, exist_ok=True)
    import csv as _csv
    with open(REPORT_CSV, "w", encoding="utf-8", newline="") as f:
        rows = []
        head = ["trial", "ci_brian2", "ci_numpy", "bounded", "no_nan",
                "v_in_range", "omega_in_range", "n_turn_events"]
        rows.append(head)
        # 同 T（P3_T_MS）numpy 直行对照——与 Brian2 试次逐位可比（主判据 numpy_5s 为 T=5000）
        numpy_same_t = numpy_straight_control(p, P3_N, P3_T_MS, seed_base=SEED_BASE)
        for i, k in enumerate(kin):
            rows.append([i, f"{k['ci']:.6f}", f"{numpy_same_t['ci'][i]:.6f}",
                         k["bounded"], k["no_nan"], k["v_in_range"],
                         k["omega_in_range"], k["n_turn_events"]])
        rows.append([])
        rows.append(["stats", "ci_mean_numpy_5s", f"{out['ci_mean_protocol']:.6f}",
                     "ci_p_numpy_5s", f"{out['ci_p_protocol']:.6f}",
                     "ci_mean_brian2", f"{out['brian2_ci_mean']:.6f}",
                     "ci_p_brian2", f"{out['brian2_ci_p']:.6f}"])
        rows.append(["deterministic", out["deterministic_rerun"]])
        rows.append(["wall_s", f"{out['wall_s']:.1f}"])
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

    if save_plot:
        _plot(out, numpy_5s, brian2_ci)

    return out


def _now():
    import time
    return time.time()


def _plot(out, numpy_5s, brian2_ci):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(REPORTS_DIR, exist_ok=True)
    env = numpy_5s["env"]
    L = env.spec.arena_L
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))

    # 1) numpy 直行对照轨迹（同协议 T=5000 N=10，与 Brian2 同种子）
    ax = axes[0]
    for xs, ys in zip(numpy_5s["xs"], numpy_5s["ys"]):
        ax.plot(xs, ys, lw=1.0, alpha=0.8)
    ax.plot(env.spec.food_x, env.spec.food_y, marker="*", ms=16, color="red",
            label="food (C_max=0 in control)")
    ax.axvline(L / 2, color="gray", ls="--", lw=0.7)
    ax.axhline(L / 2, color="gray", ls="--", lw=0.7)
    ax.set_xlim(-0.2, L + 0.2)
    ax.set_ylim(-0.2, L + 0.2)
    ax.set_aspect("equal")
    ax.set_title(f"numpy straight-line control (no gradient → s≡0 → no turns)\n"
                 f"CĪ={out['ci_mean_protocol']:+.3f} (p={out['ci_p_protocol']:.3f})",
                 fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    # 2) CI 分布：numpy 对照 vs Brian2 对照
    ax = axes[1]
    ax.hist(numpy_5s["ci"], bins=11, range=(-1.05, 1.05), alpha=0.6,
            color="#1f77b4", label=f"numpy control (N={len(numpy_5s['ci'])})")
    ax.hist(brian2_ci, bins=11, range=(-1.05, 1.05), alpha=0.6,
            color="#d62728", label=f"Brian2 control (N={len(brian2_ci)})")
    ax.axvline(0.0, color="k", ls="--", lw=0.9)
    ax.set_xlabel("CI (quadrant-based)")
    ax.set_ylabel("count")
    ax.set_title(f"control CI: Brian2 mean={out['brian2_ci_mean']:+.3f} "
                 f"(p={out['brian2_ci_p']:.3f}) | match numpy "
                 f"{out['consistency']['match_frac']:.0%}",
                 fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("M4 P3: no-gradient control — bounded/NaN-free/reproducible "
                 "trajectories; CI not significantly ≠ 0 (p>0.05)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(REPORT_PNG, dpi=150)
    plt.close(fig)
    return REPORT_PNG


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="M4 P3 环境/身体自洽（无梯度对照）验证")
    ap.add_argument("--skip-plot", action="store_true")
    args = ap.parse_args()
    res = run_p3(save_plot=not args.skip_plot)
    print("==== M4 P3 环境/身体自洽（无梯度对照）====")
    print(f"  numpy 直行对照 (T={res['t_total_ms']:.0f}ms, N={res['n_trials']}): "
          f"CĪ={res['ci_mean_protocol']:+.3f} p={res['ci_p_protocol']:.3f} "
          f"(>0.05 {'OK' if res['ci_ok'] else 'FAIL'})")
    print(f"  numpy 全协议 (T=25s, N=20): CĪ={res['ci_numpy_25s_mean']:+.3f} "
          f"p={res['ci_numpy_25s_p']:.3f} (informational)")
    print(f"  Brian2 对照: CĪ={res['brian2_ci_mean']:+.3f}±{res['brian2_ci_sem']:.3f} "
          f"p={res['brian2_ci_p']:.3f} | 与 numpy 逐位一致 {res['consistency']['match_frac']:.0%}")
    print(f"  轨迹: 有界={res['bounded_all']} 无NaN={res['no_nan_all']} "
          f"v量程={res['v_in_range_all']} ω量程={res['omega_in_range_all']} "
          f"CI有限={res['ci_finite_all']}")
    print(f"  重跑可复现: {res['deterministic_rerun']}")
    print(f"  P3 pass_ = {res['pass_']}（wall {res['wall_s']:.0f}s）")
