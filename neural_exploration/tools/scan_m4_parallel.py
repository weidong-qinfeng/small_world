"""M4 步骤 4：Brian2 闭环并行扫描协调器（B1c 执行节点）。

清单《生物仿真M4实施清单》§5（校准协议）：
  - 网格：g_ON/g_OFF、τ_win、v_fwd0、ω_max、σ、T_total（或机制 A 关键参数），
    每点 N=10 试次、确定性、无梯度对照并行；
  - 判据：有梯度 CI ∈ [0.3,0.7]（容差 [0.25,0.75]）且 无梯度对照 p>0.05；
    选最小充分组合（D4 原则）。

实现要点（B1c 实测，m4_env_notes §L16）：
  - 每试次 = 独立进程（record=[] 关闭 V 轨迹监视 → 84s/1000ms，25s 试次 ≈ 35min）；
    试次间确定性（seed=seed_base+trial）+ 机制 A 转向方向=试次种子伪随机；
  - 无梯度对照（C≡C_bg → s≡0 → 零 ASE 发放 → 零转向事件）只依赖 (v_fwd0, 协议)，
    与电路参数无关 → 由 numpy 直行运动学一次算定（每 v_fwd0），最终点真实 Brian2 复核；
  - 产出：data/m4_calibration.csv（每点 CĪ±SEM/对照/参数）+ reports/neuro/m4_calibration.png。

用法：
  .venv-neuro/bin/python -m neural_exploration.tools.scan_m4_parallel \
      --points 'json' --workers 9 --n-trials 10
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.src.chemotaxis_circuit import load_chemotaxis_params  # noqa: E402
from neural_exploration.src.chemotaxis_env import (  # noqa: E402
    ChemotaxisEnv,
    ci_group_stats,
)
from neural_exploration.tools.calibrate_m4_chemotaxis import (  # noqa: E402
    CAL_CSV,
    CAL_PNG,
    TOL_LO,
    TOL_HI,
    make_body,
    plot_calibration,
    ref_control,
    write_cal_csv,
)

PY = os.path.join(ROOT, ".venv-neuro", "bin", "python")
WORKER = os.path.join(ROOT, "neural_exploration", "tools", "_m4_trial_worker.py")


def _worker_script():
    """内联 trial worker（子进程隔离；避免仓库额外文件）。"""
    return r"""
import sys, os, time, json
sys.path.insert(0, %(root)r)
import numpy as np
from neural_exploration.src.chemotaxis_circuit import ChemotaxisCircuit
from neural_exploration.src.chemotaxis_env import ChemotaxisEnv
from neural_exploration.src.chemotaxis_loop import ChemotaxisLoop
CSV = %(csv)r
g = float(sys.argv[1]); theta_pir = float(sys.argv[2]); v_fwd0 = float(sys.argv[3])
T = float(sys.argv[4]); seed = int(sys.argv[5])
t_pir_ms = float(sys.argv[6]); omega_pir = float(sys.argv[7])
sigma = float(sys.argv[8]) if len(sys.argv) > 8 else None
tau_win = float(sys.argv[9]) if len(sys.argv) > 9 else None
t0 = time.time()
circ = ChemotaxisCircuit(csv_path=CSV)
circ.set_ase_gains(g_on=g, g_off=g)
circ.set_mechanism_a(theta_pir=theta_pir, omega_pir=omega_pir, t_pir_ms=t_pir_ms)
circ.params.body.v_fwd0 = v_fwd0
if sigma is not None: circ.params.env.sigma = sigma
if tau_win is not None: circ.params.transduction.tau_win_ms = tau_win
env = ChemotaxisEnv(**dict(circ.params.env.__dict__))
loop = ChemotaxisLoop(circ, env=env, seed=seed)
r = loop.run_trials(n_trials=1, seed_base=seed, t_total_ms=T, record=[])
out = dict(g=g, theta_pir=theta_pir, v_fwd0=v_fwd0, t_total_ms=T, seed=seed,
           t_pir_ms=t_pir_ms, omega_pir=omega_pir, sigma=circ.params.env.sigma,
           tau_win_ms=circ.params.transduction.tau_win_ms,
           ci=float(r[0].ci), n_turn_events=int(r[0].meta["n_turn_events"]),
           wall_s=round(time.time()-t0, 1),
           dist_end_food=float(r[0].meta["dist_end_food"]))
print("TRIAL " + json.dumps(out), flush=True)
""" % {"root": ROOT, "csv": os.path.join(ROOT, "neural_exploration", "data",
                                        "m4_chemotaxis_params.csv")}


def _write_worker():
    with open(WORKER, "w", encoding="utf-8") as f:
        f.write(_worker_script())
    return WORKER


def run_point(point: dict, n_trials: int, n_workers: int, seed_base: int = 0) -> dict:
    """一个校准点：n_trials 个试次，n_workers 个并发 worker（有界轮转进程池）。

    实测（B1c，m4_env_notes §L16）：3+ 个并发 Brian2 进程会因 cython 缓存锁竞争
    使单试次耗时恶化 → 并发数限制在 2–3（每试次独立进程，确定性 seed=seed_base+trial）。
    """
    g = point["g_off"]
    th = point.get("theta_pir", 4.0e-6)
    v = point.get("v_fwd0", 0.5)
    T = point.get("t_total_ms", 25000.0)
    t_pir = point.get("t_pir_ms", 1571.0)
    om = point.get("omega_pir", 1.0)
    sigma = point.get("sigma")
    tau_win = point.get("tau_win_ms")

    def _args(trial):
        args = [PY, WORKER, str(g), str(th), str(v), str(T), str(seed_base + trial),
                str(t_pir), str(om)]
        if sigma is not None:
            args.append(str(sigma))
        if tau_win is not None:
            args.append(str(tau_win))
        return args

    cis, turns, walls, d_ends = [], [], [], []
    t0 = time.time()
    done = 0
    next_trial = 0
    procs = {}   # Popen → trial index
    while next_trial < n_trials or procs:
        while next_trial < n_trials and len(procs) < max(1, n_workers):
            trial = next_trial
            p = subprocess.Popen(_args(trial), stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL, text=True, cwd=ROOT)
            procs[p] = trial
            next_trial += 1
        finished = [p for p in procs if p.poll() is not None]
        if not finished:
            time.sleep(10)
            continue
        for p in finished:
            trial = procs.pop(p)
            out, _ = p.communicate()
            for line in out.splitlines():
                if line.startswith("TRIAL "):
                    d = json.loads(line[len("TRIAL "):])
                    cis.append(d["ci"])
                    turns.append(d["n_turn_events"])
                    walls.append(d["wall_s"])
                    d_ends.append(d["dist_end_food"])
            done += 1
            if done % 5 == 0:
                print(f"  ...{done}/{n_trials} trials done "
                      f"({time.time()-t0:.0f}s)", flush=True)
    st = ci_group_stats(cis, TOL_LO, TOL_HI)
    return dict(
        params=point, ci_values=cis, ci_mean=st["mean"], ci_sem=st["sem"],
        ci_std=st["std"], p_value=st["p_value"], cohen_d=st["cohen_d"],
        n_turns=turns, wall_s=sum(walls), wall_max_s=max(walls) if walls else 0,
        dist_end_food=np.mean(d_ends) if d_ends else float("nan"),
    )


def main():
    ap = argparse.ArgumentParser(description="M4 步骤 4 Brian2 并行扫描")
    ap.add_argument("--points", required=True, help="JSON 点列表")
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--n-trials", type=int, default=10)
    args = ap.parse_args()

    points = json.loads(args.points)
    _write_worker()
    params = load_chemotaxis_params()

    # 无梯度对照（numpy 直行运动学；只依赖 v_fwd0）——先算一次
    controls = {}
    for pt in points:
        v = pt.get("v_fwd0", 0.5)
        if v not in controls:
            ctrl = ref_control(params, v, pt.get("sigma", params.env.sigma),
                               pt.get("tau_win_ms", params.transduction.tau_win_ms),
                               pt.get("t_total_ms", params.protocol.t_total_ms),
                               n=20)
            ctrl_stats = ci_group_stats(ctrl["ci"], TOL_LO, TOL_HI)
            controls[v] = dict(mean=ctrl_stats["mean"], p=ctrl_stats["p_value"],
                               sem=ctrl_stats["sem"], values=ctrl["ci"].tolist())
            print(f"[control] v_fwd0={v}: CĪ={ctrl_stats['mean']:.3f} "
                  f"p={ctrl_stats['p_value']:.3f}", flush=True)

    rows = []
    for idx, pt in enumerate(points):
        t0 = time.time()
        res = run_point(pt, args.n_trials, args.workers)
        ctrl = controls[pt.get("v_fwd0", 0.5)]
        in_tol = TOL_LO <= res["ci_mean"] <= TOL_HI
        ctrl_ok = ctrl["p"] > 0.05
        row = dict(
            point_id=idx,
            g_on=pt.get("g_on", pt["g_off"]), g_off=pt["g_off"],
            tau_win_ms=pt.get("tau_win_ms", params.transduction.tau_win_ms),
            v_fwd0=pt.get("v_fwd0", 0.5),
            omega_max=pt.get("omega_max", params.body.omega_max),
            sigma=pt.get("sigma", params.env.sigma),
            t_total_ms=pt.get("t_total_ms", 25000.0),
            theta_pir=pt.get("theta_pir", 4.0e-6),
            omega_pir=pt.get("omega_pir", 1.0),
            t_pir_ms=pt.get("t_pir_ms", 1571.0),
            ci_mean=res["ci_mean"], ci_sem=res["ci_sem"], ci_std=res["ci_std"],
            p_value=res["p_value"], cohen_d=res["cohen_d"],
            ci_values=json.dumps([round(c, 4) for c in res["ci_values"]]),
            n_turns=json.dumps(res["n_turns"]),
            wall_s=round(res["wall_s"], 1),
            ctrl_mean=ctrl["mean"], ctrl_p=ctrl["p"], ctrl_sem=ctrl["sem"],
            pass_tol=in_tol, pass_ctrl=ctrl_ok,
            note=("PASS" if (in_tol and ctrl_ok) else
                  ("band-only" if in_tol else "out-of-band")),
        )
        rows.append(row)
        write_cal_csv(rows)
        print(f"[point {idx}] g_off={pt['g_off']:.0e} θ_pir={pt.get('theta_pir',4e-6):.0e} "
              f"v={pt.get('v_fwd0',0.5)} T={pt.get('t_total_ms',25000):.0f}: "
              f"CĪ={res['ci_mean']:.3f}±{res['ci_sem']:.3f} (p={res['p_value']:.3f}) "
              f"turns={res['n_turns']} ctrl_p={ctrl['p']:.3f} → {row['note']} "
              f"({time.time()-t0:.0f}s)", flush=True)

    png = plot_calibration(rows)
    print(f"calibration csv: {CAL_CSV}")
    print(f"calibration png: {png}")
    n_pass = sum(1 for r in rows if r["pass_tol"] and r["pass_ctrl"])
    print(f"pass: {n_pass}/{len(rows)}")


if __name__ == "__main__":
    main()
