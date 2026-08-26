"""临时探针：测闭环单试次耗时 + 梯度下电路行为（ASER/SMDD 发放、s 范围、有效速度）。

不落盘、不提交；仅 B1c 校准预算依据。
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from neural_exploration.src.chemotaxis_circuit import ChemotaxisCircuit
from neural_exploration.src.chemotaxis_env import ChemotaxisEnv, TimeDiffTracker
from neural_exploration.src.chemotaxis_loop import ChemotaxisLoop

CSV = os.path.join(ROOT, "neural_exploration", "data", "m4_chemotaxis_params.csv")
T_TOTAL_MS = 2000.0


def probe(T=T_TOTAL_MS, seed=0):
    circ = ChemotaxisCircuit(csv_path=CSV)
    loop = ChemotaxisLoop(circ, seed=seed)
    t0 = time.time()
    r = loop.run_trial(t_total_ms=T, seed=seed)
    dt = time.time() - t0
    print(f"T={T:.0f}ms: wall={dt:.1f}s  ({dt / T * 1000:.2f} s per 1000ms)")
    print(f"  CI={r.ci:.3f}  start=({r.meta['start_x']:.2f},{r.meta['start_y']:.2f}) theta0={r.meta['theta0']:.2f}")
    print(f"  dist_end_food={r.meta['dist_end_food']:.2f}  dist_start_food={r.meta['dist_start_food']:.2f}")
    s_trace = r.meta["c_sensed"]
    # 重算 s（TimeDiffTracker 语义近似：tick 级差分）
    tau = circ.params.transduction.tau_win_ms
    dtb = circ.params.body.dt_b
    k = max(1, int(round(tau / dtb)))
    c = s_trace
    cp = np.concatenate([np.full(k, c[0]), c])
    s = (c - cp[:len(c)]) / tau
    print(f"  s: min={s.min():.2e} max={s.max():.2e}  (tau={tau}ms)")
    n_aser = len(r.spikes("ASER", "node3"))
    n_smddl = len(r.spikes("SMDDL", "node3"))
    n_smddr = len(r.spikes("SMDDR", "node3"))
    n_asel = len(r.spikes("ASEL", "node3"))
    print(f"  spikes: ASEL={n_asel} ASER={n_aser} SMDDL={n_smddl} SMDDR={n_smddr}")
    print(f"  muscle: C_fwd mean={r.c_fwd.mean():.4f} max={r.c_fwd.max():.3f} | "
          f"C_left mean={r.c_left.mean():.4f} C_right mean={r.c_right.mean():.4f}")
    print(f"  x range=[{r.x.min():.2f},{r.x.max():.2f}] y range=[{r.y.min():.2f},{r.y.max():.2f}]")
    return r


if __name__ == "__main__":
    probe()
