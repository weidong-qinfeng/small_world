
import sys, os, time, json
sys.path.insert(0, '/Users/weidong/ai/small_world')
import numpy as np
from neural_exploration.src.chemotaxis_circuit import ChemotaxisCircuit
from neural_exploration.src.chemotaxis_env import ChemotaxisEnv
from neural_exploration.src.chemotaxis_loop import ChemotaxisLoop
CSV = '/Users/weidong/ai/small_world/neural_exploration/data/m4_chemotaxis_params.csv'
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
