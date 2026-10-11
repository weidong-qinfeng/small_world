#!/usr/bin/env python
"""P4 会话构建分阶段计时探针（诊断 make_session 慢点；非交付物）。"""
import csv
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DATA_DIR = os.path.join(ROOT, "neural_exploration", "data")


def load_weight_rows():
    out = {}
    with open(os.path.join(DATA_DIR, "m8_larva_params.csv"), newline="",
              encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            fields = next(csv.reader([s]))
            if len(fields) < 11 or fields[0].strip().lower() != "weight":
                continue
            try:
                out[fields[1].strip()] = float(fields[9])
            except ValueError:
                continue
    return out


def main():
    w = load_weight_rows()
    gmax = float(w["gmax_scale"])
    cs = {}
    for k, v in w.items():
        if k.startswith("class_scale_"):
            p2 = k.split("_")
            if len(p2) == 4:
                cs[(p2[2], p2[3])] = v
    from neural_exploration.src.larva_circuit import LarvaCircuit
    circ = LarvaCircuit(scale=300, fidelity="two_comp", seed=0,
                        nt_fallback="class", provisional_muscles=True,
                        gmax_scale=gmax, class_scales=cs, plasticity="none")

    for trial in range(3):
        t0 = time.perf_counter()
        sess = circ.make_session(t_total_ms=30000.0)
        t1 = time.perf_counter()
        sess.reset(seed=trial)
        t2 = time.perf_counter()
        # 跑 40 个 epoch（≈1s 网络时间）计时
        n_ep = 40
        for _ in range(n_ep):
            sess.run_epoch(circ.dt_b, 0.0)
        t3 = time.perf_counter()
        print(f"trial {trial}: make_session={t1-t0:.1f}s reset={t2-t1:.1f}s "
              f"{n_ep}epoch={t3-t2:.1f}s "
              f"(net.t={float(sess.net.t):.0f})", flush=True)


if __name__ == "__main__":
    main()
