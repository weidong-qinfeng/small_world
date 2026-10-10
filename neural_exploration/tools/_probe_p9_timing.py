"""P9 前置探针：全规模回路在 MPS / CPU 上的构建墙钟与 ms/step（选设备 + 预算）。

纪律：**先探后跑**（血泪教训 8：共享高负载环境先探针测墙钟）。
用法：PYTHONHASHSEED=0 ./.venv-m9/bin/python -m neural_exploration.tools._probe_p9_timing
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

DEMO_STEPS = int(os.environ.get("M9_PROBE_STEPS", "4000"))


def probe(device: str) -> dict:
    from neural_exploration.src.adult_circuit import AdultCircuit, CircuitParams
    from neural_exploration.tools.validate_p9_resting import PARAMS, BG
    t0 = time.perf_counter()
    c = AdultCircuit(device=device, params=CircuitParams(**PARAMS), verbose=True,
                     use_compile=False)
    c.build()
    t_build = time.perf_counter() - t0
    g_ext = BG["epsp_mv"] / 0.104
    n_steps = DEMO_STEPS + 200
    c.build_background(rate_hz=BG["rate_hz"], g_ext=g_ext, n_steps=n_steps,
                       seed=BG["seed"])
    v0 = np.full(c.n_neurons, -52.0, dtype=np.float32)
    c.engine.reset(v0=v0, seed=0)
    # 预热（编译/分配器池）
    c.engine.run(200, delivery="chunk", record="counts")
    c.engine.reset_counts()
    ts = []
    done = 0
    while done < DEMO_STEPS:
        m = min(2000, DEMO_STEPS - done)
        tt = time.perf_counter()
        c.engine.run(m, delivery="chunk", record="counts", pop_trace=True)
        ts.append((time.perf_counter() - tt) / m * 1e3)
        done += m
    st = c.engine.firing_stats(DEMO_STEPS * c.params.dt_ms)
    seg_ms = float(np.median(ts))
    return {"device": device, "build_wall_s": t_build, "ms_per_step_median": seg_ms,
            "ms_per_step_max": float(np.max(ts)), "pop_rate_hz": st["pop_rate_hz"],
            "rate_median_hz": st["rate_median_hz"], "silent_frac": st["silent_frac"],
            "n_segments": len(ts)}


def main() -> int:
    devs = os.environ.get("M9_PROBE_DEVICES", "cpu,mps").split(",")
    for d in devs:
        try:
            r = probe(d.strip())
        except Exception as e:  # noqa: BLE001
            print("probe %s FAILED: %r" % (d, e), flush=True)
            continue
        print("PROBE %s: build %.1fs  ms/step 中位 %.3f (max %.3f)  pop %.3f Hz  "
              "median %.3f Hz  silent %.3f"
              % (r["device"], r["build_wall_s"], r["ms_per_step_median"],
                 r["ms_per_step_max"], r["pop_rate_hz"], r["rate_median_hz"],
                 r["silent_frac"]), flush=True)
        # 单次 1000ms 测量窗（+300ms settle）预算
        print("   → 预测 T=1000ms/settle=300ms 单次运行墙钟 ≈ %.0fs"
              % (r["ms_per_step_median"] * 26000 / 1e3), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
