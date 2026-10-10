"""M9 引擎诊断：`engine.run()` 的**分段可组合性**（外部事件在飞时跨调用边界是否等价）。

动机（P7 L38 实测发现）：同一协议、同一 seed、同一外部事件表，仅改变 `engine.run()` 的
**分段调度**（settle 1×6000 步 vs 3×2000 步），群率由 4.7680 Hz 变为 4.8867 Hz（差 2.5%）。
若成立，则 `engine.run()` 在**外部事件投递未清空**时**不是严格可组合的** → 所有分段执行的
协议数值（含 P4 T=10s 权威档）都带有"分段调度依赖"，必须作为**测量限制**登记。

设计（廉价、决定性）：固定网络/参数/事件表，仅比较分段调度：
  A0：1 × N 步                B0：2 × N/2 步
  A1：1 × N 步（重复 A0）      B1：4 × N/4 步
若 A0 == A1（确定性）而 A0 != B0（分段依赖）→ 结论成立。

输出：`data/m9_engine_segment_composability.csv`
用法：PYTHONHASHSEED=0 MPLBACKEND=Agg .venv-m9/bin/python -m neural_exploration.tools.diag_engine_segment_composability
"""

from __future__ import annotations

import csv as _csv
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUT_CSV = os.path.join(DATA_DIR, "m9_engine_segment_composability.csv")

N_STEPS = int(os.environ.get("M9_SEGCOMP_N", "4000"))
N_NEURONS_BG = 0.5      # 背景率（Hz）
SEED = 0


def main() -> int:
    from neural_exploration.src.adult_circuit import AdultCircuit, CircuitParams
    from neural_exploration.tools.validate_p9_resting import PARAMS

    print("=== M9 诊断：engine.run() 分段可组合性（外部事件在飞）===", flush=True)
    t0 = time.perf_counter()
    P = dict(PARAMS)
    c = AdultCircuit(device="mps", params=CircuitParams(**P), use_compile=False)
    c.build()
    g_ext = 2.0 / 0.104
    dt = P["dt_ms"]

    # 固定事件表（一次生成，全臂共用同一事件数组 → 排除采样差异）
    rng = np.random.default_rng(SEED + 4242)
    n_ev = rng.poisson(N_NEURONS_BG * N_STEPS * dt * 1e-3, size=c.n_neurons)
    total = int(n_ev.sum())
    neurons = np.repeat(np.arange(c.n_neurons, dtype=np.int64), n_ev)
    steps = np.sort(rng.integers(0, N_STEPS, size=total).astype(np.int64))
    amp = np.full(total, g_ext, dtype=np.float32)

    def run_split(k, tag):
        """把 N_STEPS 均分为 k 次 engine.run() 调用；返回逐步脉冲计数与总脉冲数。"""
        c.engine.set_external_events(steps, neurons, amp, N_STEPS)
        v0 = (P["v_rest"] + 2.0 * np.random.default_rng(1).standard_normal(c.n_neurons)
              ).astype(np.float32)
        c.engine.reset(v0=v0, seed=SEED)
        pops, done = [], 0
        seg = N_STEPS // k
        while done < N_STEPS:
            m = min(seg, N_STEPS - done)
            rr = c.engine.run(m, delivery="chunk", record="counts", pop_trace=True)
            pops.append(rr["pop"]); done += m
        pop = np.concatenate(pops)
        ctot = c.engine.t_count.detach().cpu().numpy()
        print("  [%s] k=%d 段 → 总脉冲 %d、pop_rate=%.4f Hz、%.1fs"
              % (tag, k, int(ctot.sum()), float(ctot.sum() / c.n_neurons / (N_STEPS * dt / 1000.0)),
                 time.perf_counter() - t0), flush=True)
        return pop, ctot

    popA0, cA0 = run_split(1, "A0")
    popA1, cA1 = run_split(1, "A1")          # 同配置复跑 → 确定性
    popB0, cB0 = run_split(2, "B0")
    popB1, cB1 = run_split(4, "B1")

    det = bool(np.array_equal(popA0, popA1))
    eq2 = bool(np.array_equal(popA0, popB0))
    eq4 = bool(np.array_equal(popA0, popB1))
    rel2 = float(np.abs(cB0.sum() - cA0.sum()) / max(cA0.sum(), 1))
    rel4 = float(np.abs(cB1.sum() - cA0.sum()) / max(cA0.sum(), 1))
    nz = int(np.sum(np.abs(cB0.astype(np.int64) - cA0.astype(np.int64)) > 0))

    rows = [["section", "key", "value", "note"],
            ["protocol", "n_steps", N_STEPS, "诊断窗步数"],
            ["protocol", "delivery", "chunk", "W=min(SLOT_EXTRA,max_delay)"],
            ["protocol", "ext_delay_steps", int(c.engine.max_delay), "外部事件延迟槽"],
            ["protocol", "n_ext_events", total, "固定事件表（全臂共用）"],
            ["measure", "spikes_k1", int(cA0.sum()), "1 次调用"],
            ["measure", "spikes_k2", int(cB0.sum()), "2 次调用"],
            ["measure", "spikes_k4", int(cB1.sum()), "4 次调用"],
            ["measure", "rel_diff_k2", "%.6f" % rel2, "|k2−k1|/k1"],
            ["measure", "rel_diff_k4", "%.6f" % rel4, "|k4−k1|/k1"],
            ["measure", "n_neurons_differ_k2", nz, "逐神经元计数不同的神经元数"],
            ["crit", "determinism_same_split", str(det), "A0 vs A1 逐位一致"],
            ["crit", "composable_k2", str(eq2), "分段与单次逐位等价"],
            ["crit", "composable_k4", str(eq4), "分段与单次逐位等价"],
            ["crit", "verdict", "NON_COMPOSABLE" if not (eq2 and eq4) else "COMPOSABLE",
             "外部事件在飞时分段调度是否影响结果"],
            ["measure", "wall_s", "%.1f" % (time.perf_counter() - t0), ""]]
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        f.write("# M9 引擎诊断：engine.run() 分段可组合性（外部事件在飞；判据带无关）\n")
        _csv.writer(f, lineterminator="\n").writerows(rows)
    print("→", OUT_CSV, flush=True)
    print("确定性(k1 复跑)=%s；k2 等价=%s；k4 等价=%s；相对差 k2=%.4f k4=%.4f"
          % (det, eq2, eq4, rel2, rel4), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
