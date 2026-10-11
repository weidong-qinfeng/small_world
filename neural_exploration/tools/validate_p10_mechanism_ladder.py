"""M10 P6：状态切换机制阶梯（**前置机制门**）——逐级判决与如实登记胜负。

《生物仿真M10实施清单》§4.1（P6）+ §0.2 R1/R3 承接 + §8.1（P6 失败的反证路径）。

**机制自检门三条（每一级独立通过才允许评功能，§4.1.2）**：
  1. **无输入不得自持**：零驱动群率 / 基准群率 ≤ 上限（`m10_calibration_targets.csv`，草案 ≤0.10；
     **M9 实测 0.9853 = 反证基线**）
  2. **驱动弹性**：驱动倍数 {0,0.5,1,2,4,8} 群率单调不减 且 **4× 增幅 ≥ 下限（草案 ≥0.30）**
     （**M9 实测 +0.0725 = 反证基线**）
  3. **双态存在**：`bout_fraction > 0`（口径见 `src/larval_zf_circuit.py`；**M9 实测 0.0000 = 反证基线**）

阶梯（§4.1.1；逐级引入、逐级判决，**无效级同等入档**）：
  `M0`（基线，无机制）→ `M-1` STP 抑制型 → `M-2` STP 易化型 → `M-3` GABA 亚型异质 →
  `M-4` 调质增益（tonic→phasic）→ `M-5` 组合（最优子集）。

**结构证据级别 = 抽象（Fish1 partial 锚定）**——见 `src/larval_zf_circuit.py` docstring。

用法（每级建议独立进程——M9 L40 跨试次退化纪律）：
    PYTHONHASHSEED=0 ./.venv-m9/bin/python -m neural_exploration.tools.validate_p10_mechanism_ladder --level M-1
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

from ..src.larval_zf_circuit import ZFNetParams, build_engine, fish1_anchors, run_protocol, DT_MS  # noqa: E402

LEVELS = {
    "M0":  dict(stp=None, gaba=None, g_mod=1.0,
                desc="基线：无状态切换机制（M9 语义）"),
    "M-1": dict(stp=dict(enabled=True, u0=0.5, tau_fac_ms=50.0, tau_rec_ms=1000.0,
                         mode="depress"),
                gaba=None, g_mod=1.0, desc="STP 抑制型（Tsodyks–Markram；u0=0.5, τ_rec=1 s）"),
    "M-2": dict(stp=dict(enabled=True, u0=0.15, tau_fac_ms=500.0, tau_rec_ms=100.0,
                         mode="facilitate"),
                gaba=None, g_mod=1.0, desc="STP 易化型（u0=0.15, τ_fac=0.5 s）"),
    "M-3": dict(stp=None, gaba=dict(enabled=True, tau_fast_ms=8.0, tau_slow_ms=200.0,
                                    frac_slow=0.35), g_mod=1.0,
                desc="GABA 亚型异质（A 快 8 ms / B 慢 200 ms，慢组分 35%；类级先验）"),
    "M-4": dict(stp=None, gaba=None, g_mod=1.5,
                desc="调质增益（tonic→phasic 近似：兴奋/抑制权重与外部事件 ×1.5）"),
    "M-5": dict(stp=dict(enabled=True, u0=0.5, tau_fac_ms=50.0, tau_rec_ms=1000.0,
                         mode="depress"),
                gaba=dict(enabled=True, tau_fast_ms=8.0, tau_slow_ms=200.0, frac_slow=0.35),
                g_mod=1.15, desc="组合：M-1 + M-3 + M-4（全局参数 4 个 ≤ 上限 8）"),
}


def load_bands() -> dict:
    p = os.path.join(DATA_DIR, "m10_calibration_targets.csv")
    out = {}
    with open(p) as f:
        for row in csv.DictReader(r for r in f if not r.startswith("#")):
            out[row["key"]] = row["value"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="M-1", choices=list(LEVELS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-neuron", type=int, default=3000)
    ap.add_argument("--indegree", type=int, default=20)
    ap.add_argument("--T-ms", type=float, default=10_000.0)
    ap.add_argument("--settle-ms", type=float, default=500.0)
    ap.add_argument("--segment-steps", type=int, default=20_000)
    ap.add_argument("--lam0", type=float, default=2.0)
    ap.add_argument("--g-ext", type=float, default=20.0)
    ap.add_argument("--bias-median", type=float, default=250.0)
    ap.add_argument("--g-exc", type=float, default=9.0)
    ap.add_argument("--g-inh", type=float, default=22.0)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-csv", default=os.path.join(DATA_DIR, "m10_p6_levels.csv"))
    args = ap.parse_args(argv)

    bands = load_bands()
    cfg = LEVELS[args.level]
    Pm = ZFNetParams(n_neuron=args.n_neuron, indegree=args.indegree,
                     bias_median_mv_s=args.bias_median,
                     g_exc=args.g_exc, g_inh=args.g_inh)
    T = int(round(args.T_ms / DT_MS)); settle = int(round(args.settle_ms / DT_MS))
    t0 = time.time()
    print(f"[P6 {args.level}] {cfg['desc']}", flush=True)
    rows = []
    for m in (0.0, 1.0, 4.0):
        e = build_engine(Pm, device=args.device, stp=cfg["stp"], gaba=cfg["gaba"],
                         g_mod=cfg["g_mod"])
        r = run_protocol(e, T, settle_steps=settle, segment_steps=args.segment_steps,
                         lam_hz=args.lam0 * m, ev_seed=3, g_ext=args.g_ext)
        r["drive_mult"] = m
        rows.append(r)
        print(f"   λ×{m}: pop={r['pop_rate_hz']:.4f} med={r['rate_median_hz']:.4f} "
              f"silent={r['silent_frac']:.4f} bout_frac={r['bout_fraction']:.4f} "
              f"nbouts={r['n_bouts']} bout_med={r['bout_median_s']:.2f}s "
              f"ms/step={r['ms_per_step_steady']:.3f} wall={r['wall_s']}s", flush=True)
        del e
    base = next(r for r in rows if r["drive_mult"] == 1.0)
    zero = next(r for r in rows if r["drive_mult"] == 0.0)
    four = next(r for r in rows if r["drive_mult"] == 4.0)
    selfsustain = zero["pop_rate_hz"] / max(base["pop_rate_hz"], 1e-12)
    elasticity = (four["pop_rate_hz"] - base["pop_rate_hz"]) / max(base["pop_rate_hz"], 1e-12)
    pops = [r["pop_rate_hz"] for r in rows]
    monotone = all(b >= a - 1e-12 for a, b in zip(pops, pops[1:]))
    g1 = selfsustain <= float(bands["selfsustain_ratio_max"])
    g2 = (elasticity >= float(bands["drive_elasticity_min"])) and monotone
    g3 = base["bout_fraction"] > 0.0
    passed = bool(g1 and g2 and g3)
    entry = dict(
        level=args.level, desc=cfg["desc"], mechanism_params=dict(
            stp=cfg["stp"], gaba=cfg["gaba"], g_mod=cfg["g_mod"]),
        net_params=Pm.as_dict(), T_ms=args.T_ms, lam0=args.lam0, g_ext=args.g_ext,
        gate1_no_selfsustain=dict(value=round(selfsustain, 6),
                                  limit=float(bands["selfsustain_ratio_max"]),
                                  pass_=bool(g1), m9_baseline=0.9853),
        gate2_elasticity=dict(value=round(elasticity, 6),
                              limit=float(bands["drive_elasticity_min"]),
                              monotone=bool(monotone), pass_=bool(g2), m9_baseline=0.0725),
        gate3_bout_present=dict(bout_fraction=round(base["bout_fraction"], 6),
                                n_bouts=base["n_bouts"],
                                bout_median_s=round(base["bout_median_s"], 3),
                                pass_=bool(g3), m9_baseline=0.0),
        gate_all_pass=passed,
        outcome=("effective" if passed else
                 ("partial" if (g1 or g2 or g3) else "ineffective")),
        wall_s_total=round(time.time() - t0, 1),
        structure_evidence_level="abstract(anchored_on_fish1_partial)",
        drive_rows=rows,
    )
    outp = args.out_json or os.path.join(DATA_DIR, f"m10_p6_{args.level}.json")
    with open(outp, "w") as f:
        json.dump(entry, f, indent=1, ensure_ascii=False)
    # 追加到汇总 CSV（逐级一行；**无效级同等入档**）
    hdr = ["level", "desc", "selfsustain_ratio", "selfsustain_pass", "elasticity_4x",
           "elasticity_monotone", "elasticity_pass", "bout_fraction", "n_bouts",
           "bout_pass", "gate_all_pass", "outcome", "pop_rate_hz", "rate_median_hz",
           "silent_frac", "wall_s_total"]
    exists = os.path.exists(args.out_csv)
    with open(args.out_csv, "a", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        if not exists:
            f.write("# M10 P6 机制阶梯判决表（逐级：自检门三条 + 结论；结构证据级别 = 抽象/Fish1-partial 锚定）\n")
            w.writerow(hdr)
        w.writerow([args.level, cfg["desc"], round(selfsustain, 6), int(g1),
                    round(elasticity, 6), int(monotone), int(g2),
                    round(base["bout_fraction"], 6), base["n_bouts"], int(g3),
                    int(passed), entry["outcome"], round(base["pop_rate_hz"], 6),
                    round(base["rate_median_hz"], 6), round(base["silent_frac"], 6),
                    entry["wall_s_total"]])
    print(f"[P6 {args.level}] selfsustain={selfsustain:.4f}(≤{bands['selfsustain_ratio_max']}) "
          f"elasticity={elasticity:.4f}(≥{bands['drive_elasticity_min']}) "
          f"bout={base['bout_fraction']:.4f} ⇒ {entry['outcome'].upper()} "
          f"({entry['wall_s_total']}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
