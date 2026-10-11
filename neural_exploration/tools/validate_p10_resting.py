"""M10 P5：静息标定与自持性（**G1 前置门**）验证。

《生物仿真M10实施清单》§4.2（P5 判据 (a)–(g)）+ §0.5 G1 门 + §8.1（失败排查顺序）。

**M9 → M10 直接承接**：M9 实测"活动自主自持"（零背景驱动仍保留 **98.5%** 群率）+
"输入通路增益过低"（4× 驱动仅 **+7.2%**）→ 本工具把这两项升为**硬判据**：
  (c) **输入关闭不得自持**：零驱动群率 / 基准群率 ≤ 上限（`m10_calibration_targets.csv`）
  (d) **驱动弹性**：驱动倍数 {0,0.5,1,2,4,8} 群率单调不减 + 4× 增幅 ≥ 下限

**判据带全部在 `data/m10_calibration_targets.csv` 运行前定稿**（本工具只读，不改带）。
**结构证据级别 = 抽象（Fish1 partial 锚定）**，见 `src/larval_zf_circuit.py` docstring。

用法：
    PYTHONHASHSEED=0 ./.venv-m9/bin/python -m neural_exploration.tools.validate_p10_resting
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
REP_DIR = os.path.join(PROJECT_ROOT, "reports", "neuro")

from ..src.larval_zf_circuit import (ZFNetParams, build_engine, fish1_anchors,  # noqa: E402
                                     run_protocol, DT_MS)


def load_bands() -> dict:
    """读判据带（唯一定稿 CSV）。缺失即报错——不做默认值静默回退。"""
    p = os.path.join(DATA_DIR, "m10_calibration_targets.csv")
    out = {}
    with open(p) as f:
        for row in csv.DictReader(r for r in f if not r.startswith("#")):
            out[row["key"]] = row["value"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-neuron", type=int, default=3000)
    ap.add_argument("--indegree", type=int, default=20)
    ap.add_argument("--T-ms", type=float, default=10_000.0)
    ap.add_argument("--settle-ms", type=float, default=500.0)
    ap.add_argument("--segment-steps", type=int, default=20_000)
    ap.add_argument("--lam0", type=float, default=2.0,
                    help="基准背景驱动率（逐神经元 Hz）")
    ap.add_argument("--g-ext", type=float, default=20.0)
    ap.add_argument("--bias-median", type=float, default=250.0)
    ap.add_argument("--g-exc", type=float, default=9.0)
    ap.add_argument("--g-inh", type=float, default=22.0)
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "m10_p5_resting.csv"))
    ap.add_argument("--out-drive", default=os.path.join(DATA_DIR, "m10_p5_drive_sensitivity.csv"))
    ap.add_argument("--out-json", default=os.path.join(DATA_DIR, "m10_p5_resting.json"))
    args = ap.parse_args(argv)

    bands = load_bands()
    t0 = time.time()
    Pm = ZFNetParams(n_neuron=args.n_neuron, indegree=args.indegree,
                     g_exc=args.g_exc, g_inh=args.g_inh,
                     bias_median_mv_s=args.bias_median)
    print(f"[P5] 网络参数 {Pm.as_dict()}  bands={ {k: bands[k] for k in list(bands)[:4]} }",
          flush=True)
    T = int(round(args.T_ms / DT_MS)); settle = int(round(args.settle_ms / DT_MS))

    # ---------------- 驱动剂量-响应（(d) 驱动弹性）----------------
    mult = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    rows = []
    for m in mult:
        e = build_engine(Pm, device=args.device)
        r = run_protocol(e, T, settle_steps=settle, segment_steps=args.segment_steps,
                         lam_hz=args.lam0 * m, ev_seed=3, g_ext=args.g_ext)
        r["drive_mult"] = m
        rows.append(r)
        print(f"  λ={args.lam0*m:.3f} Hz ({m}×): pop={r['pop_rate_hz']:.4f} Hz "
              f"med={r['rate_median_hz']:.4f} silent={r['silent_frac']:.4f} "
              f"bout_frac={r['bout_fraction']:.4f} nbouts={r['n_bouts']} "
              f"ms/step={r['ms_per_step_steady']:.3f} wall={r['wall_s']}s", flush=True)
        del e

    base = next(r for r in rows if r["drive_mult"] == 1.0)
    zero = next(r for r in rows if r["drive_mult"] == 0.0)
    four = next(r for r in rows if r["drive_mult"] == 4.0)
    selfsustain = zero["pop_rate_hz"] / max(base["pop_rate_hz"], 1e-12)
    elasticity = (four["pop_rate_hz"] - base["pop_rate_hz"]) / max(base["pop_rate_hz"], 1e-12)
    pops = [r["pop_rate_hz"] for r in rows]
    monotone = all(b >= a - 1e-12 for a, b in zip(pops, pops[1:]))
    elastic_ok = row_ok = None
    ok_ss = selfsustain <= float(bands["selfsustain_ratio_max"])
    ok_el = (elasticity >= float(bands["drive_elasticity_min"])) and monotone
    ok_rate = float(bands["rate_band_lo"]) <= base["rate_median_hz"] <= float(bands["rate_band_hi"])
    ok_sil = float(bands["silent_band_lo"]) <= base["silent_frac"] <= float(bands["silent_band_hi"])
    ok_bout = base["bout_fraction"] > 0.0
    ok_nan = base["n_nan"] == 0

    verdict = dict(
        p5_c_selfsustain=dict(value=selfsustain, limit=float(bands["selfsustain_ratio_max"]),
                              pass_=bool(ok_ss),
                              m9_baseline=0.9853),
        p5_d_elasticity=dict(value=elasticity, limit=float(bands["drive_elasticity_min"]),
                             monotone=bool(monotone), pass_=bool(ok_el),
                             m9_baseline=0.0725),
        p5_a_rate_median=dict(value=base["rate_median_hz"],
                              band=[float(bands["rate_band_lo"]), float(bands["rate_band_hi"])],
                              pass_=bool(ok_rate),
                              imaging_note=bands.get("imaging_silent_frac_k3mad", "")),
        p5_b_silent=dict(value=base["silent_frac"],
                         band=[float(bands["silent_band_lo"]), float(bands["silent_band_hi"])],
                         pass_=bool(ok_sil)),
        p5_bout_present=dict(value=base["bout_fraction"], pass_=bool(ok_bout), m9_baseline=0.0),
        p5_e_bounded=dict(nan=base["n_nan"], pass_=bool(ok_nan)),
        p5_g_determinism=None,
    )
    # (g) 确定性：同参数重跑（统计级；逐位见 §2.3(e)）
    e = build_engine(Pm, device=args.device)
    r1 = run_protocol(e, min(T, 40_000), settle_steps=settle,
                      segment_steps=args.segment_steps, lam_hz=args.lam0, ev_seed=3,
                      g_ext=args.g_ext)
    del e
    e = build_engine(Pm, device=args.device)
    r2 = run_protocol(e, min(T, 40_000), settle_steps=settle,
                      segment_steps=args.segment_steps, lam_hz=args.lam0, ev_seed=3,
                      g_ext=args.g_ext)
    del e
    det_equal = bool(r1["spk_total"] == r2["spk_total"])
    verdict["p5_g_determinism"] = dict(run1_spk=r1["spk_total"], run2_spk=r2["spk_total"],
                                       bit_identical=det_equal)

    G1 = bool(ok_ss and ok_el)
    out = dict(generated_by="tools/validate_p10_resting.py",
               generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
               structure_evidence_level="abstract(anchored_on_fish1_partial)",
               fish1_anchors=fish1_anchors(),
               net_params=Pm.as_dict(), bands_used=bands,
               drive_sweep=rows, verdict=verdict,
               G1_gate=dict(pass_=G1,
                            rule="(c) 自持比 ≤ 上限 ∧ (d) 4× 驱动增幅 ≥ 下限（清单 §4.2 G1 门判定）"),
               all_pass=bool(G1 and ok_rate and ok_sil and ok_bout and ok_nan and det_equal),
               elapsed_s=round(time.time() - t0, 1))
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    _write_csv(args.out, ["key", "value", "unit", "pass"], [
        ("p5_rate_median_hz", round(base["rate_median_hz"], 6), "Hz", ok_rate),
        ("p5_silent_frac", round(base["silent_frac"], 6), "", ok_sil),
        ("p5_selfsustain_ratio", round(selfsustain, 6), "", ok_ss),
        ("p5_elasticity_4x", round(elasticity, 6), "", ok_el),
        ("p5_elasticity_monotone", int(monotone), "bool", monotone),
        ("p5_bout_fraction", round(base["bout_fraction"], 6), "", ok_bout),
        ("p5_bout_count", base["n_bouts"], "", ok_bout),
        ("p5_pop_rate_hz", round(base["pop_rate_hz"], 6), "Hz", True),
        ("p5_n_nan", base["n_nan"], "", ok_nan),
        ("p5_ms_per_step_steady", round(base["ms_per_step_steady"], 4), "ms/step", True),
        ("p5_wall_s_total", round(out["elapsed_s"], 1), "s", True),
        ("p5_determinism_bit_identical", int(det_equal), "bool", det_equal),
        ("G1_gate_pass", int(G1), "bool", G1),
    ])
    _write_csv(args.out_drive, ["drive_mult", "lam_hz", "pop_rate_hz", "rate_median_hz",
                                "silent_frac", "bout_fraction", "n_bouts", "spk_total",
                                "ms_per_step_steady", "wall_s"], [
        (r["drive_mult"], round(r["lam_hz"], 4), round(r["pop_rate_hz"], 6),
         round(r["rate_median_hz"], 6), round(r["silent_frac"], 6),
         round(r["bout_fraction"], 6), r["n_bouts"], int(r["spk_total"]),
         round(r["ms_per_step_steady"], 4), r["wall_s"]) for r in rows])
    print(json.dumps({k: {kk: vv for kk, vv in v.items()} for k, v in verdict.items()},
                     ensure_ascii=False, indent=1))
    print(f"[P5] G1 = {G1}  all_pass = {out['all_pass']}  ({out['elapsed_s']}s)")
    return 0


def _write_csv(path: str, header, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        f.write("# M10 P5 产物（结构证据级别 = 抽象/Fish1-partial 锚定；判据带见 m10_calibration_targets.csv）\n")
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow(list(r))


if __name__ == "__main__":
    raise SystemExit(main())
