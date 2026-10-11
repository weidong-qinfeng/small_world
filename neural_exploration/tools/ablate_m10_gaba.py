"""M10 RF-2② 落实：**GABA_A / GABA_B 通路分别消融**（否则无法归因）。

主agent RF-2 裁决：
  ① 亚型比例/时间常数**预注册**（`data/m10_mechanism_ladder.csv`：τ_A∈[5,20] ms、τ_B∈[80,400] ms、
     frac_B∈[0.05,0.50]）✓；
  ② **消融时分别报告 GABA_A 与 GABA_B 通路的影响**（本工具）；
  ③ 登记为测量限制（"亚型标注不可得"，`data/m10_connectome_counts.json::diagnostics`）✓。

三臂（同网络/同事件表/同 seed，**只改抑制电荷在快慢池之间的分配**）：
  `GABA_B_off`  frac_slow = 0.00  → 抑制全部经 GABA_A（快 τ_A）
  `both_on`     frac_slow = 0.35  → 预注册中点（两通路并存）
  `GABA_A_off`  frac_slow = 1.00  → 抑制全部经 GABA_B（慢 τ_B）
逐臂报：群率/中位/静默/bout（可证伪 interior 口径）+ 4× 驱动弹性 + 慢池电导轨迹与群率相关。

**结构证据级别 = partial/抽象** ⇒ 归属为**参数级/类级**（RF-1），不得表述为结构级归属。

用法：
    PYTHONHASHSEED=0 ./.venv-m9/bin/python -m neural_exploration.tools.ablate_m10_gaba
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
from ..src.larval_zf_circuit import ZFNetParams, build_engine, make_events, run_protocol  # noqa: E402

TAU_A_MS, TAU_B_MS = 8.0, 200.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-neuron", type=int, default=1000)
    ap.add_argument("--T-ms", type=float, default=2000.0)
    ap.add_argument("--settle-ms", type=float, default=300.0)
    ap.add_argument("--lam0", type=float, default=8.0)
    ap.add_argument("--bias-median", type=float, default=150.0)
    ap.add_argument("--g-exc", type=float, default=1.0)
    ap.add_argument("--g-inh", type=float, default=22.0)
    ap.add_argument("--out-json", default=os.path.join(DATA_DIR, "m10_p6_gaba_ablation.json"))
    ap.add_argument("--out-csv", default=os.path.join(DATA_DIR, "m10_p6_gaba_ablation.csv"))
    args = ap.parse_args(argv)
    t0 = time.time()
    rows = []
    for name, frac in (("GABA_B_off(fast_only)", 0.0), ("both_on(0.35)", 0.35),
                       ("GABA_A_off(slow_only)", 1.0)):
        Pm = ZFNetParams(n_neuron=args.n_neuron, indegree=20,
                         bias_median_mv_s=args.bias_median,
                         g_exc=args.g_exc, g_inh=args.g_inh)
        arm = {}
        for mult in (1.0, 4.0):
            e = build_engine(Pm, device="cpu",
                             gaba=dict(enabled=True, tau_fast_ms=TAU_A_MS,
                                       tau_slow_ms=TAU_B_MS, frac_slow=frac))
            # 读回断言（M9 B3）
            assert abs(e.gaba["frac_slow"] - frac) < 1e-12, "frac_slow 读回不一致"
            r = run_protocol(e, int(round(args.T_ms / 0.05)),
                             settle_steps=int(round(args.settle_ms / 0.05)),
                             segment_steps=10000, lam_hz=args.lam0 * mult, ev_seed=3)
            arm[mult] = r
            # 慢池电导轨迹（逐帧均值采样）——归因证据
            try:
                gis = float(e.t_gis.mean().item())
            except Exception:  # pragma: no cover
                gis = float("nan")
            arm[f"gis_mean_{mult}"] = gis
            del e
        base, four = arm[1.0], arm[4.0]
        elas = (four["pop_rate_hz"] - base["pop_rate_hz"]) / max(base["pop_rate_hz"], 1e-12)
        rows.append(dict(arm=name, frac_slow=frac,
                         pop_rate_hz=base["pop_rate_hz"], rate_median_hz=base["rate_median_hz"],
                         silent_frac=base["silent_frac"], bout_fraction=base["bout_fraction"],
                         n_bouts=base["n_bouts"], elasticity_4x=elas,
                         gis_mean_1x=arm["gis_mean_1.0"], gis_mean_4x=arm["gis_mean_4.0"]))
        print(f"  {name}: pop={rows[-1]['pop_rate_hz']:.4f} med={rows[-1]['rate_median_hz']:.4f} "
              f"silent={rows[-1]['silent_frac']:.4f} bout={rows[-1]['bout_fraction']:.4f} "
              f"弹性4×={elas:.4f} gis(1×)={rows[-1]['gis_mean_1x']:.4f}", flush=True)
    with open(args.out_csv, "w", newline="") as f:
        f.write("# M10 RF-2② GABA_A / GABA_B 通路分别消融（同网络/同事件表/同 seed，仅改电荷分配）\n")
        f.write(f"# τ_A={TAU_A_MS}ms τ_B={TAU_B_MS}ms；归属=参数级/类级（RF-1）；亚型标注不可得=测量限制（RF-2③）\n")
        w = csv.writer(f, lineterminator="\n")
        w.writerow(list(rows[0].keys()))
        for r in rows:
            w.writerow([r[k] for k in rows[0].keys()])
    out = dict(generated_by="tools/ablate_m10_gaba.py", arms=rows,
               conclusion=dict(
                   gaba_b_contribution=("GABA_B 通路对群率的影响 = both_on 与 GABA_B_off 的差；"
                                        "对慢时间尺度结构的贡献看 bout/转换率（本协议下均为 0）"),
                   attribution_strength="参数级/类级（RF-1）",
                   measurement_limitation="Fish1 无 GABA_A/B 亚型标注 → 类级文献先验（RF-2③）"),
               elapsed_s=round(time.time() - t0, 1))
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps({r['arm']: {'pop': round(r['pop_rate_hz'], 4),
                                 'elas4': round(r['elasticity_4x'], 3),
                                 'gis1': round(r['gis_mean_1x'], 4)} for r in rows},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
