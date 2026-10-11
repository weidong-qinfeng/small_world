"""M11 引擎门附加工具：**上游点核缺陷（E-M9-4）对 M10 P5/G1 静息标定的量级影响**。

背景（见 `docs/m11_b2_notes.md` E-M9-4）：冻结件 `adult_engine._point_core_factory` 的点核形参序为
`(v, ge, gi, g_ahp, cool, …)`，而 4 个 point 调用点传入 `(t_cool, t_ahp, …)` ⇒ 不应期门实读 `t_ahp`、
AHP 电导实读 `t_cool`。M10 的 P5/G1（静息标定）与 P6（机制阶梯）**全部 point 档数值都在缺陷档下取得**。

本工具用 **M11 内核**在**同一网络/同一事件表/同一 seed** 下跑两档：
  - `REF_FROZEN`（**缺陷档复刻**，逐位等于 M10 语义，实测已验证）；
  - `REF_FIX`（文档语义，逐位等于 M10 的 `gaba_split=True, frac_slow=0` 路径）。
输出静息统计（群率/中位率/静默比/自持比/4× 弹性）与两档差值 ⇒ 供主 agent 判断 M10 结论的
"缺陷档"成分。

**不修改任何冻结件**；不改动 M10 的判据带。

用法（预计墙钟：N=1000、T=10 s、6 臂 × 2 档 ≈ 8–20 min ⇒ **托管作业**）：
    PYTHONHASHSEED=0 OMP_NUM_THREADS=1 ./.venv-m9/bin/python \\
      -m neural_exploration.tools.audit_m10_point_defect --n 1000 --t-s 10 --seeds 3 --device cpu
产物：`data/m11_m10_defect_audit.csv` + `data/m11_m10_defect_audit.json`
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import time

import numpy as np

from neural_exploration.src.mouse_cortex_engine import (MouseCortexEngine,       # noqa: E402
                                                        REF_FIX, REF_FROZEN)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
LAM_GRID = (0.0, 1.0, 4.0, 8.0)      # M10 P5 的 λ× 档（剂量-响应）


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def run_arm(n, lam, T_s, seed, form, device="cpu"):
    """M10 P5 的配置语义：N 点神经元 / indegree 20 / bias 亚阈 / λ0=8 Hz 外部事件。"""
    T = int(T_s / 0.05e-3)
    e = MouseCortexEngine(n, device=device, use_compile=False, verbose=False)
    e.configure_refractory(form)
    rng = np.random.default_rng(seed)
    E = int(n * 20)
    pre = rng.integers(0, n, E)
    post = rng.integers(0, n, E)
    g = rng.uniform(6.0, 18.0, E).astype(np.float32)
    inh = rng.random(E) < 0.2
    e.add_synapses(pre, post, g, np.full(E, 40, dtype=np.int32), inhibitory=inh,
                   syn_class=inh.astype(np.int8))
    e.finalize()
    e.set_neuron_heterogeneity(np.exp(rng.normal(np.log(230.0), 0.30, n)).astype(np.float32))
    n_ev = int(8.0 * lam * T * 0.05e-3 * n)
    st = rng.integers(0, T, max(0, n_ev))
    ne = rng.integers(0, n, max(0, n_ev))
    gg = rng.uniform(17.0, 23.0, max(0, n_ev)).astype(np.float32)
    e.set_external_events(st, ne, gg, n_steps=T)
    e.reset(seed=seed + 1)
    e.reset_counts()
    r = e.run(T, delivery="chunk", record="counts")
    cnt = e.t_count.detach().cpu().numpy().astype(np.float64)
    rate = cnt / (T * 0.05e-3)
    return dict(n=n, lam=lam, T_s=T_s, seed=seed, form=form, n_edge=E,
                spk_total=float(cnt.sum()), pop_hz=float(cnt.sum() / n / (T * 0.05e-3)),
                rate_median_hz=float(np.median(rate)), rate_mean_hz=float(rate.mean()),
                silent_frac=float(np.mean(rate < 0.5)), n_nan=int(np.sum(~np.isfinite(rate))),
                ms_per_step=round(r["ms_per_step"], 4), wall_s=round(r["wall_s"], 2),
                n_ch=int(e.n_ch), core=e.core_readback()["core_key"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--t-s", type=float, default=10.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lam", default=",".join(str(x) for x in LAM_GRID))
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "m11_m10_defect_audit.csv"))
    ap.add_argument("--out-json", default=os.path.join(DATA_DIR, "m11_m10_defect_audit.json"))
    args = ap.parse_args(argv)
    rows = []
    t0 = time.time()
    for form in (REF_FROZEN, REF_FIX):
        for lam in [float(x) for x in args.lam.split(",") if x.strip()]:
            for s in range(args.seeds):
                r = run_arm(args.n, lam, args.t_s, s, form, device=args.device)
                rows.append(r)
                print("  [audit] %-11s λ×%.1f seed%d → pop %.4f Hz 中位 %.3f 静默 %.3f ms/step %.4f"
                      % (form, lam, s, r["pop_hz"], r["rate_median_hz"], r["silent_frac"],
                         r["ms_per_step"]), flush=True)
    # 自持比（λ×0）：零驱动臂 / λ×1 臂
    summary = {}
    for lam in (0.0, 1.0):
        for form in (REF_FROZEN, REF_FIX):
            a = [r for r in rows if r["form"] == form and r["lam"] == lam]
            if a:
                summary["%s_lam%.1f" % (form, lam)] = dict(
                    pop_hz=float(np.mean([r["pop_hz"] for r in a])),
                    spk_total=float(np.sum([r["spk_total"] for r in a])),
                    silent_frac=float(np.mean([r["silent_frac"] for r in a])),
                    n_seeds=len(a))
    for form in (REF_FROZEN, REF_FIX):
        z = summary.get("%s_lam0.0" % form, {}).get("spk_total")
        o = summary.get("%s_lam1.0" % form, {}).get("spk_total")
        if z is not None and o:
            summary["%s_self_sustain_ratio" % form] = float(z / o)
    out = dict(generated_by="tools/audit_m10_point_defect.py",
               started_at=time.strftime("%Y-%m-%dT%H:%M:%S"), elapsed_s=round(time.time() - t0, 1),
               n=args.n, t_s=args.t_s, seeds=args.seeds, lam_grid=args.lam, rows=rows,
               summary=summary, rss_mb=round(rss_mb(), 1),
               note="冻结档=REF_FROZEN（逐位等于 M10 非 gaba 点核语义）；修正档=REF_FIX（文档语义）。"
                    "两档差值量化 E-M9-4 对 M10 P5/G1 数值的影响；**不改动 M10 判据带**。")
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False, default=str)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(list(rows[0].keys()))
        for r in rows:
            w.writerow([r[k] for k in rows[0].keys()])
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
    print("csv ->", args.out, "| json ->", args.out_json, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
