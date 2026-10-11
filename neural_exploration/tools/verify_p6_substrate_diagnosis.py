"""主agent 验收诊断探针：M10 P6 阴性结论的**归因判别**——网络是否存在"双稳基质"？

**为什么需要这个探针**（读码 + 已跑数据推断，非推测）：
  所有已跑 P6 阶梯臂的共同特征是**驱动-响应单调**（λ×0→0.03–0.08 Hz、λ×1→0.17–0.27 Hz、λ×4→2.4–2.5 Hz），
  即网络**没有"上态"可自持**（`g_exc=1.0` / `g_inh=22` / 10% 抑制 ⇒ 抑制主导，无兴奋自持回路）。
  若如此，则"STP/GABA/调质都不产生双态"这一阴性结论**可能根本不是关于机制**，
  而是关于**替代网络的参数化**——即该网络**没有可用于产生双态的结构基质**
  （无上态 ⇒ 慢变量无从"终止一个上态"）。

**清单 §8 反证路径明确要求**：记录反证前必须"**升阶到组合 → 检查连接组子集/权重标定**"。
本探针做的正是这一步的**最简形式**：在 (g_exc, bias) 平面上做**滞回（hysteresis）扫描**——
  * 同一网络、同一 seed，驱动先**斜坡上升**（0 → λ_max），再**斜坡下降**（λ_max → 0）；
  * 若存在双稳态（上态自持），**下降支的速率应显著高于上升支**（滞回环非零）⇒ 基质存在；
  * 若两支重合（滞回 ≈ 0）⇒ **该参数区域内无双稳基质**，阴性结论必须改写成
    "替代网络无上态 ⇒ 慢变量无用武之地"，并据此调整权重标定或抽象。

**只读** M10 修正内核（`build_engine` 等），不改任何文件；产物写
`data/_mainagent_p6_substrate.{json,csv}`。

用法：
    PYTHONHASHSEED=0 OMP_NUM_THREADS=1 ./.venv-m9/bin/python \
        -m neural_exploration.tools.verify_p6_substrate_diagnosis --g-exc 1.0 --bias 150
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
OUT_JSON = os.path.join(DATA_DIR, "_mainagent_p6_substrate.json")
OUT_CSV = os.path.join(DATA_DIR, "_mainagent_p6_substrate.csv")


def make_ramp_events(n_steps: int, n: int, lam_max_hz: float, src, dt_ms: float,
                     bin_steps: int, seed: int = 7, g_ext: float = 20.0):
    """时间变化率的事件流：λ(t) 三角斜坡 0→λ_max→0（每 bin 一次泊松抽样）。"""
    rng = np.random.default_rng(seed)
    half = n_steps // 2
    steps_l, neu_l, g_l = [], [], []
    for b0 in range(0, n_steps, bin_steps):
        b1 = min(b0 + bin_steps, n_steps)
        mid = 0.5 * (b0 + b1)
        frac = (mid / half) if mid <= half else max(0.0, (1.0 - (mid - half) / half))
        lam = lam_max_hz * frac
        dur_s = (b1 - b0) * dt_ms * 1e-3
        k = int(rng.poisson(max(lam, 0.0) * n * dur_s))
        if k <= 0:
            continue
        steps_l.append(rng.integers(b0, b1, size=k))
        neu_l.append(rng.integers(0, n, size=k))
        g_l.append(np.full(k, g_ext, dtype=np.float32))
    if not steps_l:
        return (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32))
    return (np.concatenate(steps_l).astype(np.int64), np.concatenate(neu_l).astype(np.int64),
            np.concatenate(g_l))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-neuron", type=int, default=1000)
    ap.add_argument("--indegree", type=int, default=20)
    ap.add_argument("--g-exc", type=float, default=1.0)
    ap.add_argument("--g-inh", type=float, default=22.0)
    ap.add_argument("--bias", type=float, default=150.0)
    ap.add_argument("--lam-max", type=float, default=32.0)
    ap.add_argument("--t-ramp-ms", type=float, default=10_000.0, help="单程时长（上/下各一段）")
    ap.add_argument("--settle-ms", type=float, default=1000.0)
    ap.add_argument("--bin-ms", type=float, default=200.0)
    args = ap.parse_args(argv)

    from ..src.larval_zf_circuit import ZFNetParams, build_engine, DT_MS

    t0 = time.time()
    Pm = ZFNetParams(n_neuron=args.n_neuron, indegree=args.indegree,
                     bias_median_mv_s=args.bias, g_exc=args.g_exc, g_inh=args.g_inh)
    e = build_engine(Pm, device="cpu")
    settle_steps = int(round(args.settle_ms / DT_MS))
    ramp_steps = int(round(args.t_ramp_ms / DT_MS))
    n_steps = 2 * ramp_steps
    bin_steps = max(1, int(round(args.bin_ms / DT_MS)))

    # ---- 参数读回断言（"参数是活的"）----
    rb = dict(n_neuron=int(e.n), n_edge=int(e.n_edge), g_exc=float(Pm.g_exc),
              g_inh=float(Pm.g_inh), bias_median=float(Pm.bias_median_mv_s),
              inh_frac=float(Pm.inh_frac), dt_ms=float(DT_MS))
    assert rb["n_edge"] == args.n_neuron * args.indegree, rb

    st, ne, gg = make_ramp_events(n_steps, int(e.n), args.lam_max, None, DT_MS, bin_steps)
    e.set_external_events(st, ne, gg, n_steps=n_steps)
    e.reset(seed=1)
    # settle（零驱动，丢弃）
    if settle_steps > 0:
        e.set_external_events(np.zeros(0, np.int64), np.zeros(0, np.int64),
                              np.zeros(0, np.float32), n_steps=settle_steps)
        e.run(settle_steps, delivery="step", record="counts")
    # 恢复斜坡事件并把绝对步号平移到 settle 之后
    e.set_external_events(st + settle_steps, ne, gg, n_steps=settle_steps + n_steps)
    e.reset_counts()
    pops, seg = [], 0
    while seg < n_steps:
        w = min(20_000, n_steps - seg)
        r = e.run(w, delivery="step", record="counts", pop_trace=True)
        pops.append(r["pop"])
        seg += w
    pop = np.concatenate(pops).astype(np.float64)
    assert pop.size == n_steps, (pop.size, n_steps)

    nb = n_steps // bin_steps
    prof = pop[: nb * bin_steps].reshape(nb, bin_steps).mean(axis=1) / (DT_MS * 1e-3) / e.n
    half = nb // 2
    lam_prof = np.concatenate([np.linspace(0, args.lam_max, half),
                               np.linspace(args.lam_max, 0, nb - half)])
    up, dn = slice(0, half), slice(half, nb)
    # 滞回量：在下行支的速率曲线上，用与上行支相同的 λ 网格插值比较（上行 λ 递增 ⇒ 反转后可比）
    lam_up = lam_prof[up]
    lam_dn = lam_prof[dn][::-1]
    r_up = prof[up]
    r_dn = prof[dn][::-1]
    lo, hi = 0.05 * args.lam_max, 0.95 * args.lam_max
    m = (lam_up >= lo) & (lam_up <= hi) & (lam_dn >= lo) & (lam_dn <= hi)
    grid = np.linspace(lo, hi, 24)
    ru = np.interp(grid, lam_up[m], r_up[m])
    rd = np.interp(grid, lam_dn[::-1], r_dn[::-1])
    hyst = rd - ru
    scale = max(float(np.mean(ru)), 1e-9)
    rec = dict(net=rb, protocol=dict(lam_max_hz=args.lam_max, t_ramp_ms=args.t_ramp_ms,
                                     settle_ms=args.settle_ms, bin_ms=args.bin_ms,
                                     n_steps=n_steps, n_events=int(gg.size)),
               rate_up_max_hz=float(ru.max()), rate_dn_max_hz=float(rd.max()),
               rate_up_at_half_hz=float(np.interp(0.5 * args.lam_max, lam_up[m], r_up[m])),
               rate_dn_at_half_hz=float(np.interp(0.5 * args.lam_max, lam_dn[::-1], r_dn[::-1])),
               hysteresis_mean_hz=float(hyst.mean()), hysteresis_max_hz=float(hyst.max()),
               hysteresis_rel_to_up=float(hyst.mean() / scale),
               bistable_substrate=bool(hyst.mean() / scale > 0.10),
               ms_per_step=float(r["ms_per_step"]), wall_s=round(time.time() - t0, 1),
               grid_lam=list(np.round(grid, 4)), grid_rate_up=list(np.round(ru, 6)),
               grid_rate_dn=list(np.round(rd, 6)))

    rows = []
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f:
            rows = json.load(f)
    # **勘误（主agent 自查）**：去重键原先漏了 `t_ramp_ms`，导致同配置不同斜坡时长的行被互相覆盖
    # ⇒ 斜坡时长敏感性无法累积（本探针首版即因此丢失了 2/4/10/20 s 四行）。现补入。
    key = (args.g_exc, args.g_inh, args.bias, args.n_neuron, args.indegree, args.lam_max,
           args.t_ramp_ms)
    rows = [r for r in rows if (r["net"]["g_exc"], r["net"]["g_inh"],
                                r["net"]["bias_median"], r["net"]["n_neuron"],
                                r["net"]["n_edge"] // max(r["net"]["n_neuron"], 1),
                                r["protocol"]["lam_max_hz"],
                                r["protocol"]["t_ramp_ms"]) != key]
    rows.append(rec)
    with open(OUT_JSON, "w") as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["g_exc", "g_inh", "bias", "n_neuron", "indegree", "lam_max",
                    "rate_up_at_half_hz", "rate_dn_at_half_hz", "hysteresis_mean_hz",
                    "hysteresis_rel", "bistable_substrate", "ms_per_step", "wall_s"])
        for r in rows:
            w.writerow([r["net"]["g_exc"], r["net"]["g_inh"], r["net"]["bias_median"],
                        r["net"]["n_neuron"], r["net"]["n_edge"] // max(r["net"]["n_neuron"], 1),
                        r["protocol"]["lam_max_hz"],
                        round(r["rate_up_at_half_hz"], 6), round(r["rate_dn_at_half_hz"], 6),
                        round(r["hysteresis_mean_hz"], 6), round(r["hysteresis_rel_to_up"], 4),
                        int(r["bistable_substrate"]), round(r["ms_per_step"], 4), r["wall_s"]])
    print("MAINAGENT_SUBSTRATE " + json.dumps(
        {k: rec[k] for k in ("net", "rate_up_at_half_hz", "rate_dn_at_half_hz",
                             "hysteresis_mean_hz", "hysteresis_rel_to_up",
                             "bistable_substrate", "wall_s")}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
