"""M10 追加要求 1/3 落实：**在修正内核上复测 M9 的关键反证**（跨里程碑勘误取证）。

主agent 裁决（M9 交接最重要的发现）：
  M9 冻结内核 `_deliver_step/_deliver_chunk` 的 CSR 起点公式错误（M10 L17）
  → **同一步 ≥2 个神经元发放时越界**（CPU 抛 IndexError；MPS 静默错投/读脏）
  → M9 全规模运行（pop 4.77 Hz / 静默 81.7% ⇒ 每步必然多神经元发放）**恰在该缺陷的触发区间**。

本工具用**同一 M9 定稿网络与定稿参数**（`m9_network.npz` + `validate_p9_resting.PARAMS`，
含 `ahp_g=25`），在同一协议下（T=1000 ms / settle=300 ms / segment=2000 / g_ext=19.230769，
`tools/diag_p7_drive_sensitivity.py` 原协议）跑**双引擎对照**：

  * `m9`  = 冻结引擎 `AdultEngine`（复现 M9 原值，验证对照实验本身）
  * `m10` = 修正内核 `LarvalZFEngine`（E5-FIX + 绝对步号槽）

待复测的 M9 反证：
  - **R3** 零驱动群率比（M9 0.9853）/ 4× 驱动增幅（M9 +7.25%）→ `drive_insensitive`
  - **P4 (d) / P7 (a)** bout 结构（M9 bout 占比 0.0000）
  - **P7 (b)** 昼/夜（调制）比例（M9 1.0003）→ 用 λ_high/λ_low = 3.08× 的**方向判别**臂
  - **P10** 活动统计：登记为 **not_run**（需完整钙正向模型管线，本轮预算不足；不写成未命中）

**不改任何 M9 冻结件**（只读 `m9_network.npz`、导入 `PARAMS`/`BG`、不改 CSV）。

用法（逐臂独立进程，M9 L40 纪律）：
    PYTHONHASHSEED=0 ./.venv-m9/bin/python -m neural_exploration.tools.reeval_m10_m9_rebuttals \
        --engine m10 --lam0 0.0
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
OUT_CSV = os.path.join(DATA_DIR, "m10_m9_reeval.csv")

# **口径警告（RF-9）**：本工具首版在 `reset_counts()` 时机与初值语义上与 `AdultCircuit.run_resting`
# 不一致 ⇒ 绝对速率被放大 (settle+T)/T = 1.3004 倍（主agent 独立探针否证）。已修正；旧产物见
# `data/m10_m9_reeval.csv` 顶部警告行与 `data/m10_m9_reeval_correction.json`。
T_MS = 1000.0
SETTLE_MS = 300.0
SEGMENT = 2000
G_EXT = 2.0 / 0.104          # = EPSP 2 mV / 0.104（M9 原协议）
BASE_LAM0 = 0.5
BASE_REF = {"m9": 4.7482}    # M9 P7 rhythm0 臂（λ0=0.5）


def build_m10_engine(verbose=True):
    """用 M9 定稿网络/参数装配 **M10 修正内核**（不改 M9 冻结件，只读其数据与参数）。"""
    from ..src.adult_circuit import AdultCircuit, CircuitParams
    from ..src.larval_zf_engine import LarvalZFEngine
    from .validate_p9_resting import PARAMS
    P = dict(PARAMS)
    c = AdultCircuit(device="cpu", params=CircuitParams(**P), use_compile=False,
                     verbose=False)
    delay_steps = max(1, int(round(c.params.delay_ms / c.params.dt_ms)))
    p = c.params
    ahp_kw = ({"ahp_inc": p.ahp_inc} if str(p.ahp_form) == "current"
              else {"ahp_g_inc": p.ahp_g_inc})
    n = c.n_neurons
    # 设备选择：全规模在 CPU 上不可行（M9 RB1 的报错已被本节点修复，但仍慢）→ 默认 MPS
    dev = os.environ.get("M10_REEVAL_DEVICE", "mps")
    e = LarvalZFEngine(n, device=dev, use_compile=False, verbose=False,
                       use_abs_slots=True)
    e.set_point_params(v_rest=p.v_rest, v_th=p.v_th, v_reset=p.v_reset, tau_m=p.tau_m,
                       tau_e=p.tau_e, tau_i=p.tau_i, ref_ms=p.ref_ms, dt_ms=p.dt_ms,
                       e_inh=p.e_inh, ek=p.ek, ahp_tau_ms=p.ahp_tau_ms,
                       ahp_form=p.ahp_form, **ahp_kw)
    gmax = c.gmax_vector()
    step = int(np.ceil(c.pre.size / 4))
    for k in range(4):
        a, b = k * step, min((k + 1) * step, c.pre.size)
        if a >= b:
            break
        e.add_synapses(c.pre[a:b], c.post[a:b], gmax[a:b], delay_steps=delay_steps,
                       inhibitory=c.inh[a:b], syn_class=c.nt_code[a:b])
    e.finalize()
    e.set_neuron_heterogeneity(c.bias_vector())
    if verbose:
        print(f"[reeval] M10 引擎装配：{n} 神经元 / {c.pre.size} 边 / "
              f"delay={delay_steps} 步 / 延迟组 {e.n_delay_group} / 槽 {e.n_slot} / "
              f"设备 {dev}", flush=True)
    return e, n, p


def build_m9_engine(verbose=True):
    from ..src.adult_circuit import AdultCircuit, CircuitParams
    from .validate_p9_resting import PARAMS
    c = AdultCircuit(device=os.environ.get("M10_REEVAL_DEVICE", "mps"),
                     params=CircuitParams(**dict(PARAMS)), use_compile=False,
                     verbose=False)
    st = c.build()
    if verbose:
        print(f"[reeval] M9 冻结引擎装配：{c.n_neurons} 神经元 / {c.pre.size} 边 / {st}",
              flush=True)
    return c.engine, c.n_neurons, c.params


def poisson_events(nn: int, lam0: float, n_neurons: int, seed: int = 4242):
    rng = np.random.default_rng(seed)
    n_ev = rng.poisson(max(lam0, 0.0) * nn * 0.05 * 1e-3, size=n_neurons)
    total = int(n_ev.sum())
    if total == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)
    neurons = np.repeat(np.arange(n_neurons, dtype=np.int64), n_ev)
    steps = rng.integers(0, nn, size=total).astype(np.int64)
    return steps, neurons, np.full(total, G_EXT, dtype=np.float32)


def bout_metrics(pop: np.ndarray, dt_ms: float = 0.05, bin_ms: float = 100.0,
                 min_bout_bins: int = 2):
    """与 M10 一致的可证伪口径（interior bout）；用于 P4(d)/P7(a) 的方向判别。"""
    b = max(1, int(round(bin_ms / dt_ms)))
    nb = pop.size // b
    if nb < 4:
        return dict(bout_fraction=0.0, n_bouts=0, silent_bin_frac=float("nan"))
    m = pop[: nb * b].reshape(nb, b).sum(axis=1)
    active = m > 0
    d = np.diff(np.r_[0, active.astype(np.int8), 0])
    st = np.flatnonzero(d == 1); en = np.flatnonzero(d == -1)
    interior = (st > 0) & (en < nb)
    st, en = st[interior], en[interior]
    ln = en - st
    good = ln >= min_bout_bins
    return dict(bout_fraction=float(ln[good].sum() / nb) if good.any() else 0.0,
                n_bouts=int(good.sum()), silent_bin_frac=float((~active).mean()))


def run_arm(engine, n_neurons: int, lam0: float, seed: int = 0, tag: str = "",
            v_rest: float = -52.0):
    """**口径修正（RF-9）**：严格复刻 `AdultCircuit.run_resting` 的测量语义。

    两处此前的口径缺陷（主agent 独立探针捕获，已被否证）：
      ① `reset_counts()` 必须在 **settle 之后**调用（此前在 settle 之前 ⇒ 计数含 settle 窗，
         而速率仍按 1.0 s 归一 ⇒ 全部被放大 (settle+T)/T = 1.3004 倍）；
      ② 初值必须与 `run_resting` 一致：`v0 = v_rest + 2·N(0,1)`（`rng(1)`），
         而不是 `reset(seed=...)` 的默认 v0。
    """
    n_settle = int(round(SETTLE_MS / 0.05))
    n_meas = int(round(T_MS / 0.05))
    nn = n_settle + n_meas + 8
    st, ne, gg = poisson_events(nn, lam0, n_neurons, seed=seed + 4242)
    engine.set_external_events(st, ne, gg, n_steps=nn)
    rng = np.random.default_rng(1)
    v0 = (v_rest + 2.0 * rng.standard_normal(n_neurons)).astype(np.float32)
    engine.reset(v0=v0, seed=seed)
    segs = []
    pops = []
    t0 = time.time()
    # ---- settle 段（丢弃）----
    done = 0
    while done < n_settle:
        w = min(SEGMENT, n_settle - done)
        engine.run(w, delivery="chunk", record="counts")
        done += w
    engine.reset_counts()                    # ① 测量窗起点（与 run_resting 一致）
    done = 0
    while done < n_meas:                     # 测量窗只跑 n_meas 步
        w = min(SEGMENT, n_meas - done)
        t1 = time.time()
        r = engine.run(w, delivery="chunk", record="counts", pop_trace=True)
        segs.append(round(r["ms_per_step"], 4))
        if r["pop"] is not None:
            pops.append(r["pop"])
        done += w
        print(f"    [{tag}] meas-seg {done}/{n_meas}  {time.time()-t1:.1f}s  "
              f"{segs[-1]} ms/step", flush=True)
    pop = np.concatenate(pops) if pops else np.zeros(0)
    popm = pop            # 测量窗已从 settle 之后开始（口径修正后无需再切）
    counts = engine.t_count.detach().cpu().numpy().astype(np.float64)
    t_meas_s = T_MS * 1e-3
    rate = counts / t_meas_s
    out = dict(n_events=int(gg.size), pop_rate_hz=float(counts.sum() / n_neurons / t_meas_s),
               rate_mean_hz=float(rate.mean()), rate_median_hz=float(np.median(rate)),
               silent_frac=float(np.mean(rate < 0.5)),
               wall_s=round(time.time() - t0, 1),
               ms_per_step=float(np.median(segs[1:] or segs)),
               n_nan=int(np.sum(~np.isfinite(rate))))
    out.update(bout_metrics(popm))
    return out


def append_rows(rows):
    new = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        if new:
            f.write("# M10 追加要求 1：在修正内核上复测 M9 关键反证（同一 M9 定稿网络/参数/协议）\n")
            f.write("# 协议：T=1000ms settle=300ms segment=2000 g_ext=19.230769（tools/diag_p7_drive_sensitivity.py 原协议）\n")
            f.write("# 口径：bout = 可证伪 interior 口径（100 ms 分箱，两侧须有静默箱，≥2 箱）\n")
            w.writerow(["engine", "lam0", "n_events", "pop_rate_hz", "rate_mean_hz",
                        "rate_median_hz", "silent_frac", "bout_fraction", "n_bouts",
                        "silent_bin_frac", "ms_per_step", "wall_s"])
        for r in rows:
            w.writerow([r["engine"], r["lam0"], r["n_events"], round(r["pop_rate_hz"], 6),
                        round(r["rate_mean_hz"], 6), round(r["rate_median_hz"], 6),
                        round(r["silent_frac"], 6), round(r["bout_fraction"], 6),
                        r["n_bouts"], round(r["silent_bin_frac"], 6),
                        round(r["ms_per_step"], 4), r["wall_s"]])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="m10", choices=["m9", "m10"])
    ap.add_argument("--lam0", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    t0 = time.time()
    if args.engine == "m10":
        eng, n, p = build_m10_engine()
    else:
        eng, n, p = build_m9_engine()
    _v_rest = float(getattr(p, "v_rest", -52.0))
    r = run_arm(eng, n, args.lam0, seed=args.seed, tag=f"{args.engine} λ={args.lam0}",
                v_rest=_v_rest)
    r["engine"] = args.engine
    r["lam0"] = args.lam0
    print(json.dumps(r, ensure_ascii=False), flush=True)
    append_rows([r])
    print(f"[reeval {args.engine} λ={args.lam0}] pop={r['pop_rate_hz']:.4f} "
          f"med={r['rate_median_hz']:.4f} silent={r['silent_frac']:.4f} "
          f"bout={r['bout_fraction']:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
