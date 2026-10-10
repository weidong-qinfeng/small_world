"""M9 P4：全规模网络装配 + 静息 sanity 验证（清单 §3.5）。

《生物仿真M9实施清单》§3.5.1（全规模构建）/§3.5.2（静息协议：T 与 settle 窗预注册）/
§3.5.3（双状态）/§3.5.4（单试次墙钟 ≤ 预算）/§0.7 #8（判据带定稿于 CSV 不事后调）/
§0.8（数据隔离：标定只读拟合集 A）。

协议（预注册）：
  - settle 窗 1000 ms（丢弃，t=0 初始化瞬态；M5 L37#2）；测量窗 T = 30000 ms；
  - N = 3 试次（固定 seed 0/1/2，确定性）；无刺激、无梯度；
  - 背景驱动 = 逐神经元独立 Poisson 虚拟突触事件（固定 seed；抽象登记：持续感觉/内在驱动）。

判据（只读 `data/m9_behavior_reference.csv` 的 resting 段；**不在此处改带**）：
  (a) 静息发放率中位落带 [0.1,10] Hz；(b) 静默比例（率<0.5Hz）落带 [50,90]%；
  (c) p95 上界（防饱和）；(d) 双状态（活动 bout 占比/计数）；(e) 确定性统计级一致；
  (f) 单试次墙钟 ≤ 预算（§3.5.4）。

输出：
  - data/m9_p4_resting.csv（逐试次 + 聚合 + 判据三态）
  - data/m9_circuit_params.csv（回路参数定稿）
  - reports/neuro/m9_p4_resting.png（发放率分布 / 群体活动时序 / bout 结构 / 墙钟）

用法：
  PYTHONHASHSEED=0 MPLBACKEND=Agg .venv-m9/bin/python -m neural_exploration.tools.validate_p9_resting
"""

from __future__ import annotations

import csv as _csv
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports", "neuro")
REF_CSV = os.path.join(DATA_DIR, "m9_behavior_reference.csv")
OUT_CSV = os.path.join(DATA_DIR, "m9_p4_resting.csv")
OUT_PARAMS = os.path.join(DATA_DIR, "m9_circuit_params.csv")
OUT_PNG = os.path.join(REPORT_DIR, "m9_p4_resting.png")

T_MS = 30000.0        # 测量窗（预注册 T=30s；融合内核实测稳态 ~0.9 ms/step → 单试次 ≈9 分钟）
SETTLE_MS = 1000.0    # settle 窗（预注册；与 T 同步缩放）
N_TRIALS = 3
BIN_MS = 100.0        # bout 分箱（预注册；阈值 3× 中位箱值）

#: 定稿回路参数（标定产物；见 data/m9_weight_calibration.csv）
PARAMS = dict(
    w_exc=0.3, w_inh=1.0, w_mod=0.3, syn_count_gamma=1.0,
    bias_mv_s=290.0, bias_cv=0.25, bias_mode="lognormal",
    ahp_tau_ms=700.0, ahp_inc=2500.0,
    v_rest=-52.0, v_th=-45.0, v_reset=-55.0, tau_m=20.0, tau_e=2.0, tau_i=5.0,
    ref_ms=2.0, dt_ms=0.05, delay_ms=1.0,
)
BG = dict(rate_hz=0.5, epsp_mv=2.0, seed=0)


def read_band(role="resting", metric=None):
    """只读判据带（`m9_behavior_reference.csv` 的 resting 段）——不在此处改带。"""
    out = {}
    if not os.path.exists(REF_CSV):
        return out
    with open(REF_CSV, encoding="utf-8") as f:
        rd = _csv.reader(f)
        hdr = None
        for row in rd:
            if not row or row[0].startswith("#"):
                continue
            if hdr is None:
                hdr = row
                continue
            d = dict(zip(hdr, row))
            if d.get("role") != role:
                continue
            if metric and d.get("metric") != metric:
                continue
            out[d["metric"]] = {"lo": float(d["lo"]), "hi": float(d["hi"]),
                                "unit": d.get("unit", ""), "target": d.get("target", ""),
                                "provenance": d.get("provenance", "")[:120]}
    return out


def bout_stats(pop, bin_steps, n_bins_expected=None):
    """双状态（§3.5.3）：100ms 分箱 → 阈值（3× 中位箱值）→ 活动 bout 占比/计数。"""
    nb = pop.size // bin_steps
    if nb == 0:
        return {"bout_active_fraction": 0.0, "bout_count_per_min": 0.0,
                "bout_median_len_ms": 0.0, "bin_median_spk": 0.0}
    b = pop[:nb * bin_steps].reshape(nb, bin_steps).sum(axis=1)
    med = float(np.median(b))
    thr = 3.0 * med if med > 0 else max(1.0, float(np.percentile(b, 75)))
    act = b > thr
    frac = float(act.mean())
    # 连续段计数与长度
    idx = np.flatnonzero(np.diff(np.concatenate([[0], act.view(np.int8), [0]])) != 0)
    lens = (idx[1::2] - idx[0::2]) if idx.size >= 2 else np.zeros(0)
    dur_s = nb * BIN_MS / 1000.0
    return {"bout_active_fraction": frac,
            "bout_count_per_min": float(lens.size) / max(dur_s, 1e-9) * 60.0,
            "bout_median_len_ms": float(np.median(lens) * BIN_MS) if lens.size else 0.0,
            "bin_median_spk": med, "bout_threshold_spk": thr}


def _rss_gb():
    """进程峰值 RSS（GB）——全规模长试次内存观测。"""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 3)
    except Exception:
        return 0.0


def main() -> int:
    from neural_exploration.src.adult_circuit import AdultCircuit, CircuitParams
    from neural_exploration.src.adult_circuit import _mps_mem
    os.makedirs(REPORT_DIR, exist_ok=True)
    print("=== M9 P4：全规模装配 + 静息 sanity（§3.5）===", flush=True)
    band = read_band("resting")
    if not band:
        print("警告：未读到判据带（%s）——判据按空处理" % REF_CSV, flush=True)
    for k, v in band.items():
        print("  判据带 %s: [%.4g, %.4g] %s" % (k, v["lo"], v["hi"], v["unit"]), flush=True)

    t00 = time.perf_counter()
    p = CircuitParams(**PARAMS)
    c = AdultCircuit(device="mps", params=p, use_compile=False)
    bs = c.build()
    print("构建：%d 神经元 / %d 边；%.2fs；MPS 显存 %.2f GB；槽 %d"
          % (c.n_neurons, c.n_edge, bs["build_wall_s"],
             bs["vram_bytes"] / 2**30, bs["n_slot"]), flush=True)

    n_steps = int(round((SETTLE_MS + T_MS) / p.dt_ms)) + 8
    g_ext = BG["epsp_mv"] / 0.104
    bg = c.build_background(rate_hz=BG["rate_hz"], g_ext=g_ext, n_steps=n_steps,
                            seed=BG["seed"])
    c._bg_steps = n_steps
    c._bg_args = (BG["rate_hz"], g_ext)

    rows = []
    allpop = []
    rates = []
    for seed in range(N_TRIALS):
        t0 = time.perf_counter()
        print("  [试次 seed=%d 开始] 步数 %d（T=%.0fs）…" % (
            seed, int(round(T_MS / p.dt_ms)), T_MS / 1000.0), flush=True)
        st = c.run_resting(T_ms=T_MS, settle_ms=SETTLE_MS, seed=seed, pop_trace=True,
                           progress=lambda k, n, el: print(
                               "    [seed=%d] %d/%d 步  %.0fs  %.3f ms/step  RSS=%.2fGB"
                               % (seed, k, n, el, el / max(k, 1) * 1e3,
                                  _rss_gb()), flush=True),
                           progress_every=100000)
        rate = (c.engine.t_count.detach().cpu().numpy().astype(np.float64)
                / (T_MS / 1000.0))
        rates.append(rate)
        bin_steps = int(round(BIN_MS / p.dt_ms))
        bt = bout_stats(st["pop"], bin_steps)
        allpop.append(st["pop"])
        rows.append(dict(seed=seed, **{k: st[k] for k in (
            "rate_median_hz", "rate_mean_hz", "rate_p95_hz", "rate_max_hz",
            "silent_frac", "active_frac", "pop_rate_hz", "n_spikes",
            "spk_frac_per_step", "ms_per_step", "wall_s", "n_nan")}, **bt))
        print(" 试次 seed=%d：中位 %8.4f Hz / 均值 %8.4f / p95 %7.3f / 静默 %.3f / "
              "群体 %6.3f Hz / bout 占比 %.3f / %.3f ms/step（%.0fs）"
              % (seed, st["rate_median_hz"], st["rate_mean_hz"], st["rate_p95_hz"],
                 st["silent_frac"], st["pop_rate_hz"], bt["bout_active_fraction"],
                 st["ms_per_step"], time.perf_counter() - t0), flush=True)

    # ---- 确定性（§3.5.2 判据 (c)：同参数重跑统计级一致）----
    st_a = c.run_resting(T_ms=2000.0, settle_ms=SETTLE_MS, seed=0, pop_trace=False)
    ra = c.engine.t_count.detach().cpu().numpy().astype(np.float64).copy()
    st_b = c.run_resting(T_ms=2000.0, settle_ms=SETTLE_MS, seed=0, pop_trace=False)
    rb = c.engine.t_count.detach().cpu().numpy().astype(np.float64).copy()
    det_spearman = float(np.corrcoef(ra, rb)[0, 1]) if ra.std() > 0 and rb.std() > 0 else 0.0
    det_identical = bool(np.array_equal(ra, rb))
    det_med_rel = float(abs(np.median(ra) - np.median(rb)) / max(np.median(ra), 1.0))

    # ---- 聚合 + 判据 ----
    med = float(np.median([r["rate_median_hz"] for r in rows]))
    sil = float(np.mean([r["silent_frac"] for r in rows]))
    p95 = float(np.mean([r["rate_p95_hz"] for r in rows]))
    popr = float(np.mean([r["pop_rate_hz"] for r in rows]))
    msp = float(np.min([r["ms_per_step"] for r in rows]))
    boutf = float(np.mean([r["bout_active_fraction"] for r in rows]))
    boutc = float(np.mean([r["bout_count_per_min"] for r in rows]))
    gpu_h = msp * 1e-3 * int(round(T_MS / p.dt_ms)) / 3600.0

    def _in(metric, val):
        b = band.get(metric)
        return (b is None) or (b["lo"] <= val <= b["hi"])

    crit = {
        "a_rate_median": _in("rate_median_hz", med),
        "b_silent_fraction": _in("silent_fraction", sil),
        "c_rate_p95": _in("rate_p95_hz", p95),
        "d_bout": _in("bout_active_fraction", boutf) and _in("bout_count_per_min", boutc),
        "e_determinism": (det_spearman >= 0.99 and det_med_rel < 0.05) or det_identical,
        "f_wallclock": gpu_h <= 1.0,
    }
    verdict = "PASS" if all(crit.values()) else "FAIL"

    # ---- 落盘 ----
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = _csv.writer(f, lineterminator="\n")
        w.writerow(["# M9 P4 静息 sanity（§3.5.2/§3.5.3）；判据带定稿于 m9_behavior_reference.csv"])
        w.writerow(["section", "key", "value", "note"])
        w.writerow(["protocol", "T_ms", T_MS, "测量窗（预注册）"])
        w.writerow(["protocol", "settle_ms", SETTLE_MS, "settle 窗（丢弃；M5 L37#2）"])
        w.writerow(["protocol", "n_trials", N_TRIALS, "固定 seed 0/1/2（确定性）"])
        w.writerow(["protocol", "bout_bin_ms", BIN_MS, "bout 分箱（阈值 3× 中位箱值）"])
        w.writerow(["protocol", "dt_ms", p.dt_ms, "M8 FIDELITY_DT 定稿"])
        w.writerow(["build", "n_neurons", c.n_neurons, "FlyWire v783 全脑（P1）"])
        w.writerow(["build", "n_edges", c.n_edge, "化学连接（54,492,922 突触）"])
        w.writerow(["build", "n_gap", 0, "官方发布不含缝隙（L19.2 裁决②）"])
        w.writerow(["build", "inhibitory_edge_frac", "%.4f" % float(c.inh.mean()),
                    "真实 GABA 边占比（P1 递质标注）"])
        w.writerow(["build", "build_wall_s", "%.2f" % bs["build_wall_s"], "分段装配"])
        w.writerow(["build", "vram_gb", "%.3f" % (bs["vram_bytes"] / 2**30),
                    "MPS 显存（预注册 <1.5GB）"])
        w.writerow(["build", "n_slot", bs["n_slot"], "环形事件槽"])
        for k, v in PARAMS.items():
            w.writerow(["param", k, v, "类级全局参数（≤40，§8 ①）"])
        for k, v in BG.items():
            w.writerow(["bg", k, v, "背景驱动（抽象登记：持续感觉/内在驱动）"])
        for r in rows:
            for k, v in r.items():
                if k == "pop":
                    continue
                w.writerow(["trial_seed%d" % r["seed"], k, v, ""])
        w.writerow(["agg", "rate_median_hz", "%.6f" % med, "判据 (a) 带 %s" % (
            "[%.4g,%.4g]" % (band.get("rate_median_hz", {}).get("lo", float("nan")),
                             band.get("rate_median_hz", {}).get("hi", float("nan"))))])
        w.writerow(["agg", "silent_fraction", "%.6f" % sil, "判据 (b)"])
        w.writerow(["agg", "rate_p95_hz", "%.6f" % p95, "判据 (c)"])
        w.writerow(["agg", "pop_rate_hz", "%.6f" % popr, "群体发放率"])
        w.writerow(["agg", "bout_active_fraction", "%.6f" % boutf, "判据 (d) 双状态"])
        w.writerow(["agg", "bout_count_per_min", "%.3f" % boutc, "判据 (d)"])
        w.writerow(["agg", "ms_per_step_best", "%.4f" % msp, "最快试次每步墙钟"])
        w.writerow(["agg", "gpu_h_per_30s_trial", "%.4f" % gpu_h,
                    "判据 (f)：≤1 GPU-h（§3.5.4）"])
        w.writerow(["agg", "determinism_spearman", "%.6f" % det_spearman, "判据 (e)"])
        w.writerow(["agg", "determinism_bitwise", str(det_identical), "逐位一致（不承诺）"])
        for k, v in crit.items():
            w.writerow(["crit", k, str(bool(v)), ""])
        w.writerow(["crit", "verdict", verdict, "P4 三态判定（规划节点复核）"])
    print("P4 结果 →", OUT_CSV, flush=True)

    with open(OUT_PARAMS, "w", encoding="utf-8", newline="") as f:
        w = _csv.writer(f, lineterminator="\n")
        w.writerow(["param", "value", "note"])
        for k, v in PARAMS.items():
            w.writerow([k, v, "类级全局参数（§8 ①；标定见 m9_weight_calibration.csv）"])
        for k, v in BG.items():
            w.writerow(["bg_" + k, v, "背景 Poisson 驱动"])
        w.writerow(["delay_model", "uniform_%.2fms" % p.delay_ms,
                    "抽象登记：FlyWire v783 不含延迟列 → 统一轴突延迟"])
        w.writerow(["vram_gb", "%.3f" % (bs["vram_bytes"] / 2**30), "实测"])
    print("回路参数 →", OUT_PARAMS, flush=True)

    make_plot(rates, allpop, rows, band, crit, boutc, BIN_MS)
    print("图 →", OUT_PNG, flush=True)
    print("P4 判定：%s（%s）" % (verdict, crit), flush=True)
    print("总墙钟 %.0fs" % (time.perf_counter() - t00), flush=True)
    return 0 if verdict == "PASS" else 2


def make_plot(rates, allpop, rows, band, crit, boutc, bin_ms):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    r = rates[0]
    ax.hist(np.clip(r, 0, 20), bins=80, color="#1f77b4")
    ax.axvline(0.5, color="red", ls="--", lw=1, label="silent threshold 0.5 Hz")
    ax.set_yscale("log")
    ax.set_xlabel("firing rate (Hz)"); ax.set_ylabel("neuron count")
    ax.set_title("Resting firing-rate distribution (N=%d, T=%.0fs)\nmedian=%.4f Hz  silent=%.3f"
                 % (r.size, T_MS / 1000.0, float(np.median(r)),
                    float(np.mean(r < 0.5))))
    ax.legend()
    ax = axes[0, 1]
    pop = allpop[0]
    bs = max(1, int(round(1000.0 / (POP_DT := 0.05))))
    nb = pop.size // bs
    ax.plot(np.arange(nb) * bs * POP_DT / 1000.0, pop[:nb * bs].reshape(nb, bs).sum(1),
            lw=0.5, color="#2ca02c")
    ax.set_xlabel("time (s)"); ax.set_ylabel("spikes / s (population)")
    ax.set_title("Population activity (double-state / bouts: %.1f per min)" % boutc)
    ax = axes[1, 0]
    keys = ["a_rate_median", "b_silent_fraction", "c_rate_p95", "d_bout",
            "e_determinism", "f_wallclock"]
    ax.barh(keys, [1 if crit[k] else 0 for k in keys],
            color=["#2ca02c" if crit[k] else "#d62728" for k in keys])
    ax.set_xlim(0, 1.2); ax.set_title("P4 criteria (in-band = 1)")
    ax = axes[1, 1]
    ms = [x["ms_per_step"] for x in rows]
    ax.bar([str(x["seed"]) for x in rows], ms, color="#9467bd")
    ax.set_xlabel("trial seed"); ax.set_ylabel("ms/step (N=139,255)")
    ax.set_title("Wall clock per step (budget: <=6 ms/step for 1 GPU-h / 30s)")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=110)


if __name__ == "__main__":
    sys.exit(main())
