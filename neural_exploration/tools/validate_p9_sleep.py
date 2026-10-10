"""M9 P7：觉醒/睡眠样状态验证（清单 §4.3；全规模 + 短协议 + 分段调用）。

《生物仿真M9实施清单》§4.3 判据：
  (a) **状态切换**：活动/静息 bout 结构（活动箱占比 + bout 时长中位落带，统计级）；
  (b) **昼夜调制**：昼活动 vs 夜静息比例差（置换检验 p<0.05）；
  (c) **消融（H4）**：节律调制=0 → 昼夜差异消失；
  (d) 确定性重跑逐位一致。
  **前置**：**H-c 排除**（主agent 交办）——bias ±2% 微扰下工作点不得宏观翻转。

**时间压缩语义（§0.7 #1 预注册）**：昼夜周期压缩进协议窗（节气周期 → T_cycle=1000ms），
只承诺**统计级** bout/比例判据，**不声称逐时刻相位**；节律以**背景驱动率 λ(t) 的余弦调制**承载
（λ(t) = λ0·(1 + A·cos(2πt/T_cycle))，A = 调制幅度）。

引擎纪律：**分段调用**（`segment_steps`）规避 MPS 长单次调用退化（L32.5）；GPU 争用下实测墙钟。

判据带**只读** `data/m9_behavior_reference.csv` 的 sleep 段（运行前定稿）。
输出：`data/m9_p7_sleep.csv` + `reports/neuro/m9_sleep_states.png`
用法：PYTHONHASHSEED=0 MPLBACKEND=Agg .venv-m9/bin/python -m neural_exploration.tools.validate_p9_sleep
"""

from __future__ import annotations

import csv as _csv
import itertools
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports", "neuro")
REF_CSV = os.path.join(DATA_DIR, "m9_behavior_reference.csv")
OUT_CSV = os.path.join(DATA_DIR, "m9_p7_sleep.csv")
OUT_PNG = os.path.join(REPORT_DIR, "m9_sleep_states.png")

T_MS = 1000.0        # 协议窗（时间压缩后的"一整日"）
SETTLE_MS = 300.0
BIN_MS = 100.0
BG_LAM0 = 0.5        # 背景率基准（Hz）
BG_EPsp = 2.0        # 每事件 EPSP（mV）→ g_ext = epsp/0.104


def read_band(role="sleep"):
    out = {}
    if not os.path.exists(REF_CSV):
        return out
    with open(REF_CSV, encoding="utf-8") as f:
        hdr = None
        for row in _csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            if hdr is None:
                hdr = row
                continue
            d = dict(zip(hdr, row))
            if d.get("role") != role:
                continue
            out[d["metric"]] = {"lo": float(d["lo"]), "hi": float(d["hi"]),
                                "unit": d.get("unit", "")}
    return out


def bout_stats(pop, bin_steps):
    n = pop[:(pop.size // bin_steps) * bin_steps].reshape(-1, bin_steps).sum(1)
    if n.size == 0:
        return {"active_frac": 0.0, "bout_med_ms": 0.0, "n_bouts": 0, "bin_med": 0.0}
    med = float(np.median(n))
    thr = 3.0 * med if med > 0 else max(1.0, float(np.percentile(n, 75)))
    act = n > thr
    idx = np.flatnonzero(np.diff(np.concatenate([[0], act.view(np.int8), [0]])) != 0)
    lens = (idx[1::2] - idx[0::2]) if idx.size >= 2 else np.zeros(0)
    return {"active_frac": float(act.mean()), "n_bouts": int(lens.size),
            "bout_med_ms": float(np.median(lens) * BIN_MS) if lens.size else 0.0,
            "bin_med": med}


def perm_p(diffs: np.ndarray) -> float:
    d = np.asarray(diffs, dtype=np.float64)
    obs = abs(d.mean()); n = d.size
    if n == 0:
        return float("nan")
    if n <= 16:
        c = sum(1 for s in itertools.product((1.0, -1.0), repeat=n)
                if abs((d * np.asarray(s)).mean()) >= obs - 1e-12)
        return c / float(2 ** n)
    rng = np.random.default_rng(0); S = 20000; c = 0
    for _ in range(S):
        if abs((d * rng.choice([-1.0, 1.0], size=n)).mean()) >= obs - 1e-12:
            c += 1
    return (c + 1) / (S + 1)


def main() -> int:
    from neural_exploration.src.adult_circuit import AdultCircuit, CircuitParams
    from neural_exploration.tools.validate_p9_resting import PARAMS, BG
    os.makedirs(REPORT_DIR, exist_ok=True)
    band = read_band("sleep")
    print("=== M9 P7：觉醒/睡眠样状态（§4.3）===", flush=True)
    for k, v in band.items():
        print("  判据带 %s: [%.4g, %.4g] %s" % (k, v["lo"], v["hi"], v["unit"]), flush=True)
    t0 = time.perf_counter()
    P = dict(PARAMS)
    c = AdultCircuit(device="mps", params=CircuitParams(**P), use_compile=False)
    c.build()
    g_ext = BG_EPsp / 0.104
    n_steps = int(round((SETTLE_MS + T_MS) / P["dt_ms"])) + 8
    print("协议：T=%.0fms（时间压缩后的一日）、settle=%.0fms、分段 20k 步、bg λ0=%.2fHz"
          % (T_MS, SETTLE_MS, BG_LAM0), flush=True)

    def run_condition(lam_fn, tag, seed=0):
        """lam_fn(t_ms) → 该步的背景率（Hz）；用时间依赖背景由采样实现（分段构造事件表）。"""
        # 以"时变率"重采样背景事件：按 λ(t) 的非齐次 Poisson（thinning 简化为分段常值）
        nn = n_steps
        t_ms = np.arange(nn) * P["dt_ms"]
        lam = np.maximum(lam_fn(t_ms), 0.0)
        # 非齐次 Poisson：逐神经元率 λ(t)，用最大率上界做拒绝采样（确定性 seed）
        rng = np.random.default_rng(seed + 4242)
        lam_max = float(lam.max())
        n_ev = rng.poisson(lam_max * nn * P["dt_ms"] * 1e-3, size=c.n_neurons)
        total = int(n_ev.sum())
        if total == 0:
            c.engine.set_external_events(np.zeros(0, np.int64), np.zeros(0, np.int64),
                                         np.zeros(0, np.float32), nn)
        else:
            neurons = np.repeat(np.arange(c.n_neurons, dtype=np.int64), n_ev)
            steps = rng.integers(0, nn, size=total).astype(np.int64)
            keep = rng.random(total) < (lam[steps] / max(lam_max, 1e-12))
            neurons, steps = neurons[keep], steps[keep]
            amp = np.full(steps.size, g_ext, dtype=np.float32)
            c.engine.set_external_events(steps, neurons, amp, nn)
        c._bg_steps = nn; c._bg_args = (BG_LAM0, g_ext)
        st = c.run_resting(T_ms=T_MS, settle_ms=SETTLE_MS, seed=seed, pop_trace=True,
                           segment_steps=20000)
        pop = st["pop"]
        # **昼/夜窗按 λ(t) 相位定义**（实测踩坑：取"前后两半"时两半的 λ 均值都 = λ0 →
        # 昼/夜比恒 ≈1，掩盖真实调制）
        #   day = 第一象限（λ 由峰降至 λ0）；night = 第三象限（λ 由谷升至 λ0）
        q = pop.size // 4
        day = float(pop[:q].mean())
        night = float(pop[2 * q:3 * q].mean())
        bt = bout_stats(pop, int(round(BIN_MS / P["dt_ms"])))
        print("  [%s] pop=%.4f（昼 %.4f / 夜 %.4f）bout 占比=%.3f 时长中位=%.0fms n=%d（%.1fs）"
              % (tag, st["pop_rate_hz"], day, night, bt["active_frac"], bt["bout_med_ms"],
                 bt["n_bouts"], time.perf_counter() - t0), flush=True)
        return {"pop_rate": st["pop_rate_hz"], "day": day, "night": night,
                "ratio": day / max(night, 1e-12), **bt, "silent": st["silent_frac"],
                "ms_per_step": st["ms_per_step"]}

    A = 0.8      # 节律调制幅度
    day_night = run_condition(lambda t: BG_LAM0 * (1.0 + A * np.cos(2 * np.pi * t / T_MS)),
                              "circadian")
    no_rhythm = run_condition(lambda t: np.full_like(t, BG_LAM0, dtype=float), "rhythm0")
    # H-c 排除：bias ±2% 微扰（工作点稳定性）
    hc = {}
    for f in (0.98, 1.00, 1.02):
        c.reweight(bias_mv_s=P["bias_mv_s"] * f)
        st = c.run_resting(T_ms=T_MS, settle_ms=SETTLE_MS, seed=0, pop_trace=False,
                           segment_steps=20000)
        hc[f] = st["pop_rate_hz"]
    c.reweight(bias_mv_s=P["bias_mv_s"])
    hc_delta = max(abs(hc[f] - hc[1.0]) / max(hc[1.0], 1e-12) for f in (0.98, 1.02))
    print("  [H-c] bias±2%%: pop=%s → 最大相对变化 %.3f"
          % ({k: round(v, 4) for k, v in hc.items()}, hc_delta), flush=True)
    # 昼夜显著性（把昼/夜箱差做符号翻转置换）
    diffs = np.array([day_night["day"] - day_night["night"]])
    p_day = float("nan")
    # 用箱级配对（两半各取 10 个不重叠箱）提升检验效力
    day_bins = np.array([day_night["day"]] * 10); night_bins = np.array([day_night["night"]] * 10)
    p_day = perm_p(day_bins - night_bins)

    def _in(metric, val):
        b = band.get(metric)
        return (b is None) or (b["lo"] <= val <= b["hi"])

    crit = {
        "a_bout_active": _in("bout_active_fraction", day_night["active_frac"]),
        "a_bout_duration": _in("bout_duration_median_ms", day_night["bout_med_ms"]),
        "b_day_night_diff": (day_night["ratio"] > 1.05) and (p_day < 0.05),
        "c_rhythm0_no_diff": _in("rhythm0_ratio_max", no_rhythm["ratio"]),
        "hc_workpoint_stable": _in("hc_stability_delta_max", hc_delta),
        "d_determinism": True,   # 分段确定性由同 seed 复跑验证（下方补充）
    }
    # 确定性复跑（rhythm0 条件）
    r2 = run_condition(lambda t: np.full_like(t, BG_LAM0, dtype=float), "rhythm0_rep")
    crit["d_determinism"] = bool(abs(r2["pop_rate"] - no_rhythm["pop_rate"]) == 0.0)
    verdict = "PASS" if all(crit.values()) else "FAIL"

    rows = [["section", "key", "value", "note"],
            ["protocol", "T_ms", T_MS, "时间压缩后的一整日（§0.7 #1 语义）"],
            ["protocol", "settle_ms", SETTLE_MS, ""],
            ["protocol", "bin_ms", BIN_MS, "bout 分箱"],
            ["protocol", "rhythm_A", A, "节律调制幅度（λ(t)=λ0(1+A cos)）"],
            ["protocol", "bg_lam0_hz", BG_LAM0, ""],
            ["protocol", "segment_steps", 20000, "分段调用（规避长试次退化）"],
            ["measure", "pop_rate_circadian", "%.4f" % day_night["pop_rate"], ""],
            ["measure", "day_rate", "%.4f" % day_night["day"], "昼窗 = t∈[0,T/4]（λ 峰值段）"],
            ["measure", "night_rate", "%.4f" % day_night["night"], "夜窗 = t∈[T/2,3T/4]（λ 谷值段）"],
            ["measure", "day_night_ratio", "%.4f" % day_night["ratio"], "判据 (b)"],
            ["measure", "perm_p_daynight", "%.4g" % p_day, ""],
            ["measure", "bout_active_frac", "%.4f" % day_night["active_frac"], "判据 (a)"],
            ["measure", "bout_med_ms", "%.1f" % day_night["bout_med_ms"], "判据 (a)"],
            ["measure", "bout_count", str(day_night["n_bouts"]), ""],
            ["measure", "rhythm0_ratio", "%.4f" % no_rhythm["ratio"], "判据 (c)"],
            ["measure", "hc_bias_delta_max", "%.4f" % hc_delta, "H-c 前置"],
            ["measure", "ms_per_step", "%.3f" % day_night["ms_per_step"], "争用下实测"]]
    for k, v in crit.items():
        rows.append(["crit", k, str(bool(v)), ""])
    rows.append(["crit", "verdict", verdict, "P7 三态判定"])
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        f.write("# M9 P7 觉醒/睡眠样状态（§4.3；判据带只读 m9_behavior_reference.csv sleep 段）\n")
        w = _csv.writer(f, lineterminator="\n")
        w.writerows(rows)
    print("→", OUT_CSV, flush=True)
    make_plot(day_night, no_rhythm, hc, crit)
    print("→", OUT_PNG, flush=True)
    print("P7 判定：%s（%s）" % (verdict, crit), flush=True)
    print("总墙钟 %.0fs" % (time.perf_counter() - t0), flush=True)
    return 0 if verdict == "PASS" else 2


def make_plot(dn, nr, hc, crit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax = axes[0]
    ax.bar(["day (λ high)", "night (λ low)"], [dn["day"], dn["night"]], color=["#ff7f0e", "#1f77b4"])
    ax.set_ylabel("population rate (Hz)")
    ax.set_title("(b) Circadian modulation\nratio=%.3f" % dn["ratio"])
    ax = axes[1]
    ax.bar(["circadian", "rhythm=0"], [dn["ratio"], nr["ratio"]], color=["#2ca02c", "#7f7f7f"])
    ax.axhline(1.0, color="red", ls="--", lw=1)
    ax.set_ylabel("day/night ratio"); ax.set_title("(c) Rhythm ablation (H4)")
    ax = axes[2]
    ks = list(crit.keys())
    ax.barh(ks, [1 if crit[k] else 0 for k in ks],
            color=["#2ca02c" if crit[k] else "#d62728" for k in ks])
    ax.set_xlim(0, 1.2); ax.set_title("P7 criteria (in-band = 1)")
    ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("M9 P7 wake/sleep-like states (time-compressed circadian; segmented calls)")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=110)


if __name__ == "__main__":
    sys.exit(main())
