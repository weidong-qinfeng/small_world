"""M9 P10：活动金标准的正向模型验证（发放 → GCaMP ΔF/F → 统计级对照）。

《生物仿真M9实施清单》§4.6（P10）/§0.3.5（抽象登记）/§0.7 #8（判据带定稿于 CSV 不事后调）。

**数据诚实（R9 前置声明）**：本机**成虫全脑钙成像 / 电生理数据不可得** →
  - 参考分布为**文献参数化回退**（对数正态，中位 1 Hz / σ_log 1.4），provenance 入档；
  - 只承诺**统计级**结论（分布形状 / 相关性量级 / 静默比例），**不声称逐神经元成像对应**；
  - 正向模型为**理想化线性 GCaMP**（无噪声、无运动伪影、无神经域污染）→ 结果登记为上限。

单次运行同时产出（无额外预算）：真发放率分布、ΔF/F 帧序列、成像式重建率、
成对相关（2 Hz 与 1 Hz 帧率、2 s 与 5 s 窗长——**窗口/核混淆诊断**）、相位随机化代理零分布、
Fano 因子、确定性复跑逐位比对。

输出：`data/m9_p10_activity.csv` + `reports/neuro/m9_activity_stats.png`
用法：PYTHONHASHSEED=0 MPLBACKEND=Agg ./.venv-m9/bin/python -u -m neural_exploration.tools.validate_p9_activity
"""

from __future__ import annotations

import csv as _csv
import hashlib
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports", "neuro")
REF_CSV = os.path.join(DATA_DIR, "m9_behavior_reference.csv")
OUT_CSV = os.path.join(DATA_DIR, "m9_p10_activity.csv")
OUT_PNG = os.path.join(REPORT_DIR, "m9_activity_stats.png")
OUT_LOG = os.path.join(REPORT_DIR, "m9_p10_progress.log")

T_MS = float(os.environ.get("M9_P10_T_MS", "5000"))
SETTLE_MS = float(os.environ.get("M9_P10_SETTLE_MS", "500"))
SEG_STEPS = int(os.environ.get("M9_P10_SEG", "2000"))
ABORT_S = float(os.environ.get("M9_P10_ABORT_S", "180"))
DEVICE = os.environ.get("M9_P10_DEVICE", "mps")
SEED = 0
N_PAIRS = int(os.environ.get("M9_P10_NPAIRS", "2000"))


def log(msg: str) -> None:
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(OUT_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_band(role: str) -> dict:
    out = {}
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
                                "unit": d.get("unit", ""),
                                "provenance": d.get("provenance", "")}
    return out


def main() -> int:
    from neural_exploration.src.adult_circuit import AdultCircuit, CircuitParams
    from neural_exploration.src.adult_activity import (GcampForward, GcampParams,
                                                       ks_1samp, lognormal_cdf, spearman)
    from neural_exploration.tools.validate_p9_resting import PARAMS, BG
    os.makedirs(REPORT_DIR, exist_ok=True)
    open(OUT_LOG, "w", encoding="utf-8").close()
    band = read_band("activity")
    log("=== M9 P10 活动金标准（§4.6）===")
    for k, v in band.items():
        log("  判据带 %s: [%.4g, %.4g] %s" % (k, v["lo"], v["hi"], v["unit"]))

    c = AdultCircuit(device=DEVICE, params=CircuitParams(**PARAMS), use_compile=False)
    c.build()
    pt = c.engine.point
    rb = {"e_inh": float(pt.e_inh), "ek": float(pt.ek), "ahp_form": str(pt.ahp_form),
          "ahp_g_inc": float(pt.ahp_g_inc), "v_floor": float(pt.v_floor)}
    assert rb["e_inh"] == -80.0 and rb["ek"] == -77.0 and rb["ahp_g_inc"] == 25.0, rb
    log("参数读回断言 ✓ %s" % rb)
    g_ext = BG["epsp_mv"] / 0.104
    n_steps = int(round((SETTLE_MS + T_MS) / PARAMS["dt_ms"])) + 8
    c.build_background(rate_hz=BG["rate_hz"], g_ext=g_ext, n_steps=n_steps, seed=BG["seed"])
    c._bg_steps, c._bg_args = n_steps, (BG["rate_hz"], g_ext)

    def record_run(tag, seed=SEED, record=True):
        """分段运行 + 逐 spike 记录（record='full'）；返回 (steps, idx, counts, wall)。"""
        p = c.params
        n_settle = int(round(SETTLE_MS / p.dt_ms))
        n_meas = int(round(T_MS / p.dt_ms))
        rng0 = np.random.default_rng(1)
        v0 = (p.v_rest + 2.0 * rng0.standard_normal(c.n_neurons)).astype(np.float32)
        c.engine.reset(v0=v0, seed=seed)
        done = 0
        while done < n_settle:
            m = min(SEG_STEPS, n_settle - done)
            c.engine.run(m, delivery="chunk", record="counts")
            done += m
        c.engine.reset_counts()
        S, I, off = [], [], 0
        done = 0
        t0 = time.perf_counter()
        while done < n_meas:
            m = min(SEG_STEPS, n_meas - done)
            tt = time.perf_counter()
            rr = c.engine.run(m, delivery="chunk",
                              record="full" if record else "counts")
            if time.perf_counter() - tt > ABORT_S:
                raise RuntimeError("段墙钟 %.0fs > 上限 %.0fs → 主动中止"
                                   % (time.perf_counter() - tt, ABORT_S))
            if record and rr.get("spikes") is not None:
                S.append(rr["spikes"][0] + off); I.append(rr["spikes"][1])
            off += m
            done += m
        counts = c.engine.t_count.detach().cpu().numpy().astype(np.float64)
        steps = np.concatenate(S) if S else np.zeros(0, np.int64)
        idx = np.concatenate(I) if I else np.zeros(0, np.int64)
        wall = time.perf_counter() - t0
        log("  运行 %s：%d spikes（%.0f Hz 群体）、%.0fs（%.2f ms/step）"
            % (tag, steps.size, counts.sum() / c.n_neurons / (T_MS / 1000.0), wall,
               wall / max(n_meas, 1) * 1e3))
        return steps, idx, counts, wall

    log("协议：%s、T=%.0fms、settle=%.0fms、%d 步/段、逐 spike 记录（record=full）"
        % (DEVICE, T_MS, SETTLE_MS, SEG_STEPS))
    steps, idx, counts, wall1 = record_run("main")
    g = GcampForward(GcampParams())
    r = g.analyze(steps, idx, c.n_neurons, int(round(T_MS / PARAMS["dt_ms"])),
                  PARAMS["dt_ms"], n_pairs=N_PAIRS, seed=SEED)
    log("  真率：均值 %.3f / 中位 %.3f Hz；静默 %.3f；帧数 %d（%.0f ms/帧）；可检出 %d"
        % (r["rate_mean_hz"], r["rate_median_hz"], r["silent_frac_true"], r["n_frames"],
           r["frame_ms"], r["n_detectable"]))

    # 确定性复跑（逐 spike 逐位）
    steps2, idx2, counts2, wall2 = record_run("repeat")
    det_ok = bool(np.array_equal(steps, steps2) and np.array_equal(idx, idx2)
                  and np.array_equal(counts, counts2))
    log("确定性复跑（同 seed）：逐位相同 = %s" % det_ok)

    # ---------------- 统计量 ----------------
    ks = ks_1samp(r["true_rate"], lambda x: lognormal_cdf(x, 1.0, 1.4))
    ks_rec = ks_1samp(r["rec_rate"], lambda x: lognormal_cdf(x, 1.0, 1.4))
    rho = spearman(r["rec_rate_full"], r["true_rate"])
    log("KS（真率 vs 文献对数正态 median 1Hz/σ1.4）：D=%.4f p=%.3g（带 ≤%.2f → %s）"
        % (ks["D"], ks["p"], band["ks_stat_max"]["hi"],
           "PASS" if ks["D"] <= band["ks_stat_max"]["hi"] else "FAIL"))
    log("KS（成像式重建率 vs 同参考，描述性）：D=%.4f" % ks_rec["D"])
    log("平均成对相关（2 Hz 帧 / 全窗）：%.4f（代理零分布 %.4f；可检出 %d 神经元，%d 对）"
        % (r["mean_pair_corr"], r["mean_pair_corr_surrogate"], r["n_detectable"],
           r["n_pairs_used"]))
    log("平均成对相关（100 ms 精细栅格，描述性）：%.4f" % r["mean_pair_corr_fine_100ms"])
    log("Fano 中位 %.3f / 均值 %.3f；ΔF/F 静默占比 %.3f；重建率(补零全尾) 中位 %.3f Hz、"
        "窗内偏低比 %.3f；ρ(重建,真)=%.4f"
        % (r["fano_median"], r["fano_mean"], r["silent_frac_ca"],
           r["recon_rate_median_full"], r["recon_window_bias"], rho))

    # **窗口/帧率混淆诊断**（同一记录，无额外运行）：窗长与帧率对相关估计的影响
    confuse = []
    for win_s in (2.0, 5.0):
        for fr in (1.0, 2.0):
            n_steps_w = int(round(min(win_s, T_MS / 1000.0) * 1000.0 / PARAMS["dt_ms"]))
            steps_w = steps[steps < n_steps_w]
            idx_w = idx[steps < n_steps_w]
            gg = GcampForward(GcampParams(frame_rate_hz=fr))
            rr = gg.analyze(steps_w, idx_w, c.n_neurons, n_steps_w, PARAMS["dt_ms"],
                            n_pairs=N_PAIRS, seed=SEED)
            confuse.append({"win_s": win_s, "frame_rate_hz": fr,
                            "n_frames": rr["n_frames"],
                            "mean_pair_corr": rr["mean_pair_corr"],
                            "surrogate": rr["mean_pair_corr_surrogate"],
                            "fano_median": rr["fano_median"]})
            log("  [混淆诊断] 窗 %.0fs / %.0f Hz：%d 帧，平均相关 %.4f（代理 %.4f）"
                % (win_s, fr, rr["n_frames"], rr["mean_pair_corr"],
                   rr["mean_pair_corr_surrogate"]))

    def _in(metric, val):
        b = band.get(metric)
        return (b is None) or (b["lo"] <= val <= b["hi"])

    crit = {
        "a_ks_vs_literature": bool(ks["D"] <= band["ks_stat_max"]["hi"]),
        "b_ca_tau_registered": bool(_in("ca_decay_tau_s", g.p.tau_s)),
        "c_frame_rate_registered": bool(_in("frame_rate_hz", g.p.frame_rate_hz)),
        "d_mean_pair_corr_in_band": bool(_in("mean_pairwise_corr_lo", r["mean_pair_corr"])),
        "e_fano_in_band": bool(_in("fano_factor_median", r["fano_median"])),
        "f_silent_ca_le_max": bool(r["silent_frac_ca"] <= band["silent_fraction_max"]["hi"]),
        "g_ca_rate_rank_corr": bool(rho >= band["ca_rate_rank_corr_min"]["lo"]),
        "h_determinism_bitwise": bool(det_ok),
    }
    verdict = "PASS" if all(crit.values()) else "FAIL(部分)"
    log("判定：%s（%s）" % (verdict, crit))
    n_pass = sum(crit.values())
    log("P10 判据 %d/%d 通过；总墙钟 %.0fs" % (n_pass, len(crit), wall1 + wall2))

    # ---------------- 落盘 ----------------
    T_s = T_MS / 1000.0
    rows = [["section", "key", "value", "note"],
            ["protocol", "device", DEVICE, ""],
            ["protocol", "T_ms", "%.0f" % T_MS, "成像协议窗（预算受限；登记为测量限制）"],
            ["protocol", "settle_ms", "%.0f" % SETTLE_MS, ""],
            ["protocol", "segment_steps", SEG_STEPS, "全分段"],
            ["protocol", "ca_tau_s", g.p.tau_s, "单指数核（预注册）"],
            ["protocol", "single_spike_dff", g.p.single_spike_dff, "单脉冲峰幅（预注册；Chen 2013 量级）"],
            ["protocol", "frame_rate_hz", g.p.frame_rate_hz, "预注册 1–2 Hz"],
            ["protocol", "n_frames", r["n_frames"], ""],
            ["protocol", "params_readback", str(rb), "构建后读回断言 ✓"],
            ["protocol", "reference_distribution",
             "lognormal(median=1.0 Hz, sigma_log=1.4)", "**文献参数化回退**（R9：成像数据不可得）"],
            ["measure", "rate_mean_hz", "%.4f" % r["rate_mean_hz"], ""],
            ["measure", "rate_median_hz", "%.4f" % r["rate_median_hz"], ""],
            ["measure", "silent_frac_true", "%.4f" % r["silent_frac_true"], "承接 P4 (b) 0.524"],
            ["measure", "ks_D_true_vs_lit", "%.4f" % ks["D"], "判据 (a)；p=%.3g" % ks["p"]],
            ["measure", "ks_D_ca_recon_vs_lit", "%.4f" % ks_rec["D"], "描述性（成像式重建）"],
            ["measure", "mean_pair_corr_2hz", "%.4f" % r["mean_pair_corr"], "判据 (d)"],
            ["measure", "mean_pair_corr_surrogate_null", "%.4f" % r["mean_pair_corr_surrogate"],
             "相位随机化代理（采样噪声地板对照）"],
            ["measure", "mean_pair_corr_fine_100ms", "%.4f" % r["mean_pair_corr_fine_100ms"],
             "描述性（精细时间尺度）"],
            ["measure", "fano_median", "%.4f" % r["fano_median"], "判据 (e)"],
            ["measure", "silent_frac_ca", "%.4f" % r["silent_frac_ca"], "判据 (f)"],
            ["measure", "dff_peak_median", "%.4f" % r["dff_peak_median"], ""],
            ["measure", "recon_rate_median_window", "%.4f" % r["recon_rate_median"],
             "窗内 ΔF/F 积分重建（含钙尾截断偏低偏置）"],
            ["measure", "recon_rate_median_full", "%.4f" % r["recon_rate_median_full"],
             "补零全尾重建（无边界截断；单脉冲自检精确）"],
            ["measure", "recon_window_bias", "%.4f" % r["recon_window_bias"],
             "窗内/全尾（成像式估计的系统性偏低；τ/T 效应）"],
            ["measure", "spearman_recon_vs_true_rate", "%.4f" % rho, "判据 (g)：模型自检"],
            ["measure", "n_detectable", r["n_detectable"], "真率 ≥0.5 Hz（相关估计子集；成像常规）"],
            ["measure", "wall_s", "%.0f" % (wall1 + wall2), "两次运行（主 + 确定性复跑）"]]
    for d in confuse:
        rows.append(["confound", "win%.0fs_frame%.0fHz" % (d["win_s"], d["frame_rate_hz"]),
                     "corr=%.4f surrogate=%.4f frames=%d" % (d["mean_pair_corr"],
                                                             d["surrogate"], d["n_frames"]),
                     "窗长/帧率对相关估计的混淆诊断（同一记录，无额外运行）"])
    for k, v in crit.items():
        rows.append(["crit", k, str(bool(v)), ""])
    rows.append(["crit", "verdict", verdict, "P10 三态判定（%d/%d）" % (n_pass, len(crit))])
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        f.write("# M9 P10 活动金标准（§4.6；判据带只读 m9_behavior_reference.csv activity 段）\n")
        f.write("# 成像数据不可得（R9）→ 文献参数化回退 + 只承诺统计级；不声称逐神经元成像对应\n")
        w = _csv.writer(f, lineterminator="\n")
        w.writerows(rows)
    log("→ %s" % OUT_CSV)
    make_plot(r, ks, ks_rec, rho, confuse, crit, band)
    log("→ %s" % OUT_PNG)
    log("完成标记 M9_P10_DONE verdict=%s pass=%d/%d" % (verdict, n_pass, len(crit)))
    return 0 if n_pass == len(crit) else 2


def make_plot(r, ks, ks_rec, rho, confuse, crit, band):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5))
    tr = r["true_rate"]
    ax = axes[0][0]
    ax.hist(np.log10(np.maximum(tr, 1e-3)), bins=80, color="#1f77b4")
    ax.set_xlabel("log10 firing rate (Hz)"); ax.set_title("(a) model firing-rate distribution")
    ax = axes[0][1]
    x = np.sort(tr)
    ax.plot(np.log10(np.maximum(x, 1e-3)), np.arange(1, x.size + 1) / x.size,
            label="model", color="#1f77b4")
    grid = np.logspace(-3, 2, 400)
    from neural_exploration.src.adult_activity import lognormal_cdf
    ax.plot(np.log10(grid), lognormal_cdf(grid, 1.0, 1.4), label="literature fallback",
            color="#d62728", ls="--")
    ax.set_xlabel("log10 rate (Hz)"); ax.set_ylabel("CDF")
    ax.set_title("(b) KS vs literature reference\nD=%.3f (recon D=%.3f)" % (ks["D"], ks_rec["D"]))
    ax.legend(fontsize=8)
    ax = axes[0][2]
    p = band["mean_pairwise_corr_lo"]
    ax.bar(["measured", "surrogate null"], [r["mean_pair_corr"], r["mean_pair_corr_surrogate"]],
           color=["#2ca02c" if p["lo"] <= r["mean_pair_corr"] <= p["hi"] else "#d62728", "#7f7f7f"])
    ax.axhspan(p["lo"], p["hi"], color="green", alpha=0.15, label="pre-registered band")
    ax.set_title("(c) mean pairwise correlation (2 Hz frames)"); ax.legend(fontsize=8)
    ax = axes[1][0]
    ax.scatter(np.log10(np.maximum(r["true_rate"], 1e-3)),
               np.log10(np.maximum(r["rec_rate_full"], 1e-3)), s=1, alpha=0.15)
    lim = (np.log10(np.maximum(r["true_rate"].min(), 1e-3)),
           np.log10(np.maximum(r["true_rate"].max(), 1e-3)))
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_xlabel("true log10 rate"); ax.set_ylabel("Ca-reconstructed log10 rate")
    ax.set_title("(d) imaging-style reconstruction (rho=%.3f)" % rho)
    ax = axes[1][1]
    lab = ["%.0fs/%.0fHz" % (d["win_s"], d["frame_rate_hz"]) for d in confuse]
    ax.bar(lab, [d["mean_pair_corr"] for d in confuse], color="#ff7f0e")
    ax.set_ylabel("mean pairwise corr")
    ax.set_title("(e) window/frame-rate confound on correlation")
    ax.tick_params(axis="x", labelsize=8)
    ax = axes[1][2]
    ks_ = list(crit.keys())
    ax.barh(ks_, [1 if crit[k] else 0 for k in ks_],
            color=["#2ca02c" if crit[k] else "#d62728" for k in ks_])
    ax.set_xlim(0, 1.2); ax.set_title("(f) P10 criteria (pass=1)")
    ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("M9 P10 activity gold standard (spikes -> GCaMP DeltaF/F; statistical level only; "
                 "imaging data unavailable -> literature fallback)")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=110)


if __name__ == "__main__":
    sys.exit(main())
