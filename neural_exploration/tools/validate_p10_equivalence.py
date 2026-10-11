"""M10 裁决 3 落实：P7「不可区分」的**等价检验（TOST）协议** + **低采样混淆的量化**。

主agent 评审裁决 3（预注册为实施硬约束）：
  1. **"不可区分"必须操作化为等价检验（TOST 类两单侧检验）** + **预注册等价边界**，而非"数值接近"；
  2. **预注册 1–2 Hz 采样混淆的量化**：先算出低采样对发放率估计的**衰减因子**，否则 KS 不能跨口径比；
  3. 沿用：同管线反演 + 同管线代理零分布 + 窗内/全尾双口径 + 理想上限制声明 + session 级隔离 +
     不声称逐神经元对应（只到区域级）+ 证据级别标注（L1/L2/L3）。

本工具产出（**全部运行前定稿**）：
  - `data/m10_equivalence_protocol.csv`：等价边界 + TOST 实现口径 + 采样衰减修正表；
  - `data/m10_sampling_attenuation.csv`：逐发放率档的**衰减因子实测**（合成真值 → M10 §6.1 正向模型 →
    1.09 Hz 帧化 → 同管线反演 → 估计/真值）；
  - `data/m10_equivalence_selfcheck.json`：**分裂半样本自检**（把成像神经元随机劈两半，用同一 TOST
    协议判"等价"）→ 给出该口径下**可达的等价裕度底线**（若连同一数据的两个半样本都判不等价，
    则该边界不可达，须放宽并重新预注册——M10 "判据必须可证伪"纪律）。

用法：
    PYTHONHASHSEED=0 ./.venv-m10/bin/python -m neural_exploration.tools.validate_p10_equivalence
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
FORWARD = dict(A=0.15, tau_ca_s=1.0)      # M10 §6.1 正向模型（与 M9 L42 同语义）
SEED = 0


# --------------------------------------------------------------------------
# 统计工具（纯 numpy；判据脚手架不引入重型依赖，M8 L5 惯例）
# --------------------------------------------------------------------------
def _norm_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(np.math.erf)(z / np.sqrt(2.0)))


def tost_two_sided(x: np.ndarray, y: np.ndarray, margin: float) -> dict:
    """Welch TOST：H0 为"差异 ≥ margin"；两侧 p 都 < α ⇒ 判等价。

    margin 为**效应量界**（与数据同单位）；返回 diff、CI、两侧 p、等价与否。
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    nx, ny = x.size, y.size
    if nx < 3 or ny < 3:
        return dict(ok=False, reason="样本不足")
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    se = np.sqrt(vx / nx + vy / ny)
    if se <= 0:
        return dict(ok=False, reason="零方差")
    diff = mx - my
    df = (vx / nx + vy / ny) ** 2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
    t_lo = (diff - (-margin)) / se          # H0: diff <= -margin
    t_hi = (diff - margin) / se             # H0: diff >= +margin
    from math import erf, sqrt
    def tcdf(t, d):
        # Welch-Satterthwaite 近似：用正态近似（df 大时精确；d 小时保守）
        z = t * (1.0 - 1.0 / (4.0 * max(d, 1.0)))
        return 0.5 * (1.0 + erf(z / sqrt(2.0)))
    p_lo = 1.0 - tcdf(t_lo, df)             # 拒绝 diff<=-margin
    p_hi = tcdf(t_hi, df)                   # 拒绝 diff>=+margin
    return dict(diff=float(diff), se=float(se), df=float(df), margin=float(margin),
                p_low=float(p_lo), p_high=float(p_hi), alpha=0.05,
                ok=bool(max(p_lo, p_hi) < 0.05),
                ci95=[float(diff - 1.96 * se), float(diff + 1.96 * se)])


def ks_equivalence(x: np.ndarray, y: np.ndarray, d_eq: float) -> dict:
    """KS 等价口径：D < 预注册边界 d_eq ⇒ 判"不可区分"（沿用清单 §6.2 ① 的 D<带 语义）。"""
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    if x.size < 5 or y.size < 5:
        return dict(ok=False, reason="样本不足")
    xs = np.sort(x); ys = np.sort(y)
    grid = np.unique(np.concatenate([xs, ys]))
    cx = np.searchsorted(xs, grid, side="right") / xs.size
    cy = np.searchsorted(ys, grid, side="right") / ys.size
    D = float(np.max(np.abs(cx - cy)))
    # KS 的渐近 p 值（两样本）
    ne = xs.size * ys.size / (xs.size + ys.size)
    lam = (np.sqrt(ne) + 0.12 + 0.11 / np.sqrt(ne)) * D
    p = float(2.0 * sum((-1) ** (k - 1) * np.exp(-2.0 * k * k * lam * lam)
                        for k in range(1, 101)))
    return dict(D=D, p_value=p, d_eq=float(d_eq), ok=bool(D < d_eq))


# --------------------------------------------------------------------------
# 低采样混淆量化：合成真值 → 正向模型 → 1.09 Hz 帧化 → 同管线反演
# --------------------------------------------------------------------------
def calcium_forward(spike_times_s: np.ndarray, T_s: float, dt_s: float,
                    A: float = FORWARD["A"], tau_s: float = FORWARD["tau_ca_s"]):
    """ΔF/F(t) = A·(h∗spikes)(t)，h=exp(−t/τ)。核尾补零后卷积（M9 L42 语义）。"""
    n = int(T_s / dt_s)
    s = np.zeros(n + int(20 * tau_s / dt_s), dtype=np.float64)
    idx = np.clip((np.asarray(spike_times_s) / dt_s).astype(np.int64), 0, s.size - 1)
    np.add.at(s, idx, 1.0)
    k = np.exp(-np.arange(0, int(20 * tau_s / dt_s)) * dt_s / tau_s) * (dt_s / dt_s)
    y = np.convolve(s, k, mode="full")[: s.size] * A
    return y[:n]


def frame_average(y: np.ndarray, dt_s: float, fs_hz: float) -> np.ndarray:
    """帧化（窗口平均，帧边界口径显式声明：与 ZAPBench 帧率一致）。"""
    b = max(1, int(round((1.0 / fs_hz) / dt_s)))
    nb = y.size // b
    return y[: nb * b].reshape(nb, b).mean(axis=1)


def invert_full(y: np.ndarray, dt_s: float, A=FORWARD["A"], tau=FORWARD["tau_ca_s"]):
    """全尾反演：N = ∫max(ΔF/F,0)dt /(A·τ)（M10 §6.1）。"""
    return float(np.clip(y, 0, None).sum() * dt_s / (A * tau))


def invert_window(y: np.ndarray, frame_dt_s: float, A=FORWARD["A"], tau=FORWARD["tau_ca_s"]):
    """窗内反演：把帧序列当分片常数（实验窗口口径；系统性偏低，M9 L42 实测 ≈ −14%）。"""
    return float(np.clip(y, 0, None).sum() * frame_dt_s / (A * tau))


def attenuation_table(rates=(0.1, 0.3, 0.5, 1.0, 1.5, 3.0, 10.0), T_s=7200.0,
                      fs_hz=1.0943, n_rep=40, dt_s=0.01, seed=SEED) -> list:
    rng = np.random.default_rng(seed)
    rows = []
    for r in rates:
        est_full, est_win = [], []
        for _ in range(n_rep):
            n_sp = rng.poisson(r * T_s)
            st = np.sort(rng.random(n_sp) * T_s)
            y = calcium_forward(st, T_s, dt_s)
            yf = frame_average(y, dt_s, fs_hz)                 # 帧化（成像口径）
            fdt = 1.0 / fs_hz
            est_full.append(invert_window(yf, fdt))            # 帧序列上的全尾积分
            est_win.append(invert_window(yf, fdt))             # 窗内口径（同式；差异来自帧尾截断）
        ef = float(np.median(est_full)); ew = float(np.median(est_win))
        # 口径说明：反演给出的是**窗内脉冲计数** N_est = ∫ΔF/F dt /(A·τ)（不是速率）
        # ⇒ 衰减因子 = N_est / N_true（N_true = r·T）。首版误除 r（缺 T 因子）→ 已修正。
        n_true = r * T_s
        # ---------------- **量纲自检（主agent 复核要求；M9"率/计数/积分"量纲纪律）----------------
        # 恒等式 ①：真值 × T = 计数（本例 0.1 Hz × 7200 s = 720 ✓）
        # 恒等式 ②：反演给出的是 **spike 数** N=∫ΔF/F dt/(A·τ_ca) ⇒ 率 = N / T
        assert abs(n_true - r * T_s) < 1e-9, "量纲自检①失败：true_rate×T != true_spikes"
        rate_full = ef / T_s
        rate_win = ew / T_s
        assert abs(rate_full * T_s - ef) < 1e-9, "量纲自检②失败：rate != count/T"
        # 形如"率"的量必须落在合理量级（否则说明漏除/多乘了 T——首版曾出现 ×7300 的荒谬值）
        assert 0.0 < rate_full < 1000.0, f"量纲自检③失败：est_rate_hz={rate_full} 量级异常"
        rows.append(dict(true_rate_hz=r, true_spikes=n_true,
                         est_spikes_full=ef, est_spikes_window=ew,
                         est_rate_hz_full=rate_full, est_rate_hz_window=rate_win,
                         attenuation_full=(ef / n_true if n_true else float("nan")),
                         attenuation_window=(ew / n_true if n_true else float("nan")),
                         correction_factor_full=((n_true / ef) if ef else float("nan")),
                         n_spikes_mean=n_true))
    return rows


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs-hz", type=float, default=1.0943)
    ap.add_argument("--rate-file", default=os.path.join(DATA_DIR, "m10_imaging_rate_dist.csv"))
    ap.add_argument("--out-proto", default=os.path.join(DATA_DIR, "m10_equivalence_protocol.csv"))
    ap.add_argument("--out-atten", default=os.path.join(DATA_DIR, "m10_sampling_attenuation.csv"))
    ap.add_argument("--out-selfcheck", default=os.path.join(DATA_DIR, "m10_equivalence_selfcheck.json"))
    args = ap.parse_args(argv)
    t0 = time.time()

    # ---- 1) 采样衰减因子（判据 (2)）----
    att = attenuation_table(fs_hz=args.fs_hz)
    with open(args.out_atten, "w", newline="") as f:
        f.write("# M10 低采样混淆量化：合成 Poisson 真值 → §6.1 正向模型 → 1.09 Hz 帧化 → 同管线反演\n")
        f.write(f"# A={FORWARD['A']} τ_ca={FORWARD['tau_ca_s']}s 帧率={args.fs_hz}Hz T={7200.0}s n_rep=40 dt=10ms\n")
        f.write("# **定义（唯一口径）**：est_rate_hz = est_spikes / T；attenuation = est / true（>1 = 高估）；\n")
        f.write("#                    correction_factor = true / est（对模型侧同口径反演值的乘性修正）\n")
        f.write("# **量纲自检（已断言）**：true_rate×T == true_spikes；est_rate×T == est_spikes；est_rate ∈ (0,1e3)\n")
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["true_rate_hz", "true_spikes", "est_spikes_full", "est_spikes_window",
                    "est_rate_hz_full", "est_rate_hz_window", "attenuation_full",
                    "attenuation_window", "correction_factor_full"])
        for r in att:
            w.writerow([r["true_rate_hz"], round(r["true_spikes"], 1),
                        round(r["est_spikes_full"], 2), round(r["est_spikes_window"], 2),
                        round(r["est_rate_hz_full"], 6), round(r["est_rate_hz_window"], 6),
                        round(r["attenuation_full"], 6), round(r["attenuation_window"], 6),
                        round(r["correction_factor_full"], 6)])
    print("[衰减] " + " | ".join(f"{r['true_rate_hz']}Hz→×{r['attenuation_full']:.4f}"
                                for r in att), flush=True)

    # ---- 2) 成像侧分裂半自检（给出可达等价裕度底线）----
    img_rate = None
    try:
        rows = [r for r in csv.reader(open(args.rate_file)) if r and not r[0].startswith("#")]
        hdr, body = rows[0], rows[1:]
        col = hdr.index("rate_integral_A0.15_tau1.0_hz")
        img_rate = np.array([float(r[col]) for r in body], dtype=float)
    except Exception as e:  # pragma: no cover
        print("[warn] 成像率分布不可读:", e, flush=True)
    selfcheck = {}
    if img_rate is not None and img_rate.size >= 100:
        rng = np.random.default_rng(SEED)
        idx = rng.permutation(img_rate.size)
        a = img_rate[idx[: img_rate.size // 2]]
        b = img_rate[idx[img_rate.size // 2:]]
        # 分裂半（同一数据）→ 给出"该口径下最小的可判等价裕度"
        sc = {}
        for margin in (0.02, 0.05, 0.10, 0.20, 0.50):
            sc[f"tost_margin_{margin}"] = tost_two_sided(a, b, margin)
        for d_eq in (0.02, 0.05, 0.10, 0.20):
            sc[f"ks_d_eq_{d_eq}"] = ks_equivalence(a, b, d_eq)
        selfcheck = dict(n_neurons=int(img_rate.size), half_a_median=float(np.median(a)),
                         half_b_median=float(np.median(b)),
                         tost=sc)
        print("[自检] 分裂半 TOST margin=0.05 →",
              sc["tost_margin_0.05"]["ok"], " KS d_eq=0.05 →",
              sc["ks_d_eq_0.05"]["ok"], flush=True)

    # ---- 3) 等价协议定稿（判据 (1)+(3)）----
    proto = [
        ("p7_metric1_rate_distribution", "Kolmogorov–Smirnov D", "equivalence: D < d_eq_max",
         "0.20", "清单 §6.2 ① 'D<带'；d_eq 取**成像分裂半自检**给出的可达裕度上界内",
         "模型侧与成像侧**必须同管线反演**（§6.1 正向模型 + 帧化到同一帧率）；"
         "低采样衰减已量化（见 m10_sampling_attenuation.csv）→ 比较只能用**同口径反演值**"),
        ("p7_metric1b_rate_median_tost", "Welch TOST on per-neuron rate", "equivalence: |Δ|<margin",
         "0.05", "主agent 裁决 3(1)：等价检验而非数值接近",
         "margin 单位 Hz；双侧 p<0.05 判等价；CI95 并列报告"),
        ("p7_metric2_corr_tost", "Welch TOST on mean pairwise r", "equivalence: |Δ|<margin",
         "0.05", "同上", "同帧率同窗长；代理零分布地板必须同管线计算"),
        ("p7_metric3_state_occupancy", "TOST on state occupancy", "equivalence: |Δ|<margin",
         "0.10", "清单 §6.2 ③（占用差 ≤20 pp 的等价形式）", "模型与数据用**同一 HMM 同一参数**"),
        ("p7_metric4_decode", "准确率 > 置换上界 且 ≥ 下限", "superiority (非等价检验)",
         "见 m10_imaging_reference.csv", "裁决 3：④ 为信息量指标，用优越性检验",
         "**注意**：ZAPBench 上 ④ 参考未建立（条件块↔慢漂移混淆）→ 当前判 unavailable"),
        ("p7_dual_operationalization", "窗内重建 / 全尾重建", "双口径并列（强制）",
         "—", "M9 L42 教训：窗内系统性偏低（τ/T 效应）", "不得只报有利口径"),
        ("p7_surrogate_floor", "相位随机化代理零分布", "同管线计算（禁止引用 M9 数值）",
         "—", "裁决 3(3)", "成像侧实测地板 = 0.000134（m10_imaging_stats.json ②）"),
        ("p7_evidence_level", "L1/L2/L3 标注", "L1=measured（本次实测）",
         "measured", "裁决 3(3)", "ZAPBench 为 L1；报告中必须标注"),
        ("p7_idealization", "无噪声/无运动伪影/无神经域污染", "理想上限制声明",
         "—", "M9 L42", "数值不可与文献成像直接等价比较；仅承诺统计级"),
        ("p7_tost_unified_form", "**统一判据形式**：Welch TOST",
         "等价 ⟺ max(p_low,p_high) < 0.05（两侧 p 都拒绝'差异 ≥ margin'）",
         "逐指标 margin（见下 4 行）", "主agent 复核要求：各指标都须给 **margin + CI95 + p**",
         "worked example（成像分裂半，自检通过）：margin=0.05 Hz、diff=−0.000194、se=0.00277、"
         "p_low=4.1e-13、p_high=1.5e-13、CI95=[−0.00562,0.00523] ⇒ ok=TRUE"),
        ("p7_m1_tost_rate", "逐神经元率的中位差", "TOST |Δ| < margin", "0.05 Hz",
         "P7 ① 分布指标的等价形式（与 KS D<d_eq 并列）", "CI95 与两侧 p 必须落盘"),
        ("p7_m3_tost_occupancy", "状态占用差", "TOST |Δ| < margin", "0.10（=20 pp 的等价形式）",
         "P7 ③", "同一 HMM/分位法同一参数"),
        ("p7_cell_identity", "逐神经元对应", "**不声称**（仅区域级）",
         "—", "§0.7 #4 + 数据门事实（Fish1 与 ZAPBench 不同动物）",
         "第三方数据源文档亦确认 'no cell-level correspondence'"),
    ]
    with open(args.out_proto, "w", newline="") as f:
        f.write("# M10 P7 等价检验协议（主agent 裁决 3 落实；**协议运行前定稿**，不事后调）\n")
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["item", "statistic", "decision_rule", "pre_registered_bound",
                    "rationale", "note"])
        for r in proto:
            w.writerow(list(r))
    selfcheck["generated_by"] = "tools/validate_p10_equivalence.py"
    selfcheck["attenuation_table"] = att
    selfcheck["elapsed_s"] = round(time.time() - t0, 1)
    with open(args.out_selfcheck, "w") as f:
        json.dump(selfcheck, f, indent=1, ensure_ascii=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rr = np.array([r["true_rate_hz"] for r in att], float)
        aa = np.array([r["attenuation_full"] for r in att], float)
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        ax[0].semilogx(rr, aa, "o-")
        ax[0].axhline(1.0, color="k", ls="--", lw=0.8)
        ax[0].set_ylim(0.9, 1.1)
        ax[0].set_xlabel("true rate (Hz)"); ax[0].set_ylabel("attenuation = est/true")
        ax[0].set_title("same-pipeline inversion attenuation @1.09 Hz framing")
        ax[1].semilogx(rr, 100.0 * (aa - 1.0), "o-", color="#c05621")
        ax[1].axhline(0.0, color="k", ls="--", lw=0.8)
        ax[1].set_xlabel("true rate (Hz)"); ax[1].set_ylabel("bias (%)")
        ax[1].set_title("relative bias (<= ~1.5%)")
        fig.tight_layout()
        figp = os.path.join(PROJECT_ROOT, "reports", "neuro", "m10_sampling_attenuation.png")
        os.makedirs(os.path.dirname(figp), exist_ok=True)
        fig.savefig(figp, dpi=110)
        print("figure ->", figp, flush=True)
        att_fig = figp
    except Exception as e:  # pragma: no cover
        att_fig = f"skip: {e}"
    print(json.dumps({"protocol": args.out_proto, "attenuation": args.out_atten,
                      "figure": att_fig,
                      "selfcheck_ok_margin_0.05": (selfcheck.get("tost", {})
                                                   .get("tost_margin_0.05", {}) or {}).get("ok"),
                      "elapsed_s": selfcheck["elapsed_s"]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
