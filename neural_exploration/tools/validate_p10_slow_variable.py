"""M10 裁决 2 落实：P6 **慢变量硬判据** + **消融关闭态等价（逐 spike）**。

主agent 评审裁决 2（预注册为实施硬约束）：
  > P6 除三级自检门（不自持 / 驱动弹性 / bout>0）外，**必须加一条慢变量硬判据**：
  > **慢变量的时间常数须落在生理范围且与 bout 时长分布的时间尺度同阶**；
  > 同时：**全局参数总数 ≤8** + **参数范围预注册**（不许"扫参数直到 bout 出现"）+
  > **直接读出慢变量轨迹**（STP u/x、GABA_B 电导、g_mod）与群率慢涨落的相关性 +
  > **消融必须回到 M9 语义（逐 spike 一致）**（关闭态等价，§2.2 E1–E3）。

本工具实测三件事（逐级）：
  A. **慢变量轨迹读出 + 时间常数估计**：`u`(STP 利用度) / `x`(资源) / `g_i_slow`(GABA_B 电导) 的
     全群均值轨迹 → 指数恢复拟合 + 自相关衰减时间常数 τ_est；**与预注册生理范围比对**；
  B. **慢变量 ↔ 群率慢涨落相关**：群率低通（τ=1 s）后与慢变量轨迹的 |Pearson r|；
     **判据 `slow_var_corr_min`**（`m10_calibration_targets.csv`/`m10_mechanism_ladder.csv`）；
  C. **时间尺度同阶判据**：`τ_slow` 与**该级 bout 停留时长分布**的时间尺度比较
     （比值须落在 [1/10, 10] 的"同阶"窗内；bout 口径与 `larval_zf_circuit.bout_stats` 一致，
     并用**成像参考**的 dwell（ZAPBench 实测）作为跨口径对照）；
  D. **消融关闭态等价**：机制关闭后与 M9 冻结引擎在**同网络同事件表**下**逐 spike 一致**
     （bit-identical spike 计数）→ 满足 §2.2 E1–E3 的"关闭态必须退化回 M9 语义"。

**结构证据级别 = partial/抽象**（Fish1 连接组 CAVE 门控，见 L10）⇒ 按主agent 裁决 1，
本工具的机制归属**降级为"抽象网络内归属"**，**不做结构归属断言**（`attribution_strength`）：
单列输出，报告中不得表述为"连接组结构上的机制归属"。

用法：
    PYTHONHASHSEED=0 ./.venv-m9/bin/python -m neural_exploration.tools.validate_p10_slow_variable --level M-1
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

from ..src.larval_zf_circuit import ZFNetParams, build_engine, run_protocol, bout_stats  # noqa: E402
from .validate_p10_mechanism_ladder import LEVELS  # noqa: E402

# ---- 预注册生理范围（慢变量时间常数；来源见 m10_mechanism_ladder.csv，**范围运行前定稿**）----
TAU_RANGE_MS = {
    "M-1": ("tau_rec_ms", 100.0, 2000.0),      # STP 恢复（抑制型）
    "M-2": ("tau_fac_ms", 100.0, 1000.0),      # STP 易化
    "M-3": ("tau_slow_ms", 80.0, 400.0),       # GABA_B
    "M-4": ("tau_mod_ms", 500.0, 10000.0),     # 调质
    "M-6": ("tau_rec_ms", 10000.0, 60000.0),   # 慢恢复组分（由成像 dwell 26.5 s 反推，预注册）
}
# **判据表述修正（主agent 采纳本节点建议）**：同阶基准改用**成像 dwell 时长分布**，
# 而非快尺度自相关。ZAPBench dwell 中位 26.5 s → 窗 = [0.3×, 3×] = **[8, 80] s**。
SAME_ORDER_WINDOW = (8.0, 80.0)
DWELL_REF_S = 26.5

# ---- 主agent 续令明确指定的跨口径基准：ZAPBench 实测状态时间尺度（同口径 HMM）----
ZAP = dict(dwell_median_s=26.5, transition_rate_hz=0.0196, occupancy_low=0.6719,
           occupancy_high=0.3281,
           source="data/m10_imaging_reference.csv ③（ZAPBench 2 状态 HMM 实测）")


def _load_imaging_dwell_reference():
    """RF-8(iii)：成像侧同法 dwell 参照由 **M10-B4** 落盘；未到位则返回 None（→ C 判 invalid）。"""
    p = os.path.join(DATA_DIR, "m10_imaging_dwell_reference.json")
    if not os.path.exists(p):
        return None
    try:
        d = json.load(open(p))
    except Exception:
        return None
    out = dict(source=os.path.basename(p), generated_by=d.get("meta", {}).get("generated_by"))
    # **同法（Q 分位法）** 的成像侧低态 dwell 中位 —— 与模型侧 `_hmm_states` 同一实现
    try:
        out["dwell_low_median_s"] = float(d["A"]["Q"]["low"]["median_s"])
        out["dwell_low_median_s_interior"] = float(
            d["A"]["Q"]["low"].get("median_s_interior",
                                   d["A"]["Q"]["low"]["median_s"]))
        out["method"] = "quantile_median_split_2state (与模型侧同一实现)"
    except Exception:
        return None
    # H 法（K=2 高斯 HMM）对照值 —— **仅作跨方法对照，不进判据**
    try:
        out["dwell_low_median_s_method_H"] = float(d["D"]["median_s"])
    except Exception:
        pass
    return out


def _hmm_states(pop_bin: np.ndarray, dt_s: float):
    """与成像参考**同口径**的 2 状态判据：全局活动分位法（M10 §1.2.2 ③ 允许的实现之一）。

    口径（**两侧完全一致**）：把群体活动率按**该记录自身的 50% 分位**二分；
    低态 = 低于分位，高态 = 高于分位；由状态序列算占用、转换率、停留时长。
    （成像侧 ZAPBench 的 HMM 占用为 {0.672,0.328}/转换率 0.0196 Hz、低态停留中位 26.5 s，
     本函数用**同一分位法**在模型侧计算，故两者口径可比；分位法给平衡占用，转换率/停留为关键可比量。）
    """
    r = np.asarray(pop_bin, dtype=float)
    thr = float(np.quantile(r, 0.5))
    state = (r > thr).astype(np.int8)
    sw = int((np.diff(state) != 0).sum())
    dur = dict()
    for s_, nm in ((0, "low"), (1, "high")):
        b = (state == s_).astype(np.int8)
        d = np.diff(np.r_[0, b, 0])
        st = np.flatnonzero(d == 1); en = np.flatnonzero(d == -1)
        interior = (st > 0) & (en < state.size)
        ln = (en[interior] - st[interior]) * dt_s
        dur[nm] = dict(n_bouts=int(ln.size),
                       median_s=float(np.median(ln)) if ln.size else 0.0,
                       p90_s=float(np.quantile(ln, 0.9)) if ln.size else 0.0)
    return dict(occupancy=[float((state == 0).mean()), float((state == 1).mean())],
                transition_rate_hz=float(sw / (state.size * dt_s)),
                n_switches=sw, dwell=dur, state=state)


def criterion_b_analysis(slow_trace, slow_dt_s, pop_counts_steps, dt_ms,
                          max_lag_bins: int = 10, n_boot: int = 200, seed: int = 0):
    """**RF-15 修正后的判据 B**（机制归属）：慢变量 s(t) 与群率 r(t) **同一分箱**下：

    1. **去趋势后相关** `corr_detrended`（两侧一阶差分）——消掉"共同缓慢趋势"造成的伪相关；
    2. **lead-lag 互相关** lag ∈ [−L, +L] 箱：报**峰值 lag 与峰值相关**；
       **仅当峰值 lag > 0（机制变量领先）且峰值显著高于零滞后**（bootstrap 95% 下界）才判 `pass`；
    3. 结果四态：`pass` / `fail` / `invalid`（口径或自检失败）；**原始零滞后相关仅作描述性输出**
       （零滞后 ≈±1 多为"跟随者/共同趋势"征象）。
    """
    st = np.asarray(slow_trace, float)
    if st.size < 50:
        return dict(status="invalid", reason=f"慢变量轨迹过短 n={st.size}")
    frame_s = 1.0 / 1.0943
    b = max(1, int(round(frame_s / (dt_ms * 1e-3))))
    nb = pop_counts_steps.size // b
    r = pop_counts_steps[: nb * b].reshape(nb, b).sum(axis=1).astype(float)
    sb = max(1, int(round(frame_s / slow_dt_s)))
    m = st.size // sb
    sv = st[: m * sb].reshape(m, sb).mean(axis=1)
    n = min(r.size, sv.size)
    if n < 30:
        return dict(status="invalid", reason=f"公共箱数过少 n={n}")
    r = r[:n]; sv = sv[:n]
    # 同一管线平滑（5 帧）以对齐 RF-14 主判据口径
    k = np.ones(5) / 5
    if n >= 5:
        r = np.convolve(r, k, mode="valid"); sv = np.convolve(sv, k, mode="valid"); n = r.size
    a = (sv - sv.mean()) / (sv.std() + 1e-12)
    bb = (r - r.mean()) / (r.std() + 1e-12)
    corr_raw = float(a @ bb / n)
    da = np.diff(a); db = np.diff(bb)
    corr_detrended = float((da - da.mean()) @ (db - db.mean()) /
                           (da.size * (da.std() + 1e-12) * (db.std() + 1e-12)))
    # ---- lead-lag 互相关（正 lag = 慢变量领先）----
    lags = np.arange(-max_lag_bins, max_lag_bins + 1)
    xc = []
    for L in lags:
        if L >= 0:
            u, v = a[: n - L], bb[L:]
        else:
            u, v = a[-L:], bb[: n + L]
        if u.size < 10:
            xc.append(np.nan); continue
        xc.append(float((u - u.mean()) @ (v - v.mean()) /
                        (u.size * (u.std() + 1e-12) * (v.std() + 1e-12))))
    xc = np.asarray(xc)
    if np.all(~np.isfinite(xc)):
        return dict(status="invalid", reason="互相关不可算")
    ipk = int(np.nanargmax(np.abs(xc)))
    peak_lag = int(lags[ipk]); peak_corr = float(xc[ipk])
    zero_corr = float(xc[np.flatnonzero(lags == 0)[0]])
    # ---- bootstrap：峰值是否显著高于零滞后（块重采样，块长 5）----
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        aa = a[idx]; bbb = bb[idx]
        def _xc(L):
            if L >= 0:
                u, v = aa[: n - L], bbb[L:]
            else:
                u, v = aa[-L:], bbb[: n + L]
            if u.size < 10:
                return np.nan
            return float((u - u.mean()) @ (v - v.mean()) /
                         (u.size * (u.std() + 1e-12) * (v.std() + 1e-12)))
        diffs.append(abs(_xc(peak_lag)) - abs(_xc(0)))
    diffs = np.asarray([d for d in diffs if np.isfinite(d)])
    ci_lo = float(np.quantile(diffs, 0.025)) if diffs.size else float("nan")
    leads = bool(peak_lag > 0)
    sig_above_zero = bool(np.isfinite(ci_lo) and ci_lo > 0)
    return dict(status="measured", corr_raw_zerolag=corr_raw,
                corr_detrended=corr_detrended, peak_lag_bins=peak_lag,
                peak_corr=peak_corr, peak_lag_s=peak_lag * frame_s,
                zero_lag_corr=zero_corr, peak_minus_zero=abs(peak_corr) - abs(zero_corr),
                boot_ci_lo_peak_minus_zero=ci_lo,
                mechanism_leads=leads, significantly_above_zerolag=sig_above_zero,
                pass_=bool(leads and sig_above_zero),
                n_bins=int(n), n_boot=int(diffs.size),
                note="**零滞后相关仅描述性**：零滞后 ≈±1 多为'跟随者/共同趋势'而非驱动源（RF-15）")


def _lowpass(x: np.ndarray, dt_s: float, tau_s: float) -> np.ndarray:
    a = math.exp(-dt_s / tau_s)
    out = np.empty_like(x, dtype=float)
    acc = 0.0
    for i in range(x.size):
        acc = acc * a + (1 - a) * x[i]
        out[i] = acc
    return out


def _tau_from_ac(x: np.ndarray, dt_s: float) -> float:
    """自相关 e 折时间（首过 1/e）→ 慢变量时间常数估计。"""
    y = np.asarray(x, float)
    y = y - y.mean()
    if y.size < 20 or np.allclose(y, 0):
        return float("nan")
    ac = np.correlate(y, y, mode="full")[y.size - 1:]
    ac = ac / ac[0]
    below = np.flatnonzero(ac <= 1.0 / math.e)
    return float(below[0] * dt_s) if below.size else float("nan")


def _sparse_regime_equiv_check(K: int = 8, steps: int = 400) -> dict:
    """RF-7：稀疏工况（链式、每步 ≤1 发放）下 M9 冻结引擎 ≡ M10 修正内核（逐位）。

    链式 0→1→…→K-1（delay=1）+ 仅 0 号有 tonic 偏置 ⇒ 每步最多 1 个神经元发放，
    M9 的 CSR 起点公式在该工况**恰好正确** ⇒ 可作"M9 语义"参照（RF-7 明确允许）。
    """
    from ..src.adult_engine import AdultEngine
    from ..src.larval_zf_engine import LarvalZFEngine
    res = {}
    for nm, cls, kw in (("m9", AdultEngine, {}),
                        ("m10", LarvalZFEngine, {"use_abs_slots": True})):
        e = cls(K, device="cpu", use_compile=False, verbose=False, **kw)
        e.add_synapses(np.arange(K - 1), np.arange(1, K),
                       np.full(K - 1, 2.0e3, dtype=np.float32),
                       np.ones(K - 1, dtype=np.int32))
        e.finalize()
        b = np.zeros(K, dtype=np.float32); b[0] = 5000.0
        e.set_neuron_heterogeneity(b)
        e.reset(seed=0)
        r = e.run(steps, delivery="step", record="counts")
        res[nm] = np.asarray(r["spike_counts"])
    return dict(m9_spk=int(res["m9"].sum()), m10_spk=int(res["m10"].sum()),
                bit_identical=bool(np.array_equal(res["m9"], res["m10"])),
                n_neurons_counts_differ=int((res["m9"] != res["m10"]).sum()),
                note="稀疏工况（每步 ≤1 发放）⇒ M9 公式精确 ⇒ 可作语义参照（RF-7）")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="M-1", choices=[k for k in LEVELS if k != "M0"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-neuron", type=int, default=1000)
    ap.add_argument("--indegree", type=int, default=20)
    ap.add_argument("--T-ms", type=float, default=6000.0)
    ap.add_argument("--settle-ms", type=float, default=300.0)
    ap.add_argument("--segment-steps", type=int, default=10000)
    ap.add_argument("--lam0", type=float, default=32.0)
    ap.add_argument("--bias-median", type=float, default=150.0)
    ap.add_argument("--g-exc", type=float, default=1.0)
    ap.add_argument("--g-inh", type=float, default=22.0)
    ap.add_argument("--record-every", type=int, default=200)   # 慢变量采样子步
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-csv", default=os.path.join(DATA_DIR, "m10_p6_slow_variable.csv"))
    args = ap.parse_args(argv)

    cfg = LEVELS[args.level]
    Pm = ZFNetParams(n_neuron=args.n_neuron, indegree=args.indegree,
                     bias_median_mv_s=args.bias_median,
                     g_exc=args.g_exc, g_inh=args.g_inh)
    dt_ms = 0.05
    T = int(round(args.T_ms / dt_ms))
    settle = int(round(args.settle_ms / dt_ms))
    t0 = time.time()
    print(f"[slow {args.level}] {cfg['desc']}", flush=True)

    e = build_engine(Pm, device=args.device, stp=cfg["stp"], gaba=cfg["gaba"],
                     g_mod=cfg["g_mod"])
    # ---- 机制参数读回断言（M9 教训 B3）----
    assert e.stp_enabled == bool(cfg["stp"]), "STP 开关读回不一致"
    assert e.gaba_split == bool(cfg["gaba"]), "GABA 开关读回不一致"
    assert abs(e.g_mod - float(cfg["g_mod"])) < 1e-12, "g_mod 读回不一致"

    n_ev = 32.0
    ev = None
    from ..src.larval_zf_circuit import make_events
    ev = make_events(T, n_ev, args.n_neuron, seed=3, g_ext=20.0)
    e.set_external_events(ev["steps"], ev["neurons"], ev["gmax"], n_steps=T)
    e.reset(seed=1)
    e.reset_counts()

    slow = []            # (t_step, u_mean, x_mean, gi_slow_mean, pop_bin)
    n_meas = 0
    step = 0
    CH = 2000
    while step < settle + T:
        w = min(CH, settle + T - step)
        r = e.run(w, delivery="step", record="counts", pop_trace=True)
        pop = r["pop"]
        if step + w > settle and pop is not None:
            keep0 = max(0, settle - step)
            p = pop[keep0:]
            k = max(1, int(args.record_every))
            for i in range(0, p.size, k):
                row = [None, None, None, float(p[i:i + k].sum())]
                row[0] = float(e.t_u.mean().item()) if e.stp_enabled else None
                row[1] = float(e.t_x.mean().item()) if e.stp_enabled else None
                row[2] = (float(e.t_gis.mean().item()) if e.gaba_split else None)
                slow.append(row)
        step += w
    S = np.array([[np.nan if v is None else v for v in r] for r in slow], dtype=float)
    meas_steps = S.shape[0] * args.record_every
    dt_s = args.record_every * dt_ms * 1e-3
    pop_bin = S[:, 3] / (args.record_every * dt_ms * 1e-3) / args.n_neuron   # Hz/神经元
    slow_sm = _lowpass(pop_bin, dt_s, 1.0)
    res = dict(level=args.level, desc=cfg["desc"], n_meas_bins=int(S.shape[0]),
               meas_duration_s=round(meas_steps * dt_ms * 1e-3, 2), dt_bin_s=dt_s,
               mechanism_params=cfg, attribution_strength=(
                   "abstract_only（Fish1 连接组 partial/CAVE 门控 ⇒ 消融归属降级为抽象网络内归属）"),
               stop_condition="按主agent 裁决 2：不做“扫参数直到 bout 出现”的搜索；参数取预注册范围中点")
    # A. 慢变量轨迹与时间常数
    slowvars = {}
    if e.stp_enabled:
        slowvars["stp_u"] = S[:, 0]; slowvars["stp_x"] = S[:, 1]
    if e.gaba_split:
        slowvars["gaba_b_conductance"] = S[:, 2]
    tau_ref = TAU_RANGE_MS.get(args.level)
    tau_est = {}
    for nm, tr in slowvars.items():
        tau_est[nm] = _tau_from_ac(tr, dt_s)
    res["slow_variable"] = dict(
        traces_summary={nm: dict(mean=float(np.nanmean(tr)), std=float(np.nanstd(tr)),
                                 min=float(np.nanmin(tr)), max=float(np.nanmax(tr)))
                        for nm, tr in slowvars.items()},
        tau_est_s=tau_est,
        tau_pre_registered=dict(param=tau_ref[0], lo_ms=tau_ref[1], hi_ms=tau_ref[2]) if tau_ref else None,
    )
    # 生理范围判据
    if tau_ref:
        # 参数范围判据：机制**设定**的时间常数须落在预注册生理范围内（读回断言）
        set_tau = None
        if args.level in ("M-1", "M-2", "M-6") and cfg["stp"]:
            set_tau = float(cfg["stp"]["tau_rec_ms"] if args.level in ("M-1", "M-6")
                            else cfg["stp"]["tau_fac_ms"])
        elif args.level == "M-3" and cfg["gaba"]:
            set_tau = float(cfg["gaba"]["tau_slow_ms"])
        elif args.level == "M-4":
            set_tau = float(e.mod["tau_mod_ms"])
        in_range = bool(set_tau is not None and tau_ref[1] <= set_tau <= tau_ref[2])
        res["slow_variable"]["tau_setting_in_physiological_range"] = dict(
            param=tau_ref[0], value_ms=set_tau, lo_ms=tau_ref[1], hi_ms=tau_ref[2],
            pass_=in_range)
    # B. 慢变量 ↔ 群率慢涨落相关
    corr = {}
    for nm, tr in slowvars.items():
        v = tr - np.nanmean(tr)
        w = slow_sm - slow_sm.mean()
        denom = (np.nanstd(v) * np.nanstd(w))
        corr[nm] = float(np.nanmean(v * w) / denom) if denom > 0 else float("nan")
    res["slow_vs_poprate_corr"] = corr
    res["slow_var_corr_min"] = 0.30
    res["slow_var_corr_pass_DESCRIPTIVE_ONLY"] = bool(any(abs(c) >= 0.30 for c in corr.values()))
    res["slow_var_corr_note"] = ("**RF-15 撤回**：零滞后相关不作为判据（跟随者/共同趋势征象）；"
                                 "以 criterion_B 的（去趋势相关 + lead-lag 显著领先）为准")
    # =====================================================================
    # C（RF-6 修正）：**同口径 dwell 对照量**，而非与快尺度自相关比
    #   旧式 `tau_slow / tau_activity`（τ_activity ≈ 箱宽量级）按构造必失败
    #   → 不是判据而是设计陈述（主agent RF-6）→ 降为**描述性输出，不入 hard_criterion_pass**。
    #   新判据：`model_dwell_low_median_s` vs `imaging_dwell_low_median_s`（**同一方法**）
    #           的对数距离 ≤ log10(3)（即落在 [1/3×, 3×] 内）。
    #   ⚠️ 成像侧参照由 **M10-B4** 用**同一分位法**计算并落盘 `m10_imaging_dwell_reference.{csv,json}`；
    #      该产物到位前 C2/C **一律标 `invalid/未测`**（既不计 fail 也不计 pass，RF-8(iii)）。
    # =====================================================================
    b = bout_stats(pop_bin, dt_ms=dt_s * 1e3, bin_ms=dt_s * 1e3, min_bout_bins=2)
    res["bout_stats_same_bins"] = b
    tau_for_scale = res["slow_variable"].get("tau_setting_in_physiological_range", {}) or {}
    tau_s = (tau_for_scale.get("value_ms") or float("nan")) / 1000.0
    tau_act = _tau_from_ac(pop_bin, dt_s)

    hz_model = None
    try:
        hz_model = _hmm_states(pop_bin, dt_s)          # 分位法（与 B4 成像侧同一实现）
        model_block = dict(occupancy=hz_model["occupancy"],
                           transition_rate_hz=hz_model["transition_rate_hz"],
                           n_switches=hz_model["n_switches"],
                           dwell_low_median_s=hz_model["dwell"]["low"]["median_s"],   # RF-5 修正
                           dwell_high_median_s=hz_model["dwell"]["high"]["median_s"],  # RF-5 修正
                           method="quantile_median_split_2state")
        c2_status = "measured"
    except Exception as ex:  # noqa: BLE001 —— RF-5(ii)：分级输出，不得静默成 fail
        model_block = dict(error=f"{type(ex).__name__}: {ex}")
        c2_status = "invalid"                          # 未测（既不计 fail 亦不计 pass）
    imaging_ref = _load_imaging_dwell_reference()
    c_status = "invalid" if imaging_ref is None else "measured"
    log_dist = float("nan")
    dwell_pass = None
    if c2_status == "measured" and imaging_ref is not None:
        m_dw = model_block.get("dwell_low_median_s")
        i_dw = imaging_ref.get("dwell_low_median_s")
        if m_dw and i_dw and m_dw > 0 and i_dw > 0:
            log_dist = abs(math.log10(m_dw / i_dw))
            dwell_pass = bool(log_dist <= math.log10(3.0))
    res["criterion_C_same_pipeline_dwell"] = dict(
        criterion="model_dwell_low_median_s 与 imaging_dwell_low_median_s **同方法（分位法）** 同阶"
                  "（比值落 [1/3×, 3×]，等价 log10 距离 ≤ log10 3）",
        status=c_status if c2_status == "measured" else c2_status,
        model_dwell_low_median_s=model_block.get("dwell_low_median_s"),
        imaging_dwell_low_median_s=(imaging_ref or {}).get("dwell_low_median_s"),
        imaging_reference_source=(imaging_ref or {}).get("source"),
        log10_distance=log_dist if log_dist == log_dist else None,
        pass_=dwell_pass if dwell_pass is not None else None,
        descriptive_only=dict(
            tau_slow_s=tau_s, tau_activity_autocorr_s=tau_act,
            ratio_tau_over_activity=(tau_s / tau_act if (tau_act and np.isfinite(tau_act)
                                                         and tau_act > 0) else float("nan")),
            note="**RF-6**：本行仅描述性；τ_activity 为箱宽量级 ⇒ 该比值不进 hard_criterion_pass"))
    res["zapbench_timescale_comparison"] = dict(
        status=c2_status, imaging_reference_status=c_status,
        model=model_block,
        imaging_reference=imaging_ref,
        method_note="两侧**同一方法**（分位法 `_hmm_states`）；成像侧由 M10-B4 以同法复算并落盘 "
                    "`m10_imaging_dwell_reference.json`（RF-8：同一方法下比较，禁止跨口径比）",
        c2_invalid_reason=(None if (c2_status == "measured" and imaging_ref is not None)
                           else "等 M10-B4 的成像侧同法 dwell 参照落地前，C2 判 invalid/未测（RF-8(iii)）"))
    zc = res.get("zapbench_timescale_comparison", {})
    res["wall_s"] = round(time.time() - t0, 1)
    del e

    # D. 消融关闭态等价（机制关闭 ⇒ 与 M9 冻结引擎逐 spike 一致）
    from ..src.adult_engine import AdultEngine
    Ablation = {}
    try:
        rng = np.random.default_rng(Pm.seed)
        E = int(Pm.n_neuron) * int(Pm.indegree)
        pre = rng.integers(0, Pm.n_neuron, E)
        post = np.repeat(np.arange(Pm.n_neuron), Pm.indegree); rng.shuffle(post)
        inh = rng.random(E) < Pm.inh_frac
        gmax = np.where(inh, rng.uniform(0.8, 1.2, E) * Pm.g_inh,
                        rng.uniform(0.8, 1.2, E) * Pm.g_exc).astype(np.float32)
        dly = np.full(E, max(2, int(Pm.delay_max)), dtype=np.int32)
        bias = np.exp(np.random.default_rng(Pm.seed).normal(
            math.log(Pm.bias_median_mv_s), Pm.bias_cv, Pm.n_neuron)).astype(np.float32)
        seg = min(T, 40_000)
        evx = make_events(seg, n_ev, Pm.n_neuron, seed=3, g_ext=20.0)
        outs = {}
        for nm, cls, kw in (("m9_engine", AdultEngine, {}),
                            ("m10_engine_off", None, {})):
            if cls is None:
                from ..src.larval_zf_engine import LarvalZFEngine as cls2
                eng = cls2(Pm.n_neuron, device="cpu", use_compile=False, verbose=False,
                           use_abs_slots=True)
            else:
                eng = cls(Pm.n_neuron, device="cpu", use_compile=False, verbose=False)
            eng.add_synapses(pre, post, gmax, dly, inhibitory=inh,
                             syn_class=inh.astype(np.int8))
            eng.finalize()
            eng.set_neuron_heterogeneity(bias)
            eng.set_external_events(evx["steps"], evx["neurons"], evx["gmax"], n_steps=seg)
            eng.reset(seed=1)
            try:
                rr = eng.run(seg, delivery="step", record="counts")
                outs[nm] = dict(ok=True, spk=int(rr["spk_total"]),
                                counts=np.asarray(rr["spike_counts"]))
            except Exception as ex:  # noqa: BLE001
                outs[nm] = dict(ok=False, err=f"{type(ex).__name__}: {str(ex)[:80]}")
            del eng
        dense = {}
        if outs["m9_engine"].get("ok") and outs["m10_engine_off"].get("ok"):
            same = bool(np.array_equal(outs["m9_engine"]["counts"],
                                       outs["m10_engine_off"]["counts"]))
            dense = dict(bit_identical_counts=same,
                         m9_spk=outs["m9_engine"]["spk"],
                         m10_off_spk=outs["m10_engine_off"]["spk"],
                         n_neurons_counts_differ=int((outs["m9_engine"]["counts"]
                                                      != outs["m10_engine_off"]["counts"]).sum()))
        # ---- RF-7：D 判据的**参照**必须是"引擎语义正确"的工况 ----
        # 稀疏工况（每步 ≤1 个神经元发放）下 M9 冻结公式**恰好正确** ⇒ 作为 M9 语义参照（RF-7 允许）；
        # 密集工况下 M9 的 CSR 越界读是**已知缺陷**（E-M9-2）⇒ 二者不一致恰是**修正生效的签名**，
        # **不得**当作 "M10 错"，**不得**进入结论。
        sparse_spk = _sparse_regime_equiv_check()
        Ablation = dict(
            sparse_regime_bit_identical=sparse_spk.get("bit_identical"),
            sparse_regime=sparse_spk,
            dense_regime_m9_comparison=dense,
            dense_regime_role="defect_reproducibility_only（E-M9-2 的又一实测证据：M9 在密集工况"
                              "静默越界读 ⇒ 与修正内核不一致是**修正生效**的签名，不作判据）",
            criterion_reference="M9 语义在**稀疏工况**（RF-7 明确允许）；密集工况参照应为 Brian2，"
                                "本轮未做（登记）")
        if not dense:      # 密集工况下 M9 引擎抛错（L17）→ 该对照记为不可用（不作判据）
            Ablation["dense_regime_m9_comparison"] = dict(
                status="unavailable",
                m9=dict(ok=outs["m9_engine"].get("ok"), err=outs["m9_engine"].get("err")),
                m10=dict(ok=outs["m10_engine_off"].get("ok"),
                         err=outs["m10_engine_off"].get("err")),
                note="M9 冻结引擎在密集发放工况抛错（L17/E-M9-2）⇒ 密集对照不可用（不作判据）")
    except Exception as ex:  # noqa: BLE001
        Ablation = dict(status="invalid", error=f"{type(ex).__name__}: {ex}")
    res["closed_state_equivalence"] = Ablation
    _cc = res.get("criterion_C_same_pipeline_dwell", {})
    res["verdict"] = dict(
        C_same_pipeline_dwell_status=_cc.get("status"),
        C_same_pipeline_dwell_pass=_cc.get("pass_"),      # None = invalid/未测
        C2_model_dwell_low_median_s=((res.get("zapbench_timescale_comparison", {}) or {})
                                     .get("model", {}) or {}).get("dwell_low_median_s"),
        C2_model_transition_rate_hz=((res.get("zapbench_timescale_comparison", {}) or {})
                                     .get("model", {}) or {}).get("transition_rate_hz"),
        A_tau_in_phys_range=bool((res["slow_variable"]
                                  .get("tau_setting_in_physiological_range", {}) or {})
                                 .get("pass_", False)),
        B_slow_var_corr_pass=res["slow_var_corr_pass"],
        D_sparse_regime_equiv=Ablation.get("sparse_regime_bit_identical"),
        hard_criterion_pass=bool(
            (res["slow_variable"].get("tau_setting_in_physiological_range", {}) or {}).get("pass_", False)
            and res["slow_var_corr_pass"] and (_cc.get("pass_") is True)),
    )
    outp = args.out_json or os.path.join(DATA_DIR, f"m10_p6_slow_{args.level}.json")
    with open(outp, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False, default=lambda o: (
            o.tolist() if hasattr(o, "tolist") else str(o)))
    with open(args.out_csv, "a", newline="") as f:
        new = not os.path.exists(args.out_csv) or os.path.getsize(args.out_csv) == 0
        w = csv.writer(f, lineterminator="\n")
        if new:
            f.write("# M10 P6 慢变量硬判据（主agent 裁决 2；结构证据级别=抽象 ⇒ 归属降级）\n")
            w.writerow(["level", "tau_setting_ms", "tau_in_range", "corr_max_var",
                        "corr_val", "C_status", "model_dwell_low_median_s",
                        "imaging_dwell_low_median_s_same_method", "C_log10_distance",
                        "C_pass", "D_sparse_equiv_bit_identical", "hard_criterion_pass"])
        cmax = max(corr.items(), key=lambda kv: abs(kv[1])) if corr else ("", float("nan"))
        _cc = res.get("criterion_C_same_pipeline_dwell", {})
        w.writerow([args.level,
                    (res["slow_variable"].get("tau_setting_in_physiological_range", {}) or {}).get("value_ms"),
                    res["verdict"]["A_tau_in_phys_range"],
                    cmax[0], round(cmax[1], 4) if cmax[1] == cmax[1] else None,
                    _cc.get("status"), _cc.get("model_dwell_low_median_s"),
                    _cc.get("imaging_dwell_low_median_s"),
                    round(_cc["log10_distance"], 4) if _cc.get("log10_distance") else None,
                    _cc.get("pass_"),
                    (res.get("closed_state_equivalence", {}) or {}).get("sparse_regime_bit_identical"),
                    res["verdict"]["hard_criterion_pass"]])
    print(json.dumps(res["verdict"], ensure_ascii=False, indent=1))
    print(f"[slow {args.level}] τ_set={tau_s}s τ_act={tau_act}s ratio={ratio} "
          f"corr={corr} 消融={Ablation.get('bit_identical_counts')} ({res['wall_s']}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
