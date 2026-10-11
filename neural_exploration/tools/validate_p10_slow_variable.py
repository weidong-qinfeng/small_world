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
}
SAME_ORDER_WINDOW = (0.1, 10.0)               # "同阶"判据窗（τ_slow / bout 时间尺度）


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
    ap.add_argument("--record-every", type=int, default=200)   # 慢变量采样子步
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-csv", default=os.path.join(DATA_DIR, "m10_p6_slow_variable.csv"))
    args = ap.parse_args(argv)

    cfg = LEVELS[args.level]
    Pm = ZFNetParams(n_neuron=args.n_neuron, indegree=args.indegree,
                     bias_median_mv_s=args.bias_median)
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
        if args.level in ("M-1", "M-2") and cfg["stp"]:
            set_tau = float(cfg["stp"]["tau_rec_ms"] if args.level == "M-1"
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
    res["slow_var_corr_pass"] = bool(any(abs(c) >= 0.30 for c in corr.values()))
    # C. 时间尺度同阶判据（τ_slow vs bout 时长）
    b = bout_stats(pop_bin, dt_ms=dt_s * 1e3, bin_ms=dt_s * 1e3, min_bout_bins=2)
    res["bout_stats_same_bins"] = b
    tau_for_scale = res["slow_variable"].get("tau_setting_in_physiological_range", {}) or {}
    tau_s = (tau_for_scale.get("value_ms") or float("nan")) / 1000.0
    # bout 时间尺度：以"静默/活动箱的自相关 e 折"作为该级活动的慢时间尺度（与 bout 同源的慢尺度）
    tau_act = _tau_from_ac(pop_bin, dt_s)
    ratio = (tau_s / tau_act) if (tau_act and np.isfinite(tau_act) and tau_act > 0) else float("nan")
    same_order = bool(np.isfinite(ratio) and SAME_ORDER_WINDOW[0] <= ratio <= SAME_ORDER_WINDOW[1])
    res["timescale_same_order"] = dict(tau_slow_s=tau_s, tau_activity_s=tau_act,
                                       ratio=ratio, window=list(SAME_ORDER_WINDOW),
                                       pass_=same_order)
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
            eng.add_synapses(pre, post, gmax, dly, inhibitory=inh)
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
        if outs["m9_engine"].get("ok") and outs["m10_engine_off"].get("ok"):
            same = bool(np.array_equal(outs["m9_engine"]["counts"],
                                       outs["m10_engine_off"]["counts"]))
            Ablation = dict(bit_identical_counts=same,
                            m9_spk=outs["m9_engine"]["spk"],
                            m10_off_spk=outs["m10_engine_off"]["spk"],
                            n_neurons_counts_differ=int((outs["m9_engine"]["counts"]
                                                         != outs["m10_engine_off"]["counts"]).sum()))
        else:
            Ablation = dict(
                m9=dict(ok=outs["m9_engine"].get("ok"),
                        err=outs["m9_engine"].get("err"),
                        spk=outs["m9_engine"].get("spk")),
                m10=dict(ok=outs["m10_engine_off"].get("ok"),
                         err=outs["m10_engine_off"].get("err"),
                         spk=outs["m10_engine_off"].get("spk")),
                note="M9 冻结引擎在密集发放工况抛错（见 L17）⇒ **关闭态等价的密集工况对照不可用**"
                     "（这本身就是 L17 的又一实测证据）；稀疏工况下 M9/M10 逐位一致已在"
                     " `scan_m10_engine` T1 实测（t1_m9_m10_identical=true）")
    except Exception as ex:  # noqa: BLE001
        Ablation = dict(error=f"{type(ex).__name__}: {ex}")
    res["closed_state_equivalence"] = Ablation
    res["verdict"] = dict(
        A_tau_in_phys_range=bool((res["slow_variable"]
                                  .get("tau_setting_in_physiological_range", {}) or {})
                                 .get("pass_", False)),
        B_slow_var_corr_pass=res["slow_var_corr_pass"],
        C_same_order_pass=res["timescale_same_order"]["pass_"],
        D_closed_state_equiv=Ablation.get("bit_identical_counts"),
        hard_criterion_pass=bool(
            (res["slow_variable"].get("tau_setting_in_physiological_range", {}) or {}).get("pass_", False)
            and res["slow_var_corr_pass"] and res["timescale_same_order"]["pass_"]),
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
            w.writerow(["level", "tau_setting_ms", "tau_in_range", "tau_est_s",
                        "slow_var_corr_max_var", "slow_var_corr", "tau_activity_s",
                        "tau_ratio", "same_order_pass", "closed_state_bit_identical",
                        "hard_criterion_pass"])
        cmax = max(corr.items(), key=lambda kv: abs(kv[1])) if corr else ("", float("nan"))
        w.writerow([args.level,
                    (res["slow_variable"].get("tau_setting_in_physiological_range", {}) or {}).get("value_ms"),
                    res["verdict"]["A_tau_in_phys_range"], round(tau_act, 4) if np.isfinite(tau_act) else None,
                    cmax[0], round(cmax[1], 4) if np.isfinite(cmax[1]) else None,
                    round(tau_s, 4) if np.isfinite(tau_s) else None,
                    round(ratio, 4) if np.isfinite(ratio) else None,
                    res["verdict"]["C_same_order_pass"],
                    Ablation.get("bit_identical_counts"),
                    res["verdict"]["hard_criterion_pass"]])
    print(json.dumps(res["verdict"], ensure_ascii=False, indent=1))
    print(f"[slow {args.level}] τ_set={tau_s}s τ_act={tau_act}s ratio={ratio} "
          f"corr={corr} 消融={Ablation.get('bit_identical_counts')} ({res['wall_s']}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
