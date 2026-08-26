"""M6 P2 验证：神经调质系统 + M5 反证清单落地（G1 机制门复核）。

对应《生物仿真M6实施清单》§0 P2 / §3.2 / §0 G1 门：
- 四项机制（RIM 酪胺/命令互抑/AVA→DD GABA 链/自发输入）在 302 网络落地并消融 sanity；
- **复核 M5 的 P2/P4/P6**（同协议同带，data/m5_behavior_reference.csv 预注册带，
  不做事后调）+ **P5 方向相位复核**（定稿 τ_trans=23 → touch@73ms 方向 back 是否恢复）；
- 输出 G1 判定（通过/部分/不通过）+ 复核数值入 docs/m6_env_notes.md。

定稿调质参数（唯一定稿源 data/m6_learning_params.csv mod 行；本脚本消费入口 =
`load_m6_mod_params`）：O2 组合（docs/m6_env_notes.md M6-B1a L7+）：
  - ① RIM 酪胺基线浓度 tyr_baseline=1.0（fwd_gate→floor：AVB 张力压到持续发放
    分岔点之下，打破单张力夹带；低于阈值靠自发脉冲瞬态驱动 → 前进/后退 bout）；
  - ② 命令互抑**非对称**（inh_back_on_fwd=0.80nA 强 / inh_fwd_on_back=0.05nA 弱——
    RIM/酪胺语义：后退时抑制前进，前进对后退抑制弱）；mod_dt_ms=5.0 细粒度
    （逃避 150ms 窗内互抑反应延迟 ≤5ms，方向相位修复的关键）；
  - ③ AVA→DD/VD GABA 功能链 gaba_chain_gain=0.80nA（后退 bout 隔离 fwd 池）；
  - ④ 自发输入 = 运动池输出级 bout 驱动（DA/VA + SMDD，2Hz×0.10nA×3ms，seed 固定；
    命令池不注入——注入命令池会重新点燃 86% 同步夹带，实测坑 L9）。

预算（预注册 ≤150 CPU-h；本脚本约 1-1.5 CPU-h）：
  P2 T=10s×N=5、P6 T=30s×N=10（mod_dt=None 25ms 档——调质 τ~500ms 慢动力学，
  25ms 更新等价；P5 用 5ms 档）、P4 T=15s×N=20+对照、P5 150ms×N=5。

用量：.venv-neuro/bin/python -m neural_exploration.tools.validate_p6_modulation
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.src.worm_circuit import load_weight_scales  # noqa: E402
from neural_exploration.src.neuromod import (  # noqa: E402
    ModParams,
    ModulatorPool,
    load_m6_mod_params,
    make_modulated_circuit,
)
from neural_exploration.src.worm_loop import WormLoop  # noqa: E402

DATA_DIR = os.path.join(ROOT, "neural_exploration", "data")
REPORTS_DIR = os.path.join(ROOT, "neural_exploration", "reports", "neuro")

#: 判据带（data/m5_behavior_reference.csv 预注册；不做事后调）
BAND_SILENT = (0.60, 0.80)      # resting.silent_fraction_target（post-settle）
BAND_MEDIAN = (0.0, 1.0)        # resting.median_firing_hz
BAND_MAX = (0.0, 60.0)          # resting.max_firing_hz
BAND_SPONT = dict(fwd=(60.0, 80.0), rev=(10.0, 25.0), turn=(5.0, 20.0))
D_PEAK_THR = 0.3                # escape.direction_peak
SETTLE_MS = 500.0               # L41#1 settle 窗（预注册）
P2_T_MS = 10000.0
P2_N = 5
P6_T_MS = 30000.0
P6_N = 10
P4_T_MS = 15000.0
P4_N = 20
P5_N = 5
TOUCH_TAUS = (0.0, 23.0)        # τ_trans=0（M5 参考相位）与定稿 23


def _stats(x: np.ndarray) -> dict:
    from scipy import stats as sps

    c = np.asarray(x, dtype=float)
    n = int(c.size)
    mean = float(c.mean())
    sd = float(c.std(ddof=1)) if n > 1 else 0.0
    sem = sd / np.sqrt(n) if n > 0 else float("nan")
    if n > 1 and sd > 0:
        t_stat, p_val = sps.ttest_1samp(c, 0.0)
        d = mean / sd
    else:
        t_stat, p_val, d = float("nan"), float("nan"), float("nan")
    return dict(n=n, mean=mean, sem=sem, sd=sd, t_stat=float(t_stat),
                p_value=float(p_val), cohen_d=float(d))


def build_mc(mod_dt: float, mod=None):
    """调质电路工厂：CSV 定稿参数 + mod_dt 覆盖（P5 用 5ms，P2/P6/P4 用 25ms）。"""
    if mod is None:
        p = load_m6_mod_params()
        p.mod_dt_ms = mod_dt
        mod = ModulatorPool(p)
    return make_modulated_circuit(scale=302, seed=0, mod=mod,
                                  **load_weight_scales())


# --------------------------------------------------------------------- #
def run_p2_recheck(mc) -> dict:
    """P2 静息复核（同 M5 协议：T=10s×N=5，settle 500ms；带 [60,80]/<1Hz/<60Hz）。"""
    t0 = time.perf_counter()
    runs = []
    for k in range(P2_N):
        sess = mc.make_session(t_total_ms=P2_T_MS)
        sess.reset(seed=k)
        sess.run_resting_window(P2_T_MS)
        times = sess.role_spike_times()
        post = np.array(
            [len(np.asarray(t)[np.asarray(t) >= SETTLE_MS])
             / ((P2_T_MS - SETTLE_MS) / 1000.0) for t in times.values()])
        runs.append(dict(
            silent_post=float(np.mean(post < 0.1)),
            median_post=float(np.median(post)),
            max_post=float(post.max()),
            has_nan=bool(np.any(~np.isfinite(post)))))
    m = runs[0]
    out = dict(
        silent_post=m["silent_post"], median_post=m["median_post"],
        max_post=m["max_post"], wall_s=time.perf_counter() - t0,
        in_band_silent=BAND_SILENT[0] <= m["silent_post"] <= BAND_SILENT[1],
        in_band_median=BAND_MEDIAN[0] <= m["median_post"] <= BAND_MEDIAN[1],
        in_band_max=BAND_MAX[0] <= m["max_post"] <= BAND_MAX[1],
    )
    out["pass"] = bool(out["in_band_silent"] and out["in_band_median"]
                       and out["in_band_max"] and not m["has_nan"])
    print(f"P2 复核: silent_post={out['silent_post']:.3f} "
          f"median={out['median_post']:.2f}Hz max={out['max_post']:.1f}Hz "
          f"(带 [{BAND_SILENT[0]:.0%},{BAND_SILENT[1]:.0%}]/<1/<60) "
          f"pass={out['pass']} wall={out['wall_s']:.0f}s")
    return out


def run_p6_recheck(mc) -> dict:
    """P6 自发复核（同 M5 协议：T≥30s×N≥10；fwd/rev/turn 带）。"""
    t0 = time.perf_counter()
    wl = WormLoop(mc)
    fracs = []
    for k in range(P6_N):
        r = wl.run_spontaneous(t_total_ms=P6_T_MS, seed=k)
        fracs.append(r["frac"])
    mean_frac = {s: float(np.mean([f[s] for f in fracs]))
                 for s in ("fwd", "rev", "turn", "pause")}
    out = dict(
        mean_frac=mean_frac, wall_s=time.perf_counter() - t0,
        in_band={s: BAND_SPONT[s][0] <= mean_frac[s] * 100 <= BAND_SPONT[s][1]
                 for s in BAND_SPONT},
    )
    out["pass"] = bool(all(out["in_band"].values()))
    print(f"P6 复核: fwd={mean_frac['fwd']:.1%} rev={mean_frac['rev']:.1%} "
          f"turn={mean_frac['turn']:.1%} pause={mean_frac['pause']:.1%} "
          f"(带 fwd[60,80]/rev[10,25]/turn[5,20]) pass={out['pass']} "
          f"wall={out['wall_s']:.0f}s")
    return out


def run_p4_recheck(mc) -> dict:
    """P4 趋化复核（同 M5 协议：T=15s×N=20 梯度+对照；CI 显著>0 + 对照 p>0.05）。"""
    t0 = time.perf_counter()
    wl = WormLoop(mc)
    res_grad = wl.run_trials(n_trials=P4_N, seed_base=0, t_total_ms=P4_T_MS)
    ci_grad = np.array([r.ci for r in res_grad], dtype=float)
    res_ctrl = wl.run_control(n_trials=P4_N, seed_base=1000, t_total_ms=P4_T_MS)
    ci_ctrl = np.array([r.ci for r in res_ctrl], dtype=float)
    sg = _stats(ci_grad)
    sc = _stats(ci_ctrl)
    out = dict(
        ci_grad=sg, ci_ctrl=sc, wall_s=time.perf_counter() - t0,
        sig_ok=bool(sg["p_value"] < 0.05 and sg["cohen_d"] >= 0.5),
        ctrl_ok=bool(sc["p_value"] > 0.05),
        dir_ok=bool(sg["mean"] > 0.0),
        finite_ok=bool(np.all(np.isfinite(ci_grad)) and np.all(np.isfinite(ci_ctrl))),
    )
    out["pass"] = bool(out["sig_ok"] and (out["dir_ok"]) and out["ctrl_ok"]
                       and out["finite_ok"])
    print(f"P4 复核: CĪ={sg['mean']:+.3f}±{sg['sem']:.3f} (p={sg['p_value']:.4f}, "
          f"d={sg['cohen_d']:.2f}) 对照 CĪ={sc['mean']:+.3f} (p={sc['p_value']:.4f}) "
          f"pass={out['pass']} wall={out['wall_s']:.0f}s")
    return out


def run_p5_phase_recheck() -> dict:
    """P5 方向相位复核（定稿 τ_trans=23 → touch@73ms → back？N=5 确定性）。"""
    import gc

    mc = build_mc(mod_dt=5.0)
    wl = WormLoop(mc)
    out = {"by_tau": {}}
    for tau in TOUCH_TAUS:
        dpeaks = []
        saved = wl.touch["tau_trans_ms"]
        wl.touch["tau_trans_ms"] = tau
        try:
            for k in range(P5_N):
                r = wl.run_escape(t_total_ms=150.0, seed=k)
                dpeaks.append(r["d_peak"])
                del r
                gc.collect()   # 302 会话 725MB stim 数组释放（L9 #6 内存纪律）
        finally:
            wl.touch["tau_trans_ms"] = saved
        d_arr = np.array(dpeaks)
        out["by_tau"][tau] = dict(
            d_peak=float(np.mean(d_arr)), d_peak_std=float(np.std(d_arr, ddof=1)),
            direction="back" if float(np.mean(d_arr)) > D_PEAK_THR else "not_back",
            all_back=bool(np.all(d_arr > D_PEAK_THR)))
        print(f"P5 相位 τ_trans={tau:.0f}ms: D_peak={out['by_tau'][tau]['d_peak']:+.3f}"
              f"±{out['by_tau'][tau]['d_peak_std']:.3f} → "
              f"{out['by_tau'][tau]['direction']} (阈值 {D_PEAK_THR})")
    # G1 关键判据：定稿 τ_trans=23（touch@73ms）方向 back
    out["tau23_back"] = out["by_tau"][23.0]["direction"] == "back"
    return out


def run_ablation_sanity(mc_all_on) -> dict:
    """四项机制消融 sanity（清单 §3.2：删项 → 现象消失）。"""
    out = {}
    # ① 删酪胺门控 → fwd/back 共同发放复现（逃避 D_peak 显著劣化）
    p_off_tyr = load_m6_mod_params()
    p_off_tyr.tyramine_enabled = False
    mc_notyr = build_mc(mod_dt=5.0, mod=ModulatorPool(p_off_tyr))
    wl = WormLoop(mc_notyr)
    saved = wl.touch["tau_trans_ms"]
    wl.touch["tau_trans_ms"] = 23.0
    try:
        r_notyr = wl.run_escape(t_total_ms=150.0, seed=0)
    finally:
        wl.touch["tau_trans_ms"] = saved
    out["tyramine"] = dict(
        d_peak_off=float(r_notyr["d_peak"]),
        direction_off=r_notyr["direction"])
    print(f"消融① 酪胺关: escape@73ms D_peak={out['tyramine']['d_peak_off']:+.3f} "
          f"({out['tyramine']['direction_off']})")
    # ② 删命令互抑 → 方向相位敏感复现（touch@73ms 掉回 not_back）
    p_off_inh = load_m6_mod_params()
    p_off_inh.mutual_inh_enabled = False
    mc_noinh = build_mc(mod_dt=5.0, mod=ModulatorPool(p_off_inh))
    wl = WormLoop(mc_noinh)
    saved = wl.touch["tau_trans_ms"]
    wl.touch["tau_trans_ms"] = 23.0
    try:
        r_noinh = wl.run_escape(t_total_ms=150.0, seed=0)
    finally:
        wl.touch["tau_trans_ms"] = saved
    out["mutual_inh"] = dict(
        d_peak_off=float(r_noinh["d_peak"]),
        direction_off=r_noinh["direction"])
    print(f"消融② 互抑关: escape@73ms D_peak={out['mutual_inh']['d_peak_off']:+.3f} "
          f"({out['mutual_inh']['direction_off']})")
    # ③ 删 AVA→DD/VD 链 → fwd 池隔离失效（自发 rev 比例回落）
    def _spont_rev(mc, T=6000.0):
        wl = WormLoop(mc)
        r = wl.run_spontaneous(t_total_ms=T, seed=0)
        return r["frac"]["rev"]
    out["gaba_chain"] = dict(
        rev_on=_spont_rev(mc_all_on),
        rev_off=_spont_rev(build_mc(mod_dt=None, mod=ModulatorPool(
            _with("gaba_chain_enabled", False)))))
    print(f"消融③ GABA 链关: 自发 rev on={out['gaba_chain']['rev_on']:.3f} "
          f"off={out['gaba_chain']['rev_off']:.3f}")
    # ④ 删自发输入 → 自发 bout 驱动现象消失（行为比例 → ~0，静默 → 100%）
    def _spont_frac(mc, T=6000.0):
        wl = WormLoop(mc)
        r = wl.run_spontaneous(t_total_ms=T, seed=0)
        return r["frac"]
    f_on = _spont_frac(mc_all_on)
    f_off = _spont_frac(build_mc(mod_dt=None, mod=ModulatorPool(
        _with("spont_enabled", False))))
    out["spont"] = dict(
        frac_on=f_on, frac_off=f_off,
        bout_disappears=bool(f_off["pause"] > 0.9))
    print(f"消融④ 自发关: 自发 on={f_on} → off={f_off}（bout 消失: "
          f"{out['spont']['bout_disappears']}）")
    return out


def _with(key: str, value):
    p = load_m6_mod_params()
    setattr(p, key, value)
    return p


# --------------------------------------------------------------------- #
def main() -> dict:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    t_start = time.perf_counter()
    print("== M6 G1 机制门复核（调质系统 + M5 反证清单落地）==")
    # P5 相位（5ms 档——逃避方向相位修复的关键参数）
    p5 = run_p5_phase_recheck()
    import gc
    gc.collect()
    # P2/P6/P4 复核（25ms 档——调质 τ~500ms 慢动力学，更新粒度等价）
    mc = build_mc(mod_dt=None)
    p2 = run_p2_recheck(mc)
    gc.collect()
    p6 = run_p6_recheck(mc)
    gc.collect()
    p4 = run_p4_recheck(mc)
    gc.collect()
    # 消融 sanity
    abl = run_ablation_sanity(mc)

    # ---- G1 判定（§0 G1：通过/部分通过（至少方向相位修复）/不通过）----
    direction_fixed = p5["tau23_back"]
    entrainment_relieved = p2["pass"] or p6["pass"]
    if direction_fixed and p2["pass"] and p6["pass"]:
        g1 = "PASS"
    elif direction_fixed:
        g1 = "PARTIAL"   # 方向相位修复（G1 关键前置）；P2/P6 部分/未达带
    else:
        g1 = "FAIL"
    print(f"\n== G1 判定: {g1} ==")
    print(f"  P5 方向相位（τ_trans=23 → back）: {'✓' if direction_fixed else '✗'}")
    print(f"  P2 静默带: {'✓' if p2['in_band_silent'] else '✗'} "
          f"({p2['silent_post']:.1%})")
    print(f"  P6 自发带: {'✓' if p6['pass'] else '✗'} {p6['mean_frac']}")
    print(f"  P4 趋化: {'✓' if p4['pass'] else '✗'} "
          f"(CĪ={p4['ci_grad']['mean']:+.3f} p={p4['ci_grad']['p_value']:.4f})")

    summary = dict(
        g1_verdict=g1, direction_fixed=direction_fixed,
        p5=p5, p2=p2, p6=p6, p4=p4, ablation=abl,
        params=load_m6_mod_params().__dict__,
        bands=dict(silent=BAND_SILENT, median=BAND_MEDIAN, max=BAND_MAX,
                   spont=BAND_SPONT, d_peak_thr=D_PEAK_THR),
    )
    import json
    with open(os.path.join(DATA_DIR, "m6_g1_result.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, default=str)
    print(f"\n结果已落盘 data/m6_g1_result.json（wall 总 {time.perf_counter() - t_start:.0f}s）")
    return summary


if __name__ == "__main__":
    main()
