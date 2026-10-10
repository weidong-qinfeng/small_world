"""M9 P4 附加：真实抑制边（GABA）H6 消融 sanity（清单 §3.5.5 + M8 R2 承接）。

《生物仿真M9实施清单》§3.5.3（双状态三杠杆：① 真实抑制边/GABA 平衡）/§3.5.5（递质抑制边
关闭 = H6 消融方向）/§0.2 R2（M8 反证：缺 GABA → 抑制平衡缺失；M9 数据侧已解除——真实
GABA 边 3,233,022 连接 / 21.42%，见 `data/m9_inhibition_inventory.csv`）。

**本脚本是 M9 的首要科学检验点之一**：真实抑制边是否使自发分布在正确的方向改善？
（M8 冻结基线 M9 R2 语境：M8 幼虫 run=9.9% vs 行为带 [60,85]%）。

协议（与 P4 同构，配对设计）：
  - 基线臂：真实 GABA 边（`m9_circuit_params.csv` 定稿参数）；
  - H6 消融臂：**GABA 边 g→0**（`w_inh = 0`，拓扑不变——只关权重，不删边）；
  - T=2s（§3.5.4 最短协议，同 P4）、settle=1s、N=3 试次、固定 seed、确定性。

判据（§3.5.5 H6 方向断言，sanity 级）：
  - Δrate_median ≥ 0（消融 → 发放率上升或持平）
  - Δsilent_fraction ≤ 0（消融 → 静默比例下降或持平）
  - Δpop_rate ≥ 0（消融 → 群体活动上升）
  任一方向相反 → 反证记录（真实 GABA 边未起抑制平衡作用）+ 三态裁决。

输出：`data/m9_p4_gaba_ablation.csv` + `reports/neuro/m9_p4_gaba_ablation.png`
用法：PYTHONHASHSEED=0 MPLBACKEND=Agg .venv-m9/bin/python -m neural_exploration.tools.validate_p9_gaba_ablation
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
OUT_CSV = os.path.join(DATA_DIR, "m9_p4_gaba_ablation.csv")
OUT_PNG = os.path.join(REPORT_DIR, "m9_p4_gaba_ablation.png")

T_MS = 1000.0
SETTLE_MS = 500.0
N_TRIALS = 1
#: L25.2 机制假说检验：把 v_floor 抬到 E_GABA 之上（−70 > −75 mV）→ 抑制驱动项恒 ≤0。
#: 若 H6 方向随之**转正**（消融 → 活动↑/静默↓），则 L25.2 的机制定位得到确证。
V_FLOOR_ARMS = (-80.0, -70.0)
BG = dict(rate_hz=0.5, epsp_mv=2.0, seed=0)

#: 与 P4 定稿同源参数（`tools/validate_p9_resting.py::PARAMS`）
PARAMS = dict(
    w_exc=0.3, w_inh=1.0, w_mod=0.3, syn_count_gamma=1.0,
    bias_mv_s=290.0, bias_cv=0.25, bias_mode="lognormal",
    ahp_tau_ms=700.0, ahp_inc=2500.0,
    v_rest=-52.0, v_th=-45.0, v_reset=-55.0, tau_m=20.0, tau_e=2.0, tau_i=5.0,
    ref_ms=2.0, dt_ms=0.05, delay_ms=1.0,
)


def _run_arm(c, tag, seeds, w_inh, v_floor=-80.0):
    out = []
    for seed in seeds:
        c.engine.set_point_params(v_floor=v_floor)
        c.reweight(w_exc=PARAMS["w_exc"], w_inh=w_inh, w_mod=PARAMS["w_mod"],
                   bias_mv_s=PARAMS["bias_mv_s"], bias_cv=PARAMS["bias_cv"])
        t0 = time.perf_counter()
        st = c.run_resting(T_ms=T_MS, settle_ms=SETTLE_MS, seed=seed)
        rec = dict(arm=tag, w_inh=w_inh, v_floor=v_floor, seed=seed,
                   gaba_edge_scale=("1.0" if w_inh else "0.0"),
                   **{k: st[k] for k in ("rate_median_hz", "rate_mean_hz",
                                         "rate_p95_hz", "silent_frac", "pop_rate_hz",
                                         "n_spikes", "ms_per_step", "wall_s")})
        out.append(rec)
        print("  [%s] seed=%d 中位 %8.4f Hz / 均值 %8.4f / 静默 %.4f / 群体 %7.4f Hz "
              "/ %.3f ms/step（%.0fs）"
              % (tag, seed, st["rate_median_hz"], st["rate_mean_hz"], st["silent_frac"],
                 st["pop_rate_hz"], st["ms_per_step"], time.perf_counter() - t0), flush=True)
    return out


def main() -> int:
    from neural_exploration.src.adult_circuit import AdultCircuit, CircuitParams
    os.makedirs(REPORT_DIR, exist_ok=True)
    print("=== M9 P4 附加：真实 GABA 边 H6 消融 sanity（§3.5.5）===", flush=True)
    p = CircuitParams(**PARAMS)
    c = AdultCircuit(device="mps", params=p, use_compile=False)
    bs = c.build()
    gaba_edges = int(c.inh.sum())
    gaba_syn = int(c.syn_count[c.inh].sum())
    print("装配：%d 神经元 / %d 边；真实 GABA 边 %d（%.2f%%）/ GABA 突触 %d"
          % (c.n_neurons, c.n_edge, gaba_edges, 100.0 * gaba_edges / c.n_edge,
             gaba_syn), flush=True)
    n_steps = int(round((SETTLE_MS + T_MS) / p.dt_ms)) + 8
    g_ext = BG["epsp_mv"] / 0.104
    c.build_background(rate_hz=BG["rate_hz"], g_ext=g_ext, n_steps=n_steps, seed=BG["seed"])
    c._bg_steps = n_steps
    c._bg_args = (BG["rate_hz"], g_ext)

    seeds = list(range(N_TRIALS))
    rows = []
    for vf in V_FLOOR_ARMS:
        for tag, wi in (("baseline_real_gaba", PARAMS["w_inh"]), ("h6_gaba_off", 0.0)):
            rows += _run_arm(c, "%s_vfloor%.0f" % (tag, vf), seeds, wi, v_floor=vf)

    def agg(tag, key):
        return float(np.mean([r[key] for r in rows if r["arm"] == tag]))

    def arm_pair(vf):
        b = {k: agg("baseline_real_gaba_vfloor%.0f" % vf, k) for k in
             ("rate_median_hz", "rate_mean_hz", "silent_frac", "pop_rate_hz")}
        a = {k: agg("h6_gaba_off_vfloor%.0f" % vf, k) for k in b}
        return b, a, {k: a[k] - b[k] for k in b}

    results = {vf: arm_pair(vf) for vf in V_FLOOR_ARMS}
    base, abla, d = results[V_FLOOR_ARMS[0]]
    crit = {"rate_mean_up": d["rate_mean_hz"] >= 0.0,
            "silent_down": d["silent_frac"] <= 0.0,
            "pop_rate_up": d["pop_rate_hz"] >= 0.0}
    verdict = "PASS" if all(crit.values()) else "FAIL"
    for vf in V_FLOOR_ARMS:
        b, a, dd = results[vf]
        print("v_floor=%.0f mV | 基线 均值 %.4f 静默 %.4f 群体 %.4f → 消融 均值 %.4f "
              "静默 %.4f 群体 %.4f | Δ 率均值 %+.4f 静默 %+.4f 群体 %+.4f"
              % (vf, b["rate_mean_hz"], b["silent_frac"], b["pop_rate_hz"],
                 a["rate_mean_hz"], a["silent_frac"], a["pop_rate_hz"],
                 dd["rate_mean_hz"], dd["silent_frac"], dd["pop_rate_hz"]), flush=True)
    base, abla, d = results[V_FLOOR_ARMS[0]]
    print("H6 方向判定（v_floor=%.0f 定稿档）：%s" % (V_FLOOR_ARMS[0], verdict), flush=True)

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = _csv.writer(f, lineterminator="\n")
        w.writerow(["# M9 P4 附加：真实 GABA 边 H6 消融 sanity（§3.5.5；配对设计，拓扑不变）"])
        w.writerow(["section", "key", "value", "note"])
        w.writerow(["build", "n_neurons", c.n_neurons, ""])
        w.writerow(["build", "n_edges", c.n_edge, ""])
        w.writerow(["build", "gaba_edges", gaba_edges,
                    "真实抑制边（g→0 = H6 消融边集；来源 m9_flywire_connectome.csv 递质标注）"])
        w.writerow(["build", "gaba_edge_share_pct", "%.3f" % (100.0 * gaba_edges / c.n_edge),
                    "与 data/m9_inhibition_inventory.csv 21.42% 一致"])
        w.writerow(["build", "gaba_synapses", gaba_syn, "12,755,910（23.4%）"])
        w.writerow(["protocol", "T_ms", T_MS, "与 P4 同（§3.5.4 最短协议）"])
        w.writerow(["protocol", "settle_ms", SETTLE_MS, ""])
        w.writerow(["protocol", "n_trials", N_TRIALS, "固定 seed"])
        w.writerow(["protocol", "bg", "rate=%.2fHz epsp=%.1f mV" % (BG["rate_hz"], BG["epsp_mv"]),
                    "逐神经元独立 Poisson 虚拟突触驱动"])
        for r in rows:
            for k, v in r.items():
                w.writerow(["arm_" + r["arm"] + "_seed%d" % r["seed"], k, v, ""])
        w.writerow(["section", "key", "value", "note"])
        for vf in V_FLOOR_ARMS:
            b, a, dd = results[vf]
            for k in b:
                w.writerow(["agg", "vfloor%.0f_baseline_%s" % (vf, k), "%.6f" % b[k],
                            "真实 GABA 边臂（v_floor=%.0f mV）" % vf])
                w.writerow(["agg", "vfloor%.0f_h6off_%s" % (vf, k), "%.6f" % a[k],
                            "GABA 边 g→0 臂（v_floor=%.0f mV）" % vf])
                w.writerow(["agg", "vfloor%.0f_delta_%s" % (vf, k), "%.6f" % dd[k],
                            "消融 − 基线；Δ 率均值>0 且 Δ静默<0 = H6 方向正确"])
        for k, v in crit.items():
            w.writerow(["crit", k, str(bool(v)), "H6 方向断言"])
        w.writerow(["crit", "verdict", verdict,
                    "PASS=真实 GABA 边起抑制平衡作用（消融 → 活动↑/静默↓）"])
        w.writerow(["note", "m8_context",
                    "M8 冻结基线 run=9.9% vs 行为带 [60,85]%（缺 GABA 语境）；"
                    "M9 数据侧已解除（真实 GABA 21.42%）",
                    "本表只给 P4 静息统计，行为级 run 判据属 P3/P5（下一批次）"])
    print("→", OUT_CSV, flush=True)

    try:
        make_plot(rows, base, abla, d, crit)
        print("→", OUT_PNG, flush=True)
    except Exception as e:
        print("绘图失败（不影响数据）：%s" % e, flush=True)
    return 0 if verdict == "PASS" else 2


def make_plot(rows, base, abla, d, crit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    keys = ["rate_mean_hz", "silent_frac", "pop_rate_hz"]
    titles = ["Mean firing rate (Hz)", "Silent fraction (<0.5Hz)", "Population rate (Hz)"]
    for ax, k, t in zip(axes, keys, titles):
        b = [r[k] for r in rows if r["arm"] == "baseline_real_gaba"]
        a = [r[k] for r in rows if r["arm"] == "h6_gaba_off"]
        ax.bar([0, 1], [np.mean(b), np.mean(a)], yerr=[np.std(b), np.std(a)],
               color=["#1f77b4", "#d62728"], capsize=5)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["real GABA", "GABA off (H6)"])
        ax.set_title("%s\nΔ=%+.4f" % (t, d[k]))
    fig.suptitle("M9 P4 H6 ablation (real GABA edges g→0) — verdict %s"
                 % ("PASS" if all(crit.values()) else "FAIL"))
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=110)


if __name__ == "__main__":
    sys.exit(main())
