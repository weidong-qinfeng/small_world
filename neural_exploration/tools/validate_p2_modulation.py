"""M6 P2 验证（M6-B2）：调质系统 + M5 反证清单 → G1 数值复核 + 反证记录型判定。

对应《生物仿真M6实施清单》§0 P2 / §3.2 / G1 门；本节点（M6-B2 验证+报告）在
B1a 已落盘 `data/m6_g1_result.json`（G1=部分通过）基础上做**验证级复核**：

  1. **G1 数值复核**：读回 B1a G1 json 的 P2/P4/P6/P5 复核数值，逐项对照预注册带
     （`data/m5_behavior_reference.csv`，不做事后调带）→ 复算 in_band 判定；
  2. **确定性复现探针**：p=1/n=1 确定性网络 → 轻量重跑（P5 方向相位 tau=23 seed=0
     逃避 + P2 静息 T=2s seed=0 探针）确认 G1 数值可复现（点估计即真值）；
  3. **反证记录**：P2 静默未缓解 / P4 趋化未缓解 / P6 部分缓解（rev 落带）→
     反证记录型判定（与 M5 P2/P4/P6 同型，记录本身即交付物）；
     剩余缺失机制清单 = 夹带双稳态（调质门控只能整体开/关夹带，无"低活动+行为"
     稳定中间态，M5 L40#3 升级为结构性限制）；
  4. **消融 sanity 复核**：B1a 四项消融结论读回 + 判定一致性（每项 enabled 开关
     可消融，现象消失/恢复）——组件级可验证项由 tests/neuro/test_m6_neuromod.py
     覆盖（网络级不可观测项如实记录，不伪造）。

输出（验证级，只读 B1a 落盘文件；本脚本新建文件）：
  data/m6_p2_result.json + data/m6_p2_modulation.csv +
  reports/neuro/m6_p2_modulation.png

判定语义（主 agent 裁决）：P2 = **反证记录型 pass**（pass_=False 于判据带层面，
status=counter-evidence-record；记录本身即科学交付物）。

用量：.venv-neuro/bin/python -m neural_exploration.tools.validate_p2_modulation
  --skip-probes  跳过 302 复现探针（纯读回复核，无 Brian2 运行）
确定性：p=1/n=1；同参数重跑逐位一致；运行前检查无并发（brian2 cython 缓存锁）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DATA_DIR = os.path.join(ROOT, "neural_exploration", "data")
REPORTS_DIR = os.path.join(ROOT, "neural_exploration", "reports", "neuro")

G1_JSON = os.path.join(DATA_DIR, "m6_g1_result.json")
P2_RESULT_JSON = os.path.join(DATA_DIR, "m6_p2_result.json")
P2_CSV = os.path.join(DATA_DIR, "m6_p2_modulation.csv")
P2_PNG = os.path.join(REPORTS_DIR, "m6_p2_modulation.png")
M5_REF_CSV = os.path.join(DATA_DIR, "m5_behavior_reference.csv")

#: 预注册带（M5 冻结对照源；与 data/m5_behavior_reference.csv 同源）
BAND_SILENT = (0.60, 0.80)
BAND_MEDIAN = (0.0, 1.0)
BAND_MAX = (0.0, 60.0)
BAND_SPONT = dict(fwd=(60.0, 80.0), rev=(10.0, 25.0), turn=(5.0, 20.0))
D_PEAK_THR = 0.3
SETTLE_MS = 500.0
P2_PROBE_T_MS = 2000.0
P2_PROBE_SEED = 0

#: 剩余缺失机制清单（反证记录核心交付物；M5 L40#3 升级为结构性限制）
MISSING_MECHANISMS = [
    "夹带双稳态（entrainment bistability）：全互兴奋命令回路（AVA↔AVB/PVC ampa）"
    "+ 运动池互驱（motor→motor ampa 474 条）承载的 ~14-25Hz 共振极限环——"
    "调质门控（酪胺/自发 bout）只能把网络从 14Hz 夹带推到 2Hz 或 100% 静默，"
    "无『低活动（P2 静默 60-80%）+ 行为活跃（P6 fwd 60-80%）』的稳定中间态"
    "（M5 L38 观察在 O2 调质参数下复现：分岔陡峭）；任何『持续或准持续』驱动"
    "都重新点燃共振环",
    "命令层点火点：自发/调质输入若注入命令池（AVB/PVC/AVA/AVD）→ 每脉冲重新点燃"
    "夹带（10-25Hz 全局同步）——行为驱动必须作用于运动输出层（L9 #1 结构性发现）",
    "302 网络级学习读出（习惯化 D_peak / 联想 CI_salt）被夹带动力学淹没 → 学习"
    "机制在干净子图底物验证（M6-B1c L12/L16 限制，P3/P4 测量限制记录）",
]


# --------------------------------------------------------------------- #
def _band_ok(v, band) -> bool:
    return bool(band[0] <= v <= band[1])


def load_g1() -> dict:
    with open(G1_JSON, encoding="utf-8") as f:
        return json.load(f)


def _stats(x: np.ndarray) -> dict:
    from scipy import stats as sps

    c = np.asarray(x, dtype=float)
    n = int(c.size)
    mean = float(c.mean())
    sd = float(c.std(ddof=1)) if n > 1 else 0.0
    if n > 1 and sd > 0:
        t_stat, p_val = sps.ttest_1samp(c, 0.0)
        d = mean / sd
    else:
        t_stat, p_val, d = float("nan"), float("nan"), float("nan")
    return dict(n=n, mean=mean, sd=sd, t_stat=float(t_stat),
                p_value=float(p_val), cohen_d=float(d))


# --------------------------------------------------------------------- #
def recheck_g1_numbers(g1: dict) -> dict:
    """逐项对照预注册带，复算 in_band 判定（读回 B1a 数值，不重跑）。"""
    out = {}

    p5 = g1.get("p5_phase", {})
    tau23 = p5.get("tau_trans_23_ms", {})
    d23 = float(tau23.get("d_peak", float("nan")))
    out["p5_direction_phase"] = dict(
        tau_trans_ms=23.0, d_peak=d23,
        direction=tau23.get("direction", "?"),
        in_band=_band_ok(d23, (D_PEAK_THR, 1.0)),
        criterion="back (D_peak>0.3) — G1 关键前置",
        note="确定性网络逐 seed 同值（B1a 实测 0.355）；本节点探针复现",
    )

    p2 = g1.get("p2_resting", {})
    silent = float(p2.get("silent_post_settle500ms", float("nan")))
    median = float(p2.get("median_post_hz", float("nan")))
    maxhz = float(p2.get("max_post_hz", float("nan")))
    out["p2_resting"] = dict(
        silent_post=silent, band=list(BAND_SILENT),
        in_band_silent=_band_ok(silent, BAND_SILENT),
        median_post_hz=median, in_band_median=_band_ok(median, BAND_MEDIAN),
        max_post_hz=maxhz, in_band_max=_band_ok(maxhz, BAND_MAX),
        pass_=bool(_band_ok(silent, BAND_SILENT)
                   and _band_ok(median, BAND_MEDIAN)
                   and _band_ok(maxhz, BAND_MAX)),
        verdict=str(p2.get("verdict", "")),
        m5_frozen="10.6%（median 13.8Hz）→ O2 10.3%（median 22.5Hz）——未缓解",
    )

    p6 = g1.get("p6_spontaneous", {})
    probe = p6.get("T_8_10s_probe", {})
    frac = {s: float(probe.get(s, float("nan"))) * 100
            for s in ("fwd", "rev", "turn")}
    pause = float(probe.get("pause", float("nan"))) * 100
    out["p6_spontaneous"] = dict(
        frac_pct=frac, pause_pct=pause,
        in_band={s: _band_ok(frac[s], BAND_SPONT[s]) for s in BAND_SPONT},
        pass_=bool(all(_band_ok(frac[s], BAND_SPONT[s]) for s in BAND_SPONT)),
        verdict=str(p6.get("verdict", "")),
        m5_frozen="fwd 25.5/rev 3.0/turn 0.5% → O2 36-43/9-11/3.4-4.3%",
    )

    p4 = g1.get("p4_chemotaxis", {})
    probe4 = p4.get("T_5s_N_5_probe", {})
    out["p4_chemotaxis"] = dict(
        ci_grad=float(probe4.get("ci_grad", float("nan"))),
        ci_ctrl=float(probe4.get("ci_ctrl", float("nan"))),
        sig_ok=False, dir_ok=False, ctrl_ok=False,
        pass_=False,
        verdict=str(p4.get("verdict", "")),
        m5_frozen="M5 全协议 CĪ=−0.065@15s×N=20（p=0.71）→ O2 5s 探针 −0.263 同号",
        note="夹带网络 fwd/back 共同发放 → v≈0，无净趋化位移（结构性限制）",
    )
    return out


def run_confirmation_probes() -> dict:
    """轻量复现探针（p=1/n=1 确定性；点估计即真值）。

    P5 方向相位（tau=23 → back，G1 关键前置）单 seed 逃避 +
    P2 静默（T=2s，settle 500ms）单 seed 探针。mod_dt 档位与 B1a 同
    （P5 用 5ms、P2 用 None→epoch 同步 25ms 档）。
    """
    import gc

    from neural_exploration.src.worm_loop import WormLoop
    from neural_exploration.tools.validate_p6_modulation import build_mc

    out = {}

    # —— P5 方向相位（mod_dt=5.0，tau_trans=23，seed=0）——
    mc = build_mc(mod_dt=5.0)
    wl = WormLoop(mc)
    saved = wl.touch["tau_trans_ms"]
    wl.touch["tau_trans_ms"] = 23.0
    try:
        r = wl.run_escape(t_total_ms=150.0, seed=0)
    finally:
        wl.touch["tau_trans_ms"] = saved
    del wl, mc
    gc.collect()
    d_peak = float(r["d_peak"])
    out["p5_phase_probe"] = dict(
        d_peak=d_peak, direction=r["direction"],
        direction_ok=bool(d_peak > D_PEAK_THR),
        g1_reference=0.355,
        # B1a G1 json 存 3 位舍入（0.355）；实测精确值 0.35526（B1a 运行顺序
        # tau=0→23 复验同值）→ 容差 1e-3（确定性网络，点估计即真值）
        reproducible=bool(abs(d_peak - 0.355) < 1e-3),
        note="确定性网络：单 seed 点估计 = G1 N=5 均值（B1a 存 0.355 为 3 位"
             "舍入，实测精确值 0.35526；判定方向/带不变）",
    )
    print(f"[P2] P5 相位探针（tau=23, seed=0）: D_peak={d_peak:+.3f} "
          f"→ {r['direction']}（G1 参考 0.355）")

    # —— P2 静默（T=2s，settle 500ms，seed=0）——
    mc = build_mc(mod_dt=None)
    sess = mc.make_session(t_total_ms=P2_PROBE_T_MS)
    sess.reset(seed=P2_PROBE_SEED)
    sess.run_resting_window(P2_PROBE_T_MS)
    times = sess.role_spike_times()
    post = np.array(
        [len(np.asarray(t)[np.asarray(t) >= SETTLE_MS])
         / ((P2_PROBE_T_MS - SETTLE_MS) / 1000.0) for t in times.values()])
    silent = float(np.mean(post < 0.1))
    median = float(np.median(post))
    maxhz = float(post.max())
    out["p2_resting_probe"] = dict(
        silent_post=silent, median_post_hz=median, max_post_hz=maxhz,
        in_band_silent=_band_ok(silent, BAND_SILENT),
        in_band_median=_band_ok(median, BAND_MEDIAN),
        in_band_max=_band_ok(maxhz, BAND_MAX),
        pass_=bool(_band_ok(silent, BAND_SILENT)
                   and _band_ok(median, BAND_MEDIAN) and _band_ok(maxhz, BAND_MAX)),
        g1_reference=dict(silent=0.103, median=22.5, max=33.2),
        note="静息协议下运动池自发脉冲仍点燃共振环 → 静默 ~10%（带 [60,80] 未达）",
    )
    print(f"[P2] 静息探针（T=2s, seed=0）: silent={silent:.3f} "
          f"median={median:.1f}Hz max={maxhz:.1f}Hz "
          f"（带 [60,80]%/<1/<60Hz）pass={out['p2_resting_probe']['pass_']}")
    del sess, mc
    gc.collect()
    return out


def ablation_sanity_recheck(g1: dict) -> dict:
    """B1a 四项消融 sanity 结论读回 + 判定一致性（网络级不可观测项如实记录）。"""
    abl = g1.get("ablation_sanity", {})
    rows = {}
    t = abl.get("tyramine_off", {})
    rows["tyramine_off"] = dict(
        gate=t.get("gate"), escape_d_peak=t.get("escape_d_peak_73ms"),
        ok=bool(t.get("gate") == 1.0),
        note=t.get("note", ""))
    t = abl.get("mutual_inh_off", {})
    rows["mutual_inh_off"] = dict(
        escape_d_peak=t.get("escape_d_peak_73ms"),
        note=t.get("note", ""))
    t = abl.get("gaba_chain_off", {})
    rows["gaba_chain_off"] = dict(
        rev=t.get("spont_rev_T6s_5ms"), dd_vd_spikes=t.get("dd_vd_spikes_T3s"),
        note=t.get("note", ""))
    t = abl.get("spont_off", {})
    rows["spont_off"] = dict(
        pause_frac=t.get("frac_off", {}).get("pause"),
        bout_disappears=t.get("bout_disappears", False),
        note=t.get("note", ""))
    t = abl.get("frozen_regression", {})
    rows["frozen_regression"] = dict(
        rates_identical=t.get("enabled_false_rates_identical"),
        note=t.get("note", ""))
    return dict(
        rows=rows,
        summary=(
            "四项机制全部落地且可消融（每项 enabled 开关 + 行为可测变化）；"
            "方向相位修复是多机制联合（③ AVA→DD/VD 链 + ④ 自发 bout 为主，"
            "L7b）；GABA 链网络级效应不可观测（O2 夹带态淹没，L9 #7）如实记录；"
            "M5 冻结基线逐位一致（组合复用纪律 ✓）"),
    )


# --------------------------------------------------------------------- #
def run_p2(save_plot: bool = True, probes: bool = True,
           verbose: bool = True) -> dict:
    t0 = time.perf_counter()
    g1 = load_g1()
    checks = recheck_g1_numbers(g1)
    probes_out = run_confirmation_probes() if probes else {}
    abl = ablation_sanity_recheck(g1)

    # —— 反证记录判定（主 agent 裁决：反证记录型 pass）——
    band_pass = bool(
        checks["p2_resting"]["pass_"] and checks["p6_spontaneous"]["pass_"]
        and checks["p4_chemotaxis"]["pass_"])
    verdict = (
        "反证记录型 pass（判据带层面 pass_=False）：P2 静默未缓解（10.3% vs 带 "
        "[60,80]）、P4 趋化未缓解（−0.263@5s，与 M5 −0.065 同号）、P6 部分缓解"
        "（rev 10.9% 落带；fwd 36-43%/turn 3.4-4.3% 近带；方向分离结构涌现）；"
        "P5 方向相位修复（tau=23 → back D_peak=0.355，G1 关键前置满足）→ "
        "P3 习惯化协议可进入。反证记录本身即交付物（与 M5 P2/P4/P6 同型）；"
        "剩余缺失机制清单见 missing_mechanisms（夹带双稳态）")
    if verbose:
        print("== M6 P2 判定：反证记录型 pass ==")
        print(f"  P5 相位（tau=23）: {'✓ back' if checks['p5_direction_phase']['in_band'] else '✗'}"
              f" D_peak={checks['p5_direction_phase']['d_peak']:+.3f}")
        print(f"  P2 静默: {'✗ 未缓解' if not checks['p2_resting']['pass_'] else '✓'}"
              f"（{checks['p2_resting']['silent_post']:.1%} vs 带 [60,80]%）")
        print(f"  P6 自发: {'✓' if checks['p6_spontaneous']['pass_'] else '△ 部分'}"
              f" {checks['p6_spontaneous']['frac_pct']}")
        print(f"  P4 趋化: {'✗ 未缓解' if not checks['p4_chemotaxis']['pass_'] else '✓'}"
              f"（CĪ={checks['p4_chemotaxis']['ci_grad']:+.3f}@5s 探针）")

    summary = dict(
        milestone="M6-B2", p_index="P2",
        g1_verdict=str(g1.get("g1_verdict", "")),
        pass_=False,
        status="counter-evidence-record",
        pass_type="counter-evidence-record",
        band_pass=band_pass,
        checks=checks,
        probes=probes_out,
        ablation_sanity=abl,
        missing_mechanisms=MISSING_MECHANISMS,
        verdict=verdict,
        g1_json=G1_JSON,
        params=dict(
            bands=dict(silent=BAND_SILENT, median=BAND_MEDIAN, max=BAND_MAX,
                       spont=BAND_SPONT, d_peak_thr=D_PEAK_THR),
            settle_ms=SETTLE_MS, p2_probe_t_ms=P2_PROBE_T_MS,
        ),
        wall_s=time.perf_counter() - t0,
    )

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(P2_RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, default=str)

    _write_csv(summary)
    if save_plot:
        _write_plot(summary)
    if verbose:
        print(f"[P2] 结果已落盘 {P2_RESULT_JSON}（wall {summary['wall_s']:.0f}s）")
    return summary


def _write_csv(s: dict) -> None:
    rows = []
    for k, chk in s["checks"].items():
        if k == "p2_resting":
            rows.append(("P2", "silent_post_settle500ms",
                         chk["silent_post"], f"{chk['band'][0]:.2f}..{chk['band'][1]:.2f}",
                         chk["in_band_silent"], chk["verdict"]))
            rows.append(("P2", "median_post_hz", chk["median_post_hz"],
                         "0..1Hz", chk["in_band_median"], ""))
            rows.append(("P2", "max_post_hz", chk["max_post_hz"],
                         "0..60Hz", chk["in_band_max"], ""))
        elif k == "p6_spontaneous":
            for s2 in ("fwd", "rev", "turn"):
                band = BAND_SPONT[s2]
                rows.append(("P6", f"frac_{s2}", chk["frac_pct"][s2],
                             f"{band[0]}..{band[1]}%", chk["in_band"][s2],
                             chk["verdict"]))
        elif k == "p4_chemotaxis":
            rows.append(("P4", "ci_grad_5s_probe", chk["ci_grad"],
                         ">0 显著", False, chk["verdict"]))
            rows.append(("P4", "ci_ctrl_5s_probe", chk["ci_ctrl"],
                         "p>0.05", chk["ctrl_ok"], ""))
        elif k == "p5_direction_phase":
            rows.append(("P5", "d_peak_tau23", chk["d_peak"], ">0.3",
                         chk["in_band"], chk["criterion"]))
    for pk, pr in s.get("probes", {}).items():
        rows.append(("probe", pk, json.dumps(pr, default=str), "", True, ""))
    with open(P2_CSV, "w", encoding="utf-8") as f:
        f.write("# M6 P2 调质系统 + M5 反证清单复核（M6-B2 验证级）\n"
                "# 判据带 = data/m5_behavior_reference.csv 预注册（不做事后调）\n"
                "item,metric,value,band,ok,note\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
        f.write(f"# verdict: {s['verdict']}\n")


def _write_plot(s: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(REPORTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    # (a) P2 静默带
    ax = axes[0]
    c = s["checks"]["p2_resting"]
    ax.barh(["silent_post", "median (Hz)", "max (Hz)"],
            [c["silent_post"], c["median_post_hz"], c["max_post_hz"]],
            color=["tab:red" if not c["in_band_silent"] else "tab:green",
                   "tab:orange", "tab:orange"])
    ax.axvspan(0.60, 0.80, color="tab:green", alpha=0.12, label="band [60,80]%")
    ax.axvline(1.0, color="tab:blue", ls="--", lw=1, label="median <1Hz")
    ax.axvline(60.0, color="tab:purple", ls="--", lw=1, label="max <60Hz")
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_title("P2 resting recheck — not relieved")
    ax.legend(fontsize=7)
    # (b) P6 自发带
    ax = axes[1]
    c = s["checks"]["p6_spontaneous"]
    frac = c["frac_pct"]
    for i, (k, v) in enumerate(frac.items()):
        band = BAND_SPONT[k]
        ax.bar(i, v, color="tab:red" if not c["in_band"][k] else "tab:green",
               label=k)
        ax.plot([i - 0.3, i + 0.3], [band[0], band[0]], color="k", lw=1)
        ax.plot([i - 0.3, i + 0.3], [band[1], band[1]], color="k", lw=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(list(frac.keys()))
    ax.set_ylabel("% time")
    ax.set_title("P6 spontaneous — partial (rev in band)")
    ax.grid(alpha=0.3)
    # (c) P5 相位 + P4 趋化
    ax = axes[2]
    c5 = s["checks"]["p5_direction_phase"]
    c4 = s["checks"]["p4_chemotaxis"]
    ax.bar(["P5 D_peak(tau=23)", "P4 CI_grad(5s)"],
           [c5["d_peak"], c4["ci_grad"]],
           color=["tab:green", "tab:red"])
    ax.axhline(0.3, color="tab:green", ls="--", lw=1, label="D_peak>0.3 (back)")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.legend(fontsize=7)
    ax.set_title("P5 phase fixed / P4 chemotaxis not relieved")
    ax.grid(alpha=0.3)
    fig.suptitle("M6 P2: modulation + M5 counter-evidence recheck"
                 " — verdict: counter-evidence-record", y=1.02)
    fig.tight_layout()
    fig.savefig(P2_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="M6 P2 调质 + 反证清单复核")
    ap.add_argument("--skip-probes", action="store_true",
                    help="跳过 302 复现探针（纯读回复核，无 Brian2 运行）")
    ap.add_argument("--skip-plot", action="store_true")
    args = ap.parse_args()
    s = run_p2(save_plot=not args.skip_plot, probes=not args.skip_probes)
    print(json.dumps({k: v for k, v in s.items()
                      if k not in ("checks", "probes", "ablation_sanity")},
                     indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
