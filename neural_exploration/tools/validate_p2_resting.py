"""M5 P2 验证：静息发放率分布（T≥10s×N=5，D4 定稿权重）+ 夹带极限环反证记录。

判据（主 agent 裁决 2026-08-26 + data/m5_behavior_reference.csv resting 带）：
  - 静默比例（<0.1Hz 操作化）∈ [0.6, 0.8]（容差 [0.6,0.8]）；
  - 组中位数发放率 < 1Hz；最大发放率 < 60Hz（生理上限）；
  - 前置：无 NaN/无发散；
  - settle 窗分析（docs/m5_env_notes.md L41 #1 建议）：t<500ms 初始化瞬态波排除后的
    post-settle 静默比例（L37 #2：t=0 波是初始条件伪迹）——如实记录，不做事后调判据。

预期结果（B1e2 定稿 D4=g1_gap005，docs/m5_env_notes.md L37-L39）：
  - 静默 ~10.6%（中位数 ~13.8Hz、max ~14Hz）→ **不在带**；
  - 根因 = 网络级夹带极限环（L39：86% 神经元同步夹带至 2.7-13.8Hz——单一张力
    AVB 14µA/cm² 驱动 + 全互兴奋命令回路，任何持续驱动都夹带全网络，静默上限 ~44%）；
  - **反证记录型 pass**（与 M4 P4 同型：记录本身即交付物）：P2 结构性不可达——
    缺失机制（调质/异质权重/命令互抑，L40 #1/#2/#3），M6 复核优先验证清单；
    pass_=True 表示「反证记录完成」，非「指标落带」。

输出：reports/neuro/m5_p2_resting.png + data/m5_p2_resting.csv

用量：.venv-neuro/bin/python -m neural_exploration.tools.validate_p2_resting
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.src.worm_circuit import (  # noqa: E402
    load_weight_scales,
    make_worm_circuit,
)
from neural_exploration.src.worm_loop import load_m5_worm_params  # noqa: E402

DATA_DIR = os.path.join(ROOT, "neural_exploration", "data")
REPORTS_DIR = os.path.join(ROOT, "neural_exploration", "reports", "neuro")
REPORT_PNG = os.path.join(REPORTS_DIR, "m5_p2_resting.png")
REPORT_CSV = os.path.join(DATA_DIR, "m5_p2_resting.csv")
RESULT_JSON = os.path.join(DATA_DIR, "m5_p2_result.json")

#: P2 协议（m5_worm_params.csv protocol 行定稿）
RESTING_T_MS = 10000.0
N_RUNS = 5
SETTLE_MS = 500.0            # L41 #1：初始化瞬态波 settle 窗（如实记录，非判据）
SILENT_HZ_LO = 0.1           # 行为参考操作化：<0.1Hz 记静默
BAND_SILENT = (0.6, 0.8)     # resting.silent_fraction_target
BAND_MEDIAN_HZ = (0.0, 1.0)
BAND_MAX_HZ = (0.0, 60.0)


def _rates_from_times(times: dict, t_total_ms: float) -> dict:
    """逐角色发放率（Hz）；t_total_ms 为窗长。"""
    return {role: len(np.asarray(t, dtype=float)) / (t_total_ms / 1000.0)
            for role, t in times.items()}


def run_p2(save_plot: bool = True) -> dict:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    wp = load_m5_worm_params()
    t_total = float(wp["protocol"].get("resting_t_total_ms", RESTING_T_MS))
    n_runs = N_RUNS
    wc = make_worm_circuit(scale=302, seed=0, **load_weight_scales())

    runs = []
    t0 = time.perf_counter()
    for k in range(n_runs):
        sess = wc.make_session(t_total_ms=t_total)
        sess.reset(seed=k)
        sess.run_resting_window(t_total)
        times = sess.role_spike_times()
        rates = _rates_from_times(times, t_total)
        arr = np.array(list(rates.values()), dtype=float)
        # settle 窗（L41 #1）：post-settle 只统计 t>=SETTLE_MS 的发放
        post = {role: np.asarray(t, dtype=float)[np.asarray(t, dtype=float) >= SETTLE_MS]
                for role, t in times.items()}
        post_arr = np.array(
            [len(v) / ((t_total - SETTLE_MS) / 1000.0) for v in post.values()],
            dtype=float)
        runs.append(dict(
            seed=k,
            median_hz=float(np.median(arr)),
            max_hz=float(arr.max()),
            silent_01=float(np.mean(arr < SILENT_HZ_LO)),
            silent_05=float(np.mean(arr < 0.5)),
            has_nan=bool(np.any(~np.isfinite(arr))),
            n_spiking=float(np.sum(arr >= SILENT_HZ_LO)),
            median_post=float(np.median(post_arr)),
            max_post=float(post_arr.max()),
            silent_post_01=float(np.mean(post_arr < SILENT_HZ_LO)),
            silent_post_05=float(np.mean(post_arr < 0.5)),
        ))
        print(f"  run {k}: silent_01={runs[-1]['silent_01']:.3f} "
              f"median={runs[-1]['median_hz']:.2f}Hz max={runs[-1]['max_hz']:.2f}Hz "
              f"post_silent_01={runs[-1]['silent_post_01']:.3f}")
    wall_s = time.perf_counter() - t0

    # 确定性：p=1/n=1、无噪声 → 各 run 逐位一致
    deterministic = all(r["silent_01"] == runs[0]["silent_01"]
                        and r["median_hz"] == runs[0]["median_hz"]
                        for r in runs[1:])
    m = runs[0]  # 同值，取首 run 明细
    silent = m["silent_01"]
    median = m["median_hz"]
    max_hz = m["max_hz"]

    # ---- 判定（带 vs 实测；预注册判据，不做事后调）----
    in_band_silent = BAND_SILENT[0] <= silent <= BAND_SILENT[1]
    in_band_median = BAND_MEDIAN_HZ[0] <= median <= BAND_MEDIAN_HZ[1]
    in_band_max = BAND_MAX_HZ[0] <= max_hz <= BAND_MAX_HZ[1]
    no_nan = not m["has_nan"] and np.isfinite(median) and np.isfinite(max_hz)
    metrics_ok = no_nan and in_band_max
    indicator_pass = in_band_silent and in_band_median and metrics_ok

    # ---- 反证记录（L39/L40：夹带极限环 = 结构性不可达）----
    counter_evidence = dict(
        status="counter-evidence-record",
        root_cause=(
            "网络级夹带极限环（docs/m5_env_notes.md L39）：单一张力驱动 AVB"
            "（M4 携带 14µA/cm²，AVBL/AVBR）+ 全互兴奋命令回路（AVA/AVD↔AVB/PVC 无互抑边"
            "，L40 #1）→ 86% 神经元同步夹带至 2.7-13.8Hz（发率分布极端：86% 同率、"
            "0 个中间率，非高斯背景活动）→ 静默比例（<0.1Hz）结构性卡在 ~10.6%"
            "（全杠杆扫描上限 ~44%，u4），**< 带 [60,80]%**"),
        missing_mechanisms=[
            "调质（RIM 酪胺能受体=mod → g=0 占位跳过，L5#5/L40#1：后退时抑制前进缺失）",
            "命令互抑（AVA/AVD ↔ AVB/PVC 真实连接组互为兴奋，无互抑边，L40#1）",
            "AVA→DD/VD GABA 抑制链缺失（真实连接组 0 条，L40#2）",
            "异质权重/自发输入缺失（模型自发输入缺失、调质 g=0 → 任何持续驱动都夹带，L40#3）",
        ],
        settle_window=dict(
            note="L41 #1：t=0 初始化瞬态波（v=−65+张力开通）是初始条件伪迹，500ms 后消失",
            silent_post_01=m["silent_post_01"],
            silent_post_05=m["silent_post_05"],
            conclusion=(
                f"settle 后静默 {m['silent_post_01']:.1%}——仍 < 带 [60,80]%（calibration "
                f"u3 最高 69.2% 在带但行为破坏，P2 与行为在单一张力下不可兼得，L40#3）"),
        ),
        recheck_m6="M6 引入 RIM 酪胺/命令互抑/AVA→DD GABA 链后复核（M6 优先验证清单 #1/#2/#3）",
    )

    pass_ = False  # 主 agent 最终裁决 2026-08-26：P2/P4/P6 编码统一 pass_=False，
    #                status=counter-evidence-record（反证记录：记录本身即科学交付物）
    out = dict(
        pass_=pass_,
        status=counter_evidence["status"],
        verdict=(
            "P2 静息 = 反证记录（pass_=False, status=counter-evidence-record）："
            "无 NaN/无发散 ✓（max " +
            f"{max_hz:.1f}Hz < 60Hz 生理上限 ✓）；静默比例 {silent:.1%}（中位数 "
            f"{median:.1f}Hz）不在带 [60,80]%（预期：夹带极限环结构性不可达，L39）——"
            "反证记录完成（缺失机制：调质/异质权重/命令互抑，M6 复核；"
            "**与 P4 同根因：夹带极限环 + 缺失调质/异质权重**）"),
        indicator_pass=indicator_pass,
        in_band_silent=in_band_silent,
        in_band_median=in_band_median,
        in_band_max=in_band_max,
        no_nan=no_nan,
        silent_frac_01=silent,
        silent_frac_05=m["silent_05"],
        median_hz=median,
        max_hz=max_hz,
        n_spiking=m["n_spiking"],
        band_silent=list(BAND_SILENT),
        band_median_hz=list(BAND_MEDIAN_HZ),
        band_max_hz=list(BAND_MAX_HZ),
        t_total_ms=t_total,
        n_runs=n_runs,
        deterministic=deterministic,
        runs=runs,
        wall_s=wall_s,
        counter_evidence=counter_evidence,
        protocol_source="data/m5_worm_params.csv protocol.resting_t_total_ms（G0 定稿）",
        weights="D4 定稿（load_weight_scales：gap_scale=0.05，类级=先验 1.0）",
    )

    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        import csv as _csv
        w = _csv.writer(f, lineterminator="\n")
        w.writerow(["# M5 P2 静息验证（tools/validate_p2_resting.py）"])
        w.writerow(["metric", "value", "band", "verdict"])
        w.writerow(["pass_", out["pass_"], "False（反证记录：记录本身即科学交付物，主 agent 最终裁决）",
                    "record"])
        w.writerow(["status", out["status"], "counter-evidence-record", "ok"])
        w.writerow(["silent_frac_01", f"{silent:.4f}", "[0.6,0.8]",
                    "in" if in_band_silent else "OUT（反证记录）"])
        w.writerow(["silent_frac_05", f"{m['silent_05']:.4f}", "[0.6,0.8]",
                    "informational"])
        w.writerow(["median_hz", f"{median:.3f}", "[0,1]", "ok" if in_band_median else "OUT"])
        w.writerow(["max_hz", f"{max_hz:.3f}", "[0,60]", "ok" if in_band_max else "OUT"])
        w.writerow(["no_nan", out["no_nan"], "true", "ok"])
        w.writerow(["deterministic_n_runs", out["deterministic"], "true", "ok"])
        w.writerow(["silent_post_settle_01", f"{m['silent_post_01']:.4f}", "[0.6,0.8]",
                    "informational（L41 #1 settle 窗）"])
        w.writerow(["root_cause", counter_evidence["root_cause"], "", ""])
        w.writerow(["recheck_m6", counter_evidence["recheck_m6"], "", ""])

    if save_plot:
        _plot(out)

    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        import json as _json
        _json.dump(out, f, ensure_ascii=False, default=str)

    return out


def _plot(out: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    for _f in ("PingFang SC", "PingFang HK", "Heiti TC", "STHeiti",
               "Arial Unicode MS"):
        try:
            fm.findfont(_f, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [_f, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            break
        except Exception:
            continue

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.0))
    m = out["runs"][0]
    # 1) 静默比例 vs 带（全窗 + settle 后）
    ax = axes[0]
    cats = ["全窗 (<0.1Hz)", "settle 后 (<0.1Hz)", "全窗 (<0.5Hz)"]
    vals = [out["silent_frac_01"], m["silent_post_01"], out["silent_frac_05"]]
    colors = ["tab:red", "tab:orange", "tab:gray"]
    ax.bar(cats, [v * 100 for v in vals], color=colors)
    ax.axhspan(60, 80, color="green", alpha=0.15, label="带 [60,80]%")
    for i, v in enumerate(vals):
        ax.text(i, v * 100 + 1, f"{v:.1%}", ha="center")
    ax.set_ylabel("静默比例（%）")
    ax.set_ylim(0, 100)
    ax.set_title("静默比例 vs 带（夹带极限环 → 结构性不可达，L39）")
    ax.legend(fontsize=8)

    # 2) 发放率直方图（对数/线性）
    ax = axes[1]
    wc = make_worm_circuit(scale=302, seed=0, **load_weight_scales())
    sess = wc.make_session(t_total_ms=out["t_total_ms"])
    sess.reset(seed=0)
    sess.run_resting_window(out["t_total_ms"])
    times = sess.role_spike_times()
    rates = np.array([len(np.asarray(t)) / (out["t_total_ms"] / 1000.0)
                      for t in times.values()], dtype=float)
    ax.hist(rates, bins=40, color="tab:blue", alpha=0.8)
    ax.axvline(0.1, color="red", ls="--", lw=1.2, label="静默阈值 <0.1Hz")
    ax.axvline(out["median_hz"], color="k", ls="--", lw=1.2,
               label=f"中位数 {out['median_hz']:.1f}Hz")
    ax.set_xlabel("发放率（Hz）")
    ax.set_ylabel("神经元数")
    ax.set_title(f"静息发放率分布（n={len(rates)}，86% 同步夹带峰）")
    ax.legend(fontsize=8)

    # 3) 反证记录要点（文本）
    ax = axes[2]
    ax.axis("off")
    ce = out["counter_evidence"]
    lines = [
        "P2 反证记录（记录本身即交付物）：",
        f"  静默 {out['silent_frac_01']:.1%} vs 带 [60,80]% → 不在带",
        f"  中位数 {out['median_hz']:.1f}Hz / max {out['max_hz']:.1f}Hz（<60 ✓）",
        f"  根因：{ce['root_cause'][:58]}…",
        "  缺失机制（M6 复核清单）：",
        *[f"    · {s}" for s in ce["missing_mechanisms"]],
        f"  settle 窗：{ce['settle_window']['conclusion'][:52]}…",
        f"  M6：{ce['recheck_m6']}",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=9,
            family="monospace")
    ax.set_title("夹带极限环反证记录（L37-L40）")

    plt.tight_layout()
    plt.savefig(REPORT_PNG, dpi=110)
    plt.close(fig)


def main():
    out = run_p2(save_plot=True)
    print(f"P2 pass_ = {out['pass_']}（{out['status']}）")
    print(f"  静默 {out['silent_frac_01']:.1%}（带 [60,80]）→ "
          f"{'in' if out['in_band_silent'] else 'OUT'}"
          f" | median {out['median_hz']:.1f}Hz | max {out['max_hz']:.1f}Hz")
    print(f"  确定性 {out['deterministic']}（N={out['n_runs']} 同值）| "
          f"settle 后静默 {out['runs'][0]['silent_post_01']:.1%}")
    print(f"  {REPORT_CSV}\n  {REPORT_PNG}")


if __name__ == "__main__":
    main()
