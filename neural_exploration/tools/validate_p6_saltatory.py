"""P6 验证：郎飞结跳跃传导现象。

方法（清单 §5.5）：胞体刺激 → 逐隔室记录 V(t) 与发放时刻。
判定（M1 实测后定稿，判据理由见 m1_report.md §5.5）：
  1. 每个郎飞结（node1/2/3）都有动作电位（≥1 次发放）；
  2. 跳跃时序：节点发放时刻沿轴突严格递增（结处"跳跃"再发放）；
  3. 髓鞘段无主动通道：CSV 规格断言 gNa=gK=0；
  4. 髓鞘段无全幅峰：髓鞘最大峰值 < 0.75 × 驱动源（胞体/AIS）AP 峰值
     （被动波不可能超过其驱动源幅度；-20mV 越阈为被动波动，非再生发放）。
输出：reports/neuro/m1_saltatory.png（时空热图 + 各隔室 V 轨迹堆叠）
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.src.morphology import section_distances_micron  # noqa: E402
from neural_exploration.tools.load_morphology import load_morphology  # noqa: E402

REPORTS_DIR = os.path.join(ROOT, "neural_exploration", "reports", "neuro")
REPORT_PNG = os.path.join(REPORTS_DIR, "m1_saltatory.png")
#: “无全幅峰”判据：髓鞘峰值不超过驱动源 AP 峰值的该比例
FULL_AMP_RATIO = 0.75


def run_p6(save_plot: bool = True) -> dict:
    from neural_exploration.src.neuron_model import MultiCompartmentNeuron

    spec = load_morphology()
    n = MultiCompartmentNeuron(t_total_ms=15.0)
    r = n.run_stimulus(amplitude_uA_cm2=10.0, stim_start_ms=5.0,
                       stim_end_ms=6.0, record_all=True)

    t = r.t_ms
    dt = n.dt_ms

    nodes = ["node1", "node2", "node3"]
    myelin = ["myelin1", "myelin2", "myelin3"]

    node_info = {}
    for seg in nodes:
        st = r.spike_times_ms[seg]
        v = r.v_mv[f"comp{n.label_of(seg)}"]
        node_info[seg] = dict(
            n_spikes=len(st),
            t_first=float(st[0]) if len(st) else None,
            peak=float(v.max()),
            dvdt=float(np.max(np.abs(np.diff(v) / dt))),
        )

    myelin_info = {}
    for seg in myelin:
        peaks, dvdt_maxs, n_spikes = [], [], 0
        for i in n.index_map[seg]:
            v = r.v_mv[f"comp{i}"]
            peaks.append(float(v.max()))
            dvdt_maxs.append(float(np.max(np.abs(np.diff(v) / dt))))
            n_spikes += len(r.spike_times_ms.get(f"comp{i}", []))
        myelin_info[seg] = dict(
            peak_max=max(peaks), dvdt_max=max(dvdt_maxs), n_spikes=n_spikes,
            peaks=peaks)

    # 判据 1：每个郎飞结都发放
    all_nodes_fire = all(node_info[seg]["n_spikes"] > 0 for seg in nodes)
    # 判据 2：时序严格递增（跳跃）
    t_firsts = [node_info[seg]["t_first"] for seg in nodes]
    monotonic = all(a is not None and b is not None and b > a
                    for a, b in zip(t_firsts, t_firsts[1:]))
    # 判据 3：髓鞘无主动通道（构造断言）
    no_active_channels = all(
        spec.by_name(seg).gna_mS_cm2 == 0 and spec.by_name(seg).gk_mS_cm2 == 0
        for seg in myelin)
    # 判据 4：无全幅峰（相对驱动源——胞体/AIS AP）
    source_peak = float(r.v_mv[f"comp{n.label_of('soma')}"].max())
    myelin_peak_max = max(myelin_info[seg]["peak_max"] for seg in myelin)
    no_full_amplitude = bool(myelin_peak_max < FULL_AMP_RATIO * source_peak)

    pass_ = all_nodes_fire and monotonic and no_active_channels and no_full_amplitude

    if save_plot:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        order = ["soma"] + [s.name for s in spec.dendrite_chain()] + \
                [s.name for s in spec.axon_chain()]
        comps, labels, poss = [], [], []
        dists = section_distances_micron(spec, n.index_map)
        for seg in order:
            for k, i in enumerate(n.index_map[seg]):
                comps.append(i)
                labels.append(f"{seg}[{k}]")
                poss.append(dists[seg])
        poss = np.array(poss)
        V = np.vstack([r.v_mv[f"comp{i}"] for i in comps])

        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        im = axes[0].pcolormesh(t, poss, V, shading="nearest", cmap="RdBu_r",
                                vmin=-70, vmax=40)
        axes[0].set_ylabel("distance from soma (µm)")
        axes[0].set_title("P6: saltatory conduction — AP jumps between nodes (space-time)")
        fig.colorbar(im, ax=axes[0], label="V (mV)")

        offset, step = 0.0, 55.0
        for seg in order:
            color = ("#d62728" if seg.startswith("node") else
                     "#9e9e9e" if seg.startswith("myelin") else
                     "#2ca02c" if seg.startswith("dend") else
                     "#ff7f0e" if seg == "ais" else "#1f77b4")
            for k, i in enumerate(n.index_map[seg]):
                v = r.v_mv[f"comp{i}"]
                lw = 1.6 if seg.startswith(("node", "soma", "ais")) else 0.9
                axes[1].plot(t, v + offset, color=color, lw=lw)
                axes[1].text(0.3, offset + 6, labels[comps.index(i)],
                             fontsize=6.5, color=color)
                offset += step
        axes[1].axhline(-20, color="gray", ls="--", lw=0.6, alpha=0.6)
        axes[1].set_ylabel("V (mV, offset stacked)")
        axes[1].set_xlabel("t (ms)")
        axes[1].set_xlim(7.2, 11.0)
        axes[1].set_title("Stacked V(t) per compartment (red=nodes, gray=myelin)")
        fig.tight_layout()
        fig.savefig(REPORT_PNG, dpi=150)
        plt.close(fig)

    return dict(
        pass_=pass_,
        nodes=node_info,
        myelin=myelin_info,
        all_nodes_fire=all_nodes_fire,
        monotonic=monotonic,
        no_active_channels=no_active_channels,
        no_full_amplitude=no_full_amplitude,
        full_amp_ratio=FULL_AMP_RATIO,
        source_peak_mv=float(source_peak),
        myelin_peak_max_mv=float(myelin_peak_max),
        report_png=REPORT_PNG,
    )


if __name__ == "__main__":
    import json
    res = run_p6()
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print("P6 PASS" if res["pass_"] else "P6 FAIL")

