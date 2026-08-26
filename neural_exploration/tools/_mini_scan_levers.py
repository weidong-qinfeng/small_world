"""M5-B1e 杠杆方向快速扫描：2-3 组粗缩放看各目标方向（临时脚本）。

问题：302 占位权重下 静息 8.6% 静默 / 自发全 pause / 逃避 DA/VA 触前自发发放（lat<0）。
目标：类级缩放 s_k + tonic 缩放把四目标拉进带。
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.src.worm_circuit import make_worm_circuit  # noqa: E402
from neural_exploration.src.worm_loop import WormLoop  # noqa: E402

BUCKETS = [("sensory", "sensory"), ("sensory", "inter"), ("sensory", "motor"),
           ("inter", "inter"), ("inter", "motor"), ("inter", "sensory"),
           ("motor", "inter"), ("motor", "motor"), ("motor", "sensory"),
           ("pharyngeal", "pharyngeal")]


def run_combo(cs, tonic_scale=1.0, tag="", rest_ms=5000.0, spont_ms=5000.0,
              chem_t_ms=5000.0, chem_n=5, esc_n=3, seed_base=0):
    t0 = time.perf_counter()
    circ = make_worm_circuit(scale=302, fidelity="point", class_scales=cs)
    # tonic 缩放（AVBL/AVBR M4 行为上下文）
    if tonic_scale != 1.0:
        for k in circ.sub.tonic_uA_cm2:
            circ.sub.tonic_uA_cm2[k] *= tonic_scale
    loop = WormLoop(circ)

    r = circ.run_resting(t_total_ms=rest_ms)
    rates = np.array(list(r["rates_hz"].values()), dtype=float)
    sp = circ.run_spontaneous(t_total_ms=spont_ms)
    res, meta = circ.run_chemotaxis_trials(n_trials=chem_n, t_total_ms=chem_t_ms,
                                           seed_base=seed_base)
    ci = np.array([x.ci for x in res], dtype=float)
    d_peaks, lats = [], []
    for k in range(esc_n):
        esc = loop.run_escape(t_total_ms=150.0, seed=seed_base + k)
        d_peaks.append(esc["d_peak"]); lats.append(esc["neural_latency_ms"])
    wall = time.perf_counter() - t0
    line = (f"[{tag}] tonic={tonic_scale:.2f} "
            f"rest(sil01={np.mean(rates<0.1)*100:.0f}% med={np.median(rates):.2f}Hz "
            f"max={r['max_hz']:.1f}) "
            f"spont(fwd={sp['frac']['fwd']*100:.0f} rev={sp['frac']['rev']*100:.0f} "
            f"turn={sp['frac']['turn']*100:.0f} pau={sp['frac']['pause']*100:.0f}) "
            f"ci={np.mean(ci):.3f}±{np.std(ci,ddof=1)/np.sqrt(len(ci)):.3f} "
            f"esc(D_peak={np.max(d_peaks):.3f} lat={np.nanmedian(lats):.1f}ms) "
            f"wall={wall:.0f}s")
    print(line, flush=True)
    return dict(tag=tag, cs=cs, tonic=tonic_scale, rest_sil01=float(np.mean(rates<0.1)),
                rest_med=float(np.median(rates)), rest_max=float(r["max_hz"]),
                spont=dict(sp["frac"]), ci=float(np.mean(ci)),
                esc_dpeak=float(np.max(d_peaks)), esc_lat=float(np.nanmedian(lats)),
                wall=wall)


def main():
    allc = {b: 1.0 for b in BUCKETS}
    # C1: 全局 ampa 桶 ×0.3（gaba 不变）、tonic 1.0
    cs1 = dict(allc); 
    for b in BUCKETS:
        cs1[b] = 0.3
    run_combo(cs1, tonic_scale=1.0, tag="C1_all03_t1")
    # C2: 同 C1 + tonic ×0.5
    run_combo(cs1, tonic_scale=0.5, tag="C2_all03_t05")
    # C3: 分层：inter-inter/motor-motor 0.2、inter-motor 0.4、sensory-inter 0.6、其余 0.5
    cs3 = dict(allc)
    for b in BUCKETS:
        cs3[b] = 0.5
    cs3[("inter", "inter")] = 0.2
    cs3[("motor", "motor")] = 0.2
    cs3[("inter", "motor")] = 0.4
    cs3[("sensory", "inter")] = 0.6
    run_combo(cs3, tonic_scale=0.5, tag="C3_layered_t05")
    # C4: 同 C3 但 tonic 1.0
    run_combo(cs3, tonic_scale=1.0, tag="C4_layered_t1")


if __name__ == "__main__":
    main()
