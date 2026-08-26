"""M5-B1e 权重校准：302 占位权重基线探针（临时脚本，正式校准见 scan_m5_weight_calibration.py）。

跑四个短协议（T≤5s）测当前行为 + 墙钟，确认 G0 L22 记录并定校准起点。
确定性：p=1/n=1；方差来自伪随机起点。预算：~5 min（302 grouped 编译缓存预热后）。
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


def run_combo(cs, tag, rest_ms=3000.0, spont_ms=3000.0, chem_t_ms=3000.0,
              chem_n=3, esc_n=2, seed_base=0):
    """一组类级缩放下的四行为短协议快照（返回指标 dict + 墙钟）。"""
    out = {"tag": tag, "class_scales": dict(cs) if cs else {}}
    t0 = time.perf_counter()
    circ = make_worm_circuit(scale=302, fidelity="point", class_scales=cs)
    loop = WormLoop(circ)

    # 静息（无刺激；silent 按 <0.1Hz 与 <0.5Hz 两个操作化都报告）
    r = circ.run_resting(t_total_ms=rest_ms)
    rates = np.array(list(r["rates_hz"].values()), dtype=float)
    out["rest_median"] = float(np.median(rates))
    out["rest_max"] = float(r["max_hz"])
    out["rest_silent_01"] = float(np.mean(rates < 0.1))
    out["rest_silent_05"] = float(np.mean(rates < 0.5))
    out["rest_nan"] = bool(r["has_nan"])
    out["rest_wall"] = r["wall_s"]
    # 按类发放率（诊断）
    cls_rates = {}
    for role, hz in r["rates_hz"].items():
        c = circ.sub.neurons.get(role, "?")
        cls_rates.setdefault(c, []).append(hz)
    out["rest_by_class"] = {k: (float(np.median(v)), float(np.mean(np.array(v) < 0.1)))
                            for k, v in cls_rates.items()}

    # 自发（无刺激 s=0；状态比例）
    sp = circ.run_spontaneous(t_total_ms=spont_ms)
    out["spont"] = {k: float(v) for k, v in sp["frac"].items()}
    out["spont_wall"] = sp["wall_s"]

    # 趋化短协议（T=3s × N）
    res, meta = circ.run_chemotaxis_trials(n_trials=chem_n, t_total_ms=chem_t_ms,
                                           seed_base=seed_base)
    ci = np.array([x.ci for x in res], dtype=float)
    out["ci"] = float(np.mean(ci))
    out["ci_sem"] = float(np.std(ci, ddof=1) / np.sqrt(len(ci))) if len(ci) > 1 else float("nan")
    out["ci_n"] = len(ci)
    out["chem_wall_mean"] = float(np.mean(meta["wall_s"]))

    # 逃避（302 全虫：touch→PLM/ALM→缝隙→PVC→AVA→DA→back）
    d_peaks, lats = [], []
    for k in range(esc_n):
        esc = loop.run_escape(t_total_ms=150.0, seed=seed_base + k)
        d_peaks.append(esc["d_peak"])
        lats.append(esc["neural_latency_ms"])
        out["esc_touch_roles"] = esc["touch_roles"]
    out["esc_d_peak"] = float(np.max(d_peaks))
    out["esc_lat"] = float(np.nanmedian(lats))
    out["esc_wall"] = esc["wall_s"]
    out["total_wall"] = time.perf_counter() - t0
    return out


def main():
    cs = None  # 占位（恒等）
    o = run_combo(cs, tag="baseline_placeholder")
    print("=" * 80)
    print("BASELINE 302 placeholder weights (s_k=1 all)")
    for k, v in o.items():
        if k in ("class_scales",):
            continue
        print(f"  {k}: {v}")
    print("=" * 80)


if __name__ == "__main__":
    main()
