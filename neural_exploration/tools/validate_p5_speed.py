"""P5 验证：轴突传导速度。

方法（清单 §5.4）：刺激胞体 → 记录各郎飞结首个发放时刻 →
距离（各节中心到胞体中心，µm）/ 时间差 → 传导速度（m/s）。
判定（清单 §0 P5）：无髓鞘 0.5–2 m/s；有髓鞘跳跃传导更快（报告实际值并对照文献量级）。
输出：data/m1_conduction_speed.csv + reports/neuro/m1_saltatory.png 复用
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

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
REPORT_CSV = os.path.join(DATA_DIR, "m1_conduction_speed.csv")

# 判定参考（清单 §0 P5）：有髓鞘跳跃传导的文献量级（薄髓鞘纤维 1–20 m/s）
CV_RANGE_MPS = (1.0, 20.0)


def run_p5() -> dict:
    from neural_exploration.src.neuron_model import MultiCompartmentNeuron

    spec = load_morphology()
    n = MultiCompartmentNeuron(t_total_ms=80.0)
    # 触发单次 AP 并向轴突传导
    r = n.run_stimulus(amplitude_uA_cm2=10.0, stim_start_ms=5.0,
                       stim_end_ms=6.0, record=["soma", "node1", "node2", "node3"])

    dists = section_distances_micron(spec, n.index_map)  # 各区段中心到胞体距离(µm)
    nodes = ["node1", "node2", "node3"]
    times = {}
    for nd in nodes:
        st = r.spike_times_ms[nd]
        times[nd] = float(st[0]) if len(st) else None
    t_soma = float(r.spike_times_ms["soma"][0]) if len(r.spike_times_ms["soma"]) else None

    # 逐段速度（node k-1 → node k）
    rows = []
    cv_list = []
    prev_name, prev_t, prev_d = "soma", t_soma, 0.0
    for nd in nodes:
        t_now, d_now = times[nd], dists[nd]
        if t_now is not None and prev_t is not None:
            dt_ms = t_now - prev_t
            dd_um = d_now - prev_d
            cv = (dd_um * 1e-6) / (dt_ms * 1e-3) if dt_ms > 0 else None
        else:
            cv = None
        rows.append(dict(segment=nd, distance_um=d_now, t_first_ms=t_now,
                         seg_cv_mps=cv, prev_segment=prev_name))
        if cv is not None:
            cv_list.append(cv)
        prev_name, prev_t, prev_d = nd, t_now, d_now

    mean_cv = float(np.mean(cv_list)) if cv_list else None
    pass_ = bool(mean_cv is not None and CV_RANGE_MPS[0] <= mean_cv <= CV_RANGE_MPS[1])

    with open(REPORT_CSV, "w") as f:
        f.write("segment,distance_um,t_first_ms,prev_segment,seg_cv_mps\n")
        for row in rows:
            f.write(f"{row['segment']},{row['distance_um']:.3f},"
                    f"{row['t_first_ms'] if row['t_first_ms'] is not None else ''},"
                    f"{row['prev_segment']},"
                    f"{row['seg_cv_mps'] if row['seg_cv_mps'] is not None else ''}\n")

    return dict(
        pass_=pass_,
        mean_cv_mps=mean_cv,
        cv_list_mps=cv_list,
        t_soma_ms=t_soma,
        per_node=rows,
        cv_range_mps=CV_RANGE_MPS,
        report_csv=REPORT_CSV,
    )


if __name__ == "__main__":
    import json
    res = run_p5()
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print("P5 PASS" if res["pass_"] else "P5 FAIL")
