"""M3 实验台数据生成器：真实形态学 + 真实仿真轨迹 → demo/m3_lab/runs.js。

把"真实神经元连接方式"与"真实行为数据"导出为浏览器可用的 JS 数据：
  1. 形态学（M1 定稿 data/m1_channel_map.csv）：18 隔室 ——
     soma → 树突链(dend1×2 → dend2×2，触刺激位点 dend2#1)
           + 轴突链(ais → myelin1×3 → node1 → myelin2×3 → node2
                     → myelin3×3 → node3，郎飞结 300 mS/cm²，node3 = 突触前位点)；
  2. 突触连接（M3 定稿 data/m3_reflex_params.csv）：PLM/AVM/DA/VB 四神经元
     node3 → soma 化学突触（AMPA/GABA）+ DA/VB → 肌肉 δ 驱动；
  3. 8 组确定性仿真（Brian2 2.6.0，p=1，同 M3 验收基线）：
     无刺激对照 / 0.5× / 1× / 2× / 4× / 8× / 8×I0 长按20ms / 消融(切 AVM→VB)。
     每组导出全部 18 隔室 × 4 神经元的 V(t)（dt=0.1ms，1 位小数）+ 发放时刻
     + C_back/C_fwd + 统计（潜伏期/D_peak/判定），供页面逐帧回放。

用法：
  .venv-neuro/bin/python -m neural_exploration.tools.gen_m3_lab
输出：neural_exploration/demo/m3_lab/runs.js（window.M3_DATA = {...}）
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.src.reflex_arc import ReflexArc  # noqa: E402
from neural_exploration.tools.load_morphology import load_morphology  # noqa: E402

OUT_DIR = os.path.join(ROOT, "neural_exploration", "demo", "m3_lab")
OUT_JS = os.path.join(OUT_DIR, "runs.js")

DT_EXPORT_MS = 0.1          # 导出采样步长（原 dt=0.01ms，降采样 10×）
ROUND_V = 1                 # 膜电位保留小数位（0.1mV）
ROUND_C = 3                 # 肌肉收缩保留小数位
TOUCH_WIN = (50.0, 90.0)    # 响应窗（P5）

# 8 组实验（id, intensity, dur_ms, ablation, label）
EXPERIMENTS = [
    ("ctrl", 0.0, 5.0, False, "无刺激（对照）"),
    ("i05", 0.5, 5.0, False, "轻触 0.5×I0"),
    ("i1", 1.0, 5.0, False, "基准 1×I0"),
    ("i2", 2.0, 5.0, False, "2×I0"),
    ("i4", 4.0, 5.0, False, "重触 4×I0"),
    ("i8", 8.0, 5.0, False, "重压 8×I0"),
    ("i8_long", 8.0, 20.0, False, "强刺激长按 8×I0 / 20ms"),
    ("abl", 1.0, 5.0, True, "消融：切除 AVM→VB（GABA 抑制）"),
]


def _chain_positions(seg_names, direction: int) -> list:
    """沿链（自 soma 向外）给每区段逐隔室累计中心坐标（µm）。"""
    spec = load_morphology()
    out = []
    cursor = 0.0
    for name in seg_names:
        seg = spec.by_name(name)
        step = seg.length_um / seg.n
        for i in range(seg.n):
            center = cursor + step / 2.0 + i * step
            out.append((name, i, direction * center,
                        seg.diameter_um, _comp_type(name)))
        cursor += seg.length_um
    return out


def _comp_type(seg: str) -> str:
    if seg == "soma":
        return "soma"
    if seg.startswith("dend"):
        return "dend"
    if seg == "ais":
        return "ais"
    if seg.startswith("node"):
        return "node"
    return "myelin"   # myelin1/2/3


def build_morph_data() -> dict:
    spec = load_morphology()
    dend = spec.dendrite_chain()          # [dend1, dend2]
    axon = spec.axon_chain()              # [ais, myelin1, node1, ...]
    dend_pos = _chain_positions([s.name for s in dend], direction=-1)
    axon_pos = _chain_positions([s.name for s in axon], direction=+1)
    comps_by_role = {}
    for role in ("PLM", "AVM", "DA", "VB"):
        comps = []
        for s, i, x, d, t in dend_pos + [("soma", 0, 0.0, spec.by_name("soma").diameter_um, "soma")] + axon_pos:
            comps.append({"s": s, "i": i, "x": round(x, 1), "d": d, "t": t})
        comps_by_role[role] = comps
    return {
        "roles": ["PLM", "AVM", "DA", "VB"],
        "comps": comps_by_role,
        # 突触连接（node3 → soma 化学突触 + 肌肉 δ 驱动；M3 CSV 定稿）
        "links": [
            {"from": "PLM", "to": "AVM", "type": "ampa", "g": 5.0, "delay": 0.5},
            {"from": "AVM", "to": "DA", "type": "ampa", "g": 5.0, "delay": 0.5},
            {"from": "AVM", "to": "VB", "type": "gaba", "g": 15.0, "delay": 0.5},
            {"from": "DA", "to": "MUSCLE_B", "type": "muscle", "w": 0.60, "delay": 0.1},
            {"from": "VB", "to": "MUSCLE_F", "type": "muscle", "w": 0.18, "delay": 0.1},
        ],
        "touch": {"role": "PLM", "site": "dend2#1", "i0": 60.0, "start": 50.0, "dur": 5.0},
        "tonic": {"role": "VB", "density": 14.0},
    }


def _record_labels(roles=("PLM", "AVM", "DA", "VB")) -> list:
    spec = load_morphology()
    labels = []
    for role in roles:
        for seg in spec.segments:
            for i in range(seg.n):
                labels.append(f"{role}_{seg.name}#{i}")
    return labels


def run_one(cfg) -> dict:
    rid, intensity, dur_ms, ablation, _label = cfg
    arc = ReflexArc()
    if ablation:
        arc.remove_synapse("AVM", "VB")
    if dur_ms != 5.0:
        arc.set_touch(dur_ms=dur_ms)
    r = arc.run(intensity=intensity, record=_record_labels())

    step = int(round(DT_EXPORT_MS / r.meta["dt_ms"]))
    v = {}
    for lab, arr in r.v_mv.items():
        role = lab.split("_", 1)[0].upper()
        key = lab.split("_", 1)[1]           # 'dend2#1' 等
        v.setdefault(role, {})[key] = [round(float(x), ROUND_V) for x in arr[::step]]

    spikes = {role: [round(float(t), 2) for t in r.spikes(role, "node3")]
              for role in ("PLM", "AVM", "DA", "VB")}
    touch_start = r.meta["touch_start_ms"]
    da = spikes["DA"]
    latency = (da[0] - touch_start) if da else None
    d = r.c_back - r.c_fwd
    d_peak = float(d.max())
    vb_win = [t for t in spikes["VB"] if TOUCH_WIN[0] <= t <= TOUCH_WIN[1]]
    if not da:
        verdict = "静默（无反应）"
    elif d_peak > 0.3:
        verdict = "后退"
    else:
        verdict = "前进 / 未定"

    return {
        "id": rid, "label": _label, "intensity": intensity,
        "durMs": dur_ms, "ablation": ablation,
        "t": [round(float(x), 2) for x in r.t_ms[::step]],
        "v": v,
        "spikes": spikes,
        "cBack": [round(float(x), ROUND_C) for x in r.c_back[::step]],
        "cFwd": [round(float(x), ROUND_C) for x in r.c_fwd[::step]],
        "stats": {
            "latencyMs": round(latency, 2) if latency is not None else None,
            "dPeak": round(d_peak, 3),
            "cBackPeak": round(float(r.c_back.max()), 3),
            "cFwdPeak": round(float(r.c_fwd.max()), 3),
            "vbWinCount": len(vb_win),
            "verdict": verdict,
        },
    }


def main() -> int:
    t0 = time.time()
    morph = build_morph_data()
    runs = {}
    for cfg in EXPERIMENTS:
        t1 = time.time()
        run = run_one(cfg)
        runs[run["id"]] = run
        s = run["stats"]
        print(f"[gen] {run['id']:7s} {run['label']:24s} 潜伏期={s['latencyMs']}ms "
              f"D_peak={s['dPeak']:.3f} C_back峰={s['cBackPeak']:.3f} "
              f"判定={s['verdict']}  ({time.time()-t1:.1f}s)")

    data = {
        "meta": {
            "dtMs": DT_EXPORT_MS, "tTotalMs": runs["i1"]["t"][-1],
            "dThreshold": 0.3, "engine": "Brian2 2.6.0 (rk4, dt=0.01ms)",
            "deterministic": True, "morphology": "M1 Mainen-Sejnowski 简化 18 隔室",
            "generatedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "morph": morph,
        "runs": runs,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_JS, "w", encoding="utf-8") as f:
        f.write("/* M3 实验台数据：真实仿真轨迹（确定性 p=1，与 m3_report 一致）。"
                "由 tools/gen_m3_lab.py 生成，勿手改。 */\n")
        f.write("window.M3_DATA = ")
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        f.write(";\n")
    size_mb = os.path.getsize(OUT_JS) / 1e6
    print(f"\n[gen] 完成 {len(runs)} 组实验，{time.time()-t0:.0f}s → {OUT_JS} "
          f"({size_mb:.1f}MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
