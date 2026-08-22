"""M1 多隔室参考解：NEURON 9.0.1 cvode 高精度（清单 §3 步骤 2）。

同一 `MorphologySpec`（data/m1_channel_map.csv）在 NEURON 中逐隔室复刻：
  - 区段几何（L/diam/nseg）与 Brian2 完全一致；
  - 主动段插入 `hh`（gnabar/gkbar 按 CSV；ena/ek/el 显式设 50/-77/-54.4 mV）；
  - 髓鞘段插入 `pas`（g_pas 按 CSV 的 gl；e_pas=-54.4）；
  - 胞体 Ra 取极小值近似 Brian2 球体 Soma 的零内部电阻；
  - **celsius=6.3**（NEURON hh 速率函数的参考温度，与经典 HH 1952 一致，
    否则 Q10 缩放会让动力学加快 ~23 倍、与 Brian2 不可比）；
  - cvode 变步长 + atol/rtol=1e-8 作为“参考真理”；
  - 输出：胞体/树突端/轴突端/各郎飞结/髓鞘中点 V(t)（重采样到 dt=0.01ms 均匀网格）
    → data/m1_multicomp_ref.npz

用法：
  .venv-neuro/bin/python -m neural_exploration.tools.build_neuron_ref
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.tools.load_morphology import (  # noqa: E402
    CM, RA, V0, MorphologySpec, load_morphology,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
REF_NPZ = os.path.join(DATA_DIR, "m1_multicomp_ref.npz")

# 与 Brian2 侧一致的刺激参数（清单 §2.2 / §5.1）
DT_OUT = 0.01          # ms，输出重采样步长
T_TOTAL = 80.0         # ms
STIM_START = 5.0       # ms
STIM_END = 30.0        # ms
STIM_UA_CM2 = 10.0     # µA/cm²（胞体密度 → nA 换算）

# 胞体球面积：π·d²（µm² → cm²）；d=20µm
SOMA_DIAM_UM = 20.0
SOMA_AREA_CM2 = np.pi * (SOMA_DIAM_UM * 1e-4) ** 2


def build_neuron(spec: MorphologySpec):
    """按规格在 NEURON 中构建多隔室神经元，返回 sections dict。"""
    from neuron import h

    h("forall delete_section()")
    sections = {}

    soma_seg = spec.by_name("soma")
    soma = h.Section(name="soma")
    soma.L = soma_seg.length_um
    soma.diam = soma_seg.diameter_um
    soma.nseg = 1
    soma.Ra = 0.001            # 近似球体（Brian2 Soma 内部电阻≈0）
    sections["soma"] = soma

    for seg in spec.segments:
        if seg.is_soma:
            continue
        sec = h.Section(name=seg.name)
        sec.L = seg.length_um
        sec.diam = seg.diameter_um
        sec.nseg = seg.n
        sec.Ra = RA if seg.name != "soma" else 0.001
        parent = sections[seg.parent]
        sec.connect(parent, 1, 0)
        sections[seg.name] = sec

    # 通道与电容
    for seg in spec.segments:
        sec = sections[seg.name]
        if seg.gna_mS_cm2 > 0:
            sec.insert("hh")
            # ena/ek 是离子变量（na/k 机制，段级）；el 是 hh 机制参数
            sec.ena = spec.ena
            sec.ek = spec.ek
            for s in sec:  # 逐隔室（nseg=1 时只有一个）
                s.hh.gnabar = seg.gna_mS_cm2 / 1000.0   # mS/cm² → S/cm²
                s.hh.gkbar = seg.gk_mS_cm2 / 1000.0
                s.hh.el = spec.el
        else:
            sec.insert("pas")
            for s in sec:
                s.pas.g = seg.gl_mS_cm2 / 1000.0
                s.pas.e = spec.el
        sec.cm = seg.cm_uF_cm2
    return sections


def run_reference(
    spec: MorphologySpec,
    out_npz: str = REF_NPZ,
    t_total: float = T_TOTAL,
    stim_uA_cm2: float = STIM_UA_CM2,
    stim_start: float = STIM_START,
    stim_end: float = STIM_END,
) -> str:
    """运行 NEURON cvode 高精度参考解并落盘 npz。"""
    from neuron import h

    h.load_file("stdrun.hoc")  # 提供 v_init / tstop / run() 等标准定义

    sections = build_neuron(spec)
    soma = sections["soma"]

    # 温度：经典 HH 参考温度（不设则默认 6.3，仍显式声明保证可复现）
    h.celsius = 6.3
    h("v_init = {}".format(V0))

    # 刺激：胞体 IClamp（nA，与 Brian2 point current 同单位）
    stim_amp_nA = stim_uA_cm2 * 1e-6 * SOMA_AREA_CM2 * 1e9
    stim = h.IClamp(soma(0.5))
    stim.delay = stim_start
    stim.dur = stim_end - stim_start
    stim.amp = stim_amp_nA

    # 记录位置（隔室中心，与 Brian2 一致）：soma(0.5)、dend2 末隔室中心、
    # node1/2/3(0.5)、myelin1 中隔室中心（第2/3）、myelin3 中隔室中心
    def seg_center(sec, k):  # 第 k 个隔室中心（0-based）
        return (k + 0.5) / sec.nseg

    rec_specs = {
        "soma": (sections["soma"], seg_center(sections["soma"], 0)),
        "dend_end": (sections["dend2"], seg_center(sections["dend2"], sections["dend2"].nseg - 1)),
        "node1": (sections["node1"], 0.5),
        "node2": (sections["node2"], 0.5),
        "node3": (sections["node3"], 0.5),
        "myelin1_mid": (sections["myelin1"], seg_center(sections["myelin1"], sections["myelin1"].nseg // 2)),
        "myelin3_mid": (sections["myelin3"], seg_center(sections["myelin3"], sections["myelin3"].nseg // 2)),
    }

    tvec = h.Vector()
    tvec.record(h._ref_t)
    vvecs = {}
    for name, (sec, x) in rec_specs.items():
        vv = h.Vector()
        vv.record(sec(x)._ref_v)
        vvecs[name] = vv

    cvode = h.CVode()
    cvode.active(1)
    cvode.atol(1e-8)
    cvode.rtol(1e-8)
    h.tstop = t_total
    h.run()

    # 重采样到均匀网格
    t_irr = np.array(tvec)
    t_unif = np.arange(0.0, t_total + DT_OUT / 2, DT_OUT)
    out = {"t_ms": t_unif}
    for name, vv in vvecs.items():
        v_irr = np.array(vv)
        # cvode 可能稀疏记录；用 t 对齐插值（先按 t 排序）
        order = np.argsort(t_irr)
        out[f"v_{name}_mv"] = np.interp(t_unif, t_irr[order], v_irr[order])

    meta = dict(
        engine="NEURON 9.0.1",
        method="cvode atol=1e-8 rtol=1e-8",
        celsius=6.3,
        dt_out_ms=DT_OUT,
        t_total_ms=t_total,
        stim_start_ms=stim_start,
        stim_end_ms=stim_end,
        stim_uA_cm2=stim_uA_cm2,
        stim_nA=float(stim_amp_nA),
        soma_area_cm2=float(SOMA_AREA_CM2),
        v_init_mv=V0,
        ra_ohm_cm=RA,
        cm_uF_cm2=CM,
        n_compartments=spec.total_compartments,
    )
    np.savez(out_npz, **out, meta=np.array(meta, dtype=object))
    return out_npz


if __name__ == "__main__":
    spec = load_morphology()
    out = run_reference(spec)
    d = np.load(out, allow_pickle=True)
    print(f"参考解已写入: {out}")
    print(f"t 点数: {len(d['t_ms'])}，键: {[k for k in d.files if k != 'meta']}")
    for k in d.files:
        if k.startswith("v_"):
            print(f"  {k:16s} V=[{d[k].min():7.2f},{d[k].max():7.2f}] mV")
    print("meta:", d["meta"].item())
