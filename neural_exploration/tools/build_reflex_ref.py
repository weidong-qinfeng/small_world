"""M3 参考解：NEURON 9.0.1 cvode 高精度全链（触觉反射弧 4 神经元 + 双肌肉）。

清单《生物仿真M3实施清单》§2/§3（步骤 2）：4 个 M1 形态学多隔室神经元
（PLM 感觉 → AVM 中间 → DA 后退运动 / VB 前进运动）+ ExpSyn（AMPA/GABA）
+ IClamp 触刺激（6 档强度，方案 A：注入 PLM 树突端）+ VB 张力注入；
行为潜伏期由 NEURON 发放序列经同一肌肉 ODE（引擎无关）计算。
输出 `data/m3_reflex_ref.npz`：

  - t_ms                        均匀网格（dt=0.01ms，T_TOTAL 由 CSV 定稿）
  - v_avm_soma_mv_{i}           各档强度感觉→中间 PSP（AVM soma，P2 参考）
  - spike_times_{L}_{i}         L∈{plm,avm,da,vb} 各级 node3 发放时刻（6 档）
  - c_back_{i} / c_fwd_{i} / d_{i}   双肌肉收缩轨迹与方向 D（6 档）
  - latency_nerve               神经潜伏期（触刺激开始 → DA 首发放，6 档）
  - latency_behavior            行为潜伏期（首个 C_back ≥ 0.3·C_back_peak − 触刺激开始，6 档）
  - chain_time_ms               PLM 首发放 → DA 首发放（P2 链传导时间）
  - intensities / meta

参数唯一定稿源：`data/m3_reflex_params.csv`（tau/E 电学基础沿用 m2_synapse_params.csv）；
实测依据与踩坑记录：`docs/m3_env_notes.md` §L5–L10。

用法：
  .venv-neuro/bin/python -m neural_exploration.tools.build_reflex_ref
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.src.synapse_model import load_synapse_params  # noqa: E402
from neural_exploration.tools.load_morphology import (  # noqa: E402
    V0, load_morphology,
)
from neural_exploration.tools.build_neuron_ref import (  # noqa: E402
    SOMA_AREA_CM2, build_neuron,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
REF_NPZ = os.path.join(DATA_DIR, "m3_reflex_ref.npz")
PARAMS_CSV = os.path.join(DATA_DIR, "m3_reflex_params.csv")

DT_OUT = 0.01          # ms，输出重采样步长（与 CSV dt_ms 一致）
SPIKE_THRESH_MV = -15.0   # node3 发放检测阈值（上冲）+ 峰定位
SPIKE_REF_MS = 1.5        # 发放检测去重窗口


# --------------------------------------------------------------------- #
# CSV 读取（m3_reflex_params.csv，单表：role/neuron_class/... 见文件头注释）
# --------------------------------------------------------------------- #
def load_m3_params(csv_path: str = PARAMS_CSV) -> dict:
    """读入 m3_reflex_params.csv → 神经元/突触/肌肉/参数四类。"""
    neurons, synapses, muscles = {}, [], {}
    params = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(row for row in f if not row.strip().startswith("#"))
        for r in reader:
            role = (r.get("role") or "").strip()
            stype = (r.get("synapse_type") or "").strip()
            if role == "param":
                name = (r["neuron_class"] or "").strip()
                val = (r.get("value") or "").strip()
                if not val:  # 兼容旧草案把值写在 note 列
                    val = (r.get("note") or "").strip()
                params[name] = val
            elif stype == "muscle":
                muscles[(r["synapse_to"] or "").strip()] = dict(
                    w=float(r["g_max_ns"]), delay_ms=float(r["delay_ms"]))
            elif stype in ("ampa", "gaba", "nmda"):
                synapses.append(dict(
                    pre=(r["synapse_from"] or "").strip().lower(),
                    post=(r["synapse_to"] or "").strip().lower(),
                    stype=stype,
                    g_max_ns=float(r["g_max_ns"]),
                    delay_ms=float(r["delay_ms"]),
                ))
            elif role in ("PLM", "AVM", "DA", "VB"):
                tonic = (r.get("tonic_uA_cm2") or "").strip()
                neurons[role.lower()] = dict(
                    tonic_uA_cm2=float(tonic) if tonic else 0.0)
    return dict(neurons=neurons, synapses=synapses, muscles=muscles, params=params)


def _param_f(params: dict, name: str) -> float:
    return float(params[name])


def _intensities(params: dict) -> np.ndarray:
    return np.asarray([float(x) for x in params["intensity_levels"].split(",")])


# --------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------- #
def dend_tip_area_cm2(spec) -> float:
    """PLM 树突端注入位点膜面积：dend2 末隔室侧面积 π·d·(L/n)。"""
    seg = spec.by_name("dend2")
    d_cm = seg.diameter_um * 1e-4
    l_cm = seg.length_um / seg.n * 1e-4
    return float(np.pi * d_cm * l_cm)


def detect_spikes(t_ms: np.ndarray, v_mv: np.ndarray,
                  thresh: float = SPIKE_THRESH_MV,
                  refractory_ms: float = SPIKE_REF_MS) -> np.ndarray:
    """V 上冲过 thresh 的峰时刻（ms）：边沿检测 + 窗口内峰定位 + 去重。"""
    v = np.asarray(v_mv, dtype=float)
    t = np.asarray(t_ms, dtype=float)
    above = v > thresh
    edges = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    times = []
    for e in edges:
        win = (t >= t[e]) & (t <= t[e] + refractory_ms)
        times.append(float(t[win][np.argmax(v[win])]) if win.sum() else float(t[e]))
    out = []
    for x in times:
        if not out or x - out[-1] > refractory_ms:
            out.append(x)
    return np.asarray(out)


def muscle_curve(t_ms: np.ndarray, spike_times_ms: np.ndarray,
                 w: float, tau_ms: float) -> np.ndarray:
    """肌肉解析解：C(t) = Σ_k w·H(t−t_k)·exp(−(t−t_k)/τ)（δ 注入 + 指数衰减）。"""
    c = np.zeros_like(np.asarray(t_ms, dtype=float))
    for sp in np.atleast_1d(spike_times_ms):
        c += np.where(t_ms >= sp, w * np.exp(-(t_ms - sp) / tau_ms), 0.0)
    return c


# --------------------------------------------------------------------- #
# 单档强度运行
# --------------------------------------------------------------------- #
def _run_intensity(spec, synapses, muscles, params,
                   intensity: float, t_total: float, touch_start: float,
                   touch_dur: float, i0: float, tonic: float) -> dict:
    """构建全链（每档 clear=True 重建）+ 运行 → 各级 spike_times / PSP / 肌肉轨迹。"""
    from neuron import h

    # 4 神经元（M1 形态学；unique name_prefix；clear=False 同进程多神经元——M2 已验证）
    secs = {}
    first = True
    for name in ("plm", "avm", "da", "vb"):
        secs[name] = build_neuron(spec, clear=first, name_prefix=f"{name}_")
        first = False

    h.load_file("stdrun.hoc")
    h.celsius = 6.3                     # 硬约束：Q10 参考温度（SESSION_CONTEXT §四 #2）
    h.v_init = V0

    # 点过程引用列表持有防 GC（M2 L8）
    clamps, syns, ncs = [], [], []

    # 触刺激：方案 A，注入 PLM 树突端（dend2 末隔室中心）；密度→nA 按位点面积换算
    a_tip = dend_tip_area_cm2(spec)
    if intensity > 0:
        amp = intensity * i0 * 1e-6 * a_tip * 1e9
        tip_x = (secs["plm"]["dend2"].nseg - 0.5) / secs["plm"]["dend2"].nseg
        cl = h.IClamp(secs["plm"]["dend2"](tip_x))
        cl.delay = touch_start
        cl.dur = touch_dur
        cl.amp = amp
        clamps.append(cl)

    # VB 张力注入（soma 恒定电流，维持 C_fwd 静息基线≈0.2）
    if tonic > 0:
        tcl = h.IClamp(secs["vb"]["soma"](0.5))
        tcl.delay = 0.0
        tcl.dur = t_total
        tcl.amp = tonic * 1e-6 * SOMA_AREA_CM2 * 1e9
        clamps.append(tcl)

    # 突触链：ExpSyn（AMPA/GABA，τ/E 沿 m2 行）+ NetCon(pre node3 → post soma)
    m2 = load_synapse_params()
    for s in synapses:
        base = m2[s["stype"]]
        syn = h.ExpSyn(secs[s["post"]]["soma"](0.5))
        syn.tau = base.tau_ms
        syn.e = base.e_rev_mv
        nc = h.NetCon(secs[s["pre"]]["node3"](0.5)._ref_v, syn, sec=secs[s["pre"]]["node3"])
        nc.threshold = -20.0
        nc.delay = s["delay_ms"]
        nc.weight[0] = s["g_max_ns"] * 1e-3      # ExpSyn weight 单位 µS（µS = nS×1e-3）
        syns.append(syn)
        ncs.append(nc)

    # 记录：各级 node3 + AVM soma（PSP）
    tvec = h.Vector(); tvec.record(h._ref_t)
    vrec = {}
    for name in ("plm", "avm", "da", "vb"):
        vv = h.Vector(); vv.record(secs[name]["node3"](0.5)._ref_v)
        vrec[name] = vv
    vpsp = h.Vector(); vpsp.record(secs["avm"]["soma"](0.5)._ref_v)

    cvode = h.CVode()
    cvode.active(1)
    cvode.atol(1e-8)                     # 硬约束：参考真理容差
    cvode.rtol(1e-8)
    h.tstop = t_total
    h.run()

    # 重采样到均匀网格
    t_irr = np.array(tvec)
    order = np.argsort(t_irr)
    t_u = np.arange(0.0, t_total + DT_OUT / 2, DT_OUT)
    vgrid, spikes = {}, {}
    for name, vv in vrec.items():
        v = np.interp(t_u, t_irr[order], np.array(vv)[order])
        vgrid[name] = v
        spikes[name] = detect_spikes(t_u, v)
    psp = np.interp(t_u, t_irr[order], np.array(vpsp)[order])

    # 肌肉（引擎无关 ODE；delta 注入 + 指数衰减解析解）
    tau_mus = _param_f(params, "muscle_tau_ms")
    w_back = muscles["muscle_back"]["w"]
    w_fwd = muscles["muscle_fwd"]["w"]
    c_back = muscle_curve(t_u, spikes["da"], w_back, tau_mus)
    c_fwd = muscle_curve(t_u, spikes["vb"], w_fwd, tau_mus)
    d = c_back - c_fwd

    # 潜伏期（清单 §5.3 定义）
    def _latency(cond) -> float:
        idx = np.flatnonzero(cond)
        return float(t_u[idx[0]] - touch_start) if len(idx) else float("nan")

    t_da_first = float(spikes["da"][0]) if len(spikes["da"]) else float("nan")
    t_plm_first = float(spikes["plm"][0]) if len(spikes["plm"]) else float("nan")
    latency_nerve = t_da_first - touch_start if not np.isnan(t_da_first) else float("nan")
    if c_back.max() > 0:
        latency_behavior = _latency(c_back >= 0.3 * c_back.max())
    else:
        latency_behavior = float("nan")
    chain_time = (t_da_first - t_plm_first
                  if not (np.isnan(t_da_first) or np.isnan(t_plm_first)) else float("nan"))

    return dict(
        t_ms=t_u, vgrid=vgrid, psp=psp, spikes=spikes,
        c_back=c_back, c_fwd=c_fwd, d=d,
        latency_nerve=latency_nerve, latency_behavior=latency_behavior,
        chain_time_ms=chain_time,
        d_peak=float(d.max()), c_back_peak=float(c_back.max()),
        c_fwd_baseline=float(np.median(c_fwd[(t_u > 40.0) & (t_u < 50.0)])),
        amp_nA=(intensity * i0 * 1e-6 * a_tip * 1e9) if intensity > 0 else 0.0,
    )


# --------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------- #
def run_reference(out_npz: str = REF_NPZ) -> str:
    """6 档强度全链参考解 → 落盘 npz。"""
    os.makedirs(DATA_DIR, exist_ok=True)
    cfg = load_m3_params()
    params = cfg["params"]
    spec = load_morphology()

    touch_start = _param_f(params, "touch_start_ms")
    touch_dur = _param_f(params, "touch_dur_ms")
    i0 = _param_f(params, "i0_uA_cm2")
    t_total = _param_f(params, "t_total_ms")
    tonic = cfg["neurons"]["vb"]["tonic_uA_cm2"]
    intensities = _intensities(params)

    out = {}
    runs = []
    for k, s in enumerate(intensities):
        r = _run_intensity(spec, cfg["synapses"], cfg["muscles"],
                           params, s, t_total, touch_start, touch_dur, i0, tonic)
        runs.append(r)
        i = int(k)
        out[f"v_avm_soma_mv_{i}"] = r["psp"]
        for name in ("plm", "avm", "da", "vb"):
            out[f"spike_times_{name}_{i}"] = r["spikes"][name]
        out[f"c_back_{i}"] = r["c_back"]
        out[f"c_fwd_{i}"] = r["c_fwd"]
        out[f"d_{i}"] = r["d"]
        if s == 1.0:   # I0 档额外存各级 node3 轨迹（波形对照用）
            for name in ("plm", "avm", "da", "vb"):
                out[f"v_{name}_node3_mv_ref"] = r["vgrid"][name]

    out["t_ms"] = runs[0]["t_ms"]
    out["intensities"] = intensities
    out["latency_nerve"] = np.asarray([r["latency_nerve"] for r in runs])
    out["latency_behavior"] = np.asarray([r["latency_behavior"] for r in runs])
    out["chain_time_ms"] = np.asarray([r["chain_time_ms"] for r in runs])
    out["d_peak"] = np.asarray([r["d_peak"] for r in runs])
    out["c_back_peak"] = np.asarray([r["c_back_peak"] for r in runs])
    out["c_fwd_baseline"] = np.asarray([r["c_fwd_baseline"] for r in runs])
    out["stim_amp_nA"] = np.asarray([r["amp_nA"] for r in runs])

    meta = dict(
        engine="NEURON 9.0.1 cvode（atol=rtol=1e-8, celsius=6.3, v_init=-65mV）",
        params_csv="data/m3_reflex_params.csv",
        dt_out_ms=DT_OUT,
        t_total_ms=t_total,
        touch_start_ms=touch_start, touch_dur_ms=touch_dur,
        i0_uA_cm2=i0, intensities=list(map(float, intensities)),
        inject_site="plm_dend2_tip（方案 A）",
        dend_tip_area_cm2=float(dend_tip_area_cm2(spec)),
        soma_area_cm2=float(SOMA_AREA_CM2),
        tonic_vb_uA_cm2=tonic,
        synapse_chain=[(s["pre"], s["post"], s["stype"], s["g_max_ns"], s["delay_ms"])
                       for s in cfg["synapses"]],
        muscle_tau_ms=_param_f(params, "muscle_tau_ms"),
        muscle_w_back=cfg["muscles"]["muscle_back"]["w"],
        muscle_w_fwd=cfg["muscles"]["muscle_fwd"]["w"],
        behavior_latency_definition="首个 C_back≥0.3·C_back_peak − touch_start（清单 §5.3）",
        note=("行为潜伏期≈神经潜伏期（8–14ms），结构性落不到 [25,60] 窗——见 docs/m3_env_notes.md §L7；"
              "P5 消融判据时序敏感性见 §L8"),
    )
    np.savez(out_npz, **out, meta=np.array(meta, dtype=object))
    return out_npz


if __name__ == "__main__":
    out = run_reference()
    d = np.load(out, allow_pickle=True)
    print(f"参考解已写入: {out}")
    keys = [k for k in d.files if k != "meta"]
    print(f"键数: {len(keys)}")
    print("intensities:", d["intensities"])
    print("latency_nerve(ms):", np.round(d["latency_nerve"], 2))
    print("latency_behavior(ms):", np.round(d["latency_behavior"], 2))
    print("chain_time(ms):", np.round(d["chain_time_ms"], 2))
    print("d_peak:", np.round(d["d_peak"], 3))
    print("c_back_peak:", np.round(d["c_back_peak"], 3))
    print("c_fwd_baseline:", np.round(d["c_fwd_baseline"], 3))
    for i, s in enumerate(d["intensities"]):
        sp = {L: d[f"spike_times_{L}_{i}"] for L in ("plm", "avm", "da", "vb")}
        order_ok = (len(sp["plm"]) > 0 and len(sp["avm"]) > 0 and len(sp["da"]) > 0
                    and sp["plm"][0] < sp["avm"][0] < sp["da"][0])
        print(f"  档 {s:4.1f}: PLM={sp['plm'][0] if len(sp['plm']) else '-':>6} "
              f"AVM={sp['avm'][0] if len(sp['avm']) else '-':>6} "
              f"DA={sp['da'][0] if len(sp['da']) else '-':>6} "
              f"VB(n)={len(sp['vb'])}  t_PLM<t_AVM<t_DA={order_ok}")
    print("meta:", d["meta"].item())
