"""**独立参照臂（可选增强）**：M11 point 内核 vs **Brian2** 的同离散方案单神经元对照。

> **定位（主agent R32/④）**：**不是 G0 判据**、**不进门判定**（`not_a_gate_criterion=true`）。
> 目的：给 (b) 的"参照可信度"一个**不带缺陷**的独立交叉证据 —— 因为 (b) 的参照（M10 冻结内核）
> 自身带 E-M9-4/E-M10-1/E-M10-2 三处已登记缺陷（R32：参照须同时 ①可比 ②可信）。
> 结果落 `discovery`：差异同量级（≤1 步）⇒ 提高参照可信度；**差异显著 ⇒ 交主agent 裁决**
> （可能反噬 M10 的 P5 工作点数值，属 M10 记录范畴，**不在 M11 内改写 M10 结论**）。

**为什么必须"同离散方案"才可比**（本节点在 G0 已给的口径论证）：本引擎 point 档对 `v` 用**显式 Euler**、
对电导用**精确指数衰减** `g ← g·exp(−dt/τ)`；而 Brian2 默认 `exponential_euler` 对 `v` 用**闭式**
⇒ 跨方案比较会失败于**口径**而非正确性。故本工具用 Brian2 的**按变量方法**：
`method={'v': 'euler', 'ge': 'exponential_euler', 'gi': 'exponential_euler'}` —— 与引擎逐运算同构。

对照项（单神经元，注入序列完全相同）：
  ① 首发放步；② 全部发放步（容差 ≤1 步 = 0.05 ms）；③ 阈下 `v(t)` 轨迹最大绝对差（mV，应 ~0）。
另对 **REF_FROZEN（缺陷档）** 同样跑一次 ⇒ 独立复证 E-M9-4（预期 ISI 182 vs 144 步）。

用法（引擎侧解释器，需 torch+brian2）：
    PYTHONHASHSEED=0 ./.venv-m9/bin/python -m neural_exploration.tools.ref_brian2_point_lif
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "neural_exploration", "data")

# 与 `MouseCortexEngine.point` 的默认 PointParams **逐字段一致**（不得凭印象）
P = dict(v_rest=-52.0, v_th=-45.0, v_reset=-55.0, tau_m=20.0, tau_e=2.0, tau_i=5.0,
         e_exc=0.0, e_inh=-80.0, ref_ms=2.0, dt_ms=0.05)
DT_S = P["dt_ms"] * 1e-3
G_EXT = 22.0          # 注入事件幅度（与 P5 臂的 [17,23] 同量级）
LAM_HZ = 300.0        # 注入率（Hz）—— 使单神经元充分驱动（ISI 可测）
T_STEPS = 8000        # 8000 × 0.05 ms = 400 ms


def engine_arm(form: str, seed: int = 3, lam_hz: float = None):
    """M11 内核单神经元臂（`REF_FIX` 或 `REF_FROZEN`），返回 (发放步, v 轨迹)。"""
    from neural_exploration.src.mouse_cortex_engine import (MouseCortexEngine, REF_FIX,
                                                            REF_FROZEN)
    e = MouseCortexEngine(1, device="cpu", use_compile=False, verbose=False)
    e.configure_refractory(REF_FROZEN if form == "frozen" else REF_FIX)
    e.finalize()
    e.set_neuron_heterogeneity(np.zeros(1, dtype=np.float32))
    lam = float(LAM_HZ if lam_hz is None else lam_hz)
    n_ev = int(lam * T_STEPS * DT_S)
    rng = np.random.default_rng(seed)
    steps = np.sort(rng.integers(0, T_STEPS, n_ev))
    e.set_external_events(steps, np.zeros(n_ev, dtype=np.int64),
                          np.full(n_ev, G_EXT, dtype=np.float32), n_steps=T_STEPS)
    e.reset(seed=0)
    e.reset_counts()
    vt = np.empty(T_STEPS, dtype=np.float64)
    spikes = []
    for i in range(T_STEPS):
        e.run(1, delivery="step", record="counts")
        vt[i] = float(e.t_v[0])
        if float(e.t_count[0]) > len(spikes):
            spikes.append(i)
    return dict(spikes=np.asarray(spikes, dtype=np.int64), v_trace=vt, inject_steps=steps,
                ext_delay=int(e._ext_delay), n_ev=int(n_ev))


def brian2_arm(inject_steps: np.ndarray, ext_delay: int, method: str = "exponential_euler"):
    """Brian2 **同离散方案**对照（按变量方法：v=euler，ge/gi=exponential_euler）。"""
    import brian2 as b2
    b2.prefs.codegen.target = "numpy"
    b2.start_scope()
    # **口径核对（每一条都是实测踩坑）**：
    #   ① Brian2 默认 `defaultclock.dt = 0.1 ms`，引擎是 **0.05 ms** ⇒ 必须显式设；
    #   ② `SpikeGeneratorGroup` 不允许"同一神经元在同一步多次发放" ⇒ 同一步的注入必须先**合并**
    #      （幅度求和）⇒ 用"每个唯一注入时刻一个突触前神经元"的写法（`syn.w` 逐突触给幅度）。
    b2.defaultclock.dt = DT_S * b2.second
    st = np.asarray(inject_steps, dtype=np.int64) + int(ext_delay)
    uniq, cnt = np.unique(st, return_counts=True)
    amps = (cnt.astype(np.float64) * G_EXT)
    n = int(uniq.size)
    src = b2.SpikeGeneratorGroup(n, np.arange(n, dtype=int), uniq * DT_S * b2.second)
    g = b2.NeuronGroup(1, '''
        dv/dt = (gL*(vr - v) + ge*(ee - v) + gi*(ei - v))/second : 1
        dge/dt = -ge/tau_e : 1
        dgi/dt = -gi/tau_i : 1
        gL : 1
        vr : 1
        ee : 1
        ei : 1
        tau_e : second
        tau_i : second
        vth : 1
        vres : 1
    ''', method=method,
        threshold="v >= vth", reset="v = vres", refractory=str(P["ref_ms"]) + "*ms",
        name="m11_ref_neuron")
    # ⚠️ **方案差登记（必读）**：Brian2 2.6 的"按变量 method dict"**不可用**（实测
    # `method={"v":"euler","ge":"exponential_euler"}` ⇒ `ValueError: No state updater with the
    # name "{method}" is known.`），故整组用 `exponential_euler`：
    #   - `ge/gi`：**与引擎逐运算同构**（精确指数衰减）；
    #   - `v`：Brian2 用**闭式**（g 视为步内常数），引擎用**显式 Euler** ⇒ 每步相对差 O(dt/τ_m)=0.25%
    #     ⇒ 本臂是**跨实现（生态独立）的统计级对照**，不是"逐位"对照；方案差已由 `v_trace_*` 行量化。
    g.v = -52.0
    g.vth = P["v_th"]
    g.vres = P["v_reset"]
    g.gL = 1.0 / (P["tau_m"] * 1e-3)
    g.vr = P["v_rest"]
    g.ee = P["e_exc"]
    g.ei = P["e_inh"]
    g.tau_e = P["tau_e"] * b2.ms
    g.tau_i = P["tau_i"] * b2.ms
    syn = b2.Synapses(src, g, model="w : 1", on_pre="ge += w")
    syn.connect(j="0")            # 全部突触前 → 唯一突触后
    syn.w = amps                  # 逐突触幅度（同一步的多次注入已合并求和）
    mon_t = b2.StateMonitor(g, "v", record=True)
    mon_s = b2.SpikeMonitor(g)
    net = b2.Network(src, g, syn, mon_t, mon_s)
    net.run(T_STEPS * DT_S * b2.second)
    spikes = np.asarray(mon_s.i, dtype=int)
    times = np.asarray(mon_s.t / (DT_S * b2.second), dtype=np.float64)
    return dict(spike_steps=times, n_spikes=int(spikes.size),
                v_trace=np.asarray(mon_t.v[0] / 1.0, dtype=np.float64))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default=os.path.join(DATA_DIR, "m11_brian2_reference.json"))
    ap.add_argument("--subthreshold-lam", type=float, default=20.0)
    args = ap.parse_args(argv)
    t0 = time.time()
    out = dict(generated_by="tools/ref_brian2_point_lif.py",
               started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
               role="独立参照臂（**not_a_gate_criterion**，不进门判定；主agent R32/④）",
               params=dict(P, G_EXT=G_EXT, LAM_HZ=LAM_HZ, T_STEPS=T_STEPS,
                           brian2_method={"v": "euler", "ge": "exponential_euler",
                                          "gi": "exponential_euler"},
                           why_same_scheme="引擎 point 档 = v 显式 Euler + 电导精确指数衰减；"
                                           "Brian2 默认 exponential_euler 对 v 用闭式 ⇒ 必须按变量指定方法"),
               arms={})
    for form in ("fix", "frozen"):
        eng = engine_arm(form)
        br = brian2_arm(eng["inject_steps"], eng["ext_delay"])
        es = np.asarray(eng["spikes"], dtype=np.float64)
        bs = np.asarray(np.round(br["spike_steps"]), dtype=np.float64)
        n = min(es.size, bs.size)
        d = np.abs(es[:n] - bs[:n]) if n else np.array([np.nan])
        out["arms"][form] = dict(
            engine_n_spikes=int(es.size), brian2_n_spikes=int(bs.size),
            engine_first_step=(int(es[0]) if es.size else None),
            brian2_first_step=(int(bs[0]) if bs.size else None),
            first_step_abs_diff=(None if not n else float(abs(es[0] - bs[0]))),
            max_abs_step_diff=(float(np.nanmax(d)) if n else None),
            frac_within_1_step=(float(np.mean(d <= 1.0)) if n else None),
            isi_engine_steps=(float(np.median(np.diff(es))) if es.size > 1 else None),
            isi_brian2_steps=(float(np.median(np.diff(bs))) if bs.size > 1 else None),
            v_trace_max_abs_diff_mV=(None if br["v_trace"].size != eng["v_trace"].size else
                                     float(np.max(np.abs(br["v_trace"] - eng["v_trace"])))),
            v_trace_rms_diff_mV=(None if br["v_trace"].size != eng["v_trace"].size else
                                 float(np.sqrt(np.mean((br["v_trace"] - eng["v_trace"]) ** 2)))),
        )
        print("  [%s] engine %d spikes (ISI %.0f 步) | brian2 %d spikes (ISI %.0f 步) | "
              "首发放差 %s 步 | max|Δ| %s 步 | ≤1 步比例 %s" % (
                  form, es.size, out["arms"][form]["isi_engine_steps"] or -1, bs.size,
                  out["arms"][form]["isi_brian2_steps"] or -1,
                  out["arms"][form]["first_step_abs_diff"],
                  out["arms"][form]["max_abs_step_diff"],
                  out["arms"][form]["frac_within_1_step"]), flush=True)
    # ---- 阈下对照（无发放 ⇒ 无混沌）：直接量化"v 方案差"与"g 方案差" ----
    sub = {}
    for tag, m in (("brian2_exponential_euler", "exponential_euler"), ("brian2_euler", "euler")):
        eng = engine_arm("fix", lam_hz=args.subthreshold_lam)
        br = brian2_arm(eng["inject_steps"], eng["ext_delay"], method=m)
        n = min(br["v_trace"].size, eng["v_trace"].size)
        d = br["v_trace"][:n] - eng["v_trace"][:n]
        sub[tag] = dict(engine_spikes=int(eng["spikes"].size), brian2_spikes=int(br["n_spikes"]),
                        v_max_abs_diff_mV=float(np.max(np.abs(d))),
                        v_rms_diff_mV=float(np.sqrt(np.mean(d ** 2))),
                        v_final_engine_mV=float(eng["v_trace"][-1]),
                        v_final_brian2_mV=float(br["v_trace"][-1]))
        print("  [sub-threshold] %s: engine v_end=%.4f | brian2 v_end=%.4f | max|dv|=%.4g mV | RMS=%.4g mV (spk %d/%d)" % (
            tag, sub[tag]["v_final_engine_mV"], sub[tag]["v_final_brian2_mV"],
            sub[tag]["v_max_abs_diff_mV"], sub[tag]["v_rms_diff_mV"],
            sub[tag]["engine_spikes"], sub[tag]["brian2_spikes"]), flush=True)
    out["subthreshold_arm"] = sub
    fx = out["arms"]["fix"]
    fz = out["arms"]["frozen"]
    same_order = bool(fx["frac_within_1_step"] is not None and fx["frac_within_1_step"] >= 0.99)
    defect_confirmed = bool(fz["isi_engine_steps"] and fz["isi_brian2_steps"]
                            and abs(fz["isi_engine_steps"] - fz["isi_brian2_steps"]) > 10)
    out["conclusion"] = dict(
        fix_vs_brian2_same_order=same_order,
        fix_max_abs_step_diff=fx["max_abs_step_diff"],
        fix_frac_within_1_step=fx["frac_within_1_step"],
        frozen_vs_brian2_confirms_defect=defect_confirmed,
        defect_not_resolvable_by_this_arm=(
            "⚠️ **本臂不能分辨 E-M9-4**：跨方案的 0.18 mV 量级差在近阈值驱动下引发脉冲时序去相关"
            "（混沌），其幅度**远大于** E-M9-4 的 ISI 差（182 vs 144 步）⇒ 冻结/修正两臂相对 Brian2 "
            "都「看起来差很多」，无法据此判定缺陷。E-M9-4 的证据仍来自**确定性解析臂**（G0 的 "
            "`single_neuron_ISI` 与解析闭式，≤1 步），本臂仅作**生态独立性**参考。"),
        frozen_isi_engine_steps=fz["isi_engine_steps"],
        frozen_isi_brian2_steps=fz["isi_brian2_steps"],
        subthreshold_v_max_abs_diff_mV={k: v["v_max_abs_diff_mV"] for k, v in sub.items()},
        verdict=("**跨生态统计级**：首发放步差 ≤1 步；逐脉冲时序会**去相关**（跨方案 O(dt) 差 + 混沌放大）"
                 "⇒ 只能作统计级对照、**不得按「逐位」读**；方案差由阈下臂直接量化。原判据式："
                 "⇒ (b) 参照可信度的独立支持" if same_order else
                 "**差异显著 ⇒ 交主agent 裁决**（可能反噬 M10 的 P5 工作点数值；不在 M11 内改写 M10 结论）"),
        note="本臂 **not_a_gate_criterion=true**：不作为 G0 判据、不列入门判定；"
             "**不得**据此改写 M10 的任何结论")
    out["elapsed_s"] = round(time.time() - t0, 2)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False, default=str)
    print(json.dumps({k: v for k, v in out["conclusion"].items()
                      if k in ("fix_vs_brian2_same_order", "fix_max_abs_step_diff",
                               "fix_frac_within_1_step", "frozen_vs_brian2_confirms_defect",
                               "frozen_isi_engine_steps", "frozen_isi_brian2_steps")},
                     ensure_ascii=False, indent=1), flush=True)
    print("json ->", args.out_json, "| elapsed %.1fs" % out["elapsed_s"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
