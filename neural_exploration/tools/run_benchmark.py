"""M0 引擎基准（清单 §3）：同一 HH 单神经元三引擎各跑一遍 + 20 神经元扩展性探测。

输出：
- reports/neuro/m0_benchmark_results.json   基准结果（供基准表填写）
- reports/neuro/m0_benchmark_traces.npz     各引擎 V(t) 轨迹
- reports/neuro/m0_benchmark_compare.png    参考解 vs 引擎叠加图

用法：
  python tools/run_benchmark.py                  # 跑全部已安装引擎
  python tools/run_benchmark.py --engines brian2,neuron
"""

import argparse
import json
import os
import resource
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.hh_spec import (
    CM, DT, EK, EL, ENA, GK, GL, GNA, T_STIM_END, T_STIM_START, T_TOTAL,
    V0, steady_state,
)
from tools.metrics import first_spike_time, spike_count, waveform_rmse
from tools.reference_data import rk4_reference_trace

REPORTS_DIR = os.path.join(ROOT, "reports", "neuro")
N_NET = 20          # 扩展性探测：~20 神经元随机小网络
T_NET = 100.0       # ms


def _mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


# ---------------------------------------------------------------- Brian2
def _brian2_hh_eqs(stim_expr="Iinj"):
    """HH 方程组（Brian2 语法，参考 brian2.tests 官方写法）。
    stim_expr 为注入电流表达式（A/m² 单位）。"""
    return f"""
    dv/dt = ({stim_expr} - ({GNA}*mS/cm2)*m**3*h*(v-({ENA}*mV)) - ({GK}*mS/cm2)*n**4*(v-({EK}*mV)) - ({GL}*mS/cm2)*(v-({EL}*mV))) / ({CM}*uF/cm2) : volt
    dm/dt = alpham*(1-m)-betam*m : 1
    dh/dt = alphah*(1-h)-betah*h : 1
    dn/dt = alphan*(1-n)-betan*n : 1
    alpham = (0.1/mV)*(v+40*mV)/(1-exp(-(v+40*mV)/(10*mV)))/ms : Hz
    betam = 4*exp(-(v+65*mV)/(18*mV))/ms : Hz
    alphah = 0.07*exp(-(v+65*mV)/(20*mV))/ms : Hz
    betah = 1/(1+exp(-(v+35*mV)/(10*mV)))/ms : Hz
    alphan = (0.01/mV)*(v+55*mV)/(1-exp(-(v+55*mV)/(10*mV)))/ms : Hz
    betan = 0.125*exp(-(v+65*mV)/(80*mV))/ms : Hz
    Iinj : amp/meter**2
    """


def _init_brian2_hh_group(G, mV):
    m0, h0, n0 = steady_state(V0)
    G.v = V0 * mV
    G.m = m0
    G.h = h0
    G.n = n0


def _run_brian2_hh():
    from brian2 import (NeuronGroup, Network, StateMonitor, TimedArray, amp, cm,
                        defaultclock, meter, ms, mV, start_scope, uF, mS, volt, Hz)
    start_scope()
    defaultclock.dt = DT * ms

    i_stim_am2 = 10.0 * 1e-6 * amp / cm ** 2   # 10 µA/cm² → A/m²

    n_steps = int(round(T_TOTAL / DT))
    t_stim = np.arange(n_steps) * DT
    i_values = np.where((t_stim >= T_STIM_START) & (t_stim < T_STIM_END), 1.0, 0.0) * i_stim_am2
    stim = TimedArray(i_values, dt=DT * ms)

    G = NeuronGroup(1, _brian2_hh_eqs("stim(t)"), method="rk4")
    _init_brian2_hh_group(G, mV)
    G.Iinj = 0.0 * amp / meter ** 2

    mon = StateMonitor(G, "v", record=True, dt=DT * ms)
    net = Network(G, mon)
    net.store()                       # 保存初始状态
    net.run(T_TOTAL * ms)             # 预热：触发 JIT 编译，不计时
    net.restore()                     # 恢复初始状态，保证计时轮与首次等价
    t0 = time.perf_counter()
    net.run(T_TOTAL * ms)
    elapsed = time.perf_counter() - t0
    v = np.array(mon.v[0] / mV)
    t = np.array(mon.t / ms)
    return t, v, elapsed


def _run_brian2_net():
    from brian2 import (NeuronGroup, Network, StateMonitor, Synapses, defaultclock, amp,
                        cm, meter, ms, mV, start_scope, uF, mS, volt, Hz)
    start_scope()
    defaultclock.dt = DT * ms
    G = NeuronGroup(N_NET, _brian2_hh_eqs(), method="rk4", threshold="v > -20*mV", refractory=2 * ms)
    _init_brian2_hh_group(G, mV)
    # 随机化学突触：约 20% 连接概率（固定种子，确定性）
    S = Synapses(G, G, model="ds/dt = -s/(2*ms) : 1", on_pre="s += 0.05")
    rng = np.random.default_rng(42)
    conn = rng.random((N_NET, N_NET)) < 0.2
    src, dst = np.nonzero(conn)
    S.connect(i=src, j=dst)
    S.delay = 0.5 * ms
    # 前半神经元注入恒定电流制造活动，其余静息
    G.Iinj = 0.0 * amp / meter ** 2
    G.Iinj[:N_NET // 2] = 6.0 * 1e-6 * amp / cm ** 2
    mon = StateMonitor(G, "v", record=[0, 5, 10])
    net = Network(G, S, mon)
    net.store()                       # 保存初始状态
    net.run(T_NET * ms)               # 预热：触发 JIT 编译，不计时
    net.restore()                     # 恢复初始状态
    t0 = time.perf_counter()
    net.run(T_NET * ms)
    elapsed = time.perf_counter() - t0
    n_spikes = sum(spike_count(mon.v[i] / mV) for i in range(3))
    return {"engine": "brian2", "n_net": N_NET, "net_time_s": elapsed, "net_spikes_probe": n_spikes}


# ---------------------------------------------------------------- NEURON
def _run_neuron_hh():
    from neuron import h
    h.load_file("stdrun.hoc")
    h.dt = DT
    h.tstop = T_TOTAL
    h.v_init = V0

    soma = h.Section(name="soma")
    soma.L = 50.0     # µm
    soma.diam = 50.0  # µm
    soma.insert("hh")
    soma(0.5).hh.el = EL   # 覆盖 hh 默认 -54.3，对齐清单 -54.4

    area_um2 = 3.141592653589793 * soma.diam * soma.L   # 圆柱侧面积 µm²
    area_cm2 = area_um2 * 1e-8                          # µm² → cm²
    amp_nA = 10.0e-6 * 1e9 * area_cm2                   # 10 µA/cm² × 面积 → nA

    iclamp = h.IClamp(soma(0.5))
    iclamp.delay = T_STIM_START
    iclamp.dur = T_STIM_END - T_STIM_START
    iclamp.amp = amp_nA

    v_vec = h.Vector()
    t_vec = h.Vector()
    v_vec.record(soma(0.5)._ref_v)
    t_vec.record(h._ref_t)

    t0 = time.perf_counter()
    h.finitialize(V0)
    h.continuerun(T_TOTAL)
    elapsed = time.perf_counter() - t0

    t = np.array(t_vec)
    v = np.array(v_vec)
    return t, v, elapsed


def _run_neuron_net():
    from neuron import h
    h.load_file("stdrun.hoc")
    h.dt = DT
    h.tstop = T_NET
    h.v_init = V0

    cells = []
    for _ in range(N_NET):
        sec = h.Section()
        sec.L = 50.0
        sec.diam = 50.0
        sec.insert("hh")
        cells.append(sec)

    # 随机连接：NetCon from cell i spike → ExpSyn on cell j
    rng = np.random.default_rng(42)
    ncs = []
    for i in range(N_NET):
        for j in range(N_NET):
            if i != j and rng.random() < 0.2:
                syn = h.ExpSyn(cells[j](0.5))
                syn.tau = 2.0
                syn.e = 0.0
                nc = h.NetCon(cells[i](0.5)._ref_v, syn, sec=cells[i])
                nc.threshold = -20.0
                nc.delay = 0.5
                nc.weight[0] = 0.05
                ncs.append(nc)

    t0 = time.perf_counter()
    h.finitialize(V0)
    h.continuerun(T_NET)
    elapsed = time.perf_counter() - t0
    return {"engine": "neuron", "n_net": N_NET, "net_time_s": elapsed, "net_spikes_probe": None}


# ---------------------------------------------------------------- NEST
def _run_nest_hh():
    import nest
    nest.ResetKernel()
    nest.resolution = DT
    nest.simulation_time = T_TOTAL
    nest.rng_seed = 42
    # NEST 内置 hh_psc_alpha 神经元（HH 参数近似，用于可行性/性能评估）
    neurons = nest.Create("hh_psc_alpha", 1)
    dc = nest.Create("dc_generator", 1)
    nest.SetStatus(dc, {"amplitude": 1000.0, "start": T_STIM_START, "stop": T_STIM_END})  # pA
    nest.Connect(dc, neurons, "one_to_one")
    vm = nest.Create("voltmeter", 1)
    nest.SetStatus(vm, {"interval": DT, "record_from": ["V_m"]})
    nest.Connect(neurons, vm, "one_to_one")
    t0 = time.perf_counter()
    nest.Simulate(T_TOTAL)
    elapsed = time.perf_counter() - t0
    d = nest.GetStatus(vm)[0]["events"]
    t = np.array(d["times"])
    v = np.array(d["V_m"])
    return t, v, elapsed


def _run_nest_net():
    import nest
    nest.ResetKernel()
    nest.resolution = DT
    nest.simulation_time = T_NET
    nest.rng_seed = 42
    neurons = nest.Create("hh_psc_alpha", N_NET)
    nest.Connect(neurons, neurons, {"rule": "pairwise_bernoulli", "p": 0.2},
                 {"model": "static_synapse", "weight": 50.0, "delay": 0.5})
    t0 = time.perf_counter()
    nest.Simulate(T_NET)
    elapsed = time.perf_counter() - t0
    return {"engine": "nest", "n_net": N_NET, "net_time_s": elapsed, "net_spikes_probe": None}


ENGINES = {
    "brian2": {"hh": _run_brian2_hh, "net": _run_brian2_net, "import": "brian2"},
    "neuron": {"hh": _run_neuron_hh, "net": _run_neuron_net, "import": "neuron"},
    "nest": {"hh": _run_nest_hh, "net": _run_nest_net, "import": "nest"},
}


def _importable(mod):
    try:
        __import__(mod)
        return True
    except Exception:
        return False


def run_benchmark(engines=None):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    t_ref, v_ref = rk4_reference_trace()

    results = {}
    traces = {}
    for name, cfg in ENGINES.items():
        if engines and name not in engines:
            continue
        if not _importable(cfg["import"]):
            results[name] = {"installed": False, "reason": f"import {cfg['import']} 失败"}
            print(f"[{name}] 未安装/不可导入，跳过")
            continue
        print(f"[{name}] 运行 HH 基准…")
        try:
            t, v, elapsed = cfg["hh"]()
            rmse = waveform_rmse(v, v_ref, DT)
            n_spike = spike_count(v)
            fst = first_spike_time(v, t)
            traces[name] = {"t": t, "v": v}
            res = {
                "installed": True,
                "hh_time_s": elapsed,
                "hh_spikes": n_spike,
                "hh_first_spike_ms": fst,
                "hh_rmse_mv": rmse,
                "hh_rmse_pct": rmse / (np.ptp(v_ref) + 1e-9) * 100.0,
                "v_peak_mv": float(v.max()),
                "v_min_mv": float(v.min()),
            }
            # 参考解对照
            res["ref_spikes"] = spike_count(v_ref)
            res["ref_first_spike_ms"] = first_spike_time(v_ref, t_ref)
        except Exception as e:
            res = {"installed": True, "hh_error": f"{type(e).__name__}: {e}"}
            print(f"  HH 基准失败: {e}")
        # 扩展性探测
        try:
            net = cfg["net"]()
            res.update(net)
        except Exception as e:
            res["net_error"] = f"{type(e).__name__}: {e}"
            print(f"  网络探测失败: {e}")
        results[name] = res

    out_json = os.path.join(REPORTS_DIR, "m0_benchmark_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    if traces:
        np.savez(os.path.join(REPORTS_DIR, "m0_benchmark_traces.npz"),
                 t_ref=t_ref, v_ref=v_ref,
                 **{f"{k}_t": vv["t"] for k, vv in traces.items()},
                 **{f"{k}_v": vv["v"] for k, vv in traces.items()})
        _plot_compare(t_ref, v_ref, traces)
    return results


def _plot_compare(t_ref, v_ref, traces):
    from tools.plot_trace import plot_trace
    v2 = next(iter(traces.values()))["v"]
    label2 = f"{next(iter(traces))} (RMSE {waveform_rmse(v2, v_ref, DT):.3f} mV)"
    out = plot_trace(t_ref, v_ref, title="M0 HH benchmark: RK4 reference vs engine",
                     out_png=os.path.join(REPORTS_DIR, "m0_benchmark_compare.png"),
                     v2=v2, label2=label2)
    print(f"对比图已生成: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default=None, help="逗号分隔引擎列表，默认全部")
    args = ap.parse_args()
    wanted = args.engines.split(",") if args.engines else None
    res = run_benchmark(wanted)
    print(json.dumps(res, indent=2, ensure_ascii=False))
