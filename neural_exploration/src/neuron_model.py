"""M1 正式实现：多隔室 HH 神经元组装（morphology + channels → SpatialNeuron）。

`MultiCompartmentNeuron`：由 CSV 规格（tools/load_morphology.py）驱动构建
Brian2 SpatialNeuron（胞体 + 主动树突 + 髓鞘轴突 + 郎飞结），并提供：
  - 胞体电流注入（point current，nA；与 NEURON IClamp 同单位 → 参考解直接可比）
  - 指定隔室的 V 轨迹记录与逐隔室发放时刻（SpikeMonitor）
  - 确定性运行（无随机性；同参数逐位一致 → P1）

`SingleCompartmentHH`：单隔室对照模型（M0 闭环同款方程），供 f-I 曲线对照
与 M0 结果交叉校验。

数值方法（M1 实测，见 m1_env_notes.md）：
  - 主线 rk4 / dt=0.01ms（清单 §3 指定）；修正 Im 符号后对 1.5µm 郎飞结稳定；
  - 备用 exponential_euler（对电缆扩散项无条件稳定），精度自检用。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.src.ion_channels import hh_equations, steady_state_gates  # noqa: E402
from neural_exploration.src.morphology import (  # noqa: E402
    SectionIndexMap,
    apply_channel_densities,
    build_brian2_morphology,
)
from neural_exploration.tools.load_morphology import (  # noqa: E402
    CM, DEFAULT_CSV, RA, V0, MorphologySpec, load_morphology,
)


@dataclass
class MultiCompartmentResult:
    """一次运行的输出。"""

    t_ms: np.ndarray
    v_mv: Dict[str, np.ndarray]          # 标签 → V(t)（mV）
    spike_times_ms: Dict[str, np.ndarray]  # 标签 → 发放时刻（ms）
    meta: Dict = field(default_factory=dict)


class MultiCompartmentNeuron:
    """多隔室 HH 神经元（Brian2 SpatialNeuron，CSV 驱动）。"""

    def __init__(
        self,
        csv_path: Optional[str] = None,
        dt_ms: float = 0.01,
        method: str = "rk4",
        refractory_ms: float = 2.0,
        threshold_mv: float = -20.0,
        t_total_ms: float = 100.0,
        name: str = "m1_neuron",
        channel_overrides: Optional[Dict[str, Dict[str, float]]] = None,
        extra_im_terms: str = "",
        extra_eqs: str = "",
        stim_var: str = "stim",
    ):
        self.spec: MorphologySpec = load_morphology(csv_path)
        self.dt_ms = dt_ms
        self.method = method
        self.refractory_ms = refractory_ms
        self.threshold_mv = threshold_mv
        self.t_total_ms = t_total_ms
        self.name = name
        #: 区段级参数覆盖（键: 区段名 → {'gna','gk','gl','cm'}，单位同 CSV）。
        #: 每次 build 时在 CSV 赋值之后应用（供参数扫描/调参，M1 报告记录最终值）。
        self.channel_overrides: Dict[str, Dict[str, float]] = channel_overrides or {}
        #: M2 扩展钩子：突触方程片段（见 ion_channels.hh_equations），默认空 = M1 原行为
        self.extra_im_terms = extra_im_terms
        self.extra_eqs = extra_eqs
        #: 注入电流 TimedArray 变量名（M2 神经元对分别命名，避免共享 stim）
        self.stim_var = stim_var
        self._built = False

    # ------------------------------------------------------------------ #
    # 构建
    # ------------------------------------------------------------------ #
    def build(self):
        """构建 SpatialNeuron 并赋初始条件/通道密度。重复调用前需 start_scope。"""
        from neural_exploration.src.brian_env import configure_brian2
        from brian2 import SpatialNeuron, cm, defaultclock, ms, mV, ohm, start_scope, uF

        configure_brian2()
        start_scope()
        defaultclock.dt = self.dt_ms * ms

        morpho, index_map = build_brian2_morphology(self.spec)
        eqs = hh_equations(
            with_point_current=True,
            extra_im_terms=self.extra_im_terms,
            extra_eqs=self.extra_eqs,
            stim_var=self.stim_var,
        )

        neuron = SpatialNeuron(
            morphology=morpho,
            model=eqs,
            Cm=CM * uF / cm ** 2,
            Ri=RA * ohm * cm,
            method=self.method,
            threshold=f"v > {self.threshold_mv}*mV",
            refractory=self.refractory_ms * ms,
            name=self.name,
        )
        # 初始条件：v=V0，门控=稳态（与 NEURON 参考解的 hh 稳态一致）
        neuron.v = V0 * mV
        m0, h0, n0 = steady_state_gates(V0)
        neuron.m = m0
        neuron.h = h0
        neuron.n = n0
        apply_channel_densities(neuron, self.spec, index_map)
        self._apply_overrides(neuron, index_map)

        self.neuron = neuron
        self.morpho = morpho
        self.index_map: SectionIndexMap = index_map
        self._built = True
        return self

    def _apply_overrides(self, neuron, index_map: SectionIndexMap):
        """把 channel_overrides（区段名→{gna,gk,gl,cm}）应用到隔室。"""
        from brian2 import mS, cm, uF

        for seg_name, over in self.channel_overrides.items():
            for i in index_map[seg_name]:
                if "gna" in over:
                    neuron[i].gNa = over["gna"] * mS / cm ** 2
                if "gk" in over:
                    neuron[i].gK = over["gk"] * mS / cm ** 2
                if "gl" in over:
                    neuron[i].gL = over["gl"] * mS / cm ** 2
                if "cm" in over:
                    neuron[i].Cm = over["cm"] * uF / cm ** 2

    # ------------------------------------------------------------------ #
    # 便利访问
    # ------------------------------------------------------------------ #
    def soma_area_cm2(self) -> float:
        """胞体表面积（cm²，球体 π·d²）。"""
        from brian2 import cm as bcm
        return float(self.neuron.main.area[0] / bcm ** 2)

    def density_to_nA(self, density_uA_cm2: float, compartment: int = 0) -> float:
        """电流密度（µA/cm²）→ 该隔室总电流（nA）。"""
        from brian2 import cm as bcm
        area = float(self.neuron[compartment].area[0] / bcm ** 2)
        return density_uA_cm2 * 1e-6 * area * 1e9  # µA/cm²·cm² = µA → nA

    def label_of(self, segment: str, compartment: Optional[int] = None) -> int:
        """区段（及段内隔室下标）→ 绝对隔室索引。"""
        arr = self.index_map[segment]
        if compartment is None:
            return int(arr[0])
        return int(arr[compartment])

    # ------------------------------------------------------------------ #
    # 运行
    # ------------------------------------------------------------------ #
    def run_stimulus(
        self,
        amplitude_uA_cm2: float = 10.0,
        stim_start_ms: float = 5.0,
        stim_end_ms: Optional[float] = None,
        t_total_ms: Optional[float] = None,
        record: Optional[Sequence[str]] = None,
        record_all: bool = False,
        inject_at: str = "soma",
        inject_compartment: int = 0,
    ) -> MultiCompartmentResult:
        """在指定隔室注入恒定电流（µA/cm² 密度 → nA point current）。

        amplitude<=0 时为无刺激对照（注入零电流）。
        record: 标签列表（区段名或 `区段#下标`，如 'node1'、'dend2#1'）；
        record_all: 记录全部隔室（V 与发放时刻），供跳跃传导可视化。
        """
        from brian2 import Network, StateMonitor, SpikeMonitor, TimedArray, amp, ms, mV, nA

        # 每次运行重建（fresh 网络，保证状态干净、可重复、支持参数扫描）
        self.build()
        t_total = t_total_ms or self.t_total_ms
        stim_end = stim_end_ms if stim_end_ms is not None else t_total
        n_steps = int(round(t_total / self.dt_ms))

        # 刺激轨迹（逐时间步 × 逐隔室；point current 单位 amp）
        stim2d = np.zeros((n_steps, self.spec.total_compartments)) * amp
        if amplitude_uA_cm2 > 0:
            i_total_nA = self.density_to_nA(amplitude_uA_cm2, self.label_of(inject_at, inject_compartment))
            i0 = int(round(stim_start_ms / self.dt_ms))
            i1 = int(round(stim_end / self.dt_ms))
            stim2d[i0:i1, self.label_of(inject_at, inject_compartment)] = i_total_nA * nA
        stim = TimedArray(stim2d, dt=self.dt_ms * ms)

        # 记录目标
        rec_idx: List[int] = []
        rec_label: List[str] = []
        if record_all:
            # 全部隔室：标签 comp{绝对索引}（唯一，避免同区段多隔室同名覆盖）
            rec_idx = list(range(self.spec.total_compartments))
            rec_label = [f"comp{i}" for i in rec_idx]
        if record:
            for lab in record:
                if "#" in lab:
                    seg, _, sub = lab.partition("#")
                    idx = self.label_of(seg, int(sub))
                    key = f"{seg}#{sub}"
                else:
                    idx = self.label_of(lab)
                    key = lab
                if idx not in rec_idx:
                    rec_idx.append(idx)
                    rec_label.append(key)

        mon = StateMonitor(self.neuron, "v", record=rec_idx, dt=self.dt_ms * ms)
        spmon = SpikeMonitor(self.neuron, "v")

        # 显式 Network（类内构建的对象不在 run() 调用帧，magic 收集不可靠）；
        # TimedArray 非 BrianObject，经 run namespace 传入方程解析
        net = Network(self.neuron, mon, spmon)
        net.run(t_total * ms, namespace={"stim": stim})

        t = np.array(mon.t / ms)
        v = {lab: np.array(mon.v[pos] / mV) for pos, (idx, lab) in enumerate(zip(rec_idx, rec_label))}
        # 发放时刻：按隔室标签聚合
        spikes: Dict[str, np.ndarray] = {}
        for seg in self.spec.segments:
            idxs = self.index_map[seg.name]
            times = np.array(spmon.t[spmon.i == idxs[0]] / ms) if len(idxs) else np.array([])
            spikes[seg.name] = times
        if record_all:
            for i in range(self.spec.total_compartments):
                times = np.array(spmon.t[spmon.i == i] / ms)
                spikes[f"comp{i}"] = times

        return MultiCompartmentResult(
            t_ms=t,
            v_mv=v,
            spike_times_ms=spikes,
            meta=dict(
                amplitude_uA_cm2=amplitude_uA_cm2,
                stim_start_ms=stim_start_ms,
                stim_end_ms=stim_end,
                t_total_ms=t_total,
                dt_ms=self.dt_ms,
                method=self.method,
                inject_at=f"{inject_at}#{inject_compartment}",
            ),
        )


# --------------------------------------------------------------------- #
# 单隔室对照（M0 闭环同款方程）
# --------------------------------------------------------------------- #
class SingleCompartmentHH:
    """单隔室 HH 神经元（对照用；方程与 M0 smoke_loop 逐项一致）。"""

    def __init__(
        self,
        dt_ms: float = 0.01,
        method: str = "rk4",
        refractory_ms: float = 2.0,
        t_total_ms: float = 500.0,
    ):
        self.dt_ms = dt_ms
        self.method = method
        self.refractory_ms = refractory_ms
        self.t_total_ms = t_total_ms

    def firing_rate(self, amplitude_uA_cm2: float, warmup_ms: float = 20.0) -> float:
        """恒流刺激下的稳态发放频率（Hz）。0 表示不发放。"""
        from brian2 import (
            Network, NeuronGroup, StateMonitor, TimedArray, amp, cm, defaultclock,
            meter, ms, mS, mV, start_scope, uF,
        )

        start_scope()
        defaultclock.dt = self.dt_ms * ms
        from neural_exploration.tools.hh_spec import CM, DT, EK, EL, ENA, GK, GL, GNA, steady_state

        eqs = f"""
        dv/dt = (stim(t) - ({GNA}*mS/cm2)*m**3*h*(v-({ENA}*mV)) - ({GK}*mS/cm2)*n**4*(v-({EK}*mV)) - ({GL}*mS/cm2)*(v-({EL}*mV))) / ({CM}*uF/cm2) : volt
        dm/dt = alpham*(1-m)-betam*m : 1
        dh/dt = alphah*(1-h)-betah*h : 1
        dn/dt = alphan*(1-n)-betan*n : 1
        alpham = (0.1/mV)*(v+40*mV)/(1-exp(-(v+40*mV)/(10*mV)))/ms : Hz
        betam = 4*exp(-(v+65*mV)/(18*mV))/ms : Hz
        alphah = 0.07*exp(-(v+65*mV)/(20*mV))/ms : Hz
        betah = 1/(1+exp(-(v+35*mV)/(10*mV)))/ms : Hz
        alphan = (0.01/mV)*(v+55*mV)/(1-exp(-(v+55*mV)/(10*mV)))/ms : Hz
        betan = 0.125*exp(-(v+65*mV)/(80*mV))/ms : Hz
        """
        n_steps = int(round(self.t_total_ms / self.dt_ms))
        i_am2 = amplitude_uA_cm2 * 1e-6 * amp / cm ** 2
        stim = TimedArray(np.full(n_steps, 1.0) * i_am2, dt=self.dt_ms * ms)

        g = NeuronGroup(1, eqs, method=self.method, threshold="v > -20*mV",
                        refractory=self.refractory_ms * ms)
        m0, h0, n0 = steady_state(-65.0)
        g.v = -65.0 * mV
        g.m, g.h, g.n = m0, h0, n0

        mon = StateMonitor(g, "v", record=True, dt=self.dt_ms * ms)
        net = Network(g, mon)
        # 方程中的单位/函数来自本方法局部命名空间 → 显式传入 run namespace
        ns = {k: v for k, v in locals().items() if not k.startswith("__")}
        net.run(self.t_total_ms * ms, namespace=ns)
        v = np.array(mon.v[0] / mV)
        t = np.array(mon.t / ms)
        from neural_exploration.tools.metrics import spike_count

        n_spikes = spike_count(v)
        if n_spikes == 0:
            return 0.0
        # 稳态频率：首个发放后的时间窗
        rising = (v[:-1] < -20.0) & (v[1:] >= -20.0)
        first = np.flatnonzero(rising)[0] + 1
        last = np.flatnonzero(rising)[-1] + 1
        span = (t[last] - t[first]) / 1000.0  # s
        if span <= 0:
            return 0.0
        return (np.sum(rising) - 1) / span if np.sum(rising) > 1 else (np.sum(rising) / max(span, 1e-9))
