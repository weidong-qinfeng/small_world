"""M2 组装：两个 M1 多隔室神经元 + 突触（化学/缝隙）→ 神经元对。

清单 §4.1：`neuron_pair.py` 职责：
  - 构建 pre/post 两个 MultiCompartmentNeuron（按需附加突触方程片段）；
  - 挂载 ChemicalSynapse（默认 node3 → soma）与 GapJunction（默认 soma ↔ soma）；
  - 提供刺激协议（单脉冲 / 50Hz 脉冲串 / 保持电流）与记录；
  - 提供多试次重复运行（P2 释放失败统计，store/restore 复用网络状态）。

刺激约定：
  - pre_pulses / post_pulses：`(start_ms, dur_ms, amp_uA_cm2, site)` 列表；
  - 每个神经元使用独立 TimedArray（stim_var 区分，避免共享 stim 串扰）；
  - 首脉冲建议 t ≥ 50ms（M1 已记录的 HH 静息瞬态漂移，见 m1_env_notes §L3）。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.src.neuron_model import MultiCompartmentNeuron  # noqa: E402
from neural_exploration.src.synapse_model import (  # noqa: E402
    ChemicalSynapse,
    GapJunction,
    SynapseParams,
    chemical_im_terms,
    chemical_post_eqs,
    gap_im_term,
    GAP_POST_EQ,
    load_synapse_params,
)

# 脉冲元组： (start_ms, dur_ms, amp_uA_cm2, site)
Pulse = Tuple[float, float, float, str]


@dataclass
class PairResult:
    """一次运行的输出。"""

    t_ms: np.ndarray
    v_mv: Dict[str, np.ndarray]            # 标签 → V(t)（mV），如 'pre_node3'/'post_soma'
    spike_times_ms: Dict[str, np.ndarray]  # 标签 → 发放时刻（ms）
    g: Dict[str, np.ndarray] = field(default_factory=dict)  # 电导标签 → 值（S/m² 或 A）
    meta: Dict = field(default_factory=dict)


def pulse_train(
    start_ms: float, freq_hz: float, n_pulses: int, dur_ms: float,
    amp_uA_cm2: float, site: str = "soma",
) -> List[Pulse]:
    """50Hz 等间隔脉冲串 → Pulse 列表（清单 P3 协议）。"""
    return [
        (start_ms + k * 1000.0 / freq_hz, dur_ms, amp_uA_cm2, site)
        for k in range(n_pulses)
    ]


class NeuronPair:
    """两个 M1 神经元 + 突触的组装体。"""

    def __init__(
        self,
        dt_ms: float = 0.01,
        method: str = "rk4",
        t_total_ms: float = 300.0,
        seed: int = 0,
        name_prefix: str = "pair",
        csv_path: Optional[str] = None,
    ):
        self.dt_ms = dt_ms
        self.method = method
        self.t_total_ms = t_total_ms
        self.seed = seed
        self.name_prefix = name_prefix
        self.params: Dict[str, SynapseParams] = load_synapse_params(csv_path)
        self.chemicals: Dict[str, ChemicalSynapse] = {}
        self.gaps: List[GapJunction] = []
        self.pre = None
        self.post = None
        self._built = False

    # ------------------------------------------------------------------ #
    # 突触装配
    # ------------------------------------------------------------------ #
    def add_chemical(
        self,
        synapse_type: str,
        g_max_ns: Optional[float] = None,
        p_release: Optional[float] = None,
        n_vesicles: Optional[int] = None,
        stp: Optional[Tuple[float, float, float]] = None,  # (u0, tau_fac_ms, tau_rec_ms)
        pre_site: str = "node3",
        post_site: str = "soma",
        mg_mm: Optional[float] = None,
        name: Optional[str] = None,
    ) -> "NeuronPair":
        p = self.params[synapse_type]
        if g_max_ns is not None:
            p.g_max_ns = g_max_ns
        if p_release is not None:
            p.p_release = p_release
        if n_vesicles is not None:
            p.n_vesicles = n_vesicles
        if stp is not None:
            p.u0, p.tau_fac_ms, p.tau_rec_ms = stp
        if mg_mm is not None:
            p.mg_mm = mg_mm
        self.chemicals[synapse_type] = ChemicalSynapse(
            None, None, p, pre_site=pre_site, post_site=post_site,
            name=name or f"syn_{synapse_type}",
        )
        return self

    def add_gap(self, g_gap_ns: float, pre_site: str = "soma",
                post_site: str = "soma") -> "NeuronPair":
        self.gaps.append(GapJunction(None, None, g_gap_ns, pre_site=pre_site,
                                     post_site=post_site, name=f"gap_{len(self.gaps)}"))
        return self

    # ------------------------------------------------------------------ #
    # 构建
    # ------------------------------------------------------------------ #
    def build(self):
        """构建 pre/post 神经元 + 突触（每次 run 前自动重建）。"""
        from neural_exploration.src.brian_env import configure_brian2
        from brian2 import start_scope

        configure_brian2()
        start_scope()

        post_eqs = chemical_post_eqs(self.params)
        post_im = chemical_im_terms(self.params)
        if self.gaps:
            post_eqs = "\n".join(x for x in (post_eqs, GAP_POST_EQ) if x)
        # pre 只需缝隙连接 point current
        pre_eqs = GAP_POST_EQ if self.gaps else ""

        self.pre = MultiCompartmentNeuron(
            name=f"{self.name_prefix}_pre", t_total_ms=self.t_total_ms,
            dt_ms=self.dt_ms, method=self.method, extra_eqs=pre_eqs,
            stim_var="stim_pre",
        )
        self.post = MultiCompartmentNeuron(
            name=f"{self.name_prefix}_post", t_total_ms=self.t_total_ms,
            dt_ms=self.dt_ms, method=self.method, extra_eqs=post_eqs,
            extra_im_terms=post_im, stim_var="stim_post",
        )
        self.pre.build()
        self.post.build()

        for key, cs in self.chemicals.items():
            cs.pre_neuron = self.pre
            cs.post_neuron = self.post
            cs.build()
        for g in self.gaps:
            g.pre_neuron = self.pre
            g.post_neuron = self.post
            g.build()
        self._built = True
        return self

    # ------------------------------------------------------------------ #
    # 刺激与运行
    # ------------------------------------------------------------------ #
    def _stim_arrays(self, pre_pulses, post_pulses, t_total_ms):
        """按神经元各自形态学生成 (stim_pre, stim_post) 两个 TimedArray。"""
        from brian2 import TimedArray, amp, ms, nA

        n_steps = int(round(t_total_ms / self.dt_ms))
        arr_p = np.zeros((n_steps, self.pre.spec.total_compartments)) * amp
        arr_q = np.zeros((n_steps, self.post.spec.total_compartments)) * amp
        for (s0, dur, amp_uA, site) in pre_pulses:
            i0, i1 = int(round(s0 / self.dt_ms)), int(round((s0 + dur) / self.dt_ms))
            idx = self.pre.label_of(site)
            arr_p[i0:i1, idx] = self.pre.density_to_nA(amp_uA, idx) * nA
        for (s0, dur, amp_uA, site) in post_pulses:
            i0, i1 = int(round(s0 / self.dt_ms)), int(round((s0 + dur) / self.dt_ms))
            idx = self.post.label_of(site)
            arr_q[i0:i1, idx] = self.post.density_to_nA(amp_uA, idx) * nA
        return (TimedArray(arr_p, dt=self.dt_ms * ms),
                TimedArray(arr_q, dt=self.dt_ms * ms))

    def _record_spec(self, record: Sequence[str]):
        """'pre_soma'/'post_node3' 等标签 → (group, 隔室索引, 标签) 列表。"""
        from neural_exploration.src.morphology import SectionIndexMap

        out = []
        for lab in record:
            if lab.startswith("pre_"):
                site = lab[4:]
                idx = self.pre.label_of(site)
                out.append((self.pre.neuron, idx, lab))
            elif lab.startswith("post_"):
                site = lab[5:]
                idx = self.post.label_of(site)
                out.append((self.post.neuron, idx, lab))
            else:
                raise ValueError(f"记录标签需以 pre_/post_ 开头：{lab}")
        return out

    def _spike_labels(self) -> List[str]:
        """返回所有记录隔室的发放标签（含 post 全隔室发放时刻，供 P4 双向验证）。"""
        labels = []
        for prefix, neuron in (("pre", self.pre), ("post", self.post)):
            for seg in neuron.spec.segments:
                labels.append(f"{prefix}_{seg.name}")
        return labels

    def _spike_times(self, spmon, neuron, prefix: str) -> Dict[str, np.ndarray]:
        from brian2 import ms

        t_ms_arr = np.array(spmon.t / ms)
        i_arr = np.array(spmon.i)
        out = {}
        for seg in neuron.spec.segments:
            idxs = neuron.index_map[seg.name]
            if len(idxs):
                mask = np.isin(i_arr, idxs)
                out[f"{prefix}_{seg.name}"] = t_ms_arr[mask]
        return out

    def run(
        self,
        pre_pulses: Sequence[Pulse],
        post_pulses: Sequence[Pulse] = (),
        record: Sequence[str] = ("pre_node3", "post_soma"),
        record_g: Sequence[str] = (),
        t_total_ms: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> PairResult:
        """单次运行：重建网络 → 刺激 → 记录 V/发放时刻/电导。"""
        from brian2 import Network, SpikeMonitor, StateMonitor, seed as bseed, ms, mV

        self.build()
        bseed(self.seed if seed is None else seed)
        t_total = t_total_ms or self.t_total_ms

        stim_pre, stim_post = self._stim_arrays(pre_pulses, post_pulses, t_total)

        rec = self._record_spec(record)
        # 逐组记录（pre/post 分别建 monitor，避免跨组 record 混用）
        monos = []
        pre_ids = [r[1] for r in rec if r[0] is self.pre.neuron]
        post_ids = [r[1] for r in rec if r[0] is self.post.neuron]
        pre_labels = [r[2] for r in rec if r[0] is self.pre.neuron]
        post_labels = [r[2] for r in rec if r[0] is self.post.neuron]
        if pre_ids:
            monos.append(StateMonitor(self.pre.neuron, "v", record=pre_ids,
                                      dt=self.dt_ms * ms))
        if post_ids:
            monos.append(StateMonitor(self.post.neuron, "v", record=post_ids,
                                      dt=self.dt_ms * ms))
        gmons = []
        for gvar in record_g:
            if hasattr(self.post.neuron, gvar):
                gmons.append(StateMonitor(self.post.neuron, gvar,
                                          record=[self.post.label_of("soma")],
                                          dt=self.dt_ms * ms))
        sp_pre = SpikeMonitor(self.pre.neuron, "v")
        sp_post = SpikeMonitor(self.post.neuron, "v")

        net = Network(self.pre.neuron, self.post.neuron)
        for cs in self.chemicals.values():
            net.add(cs.synapses)
        for g in self.gaps:
            net.add(g.synapses)
        for m in monos + gmons:
            net.add(m)
        net.add(sp_pre)
        net.add(sp_post)
        net.run(t_total * ms, namespace={"stim_pre": stim_pre, "stim_post": stim_post})

        t = np.array(monos[0].t / ms) if monos else np.arange(0, t_total, self.dt_ms)
        v = {}
        if monos:
            mon_pre_v = monos[0] if pre_ids else None
            mon_post_v = monos[1] if len(monos) > 1 and post_ids else (
                monos[0] if post_ids else None)
            for pos, lab in enumerate(pre_labels):
                v[lab] = np.array(mon_pre_v.v[pos] / mV)
            for pos, lab in enumerate(post_labels):
                v[lab] = np.array(mon_post_v.v[pos] / mV)
        g = {}
        for m, gvar in zip(gmons, record_g):
            g[gvar] = np.array(m.v[0])
        spikes = {}
        spikes.update(self._spike_times(sp_pre, self.pre, "pre"))
        spikes.update(self._spike_times(sp_post, self.post, "post"))

        return PairResult(
            t_ms=t,
            v_mv=v,
            spike_times_ms=spikes,
            g=g,
            meta=dict(t_total_ms=t_total, dt_ms=self.dt_ms, seed=seed if seed is not None else self.seed),
        )

    def run_trials(
        self,
        pre_pulses: Sequence[Pulse],
        n_trials: int,
        seed_base: int = 0,
        record: Sequence[str] = ("post_soma",),
        record_g: Sequence[str] = (),
        t_total_ms: Optional[float] = None,
    ) -> List[PairResult]:
        """多试次重复运行（P2 释放失败统计）：store/restore 复用网络，仅重播种。

        每次 restore 后变量回到快照（g 清零、神经元复位）；monitor 累积，
        按等长窗口切片还原每试次轨迹。
        """
        from brian2 import Network, SpikeMonitor, StateMonitor, ms, seed as bseed, mV

        self.build()
        t_total = t_total_ms or self.t_total_ms
        stim_pre, stim_post = self._stim_arrays(pre_pulses, (), t_total)

        rec = self._record_spec(record)
        pre_ids = [r[1] for r in rec if r[0] is self.pre.neuron]
        post_ids = [r[1] for r in rec if r[0] is self.post.neuron]
        pre_labels = [r[2] for r in rec if r[0] is self.pre.neuron]
        post_labels = [r[2] for r in rec if r[0] is self.post.neuron]
        monos = []
        if pre_ids:
            monos.append(StateMonitor(self.pre.neuron, "v", record=pre_ids,
                                      dt=self.dt_ms * ms))
        if post_ids:
            monos.append(StateMonitor(self.post.neuron, "v", record=post_ids,
                                      dt=self.dt_ms * ms))
        gmons = []
        for gvar in record_g:
            if hasattr(self.post.neuron, gvar):
                gmons.append(StateMonitor(self.post.neuron, gvar,
                                          record=[self.post.label_of("soma")],
                                          dt=self.dt_ms * ms))
        sp_pre = SpikeMonitor(self.pre.neuron, "v")
        sp_post = SpikeMonitor(self.post.neuron, "v")

        net = Network(self.pre.neuron, self.post.neuron)
        for cs in self.chemicals.values():
            net.add(cs.synapses)
        for g in self.gaps:
            net.add(g.synapses)
        for m in monos + gmons:
            net.add(m)
        net.add(sp_pre)
        net.add(sp_post)

        bseed(seed_base)
        net.run(0 * ms)          # 完成初始化（快照前）
        net.store()              # 保存干净状态

        n_steps = int(round(t_total / self.dt_ms))
        results = []
        for trial in range(n_trials):
            bseed(seed_base + trial)
            net.restore()
            net.run(t_total * ms, namespace={"stim_pre": stim_pre, "stim_post": stim_post})
            v = {}
            if monos:
                mon_pre_v = monos[0] if pre_ids else None
                mon_post_v = monos[1] if len(monos) > 1 and post_ids else (
                    monos[0] if post_ids else None)
                for pos, lab in enumerate(pre_labels):
                    v[lab] = np.array(mon_pre_v.v[pos] / mV)[
                        trial * n_steps:(trial + 1) * n_steps]
                for pos, lab in enumerate(post_labels):
                    v[lab] = np.array(mon_post_v.v[pos] / mV)[
                        trial * n_steps:(trial + 1) * n_steps]
            g = {}
            for m, gvar in zip(gmons, record_g):
                g[gvar] = np.array(m.v[0])[trial * n_steps:(trial + 1) * n_steps]
            results.append(PairResult(
                t_ms=np.arange(0, t_total, self.dt_ms),
                v_mv=v, spike_times_ms={}, g=g,
                meta=dict(t_total_ms=t_total, dt_ms=self.dt_ms, trial=trial,
                          seed=seed_base + trial),
            ))
        return results
