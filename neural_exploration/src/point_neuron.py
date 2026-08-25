"""M5 降阶组件：点神经元（单隔室 HH）+ 双隔室神经元（soma+轴突末梢）。

`PointNeuron`：单隔室 HH 神经元（Brian2 `NeuronGroup`，1 神经元）的薄包装，
提供与 M1 `MultiCompartmentNeuron` 一致的 M2 突触组件接口：
  - `.neuron`         → NeuronGroup（1 神经元，单隔室）
  - `.label_of(site)` → 0（任意位点 → 唯一隔室索引；ChemicalSynapse pre_site="node3"
                        / GapJunction soma / Muscle3.connect_driver node3 均映射到 0）
  - `.soma_area_cm2()`→ 点面积（球体 d=20µm ≈ 1.257e-5 cm²，与 M4 CSV ase_site 注释一致）
  - `.density_to_nA()`→ 密度 µA/cm² → nA（按点面积换算，与 M4 同一物理量）

**实测适配结论（本节点 B1b，Brian2 2.6.0 验证，docs/m5_env_notes.md L7）**：
M2 `ChemicalSynapse` / `GapJunction` / `Muscle3.connect_driver` **不经修改**即可复用
PointNeuron（事件/位点语义兼容：`label_of` 映射 + `I_gap : amp` 变量 + on_pre 写回均实测
通过——pre 发放 50.7ms → post 经 AMPA 5nS 于 52.0ms 发放，GapJunction 无报错、数值有限）。

**数值方法实测（L7）**：单隔室 HH 在 **dt=0.1ms 下 rk4 发放后发散（NaN）**，
`exponential_euler` 稳定（vmax≈42.6mV）；dt≤0.05ms 时 rk4/exponential_euler 均稳定。
→ 点档默认 method=exponential_euler、dt=0.1ms；双隔室档 dt=0.05ms 用 rk4。
（M4 L16 教训：dt/形状/命名定稿后不变——扫描按档位一次定稿。）

确定性：无随机性（p=1/n=1 纪律；随机性全部来自试次伪随机起点，M4 惯例）。

`TwoCompartmentNeuron`：双隔室档（soma + 轴突末梢 node3，2 神经元 NeuronGroup +
轴向耦合 g_ax），保留最小空间效应（缩放扫描中间档，§3.2 方案②）。
  - label_of("soma")=0、label_of("node3")=1（ChemicalSynapse node3→soma 语义保持）
"""

from __future__ import annotations

import os
import sys
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from neural_exploration.tools.hh_spec import (  # noqa: E402
    CM as _CM, EK as _EK, EL as _EL, ENA as _ENA, GK as _GK, GL as _GL,
    GNA as _GNA, V0 as _V0,
)

#: 点面积（cm²）：球体 d=20µm（M4 CSV ase_site 注释同值，保证突触电导换算一致）
SOMA_AREA_CM2 = 1.257e-5

#: 双隔室档：轴突末梢 Na 密度（M1 郎飞结值 300 mS/cm²）
_NODE_GNA = 300.0
#: 轴突末梢面积（cm²）：node3 = π·d·L，d=1.5µm、L=2µm（M1 郎飞结）
_NODE_AREA_CM2 = np.pi * 1.5 * 2.0 * 1e-8
#: 轴向耦合电导（S）：soma→node3 全程轴向电阻 R_ax = Σ Ra·L_i/A_i（M1 形态：
#: soma 9.6e4 + ais 1.4e7 + myelin1/2/3 各 5.7e7 + node1/2 各 ~1.7e6 ≈ 1.85e8 Ω）
#: → G_AX = 1/R_ax ≈ 5.4e-9 S（实测：初值 8.4e-7（只算 soma 截面）导致小隔室
#: 被轴向电流驱动发散——M5 L7 实测坑）
_AXIAL_G_S = 5.4e-9


def _point_neuron_eqs(extra_im_terms: str = "", extra_eqs: str = "",
                      stim_var: str = "stim") -> str:
    """单隔室 HH 方程串（NeuronGroup 用）。

    仅使用 Brian2 DEFAULT_UNITS 可解析的单位标识符（siemens/meter/amp/volt/mV/ms/Hz/
    uF/cm/cm2——实测 `mS` 不在 DEFAULT_UNITS，M1 同款处置：密度作状态变量在 Python 侧赋值）；
    Im 内向正约定（g·(E−v)，M1 L4/设计文档 L4）；点电流 (stim(t,i)+I_gap)/AREA 显式入
    dv/dt（不依赖 SpatialNeuron 自动注入）。exponential_euler 在 dt=0.1ms 稳定（L7）。
    """
    return f"""
Im = gL*(EL-v) + gNa*m**3*h*(ENa-v) + gK*n**4*(EK-v){extra_im_terms} : amp/meter**2
dv/dt = (Im + ({stim_var}(t, i) + I_gap + I_gap_in + I_gap_out)/AREA) / Cm : volt
dm/dt = alpham*(1-m)-betam*m : 1
dh/dt = alphah*(1-h)-betah*h : 1
dn/dt = alphan*(1-n)-betan*n : 1
alpham = (0.1/mV)*(v+40*mV)/(1-exp(-(v+40*mV)/(10*mV)))/ms : Hz
betam = 4*exp(-(v+65*mV)/(18*mV))/ms : Hz
alphah = 0.07*exp(-(v+65*mV)/(20*mV))/ms : Hz
betah = 1/(1+exp(-(v+35*mV)/(10*mV)))/ms : Hz
alphan = (0.01/mV)*(v+55*mV)/(1-exp(-(v+55*mV)/(10*mV)))/ms : Hz
betan = 0.125*exp(-(v+65*mV)/(80*mV))/ms : Hz
gNa : siemens/meter**2
gK : siemens/meter**2
gL : siemens/meter**2
EL = {_EL}*mV : volt (shared)
ENa = {_ENA}*mV : volt (shared)
EK = {_EK}*mV : volt (shared)
I_gap : amp
I_gap_in : amp
I_gap_out : amp
""" + (extra_eqs or "")


def _two_comp_eqs(extra_im_terms: str = "", extra_eqs: str = "",
                  stim_var: str = "stim") -> str:
    """双隔室 HH：soma（i=0）+ 轴突末梢 node3（i=1），轴向耦合。

    轴向电流 I_ax = G_AX·(v_peer − v)（amp），v_peer 为对侧隔室电压（linked_var
    索引交换 [1,0]，实测支持）；按隔室面积折密度入 dv/dt。node3 gNa 高（300）→
    node3 先于 soma 触发（M1 郎飞结语义的最小保留）；化学突触 node3→soma 保持。
    """
    return f"""
Im = gL*(EL-v) + gNa*m**3*h*(ENa-v) + gK*n**4*(EK-v){extra_im_terms} : amp/meter**2
dv/dt = (Im + ({stim_var}(t, i) + I_gap + I_gap_in + I_gap_out + I_ax)/AREA) / Cm : volt
dm/dt = alpham*(1-m)-betam*m : 1
dh/dt = alphah*(1-h)-betah*h : 1
dn/dt = alphan*(1-n)-betan*n : 1
alpham = (0.1/mV)*(v+40*mV)/(1-exp(-(v+40*mV)/(10*mV)))/ms : Hz
betam = 4*exp(-(v+65*mV)/(18*mV))/ms : Hz
alphah = 0.07*exp(-(v+65*mV)/(20*mV))/ms : Hz
betah = 1/(1+exp(-(v+35*mV)/(10*mV)))/ms : Hz
alphan = (0.01/mV)*(v+55*mV)/(1-exp(-(v+55*mV)/(10*mV)))/ms : Hz
betan = 0.125*exp(-(v+65*mV)/(80*mV))/ms : Hz
gNa : siemens/meter**2
gK : siemens/meter**2
gL : siemens/meter**2
AREA : meter**2
EL = {_EL}*mV : volt (shared)
ENa = {_ENA}*mV : volt (shared)
EK = {_EK}*mV : volt (shared)
I_gap : amp
I_gap_in : amp
I_gap_out : amp
I_ax = G_AX*(v_peer - v) : amp
v_peer : volt (linked)
""" + (extra_eqs or "")


class PointNeuron:
    """单隔室 HH 点神经元（M2 突触组件兼容薄包装，实测复用成功——L7）。"""

    def __init__(
        self,
        name: str = "pn",
        dt_ms: Optional[float] = None,
        method: Optional[str] = None,
        refractory_ms: float = 2.0,
        threshold_mv: float = -20.0,
        area_cm2: float = SOMA_AREA_CM2,
        gna_mS_cm2: float = _GNA,
        gk_mS_cm2: float = _GK,
        gl_mS_cm2: float = _GL,
        extra_eqs: str = "",
        extra_im_terms: str = "",
        stim_var: str = "stim",
    ):
        self.name = name
        #: dt/方法定稿纪律（M4 L16）：默认点档 = 0.1ms/exponential_euler（L7 实测稳定）
        self.dt_ms = 0.1 if dt_ms is None else float(dt_ms)
        self.method = "exponential_euler" if method is None else method
        self.refractory_ms = refractory_ms
        self.threshold_mv = threshold_mv
        self.area_cm2 = float(area_cm2)
        self.gna = gna_mS_cm2
        self.gk = gk_mS_cm2
        self.gl = gl_mS_cm2
        self.extra_eqs = extra_eqs
        self.extra_im_terms = extra_im_terms
        self.stim_var = stim_var
        self.neuron = None
        self._built = False

    # ------------------------------------------------------------------ #
    # 构建
    # ------------------------------------------------------------------ #
    def build(self):
        """构建单隔室 NeuronGroup（重复调用前需 start_scope）。"""
        from brian2 import (NeuronGroup, cm, defaultclock, meter, ms, mS, mV,
                            uF)
        from neural_exploration.src.brian_env import configure_brian2

        configure_brian2()
        from brian2 import start_scope
        start_scope()
        defaultclock.dt = self.dt_ms * ms
        eqs = _point_neuron_eqs(self.extra_im_terms, self.extra_eqs,
                                self.stim_var)
        ns = {"AREA": self.area_cm2 * 1e-4 * meter ** 2,
              "Cm": _CM * uF / cm ** 2}
        g = NeuronGroup(1, eqs, method=self.method,
                        threshold=f"v > {self.threshold_mv}*mV",
                        refractory=self.refractory_ms * ms,
                        name=self.name, namespace=ns)
        self._init_conditions(g)
        self.neuron = g
        self._built = True
        return self

    def _init_conditions(self, g):
        """初始条件：v=V0 + 门控稳态（与 NEURON/M1 参考同源）。"""
        from brian2 import cm, mS, mV
        from neural_exploration.src.ion_channels import steady_state_gates

        m0, h0, n0 = steady_state_gates(_V0)
        g.v = _V0 * mV
        g.m, g.h, g.n = m0, h0, n0
        g.gNa = self.gna * mS / cm ** 2
        g.gK = self.gk * mS / cm ** 2
        g.gL = self.gl * mS / cm ** 2

    # ------------------------------------------------------------------ #
    # M2 组件接口（与 MultiCompartmentNeuron 同签名）
    # ------------------------------------------------------------------ #
    def label_of(self, segment: str, compartment: Optional[int] = None) -> int:
        """任意位点 → 0（唯一隔室；ChemicalSynapse node3/soma 语义均落位）。"""
        return 0

    def soma_area_cm2(self) -> float:
        """点面积（cm²）。"""
        return self.area_cm2

    def density_to_nA(self, density_uA_cm2: float, compartment: int = 0) -> float:
        """电流密度（µA/cm²）→ 该神经元总电流（nA）。"""
        return float(density_uA_cm2) * 1e-6 * self.area_cm2 * 1e9


class TwoCompartmentNeuron:
    """双隔室 HH（soma + 轴突末梢 node3；缩放扫描中间档，§3.2 方案②）。

    2 神经元 NeuronGroup：i=0 soma（gNa=120）、i=1 node3（gNa=300）；
    轴向耦合 I_ax = G_AX·(v_peer − v)（amp，按隔室面积折密度）。
    接口与 PointNeuron/MultiCompartmentNeuron 一致（M2 组件复用）。
    """

    def __init__(
        self,
        name: str = "tc",
        dt_ms: Optional[float] = None,
        method: Optional[str] = None,
        refractory_ms: float = 2.0,
        threshold_mv: float = -20.0,
        area_cm2: float = SOMA_AREA_CM2,
        node_area_cm2: float = _NODE_AREA_CM2,
        extra_eqs: str = "",
        extra_im_terms: str = "",
        stim_var: str = "stim",
    ):
        self.name = name
        #: 双隔室档定稿：dt=0.05ms / exponential_euler（L7：小隔室高 gNa 使膜方程更
        #: stiff，rk4 在 dt=0.05 静息自发尖峰后发散；exponential_euler 无条件稳定）
        self.dt_ms = 0.05 if dt_ms is None else float(dt_ms)
        self.method = "exponential_euler" if method is None else method
        self.refractory_ms = refractory_ms
        self.threshold_mv = threshold_mv
        self.area_cm2 = float(area_cm2)
        self.node_area_cm2 = float(node_area_cm2)
        self.extra_eqs = extra_eqs
        self.extra_im_terms = extra_im_terms
        self.stim_var = stim_var
        self.neuron = None
        self._built = False

    def build(self):
        from brian2 import (NeuronGroup, cm, defaultclock, linked_var, meter,
                            ms, mS, mV, siemens, start_scope, uF)
        from neural_exploration.src.brian_env import configure_brian2

        configure_brian2()
        start_scope()
        defaultclock.dt = self.dt_ms * ms
        eqs = _two_comp_eqs(self.extra_im_terms, self.extra_eqs, self.stim_var)
        ns = {"Cm": _CM * uF / cm ** 2, "G_AX": _AXIAL_G_S * siemens}
        g = NeuronGroup(2, eqs, method=self.method,
                        threshold=f"v > {self.threshold_mv}*mV",
                        refractory=self.refractory_ms * ms,
                        name=self.name, namespace=ns)
        from neural_exploration.src.ion_channels import steady_state_gates
        m0, h0, n0 = steady_state_gates(_V0)
        g.v = _V0 * mV
        g.m, g.h, g.n = m0, h0, n0
        # 逐神经元面积（m²）→ 状态变量赋值（Brian2 不接受 namespace 数组，实测）
        g.AREA = np.array([self.area_cm2, self.node_area_cm2]) * 1e-4 * meter ** 2
        g.gNa = np.array([_GNA, _NODE_GNA]) * mS / cm ** 2
        g.gK = _GK * mS / cm ** 2
        g.gL = _GL * mS / cm ** 2
        # 对侧隔室电压（索引交换；实测 linked_var 支持 index 数组，L7）
        g.v_peer = linked_var(g, "v", index=np.array([1, 0]))
        self.neuron = g
        self._built = True
        return self

    def label_of(self, segment: str, compartment: Optional[int] = None) -> int:
        """soma→0、node3→1、其余→0（ChemicalSynapse node3→soma 语义保持）。"""
        seg = str(segment).lower()
        if seg == "node3":
            return 1
        return 0

    def soma_area_cm2(self) -> float:
        return self.area_cm2

    def density_to_nA(self, density_uA_cm2: float, compartment: int = 0) -> float:
        area = self.area_cm2 if compartment == 0 else self.node_area_cm2
        return float(density_uA_cm2) * 1e-6 * area * 1e9
