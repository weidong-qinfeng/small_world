"""M1 正式实现：HH Na/K/漏 离子通道（可配置密度，多隔室支持）。

M0 占位版只 re-export 了 tools/hh_spec.py 的参数；本模块在 M1 提供：

1. 标准 HH 1952 门控方程串（Brian2 语法，与 hh_spec.py 速率函数逐项一致，
   单位已无量纲化——沿用 M0 验证通过的写法）；
2. `HhChannels` 数据类：区段级通道密度（gNa/gK/gL），由 CSV 驱动；
3. 数值安全说明：Brian2 2.6.0 SpatialNeuron 的 Im 采用**内向正**约定
   （dv/dt = (Im - Iaxial)/Cm，见 docs 的 `Im=gL*(EL-v)` 写法），
   本模块方程与 tools/hh_spec.py 的 current() 相差一个整体符号，二者物理等价。

单位约定（清单 §1 L4）：电导 mS/cm²，电容 µF/cm²，电位 mV，时间 ms，
轴向电阻 Ra 单位 Ω·cm（传入 SpatialNeuron 的 Ri 参数）。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.tools.hh_spec import EK as _EK  # noqa: F401  # 与 M0 参数同源（EL/ENa 亦同）
from neural_exploration.tools.hh_spec import EL as _EL  # noqa: F401
from neural_exploration.tools.hh_spec import ENA as _ENA  # noqa: F401
from neural_exploration.tools.hh_spec import GK as _GK  # noqa: F401
from neural_exploration.tools.hh_spec import GL as _GL  # noqa: F401
from neural_exploration.tools.hh_spec import GNA as _GNA  # noqa: F401

#: 标准 HH 门控速率函数（v 单位 mV，速率单位 1/ms）——与 tools/hh_spec.py 相同
ALPHA_M = "(0.1/mV)*(v+40*mV)/(1-exp(-(v+40*mV)/(10*mV)))/ms"
BETA_M = "4*exp(-(v+65*mV)/(18*mV))/ms"
ALPHA_H = "0.07*exp(-(v+65*mV)/(20*mV))/ms"
BETA_H = "1/(1+exp(-(v+35*mV)/(10*mV)))/ms"
ALPHA_N = "(0.01/mV)*(v+55*mV)/(1-exp(-(v+55*mV)/(10*mV)))/ms"
BETA_N = "0.125*exp(-(v+65*mV)/(80*mV))/ms"

#: 逐隔室通道电导参数（每隔室一个值；Brian2 2.6.0 中不加标志即为逐隔室）
CHANNEL_PARAMS = """
gNa : siemens/meter**2
gK : siemens/meter**2
gL : siemens/meter**2
"""

#: 反转电位（全树共享；与 M0 hh_spec 参数一致）
REVERSAL_POTENTIALS = """
EL = {EL}*mV : volt (shared)
ENa = {ENA}*mV : volt (shared)
EK = {EK}*mV : volt (shared)
"""

#: HH 门控动力学（每隔室独立积分）
GATING_DYNAMICS = """
dm/dt = alpham*(1-m)-betam*m : 1
dh/dt = alphah*(1-h)-betah*h : 1
dn/dt = alphan*(1-n)-betan*n : 1
alpham = {ALPHA_M} : Hz
betam = {BETA_M} : Hz
alphah = {ALPHA_H} : Hz
betah = {BETA_H} : Hz
alphan = {ALPHA_N} : Hz
betan = {BETA_N} : Hz
"""

#: 注入电流（point current，每隔室、单位 amp；NEURON IClamp 同单位，便于参考解对齐）
POINT_CURRENT = """
I = stim(t, i) : amp (point current)
"""


@dataclass
class HhChannels:
    """一个区段的通道密度配置（mS/cm²）。"""

    gna: float
    gk: float
    gl: float


def hh_equations(
    with_point_current: bool = True,
    el: float = _EL,
    ena: float = _ENA,
    ek: float = _EK,
) -> str:
    """组装 Brian2 方程串（SpatialNeuron 用）。

    膜电流 Im 采用 Brian2 文档约定的**内向正**写法（Im = g·(E-v)），
    与工具 current() 的“外向正”差一个整体符号；物理与数值等价。
    """
    eqs = [
        "Im = gL*(EL-v) + gNa*m**3*h*(ENa-v) + gK*n**4*(EK-v) : amp/meter**2",
        CHANNEL_PARAMS,
        REVERSAL_POTENTIALS.format(EL=el, ENA=ena, EK=ek),
        GATING_DYNAMICS.format(
            ALPHA_M=ALPHA_M, BETA_M=BETA_M, ALPHA_H=ALPHA_H,
            BETA_H=BETA_H, ALPHA_N=ALPHA_N, BETA_N=BETA_N,
        ),
    ]
    if with_point_current:
        eqs.append(POINT_CURRENT)
    return "\n".join(eqs)


def steady_state_gates(v_mv: float):
    """v 处门控稳态 (m, h, n)，与 tools/hh_spec.steady_state 相同（供初始条件）。"""
    from neural_exploration.tools.hh_spec import steady_state
    return steady_state(v_mv)
