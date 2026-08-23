"""M2 正式实现：化学突触 + 电突触（缝隙连接）+ 短期可塑性（STP）。

清单《生物仿真M2实施清单》§4.1：`src/synapse_model.py` 提供：
  - `SynapseParams`：突触参数（data/m2_synapse_params.csv 驱动）；
  - `ChemicalSynapse`：量子释放（二项/伯努利）+ 受体动力学
    （AMPA 快兴奋 / GABA_A 快抑制 / NMDA 慢兴奋 + Mg²⁺ 电压依赖去阻断）；
  - `GapJunction`：缝隙连接（无延迟、双向、I = g·(V_pre - V_post)）；
  - STP（Tsodyks–Markram）：u（利用度）与 x（资源）双变量，
    由 ChemicalSynapse 的 `stp` 参数开启。

Brian2 2.6.0 实测机制要点（详见 docs/m2_env_notes.md）：
  1. 突触前触发：`Synapses` 以 M1 SpatialNeuron 为源/目标，逐隔室发放事件；
     node3（轴突末梢）跨阈值 → `on_pre` 回调（清单 L1）。
  2. 突触后电导：在 post 神经元方程里定义 `dg/dt = -g/tau : siemens/meter**2`，
     `on_pre` 用 `g_post += ...`（`_post` 后缀直接写 post 组变量）。
     Im 按 Brian2 **内向正**约定追加 `g*(E-v)`（M1 L4 踩坑的延续）。
  3. 量子释放随机性：`on_pre` 中 `int(rand() < p)`，**每条语句至多一次 rand()**
     （Brian2 2.6 不支持同一语句多次 rand：会报“more than one call of rand”）。
     p=1 时生成无 rand() 的确定性语句，避免 abstract-code 警告。
  4. NMDA Mg²⁺ 阻断：释放增量乘 B(V_post) = 1/(1+[Mg²⁺]·exp(-0.062·V/1mV)/3.57)
     （Jahr & Stevens 1990 标准式，与 NEURON 参考解同一方程，见 §3 清单）。
  5. 缝隙连接：`I_gap : amp (point current)` 挂在两侧神经元；
     Synapses 模型内 `I_couple = g*(v_pre - v_post)`，用 `(summed)` 变量
     `I_gap_post`/`I_gap_pre` 每步写回两侧 point current（双向、连续）。
  6. 单位：电导密度 S/m²（Im 为 amp/m²）；CSV 用生理点电导 nS，
     构建时按胞体面积换算（g_nS·1e-9/面积）——与 NEURON ExpSyn 的点电导同物理量。
"""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEFAULT_PARAMS_CSV = os.path.join(ROOT, "neural_exploration", "data", "m2_synapse_params.csv")

# 经典 Jahr & Stevens (1990) Mg²⁺ 阻断常数（V 单位 mV）
MG_BLOCK_A = 0.062     # /mV
MG_BLOCK_B = 3.57      # mM


@dataclass
class SynapseParams:
    """一种突触的参数（CSV 一行；单位见文档）。

    g_max_ns: 点电导 nS（化学突触 = 单量子电导；缝隙连接 = g_gap）
    tau_ms:   受体衰减时间常数（缝隙连接不用）
    e_rev_mv: 反转电位 mV
    p_release: 单囊泡释放概率（0..1）
    n_vesicles: 可用囊泡数（量子释放二项模型 k ~ Binomial(n, p)）
    mg_mm:    NMDA Mg²⁺ 浓度（mM；仅 nmda 用）
    u0:       STP 基准利用度 U
    tau_fac_ms / tau_rec_ms: STP 易化/恢复时间常数（0 表示该 STP 项关闭）
    """

    synapse_type: str
    g_max_ns: float
    tau_ms: float = 3.0
    e_rev_mv: float = 0.0
    p_release: float = 1.0
    n_vesicles: int = 1
    mg_mm: float = 1.2
    u0: float = 0.0
    tau_fac_ms: float = 0.0
    tau_rec_ms: float = 0.0

    @property
    def stp_enabled(self) -> bool:
        return self.tau_fac_ms > 0 and self.tau_rec_ms > 0

    @property
    def post_var(self) -> str:
        """post 组上的电导变量名。"""
        return {"ampa": "g_ampa", "gaba": "g_gaba", "nmda": "g_nmda"}[self.synapse_type]

    def as_dict(self) -> dict:
        return dict(
            synapse_type=self.synapse_type, g_max_ns=self.g_max_ns, tau_ms=self.tau_ms,
            e_rev_mv=self.e_rev_mv, p_release=self.p_release, n_vesicles=self.n_vesicles,
            mg_mm=self.mg_mm, u0=self.u0, tau_fac_ms=self.tau_fac_ms, tau_rec_ms=self.tau_rec_ms,
        )


def load_synapse_params(csv_path: Optional[str] = None) -> Dict[str, SynapseParams]:
    """读入 data/m2_synapse_params.csv → {synapse_type: SynapseParams}。"""
    path = csv_path or DEFAULT_PARAMS_CSV
    if not os.path.exists(path):
        raise FileNotFoundError(f"突触参数 CSV 不存在：{path}")
    out: Dict[str, SynapseParams] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(row for row in f if not row.strip().startswith("#"))
        for r in reader:
            p = SynapseParams(
                synapse_type=r["synapse_type"].strip(),
                g_max_ns=float(r["g_max_ns"]),
                tau_ms=float(r["tau_ms"]),
                e_rev_mv=float(r["e_rev_mv"]),
                p_release=float(r["p_release"]),
                n_vesicles=int(float(r["n_vesicles"])),
                mg_mm=float(r["mg_mm"]),
                u0=float(r["u0"]),
                tau_fac_ms=float(r["tau_fac_ms"]),
                tau_rec_ms=float(r["tau_rec_ms"]),
            )
            out[p.synapse_type] = p
    return out


# --------------------------------------------------------------------- #
# 方程片段（追加到 post 神经元；默认空 = M1 原行为）
# --------------------------------------------------------------------- #
def chemical_post_eqs(params: Dict[str, SynapseParams]) -> str:
    """post 组突触电导 ODE（S/m²，指数衰减）。"""
    lines = []
    for key in ("ampa", "gaba", "nmda"):
        if key not in params:
            continue
        p = params[key]
        var = p.post_var
        lines.append(f"d{var}/dt = -{var}/({p.tau_ms}*ms) : siemens/meter**2")
    return "\n".join(lines)


def chemical_im_terms(params: Dict[str, SynapseParams]) -> str:
    """Im 追加项（内向正约定：g*(E-v)，与 M1 通道项同号）。"""
    terms = []
    for key in ("ampa", "gaba", "nmda"):
        if key not in params:
            continue
        p = params[key]
        terms.append(f" + {p.post_var}*({p.e_rev_mv}*mV-v)")
    return "".join(terms)


GAP_POST_EQ = "I_gap : amp (point current)"


def gap_im_term(_params=None) -> str:
    """缝隙连接无需改 Im（point current 自动注入）。"""
    return ""


# --------------------------------------------------------------------- #
# 突触前触发（on_pre 语句生成）
# --------------------------------------------------------------------- #
# 运行期参数标识符（经 Synapses namespace 传入；值不进入生成代码串，
# 保证不同参数组合共享同一编译产物——M2 实测：格式化进字符串的数值
# 每变一次就触发 80–120s 重编译，见 m2_env_notes §L2）
NS_GMAX = "GMAXD"     # 释放电导密度 S/m²
NS_PREL = "PREL"      # 单囊泡释放概率
NS_MG = "MGMM"        # NMDA [Mg²⁺] mM
NS_U0 = "U0"          # STP 基准利用度
NS_TAUFAC = "TAUFAC"  # STP 易化时间常数 ms
NS_TAUREC = "TAUREC"  # STP 恢复时间常数 ms


def _release_statements(post_var: str, p: float, n_ves: int) -> List[str]:
    """量子释放：k ~ Binomial(n_ves, p) 个量子，各以单条语句累加。

    p=1 → 确定性语句（无 rand，无 abstract-code 警告）。
    """
    if p >= 1.0:
        return [f"{post_var}_post = {post_var}_post + {NS_GMAX}*{n_ves}"]
    stmts = []
    for _ in range(n_ves):
        stmts.append(f"{post_var}_post = {post_var}_post + "
                     f"{NS_GMAX}*int(rand() < {NS_PREL})")
    return stmts


class ChemicalSynapse:
    """化学突触：pre 神经元某隔室（默认 node3）→ post 神经元某隔室（默认 soma）。

    on_pre 语义（按 params 组合）：
      1. STP（若开启）：u 先易化 → 释放 ∝ u·x → x 耗竭；
      2. NMDA：释放增量再乘 Mg²⁺ 阻断因子 B(v_post)（电压依赖去阻断）；
      3. 量子释放：二项（n_vesicles × p_release）。
    """

    def __init__(
        self,
        pre_neuron,
        post_neuron,
        params: SynapseParams,
        pre_site: str = "node3",
        post_site: str = "soma",
        name: str = "chem_syn",
    ):
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
        self.params = params
        self.pre_site = pre_site
        self.post_site = post_site
        self.name = name
        self.synapses = None
        self._built = False

    def _g_density(self) -> float:
        """点电导 nS → 密度 S/m²（按 post 胞体面积）。"""
        area = self.post_neuron.soma_area_cm2() * 1e-4  # cm² → m²
        return self.params.g_max_ns * 1e-9 / area

    def _build_on_pre(self) -> str:
        p = self.params
        var = p.post_var
        stmts: List[str] = []
        if p.stp_enabled:
            # Tsodyks–Markram：先易化、再按 u·x 释放、后耗竭
            stmts.append(f"u = u + {NS_U0}*(1-u)")
            rel = f"u*x"
            if p.n_vesicles > 1:
                rel = f"({rel})*{p.n_vesicles}"
            if p.p_release < 1.0:
                rel = f"{rel}*int(rand() < {NS_PREL})"
            if p.synapse_type == "nmda":
                rel = f"{rel}*(1/(1+{NS_MG}*exp(-{MG_BLOCK_A}*v_post/mV)/{MG_BLOCK_B}))"
            stmts.append(f"{var}_post = {var}_post + {NS_GMAX}*{rel}")
            stmts.append(f"x = x - u*x")
        else:
            rel = _release_statements(var, p.p_release, p.n_vesicles)
            if p.synapse_type == "nmda":
                b = (f"*(1/(1+{NS_MG}*exp(-{MG_BLOCK_A}*v_post/mV)/{MG_BLOCK_B}))")
                rel = [s + b for s in rel]
            stmts.extend(rel)
        return "\n".join(stmts)

    def _namespace(self) -> dict:
        from brian2 import meter, siemens

        p = self.params
        ns = {
            NS_GMAX: self._g_density() * siemens / meter ** 2,
            NS_PREL: p.p_release,
            NS_MG: p.mg_mm,
            NS_U0: p.u0,
            NS_TAUFAC: p.tau_fac_ms,
            NS_TAUREC: p.tau_rec_ms,
        }
        return ns

    def build(self):
        """创建 Brian2 Synapses 并连接（pre_site → post_site 单突触）。"""
        from brian2 import Synapses

        p = self.params
        model = ""
        if p.stp_enabled:
            model = f"""
du/dt = ({NS_U0}-u)/({NS_TAUFAC}*ms) : 1 (clock-driven)
dx/dt = (1-x)/({NS_TAUREC}*ms) : 1 (clock-driven)
"""
        on_pre = self._build_on_pre()
        syn = Synapses(self.pre_neuron.neuron, self.post_neuron.neuron,
                       model=model, on_pre=on_pre, name=self.name,
                       namespace=self._namespace())
        i = self.pre_neuron.label_of(self.pre_site)
        j = self.post_neuron.label_of(self.post_site)
        syn.connect(i=i, j=j)
        if p.stp_enabled:
            syn.u = p.u0
            syn.x = 1.0
        self.synapses = syn
        self._built = True
        return self


class GapJunction:
    """电突触：pre/post 各一隔室（默认 soma）间的双向欧姆耦合。

    I_gap = g_gap·(V_pre - V_post)，I_gap_post = -I_gap_pre。
    Brian2 实现：Synapses 模型内逐时间步计算 `I_couple`（可引用 v_pre/v_post），
    用 `(summed)` 变量 `I_gap_post`/`I_gap_pre` 写回两侧神经元的
    `I_gap : amp (point current)` 参数（自动注入膜方程）。
    """

    def __init__(
        self,
        pre_neuron,
        post_neuron,
        g_gap_ns: float,
        pre_site: str = "soma",
        post_site: str = "soma",
        name: str = "gap",
    ):
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
        self.g_gap_ns = g_gap_ns
        self.pre_site = pre_site
        self.post_site = post_site
        self.name = name
        self.synapses = None
        self._built = False

    def build(self):
        from brian2 import Synapses

        g = self.g_gap_ns * 1e-9  # nS → S
        model = f"""
I_couple = {g!r}*siemens*(v_pre - v_post) : amp
I_gap_post = I_couple : amp (summed)
I_gap_pre = -I_couple : amp (summed)
"""
        syn = Synapses(self.pre_neuron.neuron, self.post_neuron.neuron,
                       model=model, name=self.name)
        i = self.pre_neuron.label_of(self.pre_site)
        j = self.post_neuron.label_of(self.post_site)
        syn.connect(i=i, j=j)
        self.synapses = syn
        self._built = True
        return self
