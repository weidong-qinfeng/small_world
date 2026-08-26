"""M6 可塑性组件：成对 STDP + 三因子调质门控可塑性（清单 P1/P4 机制层）。

《生物仿真M6实施清单》§2.1/§2.2（src/plasticity.py）：
  - `StdpSynapse`：成对 STDP（pre→post 因果 LTP / post→pre 反因果 LTD）——
    Δw = A₊·exp(−Δt/τ₊)（Δt>0）/ −A₋·exp(Δt/τ₋)（Δt<0）；权重有界 [0, w_max]；
    tau_plus/tau_minus/a_plus/a_minus 参数化（data/m6_learning_params.csv stdp 段定稿）；
  - `ThreeFactorSynapse`：三因子调质门控（P4 联想学习机制）——
    资格迹 elig(t)（pre/post 激活痕迹）+ 调质信号 M(t)（多巴胺/血清素池，
    来自 B1a neuromod.ModulatorPool）→ dw/dt = η·M(t)·elig(t)；
  - 网络级 STDP 装配接口（§0 预注册 #1：G1 门后启用；默认仅组件级/子图，
    **不做 3638 化学突触全图 STDP**——避免全同步饱和 + 编译预算失控）。

Brian2 2.6.0 实现要点（M3 L11 实测约束 + M6 规格）：
  1. 事件代码限制：min/max、if/else 分支在 on_pre/on_post 中不可用 →
     权重钳位一律 `clip(w, 0.0, WMAX)`（namespace 常量，M3 L11 已验证）；
  2. 成对时序规则用 pre/post 指数痕迹实现（on_pre 以 post_trace 做 LTD、
     on_post 以 pre_trace 做 LTP）——隔离脉冲对下逐位复现理论曲线
     （实测单对 |ΔW − 理论| ~ 1e-17，见 tools/validate_p1_stdp.py）；
  3. 确定性铁律：p=1/n=1（无 rand() 语句），同参数重跑逐位一致；
  4. 三因子 w 的 ODE 漂移（dw/dt = η·M(t)·elig）为规格公式（§2.2）；
     事件期 clip 提供 [0, w_max] 硬边界，漂移段有界性由 P4 协议 η 校准保证
     （预注册 Δw ≤ w_max − w0）；
  5. 命名纪律：Brian2 保留 `_pre`/`_post` 后缀（不能用作变量名），
     `e` 为内置常量（不能用作变量名）→ 痕迹变量 pre_trace/post_trace、
     资格迹 elig。

权重语义（清单 L6）：w 为无量纲电导缩放因子，w0=1.0 锚 = 连接组 g_max_ns
占位（ampa 5.0nS 语义，M5 先验；见 stdp 段 w0_semantic_ns）；ΔW 以相对量
Δw/w0 报告（与文献归一化曲线可比）。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEFAULT_STDP_PARAMS_CSV = os.path.join(
    ROOT, "neural_exploration", "data", "m6_learning_params.csv")


# --------------------------------------------------------------------- #
# 参数
# --------------------------------------------------------------------- #
@dataclass
class StdpParams:
    """STDP / 三因子参数（m6_learning_params.csv stdp 段定稿；预注册 §2.1/§2.2）。

    预注册窗（清单 §2.1/§2.2，CSV 注释同步）：
      τ₊=τ₋=20ms（文献典型值，Bi & Poo 1998 量级；[10,40] 可调窗）
      A₊=0.01·w0（[0.005, 0.02]）；A₋/A₊=0.9（[0.5, 1.0] 平衡防发散）
      w_max=2.0·w0；η（[1e-4, 1e-2]）；τ_e（[100, 500]ms）
    """

    tau_plus_ms: float = 20.0
    tau_minus_ms: float = 20.0
    a_plus: float = 0.01          # ×w0
    a_minus: float = 0.009        # = 0.9·a_plus（预注册比值）
    w0: float = 1.0               # 权重锚（无量纲；语义 = w0_semantic_ns）
    w_max: float = 2.0            # 2.0·w0 预注册
    w0_semantic_ns: float = 5.0   # ampa 权重语义锚（M5 先验；信息字段，不进入方程）
    g_max_ns: float = 0.3         # 释放基准电导（w=1 时；P1 协议亚阈值，发放时刻由刺激确定）
    tau_syn_ms: float = 3.0       # ampa 受体衰减（M2 定稿）
    e_rev_mv: float = 0.0         # ampa 反转电位
    p_release: float = 1.0        # 确定性 p=1/n=1
    n_vesicles: int = 1
    eta: float = 1e-4             # 三因子学习率（预注册 [1e-4, 1e-2]）
    tau_e_ms: float = 200.0       # 资格迹（预注册 [100, 500]）

    @property
    def a_minus_over_a_plus(self) -> float:
        return self.a_minus / self.a_plus

    @property
    def post_var(self) -> str:
        """post 组电导变量（ampa 语义，P1 规格）。"""
        return "g_ampa"


def _parse_value(s: str):
    s = s.strip()
    if s in ("", "None"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def load_stdp_params(csv_path: Optional[str] = None) -> StdpParams:
    """读 data/m6_learning_params.csv stdp 段 → StdpParams（CSV 唯一定稿源）。

    文件缺失 → 预注册默认值（并在调用处提示；验证脚本先写母版 CSV 再回读，
    保证 CSV 闭环）。a_minus_over_a_plus 行派生 a_minus（若显式 a_minus 行
    缺失）；比值一致性校验（不一致 → ValueError，CSV 是唯一定稿源）。
    """
    path = csv_path or DEFAULT_STDP_PARAMS_CSV
    if not os.path.exists(path):
        return StdpParams()
    rows: dict = {}
    with open(path, newline="", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 3 or parts[0] != "stdp":
                continue
            rows[parts[1]] = parts[2]
    if not rows:
        return StdpParams()
    kw = {}
    for field in ("tau_plus_ms", "tau_minus_ms", "a_plus", "a_minus", "w0",
                  "w_max", "w0_semantic_ns", "g_max_ns", "tau_syn_ms",
                  "e_rev_mv", "p_release", "n_vesicles", "eta", "tau_e_ms"):
        if field in rows:
            kw[field] = _parse_value(rows[field])
    p = StdpParams(**kw)
    if "a_minus_over_a_plus" in rows and "a_minus" not in rows:
        p.a_minus = float(rows["a_minus_over_a_plus"]) * p.a_plus
    elif "a_minus_over_a_plus" in rows:
        ratio = float(rows["a_minus_over_a_plus"])
        if abs(p.a_minus / p.a_plus - ratio) > 1e-9:
            raise ValueError(
                f"stdp CSV 比值不一致：a_minus/a_plus={p.a_minus/p.a_plus:.4f} "
                f"vs a_minus_over_a_plus={ratio}")
    return p


def write_stdp_params_csv(path: str, params: Optional[StdpParams] = None) -> None:
    """写 stdp 段到 m6_learning_params.csv（验证脚本生成母版；后续步骤追加
    habituation/associative 等段；已有文件的其他段保留，stdp 段覆盖为定稿值）。"""
    p = params or StdpParams()
    lines = [
        "# M6 学习参数母版（B1b 生成 stdp 段；habituation/associative 段由后续步骤追加）",
        "# 单位：ms/mV/nS；w 无量纲（ΔW 以相对量 Δw/w0 报告，L6 约定）",
        "# STDP 预注册（清单 §2.1/§2.2）：τ=20ms 量级[10,40]；A₊=0.01·w0[0.005,0.02]；",
        "#   A₋/A₊=0.9[0.5,1.0]；w_max=2.0·w0；η[1e-4,1e-2]；τ_e[100,500]ms；p=1/n=1",
        "section,param,value,unit,note",
        f"stdp,tau_plus_ms,{p.tau_plus_ms},ms,文献典型值(Bi&Poo 1998);预注册窗[10,40]",
        f"stdp,tau_minus_ms,{p.tau_minus_ms},ms,同τ₊",
        f"stdp,a_plus,{p.a_plus},,0.01·w0;预注册[0.005,0.02]",
        f"stdp,a_minus,{p.a_minus},,0.9·A₊(预注册比值)",
        f"stdp,a_minus_over_a_plus,{round(p.a_minus_over_a_plus, 6)},,LTD/LTP平衡;预注册[0.5,1.0]",
        f"stdp,w0,{p.w0},,权重锚(无量纲;ΔW 用相对量)",
        f"stdp,w_max,{p.w_max},,2.0·w0 预注册",
        f"stdp,w0_semantic_ns,{p.w0_semantic_ns},nS,ampa 权重语义锚(M5 先验)",
        f"stdp,g_max_ns,{p.g_max_ns},nS,P1 协议释放基准(亚阈值;发放时刻由刺激确定)",
        f"stdp,tau_syn_ms,{p.tau_syn_ms},ms,ampa 受体衰减(M2 定稿)",
        f"stdp,e_rev_mv,{p.e_rev_mv},mV,ampa 反转电位",
        f"stdp,p_release,{p.p_release},,确定性 p=1/n=1",
        f"stdp,n_vesicles,{p.n_vesicles},,确定性 n=1",
        f"stdp,eta,{p.eta},,三因子学习率;预注册[1e-4,1e-2]",
        f"stdp,tau_e_ms,{p.tau_e_ms},ms,资格迹;预注册[100,500]",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    existing = []
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line.startswith(("stdp,", "# stdp")) or line.strip() == "":
                    continue
                existing.append(line)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
        if existing:
            f.write("\n".join(existing) + "\n")


# --------------------------------------------------------------------- #
# StdpSynapse：成对 STDP
# --------------------------------------------------------------------- #
class StdpSynapse:
    """成对 STDP 突触（Brian2 Synapses 包装；构造接口同 M2 ChemicalSynapse）。

    规格（清单 §2.1）：
      on_pre  : w = clip(w − A₋·post_trace, 0, w_max)；pre_trace += 1；释放 g_post += gmax·w
      on_post : w = clip(w + A₊·pre_trace, 0, w_max)；post_trace += 1
      dpre_trace/dt = −pre_trace/τ₊；dpost_trace/dt = −post_trace/τ₋
    隔离脉冲对下 Δw = A₊·exp(−Δt/τ₊)（Δt>0，LTP）/ −A₋·exp(Δt/τ₋)（Δt<0，LTD），
    与文献理论曲线逐位一致（tools/validate_p1_stdp.py 实测 |ΔW−理论| ~ 1e-17）。

    enable_stdp=False → 纯传递（无痕迹/无权重更新），供 P3/P4 机制消融
    （"关 STDP"对照组）。
    """

    def __init__(
        self,
        pre_neuron,
        post_neuron,
        params: Optional[StdpParams] = None,
        pre_site: str = "node3",
        post_site: str = "soma",
        name: str = "stdp_syn",
        enable_stdp: bool = True,
    ):
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
        self.params = params or load_stdp_params()
        self.pre_site = pre_site
        self.post_site = post_site
        self.name = name
        self.enable_stdp = enable_stdp
        self.synapses = None
        self._built = False

    def _g_density(self) -> float:
        """点电导 nS → 密度 S/m²（按 post 胞体面积，M2 同款换算）。"""
        area = self.post_neuron.soma_area_cm2() * 1e-4  # cm² → m²
        return self.params.g_max_ns * 1e-9 / area

    def build(self):
        from brian2 import Synapses, meter, ms, siemens

        p = self.params
        post_var = p.post_var
        ns = {
            "TAU_PLUS": p.tau_plus_ms * ms,
            "TAU_MINUS": p.tau_minus_ms * ms,
            "A_PLUS": p.a_plus,
            "A_MINUS": p.a_minus,
            "WMAX": p.w_max,
            "GMAXD": self._g_density() * siemens / meter ** 2,
        }
        if self.enable_stdp:
            model = (
                f"w : 1\n"
                f"dpre_trace/dt = -pre_trace/TAU_PLUS : 1 (clock-driven)\n"
                f"dpost_trace/dt = -post_trace/TAU_MINUS : 1 (clock-driven)\n")
            on_pre = (
                "w = clip(w - A_MINUS*post_trace, 0.0, WMAX)\n"
                "pre_trace = pre_trace + 1\n"
                f"{post_var}_post = {post_var}_post + GMAXD*w")
            on_post = (
                "w = clip(w + A_PLUS*pre_trace, 0.0, WMAX)\n"
                "post_trace = post_trace + 1")
        else:
            model = f"w : 1\n"
            on_pre = f"{post_var}_post = {post_var}_post + GMAXD*w"
            on_post = ""
        kw = dict(model=model, on_pre=on_pre, namespace=ns, name=self.name)
        if self.enable_stdp:
            kw["on_post"] = on_post
        syn = Synapses(self.pre_neuron.neuron, self.post_neuron.neuron, **kw)
        i = self.pre_neuron.label_of(self.pre_site)
        j = self.post_neuron.label_of(self.post_site)
        syn.connect(i=i, j=j)
        syn.w = p.w0
        if self.enable_stdp:
            syn.pre_trace = 0.0
            syn.post_trace = 0.0
        self.synapses = syn
        self._built = True
        return self

    def weights(self) -> np.ndarray:
        """当前权重数组（w，无量纲缩放因子）。"""
        return np.array(self.synapses.w)


# --------------------------------------------------------------------- #
# ThreeFactorSynapse：三因子调质门控可塑性（P4 联想学习机制）
# --------------------------------------------------------------------- #
class ThreeFactorSynapse:
    """三因子调质门控可塑性（清单 §2.2，P4 联想学习机制）。

    delig/dt = −elig/τ_e : 1 (clock-driven)     资格迹（pre/post 激活痕迹）
    dw/dt   = η·M(t)·elig/second : 1           M(t)=调质浓度信号（TimedArray，
                                               来自 B1a neuromod.ModulatorPool）
    on_pre : elig += 1；w = clip(w, 0, w_max)；释放
    on_post: elig += 1
    边界：事件期 clip 硬边界；ODE 漂移段有界性由 P4 协议 η 校准保证
    （预注册：训练试次 Δw ≤ w_max − w0）。
    作用域（§0 #10）：联想学习链 ASE/AWC→AIY/AIB 子集开启，其余突触不开启。
    """

    def __init__(
        self,
        pre_neuron,
        post_neuron,
        params: Optional[StdpParams] = None,
        pre_site: str = "node3",
        post_site: str = "soma",
        name: str = "tf_syn",
    ):
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
        self.params = params or load_stdp_params()
        self.pre_site = pre_site
        self.post_site = post_site
        self.name = name
        self.synapses = None
        self._built = False

    def _g_density(self) -> float:
        area = self.post_neuron.soma_area_cm2() * 1e-4
        return self.params.g_max_ns * 1e-9 / area

    def build(self, modulation_timedarray):
        """构建；modulation_timedarray = 调质浓度 M(t) 的 Brian2 TimedArray
        （1-D，值 ∈ [0,1]，时间步 = 网络 dt；命名不限，经 namespace 解析）。"""
        from brian2 import Synapses, meter, ms, second, siemens

        p = self.params
        post_var = p.post_var
        model = (
            f"delig/dt = -elig/TAU_E : 1 (clock-driven)\n"
            f"dw/dt = ETA*M_t(t)*elig/second : 1\n")
        on_pre = (
            "elig = elig + 1\n"
            "w = clip(w, 0.0, WMAX)\n"
            f"{post_var}_post = {post_var}_post + GMAXD*w")
        on_post = "elig = elig + 1"
        ns = {
            "TAU_E": p.tau_e_ms * ms,
            "ETA": p.eta,
            "WMAX": p.w_max,
            "GMAXD": self._g_density() * siemens / meter ** 2,
            "M_t": modulation_timedarray,
        }
        syn = Synapses(self.pre_neuron.neuron, self.post_neuron.neuron,
                       model=model, on_pre=on_pre, on_post=on_post,
                       name=self.name, namespace=ns)
        i = self.pre_neuron.label_of(self.pre_site)
        j = self.post_neuron.label_of(self.post_site)
        syn.connect(i=i, j=j)
        syn.w = p.w0
        syn.elig = 0.0
        self.synapses = syn
        self._built = True
        return self

    def weights(self) -> np.ndarray:
        return np.array(self.synapses.w)


# --------------------------------------------------------------------- #
# 网络级 STDP 装配接口（§0 预注册 #1：G1 门后启用，默认不开启）
# --------------------------------------------------------------------- #
def stdp_network_connections(csv_path: Optional[str] = None
                             ) -> List[Tuple[str, str, str]]:
    """读 data/m6_learning_params.csv stdp_connections 段 →
    [(pre_role, post_role, syn_type), ...]。

    哪些连接开启 STDP 由 CSV 定稿（§0 预注册 #1）；段缺失 → []（默认不启用
    任何网络级连接）。候选行（G1 门后定稿）：习惯化=触觉 PLM/ALM→命令中间
    神经元；联想学习=ASE/AWC→AIY/AIB 趋化链。
    """
    path = csv_path or DEFAULT_STDP_PARAMS_CSV
    if not os.path.exists(path):
        return []
    out: List[Tuple[str, str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 4 and parts[0] == "stdp_connections":
                out.append((parts[1], parts[2], parts[3]))
    return out


def attach_subgraph_stdp(
    circuit,
    connections: Optional[Sequence[Tuple[str, str, str]]] = None,
    params: Optional[StdpParams] = None,
    enabled: bool = False,
    name_prefix: str = "m6_stdp",
) -> Optional[List[object]]:
    """worm 网络级 STDP 装配接口（清单 §0 预注册 #1；G1 门后启用，默认不开启）。

    预注册语义：
      (a) STDP 默认**协议窗内开启**（学习协议 run 窗口内开、基线/复核协议关），
          且**限定突触子集**（学习相关通路：习惯化=触觉 PLM/ALM→命令中间神经元；
          联想学习=ASE/AWC→AIY/AIB 趋化链）——**不做 3638 化学突触全图 STDP**
          （避免全同步饱和 + 编译预算失控）；
      (b) G1 门先行：夹带缓解前不开启网络级学习协议。

    enabled=False（默认）：no-op，返回 None——基线协议不触发 STDP 代码路径，
    M5 复核协议（无学习窗）数值不变（回归保护）。

    enabled=True：在 circuit.group（批量组装单组）上为 connections 子集构建
    STDP Synapses（逐连接 gmax/delay，同 GroupedWormCircuit.chem 批量语义；
    gmax/delay 优先取 circuit.sub.chem 连接组事实）。调用方需在装配后自
    circuit.chem_synapses 移除对应连接（learning.py 在 G1 后以子图替换方式
    组装；未改 M5 冻结文件）。

    connections: [(pre_role, post_role, syn_type), ...]；缺省 →
    stdp_network_connections()（CSV stdp_connections 段）。
    """
    if not enabled:
        return None
    params = params or load_stdp_params()
    conns = list(connections) if connections is not None \
        else stdp_network_connections()
    if not conns:
        raise ValueError("attach_subgraph_stdp 需要非空 connections"
                         "（或 stdp_connections CSV 行）；G1 门后由 CSV 定稿")
    from brian2 import Synapses, meter, ms, siemens

    group = circuit.group
    role_index = circuit.role_index
    lookup = {}
    for r in getattr(getattr(circuit, "sub", None), "chem", []):
        lookup[(r.pre, r.post, r.syn_type)] = r
    syns = []
    for stype in sorted({c[2] for c in conns}):
        rows = [c for c in conns if c[2] == stype]
        pre_i, post_i, gmax, delays = [], [], [], []
        for pre_role, post_role, st in rows:
            if pre_role not in role_index or post_role not in role_index:
                raise ValueError(
                    f"STDP 连接 {pre_role}→{post_role} 不在 circuit 神经元集合"
                    f"（{stype}）")
            row = lookup.get((pre_role, post_role, st))
            g_ns = row.g_ns if row is not None else params.g_max_ns
            delay = row.delay_ms if row is not None else 0.5
            pre_i.append(role_index[pre_role])
            post_i.append(role_index[post_role])
            gmax.append(g_ns * 1e-9 / (1.257e-5 * 1e-4))   # nS → S/m²（点面积）
            delays.append(delay)
        model = (
            "gmax : siemens/meter**2\n"
            "w : 1\n"
            "dpre_trace/dt = -pre_trace/TAU_PLUS : 1 (clock-driven)\n"
            "dpost_trace/dt = -post_trace/TAU_MINUS : 1 (clock-driven)\n")
        on_pre = (
            "w = clip(w - A_MINUS*post_trace, 0.0, WMAX)\n"
            "pre_trace = pre_trace + 1\n"
            f"g_{stype}_post = g_{stype}_post + gmax*w")
        on_post = (
            "w = clip(w + A_PLUS*pre_trace, 0.0, WMAX)\n"
            "post_trace = post_trace + 1")
        syn = Synapses(group, group, model=model, on_pre=on_pre,
                       on_post=on_post,
                       name=f"{name_prefix}_{stype}",
                       namespace={
                           "TAU_PLUS": params.tau_plus_ms * ms,
                           "TAU_MINUS": params.tau_minus_ms * ms,
                           "A_PLUS": params.a_plus,
                           "A_MINUS": params.a_minus,
                           "WMAX": params.w_max,
                       })
        syn.connect(i=np.array(pre_i, dtype=np.int32),
                    j=np.array(post_i, dtype=np.int32))
        syn.gmax = np.array(gmax) * siemens / meter ** 2
        syn.w = params.w0
        syn.pre_trace = 0.0
        syn.post_trace = 0.0
        syn.delay = np.array(delays) * ms
        syns.append(syn)
    return syns
