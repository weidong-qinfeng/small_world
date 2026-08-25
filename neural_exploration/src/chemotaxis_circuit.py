"""M4 正式实现：`ChemotaxisCircuit` —— 嗅觉/味觉趋化局部回路（~20 神经元，CSV 驱动）。

清单《生物仿真M4实施清单》§2.1/§4（拓扑与实现要点）：

    食物梯度 C(x,y)
      → ASE 感觉对（时间差分编码，电流注入）
          ASEL（ON：浓度↑ → 去极化）──AMPA──> AIYL/AIYR（前进促进：抑制转向、保持直行）
          ASER（OFF：浓度↓ → 去极化）──AMPA──> AIBL/AIBR（转向促进：促发 pirouette）
      → AIYL/AIYR ──GABA──> RIAL/RIAR（前进促进压制转向执行——互斥）
      → AIBL/AIBR ──AMPA──> RIAL/RIAR（转向促进驱动转向执行）
      → RIAL ──AMPA──> SMDDL/SMDVL ──muscle──> C_left
      → RIAR ──AMPA──> SMDDR/SMDVR ──muscle──> C_right
      → AVBL/AVBR ──AMPA──> VB/DB ──muscle──> C_fwd（正弦爬行推进）
      → 虚拟身体（chemotaxis_body.py）→ 闭环（chemotaxis_loop.py）

链定义唯一来源：`data/m4_chemotaxis_params.csv`（B1a 定稿；本模块运行期读取，
缺失时 wait_for_csv 轮询 ≤20 分钟——M3 同款机制）。CSV 列（清单 §2.5）：
    role, neuron_class, synapse_from, synapse_to, synapse_type, g_max_ns,
    delay_ms, tonic_uA_cm2, value, note
  - 神经元行：role=<20 角色名>；
  - 突触行：synapse_from + synapse_to + synapse_type(ampa|gaba|nmda|muscle)
    + g_max_ns(化学=链点电导 nS；muscle 行=收缩权重 w) + delay_ms；
    muscle 行 synapse_to = muscle_fwd / muscle_left / muscle_right；
  - 转导行：role=transduction, neuron_class=<g_ON|g_OFF|tau_win|ase_site|noise>, value=<值>；
  - 环境行：role=env, neuron_class=<arena_L|sigma|C_max|C_bg|food_x|food_y|boundary>；
  - 身体行：role=body, neuron_class=<v_fwd0|omega_max|dt_b|v_osc>；
  - 协议行：role=protocol, neuron_class=<n_trials|t_total_ms|start_x|start_y|
    ci_radius|ci_band_lo|ci_band_hi|start_jitter>；
  - 全局行：role=param, neuron_class=<t_total_ms|dt_ms|method|seed|muscle_tau_ms|muscle_cap>。

确定性铁律（清单 §4.2 #3）：**默认全链 p_release=1、n_vesicles=1**（无随机，
同参数重跑逐位一致）——CSV 中若有 p/n 列也被忽略；量子释放噪声由调用方
显式开启（`set_quantum_noise`，informational）。

M3 交接复用（清单 §1 L1）：`MultiCompartmentNeuron`（多隔室 HH，逐神经元 stim_var）、
`ChemicalSynapse`（node3→soma）、`chemical_post_eqs`/`chemical_im_terms`、
`STIM_WINDOW_MS` 固定形状 + TimedArray 显式命名 + namespace 传参（编译缓存纪律）、
store/restore + 重播种多试次机制、`wait_for_csv`。

**三通道肌肉（清单 §1 L1 方案① 定稿）**：`Muscle3` 在 chemotaxis 模块内新建
（照 `Muscle` 模式参数化通道：C_fwd / C_left / C_right）——冻结的 muscle.py 不改。

**ASE 转导电流（清单 §2.2）**：s(t) = (C(t) − C(t−τ_win))/τ_win（[ΔC/ms]）；
I_ASEL = g_ON·max(s,0)、I_ASER = g_OFF·max(−s,0)（µA/cm² 密度，按注入位点
面积换算 nA）。max/min 在 numpy 侧完成（刺激数组数值），事件代码只用 clip() +
namespace 常量（M3 L11 纪律）。

**闭环 epoch 迭代（清单 §4.2 #1）**：`ChemoSession` 提供 epoch 级运行——
每 epoch ΔT：写入固定形状 TimedArray（`STIM_WINDOW_MS` 下限 + 显式命名 +
pad 零，epoch 间仅数值变化 → 编译缓存命中），net.run(ΔT)，读三通道肌肉收缩。
试次间 store/restore + 重播种（M3 L12 语义）。

确定性：同参数重跑逐位一致（`ChemotaxisResult.__eq__` 数值比较）。
"""

from __future__ import annotations

import math
import os
import sys
import time as _time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.src.chemotaxis_env import (  # noqa: E402
    ChemotaxisEnv,
    EnvSpec,
    step_protocol,
)
from neural_exploration.src.chemotaxis_body import BodySpec  # noqa: E402
from neural_exploration.src.neuron_model import MultiCompartmentNeuron  # noqa: E402
from neural_exploration.src.neuron_pair import STIM_WINDOW_MS  # noqa: E402  # M2 L6 固定形状约定
from neural_exploration.src.synapse_model import (  # noqa: E402
    ChemicalSynapse,
    SynapseParams,
    chemical_im_terms,
    chemical_post_eqs,
    load_synapse_params,
)

DEFAULT_CHEMO_PARAMS_CSV = os.path.join(ROOT, "neural_exploration", "data",
                                        "m4_chemotaxis_params.csv")

#: 链角色花名册（清单 §2.1 规格；CSV 定稿为准，loader 要求齐全）
EXPECTED_ROLES = (
    "ASEL", "ASER",            # 感觉（ON/OFF 对）
    "AIYL", "AIYR",            # 中间-前进促进
    "AIBL", "AIBR",            # 中间-转向促进
    "RIAL", "RIAR",            # 中间-转向执行
    "AVBL", "AVBR",            # 命令（前进）
    "SMDDL", "SMDDR", "SMDVL", "SMDVR",   # 运动-头（左右转向竞争）
    "VB", "DB",                # 运动-身体（前进）
    "AVAL", "AVAR",            # 运动-反向（竞争对照/备用）
    "RMED", "RMEV",            # 缝隙-调节（默认关闭）
)

#: 三通道肌肉（§1 L1 方案①）
MUSCLE_CHANNELS = ("fwd", "left", "right")


# --------------------------------------------------------------------- #
# 链规格（CSV 驱动）
# --------------------------------------------------------------------- #
@dataclass
class ChemoSynapseSpec:
    """一条链连接（化学突触或肌肉驱动）。"""

    synapse_from: str
    synapse_to: str
    synapse_type: str            # ampa | gaba | nmda | muscle
    g_max_ns: Optional[float] = None   # None → m2_synapse_params.csv 默认（化学）
    delay_ms: float = 0.1
    note: str = ""

    @property
    def is_muscle(self) -> bool:
        return self.synapse_type == "muscle"

    @property
    def muscle_channel(self) -> str:
        """muscle 行目标通道：muscle_fwd→'fwd'，muscle_left→'left'，muscle_right→'right'。"""
        to = self.synapse_to.lower()
        m = {"muscle_fwd": "fwd", "muscle_left": "left", "muscle_right": "right"}
        if to not in m:
            raise ValueError(f"muscle 突触目标需为 muscle_fwd/left/right：{self.synapse_to}")
        return m[to]


@dataclass
class ChemoTransductionSpec:
    """ASE 转导参数（清单 §2.2；CSV transduction 行）。"""

    g_on: float = 2000.0        # ON 细胞增益（µA/cm² 每 ΔC/ms；s>0）
    g_off: float = 2000.0       # OFF 细胞增益（µA/cm² 每 ΔC/ms；s<0）
    tau_win_ms: float = 50.0    # 时间差分滑窗 τ_win（ms）
    ase_site: str = "soma"      # 注入位点（ASE 胞体；CSV 定稿）
    noise: float = 0.0          # 感受器噪声 σ_noise（informational；默认关）


@dataclass
class ChemoMechASpec:
    """机制 A（pirouette 转向事件；清单 §2.4 落地修订，主 agent 裁决 2026-08-23）。

    转向方向 = 试次种子确定性伪随机（numpy RNG seed=trial_seed：每试次固定 → 可复现，
    试次间随机——真实虫 pirouette 方向随机）；s 调制转向频率：s < −θ_pir 且
    ASER→AIB→RIA→SMDD 激活（闭环 epoch 内 SMDD 发放）→ 转向事件（持续 T_pir，
    ω=±ω_pir）；s > 0 → ASEL→AIY 压制（无触发）——偏置来自直行/转向时长不对称
    （Pierce-Shimomura 1999）。CSV role=mechanism_a 行为唯一定稿源。
    """

    theta_pir: float = 4.0e-6   # s 转向触发阈值 [ΔC/ms]（s < −θ_pir 才触发）
    omega_pir: float = 1.0      # 转向事件角速度 [rad/s]（ω = ±ω_pir）
    t_pir_ms: float = 1571.0    # 转向事件持续时长 [ms]（π/2 / ω_pir = 90° 转角）
    enabled: bool = True        # 开关（False = 退回对称电路 ω≡0，仅诊断用）


@dataclass
class ChemoProtocolSpec:
    """行为试次协议（CSV protocol 行；清单 §2.4）。"""

    n_trials: int = 20
    t_total_ms: float = 20000.0   # 试次时长（行为仿真时间）
    start_x: float = 5.0          # 起点 = 皿中心（默认）
    start_y: float = 5.0
    start_jitter: float = 0.0     # 试次间起点扰动（伪随机；0 = 同起点）
    ci_radius: float = 1.5        # 吸引圈半径（informational 圈式指标）
    ci_band_lo: float = 0.25      # 生物带容差窗（Ward 1973；Pierce-Shimomura 1999）
    ci_band_hi: float = 0.75
    p1_baseline_ms: float = 40.0  # P1 阶跃协议：静止基线期（HH 静息漂移）
    p1_seg_ms: float = 50.0       # P1 阶跃协议：每段时长（上升/静止/下降）
    p1_step_dc: float = 0.5       # P1 阶跃协议：ΔC 幅度（±）


@dataclass
class ChemotaxisParams:
    """整条趋化回路参数（唯一定稿源 = CSV）。"""

    roles: List[str] = field(default_factory=list)
    synapses: List[ChemoSynapseSpec] = field(default_factory=list)
    tonic_uA_cm2: Dict[str, float] = field(default_factory=dict)  # role → 张力注入
    transduction: ChemoTransductionSpec = field(default_factory=ChemoTransductionSpec)
    mech_a: ChemoMechASpec = field(default_factory=ChemoMechASpec)
    env: EnvSpec = field(default_factory=EnvSpec)
    body: BodySpec = field(default_factory=BodySpec)
    protocol: ChemoProtocolSpec = field(default_factory=ChemoProtocolSpec)
    muscle_tau_ms: float = 20.0
    muscle_cap: Optional[float] = 1.0
    t_total_ms: float = 20000.0
    dt_ms: float = 0.01
    method: str = "rk4"
    seed: int = 0
    csv_path: str = ""


def _parse_site(site: str) -> Tuple[str, int]:
    """'dend2#1' → ('dend2', 1)；无 '#' → (site, 0)。"""
    if "#" in site:
        seg, _, sub = site.partition("#")
        return seg, int(sub)
    return site, 0


# --------------------------------------------------------------------- #
# CSV 加载（运行期读取；唯一定稿源）
# --------------------------------------------------------------------- #
def load_chemotaxis_params(csv_path: Optional[str] = None) -> ChemotaxisParams:
    """读入 data/m4_chemotaxis_params.csv → ChemotaxisParams。

    列容差：神经元行/突触行/键值行由非空列判别；缺列用默认值（.get 容错，
    便于 B1a 定稿微调列）。键值行按 neuron_class（缺省 role）归一化后应用。
    """
    path = csv_path or DEFAULT_CHEMO_PARAMS_CSV
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"M4 链参数 CSV 不存在：{path}\n"
            "ChemotaxisCircuit 的参数唯一定稿源是 data/m4_chemotaxis_params.csv（B1a 节点产出）。\n"
            "请先确认该文件已生成（列：role/neuron_class/synapse_from/synapse_to/"
            "synapse_type/g_max_ns/delay_ms/tonic_uA_cm2/value/note），或运行 "
            "tests/neuro/test_chemotaxis_smoke.py 等待其生成。"
        )
    import csv

    p = ChemotaxisParams(csv_path=path)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(row for row in f if not row.strip().startswith("#"))
        for r in reader:
            role = (r.get("role") or "").strip()
            frm = (r.get("synapse_from") or "").strip()
            to = (r.get("synapse_to") or "").strip()
            stype = (r.get("synapse_type") or "").strip()
            note = (r.get("note") or "").strip()

            def _f(col, default=None):
                v = (r.get(col) or "").strip()
                return v if v != "" else default

            if frm and to:
                g = _f("g_max_ns")
                p.synapses.append(ChemoSynapseSpec(
                    synapse_from=frm, synapse_to=to, synapse_type=stype,
                    g_max_ns=float(g) if g is not None else None,
                    delay_ms=float(_f("delay_ms", "0.1")), note=note,
                ))
            elif role and role.lower() in ("transduction", "mechanism_a", "env", "body",
                                           "protocol", "param", "global"):
                key = (_f("neuron_class") or role).strip().lower()
                # 值优先取 value 列；空则取 note 列（M3 惯例：值写在 note）；
                # 再退到 to 列 / 表头外扩展字段（B1a 惯例同 M3）。
                val = _f("value")
                if not val:
                    val = _f("note")
                if not val:
                    val = to
                if not val:
                    for x in (r.get(None) or ()):
                        if x is not None and str(x).strip():
                            val = str(x).strip()
                            break
                _apply_kv(p, key, val)
            elif role:
                p.roles.append(role)
                t = _f("tonic_uA_cm2")
                if t is not None:
                    p.tonic_uA_cm2[role] = float(t)
            else:
                raise ValueError(f"m4_chemotaxis_params.csv 行无法识别（role/synapse 均空）：{r}")

    _validate_params(p)
    return p


def _apply_kv(p: ChemotaxisParams, key: str, val: str):
    """键值行：归一化 key → 覆盖对应 spec 字段（值尽量转 float）。"""
    def _num(v):
        try:
            return float(v)
        except ValueError:
            return v

    k = key.strip().lower()
    v = _num(val)
    tr, env, body, proto = p.transduction, p.env, p.body, p.protocol

    # 转导
    if k in ("g_on", "g_on_uacm2", "g_on_uA_cm2".lower(), "gon"):
        tr.g_on = float(v)
    elif k in ("g_off", "g_off_uacm2", "goff"):
        tr.g_off = float(v)
    elif k in ("tau_win", "tau_win_ms", "tauwin"):
        tr.tau_win_ms = float(v)
    elif k in ("ase_site", "ase_site_name"):
        tr.ase_site = str(v)
    elif k in ("noise", "noise_sigma"):
        tr.noise = float(v)
    # 机制 A（pirouette，清单 §2.4 落地修订）
    elif k in ("theta_pir", "theta_pir_thresh"):
        p.mech_a.theta_pir = float(v)
    elif k in ("omega_pir", "pirouette_omega"):
        p.mech_a.omega_pir = float(v)
    elif k in ("t_pir_ms", "t_pir", "pirouette_dur_ms"):
        p.mech_a.t_pir_ms = float(v)
    elif k in ("mech_a_enabled", "mecha_enabled", "enabled"):
        p.mech_a.enabled = bool(float(v) > 0)
    # 环境
    elif k in ("arena_l", "arena_len", "arena_size", "l"):
        env.arena_L = float(v)
    elif k == "sigma":
        env.sigma = float(v)
    elif k in ("c_max", "cmax", "concentration_max"):
        env.c_max = float(v)
    elif k in ("c_bg", "cbg", "concentration_bg"):
        env.c_bg = float(v)
    elif k in ("food_x", "food_x_position"):
        env.food_x = float(v)
    elif k in ("food_y", "food_y_position"):
        env.food_y = float(v)
    elif k in ("boundary", "boundary_mode"):
        env.boundary = str(v)
    # 身体
    elif k in ("v_fwd0", "v_fwd", "speed0"):
        body.v_fwd0 = float(v)
    elif k in ("omega_max", "w_max", "turn_rate"):
        body.omega_max = float(v)
    elif k in ("dt_b", "dt_behavior", "dtb", "epoch_ms"):
        body.dt_b = float(v)
    elif k == "v_osc":
        body.v_osc = float(v)
    # 协议
    elif k in ("n_trials", "ntrials"):
        proto.n_trials = int(float(v))
    elif k == "t_total_ms":
        proto.t_total_ms = float(v)
        p.t_total_ms = float(v)          # 全局同值（沿 M3 惯例）
    elif k in ("start_x", "start_x_position"):
        proto.start_x = float(v)
    elif k in ("start_y", "start_y_position"):
        proto.start_y = float(v)
    elif k in ("start_jitter", "start_noise"):
        proto.start_jitter = float(v)
    elif k in ("ci_radius", "attract_radius"):
        proto.ci_radius = float(v)
    elif k in ("ci_band_lo", "ci_lo"):
        proto.ci_band_lo = float(v)
    elif k in ("ci_band_hi", "ci_hi"):
        proto.ci_band_hi = float(v)
    elif k == "p1_baseline_ms":
        proto.p1_baseline_ms = float(v)
    elif k == "p1_seg_ms":
        proto.p1_seg_ms = float(v)
    elif k == "p1_step_dc":
        proto.p1_step_dc = float(v)
    # 全局
    elif k == "dt_ms":
        p.dt_ms = float(v)
    elif k == "method":
        p.method = str(v)
    elif k == "seed":
        p.seed = int(float(v))
    elif k == "muscle_tau_ms":
        p.muscle_tau_ms = float(v)
    elif k == "muscle_cap":
        p.muscle_cap = None if str(v).lower() in ("none", "null", "") else float(v)
    else:
        # 未知键：宽容忽略（B1a 扩展字段自由使用）
        pass


def _validate_params(p: ChemotaxisParams):
    """拓扑/极性校验（清单 §2.5 验收）：角色齐全、端点存在、muscle 目标合法。"""
    missing = set(EXPECTED_ROLES) - set(p.roles)
    if missing:
        raise ValueError(
            f"m4_chemotaxis_params.csv 缺少角色 {sorted(missing)}\n"
            f"（当前角色：{sorted(p.roles)}；清单 §2.1 规格 20 角色）")
    known = set(p.roles) | {"muscle_fwd", "muscle_left", "muscle_right"}
    for s in p.synapses:
        if s.synapse_from not in p.roles:
            raise ValueError(f"突触起点 {s.synapse_from} 不在角色列表 {p.roles}")
        if s.is_muscle:
            if s.synapse_to.lower() not in ("muscle_fwd", "muscle_left", "muscle_right"):
                raise ValueError(f"muscle 突触目标需为 muscle_fwd/left/right：{s.synapse_to}")
        elif s.synapse_to not in p.roles:
            raise ValueError(f"突触终点 {s.synapse_to} 不在角色列表 {p.roles}")
        if s.synapse_type not in ("ampa", "gaba", "nmda", "muscle"):
            raise ValueError(f"未知突触类型：{s.synapse_type}")
    if not any(s.is_muscle for s in p.synapses):
        raise ValueError("m4_chemotaxis_params.csv 缺少肌肉驱动行（→muscle_fwd/left/right）")
    if not any(s.is_muscle and s.muscle_channel == "fwd" for s in p.synapses):
        raise ValueError("m4_chemotaxis_params.csv 缺少 muscle_fwd 驱动（前进推进必需）")
    if not any(s.is_muscle and s.muscle_channel in ("left", "right")
               for s in p.synapses):
        raise ValueError("m4_chemotaxis_params.csv 缺少左右转向肌肉驱动")


def wait_for_csv(csv_path: Optional[str] = None, timeout_s: float = 1200.0,
                 interval_s: float = 30.0) -> str:
    """轮询等待 CSV 生成（B1a 节点可能稍后产出；测试用）。

    已存在则立即返回；超时抛 FileNotFoundError（默认 ≤20 分钟）。
    """
    path = csv_path or DEFAULT_CHEMO_PARAMS_CSV
    t0 = _time.time()
    while not os.path.exists(path):
        if _time.time() - t0 > timeout_s:
            raise FileNotFoundError(
                f"等待 {timeout_s:.0f}s 后 m4_chemotaxis_params.csv 仍未生成：{path}")
        _time.sleep(interval_s)
    return path


# --------------------------------------------------------------------- #
# 运行结果
# --------------------------------------------------------------------- #
@dataclass
class ChemotaxisResult:
    """一次运行的输出（P1–P5 判定脚本的输入）。"""

    t_ms: np.ndarray
    v_mv: Dict[str, np.ndarray]            # 标签 → V(t)（mV），如 'asel_soma'
    spike_times_ms: Dict[str, np.ndarray]  # 标签 → 发放时刻（ms），如 'asel_node3'
    c_fwd: np.ndarray
    c_left: np.ndarray
    c_right: np.ndarray
    meta: Dict = field(default_factory=dict)
    # 闭环轨迹（开环运行无）
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    theta: Optional[np.ndarray] = None

    def spikes(self, role: str, site: str = "node3") -> np.ndarray:
        """某角色某位点的发放时刻（无则空数组）。role 大小写不敏感。"""
        lab = f"{role.lower()}_{site}"
        return self.spike_times_ms.get(lab, np.array([]))

    @property
    def ci(self) -> Optional[float]:
        """闭环试次 CI（开环运行无）。"""
        return self.meta.get("ci")

    @property
    def has_trajectory(self) -> bool:
        return self.x is not None and self.y is not None

    def __eq__(self, other) -> bool:
        """确定性验证：数值逐位比较（含闭环轨迹）。"""
        if not isinstance(other, ChemotaxisResult):
            return NotImplemented
        base = (
            np.array_equal(self.t_ms, other.t_ms)
            and self.v_mv.keys() == other.v_mv.keys()
            and all(np.array_equal(self.v_mv[k], other.v_mv[k]) for k in self.v_mv)
            and self.spike_times_ms.keys() == other.spike_times_ms.keys()
            and all(np.array_equal(self.spike_times_ms[k], other.spike_times_ms[k])
                    for k in self.spike_times_ms)
            and np.array_equal(self.c_fwd, other.c_fwd)
            and np.array_equal(self.c_left, other.c_left)
            and np.array_equal(self.c_right, other.c_right)
        )
        for a, b in ((self.x, other.x), (self.y, other.y), (self.theta, other.theta)):
            if a is None or b is None:
                if not (a is None and b is None):
                    return False
            elif not np.array_equal(a, b):
                return False
        return base


# --------------------------------------------------------------------- #
# 三通道肌肉（清单 §1 L1 方案①：Muscle 模式参数化，新建于本模块）
# --------------------------------------------------------------------- #
class Muscle3:
    """Brian2 虚拟肌肉：三通道收缩积分器（C_fwd / C_left / C_right）。

    与冻结的 `Muscle`（muscle.py）同模式：每通道一个 NeuronGroup（单变量单
    神经元），`dC/dt = −C/TAU`，运动神经元发放 → on_pre 增量 w（饱和用
    `clip()` + namespace 常量——事件代码分支纪律，M3 L11）。
    """

    def __init__(self, tau_ms: float = 20.0, cap: Optional[float] = 1.0,
                 channels: Sequence[str] = MUSCLE_CHANNELS,
                 name: str = "muscle3"):
        self.tau_ms = tau_ms
        self.cap = cap
        self.channels = tuple(channels)
        self.name = name
        self._groups: Dict[str, object] = {}
        self._drivers: Dict[str, list] = {ch: [] for ch in self.channels}
        self._built = False

    def build(self):
        from brian2 import NeuronGroup, ms

        tau = self.tau_ms
        for ch in self.channels:
            var = f"c_{ch}"
            g = NeuronGroup(
                1, f"d{var}/dt = -{var}/({tau}*ms) : 1",
                method="euler", name=f"{self.name}_{ch}",
            )
            setattr(g, var, 0.0)
            self._groups[ch] = g
        self._built = True
        return self

    def connect_driver(self, pre_neuron, channel: str, weight: float,
                       name: str = "mus_drv"):
        """运动神经元 → 肌肉通道的 Synapses（on_pre 增量 w；触发位点 node3）。"""
        from brian2 import Synapses, ms

        if channel not in self.channels:
            raise ValueError(f"通道需为 {self.channels}：{channel}")
        var = f"c_{channel}"
        g = self._groups[channel]
        if self.cap is not None:
            on_pre = f"{var}_post = clip({var}_post + WMUSC, 0.0, CAP)"
            ns = {"WMUSC": weight, "CAP": self.cap}
        else:
            on_pre = f"{var}_post += WMUSC"
            ns = {"WMUSC": weight}
        syn = Synapses(pre_neuron.neuron, g, on_pre=on_pre,
                       name=f"{name}_{channel}", namespace=ns)
        i = pre_neuron.label_of("node3")
        syn.connect(i=i, j=0)
        syn.delay = 0.1 * ms   # 与化学突触同量级（发放→收缩传导延迟）
        self._drivers[channel].append(syn)
        return syn

    @property
    def groups(self) -> list:
        return list(self._groups.values())

    @property
    def drivers(self) -> list:
        return [s for lst in self._drivers.values() for s in lst]

    def get(self, channel: str):
        """返回单通道 NeuronGroup。"""
        return self._groups[channel]

    def read(self) -> Dict[str, float]:
        """当前三通道收缩值（epoch 末读取 → 运动学积分输入）。"""
        return {ch: float(getattr(self._groups[ch], f"c_{ch}")[0])
                for ch in self.channels}

    def monitor(self, dt_ms: float, name: str = "mon_muscle3"):
        """记录三通道收缩的 StateMonitor（需加入 Network）。"""
        from brian2 import StateMonitor, ms

        out = []
        for ch in self.channels:
            var = f"c_{ch}"
            out.append(StateMonitor(self._groups[ch], var, record=True,
                                    dt=dt_ms * ms, name=f"{name}_{ch}"))
        return tuple(out)


# --------------------------------------------------------------------- #
# ChemotaxisCircuit：~20 神经元链组装 + ASE 转导 + 三通道肌肉
# --------------------------------------------------------------------- #
class ChemotaxisCircuit:
    """嗅觉/味觉趋化回路（~20 神经元 + 三通道肌肉，CSV 驱动）。

    用法：
        circ = ChemotaxisCircuit()                    # 读 data/m4_chemotaxis_params.csv
        r = circ.run()                                # 开环 P1 阶跃协议（默认）
        r.spikes("ASEL", "node3"), r.spikes("ASER", "node3")
        circ.set_protocol(c_trace=..., dt_protocol_ms=...)   # 自定义开环协议
        circ.remove_synapse("AIYL", "RIAL")           # P5 消融
        loop = ChemotaxisLoop(circ)                   # 闭环（chemotaxis_loop.py）
    """

    def __init__(
        self,
        csv_path: Optional[str] = None,
        dt_ms: Optional[float] = None,
        method: Optional[str] = None,
        t_total_ms: Optional[float] = None,
        seed: Optional[int] = None,
        name_prefix: str = "chem",
    ):
        self.params: ChemotaxisParams = load_chemotaxis_params(csv_path)
        # 构造参数（显式传入）覆盖 CSV；None 默认 = 以 CSV 为准（M3 L13）
        if dt_ms is not None:
            self.params.dt_ms = dt_ms
        if method is not None:
            self.params.method = method
        if t_total_ms is not None:
            self.params.t_total_ms = t_total_ms
            self.params.protocol.t_total_ms = t_total_ms
        if seed is not None:
            self.params.seed = seed
        self.seed = seed if seed is not None else self.params.seed
        self.name_prefix = name_prefix
        self._m2 = load_synapse_params()          # ampa/gaba/nmda 基础参数（M2 定稿）
        self._protocol: Optional[Tuple[np.ndarray, Optional[np.ndarray], float]] = None
        self._release: Optional[Tuple[float, int]] = None
        self._release_overrides: Dict[Tuple[str, str], Tuple[float, int]] = {}
        self._syn_overrides: Dict[Tuple[str, str], Dict[str, float]] = {}
        self._removed: set = set()
        self._ase_noise = False

        self.neurons: Dict[str, MultiCompartmentNeuron] = {}
        self.chemicals: List[ChemicalSynapse] = []
        self.muscle3: Optional[Muscle3] = None
        self._built = False

    # ------------------------------------------------------------------ #
    # 协议覆盖钩子（P1/P5/校准）
    # ------------------------------------------------------------------ #
    def set_protocol(self, c_trace: Optional[Sequence[float]] = None,
                     s_trace: Optional[Sequence[float]] = None,
                     dt_protocol_ms: Optional[float] = None) -> "ChemotaxisCircuit":
        """覆盖开环协议（P1/验证用）。

        - c_trace : C(t) 轨迹（dt_protocol_ms 网格）→ 内部按 τ_win 算 s(t)；
        - s_trace : 直接给 s(t)（ΔC/ms）轨迹（跳过差分）。
        两者至少给一个；缺省（不调用）时 run() 用默认上升→静止→下降阶跃
        协议（清单 §2.2，时间窗见 `protocol_info`）。
        """
        if c_trace is None and s_trace is None:
            raise ValueError("set_protocol 需给 c_trace 或 s_trace 之一")
        dt = float(dt_protocol_ms or self.params.dt_ms)
        self._protocol = (
            np.asarray(c_trace, dtype=float) if c_trace is not None else None,
            np.asarray(s_trace, dtype=float) if s_trace is not None else None,
            dt,
        )
        return self

    def clear_protocol(self) -> "ChemotaxisCircuit":
        """回到默认 P1 阶跃协议。"""
        self._protocol = None
        return self

    def protocol_info(self) -> dict:
        """P1 阶跃协议的时间窗（测试/绘图用；CSV protocol p1_* 定稿）。"""
        proto = self.params.protocol
        rise_start = proto.p1_baseline_ms
        rise_end = rise_start + proto.p1_seg_ms
        fall_start = rise_end + proto.p1_seg_ms
        fall_end = fall_start + proto.p1_seg_ms
        return dict(
            c_base=0.2, delta_c=proto.p1_step_dc,
            t_baseline_ms=proto.p1_baseline_ms, t_up_ms=proto.p1_seg_ms,
            t_hold_ms=proto.p1_seg_ms, t_down_ms=proto.p1_seg_ms,
            rise_start_ms=rise_start, rise_end_ms=rise_end,
            fall_start_ms=fall_start, fall_end_ms=fall_end,
            t_total_ms=rise_start + 4 * proto.p1_seg_ms,
        )

    def set_ase_noise(self, enabled: bool = True) -> "ChemotaxisCircuit":
        """开启 ASE 感受器噪声（informational；默认关——确定性铁律不受影响）。"""
        self._ase_noise = bool(enabled)
        return self

    def set_ase_gains(self, g_on: Optional[float] = None,
                      g_off: Optional[float] = None) -> "ChemotaxisCircuit":
        """覆盖转导增益 g_ON/g_OFF（B1c 校准钩子）。"""
        if g_on is not None:
            self.params.transduction.g_on = float(g_on)
        if g_off is not None:
            self.params.transduction.g_off = float(g_off)
        return self

    def set_mechanism_a(self, theta_pir: Optional[float] = None,
                        omega_pir: Optional[float] = None,
                        t_pir_ms: Optional[float] = None,
                        enabled: Optional[bool] = None) -> "ChemotaxisCircuit":
        """覆盖机制 A 参数（B1c 校准钩子；None = 以 CSV 为准）。"""
        if theta_pir is not None:
            self.params.mech_a.theta_pir = float(theta_pir)
        if omega_pir is not None:
            self.params.mech_a.omega_pir = float(omega_pir)
        if t_pir_ms is not None:
            self.params.mech_a.t_pir_ms = float(t_pir_ms)
        if enabled is not None:
            self.params.mech_a.enabled = bool(enabled)
        return self

    def set_quantum_noise(self, p_release: float, n_vesicles: int,
                          synapse_from: Optional[str] = None,
                          synapse_to: Optional[str] = None) -> "ChemotaxisCircuit":
        """开启量子释放噪声（informational；默认确定性 p=1/n=1）。"""
        if not 0.0 < p_release <= 1.0 or int(n_vesicles) < 1:
            raise ValueError(f"非法量子释放参数：p={p_release}, n={n_vesicles}")
        if synapse_from or synapse_to:
            if not (synapse_from and synapse_to):
                raise ValueError("逐连接噪声需同时给 synapse_from 与 synapse_to")
            self._release_overrides[(synapse_from.upper(), synapse_to.upper())] = (
                float(p_release), int(n_vesicles))
        else:
            self._release = (float(p_release), int(n_vesicles))
        return self

    def set_deterministic(self) -> "ChemotaxisCircuit":
        """回到确定性模式（p=1/n=1，全部连接；ASE 噪声关闭）。"""
        self._release = None
        self._release_overrides.clear()
        self._ase_noise = False
        return self

    def override_synapse(self, synapse_from: str, synapse_to: str,
                         g_max_ns: Optional[float] = None,
                         delay_ms: Optional[float] = None) -> "ChemotaxisCircuit":
        """调参钩子：覆盖 CSV 中某连接的 g_max_ns / delay_ms（B2 参数扫描）。"""
        key = (synapse_from.upper(), synapse_to.upper())
        over = dict(self._syn_overrides.get(key, {}))
        if g_max_ns is not None:
            over["g_max_ns"] = float(g_max_ns)
        if delay_ms is not None:
            over["delay_ms"] = float(delay_ms)
        self._syn_overrides[key] = over
        return self

    def remove_synapse(self, synapse_from: str, synapse_to: str) -> "ChemotaxisCircuit":
        """消融：删除某连接（P5 反证路径）。"""
        self._removed.add((synapse_from.upper(), synapse_to.upper()))
        return self

    # ------------------------------------------------------------------ #
    # 构建
    # ------------------------------------------------------------------ #
    def _synapse_specs(self) -> List[ChemoSynapseSpec]:
        """按覆盖/消融过滤后的连接列表（不修改 params，便于重跑）。"""
        out = []
        for s in self.params.synapses:
            key = (s.synapse_from.upper(), s.synapse_to.upper())
            if key in self._removed:
                continue
            over = self._syn_overrides.get(key, {})
            if not over:
                out.append(s)
                continue
            out.append(ChemoSynapseSpec(
                synapse_from=s.synapse_from, synapse_to=s.synapse_to,
                synapse_type=s.synapse_type,
                g_max_ns=over.get("g_max_ns", s.g_max_ns),
                delay_ms=over.get("delay_ms", s.delay_ms), note=s.note))
        return out

    def _release_for(self, spec: ChemoSynapseSpec) -> Tuple[float, int]:
        """某化学连接的 (p_release, n_vesicles)：确定性默认，除非显式开启噪声。"""
        key = (spec.synapse_from.upper(), spec.synapse_to.upper())
        if key in self._release_overrides:
            return self._release_overrides[key]
        if self._release is not None:
            return self._release
        return (1.0, 1)  # 确定性铁律

    def build(self):
        """重建整条链（每次会话前自动调用）：神经元 + 化学突触 + 三通道肌肉。"""
        from neural_exploration.src.brian_env import configure_brian2
        from brian2 import ms, start_scope

        p = self.params
        configure_brian2()
        start_scope()

        # 1) 每个角色需要的突触后方程片段（按入边类型聚合）
        post_types: Dict[str, set] = {r: set() for r in p.roles}
        for s in p.synapses:
            if not s.is_muscle:
                post_types[s.synapse_to].add(s.synapse_type)

        # 2) 构建 20 个多隔室神经元（唯一 name + 逐神经元 stim_var 防串扰）
        self.neurons = {}
        for role in p.roles:
            types = post_types[role]
            sub = {t: self._m2[t] for t in types}
            eqs = chemical_post_eqs(sub)
            ims = chemical_im_terms(sub)
            stim_var = f"stim_{role.lower()}"
            neuron = MultiCompartmentNeuron(
                name=f"{self.name_prefix}_{role.lower()}",
                dt_ms=p.dt_ms, method=p.method, t_total_ms=p.t_total_ms,
                extra_eqs=eqs, extra_im_terms=ims, stim_var=stim_var,
            )
            neuron.build()
            self.neurons[role] = neuron

        # 3) 化学突触逐对连接（node3 → soma；M2 默认）
        self.chemicals = []
        for k, spec in enumerate(self._synapse_specs()):
            if spec.is_muscle:
                continue
            base = self._m2[spec.synapse_type]
            sp = SynapseParams(
                synapse_type=spec.synapse_type,
                g_max_ns=spec.g_max_ns if spec.g_max_ns is not None else base.g_max_ns,
                tau_ms=base.tau_ms, e_rev_mv=base.e_rev_mv,
                p_release=1.0, n_vesicles=1,          # 确定性默认（铁律）
                mg_mm=base.mg_mm, u0=base.u0,
                tau_fac_ms=base.tau_fac_ms, tau_rec_ms=base.tau_rec_ms,
            )
            p_rel, n_ves = self._release_for(spec)
            sp.p_release, sp.n_vesicles = p_rel, int(n_ves)
            cs = ChemicalSynapse(
                self.neurons[spec.synapse_from], self.neurons[spec.synapse_to],
                sp, pre_site="node3", post_site="soma",
                name=f"{self.name_prefix}_syn{k}_{spec.synapse_type}",
            )
            cs.build()
            cs.synapses.delay = spec.delay_ms * ms
            self.chemicals.append(cs)

        # 4) 三通道肌肉（Muscle3，§1 L1 方案①）+ 运动神经元驱动
        self.muscle3 = Muscle3(tau_ms=p.muscle_tau_ms, cap=p.muscle_cap,
                               name=f"{self.name_prefix}_muscle3")
        self.muscle3.build()
        for k, spec in enumerate(self._synapse_specs()):
            if not spec.is_muscle:
                continue
            w = spec.g_max_ns if spec.g_max_ns is not None else 0.3
            self.muscle3.connect_driver(self.neurons[spec.synapse_from],
                                        spec.muscle_channel, weight=w,
                                        name=f"{self.name_prefix}_musdrv{k}")
        self._built = True
        return self

    # ------------------------------------------------------------------ #
    # ASE 转导（numpy 侧完成 max；nA 按注入位点面积换算）
    # ------------------------------------------------------------------ #
    def _ase_site_idx(self, role: str) -> int:
        seg, comp = _parse_site(self.params.transduction.ase_site)
        return self.neurons[role].label_of(seg, comp)

    def _nA_per_density(self, role: str, idx: int) -> float:
        """该位点 1 µA/cm² → nA 换算系数（面积固定，一次计算）。"""
        return self.neurons[role].density_to_nA(1.0, idx)

    def ase_stim_nA(self, s_value: float) -> Tuple[float, float]:
        """时间差分 s → (I_ASEL nA, I_ASER nA)。

        I_ASEL = g_ON·max(s,0)、I_ASER = g_OFF·max(−s,0)（µA/cm² 密度）
        → 按注入位点面积换算 nA。max 在 numpy/python 侧完成（不进事件代码）。
        """
        tr = self.params.transduction
        s = float(s_value)
        i_on = tr.g_on * max(s, 0.0)
        i_off = tr.g_off * max(-s, 0.0)
        idx_on = self._ase_site_idx("ASEL")
        idx_off = self._ase_site_idx("ASER")
        nA_on = i_on * self._nA_per_density("ASEL", idx_on)
        nA_off = i_off * self._nA_per_density("ASER", idx_off)
        return nA_on, nA_off

    # ------------------------------------------------------------------ #
    # 刺激数组（固定形状 + 显式命名 + namespace 传参，M2 L6 纪律）
    # ------------------------------------------------------------------ #
    def _n_steps(self, t_total_ms: float) -> int:
        """固定形状纪律：至少 STIM_WINDOW_MS=500（M3 惯例）；更长试次按全长。"""
        return int(round(max(STIM_WINDOW_MS, t_total_ms) / self.params.dt_ms))

    def _stim_arrays(self, t_total_ms: float,
                     stimulated_roles: Sequence[str] = ("ASEL", "ASER")):
        """每角色一个 TimedArray（显式命名；形状固定）。

        - 受激角色（默认 ASEL/ASER + 张力角色）：全试次形状 (n_steps, n_comp)，
          数值可变（epoch 间仅数值变化 → 编译缓存命中）；
        - 其余角色：(1, n_comp) 零数组（Brian2 TimedArray 越界索引钳位 →
          恒 0，极小内存）。
        """
        from brian2 import TimedArray, amp, ms

        p = self.params
        n_steps = self._n_steps(t_total_ms)
        n_comp = self.neurons[p.roles[0]].spec.total_compartments
        stim = set(stimulated_roles) | {r for r, v in p.tonic_uA_cm2.items() if v > 0}
        arrays = {}
        for role in p.roles:
            if role in stim:
                arr = np.zeros((n_steps, n_comp)) * amp
            else:
                arr = np.zeros((1, n_comp)) * amp
            arrays[role] = TimedArray(arr, dt=p.dt_ms * ms,
                                      name=f"stim_{role.lower()}")
        return arrays

    # ------------------------------------------------------------------ #
    # 记录
    # ------------------------------------------------------------------ #
    @staticmethod
    def _default_record(roles: Sequence[str]) -> List[str]:
        return [f"{r.lower()}_soma" for r in roles] + \
               [f"{r.lower()}_node3" for r in roles]

    def _record_spec(self, record: Sequence[str]):
        """'asel_soma' 等标签 → (role, neuron, idx, label) 列表。"""
        out = []
        for lab in record:
            role_site = lab.split("_", 1)
            if len(role_site) != 2:
                raise ValueError(f"记录标签需为 <角色>_<位点>：{lab}")
            role, site = role_site[0].upper(), role_site[1]
            if role not in self.neurons:
                raise ValueError(f"记录标签角色未知：{role}")
            neuron = self.neurons[role]
            seg, comp = _parse_site(site)
            idx = neuron.label_of(seg, comp)
            out.append((role, neuron, idx, lab))
        return out

    def _spike_times(self, spmon, neuron: MultiCompartmentNeuron,
                     role: str) -> Dict[str, np.ndarray]:
        from brian2 import ms

        t_ms_arr = np.array(spmon.t / ms)
        i_arr = np.array(spmon.i)
        out = {}
        for seg in neuron.spec.segments:
            idxs = neuron.index_map[seg.name]
            if len(idxs):
                mask = np.isin(i_arr, idxs)
                out[f"{role.lower()}_{seg.name}"] = t_ms_arr[mask]
        return out

    # ------------------------------------------------------------------ #
    # 会话（开环 run / 闭环 epoch 共用同一组装路径）
    # ------------------------------------------------------------------ #
    def make_session(self, t_total_ms: Optional[float] = None,
                     record: Optional[Sequence[str]] = None,
                     stimulated_roles: Optional[Sequence[str]] = None,
                     ) -> "ChemoSession":
        """构建一个试次会话（网络 + monitor + 可变 stim 数组；已 store）。

        开环 `run()`/`run_trials()` 与闭环 `chemotaxis_loop.py` 都经由会话：
        epoch 级 `run_epoch`（闭环）或整段 `run_open_loop`（开环）。
        受激角色默认 = ASEL/ASER + 张力角色（tonic 行）。
        """
        if stimulated_roles is None:
            stimulated_roles = ("ASEL", "ASER") + self.tonic_roles
        return ChemoSession(self, t_total_ms=t_total_ms, record=record,
                            stimulated_roles=tuple(stimulated_roles))

    @property
    def tonic_roles(self) -> Tuple[str, ...]:
        """有张力注入的角色（维持静息/前进基线，M3 VB 同款机制）。"""
        return tuple(r for r, v in self.params.tonic_uA_cm2.items() if v > 0)

    def run(self, t_total_ms: Optional[float] = None,
            record: Optional[Sequence[str]] = None,
            seed: Optional[int] = None) -> ChemotaxisResult:
        """开环单次运行：注入 ASE 转导电流（默认 P1 阶跃协议）。

        t_total_ms 缺省 = 协议轨迹长度；更长则协议后保持零刺激。
        """
        s_trace, dt_proto = self._protocol_s_trace()
        t_total = float(t_total_ms or (len(s_trace) * dt_proto))
        sess = self.make_session(t_total_ms=t_total, record=record)
        sess.reset(seed=seed)
        sess.run_open_loop(s_trace, dt_proto)
        return sess.finish()

    def run_trials(self, n_trials: Optional[int] = None, seed_base: int = 0,
                   record: Optional[Sequence[str]] = None,
                   t_total_ms: Optional[float] = None) -> List[ChemotaxisResult]:
        """开环多试次（store/restore + 重播种；确定性 p=1/n=1 默认）。

        每试次 monitor 数据即该试次自身的完整轨迹（M3 L12 语义）。
        """
        p = self.params
        n = int(n_trials or p.protocol.n_trials)
        s_trace, dt_proto = self._protocol_s_trace()
        t_total = float(t_total_ms or (len(s_trace) * dt_proto))
        sess = self.make_session(t_total_ms=t_total, record=record)
        results = []
        for trial in range(n):
            sess.reset(seed=seed_base + trial)
            sess.run_open_loop(s_trace, dt_proto)
            r = sess.finish()
            r.meta["trial"] = trial
            results.append(r)
        return results

    def _protocol_s_trace(self) -> Tuple[np.ndarray, float]:
        """当前协议 → (s(t) 轨迹, dt_ms)（缺省 = P1 阶跃协议，CSV p1_* 定稿）。"""
        p = self.params
        if self._protocol is not None:
            c_trace, s_trace, dt_proto = self._protocol
        else:
            info = self.protocol_info()
            _, c_trace = step_protocol(
                c_base=info["c_base"], delta_c=info["delta_c"],
                t_baseline_ms=info["t_baseline_ms"], t_up_ms=info["t_up_ms"],
                t_hold_ms=info["t_hold_ms"], t_down_ms=info["t_down_ms"],
                dt_ms=p.dt_ms)
            s_trace, dt_proto = None, p.dt_ms
        if s_trace is None:
            s_trace = ChemotaxisEnv.time_diff_trace(
                c_trace, p.transduction.tau_win_ms, dt_proto)
        return np.asarray(s_trace, dtype=float), float(dt_proto)

    # ------------------------------------------------------------------ #
    # 元数据 / 内省（B2 验证脚本用）
    # ------------------------------------------------------------------ #
    def chain_summary(self) -> dict:
        """拓扑/极性内省（P2 断言：角色/连接数/递质类型/肌肉通道）。"""
        chems = [s for s in self.params.synapses if not s.is_muscle]
        muscs = [s for s in self.params.synapses if s.is_muscle]
        return dict(
            roles=list(self.params.roles),
            n_chemical=len(chems),
            n_muscle_drives=len(muscs),
            synapse_types={f"{s.synapse_from}->{s.synapse_to}": s.synapse_type
                           for s in chems},
            muscle_channels={f"{s.synapse_from}->{s.synapse_to}": s.muscle_channel
                             for s in muscs},
            n_ampa=sum(1 for s in chems if s.synapse_type == "ampa"),
            n_gaba=sum(1 for s in chems if s.synapse_type == "gaba"),
            n_nmda=sum(1 for s in chems if s.synapse_type == "nmda"),
            transduction=dict(self.params.transduction.__dict__),
            mech_a=dict(self.params.mech_a.__dict__),
            env=dict(self.params.env.__dict__),
            body=dict(self.params.body.__dict__),
            protocol=dict(self.params.protocol.__dict__),
            dt_ms=self.params.dt_ms, method=self.params.method,
            seed=self.params.seed,
        )

    def chain_from(self, start_role: str,
                   target_muscle_channels: Sequence[str] = ("fwd",)) -> List[str]:
        """沿出边 BFS 找 start_role → 驱动目标肌肉通道的最短角色链（按序）。

        P2 链传播/绘图用；找不到目标时返回 [start_role]。
        """
        adj: Dict[str, List[str]] = {}
        for s in self.params.synapses:
            if not s.is_muscle:
                adj.setdefault(s.synapse_from, []).append(s.synapse_to)
        muscle_drivers = {s.synapse_from: s.muscle_channel
                          for s in self.params.synapses if s.is_muscle}
        prev = {start_role: None}
        q = deque([start_role])
        target = None
        while q:
            r = q.popleft()
            if r in muscle_drivers and muscle_drivers[r] in target_muscle_channels:
                target = r
                break
            for nxt in adj.get(r, []):
                if nxt not in prev:
                    prev[nxt] = r
                    q.append(nxt)
        if target is None:
            return [start_role]
        chain = []
        r = target
        while r is not None:
            chain.append(r)
            r = prev[r]
        return chain[::-1]

    def _meta(self, t_total, seed, c_fwd, c_left, c_right, trial=None) -> dict:
        p = self.params
        tr = p.transduction
        m = dict(
            csv_path=p.csv_path, t_total_ms=t_total, dt_ms=p.dt_ms,
            method=p.method, seed=seed,
            g_on=tr.g_on, g_off=tr.g_off, tau_win_ms=tr.tau_win_ms,
            ase_site=tr.ase_site,
            mech_a=dict(p.mech_a.__dict__),
            muscle_tau_ms=p.muscle_tau_ms, muscle_cap=p.muscle_cap,
            c_fwd_peak=float(np.max(c_fwd)) if len(c_fwd) else 0.0,
            c_left_peak=float(np.max(c_left)) if len(c_left) else 0.0,
            c_right_peak=float(np.max(c_right)) if len(c_right) else 0.0,
            protocol=self.protocol_info(),
            env=dict(p.env.__dict__), body=dict(p.body.__dict__),
        )
        if trial is not None:
            m["trial"] = trial
        return m


# --------------------------------------------------------------------- #
# ChemoSession：一次试次的网络会话（epoch 迭代 + store/restore）
# --------------------------------------------------------------------- #
class ChemoSession:
    """一次试次的网络会话（清单 §4.2 #1 闭环 epoch 迭代的引擎侧）。

    用法（闭环，chemotaxis_loop.py）：
        sess = circuit.make_session(t_total_ms=...)
        sess.reset(seed=...)                     # 试次开始（restore + 重播种）
        for e in range(n_epochs):
            s = env.time_diff(...)
            mus = sess.run_epoch(dt_b, s)        # 运行 ΔT，返回肌肉收缩
            body.step(mus["fwd"], mus["left"], mus["right"])
        result = sess.finish(x=..., y=..., theta=...)

    开环（整段一次运行）：`sess.run_open_loop(s_trace, dt_protocol_ms)`。

    编译缓存纪律：stim TimedArray 固定形状 + 显式命名；epoch 间仅数值变化
    （写入 `ta.values` 的绝对时间切片，pad 零）→ 不重编译。
    """

    def __init__(self, circuit: ChemotaxisCircuit, t_total_ms: Optional[float] = None,
                 record: Optional[Sequence[str]] = None,
                 stimulated_roles: Sequence[str] = ("ASEL", "ASER")):
        from brian2 import Network, SpikeMonitor, StateMonitor, ms, seed as bseed

        self.circuit = circuit
        p = circuit.params
        self.t_total_ms = float(t_total_ms or p.protocol.t_total_ms)
        self.stimulated_roles = tuple(stimulated_roles)

        circuit.build()
        self._assemble(record)

        # 初始化 + 快照（M3 L12：restore 重置 monitor → 每试次完整轨迹）
        bseed(circuit.seed)
        self.net.run(0 * ms, namespace=self.ns)
        self.net.store()
        self._rng = np.random.default_rng(circuit.seed)

    def _assemble(self, record: Optional[Sequence[str]]):
        from brian2 import Network, SpikeMonitor, StateMonitor, ms

        circuit = self.circuit
        p = circuit.params
        record = list(record) if record is not None \
            else circuit._default_record(p.roles)
        rec = circuit._record_spec(record)

        # 逐角色 StateMonitor（同组多索引；标签按位置对应）
        monos: Dict[str, dict] = {}
        for role, neuron, idx, lab in rec:
            monos.setdefault(role, {"neuron": neuron, "idx": [], "labels": []})
            if idx not in monos[role]["idx"]:
                monos[role]["idx"].append(idx)
                monos[role]["labels"].append(lab)
        vmons = []
        for role, m in monos.items():
            vmons.append(StateMonitor(m["neuron"].neuron, "v", record=m["idx"],
                                      dt=p.dt_ms * ms,
                                      name=f"mon_{role.lower()}_v"))
        spmons = {r: SpikeMonitor(n.neuron, "v", name=f"sp_{r.lower()}")
                  for r, n in circuit.neurons.items()}
        mons_mus = circuit.muscle3.monitor(p.body.dt_b,
                                           name=f"{circuit.name_prefix}_musc")

        net = Network()
        for n in circuit.neurons.values():
            net.add(n.neuron)
        for cs in circuit.chemicals:
            net.add(cs.synapses)
        for g in circuit.muscle3.groups:
            net.add(g)
        for s in circuit.muscle3.drivers:
            net.add(s)
        for m in vmons:
            net.add(m)
        for sp in spmons.values():
            net.add(sp)
        for mm in mons_mus:
            net.add(mm)

        stims = circuit._stim_arrays(self.t_total_ms, self.stimulated_roles)
        ns = {f"stim_{rl.lower()}": ta for rl, ta in stims.items()}

        self.net, self.ns = net, ns
        self.stims = stims
        self.vmons, self.spmons = vmons, spmons
        self.mons_mus = mons_mus
        self.monos = monos
        self.rec = rec
        self._n_epochs = 0
        self._fill_tonic()

    def _fill_tonic(self):
        """张力注入（M3 VB 同款）：恒定密度按位点面积 → nA，写入整段数组。"""
        circuit = self.circuit
        for role, density in circuit.params.tonic_uA_cm2.items():
            if density <= 0 or role not in self.stims:
                continue
            idx = circuit.neurons[role].label_of("soma")
            k = circuit._nA_per_density(role, idx)
            self.stims[role].values[:, idx] = (density * k) * 1e-9

    # ------------------------------------------------------------------ #
    # 试次控制
    # ------------------------------------------------------------------ #
    def reset(self, seed: Optional[int] = None):
        """试次开始：restore 干净状态 + 重播种 + 清零 stim 数组。"""
        from brian2 import seed as bseed

        bseed(seed if seed is not None else self.circuit.seed)
        self.net.restore()
        for ta in self.stims.values():
            ta.values[:] = 0.0
        self._fill_tonic()          # 张力是常数——清零后重填（restore 不影响 TimedArray）
        self._rng = np.random.default_rng(seed if seed is not None
                                          else self.circuit.seed)
        self._n_epochs = 0

    # ------------------------------------------------------------------ #
    # 闭环 epoch 运行
    # ------------------------------------------------------------------ #
    def run_epoch(self, dt_ms: float, s_value: float) -> Dict[str, float]:
        """运行一个 epoch（ΔT = dt_ms）：s 恒定 → ASE nA → 写入固定形状数组
        的当前绝对时间切片（pad 零）→ net.run(dt)。返回 epoch 末肌肉收缩。"""
        from brian2 import ms

        circuit = self.circuit
        p = circuit.params
        s = float(s_value)
        if circuit._ase_noise and p.transduction.noise > 0:
            s = s + p.transduction.noise * float(self._rng.standard_normal())
        nA_on, nA_off = circuit.ase_stim_nA(s)
        idx_on = circuit._ase_site_idx("ASEL")
        idx_off = circuit._ase_site_idx("ASER")
        t_now_ms = float(self.net.t / ms)
        dt = float(dt_ms)
        i0 = int(round(t_now_ms / p.dt_ms))
        i1 = int(round((t_now_ms + dt) / p.dt_ms))
        ta_on = self.stims["ASEL"]
        ta_off = self.stims["ASER"]
        ta_on.values[i0:i1, idx_on] = nA_on * 1e-9
        ta_off.values[i0:i1, idx_off] = nA_off * 1e-9
        self.net.run(dt * ms, namespace=self.ns)
        self._n_epochs += 1
        return self.muscle_read()

    def run_epoch_raw(self, dt_ms: float, stim_nA: Dict[str, float]) -> Dict[str, float]:
        """底层注入：{role: nA} 按各角色注入位点写入并 run(dt)（B1c 校准钩子）。"""
        from brian2 import ms

        circuit = self.circuit
        p = circuit.params
        t_now_ms = float(self.net.t / ms)
        dt = float(dt_ms)
        i0 = int(round(t_now_ms / p.dt_ms))
        i1 = int(round((t_now_ms + dt) / p.dt_ms))
        for role, nA in stim_nA.items():
            if role not in self.stims:
                raise ValueError(f"未知受激角色：{role}")
            idx = circuit._ase_site_idx(role)
            self.stims[role].values[i0:i1, idx] = float(nA) * 1e-9
        self.net.run(dt * ms, namespace=self.ns)
        self._n_epochs += 1
        return self.muscle_read()

    def run_open_loop(self, s_trace: Sequence[float],
                      dt_protocol_ms: Optional[float] = None):
        """开环：整段 s(t) 一次写入固定形状数组 + 单次 net.run(t_total)。

        重采样到电路 dt 网格；协议短于 t_total 的部分 pad 零（静默）。
        max(±s,0) 在 numpy 侧完成（事件代码无分支，M3 L11）。
        """
        from brian2 import ms

        circuit = self.circuit
        p = circuit.params
        dt_proto = p.dt_ms if dt_protocol_ms is None else float(dt_protocol_ms)
        s_arr = np.asarray(s_trace, dtype=float)
        n_steps = self.stims["ASEL"].values.shape[0]
        t = np.arange(n_steps) * p.dt_ms
        s = np.interp(t, np.arange(len(s_arr)) * dt_proto, s_arr,
                      left=0.0, right=0.0)
        if circuit._ase_noise and p.transduction.noise > 0:
            s = s + p.transduction.noise * self._rng.standard_normal(s.shape)
        i_on = p.transduction.g_on * np.maximum(s, 0.0)     # µA/cm²
        i_off = p.transduction.g_off * np.maximum(-s, 0.0)
        idx_on = circuit._ase_site_idx("ASEL")
        idx_off = circuit._ase_site_idx("ASER")
        k_on = circuit._nA_per_density("ASEL", idx_on)      # nA per µA/cm²
        k_off = circuit._nA_per_density("ASER", idx_off)
        ta_on = self.stims["ASEL"]
        ta_off = self.stims["ASER"]
        ta_on.values[:] = 0.0
        ta_off.values[:] = 0.0
        ta_on.values[:, idx_on] = (i_on * k_on) * 1e-9
        ta_off.values[:, idx_off] = (i_off * k_off) * 1e-9
        self.net.run(self.t_total_ms * ms, namespace=self.ns)

    def muscle_read(self) -> Dict[str, float]:
        """当前三通道肌肉收缩值（epoch 末读取 → 运动学积分）。"""
        return self.circuit.muscle3.read()

    def any_spikes_in_window(self, roles, t0_ms: float, t1_ms: float) -> bool:
        """roles 中任一角色在 [t0, t1)（ms）内是否有发放。

        机制 A 转向触发判定：ASER→AIB→RIA→SMDD 链在本 epoch 内激活
        （SMDD 发放）→ 电路耦合的"转向事件"触发条件之一。
        """
        from brian2 import ms

        for role in roles:
            sp = self.spmons.get(str(role).upper())
            if sp is None:
                continue
            t = np.asarray(sp.t / ms)
            if t.size and np.any((t >= t0_ms - 1e-9) & (t < t1_ms)):
                return True
        return False

    # ------------------------------------------------------------------ #
    # 结果收集
    # ------------------------------------------------------------------ #
    def finish(self, x: Optional[Sequence[float]] = None,
               y: Optional[Sequence[float]] = None,
               theta: Optional[Sequence[float]] = None,
               meta_extra: Optional[dict] = None) -> ChemotaxisResult:
        """收集本试次结果（V / 发放 / 肌肉收缩 / 可选闭环轨迹）。"""
        from brian2 import ms, mV

        circuit = self.circuit
        p = circuit.params
        t = np.array(self.vmons[0].t / ms) if self.vmons \
            else np.arange(0.0, self.t_total_ms, p.dt_ms)
        v = {}
        for role, mon_obj in zip(self.monos, self.vmons):
            for pos, lab in enumerate(self.monos[role]["labels"]):
                v[lab] = np.array(mon_obj.v[pos] / mV)
        spikes: Dict[str, np.ndarray] = {}
        for role, sp in self.spmons.items():
            spikes.update(circuit._spike_times(sp, circuit.neurons[role], role))
        c_fwd = np.array(self.mons_mus[0].c_fwd[0])
        c_left = np.array(self.mons_mus[1].c_left[0])
        c_right = np.array(self.mons_mus[2].c_right[0])
        meta = circuit._meta(self.t_total_ms, p.seed, c_fwd, c_left, c_right)
        meta["n_epochs"] = self._n_epochs
        if meta_extra:
            meta.update(meta_extra)
        return ChemotaxisResult(
            t_ms=t, v_mv=v, spike_times_ms=spikes,
            c_fwd=c_fwd, c_left=c_left, c_right=c_right, meta=meta,
            x=None if x is None else np.asarray(x, dtype=float),
            y=None if y is None else np.asarray(y, dtype=float),
            theta=None if theta is None else np.asarray(theta, dtype=float),
        )


# --------------------------------------------------------------------- #
# 绘图（reports/neuro/m4_smoke.png）
# --------------------------------------------------------------------- #
def plot_chemotaxis(open_result: ChemotaxisResult, traj_result: Optional[ChemotaxisResult],
                    out_png: Optional[str] = None,
                    chain_a_roles: Optional[Sequence[str]] = None,
                    chain_b_roles: Optional[Sequence[str]] = None) -> str:
    """核心链各级 V（两条子链）+ 一条趋化轨迹（含象限/食物/CI 标注）。

    - 链 A：ASEL → AIY → AVB → VB/DB（前进，C_fwd）；
    - 链 B：ASER → AIB → RIA → SMDD（转向，C_left/C_right）。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reports_dir = os.path.join(ROOT, "neural_exploration", "reports", "neuro")
    os.makedirs(reports_dir, exist_ok=True)
    out_png = out_png or os.path.join(reports_dir, "m4_smoke.png")

    chain_a = list(chain_a_roles) if chain_a_roles else ["ASEL", "AIYL", "AVBL", "VB"]
    chain_b = list(chain_b_roles) if chain_b_roles else ["ASER", "AIBL", "RIAL", "SMDDL"]

    fig, axes = plt.subplots(len(chain_a) + len(chain_b) + 1, 1,
                             figsize=(11, 2.1 * (len(chain_a) + len(chain_b) + 1)),
                             sharex=False)
    t = open_result.t_ms
    info = open_result.meta.get("protocol", {})
    rise_start = info.get("rise_start_ms", 40.0)
    rise_end = info.get("rise_end_ms", 90.0)
    fall_start = info.get("fall_start_ms", 140.0)
    fall_end = info.get("fall_end_ms", 190.0)

    def _panel(ax, role, color):
        lab = f"{role.lower()}_soma"
        if lab in open_result.v_mv:
            ax.plot(t, open_result.v_mv[lab], lw=0.8, color=color)
        ax.axhline(-20.0, color="gray", ls="--", lw=0.7)
        spikes = open_result.spikes(role, "node3")
        for tk in spikes:
            ax.axvline(tk, color=color, ls=":", lw=0.8, alpha=0.7)
        ax.set_ylabel(f"{role} V")
        ax.set_ylim(-90, 60)
        ax.grid(alpha=0.3)
        ax.set_title(f"{role}: {len(spikes)} spikes @ node3", fontsize=8, loc="left")

    colors_a = ["#1f77b4", "#ff7f0e", "#2ca02c", "#8c564b"]
    colors_b = ["#d62728", "#9467bd", "#e377c2", "#17becf"]
    for ax, role, color in zip(axes[:len(chain_a)], chain_a, colors_a):
        _panel(ax, role, color)
    for ax, role, color in zip(axes[len(chain_a):len(chain_a) + len(chain_b)],
                               chain_b, colors_b):
        _panel(ax, role, color)
    # 共享阶跃窗标注（画在链 A 第一子图）
    ax0 = axes[0]
    ax0.axvspan(rise_start, rise_end, color="green", alpha=0.12)
    ax0.axvspan(fall_start, fall_end, color="red", alpha=0.12)
    ax0.text(rise_start + 1, 45, "rise ΔC>0", fontsize=7, color="green")
    ax0.text(fall_start + 1, 45, "fall ΔC<0", fontsize=7, color="red")

    # 最后子图：趋化轨迹（闭环）或肌肉收缩（开环）
    axm = axes[-1]
    if traj_result is not None and traj_result.has_trajectory:
        env_spec = traj_result.meta.get("env", {})
        L = env_spec.get("arena_L", 10.0)
        fx = env_spec.get("food_x", 7.5)
        fy = env_spec.get("food_y", 7.5)
        axm.plot(traj_result.x, traj_result.y, lw=1.4, color="#1f77b4",
                 label="worm trajectory")
        axm.plot(traj_result.x[0], traj_result.y[0], "o", color="k", label="start")
        axm.plot(fx, fy, marker="*", ms=16, color="red", label="food")
        axm.axvline(L / 2, color="gray", ls="--", lw=0.7)
        axm.axhline(L / 2, color="gray", ls="--", lw=0.7)
        axm.set_xlim(-0.2, L + 0.2)
        axm.set_ylim(-0.2, L + 0.2)
        ci = traj_result.ci
        axm.set_title(f"chemotaxis trajectory (CI = {ci:.3f})",
                      fontsize=9, loc="left")
        axm.set_aspect("equal")
    else:
        axm.plot(t, open_result.c_fwd, lw=1.2, color="#2ca02c", label="C_fwd")
        axm.plot(t, open_result.c_left, lw=1.2, color="#9467bd", label="C_left")
        axm.plot(t, open_result.c_right, lw=1.2, color="#e377c2", label="C_right")
        axm.set_ylabel("muscle C")
        axm.set_xlabel("t (ms)")
        axm.set_title("muscle contraction (open-loop run)", fontsize=9, loc="left")
    axm.grid(alpha=0.3)
    axm.legend(loc="upper right", fontsize=8)

    fig.suptitle("M4 chemotaxis circuit: ASE ON/OFF → AIY/AIB → motor → virtual body",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="M4 趋化回路开环单次运行 + 出图")
    ap.add_argument("--t-total-ms", type=float, default=None)
    ap.add_argument("--noplot", action="store_true")
    args = ap.parse_args()

    circ = ChemotaxisCircuit()
    r = circ.run(t_total_ms=args.t_total_ms)
    print(f"roles={len(circ.chain_summary()['roles'])}  "
          f"chem={circ.chain_summary()['n_chemical']}  "
          f"muscle={circ.chain_summary()['n_muscle_drives']}")
    for role in ("ASEL", "ASER", "AIYL", "AVBL", "VB"):
        st = r.spikes(role, "node3")
        print(f"  {role}: {len(st)} spikes @ node3, first={st[0]:.2f} ms" if len(st)
              else f"  {role}: 0 spikes")
    if not args.noplot:
        png = plot_chemotaxis(r, None)
        print(f"图已生成: {png}")
