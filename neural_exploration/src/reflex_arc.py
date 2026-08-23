"""M3 正式实现：`ReflexArc` —— 触觉反射弧多神经元链（清单 §2.1/§4）。

拓扑（4 神经元 + 2 肌肉，双方向通道）：
    后部触刺激 S(t)（注入 PLM 树突端）
      → PLM（感觉神经元）--AMPA--> AVM（中间神经元）
          --AMPA--> DA（后退运动神经元）→ 肌肉 C_back（后退收缩）
          --GABA--> VB（前进运动神经元，张力注入维持基线）→ 肌肉 C_fwd（前进收缩）

链定义唯一来源：`data/m3_reflex_params.csv`（B1a 定稿；本模块运行期读取，
缺失时给出清晰报错）。CSV 列（清单 §2.4）：
    role, neuron_class, synapse_from, synapse_to, synapse_type, g_max_ns,
    delay_ms, tonic_uA_cm2, value, note
  - 神经元行：role=<PLM|AVM|DA|VB>，tonic_uA_cm2 可选（VB 张力注入）；
  - 突触行：synapse_from + synapse_to + synapse_type(ampa|gaba|nmda|muscle)
    + g_max_ns(覆盖；缺省用 m2_synapse_params.csv 默认) + delay_ms；
    synapse_type=muscle 时 synapse_to 为 muscle_back/muscle_fwd，g_max_ns 即收缩权重 w；
  - 参数行：role=param, neuron_class=<参数名>, value=<值>（touch/肌肉/刺激档位/潜伏期窗…）。

确定性铁律（清单 §4.2 #5）：**默认全链 p_release=1、n_vesicles=1**（无随机，
同参数重跑逐位一致）——CSV 中若有 p/n 列也会被忽略；量子释放噪声由调用方
显式开启（`set_quantum_noise(p=0.95, n=2)`，P1/P3 重复试次实验用）。

M2 交接复用（清单 §1 L1）：`MultiCompartmentNeuron`（多隔室 HH）、
`ChemicalSynapse`（pre_site="node3"→post_site="soma"）、`chemical_post_eqs`/
`chemical_im_terms`、`STIM_WINDOW_MS` 固定形状 + TimedArray 显式命名 +
namespace 传参（编译缓存纪律）、store/restore + 重播种多试次机制。

刺激协议：触刺激 TimedArray 注入 PLM 树突端（`density_to_nA` 换算），6 档强度
{0,0.5,1,2,4,8}×I0 共用同形数组；刺激开始 ≥ 40ms（HH 静息瞬态漂移，M1 §L3）。
"""

from __future__ import annotations

import os
import sys
import time as _time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.src.muscle import (  # noqa: E402
    Muscle,
    direction_peak,
)
from neural_exploration.src.neuron_model import MultiCompartmentNeuron  # noqa: E402
from neural_exploration.src.neuron_pair import STIM_WINDOW_MS  # noqa: E402  # M2 L6 固定形状约定
from neural_exploration.src.synapse_model import (  # noqa: E402
    ChemicalSynapse,
    SynapseParams,
    chemical_im_terms,
    chemical_post_eqs,
    load_synapse_params,
)

DEFAULT_REFLEX_PARAMS_CSV = os.path.join(ROOT, "neural_exploration", "data",
                                         "m3_reflex_params.csv")

#: 链角色（CSV 应含这些 role；PLM 收触刺激、VB 收张力）
CHAIN_ROLES = ("PLM", "AVM", "DA", "VB")

#: 刺激强度档位默认（清单 P4；CSV 参数行可覆盖）
DEFAULT_INTENSITY_LEVELS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)


# --------------------------------------------------------------------- #
# 链规格（CSV 驱动）
# --------------------------------------------------------------------- #
@dataclass
class ReflexSynapseSpec:
    """一条链连接（化学突触或肌肉驱动）。"""

    synapse_from: str
    synapse_to: str
    synapse_type: str            # ampa | gaba | nmda | muscle
    g_max_ns: Optional[float] = None   # None → m2_synapse_params.csv 默认
    delay_ms: float = 0.1
    p_release: Optional[float] = None  # 存而不用于默认运行（确定性铁律）
    n_vesicles: Optional[int] = None
    note: str = ""

    @property
    def is_muscle(self) -> bool:
        return self.synapse_type == "muscle"

    @property
    def muscle_channel(self) -> str:
        """muscle 行的目标通道：muscle_back → 'back'，muscle_fwd → 'fwd'。"""
        to = self.synapse_to.lower()
        if to == "muscle_back":
            return "back"
        if to == "muscle_fwd":
            return "fwd"
        raise ValueError(f"muscle 突触目标需为 muscle_back/muscle_fwd：{self.synapse_to}")


@dataclass
class ReflexTouchSpec:
    """触刺激协议（清单 §2.2 方案 A：触电流密度）。"""

    site: str = "dend2#1"        # PLM 树突端（远端树突隔室）
    start_ms: float = 50.0       # 刺激开始 ≥ 40ms（HH 静息漂移）
    dur_ms: float = 8.0
    i0_uA_cm2: float = 20.0      # 基准强度（档位 1×I0）


@dataclass
class ReflexMuscleSpec:
    """虚拟肌肉参数（清单 §2.3）。"""

    tau_ms: float = 20.0
    w_back: float = 0.4          # 无 muscle 行时的后备权重（行优先）
    w_fwd: float = 0.3
    cap: Optional[float] = 1.0   # C∈[0,1] 可选饱和；None = 不饱和


@dataclass
class ReflexParams:
    """整条反射弧参数（唯一定稿源 = CSV）。"""

    roles: List[str] = field(default_factory=list)          # 链序
    tonic_uA_cm2: Dict[str, float] = field(default_factory=dict)  # role → 张力注入
    synapses: List[ReflexSynapseSpec] = field(default_factory=list)
    touch: ReflexTouchSpec = field(default_factory=ReflexTouchSpec)
    muscle: ReflexMuscleSpec = field(default_factory=ReflexMuscleSpec)
    intensity_levels: List[float] = field(default_factory=lambda: list(DEFAULT_INTENSITY_LEVELS))
    latency_window_ms: Tuple[float, float] = (25.0, 60.0)   # 行为潜伏期目标窗（P3 判据容差）
    chalfie_window_ms: Tuple[float, float] = (30.0, 50.0)   # Chalfie 1985 行为参考窗
    t_total_ms: float = 250.0
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


def load_reflex_params(csv_path: Optional[str] = None) -> ReflexParams:
    """读入 data/m3_reflex_params.csv → ReflexParams（运行期读取，唯一定稿源）。

    CSV 缺失 → FileNotFoundError（清晰报错）。列容差：神经元行/突触行/参数行
    由非空列判别；缺列用默认值（.get 容错，便于 B1a 定稿微调列）。
    """
    path = csv_path or DEFAULT_REFLEX_PARAMS_CSV
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"M3 链参数 CSV 不存在：{path}\n"
            "ReflexArc 的参数唯一定稿源是 data/m3_reflex_params.csv（B1a 节点产出）。\n"
            "请先确认该文件已生成（列：role/neuron_class/synapse_from/synapse_to/"
            "synapse_type/g_max_ns/delay_ms/tonic_uA_cm2/value/note），或运行 "
            "tests/neuro/test_reflex_smoke.py 等待其生成。"
        )
    import csv

    p = ReflexParams(csv_path=path)
    touch_kw, muscle_kw = {}, {}
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

            if role.lower() == "param":
                key = _f("neuron_class") or frm
                # 值可能落在 value / note / 表头外扩展字段（B1a 惯例：value 列空、
                # 值写在 note 列；intensity_levels 未加引号溢出到 row[None]）。
                # 容错：拼接 value+note+扩展字段的非空段。
                trailing = []
                for col in ("value", "note"):
                    v = (r.get(col) or "").strip()
                    if v:
                        trailing.append(v)
                for x in (r.get(None) or ()):
                    if x is not None and str(x).strip():
                        trailing.append(str(x).strip())
                val = ",".join(trailing) if trailing else to
                _apply_param(p, touch_kw, muscle_kw, key, val, note)
            elif frm and to:
                g = _f("g_max_ns")
                p.synapses.append(ReflexSynapseSpec(
                    synapse_from=frm, synapse_to=to, synapse_type=stype,
                    g_max_ns=float(g) if g is not None else None,
                    delay_ms=float(_f("delay_ms", "0.1")),
                    p_release=float(_f("p_release")) if _f("p_release") else None,
                    n_vesicles=int(float(_f("n_vesicles"))) if _f("n_vesicles") else None,
                    note=note,
                ))
            elif role:
                p.roles.append(role)
                t = _f("tonic_uA_cm2")
                if t is not None:
                    p.tonic_uA_cm2[role] = float(t)
                # neuron_class 列当前未用（全部神经元共用 M1 形态学）
            else:
                raise ValueError(f"m3_reflex_params.csv 行无法识别（role/synapse 均空）：{r}")

    if touch_kw:
        p.touch = ReflexTouchSpec(**{**p.touch.__dict__, **touch_kw})
    if muscle_kw:
        p.muscle = ReflexMuscleSpec(**{**p.muscle.__dict__, **muscle_kw})
    _validate_params(p)
    return p


def _apply_param(p: ReflexParams, touch_kw: dict, muscle_kw: dict,
                 key: str, val: str, note: str):
    """参数行：key → 目标字段。值尽量转 float；不能转的保留字符串。"""
    def _num(v):
        try:
            return float(v)
        except ValueError:
            return v

    k = key.strip().lower()
    v = _num(val)
    if k == "touch_site":
        touch_kw["site"] = val
    elif k == "touch_start_ms":
        touch_kw["start_ms"] = float(v)
    elif k == "touch_dur_ms":
        touch_kw["dur_ms"] = float(v)
    elif k == "i0_uA_cm2".lower() or k == "i0_uacm2":
        touch_kw["i0_uA_cm2"] = float(v)
    elif k == "muscle_tau_ms":
        muscle_kw["tau_ms"] = float(v)
    elif k == "muscle_w_back":
        muscle_kw["w_back"] = float(v)
    elif k == "muscle_w_fwd":
        muscle_kw["w_fwd"] = float(v)
    elif k == "muscle_cap":
        muscle_kw["cap"] = None if str(v).lower() in ("none", "null", "") else float(v)
    elif k == "intensity_levels":
        p.intensity_levels = [float(x) for x in str(v).replace(" ", "").split(",")]
    elif k == "latency_lo_ms":
        p.latency_window_ms = (float(v), p.latency_window_ms[1])
    elif k == "latency_hi_ms":
        p.latency_window_ms = (p.latency_window_ms[0], float(v))
    elif k == "chalfie_lo_ms":
        p.chalfie_window_ms = (float(v), p.chalfie_window_ms[1])
    elif k == "chalfie_hi_ms":
        p.chalfie_window_ms = (p.chalfie_window_ms[0], float(v))
    elif k == "t_total_ms":
        p.t_total_ms = float(v)
    elif k == "dt_ms":
        p.dt_ms = float(v)
    elif k == "method":
        p.method = str(v)
    elif k == "seed":
        p.seed = int(float(v))
    else:
        # 未知参数键：宽容忽略（记录到 note 之外的扩展字段由 B1a 自由使用）
        pass


def _validate_params(p: ReflexParams):
    """拓扑/极性校验：角色齐全、突触端点存在、肌肉驱动合法。"""
    for r in CHAIN_ROLES:
        if r not in p.roles:
            raise ValueError(f"m3_reflex_params.csv 缺少链角色 {r}（当前：{p.roles}）")
    known = set(p.roles) | {"muscle_back", "muscle_fwd"}
    for s in p.synapses:
        if s.synapse_from not in p.roles:
            raise ValueError(f"突触起点 {s.synapse_from} 不在角色列表 {p.roles}")
        if s.is_muscle:
            if s.synapse_to.lower() not in ("muscle_back", "muscle_fwd"):
                raise ValueError(f"muscle 突触目标需为 muscle_back/muscle_fwd：{s.synapse_to}")
        elif s.synapse_to not in p.roles:
            raise ValueError(f"突触终点 {s.synapse_to} 不在角色列表 {p.roles}")
        if s.synapse_type not in ("ampa", "gaba", "nmda", "muscle"):
            raise ValueError(f"未知突触类型：{s.synapse_type}")
    if not any(s.is_muscle for s in p.synapses):
        raise ValueError("m3_reflex_params.csv 缺少肌肉驱动行（DA→muscle_back / VB→muscle_fwd）")


def wait_for_csv(csv_path: Optional[str] = None, timeout_s: float = 900.0,
                 interval_s: float = 30.0) -> str:
    """轮询等待 CSV 生成（B1a 节点可能稍后产出；测试用）。

    已存在则立即返回；超时抛 FileNotFoundError。
    """
    path = csv_path or DEFAULT_REFLEX_PARAMS_CSV
    t0 = _time.time()
    while not os.path.exists(path):
        if _time.time() - t0 > timeout_s:
            raise FileNotFoundError(
                f"等待 {timeout_s:.0f}s 后 m3_reflex_params.csv 仍未生成：{path}")
        _time.sleep(interval_s)
    return path


# --------------------------------------------------------------------- #
# 运行结果
# --------------------------------------------------------------------- #
@dataclass
class ReflexResult:
    """一次运行的输出（P1–P5 判定脚本的输入）。"""

    t_ms: np.ndarray
    v_mv: Dict[str, np.ndarray]            # 标签 → V(t)（mV），如 'plm_soma'
    spike_times_ms: Dict[str, np.ndarray]  # 标签 → 发放时刻（ms），如 'plm_node3'
    c_back: np.ndarray
    c_fwd: np.ndarray
    meta: Dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # 判定便利属性（清单 §2.3 / P1 / P3）
    # ------------------------------------------------------------------ #
    @property
    def d_peak(self) -> float:
        """方向判定 D_peak = max(C_back − C_fwd)（> 0.3 → 后退）。"""
        return direction_peak(self.c_back, self.c_fwd)

    @property
    def c_back_peak(self) -> float:
        return float(np.max(self.c_back))

    @property
    def c_fwd_peak(self) -> float:
        return float(np.max(self.c_fwd))

    def spikes(self, role: str, site: str = "node3") -> np.ndarray:
        """某角色某位点的发放时刻（无则空数组）。role 大小写不敏感。"""
        lab = f"{role.lower()}_{site}"
        return self.spike_times_ms.get(lab, np.array([]))

    def __eq__(self, other) -> bool:
        """确定性验证：数值逐位比较（M0 SmokeResult 同款约定）。"""
        if not isinstance(other, ReflexResult):
            return NotImplemented
        return (
            np.array_equal(self.t_ms, other.t_ms)
            and self.v_mv.keys() == other.v_mv.keys()
            and all(np.array_equal(self.v_mv[k], other.v_mv[k]) for k in self.v_mv)
            and self.spike_times_ms.keys() == other.spike_times_ms.keys()
            and all(np.array_equal(self.spike_times_ms[k], other.spike_times_ms[k])
                    for k in self.spike_times_ms)
            and np.array_equal(self.c_back, other.c_back)
            and np.array_equal(self.c_fwd, other.c_fwd)
        )


# --------------------------------------------------------------------- #
# ReflexArc：N 神经元链组装 + 刺激 + 张力 + 肌肉
# --------------------------------------------------------------------- #
class ReflexArc:
    """触觉反射弧（主链 4 神经元 + 2 肌肉，CSV 驱动）。

    用法：
        arc = ReflexArc()                       # 读 data/m3_reflex_params.csv
        r = arc.run(intensity=1.0)              # 基准触刺激单次运行
        r.d_peak, r.spike_times_ms["plm_node3"] # 方向判定 / 发放时刻
        arc.set_quantum_noise(0.95, 2)          # P1/P3 开启量子释放噪声
        trials = arc.run_trials(intensity=1.0, n_trials=5, seed_base=0)
    """

    def __init__(
        self,
        csv_path: Optional[str] = None,
        dt_ms: Optional[float] = None,
        method: Optional[str] = None,
        t_total_ms: Optional[float] = None,
        seed: Optional[int] = None,
        name_prefix: str = "arc",
    ):
        self.params: ReflexParams = load_reflex_params(csv_path)
        # 构造参数（显式传入）覆盖 CSV；None 默认 = 以 CSV 为准（唯一定稿源）
        if dt_ms is not None:
            self.params.dt_ms = dt_ms
        if method is not None:
            self.params.method = method
        if t_total_ms is not None:
            self.params.t_total_ms = t_total_ms
        if seed is not None:
            self.params.seed = seed
        self.seed = seed if seed is not None else self.params.seed
        self.name_prefix = name_prefix
        self._m2 = load_synapse_params()          # ampa/gaba/nmda 基础参数（M2 定稿）
        self._release: Optional[Tuple[float, int]] = None     # 全局量子噪声 (p, n)
        self._release_overrides: Dict[Tuple[str, str], Tuple[float, int]] = {}
        self._syn_overrides: Dict[Tuple[str, str], Dict[str, float]] = {}
        self._removed: set = set()                # (frm, to) 消融连接（P5）

        self.neurons: Dict[str, MultiCompartmentNeuron] = {}
        self.chemicals: List[ChemicalSynapse] = []
        self.muscle: Optional[Muscle] = None
        self._built = False

    # ------------------------------------------------------------------ #
    # 协议覆盖（P1/P3/P5 用）
    # ------------------------------------------------------------------ #
    def set_quantum_noise(self, p_release: float, n_vesicles: int,
                          synapse_from: Optional[str] = None,
                          synapse_to: Optional[str] = None) -> "ReflexArc":
        """开启量子释放噪声（默认确定性 p=1/n=1）。

        全局（不指定 from/to）或逐连接覆盖；P1/P3 协议值 p=0.95、n=2。
        """
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

    def set_deterministic(self) -> "ReflexArc":
        """回到确定性模式（p=1/n=1，全部连接）。"""
        self._release = None
        self._release_overrides.clear()
        return self

    def override_synapse(self, synapse_from: str, synapse_to: str,
                         g_max_ns: Optional[float] = None,
                         delay_ms: Optional[float] = None) -> "ReflexArc":
        """调参钩子：覆盖 CSV 中某连接的 g_max_ns / delay_ms（B2 参数扫描）。"""
        key = (synapse_from.upper(), synapse_to.upper())
        over = dict(self._syn_overrides.get(key, {}))
        if g_max_ns is not None:
            over["g_max_ns"] = float(g_max_ns)
        if delay_ms is not None:
            over["delay_ms"] = float(delay_ms)
        self._syn_overrides[key] = over
        return self

    def remove_synapse(self, synapse_from: str, synapse_to: str) -> "ReflexArc":
        """消融：删除某连接（P5 删除 AVM→VB GABA 反证路径）。"""
        self._removed.add((synapse_from.upper(), synapse_to.upper()))
        return self

    def set_touch(self, i0_uA_cm2: Optional[float] = None, start_ms: Optional[float] = None,
                  dur_ms: Optional[float] = None, site: Optional[str] = None) -> "ReflexArc":
        """覆盖触刺激协议（调参/强度扫描）。"""
        t = self.params.touch
        if i0_uA_cm2 is not None:
            t.i0_uA_cm2 = i0_uA_cm2
        if start_ms is not None:
            t.start_ms = start_ms
        if dur_ms is not None:
            t.dur_ms = dur_ms
        if site is not None:
            t.site = site
        return self

    # ------------------------------------------------------------------ #
    # 构建
    # ------------------------------------------------------------------ #
    def _synapse_specs(self) -> List[ReflexSynapseSpec]:
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
            g = over.get("g_max_ns", s.g_max_ns)
            d = over.get("delay_ms", s.delay_ms)
            out.append(ReflexSynapseSpec(
                synapse_from=s.synapse_from, synapse_to=s.synapse_to,
                synapse_type=s.synapse_type, g_max_ns=g, delay_ms=d,
                note=s.note))
        return out

    def _release_for(self, spec: ReflexSynapseSpec) -> Tuple[float, int]:
        """某化学连接的 (p_release, n_vesicles)：确定性默认，除非显式开启噪声。"""
        key = (spec.synapse_from.upper(), spec.synapse_to.upper())
        if key in self._release_overrides:
            return self._release_overrides[key]
        if self._release is not None:
            return self._release
        return (1.0, 1)  # 确定性铁律

    def build(self):
        """重建整条链（每次 run 前自动调用）：神经元 + 化学突触 + 肌肉。"""
        from neural_exploration.src.brian_env import configure_brian2
        from brian2 import start_scope

        p = self.params
        configure_brian2()
        start_scope()

        # 1) 每个角色需要的突触后方程片段（按入边类型聚合）
        post_types: Dict[str, set] = {r: set() for r in p.roles}
        for s in p.synapses:
            if not s.is_muscle:
                post_types[s.synapse_to].add(s.synapse_type)

        # 2) 构建 N 个多隔室神经元（唯一 name + 逐神经元 stim_var 防串扰）
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
        from brian2 import ms

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

        # 4) 肌肉双通道 + 运动神经元驱动（DA→C_back、VB→C_fwd）
        self.muscle = Muscle(tau_ms=p.muscle.tau_ms, cap=p.muscle.cap,
                             name=f"{self.name_prefix}_muscle")
        self.muscle.build()
        w_by_channel = {"back": p.muscle.w_back, "fwd": p.muscle.w_fwd}
        for k, spec in enumerate(self._synapse_specs()):
            if not spec.is_muscle:
                continue
            ch = spec.muscle_channel
            w = spec.g_max_ns if spec.g_max_ns is not None else w_by_channel[ch]
            self.muscle.connect_driver(self.neurons[spec.synapse_from], ch, weight=w,
                                       name=f"{self.name_prefix}_musdrv{k}")
        self._built = True
        return self

    # ------------------------------------------------------------------ #
    # 刺激数组（固定形状 + 显式命名 + namespace 传参，M2 L6 纪律）
    # ------------------------------------------------------------------ #
    def _stim_arrays(self, intensity: float):
        """每角色一个 TimedArray（固定 STIM_WINDOW_MS 形状，显式命名）。

        PLM：触刺激（dend 端，density→nA）；VB：恒定张力注入（soma）；
        其余角色：全零。6 档强度共用同形数组 → 编译缓存命中。
        """
        from brian2 import TimedArray, amp, ms, nA

        p = self.params
        n_steps = int(round(STIM_WINDOW_MS / p.dt_ms))
        arrays = {}
        for role in p.roles:
            neuron = self.neurons[role]
            arr = np.zeros((n_steps, neuron.spec.total_compartments)) * amp
            rl = role.lower()
            if role == "PLM":
                seg, comp = _parse_site(p.touch.site)
                idx = neuron.label_of(seg, comp)
                i0 = int(round(p.touch.start_ms / p.dt_ms))
                i1 = int(round((p.touch.start_ms + p.touch.dur_ms) / p.dt_ms))
                if intensity > 0:
                    arr[i0:i1, idx] = neuron.density_to_nA(
                        p.touch.i0_uA_cm2 * intensity, idx) * nA
            elif role == "VB" and p.tonic_uA_cm2.get("VB", 0.0) > 0:
                idx = neuron.label_of("soma")
                arr[:, idx] = neuron.density_to_nA(p.tonic_uA_cm2["VB"], idx) * nA
            arrays[rl] = TimedArray(arr, dt=p.dt_ms * ms, name=f"stim_{rl}")
        return arrays

    # ------------------------------------------------------------------ #
    # 记录
    # ------------------------------------------------------------------ #
    @staticmethod
    def _default_record(roles: Sequence[str]) -> List[str]:
        return [f"{r.lower()}_soma" for r in roles] + \
               [f"{r.lower()}_node3" for r in roles]

    def _record_spec(self, record: Sequence[str]):
        """'plm_soma' 等标签 → (role, neuron, idx, label) 列表。"""
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
    # 运行
    # ------------------------------------------------------------------ #
    def run(
        self,
        intensity: float = 1.0,
        record: Optional[Sequence[str]] = None,
        t_total_ms: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> ReflexResult:
        """单次运行：重建网络 → 刺激 → 记录 V/发放时刻/肌肉收缩。

        intensity : 触刺激档位（1.0 = 基准 I0；0 = 无刺激对照；其余见 CSV 档位）
        record    : 记录标签（默认每角色 soma+node3）
        """
        from brian2 import Network, SpikeMonitor, StateMonitor, seed as bseed, ms, mV

        self.build()
        bseed(self.seed if seed is None else seed)
        p = self.params
        t_total = t_total_ms or p.t_total_ms
        record = list(record) if record is not None else self._default_record(p.roles)
        rec = self._record_spec(record)
        stims = self._stim_arrays(intensity)

        # 逐角色 StateMonitor（同组多索引；标签按位置对应）
        monos, labels_by_role = {}, {}
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
                  for r, n in self.neurons.items()}
        m_back, m_fwd = self.muscle.monitor(p.dt_ms, name=f"{self.name_prefix}_musc")

        net = Network()
        for n in self.neurons.values():
            net.add(n.neuron)
        for cs in self.chemicals:
            net.add(cs.synapses)
        for g in self.muscle.groups:
            net.add(g)
        for s in self.muscle.drivers:
            net.add(s)
        for m in vmons:
            net.add(m)
        for sp in spmons.values():
            net.add(sp)
        net.add(m_back)
        net.add(m_fwd)

        ns = {f"stim_{rl}": ta for rl, ta in stims.items()}
        net.run(t_total * ms, namespace=ns)

        # 组装结果
        t = np.array(vmons[0].t / ms) if vmons else np.arange(0, t_total, p.dt_ms)
        v = {}
        for role, mon_obj in zip(monos, vmons):
            for pos, lab in enumerate(monos[role]["labels"]):
                v[lab] = np.array(mon_obj.v[pos] / mV)
        spikes: Dict[str, np.ndarray] = {}
        for role, sp in spmons.items():
            spikes.update(self._spike_times(sp, self.neurons[role], role))
        c_back = np.array(m_back.c_back[0])
        c_fwd = np.array(m_fwd.c_fwd[0])

        return ReflexResult(
            t_ms=t, v_mv=v, spike_times_ms=spikes, c_back=c_back, c_fwd=c_fwd,
            meta=self._meta(intensity, t_total, seed if seed is not None else self.seed,
                            c_back, c_fwd),
        )

    def run_trials(
        self,
        intensity: float = 1.0,
        n_trials: int = 5,
        seed_base: int = 0,
        record: Optional[Sequence[str]] = None,
        t_total_ms: Optional[float] = None,
    ) -> List[ReflexResult]:
        """多试次重复运行（P1/P3 量子噪声协议）：store/restore 复用网络，仅重播种。

        每次 restore 后变量回到快照（g 清零、神经元/肌肉复位；monitor 记录亦被
        重置 → 每试次的 monitor 数据即该试次自身的完整轨迹）。
        """
        from brian2 import Network, SpikeMonitor, StateMonitor, seed as bseed, ms, mV

        self.build()
        bseed(seed_base)
        p = self.params
        t_total = t_total_ms or p.t_total_ms
        record = list(record) if record is not None else self._default_record(p.roles)
        rec = self._record_spec(record)
        stims = self._stim_arrays(intensity)

        monos = {}
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
                  for r, n in self.neurons.items()}
        m_back, m_fwd = self.muscle.monitor(p.dt_ms, name=f"{self.name_prefix}_musc")

        net = Network()
        for n in self.neurons.values():
            net.add(n.neuron)
        for cs in self.chemicals:
            net.add(cs.synapses)
        for g in self.muscle.groups:
            net.add(g)
        for s in self.muscle.drivers:
            net.add(s)
        for m in vmons:
            net.add(m)
        for sp in spmons.values():
            net.add(sp)
        net.add(m_back)
        net.add(m_fwd)

        ns = {f"stim_{rl}": ta for rl, ta in stims.items()}
        bseed(seed_base)
        net.run(0 * ms, namespace=ns)   # 完成初始化（快照前；需带 stim namespace 解析单位）
        net.store()          # 保存干净状态（含 monitor 记录 → restore 后 monitor 归零，
                             # 每试次的 monitor 数据即该试次自身的完整轨迹）

        results = []
        for trial in range(n_trials):
            bseed(seed_base + trial)
            net.restore()
            net.run(t_total * ms, namespace=ns)

            t = np.array(vmons[0].t / ms) if vmons else np.arange(0, t_total, p.dt_ms)
            v = {}
            for role, mon_obj in zip(monos, vmons):
                for pos, lab in enumerate(monos[role]["labels"]):
                    v[lab] = np.array(mon_obj.v[pos] / mV)
            spikes: Dict[str, np.ndarray] = {}
            for role, sp in spmons.items():
                t_all = np.array(sp.t / ms)
                i_all = np.array(sp.i)
                for seg in self.neurons[role].spec.segments:
                    idxs = self.neurons[role].index_map[seg.name]
                    if len(idxs):
                        m2 = np.isin(i_all, idxs)
                        spikes[f"{role.lower()}_{seg.name}"] = t_all[m2]
            c_back = np.array(m_back.c_back[0])
            c_fwd = np.array(m_fwd.c_fwd[0])
            results.append(ReflexResult(
                t_ms=t, v_mv=v, spike_times_ms=spikes,
                c_back=c_back, c_fwd=c_fwd,
                meta=self._meta(intensity, t_total, seed_base + trial, c_back, c_fwd,
                                trial=trial),
            ))
        return results

    # ------------------------------------------------------------------ #
    # 元数据 / 内省（B2 验证脚本用）
    # ------------------------------------------------------------------ #
    def _meta(self, intensity, t_total, seed, c_back, c_fwd, trial=None) -> dict:
        p = self.params
        m = dict(
            csv_path=p.csv_path, intensity=intensity, t_total_ms=t_total,
            dt_ms=p.dt_ms, method=p.method, seed=seed,
            touch_start_ms=p.touch.start_ms, touch_dur_ms=p.touch.dur_ms,
            i0_uA_cm2=p.touch.i0_uA_cm2, touch_site=p.touch.site,
            muscle_tau_ms=p.muscle.tau_ms,
            w_back=p.muscle.w_back, w_fwd=p.muscle.w_fwd,
            d_peak=direction_peak(c_back, c_fwd),
            c_back_peak=float(np.max(c_back)), c_fwd_peak=float(np.max(c_fwd)),
            latency_window_ms=list(p.latency_window_ms),
            chalfie_window_ms=list(p.chalfie_window_ms),
        )
        if trial is not None:
            m["trial"] = trial
        return m

    def chain_summary(self) -> dict:
        """拓扑/极性内省（P2 断言：连接数与递质类型与 CSV 规格一致）。"""
        chems = [s for s in self.params.synapses if not s.is_muscle]
        muscs = [s for s in self.params.synapses if s.is_muscle]
        return dict(
            roles=list(self.params.roles),
            n_chemical=len(chems),
            n_muscle_drives=len(muscs),
            synapse_types={f"{s.synapse_from}->{s.synapse_to}": s.synapse_type
                           for s in chems},
            tonic_uA_cm2=dict(self.params.tonic_uA_cm2),
            intensity_levels=list(self.params.intensity_levels),
        )


# --------------------------------------------------------------------- #
# 绘图（reports/neuro/m3_smoke.png）
# --------------------------------------------------------------------- #
def plot_reflex(result: ReflexResult, out_png: Optional[str] = None,
                roles: Sequence[str] = CHAIN_ROLES) -> str:
    """链各级 V（PLM/AVM/DA/VB 子图）+ 双肌肉收缩 C_back/C_fwd 叠加图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reports_dir = os.path.join(ROOT, "neural_exploration", "reports", "neuro")
    os.makedirs(reports_dir, exist_ok=True)
    out_png = out_png or os.path.join(reports_dir, "m3_smoke.png")

    t = result.t_ms
    n = len(roles)
    fig, axes = plt.subplots(n + 1, 1, figsize=(11, 2.4 * (n + 1)), sharex=True)
    touch = result.meta.get("touch_start_ms", 50.0)
    dur = result.meta.get("touch_dur_ms", 8.0)

    colors = {"PLM": "#1f77b4", "AVM": "#ff7f0e", "DA": "#d62728", "VB": "#9467bd"}
    for ax, role in zip(axes[:n], roles):
        lab = f"{role.lower()}_soma"
        if lab in result.v_mv:
            ax.plot(t, result.v_mv[lab], lw=0.9, color=colors.get(role, "k"))
        ax.axhline(-20.0, color="gray", ls="--", lw=0.7)
        ax.axvspan(touch, touch + dur, color="orange", alpha=0.15)
        spikes = result.spikes(role, "node3")
        for tk in spikes:
            ax.axvline(tk, color=colors.get(role, "k"), ls=":", lw=0.8, alpha=0.6)
        ax.set_ylabel(f"{role} V (mV)")
        ax.set_ylim(-90, 60)
        ax.grid(alpha=0.3)
        ax.set_title(f"{role}: {len(spikes)} spikes @ node3", fontsize=9, loc="left")

    axm = axes[n]
    axm.plot(t, result.c_back, lw=1.4, color="#2ca02c", label="C_back (DA -> withdraw)")
    axm.plot(t, result.c_fwd, lw=1.4, color="#8c564b", label="C_fwd (VB -> forward)")
    axm.axvspan(touch, touch + dur, color="orange", alpha=0.15)
    axm.set_ylabel("muscle C")
    axm.set_xlabel("t (ms)")
    axm.set_title(f"D_peak = {result.d_peak:.3f} (C_back_peak={result.c_back_peak:.3f}, "
                  f"C_fwd_peak={result.c_fwd_peak:.3f})", fontsize=9, loc="left")
    axm.legend(loc="upper right", fontsize=8)
    axm.grid(alpha=0.3)
    axm.set_ylim(0, 1.05)

    fig.suptitle("M3 reflex arc: touch → PLM → AVM → DA (back) / GABA→VB (fwd)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="M3 反射弧单次运行 + 出图")
    ap.add_argument("--intensity", type=float, default=1.0)
    ap.add_argument("--t-total-ms", type=float, default=None)
    ap.add_argument("--noplot", action="store_true")
    args = ap.parse_args()

    arc = ReflexArc()
    r = arc.run(intensity=args.intensity, t_total_ms=args.t_total_ms)
    print(f"intensity={args.intensity}  d_peak={r.d_peak:.3f}  "
          f"C_back_peak={r.c_back_peak:.3f}  C_fwd_peak={r.c_fwd_peak:.3f}")
    for role in ("PLM", "AVM", "DA", "VB"):
        st = r.spikes(role, "node3")
        if len(st):
            print(f"  {role}: {len(st)} spikes @ node3, first={st[0]:.2f} ms")
        else:
            print(f"  {role}: 0 spikes")
    if not args.noplot:
        png = plot_reflex(r)
        print(f"图已生成: {png}")
