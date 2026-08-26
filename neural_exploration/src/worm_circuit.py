"""M5 可缩放全虫骨架：WormCircuit（连接组驱动，规模轴 20/50/100/302 × 保真度轴
点神经元/双隔室/多隔室 HH）。

用法（G0 缩放扫描，tools/scan_m5_scaling.py）：
    wc = WormCircuit(scale=20, fidelity="point")   # 读 data/m5_connectome.csv（B1a 产出，
                                                   # 运行期 wait_for_csv 轮询）；未就绪时
                                                   # 回退 data/m4_chemotaxis_params.csv
                                                   # 20 角色趋化子图骨架（M4 冻结参数源）
    wc.build()
    ci_list, meta = wc.run_chemotaxis_trials(n=3, t_total_ms=5000, seed_base=0)
    esc = wc.run_escape()                          # M3 反射子图（同保真度）方向/潜伏期
    rates = wc.run_resting(t_total_ms=1000)        # 无刺激发放率分布
    states = wc.run_spontaneous(t_total_ms=5000)   # 自发状态比例

组件复用（冻结文件零修改，清单 L1）：
  - 神经元：PointNeuron / TwoCompartmentNeuron（本节点，src/point_neuron.py）或
    MultiCompartmentNeuron（M1 冻结，HH 档仅 ≤50 子图）；
  - 突触：M2 冻结 ChemicalSynapse / GapJunction（PointNeuron 薄包装适配，实测复用成功）；
  - 肌肉：M4 冻结 Muscle3（四通道或三通道）；
  - 环境/身体/CI/统计：M4 冻结 ChemotaxisEnv / ChemotaxisBody / TimeDiffTracker /
    ci_group_stats（两引擎共用同一统计代码——行为参考可比性保证）。

确定性：p=1/n=1、无噪声；试次方差来自伪随机起点（M4 纪律）；同参数重跑逐位一致。
构造参数默认 None（M3 L13）：None → 以数据文件为准。
编译缓存纪律（M4 L16）：dt/形状/命名定稿后不变；stim TimedArray 固定形状 +
显式命名；多进程并发 ≤2 worker（M4 L21）。

连接组模式：data/m5_connectome.csv（B1a 唯一定稿源，schema 见 M5 清单 §2.2）——
神经元行/化学突触行/缝隙行/肌肉行；规模轴按拓扑序（sensory→inter→motor→pharyngeal）
取子集；20 档优先取 M4 趋化子图 roster（若在连接组中存在）。
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.src.chemotaxis_body import ChemotaxisBody  # noqa: E402
from neural_exploration.src.chemotaxis_circuit import (  # noqa: E402
    ChemotaxisCircuit,
    ChemotaxisResult,
    ChemoSession,
    ChemotaxisParams,
    ChemoSynapseSpec,
    load_chemotaxis_params,
)
from neural_exploration.src.chemotaxis_env import (  # noqa: E402
    ChemotaxisEnv,
    TimeDiffTracker,
    ci_group_stats,
)
from neural_exploration.src.neuron_model import MultiCompartmentNeuron  # noqa: E402
from neural_exploration.src.point_neuron import (  # noqa: E402
    PointNeuron,
    TwoCompartmentNeuron,
)
from neural_exploration.src.synapse_model import (  # noqa: E402
    GapJunction,
    SynapseParams,
    chemical_im_terms,
    chemical_post_eqs,
    load_synapse_params,
)

DEFAULT_CONNECTOME_CSV = os.path.join(ROOT, "neural_exploration", "data",
                                      "m5_connectome.csv")
DEFAULT_M4_PARAMS_CSV = os.path.join(ROOT, "neural_exploration", "data",
                                     "m4_chemotaxis_params.csv")
DEFAULT_M3_PARAMS_CSV = os.path.join(ROOT, "neural_exploration", "data",
                                     "m3_reflex_params.csv")
STIM_WINDOW_MS = 500.0        # 固定形状下限（M3/M4 惯例）
#: 协议窗口（固定 stim 形状上限，ms）：探针/趋化/静息/自发共用同一 TimedArray 形状
#: （M4 L16：形状变化触发全量重编译——窗口内任意 T 复用同一编译产物，L7 实测）。
#: B1e 处置（m5_env_notes L27 请求）：G0 定稿 P4 T=15s、P6 T=30s > 6000ms →
#: 扩展窗口至 30000ms（覆盖最长自发协议；一次性冷编译预算预注册 §8，
#: grouped 模式 302 实测 ~10min 冷编译 / 稳态 ~6.6s，见 m5_scaling.csv build_wall_s）。
#: 20000 <= T < 30000 的协议仍复用同一编译产物；shape 定稿后不再变（M4 L16）。
PROTOCOL_WINDOW_MS = 30000.0

#: 规模轴（清单 §3.3：20=M4 趋化子图；50/100=命令/运动上下文扩展；302=全连接组）
SCALE_AXIS = (20, 50, 100, 302)
#: 保真度轴（清单 §3.3；dt 并入保真度档，不做独立网格）
FIDELITY_AXIS = ("point", "two_comp", "multicomp")
#: 每保真度档固定 dt（ms）与 method（本节点实测定稿，docs/m5_env_notes.md L7）
FIDELITY_DT = {"point": (0.1, "exponential_euler"),
               "two_comp": (0.05, "exponential_euler"),
               "multicomp": (0.01, "rk4")}

#: 类级缩放桶（§6 权重策略 #2：按 (pre 类, post 类) 分桶，每桶一个 s_k，
#: w_ij = w0_class · s_k；w0_class = 连接组 g_max_ns 占位 = M3/M4 子图先验
#: （ampa 5.0nS / gaba 15.0nS，L4）。连接组 302 中**非空化学类对**（B1e 实测，
#: 见 m5_connectome_counts 类对统计；mod/none 调质占位跳过不参与缩放）。
CLASS_PAIRS = (
    ("sensory", "sensory"), ("sensory", "inter"), ("sensory", "motor"),
    ("inter", "inter"), ("inter", "motor"), ("inter", "sensory"),
    ("motor", "inter"), ("motor", "motor"), ("motor", "sensory"),
    ("pharyngeal", "pharyngeal"),
)

#: 默认类级缩放（全部 1.0 = 占位权重恒等；定稿值写入 data/m5_worm_params.csv
#: weight 行 role=weight, neuron_class=class_scale_<pre>_<post>, value=s_k）
DEFAULT_CLASS_SCALES: Dict[Tuple[str, str], float] = {
    pair: 1.0 for pair in CLASS_PAIRS}


def load_class_scales(csv_path: Optional[str] = None
                      ) -> Dict[Tuple[str, str], float]:
    """读 data/m5_worm_params.csv 的类级缩放定稿（§6 定稿消费入口）。

    - weight 行：role=weight, neuron_class=class_scale_<pre>_<post>, value=s_k
      （列 schema 同 m5_worm_params.csv：value 在 fields[9]，L23 位置解析语义）；
    - 缺失桶 → 1.0（恒等）；非 class_scale_* 的 weight 行忽略。
    - 供 P2/P4/P5/P6 验证脚本消费：``cs = load_class_scales()`` →
      ``make_worm_circuit(scale=302, class_scales=cs)``（行为层降阶配置 G0）。
    """
    from neural_exploration.src.worm_loop import load_m5_worm_params

    wp = load_m5_worm_params(csv_path)
    out = dict(DEFAULT_CLASS_SCALES)
    for key, val in wp.get("weight", {}).items():
        if key.startswith("class_scale_") and isinstance(val, (int, float)):
            rest = key[len("class_scale_"):]     # "<pre>_<post>"
            pre, _, post = rest.rpartition("_")
            if (pre, post) in out:
                out[(pre, post)] = float(val)
    return out


def load_weight_scales(csv_path: Optional[str] = None) -> Dict[str, dict]:
    """读 data/m5_worm_params.csv 的 §6 权重定稿全量（类级 + 缝隙 + 突触类型）。

    返回 {"class_scales": {(pre,post): s_k}, "gap_scale": float,
          "syn_type_scales": {type: s_t}, "tonic_scale": float,
          "gL_scale": float}——可直接
    ``make_worm_circuit(scale=302, **load_weight_scales())``（M5-B1e2 校准定稿
    消费入口；行为层 P2/P4/P5/P6 验证脚本统一用它，保证 CSV 为唯一定稿源）。
    列 schema 同 m5_worm_params.csv（value 在 fields[9]，L23 位置解析语义）：
      - weight 行 neuron_class=class_scale_<pre>_<post> → 类级缩放 s_k；
      - weight 行 neuron_class=gap_scale → 缝隙全局缩放（默认 1.0）；
      - weight 行 neuron_class=syn_type_scale_<ampa|gaba|nmda> → 突触类型缩放
        （默认 1.0；M5-B1e2 校准扩展，API 兼容——缺省恒等）；
      - weight 行 neuron_class=tonic_scale → AVB 张力缩放（默认 1.0，同上）；
      - weight 行 neuron_class=gL_scale → 点神经元漏电缩放（默认 1.0，同上）。
    """
    from neural_exploration.src.worm_loop import load_m5_worm_params

    wp = load_m5_worm_params(csv_path)
    class_scales = dict(DEFAULT_CLASS_SCALES)
    gap_scale = 1.0
    syn_type_scales: Dict[str, float] = {}
    tonic_scale = 1.0
    gL_scale = 1.0
    for key, val in wp.get("weight", {}).items():
        if not isinstance(val, (int, float)):
            continue
        if key.startswith("class_scale_"):
            rest = key[len("class_scale_"):]
            pre, _, post = rest.rpartition("_")
            if (pre, post) in class_scales:
                class_scales[(pre, post)] = float(val)
        elif key == "gap_scale":
            gap_scale = float(val)
        elif key == "tonic_scale":
            tonic_scale = float(val)
        elif key == "gL_scale":
            gL_scale = float(val)
        elif key.startswith("syn_type_scale_"):
            stype = key[len("syn_type_scale_"):]
            if stype in ("ampa", "gaba", "nmda"):
                syn_type_scales[stype] = float(val)
    return {"class_scales": class_scales, "gap_scale": gap_scale,
            "syn_type_scales": syn_type_scales, "tonic_scale": tonic_scale,
            "gL_scale": gL_scale}

#: M4 趋化子图 20 角色 roster（连接组模式下 20 档优先取此集合；M4 冻结参数源的角色序）
M4_ROSTER = ("ASEL", "ASER", "AIYL", "AIYR", "AIBL", "AIBR", "RIAL", "RIAR",
             "AVBL", "AVBR", "SMDDL", "SMDDR", "SMDVL", "SMDVR", "VB", "DB",
             "AVAL", "AVAR", "RMED", "RMEV")


# --------------------------------------------------------------------- #
# 连接组规格
# --------------------------------------------------------------------- #
@dataclass
class ChemRow:
    pre: str
    post: str
    syn_type: str          # ampa|gaba|nmda
    g_ns: float
    delay_ms: float


@dataclass
class GapRow:
    a: str
    b: str
    g_ns: float


@dataclass
class MuscleRow:
    motor: str
    channel: str
    w: float


@dataclass
class ConnectomeSpec:
    """连接组规格（唯一定稿源解析产物，M5 清单 §2.2 schema）。"""

    neurons: "OrderedDict[str, str]" = field(default_factory=dict)  # name -> neuron_class
    chem: List[ChemRow] = field(default_factory=list)
    gaps: List[GapRow] = field(default_factory=list)
    muscles: List[MuscleRow] = field(default_factory=list)
    tonic_uA_cm2: Dict[str, float] = field(default_factory=dict)
    source: str = ""

    @property
    def n_neurons(self) -> int:
        return len(self.neurons)

    @property
    def n_chem(self) -> int:
        return len(self.chem)

    @property
    def n_gap(self) -> int:
        return len(self.gaps)

    def subset(self, names: Sequence[str]) -> "ConnectomeSpec":
        """按名称集合取子集（连接/缝隙/肌肉只保留两端均在子集内的）。"""
        keep = set(names)
        return ConnectomeSpec(
            neurons={n: c for n, c in self.neurons.items() if n in keep},
            chem=[r for r in self.chem if r.pre in keep and r.post in keep],
            gaps=[r for r in self.gaps if r.a in keep and r.b in keep],
            muscles=[r for r in self.muscles if r.motor in keep],
            tonic_uA_cm2={n: v for n, v in self.tonic_uA_cm2.items()
                          if n in keep},
            source=self.source,
        )


def wait_for_connectome(csv_path: Optional[str] = None, timeout_s: float = 600.0,
                        interval_s: float = 10.0) -> str:
    """轮询等待 B1a 的 m5_connectome.csv（运行期读取；超时抛 FileNotFoundError）。"""
    path = csv_path or DEFAULT_CONNECTOME_CSV
    t0 = time.time()
    while not os.path.exists(path):
        if time.time() - t0 > timeout_s:
            raise FileNotFoundError(
                f"等待 {timeout_s:.0f}s 后 m5_connectome.csv 仍未生成：{path}")
        time.sleep(interval_s)
    return path


def load_connectome(csv_path: Optional[str] = None, poll_s: float = 0.0,
                    timeout_s: float = 600.0) -> ConnectomeSpec:
    """读入连接组规格。

    - data/m5_connectome.csv 存在 → 按 M5 清单 §2.2 schema 解析（B1a 唯一定稿源；
      poll_s>0 时轮询等待）；
    - 否则回退 data/m4_chemotaxis_params.csv 的 20 角色趋化子图骨架
      （M4 冻结参数源；任务规定：CSV 到达后切换）。
    """
    path = csv_path or DEFAULT_CONNECTOME_CSV
    if not os.path.exists(path) and poll_s > 0:
        try:
            path = wait_for_connectome(path, timeout_s=timeout_s)
        except FileNotFoundError:
            path = DEFAULT_CONNECTOME_CSV
    if os.path.exists(path) and os.path.abspath(path) != os.path.abspath(DEFAULT_M4_PARAMS_CSV):
        return _parse_connectome_csv(path)
    return _fallback_from_m4(DEFAULT_M4_PARAMS_CSV)


def _parse_connectome_csv(path: str) -> ConnectomeSpec:
    """解析 m5_connectome.csv（B1a 唯一定稿源，schema：M5 清单 §2.2 实测列
    role, neuron_class, neurotransmitter, receptor, synapse_from, synapse_to,
    synapse_type, g_max_ns, delay_ms, g_gap_ns, muscle_target, note）。

    - 神经元行：role=<名>（≠muscle_drive）；
    - 化学行：synapse_type=chem + receptor ∈ {ampa, gaba}（§2.3 映射：ach/glut→ampa、
      gaba→gaba；mod/none = 调质占位/无受体 → 本档跳过并计数，L7 记录）；
    - 缝隙行：synapse_type=gap + g_gap_ns；
    - 肌肉行：role=muscle_drive + synapse_to ∈ {body_fwd, body_back, head_left,
      head_right} → 通道 fwd/back/left/right。
    """
    import csv as _csv

    spec = ConnectomeSpec(source=path)
    skipped_mod_none = 0

    def _clean_line(ln: str) -> str:
        """去注释/头行外层引号：子图 CSV（B1a）的列头行为 `"role,...,note"`
        带引号——直接过滤会把列头行丢掉 → DictReader 误把首数据行当列头
        （M5-B1d 实测，L23）；主 connectome CSV 列头无引号不受影响。"""
        s = ln.strip()
        if s.startswith('"'):
            s = s.strip('"')
        return s

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(_csv.DictReader(
            _clean_line(ln) for ln in f
            if _clean_line(ln) and not _clean_line(ln).startswith("#")))
    for r in rows:
        role = (r.get("role") or "").strip().upper()
        frm = (r.get("synapse_from") or "").strip().upper()
        stype = (r.get("synapse_type") or "").strip().lower()
        if role and role != "MUSCLE_DRIVE":
            cls = (r.get("neuron_class") or "inter").strip().lower()
            spec.neurons[role] = cls
        if stype == "chem" and frm:
            receptor = (r.get("receptor") or "").strip().lower()
            if not receptor:
                # 子图 CSV（m5_pharynx/command/chemotaxis_subgraph.csv）无 receptor 列
                # → 按 L4 递质→受体映射回退（ach/glut→ampa、gaba→gaba、调质→mod）——
                # M5-B1d 微调（API 兼容：主 connectome CSV 有 receptor 列不受影响，L23）
                receptor = {"ach": "ampa", "glut": "ampa", "gaba": "gaba",
                            "dopamine": "mod", "serotonin": "mod",
                            "other": "none"}.get(
                                (r.get("neurotransmitter") or "").strip().lower(),
                                "none")
            if receptor in ("ampa", "gaba"):
                spec.chem.append(ChemRow(
                    pre=frm, post=(r.get("synapse_to") or "").strip().upper(),
                    syn_type=receptor,
                    g_ns=float(r.get("g_max_ns") or 5.0),
                    delay_ms=float(r.get("delay_ms") or 0.5)))
            else:
                skipped_mod_none += 1
        elif stype == "gap" and frm:
            spec.gaps.append(GapRow(
                a=frm, b=(r.get("synapse_to") or "").strip().upper(),
                g_ns=float(r.get("g_gap_ns") or r.get("g_max_ns") or 0.5)))
        elif stype == "muscle" and frm:
            ch = (r.get("synapse_to") or "").strip().lower()
            for prefix in ("body_", "head_"):
                if ch.startswith(prefix):
                    ch = ch[len(prefix):]
                    break
            spec.muscles.append(MuscleRow(motor=frm, channel=ch,
                                          w=float(r.get("g_max_ns") or 0.3)))
    spec.n_skipped_mod_none = skipped_mod_none
    return spec


def _fallback_from_m4(csv_path: str) -> ConnectomeSpec:
    """M4 趋化子图骨架（20 角色 + 化学突触 + 肌肉驱动 + 张力；M4 冻结参数源）。"""
    p = load_chemotaxis_params(csv_path)
    spec = ConnectomeSpec(source=csv_path)
    for role in p.roles:
        spec.neurons[role] = "inter"
    for s in p.synapses:
        if s.is_muscle:
            spec.muscles.append(MuscleRow(motor=s.synapse_from,
                                          channel=s.muscle_channel,
                                          w=float(s.g_max_ns or 0.3)))
        else:
            spec.chem.append(ChemRow(pre=s.synapse_from, post=s.synapse_to,
                                     syn_type=s.synapse_type,
                                     g_ns=float(s.g_max_ns or 5.0),
                                     delay_ms=float(s.delay_ms or 0.5)))
    for r, v in p.tonic_uA_cm2.items():
        if v > 0:
            spec.tonic_uA_cm2[r] = v
    return spec


def _topo_order(spec: ConnectomeSpec) -> List[str]:
    """拓扑序（sensory → inter → motor → pharyngeal；类内保持 CSV/M4 顺序）。

    M4 骨架：按 M4 参数源角色序（roster 序）。
    """
    order = []
    for cls in ("sensory", "inter", "motor", "pharyngeal"):
        order += [n for n, c in spec.neurons.items() if c == cls]
    for n in spec.neurons:
        if n not in order:
            order.append(n)
    return order


def scale_names(spec: ConnectomeSpec, scale: int) -> List[str]:
    """规模轴子集（20=M4 趋化子图在连接组中的对应；其余按拓扑序取前 n）。

    - 20 档：优先取 M4 趋化子图 roster 中存在于连接组的角色（M4 的 VB/DB 在
      Cook 连接组中为 VB1..VB11/DB1..DB7——M4 无名简化，取拓扑序首个 VB*/DB*
      补足 20），再按拓扑序补足；
    - 50/100/302：拓扑序（sensory→inter→motor→pharyngeal）取前 n。
    """
    order = _topo_order(spec)
    if scale == 20:
        m4_matched = [r for r in M4_ROSTER if r in spec.neurons]
        if len(m4_matched) >= 18:
            # M4 的 VB/DB（无名简化）→ Cook 连接组 VB1..VB11 / DB1..DB7：
            # 前进命令驱动必须入子集（否则 20 档无 C_fwd 驱动，行为不完整，L7）
            pref = ([n for n in order if n.startswith("VB")]
                    + [n for n in order if n.startswith("DB")])
            filled = [n for n in pref if n not in m4_matched]
            need = scale - len(m4_matched)
            if len(filled) < need:
                filled += [n for n in order if n not in m4_matched
                           and n not in filled][:need - len(filled)]
            return m4_matched + filled[:need]
    if scale in (50, 100) and scale < len(order):
        # 类平衡分层抽样（L7 实测：纯拓扑序 50/100 无运动神经元 → 无 C_fwd、
        # CI 失真）：按 302 类构成比例（sensory≈27%/inter≈28%/motor≈38%/
        # pharyngeal≈7%）每类配额，保证含运动神经元 → 可爬行。
        classes = ("sensory", "inter", "motor", "pharyngeal")
        per = {c: [n for n in order if spec.neurons.get(n) == c]
               for c in classes}
        total = len(order)
        quota = {c: max(1, round(scale * len(v) / total))
                 for c, v in per.items()}
        while sum(quota.values()) < scale:
            for c in classes:
                if sum(quota.values()) >= scale:
                    break
                if len(per[c]) > quota[c]:
                    quota[c] += 1
        out = []
        for c in classes:
            out += per[c][:quota[c]]
        return out[:scale]
    if scale > len(order):
        return list(order)
    return order[:scale]


# --------------------------------------------------------------------- #
# WormCircuit：可缩放组装
# --------------------------------------------------------------------- #
class WormCircuit:
    """全虫电路骨架（连接组驱动 + 规模/保真度轴 + 趋化闭环/机械逃避/静息协议）。

    构造参数默认 None（M3 L13）：None → 以数据文件为准。
    """

    def __init__(
        self,
        csv_path: Optional[str] = None,
        dt_ms: Optional[float] = None,
        method: Optional[str] = None,
        t_total_ms: Optional[float] = None,
        seed: Optional[int] = None,
        name_prefix: str = "worm",
        scale: int = 20,
        fidelity: str = "point",
        connectome_poll_s: float = 0.0,
        connectome_timeout_s: float = 600.0,
        muscle_tau_ms: Optional[float] = None,
        muscle_cap: Optional[float] = 1.0,
        gap_mode: str = "auto",
        class_scales: Optional[Dict[Tuple[str, str], float]] = None,
        gap_scale: Optional[float] = None,
        syn_type_scales: Optional[Dict[str, float]] = None,
        tonic_scale: Optional[float] = None,
        gL_scale: Optional[float] = None,
    ):
        if scale not in SCALE_AXIS:
            raise ValueError(f"规模需为 {SCALE_AXIS}：{scale}")
        if fidelity not in FIDELITY_AXIS:
            raise ValueError(f"保真度需为 {FIDELITY_AXIS}：{fidelity}")
        if gap_mode not in ("auto", "component", "grouped", "none"):
            raise ValueError(f"gap_mode 需为 auto/component/grouped/none：{gap_mode}")
        self.scale = scale
        self.fidelity = fidelity
        self.name_prefix = name_prefix
        self.seed = 0 if seed is None else int(seed)
        self.connectome_poll_s = connectome_poll_s
        self.connectome_timeout_s = connectome_timeout_s
        # 类级缩放（§6 权重策略 #2；None → 恒等占位权重，行为不变——API 兼容）
        self.class_scales: Dict[Tuple[str, str], float] = dict(
            DEFAULT_CLASS_SCALES if class_scales is None else class_scales)
        self.gap_scale = 1.0 if gap_scale is None else float(gap_scale)
        # 突触类型缩放（M5-B1e2 校准扩展：syn_type_scales={"ampa":s_a,"gaba":s_g}，
        # 在类级缩放后额外乘；None → 恒等——API 兼容，默认行为不变）。
        # 依据：连接组 gaba 化学突触仅 129/2472（L20），占位 gaba=15nS 相对
        # ampa=5nS 仅 3×，静息过兴奋的"抑制不足"假说需要独立 gaba 杠杆（§6 校准）。
        self.syn_type_scales: Dict[str, float] = dict(
            syn_type_scales or {})
        # 张力缩放（M5-B1e2 校准扩展：M4 定稿 tonic=14µA/cm²（AVBL/AVBR）在 302
        # 全网络下把 AVB 推成 14-27Hz 网络级夹带振荡引擎——实测 86% 神经元同步
        # 13.8Hz（静息 P2 静默比例结构性不可达）。tonic_scale 把 AVB 降至 ~3Hz
        # 目标（C_fwd 基线 ≈0.2 参考值所需 VB/DB 总率 ~55Hz）；None → 1.0 恒等，
        # API 兼容默认行为不变）
        self.tonic_scale = 1.0 if tonic_scale is None else float(tonic_scale)
        # 漏电缩放（风险表"漏电增强"杠杆；None → 1.0 恒等，API 兼容）
        self.gL_scale = 1.0 if gL_scale is None else float(gL_scale)

        # 连接组规格（运行期读取；缺省回退 M4 骨架）
        self.spec: ConnectomeSpec = load_connectome(
            csv_path, poll_s=connectome_poll_s, timeout_s=connectome_timeout_s)
        self.names = scale_names(self.spec, scale)
        self.sub: ConnectomeSpec = self.spec.subset(self.names)

        # 缝隙模式（L7 实测：M2 GapJunction (summed) 在 ≥2 缝隙/神经元时 Brian2
        # 报 "Multiple summed variables"——真实连接组多缝隙拓扑只能用 grouped 或 none；
        # component 仅用于单对语义验证）
        self.is_connectome = os.path.abspath(self.spec.source) != \
            os.path.abspath(DEFAULT_M4_PARAMS_CSV)
        if gap_mode == "auto":
            self.gap_mode = "component" if (not self.is_connectome
                                            and self.sub.gaps) else "none"
        else:
            self.gap_mode = gap_mode

        # 行为上下文参数（M4 唯一定稿源；M5 定稿于 m5_worm_params.csv 后切换）
        self.params: ChemotaxisParams = load_chemotaxis_params(DEFAULT_M4_PARAMS_CSV)
        # 张力携带（连接组模式）：M4 行为上下文（AVBL/AVBR 14µA/cm² 维持前进基线）
        # 应用到连接组同名角色——m5_worm_params.csv 落盘前的行为上下文携带（L7 记录）
        if not self.sub.tonic_uA_cm2 and os.path.abspath(self.spec.source) != \
                os.path.abspath(DEFAULT_M4_PARAMS_CSV):
            for r, v in self.params.tonic_uA_cm2.items():
                if r in self.sub.neurons and v > 0:
                    self.sub.tonic_uA_cm2[r] = v * self.tonic_scale
        if dt_ms is not None:
            self.params.dt_ms = dt_ms
        if method is not None:
            self.params.method = method
        if t_total_ms is not None:
            self.params.t_total_ms = t_total_ms
            self.params.protocol.t_total_ms = t_total_ms
        self.dt_ms = FIDELITY_DT[fidelity][0] if dt_ms is None else float(dt_ms)
        self.method = FIDELITY_DT[fidelity][1] if method is None else method
        if fidelity == "multicomp":
            # HH 档仅 ≤50 子图（清单 §3.2 方案③）
            if scale > 50:
                raise ValueError(
                    f"多隔室 HH 档仅限 ≤50 子图（规模 {scale} 不可用）："
                    f"降阶正确性要求 HH 只做局部子图（§3.4）")
        self.muscle_tau_ms = (self.params.muscle_tau_ms if muscle_tau_ms is None
                              else float(muscle_tau_ms))
        self.muscle_cap = 1.0 if muscle_cap is None else float(muscle_cap)
        self._m2 = load_synapse_params()
        self._built = False
        self.neurons: Dict[str, object] = {}
        self.chemicals: List[object] = []
        self.gaps: List[object] = []
        self.muscle3 = None
        self._post_types: Dict[str, set] = {}
        self._build_wall_s = float("nan")

    # ------------------------------------------------------------------ #
    # 组装
    # ------------------------------------------------------------------ #
    def _role_post_types(self) -> Dict[str, set]:
        post: Dict[str, set] = {n: set() for n in self.names}
        for r in self.sub.chem:
            post[r.post].add(r.syn_type)
        return post

    def _class_scale_for(self, pre: str, post: str) -> float:
        """类级缩放 s_k(bucket)：w_ij = w0_class · s_k（§6 权重策略 #2）。

        桶 = (pre 类, post 类)（四类 sensory/inter/motor/pharyngeal）；
        缺省/未知类对 → 1.0（恒等）。
        """
        return self.class_scales.get(
            (self.sub.neurons.get(pre, ""), self.sub.neurons.get(post, "")),
            1.0)

    def _make_neuron(self, role: str, extra_eqs: str, extra_im: str,
                     stim_var: str) -> object:
        if self.fidelity == "point":
            return PointNeuron(
                name=f"{self.name_prefix}_{role.lower()}", dt_ms=self.dt_ms,
                method=self.method, extra_eqs=extra_eqs,
                extra_im_terms=extra_im, stim_var=stim_var).build()
        if self.fidelity == "two_comp":
            return TwoCompartmentNeuron(
                name=f"{self.name_prefix}_{role.lower()}", dt_ms=self.dt_ms,
                method=self.method, extra_eqs=extra_eqs,
                extra_im_terms=extra_im, stim_var=stim_var).build()
        return MultiCompartmentNeuron(
            name=f"{self.name_prefix}_{role.lower()}", dt_ms=self.dt_ms,
            method=self.method, t_total_ms=self.params.t_total_ms,
            extra_eqs=extra_eqs, extra_im_terms=extra_im,
            stim_var=stim_var).build()

    def build(self):
        """组装：神经元 + 化学突触 + 缝隙 + 肌肉（每次会话前自动调用）。"""
        from neural_exploration.src.brian_env import configure_brian2
        from brian2 import ms, start_scope

        configure_brian2()
        start_scope()
        t0 = time.perf_counter()

        self._post_types = self._role_post_types()
        self.neurons = {}
        for role in self.names:
            types = self._post_types.get(role, set())
            sub = {t: self._m2[t] for t in types}
            eqs = chemical_post_eqs(sub)
            ims = chemical_im_terms(sub)
            stim_var = f"stim_{role.lower()}"
            self.neurons[role] = self._make_neuron(role, eqs, ims, stim_var)

        self.chemicals = []
        for k, r in enumerate(self.sub.chem):
            base = self._m2[r.syn_type]
            sp = SynapseParams(
                synapse_type=r.syn_type, g_max_ns=r.g_ns
                * self._class_scale_for(r.pre, r.post)
                * self.syn_type_scales.get(r.syn_type, 1.0),
                tau_ms=base.tau_ms, e_rev_mv=base.e_rev_mv,
                p_release=1.0, n_vesicles=1,          # 确定性铁律
                mg_mm=base.mg_mm, u0=base.u0,
                tau_fac_ms=base.tau_fac_ms, tau_rec_ms=base.tau_rec_ms,
            )
            from neural_exploration.src.synapse_model import ChemicalSynapse
            cs = ChemicalSynapse(
                self.neurons[r.pre], self.neurons[r.post], sp,
                pre_site="node3", post_site="soma",
                name=f"{self.name_prefix}_syn{k}_{r.syn_type}")
            cs.build()
            cs.synapses.delay = r.delay_ms * ms
            self.chemicals.append(cs)

        self.gaps = []
        for k, g in enumerate(self.sub.gaps):
            if self.gap_mode == "none":
                break
            gj = GapJunction(self.neurons[g.a], self.neurons[g.b],
                             g_gap_ns=g.g_ns * self.gap_scale,
                             name=f"{self.name_prefix}_gap{k}")
            gj.build()
            self.gaps.append(gj)

        # 肌肉通道：按肌肉行出现的通道（趋化 fwd/left/right；逃避 back/fwd）
        channels = list(dict.fromkeys(m.channel for m in self.sub.muscles))
        if not channels:
            channels = ["fwd", "left", "right"]
        from neural_exploration.src.chemotaxis_circuit import Muscle3
        self.muscle3 = Muscle3(tau_ms=self.muscle_tau_ms, cap=self.muscle_cap,
                               channels=channels,
                               name=f"{self.name_prefix}_muscle3")
        self.muscle3.build()
        for k, m in enumerate(self.sub.muscles):
            self.muscle3.connect_driver(self.neurons[m.motor], m.channel,
                                        weight=m.w,
                                        name=f"{self.name_prefix}_musdrv{k}")
        self._built = True
        self._build_wall_s = time.perf_counter() - t0
        return self

    # ------------------------------------------------------------------ #
    # 会话（趋化闭环）
    # ------------------------------------------------------------------ #
    @property
    def tonic_roles(self) -> Tuple[str, ...]:
        return tuple(self.sub.tonic_uA_cm2.keys())

    @property
    def sens_roles(self) -> Tuple[str, str]:
        """化学转导感觉对（默认 ASEL/ASER；连接组模式可按 CSV 配置）。

        按 self.names 判定（grouped 模式不填充 self.neurons——L7 实测坑）。
        """
        if "ASEL" in self.names and "ASER" in self.names:
            return ("ASEL", "ASER")
        return ("", "")

    def _n_comp(self, role: str) -> int:
        n = self.neurons[role].neuron.N
        return int(n)

    def _stim_n_steps(self, t_total_ms: float) -> int:
        """stim 数组步数：固定协议窗口形状（M4 L16 编译缓存纪律），但多隔室
        HH（dt=0.01）内存约束 → 按 t_total 取形（该档每会话只跑单一协议）。"""
        if self.dt_ms >= 0.05:
            return int(round(max(STIM_WINDOW_MS, PROTOCOL_WINDOW_MS)
                             / self.dt_ms))
        return int(round(max(STIM_WINDOW_MS, t_total_ms) / self.dt_ms))

    def _stim_arrays(self, t_total_ms: float) -> Dict[str, object]:
        """每角色一个 TimedArray（固定形状 + 显式命名，M2 L6/M4 L12 纪律）。

        受激角色（感觉对 + 张力）：全试次形状 (n_steps, n_comp)；
        其余：(1, n_comp) 零数组（越界钳位 → 恒 0，极小内存）。
        """
        from brian2 import TimedArray, amp, ms

        p = self.params
        n_steps = self._stim_n_steps(t_total_ms)
        stim_roles = set(self.sens_roles) | set(self.tonic_roles)
        arrays = {}
        for role in self.names:
            n_comp = self._n_comp(role)
            if role in stim_roles:
                arr = np.zeros((n_steps, n_comp)) * amp
            else:
                arr = np.zeros((1, n_comp)) * amp
            arrays[role] = TimedArray(arr, dt=self.dt_ms * ms,
                                      name=f"stim_{role.lower()}")
        return arrays

    def _nA_per_density(self, role: str, idx: int) -> float:
        return self.neurons[role].density_to_nA(1.0, idx)

    def make_session(self, t_total_ms: Optional[float] = None,
                     record: Optional[Sequence[str]] = None,
                     stimulated_roles: Optional[Sequence[str]] = None
                     ) -> "WormSession":
        from brian2 import Network, SpikeMonitor, StateMonitor, ms, seed as bseed

        p = self.params
        self.t_total_ms_sess = float(t_total_ms or p.protocol.t_total_ms)
        self.build()
        sens = self.sens_roles
        stim_roles = tuple(stimulated_roles) if stimulated_roles is not None \
            else tuple(s for s in sens + self.tonic_roles if s)

        spmons = {r: SpikeMonitor(n.neuron, "v", name=f"sp_{r.lower()}")
                  for r, n in self.neurons.items()}
        mons_mus = self.muscle3.monitor(p.body.dt_b,
                                        name=f"{self.name_prefix}_musc")
        net = Network()
        for n in self.neurons.values():
            net.add(n.neuron)
        for cs in self.chemicals:
            net.add(cs.synapses)
        for gj in self.gaps:
            net.add(gj.synapses)
        for g in self.muscle3.groups:
            net.add(g)
        for s in self.muscle3.drivers:
            net.add(s)
        for sp in spmons.values():
            net.add(sp)
        for mm in mons_mus:
            net.add(mm)

        stims = self._stim_arrays(self.t_total_ms_sess)
        ns = {f"stim_{rl.lower()}": ta for rl, ta in stims.items()}
        self._sess = WormSession(self, net, ns, stims, spmons, mons_mus)
        self._sess._init(self.seed)
        return self._sess

    # ------------------------------------------------------------------ #
    # 协议：趋化闭环（M4 机制 A 语义；环境/身体/统计 M4 冻结复用）
    # ------------------------------------------------------------------ #
    def _loop_env_body(self, env: Optional[ChemotaxisEnv] = None,
                       body: Optional[ChemotaxisBody] = None):
        p = self.params
        if env is None:
            env = ChemotaxisEnv(arena_L=p.env.arena_L, sigma=p.env.sigma,
                                c_max=p.env.c_max, c_bg=p.env.c_bg,
                                food_x=p.env.food_x, food_y=p.env.food_y,
                                boundary=p.env.boundary)
        if body is None:
            body = ChemotaxisBody(
                v_fwd0=p.body.v_fwd0, omega_max=p.body.omega_max,
                dt_b=p.body.dt_b, v_osc=p.body.v_osc,
                arena_L=p.env.arena_L, boundary=p.env.boundary,
                turn_omega_pir=p.mech_a.omega_pir,
                turn_duration_ms=p.mech_a.t_pir_ms)
        return env, body

    def _chemotaxis_trial(self, sess: "WormSession", start_x: float,
                          start_y: float, theta0: float, t_total_ms: float,
                          seed: int, env: ChemotaxisEnv,
                          body: ChemotaxisBody) -> ChemotaxisResult:
        """单试次闭环（M4 ChemotaxisLoop._session_trial 同语义，点神经元版）。"""
        p = self.params
        dt_b = body.dt_b
        n_epochs = max(1, int(round(t_total_ms / dt_b)))
        tr = p.transduction
        mech = p.mech_a

        sess.reset(seed=seed)
        body.reset(start_x, start_y, theta0)
        tracker = TimeDiffTracker(tr.tau_win_ms, env.sample(start_x, start_y))
        turn_rng = np.random.default_rng(seed)
        n_turn_events = 0

        xs, ys, thetas, c_sensed = [], [], [], []
        for e in range(n_epochs):
            t_e = e * dt_b
            c_now = env.sample(body.x, body.y)
            s = tracker.s_at(t_e, c_now)
            mus = sess.run_epoch(dt_b, s)
            if mech.enabled and not body.is_turning():
                if s < -mech.theta_pir and sess.any_spikes_in_window(
                        ("SMDDL", "SMDDR"), t_e, t_e + dt_b):
                    direction = 1.0 if turn_rng.random() < 0.5 else -1.0
                    body.trigger_turn(direction, mech.omega_pir, mech.t_pir_ms)
                    n_turn_events += 1
            body.step(mus.get("fwd", 0.0), mus.get("left", 0.0),
                      mus.get("right", 0.0), dt_b, t_e)
            xs.append(body.x)
            ys.append(body.y)
            thetas.append(body.theta)
            c_sensed.append(c_now)

        xa = np.array(xs, dtype=float)
        ya = np.array(ys, dtype=float)
        env.assert_bounded(xa, ya)
        body.assert_trajectory(xa, ya)
        ci = env.ci_per_trial(xa, ya)

        meta_extra = dict(
            start_x=start_x, start_y=start_y, theta0=theta0,
            n_epochs=n_epochs, dt_b_ms=dt_b, ci=ci,
            ci_band_lo=p.protocol.ci_band_lo, ci_band_hi=p.protocol.ci_band_hi,
            c_sensed=np.array(c_sensed, dtype=float),
            n_turn_events=n_turn_events, turn_dir_seed=seed,
            dist_start_food=float(np.hypot(start_x - env.spec.food_x,
                                           start_y - env.spec.food_y)),
            dist_end_food=float(np.hypot(xa[-1] - env.spec.food_x,
                                         ya[-1] - env.spec.food_y)),
            scale=self.scale, fidelity=self.fidelity,
            dt_ms=self.dt_ms, method=self.method,
        )
        return sess.finish(x=xa, y=ya, theta=np.array(thetas, dtype=float),
                           meta_extra=meta_extra)

    def run_chemotaxis_trials(self, n_trials: Optional[int] = None,
                              seed_base: int = 0,
                              t_total_ms: Optional[float] = None,
                              start_jitter: Optional[float] = None,
                              env: Optional[ChemotaxisEnv] = None,
                              body: Optional[ChemotaxisBody] = None,
                              wall: bool = True) -> Tuple[List[ChemotaxisResult], dict]:
        """闭环趋化多试次（确定性：同 seed_base 重跑逐位一致）。

        返回 (results, meta)；meta 含 wall_s（每试次墙钟）、build_wall_s（编译）。
        """
        p = self.params
        n = int(n_trials or p.protocol.n_trials)
        t_total = float(t_total_ms or p.protocol.t_total_ms)
        jitter = float(start_jitter if start_jitter is not None
                       else p.protocol.start_jitter)
        env, body = self._loop_env_body(env, body)
        sess = self.make_session(t_total_ms=t_total)
        rng = np.random.default_rng(seed_base)
        out, walls = [], []
        for trial in range(n):
            if jitter > 0:
                sx = p.protocol.start_x + rng.normal(0.0, jitter)
                sy = p.protocol.start_y + rng.normal(0.0, jitter)
                th0 = rng.uniform(0.0, 2.0 * math.pi)
            else:
                sx, sy, th0 = p.protocol.start_x, p.protocol.start_y, 0.0
            t0w = time.perf_counter()
            res = self._chemotaxis_trial(sess, sx, sy, th0, t_total,
                                         seed_base + trial, env, body)
            res.meta["trial"] = trial
            res.meta["wall_s"] = time.perf_counter() - t0w
            walls.append(res.meta["wall_s"])
            out.append(res)
        meta = dict(wall_s=walls, mean_wall_s=float(np.mean(walls)) if walls
                    else float("nan"),
                    build_wall_s=self._build_wall_s,
                    n_neurons=self.sub.n_neurons, n_chem=self.sub.n_chem,
                    n_gap=self.sub.n_gap, scale=self.scale,
                    fidelity=self.fidelity, dt_ms=self.dt_ms,
                    method=self.method)
        return out, meta

    def run_control(self, n_trials: Optional[int] = None, seed_base: int = 1000,
                    t_total_ms: Optional[float] = None,
                    start_jitter: Optional[float] = None
                    ) -> Tuple[List[ChemotaxisResult], dict]:
        """无梯度对照（C_max 置 0；P4 主判据语义，M4 同款）。"""
        env, body = self._loop_env_body()
        return self.run_chemotaxis_trials(n_trials=n_trials, seed_base=seed_base,
                                          t_total_ms=t_total_ms,
                                          start_jitter=start_jitter,
                                          env=env.no_gradient(), body=body)

    # ------------------------------------------------------------------ #
    # 协议：静息发放率分布（无刺激）
    # ------------------------------------------------------------------ #
    def run_resting(self, t_total_ms: float = 1000.0,
                    seed: Optional[int] = None) -> dict:
        """无刺激 T 窗：逐神经元发放率（Hz）+ 静默比例 + 稳定性检查。"""
        from brian2 import ms as bms

        p = self.params
        seed = self.seed if seed is None else int(seed)
        sess = self.make_session(t_total_ms=t_total_ms)
        sess.reset(seed=seed)
        t0w = time.perf_counter()
        sess.run_resting_window(t_total_ms)
        wall = time.perf_counter() - t0w
        rates = {}
        t_end = sess.net.t / bms
        for role, times in sess.role_spike_times().items():
            rates[role] = float(len(times)) / (float(t_end) / 1000.0)
        arr = np.array(list(rates.values()), dtype=float)
        silent_frac = float(np.mean(arr < 0.5)) if arr.size else float("nan")
        return dict(
            rates_hz=rates, median_hz=float(np.median(arr)) if arr.size
            else float("nan"),
            max_hz=float(arr.max()) if arr.size else float("nan"),
            silent_frac=silent_frac, n_spiking=float(np.sum(arr >= 0.5)),
            has_nan=bool(np.any(~np.isfinite(arr))), wall_s=wall,
            t_total_ms=t_total_ms,
        )

    # ------------------------------------------------------------------ #
    # 协议：自发状态比例（无刺激短协议，P6 指标；后退需身体负速度——M5 规格）
    # ------------------------------------------------------------------ #
    def run_spontaneous(self, t_total_ms: float = 5000.0,
                        seed: Optional[int] = None,
                        v_rev0: float = 1.0,
                        classify: Optional[dict] = None) -> dict:
        """无刺激无梯度 T 窗：C_fwd/C_back/C_left/C_right 序列 → 状态比例。

        状态分类（阈值定稿于 CSV 前用本函数默认，m5_worm_params.csv 落盘后切换）：
        fwd: v > v_thr 且 |ω| < ω_thr；rev: v < -v_thr；turn: |ω| > ω_thr；
        pause: 其余。v = v_fwd0·clip(C_fwd,0,1) − v_rev0·clip(C_back,0,1)
        （M5 身体方程规格 §5.2 #3 的简版，back 通道存在时有效）。
        """
        from brian2 import ms as bms

        p = self.params
        seed = self.seed if seed is None else int(seed)
        body = ChemotaxisBody(v_fwd0=p.body.v_fwd0, omega_max=p.body.omega_max,
                              dt_b=p.body.dt_b, v_osc=0.0,
                              arena_L=p.env.arena_L, boundary=p.env.boundary)
        sess = self.make_session(t_total_ms=t_total_ms)
        sess.reset(seed=seed)
        t0w = time.perf_counter()
        n_epochs = max(1, int(round(t_total_ms / body.dt_b)))
        mus_hist = []
        for _ in range(n_epochs):
            mus_hist.append(sess.run_epoch(body.dt_b, 0.0))
        wall = time.perf_counter() - t0w

        v_thr = float((classify or {}).get("v_thr", 0.05 * p.body.v_fwd0))
        w_thr = float((classify or {}).get("omega_thr", 0.2 * p.body.omega_max))
        states = []
        for m in mus_hist:
            c_fwd = max(0.0, min(1.0, m.get("fwd", 0.0)))
            c_back = max(0.0, min(1.0, m.get("back", 0.0)))
            v = p.body.v_fwd0 * c_fwd - v_rev0 * c_back
            omega = p.body.omega_max * (m.get("left", 0.0) - m.get("right", 0.0))
            if abs(omega) > w_thr:
                states.append("turn")
            elif v > v_thr:
                states.append("fwd")
            elif v < -v_thr:
                states.append("rev")
            else:
                states.append("pause")
        n = len(states)
        frac = {s: float(states.count(s)) / n for s in ("fwd", "rev", "turn", "pause")}
        return dict(frac=frac, n_epochs=n, wall_s=wall,
                    classify_thresholds=dict(v_thr=v_thr, omega_thr=w_thr))


# --------------------------------------------------------------------- #
# WormSession：试次会话（epoch 迭代；ChemoSession 同语义，点神经元版）
# --------------------------------------------------------------------- #
class WormSession:
    """一次试次的网络会话（闭环 epoch 的引擎侧；store/restore + 重播种）。"""

    def __init__(self, circuit: WormCircuit, net, ns, stims, spmons, mons_mus):
        self.circuit = circuit
        self.net = net
        self.ns = ns
        self.stims = stims
        self.spmons = spmons
        self.mons_mus = mons_mus
        self._n_epochs = 0

    def _init(self, seed: int):
        from brian2 import seed as bseed, ms

        bseed(seed)
        self.net.run(0 * ms, namespace=self.ns)
        self.net.store()
        self._rng = np.random.default_rng(seed)
        self._fill_tonic()

    def _fill_tonic(self):
        c = self.circuit
        for role, density in c.sub.tonic_uA_cm2.items():
            if role not in self.stims:
                continue
            idx = c.neurons[role].label_of("soma")
            k = c._nA_per_density(role, idx)
            self.stims[role].values[:, idx] = (density * k) * 1e-9

    def reset(self, seed: Optional[int] = None):
        from brian2 import seed as bseed

        c = self.circuit
        bseed(seed if seed is not None else c.seed)
        self.net.restore()
        for ta in self.stims.values():
            ta.values[:] = 0.0
        self._fill_tonic()
        self._rng = np.random.default_rng(seed if seed is not None else c.seed)
        self._n_epochs = 0

    # ------------------------------------------------------------------ #
    def _ase_nA(self, s_value: float) -> Dict[str, float]:
        """时间差分 s → 感觉对注入 nA（M4 转导语义；max 在 python 侧完成）。"""
        c = self.circuit
        tr = c.params.transduction
        s = float(s_value)
        on_role, off_role = c.sens_roles
        out = {}
        if on_role:
            i_on = tr.g_on * max(s, 0.0)
            out[on_role] = i_on * c._nA_per_density(on_role, 0)
        if off_role:
            i_off = tr.g_off * max(-s, 0.0)
            out[off_role] = i_off * c._nA_per_density(off_role, 0)
        return out

    def run_epoch(self, dt_ms: float, s_value: float) -> Dict[str, float]:
        """运行一个 epoch：s → 感觉对 nA → 写入固定形状数组切片 → run(dt)。"""
        from brian2 import ms

        c = self.circuit
        p = c.params
        t_now_ms = float(self.net.t / ms)
        dt = float(dt_ms)
        i0 = int(round(t_now_ms / p.dt_ms))
        i1 = int(round((t_now_ms + dt) / p.dt_ms))
        for role, nA in self._ase_nA(s_value).items():
            idx = c.neurons[role].label_of("soma")
            self.stims[role].values[i0:i1, idx] = float(nA) * 1e-9
        self.net.run(dt * ms, namespace=self.ns)
        self._n_epochs += 1
        return self.circuit.muscle3.read()

    def run_resting_window(self, t_total_ms: float):
        """无刺激整段运行（静息协议；零数组越界钳位 → 恒 0）。"""
        from brian2 import ms

        self.net.run(t_total_ms * ms, namespace=self.ns)
        self._n_epochs += 1

    def role_spike_times(self) -> Dict[str, np.ndarray]:
        """逐角色发放时刻（ms；component 模式：整神经元所有隔室）。"""
        from brian2 import ms as bms

        out = {}
        for role, sp in self.spmons.items():
            out[role] = np.asarray(sp.t / bms)
        return out

    def any_spikes_in_window(self, roles, t0_ms: float, t1_ms: float) -> bool:
        from brian2 import ms

        for role in roles:
            sp = self.spmons.get(str(role).upper())
            if sp is None:
                continue
            t = np.asarray(sp.t / ms)
            if t.size and np.any((t >= t0_ms - 1e-9) & (t < t1_ms)):
                return True
        return False

    def muscle_read(self) -> Dict[str, float]:
        return self.circuit.muscle3.read()

    def finish(self, x=None, y=None, theta=None,
               meta_extra: Optional[dict] = None) -> ChemotaxisResult:
        """收集本试次结果（V/发放/肌肉/可选轨迹；ChemotaxisResult 兼容）。"""
        from brian2 import ms, mV

        c = self.circuit
        p = c.params
        t = np.arange(0.0, self.circuit.t_total_ms_sess, p.dt_ms)
        spikes: Dict[str, np.ndarray] = {}
        for role, sp in self.spmons.items():
            t_arr = np.asarray(sp.t / ms)
            i_arr = np.asarray(sp.i)
            n_comp = c._n_comp(role)
            if n_comp == 1:
                spikes[f"{role.lower()}_node3"] = t_arr
                spikes[f"{role.lower()}_soma"] = t_arr
            else:
                # 双隔室/多隔室：node3=末隔室、soma=0（点档简并见上）
                spikes[f"{role.lower()}_soma"] = t_arr[i_arr == 0]
                spikes[f"{role.lower()}_node3"] = t_arr[i_arr == n_comp - 1]
        mus_names = list(self.circuit.muscle3.channels)
        mus_map = {}
        for i, ch in enumerate(mus_names):
            var = f"c_{ch}"
            mus_map[ch] = np.array(getattr(self.mons_mus[i], var)[0])
        zero = np.zeros(1)
        c_fwd = mus_map.get("fwd", zero)
        meta = dict(t_total_ms=self.circuit.t_total_ms_sess, seed=p.seed,
                    n_epochs=self._n_epochs, dt_ms=c.dt_ms, method=c.method,
                    scale=c.scale, fidelity=c.fidelity)
        if meta_extra:
            meta.update(meta_extra)
        return ChemotaxisResult(
            t_ms=t, v_mv={}, spike_times_ms=spikes,
            c_fwd=c_fwd,
            c_left=mus_map.get("left", zero),
            c_right=mus_map.get("right", zero),
            meta=meta,
            x=None if x is None else np.asarray(x, dtype=float),
            y=None if y is None else np.asarray(y, dtype=float),
            theta=None if theta is None else np.asarray(theta, dtype=float),
        )


# --------------------------------------------------------------------- #
# 机械逃避：M3 反射子图（降阶正确性验证，§3.4）
# --------------------------------------------------------------------- #
@dataclass
class ReflexSpec:
    """M3 反射链规格（本地解析 m3_reflex_params.csv——M4 加载器 muscle_channel
    不接受 muscle_back，冻结文件不可改 → 本模块自解析，L7 记录）。"""

    roles: List[str] = field(default_factory=list)
    chem: List[ChemRow] = field(default_factory=list)      # 化学链（node3→soma）
    muscles: List[MuscleRow] = field(default_factory=list)  # 通道 back/fwd
    tonic_uA_cm2: Dict[str, float] = field(default_factory=dict)
    touch_density: float = 60.0      # i0_uA_cm2（µA/cm²）
    touch_start_ms: float = 50.0
    touch_dur_ms: float = 5.0
    muscle_tau_ms: float = 20.0
    source: str = ""


def load_reflex_spec(csv_path: Optional[str] = None) -> ReflexSpec:
    """本地解析 m3_reflex_params.csv（列 schema 同 m4 CSV）。"""
    import csv as _csv

    path = csv_path or DEFAULT_M3_PARAMS_CSV
    spec = ReflexSpec(source=path)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(_csv.DictReader(r for r in f
                                    if not r.strip().startswith("#")))
    for r in rows:
        role = (r.get("role") or "").strip().upper()
        frm = (r.get("synapse_from") or "").strip().upper()
        if role and role != "PARAM":
            if role not in spec.roles:
                spec.roles.append(role)
            ton = r.get("tonic_uA_cm2")
            if ton:
                spec.tonic_uA_cm2[role] = float(ton)
        if frm:
            stype = (r.get("synapse_type") or "").strip().lower()
            to = (r.get("synapse_to") or "").strip().upper()
            g = r.get("g_max_ns")
            if stype == "muscle":
                ch = to.lower()
                if ch.startswith("muscle_"):
                    ch = ch[len("muscle_"):]
                spec.muscles.append(MuscleRow(motor=frm, channel=ch,
                                              w=float(g or 0.3)))
            else:
                spec.chem.append(ChemRow(pre=frm, post=to, syn_type=stype,
                                         g_ns=float(g or 5.0),
                                         delay_ms=float(r.get("delay_ms") or 0.5)))
        elif role == "PARAM":
            key = (r.get("neuron_class") or "").strip().lower()
            val = r.get("value")
            if key == "i0_uA_cm2" or key == "i0_uacm2":
                spec.touch_density = float(val)
            elif key == "touch_start_ms":
                spec.touch_start_ms = float(val)
            elif key == "touch_dur_ms":
                spec.touch_dur_ms = float(val)
            elif key == "muscle_tau_ms":
                spec.muscle_tau_ms = float(val)
    return spec


class ReflexCircuit:
    """M3 反射弧子图（PLM/AVM/DA/VB + 双通道肌肉）在同保真度组件下的重实现。

    用于降阶正确性验证：方向（D_peak = max(C_back − C_fwd) > 0.3 → back）与
    神经潜伏期（PLM 首发放 → DA 首发放，M3 窗 [5,20]ms）对照 M3 已验证结果
    （m3_p1_direction.csv：D_peak=0.352；m3_p3_latency.csv：lat=8.23–8.93ms）。
    """

    def __init__(self, csv_path: Optional[str] = None, fidelity: str = "point",
                 dt_ms: Optional[float] = None, method: Optional[str] = None,
                 name_prefix: str = "reflex", seed: int = 0):
        if fidelity not in FIDELITY_AXIS:
            raise ValueError(f"保真度需为 {FIDELITY_AXIS}：{fidelity}")
        self.fidelity = fidelity
        self.dt_ms = FIDELITY_DT[fidelity][0] if dt_ms is None else float(dt_ms)
        self.method = FIDELITY_DT[fidelity][1] if method is None else method
        self.name_prefix = name_prefix
        self.seed = seed
        self.spec: ReflexSpec = load_reflex_spec(csv_path)
        self._m2 = load_synapse_params()
        self.neurons: Dict[str, object] = {}
        self.chemicals = []
        self.muscle3 = None
        self.touch_role = "PLM"

    def build(self):
        from neural_exploration.src.brian_env import configure_brian2
        from brian2 import ms, start_scope

        configure_brian2()
        start_scope()
        spec = self.spec
        post_types: Dict[str, set] = {r: set() for r in spec.roles}
        for s in spec.chem:
            post_types[s.post].add(s.syn_type)
        self.neurons = {}
        for role in spec.roles:
            sub = {t: self._m2[t] for t in post_types[role]}
            eqs = chemical_post_eqs(sub)
            ims = chemical_im_terms(sub)
            stim_var = f"stim_{role.lower()}"
            if self.fidelity == "point":
                nrn = PointNeuron(name=f"{self.name_prefix}_{role.lower()}",
                                  dt_ms=self.dt_ms, method=self.method,
                                  extra_eqs=eqs, extra_im_terms=ims,
                                  stim_var=stim_var).build()
            elif self.fidelity == "two_comp":
                nrn = TwoCompartmentNeuron(
                    name=f"{self.name_prefix}_{role.lower()}", dt_ms=self.dt_ms,
                    method=self.method, extra_eqs=eqs, extra_im_terms=ims,
                    stim_var=stim_var).build()
            else:
                nrn = MultiCompartmentNeuron(
                    name=f"{self.name_prefix}_{role.lower()}", dt_ms=self.dt_ms,
                    method=self.method, t_total_ms=200.0, extra_eqs=eqs,
                    extra_im_terms=ims, stim_var=stim_var).build()
            self.neurons[role] = nrn
        self.chemicals = []
        for k, s in enumerate(spec.chem):
            base = self._m2[s.syn_type]
            sp = SynapseParams(synapse_type=s.syn_type,
                               g_max_ns=s.g_ns, tau_ms=base.tau_ms,
                               e_rev_mv=base.e_rev_mv, p_release=1.0,
                               n_vesicles=1, mg_mm=base.mg_mm, u0=base.u0,
                               tau_fac_ms=base.tau_fac_ms,
                               tau_rec_ms=base.tau_rec_ms)
            from neural_exploration.src.synapse_model import ChemicalSynapse
            cs = ChemicalSynapse(self.neurons[s.pre], self.neurons[s.post], sp,
                                 pre_site="node3", post_site="soma",
                                 name=f"{self.name_prefix}_syn{k}")
            cs.build()
            cs.synapses.delay = s.delay_ms * ms
            self.chemicals.append(cs)
        from neural_exploration.src.chemotaxis_circuit import Muscle3
        self.muscle3 = Muscle3(tau_ms=spec.muscle_tau_ms, cap=1.0,
                               channels=("back", "fwd"),
                               name=f"{self.name_prefix}_muscle3")
        self.muscle3.build()
        for k, m in enumerate(spec.muscles):
            self.muscle3.connect_driver(self.neurons[m.motor], m.channel,
                                        weight=m.w,
                                        name=f"{self.name_prefix}_musdrv{k}")
        # 张力（VB 维持 C_fwd 基线，M3 语义）
        self._tonic_uA_cm2 = spec.tonic_uA_cm2
        return self

    def run(self, t_total_ms: float = 150.0, seed: Optional[int] = None
            ) -> "ReflexResult":
        """触刺激单次运行：方向（D_peak）+ 神经潜伏期（PLM→DA）。"""
        from brian2 import (Network, SpikeMonitor, StateMonitor, TimedArray,
                            amp, ms, nA, seed as bseed)

        seed = self.seed if seed is None else int(seed)
        self.build()
        from neural_exploration.src.brian_env import configure_brian2
        configure_brian2()
        from brian2 import start_scope
        start_scope()
        bseed(seed)

        t0 = time.perf_counter()
        spec = self.spec
        n_steps = int(round(max(STIM_WINDOW_MS, t_total_ms) / self.dt_ms))
        stims = {}
        tonic = dict(self._tonic_uA_cm2)
        for role in spec.roles:
            n_comp = int(self.neurons[role].neuron.N)
            arr = np.zeros((n_steps, n_comp)) * amp
            if role == self.touch_role:
                # 触刺激 = 密度 I0（µA/cm²）@ soma 等效位点（点档；M3 树突端注入的
                # 密度等效——刺激密度为生理量，按点膜面积换算 nA，L7 记录）
                i_nA = self.neurons[role].density_to_nA(spec.touch_density)
                i0 = int(round(spec.touch_start_ms / self.dt_ms))
                i1 = int(round((spec.touch_start_ms + spec.touch_dur_ms)
                               / self.dt_ms))
                arr[i0:i1, :] = i_nA * nA
            if role in tonic:
                k = self.neurons[role].density_to_nA(1.0)
                arr[:, 0] = tonic[role] * k * nA
            stims[role] = TimedArray(arr, dt=self.dt_ms * ms,
                                     name=f"stim_{role.lower()}")
        ns = {f"stim_{r.lower()}": ta for r, ta in stims.items()}

        spmons = {r: SpikeMonitor(n.neuron, "v", name=f"sp_{r.lower()}")
                  for r, n in self.neurons.items()}
        mons_mus = self.muscle3.monitor(self.dt_ms,
                                        name=f"{self.name_prefix}_musc")
        net = Network()
        for n in self.neurons.values():
            net.add(n.neuron)
        for cs in self.chemicals:
            net.add(cs.synapses)
        for g in self.muscle3.groups:
            net.add(g)
        for s in self.muscle3.drivers:
            net.add(s)
        for sp in spmons.values():
            net.add(sp)
        for mm in mons_mus:
            net.add(mm)
        net.run(t_total_ms * ms, namespace=ns)
        wall = time.perf_counter() - t0

        mus_names = list(self.muscle3.channels)
        c_series = {}
        for i, ch in enumerate(mus_names):
            var = f"c_{ch}"
            c_series[ch] = np.array(getattr(mons_mus[i], var)[0])
        c_back = c_series.get("back", np.zeros(1))
        c_fwd = c_series.get("fwd", np.zeros_like(c_back))
        d_peak = float(np.max(c_back - c_fwd)) if c_back.size else 0.0
        t_sim = np.arange(0.0, t_total_ms, self.dt_ms)

        sp_da = np.asarray(spmons["DA"].t / ms)
        lat = float(sp_da[0] - spec.touch_start_ms) if sp_da.size else float("nan")
        sp_plm = np.asarray(spmons["PLM"].t / ms)
        sp_avm = np.asarray(spmons["AVM"].t / ms)
        return ReflexResult(
            d_peak=d_peak, c_back=c_back, c_fwd=c_fwd, t_ms=t_sim,
            neural_latency_ms=lat, direction=("back" if d_peak > 0.3
                                              else "not_back"),
            plm_spikes=sp_plm, avm_spikes=sp_avm, da_spikes=sp_da,
            wall_s=wall, t_total_ms=t_total_ms, fidelity=self.fidelity,
            dt_ms=self.dt_ms, method=self.method,
        )


@dataclass
class ReflexResult:
    d_peak: float
    c_back: np.ndarray
    c_fwd: np.ndarray
    t_ms: np.ndarray
    neural_latency_ms: float
    direction: str
    plm_spikes: np.ndarray
    avm_spikes: np.ndarray
    da_spikes: np.ndarray
    wall_s: float
    t_total_ms: float
    fidelity: str
    dt_ms: float
    method: str


# --------------------------------------------------------------------- #
# GroupedWormCircuit：连接组规模的批量组装（point 保真度）
# --------------------------------------------------------------------- #
# 冷编译预算（M4 L16/L25 + M5 §8 风险表）：M2 component 模式每突触对象 ~5.2s 编译
# （实测，L7）——302 全虫 2472 化学 + 1093 缝隙 + 68 肌肉 ≈ 5-6h 冷编译不可接受。
# → grouped 模式：全部神经元合并为**一个** NeuronGroup（逐神经元参数），化学突触按
# 类型各一个 Synapses（每连接 gmax 变量），缝隙一个 Synapses（I_gap_in/I_gap_out
# 两个 summed 目标变量——M2 (summed) 单变量在 ≥2 缝隙/神经元时 Brian2 报错，L7），
# 肌肉每通道一个驱动 Synapses。方程与 component 模式逐位一致（同一 on_pre 增量 +
# 同一膜方程）——20 档两模式数值对照验证（L7）。冷编译 302 全虫降至 ~10min。
GROUPED_PN_EQ = """
Im = gL*(EL-v) + gNa*m**3*h*(ENa-v) + gK*n**4*(EK-v) + g_ampa*(0.0*mV-v) + g_gaba*(-70.0*mV-v) : amp/meter**2
dv/dt = (Im + (stim(t, i) + I_gap + I_gap_in + I_gap_out)/AREA) / Cm : volt
dm/dt = alpham*(1-m)-betam*m : 1
dh/dt = alphah*(1-h)-betah*h : 1
dn/dt = alphan*(1-n)-betan*n : 1
alpham = (0.1/mV)*(v+40*mV)/(1-exp(-(v+40*mV)/(10*mV)))/ms : Hz
betam = 4*exp(-(v+65*mV)/(18*mV))/ms : Hz
alphah = 0.07*exp(-(v+65*mV)/(20*mV))/ms : Hz
betah = 1/(1+exp(-(v+35*mV)/(10*mV)))/ms : Hz
alphan = (0.01/mV)*(v+55*mV)/(1-exp(-(v+55*mV)/(10*mV)))/ms : Hz
betan = 0.125*exp(-(v+65*mV)/(80*mV))/ms : Hz
dg_ampa/dt = -g_ampa/({TAU_AMPA}*ms) : siemens/meter**2
dg_gaba/dt = -g_gaba/({TAU_GABA}*ms) : siemens/meter**2
gNa : siemens/meter**2
gK : siemens/meter**2
gL : siemens/meter**2
AREA : meter**2
EL = {EL}*mV : volt (shared)
ENa = {ENA}*mV : volt (shared)
EK = {EK}*mV : volt (shared)
I_gap : amp
I_gap_in : amp
I_gap_out : amp
"""

_GAP_MODEL_GROUPED = """
I_couple = g_gap*(v_pre - v_post) : amp
I_gap_in_post = I_couple : amp (summed)
I_gap_out_pre = -I_couple : amp (summed)
"""


class GroupedWormCircuit(WormCircuit):
    """批量组装版 WormCircuit（point 保真度；规模 50/100/302 的默认路径）。

    协议接口与父类一致（run_chemotaxis_trials/run_resting/run_spontaneous 复用）；
    仅 build/make_session/刺激与发放记录路径不同（单组 + 批量 Synapses）。
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("fidelity", "point")
        # grouped 模式：缝隙统一走批量 (summed) I_gap_in/I_gap_out（多缝隙拓扑唯一可行
        # 路径，L7）；gap_mode="none" 用于组件/分组化学一致性对照
        if kwargs.get("gap_mode") not in ("grouped", "none"):
            kwargs["gap_mode"] = "grouped"
        super().__init__(*args, **kwargs)
        if self.fidelity != "point":
            raise ValueError("GroupedWormCircuit 仅支持 point 保真度"
                             "（双隔室/多隔室用 component 模式 ≤50 子图）")
        self.role_index: Dict[str, int] = {}
        self.group = None
        self.chem_synapses: List[object] = []
        self.gap_synapse = None
        self.mus_drivers: Dict[str, object] = {}
        self._sp = None
        self._tonic_nA: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    def build(self):
        from neural_exploration.src.brian_env import configure_brian2
        from brian2 import (NeuronGroup, SpikeMonitor, Synapses, cm, defaultclock,
                            meter, ms, mS, mV, siemens, start_scope, uF)

        configure_brian2()
        start_scope()
        t0 = time.perf_counter()
        defaultclock.dt = self.dt_ms * ms
        n = len(self.names)
        if n == 0:
            raise ValueError("空子集：连接组未包含任何神经元")
        # 突触衰减时间常数取 M2 定稿（ampa=3ms/gaba=5ms，与 component 模式同源）
        t_ampa = self._m2["ampa"].tau_ms
        t_gaba = self._m2["gaba"].tau_ms
        eqs = GROUPED_PN_EQ.format(EL=-54.4, ENA=50.0, EK=-77.0,
                                   TAU_AMPA=t_ampa, TAU_GABA=t_gaba)
        ns = {"Cm": 1.0 * uF / cm ** 2}
        self.group = NeuronGroup(n, eqs, method=self.method,
                                 threshold="v > -20*mV", refractory=2.0 * ms,
                                 name=f"{self.name_prefix}_all", namespace=ns)
        from neural_exploration.src.ion_channels import steady_state_gates
        m0, h0, n0 = steady_state_gates(-65.0)
        self.group.v = -65.0 * mV
        self.group.m, self.group.h, self.group.n = m0, h0, n0
        self.group.gNa = 120.0 * mS / cm ** 2
        self.group.gK = 36.0 * mS / cm ** 2
        # 漏电缩放（M5-B1e2 校准扩展：gL=0.3mS/cm² 基准；风险表"漏电增强"杠杆——
        # 提高阈值 → 抑制网络级夹带；None → 1.0 恒等，API 兼容）
        self.group.gL = (0.3 * self.gL_scale) * mS / cm ** 2
        self.group.AREA = np.full(n, 1.257e-5 * 1e-4) * meter ** 2
        self.role_index = {r: k for k, r in enumerate(self.names)}

        # ---- 化学突触：每类型一个 Synapses（逐连接 gmax/delay）----
        self.chem_synapses = []
        for stype in ("ampa", "gaba"):
            rows = [r for r in self.sub.chem if r.syn_type == stype]
            if not rows:
                continue
            pre_i = np.array([self.role_index[r.pre] for r in rows])
            post_i = np.array([self.role_index[r.post] for r in rows])
            gmax = np.array([r.g_ns * self._class_scale_for(r.pre, r.post)
                             * self.syn_type_scales.get(r.syn_type, 1.0)
                             * 1e-9 / (1.257e-5 * 1e-4)
                             for r in rows]) * siemens / meter ** 2
            delays = np.array([r.delay_ms for r in rows]) * ms
            syn = Synapses(self.group, self.group,
                           model="gmax : siemens/meter**2",
                           on_pre=f"g_{stype}_post = g_{stype}_post + gmax",
                           name=f"{self.name_prefix}_chem_{stype}")
            syn.connect(i=pre_i, j=post_i)
            syn.gmax = gmax
            syn.delay = delays
            self.chem_synapses.append(syn)

        # ---- 缝隙连接：一个 Synapses（I_gap_in/I_gap_out 双 summed 目标；
        # gap_mode="none" 时跳过——组件/分组化学一致性对照）----
        if self.sub.gaps and self.gap_mode == "grouped":
            a_i = np.array([self.role_index[r.a] for r in self.sub.gaps])
            b_i = np.array([self.role_index[r.b] for r in self.sub.gaps])
            gg = np.array([r.g_ns * self.gap_scale
                           for r in self.sub.gaps]) * 1e-9 * siemens
            syn = Synapses(self.group, self.group,
                           model="g_gap : siemens\n" + _GAP_MODEL_GROUPED,
                           name=f"{self.name_prefix}_gaps")
            syn.connect(i=a_i, j=b_i)
            syn.g_gap = gg
            self.gap_synapse = syn

        # ---- 肌肉：每通道一个驱动 Synapses（on_pre 增量 wm，逐连接）----
        from neural_exploration.src.chemotaxis_circuit import Muscle3
        channels = list(dict.fromkeys(m.channel for m in self.sub.muscles))
        if not channels:
            channels = ["fwd", "left", "right"]
        self.muscle3 = Muscle3(tau_ms=self.muscle_tau_ms, cap=self.muscle_cap,
                               channels=channels,
                               name=f"{self.name_prefix}_muscle3")
        self.muscle3.build()
        self.mus_drivers = {}
        for ch in channels:
            rows = [m for m in self.sub.muscles if m.channel == ch]
            if not rows:
                continue
            pre_i = np.array([self.role_index[m.motor] for m in rows])
            wm = np.array([m.w for m in rows])
            g = self.muscle3.get(ch)
            var = f"c_{ch}"
            cap = self.muscle_cap
            if cap is not None:
                on_pre = f"{var}_post = clip({var}_post + wm, 0.0, CAP)"
                ns_m = {"CAP": cap}
            else:
                on_pre = f"{var}_post += wm"
                ns_m = {}
            syn = Synapses(self.group, g, model="wm : 1", on_pre=on_pre,
                           name=f"{self.name_prefix}_musdrv_{ch}",
                           namespace=ns_m)
            syn.connect(i=pre_i, j=0)
            syn.wm = wm
            syn.delay = 0.1 * ms
            self.mus_drivers[ch] = syn

        # ---- 张力（tonic → nA，会话填充）----
        self._tonic_nA = {}
        for role, density in self.sub.tonic_uA_cm2.items():
            idx = self.role_index[role]
            self._tonic_nA[role] = density * 1e-6 * 1.257e-5 * 1e9

        self._built = True
        self._build_wall_s = time.perf_counter() - t0
        return self

    # ------------------------------------------------------------------ #
    def _n_comp(self, role: str) -> int:
        return 1

    def make_session(self, t_total_ms: Optional[float] = None,
                     record: Optional[Sequence[str]] = None,
                     stimulated_roles: Optional[Sequence[str]] = None
                     ) -> "GroupedWormSession":
        from brian2 import (Network, SpikeMonitor, StateMonitor, TimedArray,
                            amp, ms, seed as bseed)

        p = self.params
        self.t_total_ms_sess = float(t_total_ms or p.protocol.t_total_ms)
        self.build()
        sens = self.sens_roles
        stim_roles = tuple(stimulated_roles) if stimulated_roles is not None \
            else tuple(s for s in sens + self.tonic_roles if s)

        self._sp = SpikeMonitor(self.group, "v", name=f"{self.name_prefix}_sp")
        mons_mus = self.muscle3.monitor(p.body.dt_b,
                                        name=f"{self.name_prefix}_musc")
        net = Network()
        net.add(self.group)
        for syn in self.chem_synapses:
            net.add(syn)
        if self.gap_synapse is not None:
            net.add(self.gap_synapse)
        for syn in self.mus_drivers.values():
            net.add(syn)
        for g in self.muscle3.groups:
            net.add(g)
        net.add(self._sp)
        for mm in mons_mus:
            net.add(mm)

        n_steps = self._stim_n_steps(self.t_total_ms_sess)
        stim = TimedArray(np.zeros((n_steps, len(self.names))) * amp,
                          dt=self.dt_ms * ms, name="stim")
        ns = {"stim": stim}
        self._sess = GroupedWormSession(self, net, ns, stim, mons_mus)
        self._sess._init(self.seed)
        return self._sess

    def _nA_per_density(self, role: str, idx: int) -> float:
        return 1e-6 * 1.257e-5 * 1e9   # 1 µA/cm² → nA（点面积恒定）


class GroupedWormSession:
    """GroupedWormCircuit 的试次会话（单组 stim (n_steps, N)，epoch 迭代）。"""

    def __init__(self, circuit: GroupedWormCircuit, net, ns, stim, mons_mus):
        self.circuit = circuit
        self.net = net
        self.ns = ns
        self.stim = stim
        self.mons_mus = mons_mus
        self._n_epochs = 0

    def _init(self, seed: int):
        from brian2 import seed as bseed, ms

        bseed(seed)
        self.net.run(0 * ms, namespace=self.ns)
        self.net.store()
        self._rng = np.random.default_rng(seed)
        self._fill_tonic()

    def _fill_tonic(self):
        for role, nA in self.circuit._tonic_nA.items():
            idx = self.circuit.role_index[role]
            self.stim.values[:, idx] = nA * 1e-9

    def reset(self, seed: Optional[int] = None):
        from brian2 import seed as bseed

        c = self.circuit
        bseed(seed if seed is not None else c.seed)
        self.net.restore()
        self.stim.values[:] = 0.0
        self._fill_tonic()
        self._rng = np.random.default_rng(seed if seed is not None else c.seed)
        self._n_epochs = 0

    def run_epoch(self, dt_ms: float, s_value: float) -> Dict[str, float]:
        from brian2 import ms

        c = self.circuit
        p = c.params
        tr = p.transduction
        s = float(s_value)
        t_now_ms = float(self.net.t / ms)
        dt = float(dt_ms)
        i0 = int(round(t_now_ms / p.dt_ms))
        i1 = int(round((t_now_ms + dt) / p.dt_ms))
        on_role, off_role = c.sens_roles
        if on_role:
            i_on = tr.g_on * max(s, 0.0)
            idx = c.role_index[on_role]
            self.stim.values[i0:i1, idx] = (i_on * 1e-6 * 1.257e-5 * 1e9) * 1e-9
        if off_role:
            i_off = tr.g_off * max(-s, 0.0)
            idx = c.role_index[off_role]
            self.stim.values[i0:i1, idx] = (i_off * 1e-6 * 1.257e-5 * 1e9) * 1e-9
        self.net.run(dt * ms, namespace=self.ns)
        self._n_epochs += 1
        return self.circuit.muscle3.read()

    def run_resting_window(self, t_total_ms: float):
        from brian2 import ms

        self.net.run(t_total_ms * ms, namespace=self.ns)
        self._n_epochs += 1

    def role_spike_times(self) -> Dict[str, np.ndarray]:
        from brian2 import ms as bms

        c = self.circuit
        t_arr = np.asarray(c._sp.t / bms)
        i_arr = np.asarray(c._sp.i)
        return {role: t_arr[i_arr == idx]
                for role, idx in c.role_index.items()}

    def any_spikes_in_window(self, roles, t0_ms: float, t1_ms: float) -> bool:
        from brian2 import ms

        c = self.circuit
        t = np.asarray(self.circuit._sp.t / ms)
        i = np.asarray(self.circuit._sp.i)
        for role in roles:
            idx = c.role_index.get(str(role).upper())
            if idx is None:
                continue
            mask = i == idx
            if mask.any() and np.any((t[mask] >= t0_ms - 1e-9)
                                     & (t[mask] < t1_ms)):
                return True
        return False

    def muscle_read(self) -> Dict[str, float]:
        return self.circuit.muscle3.read()

    def finish(self, x=None, y=None, theta=None,
               meta_extra: Optional[dict] = None) -> ChemotaxisResult:
        from brian2 import ms

        c = self.circuit
        p = c.params
        t = np.arange(0.0, c.t_total_ms_sess, p.dt_ms)
        t_arr = np.asarray(c._sp.t / ms)
        i_arr = np.asarray(c._sp.i)
        spikes: Dict[str, np.ndarray] = {}
        for role, idx in c.role_index.items():
            times = t_arr[i_arr == idx]
            spikes[f"{role.lower()}_node3"] = times
            spikes[f"{role.lower()}_soma"] = times
        mus_map = {}
        for i, ch in enumerate(c.muscle3.channels):
            var = f"c_{ch}"
            mus_map[ch] = np.array(getattr(self.mons_mus[i], var)[0])
        meta = dict(t_total_ms=c.t_total_ms_sess, seed=p.seed,
                    n_epochs=self._n_epochs, dt_ms=c.dt_ms, method=c.method,
                    scale=c.scale, fidelity=c.fidelity, grouped=True)
        if meta_extra:
            meta.update(meta_extra)
        return ChemotaxisResult(
            t_ms=t, v_mv={}, spike_times_ms=spikes,
            c_fwd=mus_map.get("fwd", np.zeros(1)),
            c_left=mus_map.get("left", np.zeros(1)),
            c_right=mus_map.get("right", np.zeros(1)),
            meta=meta,
            x=None if x is None else np.asarray(x, dtype=float),
            y=None if y is None else np.asarray(y, dtype=float),
            theta=None if theta is None else np.asarray(theta, dtype=float),
        )


def make_worm_circuit(scale: int = 20, fidelity: str = "point", **kwargs):
    """WormCircuit 工厂。

    - point 保真度 → GroupedWormCircuit（单组批量组装：冷编译预算 M4 L16/L25 +
      M5 §8 风险表；真实连接组多缝隙拓扑只能走 grouped，L7）；
    - 双隔室/多隔室（≤50 子图）→ WormCircuit component 模式（M2 组件复用）。
    传 gap_mode="none" 时 point 档也跳过缝隙（一致性对照用）。
    """
    if fidelity == "point":
        return GroupedWormCircuit(scale=scale, fidelity=fidelity, **kwargs)
    return WormCircuit(scale=scale, fidelity=fidelity, **kwargs)
