"""M8 果蝇幼虫全脑降阶电路：LarvaCircuit（3,016 神经元 / ~0.5M 突触，grouped 批量组装）。

对应《生物仿真M8实施清单》§1 D1（降阶模型规模化）与 §4 步骤 2（铁律 C 三组缩放扫描）：
  - **grouped 批量模式唯一路径**（M5 L19 教训：302 component 3,633 对象冷编译 5-6h 不可行，
    grouped ~10min）——全部点神经元合为一个 NeuronGroup（逐神经元参数），化学突触按
    递质类型各一个 Synapses（每连接 gmax/delay 向量化），缝隙一个 Synapses（双 summed
    目标 I_gap_in/I_gap_out，M5 L18 坑），肌肉每通道一个驱动 Synapses；
  - **稀疏 stim 编码**（D1，结构性限制 §0.7 #1b）：(n_steps × 3,016) 全矩阵 float64 ≈ 7.2GB
    不可行 → 仅刺激角色列物化：(n_steps × n_cols)，n_cols = 刺激角色并集 + 1 个零列
    （≈50 列 → ~120MB@point/0.1ms/30s 窗）；神经元方程以 `stim(t, stim_col)` 引用
    （stim_col = 逐神经元整型列索引变量，本节点实测 Brian2 2.6.0 支持，_probe_m8_sparse_stim.py）；
  - **规模轴 (300, 1000, 3016)**：类平衡分层 + 功能模块保证（M5 L7 教训：纯拓扑序子集
    无运动神经元 → 行为伪迹）；302 为 C. elegans 方法论锚（非幼虫子集）；
  - **保真度轴 (point, two_comp, hh)**：point=grouped 单隔室 HH；two_comp=grouped 2N
    双隔室（soma+node3 轴向耦合，每神经元 2 个索引，linked_var 对交换）；hh=M1
    多隔室 HH 局部子图（≤300，component 模式，M5 哲学）；
  - **可塑性轴 (none, stp, stdp, stdp_homeo)**（并入行为指标）：stp=Tsodyks–Markram
    全化学突触（m6 CSV 定稿参数）；stdp=成对 STDP 限 KC→MBON 子集（M6 #1 STDP 饱和
    教训：限子集+协议窗）；stdp_homeo=+稳态（w 向基线漂移，防饱和）；
  - **夹带双稳态三杠杆（§1 D6，参数化 + enabled 开关消融 sanity）**：
    ① `lever_cmd_desync` 命令层去同步（真实 GABA 抑制边在命令层有效权重）；
    ② `lever_motor_drive` 运动层与命令层分离驱动（自发脉冲只落运动输出层，
       M6 L9#1 教训：命令池注入每脉冲重新点燃夹带）；
    ③ `lever_hetero` 异质权重/传导（类级权重 + 异质延迟，M5 恒等权重教训的反面）。

确定性：p=1/n=1；无噪声（自发脉冲表由固定 seed 伪随机生成，会话内恒定）；
试次方差来自伪随机起点（M4 纪律）；同参数重跑逐位一致。
编译缓存纪律（M4 L16）：dt/形状/命名定稿后不变；stim TimedArray 固定形状
(PROTOCOL_WINDOW_MS, n_cols) + 显式命名；多进程并发 ≤2 worker（M4 L21）。

连接组模式：data/m8_larva_connectome.csv（并行节点 B1a 唯一定稿源；运行期
wait_for_csv 轮询，超时可配；schema 沿 m5_connectome.csv 扩展 larva 列：
分区 region=brain|vnc、神经递质 neurotransmitter、受体 receptor、delay_ms）。
未就绪且 `allow_placeholder=True` 时可用确定性合成占位连接组（仅冒烟验证
装配/扫描机制，**绝不用于真实决策**——连接组是事实不动，M5 P1 纪律）。

复用（冻结文件零修改）：ChemotaxisEnv/TimeDiffTracker/ci_group_stats（M4）、
Muscle3（M4 冻结肌肉组件）、classify_state/StateThresholds/state_fractions
（virtual_body.py）、MultiCompartmentNeuron（M1，hh 档）、load_stdp_params
（plasticity.py，stdp 参数只读）、load_m6/habituation 参数语义（learning.py 只读）。
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

from neural_exploration.src.chemotaxis_env import (  # noqa: E402
    ChemotaxisEnv,
    TimeDiffTracker,
    ci_group_stats,
)
from neural_exploration.src.virtual_body import (  # noqa: E402
    StateThresholds,
    VirtualBody,
    classify_state,
    state_fractions,
)

#: 幼虫连接组唯一定稿源（并行节点 B1a 产出；运行期 wait_for_csv 轮询）
DEFAULT_CONNECTOME_CSV = os.path.join(ROOT, "neural_exploration", "data",
                                      "m8_larva_connectome.csv")
#: 协议窗口（固定 stim 形状上限，ms）：继承 M5 L27 教训——形状定稿后不变，
#: 窗口内任意 T 复用同一编译产物；幼虫协议单试次 T≤30s（清单 §0.7 #2）。
PROTOCOL_WINDOW_MS = 30000.0
#: 固定形状下限（M3/M4 惯例）
STIM_WINDOW_MS = 500.0

#: 规模轴（幼虫连接组子集；302 为 C. elegans 方法论锚，非幼虫子集——报告注明）
SCALE_AXIS = (300, 1000, 3016)
#: 保真度轴（dt 并入保真度档，不做独立网格；M8 D1：HH 仅 ≤300 档短协议）
FIDELITY_AXIS = ("point", "two_comp", "hh")
#: 每保真度档固定 dt（ms）与 method（M5 L17 实测定稿语义沿用）
FIDELITY_DT = {"point": (0.1, "exponential_euler"),
               "two_comp": (0.05, "exponential_euler"),
               "hh": (0.01, "exponential_euler")}
#: 可塑性轴（并入行为指标；LI 出现/消失阈值判据）
PLASTICITY_AXIS = ("none", "stp", "stdp", "stdp_homeo")

#: 点面积（cm²）：球体 d=20µm（M5 同值，保证突触电导换算一致）
SOMA_AREA_CM2 = 1.257e-5
#: 双隔室轴突末梢（node3）面积（cm²）：π·d·L，d=1.5µm、L=2µm（M1 郎飞结，M5 同值）
NODE_AREA_CM2 = np.pi * 1.5 * 2.0 * 1e-8
#: 轴向耦合电导（S）：soma→node3（M5 L17 实测定稿 _AXIAL_G_S=5.4e-9）
AXIAL_G_S = 5.4e-9

#: 类级缩放桶（§1 D5：按 (pre 类, post 类) 分桶；w_ij = w0_class · s_k；
#: w0_class = 连接组 g_max_ns 占位（ampa 5.0nS / gaba 15.0nS，M5 L4 惯例）。
#: 默认全部 1.0 = M5 302 先验恒等（§6 权重校准节点的起点，扫描用先验）。
CLASS_PAIRS = (
    ("sensory", "sensory"), ("sensory", "inter"), ("sensory", "motor"),
    ("inter", "inter"), ("inter", "motor"), ("inter", "sensory"),
    ("motor", "inter"), ("motor", "motor"), ("motor", "sensory"),
)
DEFAULT_CLASS_SCALES: Dict[Tuple[str, str], float] = {
    pair: 1.0 for pair in CLASS_PAIRS}

#: 缝隙全局缩放（M5 L38 定稿 gap_scale=0.05：缝隙分流使 AVB 静默；幼虫沿用先验）
DEFAULT_GAP_SCALE = 0.05
#: 突触类型缩放（ampa/gaba；None → 恒等 1.0）
DEFAULT_SYN_TYPE_SCALES: Dict[str, float] = {"ampa": 1.0, "gaba": 1.0}

#: 功能模块角色前缀（自动识别；B1a CSV 有 region/neuron_class 列时优先用列，
#: 前缀为回退/补充——命名以 Winding 2023 惯例为准，本表为常见前缀草案）
ROLE_PREFIXES = {
    "awc": ("AWC",),            # 嗅觉感觉对（AWC on/off）
    "md": ("MD",),              # IV 类伤害感受器（痛觉）
    "kc": ("KC",),              # 蘑菇体 Kenyon 细胞
    "mbon": ("MBON",),          # 蘑菇体输出神经元
    "dan": ("DAN", "MB-DAN"),   # 多巴胺能奖赏神经元
    "motor_fwd": ("DB", "VB", "MN-FWD", "MOTOR-FWD"),
    "motor_back": ("DA", "VA", "MN-BACK", "MOTOR-BACK"),
    "motor_curl": ("MN-CURL", "MOTOR-CURL"),
}

#: STP 定稿参数（m6_learning_params.csv habituation 段同值，M6 定稿）
DEFAULT_STP = dict(u0=0.6, tau_fac_ms=10.0, tau_rec_ms=1000.0)
#: 扫描探针 STDP 学习率（可塑性轴判据用；成对 STDP 振幅取自
#: data/m6_learning_params.csv stdp 段 a_plus/a_minus——扫描短协议内可读出
#: LI 出现/消失阈值；P5 全协议定稿学习率由学习验证节点定稿）
SCAN_STDP_ETA = 10.0
#: 稳态可塑性权重漂移率（+稳态档：dw/dt = η_h·(w0 − w) 防饱和，M6 #1 STDP 饱和教训）
SCAN_HOMEO_ETA = 0.02
#: 学习探针 LI 出现/消失阈值（|LI| 判据；机制级）
LI_APPEAR_THRESHOLD = 0.05
LI_DISAPPEAR_THRESHOLD = 0.01

#: 静息静默比例 G1 带（草案 [50,90]%，§1 D6；成像文献校准定稿后以
#: data/m8_behavior_reference.csv 为准——B1a 交付前用本草案）
SILENT_BAND = (0.50, 0.90)
#: 双状态「行为 bout」活动下限（fwd+turn 时间比例 > 本值 → 有行为 bout）
BOUT_ACTIVITY_FLOOR = 0.10
#: 机制 A 转向触发阈值（ΔC/ms）：M4 定稿 mechanism_a.theta_pir=1e-6
#: （data/m4_chemotaxis_params.csv；B1c L24 修正 larva 版 -0.5 不可达 bug）
PIR_THETA_S = 1.0e-6


# --------------------------------------------------------------------- #
# 连接组规格（m8 版：+ region/neurotransmitter）
# --------------------------------------------------------------------- #
@dataclass
class ChemRow:
    pre: str
    post: str
    syn_type: str          # ampa|gaba|nmda
    g_ns: float
    delay_ms: float
    pre_site: str = "node3"
    post_site: str = "soma"
    weight: int = 1        # 突触计数（多重性，informational；B1a CSV 有则填）


@dataclass
class GapRow:
    a: str
    b: str
    g_ns: float
    delay_ms: float = 0.05


@dataclass
class MuscleRow:
    motor: str
    channel: str
    w: float


@dataclass
class ConnectomeSpec:
    """幼虫连接组规格（B1a 唯一定稿源解析产物；schema 沿 m5 扩展）。"""

    neurons: "OrderedDict[str, dict]" = field(default_factory=dict)  # name -> {neuron_class, neurotransmitter, region, celltype, side, skid}
    chem: List[ChemRow] = field(default_factory=list)      # 可用（ampa/gaba）
    chem_all: List[ChemRow] = field(default_factory=list)  # 全对（含 none 占位）
    gaps: List[GapRow] = field(default_factory=list)
    muscles: List[MuscleRow] = field(default_factory=list)
    tonic_uA_cm2: Dict[str, float] = field(default_factory=dict)
    source: str = ""
    is_placeholder: bool = False
    #: B1a raw 功能注解 join 结果：role -> tags（olfactory/noci/celltype:* 等）
    functional_tags: Dict[str, set] = field(default_factory=dict)

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
            neurons={n: v for n, v in self.neurons.items() if n in keep},
            chem=[r for r in self.chem if r.pre in keep and r.post in keep],
            chem_all=[r for r in self.chem_all
                      if r.pre in keep and r.post in keep],
            gaps=[r for r in self.gaps if r.a in keep and r.b in keep],
            muscles=[r for r in self.muscles if r.motor in keep],
            tonic_uA_cm2={n: v for n, v in self.tonic_uA_cm2.items()
                          if n in keep},
            source=self.source, is_placeholder=self.is_placeholder,
            functional_tags={n: t for n, t in self.functional_tags.items()
                             if n in keep},
        )


def wait_for_csv(csv_path: Optional[str] = None, timeout_s: float = 3600.0,
                 interval_s: float = 15.0) -> str:
    """轮询等待 B1a 的 m8_larva_connectome.csv（运行期读取；超时抛 FileNotFoundError）。

    清单 §4.1：连接组数据管线（G2 门）为下游前提——扫描工具默认等待（不静默回退）。
    """
    path = csv_path or DEFAULT_CONNECTOME_CSV
    t0 = time.time()
    while not os.path.exists(path):
        if time.time() - t0 > timeout_s:
            raise FileNotFoundError(
                f"等待 {timeout_s:.0f}s 后 m8_larva_connectome.csv 仍未生成：{path}"
                f"（B1a 连接组管线未交付——G2 数据门未过，不能做真实缩放扫描）")
        time.sleep(interval_s)
    return path


def _nt_to_receptor(nt: str) -> str:
    """递质 → 受体映射（M5 L4 惯例 + B1a 全名标注；ach/glut→ampa、gaba→gaba、
    调质→mod——调质为功能门控占位（M6 惯例），不臆造受体作用域）。"""
    nt = (nt or "").strip().lower()
    return {"ach": "ampa", "glut": "ampa", "gaba": "gaba",
            "cholinergic": "ampa", "glutamatergic": "ampa",
            "gabaergic": "gaba",
            "dopamine": "mod", "serotonin": "mod", "octopamine": "mod",
            "tyramine": "mod", "dopaminergic": "mod", "serotonergic": "mod",
            "octopaminergic": "mod", "tyraminergic": "mod",
            "other": "none"}.get(nt, "none")


def load_connectome(csv_path: Optional[str] = None, poll_s: float = 0.0,
                    timeout_s: float = 3600.0,
                    annotations_path: Optional[str] = None) -> ConnectomeSpec:
    """读入幼虫连接组规格（B1a 唯一定稿源；poll_s>0 时轮询等待）。

    annotations_path：B1a raw 功能注解 CSV（winding_s1 的 annotations.csv，
    只读 join：skid → olfactory/noci 等附加标注，供功能模块识别——不改动
    连接组 CSV；缺省 None → 不 join）。
    未就绪且 poll_s=0 → 返回空 spec（source=""，调用方自行处置）。
    """
    path = csv_path or DEFAULT_CONNECTOME_CSV
    if not os.path.exists(path) and poll_s > 0:
        path = wait_for_csv(path, timeout_s=timeout_s)
    if os.path.exists(path):
        return _parse_connectome_csv(path, annotations_path=annotations_path)
    return ConnectomeSpec(source="")


def _clean_line(ln: str) -> str:
    """去注释/头行外层引号（M5-B1d L23 教训：子图 CSV 列头带引号）。"""
    s = ln.strip()
    if s.startswith('"'):
        s = s.strip('"')
    return s


def _parse_connectome_csv(path: str,
                          annotations_path: Optional[str] = None) -> ConnectomeSpec:
    """解析 m8_larva_connectome.csv（schema 沿 m5_connectome.csv 扩展 larva 列：
    role, region, neuron_class, neurotransmitter, receptor, synapse_from,
    synapse_to, synapse_type, g_max_ns, delay_ms, g_gap_ns, muscle_target,
    skid, side, celltype, weight, note）。

    - 神经元行：role=<名>（≠muscle_drive）+ neuron_class + neurotransmitter
      + region + celltype + side + skid；
    - 化学行：synapse_type=chem + receptor ∈ {ampa, gaba}（递质→受体映射：
      cholinergic/glutamatergic→ampa、GABAergic→gaba、调质/other→none 跳过——
      调质功能门控为 M6 惯例，不臆造受体作用域，清单 §3.3）；
      ⚠ B1a 当前交付：neurotransmitter 全为 other（无 CATMAID 递质标注，
      如实登记）→ 0 可用化学边——`nt_fallback` 参数提供显式临时回退；
    - 缝隙行：synapse_type=gap（幼虫脑连接组权威 = 0 缝隙，论文确认）；
    - 肌肉行：role=muscle_drive（幼虫脑连接组不含肌肉，运动神经元在 VNC
      分析图外 → B1a 0 行；`provisional_muscles` 提供显式临时映射）。
    """
    import csv as _csv

    spec = ConnectomeSpec(source=path)
    n_mod_skipped = 0
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(_csv.DictReader(
            _clean_line(ln) for ln in f
            if _clean_line(ln) and not _clean_line(ln).startswith("#")))
    # 第一遍：skid → role 映射（B1a CSV：神经元 role=名字/KC #0，而化学行
    # synapse_from/to = 原始 skeleton id——须经 skid 列解析为 role）
    skid_map: Dict[str, str] = {}
    for r in rows:
        role = (r.get("role") or "").strip().upper()
        sk = (r.get("skid") or "").strip()
        if role and role != "MUSCLE_DRIVE" and sk:
            skid_map[sk] = role

    def _resolve(sid: str) -> str:
        sid = sid.strip().upper()
        if sid in skid_map:
            return skid_map[sid]
        return sid  # 已是 role（占位/命名连接组）

    for r in rows:
        role = (r.get("role") or "").strip().upper()
        frm_raw = (r.get("synapse_from") or "").strip()
        stype = (r.get("synapse_type") or "").strip().lower()
        if role and role != "MUSCLE_DRIVE":
            spec.neurons[role] = dict(
                neuron_class=(r.get("neuron_class") or "inter").strip().lower(),
                neurotransmitter=(r.get("neurotransmitter") or "").strip().lower(),
                region=(r.get("region") or "").strip().lower(),
                receptor=(r.get("receptor") or "").strip().lower(),
                celltype=(r.get("celltype") or "").strip(),
                side=(r.get("side") or "").strip().lower(),
                skid=(r.get("skid") or "").strip(),
            )
        if stype == "chem" and frm_raw:
            frm = _resolve(frm_raw)
            receptor = (r.get("receptor") or "").strip().lower()
            if not receptor:
                receptor = _nt_to_receptor(r.get("neurotransmitter"))
            if receptor in ("ampa", "gaba"):
                pre_site = (r.get("pre_site") or "node3").strip().lower()
                post_site = (r.get("post_site") or "soma").strip().lower()
                w = r.get("synapse_count") or r.get("weight")
                row = ChemRow(
                    pre=frm, post=_resolve(r.get("synapse_to") or ""),
                    syn_type=receptor,
                    g_ns=float(r.get("g_max_ns") or 5.0),
                    delay_ms=float(r.get("delay_ms") or 0.5),
                    pre_site=pre_site, post_site=post_site,
                    weight=int(float(w)) if w else 1)
                spec.chem.append(row)
                spec.chem_all.append(row)
            else:
                # 'none'/mod 占位（B1a 递质标注不完整 → 其余行）：
                # 保留全对供 nt_fallback 显式临时回退（syn_type='none' 不入可用集）
                spec.chem_all.append(ChemRow(
                    pre=frm, post=_resolve(r.get("synapse_to") or ""),
                    syn_type="none",
                    g_ns=float(r.get("g_max_ns") or 5.0),
                    delay_ms=float(r.get("delay_ms") or 0.5),
                    weight=int(float(r.get("synapse_count")
                                     or r.get("weight") or 1))))
                n_mod_skipped += 1
        elif stype == "gap" and frm:
            spec.gaps.append(GapRow(
                a=frm, b=(r.get("synapse_to") or "").strip().upper(),
                g_ns=float(r.get("g_gap_ns") or r.get("g_max_ns") or 0.5),
                delay_ms=float(r.get("delay_ms") or 0.05)))
        elif stype == "muscle" and frm:
            ch = (r.get("synapse_to") or "").strip().lower()
            for prefix in ("body_", "head_"):
                if ch.startswith(prefix):
                    ch = ch[len(prefix):]
                    break
            spec.muscles.append(MuscleRow(motor=frm, channel=ch,
                                          w=float(r.get("g_max_ns") or 0.3)))
    spec.n_mod_skipped = n_mod_skipped
    # B1a raw 功能注解 join（只读；skid_<id> ↔ annotations left_id/right_id）
    spec.functional_tags: Dict[str, set] = {}
    if annotations_path and os.path.exists(annotations_path):
        with open(annotations_path, newline="", encoding="utf-8") as f:
            for ar in _csv.DictReader(_clean_line(ln) for ln in f
                                      if _clean_line(ln)
                                      and not _clean_line(ln).startswith("#")):
                tags = set()
                ann = (ar.get("additional_annotations") or "").strip()
                if ann and ann != "no official annotation":
                    tags.update(t.strip() for t in ann.split(";") if t.strip())
                ct = (ar.get("celltype") or "").strip()
                if ct:
                    tags.add(f"celltype:{ct}")
                for key in ("left_id", "right_id"):
                    sid = (ar.get(key) or "").strip()
                    if sid and sid != "no pair":
                        # ⚠ B1c 实测坑 L23（2026-08-28）：旧逻辑 rkey=f"SKID_{sid}"
                        # 永不命中 skid_map（键为原始 skid，如 "40045"）→
                        # functional_tags 全空 → sens_roles 回退首 2 感觉神经元
                        # （含光感受器 Rh6PR，嗅觉链断裂 → CI=0/LI=0 根因）。
                        # 修复：优先以原始 skid 命中（f"SKID_{sid}" 回退保留，
                        # 兼容占位/预置命名连接组）；未改签名/默认行为
                        # （annotations_path=None 时无任何行为变化）。
                        rkey = sid if sid in skid_map else f"SKID_{sid}"
                        if rkey in skid_map:
                            rkey = skid_map[rkey]
                        spec.functional_tags.setdefault(rkey, set()).update(tags)
    return spec


def _topo_order(spec: ConnectomeSpec) -> List[str]:
    """拓扑序（sensory → inter → motor；类内保持 CSV 顺序）。"""
    order = []
    for cls in ("sensory", "inter", "motor"):
        order += [n for n, c in spec.neurons.items()
                  if c.get("neuron_class") == cls]
    for n in spec.neurons:
        if n not in order:
            order.append(n)
    return order


def _prefix_roles(spec: ConnectomeSpec, prefixes: Sequence[str]) -> List[str]:
    """按前缀匹配角色（有序，按拓扑序）。"""
    order = _topo_order(spec)
    return [n for n in order if any(n.startswith(p) for p in prefixes)]


def _functional_must_include(spec: ConnectomeSpec) -> List[str]:
    """功能模块必保集合（§3.4 规模子集规则：嗅觉 AWC / 痛觉 MD / 蘑菇体
    KC·MBON·MBIN / 命令层 DN / 运动池——子集必须含行为驱动链，否则行为
    伪迹，M5 L7 教训）。celltype 列优先（B1a CSV：KC/MBON/MBIN/DN-*），
    名称前缀回退（占位/命名连接组）。"""
    must: List[str] = []

    def _by_celltype(*cts: str, limit: int = 8) -> List[str]:
        want = {c.lower() for c in cts}
        return [n for n, v in spec.neurons.items()
                if (v.get("celltype") or "").strip().lower() in want][:limit]

    # KC/MBON 学习底物优先（KC→MBON 直连边在子集内——可塑性轴 stdp 可测；
    # 实测：首 8 KC 可能无 MBON 出边 → stdp 底物缺失，LI 不可测，M 实测坑）
    kc_all = [n for n, v in spec.neurons.items()
              if (v.get("celltype") or "").strip().lower() == "kc"]
    mbon_all = [n for n, v in spec.neurons.items()
                if (v.get("celltype") or "").strip().lower() == "mbon"]
    mbon_set = set(mbon_all)
    kc_all_set = set(kc_all)
    # 用 chem_all（含未标注行）——KC→MBON 边多在递质未标注行，仅查 chem
    # 会漏（M 实测坑：300 子集 stdp 底物 0 边的根因）
    kc_with_out = [n for n in kc_all
                   if any(r.pre == n and r.post in mbon_set
                          for r in spec.chem_all)]
    mbon_with_in = [n for n in mbon_all
                    if any(r.post == n and r.pre in kc_all_set
                           for r in spec.chem_all)]
    must += (kc_with_out[:8] if kc_with_out else kc_all[:8])
    must += (mbon_with_in[:8] if mbon_with_in else mbon_all[:8])
    must += _by_celltype("MBIN", limit=8)
    must += _by_celltype("pre-DN-VNC", "DN-VNC", "pre-DN-SEZ", "DN-SEZ",
                         limit=8)
    # 嗅觉链（AWC 类 ORN + PN）：嗅觉 ORN → PN → KC 链在子集内完整，
    # 否则趋化 CI/学习探针链断裂（M 实测坑：300 子集 ORN→PN→KC 缺环）
    orn = [n for n, v in spec.neurons.items()
           if any(t.lower().startswith("olfactory")
                  for t in spec.functional_tags.get(n, set()))]
    must += orn[:4]
    must += _by_celltype("PN", "PN-somato", limit=8)
    for key, pref in (("awc", ROLE_PREFIXES["awc"]),
                      ("md", ROLE_PREFIXES["md"]),
                      ("kc", ROLE_PREFIXES["kc"]),
                      ("mbon", ROLE_PREFIXES["mbon"]),
                      ("motor_fwd", ROLE_PREFIXES["motor_fwd"]),
                      ("motor_back", ROLE_PREFIXES["motor_back"])):
        hits = _prefix_roles(spec, pref)
        must += hits[:8]
    # 运动神经元兜底（无前缀匹配时按 neuron_class=motor）
    if not any(n in must for n in spec.neurons
               if spec.neurons[n].get("neuron_class") == "motor"):
        for n in _topo_order(spec):
            if spec.neurons[n].get("neuron_class") == "motor":
                must.append(n)
            if sum(1 for x in must
                   if spec.neurons.get(x, {}).get("neuron_class") == "motor") >= 6:
                break
    return [n for n in dict.fromkeys(must) if n in spec.neurons]


def scale_names(spec: ConnectomeSpec, scale: int) -> List[str]:
    """规模轴子集（预注册规则，不破坏连接组事实）：

    - scale=3016：全 roster（真实 CSV 下须 = 3,016 ± 0，P1 判据）；
    - 300/1000：类平衡分层抽样（按全 roster 类构成比例配额）+ 功能模块必保集合
      （_functional_must_include）先占位，余量按拓扑序配额内补齐——保证运动
      神经元与行为链入子集（M5 L7 教训）；
    - scale > roster 大小（占位冒烟/空 CSV）：返回全 roster。
    """
    order = _topo_order(spec)
    if scale >= len(order):
        return list(order)
    must = _functional_must_include(spec)
    classes = ("sensory", "inter", "motor")
    per = {c: [n for n in order if spec.neurons.get(n, {}).get("neuron_class") == c]
           for c in classes}
    total = len(order)
    quota = {c: max(1, round(scale * len(v) / total))
             for c, v in per.items()}
    # 必保集合消耗配额
    for n in must:
        c = spec.neurons.get(n, {}).get("neuron_class", "inter")
        if quota.get(c, 0) > 0:
            quota[c] -= 1
    while sum(quota.values()) < scale - len(must):
        for c in classes:
            if sum(quota.values()) >= scale - len(must):
                break
            if len(per[c]) > quota[c]:
                quota[c] += 1
    out = list(must)
    for c in classes:
        for n in per[c]:
            if n in out or quota[c] <= 0:
                continue
            out.append(n)
            quota[c] -= 1
            if len(out) >= scale:
                break
        if len(out) >= scale:
            break
    return out[:scale]


# --------------------------------------------------------------------- #
# 确定性合成占位连接组（仅冒烟验证装配/扫描机制；**绝不用于真实决策**）
# --------------------------------------------------------------------- #
def build_placeholder_spec(n_neurons: int = 300, seed: int = 20260827
                           ) -> ConnectomeSpec:
    """确定性合成占位连接组（machinery smoke test only）。

    - 明确标注 is_placeholder=True（扫描工具据此拒绝写入真实决策）；
    - 结构：sensory 80 / inter 120 / motor 100；region brain|vnc；
      递质 ach/gaba/glut；功能角色 AWC on/off、MD 伤害感受器、KC/MBON/DAN、
      命令 inter（brain→vnc）、运动池 DA*/DB*/VA*/VB* + 肌肉行；
    - 化学/缝隙边按 (pre 类, post 类) 概率生成（确定性 seed）；g_max_ns 占位
      （ampa 5.0 / gaba 15.0）；delay 0.5±抖动（杠杆③异质延迟的数据基础）。
    """
    rng = np.random.default_rng(seed)
    spec = ConnectomeSpec(source="PLACEHOLDER_SYNTHETIC",
                          is_placeholder=True)
    names: List[Tuple[str, str, str, str]] = []  # (name, class, region, nt)

    def _add(n, cls, region, nt):
        spec.neurons[n] = dict(neuron_class=cls, region=region,
                               neurotransmitter=nt,
                               receptor=_nt_to_receptor(nt))
        names.append((n, cls, region, nt))

    # 功能角色（先加，保证必保集合命中）
    _add("AWC_ON", "sensory", "brain", "glut")
    _add("AWC_OFF", "sensory", "brain", "glut")
    for k in range(2):
        _add(f"MD{k + 1}", "sensory", "vnc", "glut")
    for k in range(24):
        _add(f"KC{k + 1}", "inter", "brain", "ach")
    for k in range(8):
        _add(f"MBON{k + 1}", "inter", "brain", "ach")
    for k in range(2):
        _add(f"DAN{k + 1}", "inter", "brain", "dopamine")
    for k in range(10):
        _add(f"CMD{k + 1}", "inter", "brain", "ach")   # 命令（brain→vnc）
    for k in range(4):
        _add(f"INH{k + 1}", "inter", "brain", "gaba")  # 抑制性命令（杠杆①数据）
    for k in range(16):
        _add(f"DA{k + 1}", "motor", "vnc", "ach")
    for k in range(16):
        _add(f"DB{k + 1}", "motor", "vnc", "ach")
    # 其余类平衡补齐
    n_add = n_neurons - len(names)
    per = dict(sensory=n_add * 80 // 300, inter=n_add * 120 // 300,
               motor=n_add * 100 // 300)
    per["inter"] += n_add - per["sensory"] - per["inter"] - per["motor"]
    for cls, cnt in (("sensory", per["sensory"]), ("inter", per["inter"]),
                     ("motor", per["motor"])):
        for k in range(cnt):
            region = "brain" if cls in ("sensory", "inter") else "vnc"
            nt = rng.choice(["ach", "gaba", "glut"], p=[0.5, 0.2, 0.3])
            _add(f"{cls.upper()}_{k + 1}", cls, region, nt)

    # 化学边（含命令层真实抑制边：CMD/INH→CMD，杠杆①数据）
    cls_of = {n: c for n, c, _r, _t in names}
    for pre, _c, _r, _t in names:
        p_edge = {"sensory": 0.25, "inter": 0.20, "motor": 0.15}[cls_of[pre]]
        posts = [n for n, c, _r, _t in names if c != "sensory" or True]
        posts = [n for n in posts if n != pre]
        rng.shuffle(posts)
        for post in posts[:max(1, int(len(posts) * p_edge))]:
            if rng.random() < 0.05:
                continue
            nt_post = spec.neurons[pre]["neurotransmitter"]
            stype = "gaba" if nt_post == "gaba" else "ampa"
            g = 15.0 if stype == "gaba" else 5.0
            delay = float(np.clip(0.5 + rng.normal(0, 0.3), 0.1, 3.0))
            spec.chem.append(ChemRow(pre=pre, post=post, syn_type=stype,
                                     g_ns=g, delay_ms=delay, weight=1))
    # KC→MBON 专桶（STDP 子集）
    for kc in [n for n, _c, _r, _t in names if n.startswith("KC")]:
        for mbon in [n for n, _c, _r, _t in names if n.startswith("MBON")]:
            spec.chem.append(ChemRow(pre=kc, post=mbon, syn_type="ampa",
                                     g_ns=5.0, delay_ms=0.5, weight=1))
    # 缝隙（稀疏）
    for _ in range(n_neurons):
        a = names[rng.integers(len(names))][0]
        b = names[rng.integers(len(names))][0]
        if a != b:
            spec.gaps.append(GapRow(a=a, b=b, g_ns=0.5, delay_ms=0.05))
    # 肌肉行（运动池 → 通道）
    for n, c, _r, _t in names:
        if c == "motor":
            if n.startswith("DA") or n.startswith("VA"):
                ch = "back"
            elif n.startswith("DB") or n.startswith("VB"):
                ch = "fwd"
            else:
                ch = ["fwd", "back", "left", "right"][
                    int(rng.integers(4))]
            spec.muscles.append(MuscleRow(motor=n, channel=ch,
                                          w=float(rng.uniform(0.1, 0.4))))
    # 去重
    seen = set()
    spec.chem = [r for r in spec.chem if not (r.pre, r.post) in seen
                 and not seen.add((r.pre, r.post))]
    return spec


# --------------------------------------------------------------------- #
# LarvaCircuit：3,016 神经元 grouped 批量组装
# --------------------------------------------------------------------- #
class LarvaCircuit:
    """幼虫全脑降阶电路（grouped 批量组装 + 稀疏 stim 编码 + 三杠杆 + 可塑性）。

    构造参数默认 None（M3 L13）：None → 以数据文件/预注册默认值为准。
    """

    def __init__(
        self,
        scale: int = 3016,
        fidelity: str = "point",
        csv_path: Optional[str] = None,
        connectome_poll_s: float = 0.0,
        connectome_timeout_s: float = 3600.0,
        allow_placeholder: bool = False,
        seed: int = 0,
        name_prefix: str = "larva",
        dt_ms: Optional[float] = None,
        method: Optional[str] = None,
        t_total_ms: Optional[float] = None,
        class_scales: Optional[Dict[Tuple[str, str], float]] = None,
        gap_scale: Optional[float] = None,
        syn_type_scales: Optional[Dict[str, float]] = None,
        gmax_scale: Optional[float] = None,
        # 可塑性轴
        plasticity: str = "none",
        stdp_edges: Optional[Sequence[Tuple[str, str]]] = None,
        stdp_eta: float = SCAN_STDP_ETA,
        homeo_eta: float = SCAN_HOMEO_ETA,
        # 夹带双稳态三杠杆（§1 D6；enabled 开关消融 sanity）
        lever_cmd_desync: bool = True,
        lever_motor_drive: bool = True,
        lever_hetero: bool = True,
        cmd_gaba_scale: float = 1.5,
        motor_drive_nA: float = 0.10,
        motor_drive_hz: float = 2.0,
        motor_drive_ms: float = 3.0,
        spont_seed: int = 20260827,
        # 行为上下文（M4/M5 参数；扫描用短协议）
        v_fwd0: float = 1.0,
        v_rev0: float = 1.0,
        omega_max: float = 1.0,
        dt_b: float = 25.0,
        arena_L: float = 10.0,
        transduction_g_on: float = 8.0e6,   # M4 定稿（µA/cm² per s；B1c2 校准 g=8e6）
        transduction_g_off: float = 8.0e6,
        transduction_tau_win_ms: float = 100.0,
        escape_density_uA_cm2: float = 60.0,
        escape_start_ms: float = 100.0,
        escape_dur_ms: float = 20.0,
        muscle_tau_ms: float = 20.0,
        muscle_cap: float = 1.0,
        spec_override: Optional[ConnectomeSpec] = None,
        annotations_path: Optional[str] = None,
        nt_fallback: Optional[str] = None,
        provisional_muscles: bool = False,
    ):
        if scale not in SCALE_AXIS:
            raise ValueError(f"规模需为 {SCALE_AXIS}：{scale}")
        if fidelity not in FIDELITY_AXIS:
            raise ValueError(f"保真度需为 {FIDELITY_AXIS}：{fidelity}")
        if plasticity not in PLASTICITY_AXIS:
            raise ValueError(f"可塑性需为 {PLASTICITY_AXIS}：{plasticity}")
        if fidelity in ("two_comp", "hh") and scale > 1000:
            raise ValueError(
                f"two_comp/hh 档仅限 ≤1000（hh 限 ≤300 短协议，D1）：规模 {scale}")
        if fidelity == "hh" and scale > 300:
            raise ValueError("HH（多隔室）档仅限 ≤300 档短协议 T≤5s（M8 D1）")
        self.scale = int(scale)
        self.fidelity = fidelity
        self.plasticity = plasticity
        self.name_prefix = name_prefix
        self.seed = int(seed)
        self.dt_ms = FIDELITY_DT[fidelity][0] if dt_ms is None else float(dt_ms)
        self.method = FIDELITY_DT[fidelity][1] if method is None else method
        self.t_total_ms = 1000.0 if t_total_ms is None else float(t_total_ms)

        # 权重杠杆（§1 D5 类级缩放 + M5 先验）
        self.class_scales: Dict[Tuple[str, str], float] = dict(
            DEFAULT_CLASS_SCALES if class_scales is None else class_scales)
        self.gap_scale = DEFAULT_GAP_SCALE if gap_scale is None else float(gap_scale)
        self.syn_type_scales: Dict[str, float] = dict(
            DEFAULT_SYN_TYPE_SCALES if syn_type_scales is None
            else syn_type_scales)
        # 全局突触电导缩放（D5 第一遍先验：110k 对 × 5nS 占位 → 平均输入
        # 电导 ≈ 数十× gL → 全网络饱和夹带（实测 silent≈0.15/median 69Hz）。
        # 扫描用第一遍全局缩放进入可工作区（校准节点步骤 4 定稿类级因子）；
        # None → 1.0 恒等（先验）。
        self.gmax_scale = 1.0 if gmax_scale is None else float(gmax_scale)

        # 三杠杆
        self.lever_cmd_desync = bool(lever_cmd_desync)
        self.lever_motor_drive = bool(lever_motor_drive)
        self.lever_hetero = bool(lever_hetero)
        self.cmd_gaba_scale = float(cmd_gaba_scale)
        self.motor_drive_nA = float(motor_drive_nA)
        self.motor_drive_hz = float(motor_drive_hz)
        self.motor_drive_ms = float(motor_drive_ms)
        self.spont_seed = int(spont_seed)

        # 行为上下文
        self.v_fwd0, self.v_rev0, self.omega_max = (float(v_fwd0),
                                                    float(v_rev0),
                                                    float(omega_max))
        self.dt_b = float(dt_b)
        self.arena_L = float(arena_L)
        self.transduction = dict(g_on=float(transduction_g_on),
                                 g_off=float(transduction_g_off),
                                 tau_win_ms=float(transduction_tau_win_ms))
        self.escape = dict(density=float(escape_density_uA_cm2),
                           start_ms=float(escape_start_ms),
                           dur_ms=float(escape_dur_ms))
        self.muscle_tau_ms = float(muscle_tau_ms)
        self.muscle_cap = float(muscle_cap)

        # 连接组（运行期读取；占位仅冒烟）
        if spec_override is not None:
            self.spec: ConnectomeSpec = spec_override
        else:
            self.spec = load_connectome(csv_path, poll_s=connectome_poll_s,
                                        timeout_s=connectome_timeout_s,
                                        annotations_path=annotations_path)
            if not self.spec.neurons and allow_placeholder:
                self.spec = build_placeholder_spec()
        if not self.spec.neurons:
            raise FileNotFoundError(
                f"连接组未就绪（{self.spec.source or DEFAULT_CONNECTOME_CSV}）——"
                f"等待 B1a 数据或 allow_placeholder=True 冒烟")
        self.is_placeholder = bool(self.spec.is_placeholder)
        # ⚠ B1a 交付：S1 矩阵 roster = 2,952 神经元（论文 3,016 含分析图外
        # 64 神经元 + 孤儿位点，两套计数语义——B1a counts.json 诊断 OUT +
        # 三态裁决请求，M5 L7 惯例）。scale=3016 = 全 roster，断言放行并记录。
        if not self.is_placeholder and self.scale == 3016 \
                and self.spec.n_neurons != 3016:
            self.roster_note = (f"B1a roster={self.spec.n_neurons} "
                                f"≠ 论文 3016（64 神经元分析图外，B1a 三态裁决请求）")
        else:
            self.roster_note = ""

        self.names = scale_names(self.spec, self.scale)
        self.sub: ConnectomeSpec = self.spec.subset(self.names)
        # 递质回退（B1a 标注不完整 → nt_fallback='class' = 仅未标注行临时
        # 类级回退，**非权威标注**，结果标记 PROVISIONAL_NT）
        self.nt_fallback = nt_fallback
        self.nt_fallback_active = False
        self.nt_fallback_n_provisional = 0
        if nt_fallback is not None:
            self._apply_nt_fallback()
        # 肌肉回退（幼虫脑连接组无肌肉行——运动神经元在 VNC 分析图外；
        # provisional_muscles=True = 显式临时通道映射，结果标记 PROVISIONAL）
        self.muscle_provisional = False
        if provisional_muscles and not self.sub.muscles:
            self._apply_provisional_muscles()
        self.role_index: Dict[str, int] = {}
        self.group = None
        self.chem_synapses: List[object] = []
        self.gap_synapse = None
        self.mus_drivers: Dict[str, object] = {}
        self.muscle3 = None
        self._sp = None
        self._sp_comp = None      # two_comp：按隔室索引分发的 SpikeMonitor
        self._built = False
        self._build_wall_s = float("nan")
        # 会话侧（make_session 填充）
        self._sess = None
        self._stim_cols: Dict[str, int] = {}
        self._zero_col = 0
        self._stim_n_cols = 0
        self._tonic_nA: Dict[str, float] = {}
        self._stdp_syn = None
        self._stdp_mask = None
        self._stp_params = dict(DEFAULT_STP)
        self._stdp_edges: List[Tuple[str, str]] = []
        self._stdp_eta = float(stdp_eta)
        self._homeo_eta = float(homeo_eta)
        if stdp_edges is not None:
            self._stdp_edges = list(stdp_edges)

    # ------------------------------------------------------------------ #
    # 角色集合（功能模块识别；celltype 优先，名称前缀回退）
    # ------------------------------------------------------------------ #
    def _roles_by_prefix(self, prefixes: Sequence[str]) -> List[str]:
        return [n for n in self.names
                if any(n.startswith(p) for p in prefixes)]

    def _roles_by_celltype(self, *celltypes: str) -> List[str]:
        """按 celltype 列匹配角色（B1a CSV：KC/MBON/MBIN/DN-*/PN/LHN…）。"""
        want = {c.strip().lower() for c in celltypes if c and c.strip()}
        return [n for n in self.names
                if (self.sub.neurons.get(n, {}).get("celltype") or "")
                .strip().lower() in want]

    def _roles_by_tag(self, tag: str) -> List[str]:
        """按 B1a raw 功能注解 tag 匹配（olfactory/noci 等；functional_tags）。"""
        tag = tag.lower()
        return [n for n in self.names
                if any(t.lower().startswith(tag) for t in
                       self.sub.functional_tags.get(n, set()))]

    @property
    def sens_roles(self) -> Tuple[str, str]:
        """嗅觉感觉对：olfactory 注解 → AWC 前缀 → 前 2 个 sensory 类（占位）。"""
        hits = self._roles_by_tag("olfactory")
        if len(hits) >= 2:
            return (hits[0], hits[1])
        hits = self._roles_by_prefix(ROLE_PREFIXES["awc"])
        if len(hits) >= 2:
            return (hits[0], hits[1])
        hits = [n for n in self.names
                if self.sub.neurons.get(n, {}).get("neuron_class") == "sensory"]
        return (hits[0], hits[1]) if len(hits) >= 2 else ("", "")

    @property
    def nociceptor_roles(self) -> Tuple[str, ...]:
        """伤害感受器：noci 注解 → MD 前缀 → 空（如实记录）。"""
        hits = self._roles_by_tag("noci")
        if hits:
            return tuple(hits)
        return tuple(self._roles_by_prefix(ROLE_PREFIXES["md"]))

    @property
    def motor_roles(self) -> Tuple[str, ...]:
        """运动池（neuron_class=motor + 前缀匹配）。"""
        pref = self._roles_by_prefix(
            ROLE_PREFIXES["motor_fwd"] + ROLE_PREFIXES["motor_back"]
            + ROLE_PREFIXES["motor_curl"])
        cls = [n for n in self.names
               if self.sub.neurons.get(n, {}).get("neuron_class") == "motor"]
        out = list(dict.fromkeys(pref + cls))
        return tuple(out)

    @property
    def kc_roles(self) -> Tuple[str, ...]:
        """Kenyon 细胞（celltype=KC 优先）。"""
        hits = self._roles_by_celltype("KC")
        if hits:
            return tuple(hits)
        return tuple(self._roles_by_prefix(ROLE_PREFIXES["kc"]))

    @property
    def mbon_roles(self) -> Tuple[str, ...]:
        """蘑菇体输出神经元（celltype=MBON 优先）。"""
        hits = self._roles_by_celltype("MBON")
        if hits:
            return tuple(hits)
        return tuple(self._roles_by_prefix(ROLE_PREFIXES["mbon"]))

    @property
    def dan_roles(self) -> Tuple[str, ...]:
        """DA 奖赏/蘑菇体输入（celltype=MBIN 优先——MB 输入含 DAN；前缀回退）。"""
        hits = self._roles_by_celltype("MBIN")
        if hits:
            return tuple(hits)
        return tuple(self._roles_by_prefix(ROLE_PREFIXES["dan"]))

    def _turn_driver_roles(self) -> Tuple[str, ...]:
        """转向驱动角色（机制 A：驱动 left/right 肌肉通道的运动池）。"""
        left = {m.motor for m in self.sub.muscles if m.channel == "left"}
        right = {m.motor for m in self.sub.muscles if m.channel == "right"}
        hits = self._roles_by_prefix(("SMD", "MN-LEFT", "MN-RIGHT"))
        out = list(dict.fromkeys(list(hits) + list(left | right)))
        return tuple(o for o in out if o in self.role_index)

    # ------------------------------------------------------------------ #
    # 临时回退（B1a 数据缺口：递质标注 / 肌肉行；**非权威，结果标记**）
    # ------------------------------------------------------------------ #
    def _apply_nt_fallback(self):
        """递质临时回退（nt_fallback='class'）：B1a 当前递质标注不完整
        （cholinergic 25185 / dopaminergic 2344 / other 83148——**无 GABA**）。
        语义：**保留已标注**（ampa/gaba 原样），仅对 'none' 未标注行按
        (pre 类) 分桶分配离子型递质：sensory→ach(ampa)、motor→ach(ampa)、
        inter→ach 80%/gaba 20%（按 role_index 确定性哈希，幼虫脑 ~20%
        GABA 中间神经元惯例）。
        连接（pre/post 对、g_max_ns、delay_ms）为真实连接组事实；未标注
        部分的递质为**临时假设**——结果标记 PROVISIONAL_NT，不作权威决策。
        """
        rows = []
        n_provisional = 0
        for r in self.sub.chem_all:
            if r.syn_type in ("ampa", "gaba"):
                rows.append(r)          # 已标注（权威部分原样保留）
                continue
            pre_cls = self.sub.neurons.get(r.pre, {}).get("neuron_class", "inter")
            if pre_cls in ("sensory", "motor"):
                stype = "ampa"
            else:
                try:
                    h = int(str(r.pre).rsplit("_", 1)[-1])
                except ValueError:
                    h = abs(hash(r.pre))
                stype = "gaba" if (h % 5 == 0) else "ampa"  # ~20% inter GABA
            rows.append(ChemRow(pre=r.pre, post=r.post, syn_type=stype,
                                g_ns=r.g_ns, delay_ms=r.delay_ms,
                                pre_site=r.pre_site, post_site=r.post_site,
                                weight=r.weight))
            n_provisional += 1
        self.sub.chem = rows
        self.nt_fallback_active = True
        self.nt_fallback_n_provisional = n_provisional

    def _apply_provisional_muscles(self):
        """运动池→虚拟通道临时映射（幼虫脑连接组无肌肉行——运动神经元在
        VNC 分析图外，B1a 0 行）：motor 类角色确定性分桶到
        fwd/back/left/right（side 列优先：left→left、right→right，
        其余按 role_index 哈希），w=0.3（M5 肌肉权重量级）。
        映射为**临时假设**（P3 larva_body 节点定稿真实映射），结果标记
        PROVISIONAL_MUSCLE。

        ⚠ B1c 实测坑 L25（2026-08-28）：初版 side 优先把**全部** motor 分到
        left/right（B1a CSV 的 VNC 运动神经元几乎都有 side=left/right），
        k%4 回退不可达 → c_fwd/c_back 恒 0 → 身体只能原地转（fwd=0/CI≡0）。
        P3 规格（清单 §5.1）要求 C_fwd/C_back 行波驱动存在——修正：side 优先
        保留 left/right 转向驱动，但按确定性周期抽 ~1/3 到 fwd/back 行波通道
        （k%3==0 → left→fwd、right→back；否则 side 通道），保证五模式
        （前进/后退/侧转）驱动齐全。未改签名/默认行为。
        """
        rows = []
        for k, n in enumerate(self.names):
            if self.sub.neurons.get(n, {}).get("neuron_class") != "motor":
                continue
            side = self.sub.neurons.get(n, {}).get("side", "")
            if side == "left":
                ch = "fwd" if k % 3 == 0 else "left"
            elif side == "right":
                ch = "back" if k % 3 == 0 else "right"
            else:
                ch = ["fwd", "back", "left", "right"][k % 4]
            rows.append(MuscleRow(motor=n, channel=ch, w=0.3))
        self.sub.muscles = rows
        self.muscle_provisional = True

    def _stdp_edge_pairs(self) -> List[Tuple[str, str]]:
        """STDP 目标边（KC→MBON；显式 stdp_edges 优先，否则自动 KC→MBON）。"""
        if self._stdp_edges:
            return [(a, b) for (a, b) in self._stdp_edges
                    if a in self.role_index and b in self.role_index]
        kc, mbon = self.kc_roles, self.mbon_roles
        pairs = [(a, b) for a in kc for b in mbon
                 if any((r.pre == a and r.post == b) for r in self.sub.chem)]
        if not pairs:
            # 无 KC→MBON 直接边（占位/小规模）→ 空集（STDP 档 LI 不可测，如实记录）
            return []
        return pairs

    def _cmd_layer_roles(self) -> Tuple[str, ...]:
        """命令层角色（GABA 抑制杠杆作用域）：celltype 下行命令类
        （pre-DN-VNC/DN-VNC/pre-DN-SEZ/DN-SEZ，幼虫脑→VNC 下行命令），
        或带命令/抑制前缀（CMD/INH/COMMAND），或 brain→vnc 下行边 pre。"""
        pref = [n for n in self.names
                if n.startswith(("CMD", "INH", "COMMAND"))]
        ct = self._roles_by_celltype("pre-DN-VNC", "DN-VNC", "pre-DN-SEZ",
                                     "DN-SEZ")
        down = set()
        for r in self.sub.chem:
            if r.post in self.role_index and r.pre in self.role_index:
                pre_region = self.sub.neurons.get(r.pre, {}).get("region", "")
                post_region = self.sub.neurons.get(r.post, {}).get("region", "")
                if pre_region == "brain" and post_region == "vnc":
                    down.add(r.pre)
        out = list(dict.fromkeys(list(ct) + pref
                                 + [n for n in self.names if n in down]))
        return tuple(out)

    # ------------------------------------------------------------------ #
    # 组装
    # ------------------------------------------------------------------ #
    def _class_scale_for(self, pre: str, post: str) -> float:
        return self.class_scales.get(
            (self.sub.neurons.get(pre, {}).get("neuron_class", ""),
             self.sub.neurons.get(post, {}).get("neuron_class", "")), 1.0)

    def _post_area(self, post: str, site: str) -> float:
        """post 位点面积（m²）：soma=点面积；node3=郎飞结面积。"""
        if self.fidelity == "point":
            return SOMA_AREA_CM2 * 1e-4
        if site == "node3":
            return float(NODE_AREA_CM2) * 1e-4
        return SOMA_AREA_CM2 * 1e-4

    def _gmax_density(self, g_ns: float, post: str, post_site: str) -> float:
        return g_ns * 1e-9 / self._post_area(post, post_site)

    def build(self):
        """grouped 批量组装（每会话前自动调用；dt/形状/命名定稿后不变）。

        - point：1 个 NeuronGroup（N）；two_comp：1 个 NeuronGroup（2N，
          i=2k soma / 2k+1 node3，linked_var 对交换 v_peer）；
        - 化学突触按递质类型各一个 Synapses（gmax/delay 向量化；行级 delay =
          异质延迟数据基础，杠杆③）；
        - 缝隙一个 Synapses（I_gap_in/I_gap_out 双 summed 目标，M5 L18）；
        - 肌肉每通道一个驱动 Synapses（Muscle3 冻结组件）；
        - 稀疏 stim 编码：n_cols = 刺激角色并集 + 1 零列；stim_col 逐神经元整型
          变量（实测支持，_probe_m8_sparse_stim.py）；
        - 可塑性：stp → 全化学突触 u/x；stdp/stdp_homeo → KC→MBON 子集替换
          （M6 L15：gmax 掩码赋值静默 no-op → 整体重建数组）。
        """
        from neural_exploration.src.brian_env import configure_brian2
        from brian2 import (NeuronGroup, SpikeMonitor, Synapses, cm, defaultclock,
                            linked_var, meter, ms, mS, mV, siemens, start_scope,
                            uF)

        # 每次 make_session 均重建（start_scope + 复用编译缓存——M5 惯例：
        # 旧 Network 对象弃用；代码串不变 → cython 缓存命中，秒级重 build）
        self._built = False
        configure_brian2()
        start_scope()
        t0 = time.perf_counter()
        defaultclock.dt = self.dt_ms * ms
        n = len(self.names)
        if n == 0:
            raise ValueError("空子集：连接组未包含任何神经元")
        if self.fidelity not in ("point", "two_comp"):
            raise ValueError(f"grouped 组装仅支持 point/two_comp：{self.fidelity}")

        two = self.fidelity == "two_comp"
        g_n = 2 * n if two else n
        eqs = self._group_eqs(two)
        ns = {"Cm": 1.0 * uF / cm ** 2, "G_AX": AXIAL_G_S * siemens}
        self.group = NeuronGroup(g_n, eqs, method=self.method,
                                 threshold="v > -20*mV", refractory=2.0 * ms,
                                 name=f"{self.name_prefix}_all", namespace=ns)
        from neural_exploration.src.ion_channels import steady_state_gates
        m0, h0, n0 = steady_state_gates(-65.0)
        self.group.v = -65.0 * mV
        self.group.m, self.group.h, self.group.n = m0, h0, n0
        self.group.gNa = 120.0 * mS / cm ** 2
        self.group.gK = 36.0 * mS / cm ** 2
        self.group.gL = 0.3 * mS / cm ** 2
        if two:
            area = np.empty(g_n)
            gna = np.empty(g_n)
            for k in range(n):
                area[2 * k] = SOMA_AREA_CM2 * 1e-4
                area[2 * k + 1] = float(NODE_AREA_CM2) * 1e-4
                gna[2 * k] = 120.0
                gna[2 * k + 1] = 300.0
            self.group.AREA = area * meter ** 2
            self.group.gNa = gna * mS / cm ** 2
            peer = np.empty(g_n, dtype=np.int64)
            peer[0::2] = np.arange(1, g_n, 2)
            peer[1::2] = np.arange(0, g_n, 2)
            self.group.v_peer = linked_var(self.group, "v", index=peer)
        else:
            self.group.AREA = np.full(n, SOMA_AREA_CM2 * 1e-4) * meter ** 2
        self.role_index = {r: k for k, r in enumerate(self.names)}

        # ---- 稀疏 stim 编码：刺激角色并集 + 零列（形状定稿，编译缓存纪律）----
        self._stim_cols, self._zero_col = self._assign_stim_cols()
        self._stim_n_cols = len(self._stim_cols) + 1  # + 零列
        stim_col_arr = np.full(g_n, self._zero_col, dtype=np.int32)
        for role, col in self._stim_cols.items():
            idx = self.role_index[role]
            if two:
                stim_col_arr[2 * idx] = col
                stim_col_arr[2 * idx + 1] = col
            else:
                stim_col_arr[idx] = col
        self.group.stim_col = stim_col_arr

        # ---- 化学突触：每递质类型一个 Synapses（杠杆③异质延迟）。
        # stp 档：直接建 u/x 突触（不重复建静态突触——避免双重连接，L 实测坑）----
        self.chem_synapses = []
        if self.plasticity == "stp":
            self._enable_stp()
        else:
            for stype in ("ampa", "gaba"):
                rows = [r for r in self.sub.chem if r.syn_type == stype]
                if not rows:
                    continue
                self._build_chem_synapse(stype, rows)

        # ---- 缝隙（一个 Synapses；杠杆②无关；gap_scale 先验 0.05）----
        if self.sub.gaps:
            a_i = np.array([self._idx_of(r.a, "soma") for r in self.sub.gaps])
            b_i = np.array([self._idx_of(r.b, "soma") for r in self.sub.gaps])
            gg = np.array([r.g_ns * self.gap_scale
                           for r in self.sub.gaps]) * 1e-9 * siemens
            syn = Synapses(self.group, self.group,
                           model="g_gap : siemens\n"
                                 "I_couple = g_gap*(v_pre - v_post) : amp\n"
                                 "I_gap_in_post = I_couple : amp (summed)\n"
                                 "I_gap_out_pre = -I_couple : amp (summed)",
                           name=f"{self.name_prefix}_gaps")
            syn.connect(i=a_i, j=b_i)
            syn.g_gap = gg
            self.gap_synapse = syn

        # ---- 肌肉：每通道一个驱动 Synapses（Muscle3 冻结组件）----
        from neural_exploration.src.chemotaxis_circuit import Muscle3
        channels = list(dict.fromkeys(m.channel for m in self.sub.muscles))
        if not channels:
            channels = ["fwd", "back", "left", "right"]
        self.muscle3 = Muscle3(tau_ms=self.muscle_tau_ms, cap=self.muscle_cap,
                               channels=channels,
                               name=f"{self.name_prefix}_muscle3")
        self.muscle3.build()
        self.mus_drivers = {}
        for ch in channels:
            rows = [m for m in self.sub.muscles if m.channel == ch]
            if not rows:
                continue
            pre_i = np.array([self._idx_of(m.motor, "node3")
                              for m in rows], dtype=np.int32)
            wm = np.array([m.w for m in rows])
            g = self.muscle3.get(ch)
            var = f"c_{ch}"
            on_pre = f"{var}_post = clip({var}_post + wm, 0.0, CAP)"
            syn = Synapses(self.group, g, model="wm : 1", on_pre=on_pre,
                           name=f"{self.name_prefix}_musdrv_{ch}",
                           namespace={"CAP": self.muscle_cap})
            syn.connect(i=pre_i, j=0)
            syn.wm = wm
            syn.delay = 0.1 * ms
            self.mus_drivers[ch] = syn

        # ---- 可塑性装配（stp 已在化学突触段构建；stdp/stdp_homeo 子集替换）----
        if self.plasticity in ("stdp", "stdp_homeo"):
            self._enable_stdp()

        # ---- 张力（tonic → nA，会话填充；幼虫默认空）----
        self._tonic_nA = {}
        for role, density in self.sub.tonic_uA_cm2.items():
            idx = self.role_index[role]
            self._tonic_nA[role] = density * 1e-6 * SOMA_AREA_CM2 * 1e9

        self._built = True
        self._build_wall_s = time.perf_counter() - t0
        return self

    def _idx_of(self, role: str, site: str = "soma") -> int:
        k = self.role_index[role]
        if self.fidelity == "two_comp":
            return 2 * k + (1 if site == "node3" else 0)
        return k

    def _assign_stim_cols(self) -> Tuple[Dict[str, int], int]:
        """刺激角色并集 → 列索引（稀疏编码核心；形状定稿后不变）。

        并集 = 嗅觉对 + 伤害感受器 + 运动池（自发脉冲，杠杆②）+ 张力角色 +
        KC 感觉输入（学习探针 CS 通路，可选）+ DA 奖赏（学习探针 US，可选）。
        运动池列**恒物化**（无论 lever_motor_drive 开关）——stim 形状在三杠杆
        消融间不变（编译缓存纪律，M4 L16）；杠杆②关闭时仅不填充脉冲。
        非刺激角色 → 零列（恒定 0，不物化）。
        """
        roles: List[str] = []
        sens = self.sens_roles
        roles += [s for s in sens if s]
        roles += list(self.nociceptor_roles)
        roles += list(self.motor_roles)          # 恒物化（形状定稿纪律）
        roles += list(self.sub.tonic_uA_cm2.keys())
        out: Dict[str, int] = {}
        for r in roles:
            if r in self.role_index and r not in out:
                out[r] = len(out)
        return out, len(out)  # 零列 = len(out)（最后一个）

    def _group_eqs(self, two: bool) -> str:
        """grouped 神经元方程（point 或 two_comp；稀疏 stim 用 stim(t, stim_col)）。

        two_comp：I_ax = G_AX·(v_peer − v)（amp，按隔室面积折密度）。
        """
        ax = " + I_ax" if two else ""
        ax_eqs = ("I_ax = G_AX*(v_peer - v) : amp\n"
                  "v_peer : volt (linked)\n") if two else ""
        return f"""
Im = gL*(EL-v) + gNa*m**3*h*(ENa-v) + gK*n**4*(EK-v) + g_ampa*(0.0*mV-v) + g_gaba*(-70.0*mV-v) : amp/meter**2
dv/dt = (Im + (stim(t, stim_col) + I_gap + I_gap_in + I_gap_out{ax})/AREA) / Cm : volt
dm/dt = alpham*(1-m)-betam*m : 1
dh/dt = alphah*(1-h)-betah*h : 1
dn/dt = alphan*(1-n)-betan*n : 1
alpham = (0.1/mV)*(v+40*mV)/(1-exp(-(v+40*mV)/(10*mV)))/ms : Hz
betam = 4*exp(-(v+65*mV)/(18*mV))/ms : Hz
alphah = 0.07*exp(-(v+65*mV)/(20*mV))/ms : Hz
betah = 1/(1+exp(-(v+35*mV)/(10*mV)))/ms : Hz
alphan = (0.01/mV)*(v+55*mV)/(1-exp(-(v+55*mV)/(10*mV)))/ms : Hz
betan = 0.125*exp(-(v+65*mV)/(80*mV))/ms : Hz
dg_ampa/dt = -g_ampa/(3.0*ms) : siemens/meter**2
dg_gaba/dt = -g_gaba/(5.0*ms) : siemens/meter**2
gNa : siemens/meter**2
gK : siemens/meter**2
gL : siemens/meter**2
AREA : meter**2
stim_col : integer
EL = -54.4*mV : volt (shared)
ENa = 50.0*mV : volt (shared)
EK = -77.0*mV : volt (shared)
I_gap : amp
I_gap_in : amp
I_gap_out : amp
{ax_eqs}"""

    def _build_chem_synapse(self, stype: str, rows: List[ChemRow]):
        """每递质类型一个 Synapses（向量化连接；gmax/delay 逐连接）。

        杠杆③（异质权重/传导）：enabled → 行级 delay（连接组事实/占位列）+
        类级缩放差异；disabled → 统一 delay=0.5ms + 无类级缩放（恒等权重）。
        杠杆①（命令层去同步）：cmd_gaba_scale 放大命令层 GABA 边有效权重
        （真实抑制边直接使用，不造边）。
        """
        from brian2 import Synapses, meter, ms, siemens

        pre_i = np.array([self._idx_of(r.pre, r.pre_site) for r in rows],
                         dtype=np.int32)
        post_i = np.array([self._idx_of(r.post, r.post_site) for r in rows],
                          dtype=np.int32)
        gmax = np.empty(len(rows))
        for k, r in enumerate(rows):
            s = self._class_scale_for(r.pre, r.post)
            s *= self.syn_type_scales.get(r.syn_type, 1.0)
            if stype == "gaba" and self.lever_cmd_desync \
                    and r.pre in self._cmd_layer_roles():
                s *= self.cmd_gaba_scale
            gmax[k] = self._gmax_density(r.g_ns * s * self.gmax_scale,
                                    r.post, r.post_site)
        if self.lever_hetero:
            delays = np.array([r.delay_ms for r in rows])
        else:
            delays = np.full(len(rows), 0.5)
        syn = Synapses(self.group, self.group,
                       model="gmax : siemens/meter**2",
                       on_pre=f"g_{stype}_post = g_{stype}_post + gmax",
                       name=f"{self.name_prefix}_chem_{stype}")
        syn.connect(i=pre_i, j=post_i)
        syn.gmax = gmax * siemens / meter ** 2
        syn.delay = delays * ms
        self.chem_synapses.append(syn)

    def _enable_stp(self):
        """STP（Tsodyks–Markram）全化学突触：u 先易化 → 释放 ∝ u·x → x 耗竭。

        参数 = m6_learning_params.csv habituation 段定稿（u0=0.6 / τ_fac=10 /
        τ_rec=1000，M6 定稿）。实现：重建 on_pre 含 u/x（冻结组件零修改，
        grouped 侧自建——与 M2 ChemicalSynapse 语义一致）。
        """
        from brian2 import Synapses, meter, ms, siemens

        u0 = self._stp_params["u0"]
        tf = self._stp_params["tau_fac_ms"]
        tr = self._stp_params["tau_rec_ms"]
        ns = {"U0": u0, "TAUFAC": tf * ms, "TAUREC": tr * ms}
        model = ("du/dt = (U0-u)/TAUFAC : 1 (clock-driven)\n"
                 "dx/dt = (1-x)/TAUREC : 1 (clock-driven)\n"
                 "gmax : siemens/meter**2")
        for stype in ("ampa", "gaba"):
            rows = [r for r in self.sub.chem if r.syn_type == stype]
            if not rows:
                continue
            pre_i = np.array([self._idx_of(r.pre, r.pre_site) for r in rows],
                             dtype=np.int32)
            post_i = np.array([self._idx_of(r.post, r.post_site) for r in rows],
                              dtype=np.int32)
            gmax = np.empty(len(rows))
            for k, r in enumerate(rows):
                s = self._class_scale_for(r.pre, r.post)
                s *= self.syn_type_scales.get(r.syn_type, 1.0)
                if stype == "gaba" and self.lever_cmd_desync \
                        and r.pre in self._cmd_layer_roles():
                    s *= self.cmd_gaba_scale
                gmax[k] = self._gmax_density(r.g_ns * s * self.gmax_scale,
                                    r.post, r.post_site)
            delays = (np.array([r.delay_ms for r in rows]) if self.lever_hetero
                      else np.full(len(rows), 0.5))
            on_pre = (f"u = u + U0*(1-u)\n"
                      f"g_{stype}_post = g_{stype}_post + gmax*u*x\n"
                      f"x = x - u*x")
            syn = Synapses(self.group, self.group, model=model, on_pre=on_pre,
                           namespace=ns,
                           name=f"{self.name_prefix}_chem_{stype}_stp")
            syn.connect(i=pre_i, j=post_i)
            syn.gmax = gmax * siemens / meter ** 2
            syn.delay = delays * ms
            syn.u = u0
            syn.x = 1.0
            self.chem_synapses.append(syn)

    def _enable_stdp(self):
        """成对 STDP（限 KC→MBON 子集）+ 可选稳态（stdp_homeo）。

        - 装配语义 = learning.py `_build`（M6 L15 教训：gmax 掩码赋值静默
          no-op → 整体重建数组）：原边 gmax 置 0 + 新建 w 缩放突触；
        - 成对规则（stdp 段定稿参数 a_plus/a_minus/tau_plus/tau_minus/w_max；
          扫描探针振幅 × stdp_eta 以在短协议内读出 LI 出现/消失阈值——
          机制级判据，P5 全协议定稿学习率由学习节点定稿）；
        - +稳态：dw/dt = η_h·(w0 − w)（防 STDP 饱和，M6 #1 教训）。
        """
        from brian2 import Synapses, TimedArray, meter, ms, second, siemens

        try:
            from neural_exploration.src.plasticity import load_stdp_params
            sp = load_stdp_params()
            a_plus = float(sp.a_plus) * self._stdp_eta
            a_minus = float(sp.a_minus) * self._stdp_eta
            tau_plus = float(sp.tau_plus_ms)
            tau_minus = float(sp.tau_minus_ms)
            w_max = float(sp.w_max)
        except Exception:  # noqa: BLE001 —— 冻结参数文件缺失时回退预注册默认
            a_plus, a_minus = 0.01 * self._stdp_eta, 0.009 * self._stdp_eta
            tau_plus, tau_minus, w_max = 20.0, 20.0, 2.0
        pairs = self._stdp_edge_pairs()
        self._stdp_pairs = pairs

        # 目标边（KC→MBON 的 ampa 边）
        syn_ampa = next((s for s in self.chem_synapses
                         if getattr(s, "name", "").endswith("_chem_ampa")),
                        None)
        if syn_ampa is None or not pairs:
            # 无目标边 → STDP 空转（可塑性档在无 KC→MBON 子集的子集上如实记录）
            self._stdp_syn = None
            return
        i_arr = np.asarray(syn_ampa.i)
        j_arr = np.asarray(syn_ampa.j)
        mask = np.zeros(i_arr.size, dtype=bool)
        pre_i, post_i = [], []
        for pre, post in pairs:
            m = (i_arr == self._idx_of(pre, "node3")) & \
                (j_arr == self._idx_of(post, "soma"))
            if m.any():
                mask |= m
                pre_i.append(self._idx_of(pre, "node3"))
                post_i.append(self._idx_of(post, "soma"))
        if not mask.any():
            self._stdp_syn = None
            return
        _g = np.array(np.asarray(syn_ampa.gmax, dtype=float), dtype=float)
        _g[mask] = 0.0
        syn_ampa.gmax = _g * siemens / meter ** 2

        g_density = self._gmax_density(5.0 * self.gmax_scale,
                                      pairs[0][1], "soma")
        ns = {"A_PLUS": a_plus, "A_MINUS": a_minus, "TAU_PLUS": tau_plus * ms,
              "TAU_MINUS": tau_minus * ms, "WMAX": w_max, "WMIN": 0.0,
              "GMAXD": g_density * siemens / meter ** 2,
              "W0": 1.0, "ETAH": self._homeo_eta}
        model = ("dApre/dt = -Apre/TAU_PLUS : 1 (clock-driven)\n"
                 "dApost/dt = -Apost/TAU_MINUS : 1 (clock-driven)\n")
        if self.plasticity == "stdp_homeo":
            model += "dw/dt = ETAH*(W0 - w)/second : 1 (clock-driven)\n"
        else:
            model += "w : 1\n"
        on_pre = ("w = clip(w + A_MINUS*Apost, WMIN, WMAX)\n"
                  "Apre = Apre + A_PLUS\n"
                  "g_ampa_post = g_ampa_post + GMAXD*w")
        on_post = ("Apost = Apost + A_MINUS\n"
                   "w = clip(w + A_PLUS*Apre, WMIN, WMAX)")
        syn = Synapses(self.group, self.group, model=model, on_pre=on_pre,
                       on_post=on_post, namespace=ns,
                       name=f"{self.name_prefix}_stdp_kcmbon")
        syn.connect(i=np.array(pre_i, dtype=np.int32),
                    j=np.array(post_i, dtype=np.int32))
        syn.delay = 0.5 * ms
        syn.w = 1.0
        syn.Apre = 0.0
        syn.Apost = 0.0
        self._stdp_syn = syn

    # ------------------------------------------------------------------ #
    # 会话
    # ------------------------------------------------------------------ #
    def _stim_n_steps(self) -> int:
        return int(round(max(STIM_WINDOW_MS, PROTOCOL_WINDOW_MS) / self.dt_ms))

    def make_session(self, t_total_ms: Optional[float] = None,
                     record: Optional[Sequence[str]] = None,
                     stimulated_roles: Optional[Sequence[str]] = None
                     ) -> "LarvaSession":
        from brian2 import Network, SpikeMonitor, TimedArray, amp, ms

        p = self
        self.t_total_ms_sess = float(t_total_ms or self.t_total_ms)
        self.build()
        n_steps = self._stim_n_steps()
        n_cols = self._stim_n_cols
        stim = TimedArray(np.zeros((n_steps, n_cols)) * amp,
                          dt=self.dt_ms * ms, name="stim")
        net = Network()
        net.add(self.group)
        for syn in self.chem_synapses:
            net.add(syn)
        if self.gap_synapse is not None:
            net.add(self.gap_synapse)
        if self._stdp_syn is not None:
            net.add(self._stdp_syn)
        for syn in self.mus_drivers.values():
            net.add(syn)
        for g in self.muscle3.groups:
            net.add(g)
        self._sp = SpikeMonitor(self.group, "v",
                                name=f"{self.name_prefix}_sp")
        net.add(self._sp)
        mons_mus = self.muscle3.monitor(p.dt_b,
                                        name=f"{self.name_prefix}_musc")
        for mm in mons_mus:
            net.add(mm)
        ns = {"stim": stim}
        self._sess = LarvaSession(self, net, ns, stim, mons_mus)
        self._sess._init(self.seed)
        return self._sess

    # ------------------------------------------------------------------ #
    # 协议：静息（无刺激发放率分布，G1 门输入）
    # ------------------------------------------------------------------ #
    def run_resting(self, t_total_ms: float = 2000.0,
                    settle_ms: float = 500.0, seed: Optional[int] = None
                    ) -> dict:
        """静息低活动协议（G1 门输入）。

        语义（预注册）：M8『静息低活动』= **自然自发驱动在场**的低活动背景
        （运动层自发脉冲 = 幼虫的自发活动源，M5 302 的 AVB 张力背景同构——
        M8 无单一张力，杠杆②运动层自发脉冲为背景驱动）。测量 settle 后的
        静默比例：**无夹带**的健康网络在驱动下仍保持大部分神经元安静
        （静默 ∈ [50,90]%，G1 带）；全互兴奋夹带网络驱动扩散 → 静默 <50%。
        （注：完全无驱动 = 100% 静默 ≠『低活动』，落不出带——故驱动在场，
        与 M5 P2 的张力驱动语义一致，M5 L41#1 settle 窗沿用。）

        Returns dict(median_hz/max_hz/silent_frac/n_spiking/has_nan/wall_s)。
        """
        from brian2 import ms as bms

        seed = self.seed if seed is None else int(seed)
        sess = self.make_session(t_total_ms=t_total_ms)
        sess.reset(seed=seed, motor_drive=True)    # 静息 = 自然自发驱动在场
        t0w = time.perf_counter()
        sess.run_resting_window(settle_ms)         # 初始化瞬态波排除（M5 L41#1）
        sess.reset(seed=seed, motor_drive=True)    # 清监视器（store/restore）再测
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
            t_total_ms=t_total_ms, settle_ms=settle_ms,
        )

    # ------------------------------------------------------------------ #
    # 协议：自发行为状态比例（G1 双状态门输入；三杠杆②驱动运动输出层）
    # ------------------------------------------------------------------ #
    def run_spontaneous(self, t_total_ms: float = 5000.0,
                        seed: Optional[int] = None,
                        s_override: float = 0.0) -> dict:
        """无刺激无梯度 T 窗：运动输出层自发脉冲（杠杆②）→ 肌肉序列 →
        classify_state 状态比例（virtual_body 冻结语义；阈值 CSV 定稿语义）。

        Returns dict(frac/states/v/omega/n_epochs/wall_s)。
        """
        from brian2 import ms as bms

        seed = self.seed if seed is None else int(seed)
        body = VirtualBody(v_fwd0=self.v_fwd0, v_rev0=self.v_rev0,
                           omega_max=self.omega_max, dt_b=self.dt_b,
                           arena_L=self.arena_L, boundary="reflect")
        sess = self.make_session(t_total_ms=t_total_ms)
        sess.reset(seed=seed)
        t0w = time.perf_counter()
        n_epochs = max(1, int(round(t_total_ms / self.dt_b)))
        mus_hist, states, vs, omegas = [], [], [], []
        for e in range(n_epochs):
            mus = sess.run_epoch(self.dt_b, s_override)
            c_fwd = float(mus.get("fwd", 0.0))
            c_back = float(mus.get("back", 0.0))
            c_left = float(mus.get("left", 0.0))
            c_right = float(mus.get("right", 0.0))
            v = body.speed(c_fwd, c_back)
            omega = body.turn_rate(c_left, c_right, e * self.dt_b)
            st = classify_state(v, omega, c_fwd, c_back,
                                v_fwd0=self.v_fwd0, omega_max=self.omega_max)
            body.step(c_fwd, c_back, c_left, c_right, self.dt_b, e * self.dt_b)
            mus_hist.append(mus)
            states.append(st)
            vs.append(v)
            omegas.append(omega)
        wall = time.perf_counter() - t0w
        frac = state_fractions(states)
        return dict(frac=frac, states=states, v=np.asarray(vs),
                    omega=np.asarray(omegas), n_epochs=n_epochs,
                    wall_s=wall, n_epochs_total=n_epochs)

    # ------------------------------------------------------------------ #
    # 协议：趋化短协议（CI 符号；M4/M5 趋化闭环语义的幼虫版）
    # ------------------------------------------------------------------ #
    def run_chemotaxis_trials(self, n_trials: int = 1, t_total_ms: float = 5000.0,
                              seed_base: int = 0, start_jitter: float = 0.3,
                              gradient: bool = True) -> Tuple[List[dict], dict]:
        """气味梯度闭环（AWC on/off 时间差分编码，M4 转导语义）→ CI。

        Returns (results, meta)；result dict 含 ci/turn_events/wall_s。
        """
        from brian2 import ms as bms

        env = ChemotaxisEnv(arena_L=self.arena_L, sigma=1.25,
                            c_max=1.0, c_bg=0.0,
                            food_x=self.arena_L * 0.75,
                            food_y=self.arena_L * 0.75, boundary="reflect")
        if not gradient:
            env = env.no_gradient()
        body = VirtualBody(v_fwd0=self.v_fwd0, v_rev0=self.v_rev0,
                           omega_max=self.omega_max, dt_b=self.dt_b,
                           arena_L=self.arena_L, boundary="reflect")
        sess = self.make_session(t_total_ms=t_total_ms)
        rng = np.random.default_rng(seed_base)
        out = []
        for trial in range(int(n_trials)):
            seed = seed_base + trial
            if start_jitter > 0:
                sx = self.arena_L / 2.0 + rng.normal(0.0, start_jitter)
                sy = self.arena_L / 2.0 + rng.normal(0.0, start_jitter)
                th0 = rng.uniform(0.0, 2.0 * math.pi)
            else:
                sx, sy, th0 = self.arena_L / 2.0, self.arena_L / 2.0, 0.0
            sess.reset(seed=seed)
            body.reset(sx, sy, th0)
            tracker = TimeDiffTracker(self.transduction["tau_win_ms"],
                                      env.sample(sx, sy))
            t0w = time.perf_counter()
            n_epochs = max(1, int(round(t_total_ms / self.dt_b)))
            xs, ys, n_turn = [], [], 0
            turn_rng = np.random.default_rng(seed)
            for e in range(n_epochs):
                c_now = env.sample(body.x, body.y)
                s = tracker.s_at(e * self.dt_b, c_now)
                mus = sess.run_epoch(self.dt_b, s)
                # 机制 A（M4 语义）：s<−θ_pir 且转向头运动发放 → 转向事件
                # ⚠ B1c 实测坑 L24（2026-08-28）：旧阈值 -0.5 不可达
                # （|s|≤ΔC/τ_win≈0.01，C∈[0,1]）→ 转向事件永不触发 → CI≡0。
                # 按 M4 定稿 mechanism_a.theta_pir=1e-6（ΔC/ms，
                # data/m4_chemotaxis_params.csv）修正（未改签名/默认行为）。
                if s < -PIR_THETA_S and not body.is_turning():
                    turn_roles = self._turn_driver_roles()
                    if turn_roles and sess.any_spikes_in_window(
                            turn_roles, e * self.dt_b,
                            e * self.dt_b + self.dt_b):
                        direction = 1.0 if turn_rng.random() < 0.5 else -1.0
                        body.trigger_turn(direction, 1.0, 1571.0)
                        n_turn += 1
                body.step(float(mus.get("fwd", 0.0)),
                          float(mus.get("back", 0.0)),
                          float(mus.get("left", 0.0)),
                          float(mus.get("right", 0.0)), self.dt_b,
                          e * self.dt_b)
                xs.append(body.x)
                ys.append(body.y)
            xa, ya = np.asarray(xs), np.asarray(ys)
            env.assert_bounded(xa, ya)
            body.assert_trajectory(xa, ya)
            ci = env.ci_per_trial(xa, ya)
            out.append(dict(ci=float(ci), n_turn_events=n_turn,
                            wall_s=time.perf_counter() - t0w,
                            trial=trial, seed=seed))
        meta = dict(wall_s=[r["wall_s"] for r in out],
                    mean_wall_s=float(np.mean([r["wall_s"] for r in out])),
                    build_wall_s=self._build_wall_s,
                    n_neurons=self.sub.n_neurons, n_chem=self.sub.n_chem,
                    n_gap=self.sub.n_gap, scale=self.scale,
                    fidelity=self.fidelity, dt_ms=self.dt_ms,
                    method=self.method, plasticity=self.plasticity,
                    gradient=gradient)
        return out, meta

    # ------------------------------------------------------------------ #
    # 协议：痛觉逃避短协议（MD 伤害感受器 → 方向）
    # ------------------------------------------------------------------ #
    def run_escape(self, t_total_ms: float = 1000.0,
                   seed: Optional[int] = None) -> dict:
        """伤害性刺激（MD 前缀角色电流注入）→ C_back/C_curl 方向。

        D_peak = max(C_back − C_fwd)；curl_peak = max(C_curl)。
        Returns dict(d_peak/direction/curl_peak/c_back/c_fwd/c_curl/wall_s)。
        """
        from brian2 import ms as bms

        seed = self.seed if seed is None else int(seed)
        roles = self.nociceptor_roles
        if not roles:
            return dict(d_peak=float("nan"), direction="no_nociceptor",
                        curl_peak=float("nan"), c_back=[], c_fwd=[],
                        c_curl=[], wall_s=0.0, note="无 MD 伤害感受器角色")
        sess = self.make_session(t_total_ms=t_total_ms)
        sess.reset(seed=seed)
        i_nA = self.escape["density"] * 1e-6 * SOMA_AREA_CM2 * 1e9
        i0 = int(round(self.escape["start_ms"] / self.dt_ms))
        i1 = int(round((self.escape["start_ms"] + self.escape["dur_ms"])
                       / self.dt_ms))
        n_steps = sess.stim.values.shape[0]
        i0, i1 = max(0, min(i0, n_steps)), max(i0, min(i1, n_steps))
        for r in roles:
            col = self._stim_cols.get(r)
            if col is not None:
                sess.stim.values[i0:i1, col] = i_nA * 1e-9
        t0w = time.perf_counter()
        n_epochs = max(1, int(round(t_total_ms / self.dt_b)))
        cbs, cfs, ccs, t_ms = [], [], [], []
        for e in range(n_epochs):
            mus = sess.run_epoch(self.dt_b, 0.0)
            cbs.append(float(mus.get("back", 0.0)))
            cfs.append(float(mus.get("fwd", 0.0)))
            ccs.append(float(mus.get("curl", 0.0)))
            t_ms.append(e * self.dt_b)
        wall = time.perf_counter() - t0w
        c_back, c_fwd, c_curl = (np.asarray(cbs), np.asarray(cfs),
                                 np.asarray(ccs))
        d_peak = float(np.max(c_back - c_fwd)) if c_back.size else 0.0
        curl_peak = float(np.max(c_curl)) if c_curl.size else 0.0
        return dict(d_peak=d_peak,
                    direction=("back" if d_peak > 0.3 else "not_back"),
                    curl_peak=curl_peak, c_back=cbs, c_fwd=cfs, c_curl=ccs,
                    t_ms=t_ms, wall_s=wall, nociceptor_roles=roles,
                    t_total_ms=t_total_ms)

    # ------------------------------------------------------------------ #
    # 协议：学习探针（可塑性轴 LI；CS=气味/AWC，US=DA 奖赏，KC→MBON）
    # ------------------------------------------------------------------ #
    def run_learning_probe(self, t_test_ms: float = 2000.0,
                           t_train_ms: float = 2000.0,
                           seed: Optional[int] = None) -> dict:
        """机制级联想学习探针：KC→MBON 权重的 CS-US 配对获得 → LI。

        - CS：AWC 感觉对恒定气味（s=+1）注入（蘑菇体通路驱动 KC→MBON）；
        - US：DA 奖赏角色（DAN 前缀）在训练窗恒定注入（调质上下文占位——
          扫描探针用**成对 STDP** 测 CS 驱动相关性的权重获得，机制级 LI；
          三因子门控（M(t) 门控）为 P5 全协议机制，M6 L24 语义）；
        - 相位纪律（M6 L13 教训）：US 写**绝对网络时间**索引；
        - 测量：基线 CS 窗 → w_pre；训练（CS+US）→ w_post；
          stdp/stdp_homeo 档：LI = (mean(w_post)−mean(w_pre))/(w_max−w0)
          clip [−1,1]（权重档，机制级）；none/stp 档：LI = MBON 发放率
          变化归一（应为 0——STP 无持久权重变化）。

        Returns dict(li/dw/li_mode/w_pre/w_post/mbon_rate_pre/mbon_rate_post/
        n_stdp_edges/wall_s)。
        """
        from brian2 import ms as bms

        seed = self.seed if seed is None else int(seed)
        t_tot = float(max(t_test_ms, 1.0)) + float(max(t_train_ms, 1.0))
        sess = self.make_session(t_total_ms=t_tot)
        sess.reset(seed=seed)          # w = 存储快照（=1.0 基线）
        mbon = self.mbon_roles
        dan = self._roles_by_prefix(ROLE_PREFIXES["dan"])
        t0w = time.perf_counter()

        # US（DA）训练窗绝对索引预填充（M6 L13：网络时钟 ≠ 0 时按绝对时间写）
        n_steps = sess.stim.values.shape[0]
        i0t = int(round(float(t_test_ms) / self.dt_ms))
        i1t = int(round((float(t_test_ms) + float(t_train_ms)) / self.dt_ms))
        i1t = min(i1t, n_steps)
        for r in dan:
            col = self._stim_cols.get(r)
            if col is not None:
                sess.stim.values[i0t:i1t, col] = 0.10 * 1e-9  # 0.1nA 奖赏

        def _run_cs_window(t_ms: float) -> Tuple[int, float]:
            """CS 窗（s=+1）运行；返回 (mbon 发放数, 窗结束网络时刻 ms)。"""
            n_ep = max(1, int(round(t_ms / self.dt_b)))
            t_start = float(sess.net.t / bms)
            for _ in range(n_ep):
                sess.run_epoch(self.dt_b, 1.0)
            t_end = float(sess.net.t / bms)
            n_mbon = 0
            times = sess.role_spike_times()
            for r in mbon:
                t_arr = times.get(r, [])
                n_mbon += int(np.sum((t_arr >= t_start) & (t_arr < t_end)))
            return n_mbon, t_end

        def _run_settle(t_ms: float) -> None:
            """无刺激 settle 窗（s=0）：训练后突触电导回落，剔除瞬态伪迹——
            none/stp 档 LI 应 ≈0（无持久权重变化），否则瞬态污染测量（M 实测坑）。"""
            n_ep = max(1, int(round(t_ms / self.dt_b)))
            for _ in range(n_ep):
                sess.run_epoch(self.dt_b, 0.0)

        # 基线 CS 窗 → w_pre（stdp 档：基线暴露本身产生少量相关发放，如实测量）
        n_pre, t_end_pre = _run_cs_window(float(t_test_ms))
        mbon_pre = n_pre / (max(t_end_pre, 1e-9) / 1000.0)
        w_pre = (np.array(self._stdp_syn.w, dtype=float)
                 if self._stdp_syn is not None else None)

        # 训练（CS+US 配对；DA 列已预填充）
        n_tr, t_end_tr = _run_cs_window(float(t_train_ms))
        w_post = (np.array(self._stdp_syn.w, dtype=float)
                  if self._stdp_syn is not None else None)

        # settle（瞬态回落）→ 训练后 CS 测试窗 → mbon_post
        _run_settle(500.0)
        n_post, t_end_post = _run_cs_window(float(t_test_ms))
        mbon_post = n_post / (max(t_end_post - t_end_tr - 500.0, 1e-9) / 1000.0)
        wall = time.perf_counter() - t0w

        if w_post is not None and w_pre is not None:
            dw = float(np.mean(w_post) - np.mean(w_pre))
            li = float(np.clip(dw / (2.0 - 1.0), -1.0, 1.0))
            li_mode = "weight"
        else:
            # none/stp 档：无持久可塑性机制 → LI 按构造为 0（机制级判据，
            # M6 L16 语义）；MBON 发放率前后差为瞬态伪迹（settle 后仍可能
            # 有慢集体模态残余），如实入 mbon_rate_pre/post 供参考。
            dw = float("nan")
            li = 0.0
            li_mode = "no_plasticity"
        return dict(li=li, dw=dw, li_mode=li_mode,
                    w_pre=(None if w_pre is None else w_pre.tolist()),
                    w_post=(None if w_post is None else w_post.tolist()),
                    mbon_rate_pre=float(mbon_pre),
                    mbon_rate_post=float(mbon_post),
                    wall_s=wall, t_test_ms=t_test_ms, t_train_ms=t_train_ms,
                    n_stdp_edges=(0 if self._stdp_syn is None
                                  else len(self._stdp_syn.w)))


# --------------------------------------------------------------------- #
# LarvaSession：试次会话（epoch 迭代；稀疏 stim 写列）
# --------------------------------------------------------------------- #
class LarvaSession:
    """LarvaCircuit 的试次会话（单组 stim (n_steps, n_cols) 稀疏编码）。"""

    def __init__(self, circuit: LarvaCircuit, net, ns, stim, mons_mus):
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
            col = self.circuit._stim_cols.get(role)
            if col is not None:
                self.stim.values[:, col] = nA * 1e-9

    def reset(self, seed: Optional[int] = None, motor_drive: bool = True):
        """store/restore + 重播种；stim 清零 + 张力 + 杠杆②运动池脉冲。

        motor_drive=False：不填充运动池自发脉冲（静息协议——G1 双状态的
        「静息低活动」态 = 无驱动；行为 bout 态 = 运动层驱动，M6 L9#1）。
        """
        from brian2 import seed as bseed

        c = self.circuit
        bseed(seed if seed is not None else c.seed)
        self.net.restore()
        self.stim.values[:] = 0.0
        self._fill_tonic()
        # 杠杆②：运动输出层自发脉冲表（固定 seed 伪随机，p=1/n=1 确定性；
        # 只落运动池列——命令池**不**注入，M6 L9#1 教训）
        if c.lever_motor_drive and motor_drive:
            self._fill_motor_pulses()
        self._rng = np.random.default_rng(seed if seed is not None else c.seed)
        self._n_epochs = 0

    def _fill_motor_pulses(self):
        """确定性伪随机脉冲（2Hz×0.10nA×3ms）→ 运动输出层 stim 列。"""
        c = self.circuit
        n_steps = self.stim.values.shape[0]
        rng = np.random.default_rng(c.spont_seed)
        for role, col in c._stim_cols.items():
            if role not in c.motor_roles:
                continue
            period = 1000.0 / max(c.motor_drive_hz, 1e-9)
            n_pulses = int(n_steps * c.dt_ms / period)
            for _ in range(n_pulses):
                t_on = rng.uniform(0.0, n_steps * c.dt_ms)
                i0 = int(t_on / c.dt_ms)
                i1 = min(n_steps, i0 + int(c.motor_drive_ms / c.dt_ms))
                self.stim.values[i0:i1, col] = c.motor_drive_nA * 1e-9

    def run_epoch(self, dt_ms: float, s_value: float) -> Dict[str, float]:
        """运行一个 epoch：s → 感觉对 nA → 稀疏列切片 → run(dt)。"""
        from brian2 import ms

        c = self.circuit
        s = float(s_value)
        t_now_ms = float(self.net.t / ms)
        dt = float(dt_ms)
        i0 = int(round(t_now_ms / c.dt_ms))
        i1 = int(round((t_now_ms + dt) / c.dt_ms))
        n_steps = self.stim.values.shape[0]
        i0, i1 = max(0, min(i0, n_steps)), max(i0, min(i1, n_steps))
        on_role, off_role = c.sens_roles
        tr = c.transduction
        if on_role and (col := c._stim_cols.get(on_role)) is not None:
            i_on = tr["g_on"] * max(s, 0.0) * 1e-6 * SOMA_AREA_CM2 * 1e9
            self.stim.values[i0:i1, col] = i_on * 1e-9
        if off_role and (col := c._stim_cols.get(off_role)) is not None:
            i_off = tr["g_off"] * max(-s, 0.0) * 1e-6 * SOMA_AREA_CM2 * 1e9
            self.stim.values[i0:i1, col] = i_off * 1e-9
        self.net.run(dt * ms, namespace=self.ns)
        self._n_epochs += 1
        return self.circuit.muscle3.read()

    def run_resting_window(self, t_total_ms: float):
        from brian2 import ms

        self.net.run(t_total_ms * ms, namespace=self.ns)
        self._n_epochs += 1

    def role_spike_times(self) -> Dict[str, np.ndarray]:
        """逐角色发放时刻（ms；two_comp：soma/node3 合并为该角色发放）。"""
        from brian2 import ms as bms

        c = self.circuit
        t_arr = np.asarray(c._sp.t / bms)
        i_arr = np.asarray(c._sp.i)
        out: Dict[str, np.ndarray] = {}
        for role, idx in c.role_index.items():
            if c.fidelity == "two_comp":
                mask = (i_arr == 2 * idx) | (i_arr == 2 * idx + 1)
            else:
                mask = i_arr == idx
            out[role] = t_arr[mask]
        return out

    def any_spikes_in_window(self, roles, t0_ms: float, t1_ms: float) -> bool:
        from brian2 import ms

        c = self.circuit
        t = np.asarray(c._sp.t / ms)
        i = np.asarray(c._sp.i)
        for role in roles:
            idx = c.role_index.get(str(role).upper())
            if idx is None:
                continue
            if c.fidelity == "two_comp":
                mask = (i == 2 * idx) | (i == 2 * idx + 1)
            else:
                mask = i == idx
            if mask.any() and np.any((t[mask] >= t0_ms - 1e-9)
                                     & (t[mask] < t1_ms)):
                return True
        return False

    def muscle_read(self) -> Dict[str, float]:
        return self.circuit.muscle3.read()


def make_larva_circuit(scale: int = 3016, fidelity: str = "point", **kwargs):
    """LarvaCircuit 工厂（point/two_comp → grouped；hh → HH 子图路径）。"""
    if fidelity == "hh":
        return LarvaHHSubgraph(scale=scale, **kwargs)
    return LarvaCircuit(scale=scale, fidelity=fidelity, **kwargs)


# --------------------------------------------------------------------- #
# HH（多隔室）档：≤300 局部子图（component 模式，M5 哲学）
# --------------------------------------------------------------------- #
class LarvaHHSubgraph:
    """HH（M1 MultiCompartmentNeuron）局部子图（≤300 档短协议 T≤5s）。

    与 M5 ReflexCircuit 同构：按 spec 子集（默认功能必保集合前 ~40 角色 +
    其诱导闭包）用 M1 多隔室神经元 + M2 ChemicalSynapse/GapJunction 组件
    逐对象组装（component 模式——HH 档仅局部子图，预算纪律 M5 同款：
    扫描默认跳过该格点，`--run-hh` 强制运行时用最短协议）。
    """

    def __init__(self, scale: int = 300, csv_path: Optional[str] = None,
                 name_prefix: str = "larva_hh", seed: int = 0,
                 dt_ms: Optional[float] = None,
                 method: Optional[str] = None,
                 hh_subgraph_size: int = 40,
                 spec_override: Optional[ConnectomeSpec] = None,
                 **kwargs):
        self.scale = int(scale)
        if self.scale > 300:
            raise ValueError("HH 档仅限 ≤300 规模（M8 D1）")
        self.dt_ms = FIDELITY_DT["hh"][0] if dt_ms is None else float(dt_ms)
        self.method = FIDELITY_DT["hh"][1] if method is None else method
        self.name_prefix = name_prefix
        self.seed = int(seed)
        self.hh_subgraph_size = int(hh_subgraph_size)
        if spec_override is not None:
            self.spec = spec_override
        else:
            self.spec = load_connectome(csv_path, poll_s=0.0)
        self.is_placeholder = bool(self.spec.is_placeholder)
        self.names = scale_names(self.spec, min(scale, self.hh_subgraph_size))
        self.sub = self.spec.subset(self.names)
        self.neurons: Dict[str, object] = {}
        self.chemicals: List[object] = []
        self.muscle3 = None
        self._built = False
        self._build_wall_s = float("nan")

    def build(self):
        from neural_exploration.src.brian_env import configure_brian2
        from neural_exploration.src.neuron_model import MultiCompartmentNeuron
        from neural_exploration.src.synapse_model import (ChemicalSynapse,
                                                          SynapseParams,
                                                          chemical_im_terms,
                                                          chemical_post_eqs,
                                                          load_synapse_params)
        from brian2 import ms, start_scope

        if self._built:
            return self
        configure_brian2()
        start_scope()
        t0 = time.perf_counter()
        m2 = load_synapse_params()
        post_types: Dict[str, set] = {n: set() for n in self.names}
        for s in self.sub.chem:
            post_types[s.post].add(s.syn_type)
        self.neurons = {}
        for role in self.names:
            sub = {t: m2[t] for t in post_types[role]}
            eqs = chemical_post_eqs(sub)
            ims = chemical_im_terms(sub)
            self.neurons[role] = MultiCompartmentNeuron(
                name=f"{self.name_prefix}_{role.lower()}", dt_ms=self.dt_ms,
                method=self.method, t_total_ms=500.0, extra_eqs=eqs,
                extra_im_terms=ims, stim_var=f"stim_{role.lower()}").build()
        self.chemicals = []
        for k, s in enumerate(self.sub.chem):
            base = m2[s.syn_type]
            sp = SynapseParams(synapse_type=s.syn_type, g_max_ns=s.g_ns,
                               tau_ms=base.tau_ms, e_rev_mv=base.e_rev_mv,
                               p_release=1.0, n_vesicles=1,
                               mg_mm=base.mg_mm, u0=base.u0,
                               tau_fac_ms=base.tau_fac_ms,
                               tau_rec_ms=base.tau_rec_ms)
            cs = ChemicalSynapse(self.neurons[s.pre], self.neurons[s.post], sp,
                                 pre_site=s.pre_site, post_site=s.post_site,
                                 name=f"{self.name_prefix}_syn{k}")
            cs.build()
            cs.synapses.delay = s.delay_ms * ms
            self.chemicals.append(cs)
        from neural_exploration.src.chemotaxis_circuit import Muscle3
        channels = list(dict.fromkeys(m.channel for m in self.sub.muscles))
        if not channels:
            channels = ["fwd", "back"]
        self.muscle3 = Muscle3(tau_ms=20.0, cap=1.0, channels=channels,
                               name=f"{self.name_prefix}_muscle3")
        self.muscle3.build()
        for k, m in enumerate(self.sub.muscles):
            self.muscle3.connect_driver(self.neurons[m.motor], m.channel,
                                        weight=m.w,
                                        name=f"{self.name_prefix}_musdrv{k}")
        self._built = True
        self._build_wall_s = time.perf_counter() - t0
        return self

    def run_escape(self, t_total_ms: float = 500.0, seed: Optional[int] = None
                   ) -> dict:
        """MD 伤害感受器刺激 → 方向（HH 档唯一协议：最短短协议）。"""
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
        roles = [n for n in self.names if n.startswith("MD")]
        n_steps = int(round(max(500.0, t_total_ms) / self.dt_ms))
        stims = {}
        for role in self.names:
            n_comp = int(self.neurons[role].neuron.N)
            arr = np.zeros((n_steps, n_comp)) * amp
            if role in roles:
                i_nA = self.neurons[role].density_to_nA(60.0)
                i0 = int(round(100.0 / self.dt_ms))
                i1 = int(round(120.0 / self.dt_ms))
                arr[i0:i1, :] = i_nA * nA
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
        return dict(d_peak=d_peak,
                    direction=("back" if d_peak > 0.3 else "not_back"),
                    curl_peak=float(np.max(c_series.get("curl",
                                                        np.zeros(1)))),
                    wall_s=wall, t_total_ms=t_total_ms,
                    fidelity="hh", n_neurons=len(self.names),
                    build_wall_s=self._build_wall_s)


# --------------------------------------------------------------------- #
# 降阶正确性锚（§3.4：幼虫已验子回路行为一致性 Δ 判据带预注册）
# --------------------------------------------------------------------- #
# M4/M5 已验子回路 → 幼虫网络的行为一致性（G0 门组件；判据带预注册于
# data/m8_larva_params.csv 校准段——扫描工具据此判定 Δ 判据）：
#   - AWC 嗅觉趋化：CI 方向（正趋化）随规模一致（ΔCI ≤ 0.15 或方向一致，
#     M5 G0 §3.4 语义沿用——幼虫无独立参考解，锚 = M4/M5 机制模块语义）；
#   - MD 痛觉逃避：伤害性刺激 → back/curl 方向（D_peak > 0.3 或 C_curl 主导，
#     M3/M5 逃避方向语义）；
#   - 运动节律：自发 bout 结构（双状态 G1 门，§1 D6）。
CORRECTNESS_ANCHORS = {
    "chemotaxis_direction": dict(criterion="sign_match_or_delta_le_0.15",
                                 ref_ci_sign=1.0),
    "escape_direction": dict(criterion="d_peak_gt_0.3_or_curl_dominant",
                             d_peak_threshold=0.3),
    "dual_state": dict(criterion="silent_in_band_and_bout_activity",
                       silent_band=SILENT_BAND,
                       bout_floor=BOUT_ACTIVITY_FLOOR),
}


def g1_dual_state_check(rest: dict, spont: dict) -> dict:
    """G1 双状态门判据：静息低活动（静默比例落带）+ 行为 bout 双状态。

    Returns dict(silent_frac, silent_in_band, bout_activity, bout_ok,
    dual_state, verdict, detail)。
    """
    silent = float(rest.get("silent_frac", float("nan")))
    silent_in_band = bool(SILENT_BAND[0] <= silent <= SILENT_BAND[1])
    frac = spont.get("frac", {})
    bout = float(frac.get("fwd", 0.0)) + float(frac.get("turn", 0.0)) \
        + float(frac.get("rev", 0.0))
    bout_ok = bool(bout >= BOUT_ACTIVITY_FLOOR)
    dual = bool(silent_in_band and bout_ok)
    return dict(
        silent_frac=round(silent, 4), silent_band=list(SILENT_BAND),
        silent_in_band=silent_in_band, bout_activity=round(bout, 4),
        bout_floor=BOUT_ACTIVITY_FLOOR, bout_ok=bout_ok,
        state_frac={k: round(float(v), 4) for k, v in frac.items()},
        dual_state=dual,
        verdict="PASS" if dual else "FAIL",
        detail=("静默比例落带 [50,90]% 且行为 bout 活动 ≥10%"
                if dual else
                f"静默 {silent:.2f}（带 {SILENT_BAND}）"
                f"或 bout 活动 {bout:.2f}（下限 {BOUT_ACTIVITY_FLOOR}）未达"),
    )
