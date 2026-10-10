"""M9 P9：全规模回路扰动（类型级锚 → 代理目标池）与因果后果读取（`src/adult_perturb.py`，新建）。

《生物仿真M9实施清单》§1.4（扰动锚库：类型级锚 ≥20 下限）/§4.5（P9 判据：命中率 ≥70% +
sham 零臂 + 确定性）/§0.3.5（抽象登记）/§0.7 #8（判据带定稿于 CSV 不事后调）。

**前置警示（P7 R3 的机制性反证）**：`data/m9_p7_drive_sensitivity.csv` 证明
——零背景驱动仍保留 98.5% 群发放率、4× 驱动仅 +7.2% → **活动自主自持（tonic 偏置主导）、
输入通路增益过低**。因此**类型级扰动可能不显著**；该结果本身即 M9 的正当科学结论
（"静态/结构层面成立、状态切换动力学不涌现"），**不得为凑命中率扭曲判据或事后改带**。

扰动语义（预注册，见 `data/m9_behavior_reference.csv` 的 perturbation 段）：
  - **沉默**：目标池**输出边** gmax → 0（`AdultEngine.scale_synapses`，拓扑不变）；
  - **激活**：目标池**逐神经元 tonic 偏置** += mult × median(bias)（`set_neuron_heterogeneity`，
    读回断言生效）；
  - **sham 零臂自检**：经同一代码路径施加 factor=1.0（乘法恒等）→ 必须**逐位无变化**；
  - **随机 size-matched 对照**：同一操作施加到同规模随机神经元集（**特异性/效应量对照**，
    非"必须无变化"臂）。

后果类（预注册，`effect_threshold_rel`）：相对 sham 基线 |Δ|/base ≥ 0.05 → `activity_up` /
`activity_down`；否则 `no_change`。预测为**机械规则**（非拟合）：

    操纵边的主导递质类 C（由连接组数据读出）→ 激活：C 抑制 ⇒ 伙伴池 ↓；C 兴奋/调质 ⇒ 伙伴池 ↑
    沉默：方向相反；自身池预测：激活 ⇒ ↑（**操作有效性前置**）；沉默 ⇒ no_change

**代理池与塌缩（诚实登记）**：v783 无 cell_type（Schlegel parquet 不可得，R7）→ 类型级锚
（如 MBON-α1 与 MBON-γ5β'2a）**无法分辨**，只能落到**同一区域级代理池** →
`pool_collapse` 如实记录，命中率**同时报告锚级与池级**（池级为统计诚实单位）。

用法：
    from neural_exploration.src.adult_perturb import AdultPerturb, ANCHOR_MAP
    from neural_exploration.src.adult_circuit import AdultCircuit, CircuitParams
    c = AdultCircuit(device="mps", params=CircuitParams(**PARAMS)); c.build()
    p = AdultPerturb(c); p.prepare_silence("MBON"); r = p.run(T_ms=1000, settle_ms=300)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

CLS_EXC, CLS_INH, CLS_MOD = 0, 1, 2
CLASS_NAMES = {CLS_EXC: "excitatory", CLS_INH: "inhibitory", CLS_MOD: "modulatory"}

#: 后果类枚举
UP, DOWN, NOCHANGE = "activity_up", "activity_down", "no_change"


# ---------------------------------------------------------------------------
# 类型级锚 → 代理目标（预注册映射；不可映射者登记 unavailable → no-experiment，不入分母）
# ---------------------------------------------------------------------------
#: 代理池规格：regions = 主脑区前缀（neuron table region_names）；nt_class = 逐神经元主导递质类码
POOL_SPECS: Dict[str, Dict[str, Any]] = {
    "MBON": {"regions": ("MB_ML", "MB_VL", "MB_PED"), "desc":
             "MBON 输出区（蘑菇体叶/梗）——承接 src/adult_mb.py 的 MBON 代理（1374 神经元）"},
    "MOD_MB": {"nt_class": CLS_MOD, "regions": ("MB_CA", "MB_ML", "MB_VL", "MB_PED"),
               "desc": "投射到蘑菇体的调质类神经元（PAM/PPL1 DAN 代理；递质类级不可分 DA/5HT/OA）"},
    "MOD_ALL": {"nt_class": CLS_MOD, "desc":
                "全脑调质类神经元（dom_nt=modulatory，3300）——OA/DA/5HT 系统代理"},
    "CX_EB": {"regions": ("EB",), "desc": "椭球体（EB）——EPG 环状吸引子代理（381 神经元）"},
    "CX_ALL": {"regions": ("EB", "PB", "FB"), "desc":
               "中央复合体（EB+PB+FB）——转向回路代理（承接 P6）"},
}

#: 边级锚：目标 = pre∈from_regions & post∈to_pool 的边（沉默其 gmax）
EDGE_SPECS: Dict[str, Dict[str, Any]] = {
    "KC_TO_MBON": {"from_regions": ("MB_CA",), "to_pool": "MBON",
                   "desc": "KC（MB_CA）→ MBON 真实边（P5 承接：100,533 条）"},
}

#: 锚 → 目标映射（预注册；参考锚库 data/m9_perturbation_plan.csv 的 26 条类型级锚）
#: pred_act = **激活**该锚目标时期待的"伙伴池（直接突触后）后果方向"（+1 ↑ / −1 ↓）；
#:           沉默 = 反号。方向来源（pred_source）为**文献机制**或**已登记的抽象（§0.3.5）**，
#:           **不可**由实测结果反推（避免循环论证）。
ANCHOR_MAP: Dict[str, Dict[str, Any]] = {}


def _a(anchor_id: str, kind: str, target: Optional[str], lit: str, reason: str,
       pred_act: int = 0, pred_source: str = "") -> None:
    ANCHOR_MAP[anchor_id] = {"kind": kind, "target": target, "lit_direction": lit,
                             "map_reason": reason, "pred_act": int(pred_act),
                             "pred_source": pred_source}


_SRC_MBON_GABA = ("文献机制：MBON 为抑制性（GABA 能）输出神经元 → 激活抑制下游 ⇒ 伙伴池 ↓；"
                  "沉默解除抑制 ⇒ 伙伴池 ↑")
_SRC_MOD = ("登记抽象 T2（§0.3.5）：调质类（DA/5HT/OA）在本模型中由**弱兴奋电导**承载"
            "（受体映射 v783 不可得）→ 激活 ⇒ 伙伴池 ↑")
_SRC_EXC = "连接组结构：该通路主导递质为兴奋性（ACh/谷氨酸）→ 激活 ⇒ 伙伴池 ↑"
_SRC_KC = ("连接组结构：KC→MBON 边为兴奋性（KC 胆碱能）→ 边沉默 ⇒ MBON（post）池 ↓；"
           "承接 P5 真实边 100,533 条")


# ---- Aso 2014：MBON 价效（13 个区间锚，价格效/抑制性输出）----
for _aid, _lit in [
    ("A14-MBON-a1", "avoidance"), ("A14-MBON-a2", "avoidance"),
    ("A14-MBON-a3", "avoidance"), ("A14-MBON-a1p", "avoidance"),
    ("A14-MBON-a2p", "avoidance"), ("A14-MBON-a3p", "approach"),
    ("A14-MBON-b1", "approach"), ("A14-MBON-b2", "approach"),
    ("A14-MBON-g1pedc", "avoidance"), ("A14-MBON-g2a1p", "approach"),
    ("A14-MBON-g3", "avoidance"), ("A14-MBON-g4g5", "mixed"),
    ("A14-MBON-g5b2a", "avoidance"),
]:
    _a(_aid, "pool", "MBON", _lit, "MBON 区间锚 → 区域级 MBON 池代理（无 cell_type，区间不可分）",
       -1, _SRC_MBON_GABA)
# ---- Aso 2014：DAN / 调质 ----
_a("A14-PAM-reward", "pool", "MOD_MB", "reward",
   "PAM 簇（奖赏 DAN）→ 投射 MB 的调质类神经元代理", +1, _SRC_MOD)
_a("A14-PPL1-punish", "pool", "MOD_MB", "punishment",
   "PPL1 簇（惩罚 DAN）→ 投射 MB 的调质类神经元代理（与 PAM 同池：递质类级不可分）",
   +1, _SRC_MOD)
_a("A14-MBON-OA", "pool", "MOD_ALL", "arousal",
   "OA-VUM → 全脑调质类代理（递质类级不可分 DA/5HT/OA）", +1, _SRC_MOD)
# ---- Robie 2017（optogenetic 激活-行为图谱）----
_a("R17-GF-escape", "unavailable", None, "escape/jump",
   "GF 为单对神经元；无 super_class/cell_type（R7）→ 无法隔离 → no-experiment")
_a("R17-DN-downstream", "unavailable", None, "walking/turn/grooming",
   "descending neuron 身份需 cell_type/super_class；神经域标注无 VNC/SEG → 无法映射 → no-experiment")
_a("R17-MBON-behavior", "pool", "MBON", "approach/avoidance 转向",
   "MBON 驱动线集合 → MBON 池代理（**行为通道读出 P9 不可测**：P9 仅活动级，行为级登记限制）",
   -1, _SRC_MBON_GABA)
_a("R17-turning-lines", "pool", "CX_ALL", "turning bias",
   "CX 驱动线（EPG/PFN 类）→ 中央复合体（EB+PB+FB）代理", +1, _SRC_EXC)
_a("R17-walking-lines", "unavailable", None, "walking initiation",
   "‘多驱动线’无类型/区域可分辨代理 → no-experiment")
# ---- Claudi 2024（预印本；原始表网络受限不可得）----
_a("C24-cluster-atlas", "unavailable", None, "行为簇",
   "原始数据表不可得（anchor_mapping=unavailable 由锚库自身登记）→ no-experiment")
_a("C24-larval-compare", "unavailable", None, "跨龄期对照",
   "方法学对照项，无神经元目标 → no-experiment")
# ---- 其他机制锚 ----
_a("MISC-KC-MBON-STDP", "edge", "KC_TO_MBON", "Δw 符号（DA 门控）",
   "KC→MBON 突触锚 → **边级**目标（pre∈MB_CA & post∈MBON），沉默其 gmax 读 MBON 池后果",
   -1, _SRC_KC)
_a("MISC-MBON-US", "pool", "MBON", "ΔMBON 输出 ∝ 偏好",
   "MBON 输出差分读出锚 → MBON 池代理（价效差分本身为 P8 读出，P9 只测活动级后果）",
   -1, _SRC_MBON_GABA)
_a("MISC-EPG-ring", "pool", "CX_EB", "bump ↔ 航向；EPG 消融 → 丢失",
   "EPG 环锚 → EB 池代理（缝隙连接 v783=0：环序为图论代理，R8）", +1, _SRC_EXC)
_a("MISC-SLEEP-OA", "pool", "MOD_ALL", "OA 增益 ↑ → 觉醒",
   "OA/DA 调质系统锚 → 全脑调质类代理", +1, _SRC_MOD)
_a("MISC-NOCICEPTION", "unavailable", None, "逃避/蜷缩 ↑",
   "成虫伤害性类型（class IV md 同源）无 cell_type 可分辨 → no-experiment")


@dataclass
class PerturbResult:
    """单次扰动运行的读出口径（全部为测量窗内统计）。"""
    tag: str = ""
    mode: str = ""
    pool: str = ""
    factor: float = 0.0
    delta_bias_mult: float = 0.0
    n_target: int = 0
    n_target_edges: int = 0
    target_nt_class: int = -1
    n_partner: int = 0
    self_rate_hz: float = 0.0
    partner_rate_hz: float = 0.0
    pop_rate_hz: float = 0.0
    n_spikes: float = 0.0
    counts_hash: str = ""
    counts: Optional[np.ndarray] = field(default=None, repr=False)
    wall_s: float = 0.0
    ms_per_step: float = 0.0

    def brief(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "counts"}
        return d

class AdultPerturb:
    """全规模回路的类型级锚扰动管线（沉默 / 激活 / sham / 随机对照）。"""

    def __init__(self, circuit, verbose: bool = True):
        self.c = circuit
        self.verbose = verbose
        if not circuit._built:
            raise RuntimeError("先 circuit.build()")
        self._pristine = np.array(circuit._gmax, dtype=np.float32, copy=True)
        self._bias0 = np.array(circuit.bias_vector(), dtype=np.float32)
        self._masks: Dict[str, np.ndarray] = {}
        self._edge_masks: Dict[str, np.ndarray] = {}
        self._mode = "none"
        self._active: Optional[Dict[str, Any]] = None

    # ---------------- 目标集 ----------------
    def pool_mask(self, key: str) -> np.ndarray:
        if key in self._masks:
            return self._masks[key]
        spec = POOL_SPECS[key]
        m = np.ones(self.c.n_neurons, dtype=bool)
        if spec.get("regions"):
            m &= self.c.region_mask(tuple(spec["regions"]))
        if spec.get("nt_class") is not None:
            m &= (self.c.dom_nt_code == int(spec["nt_class"]))
        self._masks[key] = m
        return m

    def edge_mask(self, key: str) -> np.ndarray:
        """边级目标掩码（原始边序）：pre∈from_regions & post∈to_pool。"""
        if key in self._edge_masks:
            return self._edge_masks[key]
        spec = EDGE_SPECS[key]
        pre_m = self.c.region_mask(tuple(spec["from_regions"]))
        post_m = self.pool_mask(spec["to_pool"])
        m = pre_m[self.c.pre] & post_m[self.c.post]
        self._edge_masks[key] = m
        return m

    def out_edge_mask(self, pool: str) -> np.ndarray:
        """目标池的**输出边**掩码（原始边序）。"""
        return self.pool_mask(pool)[self.c.pre]

    def partner_mask(self, pool: str) -> np.ndarray:
        """伙伴池 = 目标池输出边的突触后神经元（**直接**后果读取面）。"""
        m = np.zeros(self.c.n_neurons, dtype=bool)
        m[self.c.post[self.out_edge_mask(pool)]] = True
        return m

    def dominant_nt_class(self, edge_mask: np.ndarray) -> int:
        """操纵边的主导递质类（由连接组数据读出 → 预测方向规则的输入）。"""
        nt = self.c.nt_code[edge_mask]
        if nt.size == 0:
            return -1
        cnt = np.bincount(nt.astype(np.int64), minlength=3)
        return int(np.argmax(cnt))

    def random_matched_mask(self, pool: str, seed: int = 7717) -> np.ndarray:
        """同规模随机对照集（确定性；排除目标池与伙伴池以免污染特异性对照）。"""
        tgt = self.pool_mask(pool)
        excl = tgt | self.partner_mask(pool)
        cand = np.flatnonzero(~excl)
        rng = np.random.default_rng(seed)
        k = int(tgt.sum())
        pick = rng.choice(cand, size=min(k, cand.size), replace=False)
        m = np.zeros(self.c.n_neurons, dtype=bool)
        m[pick] = True
        return m

    # ---------------- 施加扰动 ----------------
    def restore(self) -> None:
        """恢复原始权重与偏置（sham 零臂的前置）。"""
        self.c.engine.set_gmax(self._pristine)
        self.c.engine.set_neuron_heterogeneity(i_bias=self._bias0)
        self._mode = "none"
        self._active = None

    def _partner_of(self, tgt: np.ndarray) -> np.ndarray:
        m = np.zeros(self.c.n_neurons, dtype=bool)
        m[self.c.post[tgt[self.c.pre]]] = True
        return m

    def prepare_silence(self, pool: Optional[str] = None, factor: float = 0.0,
                        mask: Optional[np.ndarray] = None,
                        label: str = "") -> Dict[str, Any]:
        """沉默：目标池输出边 gmax ×= factor（factor=0 → 完全沉默；factor=1.0 → sham 零臂）。"""
        self.restore()
        tgt = (self.pool_mask(pool) if mask is None else np.asarray(mask, bool))
        em = tgt[self.c.pre]
        self.c.engine.scale_synapses(em, float(factor))
        self.c.engine.set_neuron_heterogeneity(i_bias=self._bias0)  # 零臂：偏置显式复位
        info = {"mode": "silence", "pool": pool or label, "factor": float(factor),
                "n_target": int(tgt.sum()), "n_target_edges": int(em.sum()),
                "target_nt_class": self.dominant_nt_class(em),
                "self_mask": tgt, "partner_mask": self._partner_of(tgt)}
        self._mode = "silence"
        self._active = info
        return info

    def prepare_edge_silence(self, edge_key: Optional[str] = None, factor: float = 0.0,
                             edge_mask: Optional[np.ndarray] = None,
                             label: str = "") -> Dict[str, Any]:
        """边级沉默（如 KC→MBON）：仅目标边 gmax ×= factor（拓扑与神经元集不变）。"""
        self.restore()
        em = (self.edge_mask(edge_key) if edge_mask is None
              else np.asarray(edge_mask, bool))
        self.c.engine.scale_synapses(em, float(factor))
        self.c.engine.set_neuron_heterogeneity(i_bias=self._bias0)
        pre_m = np.zeros(self.c.n_neurons, dtype=bool)
        pre_m[self.c.pre[em]] = True
        post_m = np.zeros(self.c.n_neurons, dtype=bool)
        post_m[self.c.post[em]] = True
        info = {"mode": "edge_silence", "pool": edge_key or label,
                "factor": float(factor), "n_target": int(pre_m.sum()),
                "n_target_edges": int(em.sum()),
                "target_nt_class": self.dominant_nt_class(em),
                "self_mask": pre_m, "partner_mask": post_m}
        self._mode = "edge_silence"
        self._active = info
        return info

    def random_matched_edge_mask(self, edge_key: str, seed: int = 991) -> np.ndarray:
        """同规模随机边集（确定性；排除目标边）——边级操作的特异性对照。"""
        em = self.edge_mask(edge_key)
        cand = np.flatnonzero(~em)
        rng = np.random.default_rng(seed)
        pick = rng.choice(cand, size=int(em.sum()), replace=False)
        out = np.zeros(em.shape, dtype=bool)
        out[pick] = True
        return out

    def prepare_activation(self, pool: Optional[str] = None, mult: float = 1.0,
                           mask: Optional[np.ndarray] = None,
                           label: str = "") -> Dict[str, Any]:
        """激活：目标池逐神经元 tonic 偏置 += mult × median(bias)（读回断言生效）。"""
        self.restore()
        tgt = (self.pool_mask(pool) if mask is None else np.asarray(mask, bool))
        delta = float(mult) * float(np.median(self._bias0))
        bias = self._bias0.copy()
        bias[tgt] += delta
        self.c.engine.set_neuron_heterogeneity(i_bias=bias)
        # **读回断言**（血泪教训 3：字段存在但作用域不匹配 = 静默失效）
        back = self.c.engine.t_iext.detach().cpu().numpy()
        got = float(back[tgt].mean()) - float(self._bias0[tgt].mean()) if tgt.any() else 0.0
        if tgt.any() and abs(got - delta) > 1e-3:
            raise RuntimeError("激活读回失败：期望 Δ=%.4f 实测 %.4f（静默失效防护）"
                               % (delta, got))
        em = tgt[self.c.pre]
        info = {"mode": "activation", "pool": pool or label, "factor": float(mult),
                "delta_bias_mv_s": delta, "n_target": int(tgt.sum()),
                "n_target_edges": int(em.sum()),
                "target_nt_class": self.dominant_nt_class(em),
                "readback_delta_mean": got,
                "self_mask": tgt, "partner_mask": self._partner_of(tgt)}
        self._mode = "activation"
        self._active = info
        return info

    # ---------------- 运行（全分段） ----------------
    def run(self, T_ms: float = 1000.0, settle_ms: float = 300.0, seed: int = 0,
            segment_steps: int = 2000, abort_s: float = 180.0,
            pop_trace: bool = False, tag: str = "") -> PerturbResult:
        """分段静息协议（复刻 `AdultCircuit.run_resting` 语义，全分段规避长单次调用退化）。"""
        c = self.c
        p = c.params
        dt = p.dt_ms
        n_settle = int(round(settle_ms / dt))
        n_meas = int(round(T_ms / dt))
        rng0 = np.random.default_rng(1)
        v0 = (p.v_rest + 2.0 * rng0.standard_normal(c.n_neurons)).astype(np.float32)
        c.engine.reset(v0=v0, seed=seed)
        t0 = time.perf_counter()
        done = 0
        while done < n_settle:
            m = min(segment_steps, n_settle - done)
            c.engine.run(m, delivery="chunk", record="counts")
            done += m
        c.engine.reset_counts()
        done = 0
        t_seg = time.perf_counter()
        while done < n_meas:
            m = min(segment_steps, n_meas - done)
            tt = time.perf_counter()
            c.engine.run(m, delivery="chunk", record="counts", pop_trace=pop_trace)
            if time.perf_counter() - tt > abort_s:
                raise RuntimeError("段墙钟 %.0fs > 上限 %.0fs → 引擎退化，主动中止"
                                   % (time.perf_counter() - tt, abort_s))
            done += m
        counts = c.engine.t_count.detach().cpu().numpy().astype(np.float64)
        ms_per_step = (time.perf_counter() - t_seg) / max(n_meas, 1) * 1e3
        act = self._active or {}
        pool = act.get("pool")
        z = np.zeros(c.n_neurons, dtype=bool)
        tgt = np.asarray(act.get("self_mask", z), bool)
        partner = np.asarray(act.get("partner_mask", z), bool)
        T_s = T_ms / 1000.0
        res = PerturbResult(
            tag=tag, mode=str(act.get("mode", "none")), pool=str(pool),
            factor=float(act.get("factor", 0.0)),
            delta_bias_mult=float(act.get("factor", 0.0)) if act.get("mode") == "activation" else 0.0,
            n_target=int(act.get("n_target", 0)), n_target_edges=int(act.get("n_target_edges", 0)),
            target_nt_class=int(act.get("target_nt_class", -1)),
            n_partner=int(partner.sum()),
            self_rate_hz=float(counts[tgt].sum() / max(tgt.sum(), 1) / T_s),
            partner_rate_hz=float(counts[partner].sum() / max(partner.sum(), 1) / T_s),
            pop_rate_hz=float(counts.sum() / c.n_neurons / T_s),
            n_spikes=float(counts.sum()), counts=counts,
            counts_hash=_hash(counts), wall_s=time.perf_counter() - t0,
            ms_per_step=ms_per_step)
        if self.verbose:
            print("    [%s] %s pool=%s 目标 %d 边 %d 伙伴 %d | 自身 %.3f 伙伴 %.3f 群体 %.3f Hz "
                  "| %.0fs (%.2f ms/step)"
                  % (tag, res.mode, res.pool, res.n_target, res.n_target_edges,
                     res.n_partner, res.self_rate_hz, res.partner_rate_hz,
                     res.pop_rate_hz, res.wall_s, res.ms_per_step), flush=True)
        return res


def _hash(counts: np.ndarray) -> str:
    import hashlib
    return hashlib.sha1(np.asarray(counts, dtype=np.float64).tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 预测 / 判定（机械规则；预注册）
# ---------------------------------------------------------------------------
def predict_direction(manip: str, nt_class: int) -> int:
    """机械预测规则：返回伙伴池发放率的预期变化方向（+1 ↑ / −1 ↓ / 0 no_change）。

    规则（运行前定稿）：激活（tonic 注入）目标池并向其输出边递质类 C：
      C 抑制（GABA）⇒ 伙伴池 ↓；C 兴奋/调质 ⇒ 伙伴池 ↑；沉默则方向相反。
    """
    if nt_class < 0:
        return 0
    sgn = -1 if nt_class == CLS_INH else +1
    if manip == "activation":
        return sgn
    if manip in ("silence", "edge_silence"):
        return -sgn
    return 0


def classify(delta: float, thr: float) -> str:
    """后果类判定（预注册阈值）：|Δ|/base ≥ thr → up/down；否则 no_change。"""
    if delta >= thr:
        return UP
    if delta <= -thr:
        return DOWN
    return NOCHANGE


def dir_to_class(d: int) -> str:
    return UP if d > 0 else (DOWN if d < 0 else NOCHANGE)


def rel_delta(x: float, base: float) -> float:
    return (x - base) / max(abs(base), 1e-12)


def map_predictions(anchors: List[Dict[str, Any]], pools_nt: Dict[str, int]
                    ) -> List[Dict[str, Any]]:
    """为每个锚生成**可测预测槽**（预注册；运行前写盘）。

    方向来源 = 锚自身的**文献机制**或**已登记抽象**（`ANCHOR_MAP.pred_act`），**不由实测反推**；
    `pools_nt`（代理池主导递质类）仅作**诊断列**——区域代理混类时如实登记（如 MBON 代理池
    输出边兴奋性占优 ⇒ 区域代理含非 MBON 神经元，为测量限制 R7 的直接证据）。
    """
    out = []
    for a in anchors:
        kind, tgt = a["kind"], a["target"]
        pred_act = int(a.get("pred_act", 0))
        if kind == "unavailable" or pred_act == 0:
            out.append({**a, "slot": "", "readout": "", "predicted": "no-experiment",
                        "predicted_dir": 0})
            continue
        if kind == "pool":
            out.append({**a, "slot": "S1_silence_partner", "readout": f"{tgt}.partner",
                        "predicted": dir_to_class(-pred_act), "predicted_dir": -pred_act})
            out.append({**a, "slot": "S2_silence_self", "readout": f"{tgt}.self",
                        "predicted": NOCHANGE, "predicted_dir": 0})
            out.append({**a, "slot": "A1_activation_partner", "readout": f"{tgt}.partner",
                        "predicted": dir_to_class(pred_act), "predicted_dir": pred_act})
        elif kind == "edge":
            out.append({**a, "slot": "S1_edge_silence_post", "readout": f"{tgt}.post",
                        "predicted": dir_to_class(-pred_act), "predicted_dir": -pred_act})
            out.append({**a, "slot": "S2_edge_silence_pre", "readout": f"{tgt}.pre",
                        "predicted": NOCHANGE, "predicted_dir": 0})
    return out
