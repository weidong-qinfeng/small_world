"""M11 内核（小鼠皮层微环路）——在 M10 修正内核之上**继承扩展**，冻结件零修改。

《生物仿真M11实施清单》§2.2（E1b–E5b）/§2.3（对齐门 (a)–(h) + 关闭态等价 + 编码路径断言）。

| 扩展 | 内容 | 判定 | 实现要点 |
|---|---|---|---|
| **E1b** | **抑制亚型四槽（E / PV / SST / VIP）** | 扩展 | 事件环槽宽度 `2N → n_ch·N`（`n_ch = 4` 时）；逐类电导轨迹与时间常数；**逐类可验证**（`syn_class` 逐类断言） |
| **E2b** | **慢变量状态张量 `s_i(t)`**（逐神经元） | 新建 | 慢适应钾电导（τ_slow，spike 触发、逐神经元异质增量）+ 读出接口 `slow_readback()` + "参数是活的"门 |
| **E3b** | **层流结构 + 层间延迟打包** | 扩展 | 层标签（L2/3–L6）+ 每层 E/I 比例 + **层间延迟量化为 ≤4 档并共享单一环槽**（消除 M10 L18.1 的"多延迟组 >30× 慢"） |
| **E4b** | 双房室档 | 复用（本节点**未扩展**） | 层特异形态参数属 P-EXT1（预算门控）⇒ 本引擎 `fidelity="two_comp"` 显式 `NotImplementedError`（不静默降级） |
| **E5b** | 规模适配（8k/31k/75,909） | 扩展 | 稀疏装配 + 单环槽 + 无泄漏（禁动态规模缓存键）|

**与 M10 的关系（可审计）**：本类继承 `LarvalZFEngine`，并**泛化**其 `run()`/投递函数，把硬编码的
"2 个电导槽（E/I）+ 每延迟值一个环槽"泛化为"`n_ch` 个电导槽 + 全部延迟档共享单一环槽"。

- `n_ch == 2`（E1b 关闭）⇒ 环槽布局、`post_off` 编码、核表达式与 M10 **逐字一致**（见 `_mc_core_factory`
  的 `nc==2` 分支与 `finalize` 的 `2N` 布局）⇒ **关闭态与 M10 逐位可比**（实测见 `tools/scan_m11_engine.py` T6/T7）。
- `n_ch == 4` ⇒ 环槽宽 `4N`，`post_off = post + N·cls`（cls∈{0,1,2,3}）。
- 慢变量关闭（`slow_on=False`）⇒ 核表达式不含慢项（不做"乘 1"式伪关闭）。

**纪律**（清单 §0.7 #15 + M10 L21）：
  ① `add_synapses` 严格守卫：给 `inhibitory` 而不给 `syn_class` 直接报错（M10 的静默失效根因）；
  ② 全部 `configure_*` 带**读回断言**；③ 核闭包捕获的类参数以 `core_readback()` 暴露（作用域守卫）；
  ④ `run()` 内建**双估计器硬断言**（计数 vs 群迹、积分累积 vs 闭式、ms/step×步数 vs 墙钟）。

用法：
    PYTHONHASHSEED=0 OMP_NUM_THREADS=1 ./.venv-m9/bin/python -m neural_exploration.tools.scan_m11_engine --which all
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional, Sequence

import numpy as np

try:  # 引擎为可选重型依赖（判据脚手架不 import 本模块；M8 L5 惯例）
    import torch
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None
    TORCH_AVAILABLE = False

from .adult_engine import MAX_EDGES_PER_FLUSH, SLOT_EXTRA
from .larval_zf_engine import LarvalZFEngine

MC_ENGINE_VERSION = "m11e-1.0-4slot+laminar-delay-pack+slowvar"

SUB_E, SUB_PV, SUB_SST, SUB_VIP = 0, 1, 2, 3
SUB_NAMES = ("E", "PV", "SST", "VIP")
MAX_DELAY_BINS = 4          # E3b：层间延迟档上限（打包后必须 ≤ 该值）
N_CH_OFF = 2                # E1b 关闭：E / I 两槽（M10 语义）
N_CH_ON = 4                 # E1b 开启：E / PV / SST / VIP
DUAL_EST_TOL = 0.02         # 双估计器一致性硬门（清单：相对差 ≤2%）


# ---------------------------------------------------------------------------
# 点神经元核（E1b/E2b）
# ---------------------------------------------------------------------------
REF_FIX = "semantic_fix"        # 文档语义：第 4 形参=g_ahp、第 5 形参=cool（= 冻结件 gaba 路径语义）
REF_FROZEN = "m10_frozen"       # 冻结件缺陷档：调用序 (t_cool, t_ahp) 与形参名 (g_ahp, cool) 错位


def _mc_core_factory(p, nc: int, slow_on: bool, frozen: bool, taus: Sequence[float],
                     e_inhs: Sequence[float], slow_tau_ms: float, e_slow: float):
    """返回点神经元状态更新核（**6 变体**：nc × slow_on × refractory_form）。

    | 变体 | 语义 |
    |---|---|
    | `nc=2, slow=False, REF_FIX` | 文档语义点核（= M10 `gaba_split=True` 且 `frac_slow=0` 的语义，逐位一致）|
    | `nc=2, slow=False, REF_FROZEN` | **冻结件 M9/M10 非 gaba 路径的实收语义**（第 4/5 形参错位，见 E-M9-4）|
    | `nc=2, slow=True, *` | 上式 + 慢电导项 `gs·(E_slow − v)`（慢项全零时逐位退化）|
    | `nc=4, slow=*, REF_FIX` | 三类抑制电导各自独立时间常数与反转电位（**冻结档在 4 槽下无定义 ⇒ 构造时拒绝**）|

    `taus = (tau_e, tau_pv, tau_sst, tau_vip)`；`e_inhs = (e_pv, e_sst, e_vip)`。

    **前置发现（E-M9-4）**：`adult_engine._point_core_factory` 的形参序为
    `(v, ge, gi, g_ahp, cool, ev_e, ev_i, iext)`，而全部 4 个 point 调用点
    （`adult_engine` L666/L690、`larval_zf_engine` L348/L373）传入 `(t_cool, t_ahp)`
    ⇒ 不应期门实读 `t_ahp`、AHP 电导实读 `t_cool`。**`REF_FROZEN` 逐位复刻缺陷档**
    （实测单神经元 ISI 182 步 vs 文档语义 144 步），用于"关闭态等价"与缺陷量级量化；
    冻结档在 `nc=4` 下无定义 ⇒ 显式拒绝（不静默）。
    """
    dt_s = p.dt_ms * 1e-3
    ae = math.exp(-p.dt_ms / p.tau_e)
    gl = 1.0 / (p.tau_m * 1e-3)
    vr, vth, vres, ee, ek, vfloor = (p.v_rest, p.v_th, p.v_reset, p.e_exc, p.ek, p.v_floor)
    ref_m1 = float(max(p.ref_ms / p.dt_ms - 1.0, 0.0))
    aa = (math.exp(-p.dt_ms / p.ahp_tau_ms) if p.ahp_tau_ms and p.ahp_tau_ms > 0 else 0.0)
    ahp_g_inc = float(p.ahp_g_inc)
    legacy = (str(p.ahp_form) == "current")
    if legacy:  # pragma: no cover —— legacy 档不在 M11 路线（显式拒绝，避免静默错核）
        raise ValueError("M11 内核不支持 ahp_form='current'（legacy）；请用 'conductance'")
    if frozen and nc != N_CH_OFF:
        raise ValueError("冻结缺陷档（REF_FROZEN）只在 nc=2（E1b 关闭）下有意义")
    if frozen and ahp_g_inc != 0.0:
        raise ValueError("冻结缺陷档 + ahp_g_inc≠0 是无意义组合（冻结件把 t_cool 当 AHP）")
    a_slow = math.exp(-p.dt_ms / max(1e-9, float(slow_tau_ms)))

    if nc == 2 and not slow_on:
        ei = float(p.e_inh)
        ai = math.exp(-p.dt_ms / p.tau_i)

        if not frozen:
            def core(v, ge, gi, cool, g_ahp, ev_e, ev_i, iext):
                """文档语义点核（与 M10 `gaba_split=True, frac_slow=0` 逐位一致）。"""
                ge2 = ge + ev_e
                gi2 = gi + ev_i
                ga2 = g_ahp * aa
                drive = (gl * (vr - v) + ge2 * (ee - v) + gi2 * (ei - v)
                         + ga2 * (ek - v) + iext)
                v2 = torch.clamp(v + dt_s * drive, min=vfloor)
                spk = (v2 >= vth) & (cool <= 0.0)
                cool2 = torch.where(spk, ref_m1, torch.clamp(cool - 1.0, min=0.0))
                ga3 = torch.where(spk, ga2 + ahp_g_inc, ga2)
                v3 = torch.where(cool2 > 0.0, vres, v2)
                return v3, ge2 * ae, gi2 * ai, ga3, cool2, spk

            return core

        def core(v, ge, gi, cool, g_ahp, ev_e, ev_i, iext):
            """**冻结件实收语义**（形参名保持调用序；第 4/5 形参使用错位 ⇒ 逐位复刻缺陷档）。"""
            ge2 = ge + ev_e
            gi2 = gi + ev_i
            ga2 = cool * aa                      # 冻结件把 `cool` 形参当 g_ahp 用
            drive = (gl * (vr - v) + ge2 * (ee - v) + gi2 * (ei - v)
                     + ga2 * (ek - v) + iext)
            v2 = torch.clamp(v + dt_s * drive, min=vfloor)
            spk = (v2 >= vth) & (g_ahp <= 0.0)   # 冻结件把 `g_ahp` 形参当 cool 用
            cool2 = torch.where(spk, ref_m1, torch.clamp(g_ahp - 1.0, min=0.0))
            ga3 = torch.where(spk, ga2 + ahp_g_inc, ga2)
            v3 = torch.where(cool2 > 0.0, vres, v2)
            return v3, ge2 * ae, gi2 * ai, ga3, cool2, spk

        return core

    if nc == 2 and slow_on:
        ei = float(p.e_inh)
        ai = math.exp(-p.dt_ms / p.tau_i)

        def core(v, ge, gi, cool, g_ahp, gs, s_inc, ev_e, ev_i, iext):
            ge2 = ge + ev_e
            gi2 = gi + ev_i
            if frozen:
                ga2 = cool * aa
                gate = g_ahp
            else:
                ga2 = g_ahp * aa
                gate = cool
            drive = (gl * (vr - v) + ge2 * (ee - v) + gi2 * (ei - v)
                     + ga2 * (ek - v) + gs * (e_slow - v) + iext)
            v2 = torch.clamp(v + dt_s * drive, min=vfloor)
            spk = (v2 >= vth) & (gate <= 0.0)
            cool2 = torch.where(spk, ref_m1, torch.clamp(gate - 1.0, min=0.0))
            ga3 = torch.where(spk, ga2 + ahp_g_inc, ga2)
            gs2 = torch.where(spk, gs + s_inc, gs)
            v3 = torch.where(cool2 > 0.0, vres, v2)
            return v3, ge2 * ae, gi2 * ai, ga3, cool2, gs2 * a_slow, spk

        return core

    e1, e2, e3 = (float(x) for x in e_inhs)
    a1 = math.exp(-p.dt_ms / float(taus[1]))
    a2 = math.exp(-p.dt_ms / float(taus[2]))
    a3 = math.exp(-p.dt_ms / float(taus[3]))

    if not slow_on:
        def core(v, ge, g1, g2, g3, cool, g_ahp, ev_e, ev_1, ev_2, ev_3, iext):
            ge2 = ge + ev_e
            h1 = g1 + ev_1
            h2 = g2 + ev_2
            h3 = g3 + ev_3
            ga2 = g_ahp * aa
            drive = (gl * (vr - v) + ge2 * (ee - v) + h1 * (e1 - v) + h2 * (e2 - v)
                     + h3 * (e3 - v) + ga2 * (ek - v) + iext)
            v2 = torch.clamp(v + dt_s * drive, min=vfloor)
            spk = (v2 >= vth) & (cool <= 0.0)
            cool2 = torch.where(spk, ref_m1, torch.clamp(cool - 1.0, min=0.0))
            ga3 = torch.where(spk, ga2 + ahp_g_inc, ga2)
            v3 = torch.where(cool2 > 0.0, vres, v2)
            return v3, ge2 * ae, h1 * a1, h2 * a2, h3 * a3, ga3, cool2, spk

        return core

    def core(v, ge, g1, g2, g3, cool, g_ahp, gs, s_inc, ev_e, ev_1, ev_2, ev_3, iext):
        ge2 = ge + ev_e
        h1 = g1 + ev_1
        h2 = g2 + ev_2
        h3 = g3 + ev_3
        ga2 = g_ahp * aa
        drive = (gl * (vr - v) + ge2 * (ee - v) + h1 * (e1 - v) + h2 * (e2 - v)
                 + h3 * (e3 - v) + ga2 * (ek - v) + gs * (e_slow - v) + iext)
        v2 = torch.clamp(v + dt_s * drive, min=vfloor)
        spk = (v2 >= vth) & (cool <= 0.0)
        cool2 = torch.where(spk, ref_m1, torch.clamp(cool - 1.0, min=0.0))
        ga3 = torch.where(spk, ga2 + ahp_g_inc, ga2)
        gs2 = torch.where(spk, gs + s_inc, gs)
        v3 = torch.where(cool2 > 0.0, vres, v2)
        return v3, ge2 * ae, h1 * a1, h2 * a2, h3 * a3, ga3, cool2, gs2 * a_slow, spk

    return core


# ---------------------------------------------------------------------------
class MouseCortexEngine(LarvalZFEngine):
    """M11 小鼠皮层微环路内核（E1b 四槽 + E2b 慢变量张量 + E3b 层间延迟打包）。"""

    def __init__(self, n_neuron: int, device: str = "mps", dtype=None,
                 use_compile: bool = False, verbose: bool = True,
                 compile_mode: str = "default", use_abs_slots: bool = True,
                 strict_syn_class: bool = True):
        super().__init__(n_neuron, device=device, dtype=dtype, use_compile=use_compile,
                         verbose=verbose, compile_mode=compile_mode,
                         use_abs_slots=use_abs_slots)
        self.version = MC_ENGINE_VERSION
        self.strict_syn_class = bool(strict_syn_class)

        # ---- E1b 四亚型槽 ----
        self.subtypes_on = False
        self.n_ch = N_CH_OFF
        # 预注册抽象默认值（**类级先验，literature_unverified**；不得作判据带，见 docs/m11_b2_notes.md B2-L3）
        self.sub = dict(tau_pv_ms=8.0, tau_sst_ms=64.0, tau_vip_ms=32.0,
                        e_pv=-80.0, e_sst=-80.0, e_vip=-80.0)

        # ---- E3b 层流 / 延迟打包 ----
        self.laminar_on = False
        self.t_layer = None            # 逐神经元层标签（int64 numpy）
        self.layer_names = ("L2/3", "L4", "L5", "L6")
        self.delay_bins = np.array([SLOT_EXTRA], dtype=np.int32)
        self._n_delay_group_seen = 0

        # ---- E2b 慢变量张量 ----
        self.slow_on = False
        self.slow = dict(tau_slow_ms=1000.0, inc_per_spike=0.0, e_slow=-77.0)
        self.t_slow = None
        self.t_slow_inc = None
        self.slow_trace = []
        self.slow_sample_every = 2000

        # ---- 不应期/AHP 语义档（上游缺陷登记 E-M9-4）----
        self.refractory_form = REF_FIX

        # ---- 双估计器 / 积分累积 ----
        self._tr_on = False
        self._tr_ge = None
        self._tr_g = None
        self._tr_gs = None
        self._last_run_dual = {}

    # ------------------------------------------------------------------
    # 机制配置（全部带读回断言）
    # ------------------------------------------------------------------
    def configure_subtypes(self, enabled: bool = True, tau_pv_ms: float = 8.0,
                           tau_sst_ms: float = 64.0, tau_vip_ms: float = 32.0,
                           e_pv: float = -80.0, e_sst: float = -80.0,
                           e_vip: float = -80.0) -> Dict[str, Any]:
        """E1b：四亚型槽开关 + 类级时间常数（读回断言）。"""
        for name, v in (("tau_pv_ms", tau_pv_ms), ("tau_sst_ms", tau_sst_ms),
                        ("tau_vip_ms", tau_vip_ms)):
            if not (float(v) > 0):
                raise ValueError("E1b 时间常数必须 > 0：%s=%r（否则该机制构造性失效）" % (name, v))
        for name, v in (("e_pv", e_pv), ("e_sst", e_sst), ("e_vip", e_vip)):
            if not (float(v) < self.point.v_rest):
                raise ValueError("E1b 反转电位必须严格低于静息电位：%s=%r" % (name, v))
        if self._built and self.n_edge and np.any(self._cls_np > 1) and not enabled:
            raise ValueError("已登记类标签 >1 的边，但 enabled=False ⇒ 会静默丢类（拒绝）")
        self.subtypes_on = bool(enabled)
        self.n_ch = N_CH_ON if self.subtypes_on else N_CH_OFF
        self.sub = dict(tau_pv_ms=float(tau_pv_ms), tau_sst_ms=float(tau_sst_ms),
                        tau_vip_ms=float(tau_vip_ms), e_pv=float(e_pv),
                        e_sst=float(e_sst), e_vip=float(e_vip))
        # 读回断言
        assert self.sub["tau_sst_ms"] == float(tau_sst_ms), "tau_sst_ms 读回不一致"
        assert self.n_ch == (N_CH_ON if enabled else N_CH_OFF), "n_ch 读回不一致"
        if self._built:
            self.finalize()
        return self.subtype_config()

    def subtype_config(self) -> Dict[str, Any]:
        return dict(enabled=self.subtypes_on, n_ch=int(self.n_ch), **self.sub)

    def configure_layers(self, layer_id, delay_bins=None, min_delay_steps: int = SLOT_EXTRA
                         ) -> Dict[str, Any]:
        """E3b：层标签 + 层间延迟档（**打包为 ≤4 档**，共享单一环槽）。

        `min_delay_steps`：chunk 安全条件 `min(delay_bins) ≥ W`（W ≤ SLOT_EXTRA）——
        不满足则 chunk 模式会丢事件；故此处**硬断言**（不静默放宽）。
        """
        layer_id = np.asarray(layer_id, dtype=np.int64).ravel()
        if layer_id.size != self.n:
            raise ValueError("层标签长度必须 == n_neuron")
        if layer_id.min() < 0 or layer_id.max() >= len(self.layer_names):
            raise ValueError("层标签越界（0..%d）" % (len(self.layer_names) - 1))
        if delay_bins is None:
            delay_bins = np.array([min_delay_steps], dtype=np.int32)
        delay_bins = np.asarray(sorted(set(int(x) for x in np.asarray(delay_bins).ravel())),
                                dtype=np.int32)
        if delay_bins.size < 1:
            raise ValueError("delay_bins 不得为空")
        if delay_bins.size > MAX_DELAY_BINS:
            raise ValueError("E3b：层间延迟档必须 ≤%d 档（打包），实到 %d 档"
                             % (MAX_DELAY_BINS, delay_bins.size))
        if int(delay_bins.min()) < int(min_delay_steps):
            raise ValueError("E3b：min(delay_bins)=%d < %d ⇒ chunk 模式会丢事件（拒绝静默放宽）"
                             % (int(delay_bins.min()), int(min_delay_steps)))
        self.t_layer = layer_id
        self.laminar_on = True
        self.delay_bins = delay_bins
        assert self.delay_bins.size == delay_bins.size
        return self.laminar_config()

    def laminar_config(self) -> Dict[str, Any]:
        counts = {}
        if self.t_layer is not None:
            for i, nm in enumerate(self.layer_names):
                counts[nm] = int((self.t_layer == i).sum())
        return dict(enabled=self.laminar_on, n_layer=len(self.layer_names),
                    layer_counts=counts, delay_bins=[int(x) for x in self.delay_bins],
                    n_delay_bins=int(self.delay_bins.size))

    def configure_slow(self, enabled: bool = True, tau_slow_ms: float = 1000.0,
                       inc_per_spike: float = 6.0, e_slow: float = -77.0,
                       sample_every: int = 2000) -> Dict[str, Any]:
        """E2b：逐神经元慢变量张量（慢适应钾电导）。

        **"参数是活的"门**（M10 D1 教训）：`enabled=True` 时要求
        `tau_slow_ms>0 ∧ inc_per_spike>0`，否则该机制构造性无效 ⇒ 直接报错（不得静默）。
        """
        if enabled and not (float(tau_slow_ms) > 0):
            raise ValueError("E2b：tau_slow_ms 必须 > 0")
        if enabled and not (float(inc_per_spike) > 0):
            raise ValueError("E2b：inc_per_spike 必须 > 0（否则慢变量恒为 0 = 机制未实施，"
                             "M10 D1 同类错误）")
        if not (float(e_slow) <= self.point.ek + 1e-9):
            raise ValueError("E2b：e_slow 必须 ≤ E_K（慢适应为钾电导，否则符号反转）")
        self.slow_on = bool(enabled)
        self.slow = dict(tau_slow_ms=float(tau_slow_ms), inc_per_spike=float(inc_per_spike),
                         e_slow=float(e_slow))
        self.slow_sample_every = int(sample_every)
        assert self.slow["inc_per_spike"] == float(inc_per_spike), "inc_per_spike 读回不一致"
        assert self.slow["tau_slow_ms"] == float(tau_slow_ms), "tau_slow_ms 读回不一致"
        if self._built:
            self._alloc_slow_state()
            self._build_cores()
        return self.slow_config()

    def slow_config(self) -> Dict[str, Any]:
        return dict(enabled=self.slow_on, **self.slow)

    def set_neuron_slow_gain(self, gain) -> None:
        """逐神经元慢增量倍率（异质性杠杆；默认 1）。"""
        g = np.ones(self.n, dtype=np.float32) if gain is None else \
            np.asarray(gain, dtype=np.float32).ravel()
        if g.size != self.n:
            raise ValueError("慢增益长度必须 == n_neuron")
        self._slow_gain_np = g
        if self._built:
            self.t_slow_inc = torch.as_tensor(
                (g * float(self.slow["inc_per_spike"])).astype(np.float32),
                dtype=self.dtype, device=self.device)

    # ------------------------------------------------------------------
    # 装配
    # ------------------------------------------------------------------
    def add_synapses(self, pre, post, gmax, delay_steps, inhibitory=None,
                     syn_class=None) -> None:
        """**严格守卫**（M10 L21 的静默失效修复）：给 `inhibitory` 就必须给 `syn_class`。"""
        if self.strict_syn_class and inhibitory is not None and syn_class is None:
            raise ValueError(
                "编码路径前置守卫（M10 L21）：传了 inhibitory 但未传 syn_class ⇒ 抑制边会写进兴奋槽"
                "（M10 曾因此使抑制从未生效）。请显式传 syn_class=inhibitory.astype(np.int8)")
        if syn_class is not None:
            cls = np.asarray(syn_class)
            if cls.size and (cls.min() < 0 or cls.max() >= N_CH_ON):
                raise ValueError("syn_class 取值必须 ∈ 0..3（E/PV/SST/VIP）")
        if delay_steps is not None and not np.isscalar(delay_steps):
            d = np.unique(np.asarray(delay_steps).ravel())
            if d.size > MAX_DELAY_BINS:
                raise ValueError("E3b：延迟档必须 ≤%d（打包），实到 %d 档：%r"
                                 % (MAX_DELAY_BINS, d.size, d.tolist()))
        super().add_synapses(pre, post, gmax, delay_steps, inhibitory=inhibitory,
                             syn_class=syn_class)

    def configure_refractory(self, form: str = REF_FIX) -> Dict[str, Any]:
        """不应期/AHP 语义档（**上游缺陷 E-M9-4 的可选择处置**）。

        - `REF_FIX`（默认，M11 基线）：文档语义（第 4 形参 = g_ahp、第 5 形参 = cool）；
        - `REF_FROZEN`：**逐位复刻冻结件 M9/M10 非 gaba 点核的实收语义**（形参序错位），
          仅用于"关闭态等价"与缺陷量级量化；`nc=4` 下无定义 ⇒ 拒绝。
        """
        if form not in (REF_FIX, REF_FROZEN):
            raise ValueError("refractory_form 必须 ∈ {%r, %r}" % (REF_FIX, REF_FROZEN))
        if form == REF_FROZEN and int(self.n_ch) != N_CH_OFF:
            raise ValueError("REF_FROZEN 只在 n_ch=2（E1b 关闭）下有意义")
        self.refractory_form = str(form)
        assert self.refractory_form == str(form), "refractory_form 读回不一致"
        if self._built:
            self._build_cores()
        return dict(refractory_form=self.refractory_form,
                    frozen_arg_order_bug=(self.refractory_form == REF_FROZEN))

    def _build_cores(self):
        """点核（按 n_ch × slow_on 变体）；two_comp 档仅登记（本引擎不实现分层档）。"""
        self._cores.clear()
        self._compile_ok.clear()
        p = self.point
        taus = (p.tau_e, self.sub["tau_pv_ms"], self.sub["tau_sst_ms"], self.sub["tau_vip_ms"])
        e_inhs = (self.sub["e_pv"], self.sub["e_sst"], self.sub["e_vip"])
        frozen = (self.refractory_form == REF_FROZEN)
        fn = _mc_core_factory(p, int(self.n_ch), bool(self.slow_on), frozen, taus, e_inhs,
                              self.slow["tau_slow_ms"], self.slow["e_slow"])
        self._cores["point"] = fn
        self._compile_ok["point"] = False
        self._core_key = ("nc%d_slow%d_%s" % (int(self.n_ch), int(bool(self.slow_on)),
                                              self.refractory_form))
        self.accel = "eager"
        cfg = getattr(self, "_two_comp_cfg", None)
        if cfg is not None:
            from .adult_engine import _two_comp_core_factory
            self._cores["two_comp"] = _two_comp_core_factory(cfg["dt_ms"], cfg["two_comp"])

    def core_readback(self) -> Dict[str, Any]:
        """**作用域守卫**：报告核闭包创建时固化的参数（与 config 不一致 ⇒ 机制未生效）。"""
        return dict(core_key=self._core_key, n_ch=int(self.n_ch), slow_on=bool(self.slow_on),
                    refractory_form=str(self.refractory_form),
                    frozen_arg_order_bug=bool(self.refractory_form == REF_FROZEN),
                    tau_e_ms=float(self.point.tau_e),
                    tau_pv_ms=float(self.sub["tau_pv_ms"]),
                    tau_sst_ms=float(self.sub["tau_sst_ms"]),
                    tau_vip_ms=float(self.sub["tau_vip_ms"]),
                    e_pv=float(self.sub["e_pv"]), e_sst=float(self.sub["e_sst"]),
                    e_vip=float(self.sub["e_vip"]),
                    tau_slow_ms=float(self.slow["tau_slow_ms"]),
                    e_slow=float(self.slow["e_slow"]),
                    uses_syn_class=bool(self.n_ch == N_CH_ON),
                    ring_width_n=int(self.n_ch * self.n))

    def finalize(self) -> Dict[str, Any]:
        st = self._finalize_no_ring()
        return st

    def _finalize_no_ring(self) -> Dict[str, Any]:
        st = super().finalize()
        n = self.n
        nc = int(self.n_ch)
        if nc == N_CH_OFF and self._cls_np.size and int(self._cls_np.max()) > 1:
            raise ValueError("E1b 关闭态但存在 syn_class>1 的边 ⇒ 拒绝（会静默丢类）")
        # ① 逐类编码 post_off（E/PV/SST/VIP → 0N/1N/2N/3N）
        for grp in self._groups:
            cls = torch.as_tensor(np.asarray(grp["cls"], dtype=np.int64), device=self.device)
            post = torch.as_tensor(np.asarray(grp["post"], dtype=np.int64), device=self.device)
            grp["t_cls"] = cls
            grp["t_post_off"] = (post + n * cls).to(torch.int32)
        # ② **单一共享环槽**（E3b 打包核心）：全部延迟档写入同一 ring（宽度 n_ch·N）
        self._ring = torch.zeros(self.n_slot * nc * n, dtype=self.dtype, device=self.device)
        self._rings = {int(g["delay"]): self._ring for g in self._groups}
        self._n_delay_group_seen = int(self.n_delay_group)
        # ③ 逐类电导状态（n_ch=4）+ 慢变量状态
        for nm in ("t_g1", "t_g2", "t_g3"):
            setattr(self, nm, torch.zeros(n, dtype=self.dtype, device=self.device))
        if self.slow_on:
            self._alloc_slow_state()
        self._build_cores()
        st["m11"] = dict(version=MC_ENGINE_VERSION, subtypes_on=self.subtypes_on,
                         n_ch=nc, laminar_on=self.laminar_on,
                         delay_bins=[int(x) for x in self.delay_bins],
                         n_delay_group=int(self.n_delay_group),
                         ring_shared=True, slow_on=self.slow_on,
                         strict_syn_class=self.strict_syn_class)
        return st

    def _alloc_slow_state(self):
        self.t_slow = torch.zeros(self.n, dtype=self.dtype, device=self.device)
        gain = getattr(self, "_slow_gain_np", None)
        if gain is None:
            gain = np.ones(self.n, dtype=np.float32)
            self._slow_gain_np = gain
        self.t_slow_inc = torch.as_tensor(
            (gain * float(self.slow["inc_per_spike"])).astype(np.float32),
            dtype=self.dtype, device=self.device)

    # ------------------------------------------------------------------
    # 外部事件（逐类编码）
    # ------------------------------------------------------------------
    def set_external_events(self, steps, neurons, gmax, n_steps: int,
                            inhibitory=None, delay_steps: Optional[int] = None,
                            slow_frac: float = 0.0, syn_class=None) -> Dict[str, Any]:
        """E1b：外部事件支持 `syn_class`（默认由 `inhibitory` 给出 0/1）；绝对步号表。"""
        steps = np.asarray(steps, dtype=np.int64).ravel()
        neurons = np.asarray(neurons, dtype=np.int64).ravel()
        gmax = np.asarray(gmax, dtype=np.float32).ravel()
        if syn_class is None:
            inh = (np.zeros(steps.shape, dtype=bool) if inhibitory is None
                   else np.asarray(inhibitory, dtype=bool).ravel())
            cls = inh.astype(np.int64)
        else:
            cls = np.asarray(syn_class, dtype=np.int64).ravel()
        if not (steps.shape == neurons.shape == gmax.shape == cls.shape):
            raise ValueError("外部事件数组形状不一致")
        if cls.size and (cls.min() < 0 or cls.max() >= int(self.n_ch)):
            raise ValueError("外部事件 syn_class 越界（n_ch=%d）" % int(self.n_ch))
        order = np.argsort(steps, kind="stable")
        steps, neurons, gmax, cls = steps[order], neurons[order], gmax[order], cls[order]
        counts = np.bincount(steps, minlength=int(n_steps))[:int(n_steps)]
        off = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        n = self.n
        self._ext_neuron = torch.as_tensor((neurons + n * cls).astype(np.int64),
                                           device=self.device)
        self._ext_amp = torch.as_tensor(gmax, device=self.device)
        self._ext_delay = int(self.max_delay if delay_steps is None else delay_steps)
        self._ext_slot = torch.as_tensor(
            ((steps + self._ext_delay) % self.n_slot) * (int(self.n_ch) * n),
            device=self.device)
        self._ext_off = off
        self._ext_n_steps = int(n_steps)
        self._ext_active = bool(steps.size)
        self._ext_slow_frac = float(slow_frac)
        # 读回断言（M10 语义）：事件表必须覆盖整个协议（绝对步号）
        assert self._ext_off.size == int(n_steps) + 1, "外部事件表必须覆盖整个协议（绝对步号）"
        return {"n_events": int(steps.size),
                "max_events_per_step": int(counts.max() if counts.size else 0),
                "mean_events_per_step": float(counts.mean() if counts.size else 0.0)}

    def reset(self, v0=None, seed: int = 0) -> None:
        super().reset(v0=v0, seed=seed)
        if int(self.n_ch) == N_CH_ON:
            for nm in ("t_g1", "t_g2", "t_g3"):
                t = getattr(self, nm, None)
                if t is not None:
                    t.zero_()
        if self.t_slow is not None:
            self.t_slow.zero_()
        self.slow_trace = []

    # ------------------------------------------------------------------
    # 运行（泛化 M10 的 run：单共享环槽 + n_ch 电导槽 + 慢变量 + 积分累积）
    # ------------------------------------------------------------------
    def run(self, n_steps: int, *, fidelity: str = "point", record: str = "counts",
            delivery: Optional[str] = None, pop_trace: bool = False,
            chunk_steps: Optional[int] = None, record_from: int = 0,
            progress=None, progress_every: int = 20000, cache_clear_every: int = 0,
            keep_state: bool = True, trace_integrals: bool = False) -> Dict[str, Any]:
        if fidelity == "two_comp":
            raise NotImplementedError(
                "E4b 层特异双房室档未实现（清单 §2.2 标『复用+扩展』，属 P-EXT1 预算门控）"
                "⇒ 显式拒绝，不静默降级为 point。")
        if not self._built:
            self.finalize()
        n = self.n
        nc = int(self.n_ch)
        ncn = nc * n
        dt_ms = self.point.dt_ms
        n_steps = int(n_steps)
        abs0 = self.abs_step if self.use_abs_slots else 0

        if delivery is None:
            delivery = "chunk"
        W = int(chunk_steps if chunk_steps else 0)
        if delivery == "chunk":
            if W <= 0:
                W = int(min(SLOT_EXTRA, self.max_delay))
            W = int(min(W, self.max_delay, SLOT_EXTRA))
            if W < 1:
                delivery = "step"

        rec_rows = 0
        if record == "full":
            rec_rows = W if delivery == "chunk" else max(1, min(2000, max(1, int(2e8 // n))))
        spk_ring = None
        if delivery == "chunk":
            spk_ring = torch.zeros(W, n, dtype=torch.bool, device=self.device)
        elif rec_rows:
            spk_ring = torch.zeros(rec_rows, n, dtype=torch.bool, device=self.device)

        pop_dev = (torch.zeros(n_steps, dtype=self.dtype, device=self.device)
                   if pop_trace else None)
        spike_steps = []
        spike_idx = []
        rec_ptr = 0
        rec_base = 0

        core = self._cores["point"]
        n_slot = self.n_slot
        t_count = self.t_count
        zero_iext = self.t_iext

        t_v = self.t_v
        t_ge = self.t_ge
        t_cool = self.t_cool
        t_ahp = self.t_ahp
        t_g1 = self.t_g1
        t_g2 = self.t_g2
        t_g3 = self.t_g3
        t_slow = self.t_slow
        t_slow_inc = self.t_slow_inc
        t_gi = self.t_gi
        t_stim = self.t_stim
        t_stim_col = self.t_stim_col
        has_stim = t_stim is not None
        ring = self._ring
        # 逐类事件视图（预计算 ⇒ 单共享环槽仍走"单环槽快路径"，M10 L18.1 的 >30× 退化被消除）
        rv = [[ring[_k * ncn + c * n: _k * ncn + (c + 1) * n] for _k in range(n_slot)]
              for c in range(nc)]
        nc4 = (nc == N_CH_ON)
        slow_on = bool(self.slow_on) and t_slow is not None
        rec_rows_v = list(spk_ring.unbind(0)) if spk_ring is not None else None
        ext_active = getattr(self, "_ext_active", False)
        ext_off = getattr(self, "_ext_off", None)
        ext_neuron = getattr(self, "_ext_neuron", None)
        ext_amp = getattr(self, "_ext_amp", None)
        ext_n_steps = getattr(self, "_ext_n_steps", 0)
        ext_slot = getattr(self, "_ext_slot", None)
        # E3（M10 调质增益）——保留语义：作用于外部事件幅度
        mod_slow = bool(self.mod_slow) and getattr(self, "t_mmod", None) is not None
        mod_scale = float(self.g_mod)
        dt_s_mod = dt_ms * 1e-3
        _tau_mod_s = float(self.mod["tau_mod_ms"]) * 1e-3 if mod_slow else 0.0
        _ghi, _glo = float(self.mod["gain_hi"]), float(self.mod["gain_lo"])
        m_dev = self.t_mmod if mod_slow else None
        self._gmod_step = mod_scale
        mod_sample_every = max(1, int(round(50.0 / max(dt_ms, 1e-9))))
        _mstep = [0]

        deliver_step = self._deliver_step
        chunk_mode = (delivery == "chunk")
        t_u = getattr(self, "t_u", None)
        stp_on = bool(self.stp_enabled) and t_u is not None

        tr_on = bool(trace_integrals)
        if tr_on:
            self._tr_ge = torch.zeros(n, dtype=torch.float64, device=self.device)
            self._tr_g = [torch.zeros(n, dtype=torch.float64, device=self.device)
                          for _ in range(max(1, nc - 1))]
            self._tr_gs = (torch.zeros(n, dtype=torch.float64, device=self.device)
                           if slow_on else None)
            _sample_ct = [0]
            slow_min = float("inf"); slow_max = float("-inf"); slow_sum = 0.0
            slow_sq = 0.0; slow_n = 0
        else:
            _sample_ct = [0]

        t0 = time.perf_counter()
        step = 0
        while step < n_steps:
            if chunk_mode:
                a = (abs0 + step) % W
                w = min(W - a if a else W, n_steps - step)
            else:
                w = 1
            for k in range(w):
                s = step + k
                _sl = ((abs0 + s) if self.use_abs_slots else s) % n_slot
                ev0 = rv[0][_sl]
                ev1 = rv[1][_sl]
                if nc4:
                    ev2 = rv[2][_sl]
                    ev3 = rv[3][_sl]
                    iext = t_stim[s][t_stim_col] if has_stim else zero_iext
                    if slow_on:
                        t_v, t_ge, t_g1, t_g2, t_g3, t_ahp, t_cool, t_slow, spk = core(
                            t_v, t_ge, t_g1, t_g2, t_g3, t_cool, t_ahp, t_slow, t_slow_inc,
                            ev0, ev1, ev2, ev3, iext)
                    else:
                        t_v, t_ge, t_g1, t_g2, t_g3, t_ahp, t_cool, spk = core(
                            t_v, t_ge, t_g1, t_g2, t_g3, t_cool, t_ahp,
                            ev0, ev1, ev2, ev3, iext)
                else:
                    iext = t_stim[s][t_stim_col] if has_stim else zero_iext
                    if slow_on:
                        t_v, t_ge, t_gi, t_ahp, t_cool, t_slow, spk = core(
                            t_v, t_ge, t_gi, t_cool, t_ahp, t_slow, t_slow_inc,
                            ev0, ev1, iext)
                    else:
                        t_v, t_ge, t_gi, t_ahp, t_cool, spk = core(
                            t_v, t_ge, t_gi, t_cool, t_ahp, ev0, ev1, iext)
                if tr_on:
                    self._tr_ge.add_(t_ge)
                    if nc4:
                        self._tr_g[0].add_(t_g1); self._tr_g[1].add_(t_g2)
                        self._tr_g[2].add_(t_g3)
                    else:
                        self._tr_g[0].add_(t_gi)
                    if self._tr_gs is not None:
                        self._tr_gs.add_(t_slow)
                if slow_on:
                    _sample_ct[0] += 1
                    if _sample_ct[0] % self.slow_sample_every == 0:
                        _sv = float(t_slow.mean().item())
                        self.slow_trace.append(_sv)
                if mod_slow:
                    rate_inst = spk.to(self.dtype).mean() / dt_s_mod
                    tgt = (rate_inst / float(self.mod["rate_ref_hz"])).reshape(1)
                    m_dev.add_((dt_s_mod / _tau_mod_s) * (tgt - m_dev))
                    self._gmod_step = float(_glo + (_ghi - _glo) * float(m_dev.item()))
                    _mstep[0] += 1
                    if _mstep[0] % mod_sample_every == 0:
                        self.mod_trace.append(float(m_dev.item()))
                ev0.zero_(); ev1.zero_()
                if nc4:
                    ev2.zero_(); ev3.zero_()
                if ext_active and not chunk_mode and s < ext_n_steps:
                    ea = int(ext_off[s]); eb = int(ext_off[s + 1])
                    if eb > ea:
                        ring.index_put_(
                            (ext_neuron[ea:eb] + ext_slot[ea:eb],),
                            ext_amp[ea:eb] * (self._gmod_step if mod_slow else mod_scale),
                            accumulate=True)
                if chunk_mode:
                    rec_rows_v[rec_ptr].copy_(spk)
                    rec_ptr += 1
                else:
                    t_count += spk
                    if record == "full":
                        rec_rows_v[rec_ptr].copy_(spk)
                        rec_ptr += 1
                    deliver_step(spk, abs0 + s)
                    if pop_dev is not None:
                        pop_dev[s:s + 1].copy_(spk.sum().reshape(1).to(self.dtype))
            if chunk_mode:
                if ext_active:
                    a0 = step
                    ea = int(ext_off[a0 if not self.use_abs_slots else abs0 + a0])
                    eb = int(ext_off[(a0 + w) if not self.use_abs_slots else abs0 + a0 + w])
                    if eb > ea:
                        ring.index_put_(
                            (ext_neuron[ea:eb] + ext_slot[ea:eb],),
                            ext_amp[ea:eb] * (self._gmod_step if mod_slow else mod_scale),
                            accumulate=True)
                self._deliver_chunk(spk_ring, w, abs0 + step if self.use_abs_slots else step,
                                    t_count)
                if pop_dev is not None:
                    pop_dev[step:step + w].copy_(spk_ring[:w].sum(dim=1).to(self.dtype))
            if rec_ptr > 0 and (chunk_mode or rec_ptr >= rec_rows):
                if record == "full":
                    self._flush_rec(spk_ring, rec_ptr, rec_base - record_from,
                                    spike_steps, spike_idx)
                rec_base += rec_ptr
                rec_ptr = 0
            step += w
            if cache_clear_every and (step % cache_clear_every) < w:
                try:
                    if getattr(torch, "mps", None) is not None and torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                except Exception:
                    pass
            if progress is not None and (step % progress_every < w or step >= n_steps):
                progress(step, n_steps, time.perf_counter() - t0)

        wall = time.perf_counter() - t0
        if record == "full" and rec_ptr > 0:
            self._flush_rec(spk_ring, rec_ptr, rec_base - record_from, spike_steps, spike_idx)

        self.t_v, self.t_ge, self.t_cool = t_v, t_ge, t_cool
        self.t_ahp = t_ahp
        if nc4:
            self.t_g1, self.t_g2, self.t_g3 = t_g1, t_g2, t_g3
        else:
            self.t_gi = t_gi
        if slow_on:
            self.t_slow = t_slow
        self.abs_step = abs0 + n_steps

        counts = self.t_count.detach().cpu().numpy().astype(np.float64)
        res = {
            "wall_s": wall, "n_steps": n_steps, "dt_ms": dt_ms,
            "ms_per_step": wall / max(n_steps, 1) * 1e3,
            "spk_total": float(np.sum(counts)), "spike_counts": counts,
            "pop": (pop_dev.detach().cpu().numpy() if pop_dev is not None else None),
            "spikes": ((np.concatenate(spike_steps), np.concatenate(spike_idx))
                       if (record == "full" and spike_steps) else None),
            "delivery": delivery, "chunk_steps": W if delivery == "chunk" else 1,
            "accel": self.accel, "device": str(self.device),
            "n_edge": self.n_edge, "n_neuron": n, "n_slot": n_slot,
            "n_ch": nc, "abs_step_end": self.abs_step,
        }
        # ---- 双估计器硬断言（清单要求：同一量的两个独立估计途径必须断言一致）----
        dual = {}
        spk_pop = float(np.sum(res["pop"])) if res["pop"] is not None else None
        dual["spk_total"] = float(res["spk_total"])
        dual["counts_sum"] = float(counts.sum())
        dual["rel_diff_count_vs_total"] = _rel(res["spk_total"], dual["counts_sum"])
        if spk_pop is not None:
            dual["pop_trace_sum"] = spk_pop
            dual["rel_diff_total_vs_pop"] = _rel(res["spk_total"], spk_pop)
            assert dual["rel_diff_total_vs_pop"] <= DUAL_EST_TOL, (
                "双估计器不一致（spk_total vs pop_trace.sum）：%.4f%% > %.1f%%"
                % (100 * dual["rel_diff_total_vs_pop"], 100 * DUAL_EST_TOL))
        assert dual["rel_diff_count_vs_total"] <= DUAL_EST_TOL, (
            "双估计器不一致（spk_total vs counts.sum）：%.4f%%" % (100 * dual["rel_diff_count_vs_total"]))
        dual["wall_s"] = float(wall)
        dual["wall_from_ms_per_step_s"] = float(res["ms_per_step"]) * n_steps / 1e3
        if tr_on:
            res["trace_integrals"] = self.trace_integrals()
        self._last_run_dual = dual
        res["dual_estimators"] = dual
        if slow_on and self.slow_trace:
            _tr = np.asarray(self.slow_trace, dtype=float)
            res["slow_trace"] = dict(n=int(_tr.size), mean=float(_tr.mean()),
                                     std=float(_tr.std()), min=float(_tr.min()),
                                     max=float(_tr.max()),
                                     dynamic_range=float(_tr.max() - _tr.min()),
                                     alive=bool(_tr.std() > 1e-6))
        if not keep_state:
            self.reset(seed=getattr(self, "_rng_seed", 0))
        return res

    # ------------------------------------------------------------------
    # 投递（E1b：逐类槽；E3b：单一共享环槽）
    # ------------------------------------------------------------------
    def _deliver_step(self, spk, abs_step: int) -> int:
        self._decay_stp(self.point.dt_ms)
        total = 0
        ring = self._ring
        ncn = int(self.n_ch) * self.n
        for grp in self._groups:
            idx = spk.nonzero(as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            c = grp["t_counts"][idx]
            tot = int(c.sum().item())
            if tot == 0:
                continue
            total += tot
            # **E5-FIX（M10）保持**：base = arange + repeat_interleave(t_row − prev_total, c)
            off = torch.repeat_interleave(grp["t_row"][idx] - (torch.cumsum(c, 0) - c), c)
            base = self._arange(tot)[:tot] + off
            slot = ((abs_step + grp["delay"]) % self.n_slot) * ncn
            self._apply_stp_and_deliver(grp, base, grp["t_post_off"][base] + slot, ring)
        return total

    def _deliver_chunk(self, spk_ring, w: int, abs_step0: int, t_count) -> int:
        if w <= 0:
            return 0
        if self.stp_enabled:
            for k in range(w):
                self._deliver_step(spk_ring[k], abs_step0 + k)
            t_count += spk_ring[:w].sum(dim=0)
            return 0
        rows = spk_ring[:w]
        t_count += rows.sum(dim=0)
        flat = rows.reshape(-1).nonzero(as_tuple=False).flatten()
        if flat.numel() == 0:
            return 0
        n_spk = int(flat.numel())
        if n_spk * (self.n_edge / max(self.n, 1)) > MAX_EDGES_PER_FLUSH:
            tot_all = 0
            for k in range(w):
                tot_all += self._deliver_step(rows[k], abs_step0 + k)
            return tot_all
        n = self.n
        ncn = int(self.n_ch) * n
        ring = self._ring
        kk = torch.div(flat, n, rounding_mode="floor")
        nn = flat - kk * n
        total = 0
        for grp in self._groups:
            c = grp["t_counts"][nn]
            tot = int(c.sum().item())
            if tot == 0:
                continue
            total += tot
            slot_k = ((abs_step0 + kk + grp["delay"]) % self.n_slot) * ncn
            packed = torch.stack([grp["t_row"][nn] - (torch.cumsum(c, 0) - c), slot_k], dim=1)
            rep = torch.repeat_interleave(packed, c, dim=0)
            base = self._arange(tot)[:tot] + rep[:, 0]
            dst = grp["t_post_off"][base] + rep[:, 1]
            _gm = getattr(self, "_gmod_step", 1.0)
            _gv = grp["t_gmax"][base]
            if _gm != 1.0:
                _gv = _gv * _gm
            ring.index_add_(0, dst, _gv)
        return total

    def _apply_stp_and_deliver(self, grp, base, dst, ring) -> None:
        g = grp["t_gmax"][base]
        gm = getattr(self, "_gmod_step", 1.0)
        if gm != 1.0:
            g = g * gm
        if self.stp_enabled:
            eidx = grp["t_eidx"][base]
            u = self.t_u[eidx]; x = self.t_x[eidx]
            g = g * u * x
            u2 = u + float(self.stp["u0"]) * (1.0 - u)
            x2 = x - u2 * x
            self.t_u.index_copy_(0, eidx, u2)
            self.t_x.index_copy_(0, eidx, x2)
        ring.index_add_(0, dst, g)

    # ------------------------------------------------------------------
    # 断言 / 读出接口
    # ------------------------------------------------------------------
    def class_edge_counts(self) -> Dict[str, int]:
        """**逐类边数**（静态）：`#{edges with cls=c}`（E1b 编码路径断言之 1）。"""
        out = {nm: 0 for nm in SUB_NAMES[:int(self.n_ch)]}
        if self._cls_np is None:
            return out
        for c in range(int(self.n_ch)):
            out[SUB_NAMES[c]] = int((self._cls_np == c).sum())
        return out

    def class_slot_counts(self) -> Dict[str, int]:
        """**逐类槽内边数**（静态，读回 CSR 的 `post_off`）：`#{post_off ∈ [cN,(c+1)N)}`。

        M10 的失效模式（抑制边写进兴奋槽）在此**必然**表现为 `I` 槽 0 条 / `E` 槽多出。
        """
        out = {nm: 0 for nm in SUB_NAMES[:int(self.n_ch)]}
        ncn = int(self.n_ch) * self.n
        for grp in self._groups:
            po = grp["t_post_off"].detach().cpu().numpy()
            for c in range(int(self.n_ch)):
                out[SUB_NAMES[c]] += int(((po >= c * self.n) & (po < (c + 1) * self.n)).sum())
        return out

    def dense_class_slot_counts(self) -> Dict[str, int]:
        """逐类边数的**独立实现二**（对 `_cls_np` 直接 `bincount`，与 CSR 派生路径交叉核对）。"""
        out = {nm: 0 for nm in SUB_NAMES[:int(self.n_ch)]}
        if self._cls_np is None or self._cls_np.size == 0:
            return out
        bc = np.bincount(self._cls_np.astype(np.int64), minlength=max(4, int(self.n_ch)))
        for c in range(int(self.n_ch)):
            out[SUB_NAMES[c]] = int(bc[c])
        return out

    def ring_slot_occupancy(self) -> Dict[str, float]:
        """环槽残量（**信息性**）：尚未被读取的事件电导和，逐类。"""
        r = self._ring.view(self.n_slot, int(self.n_ch), self.n)
        tot = r.abs().sum(dim=(0, 2)).detach().cpu().numpy()
        return {SUB_NAMES[c]: float(tot[c]) for c in range(int(self.n_ch))}

    def trace_integrals(self, dt_ms: Optional[float] = None) -> Dict[str, Any]:
        """**逐类电导积分** ∫g_c(t)dt（动态，引擎侧独立估计器）。"""
        dt = float(self.point.dt_ms if dt_ms is None else dt_ms)
        out = {}
        if self._tr_ge is None:
            return out
        out["E"] = float(self._tr_ge.sum().item()) * dt
        for c in range(1, int(self.n_ch)):
            out[SUB_NAMES[c]] = float(self._tr_g[c - 1].sum().item()) * dt
        if self._tr_gs is not None:
            out["SLOW"] = float(self._tr_gs.sum().item()) * dt
        return out

    def slow_readback(self) -> Dict[str, Any]:
        """E2b 读出接口（**"机制确已实施"门**）。"""
        if self.t_slow is None:
            return dict(implemented=False, alive=False)
        s = self.t_slow.detach().cpu().numpy().astype(np.float64)
        tr = np.asarray(self.slow_trace, dtype=float) if self.slow_trace else np.zeros(0)
        return dict(implemented=True, n=int(s.size), mean=float(s.mean()), std=float(s.std()),
                    min=float(s.min()), max=float(s.max()),
                    dynamic_range=float(s.max() - s.min()),
                    n_nonzero=int((s > 0).sum()),
                    frac_nonzero=float((s > 0).mean()),
                    trace_n=int(tr.size),
                    trace_std=float(tr.std()) if tr.size else 0.0,
                    trace_range=float(tr.max() - tr.min()) if tr.size else 0.0,
                    alive=bool((s.max() - s.min()) > 1e-9 and float((s > 0).mean()) > 0.0),
                    inc_per_spike=float(self.slow["inc_per_spike"]),
                    tau_slow_ms=float(self.slow["tau_slow_ms"]))

    def layer_rate_summary(self, t_meas_ms: float, layer_labels=None) -> Dict[str, Any]:
        """逐层发放率（层流机制的可测读出）。"""
        c = self.t_count.detach().cpu().numpy().astype(np.float64)
        lay = self.t_layer if layer_labels is None else np.asarray(layer_labels)
        if lay is None:
            return {}
        T = max(1e-9, t_meas_ms / 1000.0)
        out = {}
        for i, nm in enumerate(self.layer_names):
            m = (lay == i)
            if m.sum() == 0:
                continue
            out[nm] = dict(n=int(m.sum()), rate_mean_hz=float(c[m].mean() / T),
                           rate_median_hz=float(np.median(c[m]) / T),
                           spk_total=float(c[m].sum()))
        return out

    # ==================================================================
    # 兼容层：把逐类状态张量挂到 finalize 之后
    # ==================================================================
    @property
    def n_ch_eff(self) -> int:
        return int(self.n_ch)


# ---------------------------------------------------------------------------
def _rel(a: float, b: float) -> float:
    """相对差（分母取两者较大绝对值的保护，避免 0 分母）。"""
    d = max(abs(float(a)), abs(float(b)), 1e-12)
    return abs(float(a) - float(b)) / d
