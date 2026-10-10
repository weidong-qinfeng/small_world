"""M9 全规模向量化引擎（torch / Apple MPS）——B2 节点交付 ①（清单 §2）。

《生物仿真M9实施清单》§2.3（GPU 正确性验证）/§2.4（point-neuron 为主）/§2.5（引擎冻结规则：
kernel 改动必须重过对齐探针）/§3.5.1（全规模装配）/§0.9 R10（探针先行 → 裁决 → 再烧）。

语义基线 = `tools/scan_m9_engine.py::run_gpu`（G0 门 100% 对齐的参考实现，L15.2/L15.3）。
本模块在**保持其全部语义**的前提下做 5 项向量化优化（L19.2 执行层优化要求）：

  ① **事件交付向量化**：CSR-by-pre（按 pre 稳定排序的出边表）+ 活跃集压缩 + GPU
     `index_add_` scatter-add，替代 Python `O(spikes × out-degree)` 循环。两种投递调度：
       - `delivery="step"`  每步压缩 + 投递（逐步语义＝参考实现语义）；
       - `delivery="chunk"` 每 W 步压缩一次批量投递（要求 W ≤ 延迟步数，语义等价）——
         把 host 同步 / `nonzero` / `repeat_interleave` 等**固定开销**摊薄 W 倍。
     实测 MPS 固定开销（本机 M1 Pro，min-of-N）：`nonzero`(139k) ≈1.0ms、
     `repeat_interleave` ≈0.73ms（与长度无关）、`index_add_` 15.1M ≈8.6ms → 必须摊薄。
  ② **算子融合**：状态更新经 `torch.compile`（inductor/MPS）融合（实测 8 op 链
     0.564 → 0.094 ms/步，6×）。
  ③ **消除每步同步**：不应期改用**倒计时** `cool`（与 `s >= next_allowed` 等价，推演见
     `_point_core_factory`），免传 step；记录/spike 计数全程设备侧累积。
  ④ **消除 Python 侧开销**：张量全部预绑定为实例属性；step 循环内无字典查找/字符串拼接。
  ⑤ **稀疏事件驱动**：只处理 spiking 神经元的出边（静默 81.67% → 每步活跃边 ~1% 全量）。

语义保持（L15.3 实测，**不得改动**）：
  - 事件在**步起点**生效（先加事件再更新状态）；全变量**步起点同步估值**；
  - spike 在状态更新后检测、**时间记步骤起点**、不应期自检测步终点起算；无 v 重置（HH 档）；
  - 事件**按 post 隔室索引**交付；轴耦合 v 相关项入 `A_v`、常数项入 `B_v`。

环形槽说明（**关键**）：槽数 `n_slot = max_delay + SLOT_EXTRA + 1`（参考实现为
`max_delay = max(delay)+1`）。放宽槽数对语义**无影响**（事件写槽 `(s+d)%R` 在 `d<R` 时
首次被读即为步 `s+d`），但为 chunk 模式所必需：批量写发生在 chunk 末步之后，要求
`d ≥ W` 且 `R ≥ d + W`（否则事件会被本 chunk 已读并清零的槽吞掉）。

公共 API：
  - `AdultEngine(n_neuron, device="mps")` → `add_synapses(...)` → `finalize()`
    → `reset()` → `run(n_steps, ...)`；
  - `configure_two_comp(ex)`：由 `scan_m9_engine.extract_network` 提取字典配置对齐档；
  - `scale_synapses(mask, factor)`：消融（P5–P9 复用；拓扑不变）；
  - `firing_stats(t_meas_ms)`：P4 静息判据量。

用法（探针纪律：先小窗冒烟 → 再全规模）：
    PYTHONHASHSEED=0 MPLBACKEND=Agg .venv-m9/bin/python -m neural_exploration.tools.scan_m9_engine_vec
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:  # 引擎为可选重型依赖（判据脚手架不 import 本模块；M8 L5 惯例）
    import torch
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None
    TORCH_AVAILABLE = False

M9_ENGINE_VERSION = "m9e-2.0-vec"

# ---------------------------------------------------------------------------
# two_comp（HH）档常量——与 tools/scan_m9_engine.py 逐字一致（对齐基线，勿改）
# ---------------------------------------------------------------------------
TH_V = -20.0e-3
REF_MS = 2.0
CM_SI = 1e-2
EL_V, ENA_V, EK_V = -54.4e-3, 50.0e-3, -77.0e-3
E_GABA_V = -70.0e-3
G_AX_S = 5.4e-9
TAU_AMPA_S, TAU_GABA_S = 3.0e-3, 5.0e-3

SLOT_EXTRA = 40          # 环形槽余量（chunk 模式 W ≤ SLOT_EXTRA）
MAX_EDGES_PER_FLUSH = 4_000_000   # 单次批量投递的边数上限（超限退化为逐步，防 OOM）


# ---------------------------------------------------------------------------
# point 档（§2.4：M9 全规模唯一可行保真度）参数
# ---------------------------------------------------------------------------
@dataclass
class PointParams:
    """conductance-based LIF（CUBA 类）点神经元参数（类级定稿 ≤40 全局参数，§8 ①）。"""
    v_rest: float = -52.0      # mV
    v_th: float = -45.0        # mV
    v_reset: float = -55.0     # mV
    tau_m: float = 20.0        # ms
    tau_e: float = 2.0         # ms（AMPA/cholinergic）
    tau_i: float = 5.0         # ms（GABA）
    e_exc: float = 0.0         # mV
    e_inh: float = -80.0       # mV（GABA_A 反转电位；预注册：主agent 裁决 2026-10-10
                               #  ——须严格负于 E_K=-77 mV，使静息与 AHP 期间均满足
                               #  v > E_GABA ⇒ 抑制边恒为抑制）
    ref_ms: float = 2.0        # ms 绝对不应期
    dt_ms: float = 0.05        # ms（与 two_comp 档一致，M8 FIDELITY_DT 定稿）
    v_floor: float = -85.0     # mV（**纯数值**保护；低于全部反转电位 → 正常运行不触发）
    ek: float = -77.0          # mV 钾平衡电位（AHP 电导的反转电位）
    ahp_tau_ms: float = 700.0  # ms AHP 时间常数（0 = 关闭）
    ahp_g_inc: float = 0.0     # 1/s 每 spike 的 AHP **电导**增量（ahp_form="conductance"）
    ahp_inc: float = 0.0       # mV/s 每 spike 的 AHP **恒流**增量（ahp_form="current"，legacy 复现用）
    ahp_form: str = "conductance"   # "conductance"（物理修复档）| "current"（v1 legacy 档）

    def as_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self.__dict__.items()}


def _point_core_factory(p: PointParams):
    """point 档状态更新核（含**物理自洽的 AHP 钾电导**）。

    AHP 修复（主agent 裁决 2026-10-10；替代 v1 的恒流注入）：
        旧：`drive -= a_ahp`（a 为 mV/s 恒流）→ 平衡点 `v_rest − a/g_L` 可达 **−102 mV**，
            低于 E_K（−77）与 E_GABA（−75）→ **GABA 驱动项符号反转**（L25 机制定位）。
        新：`I_ahp = g_ahp·(E_K − v)`（钾电导，τ_ahp 动力学；spike 时 g_ahp += ahp_g_inc）
            → v 被 E_K **硬性限制**在 −77 mV 之上，永不越过 E_GABA（−80）→ 抑制边恒为抑制。

    不应期倒计时等价性（不变）：spike 于步 s → `cool = ref_steps − 1`；步 s+k 判定用
    `cool = ref − k`；阻塞 ⟺ s+k < s+ref（与参考实现一致）。
    v 用显式 Euler（dt=0.05 ms / τ_m=20 ms → dt/τ=0.0025，精度充分且算子数最少）。
    """
    dt_s = p.dt_ms * 1e-3
    ae = math.exp(-p.dt_ms / p.tau_e)
    ai = math.exp(-p.dt_ms / p.tau_i)
    gl = 1.0 / (p.tau_m * 1e-3)
    vr, vth, vres = p.v_rest, p.v_th, p.v_reset
    ee, ei, ek, vfloor = p.e_exc, p.e_inh, p.ek, p.v_floor
    ref_m1 = float(max(p.ref_ms / p.dt_ms - 1.0, 0.0))
    aa = (math.exp(-p.dt_ms / p.ahp_tau_ms) if p.ahp_tau_ms and p.ahp_tau_ms > 0
          else 0.0)
    ahp_g_inc = float(p.ahp_g_inc)

    legacy = (str(p.ahp_form) == "current")
    ahp_inc_c = float(p.ahp_inc)

    if legacy:
        def core(v, ge, gi, a_ahp, cool, ev_e, ev_i, iext):
            ge2 = ge + ev_e
            gi2 = gi + ev_i
            a2 = a_ahp * aa                   # legacy：AHP 为**恒流**(mV/s) 衰减
            drive = (gl * (vr - v) + ge2 * (ee - v) + gi2 * (ei - v)
                     - a2 + iext)
            v2 = torch.clamp(v + dt_s * drive, min=vfloor)
            spk = (v2 >= vth) & (cool <= 0.0)
            cool2 = torch.where(spk, ref_m1, torch.clamp(cool - 1.0, min=0.0))
            a3 = torch.where(spk, a2 + ahp_inc_c, a2)
            v3 = torch.where(cool2 > 0.0, vres, v2)
            return v3, ge2 * ae, gi2 * ai, cool2, a3, spk
    else:
        def core(v, ge, gi, g_ahp, cool, ev_e, ev_i, iext):
            ge2 = ge + ev_e                   # 步起点交付（事件在步起点生效）
            gi2 = gi + ev_i
            ga2 = g_ahp * aa                  # 物理档：AHP 为**钾电导**(1/s)，反转电位 E_K
            drive = (gl * (vr - v) + ge2 * (ee - v) + gi2 * (ei - v)
                     + ga2 * (ek - v) + iext)
            v2 = torch.clamp(v + dt_s * drive, min=vfloor)
            spk = (v2 >= vth) & (cool <= 0.0)
            cool2 = torch.where(spk, ref_m1, torch.clamp(cool - 1.0, min=0.0))
            ga3 = torch.where(spk, ga2 + ahp_g_inc, ga2)
            v3 = torch.where(cool2 > 0.0, vres, v2)
            return v3, ge2 * ae, gi2 * ai, cool2, ga3, spk

    return core


def _two_comp_core_factory(dt_ms: float, two_comp: bool):
    """two_comp（HH）档状态更新核——**B1 融合手段**（语义逐字等价，仅改实现）：

    ① 门控闭式 `x ← (x − α/(α+β))·e^{−(α+β)dt} + α/(α+β)`（替代 stack + where 分支）；
    ② v 闭式 `v ← (v − B_v/gsum)·e^{−gsum·dt/Cm} + B_v/gsum`；
    ③ **删除 `|A|<1e-12` 守卫**（HH 下 α+β>0、gsum ≥ gL > 0 恒成立 → 守卫是纯 launch 开销）；
    ④ m/h/n 用**独立张量**（免 stack/unstack）；⑤ 电导衰减 `mul_` in-place。

    等价性：`B/A` 在 A=−s 时为 `−α/s`（IEEE 精确取负）→ 数值与参考实现同值；
    对齐由 §2.5 对齐门实测（`tools/scan_m9_engine_vec.py`）复核。
    """
    dt_s = dt_ms * 1e-3
    ae = math.exp(-dt_s / TAU_AMPA_S)
    ai = math.exp(-dt_s / TAU_GABA_S)
    ref_m1 = float(max(REF_MS / dt_ms - 1.0, 0.0))

    def core(v, m, h, n, ge, gi, cool, ev_e, ev_i, iext,
             gNa, gK, gL, AREA, peer, g_ax_dens):
        ge2 = ge + ev_e                       # 步起点交付（事件在步起点生效）
        gi2 = gi + ev_i
        vm = v * 1e3
        am = (0.1 * (vm + 40.0) / (1.0 - torch.exp(-(vm + 40.0) / 10.0))) * 1e3
        bm = 4.0 * torch.exp(-(vm + 65.0) / 18.0) * 1e3
        sm = am + bm
        rm = am / sm
        m_new = (m - rm) * torch.exp(-sm * dt_s) + rm
        ah = 0.07 * torch.exp(-(vm + 65.0) / 20.0) * 1e3
        bh = 1.0 / (1.0 + torch.exp(-(vm + 35.0) / 10.0)) * 1e3
        sh = ah + bh
        rh = ah / sh
        h_new = (h - rh) * torch.exp(-sh * dt_s) + rh
        an = (0.01 * (vm + 55.0) / (1.0 - torch.exp(-(vm + 55.0) / 10.0))) * 1e3
        bn = 0.125 * torch.exp(-(vm + 65.0) / 80.0) * 1e3
        sn = an + bn
        rn = an / sn
        n_new = (n - rn) * torch.exp(-sn * dt_s) + rn
        m3h = m ** 3 * h                      # 步起点 m/h（Brian2 exponential_euler 语义）
        n4 = n ** 4
        gsum = gL + gNa * m3h + gK * n4 + ge2 + gi2
        B_v = (gL * EL_V + gNa * m3h * ENA_V + gK * n4 * EK_V + gi2 * E_GABA_V
               + iext / AREA) / CM_SI
        if two_comp:
            gsum = gsum + g_ax_dens
            B_v = B_v + (g_ax_dens * v[peer]) / CM_SI
        # 参考实现：A_v = −gsum/Cm，B/A = −B_v·Cm/gsum ⇒ 闭式 (v − B/A)·e^{A·dt} + B/A
        Bv_g = (B_v * CM_SI) / gsum
        v_new = (v - Bv_g) * torch.exp(-gsum * (dt_s / CM_SI)) + Bv_g
        spk = (v_new > TH_V) & (cool <= 0.0)
        cool2 = torch.where(spk, ref_m1, torch.clamp(cool - 1.0, min=0.0))
        return v_new, m_new, h_new, n_new, ge2 * ae, gi2 * ai, cool2, spk

    return core


# ---------------------------------------------------------------------------
class AdultEngine:
    """M9 全规模向量化神经引擎（torch MPS/CUDA/CPU）。

    两种保真度：
      - `point`（默认，§2.4 全规模）：conductance-based LIF，状态 v/g_e/g_i/cool；
      - `two_comp`：M8 冻结 300 two_comp 定稿配置（HH + 轴耦合），G0 对齐参考档。
    """

    def __init__(self, n_neuron: int, device: str = "mps", dtype=None,
                 use_compile: bool = True, verbose: bool = True,
                 compile_mode: str = "default"):
        if not TORCH_AVAILABLE:  # pragma: no cover
            raise RuntimeError("torch 不可用——M9 引擎需 .venv-m9（torch 2.8.0 + MPS）")
        self.n = int(n_neuron)
        if device == "mps" and not torch.backends.mps.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.dtype = dtype if dtype is not None else torch.float32
        self.use_compile = bool(use_compile)
        self.compile_mode = str(compile_mode)
        self.verbose = bool(verbose)
        self.version = M9_ENGINE_VERSION

        self._edge_pre: List[np.ndarray] = []
        self._edge_post: List[np.ndarray] = []
        self._edge_gmax: List[np.ndarray] = []
        self._edge_delay: List[np.ndarray] = []
        self._edge_inh: List[np.ndarray] = []
        self._edge_class: List[np.ndarray] = []
        self._gmax_np: Optional[np.ndarray] = None
        self._built = False
        self.stats: Dict[str, Any] = {}
        self._cores: Dict[str, Any] = {}
        self._compile_ok: Dict[str, bool] = {}
        self.point = PointParams()

    # ---------------- 装配 ----------------
    def add_synapses(self, pre, post, gmax, delay_steps, inhibitory=None,
                     syn_class=None) -> None:
        """登记化学边（pre/post 为**神经元索引**；可多次调用，finalize 时拼接）。"""
        pre = np.asarray(pre, dtype=np.int64).ravel()
        post = np.asarray(post, dtype=np.int64).ravel()
        gmax = np.asarray(gmax, dtype=np.float32).ravel()
        if pre.shape != post.shape or pre.shape != gmax.shape:
            raise ValueError("pre/post/gmax 形状不一致")
        if np.isscalar(delay_steps):
            dly = np.full(pre.shape, int(delay_steps), dtype=np.int32)
        else:
            dly = np.asarray(delay_steps, dtype=np.int32).ravel()
        inh = (np.zeros(pre.shape, dtype=bool) if inhibitory is None
               else np.asarray(inhibitory, dtype=bool).ravel())
        cls = (np.zeros(pre.shape, dtype=np.int8) if syn_class is None
               else np.asarray(syn_class, dtype=np.int8).ravel())
        if pre.size:
            if pre.min() < 0 or pre.max() >= self.n or post.max() >= self.n:
                raise ValueError("边索引越界")
        self._edge_pre.append(pre)
        self._edge_post.append(post)
        self._edge_gmax.append(gmax)
        self._edge_delay.append(dly)
        self._edge_inh.append(inh)
        self._edge_class.append(cls)

    def finalize(self) -> Dict[str, Any]:
        """构建 CSR（按 delay 分组）+ 环形缓冲 + 编译内核；返回构建统计。"""
        t0 = time.perf_counter()
        n = self.n
        if self._edge_pre:
            pre = np.concatenate(self._edge_pre)
            post = np.concatenate(self._edge_post)
            gmax = np.concatenate(self._edge_gmax)
            dly = np.concatenate(self._edge_delay)
            inh = np.concatenate(self._edge_inh)
            cls = np.concatenate(self._edge_class)
        else:
            pre = np.zeros(0, dtype=np.int64); post = pre.copy()
            gmax = np.zeros(0, dtype=np.float32); dly = np.zeros(0, dtype=np.int32)
            inh = np.zeros(0, dtype=bool); cls = np.zeros(0, dtype=np.int8)
        self._gmax_np = gmax
        self._inh_np = inh
        self._cls_np = cls
        self._edge_pre_all = pre
        self._edge_post_all = post
        self._edge_dly_all = dly
        self.n_edge = int(pre.size)

        # 延迟 0 → 下一步生效（= 参考实现"直接加入 g"语义）
        self._dly_eff = np.maximum(dly, 1).astype(np.int32)
        self.max_delay = int(self._dly_eff.max()) if pre.size else 1
        self.n_slot = self.max_delay + SLOT_EXTRA + 1
        self.n_delay_group = int(len(np.unique(self._dly_eff))) if pre.size else 1

        self._groups: List[Dict[str, Any]] = []
        if pre.size:
            for d in np.unique(self._dly_eff):
                m = self._dly_eff == d
                eidx = np.flatnonzero(m)
                gp = pre[m]
                order = np.argsort(gp, kind="stable")
                eidx = eidx[order]
                gpost = post[m][order]
                gg = gmax[m][order]
                gcls = cls[m][order]
                counts = np.bincount(gp, minlength=n).astype(np.int64)
                row = np.zeros(n + 1, dtype=np.int64)
                np.cumsum(counts, out=row[1:])
                self._groups.append({
                    "delay": int(d), "edge_idx": eidx, "row": row, "counts": counts,
                    "post": gpost, "gmax": gg, "cls": gcls,
                    "t_row": torch.as_tensor(row, device=self.device),
                    "t_counts": torch.as_tensor(counts, device=self.device),
                    "t_post_off": torch.as_tensor(
                        (gpost + n * (gcls > 0).astype(np.int64)).astype(np.int32),
                        device=self.device),
                    "t_gmax": torch.as_tensor(gg, device=self.device),
                })
        else:
            row = np.zeros(n + 1, dtype=np.int64)
            self._groups.append({
                "delay": 1, "edge_idx": np.zeros(0, dtype=np.int64), "row": row,
                "counts": np.zeros(n, dtype=np.int64),
                "post": np.zeros(0, dtype=np.int64), "gmax": np.zeros(0, dtype=np.float32),
                "cls": np.zeros(0, dtype=np.int8),
                "t_row": torch.zeros(n + 1, dtype=torch.int64, device=self.device),
                "t_counts": torch.zeros(n, dtype=torch.int64, device=self.device),
                "t_post_off": torch.zeros(0, dtype=torch.int32, device=self.device),
                "t_gmax": torch.zeros(0, dtype=self.dtype, device=self.device),
            })

        # 事件环形缓冲：(n_slot, 2N) 扁平化，索引 = slot*2N + post_off
        self._rings = {g["delay"]: torch.zeros(self.n_slot * 2 * n, dtype=self.dtype,
                                               device=self.device)
                       for g in self._groups}

        self.t_v = torch.full((n,), self.point.v_rest, dtype=self.dtype, device=self.device)
        self.t_ge = torch.zeros(n, dtype=self.dtype, device=self.device)
        self.t_gi = torch.zeros(n, dtype=self.dtype, device=self.device)
        self.t_cool = torch.zeros(n, dtype=self.dtype, device=self.device)
        self.t_ahp = torch.zeros(n, dtype=self.dtype, device=self.device)
        self.t_count = torch.zeros(n, dtype=self.dtype, device=self.device)
        self.t_iext = torch.zeros(n, dtype=self.dtype, device=self.device)
        self.t_stim = None
        self.t_stim_col = None
        self._ar_buf = None
        self._build_cores()
        self._built = True
        self.stats["build_s"] = time.perf_counter() - t0
        self.stats["n_edge"] = self.n_edge
        self.stats["n_delay_group"] = self.n_delay_group
        self.stats["n_slot"] = self.n_slot
        return dict(self.stats)

    def _build_cores(self):
        self._cores.clear()
        self._compile_ok.clear()
        fns = [("point", _point_core_factory(self.point))]
        cfg = getattr(self, "_two_comp_cfg", None)
        if cfg is not None:
            fns.append(("two_comp", _two_comp_core_factory(cfg["dt_ms"], cfg["two_comp"])))
        for name, fn in fns:
            if self.use_compile and hasattr(torch, "compile"):
                self._cores[name] = torch.compile(fn, dynamic=False)
                self._compile_ok[name] = True
            else:
                self._cores[name] = fn
                self._compile_ok[name] = False
        self.accel = "torch.compile" if any(self._compile_ok.values()) else "eager"

    # ---------------- 保真度配置 ----------------
    def set_point_params(self, **kw) -> None:
        # 静默失效守卫（M9-B2 实测教训）：ahp_inc 仅在 ahp_form="current" 时生效；
        # 反之 ahp_g_inc 仅在 "conductance" 时生效 → 误设必须**显式报错**，不得静默忽略。
        form = str(kw.get("ahp_form", getattr(self.point, "ahp_form", "conductance")))
        if "ahp_inc" in kw and form != "current":
            raise ValueError("ahp_inc 仅在 ahp_form='current' 时生效；当前 ahp_form=%r "
                             "→ 请改用 ahp_g_inc（或显式设 ahp_form='current'）" % form)
        if "ahp_g_inc" in kw and form != "conductance":
            raise ValueError("ahp_g_inc 仅在 ahp_form='conductance' 时生效；当前 ahp_form=%r"
                             % form)
        for k, v in kw.items():
            if not hasattr(self.point, k):
                raise KeyError("未知 point 参数：%s" % k)
            if isinstance(v, str) or isinstance(getattr(self.point, k), str):
                setattr(self.point, k, str(v))     # 选择型参数（如 ahp_form）
            else:
                setattr(self.point, k, float(v))
        if self._built:
            self._build_cores()

    def set_neuron_heterogeneity(self, i_bias=None) -> None:
        """逐神经元异质 tonic 偏置（固定 seed 由调用方保证；M8 异质杠杆语义）。"""
        if i_bias is not None:
            self.t_iext = torch.as_tensor(np.asarray(i_bias, dtype=np.float32),
                                          device=self.device)

    def set_stim(self, stim_matrix, stim_col) -> None:
        """稀疏 stim 编码（§3.5.1）：(n_steps, n_col) 矩阵 + role→column 索引。

        非刺激神经元映射到末尾**恒零列**（物化保证形状不变——M4 L16 编译缓存纪律）。
        """
        s = np.asarray(stim_matrix, dtype=np.float32)
        if s.ndim == 1:
            s = s[:, None]
        s = np.concatenate([s, np.zeros((s.shape[0], 1), dtype=np.float32)], axis=1)
        self.t_stim = torch.as_tensor(s, device=self.device)
        col = np.asarray(stim_col, dtype=np.int64).copy()
        col = np.where((col < 0) | (col >= s.shape[1] - 1), s.shape[1] - 1, col)
        self.t_stim_col = torch.as_tensor(col, device=self.device)

    def configure_two_comp(self, ex: Dict[str, Any]) -> None:
        """由 `scan_m9_engine.extract_network` 的提取字典配置 two_comp 档（对齐探针用）。"""
        self._two_comp_cfg = {"dt_ms": float(ex["dt_ms"]),
                              "two_comp": bool(ex["two_comp"])}
        for stype, s in ex["syn"].items():
            self.add_synapses(s["pre"], s["post"], s["gmax"], int(s["delay_steps"]),
                              inhibitory=(stype == "gaba"),
                              syn_class=np.full(len(s["pre"]),
                                                1 if stype == "gaba" else 0, dtype=np.int8))
        self.finalize()
        dev, dt = self.device, self.dtype
        self.t_v = torch.as_tensor(np.asarray(ex["state"]["v"], dtype=np.float32),
                                   dtype=dt, device=dev)
        self._mhn0 = [torch.as_tensor(np.asarray(ex["state"][k], dtype=np.float32),
                                      dtype=dt, device=dev) for k in ("m", "h", "n")]
        self.t_m, self.t_h, self.t_n = (x.clone() for x in self._mhn0)
        self.t_ge = torch.as_tensor(np.asarray(ex["state"]["g_ampa"], dtype=np.float32),
                                    dtype=dt, device=dev)
        self.t_gi = torch.as_tensor(np.asarray(ex["state"]["g_gaba"], dtype=np.float32),
                                    dtype=dt, device=dev)
        for k in ("gNa", "gK", "gL", "AREA"):
            setattr(self, "t_" + k,
                    torch.as_tensor(np.asarray(ex[k], dtype=np.float32), dtype=dt, device=dev))
        self.t_peer = torch.as_tensor(np.asarray(ex["peer"], dtype=np.int64), device=dev)
        self.t_g_ax_dens = G_AX_S / self.t_AREA
        self.set_stim(ex["stim"], np.asarray(ex["stim_col"], dtype=np.int64))
        self._build_cores()

    # ---------------- 消融 / 权重缩放 ----------------
    def scale_synapses(self, mask: np.ndarray, factor: float) -> None:
        """按边掩码（**原始边序**）缩放权重——消融：GABA 边关闭 / 子图断开（拓扑不变）。"""
        if self._gmax_np is None:
            raise RuntimeError("finalize() 之前不可缩放")
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self._gmax_np.shape:
            raise ValueError("掩码形状与边数不一致")
        self._gmax_np = np.where(mask, self._gmax_np * float(factor),
                                 self._gmax_np).astype(np.float32)
        for grp in self._groups:
            grp["t_gmax"] = torch.as_tensor(self._gmax_np[grp["edge_idx"]],
                                            device=self.device)

    def set_gmax(self, values: np.ndarray) -> None:
        """整体替换逐边权重（标定重扫用；不改拓扑/排序，仅重排到各组 CSR 序）。"""
        values = np.asarray(values, dtype=np.float32)
        if values.shape != self._gmax_np.shape:
            raise ValueError("权重形状与边数不一致")
        self._gmax_np = values
        for grp in self._groups:
            grp["t_gmax"] = torch.as_tensor(values[grp["edge_idx"]], device=self.device)

    def set_external_events(self, steps, neurons, gmax, n_steps: int,
                            inhibitory=None, delay_steps: Optional[int] = None
                            ) -> Dict[str, Any]:
        """外部 Poisson 背景驱动（虚拟突触事件；逐神经元独立，固定 seed 确定性）。

        语义：事件于步 s 投递 → 步 s+1 生效（进入兴奋/抑制电导累加器，随 tau 衰减）——
        等价于一个虚拟突触前神经元在步 s 发放。**无 host 同步**：事件按步预排序，
        逐步计数前缀和存为 numpy 数组（host 侧整数索引，零同步）。
        抽象登记：持续感觉/内在驱动（真实值不可得，按拟合集 A 标定）。
        性能：投递按 **chunk 批量 + `index_put_(accumulate=True)`**（实测 `index_add_` 成本
        随**目标张量尺寸**线性（5.85M 元 ≈0.20ms，与索引数无关）→ 每步一次不可接受）。
        """
        steps = np.asarray(steps, dtype=np.int64).ravel()
        neurons = np.asarray(neurons, dtype=np.int64).ravel()
        gmax = np.asarray(gmax, dtype=np.float32).ravel()
        inh = (np.zeros(steps.shape, dtype=bool) if inhibitory is None
               else np.asarray(inhibitory, dtype=bool).ravel())
        if not (steps.shape == neurons.shape == gmax.shape == inh.shape):
            raise ValueError("外部事件数组形状不一致")
        order = np.argsort(steps, kind="stable")
        steps, neurons, gmax, inh = steps[order], neurons[order], gmax[order], inh[order]
        counts = np.bincount(steps, minlength=int(n_steps))[:int(n_steps)]
        off = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self._ext_neuron = torch.as_tensor(
            (neurons + self.n * inh.astype(np.int64)).astype(np.int64),
            device=self.device)
        self._ext_amp = torch.as_tensor(gmax, device=self.device)
        # 目标槽：与网络事件同延迟（保证 chunk 末批量写入后**才**被读取——见环形槽说明）
        self._ext_delay = int(delay_steps)
        self._ext_slot = torch.as_tensor(
            ((steps + self._ext_delay) % self.n_slot) * (2 * self.n),
            device=self.device)
        self._ext_off = off
        self._ext_n_steps = int(n_steps)
        self._ext_active = bool(steps.size)
        return {"n_events": int(steps.size), "max_events_per_step": int(counts.max() if counts.size else 0),
                "mean_events_per_step": float(counts.mean() if counts.size else 0.0)}

    def reset_counts(self) -> None:
        """清零逐神经元 spike 计数与群体计数（静息协议 settle 窗后用）。"""
        self.t_count.zero_()

    # ---------------- 运行 ----------------
    def reset(self, v0=None, seed: int = 0) -> None:
        if not self._built:
            self.finalize()
        n = self.n
        if v0 is None:
            v0 = np.full(n, self.point.v_rest, dtype=np.float32)
        self.t_v = torch.as_tensor(np.asarray(v0, dtype=np.float32),
                                   dtype=self.dtype, device=self.device).clone()
        if getattr(self, "_two_comp_cfg", None) is not None and hasattr(self, "_mhn0"):
            self.t_m, self.t_h, self.t_n = (x.clone() for x in self._mhn0)
        self.t_ge = torch.zeros(n, dtype=self.dtype, device=self.device)
        self.t_gi = torch.zeros(n, dtype=self.dtype, device=self.device)
        self.t_cool = torch.zeros(n, dtype=self.dtype, device=self.device)
        self.t_ahp = torch.zeros(n, dtype=self.dtype, device=self.device)
        self.t_count = torch.zeros(n, dtype=self.dtype, device=self.device)
        for r in self._rings.values():
            r.zero_()
        self._rng_seed = int(seed)

    def _arange(self, cap: int):
        """单调增长的 arange 缓冲（**单一**缓冲，按容量扩容）。

        踩坑（本节点实测，性能/内存双重 bug）：初版以 `cap` 为键做缓存
        （`_ar_cache[cap] = arange(max(cap,1024))`）——但 chunk 的活跃边数 `tot` **每 chunk 都不同**
        → 缓存条目无界增长（30s 试次 3 万 chunk → GB 级滞留张量）+ 每次新分配，
        实测使全规模长试次从 0.9 ms/step 退化到 >15 ms/step 且 RSS 持续膨胀（7.5→14GB）。
        正确做法：只保留**一个**可增长缓冲，超出容量才重新分配（摊薄 O(1)）。
        """
        buf = self._ar_buf
        if buf is None or buf.numel() < cap:
            self._ar_buf = torch.arange(max(int(cap * 1.25), 4096), dtype=torch.int64,
                                        device=self.device)
            buf = self._ar_buf
        return buf

    def run(self, n_steps: int, *, fidelity: str = "point", record: str = "counts",
            delivery: Optional[str] = None, pop_trace: bool = False,
            chunk_steps: Optional[int] = None, record_from: int = 0,
            progress=None, progress_every: int = 20000,
            keep_state: bool = True) -> Dict[str, Any]:
        """执行 n_steps 步。

        delivery: "step"（逐步投递＝参考语义）/"chunk"（每 W 步批量投递）。
        record:   "counts"（逐神经元计数 + 可选群体序列）/"full"（逐 spike（step, idx））。
        """
        if not self._built:
            self.finalize()
        n = self.n
        two = (fidelity == "two_comp")
        dt_ms = self._two_comp_cfg["dt_ms"] if two else self.point.dt_ms
        n_steps = int(n_steps)

        if delivery is None:
            delivery = "step" if two else "chunk"
        W = int(chunk_steps if chunk_steps else 0)
        if delivery == "chunk":
            if W <= 0:
                W = int(min(SLOT_EXTRA, self.max_delay))
            W = int(min(W, self.max_delay, SLOT_EXTRA))
            if W < 1:
                delivery = "step"
        ring_items = list(self._rings.items())
        n2 = 2 * n

        # 记录缓冲（full 记录：以位掩码环形行累积，批量刷回 host）
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
        spike_steps: List[np.ndarray] = []
        spike_idx: List[np.ndarray] = []
        rec_ptr = 0
        rec_base = 0

        core = self._cores[fidelity]
        groups = self._groups
        rings = [r for _, r in ring_items]
        delays = [d for d, _ in ring_items]
        n_slot = self.n_slot
        t_count = self.t_count
        zero_iext = self.t_iext
        len_rings = len(rings)

        # 预绑定（④ 消除 Python 侧开销：循环内无属性查找/字典查找）
        t_v = self.t_v; t_ge = self.t_ge; t_gi = self.t_gi; t_cool = self.t_cool
        t_ahp = self.t_ahp
        t_m = getattr(self, "t_m", None); t_h = getattr(self, "t_h", None)
        t_n = getattr(self, "t_n", None)
        t_gNa = getattr(self, "t_gNa", None); t_gK = getattr(self, "t_gK", None)
        t_gL = getattr(self, "t_gL", None); t_AREA = getattr(self, "t_AREA", None)
        t_peer = getattr(self, "t_peer", None); t_axd = getattr(self, "t_g_ax_dens", None)
        t_stim = self.t_stim; t_stim_col = self.t_stim_col
        has_stim = t_stim is not None
        ring_flat = rings[0] if len_rings == 1 else None
        # 预计算槽/行**视图**（避免每步 Python 切片对象创建——Python 侧开销实测主导）
        rv_e = [None] * n_slot
        rv_i = [None] * n_slot
        if ring_flat is not None:
            rf = ring_flat
            for _k in range(n_slot):
                _o = _k * n2
                rv_e[_k] = rf[_o:_o + n]
                rv_i[_k] = rf[_o + n:_o + n2]
        rec_rows_v = list(spk_ring.unbind(0)) if spk_ring is not None else None
        zero_ev = None if ring_flat is None else ring_flat[n2:2 * n2]
        ext_active = getattr(self, "_ext_active", False) and ring_flat is not None
        ext_off = getattr(self, "_ext_off", None)
        ext_neuron = getattr(self, "_ext_neuron", None)
        ext_amp = getattr(self, "_ext_amp", None)
        ext_n_steps = getattr(self, "_ext_n_steps", 0)
        ext_slot = getattr(self, "_ext_slot", None)

        deliver_step = self._deliver_step
        chunk_mode = (delivery == "chunk")
        _prof = bool(os.environ.get("M9_ENGINE_PROF"))
        _pt = {"core": 0.0, "ring": 0.0, "copy": 0.0, "ext": 0.0,
               "chunk": 0.0, "flush": 0.0, "py": 0.0}

        t0 = time.perf_counter()
        step = 0
        while step < n_steps:
            w = min(W, n_steps - step) if chunk_mode else 1
            for k in range(w):
                s = step + k
                _sl = s % n_slot
                if _prof:
                    _tprev = time.perf_counter()
                if len_rings == 1:
                    ev_e = rv_e[_sl]
                    ev_i = rv_i[_sl]
                    if two:
                        t_v, t_m, t_h, t_n, t_ge, t_gi, t_cool, spk = core(
                            t_v, t_m, t_h, t_n, t_ge, t_gi, t_cool, ev_e, ev_i,
                            t_stim[s][t_stim_col] if has_stim else zero_iext,
                            t_gNa, t_gK, t_gL, t_AREA, t_peer, t_axd)
                    else:
                        t_v, t_ge, t_gi, t_cool, t_ahp, spk = core(
                            t_v, t_ge, t_gi, t_cool, t_ahp, ev_e, ev_i,
                            t_stim[s][t_stim_col] if has_stim else zero_iext)
                    if _prof:
                        _b = time.perf_counter(); _pt["core"] += _b - _tprev; _tprev = _b
                    ev_e.zero_()
                    ev_i.zero_()             # 读后清零（预计算槽视图；等价整槽清零）
                    if _prof:
                        _c = time.perf_counter(); _pt["ring"] += _c - _tprev; _tprev = _c
                else:
                    slot = _sl * n2
                    ev_e = None
                    ev_i = None
                    for r in rings:
                        sl = r[slot:slot + n]
                        ev_e = sl if ev_e is None else ev_e + sl
                        sl = r[slot + n:slot + n2]
                        ev_i = sl if ev_i is None else ev_i + sl
                    iext = t_stim[s][t_stim_col] if has_stim else zero_iext
                    if two:
                        t_v, t_m, t_h, t_n, t_ge, t_gi, t_cool, spk = core(
                            t_v, t_m, t_h, t_n, t_ge, t_gi, t_cool, ev_e, ev_i, iext,
                            t_gNa, t_gK, t_gL, t_AREA, t_peer, t_axd)
                    else:
                        t_v, t_ge, t_gi, t_cool, t_ahp, spk = core(
                            t_v, t_ge, t_gi, t_cool, t_ahp, ev_e, ev_i, iext)
                    for r in rings:
                        r[slot:slot + n2].zero_()
                if ext_active and not chunk_mode and s < ext_n_steps:
                    ea = int(ext_off[s]); eb = int(ext_off[s + 1])
                    if eb > ea:
                        ring_flat.index_put_(
                            (ext_neuron[ea:eb] + ext_slot[ea:eb],), ext_amp[ea:eb],
                            accumulate=True)
                if chunk_mode:
                    if _prof:
                        _a = time.perf_counter(); _pt["py"] += _a - _tprev; _tprev = _a
                    rec_rows_v[rec_ptr].copy_(spk)
                    rec_ptr += 1
                else:
                    t_count += spk
                    if record == "full":
                        rec_rows_v[rec_ptr].copy_(spk)
                        rec_ptr += 1
                    deliver_step(spk, s)
                    if pop_dev is not None:
                        pop_dev[s:s + 1].copy_(
                            spk.sum().reshape(1).to(self.dtype))
            # ---- 批量投递（chunk：固定开销摊薄 w 倍） ----
            if chunk_mode:
                if _prof:
                    _d = time.perf_counter(); _pt["copy"] += _d - _tprev; _tprev = _d
                if ext_active:
                    ea = int(ext_off[step]); eb = int(ext_off[step + w])
                    if eb > ea:
                        ring_flat.index_put_(
                            (ext_neuron[ea:eb] + ext_slot[ea:eb],), ext_amp[ea:eb],
                            accumulate=True)
                if _prof:
                    _e = time.perf_counter(); _pt["ext"] += _e - _tprev; _tprev = _e
                self._deliver_chunk(spk_ring, w, step, t_count)
                if _prof:
                    _f = time.perf_counter(); _pt["chunk"] += _f - _tprev; _tprev = _f
                if pop_dev is not None:
                    pop_dev[step:step + w].copy_(spk_ring[:w].sum(dim=1).to(self.dtype))
            # ---- 记录刷回 host ----
            if rec_ptr > 0 and (chunk_mode or rec_ptr >= rec_rows):
                if record == "full":
                    self._flush_rec(spk_ring, rec_ptr, rec_base - record_from,
                                    spike_steps, spike_idx)
                rec_base += rec_ptr
                rec_ptr = 0
            step += w
            if progress is not None and (step % progress_every < w or step >= n_steps):
                progress(step, n_steps, time.perf_counter() - t0)

        wall = time.perf_counter() - t0
        if _prof:
            self.stats["phase_ms_per_step"] = {
                k: v / max(n_steps, 1) * 1e3 for k, v in _pt.items()}
            print("  [engine 相位] " + " ".join(
                "%s=%.3f" % (k, v / max(n_steps, 1) * 1e3) for k, v in _pt.items()),
                flush=True)
        if record == "full" and rec_ptr > 0:
            self._flush_rec(spk_ring, rec_ptr, rec_base - record_from,
                            spike_steps, spike_idx)

        self.t_v, self.t_ge, self.t_gi, self.t_cool = t_v, t_ge, t_gi, t_cool
        self.t_ahp = t_ahp
        if two:
            self.t_m, self.t_h, self.t_n = t_m, t_h, t_n
        counts = self.t_count.detach().cpu().numpy().astype(np.float64)
        res = {
            "wall_s": wall, "n_steps": n_steps, "dt_ms": dt_ms,
            "ms_per_step": wall / max(n_steps, 1) * 1e3,
            "spk_total": float(np.sum(counts)),
            "spike_counts": counts,
            "pop": (pop_dev.detach().cpu().numpy() if pop_dev is not None else None),
            "spikes": ((np.concatenate(spike_steps), np.concatenate(spike_idx))
                       if (record == "full" and spike_steps) else None),
            "delivery": delivery, "chunk_steps": W if delivery == "chunk" else 1,
            "accel": self.accel, "device": str(self.device),
            "n_edge": self.n_edge, "n_neuron": n, "n_slot": n_slot,
        }
        if record == "full":
            res["spike_counts"] = None
            res["spk_total"] = float(res["spikes"][0].size) if res["spikes"] else 0.0
        if not keep_state:
            self.reset(seed=getattr(self, "_rng_seed", 0))
        return res

    # ---------------- 内部 ----------------
    def _deliver_step(self, spk, s: int) -> int:
        """逐步投递：spike 集 → CSR 出边 → scatter-add 到环形槽。"""
        total = 0
        for grp in self._groups:
            idx = spk.nonzero(as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            c = grp["t_counts"][idx]
            tot = int(c.sum().item())
            if tot == 0:
                continue
            total += tot
            off = torch.repeat_interleave(grp["t_row"][idx] + torch.cumsum(c, 0) - c, c)
            base = self._arange(tot)[:tot] + off
            slot = ((s + grp["delay"]) % self.n_slot) * (2 * self.n)
            self._rings[grp["delay"]].index_add_(
                0, grp["t_post_off"][base] + slot, grp["t_gmax"][base])
        return total

    def _deliver_chunk(self, spk_ring, w: int, step0: int, t_count) -> int:
        """批量投递：一次 nonzero 覆盖 w 步（固定开销摊薄 w 倍）+ 同步更新逐神经元计数。"""
        if w <= 0:
            return 0
        rows = spk_ring[:w]
        t_count += rows.sum(dim=0)
        flat = rows.reshape(-1).nonzero(as_tuple=False).flatten()
        if flat.numel() == 0:
            return 0
        n_spk = int(flat.numel())
        # 过载保护：spike 过密时退化为逐步投递（避免 tot 级临时张量 OOM——
        # 全活跃网络 chunk 投递需 150M 边 → 数个 GB 临时量）
        if n_spk * (self.n_edge / max(self.n, 1)) > MAX_EDGES_PER_FLUSH:
            tot_all = 0
            for k in range(w):
                tot_all += self._deliver_step(rows[k], step0 + k)
            return tot_all
        n = self.n
        kk = torch.div(flat, n, rounding_mode="floor")
        nn = flat - kk * n
        total = 0
        for grp in self._groups:
            c = grp["t_counts"][nn]
            tot = int(c.sum().item())
            if tot == 0:
                continue
            total += tot
            slot_k = ((step0 + kk + grp["delay"]) % self.n_slot) * (2 * n)
            packed = torch.stack([grp["t_row"][nn] + torch.cumsum(c, 0) - c, slot_k], dim=1)
            rep = torch.repeat_interleave(packed, c, dim=0)
            base = self._arange(tot)[:tot] + rep[:, 0]
            dst = grp["t_post_off"][base] + rep[:, 1]
            self._rings[grp["delay"]].index_add_(0, dst, grp["t_gmax"][base])
        return total

    def _flush_rec(self, spk_ring, rows: int, t0_rel, spike_steps, spike_idx):
        nz = spk_ring[:rows].nonzero(as_tuple=False)
        if nz.numel():
            r = nz[:, 0].cpu().numpy().astype(np.int64)
            c = nz[:, 1].cpu().numpy().astype(np.int64)
            spike_steps.append(t0_rel + r)
            spike_idx.append(c)

    # ---------------- 统计辅助 ----------------
    def firing_stats(self, t_meas_ms: float) -> Dict[str, float]:
        """静息协议判据量（§3.5.2）：发放率中位/均值、静默比例、群体发放率。"""
        c = self.t_count.detach().cpu().numpy().astype(np.float64)
        rate = c / (t_meas_ms / 1000.0)
        return {
            "rate_median_hz": float(np.median(rate)),
            "rate_mean_hz": float(np.mean(rate)),
            "rate_p95_hz": float(np.percentile(rate, 95)),
            "rate_max_hz": float(np.max(rate)),
            "silent_frac": float(np.mean(rate < 0.5)),
            "active_frac": float(np.mean(rate >= 0.5)),
            "n_spikes": float(c.sum()),
            "pop_rate_hz": float(c.sum() / self.n / (t_meas_ms / 1000.0)),
            "n_nan": int(np.sum(~np.isfinite(rate))),
            "n_neuron": int(self.n),
        }


# ---------------------------------------------------------------------------
def engine_from_extracted(ex: Dict[str, Any], device: str = "cpu",
                          use_compile: bool = False) -> AdultEngine:
    """从 `scan_m9_engine.extract_network` 提取字典构建 two_comp 对齐引擎。"""
    e = AdultEngine(ex["n_comp"], device=device, use_compile=use_compile)
    e.configure_two_comp(ex)
    return e
