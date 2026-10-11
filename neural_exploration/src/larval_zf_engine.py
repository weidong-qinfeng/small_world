"""M10 内核扩展（斑马鱼幼鱼全脑）——**在 M9 冻结内核之上继承扩展，不改一行冻结件**。

《生物仿真M10实施清单》§2.2（E1–E5）/§2.3（对齐门 (a)–(e) + 关闭态等价）/§2.5（冻结规则）。

| 扩展 | 内容 | 承接的 M9 反证 |
|---|---|---|
| **E4** | **延迟槽改绝对步号驱动**：环形槽读写一律用 `abs_step % n_slot`；外部事件表按**绝对步号**索引；chunk 切分锚定在**绝对 W 栅格**上 → 分段调用（k=1/2/5）**逐位可组合** | M9 R6（`m9_engine_segment_composability.csv`：0.2–0.7% 依赖 / 17.5% 神经元计数不同） |
| **E1** | **突触短期可塑性（Tsodyks–Markram）**：逐突触 `u`（利用度）/`x`（资源），`u0/τ_fac/τ_rec`；语义复用 M2 冻结件 `src/synapse_model.py`；按边向量化（禁逐突触 Python 循环，M9 L16#1） | M9 R1/R3（状态切换的**慢变量**） |
| **E2** | **GABA 亚型异质**：抑制边分 `GABA_A`(快 τ≈5–20 ms) / `GABA_B`(慢 τ≈80–400 ms) 双电导轨迹（`g_i_fast`/`g_i_slow`，独立反转电位与时间常数） | M9 R1（慢抑制 = 双态时长候选） |
| **E3** | **调质增益（tonic→phasic）**：慢变量 `m(t)` → 全局增益 `g_mod`，作用于**状态变量**（兴奋/抑制权重、AHP 电导、外部事件幅度），**不是只调背景率** | M9 R2/R3（仅调背景率无效：4× 驱动仅 +7.2%） |

纪律（M10 §2.5）：
  - 内核任何改动必须重过对齐门 (a)–(e) + **E1–E3 关闭态等价**（关闭 = 退化回 M9 语义，逐 spike 一致）；
  - `use_abs_slots=False` + `stp_enabled=False` + `gaba_split=False` + `g_mod=1` ⇒ **E4 关闭**（退化为 M9 环槽语义）
    与 E1–E3 关闭态；该组合下须与 `AdultEngine` 逐位一致（本模块自检项，见 `tools/scan_m10_engine.py`）。

用法：
    PYTHONHASHSEED=0 ./.venv-m9/bin/python -m neural_exploration.tools.scan_m10_engine
"""
from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, Optional

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None
    TORCH_AVAILABLE = False

from .adult_engine import (AdultEngine, PointParams, SLOT_EXTRA, MAX_EDGES_PER_FLUSH,
                           _point_core_factory)

M10_ENGINE_VERSION = "m10e-1.0-abs-step+stp+gaba-subtype+mod"


class LarvalZFEngine(AdultEngine):
    """M10 内核：M9 语义 + E1–E4（默认 E1–E3 关闭、E4 开启）。"""

    def __init__(self, n_neuron: int, device: str = "mps", dtype=None,
                 use_compile: bool = True, verbose: bool = True,
                 compile_mode: str = "default", use_abs_slots: bool = True):
        super().__init__(n_neuron, device=device, dtype=dtype, use_compile=use_compile,
                         verbose=verbose, compile_mode=compile_mode)
        self.version = M10_ENGINE_VERSION
        self.use_abs_slots = bool(use_abs_slots)
        self.abs_step = 0                      # 绝对步号（跨 run() 调用累积）
        # ---- E1 STP ----
        self.stp_enabled = False
        self.stp = dict(u0=0.5, tau_fac_ms=50.0, tau_rec_ms=500.0, mode="depress")
        # ---- E2 GABA 亚型 ----
        self.gaba_split = False
        self.gaba = dict(tau_fast_ms=5.0, tau_slow_ms=150.0, frac_slow=0.2)
        # ---- E3 调质增益 ----
        self.g_mod = 1.0
        self.mod_slow = False
        self.t_mmod = None          # E3 慢变量 m(t)（标量）
        self.mod_trace = []         # m(t) 采样（"参数是活的"自检用）
        self.mod = dict(tau_mod_ms=2000.0, gain_hi=1.3, gain_lo=0.8, source=None)

    # ------------------------------------------------------------------
    # 机制配置（**全部带读回断言**，M9 教训 B3）
    # ------------------------------------------------------------------
    def configure_stp(self, enabled: bool = True, u0: float = 0.5,
                      tau_fac_ms: float = 50.0, tau_rec_ms: float = 500.0,
                      mode: str = "depress") -> dict:
        self.stp_enabled = bool(enabled)
        self.stp = dict(u0=float(u0), tau_fac_ms=float(tau_fac_ms),
                        tau_rec_ms=float(tau_rec_ms), mode=str(mode))
        if self.stp_enabled and self._built:
            self._init_stp_state()
        # 读回断言
        assert self.stp["u0"] == float(u0) and self.stp["tau_rec_ms"] == float(tau_rec_ms)
        return dict(self.stp, enabled=self.stp_enabled)

    def configure_gaba(self, enabled: bool = True, tau_fast_ms: float = 5.0,
                       tau_slow_ms: float = 150.0, frac_slow: float = 0.2) -> dict:
        self.gaba_split = bool(enabled)
        self.gaba = dict(tau_fast_ms=float(tau_fast_ms), tau_slow_ms=float(tau_slow_ms),
                         frac_slow=float(frac_slow))
        if self._built:
            self._build_cores()
            self._alloc_gaba_state()
        assert self.gaba["tau_slow_ms"] == float(tau_slow_ms)
        return dict(self.gaba, enabled=self.gaba_split)

    def configure_mod(self, g_mod: float = 1.0, slow: bool = False,
                      tau_mod_ms: float = 2000.0, gain_hi: float = 1.3,
                      gain_lo: float = 0.8, rate_ref_hz: float = 5.0) -> dict:
        self.g_mod = float(g_mod)
        self.mod_slow = bool(slow)
        self.mod = dict(tau_mod_ms=float(tau_mod_ms), gain_hi=float(gain_hi),
                        gain_lo=float(gain_lo), rate_ref_hz=float(rate_ref_hz))
        # 读回断言（M9 B3 / M10 L21 教训）：写入值 == 读回值，且开关与实现一致
        assert self.g_mod == float(g_mod), "g_mod 读回不一致"
        assert self.mod_slow == bool(slow), "mod_slow 读回不一致"
        assert self.mod["tau_mod_ms"] == float(tau_mod_ms), "tau_mod_ms 读回不一致"
        if self.mod_slow:
            assert self.mod["tau_mod_ms"] > 0, "slow=True 但 tau_mod_ms<=0"
            if self._built:
                self._init_mod_state()
        return dict(g_mod=self.g_mod, slow=self.mod_slow, **self.mod)

    def _init_mod_state(self):
        self.t_mmod = torch.zeros(1, dtype=self.dtype, device=self.device)
        self.mod_trace = []

    # ------------------------------------------------------------------
    # 装配
    # ------------------------------------------------------------------
    def finalize(self) -> Dict[str, Any]:
        st = super().finalize()
        for grp in self._groups:                     # E1：逐边 STP 状态索引（组内 → 全局边号）
            grp["t_eidx"] = torch.as_tensor(np.asarray(grp["edge_idx"], dtype=np.int64),
                                            device=self.device)
        if self.stp_enabled:
            self._init_stp_state()
        if self.gaba_split:
            self._alloc_gaba_state()
        if self.mod_slow and self.t_mmod is None:
            self._init_mod_state()
        st["m10"] = dict(use_abs_slots=self.use_abs_slots, stp=self.stp_enabled,
                         gaba_split=self.gaba_split, g_mod=self.g_mod,
                         mod_slow=self.mod_slow, tau_mod_ms=self.mod["tau_mod_ms"])
        return st

    def _init_stp_state(self):
        n_edge = max(1, int(self.n_edge))
        self.t_u = torch.full((n_edge,), float(self.stp["u0"]), dtype=self.dtype,
                              device=self.device)
        self.t_x = torch.ones(n_edge, dtype=self.dtype, device=self.device)

    def _alloc_gaba_state(self):
        n = self.n
        self.t_gif = torch.zeros(n, dtype=self.dtype, device=self.device)
        self.t_gis = torch.zeros(n, dtype=self.dtype, device=self.device)

    def _build_cores(self):
        """E2：GABA 亚型开启时安装双电导 point 核；否则用 M9 原核（**逐字等价**）。"""
        if not self.gaba_split:
            return super()._build_cores()
        p = self.point
        dt_s = p.dt_ms * 1e-3
        ae = math.exp(-p.dt_ms / p.tau_e)
        aif = math.exp(-p.dt_ms / self.gaba["tau_fast_ms"])
        ais = math.exp(-p.dt_ms / self.gaba["tau_slow_ms"])
        gl = 1.0 / (p.tau_m * 1e-3)
        vr, vth, vres, ee, ei, ek, vfloor = (p.v_rest, p.v_th, p.v_reset, p.e_exc,
                                             p.e_inh, p.ek, p.v_floor)
        ref_m1 = float(max(p.ref_ms / p.dt_ms - 1.0, 0.0))
        aa = (math.exp(-p.dt_ms / p.ahp_tau_ms) if p.ahp_tau_ms and p.ahp_tau_ms > 0
              else 0.0)
        ahp_g_inc = float(p.ahp_g_inc)
        legacy = (str(p.ahp_form) == "current")
        ahp_inc_c = float(p.ahp_inc)
        slow_frac = float(self.gaba["frac_slow"])

        if legacy:  # pragma: no cover
            def core(v, ge, gif, gis, g_ahp, cool, ev_e, ev_i, iext):
                ge2 = ge + ev_e
                gif2 = gif + (1.0 - slow_frac) * ev_i
                gis2 = gis + slow_frac * ev_i
                a2 = g_ahp * aa
                drive = (gl * (vr - v) + ge2 * (ee - v) + gif2 * (ei - v)
                         + gis2 * (ei - v) - a2 + iext)
                v2 = torch.clamp(v + dt_s * drive, min=vfloor)
                spk = (v2 >= vth) & (cool <= 0.0)
                cool2 = torch.where(spk, ref_m1, torch.clamp(cool - 1.0, min=0.0))
                a3 = torch.where(spk, a2 + ahp_inc_c, a2)
                v3 = torch.where(cool2 > 0.0, vres, v2)
                return v3, ge2 * ae, gif2 * aif, gis2 * ais, cool2, a3, spk
        else:
            def core(v, ge, gif, gis, g_ahp, cool, ev_e, ev_i, iext):
                ge2 = ge + ev_e
                gif2 = gif + (1.0 - slow_frac) * ev_i   # E2 类级拆分（无亚型标注 → 类级先验）
                gis2 = gis + slow_frac * ev_i
                ga2 = g_ahp * aa
                drive = (gl * (vr - v) + ge2 * (ee - v) + gif2 * (ei - v)
                         + gis2 * (ei - v) + ga2 * (ek - v) + iext)
                v2 = torch.clamp(v + dt_s * drive, min=vfloor)
                spk = (v2 >= vth) & (cool <= 0.0)
                cool2 = torch.where(spk, ref_m1, torch.clamp(cool - 1.0, min=0.0))
                ga3 = torch.where(spk, ga2 + ahp_g_inc, ga2)
                v3 = torch.where(cool2 > 0.0, vres, v2)
                return v3, ge2 * ae, gif2 * aif, gis2 * ais, cool2, ga3, spk
        self._cores["point"] = core
        self._compile_ok["point"] = False
        self.accel = "eager"
        return None

    # ------------------------------------------------------------------
    # 外部事件（E4：**绝对步号**索引）
    # ------------------------------------------------------------------
    def set_external_events(self, steps, neurons, gmax, n_steps: int,
                            inhibitory=None, delay_steps: Optional[int] = None,
                            slow_frac: float = 0.0) -> Dict[str, Any]:
        """同 M9 语义，但 `steps` 是**协议的绝对步号**（0..n_steps-1，跨分段一致）。

        `slow_frac`：抑制事件中属 GABA_B（慢）的比例（E2）；0 ⇒ 全走快轨迹。
        """
        meta = super().set_external_events(steps, neurons, gmax, n_steps,
                                           inhibitory=inhibitory,
                                           delay_steps=delay_steps)
        self._ext_slow_frac = float(slow_frac)
        # 读回断言：事件表长度 == 协议总步数（跨分段一致）
        assert self._ext_off.size == int(n_steps) + 1, "外部事件表必须覆盖整个协议（绝对步号）"
        return meta

    def reset(self, v0=None, seed: int = 0) -> None:
        super().reset(v0=v0, seed=seed)
        self.abs_step = 0
        if getattr(self, "t_u", None) is not None:
            self.t_u.fill_(float(self.stp["u0"]))
            self.t_x.fill_(1.0)
        if getattr(self, "t_gif", None) is not None:
            self.t_gif.zero_()
            self.t_gis.zero_()
        if getattr(self, "t_mmod", None) is not None:
            self.t_mmod.zero_()
            self.mod_trace = []

    # ------------------------------------------------------------------
    # 运行（**派生自冻结内核 `AdultEngine.run`**，仅四处 delta：绝对槽/绝对事件/chunk 绝对栅格/STP 投递）
    # ------------------------------------------------------------------
    def run(self, n_steps: int, *, fidelity: str = "point", record: str = "counts",
            delivery: Optional[str] = None, pop_trace: bool = False,
            chunk_steps: Optional[int] = None, record_from: int = 0,
            progress=None, progress_every: int = 20000, cache_clear_every: int = 0,
            keep_state: bool = True) -> Dict[str, Any]:
        if not self._built:
            self.finalize()
        n = self.n
        two = (fidelity == "two_comp")
        dt_ms = self._two_comp_cfg["dt_ms"] if two else self.point.dt_ms
        n_steps = int(n_steps)
        abs0 = self.abs_step if self.use_abs_slots else 0

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

        core = self._cores[fidelity]
        rings = [r for _, r in ring_items]
        n_slot = self.n_slot
        t_count = self.t_count
        zero_iext = self.t_iext
        len_rings = len(rings)

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
        rv_e = [None] * n_slot
        rv_i = [None] * n_slot
        if ring_flat is not None:
            for _k in range(n_slot):
                _o = _k * n2
                rv_e[_k] = ring_flat[_o:_o + n]
                rv_i[_k] = ring_flat[_o + n:_o + n2]
        rec_rows_v = list(spk_ring.unbind(0)) if spk_ring is not None else None
        ext_active = getattr(self, "_ext_active", False) and ring_flat is not None
        ext_off = getattr(self, "_ext_off", None)
        ext_neuron = getattr(self, "_ext_neuron", None)
        ext_amp = getattr(self, "_ext_amp", None)
        ext_n_steps = getattr(self, "_ext_n_steps", 0)
        ext_slot = getattr(self, "_ext_slot", None)
        ext_slow_frac = float(getattr(self, "_ext_slow_frac", 0.0))
        gaba = self.gaba_split
        mod_slow = bool(self.mod_slow) and getattr(self, "t_mmod", None) is not None
        mod_scale = float(self.g_mod)
        dt_s_mod = dt_ms * 1e-3
        _tau_mod_s = float(self.mod["tau_mod_ms"]) * 1e-3 if mod_slow else 0.0
        _ghi, _glo = float(self.mod["gain_hi"]), float(self.mod["gain_lo"])
        m_dev = self.t_mmod if mod_slow else None
        self._gmod_step = mod_scale
        mod_sample_every = max(1, int(round(50.0 / max(dt_ms, 1e-9))))   # 每 50 ms 采样一次
        _mstep = [0]

        deliver_step = self._deliver_step
        chunk_mode = (delivery == "chunk")
        t_u = getattr(self, "t_u", None)
        t_x = getattr(self, "t_x", None)
        stp_on = bool(self.stp_enabled) and t_u is not None
        U0 = float(self.stp["u0"]); tau_f = self.stp["tau_fac_ms"]; tau_r = self.stp["tau_rec_ms"]
        stp_mode = str(self.stp["mode"])

        t0 = time.perf_counter()
        step = 0
        while step < n_steps:
            if chunk_mode:
                # E4：chunk 边界锚定**绝对 W 栅格** → 分段切分不影响分段调度（逐位可组合的关键）
                a = (abs0 + step) % W
                w = min(W - a if a else W, n_steps - step)
            else:
                w = 1
            for k in range(w):
                s = step + k
                _sl = ((abs0 + s) if self.use_abs_slots else s) % n_slot
                if len_rings == 1:
                    ev_e = rv_e[_sl]
                    ev_i = rv_i[_sl]
                    if two:
                        t_v, t_m, t_h, t_n, t_ge, t_gi, t_cool, spk = core(
                            t_v, t_m, t_h, t_n, t_ge, t_gi, t_cool, ev_e, ev_i,
                            t_stim[s][t_stim_col] if has_stim else zero_iext,
                            t_gNa, t_gK, t_gL, t_AREA, t_peer, t_axd)
                    elif gaba:
                        t_v, t_ge, t_gif, t_gis, t_cool, t_ahp, spk = core(
                            t_v, t_ge, self.t_gif, self.t_gis, t_ahp, t_cool,
                            ev_e, ev_i, t_stim[s][t_stim_col] if has_stim else zero_iext)
                        self.t_gif, self.t_gis = t_gif, t_gis
                    else:
                        t_v, t_ge, t_gi, t_cool, t_ahp, spk = core(
                            t_v, t_ge, t_gi, t_cool, t_ahp, ev_e, ev_i,
                            t_stim[s][t_stim_col] if has_stim else zero_iext)
                    if mod_slow:
                        # E3：m ← m + (dt/τ)·(target − m)，target = 归一化群体发放率
                        # **归一化（D1 自检的产物）**：target = 瞬时群体率(Hz) / rate_ref
                        # 未归一化时 target ≈ rate·dt ≈ 3e-5 ⇒ m≈0 ⇒ 等效静态增益（"参数不活"）
                        rate_inst = spk.to(self.dtype).mean() / dt_s_mod
                        tgt = (rate_inst / float(self.mod["rate_ref_hz"])).reshape(1)
                        m_dev.add_((dt_s_mod / _tau_mod_s) * (tgt - m_dev))
                        self._gmod_step = float(_glo + (_ghi - _glo) * float(m_dev.item()))
                        _mstep[0] += 1
                        if _mstep[0] % mod_sample_every == 0:
                            self.mod_trace.append(float(m_dev.item()))
                    ev_e.zero_()
                    ev_i.zero_()
                else:  # pragma: no cover（多延迟分组：M9 路径，保持语义）
                    slot = _sl * n2
                    ev_e = None; ev_i = None
                    for r in rings:
                        sl = r[slot:slot + n]
                        ev_e = sl if ev_e is None else ev_e + sl
                        sl = r[slot + n:slot + n2]
                        ev_i = sl if ev_i is None else ev_i + sl
                    iext = t_stim[s][t_stim_col] if has_stim else zero_iext
                    t_v, t_ge, t_gi, t_cool, t_ahp, spk = core(
                        t_v, t_ge, t_gi, t_cool, t_ahp, ev_e, ev_i, iext)
                    for r in rings:
                        r[slot:slot + n2].zero_()
                if ext_active and not chunk_mode and s < ext_n_steps:
                    ea = int(ext_off[s]); eb = int(ext_off[s + 1])
                    if eb > ea:
                        ring_flat.index_put_(
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
                        ring_flat.index_put_(
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

        self.t_v, self.t_ge, self.t_gi, self.t_cool = t_v, t_ge, t_gi, t_cool
        self.t_ahp = t_ahp
        if two:
            self.t_m, self.t_h, self.t_n = t_m, t_h, t_n
        self.abs_step = abs0 + n_steps          # E4：跨调用累积绝对步号
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
            "abs_step_end": self.abs_step,
        }
        if mod_slow and self.mod_trace:
            _tr = np.asarray(self.mod_trace, dtype=float)
            res["mod_trace"] = dict(n=len(self.mod_trace), mean=float(_tr.mean()),
                                    std=float(_tr.std()), min=float(_tr.min()),
                                    max=float(_tr.max()),
                                    dynamic_range=float(_tr.max() - _tr.min()),
                                    gain_range=[float(_glo + (_ghi - _glo) * _tr.min()),
                                                float(_glo + (_ghi - _glo) * _tr.max())],
                                    alive=bool(_tr.std() > 1e-3))   # **"参数是活的"门**（D1）
        if record == "full":
            res["spike_counts"] = None
            res["spk_total"] = float(res["spikes"][0].size) if res["spikes"] else 0.0
        if not keep_state:
            self.reset(seed=getattr(self, "_rng_seed", 0))
        return res

    # ------------------------------------------------------------------
    # E1/E2 投递（覆盖父类：STP 有效权重 + 亚型拆分）
    # ------------------------------------------------------------------
    def _apply_stp_and_deliver(self, grp, base, dst):
        """按边更新 STP 状态并投递有效电导。`base` 已按 `_arange` 偏移就位。"""
        g = grp["t_gmax"][base]
        gm = getattr(self, "_gmod_step", 1.0)
        if gm != 1.0:
            g = g * gm                    # E3：增益作用于**状态变量**（突触电导）
        if self.stp_enabled:
            eidx = grp["t_eidx"][base]
            u = self.t_u[eidx]; x = self.t_x[eidx]
            g = g * u * x
            # TM 更新：u += U0(1-u)（易化）/ release = u·x / x -= u·x
            u2 = u + float(self.stp["u0"]) * (1.0 - u)
            x2 = x - u2 * x
            self.t_u.index_copy_(0, eidx, u2)
            self.t_x.index_copy_(0, eidx, x2)
        self._rings[grp["delay"]].index_add_(0, dst, g)
        return g

    def _decay_stp(self, dt_ms: float):
        if not self.stp_enabled:
            return
        u0 = float(self.stp["u0"])
        if str(self.stp["mode"]) == "facilitate":
            b = math.exp(-dt_ms / max(1e-9, float(self.stp["tau_fac_ms"])))
            self.t_u.mul_(b).add_(u0 * (1.0 - b))
        else:
            self.t_u.mul_(math.exp(-dt_ms / max(1e-9, float(self.stp["tau_fac_ms"]))))
        b2 = math.exp(-dt_ms / max(1e-9, float(self.stp["tau_rec_ms"])))
        self.t_x.mul_(b2).add_(1.0 - b2)

    def _deliver_step(self, spk, abs_step: int) -> int:
        self._decay_stp(self.point.dt_ms)
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
            # **CSR 投递索引修正（M10 E5-FIX）**：M9 冻结件用 `t_row[idx] + cumsum(c)-c`
            # → 第 j 个发放神经元的块起点被平移了 `prev_total`（= 之前所有发放神经元的边数），
            # 当**≥2 个神经元在同一步发放**时越界（CPU 抛 IndexError / MPS 静默错投）。
            # 正确形式：`base = arange(tot) + repeat_interleave(t_row[idx] - (cumsum(c)-c), c)`。
            # 最小复现见 `docs/m10_env_notes.md` L17。
            off = torch.repeat_interleave(grp["t_row"][idx] - (torch.cumsum(c, 0) - c), c)
            base = self._arange(tot)[:tot] + off
            slot = ((abs_step + grp["delay"]) % self.n_slot) * (2 * self.n)
            self._apply_stp_and_deliver(grp, base, grp["t_post_off"][base] + slot)
        return total

    def _deliver_chunk(self, spk_ring, w: int, abs_step0: int, t_count) -> int:
        if w <= 0:
            return 0
        if self.stp_enabled:
            for k in range(w):                    # 逐 step 投递（STP 依赖事件历史，无法跨 step 批量）
                self._deliver_step(spk_ring[k], abs_step0 + k)
            t_count += spk_ring[:w].sum(dim=0)
            return 0
        # ---- 无 STP ⇒ 走父类批量路径（语义与 M9 逐字一致，仅步号改绝对）----
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
        kk = torch.div(flat, n, rounding_mode="floor")
        nn = flat - kk * n
        total = 0
        for grp in self._groups:
            c = grp["t_counts"][nn]
            tot = int(c.sum().item())
            if tot == 0:
                continue
            total += tot
            slot_k = ((abs_step0 + kk + grp["delay"]) % self.n_slot) * (2 * n)
            # E5-FIX（同 _deliver_step）：块起点用 `t_row - prev_total`
            packed = torch.stack([grp["t_row"][nn] - (torch.cumsum(c, 0) - c), slot_k], dim=1)
            rep = torch.repeat_interleave(packed, c, dim=0)
            base = self._arange(tot)[:tot] + rep[:, 0]
            dst = grp["t_post_off"][base] + rep[:, 1]
            _gm = getattr(self, "_gmod_step", 1.0)
            _gv = grp["t_gmax"][base]
            if _gm != 1.0:
                _gv = _gv * _gm           # E3（chunk 路径）
            self._rings[grp["delay"]].index_add_(0, dst, _gv)
        return total
