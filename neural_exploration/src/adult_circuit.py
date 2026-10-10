"""M9 全规模成年果蝇脑回路装配（清单 §3.5 P4；`src/adult_circuit.py`，新建）。

《生物仿真M9实施清单》§3.5.1（全规模构建）/§3.5.2（静息协议）/§3.5.3（双状态）/
§3.5.4（单试次墙钟预算）/§2.4（point-neuron 为主）。

网络：
  - 139,255 点神经元（conductance-based LIF，§2.4）——1 隔室/神经元（point 档）；
  - 15,091,983 化学连接（54,492,922 突触，syn_count 加权）+ **0 缝隙**（官方 v783 发布不含，
    L19.2 裁决②：权威解析 = 0/不可得，测量限制）；
  - **真实递质标注**（L14.3）：兴奋性 11,042,578 边（ach/glut）、**抑制性 3,233,022 边
    （真实 GABA，12,755,910 突触 = 23.4%）**、调质类 816,383 边（DA/5HT/OA）。

权重（§8 ① 类级缩放 ≤40 全局参数，行为学反推；不做逐突触拟合）：
    gmax_e = w_exc · syn_count^γ,  gmax_i = w_inh · syn_count^γ,
    gmax_m = w_mod · syn_count^γ（调质类，抽象登记 T2：P4 以弱兴奋电导承载；
    DA/5HT/OA 的受体映射在 v783 不可得 → M6 功能门控语义 + 测量限制）

抽象登记（§0.3.5；每抽象登记替代对象/误差/回归条件）：
  1. **point-neuron 替代 HH**：误差 = 无树突/簇发放计算；回归条件 = 某行为判据在点神经元下
     不达且机制指向树突（§2.4 铁律 A 回退路径）；
  2. **统一轴突延迟 0.5ms**：替代对象 = 逐连接延迟（FlyWire v783 连接组**不含延迟列**）；
     误差 = 无传导速度异质性；回归条件 = 延迟敏感判据（P6 航向补偿）不达时按类型级延迟回退；
  3. **调质→弱兴奋电导**：替代对象 = 代谢型调质（受体映射 v783 不可得）；误差 = 无增益调制；
     回归条件 = P5/P8 调质消融不成立的机制归属检验；
  4. **递质类级权重**：替代对象 = 逐突触电导（无可得数据）；误差 = 类内异质缺失；
     回归条件 = 类级缩放无法落判据带时引入度/区域调制。

静息协议（§3.5.2/§3.5.3）：
  - 无刺激、无梯度；settle 窗丢弃（t=0 初始化瞬态，M5 L37#2）；测量窗统计；
  - 判据带**定稿于 `data/m9_behavior_reference.csv`**（协议运行前，不事后调，§0.7 #8）；
  - N≥3 试次（固定 seed 确定性）；双状态 = 静默比例 + 活动 bout 结构。

用法：
    from neural_exploration.src.adult_circuit import AdultCircuit
    c = AdultCircuit(device="mps"); c.build()
    r = c.run_resting(T_ms=30000.0, settle_ms=1000.0, seed=0)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
NET_NPZ = os.path.join(DATA_DIR, "m9_network.npz")
NEURON_NPZ = os.path.join(DATA_DIR, "m9_neuron_table.npz")
PARAMS_CSV = os.path.join(DATA_DIR, "m9_circuit_params.csv")

#: 递质类码（与 tools/build_m9_network.py 一致）
CLS_EXC, CLS_INH, CLS_MOD = 0, 1, 2


@dataclass
class CircuitParams:
    """全规模回路参数（类级定稿；标定纪律：只读拟合集 A = 静息判据带，§0.8）。"""
    w_exc: float = 1.0          # 兴奋性边权重系数（gmax = w·syn_count^γ）
    w_inh: float = 4.0          # 抑制性（真实 GABA）边权重系数
    w_mod: float = 0.3          # 调质类边权重系数（抽象 T2）
    syn_count_gamma: float = 1.0
    w_jitter_cv: float = 0.0    # 逐边权重抖动（0 = 关；固定 seed 确定性）
    bias_mv_s: float = 355.0    # tonic 偏置电流（mV/s）中位（对数正态）
    bias_cv: float = 0.60       # 偏置对数正态 sigma（异质兴奋性分布宽度）
    v_rest: float = -52.0
    v_th: float = -45.0
    v_reset: float = -55.0
    tau_m: float = 20.0
    tau_e: float = 2.0
    tau_i: float = 5.0
    ref_ms: float = 2.0
    e_inh: float = -80.0        # mV GABA_A 反转电位（必须 < E_K = -77 mV）
    ek: float = -77.0           # mV 钾平衡电位（AHP 电导反转电位）
    ahp_tau_ms: float = 700.0   # ms AHP 时间常数（发放率上限杠杆）
    ahp_g_inc: float = 100.0    # 1/s 每 spike 的 AHP 电导增量（conductance 档）
    ahp_inc: float = 0.0        # mV/s 每 spike 的 AHP 恒流增量（current/legacy 档）
    ahp_form: str = "conductance"   # "conductance" | "current"（legacy 复现）
    dt_ms: float = 0.05
    delay_ms: float = 1.0
    bias_mode: str = "lognormal"
    seed: int = 0

    def as_dict(self) -> Dict[str, float]:
        return {k: (float(v) if not isinstance(v, int) else int(v))
                for k, v in asdict(self).items()}


class AdultCircuit:
    """139,255 point 神经元全规模成年果蝇脑回路（GPU 装配 + 静息协议）。"""

    def __init__(self, device: str = "mps", params: Optional[CircuitParams] = None,
                 verbose: bool = True, use_compile: bool = True):
        from neural_exploration.src.adult_engine import AdultEngine
        self.device = device
        self.params = params or CircuitParams()
        self.verbose = verbose
        self.use_compile = use_compile
        d = np.load(NET_NPZ, allow_pickle=False)
        self.pre = d["pre"].astype(np.int64)
        self.post = d["post"].astype(np.int64)
        self.syn_count = d["syn_count"].astype(np.float32)
        self.nt_code = d["nt_code"].astype(np.int8)
        self.root_ids = d["root_ids"].astype(np.int64)
        n = self.n_neurons = int(self.root_ids.size)
        self.n_edge = int(self.pre.size)
        try:
            nt = np.load(NEURON_NPZ, allow_pickle=False)
            self.region_code = nt["region_code"].astype(np.int32)
            self.region_names = np.asarray(nt["region_names"])
            self.dom_nt_code = nt["dom_nt_code"].astype(np.int8)
        except Exception:
            self.region_code = np.zeros(n, dtype=np.int32)
            self.region_names = np.asarray([""], dtype="U32")
            self.dom_nt_code = np.full(n, -1, dtype=np.int8)
        self.inh = self.nt_code == CLS_INH
        self.engine = AdultEngine(n, device=device, use_compile=use_compile,
                                  verbose=verbose)
        self._built = False
        self.build_stats: Dict[str, Any] = {}

    # ---------------- 权重 ----------------
    def gmax_vector(self, w_exc=None, w_inh=None, w_mod=None, gamma=None,
                    jitter_cv=None) -> np.ndarray:
        """按类级系数生成逐边 gmax（确定性；jitter 用固定 seed）。"""
        p = self.params
        we = p.w_exc if w_exc is None else float(w_exc)
        wi = p.w_inh if w_inh is None else float(w_inh)
        wm = p.w_mod if w_mod is None else float(w_mod)
        g = p.syn_count_gamma if gamma is None else float(gamma)
        jc = p.w_jitter_cv if jitter_cv is None else float(jitter_cv)
        sc = np.power(self.syn_count, g, dtype=np.float32)
        w = np.where(self.nt_code == CLS_EXC, we,
                     np.where(self.nt_code == CLS_INH, wi, wm)).astype(np.float32)
        gmax = w * sc
        if jc > 0:
            rng = np.random.default_rng(p.seed + 7717)
            gmax = gmax * (1.0 + jc * rng.standard_normal(gmax.size)).astype(np.float32)
            gmax = np.maximum(gmax, 0.0)
        return gmax.astype(np.float32)

    def bias_vector(self, bias_mv_s=None, cv=None, mode=None) -> np.ndarray:
        """逐神经元 tonic 偏置（确定性；异质兴奋性杠杆）。

        `mode="lognormal"`（默认）：b = b_med · exp(σ z) —— 重尾正偏（脑区间兴奋性差异，
        如视叶 vs 中央脑）→ 静息发放率分布呈"多数极低 + 少数较高"形状（拟合集 A 标定）。
        `mode="normal"`：b ~ N(b_med, σ·b_med)。
        """
        p = self.params
        b0 = p.bias_mv_s if bias_mv_s is None else float(bias_mv_s)
        c = p.bias_cv if cv is None else float(cv)
        m = getattr(p, "bias_mode", "lognormal") if mode is None else mode
        rng = np.random.default_rng(p.seed + 11)
        z = rng.standard_normal(self.n_neurons)
        if m == "lognormal":
            b = b0 * np.exp(c * z)
        else:
            b = b0 + c * abs(b0) * z
        return np.maximum(b, 0.0).astype(np.float32)

    # ---------------- 装配 ----------------
    def build(self, gmax=None, bias=None) -> Dict[str, Any]:
        from neural_exploration.src.adult_engine import AdultEngine  # noqa: F401
        p = self.params
        t0 = time.perf_counter()
        gmax = self.gmax_vector() if gmax is None else np.asarray(gmax, dtype=np.float32)
        delay_steps = max(1, int(round(p.delay_ms / p.dt_ms)))
        self.engine.set_point_params(v_rest=p.v_rest, v_th=p.v_th, v_reset=p.v_reset,
                                     tau_m=p.tau_m, tau_e=p.tau_e, tau_i=p.tau_i,
                                     ref_ms=p.ref_ms, dt_ms=p.dt_ms,
                                     e_inh=p.e_inh, ek=p.ek,
                                     ahp_tau_ms=p.ahp_tau_ms, ahp_g_inc=p.ahp_g_inc,
                                     ahp_inc=p.ahp_inc, ahp_form=p.ahp_form)
        n_chunk = 4
        step = int(np.ceil(self.pre.size / n_chunk))
        for k in range(n_chunk):        # 分段装配（M8 分批语义；构建墙钟探针）
            a, b = k * step, min((k + 1) * step, self.pre.size)
            if a >= b:
                break
            self.engine.add_synapses(self.pre[a:b], self.post[a:b], gmax[a:b],
                                     delay_steps=delay_steps,
                                     inhibitory=self.inh[a:b],
                                     syn_class=self.nt_code[a:b])
        self.engine.finalize()
        self._gmax = gmax
        self.engine.set_neuron_heterogeneity(
            i_bias=self.bias_vector() if bias is None else np.asarray(bias, np.float32))
        self.build_stats = dict(self.engine.stats)
        self.build_stats["build_wall_s"] = time.perf_counter() - t0
        self.build_stats["delay_steps"] = delay_steps
        self.build_stats["vram_bytes"] = _mps_mem()
        self.build_stats["n_neurons"] = self.n_neurons
        self.build_stats["n_edges"] = int(self.pre.size)
        self._built = True
        if self.verbose:
            print("全规模装配：%d 神经元 / %d 边 / %d 延迟组 / 槽 %d；构建 %.1fs；"
                  "MPS 显存 %.2f GB" % (self.n_neurons, self.pre.size,
                                        self.build_stats["n_delay_group"],
                                        self.build_stats["n_slot"],
                                        self.build_stats["build_wall_s"],
                                        self.build_stats["vram_bytes"] / 2**30), flush=True)
        return dict(self.build_stats)

    def reweight(self, w_exc=None, w_inh=None, w_mod=None, gamma=None,
                 bias_mv_s=None, bias_cv=None) -> None:
        """标定用：重设权重/偏置（不重建 CSR；秒级）。"""
        if not self._built:
            raise RuntimeError("先 build()")
        self._gmax = self.gmax_vector(w_exc, w_inh, w_mod, gamma)
        self.engine.set_gmax(self._gmax)
        self.engine.set_neuron_heterogeneity(
            i_bias=self.bias_vector(bias_mv_s, bias_cv))

    # ---------------- 背景驱动（虚拟突触前 Poisson 事件） ----------------
    def build_background(self, rate_hz: float = 1.0, g_ext: float = 0.04,
                         n_steps: int = 40000, seed: int = 0, cv: float = 0.0,
                         inh_frac: float = 0.0, rate_cv: float = 0.0) -> Dict[str, Any]:
        """逐神经元独立 Poisson 背景突触驱动（确定性，固定 seed）。

        抽象登记：持续感觉/内在驱动（果蝇中枢脑永不静默——视觉 ME 占 49% 神经元）；
        真实背景率/电导不可得 → 标定于拟合集 A（静息判据带）。
        """
        p = self.params
        T_s = n_steps * p.dt_ms * 1e-3
        rng = np.random.default_rng(seed + 9001)
        lam = np.full(self.n_neurons, float(rate_hz), dtype=np.float64)
        if rate_cv > 0:
            lam = np.maximum(lam * (1.0 + rate_cv * rng.standard_normal(self.n_neurons)),
                             0.0)
        n_ev = rng.poisson(lam * T_s)
        total = int(n_ev.sum())
        if total == 0:
            self.engine.set_external_events(np.zeros(0, np.int64), np.zeros(0, np.int64),
                                            np.zeros(0, np.float32), n_steps)
            return {"n_events": 0}
        neurons = np.repeat(np.arange(self.n_neurons, dtype=np.int64), n_ev)
        steps = rng.integers(0, n_steps, size=total).astype(np.int64)
        inh = rng.random(total) < float(inh_frac)
        amp = (g_ext * (1.0 + 0.15 * rng.standard_normal(total))).astype(np.float32)
        amp = np.maximum(amp, 0.0)
        delay_steps = max(1, int(round(p.delay_ms / p.dt_ms)))
        st = self.engine.set_external_events(steps, neurons, amp, n_steps,
                                             inhibitory=inh, delay_steps=delay_steps)
        st.update({"rate_hz": rate_hz, "g_ext": g_ext, "T_s": T_s})
        if self.verbose:
            print("背景驱动：%d 事件（%.2f 事件/步；率 %.2f Hz；g_ext %.4f）"
                  % (total, st["mean_events_per_step"], rate_hz, g_ext), flush=True)
        return st

    # ---------------- 运行 ----------------
    def run_resting(self, T_ms: float = 1000.0, settle_ms: float = 200.0,
                    seed: int = 0, delivery: str = "chunk", pop_trace: bool = False,
                    v0_seed=None, progress=None, progress_every: int = 100000
                    ) -> Dict[str, Any]:
        """静息协议：settle 窗（丢弃）→ 测量窗统计（§3.5.2）。"""
        if not self._built:
            self.build()
        p = self.params
        dt = p.dt_ms
        n_settle = int(round(settle_ms / dt))
        n_meas = int(round(T_ms / dt))
        rng = np.random.default_rng(1 if v0_seed is None else v0_seed)
        v0 = (p.v_rest + 2.0 * rng.standard_normal(self.n_neurons)).astype(np.float32)
        if getattr(self, "_bg_steps", 0) < n_settle + n_meas:
            self.build_background(*(self._bg_args or ()), n_steps=n_settle + n_meas)
        self.engine.reset(v0=v0, seed=seed)
        t0 = time.perf_counter()
        if n_settle > 0:
            self.engine.run(n_settle, delivery=delivery, record="counts")
        self.engine.reset_counts()
        r = self.engine.run(n_meas, delivery=delivery, record="counts",
                            pop_trace=pop_trace, progress=progress,
                            progress_every=progress_every)
        wall = time.perf_counter() - t0
        st = self.engine.firing_stats(T_ms)
        st.update({"T_ms": T_ms, "settle_ms": settle_ms, "seed": seed,
                   "wall_s": wall, "wall_settle_s": wall,
                   "ms_per_step": r["ms_per_step"],
                   "spk_frac_per_step": st["n_spikes"] / max(self.n_neurons * n_meas, 1),
                   "n_steps": n_meas, "delivery": r["delivery"],
                   "pop": r["pop"]})
        return st

    # ---------------- 角色/子图查询（P5–P9 复用） ----------------
    def region_mask(self, prefixes: Tuple[str, ...]) -> np.ndarray:
        """按主脑区前缀取神经元掩码（cell_type 不可得 → 区域级角色代理，测量限制登记）。"""
        names = np.asarray([str(x) for x in self.region_names])
        codes = [i for i, nm in enumerate(names) if any(nm.startswith(p) for p in prefixes)]
        return np.isin(self.region_code, codes)

    def nt_mask(self, cls: int) -> np.ndarray:
        return self.dom_nt_code == cls

    def summary(self) -> Dict[str, Any]:
        return {"n_neurons": self.n_neurons, "n_edges": int(self.pre.size),
                "build_stats": self.build_stats, "params": self.params.as_dict(),
                "regions": int(len(self.region_names))}


def _mps_mem() -> int:
    try:
        import torch
        if torch.backends.mps.is_available():
            return int(torch.mps.driver_allocated_memory())
    except Exception:
        pass
    return 0


def write_params_csv(path: str = PARAMS_CSV, rows: Optional[List[List[str]]] = None) -> None:
    import csv as _csv
    rows = rows or []
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("# M9 全规模回路参数定稿（§8 ① 类级缩放 ≤40 全局参数；标定只读拟合集 A）\n")
        w = _csv.writer(f, lineterminator="\n")
        w.writerow(["param", "value", "note"])
        w.writerows(rows)
    print("回路参数 →", path, flush=True)
