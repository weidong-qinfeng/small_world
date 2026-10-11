"""M10 斑马鱼幼鱼静息网络装配与协议运行（P5/P6 共用）。

《生物仿真M10实施清单》§4.2（P5 静息标定，G1 前置门）+ §4.1（P6 机制阶梯，前置机制门）+ §0.7（可测性）。

**结构证据级别声明（必读）**：Fish1 全脑逐突触表不可得（CAVE 门控，见 `docs/m10_env_notes.md` L10）
⇒ 完整度 = `partial`。本模块以 **Fish1 公开子图统计量**（实测 E/I 标注比例、roster 规模量级）
为锚，构建**类级递归网络**（随机入度、类级权重、确定性 seed）——**这是抽象，不是连接组**
（§1.1.3 `none` 级的诚实处置语义：不得把抽象当连接组；报告须标注"结构证据级别 = 抽象(partial 锚定)"）。
抽象登记（§0.3.5）：
  - 拓扑：类级随机入度（不含真实 Fish1 逐突触结构）；回归条件 = 取得 CAVE 全脑突触表；
  - 权重：类级（g_exc/g_inh 两个全局参数）→ §0.7 #7 "权重不可辨识，只做类级缩放"；
  - 递质亚型：GABA_A/B 按**类级先验**拆分（无亚型标注 → 测量限制登记）；
  - 背景驱动：Poisson 虚拟突触事件（M9 `set_external_events` 语义，抽象登记）。

协议/判据口径（**运行前定稿**，见 `data/m10_calibration_targets.csv`）：
  - settle 窗 = 500 ms（t=0 初始化瞬态剔除，M5 L37#2）；判据只看 settle 之后；
  - 分段调用：一律 `segment_steps` 分段（M9 L38.1 教训），段间记 ms/step 与 RSS；
  - bout 定义（**可证伪**）：100 ms 分箱；`silent_bin ⟺ 0 spike`；
    `bout ⟺ 由静默峰（或记录起点）分隔的**连续非静默 ≥2 箱**（≥200 ms）`；
    `bout_fraction = Σ bout 箱 / 总箱`。**匀速平坦 Poisson 网络无静默峰 → bout_fraction ≈ 0**
    （与 M9 实测 0.0000 同语义，故该判据可证伪，非"总能过"）。
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

DT_MS = 0.05


@dataclass
class ZFNetParams:
    """类级网络参数（≤8 个全局参数，M10 §4.1.1 组合上限）。"""
    n_neuron: int = 3000
    indegree: int = 20
    inh_frac: float = 0.30            # 抑制边比例（类级；未用真实 Fish1 逐突触结构，抽象登记）
    g_exc: float = 9.0                # 类级兴奋峰电导（1/s）
    g_inh: float = 22.0               # 类级抑制峰电导（1/s）
    delay_max: int = 40               # 统一延迟（步，=2 ms）；单延迟组 ⇒ 冻结内核 chunk 快路径
    bias_median_mv_s: float = 250.0   # tonic 偏置中位（**亚阈**：v* = v_rest + b/gl < v_th）
    bias_cv: float = 0.30             # 对数正态离散度
    seed: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_engine(p: ZFNetParams, device: str = "cpu", use_abs_slots: bool = True,
                 stp: Optional[Dict[str, Any]] = None,
                 gaba: Optional[Dict[str, Any]] = None,
                 g_mod: float = 1.0, use_compile: bool = False):
    """装配类级递归网络（确定性：固定 seed；无逐突触拟合）。"""
    from ..src.larval_zf_engine import LarvalZFEngine  # noqa: F401
    e = LarvalZFEngine(p.n_neuron, device=device, use_compile=use_compile,
                       verbose=False, use_abs_slots=use_abs_slots)
    rng = np.random.default_rng(p.seed)
    E = int(p.n_neuron) * int(p.indegree)
    pre = rng.integers(0, p.n_neuron, E)
    post = np.repeat(np.arange(p.n_neuron), p.indegree)
    rng.shuffle(post)
    inh = rng.random(E) < p.inh_frac
    gmax = np.where(inh, rng.uniform(0.8, 1.2, E) * p.g_inh,
                    rng.uniform(0.8, 1.2, E) * p.g_exc).astype(np.float32)
    dly = np.full(E, max(2, int(p.delay_max)), dtype=np.int32)   # 统一延迟（单组 ⇒ chunk 快路径）
    e.add_synapses(pre, post, gmax, dly, inhibitory=inh)
    e.finalize()
    bias = np.exp(rng.normal(math.log(p.bias_median_mv_s), p.bias_cv,
                             p.n_neuron)).astype(np.float32)
    e.set_neuron_heterogeneity(bias)
    if stp is not None:
        e.configure_stp(**stp)
    if gaba is not None:
        e.configure_gaba(**gaba)
    if g_mod != 1.0:
        e.configure_mod(g_mod=g_mod)
    return e


def make_events(T: int, lam_hz: float, n: int, seed: int = 3,
                g_ext: float = 20.0, slow_frac: float = 0.0) -> Dict[str, Any]:
    """Poisson 背景事件（**绝对步号**，E4 语义：跨分段一致）。

    lam_hz 为**逐神经元**平均背景率。抑制事件比例 = 0（背景为兴奋性驱动；M9 同语义）。
    """
    rng = np.random.default_rng(seed)
    t_s = T * DT_MS * 1e-3
    n_ev = int(round(lam_hz * t_s * n))
    steps = rng.integers(0, int(T), max(n_ev, 0)).astype(np.int64)
    neurons = rng.integers(0, n, max(n_ev, 0)).astype(np.int64)
    g = (rng.uniform(0.85, 1.15, max(n_ev, 0)) * g_ext).astype(np.float32)
    return dict(steps=steps, neurons=neurons, gmax=g, n_events=int(n_ev),
                lam_hz=lam_hz, g_ext=g_ext)


def bout_stats(pop_counts: np.ndarray, dt_ms: float = DT_MS, bin_ms: float = 100.0,
               min_bout_bins: int = 2):
    """bout 统计（**运行前定稿口径**，见模块 docstring）。pop_counts: 逐步全网络脉冲数。"""
    b = int(round(bin_ms / dt_ms))
    nb = pop_counts.size // b
    if nb < 4:
        return dict(bout_fraction=0.0, n_bouts=0, silent_bin_frac=1.0,
                    bout_median_s=0.0, nbins=nb)
    m = pop_counts[:nb * b].reshape(nb, b).sum(axis=1)
    silent = m == 0
    active = ~silent
    # **判据可证伪性修正（M10 实测，见 docs/m10_env_notes.md L19）**：
    # 首版把"记录起点/终点"也算作分隔 → **完全无静默的记录被算成 1 个贯穿全程的 bout**
    # （饱和网络的 bout_fraction = 1.0）⇒ 该判据无法证伪（M9 B1 纪律）。
    # 修正：bout **必须两侧都被静默箱界定**（interior bout）⇒ 无静默 ⇒ 0 bout ⇒ bout_fraction = 0。
    d = np.diff(np.r_[0, active.astype(np.int8), 0])
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    interior = (starts > 0) & (ends < nb)          # 两侧均有静默箱
    starts = starts[interior]; ends = ends[interior]
    lengths = ends - starts
    good = lengths >= min_bout_bins
    n_bouts = int(good.sum())
    in_bout = int(lengths[good].sum()) if n_bouts else 0
    med_s = float(np.median(lengths[good]) * bin_ms / 1000.0) if n_bouts else 0.0
    return dict(bout_fraction=float(in_bout / nb), n_bouts=n_bouts,
                silent_bin_frac=float(silent.mean()), bout_median_s=med_s, nbins=int(nb),
                mean_bin_spikes=float(m.mean()))


def run_protocol(engine, T_steps: int, settle_steps: int = 0, segment_steps: int = 20_000,
                 lam_hz: float = 0.0, ev_seed: int = 3, g_ext: float = 20.0,
                 record_pop: bool = True, progress=None) -> Dict[str, Any]:
    """分段执行协议（settle 亦分段）；返回判据量 + 逐段墙钟。"""
    n = engine.n
    ev = make_events(T_steps, lam_hz, n, seed=ev_seed, g_ext=g_ext)
    engine.set_external_events(ev["steps"], ev["neurons"], ev["gmax"], n_steps=T_steps)
    engine.reset(seed=1)
    engine.reset_counts()
    segs = []
    pop_chunks = []
    total = settle_steps + T_steps
    done = 0
    t_all = time.time()
    while done < total:
        w = min(segment_steps, total - done)
        t0 = time.time()
        r = engine.run(w, delivery="step", record="counts",
                       pop_trace=record_pop)
        segs.append(dict(step=done, w=w, wall_s=round(time.time() - t0, 3),
                         ms_per_step=round(r["ms_per_step"], 4)))
        if record_pop and r["pop"] is not None:
            pop_chunks.append(r["pop"])
        done += w
        if progress:
            progress(done, total, time.time() - t_all)
    counts = engine.t_count.detach().cpu().numpy().astype(np.float64)
    t_meas_s = T_steps * DT_MS * 1e-3
    rate = counts / t_meas_s
    pop = np.concatenate(pop_chunks) if pop_chunks else np.zeros(0)
    pop_meas = pop[settle_steps:] if pop.size >= settle_steps else pop
    out = dict(
        n_neuron=int(n), T_ms=T_steps * DT_MS, settle_ms=settle_steps * DT_MS,
        lam_hz=lam_hz, n_events=ev["n_events"], g_ext=g_ext,
        spk_total=float(counts.sum()),
        pop_rate_hz=float(counts.sum() / n / t_meas_s),
        rate_median_hz=float(np.median(rate)), rate_mean_hz=float(rate.mean()),
        rate_p95_hz=float(np.percentile(rate, 95)),
        silent_frac=float(np.mean(rate < 0.5)),
        n_nan=int(np.sum(~np.isfinite(rate))),
        wall_s=round(time.time() - t_all, 2),
        ms_per_step_steady=float(np.median([s["ms_per_step"] for s in segs[1:]] or
                                           [s["ms_per_step"] for s in segs])),
        segments=segs,
    )
    out.update(bout_stats(pop_meas))
    out["pop_trace_mean_hz"] = float(pop_meas.mean() / (DT_MS * 1e-3) / n) if pop_meas.size else 0.0
    return out


# --------------------------------------------------------------------------
# Fish1 实测锚（用于类级抽象的参数锚定与报告标注）
# --------------------------------------------------------------------------
def fish1_anchors() -> Dict[str, Any]:
    p = os.path.join(DATA_DIR, "m10_connectome_counts.json")
    d = {"source": "data/m10_connectome_counts.json"}
    try:
        import json
        with open(p) as f:
            c = json.load(f)
        d.update(roster=int(c["parse"]["n_roster"]),
                 cells=int(c["parse"]["n_chem_pairs_in_roster"]),
                 synapses=int(c["parse"]["n_chem_synapses_in_roster"]),
                 cell_type_counts=c["parse"]["cell_type_counts"],
                 completeness=c["completeness_level"])
    except Exception:  # pragma: no cover
        pass
    return d
