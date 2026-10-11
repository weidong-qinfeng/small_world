"""M11 分层柱装配（**引擎门夹具**）——`src/mouse_cortex_engine.py` 的可测输入。

《生物仿真M11实施清单》§2.2（E1b 四亚型槽 / E3b 层流 + 层间延迟打包）、§0.10 L1
（`src/mouse_cortex_circuit.py` = 分层柱装配）。

> **定位（务必区分，防越界）**：本模块是 **G0 引擎门的最小可测夹具**，只保证
> ① 层标签 / ② 每层 E:I 比例 / ③ E/PV/SST/VIP 四类标签 / ④ 层间延迟量化为 ≤4 档（打包）
> 这四项**对内核可见**，用于 E1b/E3b 的编码路径断言与阳性对照。
> **它不是 P4 的交付物**：与 MICrONS/文献连接概率的逐格一致性检验（P4 判据）**不在本节点范围**。
> 全部比例/权重/延迟值为**预注册抽象默认值**（`evidence=abstraction_default`），
> **不得**当作判据带或 `measured` 数据（清单 §0.7 #2 禁令）。

用法：
    from neural_exploration.src.mouse_cortex_circuit import build_laminar_probe_column
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

LAYER_NAMES = ("L2/3", "L4", "L5", "L6")
# 抽象默认（预注册；**pending G2 prior 定稿**，不得作判据带）
DEFAULT_LAYER_FRAC = (0.35, 0.15, 0.30, 0.20)     # 每层层内神经元占比
DEFAULT_E_FRAC = (0.80, 0.80, 0.80, 0.85)         # 每层兴奋细胞占该层比例
DEFAULT_SUB_FRAC = (0.35, 0.30, 0.35)             # PV / SST / VIP 占中间神经元比例
# 层间延迟档（步；dt=0.05 ms ⇒ 2 / 3 / 4 ms）——**档数 3 ≤ MAX_DELAY_BINS=4**（E3b 打包）
DEFAULT_DELAY_BINS = (40, 60, 80)
# 类级权重带（1/s）；E→E 取 M10 探针带，抑制类按 PV>SST>VIP 排序（抽象默认）
DEFAULT_GMAX_BAND = dict(E=(6.0, 18.0), PV=(40.0, 90.0), SST=(20.0, 60.0), VIP=(8.0, 24.0))

SUB_E, SUB_PV, SUB_SST, SUB_VIP = 0, 1, 2, 3
SUB_NAMES = ("E", "PV", "SST", "VIP")


def build_laminar_probe_column(n_total: int, indegree: int = 20, seed: int = 0,
                               layer_frac: Sequence[float] = DEFAULT_LAYER_FRAC,
                               e_frac: Sequence[float] = DEFAULT_E_FRAC,
                               sub_frac: Sequence[float] = DEFAULT_SUB_FRAC,
                               delay_bins: Sequence[int] = DEFAULT_DELAY_BINS,
                               inh_frac: float = 0.20,
                               gmax_band: Optional[Dict[str, Any]] = None,
                               gmax_scale: float = 1.0,
                               bias_log_mean: float = 230.0,
                               bias_log_sigma: float = 0.30,
                               bias_scale: float = 1.0) -> Dict[str, Any]:
    """装配一个"分层柱"探针网络（**确定性**：固定 seed；禁 Python `hash()`）。

    返回字典：`pre/post/gmax/delay/syn_class/inhibitory`（边表）+ `layer_of/type_of/bias`
    （逐神经元）+ `meta`（统计与装配参数，供报告与断言复核）。
    """
    n_total = int(n_total)
    if n_total < 8:
        raise ValueError("n_total 过小")
    rng = np.random.default_rng(int(seed))
    lf = np.asarray(layer_frac, dtype=np.float64)
    lf = lf / lf.sum()
    ef = np.asarray(e_frac, dtype=np.float64)
    sf = np.asarray(sub_frac, dtype=np.float64)
    sf = sf / sf.sum()
    dly_bins = np.asarray(sorted(int(x) for x in np.asarray(delay_bins).ravel()), dtype=np.int32)

    # ① 层分配
    counts = np.floor(lf * n_total).astype(np.int64)
    counts[0] += n_total - int(counts.sum())
    layer_of = np.concatenate([np.full(int(c), i, dtype=np.int64) for i, c in enumerate(counts)])

    # ② 层内类型（E / PV / SST / VIP）
    type_of = np.zeros(n_total, dtype=np.int64)
    for i in range(len(LAYER_NAMES)):
        idx = np.flatnonzero(layer_of == i)
        n_e = int(round(len(idx) * ef[i]))
        perm = rng.permutation(len(idx))
        type_of[idx[perm[:n_e]]] = SUB_E
        inter = idx[perm[n_e:]]
        if inter.size:
            u = rng.random(inter.size)
            type_of[inter] = np.where(u < sf[0], SUB_PV,
                                      np.where(u < sf[0] + sf[1], SUB_SST, SUB_VIP))
    is_inh = type_of > 0

    # ③ 边表：逐突触后神经元抽 indegree 个突触前（层邻近度加权 + 类型加权）
    pre_l, post_l, dly_l, cls_l = [], [], [], []
    lay_src = [np.flatnonzero(layer_of == i) for i in range(len(LAYER_NAMES))]
    lay_src = [x if x.size else np.arange(n_total) for x in lay_src]
    for post in range(n_total):
        pl = int(layer_of[post])
        # 层邻近度 → 延迟档（同层=档0；相邻层=档1；隔层=档2）
        dist = np.abs(np.arange(len(LAYER_NAMES)) - pl)
        bin_idx = np.clip(dist, 0, len(dly_bins) - 1)
        w = np.array([0.45, 0.25, 0.15, 0.10][:len(LAYER_NAMES)], dtype=np.float64)
        w = w * (1.0 / (1.0 + dist))
        w = w / w.sum()
        k = int(indegree)
        src = np.concatenate([rng.choice(lay_src[i], size=max(1, int(round(k * w[i]))), replace=True)
                              for i in range(len(LAYER_NAMES))])[:k]
        sbin = np.concatenate([np.full(max(1, int(round(k * w[i]))), bin_idx[i]) 
                               for i in range(len(LAYER_NAMES))])[:k]
        for s, b in zip(src, sbin):
            inh = bool(rng.random() < inh_frac)
            if inh:
                cls = int(rng.choice([SUB_PV, SUB_SST, SUB_VIP], p=list(sf)))
            else:
                cls = SUB_E
            pre_l.append(int(s)); post_l.append(int(post)); cls_l.append(cls)
            dly_l.append(int(dly_bins[int(b)]))
    pre = np.asarray(pre_l, dtype=np.int64)
    post = np.asarray(post_l, dtype=np.int64)
    dly = np.asarray(dly_l, dtype=np.int32)
    cls = np.asarray(cls_l, dtype=np.int8)
    band = dict(DEFAULT_GMAX_BAND if gmax_band is None else gmax_band)
    lo = np.array([band[SUB_NAMES[c]][0] for c in cls], dtype=np.float64)
    hi = np.array([band[SUB_NAMES[c]][1] for c in cls], dtype=np.float64)
    gmax = (lo + (hi - lo) * rng.random(pre.size)).astype(np.float32) * float(gmax_scale)
    inh_flag = (cls > 0)

    # ④ 亚阈 tonic 偏置（异质；用于密度标定）
    bias = (np.exp(rng.normal(np.log(max(1e-6, float(bias_log_mean))), float(bias_log_sigma),
                              n_total)) * float(bias_scale)).astype(np.float32)

    meta = dict(n_total=n_total, indegree=int(indegree), seed=int(seed), n_edge=int(pre.size),
                layer_frac=[float(x) for x in lf], e_frac=[float(x) for x in ef],
                sub_frac=[float(x) for x in sf], inh_frac=float(inh_frac),
                delay_bins=[int(x) for x in dly_bins],
                gmax_scale=float(gmax_scale), bias_scale=float(bias_scale),
                evidence="abstraction_default(pending G2 prior)",
                layer_counts={LAYER_NAMES[i]: int(counts[i]) for i in range(len(LAYER_NAMES))},
                type_counts={SUB_NAMES[c]: int((type_of == c).sum()) for c in range(4)},
                class_edge_counts={SUB_NAMES[c]: int((cls == c).sum()) for c in range(4)},
                delay_edge_counts={int(d): int((dly == d).sum()) for d in np.unique(dly)},
                is_probe_fixture=True)
    return dict(pre=pre, post=post, gmax=gmax, delay=dly, syn_class=cls, inhibitory=inh_flag,
                layer_of=layer_of, type_of=type_of, bias=bias, meta=meta)


def attach_to_engine(engine, net: Dict[str, Any]) -> Dict[str, Any]:
    """把夹具网络装配进 M11 内核（**逐类 syn_class 必须显式传入**——M10 L21 硬条款）。"""
    engine.add_synapses(net["pre"], net["post"], net["gmax"], net["delay"],
                        inhibitory=net["inhibitory"], syn_class=net["syn_class"])
    st = engine.finalize()
    engine.set_neuron_heterogeneity(net["bias"])   # **必须在 finalize 之后**（finalize 会清零 t_iext）
    return st
