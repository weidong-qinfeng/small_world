"""M1 形态学构建：`MorphologySpec`（CSV）→ Brian2 `Soma`/`Cylinder` 树。

职责：
  1. 按规格的区段顺序构建 Soma + Cylinder 树（含递归子区段）；
  2. 返回 `SectionIndexMap`：区段名 → 该区段隔室在整树中的绝对索引数组，
     供逐隔室赋值通道密度、注入电流与记录；
  3. 与 NEURON 侧（tools/build_neuron_ref.py）共用同一 `MorphologySpec`，
     保证两引擎离散化逐隔室一致（P2 可比性的前提）。

Brian2 2.6.0 API 说明（M1 实测结论，记录在 m1_env_notes.md）：
  - Soma 为球体（面积 π·d²，无轴向电阻）；Cylinder 每隔室等长等径；
  - 根区段（soma）不参与命名子区段访问：用 `morpho._indices()`；
    子树用 `morpho['name']`（或 `morpho.name`）；
  - 逐隔室索引：`morpho['name']._indices()`（n=1 时返回标量，统一转数组）；
  - 2.6.0 不支持新版 `(membrane)` 标志：非 shared 变量天然逐隔室。
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.tools.load_morphology import MorphologySpec, Segment  # noqa: E402


@dataclass
class SectionIndexMap:
    """区段名 → 隔室绝对索引数组（整树扁平化后的下标）。"""

    index: Dict[str, np.ndarray] = field(default_factory=dict)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.index[name]

    def compartment_label(self, idx: int) -> str:
        for name, arr in self.index.items():
            if idx in arr:
                return name
        return "?"


def build_brian2_morphology(spec: MorphologySpec):
    """按 MorphologySpec 构建 Brian2 形态学树。

    Returns
    -------
    morpho : brian2.spatialneuron.morphology.Soma（根）
    index_map : SectionIndexMap
    """
    from brian2 import Soma, Cylinder, um

    # 根：soma（球体）
    soma_seg = spec.by_name("soma")
    morpho = Soma(diameter=soma_seg.diameter_um * um)

    # 区段顺序 = CSV 出现顺序（拓扑序已由 load_morphology 保证）
    built: Dict[str, object] = {"soma": morpho}
    for seg in spec.segments:
        if seg.is_soma:
            continue
        parent = built[seg.parent]
        cyl = Cylinder(diameter=seg.diameter_um * um,
                       length=seg.length_um * um,
                       n=seg.n)
        setattr(parent, seg.name, cyl)
        built[seg.name] = cyl

    # 区段 → 绝对索引（构建时索引与 SpatialNeuron 扁平化一致：确定性深度优先）
    index_map = SectionIndexMap()
    for seg in spec.segments:
        node = built[seg.name]
        idx = node._indices()
        index_map.index[seg.name] = np.atleast_1d(idx)
    return morpho, index_map


def apply_channel_densities(neuron, spec: MorphologySpec, index_map: SectionIndexMap):
    """按规格把 gNa/gK/gL 与 Cm 赋到各隔室（逐隔室赋值，避开子树跨段问题）。

    Cm 为逐隔室常量（Brian2 文档明确支持“useful to model myelinated axons”），
    髓鞘段由 CSV 的 cm_uF_cm2=0.02 降低电容负载。
    """
    from brian2 import mS, cm, uF

    for seg in spec.segments:
        for i in index_map[seg.name]:
            neuron[i].gNa = seg.gna_mS_cm2 * mS / cm ** 2
            neuron[i].gK = seg.gk_mS_cm2 * mS / cm ** 2
            neuron[i].gL = seg.gl_mS_cm2 * mS / cm ** 2
            neuron[i].Cm = seg.cm_uF_cm2 * uF / cm ** 2
    return neuron


def section_distances_micron(spec: MorphologySpec, index_map: SectionIndexMap) -> Dict[str, float]:
    """各区段（首个隔室中心）到胞体中心的距离（µm），用于传导速度计算。"""
    d: Dict[str, float] = {"soma": 0.0}
    # 沿树累积：距离 = 父区段距离 + 父区段半长 + 本区段半长
    for seg in spec.segments:
        if seg.is_soma:
            continue
        parent_dist = d[seg.parent]
        parent = spec.by_name(seg.parent)
        d[seg.name] = parent_dist + parent.length_um / 2.0 + seg.length_um / 2.0
    return d


if __name__ == "__main__":
    from neural_exploration.tools.load_morphology import load_morphology

    spec = load_morphology()
    morpho, idx = build_brian2_morphology(spec)
    print("形态学树构建 OK，总隔室:", morpho.total_compartments)
    for name, arr in idx.index.items():
        print(f"  {name:8s} indices={arr.tolist()}")
    print("区段距离(µm):", section_distances_micron(spec, idx))
