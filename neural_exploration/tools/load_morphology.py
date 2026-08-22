"""M1 形态学/通道规格加载器（清单 §2.3 验收：CSV 存在且可被本模块读入）。

读取 `data/m1_channel_map.csv` → `MorphologySpec`（区段树 + 逐隔室通道密度），
供两条构建路径共用：
  - Brian2（src/morphology.py  → SpatialNeuron）
  - NEURON（tools/build_neuron_ref.py → 参考解）

CSV 约定（每行一个隔室）：
  segment,compartment_index,parent,gna_mS_cm2,gk_mS_cm2,gl_mS_cm2,diameter_um,length_um
  parent = 本隔室所挂接的区段名；'root' 表示根（胞体）。
  区段出现顺序即树构建顺序；同一区段多隔室时父级为该区段自身（链式）。

单位约定（清单 §1 L4）：电导 mS/cm²、长度/直径 µm、时间 ms、电位 mV；
多隔室新增轴向电阻 Ra（Ω·cm，默认 150）与 Cm（µF/cm²，默认 1.0）。
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CSV = os.path.join(ROOT, "neural_exploration", "data", "m1_channel_map.csv")

# 全局电学参数（清单 §2.2）
RA = 150.0          # Ω·cm
CM = 1.0            # µF/cm²
ENA = 50.0          # mV
EK = -77.0          # mV
EL = -54.4          # mV
V0 = -65.0          # mV 初始膜电位


@dataclass
class Segment:
    """一个形态学区段（一节 Cylinder/Section）。"""

    name: str
    parent: str                      # 父区段名；'root' = 胞体
    n: int                           # 隔室数
    diameter_um: float               # µm（区段内恒定）
    length_um: float                 # µm（区段总长）
    gna_mS_cm2: float                # 区段内恒定
    gk_mS_cm2: float
    gl_mS_cm2: float
    cm_uF_cm2: float = 1.0           # 膜电容（髓鞘段降低模拟绝缘，默认 1.0）

    @property
    def is_soma(self) -> bool:
        return self.name == "soma"


@dataclass
class MorphologySpec:
    """整树规格：区段列表（按 CSV 出现顺序）+ 全局参数。"""

    segments: List[Segment] = field(default_factory=list)
    ra: float = RA
    cm: float = CM
    ena: float = ENA
    ek: float = EK
    el: float = EL
    v0: float = V0

    def by_name(self, name: str) -> Segment:
        for seg in self.segments:
            if seg.name == name:
                return seg
        raise KeyError(f"区段 {name} 不存在于形态学规格中")

    @property
    def total_compartments(self) -> int:
        return sum(s.n for s in self.segments)

    def dendrite_chain(self) -> List[Segment]:
        """树突链（自近端到远端）。"""
        out, cur = [], self.by_name("dend1")
        while True:
            out.append(cur)
            children = [s for s in self.segments if s.parent == cur.name and s.name != cur.name]
            if not children:
                break
            cur = children[0]
        return out

    def axon_chain(self) -> List[Segment]:
        """轴突链（AIS 起至末端）。"""
        out, cur = [], self.by_name("ais")
        while True:
            out.append(cur)
            children = [s for s in self.segments if s.parent == cur.name and s.name != cur.name]
            if not children:
                break
            cur = children[0]
        return out


def load_morphology(csv_path: Optional[str] = None) -> MorphologySpec:
    """读入 m1_channel_map.csv → MorphologySpec。

    校验：soma 存在且为根；每区段 parent 必须已出现（拓扑序合法）；
    每区段直径/长度/通道密度恒定。
    """
    path = csv_path or DEFAULT_CSV
    if not os.path.exists(path):
        raise FileNotFoundError(f"形态学 CSV 不存在：{path}")
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(row for row in f if not row.strip().startswith("#"))
        for r in reader:
            rows.append({
                "segment": r["segment"].strip(),
                "compartment_index": int(r["compartment_index"]),
                "parent": r["parent"].strip(),
                "gna": float(r["gna_mS_cm2"]),
                "gk": float(r["gk_mS_cm2"]),
                "gl": float(r["gl_mS_cm2"]),
                "cm": float(r.get("cm_uF_cm2", "1.0")),
                "diameter_um": float(r["diameter_um"]),
                "length_um": float(r["length_um"]),
            })

    # 按区段聚合
    seg_rows: Dict[str, List[dict]] = {}
    for r in rows:
        seg_rows.setdefault(r["segment"], []).append(r)

    seen = set()
    segments: List[Segment] = []
    for r in rows:  # CSV 顺序即构建顺序；用首个出现的 parent 判定
        name = r["segment"]
        if name in seen:
            continue
        comps = seg_rows[name]
        parent = comps[0]["parent"]
        if parent != "root" and parent not in seen:
            raise ValueError(f"区段 {name} 的父区段 {parent} 未先出现（CSV 需拓扑序）")
        # 校验一致性
        if len(comps) != comps[-1]["compartment_index"] + 1:
            raise ValueError(f"区段 {name} 的 compartment_index 不连续")
        d = {c["diameter_um"] for c in comps}
        gna = {c["gna"] for c in comps}
        if len(d) != 1 or len(gna) != 1:
            raise ValueError(f"区段 {name} 的直径/通道密度必须恒定")
        seg = Segment(
            name=name, parent=parent, n=len(comps),
            diameter_um=comps[0]["diameter_um"],
            length_um=sum(c["length_um"] for c in comps),
            gna_mS_cm2=comps[0]["gna"], gk_mS_cm2=comps[0]["gk"],
            gl_mS_cm2=comps[0]["gl"], cm_uF_cm2=comps[0]["cm"],
        )
        segments.append(seg)
        seen.add(name)

    return MorphologySpec(segments=segments)


if __name__ == "__main__":
    spec = load_morphology()
    print(f"区段数: {len(spec.segments)}，总隔室数: {spec.total_compartments}")
    print(f"Ra={spec.ra} Ω·cm, Cm={spec.cm} µF/cm²")
    for s in spec.segments:
        print(f"  {s.name:8s} n={s.n} d={s.diameter_um}µm L={s.length_um}µm "
              f"gNa={s.gna_mS_cm2} gK={s.gk_mS_cm2} gL={s.gl_mS_cm2} parent={s.parent}")
    dend = spec.dendrite_chain()
    ax = spec.axon_chain()
    print("树突链:", [s.name for s in dend])
    print("轴突链:", [s.name for s in ax])
    print("轴突总长(µm):", sum(s.length_um for s in ax))
