"""M9 抑制/递质清单（真实 GABA 抑制边承接 M8 R2/H6；P4 静息 sanity 与 P5/P8 机制前置）。

《生物仿真M9实施清单》§0.2 R2/R6 承接 + §1.2 递质完整性 + §3.5.3 双状态（真实抑制边）。

M8 反证 R2：inter 递质经 class hash 回退分配（非真实）→ 抑制平衡缺失。M9 v783 官方发布
含 Eckstein 2024 逐连接递质预测 → **真实抑制边可得**。本清单把该事实整理为下游可直接消费的
机器可读表（引擎装配/权重校准/H6 消融判据用）：

输出 `data/m9_inhibition_inventory.csv`（行 = 指标，列 = 数值/来源）：
  1. 逐递质突触总量/连接数/占比（含真实 GABA 抑制边规模）；
  2. 受体映射分布（gaba 抑制 / ampa 兴奋 / mod 调质）——GPU 引擎装配的边类型计数；
  3. 神经元级：dominant 递质构成（出边）、GABA 能神经元数、接收 GABA 输入的神经元数；
  4. GABA 边 neuropil 分布 top-N（结构来源）；
  5. H6 消融接口提示（抑制边 g→0 的边集规模）。

用法：PYTHONHASHSEED=0 MPLBACKEND=Agg .venv-m9/bin/python -m neural_exploration.tools.gen_m9_inhibition_inventory
"""

from __future__ import annotations

import csv
import os
from collections import Counter

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CONNECTOME = os.path.join(DATA_DIR, "m9_flywire_connectome.csv")
OUT = os.path.join(DATA_DIR, "m9_inhibition_inventory.csv")

RECEPTOR_OF = {"GABA": "gaba", "cholinergic": "ampa", "glutamatergic": "ampa",
               "dopaminergic": "mod", "serotonergic": "mod", "octopaminergic": "mod"}


def read_chem_columns():
    """流式读连接组 CSV 的 chem 行（只取所需列；1.89GB 文件分块）。"""
    import pyarrow.csv as pacsv
    cols = ["kind", "neurotransmitter", "syn_count", "neuropil",
            "synapse_from", "synapse_to"]
    kinds, nts, syns, nps, pres, posts = [], [], [], [], [], []
    ro = pacsv.ReadOptions(block_size=1 << 26, skip_rows=5)  # 跳过 5 行 # 注释
    po = pacsv.ParseOptions(delimiter=",")
    co = pacsv.ConvertOptions(include_columns=cols)
    with open(CONNECTOME, "rb") as f:
        reader = pacsv.open_csv(f, read_options=ro, parse_options=po,
                                convert_options=co)
        for batch in reader:
            t = batch.to_pydict()
            for i, k in enumerate(t["kind"]):
                if k == "chem":
                    kinds.append(0)
                    nts.append(t["neurotransmitter"][i])
                    syns.append(t["syn_count"][i])
                    nps.append(t["neuropil"][i])
                    pres.append(t["synapse_from"][i])
                    posts.append(t["synapse_to"][i])
    return (np.asarray(nts, dtype=object), np.asarray(syns, dtype=np.int64),
            np.asarray(nps, dtype=object), np.asarray(pres, dtype=np.int64),
            np.asarray(posts, dtype=np.int64))


def main() -> int:
    print("=== M9 抑制/递质清单（gen_m9_inhibition_inventory.py）===", flush=True)
    nt, syn, npil, pre, post = read_chem_columns()
    n_pairs = int(nt.size)
    total_syn = int(syn.sum())
    print("化学连接行：%d；突触总量：%d" % (n_pairs, total_syn), flush=True)

    nt_order = ["cholinergic", "GABA", "glutamatergic", "dopaminergic",
                "serotonergic", "octopaminergic"]
    rows = [["category", "item", "metric", "value", "share_pct", "note"]]

    # 1) 逐递质总量/连接数
    for k in nt_order:
        m = nt == k
        rows.append(["neurotransmitter", k, "synapses", int(syn[m].sum()),
                     round(100.0 * syn[m].sum() / total_syn, 3),
                     "Eckstein 2024 逐连接 dominant 递质（v783 官方发布）"])
        rows.append(["neurotransmitter", k, "connections", int(m.sum()),
                     round(100.0 * m.sum() / n_pairs, 3), ""])

    # 2) 受体映射分布（GPU 引擎边类型）
    rec = np.array([RECEPTOR_OF[x] for x in nt], dtype=object)
    for r in ("gaba", "ampa", "mod"):
        m = rec == r
        rows.append(["receptor", r, "connections", int(m.sum()),
                     round(100.0 * m.sum() / n_pairs, 3),
                     "gaba=抑制性离子通道 / ampa=兴奋性占位 / mod=调质门控（M6 语义）"])
        rows.append(["receptor", r, "synapses", int(syn[m].sum()),
                     round(100.0 * syn[m].sum() / total_syn, 3), ""])

    # 3) 神经元级（出边 dominant）
    pre_nt = {}
    pre_syn = {}
    for i in range(n_pairs):
        p = int(pre[i])
        # 按突触数加权的 dominant（与 build 一致：逐 pre 统计）
        pre_syn[p] = pre_syn.get(p, 0) + int(syn[i])
        d = pre_nt.setdefault(p, Counter())
        d[nt[i]] += int(syn[i])
    n_pre = len(pre_nt)
    dom = Counter()
    for p, c in pre_nt.items():
        dom[c.most_common(1)[0][0]] += 1
    for k in nt_order:
        if dom[k]:
            rows.append(["neuron_output", k, "neurons_dominant", int(dom[k]),
                         round(100.0 * dom[k] / n_pre, 3),
                         "出边 syn 加权 dominant 递质（有出边神经元 %d 个）" % n_pre])
    # GABA 输入侧
    gaba_edges = nt == "GABA"
    gaba_post_neurons = int(np.unique(post[gaba_edges]).size)
    gaba_pre_neurons = int(np.unique(pre[gaba_edges]).size)
    rows.append(["inhibition", "GABA edges", "connections", int(gaba_edges.sum()),
                 round(100.0 * gaba_edges.sum() / n_pairs, 3), "真实抑制边（M8 R2 承接）"])
    rows.append(["inhibition", "GABA edges", "synapses",
                 int(syn[gaba_edges].sum()),
                 round(100.0 * syn[gaba_edges].sum() / total_syn, 3),
                 "H6 消融接口：该边集 g→0 = 抑制平衡消融（§3.5.5）"])
    rows.append(["inhibition", "GABA presynaptic neurons", "neurons",
                 gaba_pre_neurons, 0.0, "GABA 能输出神经元数"])
    rows.append(["inhibition", "GABA-receiving neurons", "neurons",
                 gaba_post_neurons, 0.0, "接收至少 1 条 GABA 边的神经元数"])

    # 4) GABA 边 neuropil 分布 top-12
    cnt = Counter()
    for i in np.nonzero(gaba_edges)[0]:
        cnt[str(npil[i])] += int(syn[i])
    for name, v in cnt.most_common(12):
        rows.append(["GABA_by_neuropil", name, "synapses", int(v),
                     round(100.0 * v / max(syn[gaba_edges].sum(), 1), 3),
                     "GABA 突触的脑区来源（结构分布）"])

    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        f.write("# M9 抑制/递质清单（真实抑制边承接 M8 R2；P4 双状态 H6 消融接口）\n")
        f.write("# 来源：data/m9_flywire_connectome.csv（Zenodo 10676866 v783 官方解析，"
                "Eckstein 2024 递质预测）\n")
        w.writerows(rows)
    print("→", OUT, flush=True)
    g = nt == "GABA"
    print("真实 GABA：%d 突触（%.1f%%）/ %d 连接；GABA 能神经元 %d；受 GABA 神经元 %d" % (
        int(syn[g].sum()), 100.0 * syn[g].sum() / total_syn, int(g.sum()),
        gaba_pre_neurons, gaba_post_neurons), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
