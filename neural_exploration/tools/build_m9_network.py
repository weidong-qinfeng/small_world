"""M9 全规模网络紧凑装配（2.4GB 连接组 CSV → npz 索引表；§3.5.1 前置）。

《生物仿真M9实施清单》§3.5.1（全规模构建）：139,255 点神经元 + 54.5M 化学突触
（15,091,983 条连接行，逐行 syn_count 加权）+ 0 缝隙（官方发布不可得，L19.2 裁决②）。

纪律（L16#1）：千万行级一律向量化（pyarrow 列选择 + numpy searchsorted 映射），
禁止逐行 Python 循环；流式/一次性写出 npz，避免 2.4GB CSV 反复解析。

递质类映射（真实标注，L14.3；不使用 hash 分配）：
  - cholinergic / glutamatergic → 兴奋性（ionotropic，类 0）
  - GABA → 抑制性（ionotropic，类 1）——**真实 GABA 12,755,910 突触（23.4%）**
  - dopaminergic / serotonergic / octopaminergic → 调质类（类 2）——P4 静息以弱兴奋电导
    承载（抽象登记：T2 调质显式建模在 P5+ 落地；此处登记为抽象 + 测量限制）

输出：
  - data/m9_network.npz：pre/post（int32 神经元索引）+ syn_count（int16）+ nt_code（int8）
    + root_ids（int64 神经元根 id）+ edge_row（int32 原始 CSV 行序，消融/审计用）
  - data/m9_network_stats.json：装配统计（边数/NT 分布/度分布/构建墙钟）
  - data/m9_neuron_table.npz：root_id + region 码 + dominant NT 码（角色映射用）

用法：
  PYTHONHASHSEED=0 MPLBACKEND=Agg .venv-m9/bin/python -m neural_exploration.tools.build_m9_network
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CSV_PATH = os.path.join(DATA_DIR, "m9_flywire_connectome.csv")
OUT_NPZ = os.path.join(DATA_DIR, "m9_network.npz")
OUT_NEURON = os.path.join(DATA_DIR, "m9_neuron_table.npz")
OUT_STATS = os.path.join(DATA_DIR, "m9_network_stats.json")

NT_CODES = {"cholinergic": 0, "glutamatergic": 0, "GABA": 1,
            "dopaminergic": 2, "serotonergic": 2, "octopaminergic": 2}
NT_NAMES = ["cholinergic", "glutamatergic", "GABA", "dopaminergic", "serotonergic",
            "octopaminergic"]
CLASS_NAMES = {0: "excitatory", 1: "inhibitory", 2: "modulatory"}


def _read_csv_streaming(path: str, chunk_rows: int = 2_000_000):
    """流式读取连接组 CSV（pyarrow 优先，pandas 回退）→ 逐块 yield dict of arrays。"""
    import pyarrow.csv as pcsv
    import pyarrow as pa
    t0 = time.perf_counter()
    skip = 0
    with open(path, "r", encoding="utf-8") as f:      # 前导注释行（# 头）需跳过
        for line in f:
            if line.startswith("#"):
                skip += 1
            else:
                break
    read_opts = pcsv.ReadOptions(block_size=64 << 20, skip_rows=skip)
    parse_opts = pcsv.ParseOptions(delimiter=",")
    conv_opts = pcsv.ConvertOptions(
        include_columns=["kind", "root_id", "synapse_from", "synapse_to",
                         "neurotransmitter", "syn_count", "region", "nt_confidence"],
        column_types={"kind": pa.string(), "root_id": pa.string(),
                      "synapse_from": pa.string(), "synapse_to": pa.string(),
                      "neurotransmitter": pa.string(), "syn_count": pa.int32(),
                      "region": pa.string(), "nt_confidence": pa.float32()},
        strings_can_be_null=True)
    n = 0
    with pcsv.open_csv(path, read_options=read_opts, parse_options=parse_opts,
                       convert_options=conv_opts) as reader:
        for batch in reader:
            t = pa.Table.from_batches([batch])
            yield {c: t.column(c).to_numpy(zero_copy_only=False) for c in t.column_names}
            n += t.num_rows
    print("  CSV 流式读取完成：%d 行，%.1fs" % (n, time.perf_counter() - t0), flush=True)


def build(force: bool = False) -> int:
    if os.path.exists(OUT_NPZ) and not force:
        print("已存在 →", OUT_NPZ, "（force=True 重建）", flush=True)
        return 0
    t0 = time.perf_counter()
    root_ids = None
    regions = None
    nt_dom = None
    nt_conf = None
    chem_from, chem_to, chem_syn, chem_nt, chem_row = [], [], [], [], []
    row_off = 0
    diag = {}
    for blk in _read_csv_streaming(CSV_PATH):
        kind = blk["kind"]
        is_neuron = kind == "neuron"
        is_chem = kind == "chem"
        if is_neuron.any():
            ids = blk["root_id"][is_neuron].astype(np.int64)
            root_ids = ids if root_ids is None else np.concatenate([root_ids, ids])
            reg = blk["region"][is_neuron].astype("U32")
            regions = reg if regions is None else np.concatenate([regions, reg])
            nd = blk["neurotransmitter"][is_neuron].astype("U20")
            nt_dom = nd if nt_dom is None else np.concatenate([nt_dom, nd])
            nc = blk["nt_confidence"][is_neuron].astype(np.float32)
            nt_conf = nc if nt_conf is None else np.concatenate([nt_conf, nc])
        if is_chem.any():
            nrow = int(is_chem.sum())
            chem_from.append(blk["synapse_from"][is_chem].astype(np.int64))
            chem_to.append(blk["synapse_to"][is_chem].astype(np.int64))
            chem_syn.append(blk["syn_count"][is_chem].astype(np.int32))
            chem_nt.append(blk["neurotransmitter"][is_chem].astype("U20"))
            chem_row.append(np.arange(row_off, row_off + nrow, dtype=np.int64))
        row_off += int(blk["kind"].shape[0])
    if root_ids is None or not chem_from:
        raise RuntimeError("CSV 解析为空——检查路径 %s" % CSV_PATH)
    print("  神经元 %d；化学行 %d；解析 %.1fs"
          % (root_ids.size, sum(a.size for a in chem_from), time.perf_counter() - t0),
          flush=True)

    # ---- 神经元索引映射（int64 统一 dtype，规避 L14.2#2 float64 精度陷阱） ----
    root_ids = np.asarray(root_ids, dtype=np.int64)
    order = np.argsort(root_ids, kind="stable")
    sorted_ids = root_ids[order]
    if np.unique(root_ids).size != root_ids.size:
        raise RuntimeError("root_id 非唯一——数据异常")
    pre_root = np.concatenate(chem_from)
    post_root = np.concatenate(chem_to)
    nt_str = np.concatenate(chem_nt)
    syn_cnt = np.concatenate(chem_syn)
    edge_rows = np.concatenate(chem_row)

    def _map(ids):
        pos = np.searchsorted(sorted_ids, ids)
        bad = (pos >= sorted_ids.size) | (sorted_ids[np.minimum(pos, sorted_ids.size - 1)] != ids)
        return pos.astype(np.int32), int(bad.sum())

    pre_idx, bad_pre = _map(pre_root)
    post_idx, bad_post = _map(post_root)
    if bad_pre or bad_post:
        raise RuntimeError("化学边节点不在 roster 内（%d/%d）" % (bad_pre, bad_post))

    nt_code = np.zeros(nt_str.size, dtype=np.int8)
    for k, v in NT_CODES.items():
        nt_code[nt_str == k] = v
    unknown = ~np.isin(nt_str, list(NT_CODES.keys()))
    if unknown.any():
        diag["unknown_nt_rows"] = int(unknown.sum())

    # ---- 神经元表：region 码 + dominant NT 码 ----
    reg_codes, reg_uniques = _encode_categories(regions[order])
    dom_code = np.zeros(root_ids.size, dtype=np.int8)   # = 神经元行 neurotransmitter
    neuron_nt = nt_dom[order]
    for k, v in NT_CODES.items():
        dom_code[neuron_nt == k] = v
    dom_code[neuron_nt == ""] = -1

    np.savez_compressed(OUT_NPZ, pre=pre_idx, post=post_idx, syn_count=syn_cnt.astype(np.int16),
                        nt_code=nt_code, root_ids=sorted_ids, edge_row=edge_rows.astype(np.int32),
                        nt_str=nt_str.astype("U20"))
    np.savez_compressed(OUT_NEURON, root_ids=sorted_ids, region_code=reg_codes,
                        region_names=np.asarray(reg_uniques, dtype="U32"),
                        dom_nt_code=dom_code, nt_confidence=nt_conf[order].astype(np.float32))
    print("  → %s / %s（%.1fs）" % (OUT_NPZ, OUT_NEURON, time.perf_counter() - t0), flush=True)

    # ---- 统计 ----
    outdeg = np.bincount(pre_idx, minlength=root_ids.size)
    indeg = np.bincount(post_idx, minlength=root_ids.size)
    stats = {
        "generated_by": "tools/build_m9_network.py",
        "n_neurons": int(root_ids.size),
        "n_chem_connections": int(pre_idx.size),
        "n_chem_synapses": int(syn_cnt.sum()),
        "n_gap": 0,
        "nt_class_edge_counts": {
            CLASS_NAMES[c]: int((nt_code == c).sum()) for c in (0, 1, 2)},
        "nt_class_synapse_counts": {
            CLASS_NAMES[c]: int(syn_cnt[nt_code == c].sum()) for c in (0, 1, 2)},
        "inhibitory_edge_fraction_pct": float(100.0 * (nt_code == 1).mean()),
        "syn_count": {"mean": float(syn_cnt.mean()), "median": float(np.median(syn_cnt)),
                      "max": int(syn_cnt.max())},
        "degree": {"out_mean": float(outdeg.mean()), "out_p99": float(np.percentile(outdeg, 99)),
                   "out_max": int(outdeg.max()), "in_mean": float(indeg.mean()),
                   "in_p99": float(np.percentile(indeg, 99)), "in_max": int(indeg.max()),
                   "isolated": int(((outdeg == 0) & (indeg == 0)).sum())},
        "region_coverage_pct": float(100.0 * np.mean(
            np.asarray([r != "" for r in regions[order]]))),
        "n_regions": int(len(reg_uniques)),
        "region_top": _top_counts(regions[order], k=15),
        "diagnostics": diag,
        "build_wall_s": time.perf_counter() - t0,
    }
    with open(OUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("  → %s" % OUT_STATS, flush=True)
    print("  化学边 %d（兴奋 %d / 抑制 %d / 调质 %d）；突触 %d；出度中位 %.0f 最大 %d"
          % (pre_idx.size, stats["nt_class_edge_counts"]["excitatory"],
             stats["nt_class_edge_counts"]["inhibitory"],
             stats["nt_class_edge_counts"]["modulatory"],
             stats["n_chem_synapses"], np.median(outdeg), outdeg.max()), flush=True)
    return 0


def _encode_categories(arr):
    uniq, inv = np.unique(np.asarray(arr), return_inverse=True)
    return inv.astype(np.int16), uniq


def _top_counts(arr, k=15):
    u, c = np.unique(np.asarray(arr), return_counts=True)
    o = np.argsort(-c)[:k]
    return {str(u[i]): int(c[i]) for i in o}


if __name__ == "__main__":
    sys.exit(build(force="--force" in sys.argv))
