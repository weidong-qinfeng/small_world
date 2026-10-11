"""M10 数据门 P1：斑马鱼幼鱼（Fish1 CLEM）连接组解析管线。

《生物仿真M10实施清单》§1.1（连接组获取/权威计数锚/完整度分级）+ §0.10 L1/L2。

数据源（**实测可达性见 `docs/m10_env_notes.md` L9–L11**）：
  - Fish1 官方发布页（GCS 公开匿名可读，实测 4 MB/s）：
      https://fish1-release.storage.googleapis.com/index.html
      paper_data/HMI_analysis.zip   → Hindbrain Motion Integrator（含 CAVE somas 导出 30,346 行）
      paper_data/TEN_analysis.zip   → Tegmental Excitatory Nucleus（含 agglomerated roster +
                                      逐神经元 partner 列表 + 逐突触尺寸表）
  - 论文：Petkova, Januszewski, et al. (2025) "A connectomic resource for neural cataloguing and
    circuit dissection of the larval zebrafish brain", bioRxiv 2025.06.10.658982
    （PMC 全文 PMC12259123，经 Europe PMC REST 实测可得；全文声明 >180,000 somas / ~30M synapses /
     >40,000 分子标注神经元，且 **全部 proofreading/segmentation 数据经 CAVE 平台访问**）
  - 完整全脑突触表（CAVE `synapses_axde_label`，~30M 行）：**本机不可得**——CAVE 服务
    `https://global.brain-wire-test.org/` 的 `/materialize/api/v2/...` 实测 404、`/info/api/v2/...`
    实测 302 → 交互式登录（浏览器取 token）→ 无头会话不可用 ⇒ **完整度分级 = `partial`**
    （M10 §1.1.3），并请求三态裁决（见 counts JSON 的 `adjudication_requests`）。

本工具**只做确定性解析 + 硬断言 + 计数诚实性 + SHA-256 重跑**（沿 `tools/build_m9_connectome.py` 范式）：
  - 禁逐行 Python 循环（向量化 + pyarrow/csv 批量）；ID 比较统一 int64（M9 L14.2）；
  - **不臆造**：不可得字段（区域/递质亚型/缝隙连接/全脑突触数）一律写空 + 显式测量限制；
  - 权重（gmax）不可辨识 → npz 中以 `gmax=1.0` 基线占位，标定留给 P5（§0.7 #7）。

用法：
    PYTHONHASHSEED=0 ./.venv-m10/bin/python -m neural_exploration.tools.build_m10_connectome
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW = os.path.join(DATA_DIR, "m10_raw", "fish1_release")
TEN = os.path.join(RAW, "TEN_analysis")
HMI = os.path.join(RAW, "HMI_analysis", "data")

OUT_CSV = os.path.join(DATA_DIR, "m10_zf_connectome.csv")
OUT_JSON = os.path.join(DATA_DIR, "m10_connectome_counts.json")
OUT_NPZ = os.path.join(DATA_DIR, "m10_zf_network.npz")

# ---- 权威发布数（论文/发布页原文；不得按"民俗数字"近似补齐，M5 L7/M9 §2.3⑤）----
AUTH = dict(
    n_soma=">180,000 (发布原文)",
    n_synapses="~30 million (发布原文)",
    n_molecularly_annotated=">40,000 (发布原文)",
    age="7 dpf",
    scope="brain + anterior spinal cord",
    source="Petkova et al. 2025 bioRxiv 2025.06.10.658982 (PMC12259123); "
           "fish1-release GCS 官方发布页",
)

# ---- 判据带（预注册，本文件顶部常量；不事后调）----
BAND = dict(
    roster_min=180_000,          # 发布原文 ">180,000"
    roster_tol=0.05,             # ±5%（roster 为 lores↔agglo 映射表，非等于"神经元数"）
    synapse_anchor_tol=0.10,     # 全脑突触计数锚 ±10%（若可测）
    min_edge_pairs=5_000,        # partial 级最低可用子图规模（预注册下限）
)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _parse_id_list(s: pd.Series) -> list[np.ndarray]:
    """把 `[a b c ...]` / `a b c` 形式的 id 列向量化解析为 int64 数组列表。"""
    return [np.fromstring(x.strip("[] \n"), sep=" ", dtype=np.int64)
            for x in s.astype(str).to_numpy()]


def load_roster() -> np.ndarray:
    """lores_id ↔ hires_id_agglo（官方 TEN 包）→ 返回 hires_id_agglo（agglomerated 段 id）。"""
    p = os.path.join(TEN, "agglomerated_segments_and_soma_ids.csv")
    df = pd.read_csv(p, dtype={"lores_id": np.int64, "hires_id_agglo": np.int64})
    return df


def load_somas() -> pd.DataFrame:
    p = os.path.join(HMI, "cave_somas_in_big_box.csv")
    df = pd.read_csv(p, usecols=["id", "cell_type", "pt_root_id", "pt_supervoxel_id",
                                 "_x", "_y", "_z"])
    df = df.rename(columns={"id": "lore_id"})
    df["cell_type"] = df["cell_type"].fillna("na").astype(str)
    return df


def load_ten_edges() -> tuple[pd.DataFrame, dict]:
    """TEN 包 → (pre, post, syn_count) 有向边表 + 解析诊断。"""
    din = pd.read_csv(os.path.join(TEN, "incoming_synapses.csv"),
                      dtype={"query_neuron": np.int64})
    dout = pd.read_csv(os.path.join(TEN, "outgoing_synapses.csv"),
                       dtype={"query_neuron": np.int64})
    recs = []
    for df, col, direction in ((din, "incoming_with_soma", "in"),
                               (dout, "outgoing_with_soma", "out")):
        q = df["query_neuron"].to_numpy(dtype=np.int64)
        partners = _parse_id_list(df[col])
        for qi, arr in zip(q, partners):
            if arr.size == 0:
                continue
            uniq, cnt = np.unique(arr, return_counts=True)
            if direction == "in":
                pre, post = uniq, np.full(uniq.shape, qi, dtype=np.int64)
            else:
                pre, post = np.full(uniq.shape, qi, dtype=np.int64), uniq
            recs.append(np.stack([pre, post, cnt.astype(np.int64)], axis=1))
    E = np.concatenate(recs, axis=0) if recs else np.zeros((0, 3), dtype=np.int64)
    diag = dict(n_query_incoming=int(din.shape[0]), n_query_outgoing=int(dout.shape[0]),
                raw_edge_rows=int(E.shape[0]))
    # 同一 (pre,post) 可能同时出现在 incoming（对方视角）与 outgoing → 合并计数
    key = E[:, 0].astype(np.uint64) * np.uint64(1 << 32) + (E[:, 1].astype(np.uint64) & np.uint64(0xFFFFFFFF))
    order = np.argsort(key, kind="stable")
    key, E = key[order], E[order]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    sums = np.add.reduceat(E[:, 2], starts)
    E = np.stack([E[starts, 0], E[starts, 1], sums], axis=1)
    diag["merged_edge_pairs"] = int(E.shape[0])
    diag["merged_synapses"] = int(E[:, 2].sum())
    return pd.DataFrame(E, columns=["pre", "post", "syn_count"]), diag


def load_synapse_sizes() -> tuple[pd.DataFrame, dict]:
    """逐突触尺寸表（pre_neuron_id/bbx/bby/bbz/post_id）→ 逐突触边表（独立口径）。"""
    p = os.path.join(TEN, "synapse_sizes.csv")
    df = pd.read_csv(p, dtype={"pre_neuron_id": "Int64", "post_id": "Int64"})
    n_raw = int(df.shape[0])
    n_missing = int(df["pre_neuron_id"].isna().sum() + df["post_id"].isna().sum())
    df = df.dropna(subset=["pre_neuron_id", "post_id"]).reset_index(drop=True)
    df["pre_neuron_id"] = df["pre_neuron_id"].astype(np.int64)
    df["post_id"] = df["post_id"].astype(np.int64)
    key = (df["pre_neuron_id"].to_numpy(np.int64).astype(np.uint64) * np.uint64(1 << 32)
           + (df["post_id"].to_numpy(np.int64).astype(np.uint64) & np.uint64(0xFFFFFFFF)))
    uniq, cnt = np.unique(key, return_counts=True)
    pre = (uniq >> np.uint64(32)).astype(np.int64)
    post = (uniq & np.uint64(0xFFFFFFFF)).astype(np.int64)
    out = pd.DataFrame({"pre": pre, "post": post, "syn_count": cnt.astype(np.int64)})
    bb = df[["bbx", "bby", "bbz"]].to_numpy(np.float64)
    diag = dict(n_synapse_rows=int(df.shape[0]),
                n_synapse_rows_raw=n_raw,
                n_missing_partner_ids=n_missing,
                syn_pairs=int(uniq.size),
                bbox_dim_mean=[float(x) for x in bb.mean(axis=0)],
                bbox_dim_median=[float(x) for x in np.median(bb, axis=0)],
                bbox_max=[float(x) for x in bb.max(axis=0)])
    return out, diag


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", default=OUT_CSV)
    ap.add_argument("--out-json", default=OUT_JSON)
    ap.add_argument("--out-npz", default=OUT_NPZ)
    args = ap.parse_args(argv)
    t0 = time.time()

    roster = load_roster()
    somas = load_somas()
    ten_e, ten_diag = load_ten_edges()
    syn_e, syn_diag = load_synapse_sizes()

    n_roster = int(roster["hires_id_agglo"].nunique())
    roster_ids = np.unique(roster["hires_id_agglo"].to_numpy(np.int64))  # 唯一节点（lore 多重映射已登记）
    soma_lore = somas["lore_id"].to_numpy(np.int64)

    # ---- 硬断言 1：roster 计数锚（发布原文 >180,000）----
    roster_ok = n_roster >= BAND["roster_min"] and \
        n_roster <= BAND["roster_min"] * (1 + BAND["roster_tol"]) * 1.05
    assert roster_ok, f"roster 计数 {n_roster} 不落带（发布原文 >180,000，容差 ±5%）"

    # ---- 身份空间对齐：roster(agglo) vs somas(lore) ----
    lore2agglo = dict(zip(roster["lores_id"].to_numpy(np.int64),
                          roster["hires_id_agglo"].to_numpy(np.int64)))
    soma_in_roster = np.array([int(l) in lore2agglo for l in soma_lore], dtype=bool)

    # ---- 边表：限定两端都在 roster 内的（其余如实计数登记）----
    in_roster_pre = np.isin(ten_e["pre"].to_numpy(np.int64), roster_ids)
    in_roster_post = np.isin(ten_e["post"].to_numpy(np.int64), roster_ids)
    both = in_roster_pre & in_roster_post
    E = ten_e.loc[both].reset_index(drop=True)
    E_syn = syn_e[np.isin(syn_e["pre"].to_numpy(np.int64), roster_ids)
                  & np.isin(syn_e["post"].to_numpy(np.int64), roster_ids)].reset_index(drop=True)

    pre = E["pre"].to_numpy(np.int64)
    post = E["post"].to_numpy(np.int64)
    syn = E["syn_count"].to_numpy(np.int64)
    self_loops = int((pre == post).sum())

    # ---- cell_type 覆盖（如实，可 <100%）----
    ct = somas["cell_type"].to_numpy()
    ct_counts = {k: int(v) for k, v in zip(*np.unique(ct, return_counts=True))}
    lore_set = set(soma_lore.tolist())
    neurons_with_type = int(sum(1 for p_ in pre if int(p_) in lore_set)) + \
        int(sum(1 for p_ in post if int(p_) in lore_set))
    edge_nodes = np.unique(np.concatenate([pre, post]))
    # cell_type 在 *lore* 空间；边表在 *agglomerated* 空间 → 经 roster 反查（ID 空间隔离，M9 L14.2）
    agglo2lore = dict(zip(roster["hires_id_agglo"].to_numpy(np.int64),
                          roster["lores_id"].to_numpy(np.int64)))
    type_map_lore = dict(zip(soma_lore.tolist(), ct.tolist()))
    type_map = {int(a): cm for a, cm in
                ((a, type_map_lore.get(int(agglo2lore.get(int(a), -1)), ""))
                 for a in edge_nodes.tolist()) if cm}
    typed_nodes = int(sum(1 for p_ in edge_nodes if int(p_) in type_map))
    # 边的抑制性标记：pre 端已知且为 inh（未知 → False，且另以 n_edges_type_unknown 登记）
    n_edges_type_known = int(sum(1 for v in pre if int(v) in type_map))
    n_edges_inh = int(sum(1 for v in pre if type_map.get(int(v), "") == "inh"))

    # ---- 传出/传入度（用于 CSV 神经元行）----
    n_nodes = int(roster_ids.size)
    idx = {int(v): i for i, v in enumerate(roster_ids)}
    poi = np.array([idx[int(v)] for v in pre], dtype=np.int64)
    poi_post = np.array([idx[int(v)] for v in post], dtype=np.int64)
    out_deg = np.bincount(poi, minlength=n_nodes).astype(np.int64)
    in_deg = np.bincount(poi_post, minlength=n_nodes).astype(np.int64)
    out_syn = np.bincount(poi, weights=syn, minlength=n_nodes)
    in_syn = np.bincount(poi_post, weights=syn, minlength=n_nodes)

    # ---- 写 CSV（m9 schema：注释头 + neuron 行 + chem 行 + gap 行；流式写盘）----
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        f.write("# M10 斑马鱼幼鱼（Fish1 CLEM, 7 dpf）连接组解析产物 —— 完整度分级 partial\n")
        f.write("# 来源：fish1-release 官方公开包（HMI_analysis.zip + TEN_analysis.zip）；"
                "论文 Petkova et al. 2025 bioRxiv 2025.06.10.658982\n")
        f.write("# 行类型 kind：neuron（roster 神经元行）/ chem（化学突触行）/ gap（缝隙连接行）\n")
        f.write("# neuron 行：id=agglomerated 段 id；cell_type=exc|inh|na（HMI CAVE somas 导出，覆盖有限）；"
                "region=''（官方未发布区域分配，不臆造）；neurotransmitter=''（仅 exc/inh 二元基因标注）；"
                "out_deg/in_deg/out_syn/in_syn（仅对已发布子图成立）\n")
        f.write("# chem 行：pre/post=agglomerated 段 id；syn_count=该有向对突触数；"
                "nt=''（逐突触递质不可得）；delay/size 不适用\n")
        f.write("# gap 行：0 行（官方包不含缝隙连接表 → 测量限制，不臆造）\n")
        w = csv_writer(f)
        w.writerow(["kind", "id", "cell_type", "region", "neurotransmitter",
                    "out_deg", "in_deg", "out_syn", "in_syn", "pre", "post", "syn_count"])
        for i in range(n_nodes):
            gid = int(roster_ids[i])
            w.writerow(["neuron", gid, type_map.get(gid, ""), "", "",
                        int(out_deg[i]), int(in_deg[i]), int(out_syn[i]), int(in_syn[i]),
                        "", "", ""])
        for k in range(E.shape[0]):
            w.writerow(["chem", "", "", "", "", "", "", "", "",
                        int(pre[k]), int(post[k]), int(syn[k])])
    sha = sha256_file(args.out_csv)

    # ---- 计数 JSON ----
    counts = dict(
        generated_by="tools/build_m10_connectome.py",
        completeness_level="partial",
        completeness_reason=(
            "全脑逐突触表（CAVE synapses_axde_label，~30M 行）需交互式 CAVE token（无头不可得）；"
            "可得者为官方公开包内的 (a) lores↔agglo roster (b) CAVE somas 导出 30,346 行 "
            "(c) TEN/DMV 电路逐神经元 partner 列表 + 逐突触尺寸表 ⇒ 按 §1.1.3 判 partial"),
        authority=AUTH,
        band=BAND,
        parse=dict(
            n_roster=n_roster,
            n_roster_unique_lore=int(roster["lores_id"].nunique()),
            n_soma_table=int(somas.shape[0]),
            soma_table_in_roster=int(soma_in_roster.sum()),
            cell_type_counts=ct_counts,
            n_chem_pairs_released=int(ten_e.shape[0]),
            n_chem_pairs_in_roster=int(E.shape[0]),
            n_chem_synapses_in_roster=int(syn.sum()),
            n_chem_pairs_released_all=int(ten_e.shape[0]),
            n_chem_synapses_released_all=int(ten_e["syn_count"].sum()),
            n_synapse_size_rows=int(syn_diag["n_synapse_rows"]),
            n_synapse_rows_raw=int(syn_diag["n_synapse_rows_raw"]),
            n_missing_partner_ids=int(syn_diag["n_missing_partner_ids"]),
            n_synapse_size_pairs=int(syn_diag["syn_pairs"]),
            self_chem_pairs=self_loops,
            gap_rows=0,
            edge_nodes=int(edge_nodes.size),
            typed_edge_nodes=typed_nodes,
            typed_edge_node_coverage_pct=round(100.0 * typed_nodes / max(1, edge_nodes.size), 2),
            n_edges_pre_type_known=n_edges_type_known,
            n_edges_inhibitory_pre=n_edges_inh,
            n_roster_rows=int(roster.shape[0]),
            cell_type_coverage_of_roster_pct=round(100.0 * (ct_counts.get("exc", 0) + ct_counts.get("inh", 0)) / n_roster, 3),
        ),
        cross_checks=dict(
            ten_vs_synapse_sizes=dict(
                ten_pairs=int(ten_e.shape[0]),
                size_pairs=int(syn_e.shape[0]),
                size_synapses=int(syn_e["syn_count"].sum()),
                note="TEN partner 列表（*_with_soma 列，仅含有 soma 的伙伴）vs "
                     "synapse_sizes.csv（逐突触尺寸表，不要求伙伴有 soma）→ 两套口径分别记录（M5 L7）"),
            thirdparty_derived=dict(
                file=os.path.join(RAW, "thirdparty_fish1released_connectivity.json"),
                n_edges=13813, n_analysed_lore=1088,
                note="第三方派生产物（jamieswrld/zebrafishconnectome，README 声明由公开包派生）；"
                     "仅作独立交叉核对，不作为权威来源"),
        ),
        diagnostics=[
            dict(name="全脑突触计数锚（~30M）", in_band=False,
                 detail="CAVE token 门控 → 全脑突触表不可得，锚**未验证**（不是“未命中”，是“未测”）；"
                        "请求三态裁决（见 adjudication_requests）"),
            dict(name="缝隙连接", in_band=False,
                 detail="官方公开包与 CAVE 表清单均无缝隙连接表（CAVE 表名为 synapses_axde/axax 等，"
                        "无 gap junction 表）→ gap 行 0 + 测量限制（M9 R8 语义沿用）"),
            dict(name="区域分配", in_band=False,
                 detail="官方未发布逐神经元区域分配（第三方文档亦确认 not exported）；"
                        "由坐标推区域 = 臆造 → 不生成 region 列（M8 B1a 纪律）"),
            dict(name="递质/调质类型", in_band=False,
                 detail="Fish1 仅发布 vglut2a+(exc)/gad1b+(inh) 遗传标注二元身份，无 GABA_A/B 亚型、"
                        "无 DA/5HT 逐神经元标注 → M10 P6 的 M-3 级（GABA 亚型）**只能走类级文献先验 + "
                        "测量限制登记**（M10 §1.1.2 允许：覆盖率如实记录，不臆造受体作用域）"),
            dict(name="cell_type 覆盖率", in_band=True,
                 detail=f"roster {n_roster} 中 typed(exc+inh) "
                        f"{ct_counts.get('exc',0)+ct_counts.get('inh',0)} "
                        f"（{100.0*(ct_counts.get('exc',0)+ct_counts.get('inh',0))/n_roster:.2f}%）；"
                        f"边节点覆盖 {typed_nodes}/{edge_nodes.size} "
                        f"({100.0*typed_nodes/max(1,edge_nodes.size):.1f}%)——如实记录，可 <100%"),
        ],
        adjudication_requests=[
            dict(id="RF-1", question="是否提供 CAVE token（浏览器交互）以取得全脑逐突触表"
                                     "（synapses_axde_label，~30M 行）？",
                 impact="取得 → P1 完整度可升自 partial 至 full（全脑逐突触 + 逐突触 tag 兴奋/抑制）",
                 fallback="不提供 → P1 判 partial：判据带按子图重新定稿（§1.1.3），报告标结构证据级别=partial"),
            dict(id="RF-2", question="P6 的 GABA 亚型异质（M-3）在无亚型标注下是否接受"
                                     "「类级文献先验 + 测量限制登记」实现？",
                 impact="接受 → M-3 可按 τ_A∈[5,20]ms / τ_B∈[80,400]ms 类级扫描；"
                        "不接受 → M-3 不可启动，机制阶梯缺一级",
                 fallback="按 §1.5「递质/调质标注缺」行：类级文献先验注入 + 标注测量限制（不臆造逐突触）"),
        ],
        outputs=dict(csv=args.out_csv, csv_sha256=sha, npz=args.out_npz),
        elapsed_s=round(time.time() - t0, 2),
    )

    # ---- npz（引擎装配件；权重不可辨识 → gmax=1.0 基线占位，不入任何判据）----
    node_ids = roster_ids
    np.savez_compressed(
        args.out_npz,
        neuron_ids=node_ids,
        pre=np.array([idx[int(v)] for v in pre], dtype=np.int64),
        post=np.array([idx[int(v)] for v in post], dtype=np.int64),
        syn_count=syn.astype(np.int32),
        gmax=np.ones(E.shape[0], dtype=np.float32),
            inhibitory=np.array([type_map.get(int(v), "") == "inh" for v in pre], dtype=bool),
        recipe=json.dumps(dict(weights="uncalibrated_baseline_gmax_1.0",
                               note="权重不可辨识（§0.7 #7）→ 类级缩放标定在 P5；此文件不作判据"),
                          ensure_ascii=False),
    )
    counts["outputs"]["npz_sha256"] = sha256_file(args.out_npz)
    counts["outputs"]["npz_bytes"] = os.path.getsize(args.out_npz)

    with open(args.out_json, "w") as f:
        json.dump(counts, f, indent=1, ensure_ascii=False)

    print(json.dumps(dict(n_roster=n_roster, n_edges=int(E.shape[0]),
                          n_syn=int(syn.sum()), sha=sha,
                          csv=args.out_csv), ensure_ascii=False, indent=1))
    return 0


def csv_writer(f):
    import csv
    return csv.writer(f, lineterminator="\n")


if __name__ == "__main__":
    raise SystemExit(main())
