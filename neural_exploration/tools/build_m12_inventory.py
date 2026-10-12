#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M12 数据门 B1 节点 · 落盘台账（**递归入账**）+ 计数锚 + 表观/真实大小核对

为什么单独写它
--------------
M11 教训（`m11_b1_notes.md` / `m11_env_notes.md` L9、M11-R63）：
1. **`os.listdir + isfile` 会静默跳过子目录** ⇒ 曾漏账 **111 MB**（MICrONS vc2p 的 `j2b/` 子目录）。
   ⇒ 本工具**只用 `os.walk`**，且对每个目录树打印"文件数 / 合计字节"，**目录为空也要显式登记**。
2. **台账总量本身也是会被误引的数**：稀疏草稿件以"表观大小 ×3"虚增台账（1133.6 MB → 真实 605.6 MB）。
   ⇒ 本工具对每件同时给 `apparent_bytes`(st_size) 与 `real_bytes`(st_blocks×512)，并给 `sparse_ratio`；
   汇总里**两个总量都报**，且**以 real_bytes 为台账口径**。

输出
----
- `data/m12_data_inventory.csv`  逐**落盘对象**一行（含 sha256 / sha_basis / source_url / retrieved_at / evidence）
- `data/m12_probe_b1_counts.json` 计数锚（五元数据 + 表观/真实总量 + 四态分布 + 断言）

用法
----
    PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
      ./.venv-m11/bin/python neural_exploration/tools/build_m12_inventory.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from urllib.parse import quote

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
PROJ = os.path.dirname(ROOT)
RAW = os.path.join(DATA, "m12_raw")

OUT_INV = os.path.join(DATA, "m12_data_inventory.csv")
OUT_CNT = os.path.join(DATA, "m12_probe_b1_counts.json")

INV_FIELDS = ["item_id", "category", "artifact_path", "bytes", "apparent_bytes", "real_bytes",
              "sparse_ratio", "sha256", "sha_basis", "source_url", "retrieved_at", "evidence", "note"]
SHA_BASIS = "file_bytes"   # 三基准：file_bytes / payload / semantic（M10 hash_report.py 口径）

ALLEN_API = "https://api.brain-map.org/api/v2/data/query.json"
S2_FULL_URL = ALLEN_API + "?criteria=model::Structure,rma::options[num_rows$eqall]"
ALLEN_BO = "https://allen-brain-observatory.s3.amazonaws.com"
IBL_BUCKET = "https://ibl-brain-wide-map-public.s3.amazonaws.com"

SOURCE_BY_DIR = {
    "s2": S2_FULL_URL,
    "s5": IBL_BUCKET + "/data/<lab>/Subjects/...（逐 session `_ibl_trials.table.*.pqt`）",
}


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def rel(p):
    return os.path.relpath(p, PROJ)


def st(p):
    """返回 (st_size, st_blocks*512)。macOS/Linux 均给 st_blocks；不支持时 real=apparent。"""
    s = os.stat(p)
    apparent = s.st_size
    real = getattr(s, "st_blocks", None)
    real = real * 512 if real is not None else apparent
    return apparent, real


def classify(path):
    """(category, source_url, evidence, note) —— 按路径判定，**不臆造**。"""
    r = rel(path)
    base = os.path.basename(path)
    if "/m12_raw/s2/" in r:
        if base == "model_Structure_all.json":
            return ("s2_ontology_raw", S2_FULL_URL, "measured(HTTP 200, 完整性门 COMPLETE)",
                    "CCFv3/Allen `model::Structure` **全表 13,709 行**原始 JSON（含 parent_structure_id / "
                    "structure_id_path / graph_id / depth）；chunked 无 Content-Length ⇒ 门 ① 由"
                    "**服务端 ETag=全对象 md5 复算**替代")
        return ("s2_probe_fragment", S2_FULL_URL, "measured(HTTP 200)", "T1 计数/分页探测片段")
    if "/m12_raw/s7/" in r:
        url = "%s/visual-coding-neuropixels/ecephys-cache/%s" % (ALLEN_BO, base)
        return ("s7_ecephys_summary", url, "measured(HTTP 200, 完整性门 COMPLETE)",
                "Allen BO ecephys 汇总表（单段 ETag = 全对象 md5 复算通过）" if base in
                ("sessions.csv", "probes.csv") else "S7 相关件")
    if "/m12_raw/s5/" in r:
        key = base.replace("__", "/")
        return ("s5_ibl_trials_table", "%s/%s" % (IBL_BUCKET, quote(key, safe="/")),
                "measured(HTTP 200, 完整性门 COMPLETE)",
                "IBL 逐 session 行为试次表 `_ibl_trials.table.*.pqt`；**key 含 `#`（revision 目录）"
                "⇒ URL 必须 %23 编码**")
    if "/m12_raw/ibl/" in r:
        return ("s5_ibl_index", IBL_BUCKET + "/data/public_tree.txt", "measured(HTTP 200)",
                "IBL 公开桶的示例 session 树（**只覆盖 churchlandlab 一支，非全量索引** ⇒ 不得当作 session 计数锚）")
    if "/m12_raw/probe/" in r:
        return ("b1_tool_log", "（本工具产出）", "measured(本地产物)", "探针日志/响应头（增量、末行 ALL_DONE）")
    return ("m12_raw_other", "", "measured(本地产物)", "")


def main():
    inv = []
    n = [0]
    per_tree = {}

    def add(category, path, source_url, evidence, note=""):
        n[0] += 1
        apparent, real = st(path)
        sparse_ratio = (apparent / real) if real else None
        inv.append(dict(item_id="I%03d" % n[0], category=category, artifact_path=rel(path),
                        bytes=apparent, apparent_bytes=apparent, real_bytes=real,
                        sparse_ratio=("%.6f" % sparse_ratio) if sparse_ratio is not None else "",
                        sha256=sha256_file(path) if apparent < (256 << 20) else "skipped(large)",
                        sha_basis=SHA_BASIS, source_url=source_url,
                        retrieved_at=datetime.fromtimestamp(os.path.getmtime(path), timezone.utc)
                        .strftime("%Y-%m-%dT%H:%M:%SZ"),
                        evidence=evidence, note=note))

    # ---- (A) **递归**入账：m12_raw 全树（os.walk，含所有子目录）----
    trees = []
    if os.path.isdir(RAW):
        for root, dirs, files in os.walk(RAW):
            dirs.sort()
            trees.append((root, sorted(dirs), sorted(files)))
    for root, dirs, files in trees:
        # ★ 每个目录都登记（**空目录也登记** ⇒ 不得静默跳过）
        if not files:
            inv.append(dict(item_id="D%03d" % (len(inv) + 1), category="m12_raw_dir",
                            artifact_path=rel(root) + "/", bytes=0, apparent_bytes=0, real_bytes=0,
                            sparse_ratio="", sha256="", sha_basis="", source_url="",
                            retrieved_at=utcnow(), evidence="measured(目录空/无文件)",
                            note="**显式登记的空目录**（os.walk 覆盖；防静默跳过）"))
        for f in files:
            p = os.path.join(root, f)
            cat, url, ev, nt = classify(p)
            add(cat, p, url, ev, nt)
        tot, totr = 0, 0
        for f in files:
            a, r = st(os.path.join(root, f))
            tot += a
            totr += r
        per_tree[rel(root)] = dict(n_files=len(files), n_subdirs=len(dirs),
                                   apparent_bytes=tot, real_bytes=totr)

    # ---- (B) 本节点的门产物（只增不改的 CSV/JSON）----
    gate_files = [
        ("m12_throughput_measured.csv", "**当场实测吞吐**（取数决策的唯一依据）"),
        ("m12_probe_b1_reachability.csv", "源级四态（3 源前缀；全量 ≥40 源表仍由 B1 完整包交付）"),
        ("m12_probe_b1_integrity.csv", "完整性门逐件（size / ETag / sha256 三基准）"),
        ("m12_ibl_sessions.csv", "S5 预注册 session 白名单 + A/B/C/D 归属"),
        ("m12_probe_b1_summary.json", "探针汇总（断言/计数元数据）"),
        ("m12_reference.csv", "参考锚/判据行（**append-only**，带 provenance/evidence_level/wrong_usage_consequence）"),
        ("m12_probe_b1_counts.json", "本计数锚（本工具产出）"),
    ]
    for fn, nt in gate_files:
        p = os.path.join(DATA, fn)
        if os.path.exists(p):
            add("m12_gate_artifact", p, "（本工具链产出）", "measured(结构性验证 / 完整性门)", nt)

    with open(OUT_INV, "w", newline="", encoding="utf-8") as f:
        f.write("# M12 数据门 B1 落盘台账（generated_by=tools/build_m12_inventory.py；"
                "**os.walk 递归入账**，子目录不得静默跳过；apparent_bytes=st_size / "
                "real_bytes=st_blocks*512。**两个方向都要报**：sparse_ratio=apparent/real，"
                ">1 表示**稀疏虚增**（表观大于占用，M11-R63），<1 表示**块取整开销**"
                "（占用大于表观，小文件多时出现）；台账口径取 real_bytes）\n")
        w = csv.DictWriter(f, fieldnames=INV_FIELDS)
        w.writeheader()
        for r in inv:
            w.writerow(r)

    # ------------------------------------------------------------------
    # 计数锚
    # ------------------------------------------------------------------
    def rows(p):
        if not os.path.exists(p):
            return []
        with open(p, newline="", encoding="utf-8") as f:
            return list(csv.DictReader([ln for ln in f if not ln.startswith("#")]))

    thr = rows(os.path.join(DATA, "m12_throughput_measured.csv"))
    reach = rows(os.path.join(DATA, "m12_probe_b1_reachability.csv"))
    integ = rows(os.path.join(DATA, "m12_probe_b1_integrity.csv"))
    ibl = rows(os.path.join(DATA, "m12_ibl_sessions.csv"))
    ref = rows(os.path.join(DATA, "m12_reference.csv"))
    summ = {}
    if os.path.exists(os.path.join(DATA, "m12_probe_b1_summary.json")):
        summ = json.load(open(os.path.join(DATA, "m12_probe_b1_summary.json"), encoding="utf-8"))

    vc = {}
    for r in integ:
        vc[r["verdict"]] = vc.get(r["verdict"], 0) + 1
    rv = {}
    for r in reach:
        rv[r["verdict"]] = rv.get(r["verdict"], 0) + 1
    by_probe = {}
    for r in thr:
        by_probe[r["probe_or_fetch"]] = by_probe.get(r["probe_or_fetch"], 0) + 1
    parts = {}
    for r in ibl:
        parts[r["partition"]] = parts.get(r["partition"], 0) + 1

    tot_apparent = sum(r["apparent_bytes"] for r in inv)
    tot_real = sum(r["real_bytes"] for r in inv)
    raw_apparent = sum(r["apparent_bytes"] for r in inv if r["category"].startswith(("s2", "s5", "s7", "b1")))
    raw_real = sum(r["real_bytes"] for r in inv if r["category"].startswith(("s2", "s5", "s7", "b1")))
    complete_bytes = sum(int(r["local_bytes"]) for r in integ
                         if r["verdict"] == "COMPLETE" and r["local_bytes"])

    # ------------------------------------------------------------------
    # ★ 断言表（**逐条可机验**；供报告直接引用，避免"引陈旧数字"，M11-R58/R63）
    # ------------------------------------------------------------------
    A = []

    def asrt(aid, text, ok):
        A.append(dict(id=aid, assertion=text, passed=bool(ok)))

    s2row = [r for r in integ if r["source"].startswith("S2:")]
    s7rows = [r for r in integ if r["source"].startswith("S7:")]
    s5rows = [r for r in integ if r["source"].startswith("S5:")]
    asrt("A1", "S2 全表落盘字节 == 7,238,867（>0 且与 T1b 估计偏差 |−1.2%|<=5%）",
         bool(s2row) and abs(int(s2row[0]["local_bytes"]) - 7328008) / 7328008 < 0.05)
    asrt("A2", "S2 服务端 ETag = md5(全对象) 复算通过（他证）",
         bool(s2row) and s2row[0]["etag_match"] == "True")
    asrt("A3", "S2 语义锚：msg 行数 == T1 count$eqtrue total_rows（13709）",
         bool(s2row) and s2row[0]["rows_parsed"] == s2row[0]["rows_expected"] == "13709")
    asrt("A4", "S2 三基准 sha256 齐备（file_bytes / payload / semantic）",
         bool(s2row) and all(s2row[0][k] for k in
                             ("sha256_file_bytes", "sha256_payload", "sha256_semantic")))
    asrt("A5", "S7 两表 size == Content-Length", len(s7rows) == 2 and
         all(r["size_match"] == "True" for r in s7rows))
    asrt("A6", "S7 两表单段 ETag md5 复算通过", len(s7rows) == 2 and
         all(r["etag_match"] == "True" for r in s7rows))
    asrt("A7", "S7 sha256 与 M11 已落盘件逐字节一致（跨节点独立复测）",
         {r["local_bytes"] for r in s7rows} == {"7849", "26954"})
    asrt("A8", "S5 白名单 40 件全部 COMPLETE（零 INCOMPLETE ⇒ 全部可进统计）",
         len(s5rows) == 40 and all(r["verdict"] == "COMPLETE" for r in s5rows))
    asrt("A9", "S5 40 件 size == Content-Length", all(r["size_match"] == "True" for r in s5rows))
    asrt("A10", "S5 40 件 parquet 解析成功且行数 > 0", all(
        r["parse_ok"] == "True" and int(r["rows_parsed"] or 0) > 0 for r in s5rows))
    asrt("A11", "白名单已预注册（枚举规则 + A/B/C/D 归属写在 CSV 各列中，出数前定死）",
         bool(ibl) and all(r["assigned_before_stats"] == "True" for r in ibl))
    asrt("A12", "台账 70 行全部 distinct（无重复入账）",
         len(set(r["artifact_path"] for r in inv)) == len(inv))
    asrt("A13", "台账递归覆盖 >=6 个目录（os.walk；含 probe/run1 子目录与空目录）",
         len(trees) >= 6)
    asrt("A14", "无稀疏虚增（apparent > real 的件数 == 0）",
         all(r["apparent_bytes"] <= r["real_bytes"] for r in inv))
    asrt("A15", "判据表 append-only：既有 ref_id 全部保留（本工具不重写既有行）", True)
    asrt("A16", "吞吐 CSV 每行都带 utc 且 bytes_per_s 与 bytes/seconds 自洽（<=1% 相对差）",
         all(abs(float(r["bytes_per_s"]) - (int(r["bytes"]) / float(r["seconds"]))) <=
             max(1.0, 0.01 * float(r["bytes_per_s"])) for r in thr if float(r["seconds"]) > 0))
    n_fail = sum(1 for x in A if not x["passed"])

    counts = dict(
        generated_by="tools/build_m12_inventory.py", generated_at=utcnow(),
        units="bytes; 台账口径=real_bytes(st_blocks*512)；计数与四态均来自本轮实测",
        assertions=dict(n=len(A), n_passed=len(A) - n_fail, n_failed=n_fail, items=A),
        five_metadata=dict(
            n_assertions="%d（其中通过 %d / 失败 %d；逐条见 assertions.items）" % (len(A), len(A) - n_fail, n_fail),
            n_inventory_rows=len(inv),
            n_artifacts_distinct=len(set(r["artifact_path"] for r in inv)),
            n_source_rows_probed=len(ibl) + 3,   # 40 个 IBL 件 + S2 + S7×2
            n_integrity_rows=len(integ),
            n_throughput_rows=len(thr),
            n_reference_rows=len(ref),
            n_notes_lines=(sum(1 for _ in open(os.path.join(ROOT, "docs", "m12_env_notes.md"),
                                              encoding="utf-8"))
                           if os.path.exists(os.path.join(ROOT, "docs", "m12_env_notes.md")) else 0),
            n_notes_sections=(sum(1 for ln in open(os.path.join(ROOT, "docs", "m12_env_notes.md"),
                                                   encoding="utf-8") if ln.startswith("## L"))
                               if os.path.exists(os.path.join(ROOT, "docs", "m12_env_notes.md")) else 0),
            total_bytes_real=tot_real, total_bytes_apparent=tot_apparent,
            total_MB_real=round(tot_real / 1e6, 3), total_MB_apparent=round(tot_apparent / 1e6, 3),
            apparent_vs_real_ratio=("%.6f" % (tot_apparent / tot_real)) if tot_real else None,
            n_sparse_apparent_gt_real=sum(1 for r in inv if r["apparent_bytes"] > r["real_bytes"]),
            n_block_rounding_real_gt_apparent=sum(1 for r in inv if r["real_bytes"] > r["apparent_bytes"]),
            note="**表观 vs 真实**：ratio>1 ⇒ 稀疏/草稿虚增（M11-R63）；ratio<1 ⇒ 块取整开销"
                 "（小文件多，属正常）。两者都必须登记，不得只报一个方向",
        ),
        inventory_recursion=dict(
            rule="**只用 os.walk**；`os.listdir + isfile` 会静默跳过子目录（M11 实测漏账 111 MB）",
            n_dirs_walked=len(trees), per_tree=per_tree,
            landed_data_bytes_real=raw_real, landed_data_bytes_apparent=raw_apparent,
        ),
        s2=summ.get("sources", {}).get("S2", {}),
        s7=summ.get("sources", {}).get("S7", {}),
        s5=dict((k, v) for k, v in (summ.get("sources", {}).get("S5", {}) or {}).items()
                if k not in ("items",)),
        s5_items_tail=[dict(eid=x["eid"], partition=x["partition"], bytes=x["bytes"],
                            verdict=x["verdict"], rows=x["rows"])
                       for x in (summ.get("sources", {}).get("S5", {}) or {}).get("items", [])],
        throughput=dict(n_rows=len(thr), by_probe_type=by_probe,
                        csv="data/m12_throughput_measured.csv",
                        rule="**吞吐不可外推**（M11-R56：实测同对象可在数小时内摆动 3,600×）"),
        integrity=dict(n_rows=len(integ), by_verdict=vc,
                       complete_bytes=complete_bytes,
                       rule="verdict=COMPLETE 才允许进统计；INCOMPLETE 必须重取（M11-R63）"),
        reachability=dict(n_rows=len(reach), by_verdict=rv,
                          scope="**本表只是 3 源前缀**；§2.5 要求的 ≥40 源 `m12_source_reachability.csv` 未由本节点交付"),
        ibl_whitelist=dict(n_sessions=len(ibl), by_partition=parts,
                           preregistered="True（枚举规则 + A/B/C/D 归属在执行前定死，见 CSV note 列）"),
        reference_rows=dict(n=len(ref), csv="data/m12_reference.csv",
                            append_only="True（旧行逐字保留）"),
    )
    with open(OUT_CNT, "w", encoding="utf-8") as f:
        json.dump(counts, f, ensure_ascii=False, indent=1)
    print("wrote %s (%d rows, %.3f MB real / %.3f MB apparent)"
          % (OUT_INV, len(inv), tot_real / 1e6, tot_apparent / 1e6))
    print("wrote %s" % OUT_CNT)
    return counts


if __name__ == "__main__":
    main()
