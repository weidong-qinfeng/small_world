#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M11 · DQ-4 解析器：MICrONS `functional_connectomics` 逐细胞/逐对表（**只解析，不下载**）

背景（主 agent DQ-4 裁决）
--------------------------
主 agent 已亲自启动托管下载（`data/m11_raw/microns/functional_connectomics/`），
并明确「**你负责在下载完成后解析**（pickle 载入 + schema 逐列核对 + 落盘为 parquet/csv + 记录 sha256/行数）。
**不要自己再起下载**」。本工具严格遵守：**只读本地文件**，任何"文件不存在/尺寸不符"一律判 `INCOMPLETE` 并**退出非零**。

对象与期望尺寸（**实测**，Range `Content-Range` 给出）
- `node_data_v1.pkl`  期望 **294,635,032 B**（逐神经元：EM 坐标 + `layer`/`brain_area` + `pref_ori`/`osi_iv`/…）
- `edge_data_v1.pkl`  期望 **201,511,086 B**（逐对：`population`/`n_synapses`/`synapse_size`/`in_vivo_sig_corr_iv`/…）
- `readme_v1.md`      5,594 B（已在位，作为 schema 依据）

解析产物
--------
- `data/m11_fc_nodes.parquet`（+ `.csv` 可选）/ `data/m11_fc_edges.parquet`
- `data/m11_functional_connectomics_counts.json`：行数/列数/schema 核对/逐列非空率/关键列取值分布/sha256/证据级别

用法（**pandas 在 `.venv-m10`**；数据侧按清单 L5 复用该 venv）
    ./.venv-m10/bin/python neural_exploration/tools/parse_m11_functional_connectomics.py            # 解析（若完整）
    ./.venv-m10/bin/python neural_exploration/tools/parse_m11_functional_connectomics.py --csv      # 同时落 CSV
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJ = os.path.dirname(ROOT)
DATA = os.path.join(ROOT, "data")
FC = os.path.join(DATA, "m11_raw", "microns", "functional_connectomics")
OUT_JSON = os.path.join(DATA, "m11_functional_connectomics_counts.json")
README = os.path.join(FC, "readme_v1.md")

EXPECT = {
    "node_data_v1.pkl": 294635032,
    "edge_data_v1.pkl": 201511086,
    "readme_v1.md": 5594,
}
URL_BASE = ("https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/functional_data/"
            "functional_connectomics/node_and_edge_properties/v1/")


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def schema_from_readme():
    """从 readme（已落盘）解析出 node/edge 的列名清单——**schema 核对依据**。"""
    if not os.path.exists(README):
        return {}
    txt = open(README, encoding="utf-8", errors="replace").read()
    out = {}
    for sect, pat in (("node", r"### Node Data Columns(.*?)##"), ("edge", r"### Edge Data Columns(.*)")):
        m = re.search(pat, txt, re.S)
        if not m:
            continue
        cols = re.findall(r"^\|\s*`([^`]+)`\s*\|", m.group(1), re.M)
        out[sect] = cols
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="store_true", help="同时落 CSV（体积大）")
    args = ap.parse_args()

    out = dict(generated_by="tools/parse_m11_functional_connectomics.py", generated_at=utcnow(),
               rule="**只解析不下载**（DQ-4：下载由主 agent 托管）；文件缺失/尺寸不符一律 INCOMPLETE 且退出非零",
               expected_bytes=EXPECT, url_base=URL_BASE,
               readme_present=os.path.exists(README),
               coverage_note="逐细胞/逐对表 = **属性级同源**（R8）：提供 P7 调谐分布 / P8 层·区差异 / P14 群体相关"
                             "的 `measured` 依据，并自带 `population` 匹配负对照（Connected/ADP/Same region）；"
                             "**但不含时间序列** ⇒ S 门与 dwell/τ 目标带仍走 Allen 2p（R6 不变）")
    sched = schema_from_readme()
    out["schema_from_readme"] = sched

    status = {}
    for fn, exp in EXPECT.items():
        p = os.path.join(FC, fn)
        st = dict(expected_bytes=exp, url=URL_BASE + fn)
        if not os.path.exists(p):
            st.update(bytes=0, verdict="INCOMPLETE", note="本地不存在（下载未完成）")
        else:
            n = os.path.getsize(p)
            st.update(bytes=n, sha256=sha256_file(p) if n == exp else None)
            st["verdict"] = "COMPLETE" if n == exp else "INCOMPLETE"
            if n != exp:
                st["note"] = ("尺寸不符：%d / %d（**未完成**；不解析、不入统计）" % (n, exp))
        # 分片目录进度（主 agent 的托管作业按段写 parts/）
        parts = sorted(__import__("glob").glob(os.path.join(FC, "parts", fn + ".p*")))
        if parts and st["verdict"] == "INCOMPLETE":
            st["parts_n"] = len(parts)
            st["parts_bytes"] = sum(os.path.getsize(x) for x in parts)
            st["parts_progress_pct"] = round(100.0 * st["parts_bytes"] / exp, 3)
        status[fn] = st
    out["files"] = status

    # ---------- 解析（仅当两个主表都完整）----------
    node_p, edge_p = os.path.join(FC, "node_data_v1.pkl"), os.path.join(FC, "edge_data_v1.pkl")
    ok = all(status[f]["verdict"] == "COMPLETE" for f in ("node_data_v1.pkl", "edge_data_v1.pkl"))
    if not ok:
        out["parse_status"] = "not_run（文件未完整）"
        with open(OUT_JSON, "w") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        print("INCOMPLETE：", json.dumps({k: (v["verdict"], v.get("parts_progress_pct"))
                                          for k, v in status.items()}, ensure_ascii=False))
        print("wrote", OUT_JSON, "（未解析；等待下载完成）")
        return 2

    import pandas as pd  # noqa: E402
    res = {}
    for fn, p, kind in (("node_data_v1.pkl", node_p, "node"), ("edge_data_v1.pkl", edge_p, "edge")):
        df = pd.read_pickle(p)
        cols = list(df.columns)
        exp_cols = sched.get(kind) or []
        d = dict(rows=int(df.shape[0]), columns=int(df.shape[1]),
                 schema_match=bool(set(cols) == set(exp_cols)) if exp_cols else None,
                 missing_vs_readme=sorted(set(exp_cols) - set(cols)) if exp_cols else None,
                 extra_vs_readme=sorted(set(cols) - set(exp_cols)) if exp_cols else None,
                 nonnull_frac={c: round(float(df[c].notna().mean()), 6) for c in cols})
        # 关键列取值分布（按 readme 的语义列）
        keys = (["layer", "brain_area", "hva"] if kind == "node"
                else ["population", "pre_layer", "post_layer", "pre_hva", "post_hva"])
        d["value_counts"] = {k: {str(a): int(b) for a, b in df[k].value_counts().head(12).items()}
                             for k in keys if k in cols}
        for k in (["osi_iv", "gosi_iv", "pref_ori", "pref_dir"] if kind == "node"
                  else ["n_synapses", "synapse_size"]):
            if k in cols:
                s = pd.to_numeric(df[k], errors="coerce")
                d.setdefault("numeric_summary", {})[k] = dict(
                    n=int(s.notna().sum()), min=float(s.min()), median=float(s.median()),
                    p90=float(s.quantile(0.9)), max=float(s.max()))
        outp = os.path.join(DATA, "m11_fc_%ss.parquet" % kind)
        df.to_parquet(outp, index=False)
        d["parquet"] = os.path.relpath(outp, PROJ)
        if args.csv:
            csvp = os.path.join(DATA, "m11_fc_%ss.csv" % kind)
            df.to_csv(csvp, index=False)
            d["csv"] = os.path.relpath(csvp, PROJ)
        res[kind] = d
    out["parse_status"] = "ok"
    out["tables"] = res
    out["evidence_level"] = "measured（作者发布表；DOI 10.1038/s41586-025-08840-3）"
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("parsed:", json.dumps({k: (v["rows"], v["columns"], v.get("schema_match"))
                                 for k, v in res.items()}, ensure_ascii=False))
    print("wrote", OUT_JSON)
    return 0


if __name__ == "__main__":
    sys.exit(main())
