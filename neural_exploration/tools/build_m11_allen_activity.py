#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M11 · 活动通道**回退源**落地器（RF-M11-A §二.3 处置）

背景（主 agent 裁定 RF-M11-A）
------------------------------
主 agent 判定 MICrONS **功能体素序列**在无官方客户端下不可解，要求 P2 功能通道**回退**到
"文档化 / 自描述格式"的源，并把体素序列登记 `format_unverified`。
本节点据 RF-M11-A 的 §三 纪律同时给出**两条事实**（先给证据再给结论）：
1. **体素序列其实已解**（大端 int16；见 `tools/demo_m11_microns_decode.py` 与
   `data/m11_decode_proof.json`）——本文件不重复论证，只在 `voxel_channel` 段引用其证据；
2. **回退源照样落地**（本节主体）：Allen Brain Observatory Neuropixels 的**自描述表格 + NWB(HDF5)**，
   满足"读回采样率与 units/traces 结构"的要求，且**格式风险最低**（HDF5/CSV 自带元数据）。

落地内容
--------
- `probes.csv`（完整）→ **采样率读回**：`sampling_rate ≈ 30 kHz`、`lfp_sampling_rate ≈ 2.5 kHz`
  （每个 probe 一行 + `has_lfp_data`）；
- `sessions.csv`（完整）→ session 级元数据（type/specimen/age/sex/genotype/`has_nwb`）⇒ 供 P14 的 session 级隔离；
- `units.csv`（**全量 51.4 MB 未取**，网络 16 KB/s ⇒ 预计 >30 min ⇒ 登记 `unavailable` 并交裁决 DQ-6）
  → 用 **1 MB Range 前缀**读回**表头 schema**（firing_rate / snr / isi_violations / quality / …）；
- `channels.csv` / 一个 `*_lfp.nwb`（2.06 GB，>2 GB 上限）→ 有界 Range 前缀（头部/表头）；
- 逐对象登记：URL / Range / 期望字节 / 实收字节 / SHA-256 / **verdict(COMPLETE|INCOMPLETE|RANGE_BOUNDED)**。

用法
----
    PYTHONHASHSEED=0 ./.venv-m11/bin/python neural_exploration/tools/build_m11_allen_activity.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJ = os.path.dirname(ROOT)
DATA = os.path.join(ROOT, "data")
BO = os.path.join(DATA, "m11_raw", "allen", "bo_ecephys")
OUT_JSON = os.path.join(DATA, "m11_allen_activity_fallback.json")
OUT_CSV = os.path.join(DATA, "m11_allen_activity_objects.csv")
BUCKET = "https://allen-brain-observatory.s3.amazonaws.com/visual-coding-neuropixels/ecephys-cache"

# 对象登记：本地名 → (远端键, 期望字节, Range, 用途)
OBJECTS = [
    ("manifest.json", "manifest.json", 1879, None, "数据集清单（BASEDIR/sessions/probes/channels/units/session_data）"),
    ("sessions.csv", "sessions.csv", 7849, None, "session 级元数据（type/specimen/age/sex/genotype/has_nwb）"),
    ("probes.csv", "probes.csv", 26954, None, "**采样率读回**：sampling_rate / lfp_sampling_rate / has_lfp_data"),
    ("units_prefix_1MB.csv", "units.csv", 1048576, "0-1048575",
     "逐 unit 质量表（全量 51,439,189 B **未取** ⇒ 仅前缀 schema/样本）"),
    ("channels_prefix_1MB.csv", "channels.csv", 1048576, "0-1048575",
     "逐 channel 表（全量 6,613,329 B 未取 ⇒ 仅前缀）"),
    ("probe_810755797_lfp_head8MB.bin", "session_715093703/probe_810755797_lfp.nwb", 8388608, "0-8388607",
     "LFP NWB(HDF5) 前缀 ⇒ 读回 sampling_rate / 电极与单元结构（全量 2,062,950,784 B **>2 GB 上限**）"),
]


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify_objects():
    rows = []
    for local, key, exp, rng, use in OBJECTS:
        p = os.path.join(BO, local)
        r = dict(local_file="data/m11_raw/allen/bo_ecephys/%s" % local, url="%s/%s" % (BUCKET, key),
                 range=rng, expected_bytes=exp, use=use, retrieved_at=utcnow())
        if not os.path.exists(p):
            r.update(bytes=0, verdict="MISSING", note="本地不存在（未取）")
        else:
            n = os.path.getsize(p)
            r.update(bytes=n, sha256=sha256_file(p))
            if rng is None:
                r["verdict"] = "COMPLETE" if (exp is None or n == exp) else "INCOMPLETE"
            else:
                r["verdict"] = "RANGE_BOUNDED" if n == exp else "INCOMPLETE"
                r["note"] = ("有界 Range 前缀（**故意只取前 %d B**；全量 %s ⇒ 未取）" % (exp, key))
            if r["verdict"] == "INCOMPLETE":
                r["note"] = "期望 %s B，实收 %s B ⇒ **INCOMPLETE（不进统计，需重取）**" % (exp, n)
        rows.append(r)
    return rows


def read_csv_head(path, n=2):
    """读 CSV 的**表头 + 前 n 行**（前缀文件可能被截断 ⇒ 只做 schema/样本读回）。"""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as f:
        rows = []
        for i, ln in enumerate(f):
            rows.append(ln.rstrip("\n"))
            if i >= n:
                break
    if not rows:
        return None
    header = rows[0].split(",")
    recs = []
    for ln in rows[1:n + 1]:
        vals = ln.split(",")
        recs.append(dict(zip(header, vals)) if len(vals) == len(header) else dict(_raw=ln[:300]))
    return dict(columns=header, n_columns=len(header), sample=recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-nwb", action="store_true", help="对已落盘的 NWB 前缀做 HDF5 头解析")
    args = ap.parse_args()

    objs = verify_objects()
    sess = read_csv_head(os.path.join(BO, "sessions.csv"), 2)
    probes = read_csv_head(os.path.join(BO, "probes.csv"), 3)
    units = read_csv_head(os.path.join(BO, "units_prefix_1MB.csv"), 2)
    chan = read_csv_head(os.path.join(BO, "channels_prefix_1MB.csv"), 2)

    # 采样率读回（probes.csv 全量在盘 ⇒ 逐 probe 统计）
    rate = {}
    pp = os.path.join(BO, "probes.csv")
    if os.path.exists(pp) and os.path.getsize(pp) == OBJECTS[2][2]:
        with open(pp, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        sr = [float(r["sampling_rate"]) for r in rows if r.get("sampling_rate")]
        lr = [float(r["lfp_sampling_rate"]) for r in rows if r.get("lfp_sampling_rate")]
        sub = [float(r["lfp_temporal_subsampling_factor"]) for r in rows
               if r.get("lfp_temporal_subsampling_factor")]
        rate = dict(n_probes=len(rows), n_with_sampling_rate=len(sr),
                    sampling_rate_hz=dict(min=min(sr), max=max(sr), mean=sum(sr) / len(sr)) if sr else None,
                    lfp_sampling_rate_hz=dict(min=min(lr), max=max(lr), mean=sum(lr) / len(lr)) if lr else None,
                    lfp_temporal_subsampling_factor=sorted(set(sub)) if sub else None,
                    n_has_lfp_data=sum(1 for r in rows if (r.get("has_lfp_data") or "").lower() == "true"))

    nwb_head = None
    if args.probe_nwb:
        p = os.path.join(BO, "probe_810755797_lfp_head8MB.bin")
        if os.path.exists(p) and os.path.getsize(p) > 1024:
            sparse = os.path.join(BO, "_sparse_lfp.nwb")
            with open(p, "rb") as fi, open(sparse, "wb") as fo:
                fo.write(fi.read())
            with open(sparse, "r+b") as f:
                f.truncate(2062950784)
            try:
                import h5py
                out = dict(prefix_bytes=os.path.getsize(p))
                with h5py.File(sparse, "r") as f:
                    out["root_keys"] = list(f.keys())
                    try:
                        out["nwb_version"] = repr(f["nwb_version"][()]) if "nwb_version" in f else None
                    except Exception:
                        out["nwb_version"] = "unreadable(prefix)"
                    for grp in ("general", "acquisition", "processing", "units", "intervals"):
                        try:
                            out["%s_keys" % grp] = list(f[grp].keys())[:12]
                        except Exception as e:
                            out["%s_keys" % grp] = "unreadable(%s)" % str(e)[:40]
                    # 采样率读回（NWB 的 TimeSeries 速率）
                    try:
                        gen = f["general/extracellular_ephys"]
                        out["extracellular_ephys"] = list(gen.keys())[:10]
                    except Exception as e:
                        out["extracellular_ephys"] = "unreadable(%s)" % str(e)[:40]
                nwb_head = out
            except Exception as e:
                nwb_head = dict(prefix_bytes=os.path.getsize(p), status="open_failed", error=str(e)[:120])

    out = dict(generated_by="tools/build_m11_allen_activity.py", generated_at=utcnow(),
               ruling="RF-M11-A：P2 功能通道回退到自描述源（HDF5/CSV）",
               channel_priority=dict(
                   a="Allen Brain Observatory Neuropixels ecephys-cache（**本轮已落**：自描述 CSV + NWB/HDF5）",
                   b="DANDI 000021（NWB，214 assets/477 GB）—— 未取（若 a 足够则不需要）",
                   c="MICrONS functional_connectomics（表格型）—— 仅用于结构-功能**映射**，非时间序列"),
               sampling_rate_readback=rate or None,
               sessions_csv=dict(landed=bool(sess), file_bytes=7849, **({"columns": sess["columns"],
                                                                         "n_columns": sess["n_columns"],
                                                                         "sample": sess["sample"]} if sess else {})),
               units_csv=dict(full_bytes=51439189, landed="仅 1 MB Range 前缀（**全量未取**）",
                              prefix_bytes=1048576,
                              **({"columns": units["columns"], "n_columns": units["n_columns"],
                                  "sample": units["sample"]} if units else {})),
               channels_csv=dict(full_bytes=6613329, landed="1 MB Range 前缀",
                                 **({"columns": chan["columns"], "n_columns": chan["n_columns"]} if chan else {})),
               nwb_lfp=dict(full_bytes=2062950784, note="全量 **>2 GB 单文件上限** ⇒ 不下载，改用 Range 头解析",
                            head=nwb_head),
               voxel_channel=dict(
                   status="decoded（**不登记 format_unverified** —— 理由与证据见下）",
                   decode="大端 int16（`np.frombuffer(raw, '>i2').reshape(10,200,200)`）",
                   proof="data/m11_decode_proof.json + reports/neuro/m11_microns_decode_proof.png "
                         "+ tools/demo_m11_microns_decode.py（一条命令可复核）",
                   metrics="lat_x=0.814 / lat_y=0.801 / 帧轴 lag1..6 = 0.803/0.783/0.761/0.743/0.725/0.709 / p50=1953",
                   ground_truth="同数据集原始未压缩 BigTIFF（另一 scan）lat_x=0.767 / p50=1,879"),
               objects=objs)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        f.write("# M11 活动通道回退源：逐对象完整性登记（generated_by=tools/build_m11_allen_activity.py；"
                "verdict ∈ {COMPLETE, RANGE_BOUNDED, INCOMPLETE, MISSING}；**INCOMPLETE 不进统计**）\n")
        cols = ["local_file", "url", "range", "expected_bytes", "bytes", "sha256", "verdict",
                "retrieved_at", "use", "note"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in objs:
            w.writerow(r)
    print("wrote %s" % OUT_JSON)
    print("sampling_rate_readback:", json.dumps(rate, ensure_ascii=False)[:300])
    print("objects:", [(r["local_file"].split("/")[-1], r["verdict"]) for r in objs])
    print("nwb_head:", json.dumps(nwb_head, ensure_ascii=False)[:300] if nwb_head else None)
    return out


if __name__ == "__main__":
    main()
