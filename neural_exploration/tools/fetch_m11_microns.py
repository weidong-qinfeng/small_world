#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M11 数据门（G2/P1/P2）· MICrONS 结构与功能数据获取器（可复跑、有界、增量落盘）

定位（清单 §1.1 / §1.7 P1/P2 验收）
------------------------------------
本工具只做 G2 允许的三件事：**元数据**、**有界子集**、**权威计数锚**。
**禁止下载全量 EM/seg（TB 级）**；功能侧单个子集上限 `--subset-max-bytes`（默认 2 GB）。

子命令
------
  --struct      结构侧：em / seg / seg_m1300 的 `info`（GCS + S3 镜像）→ 解析分辨率/体素尺寸/
                bBox/channel → `data/m11_microns_structure.json`；并做**有界 shard 清单实测**
                （最细 scale 分片对象数/字节，`truncated` 如实标注）。
  --func-meta   功能侧元数据：功能数据根/各前缀列举、`functional_series/*/attributes.json`、
                `field_summaries/*/attributes.json`（逐 field）、`field_seg_masks/*/info`、
                原始 BigTIFF 头（Range 读取）、技术文档 PDF（v3/v8）→ 采集参数读回
                → `data/m11_microns_protocol.json`。
  --func-subset 功能侧**有界子集**：按预注册块数下载 `functional_series` chunk（默认 200 片
                ≈ 2000 帧 ≈ 317 s 电影），逐对象记录 URL/字节/SHA-256 →
                `data/m11_functional_subset_manifest.csv`（增量落盘 + `ALL_DONE`）。
  --pdf-text    从技术文档 PDF 抽取正文到 `data/m11_raw/microns/pdf_text/`（provenance 用）。

纪律
----
- 传输后端 = curl 子进程（M10 L11）；`--max-time` 硬超时；`PYTHONHASHSEED=0`
- 增量日志（`data/m11_raw/microns/fetch_m11_microns.log`）+ 结束 `ALL_DONE` 标记
- 下载并发上限 `--jobs`（默认 3；**不是**"并行起多个全量下载"——总子集上限 2 GB）

用法
----
    cd /Users/weidong/ai/small_world
    PYTHONHASHSEED=0 OMP_NUM_THREADS=1 ./.venv-m11/bin/python neural_exploration/tools/fetch_m11_microns.py --struct
    PYTHONHASHSEED=0 OMP_NUM_THREADS=1 ./.venv-m11/bin/python neural_exploration/tools/fetch_m11_microns.py --func-meta
    PYTHONHASHSEED=0 OMP_NUM_THREADS=1 ./.venv-m11/bin/python neural_exploration/tools/fetch_m11_microns.py --func-subset
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import struct as _struct
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "m11_raw", "microns")
STRUCT_RAW = os.path.join(RAW, "struct")
FUNC_RAW = os.path.join(RAW, "functional")
SUBSET_DIR = os.path.join(RAW, "functional_subset")
PDF_TEXT = os.path.join(RAW, "pdf_text")
LOG = os.path.join(RAW, "fetch_m11_microns.log")

GCS = "https://storage.googleapis.com/iarpa_microns"
S3 = "https://bossdb-open-data.s3.amazonaws.com"
S3_FUNC = S3 + "/iarpa_microns/minnie/functional_data"

STRUCT_VOLS = {
    "em": dict(gcs_key="minnie/minnie65/em", s3_key="iarpa_microns/minnie/minnie65/em", s3_scales=True),
    "seg": dict(gcs_key="minnie/minnie65/seg", s3_key="iarpa_microns/minnie/minnie65/seg", s3_scales=True),
    "seg_m1300": dict(gcs_key="minnie/minnie65/seg_m1300", s3_key="iarpa_microns/minnie/minnie65/seg_m1300",
                      s3_scales=False),
}


def log(msg, echo=True):
    line = "[%s] %s" % (datetime.now(timezone.utc).strftime("%H:%M:%S"), msg)
    os.makedirs(RAW, exist_ok=True)
    with open(LOG, "a", buffering=1) as f:
        f.write(line + "\n")
    if echo:
        print(line, flush=True)


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
_TMPSEQ = [0]


def curl(url, timeout=20, byte_range=None, out=None, extra=None):
    """返回 (http_code, nbytes, content_type, exit, time_total, body_bytes_or_None)."""
    cmd = ["curl", "-sS", "-L", "--connect-timeout", "8", "--max-time", str(timeout),
           "-w", "%{http_code}|%{size_download}|%{content_type}|%{time_total}"]
    if byte_range is not None:
        cmd += ["-r", "%d-%d" % byte_range]
    if extra:
        cmd += list(extra)
    if out:
        cmd += ["-o", out]
    cmd.append(url)
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=timeout + 10)
    except subprocess.TimeoutExpired:
        return (0, 0, "", 124, 0.0, None)
    parts = p.stdout.decode("utf-8", "replace").strip().split("|")
    code, n, ct, t = 0, 0, "", 0.0
    if len(parts) >= 4:
        try:
            code = int(parts[0]) if parts[0] else 0
            n = int(float(parts[1]))
            t = float(parts[3])
        except ValueError:
            pass
        ct = parts[2]
    body = None
    if not out and p.returncode == 0:
        body = None  # 未落盘的响应体不可得（curl 默认 stdout），本工具一律用 -o
    return (code, n, ct, p.returncode, t, body)


def get_bytes(url, timeout=20, byte_range=None):
    """下载到**唯一临时文件**（并发安全）后读回；文件随即删除。"""
    _TMPSEQ[0] += 1
    tmp = os.path.join(RAW, "_tmp_get_%d_%d.bin" % (os.getpid(), _TMPSEQ[0]))
    code, n, ct, rc, t, _ = curl(url, timeout=timeout, byte_range=byte_range, out=tmp)
    b = None
    if os.path.exists(tmp):
        with open(tmp, "rb") as f:
            b = f.read()
        os.remove(tmp)
    return code, n, ct, rc, t, b


def get_json(url, timeout=20):
    code, n, ct, rc, t, b = get_bytes(url, timeout=timeout)
    if b is None:
        return None, dict(http_code=code, nbytes=n, content_type=ct, curl_exit=rc, time_total_s=round(t, 3),
                          retrieved_at=utcnow(), url=url, verdict="BLOCKED")
    try:
        return json.loads(b.decode("utf-8")), dict(http_code=code, nbytes=n, content_type=ct, curl_exit=rc,
                                                   time_total_s=round(t, 3), retrieved_at=utcnow(), url=url,
                                                   sha256=hashlib.sha256(b).hexdigest(), verdict="REACHABLE")
    except Exception as e:
        return None, dict(http_code=code, nbytes=n, content_type=ct, curl_exit=rc, time_total_s=round(t, 3),
                          retrieved_at=utcnow(), url=url, verdict="REACHABLE_NO_DATA",
                          note="非 JSON（%s）" % str(e)[:80])


def list_prefix(bucket_url, prefix, delimiter="/", page_cap=10, max_keys=1000, timeout=20):
    """分页列举。返回 dict(n_objects, n_bytes, keys[:5], prefixes[:5], pages, truncated, page_sizes)."""
    tok, n_obj, n_by, pages, keys, prefixes, sizes = None, 0, 0, 0, [], [], []
    truncated = False
    while pages < page_cap:
        u = "%s/?list-type=2&max-keys=%d&prefix=%s" % (bucket_url, max_keys, prefix)
        if delimiter:
            u += "&delimiter=%s" % delimiter
        if tok:
            u += "&continuation-token=%s" % tok
        code, n, ct, rc, t, b = get_bytes(u, timeout=timeout)
        if b is None or code != 200:
            return dict(error="list failed http=%s curl=%s" % (code, rc), n_objects=n_obj, n_bytes=n_by,
                        pages=pages, truncated=True)
        x = b.decode("utf-8", "replace")
        ks = re.findall(r"<Key>([^<]*)</Key>", x)
        ss = [int(v) for v in re.findall(r"<Size>(\d+)</Size>", x)]
        n_obj += len(ks)
        n_by += sum(ss)
        sizes += ss
        keys += ks[:max(0, 5 - len(keys))]
        prefixes += re.findall(r"<CommonPrefixes><Prefix>([^<]*)</Prefix>", x)
        pages += 1
        m = re.search(r"<IsTruncated>(true|false)</IsTruncated>", x)
        m2 = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", x)
        if m and m.group(1) == "false":
            truncated = False
            break
        if not m2:
            truncated = False
            break
        tok = m2.group(1)
        truncated = True
    return dict(n_objects=n_obj, n_bytes=n_by, keys_sample=keys, prefixes_sample=prefixes[:5],
                prefixes_all=prefixes, n_prefixes=len(prefixes), pages=pages,
                truncated=truncated, size_min=min(sizes) if sizes else None, size_max=max(sizes) if sizes else None,
                size_mean=(sum(sizes) // len(sizes)) if sizes else None)


# ---------------------------------------------------------------------------
def derive_scale(s, dtype="uint8"):
    """由 precomputed info 的一个 scale 推 chunk/体素统计（**derived**，非实测）。"""
    res = [float(v) for v in s["resolution"]]
    size = [int(v) for v in s["size"]]
    ck = s.get("chunk_sizes", [[1, 1, 1]])[0]
    ck = [int(v) for v in ck]
    nchunks = 1
    for i in range(3):
        nchunks *= -(-size[i] // ck[i])
    vox_bytes = {"uint8": 1, "uint16": 2, "int16": 2, "uint32": 4, "float32": 4, "uint64": 8,
                 "int32": 4, "float64": 8}.get(dtype, 1)
    enc = s.get("encoding")
    raw_bytes_per_chunk = ck[0] * ck[1] * ck[2] * vox_bytes
    out = dict(key=s.get("key"), resolution_nm=res, size_voxels=size,
               voxel_offset=s.get("voxel_offset"),
               bbox_nm=[res[i] * size[i] for i in range(3)],
               bbox_mm=[round(res[i] * size[i] / 1e6, 4) for i in range(3)],
               chunk_sizes_voxels=ck, encoding=enc,
               n_chunks_derived=nchunks,
               raw_bytes_per_chunk_derived=raw_bytes_per_chunk,
               n_chunks_or_index_derived=nchunks)
    return out


def cmd_struct(args):
    out = dict(generated_by="tools/fetch_m11_microns.py --struct", generated_at=utcnow(),
               authority="MICrONS minimie65 precomputed CloudVolume info (GCS iarpa_microns + bossdb-open-data S3 镜像)",
               evidence_level="measured(元数据) / derived(chunk 计数)", volumes={}, sources={})
    os.makedirs(STRUCT_RAW, exist_ok=True)
    for name, cfg in STRUCT_VOLS.items():
        entry = dict(gcs={}, s3={})
        # GCS info（主通道）
        u = "%s/%s/info" % (GCS, cfg["gcs_key"])
        d, meta = get_json(u, timeout=args.timeout)
        meta["host"] = "gcs"; meta["volume"] = name
        out["sources"]["%s_info_gcs" % name] = meta
        if d:
            with open(os.path.join(STRUCT_RAW, "%s_info_gcs.json" % name), "w") as f:
                json.dump(d, f, indent=1, sort_keys=True)
            entry["gcs"] = dict(type=d.get("type"), data_type=d.get("data_type"),
                                num_channels=d.get("num_channels"),
                                scales=[derive_scale(sx, d.get("data_type", "uint8")) for sx in d.get("scales", [])])
        # S3 镜像 info
        u = "%s/%s/info" % (S3, cfg["s3_key"])
        d2, meta2 = get_json(u, timeout=args.timeout)
        meta2["host"] = "s3"; meta2["volume"] = name
        out["sources"]["%s_info_s3" % name] = meta2
        if d2:
            with open(os.path.join(STRUCT_RAW, "%s_info_s3.json" % name), "w") as f:
                json.dump(d2, f, indent=1, sort_keys=True)
            entry["s3"] = dict(type=d2.get("type"), data_type=d2.get("data_type"),
                               num_channels=d2.get("num_channels"),
                               scales=[derive_scale(sx, d2.get("data_type", "uint8"))
                                       for sx in d2.get("scales", [])])
        out["volumes"][name] = entry
        log("struct %s: gcs=%s s3=%s scales(gcs)=%d"
            % (name, meta["verdict"], meta2["verdict"],
               len(entry.get("gcs", {}).get("scales", []))))

    # ---- 有界 shard 清单实测（最细 scale）----
    inv = {}
    for name, cfg in STRUCT_VOLS.items():
        sc = out["volumes"][name].get("gcs", {}).get("scales", [])
        if not sc:
            continue
        finest = sc[0]["key"]
        pre = "%s/%s/" % (cfg["gcs_key"], finest)
        r = list_prefix(GCS, pre, delimiter="", page_cap=args.shard_pages, max_keys=1000)
        r["scale"] = finest
        r["prefix"] = pre
        r["host"] = "gcs"
        r["page_cap"] = args.shard_pages
        r["note"] = ("有界实测：仅列举前 %d 页（每页 1000 键）；truncated=true 表示**总量未测**，"
                     "不得写成总数" % args.shard_pages)
        inv[name] = r
        log("shard[%s/%s] objects=%s bytes=%s pages=%s truncated=%s"
            % (name, finest, r.get("n_objects"), r.get("n_bytes"), r.get("pages"), r.get("truncated")))
    out["shard_inventory_measured_bounded"] = inv
    out["chunk_count_derived_all_scales"] = {
        name: {s["key"]: dict(n_chunks=s["n_chunks_derived"], raw_B=s["raw_bytes_per_chunk_derived"])
               for s in out["volumes"][name].get("gcs", {}).get("scales", [])}
        for name in STRUCT_VOLS}
    with open(os.path.join(DATA, "m11_microns_structure.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=False)
    log("wrote data/m11_microns_structure.json")
    return out


# ---------------------------------------------------------------------------
def parse_attributes(js, base_url, retrieved_at):
    """functional_series / field_summaries 的 attributes.json → 采集参数读回（measured）。"""
    ax = js.get("axes", [])
    res = js.get("resolution", [])
    dims = js.get("dimensions", [])
    units = js.get("units", [])
    d = dict(source_url=base_url, retrieved_at=retrieved_at, axes=ax, dimensions=dims, resolution=res, units=units,
             data_type=js.get("dataType"), block_size=js.get("blockSize"),
             compression=(js.get("compression") or {}).get("type"))
    if "t" in ax and len(res) == len(ax):
        ti = ax.index("t")
        dt = float(res[ti])
        ni = ax.index("t") if "t" in ax else None
        n_t = int(dims[ti]) if ti < len(dims) else None
        d["frame_interval_s"] = dt
        d["frame_rate_hz"] = round(1.0 / dt, 6) if dt > 0 else None
        d["n_timepoints"] = n_t
        d["duration_s"] = round(dt * n_t, 3) if n_t else None
        d["duration_min"] = round(dt * n_t / 60.0, 3) if n_t else None
        zi = ax.index("z") if "z" in ax else None
        n_z = int(dims[zi]) if zi is not None and zi < len(dims) else 1
        d["n_planes"] = n_z
        d["volume_rate_hz"] = round((1.0 / dt) / max(1, n_z), 6)
        d["plane_interval_s"] = round(dt * max(1, n_z), 6)
        if "x" in ax and "y" in ax:
            xi, yi = ax.index("x"), ax.index("y")
            d["fov_um"] = [round(float(res[xi]) * int(dims[xi]), 3), round(float(res[yi]) * int(dims[yi]), 3)]
            d["voxel_um_xyz"] = [float(res[xi]), float(res[yi]), float(res[zi]) if zi is not None else None]
    return d


def cmd_func_meta(args):
    out = dict(generated_by="tools/fetch_m11_microns.py --func-meta", generated_at=utcnow(),
               evidence_level="measured（元数据 200 + 字节数 + SHA-256）", sources={}, fields={}, protocol={})
    os.makedirs(FUNC_RAW, exist_ok=True)

    # (a) 功能数据根 + 主要前缀
    for tag, pre in [("root", "iarpa_microns/minnie/functional_data/"),
                     ("functional_series", "iarpa_microns/minnie/functional_data/functional_series/"),
                     ("field_summaries", "iarpa_microns/minnie/functional_data/field_summaries/"),
                     ("field_seg_masks", "iarpa_microns/minnie/functional_data/field_seg_masks/"),
                     ("platinum_stimulus_movies", "iarpa_microns/minnie/functional_data/platinum_stimulus_movies/"),
                     ("platinum_camovies", "iarpa_microns/minnie/functional_data/platinum_camovies/"),
                     ("stimulus_movies", "iarpa_microns/minnie/functional_data/stimulus_movies/"),
                     ("two_photon_functional_scans", "iarpa_microns/minnie/functional_data/two_photon_functional_scans/"),
                     ("coregistration", "iarpa_microns/minnie/functional_coregistration/")]:
        r = list_prefix(S3, pre, delimiter="/", page_cap=args.list_pages, max_keys=1000)
        r.update(prefix=pre, retrieved_at=utcnow())
        out["sources"]["listing_%s" % tag] = r
        log("list %-28s objects=%-6s prefixes=%s truncated=%s"
            % (tag, r.get("n_objects"), r.get("n_prefixes"), r.get("truncated")))

    # (b) 逐 field attributes.json（field_summaries 是采集参数的权威落点）
    field_prefixes = out["sources"]["listing_field_summaries"].get("prefixes_all", [])
    recs = []
    todo = []
    for p in field_prefixes[:args.max_fields]:
        u = "%s/%sattributes.json" % (S3, p)
        todo.append(u)
    def one(u):
        d, meta = get_json(u, timeout=args.timeout)
        return u, d, meta
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for u, d, meta in ex.map(one, todo):
            if d is None:
                recs.append(dict(url=u, verdict=meta["verdict"], note="attributes 不可得"))
                continue
            pa = parse_attributes(d, u, meta["retrieved_at"])
            pa["verdict"] = meta["verdict"]
            pa["nbytes"] = meta["nbytes"]
            pa["sha256"] = meta.get("sha256")
            recs.append(pa)
    out["fields"]["n_field_summary_prefixes"] = len(field_prefixes)
    out["fields"]["n_attributes_fetched"] = len(recs)
    out["fields"]["attributes"] = recs
    rates = [r["frame_rate_hz"] for r in recs if r.get("frame_rate_hz")]
    planes = [r.get("n_planes") for r in recs if r.get("n_planes")]
    if rates:
        out["fields"]["frame_rate_hz_stats"] = dict(n=len(rates), min=min(rates), max=max(rates),
                                                    mean=round(sum(rates) / len(rates), 6))
    if planes:
        out["fields"]["n_planes_stats"] = dict(n=len(planes), min=min(planes), max=max(planes))

    # (c) 有 functional_series 的 field（原始钙电影体数据）——attributes + info
    series = out["sources"]["listing_functional_series"].get("prefixes_all", [])
    out["protocol"]["functional_series"] = []
    for p in series:
        d, meta = get_json("%s/%sattributes.json" % (S3, p), timeout=args.timeout)
        if d is None:
            continue
        pa = parse_attributes(d, "%s/%sattributes.json" % (S3, p), meta["retrieved_at"])
        pa["sha256"] = meta.get("sha256")
        pa["nbytes"] = meta["nbytes"]
        out["protocol"]["functional_series"].append(pa)
        log("series %s: fps=%s n_t=%s dur=%.1f min planes=%s"
            % (p.split("/")[-2], pa.get("frame_rate_hz"), pa.get("n_timepoints"),
               pa.get("duration_s", 0) / 60.0, pa.get("n_planes")))

    # (d) field_seg_masks info（神经域/掩膜定义）
    mask_prefs = out["sources"]["listing_field_seg_masks"].get("prefixes_all", [])
    out["protocol"]["field_seg_masks"] = []
    for p in mask_prefs[:args.max_fields]:
        d, meta = get_json("%s/%sinfo" % (S3, p), timeout=args.timeout)
        if d is None:
            continue
        sc = (d.get("scales") or [{}])[0]
        out["protocol"]["field_seg_masks"].append(dict(
            field=p.split("/")[-2], url="%s/%sinfo" % (S3, p), retrieved_at=meta["retrieved_at"],
            sha256=meta.get("sha256"), data_type=d.get("data_type"), type=d.get("type"),
            resolution_nm=sc.get("resolution"), size_voxels=sc.get("size"), chunk_sizes=sc.get("chunk_sizes"),
            encoding=sc.get("encoding"), key=sc.get("key"),
            n_masks_derived=int(sc.get("size", [1, 1, 1])[0] / sc.get("chunk_sizes", [[1, 1, 1]])[0][0])
            * int(sc.get("size", [1, 1, 1])[1] / sc.get("chunk_sizes", [[1, 1, 1]])[0][1])
            if sc.get("size") else None))

    # (e) 原始 BigTIFF 头（Range 读取；不下载 70 GB 大档）
    tif_url = S3_FUNC + "/two_photon_functional_scans/functional_scan_17797_4_10_v1.tif"
    code, n, ct, rc, t, b = get_bytes(tif_url, timeout=args.timeout, byte_range=(0, 65535))
    tif = dict(url=tif_url, byte_range="0-65535", http_code=code, nbytes=n, content_type=ct, curl_exit=rc,
               retrieved_at=utcnow())
    if b:
        with open(os.path.join(FUNC_RAW, "tif_head_17797_4_10_v1.bin"), "wb") as f:
            f.write(b)
        tif.update(sha256=hashlib.sha256(b).hexdigest(), bigtiff_header_hex=b[:32].hex())
        tif.update(parse_bigtiff_header(b, base_offset=0))
    out["protocol"]["raw_scan_bigtiff"] = tif
    log("raw BigTIFF head: http=%s n=%s shape=%s" % (code, n, tif.get("image_description_json")))

    # (f) 技术文档 PDF（v3 / v8）+ 文本抽取
    pdfs = [("functional_data_technical_documentation_v3.pdf",
             S3_FUNC + "/functional_data_technical_documentation_v3.pdf"),
            ("two_photon_processed_data_and_metadata_technical_documentation_v8.pdf",
             S3_FUNC + "/two_photon_processed_data_and_metadata/database_v8/"
                       "two_photon_processed_data_and_metadata_technical_documentation_v8.pdf")]
    out["protocol"]["tech_docs"] = []
    os.makedirs(PDF_TEXT, exist_ok=True)
    for name, u in pdfs:
        path = os.path.join(FUNC_RAW, name)
        code, n, ct, rc, t, b = get_bytes(u, timeout=args.timeout)
        ent = dict(url=u, http_code=code, nbytes=n, content_type=ct, retrieved_at=utcnow(),
                   local_path="data/m11_raw/microns/functional/%s" % name)
        if b:
            with open(path, "wb") as f:
                f.write(b)
            ent["sha256"] = hashlib.sha256(b).hexdigest()
            txt = pdf_to_text(path)
            if txt:
                with open(os.path.join(PDF_TEXT, name + ".txt"), "w") as f:
                    f.write(txt)
                ent["text_chars"] = len(txt)
                ent["text_local_path"] = "data/m11_raw/microns/pdf_text/%s.txt" % name
                ent["mentions"] = pdf_mentions(txt)
        out["protocol"]["tech_docs"].append(ent)
        log("tech doc %s http=%s n=%s text_chars=%s" % (name[:40], code, n, ent.get("text_chars")))

    # (g) 从文档正文中提取采集参数候选句（provenance；**不臆造**）
    out["protocol"]["doc_evidence"] = extract_acquisition_evidence()
    with open(os.path.join(DATA, "m11_microns_protocol.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=False)
    log("wrote data/m11_microns_protocol.json")
    return out


def parse_bigtiff_header(b, base_offset=0):
    """解析 BigTIFF 头（首个 IFD）→ 关键标签。Range 读到的前 64 KB 足够。"""
    try:
        if b[:2] == b"II":
            e = "<"
        elif b[:2] == b"MM":
            e = ">"
        else:
            return dict(parse_error="非 TIFF 魔数")
        magic = _struct.unpack_from(e + "H", b, 2)[0]
        if magic != 43:
            return dict(parse_error="非 BigTIFF（magic=%d）" % magic)
        offb, _res, first = _struct.unpack_from(e + "HHQ", b, 4)
        n = _struct.unpack_from(e + "Q", b, first)[0]
        out = dict(is_bigtiff=True, first_ifd=first, n_ifd_entries=n, tags={})
        p = first + 8
        for _ in range(n):
            tag, typ, cnt = _struct.unpack_from(e + "HHQ", b, p)
            vp = p + 12
            if typ == 2:
                raw = b[vp:vp + cnt] if cnt <= 8 else b[_struct.unpack_from(e + "Q", b, vp)[0]:][:cnt]
                s = raw.rstrip(b"\x00").decode("latin1", "replace")
                out["tags"][str(tag)] = s
            else:
                sz = {1: 1, 3: 2, 4: 4, 16: 8, 5: 8}.get(typ, 1)
                if cnt * sz <= 8:
                    if typ == 3:
                        out["tags"][str(tag)] = _struct.unpack_from(e + "H", b, vp)[0]
                    elif typ in (4,):
                        out["tags"][str(tag)] = _struct.unpack_from(e + "I", b, vp)[0]
                    else:
                        out["tags"][str(tag)] = _struct.unpack_from(e + "Q", b, vp)[0]
                else:
                    o = _struct.unpack_from(e + "Q", b, vp)[0]
                    if typ == 3:
                        out["tags"][str(tag)] = list(_struct.unpack_from(e + "H" * min(cnt, 4), b, o))
                    elif typ == 16:
                        out["tags"][str(tag)] = list(_struct.unpack_from(e + "Q" * min(cnt, 4), b, o))
                    else:
                        out["tags"][str(tag)] = "off=%d cnt=%d type=%d" % (o, cnt, typ)
            p += 20
        idj = out["tags"].get("270")
        if idj:
            try:
                out["image_description_json"] = json.loads(idj)
            except Exception:
                out["image_description"] = idj
        return out
    except Exception as ex:
        return dict(parse_error=str(ex)[:200])


def pdf_to_text(path):
    try:
        from pypdf import PdfReader
    except Exception:
        return None
    try:
        r = PdfReader(path)
        return "\n".join((pg.extract_text() or "") for pg in r.pages)
    except Exception:
        return None


PDF_KEYWORDS = ["frame rate", "frames per second", "fps", "Hz", "dwell", "scan", "volume rate", "plane",
                "neuropil", "d FoF", "dF/F", "GCaMP", "stimulus", "session", "field of view", "resonant",
                "galvo", "2.5", "0.158", "6.3", "mesoscope"]


def pdf_mentions(txt):
    return {k: txt.lower().count(k.lower()) for k in PDF_KEYWORDS if txt.lower().count(k.lower()) > 0}


EVID_PAT = re.compile(r"[^.\n]{0,160}(?:frame rate|frames per second|volume rate|dwell time|scan rate|"
                      r"resonant|galvo|neuropil|field of view|planes|sampling rate)[^.\n]{0,200}", re.I)


def extract_acquisition_evidence():
    """从已抽取的 PDF 文本中抓取"采集参数"相关句（关键词窗口，保留原文，不加工）。"""
    ev = {}
    if not os.path.isdir(PDF_TEXT):
        return ev
    for fn in sorted(os.listdir(PDF_TEXT)):
        if not fn.endswith(".txt"):
            continue
        txt = open(os.path.join(PDF_TEXT, fn), errors="replace").read()
        flat = re.sub(r"\s+", " ", txt)
        hits = EVID_PAT.findall(flat)
        ev[fn] = [h.strip() for h in hits[:20]]
    return ev


def cmd_func_subset(args):
    """有界子集：预注册块区间 [start, start+n_chunks) 逐片下载 + 逐对象 manifest。"""
    series = args.series
    base = "%s/functional_series/%s/0/0/0/" % (S3_FUNC, series)
    os.makedirs(SUBSET_DIR, exist_ok=True)
    man_path = os.path.join(DATA, "m11_functional_subset_manifest.csv")
    fields = ["idx", "t_chunk", "url", "http_code", "bytes", "sha256", "content_type", "time_total_s",
              "retrieved_at", "local_path", "verdict", "note"]
    log("subset start series=%s chunks=%d start=%d jobs=%d max_bytes=%d"
        % (series, args.n_chunks, args.start, args.jobs, args.subset_max_bytes))
    rows, total = [], 0
    with open(man_path, "w", newline="", encoding="utf-8") as f:
        f.write("# M11 MICrONS functional_series 有界子集 manifest（generated_by=tools/fetch_m11_microns.py "
                "--func-subset；series=%s；t_chunk ∈ [%d, %d)）\n"
                % (series, args.start, args.start + args.n_chunks))
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        f.flush()

        def fetch(i):
            tc = args.start + i
            url = base + str(tc)
            dst = os.path.join(SUBSET_DIR, "%s_%06d.bin" % (series, tc))
            code, n, ct, rc, t, _ = curl(url, timeout=args.timeout, out=dst)
            sha = None
            if os.path.exists(dst):
                h = hashlib.sha256()
                with open(dst, "rb") as fh:
                    for blk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(blk)
                sha = h.hexdigest()
            # **根因修复（主 agent 复核发现）**：只看 http_code+bytes 会把"curl 超时留下的半成品"
            # 记成 REACHABLE ⇒ 必须同时要求 curl_exit == 0；并就地做 gzip 完整性预检
            partial = False
            if code in (200, 206) and n > 0:
                try:
                    import gzip as _gz
                    _b = open(dst, "rb").read() if os.path.exists(dst) else b""
                    _off = _b.find(b"\x1f\x8b\x08", 0, 64)
                    _raw = _gz.decompress(_b[_off:]) if _off >= 0 else None
                    if _raw is None or len(_raw) != args.raw_chunk_bytes:
                        partial = True
                except Exception:
                    partial = True
            ok = (code in (200, 206)) and n > 0 and rc == 0 and not partial
            return dict(idx=i, t_chunk=tc, url=url, http_code=code, bytes=n, sha256=sha, content_type=ct,
                        time_total_s=round(t, 3), retrieved_at=utcnow(),
                        local_path=os.path.relpath(dst, os.path.dirname(ROOT)),
                        verdict="REACHABLE" if ok else "BLOCKED",
                        note="curl_exit=%s%s" % (rc, "; PARTIAL(解压/尺寸不符，需重取)" if partial else ""))

        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            for r in ex.map(fetch, range(args.n_chunks)):
                if r["verdict"] == "REACHABLE":
                    total += r["bytes"]
                w.writerow(r)
                f.flush()
                rows.append(r)
                if r["idx"] % 25 == 0 or r["idx"] == args.n_chunks - 1:
                    log("  subset %d/%d bytes_total=%.1f MB" % (r["idx"] + 1, args.n_chunks, total / 1e6))
                if total > args.subset_max_bytes:
                    log("!! subset_max_bytes 超限（%.1f MB）→ 停止（不静默超支）" % (total / 1e6))
                    break
    ok = sum(1 for r in rows if r["verdict"] == "REACHABLE")
    summary = dict(generated_by="tools/fetch_m11_microns.py --func-subset", generated_at=utcnow(),
                   series=series, t_chunk_range=[args.start, args.start + args.n_chunks],
                   n_requested=args.n_chunks, n_ok=ok, n_objects_landed=ok,
                   total_bytes=total, total_MB=round(total / 1e6, 3),
                   manifest="data/m11_functional_subset_manifest.csv",
                   block_shape_note="functional_series blockSize=[200,200,1,10] ⇒ 每片 = 10 帧（预注册）")
    with open(os.path.join(DATA, "m11_functional_subset_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True)
    log("subset done: ok=%d/%d bytes=%.1f MB" % (ok, args.n_chunks, total / 1e6))
    log("ALL_DONE %s" % utcnow())
    return summary


def cmd_retry_missing(args):
    """重取 `m11_functional_subset_manifest.csv` 中 verdict != REACHABLE 的片（**不重下已成功者**）。"""
    man = os.path.join(DATA, "m11_functional_subset_manifest.csv")
    with open(man, encoding="utf-8") as f:
        head_line = f.readline().rstrip("\n")
        rows = list(csv.DictReader(f))
    todo = [r for r in rows if r.get("verdict") != "REACHABLE"]
    log("retry-missing: n_total=%d n_missing=%d" % (len(rows), len(todo)))
    fixed = 0
    for r in todo:
        dst = os.path.join(os.path.dirname(ROOT), r["local_path"])
        code, n, ct, rc, t, _ = curl(r["url"], timeout=args.timeout, out=dst)
        sha = None
        if os.path.exists(dst) and n > 0:
            h = hashlib.sha256()
            with open(dst, "rb") as fh:
                for blk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(blk)
            sha = h.hexdigest()
        r.update(http_code=code, bytes=n, sha256=sha, content_type=ct, time_total_s=round(t, 3),
                 retrieved_at=utcnow(), note="retry: curl_exit=%s" % rc,
                 verdict="REACHABLE" if (code in (200, 206) and n > 0 and rc == 0) else "BLOCKED")
        fixed += 1 if r["verdict"] == "REACHABLE" else 0
        log("  retry t_chunk=%s http=%s bytes=%s verdict=%s" % (r["t_chunk"], code, n, r["verdict"]))
    with open(man, "w", newline="", encoding="utf-8") as f:
        f.write(head_line + "\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    ok = sum(1 for r in rows if r["verdict"] == "REACHABLE")
    total = sum(int(r["bytes"] or 0) for r in rows if r["verdict"] == "REACHABLE")
    summary = dict(generated_by="tools/fetch_m11_microns.py --func-subset(+retry-missing)",
                   generated_at=utcnow(), series=args.series,
                   t_chunk_range=[int(rows[0]["t_chunk"]), int(rows[-1]["t_chunk"]) + 1],
                   n_requested=len(rows), n_ok=ok, n_objects_landed=ok, total_bytes=total,
                   total_MB=round(total / 1e6, 3), manifest="data/m11_functional_subset_manifest.csv",
                   block_shape_note="functional_series blockSize=[200,200,1,10] ⇒ 每片 = 10 帧（预注册）")
    with open(os.path.join(DATA, "m11_functional_subset_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True)
    log("retry-missing done: ok=%d/%d bytes=%.1f MB" % (ok, len(rows), total / 1e6))
    return summary


def cmd_pdf_text(args):
    os.makedirs(PDF_TEXT, exist_ok=True)
    for fn in sorted(os.listdir(FUNC_RAW)):
        if fn.endswith(".pdf"):
            t = pdf_to_text(os.path.join(FUNC_RAW, fn))
            if t:
                with open(os.path.join(PDF_TEXT, fn + ".txt"), "w") as f:
                    f.write(t)
                log("pdf text %s -> %d chars" % (fn, len(t)))
    return extract_acquisition_evidence()


def cmd_shard_probe(args):
    """结构侧**对象级**验证：对最细 scale 的前若干 shard 做 Range 读取（**不下载大档**）。"""
    with open(os.path.join(DATA, "m11_microns_structure.json")) as f:
        st = json.load(f)
    out = {}
    for name, cfg in STRUCT_VOLS.items():
        sc = st["volumes"][name]["gcs"]["scales"][0]["key"]
        pre = "%s/%s/" % (cfg["gcs_key"], sc)
        lst = list_prefix(GCS, pre, delimiter="", page_cap=1, max_keys=5)
        keys = [k for k in lst.get("keys_sample", [])]
        ent = dict(scale=sc, prefix=pre, n_objects_first_page=lst.get("n_objects"), probes=[])
        for k in keys[:2]:
            u = "%s/%s" % (GCS, k)
            code, n, ct, rc, t, b = get_bytes(u, timeout=args.timeout, byte_range=(0, 1023))
            d = dict(key=k, url=u, http_code=code, bytes=n, content_type=ct, curl_exit=rc,
                     time_total_s=round(t, 3), retrieved_at=utcnow())
            if b:
                d["sha256_prefix"] = hashlib.sha256(b).hexdigest()
                d["first8_hex"] = b[:8].hex()
                # 自解压 shard 头（BigShard/压缩 shard）：看前 2 字节是否为 gzip magic
                d["gzip_magic"] = b[:2] == b"\x1f\x8b"
            d["verdict"] = "REACHABLE" if code in (200, 206) and n > 0 else "BLOCKED"
            ent["probes"].append(d)
        out[name] = ent
        log("shard probe %s: %s" % (name, [(x["http_code"], x.get("first8_hex")) for x in ent["probes"]]))
    st["shard_object_probe"] = out
    st["shard_object_probe_note"] = ("对象级 Range 读取（0-1023 B）证明 shard **真实可取**；"
                                     "**未下载**任何完整 shard（最大 4.4 GB 级）")
    with open(os.path.join(DATA, "m11_microns_structure.json"), "w") as f:
        json.dump(st, f, indent=1, sort_keys=False)
    log("wrote shard_object_probe into data/m11_microns_structure.json")
    return out


def cmd_verify_subset(args):
    """功能子集的**可解压验证**：gzip 解压若干片 → 核对 raw 字节数 == 块形状×dtype。"""
    import gzip
    man = os.path.join(DATA, "m11_functional_subset_manifest.csv")
    rows = []
    with open(man, encoding="utf-8") as f:
        for r in csv.DictReader([ln for ln in f if not ln.startswith("#")]):
            rows.append(r)
    specs = {s.get("source_url", ""): s for s in
             (json.load(open(os.path.join(DATA, "m11_microns_protocol.json")))
              .get("protocol", {}).get("functional_series") or [])}
    out = dict(generated_by="tools/fetch_m11_microns.py --verify-subset", generated_at=utcnow(),
               n_chunks_in_manifest=len(rows), checks=[])
    for r in rows[:args.verify_n]:
        p = os.path.join(os.path.dirname(ROOT), r["local_path"])
        d = dict(t_chunk=r["t_chunk"], url=r["url"], gz_bytes=os.path.getsize(p) if os.path.exists(p) else 0)
        # 实测口径（本轮新发现）：chunk = 20 B 大端头 [ndim,x,y,z,t] + **gzip 成员**
        #   （attributes.json 写 compression{type:gzip,useZlib:false}；实际为 gzip 头 1f 8b 08，
        #    前缀 20 B 为块形状声明）→ 本函数自动定位 gzip 魔数，**不硬编码偏移**
        blob = open(p, "rb").read() if os.path.exists(p) else b""
        raw, off, codec = None, -1, None
        try:
            import gzip as _gz
            import struct as _st
            off = blob.find(b"\x1f\x8b\x08", 0, 64)
            if off > 0:
                words = _st.unpack(">%dI" % (off // 4), blob[:off]) if off % 4 == 0 else ()
                d["chunk_prefix_bytes"] = off
                d["chunk_prefix_words"] = list(words)
            raw = _gz.decompress(blob[off:]) if off >= 0 else None
            codec = "gzip(at offset %d)" % off
        except Exception as e:
            d["decompress_error"] = str(e)[:160]
        d["decompress_codec"] = codec
        d["decompress_ok"] = raw is not None
        if raw is None:
            out["checks"].append(d)
            continue
        d["raw_bytes"] = len(raw)
        d["expected_raw_bytes"] = args.raw_chunk_bytes
        d["bytes_match"] = (len(raw) == args.raw_chunk_bytes)
        try:
            import numpy as np
            a = np.frombuffer(raw, dtype="<i2")
            u = a.view(np.uint16)
            d.update(n_voxels=int(a.size), int16_min=int(a.min()), int16_max=int(a.max()),
                     int16_mean=round(float(a.mean()), 4), nonzero_frac=round(float((a != 0).mean()), 4),
                     dtype_declared="int16（attributes.json dataType）",
                     uint16_alt=dict(min=int(u.min()), max=int(u.max()), mean=round(float(u.mean()), 3)),
                     frac_negative_int16=round(float((a < 0).mean()), 4))
            d["note"] = ("⚠️ **量纲/口径待核（NEEDS_VERIFY）**：按 int16 解释均值≈%.0f 且 %.0f%% 为负，"
                         "按 uint16 解释均值≈%.0f —— 两解均自洽；attributes.json 声明 int16 ⇒ "
                         "P2 定稿前必须核对（本节点不裁决）" % (a.mean(), 100 * (a < 0).mean(), u.mean()))
        except Exception as e:
            d["numpy_error"] = str(e)[:120]
        d["verdict"] = ("REACHABLE(可解压+字节数精确匹配)" if d.get("bytes_match")
                        else "REACHABLE_NO_DATA(可解压但字节数不匹配)")
        out["checks"].append(d)
    with open(os.path.join(DATA, "m11_functional_subset_verify.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    log("wrote data/m11_functional_subset_verify.json (%d checks)" % len(out["checks"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--struct", action="store_true")
    ap.add_argument("--func-meta", action="store_true")
    ap.add_argument("--func-subset", action="store_true")
    ap.add_argument("--pdf-text", action="store_true")
    ap.add_argument("--shard-probe", action="store_true")
    ap.add_argument("--verify-subset", action="store_true")
    ap.add_argument("--retry-missing", action="store_true")
    ap.add_argument("--verify-n", type=int, default=3)
    ap.add_argument("--raw-chunk-bytes", type=int, default=800000)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--jobs", type=int, default=3, help="下载并发（有界；默认 3）")
    ap.add_argument("--shard-pages", type=int, default=3, help="shard 清单最多列举页数（每页 1000 键）")
    ap.add_argument("--list-pages", type=int, default=10)
    ap.add_argument("--max-fields", type=int, default=300, help="逐 field attributes.json 上限")
    ap.add_argument("--series", default="session4_scan7_field4")
    ap.add_argument("--n-chunks", type=int, default=200)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--subset-max-bytes", type=int, default=2 * 1024 ** 3)
    args = ap.parse_args()
    for d in (RAW, STRUCT_RAW, FUNC_RAW, SUBSET_DIR, PDF_TEXT):
        os.makedirs(d, exist_ok=True)
    log("=== fetch_m11_microns start argv=%s PYTHONHASHSEED=%s" % (" ".join(sys.argv[1:]),
                                                                   os.environ.get("PYTHONHASHSEED", "(unset)")))
    if args.all or args.struct:
        cmd_struct(args)
    if args.all or args.func_meta:
        cmd_func_meta(args)
    if args.pdf_text:
        cmd_pdf_text(args)
    if args.shard_probe:
        cmd_shard_probe(args)
    if args.verify_subset:
        cmd_verify_subset(args)
    if args.all or args.func_subset:
        cmd_func_subset(args)
    if args.retry_missing:
        cmd_retry_missing(args)
    log("ALL_DONE %s" % utcnow())


if __name__ == "__main__":
    main()
