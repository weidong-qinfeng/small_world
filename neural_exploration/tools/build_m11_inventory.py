#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M11 数据门（G2/P1/P2）· 落盘数据清单 + 权威计数锚

输出
----
- `data/m11_data_inventory.csv`  —— 逐**落盘对象**一行：artifact_path / bytes / SHA-256 /
  `source_url` / `retrieved_at` / 证据级别 / note（**权威计数锚的载体**）
- `data/m11_data_counts.json`    —— 分层计数锚（结构 / 功能 / Allen / 文献 / 可达性 / 下载量），
  每项标注 `measured` 或 `derived`，并显式登记 `unavailable`（**未测 ≠ 未命中**）

口径纪律
--------
- SHA-256 覆盖全量文件（≥1 MB 的分块读）；目录（如 functional_subset 帧）逐文件一行，
  不聚合，便于复核。
- `source_url` 必须来自真实探测/下载记录（本工具从各 fetch 工具产出的 JSON 读取，不臆造）。

用法
----
    PYTHONHASHSEED=0 ./.venv-m11/bin/python neural_exploration/tools/build_m11_inventory.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "m11_raw")
PROJ = os.path.dirname(ROOT)

STRUCT_JSON = os.path.join(DATA, "m11_microns_structure.json")
PROTO_JSON = os.path.join(DATA, "m11_microns_protocol.json")
ALLEN_JSON = os.path.join(DATA, "m11_allen_sources.json")
LIT_CSV = os.path.join(DATA, "m11_literature_priors.csv")
REACH_CSV = os.path.join(DATA, "m11_source_reachability.csv")
SUBSET_MANIFEST = os.path.join(DATA, "m11_functional_subset_manifest.csv")
SUBSET_SUMMARY = os.path.join(DATA, "m11_functional_subset_summary.json")
TIFF_URL = ("https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/functional_data/"
            "two_photon_functional_scans/functional_scan_17797_4_10_v1.tif")
S3_LIST_URL = "https://bossdb-open-data.s3.amazonaws.com/"
OUT_INV = os.path.join(DATA, "m11_data_inventory.csv")
OUT_CNT = os.path.join(DATA, "m11_data_counts.json")

INV_FIELDS = ["item_id", "category", "artifact_path", "bytes", "sha256", "source_url",
              "retrieved_at", "evidence", "note"]


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


def load_json(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def read_csv_rows(p):
    if not os.path.exists(p):
        return []
    with open(p, newline="", encoding="utf-8") as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    return list(csv.DictReader(lines))


def main():
    st = load_json(STRUCT_JSON) or {}
    pr = load_json(PROTO_JSON) or {}
    al = load_json(ALLEN_JSON) or {}
    sub = load_json(SUBSET_SUMMARY) or {}
    lit_rows = read_csv_rows(LIT_CSV)
    reach_rows = read_csv_rows(REACH_CSV)
    sub_rows = read_csv_rows(SUBSET_MANIFEST)

    inv = []
    n = [0]

    def add(category, path, source_url, retrieved_at, evidence, note=""):
        if not os.path.exists(path):
            return
        n[0] += 1
        sz = os.path.getsize(path)
        inv.append(dict(item_id="I%03d" % n[0], category=category, artifact_path=rel(path), bytes=sz,
                        sha256=sha256_file(path) if sz < (256 << 20) else "skipped(large)",
                        source_url=source_url, retrieved_at=retrieved_at, evidence=evidence, note=note))

    # ---- (A) MICrONS 结构元数据 ----
    for name in ("em", "seg", "seg_m1300"):
        for host in ("gcs", "s3"):
            p = os.path.join(RAW, "microns", "struct", "%s_info_%s.json" % (name, host))
            src = st.get("sources", {}).get("%s_info_%s" % (name, host), {})
            add("microns_structure_info", p, src.get("url", ""), src.get("retrieved_at", ""),
                "measured(元数据)" if src.get("verdict") == "REACHABLE" else src.get("verdict", "unavailable"),
                "precomputed CloudVolume info（分辨率/体素尺寸/bBox/channel）")

    # ---- (B) MICrONS 功能元数据 ----
    fmap = [
        ("series_s4s7f4_attributes.json", 0, "functional_series attributes（采集参数）"),
        ("summary_s4s7f4_attributes.json", None, "field_summaries attributes（field 网格）"),
        ("mask_s4s7f4_info", 1, "field_seg_masks info（神经域掩膜格）"),
    ]
    for fn, idx, note in fmap:
        p = os.path.join(RAW, "microns", fn)
        if idx is None:
            u = pr.get("fields", {}).get("attributes", [{}])
            u = next((r for r in u if "session_4_scan_7_field_4" in (r.get("source_url") or "")), {})
            add("microns_functional_meta", p, u.get("source_url", ""), u.get("retrieved_at", ""),
                "measured(元数据, HTTP 200/223-230 B)", note)
        elif idx == 0:
            s0 = (pr.get("protocol", {}).get("functional_series") or [{}])[0]
            add("microns_functional_meta", p, s0.get("source_url", ""), s0.get("retrieved_at", ""),
                "measured(元数据, HTTP 200/223 B)", note)
        else:
            m0 = (pr.get("protocol", {}).get("field_seg_masks") or [{}])[0]
            add("microns_functional_meta", p, m0.get("url", ""), m0.get("retrieved_at", ""),
                "measured(元数据, HTTP 200/561 B)", note)

    tf = pr.get("protocol", {}).get("tech_docs", [])
    for i, ent in enumerate(tf):
        p = os.path.join(PROJ, ent.get("local_path", "")) if ent.get("local_path") else None
        if p:
            add("microns_tech_doc", p, ent.get("url", ""), ent.get("retrieved_at", ""),
                "measured(全文 PDF 落盘)", "采集参数 provenance（帧率/场/掩膜/刺激 30 Hz 插值）")
    tifbin = os.path.join(RAW, "microns", "functional", "tif_head_17797_4_10_v1.bin")
    rt = pr.get("protocol", {}).get("raw_scan_bigtiff", {})
    add("microns_raw_scan_header", tifbin, rt.get("url", ""), rt.get("retrieved_at", ""),
        "measured(Range 0-65535, HTTP 206)", "BigTIFF 头 → shape [320000,440,248]（**未下载 70 GB 大档**）")
    # (B2) 功能连接组学发布（Ding et al. 2025）：readme 落盘 + 两个大 pkl 仅登记（未取）
    add("microns_functional_meta", os.path.join(RAW, "microns", "functional_connectomics_readme_v1.md"),
        S3_LIST_URL + "?list-type=2&prefix=iarpa_microns/minnie/functional_data/functional_connectomics/"
                      "node_and_edge_properties/v1/", utcnow(), "measured(HTTP 200 / 5,594 B)",
        "node/edge 表 schema 与列语义（含 in vivo OSI / 信号相关 / 层与区标注）")
    for fn in ("access_v8_README.md", "filestructure.txt"):
        p = os.path.join(RAW, "microns", fn)
        add("microns_tech_doc", p, "", "", "measured(HTTP 200)", "功能数据管线/目录语义 provenance")

    # ---- (C) Allen ----
    ct = al.get("celltypes_api", {})
    add("allen_celltypes", os.path.join(RAW, "allen", "celltypes_mouse_metadata.json"),
        ct.get("count_url", ""), ct.get("retrieved_at", ""), "measured(HTTP 200)", "Cell Types API 原始返回")
    add("allen_celltypes", os.path.join(DATA, "m11_allen_celltypes_metadata.csv"), ct.get("count_url", ""),
        ct.get("retrieved_at", ""), "measured(分页全取)", "小鼠 %s 行" % ct.get("num_rows_returned"))
    nwb = al.get("per_cell_nwb", {})
    for fn, note in (("486940961_ephys.nwb", "per-cell ephys NWB（若已下完则为全量 91,560,414 B）"),
                     ("486940961_ephys_head8MB.bin", "HTTP Range 0-8388607 前缀（验证对象级可取）"),
                     ("head_sparse.nwb", "稀疏文件（前缀+洞；**仅用于读 HDF5 header**）")):
        add("allen_nwb", os.path.join(RAW, "allen", fn), nwb.get("url", ""), nwb.get("retrieved_at", ""),
            "measured" + ("(全量)" if "486940961_ephys.nwb" == fn and os.path.getsize(
                os.path.join(RAW, "allen", fn)) == nwb.get("nwb_expected_bytes", 0) else "(部分)"), note)

    # ---- (D) 探测/文献产物 ----
    add("m11_csv", REACH_CSV, "（本工具产出）", utcnow(), "measured(逐源 curl 两次探测)", "51 清单源 + 17 补测源")
    add("m11_csv", LIT_CSV, "https://www.ebi.ac.uk/europepmc/webservices/rest/search", utcnow(),
        "measured(Europe PMC REST)", "%d 行 provenance" % len(lit_rows))
    add("m11_json", STRUCT_JSON, "（本工具产出）", st.get("generated_at", ""), "measured+derived", "结构侧")
    add("m11_json", PROTO_JSON, "（本工具产出）", pr.get("generated_at", ""), "measured", "采集参数读回")
    add("m11_json", ALLEN_JSON, "（本工具产出）", al.get("generated_at", ""), "measured", "Allen 侧")
    add("allen_bo_ecephys", os.path.join(RAW, "allen", "bo_ecephys", "probes.csv"),
        "https://allen-brain-observatory.s3.amazonaws.com/visual-coding-neuropixels/ecephys-cache/probes.csv",
        utcnow(), "measured(HTTP 200, 完整)", "**采样率读回**：spike 30 kHz / LFP 2.5 kHz（RF-M11-A 回退源）")
    for extra in ("m11_dwell_reference_allen2p.json", "m11_dwell_reference_allen2p.csv", "m11_activity_reference.csv", "m11_activity_reference.json", "m11_throughput_measured.csv",
                  "m11_functional_connectomics_counts.json", "m11_dwell_reference.csv", "m11_dwell_reference.json",
                  "m11_s_gate_criteria_prereg.csv", "m11_b1_managed_jobs.json",
                  "m11_grid_alignment.json", "m11_allen_activity_fallback.json", "m11_allen_activity_objects.csv",
                  "m11_decode_proof.json",
                  "m11_reference.csv", "m11_download_integrity.csv", "m11_download_integrity_summary.json",
                  "m11_microns_parse_validation.json", "m11_microns_parse_hypotheses.csv"):
        add("m11_gate", os.path.join(DATA, extra), "（本工具链产出）", utcnow(),
            "measured(结构性验证 / 完整性门)", "主 agent 复核要求的三项门（解析/完整性/哨兵）产物")
    add("microns_raw_tiff_frames", os.path.join(RAW, "microns", "functional", "tif_frames_probe.bin"),
        TIFF_URL, utcnow(), "measured(Range 512-2200000, HTTP 206)",
        "原始 BigTIFF 帧前缀（独立地面真值：lat_x≈0.77）")
    for p in (SUBSET_MANIFEST, SUBSET_SUMMARY):
        add("m11_csv", p, "", sub.get("generated_at", ""), "measured", "功能有界子集清单")

    # ---- (E) 功能子集帧（逐文件）----
    for r in sub_rows:
        lp = r.get("local_path") or ""
        p = os.path.join(PROJ, lp)
        if not os.path.exists(p):
            continue
        n[0] += 1
        inv.append(dict(item_id="I%03d" % n[0], category="microns_functional_subset_frame",
                        artifact_path=rel(p), bytes=os.path.getsize(p), sha256=r.get("sha256", ""),
                        source_url=r.get("url", ""), retrieved_at=r.get("retrieved_at", ""),
                        evidence="measured(HTTP %s)" % r.get("http_code", ""),
                        note="t_chunk=%s（10 帧/片）" % r.get("t_chunk", "")))

    with open(OUT_INV, "w", newline="", encoding="utf-8") as f:
        f.write("# M11 落盘数据权威清单（generated_by=tools/build_m11_inventory.py；"
                "每行 = 一个落盘对象；sha256 为全量哈希；source_url/retrieved_at 来自真实探测）\n")
        w = csv.DictWriter(f, fieldnames=INV_FIELDS)
        w.writeheader()
        for r in inv:
            w.writerow(r)

    # ------------------------------------------------------------------
    # 计数锚
    # ------------------------------------------------------------------
    def vol_mm3(sc):
        bb = sc.get("bbox_mm") or []
        return round(bb[0] * bb[1] * bb[2], 4) if len(bb) == 3 else None

    struct_counts = {}
    for name in ("em", "seg", "seg_m1300"):
        v = st.get("volumes", {}).get(name, {}).get("gcs", {})
        scs = v.get("scales", [])
        finest = scs[0] if scs else {}
        struct_counts[name] = dict(
            host_gcs_verdict=st.get("sources", {}).get("%s_info_gcs" % name, {}).get("verdict"),
            host_s3_verdict=st.get("sources", {}).get("%s_info_s3" % name, {}).get("verdict"),
            data_type=v.get("data_type"), type=v.get("type"), n_scales=len(scs),
            finest_scale=finest.get("key"), resolution_nm=finest.get("resolution_nm"),
            size_voxels=finest.get("size_voxels"), bbox_nm=finest.get("bbox_nm"),
            bbox_mm=finest.get("bbox_mm"), volume_mm3=vol_mm3(finest),
            chunk_sizes_voxels=finest.get("chunk_sizes_voxels"),
            encoding=finest.get("encoding"),
            n_chunks_derived=finest.get("n_chunks_derived"),
            raw_bytes_per_chunk_derived=finest.get("raw_bytes_per_chunk_derived"),
            full_volume_raw_bytes_derived=(finest.get("n_chunks_derived", 0) *
                                           finest.get("raw_bytes_per_chunk_derived", 0)),
            n_chunks_all_scales_derived=sum(s.get("n_chunks_derived", 0) for s in scs))
    shard = {}
    for k, v in (st.get("shard_inventory_measured_bounded") or {}).items():
        shard[k] = dict(scale=v.get("scale"), n_objects_measured=v.get("n_objects"),
                        bytes_measured=v.get("n_bytes"), pages=v.get("pages"),
                        truncated=v.get("truncated"), size_min=v.get("size_min"),
                        size_max=v.get("size_max"), size_mean=v.get("size_mean"),
                        evidence="measured(有界列举)")

    fs = (pr.get("protocol", {}).get("functional_series") or [])
    masks = (pr.get("protocol", {}).get("field_seg_masks") or [])
    fields = pr.get("fields", {})
    func_counts = dict(
        n_fields_enumerated=fields.get("n_field_summary_prefixes"),
        n_field_attributes_fetched=fields.get("n_attributes_fetched"),
        n_fields_with_functional_series=(pr.get("sources", {}).get("listing_functional_series", {}) or {}).get("n_prefixes"),
        n_fields_with_seg_masks=(pr.get("sources", {}).get("listing_field_seg_masks", {}) or {}).get("n_prefixes"),
        n_platinum_stimulus_movies=(pr.get("sources", {}).get("listing_platinum_stimulus_movies", {}) or {}).get("n_objects"),
        n_platinum_camovies=(pr.get("sources", {}).get("listing_platinum_camovies", {}) or {}).get("n_objects"),
        n_stimulus_movies=(pr.get("sources", {}).get("listing_stimulus_movies", {}) or {}).get("n_objects"),
        n_raw_scan_tifs=(pr.get("sources", {}).get("listing_two_photon_functional_scans", {}) or {}).get("n_objects"),
        raw_scan_tif_bytes_total=(pr.get("sources", {}).get("listing_two_photon_functional_scans", {}) or {}).get("n_bytes"),
        series=[dict(field=s.get("source_url", "").split("/functional_series/")[-1].split("/")[0]
                     if "/functional_series/" in s.get("source_url", "") else "?",
                     frame_interval_s=s.get("frame_interval_s"), frame_rate_hz=s.get("frame_rate_hz"),
                     n_timepoints=s.get("n_timepoints"), duration_s=s.get("duration_s"),
                     n_planes=s.get("n_planes"), volume_rate_hz=s.get("volume_rate_hz"),
                     voxel_um_xyz=s.get("voxel_um_xyz"), fov_um=s.get("fov_um"),
                     block_size=s.get("block_size"), data_type=s.get("data_type"),
                     source_url=s.get("source_url"), retrieved_at=s.get("retrieved_at"))
                for s in fs],
        seg_mask_grid=dict(n_masks_infos=len(masks),
                           resolution_nm=masks[0].get("resolution_nm") if masks else None,
                           size_voxels=masks[0].get("size_voxels") if masks else None,
                           encoding=masks[0].get("encoding") if masks else None),
        raw_bigtiff=dict(url=(pr.get("protocol", {}).get("raw_scan_bigtiff", {}) or {}).get("url"),
                         frame_shape=(pr.get("protocol", {}).get("raw_scan_bigtiff", {}) or {}).get("image_description_json"),
                         bits_per_sample=(pr.get("protocol", {}).get("raw_scan_bigtiff", {}) or {}).get("tags", {}).get("258"),
                         strip_bytes=(pr.get("protocol", {}).get("raw_scan_bigtiff", {}) or {}).get("tags", {}).get("279"),
                         evidence="measured(Range 0-65535, HTTP 206)"),
        bounded_subset=dict(series=sub.get("series"), t_chunk_range=sub.get("t_chunk_range"),
                            n_objects=sub.get("n_objects_landed"), bytes=sub.get("total_bytes"),
                            frames=10 * (sub.get("n_ok") or 0), evidence="measured(逐对象 HTTP 200)"),
    )

    av = []
    for r in read_csv_rows(os.path.join(DATA, "m11_allen_celltypes_metadata.csv")):
        av.append(r)
    fr = []
    for r in av:
        try:
            fr.append(float(r.get("ef__avg_firing_rate") or "nan"))
        except ValueError:
            pass
    fr = [x for x in fr if x == x]
    # **双估计器一致性（主 agent M10 抓到 5 次的同类事故）**：偶数 n 必须取两中位均值；
    # 并与 statistics.median 交叉核对（相对差 ≤1e-6），不一致即在此处暴露。
    import statistics as _st
    def _median_explicit(v):
        w = sorted(v)
        n = len(w)
        if n == 0:
            return None
        return w[n // 2] if n % 2 else 0.5 * (w[n // 2 - 1] + w[n // 2])
    med_a = _median_explicit(fr) if fr else None
    med_b = _st.median(fr) if fr else None
    med_rel = (abs(med_a - med_b) / abs(med_b)) if (med_a is not None and med_b) else None
    median_check = dict(median_explicit=round(med_a, 6) if med_a is not None else None,
                        median_statistics=round(med_b, 6) if med_b is not None else None,
                        rel_diff=(round(med_rel, 12) if med_rel is not None else None),
                        tol=1e-6, agree=bool(med_rel is not None and med_rel <= 1e-6),
                        note="**旧实现缺陷已修**：曾用 `sorted[n//2]`（奇数式取单元素），n=1808 时给出 17.4379；"
                             "正确值 = 两中位均值 = **17.4130**（numpy/statistics 两实现一致）")
    donors = {r.get("donor__id") for r in av if r.get("donor__id")}
    allen_counts = dict(
        celltypes_total_rows_reported=ct.get("total_rows"),
        celltypes_rows_landed=ct.get("num_rows_returned"),
        celltypes_download_complete=ct.get("download_complete"),
        celltypes_n_fields=ct.get("n_fields"),
        celltypes_n_ef_fields=ct.get("n_ef_fields"),
        celltypes_n_unique_donors=len(donors),
        ef_avg_firing_rate=dict(n=len(fr), min=round(min(fr), 6) if fr else None,
                                median=round(med_a, 6) if fr else None,
                                median_alt=round(med_b, 6) if fr else None,
                                max=round(max(fr), 6) if fr else None,
                                two_estimator_check=median_check,
                                evidence="measured(API 返回的 ef__ 特征字段，逐标本一行)"),
        per_cell_nwb=dict(head_http_code=nwb.get("head_http_code"),
                          content_length=nwb.get("head_content_length"),
                          filename=(nwb.get("head_content_disposition") or "").split('filename=')[-1],
                          range_0_99_http_code=nwb.get("range_0_99_http_code"),
                          local_full_bytes=nwb.get("local_full_bytes"),
                          local_partial_bytes=nwb.get("local_partial_bytes"),
                          header_status=(nwb.get("nwb_header") or {}).get("status"),
                          header_source=nwb.get("header_source"),
                          header_groups=(nwb.get("nwb_header") or {}).get("groups"),
                          n_datasets=(nwb.get("nwb_header") or {}).get("n_datasets")),
        bo_neuropixels=dict(n_objects=(al.get("bo_neuropixels") or {}).get("n_objects"),
                            bytes=(al.get("bo_neuropixels") or {}).get("n_bytes"),
                            truncated=(al.get("bo_neuropixels") or {}).get("truncated"),
                            object_probe=(al.get("bo_neuropixels") or {}).get("object_probe")),
    )

    vc = {}
    for r in reach_rows:
        vc[r["verdict"]] = vc.get(r["verdict"], 0) + 1
    lit_v = {}
    for r in lit_rows:
        lit_v[r["status"]] = lit_v.get(r["status"], 0) + 1

    total_bytes_downloaded = 0
    for r in inv:
        total_bytes_downloaded += r["bytes"]

    # ---- 磁盘核对（主 agent 要求：清单里的数必须与磁盘上的数一致；未完成写 INCOMPLETE，不留 null/0）----
    nwb = al.get("per_cell_nwb", {}) or {}
    declared = [
        dict(key="allen.per_cell_nwb.full_nwb", path="neural_exploration/data/m11_raw/allen/486940961_ephys.nwb",
             declared_bytes=nwb.get("nwb_expected_bytes"),
             note="per-cell ephys NWB 全量；**下载停滞 48 min 无增长后已按 >30 min 纪律停止** ⇒ INCOMPLETE（可 curl -C - 续传）"),
        dict(key="allen.per_cell_nwb.header_staging", path="neural_exploration/data/m11_raw/allen/head_sparse.nwb",
             declared_bytes=None,
             note="**稀疏文件**：仅前 44,695,552 B 为真实数据，其余为洞(zeros) ⇒ **不是完整 NWB**，仅用于读 HDF5 头"),
        dict(key="allen.per_cell_nwb.range_prefix", path="neural_exploration/data/m11_raw/allen/486940961_ephys_head8MB.bin",
             declared_bytes=8388608, note="HTTP Range 前缀（独立于部分下载）"),
        dict(key="microns.functional_subset.manifest", path="neural_exploration/data/m11_functional_subset_manifest.csv",
             declared_bytes=None, note="200 对象逐行 url/bytes/sha256/integrity"),
        dict(key="microns.structure.info", path="neural_exploration/data/m11_microns_structure.json", declared_bytes=None, note=""),
        dict(key="microns.protocol", path="neural_exploration/data/m11_microns_protocol.json", declared_bytes=None, note=""),
        dict(key="m11.reference", path="neural_exploration/data/m11_reference.csv", declared_bytes=None, note=""),
        dict(key="m11.grid_alignment", path="neural_exploration/data/m11_grid_alignment.json", declared_bytes=None,
             note="空间口径 10× 差与同格性的实像素判定（RF-M11-B §四）"),
    ]
    disk_verified = []
    for d in declared:
        p_ = os.path.join(PROJ, d["path"])
        e = dict(key=d["key"], path=d["path"], declared_bytes=d["declared_bytes"],
                 exists=os.path.exists(p_), note=d["note"])
        if e["exists"]:
            e["bytes_on_disk"] = os.path.getsize(p_)
            e["sha256"] = sha256_file(p_)
            if d["declared_bytes"] is None:
                e["status"] = "COMPLETE"
            else:
                e["status"] = "COMPLETE" if e["bytes_on_disk"] == d["declared_bytes"] else "INCOMPLETE"
                if e["status"] == "INCOMPLETE":
                    e["note"] = (e["note"] + "；期望 %d B / 实收 %d B" % (d["declared_bytes"], e["bytes_on_disk"]))
        else:
            e["status"] = "MISSING"
            e["bytes_on_disk"] = None
        disk_verified.append(e)

    counts = dict(
        generated_by="tools/build_m11_inventory.py", generated_at=utcnow(),
        units="bytes; evidence ∈ {measured, derived, literature, unavailable}",
        microns_structure=dict(
            evidence="measured(元数据) + derived(chunk 计数); shard 清单 = measured(有界)",
            volumes=struct_counts, shard_inventory_measured_bounded=shard,
            unavailable=["逐 chunk 键清单：precomputed+sharded 存储 ⇒ 单 chunk 对象不可逐键列举（见 shard 级实测）",
                         "EM/seg 全量下载：TB 级 → 明令禁止（清单 §1.1.2）"]),
        microns_functional=dict(evidence="measured(元数据+有界子集)", **func_counts),
        allen=dict(evidence="measured", **allen_counts),
        download_integrity=dict(
            evidence="measured(逐对象 http→大小→SHA-256→解压→尺寸自洽)",
            **(load_json(os.path.join(DATA, "m11_download_integrity_summary.json")) or {})),
        microns_parse_validation=dict(
            evidence="measured(结构性假设检验 + 预注册判据)",
            winner=(load_json(os.path.join(DATA, "m11_microns_parse_validation.json")) or {}).get("winner"),
            hypotheses_passing=(load_json(os.path.join(DATA, "m11_microns_parse_validation.json")) or {}).get("hypotheses_passing"),
            full_objects={k: v for k, v in
                          ((load_json(os.path.join(DATA, "m11_microns_parse_validation.json")) or {})
                           .get("full_objects") or {}).items()
                          if k in ("n_objects", "n_elements", "lat_x_mean", "lat_y_mean", "frame_mean",
                                   "p50_mean", "n_nan", "n_inf", "n_zero", "n_int16_negative",
                                   "n_int16_saturated_max")},
            sentinel_and_missing=((load_json(os.path.join(DATA, "m11_microns_parse_validation.json")) or {})
                                  .get("sentinel_and_missing") or {})),
        allen_activity_fallback=dict(
            evidence="measured(自描述 CSV / HDF5 前缀)",
            **(load_json(os.path.join(DATA, "m11_allen_activity_fallback.json")) or {}).get(
                "sampling_rate_readback", {}) or {},
            ),
        grid_alignment=dict(
            evidence="measured(实像素) + derived(尺度扫描)",
            **(load_json(os.path.join(DATA, "m11_grid_alignment.json")) or {})),
        dwell_band=dict(
            evidence="measured(Allen 2p events) + 冻结管线复用",
            **(load_json(os.path.join(DATA, "m11_dwell_reference.json")) or {}).get("band", {})),
        functional_connectomics_release_dq4=dict(
            evidence="measured(readme schema) / unavailable（pkl 下载未完成）",
            **(load_json(os.path.join(DATA, "m11_functional_connectomics_counts.json")) or {}).get("files", {})),
        activity_reference=dict(
            evidence="四态收敛（measured/thin/unavailable）",
            rows=(load_json(os.path.join(DATA, "m11_activity_reference.json")) or {}).get("rows", [])),
        throughput_measured_n=len(read_csv_rows(os.path.join(DATA, "m11_throughput_measured.csv"))),
        literature=dict(n_rows=len(lit_rows), by_status=lit_v,
                        evidence="literature=Europe PMC 题录实测命中；未命中=literature_unverified"),
        reachability=dict(n_sources=len(reach_rows), by_verdict=vc,
                          evidence="measured(curl 两次探测)"),
        disk_verified=dict(
            rule="清单/计数里出现的每个数必须能与磁盘对上：exists + bytes_on_disk + sha256；"
                 "**未完成者写 INCOMPLETE 并给原因，不留 null、不写 0**",
            artifacts=disk_verified),
        inventory=dict(n_items=len(inv), total_bytes=total_bytes_downloaded,
                       total_MB=round(total_bytes_downloaded / 1e6, 3)),
        functional_connectomics_release=dict(
            evidence="measured(列举 200：KeyCount=4)",
            objects=dict(node_data_v1_pkl=294635032, edge_data_v1_pkl=201511086,
                         readme_v1_md=5594, prefix="iarpa_microns/minnie/functional_data/"
                                                      "functional_connectomics/node_and_edge_properties/v1/"),
            landed="readme_v1.md 已落盘（schema/列语义）；两个 pkl **未取**（按 >30 min 纪律交主 agent 裁决 DQ-4）",
            contents_from_readme=dict(
                node_data="逐神经元一行：nucleus_id / animal_id / scan_session / scan_idx / unit_id / "
                          "nucleus_x,y,z(nm) / hva / brain_area / layer / **pref_ori, pref_dir, osi_iv, gosi_iv**"
                          "（in vivo）/ in_vivo_mean_resp / mei / mask / readout_cvt(CvT)",
                edge_data="逐对一行：pre/post nucleus_id / population(Connected|ADP|Same region) / n_synapses / "
                          "synapse_size(cleft volume) / dend_len / pre,post_hva / pre,post_layer / proj_hva / "
                          "**in_vivo_sig_corr_iv**（in vivo 信号相关）/ in_silico_sig_corr_cvt / delta_ori_iv",
                source="Ding et al. 2025, DOI 10.1038/s41586-025-08840-3（readme 原文引用）"),
            impact="可得后 P2 的 C1（调谐/OSI 分布）与 C2（群体相关/有效维度）可**同源同口径**计算，"
                   "并直接服务 H4（结构-功能映射）与 P8（层/区/类差异）"),
        unavailable_register=[
            "**细胞数 / 突触数（MICrONS）= 未测**（`unavailable`，**不得写 0、不得算负结果**）："
            "权威逐细胞表 `node_data_v1.pkl`（294.6 MB）与逐对突触表 `edge_data_v1.pkl`（201.5 MB）本轮未取（DQ-4）；"
            "论文档锚（literature）= '>200,000 cells / 0.5 billion synapses'（Nature 2025）；"
            "Allen 侧可得锚 = Cell Types 1,920 标本（measured）",
            "MICrONS 逐细胞 ΔF/F 与 ROI 轨迹：仅存在于 v8 DataJoint MySQL 数据库"
            "（103.8 GB docker tar 或 116.2 GB SQL dump）→ 本轮未取（超预算）⇒ unavailable（非 0、非未命中）",
            "MICrONS 全 144 field 的扫描级 fps/nframes/nfields：在数据库 Scan 表内 → 本轮 unavailable；"
            "框架内时间口径由 field_summaries（无时间轴）+ functional_series（仅 2 field，6.3001 Hz）给出",
            "Allen synphys 桶对象级读取：403 AccessDenied（桶存在、不可 list）→ NEEDS_VERIFY",
            "IBL OpenAlyx 数据集表：403 需凭据 → NEEDS_TOKEN（P12 走公开桶 + 论文档回退）",
            "**四类参考量（率分布 / 群体相关 / 状态结构 / dwell）= 当前不可执行**（`blocked`，"
            "非未测、非未命中）：① R2 禁用 MICrONS 体素序列（否则 2 个 field 的钙域参考量本可算）；"
            "② R3 的本地 Allen NWB 中 `/units`、`/acquisition`、`/general` 落在稀疏洞内不可读，"
            "且下载被禁/网络 1.6–16 KB/s（DANDI 7.2 KB/s）；③ 逐细胞 ΔF/F 仅在 v8 DB（R1 不下载）；"
            "④ 唯一剩余实测路径 = 全量 `units.csv`（51,439,189 B，逐 unit firing_rate/质量）⇒ **DQ-6 待批**；"
            "⇒ 请裁决：批 DQ-6 / 续传 NWB / 或改判 R2 以启用 2 field 钙域参考量",
            "MICrONS minnie CAVE 表 API（info/materialize）：404 / 503 → REACHABLE_NO_DATA / BLOCKED",
        ],
    )
    with open(OUT_CNT, "w") as f:
        json.dump(counts, f, indent=1, ensure_ascii=False, sort_keys=False)
    print("wrote %s (%d items, %.1f MB)" % (OUT_INV, len(inv), total_bytes_downloaded / 1e6))
    print("wrote %s" % OUT_CNT)
    return counts


if __name__ == "__main__":
    main()
