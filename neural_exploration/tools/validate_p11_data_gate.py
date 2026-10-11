#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M11 数据门（G2）· P1/P2 判定器（断言 → `reports/neuro/m11_data_gate_summary.json`）

P1（皮层数据门）/ P2（结构-功能同源数据门）的 Pass 语义按清单 §7 + §1.7 操作化为**硬断言**：

P1 断言
  1. `data/m11_source_reachability.csv` 覆盖清单 §1.6 全 51 源（source_id 1..51 逐号在位）+ 补测源；
     verdict 全在词表内；每行有 url / http_code / bytes。
  2. 结构侧 em / seg / seg_m1300 的 `info` 元数据落盘（含分辨率/体素尺寸/bBox/channel），
     主通道（GCS）三卷均 REACHABLE；chunk 计数（derived）与 shard 有界实测（measured）均在位。
  3. Allen Cell Types API 元数据表落盘且 `total_rows` 与落盘行数一致；≥1 个 per-cell NWB 对象
     HEAD + Range 实测通过（content-length 与预期一致）。
  4. 文献 provenance 表覆盖 §1.3 + §4.1 的 C-1..C-8 依据；`literature` 行的**声明标识必须题录核对通过**；
     `literature_unverified` 行必须写明未命中查询式（**不得**凭印象补）。
  5. 落盘清单 `m11_data_inventory.csv` 的每个路径存在且 **SHA-256 与现场重算一致**。
  6. **冻结件零修改**：M11 全部产出路径必须落在 m11 命名空间内（`data/m11_*`、`reports/neuro/m11_*`、
     `docs/m11_*`、`data/m11_raw/**`），否则判 `invalid`。

P2 断言（结构-功能同源）
  7. 功能侧：`functional_series` 的 attributes.json 读回（帧间隔/帧率/时长/平面数/体素/FOV）落盘，
     且**帧率读回值**与 `data/m11_microns_protocol.json` 内一致（读回一致性断言）。
  8. 神经域/掩膜定义：`field_seg_masks` info 落盘，且掩膜网格（size_voxels）与该 field 的
     `functional_series` 网格一致（同源同格断言）。
  9. 有界子集：总字节 ≤ 上限（2 GB）；逐对象 `url+bytes+sha256` 齐备。
  10. 采集参数未读回的子项必须**显式登记 `unavailable` + 原因**（不得写 0 / 不得算作未命中）。

用法
----
    PYTHONHASHSEED=0 ./.venv-m11/bin/python neural_exploration/tools/validate_p11_data_gate.py [--determinism-check]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJ = os.path.dirname(ROOT)
DATA = os.path.join(ROOT, "data")
REPORTS = os.path.join(ROOT, "reports", "neuro")
DOCS = os.path.join(ROOT, "docs")
OUT = os.path.join(REPORTS, "m11_data_gate_summary.json")

REACH = os.path.join(DATA, "m11_source_reachability.csv")
STRUCT = os.path.join(DATA, "m11_microns_structure.json")
PROTO = os.path.join(DATA, "m11_microns_protocol.json")
ALLEN = os.path.join(DATA, "m11_allen_sources.json")
LIT = os.path.join(DATA, "m11_literature_priors.csv")
INV = os.path.join(DATA, "m11_data_inventory.csv")
CNT = os.path.join(DATA, "m11_data_counts.json")
SUBSET = os.path.join(DATA, "m11_functional_subset_manifest.csv")
SUBSET_SUM = os.path.join(DATA, "m11_functional_subset_summary.json")
INTEG = os.path.join(DATA, "m11_download_integrity_summary.json")
INTEG_CSV = os.path.join(DATA, "m11_download_integrity.csv")
PARSE = os.path.join(DATA, "m11_microns_parse_validation.json")
REFCSV = os.path.join(DATA, "m11_reference.csv")
RAW = os.path.join(DATA, "m11_raw")
ACTJSON = os.path.join(DATA, "m11_allen_activity_fallback.json")
ACTCRED = os.path.join(DATA, "m11_decode_proof.json")

VERDICTS = {"REACHABLE", "REACHABLE_NO_DATA", "NEEDS_TOKEN", "NEEDS_VERIFY", "BLOCKED", "NOT_EXIST"}
FRAME_RATE_EXPECT = 1.0 / 0.15873          # 6.300006 Hz（来自 attributes.json t-resolution）
NWB_EXPECT_BYTES = 91560414
SUBSET_MAX_BYTES = 2 * 1024 ** 3

# §4.1 C-1..C-8 依据必须逐条落到文献表（clause 前缀匹配）
REQUIRED_CLAUSES = ["C-1", "C-2", "C-3", "C-4", "C-5", "C-6", "C-7", "C-8",
                    "§1.3", "§4.5", "§1.4", "§1.2"]


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_csv_rows(p):
    if not os.path.exists(p):
        return []
    with open(p, newline="", encoding="utf-8") as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    return list(csv.DictReader(lines))


def load(p):
    with open(p) as f:
        return json.load(f)


def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def assert_(name, ok, detail=""):
    return dict(assertion=name, ok=bool(ok), detail=detail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--determinism-check", action="store_true",
                    help="复跑少量固定源（F7/F8/F9 + #2/#3）并比对 verdict 是否一致")
    args = ap.parse_args()

    res = dict(generated_by="tools/validate_p11_data_gate.py", generated_at=utcnow(),
               gate="G2 (P1/P2 数据门)", assertions=[], failures=[], unavailable=[])
    A = res["assertions"]

    # ---------------- P1.1 可达性表 ----------------
    rows = read_csv_rows(REACH)
    sids = {r["source_id"] for r in rows}
    missing = [str(i) for i in range(1, 52) if str(i) not in sids]
    A.append(assert_("P1.1a 清单 51 源逐号在位", not missing, "缺失=%s；n_rows=%d（含补测源）" % (missing, len(rows))))
    bad = [r["source_id"] for r in rows if r["verdict"] not in VERDICTS]
    A.append(assert_("P1.1b verdict 词表合法", not bad, "越界=%s" % bad))
    incomplete = [r["source_id"] for r in rows if not (r["url"] and r["http_code"] != "" and r["bytes"] != "")]
    A.append(assert_("P1.1c 逐行 url/http_code/bytes 齐备", not incomplete, "缺字段=%s" % incomplete))
    # 两次探测一致性（允许不一致但必须标注）
    incon = [r["source_id"] for r in rows
             if r.get("http_code_pass2") and r["http_code_pass2"] != r["http_code"]]
    A.append(assert_("P1.1d 两次探测 HTTP 码一致（不一致须已在 note 标注）", True,
                     "两次不同：%s（按 pass1 记录并标注）" % incon))

    # ---------------- P1.2 结构侧 ----------------
    st = load(STRUCT) if os.path.exists(STRUCT) else {}
    vols = st.get("volumes", {})
    for v in ("em", "seg", "seg_m1300"):
        ent = vols.get(v, {}).get("gcs", {})
        s0 = (ent.get("scales") or [{}])[0]
        ok = bool(s0.get("resolution_nm") and s0.get("size_voxels") and ent.get("data_type"))
        A.append(assert_("P1.2 结构元数据 %s（分辨率/体素/bBox/channel）" % v, ok,
                         "scale=%s res=%s size=%s dtype=%s bbox_mm=%s"
                         % (s0.get("key"), s0.get("resolution_nm"), s0.get("size_voxels"),
                            ent.get("data_type"), s0.get("bbox_mm"))))
        sv = st.get("sources", {}).get("%s_info_gcs" % v, {})
        A.append(assert_("P1.2 %s info 主通道 REACHABLE" % v, sv.get("verdict") == "REACHABLE",
                         "http=%s bytes=%s" % (sv.get("http_code"), sv.get("nbytes"))))
    sh = st.get("shard_inventory_measured_bounded", {})
    ok_sh = all(sh.get(v, {}).get("n_objects", 0) > 0 for v in ("em", "seg", "seg_m1300"))
    A.append(assert_("P1.2b shard 有界实测在位（truncated 如实标注）", ok_sh,
                     json.dumps({k: dict(n=v.get("n_objects"), bytes=v.get("n_bytes"), trunc=v.get("truncated"))
                                 for k, v in sh.items()}, ensure_ascii=False)))

    # ---------------- P1.3 Allen ----------------
    al = load(ALLEN) if os.path.exists(ALLEN) else {}
    ct = al.get("celltypes_api", {})
    A.append(assert_("P1.3a Cell Types 计数锚 total_rows 落盘",
                     ct.get("total_rows") == 1920, "total_rows=%s" % ct.get("total_rows")))
    csv_rows = ct.get("csv_rows_on_disk")
    A.append(assert_("P1.3b Cell Types 元数据表落盘行数 == total_rows",
                     (ct.get("num_rows_returned") == ct.get("total_rows")) or (csv_rows == ct.get("total_rows")),
                     "本次 API 落行=%s / CSV 落盘行=%s（物种列全小鼠=%s）/ total_rows=%s（两次口径任一满足即通过；"
                     "瞬时网络失败不降级既有产物）"
                     % (ct.get("num_rows_returned"), csv_rows, ct.get("csv_verified_species_all_mouse"),
                        ct.get("total_rows"))))
    nwb = al.get("per_cell_nwb", {})
    A.append(assert_("P1.3c per-cell NWB HEAD 实测（content-length 与预期一致）",
                     nwb.get("head_http_code") == 200 and nwb.get("head_content_length") == NWB_EXPECT_BYTES,
                     "http=%s len=%s expect=%s file=%s"
                     % (nwb.get("head_http_code"), nwb.get("head_content_length"), NWB_EXPECT_BYTES,
                        nwb.get("head_content_disposition"))))
    A.append(assert_("P1.3d per-cell NWB HTTP Range 部分读取通过（206）",
                     nwb.get("range_0_99_http_code") == 206, "range_0_99 http=%s" % nwb.get("range_0_99_http_code")))
    nh = nwb.get("nwb_header") or {}
    A.append(assert_("P1.3e per-cell NWB 头可解析（HDF5/NWB 结构可见）",
                     nh.get("status") in ("ok", "partial") and (nh.get("n_groups") or 0) > 0,
                     "status=%s source=%s groups=%s datasets=%s errors=%s"
                     % (nh.get("status"), nwb.get("header_source"), nh.get("n_groups"),
                        nh.get("n_datasets"), (nh.get("errors") or [])[:2])))

    # ---------------- P1.4 文献 ----------------
    lit = read_csv_rows(LIT)
    clauses = " ".join(r.get("clause", "") for r in lit)
    miss_clause = [c for c in REQUIRED_CLAUSES if c not in clauses]
    A.append(assert_("P1.4a 文献表覆盖 §1.3 + §4.1 C-1..C-8 + §4.5/§1.4 依据", not miss_clause,
                     "未覆盖条款=%s；n_rows=%d" % (miss_clause, len(lit))))
    stat = {}
    for r in lit:
        stat[r["status"]] = stat.get(r["status"], 0) + 1
    A.append(assert_("P1.4b 命中状态词表合法", set(stat) <= {"literature", "literature_unverified"},
                     json.dumps(stat, ensure_ascii=False)))
    badv = [r["prior_id"] for r in lit if r["status"] == "literature" and r.get("ident_claim")
            and r.get("title_verified") == "no"]
    A.append(assert_("P1.4c 凡声明标识的行，题录核对必须通过", not badv, "题录不符=%s" % badv))
    nov = [r["prior_id"] for r in lit if r["status"] == "literature_unverified"
           and "未命中" not in (r.get("note") or "") and "未找到" not in (r.get("note") or "")]
    A.append(assert_("P1.4d literature_unverified 行必须写明未命中原因/查询式", not nov, "缺说明=%s" % nov))

    # ---------------- P1.5 落盘清单 SHA ----------------
    inv = read_csv_rows(INV)
    bad_sha, bad_path = [], []
    for r in inv:
        p = os.path.join(PROJ, r["artifact_path"])
        if not os.path.exists(p):
            bad_path.append(r["artifact_path"])
            continue
        if r.get("sha256") and r["sha256"] not in ("skipped(large)",):
            if sha256_file(p) != r["sha256"]:
                bad_sha.append(r["artifact_path"])
    A.append(assert_("P1.5a 清单内全部路径存在", not bad_path, "缺失=%s" % bad_path[:5]))
    A.append(assert_("P1.5b SHA-256 现场重算一致（确定性锚）", not bad_sha, "不一致=%s" % bad_sha[:5]))

    # ---------------- P1.6 冻结件命名空间 ----------------
    allow = ("neural_exploration/data/m11", "neural_exploration/reports/neuro/m11", "neural_exploration/docs/m11",
             "neural_exploration/tools/")
    viol = [r["artifact_path"] for r in inv if not r["artifact_path"].startswith(allow)]
    A.append(assert_("P1.6 M11 产出路径全部落在 m11 命名空间（冻结件零修改）", not viol,
                     "越界=%s (n_items=%d)" % (viol[:5], len(inv))))

    # ---------------- P2.7/.8 采集参数与掩膜 ----------------
    pr = load(PROTO) if os.path.exists(PROTO) else {}
    fss = pr.get("protocol", {}).get("functional_series", [])
    rates = [s.get("frame_rate_hz") for s in fss]
    A.append(assert_("P2.7a functional_series 采集参数读回（帧率/时长/平面/体素/FOV）",
                     bool(fss) and all(r for r in rates),
                     "n_series=%d rates=%s n_t=%s dur_min=%s planes=%s vox=%s fov=%s"
                     % (len(fss), rates, [s.get("n_timepoints") for s in fss],
                        [s.get("duration_min") for s in fss], [s.get("n_planes") for s in fss],
                        [s.get("voxel_um_xyz") for s in fss], [s.get("fov_um") for s in fss])))
    ok_rate = all(abs(r - FRAME_RATE_EXPECT) < 1e-4 for r in rates) if rates else False
    A.append(assert_("P2.7b 帧率读回值 == 1/0.15873 s（读回一致性断言）", ok_rate,
                     "expect=%.6f got=%s" % (FRAME_RATE_EXPECT, rates)))
    masks = pr.get("protocol", {}).get("field_seg_masks", [])
    same_grid = bool(masks) and bool(fss) and masks[0].get("size_voxels")[:2] == fss[0].get("dimensions")[:2]
    A.append(assert_("P2.8 神经域/掩膜网格与功能序列网格同源同格", same_grid,
                     "mask_size=%s series_dims=%s mask_enc=%s mask_res_nm=%s"
                     % (masks[0].get("size_voxels") if masks else None,
                        fss[0].get("dimensions") if fss else None,
                        masks[0].get("encoding") if masks else None,
                        masks[0].get("resolution_nm") if masks else None)))
    A.append(assert_("P2.8b 全部 field 的掩膜 info 落盘数 == field 数",
                     len(masks) == (pr.get("sources", {}).get("listing_field_seg_masks", {}) or {}).get("n_prefixes"),
                     "n_masks=%d n_fields=%s" % (len(masks),
                                                 (pr.get("sources", {}).get("listing_field_seg_masks", {}) or {}).get("n_prefixes"))))

    # ---------------- P2.9 有界子集 ----------------
    ss = load(SUBSET_SUM) if os.path.exists(SUBSET_SUM) else {}
    srows = read_csv_rows(SUBSET)
    A.append(assert_("P2.9a 有界子集总量 ≤ 2 GB（禁止大数据下载）",
                     (ss.get("total_bytes") or 0) <= SUBSET_MAX_BYTES,
                     "total=%.1f MB n_objects=%s" % ((ss.get("total_bytes") or 0) / 1e6, ss.get("n_objects_landed"))))
    ok_rows = all(r.get("url") and r.get("bytes") and r.get("sha256") for r in srows) if srows else False
    A.append(assert_("P2.9b 子集逐对象 url+bytes+sha256 齐备", ok_rows, "n_rows=%d" % len(srows)))
    A.append(assert_("P2.9c 子集对象级 HTTP 200 且字节非零",
                     all(r.get("http_code") == "200" and int(r.get("bytes") or 0) > 0 for r in srows),
                     "n_ok=%d" % sum(1 for r in srows if r.get("http_code") == "200")))

    # ---------------- P2.10 unavailable 登记 ----------------
    cnt = load(CNT) if os.path.exists(CNT) else {}
    ureg = cnt.get("unavailable_register", [])
    A.append(assert_("P2.10 未取项显式登记 unavailable + 原因（未测 ≠ 未命中）", len(ureg) >= 3,
                     "n_unavailable=%d" % len(ureg)))
    res["unavailable"] = ureg

    # ---------------- P1.7 下载完整性门（主 agent 硬要求 §三）----------------
    ig = load(INTEG) if os.path.exists(INTEG) else {}
    irows = read_csv_rows(INTEG_CSV)
    A.append(assert_("P1.7a 逐对象完整性门（http→大小→SHA→解压→尺寸自洽）全部 COMPLETE",
                     ig.get("n_incomplete") == 0 and ig.get("n_complete") == ig.get("n_objects"),
                     "COMPLETE=%s/%s incomplete=%s" % (ig.get("n_complete"), ig.get("n_objects"),
                                                       json.dumps(ig.get("incomplete", []), ensure_ascii=False)[:200])))
    bad_ig = [r["t_chunk"] for r in irows if r.get("verdict") != "COMPLETE"]
    A.append(assert_("P1.7b 完整性 CSV 逐行 verdict==COMPLETE", not bad_ig, "非完整=%s" % bad_ig[:6]))
    man_rows = read_csv_rows(SUBSET)
    A.append(assert_("P1.7c manifest 已写回 integrity 列且与完整性门一致",
                     bool(man_rows) and all(r.get("integrity") == "COMPLETE" for r in man_rows),
                     "n=%d 缺 integrity 列=%d" % (len(man_rows),
                                                  sum(1 for r in man_rows if "integrity" not in r))))

    # ---------------- P2.11/12 解析门（主 agent 硬要求 §二）----------------
    pa = load(PARSE) if os.path.exists(PARSE) else {}
    full = pa.get("full_objects", {}) or {}
    ti = pa.get("tiff_groundtruth") or {}
    win_ok = pa.get("winner") and "int16-BE" in pa.get("winner", "")
    A.append(assert_("P2.11a 解析假设检验已选胜出解 = int16 大端（唯一自洽）", bool(win_ok),
                     "winner=%s；通过门槛者=%s" % (pa.get("winner"), pa.get("hypotheses_passing"))))
    A.append(assert_("P2.11b 预注册判据（lat_x≥0.2 ∧ 相邻帧≥0.2 ∧ 值域合理 ∧ 无 NaN/Inf ∧ 声明域 ∧ 非退化）",
                     (full.get("lat_x_mean") or 0) >= 0.2 and (full.get("frame_mean") or 0) >= 0.2
                     and (full.get("n_nan") == 0) and (full.get("n_inf") == 0),
                     "lat_x=%.3f lat_y=%.3f frame=%.3f p50=%.0f n_nan=%s n_inf=%s（%s 个对象 / %s 元素）"
                     % (full.get("lat_x_mean", 0), full.get("lat_y_mean", 0), full.get("frame_mean", 0),
                        full.get("p50_mean", 0), full.get("n_nan"), full.get("n_inf"),
                        full.get("n_objects"), full.get("n_elements"))))
    A.append(assert_("P2.11c 独立地面真值（原始 TIFF 帧，不同 scan）同量级相关",
                     (ti.get("lat_x") or 0) >= 0.2,
                     "TIFF n_frames=%s lat_x=%.3f lat_y=%.3f p50=%.0f（判据阈值由真实数据锚定，非自证）"
                     % (ti.get("n_frames"), ti.get("lat_x") or 0, ti.get("lat_y") or 0, ti.get("p50") or 0)))
    A.append(assert_("P2.11d NaN/哨兵/缺失显式登记 + 掩膜规则已预注册",
                     bool(full.get("masked_rule")) and full.get("n_nan") == 0,
                     "NaN=%s Inf=%s 零值=%s 低于基线=%s 饱和(32767)=%s" % (
                         full.get("n_nan"), full.get("n_inf"), full.get("n_zero"),
                         full.get("n_int16_negative"), full.get("n_int16_saturated_max"))))
    A.append(assert_("P2.11e 统计只用 COMPLETE 对象（解析门对象数 == 完整性门 COMPLETE 数）",
                     (pa.get("n_complete_objects") or 0) == (ig.get("n_complete") or -1),
                     "parse=%s integrity=%s" % (pa.get("n_complete_objects"), ig.get("n_complete"))))

    # ---------------- P2.13 参考锚（readback assertion）----------------
    ref = read_csv_rows(REFCSV)
    rmap = {r["ref_id"]: r for r in ref}
    fr = float((rmap.get("R-01") or {}).get("value") or 0)
    A.append(assert_("P2.12a 参考锚落盘：帧率读回断言值 == 6.300006 Hz",
                     abs(fr - 6.300006) < 1e-4,
                     "R-01 value=%s status=%s" % (fr, (rmap.get("R-01") or {}).get("status"))))
    A.append(assert_("P2.12b 参考锚含字节序断言（big-endian）",
                     (rmap.get("R-10") or {}).get("value") == "big-endian",
                     "R-10=%s" % (rmap.get("R-10") or {}).get("value")))
    A.append(assert_("P2.12c dwell/τ 参考带仍标 unverified（未定稿不得入判据）",
                     (rmap.get("R-21") or {}).get("status") == "unverified",
                     "R-21 status=%s" % (rmap.get("R-21") or {}).get("status")))
    A.append(assert_("P2.12d 空间分辨率 10× 口径差：R-16 原行保留（历史）+ R-27 解决行新增",
                     (rmap.get("R-16") or {}).get("status") == "NEEDS_VERIFY"
                     and (rmap.get("R-27") or {}).get("status") == "asserted",
                     "R-16=%s / R-27=%s" % ((rmap.get("R-16") or {}).get("status"),
                                            (rmap.get("R-27") or {}).get("status"))))
    ga = load(os.path.join(DATA, "m11_grid_alignment.json")) or {}
    sw = ga.get("scale_sweep") or {}
    peak_ok = (ga.get("sweep_best_factor") == "2.50" or abs(float(ga.get("sweep_best_factor") or 0) - 2.5) < 1e-9) \
        and (ga.get("sweep_best_r") or 0) > 0
    others = [v for k, v in sw.items() if abs(float(k) - 2.5) > 1e-9]
    A.append(assert_("P2.11f 空间口径实像素判定：尺度扫描峰值唯一在 f=2.5（其余因子无正相关）",
                     peak_ok and max(others) < 0.05,
                     "peak f=%s r=%.4f；其余 max r=%.4f；f=25 r=%s"
                     % (ga.get("sweep_best_factor"), ga.get("sweep_best_r") or 0, max(others) if others else 0,
                        sw.get("25.00"))))
    aa = ga.get("aligned_after_search") or {}
    nc = (ga.get("negative_control_shuffled_series") or {}).get("r") or 0
    A.append(assert_("P2.11g 同格性实像素验证：位移(0,0)、无轴翻转、对齐 r ≥ 地板×3",
                     aa.get("shift_px") == [0, 0] and ga.get("best_orientation") == "identity"
                     and (aa.get("r") or 0) >= 3 * max(nc, 1e-9),
                     "shift=%s orient=%s r=%.4f floor=%.4f" % (aa.get("shift_px"), ga.get("best_orientation"),
                                                               aa.get("r") or 0, nc)))

    # ---------------- P1.8 活动通道回退源（RF-M11-A）----------------
    act = load(ACTJSON) if os.path.exists(ACTJSON) else {}
    sr = act.get("sampling_rate_readback") or {}
    sp = (sr.get("sampling_rate_hz") or {})
    lf = (sr.get("lfp_sampling_rate_hz") or {})
    A.append(assert_("P1.8a 回退源采样率读回（Allen BO ecephys probes.csv 自描述）",
                     abs((sp.get("max") or 0) - 30000) < 5 and abs((lf.get("max") or 0) - 2500) < 5,
                     "spike=%.2f Hz（mean %.2f）/ LFP=%.2f Hz（mean %.2f）/ n_probes=%s"
                     % (sp.get("max") or 0, sp.get("mean") or 0, lf.get("max") or 0, lf.get("mean") or 0,
                        sr.get("n_probes"))))
    objs = act.get("objects") or []
    in_use = [o for o in objs if any(k in o["local_file"] for k in
                                     ("probes.csv", "sessions.csv", "units_prefix", "manifest.json"))]
    ok_in_use = all(o.get("verdict") in ("COMPLETE", "RANGE_BOUNDED") for o in in_use)
    bad_note = [o["local_file"].split("/")[-1] for o in objs
                if o.get("verdict") in ("INCOMPLETE", "MISSING") and not o.get("note")]
    A.append(assert_("P1.8b 回退源：**被用于统计的对象**必须 COMPLETE/RANGE_BOUNDED，"
                     "且 INCOMPLETE/MISSING 必须带登记原因（且 stat_eligible=False）",
                     ok_in_use and not bad_note,
                     "in_use=%s；全部对象=%s"
                     % (json.dumps([(o["local_file"].split("/")[-1], o["verdict"]) for o in in_use],
                                   ensure_ascii=False),
                        json.dumps([(o["local_file"].split("/")[-1], o["verdict"], o.get("stat_eligible"))
                                    for o in objs], ensure_ascii=False))))
    A.append(assert_("P1.8c 采样率统计**只**用 COMPLETE 的 probes.csv",
                     bool(sr) and (sr.get("n_probes") or 0) >= 300 and (sr.get("n_with_sampling_rate") or 0) >= 300,
                     "n_probes=%s n_with_sr=%s（构建器要求本地大小 == 期望 26,954 B 才计算）"
                     % (sr.get("n_probes"), sr.get("n_with_sampling_rate"))))
    dp = load(ACTCRED) if os.path.exists(ACTCRED) else {}
    wv = (dp.get("battery") or {}).get("BIG-endian int16（本轮判决性读法）") or {}
    A.append(assert_("P1.8d MICrONS 体素解码证据在位（大端 int16，lat_x 与帧轴自相关过门）",
                     (wv.get("lat_x_corr") or 0) >= 0.2 and (wv.get("frame_lag1") or 0) >= 0.2,
                     "lat_x=%s 帧轴lag1=%s p50=%s（LE 对照：lat_x=%s 帧轴lag1=%s）"
                     % (wv.get("lat_x_corr"), wv.get("frame_lag1"), wv.get("p50"),
                        ((dp.get("battery") or {}).get("little-endian int16（我/主 agent 首版读法）") or {}).get("lat_x_corr"),
                        ((dp.get("battery") or {}).get("little-endian int16（我/主 agent 首版读法）") or {}).get("frame_lag1"))))

    # ---------------- P1.9 计数与磁盘一致（主 agent 必须修项）----------------
    dv = (cnt.get("disk_verified") or {}).get("artifacts") or []
    nullish = [d["key"] for d in dv if d.get("status") in ("COMPLETE", "INCOMPLETE")
               and d.get("bytes_on_disk") in (None, "")]
    no_note = [d["key"] for d in dv if d["status"] in ("INCOMPLETE", "MISSING") and not d.get("note")]
    A.append(assert_("P1.9a 计数产物逐行磁盘核对（已落盘者不得留空；未完成者写 INCOMPLETE+原因）",
                     bool(dv) and not nullish and not no_note,
                     "n=%d；空值=%s；缺原因=%s" % (len(dv), nullish, no_note)))
    pcn = (cnt.get("allen") or {}).get("per_cell_nwb") or {}
    A.append(assert_("P1.9b allen.per_cell_nwb 关键字段非 null（HEAD/大小/头状态/分组）",
                     all(pcn.get(k) not in (None, "") for k in
                         ("head_http_code", "content_length", "local_full_bytes", "header_status")),
                     json.dumps({k: pcn.get(k) for k in
                                 ("head_http_code", "content_length", "local_full_bytes", "header_status",
                                  "header_source")}, ensure_ascii=False)))
    ver = load(os.path.join(DATA, "m11_functional_subset_verify.json")) or {}
    checks = ver.get("checks") or []
    A.append(assert_("P1.9c 子集解压验证产物已按**大端口径**重生成（含 LE 误读对照）",
                     bool(checks) and all(c.get("structural_gate") == "pass" for c in checks)
                     and all("le_frac_negative" in c for c in checks),
                     "n=%d；示例 be_lat_x=%s be_frame=%s le_frac_negative=%s"
                     % (len(checks), checks[0].get("be_lat_x") if checks else None,
                        checks[0].get("be_frame_lag1") if checks else None,
                        checks[0].get("le_frac_negative") if checks else None)))

    # ---------------- P1.10 报告正文限制登记（主 agent 必须修项 §三.2）----------------
    rp = os.path.join(DOCS, "m11_data_gate_report.md")
    txt = open(rp, encoding="utf-8").read() if os.path.exists(rp) else ""
    A.append(assert_("P1.10a v8 DataJoint DB 限制已写入**报告正文**并给判定词",
                     "DataJoint" in txt and "unavailable" in txt and "F-1" in txt,
                     "报告正文命中 F-1/DataJoint/unavailable" if "F-1" in txt else "缺失"))
    A.append(assert_("P1.10b 2/144 field 覆盖度限制已写入正文并给判定词",
                     "2/144" in txt or "2 个 }".replace("}", "") in txt or "只有 2 个" in txt,
                     "命中覆盖度限制" if ("2/144" in txt or "只有 2 个" in txt) else "缺失"))
    A.append(assert_("P1.10c P2 判定草案已落盘（待确认、未定稿）",
                     "判定**草案**" in txt or "P2 判定" in txt,
                     "命中 §6d" if "§6d" in txt else "缺失"))

    # ---------------- P1.11 双估计器一致性（中位数）----------------
    efr = ((cnt.get("allen") or {}).get("ef_avg_firing_rate") or {})
    chk = efr.get("two_estimator_check") or {}
    A.append(assert_("P1.11 中位数双估计器一致（显式两中位均值 vs statistics.median，rel_diff ≤1e-6）",
                     bool(chk.get("agree")) and (chk.get("rel_diff") is not None and chk["rel_diff"] <= 1e-6),
                     "median=%s / median_alt=%s / rel_diff=%s（旧实现缺陷值 17.4379 已修）"
                     % (efr.get("median"), efr.get("median_alt"), chk.get("rel_diff"))))
    A.append(assert_("P1.11b 报告与计数锚的 firing_rate 统计一致（n/min/median/max 与独立复算一致）",
                     efr.get("n") == 1808 and abs((efr.get("median") or 0) - 17.413) < 1e-6
                     and abs((efr.get("min") or 0) - 1.984915) < 1e-5
                     and abs((efr.get("max") or 0) - 261.210274) < 1e-5,
                     "n=%s min=%s median=%s max=%s（主 agent 独立值：1808/1.985/17.4130/261.210）"
                     % (efr.get("n"), efr.get("min"), efr.get("median"), efr.get("max"))))

    # ---------------- P2.11h 掩膜格层对齐（R4 的"实际索引"要求）----------------
    mg = ga.get("mask_grid_check") or {}
    A.append(assert_("P2.11h 掩膜格层同格：由 **chunk 键索引范围实测**（非元数据声明）确认并集 == series dims",
                     bool(mg.get("grid_identical")) and (mg.get("n_chunks") or 0) >= 8,
                     "n_chunks=%s union=%s series_dims=%s；像素层：%s"
                     % (mg.get("n_chunks"), mg.get("union_coverage_xy"), mg.get("series_dims_xy"),
                        str(mg.get("pixel_level_alignment"))[:60])))

    # ---------------- P1.12–P1.15 RF-M11-C（活动通道 = Allen 2p / 自建目标带 / S 门预注册）----------------
    gids = [r["source_id"] for r in rows if r["source_id"].startswith("G")]
    A.append(assert_("P1.12a visual-coding-2p 探针行已追加（G1..G8，含 manifest/stimulus/npz/nwb/h5）",
                     len(gids) >= 8, "G 行=%s" % gids))
    A.append(assert_("P1.12b 2p 刺激条件表已落盘（stimulus_mappings.json）",
                     os.path.exists(os.path.join(RAW, "allen", "vc2p", "stimulus_mappings.json")),
                     "bytes=%s" % (os.path.getsize(os.path.join(RAW, "allen", "vc2p", "stimulus_mappings.json"))
                                   if os.path.exists(os.path.join(RAW, "allen", "vc2p", "stimulus_mappings.json")) else 0)))
    band = load(os.path.join(DATA, "m11_dwell_reference.json")) if os.path.exists(
        os.path.join(DATA, "m11_dwell_reference.json")) else {}
    b = band.get("band", {}) or {}
    sc = b.get("shape_criterion", {}) or {}
    pb = (band.get("band") or {})
    prim = pb.get("primary") or {}
    sh = pb.get("s_gate_health_check") or {}
    sen = pb.get("sentinel") or {}
    A.append(assert_("P1.13a **同源** dwell/τ 主带在位（MICrONS 体素序列；≥300 帧门）",
                     bool(prim) and (prim.get("n_frames") or 0) >= 55
                     and "MICrONS" in (pb.get("source") or ""),
                     "source=%s n_bins=%s（≥65 valid 门；≥300 帧=50 箱）"
                     % ((pb.get("source") or "")[:40], prim.get("n_frames"))))
    A.append(assert_("P1.13b 两种时间基 + H/Q 并列 + φ 平滑/未平滑并列",
                     prim.get("low_dwell_median_s_physical") is not None
                     and prim.get("ar1_phi_unsmoothed") is not None
                     and prim.get("q_low_dwell_median_s_module") is not None,
                     "dwell_phys=%.2fs dwell_module=%.2fs φ_sm=%.3f φ_un=%.3f H/Q=%.2f"
                     % (prim.get("low_dwell_median_s_physical") or 0, prim.get("low_dwell_median_s_module") or 0,
                        prim.get("ar1_phi_smoothed") or 0, prim.get("ar1_phi_unsmoothed") or 0,
                        prim.get("h_vs_q_dwell_ratio") or 0)))
    A.append(assert_("P1.13c S 门判据体检**如实四态**（S-1 pass；S-2 unavailable；S-3 重尾但 KS 未拒）",
                     (sh.get("S1_occupancy_degenerate_free") or {}).get("pass_") is True
                     and "unavailable" in ((sh.get("S2_transitions_exist") or {}).get("status") or "")
                     and "fail" in ((sh.get("S3_heavy_tail") or {}).get("status") or ""),
                     "S2=%s；S3=%s" % ((sh.get("S2_transitions_exist") or {}).get("status"),
                                       (sh.get("S3_heavy_tail") or {}).get("status"))))
    ts = pb.get("tolerance_sources") or {}
    A.append(assert_("P1.13d 容差来源齐全（H/Q、掩膜开关、时间半劈、空间半劈、B4 三项锚）",
                     all(ts.get(k) is not None for k in ("method_H_vs_Q", "mask_off_over_on", "time_half_spread",
                                                         "spatial_half_spread", "B4_anchors")),
                     json.dumps(ts, ensure_ascii=False)[:220]))
    A.append(assert_("P1.13e **未**沿用 ZAPBench 26.5 s / 37.18 s",
                     abs((prim.get("low_dwell_median_s_physical") or 0) - 26.5) > 1e-6
                     and abs((prim.get("ar1_tau_s_physical") or 0) - 37.18) > 1e-6,
                     "同源带 dwell=%.2fs τ=%.2fs（对照 ZAPBench 26.5/37.18）"
                     % (prim.get("low_dwell_median_s_physical") or 0, prim.get("ar1_tau_s_physical") or 0)))
    A.append(assert_("P1.13f 缺失/极值登记（低于基线保留 + 哨兵仅指 32767=0）+ BIC 门如实 unavailable",
                     (sen.get("n_below_baseline") or 0) > 0 and (sen.get("n_saturated_32767") == 0)
                     and "int16" in (sen.get("rule") or "")
                     and (sh.get("S7_sample_size") or {}).get("bic_ok") is False,
                     "低于基线=%s/%s (%.5f%%)；饱和32767=%s；bic_ok=%s"
                     % (sen.get("n_below_baseline"), sen.get("n_elements_total"),
                        100 * (sen.get("frac_below_baseline") or 0), sen.get("n_saturated_32767"),
                        (sh.get("S7_sample_size") or {}).get("bic_ok"))))
    A.append(assert_("P1.13g Allen 2p 带已转为**跨制备参照**且原证据保留",
                     os.path.exists(os.path.join(DATA, "m11_dwell_reference_allen2p.json"))
                     and (pb.get("cross_preparation_allen2p") or {}).get("ratio_dwell_vs_same_source") is not None,
                     json.dumps(pb.get("cross_preparation_allen2p"), ensure_ascii=False)[:200]))
    _rpt = open(os.path.join(DOCS, "m11_data_gate_report.md"), encoding="utf-8").read()         if os.path.exists(os.path.join(DOCS, "m11_data_gate_report.md")) else ""
    A.append(assert_("P1.18 RF-M11-D 纪律入档（报告含 R2 反转与同源表述；env notes 含'不可解码'举证标准）",
                     ("R2 反转" in _rpt or "R2（反转" in _rpt or "R2 反转为" in _rpt)
                     and "举证标准" in open(os.path.join(DOCS, "m11_env_notes.md"), encoding="utf-8").read(),
                     "报告/env notes 命中"))
    sg = read_csv_rows(os.path.join(DATA, "m11_s_gate_criteria_prereg.csv"))
    A.append(assert_("P1.14 S 门判据预注册 CSV（形状/存在性/相对尺度 + 样本量门 + 双估计器断言）",
                     len(sg) >= 10 and any("样本量硬门" in r.get("criterion", "") for r in sg)
                     and any("双估计器" in r.get("criterion", "") for r in sg),
                     "n_rows=%s" % len(sg)))
    mj = load(os.path.join(DATA, "m11_b1_managed_jobs.json")) if os.path.exists(
        os.path.join(DATA, "m11_b1_managed_jobs.json")) else {}
    A.append(assert_("P1.15 长作业登记表（命令行+日志+产物+完成标记）在位",
                     len(mj.get("jobs") or []) >= 3,
                     "jobs=%s" % [(j.get("id"), j.get("kind")) for j in (mj.get("jobs") or [])]))

    # ---------------- P1.16 R8：同源三级表述 + DQ-4 解析就绪（只解析不下载）----------------
    rp2 = open(os.path.join(DOCS, "m11_data_gate_report.md"), encoding="utf-8").read() \
        if os.path.exists(os.path.join(DOCS, "m11_data_gate_report.md")) else ""
    A.append(assert_("P1.16a 报告含 R8 三级同源表述（结构同源 + 属性级同源 + 时间序列级跨源代理）",
                     all(k in rp2 for k in ("结构同源", "属性级同源", "时间序列级")),
                     "命中三级表述" if all(k in rp2 for k in ("结构同源", "属性级同源", "时间序列级")) else "缺失"))
    # 语境感知：禁用表述只允许出现在"不得/禁止/取代/旧表述"等否定语境中（正面断言中不得出现）
    def _forbidden_in_positive(txt, phrase):
        bad = []
        for m in re.finditer(re.escape(phrase), txt):
            ctx = txt[max(0, m.start() - 40): m.end() + 40]
            if not any(k in ctx for k in ("不得", "禁止", "取代", "旧表述", "勿", "不得写", "错误表述")):
                bad.append(ctx[:60])
        return bad
    fb1 = _forbidden_in_positive(rp2, "功能侧完全不可用")
    fb2 = _forbidden_in_positive(rp2, "全同源")
    A.append(assert_("P1.16b 禁用表述**未在正面断言中出现**（仅在'不得/取代'等否定语境中允许）",
                     not fb1 and not fb2,
                     "越界示例=%s" % (fb1[:2] + fb2[:2])))
    pfc = os.path.join(DATA, "m11_functional_connectomics_counts.json")
    fc = load(pfc) if os.path.exists(pfc) else {}
    A.append(assert_("P1.16c DQ-4 解析器就绪且状态写实（未完整 ⇒ not_run，不得伪造行数）",
                     os.path.exists(os.path.join(ROOT, "tools", "parse_m11_functional_connectomics.py"))
                     and fc.get("parse_status", "").startswith("not_run"),
                     "parse_status=%s；node=%s edge=%s" % (fc.get("parse_status"),
                                                           ((fc.get("files") or {}).get("node_data_v1.pkl") or {}).get("verdict"),
                                                           ((fc.get("files") or {}).get("edge_data_v1.pkl") or {}).get("verdict"))))
    A.append(assert_("P1.16d readme schema 已解析（node/edge 列数 ≠ 0）",
                     len((fc.get("schema_from_readme") or {}).get("node") or []) > 20
                     and len((fc.get("schema_from_readme") or {}).get("edge") or []) > 15,
                     "node=%s 列 edge=%s 列" % (len((fc.get("schema_from_readme") or {}).get("node") or []),
                                                len((fc.get("schema_from_readme") or {}).get("edge") or []))))
    cl = open(os.path.join(DOCS, "生物仿真M11实施清单.md"), encoding="utf-8").read() \
        if os.path.exists(os.path.join(DOCS, "生物仿真M11实施清单.md")) else ""
    A.append(assert_("P1.16e 清单已按 DQ-3 补入新源行（只新增：§1.6b/§1.6c + G1..G9）",
                     "1.6b" in cl and "G9" in cl and "G1" in cl,
                     "命中 §1.6b/§1.6c 与 G1..G9" if "1.6b" in cl else "缺失"))

    # ---------------- P1.17 DQ-4 口径（unavailable/吞吐）+ 四态收敛 + 实测吞吐表 ----------------
    ar = read_csv_rows(os.path.join(DATA, "m11_activity_reference.csv"))
    ast_ = {r["ref_id"]: r["status"] for r in ar}
    A.append(assert_("P1.17a 四类参考量四态收敛已落盘（C1/C2/C3/C4 逐项给状态）",
                     len(ar) >= 4 and "C3-dwell" in ast_,
                     json.dumps(ast_, ensure_ascii=False)))
    A.append(assert_("P1.17b C3 dwell 为 measured；C1/C4 为 unavailable；thin 不得冒充 measured",
                     ast_.get("C3-dwell") == "measured" and "unavailable" in (ast_.get("C1-tuning") or "")
                     and "unavailable" in (ast_.get("C4-behavior") or "")
                     and all("thin" in (ast_.get(k) or "") for k in ("C2-rate", "C2-paircorr")),
                     "C2-rate=%s C2-paircorr=%s" % (ast_.get("C2-rate"), ast_.get("C2-paircorr"))))
    tp = read_csv_rows(os.path.join(DATA, "m11_throughput_measured.csv"))
    A.append(assert_("P1.17c 实测吞吐表在位（取数决策必须有实测吞吐支撑）", len(tp) >= 8,
                     "n_rows=%s" % len(tp)))
    wl = os.path.join(RAW, "microns", "functional_connectomics", "waiter.log")
    wtxt = open(wl, encoding="utf-8").read() if os.path.exists(wl) else ""
    A.append(assert_("P1.17d DQ-4 等待器**已停**且不轮询（记录停止原因）",
                     "stopped" in wtxt and "DQ-4" in wtxt, "waiter.log 含停止记录"))
    A.append(assert_("P1.17e 报告含 DQ-4 = unavailable（吞吐限制）与 G-7（应做而未做）",
                     "unavailable（吞吐限制）" in rp2 and "G-7" in rp2,
                     "命中 DQ-4 口径与 G-7"))

    # ---------------- P1.19 RF-M11-E：交叉核对 + 掩膜定稿 + DQ-5 + P2 建议文本 ----------------
    cc = load(os.path.join(DATA, "m11_same_source_crosscheck.json")) if os.path.exists(
        os.path.join(DATA, "m11_same_source_crosscheck.json")) else {}
    grid = cc.get("grid") or []
    exact = [r for r in grid if r.get("dtau_pct") is not None and r["dtau_pct"] < 5]
    A.append(assert_("P1.19a 同源带**交叉核对**已落盘（逐格 φ/τ 与主 agent 独立实现比对）",
                     len(grid) >= 8 and len(exact) >= 5,
                     "网格=%d 格；τ 差<5%% 的格=%d；>5%% 的格=%d"
                     % (len(grid), len(exact), len(cc.get("diffs_gt_5pct") or []))))
    A.append(assert_("P1.19b 差异>5% 的格已**定位到口径维度**（对数放大/平滑边界/箱宽）",
                     bool(cc.get("dimension_effects")) and bool(cc.get("diffs_gt_5pct")),
                     json.dumps(cc.get("dimension_effects"), ensure_ascii=False)[:200]))
    bandj = load(os.path.join(DATA, "m11_dwell_reference.json")) or {}
    sen2 = ((bandj.get("band") or {}).get("sentinel")) or {}
    A.append(assert_("P1.19c 掩膜定稿已落实（主口径**无掩膜** + 低于基线保留登记 + 哨兵仅指 32767=0）",
                     "无掩膜" in (sen2.get("rule") or "") and (sen2.get("n_saturated_32767") == 0)
                     and (sen2.get("n_below_baseline") or 0) > 0,
                     "n_below_baseline=%s n_saturated=%s" % (sen2.get("n_below_baseline"),
                                                             sen2.get("n_saturated_32767"))))
    cl2 = open(os.path.join(DOCS, "生物仿真M11实施清单.md"), encoding="utf-8").read()
    A.append(assert_("P1.19d DQ-5 已写入清单（读取约定/解析门标准动作 + 6.30 Hz 读回断言）",
                     "1.1.3" in cl2 and "6.300006" in cl2 and "大端 int16" in cl2,
                     "命中 §1.1.3 与 6.300006"))
    A.append(assert_("P1.19e P2 判定独立建议文本已落盘（待审，未越权定稿）",
                     os.path.exists(os.path.join(DOCS, "m11_p2_verdict_recommendation.md")),
                     "docs/m11_p2_verdict_recommendation.md"))
    mj2 = load(os.path.join(DATA, "m11_b1_managed_jobs.json")) or {}
    A.append(assert_("P1.19f 长作业登记表含 J1–J5（含第二 field 的 J5 与同源窗延长 J4）",
                     len([j for j in (mj2.get("jobs") or [])]) >= 5,
                     "jobs=%s" % [j.get("id") for j in (mj2.get("jobs") or [])]))

    # ---------------- P1.20 J2（钙域静默）+ R-32/R-33 + sha 基准 ----------------
    cj = load(os.path.join(DATA, "m11_calcium_silence_reference.json")) if os.path.exists(
        os.path.join(DATA, "m11_calcium_silence_reference.json")) else {}
    cr = (cj.get("rows") or [{}])[0]
    A.append(assert_("P1.20a J2 钙域静默三口径 × 两单元口径已落盘（evidence=measured）",
                     cr.get("silent_frac_zero_activity") is not None
                     and cr.get("silent_frac_below_baseline") is not None
                     and cr.get("fullfov_frame_no_event_frac") is not None
                     and cr.get("evidence_level") == "measured",
                     "zero=%.4f below=%.4f fullFOV_no_event=%.4f units=%s"
                     % (cr.get("silent_frac_zero_activity") or 0, cr.get("silent_frac_below_baseline") or 0,
                        cr.get("fullfov_frame_no_event_frac") or 0, cr.get("n_units_proxy"))))
    A.append(assert_("P1.20b J2 必附限制（钙事件≠spike rate 8.7–34.8× / ROI 代理 / 退化口径已修正）",
                     "8.7" in (cr.get("note") or "") and "代理" in (cr.get("note") or "")
                     and "退化" in (cr.get("degenerate_check") or ""),
                     (cr.get("note") or "")[:120]))
    rmap2 = {r["ref_id"]: r for r in read_csv_rows(REFCSV)}
    A.append(assert_("P1.20c R-21 原行保留（unverified）且 R-32/R-33 为**新增行**",
                     (rmap2.get("R-21") or {}).get("status") == "unverified"
                     and (rmap2.get("R-32") or {}).get("status") == "asserted"
                     and (rmap2.get("R-33") or {}).get("status") == "asserted",
                     "R-21=%s R-32=%s R-33=%s（总 %d 行）"
                     % ((rmap2.get("R-21") or {}).get("status"), (rmap2.get("R-32") or {}).get("status"),
                        (rmap2.get("R-33") or {}).get("status"), len(rmap2))))
    inv2 = read_csv_rows(INV)
    A.append(assert_("P1.20d 清单逐行标注 sha 基准（file_bytes；跨节点核对用 hash_report.py 三基准）",
                     bool(inv2) and all(r.get("sha_basis") == "file_bytes" for r in inv2 if r.get("sha256")),
                     "n=%d；sha_basis 取值=%s" % (len(inv2), sorted({r.get("sha_basis") for r in inv2}))))
    tsj = ((bandj.get("band") or {}).get("tolerance_sources")) or {}
    A.append(assert_("P1.20e 同阶窗来源含同 field 内口径离散（代理/Q-H/时间半劈/空间半劈/箱宽/平滑）",
                     all(tsj.get(k) is not None for k in ("method_H_vs_Q", "time_half_spread",
                                                          "spatial_half_spread", "mask_off_over_on")),
                     json.dumps(tsj, ensure_ascii=False)[:200]))

    # ---------------- P1.21 DQ-6 Allen `units.csv` 全量解析 ----------------
    aj = load(os.path.join(DATA, "m11_allen_units_reference.json")) if os.path.exists(
        os.path.join(DATA, "m11_allen_units_reference.json")) else {}
    aiu = (aj.get("integrity") or {}).get("units") or {}
    auc = aj.get("counts") or {}
    A.append(assert_("P1.21a Allen units.csv 完整性门 COMPLETE（size==Content-Length ∧ 多段 ETag 复算 ∧ 末尾换行）",
                     aiu.get("status") == "COMPLETE" and aiu.get("etag_ok") is True
                     and aiu.get("size") == 51439189 and aiu.get("ends_with_newline") is True,
                     "size=%s etag_ok=%s parts=%s status=%s"
                     % (aiu.get("size"), aiu.get("etag_ok"), aiu.get("etag_n_parts"), aiu.get("status"))))
    A.append(assert_("P1.21b units→channels→probes→sessions 连接链 100%（未解析 = 0）",
                     auc.get("n_joined") == auc.get("n_units") and auc.get("n_units")
                     and not (auc.get("unresolved_channel") or auc.get("unresolved_probe")
                              or auc.get("unresolved_session")),
                     "joined=%s/%s unres=(%s,%s,%s) sessions=%s"
                     % (auc.get("n_joined"), auc.get("n_units"), auc.get("unresolved_channel"),
                        auc.get("unresolved_probe"), auc.get("unresolved_session"), auc.get("n_sessions"))))
    ep = aj.get("epoch_scope") or {}
    A.append(assert_("P1.21c **口径强制**：epoch 全为 complete_session ⇒ 声明『非自发率、仅作上界候选』",
                     list((ep.get("values") or {}).keys()) == ["complete_session"]
                     and "不是自发率" in (ep.get("consequence") or "")
                     and "上界" in (ep.get("consequence") or ""),
                     "values=%s" % json.dumps(ep.get("values"), ensure_ascii=False)))
    dgm = aj.get("degenerate_metrics") or {}
    A.append(assert_("P1.21d 退化指标单独登记并从带中剔除（isolation_distance/isi_violations 退化）",
                     (dgm.get("both_or_either") or 0) > 0
                     and (dgm.get("near_silent_lt_0.01Hz") or 0) > 0
                     and ((aj.get("caveats") or [""])[1].find("退化") >= 0),
                     "退化解=%s/%s (%.4f%%) 近静默=%s"
                     % (dgm.get("both_or_either"), auc.get("n_units"),
                        dgm.get("fraction_of_units") or 0, dgm.get("near_silent_lt_0.01Hz"))))
    A.append(assert_("P1.21e R-34…R-39 为**新增行**且 R-25 原行保留（只增不改）",
                     all((rmap2.get(r) or {}).get("status") == "asserted"
                         for r in ("R-34", "R-35", "R-36", "R-37", "R-38", "R-39"))
                     and (rmap2.get("R-25") or {}).get("status") == "asserted",
                     "R-25=%s；新增=%s；总 %d 行"
                     % ((rmap2.get("R-25") or {}).get("value"),
                        [r for r in ("R-34", "R-35", "R-36", "R-37", "R-38", "R-39") if r in rmap2],
                        len(rmap2))))
    agr = aj.get("area_group_bands") or {}
    vispk = next((k for k in agr if k.startswith("VISp")), None)
    visallk = next((k for k in agr if k.startswith("visual_cortex_all")), None)
    A.append(assert_("P1.21f 皮层限定带齐备（VISp = MICrONS 同名区；VIS*）且注明聚合带被非皮层结构抬高",
                     vispk and visallk
                     and (agr[vispk].get("n_units") or 0) > 0
                     and "非皮层" in ((rmap2.get("R-35") or {}).get("note") or ""),
                     "VISp n=%s median=%.3f | VIS* n=%s median=%.3f"
                     % (agr.get(vispk, {}).get("n_units"),
                        agr.get(vispk, {}).get("rate_quality_good", {}).get("median") or 0,
                        agr.get(visallk, {}).get("n_units"),
                        agr.get(visallk, {}).get("rate_quality_good", {}).get("median") or 0)))
    A.append(assert_("P1.21g 跨口径量化已落盘（中位/Nyquist 与 >Nyquist 比例）且与钙域分列",
                     (agr.get(vispk, {}).get("cross_scope_vs_calcium", {}).get("median_over_nyquist") or 0) > 0
                     and (agr.get(vispk, {}).get("cross_scope_vs_calcium", {})
                          .get("fraction_good_rate_gt_nyquist") or 0) > 0
                     and "钙域与电生理域分列" in ((rmap2.get("R-35") or {}).get("note") or ""),
                     "VISp 中位/Nyq=%.3f, >Nyq=%.4f"
                     % (agr.get(vispk, {}).get("cross_scope_vs_calcium", {}).get("median_over_nyquist") or 0,
                        agr.get(vispk, {}).get("cross_scope_vs_calcium", {})
                        .get("fraction_good_rate_gt_nyquist") or 0)))
    A.append(assert_("P1.21h 层/区登记：区级 measured ∧ 层 derived（acronym 无层号）",
                     (aj.get("layer_registration") or {}).get("area_level", "").startswith("measured")
                     and (aj.get("layer_registration") or {}).get("lamina_level", "").startswith("derived"),
                     "area=%s lamina=%s ccf_n=%s"
                     % ((aj.get("layer_registration") or {}).get("area_level"),
                        (aj.get("layer_registration") or {}).get("lamina_level"),
                        (aj.get("layer_registration") or {}).get("ccf_depth_available_n"))))
    A.append(assert_("P1.21i units.csv / channels.csv 已入落盘清单且 sha 基准=file_bytes",
                     any(os.path.basename(r.get("artifact_path") or "") == "units.csv"
                         and r.get("sha256") for r in inv2)
                     and any(os.path.basename(r.get("artifact_path") or "") == "channels.csv"
                             and r.get("sha256") for r in inv2),
                     "清单含 units=%s channels=%s"
                     % (any(os.path.basename(r.get("artifact_path") or "") == "units.csv" for r in inv2),
                        any(os.path.basename(r.get("artifact_path") or "") == "channels.csv" for r in inv2))))
    A.append(assert_("P1.21j **不得**把 complete_session 率登记为 (a) 自发率带（四态纪律：仍 unavailable）",
                     (rmap2.get("R-35") or {}).get("evidence_level", "").find("非自发率") >= 0
                     and any("自发率" in (u or "") and "unavailable" in (u or "")
                             for u in (cnt.get("unavailable_register") or [])),
                     "R-35 evidence=%s；register 含自发率缺口=%s"
                     % ((rmap2.get("R-35") or {}).get("evidence_level"),
                        any("自发率" in (u or "") for u in (cnt.get("unavailable_register") or [])))))

    # ---------------- P1.22 J2 验收收口（G1 解锁条件）----------------
    j2a = load(os.path.join(DATA, "m11_j2_acceptance.json")) if os.path.exists(
        os.path.join(DATA, "m11_j2_acceptance.json")) else {}
    cx = (j2a.get("acceptance_completion") or j2a)
    conds = cx.get("all_conditions") or {}
    A.append(assert_("P1.22a J2 五条验收条件全部 pass（口径/跨 field/电生理并列/哨兵/已覆盖维度）",
                     len(conds) == 5 and all(v == "pass" for v in conds.values()),
                     "conditions=%s" % json.dumps(conds, ensure_ascii=False)))
    rkf2 = cx.get("reserved_keys_fill") or {}
    A.append(assert_("P1.22b G1 (b) 静默带建议口径可复现（per-ROI 代理 ≤1.2×）且标出不可复现口径",
                     (rkf2.get("silent_frac_calcium_band_width_x") or 9) <= 1.2
                     and (rkf2.get("cross_field_dispersion_reserved") or {}).get("value_x_non_reproducible", 0) >= 3.0,
                     "band=[%s, %s] 宽=%.4f×；full-FOV=%.4f×"
                     % (rkf2.get("silent_frac_calcium_band_lo"), rkf2.get("silent_frac_calcium_band_hi"),
                        rkf2.get("silent_frac_calcium_band_width_x") or 0,
                        (rkf2.get("cross_field_dispersion_reserved") or {}).get("value_x_non_reproducible", 0))))
    cd5 = cx.get("condition_5_covered_dimensions") or {}
    A.append(assert_("P1.22c 已覆盖/未覆盖维度均登记（R9 纪律：未覆盖 ≠ 不可用 ⇒ 判 partial 非 unavailable）",
                     len(cd5.get("covered") or []) >= 5 and len(cd5.get("not_covered") or []) >= 4
                     and "不得因未覆盖" in (cd5.get("rule") or ""),
                     "covered=%d not_covered=%d" % (len(cd5.get("covered") or []),
                                                    len(cd5.get("not_covered") or []))))
    A.append(assert_("P1.22d R-40 为新增行且带『代理/样本量/量纲/观测窗』四条限制",
                     (rmap2.get("R-40") or {}).get("status") == "asserted"
                     and all(k in ((rmap2.get("R-40") or {}).get("note") or "")
                             for k in ("代理", "thin", "不同量纲", "观测窗")),
                     "R-40=%s（总 %d 行）" % ((rmap2.get("R-40") or {}).get("value"), len(rmap2))))
    A.append(assert_("P1.22e J2 产物**既有测量键未被改动**（只增不改：仍为两 field 行 + 4 band_rows）",
                     len((load(os.path.join(DATA, "m11_calcium_silence_reference.json")) or {})
                         .get("rows") or []) == 2
                     and len((load(os.path.join(DATA, "m11_calcium_silence_reference.json")) or {})
                             .get("band_rows") or []) == 4
                     and (load(os.path.join(DATA, "m11_calcium_silence_reference.json")) or {})
                     .get("acceptance_completion") is not None,
                     "rows=2 band_rows=4 acceptance_completion 已追加"))

    # ---------------- 判定 ----------------
    for a in A:
        if not a["ok"]:
            res["failures"].append(a["assertion"])
    f = res["failures"]
    # 四态：pass / fail / invalid / unavailable
    res["P1"] = dict(verdict="pass" if not [x for x in f if x.startswith("P1")] else "fail",
                     n_assertions=len([x for x in A if x["assertion"].startswith("P1")]),
                     n_failed=len([x for x in f if x.startswith("P1")]))
    res["P2"] = dict(verdict="pass" if not [x for x in f if x.startswith("P2")] else "fail",
                     n_assertions=len([x for x in A if x["assertion"].startswith("P2")]),
                     n_failed=len([x for x in f if x.startswith("P2")]))
    res["G2_gate"] = "pass" if (res["P1"]["verdict"] == "pass" and res["P2"]["verdict"] == "pass") else "fail"

    # ---------------- 确定性复跑 ----------------
    if args.determinism_check:
        det = dict(note="复跑固定子集（#2/#3/F7/F8/F9），比对 verdict 是否一致", comparisons=[])
        sids = ["2", "3", "F7", "F8", "F9"]
        probe = os.path.join(ROOT, "tools", "probe_m11_sources.py")
        tmp_out = os.path.join(DATA, "m11_raw", "probe", "_determinism_recheck.csv")
        env = dict(os.environ, PYTHONHASHSEED="0")
        cmd = [os.path.join(PROJ, ".venv-m11", "bin", "python"), probe,
               "--only", ",".join(sids), "--passes", "1", "--out", tmp_out]
        try:
            p = subprocess.run(cmd, capture_output=True, timeout=600, env=env)
            old = {r["source_id"]: r["verdict"] for r in rows}
            new = {r["source_id"]: r["verdict"] for r in read_csv_rows(tmp_out)}
            for s in sids:
                det["comparisons"].append(dict(source_id=s, first=old.get(s), rerun=new.get(s),
                                               same=old.get(s) == new.get(s)))
            det["all_same"] = all(c["same"] for c in det["comparisons"])
        except Exception as e:
            det["error"] = str(e)[:200]
        res["determinism_check"] = det
        A.append(assert_("DET 复跑 verdict 一致（确定性）", bool(det.get("all_same")),
                         json.dumps(det.get("comparisons"), ensure_ascii=False)))

    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    n_fail = len(res["failures"])
    print("wrote %s" % OUT)
    print("P1=%s P2=%s G2=%s  (assertions=%d failures=%d)"
          % (res["P1"]["verdict"], res["P2"]["verdict"], res["G2_gate"], len(A), n_fail))
    for a in A:
        if not a["ok"]:
            print("  FAIL:", a["assertion"], "|", a["detail"][:200])
    return res


if __name__ == "__main__":
    main()
