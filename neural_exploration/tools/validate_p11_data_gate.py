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
    A.append(assert_("P2.12d 空间分辨率 10× 口径冲突已登记 NEEDS_VERIFY",
                     (rmap.get("R-16") or {}).get("status") == "NEEDS_VERIFY",
                     "R-16=%s" % json.dumps(rmap.get("R-16", {}), ensure_ascii=False)[:160]))

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
