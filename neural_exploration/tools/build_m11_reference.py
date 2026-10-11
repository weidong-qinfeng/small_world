#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M11 · 参考锚落盘器（`data/m11_reference.csv`）——**读回断言值 + 证据级别 + 待核状态**

用途（主 agent 复核要求）
-------------------------
把 MICrONS 功能侧的**读回断言值**固化为 `data/m11_reference.csv`（每行 = 一个可断言的锚），
并在同一表内标注：① 值 ② 断言形式 ③ 出处 URL ④ 取回时间 ⑤ 证据级别 ⑥ **状态**
（`asserted` / `NEEDS_VERIFY` / `unverified`）。**`unverified` 行不得作为任何判据带。**

关键纪律
--------
- `frame_rate_hz = 6.300006`（= 1/0.15873 s）为**读回断言值**：与 ZAPBench 的 1.0943 Hz 差 **5.8×**
  ⇒ **RF-M11-2 的"自建 dwell/τ 带"是强制的**（跨物种 + 跨帧率 + 跨箱宽三重错配）。
- **dwell/τ 参考带仍标 `unverified`**：解析门（`validate_m11_microns_parse.py`）已过，但 dwell 带需
  P2 的参考量重建 + RF-M11-2 的"劈半 + 同协议等价"边界可达性自检后才允许定稿。
- 本表只**新增行**（清单 §0.3 #8 判据带定稿纪律）；修改只允许追加。

用法
----
    PYTHONHASHSEED=0 ./.venv-m11/bin/python neural_exploration/tools/build_m11_reference.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
PROTO = os.path.join(DATA, "m11_microns_protocol.json")
PARSE = os.path.join(DATA, "m11_microns_parse_validation.json")
INTEG = os.path.join(DATA, "m11_download_integrity_summary.json")
OUT = os.path.join(DATA, "m11_reference.csv")
BUCKET_URL = "https://allen-brain-observatory.s3.amazonaws.com/visual-coding-neuropixels/ecephys-cache"

FIELDS = ["ref_id", "quantity", "value", "unit", "assertion", "status", "evidence_level",
          "source_url", "retrieved_at", "note"]


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main():
    pr = json.load(open(PROTO))
    pa = json.load(open(PARSE)) if os.path.exists(PARSE) else {}
    ig = json.load(open(INTEG)) if os.path.exists(INTEG) else {}
    fs0 = (pr.get("protocol", {}).get("functional_series") or [{}])[0]
    masks = (pr.get("protocol", {}).get("field_seg_masks") or [{}])[0]
    raw = (pr.get("protocol", {}).get("raw_scan_bigtiff") or {})
    full = pa.get("full_objects", {})
    ti = pa.get("tiff_groundtruth") or {}
    rows = []

    def add(rid, q, val, unit, assertion, status, ev, url, at, note):
        rows.append(dict(ref_id=rid, quantity=q, value=val, unit=unit, assertion=assertion, status=status,
                         evidence_level=ev, source_url=url, retrieved_at=at, note=note))

    src = fs0.get("source_url", "")
    at = fs0.get("retrieved_at", "")
    add("R-01", "functional_series.frame_rate_hz", "%.6f" % fs0.get("frame_rate_hz", 0), "Hz",
        "abs(measured - 1/0.15873) < 1e-4", "asserted", "measured", src, at,
        "**主 agent 指定的读回断言值**；与 ZAPBench 1.0943 Hz 相差 5.8× ⇒ RF-M11-2 自建 dwell/τ 带为强制项")
    add("R-02", "functional_series.frame_interval_s", "%.5f" % fs0.get("frame_interval_s", 0), "s",
        "== 0.15873", "asserted", "measured", src, at, "attributes.json resolution[3]（单位 s）")
    add("R-03", "functional_series.n_timepoints", fs0.get("n_timepoints"), "frames",
        "== 40000", "asserted", "measured", src, at, "dimensions[3]")
    add("R-04", "functional_series.duration_s", "%.1f" % fs0.get("duration_s", 0), "s",
        "== 40000*0.15873", "asserted", "measured", src, at, "单 field 记录时长 105.82 min")
    add("R-05", "functional_series.n_planes", fs0.get("n_planes"), "planes",
        "== 1（该 series 单平面）", "asserted", "measured", src, at,
        "整卷体积率不可由此推出（多深度在 DB Scan 表）")
    add("R-06", "functional_series.volume_rate_hz", "%.6f" % fs0.get("volume_rate_hz", 0), "Hz",
        "== frame_rate / n_planes", "asserted", "measured", src, at, "z=1 ⇒ 体积率=帧率（**仅该 series**）")
    add("R-07", "chunk.block_frames", 10, "frames/chunk", "== 1.5873 s/chunk", "asserted", "measured",
        src, at, "blockSize=[200,200,1,10]（session5 为 [100,100,1,10]）")
    add("R-08", "chunk.prefix_bytes", 20, "bytes", "gzip 魔数出现在偏移 20", "asserted", "measured",
        src, at, "大端 uint32 头 [ndim=4, 200,200,1,10] ⇒ **不得按裸 gzip/裸 deflate 解**")
    add("R-09", "chunk.dtype_declared", "int16", "-", "attributes.json dataType", "asserted", "measured",
        src, at, "声明值；字节序见 R-10")
    add("R-10", "chunk.byte_order", "big-endian", "-",
        "结构门胜出（lat_x %.3f / frame %.3f，判据 ≥0.2）" % (full.get("lat_x_mean", 0), full.get("frame_mean", 0)),
        "asserted", "measured", "data/m11_microns_parse_validation.json", pa.get("generated_at", ""),
        "**本节点实测纠正**：元数据未写字节序，小端直读给出 lat≈0（假象）；大端直读 = 唯一自洽解")
    add("R-11", "chunk.parse_structural_gate", "pass", "-",
        "lat_x≥0.2 ∧ 相邻帧相关≥0.2 ∧ 值域合理 ∧ 无 NaN/Inf ∧ 声明域约束 ∧ 非退化",
        "asserted", "measured", "data/m11_microns_parse_validation.json", pa.get("generated_at", ""),
        "全量 %s 个 COMPLETE 对象复核：lat_x=%.3f lat_y=%.3f frame=%.3f"
        % (full.get("n_objects"), full.get("lat_x_mean", 0), full.get("lat_y_mean", 0), full.get("frame_mean", 0)))
    add("R-12", "chunk.value_stats", "p1=%.0f p50=%.0f p99=%.0f min=%s max=%s"
        % (full.get("p1_mean", 0), full.get("p50_mean", 0), full.get("p99_mean", 0),
           full.get("vmin_global"), full.get("vmax_global")), "counts",
        "p50>0 ∧ 非全 0 ∧ 非饱和（32767 计数=0）", "asserted", "measured",
        "data/m11_microns_parse_validation.json", pa.get("generated_at", ""),
        "与原始 TIFF 独立地面真值 p50=%.0f 同量级（不同 scan、未压缩）" % (ti.get("p50") or 0))
    add("R-13", "chunk.nan_inf_counts", "NaN=0, Inf=0", "samples",
        "整型域结构性为 0", "asserted", "measured", "data/m11_microns_parse_validation.json",
        pa.get("generated_at", ""),
        "主 agent 检出的 106 个 NaN 与 |x|>1e30 属 **float 误解读副产物**（本节点 float 假设行判 invalid）")
    add("R-14", "chunk.mask_rule", "无掩膜", "-",
        "below-baseline 样本保留并登记（%s 个 / 0.01298%%）" % full.get("n_int16_negative"),
        "asserted", "measured", "data/m11_microns_parse_validation.json", pa.get("generated_at", ""),
        "低于基线样本空间**成簇**（邻域负值比例 p50=%.4f、孤立比例 %.3f）且位于暗区（邻域均值≈395）"
        % (full.get("below_nb_negfrac_p50") or 0, full.get("below_isolated_frac") or 0))
    add("R-15", "voxel_size_um", "2.5 x 2.5 x 10", "um", "attributes.json resolution+units",
        "asserted", "measured", src, at, "**与文档 4 px/µm(0.25 µm/px) 相差 10× ⇒ 见 R-16**")
    add("R-16", "voxel_size_doc_conflict", "10x", "ratio",
        "doc 4 px/µm vs series 2.5 µm/px", "NEEDS_VERIFY", "measured(冲突)",
        "data/m11_raw/microns/pdf_text/functional_data_technical_documentation_v3.pdf.txt", at,
        "**P2 定稿前必须解决**：两种可能（series 为 xy 10× 降采样；或 resolution 字段语义不同）")
    add("R-17", "fov_um", "620 x 1100", "um", "dims x resolution", "asserted", "measured", src, at,
        "文档写 1100x1200 µm²（见 R-16 冲突）")
    add("R-18", "neuropil_mask_grid", "%s" % (masks.get("size_voxels")), "voxels",
        "== functional_series dims[:2]", "asserted", "measured", masks.get("url", ""),
        masks.get("retrieved_at", ""),
        "field_seg_masks（compressed_segmentation, uint64, %s nm）与 series **同格** ⇒ 同口径 ROI 可算"
        % (masks.get("resolution_nm")))
    add("R-19", "raw_scan.frame_shape", "%s" % (raw.get("image_description_json")), "frames x rows x cols",
        "BigTIFF ImageDescription (tag 270)", "asserted", "measured", raw.get("url", ""),
        raw.get("retrieved_at", ""), "Range 0-65535 只读头（**未下载 70 GB 大档**）；strip 218240 B/帧")
    add("R-20", "bounded_subset.complete_objects", "%s/%s" % (ig.get("n_complete"), ig.get("n_objects")),
        "objects", "verdict==COMPLETE（http→size→sha256→解压→尺寸自洽）", "asserted", "measured",
        "data/m11_download_integrity.csv", ig.get("generated_at", ""),
        "**完整性门**：首轮 197/200（3 片 curl_exit=28 半成品曾被误记 REACHABLE）→ 重取后 200/200")
    add("R-21", "dwell_tau_reference_band", "N/A", "-",
        "RF-M11-2：目标数据劈半 + 同协议等价检验", "unverified", "unavailable(未定稿)", "", "",
        "**强制**：M11 自建 dwell/τ 带；解析门已过但参考量重建（C1/C2）与劈半自检未完成 ⇒ "
        "**不得**沿用 M10/ZAPBench 的 26.5 s（跨物种+跨帧率+跨箱宽三重错配）")
    add("R-22", "tiff_groundtruth_coherence", "lat_x=%.3f lat_y=%.3f"
        % (ti.get("lat_x") or 0, ti.get("lat_y") or 0), "corr",
        "原始 TIFF 帧（未压缩、不同 scan）亦为高相关图像", "asserted", "measured", ti.get("url", ""),
        utcnow(), "独立地面真值 ⇒ 判据阈值 0.2 由真实数据锚定（非自证）；n_frames=%s（Range 前缀）"
                  % ti.get("n_frames"))

    # ---- RF-M11-A：活动通道回退源（Allen BO Neuropixels 自描述表）采样率读回 ----
    act = json.load(open(os.path.join(DATA, "m11_allen_activity_fallback.json"))) \
        if os.path.exists(os.path.join(DATA, "m11_allen_activity_fallback.json")) else {}
    sr = (act.get("sampling_rate_readback") or {})
    if sr:
        sp = sr.get("sampling_rate_hz") or {}
        lf = sr.get("lfp_sampling_rate_hz") or {}
        add("R-23", "allen_ecephys.spike_sampling_rate_hz", "%.4f" % (sp.get("max") or 0), "Hz",
            "== 30000（spike-band；probes.csv 自描述）", "asserted", "measured",
            BUCKET_URL + "/probes.csv", act.get("generated_at", ""),
            "**RF-M11-A 回退源采样率读回**（%d 个 probe 中 %d 个有值；均值 %.2f Hz）；"
            "min=0 属无 spike 数据的占位行" % (sr.get("n_probes", 0), sr.get("n_with_sampling_rate", 0),
                                              sp.get("mean") or 0))
        add("R-24", "allen_ecephys.lfp_sampling_rate_hz", "%.4f" % (lf.get("max") or 0), "Hz",
            "== 2500（LFP；probes.csv 自描述）", "asserted", "measured", BUCKET_URL + "/probes.csv",
            act.get("generated_at", ""),
            "LFP 采样率；`lfp_temporal_subsampling_factor=%s`；has_lfp_data 的 probe 数=%s"
            % (sr.get("lfp_temporal_subsampling_factor"), sr.get("n_has_lfp_data")))
        add("R-25", "allen_ecephys.units_table_schema", "%s 列" % ((act.get("units_csv") or {}).get("n_columns")),
            "columns", "含 firing_rate / snr / isi_violations / quality", "asserted", "measured(前缀)",
            BUCKET_URL + "/units.csv", act.get("generated_at", ""),
            "**全量 51,439,189 B 未取**（网络 16 KB/s ⇒ 预计 >30 min，DQ-6）；1 MB Range 前缀读回 schema+样本")
        add("R-26", "microns.voxel_channel_status", "decoded(big-endian int16)", "-",
            "lat_x≥0.2 ∧ 帧轴 lag1≥0.2 ∧ 声明域内", "asserted", "measured",
            "data/m11_decode_proof.json", utcnow(),
            "**不登记 format_unverified**：大端 int16 直读给出 lat_x=0.814 / 帧轴 lag1..6=0.803…0.709 / p50=1953，"
            "并有独立地面真值（原始 TIFF lat_x=0.767）佐证；一条命令可复核（tools/demo_m11_microns_decode.py）")

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        f.write("# M11 参考锚（generated_by=tools/build_m11_reference.py；**只允许新增行**；"
                "status ∈ {asserted, NEEDS_VERIFY, unverified}；**unverified 行不得作为判据带**）\n")
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    n_a = sum(1 for r in rows if r["status"] == "asserted")
    print("wrote %s (%d rows: asserted=%d NEEDS_VERIFY=%d unverified=%d)"
          % (OUT, len(rows), n_a, sum(1 for r in rows if r["status"] == "NEEDS_VERIFY"),
             sum(1 for r in rows if r["status"] == "unverified")))
    return rows


if __name__ == "__main__":
    main()
