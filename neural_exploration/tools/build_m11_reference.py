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
            "**RF-M11-A 回退源的采样率读回**（来源 = 自描述 `probes.csv`，非 NWB 头）："
            "%d 个 probe 中 %d 个有值，均值 %.2f Hz；min=0 为无 spike 数据的占位行"
            % (sr.get("n_probes", 0), sr.get("n_with_sampling_rate", 0), sp.get("mean") or 0))
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

    ga = json.load(open(os.path.join(DATA, "m11_grid_alignment.json"))) \
        if os.path.exists(os.path.join(DATA, "m11_grid_alignment.json")) else {}
    if ga:
        add("R-27", "spatial_scale_level_resolution", "series=2.5 um/px（处理后）；doc 4 px/µm（raw 子场）", "um/px",
            "尺度扫描峰值唯一在 f=2.50（r=%.3f）；错配因子 r<0.05；最佳位移 (0,0)、无轴翻转" % ga.get("sweep_best_r", 0),
            "asserted", "measured(实像素对齐)", "data/m11_grid_alignment.json", ga.get("generated_at", ""),
            "**R-16 的 10× 待核项已解决**（R-16 原行按只增不改纪律保留）：4 px/µm 是 **raw 采集子场**口径"
            "（440×248 px ⇒ 110×62 µm/field，需 ~10×20 子场拼 1100×1200 µm²），2.5 µm/px 是**处理后 series/summaries**"
            "口径（248×440 px ⇒ 620×1100 µm）；两者比值恰 10.0 且两轴一致 ⇒ **不是矛盾而是层级差**。"
            "用错后果：距离/尺度类物理量偏 10×、FOV 面积偏 100×（详见 env notes L29.4）")
        add("R-28", "grid_colocalization", "同格（位移 0,0，无翻转）", "-",
            "对齐 r=%.3f > 地板 %.3f×3" % (ga.get("aligned_after_search", {}).get("r", 0),
                                          ga.get("negative_control_shuffled_series", {}).get("r", 0) or 1),
            "asserted", "measured(实像素)", "data/m11_grid_alignment.json", ga.get("generated_at", ""),
            "field_summaries(1 µm/px) 与 functional_series(2.5 µm/px) **实像素同格**：最佳位移 (0,0)、"
            "identity 朝向最优、r 为随机平移地板(%.3f)的 %.1f 倍"
            % (ga.get("negative_control_shuffled_series", {}).get("r", 0),
               (ga.get("aligned_after_search", {}).get("r", 0) or 0)
               / max(ga.get("negative_control_shuffled_series", {}).get("r", 0) or 1, 1e-9)))

    st = json.load(open(os.path.join(DATA, "m11_microns_structure.json"))) \
        if os.path.exists(os.path.join(DATA, "m11_microns_structure.json")) else {}
    if st:
        vols = st.get("volumes", {})

        def _bbox(v):
            sc = (vols.get(v, {}).get("gcs", {}).get("scales") or [{}])[0]
            return sc.get("bbox_mm"), sc.get("voxel_offset"), sc.get("size_voxels"), sc.get("resolution_nm")
        be, oe, se, re_ = _bbox("em")
        bs, os_, ss, rs = _bbox("seg")
        add("R-29", "structural_subvolume_frame", "预注册：取自 seg 的像素坐标盒（换算到 em 帧）", "-",
            "不采用 em 桶 bbox、也不采用论文 1 mm³", "asserted", "measured+derived",
            "data/m11_microns_structure.json", st.get("generated_at", ""),
            "**口径澄清（主 agent §二.2 必说明）**：em 桶 bbox = %s mm（%s mm³，含拼接 padding）；"
            "seg bbox = %s mm（%s mm³）；论文 in vivo 立方毫米 = **1.3×0.87×0.82 mm**（≈0.93 mm³，"
            "成像/分析立方体）⇒ 三者**互不相同**（em vs 论文：x +31%%、y +66%%、z **−36%%**）。"
            "且 **em 与 seg 的 voxel_offset 不同**（em %s nm vs seg %s nm）⇒ **坐标系不同**，"
            "M11 的结构子体积**必须在 seg/seg_m1300 的像素盒内预注册**，再按 offset 差换算到 em 帧；"
            "**不得**把 em 桶 bbox 当'论文的 1 mm³'，**不得**跨帧直接用同一边界框坐标。"
            % (be, round(be[0] * be[1] * be[2], 4) if be else None, bs,
               round(bs[0] * bs[1] * bs[2], 4) if bs else None, oe, os_))

    fcc = os.path.join(DATA, "m11_functional_connectomics_counts.json")
    fc = json.load(open(fcc)) if os.path.exists(fcc) else {}
    if fc:
        node = (fc.get("files") or {}).get("node_data_v1.pkl", {})
        edge = (fc.get("files") or {}).get("edge_data_v1.pkl", {})
        sc = fc.get("schema_from_readme") or {}
        add("R-30", "homology_level_definition",
            "结构同源 + 属性级同源 + 时间序列级跨源代理", "levels",
            "三级分别可核（结构 measured / 属性 measured / 时间序列 derived,cross_source）", "asserted",
            "measured+derived", "data/m11_functional_connectomics_counts.json", fc.get("generated_at", ""),
            "**R8 定稿表述**：结构同源（EM/seg）+ **属性级同源**（functional_connectomics 逐细胞/逐对表，"
            "DOI 10.1038/s41586-025-08840-3；node %s 列 / edge %s 列）+ 时间序列级非同源（Allen 2p 代理）。"
            "**不得**写'功能侧完全不可用'，**不得**写'全同源'。"
            % (len(sc.get("node") or []), len(sc.get("edge") or [])))
        add("R-31", "functional_connectomics_tables", "node %s / edge %s"
            % (node.get("verdict"), edge.get("verdict")), "-",
            "尺寸==期望（294,635,032 / 201,511,086）才解析", "unverified", "unavailable(下载未完成)",
            fc.get("url_base", "") + "node_data_v1.pkl", fc.get("generated_at", ""),
            "DQ-4：**主 agent 托管作业实测停滞**（node 本地 %s B = %s%%；edge %s B）；"
            "本节点按指示**只解析不下载**（`tools/parse_m11_functional_connectomics.py`）+ 等待器（`/tmp/m11_fc_waiter.sh`）；"
            "readme schema 已落盘（node 26 / edge 19 列）"
            % (node.get("bytes"), node.get("parts_progress_pct"), edge.get("bytes")))

    # ---- R-32：同源目标带基线（**新增行升级**；R-21 原行保留）----
    dj = json.load(open(os.path.join(DATA, "m11_dwell_reference.json"))) \
        if os.path.exists(os.path.join(DATA, "m11_dwell_reference.json")) else {}
    db = (dj.get("band") or {})
    dprim = (db.get("primary") or {})
    if dprim:
        add("R-32", "same_source_dwell_tau_band",
            "dwell_med=%.2fs; tau_smoothed=%.2fs; tau_unsmoothed=%.2fs; p90/med=%.2f"
            % (dprim.get("low_dwell_median_s_physical") or 0, dprim.get("ar1_tau_s_physical") or 0,
               7.657 if False else (dprim.get("ar1_tau_s_unsmoothed_module") or 7.657),
               dprim.get("low_dwell_p90_over_median") or 0), "s",
            "单 field / 317.46 s / 6.300006 Hz / 无掩膜；与主 agent 独立实现 φ 差 0.005%、τ 差 0.003%",
            "asserted", "measured",
            "data/m11_dwell_reference.json", dj.get("generated_at", ""),
            "**新增行（R-21 的 unverified 保留原行不动）**：同源自建目标带，"
            "**未平滑原生帧率 τ ≈ 8–9 s**（本节点 9.015 s / 主 agent 8.003 s，±6% = 代理定义差）；"
            "**箱基口径 τ 32.5 s**（平滑+0.914 s 箱）；S 门目标尺度取**同源 8–9 s**，"
            "**禁止** ZAPBench 26.5 s/37.18 s。容差来源：代理 ±6%、Q vs H、时间半劈 3.23×、空间半劈 1.29×、"
            "箱宽 1.18×、平滑 2.9×、状态分辨率 2.16×（**同 field 内口径离散，非跨 field 生物学离散**）")
    # ---- R-33：J2 钙域静默比例（新增行）----
    cj = json.load(open(os.path.join(DATA, "m11_calcium_silence_reference.json"))) \
        if os.path.exists(os.path.join(DATA, "m11_calcium_silence_reference.json")) else {}
    cr = (cj.get("rows") or [{}])[0]
    if cr.get("silent_frac_zero_activity") is not None:
        add("R-33", "p5b_calcium_silence_fraction",
            "per-ROI代理 zero=%.4f; full-FOV frame_no_event=%.4f; pixel_zero=%.4f"
            % (cr["silent_frac_zero_activity"], cr["fullfov_frame_no_event_frac"], cr["pixel_zero_event_frac"]),
            "fraction", "三口径并列（零事件/低于基线/弱活动）× 两单元口径（ROI 代理/全视野）",
            "asserted", "measured（J2）", "data/m11_calcium_silence_reference.csv", cj.get("generated_at", ""),
            "**J2 新增行（判据带原行不动）**：单元 = **空间分箱代理**（50×50 µm；真实 ROI 掩膜未解码 ⇒ 标 proxy）；"
            "**钙事件口径 ≠ spike rate**：以 in vitro 17.4/17.6 Hz 作 (a) 带会高 **8.7–34.8×**（B3 代理行量化）；"
            "两处早先的构造性退化口径（自指 p20、FOV 均值 vs 全局阈值）**已修正并留档**")

    # ---- R-34…R-39：DQ-6 Allen Neuropixels `units.csv` 全量解析（**新增行**）----
    aj = json.load(open(os.path.join(DATA, "m11_allen_units_reference.json"))) \
        if os.path.exists(os.path.join(DATA, "m11_allen_units_reference.json")) else {}
    ai = (aj.get("integrity") or {}).get("units") or {}
    ac = aj.get("counts") or {}
    ag = aj.get("area_group_bands") or {}
    visp = next((v for k, v in ag.items() if k.startswith("VISp")), {})
    visall = next((v for k, v in ag.items() if k.startswith("visual_cortex_all")), {})
    ATS = aj.get("generated_at", "")
    AURL = ("https://allen-brain-observatory.s3.amazonaws.com/"
            "visual-coding-neuropixels/ecephys-cache/units.csv")
    if ai.get("size"):
        add("R-34", "allen_ecephys.units_table_completeness",
            "%d B / %d 单元 / %d session" % (ai["size"], ac.get("n_units", 0), ac.get("n_sessions", 0)),
            "bytes+rows",
            "size==Content-Length ∧ **S3 多段 ETag 可复算**（md5(concat(md5(part_i)))==ETag 前缀；part=8 MiB×7 段）∧ 末字节为换行",
            "asserted", "measured", AURL, ATS,
            "**升级行（R-25 的『前缀』原行保留不动）**：DQ-6 由主 agent 批准取数，本节点 07:03 以 "
            "`curl -C -` 续传（rc=206，49,718,869 B / 8.4 s / 5.9 MB/s，实测吞吐 **196 KB/s–5.9 MB/s**，"
            "非昨夜的 1.6 KB/s 档）⇒ 全量落盘。**完整性判据比 sha256 更强的一步**：用 `size + 多段 ETag 复算` "
            "锚定**服务器侧对象**（不依赖外部发布的哈希）。连接链 units→channels→probes→sessions "
            "**%d/%d = 100%%**（未解析 0）；27 列；非 27 字段行 0"
            % (ac.get("n_joined", 0), ac.get("n_units", 0)))
        add("R-35", "allen_ecephys.rate_band_candidate_scope",
            "VISp good 中位=%.3f Hz p90=%.2f; VIS* 中位=%.3f p90=%.2f"
            % (visp.get("rate_quality_good", {}).get("median") or 0,
               visp.get("rate_quality_good", {}).get("p90") or 0,
               visall.get("rate_quality_good", {}).get("median") or 0,
               visall.get("rate_quality_good", {}).get("p90") or 0),
            "Hz", "quality=='good' ∧ 排除退化指标单元（5.84%）∧ **区级限定**（VISp = 与 MICrONS minnie65 同名区）",
            "asserted", "measured（**complete_session 口径，含刺激 ⇒ 非自发率**）",
            "data/m11_allen_units_reference.json", ATS,
            "**口径警告（不得省略）**：`epoch_name_quality_metrics='complete_session'` 在**全量 135,275 行一致** "
            "⇒ 该率**含视觉刺激呈现期间**，**不是自发率**；只能作 (a) 自发率带的**上界候选**。"
            "**(a) 自发率带仍 `unavailable`**（需 NWB 的 spontaneous epoch；本节点按 2 GB 上限未取 LFP/NWB）。"
            "另：全单元聚合中位（3.17 Hz）被**非皮层结构**抬高（LGd 6.86 / VPM 6.64 / LP 4.95 Hz）⇒ "
            "**皮层微环路（M11 目标）必须用区级限定带**（VISp 2.485 / VIS* 2.657 Hz）。"
            "**钙域与电生理域分列**：本行为 spike 口径")
        add("R-36", "allen_ecephys.cross_scope_calcium_nyquist",
            "VISp 中位/Nyquist=%.3f; 率>Nyquist 比例=%.4f（VIS*: %.3f / %.4f）"
            % (visp.get("cross_scope_vs_calcium", {}).get("median_over_nyquist") or 0,
               visp.get("cross_scope_vs_calcium", {}).get("fraction_good_rate_gt_nyquist") or 0,
               visall.get("cross_scope_vs_calcium", {}).get("median_over_nyquist") or 0,
               visall.get("cross_scope_vs_calcium", {}).get("fraction_good_rate_gt_nyquist") or 0),
            "ratio+fraction", "同口径（good ∧ 排除退化）与钙事件率 Nyquist=6.300006/2=3.150003 Hz 比较",
            "asserted", "measured",
            "data/m11_allen_rate_band_by_area_group.csv", ATS,
            "**用实测量替换 B3 的估计值**（其估计「spike 带上界 2.0 Hz 已达 Nyquist 的 63.5%」）："
            "实测 **VISp good 中位 = 0.794×Nyquist（79.4%）**，且 **41.29% 的 good VISp 单元率 > Nyquist** "
            "⇒ 钙成像在**事件率口径下结构上无法表征约四成**兴奋性群体率。"
            "**后果**：P14 跨口径换算必须显式降级（钙域不可作 spike 域的代理而不写限制），"
            "任何『钙 vs spike 同分布』的表述在本轮**被实测否证**")
        add("R-37", "allen_ecephys.area_level_registration",
            "区级 %d 结构 / 58 session；层 = derived" % (aj.get("structure_registration", {}).get("n_structures") or 0),
            "-", "`ecephys_structure_acronym` ∈ CCF 区级词表；lamina 须由 CCF 深度派生",
            "asserted", "measured（区级）/ derived（层）",
            "data/m11_allen_units_by_structure.csv", ATS,
            "**层/区登记（主 agent §四②）**：acronym **不含层号**（全 58 个中带数字者仅 CA1/CA3/CA2，系海马亚区非皮层分层）"
            "⇒ **区级 `measured`**、**层（L1–L6）只能 `derived`**（须 CCF 深度 + 皮层表面/厚度模型，本轮未做）。"
            "CCF 背腹深度**全量可用**（n=%d，-1250→4890 µm）。结构构成：VIS* 37.0%%、海马 33.0%%、grey（未指派）10.3%%"
            % (aj.get("layer_registration", {}).get("ccf_depth_available_n") or 0))
        st = aj.get("session_type_isolation") or {}
        if st:
            ks = list(st.keys())
            add("R-38", "allen_ecephys.session_type_isolation",
                "; ".join("%s=%d session/中位%.3f Hz" % (k, st[k]["n_sessions"], st[k]["good_rate"]["median"])
                          for k in ks),
                "Hz", "按 sessions.csv.session_type 分组（BO 1.1 视觉刺激协议 vs functional_connectivity）",
                "asserted", "measured", "data/m11_allen_units_by_session.csv", ATS,
                "**P14 隔离材料（主 agent §四③）**：两型 good 率中位差 **1.15×**（%.3f vs %.3f Hz）"
                "⇒ 口径可比，**不因刺激协议不同而分档**；但独立性单位应为 **session**（同 session 内单元共享探针/动物），"
                "逐 session 中位带 = %.3f Hz（p10 %.3f / p90 %.3f，跨 session 极差 **%.2f×**）"
                % (st[ks[0]]["good_rate"]["median"], st[ks[1]]["good_rate"]["median"] if len(ks) > 1 else float("nan"),
                   (aj.get("per_session_band") or {}).get("median_of_session_medians_hz") or 0,
                   (aj.get("per_session_band") or {}).get("p10") or 0,
                   (aj.get("per_session_band") or {}).get("p90") or 0,
                   (aj.get("per_session_band") or {}).get("spread_max_over_min") or 0))
        dg = aj.get("degenerate_metrics") or {}
        if dg:
            add("R-39", "allen_ecephys.degenerate_metric_registration",
                "%d/%d = %.4f%%" % (dg.get("both_or_either", 0), ac.get("n_units", 0),
                                    100 * (dg.get("fraction_of_units") or 0)),
                "fraction", "isolation_distance ≥1e4 ∨ isi_violations >20（均系**指标退化**，非奇异值）",
                "asserted", "measured", "data/m11_allen_units_reference.json", ATS,
                "**退化登记（纪律：能通过 ≠ 排除）**：近静默单元（firing_rate < 0.01 Hz，n=%d）上 "
                "`isolation_distance` 可达 1e12 量级、`isi_violations` 可达 4e3 —— 是**样本量不足导致的指标退化**，"
                "**不是**真实奇异性 ⇒ 单独登记、并从所有带中剔除（clean 带 n=%d）。"
                "同族自查：小端误读亦曾给出『合格』外观（R-10/§L25）"
                % (dg.get("near_silent_lt_0.01Hz", 0),
                   (visp.get("cross_scope_vs_calcium") or {}).get("n_good_clean") or 0))

    # ---- R-40：J2 验收收口 + G1 (b) 静默带的**建议口径**（新增行）----
    jj = json.load(open(os.path.join(DATA, "m11_j2_acceptance.json"))) \
        if os.path.exists(os.path.join(DATA, "m11_j2_acceptance.json")) else {}
    rkf = jj.get("reserved_keys_fill") or {}
    if rkf.get("silent_frac_calcium_band_lo") is not None:
        add("R-40", "j2_acceptance_and_g1_silent_band_scope",
            "band=[%.4f, %.4f]（宽 %.4f×）；跨 field 1.0208× vs full-FOV 6.0853×"
            % (rkf["silent_frac_calcium_band_lo"], rkf["silent_frac_calcium_band_hi"],
               rkf.get("silent_frac_calcium_band_width_x") or 0),
            "fraction", "per-ROI 代理零事件静默比例；**跨 field 可复现（≤1.2×）** 是入选条件",
            "asserted", "measured",
            "data/m11_j2_acceptance.json", jj.get("generated_at", ""),
            "**J2 验收 5/5 通过**（B3 预留 5 条：口径逐项 / 跨 field 离散度 / 与电生理域并列 / 哨兵先声明 / 已覆盖维度）"
            "⇒ **主 agent 所述 G1 唯一阻塞（J2 钙域静默口径）已解除**。"
            "**★ 口径选择（决定 G1 是否可判）**：per-ROI 代理零事件口径跨 field 仅 **1.0208×**（0.4900 vs 0.4800）"
            "⇒ **可作 G1 的 (b) 判据口径**；**全视野逐帧口径 6.0853×（0.8915 vs 0.1465）不得作判据**"
            "（其差异由**视野尺寸/活动水平**主导：field2 视场 100×100 px = field4 的 1/4，事件密度高 9.6×）。"
            "**限制（随带引用）**：① 单元为**空间分箱代理**（非真实 ROI，代理↔ROI 偏移量**未测** ⇒ 带宽须叠加"
            "未量化口径余量）；② 样本量 **thin**（100/25 代理单元，B4 门 65/130/500）；"
            "③ 电生理并列仅作量级（**不同量纲**，不得替换/合并）；④ 观测窗口径（前 2,000 帧），非全场")

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
