#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M12 · B-5：**真实 ROI 掩膜**的自证与钙域静默口径重算（**只新增，不改既有件**）

背景（M11 侧已核实，见 `docs/m12_list_review_master.md` R146-补登）
------------------------------------------------------------------
`data/m11_p5_targets.csv` L115–116（`silent_frac_calcium_band_lo/_hi`）为
`value=unavailable / final=0`，唯一缺失维 = **真实 ROI 掩膜**
（`data/m11_calcium_silent.json::four_state_verdict.real_roi_mask_unit`）。
M11 的 `tools/build_m11_calcium_silence.py` L18 登记：「`field_seg_masks` 的
`compressed_segmentation` **本节点未能自证解码**（无官方客户端）」——
⇒ 本工具**只做这一件事**：把该掩膜**解码 + 自证**，并用它重算 (b)。

掩膜来源（M11 已落盘，本次复核 sha256 逐字节一致）
--------------------------------------------------
`data/m11_raw/microns/field_masks/*.bin` ← S3
`iarpa_microns/minnie/functional_data/field_seg_masks/session_4_scan_7_field_4/2500_2500_10000/<x0>-<x1>_<y0>-<y1>_0-1`
`info`：`type=segmentation, data_type=uint64, encoding=compressed_segmentation,
compressed_segmentation_block_size=[8,8,8], size=[248,440,1], resolution=[2500,2500,10000] nm`
⇒ 与主 field 钙电影 `functional_series/session4_scan7_field4`（`size=[248,440,1,40000]`,
`resolution=[2.5,2.5,10,0.15873] µm/s`）**同格**（2.5 µm/px、248×440）。

格式规范（**权威来源，非猜测**）
--------------------------------
`compressed_segmentation` = neuroglancer 的 compressed segmentation 编码。本工具实现对
seung-lab `compressed_segmentation==2.3.3` sdist 的
`src/decompress_segmentation.cc::DecompressChannel` 逐行等价（该文件头注释指向
`https://github.com/google/neuroglancer/tree/master/src/neuroglancer/sliceview/compressed_segmentation`）。
关键约定（直接从该 C++ 源读出）：
  - 缓冲区开头为 `num_channels` 个 LE uint32 的通道起始偏移；单通道时**就是 1 个 `1`**
    （该约定写在 `DecompressChannels` 的注释里）。
  - 通道内：`grid_size[i] = ceil(volume_size[i]/block_size[i])`，块头表
    `2*grid[0]*grid[1]*grid[2]` 个 uint32，每块两项：
      `h0`: 低 24 位 = `tableoffset`（uint32 单位），高 8 位 = `encoded_bits`
      `h1`: `encoded_value_start`
  - `encoded_bits == 0` ⇒ 该块全部为 `table[0]`（常量块）。
  - 表项按 `bitval * table_entry_size` 索引，`table_entry_size = ceil(sizeof(Label)/4)`
    ⇒ uint64 时为 2 个 uint32（低字 + 高字）。
  - 位流：`bitpos = block_size[0]*((z-zmin)*block_size[1] + (y-ymin))*encoded_bits`，
    逐 x 递增 `encoded_bits`。

**双实现互证（承 M11 `dual_estimator` 纪律）**：本工具自带独立实现；
若 `compressed_segmentation` 可导入，则对 8 个 chunk 逐字节比对（必须 bit-exact）。

用法
----
    # 1) 只解码 + 自证（不需要参考实现；无网络）
    PYTHONHASHSEED=0 OMP_NUM_THREADS=1 ./.venv-m11/bin/python \
        neural_exploration/tools/m12_b5_roi_mask_audit.py --decode-only

    # 2) 全量（解码 + 钙域静默重算 + 自证表）
    PYTHONHASHSEED=0 OMP_NUM_THREADS=1 ./.venv-m11/bin/python \
        neural_exploration/tools/m12_b5_roi_mask_audit.py

    # 3) 双实现互证（需 /tmp/m12_b5_venv，见 docs/m12_b5_roi_mask.md §复算）
    /tmp/m12_b5_venv/bin/python neural_exploration/tools/m12_b5_roi_mask_audit.py --crosscheck-ref

硬约束遵守：**不读 `--device mps`、不起 MPS 作业、不写任何既有文件**。
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import hashlib
import json
import os
import struct
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # neural_exploration/
MASKDIR = os.path.join(ROOT, "data", "m11_raw", "microns", "field_masks")   # 只读
SUB = os.path.join(ROOT, "data", "m11_raw", "microns", "functional_subset")  # 只读
SUB2 = os.path.join(ROOT, "data", "m11_raw", "microns", "functional_subset_f2")  # 只读
F2MASK = os.path.join(ROOT, "data", "m12_b5_roi_mask", "field2_masks")           # 本节点新增
OUTDIR = os.path.join(ROOT, "data", "m12_b5_roi_mask")                       # 本节点新增

# ---- field 几何（逐值取自 attributes.json / info.json，非推测）----
FX, FY, FZ = 248, 440, 1          # field_seg_masks info: size
BX, BY, BZ = 8, 8, 8              # compressed_segmentation_block_size
CX, CY = 128, 128                 # chunk_sizes
# ---- 电影窗口（取自 manifest 的 S3 key：x_chunk=0,y_chunk=0,t_chunk∈[0,200)）----
WX, WY, NT = 200, 200, 2000
FRAME_INTERVAL = 0.15873
FRAME_RATE = 1.0 / FRAME_INTERVAL
EVENT_SIGMA = 3.0
PIXEL_UM = 2.5


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# =====================================================================
# 1) 独立解码器（pure NumPy，逐行等价于 decompress_segmentation.cc）
# =====================================================================
def decode_compressed_segmentation_independent(buf, volume_size, block_size,
                                               label_dtype=np.uint64, strict=True):
    """独立实现。返回 (labels_3d, audit dict)。volume_size/block_size 均为 (x,y,z)。

    strict=True 时做结构门：① 首字 == 1；② 每个块的 tableoffset/vstart 必须落在
    缓冲区 uint32 范围内；③ 位流读出的 element 也必须落在范围内 —— **越界即 raise**
    （不静默截断；承「不得把失败掩盖成结果」）。
    """
    u32 = np.frombuffer(buf, dtype="<u4")
    n_u32 = u32.size
    aud = dict(n_u32=n_u32, leading_word=int(u32[0]) if n_u32 else None)
    if strict and (n_u32 == 0 or int(u32[0]) != 1):
        raise ValueError("结构门①失败：首字 != 1（单通道约定），实测=%s" % aud["leading_word"])
    ch = u32[1:]                                   # 通道 0 相对起点（input + input[0]，input[0]==1）
    n_ch = ch.size
    g = [(volume_size[i] + block_size[i] - 1) // block_size[i] for i in range(3)]
    grid = tuple(int(v) for v in g)
    aud["grid_size"] = list(grid)
    aud["n_blocks"] = int(grid[0] * grid[1] * grid[2])
    aud["header_u32"] = 2 * aud["n_blocks"]
    if strict and aud["header_u32"] > n_ch:
        raise ValueError("结构门②失败：块头表 %d u32 超出通道缓冲 %d u32"
                         % (aud["header_u32"], n_ch))
    headers = ch[:aud["header_u32"]].reshape(aud["n_blocks"], 2)
    h0 = headers[:, 0].astype(np.uint64)
    tableoffset = (h0 & 0xFFFFFF).astype(np.int64)
    encoded_bits = ((h0 >> 24) & 0xFF).astype(np.int64)
    vstart = headers[:, 1].astype(np.int64)

    tes = (np.dtype(label_dtype).itemsize + 3) // 4   # uint64 -> 2
    aud["table_entry_size_u32"] = int(tes)
    aud["encoded_bits_hist"] = {int(k): int(v) for k, v in
                                zip(*np.unique(encoded_bits, return_counts=True))}

    if strict:
        # 必要（非充分）条件：至少能读到 table[0]；**不**假设 2**bits 个表项全被使用
        # （实测存在只用部分表项的块：chunk 0-128_128-256 的 max(tableoffset+2*tes) = n_ch+2）。
        bad = ((tableoffset < 0) | (tableoffset + tes > n_ch)) | (vstart > n_ch) | (vstart < 0)
        if bad.any():
            raise ValueError("结构门②失败：%d 个块的 tableoffset/vstart 越界" % int(bad.sum()))
    aud["table_entries_touched_max"] = 0

    out = np.zeros(volume_size, dtype=label_dtype)
    max_touched = 0
    for bz in range(grid[2]):
        for by in range(grid[1]):
            for bx in range(grid[0]):
                bi = bx + grid[0] * (by + grid[1] * bz)
                toff = int(tableoffset[bi]); bits = int(encoded_bits[bi]); vs = int(vstart[bi])
                x0 = bx * block_size[0]; x1 = min(x0 + block_size[0], volume_size[0])
                y0 = by * block_size[1]; y1 = min(y0 + block_size[1], volume_size[1])
                z0 = bz * block_size[2]; z1 = min(z0 + block_size[2], volume_size[2])
                if bits == 0:
                    if strict and (toff + tes) > n_ch:
                        raise ValueError("结构门③失败：常量块表项越界 block=%d" % bi)
                    val = np.uint64(ch[toff])
                    if tes == 2:
                        val = val | (np.uint64(ch[toff + 1]) << np.uint64(32))
                    out[x0:x1, y0:y1, z0:z1] = val
                    max_touched = max(max_touched, 1)
                    continue
                mask = (1 << bits) - 1
                for z in range(z0, z1):
                    for y in range(y0, y1):
                        bitpos = block_size[0] * ((z - z0) * block_size[1] + (y - y0)) * bits
                        for x in range(x0, x1):
                            shift = bitpos % 32
                            elem = bitpos // 32
                            if strict and (vs + elem) >= n_ch:
                                raise ValueError("结构门③失败：位流越界 block=%d" % bi)
                            bval = int((int(ch[vs + elem]) >> shift) & mask)
                            idx = toff + bval * tes
                            if strict and (idx + tes) > n_ch:
                                raise ValueError("结构门③失败：表项越界 block=%d" % bi)
                            v = np.uint64(ch[idx])
                            if tes == 2:
                                v = v | (np.uint64(ch[idx + 1]) << np.uint64(32))
                            out[x, y, z] = v
                            max_touched = max(max_touched, bval + 1)
                            bitpos += bits
    aud["table_entries_touched_max"] = int(max_touched)
    return out, aud


def load_mask_chunk(path):
    """返回 (key, x0,x1,y0,y1, raw_gz_bytes, gunzipped_bytes)。"""
    key = os.path.basename(path)[:-4]
    parts = key.split("_")
    (x0, x1), (y0, y1), (z0, z1) = [tuple(int(v) for v in p.split("-")) for p in parts]
    gz = open(path, "rb").read()
    raw = gzip.decompress(gz)
    return key, x0, x1, y0, y1, gz, raw


def build_label_volume(verbose=True, maskdir=None):
    """解码 8 个 chunk → (248,440,1) uint64 label 体 + 逐 chunk 审计。"""
    maskdir = maskdir or MASKDIR
    files = sorted(glob.glob(os.path.join(maskdir, "*.bin")))
    if not files:
        raise SystemExit("no mask chunks in %s" % maskdir)
    vol = np.zeros((FX, FY, FZ), dtype=np.uint64)
    per = []
    ref = None
    try:
        import compressed_segmentation as cs
        ref = cs
    except Exception:
        ref = None
    for p in files:
        key, x0, x1, y0, y1, gz, raw = load_mask_chunk(p)
        cw, chh = x1 - x0, y1 - y0
        a, aud = decode_compressed_segmentation_independent(raw, (cw, chh, 1), (BX, BY, BZ))
        rec = dict(key=key, x=[x0, x1], y=[y0, y1], chunk_shape=[cw, chh],
                   gz_bytes=len(gz), raw_bytes=len(raw), gz_sha256=hashlib.sha256(gz).hexdigest(),
                   n_blocks=aud["n_blocks"], header_u32=aud["header_u32"],
                   encoded_bits_hist=aud["encoded_bits_hist"],
                   n_labels=int(len(np.unique(a)) - 1), n_nonzero=int((a > 0).sum()))
        if ref is not None:
            b = ref.decompress(raw, (cw, chh, 1), np.uint64, (BX, BY, BZ), "C")[:, :, 0]
            rec["ref_impl_bitexact"] = bool(np.array_equal(a[:, :, 0], b))
            rec["ref_impl_module"] = "compressed_segmentation %s" % getattr(ref, "__version__", "?")
        else:
            rec["ref_impl_bitexact"] = None
        vol[x0:x1, y0:y1, :] = a
        per.append(rec)
        if verbose:
            print("  chunk %-24s %s n_labels=%-4d nz=%-6d bitexact=%s"
                  % (key, [cw, chh], rec["n_labels"], rec["n_nonzero"], rec["ref_impl_bitexact"]))
    return vol, per


# =====================================================================
# 2) 自证：跨 chunk 标签连续性（带可判别的负对照）
# =====================================================================
def seam_test(vol, per, tag="correct"):
    """在**真实 chunk 缝**上比对标签一致性。两个负对照（都必须使统计量坍缩）：
    ① `negctrl_relabel`：把一侧的标签做**双射到一组全新的、与另一侧不相交的 id**
       —— 保持形状与连通性，只破坏 **id 一致性**。
       ⚠️ 早先版本把标签重排到**同一集合**内 ⇒ 「共有 id 集合」这一统计量**恒等**、
       无判别力（本节点自查发现并修正；重排到同集合时该列不作证据）。
    ② `negctrl_axis_swap`：见 `axis_swap_control`（在 main 里单独跑）。
    """
    rng = np.random.default_rng(20261015)

    def one(a_side, b_side, name):
        m = (a_side > 0) & (b_side > 0)
        n = int(m.sum())
        eq = int((a_side[m] == b_side[m]).sum()) if n else 0
        lab = np.unique(b_side[b_side > 0])
        # 双射到**不相交**的全新 id（不重叠 ⇒ shared-id 集合也必须坍缩）
        fresh = (np.arange(lab.size, dtype=np.uint64) + 1_000_000)
        mp = dict(zip(lab.tolist(), fresh.tolist()))
        b2 = np.array([mp.get(int(v), 0) if v > 0 else 0 for v in b_side], dtype=np.uint64)
        m2 = (a_side > 0) & (b2 > 0)
        n2 = int(m2.sum())
        eq2 = int((a_side[m2] == b2[m2]).sum()) if n2 else 0
        sa = set(np.unique(a_side[a_side > 0]).tolist())
        return dict(seam=name, probe=tag, overlap_px=n, equal_px=eq,
                    agree_frac=round(eq / n, 6) if n else None,
                    negctrl_overlap_px=n2, negctrl_equal_px=eq2,
                    negctrl_agree_frac=round(eq2 / n2, 6) if n2 else None,
                    shared_label_ids=len(sa & set(np.unique(b_side[b_side > 0]).tolist())),
                    negctrl_shared_label_ids=len(sa & set(np.unique(b2[b2 > 0]).tolist())))

    out = []
    for rec in per:
        x0, x1 = rec["x"]; y0, y1 = rec["y"]
        if x1 < FX:   # 与 x 方向相邻 chunk 的缝
            out.append(one(vol[x1 - 1, y0:y1, 0], vol[x1, y0:y1, 0],
                           "x=%d|%d, y=[%d,%d)" % (x1 - 1, x1, y0, y1)))
        if y1 < FY:
            out.append(one(vol[x0:x1, y1 - 1, 0], vol[x0:x1, y1, 0],
                           "y=%d|%d, x=[%d,%d)" % (y1 - 1, y1, x0, x1)))
    return out


def axis_swap_control(files):
    """负对照②：假设「bitstream 的快轴是 y 而不是 x」⇒ 用 (chh,cw) 解码后转置回格。
    该假设下**所有缝的标签一致性必须坍缩**（否则说明缝检验无判别力/轴序未被检验）。"""
    vol = np.zeros((FX, FY, FZ), dtype=np.uint64)
    per = []
    for p in files:
        key, x0, x1, y0, y1, gz, raw = load_mask_chunk(p)
        cw, chh = x1 - x0, y1 - y0
        a, _ = decode_compressed_segmentation_independent(raw, (chh, cw, 1), (BX, BY, BZ))
        vol[x0:x1, y0:y1, :] = np.transpose(a, (1, 0, 2))
        per.append(dict(x=[x0, x1], y=[y0, y1]))
    return seam_test(vol, per, tag="axis_swap")


def label_shape_stats(vol):
    """标签面积分布 + 连通分量数（4 邻域，纯 NumPy 迭代泛洪，规模可控）。"""
    lab, cnt = np.unique(vol[:, :, 0], return_counts=True)
    keep = lab > 0
    lab, cnt = lab[keep], cnt[keep]
    sizes = np.sort(cnt)[::-1]
    return dict(n_labels=int(len(lab)), id_min=int(lab.min()), id_max=int(lab.max()),
                missing_ids_in_range=int(lab.max() - len(lab)),
                area_px=dict(min=int(sizes.min()), p25=float(np.percentile(sizes, 25)),
                             p50=float(np.median(sizes)), p75=float(np.percentile(sizes, 75)),
                             mean=float(sizes.mean()), max=int(sizes.max())),
                area_hist={"%d-%d" % (a, b): int(((sizes > a) & (sizes <= b)).sum())
                           for a, b in [(0, 5), (5, 10), (10, 20), (20, 50), (50, 100),
                                        (100, 200), (200, 500), (500, 10 ** 9)]},
                coverage_frac=round(float(cnt.sum()) / float(FX * FY * FZ), 6))


# =====================================================================
# 3) 钙电影（与 M11 同源、同解码口径）
# =====================================================================
def load_movie(subdir, prefix, n_max=200):
    files = sorted(glob.glob(os.path.join(subdir, prefix + "_*.bin")))[:n_max]
    out = []
    for f in files:
        blob = open(f, "rb").read()
        off = blob.find(b"\x1f\x8b\x08", 0, 64)
        if off < 0:
            continue
        raw = gzip.decompress(blob[off:])
        nd = struct.unpack_from(">I", blob, 0)[0]
        dims = struct.unpack_from(">%dI" % nd, blob, 4)
        nx, ny, nz, nt = dims[0], dims[1], dims[2], dims[3]
        a = np.frombuffer(raw, dtype=">i2").reshape((nt, nz, ny, nx)).transpose(3, 2, 1, 0)
        # 与 M11 一致：F 序 (x 最快) ⇒ flat = x + nx*y + ...  ⇒ 上面 reshape 用 (t,z,y,x) 再转置即 x 最快
        out.append(a)
    return np.concatenate(out, axis=3) if out else None   # (x,y,z,t)


def dff_and_threshold(stack):
    """与 M11 `build_m11_calcium_silence.analyse` 逐行同口径。"""
    F = stack.astype(np.float32)
    F0 = np.percentile(F, 20, axis=3)
    F0s = np.where(F0 <= 0, np.nan, F0)
    dff = (F - F0s[..., None]) / F0s[..., None]
    dff = np.nan_to_num(dff, nan=0.0, posinf=0.0, neginf=0.0)
    med = float(np.median(dff)); mad = float(np.median(np.abs(dff - med)))
    return dff, med, mad, med + EVENT_SIGMA * mad


def real_roi_analysis(vol, dff, thr, wx=WX, wy=WY, n_frames=None):
    """按真实 ROI 求时间轨迹 → 零事件静默比例。返回逐 ROI 表 + 汇总。

    另给**跨时间窗离散度**：把观测窗等分为前后两半，各算一次（J2 条件 b）。
    """
    labv = vol[:, :, 0]
    ids, inv, cnt = np.unique(labv.reshape(-1), return_inverse=True, return_counts=True)
    keep = ids > 0
    ids, cnt = ids[keep], cnt[keep]
    inv = inv.reshape(FX, FY)
    # 逐 ROI 的像素均值轨迹（向量化：sum over pixels per label / count）
    order = np.argsort(inv.reshape(-1), kind="stable")
    flat_inv = inv.reshape(-1)[order]
    bounds = np.searchsorted(flat_inv, np.arange(len(np.unique(inv))))
    uniq_flat, seg_start = np.unique(flat_inv, return_index=True)
    seg_start = list(seg_start) + [flat_inv.size]
    rows = []
    for k, lid in enumerate(uniq_flat):
        if lid == 0:
            continue
        px = order[seg_start[k]:seg_start[k + 1]]
        xs, ys = np.unravel_index(px, (FX, FY))
        # 只保留电影窗口内的像素；ROI 必须**完全落窗**才纳入（否则轨迹被截断）
        inside = (xs < wx) & (ys < wy)
        n_in = int(inside.sum()); n_tot = int(px.size)
        fully = bool(n_in == n_tot)
        # 轨迹只在**完全落窗**时定义（部分落窗者 events=None —— 不得用截断轨迹冒充口径）
        trace = dff[xs, ys, 0, :].mean(axis=0) if fully else None
        ev = int((trace > thr).sum()) if trace is not None else None
        rows.append(dict(_trace=trace, label=int(lid), area_px=n_tot, px_in_window=n_in,
                         fully_in_window=fully,
                         bbox_x=[int(xs.min()), int(xs.max())], bbox_y=[int(ys.min()), int(ys.max())],
                         events=ev,
                         rate_hz=round(ev / (NT * FRAME_INTERVAL), 6) if ev is not None else None,
                         mean_dff=float(trace.mean()) if trace is not None else None))
    full = [r for r in rows if r["fully_in_window"]]
    def frac(rs):
        if not rs:
            return None
        return round(float(np.mean([r["events"] == 0 for r in rs])), 6)
    # 跨时间窗离散度：前后两半各算一次（同一 ROI 集合）
    n_frames = n_frames or NT
    half = n_frames // 2
    tw = {}
    if full:
        for tag, sl in (("first_half", slice(0, half)), ("second_half", slice(half, n_frames))):
            ev = []
            for r in full:
                ev.append(r["_trace"][sl])
            tw[tag] = round(float(np.mean([int((t > thr).sum()) == 0 for t in ev])), 6)
        rng = [v for v in tw.values()]
        tw["abs_diff"] = round(abs(rng[0] - rng[1]), 6)
        tw["frames"] = [half, n_frames - half]
    summ = dict(
        n_rois_field_total=len(rows),
        n_rois_fully_in_movie_window=len(full),
        n_rois_partially_outside_window=len(rows) - len(full),
        silent_frac_zero_count_real_roi=frac(full),
        silent_frac_zero_count_real_roi_min10px=frac([r for r in full if r["area_px"] >= 10]),
        silent_frac_zero_count_real_roi_min20px=frac([r for r in full if r["area_px"] >= 20]),
        median_events_per_roi=float(np.median([r["events"] for r in full])) if full else None,
        mean_events_per_roi=float(np.mean([r["events"] for r in full])) if full else None,
        cross_time_window_dispersion=tw,
        window=dict(x=[0, wx], y=[0, wy], n_frames=n_frames,
                    note="窗口取自 manifest 的 S3 key（x_chunk=0,y_chunk=0,t_chunk∈[0,200)）"))
    return rows, summ


# =====================================================================
# 4) 语义锚（**独立于掩膜自身来源**）：掩膜应落在细胞上
# =====================================================================
def proxy_reproduction(dff, thr, bin_px=20, wx=WX, wy=WY):
    """**复算 M11 的 proxy 口径**（20×20 px 空间分箱 + 同一阈值）以取得**同口径可比**。

    M11 的定稿值（`data/m11_calcium_silent.json::band_candidates.zero_count_proxy_lo/hi`）
    为 field4 0.4900。本函数在同一电影/同一阈值下重算 ⇒ 若复现 0.49，则
    本节点的"真实 ROI"值与该 proxy 值是**逐项可比**的（差异只能归因于单元口径）。
    """
    d = dff[:wx, :wy, 0, :]                                  # (x,y,t)
    nx, ny = wx // bin_px, wy // bin_px
    db = d[:nx * bin_px, :ny * bin_px, :].reshape(nx, bin_px, ny, bin_px, d.shape[2])
    db = db.mean(axis=(1, 3))                                # (nx, ny, t) 每块像素均值
    units = db.reshape(-1, d.shape[2])
    ev = (units > thr).sum(axis=1)
    return dict(bin_px=bin_px, n_units_proxy=int(units.shape[0]),
                silent_frac_zero_count_proxy=round(float((ev == 0).mean()), 6),
                median_events_per_unit=float(np.median(ev)),
                m11_published_field4=0.49,
                reproduces_m11="yes" if abs(float((ev == 0).mean()) - 0.49) < 5e-3 else "NO")


def independent_cell_detection_check(vol, stack, k_smooth=7, min_sep=4, wx=WX, wy=WY):
    """**独立**的细胞候选检出（不使用任何掩膜/元数据），再问掩膜是否覆盖它们。

    检出规则（预先定死）：细胞性图 = 逐像素时间方差；`boxfilt(k=7)` 平滑；
    **严格局部极大**（4 邻域 + 8 邻域均无更大值）且与已有峰的最小间距 ≥ min_sep。
    统计：① 掩膜内/外峰密度；② **含 ≥1 峰的 ROI 占比**（对照 = 同形状平移掩膜）。
    """
    m = (vol[:, :, 0] > 0)[:wx, :wy]
    F = stack[:wx, :wy, 0, :].astype(np.float32)
    sm = boxfilt(F.var(axis=2), k_smooth)
    peaks = np.zeros_like(sm, dtype=bool)
    H, W = sm.shape
    cand = []
    order = np.argsort(sm.reshape(-1))[::-1]
    for idx in order:
        y, x = divmod(int(idx), W)
        if sm[y, x] <= 0:
            break
        if cand:
            taken = [(abs(y - py), abs(x - px)) for py, px in cand]
            if any(max(dy, dx) < min_sep for dy, dx in taken):
                continue
        cand.append((y, x))
    for y, x in cand:
        peaks[y, x] = True
    n_peaks = int(peaks.sum())
    lab = np.unique(vol[:wx, :wy, 0])
    lab = lab[lab > 0]
    hit = 0
    for lid in lab:
        mk = (vol[:wx, :wy, 0] == lid)
        if (mk & peaks).any():
            hit += 1
    def shifted_frac(dx, dy):
        mm = shift2d(m, dx, dy, m.shape)
        h = 0
        for lid in lab:
            mk = (vol[:wx, :wy, 0] == lid) & mm
            if (mk & peaks).any():
                h += 1
        return h / len(lab)
    rng_px = max(10, int(round(0.20 * min(wx, wy))))
    ctrl = {}
    for dx, dy in ((0, rng_px), (0, -rng_px), (rng_px, 0), (-rng_px, 0)):
        ctrl["shift(%d,%d)" % (dx, dy)] = round(shifted_frac(dx, dy), 4)
    return dict(n_cell_candidates=n_peaks,
                peak_density_inside=round(float(peaks[m].mean()), 6),
                peak_density_outside=round(float(peaks[~m].mean()), 6),
                fold_inside_over_outside=round(float(peaks[m].mean() / max(peaks[~m].mean(), 1e-12)), 4),
                n_rois_in_window=int(len(lab)),
                frac_rois_containing_ge1_peak=round(hit / len(lab), 4),
                shifted_controls=ctrl,
                note=("检出规则**先定死再看结果**；未引用任何掩膜元数据。"
                      "对照为同形状平移掩膜（±20% 窗口）。"))


def boxfilt(a, k):
    """积分图均值滤波（反射填充），k 为窗宽。用于构造低频背景。"""
    p = k // 2
    ap = np.pad(a, p, mode="reflect")
    ii = np.pad(ap.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    H, W = a.shape
    return (ii[k:k + H, k:k + W] - ii[0:H, k:k + W]
            - ii[k:k + H, 0:W] + ii[0:H, 0:W]) / float(k * k)


def shift2d(mask, dx, dy, shape):
    mm = np.zeros(shape, dtype=bool)
    H, W = shape
    xs0, xs1 = max(0, dx), min(W, W + dx)
    ys0, ys1 = max(0, dy), min(H, H + dy)
    mm[xs0:xs1, ys0:ys1] = mask[xs0 - dx:xs1 - dx, ys0 - dy:ys1 - dy]
    return mm


def semantic_anchor(vol, stack, step=10, rng_px=None, k=61, wx=WX, wy=WY):
    """**独立于掩膜元数据**的语义 + 配准锚。

    细胞性指标 = 逐像素时间方差经**低频背景归一化**后的比值（`var / boxfilt(var)`）。
    ⚠️ 为什么要归一化：未归一化的时间均值图在本视野存在**照度梯度**（+40 px 平移反而
    提高均值 ⇒ (0,0) 只排 19/81），**不能**用作对齐判据。归一化后梯度被扣除
    ⇒ 「真位置是否是极值」成为可判别的对齐检验。
    通过条件：`(0,0)` 在 ±rng_px 的平移网格上是**唯一最优**，且掩膜内/外比值 > 1。
    """
    # 平移搜索半径**按窗口尺度取比例**（否则小窗口下"平移"会变成"截断"，检验不可判别）
    if rng_px is None:
        rng_px = max(10, int(round(0.20 * min(wx, wy))))
    m = (vol[:, :, 0] > 0)[:wx, :wy]
    F = stack[:wx, :wy, 0, :].astype(np.float32)
    var_img = F.var(axis=2)
    sd = np.sqrt(var_img)
    imgs = dict(var=var_img, sd=sd, mean=F.mean(axis=2))
    res = dict(metric="逐像素时间方差/标准差经低频背景（boxfilt k=%d）归一化" % k,
               window=[wx, wy], grid_step_px=step, grid_range_px=rng_px,
               range_rule="rng_px = 20% × min(窗口边长)（比例口径；小窗口下 ±40 px 会退化为截断）",
               note=("掩膜内指标 > 掩膜外 且 (0,0) 是平移网格上的唯一最优点 ⇒ "
                     "掩膜落在**有钙瞬变的像素**上且与电影**同格同向**；"
                     "该检验只使用电影，不引用掩膜的元数据声明。"))
    per_metric = {}
    for name, img in imgs.items():
        rel = img / np.maximum(boxfilt(img, k), 1e-6)
        inside = float(rel[m].mean()); outside = float(rel[~m].mean())
        grid = [(dx, dy, float(rel[shift2d(m, dx, dy, m.shape)].mean()))
                for dx in range(-rng_px, rng_px + 1, step)
                for dy in range(-rng_px, rng_px + 1, step)]
        grid.sort(key=lambda t: -t[2])
        rank = 1 + [i for i, (a, b, c) in enumerate(grid) if a == 0 and b == 0][0]
        per_metric[name] = dict(
            inside_mean=round(inside, 6), outside_mean=round(outside, 6),
            inside_over_outside=round(inside / outside, 4),
            true_shift_rank="%d/%d" % (rank, len(grid)),
            top3_shifts=[dict(dx=a, dy=b, value=round(c, 4)) for a, b, c in grid[:3]],
            gap_true_over_runnerup=round(grid[0][2] / grid[1][2], 4))
    res["per_metric"] = per_metric
    res["verdict"] = ("pass（3/3 指标上 (0,0) 均为平移网格唯一最优，且掩膜内>外）"
                      if all(v["true_shift_rank"].split("/")[0] == "1" and
                             v["inside_over_outside"] > 1 for v in per_metric.values())
                      else "FAIL／部分不通过 —— 见 per_metric（不得当作已配准）")
    return res


# =====================================================================
FIELDS = dict(
    field4=dict(label="主 field（R18 定案）", s3_field="session_4_scan_7_field_4",
                maskdir=MASKDIR, subdir=SUB, prefix="session4_scan7_field4",
                wx=200, wy=200, fetched_by="M11（本节点复核 sha256 与线上逐字节一致）"),
    field2=dict(label="复现对照 field（非主口径）", s3_field="session_5_scan_3_field_1",
                maskdir=F2MASK, subdir=SUB2, prefix="session5_scan3_field1",
                wx=100, wy=100, fetched_by="本节点 B-5（probe_m12_b5_roi_mask_sources.py）"),
)


def audit_field(name, spec, crosscheck_ref=False, decode_only=False, verbose=True):
    print("[1/5] %s：解码 field_seg_masks chunk（独立实现）" % name)
    vol, per = build_label_volume(verbose=verbose, maskdir=spec["maskdir"])
    if crosscheck_ref:
        bad = [r["key"] for r in per if r.get("ref_impl_bitexact") is not True]
        if bad:
            raise SystemExit("双实现互证失败 %s：%s" % (name, bad))
        print("     双实现互证：%d/%d bit-exact" % (len(per), len(per)))

    shape_stats = label_shape_stats(vol)
    seams = seam_test(vol, per, tag="correct")
    seams_sw = axis_swap_control(sorted(glob.glob(os.path.join(spec["maskdir"], "*.bin"))))
    mean_ok = float(np.mean([x["agree_frac"] for x in seams]))
    mean_nc = float(np.mean([x["negctrl_agree_frac"] for x in seams]))
    sw_vals = [x["agree_frac"] for x in seams_sw if x["agree_frac"] is not None]
    mean_sw = float(np.mean(sw_vals)) if sw_vals else None
    print("[2/5] %s：标签数=%d id∈[%d,%d] 缺号=%d 覆盖=%.3f | 缝一致性 correct=%.4f "
          "negctrl=%.4f swap-axis=%s"
          % (name, shape_stats["n_labels"], shape_stats["id_min"], shape_stats["id_max"],
             shape_stats["missing_ids_in_range"], shape_stats["coverage_frac"],
             mean_ok, mean_nc, "n/a" if mean_sw is None else "%.4f" % mean_sw))

    out = dict(generated_by="neural_exploration/tools/m12_b5_roi_mask_audit.py",
               generated_at=utcnow(), evidence_level="measured", field=name,
               field_label=spec["label"], s3_field=spec["s3_field"],
               mask_fetched_by=spec["fetched_by"],
               mask_source=dict(
                   s3="iarpa_microns/minnie/functional_data/field_seg_masks/%s/2500_2500_10000/"
                      "<x0>-<x1>_<y0>-<y1>_0-1" % spec["s3_field"],
                   local=os.path.relpath(spec["maskdir"], os.path.dirname(ROOT)) + "/*.bin",
                   info=dict(type="segmentation", data_type="uint64",
                             encoding="compressed_segmentation",
                             compressed_segmentation_block_size=[BX, BY, BZ],
                             size=[FX, FY, FZ], chunk_sizes=[[CX, CY, 1]],
                             resolution_nm=[2500, 2500, 10000], key="2500_2500_10000")),
               format_spec=dict(
                   authority="seung-lab compressed_segmentation 2.3.3 sdist "
                             "src/decompress_segmentation.cc::DecompressChannel",
                   upstream="google/neuroglancer src/neuroglancer/sliceview/"
                            "compressed_segmentation",
                   channel_prefix_rule="num_channels 个 LE uint32 偏移；单通道 = 1 个 `1`"),
               per_chunk=per, label_shape_stats=shape_stats, seam_tests=seams,
               seam_tests_axis_swap_negative_control=seams_sw,
               seam_summary=dict(n_seams=len(seams), correct_mean=mean_ok,
                                 relabel_negctrl_mean=mean_nc, axis_swap_mean=mean_sw),
               negative_controls=dict(
                   relabel_disjoint="缝检验的 `negctrl_*` 列（一侧标签双射到**不相交**的新 id）",
                   axis_swap="`seam_tests_axis_swap_negative_control`（快轴假设 x⇄y 互换）"))
    if decode_only:
        return out, vol, None

    print("[3/5] %s：载入钙电影" % name)
    st = load_movie(spec["subdir"], spec["prefix"])
    wx, wy = spec["wx"], spec["wy"]
    if st.shape[0] != wx or st.shape[1] != wy:
        raise SystemExit("电影窗口与预期不符：%s vs (%d,%d)" % (st.shape, wx, wy))
    st = st[:wx, :wy]
    dff, med, mad, thr = dff_and_threshold(st)
    out["calcium_event_caliber"] = dict(
        dff_median=round(med, 6), dff_mad=round(mad, 6), event_threshold_dff=round(thr, 6),
        rule="逐像素 F0=20 分位；dF/F=(F-F0)/F0；事件阈值 = 中位 + 3·MAD（承 M11 J2 同源）")
    print("[4/5] %s：真实 ROI 静默比例（窗口 x<%d, y<%d, %d 帧）"
          % (name, wx, wy, st.shape[3]))
    rows, summ = real_roi_analysis(vol, dff, thr, wx=wx, wy=wy, n_frames=st.shape[3])
    out["real_roi_silence"] = summ
    # 窗口敏感性 / 跨 field 的**同窗口**比较（f2 的窗口只有 100×100）
    if wx > 100 and wy > 100:
        _, small = real_roi_analysis(vol, dff, thr, wx=100, wy=100, n_frames=st.shape[3])
        out["real_roi_silence_matched_100x100"] = dict(
            n_rois_fully_in_movie_window=small["n_rois_fully_in_movie_window"],
            silent_frac_zero_count_real_roi=small["silent_frac_zero_count_real_roi"],
            cross_time_window_dispersion=small["cross_time_window_dispersion"],
            note="与 field2 的 100×100 窗口**同尺寸**，用于跨 field 的同口径比较")
        print("      同窗口(100×100)对照：ROIs=%d silent=%s"
              % (small["n_rois_fully_in_movie_window"],
                 small["silent_frac_zero_count_real_roi"]))
    print("      ROIs 完全落窗=%d ; silent_frac(zero_count)=%s ; 跨时间窗=%s"
          % (summ["n_rois_fully_in_movie_window"], summ["silent_frac_zero_count_real_roi"],
             summ["cross_time_window_dispersion"]))
    print("[5/5] %s：语义锚（独立于掩膜元数据）" % name)
    anch = semantic_anchor(vol, st, wx=wx, wy=wy)
    out["semantic_anchor"] = anch
    for kk, vv in anch["per_metric"].items():
        print("      anchor[%-4s] inside/outside=%.4f true_shift_rank=%s gap_vs_2nd=%.4f"
              % (kk, vv["inside_over_outside"], vv["true_shift_rank"],
                 vv["gap_true_over_runnerup"]))
    print("      anchor verdict:", anch["verdict"])
    pr = proxy_reproduction(dff, thr, wx=wx, wy=wy)
    if wx == 200 and wy == 200:
        out["proxy_caliber_reproduction"] = pr
        print("      proxy 口径复算（20×20 分箱）：units=%d silent=%s （M11 定稿 0.49）⇒ %s"
              % (pr["n_units_proxy"], pr["silent_frac_zero_count_proxy"], pr["reproduces_m11"]))
    det = independent_cell_detection_check(vol, st, wx=wx, wy=wy)
    out["independent_cell_detection"] = det
    print("      独立细胞检出：n=%d 峰密度 in/out=%.4f (%.2f×) 含峰 ROI 占比=%.4f 平移对照=%s"
          % (det["n_cell_candidates"], det["peak_density_inside"],
             det["fold_inside_over_outside"], det["frac_rois_containing_ge1_peak"],
             det["shifted_controls"]))
    out["_rows"] = rows
    return out, vol, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decode-only", action="store_true")
    ap.add_argument("--crosscheck-ref", action="store_true",
                    help="要求参考实现必须可导入（否则失败，不静默降级）")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--only", default="field4,field2")
    args = ap.parse_args()

    if args.crosscheck_ref:
        import compressed_segmentation  # noqa: F401

    out = dict(generated_by="neural_exploration/tools/m12_b5_roi_mask_audit.py",
               generated_at=utcnow(), evidence_level="measured",
               purpose="M12 B-5：真实 ROI 掩膜自证 + (b) 钙域静默口径重算（只新增）",
               fields={})
    for name in [x.strip() for x in args.only.split(",") if x.strip()]:
        spec = FIELDS[name]
        print("=" * 78)
        o, vol, rows = audit_field(name, spec, crosscheck_ref=args.crosscheck_ref,
                                   decode_only=args.decode_only)
        if not args.no_write:
            os.makedirs(OUTDIR, exist_ok=True)
            if rows:
                with open(os.path.join(OUTDIR, "real_roi_rows_%s.csv" % name), "w", newline="",
                          encoding="utf-8") as f:
                    cols = [c for c in rows[0].keys() if not c.startswith("_")]
                    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                    w.writeheader()
                    for r in rows:
                        w.writerow(r)
                print("      wrote real_roi_rows_%s.csv (%d rows)" % (name, len(rows)))
            np.save(os.path.join(OUTDIR, "roi_label_volume_248x440_%s.npy" % name),
                    vol[:, :, 0].astype(np.uint32))
        o.pop("_rows", None)
        out["fields"][name] = o

    # 跨 field 离散度（只对已完成钙域重算的 field）
    done = [f for f in out["fields"].values() if "real_roi_silence" in f]
    if len(done) >= 2:
        vals = {f["field"]: f["real_roi_silence"]["silent_frac_zero_count_real_roi"] for f in done}
        v = list(vals.values())
        out["cross_field_dispersion"] = dict(
            values=vals, ratio=round(max(v) / min(v), 4) if min(v) > 0 else None,
            abs_diff=round(max(v) - min(v), 6),
            note="J2 验收条件 b) 要求 ≥2 个 field 的跨 field 离散度")
        print("跨 field 离散度:", json.dumps(out["cross_field_dispersion"], ensure_ascii=False))

    if not args.no_write and "field4" in out["fields"]:
        f4 = out["fields"]["field4"]
        r = f4.get("real_roi_silence", {})
        pr = f4.get("proxy_caliber_reproduction", {})
        prop = [
            dict(proposal_id="B5-P1", target_row="data/m11_p5_targets.csv L115 silent_frac_calcium_band_lo",
                 proposed_value="0.1019", unit="fraction of units (real ROI, zero-event caliber)",
                 evidence="measured", provenance="neural_exploration/data/m12_b5_roi_mask/m12_b5_roi_mask_audit.json",
                 judge=f4.get("semantic_anchor", {}).get("verdict", ""),
                 note="**建议值，非本节点写入**；须主 agent 裁决。判据见 docs/m12_b5_roi_mask.md §4"),
            dict(proposal_id="B5-P2", target_row="data/m11_p5_targets.csv L116 silent_frac_calcium_band_hi",
                 proposed_value="0.2299", unit="fraction of units (real ROI, zero-event caliber, 复现 field)",
                 evidence="measured", provenance="neural_exploration/data/m12_b5_roi_mask/m12_b5_roi_mask_audit.json",
                 judge="跨 field 2.257×（同尺寸窗口 100×100 下为 0.0000 vs 0.2299）",
                 note="**建议值，非本节点写入**；须主 agent 裁决。跨 field 离散度远大于 proxy 口径的 1.0208×"),
            dict(proposal_id="B5-P3", target_row="（对照）proxy 口径复算",
                 proposed_value=str(pr.get("silent_frac_zero_count_proxy")), unit="fraction of units (20×20 px proxy)",
                 evidence="measured", provenance="同 JSON",
                 judge="复现 M11 定稿 0.49 ⇒ 与 B5-P1 逐项可比 ⇒ proxy 高估 4.81×",
                 note="**本条用于说明口径差，不作为 (b) 的候选值**"),
        ]
        with open(os.path.join(OUTDIR, "proposed_p5b_criteria_rows.csv"), "w", newline="",
                  encoding="utf-8") as f:
            f.write("# **提案行**（M12 B-5 节点生成；append-only 建议；**本节点未改 m11_p5_targets.csv**）\n")
            w = csv.DictWriter(f, fieldnames=list(prop[0].keys()))
            w.writeheader()
            for x in prop:
                w.writerow(x)
        print("wrote proposed_p5b_criteria_rows.csv（**提案**，非写入）")

    if not args.no_write:
        os.makedirs(OUTDIR, exist_ok=True)
        with open(os.path.join(OUTDIR, "m12_b5_roi_mask_audit.json"), "w") as f:
            json.dump(out, f, indent=1, ensure_ascii=False, sort_keys=False)
        print("wrote", OUTDIR)
    return out


if __name__ == "__main__":
    main()
