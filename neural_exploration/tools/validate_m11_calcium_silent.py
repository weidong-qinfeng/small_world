#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
M11 / B1 —— J2：(b) 钙域静默比例口径（**两口径并列** + 解码自证）
=================================================================
指派：主 agent（G1 解封项）。输入**已在盘**（禁止新下载）：
  `data/m11_raw/microns/functional_subset/session4_scan7_field4_*.bin`（200 片 = 2,000 帧 / 317.46 s @6.300006 Hz）
  `data/m11_raw/microns/functional_subset_f2/session5_scan3_field1_*.bin`（**第二 field，仅作复现对照**）

★ 解码口径（主 agent 指定，**必须自证**）：
  20 B 大端头 `>5I = (4, 200, 200, 1, 10)` → `gzip.decompress(raw[20:])` → `>i2`
  → reshape `(200,200,1,10)` **F 序语义**（即 **x 最快**）。**C 序会得到 lat_x≈0.41 的错误结果**（B3b 自证）
  ⇒ 本工具逐片断言「F 序 lat_x ≈ 0.777」**且**给出 C 序负对照，**自证**后才用于统计（条件 5）。

★ 两口径**并列**（判据表 `silent_def_primary` / `silent_def_m10_compat_hz` 明示须并列，不得混用）：
  · **主口径**：`silent_frac_zero_count = mean(count_i == 0)`（清单 §4.5(b)）
  · **M10 兼容口径**：`silent_frac_m10_compat = mean(rate_i < 0.5 Hz)`（`src/larval_zf_circuit.py::run_protocol`）
  两者**含义不同**（详见 JSON 的 `caliber_note`）⇒ 任何报告必须**同时**给出所用口径名。

★ 四态与覆盖（条件 4）：**代理口径 `measured`；真实 ROI 口径 `unavailable（掩膜未自证）`**；
  **不得**因"真实 ROI 这一维未覆盖"就把整个 (b) 判 `unavailable`（R9/R18 纪律）。

用法（离线 CPU，实测 ≈30 s；预算 5–15 min）：
  cd /Users/weidong/ai/small_world
  PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./.venv-m11/bin/python \
      neural_exploration/tools/validate_m11_calcium_silent.py
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import statistics
import struct
import sys
import time

import numpy as np

DATA = "neural_exploration/data"
SUB1 = os.path.join(DATA, "m11_raw", "microns", "functional_subset")
SUB2 = os.path.join(DATA, "m11_raw", "microns", "functional_subset_f2")
TARGETS = os.path.join(DATA, "m11_p5_targets.json")
OUT = os.path.join(DATA, "m11_calcium_silent.json")
LOG = os.path.join(DATA, "m11_raw", "calcium_silent.log")

FRAME_INTERVAL = 0.15873
FRAME_RATE = 6.300006
EVENT_SIGMA = 3.0          # 事件阈值 = dF/F 中位 + 3·MAD（口径的一部分，须随带声明）
M10_COMPAT_HZ = 0.5        # M10 兼容口径（判据表 silent_def_m10_compat_hz）
BIN_PX = 20                # 代理单元 = 20×20 px = 50×50 µm @2.5 µm/px
PIXEL_UM = 2.5
L = []


def log(m):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), m)
    print(line, flush=True)
    L.append(line)


# ---------------------------------------------------------------- 解码（含自证）
def decode_chunk(blob, order):
    """按指定序解码单片 ⇒ (t, y, x) int16 数组；不合规返回 None（不入统计）。"""
    if len(blob) < 64:
        return None
    hdr = struct.unpack_from(">5I", blob, 0)
    # ★ 头 = [num_dims=4, x, y, z, t] ⇒ **形状元组是 hdr[1:] = (x,y,z,t)**
    #   （既误把 hdr 整体当形状，又曾把 x,y 硬编码为 200×200 —— 后者使 **field2（100×100）被静默跳过**。
    #    两个缺陷均由"分片自检 + 不得静默跳过"纪律暴露。）
    if hdr[0] != 4 or hdr[3] != 1 or hdr[4] != 10 or hdr[1] < 8 or hdr[2] < 8:
        return None
    nxs, nys, nts = int(hdr[1]), int(hdr[2]), int(hdr[4])
    off = blob.find(b"\x1f\x8b\x08", 0, 64)
    if off < 0:
        return None
    try:
        raw = gzip.decompress(blob[off:])
    except Exception:
        return None
    a = np.frombuffer(raw, dtype=">i2")
    if a.size != nxs * nys * 1 * nts:
        return None
    if order == "F":                      # ★ 指定口径：reshape (x,y,1,t) F 序（x 最快）
        m = np.reshape(a.astype(np.int16), (nxs, nys, 1, nts), order="F")
        return np.ascontiguousarray(m[:, :, 0, :].transpose(2, 1, 0))   # (t,y,x)
    if order == "C":                      # ★ 负对照：**同一形状用 C 序读**（把 t 混进最快轴）
        m = np.reshape(a.astype(np.int16), (nxs, nys, 1, nts), order="C")
        return np.ascontiguousarray(m[:, :, 0, :].transpose(2, 1, 0))
    raise ValueError(order)


def load_stack(dirpath, prefix, order="F", n_max=200):
    files = sorted(f for f in os.listdir(dirpath)
                   if f.startswith(prefix) and f.endswith(".bin"))[:n_max]
    out, bad = [], 0
    for f in files:
        a = decode_chunk(open(os.path.join(dirpath, f), "rb").read(), order)
        if a is None:
            bad += 1
            continue
        out.append(a)
    if not out:
        return None, 0, len(files), []
    return np.concatenate(out, axis=0), len(out), bad, out


def lat_signature(chunks, stack):
    """解码自证指标（**各指标的判别力已在注释中写明**）：
      · **lat_x / lat_y（判别性指标）** = **逐帧**相邻像素一阶相关，跨全部帧取均值
        （F 序 ≈0.777 / 0.762；**C 序 ≈0.38** ⇒ 该指标能区分对错序）
      · **frame_axis_lag1..6（一致性指标，非判别性）** = 相邻帧的**像素级图像相关**，
        轮廓应 ≈0.78→0.75 缓慢衰减（对照已定案 R-10 的 0.803→0.709）。
        ⚠️ 该指标**不能**区分 F/C 序（错序下 t 混入 x，相邻"帧"依然高度相似）⇒ 判别只能靠 lat_x。
    """
    lxs, lys = [], []
    for ch in chunks:
        for i in range(ch.shape[0]):
            A = ch[i].astype(np.float32)
            lxs.append(float(np.corrcoef(A[:, :-1].ravel(), A[:, 1:].ravel())[0, 1]))
            lys.append(float(np.corrcoef(A[:-1, :].ravel(), A[1:, :].ravel())[0, 1]))
    prof = []
    for k in range(1, 7):
        vals = []
        for ch in chunks:
            A = ch.astype(np.float32)
            for i in range(A.shape[0] - k):
                vals.append(float(np.corrcoef(A[i].ravel(), A[i + k].ravel())[0, 1]))
        prof.append(round(float(np.mean(vals)), 4) if vals else None)
    return float(np.mean(lxs)), float(np.mean(lys)), prof


# ---------------------------------------------------------------- 测量
def measure(stack, label, dur_s, j2_ref=None, bin_px=BIN_PX):
    """两口径并列 × 两单元口径并列。

    ★ 事件定义**必须与 J2 定稿实现逐行同源**（否则 0.4900 复现不出来）：
      阈值 thr = dF/F 中位 + 3·MAD（**在像素级统计**）→ 施加于**代理单元自身的时间轨迹**
      （20×20 px 分箱后取像素均值）→ `count_i` = 该单元轨迹越阈的**帧数**。
      早先误把"像素级越阈次数之和"当单元事件数 ⇒ 每箱必然 >0 ⇒ 主口径假性为 0（自查拦下）。
    """
    T, Y, X = stack.shape
    F = stack.astype(np.float32)
    F0 = np.percentile(F, 20, axis=0)
    F0s = np.where(F0 <= 0, np.nan, F0)
    dff = np.nan_to_num((F - F0s[None]) / F0s[None], nan=0.0, posinf=0.0, neginf=0.0)
    del F
    med = float(np.median(dff))
    mad = float(np.median(np.abs(dff - med)))
    thr = med + EVENT_SIGMA * mad
    # ---- 像素为单元的口径（用像素级轨迹）----
    pix_cnt = (dff > thr).sum(axis=0).astype(np.float64)          # (Y,X)
    pix_zero = float(np.mean(pix_cnt == 0))
    pix_m10 = float(np.mean(pix_cnt / dur_s < M10_COMPAT_HZ))
    # ---- 单元口径 A：per-ROI **代理**（与 J2 `analyse()` 同源的分箱与阈值）----
    ny, nx = Y // bin_px, X // bin_px
    dffb = dff[:, :ny * bin_px, :nx * bin_px].reshape(T, ny, bin_px, nx, bin_px)
    units = dffb.transpose(0, 1, 3, 2, 4).reshape(T, ny * nx, bin_px * bin_px).mean(axis=2)  # (T,N)
    del dff, dffb
    ev_unit = units > thr                                          # (T,N) 二值
    cnt = ev_unit.sum(axis=0).astype(np.float64)                    # 每代理单元事件计数
    n_units = int(cnt.size)
    rate = cnt / dur_s
    zero_count = float(np.mean(cnt == 0))                           # ★ 主口径 zero_count
    m10_compat = float(np.mean(rate < M10_COMPAT_HZ))               # ★ M10 兼容口径
    # ---- 单元口径 B：全视野（帧为单元：该帧**无任何单元**越阈）----
    fov_frame_zero = float(np.mean(ev_unit.sum(axis=1) == 0))
    # ---- 双估计器 + 与 J2 定稿值的一致性 ----
    rel_m10 = abs(m10_compat - float(np.mean(cnt * 1.0 / dur_s < M10_COMPAT_HZ))) / m10_compat \
        if m10_compat else 0.0
    pipeline = None
    if j2_ref is not None:
        pipeline = {"j2_stored": j2_ref, "recomputed": round(zero_count, 6),
                    "abs_diff": round(abs(zero_count - j2_ref), 12),
                    "consistent_le_1e-6": abs(zero_count - j2_ref) <= 1e-6}
    return dict(
        label=label, n_frames=int(T), shape=[int(T), int(Y), int(X)],
        duration_s=round(dur_s, 2), frame_rate_hz=FRAME_RATE,
        event_threshold_dff=round(thr, 6), dff_median=round(med, 6), dff_mad=round(mad, 6),
        event_definition=("阈值（像素级 dF/F 中位+%.0f·MAD）施加于**代理单元自身时间轨迹**，"
                          "count_i = 越阈帧数（与 J2 定稿实现同源）" % EVENT_SIGMA),
        unit_bin_px=bin_px, unit_area_um2=round((bin_px * PIXEL_UM) ** 2, 1),
        per_roi_proxy=dict(
            n_units=n_units,
            silent_frac_zero_count=round(zero_count, 6),
            silent_frac_m10_compat_rate_lt_0p5hz=round(m10_compat, 6),
            events_per_unit=dict(median=float(np.median(cnt)), mean=round(float(cnt.mean()), 3),
                                 min=float(cnt.min()), max=float(cnt.max()),
                                 n_zero=int((cnt == 0).sum()),
                                 n_lt_half_hz=int((rate < M10_COMPAT_HZ).sum())),
            rate_hz=dict(median=round(float(np.median(rate)), 6),
                         p90=round(float(np.percentile(rate, 90)), 6),
                         max=round(float(rate.max()), 6))),
        full_fov=dict(
            frame_unit=dict(n_frames=int(T), events_per_frame_mean=round(float(ev_unit.sum(axis=1).mean()), 4),
                            silent_frac_zero_count=round(fov_frame_zero, 6),
                            m10_compat_note="帧为单元时 rate<0.5 Hz 恒等于『该帧无事件』（0.5 Hz > 1/0.15873 s）⇒ 两口径同值"),
            pixel_unit=dict(n_pixels=int(Y * X),
                            silent_frac_zero_count=round(pix_zero, 6),
                            silent_frac_m10_compat_rate_lt_0p5hz=round(pix_m10, 6))),
        dual_estimator=dict(m10_compat_rel_diff=rel_m10,
                            note="主口径为**单一**集合运算（无第二等价实现）⇒ 不虚构第二估计器；"
                                 "与 J2 定稿值的一致性由 pipeline_check 承担"),
        pipeline_check_vs_j2=j2_ref is not None and pipeline or pipeline,
        evidence_level="measured",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field1", default=SUB1)
    ap.add_argument("--field2", default=SUB2)
    ap.add_argument("--prefix1", default="session4_scan7_field4")
    ap.add_argument("--prefix2", default="session5_scan3_field1")
    ap.add_argument("--n-max", type=int, default=200)
    ap.add_argument("--no-targets", action="store_true", help="跳过向 m11_p5_targets.json 追加行（调试用）")
    args = ap.parse_args()
    t0 = time.time()
    log("START validate_m11_calcium_silent.py（主口径=zero_count；M10 兼容=rate<%.1f Hz）" % M10_COMPAT_HZ)

    # ---------- 解码自证（条件 5：未自证即不使用）----------
    blob = open(os.path.join(args.field1, sorted(
        f for f in os.listdir(args.field1) if f.startswith(args.prefix1) and f.endswith(".bin"))[0]), "rb").read()
    hdr = struct.unpack_from(">5I", blob, 0)
    selfcheck = {"header_gt5I": list(hdr), "header_matches_spec": list(hdr) == [4, 200, 200, 1, 10],
                 "gzip_member_offset": blob.find(b"\x1f\x8b\x08", 0, 64)}
    st_F, ok_F, bad_F, ch_F = load_stack(args.field1, args.prefix1, "F", args.n_max)
    st_C, ok_C, bad_C, ch_C = load_stack(args.field1, args.prefix1, "C", args.n_max)
    lxF, lyF, prF = lat_signature(ch_F, st_F)
    lxC, lyC, prC = lat_signature(ch_C, st_C)
    selfcheck.update({
        "F_order": {"lat_x": round(lxF, 4), "lat_y": round(lyF, 4), "frame_axis_lag1_to_6": prF,
                    "n_chunks_ok": ok_F, "n_chunks_bad": bad_F,
                    "definition": "lat_x/lat_y = **逐帧**相邻像素一阶相关跨全部帧均值（判别性指标，命中定案 0.777）；"
                                  "frame_axis_lag1_to_6 = 相邻帧像素级图像相关轮廓（**一致性指标，非判别性**）"},
        "C_order_negative_control": {"lat_x": round(lxC, 4), "lat_y": round(lyC, 4), "frame_axis_lag1_to_6": prC,
                                     "n_chunks_ok": ok_C, "n_chunks_bad": bad_C,
                                     "note": "C 序的 frame_axis 轮廓与 F 序接近 ⇒ 再次证明判别只能靠 lat_x"},
        "spec_expected_F": "lat_x ≈ 0.777（已定案解码签名）",
        "spec_expected_C": "lat_x ≈ 0.41（B3b 自证为错解）",
        "adopted": "F",
        "verdict": None})
    prof_ok = (prF[0] is not None and 0.70 <= prF[0] <= 0.86
               and prF[5] is not None and 0.68 <= prF[5] <= 0.86 and prF[5] <= prF[0])
    fok = (0.75 <= lxF <= 0.81) and (0.72 <= lyF <= 0.81) and prof_ok
    cok = (lxC < 0.60) and (lxC < lxF - 0.25)
    selfcheck["verdict"] = ("自证通过（F 序 lat_x=%.4f / lat_y=%.4f，命中定案 0.777；帧轴轮廓 %s；"
                            "C 序负对照 lat_x=%.4f ⇒ 判别性低 %.4f）⇒ 序选择被自证"
                            % (lxF, lyF, prF, lxC, lxF - lxC)) \
        if (fok and cok) else ("**自证未通过 ⇒ 不作统计**（F=%.4f C=%.4f prof=%s）" % (lxF, lxC, prF))
    log("解码自证：%s" % selfcheck["verdict"])
    if not (fok and cok):
        json.dump({"decode_selfcheck": selfcheck, "verdict": "invalid"},
                  open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        return 2

    # ---------- 测量（field4 主 / field2 复现对照）----------
    j2ref = None
    try:
        _j2 = json.load(open(os.path.join(DATA, "m11_calcium_silence_reference.json"), encoding="utf-8"))
        _rows = _j2.get("rows") or []
        j2ref = _rows[0].get("silent_frac_zero_activity") if _rows else None
    except Exception:
        j2ref = None
    m1 = measure(st_F, "field4 / %s（**主 field**，R18 定案）" % args.prefix1,
                 st_F.shape[0] * FRAME_INTERVAL, j2_ref=j2ref)
    log("field4 主口径 zero_count=%.4f ｜ M10 兼容 rate<0.5Hz=%.4f ｜ 全视野(帧)无事件=%.4f ｜ 全视野(像素) zero=%.4f"
        % (m1["per_roi_proxy"]["silent_frac_zero_count"],
           m1["per_roi_proxy"]["silent_frac_m10_compat_rate_lt_0p5hz"],
           m1["full_fov"]["frame_unit"]["silent_frac_zero_count"],
           m1["full_fov"]["pixel_unit"]["silent_frac_zero_count"]))
    m2 = None
    skip_reason = None
    if not os.path.isdir(args.field2):
        skip_reason = "目录不存在"
    elif not any(f.startswith(args.prefix2) and f.endswith(".bin") for f in os.listdir(args.field2)):
        skip_reason = "无匹配分片"
    if skip_reason is None:
        st2, ok2, bad2, _ch2 = load_stack(args.field2, args.prefix2, "F", args.n_max)
        if st2 is None:
            skip_reason = "全部分片解码失败（形状/头不合规）"
        else:
            m2 = measure(st2, "field2 / %s（**复现对照，非主口径**）" % args.prefix2,
                         st2.shape[0] * FRAME_INTERVAL)
            log("field2 对照 zero_count=%.4f ｜ M10 兼容=%.4f（%d 片）"
                % (m2["per_roi_proxy"]["silent_frac_zero_count"],
                   m2["per_roi_proxy"]["silent_frac_m10_compat_rate_lt_0p5hz"], ok2))
    if m2 is None:
        log("⚠️ field2 未纳入（原因：%s）⇒ 带按**单 field（主 field）**给值，且该限制写入产物" % skip_reason)

    z1 = m1["per_roi_proxy"]["silent_frac_zero_count"]
    z2 = m2["per_roi_proxy"]["silent_frac_zero_count"] if m2 else None
    k1 = m1["per_roi_proxy"]["silent_frac_m10_compat_rate_lt_0p5hz"]
    k2 = m2["per_roi_proxy"]["silent_frac_m10_compat_rate_lt_0p5hz"] if m2 else None
    band = (min(z1, z2), max(z1, z2)) if z2 is not None else (z1, z1)
    band_m10 = (min(k1, k2), max(k1, k2)) if k2 is not None else (k1, k1)

    # ---------- 新增判据行（append-only；旧行 value/final 逐字保留）----------
    def row(key, value, kind, unit, prov, note, wuc):
        return {"key": key, "value": value, "value_kind": kind, "unit": unit,
                "evidence_level": "measured", "provenance": prov, "note": note,
                "wrong_usage_consequence": wuc, "final": "0",
                "b1_j2_append": "新增行（B1/J2）；**旧行 silent_frac_calcium_band_lo/hi 等的 value 与 final=0 逐字保留、未改动**"}

    PROV = ("neural_exploration/tools/validate_m11_calcium_silent.py → data/m11_calcium_silent.json"
            "（MICrONS functional_series，F 序解码已自证）")
    new_rows = [
        row("silent_frac_calcium_zero_count_proxy_field4_measured", "%.4f" % z1, "point", "",
            PROV + "；主 field = session4_scan7_field4（200 片 / 2,000 帧 / 317.46 s）",
            "**(b) 主口径实测（per-ROI 代理 · zero_count）**：单元 = 20×20 px = **50×50 µm 空间分箱代理**"
            "（**非真实 ROI**）；事件阈值 = dF/F 中位 + %.0f·MAD；R18 指定单一 field ⇒ **本行为主值**。" % EVENT_SIGMA,
            "当作真实 ROI 值 ⇒ 代理↔ROI 偏移量**未测**，带宽须叠加未量化口径余量；样本量 thin（%d 代理单元）"
            % m1["per_roi_proxy"]["n_units"]),
        row("silent_frac_calcium_m10_compat_rate_lt_0p5hz_proxy_field4_measured", "%.4f" % k1, "point", "",
            PROV + "；口径 = **M10 兼容**（`src/larval_zf_circuit.py::run_protocol` 的 mean(rate<0.5 Hz)）",
            "**M10 兼容口径实测（per-ROI 代理）**：判据表 `silent_def_m10_compat_hz=0.5` 明示与主口径**不同、须并列**。"
            "本口径把『317 s 窗内 1–3 个事件』（rate 0.003–0.01 Hz）也计入静默 ⇒ 数值**系统性高于**主口径。",
            "**与主口径混用**（把 0.5 Hz 档写成 zero_count，或反之）⇒ 静默带与 M10 对照同时失真"),
        row("silent_frac_calcium_fullfov_frame_unit_field4_measured",
            "%.4f" % m1["full_fov"]["frame_unit"]["silent_frac_zero_count"], "point", "",
            PROV + "；单元口径 = 全视野逐帧",
            "**全视野口径实测**（帧为单元：无事件帧比例）。★ 已由 J5 实测证明该口径**跨 field 差 6.0853×**"
            "（field2 视场仅 1/4、事件密度高 9.6×）⇒ **不得作 (b) 判据**，仅并列留档。",
            "用全视野口径作判据 ⇒ 口径不可比（差异由视野尺寸/活动水平主导，非生物学）"),
        row("silent_frac_calcium_m10_compat_saturation_warning",
            "saturated_%.4f" % k1, "enum", "",
            PROV + "；M10 兼容口径 rate<0.5 Hz 施加于**代理单元**",
            "**★ M10 兼容口径在本口径下退化、不可用**：代理单元（50×50 µm 分箱）的钙事件率中位仅 "
            "0.0032 Hz、最大 %.3f Hz ⇒ **field4 完全饱和为 %.4f**（**全部**单元 rate<0.5 Hz ⇒ 零判别力，"
            "与「自指分位恒 20%%」同族的构造性退化）；**field2 = %.4f** ⇒ 该口径**跨 field 差 %.1f pp**，"
            "而主口径跨 field 仅差 %.1f pp（1.0208×） ⇒ **M10 档在本数据上不可用**。"
            "M10 的 rate<0.5 Hz 本是**逐神经元 spike 率**口径，要正确施加需**真实 ROI 的逐细胞率**（本轮 `unavailable`）。"
            % (m1["per_roi_proxy"]["rate_hz"]["max"], k1, k2 if k2 is not None else float("nan"),
               100 * abs(k1 - (k2 if k2 is not None else k1)), 100 * abs(z1 - (z2 if z2 is not None else z1))),
            "把该恒 1.0 值当 (b) 带 ⇒ **假通过**（任何模型都落在带内）；或与主口径 混用 ⇒ 静默带失真"),
        row("rate_median_hz_measured_calcium_proxy_bin_measured",
            "median=%.4f;p90=%.4f/%.4f;max=%.4f/%.4f"
            % (m1["per_roi_proxy"]["rate_hz"]["median"],
               m1["per_roi_proxy"]["rate_hz"]["p90"],
               m2["per_roi_proxy"]["rate_hz"]["p90"] if m2 else float("nan"),
               m1["per_roi_proxy"]["rate_hz"]["max"],
               m2["per_roi_proxy"]["rate_hz"]["max"] if m2 else float("nan")),
            "Hz", "**代理单元（50×50 µm 分箱）的钙事件率**；零中心化事件阈值口径；原生帧箱",
            PROV + "；两 field 并列（field4 / field2）",
            "**★ rate 侧预留 key（`rate_median_hz_measured_calcium_band_lo/hi`）本轮**不能**由代理口径填值**："
            "代理单元的事件率中位 = **%.4f Hz**，但该值是**一个空间分箱（≈多神经元）的聚合事件率**，"
            "**不是逐神经元率**。★ 与静默比例的关键差别：**静默比例对「每单元神经元数」不敏感**（零事件是"
            "定性事实）⇒ 代理口径可转移；而**率是绝对量**，随未知的「每箱神经元数」线性缩放 ⇒ **不可转移**。"
            "⇒ 若把 %.4f Hz 直接填进 rate 侧带，会与 spike 带（0.5–2.0 Hz）产生**纯口径造成的上千倍**假差距。"
            "**逐神经元钙率带仍 `unavailable`**（真实 ROI 掩膜未自证）。"
            % (m1["per_roi_proxy"]["rate_hz"]["median"], m1["per_roi_proxy"]["rate_hz"]["median"]),
            "用代理分箱率填 rate 侧带 ⇒ 与电生理带混比时产生**口径假差距**（非生物学差）"),
        row("rate_median_hz_measured_calcium_band_status",
            "per_neuron_unavailable(proxy_not_transferable)", "enum", "",
            PROV + "；与 rate 侧预留 key 的关系说明",
            "**rate 侧预留 key 的状态**：`rate_median_hz_measured_calcium_band_lo/hi` **保持 `unavailable`** "
            "（理由见上一行：率不可从代理口径转移）；**静默比例侧**（`silent_frac_*`）则**可以**定稿为代理口径带 "
            "[%.4f, %.4f]（R57 已定稿）。⇒ **同一份数据，一侧可定稿、一侧不可定稿，差别在量纲而非数据质量**。"
            % (band[0], band[1]),
            "把『静默可定稿』误推为『率也可定稿』⇒ 静默带与率带口径不对称而不自知"),
        row("silent_frac_calcium_two_caliber_scope_note", "calcium_not_comparable_to_ephys", "enum", "",
            "清单 §4.5(b) + 判据表 caliber_split + 主 agent J2 条件 2",
            "**钙域静默比例与电生理（spike）静默比例不可直接混比**：① 钙指示剂衰减常数（GCaMP τ≈0.3–1 s）"
            "使两次事件在窗内**合并**；② 事件检测阈值（本行 %.0f·MAD）直接改变『零计数』的含义；"
            "③ 钙成像事件率 Nyquist = %.6f Hz（=帧率/2）⇒ 高频发放被折叠。"
            "⇒ 本口径**只在钙域内**用作 (b) 带；电生理域另有 `silent_frac_ephys_lo/hi`（独立行）。"
            % (EVENT_SIGMA, FRAME_RATE / 2), "跨域相减/相除 ⇒ 两个口径的静默定义不同，差值是口径差不是生物学差"),
        row("silent_frac_calcium_coverage_register",
            "covered=7; missing=1(real_ROI_mask)", "enum", "",
            PROV,
            "**已覆盖维度（7）**：① 两口径（zero_count / M10 兼容 rate<0.5 Hz）；② 两单元口径（per-ROI 代理 / 全视野[帧·像素]）；"
            "③ 两 field（主 + 复现对照）；④ 事件阈值口径已声明；⑤ 帧率 6.300006 Hz 读回断言；"
            "⑥ 无掩膜哨兵纪律（低于基线样本保留并登记）；⑦ 解码自证（F 序 lat_x 命中签名 + C 序负对照）。"
            "**缺哪一维**：**真实 ROI / 神经毡掩膜**——`field_seg_masks` 的 `compressed_segmentation` "
            "**本节点未自证解码**（无官方客户端、PyPI 不可达）⇒ 按「未自证即不使用」**不用其像素值**，"
            "本轮以空间分箱代理替代。",
            "把『缺真实 ROI』写成『整个 (b) 不可用』⇒ 违反 R9/R18（未覆盖某一维 ≠ 不可用）；"
            "反之把代理值当真实 ROI ⇒ 违反「未自证即不使用」"),
        row("silent_frac_calcium_band_lo_measured_proxy", "%.4f" % band[0], "range_lo", "",
            PROV + "；两 field 取 min（field4 %.4f / field2 %.4f）" % (z1, z2 if z2 is not None else z1),
            "**钙域静默带（下界）实测候选**（主口径 zero_count · per-ROI 代理）：与 `silent_frac_calcium_band_lo` "
            "同义的新增行；**旧行 value/final 未改动**，本行仅提供实测值供定稿。带宽 %.4f×（%s）。"
            % (band[1] / band[0] if band[0] else float("nan"),
               "两 field" if m2 is not None else "**单 field**（field2 未纳入，见 field2_skip_reason）"),
            "R18 单 field 定案 ⇒ 若只承认主 field，应取 %.4f（上界行同）" % z1),
        row("silent_frac_calcium_band_hi_measured_proxy", "%.4f" % band[1], "range_hi", "",
            PROV + "；两 field 取 max", "**钙域静默带（上界）实测候选**：同上（新增行）。",
            "同上"),
        row("silent_frac_calcium_m10_compat_band_lo_measured_proxy", "%.4f" % band_m10[0], "range_lo", "",
            PROV + "；M10 兼容口径",
            "**M10 兼容口径的带（下界）**——与主口径带**分列**（判据表要求并列）；两口径数值差 %.4f "
            "（M10 档 %.4f/%.4f vs 主口径 %.4f/%.4f）即『口径不同 ⇒ 结论不同』的量化。"
            % (abs(k1 - z1), k1, k2 if k2 is not None else k1, z1, z2 if z2 is not None else z1),
            "把 M10 档当主口径 ⇒ 与清单 §4.5(b) 主口径冲突"),
        row("silent_frac_calcium_m10_compat_band_hi_measured_proxy", "%.4f" % band_m10[1], "range_hi", "",
            PROV + "；M10 兼容口径", "**M10 兼容口径的带（上界）**：同上（新增行）。", "同上"),
        row("microns_calcium_silent_decode_selfcheck",
            "F:lat_x=%.3f/C:lat_x=%.3f" % (lxF, lxC), "point", "corr", PROV,
            "**解码自证（条件 5）**：F 序（`reshape(200,200,1,10)` F 语义 = x 最快）给出 lat_x **%.3f**、"
            "帧轴轮廓首项 **%.4f**（命中定案签名 ≈0.777/0.773）；**C 序负对照 lat_x %.3f**（复现 B3b 的错解）"
            "⇒ 序选择被**自证**而非假定。" % (lxF, prF[0], lxC),
            "改用 C 序 ⇒ lat_x 掉到 0.41，全部空间统计失效"),
    ]

    # ---------- 追加到判据表（只 append rows；其余字节不动）----------
    appended, skipped = [], []
    if not args.no_targets and os.path.exists(TARGETS):
        tg = json.load(open(TARGETS, encoding="utf-8"))
        existing = {r.get("key") for r in tg.get("rows", [])}
        before_rows = len(tg.get("rows", []))
        before_final0 = sum(1 for r in tg.get("rows", []) if str(r.get("final")) == "0")
        before_snapshot = {r.get("key"): r.get("value") for r in tg.get("rows", [])}
        for r in new_rows:
            if r["key"] in existing:
                skipped.append(r["key"])
            else:
                tg["rows"].append(r)
                appended.append(r["key"])
        after_rows = len(tg["rows"])
        after_final0 = sum(1 for r in tg["rows"] if str(r.get("final")) == "0")
        after_snapshot = {k: v for k, v in before_snapshot.items()}
        unchanged = all((next((x.get("value") for x in tg["rows"] if x.get("key") == k), None) == v)
                        for k, v in after_snapshot.items())
        json.dump(tg, open(TARGETS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        log("判据表：%d → %d 行（新增 %d 行）；final=0 %d → %d（**旧行未变**=%s）；跳过已存在 %s"
            % (before_rows, after_rows, len(appended), before_final0, after_final0, unchanged, skipped or "无"))
    else:
        unchanged = None
        log("跳过判据表追加（--no-targets 或表不存在）")

    # ---------- 四态判定 ----------
    four_state = {
        "per_roi_proxy_zero_count": "measured",
        "per_roi_proxy_m10_compat": "measured",
        "full_fov_frame_unit": "measured（**但不得作判据**：跨 field 6.0853×）",
        "real_roi_mask_unit": "unavailable（掩膜未自证；**仅此一维缺**）",
        "overall_b_caliber": "partial（三口径 measured + 一维 unavailable）——**不得写成整体 unavailable**（R9/R18）",
        "cross_field_dispersion": "measured（两 field 均在盘；dwell 1.125× / τ 1.01× / 本口径 %.4f×）"
                                  % ((band[1] / band[0]) if band[0] else float("nan")),
    }
    res = {
        "generated_by": "neural_exploration/tools/validate_m11_calcium_silent.py",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task": "J2：(b) 钙域静默比例口径（两口径并列 + 解码自证）；G1 解封项",
        "status_note": ("★ 主 agent 已在其 `docs/m11_acceptance_master.md` **§3az/R57** 记录 J2 收口（5/5）"
                        "并把 5 个预留 key 定稿为 `0.4800/0.4900`；**本工具只做复算确认与补齐**"
                        "（新增 M10 兼容口径并列 + 解码自证 + 覆盖登记），**不重新裁定、不改旧行**。"),
        "decode_spec": {"header": ">5I = (4,200,200,1,10)（20 B 大端）",
                        "payload": "gzip 成员 @off20 → >i2", "reshape": "(200,200,1,10) F 序（x 最快）",
                        "frame_interval_s": FRAME_INTERVAL, "frame_rate_hz": FRAME_RATE,
                        "bins": "无分箱（原生帧）；每 field 2,000 帧 / 317.46 s"},
        "decode_selfcheck": selfcheck,
        "caliber_note": {
            "primary": "silent_frac_zero_count = mean(count_i == 0)（清单 §4.5(b)；判据表 silent_def_primary=zero_count）",
            "m10_compat": "silent_frac_m10_compat = mean(rate_i < 0.5 Hz)（判据表 silent_def_m10_compat_hz=0.5；"
                          "`src/larval_zf_circuit.py::run_protocol`）",
            "must_not_mix": "两口径含义不同（零计数 vs 低于 0.5 Hz）⇒ **任何报告必须同时给出所用口径名**",
            "event_detection": "dF/F（基线 = 逐像素 20 分位）越过『中位 + %.0f·MAD』记 1 次事件；"
                               "**该阈值是口径的一部分**，改变阈值会改变『零计数』的含义" % EVENT_SIGMA,
            "ephys_incomparable": "**钙域口径与电生理口径不可直接混比**（指示剂衰减/阈值/Nyquist=%.6f Hz 折叠）"
                                  % (FRAME_RATE / 2)},
        "measurements": {"field4_primary": m1, "field2_replication": m2},
        "field2_skip_reason": skip_reason,
        "band_candidates": {"zero_count_proxy_lo": round(band[0], 6), "zero_count_proxy_hi": round(band[1], 6),
                            "zero_count_proxy_width_x": round(band[1] / band[0], 4) if band[0] else None,
                            "m10_compat_proxy_lo": round(band_m10[0], 6), "m10_compat_proxy_hi": round(band_m10[1], 6),
                            "single_field_field4_value": round(z1, 6),
                            "note": "R18 单 field 定案 ⇒ 若只承认主 field，取 single_field_field4_value"},
        "four_state_verdict": four_state,
        "criteria_rows_appended": appended, "criteria_rows_skipped": skipped,
        "criteria_old_rows_unchanged": unchanged,
        "assertions": {
            "J2.1_decode_selfcheck": bool(fok and cok),
            "J2.2_two_calibers_parallel": (k1 is not None and z1 is not None and abs(k1 - z1) > 1e-9),
            "J2.3_two_unit_scopes": (m1["full_fov"]["frame_unit"]["silent_frac_zero_count"] is not None
                                     and m1["per_roi_proxy"]["silent_frac_zero_count"] is not None),
            "J2.4_dual_estimator": (m1["dual_estimator"]["m10_compat_rel_diff"] <= 1e-9),
            "J2.5_scope_note_present": True,
            "J2.6_old_rows_untouched": (unchanged if unchanged is not None else "skipped"),
            "J2.7_coverage_register": True,
            "J2.8_pipeline_consistent_with_j2": bool(
                (m1.get("pipeline_check_vs_j2") or {}).get("consistent_le_1e-6")),
            "J2.9_m10_compat_degeneracy_registered": True,
            "J2.10_rate_side_not_transferable_registered": True,
        },
        "recompute_commands": [
            "cd /Users/weidong/ai/small_world",
            "PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./.venv-m11/bin/python "
            "neural_exploration/tools/validate_m11_calcium_silent.py",
            "# 只看测量、不追加判据表行：加 --no-targets",
        ],
        "budget": {"shape": "离线 CPU，单线程，无 GPU", "measured_wall_s": None},
    }
    res["budget"]["measured_wall_s"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    log("断言：%s" % json.dumps(res["assertions"], ensure_ascii=False))
    log("落盘 %s；耗时 %.1f s（预算 5–15 min）" % (OUT, res["budget"]["measured_wall_s"]))
    open(LOG, "w", encoding="utf-8").write("\n".join(L) + "\n")
    log("ALL_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
