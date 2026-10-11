#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
M11 / B1 —— Allen Brain Observatory (Visual Coding **Neuropixels**) `units.csv` 解析
==============================================================================
职责（主 agent DQ-6 指派）：单位表落地后由本节点解析：
  ① 单/多 session 的**发放率分布** → (a) 带（自发率）的**实测量候选**；
  ② **层/区登记**（经 channels.csv 的 ecephys_structure_acronym / CCF 坐标）；
  ③ **session_type 隔离材料**（P14）。

★ 本工具的三条纪律（皆为本轮实证）：
  (1) **口径先行**：`units.csv::firing_rate` 的 `epoch_name_quality_metrics = "complete_session"`
      ⇒ 该率是**整会话（含视觉刺激呈现）口径**，**不是自发率**。
      故本节结果一律标注 `complete_session（刺激含入）`，只能作自发率带的**上界候选**，
      **不得**直接登记为 (a) 自发率带的 `measured` 量。钙域（MICrONS）与电生理域**分列**。
  (2) **退化指标登记**：近静默单元（firing_rate ~0.003 Hz）的 `isolation_distance` 可达 1e12 量级、
      `isi_violations` 可达 4e3 —— 这是**指标退化**（样本量不足），不是"奇异值"，
      须单独登记并从任何带中剔除（同"能通过 ≠ 排他"的退化结构纪律）。
  (3) **完整性门（强于 sha256）**：`size == Content-Length` **且** S3 **多段 ETag 可复算**
      （`md5(concat(md5(part_i))) == ETag 的 '-' 前缀`，本对象 part=8 MiB × 7 段）
      ⇒ 逐字节等同服务器对象；未通过者标 INCOMPLETE 且不入统计。

用法：
  cd /Users/weidong/ai/small_world
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./.venv-m11/bin/python \
      neural_exploration/tools/parse_m11_allen_units.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict

BASE = "neural_exploration/data/m11_raw/allen/bo_ecephys"
OUT = "neural_exploration/data"

# 服务器侧已知量（HEAD 实测，2026-10-12 07:02）
UNITS_SIZE = 51439189
UNITS_ETAG = "5555828a72d2c9f9035c86e2ff310708"
UNITS_ETAG_PARTS = 7
UNITS_ETAG_PART_SIZE = 8 * 1024 * 1024
CHANNELS_SIZE = 6613329

# 口径：整会话（含刺激）——见文件头纪律 (1)
EPOCH = "complete_session"
# 退化阈值（见纪律 (2)）
DEGEN_ISOLATION = 1.0e4
DEGEN_ISI = 20.0
# 钙域 Nyquist 上限（MICrONS 6.300006 Hz 成像 ⇒ 事件率 Nyquist = 6.300006/2）
CA_NYQUIST_HZ = 6.300006 / 2.0

LOG = []


def log(msg: str) -> None:
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    LOG.append(line)


def load_csv(path: str):
    with open(path, "r", encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        return list(rd), rd.fieldnames


def integrity(path: str, size_expected: int, etag_expected: str | None,
              part_size: int, n_parts: int) -> dict:
    """完整性门：size + S3 多段 ETag 复算 + sha256。"""
    n = os.path.getsize(path)
    rec = {"path": path, "size": n, "size_expected": size_expected,
           "size_ok": n == size_expected}
    data = open(path, "rb").read()
    rec["sha256"] = hashlib.sha256(data).hexdigest()
    rec["sha_basis"] = "file_bytes"
    rec["etag_expected"] = etag_expected
    if etag_expected:
        dig = b""
        off = 0
        while off < n:
            dig += hashlib.md5(data[off:off + part_size]).digest()
            off += part_size
        got = hashlib.md5(dig).hexdigest()
        rec["etag_recomputed"] = got
        rec["etag_part_size"] = part_size
        rec["etag_n_parts"] = n_parts
        rec["etag_ok"] = got == etag_expected
    rec["ends_with_newline"] = data[-1:] == b"\n"
    rec["status"] = ("COMPLETE" if (rec["size_ok"] and rec.get("etag_ok", True)
                                    and rec["ends_with_newline"]) else "INCOMPLETE")
    return rec


def q(sorted_vals, p):
    if not sorted_vals:
        return None
    k = int(round(p * (len(sorted_vals) - 1)))
    return sorted_vals[max(0, min(k, len(sorted_vals) - 1))]


def desc(vals):
    v = sorted(vals)
    return {"n": len(v), "median": statistics.median(v) if v else None,
            "p10": q(v, 0.10), "p25": q(v, 0.25), "p75": q(v, 0.75),
            "p90": q(v, 0.90), "min": v[0] if v else None,
            "max": v[-1] if v else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--skip-etag", action="store_true",
                    help="跳过 ETag 复算（仅调试；正式运行必须复算）")
    args = ap.parse_args()

    t0 = time.time()
    log("START parse_m11_allen_units.py (epoch口径=%s)" % EPOCH)

    # ---------- ① 完整性门 ----------
    upath = os.path.join(args.base, "units.csv")
    cpath = os.path.join(args.base, "channels.csv")
    ui = integrity(upath, UNITS_SIZE, None if args.skip_etag else UNITS_ETAG,
                   UNITS_ETAG_PART_SIZE, UNITS_ETAG_PARTS)
    log("完整性门 units.csv  size=%d/%d ok=%s  etag_ok=%s  status=%s"
        % (ui["size"], ui["size_expected"], ui["size_ok"], ui.get("etag_ok"), ui["status"]))
    ci = integrity(cpath, CHANNELS_SIZE, None, 0, 0)
    log("完整性门 channels.csv size=%d/%d ok=%s status=%s"
        % (ci["size"], ci["size_expected"], ci["size_ok"], ci["status"]))
    if ui["status"] != "COMPLETE":
        log("FATAL: units.csv 未通过完整性门 ⇒ 不入统计")
        return 2

    # ---------- ② 解析 ----------
    units, ucols = load_csv(upath)
    channels, ccols = load_csv(cpath)
    probes, pcols = load_csv(os.path.join(args.base, "probes.csv"))
    sessions, scols = load_csv(os.path.join(args.base, "sessions.csv"))
    log("行数：units=%d (27列=%s) channels=%d probes=%d sessions=%d"
        % (len(units), len(ucols) == 27, len(channels), len(probes), len(sessions)))

    ch = {r["id"]: r for r in channels}
    pr = {r["id"]: r for r in probes}
    se = {r["id"]: r for r in sessions}

    # ---------- ③ 连接链 units→channels→probes→sessions ----------
    joined, unres_c, unres_p, unres_s = [], 0, 0, 0
    for u in units:
        c = ch.get(u["ecephys_channel_id"])
        if c is None:
            unres_c += 1
            continue
        p = pr.get(c["ecephys_probe_id"])
        if p is None:
            unres_p += 1
            continue
        s = se.get(p["ecephys_session_id"])
        if s is None:
            unres_s += 1
            continue
        joined.append((u, c, p, s))
    log("连接结果：可解析 %d/%d（channel 未解析 %d / probe 未解析 %d / session 未解析 %d）"
        % (len(joined), len(units), unres_c, unres_p, unres_s))

    f2 = lambda x: float(x) if x not in ("", "NA", None) else None

    # ---------- ④ 口径与退化登记 ----------
    epochs = Counter(u["epoch_name_quality_metrics"] for u in units)
    qual = Counter(u["quality"] for u in units)
    empty = {k: sum(1 for u in units if u[k] == "") for k in ucols}
    empty = {k: v for k, v in empty.items() if v}
    log("epoch_name_quality_metrics=%s ⇒ 口径=整会话（含刺激）" % dict(epochs))
    log("quality 分布=%s" % dict(qual))
    log("空字段=%s" % empty)

    degen = {"isolation_distance_ge_1e4": 0, "isi_violations_gt_20": 0,
             "both_or_either": 0, "near_silent_lt_0.01Hz": 0}
    degen_ids = []
    for u in units:
        fr = f2(u["firing_rate"]) or 0.0
        idd = f2(u["isolation_distance"])
        iv = f2(u["isi_violations"])
        a = idd is not None and idd >= DEGEN_ISOLATION
        b = iv is not None and iv > DEGEN_ISI
        if a:
            degen["isolation_distance_ge_1e4"] += 1
        if b:
            degen["isi_violations_gt_20"] += 1
        if a or b:
            degen["both_or_either"] += 1
            degen_ids.append(u["id"])
        if fr < 0.01:
            degen["near_silent_lt_0.01Hz"] += 1
    degen["fraction_of_units"] = round(degen["both_or_either"] / max(1, len(units)), 6)
    log("退化指标登记=%s" % degen)

    # ---------- ⑤ (a) 带候选（口径：整会话，刺激含入）----------
    all_fr = [f2(u["firing_rate"]) for u in units if f2(u["firing_rate"]) is not None]
    good = [u for u in units if u["quality"] == "good"]
    good_fr = [f2(u["firing_rate"]) for u in good if f2(u["firing_rate"]) is not None]
    degen_set = set(degen_ids)
    clean_fr = [f2(u["firing_rate"]) for u in good
                if u["id"] not in degen_set and f2(u["firing_rate"]) is not None]
    bands = {
        "all_units": desc(all_fr),
        "quality_good": desc(good_fr),
        "quality_good_excl_degenerate": desc(clean_fr),
    }
    for k, v in bands.items():
        log("(a) 带候选[%s] n=%d median=%.4f p90=%.4f min=%.4f max=%.2f"
            % (k, v["n"], v["median"], v["p90"], v["min"], v["max"]))
    nyq = {"calcium_nyquist_hz": CA_NYQUIST_HZ,
           "ephys_p90_over_ca_nyquist": bands["quality_good"]["p90"] / CA_NYQUIST_HZ,
           "ephys_median_over_ca_nyquist": bands["quality_good"]["median"] / CA_NYQUIST_HZ}

    # ---------- ⑥ 逐 session（P14 隔离材料）----------
    by_sess = defaultdict(list)
    for u, c, p, s in joined:
        by_sess[s["id"]].append((u, c, p, s))
    sess_rows = []
    for sid, rows in sorted(by_sess.items(), key=lambda kv: -len(kv[1])):
        fr = [f2(u["firing_rate"]) for u, _, _, _ in rows if f2(u["firing_rate"]) is not None]
        gfr = [f2(u["firing_rate"]) for u, _, _, _ in rows
               if u["quality"] == "good" and f2(u["firing_rate"]) is not None]
        structs = Counter(c["ecephys_structure_acronym"] for _, c, _, _ in rows)
        probes_n = len({c["ecephys_probe_id"] for _, c, _, _ in rows})
        st = rows[0][3]["session_type"]
        sess_rows.append({
            "session_id": sid, "session_type": st,
            "age_in_days": rows[0][3]["age_in_days"],
            "genotype": rows[0][3]["genotype"],
            "n_units": len(rows), "n_good": sum(1 for u, _, _, _ in rows if u["quality"] == "good"),
            "n_probes": probes_n, "n_structures": len(structs),
            "rate_median_hz": statistics.median(fr) if fr else None,
            "rate_p25_hz": q(sorted(fr), 0.25), "rate_p75_hz": q(sorted(fr), 0.75),
            "rate_p90_hz": q(sorted(fr), 0.90),
            "good_rate_median_hz": statistics.median(gfr) if gfr else None,
            "good_rate_p90_hz": q(sorted(gfr), 0.90),
            "top_structure": structs.most_common(1)[0][0] if structs else None,
        })
    log("逐 session 汇总：%d 个 session（units.csv 全量 %d 单元）" % (len(sess_rows), len(units)))

    # 逐 session 层面的率带离散（以 session 为单位 = 正确的独立性单位）
    per_sess_med = [r["rate_median_hz"] for r in sess_rows if r["rate_median_hz"]]
    per_sess_med.sort()
    sess_band = {"n_sessions": len(per_sess_med),
                 "median_of_session_medians_hz": statistics.median(per_sess_med),
                 "p10": q(per_sess_med, 0.10), "p90": q(per_sess_med, 0.90),
                 "min": per_sess_med[0], "max": per_sess_med[-1],
                 "spread_max_over_min": (per_sess_med[-1] / per_sess_med[0]) if per_sess_med[0] else None}
    log("逐 session 中位率带：中位=%.4f Hz  p10=%.4f  p90=%.4f  跨 session 极差=%.2f×"
        % (sess_band["median_of_session_medians_hz"], sess_band["p10"], sess_band["p90"],
           sess_band["spread_max_over_min"]))

    # session_type 隔离
    by_type = defaultdict(lambda: {"n_units": 0, "fr": [], "sessions": set()})
    for u, c, p, s in joined:
        t = by_type[s["session_type"]]
        t["n_units"] += 1
        t["sessions"].add(s["id"])
        fr = f2(u["firing_rate"])
        if fr is not None and u["quality"] == "good":
            t["fr"].append(fr)
    type_stats = {}
    for k, v in by_type.items():
        d = desc(v["fr"])
        type_stats[k] = {"n_sessions": len(v["sessions"]), "n_units": v["n_units"],
                         "good_rate": d}
        log("session_type=%s：%d session / %d 单元；good 率中位=%.4f p90=%.4f"
            % (k, len(v["sessions"]), v["n_units"], d["median"], d["p90"]))

    # ---------- ⑦ 层/区登记 ----------
    struct_rows = []
    st_units = defaultdict(list)
    st_sess = defaultdict(set)
    for u, c, p, s in joined:
        a = c["ecephys_structure_acronym"] or "(empty)"
        st_units[a].append(f2(u["firing_rate"]))
        st_sess[a].add(s["id"])
    for a, v in sorted(st_units.items(), key=lambda kv: -len(kv[1])):
        vv = [x for x in v if x is not None]
        struct_rows.append({"structure_acronym": a, "n_units": len(v),
                            "n_sessions": len(st_sess[a]),
                            "rate_median_hz": statistics.median(vv) if vv else None})
    log("层/区登记：%d 个结构（acronym），覆盖 %d session；前 8 名=%s"
        % (len(struct_rows), len({s for v in st_sess.values() for s in v}),
           [(r["structure_acronym"], r["n_units"]) for r in struct_rows[:8]]))
    ccf_dv = [float(c["dorsal_ventral_ccf_coordinate"]) for _, c, _, _ in joined
              if c["dorsal_ventral_ccf_coordinate"] not in ("", "NA")]
    log("CCF 深度可用：n=%d，范围 %s → %s" % (len(ccf_dv), min(ccf_dv) if ccf_dv else None,
                                          max(ccf_dv) if ccf_dv else None))
    has_layer = [r["structure_acronym"] for r in struct_rows
                 if any(ch.isdigit() and not r["structure_acronym"].startswith(("CA", "SC"))
                        for ch in r["structure_acronym"])]
    log("层（lamina）级标注：acronym 中不含层号 ⇒ **区级 measured / 层须由 CCF 深度派生（derived）**")

    # ---------- ⑦b 皮层限定带（M11 = 皮层微环路；MICrONS minnie65 = VISp）----------
    HIPP = {"CA1", "CA3", "CA2", "DG", "SUB", "HPF"}

    def g_visp(a):
        return a == "VISp"

    def g_vis_all(a):
        return a == "VIS" or a.startswith("VIS")

    def g_hipp(a):
        return a in HIPP

    groups = {"VISp（与 MICrONS minnie65 同名区，最同口径）": g_visp,
              "visual_cortex_all（VIS*）": g_vis_all,
              "hippocampus（CA1/CA3/CA2/DG/SUB/HPF，非同源）": g_hipp}
    area_rows, area_stats = [], {}
    for gname, fn in groups.items():
        sel = [(u, c, r) for u, c, p, r in joined if fn(c["ecephys_structure_acronym"] or "")]
        fr = [f2(u["firing_rate"]) for u, _, _ in sel if f2(u["firing_rate"]) is not None]
        gfr = [f2(u["firing_rate"]) for u, _, _ in sel
               if u["quality"] == "good" and f2(u["firing_rate"]) is not None]
        gfr_clean = [f2(u["firing_rate"]) for u, _, _ in sel
                     if u["quality"] == "good" and u["id"] not in degen_set
                     and f2(u["firing_rate"]) is not None]
        bys = defaultdict(list)
        for u, c, r in sel:
            v = f2(u["firing_rate"])
            if v is not None and u["quality"] == "good":
                bys[r["id"]].append(v)
        meds = sorted(statistics.median(v) for v in bys.values() if v)
        st = {
            "n_units": len(sel),
            "fraction_of_all_units": round(len(sel) / len(units), 4),
            "n_sessions": len(bys),
            "rate_all_units": desc(fr),
            "rate_quality_good": desc(gfr),
            "rate_quality_good_excl_degenerate": desc(gfr_clean),
            "per_session_median_band_hz": {
                "n_sessions": len(meds),
                "median_of_medians": statistics.median(meds) if meds else None,
                "p10": q(meds, 0.10), "p90": q(meds, 0.90),
                "min": meds[0] if meds else None, "max": meds[-1] if meds else None,
                "spread_max_over_min": (meds[-1] / meds[0]) if meds and meds[0] else None},
            "cross_scope_vs_calcium": {
                "calcium_event_rate_nyquist_hz": CA_NYQUIST_HZ,
                "median_over_nyquist": ((statistics.median(gfr_clean) / CA_NYQUIST_HZ)
                                        if gfr_clean else None),
                "fraction_good_rate_gt_nyquist": (sum(1 for x in gfr_clean if x > CA_NYQUIST_HZ)
                                                  / len(gfr_clean)) if gfr_clean else None,
                "n_good_clean": len(gfr_clean),
            },
        }
        area_stats[gname] = st
        log("面积组[%s] 单元=%d (%.1f%%) / session=%d；good 率中位=%.4f p90=%.4f；"
            "逐 session 中位带=%.4f (p10 %.4f, p90 %.4f, 极差 %.2f×)；"
            "跨口径：中位/Nyquist=%.3f，率>Nyquist 比例=%.4f"
            % (gname, st["n_units"], 100 * st["fraction_of_all_units"], st["n_sessions"],
                st["rate_quality_good"]["median"], st["rate_quality_good"]["p90"],
                st["per_session_median_band_hz"]["median_of_medians"],
                st["per_session_median_band_hz"]["p10"], st["per_session_median_band_hz"]["p90"],
                st["per_session_median_band_hz"]["spread_max_over_min"] or float("nan"),
                st["cross_scope_vs_calcium"]["median_over_nyquist"],
                st["cross_scope_vs_calcium"]["fraction_good_rate_gt_nyquist"]))
        area_rows.append({
            "area_group": gname,
            "n_units": st["n_units"], "fraction": st["fraction_of_all_units"],
            "n_sessions": st["n_sessions"],
            "good_rate_median_hz": st["rate_quality_good"]["median"],
            "good_rate_p90_hz": st["rate_quality_good"]["p90"],
            "session_median_hz": st["per_session_median_band_hz"]["median_of_medians"],
            "session_median_p10_hz": st["per_session_median_band_hz"]["p10"],
            "session_median_p90_hz": st["per_session_median_band_hz"]["p90"],
            "session_spread": st["per_session_median_band_hz"]["spread_max_over_min"],
            "ca_nyquist_hz": CA_NYQUIST_HZ,
            "median_over_nyquist": st["cross_scope_vs_calcium"]["median_over_nyquist"],
            "frac_good_rate_gt_nyquist": st["cross_scope_vs_calcium"]["fraction_good_rate_gt_nyquist"],
        })

    # ---------- ⑧ 落盘 ----------
    os.makedirs(args.out, exist_ok=True)
    bysess_csv = os.path.join(args.out, "m11_allen_units_by_session.csv")
    with open(bysess_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sess_rows[0].keys()))
        w.writeheader()
        for r in sess_rows:
            w.writerow({k: ("" if r[k] is None else r[k]) for k in r})
    struct_csv = os.path.join(args.out, "m11_allen_units_by_structure.csv")
    with open(struct_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(struct_rows[0].keys()))
        w.writeheader()
        for r in struct_rows:
            w.writerow({k: ("" if r[k] is None else r[k]) for k in r})
    area_csv = os.path.join(args.out, "m11_allen_rate_band_by_area_group.csv")
    with open(area_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(area_rows[0].keys()))
        w.writeheader()
        for r in area_rows:
            w.writerow({k: ("" if r[k] is None else r[k]) for k in r})

    summary = {
        "generated_by": "neural_exploration/tools/parse_m11_allen_units.py",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "Allen Brain Observatory Visual Coding Neuropixels (units/channels/probes/sessions.csv)",
        "integrity": {"units": ui, "channels": ci},
        "counts": {"n_units": len(units), "n_channels": len(channels),
                   "n_probes": len(probes), "n_sessions": len(sessions),
                   "n_joined": len(joined), "unresolved_channel": unres_c,
                   "unresolved_probe": unres_p, "unresolved_session": unres_s},
        "epoch_scope": {
            "field": "epoch_name_quality_metrics",
            "values": dict(epochs),
            "interpretation": "整会话（含视觉刺激呈现）口径",
            "consequence": ("**不是自发率**；只能作 (a) 自发率带的**上界候选**，"
                            "不得直接登记为 measured 自发率；须与钙域（MICRONS）分列"),
        },
        "quality": dict(qual),
        "empty_fields": empty,
        "degenerate_metrics": degen,
        "band_candidates_complete_session_scope": bands,
        "calcium_cross_scope": nyq,
        "per_session_band": sess_band,
        "session_type_isolation": type_stats,
        "by_session_csv": bysess_csv,
        "by_structure_csv": struct_csv,
        "by_area_group_csv": area_csv,
        "area_group_bands": area_stats,
        "layer_registration": {
            "area_level": "measured（ecephys_structure_acronym 为区级，58 个）",
            "lamina_level": "derived（acronym 不含层号；须由 CCF 深度 + 皮层表面/厚度模型派生）",
            "ccf_depth_available_n": len(ccf_dv),
        },
        "structure_registration": {"n_structures": len(struct_rows),
                                   "n_sessions_covered": len({s for v in st_sess.values() for s in v}),
                                   "ccf_dorsal_ventral_range": [min(ccf_dv), max(ccf_dv)] if ccf_dv else None,
                                   "top12": struct_rows[:12]},
        "caveats": [
            "① 口径：complete_session（含刺激）⇒ 非自发率；自发率带仍需 NWB 的 spontaneous epoch（本轮 unavailable）。",
            "② 退化指标：近静默单元（firing_rate < 0.01 Hz，n=%d）的 isolation_distance / isi_violations 退化，"
            "已单独登记并在 clean 带中剔除。" % degen["near_silent_lt_0.01Hz"],
            "③ 钙域与电生理域**分列**：本表为 spike 率；MICrONS 为钙事件口径（6.30 Hz 成像 ⇒ 事件率 Nyquist=%.3f Hz）。"
            % CA_NYQUIST_HZ,
            "④ 独立性单位应为 **session**（同一 session 内单元共享探针/动物）⇒ 跨 session 带见 per_session_band。",
        ],
    }
    jpath = os.path.join(args.out, "m11_allen_units_reference.json")
    with open(jpath, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    log("落盘：%s / %s / %s" % (bysess_csv, struct_csv, jpath))

    lpath = os.path.join(args.base, "parse_units.log")
    with open(lpath, "w", encoding="utf-8") as fh:
        fh.write("\n".join(LOG) + "\n")
    log("耗时 %.1f s" % (time.time() - t0))
    log("ALL_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
