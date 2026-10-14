#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M11 L3 补充报告器（**只读**） —— 把 L3 正式跑产物汇总成登记要求的四条报告量。

为什么需要它（登记依据）
------------------------------------------------------------------
* ``neural_exploration/docs/m11_l3_resume.md``（L3 交接件：判据带与 (c) 窗口定义）
* ``docs/m11_acceptance_master.md`` §3de(R114) / §3df(R115) / §3di(R118)
* ``data/m11_p5_prereg_amend_r109_a_reporting.json``（(a) 四条并报的**事前修订**）
  ⇒ 四条量均为**描述性并报**，**不改** (a) 的频带与 pass/fail 规则；
  **不得**用 ``a_active_frac`` 代替中位率判 (a)。
* 既有评估器 ``tools/validate_p11_resting.py`` 不提供这四条量（且按「每 config 一个值」设计，
  对杠杆扫描计划有结构性错配，见 R108 §⑤）⇒ 本工具是**补充**报告器。

硬约束（本脚本自证）
------------------------------------------------------------------
1. **只读**：只读 ``--runs`` 下产物；只写 ``--out-csv`` / ``--out-json``。
   若任一输出路径落在 ``--runs`` 目录内 ⇒ **拒绝运行**（非零退出）。
2. **缺件显式**：缺 ``counts.csv`` / 有 ``.fail`` 无 ``.json`` ⇒ 进 ``anomalies`` 清单并计数，
   **绝不静默跳过**。
3. **确定性**：所有集合一律 ``sorted``；输出**不含**时间戳/主机名/pid ⇒ 同输入两次运行逐字节相同。
4. **无子进程**：只用标准库（不依赖 numpy / torch / ``sys.executable``），不 spawn 任何进程。

单位口径（R80 §⑥：跨接口的量必须写明单位）
------------------------------------------------------------------
* ``counts.csv`` 的 ``count`` = **测量窗内**（``T_ms``，**不含 settle**）的逐神经元脉冲数。
  实测核对：``n_steps_meas=200000`` × 0.05 ms/step = 10000 ms = ``T_ms``；
  且逐臂核对 ``sum(counts) == spikes_total``（不一致即进 ``anomalies``）。
* ``T_s = T_ms/1000``（秒）；``a_median_hz = median_i(count_i/T_s)``（**Hz**）。
* ``pooled_*`` 单位 **脉冲数**（整数）；``R`` / ``a_active_frac`` / ``frac_nonzero`` **无量纲**。

命令行
------------------------------------------------------------------
    python3 tools/report_m11_l3.py
    python3 tools/report_m11_l3.py --runs data/m11_p5_l1_runs \
        --out-csv /tmp/x.csv --out-json /tmp/x.json
"""

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys

# ---------------------------------------------------------------- 登记常数
WINDOW_LO = 300.0            # (c) 硬门①：selfsustain_zero_drive_min_spikes
R_MAX = 0.10                 # (c) 上界
A_BAND = (0.5, 2.0)          # (a) 带：逐神经元中位率 Hz
E_BAND = (0.5, 3.0)          # (e) 带：基准臂 i_inh/i_exc 积分比
TRIVIAL_ACTIVE_FRAC = 0.35   # R118 §④4 反平凡性阈值
SUBTYPES = ("E", "PV", "SST", "VIP")
KNOWN_ROLES = ("baseline", "zero_drive")
ROLE_RANK = {"baseline": 0, "zero_drive": 1}

TOOL_VERSION = "1.0"

TRIVIALITY_WARNING = (
    "!! 显著告警（R118 §④4 反平凡性）：c_window_ok=True 且 a_active_frac < %.2f"
    " ⇒ 疑似「用把网络弄死换取 R→0」 ⇒ **该点不得记为工作点命中**。"
    % TRIVIAL_ACTIVE_FRAC
)

MISSING_CLASSES = (
    "fail_marker",
    "arm_json_without_counts",
    "counts_without_arm_json",
    "done_without_arm_json",
    "json_parse_error",
    "counts_unreadable",
    "spikes_total_missing",
    "arm_json_missing_config_or_role",
)


# ---------------------------------------------------------------- 小工具
def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def med(xs):
    """统计中位数（stdlib）。空列表 ⇒ None。"""
    if not xs:
        return None
    return float(statistics.median(xs))


def r6(x):
    """收敛到 6 位小数（确定性输出）。None 原样返回。"""
    if x is None:
        return None
    return round(float(x), 6)


def mean(xs):
    if not xs:
        return None
    return float(sum(xs)) / len(xs)


def is_inside(child, parent):
    child = os.path.abspath(child)
    parent = os.path.abspath(parent)
    return child == parent or child.startswith(parent + os.sep)


# ---------------------------------------------------------------- 输入读取
def read_counts_csv(path):
    """读 counts.csv ⇒ dict；表头按**列名**定位（不靠位置）。"""
    with open(path, "r", newline="") as fh:
        rd = csv.reader(fh)
        try:
            header = next(rd)
        except StopIteration:
            return {"ok": False, "error": "empty counts.csv"}
        idx = {}
        for want in ("neuron_id", "count", "cell_type"):
            if want in header:
                idx[want] = header.index(want)
        if "count" not in idx:
            return {"ok": False, "error": "no 'count' column", "header": header}
        nid_i = idx.get("neuron_id")
        ct_i = idx.get("cell_type")
        neuron_ids, counts, cell_types = [], [], []
        for row in rd:
            if not row:
                continue
            counts.append(int(row[idx["count"]]))
            if nid_i is not None:
                neuron_ids.append(int(row[nid_i]))
            if ct_i is not None:
                cell_types.append(row[ct_i])
    return {
        "ok": True,
        "header": header,
        "neuron_ids": neuron_ids,
        "counts": counts,
        "cell_types": cell_types,
    }


def discover(runs_dir):
    """扫描 runs 目录 ⇒ (arms, anomalies, inventory)。**只看不写**。"""
    anomalies = []
    files = sorted(os.listdir(runs_dir))

    spec_ids, done_ids, fail_ids, counts_ids = set(), set(), {}, set()
    arm_json_paths = []
    for name in files:
        p = os.path.join(runs_dir, name)
        if not os.path.isfile(p):
            continue
        if name.endswith(".spec.json"):
            spec_ids.add(name[: -len(".spec.json")])
        elif name.endswith("_counts.csv"):
            counts_ids.add(name[: -len("_counts.csv")])
        elif name.endswith(".done"):
            done_ids.add(name[: -len(".done")])
        elif name.endswith(".fail"):
            rid = name[: -len(".fail")]
            try:
                with open(p, "r", errors="replace") as fh:
                    head = fh.read(600)
            except OSError as exc:
                head = "unreadable: %s" % exc
            fail_ids[rid] = head
            line = head.splitlines()[0] if head.splitlines() else "(空文件)"
            anomalies.append(
                {
                    "class": "fail_marker",
                    "run_id": rid,
                    "detail": "存在 .fail 标记（该臂无 .json 产物）: %s" % line[:200],
                }
            )
        elif name.endswith(".json"):
            arm_json_paths.append((name[: -len(".json")], p))

    arms = []
    for rid, path in sorted(arm_json_paths):
        rec = {
            "run_id": rid,
            "json_path": os.path.abspath(path),
            "json_sha256": sha256_file(path),
            "json_bytes": os.path.getsize(path),
        }
        try:
            with open(path, "r") as fh:
                doc = json.load(fh)
        except (ValueError, OSError) as exc:
            anomalies.append(
                {"class": "json_parse_error", "run_id": rid, "detail": str(exc)}
            )
            continue
        if not isinstance(doc, dict):
            anomalies.append(
                {
                    "class": "json_not_an_object",
                    "run_id": rid,
                    "detail": "top-level type=%s" % type(doc).__name__,
                }
            )
            continue

        rec["config_id"] = doc.get("config_id")
        rec["role"] = doc.get("role")
        rec["T_ms"] = doc.get("T_ms")
        rec["n_neuron"] = doc.get("n_neuron")
        rec["seed"] = doc.get("seed")
        rec["drive_mult"] = doc.get("drive_mult")

        # ⚠️ 实测键位置：bias 档位**不在顶层**，在 net.* 内
        net = doc.get("net") if isinstance(doc.get("net"), dict) else {}
        rec["bias_scale"] = net.get("bias_scale")
        rec["bias_log_sigma"] = net.get("bias_log_sigma")
        # ⚠️ 实测 net_seed **不在顶层**（在 seeds.net_seed，另见 net.net_seed）
        seeds = doc.get("seeds") if isinstance(doc.get("seeds"), dict) else {}
        rec["net_seed"] = seeds.get("net_seed")
        rec["event_seed"] = doc.get("event_seed")

        # spikes_total：优先顶层；缺则退回 agg.spk_total（显式登记）
        agg = doc.get("agg") if isinstance(doc.get("agg"), dict) else {}
        agg_spk = agg.get("spk_total")
        if isinstance(doc.get("spikes_total"), int):
            rec["spikes_total"] = doc["spikes_total"]
            rec["spikes_total_source"] = "top_level:spikes_total"
        elif isinstance(agg_spk, int):
            rec["spikes_total"] = agg_spk
            rec["spikes_total_source"] = "fallback:agg.spk_total"
            anomalies.append(
                {
                    "class": "spikes_total_from_agg_fallback",
                    "run_id": rid,
                    "detail": "顶层 spikes_total 缺失，回退 agg.spk_total=%d" % agg_spk,
                }
            )
        else:
            rec["spikes_total"] = None
            rec["spikes_total_source"] = "MISSING"
            anomalies.append(
                {
                    "class": "spikes_total_missing",
                    "run_id": rid,
                    "detail": "顶层 spikes_total 与 agg.spk_total 均缺失/非整数",
                }
            )
        if (
            isinstance(doc.get("spikes_total"), int)
            and isinstance(agg_spk, int)
            and doc["spikes_total"] != agg_spk
        ):
            anomalies.append(
                {
                    "class": "spikes_total_agg_mismatch",
                    "run_id": rid,
                    "detail": "top_level=%d vs agg.spk_total=%d（采用 top_level）"
                    % (doc["spikes_total"], agg_spk),
                }
            )

        # ei_ratio：实测在 ei 子字典内，可能为 null
        ei = doc.get("ei") if isinstance(doc.get("ei"), dict) else {}
        rec["ei_ratio"] = ei.get("ei_ratio")
        rec["ei_ratio_present"] = "ei_ratio" in ei
        rec["i_exc_integral"] = ei.get("i_exc_integral")
        rec["i_inh_integral"] = ei.get("i_inh_integral")

        # zero_activity：顶层键（R116 新分支才写；现有臂实测全部缺失）
        rec["zero_activity_present"] = "zero_activity" in doc
        rec["zero_activity"] = doc.get("zero_activity")

        rec["n_steps_meas"] = doc.get("n_steps_meas")
        rec["n_steps_settle"] = doc.get("n_steps_settle")

        # counts.csv：优先同目录兄弟文件，其次按 json 内记录的 counts 路径（相对仓库根）
        cpath = os.path.join(runs_dir, rid + "_counts.csv")
        csrc = "sibling"
        if not os.path.isfile(cpath):
            rec_counts = doc.get("counts")
            cand = None
            if isinstance(rec_counts, str):
                cand = rec_counts
                if not os.path.isabs(cand):
                    cand = os.path.join(runs_dir, cand)
                    if not os.path.isfile(cand):
                        repo_root = os.path.abspath(
                            os.path.join(runs_dir, os.pardir, os.pardir, os.pardir)
                        )
                        cand = os.path.join(repo_root, rec_counts)
            if cand and os.path.isfile(cand):
                cpath, csrc = cand, "recorded_path"
        rec["counts_path"] = os.path.abspath(cpath)
        rec["counts_path_source"] = csrc if os.path.isfile(cpath) else "MISSING"
        rec["counts_sha256"] = None
        rec["n_counts_rows"] = None
        rec["counts_sum"] = None
        rec["counts_header"] = None
        rec["count_vals"] = None
        rec["cell_types"] = None
        rec["neuron_ids"] = None
        rec["subtype_counts"] = None
        if not os.path.isfile(cpath):
            anomalies.append(
                {
                    "class": "arm_json_without_counts",
                    "run_id": rid,
                    "detail": "有 %s.json 但缺 counts.csv（预期同目录 %s_counts.csv）"
                    % (rid, rid),
                }
            )
        else:
            rec["counts_sha256"] = sha256_file(cpath)
            try:
                got = read_counts_csv(cpath)
            except (OSError, ValueError) as exc:
                got = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
            if not got.get("ok"):
                anomalies.append(
                    {
                        "class": "counts_unreadable",
                        "run_id": rid,
                        "detail": str(got.get("error")),
                    }
                )
            else:
                rec["counts_header"] = got["header"]
                if got["header"][:3] != ["neuron_id", "count", "cell_type"]:
                    anomalies.append(
                        {
                            "class": "counts_header_unexpected",
                            "run_id": rid,
                            "detail": "实测表头=%r（预期 ['neuron_id','count','cell_type']）"
                            % (got["header"],),
                        }
                    )
                rec["count_vals"] = got["counts"]
                rec["cell_types"] = got["cell_types"] or None
                rec["neuron_ids"] = got["neuron_ids"] or None
                rec["n_counts_rows"] = len(got["counts"])
                rec["counts_sum"] = int(sum(got["counts"]))
                if rec["spikes_total"] is not None and rec["counts_sum"] != rec["spikes_total"]:
                    anomalies.append(
                        {
                            "class": "sum_counts_vs_spikes_total_mismatch",
                            "run_id": rid,
                            "detail": "sum(counts)=%d vs spikes_total=%d"
                            % (rec["counts_sum"], rec["spikes_total"]),
                        }
                    )
                if rec["cell_types"]:
                    sc = {}
                    for ct in rec["cell_types"]:
                        sc[ct] = sc.get(ct, 0) + 1
                    rec["subtype_counts"] = sc
                    for ct in sorted(sc):
                        if ct not in SUBTYPES:
                            anomalies.append(
                                {
                                    "class": "cell_type_not_in_registered_set",
                                    "run_id": rid,
                                    "detail": "实测 cell_type=%r（登记集合 %s），%d 个神经元"
                                    % (ct, list(SUBTYPES), sc[ct]),
                                }
                            )

        if rec.get("config_id") is None or rec.get("role") is None:
            anomalies.append(
                {
                    "class": "arm_json_missing_config_or_role",
                    "run_id": rid,
                    "detail": "config_id=%r role=%r ⇒ 该臂不进任何 (config_id, role) 单元"
                    % (rec.get("config_id"), rec.get("role")),
                }
            )
        if rec.get("role") is not None and rec["role"] not in KNOWN_ROLES:
            anomalies.append(
                {
                    "class": "unexpected_role",
                    "run_id": rid,
                    "detail": "role=%r 不在登记角色 %s" % (rec["role"], list(KNOWN_ROLES)),
                }
            )
        dm = rec.get("drive_mult")
        if rec.get("role") == "baseline" and dm == 0:
            anomalies.append(
                {
                    "class": "drive_mult_unexpected_for_role",
                    "run_id": rid,
                    "detail": "role=baseline 但 drive_mult=%r（期望 1.0）" % dm,
                }
            )
        if rec.get("role") == "zero_drive" and dm not in (None, 0, 0.0):
            anomalies.append(
                {
                    "class": "drive_mult_unexpected_for_role",
                    "run_id": rid,
                    "detail": "role=zero_drive 但 drive_mult=%r（期望 0.0）" % dm,
                }
            )
        if rid not in done_ids:
            anomalies.append(
                {
                    "class": "arm_json_without_done",
                    "run_id": rid,
                    "detail": "有 .json 但缺 .done（可能是在飞/部分写入的臂）",
                }
            )
        arms.append(rec)

    have_json = {a["run_id"] for a in arms}
    for rid in sorted(counts_ids - have_json):
        anomalies.append(
            {
                "class": "counts_without_arm_json",
                "run_id": rid,
                "detail": "有 %s_counts.csv 但无 %s.json ⇒ 无身份三元组，不进任何单元" % (rid, rid),
            }
        )
    for rid in sorted(done_ids - have_json):
        anomalies.append(
            {
                "class": "done_without_arm_json",
                "run_id": rid,
                "detail": "有 .done 但无 .json（产物被删/写失败）",
            }
        )
    spec_only = sorted(spec_ids - have_json - set(fail_ids))
    for rid in spec_only:
        anomalies.append(
            {
                "class": "spec_without_arm_json",
                "run_id": rid,
                "detail": "有计划 spec 但无 .json/.fail ⇒ 尚未开始或未完成（登记，非失败）",
            }
        )

    inventory = {
        "n_files_listed": len(files),
        "n_arm_json": len(arms),
        "n_spec_json": len(spec_ids),
        "n_done": len(done_ids),
        "n_fail": len(fail_ids),
        "n_counts_csv": len(counts_ids),
        "n_arm_json_with_counts": sum(1 for a in arms if a.get("n_counts_rows") is not None),
        "n_arm_json_without_counts": sum(1 for a in arms if a.get("n_counts_rows") is None),
        "spec_ids_without_arm_json": spec_only,
        "fail_ids": sorted(fail_ids),
        "run_ids_with_arm_json": sorted(have_json),
    }
    return arms, anomalies, inventory


# ---------------------------------------------------------------- 单元统计
def cell_stats(arms_of_cell):
    """给定一个 (config_id, role) 单元的全部臂 ⇒ 该单元统计量。"""
    spikes = [a["spikes_total"] for a in arms_of_cell if a["spikes_total"] is not None]
    with_counts = [a for a in arms_of_cell if a.get("n_counts_rows") is not None]
    out = {
        "n_impl": len(arms_of_cell),
        "n_impl_with_spikes": len(spikes),
        "n_impl_with_counts": len(with_counts),
        "pooled_spikes": int(sum(spikes)) if spikes else None,
        "run_ids": sorted(a["run_id"] for a in arms_of_cell),
        "T_ms": None,
        "T_s": None,
        "n_neuron": None,
        "T_ms_values": sorted({a["T_ms"] for a in arms_of_cell if a.get("T_ms") is not None}),
        "n_neuron_values": sorted(
            {a["n_neuron"] for a in arms_of_cell if a.get("n_neuron") is not None}
        ),
        "seed_values": sorted({a["seed"] for a in arms_of_cell if a.get("seed") is not None}),
        "a_median_hz_per_impl": [],
        "a_active_frac_per_impl": [],
        "subtype_per_impl": {},
        "ei_ratio_per_impl": [],
        "ei_ratio_absent_or_null": [],
        "sum_counts_per_impl": {},
        "spikes_total_per_impl": {},
    }
    if not arms_of_cell:
        return out
    ts_vals = sorted({a["T_ms"] for a in arms_of_cell if a.get("T_ms") is not None})
    n_vals = sorted({a["n_neuron"] for a in arms_of_cell if a.get("n_neuron") is not None})
    if len(ts_vals) == 1:
        out["T_ms"] = ts_vals[0]
        out["T_s"] = float(ts_vals[0]) / 1000.0
    if len(n_vals) == 1:
        out["n_neuron"] = n_vals[0]

    for a in sorted(with_counts, key=lambda x: x["run_id"]):
        if a.get("T_ms") in (None, 0):
            continue
        ts = float(a["T_ms"]) / 1000.0
        cnts = a["count_vals"]
        out["sum_counts_per_impl"][a["run_id"]] = a["counts_sum"]
        out["spikes_total_per_impl"][a["run_id"]] = a["spikes_total"]
        rates = [c / ts for c in cnts]
        m_rate = med(rates)
        med_count = med([float(c) for c in cnts])
        out["a_median_hz_per_impl"].append(
            {
                "run_id": a["run_id"],
                "T_s": ts,
                "n_neuron": len(cnts),
                "median_count": r6(med_count),
                "median_hz": r6(m_rate),
                "median_hz_from_median_count": r6(med_count / ts) if med_count is not None else None,
                "active_frac": r6(
                    (sum(1 for c in cnts if c >= 1) / float(len(cnts))) if cnts else None
                ),
                "frac_nonzero": r6(
                    (sum(1 for c in cnts if c > 0) / float(len(cnts))) if cnts else None
                ),
                "sum_counts": a["counts_sum"],
                "spikes_total": a["spikes_total"],
            }
        )
        af = (sum(1 for c in cnts if c >= 1) / float(len(cnts))) if cnts else None
        out["a_active_frac_per_impl"].append({"run_id": a["run_id"], "value": r6(af)})
        if a.get("cell_types") and len(a["cell_types"]) == len(cnts):
            per = {}
            for ct in sorted(set(a["cell_types"])):
                vals = [c for c, k in zip(cnts, a["cell_types"]) if k == ct]
                if not vals:
                    continue
                per[ct] = {
                    "n_neuron": len(vals),
                    "median_hz": r6(med([v / ts for v in vals])),
                    "active_frac": r6(sum(1 for v in vals if v >= 1) / float(len(vals))),
                }
            out["subtype_per_impl"][a["run_id"]] = per
        if a.get("ei_ratio") is None:
            out["ei_ratio_absent_or_null"].append(
                {
                    "run_id": a["run_id"],
                    "ei_ratio_key_present": a.get("ei_ratio_present"),
                    "i_exc_integral": a.get("i_exc_integral"),
                    "i_inh_integral": a.get("i_inh_integral"),
                }
            )
        else:
            out["ei_ratio_per_impl"].append({"run_id": a["run_id"], "value": r6(a["ei_ratio"])})
    return out


def aggregate_subtypes(cell):
    """逐实现的亚型读数 ⇒ 中位数；亚型集合按登记顺序 + 实测额外值（显式带上）。"""
    keys = set()
    for per in cell["subtype_per_impl"].values():
        keys.update(per.keys())
    ordered = [k for k in SUBTYPES if k in keys] + sorted(keys - set(SUBTYPES))
    out = {}
    for ct in ordered:
        rows = [per[ct] for per in cell["subtype_per_impl"].values() if ct in per]
        if not rows:
            continue
        out[ct] = {
            "registered_subtype": ct in SUBTYPES,
            "n_impl": len(rows),
            "n_neuron_values": sorted({r["n_neuron"] for r in rows}),
            "median_hz": r6(med([r["median_hz"] for r in rows])),
            "median_hz_impl_mean": r6(mean([r["median_hz"] for r in rows])),
            "active_frac": r6(med([r["active_frac"] for r in rows])),
            "active_frac_impl_mean": r6(mean([r["active_frac"] for r in rows])),
            "median_hz_per_impl": {
                rid: per[ct]["median_hz"]
                for rid, per in sorted(cell["subtype_per_impl"].items())
                if ct in per
            },
            "active_frac_per_impl": {
                rid: per[ct]["active_frac"]
                for rid, per in sorted(cell["subtype_per_impl"].items())
                if ct in per
            },
        }
    return out


def pooled_median_hz(arms_of_cell):
    """交叉核对口径：**先合池、再取中位数**（按 neuron_id 对齐合池计数 / T_s 的中位数）。

    ⚠️ 该口径与本工具主口径（逐实现再取中位数）**不同**，只作交叉核对。
    实测注意：``neuron_id → cell_type`` 的映射**逐实现不同**（每个实现用不同 net_seed 重建网络）
    ⇒ 合池不是「同一个神经元跨实现相加」，而是「同一个索引跨实现相加」；
    本函数仍按登记要求算出该量，但把 `neuron_id` 集合/行序/`cell_type` 一致性
    **逐项显式登记**，供读者判断该交叉核对的可解释性。
    """
    ws = [a for a in arms_of_cell if a.get("n_counts_rows") is not None]
    if not ws:
        return {"available": False, "reason": "no arm in this cell has counts.csv"}
    if any(a.get("neuron_ids") is None for a in ws):
        return {"available": False, "reason": "neuron_id column absent"}
    sets = [set(a["neuron_ids"]) for a in ws]
    if any(s != sets[0] for s in sets[1:]):
        return {
            "available": False,
            "reason": "neuron_id 集合跨实现不一致（无法按索引合池）",
            "n_neuron_values": sorted(len(s) for s in sets),
        }
    row_order_same = all(a["neuron_ids"] == ws[0]["neuron_ids"] for a in ws[1:])
    ct_maps = [dict(zip(a["neuron_ids"], a["cell_types"])) for a in ws if a.get("cell_types")]
    ct_same = (
        all(m == ct_maps[0] for m in ct_maps[1:]) if len(ct_maps) == len(ws) else None
    )
    acc = {}
    for a in ws:
        for nid, c in zip(a["neuron_ids"], a["count_vals"]):
            acc[nid] = acc.get(nid, 0) + c
    tot = [acc[nid] for nid in sorted(acc)]
    ts = float(sorted({float(a["T_ms"]) for a in ws})[0]) / 1000.0
    out = {
        "available": True,
        "n_impl_pooled": len(ws),
        "T_s": ts,
        "median_hz": r6(med([c / ts for c in tot])),
        "active_frac": r6(sum(1 for c in tot if c >= 1) / float(len(tot))),
        "pooled_sum_check": int(sum(tot)),
        "n_neuron_values": sorted({len(a["neuron_ids"]) for a in ws}),
        "run_ids_pooled": sorted(a["run_id"] for a in ws),
        "neuron_id_row_order_identical_across_impls": row_order_same,
        "cell_type_by_neuron_id_identical_across_impls": ct_same,
        "definition": "按 neuron_id 对齐后跨实现相加的合池计数 / T_s 的中位数",
        "caveat": (
            "合池按 neuron_id（= 索引）对齐；实测 cell_type 映射逐实现不同 ⇒ 这不是"
            "「同一生物神经元跨实现相加」。本字段只作登记要求的交叉核对，"
            "**主口径仍是 a_median_hz（逐实现再取中位数）**。"
        )
        if ct_same is False
        else "合池按 neuron_id 对齐；各实现 cell_type 映射一致",
    }
    return out


# ---------------------------------------------------------------- 报告装配
def build_configs(arms):
    by_cell = {}
    for a in arms:
        if a.get("config_id") is None or a.get("role") is None:
            continue
        by_cell.setdefault((a["config_id"], a["role"]), []).append(a)

    cells = {}
    for key in sorted(by_cell):
        cells[key] = cell_stats(by_cell[key])

    configs = []
    warnings = []
    for cid in sorted({k[0] for k in cells}):
        base = cells.get((cid, "baseline"))
        zd = cells.get((cid, "zero_drive"))
        extra_roles = sorted(r for (c, r) in cells if c == cid and r not in KNOWN_ROLES)
        cells_extra = {r: cells[(cid, r)] for r in extra_roles}

        pooled_zd = zd["pooled_spikes"] if zd else None
        pooled_bl = base["pooled_spikes"] if base else None

        # ---- (c)
        if pooled_bl is None:
            R, r_note = None, "pooled_baseline 不可用（无 baseline 臂或无 spikes_total）"
        elif pooled_bl == 0:
            R, r_note = None, "pooled_baseline==0 ⇒ R=null（不除零，显式标注）"
        elif pooled_zd is None:
            R, r_note = None, "pooled_zero_drive 不可用（无 zero_drive 臂或无 spikes_total）"
        else:
            R, r_note = r6(pooled_zd / float(pooled_bl)), None
        window_hi = r6(R_MAX * pooled_bl) if pooled_bl is not None else None
        c_judgeable = (pooled_zd is not None) and (pooled_zd >= WINDOW_LO)
        c_window_ok = bool(
            (pooled_zd is not None)
            and (pooled_zd >= WINDOW_LO)
            and (R is not None)
            and (R <= R_MAX)
        )
        if R is None or not c_judgeable:
            c_verdict = "unavailable"
        elif R <= R_MAX:
            c_verdict = "pass"
        else:
            c_verdict = "fail"
        in_window = (
            None
            if (pooled_zd is None or window_hi is None)
            else bool(WINDOW_LO <= pooled_zd <= window_hi)
        )
        if in_window is not None and in_window != c_window_ok:
            warnings.append(
                "%s: in_window(%r) != c_window_ok(%r) ⇒ 内部不一致，须查"
                % (cid, in_window, c_window_ok)
            )

        # ---- (a) 三口径（基准臂）
        a_med_impl = (
            med([r["median_hz"] for r in base["a_median_hz_per_impl"]]) if base else None
        )
        a_med_impl_mean = (
            mean([r["median_hz"] for r in base["a_median_hz_per_impl"]])
            if base and base["a_median_hz_per_impl"]
            else None
        )
        a_active_frac = (
            med([r["value"] for r in base["a_active_frac_per_impl"]]) if base else None
        )
        a_active_frac_mean = (
            mean([r["value"] for r in base["a_active_frac_per_impl"]])
            if base and base["a_active_frac_per_impl"]
            else None
        )
        pooled_ck = pooled_median_hz(by_cell.get((cid, "baseline"), []))
        a_med_pooled = pooled_ck.get("median_hz") if pooled_ck.get("available") else None
        crosscheck_diff = (
            r6(a_med_impl - a_med_pooled)
            if (a_med_impl is not None and a_med_pooled is not None)
            else None
        )
        a_pop_mean_hz = None
        if pooled_bl is not None and base and base.get("n_impl_with_spikes") and base.get(
            "n_neuron"
        ) and base.get("T_s"):
            a_pop_mean_hz = r6(
                pooled_bl / (base["n_impl_with_spikes"] * float(base["n_neuron"]) * base["T_s"])
            )
        a_in_band = (
            None if a_med_impl is None else bool(A_BAND[0] <= a_med_impl <= A_BAND[1])
        )

        # ---- (e) 只判基准臂
        e_ratio = med([r["value"] for r in base["ei_ratio_per_impl"]]) if base else None
        e_in_band = None if e_ratio is None else bool(E_BAND[0] <= e_ratio <= E_BAND[1])
        zd_ei = [
            {
                "run_id": r["run_id"],
                "ei_ratio": r["value"],
                "is_a_criterion": False,
                "why_not": "R118 §④2：零驱动下若只剩极少数细胞发放 ⇒ i_exc_integral 极小 ⇒ 比值爆炸；"
                "**不得**与基准臂 (e) 混用",
            }
            for r in (zd["ei_ratio_per_impl"] if zd else [])
        ]
        zd_ei_absent = [
            {"run_id": r["run_id"], "ei_ratio": None, "is_a_criterion": False,
             "why_not": "ei.ei_ratio 为 null 或缺键（显式记录，不插值）"}
            for r in (zd["ei_ratio_absent_or_null"] if zd else [])
        ]

        # ---- 反平凡性
        triviality_flag = bool(
            c_window_ok and (a_active_frac is not None) and (a_active_frac < TRIVIAL_ACTIVE_FRAC)
        )

        subtypes = aggregate_subtypes(base) if base else {}

        if pooled_ck.get("available") and pooled_ck.get(
            "cell_type_by_neuron_id_identical_across_impls"
        ) is False:
            warnings.append(
                "%s: 基准臂的 neuron_id→cell_type 映射**逐实现不同**（各实现 net_seed 不同）"
                "⇒ 「先合池再取中位数」的交叉核对按索引对齐、语义受限（已在该字段 caveat 中登记）"
                % cid
            )
        if (base is not None) and base["n_impl_with_counts"] < base["n_impl"]:
            warnings.append(
                "%s: 基准臂缺 counts.csv 的实现数 %d/%d ⇒ 中位数口径样本不足"
                % (cid, base["n_impl"] - base["n_impl_with_counts"], base["n_impl"])
            )
        elif (base is not None) and 0 < base["n_impl"] < 3:
            warnings.append(
                "%s: 基准臂实现数 %d < 3（登记要求 3 个独立实现）"
                % (cid, base["n_impl"])
            )
        if (zd is not None) and zd["n_impl_with_counts"] < zd["n_impl"]:
            warnings.append(
                "%s: 零驱动臂缺 counts.csv 的实现数 %d/%d"
                % (cid, zd["n_impl"] - zd["n_impl_with_counts"], zd["n_impl"])
            )

        for row in (base["a_median_hz_per_impl"] if base else []):
            if row.get("median_hz") != row.get("median_hz_from_median_count"):
                warnings.append(
                    "%s/%s: median(count/T_s)=%s != median(count)/T_s=%s（口径字面差异）"
                    % (cid, row["run_id"], row["median_hz"], row["median_hz_from_median_count"])
                )
            if row.get("active_frac") != row.get("frac_nonzero"):
                warnings.append(
                    "%s/%s: active_frac(%s) != frac_nonzero(%s)（应同义，须查）"
                    % (cid, row["run_id"], row["active_frac"], row["frac_nonzero"])
                )

        cfg = {
            "config_id": cid,
            "roles_present": sorted(
                {r for (c, r) in cells if c == cid}, key=lambda r: (ROLE_RANK.get(r, 9), r)
            ),
            "roles_absent": [r for r in KNOWN_ROLES if (cid, r) not in cells],
            "unexpected_roles": extra_roles,
            "n_impl_baseline": base["n_impl"] if base else 0,
            "n_impl_baseline_with_spikes": base["n_impl_with_spikes"] if base else 0,
            "n_impl_baseline_with_counts": base["n_impl_with_counts"] if base else 0,
            "n_impl_zero_drive": zd["n_impl"] if zd else 0,
            "n_impl_zero_drive_with_spikes": zd["n_impl_with_spikes"] if zd else 0,
            "n_impl_zero_drive_with_counts": zd["n_impl_with_counts"] if zd else 0,
            "pooled_zero_drive": pooled_zd,
            "pooled_baseline": pooled_bl,
            "R": R,
            "R_note": r_note,
            "window_lo": WINDOW_LO,
            "window_hi": window_hi,
            "window_hi_definition": "0.10 * pooled_baseline",
            "in_window": in_window,
            "c_window_ok": c_window_ok,
            "c_judgeable": c_judgeable,
            "c_judgeable_definition": "pooled_zero_drive >= 300",
            "c_judgeable_note": "按登记定义只看零驱动绝对值；若基准臂缺失/为 0（R=null），"
            "c_judgeable 仍可能为 True，而 c_window_ok 必为 False、c_verdict='unavailable'",
            "c_verdict": c_verdict,
            "c_verdict_rule": "not c_judgeable ⇒ 'unavailable'（R115 §②：不得判 pass 也不得判 fail）；"
            "否则 R<=0.10 ⇒ 'pass'，R>0.10 ⇒ 'fail'",
            "T_ms": (base or zd or {}).get("T_ms"),
            "T_s": (base or zd or {}).get("T_s"),
            "T_ms_values": sorted(set((base or {}).get("T_ms_values", []) + (zd or {}).get("T_ms_values", []))),
            "n_neuron": (base or zd or {}).get("n_neuron"),
            "n_neuron_values": sorted(
                set((base or {}).get("n_neuron_values", []) + (zd or {}).get("n_neuron_values", []))
            ),
            "a_median_hz": r6(a_med_impl),
            "a_median_hz_definition": "**逐实现**先算 median_i(count_i / T_s)，再对**实现**取中位数（登记口径）",
            "a_median_hz_impl_mean": r6(a_med_impl_mean),
            "a_median_hz_pooled_counts": a_med_pooled,
            "a_median_hz_crosscheck": pooled_ck,
            "a_median_hz_crosscheck_diff": crosscheck_diff,
            "a_median_hz_crosscheck_note": "两者不同必须显式报告；本字段给出差值（逐实现口径 − 先合池口径）",
            "a_active_frac": r6(a_active_frac),
            "a_active_frac_definition": "**逐实现**先算 count>=1 的神经元占比，再对**实现**取中位数",
            "a_active_frac_impl_mean": r6(a_active_frac_mean),
            "a_pop_mean_hz": a_pop_mean_hz,
            "a_pop_mean_hz_definition": "pooled_baseline / (n_impl_with_spikes * N * T_s)",
            "a_pop_mean_hz_n_impl": base["n_impl_with_spikes"] if base else None,
            "a_in_band": a_in_band,
            "a_band_hz": list(A_BAND),
            "frac_nonzero": r6(a_active_frac),
            "frac_nonzero_definition": "count > 0 的神经元占比（基准臂，逐实现再取中位数）",
            "frac_nonzero_note": "计数为非负整数 ⇒ count>=1 ⇔ count>0 ⇒ 与 a_active_frac **同义**；"
            "本报告不另设第二个量（避免同名双义）",
            "subtype_baseline": subtypes,
            "e_ei_ratio": r6(e_ratio),
            "e_ei_ratio_definition": "基准臂 ei.ei_ratio 的中位数（逐实现再取中位数）",
            "e_in_band": e_in_band,
            "e_band": list(E_BAND),
            "zero_drive_ei_ratio_list": zd_ei + zd_ei_absent,
            "zero_drive_ei_ratio_is_a_criterion": False,
            "zero_drive_ei_ratio_note": "R118 §④2 明文：零驱动臂 ei_ratio **不是**判据，"
            "**不得**与基准臂 (e) 混用；此处只作并报",
            "triviality_flag": triviality_flag,
            "triviality_rule": "c_window_ok and (a_active_frac < %.2f)" % TRIVIAL_ACTIVE_FRAC,
            "triviality_warning": TRIVIALITY_WARNING if triviality_flag else None,
            "extra_role_cells": {
                r: {
                    "n_impl": cells_extra[r]["n_impl"],
                    "pooled_spikes": cells_extra[r]["pooled_spikes"],
                    "median_hz": r6(med([x["median_hz"] for x in cells_extra[r]["a_median_hz_per_impl"]])),
                    "active_frac": r6(
                        med([x["value"] for x in cells_extra[r]["a_active_frac_per_impl"]])
                    ),
                }
                for r in extra_roles
            },
        }
        configs.append(cfg)
        if triviality_flag:
            warnings.append("%s: %s" % (cid, TRIVIALITY_WARNING))
        if base and len(base["T_ms_values"]) > 1:
            warnings.append("%s: baseline 臂 T_ms 不一致 %s" % (cid, base["T_ms_values"]))
        if zd and len(zd["T_ms_values"]) > 1:
            warnings.append("%s: zero_drive 臂 T_ms 不一致 %s" % (cid, zd["T_ms_values"]))
    return configs, cells, warnings


# ---------------------------------------------------------------- CSV
CSV_COLUMNS = [
    "config_id",
    "role",
    "cell_n_impl",
    "cell_n_impl_with_spikes",
    "cell_n_impl_with_counts",
    "cell_pooled_spikes",
    "cell_median_hz",
    "cell_active_frac",
    "cell_pop_mean_hz",
    "cell_ei_ratio_median",
    "cell_T_ms",
    "cell_T_s",
    "cell_n_neuron",
    "subtype_E_n_neuron",
    "subtype_E_median_hz",
    "subtype_E_active_frac",
    "subtype_PV_n_neuron",
    "subtype_PV_median_hz",
    "subtype_PV_active_frac",
    "subtype_SST_n_neuron",
    "subtype_SST_median_hz",
    "subtype_SST_active_frac",
    "subtype_VIP_n_neuron",
    "subtype_VIP_median_hz",
    "subtype_VIP_active_frac",
    "cfg_pooled_zero_drive",
    "cfg_pooled_baseline",
    "cfg_R",
    "cfg_window_lo",
    "cfg_window_hi",
    "cfg_in_window",
    "cfg_c_window_ok",
    "cfg_c_judgeable",
    "cfg_c_verdict",
    "cfg_a_median_hz",
    "cfg_a_median_hz_pooled_counts",
    "cfg_a_median_hz_crosscheck_diff",
    "cfg_a_active_frac",
    "cfg_a_pop_mean_hz",
    "cfg_a_in_band",
    "cfg_frac_nonzero",
    "cfg_e_ei_ratio",
    "cfg_e_in_band",
    "cfg_triviality_flag",
    "cfg_n_impl_baseline_with_counts",
    "cfg_n_impl_zero_drive_with_counts",
]


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "True" if v else "False"
    return v


def write_csv(cells, cfg_by_id, path):
    keys = sorted(
        cells.keys(), key=lambda k: (k[0], ROLE_RANK.get(k[1], 9), k[1])
    )
    n = 0
    tmp = path + ".tmp"
    with open(tmp, "w", newline="\n") as fh:
        wr = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, lineterminator="\n")
        wr.writeheader()
        for cid, role in keys:
            cell = cells[(cid, role)]
            cfg = cfg_by_id.get(cid, {})
            ts = cell.get("T_s")
            N = cell.get("n_neuron")
            pop = None
            if (
                cell.get("pooled_spikes") is not None
                and cell.get("n_impl_with_spikes")
                and N
                and ts
            ):
                pop = r6(
                    cell["pooled_spikes"] / (cell["n_impl_with_spikes"] * float(N) * ts)
                )
            cnt_sub = aggregate_subtypes(cell)
            row = {
                "config_id": cid,
                "role": role,
                "cell_n_impl": cell["n_impl"],
                "cell_n_impl_with_spikes": cell["n_impl_with_spikes"],
                "cell_n_impl_with_counts": cell["n_impl_with_counts"],
                "cell_pooled_spikes": cell["pooled_spikes"],
                "cell_median_hz": r6(med([r["median_hz"] for r in cell["a_median_hz_per_impl"]])),
                "cell_active_frac": r6(
                    med([r["value"] for r in cell["a_active_frac_per_impl"]])
                ),
                "cell_pop_mean_hz": pop,
                "cell_ei_ratio_median": r6(
                    med([r["value"] for r in cell["ei_ratio_per_impl"]])
                ),
                "cell_T_ms": cell.get("T_ms"),
                "cell_T_s": r6(ts),
                "cell_n_neuron": N,
                "cfg_pooled_zero_drive": cfg.get("pooled_zero_drive"),
                "cfg_pooled_baseline": cfg.get("pooled_baseline"),
                "cfg_R": cfg.get("R"),
                "cfg_window_lo": cfg.get("window_lo"),
                "cfg_window_hi": cfg.get("window_hi"),
                "cfg_in_window": cfg.get("in_window"),
                "cfg_c_window_ok": cfg.get("c_window_ok"),
                "cfg_c_judgeable": cfg.get("c_judgeable"),
                "cfg_c_verdict": cfg.get("c_verdict"),
                "cfg_a_median_hz": cfg.get("a_median_hz"),
                "cfg_a_median_hz_pooled_counts": cfg.get("a_median_hz_pooled_counts"),
                "cfg_a_median_hz_crosscheck_diff": cfg.get("a_median_hz_crosscheck_diff"),
                "cfg_a_active_frac": cfg.get("a_active_frac"),
                "cfg_a_pop_mean_hz": cfg.get("a_pop_mean_hz"),
                "cfg_a_in_band": cfg.get("a_in_band"),
                "cfg_frac_nonzero": cfg.get("frac_nonzero"),
                "cfg_e_ei_ratio": cfg.get("e_ei_ratio"),
                "cfg_e_in_band": cfg.get("e_in_band"),
                "cfg_triviality_flag": cfg.get("triviality_flag"),
                "cfg_n_impl_baseline_with_counts": cfg.get("n_impl_baseline_with_counts"),
                "cfg_n_impl_zero_drive_with_counts": cfg.get("n_impl_zero_drive_with_counts"),
            }
            for ct in SUBTYPES:
                v = cnt_sub.get(ct) or {}
                row["subtype_%s_n_neuron" % ct] = (
                    v["n_neuron_values"][0] if v.get("n_neuron_values") else None
                )
                row["subtype_%s_median_hz" % ct] = v.get("median_hz")
                row["subtype_%s_active_frac" % ct] = v.get("active_frac")
            wr.writerow({k: fmt(row.get(k)) for k in CSV_COLUMNS})
            n += 1
    os.replace(tmp, path)
    return n


# ---------------------------------------------------------------- 主流程
def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    d = os.path.abspath(os.path.join(here, os.pardir, "data"))
    default_runs = os.path.join(d, "m11_p5_l3_runs")
    default_csv = os.path.join(d, "m11_p5_l3_report.csv")
    default_json = os.path.join(d, "m11_p5_l3_report.json")

    ap = argparse.ArgumentParser(
        description="M11 L3 补充报告器（只读）：四条报告量 + (a)/(c)/(e) 并报 + 缺件清单"
    )
    ap.add_argument("--runs", default=default_runs, help="逐臂产物目录（默认 %s）" % default_runs)
    ap.add_argument("--out-csv", default=default_csv, help="CSV 输出（默认 %s）" % default_csv)
    ap.add_argument("--out-json", default=default_json, help="JSON 输出（默认 %s）" % default_json)
    args = ap.parse_args(argv)

    runs_dir = os.path.abspath(args.runs)
    out_csv = os.path.abspath(args.out_csv)
    out_json = os.path.abspath(args.out_json)
    script_path = os.path.abspath(__file__)

    if not os.path.isdir(runs_dir):
        sys.stderr.write("ERROR: --runs 目录不存在: %s\n" % runs_dir)
        return 2
    for p in (out_csv, out_json):
        if is_inside(p, runs_dir):
            sys.stderr.write(
                "ERROR: 拒绝运行 —— 输出路径落在 --runs 目录内（只读约束）: %s\n" % p
            )
            return 3

    arms, anomalies, inventory = discover(runs_dir)
    configs, cells, warnings = build_configs(arms)
    cfg_by_id = {c["config_id"]: c for c in configs}

    anomalies = sorted(
        anomalies,
        key=lambda a: (a.get("class", ""), a.get("run_id", ""), a.get("detail", "")),
    )
    by_class = {}
    for a in anomalies:
        by_class[a["class"]] = by_class.get(a["class"], 0) + 1
    n_missing = sum(by_class.get(c, 0) for c in MISSING_CLASSES)

    zero_activity_present = sorted(a["run_id"] for a in arms if a.get("zero_activity_present"))

    report = {
        "tool": {
            "name": os.path.basename(script_path),
            "path": script_path,
            "sha256": sha256_file(script_path),
            "version": TOOL_VERSION,
            "role": "**补充**报告器（只读）：把 L3 正式跑产物汇总为登记要求的四条报告量；"
            "**不替代** tools/validate_p11_resting.py，**不改**任何判据",
            "dependencies": "仅 Python 标准库（无 numpy / torch / 子进程 / sys.executable）",
        },
        "registration_basis": {
            "l3_resume": "neural_exploration/docs/m11_l3_resume.md §3.2 / §3.3 / §3.4",
            "r114": "docs/m11_acceptance_master.md §3de（a_median_hz / a_active_frac / a_pop_mean_hz 三口径并报）",
            "r115": "docs/m11_acceptance_master.md §3df §④「必须同时报」（pooled_zero_drive 绝对值 + 亚型 + frac_nonzero）",
            "r118": "docs/m11_acceptance_master.md §3di §④2（零驱动 ei_ratio 非判据）/ §④4（反平凡性）",
            "prereg_amend": "data/m11_p5_prereg_amend_r109_a_reporting.json（四条并报为**描述性**；"
            "(a) 的频带与 pass/fail 规则逐字不变；不得用 a_active_frac 代替中位率判 (a)）",
            "why_not_existing_evaluator": "tools/validate_p11_resting.py 不提供这四条量，"
            "且按「每 config 一个值」设计、对杠杆扫描计划结构性错配（R108 §⑤）",
        },
        "notation_and_units": {
            "T_ms": "测量窗毫秒数（**不含 settle**）；实测 n_steps_meas=200000 × 0.05 ms/step = 10000 ms",
            "T_s": "T_ms / 1000（秒）",
            "N": "n_neuron",
            "a_median_hz": "Hz；median_i(count_i / T_s)，逐实现再取中位数（登记主判口径）",
            "a_active_frac": "无量纲 0–1；count>=1 的神经元占比（逐实现再取中位数）",
            "a_pop_mean_hz": "Hz；pooled_baseline / (n_impl * N * T_s)",
            "R": "无量纲；pooled_zero_drive / pooled_baseline",
            "pooled_zero_drive / pooled_baseline": "**整数脉冲数**（同 config_id 同角色全部臂 spikes_total 之和）",
            "frac_nonzero": "无量纲；与 a_active_frac 同义",
            "criterion_scopes": {
                "(a)": "基准臂（role=baseline）逐神经元中位率，带 [0.5, 2.0] Hz",
                "(c)": "跨角色合成（pooled_zero_drive vs pooled_baseline），带 R<=0.10 且 pooled_zero_drive>=300",
                "(e)": "**仅**基准臂 ei.ei_ratio，带 [0.5, 3.0]；零驱动臂只作并报",
            },
        },
        "inputs": {
            "runs_dir": runs_dir,
            "inventory": inventory,
            "arm_manifest": [
                {
                    "run_id": a["run_id"],
                    "config_id": a.get("config_id"),
                    "role": a.get("role"),
                    "json_bytes": a.get("json_bytes"),
                    "json_sha256": a.get("json_sha256"),
                    "counts_path": a.get("counts_path"),
                    "counts_path_source": a.get("counts_path_source"),
                    "counts_sha256": a.get("counts_sha256"),
                    "counts_header": a.get("counts_header"),
                    "spikes_total": a.get("spikes_total"),
                    "spikes_total_source": a.get("spikes_total_source"),
                    "sum_counts": a.get("counts_sum"),
                    "n_counts_rows": a.get("n_counts_rows"),
                    "T_ms": a.get("T_ms"),
                    "n_neuron": a.get("n_neuron"),
                    "seed": a.get("seed"),
                    "net_seed": a.get("net_seed"),
                    "event_seed": a.get("event_seed"),
                    "drive_mult": a.get("drive_mult"),
                    "bias_scale": a.get("bias_scale"),
                    "bias_log_sigma": a.get("bias_log_sigma"),
                    "ei_ratio": a.get("ei_ratio"),
                    "ei_ratio_present": a.get("ei_ratio_present"),
                    "i_exc_integral": a.get("i_exc_integral"),
                    "i_inh_integral": a.get("i_inh_integral"),
                    "zero_activity_present": a.get("zero_activity_present"),
                    "zero_activity": a.get("zero_activity"),
                    "subtype_counts": a.get("subtype_counts"),
                    "n_steps_meas": a.get("n_steps_meas"),
                    "n_steps_settle": a.get("n_steps_settle"),
                }
                for a in sorted(arms, key=lambda x: x["run_id"])
            ],
        },
        "keys_actually_read": {
            "note": "**实测读到的键/列名**（用于核对接口描述；与描述不同处在此显式标注）",
            "arm_json_top_level": [
                "arm_id", "config_id", "T_ms", "settle_ms", "role", "mechanism", "drive_mult",
                "seed", "replica_of", "n_neuron", "pop_trace", "counts", "agg", "ei", "wall_s",
                "ms_per_step", "device", "host", "python", "numpy", "torch", "ms_per_step_steady",
                "event_seed", "spikes_total", "seeds", "sha256_of_trace", "n_steps_meas",
                "n_steps_settle", "delivery", "engine_version", "n_ch", "interpreter",
                "drive_rate_hz", "drive_rate_hz_eff", "n_events", "rss_mb", "pid", "finished_at", "net",
            ],
            "keys_used": {
                "config_id": "顶层",
                "role": "顶层",
                "T_ms": "顶层",
                "spikes_total": "顶层（整数）",
                "agg.spk_total": "顶层 agg 子字典内（与 spikes_total 实测一致；仅作回退）",
                "n_neuron": "顶层",
                "seed": "顶层",
                "drive_mult": "顶层",
                "bias_scale": "**net.bias_scale**（**不在顶层**）",
                "bias_log_sigma": "**net.bias_log_sigma**（**不在顶层**）",
                "net_seed": "**seeds.net_seed**（**不在顶层**；另见 net.net_seed）",
                "event_seed": "顶层",
                "ei.ei_ratio": "**ei 子字典内**，可为 null",
                "zero_activity": "顶层；**本仓库现有臂实测全部缺失该键**",
                "n_steps_meas / n_steps_settle": "顶层",
            },
            "counts_csv_columns_measured": ["neuron_id", "count", "cell_type"],
            "counts_csv_cell_type_values_measured": sorted(
                {ct for a in arms if a.get("subtype_counts") for ct in a["subtype_counts"]}
            ),
            "n_arms_with_zero_activity_key": len(zero_activity_present),
            "arms_with_zero_activity_key": zero_activity_present,
            "arms_with_zero_activity_true": [
                a["run_id"] for a in arms if a.get("zero_activity") is True
            ],
            "zero_activity_explanation": "R116 的修复只在「零驱动臂 i_exc_integral==0」分支写 "
            "zero_activity；本仓库该分支无成功产物（两条该分支探针以 .fail 结束，见 anomalies）"
            "⇒ 实测无 zero_activity 样例（**登记为未覆盖，不推断其取值**）",
            "spec_json": "独立 <run_id>.spec.json 含顶层 bias_scale/bias_log_sigma"
            "（data/m11_p5_l1_runs 有；data/m11_p5_l3_feasibility **没有** ⇒ 本工具不依赖 spec）",
        },
        "configs": configs,
        "anomalies": {
            "n_items": len(anomalies),
            "n_items_for_missing_pieces": n_missing,
            "n_missing_definition": "下列类别之和: " + ", ".join(MISSING_CLASSES),
            "by_class": {k: by_class[k] for k in sorted(by_class)},
            "items": anomalies,
        },
        "warnings": sorted(set(warnings)),
        "self_check": {
            "read_only": "只打开 --runs 下文件读取；输出路径若落在 --runs 内则拒绝运行（退出码 3）",
            "determinism": "全部集合 sorted；输出无时间戳/主机/pid ⇒ 同输入两次运行逐字节相同",
            "no_subprocess": True,
            "no_numpy": True,
            "python_version": sys.version.split()[0],
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", "<unset>"),
        },
    }

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    n_rows = write_csv(cells, cfg_by_id, out_csv)
    tmpj = out_json + ".tmp"
    with open(tmpj, "w") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1, sort_keys=False)
        fh.write("\n")
    os.replace(tmpj, out_json)

    # ---------------- 控制台摘要（人读；不进产物 ⇒ 不影响产物确定性）
    print("== M11 L3 补充报告（只读） ==")
    print("runs_dir       : %s" % runs_dir)
    print(
        "arm json       : %d（有 counts %d / 缺 counts %d）"
        % (inventory["n_arm_json"], inventory["n_arm_json_with_counts"], inventory["n_arm_json_without_counts"])
    )
    print(
        "spec/done/fail : %d / %d / %d"
        % (inventory["n_spec_json"], inventory["n_done"], inventory["n_fail"])
    )
    print(
        "anomalies      : %d 项（其中缺件类 %d 项）"
        % (report["anomalies"]["n_items"], n_missing)
    )
    for k in sorted(by_class):
        print("    %-40s %d" % (k, by_class[k]))
    print("cells (CSV 行) : %d -> %s" % (n_rows, out_csv))
    print("json           : %s" % out_json)
    for cfg in configs:
        print("-" * 78)
        print(
            "%s  roles=%s  缺角色=%s"
            % (cfg["config_id"], cfg["roles_present"], cfg["roles_absent"])
        )
        print(
            "  pooled_zero_drive=%s  pooled_baseline=%s  R=%s  (c)=%s"
            % (cfg["pooled_zero_drive"], cfg["pooled_baseline"], cfg["R"], cfg["c_verdict"])
        )
        print(
            "  window=[%s, %s]  c_window_ok=%s  c_judgeable=%s  in_window=%s"
            % (cfg["window_lo"], cfg["window_hi"], cfg["c_window_ok"], cfg["c_judgeable"], cfg["in_window"])
        )
        print(
            "  a_median_hz=%s (先合池=%s, diff=%s)  a_active_frac=%s  a_pop_mean_hz=%s  a_in_band=%s"
            % (
                cfg["a_median_hz"],
                cfg["a_median_hz_pooled_counts"],
                cfg["a_median_hz_crosscheck_diff"],
                cfg["a_active_frac"],
                cfg["a_pop_mean_hz"],
                cfg["a_in_band"],
            )
        )
        print(
            "  frac_nonzero=%s (=a_active_frac)  e_ei_ratio=%s  e_in_band=%s  triviality=%s"
            % (cfg["frac_nonzero"], cfg["e_ei_ratio"], cfg["e_in_band"], cfg["triviality_flag"])
        )
        if cfg["zero_drive_ei_ratio_list"]:
            print(
                "  零驱动臂 ei_ratio（**非判据**，R118 §④2）: %s"
                % [(r["run_id"], r["ei_ratio"]) for r in cfg["zero_drive_ei_ratio_list"]]
            )
        if cfg["subtype_baseline"]:
            print(
                "  亚型(基准臂) 中位率/活跃占比: %s"
                % {
                    k: (v["median_hz"], v["active_frac"])
                    for k, v in cfg["subtype_baseline"].items()
                }
            )
        if cfg["triviality_flag"]:
            print("  " + TRIVIALITY_WARNING)
    for w in report["warnings"]:
        print("WARNING: %s" % w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
