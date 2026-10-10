"""M9 P9：类型级锚扰动预测验证（清单 §1.4/§4.5；全规模 + 分段调用）。

判据（**只读** `data/m9_behavior_reference.csv` 的 perturbation 段，运行前定稿，不事后调）：
  - **命中率 ≥70%**（有锚子集）：预注册因果方向预测 vs 实测后果类一致率；
  - **sham 零臂自检**：同代码路径 factor=1.0 → **逐神经元脉冲计数逐位相同**；
  - **确定性逐位**：同配置重跑逐位一致；
  - 操作有效性前置（非科学判据）：激活 → 目标池自身率↑；
  - 特异性对照（描述性）：同规模随机目标集施加同一操作的效应量对照。

协议：T=1000ms / settle=300ms / 分段 2000 步 / 定稿物理档（e_inh=−80、ek=−77、ahp_g=25、
v_floor=−85，**构建后读回断言**）/ MPS。

**前置诚实声明**：P7 R3 已证明"活动自主自持、输入增益过低"→ 类型级扰动可能不显著。
本工具**不得为凑命中率改带或选择性报告**；不显著/方向不符**如实记录为 M9 的正当反证**
（静态结构成立、状态切换动力学不涌现 → 缺失机制 STP/GABA 亚型异质/调质增益）。

输出：`data/m9_p9_perturbation.csv` + `data/m9_p9_target_mapping.csv`（**运行前写盘的预测**）
      + `reports/neuro/m9_perturbation_hitrate.png`
用法：PYTHONHASHSEED=0 MPLBACKEND=Agg ./.venv-m9/bin/python -u -m neural_exploration.tools.validate_p9_perturbation
"""

from __future__ import annotations

import csv as _csv
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports", "neuro")
REF_CSV = os.path.join(DATA_DIR, "m9_behavior_reference.csv")
PLAN_CSV = os.path.join(DATA_DIR, "m9_perturbation_plan.csv")
OUT_CSV = os.path.join(DATA_DIR, "m9_p9_perturbation.csv")
OUT_MAP = os.path.join(DATA_DIR, "m9_p9_target_mapping.csv")
OUT_PNG = os.path.join(REPORT_DIR, "m9_perturbation_hitrate.png")
OUT_LOG = os.path.join(REPORT_DIR, "m9_p9_progress.log")

#: 协议（env 可覆盖以便先探后跑；**判据带数值不可覆盖**）
T_MS = float(os.environ.get("M9_P9_T_MS", "1000"))
SETTLE_MS = float(os.environ.get("M9_P9_SETTLE_MS", "300"))
SEG_STEPS = int(os.environ.get("M9_P9_SEG", "2000"))
ABORT_S = float(os.environ.get("M9_P9_ABORT_S", "180"))
DEVICE = os.environ.get("M9_P9_DEVICE", "mps")
POOLS = os.environ.get("M9_P9_POOLS",
                       "MBON,MOD_MB,MOD_ALL,CX_EB,CX_ALL,KC_TO_MBON").split(",")
SEED = 0


def log(msg: str, path: str = OUT_LOG) -> None:
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_band(role: str) -> dict:
    out = {}
    with open(REF_CSV, encoding="utf-8") as f:
        hdr = None
        for row in _csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            if hdr is None:
                hdr = row
                continue
            d = dict(zip(hdr, row))
            if d.get("role") != role:
                continue
            out[d["metric"]] = {"lo": float(d["lo"]), "hi": float(d["hi"]),
                                "unit": d.get("unit", ""), "provenance": d.get("provenance", "")}
    return out


def load_anchors() -> list:
    rows = []
    with open(PLAN_CSV, encoding="utf-8") as f:
        hdr = None
        for row in _csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            if hdr is None:
                hdr = row
                continue
            rows.append(dict(zip(hdr, row)))
    return rows


def rate(counts: np.ndarray, mask: np.ndarray, T_ms: float) -> float:
    if not mask.any():
        return float("nan")
    return float(counts[mask].sum() / mask.sum() / (T_ms / 1000.0))


def main() -> int:
    from neural_exploration.src.adult_circuit import AdultCircuit, CircuitParams
    from neural_exploration.src.adult_perturb import (ANCHOR_MAP, CLASS_NAMES, EDGE_SPECS,
                                                     POOL_SPECS, UP, DOWN, NOCHANGE,
                                                     AdultPerturb, classify, map_predictions,
                                                     rel_delta)
    from neural_exploration.tools.validate_p9_resting import PARAMS, BG
    os.makedirs(REPORT_DIR, exist_ok=True)
    open(OUT_LOG, "w", encoding="utf-8").close()
    band = read_band("perturbation")
    thr = band["effect_threshold_rel"]["lo"]
    hit_min = band["hit_rate_min"]["lo"]
    log("=== M9 P9 扰动预测（§1.4/§4.5）===")
    log("判据带（只读 CSV perturbation 段）：命中率 ≥%.2f / 阈值 %.2f / sham=1 / deterministic=1"
        % (hit_min, thr))

    # ---------------- 装配 + 参数读回断言 ----------------
    c = AdultCircuit(device=DEVICE, params=CircuitParams(**PARAMS), use_compile=False)
    c.build()
    pt = c.engine.point
    rb = {"e_inh": float(pt.e_inh), "ek": float(pt.ek), "ahp_form": str(pt.ahp_form),
          "ahp_g_inc": float(pt.ahp_g_inc), "v_floor": float(pt.v_floor)}
    assert rb["e_inh"] == -80.0 and rb["ek"] == -77.0, rb
    assert rb["ahp_form"] == "conductance" and rb["ahp_g_inc"] == 25.0, rb
    assert rb["v_floor"] == -85.0, rb
    log("参数读回断言 ✓ %s（血泪教训 3）" % rb)

    g_ext = BG["epsp_mv"] / 0.104
    n_steps = int(round((SETTLE_MS + T_MS) / PARAMS["dt_ms"])) + 8
    c.build_background(rate_hz=BG["rate_hz"], g_ext=g_ext, n_steps=n_steps, seed=BG["seed"])
    c._bg_steps, c._bg_args = n_steps, (BG["rate_hz"], g_ext)
    p = AdultPerturb(c)

    # ---------------- 预注册映射 + 预测（**运行前写盘**） ----------------
    pools_nt = {}
    pool_info = {}
    # 诊断覆盖**全部被锚引用的池**（运行子集由 M9_P9_POOLS 控制；未跑的池不参与本会话判定）
    all_keys = sorted({ANCHOR_MAP[a["anchor_id"]]["target"] for a in load_anchors()
                       if a["anchor_id"] in ANCHOR_MAP
                       and ANCHOR_MAP[a["anchor_id"]]["target"]})
    for key in all_keys:
        if key in POOL_SPECS:
            m = p.pool_mask(key)
            em = m[c.pre]
            nt = p.dominant_nt_class(em)
            frac = np.bincount(c.nt_code[em].astype(np.int64), minlength=3) / max(int(em.sum()), 1)
            pool_info[key] = {"kind": "pool", "n_target": int(m.sum()),
                              "n_target_edges": int(em.sum()), "nt_class": nt,
                              "n_partner": int(p.partner_mask(key).sum()),
                              "nt_frac": [float(x) for x in frac]}
        else:
            em = p.edge_mask(key)
            nt = p.dominant_nt_class(em)
            frac = np.bincount(c.nt_code[em].astype(np.int64), minlength=3) / max(int(em.sum()), 1)
            pool_info[key] = {"kind": "edge", "n_target_edges": int(em.sum()),
                              "nt_class": nt,
                              "n_post": int(p.pool_mask(EDGE_SPECS[key]["to_pool"]).sum()),
                              "nt_frac": [float(x) for x in frac]}
        pools_nt[key] = nt
        log("代理目标 %s（%s，本会话%s）：%s" % (
            key, CLASS_NAMES.get(nt, "?"), "运行" if key in POOLS else "仅诊断",
            {k: v for k, v in pool_info[key].items() if k != "nt_frac"}))
        log("  操纵边递质类构成 exc %.3f / inh %.3f / mod %.3f（区域代理混类 → 限制 R7 直接证据）"
            % tuple(pool_info[key]["nt_frac"]))
    anchors = load_anchors()
    preds = map_predictions([{"anchor_id": a["anchor_id"],
                              "kind": ANCHOR_MAP[a["anchor_id"]]["kind"],
                              "target": ANCHOR_MAP[a["anchor_id"]]["target"],
                              "lit": ANCHOR_MAP[a["anchor_id"]]["lit_direction"],
                              "reason": ANCHOR_MAP[a["anchor_id"]]["map_reason"],
                              "pred_act": ANCHOR_MAP[a["anchor_id"]]["pred_act"],
                              "pred_source": ANCHOR_MAP[a["anchor_id"]]["pred_source"]}
                             for a in anchors if a["anchor_id"] in ANCHOR_MAP],
                            pools_nt)
    n_mapped = len({d["anchor_id"] for d in preds if d["predicted"] != "no-experiment"})
    n_unavail = len({d["anchor_id"] for d in preds if d["predicted"] == "no-experiment"})
    n_slots = sum(1 for d in preds if d["predicted"] != "no-experiment")
    log("锚映射：可测锚 %d（≥%.0f 下限）/ 不可映射 %d（no-experiment，不入分母）/ 预测槽 %d"
        % (n_mapped, band["n_anchor_min"]["lo"], n_unavail, n_slots))
    with open(OUT_MAP, "w", encoding="utf-8", newline="") as f:
        f.write("# M9 P9 类型级锚 → 代理目标 + **预注册因果方向预测**（运行前写盘；不做事后调）\n")
        f.write("# 预测来源：锚自身的**文献机制/已登记抽象**（pred_source 列；**不由实测反推**）\n")
        f.write("# target_nt_class_diag = 代理目标输出边主导递质类（**仅诊断**：混类 = 限制 R7 的直接证据）\n")
        f.write("# 代理塌缩：无 cell_type（R7）→ 同池多锚共享同一目标集（池级命中率为统计诚实单位）\n")
        w = _csv.writer(f, lineterminator="\n")
        w.writerow(["anchor_id", "kind", "target", "target_nt_class_diag", "slot",
                    "readout", "predicted", "pred_source", "lit_direction", "map_reason"])
        for d in preds:
            w.writerow([d["anchor_id"], d["kind"], d["target"] or "",
                        CLASS_NAMES.get(pools_nt.get(d["target"], -1), "-") if d["target"] else "-",
                        d["slot"], d["readout"], d["predicted"], d.get("pred_source", ""),
                        d.get("lit", ""), d.get("reason", "")])
    log("→ %s（预测已落盘，先于任何运行）" % OUT_MAP)

    # ---------------- 运行 ----------------
    t0 = time.perf_counter()
    results = {}

    def do_run(tag: str) -> dict:
        r = p.run(T_ms=T_MS, settle_ms=SETTLE_MS, seed=SEED, segment_steps=SEG_STEPS,
                  abort_s=ABORT_S, tag=tag)
        log("  运行 %s：自身 %.4f / 伙伴 %.4f / 群体 %.4f Hz，%.0fs（%.2f ms/step）"
            % (tag, r.self_rate_hz, r.partner_rate_hz, r.pop_rate_hz, r.wall_s, r.ms_per_step))
        return r

    log("协议：设备 %s、T=%.0fms、settle=%.0fms、%d 步/段、bg λ0=%.2f Hz、g_ext=%.2f"
        % (DEVICE, T_MS, SETTLE_MS, SEG_STEPS, BG["rate_hz"], g_ext))
    # 1) 基线（无扰动）
    p.restore()
    base = do_run("baseline")
    base_counts = base.counts
    # 2) sham 零臂（同代码路径 factor=1.0）
    p.prepare_silence(pool="MBON", factor=1.0)
    sham = do_run("sham_noop")
    sham_ok = bool(np.array_equal(base_counts, sham.counts))
    log("sham 零臂自检（factor=1.0 经同一路径）：逐位相同 = %s" % sham_ok)
    # 3) 确定性复跑
    p.restore()
    rep = do_run("baseline_rep")
    det_ok = bool(np.array_equal(base_counts, rep.counts))
    log("确定性复跑（同 seed）：逐位相同 = %s" % det_ok)

    # 4) 逐池扰动 + 随机 size-matched 对照
    for key in POOLS:
        info = pool_info[key]
        log("---- 目标 %s（%s）----" % (key, info["kind"]))
        if info["kind"] == "pool":
            p.prepare_silence(pool=key, factor=0.0)
            results[key + "|silence"] = do_run(key + "|silence")
            p.prepare_activation(pool=key, mult=1.0)
            results[key + "|activation"] = do_run(key + "|activation")
            rnd = p.random_matched_mask(key)
            p.prepare_silence(mask=rnd, factor=0.0, label=key + "|random")
            results[key + "|random_silence"] = do_run(key + "|random_silence")
            p.prepare_activation(mask=rnd, mult=1.0, label=key + "|random")
            results[key + "|random_activation"] = do_run(key + "|random_activation")
        else:
            p.prepare_edge_silence(key, factor=0.0)
            results[key + "|edge_silence"] = do_run(key + "|edge_silence")
            rem = p.random_matched_edge_mask(key)
            p.prepare_edge_silence(edge_mask=rem, factor=0.0, label=key + "|random")
            results[key + "|random_edge_silence"] = do_run(key + "|random_edge_silence")
    p.restore()

    # ---------------- 判定 ----------------
    def readout(counts, mask):
        return rate(counts, mask, T_MS)

    def self_partner_masks(key, res):
        """从 prepare info 取目标/伙伴掩码（激活与沉默的目标集一致）。"""
        if key in POOL_SPECS:
            tgt = p.pool_mask(key)
            return tgt, p.partner_mask(key)
        em = p.edge_mask(key)
        pre_m = np.zeros(c.n_neurons, bool)
        pre_m[c.pre[em]] = True
        post_m = np.zeros(c.n_neurons, bool)
        post_m[c.post[em]] = True
        return pre_m, post_m

    meas = {}
    for key in POOLS:
        tgt, partner = self_partner_masks(key, None)
        b_self = readout(base_counts, tgt)
        b_part = readout(base_counts, partner)
        row = {"self_base": b_self, "partner_base": b_part}
        if key in POOL_SPECS:
            s = results[key + "|silence"]
            a = results[key + "|activation"]
            rs = results[key + "|random_silence"]
            ra = results[key + "|random_activation"]
            row.update({
                "silence_self": readout(s.counts, tgt),
                "silence_partner": readout(s.counts, partner),
                "activation_self": readout(a.counts, tgt),
                "activation_partner": readout(a.counts, partner),
                "random_silence_partner": readout(rs.counts, partner),
                "random_activation_partner": readout(ra.counts, partner),
                "n_target": int(tgt.sum()), "n_partner": int(partner.sum()),
                "nt_class": pool_info[key]["nt_class"],
                "n_target_edges": pool_info[key]["n_target_edges"]})
            row["d_silence_partner"] = rel_delta(row["silence_partner"], b_part)
            row["d_silence_self"] = rel_delta(row["silence_self"], b_self)
            row["d_activation_partner"] = rel_delta(row["activation_partner"], b_part)
            row["d_activation_self"] = rel_delta(row["activation_self"], b_self)
            row["d_random_silence_partner"] = rel_delta(row["random_silence_partner"], b_part)
            row["d_random_activation_partner"] = rel_delta(row["random_activation_partner"], b_part)
            row["class_silence_partner"] = classify(row["d_silence_partner"], thr)
            row["class_silence_self"] = classify(row["d_silence_self"], thr)
            row["class_activation_partner"] = classify(row["d_activation_partner"], thr)
        else:
            s = results[key + "|edge_silence"]
            rs = results[key + "|random_edge_silence"]
            row.update({
                "silence_self": readout(s.counts, tgt),
                "silence_partner": readout(s.counts, partner),
                "random_silence_partner": readout(rs.counts, partner),
                "n_target": int(tgt.sum()), "n_partner": int(partner.sum()),
                "nt_class": pool_info[key]["nt_class"],
                "n_target_edges": pool_info[key]["n_target_edges"]})
            row["d_silence_partner"] = rel_delta(row["silence_partner"], b_part)
            row["d_silence_self"] = rel_delta(row["silence_self"], b_self)
            row["d_random_silence_partner"] = rel_delta(row["random_silence_partner"], b_part)
            row["class_silence_partner"] = classify(row["d_silence_partner"], thr)
            row["class_silence_self"] = classify(row["d_silence_self"], thr)
            row["class_activation_partner"] = "n/a(edge)"
            row["d_activation_partner"] = float("nan")
        meas[key] = row
        log("  %s 基线：自身 %.4f / 伙伴 %.4f Hz；Δ：静默伙伴 %+.3f 自身 %+.3f 激活伙伴 %s 自身 %s"
            % (key, row["self_base"], row["partner_base"], row["d_silence_partner"],
               row["d_silence_self"], row.get("d_activation_partner"),
               row.get("d_activation_self")))

    # 槽级判定 → 锚级 / 池级命中率
    slot_rows = []
    for d in preds:
        if d["predicted"] == "no-experiment":
            slot_rows.append({**d, "measured": "no-experiment", "hit": ""})
            continue
        key = d["target"]
        if key not in meas:
            slot_rows.append({**d, "measured": "not_run_this_session", "hit": ""})
            continue
        m = meas[key]
        if d["slot"] == "S1_silence_partner":
            got = m["class_silence_partner"]
        elif d["slot"] == "S2_silence_self":
            got = m["class_silence_self"]
        elif d["slot"] == "A1_activation_partner":
            got = m["class_activation_partner"]
        elif d["slot"] == "S1_edge_silence_post":
            got = m["class_silence_partner"]
        elif d["slot"] == "S2_edge_silence_pre":
            got = m["class_silence_self"]
        else:
            got = "?"
        hit = bool(got == d["predicted"])
        slot_rows.append({**d, "measured": got, "hit": hit})
    testable = [r for r in slot_rows if r["predicted"] != "no-experiment"
                and r["measured"] != "not_run_this_session"]
    n_hit = sum(1 for r in testable if r["hit"])
    hit_rate = n_hit / max(len(testable), 1)
    # 池级（统计诚实单位：同池多锚共享测量）
    pool_slots = {}
    for r in testable:
        pool_slots.setdefault((r["target"], r["slot"]), []).append(r)
    pool_hit = sum(1 for k, v in pool_slots.items() if v[0]["hit"])
    pool_rate = pool_hit / max(len(pool_slots), 1)
    # 操作有效性（激活 → 自身 ↑）
    eff = [(k, meas[k].get("d_activation_self", float("nan")))
           for k in POOLS if k in POOL_SPECS and k in meas]
    eff_frac = (sum(1 for _, v in eff if v > 0) / max(len(eff), 1)) if eff else float("nan")
    n_eff = sum(1 for _, v in eff if v > 0)
    n_eff_total = len(eff)
    # 特异性：目标效应量 vs 随机对照效应量（描述性）
    spec = {}
    for k in POOLS:
        m = meas[k]
        spec[k] = {"target": abs(m["d_silence_partner"]),
                   "control": abs(m["d_random_silence_partner"])}
    crit = {
        "hit_rate": bool(hit_rate >= hit_min),
        "sham_noop_identical": bool(sham_ok),
        "determinism_bitwise": bool(det_ok),
        "efficacy_activation_self_up": bool(n_eff_total == 0 or n_eff / n_eff_total >= 0.8),
        "anchor_min_20": bool(n_mapped >= 20),
    }
    verdict = "PASS" if all(crit.values()) else "FAIL"
    log("命中率：锚级 %.3f（%d/%d）｜池级 %.3f（%d/%d）｜判据 ≥%.2f → %s"
        % (hit_rate, n_hit, len(testable), pool_rate, pool_hit, len(pool_slots), hit_min,
           "PASS" if crit["hit_rate"] else "FAIL"))
    log("操作有效性（激活自身↑）：%d/%d 池" % (n_eff, n_eff_total))
    log("判定：%s（%s）" % (verdict, crit))
    log("总墙钟 %.0fs" % (time.perf_counter() - t0))

    # ---------------- 落盘 ----------------
    rows = [["section", "key", "value", "note"],
            ["protocol", "device", DEVICE, ""],
            ["protocol", "T_ms", "%.0f" % T_MS, "测量窗；率判据与 T 无关"],
            ["protocol", "settle_ms", "%.0f" % SETTLE_MS, ""],
            ["protocol", "segment_steps", SEG_STEPS, "全分段（规避长单次调用退化）"],
            ["protocol", "bg_rate_hz", BG["rate_hz"], ""],
            ["protocol", "g_ext", "%.4f" % g_ext, "每事件电导（EPSP %.1f mV）" % BG["epsp_mv"]],
            ["protocol", "seed", SEED, "确定性"],
            ["protocol", "params_readback", str(rb), "构建后读回断言 ✓"],
            ["protocol", "effect_threshold_rel", thr, "后果类判定阈值（运行前定稿）"],
            ["protocol", "activation_delta_bias_mv_s", float(np.median(p._bias0)),
             "激活 = 目标池 tonic 偏置 + 1.0×median(bias)"],
            ["anchor", "n_anchors_plan", len(anchors), "锚库类型级锚（含 unavailable 行）"],
            ["anchor", "n_anchors_mapped", n_mapped, "可映射到代理池（≥20 下限）"],
            ["anchor", "n_anchors_unavailable", n_unavail, "no-experiment（不入分母）"],
            ["anchor", "n_prediction_slots", n_slots, "每池锚 3 槽 / 边级锚 2 槽"],
            ["anchor", "pool_collapse_log", json_dumps({k: sum(1 for d in preds if d["target"] == k)
                                                        for k in POOLS}),
             "同池多锚共享目标集（无 cell_type → 区间不可分）"]]
    for k in POOLS:
        m = meas[k]
        rows.append(["pool", k + "_n_target", m["n_target"], ""])
        rows.append(["pool", k + "_n_partner", m["n_partner"], ""])
        rows.append(["pool", k + "_n_target_edges", m["n_target_edges"], ""])
        rows.append(["pool", k + "_nt_class", CLASS_NAMES.get(m["nt_class"], "?"),
                     "操纵边主导递质类（预测规则输入）"])
        rows.append(["pool", k + "_self_base_hz", "%.4f" % m["self_base"], "基线测量窗"])
        for tag, key in [("silence_partner", "silence_partner"), ("silence_self", "silence_self"),
                         ("activation_partner", "activation_partner"),
                         ("activation_self", "activation_self"),
                         ("random_silence_partner", "random_silence_partner"),
                         ("random_activation_partner", "random_activation_partner")]:
            if key in m:
                rows.append(["measure", "%s_%s_rel_delta" % (k, tag), "%.4f" % m["d_" + key]
                             if isinstance(m.get("d_" + key), float) else "",
                             "相对 sham 基线（同 seed 配对）"])
        rows.append(["measure", "%s_class_silence_partner" % k, m["class_silence_partner"], ""])
        rows.append(["measure", "%s_class_activation_partner" % k,
                     m["class_activation_partner"], ""])
    for r in slot_rows:
        rows.append(["slot", "%s|%s" % (r["anchor_id"], r["slot"] or "-"),
                     "pred=%s meas=%s hit=%s" % (r["predicted"], r["measured"], r["hit"]),
                     "target=%s nt=%s" % (r["target"], CLASS_NAMES.get(pools_nt.get(r["target"], -1), "-"))])
    rows += [["agg", "hit_rate_anchor_level", "%.4f" % hit_rate, "%d/%d 槽" % (n_hit, len(testable))],
             ["agg", "hit_rate_pool_level", "%.4f" % pool_rate, "%d/%d 池×槽" % (pool_hit, len(pool_slots))],
             ["agg", "efficacy_activation_self_up_frac", "%.4f" % eff_frac,
              "%d/%d 池自身率↑" % (n_eff, n_eff_total)],
             ["agg", "sham_noop_bitwise", str(sham_ok), "零臂自检"],
             ["agg", "determinism_bitwise", str(det_ok), ""],
             ["agg", "specificity_target_gt_control_pools",
              str(sum(1 for k in POOLS if spec[k]["target"] > spec[k]["control"])) + "/%d" % len(POOLS),
              "描述性（目标效应量 > 随机对照）"],
             ["agg", "wall_s", "%.0f" % (time.perf_counter() - t0), ""]]
    for k, v in crit.items():
        rows.append(["crit", k, str(bool(v)), "P9 判据"])
    rows.append(["crit", "verdict", verdict,
                 "三态：PASS / FAIL（反证记录）" if verdict == "PASS" else
                 "**FAIL → 反证记录（不显著/方向不符 = M9 正当结论：输入增益过低）**"])
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        f.write("# M9 P9 扰动预测（§4.5；判据带只读 m9_behavior_reference.csv perturbation 段）\n")
        f.write("# 槽级预测与实测逐条留档（含未命中项）；不显著/不符方向如实记录，未做选择性报告\n")
        w = _csv.writer(f, lineterminator="\n")
        w.writerows(rows)
    log("→ %s" % OUT_CSV)
    make_plot(meas, hit_rate, pool_rate, hit_min, eff, crit, POOLS)
    log("→ %s" % OUT_PNG)
    log("完成标记 M9_P9_DONE verdict=%s hit_rate=%.4f" % (verdict, hit_rate))
    return 0 if verdict == "PASS" else 2


def json_dumps(o) -> str:
    import json
    return json.dumps(o, ensure_ascii=False)


def make_plot(meas, hit_rate, pool_rate, hit_min, eff, crit, pools):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    ax = axes[0][0]
    ax.bar(["anchor-level", "pool-level"], [hit_rate, pool_rate],
           color=["#2ca02c" if hit_rate >= hit_min else "#d62728", "#1f77b4"])
    ax.axhline(hit_min, color="red", ls="--", lw=1, label="≥%.2f" % hit_min)
    ax.set_ylim(0, 1.05); ax.set_ylabel("hit rate")
    ax.set_title("(a) prediction hit rate (pre-registered predictions)")
    ax.legend()
    ax = axes[0][1]
    keys = [k for k in pools]
    x = np.arange(len(keys))
    tgt = [meas[k]["d_silence_partner"] for k in keys]
    ctl = [meas[k]["d_random_silence_partner"] for k in keys]
    act = [meas[k].get("d_activation_partner", float("nan")) for k in keys]
    ax.bar(x - 0.25, tgt, 0.25, label="silence: partner Δ/base", color="#d62728")
    ax.bar(x, ctl, 0.25, label="silence: random control", color="#7f7f7f")
    ax.bar(x + 0.25, act, 0.25, label="activation: partner Δ/base", color="#1f77b4")
    ax.axhline(0.05, color="k", ls=":", lw=1); ax.axhline(-0.05, color="k", ls=":", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(keys, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("relative Δ"); ax.set_title("(b) downstream (partner-pool) consequence + controls")
    ax.legend(fontsize=8)
    ax = axes[1][0]
    if eff:
        ax.bar([k for k, _ in eff], [v for _, v in eff], color="#2ca02c")
    ax.axhline(0, color="k", lw=1)
    ax.set_ylabel("relative Δ of target pool rate")
    ax.set_title("(c) manipulation efficacy: activation → target pool ↑")
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    ax = axes[1][1]
    ks = list(crit.keys())
    ax.barh(ks, [1 if crit[k] else 0 for k in ks],
            color=["#2ca02c" if crit[k] else "#d62728" for k in ks])
    ax.set_xlim(0, 1.2); ax.set_title("(d) P9 criteria (pass=1)")
    ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("M9 P9 type-level perturbation prediction (real FlyWire whole-brain; segmented calls)")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=110)


if __name__ == "__main__":
    sys.exit(main())
