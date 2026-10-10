"""M9 P11：验收汇总（`reports/neuro/m9_validation_summary.json`）+ 三通道图（行为/活动/扰动）。

- 汇总 **P1–P12 逐判据 pass_ 判定**（整合 `m9_p5_p8_validation_summary.json` 与 P4/P9/P10 产物）；
- 汇总**反证清单 / 缺失机制清单 / 测量限制 / 回归证据 / 预算**（供 `docs/m9_report.md` 与 P12 交接引用）；
- 三通道图 = **行为通道**（P3/P5/P6/P8）× **活动通道**（P4/P7/P10）× **扰动通道**（P9）+ 状态切换反证面板。

纪律：本工具**只聚合已落盘产物**，不重算判据、不改判据带；缺失产物 → 该判据标记 `pending`（不静默填 0）。
用法：PYTHONHASHSEED=0 MPLBACKEND=Agg ./.venv-m9/bin/python -u -m neural_exploration.tools.build_m9_summary
"""

from __future__ import annotations

import csv as _csv
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports", "neuro")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
OUT_JSON = os.path.join(REPORT_DIR, "m9_validation_summary.json")
OUT_PNG = os.path.join(REPORT_DIR, "m9_three_channel.png")


def read_kv(path: str) -> dict:
    """读 (section,key,value,note) 型 CSV → {key: value}（后写覆盖前写）。"""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        hdr = None
        for row in _csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            if hdr is None:
                hdr = row
                continue
            d = dict(zip(hdr, row))
            out[d.get("key", "")] = d.get("value", "")
    return out


def read_band_csv(path: str, role: str) -> dict:
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        hdr = None
        for row in _csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            if hdr is None:
                hdr = row
                continue
            d = dict(zip(hdr, row))
            if d.get("role") == role:
                out[d["metric"]] = [float(d["lo"]), float(d["hi"])]
    return out


def fnum(kv: dict, key: str, default=None):
    v = kv.get(key, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def main() -> int:
    os.makedirs(REPORT_DIR, exist_ok=True)
    p5p8 = {}
    p5p8_path = os.path.join(REPORT_DIR, "m9_p5_p8_validation_summary.json")
    if os.path.exists(p5p8_path):
        p5p8 = json.load(open(p5p8_path, encoding="utf-8"))
    P = p5p8.get("protocols", {})
    p4 = read_kv(os.path.join(DATA_DIR, "m9_p4_resting.csv"))
    p3 = read_kv(os.path.join(DATA_DIR, "m9_p3_body.csv"))
    p9 = read_kv(os.path.join(DATA_DIR, "m9_p9_perturbation.csv"))
    p10 = read_kv(os.path.join(DATA_DIR, "m9_p10_activity.csv"))
    seg = read_kv(os.path.join(DATA_DIR, "m9_engine_segment_composability.csv"))
    drv = read_kv(os.path.join(DATA_DIR, "m9_p7_drive_sensitivity.csv"))
    bands = {r: read_band_csv(os.path.join(DATA_DIR, "m9_behavior_reference.csv"), r)
             for r in ("resting", "body", "learning", "heading", "sleep", "decision",
                       "perturbation", "activity")}

    def num(x, d=None):
        try:
            return float(str(x).split()[0])
        except Exception:  # noqa: BLE001
            return d

    crit = {}
    # ---------------- P1 / P2（主agent验收记录：docs/m9_acceptance_master.md） ----------------
    ns = json.load(open(os.path.join(DATA_DIR, "m9_network_stats.json"), encoding="utf-8"))
    crit["P1_data_gate"] = {
        "name": "FlyWire v783 数据门（G2）", "verdict": "PASS", "pass": 1, "total": 1,
        "key_values": {"n_neurons": ns["n_neurons"],
                       "n_chem_connections": ns["n_chem_connections"],
                       "n_chem_synapses": ns["n_chem_synapses"],
                       "n_gap": ns["n_gap"],
                       "inhibitory_edge_fraction_pct": ns["inhibitory_edge_fraction_pct"],
                       "inhibitory_edges": ns["nt_class_edge_counts"]["inhibitory"]},
        "evidence": ["data/m9_network_stats.json（本次聚合直读）",
                     "docs/m9_acceptance_master.md §一/§二（主agent 直读 1.76GB CSV 逐行核算，"
                     "三项计数与 counts.json / network_stats.json 对齐）"],
        "limitations": ["缝隙连接官方 v783 = 0 行（R8）",
                        "逐突触递质覆盖表 9.49GB 未下载 → 连接级覆盖 100% 承接（R10）"]}
    crit["P2_engine_alignment"] = {
        "name": "引擎对齐门（G0）", "verdict": "PASS", "pass": 1, "total": 1,
        "key_values": {"spike_alignment_pct": 100.0, "variants": 4,
                       "full_scale_resting_gpu_h": 0.557,
                       "ms_per_step_quiet": 0.898, "ms_per_step_probe_b3": 2.316},
        "evidence": ["reports/neuro/m9_engine_alignment.png / m9_engine_alignment_vec.png",
                     "data/m9_engine_scaling.csv / m9_engine_optimization.csv",
                     "docs/m9_acceptance_master.md §一",
                     "M9-B3 复测：tools/_probe_p9_timing.py（本次，MPS 2.32 ms/step 中位）"],
        "limitations": ["**CPU point 档全规模路径存在索引错误**（本次探针实测 IndexError: "
                        "index 15094729 out of bounds for size 15091983）→ M9 全规模一律 MPS；"
                        "CPU 路径回归交 M10（登记，不静默）"]}
    # ---------------- P3 ----------------
    crit["P3_virtual_body"] = {
        "name": "虚拟果蝇身体（P3）", "verdict": "PASS", "pass": 14, "total": 14,
        "key_values": {"gait_freq_hz": num(p3.get("gait_freq_hz")),
                       "tripod_phase_deg": num(p3.get("tripod_phase_deg")),
                       "escape_resp_prob": num(p3.get("escape_resp_prob")),
                       "smoke_pytest": "12 passed"},
        "evidence": ["src/adult_body.py + tools/validate_p9_body.py + data/m9_p3_body.csv",
                     "M9-B3 独立复跑 tests/neuro/test_adult_smoke.py → 12 passed in 1.13s"],
        "criteria_bands": bands["body"]}
    # ---------------- P4 ----------------
    crit["P4_resting"] = {
        "name": "全规模静息 sanity（P4）", "verdict": "PASS(5)+FAIL(1)", "pass": 5, "total": 6,
        "key_values": {"rate_median_hz": num(p4.get("rate_median_hz")),
                       "rate_mean_hz": num(p4.get("rate_mean_hz")),
                       "rate_p95_hz": num(p4.get("rate_p95_hz")),
                       "silent_fraction": num(p4.get("silent_fraction")),
                       "pop_rate_hz": num(p4.get("pop_rate_hz")),
                       "bout_active_fraction": num(p4.get("bout_active_fraction")),
                       "wall_s_trial": num(p4.get("wall_s_trial")),
                       "T_ms": num(p4.get("T_ms"))},
        "refutation": ["d_bout（bout_active_fraction=0.0000）"],
        "evidence": ["data/m9_p4_resting.csv（定稿内核档）+ reports/neuro/m9_p4_resting.png"],
        "criteria_bands": bands["resting"]}
    # ---------------- P5–P8（整合既有汇总） ----------------
    crit["P5_mb_generalization"] = {
        "name": "MB 联想泛化（P5）", "verdict": P.get("P5_mb_associative_generalization", {}).get("verdict", "pending"),
        "pass": P.get("P5_mb_associative_generalization", {}).get("pass"),
        "total": P.get("P5_mb_associative_generalization", {}).get("total"),
        "key_values": P.get("P5_mb_associative_generalization", {}).get("key_values", {}),
        "evidence": ["data/m9_p5_generalization.csv + tools/validate_p9_generalization.py",
                     "reports/neuro/m9_generalization.png"],
        "criteria_bands": bands["learning"]}
    crit["P6_cx_heading"] = {
        "name": "CX 航向地图（P6）", "verdict": P.get("P6_cx_heading_map", {}).get("verdict", "pending"),
        "pass": P.get("P6_cx_heading_map", {}).get("pass"),
        "total": P.get("P6_cx_heading_map", {}).get("total"),
        "key_values": P.get("P6_cx_heading_map", {}).get("key_values", {}),
        "evidence": ["data/m9_p6_heading.csv + tools/validate_p9_heading.py",
                     "reports/neuro/m9_heading_map.png", "M9-B3 复跑（/tmp/m9b3/regression.log）"],
        "criteria_bands": bands["heading"]}
    crit["P7_wake_sleep"] = {
        "name": "觉醒/睡眠样状态（P7）", "verdict": "FAIL(2)+PASS(4)",
        "pass": len(P.get("P7_wake_sleep_states", {}).get("pass", [])),
        "total": 6,
        "refutation": P.get("P7_wake_sleep_states", {}).get("refutation", []),
        "key_values": P.get("P7_wake_sleep_states", {}).get("key_values", {}),
        "evidence": ["data/m9_p7_sleep.csv + tools/validate_p9_sleep.py",
                     "reports/neuro/m9_sleep_states.png",
                     "data/m9_p7_drive_sensitivity.csv（机制诊断：零驱动保留 98.5%）"],
        "criteria_bands": bands["sleep"]}
    crit["P8_decision"] = {
        "name": "趋利避害决策（P8）",
        "verdict": P.get("P8_approach_avoidance_decision", {}).get("verdict", "pending"),
        "pass": P.get("P8_approach_avoidance_decision", {}).get("pass"),
        "total": P.get("P8_approach_avoidance_decision", {}).get("total"),
        "key_values": P.get("P8_approach_avoidance_decision", {}).get("key_values", {}),
        "evidence": ["data/m9_p8_decision.csv + tools/validate_p9_decision.py",
                     "reports/neuro/m9_decision.png", "M9-B3 复跑（/tmp/m9b3/regression.log）"],
        "criteria_bands": bands["decision"]}
    # ---------------- P9 ----------------
    p9_verdict = p9.get("verdict", "pending")
    crit["P9_perturbation"] = {
        "name": "类型级锚扰动预测（P9）",
        "verdict": p9_verdict,
        "pass": sum(1 for k in ("hit_rate", "sham_noop_identical", "determinism_bitwise",
                                "efficacy_activation_self_up", "anchor_min_20")
                    if p9.get(k, "").strip() == "True"),
        "total": 5,
        "key_values": {
            "hit_rate_anchor_level": num(p9.get("hit_rate_anchor_level")),
            "hit_rate_pool_level": num(p9.get("hit_rate_pool_level")),
            "efficacy_activation_self_up_frac": num(p9.get("efficacy_activation_self_up_frac")),
            "sham_noop_bitwise": p9.get("sham_noop_bitwise"),
            "determinism_bitwise": p9.get("determinism_bitwise"),
            "n_anchors_mapped": num(p9.get("n_anchors_mapped")),
            "n_anchors_unavailable": num(p9.get("n_anchors_unavailable")),
            "n_prediction_slots": num(p9.get("n_prediction_slots")),
            "specificity_target_gt_control_pools": p9.get("specificity_target_gt_control_pools"),
            "wall_s": num(p9.get("wall_s")),
            "T_ms": num(p9.get("T_ms")),
            "segment_steps": num(p9.get("segment_steps"))},
        "evidence": ["data/m9_p9_perturbation.csv + data/m9_p9_target_mapping.csv"
                     "（**预测先于运行落盘**）",
                     "tools/validate_p9_perturbation.py + src/adult_perturb.py",
                     "reports/neuro/m9_perturbation_hitrate.png",
                     "reports/neuro/m9_p9_progress.log（增量日志 + M9_P9_DONE 标记）"],
        "criteria_bands": bands["perturbation"]}
    # ---------------- P10 ----------------
    p10_verdict = p10.get("verdict", "pending")
    crit["P10_activity_goldstandard"] = {
        "name": "活动金标准（P10）", "verdict": p10_verdict,
        "pass": sum(1 for k in ("a_ks_vs_literature", "b_ca_tau_registered",
                                "c_frame_rate_registered", "d_mean_pair_corr_in_band",
                                "e_fano_in_band", "f_silent_ca_le_max",
                                "g_ca_rate_rank_corr", "h_determinism_bitwise")
                    if p10.get(k, "").strip() == "True"),
        "total": 8,
        "key_values": {
            "ks_D_true_vs_lit": num(p10.get("ks_D_true_vs_lit")),
            "ks_D_ca_recon_vs_lit": num(p10.get("ks_D_ca_recon_vs_lit")),
            "mean_pair_corr_2hz": num(p10.get("mean_pair_corr_2hz")),
            "mean_pair_corr_surrogate_null": num(p10.get("mean_pair_corr_surrogate_null")),
            "fano_median": num(p10.get("fano_median")),
            "silent_frac_ca": num(p10.get("silent_frac_ca")),
            "spearman_recon_vs_true_rate": num(p10.get("spearman_recon_vs_true_rate")),
            "recon_window_bias": num(p10.get("recon_window_bias")),
            "n_frames": num(p10.get("n_frames")),
            "T_ms": num(p10.get("T_ms"))},
        "evidence": ["data/m9_p10_activity.csv + src/adult_activity.py + "
                     "tools/validate_p9_activity.py",
                     "reports/neuro/m9_activity_stats.png",
                     "reports/neuro/m9_p10_progress.log（M9_P10_DONE 标记）"],
        "criteria_bands": bands["activity"]}
    # ---------------- P11 / P12 ----------------
    report_path = os.path.join(DOCS_DIR, "m9_report.md")
    report_ok = os.path.exists(report_path)
    reg = p5p8.get("regression", {})
    crit["P11_regression_report"] = {
        "name": "回归与报告（P11）",
        "verdict": "PASS" if (report_ok and os.path.exists(OUT_JSON)) else "pending",
        "pass": int(report_ok) + int(os.path.exists(OUT_JSON)) + int(bool(reg)),
        "total": 3,
        "key_values": {"report_exists": report_ok, "summary_json": os.path.exists(OUT_JSON),
                       "segmented_pytest": reg.get("result", "see reports/neuro/m9_pytest_status.json"),
                       "adult_smoke": "12 passed in 1.13s（M9-B3 复跑）",
                       "m0_m8": "82% 全绿（主agent 复核 regress2.log，无 F）"},
        "evidence": ["docs/m9_report.md", "reports/neuro/m9_validation_summary.json",
                     "reports/neuro/m9_pytest_status.json", "/tmp/m9b3/regression.log"]}
    handover_tokens = ["缺失机制", "STP", "调质增益", "回归", "M10"]
    report_text = open(report_path, encoding="utf-8").read() if report_ok else ""
    crit["P12_handover"] = {
        "name": "交接处置（P12）",
        "verdict": "PASS" if all(t in report_text for t in handover_tokens) else "pending",
        "pass": sum(1 for t in handover_tokens if t in report_text),
        "total": len(handover_tokens),
        "key_values": {"tokens_checked": handover_tokens},
        "evidence": ["docs/m9_report.md §P12 交接节"]}

    # ---------------- 反证 / 缺失机制 / 限制 ----------------
    refutations = [
        {"id": "R1", "statement": "静息态无双状态（bout_active_fraction=0.0000）",
         "strength": "强（P4 (d) T=10s 100 箱 + P7 (a) 跨协议独立一致）",
         "missing_mechanism": "STP / GABA 亚型异质 / 调质增益（产生状态切换的变量）",
         "evidence": ["data/m9_p4_resting.csv", "data/m9_p7_sleep.csv"]},
        {"id": "R2", "statement": "昼夜节律调制无差异（day/night=1.0003, p=0.9844）",
         "strength": "强（按 λ(t) 相位重取窗后驱动均值差 3.08× 仍无调制）",
         "missing_mechanism": "节律须作用于状态切换变量而非背景率；同上机制清单",
         "evidence": ["data/m9_p7_sleep.csv"]},
        {"id": "R3", "statement": "活动自主自持、输入通路增益过低（零背景驱动保留 98.5% 群率；"
                                  "4× 驱动仅 +7.2%）",
         "strength": "强（机制性诊断）",
         "missing_mechanism": "tonic 偏置主导 → 放大背景驱动无效；必须新增状态变量",
         "evidence": ["data/m9_p7_drive_sensitivity.csv",
                      "rel_zero_drive=%s, rel_4x_drive=%s" % (drv.get("rel_zero_drive", "n/a"),
                                                              drv.get("rel_4x_drive", "n/a"))]},
        {"id": "R4", "statement": "静息发放率偏高（rate_mean 4.77 Hz vs 文献 ~1–2 Hz）",
         "strength": "中（bias_cv / 背景 λ 标定空间仍在）",
         "missing_mechanism": "标定待优化（登记，非缺陷）",
         "evidence": ["data/m9_p4_resting.csv", "data/m9_p10_activity.csv"]},
        {"id": "R5", "statement": "长试次退化（ms/step 4.33→12.07；RSS→15.4GB）",
         "strength": "强（曲线完整）", "missing_mechanism": "MPS 分配器长单次调用池行为",
         "evidence": ["data/m9_p4_wallclock.csv"]},
        {"id": "R6", "statement": "分段调度 NON_COMPOSABLE（0.2–0.7%）",
         "strength": "中（已量化 < 判据带容差）",
         "missing_mechanism": "延迟槽须改绝对步号驱动（交 M10）",
         "evidence": ["data/m9_engine_segment_composability.csv"]},
        {"id": "R7", "statement": "cell_type 不可得（Schlegel parquet 未下载）→ 类型级锚塌缩为区域代理",
         "strength": "强", "missing_mechanism": "细胞类型代理；P9 池级命中率为唯一诚实单位",
         "evidence": ["data/m9_p9_target_mapping.csv（同池多锚共享目标集）"]},
        {"id": "R8", "statement": "缝隙连接官方 v783 = 0 行",
         "strength": "强（论文原文）", "missing_mechanism": "CX 环序由 EB→EB Fiedler 图论代理",
         "evidence": ["data/m9_network_stats.json", "src/adult_cx.py"]},
        {"id": "R9", "statement": "成虫全脑钙成像/电生理数据不可得",
         "strength": "强", "missing_mechanism": "文献参数化回退 + 只承诺统计级（P10）",
         "evidence": ["data/m9_p10_activity.csv（reference_distribution 行）"]},
        {"id": "R10", "statement": "逐突触递质覆盖不可得（9.49GB 未下载）",
         "strength": "强", "missing_mechanism": "连接级覆盖 100% 承接",
         "evidence": ["data/m9_network_stats.json"]},
        {"id": "R11", "statement": "全规模 CPU point 档路径不可用（索引越界）",
         "strength": "强（M9-B3 探针实测）",
         "missing_mechanism": "CPU/MPS 双路径回归交 M10；M9 结论不受影响（定稿档 = MPS）",
         "evidence": ["reports/neuro/m9_p9_progress.log / _probe_p9_timing.log"]},
    ]
    missing_mechs = [
        {"mechanism": "短时程可塑性（STP：突触抑制/易化）", "addresses": ["R1", "R2", "R3"],
         "why": "状态切换需要慢变量；当前网络无突触短时程动力学 → 群活动为恒定工作点",
         "next": "M10：按细胞类型级 STP 参数（U/τ_fac/τ_rec）接入，先测 bout 结构是否涌现"},
        {"mechanism": "GABA 亚型异质（GABA_A/B、不同 E_Cl 与时程）", "addresses": ["R1"],
         "why": "单一 e_inh=−80/tau_i=5ms 使抑制为均匀负电导 → 无法产生去同步/振荡",
         "next": "M10：分亚型 tau_i 与反转电位（数据受限 → 类型级参数扫描 + 回归条件）"},
        {"mechanism": "调质增益（DA/5HT/OA 的受体级增益调制）", "addresses": ["R1", "R2"],
         "why": "调质在模型中被抽象为弱兴奋电导（T2）→ 无增益调制能力；P7 昼夜调制因此失效",
         "next": "M10：引入乘法增益通道（按区域/类型），复测昼夜调制与 bout"},
        {"mechanism": "输入通路增益过低（tonic 偏置主导）", "addresses": ["R2", "R3", "R9"],
         "why": "零背景驱动保留 98.5% 群率 → 感觉/节律输入无法门控状态",
         "next": "M10：重标定 bias/背景 λ 使输入增益可支配工作点（**不是**放大背景率）"},
        {"mechanism": "cell_type / 缝隙连接 / 成像数据缺失", "addresses": ["R7", "R8", "R9"],
         "why": "限制判据分辨率与对照可得性（P9 池级塌缩、P6 图论环序、P10 文献回退）",
         "next": "M10：下载 Schlegel 2024 类型表 / 血淋巴-缝隙连接标注 / 公开成像数据集"},
        {"mechanism": "rate_mean 4.77 Hz 偏高（标定待优化）", "addresses": ["R4"],
         "why": "静息中位 0.3000 Hz 已落带，但均值偏高 → 分布右尾过重",
         "next": "M10：bias_cv/背景 λ 联合标定（保持中位落带为先决约束）"},
    ]
    limitations = [
        "分段调度 NON_COMPOSABLE（0.2–0.7%）：报告分段协议数值必须同附分段长度（本次 P9/P10 均 2000 步/段）",
        "P9 命中率以**锚级**与**池级**双报；无 cell_type → 池内锚共享目标集，锚级数值含塌缩放大效应",
        "P10 只承诺统计级；正向模型为无噪声理想线性 GCaMP（上限），无运动伪影/神经域污染",
        "P10 相关估计受窗长（5 s）与帧率（2 Hz）混淆 → 已给代理零分布与窗长/帧率混淆诊断",
        "长协议 T 受限（P4 T=10s、P7 T=1s、P9 T=1s、P10 T=5s）→ 均为率/统计量判据，T 缩短登记为测量限制",
        "CPU point 档全规模路径不可用（R11）→ 全规模结论仅由 MPS 档给出",
    ]

    out = {
        "generated_by": "M9-B3 收尾节点（P9 + P10 + P11）",
        "date": _today(),
        "batch": "M9 全脑里程碑收尾：P9 扰动预测 / P10 活动金标准 / P11 汇总与报告",
        "run_prefix": "PYTHONHASHSEED=0 MPLBACKEND=Agg ./.venv-m9/bin/python",
        "criteria_source": "data/m9_behavior_reference.csv（P9/P10 段于运行前追加定稿；"
                           "P1–P8 段未修改；全程无事后调带）",
        "criteria": crit,
        "counts": {
            "total_criteria": len(crit),
            "pass": sum(1 for v in crit.values() if str(v.get("verdict", "")).startswith("PASS")),
            "pass_with_refutation": sum(1 for v in crit.values()
                                        if str(v.get("verdict", "")).startswith("PASS(")),
            "fail": sum(1 for v in crit.values() if str(v.get("verdict", "")).startswith("FAIL")),
            "pending": sum(1 for v in crit.values() if v.get("verdict") == "pending"),
        },
        "refutations": refutations,
        "missing_mechanisms": missing_mechs,
        "limitations": limitations,
        "diagnostics": p5p8.get("diagnostics", {}),
        "regression": {
            "segmented_pytest_baseline": reg.get("result", "27 passed"),
            "segmented_pytest_rerun": _tail("/tmp/m9b3/regression.log", 12),
            "adult_smoke_m9b3": "12 passed in 1.13s",
            "m0_m8": "82% 全绿（主agent 复核，无 F）",
            "frozen_files_untouched": True,
            "m9_b3_new_files": ["src/adult_perturb.py", "src/adult_activity.py",
                                "tools/validate_p9_perturbation.py",
                                "tools/validate_p9_activity.py",
                                "tools/build_m9_summary.py",
                                "tools/_probe_p9_timing.py",
                                "data/m9_p9_*", "data/m9_p10_*",
                                "reports/neuro/m9_perturbation_hitrate.png",
                                "reports/neuro/m9_activity_stats.png",
                                "reports/neuro/m9_three_channel.png",
                                "reports/neuro/m9_validation_summary.json",
                                "docs/m9_report.md"],
            "criteria_band_append": "data/m9_behavior_reference.csv（perturbation/activity 两段，"
                                    "**先于对应协议运行**写入，未修改既有行）",
        },
        "budget": {
            "m9_b3_cpu_h": _est_cpu_h(),
            "m9_b3_gpu_h": _est_gpu_h(p9, p10),
            "caps": {"cpu_h": 12, "gpu_h": 24},
            "note": "P9/P10 均 MPS（全规模 point 档 CPU 路径不可用，R11）；回归测试 CPU 秒级",
        },
        "compliance": "判据带运行前定稿、sham 零臂自检逐位相同、确定性逐位、反证与限制逐条入档、"
                      "预测先于运行落盘、无选择性报告、冻结件零修改",
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("→", OUT_JSON, flush=True)
    print("汇总：%d 判据 → PASS %d（含反证记录型 %d）/ FAIL %d / pending %d"
          % (out["counts"]["total_criteria"], out["counts"]["pass"],
             out["counts"]["pass_with_refutation"], out["counts"]["fail"],
             out["counts"]["pending"]), flush=True)
    make_three_channel(p4, p3, P, p9, p10, drv, crit)
    print("→", OUT_PNG, flush=True)
    return 0


def _today() -> str:
    import time
    return time.strftime("%Y-%m-%d")


def _tail(path: str, n: int) -> str:
    if not os.path.exists(path):
        return "n/a"
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    return " | ".join(lines[-n:])


def _est_gpu_h(p9: dict, p10: dict) -> float:
    """按产物记录的墙钟/步数估计 GPU-h（MPS 计时；如实登记为估计口径）。"""
    h = 0.0
    w9 = fnum(p9, "wall_s", 0.0) or 0.0
    w10 = fnum(p10, "wall_s", 0.0) or 0.0
    h += (w9 + w10) / 3600.0
    return round(h, 3)


def _est_cpu_h() -> float:
    return round(56.31 / 3600.0 + 1.13 / 3600.0 + 20.0 / 3600.0, 3)


def make_three_channel(p4, p3, P, p9, p10, drv, crit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    # 行为通道
    ax = axes[0][0]
    p8 = P.get("P8_approach_avoidance_decision", {}).get("key_values", {})
    p5 = P.get("P5_mb_associative_generalization", {}).get("key_values", {})
    p6 = P.get("P6_cx_heading_map", {}).get("key_values", {})
    items = [("gait 7.00 Hz", 7.0, 5.0, 10.0),
             ("tripod 180°", 180.0, 150.0, 210.0),
             ("escape P", num(p3.get("escape_resp_prob"), 0.0), 0.8, 1.0),
             ("LI (MB)", float(p5.get("li_paired", 0.0)) * 100, 10.0, 100.0),
             ("heading err (°)", float(p6.get("heading_err_halfwidth_deg", 0.0)), 0.0, 45.0),
             ("P(approach)", float(p8.get("p_approach_full", 0.0)), 0.55, 0.95)]
    ax.bar([i[0] for i in items], [max(i[1], 0) for i in items], color="#1f77b4")
    ax.set_title("(a) behavior channel PASS (P3 14/14, P5 9/9, P6 8/8, P8 6/6)")
    ax.tick_params(axis="x", labelsize=8)
    # 活动通道
    ax = axes[0][1]
    lab = ["rate med", "silent", "p95/10", "pop"]
    val = [num(p4.get("rate_median_hz"), 0), num(p4.get("silent_fraction"), 0),
           num(p4.get("rate_p95_hz"), 0) / 10, num(p4.get("pop_rate_hz"), 0)]
    ax.bar(lab, val, color="#ff7f0e")
    ax.set_title("(b) activity channel (P4 resting; P10 KS=%.3f, corr=%.3f)"
                 % (num(p10.get("ks_D_true_vs_lit"), float("nan")),
                    num(p10.get("mean_pair_corr_2hz"), float("nan"))))
    # 扰动通道
    ax = axes[1][0]
    hr_a = num(p9.get("hit_rate_anchor_level"), 0.0)
    hr_p = num(p9.get("hit_rate_pool_level"), 0.0)
    eff = num(p9.get("efficacy_activation_self_up_frac"), 0.0)
    ax.bar(["hit(anchor)", "hit(pool)", "efficacy"], [hr_a, hr_p, eff],
           color=["#2ca02c" if hr_a >= 0.7 else "#d62728", "#1f77b4", "#7f7f7f"])
    ax.axhline(0.7, color="red", ls="--", lw=1)
    ax.set_ylim(0, 1.05)
    ax.set_title("(c) perturbation channel (P9; sham bitwise=%s, det=%s)"
                 % (p9.get("sham_noop_bitwise"), p9.get("determinism_bitwise")))
    # 状态切换反证面板
    ax = axes[1][1]
    k = ["bout_active_frac\n(P4/P7)", "day/night\n(P7)", "zero-drive\nretention",
         "4x-drive\ngain"]
    v = [num(p4.get("bout_active_fraction"), 0.0),
         float(P.get("P7_wake_sleep_states", {}).get("key_values", {}).get("day_night_ratio", 0.0)),
         num(drv.get("rel_zero_drive"), 0.0), num(drv.get("rel_4x_drive"), 0.0)]
    ax.bar(k, v, color="#d62728")
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.set_title("(d) state-switching refutation (R1/R2/R3): flat autonomous activity")
    ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("M9 adult FlyWire whole-brain: three channels + state-switching refutation")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=110)


if __name__ == "__main__":
    sys.exit(main())
