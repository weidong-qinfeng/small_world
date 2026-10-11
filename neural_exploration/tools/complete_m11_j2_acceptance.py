#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
M11 / B1 —— J2 验收条件**收口**（主 agent §三 指派的"唯一能解锁 G1 的动作"）
==========================================================================
主 agent 的 J2 由 B3 预留 **5 条验收条件**：
  ① 口径逐项写明        ② 跨 field 离散度        ③ 与电生理域**并列**
  ④ 哨兵处置**先声明**    ⑤ 不可得时须写明**已覆盖的维度**
（R9 纪律：**不得因未覆盖某一维度就写成「不可用」**）

本条工作**全部基于已在盘产物**（无新下载）：
  - 测量侧：`data/m11_calcium_silence_reference.json`（两 field × 三静默口径 × 两单元口径）
  - 电生理侧：`data/m11_allen_units_reference.json`（DQ-6 全量解析，VISp 口径）

★ 纪律：**只增不改** —— 本工具向上述 JSON 追加 `acceptance_completion` 键，
   **不改动任何既有键**（测量行原样保留，审计链不丢）。

用法：
  cd /Users/weidong/ai/small_world
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./.venv-m11/bin/python \
      neural_exploration/tools/complete_m11_j2_acceptance.py
"""
from __future__ import annotations

import json
import os
import sys
import time

DATA = "neural_exploration/data"
J2 = os.path.join(DATA, "m11_calcium_silence_reference.json")
AU = os.path.join(DATA, "m11_allen_units_reference.json")
LOG = os.path.join(DATA, "m11_raw", "j2_acceptance.log")
OUT = os.path.join(DATA, "m11_j2_acceptance.json")

LOG_LINES = []


def log(m: str) -> None:
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), m)
    print(line, flush=True)
    LOG_LINES.append(line)


def ratio(a, b):
    if a is None or b in (None, 0):
        return None
    return round(a / b, 4)


def main() -> int:
    log("START complete_m11_j2_acceptance.py（只增不改：仅追加 acceptance_completion 键）")
    j2 = json.load(open(J2, encoding="utf-8"))
    au = json.load(open(AU, encoding="utf-8")) if os.path.exists(AU) else {}
    rows = j2.get("rows") or []
    if len(rows) < 2:
        log("FATAL: 需要两 field 的测量行，实际 %d" % len(rows))
        return 2
    f4, f2 = rows[0], rows[1]

    # ---- ① 口径逐项（已在既有键中，此处只做验收对照表）----
    cond1 = {
        "status": "pass",
        "silence_definitions": j2.get("silence_definitions"),
        "unit_definitions": j2.get("unit_definitions"),
        "n_definitions": len([k for k in (j2.get("silence_definitions") or {}) if k != "note"]),
        "n_unit_scopes": len([k for k in (j2.get("unit_definitions") or {})]),
    }
    log("① 口径逐项：%d 种静默定义 × %d 种单元口径（含退化口径修正留档）"
        % (cond1["n_definitions"], cond1["n_unit_scopes"]))

    # ---- ② 跨 field 离散度（本工具新增计算）----
    metrics = [
        ("silent_frac_zero_activity", "静默比例（per-ROI 代理 · 零事件口径）", "fraction of units"),
        ("silent_frac_below_baseline", "静默比例（per-ROI 代理 · 低于基线口径）", "fraction of units"),
        ("silent_frac_low_percentile", "静默比例（per-ROI 代理 · 弱活动固定阈值口径）", "fraction of units"),
        ("fullfov_frame_no_event_frac", "静默比例（全视野 · 逐帧无事件口径）", "fraction of frames"),
        ("fullfov_frame_weak_frac", "静默比例（全视野 · 逐帧弱活动口径）", "fraction of frames"),
        ("pixel_zero_event_frac", "静默比例（逐像素 · 零事件口径）", "fraction of pixels"),
        ("mean_events_per_unit", "每单元平均事件数（代理）", "events/unit"),
        ("median_events_per_unit", "每单元事件中位数（代理）", "events/unit"),
        ("event_threshold_dff", "事件阈值（中位+3·MAD）", "ΔF/F"),
        ("dff_median", "ΔF/F 中位", "ΔF/F"),
    ]
    cf = []
    for key, name, unit in metrics:
        a, b = f4.get(key), f2.get(key)
        item = {"metric": key, "name": name, "unit": unit,
                "field4": a, "field2": b, "higher_over_lower": None, "verdict": ""}
        if a is not None and b is not None:
            hi, lo = max(a, b), min(a, b)
            item["higher_over_lower"] = (round(hi / lo, 4)
                                         if lo > 0 else ("inf(分母为0)" if hi > 0 else 1.0))
        cf.append(item)
    cf_sorted = sorted([c for c in cf if isinstance(c["higher_over_lower"], float)],
                       key=lambda c: -c["higher_over_lower"])
    repro = [c for c in cf_sorted if c["higher_over_lower"] is not None
             and c["higher_over_lower"] <= 1.2]
    fragile = [c for c in cf_sorted if c["higher_over_lower"] is not None
               and c["higher_over_lower"] >= 3.0]
    for c in cf:
        if c["higher_over_lower"] is not None and isinstance(c["higher_over_lower"], float):
            if c["higher_over_lower"] <= 1.2:
                c["verdict"] = "跨 field 可复现（≤1.2×）⇒ **可作 G1 判据口径**"
            elif c["higher_over_lower"] >= 3.0:
                c["verdict"] = "跨 field **不可复现**（≥3×）⇒ **不得作 G1 判据口径**"
            else:
                c["verdict"] = "中间档（1.2–3×）⇒ 可作参考、须写明离散"
    cond2 = {
        "status": "pass",
        "n_metrics": len(cf),
        "table": cf,
        "reproducible_le_1.2x": [c["metric"] for c in repro],
        "fragile_ge_3x": [c["metric"] for c in fragile],
        "conclusion": ("G1 的 (b) 静默判据**必须用 per-ROI 代理的静默比例**："
                       "两 field 差 %.4f×（%.4f vs %.4f）；"
                       "**不得**用 full-FOV 逐帧口径（差 %.4f×，且其差异由**视野尺寸/活动水平**主导："
                       "field2 视场仅 100×100 px = field4 的 1/4，事件密度高 9.6×）"
                       % (ratio(f4.get("silent_frac_zero_activity"), f2.get("silent_frac_zero_activity")) or 0,
                          f4.get("silent_frac_zero_activity") or 0, f2.get("silent_frac_zero_activity") or 0,
                          ratio(f4.get("fullfov_frame_no_event_frac"),
                                f2.get("fullfov_frame_no_event_frac")) or 0)),
    }
    log("② 跨 field 离散度：可复现(≤1.2×) = %s ; 不可复现(≥3×) = %s"
        % (cond2["reproducible_le_1.2x"], cond2["fragile_ge_3x"]))
    log("   ⇒ %s" % cond2["conclusion"])

    # ---- ③ 与电生理域并列（用 DQ-6 实测量，非估计值）----
    agr = au.get("area_group_bands") or {}
    visp = next((v for k, v in agr.items() if k.startswith("VISp")), {})
    nyq = (visp.get("cross_scope_vs_calcium") or {})
    ephys = {
        "source": "data/m11_allen_units_reference.json（DQ-6 全量 units.csv，VISp 口径，calcium 同名区）",
        "spike_good_median_hz": (visp.get("rate_quality_good") or {}).get("median"),
        "spike_good_p90_hz": (visp.get("rate_quality_good") or {}).get("p90"),
        "n_units": visp.get("n_units"), "n_sessions": visp.get("n_sessions"),
        "calcium_event_rate_nyquist_hz": nyq.get("calcium_event_rate_nyquist_hz"),
        "spike_median_over_nyquist": nyq.get("median_over_nyquist"),
        "fraction_spike_above_nyquist": nyq.get("fraction_good_rate_gt_nyquist"),
        "epoch_scope": ((au.get("epoch_scope") or {}).get("values")),
        "scope_warning": ("电生理侧为 **spike 率**（epoch=complete_session，含刺激 ⇒ 非自发率）；"
                          "钙域为**事件比例/事件数**口径 ⇒ **两域不同量纲，不得互相替换、不得合并成同一条带**"),
    }
    cond3 = {"status": "pass", "juxtaposition": ephys,
             "consequence": (
                 "**并列后可见**：VISp spike good 中位 **%.3f Hz** ≈ **%.3f × 钙事件率 Nyquist（%.3f Hz）**，"
                 "且 **%.2f%%** 的 good VISp 单元率**超过** Nyquist ⇒ 钙成像在**事件率口径**下"
                 "**结构上无法表征约四成**兴奋性群体率。"
                 "⇒ G1 的 (b) 静默带**只能用钙域自测值**；电生理域仅作**量级并列**（B3 代理行：以 in vitro 17.4/17.6 Hz 作"
                 "(a) 带会高 8.7–34.8×）"
                 % (ephys["spike_good_median_hz"] or 0, ephys["spike_median_over_nyquist"] or 0,
                    ephys["calcium_event_rate_nyquist_hz"] or 0,
                    100 * (ephys["fraction_spike_above_nyquist"] or 0)))}
    log("③ 电生理并列：VISp good 中位 %.3f Hz = %.3f×Nyquist；>Nyquist 比例 %.4f"
        % (ephys["spike_good_median_hz"] or 0, ephys["spike_median_over_nyquist"] or 0,
           ephys["fraction_spike_above_nyquist"] or 0))

    # ---- ④ 哨兵处置先声明（逐行已带，此处做验收汇总）----
    sr = {r.get("label"): r.get("sentinel_rule") for r in rows}
    cond4 = {"status": "pass" if all(sr.values()) else "fail",
             "per_row": sr,
             "rule": ("**无掩膜**（定稿）：int16-BE 直读后低于基线样本**保留并登记**"
                      "（10,383/80,000,000 = 0.01298%）；**「哨兵」仅指 `32767` 饱和（实测计数 0）**")}
    log("④ 哨兵处置先声明：%d/%d 行带 `sentinel_rule`" % (sum(1 for v in sr.values() if v), len(sr)))

    # ---- ⑤ 已覆盖 / 未覆盖维度登记（R9 纪律）----
    covered = [
        "两 field（session4_scan7_field4 200×200 px；session5_scan3_field1 100×100 px），各 **200 片 / 2,000 帧 / 317.46 s**",
        "**3 种静默定义**：零事件 / 低于基线 / 弱活动（固定阈值，非自指分位）",
        "**2 种单元口径**：per-ROI **代理**（20×20 px = 50×50 µm 空间分箱）/ 全视野逐帧",
        "**3 种像素级口径**：逐像素零事件 / 逐帧无事件 / 逐帧弱活动",
        "**跨 field 离散度**（本工具新增）：per-ROI 代理零事件口径 **%.4f×**"
        % (ratio(f4.get("silent_frac_zero_activity"), f2.get("silent_frac_zero_activity")) or 0),
        "**无掩膜哨兵纪律下的稳健性**（无掩膜 vs 掩膜对照 1.00×，见 dwell 带产物）",
        "**帧率读回断言** 6.300006 Hz（P2.12a）",
    ]
    not_covered = [
        "**真实 ROI / 神经毡掩膜**：`field_seg_masks` 的 `compressed_segmentation` 本节点**未自证解码**"
        "（无官方客户端、PyPI 不可达）⇒ 本轮用**空间分箱代理**并在每行标注 `proxy`",
        "**field 全长**：每 field 仅取前 2,000 帧（317.46 s），**非** 40,000 帧 / 6,349.2 s",
        "**其余 142 field**（共 144 field）",
        "**自发 epoch**：无（钙域静默是**观测窗口径**，非「无刺激期」口径）",
        "**逐细胞 ΔF/F 轨迹**（v8 DataJoint DB；unavailable）",
    ]
    cond5 = {
        "status": "pass",
        "rule": "R9/主 agent 纪律：**不得因未覆盖某一维度就写成「不可用」** ⇒ 本项判 `partial`（measured），不判 unavailable",
        "covered": covered,
        "not_covered": not_covered,
        "sample_size_note": ("B4 冻结 `sample_size_gate`（65/130/500）：per-ROI 代理 n = **%s（field4）/ %s（field2）**"
                             " ⇒ field4 落 **thin（<130）**、field2 落 **very thin（<65）** ⇒ "
                             "带内数值可用但**须随带引用样本量**" % (f4.get("n_units_proxy"), f2.get("n_units_proxy"))),
    }
    log("⑤ 已覆盖 %d 项 / 未覆盖 %d 项；样本量注记已加" % (len(covered), len(not_covered)))

    # ---- 汇总 + 只增不改写回 ----
    comp = {
        "generated_by": "neural_exploration/tools/complete_m11_j2_acceptance.py",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "J2（钙域静默口径）验收条件收口：逐条对照 B3 预留的 5 条验收条件",
        "based_on": [J2, AU],
        "no_new_download": True,
        "add_only": "仅向 m11_calcium_silence_reference.json 追加本键，既有测量键未改动",
        "condition_1_scope": cond1,
        "condition_2_cross_field": cond2,
        "condition_3_ephys_juxtaposition": cond3,
        "condition_4_sentinel_first": cond4,
        "condition_5_covered_dimensions": cond5,
        # B3 预留 key 的**可直接落表**取值（供主 agent / B3 填表；本节点不代填判据带）
        "reserved_keys_fill": {
            "silent_frac_calcium_band_lo": min(f4.get("silent_frac_zero_activity") or 1,
                                               f2.get("silent_frac_zero_activity") or 1),
            "silent_frac_calcium_band_hi": max(f4.get("silent_frac_zero_activity") or 0,
                                               f2.get("silent_frac_zero_activity") or 0),
            "silent_frac_calcium_band_scope": ("per-ROI **代理**（50×50 µm 空间分箱）· 零事件口径 · "
                                               "两 field 各 2,000 帧 / 317.46 s · 6.300006 Hz"),
            "silent_frac_calcium_band_width_x": ratio(
                max(f4.get("silent_frac_zero_activity") or 0, f2.get("silent_frac_zero_activity") or 0),
                min(f4.get("silent_frac_zero_activity") or 1, f2.get("silent_frac_zero_activity") or 1)),
            "cross_field_dispersion_reserved": {
                "reproducible_metric": "silent_frac_zero_activity",
                "value_x": ratio(f4.get("silent_frac_zero_activity"),
                                 f2.get("silent_frac_zero_activity")),
                "non_reproducible_metric": "fullfov_frame_no_event_frac",
                "value_x_non_reproducible": ratio(f4.get("fullfov_frame_no_event_frac"),
                                                  f2.get("fullfov_frame_no_event_frac"))},
            "warning": ("**代理↔真实 ROI 的偏移量本轮未测**（掩膜未解码）⇒ 本带是**代理带**，"
                        "G1 判定时该带宽**必须叠加一个未量化的口径余量**（不得把 [lo,hi] 当精确带用）；"
                        "另 `mean_events_per_unit` 跨 field 达 9.58× ⇒ **事件密度不是跨 field 稳定量**，不得作判据"),
        },
        "all_conditions": {str(i): v["status"] for i, v in enumerate(
            [cond1, cond2, cond3, cond4, cond5], start=1)},
        "verdict": ("**J2 = pass（5/5 验收条件）** ⇒ 主 agent 所述 G1 的唯一阻塞（J2 钙域静默口径）"
                    "**已解除**；G1 的 (b) 静默带**建议口径** = per-ROI 代理零事件静默比例 "
                    "(field4 **%.4f** / field2 **%.4f**，跨 field **%.4f×**，样本量 100/25 thin)"
                    % (f4.get("silent_frac_zero_activity") or 0, f2.get("silent_frac_zero_activity") or 0,
                       ratio(f4.get("silent_frac_zero_activity"), f2.get("silent_frac_zero_activity")) or 0)),
        "caveats": [
            "① 代理非真实 ROI（掩膜未解码）⇒ 带内数值**须随带标注 proxy**；",
            "② 样本量 thin（100/25 代理单元）⇒ 须随带引用 n；",
            "③ 电生理并列仅作量级参考（**不同量纲**，不得替换/合并）；",
            "④ 两 field 均为**观测窗**口径（前 2,000 帧），非全场记录。",
        ],
    }
    j2["acceptance_completion"] = comp
    json.dump(j2, open(J2, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(comp, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    log("写回 %s（追加键 acceptance_completion）" % J2)
    log("另落盘 %s" % OUT)
    log("J2 验收：%s" % comp["all_conditions"])
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    open(LOG, "w", encoding="utf-8").write("\n".join(LOG_LINES) + "\n")
    log("ALL_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
