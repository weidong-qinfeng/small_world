#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M11 · 活动通道参考量（**只用现阶段可得数据**；四态如实登记）

口径（RF-M11-C + 主 agent DQ-4 裁决）
-------------------------------------
- 时间序列级活动：**Allen BO 2p `events.npz`**（自描述 numpy；本轮到手 1-cell / 4-cell 两件）；
- 属性级同源表（`functional_connectomics` node/edge pkl）：**设计上可达、本轮 `unavailable（吞吐限制）`**
  ⇒ P7/P8/P14 的同源属性对照与其匹配负对照材料（`Connected/ADP/Same region`）**本轮未得**；
- 逐细胞 ΔF/F（v8 DataJoint DB）：`unavailable`（DQ-1/R1）；
- IBL 行为：`NEEDS_TOKEN`（C4）。
**样本量门**：每行必须报 n 与门限；n 不足者标 `thin`/`unavailable`，**不得**记 pass。

用法：PYTHONHASHSEED=0 OMP_NUM_THREADS=1 ./.venv-m11/bin/python neural_exploration/tools/build_m11_activity_reference.py
"""
from __future__ import annotations
import csv, glob, json, os
from datetime import datetime, timezone
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJ = os.path.dirname(ROOT)
DATA = os.path.join(ROOT, "data")
VC2P = os.path.join(DATA, "m11_raw", "allen", "vc2p")
OUT = os.path.join(DATA, "m11_activity_reference.csv")
OUTJ = os.path.join(DATA, "m11_activity_reference.json")


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load(path):
    with np.load(path) as z:
        ev = np.asarray(z["ev.npy"] if "ev.npy" in z.files else z[z.files[0]], dtype=np.float64)
    return ev.reshape(1, -1) if ev.ndim == 1 else ev


def main():
    files = [f for f in sorted(glob.glob(os.path.join(VC2P, "*_events.npz")))]
    evs = {}
    for f in files:
        try:
            evs[os.path.basename(f)] = load(f)
        except Exception:
            pass  # 不完整者不入统计（完整性门）
    rates, ncells, npairs = [], 0, 0
    pair_corrs = []
    for name, ev in evs.items():
        ncells += ev.shape[0]
        # 逐细胞事件率（事件幅度和 / 帧数；帧率 30.9 Hz 主口径，NEEDS_VERIFY）
        rates += list(ev.sum(axis=1) / ev.shape[1] * 30.9)
        if ev.shape[0] >= 2:
            # 成对相关（在 0.914 s 箱上，避免帧级稀疏零导致的相关 = 0）
            evb = ev[:, : (ev.shape[1] // 28) * 28].reshape(ev.shape[0], -1, 28).sum(axis=2)
            C = np.corrcoef(evb)
            iu = np.triu_indices(ev.shape[0], 1)
            # **修正**：np.triu_indices 返回 (row,col) **元组** ⇒ 对数 = len(iu[0])（曾误用 len(iu)=2）
            pair_corrs += list(C[iu]); npairs += int(len(iu[0]))
    rows = [
        dict(ref_id="C1-tuning", metric="取向/方向调谐分布（OSI/DSI/半宽）",
             value_or_band="", n=0, sample_size_gate="需刺激条件**时序**（在 NWB 内，读回受阻）",
             status="unavailable", evidence="unavailable",
             source="", note="同源属性表（node_data: osi_iv/pref_ori）**设计上可达、本轮因吞吐 unavailable**；"
                             "Allen 2p events 无刺激标签 ⇒ 两条路皆不可算 ⇒ 不得写成'已具备'"),
        dict(ref_id="C2-rate", metric="逐细胞事件率（0.914 s 箱外的帧级事件幅度和 × 30.9 Hz，**帧率 NEEDS_VERIFY**）",
             value_or_band="median=%.6f" % (float(np.median(rates)) if rates else 0),
             n=len(rates), sample_size_gate="形状类判据需 n≥130；此处 n=%d" % len(rates),
             status="thin（样本量远不足）" if rates else "unavailable", evidence="measured(thin)",
             source="Allen 2p events（%d 件）" % len(evs),
             note="只作**量级**参考；**不得**作为率分布判据带（n=%d ≪ 130）" % len(rates)),
        dict(ref_id="C2-paircorr", metric="成对相关（0.914 s 箱上，4-cell 件的 6 对）",
             value_or_band="median=%.4f" % (float(np.median(pair_corrs)) if pair_corrs else 0),
             n=npairs, sample_size_gate="需 ≥130 对（或足够配对结构）",
             status="thin（仅 %d 对）" % npairs if npairs else "unavailable", evidence="measured(thin)",
             source="Allen 2p events（4-cell 件）",
             note="有效维度/相位随机化代理地板**未算**（需更多细胞与同管线地板）"),
        dict(ref_id="C3-dwell", metric="状态结构 dwell 中位 / p90÷中位 / 转换率 / AR(1) τ",
             value_or_band="dwell_med=%.2f s; p90/med=%.2f; trans=%.4f Hz; tau=%.2f s"
                           % tuple(json.load(open(os.path.join(DATA, "m11_dwell_reference.json")))["band"][k]
                                   for k in ("low_dwell_median_s", "low_dwell_p90_over_median",
                                             "transition_rate_hz_median", "ar1_tau_s_median")),
             n=json.load(open(os.path.join(DATA, "m11_dwell_reference.json")))["band"]["n_bins_0914s"],
             sample_size_gate="n_bins(0.914s)≥130 ✅（实测 4,070）；low 态完整段≥5 ✅",
             status="measured", evidence="measured（B4 冻结管线）",
             source="data/m11_dwell_reference.json",
             note="**S 门目标带**；形状判据（重尾∧拒指数）两实验均过；**帧率 30.9 Hz 为 NEEDS_VERIFY**（11 Hz 敏感性并列）"),
        dict(ref_id="C4-behavior", metric="正确率 / RT 分布 / psychometric / ramping",
             value_or_band="", n=0, sample_size_gate="—", status="unavailable（NEEDS_TOKEN）",
             evidence="unavailable", source="IBL OpenAlyx（403 需凭据）",
             note="P12 需 token 或公开桶衍生统计（清单 §1.4 回退）"),
    ]
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        f.write("# M11 活动通道参考量（generated_by=tools/build_m11_activity_reference.py；"
                "**四态如实**：measured/thin/unavailable；样本量门逐行标注；不得把 thin 记 pass）\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow(r)
    out = dict(generated_by="tools/build_m11_activity_reference.py", generated_at=utcnow(),
               n_event_files=len(evs), n_cells_total=ncells, n_pairs=npairs,
               rows=rows, evidence_level="mixed（measured/thin/unavailable）")
    json.dump(out, open(OUTJ, "w"), indent=1, ensure_ascii=False)
    print("wrote", OUT, "/", OUTJ, "| cells=%d pairs=%d" % (ncells, npairs))
    for r in rows:
        print("  %-12s %-10s n=%-6s %s" % (r["ref_id"], r["status"], r["n"], r["value_or_band"][:60]))
    return out


if __name__ == "__main__":
    main()
