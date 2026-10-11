"""J1 产物 → **E-M9-4 影响量级表**（供 M10 勘误节与验收记录引用）。

输入：`data/m11_m10_defect_audit.json`（主agent 托管作业 J1 的产物；
      `tools/audit_m10_point_defect.py`，N=1000 / T=10 s / 3 seed / λ∈{0,1,4,8}）。
输出：`data/m11_m10_defect_impact_table.csv` + 本文件打印的 markdown 表。

口径（必须与引用者一致）：
  - 两档 = **REF_FROZEN**（冻结件实收语义 = M10 的 point 非 gaba 路径，逐位可复现）与
    **REF_FIX**（文档语义；与 M10 的 `gaba_split=True, frac_slow=0` 路径逐位一致）；
  - **同一网络/同一事件表/同一 seed** ⇒ 差值即"缺陷档"的净影响；
  - 本审计的绝对水平**不等于 M10 P5 的工作点**（bias/权重不同）⇒ **只引用相对差**，
    绝对水平不可与 M10 P5 数值直接比较（引自 M10 P5 的数值须另标"缺陷档"）。
"""
from __future__ import annotations

import csv
import json
import os

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def main() -> int:
    with open(os.path.join(DATA, "m11_m10_defect_audit.json")) as f:
        d = json.load(f)
    rows = d["rows"]
    lams = sorted({r["lam"] for r in rows})
    out = []
    for lam in lams:
        fr = [r for r in rows if r["lam"] == lam and r["form"] == "m10_frozen"]
        fx = [r for r in rows if r["lam"] == lam and r["form"] == "semantic_fix"]
        if not fr or not fx:
            continue
        def m(a, k):
            return sum(x[k] for x in a) / len(a)
        row = dict(
            lam=lam, n_seeds=len(fr),
            frozen_pop_hz=round(m(fr, "pop_hz"), 6), fix_pop_hz=round(m(fx, "pop_hz"), 6),
            pop_rel_diff_pct=round(100.0 * (m(fx, "pop_hz") - m(fr, "pop_hz"))
                                   / max(1e-12, m(fr, "pop_hz")), 3),
            frozen_spk=int(sum(x["spk_total"] for x in fr)),
            fix_spk=int(sum(x["spk_total"] for x in fx)),
            spk_rel_diff_pct=round(100.0 * (sum(x["spk_total"] for x in fx)
                                            - sum(x["spk_total"] for x in fr))
                                   / max(1.0, sum(x["spk_total"] for x in fr)), 3),
            frozen_silent_frac=round(m(fr, "silent_frac"), 5),
            fix_silent_frac=round(m(fx, "silent_frac"), 5),
            silent_diff_pp=round(abs(m(fx, "silent_frac") - m(fr, "silent_frac")) * 100, 4),
            frozen_rate_median_hz=round(m(fr, "rate_median_hz"), 4),
            fix_rate_median_hz=round(m(fx, "rate_median_hz"), 4),
            n_nan_frozen=int(sum(x["n_nan"] for x in fr)),
            n_nan_fix=int(sum(x["n_nan"] for x in fx)))
        out.append(row)
    path = os.path.join(DATA, "m11_m10_defect_impact_table.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()), lineterminator="\n")
        w.writeheader()
        for r in out:
            w.writerow(r)
    print("| λ× | 冻结档 pop (Hz) | 修正档 pop (Hz) | 群率相对差 | 冻结/修正脉冲 | 脉冲相对差 | "
          "静默比差 (pp) | 中位率 冻结/修正 |")
    print("|---|---|---|---|---|---|---|---|")
    for r in out:
        print("| %.1f | %.4f | %.4f | **%+.3f%%** | %d / %d | **%+.3f%%** | %.4f | %.3f / %.3f |" % (
            r["lam"], r["frozen_pop_hz"], r["fix_pop_hz"], r["pop_rel_diff_pct"],
            r["frozen_spk"], r["fix_spk"], r["spk_rel_diff_pct"], r["silent_diff_pp"],
            r["frozen_rate_median_hz"], r["fix_rate_median_hz"]))
    print()
    print("自持比（λ×0 脉冲 / λ×1 脉冲）：冻结档 %.4f vs 修正档 %.4f（相对差 %+.2f%%）" % (
        d["summary"]["m10_frozen_self_sustain_ratio"], d["summary"]["semantic_fix_self_sustain_ratio"],
        100.0 * (d["summary"]["semantic_fix_self_sustain_ratio"]
                 - d["summary"]["m10_frozen_self_sustain_ratio"])
        / max(1e-12, d["summary"]["m10_frozen_self_sustain_ratio"])))
    print("审计配置：N=%s / T=%ss / seeds=%s / λ=%s / 墙钟 %ss / RSS %s MB（主agent 托管 J1）" % (
        d["n"], d["t_s"], d["seeds"], d["lam_grid"], d["elapsed_s"], d["rss_mb"]))
    print("csv ->", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
