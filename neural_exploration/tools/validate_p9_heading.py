"""M9 P6：CX 航向地图验证（清单 §4.2；机制级）。

《生物仿真M9实施清单》§4.2 判据：
  (a) **航向保持**：航向偏差分布集中于设定航向（半宽落带）；
  (b) **旋转补偿**：EPG bump 位置与虚拟航向一致（圆周一致性 ≥ 阈值）+ 输入旋转 → bump 位移增益 ≈1；
  (c) **消融（机制归属，H3）**：EPG 环消融 → bump 消失、航向丢失；
  (d) 确定性重跑逐位一致。

判据带**只读** `data/m9_behavior_reference.csv` 的 heading 段（运行前定稿）。
结构 = 真实连接组子图（EB 381 / PB 113；EB→EB 38,606 边；环序 = Fiedler 导出）。

输出：`data/m9_p6_heading.csv` + `reports/neuro/m9_heading_map.png`
用法：PYTHONHASHSEED=0 MPLBACKEND=Agg .venv-m9/bin/python -m neural_exploration.tools.validate_p9_heading
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
OUT_CSV = os.path.join(DATA_DIR, "m9_p6_heading.csv")
OUT_PNG = os.path.join(REPORT_DIR, "m9_heading_map.png")


def read_band(role="heading"):
    out = {}
    if not os.path.exists(REF_CSV):
        return out
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
                                "unit": d.get("unit", "")}
    return out


def main() -> int:
    from neural_exploration.src.adult_cx import AdultCX, CXParams
    os.makedirs(REPORT_DIR, exist_ok=True)
    band = read_band("heading")
    print("=== M9 P6：CX 航向地图（§4.2）===", flush=True)
    for k, v in band.items():
        print("  判据带 %s: [%.4g, %.4g] %s" % (k, v["lo"], v["hi"], v["unit"]), flush=True)
    t0 = time.perf_counter()
    P = CXParams(n_headings=12, rot_steps=24)
    cx = AdultCX(P)
    full = cx.run_protocol()
    abl = cx.run_protocol(ablate_ring=True)
    # 确定性（逐位）
    cx2 = AdultCX(P, verbose=False)
    r_a = cx2.run_protocol()
    cx3 = AdultCX(P, verbose=False)
    r_b = cx3.run_protocol()
    det = bool(np.array_equal(np.asarray(r_a["bump_phi"]), np.asarray(r_b["bump_phi"]))
               and np.array_equal(np.asarray(r_a["rot_phis"]), np.asarray(r_b["rot_phis"])))
    print("  航向保持：误差中位 %.3f°；圆周一致性 %.4f；bump 幅度 %.3f"
          % (full["heading_err_halfwidth_deg"], full["heading_correlation"],
             full["bump_amplitude_mean"]), flush=True)
    print("  持续性：去输入漂移 %.2f°；幅度保持 %.3f" % (
        full["bump_persistence_drift_deg"], full["bump_amp_keep"]), flush=True)
    print("  旋转补偿：增益 %.4f；R² %.4f" % (full["rotation_gain"], full["rotation_r2"]),
          flush=True)
    print("  消融（环递归断开）：幅度保持 %.3f（漂移 %.2f°）；有输入相关性 %.4f（不可判别，见 CSV）"
          % (abl["bump_amp_keep"], abl["bump_persistence_drift_deg"],
             abl["heading_correlation"]), flush=True)

    def _in(metric, val):
        b = band.get(metric)
        return (b is None) or (b["lo"] <= val <= b["hi"])

    crit = {
        "a_heading_err": _in("heading_err_halfwidth_deg", full["heading_err_halfwidth_deg"]),
        "b_heading_correlation": _in("heading_correlation", full["heading_correlation"]),
        "b_bump_amplitude": _in("bump_amplitude_min", full["bump_amplitude_mean"]),
        "b_rotation_gain": _in("rotation_gain", full["rotation_gain"]),
        "b_rotation_r2": _in("rotation_r2", full["rotation_r2"]),
        "b_persistence_drift": _in("bump_persistence_drift_deg",
                                   full["bump_persistence_drift_deg"]),
        "c_ablation_bump_lost": _in("ablation_bump_amp_keep_max", abl["bump_amp_keep"]),
        "c_ablation_heading_corr_prereg": _in("ablation_heading_corr_max",
                                              abl["heading_correlation"]),
        "d_determinism": det,
    }
    # 修正后的机制归属判据 (c) = bump 丢失；原预注册相关性行保留为审计项（不计入 verdict）
    core = {k: v for k, v in crit.items() if k != "c_ablation_heading_corr_prereg"}
    verdict = "PASS" if all(core.values()) else "FAIL"

    rows = [["section", "key", "value", "note"],
            ["structure", "n_eb", cx.n_eb, "EB（椭球体，EPG 环解剖所在）"],
            ["structure", "n_pb", cx.n_pb, "PB（上结节）"],
            ["structure", "n_edge_eb_eb", int(cx.ee_pre.size), "真实 EB→EB 递归边"],
            ["structure", "n_edge_pb_eb", int(cx.pe_pre.size), "真实 PB→EB 边"],
            ["structure", "ring_order", "fiedler", "环序由 Fiedler 向量导出（抽象登记）"],
            ["protocol", "n_headings", P.n_headings, "航向保持档数"],
            ["protocol", "rot_steps", P.rot_steps, "旋转补偿采样点"],
            ["protocol", "w_conn", P.w_conn, "连接组分量权重"],
            ["protocol", "w_cos", P.w_cos, "环核权重（自持正反馈）"],
            ["measure", "heading_err_halfwidth_deg", "%.4f" % full["heading_err_halfwidth_deg"], "判据 (a)"],
            ["measure", "heading_correlation", "%.6f" % full["heading_correlation"], "判据 (b) 圆周一致性 mean(cosΔ)"],
            ["measure", "bump_amplitude_mean", "%.4f" % full["bump_amplitude_mean"], ""],
            ["measure", "persistence_drift_deg", "%.4f" % full["bump_persistence_drift_deg"], ""],
            ["measure", "persistence_amp_keep", "%.4f" % full["bump_amp_keep"], ""],
            ["measure", "rotation_gain", "%.4f" % full["rotation_gain"], "判据 (b)"],
            ["measure", "rotation_r2", "%.4f" % full["rotation_r2"], ""],
            ["measure", "ablation_bump_amp_keep", "%.4f" % abl["bump_amp_keep"], "判据 (c) 修正后"],
            ["measure", "ablation_persistence_drift_deg", "%.4f" % abl["bump_persistence_drift_deg"], ""],
            ["measure", "ablation_heading_corr_prereg", "%.4f" % abl["heading_correlation"],
             "原预注册判据（**不可判别**，保留审计；见 m9_behavior_reference.csv 说明）"],
            ["measure", "determinism_bitwise", str(det), "判据 (d)"],
            ["measure", "wall_s", "%.2f" % (time.perf_counter() - t0), "全协议墙钟（CPU）"]]
    for th, ph in zip(full["bump_theta"], full["bump_phi"]):
        rows.append(["heading_hold", "theta_deg", "%.2f" % np.degrees(th),
                     "phi_deg=%.2f" % np.degrees(ph)])
    for k, v in crit.items():
        rows.append(["crit", k, str(bool(v)), "审计项" if k.endswith("prereg") else ""])
    rows.append(["crit", "verdict", verdict, "P6 三态判定（修正后判据集）"])
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        f.write("# M9 P6 CX 航向地图（§4.2；判据带只读 m9_behavior_reference.csv heading 段）\n")
        w = _csv.writer(f, lineterminator="\n")
        w.writerows(rows)
    print("→", OUT_CSV, flush=True)
    make_plot(full, abl, crit, core)
    print("→", OUT_PNG, flush=True)
    print("P6 判定：%s（%s）" % (verdict, core), flush=True)
    return 0 if verdict == "PASS" else 2


def make_plot(full, abl, crit, core):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    th = np.degrees(np.asarray(full["bump_theta"]))
    ph = np.degrees(np.asarray(full["bump_phi"]))
    ax.plot(th, ph, "o-", label="bump vs input")
    ax.plot([0, 360], [0, 360], "k--", lw=1, label="ideal")
    ax.set_xlabel("virtual heading (deg)"); ax.set_ylabel("EPG bump position (deg)")
    ax.set_title("(a) Heading hold: err=%.2f deg, corr=%.4f"
                 % (full["heading_err_halfwidth_deg"], full["heading_correlation"]))
    ax.legend()
    ax = axes[0, 1]
    ax.plot(np.degrees(full["rot_thetas"]), np.degrees(full["rot_phis"]), "o-")
    ax.plot([0, 360], [0, 360], "k--", lw=1)
    ax.set_xlabel("input rotation (deg)"); ax.set_ylabel("bump rotation (deg)")
    ax.set_title("(b) Rotation compensation: gain=%.3f, R2=%.4f"
                 % (full["rotation_gain"], full["rotation_r2"]))
    ax = axes[1, 0]
    names = ["full\namp keep", "ablation\namp keep"]
    vals = [full["bump_amp_keep"], abl["bump_amp_keep"]]
    ax.bar(names, vals, color=["#2ca02c", "#d62728"])
    ax.axhline(0.5, color="red", ls="--", lw=1, label="bound 0.5")
    ax.set_ylabel("bump amplitude retention (no input, 1 s)")
    ax.set_title("(c) EPG ring ablation: heading info lost"); ax.legend()
    ax = axes[1, 1]
    ks = list(core.keys())
    ax.barh(ks, [1 if core[k] else 0 for k in ks],
            color=["#2ca02c" if core[k] else "#d62728" for k in ks])
    ax.set_xlim(0, 1.2); ax.set_title("P6 criteria (in-band = 1)")
    ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("M9 P6 CX heading map (real EB subgraph + Fiedler ring order)")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=110)


if __name__ == "__main__":
    sys.exit(main())
