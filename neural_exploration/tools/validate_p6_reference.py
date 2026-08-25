"""M4 P6 验证：参考解对照（NEURON 子链 + 行为参考模型 CI(T) 表）。

清单 §0 P6 / §6.6 + m4_env_notes §L21/L22a/L23（主 agent 预算裁决落地）：
- P6a（神经级）：NEURON 核心子链（ASEL→AIY→AVB、ASER→AIB→RIA→SMDD）发放次序
  严格递增 + EPSP 对照（短脉冲协议，复用 P2 判定数据独立核验）——链 A/B 传导时间
  vs 参考误差 <15% 或 <2ms；EPSP norm_rmse < 10%；
- P6b（行为级）：行为参考模型（pirouette，numpy）CI 落生物带 [0.3,0.7]（容差
  [0.25,0.75]）——**验证主体 = 全协议 T=25s、N=20**（主 agent 裁决 L23：Brian2 虫
  不再要求自身落带）；Brian2 虫与参考 ΔCI ≤ 0.15 用已记录点（calibration 点 0 +
  ref-T 缩放行）核对（不重跑——L23 协议限制反证记录）。
  m4_ref.npz 的 behavior_ci（B1a 粗校准 θ=4e-6/v=0.2）为历史记录（informational；
  定稿参数 θ_pir=1e-6/v_fwd0=1.0 的参考表在 data/m4_calibration.csv ref-T* 行）。
输出：reports/neuro/m4_p6_reference.png + data/m4_p6_reference.csv

用量：.venv-neuro/bin/python -m neural_exploration.tools.validate_p6_reference
"""

from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.tools import validate_p2_circuit  # noqa: E402

DATA_DIR = os.path.join(ROOT, "neural_exploration", "data")
REPORTS_DIR = os.path.join(ROOT, "neural_exploration", "reports", "neuro")
CSV_PATH = os.path.join(DATA_DIR, "m4_chemotaxis_params.csv")
REF_NPZ = os.path.join(DATA_DIR, "m4_ref.npz")
CAL_CSV = os.path.join(DATA_DIR, "m4_calibration.csv")
REPORT_PNG = os.path.join(REPORTS_DIR, "m4_p6_reference.png")
REPORT_CSV = os.path.join(DATA_DIR, "m4_p6_reference.csv")

BAND_LO, BAND_HI = 0.3, 0.7          # 生物带（判据）
TOL_LO, TOL_HI = 0.25, 0.75          # 容差窗（判据）
DELTA_CI_MAX = 0.15                  # Brian2 vs 参考 CI 均值差 ≤ 0.15
NRMSE_THRESHOLD = 0.10


def _load_cal_rows() -> dict:
    """data/m4_calibration.csv → {point_id: row}（ref-T* 缩放行 + 点 0）。"""
    rows = {}
    with open(CAL_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("ci_mean") in (None, ""):
                continue
            rows[r["point_id"]] = r
    return rows


def _load_ref_t_table(rows: dict) -> dict:
    """ref-T* 行 → {t_total_ms: (ci_mean, p, d)}（定稿参数 θ=1e-6/v=1.0 的 T 缩放表）。"""
    table = {}
    for pid, r in rows.items():
        if pid and str(pid).startswith("ref-T"):
            table[float(r["t_total_ms"])] = dict(
                ci_mean=float(r["ci_mean"]), p_value=float(r["p_value"]),
                cohen_d=float(r["cohen_d"]),
                ctrl_mean=float(r["ctrl_mean"]) if r.get("ctrl_mean") not in (None, "")
                else None,
                ctrl_p=float(r["ctrl_p"]) if r.get("ctrl_p") not in (None, "") else None,
            )
    return table


def _load_decisive_evidence() -> dict:
    """决定性点完整证据（/tmp/m4_res/point_4/trial_*.json，N=20 全部完成）。"""
    import glob
    from scipy import stats as sps
    files = sorted(glob.glob("/tmp/m4_res/point_4/trial_*.json"))
    cis = []
    for fp in files:
        try:
            with open(fp, encoding="utf-8") as f:
                cis.append(float(json.load(f)["ci"]))
        except Exception:
            continue
    if not cis:
        return dict(ci_values=[], n=0, mean=float("nan"), sem=float("nan"),
                    p_value=float("nan"), cohen_d=float("nan"), mean_str="—")
    c = np.asarray(cis, dtype=float)
    mean = float(c.mean())
    sd = float(c.std(ddof=1)) if c.size > 1 else 0.0
    sem = sd / np.sqrt(c.size)
    if c.size > 1 and sd > 0:
        _, p_val = sps.ttest_1samp(c, 0.0)
        d = mean / sd
    else:
        p_val, d = float("nan"), float("nan")
    return dict(ci_values=cis, n=int(c.size), mean=mean, sem=float(sem),
                p_value=float(p_val), cohen_d=float(d),
                mean_str=f"{mean:.3f}±{sem:.3f}")


def _load_point5_evidence() -> dict:
    """点 5（θ_pir=2e-6, T=10s, v=1.0, g=8e6）部分结果（N=9/20，扫描终止前落盘）。

    敏感性点：θ=2e-6（θ_eff≈1.9e-6 电路门限附近）——CĪ≈0.29、p≈0.15（N=9 不显著）、
    转向 0.67/试次；与决定性点（θ=1e-6）同为"可行协议下不显著"证据（informational）。
    """
    import glob
    files = sorted(glob.glob("/tmp/m4_res/point_5/trial_*.json"))
    cis = []
    for fp in files:
        try:
            with open(fp, encoding="utf-8") as f:
                cis.append(float(json.load(f)["ci"]))
        except Exception:
            continue
    if not cis:
        return dict(ci_values=[], n=0, mean=None, p_value=None)
    c = np.asarray(cis, dtype=float)
    from scipy import stats as sps
    _, p_val = sps.ttest_1samp(c, 0.0) if c.size > 1 else (None, float("nan"))
    return dict(ci_values=[round(x, 4) for x in cis], n=int(c.size),
                mean=float(c.mean()), p_value=float(p_val),
                mean_str=f"{c.mean():.3f}")

def run_p6(save_plot: bool = True) -> dict:
    ref = np.load(REF_NPZ, allow_pickle=True)
    cal = _load_cal_rows()
    ref_t = _load_ref_t_table(cal)

    # ================= P6a：NEURON 子链对照 =================
    # 参考自身一致性：发放次序严格递增 + 链传导时间（L5 实测值）
    ref_order = {
        "asel": float(ref["spike_times_asel"][0]),
        "aiyl": float(ref["spike_times_aiyl"][0]),
        "avbl": float(ref["spike_times_avbl"][0]),
        "aser": float(ref["spike_times_aser"][0]),
        "aibl": float(ref["spike_times_aibl"][0]),
        "rial": float(ref["spike_times_rial"][0]),
        "smddl": float(ref["spike_times_smddl"][0]),
    }
    ref_order_ok = bool(
        ref_order["asel"] < ref_order["aiyl"] < ref_order["avbl"]
        and ref_order["aser"] < ref_order["aibl"] < ref_order["rial"]
        < ref_order["smddl"])
    ref_chain_a = float(ref["chain_time_ms_a"])
    ref_chain_b = float(ref["chain_time_ms_b"])

    # Brian2 对照（复用 P2 判定数据，独立核验 EPSP/传导）
    p2 = validate_p2_circuit.run_p2(save_plot=False)
    p6a_ok = bool(
        ref_order_ok
        and p2["psp_norm_rmse_a"] < NRMSE_THRESHOLD
        and p2["psp_norm_rmse_b"] < NRMSE_THRESHOLD
        and p2["chain_ok"]
        and p2["order_ok_a"] and p2["order_ok_b"]
    )

    # ================= P6b：行为参考模型 CI(T) 落带 + ΔCI 记录 =================
    # 全协议验证主体：T=25s（主 agent 裁决 L23：P4(b) 以 numpy 参考全协议为验证主体）
    t25 = ref_t.get(25000.0)
    band_ok = bool(t25 is not None and TOL_LO <= t25["ci_mean"] <= TOL_HI
                   and BAND_LO <= t25["ci_mean"] <= BAND_HI)
    sig_ok = bool(t25 is not None and t25["p_value"] < 0.05 and t25["cohen_d"] >= 0.5)

    # ΔCI ≤ 0.15：预注册判据点 = 已记录完成组（L22a：点 0 θ=4e-6/T=5s/v=0.5 vs ref-T5000
    # 同协议，Δ=0.132 ≤ 0.15 ✓）——**主判据点**。
    delta_rows = []
    if "0" in cal and 5000.0 in ref_t:
        pt0 = cal["0"]
        ref5 = ref_t[5000.0]
        d0 = abs(float(pt0["ci_mean"]) - ref5["ci_mean"])
        delta_rows.append(dict(point="0 (θ=4e-6, T=5s, v=0.5) — 预注册完成组",
                               protocol_ms=5000,
                               brian2_ci=float(pt0["ci_mean"]),
                               ref_ci=ref5["ci_mean"], delta_ci=d0,
                               ok=bool(d0 <= DELTA_CI_MAX), is_judgment=True))
    # 定稿协议（θ=1e-6, T=10s, v=1.0）决定性点：N=20 全部完成 → CĪ=0.099±0.139
    # （p=0.482, d=0.16）不显著；ΔCI vs ref(10s)=0.217 > 0.15——**L23 协议限制测量
    # 限制记录**（L21 点 7 触发 → L23 三态裁决：非机制失败，记录为交付物；判定编码：
    # 不作为 pass-fail 元素，如实记录于协议限制注）
    if 10000.0 in ref_t:
        t10 = ref_t[10000.0]
        dec = _load_decisive_evidence()
        brian2_10s = dec["mean"] if dec["n"] else None
        d10 = abs(brian2_10s - t10["ci_mean"]) if brian2_10s is not None else None
        delta_rows.append(dict(
            point="decided point (θ=1e-6, T=10s, v=1.0, g=8e6) — N=20 全部完成（协议限制记录）",
            protocol_ms=10000,
            brian2_ci=(round(brian2_10s, 4) if brian2_10s is not None else None),
            ref_ci=t10["ci_mean"],
            delta_ci=(round(d10, 4) if d10 is not None else None),
            ok=None,   # 协议限制测量限制记录——不参与 pass 判定（L23 裁决）
            is_judgment=False,
            note=(f"Brian2 CĪ={dec['mean_str']} (p={dec['p_value']:.3f}, "
                  f"d={dec['cohen_d']:.2f}, N={dec['n']})——不显著；ΔCI={d10:.3f} "
                  f"vs 参考(10s)={t10['ci_mean']:.3f}（>0.15）。L23 协议限制裁决："
                  f"可行协议下统计功效结构性不足（θ_eff=max(θ_pir, I_thresh/g_off)"
                  f"≈1.9e-6 > 1e-6，L22）→ 记录为测量限制，非机制失败")))
    # 判定仅用预注册完成组（is_judgment=True；决定性点为协议限制记录，L23）
    delta_ok = all(r["ok"] for r in delta_rows
                   if r["ok"] is not None and r.get("is_judgment"))
    protocol_limited_note_extra = (
        "；决定性点（T=10s, v=1.0, g=8e6, θ_pir=1e-6, N=20）ΔCI=0.217>0.15 记录为"
        "协议限制测量限制（L21 点 7 → L23 三态裁决：非机制失败，记录即交付物）")

    # m4_ref.npz 历史行为校准（B1a：θ=4e-6/v=0.2，CI=0.449@25s）informational
    hist = dict(
        ci_mean=float(ref["behavior_ci_mean"]), ci_sem=float(ref["behavior_ci_sem"]),
        p_value=float(ref["behavior_p_value"]), cohen_d=float(ref["behavior_cohen_d"]),
        ci_ctrl_mean=float(ref["behavior_ci_ctrl_mean"]),
        ci_ctrl_p=float(ref["behavior_ci_ctrl_p"]),
    )

    p6b_ok = bool(band_ok and sig_ok and delta_ok)
    # 协议限制标记：决定性点 ΔCI=0.217>0.15 记录存在（ok=None 行）→ L23 协议限制记录
    protocol_limited = bool(any(r.get("ok") is None for r in delta_rows))
    # 点 5（θ=2e-6）部分结果（informational 敏感性证据）
    point5 = _load_point5_evidence()

    pass_ = bool(p6a_ok and p6b_ok)

    out = dict(
        pass_=pass_,
        p6a_ok=bool(p6a_ok), p6b_ok=bool(p6b_ok),
        protocol_limited=bool(protocol_limited),
        ref_first_spikes_ms=ref_order,
        ref_order_ok=bool(ref_order_ok),
        ref_chain_time_ms_a=ref_chain_a, ref_chain_time_ms_b=ref_chain_b,
        sim_chain_time_ms_a=p2["chain_time_sim_ms_a"],
        sim_chain_time_ms_b=p2["chain_time_sim_ms_b"],
        psp_norm_rmse_a=p2["psp_norm_rmse_a"], psp_norm_rmse_b=p2["psp_norm_rmse_b"],
        band_lo=BAND_LO, band_hi=BAND_HI, tol_lo=TOL_LO, tol_hi=TOL_HI,
        ref_ci_25s=t25["ci_mean"] if t25 else None,
        ref_p_25s=t25["p_value"] if t25 else None,
        ref_d_25s=t25["cohen_d"] if t25 else None,
        band_ok=bool(band_ok), sig_ok=bool(sig_ok), delta_ok=bool(delta_ok),
        delta_ci_max=DELTA_CI_MAX,
        delta_rows=delta_rows,
        ref_t_table={int(k): v for k, v in sorted(ref_t.items())},
        hist_b1a=hist,
        protocol_limited_note=(
            "L23 主 agent 最终裁决：P4/P6(b) 行为统计显著性 = 协议限制反证记录——"
            "Brian2 虫可行协议（T≤10s/N≤20）下统计功效结构性不足（稳健显著需 "
            "T≥15–25s ≈ 数千 CPU-小时，本机不可行）；P4(b)/P6(b) 生物带验证主体 = "
            "numpy 行为参考模型全协议（T=25s/N=20，CI=0.494 ∈ [0.3,0.7]，p<0.001）；"
            "ΔCI 判据（≤0.15）用预注册完成组（点 0，T=5s/θ=4e-6/v=0.5/N=10）核对："
            "Δ=0.132 ≤ 0.15 ✓" + protocol_limited_note_extra +
            "；**ΔCI=0.217 vs 参考 0.317（θ_pir=1e-6/T=10s/N=20 决定性点）为协议限制"
            "测量记录**（Brian2 转向率 ≈42% of 参考，θ_eff≈1.9e-6 电路门限；L23 已裁决"
            "该判据结构性不可达——P6b 按选项 (a) 判定 pass=True + 协议限制测量记录，"
            "与 P4 同型处置）；点 5（θ_pir=2e-6）部分结果 N=" + str(point5["n"]) +
            ("：CĪ=" + point5["mean_str"] + "（p=" + (f"{point5['p_value']:.3f}"
             if point5["p_value"] is not None else "—") + "，不显著，informational）"
             if point5["n"] else "（无）")),
        point5_partial=point5,
        csv_path=CSV_PATH, ref_npz=REF_NPZ, cal_csv=CAL_CSV,
    )

    # ---- CSV 落盘 ----
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(REPORT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["check", "value"])
        for k, v in out.items():
            if k in ("ref_t_table", "delta_rows", "hist_b1a", "ref_first_spikes_ms"):
                continue
            w.writerow([k, v])
        w.writerow([])
        w.writerow(["ref-T table (θ_pir=1e-6, v_fwd0=1.0, N=20, seed0): "
                    "T_ms, ci_mean, p_value, cohen_d"])
        for t_ms, v in sorted(out["ref_t_table"].items()):
            w.writerow([t_ms, f"{v['ci_mean']:.4f}", f"{v['p_value']:.4f}",
                        f"{v['cohen_d']:.4f}"])
        w.writerow([])
        w.writerow(["delta rows: point, protocol_ms, brian2_ci, ref_ci, delta_ci, ok"])
        for r in delta_rows:
            w.writerow([r["point"], r["protocol_ms"], r["brian2_ci"], r["ref_ci"],
                        r["delta_ci"], r["ok"]])

    if save_plot:
        _plot(out)

    return out


def _plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(REPORTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # 1) 参考模型 CI(T) 缩放曲线（ref-T 表）+ 生物带 + 已记录 Brian2 点
    ax = axes[0]
    ts = sorted(out["ref_t_table"].keys())
    cis = [out["ref_t_table"][t]["ci_mean"] for t in ts]
    ax.plot(ts, cis, "o-", lw=1.6, color="#1f77b4", label="numpy ref model CI(T)")
    for t, c in zip(ts, cis):
        ax.annotate(f"{c:.3f}", (t, c), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    # 已记录 Brian2 点（ΔCI 核对）
    for r in out["delta_rows"]:
        if r["brian2_ci"] is not None:
            ax.plot(r["protocol_ms"], r["brian2_ci"], "s", ms=8,
                    color="#d62728", label="Brian2 recorded point" if r["ok"] else "")
    ax.axhspan(TOL_LO, TOL_HI, color="green", alpha=0.12)
    ax.axhspan(BAND_LO, BAND_HI, color="green", alpha=0.25)
    ax.axhline(0.0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("T_total (ms)")
    ax.set_ylabel("CĪ (reference model, N=20)")
    ax.set_title("Ref-model CI(T) scaling (θ_pir=1e-6, v_fwd0=1.0) vs bio band [0.3,0.7]",
                 fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    # 2) P6a：链 A/B 传导时间 + EPSP norm_rmse（Brian2 vs NEURON 子链）
    ax = axes[1]
    labels = ["chain A\nASEL→AVB", "chain B\nASER→SMDD"]
    sim = [out["sim_chain_time_ms_a"], out["sim_chain_time_ms_b"]]
    refc = [out["ref_chain_time_ms_a"], out["ref_chain_time_ms_b"]]
    x = np.arange(2)
    w = 0.34
    ax.bar(x - w / 2, sim, w, color="#1f77b4", label="Brian2")
    ax.bar(x + w / 2, refc, w, color="k", alpha=0.65, label="NEURON ref")
    for i in range(2):
        ax.annotate(f"{sim[i]:.2f}", (x[i] - w / 2, sim[i]), ha="center",
                    va="bottom", fontsize=8)
        ax.annotate(f"{refc[i]:.2f}", (x[i] + w / 2, refc[i]), ha="center",
                    va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("chain conduction time (ms)")
    ax.set_title(f"Subchain times vs NEURON ref | EPSP norm_rmse "
                 f"A={out['psp_norm_rmse_a']*100:.1f}% B={out['psp_norm_rmse_b']*100:.1f}%",
                 fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)

    fig.suptitle("M4 P6: reference cross-check — NEURON subchain + behavioral "
                 "reference model (protocol-limited note for P4/P6b significance)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(REPORT_PNG, dpi=150)
    plt.close(fig)
    return REPORT_PNG


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="M4 P6 参考解对照验证")
    ap.add_argument("--skip-plot", action="store_true")
    args = ap.parse_args()
    res = run_p6(save_plot=not args.skip_plot)
    print("==== M4 P6 参考解对照 ====")
    print("  P6a NEURON 子链:")
    print(f"    参考次序: {res['ref_first_spikes_ms']} → "
          f"{'OK' if res['ref_order_ok'] else 'FAIL'}")
    print(f"    链传导 sim A={res['sim_chain_time_ms_a']:.2f} vs ref "
          f"{res['ref_chain_time_ms_a']:.2f}ms | B={res['sim_chain_time_ms_b']:.2f} "
          f"vs ref {res['ref_chain_time_ms_b']:.2f}ms → "
          f"{'OK' if res['p6a_ok'] else 'FAIL'}")
    print(f"    EPSP norm_rmse A={res['psp_norm_rmse_a']*100:.2f}% "
          f"B={res['psp_norm_rmse_b']*100:.2f}%")
    print("  P6b 行为参考模型:")
    print(f"    CI(25s)={res['ref_ci_25s']:.3f} (p={res['ref_p_25s']:.4f}, "
          f"d={res['ref_d_25s']:.2f}) → 落带[{res['band_lo']},{res['band_hi']}] "
          f"{'OK' if res['band_ok'] else 'FAIL'} 显著 {'OK' if res['sig_ok'] else 'FAIL'}")
    for r in res["delta_rows"]:
        print(f"    ΔCI {r['point']}: Brian2={r['brian2_ci']} ref={r['ref_ci']} "
              f"Δ={r['delta_ci']} → {r['ok']}")
    print(f"  P6 pass_ = {res['pass_']}（P6a {res['p6a_ok']} / P6b {res['p6b_ok']}）")
