"""P2 验证：多神经元链传播正确性（Brian2 vs NEURON 全链参考解）。

清单 §0 P2 / §5.2：
- 发放次序严格递增：t_PLM < t_AVM < t_DA（node3 首发放）；
- 感觉→中间 EPSP（AVM soma V 减 PLM 发放前基线）vs 参考链 v_avm_soma_mv_3
  norm_rmse < 10%（对齐：PLM 发放时刻归零 + 峰值对齐；同列记录 I0 档
  参考索引 2 的对比值作为透明性补充）；
- 链传导时间（PLM 首发放 → DA 首发放）vs 参考链 chain_time_ms[3]
  误差 < 15% 或绝对 < 2 ms；
- 拓扑/极性断言：chain_summary() 3 化学突触 + 2 肌肉驱动 + VB 张力 > 0。
输出：reports/neuro/m3_p2_chain.png + data/m3_p2_chain.csv
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.src.reflex_arc import ReflexArc  # noqa: E402
from neural_exploration.tools.synapse_metrics import norm_rmse  # noqa: E402

DATA_DIR = os.path.join(ROOT, "neural_exploration", "data")
REPORTS_DIR = os.path.join(ROOT, "neural_exploration", "reports", "neuro")
CSV_PATH = os.path.join(DATA_DIR, "m3_reflex_params.csv")
REF_NPZ = os.path.join(DATA_DIR, "m3_reflex_ref.npz")
REPORT_PNG = os.path.join(REPORTS_DIR, "m3_p2_chain.png")
REPORT_CSV = os.path.join(DATA_DIR, "m3_p2_chain.csv")

PSP_WINDOW_MS = 25.0       # 对齐后比较窗口（PLM 发放归零后 0–25ms）
NRMSE_THRESHOLD = 0.10     # 判据：norm_rmse < 10%
CHAIN_REF_INDEX = 3        # 清单 §5.2 指定的参考链索引（v_avm_soma_mv_3 / chain_time_ms[3]）
CHAIN_REF_INDEX_I0 = 2     # 真 I0 档参考索引（intensities[2]=1.0），透明性补充


def _align_psp(t_sim, v_sim, t_plm_sim, t_ref, v_ref, t_plm_ref):
    """PLM 发放归零 + 峰值对齐 → 公共网格上的 (t, sim, ref)。

    基线 = PLM 发放前 [−3, −0.1]ms 中位数（消除 HH 静息漂移，M1 L3）；
    峰值对齐取 |v − 基线| 极值（含 AVM 自身尖峰上冲，两引擎同 HH 模型）。
    """
    def rel(t, v, t_spike):
        m = (t >= t_spike - 1.0) & (t <= t_spike + PSP_WINDOW_MS)
        base_win = (t >= t_spike - 3.0) & (t < t_spike - 0.1)
        base = float(np.median(v[base_win])) if base_win.sum() else float(v[0])
        return t[m] - t_spike, v[m] - base

    t0, v0 = rel(t_sim, v_sim, t_plm_sim)
    t1, v1 = rel(t_ref, v_ref, t_plm_ref)
    i0, i1 = int(np.argmax(np.abs(v0))), int(np.argmax(np.abs(v1)))
    t0, t1 = t0 - t0[i0], t1 - t1[i1]
    t_common = np.arange(0.0, PSP_WINDOW_MS + 0.005, 0.01)
    a = np.interp(t_common, t0, v0)
    b = np.interp(t_common, t1, v1)
    return t_common, a, b


def run_p2(save_plot: bool = True) -> dict:
    ref = np.load(REF_NPZ, allow_pickle=True)
    t_ref = ref["t_ms"]
    intens = ref["intensities"]

    arc = ReflexArc(csv_path=CSV_PATH)   # 确定性默认 p=1/n=1
    r = arc.run(intensity=1.0)

    # ---- 发放次序严格递增 ----
    t_plm = r.spikes("PLM", "node3")
    t_avm = r.spikes("AVM", "node3")
    t_da = r.spikes("DA", "node3")
    order_ok = bool(
        len(t_plm) >= 1 and len(t_avm) >= 1 and len(t_da) >= 1
        and t_plm[0] < t_avm[0] < t_da[0]
    )
    firsts = dict(plm=float(t_plm[0]) if len(t_plm) else None,
                  avm=float(t_avm[0]) if len(t_avm) else None,
                  da=float(t_da[0]) if len(t_da) else None)

    # ---- 链传导时间 vs 参考 ----
    chain_sim = float(t_da[0] - t_plm[0])
    chain_ref = float(ref["chain_time_ms"][CHAIN_REF_INDEX])
    chain_err_rel = abs(chain_sim - chain_ref) / chain_ref
    chain_ok = bool(chain_err_rel < 0.15 or abs(chain_sim - chain_ref) < 2.0)
    chain_ref_i0 = float(ref["chain_time_ms"][CHAIN_REF_INDEX_I0])

    # ---- 感觉→中间 PSP vs 参考（索引 3 主判据；索引 2=I0 补充）----
    t_c, a, b = _align_psp(r.t_ms, r.v_mv["avm_soma"], t_plm[0],
                           t_ref, ref[f"v_avm_soma_mv_{CHAIN_REF_INDEX}"],
                           ref[f"spike_times_plm_{CHAIN_REF_INDEX}"][0])
    rmse3 = norm_rmse(a, b, 0.01)
    # I0 档补充对比
    t_c2, a2, b2 = _align_psp(r.t_ms, r.v_mv["avm_soma"], t_plm[0],
                              t_ref, ref[f"v_avm_soma_mv_{CHAIN_REF_INDEX_I0}"],
                              ref[f"spike_times_plm_{CHAIN_REF_INDEX_I0}"][0])
    rmse2 = norm_rmse(a2, b2, 0.01)
    psp_ok = bool(rmse3 < NRMSE_THRESHOLD)

    # ---- 拓扑/极性断言（chain_summary）----
    s = arc.chain_summary()
    topo_ok = bool(
        s["roles"] == ["PLM", "AVM", "DA", "VB"]
        and s["n_chemical"] == 3 and s["n_muscle_drives"] == 2
        and s["synapse_types"] == {"PLM->AVM": "ampa", "AVM->DA": "ampa", "AVM->VB": "gaba"}
        and s["tonic_uA_cm2"].get("VB", 0.0) > 0
    )

    pass_ = bool(order_ok and chain_ok and psp_ok and topo_ok)

    if save_plot:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
        # 1) Brian2 vs NEURON 参考 PSP（对齐后）
        ax = axes[0]
        ax.plot(t_c, a, lw=1.3, color="#1f77b4", label="Brian2 (rk4)")
        ax.plot(t_c, b, lw=1.1, ls="--", color="k", label="NEURON cvode (ref idx 3)")
        ax.axvline(0, color="gray", ls=":", lw=0.8)
        ax.set_title(f"AVM soma PSP vs reference: normRMSE={rmse3*100:.2f}% "
                     f"(I0 ref idx2: {rmse2*100:.2f}%)")
        ax.set_xlabel("t − PLM spike (ms, peak-aligned)")
        ax.set_ylabel("ΔV (mV)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        axins = ax.inset_axes([0.55, 0.55, 0.4, 0.35])
        axins.plot(t_c, a - b, lw=0.8, color="#d62728")
        axins.set_title("residual", fontsize=8); axins.grid(alpha=0.3)
        # 2) 链发放栅栏图（Brian2 实测）+ 参考链时刻
        ax = axes[1]
        roles = ["PLM", "AVM", "DA", "VB"]
        colors = {"PLM": "#1f77b4", "AVM": "#ff7f0e", "DA": "#d62728", "VB": "#9467bd"}
        for j, role in enumerate(roles):
            st = r.spikes(role, "node3")
            ax.eventplot(st, lineoffsets=j, linewidths=2, colors=colors[role])
            rl = role.lower()
            ref_sp = ref.get(f"spike_times_{rl}_{CHAIN_REF_INDEX}")
            if ref_sp is not None and len(ref_sp):
                ax.plot(ref_sp, np.full(len(ref_sp), j + 0.32), ls="none", marker="v",
                        ms=4, color=colors[role], alpha=0.5,
                        label=f"{role} ref" if j == 0 else None)
        touch = r.meta["touch_start_ms"]; dur = r.meta["touch_dur_ms"]
        ax.axvspan(touch, touch + dur, color="orange", alpha=0.15, label="touch")
        ax.set_yticks(range(len(roles)))
        ax.set_yticklabels(roles)
        ax.set_xlabel("t (ms)")
        ax.set_title(f"chain order: {firsts['plm']:.2f} < {firsts['avm']:.2f} < "
                     f"{firsts['da']:.2f} ms; chain time {chain_sim:.2f} vs ref "
                     f"{chain_ref:.2f} ms")
        ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3)
        fig.suptitle("P2: multi-neuron chain propagation vs NEURON reference", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(REPORT_PNG, dpi=150)
        plt.close(fig)

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(REPORT_CSV, "w", encoding="utf-8") as f:
        f.write("metric,value,pass\n")
        f.write(f"t_plm_ms,{firsts['plm']},{order_ok}\n")
        f.write(f"t_avm_ms,{firsts['avm']},{order_ok}\n")
        f.write(f"t_da_ms,{firsts['da']},{order_ok}\n")
        f.write(f"order_strict_increasing,{int(order_ok)},,\n")
        f.write(f"chain_time_sim_ms,{chain_sim:.5f},,\n")
        f.write(f"chain_time_ref_ms,{chain_ref:.5f},,\n")
        f.write(f"chain_time_ref_i0_ms,{chain_ref_i0:.5f},,\n")
        f.write(f"chain_err_rel,{chain_err_rel:.5f},{chain_ok}\n")
        f.write(f"psp_norm_rmse_idx3,{rmse3:.6f},{psp_ok}\n")
        f.write(f"psp_norm_rmse_idx2_i0,{rmse2:.6f},,\n")
        f.write(f"n_chemical,{s['n_chemical']},{topo_ok}\n")
        f.write(f"n_muscle_drives,{s['n_muscle_drives']},{topo_ok}\n")
        f.write(f"synapse_types,{s['synapse_types']},{topo_ok}\n")
        f.write(f"vb_tonic_uA_cm2,{s['tonic_uA_cm2'].get('VB', 0.0)},{topo_ok}\n")
        f.write(f"pass,{int(pass_)},,\n")

    return dict(
        pass_=pass_,
        first_spikes_ms=firsts, order_ok=order_ok,
        chain_time_sim_ms=chain_sim, chain_time_ref_ms=chain_ref,
        chain_time_ref_i0_ms=chain_ref_i0, chain_err_rel=chain_err_rel,
        chain_ok=chain_ok,
        psp_norm_rmse_idx3=rmse3, psp_norm_rmse_idx2_i0=rmse2,
        psp_ok=psp_ok, topo_ok=topo_ok,
        topology=dict(n_chemical=s["n_chemical"], n_muscle_drives=s["n_muscle_drives"],
                      synapse_types=s["synapse_types"],
                      vb_tonic_uA_cm2=s["tonic_uA_cm2"].get("VB", 0.0)),
        report_png=REPORT_PNG, report_csv=REPORT_CSV,
    )


if __name__ == "__main__":
    import json
    res = run_p2()
    print(json.dumps(res, indent=2, ensure_ascii=False, default=str))
    print("P2 PASS" if res["pass_"] else "P2 FAIL")
