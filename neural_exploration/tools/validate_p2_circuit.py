"""M4 P2 验证：回路拓扑/递质极性/链传播正确性（Brian2 vs NEURON 子链参考）。

清单 §0 P2 / §6.2（L11/L5 语义落地）：
- CSV 拓扑断言：20 角色齐全、连接数与递质类型与规格一致（chain_summary）；
- 核心子链发放次序严格递增（first_after 语义——AVB/VB 有张力自发发放，L11）：
  链 A：ASEL < AIYL < AVBL < VB（→C_fwd）；链 B：ASER < AIBL < RIAL < SMDDL（→C_left/right）；
- ASE→AIY/AIB EPSP vs NEURON 参考 `norm_rmse < 10%`（psp_amplitudes/norm_rmse 复用；
  对齐 = 发放时刻归零 + 峰值对齐，M3 P2 同款；用短脉冲协议（I=60µA/cm²×5ms@50ms，
  与 NEURON 参考同注入）得到干净单发放）；
- 核心链传导时间 vs 参考误差 < 15% 或绝对 < 2ms（链 A：ASEL→AVBL 参考 5.93ms；
  链 B：ASER→SMDDL 参考 8.90ms；L5）；
- 确定性重跑逐位一致。
输出：reports/neuro/m4_p2_circuit.png + data/m4_p2_circuit.csv

用量：.venv-neuro/bin/python -m neural_exploration.tools.validate_p2_circuit
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.src.chemotaxis_circuit import ChemotaxisCircuit, EXPECTED_ROLES  # noqa: E402
from neural_exploration.tools.synapse_metrics import norm_rmse  # noqa: E402

DATA_DIR = os.path.join(ROOT, "neural_exploration", "data")
REPORTS_DIR = os.path.join(ROOT, "neural_exploration", "reports", "neuro")
CSV_PATH = os.path.join(DATA_DIR, "m4_chemotaxis_params.csv")
REF_NPZ = os.path.join(DATA_DIR, "m4_ref.npz")
REPORT_PNG = os.path.join(REPORTS_DIR, "m4_p2_circuit.png")
REPORT_CSV = os.path.join(DATA_DIR, "m4_p2_circuit.csv")

NRMSE_THRESHOLD = 0.10      # 判据：norm_rmse < 10%
PSP_WINDOW_MS = 25.0        # 对齐后比较窗口（发放归零后 0–25ms，M3 P2 同款）
PULSE_START_MS = 50.0       # 脉冲协议：链 B 与 NEURON 参考同注入时序（STIM_START_MS）
PULSE_START_A_MS = 110.0    # 链 A 相位避让（B2b 实测：pulse@50ms 时 AVBL 张力自发
                            # 发放 54.34ms 恰在 AIYL 驱动发放 54.47ms 前 0.13ms → EPSP
                            # 落入 AVBL 不应期 → 无驱动发放 → first_after=inf；@110ms
                            # AVBL 张力 106.38 → 驱动发放 117.67，链传导 6.19ms vs 参考
                            # 5.93ms（4.4%）——见 m4_env_notes L25）
PULSE_DUR_MS = 5.0          # 参考 IClamp 时长（STIM_DUR_MS）
I_REF_UA_CM2 = 60.0         # 参考注入密度（build_chemotaxis_ref.STIM/I_ASE_UA_CM2）
PULSE_T_TOTAL_MS = 150.0    # 参考子链仿真窗口（SUBCHAIN_T_MS）
# 因果窗驱动尖峰搜索（B2b：链 A 传导时间的正确测法——张力角色的 first_after 会被
# 自发发放污染，改用"上游首发放后 [lo,hi]ms 窗口内的首个驱动发放"，L11 语义落地）
DRIVEN_WIN_LO_MS = 1.5
DRIVEN_WIN_HI_MS = 14.0


def _first_after(spike_times: np.ndarray, t_ref: float) -> float:
    """该角色在 t_ref 之后的首个发放时刻（张力角色用：取上游发放后的下一发放，L11）。"""
    st = np.asarray(spike_times, dtype=float)
    later = st[st > t_ref + 1e-6]
    return float(later[0]) if len(later) else float("inf")


def _first_driven(spike_times: np.ndarray, t_anchor: float) -> float:
    """因果窗驱动尖峰：上游首发放 t_anchor 后 [lo, hi]ms 内的首个下游发放。

    链传导时间 = 该发放 − t_anchor。窗口 [1.5, 14]ms 覆盖生理传导（NEURON 参考
    5.93/8.90ms，Brian2 预期 6–9ms）；张力自发发放若落入窗口（相位碰撞）会误判
    ——B2b 用相位避让脉冲（PULSE_START_A_MS=110）使驱动发放落入窗口（实测 117.67ms）。
    """
    st = np.asarray(spike_times, dtype=float)
    win = st[(st >= t_anchor + DRIVEN_WIN_LO_MS) & (st <= t_anchor + DRIVEN_WIN_HI_MS)]
    return float(win[0]) if len(win) else float("inf")


def _align_psp(t_sim, v_sim, t_spike_sim, t_ref, v_ref, t_spike_ref):
    """发放时刻归零 + 峰值对齐 → 公共网格 (t, sim, ref)（M3 P2 同款）。"""
    def rel(t, v, t_spike):
        m = (t >= t_spike - 1.0) & (t <= t_spike + PSP_WINDOW_MS)
        base_win = (t >= t_spike - 3.0) & (t < t_spike - 0.1)
        base = float(np.median(v[base_win])) if base_win.sum() else float(v[0])
        return t[m] - t_spike, v[m] - base

    t0, v0 = rel(t_sim, v_sim, t_spike_sim)
    t1, v1 = rel(t_ref, v_ref, t_spike_ref)
    i0, i1 = int(np.argmax(np.abs(v0))), int(np.argmax(np.abs(v1)))
    t0, t1 = t0 - t0[i0], t1 - t1[i1]
    t_common = np.arange(0.0, PSP_WINDOW_MS + 0.005, 0.01)
    return t_common, np.interp(t_common, t0, v0), np.interp(t_common, t1, v1)


def _pulse_s_trace(circ, sign: float, dt_ms: float,
                   start_ms: float = PULSE_START_MS) -> np.ndarray:
    """短脉冲 s(t)：±I_ref/g（编码 ON/OFF 阶跃，与 NEURON 参考同注入密度）。

    s = ±I_REF_UA_CM2 / g（g=CSV g_ON/g_OFF）→ 转导电流 I = g·s = 60µA/cm²（参考值）。
    start_ms：链 A 用相位避让时序（PULSE_START_A_MS），链 B 用参考时序（50ms）。
    """
    g = circ.params.transduction.g_on if sign > 0 else circ.params.transduction.g_off
    s_pulse = sign * I_REF_UA_CM2 / g
    n = int(round(PULSE_T_TOTAL_MS / dt_ms))
    s = np.zeros(n)
    i0 = int(round(start_ms / dt_ms))
    i1 = int(round((start_ms + PULSE_DUR_MS) / dt_ms))
    s[i0:i1] = s_pulse
    return s


def _reuse_p2_from_csv() -> dict:
    """M4_REUSE=1 且 CSV 已存在 → 从 data/m4_p2_circuit.csv 读回判定（不重跑 Brian2）。"""
    import ast
    import csv as _csv
    with open(REPORT_CSV, newline="", encoding="utf-8") as f:
        row = next(_csv.DictReader(f))
    out = dict(row)
    for k, v in list(out.items()):
        s = str(v)
        if s.startswith(("[", "{", "'")):
            try:
                out[k] = ast.literal_eval(s)
            except (ValueError, SyntaxError):
                pass
    for k in ("psp_norm_rmse_a", "psp_norm_rmse_b", "psp_peak_sim_a",
              "psp_peak_ref_a", "psp_peak_sim_b", "psp_peak_ref_b",
              "chain_time_sim_ms_a", "chain_time_ref_ms_a",
              "chain_time_sim_ms_b", "chain_time_ref_ms_b",
              "chain_err_rel_a", "chain_err_rel_b"):
        try:
            out[k] = float(out[k])
        except (TypeError, ValueError):
            pass
    for k in ("n_roles", "n_chemical", "n_muscle_drives", "n_ampa",
              "n_gaba", "n_nmda"):
        out[k] = int(out[k])
    for k in ("topo_ok", "params_ok", "order_ok_a", "order_ok_b", "psp_ok",
              "chain_sane_a", "chain_sane_b", "chain_ok", "deterministic",
              "pass_"):
        out[k] = str(out[k]).lower() in ("true", "1")
    return out


def run_p2(save_plot: bool = True) -> dict:
    if os.environ.get("M4_REUSE") and os.path.exists(REPORT_CSV):
        return _reuse_p2_from_csv()

    circ = ChemotaxisCircuit(csv_path=CSV_PATH)   # 确定性默认 p=1/n=1
    ref = np.load(REF_NPZ, allow_pickle=True)

    # ================= 1) CSV 拓扑/极性断言 =================
    s = circ.chain_summary()
    st = s["synapse_types"]
    pairs = [(k.split("->")[0], k.split("->")[1], v) for k, v in st.items()]
    topo_ok = bool(
        len(s["roles"]) == 20 and set(s["roles"]) == set(EXPECTED_ROLES)
        and s["n_chemical"] == 18 and s["n_muscle_drives"] == 6
        and s["n_ampa"] == 14 and s["n_gaba"] == 4 and s["n_nmda"] == 0
        and any(f == "ASEL" and t.startswith("AIY") and ty == "ampa" for f, t, ty in pairs)
        and any(f == "ASER" and t.startswith("AIB") and ty == "ampa" for f, t, ty in pairs)
        and any(f.startswith("AIY") and t.startswith("RIA") and ty == "gaba"
                for f, t, ty in pairs)
        and any(f.startswith("AIB") and t.startswith("RIA") and ty == "ampa"
                for f, t, ty in pairs)
        and any(f.startswith("RIA") and t.startswith("SMDD") and ty == "ampa"
                for f, t, ty in pairs)
        and any(f.startswith("AVB") and t in ("VB", "DB") and ty == "ampa"
                for f, t, ty in pairs)
        and {"fwd", "left", "right"} <= set(s["muscle_channels"].values())
    )
    # 定稿参数核对（L23：θ_pir=1e-6, T=10000ms, v_fwd0=1.0, g=8e6）
    params_ok = bool(
        abs(s["transduction"]["g_on"] - 8.0e6) < 1.0
        and abs(s["transduction"]["g_off"] - 8.0e6) < 1.0
        and abs(s["mech_a"]["theta_pir"] - 1.0e-6) < 1e-12
        and abs(s["body"]["v_fwd0"] - 1.0) < 1e-9
        and s["protocol"]["t_total_ms"] == 10000.0
    )
    topo_ok = topo_ok and params_ok

    # ================= 2) 核心子链发放次序（默认阶跃协议单次运行） =================
    r_step = circ.run()
    r_step2 = circ.run()
    chain_a = circ.chain_from("ASEL", ("fwd",))          # [ASEL, AIYL, AVBL, VB]
    chain_b = circ.chain_from("ASER", ("left", "right"))  # [ASER, AIBL, RIAL, SMDDL]

    def _order_ok(chain, r):
        t_prev_arr = r.spikes(chain[0], "node3")
        if len(t_prev_arr) == 0:
            return False, {}
        t_prev = float(t_prev_arr[0])
        times = {chain[0]: t_prev}
        for role in chain[1:]:
            t_next = _first_after(r.spikes(role, "node3"), t_prev)
            if not (t_next < float("inf") and t_next > t_prev + 1e-6):
                return False, times
            times[role] = t_next
            t_prev = t_next
        return True, times

    order_ok_a, times_a = _order_ok(chain_a, r_step)
    order_ok_b, times_b = _order_ok(chain_b, r_step)
    order_ok = bool(order_ok_a and order_ok_b)

    # ================= 3) EPSP vs NEURON 参考（短脉冲协议，干净单发放） =================
    dt = circ.params.dt_ms
    # 链 A：正向脉冲（相位避让 @110ms）→ ASEL 单发放 → AIYL soma PSP
    s_a = _pulse_s_trace(circ, +1.0, dt, start_ms=PULSE_START_A_MS)
    circ.set_protocol(s_trace=s_a, dt_protocol_ms=dt)
    try:
        r_a = circ.run(t_total_ms=PULSE_T_TOTAL_MS, record=["aiyl_soma"])
    finally:
        circ.clear_protocol()
    # 链 B：负向脉冲（参考时序 @50ms）→ ASER 单发放 → AIBL soma PSP
    s_b = _pulse_s_trace(circ, -1.0, dt, start_ms=PULSE_START_MS)
    circ.set_protocol(s_trace=s_b, dt_protocol_ms=dt)
    try:
        r_b = circ.run(t_total_ms=PULSE_T_TOTAL_MS, record=["aibl_soma"])
    finally:
        circ.clear_protocol()

    t_ref = np.asarray(ref["t_ms"], dtype=float)
    t_asel_ref = float(ref["spike_times_asel"][0])
    t_aser_ref = float(ref["spike_times_aser"][0])
    v_ref_a = np.asarray(ref["v_aiyl_soma_mv"], dtype=float)
    v_ref_b = np.asarray(ref["v_aibl_soma_mv"], dtype=float)

    def _psp_check(t_sim, v_sim, t_spike_sim, v_ref, t_spike_ref):
        t_c, a, b = _align_psp(t_sim, v_sim, t_spike_sim, t_ref, v_ref, t_spike_ref)
        rmse = norm_rmse(a, b, 0.01)
        return rmse, (a, b, t_c)

    t_asel_sim = r_a.spikes("ASEL", "node3")
    t_aser_sim = r_b.spikes("ASER", "node3")
    if len(t_asel_sim) == 0 or len(t_aser_sim) == 0:
        raise RuntimeError(f"脉冲协议未发放：ASEL={len(t_asel_sim)} ASER={len(t_aser_sim)}")
    t_asel_sim, t_aser_sim = float(t_asel_sim[0]), float(t_aser_sim[0])

    rmse_a, (ca, sa, ra) = _psp_check(r_a.t_ms, r_a.v_mv["aiyl_soma"], t_asel_sim,
                                      v_ref_a, t_asel_ref)
    rmse_b, (cb, sb_, rb_) = _psp_check(r_b.t_ms, r_b.v_mv["aibl_soma"], t_aser_sim,
                                        v_ref_b, t_aser_ref)
    psp_ok = bool(rmse_a < NRMSE_THRESHOLD and rmse_b < NRMSE_THRESHOLD)

    # ================= 4) 核心链传导时间 vs 参考 =================
    # 链 A：t_AVBL(驱动尖峰, 因果窗) − t_ASEL（AVB 张力自发发放 → 相位避让脉冲 +
    # 因果窗搜索，B2b L25；NEURON 参考 ASEL→AVBL=5.93ms）
    t_avbl_a = _first_driven(r_a.spikes("AVBL", "node3"), t_asel_sim)
    chain_time_sim_a = t_avbl_a - t_asel_sim if t_avbl_a < float("inf") else float("inf")
    # 链 B：t_SMDDL(驱动尖峰, 因果窗) − t_ASER（无张力角色，参考 ASER→SMDDL=8.90ms）
    t_smddl_b = _first_driven(r_b.spikes("SMDDL", "node3"), t_aser_sim)
    chain_time_sim_b = t_smddl_b - t_aser_sim if t_smddl_b < float("inf") else float("inf")

    chain_ref_a = float(ref["chain_time_ms_a"])
    chain_ref_b = float(ref["chain_time_ms_b"])
    err_a = abs(chain_time_sim_a - chain_ref_a) / chain_ref_a
    err_b = abs(chain_time_sim_b - chain_ref_b) / chain_ref_b
    # 合理性守卫（L11 张力相位：链 A 传导应在生理量级 [3,12]ms，越界 = first_after
    # 被张力自发发放污染 → 标记不通过并记录，不静默放行）
    sane_a = bool(3.0 <= chain_time_sim_a <= 12.0)
    sane_b = bool(3.0 <= chain_time_sim_b <= 12.0)
    chain_ok = bool(sane_a and sane_b
                    and (err_a < 0.15 or abs(chain_time_sim_a - chain_ref_a) < 2.0)
                    and (err_b < 0.15 or abs(chain_time_sim_b - chain_ref_b) < 2.0))

    # ================= 5) 确定性重跑（默认阶跃协议） =================
    det_ok = bool(r_step == r_step2)

    pass_ = bool(topo_ok and order_ok and psp_ok and chain_ok and det_ok)

    out = dict(
        pass_=pass_,
        n_roles=len(s["roles"]), n_chemical=s["n_chemical"],
        n_muscle_drives=s["n_muscle_drives"],
        n_ampa=s["n_ampa"], n_gaba=s["n_gaba"], n_nmda=s["n_nmda"],
        topo_ok=topo_ok, params_ok=params_ok,
        chain_a=chain_a, chain_b=chain_b,
        order_ok_a=order_ok_a, order_ok_b=order_ok_b,
        first_spikes_a={k: round(float(v), 3) for k, v in times_a.items()},
        first_spikes_b={k: round(float(v), 3) for k, v in times_b.items()},
        psp_norm_rmse_a=float(rmse_a), psp_norm_rmse_b=float(rmse_b),
        psp_peak_sim_a=float(sa.max()), psp_peak_ref_a=float(ra.max()),
        psp_peak_sim_b=float(sb_.max()), psp_peak_ref_b=float(rb_.max()),
        chain_time_sim_ms_a=float(chain_time_sim_a),
        chain_time_ref_ms_a=float(chain_ref_a),
        chain_time_sim_ms_b=float(chain_time_sim_b),
        chain_time_ref_ms_b=float(chain_ref_b),
        chain_err_rel_a=float(err_a), chain_err_rel_b=float(err_b),
        chain_sane_a=bool(sane_a), chain_sane_b=bool(sane_b),
        chain_ok=chain_ok, psp_ok=psp_ok, deterministic=bool(det_ok),
        pulse=dict(i_ref_ua_cm2=I_REF_UA_CM2, start_a_ms=PULSE_START_A_MS,
                   start_b_ms=PULSE_START_MS,
                   dur_ms=PULSE_DUR_MS, t_total_ms=PULSE_T_TOTAL_MS,
                   driven_win_ms=[DRIVEN_WIN_LO_MS, DRIVEN_WIN_HI_MS]),
        csv_path=CSV_PATH, ref_npz=REF_NPZ,
    )

    # ---- CSV 落盘 ----
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(REPORT_CSV, "w", encoding="utf-8", newline="") as f:
        import csv as _csv
        w = _csv.DictWriter(f, fieldnames=list(out.keys()))
        w.writeheader()
        w.writerow({k: (v if not isinstance(v, (list, dict)) else str(v))
                    for k, v in out.items()})

    if save_plot:
        _plot(r_a, r_b, ref, out, (ca, sa, ra), (cb, sb_, rb_))

    return out


def _plot(r_a, r_b, ref, out, psp_a, psp_b):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(REPORTS_DIR, exist_ok=True)
    (t_c, sim_a, ref_a), (t_c2, sim_b, ref_b) = psp_a, psp_b
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    # 1) 链 A EPSP（对齐后）sim vs ref
    ax = axes[0]
    ax.plot(t_c, sim_a, lw=1.2, color="#1f77b4", label="Brian2 ASEL→AIYL")
    ax.plot(t_c, ref_a, lw=1.2, color="k", ls="--", label="NEURON ref")
    ax.set_title(f"Chain A EPSP (ASEL→AIYL): norm_rmse = "
                 f"{out['psp_norm_rmse_a']*100:.2f}%", fontsize=9)
    ax.set_xlabel("t − ASEL spike (ms)")
    ax.set_ylabel("V − baseline (mV)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    # 2) 链 B EPSP
    ax = axes[1]
    ax.plot(t_c2, sim_b, lw=1.2, color="#d62728", label="Brian2 ASER→AIBL")
    ax.plot(t_c2, ref_b, lw=1.2, color="k", ls="--", label="NEURON ref")
    ax.set_title(f"Chain B EPSP (ASER→AIBL): norm_rmse = "
                 f"{out['psp_norm_rmse_b']*100:.2f}%", fontsize=9)
    ax.set_xlabel("t − ASER spike (ms)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    # 3) 两条子链发放次序（raster + 连线）
    ax = axes[2]
    chains = [(out["chain_a"], r_a, "#1f77b4", "chain A (ASEL→fwd)"),
              (out["chain_b"], r_b, "#d62728", "chain B (ASER→turn)")]
    for chain, r, color, label in chains:
        prev = None
        for i, role in enumerate(chain):
            st = r.spikes(role, "node3")
            for tk in st[:3]:
                ax.plot(tk, role, "|", color=color, ms=14, mew=2)
            if len(st):
                tk = float(st[0])
                if prev is not None:
                    ax.plot([prev, tk], [chain[i - 1], role],
                            color=color, lw=1.0, alpha=0.5)
                prev = tk
    ax.set_title(f"Subchain propagation: chain A {out['chain_time_sim_ms_a']:.2f}ms "
                 f"(ref {out['chain_time_ref_ms_a']:.2f}) | "
                 f"chain B {out['chain_time_sim_ms_b']:.2f}ms (ref "
                 f"{out['chain_time_ref_ms_b']:.2f})", fontsize=8)
    ax.set_xlabel("t (ms)")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("M4 P2: circuit topology + chain propagation vs NEURON subchain reference",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(REPORT_PNG, dpi=150)
    plt.close(fig)
    return REPORT_PNG


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="M4 P2 回路拓扑/链传播验证")
    ap.add_argument("--skip-plot", action="store_true")
    args = ap.parse_args()
    res = run_p2(save_plot=not args.skip_plot)
    print("==== M4 P2 回路拓扑/链传播 ====")
    print(f"  拓扑: 20 角色 / {res['n_chemical']} 化学 / {res['n_muscle_drives']} 肌肉 "
          f"(ampa {res['n_ampa']}/gaba {res['n_gaba']}) → {'OK' if res['topo_ok'] else 'FAIL'}")
    print(f"  链次序 A {res['chain_a']}: {res['first_spikes_a']} → "
          f"{'OK' if res['order_ok_a'] else 'FAIL'}")
    print(f"  链次序 B {res['chain_b']}: {res['first_spikes_b']} → "
          f"{'OK' if res['order_ok_b'] else 'FAIL'}")
    print(f"  EPSP norm_rmse: A={res['psp_norm_rmse_a']*100:.2f}% "
          f"B={res['psp_norm_rmse_b']*100:.2f}%（<10%）→ "
          f"{'OK' if res['psp_ok'] else 'FAIL'}")
    print(f"  链传导 A: {res['chain_time_sim_ms_a']:.2f} vs ref "
          f"{res['chain_time_ref_ms_a']:.2f}ms (err {res['chain_err_rel_a']*100:.1f}%) | "
          f"B: {res['chain_time_sim_ms_b']:.2f} vs ref "
          f"{res['chain_time_ref_ms_b']:.2f}ms (err {res['chain_err_rel_b']*100:.1f}%) → "
          f"{'OK' if res['chain_ok'] else 'FAIL'}")
    print(f"  确定性: {res['deterministic']}")
    print(f"  P2 pass_ = {res['pass_']}")
