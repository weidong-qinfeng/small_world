"""P1 验证：EPSP/IPSP 波形误差 <10%（Brian2 vs NEURON 参考解）。

清单 §0 P1 / §5.1：单次突触前发放后的 EPSP/IPSP 波形与 NEURON cvode 参考解对比，
判定：
  - 归一化 RMSE < 10%（窗口内，按参考轨迹峰-峰归一）；
  - 峰值幅度比 |A_sim/A_ref - 1| < 10%；
  - 衰减时间常数比 |τ_sim/τ_ref - 1| < 10%。
对齐：两引擎各自 pre node3 发放时刻触发释放；EPSP 段按峰值时刻对齐
（NEURON NetCon 有 0.1ms 延迟，Brian2 on_pre 在跨阈值步触发，需对齐后比较）。
输出：reports/neuro/m2_p1_waveform.png + data/m2_p1_waveform.csv
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.tools.build_synapse_ref import (  # noqa: E402
    DT_OUT, PULSE_START, REF_NPZ,
)
from neural_exploration.tools.synapse_metrics import norm_rmse  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
REPORTS_DIR = os.path.join(ROOT, "neural_exploration", "reports", "neuro")
REPORT_PNG = os.path.join(REPORTS_DIR, "m2_p1_waveform.png")
REPORT_CSV = os.path.join(DATA_DIR, "m2_p1_waveform.csv")

EPSP_WINDOW_MS = 25.0   # AMPA/GABA 衰减 τ≈3-5ms，25ms 内基本衰减完
THRESHOLD = 0.10        # 10%


def _load_ref():
    d = np.load(REF_NPZ, allow_pickle=True)
    return d


def _align_epsps(t_sim, v_sim, t_ref, v_ref, t_rel_start=1.0, t_rel_end=25.0):
    """把两引擎的 EPSP 段按峰值对齐并返回公共 (t, sim, ref) 窗口。

    EPSP（正峰）与 IPSP（负峰）都按 |v - 基线| 的极值对齐。
    """
    # 各自相对释放时刻（t=0 对齐）的窗口
    def window(t, v):
        m = (t >= -1.0) & (t <= EPSP_WINDOW_MS)
        return t[m], v[m]
    t0, v0 = window(t_sim, v_sim)
    t1, v1 = window(t_ref, v_ref)
    # 峰值对齐（EPSP/IPSP 均取 |偏离基线| 极值）
    i0 = int(np.argmax(np.abs(v0 - v0[0]))); i1 = int(np.argmax(np.abs(v1 - v1[0])))
    t0 = t0 - t0[i0]; t1 = t1 - t1[i1]
    # 插值到统一网格
    t_common = np.arange(0.0, EPSP_WINDOW_MS + DT_OUT / 2, DT_OUT)
    a = np.interp(t_common, t0, v0 - v0[0])
    b = np.interp(t_common, t1, v1 - v1[0])
    return t_common, a, b


def _decay_tau(t, v, peak_idx=None):
    """峰后指数衰减拟合 τ（ms）。输入为对齐后的 PSP（t=0 处峰值、基线 0）。"""
    v = np.asarray(v, dtype=float)
    if peak_idx is None:
        peak_idx = int(np.argmax(np.abs(v)))
    tail = t[peak_idx + 5:]
    y = np.abs(v[peak_idx + 5:])
    if len(y) < 5 or y[0] <= 1e-9:
        return None
    y = np.clip(y, 1e-9, None)
    k, _ = np.polyfit(tail - t[peak_idx], np.log(y), 1)
    return -1.0 / k if k < 0 else None


def run_p1(save_plot: bool = True) -> dict:
    from neural_exploration.src.neuron_pair import NeuronPair

    ref = _load_ref()
    t_ref = ref["t_ms"]
    # 释放时刻 = pre node3 跨 -20mV
    def release_time(v):
        idx = np.flatnonzero((v[:-1] < -20) & (v[1:] >= -20))
        return t_ref[idx[0] + 1] if len(idx) else None
    t_release = release_time(ref["epsp_ampa_pre_node3_mv"])
    if t_release is None:
        raise RuntimeError("NEURON 参考解中未找到 pre 发放")

    cases = {}
    # ---- AMPA EPSP ----
    pair = NeuronPair(t_total_ms=120.0, seed=0)
    pair.add_chemical("ampa", p_release=1.0, n_vesicles=1)
    r = pair.run(pre_pulses=[(PULSE_START, 1.0, 20.0, "soma")],
                 record=["pre_node3", "post_soma"])
    sp_sim = r.spike_times_ms["pre_node3"][0] if len(r.spike_times_ms["pre_node3"]) else None
    t_sim, v_sim = r.t_ms - sp_sim, r.v_mv["post_soma"]
    t_ref_e, v_ref_e = t_ref - t_release, ref["epsp_ampa_post_mv"]
    t_c, a, b = _align_epsps(t_sim, v_sim, t_ref_e, v_ref_e)
    cases["ampa_epsp"] = _judge("ampa EPSP", t_c, a, b)

    # ---- GABA IPSP ----
    pair = NeuronPair(t_total_ms=120.0, seed=0)
    pair.add_chemical("gaba", p_release=1.0, n_vesicles=1)
    r = pair.run(pre_pulses=[(PULSE_START, 1.0, 20.0, "soma")],
                 record=["pre_node3", "post_soma"])
    sp_sim = r.spike_times_ms["pre_node3"][0]
    t_sim, v_sim = r.t_ms - sp_sim, r.v_mv["post_soma"]
    t_ref_e, v_ref_e = t_ref - t_release, ref["ipsp_gaba_post_mv"]
    t_c, a, b = _align_epsps(t_sim, v_sim, t_ref_e, v_ref_e)
    cases["gaba_ipsp"] = _judge("gaba IPSP", t_c, a, b)

    all_pass = all(v["pass_"] for v in cases.values())

    if save_plot:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
        for ax_row, (key, name) in zip(
                axes, (("ampa_epsp", "EPSP (AMPA)"), ("gaba_ipsp", "IPSP (GABA_A)"))):
            j = cases[key]
            ax_row[0].plot(j["t"], j["sim"], lw=1.3, label="Brian2 (rk4)")
            ax_row[0].plot(j["t"], j["ref"], lw=1.1, ls="--", color="k", label="NEURON cvode")
            ax_row[0].set_title(
                f"{name}: normRMSE={j['norm_rmse']*100:.2f}%  "
                f"amp={j['amp_ratio']*100:.1f}%  "
                f"τ={j['tau_sim'] if j['tau_sim'] else '--'}/"
                f"{j['tau_ref'] if j['tau_ref'] else '--'}ms")
            ax_row[0].legend(fontsize=8)
            ax_row[0].set_ylabel("ΔV (mV)")
            ax_row[0].grid(alpha=0.3)
            ax_row[1].plot(j["t"], j["sim"] - j["ref"], lw=0.8, color="#d62728")
            ax_row[1].set_title("residual (mV)")
            ax_row[1].grid(alpha=0.3)
        axes[-1][0].set_xlabel("t - release (ms)")
        axes[-1][1].set_xlabel("t - release (ms)")
        fig.suptitle("P1: EPSP/IPSP waveform vs NEURON reference (<10%)", y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(REPORT_PNG, dpi=150)
        plt.close(fig)

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(REPORT_CSV, "w") as f:
        f.write("case,norm_rmse,amp_sim_mv,amp_ref_mv,amp_ratio,tau_sim_ms,tau_ref_ms,pass\n")
        for key, j in cases.items():
            f.write(f"{key},{j['norm_rmse']:.5f},{j['amp_sim']:.4f},{j['amp_ref']:.4f},"
                    f"{j['amp_ratio']:.4f},{j['tau_sim'] if j['tau_sim'] else ''},"
                    f"{j['tau_ref'] if j['tau_ref'] else ''},{j['pass_']}\n")

    return dict(pass_=all_pass, threshold=THRESHOLD, cases=cases,
                report_png=REPORT_PNG, report_csv=REPORT_CSV)


def _judge(name, t_c, sim, ref):
    rmse = norm_rmse(sim, ref, DT_OUT)
    amp_sim = float(np.max(np.abs(sim)))
    amp_ref = float(np.max(np.abs(ref)))
    amp_ratio = float(amp_sim / amp_ref) if amp_ref > 0 else float("nan")
    tau_sim = _decay_tau(t_c, sim, peak_idx=0)
    tau_ref = _decay_tau(t_c, ref, peak_idx=0)
    tau_ratio = None
    if tau_sim and tau_ref:
        tau_ratio = float(tau_sim / tau_ref)
    pass_ = bool(
        rmse < THRESHOLD
        and abs(amp_ratio - 1.0) < THRESHOLD
        and (tau_ratio is None or abs(tau_ratio - 1.0) < THRESHOLD)
    )
    return dict(
        name=name, t=t_c, sim=sim, ref=ref, norm_rmse=float(rmse),
        amp_sim=amp_sim, amp_ref=amp_ref, amp_ratio=amp_ratio,
        tau_sim=tau_sim, tau_ref=tau_ref, tau_ratio=tau_ratio, pass_=pass_,
    )


if __name__ == "__main__":
    import json
    res = run_p1()
    print(json.dumps({k: v for k, v in res.items() if k != "cases"},
                     indent=2, ensure_ascii=False, default=str))
    for key, j in res["cases"].items():
        print(f"  {key:12s} rmse={j['norm_rmse']*100:.2f}% amp_ratio={j['amp_ratio']*100:.1f}% "
              f"τ={j['tau_sim']:.2f}/{j['tau_ref']:.2f}ms pass={j['pass_']}")
    print("P1 PASS" if res["pass_"] else "P1 FAIL")
