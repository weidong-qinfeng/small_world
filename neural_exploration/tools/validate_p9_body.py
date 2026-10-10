"""M9 P3：虚拟成年果蝇身体验证（清单 §3.3；纯 CPU，无 GPU 依赖）。

《生物仿真M9实施清单》§3.3（身体层验证判据：**先身体模式可算 → 再耦合**，铁律）：
  (a) **每行为通道单通道驱动冒烟**（纯驱动 → 对应运动学响应正确）：
      tripod 相序 / 位移方向 / 翅拍信号 / curl 曲率饱和 / escape 触发；
  (b) **状态分类阈值 CSV 定稿 + 分类与驱动通道语义一致**；
  (c) **巨纤维逃跑反射 sanity**（P-EXT1 机制前置）：光刺激 → 巨纤维 → 跳跃/翅展，
      反应概率 **≥0.8**（承接 M8 P6 逃避基线 sanity 语义）+ 亚阈值副判据；
  (d) **确定性**：同参数重跑**逐位一致**（p=1/n=1）；轨迹有界、无 NaN。

判据带**只读** `data/m9_behavior_reference.csv` 的 body 段（协议运行前定稿，不事后调）；
参数**只读** `data/m9_adult_body_params.csv`。

输出：`data/m9_p3_body.csv` + `reports/neuro/m9_p3_body.png`
用法：PYTHONHASHSEED=0 MPLBACKEND=Agg .venv-m9/bin/python -m neural_exploration.tools.validate_p9_body
"""

from __future__ import annotations

import csv as _csv
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports", "neuro")
REF_CSV = os.path.join(DATA_DIR, "m9_behavior_reference.csv")
OUT_CSV = os.path.join(DATA_DIR, "m9_p3_body.csv")
OUT_PNG = os.path.join(REPORT_DIR, "m9_p3_body.png")

T_SMOKE_MS = 2000.0
N_ESCAPE_TRIALS = 20


def read_band(role="body"):
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
                                "unit": d.get("unit", ""),
                                "provenance": d.get("provenance", "")[:100]}
    return out


def phase_diff_deg(sig_a: np.ndarray, sig_b: np.ndarray, f_hz: float, dt_ms: float):
    """两信号在给定频率上的相位差（deg，取 [0,360)）——FFT 主峰相位差。"""
    n = len(sig_a)
    a = sig_a - sig_a.mean()
    b = sig_b - sig_b.mean()
    if np.allclose(a, 0) or np.allclose(b, 0):
        return float("nan")
    freqs = np.fft.rfftfreq(n, d=dt_ms * 1e-3)
    k = int(np.argmin(np.abs(freqs - f_hz)))
    A = np.fft.rfft(a)[k]
    B = np.fft.rfft(b)[k]
    d = np.angle(B) - np.angle(A)
    return float(np.degrees(d) % 360.0)


def main_peak_freq(sig: np.ndarray, dt_ms: float) -> float:
    n = len(sig)
    a = sig - sig.mean()
    if np.allclose(a, 0):
        return float("nan")
    sp = np.abs(np.fft.rfft(a))
    sp[0] = 0.0
    freqs = np.fft.rfftfreq(n, d=dt_ms * 1e-3)
    return float(freqs[int(np.argmax(sp))])


def main() -> int:
    from neural_exploration.src.adult_body import (AdultFlyBody, AdultBodyParams,
                                                   LEGS, TRIPOD_A, TRIPOD_B,
                                                   state_fractions)
    os.makedirs(REPORT_DIR, exist_ok=True)
    print("=== M9 P3：虚拟成年果蝇身体验证（§3.3）===", flush=True)
    band = {k: v for k, v in read_band("body").items()}
    for k, v in band.items():
        print("  判据带 %s: [%.4g, %.4g] %s" % (k, v["lo"], v["hi"], v["unit"]), flush=True)
    p = AdultBodyParams.from_csv()
    print("身体参数（CSV 定稿）：步频 %.1f Hz / tripod 相位差 %.0f° / 翅拍 %.0f Hz / "
          "球径 %.1f mm / 逃跑阈值 %.2f（潜伏 %.0fms，不应期 %.0fms）"
          % (p.gait_freq_hz, p.tripod_phase_deg, p.wing_freq_hz, p.ball_diameter_mm,
             p.escape_thr, p.escape_latency_ms, p.escape_refractory_ms), flush=True)
    dt = p.dt_ms
    rows = [["section", "key", "value", "note"]]

    b = AdultFlyBody(p, seed=0)
    b.reset()

    # ---------------- (a1) 步态：tripod 相序 + 步频 ----------------
    r_fwd = b.run({"C_leg_fwd": 1.0}, T_SMOKE_MS)
    lift = {leg: r_fwd["joints_deg"][:, i, 1] for i, leg in enumerate(LEGS)}
    f_meas = main_peak_freq(lift["L1"], dt)
    pd_cross = phase_diff_deg(lift[TRIPOD_A[0]], lift[TRIPOD_B[0]], p.gait_freq_hz, dt)
    intra = [phase_diff_deg(lift[a], lift[c], p.gait_freq_hz, dt)
             for a, c in (("L1", "R2"), ("R2", "L3"), ("L1", "L3"))]
    intra = [min(v, 360.0 - v) for v in intra]
    print("  (a1) 步频 %.2f Hz（带 %.1f–%.1f）；组间相位差 %.1f°（目标 %.0f）；"
          "组内最大相位差 %.1f°" % (f_meas, p.gait_freq_hz * 0.7, p.gait_freq_hz * 1.4,
                                    pd_cross, p.tripod_phase_deg, max(intra)), flush=True)

    # ---------------- (a2) 位移方向 / 量级 ----------------
    b.reset(); r_f = b.run({"C_leg_fwd": 1.0}, 1000.0)
    b.reset(); r_bk = b.run({"C_leg_back": 1.0}, 1000.0)
    d_fwd = float(r_f["x"][-1]); d_back = float(r_bk["x"][-1])
    print("  (a2) C_leg_fwd 净位移 %+.3f mm；C_leg_back 净位移 %+.3f mm（应反号）"
          % (d_fwd, d_back), flush=True)

    # ---------------- (a3) 转弯方向 ----------------
    b.reset(); r_tl = b.run({"C_leg_left": 1.0, "C_leg_fwd": 1.0}, 1000.0)
    b.reset(); r_tr = b.run({"C_leg_right": 1.0, "C_leg_fwd": 1.0}, 1000.0)
    print("  (a3) C_leg_left 末 θ=%+.4f rad；C_leg_right 末 θ=%+.4f rad（应反号）"
          % (r_tl["theta"][-1], r_tr["theta"][-1]), flush=True)

    # ---------------- (a4) curl 曲率饱和 ----------------
    b.reset(); r_c = b.run({"C_curl": 1.0}, 1000.0)
    curl_max = float(np.max(r_c["curl"]))
    curl_disp = float(abs(r_c["x"][-1]))
    print("  (a4) C_curl=1.0：curl 峰值 %.3f（应饱和于有界值）；位移 %.4f mm（防御态≈0）"
          % (curl_max, curl_disp), flush=True)

    # ---------------- (a5) 翅拍 + 翅展 ----------------
    b.reset(); r_w = b.run({"C_wing": 1.0}, 200.0)
    wf = main_peak_freq(r_w["wing_ang"], dt)
    b.reset(); r_e = b.run({"C_escape": 1.0}, 200.0)
    print("  (a5) 翅拍主频 %.1f Hz（带 150–250，含采样限制）；C_escape→翅展 %.3f"
          % (wf, float(np.max(r_e["wing_spread"]))), flush=True)

    # ---------------- (c) 巨纤维逃跑 sanity ----------------
    resp = 0
    for k in range(N_ESCAPE_TRIALS):
        bb = AdultFlyBody(p, seed=k)
        bb.reset()
        rr = bb.run({}, 300.0, stimuli={"light": 1.0})
        if len(rr["escape_idx"]) > 0:
            resp += 1
    prob = resp / float(N_ESCAPE_TRIALS)
    resp_sub = 0
    for k in range(N_ESCAPE_TRIALS):
        bb = AdultFlyBody(p, seed=k)
        bb.reset()
        rr = bb.run({}, 300.0, stimuli={"light": p.escape_thr * 0.4})
        if len(rr["escape_idx"]) > 0:
            resp_sub += 1
    prob_sub = resp_sub / float(N_ESCAPE_TRIALS)
    print("  (c) 逃跑反应概率 %.2f（带 ≥0.8）；亚阈值刺激（0.4×阈值）概率 %.2f（带 ≤0.2）"
          % (prob, prob_sub), flush=True)

    # ---------------- (b) 状态分类语义 ----------------
    b.reset(); s_run = state_fractions(b.run({"C_leg_fwd": 1.0}, 1000.0)["states"])
    b.reset(); s_curl = state_fractions(b.run({"C_curl": 1.0}, 1000.0)["states"])
    b.reset(); s_pause = state_fractions(b.run({}, 1000.0)["states"])
    print("  (b) 状态比例：fwd驱动→run %.2f；curl驱动→curl %.2f；无驱动→pause %.2f"
          % (s_run["run"], s_curl["curl"], s_pause["pause"]), flush=True)

    # ---------------- (d) 确定性 + 有界/无 NaN ----------------
    b1 = AdultFlyBody(p, seed=0); b1.reset()
    b2 = AdultFlyBody(p, seed=0); b2.reset()
    r1 = b1.run({"C_leg_fwd": 1.0}, 500.0)
    r2 = b2.run({"C_leg_fwd": 1.0}, 500.0)
    det = bool(np.array_equal(r1["joints_deg"], r2["joints_deg"])
               and np.array_equal(r1["x"], r2["x"]))
    finite = bool(np.all(np.isfinite(r1["joints_deg"])) and np.all(np.isfinite(r1["x"]))
                  and np.all(np.isfinite(r1["y"])) and np.all(np.isfinite(r1["v"])))
    ang_ok = bool(np.max(np.abs(r1["joints_deg"])) <= 180.0)
    print("  (d) 确定性逐位一致 %s；轨迹有限 %s；关节角有界(≤180°) %s" % (det, finite, ang_ok),
          flush=True)

    # ---------------- 判据 ----------------
    def _in(metric, val):
        bb = band.get(metric)
        return (bb is None) or (bb["lo"] <= val <= bb["hi"])

    crit = {
        "a1_gait_freq": _in("gait_freq_hz", f_meas),
        "a1_tripod_phase": _in("tripod_phase_deg", pd_cross),
        "a1_intragroup": _in("tripod_intragroup_phase_deg", max(intra)),
        "a2_v_fwd_sign": (d_fwd > 0) and (d_back < 0),
        "a2_v_magnitude": _in("v_fwd_magnitude", abs(d_fwd)),
        "a3_turn_sign": (r_tl["theta"][-1] > 0) and (r_tr["theta"][-1] < 0),
        "a4_curl_saturate": _in("curl_curvature_max", curl_max) and curl_disp < 0.5,
        "a5_wing_freq": _in("wing_freq_hz", wf),
        "a5_wing_spread": _in("wing_spread_on_escape", float(np.max(r_e["wing_spread"]))),
        "b_state_semantics": (s_run["run"] > 0.5) and (s_curl["curl"] > 0.5)
                             and (s_pause["pause"] > 0.5),
        "c_escape_resp": _in("escape_resp_prob", prob),
        "c_escape_subthr": _in("escape_resp_prob_subthr", prob_sub),
        "d_determinism": det,
        "d_finite_bounded": finite and ang_ok,
    }
    verdict = "PASS" if all(crit.values()) else "FAIL"

    rows += [["protocol", "T_smoke_ms", T_SMOKE_MS, "单通道驱动冒烟窗"],
             ["protocol", "n_escape_trials", N_ESCAPE_TRIALS, "逃跑 sanity 试次数"],
             ["params", "gait_freq_hz", p.gait_freq_hz, "CSV 定稿"],
             ["params", "tripod_phase_deg", p.tripod_phase_deg, "CSV 定稿"],
             ["params", "wing_freq_hz", p.wing_freq_hz, "CSV 定稿（采样限制）"],
             ["params", "ball_diameter_mm", p.ball_diameter_mm, "VR 协议参数（§3.2）"],
             ["measure", "gait_freq_measured_hz", "%.4f" % f_meas, "FFT 主峰"],
             ["measure", "tripod_cross_phase_deg", "%.2f" % pd_cross, "组 A/B 相位差"],
             ["measure", "tripod_intragroup_max_deg", "%.2f" % max(intra), "组内最大相位差"],
             ["measure", "disp_fwd_mm", "%.4f" % d_fwd, "C_leg_fwd=1 净位移"],
             ["measure", "disp_back_mm", "%.4f" % d_back, "C_leg_back=1 净位移"],
             ["measure", "theta_left_rad", "%.4f" % r_tl["theta"][-1], ""],
             ["measure", "theta_right_rad", "%.4f" % r_tr["theta"][-1], ""],
             ["measure", "curl_peak", "%.4f" % curl_max, ""],
             ["measure", "curl_disp_mm", "%.4f" % curl_disp, ""],
             ["measure", "wing_freq_measured_hz", "%.4f" % wf, "采样限制登记"],
             ["measure", "wing_spread_escape", "%.4f" % float(np.max(r_e["wing_spread"])), ""],
             ["measure", "escape_resp_prob", "%.4f" % prob, ""],
             ["measure", "escape_resp_prob_subthr", "%.4f" % prob_sub, ""],
             ["measure", "state_frac_run", "%.4f" % s_run["run"], ""],
             ["measure", "state_frac_curl", "%.4f" % s_curl["curl"], ""],
             ["measure", "state_frac_pause", "%.4f" % s_pause["pause"], ""],
             ["measure", "determinism_bitwise", str(det), ""],
             ["measure", "trajectory_finite", str(finite), ""],
             ["measure", "joint_angle_bounded", str(ang_ok), ""]]
    for k, v in crit.items():
        rows.append(["crit", k, str(bool(v)), ""])
    rows.append(["crit", "verdict", verdict, "P3 三态判定（规划节点复核）"])
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        f.write("# M9 P3 虚拟身体验证（§3.3；判据带只读 m9_behavior_reference.csv body 段）\n")
        w = _csv.writer(f, lineterminator="\n")
        w.writerows(rows)
    print("→", OUT_CSV, flush=True)

    make_plot(r_fwd, r_f, r_bk, r_tl, r_tr, r_c, r_e, lift, p, f_meas, pd_cross)
    print("→", OUT_PNG, flush=True)
    print("P3 判定：%s（%s）" % (verdict, crit), flush=True)
    return 0 if verdict == "PASS" else 2


def make_plot(r_fwd, r_f, r_bk, r_tl, r_tr, r_c, r_e, lift, p, f_meas, pd_cross):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    t = r_fwd["t_ms"]
    ax = axes[0, 0]
    for leg, style in (("L1", "-"), ("R2", "-"), ("L3", "-"), ("R1", "--"),
                       ("L2", "--"), ("R3", "--")):
        ax.plot(t, lift[leg] - p.femur_bias_deg, style, lw=1, label=leg)
    ax.set_title("(a1) Tripod gait: femur lift\nf=%.2f Hz, A/B phase=%.1f deg"
                 % (f_meas, pd_cross))
    ax.set_xlabel("t (ms)"); ax.set_ylabel("lift - bias (deg)"); ax.legend(fontsize=7, ncol=2)
    ax = axes[0, 1]
    ax.plot(r_f["t_ms"], r_f["x"], label="C_leg_fwd")
    ax.plot(r_bk["t_ms"], r_bk["x"], label="C_leg_back")
    ax.set_title("(a2) Displacement by drive channel")
    ax.set_xlabel("t (ms)"); ax.set_ylabel("x (mm)"); ax.legend()
    ax = axes[0, 2]
    ax.plot(r_tl["t_ms"], r_tl["theta"], label="C_leg_left")
    ax.plot(r_tr["t_ms"], r_tr["theta"], label="C_leg_right")
    ax.set_title("(a3) Turn direction"); ax.set_xlabel("t (ms)")
    ax.set_ylabel("theta (rad)"); ax.legend()
    ax = axes[1, 0]
    ax.plot(r_c["t_ms"], r_c["curl"], label="curl (C_curl=1)")
    ax.plot(r_e["t_ms"], r_e["wing_spread"], label="wing_spread (C_escape=1)")
    ax.set_title("(a4/a5) curl saturation & escape wing spread")
    ax.set_xlabel("t (ms)"); ax.legend()
    ax = axes[1, 1]
    ax.plot(r_fwd["t_ms"], r_fwd["wing_ang"], lw=0.6)
    ax.set_title("(a5) Wing beat (%.0f Hz, dt-limited)" % p.wing_freq_hz)
    ax.set_xlabel("t (ms)"); ax.set_ylabel("wing angle (deg)")
    ax = axes[1, 2]
    st = r_fwd["states"]
    from collections import Counter
    c = Counter(st)
    keys = ["run", "turn", "pause", "curl", "escape"]
    ax.bar(keys, [c.get(k, 0) / max(len(st), 1) for k in keys], color="#2ca02c")
    ax.set_title("(b) State fractions under C_leg_fwd drive")
    ax.set_ylabel("fraction")
    fig.suptitle("M9 P3 virtual adult fly body (gait / channels / states)")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=110)


if __name__ == "__main__":
    sys.exit(main())
