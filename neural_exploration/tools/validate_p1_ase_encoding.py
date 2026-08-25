"""M4 P1 验证：ASE ON/OFF 时间差分编码（环境→感觉接口，开环阶跃）。

清单 §0 P1 / §6.1：
- 协议：开环阶跃（虫位固定）：基线 40ms → 上升 ΔC=+0.5（50ms）→ 静止 50ms
  → 下降 ΔC=−0.5（50ms）→ 静止（CSV protocol p1_* 定稿，dt=0.01ms，T≈190ms）；
- 判据（§0 P1 + m4_env_notes §L4/L10 操作化）：
  1. 上升段 [rise_start, rise_end]：ASEL 发放 ≥1 且 ASER 静默（0 发放早于下降起点——
     滑窗差分 τ_win 记忆：上升段 s>0，ASER 无负差分 → 静默）；
  2. 下降段 [fall_start, fall_end]：ASER 发放 ≥1 且 ASEL 静默（留 ≥10ms 在途发放边界，
     L10：ASEL 注入持续到下降起点）；
  3. 静止段（ΔC=0）：两者均静默——在**无阶跃对照试次**（stationary_protocol，C≡C_base）
     上评估（τ_win 记忆：阶跃协议内静止段仍见差分响应，L4 预注册）；
  4. 确定性：p=1/n=1，同参数重跑逐位一致（ChemotaxisResult.__eq__）。
输出：reports/neuro/m4_p1_ase_encoding.png + data/m4_p1_ase_encoding.csv

用量：.venv-neuro/bin/python -m neural_exploration.tools.validate_p1_ase_encoding
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.src.chemotaxis_circuit import ChemotaxisCircuit  # noqa: E402
from neural_exploration.src.chemotaxis_env import stationary_protocol  # noqa: E402

DATA_DIR = os.path.join(ROOT, "neural_exploration", "data")
REPORTS_DIR = os.path.join(ROOT, "neural_exploration", "reports", "neuro")
CSV_PATH = os.path.join(DATA_DIR, "m4_chemotaxis_params.csv")
REPORT_PNG = os.path.join(REPORTS_DIR, "m4_p1_ase_encoding.png")
REPORT_CSV = os.path.join(DATA_DIR, "m4_p1_ase_encoding.csv")


def _count_in(t_spikes: np.ndarray, t0: float, t1: float) -> int:
    return int(np.sum((t_spikes >= t0 - 1e-9) & (t_spikes <= t1 + 1e-9)))


def _reuse_p1_from_csv() -> dict:
    """M4_REUSE=1 且 CSV 已存在 → 从 data/m4_p1_ase_encoding.csv 读回判定（不重跑）。"""
    import csv as _csv
    with open(REPORT_CSV, newline="", encoding="utf-8") as f:
        row = next(_csv.DictReader(f))
    n = lambda k: None if row[k] in ("", "None") else float(row[k])  # noqa: E731
    return dict(
        pass_=str(row["pass_"]).lower() == "true",
        n_asel_rise=int(float(row["n_asel_rise"])),
        n_aser_rise=int(float(row["n_aser_rise"])),
        n_aser_fall=int(float(row["n_aser_fall"])),
        n_asel_fall_late=int(float(row["n_asel_fall_late"])),
        n_stat_asel=int(float(row["n_stat_asel"])),
        n_stat_aser=int(float(row["n_stat_aser"])),
        n_stat_down=int(float(row["n_stat_down"])),
        deterministic=str(row["deterministic"]).lower() == "true",
        first_asel_ms=n("first_asel_ms"), first_aser_ms=n("first_aser_ms"),
        rise_ok=str(row["pass_"]).lower() == "true",
        fall_ok=str(row["pass_"]).lower() == "true",
        stat_ok=str(row["pass_"]).lower() == "true",
        protocol=None, csv_path=CSV_PATH,
    )


def run_p1(save_plot: bool = True) -> dict:
    if os.environ.get("M4_REUSE") and os.path.exists(REPORT_CSV):
        return _reuse_p1_from_csv()

    circ = ChemotaxisCircuit(csv_path=CSV_PATH)   # 确定性默认 p=1/n=1
    info = circ.protocol_info()
    rise_start, rise_end = info["rise_start_ms"], info["rise_end_ms"]
    fall_start, fall_end = info["fall_start_ms"], info["fall_end_ms"]

    # ---- 1) 默认全阶跃协议（上升→静止→下降→静止）单次运行 ----
    r1 = circ.run()
    # ---- 2) 确定性重跑（同参数同种子 → 逐位一致） ----
    r2 = circ.run()

    t_asel, t_aser = r1.spikes("ASEL", "node3"), r1.spikes("ASER", "node3")

    # 判据 1：上升段 ASEL ≥1；ASER 在上升+静止段（下降起点前）静默
    n_asel_rise = _count_in(t_asel, rise_start - 1.0, rise_end + 1.0)
    n_aser_before_fall = int(np.sum(t_aser < fall_start))
    rise_ok = bool(n_asel_rise >= 1 and n_aser_before_fall == 0)

    # 判据 2：下降段 ASER ≥1；ASEL 静默（留 10ms 在途发放边界，L10）
    n_aser_fall = _count_in(t_aser, fall_start - 1.0, fall_end + 1.0)
    n_asel_late = _count_in(t_asel, fall_start + 10.0, fall_end)
    fall_ok = bool(n_aser_fall >= 1 and n_asel_late == 0)

    # ---- 3) 静止浓度对照试次（C≡C_base，ΔC=0 → 两者均静默） ----
    t_stat, c_stat = stationary_protocol(t_total_ms=150.0, dt_ms=circ.params.dt_ms)
    circ.set_protocol(c_trace=c_stat, dt_protocol_ms=circ.params.dt_ms)
    try:
        r_stat = circ.run()
    finally:
        circ.clear_protocol()
    n_stat_asel = len(r_stat.spikes("ASEL", "node3"))
    n_stat_aser = len(r_stat.spikes("ASER", "node3"))
    # 下游链在静止时亦静默（无自发活动；AVB/VB 张力基线除外）
    n_stat_down = sum(len(r_stat.spikes(role, "node3"))
                      for role in ("AIYL", "AIBL", "RIAL", "SMDDL"))
    stat_ok = bool(n_stat_asel == 0 and n_stat_aser == 0 and n_stat_down == 0)

    # ---- 4) 确定性重跑逐位一致 ----
    det_ok = bool(r1 == r2)
    det_ci = r1.meta["seed"] == r2.meta["seed"]

    pass_ = bool(rise_ok and fall_ok and stat_ok and det_ok)

    out = dict(
        pass_=pass_,
        n_asel_rise=n_asel_rise, n_aser_rise=int(np.sum(t_aser < fall_start)),
        n_aser_fall=n_aser_fall,
        n_asel_fall_late=n_asel_late,
        n_stat_asel=n_stat_asel, n_stat_aser=n_stat_aser, n_stat_down=n_stat_down,
        deterministic=bool(det_ok and det_ci),
        first_asel_ms=float(t_asel[0]) if len(t_asel) else None,
        first_aser_ms=float(t_aser[0]) if len(t_aser) else None,
        rise_ok=rise_ok, fall_ok=fall_ok, stat_ok=stat_ok,
        protocol=info, csv_path=CSV_PATH,
    )

    # ---- CSV 落盘 ----
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(REPORT_CSV, "w", encoding="utf-8", newline="") as f:
        import csv as _csv
        w = _csv.DictWriter(f, fieldnames=[
            "check", "n_asel_rise", "n_aser_rise", "n_aser_fall", "n_asel_fall_late",
            "n_stat_asel", "n_stat_aser", "n_stat_down", "deterministic",
            "first_asel_ms", "first_aser_ms", "pass_"])
        w.writeheader()
        w.writerow(dict(
            check="P1 ASE ON/OFF encoding", n_asel_rise=n_asel_rise,
            n_aser_rise=int(np.sum(t_aser < fall_start)), n_aser_fall=n_aser_fall,
            n_asel_fall_late=n_asel_late, n_stat_asel=n_stat_asel,
            n_stat_aser=n_stat_aser, n_stat_down=n_stat_down,
            deterministic=out["deterministic"], first_asel_ms=out["first_asel_ms"],
            first_aser_ms=out["first_aser_ms"], pass_=pass_))

    if save_plot:
        _plot(r1, r_stat, info, out)

    return out


def _plot(r1, r_stat, info: dict, out: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(REPORTS_DIR, exist_ok=True)
    rise_start, rise_end = info["rise_start_ms"], info["rise_end_ms"]
    fall_start, fall_end = info["fall_start_ms"], info["fall_end_ms"]

    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=False)
    t = r1.t_ms

    # 1) 阶跃协议 C(t)（展示用；协议已知：c_base=0.2, ΔC=0.5，重建自 protocol_info）
    ax = axes[0]
    c_base = info["c_base"]
    c = np.full_like(t, c_base)
    i0 = int(round(info["t_baseline_ms"] / r1.meta["dt_ms"]))
    i1 = int(round((info["t_baseline_ms"] + info["t_up_ms"]) / r1.meta["dt_ms"]))
    i2 = int(round((info["t_baseline_ms"] + info["t_up_ms"] + info["t_hold_ms"]) / r1.meta["dt_ms"]))
    c[i0:i1] = c_base + info["delta_c"]
    c[i1:i2] = c_base + info["delta_c"]
    c[i2:] = c_base
    ax.plot(t, c, lw=1.4, color="k", label="C(t)")
    ax.axvspan(rise_start, rise_end, color="green", alpha=0.15)
    ax.axvspan(fall_start, fall_end, color="red", alpha=0.15)
    ax.text(rise_start + 1, 0.55, "rise ΔC>0", fontsize=8, color="green")
    ax.text(fall_start + 1, 0.55, "fall ΔC<0", fontsize=8, color="red")
    ax.set_ylabel("relative C")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("P1 open-loop step protocol (fixed worm position)", fontsize=10, loc="left")

    # 2) ASEL / 3) ASER V（soma）+ 发放标注
    for ax, role, color in ((axes[1], "ASEL", "#1f77b4"), (axes[2], "ASER", "#d62728")):
        lab = f"{role.lower()}_soma"
        ax.plot(t, r1.v_mv[lab], lw=0.7, color=color)
        st = r1.spikes(role, "node3")
        for tk in st:
            ax.axvline(tk, color=color, ls=":", lw=0.9, alpha=0.8)
        ax.axhline(-20.0, color="gray", ls="--", lw=0.6)
        ax.axvspan(rise_start, rise_end, color="green", alpha=0.12)
        ax.axvspan(fall_start, fall_end, color="red", alpha=0.12)
        ax.set_ylabel(f"{role} V (mV)")
        ax.set_ylim(-90, 60)
        ax.set_title(f"{role}: {len(st)} spikes @node3 (first={st[0]:.2f}ms)" if len(st)
                     else f"{role}: 0 spikes @node3", fontsize=8, loc="left")
        ax.grid(alpha=0.3)
    axes[2].set_xlabel("t (ms)")

    fig.suptitle("M4 P1: ASE ON/OFF time-difference encoding (rise→ASEL, fall→ASER, "
                 "stationary→silent; deterministic rerun bitwise-equal)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(REPORT_PNG, dpi=150)
    plt.close(fig)
    return REPORT_PNG


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="M4 P1 ASE 编码验证")
    ap.add_argument("--skip-plot", action="store_true")
    args = ap.parse_args()
    res = run_p1(save_plot=not args.skip_plot)
    print("==== M4 P1 ASE ON/OFF 编码 ====")
    print(f"  上升段: ASEL 发放 {res['n_asel_rise']} 次 (≥1 ✓/✗), "
          f"ASER 下降前 {res['n_aser_rise']} 次 (=0 ✓/✗) → {'OK' if res['rise_ok'] else 'FAIL'}")
    print(f"  下降段: ASER 发放 {res['n_aser_fall']} 次 (≥1 ✓/✗), "
          f"ASEL 晚发放 {res['n_asel_fall_late']} 次 (=0 ✓/✗) → {'OK' if res['fall_ok'] else 'FAIL'}")
    print(f"  静止段: ASEL={res['n_stat_asel']} ASER={res['n_stat_aser']} "
          f"下游={res['n_stat_down']}（均静默）→ {'OK' if res['stat_ok'] else 'FAIL'}")
    print(f"  确定性: 同参数重跑逐位一致 = {res['deterministic']}")
    print(f"  P1 pass_ = {res['pass_']}")
    if not args.skip_plot:
        print(f"  图: {REPORT_PNG}")
    print(f"  CSV: {REPORT_CSV}")
