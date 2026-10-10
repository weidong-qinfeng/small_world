"""M9 P7 机制诊断：群体率对背景驱动强度的敏感性（**诊断臂，非判据**）。

动机（P7 判据 (b) 反证后的机制定位）：
  修正昼/夜窗后实测 `day_night_activity_ratio = 1.029`（带 [1.2, 10]）——
  背景率 λ(t) 的昼/夜均值比 **3.08×**，而群体率几乎不动（33.66 vs 32.71）。
  必须区分两种解释：
    (i)  群体活动**自主自持**（tonic 偏置驱动），背景输入是可忽略微扰；
    (ii) 实现 bug（外部事件未真正投递 / 被覆盖）。
  本诊断给出**判别证据**：扫描**恒定**背景率 λ0 ∈ {0, 0.25, 0.5, 1.0, 2.0, 4.0}，
  若 pop_rate 对 λ0 近乎不变（含 λ0=0 时仍 ≈ 基线），则 (i) 成立 → (b) 为**机制性反证**而非 bug。

**本脚本不改任何判据带**（`m9_behavior_reference.csv` 只读）；输出仅诊断 CSV。
输出：`data/m9_p7_drive_sensitivity.csv`
用法：PYTHONHASHSEED=0 MPLBACKEND=Agg .venv-m9/bin/python -m neural_exploration.tools.diag_p7_drive_sensitivity
"""

from __future__ import annotations

import csv as _csv
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUT_CSV = os.path.join(DATA_DIR, "m9_p7_drive_sensitivity.csv")

T_MS = 1000.0
SETTLE_MS = 300.0
SEGMENT = 2000
#: 预算纪律：λ0=0.5 的基线已由 P7 本体测出（见 `data/m9_p7_sleep.csv`），
#: 故此处只补 **两臂**（零驱动 / 4× 驱动）→ 3 点敏感性曲线，最省预算。
LAM0_LIST = (0.0, 2.0)
BASE_LAM0_REF = 4.7482   # P7 rhythm0 臂 pop_rate（λ0=0.5 Hz，参考值，来自 m9_p7_sleep.csv）


def main() -> int:
    from neural_exploration.src.adult_circuit import AdultCircuit, CircuitParams
    from neural_exploration.tools.validate_p9_resting import PARAMS, BG

    print("=== M9 P7 诊断：群体率 vs 背景驱动强度（诊断臂，非判据）===", flush=True)
    t0 = time.perf_counter()
    P = dict(PARAMS)
    c = AdultCircuit(device="mps", params=CircuitParams(**P), use_compile=False)
    c.build()
    g_ext = BG["epsp_mv"] / 0.104
    nn = int(round((SETTLE_MS + T_MS) / P["dt_ms"])) + 8
    dt = P["dt_ms"]

    def run_lam(lam0, seed=0):
        rng = np.random.default_rng(seed + 4242)
        n_ev = rng.poisson(max(lam0, 0.0) * nn * dt * 1e-3, size=c.n_neurons)
        total = int(n_ev.sum())
        if total == 0:
            c.engine.set_external_events(np.zeros(0, np.int64), np.zeros(0, np.int64),
                                         np.zeros(0, np.float32), nn)
        else:
            neurons = np.repeat(np.arange(c.n_neurons, dtype=np.int64), n_ev)
            steps = rng.integers(0, nn, size=total).astype(np.int64)
            c.engine.set_external_events(steps, neurons,
                                         np.full(total, g_ext, dtype=np.float32), nn)
        c._bg_steps, c._bg_args = nn, (BG["rate_hz"], g_ext)
        st = c.run_resting(T_ms=T_MS, settle_ms=SETTLE_MS, seed=seed, pop_trace=True,
                           segment_steps=SEGMENT)
        pop = st["pop"]
        return {"n_events": total, "pop_rate_hz": st["pop_rate_hz"],
                "rate_mean": st["rate_mean_hz"], "rate_median": st["rate_median_hz"],
                "silent": st["silent_frac"], "binsum_median": float(np.median(pop)),
                "ms_per_step": st["ms_per_step"]}

    rows = [["section", "key", "value", "note"],
            ["protocol", "T_ms", T_MS, ""],
            ["protocol", "settle_ms", SETTLE_MS, ""],
            ["protocol", "g_ext", "%.6f" % g_ext, "EPSP 2mV / 0.104"],
            ["protocol", "segment_steps", SEGMENT, "分段调用（< n_meas 才走分段分支）"],
            ["reference", "lam0_0.50_pop_rate_hz", "%.4f" % BASE_LAM0_REF,
             "基线臂由 P7 本体测得（data/m9_p7_sleep.csv rhythm0）"],
            ["note", "purpose", "diagnostic",
             "判别 (b) 反证是 机制性 还是 实现 bug：驱动强度扫描"]]
    for lam in LAM0_LIST:
        r = run_lam(lam)
        print("  λ0=%.2f Hz → 事件 %d、pop_rate=%.4f Hz、rate_median=%.4f、silent=%.4f（%.1f ms/步）"
              % (lam, r["n_events"], r["pop_rate_hz"], r["rate_median"], r["silent"],
                 r["ms_per_step"]), flush=True)
        for k, v in r.items():
            rows.append(["scan", "lam0_%.2f_%s" % (lam, k),
                         ("%d" % v) if k == "n_events" else "%.6f" % v, ""])
        rows.append(["derived", "lam0_%.2f_rel_vs_baseline" % lam,
                     "%.4f" % (r["pop_rate_hz"] / BASE_LAM0_REF),
                     "1.0 = 群体率与驱动无关"])
        rows.append(["scan", "lam0_%.2f_wall_s" % lam,
                     "%.1f" % (time.perf_counter() - t0), ""])
    scan = {r[1]: float(r[2]) for r in rows if r[0] == "scan" and r[1].endswith("pop_rate_hz")}
    zero_rel = scan["lam0_0.00_pop_rate_hz"] / BASE_LAM0_REF
    four_rel = scan["lam0_2.00_pop_rate_hz"] / BASE_LAM0_REF
    rows.append(["derived", "rel_zero_drive", "%.4f" % zero_rel,
                 "零背景驱动 / λ0=0.5 基线"])
    rows.append(["derived", "rel_4x_drive", "%.4f" % four_rel, "4× 驱动 / 基线"])
    rows.append(["derived", "drive_insensitive",
                 str(abs(zero_rel - 1.0) < 0.15 and abs(four_rel - 1.0) < 0.15),
                 "0×–4× 驱动下群体率变化 <15% → 活动**自主自持**（解释 (i)），非实现 bug"])
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        f.write("# M9 P7 机制诊断：群体率 vs 背景驱动强度（诊断臂；判据带未改）\n")
        _csv.writer(f, lineterminator="\n").writerows(rows)
    print("→", OUT_CSV, flush=True)
    print("总墙钟 %.0fs" % (time.perf_counter() - t0), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
