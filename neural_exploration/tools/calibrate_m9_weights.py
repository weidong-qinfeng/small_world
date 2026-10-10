"""M9 全规模回路权重/驱动标定（拟合集 A：静息判据带；§0.8 数据隔离 / §8 ① 类级缩放）。

《生物仿真M9实施清单》§3.5.2（静息协议判据带定稿于 `data/m9_behavior_reference.csv`）/
§0.8（标定只读拟合集 A）/§0.9 R10（先探针 → 裁决 → 再烧）。

纪律：
  - **判据带先定稿**（`m9_behavior_reference.csv` 的 resting 段），标定只调参数不改带；
  - 标定参数 = 类级全局参数（≤40，§8 ①）：w_exc / w_inh / w_mod / γ（syn_count 幂）/
    b0（tonic 偏置均值）/ cv（偏置 CV）/ 背景率 λ / 背景振幅 A。**不做逐突触拟合**；
  - 落盘 `data/m9_weight_calibration.csv`（每配置实测统计 + 是否落带），可审计。

用法：
  PYTHONHASHSEED=0 MPLBACKEND=Agg .venv-m9/bin/python -m neural_exploration.tools.calibrate_m9_weights
"""

from __future__ import annotations

import csv as _csv
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUT_CSV = os.path.join(DATA_DIR, "m9_weight_calibration.csv")

#: 预注册判据带（只读 `data/m9_behavior_reference.csv` 的 resting 段；不在此处改带）
BAND = {"rate_median_hz": (0.1, 10.0), "silent_fraction": (0.5, 0.9),
        "rate_p95_hz": (0.0, 50.0)}

#: 标定网格（阶段 1 粗扫 → 阶段 2 细化；类级全局参数）
GRID_STAGE1 = [
    # w_exc, w_inh, b0(mV/s 中位), cv(lognormal σ), lam(Hz), epsp(mV), ahp_inc(mV/s)
    # 标定网格（最终档 1.0/4.0/355/0.60 的出带记录 + 定稿档 0.3/1.0/290/0.25 的落带记录；
    #  背景率 λ 与 bias 中位为两个主杠杆，见 docs/m9_env_notes.md L21.1）
    (0.3, 1.0, 290.0, 0.25, 0.50, 2.0, 100.0),
    (0.3, 1.0, 290.0, 0.25, 0.50, 2.0, 50.0),
    (0.3, 1.0, 290.0, 0.25, 0.50, 2.0, 25.0),
    (0.3, 1.0, 290.0, 0.25, 0.50, 2.0, 0.0),
]


def _run_config(c, cfg, T_ms=2000.0, settle_ms=200.0, seed=0):
    we, wi, b0, cv, lam, epsp, ahp = cfg
    g_ext = epsp / 0.104
    c.params.bias_cv = cv
    if abs(c.params.ahp_inc - ahp) > 1e-9:
        c.params.ahp_g_inc = ahp
        c.engine.set_point_params(ahp_g_inc=ahp)
    n_steps = int(round((settle_ms + T_ms) / c.params.dt_ms)) + 4
    c.build_background(rate_hz=lam, g_ext=g_ext, n_steps=n_steps, seed=seed)
    c._bg_steps = n_steps
    c._bg_args = (lam, g_ext)
    c.reweight(w_exc=we, w_inh=wi, bias_mv_s=b0, bias_cv=cv)
    st = c.run_resting(T_ms=T_ms, settle_ms=settle_ms, seed=seed)
    return st


def main() -> int:
    from neural_exploration.src.adult_circuit import AdultCircuit, CircuitParams
    print("=== M9 全规模回路标定（拟合集 A：静息判据带）===", flush=True)
    t00 = time.perf_counter()
    c = AdultCircuit(device="mps",
                     params=CircuitParams(bias_mv_s=0.0, bias_cv=0.0,
                                          syn_count_gamma=1.0, delay_ms=1.0,
                                          ahp_tau_ms=700.0, ahp_g_inc=100.0),
                     use_compile=False)
    c.build()
    rows = []
    hdr = ["w_exc", "w_inh", "b0_mv_s", "bias_cv", "bg_rate_hz", "bg_epsp_mv",
           "ahp_g_inc",
           "rate_median_hz", "rate_mean_hz", "rate_p95_hz", "silent_fraction",
           "pop_rate_hz", "spk_frac_per_step", "ms_per_step", "in_band", "wall_s"]
    for cfg in GRID_STAGE1:
        t0 = time.perf_counter()
        st = _run_config(c, cfg, T_ms=1000.0, settle_ms=200.0)
        in_band = (BAND["rate_median_hz"][0] <= st["rate_median_hz"]
                   <= BAND["rate_median_hz"][1]
                   and BAND["silent_fraction"][0] <= st["silent_frac"]
                   <= BAND["silent_fraction"][1])
        rows.append(list(cfg) + ["%.4f" % st["rate_median_hz"],
                                 "%.4f" % st["rate_mean_hz"],
                                 "%.3f" % st["rate_p95_hz"],
                                 "%.4f" % st["silent_frac"],
                                 "%.4f" % st["pop_rate_hz"],
                                 "%.6f" % st["spk_frac_per_step"],
                                 "%.3f" % st["ms_per_step"],
                                 str(bool(in_band)), "%.1f" % (time.perf_counter() - t0)])
        print("we=%.1f wi=%.1f b0=%5.0f cv=%.2f lam=%.1f epsp=%.1f ahp=%5.0f → med=%9.4f mean=%9.3f "
              "silent=%.3f pop=%8.4f %s (%.0fs)"
              % (cfg[0], cfg[1], cfg[2], cfg[3], cfg[4], cfg[5], cfg[6], st["rate_median_hz"],
                 st["rate_mean_hz"], st["silent_frac"], st["pop_rate_hz"],
                 "IN-BAND" if in_band else "", time.perf_counter() - t0), flush=True)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        f.write("# M9 全规模回路标定（拟合集 A = 静息判据带；类级全局参数 ≤40，§8 ①）\n")
        f.write("# 判据带定稿于 data/m9_behavior_reference.csv（resting 段），标定不改带\n")
        w = _csv.writer(f, lineterminator="\n")
        w.writerow(hdr)
        w.writerows(rows)
    print("标定结果 →", OUT_CSV, "（总 %.0fs）" % (time.perf_counter() - t00), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
