"""M7 P-A1 等价性验证（M7-B1a）：机制封装模块 vs 神经仿真冻结协议/行为参考。

对应《生物仿真M7实施清单》§0 P-A1（行为量等价）+ §2.1 等价性锚：
  - M-1 反射：D_peak∈[0.352,0.369]（M3 定稿带中点锚 0.360）→ back（>0.3）；
  - M-2 趋化：CI 符号正（M4 参考 CI(25s)=0.494 / CI(15s)=0.417；生物带
    [0.3,0.7]、容差窗 [0.25,0.75]）；
  - M-3 CPG：无食物 0.400Hz∈[0.1,2] / 有食物 2.167Hz∈[2,5]（M5 P3 冻结）；
  - M-4 习惯化：指数拟合 R²≥0.5（冻结 0.787）+ 后半均值<0.5×前半均值 +
    消融 STP 关 → 无衰减 + 恢复 R_rest≥0.3×R(1)（冻结 1.17）；
  - M-5 联想：Δw_train>0.1（冻结 +0.4325）、η=0 → Δw=0、Δw_ext<0
    （冻结 −0.108；绝对值不作硬判据——清单 §2.1）；
  - M-6 调质：fwd_gate ∈ [tyr_floor,1.2]、动机单调、酪胺关 → gate≡1
    （冻结消融 sanity）；
  全部探针确定性重跑**逐位一致**（清单 §0 #7 确定性纪律）。

参数源：`data/m7_innate_params.csv`（唯一定稿源；模块只读，不重训/校准）。
输出：reports/neuro/m7_equivalence_summary.json（供 P5 报告消费）。

用法：.venv-neuro/bin/python -m neural_exploration.tools.validate_m7_equivalence
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 机制模块在 digital_brain 侧（M7 步骤 1 定稿位置；纯 stdlib 无 brian2 依赖）
from digital_brain.src.innate import (  # noqa: E402
    Stimulus, make_all,
)

REPORTS_DIR = os.path.join(ROOT, "reports", "neuro")
SUMMARY_PATH = os.path.join(REPORTS_DIR, "m7_equivalence_summary.json")


def _check(name: str, ok: bool, detail: str, results: dict) -> None:
    results["checks"].append({"mechanism": name, "pass": bool(ok),
                              "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def _deterministic(fn, results: dict, label: str) -> bool:
    """探针重跑逐位一致（repr 级比较）。"""
    a = repr(fn())
    b = repr(fn())
    ok = a == b
    results["determinism"][label] = bool(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] determinism {label}: "
          f"{'bit-identical' if ok else 'MISMATCH'}")
    return ok


def main() -> int:
    t0 = time.perf_counter()
    results: dict = {
        "tool": "validate_m7_equivalence",
        "node": "M7-B1a",
        "params_csv": "neural_exploration/data/m7_innate_params.csv",
        "pass": True,
        "checks": [],
        "determinism": {},
        "anchors": {},
    }
    ms = make_all()
    print("M7 P-A1 等价性验证（机制封装 vs 冻结锚）\n")

    # ---------------- M-1 反射弧 ----------------
    print("M-1 反射弧（M3 定稿 D_peak 带 [0.352,0.369] >0.3 → back）")
    m = ms["reflex"]
    r = m.respond(Stimulus(intensity=1.0))
    d_peak = r.value
    in_band = 0.352 <= d_peak <= 0.369
    _check("reflex.d_peak_in_band", in_band,
           f"D_peak(1.0)={d_peak:.4f} ∈ [0.352,0.369]", results)
    _check("reflex.direction_back", r.direction == "back" and d_peak > 0.3,
           f"direction={r.direction} (D_peak {d_peak:.4f} > 0.3)", results)
    r0 = m.respond(Stimulus(intensity=0.0))
    _check("reflex.intensity0_none", r0.value == 0.0 and r0.direction == "none",
           f"I=0 → D_peak={r0.value} dir={r0.direction}", results)
    r2 = m.respond(Stimulus(intensity=2.0))
    _check("reflex.monotone", r2.value > d_peak,
           f"I=2 → D_peak={r2.value:.4f} > D_peak(1)={d_peak:.4f}", results)
    _deterministic(lambda: m.respond(Stimulus(intensity=1.0)).as_dict(),
                   results, "reflex")
    results["anchors"]["reflex"] = {
        "d_peak_at_1": float(d_peak), "direction": r.direction,
        "frozen_band": [0.352, 0.369], "frozen_latency_ms": 32.6,
        "behavior_latency_ms": float(m.p("behavior_latency_ms", 32.6))}

    # ---------------- M-2 趋化 ----------------
    print("\nM-2 嗅觉趋化（M4 冻结：正向梯度趋利；CI 符号正）")
    m = ms["chemotaxis"]
    cii = {}
    for th, label in ((0.0, "east"), (math.pi / 2, "north"),
                      (math.pi, "west"), (-math.pi / 2, "south")):
        xs, ys = m.run_trial(theta0=th)
        cii[label] = m.ci(xs, ys)
    all_pos = all(v > 0.0 for v in cii.values())
    _check("chemotaxis.ci_sign", all_pos,
           f"CI(10s) 四朝向 = { {k: round(v, 3) for k, v in cii.items()} } 全正",
           results)
    xs, ys = m.run_trial()  # 默认 theta0=π（不利朝向探针）
    ci_canon = m.ci(xs, ys)
    _check("chemotaxis.ci_in_tolerance",
           0.25 <= ci_canon <= 0.75,
           f"CI_canon(10s,θ=π)={ci_canon:.4f} ∈ 容差窗 [0.25,0.75] "
           f"（冻结参考 CI(25s)=0.494 / CI(15s)=0.417）", results)
    gx, gy = m.gradient(5.0, 5.0)
    _check("chemotaxis.gradient_toward_food",
           math.hypot(gx, gy) > 0 and math.atan2(gy, gx) ==
           math.atan2(m.p("food_y") - 5.0, m.p("food_x") - 5.0),
           f"∇C(中心) 指向食物滴 (∇=({gx:.3f},{gy:.3f}))", results)
    _deterministic(lambda: (m.ci(*m.run_trial())), results, "chemotaxis")
    results["anchors"]["chemotaxis"] = {
        "ci_by_heading": {k: round(v, 4) for k, v in cii.items()},
        "ci_canon": float(ci_canon),
        "frozen_ci_25s_ref": 0.494, "frozen_ci_15s_ref": 0.417,
        "ci_band": [0.3, 0.7], "ci_tolerance": [0.25, 0.75]}

    # ---------------- M-3 CPG ----------------
    print("\nM-3 咽部 CPG（M5 P3 冻结：0.400 / 2.167Hz 双带）")
    m = ms["cpg"]
    f0 = m.respond(Stimulus(kind="time", t_ms=1000, food_present=False))
    f1 = m.respond(Stimulus(kind="time", t_ms=1000, food_present=True))
    ok0 = f0.extra["in_band"] and f0.value == m.p("f_no_food_hz")
    ok1 = f1.extra["in_band"] and f1.value == m.p("f_with_food_hz")
    _check("cpg.no_food_band", ok0,
           f"f_no_food={f0.value:.3f}Hz ∈ [0.1,2]（冻结 0.400）", results)
    _check("cpg.with_food_band", ok1,
           f"f_with_food={f1.value:.3f}Hz ∈ [2,5]（冻结 2.167）", results)
    _deterministic(lambda: m.respond(Stimulus(kind="time", t_ms=1234,
                                              food_present=True)).as_dict(),
                   results, "cpg")
    results["anchors"]["cpg"] = {"f_no_food": float(f0.value),
                                 "f_with_food": float(f1.value),
                                 "frozen": [0.400, 2.167],
                                 "bands": [[0.1, 2.0], [2.0, 5.0]]}

    # ---------------- M-4 习惯化 ----------------
    print("\nM-4 习惯化（M6 P3 冻结：R²=0.787 / τ≈2 / 消融 / 恢复）")
    m = ms["habituation"]
    res = m.run_sequence(n_stim=6, rest_ms=2000.0)
    fit = res["fit"]
    _check("habituation.r2", fit["r2_ok"] and fit["r2"] >= 0.5,
           f"指数拟合 R²={fit['r2']:.4f} ≥ 0.5（冻结 0.787）", results)
    _check("habituation.decay", res["decay_ok"],
           f"decay=R(1)−R(N)={res['decay']:.4f} > 0 "
           f"R(n)={[round(v, 3) for v in res['r_seq']]}", results)
    _check("habituation.half_criterion", res["half_criterion_ok"],
           f"后半均值 {res['last_half_mean']:.4f} < 0.5×前半均值 "
           f"{0.5 * res['first_half_mean']:.4f}（冻结 −0.183 vs 0.093）", results)
    _check("habituation.direction_sanity", res["direction_ok"],
           f"R(1)={res['r_seq'][0]:.4f} > 0.3（冻结 0.353）", results)
    res_off = m.run_sequence(n_stim=6, stp_enabled=False)
    _check("habituation.stp_off_ablation", res_off["decay"] == 0.0,
           f"STP 关 → R(n) 常数 decay={res_off['decay']:.4f}（冻结消融无衰减）",
           results)
    _check("habituation.recovery", res["recover_ok"],
           f"R_rest={res['r_rest']:.4f} ≥ 0.3×R(1)（冻结 R_rest=0.411 "
           f"frac=1.17）", results)
    _deterministic(lambda: m.run_sequence(n_stim=6, rest_ms=2000.0),
                   results, "habituation")
    results["anchors"]["habituation"] = {
        "r_seq": [round(v, 4) for v in res["r_seq"]], "r2": float(fit["r2"]),
        "tau_hab": float(fit["tau_hab"]), "decay": float(res["decay"]),
        "r_rest": float(res["r_rest"]),
        "frozen_r_seq": [0.3529, 0.3752, -0.1684, -0.1832, -0.1833, -0.1833],
        "frozen_r2": 0.787, "frozen_tau": 2.0, "frozen_r_rest": 0.411}

    # ---------------- M-5 联想学习 ----------------
    print("\nM-5 联想学习（M6 P4 冻结：Δw_train=+0.4325 / η=0→0 / Δw_ext=−0.108）")
    m = ms["associative"]
    res = m.full_protocol()
    _check("associative.acquisition", res["acquisition_ok"],
           f"Δw_train={res['dw_train']:.4f} > 0.1（冻结 +0.4325；"
           f"同号，幅度差如实记录——清单 §2.1 绝对值不作硬判据）", results)
    _check("associative.extinction", res["extinction_ok"],
           f"Δw_ext={res['dw_ext']:.4f} < 0（冻结 −0.108）", results)
    _check("associative.eta0_ablation", res["eta0_ok"],
           f"η=0 → Δw={res['dw_eta0']:.6f} ≈ 0（冻结 η=0 Δw=0）", results)
    _deterministic(m.full_protocol, results, "associative")
    results["anchors"]["associative"] = {
        "dw_train": float(res["dw_train"]), "dw_ext": float(res["dw_ext"]),
        "dw_eta0": float(res["dw_eta0"]),
        "frozen_dw_train": 0.4325, "frozen_dw_ext": -0.108}

    # ---------------- M-6 调质层 ----------------
    print("\nM-6 调质层（M6 P2 冻结：gate ∈ [0.3,1.2]；酪胺关 → gate≡1）")
    m = ms["modulation"]
    gains = [m.gate(v) for v in (0.0, 0.25, 0.5, 0.75, 1.0)]
    in_range = all(0.3 <= g <= 1.2 for g in gains)
    _check("modulation.range", in_range,
           f"gate(mot)={[round(g, 3) for g in gains]} ⊆ [0.3, 1.2]", results)
    _check("modulation.monotone",
           all(gains[i] <= gains[i + 1] for i in range(len(gains) - 1)),
           "动机↑ → 增益单调↑（5HT 促进通道）", results)
    m2 = make_all()["modulation"]
    m2.params["tyramine_enabled"] = 0.0
    g_off = m2.gate(0.0)
    _check("modulation.tyramine_off", abs(g_off - 1.0) < 0.02,
           f"酪胺关 → gate={g_off:.4f} ≈ 1（冻结 sanity：gate≡1）", results)
    _deterministic(lambda: m.gate(0.5), results, "modulation")
    results["anchors"]["modulation"] = {
        "gates": [round(g, 4) for g in gains],
        "gate_tyramine_off": float(g_off),
        "gate_floor": 0.3, "gate_cap": 1.2,
        "frozen_o2_floor_gate": 0.4}

    # ---------------- 汇总 ----------------
    results["pass"] = all(c["pass"] for c in results["checks"]) and \
        all(results["determinism"].values())
    results["wall_s"] = round(time.perf_counter() - t0, 3)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    n_pass = sum(1 for c in results["checks"] if c["pass"])
    print(f"\n等价性断言 {n_pass}/{len(results['checks'])} 通过；"
          f"确定性重跑 {sum(results['determinism'].values())}/"
          f"{len(results['determinism'])} 逐位一致")
    print(f"summary → {SUMMARY_PATH}")
    print(f"总体：{'PASS ✅' if results['pass'] else 'FAIL ❌'}")
    return 0 if results["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
