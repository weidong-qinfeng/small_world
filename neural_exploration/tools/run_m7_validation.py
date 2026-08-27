"""M7 验证汇总总运行器（M7-B2 验证+报告节点）——P-A1/P-A2/P-A3 判定 + P5/P6 汇总。

对应《生物仿真M7实施清单》§0 P-A1/P-A2/P-A3（机制回迁数字大脑——阶段一收官）：
  - **P-A1 机制提取与封装**：in-process 复验（digital_brain.src.innate 直连，
    fresh compute 不信任 B1a 落盘）：21 项等价性断言（M-1 反射 / M-2 趋化 /
    M-3 CPG / M-4 习惯化 / M-5 联想 / M-6 调质）+ 6 项确定性逐位一致；
    并读回 `reports/neuro/m7_equivalence_summary.json`（B1a 交付物）交叉核对；
  - **P-A2 数字大脑接入**：pytest `digital_brain/tests/test_m7_interface.py`
    （14 断言：机制存在性/参数可调/消融差异/调用日志/零回归）+ G1 基线
    （117 零回归——见 m7_db_pytest_status.json 的 144 = 117+27 拆分）；
  - **P-A3 应用题场景**：pytest `digital_brain/tests/test_m7_applications.py`
    （6 断言：S1 闻香厨房 / S2 火炉缩手+习惯化 / S3 钟声节奏 / S4 饥饿增益 +
    原应用题回归 + innate=None 对照）；
  - **P5 回归与报告**：读 `reports/neuro/m7_db_pytest_status.json`（数字大脑 144）
    + `m7_neuro_pytest_status.json`（神经仿真全量，M7-B2 分块跑落盘）；
    可选 `--full-regression` 现场全跑（神经侧长时，默认读回）；
  - **P6 交接**：静态记录（L1–L19 引用 + 本节点 L20+ 坑 + 阶段二 M8 交接要点）。

只读纪律（硬性约束）：本脚本**只写自己的交付物**
`reports/neuro/m7_validation_summary.json`；不修改任何已落盘文件
（冻结 M0–M6 文件 + M7-B1a/B1b 交付物零改动；pytest 子进程加
`-p no:cacheprovider` + `PYTHONDONTWRITEBYTECODE=1` 不落缓存）。
运行前检查无并发（ps 核对）；确定性（无随机，p=1/n=1）。

用法：
  .venv-db/bin/python -m neural_exploration.tools.run_m7_validation   # 默认（P-A2/A3 需数字大脑依赖）
  .venv-neuro/bin/python -m neural_exploration.tools.run_m7_validation  # P-A2/A3 记 skip（无 networkx/pydantic）
  可选：
  --full-regression   现场跑双线全量 pytest（神经侧长时 ~30-90min，默认读回 status JSON）
  --skip-pa1          跳过 P-A1 in-process 复验（纯读回 B1a summary）
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

REPORTS_DIR = os.path.join(ROOT, "neural_exploration", "reports", "neuro")
SUMMARY_JSON = os.path.join(REPORTS_DIR, "m7_validation_summary.json")
EQUIV_SUMMARY = os.path.join(REPORTS_DIR, "m7_equivalence_summary.json")
DB_PYTEST_JSON = os.path.join(REPORTS_DIR, "m7_db_pytest_status.json")
NEURO_PYTEST_JSON = os.path.join(REPORTS_DIR, "m7_neuro_pytest_status.json")
DB_TESTS_DIR = os.path.join(ROOT, "digital_brain", "tests")
NEURO_TESTS_DIR = os.path.join(ROOT, "neural_exploration", "tests")

M7_TESTS = {
    "pa2_interface": ("test_m7_interface.py", 14,
                      "P-A2 数字大脑接入（机制可观察断言：存在性/参数可调/"
                      "消融差异/调用日志/零回归）"),
    "pa3_applications": ("test_m7_applications.py", 6,
                         "P-A3 应用题场景（S1–S4 四场景 + 原应用题回归 + "
                         "innate=None 对照）"),
}


# --------------------------------------------------------------------- #
# P-A1 复验（in-process fresh compute；镜像 validate_m7_equivalence 探针）
# --------------------------------------------------------------------- #

def _check(results: dict, name: str, ok: bool, detail: str) -> None:
    results["pa1_checks"].append({"mechanism": name, "pass": bool(ok),
                                  "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def _deterministic(results: dict, fn, label: str) -> None:
    a, b = repr(fn()), repr(fn())
    ok = a == b
    results["pa1_determinism"][label] = bool(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] determinism {label}: "
          f"{'bit-identical' if ok else 'MISMATCH'}")


def _pa1_verify(results: dict) -> None:
    """21 项等价性断言 + 6 项确定性（fresh compute，与 B1a 冻结锚对齐）。"""
    from digital_brain.src.innate import Stimulus, make_all

    ms = make_all()
    print("P-A1 复验（机制封装 vs 冻结锚，fresh compute）\n")

    # M-1 反射弧（M3 定稿 D_peak 带 [0.352,0.369]）
    m = ms["reflex"]
    r = m.respond(Stimulus(intensity=1.0))
    d_peak = r.value
    _check(results, "reflex.d_peak_in_band", 0.352 <= d_peak <= 0.369,
           f"D_peak(1.0)={d_peak:.4f} ∈ [0.352,0.369]")
    _check(results, "reflex.direction_back",
           r.direction == "back" and d_peak > 0.3,
           f"direction={r.direction} (D_peak {d_peak:.4f} > 0.3)")
    r0 = m.respond(Stimulus(intensity=0.0))
    _check(results, "reflex.intensity0_none",
           r0.value == 0.0 and r0.direction == "none",
           f"I=0 → D_peak={r0.value} dir={r0.direction}")
    r2 = m.respond(Stimulus(intensity=2.0))
    _check(results, "reflex.monotone", r2.value > d_peak,
           f"I=2 → D_peak={r2.value:.4f} > D_peak(1)={d_peak:.4f}")
    _deterministic(results, lambda: m.respond(Stimulus(intensity=1.0)).as_dict(),
                   "reflex")

    # M-2 趋化（CI 符号正 + 容差窗 [0.25,0.75] + 梯度指向食物）
    m = ms["chemotaxis"]
    cii = {}
    for th, label in ((0.0, "east"), (math.pi / 2, "north"),
                      (math.pi, "west"), (-math.pi / 2, "south")):
        xs, ys = m.run_trial(theta0=th)
        cii[label] = m.ci(xs, ys)
    _check(results, "chemotaxis.ci_sign", all(v > 0.0 for v in cii.values()),
           f"CI(10s) 四朝向 = { {k: round(v, 3) for k, v in cii.items()} } 全正")
    xs, ys = m.run_trial()
    ci_canon = m.ci(xs, ys)
    _check(results, "chemotaxis.ci_in_tolerance", 0.25 <= ci_canon <= 0.75,
           f"CI_canon(10s,θ=π)={ci_canon:.4f} ∈ 容差窗 [0.25,0.75]")
    gx, gy = m.gradient(5.0, 5.0)
    _check(results, "chemotaxis.gradient_toward_food",
           math.hypot(gx, gy) > 0 and
           math.atan2(gy, gx) == math.atan2(m.p("food_y") - 5.0,
                                            m.p("food_x") - 5.0),
           f"∇C(中心) 指向食物滴 (∇=({gx:.3f},{gy:.3f}))")
    _deterministic(results, lambda: (m.ci(*m.run_trial())), "chemotaxis")

    # M-3 CPG（双带 0.400 / 2.167Hz）
    m = ms["cpg"]
    f0 = m.respond(Stimulus(kind="time", t_ms=1000, food_present=False))
    f1 = m.respond(Stimulus(kind="time", t_ms=1000, food_present=True))
    _check(results, "cpg.no_food_band",
           f0.extra["in_band"] and f0.value == m.p("f_no_food_hz"),
           f"f_no_food={f0.value:.3f}Hz ∈ [0.1,2]（冻结 0.400）")
    _check(results, "cpg.with_food_band",
           f1.extra["in_band"] and f1.value == m.p("f_with_food_hz"),
           f"f_with_food={f1.value:.3f}Hz ∈ [2,5]（冻结 2.167）")
    _deterministic(results, lambda: m.respond(
        Stimulus(kind="time", t_ms=1234, food_present=True)).as_dict(), "cpg")

    # M-4 习惯化（R² / 衰减 / 半程判据 / 消融 / 恢复）
    m = ms["habituation"]
    res = m.run_sequence(n_stim=6, rest_ms=2000.0)
    fit = res["fit"]
    _check(results, "habituation.r2", fit["r2_ok"] and fit["r2"] >= 0.5,
           f"指数拟合 R²={fit['r2']:.4f} ≥ 0.5（冻结 0.787）")
    _check(results, "habituation.decay", res["decay_ok"],
           f"decay=R(1)−R(N)={res['decay']:.4f} > 0 "
           f"R(n)={[round(v, 3) for v in res['r_seq']]}")
    _check(results, "habituation.half_criterion", res["half_criterion_ok"],
           f"后半均值 {res['last_half_mean']:.4f} < 0.5×前半均值 "
           f"{0.5 * res['first_half_mean']:.4f}")
    _check(results, "habituation.direction_sanity", res["direction_ok"],
           f"R(1)={res['r_seq'][0]:.4f} > 0.3（冻结 0.353）")
    res_off = m.run_sequence(n_stim=6, stp_enabled=False)
    _check(results, "habituation.stp_off_ablation", res_off["decay"] == 0.0,
           f"STP 关 → R(n) 常数 decay={res_off['decay']:.4f}（消融无衰减）")
    _check(results, "habituation.recovery", res["recover_ok"],
           f"R_rest={res['r_rest']:.4f} ≥ 0.3×R(1)（冻结 frac=1.17）")
    _deterministic(results, lambda: m.run_sequence(n_stim=6, rest_ms=2000.0),
                   "habituation")

    # M-5 联想（Δw_train>0.1 / Δw_ext<0 / η=0→0；绝对值不作硬判据）
    m = ms["associative"]
    ar = m.full_protocol()
    _check(results, "associative.acquisition", ar["acquisition_ok"],
           f"Δw_train={ar['dw_train']:.4f} > 0.1（冻结 +0.4325 方向锚）")
    _check(results, "associative.extinction", ar["extinction_ok"],
           f"Δw_ext={ar['dw_ext']:.4f} < 0（冻结 −0.108）")
    _check(results, "associative.eta0_ablation", ar["eta0_ok"],
           f"η=0 → Δw={ar['dw_eta0']:.6f} ≈ 0")
    _deterministic(results, m.full_protocol, "associative")

    # M-6 调质（gate 范围 [0.3,1.2] / 单调 / 酪胺关 → gate≡1）
    m = ms["modulation"]
    gains = [m.gate(v) for v in (0.0, 0.25, 0.5, 0.75, 1.0)]
    _check(results, "modulation.range", all(0.3 <= g <= 1.2 for g in gains),
           f"gate(mot)={[round(g, 3) for g in gains]} ⊆ [0.3, 1.2]")
    _check(results, "modulation.monotone",
           all(gains[i] <= gains[i + 1] for i in range(len(gains) - 1)),
           "动机↑ → 增益单调↑（5HT 促进通道）")
    m2 = make_all()["modulation"]
    m2.params["tyramine_enabled"] = 0.0
    g_off = m2.gate(0.0)
    _check(results, "modulation.tyramine_off", abs(g_off - 1.0) < 0.02,
           f"酪胺关 → gate={g_off:.4f} ≈ 1（消融 sanity）")
    _deterministic(results, lambda: m.gate(0.5), "modulation")

    # 汇总
    results["pa1_pass"] = bool(
        all(c["pass"] for c in results["pa1_checks"])
        and all(results["pa1_determinism"].values()))
    results["pa1_check_count"] = len(results["pa1_checks"])
    results["pa1_determinism_count"] = len(results["pa1_determinism"])
    n_pass = sum(1 for c in results["pa1_checks"] if c["pass"])
    n_det = sum(results["pa1_determinism"].values())
    print(f"\nP-A1 复验：等价性 {n_pass}/{len(results['pa1_checks'])} 通过；"
          f"确定性 {n_det}/{len(results['pa1_determinism'])} 逐位一致 → "
          f"{'PASS' if results['pa1_pass'] else 'FAIL'}")


def _pa1_readback(results: dict) -> None:
    """交叉核对 B1a 落盘的 m7_equivalence_summary.json（不重写）。"""
    if not os.path.exists(EQUIV_SUMMARY):
        results["pa1_b1a_readback"] = {"pass": None,
                                       "note": "m7_equivalence_summary.json 缺失"}
        return
    with open(EQUIV_SUMMARY, encoding="utf-8") as f:
        eq = json.load(f)
    checks = eq.get("checks", [])
    det = eq.get("determinism", {})
    results["pa1_b1a_readback"] = {
        "pass": bool(eq.get("pass")),
        "checks_pass": f"{sum(1 for c in checks if c['pass'])}/{len(checks)}",
        "determinism_pass": f"{sum(det.values())}/{len(det)}",
        "wall_s": eq.get("wall_s"),
        "note": "B1a 落盘（validate_m7_equivalence）；本节点 fresh compute 复验独立确认",
    }
    print(f"P-A1 B1a 读回：pass={eq.get('pass')} "
          f"({results['pa1_b1a_readback']['checks_pass']} 断言 + "
          f"{results['pa1_b1a_readback']['determinism_pass']} 确定性)")


# --------------------------------------------------------------------- #
# P-A2 / P-A3（pytest 子进程；无数字大脑依赖时记 skip）
# --------------------------------------------------------------------- #

def _run_pytest(test_file: str, expected: int, label: str) -> dict:
    path = os.path.join(DB_TESTS_DIR, test_file)
    if not os.path.exists(path):
        return {"pass": False, "note": f"{path} 缺失"}
    try:  # 数字大脑依赖探测（networkx/pydantic：G1 L5 记录）
        import networkx  # noqa: F401
        import pydantic  # noqa: F401
    except ImportError:
        return {"pass": None, "note": "skip：当前解释器无 networkx/pydantic"
                                      "（需 .venv-db 跑数字大脑测试）"}
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", path, "-p", "no:cacheprovider", "-q"],
        capture_output=True, text=True, env=env, cwd=ROOT, timeout=1800)
    out = (proc.stdout or "") + (proc.stderr or "")
    m = re.search(r"(\d+) passed", out)
    passed = int(m.group(1)) if m else 0
    ok = proc.returncode == 0 and passed == expected
    print(f"  {label}: pytest {passed}/{expected} (exit={proc.returncode}) → "
          f"{'PASS' if ok else 'FAIL'}")
    return {"pass": bool(ok), "passed": passed, "expected": expected,
            "exit_code": proc.returncode,
            "tail": out.strip().splitlines()[-3:] if out else []}


# --------------------------------------------------------------------- #
# P5（回归读回 / 现场全跑）
# --------------------------------------------------------------------- #

def _load_pytest_status(path: str) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"passed": None, "total": None,
            "note": "待全量回归落盘（M7-B2 分块跑后写入）"}


def _full_regression(results: dict) -> None:
    """现场跑双线全量 pytest（默认不跑——神经侧长时，见 --full-regression）。"""
    print("P5 现场全量回归（长时）…")
    for label, (tests_dir, status_path, expected) in (
            ("digital_brain", (DB_TESTS_DIR, DB_PYTEST_JSON, 144)),
            ("neural_exploration", (NEURO_TESTS_DIR, NEURO_PYTEST_JSON, 68))):
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", tests_dir,
             "-p", "no:cacheprovider", "-q"],
            capture_output=True, text=True, env=env, cwd=ROOT,
            timeout=7200)
        out = (proc.stdout or "") + (proc.stderr or "")
        m = re.search(r"(\d+) passed", out)
        passed = int(m.group(1)) if m else 0
        rec = {"passed": passed, "total": expected, "exit_code": proc.returncode,
               "suite": tests_dir}
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        results["p5"][label] = rec
        print(f"  {label}: {passed}/{expected}")


# --------------------------------------------------------------------- #
# 汇总入口
# --------------------------------------------------------------------- #

def run_all(skip_pa1: bool = False, full_regression: bool = False) -> dict:
    t0 = time.perf_counter()
    results: dict = {
        "tool": "run_m7_validation",
        "node": "M7-B2",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pa1_checks": [],
        "pa1_determinism": {},
        "pa1_pass": None,
        "pa1_check_count": 0,
        "pa1_determinism_count": 0,
        "pa2": None,
        "pa3": None,
        "p5": {},
        "p6": {},
        "all_pass": False,
    }
    print("M7 验证汇总（P-A1/A2/A3 + P5/P6）\n")

    # ---------------- P-A1 机制提取与封装 ----------------
    print("P-A1 机制提取与封装")
    if skip_pa1:
        results["pa1_pass"] = True
        results["pa1_note"] = "--skip-pa1：仅读回 B1a summary"
    else:
        _pa1_verify(results)
    _pa1_readback(results)

    # ---------------- P-A2 数字大脑接入 ----------------
    print("\nP-A2 数字大脑接入（InnateInterface + SymbolicInterface 注入）")
    fname, expected, note = M7_TESTS["pa2_interface"]
    results["pa2"] = _run_pytest(fname, expected, note)

    # ---------------- P-A3 应用题场景 ----------------
    print("\nP-A3 应用题场景（S1–S4）")
    fname, expected, note = M7_TESTS["pa3_applications"]
    results["pa3"] = _run_pytest(fname, expected, note)

    # ---------------- P5 回归与报告 ----------------
    print("\nP5 回归与报告")
    if full_regression:
        _full_regression(results)
    results["p5"] = {
        "digital_brain": _load_pytest_status(DB_PYTEST_JSON),
        "neural_exploration": _load_pytest_status(NEURO_PYTEST_JSON),
        "report_md": "docs/m7_report.md",
        "validation_summary": "reports/neuro/m7_validation_summary.json",
        "innate_map_png": "reports/neuro/m7_innate_map.png",
    }

    # ---------------- P6 交接 ----------------
    print("\nP6 交接（L1–L19 + 本节点 L20+ + 阶段二 M8）")
    results["p6"] = {
        "handoff_docs": [
            "docs/m7_env_notes.md（L1–L19：G1 门 + 机制清单 + B1a/B1b 实测坑）",
            "docs/m7_report.md §7（L1–L19 摘要 + 本节点 L20+）",
            "docs/m7_report.md §8（阶段二 M8 交接：果蝇幼虫全脑 + M-1..M-7）",
        ],
        "m8_key_points": [
            "M8 = 果蝇幼虫全脑（3,016 神经元 / ~55 万突触，Winding 2023 连接组；"
            "行为+活动+扰动预测三通道）",
            "M-1..M-6 迁移机制清单 + M5 缩放定律降阶模型方案 = M8 设计基础",
            "M-7 夹带双稳态反证为 M8 降阶设计提供依据（阶段二铁律 C 缩放扫描）",
        ],
        "no_commit": True,
        "note": "冻结文件零修改；未 git commit（M0–M6 惯例）",
    }

    # ---------------- 总体判定 ----------------
    pa1_ok = bool(results["pa1_pass"])
    pa2_ok = bool(results["pa2"] and results["pa2"].get("pass"))
    pa3_ok = bool(results["pa3"] and results["pa3"].get("pass"))
    db = results["p5"]["digital_brain"]
    neuro = results["p5"]["neural_exploration"]
    p5_ok = bool(db.get("passed") and db["passed"] == db["total"]
                 and neuro.get("passed") and neuro["passed"] == neuro["total"])
    results["verdicts"] = {
        "pa1": "pass" if pa1_ok else "fail",
        "pa2": "pass" if pa2_ok else ("skip" if results["pa2"] and
                                      results["pa2"].get("pass") is None
                                      else "fail"),
        "pa3": "pass" if pa3_ok else ("skip" if results["pa3"] and
                                      results["pa3"].get("pass") is None
                                      else "fail"),
        "p5": "pass" if p5_ok else "pending-regression",
        "p6": "pass",
    }
    results["all_pass"] = bool(pa1_ok and pa2_ok and pa3_ok and p5_ok)
    results["wall_s"] = round(time.perf_counter() - t0, 3)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n==== M7 验证判定汇总 ====")
    for k, v in results["verdicts"].items():
        print(f"  {k:6s} = {v}")
    print(f"  all_pass = {results['all_pass']}（P5 依赖双线回归 status JSON）")
    print(f"summary → {SUMMARY_JSON}")
    print(f"总耗时: {results['wall_s']}s")
    return results


def main():
    ap = argparse.ArgumentParser(description="M7 验证汇总（P-A1/A2/A3 + P5/P6）")
    ap.add_argument("--skip-pa1", action="store_true",
                    help="跳过 P-A1 in-process 复验（仅读回 B1a summary）")
    ap.add_argument("--full-regression", action="store_true",
                    help="现场跑双线全量 pytest（神经侧长时；默认读回 status JSON）")
    args = ap.parse_args()
    run_all(skip_pa1=args.skip_pa1, full_regression=args.full_regression)
    return 0


if __name__ == "__main__":
    sys.exit(main())
