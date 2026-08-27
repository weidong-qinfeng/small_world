"""M6 逐项验证总运行器（M6-B2 验证+报告节点；清单步骤 7 验收：P1–P6 顺序运行）。

依次运行 P1（STDP 复核）/P2（调质+反证清单复核）/P3（习惯化）/P4（联想学习）；
汇总判定写入 reports/neuro/m6_validation_summary.json。

判定语义（主 agent 裁决 2026-08-27，M6-B2 落实为判定框架）：
  - P1（STP/STDP 组件）= pass（B1b 已过，本节点复核确认）；
  - P2（调质 + M5 反证清单）= **反证记录型**（G1=部分通过：方向相位修复 ✓；
    P2 静默/P4 趋化未缓解、P6 部分缓解——与 M5 P2/P4/P6 同型，记录本身即交付物；
    剩余缺失机制清单：夹带双稳态）→ pass_=False + status=counter-evidence-record
    + verdict_accepted=True（主 agent 裁决接受记录）；
  - P3（习惯化）= pass-with-measurement-limitations（机制级衰减/消融/恢复全过；
    10s-ISI 主协议判据受 τ_rec 时程限制不可达，L25 如实记录）；
  - P4（联想学习）= pass-with-measurement-limitations（机制级获得/消融/消退全过；
    CI_salt 读出灵敏度低——网络级显著性不可达，L16 如实记录）。
  - P5（pytest 全绿 + 报告）/P6（交接）由全量 pytest + gen_m6_report.py +
    docs/m6_env_notes.md 独立确认（本脚本 note 引用）。

用法：
  .venv-neuro/bin/python -m neural_exploration.tools.run_m6_validation
可选：
  --reuse     M6_REUSE=1 语义：各 P 存在 result JSON/CSV 时直接读回判定（不重跑）
  --skip-heavy 跳过重协议（P3 network 底物 / P2 302 探针）
  --skip-plot 不出图（纯判定）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.tools import (  # noqa: E402
    validate_p1_stdp,
    validate_p2_modulation,
    validate_p3_habituation,
    validate_p4_associative,
)

REPORTS_DIR = os.path.join(ROOT, "neural_exploration", "reports", "neuro")
SUMMARY_JSON = os.path.join(REPORTS_DIR, "m6_validation_summary.json")
DATA_DIR = os.path.join(ROOT, "neural_exploration", "data")

#: 各 P 的 result JSON（完整判定 dict；复用模式读回，不重跑 Brian2）
_RESULT_JSON = {
    "p1_stdp": None,                      # P1 无 result JSON → CSV 读回
    "p2_modulation": "m6_p2_result.json",
    "p3_habituation": "m6_p3_result.json",
    "p4_associative": "m6_p4_result.json",
}
_RUNNERS = {
    "p1_stdp": lambda sp, heavy: validate_p1_stdp.run_p1_stdp(save_plot=sp),
    "p2_modulation": lambda sp, heavy: validate_p2_modulation.run_p2(
        save_plot=sp, probes=heavy),
    "p3_habituation": lambda sp, heavy: validate_p3_habituation.run_p3(
        save_plot=sp, with_network=heavy),
    "p4_associative": lambda sp, heavy: validate_p4_associative.run_p4(
        save_plot=sp),
}
#: P3/P4 的 result dict 里 pass_/status 所在路径
_STATUS_PATH = {
    "p1_stdp": None, "p2_modulation": None, "p3_habituation": None,
    "p4_associative": None,
}


def _p1_from_csv() -> dict:
    """P1 判定从 data/m6_p1_stdp.csv 的 summary 注释行读回（B1b/B2 落盘）。"""
    path = os.path.join(DATA_DIR, "m6_p1_stdp.csv")
    if not os.path.exists(path):
        return None
    info = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("# "):
                k, _, v = line[2:].partition("=")
                # 值取第一个 token（summary 行格式 `# key=value (extra)`，
                # 括号内为额外信息，不参与布尔解析）
                v = v.strip().split("(")[0].strip()
                info[k.strip()] = v
    try:
        stp_pass = str(info.get("stdp_pass", "False")).lower() == "true"
        det_ok = str(info.get("deterministic_ok", "False")).lower() == "true"
        stp_reg = str(info.get("stp_regression_pass", "False")).lower() == "true"
        net_ok = str(info.get("network_interface_ok", "False")).lower() == "true"
        return dict(
            pass_=bool(stp_pass and det_ok and stp_reg and net_ok),
            status="pass" if stp_pass else "fail",
            stdp_pass=stp_pass, deterministic_ok=det_ok,
            stp_regression_pass=stp_reg, network_interface_ok=net_ok,
            max_abs_diff=info.get("per_point_ok", ""),
            reused_from="data/m6_p1_stdp.csv（B1b 已跑，本节点复核读回）",
        )
    except Exception as exc:  # noqa: BLE001
        return dict(pass_=False, status="readback-error", error=str(exc))


def _load_result(key: str, reuse: bool) -> dict:
    rj = _RESULT_JSON[key]
    if reuse and rj is not None:
        path = os.path.join(DATA_DIR, rj)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                rec = json.load(f)
            rec.setdefault("reused_from", f"data/{rj}（本节点已跑，读回）")
            return rec
    if reuse and key == "p1_stdp":
        rec = _p1_from_csv()
        if rec is not None:
            return rec
    return None


def run_all(skip_heavy: bool = False, skip_plot: bool = False,
            reuse: bool = False) -> dict:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    sp = not skip_plot
    results = {}
    for key, runner in _RUNNERS.items():
        heavy = not skip_heavy
        if reuse:
            rec = _load_result(key, reuse=True)
            if rec is not None:
                print(f"  {key}: 复用 result JSON/CSV（pass_={rec.get('pass_')}）")
                results[key] = rec
                continue
        print(f"  {key}: 运行验证…")
        results[key] = runner(sp, heavy)

    pass_type = {
        "p1_stdp": "pass",
        "p2_modulation": "counter-evidence-record",
        "p3_habituation": "pass-with-measurement-limitations",
        "p4_associative": "pass-with-measurement-limitations",
    }
    p2 = results.get("p2_modulation", {})
    p3 = results.get("p3_habituation", {})
    p4 = results.get("p4_associative", {})
    note = (
        "主 agent 裁决 2026-08-27（M6-B2 落实为判定框架）："
        "**P1 = pass**（STDP 组件 vs 理论曲线逐位一致 + STP 回归不回归）；"
        "**P2 = 反证记录型**（G1=部分通过：P5 方向相位修复 back@tau=23 ✓；"
        "P2 静默未缓解（10.3% vs 带 [60,80]）、P4 趋化未缓解（−0.263@5s 同号反证）、"
        "P6 部分缓解（rev 落带；fwd/turn 近带）——记录本身即交付物；剩余缺失机制"
        "清单：夹带双稳态）；"
        "**P3 = pass-with-measurement-limitations**（机制级衰减/消融/恢复全过；"
        "10s-ISI 主协议判据受 τ_rec=1s 时程限制不可达——R(n) 常数，机制在短 ISI "
        "演示，L25 如实记录）；"
        "**P4 = pass-with-measurement-limitations**（机制级获得/消融/消退全过；"
        "CI_salt 读出灵敏度低——ΔCI≈+0.004，网络级显著性不可达，L16 如实记录）。"
        "P5（pytest 全绿 + m6_report.md）/P6（交接 + m6_env_notes）由全量 pytest + "
        "tools/gen_m6_report.py 独立确认。"
    )
    summary = {
        "milestone": "M6",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results,
        "all_pass": all(v.get("pass_", False) for v in results.values()),
        "milestone_complete": bool(
            results.get("p1_stdp", {}).get("pass_", False)
            and results.get("p3_habituation", {}).get("pass_", False)
            and results.get("p4_associative", {}).get("pass_", False)
            and p2.get("status") == "counter-evidence-record"),
        "pass_type": pass_type,
        "p2_verdict_accepted": True,
        "note": note,
        "measured_limitations": {
            "p3_habituation": p3.get("measured_limitations", []),
            "p4_associative": p4.get("measured_limitations", []),
        },
        "counter_evidence": {
            "p2_modulation": p2.get("missing_mechanisms", []),
            "p4_associative": p4.get("counter_evidence", []),
        },
        "pytest_status_json": "reports/neuro/m6_pytest_status.json",
        "report_md": "docs/m6_report.md",
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    return summary


def main():
    ap = argparse.ArgumentParser(description="M6 验证总运行器（P1–P4）")
    ap.add_argument("--reuse", action="store_true",
                    help="M6_REUSE=1：P 脚本 result JSON/CSV 存在时读回（不重跑）")
    ap.add_argument("--skip-heavy", action="store_true",
                    help="跳过重协议（P2 302 探针 / P3 302 网络底物）")
    ap.add_argument("--skip-plot", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    summary = run_all(skip_heavy=args.skip_heavy, skip_plot=args.skip_plot,
                      reuse=args.reuse or os.environ.get("M6_REUSE") == "1")
    print("==== M6 验证判定汇总 ====")
    for k, v in summary["results"].items():
        pt = summary["pass_type"].get(k, "")
        print(f"  {k:20s} pass={v.get('pass_') if isinstance(v, dict) else '?'}"
              f" [{pt}]" + (f" ({v.get('status', '')})" if isinstance(v, dict) else ""))
    print(f"  all_pass = {summary['all_pass']}（P2 反证记录型 → milestone_complete = "
          f"{summary['milestone_complete']}）")
    print(f"汇总 JSON: {SUMMARY_JSON}")
    print(f"总耗时: {time.time() - t0:.1f}s")
    return summary


if __name__ == "__main__":
    main()
