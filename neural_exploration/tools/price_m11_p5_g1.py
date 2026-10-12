"""**★（G1 正式矩阵）GPU-h 精确重算**（全部由脚本计算并落盘；**不得手算** —— R76）。

**输入（逐行给 provenance；实测项一律标明是谁测的/哪个口径）**
| 量 | 值 | 来源 |
|---|---|---|
| `ms/step`（**step** 口径，N=8,192）| dose 0/1/4 = 2.5585 / 3.2583 / 4.2056 | **主agent 亲跑**（本任务 §② 给定）|
| `ms/step`（**chunk** 口径，N=8,192）| dose 0/4 = 0.6545 / 0.9720 | **主agent 亲跑** |
| `ms/step`（chunk，N=8,192）| dose 0.5 / 1 / 2 = **本轮实测**（`price_8192_T2000ms_d*.json`）| 本节点（生产者产物；**T 缩短为 2 s，仅测 ms/step**，已在 run_id/笔记标注）|
| 档因子（8k→31k，step）| dose 0/1/4 = 0.6817 / 0.6017 / 0.4143 | **主agent 实测**（强剂量依赖 ⇒ **不作外推**；★ 只用 8k 档）|

**口径纪律（R77）**：**"加速比"必须写全口径** ——
  `wall_ratio` = 全网格总墙钟之比（含 settle 与全部臂），
  `ms_per_step_ratio` = 逐剂量 `ms/step` 比值（按臂步数加权）。
  两者**不是同一个量**，必须在同一行里分别给出（R77：墙钟 3.35× vs `ms/step` 3.91×）。

**口径（R78）**：生产口径 = `chunk`（判据口径 = `step`，评估器只读产物不跑仿真）；
E 段剂量两端孪生臂 `pop_trace` 逐位相同（`ndiff=0/200000`）⇒ 全网格可用 chunk。

用法：
    ./.venv-m9/bin/python -m neural_exploration.tools.price_m11_p5_g1
产物：`data/m11_p5_g1_pricing.csv` / `.json`
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "neural_exploration", "data")
DT_MS = 0.05
N = 8192
BUDGET_GPU_H = 10.0

# ---- 实测锚（step 口径；主agent 亲跑）----
MS_STEP = {0.0: 2.5585, 1.0: 3.2583, 4.0: 4.2056}
# ---- 实测锚（chunk 口径；0/4 主agent 亲跑，0.5/1/2 由本轮生产者产物读回）----
MS_CHUNK_MASTER = {0.0: 0.6545, 4.0: 0.9720}


def _load_price_arms() -> dict:
    """读回本轮实测的 chunk ms/step（dose=0.5/1/2）。"""
    out = {}
    d = os.path.join(DATA_DIR, "m11_p5_runs")
    if not os.path.isdir(d):
        return out
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json") or fn.endswith(".spec.json"):
            continue
        try:
            with open(os.path.join(d, fn)) as f:
                a = json.load(f)
        except Exception:  # noqa: BLE001
            continue
        if a.get("delivery") != "chunk" or int(a.get("n_neuron", 0)) != N:
            continue
        dose = float(a.get("drive_mult"))
        ms = a.get("ms_per_step_steady")
        if ms is None:
            continue
        out.setdefault(dose, []).append(dict(arm=a.get("arm_id"), ms=float(ms),
                                             T_ms=a.get("T_ms"), wall_s=a.get("wall_s"),
                                             source="producer artifact"))
    return out


def interp_chunk(dose: float, pts: dict) -> tuple:
    """**实测优先、插值兜底**（标注返回的 method；log-dose 线性插值）。"""
    if dose in pts:
        return float(pts[dose]), "measured"
    xs = sorted(pts)
    if dose <= xs[0]:
        return float(pts[xs[0]]), "edge-hold(measured@%.2f)" % xs[0]
    if dose >= xs[-1]:
        return float(pts[xs[-1]]), "edge-hold(measured@%.2f)" % xs[-1]
    lo = max(x for x in xs if x < dose)
    hi = min(x for x in xs if x > dose)
    import math
    t = (math.log(dose + 1.0) - math.log(lo + 1.0)) / (math.log(hi + 1.0) - math.log(lo + 1.0))
    v = math.exp(math.log(pts[lo]) + t * (math.log(pts[hi]) - math.log(pts[lo])))
    return float(v), "interp_logdose(%.2f..%.2f)" % (lo, hi)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=os.path.join(DATA_DIR, "m11_p5_plan_g1.json"))
    ap.add_argument("--budget-gpu-h", type=float, default=BUDGET_GPU_H)
    ap.add_argument("--out-json", default=os.path.join(DATA_DIR, "m11_p5_g1_pricing.json"))
    ap.add_argument("--out-csv", default=os.path.join(DATA_DIR, "m11_p5_g1_pricing.csv"))
    args = ap.parse_args(argv)
    with open(args.plan) as f:
        plan = json.load(f)
    settle = float(plan["settle_ms"])

    # ---- 组装 chunk 锚点（实测优先）----
    measured = _load_price_arms()
    ms_chunk = dict(MS_CHUNK_MASTER)
    ms_chunk_src = {0.0: "主agent实测", 4.0: "主agent实测"}
    for dose, rows in measured.items():
        if rows:
            ms_chunk[dose] = rows[0]["ms"]
            _arm = str(rows[0]["arm"])
            _who = ("主agent E 段产物" if _arm.startswith("e_") else "本轮生产者产物")
            ms_chunk_src[dose] = "%s(%s, T=%sms)" % (_who, _arm, rows[0]["T_ms"])
    for k in MS_STEP:
        if k in ms_chunk and k not in ms_chunk_src:
            ms_chunk_src[k] = "本轮实测/派生"

    rows = []
    for a in plan["arms"]:
        dose = float(a["drive_mult"])
        steps = int(round((settle + float(a["T_ms"])) / DT_MS))
        ms_c, src_c = interp_chunk(dose, ms_chunk)
        ms_s, _ = (MS_STEP[dose], "主agent实测") if dose in MS_STEP else \
            interp_chunk(dose, MS_STEP)
        wall_c = steps * ms_c / 1000.0
        wall_s = steps * ms_s / 1000.0
        rows.append(dict(stage=a["stage"], run_id=a["run_id"], config_id=a["config_id"],
                         T_ms=float(a["T_ms"]), drive_mult=dose, role=a["role"],
                         steps=steps, ms_per_step_chunk=round(ms_c, 4),
                         ms_per_step_step=round(ms_s, 4), chunk_ms_source=src_c,
                         wall_s_chunk=round(wall_c, 1), wall_s_step=round(wall_s, 1),
                         gpu_h_chunk=round(wall_c / 3600.0, 5),
                         gpu_h_step=round(wall_s / 3600.0, 5)))
    segs = {}
    for r in rows:
        s = segs.setdefault(r["stage"], dict(n=0, steps=0, gpu_h_chunk=0.0, gpu_h_step=0.0))
        s["n"] += 1
        s["steps"] += r["steps"]
        s["gpu_h_chunk"] += r["gpu_h_chunk"]
        s["gpu_h_step"] += r["gpu_h_step"]
    tot = dict(n_arms=len(rows), steps=sum(r["steps"] for r in rows),
               gpu_h_chunk=round(sum(r["gpu_h_chunk"] for r in rows), 4),
               gpu_h_step=round(sum(r["gpu_h_step"] for r in rows), 4))
    # ---- 加速比：两个口径分列（R77）----
    w_ratio = tot["gpu_h_step"] / tot["gpu_h_chunk"] if tot["gpu_h_chunk"] else None
    num = sum(r["steps"] * r["ms_per_step_step"] for r in rows)
    den = sum(r["steps"] * r["ms_per_step_chunk"] for r in rows)
    ms_ratio = (num / den) if den else None
    # 逐剂量 ms/step 比值
    per_dose = {str(d): round(MS_STEP[d] / interp_chunk(d, ms_chunk)[0], 4) for d in sorted(MS_STEP)}
    # ---- 敏感性：**插值剂量**（未实测者）±15% ⇒ 合计范围（脚本算，不手算；R76）----
    interp_doses = sorted({float(r["drive_mult"]) for r in rows if r["chunk_ms_source"].startswith("interp")})
    _share = sum(r["gpu_h_chunk"] for r in rows if float(r["drive_mult"]) in interp_doses) / max(1e-12, tot["gpu_h_chunk"])
    sens = dict(interpolated_doses=interp_doses, n_arms_interpolated=len([r for r in rows if float(r["drive_mult"]) in interp_doses]),
                interpolated_share_of_total=round(_share, 4),
                total_lo=round(tot["gpu_h_chunk"] * (1 - 0.15 * _share), 4),
                total_hi=round(tot["gpu_h_chunk"] * (1 + 0.15 * _share), 4),
                still_within_10_at_hi=bool(tot["gpu_h_chunk"] * (1 + 0.15 * _share) <= args.budget_gpu_h),
                note="±15% 取自本轮实测的插值误差量级（dose=1 实测 0.9482 vs 插值 ~0.82 ⇒ 约 +15%）")

    out = dict(generated_by="tools/price_m11_p5_g1.py", plan=os.path.basename(args.plan),
               plan_sha256=hashlib.sha256(open(args.plan, "rb").read()).hexdigest()[:32],
               n_neuron=N, production_delivery="chunk", criterion_delivery="step",
               ms_step_measured=MS_STEP, ms_chunk_used=ms_chunk, ms_chunk_source=ms_chunk_src,
               segments={k: dict(n=v["n"], steps=v["steps"], gpu_h_chunk=round(v["gpu_h_chunk"], 4),
                                 gpu_h_step=round(v["gpu_h_step"], 4)) for k, v in segs.items()},
               total=tot, speedup=dict(wall_ratio=round(w_ratio, 4) if w_ratio else None,
                                       ms_per_step_ratio_weighted=round(ms_ratio, 4) if ms_ratio else None,
                                       per_dose_ms_ratio=per_dose,
                                       discipline="R77：wall_ratio 与 ms_per_step_ratio 是两个量，分列"),
               budget_gpu_h=args.budget_gpu_h,
               verdict=dict(gpu_h=dict(chunk=tot["gpu_h_chunk"], step=tot["gpu_h_step"]),
                            within_10_gpu_h_chunk=bool(tot["gpu_h_chunk"] <= args.budget_gpu_h),
                            within_10_gpu_h_step=bool(tot["gpu_h_step"] <= args.budget_gpu_h),
                            margin_chunk_gpu_h=round(args.budget_gpu_h - tot["gpu_h_chunk"], 4),
                            note="chunk = 生产口径（R78）⇒ 以 chunk 行为准判定；step 行并列供审计"),
               sensitivity=sens,
               rows=rows)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("段 | 臂数 | 步数 | chunk GPU-h | step GPU-h")
    for k in ("p1", "p2", "b_r1", "b_r2", "b_r3", "b_r4", "c", "d"):
        if k in segs:
            v = segs[k]
            print("%-6s | %3d | %8d | %8.4f | %8.4f" % (k, v["n"], v["steps"],
                                                        v["gpu_h_chunk"], v["gpu_h_step"]))
    print("%-6s | %3d | %8d | %8.4f | %8.4f" % ("合计", tot["n_arms"], tot["steps"],
                                                tot["gpu_h_chunk"], tot["gpu_h_step"]))
    print("加速比（两个量，R77）：wall_ratio=%.4f | ms/step 加权比=%.4f | 逐剂量=%s" % (
        w_ratio, ms_ratio, per_dose))
    print("chunk 锚点来源：", ms_chunk_src)
    print("判定（budget=%.1f GPU-h）：chunk %s（余量 %.4f）｜step %s" % (
        args.budget_gpu_h, "≤ 预算 ✅" if tot["gpu_h_chunk"] <= args.budget_gpu_h else "超预算 ❌",
        args.budget_gpu_h - tot["gpu_h_chunk"],
        "≤ 预算 ✅" if tot["gpu_h_step"] <= args.budget_gpu_h else "超预算 ❌"))
    print("敏感性（插值剂量 %s，%d/%d 臂，占合计 %.1f%%）⇒ 合计 %s .. %s GPU-h，"
          "上界仍 ≤10 = %s" % (sens["interpolated_doses"], sens["n_arms_interpolated"],
                              tot["n_arms"], 100 * sens["interpolated_share_of_total"],
                              sens["total_lo"], sens["total_hi"], sens["still_within_10_at_hi"]))
    print("json ->", args.out_json, "| csv ->", args.out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
