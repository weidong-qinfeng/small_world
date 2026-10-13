"""**L1（σ 形状杠杆）计划与全段预算**（R76：中间值脚本算；R80 §⑤2：先做量纲自检）。

单位价口径（B3 预注册 `L1_design.budget.unit_prices`）：
- `chunk@8k`、**bias=1.00**（G1 实测）：dose 0/0.5/1/2/4 = 0.6545 / 0.8130 / 0.9482 / 0.9587 / 0.9720 ms/step；
- **bias=0.70**：L0 干跑**实测** dose=1 ⇒ 0.7964 ms/step（比值 0.8399 相对 bias=1.00 同剂量）；
- 预注册取 **mid = 全臂 0.7964**、**upper = 全臂 0.9720**；本工具**都算**，另给"按剂量感知"的下界。

**量纲恒等式**：`gpu_h = steps × ms_per_step / 3.6e6`（1 GPU-h = 3600 s = 3.6e6 ms；因子 2.7778e-7）。
自检算例：24 × 210,000 × 0.7964 / 3.6e6 必须等于脚本算出的 `scan_mid`，并与预注册的 1.115 对照。

用法：`./.venv-m11/bin/python -m neural_exploration.tools.price_m11_p5_l1`
产物：`data/m11_p5_l1_pricing.json` / `.csv`
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "neural_exploration", "data")
MS_PER_GPU_H = 3.6e6
BIAS1_CHUNK = {0.0: 0.6545, 0.5: 0.8130, 1.0: 0.9482, 2.0: 0.9587, 4.0: 0.9720}   # G1 实测
BIAS07_DOSE1 = 0.7964      # L0 干跑实测（bias=0.70/dose=1/chunk/MPS）
UPPER = 0.9720             # 预注册上界：全臂按最贵实测 chunk 单价
RATIO = BIAS07_DOSE1 / BIAS1_CHUNK[1.0]
PREREG = dict(scan=1.115, scan_upper=1.361, final_confirm=4.745, total_mid=6.907, total_upper=8.322,
              envelope=10.0)
FINAL_CONFIRM_DOSES = (0.0, 0.5, 1.0, 2.0, 4.0)
TS = (10_000.0, 20_000.0, 40_000.0)


def steps_of(T_ms: float) -> int:
    return int(round((500.0 + T_ms) / 0.05))


def main() -> int:
    plan_p = os.path.join(DATA_DIR, "m11_p5_plan_l1.json")
    with open(plan_p) as f:
        plan = json.load(f)
    psha = hashlib.sha256(open(plan_p, "rb").read()).hexdigest()
    print("== L1 预算（脚本算；plan sha256 %s）==" % psha[:16])
    # ---- 量纲自检 ----
    w = steps_of(10_000.0)
    self_check = w * BIAS07_DOSE1 / MS_PER_GPU_H
    print("  [量纲自检] gpu_h = steps×ms/3.6e6 ⇒ 210000×0.7964 = %.9f GPU-h；bias 校正比值 = %.6f"
          % (self_check, RATIO))

    rows = []
    scan_steps = sum(steps_of(a["T_ms"]) for a in plan["arms"])
    n_scan = len(plan["arms"])
    for a in plan["arms"]:
        d = float(a["drive_mult"])
        rows.append(dict(segment="L1_sigma_scan", run_id=a["run_id"], config_id=a["config_id"],
                         bias_log_sigma=a["bias_log_sigma"], drive_mult=d, role=a["role"],
                         steps=steps_of(a["T_ms"]),
                         ms_mid=BIAS07_DOSE1, ms_dose_aware=BIAS1_CHUNK[d] * RATIO,
                         gpu_h_mid=steps_of(a["T_ms"]) * BIAS07_DOSE1 / MS_PER_GPU_H,
                         gpu_h_upper=steps_of(a["T_ms"]) * UPPER / MS_PER_GPU_H,
                         gpu_h_dose_aware=steps_of(a["T_ms"]) * BIAS1_CHUNK[d] * RATIO / MS_PER_GPU_H))
    scan_mid = sum(r["gpu_h_mid"] for r in rows)
    scan_upper = sum(r["gpu_h_upper"] for r in rows)
    scan_dose = sum(r["gpu_h_dose_aware"] for r in rows)
    # ---- final confirm（45 run：5 剂量 × 3 实现 × 3 T，最优 σ 档）----
    fc_steps = sum(steps_of(T) for T in TS for _ in FINAL_CONFIRM_DOSES for _ in range(3))
    fc_mid = sum(steps_of(T) * BIAS1_CHUNK[d] * RATIO for T in TS for d in FINAL_CONFIRM_DOSES
                 for _ in range(3)) / MS_PER_GPU_H
    # ---- 降级方案（3 剂量 d0/d1/d4：27 run）----
    fc_dg_steps = sum(steps_of(T) for T in TS for _ in (0.0, 1.0, 4.0) for _ in range(3))
    fc_dg_mid = sum(steps_of(T) * BIAS1_CHUNK[d] * RATIO for T in TS for d in (0.0, 1.0, 4.0)
                    for _ in range(3)) / MS_PER_GPU_H
    # ---- 孪生臂（新工况 bias=0.70：剂量端点 d0/d4 × {step,chunk} = 4 run）----
    twin_steps = 4 * steps_of(10_000.0)
    twin_mid = sum(steps_of(10_000.0) * BIAS1_CHUNK[d] * RATIO for d in (0.0, 4.0)
                   for _ in range(2)) / MS_PER_GPU_H
    refine_steps = 12 * steps_of(10_000.0)
    refine_mid = 12 * steps_of(10_000.0) * BIAS07_DOSE1 / MS_PER_GPU_H

    # ---- 实测旁证：本次 CPU 探针的 chunk 单价（**不同环境，非同口径价格**）----
    cpu_prices = []
    for p in glob.glob(os.path.join(DATA_DIR, "m11_p5_l1_sigma_probe", "*.json")):
        if p.endswith("spec.json"):
            continue
        a = json.load(open(p))
        cpu_prices.append(dict(run_id=a["arm_id"], device=a["device"],
                               ms_per_step=a["ms_per_step_steady"], sigma=a["net"].get("bias_log_sigma")))
    cpu_note = ("本次 σ 探针在 **CPU** 侧实测 chunk = %s ms/step（N=8192/T=1 s），而预算用的是 **MPS** "
                "实测 0.7964/0.9482 ⇒ **两者不是同口径、不可直接比价**；且 R78 要求跨设备可比性须由"
                "孪生臂（同 spec、两设备、pop_trace 逐位/统计一致）先证 —— 属主 agent 的 MPS 权限范围"
                % ", ".join("%.4f(σ=%s)" % (c["ms_per_step"], c["sigma"]) for c in cpu_prices)
                if cpu_prices else "未找到探针产物")
    print("  [实测旁证] " + cpu_note)

    fc_upper = sum(steps_of(T) * UPPER for T in TS for _ in FINAL_CONFIRM_DOSES
                   for _ in range(3)) / MS_PER_GPU_H
    twin_upper = sum(steps_of(10_000.0) * UPPER for _ in range(4)) / MS_PER_GPU_H
    refine_upper = 12 * steps_of(10_000.0) * UPPER / MS_PER_GPU_H
    # **保守口径**（全臂按 bias=1.00 的 G1 单价、不做 bias 修正）——与预注册/G1 预算同口径，便于对照
    fc_b1 = sum(steps_of(T) * BIAS1_CHUNK[d] for T in TS for d in FINAL_CONFIRM_DOSES
                for _ in range(3)) / MS_PER_GPU_H
    twin_b1 = sum(steps_of(10_000.0) * BIAS1_CHUNK[d] for d in (0.0, 4.0)
                  for _ in range(2)) / MS_PER_GPU_H
    refine_b1 = 12 * steps_of(10_000.0) * BIAS1_CHUNK[1.0] / MS_PER_GPU_H
    total_conservative = scan_upper + fc_b1 + twin_b1 + refine_b1
    total_mid = scan_mid + fc_mid + twin_mid + refine_mid
    total_upper = scan_upper + fc_upper + twin_upper + refine_upper
    total_dg = scan_dose + fc_dg_mid + twin_mid + refine_mid
    out = dict(tool="tools/price_m11_p5_l1.py", generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
               plan_path="neural_exploration/data/m11_p5_plan_l1.json", plan_sha256=psha,
               units=dict(gpu_h="GPU-hour=3.6e6 ms", ms_per_step="millisecond per step",
                          note="gpu_h = steps × ms_per_step / 3.6e6"),
               unit_prices=dict(bias1_chunk_ms=BIAS1_CHUNK, bias07_dose1_ms=BIAS07_DOSE1,
                                bias07_over_bias1_ratio=RATIO, upper_ms=UPPER,
                                source="G1 实测 + L0 干跑实测（MPS）"),
               dimension_self_check=dict(identity="gpu_h = steps×ms_per_step/3.6e6",
                                         worked_example=dict(steps=w, ms=BIAS07_DOSE1,
                                                             gpu_h=self_check),
                                         prereg_scan_mid=PREREG["scan"]),
               scan=dict(n_runs=n_scan, steps=scan_steps, gpu_h_mid=scan_mid,
                         gpu_h_upper=scan_upper, gpu_h_dose_aware=scan_dose,
                         prereg_mid=PREREG["scan"], prereg_upper=PREREG["scan_upper"],
                         delta_vs_prereg_mid=scan_mid - PREREG["scan"]),
               final_confirm=dict(n_runs=45, steps=fc_steps, gpu_h_mid=fc_mid,
                                  prereg_mid=PREREG["final_confirm"],
                                  downgrade_3doses=dict(n_runs=27, steps=fc_dg_steps, gpu_h_mid=fc_dg_mid)),
               twin_arms=dict(n_runs=4, steps=twin_steps, gpu_h_mid=twin_mid),
               refine_round=dict(n_runs=12, steps=refine_steps, gpu_h_mid=refine_mid),
               totals=dict(mid_gpu_h=total_mid, upper_gpu_h=total_upper,
                           conservative_gpu_h=total_conservative,
                           conservative_parts=dict(scan=scan_upper, final_confirm=fc_b1,
                                                   twin=twin_b1, refine=refine_b1),
                           upper_parts=dict(scan=scan_upper, final_confirm=fc_upper,
                                            twin=twin_upper, refine=refine_upper),
                           downgraded_total_gpu_h=total_dg,
                           prereg_total_mid=PREREG["total_mid"], prereg_total_upper=PREREG["total_upper"],
                           envelope_gpu_h=PREREG["envelope"],
                           within_envelope=total_mid <= PREREG["envelope"],
                           wall_clock_note="MPS 串行 ≈ mid GPU-h 小时；另加逐臂新进程开销（本节点实测 "
                                           "8k 短臂 6–7 s，10 s 档约 30–60 s/臂）"),
               cpu_probe_cross_environment=cpu_note, cpu_probe_prices=cpu_prices,
               rows=rows)
    jp = os.path.join(DATA_DIR, "m11_p5_l1_pricing.json")
    cp = os.path.join(DATA_DIR, "m11_p5_l1_pricing.csv")
    with open(jp, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    with open(cp, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        wtr.writeheader()
        for r in rows:
            wtr.writerow(r)
    print("  L1 σ 扫描（%d 臂/%d 步）：mid %.4f｜upper %.4f｜剂量感知 %.4f GPU-h（预注册 mid %.3f / upper %.3f）"
          % (n_scan, scan_steps, scan_mid, scan_upper, scan_dose, PREREG["scan"], PREREG["scan_upper"]))
    print("  定稿确认 45 run：%.4f（降级 27 run：%.4f）｜孪生臂 4 run：%.4f｜σ 细化 12 run：%.4f GPU-h"
          % (fc_mid, fc_dg_mid, twin_mid, refine_mid))
    print("  保守口径（不做 bias 修正、全臂 bias=1.00 单价）= %.4f GPU-h" % total_conservative)
    print("  合计 mid = **%.4f GPU-h**（预注册 %.3f）｜upper = %.4f（预注册 %.3f）｜降级方案 %.4f"
          "｜包络 %.1f ⇒ %s"
          % (total_mid, PREREG["total_mid"], total_upper, PREREG["total_upper"], total_dg,
             PREREG["envelope"],
             "在包络内" if total_upper <= PREREG["envelope"] else "**upper 超包络 ⇒ 须启用降级方案**"))
    print("产物 -> %s\n     -> %s" % (jp, cp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
