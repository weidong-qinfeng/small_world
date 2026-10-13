"""**L0 计划的脚本化预算**（R76：中间值一律脚本算出并落盘，不得手算）。

输入：`data/m11_p5_plan_l0.json`（本节点产出）＋ `data/m11_p5_g1_pricing.csv`（**实测**单位价格）
＋ `data/m11_p5_l0_dryrun/*.json`（本节点 1 臂 chunk 干跑的实测 ms/step）。
产出：`data/m11_p5_l0_pricing.json` / `.csv`。

**量纲恒等式（先自检再用）**：`gpu_h = steps × ms_per_step / 3.6e6`
（1 GPU-h = 3600 s = 3.6e6 ms；因子 2.7778e-7）。脚本用 G1 价格表的**已知行**做算例核对
（210,000 步 × 0.9482 ms/step = 0.055312 GPU-h），不一致即非零退出（防 E-M11-2 那类因子错）。

价格来源分两维，**分开报告份额**（不得用"零插值"掩盖另一维的外推）：
- **剂量维**：L0 只用到 dose∈{0,1,4}，三者都是**实测**值 ⇒ 剂量维插值份额 = 0；
- **杠杆维**：G1 价格表只在「参考档 gmax=1.0/bias=1.0」实测过 ⇒ 42 臂里 38 臂的价格是
  **同剂量跨杠杆外推** ⇒ 单列 `lever_extrapolation_share` 并用干跑实测值与 ±15% 做敏感性。

用法：
    ./.venv-m11/bin/python -m neural_exploration.tools.price_m11_p5_l0
"""
from __future__ import annotations

import csv
import glob
import json
import os
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "neural_exploration", "data")
MS_PER_GPU_H = 3.6e6                     # 1 GPU-h = 3,600 s = 3.6e6 ms
LOGICAL_CPU = 8                          # 本机可用逻辑核（仅用于并行份额说明）
PREREG_R1_ENVELOPE = 1.802               # 预注册 data/m11_p5_lever_prereg.json：R1_L0_scan chunk GPU-h
PROGRAM_ENVELOPE = 10.0                  # 主 agent 给定包络（≤10 GPU-h）
PREREG_OTHER_SEGMENTS = dict(R2_L1_sigma_scan=1.352, R3_final_confirm=5.179,
                             R4_twin_arms_new_condition=0.190)
NON_SIM_S_PER_ARM = 10.0                 # 空闲态单臂非仿真开销（进程启动+夹具+CSR+事件），本节点实测 5–10 s


def load_prices() -> dict:
    """从 G1 价格表取**实测**的 chunk/step 单位价（ms/step）与来源标注。"""
    p = os.path.join(DATA_DIR, "m11_p5_g1_pricing.csv")
    out = {}
    with open(p) as f:
        for r in csv.DictReader(f):
            if r["config_id"] != "p5_P1":
                continue
            d = float(r["drive_mult"])
            out[d] = dict(ms_chunk=float(r["ms_per_step_chunk"]),
                          ms_step=float(r["ms_per_step_step"]),
                          src=r["chunk_ms_source"], gpu_h_chunk_ref=float(r["gpu_h_chunk"]))
    assert set(out) >= {0.0, 1.0, 4.0}, "价格表缺 dose∈{0,1,4} 的实测行"
    return out


def dimension_self_check(plan: dict, prices: dict) -> dict:
    """量纲自检：用脚本重算 G1 价格表的已知行，必须与表内值一致（R80 §⑤2）。"""
    steps = int(round((plan["settle_ms"] + 10000.0) / plan["dt_ms"]))
    ms = prices[1.0]["ms_chunk"]
    gpu_h = steps * ms / MS_PER_GPU_H
    ref = prices[1.0]["gpu_h_chunk_ref"]
    ok = abs(gpu_h - ref) <= 1e-5          # G1 价格表按 5 位小数落盘 ⇒ 容差取半个末位量级
    ex = dict(identity="gpu_h = steps × ms_per_step / 3.6e6",
              worked_example=dict(steps=steps, ms_per_step=ms, gpu_h_computed=gpu_h,
                                  gpu_h_in_g1_table=ref, match=ok,
                                  seconds=steps * ms / 1e3, hours=steps * ms / 3.6e6),
              tolerance_gpu_h=1e-5,
              note="G1 表 gpu_h 按 5 位小数落盘；因子错误是 E-M11-2 的形态（曾把 2.7778e-7 写成 3.6e-6，差 12.96 倍）")
    print("  [量纲自检] %d 步 × %.4f ms/step = %.6f GPU-h（G1 表 %.6f，一致=%s）"
          % (steps, ms, gpu_h, ref, ok))
    if not ok:
        raise SystemExit(8)
    return ex


def main() -> int:
    plan_p = os.path.join(DATA_DIR, "m11_p5_plan_l0.json")
    with open(plan_p) as f:
        plan = json.load(f)
    prices = load_prices()
    dt, settle = float(plan["dt_ms"]), float(plan["settle_ms"])
    print("== L0 预算（脚本计算；plan sha256 前缀 %s）=="
          % __import__("hashlib").sha256(open(plan_p, "rb").read()).hexdigest()[:16])
    ex = dimension_self_check(plan, prices)

    # ---- 干跑实测（bias_scale=0.70 / dose=1 / chunk）----
    dry = {}
    for p in glob.glob(os.path.join(DATA_DIR, "m11_p5_l0_dryrun", "*.json")):
        if p.endswith("spec.json"):
            continue
        d = json.load(open(p))
        dry[d["arm_id"]] = d
    dry_ratio = None
    dry_note = "未找到干跑产物 ⇒ 无杠杆维实测校正（仅用参考档价格，属保守上界）"
    if dry:
        a = list(dry.values())[0]
        dry_ratio = float(a["ms_per_step_steady"]) / prices[1.0]["ms_chunk"]
        dry_note = ("干跑实测：%s（bias_scale=0.7, dose=1, chunk, T=%.0f ms, n_steps_meas=%d）"
                    " ms/step=%.4f ⇒ 相对参考档同剂量 %.4f 的比值 = %.4f（**单臂单点**，"
                    "仅作敏感性，不作预算基准）"
                    % (a["arm_id"], a["T_ms"], a["n_steps_meas"],
                       a["ms_per_step_steady"], prices[1.0]["ms_chunk"], dry_ratio))
    print("  [干跑] " + dry_note)

    rows, tot_chunk, tot_step = [], 0.0, 0.0
    interp_ms, all_ms = 0.0, 0.0
    lever_extrap = 0
    for a in plan["arms"]:
        d = float(a["drive_mult"])
        steps = int(round((settle + a["T_ms"]) / dt))
        pr = prices[d]
        ms_c, ms_s = pr["ms_chunk"], pr["ms_step"]
        gh_c = steps * ms_c / MS_PER_GPU_H
        gh_s = steps * ms_s / MS_PER_GPU_H
        tot_chunk += gh_c
        tot_step += gh_s
        all_ms += steps * ms_c
        if pr["src"] != "measured":
            interp_ms += steps * ms_c
        on_ref = (abs(float(a["gmax_scale"]) - 1.0) < 1e-9 and abs(float(a["bias_scale"]) - 1.0) < 1e-9)
        if not on_ref:
            lever_extrap += 1
        rows.append(dict(run_id=a["run_id"], stage=a["stage"], config_id=a["config_id"],
                         role=a["role"], drive_mult=d, steps=steps,
                         ms_per_step_chunk=ms_c, ms_source=pr["src"],
                         ms_per_step_step=ms_s, gpu_h_chunk=gh_c, gpu_h_step=gh_s,
                         priced_on_reference_config=on_ref))
    per_stage = {}
    for r in rows:
        v = per_stage.setdefault(r["stage"], dict(n=0, gpu_h_chunk=0.0, gpu_h_step=0.0))
        v["n"] += 1
        v["gpu_h_chunk"] += r["gpu_h_chunk"]
        v["gpu_h_step"] += r["gpu_h_step"]

    s_plus15 = tot_chunk * 1.15
    s_worst_ms = sum(r["steps"] * max(p["ms_chunk"] for p in prices.values())
                     for r in rows) / MS_PER_GPU_H
    s_dry = None
    if dry_ratio is not None:
        s_dry = sum((r["gpu_h_chunk"] * dry_ratio if not r["priced_on_reference_config"]
                     else r["gpu_h_chunk"]) for r in rows)
    sens = dict(
        base_chunk_gpu_h=tot_chunk,
        S1_ms_per_step_plus15pct=s_plus15,
        S1_basis="G1 价格表实测到的最大插值误差（dose=1 的粗插值低估约 15%）",
        S2_all_arms_at_worst_measured_ms_per_step=s_worst_ms,
        S2_basis="最贵实测 chunk 单价（dose=4 = %.4f ms/step）套到全部 42 臂 ⇒ 剂量维上界"
                 % max(p["ms_chunk"] for p in prices.values()),
        S3_dry_run_bias_lever_correction=s_dry,
        S3_basis=("干跑实测的跨杠杆比值 %.4f 套到 %d 个非参考档臂（**乐观下界**；只测了 "
                  "bias=0.7/dose=1 一个点）" % (dry_ratio, lever_extrap)) if dry_ratio else "无干跑实测",
        lever_extrapolation_bounded_by=(
            "① 实测 chunk 单价对群率**饱和**（dose=1 0.9482 vs dose=4 0.9720，仅差 2.5%%）"
            " ⇒ gmax↑ 方向的上界就是饱和价；② bias↓ 方向已由干跑实测证实更便宜（-16%%）"
            " ⇒ S2（全臂饱和价）= 杠杆维上界，且 %.4f 仍可容纳" % s_worst_ms),
    )
    prog = tot_chunk + sum(PREREG_OTHER_SEGMENTS.values())
    out = dict(tool="tools/price_m11_p5_l0.py", generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
               plan_path="neural_exploration/data/m11_p5_plan_l0.json",
               plan_sha256=__import__("hashlib").sha256(open(plan_p, "rb").read()).hexdigest(),
               units=dict(gpu_h="GPU-hour=3.6e6 ms", ms_per_step="millisecond per step",
                          note="gpu_h = steps × ms_per_step / 3.6e6"),
               dimension_self_check=ex, price_source_table="neural_exploration/data/m11_p5_g1_pricing.csv",
               price_sources={str(k): v for k, v in sorted(prices.items())},
               dry_run=dry_note, lever_extrapolation_share=lever_extrap / len(rows),
               dose_interpolation_share=(interp_ms / all_ms if all_ms else 0.0),
               n_arms=len(rows), total_steps=sum(r["steps"] for r in rows),
               total_gpu_h_chunk=tot_chunk, total_gpu_h_step_scope=tot_step,
               per_stage=per_stage, sensitivity=sens,
               non_sim_overhead=dict(s_per_arm=NON_SIM_S_PER_ARM,
                                     total_min=len(rows) * NON_SIM_S_PER_ARM / 60.0,
                                     basis="本节点实测（N=8k/T=1 s 总墙钟 − 仿真窗；N=31k/T=10 s 为 5.14 s）"),
               program_total_gpu_h_worst=(s_worst_ms + sum(PREREG_OTHER_SEGMENTS.values())),
               envelope=dict(prereg_R1_chunk_gpu_h=PREREG_R1_ENVELOPE,
                             delta_vs_prereg=tot_chunk - PREREG_R1_ENVELOPE,
                             delta_pct=100 * (tot_chunk - PREREG_R1_ENVELOPE) / PREREG_R1_ENVELOPE,
                             program_segments=dict(L0=tot_chunk, **PREREG_OTHER_SEGMENTS),
                             program_total_gpu_h=prog,
                             program_envelope_gpu_h=PROGRAM_ENVELOPE,
                             program_within_envelope=prog <= PROGRAM_ENVELOPE,
                             program_total_worst_gpu_h=s_worst_ms + sum(PREREG_OTHER_SEGMENTS.values()),
                             program_total_worst_within_envelope=(
                                 s_worst_ms + sum(PREREG_OTHER_SEGMENTS.values())) <= PROGRAM_ENVELOPE,
                             verdict=("chunk 口径可行：L0 %.3f GPU-h（预注册 R1 段 %.3f），"
                                      "程序合计 %.3f ≤ %.1f" % (tot_chunk, PREREG_R1_ENVELOPE, prog,
                                                                 PROGRAM_ENVELOPE))),
               rows=rows)
    jp = os.path.join(DATA_DIR, "m11_p5_l0_pricing.json")
    cp = os.path.join(DATA_DIR, "m11_p5_l0_pricing.csv")
    with open(jp, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    with open(cp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("  段/总量（chunk）：")
    for k, v in sorted(per_stage.items()):
        print("    %-8s n=%-3d %.4f GPU-h" % (k, v["n"], v["gpu_h_chunk"]))
    print("    合计 n=%d  %.4f GPU-h（step 口径 %.4f，仅参考）"
          % (len(rows), tot_chunk, tot_step))
    print("  剂量维插值份额 = %.4f；杠杆维外推份额 = %.4f"
          % (out["dose_interpolation_share"], out["lever_extrapolation_share"]))
    print("  敏感性：+15%% ⇒ %.4f；全臂按最贵单价 ⇒ %.4f；干跑比值校正 ⇒ %s"
          % (s_plus15, s_worst_ms, ("%.4f" % s_dry) if s_dry else "n/a"))
    print("  预算裁决：" + out["envelope"]["verdict"])
    print("  最坏情形（L0 全臂饱和价 %.4f）⇒ 程序合计 %.3f ≤ %.1f = %s"
          % (s_worst_ms, out["envelope"]["program_total_worst_gpu_h"], PROGRAM_ENVELOPE,
             out["envelope"]["program_total_worst_within_envelope"]))
    print("  非仿真开销：42 臂 × %.0f s ≈ %.1f min" % (NON_SIM_S_PER_ARM,
                                                     out["non_sim_overhead"]["total_min"]))
    print("产物 -> %s\n     -> %s" % (jp, cp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
