"""M8 铁律 C 三组缩放扫描：规模 {300,1000,3016} × 保真度 {点,双隔室,HH} ×
可塑性 {无,STP,STDP,+稳态} → 行为指标 + 性能轴 + G0/G1 决策（G0 门）。

对应《生物仿真M8实施清单》§4 步骤 2（P2 验证对象，第一关键决策步）：
  - **规模轴** {300, 1000, 3016}（幼虫连接组子集，类平衡+功能模块规则，
    src/larva_circuit.py scale_names）+ 302 锚行（C. elegans 方法论对照，
    非幼虫子集——报告注明，`--run-anchor` 用 M5 WormCircuit 跑）；
  - **保真度轴** {point, two_comp, hh}（dt 并入档位：0.1/0.05/0.01ms；
    HH 仅 ≤300 档短协议 T≤5s，M5 同哲学；two_comp ≤1000 可选；
    hh/1000-two_comp 默认记 skipped（预算纪律，M5 multicomp 先例）；
  - **可塑性轴** {none, stp, stdp, stdp_homeo}（学习探针 LI 出现/消失阈值，
    机制级判据——M6 L16 网络级 CI 读出不可见的测量限制语义）；
  - **行为指标轴**：嗅觉 CI（气味梯度短协议，符号）、逃避/蜷缩方向（MD
    伤害性刺激短协议）、自发状态比例（无刺激短协议）、学习指数 LI；
  - **性能轴**：单试次墙钟（每格点实测）+ 冷编译墙钟；
  - **G1 双状态门**（§0.5/§1 D6）：3,016 定稿配置（全杠杆）静息静默比例
    落带 [50,90]% + 自发 bout 活动 ≥10% 双状态；三杠杆消融 sanity
    （删杠杆 → 双状态破坏断言）；不通过 → 反证记录 + 三态裁决；
  - **G0 决策**（§4.4）：定稿规模/保真度/dt/协议 T/预算 →
    data/m8_larva_params.csv（role=model/protocol/g0）；
  - 产出：data/m8_scaling.csv + reports/neuro/m8_scaling_curves.png。

确定性：p=1/n=1；无噪声；试次方差来自伪随机起点（seed_base 派生）；
同参数重跑逐位一致。总格点 ≤36 预算（本网格 13 行）。

运行：
  .venv-neuro/bin/python -m neural_exploration.tools.scan_m8_scaling
    --wait            # 默认：轮询等待 B1a m8_larva_connectome.csv（G2 数据门）
    --smoke           # 冒烟：内存合成占位连接组（仅验证机制，不出真实决策）
    --rows 300p,1000p # 格点子集（<scale><f>，f∈{p,t,h}；可塑性轴按 --plasticity 全集）
    --run-hh --run-anchor --run-1000-two-comp   # 预算敏感格点强制
  长任务：每格点结果即时落盘 CSV（断点续跑），完成后 touch 完成标记（M4 L25）。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.src.larva_circuit import (  # noqa: E402
    FIDELITY_DT,
    LI_APPEAR_THRESHOLD,
    LI_DISAPPEAR_THRESHOLD,
    PLASTICITY_AXIS,
    PROTOCOL_WINDOW_MS,
    SCALE_AXIS,
    LarvaCircuit,
    build_placeholder_spec,
    g1_dual_state_check,
    wait_for_csv,
)

DATA_DIR = os.path.join(ROOT, "neural_exploration", "data")
REPORTS_DIR = os.path.join(ROOT, "neural_exploration", "reports", "neuro")
SCALING_CSV = os.path.join(DATA_DIR, "m8_scaling.csv")
SCALING_PNG = os.path.join(REPORTS_DIR, "m8_scaling_curves.png")
G1_RESULT_JSON = os.path.join(DATA_DIR, "m8_g1_result.json")
LARVA_PARAMS_CSV = os.path.join(DATA_DIR, "m8_larva_params.csv")
DONE_MARKER = os.path.join(DATA_DIR, ".m8_scaling_DONE")

#: 降阶正确性锚（§3.4；M4/M5 机制模块语义——幼虫无独立参考解，锚 =
#: CI 符号一致 / 逃避方向 back / 双状态结构，Δ 判据带预注册）
CI_DELTA_BAND = 0.15
D_PEAK_THRESHOLD = 0.3
#: G0 协议 T 草案（M5 P4 定稿值；B1c 幼虫行为参考交付后按「参考模型 N=20
#: 通过率 ≥80% 最短 T」公式复核——M5 §3.5）
PROTOCOL_T_DRAFT_MS = 15000.0
#: 单试次墙钟上限（协议 T=15s ≤ 20min，§0.7 #1c 草案；超出 → 最短协议 +
#: 测量限制 + 三态裁决）
TRIAL_WALL_BUDGET_S = 1200.0

#: 网格（scale, fidelity, plasticity, proto overrides, note）——总 13 行 ≤ 36 预算。
#: two_comp=1000 与 hh=300 默认记 skipped（预算纪律；--run-hh/--run-1000-two-comp 强制）。
GRID = [
    (300, "point", "none", {}, "300 点神经元基线（grouped）"),
    (300, "point", "stp", {}, "300 点 + STP（全化学突触）"),
    (300, "point", "stdp", {}, "300 点 + STDP（KC→MBON 子集）"),
    (300, "point", "stdp_homeo", {}, "300 点 + STDP+稳态（防饱和）"),
    (300, "two_comp", "none", {"t_scale": 0.5}, "300 双隔室（grouped 2N；短协议）"),
    (300, "hh", "none", {}, "300 HH（M1 多隔室局部子图；预算敏感，默认跳过）"),
    (1000, "point", "none", {"t_scale": 0.8}, "1000 点神经元基线"),
    (1000, "point", "stp", {"t_scale": 0.8}, "1000 点 + STP"),
    (1000, "point", "stdp", {"t_scale": 0.8}, "1000 点 + STDP"),
    (1000, "two_comp", "none", {}, "1000 双隔室（预算敏感，默认跳过）"),
    (3016, "point", "none", {"t_scale": 0.4}, "3,016 全脑点神经元（grouped 稀疏 stim）"),
    (3016, "point", "stp", {"t_scale": 0.4}, "3,016 全脑 + STP"),
    (302, "point", "none", {"anchor": True}, "302 锚行（C. elegans 方法论对照，非幼虫子集）"),
]

CSV_HEADER = [
    "scale", "fidelity", "plasticity", "source", "dt_ms", "method", "status",
    "n_neurons", "n_chem", "n_gap", "n_muscle",
    "build_wall_s", "probe_T_ms", "probe_wall_s",
    "ci_mean", "ci_sem", "ci_n", "ci_direction",
    "escape_direction", "escape_d_peak", "escape_curl_peak",
    "resting_median_hz", "resting_silent_frac", "resting_max_hz",
    "spont_fwd", "spont_rev", "spont_turn", "spont_pause",
    "li", "li_mode", "dw_stdp",
    "chem_wall_s_mean", "total_wall_s", "notes",
]


def _row_defaults(scale, fidelity, plasticity, source, status="pending"):
    dt, method = FIDELITY_DT[fidelity]
    return dict(scale=scale, fidelity=fidelity, plasticity=plasticity,
                source=source, dt_ms=dt, method=method, status=status,
                n_neurons=0, n_chem=0, n_gap=0, n_muscle=0,
                build_wall_s=float("nan"), probe_T_ms=1000.0,
                probe_wall_s=float("nan"), ci_mean=float("nan"),
                ci_sem=float("nan"), ci_n=0, ci_direction="",
                escape_direction="", escape_d_peak=float("nan"),
                escape_curl_peak=float("nan"), resting_median_hz=float("nan"),
                resting_silent_frac=float("nan"), resting_max_hz=float("nan"),
                spont_fwd=float("nan"), spont_rev=float("nan"),
                spont_turn=float("nan"), spont_pause=float("nan"),
                li=float("nan"), li_mode="", dw_stdp=float("nan"),
                chem_wall_s_mean=float("nan"), total_wall_s=float("nan"),
                notes="")


def _load_existing() -> dict:
    out = {}
    if not os.path.exists(SCALING_CSV):
        return out
    with open(SCALING_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(row for row in f
                                if not row.strip().startswith("#")):
            out[(int(r["scale"]), r["fidelity"], r["plasticity"])] = r
    return out


def _save_rows(rows):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(SCALING_CSV, "w", newline="", encoding="utf-8") as f:
        f.write("# M8 铁律 C 三组缩放扫描结果（tools/scan_m8_scaling.py 生成，G0 门）\n"
                "# 列语义：scale=规模轴（300/1000/3016 幼虫子集；302=C. elegans 锚行）；\n"
                "#   fidelity=保真度轴（point/two_comp/hh，dt 并入档位）；\n"
                "#   plasticity=可塑性轴（none/stp/stdp/stdp_homeo）；\n"
                "#   ci_direction=趋化 CI 符号（正趋化=+）；escape_*=痛觉逃避方向（back/not_back）；\n"
                "#   resting_silent_frac=静默比例（G1 带 [50,90]%）；spont_*=自发状态比例；\n"
                "#   li/li_mode=学习探针 LI（weight=KC→MBON 权重档，mbon_rate=发放率档）；\n"
                "#   probe= T=1s 探针墙钟；chem_wall_s_mean=趋化短协议单试次墙钟。\n")
        w = csv.DictWriter(f, fieldnames=CSV_HEADER, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_HEADER})


def _spec_for(args) -> object:
    """连接组规格：真实 CSV（wait_for_csv 轮询）或占位（--smoke）。"""
    if args.smoke:
        spec = build_placeholder_spec(300)
        print(f"[smoke] 占位连接组 {spec.n_neurons} 神经元（仅机制冒烟，"
              f"不出真实决策）", flush=True)
        return spec
    path = wait_for_csv(None, timeout_s=args.connectome_timeout)
    print(f"[data] m8_larva_connectome.csv 就绪：{path}", flush=True)
    return None  # LarvaCircuit 自行 load_connectome


def run_grid_point(scale, fidelity, plasticity, proto, args, spec) -> dict:
    row = _row_defaults(scale, fidelity, plasticity,
                        "placeholder_smoke" if args.smoke else "connectome")
    t0 = time.perf_counter()
    t_scale = proto.get("t_scale", 1.0)
    try:
        kw = dict(scale=scale, fidelity=fidelity, plasticity=plasticity,
                  allow_placeholder=args.smoke, seed=0,
                  connectome_poll_s=0.0,
                  nt_fallback=args.nt_fallback,
                  provisional_muscles=args.provisional_muscles,
                  annotations_path=args.annotations,
                  gmax_scale=args.gmax_scale)
        if args.smoke:
            kw["spec_override"] = spec
        if proto.get("anchor"):
            # 302 锚行：C. elegans 全虫（M5 冻结路径，方法论对照）
            from neural_exploration.src.worm_circuit import make_worm_circuit
            circ = make_worm_circuit(scale=302, fidelity="point")
            probe_res, probe_meta = circ.run_chemotaxis_trials(
                n_trials=1, t_total_ms=1000.0, seed_base=9000)
            row["probe_wall_s"] = probe_meta["wall_s"][0]
            row["build_wall_s"] = circ._build_wall_s
            row["n_neurons"] = circ.sub.n_neurons
            row["n_chem"] = circ.sub.n_chem
            row["n_gap"] = circ.sub.n_gap
            row["n_muscle"] = len(circ.sub.muscles)
            res, meta = circ.run_chemotaxis_trials(
                n_trials=1, t_total_ms=5000.0, seed_base=0)
            row["ci_mean"] = float(res[0]["ci"])
            row["ci_n"] = 1
            row["ci_direction"] = "+" if row["ci_mean"] > 0 else "-"
            row["chem_wall_s_mean"] = float(meta["wall_s"][0])
            rest = circ.run_resting(t_total_ms=2000.0)
            row["resting_median_hz"] = round(rest["median_hz"], 3)
            row["resting_silent_frac"] = round(rest["silent_frac"], 3)
            row["resting_max_hz"] = round(rest["max_hz"], 3)
            sp = circ.run_spontaneous(t_total_ms=5000.0)
            for k in ("fwd", "rev", "turn", "pause"):
                row[f"spont_{k}"] = round(sp["frac"].get(k, 0.0), 3)
            row["status"] = "done"
            row["notes"] = ("C. elegans 302 方法论锚行（m5 冻结路径；"
                            "非幼虫子集——M8 D1 注明）")
            row["total_wall_s"] = round(time.perf_counter() - t0, 1)
            return row

        circ = LarvaCircuit(**kw)
        # 探针（T=1s 单试次墙钟）
        pr, pmeta = circ.run_chemotaxis_trials(n_trials=1, t_total_ms=1000.0,
                                               seed_base=9000)
        row["probe_wall_s"] = round(pr[0]["wall_s"], 3)
        row["build_wall_s"] = round(circ._build_wall_s, 3)
        row["n_neurons"] = circ.sub.n_neurons
        row["n_chem"] = circ.sub.n_chem
        row["n_gap"] = circ.sub.n_gap
        row["n_muscle"] = len(circ.sub.muscles)
        row["probe_T_ms"] = 1000.0
        # 数据缺口/临时回退标记（如实记录，不静默）
        if getattr(circ, "roster_note", ""):
            row["notes"] += circ.roster_note + "; "
        if getattr(circ, "nt_fallback_active", False):
            row["notes"] += (f"⚠ PROVISIONAL_NT：未标注化学边临时类级回退 "
                             f"n={circ.nt_fallback_n_provisional} "
                             f"（B1a 递质标注不完整，非权威）; ")
        if getattr(circ, "muscle_provisional", False):
            row["notes"] += ("⚠ PROVISIONAL_MUSCLE：运动池→通道临时映射 "
                             "（幼虫脑连接组无肌肉行，P3 节点定稿真实映射）; ")

        # 趋化短协议（T 按规模缩放；CI 符号）
        chem_t = float(proto.get("chem_t_ms", 5000.0 * t_scale))
        res, meta = circ.run_chemotaxis_trials(n_trials=1,
                                               t_total_ms=chem_t, seed_base=0)
        row["ci_mean"] = round(float(res[0]["ci"]), 4)
        row["ci_sem"] = float("nan")
        row["ci_n"] = 1
        row["ci_direction"] = "+" if row["ci_mean"] > 0 else "-"
        row["chem_wall_s_mean"] = round(float(meta["wall_s"][0]), 3)

        # 痛觉逃避（MD 伤害性刺激短协议）
        esc = circ.run_escape(t_total_ms=1000.0 * t_scale)
        row["escape_direction"] = esc["direction"]
        row["escape_d_peak"] = round(esc["d_peak"], 4)
        row["escape_curl_peak"] = round(esc["curl_peak"], 4)

        # 静息（settle 500ms；G1 输入）
        rest = circ.run_resting(t_total_ms=2000.0 * t_scale, settle_ms=500.0)
        row["resting_median_hz"] = round(rest["median_hz"], 3)
        row["resting_silent_frac"] = round(rest["silent_frac"], 4)
        row["resting_max_hz"] = round(rest["max_hz"], 3)
        if rest["has_nan"]:
            row["notes"] += "⚠ 静息 NaN/发散; "

        # 自发状态比例（G1 输入）
        sp = circ.run_spontaneous(t_total_ms=5000.0 * t_scale)
        for k in ("fwd", "rev", "turn", "pause"):
            row[f"spont_{k}"] = round(sp["frac"].get(k, 0.0), 3)

        # 学习探针（LI 出现/消失阈值）
        lp = circ.run_learning_probe(t_test_ms=2000.0 * t_scale,
                                     t_train_ms=2000.0 * t_scale)
        row["li"] = round(lp["li"], 4)
        row["li_mode"] = lp["li_mode"]
        if lp["dw"] is not None and np.isfinite(lp["dw"]):
            row["dw_stdp"] = round(lp["dw"], 5)

        row["status"] = "done"
        row["total_wall_s"] = round(time.perf_counter() - t0, 1)
        if args.smoke:
            row["source"] = "placeholder_smoke"
            row["notes"] += "冒烟（占位数据，不出真实决策）; "
    except Exception as exc:  # noqa: BLE001 —— 格点失败如实记录，不静默
        row["status"] = "failed"
        row["notes"] += f"FAIL: {type(exc).__name__}: {str(exc)[:300]}"
    return row


# --------------------------------------------------------------------- #
# G1 双状态门（三杠杆消融 sanity）
# --------------------------------------------------------------------- #
def run_g1_gate(args, spec) -> dict:
    """G1：3,016（或冒烟：最大可用规模）全杠杆双状态 + 三杠杆消融 sanity。

    Returns dict(verdict/lever_results/ablation/decision)。
    """
    scale = int(getattr(args, "g1_scale", 3016) or 3016)
    if args.smoke:
        scale = spec.n_neurons  # 冒烟：占位规模（机制验证）
    out: dict = dict(scale=scale, plasticity="none",
                     verdict="PENDING_DATA", detail="")

    def _dual(lever_cmd, lever_motor, lever_hetero):
        kw = dict(scale=scale, fidelity="point", plasticity="none",
                  allow_placeholder=args.smoke, seed=0,
                  connectome_poll_s=0.0,
                  lever_cmd_desync=lever_cmd,
                  lever_motor_drive=lever_motor,
                  lever_hetero=lever_hetero,
                  nt_fallback=args.nt_fallback,
                  provisional_muscles=args.provisional_muscles,
                  annotations_path=args.annotations,
                  gmax_scale=args.gmax_scale)
        if args.smoke:
            kw["spec_override"] = spec
        circ = LarvaCircuit(**kw)
        rest = circ.run_resting(t_total_ms=2000.0, settle_ms=500.0)
        sp = circ.run_spontaneous(t_total_ms=5000.0)
        return g1_dual_state_check(rest, sp)

    base = _dual(True, True, True)
    out["base"] = base
    out["levers"] = dict(cmd_desync=True, motor_drive=True, hetero=True)
    # 三杠杆消融 sanity（删杠杆 → 双状态破坏的断言/记录）
    ablations = {
        "no_cmd_desync": _dual(False, True, True),
        "no_motor_drive": _dual(True, False, True),
        "no_hetero": _dual(True, True, False),
    }
    out["ablation"] = {}
    for name, res in ablations.items():
        out["ablation"][name] = dict(
            dual_state=res["dual_state"], silent_frac=res["silent_frac"],
            bout_activity=res["bout_activity"],
            detail=res["detail"])
    broken = {k: v for k, v in ablations.items() if not v["dual_state"]}
    if base["dual_state"]:
        out["verdict"] = "PASS" if broken else "PASS_WEAK_ABLATION"
        if not broken:
            out["detail"] = ("双状态成立但三杠杆消融均未破坏（消融 sanity 弱——"
                             "如实记录，不伪造归因）")
        else:
            out["detail"] = (f"双状态成立；消融破坏：{list(broken.keys())}——"
                             "三杠杆生效（G1 正面设计验证）")
    else:
        out["verdict"] = "FAIL"
        out["detail"] = (f"双状态不成立：{base['detail']}；消融结果 "
                         f"{list(ablations.items())}——按三杠杆顺序排查 "
                         "（命令层去同步 → 运动层分离驱动 → 异质权重/传导），"
                         "仍不通过 → 反证记录 + 三态裁决（不烧协议预算，§0.5 G1）")
    out["li_thresholds"] = dict(li_appear=LI_APPEAR_THRESHOLD,
                                li_disappear=LI_DISAPPEAR_THRESHOLD)
    if args.smoke:
        out["verdict"] = f"SMOKE({out['verdict']})"
        out["detail"] = "冒烟机制验证（占位数据；真实判定待 B1a CSV）——" + out["detail"]
    return out


# --------------------------------------------------------------------- #
# G0 决策（§4.4：定稿规模/保真度/dt/协议 T/预算）
# --------------------------------------------------------------------- #
def g0_decision(rows, g1, smoke: bool = False) -> dict:
    """G0 决策规则（预注册，确定性；数据不足 → PENDING_DATA 不静默）。

    - 规模：3016 档 CI 方向与 300/1000 一致 且 单试次墙钟 ≤ 预算上限
      （协议 T=15s ≤ 20min）→ 3016；否则 1000（记录三态裁决请求）；
    - 保真度：300 档 point vs two_comp ΔCI ≤ 0.15 → point（M5 §3.3 收敛规则）；
    - dt/method：FIDELITY_DT[保真度] 定稿后不变（M4 L16）；
    - 协议 T：草案 15s（幼虫行为参考交付后按 M5 §3.5 公式复核）；
    - 预算：缩放扫描 ≤120 CPU-h（§0.7 #14 分解）。
    """
    done = [r for r in rows if r["status"] == "done" and r["plasticity"] == "none"
            and r["fidelity"] == "point"]
    pts = {int(r["scale"]): r for r in done}
    d = dict(decision="PENDING_DATA", scale=3016, fidelity="point",
             dt_ms=FIDELITY_DT["point"][0], method=FIDELITY_DT["point"][1],
             protocol_t_ms=PROTOCOL_T_DRAFT_MS, budget_cpu_h=120.0,
             plasticity="none", notes="")
    if 3016 in pts and 300 in pts and pts[3016]["ci_direction"] \
            and pts[3016]["ci_direction"] == pts[300]["ci_direction"]:
        d["scale"] = 3016
        d["decision"] = "PASS"
    elif 1000 in pts and 300 in pts and pts[1000]["ci_direction"] \
            and pts[1000]["ci_direction"] == pts[300]["ci_direction"]:
        d["scale"] = 1000
        d["decision"] = "PARTIAL"
        d["notes"] = "3016 档不可行/方向不一致 → 回退 1000（请求规划节点裁决）"
    else:
        d["decision"] = "FAIL"
        d["notes"] = "规模扫描方向不一致/数据不足 → 三态裁决（降保真度/缩行为集/反证）"
    if g1.get("verdict", "").startswith("FAIL"):
        d["decision"] = "G1_FAIL"
        d["notes"] += "；G1 双状态门不通过 → 按三杠杆顺序排查后三态裁决"
    two = [r for r in rows if r["status"] == "done" and r["fidelity"] == "two_comp"
           and r["scale"] == 300]
    if two and pts.get(300) and two[0].get("ci_mean") != "nan" \
            and pts[300].get("ci_mean") != "nan":
        try:
            dc = abs(float(two[0]["ci_mean"]) - float(pts[300]["ci_mean"]))
            if dc <= CI_DELTA_BAND:
                d["fidelity"] = "point"
                d["notes"] += f"；保真度收敛 ΔCI={dc:.3f}≤0.15 → point"
            else:
                d["fidelity"] = "two_comp"
                d["notes"] += f"；保真度不收敛 ΔCI={dc:.3f}>0.15 → two_comp（请求裁决）"
        except (TypeError, ValueError):
            pass
    if smoke:
        d["decision"] = f"SMOKE({d['decision']})"
        d["notes"] = "冒烟机制验证（占位数据；真实决策待 B1a CSV）——" + d["notes"]
    return d


def write_larva_params(g0: dict, g1: dict, args):
    """G0/G1 决策 → data/m8_larva_params.csv（role=model/protocol/g0 行）。"""
    os.makedirs(DATA_DIR, exist_ok=True)
    lines = [
        "# M8 幼虫参数与 G0/G1 决策定稿（唯一定稿源，可修改复现）——",
        "# tools/scan_m8_scaling.py 生成。列语义同 m5_worm_params.csv",
        "# （value 在 fields[9]，位置解析，M5-B1d L23 语义）。",
        "# G0 依据：data/m8_scaling.csv + reports/neuro/m8_scaling_curves.png。",
        "role,neuron_class,synapse_from,synapse_to,synapse_type,g_max_ns,delay_ms,tonic_uA_cm2,value,note",
        "# ---- 降阶模型行（G0 定稿）----",
        f"model,scale_behavior,,,,,,,,{g0['scale']},G0 定稿规模（300/1000/3016；302 为 C. elegans 锚行非幼虫子集）",
        f"model,fidelity_behavior,,,,,,,,{g0['fidelity']},G0 定稿保真度（point/two_comp/hh；dt 并入档位）",
        f"model,dt_ms,,,,,,,,{g0['dt_ms']},定稿 dt（M4 L16：定稿后不变）",
        f"model,method,,,,,,,,{g0['method']},定稿方法（M5 L17 实测定稿语义）",
        "model,plasticity_default,,,,,,,,none,行为协议默认可塑性（P5 学习协议单独启用）",
        "model,gap_scale,,,,,,,,0.05,缝隙全局缩放（M5 L38 定稿先验）",
        "model,lever_cmd_desync,,,,,,,,1,夹带双稳态杠杆①命令层去同步（真实 GABA 抑制边）",
        "model,lever_motor_drive,,,,,,,,1,杠杆②运动层与命令层分离驱动（M6 L9#1）",
        "model,lever_hetero,,,,,,,,1,杠杆③异质权重/传导（类级权重+异质延迟）",
        "model,budget_cpu_hours,,,,,,,,120.0,缩放扫描子预算（§0.7 #14：缩放 ≤120 CPU-h）",
        "# ---- 协议行（G0 定稿草案；B1c 行为参考交付后按 M5 §3.5 复核）----",
        f"protocol,t_total_ms,,,,,,,,{g0['protocol_t_ms']},单试次仿真时长草案（M5 P4 定稿值 15s）",
        "protocol,resting_t_total_ms,,,,,,,,2000.0,静息短协议（扫描档）",
        "protocol,resting_settle_ms,,,,,,,,500.0,settle 窗（M5 L41#1：t=0 瞬态波排除）",
        "protocol,spont_t_total_ms,,,,,,,,5000.0,自发短协议",
        "protocol,escape_t_total_ms,,,,,,,,1000.0,痛觉逃避短协议",
        "protocol,escape_start_ms,,,,,,,,100.0,伤害性刺激开始（静息漂移纪律 ≥40ms）",
        "protocol,escape_dur_ms,,,,,,,,20.0,伤害性刺激时长",
        "protocol,escape_i0_uA_cm2,,,,,,,,60.0,伤害性刺激密度（M3 定稿量级）",
        "protocol,learning_t_test_ms,,,,,,,,2000.0,学习探针测试窗（扫描档）",
        "protocol,learning_t_train_ms,,,,,,,,2000.0,学习探针训练窗（扫描档）",
        "protocol,n_trials,,,,,,,,1,扫描格点试次数（确定性 p=1/n=1）",
        "# ---- G0/G1 决策行 ----",
        f"g0,decision,,,,,,,,{g0['decision']},G0 决策（PASS/PARTIAL/FAIL/G1_FAIL/PENDING_DATA）",
        f"g0,scale,,,,,,,,{g0['scale']},G0 定稿规模",
        f"g0,fidelity,,,,,,,,{g0['fidelity']},G0 定稿保真度",
        f"g0,protocol_t_ms,,,,,,,,{g0['protocol_t_ms']},G0 定稿协议 T",
        f"g0,budget_cpu_h,,,,,,,,{g0['budget_cpu_h']},G0 定稿预算",
        f"g0,notes,,,,,,,,{g0['notes']},G0 依据/三态裁决请求",
        f"g1,verdict,,,,,,,,{g1.get('verdict','PENDING_DATA')},G1 双状态门判定",
        f"g1,silent_frac,,,,,,,,{g1.get('base',{}).get('silent_frac','nan')},静默比例（带 [50,90]%）",
        f"g1,bout_activity,,,,,,,,{g1.get('base',{}).get('bout_activity','nan')},行为 bout 活动（下限 10%）",
        f"g1,detail,,,,,,,,{g1.get('detail','')},G1 依据/三杠杆消融",
    ]
    with open(LARVA_PARAMS_CSV, "w", newline="", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return LARVA_PARAMS_CSV


# --------------------------------------------------------------------- #
# 出图
# --------------------------------------------------------------------- #
def plot_curves(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    for _f in ("PingFang SC", "PingFang HK", "Heiti TC", "STHeiti",
               "Arial Unicode MS"):
        try:
            fm.findfont(_f, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [_f, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            break
        except Exception:
            continue

    os.makedirs(REPORTS_DIR, exist_ok=True)
    done = [r for r in rows if r["status"] == "done"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    def _ser(scale, fid, field):
        return [r for r in done if int(r["scale"]) == scale
                and r["fidelity"] == fid and r["plasticity"] == "none"]

    scales = [300, 1000, 3016]
    colors = {"point": "#1f77b4", "two_comp": "#ff7f0e", "hh": "#2ca02c"}
    plast_colors = {"none": "#555555", "stp": "#1f77b4",
                    "stdp": "#ff7f0e", "stdp_homeo": "#2ca02c"}

    # 1) CI vs 规模（point，plasticity=none）
    ax = axes[0, 0]
    xs, ys = [], []
    for s in scales:
        r = _ser(s, "point", "ci_mean")
        if r:
            xs.append(s)
            ys.append(float(r[0]["ci_mean"]))
    ax.plot(xs, ys, "o-", color=colors["point"], label="point/none")
    for s, y in zip(xs, ys):
        ax.annotate(f"{y:+.2f}", (s, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("规模（神经元数）")
    ax.set_ylabel("CI（趋化短协议）")
    ax.set_title("铁律 C① 规模：CI vs 规模（point）")
    ax.set_xticks(scales)
    ax.grid(True, alpha=0.3)

    # 2) 可塑性轴：LI vs 规模
    ax = axes[0, 1]
    for s in scales:
        for p in PLASTICITY_AXIS:
            r = next((x for x in done if int(x["scale"]) == s
                      and x["plasticity"] == p and x["fidelity"] == "point"), None)
            if r is None or r.get("li") in (None, ""):
                continue
            try:
                li = float(r["li"])
            except (TypeError, ValueError):
                continue
            ax.plot([s], [li], "o", color=plast_colors[p], label=p if s == 300 else None)
    ax.axhline(LI_APPEAR_THRESHOLD, color="gray", ls="--", lw=1,
               label=f"LI 出现阈值 {LI_APPEAR_THRESHOLD}")
    ax.axhline(-LI_APPEAR_THRESHOLD, color="gray", ls="--", lw=1)
    ax.set_xlabel("规模（神经元数）")
    ax.set_ylabel("LI（学习探针）")
    ax.set_title("铁律 C③ 可塑性：LI vs 规模（分色可塑性档）")
    ax.set_xticks(scales)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3) 性能轴：墙钟 vs 规模（对数）
    ax = axes[0, 2]
    for fid in ("point", "two_comp"):
        xs, ys = [], []
        for s in scales:
            r = _ser(s, fid, "probe_wall_s")
            if r and np.isfinite(float(r[0]["probe_wall_s"])):
                xs.append(s)
                ys.append(float(r[0]["probe_wall_s"]))
        if xs:
            ax.plot(xs, ys, "s-", color=colors[fid], label=f"{fid} 探针 T=1s")
    ax.set_yscale("log")
    ax.set_xlabel("规模（神经元数）")
    ax.set_ylabel("单试次墙钟（s，T=1s 探针，对数轴）")
    ax.set_title("性能轴：墙钟 vs 规模")
    ax.set_xticks(scales)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # 4) 静默比例 vs 规模（G1 输入）
    ax = axes[1, 0]
    xs, ys = [], []
    for s in scales:
        r = _ser(s, "point", "resting_silent_frac")
        if r and np.isfinite(float(r[0]["resting_silent_frac"])):
            xs.append(s)
            ys.append(float(r[0]["resting_silent_frac"]))
    if xs:
        ax.plot(xs, ys, "^-", color=colors["point"], label="静默比例")
    ax.axhspan(*[0.50, 0.90], color="green", alpha=0.12,
               label="G1 带 [50,90]%")
    ax.set_xlabel("规模（神经元数）")
    ax.set_ylabel("静默比例（静息，settle 500ms）")
    ax.set_title("G1 门输入：静息静默比例 vs 规模")
    ax.set_xticks(scales)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5) 自发状态比例（3016 档，堆叠）
    ax = axes[1, 1]
    r3016 = _ser(3016, "point", "spont_fwd") or _ser(1000, "point", "spont_fwd")
    if r3016:
        r = r3016[0]
        parts = [("fwd", r["spont_fwd"]), ("rev", r["spont_rev"]),
                 ("turn", r["spont_turn"]), ("pause", r["spont_pause"])]
        vals = [float(v) if v not in ("", "nan") else 0.0 for _k, v in parts]
        labs = [k for k, _v in parts]
        ax.bar(labs, vals, color=["#4c72b0", "#dd8452", "#55a868", "#c44e52"])
        ax.set_ylabel("时间比例")
        ax.set_title(f"自发状态分布（{r['scale']} 档）")
    ax.grid(True, alpha=0.3)

    # 6) 逃避方向 D_peak vs 规模
    ax = axes[1, 2]
    xs, ys = [], []
    for s in scales:
        r = _ser(s, "point", "escape_d_peak")
        if r and np.isfinite(float(r[0]["escape_d_peak"])):
            xs.append(s)
            ys.append(float(r[0]["escape_d_peak"]))
    if xs:
        ax.plot(xs, ys, "o-", color=colors["point"])
    ax.axhline(D_PEAK_THRESHOLD, color="r", ls="--", lw=1,
               label=f"D_peak 阈值 {D_PEAK_THRESHOLD}")
    ax.set_xlabel("规模（神经元数）")
    ax.set_ylabel("D_peak = max(C_back − C_fwd)")
    ax.set_title("降阶正确性：痛觉逃避方向 vs 规模")
    ax.set_xticks(scales)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("M8 铁律 C 三组缩放扫描（G0 门）：data/m8_scaling.csv + "
                 "data/m8_larva_params.csv", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(SCALING_PNG, dpi=130)
    plt.close(fig)
    return SCALING_PNG


# --------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="M8 铁律 C 三组缩放扫描（G0/G1 门）")
    ap.add_argument("--rows", default="all",
                    help="格点子集：all / 逗号分隔 <scale><f>（如 300p,1000p,3016p）")
    ap.add_argument("--plasticity", default="all",
                    help="可塑性轴：all / none,stp,stdp,stdp_homeo")
    ap.add_argument("--wait", action="store_true",
                    help="轮询等待 B1a m8_larva_connectome.csv（G2 数据门；默认）")
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟：内存合成占位连接组（仅机制验证，不出真实决策）")
    ap.add_argument("--connectome-timeout", type=float, default=3600.0,
                    help="等待连接组超时（s）")
    ap.add_argument("--run-hh", action="store_true",
                    help="强制跑 300-HH 格点（预算敏感：多隔室局部子图）")
    ap.add_argument("--run-1000-two-comp", action="store_true",
                    help="强制跑 1000-two_comp 格点（预算敏感）")
    ap.add_argument("--run-anchor", action="store_true",
                    help="跑 302 C. elegans 锚行（方法论对照；~10min 冷编译）")
    ap.add_argument("--nt-fallback", default=None, choices=["class"],
                    help="递质临时回退：class=未标注化学边按类级临时分配"
                         "（PROVISIONAL_NT，非权威；B1a 标注补齐后移除）")
    ap.add_argument("--provisional-muscles", action="store_true",
                    help="运动池→虚拟通道临时映射（PROVISIONAL_MUSCLE；"
                         "幼虫脑连接组无肌肉行，P3 节点定稿真实映射）")
    ap.add_argument("--annotations", default=None,
                    help="B1a raw 功能注解 CSV 路径（olfactory/noci 等，"
                         "默认 winding_s1/Supplementary-Data-S1/annotations.csv）")
    ap.add_argument("--gmax-scale", type=float, default=None,
                    help="全局突触电导缩放（D5 第一遍先验；默认 1.0 恒等）")
    ap.add_argument("--g1-scale", type=int, default=3016,
                    help="G1 门规模（部分扫描时用小规模验证三杠杆机制；"
                         "默认 3016 全脑）")
    args = ap.parse_args()

    if args.smoke and args.wait:
        print("--smoke 与 --wait 互斥：冒烟用占位连接组，不等待真实数据")
        sys.exit(2)
    if args.smoke and (args.nt_fallback or args.provisional_muscles):
        print("--smoke 与 --nt-fallback/--provisional-muscles 互斥"
              "（临时回退只用于真实连接组数据缺口）")
        sys.exit(2)
    # 默认 annotations 路径（B1a raw；存在才 join）
    if args.annotations is None:
        cand = os.path.join(DATA_DIR, "m8_raw", "winding_s1",
                            "Supplementary-Data-S1", "annotations.csv")
        args.annotations = cand if os.path.exists(cand) else None
    t_start = time.perf_counter()
    spec = _spec_for(args)
    existing = _load_existing()
    if not args.smoke:
        # 真实数据运行：冒烟行（placeholder_smoke）作废，需重跑
        existing = {k: v for k, v in existing.items()
                    if v.get("source") != "placeholder_smoke"}
    rows = []
    selected = []
    plast_all = (list(PLASTICITY_AXIS) if args.plasticity == "all"
                 else [p.strip() for p in args.plasticity.split(",")])
    for (scale, fid, plast, proto, note) in GRID:
        if args.rows != "all" and f"{scale}{fid[0]}" not in args.rows.split(","):
            continue
        if plast not in plast_all:
            continue
        if fid == "hh" and not args.run_hh:
            continue
        if scale == 1000 and fid == "two_comp" and not args.run_1000_two_comp:
            continue
        if proto.get("anchor") and not args.run_anchor:
            continue
        if args.smoke and (scale > spec.n_neurons
                           or fid in ("hh", "two_comp")):
            # 冒烟：占位 300 神经元——只跑 point 档（two_comp/hh 由
            # test_larva_smoke 另行冒烟，避免扫描冒烟膨胀）
            if fid != "point":
                continue
        selected.append((scale, fid, plast, proto, note))
    if not selected:
        print("无选中格点（--rows 语法：300p,1000p,3016p；"
              "--plasticity none,stp,stdp,stdp_homeo）")
        sys.exit(1)

    for (scale, fid, plast, proto, note) in selected:
        key = (scale, fid, plast)
        if key in existing and existing[key]["status"] == "done":
            print(f"[reuse] {scale}/{fid}/{plast} 已落盘，跳过", flush=True)
            rows.append(existing[key])
            continue
        print(f"[run] {scale}/{fid}/{plast} ({note}) 开始 "
              f"{time.strftime('%H:%M:%S')}", flush=True)
        row = run_grid_point(scale, fid, plast, proto, args, spec)
        row["notes"] = (row.get("notes", "") + note).strip()
        rows.append(row)
        _save_rows(rows)
        print(f"[done] {scale}/{fid}/{plast}: status={row['status']} "
              f"CI={row['ci_mean']} silent={row['resting_silent_frac']} "
              f"LI={row['li']} probe={row['probe_wall_s']}s "
              f"({time.perf_counter() - t_start:.0f}s elapsed)", flush=True)

    seen = {(int(r["scale"]), r["fidelity"], r["plasticity"]) for r in rows}
    for (scale, fid, plast, proto, note) in GRID:
        key = (scale, fid, plast)
        if key in seen:
            continue
        if key in existing and existing[key]["status"] == "done":
            rows.append(existing[key])
            continue
        r = _row_defaults(scale, fid, plast,
                          "placeholder_smoke" if args.smoke else "connectome",
                          status="skipped")
        r["notes"] = f"skipped: {note}"
        rows.append(r)
    _save_rows(rows)

    # G1 双状态门（三杠杆消融 sanity）
    g1 = run_g1_gate(args, spec)
    with open(G1_RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(g1, f, ensure_ascii=False, indent=2)
    print(f"[g1] verdict={g1['verdict']}: {g1['detail']}", flush=True)

    # G0 决策
    g0 = g0_decision(rows, g1, smoke=args.smoke)
    params_csv = write_larva_params(g0, g1, args)
    print(f"[g0] decision={g0['decision']} scale={g0['scale']} "
          f"fidelity={g0['fidelity']} → {params_csv}", flush=True)

    png = plot_curves(rows)
    print(f"CSV: {SCALING_CSV}")
    print(f"PNG: {png}")
    with open(DONE_MARKER, "w") as f:
        f.write(f"scan finished {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"DONE marker: {DONE_MARKER}")


if __name__ == "__main__":
    main()
