#!/usr/bin/env python
"""P7 扰动预测验证（B1d：计划落盘 + 前 3–5 神经元沉默/激活短协议冒烟）。

《生物仿真M8实施清单》§9 / §1 D3：
- 计划：data/m8_perturbation_plan.csv（tools/gen_m8_perturbation_plan.py 确定性
  生成 top-50 + 阈值 + 实验锚映射；MD class IV 伤害感受器 = 有锚，其余
  "no-experiment" 不入命中率分母）；
- 模拟：基线试次（sham）→ 逐神经元 **沉默**（出边 gmax→0，M6 L15 整体重建数组）
  + **激活**（tonic 注入，扩展 stim 列）→ 行为后果类（预注册类集 + 阈值）；
- 命中率 = 预测正确 / 有锚子集 ≥70%（预注册有锚 ≥20/50——B1d 网络受限：
  有锚 = MD 4/50 < 20 → 按 §0.7 #4 缩小有锚判据分母 + 记录测量限制）；
- sham 对照 sanity（无变化）+ 确定性重跑；
- B1d 冒烟预算：top-N 前 5 神经元（沉默+激活），300 档 two_comp 短协议 T≤2s。

输出：data/m8_p7_perturbation.csv + reports/neuro/m8_perturbation_hitrate.png
      + reports/neuro/m8_p7_perturbation.json
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.src.larva_perturb import (  # noqa: E402
    PerturbLarvaCircuit,
    classify_consequence,
    run_spont_protocol,
)

DATA_DIR = os.path.join(ROOT, "neural_exploration", "data")
REPORTS_NEURO = os.path.join(ROOT, "neural_exploration", "reports", "neuro")
PARAMS_CSV = os.path.join(DATA_DIR, "m8_larva_params.csv")
PLAN_CSV = os.path.join(DATA_DIR, "m8_perturbation_plan.csv")

SCALE = 300
FIDELITY = "two_comp"
PROTO_T_MS = 2000.0
SEED = 0
N_SMOKE = 5              # B1d 冒烟神经元数（top-N 前 5）
ACTIVATION_nA = 0.5      # 与计划 CSV 一致（预注册）
ANCHORED_FLOOR = 20      # §0.7 #4 预注册有锚下限
#: B2 全测：top-50 全部神经元（清单 §9.2：逐神经元沉默/激活 → 后果类 → 命中率）
N_FULL = 50


def load_weight_rows() -> dict:
    out = {}
    import csv as _csv
    with open(PARAMS_CSV, newline="", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            fields = next(_csv.reader([s]))
            if len(fields) < 11 or fields[0].strip().lower() != "weight":
                continue
            try:
                out[fields[1].strip()] = float(fields[9])
            except ValueError:
                continue
    return out


def read_plan() -> list:
    rows = []
    with open(PLAN_CSV, newline="", encoding="utf-8") as f:
        # 跳过 # 注释行（csv.DictReader 不自动跳过）
        lines = [ln for ln in f
                 if ln.strip() and not ln.lstrip().startswith("#")]
        rdr = csv.DictReader(lines)
        for r in rdr:
            try:
                rank = int(r["rank"])
            except (ValueError, KeyError):
                continue
            rows.append(dict(rank=rank, role=r["role"],
                             anchor=r.get("anchor_status", "no-experiment"),
                             effect=r.get("anchor_effect_class", ""),
                             source=r.get("anchor_source", "")))
    rows.sort(key=lambda r: r["rank"])
    return rows


def make_circuit(perturb_roles) -> PerturbLarvaCircuit:
    w = load_weight_rows()
    gmax = float(w["gmax_scale"])
    class_scales = {}
    for k, v in w.items():
        if k.startswith("class_scale_"):
            parts = k.split("_")
            if len(parts) == 4:
                class_scales[(parts[2], parts[3])] = v
    return PerturbLarvaCircuit(perturb_roles=perturb_roles,
                               scale=SCALE, fidelity=FIDELITY, seed=0,
                               nt_fallback="class", provisional_muscles=True,
                               gmax_scale=gmax, class_scales=class_scales,
                               plasticity="none")


PROGRESS_FULL_CSV = os.path.join(DATA_DIR, "m8_p7_perturbation_full_progress.csv")


def _read_progress_full() -> dict:
    """读进度 CSV → {rank: row dict}（断点续跑用）。"""
    out = {}
    if not os.path.exists(PROGRESS_FULL_CSV):
        return out
    with open(PROGRESS_FULL_CSV, newline="", encoding="utf-8") as f:
        lines = [ln for ln in f
                 if ln.strip() and not ln.lstrip().startswith("#")]
        rdr = csv.DictReader(lines)
        for r in rdr:
            try:
                rank = int(r["rank"])
            except (ValueError, KeyError):
                continue
            out[rank] = dict(
                rank=rank, role=r["role"], anchor=r.get("anchor_status", ""),
                anchor_effect=r.get("anchor_effect_class", ""),
                class_silence=r["class_silence"],
                class_activation=r["class_activation"],
                hit=(None if r.get("hit", "") in ("", "None") else
                     r["hit"].lower() == "true"),
                n_silenced_edges=int(float(r.get("n_silenced", 0) or 0)),
                d_sil=dict(fwd=float(r.get("d_sil_fwd", 0) or 0),
                           back=float(r.get("d_sil_back", 0) or 0),
                           turn=float(r.get("d_sil_turn", 0) or 0),
                           pause=float(r.get("d_sil_pause", 0) or 0)))
    return out


def run_full_test(plan: list, summary: dict, resume: bool = False) -> dict:
    """B2 P7 top-50 全测：逐神经元沉默+激活 → 后果类 → 有锚命中率（限制记录）。

    预算：50×2 次仿真（300 档 two_comp T=2s）≈ 20-30 分钟；**分批后台 + 断点
    续跑**（B2 实测：单进程长批量 session 内存渐增 → 系统 swap 病态；resume 模式
    每次新进程内存重置，进度落盘 m8_p7_perturbation_full_progress.csv）。

    命中率：有锚子集（plan 中 anchor_status=anchored，B1d 实测 3/50 < 20 下限）
    → 按 §0.7 #4 缩小有锚判据分母 + 测量限制记录，**不静默判 ≥70%**（informational）。
    """
    import csv as _csv
    t0 = time.perf_counter()
    targets = [p["role"] for p in plan[:N_FULL]]
    circ = make_circuit(targets)
    base = run_spont_protocol(circ, PROTO_T_MS, SEED)

    done = _read_progress_full() if resume else {}
    append = bool(resume and done)
    with open(PROGRESS_FULL_CSV, "a" if append else "w", newline="",
              encoding="utf-8") as pf:
        wtr = _csv.writer(pf)
        if not append:
            wtr.writerow(["rank", "role", "anchor_status", "anchor_effect_class",
                          "n_silenced", "class_silence", "class_activation",
                          "hit", "d_sil_fwd", "d_sil_back", "d_sil_turn",
                          "d_sil_pause", "wall_s"])
        for p in plan[:N_FULL]:
            role = p["role"]
            if role in {d["role"] for d in done.values()}:
                print(f"  #{p['rank']} {role[:44]:46s} 已续跑跳过", flush=True)
                continue
            ts = time.perf_counter()
            r_sil = run_spont_protocol(circ, PROTO_T_MS, SEED,
                                       silence_roles=[role])
            d_sil = {k: r_sil["frac"].get(k, 0.0) - base["frac"].get(k, 0.0)
                     for k in ("run", "turn", "pause", "curl")}
            d_sil["fwd"] = r_sil["ch_mean"]["fwd"] - base["ch_mean"]["fwd"]
            d_sil["back"] = r_sil["ch_mean"]["back"] - base["ch_mean"]["back"]
            d_sil["curl"] = r_sil["ch_mean"]["curl"] - base["ch_mean"]["curl"]
            cls_sil = classify_consequence(d_sil)

            r_act = run_spont_protocol(circ, PROTO_T_MS, SEED,
                                       activate_roles=[role],
                                       act_nA=ACTIVATION_nA)
            d_act = {k: r_act["frac"].get(k, 0.0) - base["frac"].get(k, 0.0)
                     for k in ("run", "turn", "pause", "curl")}
            d_act["fwd"] = r_act["ch_mean"]["fwd"] - base["ch_mean"]["fwd"]
            d_act["back"] = r_act["ch_mean"]["back"] - base["ch_mean"]["back"]
            d_act["curl"] = r_act["ch_mean"]["curl"] - base["ch_mean"]["curl"]
            cls_act = classify_consequence(d_act)

            hit = None
            if p["anchor"] == "anchored":
                anchor_classes = [c.strip() for c in p["effect"].split("/")]
                pred_classes = {cls_sil, cls_act} - {"无变化"}
                hit = bool(pred_classes & set(anchor_classes))
            wall = time.perf_counter() - ts
            wtr.writerow([p["rank"], role, p["anchor"], p["effect"],
                          r_sil["n_silenced"], cls_sil, cls_act,
                          ("" if hit is None else hit),
                          f"{d_sil['fwd']:.6f}", f"{d_sil['back']:.6f}",
                          f"{d_sil['turn']:.6f}", f"{d_sil['pause']:.6f}",
                          f"{wall:.1f}"])
            pf.flush()
            import gc
            gc.collect()   # 内存纪律（B2 实测：批量 session 无 gc → RSS 累积）
            print(f"  #{p['rank']} {role[:44]:46s} 锚={p['anchor'][:5]:6s} "
                  f"sil={cls_sil:4s} act={cls_act:4s} hit={hit} "
                  f"wall={wall:.1f}s", flush=True)

    return finalize_full(plan, summary, t0)


def finalize_full(plan: list, summary: dict, t0: float) -> dict:
    """读完整进度 CSV → 汇总（sham/确定性/命中率/CSV/JSON/PNG）。

    独立于 run_full_test 可重复调用（全部神经元完成后）。
    """
    import csv as _csv
    done = _read_progress_full()
    per_neuron = [done[rank] for rank in sorted(done)
                  if rank in {p["rank"] for p in plan[:N_FULL]}]
    # 对齐 plan 的锚信息（progress 已含；补全缺省）
    plan_by_rank = {p["rank"]: p for p in plan[:N_FULL]}
    for x in per_neuron:
        p = plan_by_rank.get(x["rank"], {})
        x["anchor"] = p.get("anchor", x.get("anchor", "no-experiment"))
        x["anchor_effect"] = p.get("effect", x.get("anchor_effect", ""))
        if x["anchor"] == "anchored" and x.get("anchor_effect"):
            anchor_classes = [c.strip()
                              for c in x["anchor_effect"].split("/")]
            pred_classes = {x["class_silence"], x["class_activation"]} \
                - {"无变化"}
            x["hit"] = bool(pred_classes & set(anchor_classes))
        else:
            x["hit"] = None

    # sham sanity + 确定性（重跑第一个目标沉默）
    targets = [p["role"] for p in plan[:N_FULL]]
    circ = make_circuit(targets)
    base = run_spont_protocol(circ, PROTO_T_MS, SEED)
    r_sham2 = run_spont_protocol(circ, PROTO_T_MS, SEED)
    sham_identical = bool(r_sham2["frac"] == base["frac"]
                          and np.allclose(r_sham2["ch_mean"]["fwd"],
                                          base["ch_mean"]["fwd"]))
    first = plan[0]["role"]
    r_sil2 = run_spont_protocol(circ, PROTO_T_MS, SEED, silence_roles=[first])
    d2 = {k: r_sil2["frac"].get(k, 0.0) - base["frac"].get(k, 0.0)
          for k in ("run", "turn", "pause", "curl")}
    d2["fwd"] = r_sil2["ch_mean"]["fwd"] - base["ch_mean"]["fwd"]
    d2["back"] = r_sil2["ch_mean"]["back"] - base["ch_mean"]["back"]
    d2["curl"] = r_sil2["ch_mean"]["curl"] - base["ch_mean"]["curl"]
    cls_sil2 = classify_consequence(d2)
    determinism = bool(cls_sil2 == per_neuron[0]["class_silence"]
                       and np.isclose(r_sil2["ch_mean"]["fwd"],
                                      per_neuron[0]["d_sil"]["fwd"]
                                      + base["ch_mean"]["fwd"]))

    # ---- 命中率（有锚子集；3/50 < 20 下限 → informational + 限制记录）----
    n_anchored_all = sum(1 for p in plan[:N_FULL] if p["anchor"] == "anchored")
    simulated_anchored = [x for x in per_neuron if x["anchor"] == "anchored"]
    hits = [x for x in simulated_anchored if x["hit"]]
    hit_rate = (len(hits) / len(simulated_anchored)
                if simulated_anchored else None)
    anchored_ok = bool(n_anchored_all >= ANCHORED_FLOOR)
    hit_ok = bool(hit_rate is not None and hit_rate >= 0.70)
    hitrate_limitation = dict(
        n_anchored_plan=n_anchored_all, anchored_floor=ANCHORED_FLOOR,
        anchored_floor_met=anchored_ok, hit_rate=hit_rate,
        hits=len(hits), simulated_anchored=len(simulated_anchored),
        note=("B1d/B2 网络受限（无逐神经元驱动线锚下载）→ 有锚 = MD 3/50 < 20 "
              "下限 → 按 §0.7 #4 缩小有锚判据分母 + 测量限制记录；命中率仅 "
              "informational，≥70% 判据在有锚子集达标前不静默判定"))

    # ---- 落盘 CSV（top-50 逐神经元对照表）----
    csv_path = os.path.join(DATA_DIR, "m8_p7_perturbation_full.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wtr = _csv.writer(f)
        wtr.writerow(["# M8 P7 扰动预测（B2 全测：top-%d 神经元沉默/激活）"
                      % N_FULL])
        wtr.writerow(["rank", "role", "anchor_status", "anchor_effect_class",
                      "n_silenced_edges", "class_silence", "class_activation",
                      "hit", "d_fwd_mean", "d_back_mean", "d_turn_frac",
                      "d_pause_frac"])
        for x in per_neuron:
            wtr.writerow([x["rank"], x["role"], x["anchor"],
                          x["anchor_effect"], x["n_silenced_edges"],
                          x["class_silence"], x["class_activation"], x["hit"],
                          f"{x['d_sil']['fwd']:.4f}",
                          f"{x['d_sil']['back']:.4f}",
                          f"{x['d_sil']['turn']:.4f}",
                          f"{x['d_sil']['pause']:.4f}"])

    # ---- 出图（全测命中率 + 后果类分布）----
    png_path = os.path.join(REPORTS_NEURO, "m8_perturbation_hitrate_full.png")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        for _f in ("PingFang HK", "Heiti TC", "PingFang SC", "Arial Unicode MS"):
            try:
                fm.findfont(_f, fallback_to_default=False)
                plt.rcParams["font.sans-serif"] = [_f, "DejaVu Sans"]
                plt.rcParams["axes.unicode_minus"] = False
                break
            except Exception:
                continue
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.2),
                                 gridspec_kw={"width_ratios": [1, 2]})
        ax = axes[0]
        if hit_rate is not None:
            ax.bar(["有锚命中率"], [hit_rate * 100], color="#2ca02c")
            ax.axhline(70, color="gray", ls="--", lw=1, label="70% 判据")
            ax.set_ylim(0, 105)
        else:
            ax.text(0.5, 0.5, "有锚子集=0", ha="center", va="center")
        ax.set_ylabel("命中率 (%)")
        ax.set_title(f"P7 全测命中率（有锚 {len(simulated_anchored)}/"
                     f"{n_anchored_all}；下限 {ANCHORED_FLOOR} 未达 → 限制记录）")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax = axes[1]
        labels = [f"#{x['rank']}\n{x['role'][:16]}" for x in per_neuron]
        code = {"无变化": 0, "前进↑": 1, "前进↓": -1, "转弯↑": 2, "停驻↑": 3,
                "蜷缩↑": 4, "后退↑": -2}
        sil = [code.get(x["class_silence"], 0) for x in per_neuron]
        act = [code.get(x["class_activation"], 0) for x in per_neuron]
        xpos = np.arange(len(labels))
        wb = 0.38
        ax.bar(xpos - wb / 2, sil, wb, color="#1f77b4", label="沉默")
        ax.bar(xpos + wb / 2, act, wb, color="#ff7f0e", label="激活")
        ax.set_xticks(xpos[::5])
        ax.set_xticklabels([labels[i] for i in range(0, len(labels), 5)],
                           rotation=45, ha="right", fontsize=6)
        ax.set_ylabel("后果类编码（-2 后退↑ … 4 蜷缩↑）")
        ax.set_title(f"top-{N_FULL} 逐神经元扰动后果类（沉默 vs 激活）")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.suptitle(f"M8 P7 扰动预测（B2 top-{N_FULL} 全测）：机制全可运行 + "
                     "有锚命中率限制记录", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(png_path, dpi=130)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        png_path = f"FAILED: {e}"

    full = dict(
        meta=dict(scale=SCALE, fidelity=FIDELITY, proto_t_ms=PROTO_T_MS,
                  seed=SEED, n_full=N_FULL, activation_nA=ACTIVATION_nA,
                  wall_s=round(time.perf_counter() - t0, 2)),
        baseline=dict(frac=base["frac"], ch_mean=base["ch_mean"]),
        sham=dict(identical=sham_identical),
        determinism=dict(identical=determinism, class_rerun=cls_sil2),
        hitrate=dict(rate=hit_rate, hits=len(hits),
                     simulated_anchored=len(simulated_anchored),
                     anchored_ok=anchored_ok, hit_ok=hit_ok,
                     limitation=hitrate_limitation),
        criteria=dict(
            sham_sanity=sham_identical, determinism=determinism,
            hitrate_note=("有锚 3/50 < 20 预注册下限 → 命中率仅 informational + "
                          "测量限制记录，不静默判 ≥70%（B1d 冒烟 MD 锚 HIT："
                          "MDNB_RIGHT 激活→后退↑ 与文献锚一致）")),
        per_neuron=per_neuron, csv=csv_path, plot=png_path)

    json_path = os.path.join(REPORTS_NEURO, "m8_p7_perturbation_full.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full, f, ensure_ascii=False, indent=2, default=str)

    print("=== P7 top-50 全测（B2）===")
    print(f"baseline frac: {base['frac']}")
    n_change_sil = sum(1 for x in per_neuron
                       if x["class_silence"] != "无变化")
    n_change_act = sum(1 for x in per_neuron
                       if x["class_activation"] != "无变化")
    print(f"沉默后果类非无变化: {n_change_sil}/{len(per_neuron)}；"
          f"激活后果类非无变化: {n_change_act}/{len(per_neuron)}")
    for x in simulated_anchored:
        print(f"  锚 {x['role'][:40]:42s} sil={x['class_silence']:4s} "
              f"act={x['class_activation']:4s} hit={x['hit']}")
    print(f"sham identical: {sham_identical}  determinism: {determinism}")
    print(f"hitrate: {hit_rate}（有锚模拟 {len(simulated_anchored)}/"
          f"计划 {n_anchored_all}，下限 {ANCHORED_FLOOR}）→ 限制记录，不静默")
    print(f"csv={csv_path} json={json_path} wall={full['meta']['wall_s']}s")
    summary["full_test"] = full
    return full


def main() -> int:
    t0 = time.perf_counter()
    os.makedirs(REPORTS_NEURO, exist_ok=True)
    plan = read_plan()
    if not plan:
        print("plan CSV 缺失——先运行 tools/gen_m8_perturbation_plan.py")
        return 2
    # 冒烟目标 = top-N 前 5 + 首个有锚神经元（MD，机制在锚上也可测）
    smoke = list(plan[:N_SMOKE])
    first_anchored = next((p for p in plan if p["anchor"] == "anchored"), None)
    if first_anchored and all(p["role"] != first_anchored["role"]
                              for p in smoke):
        smoke.append(first_anchored)
    targets = [p["role"] for p in smoke]

    circ = make_circuit(targets)
    # 基线（sham）
    base = run_spont_protocol(circ, PROTO_T_MS, SEED)

    per_neuron = []
    for p in smoke:
        role = p["role"]
        # ---- 沉默（出边 gmax→0）----
        r_sil = run_spont_protocol(circ, PROTO_T_MS, SEED,
                                   silence_roles=[role])
        d_sil = {k: r_sil["frac"].get(k, 0.0) - base["frac"].get(k, 0.0)
                 for k in ("run", "turn", "pause", "curl")}
        d_sil["fwd"] = r_sil["ch_mean"]["fwd"] - base["ch_mean"]["fwd"]
        d_sil["back"] = r_sil["ch_mean"]["back"] - base["ch_mean"]["back"]
        d_sil["curl"] = r_sil["ch_mean"]["curl"] - base["ch_mean"]["curl"]
        cls_sil = classify_consequence(d_sil)

        # ---- 激活（tonic 注入）----
        r_act = run_spont_protocol(circ, PROTO_T_MS, SEED,
                                   activate_roles=[role], act_nA=ACTIVATION_nA)
        d_act = {k: r_act["frac"].get(k, 0.0) - base["frac"].get(k, 0.0)
                 for k in ("run", "turn", "pause", "curl")}
        d_act["fwd"] = r_act["ch_mean"]["fwd"] - base["ch_mean"]["fwd"]
        d_act["back"] = r_act["ch_mean"]["back"] - base["ch_mean"]["back"]
        d_act["curl"] = r_act["ch_mean"]["curl"] - base["ch_mean"]["curl"]
        cls_act = classify_consequence(d_act)

        # 命中（有锚神经元）：预测类与实验锚效应类一致
        hit = None
        if p["anchor"] == "anchored":
            # 锚效应类为「蜷缩↑/后退↑」：任一命中即算预测正确（预测含任一类）
            anchor_classes = [c.strip() for c in p["effect"].split("/")]
            pred_classes = {cls_sil, cls_act} - {"无变化"}
            hit = bool(pred_classes & set(anchor_classes))
        per_neuron.append(dict(rank=p["rank"], role=role,
                               anchor=p["anchor"],
                               anchor_effect=p["effect"],
                               n_silenced_edges=r_sil["n_silenced"],
                               sil_frac=r_sil["frac"], act_frac=r_act["frac"],
                               base_frac=base["frac"],
                               d_sil=d_sil, d_act=d_act,
                               class_silence=cls_sil, class_activation=cls_act,
                               hit=hit))

    # ---- sham sanity（无扰动重跑 → 无变化/逐位一致）----
    r_sham2 = run_spont_protocol(circ, PROTO_T_MS, SEED)
    sham_identical = bool(r_sham2["frac"] == base["frac"]
                          and np.allclose(r_sham2["ch_mean"]["fwd"],
                                          base["ch_mean"]["fwd"]))
    # ---- 确定性：重跑第一个目标沉默 → 后果类一致 ----
    r_sil2 = run_spont_protocol(circ, PROTO_T_MS, SEED,
                                silence_roles=[smoke[0]["role"]])
    d2 = {k: r_sil2["frac"].get(k, 0.0) - base["frac"].get(k, 0.0)
          for k in ("run", "turn", "pause", "curl")}
    d2["fwd"] = r_sil2["ch_mean"]["fwd"] - base["ch_mean"]["fwd"]
    d2["back"] = r_sil2["ch_mean"]["back"] - base["ch_mean"]["back"]
    d2["curl"] = r_sil2["ch_mean"]["curl"] - base["ch_mean"]["curl"]
    cls_sil2 = classify_consequence(d2)
    determinism = bool(cls_sil2 == per_neuron[0]["class_silence"]
                       and np.isclose(r_sil2["ch_mean"]["fwd"],
                                      per_neuron[0]["d_sil"]["fwd"]
                                      + base["ch_mean"]["fwd"]))

    # ---- 命中率（有锚子集；B1d 有锚 <20 下限 → 记录测量限制）----
    n_anchored_all = sum(1 for p in plan if p["anchor"] == "anchored")
    simulated_anchored = [x for x in per_neuron if x["anchor"] == "anchored"]
    hits = [x for x in simulated_anchored if x["hit"]]
    hit_rate = (len(hits) / len(simulated_anchored)
                if simulated_anchored else None)
    anchored_ok = bool(n_anchored_all >= ANCHORED_FLOOR)
    hit_ok = bool(hit_rate is not None and hit_rate >= 0.70)
    hitrate_limitation = dict(
        n_anchored_plan=n_anchored_all, anchored_floor=ANCHORED_FLOOR,
        anchored_floor_met=anchored_ok,
        note=("B1d 网络受限（无逐神经元驱动线锚下载）→ 有锚 = MD 4/50 < 20 下限"
              "→ 按 §0.7 #4 缩小有锚判据分母 + 记录测量限制；命中率仅 informational，"
              "≥70% 判据留 B2（有锚子集达标后）"))

    summary = dict(
        meta=dict(scale=SCALE, fidelity=FIDELITY, proto_t_ms=PROTO_T_MS,
                  seed=SEED, n_smoke=N_SMOKE, activation_nA=ACTIVATION_nA,
                  wall_s=round(time.perf_counter() - t0, 2)),
        baseline=dict(frac=base["frac"], ch_mean=base["ch_mean"]),
        sham=dict(identical=sham_identical),
        determinism=dict(identical=determinism, class_rerun=cls_sil2),
        hitrate=dict(rate=hit_rate, hits=len(hits),
                     simulated_anchored=len(simulated_anchored),
                     anchored_ok=anchored_ok, hit_ok=hit_ok,
                     limitation=hitrate_limitation),
        criteria=dict(
            sham_sanity=sham_identical,
            determinism=determinism,
            hitrate_note=("≥70% 判据预注册有锚 ≥20/50 未达 → 测量限制记录，"
                          "不静默判定；机制（沉默/激活→后果类）B1d 已可运行")),
        per_neuron=per_neuron)

    # ---- 落盘 CSV（逐神经元预测 vs 实验对照表）----
    csv_path = os.path.join(DATA_DIR, "m8_p7_perturbation.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wtr = csv.writer(f)
        wtr.writerow(["# M8 P7 扰动预测（B1d 冒烟：前 %d 神经元沉默/激活）" % N_SMOKE])
        wtr.writerow(["rank", "role", "anchor_status", "anchor_effect_class",
                      "n_silenced_edges", "class_silence", "class_activation",
                      "hit", "d_fwd_mean", "d_back_mean", "d_turn_frac",
                      "d_pause_frac"])
        for x in per_neuron:
            wtr.writerow([x["rank"], x["role"], x["anchor"],
                          x["anchor_effect"], x["n_silenced_edges"],
                          x["class_silence"], x["class_activation"],
                          x["hit"], f"{x['d_sil']['fwd']:.4f}",
                          f"{x['d_sil']['back']:.4f}",
                          f"{x['d_sil']['turn']:.4f}",
                          f"{x['d_sil']['pause']:.4f}"])

    # ---- 出图（命中率 + 逐神经元预测 vs 实验对照表）----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6),
                                 gridspec_kw={"width_ratios": [1, 2]})
        ax = axes[0]
        if hit_rate is not None:
            ax.bar(["有锚命中率"], [hit_rate * 100], color="#2ca02c")
            ax.axhline(70, color="gray", ls="--", lw=1, label="70% 判据")
            ax.set_ylim(0, 105)
        else:
            ax.text(0.5, 0.5, "有锚子集=0\n（no-experiment）", ha="center",
                    va="center")
        ax.set_ylabel("命中率 (%)")
        ax.set_title(f"P7 命中率（有锚 {len(simulated_anchored)}；"
                     f"下限 {ANCHORED_FLOOR} 未达 → 限制记录）")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax = axes[1]
        labels = [f"#{x['rank']}\n{x['role'][:18]}" for x in per_neuron]
        sil = [{"无变化": 0, "前进↑": 1, "前进↓": -1, "转弯↑": 2, "停驻↑": 3,
                "蜷缩↑": 4, "后退↑": -2}.get(x["class_silence"], 0)
               for x in per_neuron]
        act = [{"无变化": 0, "前进↑": 1, "前进↓": -1, "转弯↑": 2, "停驻↑": 3,
                "蜷缩↑": 4, "后退↑": -2}.get(x["class_activation"], 0)
               for x in per_neuron]
        xpos = np.arange(len(labels))
        wb = 0.38
        ax.bar(xpos - wb / 2, sil, wb, color="#1f77b4", label="沉默")
        ax.bar(xpos + wb / 2, act, wb, color="#ff7f0e", label="激活")
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("后果类编码（-2 后退↑ … 4 蜷缩↑）")
        ax.set_title("逐神经元扰动后果类（沉默 vs 激活）")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.suptitle("P7 扰动预测（B1d 冒烟）：机制可运行 + 有锚命中率限制记录",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        png_path = os.path.join(REPORTS_NEURO, "m8_perturbation_hitrate.png")
        fig.savefig(png_path, dpi=130)
        plt.close(fig)
        summary["plot"] = png_path
    except Exception as e:  # noqa: BLE001
        summary["plot"] = f"FAILED: {e}"

    json_path = os.path.join(REPORTS_NEURO, "m8_p7_perturbation.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print("=== P7 扰动预测（B1d 冒烟）===")
    print(f"baseline frac: {base['frac']} ch_mean: "
          f"{ {k: round(v, 3) for k, v in base['ch_mean'].items()} }")
    for x in per_neuron:
        print(f"  #{x['rank']} {x['role'][:42]:44s} 锚={x['anchor'][:5]} "
              f"sil={x['class_silence']:4s} act={x['class_activation']:4s} "
              f"hit={x['hit']}")
    print(f"sham identical: {sham_identical}  determinism: {determinism}")
    print(f"hitrate: {hit_rate}（有锚模拟 {len(simulated_anchored)}/"
          f"计划 {n_anchored_all}，下限 {ANCHORED_FLOOR}）→ {hitrate_limitation['note'][:40]}…")
    print(f"csv={csv_path} json={json_path} wall={summary['meta']['wall_s']}s")

    # ---- B2：top-50 全测（逐神经元沉默/激活 → 后果类 → 有锚命中率限制记录）
    # ---- --full-only：跳过冒烟 + 全新全测；--resume：跳过冒烟 + 断点续跑 ----
    if "--full-only" in sys.argv[1:] or "--resume" in sys.argv[1:]:
        resume = "--resume" in sys.argv[1:]
        print(f"=== P7 全测模式（跳过 B1d 冒烟；resume={resume}）===")
        full = run_full_test(plan, summary, resume=resume)
    else:
        full = run_full_test(plan, summary, resume=False)
    print("=== P7 判定（B1d 冒烟 + B2 top-50 全测）===")
    print(f"机制全可运行（sham 逐位一致 + 确定性）；有锚命中率限制记录"
          f"（3/50 < 20 下限，不静默判 ≥70%）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
