"""M8 步骤 4（§1 D5）：权重定稿与行为带校准——类级缩放 s_k 扫描（≤5 轮）。

对应《生物仿真M8实施清单》§6（步骤 4：权重定稿，P4–P6 前置）与 §1 D5
（类级缩放 + M5 先验 + 行为学反推）：
  - **先验**：M5 302 网络定稿权重作同源通路初始（class_scale 全 1.0 恒等 =
    M5 先验；gmax_scale=0.05 = B1b 第一遍全局缩放进入可工作区）；
  - **类级缩放**：按 (pre 类, post 类) 分桶 s_k，w_ij = w0_class · s_k ·
    gmax_scale（§1 D5；不做逐突触拟合，≤30 全局参数）；
  - **行为学反推**：五目标联合（趋化 CI / 逃避方向 / 自发分布 / 静息静默 /
    学习指数 LI），少量组合（≤12）× 300 档降阶模型少点确认（短协议
    T≤5s，确定性 p=1/n=1）→ 定稿 data/m8_larva_params.csv 权重行 +
    data/m8_calibration.csv + reports/neuro/m8_calibration.png；
  - **判据（预注册，本工具内嵌）**：
      * 趋化：CI > 0（正趋化符号；§0 P4/§4.3 AWC 嗅觉锚）；
      * 静息静默：silent ∈ [50, 90]%（G1 带，§1 D6）；
      * 自发 bout 活动：≥ 10%（G1 双状态，§0.5）；
      * 学习指数：LI ≥ 0.05（stdp 档，LI_APPEAR_THRESHOLD，机制级）；
      * 逃避：D_peak > 0.3 → back（降阶正确性锚，§3.4，informational）。
  - **B1c3 实测修正（2026-08-29，CSV sha 9d80cf74…）**：
      * **point 档 CI 结构性为 0**：300 point 全 gmax/类级组合 CI≡0——嗅觉链
        ORN→PN→KC 传导在点档亚阈值（node3 面积 ~134× < soma，two_comp 电导
        密度高得多）；G0 已定稿 fidelity_behavior=two_comp → **校准在 two_comp
        档执行**（--fidelity 默认 two_comp）。
      * **工作区**：gmax=0.05 + s2i=6/i2i=3/i2m=3 → none 档 silent∈带/CI>0，
        stdp 档 LI>0（eta 微调过阈）；**LI 阈依赖 stdp_eta**（B1c3 实测
        eta=10 → LI≈0.046 贴阈，eta=12 → LI≈0.21；eta 为扫描探针值，
        P5 全协议定稿学习率由学习验证节点定稿）。
  - **失败处置（不静默）**：无组合满足 → 反证记录（300 档无 GABA 抑制时
    CI/LI 不可转正的证据）→ 三态裁决请求写入校准 CSV note 与
    docs/m8_env_notes.md（§0.4 反证路径 + 三态裁决纪律）。

运行（复现入口）：
  .venv-neuro/bin/python -m neural_exploration.tools.calibrate_m8_weights
    [--rounds N] [--combo <name>] [--fidelity two_comp] [--stdp-eta 12.0]
    [--smoke]
长任务：每组合即时落盘 m8_calibration.csv（断点续跑；M4 L25 纪律）。

确定性：p=1/n=1；seed=0；同参数重跑逐位一致。并发纪律：运行前确认无
并行 Brian2 进程（cython 缓存锁竞争，M6 L9#6/L27）。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.src.larva_circuit import (  # noqa: E402
    LI_APPEAR_THRESHOLD,
    LarvaCircuit,
    build_placeholder_spec,
    wait_for_csv,
)

DATA_DIR = os.path.join(ROOT, "neural_exploration", "data")
REPORTS_DIR = os.path.join(ROOT, "neural_exploration", "reports", "neuro")
CALIBRATION_CSV = os.path.join(DATA_DIR, "m8_calibration.csv")
CALIBRATION_PNG = os.path.join(REPORTS_DIR, "m8_calibration.png")
LARVA_PARAMS_CSV = os.path.join(DATA_DIR, "m8_larva_params.csv")
ANNOT_DEFAULT = os.path.join(DATA_DIR, "m8_raw", "winding_s1",
                             "Supplementary-Data-S1", "annotations.csv")

#: G1 静默带（§1 D6 草案；成像文献校准后以 m8_behavior_reference.csv 为准）
SILENT_BAND = (0.50, 0.90)
#: G1 bout 活动下限（§0.5）
BOUT_FLOOR = 0.10
#: 逃避方向阈值（§3.4 降阶正确性锚）
D_PEAK_THRESHOLD = 0.3

#: 校准网格（≤5 轮，每轮一个组合；类级缩放 + 全局缩放）。
#: B1c3 实测（two_comp 300，CSV sha 9d80cf74…，2026-08-29）：
#:   - two_comp identity g=0.05：silent=0.817/CI=0.51（正！）但 LI=0
#:     （KC→MBON 底物未驱动，mbon_pre=0——STDP 底物 gmax=5.0·gmax_scale 太小）；
#:   - D5(s2i6/i2i3/i2m3)@g=0.05：silent=0.510/CI=0.445/CI_nograd=0.015
#:     （气味驱动 ✓）LI≈0.046（贴阈，eta=10）；
#:   - eta=12 → LI≈0.21（过阈 ✓；eta 为扫描探针值，P5 定稿学习率另定）；
#:   - 无梯度对照 CI_nograd≈0.015≪CI=0.445 → CI 为气味驱动（M4 no-gradient
#:     主判据语义）。
#: 定稿候选：gmax=0.05 + s2i6/i2i3/i2m3（two_comp），eta 取 12 过 LI 阈。
GRID = [
    dict(name="prior_base", gmax_scale=0.05, class_scales={},
         note="M5 先验恒等 + B1b 第一遍全局缩放（two_comp 对照基线）"),
    dict(name="d5_s2i8", gmax_scale=0.05,
         class_scales={("sensory", "inter"): 8.0},
         note="感官→inter 放大（嗅觉链第一级；s2i 单桶）"),
    dict(name="d5_g050", gmax_scale=0.05,
         class_scales={("sensory", "inter"): 6.0,
                       ("inter", "inter"): 3.0,
                       ("inter", "motor"): 3.0},
         note="D5 反推工作区（B1c3 实测：two_comp silent=0.510/CI=0.445/LI=0.046@eta10）"),
    dict(name="d5_g049", gmax_scale=0.049,
         class_scales={("sensory", "inter"): 6.0,
                       ("inter", "inter"): 3.0,
                       ("inter", "motor"): 3.0},
         note="稳健性邻点（gmax 0.049，工作区下沿）"),
    dict(name="d5_g051", gmax_scale=0.051,
         class_scales={("sensory", "inter"): 6.0,
                       ("inter", "inter"): 3.0,
                       ("inter", "motor"): 3.0},
         note="稳健性邻点（gmax 0.051，工作区上沿）"),
]

CSV_HEADER = [
    "combo", "gmax_scale", "class_scales", "fidelity", "plasticity", "status",
    "silent_frac", "bout_activity", "ci", "escape_direction", "escape_d_peak",
    "li", "li_mode", "mbon_rate_pre", "mbon_rate_post", "wall_s", "notes",
]


def _load_existing() -> dict:
    out = {}
    if not os.path.exists(CALIBRATION_CSV):
        return out
    with open(CALIBRATION_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(row for row in f
                                if not row.strip().startswith("#")):
            out[(r["combo"], r["plasticity"])] = r
    return out


def _save_rows(rows):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CALIBRATION_CSV, "w", newline="", encoding="utf-8") as f:
        f.write("# M8 步骤 4（D5）权重校准结果（tools/calibrate_m8_weights.py 生成）\n"
                "# 判据（预注册）：silent∈[50,90]%（G1）∧ bout≥10% ∧ CI>0 ∧ LI≥0.05\n"
                "#   （stdp 档；LI_APPEAR_THRESHOLD）；escape back（informational）。\n")
        w = csv.DictWriter(f, fieldnames=CSV_HEADER, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_HEADER})


def _fmt_class_scales(cs: dict) -> str:
    if not cs:
        return "identity(1.0)"
    return ";".join(f"{a}->{b}={v}" for (a, b), v in sorted(cs.items()))


def run_combo(entry, args, spec, plasticity="none") -> dict:
    """单个组合 × 单个可塑性档：短协议（T≤5s）五指标。

    plasticity='none'：静息/自发/趋化/逃避（行为默认档，G1 输入）；
    plasticity='stdp'：学习探针（LI，机制级判据）。
    """
    t0 = time.perf_counter()
    row = dict(combo=entry["name"], gmax_scale=entry["gmax_scale"],
               class_scales=_fmt_class_scales(entry["class_scales"]),
               fidelity=args.fidelity, plasticity=plasticity, status="pending",
               silent_frac="", bout_activity="", ci="",
               escape_direction="", escape_d_peak="",
               li="", li_mode="", mbon_rate_pre="", mbon_rate_post="",
               wall_s="", notes="")
    try:
        kw = dict(scale=args.scale, fidelity=args.fidelity, plasticity=plasticity,
                  allow_placeholder=args.smoke, seed=0,
                  connectome_poll_s=0.0,
                  nt_fallback=args.nt_fallback,
                  provisional_muscles=args.provisional_muscles,
                  annotations_path=args.annotations,
                  gmax_scale=entry["gmax_scale"],
                  class_scales=entry["class_scales"],
                  stdp_eta=args.stdp_eta)
        if args.smoke:
            kw["spec_override"] = spec
        circ = LarvaCircuit(**kw)
        if plasticity == "stdp":
            lp = circ.run_learning_probe(t_test_ms=2000.0, t_train_ms=2000.0)
            row["li"] = round(lp["li"], 4)
            row["li_mode"] = lp["li_mode"]
            row["mbon_rate_pre"] = round(lp["mbon_rate_pre"], 2)
            row["mbon_rate_post"] = round(lp["mbon_rate_post"], 2)
        else:
            rest = circ.run_resting(t_total_ms=2000.0, settle_ms=500.0)
            row["silent_frac"] = round(rest["silent_frac"], 4)
            sp = circ.run_spontaneous(t_total_ms=3000.0)
            bout = (sp["frac"].get("fwd", 0.0) + sp["frac"].get("turn", 0.0)
                    + sp["frac"].get("rev", 0.0))
            row["bout_activity"] = round(bout, 4)
            res, _ = circ.run_chemotaxis_trials(n_trials=1,
                                                t_total_ms=5000.0, seed_base=0)
            row["ci"] = round(res[0]["ci"], 4)
            esc = circ.run_escape(t_total_ms=1000.0)
            row["escape_direction"] = esc["direction"]
            row["escape_d_peak"] = round(esc["d_peak"], 4)
        row["status"] = "done"
    except Exception as exc:  # noqa: BLE001 —— 组合失败如实记录，不静默
        row["status"] = "failed"
        row["notes"] = f"FAIL: {type(exc).__name__}: {str(exc)[:300]}"
    row["wall_s"] = round(time.perf_counter() - t0, 1)
    return row


def _pass(rows: dict, entry_name: str) -> bool:
    none_r = rows.get((entry_name, "none"))
    stdp_r = rows.get((entry_name, "stdp"))
    if none_r is None or stdp_r is None:
        return False
    if none_r["status"] != "done" or stdp_r["status"] != "done":
        return False
    try:
        silent = float(none_r["silent_frac"])
        bout = float(none_r["bout_activity"])
        ci = float(none_r["ci"])
        li = float(stdp_r["li"])
    except (TypeError, ValueError):
        return False
    return (SILENT_BAND[0] <= silent <= SILENT_BAND[1]
            and bout >= BOUT_FLOOR and ci > 0.0
            and li >= LI_APPEAR_THRESHOLD)


def decide(rows: dict) -> Optional[dict]:
    """确定性选择：通过组合中 silent 最接近带中值（0.7）者。"""
    cands = [e for e in GRID if _pass(rows, e["name"])]
    if not cands:
        return None
    best = min(cands, key=lambda e: abs(
        float(rows[(e["name"], "none")]["silent_frac"]) - 0.7))
    return best


def append_weight_rows(entry: dict, args) -> str:
    """定稿权重 → data/m8_larva_params.csv（role=weight 行追加，不动既有行）。

    写：gmax_scale（全局）+ class_scale_<pre>_<post>（类级 s_k）。
    """
    lines = [
        f"weight,gmax_scale,,,,,,,,{entry['gmax_scale']},D5 定稿全局突触电导缩放（B1c3 校准：two_comp 档 CI>0 正趋化工作区；point 档 CI 结构性为 0——嗅觉链亚阈值）",
        f"weight,fidelity_calib,,,,,,,,{args.fidelity},D5 定稿校准保真度（G0 定稿 two_comp）",
        f"weight,stdp_eta_scan,,,,,,,,{args.stdp_eta},D5 扫描探针 STDP 学习率（P5 全协议定稿学习率由学习验证节点定稿）",
    ]
    for (pre, post), v in sorted(entry["class_scales"].items()):
        lines.append(
            f"weight,class_scale_{pre}_{post},,,,,,,,{v},D5 定稿类级缩放 s_k（{pre}→{post}；w_ij=w0_class·s_k·gmax_scale）")
    for k, v in [("sensory", "sensory"), ("sensory", "motor"),
                 ("inter", "sensory"), ("motor", "inter"),
                 ("motor", "motor"), ("motor", "sensory")]:
        if (k, v) not in entry["class_scales"]:
            lines.append(
                f"weight,class_scale_{k}_{v},,,,,,,,1.0,D5 定稿类级缩放 s_k（{k}→{v}；先验恒等）")
    with open(LARVA_PARAMS_CSV, "a", newline="", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return LARVA_PARAMS_CSV


def plot_calibration(rows, best: Optional[dict]):
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
    names = [e["name"] for e in GRID]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    def _val(name, plastic, field):
        r = next((x for x in done if x["combo"] == name
                  and x["plasticity"] == plastic), None)
        if r is None or r.get(field) in ("", "nan"):
            return None
        try:
            return float(r[field])
        except (TypeError, ValueError):
            return None

    # 1) silent + CI（none 档）
    ax = axes[0]
    sil = [_val(n, "none", "silent_frac") for n in names]
    ci = [_val(n, "none", "ci") for n in names]
    x = np.arange(len(names))
    ax.plot(x, sil, "o-", color="tab:green", label="静默比例（G1 带 [50,90]%）")
    ax.axhspan(*SILENT_BAND, color="green", alpha=0.12)
    ax2 = ax.twinx()
    ax2.plot(x, ci, "s-", color="tab:blue", label="CI（趋化短协议）")
    ax2.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, fontsize=8)
    ax.set_ylabel("静默比例")
    ax2.set_ylabel("CI")
    ax.set_title("D5 校准：静默（G1）与趋化 CI（none 档）")
    ax.grid(True, alpha=0.3)

    # 2) LI（stdp 档）
    ax = axes[1]
    li = [_val(n, "stdp", "li") for n in names]
    ax.plot(x, li, "o-", color="tab:red", label="LI（学习探针，stdp 档）")
    ax.axhline(LI_APPEAR_THRESHOLD, color="gray", ls="--", lw=1,
               label=f"LI 出现阈值 {LI_APPEAR_THRESHOLD}")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, fontsize=8)
    ax.set_ylabel("LI")
    ax.set_title("D5 校准：学习指数 LI（stdp 档，机制级）")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3) 逃避 D_peak（none 档）
    ax = axes[2]
    dp = [_val(n, "none", "escape_d_peak") for n in names]
    ax.plot(x, dp, "^-", color="tab:orange", label="D_peak = max(C_back−C_fwd)")
    ax.axhline(D_PEAK_THRESHOLD, color="r", ls="--", lw=1,
               label=f"阈值 {D_PEAK_THRESHOLD}")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, fontsize=8)
    ax.set_ylabel("D_peak")
    ax.set_title("降阶正确性：痛觉逃避方向（none 档）")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    title = "M8 D5 权重校准（≤5 轮；300 档短协议 T≤5s，two_comp 档）"
    if best is not None:
        title += f" —— 定稿: {best['name']}"
    else:
        title += " —— 无组合通过 → 反证记录 + 三态裁决"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(CALIBRATION_PNG, dpi=130)
    plt.close(fig)
    return CALIBRATION_PNG


def main():
    ap = argparse.ArgumentParser(description="M8 D5 权重校准（类级缩放 s_k 扫描）")
    ap.add_argument("--rounds", type=int, default=5,
                    help="扫描轮数上限（每轮一个组合；默认 5）")
    ap.add_argument("--combo", default=None,
                    help="只跑指定组合（逗号分隔 name；默认全部 GRID）")
    ap.add_argument("--scale", type=int, default=300,
                    help="校准规模档（300 档短协议；D5 先 300 档确认）")
    ap.add_argument("--fidelity", default="two_comp", choices=["point", "two_comp"],
                    help="校准保真度档（G0 定稿 two_comp——B1c3 实测 point 档 "
                         "CI 结构性为 0，嗅觉链亚阈值；默认 two_comp）")
    ap.add_argument("--stdp-eta", type=float, default=12.0,
                    help="扫描探针 STDP 学习率（B1c3 实测：eta=10 → LI≈0.046 "
                         "贴阈、eta=12 → LI≈0.21；P5 全协议定稿学习率另定）")
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟：占位连接组（仅机制，不出真实决策）")
    ap.add_argument("--wait", action="store_true",
                    help="轮询等待 B1a m8_larva_connectome.csv（G2 数据门）")
    ap.add_argument("--nt-fallback", default="class", choices=["class"],
                    help="递质临时回退（B1a 标注不完整；PROVISIONAL_NT）")
    ap.add_argument("--provisional-muscles", action="store_true", default=True,
                    help="运动池→通道临时映射（PROVISIONAL_MUSCLE；幼虫脑连接组"
                         "无肌肉行——B1a 数据缺口，默认开，与复现入口"
                         "scan_m8_scaling --provisional-muscles 一致）")
    ap.add_argument("--annotations", default=ANNOT_DEFAULT,
                    help="B1a raw 功能注解 CSV（olfactory/noci 标签）")
    args = ap.parse_args()

    if args.smoke and args.wait:
        print("--smoke 与 --wait 互斥")
        sys.exit(2)
    spec = None
    if args.smoke:
        spec = build_placeholder_spec(300)
        print(f"[smoke] 占位连接组 {spec.n_neurons} 神经元（仅机制冒烟）",
              flush=True)
    else:
        path = wait_for_csv(None, timeout_s=3600.0)
        print(f"[data] m8_larva_connectome.csv 就绪：{path}", flush=True)
    if not os.path.exists(args.annotations):
        print(f"[warn] 功能注解缺失：{args.annotations}（无 olfactory/noci 标签）",
              flush=True)

    selected = GRID[:max(1, min(args.rounds, len(GRID)))]
    if args.combo:
        want = {c.strip() for c in args.combo.split(",")}
        selected = [e for e in GRID if e["name"] in want]
    if not selected:
        print("无选中组合")
        sys.exit(1)

    existing = _load_existing()
    rows = list(existing.values())
    for entry in selected:
        for plastic in ("none", "stdp"):
            key = (entry["name"], plastic)
            if key in existing and existing[key]["status"] == "done":
                print(f"[reuse] {entry['name']}/{plastic} 已落盘", flush=True)
                continue
            print(f"[run] {entry['name']}/{plastic} "
                  f"(gmax={entry['gmax_scale']}) 开始 "
                  f"{time.strftime('%H:%M:%S')}", flush=True)
            row = run_combo(entry, args, spec, plasticity=plastic)
            row["notes"] = (row.get("notes", "") + entry["note"]).strip()
            rows.append(row)
            _save_rows(rows)
            print(f"[done] {entry['name']}/{plastic}: status={row['status']} "
                  f"silent={row['silent_frac']} CI={row['ci']} LI={row['li']}",
                  flush=True)

    rows_by = {(r["combo"], r["plasticity"]): r for r in rows}
    best = decide(rows_by)
    png = plot_calibration(rows, best)
    print(f"PNG: {png}", flush=True)

    if best is not None:
        csv_path = append_weight_rows(best, args)
        print(f"[D5 定稿] {best['name']}: gmax={best['gmax_scale']} "
              f"class={_fmt_class_scales(best['class_scales'])} → {csv_path}",
              flush=True)
        print("完成标准：CI>0 ✓  LI≥0.05 ✓  silent∈[50,90]% ✓  bout≥10% ✓",
              flush=True)
    else:
        # 反证记录（§0.4 反证路径 + 三态裁决，不静默）
        note = ("D5 校准无组合通过：300 档（nt_fallback=class，无真实 GABA 标注）"
                "下 CI/LI 不可同时转正/出现——"
                "反证：① gmax 过低 → 嗅觉链亚阈值（LI=0）；② gmax 过高 → "
                "全网络饱和静默出带 + CI 转负；③ 工作区（gmax≈0.06+s2i6/i2i3/i2m3）"
                "三指标同时成立时方通过——若本运行全部失败，按三态裁决："
                "（a）接受 B1a 无 GABA 标注限制记录为缺失机制；（b）请求 B1a 补齐"
                "递质标注后重校准；（c）降级为仅 G1 结构交付 + 行为判据反证。"
                "（M8 清单 §0.4/§6 校准失败处置，不静默）")
        with open(CALIBRATION_CSV, "a", newline="", encoding="utf-8") as f:
            f.write(f"# {note}\n")
        with open(os.path.join(DATA_DIR, "m8_calibration_FAIL.md"), "w",
                  encoding="utf-8") as f:
            f.write("# M8 D5 权重校准失败记录（反证）\n\n" + note + "\n")
        print("[D5 FAIL] 无组合通过 → 反证记录 + 三态裁决请求（不静默）",
              flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
