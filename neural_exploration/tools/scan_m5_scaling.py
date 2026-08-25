"""M5 铁律 C 缩放扫描：规模 {20,50,100,302} × 保真度 {点, 双隔室, HH} → 行为指标 + 墙钟。

G0 门组件（M5 清单 §3.3/§3.4/§3.5，第一关键决策步）：
  - 每格点：探针（T=1s 单试次墙钟，record=[]）→ 短协议（T≤5s）行为指标
    （趋化 CI / 逃避方向与潜伏期 / 静息发放率 / 自发状态比例）+ 单试次墙钟；
  - 总格点 ≤ 30；预算控制：探针先行 + 长任务完成标记（M4 L25）+ 单 worker
    （Brian2 编译缓存锁竞争，M4 L21）；
  - 降阶正确性（§3.4）：20 档 CI vs M4 记录（决定性点 CI=0.099@10s、点0 CI=0.043@5s、
    参考模型 CI=0.175@5s，ΔCI≤0.15 或方向一致）；M3 反射链方向/潜伏期
    （D_peak>0.3、神经潜伏期 [5,20]ms）vs M3（0.352 / 8.23–8.93ms）；
  - 产出：data/m5_scaling.csv + reports/neuro/m5_scaling_curves.png；
  - 决策（§3.3 预注册）：行为层配置 = 使行为指标与高保真档收敛（ΔCI≤0.15、
    逃避方向一致、静默比例差<10pp）的最小规模×最低保真度；T 按参考模型
    N=20 通过率 ≥80% 的最短 T；预算 ≤ 200 CPU-小时。

确定性：p=1/n=1；试次方差来自伪随机起点（seed_base 派生）；同参数重跑逐位一致。
运行：.venv-neuro/bin/python -m neural_exploration.tools.scan_m5_scaling
  --rows 20p,20t,50p,50t,100p,302p（默认全集）--skip-heavy（跳过多隔室探针）
  长任务：本脚本每格点结果即时落盘 CSV（断点续跑），完成后 touch 完成标记。
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

from neural_exploration.src.chemotaxis_env import ci_group_stats  # noqa: E402
from neural_exploration.src.worm_circuit import (  # noqa: E402
    FIDELITY_DT,
    FIDELITY_AXIS,
    ReflexCircuit,
    WormCircuit,
    load_connectome,
    make_worm_circuit,
)

DATA_DIR = os.path.join(ROOT, "neural_exploration", "data")
REPORTS_DIR = os.path.join(ROOT, "neural_exploration", "reports", "neuro")
SCALING_CSV = os.path.join(DATA_DIR, "m5_scaling.csv")
SCALING_PNG = os.path.join(REPORTS_DIR, "m5_scaling_curves.png")
DONE_MARKER = os.path.join(DATA_DIR, ".m5_scaling_DONE")

#: 参考记录（M4/M3 冻结基线，§3.4 对照；来源见 m4_env_notes L22a/L24c、m3_p*_*.csv）
M4_CI_BRIAN2_5S = 0.043        # M4 点0：T=5s/N=10（L22a）
M4_CI_BRIAN2_10S = 0.099       # M4 决定性点：T=10s/N=20（L24c）
M4_CI_REF_5S = 0.175           # 参考模型 T=5s（ref-T5000 行）
M3_D_PEAK = 0.352              # m3_p1_direction.csv（>0.3 → back）
M3_LAT_LO, M3_LAT_HI = 5.0, 20.0   # 神经潜伏期窗（M3 P3 / M5 P5）
D_PEAK_THRESHOLD = 0.3

#: 网格（scale, fidelity, source, 说明, 协议覆盖）——总格点 7（≤30 预算）。
#: component 模式（two_comp）比 grouped 慢 ~15×（L7 实测：20 档 T=0.5s=25s vs 3s）
#: → two_comp 档用短协议；100-two_comp 预算不可行（~30min 编译 + 10min/T1s）→ skipped。
GRID = [
    (20, "point", "connectome", "M4 趋化子图（连接组真实接线，grouped）", {}),
    (20, "two_comp", "connectome", "M4 趋化子图 + 双隔室（component）",
     {"chem_t_ms": 5000.0, "chem_n": 2}),
    (20, "multicomp", "m4_fallback", "M4 子图 HH（重档复用 M4 记录）", {}),
    (50, "point", "connectome", "命令/运动上下文扩展（grouped）", {}),
    (50, "two_comp", "connectome", "50 双隔室（component）",
     {"chem_t_ms": 2000.0, "chem_n": 2}),
    (100, "point", "connectome", "100 规模（grouped）", {}),
    (302, "point", "connectome", "全连接组（grouped；冷编译 ~10min）", {}),
]

CSV_HEADER = [
    "scale", "fidelity", "source", "dt_ms", "method", "status",
    "n_neurons", "n_chem", "n_gap", "n_muscle",
    "build_wall_s", "probe_T_ms", "probe_wall_s",
    "ci_mean", "ci_sem", "ci_n", "ci_control_mean",
    "ci_delta_vs_m4_brian2_5s", "ci_delta_vs_m4_ref_5s", "ci_direction_same",
    "escape_direction", "escape_d_peak", "escape_lat_ms", "escape_lat_ok",
    "resting_median_hz", "resting_silent_frac", "resting_max_hz",
    "spont_fwd", "spont_rev", "spont_turn", "spont_pause",
    "chem_wall_s_mean", "total_wall_s", "notes",
]


def _row_defaults(scale, fidelity, source, status="pending"):
    dt, method = FIDELITY_DT[fidelity]
    return dict(scale=scale, fidelity=fidelity, source=source, dt_ms=dt,
                method=method, status=status, n_neurons=0, n_chem=0, n_gap=0,
                n_muscle=0, build_wall_s=float("nan"), probe_T_ms=1000.0,
                probe_wall_s=float("nan"), ci_mean=float("nan"),
                ci_sem=float("nan"), ci_n=0, ci_control_mean=float("nan"),
                ci_delta_vs_m4_brian2_5s=float("nan"),
                ci_delta_vs_m4_ref_5s=float("nan"), ci_direction_same="",
                escape_direction="", escape_d_peak=float("nan"),
                escape_lat_ms=float("nan"), escape_lat_ok="",
                resting_median_hz=float("nan"), resting_silent_frac=float("nan"),
                resting_max_hz=float("nan"), spont_fwd=float("nan"),
                spont_rev=float("nan"), spont_turn=float("nan"),
                spont_pause=float("nan"), chem_wall_s_mean=float("nan"),
                total_wall_s=float("nan"), notes="")


def _load_existing() -> dict:
    """读回已落盘行（断点续跑：按 (scale, fidelity) 键）。"""
    out = {}
    if not os.path.exists(SCALING_CSV):
        return out
    with open(SCALING_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(row for row in f
                                if not row.strip().startswith("#")):
            out[(int(r["scale"]), r["fidelity"])] = r
    return out


def _save_rows(rows):
    with open(SCALING_CSV, "w", newline="", encoding="utf-8") as f:
        f.write("# M5 铁律 C 缩放扫描结果（tools/scan_m5_scaling.py 生成，G0 门）\n"
                "# 列语义：scale=规模轴；fidelity=保真度轴（point/two_comp/multicomp）；\n"
                "#   source=connectome（m5_connectome.csv 子集）/ m4_fallback（M4 参数源）；\n"
                "#   ci_delta_vs_m4_brian2_5s = |CI − M4 点0(0.043@5s)|；\n"
                "#   ci_delta_vs_m4_ref_5s = |CI − 参考模型(0.175@5s)|；§3.4 判据 ΔCI≤0.15 或方向一致；\n"
                "#   escape_* 来自同保真度 M3 反射子图（ReflexCircuit，方向/潜伏期 vs M3）；\n"
                "#   chem_wall_s_mean = 趋化短协议单试次墙钟（含编译后的稳态）；probe = T=1s 探针。\n")
        w = csv.DictWriter(f, fieldnames=CSV_HEADER, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_HEADER})


#: 逃避结果缓存（每保真度跑一次反射子图，跨规模复用——避免重复冷编译）
_ESCAPE_CACHE: dict = {}


def _escape_for(fidelity: str) -> dict:
    """同保真度 M3 反射子图（方向/潜伏期 vs M3）；每保真度只跑一次。"""
    if fidelity in _ESCAPE_CACHE:
        return _ESCAPE_CACHE[fidelity]
    rc = ReflexCircuit(fidelity=fidelity)
    esc = rc.run(t_total_ms=150.0)
    out = dict(
        escape_direction=esc.direction,
        escape_d_peak=round(esc.d_peak, 4),
        escape_lat_ms=round(esc.neural_latency_ms, 3),
        escape_lat_ok=(str(M3_LAT_LO <= esc.neural_latency_ms <= M3_LAT_HI)
                       if np.isfinite(esc.neural_latency_ms)
                       else "no_da_spike"),
    )
    _ESCAPE_CACHE[fidelity] = out
    return out


# --------------------------------------------------------------------- #
# 单格点协议
# --------------------------------------------------------------------- #
def run_grid_point(scale, fidelity, source, chem_t_ms=5000.0, chem_n=3,
                   control_n=2, resting_ms=1000.0, spont_ms=5000.0,
                   run_multicomp_probe=False) -> dict:
    row = _row_defaults(scale, fidelity, source)
    t0 = time.perf_counter()
    try:
        if source == "m4_fallback":
            # HH 档 = M4 子图重实现（component，M4 参数源强制）；重档复用 M4 记录
            from neural_exploration.src.worm_circuit import DEFAULT_M4_PARAMS_CSV
            circ = WormCircuit(scale=20, fidelity=fidelity,
                               csv_path=DEFAULT_M4_PARAMS_CSV,
                               connectome_poll_s=0.0, gap_mode="component")
            _ = circ.make_session(t_total_ms=1000.0)
            if run_multicomp_probe:
                row["probe_wall_s"] = circ.run_resting(t_total_ms=1000.0)["wall_s"]
                row["notes"] += "multicomp 探针实测; "
            # M4 记录（复用，不重跑闭环——M4 L23/L25 预算纪律）
            row["ci_mean"] = M4_CI_BRIAN2_5S
            row["ci_n"] = 10
            row["ci_control_mean"] = -0.0725
            row["ci_delta_vs_m4_brian2_5s"] = 0.0
            row["ci_delta_vs_m4_ref_5s"] = abs(M4_CI_BRIAN2_5S - M4_CI_REF_5S)
            row["ci_direction_same"] = "reuse_m4_record"
            row["n_neurons"] = circ.sub.n_neurons
            row["n_chem"] = circ.sub.n_chem
            row["n_gap"] = circ.sub.n_gap
            row["n_muscle"] = len(circ.sub.muscles)
            row["build_wall_s"] = circ._build_wall_s
            row["status"] = "done"
            row["notes"] += ("CI/对照 = M4 记录复用（L22a 点0：CI=0.043±0.199@5s/N=10；"
                             "决定性点 0.099@10s——见 m5_env_notes L8）；")
            return row

        circ = make_worm_circuit(scale=scale, fidelity=fidelity)
        # ---- 探针（T=1s 单试次；编译后的稳态墙钟）----
        probe_res, probe_meta = circ.run_chemotaxis_trials(
            n_trials=1, t_total_ms=1000.0, seed_base=9000)
        row["probe_wall_s"] = probe_meta["wall_s"][0]
        row["build_wall_s"] = circ._build_wall_s
        row["n_neurons"] = circ.sub.n_neurons
        row["n_chem"] = circ.sub.n_chem
        row["n_gap"] = circ.sub.n_gap
        row["n_muscle"] = len(circ.sub.muscles)

        # ---- 趋化短协议（T=5s × N，确定性）----
        res, meta = circ.run_chemotaxis_trials(n_trials=chem_n,
                                               t_total_ms=chem_t_ms,
                                               seed_base=0)
        ci_vals = np.array([r.ci for r in res], dtype=float)
        row["ci_mean"] = float(np.mean(ci_vals))
        row["ci_sem"] = float(np.std(ci_vals, ddof=1) / math.sqrt(len(ci_vals))
                              if len(ci_vals) > 1 else float("nan"))
        row["ci_n"] = len(ci_vals)
        row["chem_wall_s_mean"] = float(np.mean(meta["wall_s"]))
        # 对照（无梯度；20 档必跑，其余档 budget 允许时）
        if control_n > 0 and scale == 20:
            ctrl_res, _ = circ.run_control(n_trials=control_n,
                                           t_total_ms=chem_t_ms, seed_base=1000)
            row["ci_control_mean"] = float(np.mean([r.ci for r in ctrl_res]))
        # ΔCI vs M4 记录（§3.4：≤0.15 或方向一致）
        d1 = abs(row["ci_mean"] - M4_CI_BRIAN2_5S)
        d2 = abs(row["ci_mean"] - M4_CI_REF_5S)
        row["ci_delta_vs_m4_brian2_5s"] = round(d1, 4)
        row["ci_delta_vs_m4_ref_5s"] = round(d2, 4)
        same_dir = bool(np.sign(row["ci_mean"]) == np.sign(M4_CI_REF_5S)) \
            or abs(row["ci_mean"]) < 0.05
        row["ci_direction_same"] = str(same_dir)

        # ---- 逃避（同保真度 M3 反射子图；缓存跨规模复用）----
        row.update(_escape_for(fidelity))

        # ---- 静息（无刺激 T=1s）----
        rest = circ.run_resting(t_total_ms=resting_ms)
        row["resting_median_hz"] = round(rest["median_hz"], 3)
        row["resting_silent_frac"] = round(rest["silent_frac"], 3)
        row["resting_max_hz"] = round(rest["max_hz"], 3)
        if rest["has_nan"]:
            row["notes"] += "⚠ 静息 NaN/发散; "

        # ---- 自发状态比例（无刺激 T=5s）----
        sp = circ.run_spontaneous(t_total_ms=spont_ms)
        for k in ("fwd", "rev", "turn", "pause"):
            row[f"spont_{k}"] = round(sp["frac"].get(k, 0.0), 3)

        row["status"] = "done"
        row["total_wall_s"] = round(time.perf_counter() - t0, 1)
    except Exception as exc:  # noqa: BLE001 —— 格点失败如实记录，不静默
        row["status"] = "failed"
        row["notes"] += f"FAIL: {type(exc).__name__}: {str(exc)[:300]}"
    return row


# --------------------------------------------------------------------- #
# 出图
# --------------------------------------------------------------------- #
def plot_curves(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    # CJK 标签字体（macOS PingFang；DejaVu Sans 缺 CJK 字形 → 方框，L7）
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
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 1) CI vs 规模（按保真度分色；M4 记录参考线）
    ax = axes[0, 0]
    colors = {"point": "#1f77b4", "two_comp": "#ff7f0e", "multicomp": "#2ca02c"}
    for fid in ("point", "two_comp", "multicomp"):
        xs = [r["scale"] for r in done if r["fidelity"] == fid]
        ys = [r["ci_mean"] for r in done if r["fidelity"] == fid]
        if xs:
            ax.plot(xs, ys, "o-", color=colors[fid], label=fid)
    ax.axhline(M4_CI_REF_5S, color="gray", ls="--", lw=1,
               label=f"参考模型 @5s ({M4_CI_REF_5S})")
    ax.axhline(M4_CI_BRIAN2_5S, color="gray", ls=":", lw=1,
               label=f"M4 Brian2 @5s ({M4_CI_BRIAN2_5S})")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("规模（神经元数）")
    ax.set_ylabel("CI（趋化短协议 T=5s）")
    ax.set_title("铁律 C：行为复现度 vs 规模（按保真度分色）")
    ax.legend(fontsize=8)
    ax.set_xticks([20, 50, 100, 302])

    # 2) 墙钟 vs 规模（对数轴，双保真度）
    ax = axes[0, 1]
    for fid in ("point", "two_comp", "multicomp"):
        xs = [r["scale"] for r in done if r["fidelity"] == fid]
        ys = [r["probe_wall_s"] for r in done if r["fidelity"] == fid]
        if xs and any(np.isfinite(y) for y in ys):
            ax.plot(xs, ys, "s-", color=colors[fid],
                    label=f"{fid} 探针 T=1s")
    ax.set_yscale("log")
    ax.set_xlabel("规模（神经元数）")
    ax.set_ylabel("单试次墙钟（s，T=1s 探针，对数轴）")
    ax.set_title("性能曲线：墙钟 vs 规模")
    ax.legend(fontsize=8)
    ax.set_xticks([20, 50, 100, 302])
    ax.grid(True, which="both", alpha=0.3)

    # 3) 逃避（M3 反射子图，按保真度）
    ax = axes[1, 0]
    for fid in ("point", "two_comp", "multicomp"):
        r = next((x for x in done if x["fidelity"] == fid
                  and np.isfinite(x["escape_d_peak"])), None)
        if r is None:
            continue
        ax.bar(fid, r["escape_d_peak"], color=colors[fid], alpha=0.8,
               label=f"D_peak={r['escape_d_peak']:.2f}")
        ax.text(fid, r["escape_d_peak"] + 0.02,
                f"lat={r['escape_lat_ms']:.1f}ms", ha="center", fontsize=8)
    ax.axhline(D_PEAK_THRESHOLD, color="r", ls="--", lw=1,
               label=f"D_peak 阈值 {D_PEAK_THRESHOLD}")
    ax.axhline(M3_D_PEAK, color="gray", ls=":", lw=1, label=f"M3 记录 {M3_D_PEAK}")
    ax.set_ylabel("D_peak = max(C_back − C_fwd)")
    ax.set_title("降阶正确性：M3 反射子图方向/潜伏期（点/双隔室/M3 记录）")
    ax.legend(fontsize=8)

    # 4) 静息静默比例 vs 规模
    ax = axes[1, 1]
    for fid in ("point", "two_comp"):
        xs = [r["scale"] for r in done if r["fidelity"] == fid]
        ys = [r["resting_silent_frac"] for r in done if r["fidelity"] == fid]
        if xs:
            ax.plot(xs, ys, "^-", color=colors[fid], label=f"{fid} 静默比例")
    ax.axhline(0.7, color="gray", ls="--", lw=1,
               label="电生理带下限 70%（预注册 #4）")
    ax.set_xlabel("规模（神经元数）")
    ax.set_ylabel("静默比例（无刺激 T=1s）")
    ax.set_title("静息发放率分布：静默比例 vs 规模")
    ax.legend(fontsize=8)
    ax.set_xticks([20, 50, 100, 302])

    fig.suptitle("M5 铁律 C 缩放扫描（G0）：data/m5_scaling.csv + "
                 "docs/m5_env_notes.md G0 结论", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(SCALING_PNG, dpi=130)
    plt.close(fig)
    return SCALING_PNG


# --------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="M5 铁律 C 缩放扫描（G0）")
    ap.add_argument("--rows", default="all",
                    help="格点子集：all / 逗号分隔的 <scale><f>（如 20p,50p,302p）")
    ap.add_argument("--chem-t-ms", type=float, default=5000.0,
                    help="趋化短协议 T（默认 5000ms）")
    ap.add_argument("--chem-n", type=int, default=3, help="趋化试次数")
    ap.add_argument("--run-multicomp-probe", action="store_true",
                    help="重档：实测 multicomp T=1s 探针（~20-30min）")
    args = ap.parse_args()

    t_start = time.perf_counter()
    existing = _load_existing()
    rows = []
    selected = []
    for (scale, fid, source, note, proto) in GRID:
        if args.rows == "all" or f"{scale}{fid[0]}" in args.rows.split(","):
            selected.append((scale, fid, source, note, proto))
    if not selected:
        print("无选中格点（--rows 语法：20p,50t,302p）")
        sys.exit(1)

    for (scale, fid, source, note, proto) in selected:
        key = (scale, fid)
        if key in existing and existing[key]["status"] == "done":
            print(f"[reuse] {scale}/{fid} 已落盘，跳过", flush=True)
            rows.append(existing[key])
            continue
        print(f"[run] {scale}/{fid} ({note}) 开始 {time.strftime('%H:%M:%S')}",
              flush=True)
        row = run_grid_point(scale, fid, source,
                             chem_t_ms=proto.get("chem_t_ms", args.chem_t_ms),
                             chem_n=proto.get("chem_n", args.chem_n),
                             run_multicomp_probe=args.run_multicomp_probe)
        row["notes"] = (row.get("notes", "") + note).strip()
        rows.append(row)
        _save_rows(rows)          # 每格点落盘（断点续跑）
        print(f"[done] {scale}/{fid}: status={row['status']} "
              f"CI={row['ci_mean']} probe={row['probe_wall_s']}s "
              f"({time.perf_counter()-t_start:.0f}s elapsed)", flush=True)

    # 补齐：既有 done 行（未重跑）保留；真正缺的格点记 skipped（CSV 完整性）
    seen = {(int(r["scale"]), r["fidelity"]) for r in rows}
    for (scale, fid, source, note, _proto) in GRID:
        key = (scale, fid)
        if key in seen:
            continue
        if key in existing and existing[key]["status"] == "done":
            rows.append(existing[key])
            continue
        r = _row_defaults(scale, fid, source, status="skipped")
        r["notes"] = f"skipped: {note}"
        rows.append(r)
    _save_rows(rows)

    png = plot_curves(rows)
    print(f"CSV: {SCALING_CSV}")
    print(f"PNG: {png}")
    # 完成标记（M4 L25 纪律）
    with open(DONE_MARKER, "w") as f:
        f.write(f"scan finished {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"DONE marker: {DONE_MARKER}")


if __name__ == "__main__":
    main()
