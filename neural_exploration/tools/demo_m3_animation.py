"""M3 成果展示：触觉反射弧动画演示（最小行为回路——感觉→中间→运动→肌肉）。

真实运行一次 Brian2 反射弧（确定性 p=1，同 m3_report P1 判定基准），
把整条链的动力学做成 GIF 动画：
  · 左上拓扑图：触刺激注入 PLM → AMPA 兴奋链 PLM→AVM→DA（后退）
    + AVM→VB GABA 抑制（互斥方向）+ 张力维持前进基线；
    发放时节点发光，肌肉条随 C_back/C_fwd 实时收缩；
  · 中：各级 soma 膜电位（错位叠加）+ node3 发放标记；
  · 下：C_back/C_fwd 收缩曲线 + D = C_back − C_fwd 方向判定
    （D_peak > 0.3 → 后退，P1 判据）。

不是示意图：每一帧都来自真实仿真轨迹（reports/neuro/m3_animation.gif）。

用法：
  .venv-neuro/bin/python -m neural_exploration.tools.demo_m3_animation
    [--intensity 1.0] [--t-lo 46] [--t-hi 104] [--step-ms 0.5]
    [--fps 24] [--dpi 100] [--out reports/neuro/m3_animation.gif]
    [--no-gif]        # 只跑仿真 + 打印级联摘要，不写动画
    [--t-total-ms N]  # 覆盖仿真总时长（默认取 CSV 150ms）
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.src.reflex_arc import ReflexArc  # noqa: E402

REPORTS_DIR = os.path.join(ROOT, "neural_exploration", "reports", "neuro")
DEFAULT_OUT = os.path.join(REPORTS_DIR, "m3_animation.gif")

# ---- 配色（与 reflex_arc.plot_reflex / 验证图一致）----
COLORS = {"PLM": "#1f77b4", "AVM": "#ff7f0e", "DA": "#d62728", "VB": "#9467bd"}
C_BACK_C, C_FWD_C = "#2ca02c", "#8c564b"
GLOW_C = "#ffd700"
TOUCH_C = "#ff7f0e"
D_THRESHOLD = 0.3  # P1 判据：D_peak > 0.3 → 后退

# 拓扑图节点坐标（数据坐标 0..1）
NODE_POS = {"PLM": (0.10, 0.66), "AVM": (0.40, 0.66),
            "DA": (0.70, 0.82), "VB": (0.70, 0.44)}
NODE_R = 0.052
BAR_X, BAR_W = 0.905, 0.042
BACK_BAR = {"y0": 0.60, "h": 0.42}    # C_back 条区域 [y0, y0+h]
FWD_BAR = {"y0": 0.28, "h": 0.42}     # C_fwd 条区域


def _setup_fonts():
    import matplotlib
    matplotlib.rcParams["font.sans-serif"] = [
        "PingFang HK", "Hiragino Sans GB", "Arial Unicode MS", "Heiti TC",
        "Songti SC", "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["font.size"] = 9


def _draw_schematic(ax, params):
    """静态拓扑：节点 + 连接 + 触刺激盒 + 张力注记；返回动画句柄字典。"""
    from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")

    nodes, glows = {}, {}
    for role, (x, y) in NODE_POS.items():
        glow = Circle((x, y), NODE_R * 1.5, fc=GLOW_C, alpha=0.0, zorder=1)
        ax.add_patch(glow)
        body = Circle((x, y), NODE_R, fc=COLORS[role], ec="k", lw=1.0,
                      zorder=2)
        ax.add_patch(body)
        ax.text(x, y, role, ha="center", va="center", color="white",
                fontsize=9, fontweight="bold", zorder=3)
        nodes[role] = body
        glows[role] = glow

    def arrow(p0, p1, color="#555555", lw=1.4):
        a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13,
                            color=color, lw=lw, zorder=1)
        ax.add_patch(a)
        return a

    # 化学突触（node3 → soma 约定；label 用定稿值）
    arrow((0.155, 0.66), (0.348, 0.66))           # PLM → AVM
    ax.text(0.25, 0.745, "AMPA 谷氨酸 5.0nS", ha="center", fontsize=7.5,
            color="#555555")
    arrow((0.452, 0.700), (0.648, 0.800))         # AVM → DA
    ax.text(0.545, 0.855, "AMPA 5.0nS", ha="center", fontsize=7.5,
            color="#555555")
    arrow((0.452, 0.620), (0.648, 0.485))         # AVM → VB（GABA 抑制）
    ax.text(0.545, 0.515, "GABA 抑制 15nS", ha="center", fontsize=7.5,
            color="#555555")
    # 肌肉驱动
    arrow((0.752, 0.820), (0.900, 0.845), lw=1.2)
    ax.text(0.828, 0.905, "w=0.60", ha="center", fontsize=7.5, color="#2ca02c")
    arrow((0.752, 0.440), (0.900, 0.410), lw=1.2)
    ax.text(0.828, 0.375, "w=0.18", ha="center", fontsize=7.5, color="#8c564b")

    # 触刺激盒（PLM 左侧，动画中闪烁）
    touch_box = Rectangle((0.012, 0.600), 0.052, 0.12, fc=TOUCH_C, alpha=0.0,
                          ec="k", lw=0.8, zorder=1)
    ax.add_patch(touch_box)
    ax.text(0.038, 0.66, "I0 触刺激", ha="center", va="center", fontsize=7,
            color="#333333", zorder=3)
    arrow((0.064, 0.66), (0.080, 0.66), color=TOUCH_C, lw=1.6)

    # 张力注记（VB 下方）
    ax.annotate("张力注入 14µA/cm²\n维持前进基线 ≈0.2",
                xy=(0.700, 0.352), xytext=(0.700, 0.235),
                ha="center", fontsize=7, color="#666666",
                arrowprops=dict(arrowstyle="->", color="#666666", lw=0.9,
                                shrinkA=0, shrinkB=2))

    # 肌肉条（收缩动画：高度 ∝ C）
    bar_back = Rectangle((BAR_X, BACK_BAR["y0"]), BAR_W, 0.0,
                         fc=C_BACK_C, ec="k", lw=0.8, zorder=2)
    bar_fwd = Rectangle((BAR_X, FWD_BAR["y0"]), BAR_W, 0.0,
                        fc=C_FWD_C, ec="k", lw=0.8, zorder=2)
    ax.add_patch(bar_back)
    ax.add_patch(bar_fwd)
    ax.text(BAR_X + BAR_W / 2, 1.05, "C_back\n后退收缩", ha="center",
            fontsize=7.5, color=C_BACK_C)
    ax.text(BAR_X + BAR_W / 2, 0.18, "C_fwd\n前进收缩", ha="center",
            fontsize=7.5, color=C_FWD_C)
    lab_back = ax.text(0.955, 0.83, "C_back=0.000", fontsize=7, color=C_BACK_C)
    lab_fwd = ax.text(0.955, 0.47, "C_fwd=0.000", fontsize=7, color=C_FWD_C)

    # 图例注：递质极性说明
    ax.text(0.50, 0.075,
            "White 1986 语义简化：感觉→中间谷氨酸兴奋 · 中间→前进 GABA 抑制（互斥方向）"
            " · 肌肉 δ 驱动 dC/dt = −C/τ + Σw·δ",
            ha="center", fontsize=7, color="#888888")

    return dict(nodes=nodes, glows=glows, touch_box=touch_box,
                bar_back=bar_back, bar_fwd=bar_fwd,
                lab_back=lab_back, lab_fwd=lab_fwd,
                touch_start=params.touch.start_ms, touch_end=params.touch.start_ms
                + params.touch.dur_ms)


def _draw_voltage(ax, t_lo, t_hi, result, roles):
    """膜电位错位叠加 + node3 发放标记；返回 (光标, 每条曲线的标签)。"""
    offsets = {"PLM": 0.0, "AVM": 130.0, "DA": 260.0, "VB": 390.0}
    for role in roles:
        lab = f"{role.lower()}_soma"
        off = offsets[role]
        ax.plot(result.t_ms, result.v_mv[lab] + off, lw=0.8, color=COLORS[role])
        ax.text(t_hi - 1.2, off + 42, role, fontsize=9, color=COLORS[role],
                fontweight="bold")
        for s in result.spikes(role, "node3"):
            if t_lo <= s <= t_hi:
                ax.plot([s], [off + 68], marker="v", ms=5, color=COLORS[role],
                        alpha=0.8)
    ax.set_xlim(t_lo, t_hi)
    ax.set_ylim(-80, 540)
    ax.set_ylabel("V soma (mV, 错位叠加)", fontsize=8)
    ax.grid(alpha=0.3)
    cursor = ax.axvline(t_lo, color="#333333", lw=1.0, alpha=0.8)
    return cursor


def _draw_muscle(ax_d, t_lo, t_hi, result):
    """肌肉收缩 + D 方向判定（twin 轴）；返回光标/星标/判定文本等句柄。"""
    t, cb, cf = result.t_ms, result.c_back, result.c_fwd
    d = cb - cf
    ax_d.plot(t, cb, lw=1.6, color=C_BACK_C, label="C_back 后退收缩 (DA)")
    ax_d.plot(t, cf, lw=1.6, color=C_FWD_C, label="C_fwd 前进收缩 (VB)")
    ax_d.fill_between(t, cb, cf, where=cb >= cf, color=C_BACK_C, alpha=0.12)
    ax_d.fill_between(t, cb, cf, where=cb < cf, color=C_FWD_C, alpha=0.12)
    ax_d.axhline(D_THRESHOLD, color="#d62728", ls="--", lw=1.0)
    ax_d.text(t_hi - 1.5, D_THRESHOLD + 0.02, "P1 判据 D_peak > 0.3",
              fontsize=7, color="#d62728", ha="right")
    touch = result.meta.get("touch_start_ms", 50.0)
    dur = result.meta.get("touch_dur_ms", 5.0)
    ax_d.axvspan(touch, touch + dur, color=TOUCH_C, alpha=0.14)
    ax_d.set_xlim(t_lo, t_hi)
    ax_d.set_ylim(-0.05, 1.05)
    ax_d.set_ylabel("肌肉收缩 C", fontsize=8)
    ax_d.grid(alpha=0.3)
    ax_d.legend(loc="upper left", fontsize=7.5, ncol=2)

    ax_d2 = ax_d.twinx()
    (dline,) = ax_d2.plot(t, d, lw=1.3, color="#d62728", alpha=0.9,
                          label="D = C_back − C_fwd")
    ax_d2.set_ylim(-0.45, 1.1)
    ax_d2.set_ylabel("D = C_back − C_fwd", fontsize=8, color="#d62728")
    ax_d2.tick_params(axis="y", labelcolor="#d62728")
    ax_d2.legend(loc="upper right", fontsize=7.5)

    i_peak = int(np.argmax(d))
    t_peak, d_peak = t[i_peak], d[i_peak]
    star, = ax_d2.plot([], [], marker="*", ms=15, color="#d62728", zorder=5)
    verdict = ax_d.text(0.5, 0.97, "等待方向判定…", transform=ax_d.transAxes,
                        ha="center", va="top", fontsize=10, color="#666666")
    status = ax_d.text(0.01, 0.06, "", transform=ax_d.transAxes,
                       ha="left", va="bottom", fontsize=8, color="#333333",
                       family="monospace")
    cursor = ax_d.axvline(t_lo, color="#333333", lw=1.0, alpha=0.8)

    return dict(cursor=cursor, star=star, verdict=verdict, status=status,
                t_peak=t_peak, d_peak=d_peak, dline=dline)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--intensity", type=float, default=1.0)
    ap.add_argument("--t-lo", type=float, default=46.0)
    ap.add_argument("--t-hi", type=float, default=104.0)
    ap.add_argument("--step-ms", type=float, default=0.5)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--t-total-ms", type=float, default=None)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args()

    _setup_fonts()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    # ---- 1) 真实仿真（确定性 p=1，同 P1 判定基准）----
    t0 = time.time()
    print(f"[demo] 运行 Brian2 反射弧仿真 intensity={args.intensity} ...")
    arc = ReflexArc()
    r = arc.run(intensity=args.intensity, t_total_ms=args.t_total_ms)
    print(f"[demo] 仿真完成：{time.time() - t0:.1f}s，"
          f"t ∈ [{r.t_ms[0]:.0f}, {r.t_ms[-1]:.0f}] ms，{len(r.t_ms)} 步")

    # ---- 2) 控制台故事线（级联/潜伏期/方向判定）----
    cascade = {role: r.spikes(role, "node3") for role in ("PLM", "AVM", "DA", "VB")}
    print("\n=== M3 触觉反射弧 演示故事线 ===")
    meta = r.meta
    print(f"触刺激: I0={meta['i0_uA_cm2']:.0f}µA/cm² × {meta['touch_dur_ms']:.0f}ms "
          f"@ t={meta['touch_start_ms']:.0f}ms → PLM 树突端 ({meta['touch_site']})")
    prev_t = meta["touch_start_ms"]
    for role in ("PLM", "AVM", "DA"):
        sp = cascade[role]
        if len(sp):
            delta = sp[0] - prev_t
            print(f"  {role:>3s} 首发放 @ {sp[0]:7.2f} ms  "
                  f"({delta:+.2f} ms，共 {len(sp)} 发放)  [{COLORS[role]}]")
            prev_t = sp[0]
        else:
            print(f"  {role:>3s} 无发放")
    vb_sp = cascade["VB"]
    win_lo, win_hi = meta["touch_start_ms"], meta["touch_start_ms"] + 40.0
    n_win = int(np.sum((vb_sp >= win_lo) & (vb_sp <= win_hi)))
    print(f"  VB  张力基线发放 ~60Hz（共 {len(vb_sp)} 次；响应窗 "
          f"[{win_lo:.0f},{win_hi:.0f}]ms 内 {n_win} 次 —— AVM→VB GABA 抑制"
          f"使其跳过 1 个发放）")
    lat = (cascade["DA"][0] - meta["touch_start_ms"]) if len(cascade["DA"]) else None
    if lat is not None:
        print(f"  神经潜伏期（触刺激→DA 首发放）= {lat:.2f} ms"
              f"（NEURON 参考 10.10ms，误差 {(lat-10.10)/10.10*100:.1f}%）")
    base_mask = (r.t_ms >= 10.0) & (r.t_ms < meta["touch_start_ms"])
    c_fwd_base = float(np.median(r.c_fwd[base_mask])) if np.any(base_mask) else 0.0
    print(f"  肌肉: C_back 峰 = {r.c_back_peak:.3f}   "
          f"C_fwd 稳态基线 ≈ {c_fwd_base:.3f} / 峰 {r.c_fwd_peak:.3f}")
    verdict = "后退（通过）" if r.d_peak > D_THRESHOLD else "未达判据（不通过）"
    print(f"  D_peak = {r.d_peak:.3f}  >  {D_THRESHOLD}  →  判定: {verdict}")

    if args.no_gif:
        print("\n[--no-gif] 跳过动画生成。")
        return 0

    # ---- 3) 动画帧 ---- 
    frames = np.arange(args.t_lo, args.t_hi + 1e-9, args.step_ms)
    idxs = np.searchsorted(r.t_ms, frames, side="left")

    fig = plt.figure(figsize=(12.5, 8.2), dpi=args.dpi)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.05, 1.15, 1.0], hspace=0.32)
    ax_schem = fig.add_subplot(gs[0])
    ax_v = fig.add_subplot(gs[1])
    ax_m = fig.add_subplot(gs[2])

    fig.suptitle("M3 最小行为回路 · 触觉反射弧 —— "
                 "触刺激 → 感觉(PLM) → 中间(AVM) → 运动(DA/VB) → 肌肉",
                 fontsize=13, fontweight="bold")
    ax_schem.set_title("① 网络拓扑（4 神经元 + 2 肌肉双通道 · 参数唯一定稿源 "
                       "data/m3_reflex_params.csv）", fontsize=9, loc="left")
    ax_v.set_title("② 各级 soma 膜电位 V(t)（错位叠加 · ▼ 为 node3 发放时刻）",
                   fontsize=9, loc="left")
    ax_m.set_title("③ 肌肉收缩 C(t) 与方向判定 D(t)（基准强度 I0 单次运行，确定性 p=1）",
                   fontsize=9, loc="left")

    sch = _draw_schematic(ax_schem, arc.params)
    v_cur = _draw_voltage(ax_v, args.t_lo, args.t_hi, r, ("PLM", "AVM", "DA", "VB"))
    mus = _draw_muscle(ax_m, args.t_lo, args.t_hi, r)

    # 帧内最近时刻的发放归属（供拓扑图发光）
    spike_windows = {role: (sp, sp + 2.5) for role, sp in cascade.items()}

    def update(frame_t, idx):
        t_now = frame_t
        # 膜电位 / 肌肉光标
        v_cur.set_xdata([t_now, t_now])
        mus["cursor"].set_xdata([t_now, t_now])
        # 拓扑图：节点发光（3ms 窗口）
        for role, (win_start, win_end) in spike_windows.items():
            active = np.any((win_start <= t_now) & (t_now <= win_end))
            sch["glows"][role].set_alpha(0.75 if active else 0.0)
        # 触刺激闪烁
        if sch["touch_start"] <= t_now <= sch["touch_end"]:
            sch["touch_box"].set_alpha(0.85)
        else:
            sch["touch_box"].set_alpha(0.0)
        # 肌肉条
        i = idx
        cb, cf = r.c_back[i], r.c_fwd[i]
        sch["bar_back"].set_height(cb * BACK_BAR["h"])
        sch["bar_back"].set_y(BACK_BAR["y0"])
        sch["bar_fwd"].set_height(cf * FWD_BAR["h"])
        sch["bar_fwd"].set_y(FWD_BAR["y0"])
        sch["lab_back"].set_text(f"C_back={cb:.3f}")
        sch["lab_fwd"].set_text(f"C_fwd={cf:.3f}")
        # 状态读数
        d_now = cb - cf
        mus["status"].set_text(
            f"t={t_now:6.1f}ms  C_back={cb:.3f}  C_fwd={cf:.3f}  D={d_now:+.3f}")
        # 方向判定：到达 D_peak 时刻 → 打星 + 判语
        if t_now >= mus["t_peak"]:
            mus["star"].set_data([mus["t_peak"]], [mus["d_peak"]])
            ok = mus["d_peak"] > D_THRESHOLD
            mus["verdict"].set_text(
                f"D_peak = {mus['d_peak']:.3f} > {D_THRESHOLD} → 判定：后退（通过）"
                if ok else f"D_peak = {mus['d_peak']:.3f} ≤ {D_THRESHOLD} → 判定：不后退（未达标）")
            mus["verdict"].set_color("#2ca02c" if ok else "#d62728")
            mus["verdict"].set_fontweight("bold")
        return (v_cur, mus["cursor"], sch["glows"]["PLM"], sch["glows"]["AVM"],
                sch["glows"]["DA"], sch["glows"]["VB"], sch["touch_box"],
                sch["bar_back"], sch["bar_fwd"], sch["lab_back"], sch["lab_fwd"],
                mus["status"], mus["star"], mus["verdict"])

    anim = FuncAnimation(fig, lambda i: update(frames[i], idxs[i]),
                         frames=len(frames), interval=1000 / args.fps, blit=False)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(f"\n[demo] 渲染动画 {len(frames)} 帧 → {args.out} ...")
    t1 = time.time()
    anim.save(args.out, writer=PillowWriter(fps=args.fps))
    print(f"[demo] 动画已保存: {args.out}  ({time.time() - t1:.1f}s, "
          f"{len(frames)} 帧 @ {args.fps}fps, 时间窗 [{args.t_lo}, {args.t_hi}]ms)")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
