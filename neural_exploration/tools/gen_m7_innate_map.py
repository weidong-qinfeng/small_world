"""M7 先天机制 → 数字大脑接入点映射图（M7-B2 交付物）。

输出：reports/neuro/m7_innate_map.png
图内容（三层架构 + 7 项迁移机制映射，M7 清单 D2/D3）：
  - 顶层：认知层（SymbolicInterface.solve：应用题文本 → DAG 推理链）；
  - 中层：机制层（InnateInterface 四方法：sense / actuate / adapt / gate）；
  - 底层：环境刺激（触刺激 / 气味梯度 / 时间 / 配对 / 动机）；
  - 7 项迁移机制（M-1..M-7）→ 接入点映射：
      M-1 反射   → actuate(escape)          先天运动反应
      M-2 趋化   → sense(odor) + actuate(approach)  环境感知
      M-3 CPG    → actuate(rhythm)          行为节奏
      M-4 习惯化 → adapt(n)                 先天适应
      M-5 联想   → mechanisms["associative"]（机制层可观察，不过四方法路由）
      M-6 调质   → gate(motivation) → 注入 actuate  运动增益门控
      M-7 反证   → 设计依据 → 阶段二 M8（不接入；虚线标注）
  - 消融语义：set_enabled(name, False) → 该机制贡献归零（行为差异可测）；
  - 确定性：无随机（p=1/n=1），同参数重跑逐位一致。

用法：.venv-neuro/bin/python -m neural_exploration.tools.gen_m7_innate_map
（matplotlib 在 .venv-neuro；.venv-db 无 matplotlib——如用 .venv-db 需先装）
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.font_manager as fm  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

REPORTS_DIR = os.path.join(ROOT, "neural_exploration", "reports", "neuro")
OUT_PNG = os.path.join(REPORTS_DIR, "m7_innate_map.png")

# CJK 字体候选（macOS 系统字体；TTC 取 face 0，matplotlib addfont 支持）
_CJK_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
]


def _setup_cjk_font():
    for p in _CJK_FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                fm.fontManager.addfont(p)
                name = fm.FontProperties(fname=p).get_name()
                plt.rcParams["font.family"] = "sans-serif"
                plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
                plt.rcParams["axes.unicode_minus"] = False
                print(f"CJK 字体：{name}（{p}）")
                return name
            except Exception:  # noqa: BLE001
                continue
    print("警告：未找到 CJK 字体，中文将显示为方框")
    return None

# 颜色
C_ENV = "#eaf3e1"      # 环境（浅绿）
C_MECH = "#e8f0fb"     # 机制层（浅蓝）
C_COG = "#fdf0e0"      # 认知层（浅橙）
C_EDGE = "#5b7a9d"
C_OK = "#1d7a3e"
C_REFUTE = "#b05030"
C_MUTED = "#8a8a8a"


def _box(ax, x, y, w, h, fc, title, lines, fs=9, tc="#222"):
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.05",
                       linewidth=1.4, edgecolor=C_EDGE, facecolor=fc)
    ax.add_patch(p)
    ax.text(x, y + h / 2 - 0.045, title, ha="center", va="top",
            fontsize=fs + 1, fontweight="bold", color=tc)
    for i, ln in enumerate(lines):
        ax.text(x, y - 0.045 - 0.052 * (i + 1), ln, ha="center", va="top",
                fontsize=fs - 1.5, color=tc)


def _arrow(ax, x1, y1, x2, y2, color=C_EDGE, style="-|>", lw=1.6,
           ls="-", label=None, lx=0.0, ly=0.0, fs=8.5):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                        mutation_scale=14, linewidth=lw, color=color,
                        linestyle=ls, shrinkA=2, shrinkB=2)
    ax.add_patch(a)
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label,
                fontsize=fs, color=color, ha="center", va="center")


def gen_map() -> str:
    _setup_cjk_font()
    fig, ax = plt.subplots(figsize=(12.4, 9.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # ---- 三层架构 -------------------------------------------------- #
    _box(ax, 6.0, 8.55, 11.4, 1.15, C_COG, "认知层 SymbolicInterface",
         ["solve(应用题文本) → srl_parse → DAG 构建 → call_algorithm → answer",
          "（0-20 加法已验证模板；场景叙事由机制层断言承载——D3 层位）"], fs=9.5)
    _box(ax, 6.0, 5.1, 11.4, 2.4, C_MECH, "机制层 InnateInterface（先天感知/运动底座）",
         ["sense(odor/touch)   环境感知（浓度/梯度/触强度）",
          "actuate(escape/approach/rhythm)   动作选择（回避/趋利/节律）",
          "adapt(n)   适应（重复刺激 → 响应衰减）   gate(motivation)   增益门控",
          "可观察性：calls 调用日志 + set_enabled 消融（行为差异可测）+ 确定性逐位一致"],
         fs=9.5)
    _box(ax, 6.0, 1.75, 11.4, 1.3, C_ENV, "环境刺激（M4 冻结梯度场 / M3 触刺激协议）",
         ["touch(I)   odor(x,y)   time(t, food_present)   pairing(cs,us)   motivation(m)"],
         fs=9.5)

    # ---- 7 项迁移机制（左列）→ 接入点（右列） ------------------------- #
    mech_rows = [
        ("M-1 反射弧", "actuate(escape)\n触刺激→定向回避硬连线", C_OK),
        ("M-2 趋化", "sense(odor) + actuate(approach)\n正向梯度趋利", C_OK),
        ("M-3 咽部 CPG", "actuate(rhythm)\n节律相位/频率（双带）", C_OK),
        ("M-4 习惯化", "adapt(n)\nR(n) 指数衰减（STP 消融）", C_OK),
        ("M-5 联想", "mechanisms[\"associative\"]\n机制层可观察（L16 选择性路由）", C_OK),
        ("M-6 调质", "gate(motivation) → 注入 actuate\n增益∈[floor,1.2] 单调", C_OK),
        ("M-7 反证清单", "设计依据 → 阶段二 M8\n夹带双稳态（不接入）", C_REFUTE),
    ]
    mech_x, mid_x = 3.05, 8.55
    y0, dy = 7.62, 0.72
    for i, (name, target, color) in enumerate(mech_rows):
        y = y0 - i * dy
        ax.text(mech_x, y, name, ha="center", va="center", fontsize=10,
                fontweight="bold", color=color)
        ax.plot([mech_x + 0.75, mid_x - 0.75], [y, y], color=color,
                linewidth=1.4, linestyle="-")
        ax.add_patch(FancyArrowPatch((mid_x - 0.75, y), (mid_x + 0.02, y),
                                     arrowstyle="-|>", mutation_scale=13,
                                     linewidth=1.4, color=color))
        ax.text(mid_x + 0.25, y, target, ha="left", va="center", fontsize=8.6,
                color="#333")

    # ---- 环境 → 机制 / 机制 → 认知 纵向流 ---------------------------- #
    _arrow(ax, 2.4, 2.4, 2.4, 3.9, color=C_MUTED, label="刺激→感知", lx=-0.7)
    _arrow(ax, 6.0, 2.4, 6.0, 3.9, color=C_MUTED, label="刺激→行为")
    _arrow(ax, 9.6, 2.4, 9.6, 3.9, color=C_MUTED, label="动机→门控")
    _arrow(ax, 2.0, 6.3, 2.0, 7.95, color=C_EDGE, lw=1.8)
    _arrow(ax, 10.0, 6.3, 10.0, 7.95, color=C_EDGE, lw=1.8)
    ax.text(2.0, 7.1, "感知/行为量\n（D_peak·CI·R(n)·gain）", fontsize=8,
            color=C_EDGE, ha="center")
    ax.text(10.0, 7.1, "行为语义供认知层\n（场景语境，不替换推理链）", fontsize=8,
            color=C_EDGE, ha="center")

    # ---- 标题与图例 -------------------------------------------------- #
    ax.text(6.0, 9.72, "M7 机制回迁数字大脑——7 项迁移机制 → 接入点映射",
            ha="center", va="center", fontsize=14, fontweight="bold", color="#111")
    ax.text(6.0, 0.62,
            "P-A1 6/6 机制封装 · 等价性 21/21 · 确定性 6/6 逐位一致 | P-A2 接入 144 全绿"
            "（117 零回归+27 新增）| P-A3 4 场景 ≥3 达标 | M-7 为阶段二 M8 设计依据（铁律 C）",
            ha="center", va="center", fontsize=8.4, color="#555")
    leg = [plt.Line2D([0], [0], color=C_OK, lw=2, label="已回迁（M-1..M-6）"),
           plt.Line2D([0], [0], color=C_REFUTE, lw=2, label="反证记录（M-7，设计依据）"),
           plt.Line2D([0], [0], color=C_EDGE, lw=2, label="机制→认知（行为语义，不替换推理链）")]
    ax.legend(handles=leg, loc="lower left", fontsize=8.4, frameon=True,
              bbox_to_anchor=(0.02, 0.10))

    fig.tight_layout()
    os.makedirs(REPORTS_DIR, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"映射图已写盘 {OUT_PNG}")
    return OUT_PNG


if __name__ == "__main__":
    gen_map()
