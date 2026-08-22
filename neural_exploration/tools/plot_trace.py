"""轨迹可视化（清单 §5.3）：输入 (t, v) → matplotlib 存图到 reports/neuro/。

用法：
  python tools/plot_trace.py --demo                          # 生成示例波形图
  python tools/plot_trace.py --csv data/hh1952_trace.csv     # 画参考解
  python tools/plot_trace.py --csv <f1> --csv2 <f2> --labels 参考,Brian2
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

REPORTS_DIR = os.path.join(ROOT, "reports", "neuro")


def plot_trace(t, v, title="Membrane potential", out_png=None, v2=None, label2=None,
               xlabel="t (ms)", ylabel="V (mV)"):
    """画单条（或两条叠加）轨迹并落盘，返回保存路径。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(REPORTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, v, lw=1.2, label="reference" if v2 is not None else None, color="#1f77b4")
    if v2 is not None:
        ax.plot(t[:len(v2)], v2, lw=0.8, alpha=0.85, label=label2 or "sim", color="#d62728")
        ax.legend()
    ax.axhline(-20.0, color="gray", ls="--", lw=0.8, label="spike threshold" if v2 is None else None)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    out_png = out_png or os.path.join(REPORTS_DIR, "trace.png")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def _demo():
    from tools.reference_data import rk4_reference_trace
    t, v = rk4_reference_trace()
    out = plot_trace(t, v, title="HH 1952 RK4 reference (M0 demo)", out_png=os.path.join(REPORTS_DIR, "m0_example.png"))
    print(f"示例图已生成: {out}")


def _load_csv(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="生成示例波形图")
    ap.add_argument("--csv", help="主轨迹 csv (t,v)")
    ap.add_argument("--csv2", help="叠加轨迹 csv (t,v)")
    ap.add_argument("--labels", default="", help="两条轨迹标签，逗号分隔")
    ap.add_argument("--out", default=None, help="输出 png 路径")
    args = ap.parse_args()

    if args.demo:
        _demo()
    elif args.csv:
        t, v = _load_csv(args.csv)
        v2, lab2 = None, None
        if args.csv2:
            _, v2 = _load_csv(args.csv2)
            labels = [s.strip() for s in args.labels.split(",")] if args.labels else ["ref", "sim"]
            lab2 = labels[1] if len(labels) > 1 else "sim"
        out = plot_trace(t, v, out_png=args.out, v2=v2, label2=lab2)
        print(f"图已生成: {out}")
    else:
        ap.print_help()
