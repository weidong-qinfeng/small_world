"""M8 幼虫闭环（LarvaLoop）：LarvaCircuit ↔ 行为判据带 ↔ 虚拟身体（P4/P5 前置）。

对应《生物仿真M8实施清单》§5（步骤 3 身体）/§6（步骤 4 权重定稿）/§7（步骤 5
行为验证）：
  - **P4 自发行为协议**（§7.1）：无刺激无梯度 T 窗 → `run_spontaneous` 状态比例
    （run/turn/pause/curl）vs `data/m8_behavior_reference.csv` 带（唯一定稿源，
    M3 P5 ×1.2 教训：不做事后调带）；
  - **P5 学习探针**（§7.2 前置）：AWC→KC→MBON 蘑菇体通路 LI（机制级判据，
    M6 L16 语义——网络级 CI 读出不可见的测量限制已由 `run_learning_probe` 的
    weight 档 LI 处理）；
  - **G1 双状态**（§0.5）：静息静默落带 + 自发 bout 活动（`g1_dual_state_check`）；
  - **CI 方向**（§3.4 降阶正确性锚）：AWC 嗅觉趋化正符号；
  - **确定性**：p=1/n=1、seed 固定；同参数重跑逐位一致（M4 纪律）。

复用（冻结文件零修改）：LarvaCircuit（`src/larva_circuit.py`，B1b 交付）、
VirtualLarvaBody/classify_larva_state（`src/larva_body.py`，B1c2 交付）、
ChemotaxisEnv/ci_group_stats（M4）、g1_dual_state_check（B1b）。

构造参数默认 None（M3 L13）；本模块无 Brian2 编译（session 由 circuit 提供）。
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.src.larva_body import (  # noqa: E402
    VirtualLarvaBody,
    larva_state_fractions,
    load_larva_body_params,
)
from neural_exploration.src.larva_circuit import (  # noqa: E402
    LI_APPEAR_THRESHOLD,
    LarvaCircuit,
    g1_dual_state_check,
)

DEFAULT_BEHAVIOR_REF_CSV = os.path.join(ROOT, "neural_exploration", "data",
                                        "m8_behavior_reference.csv")


def load_behavior_reference(csv_path: Optional[str] = None) -> Dict[str, dict]:
    """解析 data/m8_behavior_reference.csv（唯一定稿源）→ {(role, neuron_class): band}。

    band dict：{lo, hi, unit, tol_lo, tol_hi, target, provenance, note}。
    文件缺失 → 返回空 dict（调用方回退代码内预注册默认并记录）。
    """
    path = csv_path or DEFAULT_BEHAVIOR_REF_CSV
    out: Dict[Tuple[str, str], dict] = {}

    def _clean(ln: str) -> str:
        s = ln.strip()
        if s.startswith('"'):
            s = s.strip('"')
        return s

    if not os.path.exists(path):
        return out
    import csv as _csv

    with open(path, newline="", encoding="utf-8") as f:
        for ln in f:
            s = _clean(ln)
            if not s or s.startswith("#"):
                continue
            fields = next(_csv.reader([s]))
            if len(fields) < 10:
                continue
            role = fields[0].strip().lower()
            key = fields[1].strip().lower()
            if role in ("", "role") or not key:
                continue

            def _f(x):
                try:
                    return float(x) if x not in ("", "nan") else None
                except ValueError:
                    return None

            out[(role, key)] = dict(
                lo=_f(fields[2]), hi=_f(fields[3]), unit=fields[4].strip(),
                tol_lo=_f(fields[5]), tol_hi=_f(fields[6]),
                target=_f(fields[7]), provenance=fields[8].strip(),
                note=fields[9].strip() if len(fields) > 9 else "")
    return out


def band_check(value: float, band: Optional[dict],
               tol: bool = True) -> Dict[str, object]:
    """判据带判定（M5 §0 #5 语义：容差窗 = tol_lo/tol_hi，无 tol → lo/hi）。

    Returns dict(in_band, band, value)。
    """
    if band is None:
        return dict(in_band=False, band=None, value=value)
    lo = band.get("tol_lo" if tol else "lo")
    hi = band.get("tol_hi" if tol else "hi")
    if lo is None or hi is None:
        return dict(in_band=False, band=band, value=value)
    return dict(in_band=bool(lo <= value <= hi), band=band, value=value)


class LarvaLoop:
    """幼虫闭环运行器：电路协议 + 行为判据带（P4/P5/G1/CI，确定性）。

    协议参数优先取 data/m8_larva_body_params.csv + m8_behavior_reference.csv
    （唯一定稿源），缺省回退预注册默认（B1c2 定稿草案）。
    """

    def __init__(
        self,
        scale: int = 300,
        fidelity: str = "point",
        plasticity: str = "none",
        seed: int = 0,
        behavior_ref_csv: Optional[str] = None,
        body_params_csv: Optional[str] = None,
        circuit_kw: Optional[dict] = None,
    ):
        self.scale = int(scale)
        self.fidelity = fidelity
        self.plasticity = plasticity
        self.seed = int(seed)
        self.behavior_ref = load_behavior_reference(behavior_ref_csv)
        bp = load_larva_body_params(body_params_csv)
        self.body_params = bp
        self.circuit_kw = dict(circuit_kw or {})
        self.circuit_kw.setdefault("scale", self.scale)
        self.circuit_kw.setdefault("fidelity", self.fidelity)
        self.circuit_kw.setdefault("plasticity", self.plasticity)
        self.circuit_kw.setdefault("seed", self.seed)
        self.circuit: Optional[LarvaCircuit] = None

    # ------------------------------------------------------------------ #
    # 电路构造（惰性；每协议 make_session 由 circuit 处理）
    # ------------------------------------------------------------------ #
    def make_circuit(self, plasticity: Optional[str] = None) -> LarvaCircuit:
        kw = dict(self.circuit_kw)
        if plasticity is not None:
            kw["plasticity"] = plasticity
        self.circuit = LarvaCircuit(**kw)
        return self.circuit

    # ------------------------------------------------------------------ #
    # P4：自发行为分布（vs m8_behavior_reference.csv 带）
    # ------------------------------------------------------------------ #
    def run_spontaneous(self, t_total_ms: float = 5000.0,
                        seed: Optional[int] = None) -> Dict[str, object]:
        """无刺激无梯度 T 窗 → 状态比例 + 带判定（P4 判据）。

        ⚠ 状态名映射（B1c3 实测坑）：LarvaCircuit.run_spontaneous 用 M5
        classify_state（virtual_body 冻结语义）→ 键为 fwd/rev/turn/pause；
        P4 判据用幼虫状态（run/turn/pause/curl，larva_body 语义）——映射：
        run=fwd、turn=turn+rev（反转并入 turn，m8_larva_body_params note）、
        pause=pause、curl=0（本协议无 curl 驱动）。

        Returns dict(frac, states, band_checks, g1_dual_state, wall_s)。
        """
        seed = self.seed if seed is None else int(seed)
        circ = self.make_circuit(plasticity="none")
        sp = circ.run_spontaneous(t_total_ms=t_total_ms, seed=seed)
        m5 = sp["frac"]
        frac = dict(run=m5.get("fwd", 0.0),
                    turn=m5.get("turn", 0.0) + m5.get("rev", 0.0),
                    pause=m5.get("pause", 0.0),
                    curl=0.0)
        checks = {}
        for state in ("run", "turn", "pause", "curl"):
            band = self.behavior_ref.get(("spontaneous",
                                          f"time_fraction_{state}"))
            checks[state] = band_check(frac.get(state, 0.0) * 100.0, band)
        bout = sum(frac.get(k, 0.0) for k in ("run", "turn"))
        checks["bout_activity"] = dict(
            in_band=bool(bout >= 0.10), value=round(bout, 4), band=None)
        return dict(frac=frac, states=sp["states"], band_checks=checks,
                    bout_activity=round(bout, 4), wall_s=sp["wall_s"],
                    m5_frac=m5)

    # ------------------------------------------------------------------ #
    # P5 前置：学习探针（AWC→KC→MBON LI；机制级判据）
    # ------------------------------------------------------------------ #
    def run_learning_probe(self, t_test_ms: float = 2000.0,
                           t_train_ms: float = 2000.0,
                           seed: Optional[int] = None) -> Dict[str, object]:
        """学习探针（stdp 档）：LI + 带判定（P5 前置判据）。

        Returns dict(li, li_mode, n_stdp_edges, band_check, wall_s)。
        """
        seed = self.seed if seed is None else int(seed)
        circ = self.make_circuit(plasticity="stdp")
        lp = circ.run_learning_probe(t_test_ms=t_test_ms,
                                     t_train_ms=t_train_ms, seed=seed)
        li = float(lp["li"])
        band = self.behavior_ref.get(("learning", "li_gain_band"))
        check = band_check(li, band)
        return dict(li=li, li_mode=lp["li_mode"], n_stdp_edges=lp["n_stdp_edges"],
                    band_check=check, wall_s=lp["wall_s"])

    # ------------------------------------------------------------------ #
    # CI：嗅觉趋化短协议（§3.4 降阶正确性锚，正符号）
    # ------------------------------------------------------------------ #
    def run_chemotaxis_ci(self, t_total_ms: float = 5000.0,
                          n_trials: int = 1,
                          seed_base: int = 0) -> Dict[str, object]:
        """气味梯度闭环 → CI（方向条款：CI>0 为正趋化）。

        Returns dict(ci, direction, n_turn_events, wall_s)。
        """
        circ = self.make_circuit(plasticity="none")
        res, meta = circ.run_chemotaxis_trials(
            n_trials=n_trials, t_total_ms=t_total_ms, seed_base=seed_base)
        ci = float(res[0]["ci"])
        return dict(ci=ci, direction="+" if ci > 0 else "-",
                    n_turn_events=res[0]["n_turn_events"],
                    wall_s=meta["mean_wall_s"])

    # ------------------------------------------------------------------ #
    # G1：静息静默 + 自发 bout 双状态（§0.5）
    # ------------------------------------------------------------------ #
    def run_g1(self, resting_t_ms: float = 2000.0,
               spont_t_ms: float = 5000.0,
               settle_ms: float = 500.0) -> Dict[str, object]:
        circ = self.make_circuit(plasticity="none")
        rest = circ.run_resting(t_total_ms=resting_t_ms, settle_ms=settle_ms)
        sp = circ.run_spontaneous(t_total_ms=spont_t_ms)
        g1 = g1_dual_state_check(rest, sp)
        return dict(rest=rest, spont=sp, g1=g1, silent_frac=rest["silent_frac"],
                    bout_activity=g1["bout_activity"], dual_state=g1["dual_state"])

    # ------------------------------------------------------------------ #
    # 静息发放率（G1 输入 + 无 NaN/无发散前置）
    # ------------------------------------------------------------------ #
    def run_resting(self, t_total_ms: float = 2000.0,
                    settle_ms: float = 500.0) -> Dict[str, object]:
        circ = self.make_circuit(plasticity="none")
        return circ.run_resting(t_total_ms=t_total_ms, settle_ms=settle_ms)


# --------------------------------------------------------------------- #
# 冒烟出图（reports/neuro/m8_smoke.png）
# --------------------------------------------------------------------- #
def plot_smoke(spont: dict, lp: dict, ci: dict, g1: dict,
               png_path: str) -> str:
    """M8 冒烟图：自发状态分布 + CI/LI/G1 指标。"""
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

    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # 1) 自发状态分布（P4）
    ax = axes[0]
    frac = spont["frac"]
    states = [s for s in ("run", "turn", "pause", "curl") if s in frac]
    vals = [frac[s] for s in states]
    colors = {"run": "#4c72b0", "turn": "#55a868", "pause": "#c44e52",
              "curl": "#8172b2"}
    ax.bar(states, vals, color=[colors.get(s, "#999999") for s in states])
    ax.set_ylabel("时间比例")
    ax.set_title(f"自发状态分布（{spont.get('scale', 300)} 档）")
    for s, v in zip(states, vals):
        ax.annotate(f"{v:.2f}", (s, v), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2) CI / LI / G1 指标
    ax = axes[1]
    labels = ["CI", "LI", "bout", "1-silent"]
    g1b = g1.get("g1", {})
    vals2 = [ci["ci"], lp["li"], g1b.get("bout_activity", float("nan")),
             1.0 - float(g1b.get("silent_frac", float("nan")))]
    ax.bar(labels, vals2, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"])
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(LI_APPEAR_THRESHOLD, color="gray", ls="--", lw=1,
               label=f"LI 出现 {LI_APPEAR_THRESHOLD}")
    ax.set_ylabel("指标值")
    ax.set_title(f"CI={ci['ci']:+.3f}（{ci['direction']}）  LI={lp['li']:+.3f}"
                 f"（{lp['li_mode']}）")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3) G1 双状态（静默 vs bout）
    ax = axes[2]
    silent = float(g1b.get("silent_frac", float("nan")))
    bout = float(g1b.get("bout_activity", float("nan")))
    ax.bar(["静默比例", "bout 活动"], [silent, bout],
           color=["#1f77b4", "#2ca02c"])
    ax.axhspan(0.50, 0.90, color="green", alpha=0.12, label="G1 静默带")
    ax.axhline(0.10, color="orange", ls="--", lw=1, label="bout 下限")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("比例")
    ax.set_title(f"G1 双状态：{'PASS' if g1b.get('dual_state') else 'FAIL'}"
                 f"（{g1.get('scale', 300)} 档）")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("M8 幼虫闭环冒烟（larva_loop.py）：自发分布 + CI/LI + G1 双状态",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    return png_path
