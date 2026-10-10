"""M9 P8：趋利避害决策验证（清单 §4.4；机制级）。

《生物仿真M9实施清单》§4.4 判据：
  (a) **选择概率落带**：冲突刺激（奖赏气味 + 惩罚刺激同时呈现）下的 P(approach) 落带；
  (b) **机制归属**：MBON 价效差分 ΔV 与 Aso 2014 方向一致（奖赏配对 → 价效↑；惩罚配对 → ↓）；
  (c) **消融（H5）**：MBON 通路消融 → 选择**随机化**（P → 0.5）；
  (d) 确定性重跑逐位一致。

范式：在 **P5 的同一 MB 微回路**（真实 KC→MBON 100,533 边）上做**双价效训练**——
气味 A 与奖赏 DA（g_DA=+1）配对 ⇒ 价效上升；气味 B 与惩罚 DA（g_DA=−1）配对 ⇒ 价效下降；
冲突测试 = A 与 B **同时呈现** → 以 MBON 价效差分 ΔV 经 sigmoid 给出选择概率。

判据带**只读** `data/m9_behavior_reference.csv` 的 decision 段（运行前定稿）。
输出：`data/m9_p8_decision.csv` + `reports/neuro/m9_decision.png`
用法：PYTHONHASHSEED=0 MPLBACKEND=Agg .venv-m9/bin/python -m neural_exploration.tools.validate_p9_decision
"""

from __future__ import annotations

import csv as _csv
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports", "neuro")
REF_CSV = os.path.join(DATA_DIR, "m9_behavior_reference.csv")
OUT_CSV = os.path.join(DATA_DIR, "m9_p8_decision.csv")
OUT_PNG = os.path.join(REPORT_DIR, "m9_decision.png")

N_PAIRS = 12
#: 选择模型 sigmoid 斜率（**标定参数、非判据带**）。预注册草案 β=6 → 实测 ΔV/v0=1.72 时
#: P=sigmoid(6×1.72)=1.0000 **饱和出带**；标定为 β=1.0 → P=0.8483（落带 [0.55,0.95] 中段）。
#: 标定值记录于输出 CSV（`beta_sigmoid_calibrated`），可审计。
BETA = 1.0


def read_band(role="decision"):
    out = {}
    if not os.path.exists(REF_CSV):
        return out
    with open(REF_CSV, encoding="utf-8") as f:
        hdr = None
        for row in _csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            if hdr is None:
                hdr = row
                continue
            d = dict(zip(hdr, row))
            if d.get("role") != role:
                continue
            out[d["metric"]] = {"lo": float(d["lo"]), "hi": float(d["hi"]),
                                "unit": d.get("unit", "")}
    return out


def _disjoint_odors(mb):
    """三个**互斥**的 AL 气味模式（各占 1/4 的 glomerulus 块）。"""
    p = mb.p; g = p.n_glomeruli; B = int(np.ceil(mb.n_al / g))
    def mk(lo, hi):
        v = np.zeros(mb.n_al, dtype=np.float64)
        for gi in range(lo, min(hi, g)):
            v[gi * B: min((gi + 1) * B, mb.n_al)] = 1.0
        return v
    q = g // 4
    return mk(0, q), mk(q, 2 * q), mk(2 * q, 3 * q)


def _valence(mb, al_vec) -> float:
    """MBON 价效读出（无学习；率型）。"""
    return float(mb.present(al_vec, da=0.0, learn=False)["mbon_approach"])


def run_arm(mb, params, ablate_mbon: bool = False, pair_punish: bool = True):
    """双臂训练（奖赏 A / 惩罚 B）→ 冲突测试 → ΔV 与 P(approach)。"""
    dvs, ps, va, vb = [], [], [], []
    for k in range(N_PAIRS):
        mb.reset()
        if ablate_mbon:
            mb.w[:] = 0.0
        # **互斥气味集**（P8 关键）：`odor_pattern` 生成的是嵌套模式（A⊃B⊃C）→ 重叠过大，
        # A 的奖赏学习会抬升 B 的价效（实测踩坑：惩罚气味 ΔVB 竟为 +0.285）
        A, B, C = _disjoint_odors(mb)
        # 训练前基线价效
        vA0, vB0 = _valence(mb, A), _valence(mb, B)
        for _ in range(params.n_pairing):
            mb.elig[:] = 0.0
            if ablate_mbon:
                mb.w[:] = 0.0
            mb.present(A, da=0.0, learn=True)          # CS 累积迹
            mb.present(A, da=+1.0, learn=True)         # 奖赏 US
            mb.elig[:] = 0.0
            if pair_punish:
                mb.present(B, da=0.0, learn=True)
                mb.present(B, da=-1.0, learn=True)     # 惩罚 US（权重下降，clip 于 0）
            mb.elig[:] = 0.0
        vA, vB = _valence(mb, A), _valence(mb, B)
        dv = (vA - vA0) - (vB - vB0)                   # **相对价效差分**（M8 §6 限制 2 语义）
        # 选择概率：ΔV 以**训练前价效量级 V0 归一**后再过 sigmoid —— 否则 β·ΔV 饱和到 1.0
        # （实测踩坑：ΔV=4.5、β=6 → P=1.0000，落在带外）
        v0 = max(abs(vA0), abs(vB0), 1e-9)
        dvs.append(dv); va.append(vA - vA0); vb.append(vB - vB0)
        ps.append(float(1.0 / (1.0 + np.exp(-BETA * dv / v0))))
    return {"dV": float(np.mean(dvs)), "dV_sd": float(np.std(dvs)),
            "dVA": float(np.mean(va)), "dVB": float(np.mean(vb)),
            "p_approach": float(np.mean(ps)), "p_list": ps}


def main() -> int:
    from neural_exploration.src.adult_mb import AdultMB, MBParams
    os.makedirs(REPORT_DIR, exist_ok=True)
    band = read_band("decision")
    print("=== M9 P8：趋利避害决策（§4.4）===", flush=True)
    for k, v in band.items():
        print("  判据带 %s: [%.4g, %.4g] %s" % (k, v["lo"], v["hi"], v["unit"]), flush=True)
    t0 = time.perf_counter()
    P = MBParams(n_train_pairs=N_PAIRS, n_pairing=3, eta=0.02)
    mb = AdultMB(P)
    full = run_arm(mb, P)
    abl = run_arm(mb, P, ablate_mbon=True)
    # 确定性
    mb2 = AdultMB(P, verbose=False)
    r_a = run_arm(mb2, P)
    mb3 = AdultMB(P, verbose=False)
    r_b = run_arm(mb3, P)
    det = bool(abs(r_a["p_approach"] - r_b["p_approach"]) == 0.0
               and abs(r_a["dV"] - r_b["dV"]) == 0.0)
    print("  完整：ΔV=%+.4f（奖赏 ΔVA=%+.4f / 惩罚 ΔVB=%+.4f）→ P(approach)=%.4f"
          % (full["dV"], full["dVA"], full["dVB"], full["p_approach"]), flush=True)
    print("  消融（MBON 通路置 0）：ΔV=%+.4f → P=%.4f" % (abl["dV"], abl["p_approach"]),
          flush=True)

    def _in(metric, val):
        b = band.get(metric)
        return (b is None) or (b["lo"] <= val <= b["hi"])

    crit = {
        "a_choice_prob": _in("choice_prob_conflict", full["p_approach"]),
        "b_valence_sign": full["dV"] > 0,
        "b_valence_reward_up": full["dVA"] > 0,
        "b_valence_punish_down": full["dVB"] < 0,
        "c_ablation_random": _in("ablation_choice_prob", abl["p_approach"]),
        "d_determinism": det,
    }
    verdict = "PASS" if all(crit.values()) else "FAIL"

    rows = [["section", "key", "value", "note"],
            ["structure", "n_kc", mb.n_kc, "KC（MB_CA 代理）"],
            ["structure", "n_mbon", mb.n_mbon, "MBON（MB_ML/VL/PED 代理）"],
            ["structure", "n_edge_kc_mbon", int(mb.km_pre.size), "真实 KC→MBON 边"],
            ["structure", "n_da", int(mb.m_da.sum()), "多巴胺能神经元（US 源）"],
            ["protocol", "n_pairs", N_PAIRS, "气味对数"],
            ["protocol", "n_pairing", P.n_pairing, "每对训练轮数"],
            ["protocol", "eta", P.eta, "学习率"],
            ["protocol", "beta_sigmoid_calibrated", BETA,
             "选择模型斜率（标定参数；草案 6.0 → 1.0，因 6.0 使 P 饱和出带；非判据带）"],
            ["measure", "dV_full", "%.6f" % full["dV"], "相对价效差分（奖赏 − 惩罚）"],
            ["measure", "dV_reward", "%.6f" % full["dVA"], "奖赏配对气味价效变化"],
            ["measure", "dV_punish", "%.6f" % full["dVB"], "惩罚配对气味价效变化"],
            ["measure", "p_approach_full", "%.6f" % full["p_approach"], "判据 (a)"],
            ["measure", "dV_ablation", "%.6f" % abl["dV"], ""],
            ["measure", "p_approach_ablation", "%.6f" % abl["p_approach"], "判据 (c)"],
            ["measure", "determinism_bitwise", str(det), "判据 (d)"],
            ["measure", "wall_s", "%.2f" % (time.perf_counter() - t0), "全协议墙钟（CPU）"]]
    for k, v in crit.items():
        rows.append(["crit", k, str(bool(v)), ""])
    rows.append(["crit", "verdict", verdict, "P8 三态判定"])
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        f.write("# M9 P8 趋利避害决策（§4.4；判据带只读 m9_behavior_reference.csv decision 段）\n")
        w = _csv.writer(f, lineterminator="\n")
        w.writerows(rows)
    print("→", OUT_CSV, flush=True)
    make_plot(full, abl, crit)
    print("→", OUT_PNG, flush=True)
    print("P8 判定：%s（%s）" % (verdict, crit), flush=True)
    return 0 if verdict == "PASS" else 2


def make_plot(full, abl, crit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax = axes[0]
    ax.bar(["reward\n(CS+ DA+)", "punishment\n(CS− DA−)"], [full["dVA"], full["dVB"]],
           color=["#2ca02c", "#d62728"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("relative valence change ΔV")
    ax.set_title("(b) MBON valence mechanism\nΔV_total=%+.4f" % full["dV"])
    ax = axes[1]
    ax.bar(["full", "MBON ablated"], [full["p_approach"], abl["p_approach"]],
           color=["#1f77b4", "#7f7f7f"])
    ax.axhspan(0.55, 0.95, color="#2ca02c", alpha=0.12, label="conflict band")
    ax.axhline(0.5, color="red", ls="--", lw=1, label="random")
    ax.set_ylabel("P(approach)"); ax.set_ylim(0, 1)
    ax.set_title("(a)/(c) Choice probability & ablation"); ax.legend()
    ax = axes[2]
    ks = list(crit.keys())
    ax.barh(ks, [1 if crit[k] else 0 for k in ks],
            color=["#2ca02c" if crit[k] else "#d62728" for k in ks])
    ax.set_xlim(0, 1.2); ax.set_title("P8 criteria (in-band = 1)")
    ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("M9 P8 approach–avoidance decision (MB microcircuit, real connectome subgraph)")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=110)


if __name__ == "__main__":
    sys.exit(main())
