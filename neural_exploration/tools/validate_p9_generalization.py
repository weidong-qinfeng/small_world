"""M9 P5：MB 联想泛化验证（清单 §4.1；机制级 + 行为级双轨）。

《生物仿真M9实施清单》§4.1（P5 判据）：
  (a) 训练后偏好显著 > 未配对对照（**符号翻转置换检验** exact p<0.05 且 Cohen d≥0.5）；
  (b) **泛化梯度与真实一致**：相似度-偏好响应曲线单调（Spearman ρ≥0.9）且半宽落带；
  (c) **消融（机制归属）**：η=0 无学习 / DA 门控=0 无学习（H2）/ KC→MBON 子图消融无学习（H1）；
  (d) **机制锚**：MBON 价效编码方向正确（奖赏配对 → MBON 读出增强）；
  (e) 确定性重跑逐位一致。

判据带**只读** `data/m9_behavior_reference.csv` 的 learning 段（运行前定稿）。
结构 = 真实连接组子图（KC 3,819 / MBON 1,374 / AL 3,449；KC→MBON 100,533 边）。

输出：`data/m9_p5_generalization.csv` + `reports/neuro/m9_generalization.png`
用法：PYTHONHASHSEED=0 MPLBACKEND=Agg .venv-m9/bin/python -m neural_exploration.tools.validate_p9_generalization
"""

from __future__ import annotations

import csv as _csv
import itertools
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports", "neuro")
REF_CSV = os.path.join(DATA_DIR, "m9_behavior_reference.csv")
OUT_CSV = os.path.join(DATA_DIR, "m9_p5_generalization.csv")
OUT_PNG = os.path.join(REPORT_DIR, "m9_generalization.png")

N_PAIRS = 12
N_GEN = 12


def read_band(role="learning"):
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


def sign_flip_perm_p(diffs: np.ndarray) -> float:
    """符号翻转置换检验（exact，2^n；确定性、无 scipy 依赖）。"""
    d = np.asarray(diffs, dtype=np.float64)
    obs = abs(d.mean())
    n = d.size
    if n == 0:
        return float("nan")
    if n <= 16:
        cnt = 0
        for signs in itertools.product((1.0, -1.0), repeat=n):
            if abs((d * np.asarray(signs)).mean()) >= obs - 1e-12:
                cnt += 1
        return cnt / float(2 ** n)
    rng = np.random.default_rng(0)
    S = 20000
    cnt = 0
    for _ in range(S):
        if abs((d * rng.choice([-1.0, 1.0], size=n)).mean()) >= obs - 1e-12:
            cnt += 1
    return (cnt + 1) / (S + 1)


def cohen_d_paired(diffs: np.ndarray) -> float:
    d = np.asarray(diffs, dtype=np.float64)
    sd = d.std(ddof=1) if d.size > 1 else 0.0
    return float(d.mean() / sd) if sd > 0 else float("inf")


def spearman(x, y) -> float:
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    if x.size < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean(); ry -= ry.mean()
    den = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def main() -> int:
    from neural_exploration.src.adult_mb import AdultMB, MBParams
    os.makedirs(REPORT_DIR, exist_ok=True)
    band = read_band("learning")
    print("=== M9 P5：MB 联想泛化（§4.1）===", flush=True)
    for k, v in band.items():
        print("  判据带 %s: [%.4g, %.4g] %s" % (k, v["lo"], v["hi"], v["unit"]), flush=True)
    t0 = time.perf_counter()
    P = MBParams(n_train_pairs=N_PAIRS, n_pairing=3, eta=0.02, tau_e_ms=200.0)
    mb = AdultMB(P)
    rows = [["section", "key", "value", "note"]]
    rows += [["structure", "n_kc", mb.n_kc, "KC（主脑区 MB_CA 代理；抽象登记①）"],
             ["structure", "n_mbon", mb.n_mbon, "MBON（MB_ML/MB_VL/MB_PED 代理）"],
             ["structure", "n_al", mb.n_al, "AL（触角叶）"],
             ["structure", "n_da", int(mb.m_da.sum()), "多巴胺能神经元（US 门控源）"],
             ["structure", "n_edge_kc_mbon", int(mb.km_pre.size), "真实 KC→MBON 连接组边"],
             ["structure", "n_edge_al_kc", int(mb.ak_pre.size), "真实 AL→KC 边"]]

    # ---- 主协议（配对 + 未配对对照 + 泛化梯度）----
    # **修正口径（运行后，因原口径发现不可判别/伪重复；原口径结果并列保留在档）**：
    #   原有实现每对都用**同一组**气味模式（start=0）→ 12 个 ΔLI **逐位相同**（sd≈3e-17）
    #   → 伪重复：置换检验有效样本量实为 1、Cohen d 退化为 1e15。
    #   修正：`independent_patterns=True`（每对抽不同 glomerulus 起始位置）→ 真 12 个独立重复。
    proto = mb.run_protocol(n_pairs=N_PAIRS, n_generalization=N_GEN,
                            independent_patterns=True)
    li_paired = float(proto["li"])
    d_obs = np.asarray(proto["paired"], dtype=np.float64)
    eff_n = int(np.unique(np.round(d_obs, 12)).size)
    gen = np.asarray(proto["gen"], dtype=np.float64)
    sims = np.asarray(proto["sims"], dtype=np.float64)
    print("  主协议（修正口径 independent_patterns）：LI=%+.4f  Δw=%.0f  泛化半宽=%.3f"
          "  有效独立对数=%d/%d（%.1fs）"
          % (li_paired, proto["dw_l1"], proto["gen_halfwidth"], eff_n, N_PAIRS,
             time.perf_counter() - t0), flush=True)
    print("    泛化曲线:", [round(v, 3) for v in gen], flush=True)

    # ---- 原口径（审计：同一组气味模式，伪重复）----
    mb_l = AdultMB(P, verbose=False)
    legacy = mb_l.run_protocol(n_pairs=N_PAIRS, n_generalization=N_GEN,
                               independent_patterns=False)
    d_leg = np.asarray(legacy["paired"], dtype=np.float64)
    leg_p = sign_flip_perm_p(d_leg - np.zeros_like(d_leg))
    leg_d = cohen_d_paired(d_leg - np.zeros_like(d_leg))
    print("  原口径（审计）：LI=%+.6f、有效独立对数=%d/%d、Cohen d=%.3g → **伪重复**（不可作显著性证据）"
          % (float(legacy["li"]), int(np.unique(np.round(d_leg, 12)).size), N_PAIRS, leg_d),
          flush=True)

    # ---- 未配对对照臂（同一网络，US 关闭）----
    mb_c = AdultMB(P, verbose=False)
    ctrl = mb_c.run_protocol(n_pairs=N_PAIRS, eta=0.0, da_gain=0.0,
                             n_generalization=N_GEN, independent_patterns=True)
    li_ctrl = float(ctrl["li"])
    # 逐"试次对"的差值（配对 vs 对照），用于置换检验
    d_pairs = d_obs
    d_ctrl = np.asarray(ctrl["paired"], dtype=np.float64)
    diffs = d_pairs - d_ctrl
    p_val = sign_flip_perm_p(diffs)
    d_eff = cohen_d_paired(diffs)
    print("  对照臂：LI=%+.4f（Δw=%.0f，**三因子规则下结构上恒为 0，非测量值**）；"
          "配对−对照 置换检验 p=%.4g，Cohen d=%.3f（有效 n=%d）"
          % (li_ctrl, ctrl["dw_l1"], p_val, d_eff, eff_n), flush=True)

    # ---- 消融（机制归属）----
    abl = {}
    abl_dw = {}
    mb2 = AdultMB(P, verbose=False)
    r2a = mb2.run_protocol(n_pairs=N_PAIRS, eta=0.0,
                           n_generalization=N_GEN, independent_patterns=True)
    abl["eta0"] = float(r2a["li"]); abl_dw["eta0"] = float(r2a["dw_l1"])
    mb3 = AdultMB(P, verbose=False)
    r3a = mb3.run_protocol(n_pairs=N_PAIRS, da_gain=0.0,
                           n_generalization=N_GEN, independent_patterns=True)
    abl["da0"] = float(r3a["li"]); abl_dw["da0"] = float(r3a["dw_l1"])
    mb4 = AdultMB(P, verbose=False)
    r4a = mb4.run_protocol(n_pairs=N_PAIRS, ablate_subgraph=True,
                           n_generalization=N_GEN, independent_patterns=True)
    abl["subgraph_off"] = float(r4a["li"]); abl_dw["subgraph_off"] = float(r4a["dw_l1"])
    abl_dw["paired"] = float(proto["dw_l1"])
    print("  消融：η=0 → LI=%+.4f（Δw=%.0f）；DA=0 → %+.4f（Δw=%.0f）；KC→MBON=0 → %+.4f（Δw=%.0f）"
          % (abl["eta0"], abl_dw["eta0"], abl["da0"], abl_dw["da0"],
             abl["subgraph_off"], abl_dw["subgraph_off"]), flush=True)
    # **读回式判别力**：ΔLI=0 在相对读法下由"无权重变化"**结构上保证**、不可能失败 →
    # 消融的判别力必须来自 `Δw_l1 == 0` 这一**读回**（原实现从未记录）
    ablation_dw_zero = all(abl_dw[k] == 0.0 for k in ("eta0", "da0", "subgraph_off"))
    print("  读回：三消融臂 Δw_l1 == 0 → %s（配对臂 Δw_l1=%.0f）"
          % (ablation_dw_zero, abl_dw["paired"]), flush=True)

    # ---- 确定性（逐位）----
    mb5 = AdultMB(P, verbose=False)
    r5 = mb5.run_protocol(n_pairs=3, n_generalization=N_GEN, independent_patterns=True)
    mb6 = AdultMB(P, verbose=False)
    r6 = mb6.run_protocol(n_pairs=3, n_generalization=N_GEN, independent_patterns=True)
    det = bool(np.array_equal(np.asarray(r5["gen"]), np.asarray(r6["gen"]))
               and abs(r5["li"] - r6["li"]) == 0.0)

    # ---- 判据 ----
    def _in(metric, val):
        b = band.get(metric)
        return (b is None) or (b["lo"] <= val <= b["hi"])

    rho = spearman(sims, gen)
    crit = {
        "a_li_in_band": _in("li_gain", li_paired),
        "a_paired_gt_control": (p_val < 0.05) and (d_eff >= 0.5),
        "b_gen_halfwidth": _in("generalization_halfwidth", proto["gen_halfwidth"]),
        "b_gen_monotonic": rho >= 0.9,
        "c_ablation_eta0": abs(abl["eta0"]) <= 0.05,
        "c_ablation_da0": abs(abl["da0"]) <= 0.05,
        "c_ablation_subgraph": abs(abl["subgraph_off"]) <= 0.05,
        # **可判别性补强（原判据不可判别 → 新增读回判据；运行后修正，原行保留在档）**
        "c_ablation_dw_readback": ablation_dw_zero and (abl_dw["paired"] > 0),
        "d_mbon_valence_sign": li_paired > 0,
        "e_determinism": det,
    }
    verdict = "PASS" if all(crit.values()) else "FAIL"
    print("  Spearman(相似度,响应)=%.4f；判定=%s" % (rho, verdict), flush=True)

    rows += [["protocol", "n_pairs", N_PAIRS, "气味对总数（配对组/对照组各 12）"],
             ["protocol", "n_pairing", P.n_pairing, "配对训练轮数"],
             ["protocol", "eta", P.eta, "学习率（三因子）"],
             ["protocol", "tau_e_ms", P.tau_e_ms, "eligibility 时间常数"],
             ["protocol", "kc_sparsity", P.kc_sparsity, "KC 稀疏码比例"],
             ["protocol", "n_generalization", N_GEN, "泛化梯度档数"],
             ["protocol", "independent_patterns", "True",
              "**修正口径**：每对抽不同 glomerulus 起始位置（原口径 False → 12 对逐位相同 = 伪重复）"],
             ["audit", "legacy_li_same_patterns", "%.6f" % float(legacy["li"]),
              "**原口径（保留在档）**：同一组气味模式 → 12 个 ΔLI 逐位相同"],
             ["audit", "legacy_effective_n", int(np.unique(np.round(d_leg, 12)).size),
              "原口径的**有效独立对数**（=1 → 置换检验/Cohen d 无效）"],
             ["audit", "legacy_cohen_d", "%.6g" % leg_d,
              "原口径 Cohen d 退化（sd≈3e-17）→ **不可解释**"],
             ["audit", "legacy_perm_p", "%.6g" % leg_p,
              "原口径 p=2/4096 只反映『值≠0』，**不是** 12 个独立重复的显著性"],
             ["measure", "effective_n_pairs", eff_n,
              "修正口径下**互不相同**的 ΔLI 个数（= 有效样本量）"],
             ["measure", "li_paired", "%.6f" % li_paired,
              "判据 (a) ΔLI = LI_after − LI_base（**相对读法**）"],
             ["measure", "li_paired_sd", "%.6f" % float(np.std(d_obs)),
              "修正口径下 12 对 ΔLI 的标准差（原口径 2.8e-17）"],
             ["measure", "li_abs", "%.6f" % proto.get("li_abs", float("nan")),
              "**双读数**：绝对读法 LI_after（= li_baseline + ΔLI）"],
             ["measure", "li_baseline", "%.6f" % proto.get("li_base", float("nan")),
              "**训练前基线 LI**（主agent 所见 0.163672 的来源；由 CS+⊃CS− 嵌套的**气味强度序**造成）"],
             ["measure", "li_control", "%.6f" % li_ctrl,
              "未配对对照臂 ΔLI（**三因子规则下结构上恒为 0，非测量值** → 不可判别，判别力靠 Δw 读回）"],
             ["measure", "perm_p", "%.6g" % p_val, "符号翻转置换检验（exact，n=%d 独立重复）" % eff_n],
             ["measure", "cohen_d", "%.6f" % d_eff, "配对效应量（修正口径）"],
             ["measure", "dw_l1", "%.2f" % proto["dw_l1"], "权重 L1 变化（机制级）"],
             ["measure", "ablation_dw_l1_eta0", "%.10g" % abl_dw["eta0"], "**读回**：η=0 臂 Δw（=0 证明消融生效）"],
             ["measure", "ablation_dw_l1_da0", "%.10g" % abl_dw["da0"], "**读回**：DA=0 臂 Δw"],
             ["measure", "ablation_dw_l1_subgraph", "%.10g" % abl_dw["subgraph_off"], "**读回**：子图=0 臂 Δw"],
             ["measure", "gen_halfwidth", "%.4f" % proto["gen_halfwidth"], "泛化半宽（相似度）"],
             ["measure", "gen_spearman", "%.4f" % rho, "相似度-响应单调性"],
             ["measure", "ablation_eta0_li", "%.6f" % abl["eta0"], "η=0（**ΔLI 恒 0，不可判别**）"],
             ["measure", "ablation_da0_li", "%.6f" % abl["da0"], "DA 门控=0（H2）（**ΔLI 恒 0，不可判别**）"],
             ["measure", "ablation_subgraph_li", "%.6f" % abl["subgraph_off"], "KC→MBON=0（H1）（**ΔLI 恒 0，不可判别**）"],
             ["measure", "determinism_bitwise", str(det), ""],
             ["measure", "wall_s", "%.1f" % (time.perf_counter() - t0), "全协议墙钟（CPU）"]]
    for i, (s, g) in enumerate(zip(sims, gen)):
        rows.append(["gen_curve", "sim%.3f" % s, "%.6f" % g, "档 %d" % i])
    for k, v in crit.items():
        rows.append(["crit", k, str(bool(v)), ""])
    rows.append(["crit", "verdict", verdict, "P5 三态判定"])
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        f.write("# M9 P5 MB 联想泛化（§4.1；判据带只读 m9_behavior_reference.csv learning 段）\n")
        w = _csv.writer(f, lineterminator="\n")
        w.writerows(rows)
    print("→", OUT_CSV, flush=True)
    make_plot(sims, gen, li_paired, li_ctrl, abl, crit, p_val, d_eff,
              li_base=proto.get("li_base"), li_abs=proto.get("li_abs"),
              abl_dw=abl_dw, eff_n=eff_n)
    print("→", OUT_PNG, flush=True)
    print("P5 判定：%s（%s）" % (verdict, crit), flush=True)
    return 0 if verdict == "PASS" else 2


def make_plot(sims, gen, li_paired, li_ctrl, abl, crit, p_val, d_eff,
              li_base=None, li_abs=None, abl_dw=None, eff_n=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    ax.plot(sims, gen, "o-", color="#1f77b4")
    ax.axhline(0.5, color="red", ls="--", lw=1, label="50% (half-width)")
    ax.set_xlabel("odor similarity to CS+ (abstract)"); ax.set_ylabel("normalized MBON response")
    ax.set_title("(b) Generalization gradient (monotonic, half-width)")
    ax.invert_xaxis(); ax.legend()
    ax = axes[0, 1]
    # **双读数**：绝对 LI（= 基线 + ΔLI）与 ΔLI 并列，避免把基线当成消融残留
    labels = ["paired\nΔLI", "baseline\n(pre-train)", "paired\nabs LI"]
    vals = [li_paired, li_base if li_base is not None else 0.0,
            li_abs if li_abs is not None else 0.0]
    ax.bar(labels, vals, color=["#2ca02c", "#7f7f7f", "#1f77b4"])
    ax.axhspan(0.1, 1.0, color="#2ca02c", alpha=0.12, label="LI band [0.1,1.0]")
    ax.set_ylabel("LI (learning index)")
    ax.set_title("(a) Dual reading: ΔLI is the criterion\nperm p=%.3g, Cohen d=%.2f (n=%s)"
                 % (p_val, d_eff, str(eff_n)))
    ax.legend()
    ax = axes[1, 0]
    names = ["paired", "η=0", "DA=0", "KC→MBON=0"]
    ks = ["paired", "eta0", "da0", "subgraph_off"]
    vals = [1.0 if (abl_dw and abl_dw[k] > 0) else 0.0 for k in ks]
    ax.bar(names, vals, color=["#2ca02c", "#7f7f7f", "#7f7f7f", "#7f7f7f"])
    ax.set_ylabel("Δw_l1 (weight change, read-back)")
    ax.set_title("(c) Ablations: judgment by Δw READ-BACK\n(ΔLI≡0 is structural, non-discriminative)")
    ax = axes[1, 1]
    ck = list(crit.keys())
    ax.barh(ck, [1 if crit[k] else 0 for k in ck],
            color=["#2ca02c" if crit[k] else "#d62728" for k in ck])
    ax.set_xlim(0, 1.2); ax.set_title("P5 criteria (in-band = 1)")
    ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("M9 P5 MB associative generalization (KC→MBON, real connectome subgraph)")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=110)


if __name__ == "__main__":
    sys.exit(main())
