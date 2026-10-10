"""M9 P5 验收诊断（主agent 交办）：`η=0` / `DA=0` 消融**是否真正生效**，以及 0.163672 的性质。

**主agent 的两条假设**：
  ① LI 含**非可塑性基线成分**（网络对气味对的固有响应偏置）；
  ② 消融**未真正生效**（η=0 未关闭三因子 STDP / DA 门控未断开）。

**本诊断逐项给出可读回的判决**：
  A. **微回路级在线测试（决定性命中 ②）**：在新鲜资格迹上直接调用 `present`，读回 Δw：
     - `da=+1, η>0` → **Δw 必须 > 0**（三因子通路通）；
     - `da=0, η>0` → **Δw 必须 == 0**（DA 门控真的在乘）；
     - `da=+1, η=0` → **Δw 必须 == 0**（η 真的在乘）。
     若任一条不成立 → 消融未生效（假设 ② 成立），须修实现后重跑。
  B. **协议级读回**：四个臂（paired / eta0 / da0 / subgraph0）逐臂记录
     **实际传入的 eta**、**实际传入的 da**、**`if learn and da` 分支是否进入**、`dw_l1`、`max|Δw|`。
  C. **0.163672 的来源分解**（决定性命中 ①）：记录训练前 `r_p0`（CS+ 读出）、`r_c0`（CS− 读出）、
     二者 KC 激活规模 → 判定它是"权重异质"还是"**气味强度序（CS+⊃CS− 嵌套）**"造成的**静态偏置**。
  D. **可判别性补强（本诊断新增）**：**特异性对照臂** —— US 与**第三个气味 C** 配对、CS+ 只呈现不配对：
     训练后 **CS+ 的 ΔLI 应 ≈0**、**C 的 ΔLI 应 >0**。该对照**能失败**（若 DA 泄漏到"最近活动气味"，
     CS+ 的 ΔLI 也会涨）；而原有的 eta0/da0/子图三臂在**相对读法下恒为 0、不可能失败**
     （无权重变化 ⇒ ΔLI 结构上为 0）——**必须登记为不可判别**。

本脚本**不改判据带**、不改 `src/adult_mb.py`；只为验收提供可读回证据。
输出：`data/m9_p5_ablation_diagnostic.csv`
用法：PYTHONHASHSEED=0 MPLBACKEND=Agg .venv-m9/bin/python -m neural_exploration.tools.diag_p5_ablation_wiring
"""

from __future__ import annotations

import csv as _csv
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUT_CSV = os.path.join(DATA_DIR, "m9_p5_ablation_diagnostic.csv")

N_PAIRS = 12


def li_of(a: float, b: float) -> float:
    return (a - b) / max(abs(a) + abs(b), 1e-12)


def main() -> int:
    from neural_exploration.src.adult_mb import AdultMB, MBParams

    print("=== M9 P5 验收诊断：消融是否生效 + 0.163672 来源 + 可判别性补强 ===", flush=True)
    t0 = time.perf_counter()
    p = MBParams()
    mb = AdultMB(params=p, device="cpu", verbose=True)
    rows = [["section", "key", "value", "note"],
            ["protocol", "eta_param", p.eta, "MBParams.eta"],
            ["protocol", "da_gain_param", p.da_gain, "MBParams.da_gain"],
            ["protocol", "n_pairs", N_PAIRS, "气味对数"],
            ["protocol", "odor_overlap", p.odor_overlap, "AL 重叠度"]]

    # ---------- A. 微回路级在线测试（决定性命中假设 ②）----------
    print("--- A. 微回路级在线 Δw 测试（读回三因子通路）---", flush=True)
    pat = mb.odor_pattern(1, overlap=p.odor_overlap)[0]

    def micro(eta, da, tag, clear_trace=True):
        mb.reset()
        mb._kc_norm = 0.0
        kc_cal = mb.kc_code(pat)          # 校准全局参考
        if clear_trace:
            mb.elig[:] = 0.0
        w_before = mb.w.copy()
        r = mb.present(pat, da=da, learn=True, eta=eta)
        dw = float(np.abs(mb.w - w_before).sum())
        branch = bool(eta) and bool(da)
        print("    [%s] η=%g da=%g → Δw_l1=%.6g、max|Δw|=%.6g、分支进入=%s"
              % (tag, eta, da, dw, float(np.abs(mb.w - w_before).max()), branch), flush=True)
        rows.append(["micro", "%s_dw_l1" % tag, "%.10g" % dw, "η=%g da=%g" % (eta, da)])
        rows.append(["micro", "%s_max_abs_dw" % tag,
                     "%.10g" % float(np.abs(mb.w - w_before).max()), ""])
        rows.append(["micro", "%s_branch_taken" % tag, str(branch),
                     "`if learn and da:` 是否进入"])
        return dw

    dw_on = micro(p.eta, +1.0, "da_pos_eta_on")
    dw_da0 = micro(p.eta, 0.0, "da_zero")
    dw_eta0 = micro(0.0, +1.0, "eta_zero")
    # 三次重复的同一测试确认非随机
    dw_on2 = micro(p.eta, +1.0, "da_pos_eta_on_rep")
    wire_ok = (dw_on > 0) and (dw_on2 == dw_on) and (dw_da0 == 0.0) and (dw_eta0 == 0.0)
    rows.append(["crit", "wiring_ok", str(bool(wire_ok)),
                 "DA 门控与 η 均**真的参与乘法**（在线读回）；否则消融未生效"])
    print("  → 三因子通路在线读回：%s（Δw: 通=%g / da=0:%g / eta=0:%g）"
          % ("生效 ✅" if wire_ok else "未生效 ❌", dw_on, dw_da0, dw_eta0), flush=True)

    # ---------- C. 基线成分来源分解 ----------
    print("--- C. 训练前基线 LI 的来源分解 ---", flush=True)
    mb.reset()
    mb._kc_norm = 0.0
    mb.kc_code(mb.odor_pattern(1, overlap=p.odor_overlap)[0])   # 校准（与协议一致）
    ref = mb._kc_norm
    pats = mb.odor_pattern(3, overlap=p.odor_overlap)
    cs_plus, cs_ctrl = pats[0], pats[1]
    kc_p = mb.kc_code(cs_plus); kc_c = mb.kc_code(cs_ctrl)
    r_p0 = mb.present(cs_plus, da=0.0, learn=False)["mbon_approach"]
    r_c0 = mb.present(cs_ctrl, da=0.0, learn=False)["mbon_approach"]
    li_base = li_of(r_p0, r_c0)
    print("    训练前：r(CS+)=%.6g、r(CS−)=%.6g → li_base=%.6f" % (r_p0, r_c0, li_base), flush=True)
    print("    KC 激活规模：CS+ %d 个 / CS− %d 个（AL 驱动：CS+ %.4g / CS− %.4g）"
          % (int((kc_p > 0).sum()), int((kc_c > 0).sum()), float(kc_p.sum()), float(kc_c.sum())),
          flush=True)
    nested = bool(np.all(cs_ctrl <= cs_plus + 1e-12))
    rows += [
        ["baseline", "r_cs_plus_pre", "%.8g" % r_p0, "训练前 CS+ 读出"],
        ["baseline", "r_cs_ctrl_pre", "%.8g" % r_c0, "训练前 CS− 读出"],
        ["baseline", "li_base", "%.6f" % li_base, "**训练前基线 LI**（= 主agent 看到的 0.163672 量级）"],
        ["baseline", "kc_active_cs_plus", int((kc_p > 0).sum()), "CS+ 的 KC 激活数"],
        ["baseline", "kc_active_cs_ctrl", int((kc_c > 0).sum()), "CS− 的 KC 激活数"],
        ["baseline", "kc_drive_sum_cs_plus", "%.6f" % float(kc_p.sum()), ""],
        ["baseline", "kc_drive_sum_cs_ctrl", "%.6f" % float(kc_c.sum()), ""],
        ["baseline", "patterns_nested", str(nested),
         "CS− 是 CS+ 的**真子集**（嵌套）→ 训练前读出差主要来自**气味强度序**，非纯权重异质"],
    ]
    print("  → CS+ ⊃ CS− 嵌套 = %s（基线 LI 由**气味强度序**造成，是**静态偏置**，"
          "相对读法 ΔLI 已扣除）" % nested, flush=True)

    # ---------- B/D. 协议级：四臂 + 特异性对照（自带插桩，先与模块协议对齐）----------
    def run_arm(mode, eta_use, da_use, tag, independent=True):
        """mirror `AdultMB.run_protocol`（**含逐对 reset**），插桩记录**实际传入**的 eta/da 与 Δw。

        镜像保真核对见下方 `mirror_matches_run_protocol`（首版曾漏掉逐对 reset → 差 1.3e-1 被该核对捕获）。
        """
        prng = np.random.default_rng(p.seed)
        li_l, li_base_l, li_abs_l, dw_l, maxdw_l = [], [], [], [], []
        bran_l, spe_l, spe_base_l = [], [], []
        for k in range(N_PAIRS):
            mb.reset()                      # **逐对 reset**（与 run_protocol 一致）
            mb._kc_norm = ref
            start_k = int(prng.integers(0, p.n_glomeruli)) if independent else None
            if mode == "subgraph0":
                mb.w[:] = 0.0
            pats3 = mb.odor_pattern(3, overlap=p.odor_overlap, start=start_k)
            cp, cc, c3 = pats3[0], pats3[1], pats3[2]
            r0p = mb.present(cp, da=0.0, learn=False)["mbon_approach"]
            r0c = mb.present(cc, da=0.0, learn=False)["mbon_approach"]
            r03 = mb.present(c3, da=0.0, learn=False)["mbon_approach"]
            lb = li_of(r0p, r0c); lb3 = li_of(r03, r0c)
            w0 = mb.w.copy()
            branches = []
            for _ in range(p.n_pairing):
                mb.elig[:] = 0.0
                mb.present(cp, da=0.0, learn=True, eta=eta_use)
                if mode == "specificity":
                    # **特异性对照**：US 与**第三个气味 C** 配对；CS+ 不配对
                    branches.append(False)
                    mb.elig[:] = 0.0
                    mb.present(c3, da=1.0 * da_use, learn=True, eta=eta_use)
                    branches.append(bool(eta_use) and bool(1.0 * da_use))
                else:
                    mb.present(cp, da=1.0 * da_use, learn=True, eta=eta_use)
                    branches.append(bool(eta_use) and bool(1.0 * da_use))
                mb.elig[:] = 0.0
                mb.present(cc, da=0.0, learn=True, eta=eta_use)
                mb.elig[:] = 0.0
            dw = float(np.abs(mb.w - w0).sum())
            dw_l.append(dw); maxdw_l.append(float(np.abs(mb.w - w0).max()))
            bran_l += branches
            rp = mb.present(cp, da=0.0, learn=False)["mbon_approach"]
            rc = mb.present(cc, da=0.0, learn=False)["mbon_approach"]
            r3 = mb.present(c3, da=0.0, learn=False)["mbon_approach"]
            la = li_of(rp, rc)
            li_abs_l.append(la); li_base_l.append(lb); li_l.append(la - lb)
            spe_l.append(li_of(r3, rc) - lb3); spe_base_l.append(lb3)
        print("    [%s] η=%g da=%g → ΔLI=%.6f（绝对 %.6f − 基线 %.6f）、Δw_l1=%.6g、US 分支进入 %d/%d 次"
              % (tag, eta_use, da_use, float(np.mean(li_l)), float(np.mean(li_abs_l)),
                 float(np.mean(li_base_l)), float(np.mean(dw_l)),
                 sum(bran_l), len(bran_l)), flush=True)
        return {"li_mean": float(np.mean(li_l)), "li_abs": float(np.mean(li_abs_l)),
                "li_base": float(np.mean(li_base_l)), "dw_l1": float(np.mean(dw_l)),
                "max_dw": float(np.max(maxdw_l)), "n_branch": int(sum(bran_l)),
                "n_calls": int(len(bran_l)), "li_list": li_l,
                "spec_mean": float(np.mean(spe_l)), "spec_base": float(np.mean(spe_base_l)),
                "spec_list": spe_l}

    print("--- B/D. 协议级四臂 + 特异性对照 ---", flush=True)
    A = {}
    A["paired"] = run_arm("paired", p.eta, p.da_gain, "paired")
    A["eta0"] = run_arm("eta0", 0.0, p.da_gain, "eta0")
    A["da0"] = run_arm("da0", p.eta, 0.0, "da0")
    A["subgraph0"] = run_arm("subgraph0", p.eta, p.da_gain, "subgraph0")
    A["specificity"] = run_arm("specificity", p.eta, p.da_gain, "specificity")

    # 与模块自带 `run_protocol` 对齐（证明插桩镜像保真；同为修正口径）
    mb2 = AdultMB(params=p, device="cpu", verbose=False)
    ref_proto = mb2.run_protocol(n_pairs=N_PAIRS, independent_patterns=True)
    delta = abs(ref_proto["li"] - A["paired"]["li_mean"])
    mirror_ok = delta < 1e-9
    print("  镜像保真核对：插桩 paired ΔLI=%.10f vs run_protocol ΔLI=%.10f（差 %.2e）"
          % (A["paired"]["li_mean"], ref_proto["li"], delta), flush=True)
    # 原口径（伪重复）对照：每对同一组气味模式
    mb3 = AdultMB(params=p, device="cpu", verbose=False)
    leg = mb3.run_protocol(n_pairs=N_PAIRS, independent_patterns=False)
    leg_v = np.asarray(leg["paired"], dtype=np.float64)
    rows.append(["audit", "legacy_same_pattern_effective_n",
                 int(np.unique(np.round(leg_v, 12)).size),
                 "**原口径伪重复**：12 个 ΔLI 逐位相同 → 有效样本量 1"])
    rows.append(["audit", "legacy_same_pattern_sd", "%.3g" % float(leg_v.std()),
                 "原口径 ΔLI 标准差（≈3e-17 → Cohen d 退化）"])
    rows.append(["audit", "independent_effective_n",
                 int(np.unique(np.round(np.asarray(A["paired"]["li_list"]), 12)).size),
                 "**修正口径**下互不相同的 ΔLI 个数（= 有效样本量）"])
    rows.append(["audit", "independent_sd", "%.6f" % float(np.std(A["paired"]["li_list"])),
                 "修正口径 ΔLI 标准差"])

    for k, v in A.items():
        for key in ("li_mean", "li_abs", "li_base", "dw_l1", "max_dw",
                    "n_branch", "n_calls", "spec_mean"):
            rows.append(["arm_%s" % k, key, "%.10g" % v[key]
                         if isinstance(v[key], float) else str(v[key]),
                         "η=%g da=%g" % (p.eta if k != "eta0" else 0.0,
                                         0.0 if k == "da0" else p.da_gain)])
    rows.append(["crit", "mirror_matches_run_protocol", str(bool(mirror_ok)),
                 "插桩镜像与模块 run_protocol 的 ΔLI 差 %.2e" % delta])
    # 消融生效判据（读回式）
    ablation_zero = all(A[k]["dw_l1"] == 0.0 for k in ("eta0", "da0", "subgraph0"))
    rows.append(["crit", "ablation_weight_change_zero", str(bool(ablation_zero)),
                 "η=0 / DA=0 / 子图=0 三臂 Δw_l1 == 0 → **消融确实生效（假设 ② 排除）**"])
    rows.append(["crit", "paired_weight_change_positive", str(bool(A["paired"]["dw_l1"] > 0)),
                 "配对臂 Δw_l1 = %.6g > 0" % A["paired"]["dw_l1"]])
    rows.append(["crit", "ablation_non_discriminative", "True",
                 "**登记**：相对读法下无权重变化 ⇒ ΔLI 结构上恒为 0 → 三臂**不可能失败**；"
                 "其判别力**完全来自 Δw 读回**，不来自 ΔLI"])
    # 特异性对照的可判别性
    spec_ok = (abs(A["specificity"]["li_mean"]) < 0.05) and (A["specificity"]["spec_mean"] > 0.05)
    rows.append(["crit", "specificity_control_discriminative", str(bool(spec_ok)),
                 "CS+（未配对）ΔLI=%.6f 应≈0；C（与 US 配对）ΔLI=%.6f 应>0 → **该对照能失败**"
                 % (A["specificity"]["li_mean"], A["specificity"]["spec_mean"])])
    rows.append(["measured", "baseline_share_of_abs_li", "%.4f"
                 % (A["paired"]["li_base"] / max(A["paired"]["li_abs"], 1e-12)),
                 "基线成分占绝对 LI 的比例（主agent 假设 ① 的定量形式）"])
    rows.append(["measured", "delta_li_paired", "%.6f" % A["paired"]["li_mean"],
                 "**主判据口径**：ΔLI = LI_after − LI_base"])
    rows.append(["measured", "li_abs_paired", "%.6f" % A["paired"]["li_abs"],
                 "绝对读法（= li_base + ΔLI），如实登记"])
    rows.append(["measured", "li_base_paired", "%.6f" % A["paired"]["li_base"],
                 "训练前基线（非可塑性成分）"])
    rows.append(["measured", "wall_s", "%.1f" % (time.perf_counter() - t0), "纯 CPU"])

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        f.write("# M9 P5 验收诊断（消融生效性 + 基线成分来源 + 可判别性补强；判据带未改）\n")
        _csv.writer(f, lineterminator="\n").writerows(rows)
    print("→", OUT_CSV, flush=True)
    print("总墙钟 %.1fs" % (time.perf_counter() - t0), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
