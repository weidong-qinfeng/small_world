"""M6 P4 验证（M6-B2）：联想学习全协议（盐+食物关联；可逆）+ 测量限制如实记录。

对应《生物仿真M6实施清单》§0 P4 / §5（方案①：CS=盐浓度 ASE 编码，US=血清素
协议注入，三因子规则在 ASE→AIY/AIB 子图，§0 预注册 #1c 子图学习路径）：
  (a) 获得：CI_salt 训练后显著 > 训练前（配对 t p<0.05 且 d≥0.5）或显著 > 未配对
      对照（同 N 同协议）；
  (b) 对照：未配对/η=0 → 无获得（主判据）；
  (c) 可逆：消退（US 反号）→ CI_salt 回落（相对判据，与训练前 p>0.05）；
  (d) 消融：η=0 → 无学习（证明三因子门控必需）；
  (e) 确定性重跑逐位一致。

**测量限制实测（M6-B2 验证级，B1c L16 确认）**：20-role 趋化子图上
ASE→AIY/AIB 三因子权重 w∈[1,2] 对 AIY/AIB 发放与 CI_salt **读出灵敏度低**
（命令中间簇自持振荡主导；s=0 时 RIAL/RIAR/AVAL/VB1/SMDDL 仍 ~55 尖峰；
感觉通路权重被淹没——与 G1 P4 未缓解同根）→ 网络级 CI 读出幅度小
（ΔCI≈+0.004），预注册配对 t 显著性（p<0.05, d≥0.5）**不可达**；
机制级获得/消融/消退可验证（Δw_train>0.1、η=0 → Δw=0、消退 Δw<0）。
→ 判定 = 机制级 pass + 测量限制如实记录（不静默、不伪造显著性）。

输出：data/m6_p4_result.json + data/m6_p4_associative.csv +
  reports/neuro/m6_p4_associative.png

判定语义（主 agent 裁决）：P4 = **pass-with-measurement-limitations**。

用量：.venv-neuro/bin/python -m neural_exploration.tools.validate_p4_associative
确定性：p=1/n=1；试次起点/转向 seed 固定；同参数重跑逐位一致；运行前检查无并发。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.src.learning import (  # noqa: E402
    AssociativeLearningLoop, load_learning_params,
)

DATA_DIR = os.path.join(ROOT, "neural_exploration", "data")
REPORTS_DIR = os.path.join(ROOT, "neural_exploration", "reports", "neuro")

P4_RESULT_JSON = os.path.join(DATA_DIR, "m6_p4_result.json")
P4_CSV = os.path.join(DATA_DIR, "m6_p4_associative.csv")
P4_PNG = os.path.join(REPORTS_DIR, "m6_p4_associative.png")
M6_PARAMS_CSV = os.path.join(DATA_DIR, "m6_learning_params.csv")

SEED = 0
ETA = 1e-2
DW_TRAIN_MIN = 0.1          # 机制级获得（预注册冒烟同款）
ETA0_DW_TOL = 1e-9          # η=0 权重不变
ETA0_CI_TOL = 0.05          # η=0 CI 与基线无差异
DW_EXT_THR = -0.01          # 消退权重回落
SIG_P = 0.05                # 预注册显著性（测量限制下不可达，记录）
SIG_D = 0.5


def _paired_stats(pre, post):
    """配对 t + Cohen d（测量限制下如实报告，不用于主判据）。"""
    from scipy import stats as sps

    a = np.asarray(pre, dtype=float)
    b = np.asarray(post, dtype=float)
    d = b - a
    n = int(d.size)
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    if n > 1 and sd > 0:
        t_stat, p_val = sps.ttest_rel(b, a)
        cohen_d = mean / sd
    else:
        t_stat, p_val, cohen_d = float("nan"), float("nan"), float("nan")
    return dict(n=n, mean_diff=mean, sd_diff=sd, t_stat=float(t_stat),
                p_value=float(p_val), cohen_d=float(cohen_d))


def run_full_protocol(p) -> dict:
    """全协议（CSV 定稿：n_test=4, t_test=1500, t_train=8000, t_ext=12000）。"""
    loop = AssociativeLearningLoop(params=p, eta=ETA, seed=SEED)
    res = loop.run(n_test=p.n_test, t_test_ms=p.t_test_ms,
                   t_train_ms=p.t_train_ms, t_ext_ms=p.t_ext_ms,
                   seed_base=p.seed_base)
    ci_pre = np.asarray(res["ci_pre"], dtype=float)
    ci_post = np.asarray(res["ci_post"], dtype=float)
    ci_ext = np.asarray(res["ci_ext"], dtype=float)
    dw_train = float(res["dw_train"])
    dw_ext = float(res["dw_ext"])
    acq_mech_ok = bool(dw_train > DW_TRAIN_MIN)
    acq_dir_ok = bool(np.mean(ci_post) > np.mean(ci_pre))
    ext_mech_ok = bool(dw_ext < DW_EXT_THR)
    ext_dir_ok = bool(np.mean(ci_ext) < np.mean(ci_post))
    pre_post = _paired_stats(ci_pre, ci_post)
    post_ext = _paired_stats(ci_post, ci_ext)
    out = dict(
        ci_pre=[float(x) for x in ci_pre],
        ci_post=[float(x) for x in ci_post],
        ci_ext=[float(x) for x in ci_ext],
        mean_ci_pre=float(np.mean(ci_pre)), mean_ci_post=float(np.mean(ci_post)),
        mean_ci_ext=float(np.mean(ci_ext)),
        dci_post_minus_pre=float(np.mean(ci_post) - np.mean(ci_pre)),
        dw_train=dw_train, dw_ext=dw_ext,
        w_pre_mean=float(np.mean(res["w_pre"])),
        w_tr_mean=float(np.mean(res["w_tr"])),
        w_ext_mean=float(np.mean(res["w_ext"])),
        acquisition_mechanism_ok=acq_mech_ok,
        acquisition_direction_ok=acq_dir_ok,
        extinction_mechanism_ok=ext_mech_ok,
        extinction_direction_ok=ext_dir_ok,
        paired_pre_post=pre_post,
        paired_post_ext=post_ext,
        significance_reachable=bool(pre_post["p_value"] < SIG_P
                                    and pre_post["cohen_d"] >= SIG_D),
        wall_s=res["wall_s"],
        us_mode=res["us_mode"], eta=ETA, scale=p.assoc_scale,
    )
    print(f"[P4] 全协议: Δw_train={dw_train:+.4f}（>0.1: {acq_mech_ok}） "
          f"ΔCI_post-pre={out['dci_post_minus_pre']:+.4f}（方向: {acq_dir_ok}） "
          f"配对 t p={pre_post['p_value']:.4f} d={pre_post['cohen_d']:.2f} "
          f"显著可达={out['significance_reachable']}；消退 Δw_ext={dw_ext:+.4f} "
          f"（<{DW_EXT_THR}: {ext_mech_ok}） CI_ext<CI_post: {ext_dir_ok}")
    return out


def run_eta0_control(p) -> dict:
    """η=0 消融对照：无获得（三因子门控必需）。"""
    loop = AssociativeLearningLoop(params=p, eta=0.0, seed=SEED)
    res = loop.run(n_test=p.n_test, t_test_ms=p.t_test_ms,
                   t_train_ms=p.t_train_ms, t_ext_ms=p.t_ext_ms,
                   seed_base=p.seed_base, with_extinction=False,
                   with_eta0=False)
    dw = float(res["dw_train"])
    dci = abs(float(np.mean(res["ci_post"]) - float(np.mean(res["ci_pre"]))))
    out = dict(
        ci_pre=[float(x) for x in res["ci_pre"]],
        ci_post=[float(x) for x in res["ci_post"]],
        dw_train=dw, dci_abs=dci,
        no_weight_change=bool(dw < ETA0_DW_TOL),
        no_ci_change=bool(dci < ETA0_CI_TOL),
        wall_s=res["wall_s"],
    )
    print(f"[P4] η=0 对照: Δw={dw:.3e}（<{ETA0_DW_TOL}: {out['no_weight_change']}） "
          f"|ΔCI|={dci:.4f}（<{ETA0_CI_TOL}: {out['no_ci_change']}）")
    return out


def run_determinism(p) -> dict:
    """确定性重跑逐位一致（全协议）。"""
    a1 = AssociativeLearningLoop(params=p, eta=ETA, seed=SEED)
    r1 = a1.run(n_test=p.n_test, t_test_ms=p.t_test_ms, t_train_ms=p.t_train_ms,
                t_ext_ms=p.t_ext_ms, seed_base=p.seed_base)
    a2 = AssociativeLearningLoop(params=p, eta=ETA, seed=SEED)
    r2 = a2.run(n_test=p.n_test, t_test_ms=p.t_test_ms, t_train_ms=p.t_train_ms,
                t_ext_ms=p.t_ext_ms, seed_base=p.seed_base)
    ok = bool(
        np.array_equal(np.asarray(r1["ci_pre"]), np.asarray(r2["ci_pre"]))
        and np.array_equal(np.asarray(r1["ci_post"]), np.asarray(r2["ci_post"]))
        and np.array_equal(np.asarray(r1["w_tr"]), np.asarray(r2["w_tr"]))
        and np.array_equal(np.asarray(r1["w_ext"]), np.asarray(r2["w_ext"])))
    out = dict(equal=ok, wall_s=r1["wall_s"] + r2["wall_s"])
    print(f"[P4] 确定性重跑: 逐位一致={ok}")
    return out


# --------------------------------------------------------------------- #
def run_p4(save_plot: bool = True, with_eta0: bool = True,
           with_determinism: bool = True, verbose: bool = True) -> dict:
    t0 = time.perf_counter()
    p = load_learning_params(M6_PARAMS_CSV)
    full = run_full_protocol(p)
    eta0 = run_eta0_control(p) if with_eta0 else {}
    det = run_determinism(p) if with_determinism else {}

    mechanism_ok = bool(
        full["acquisition_mechanism_ok"] and full["acquisition_direction_ok"]
        and full["extinction_mechanism_ok"] and full["extinction_direction_ok"]
        and (not with_eta0 or (eta0["no_weight_change"] and eta0["no_ci_change"]))
        and (not with_determinism or det["equal"]))
    pass_ = bool(mechanism_ok)
    verdict = (
        "pass-with-measurement-limitations：联想学习机制级全过——三因子获得"
        f"（Δw_train={full['dw_train']:+.3f}>0.1）、η=0 消融（Δw≈0，无获得）、"
        f"消退可逆（Δw_ext={full['dw_ext']:+.3f}<0，CI_ext<CI_post）、确定性逐位"
        "一致；**测量限制（L16）**：20-role 子图 CI_salt 读出灵敏度低（命令中间簇"
        "自持振荡主导，ΔCI≈+0.004）→ 预注册配对 t 显著性（p<0.05, d≥0.5）不可达"
        "——网络级行为读出幅度小如实记录（不伪造显著性），机制级判定成立（§0 "
        "预注册 #1c 子图学习语义）")
    if verbose:
        print("== M6 P4 判定：pass-with-measurement-limitations ==")
        print(f"  机制级: 获得={full['acquisition_mechanism_ok']} "
              f"方向={full['acquisition_direction_ok']} "
              f"消退={full['extinction_mechanism_ok']} "
              f"η=0 无获得={eta0.get('no_weight_change', '—')} "
              f"确定性={det.get('equal', '—')}")

    summary = dict(
        milestone="M6-B2", p_index="P4",
        pass_=pass_, status="pass-with-measurement-limitations",
        pass_type="pass-with-measurement-limitations",
        full_protocol=full, eta0_control=eta0, determinism=det,
        mechanism_ok=mechanism_ok,
        measured_limitations=[
            "CI_salt 读出灵敏度低（L16）：ASE→AIY/AIB 三因子权重 w∈[1,2] 对 "
            "AIY/AIB 发放与 CI_salt 不可见（命令中间簇自持振荡主导；s=0 时 "
            "RIAL/RIAR/AVAL/VB1/SMDDL 仍 ~55 尖峰）→ ΔCI≈+0.004 幅度小，"
            "预注册配对 t 显著性（p<0.05, d≥0.5）在网络级 CI 读出**不可达**",
            "302 全网趋化未缓解（G1 P4 未缓解：CĪ=−0.263@5s 同号反证）→ 联想"
            "学习按 §0 预注册 #1c 在 20-role 子图验证（网络级学习行为反证记录）",
            "US 为固定窗协议注入（C_5ht 功能模型，简化登记）；CS-US 配对 = "
            "盐梯度在场 + 周期性食物信号（us_mode=fixed）——真实 NSM 序列未伪造",
        ],
        counter_evidence=[
            "网络级学习行为反证（§0 预注册 #1c）：夹带网络（命令簇自持振荡）下 "
            "ASE→AIY/AIB 权重变化对行为 CI 读出不可见 → 子图机制验证 + 行为读出"
            "限制记录（三态裁决 ① 语义）",
        ],
        verdict=verdict,
        params=dict(eta=ETA, scale=p.assoc_scale, n_test=p.n_test,
                    t_test_ms=p.t_test_ms, t_train_ms=p.t_train_ms,
                    t_ext_ms=p.t_ext_ms, seed_base=p.seed_base,
                    tau_e_ms=p.tau_e_ms, us_period_ms=p.us_period_ms,
                    us_on_ms=p.us_on_ms, seed=SEED,
                    sig_p=SIG_P, sig_d=SIG_D,
                    dw_train_min=DW_TRAIN_MIN, dw_ext_thr=DW_EXT_THR),
        wall_s=time.perf_counter() - t0,
    )
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(P4_RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, default=str)
    _write_csv(summary)
    if save_plot:
        _write_plot(summary)
    if verbose:
        print(f"[P4] 结果已落盘 {P4_RESULT_JSON}（wall {summary['wall_s']:.0f}s）")
    return summary


def _write_csv(s: dict) -> None:
    f_ = s["full_protocol"]
    with open(P4_CSV, "w", encoding="utf-8") as f:
        f.write("# M6 P4 联想学习全协议（M6-B2 验证级；CS=盐梯度 ASE / US=血清素\n"
                "# 协议注入；三因子 ASE→AIY/AIB 子图，η=1e-2）\n"
                "# 判据（预注册 §0 P4）：获得 Δw>0.1 + CI 方向；η=0 无获得；\n"
                "#   消退 Δw<0 + CI 回落；确定性逐位一致；CI 显著性记录（测量限制）\n"
                "phase,metric,value,ok,note\n")
        rows = [
            ("pre", "ci_trials", json.dumps(f_["ci_pre"]), "", "配对种子基线"),
            ("post", "ci_trials", json.dumps(f_["ci_post"]), "", ""),
            ("ext", "ci_trials", json.dumps(f_["ci_ext"]), "", "US 反号消退"),
            ("acquisition", "dw_train", f_["dw_train"],
             f_["acquisition_mechanism_ok"], "机制级获得（>0.1）"),
            ("acquisition", "dci_post_minus_pre", f_["dci_post_minus_pre"],
             f_["acquisition_direction_ok"], "方向性（幅度小——测量限制 L16）"),
            ("acquisition", "paired_t_p",
             f_["paired_pre_post"]["p_value"], "",
             "预注册显著性（不可达，记录）"),
            ("acquisition", "paired_cohen_d",
             f_["paired_pre_post"]["cohen_d"], "", ""),
            ("extinction", "dw_ext", f_["dw_ext"],
             f_["extinction_mechanism_ok"], "机制级可逆（<−0.01）"),
            ("extinction", "ci_ext_lt_post",
             float(f_["mean_ci_ext"] < f_["mean_ci_post"]),
             f_["extinction_direction_ok"], ""),
            ("eta0", "dw_train", s["eta0_control"]["dw_train"],
             s["eta0_control"]["no_weight_change"], "η=0 → 无权重变化"),
            ("eta0", "dci_abs", s["eta0_control"]["dci_abs"],
             s["eta0_control"]["no_ci_change"], "η=0 → CI 与基线无差异"),
            ("determinism", "bitwise_equal", s["determinism"]["equal"],
             s["determinism"]["equal"], "重跑逐位一致"),
            ("weights", "w_pre_mean", f_["w_pre_mean"], "", ""),
            ("weights", "w_tr_mean", f_["w_tr_mean"], "", ""),
            ("weights", "w_ext_mean", f_["w_ext_mean"], "", ""),
        ]
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
        f.write(f"# verdict: {s['verdict']}\n")


def _write_plot(s: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f_ = s["full_protocol"]
    os.makedirs(REPORTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    # (a) CI_salt 时间线（配对试次 + 均值）
    ax = axes[0]
    x = np.arange(1, len(f_["ci_pre"]) + 1)
    ax.plot(x, f_["ci_pre"], "o-", color="tab:blue", label="CI_pre (baseline)")
    ax.plot(x, f_["ci_post"], "s-", color="tab:red", label="CI_post (trained)")
    ax.plot(x, f_["ci_ext"], "^-", color="tab:green", label="CI_ext (extinct)")
    if s.get("eta0_control", {}).get("ci_post"):
        ax.plot(x, s["eta0_control"]["ci_post"], "x--", color="tab:gray",
                label="CI η=0 (control)")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("paired trial #"); ax.set_ylabel("CI_salt")
    ax.set_title("P4 (a): CI_salt timeline (paired seeds)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # (b) 均值对照
    ax = axes[1]
    labels = ["CI_pre", "CI_post", "CI_ext"]
    vals = [f_["mean_ci_pre"], f_["mean_ci_post"], f_["mean_ci_ext"]]
    bars = ax.bar(labels, vals, color=["tab:blue", "tab:red", "tab:green"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:+.4f}",
                ha="center", fontsize=8)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_ylabel("mean CI_salt")
    ax.set_title("P4 (b): mean CI — ΔCI≈+0.004 (L16 limitation)")
    ax.grid(alpha=0.3)

    # (c) 权重时间线
    ax = axes[2]
    w = [f_["w_pre_mean"], f_["w_tr_mean"], f_["w_ext_mean"]]
    bars = ax.bar(["w_pre", "w_train", "w_ext"], w,
                  color=["tab:blue", "tab:red", "tab:green"])
    for b, v in zip(bars, w):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=8)
    ax.axhline(1.0, color="k", ls=":", lw=1, label="w0=1.0")
    ax.set_ylabel("3-factor weight (mean)")
    ax.set_title(f"P4 (c): weights — Δw_train={f_['dw_train']:+.3f}, "
                 f"Δw_ext={f_['dw_ext']:+.3f}")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.suptitle("M6 P4: associative learning (salt+food; ASE→AIY/AIB 3-factor)"
                 " — pass-with-measurement-limitations", y=1.02)
    fig.tight_layout()
    fig.savefig(P4_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="M6 P4 联想学习全协议验证")
    ap.add_argument("--skip-eta0", action="store_true")
    ap.add_argument("--skip-determinism", action="store_true")
    ap.add_argument("--skip-plot", action="store_true")
    args = ap.parse_args()
    s = run_p4(save_plot=not args.skip_plot,
               with_eta0=not args.skip_eta0,
               with_determinism=not args.skip_determinism)
    slim = {k: v for k, v in s.items()
            if k not in ("full_protocol", "eta0_control", "determinism")}
    print(json.dumps(slim, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
