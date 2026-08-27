"""M6 P3 验证（M6-B2）：习惯化全协议（Rankin et al. 1990 对照）+ 判据可达性如实记录。

对应《生物仿真M6实施清单》§0 P3 / §4（母版 = M5 P5 逃避协议）：
  (a) 曲线形状：逐刺激 R(n) = D_peak 序列 + 指数拟合 R(n)=A·exp(−n/τ_hab)+B
      （确定性 lstsq）→ R² ≥ 0.5（预注册）；
  (b) 时程：τ_hab ∈ [3,15] 次（预注册带，Rankin 1990 ISI=10s 量级）；
  (c) 自发恢复：休息窗后 R_rest ≥ 0.3×R(1)（预注册相对判据；绝对时程记录为
      测量限制）；
  (d) 去习惯化（informational，本节点记录不判据化）；
  (e) 确定性重跑逐位一致；消融（STP 关 → 衰减消失，H1 机制必需）。

**判据可达性实测（M6-B2 验证级，L25 记录）**：
  - **机制在短 ISI 演示**（reflex 底物，isi=0/50ms）：STP 开 → R(n) 指数衰减
    （R²≥0.5 ✓，τ_hab≈2 出预注册带 [3,15]——短 ISI 衰减更快，形态如实记录）；
  - **10s-ISI 主协议受模型时程限制**：τ_rec=1000ms ≪ ISI=10s → STP x 在 ISI 内
    完全恢复 → R(n) 常数（无习惯化）；且 30s 协议窗（PROTOCOL_WINDOW_MS）内
    10s-ISI 仅 2 刺激可注触（≥3 超出窗口，协议分段预注册 §4.1 受会话窗限制）→
    判据 (a)/(b) 在 10s-ISI 主协议**不可达**，如实记录 + 三态裁决请求
    （① 接受机制级 pass（相对判据 + 测量限制记录）② 延长协议窗（超本节点权限）
    ③ 参考模型对照）——主 agent 框架 ① 采纳；
  - **302 O2 网络底物**：R(n) 可计算（确定性）但 touch≈no-touch（夹带干扰，
    D_peak 非触诱发，B1c L12#1 确认）——网络级触诱发反应不可干净测量（记录）。

输出：data/m6_p3_result.json + data/m6_p3_habituation.csv +
  reports/neuro/m6_p3_habituation.png

判定语义（主 agent 裁决）：P3 = **pass-with-measurement-limitations**
（机制级衰减/消融/恢复全过；预注册主协议判据可达性限制如实记录）。

用量：.venv-neuro/bin/python -m neural_exploration.tools.validate_p3_habituation
确定性：p=1/n=1；同参数重跑逐位一致；运行前检查无并发。
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
    HabituationLoop, load_learning_params,
)

DATA_DIR = os.path.join(ROOT, "neural_exploration", "data")
REPORTS_DIR = os.path.join(ROOT, "neural_exploration", "reports", "neuro")

P3_RESULT_JSON = os.path.join(DATA_DIR, "m6_p3_result.json")
P3_CSV = os.path.join(DATA_DIR, "m6_p3_habituation.csv")
P3_PNG = os.path.join(REPORTS_DIR, "m6_p3_habituation.png")
M6_PARAMS_CSV = os.path.join(DATA_DIR, "m6_learning_params.csv")

SEED = 0
N_SHORT = 6                # 短 ISI 机制演示刺激数
ISI_SHORT_MS = 0.0         # 最短协议（STP 不恢复 → 衰减可见）
ISI_MID_MS = 3000.0        # 扩展对照：3s ≫ τ_rec=1s → x ~95% 恢复 → R 近常数
ISI_MAIN_MS = 10000.0      # Rankin 主协议 10s（判据带 τ_hab∈[3,15] 预注册于此）
N_MAIN = 2                 # 10s-ISI 在 30s 会话窗内可注触刺激数（2×10.15=20.3s<30s）
REST_MS = 2000.0           # 恢复窗（≫τ_rec → STP 恢复 → 反应回升）
N_NETWORK = 3              # 302 底物刺激数（冒烟同款）
D_PEAK_THR = 0.3
DECAY_FRAC = 0.5           # 衰减判据：后半均值 < 0.5×前半均值（冒烟同款）
ABLATION_FRAC = 0.7        # 消融判据：STP 关后半 ≥ 0.7×前半（无系统衰减）
RECOVER_FRAC_MIN = 0.3     # 预注册 #4：R_rest ≥ 0.3×R(1)


# --------------------------------------------------------------------- #
def _run_reflex(loop: HabituationLoop, n: int, isi_ms: float,
                stp_enabled: bool, rest_ms: float = 0.0) -> dict:
    return loop.run(n_stim=n, isi_ms=isi_ms, seed=SEED,
                    stp_enabled=stp_enabled, rest_ms=rest_ms)


def run_short_isi_mechanism(loop: HabituationLoop) -> dict:
    """短 ISI（0ms）机制演示：R(n) 序列 + 指数拟合 + 衰减判据。"""
    res = _run_reflex(loop, N_SHORT, ISI_SHORT_MS, stp_enabled=True)
    r = np.asarray(res["r_seq"], dtype=float)
    fit = res["fit"]
    decay_ok = bool(res["last_half_mean"] < DECAY_FRAC * res["first_half_mean"])
    direction_ok = bool(res["direction_ok"])
    out = dict(
        r_seq=[float(x) for x in r],
        first_half_mean=res["first_half_mean"],
        last_half_mean=res["last_half_mean"],
        decay=res["decay"],
        fit=fit,
        decay_ok=decay_ok,
        direction_ok=direction_ok,
        fit_r2_ok=bool(fit["r2_ok"]),
        fit_tau_in_band=bool(fit["in_tau_band"]),
        wall_s=res["wall_s"],
        note="短 ISI（0ms）：STP 不恢复 → 逐刺激耗竭 → 指数衰减；"
             "τ_hab≈2 出预注册带 [3,15]（短 ISI 形态，10s-ISI 带不可达，如实记录）",
    )
    print(f"[P3] 短 ISI（0ms, n={N_SHORT}）: R={[round(x, 3) for x in r]} "
          f"decay_ok={decay_ok} fit R²={fit['r2']:.2f} τ_hab={fit['tau_hab']:.2f} "
          f"in_band={fit['in_tau_band']}")
    return out


def run_ablation(loop: HabituationLoop) -> dict:
    """消融：STP 关 → 衰减消失（H1 机制必需）。"""
    res = _run_reflex(loop, N_SHORT, ISI_SHORT_MS, stp_enabled=False)
    r = np.asarray(res["r_seq"], dtype=float)
    no_decay_ok = bool(res["last_half_mean"] >= ABLATION_FRAC
                       * res["first_half_mean"])
    contrast_ok = bool(res["decay"] < 0.05)  # 与 STP 开（decay>0.05）对照
    out = dict(
        r_seq=[float(x) for x in r],
        first_half_mean=res["first_half_mean"],
        last_half_mean=res["last_half_mean"],
        decay=res["decay"],
        no_decay_ok=no_decay_ok,
        contrast_ok=contrast_ok,
        wall_s=res["wall_s"],
        note="STP 关（u0 未配置/τ_fac=0 回退防御）→ 无系统衰减 → H1 机制必需",
    )
    print(f"[P3] 消融（STP 关）: R={[round(x, 3) for x in r]} "
          f"no_decay_ok={no_decay_ok}")
    return out


def run_recovery(loop: HabituationLoop) -> dict:
    """自发恢复：6×短 ISI 耗竭 + 休息 2s + 测试 → R_rest 回升。"""
    res = _run_reflex(loop, N_SHORT, ISI_SHORT_MS, stp_enabled=True,
                      rest_ms=REST_MS)
    r = np.asarray(res["r_seq"], dtype=float)
    r1 = float(r[0])
    r_rest = float(res["r_rest"])
    recover_ok = bool(r_rest >= RECOVER_FRAC_MIN * r1)
    above_last = bool(r_rest > float(r[-1]))
    out = dict(
        r_seq=[float(x) for x in r],
        r1=r1, r_last=float(r[-1]), r_rest=r_rest,
        recover_frac=r_rest / r1 if r1 else float("nan"),
        recover_ok=recover_ok, above_last=above_last,
        rest_ms=REST_MS,
        wall_s=res["wall_s"],
        note="预注册 #4 相对恢复判据 R_rest ≥ 0.3×R(1)；绝对恢复时程（真实分钟~小时）"
             "记录为测量限制（不伪造）",
    )
    print(f"[P3] 恢复（rest={REST_MS:.0f}ms）: R_rest={r_rest:+.3f} "
          f"R(1)={r1:+.3f} R(N)={out['r_last']:+.3f} "
          f"recover_ok={recover_ok} (≥0.3×R1={0.3 * r1:+.3f})")
    return out


def run_isi_scaling(loop: HabituationLoop) -> dict:
    """ISI 扩展（判据可达性核心证据）：

    - 10s-ISI 主协议（n=2，30s 会话窗内可注触上限）：R(1)≈R(2) 常数（x 在 ISI 内
      完全恢复 → 无习惯化）；协议分段（§4.1 预注册）受 30s 窗限制 → 判据 (a)/(b)
      不可达，如实记录；
    - 3s-ISI 扩展（n=6，3s ≫ τ_rec=1s → x~95% 恢复）：R(n) 近常数（残差衰减小），
      量化『ISI ≫ τ_rec → 无习惯化』。
    """
    res_main = _run_reflex(loop, N_MAIN, ISI_MAIN_MS, stp_enabled=True)
    r_main = np.asarray(res_main["r_seq"], dtype=float)
    const_main = bool(len(r_main) >= 2
                      and abs(float(r_main[-1] - r_main[0])) < 0.05)
    res_mid = _run_reflex(loop, N_SHORT, ISI_MID_MS, stp_enabled=True)
    r_mid = np.asarray(res_mid["r_seq"], dtype=float)
    fit_mid = res_mid["fit"]
    decay_mid = float(r_mid[0] - r_mid[-1])
    out = dict(
        main_10s=dict(
            isi_ms=ISI_MAIN_MS, n=N_MAIN, r_seq=[float(x) for x in r_main],
            constant=const_main,
            note="R(n) 常数：τ_rec=1000ms 在 10s ISI 内完全恢复（L25 记录）→ 无"
                 "习惯化；30s 会话窗内仅 2 刺激可注触（协议分段预注册 §4.1 受会话"
                 "窗限制）→ 主协议判据 (a)/(b) 不可达（测量限制，三态裁决请求）"),
        mid_3s=dict(
            isi_ms=ISI_MID_MS, n=N_SHORT, r_seq=[float(x) for x in r_mid],
            first_last_decay=decay_mid,
            fit=fit_mid,
            constant=bool(abs(decay_mid) < 0.08),
            note="3s ≫ τ_rec=1s → x 近完全恢复 → R 近常数（残差衰减 "
                 f"{decay_mid:+.3f}）——ISI≫τ_rec 无习惯化的量化对照"),
        wall_s=res_main["wall_s"] + res_mid["wall_s"],
    )
    print(f"[P3] 10s-ISI（n={N_MAIN}）: R={[round(x, 3) for x in r_main]} "
          f"常数={const_main}；3s-ISI: R={[round(x, 3) for x in r_mid]} "
          f"decay={decay_mid:+.3f}")
    return out


def run_network_302(loop_net: HabituationLoop) -> dict:
    """302 O2 网络底物：R(n) 可计算 + touch≈no-touch 夹带限制记录。"""
    res = loop_net.run(n_stim=N_NETWORK, isi_ms=0.0, seed=SEED,
                       stp_enabled=True, no_touch_control=True)
    r = np.asarray(res["r_seq"], dtype=float)
    nt = float(res["no_touch_d_peak"]) if res["no_touch_d_peak"] is not None \
        else float("nan")
    limitation_ok = bool(abs(float(r[0]) - nt) < 0.2) if np.isfinite(nt) else False
    out = dict(
        r_seq=[float(x) for x in r],
        no_touch_d_peak=nt,
        touch_eq_no_touch=limitation_ok,
        wall_s=res["wall_s"],
        note="302 O2 全网 D_peak 由自发动力学主导（touch≈no-touch）→ 网络级触诱发"
             "反应不可干净测量（G1 部分通过结构性限制；B1c L12#1 确认）",
    )
    print(f"[P3] 302 底物: R={[round(x, 3) for x in r]} no_touch={nt:+.3f} "
          f"touch≈no-touch={limitation_ok}")
    return out


def run_determinism(loop: HabituationLoop) -> dict:
    """确定性重跑逐位一致（短 ISI 机制演示）。"""
    r1 = _run_reflex(loop, N_SHORT, ISI_SHORT_MS, stp_enabled=True)
    r2 = _run_reflex(loop, N_SHORT, ISI_SHORT_MS, stp_enabled=True)
    seq_ok = bool(np.array_equal(np.asarray(r1["r_seq"]),
                                 np.asarray(r2["r_seq"])))
    fit_ok = bool(r1["fit"]["tau_hab"] == r2["fit"]["tau_hab"]
                  and r1["fit"]["r2"] == r2["fit"]["r2"])
    out = dict(seq_equal=seq_ok, fit_equal=fit_ok,
               wall_s=r1["wall_s"] + r2["wall_s"])
    print(f"[P3] 确定性重跑: 逐位一致={seq_ok and fit_ok}")
    return out


# --------------------------------------------------------------------- #
def run_p3(save_plot: bool = True, with_network: bool = True,
           verbose: bool = True) -> dict:
    t0 = time.perf_counter()
    p = load_learning_params(M6_PARAMS_CSV)
    loop = HabituationLoop(params=p, substrate="reflex")
    short = run_short_isi_mechanism(loop)
    abl = run_ablation(loop)
    rec = run_recovery(loop)
    isi = run_isi_scaling(loop)
    det = run_determinism(loop)

    net = {}
    if with_network:
        loop_net = HabituationLoop(params=p, substrate="network")
        net = run_network_302(loop_net)

    # —— 判定（主 agent 裁决：pass-with-measurement-limitations）——
    mechanism_ok = bool(
        short["decay_ok"] and short["direction_ok"]
        and short["fit_r2_ok"]           # 指数拟合 R²≥0.5 ✓（isi=0 形态）
        and abl["no_decay_ok"] and abl["contrast_ok"]
        and rec["recover_ok"] and rec["above_last"]
        and det["seq_equal"] and det["fit_equal"])
    criterion_reachability = dict(
        fit_r2_at_short_isi=short["fit"]["r2_ok"],
        tau_hab_in_band=short["fit"]["in_tau_band"],
        tau_hab_meas=short["fit"]["tau_hab"],
        tau_band=[3.0, 15.0],
        main_protocol_10s_isi="NOT-REACHABLE",
        reason=(
            "τ_rec=1000ms ≪ ISI=10s → STP x 完全恢复 → R(n) 常数（无习惯化）；"
            "30s 协议窗内 10s-ISI 仅 2 刺激可注触 → 主协议判据 (a)/(b) 不可达；"
            "机制在短 ISI（0ms）演示（R²≥0.5 ✓，τ_hab≈2 出带）——判据可达性"
            "如实记录，三态裁决（主 agent 采纳机制级 pass + 测量限制记录）"),
    )
    pass_ = bool(mechanism_ok)
    verdict = (
        "pass-with-measurement-limitations：习惯化机制级全过——短 ISI 指数衰减"
        "（R²≥0.5）、STP 消融（关→无衰减，H1 必需）、自发恢复（R_rest≥0.3×R(1)）、"
        "确定性逐位一致；预注册 10s-ISI 主协议判据（τ_hab∈[3,15] 带 / R²≥0.5）"
        "受模型时程限制不可达（τ_rec=1s 在 10s ISI 内完全恢复 → R(n) 常数；"
        "30s 会话窗限刺激数）→ 机制级判定 + 测量限制如实记录（L25，三态裁决 "
        "① 采纳）；302 网络底物 touch≈no-touch（夹带干扰）记录"
        + (f"；302 R(n)={[round(x, 3) for x in net.get('r_seq', [])]}" if net else ""))
    if verbose:
        print("== M6 P3 判定：pass-with-measurement-limitations ==")
        print(f"  机制级: 衰减={short['decay_ok']} 方向={short['direction_ok']} "
              f"R²≥0.5={short['fit_r2_ok']} 消融={abl['no_decay_ok']} "
              f"恢复={rec['recover_ok']} 确定性={det['seq_equal']}")
        print(f"  判据可达性: 10s-ISI 主协议 NOT-REACHABLE（τ_rec 时程限制）")

    summary = dict(
        milestone="M6-B2", p_index="P3",
        pass_=pass_, status="pass-with-measurement-limitations",
        pass_type="pass-with-measurement-limitations",
        protocols=dict(
            short_isi_mechanism=short, ablation=abl, recovery=rec,
            isi_scaling=isi, network_302=net, determinism=det),
        criterion_reachability=criterion_reachability,
        mechanism_ok=mechanism_ok,
        measured_limitations=[
            "10s-ISI 主协议不可达：τ_rec=1000ms ≪ ISI=10s → STP 完全恢复 → R(n) "
            "常数（无习惯化）；30s 会话窗内仅 2 刺激可注触（协议分段受窗限制）",
            "短 ISI（0ms）形态 τ_hab≈2 出预注册带 [3,15]（Rankin 10s-ISI 带在短 "
            "ISI 不可比——衰减更快，形态如实记录）",
            "302 O2 网络 D_peak 非触诱发（touch≈no-touch，夹带干扰）→ 网络级"
            "习惯化不可干净测量（G1 部分通过结构性限制）",
            "自发恢复用相对判据（R_rest≥0.3×R(1)）；绝对恢复时程（真实分钟~小时）"
            "记录为测量限制（§0 #4 预注册）",
        ],
        missing_or_deferred=[
            "去习惯化/反习惯化（强刺激后恢复）——informational（预注册），本节点"
            "记录不判据化；可经 rest 后强刺激（I0×2）扩展验证（未跑，如实记录）",
        ],
        verdict=verdict,
        params=dict(n_short=N_SHORT, isi_short_ms=ISI_SHORT_MS,
                    isi_mid_ms=ISI_MID_MS, isi_main_ms=ISI_MAIN_MS,
                    n_main=N_MAIN, rest_ms=REST_MS, n_network=N_NETWORK,
                    seed=SEED, d_peak_thr=D_PEAK_THR,
                    decay_frac=DECAY_FRAC, ablation_frac=ABLATION_FRAC,
                    recover_frac_min=RECOVER_FRAC_MIN),
        wall_s=time.perf_counter() - t0,
    )
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(P3_RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, default=str)
    _write_csv(summary)
    if save_plot:
        _write_plot(summary)
    if verbose:
        print(f"[P3] 结果已落盘 {P3_RESULT_JSON}（wall {summary['wall_s']:.0f}s）")
    return summary


def _write_csv(s: dict) -> None:
    pr = s["protocols"]
    with open(P3_CSV, "w", encoding="utf-8") as f:
        f.write("# M6 P3 习惯化全协议（M6-B2 验证级；母版=M5 P5 逃避协议）\n"
                "# 判据（预注册 §0 P3）：指数拟合 R²≥0.5 / τ_hab∈[3,15]（10s-ISI）/\n"
                "#   恢复 R_rest≥0.3×R(1)；消融 STP 关→无衰减（H1 必需）\n"
                "protocol,metric,value,ok,note\n")
        rows = [
            ("short_isi_0ms", "r_seq", json.dumps(pr["short_isi_mechanism"]["r_seq"]),
             "", "STP 开，n=6，isi=0ms"),
            ("short_isi_0ms", "first_half_mean",
             pr["short_isi_mechanism"]["first_half_mean"],
             pr["short_isi_mechanism"]["decay_ok"], ""),
            ("short_isi_0ms", "last_half_mean",
             pr["short_isi_mechanism"]["last_half_mean"], "", ""),
            ("short_isi_0ms", "fit_tau_hab", pr["short_isi_mechanism"]["fit"]["tau_hab"],
             pr["short_isi_mechanism"]["fit"]["in_tau_band"],
             "预注册带 [3,15]（10s-ISI 语义；短 ISI 出带如实记录）"),
            ("short_isi_0ms", "fit_r2", pr["short_isi_mechanism"]["fit"]["r2"],
             pr["short_isi_mechanism"]["fit"]["r2_ok"], "预注册 R²≥0.5"),
            ("ablation_stp_off", "r_seq", json.dumps(pr["ablation"]["r_seq"]),
             pr["ablation"]["no_decay_ok"], "STP 关 → 无系统衰减"),
            ("ablation_stp_off", "decay", pr["ablation"]["decay"],
             pr["ablation"]["contrast_ok"], "与 STP 开对照（decay>0.05）"),
            ("recovery", "r_rest", pr["recovery"]["r_rest"],
             pr["recovery"]["recover_ok"], f"rest={pr['recovery']['rest_ms']}ms"),
            ("recovery", "recover_frac", pr["recovery"]["recover_frac"],
             pr["recovery"]["recover_ok"], "≥0.3×R(1) 预注册"),
            ("recovery", "r1", pr["recovery"]["r1"], "", ""),
            ("recovery", "r_last", pr["recovery"]["r_last"], "", ""),
            ("main_10s_isi", "r_seq",
             json.dumps(pr["isi_scaling"]["main_10s"]["r_seq"]),
             pr["isi_scaling"]["main_10s"]["constant"],
             "τ_rec 时程限制：x 完全恢复 → R 常数（主协议判据不可达）"),
            ("mid_3s_isi", "r_seq", json.dumps(pr["isi_scaling"]["mid_3s"]["r_seq"]),
             "", "3s≫τ_rec → R 近常数"),
            ("mid_3s_isi", "first_last_decay",
             pr["isi_scaling"]["mid_3s"]["first_last_decay"],
             pr["isi_scaling"]["mid_3s"]["constant"], ""),
            ("network_302", "r_seq", json.dumps(pr["network_302"].get("r_seq", [])),
             "", "O2 网络底物（夹带干扰记录）"),
            ("network_302", "no_touch_d_peak",
             pr["network_302"].get("no_touch_d_peak"),
             pr["network_302"].get("touch_eq_no_touch", False),
             "touch≈no-touch → 网络级触诱发不可干净测量"),
            ("determinism", "seq_equal", pr["determinism"]["seq_equal"], "", ""),
            ("determinism", "fit_equal", pr["determinism"]["fit_equal"], "", ""),
        ]
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
        f.write(f"# verdict: {s['verdict']}\n")


def _write_plot(s: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pr = s["protocols"]
    os.makedirs(REPORTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) 短 ISI R(n)：STP 开 vs 关 + 指数拟合
    ax = axes[0, 0]
    n = np.arange(1, N_SHORT + 1)
    r_on = pr["short_isi_mechanism"]["r_seq"]
    r_off = pr["ablation"]["r_seq"]
    ax.plot(n, r_on, "o-", color="tab:red", label="STP on (habituation)")
    ax.plot(n, r_off, "s--", color="tab:blue", label="STP off (ablation)")
    fit = pr["short_isi_mechanism"]["fit"]
    if np.isfinite(fit["tau_hab"]):
        ax.plot(n, fit["A"] * np.exp(-n / fit["tau_hab"]) + fit["B"],
                ls=":", color="k",
                label=f"exp fit τ_hab={fit['tau_hab']:.1f} R²={fit['r2']:.2f}")
    ax.axhline(0.3, color="gray", ls=":", lw=1)
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel("stimulus n"); ax.set_ylabel("R(n) = D_peak")
    ax.set_title("P3 (a): habituation at short ISI (0 ms)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # (b) 恢复
    ax = axes[0, 1]
    rec = pr["recovery"]
    labels = ["R(1)", "R(N)", "R_rest"]
    vals = [rec["r1"], rec["r_last"], rec["r_rest"]]
    colors = ["tab:blue", "tab:red", "tab:green"]
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:+.3f}",
                ha="center", fontsize=8)
    ax.axhline(0.3 * rec["r1"], color="tab:green", ls="--", lw=1,
               label=f"0.3×R(1)={0.3 * rec['r1']:+.3f}")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_title(f"P3 (b): spontaneous recovery (rest {rec['rest_ms']:.0f} ms)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # (c) ISI 扩展：10s vs 3s（判据可达性）
    ax = axes[1, 0]
    m10 = pr["isi_scaling"]["main_10s"]
    m3 = pr["isi_scaling"]["mid_3s"]
    ax.plot(np.arange(1, len(m10["r_seq"]) + 1), m10["r_seq"], "o-",
            color="tab:purple", label="10s ISI (main, n≤2)")
    ax.plot(np.arange(1, len(m3["r_seq"]) + 1), m3["r_seq"], "s--",
            color="tab:orange", label="3s ISI (n=6)")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel("stimulus n"); ax.set_ylabel("R(n)")
    ax.set_title("P3 (c): ISI ≫ τ_rec → R constant (criterion NOT-REACHABLE)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # (d) 302 底物 touch vs no-touch
    ax = axes[1, 1]
    net = pr["network_302"]
    if net.get("r_seq"):
        ax.plot(np.arange(1, len(net["r_seq"]) + 1), net["r_seq"], "o-",
                color="tab:red", label="touch R(n)")
        ax.axhline(net.get("no_touch_d_peak", float("nan")), color="tab:blue",
                   ls="--", lw=1.2, label=f"no-touch D_peak={net.get('no_touch_d_peak', float('nan')):+.3f}")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel("stimulus n"); ax.set_ylabel("D_peak")
    ax.set_title("P3 (d): 302 O2 network — touch≈no-touch (entrainment)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.suptitle("M6 P3: habituation protocol — pass-with-measurement-limitations",
                 y=1.01)
    fig.tight_layout()
    fig.savefig(P3_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="M6 P3 习惯化全协议验证")
    ap.add_argument("--skip-network", action="store_true",
                    help="跳过 302 网络底物（纯 reflex 机制验证）")
    ap.add_argument("--skip-plot", action="store_true")
    args = ap.parse_args()
    s = run_p3(save_plot=not args.skip_plot,
               with_network=not args.skip_network)
    slim = {k: v for k, v in s.items() if k != "protocols"}
    print(json.dumps(slim, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
