"""M6 学习协议冒烟测试（清单 §6.2 #6：习惯化 + 联想学习协议运行器）。

覆盖（≥6 断言；确定性 p=1/n=1，同参数重跑逐位一致）：
  1. P3 习惯化（reflex 底物，STP 开）：R(n) = D_peak 序列可测（有限/长度正确/
     首刺激方向 sanity D_peak>0.3——母版 = M5 P5 逃避协议 touch@73ms）；
  2. P3 习惯化衰减：STP 开 + 短 ISI → 后半均值 < 前半均值×0.5（衰减趋势，
     Rankin 指数衰减方向；τ_hab 拟合失败时记录形状偏差，不静默重试）；
  3. P3 消融：STP 关 → 衰减消失（后半均值 ≥ 前半均值×0.7）——H1 机制必需；
  4. P3 302 O2 网络底物：R(n) 可计算（确定性）+ touch≈no-touch（夹带干扰
     测量限制如实记录——D_peak 非触诱发，L23）；
  5. P4 联想学习获得：训练后 w 上升（Δw_train>0.1）+ CI_salt 训练后均值 >
     训练前（方向性）；
  6. P4 η=0 消融：无获得（权重不变 + CI 与基线无差异）——三因子门控必需；
  7. P4 消退可逆：US 反号后 w 下降 + CI_salt 回落（趋势性）；
  8. 确定性重跑逐位一致（R 序列 / CI / 权重逐位相等）；
  9. 出图 reports/neuro/m6_learning_smoke.png。

实测坑（M6-B1c L23+，见 docs/m6_env_notes.md）：
  - 302 O2 全网 D_peak 由自发动力学主导（touch@73ms +0.355 vs no-touch
    +0.357）→ 网络级触诱发反应不可干净测量（G1 部分通过结构性限制）；
  - 反射子图 STP 二值坍缩（命令阈值）→ 机制在短 ISI（≤50ms）演示；
    Rankin 10s-ISI 主协议受模型时程限制（τ_rec 在 ISI 内完全恢复）。

数据依赖：data/m5_connectome.csv、data/m3_reflex_params.csv、
data/m5_worm_params.csv、data/m6_learning_params.csv（learning 段由
src/learning.py 追加）。
"""

import os

import numpy as np
import pytest

from neural_exploration.src.learning import (
    AssociativeLearningLoop,
    HabituationLoop,
    load_learning_params,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data")
REPORTS_NEURO = os.path.join(os.path.dirname(DATA_DIR), "reports", "neuro")
M6_PARAMS_CSV = os.path.join(DATA_DIR, "m6_learning_params.csv")

#: 冒烟模式参数（机制演示用短 ISI；完整协议参数在 CSV，见 L23 时程限制）
SMOKE_N_STIM = 6
SMOKE_ISI_MS = 0.0          # 短 ISI（Rankin 10s 主协议的时程限制记录于 L23）
SMOKE_SEED = 0


@pytest.fixture(scope="module")
def lp():
    """CSV 唯一定稿源参数（learning 段）。"""
    p = load_learning_params(M6_PARAMS_CSV)
    assert p.n_stim == 20 and p.isi_ms == 10000.0      # 协议定稿值（冒烟覆盖）
    assert p.stp_u0 == pytest.approx(0.6)              # H1 预注册 u0≈0.6
    assert p.eta == pytest.approx(1e-2)                # 三因子 η 上界
    assert len(p.tf_edges) == 8                        # ASE→AIY/AIB 8 条
    return p


# --------------------------------------------------------------------- #
# 1) P3 习惯化：R(n) 序列可测（reflex 底物）
# --------------------------------------------------------------------- #
def test_habituation_r_sequence_measurable(lp):
    loop = HabituationLoop(params=lp, substrate="reflex")
    res = loop.run(n_stim=SMOKE_N_STIM, isi_ms=SMOKE_ISI_MS, seed=SMOKE_SEED,
                   stp_enabled=True)
    r = np.asarray(res["r_seq"], dtype=float)
    assert r.size == SMOKE_N_STIM, "R(n) 序列长度应为 n_stim"
    assert np.all(np.isfinite(r)), "R(n) 应全有限（无 NaN）"
    assert res["direction_ok"], (
        f"首刺激方向 sanity 应 D_peak>0.3（母版 escape），实测 {r[0]:+.3f}")
    assert res["substrate"] == "reflex"
    # 拟合可计算（退化 → 形状偏差如实记录，不静默重试）
    assert "tau_hab" in res["fit"] and "r2" in res["fit"]


# --------------------------------------------------------------------- #
# 2) P3 习惯化衰减（STP 开 → R(n) 下降）
# --------------------------------------------------------------------- #
def test_habituation_decay_with_stp(lp):
    loop = HabituationLoop(params=lp, substrate="reflex")
    res = loop.run(n_stim=SMOKE_N_STIM, isi_ms=SMOKE_ISI_MS, seed=SMOKE_SEED,
                   stp_enabled=True)
    assert res["last_half_mean"] < 0.5 * res["first_half_mean"], (
        f"STP 开应衰减（后半均值 ≪ 前半均值），实测 first={res['first_half_mean']:+.3f}"
        f" last={res['last_half_mean']:+.3f} R={[round(x,3) for x in res['r_seq']]}")
    assert res["decay"] > 0.05, f"R(1)−R(N) 应为正（衰减），实测 {res['decay']:+.3f}"


# --------------------------------------------------------------------- #
# 3) P3 消融：STP 关 → 衰减消失（H1 机制必需）
# --------------------------------------------------------------------- #
def test_habituation_ablation_stp_off(lp):
    loop = HabituationLoop(params=lp, substrate="reflex")
    res = loop.run(n_stim=SMOKE_N_STIM, isi_ms=SMOKE_ISI_MS, seed=SMOKE_SEED,
                   stp_enabled=False)
    assert res["last_half_mean"] >= 0.7 * res["first_half_mean"], (
        f"STP 关应无系统衰减（后半 ≥ 0.7×前半），实测 "
        f"first={res['first_half_mean']:+.3f} last={res['last_half_mean']:+.3f}"
        f" R={[round(x,3) for x in res['r_seq']]}")


# --------------------------------------------------------------------- #
# 4) P3 302 O2 网络底物：R(n) 可计算 + 夹带限制记录（touch≈no-touch）
# --------------------------------------------------------------------- #
def test_habituation_network_302_measured_limitation(lp):
    loop = HabituationLoop(params=lp, substrate="network")
    res = loop.run(n_stim=3, isi_ms=0.0, seed=SMOKE_SEED,
                   stp_enabled=True, no_touch_control=True)
    r = np.asarray(res["r_seq"], dtype=float)
    assert r.size == 3 and np.all(np.isfinite(r)), "302 底物 R(n) 应可计算"
    assert res["no_touch_d_peak"] is not None
    # 夹带干扰限制（L23）：touch 与 no-touch 的 D_peak 应接近（自发动力学主导）
    assert abs(r[0] - res["no_touch_d_peak"]) < 0.2, (
        "O2 全网 D_peak 应自发主导（touch≈no-touch，夹带限制如实记录），实测 "
        f"touch={r[0]:+.3f} no_touch={res['no_touch_d_peak']:+.3f}")


# --------------------------------------------------------------------- #
# 5) P4 联想学习：三因子获得（机制级方向性 Δw>0）+ CI_salt 实测入档
# --------------------------------------------------------------------- #
def test_associative_acquisition_directional(lp):
    """训练后三因子权重上升（Δw>0.1）+ CI_salt 训练后 > 训练前（方向性）。

    ⚠ 实测限制（L23）：CI_salt 读出灵敏度低——20-role 子图命令中间簇自持
    振荡主导（G1 P4 未缓解的结构性延续），感觉通路权重部分被淹没 →
    ΔCI 实测为正但幅度小（如实入档，不静默）；机制级获得/消融/消退可验证。
    """
    loop = AssociativeLearningLoop(params=lp, eta=1e-2, seed=0)
    res = loop.run(n_test=4, t_test_ms=1500.0, t_train_ms=8000.0,
                   t_ext_ms=12000.0, seed_base=0)
    assert res["dw_train"] > 0.1, (
        f"训练后三因子权重应上升（Δw>0.1，机制级获得），实测 Δw={res['dw_train']:+.4f}")
    assert res["mean_ci_post"] > res["mean_ci_pre"], (
        f"训练后 CI_salt 应 > 训练前（方向性；幅度小——限制见 L23），实测 "
        f"CI_pre={res['mean_ci_pre']:+.3f} CI_post={res['mean_ci_post']:+.3f}")


# --------------------------------------------------------------------- #
# 6) P4 η=0 消融：无获得（三因子门控必需）
# --------------------------------------------------------------------- #
def test_associative_eta0_ablation(lp):
    loop = AssociativeLearningLoop(params=lp, eta=0.0, seed=0)
    res = loop.run(n_test=4, t_test_ms=1500.0, t_train_ms=8000.0,
                   t_ext_ms=12000.0, seed_base=0, with_extinction=False,
                   with_eta0=False)
    assert res["dw_train"] < 1e-9, f"η=0 权重应不变，实测 Δw={res['dw_train']:+.6f}"
    assert abs(res["mean_ci_post"] - res["mean_ci_pre"]) < 0.05, (
        f"η=0 应无获得（CI 与基线无差异），实测 "
        f"CI_pre={res['mean_ci_pre']:+.3f} CI_post={res['mean_ci_post']:+.3f}")


# --------------------------------------------------------------------- #
# 7) P4 消退可逆：US 反号 → 权重回落（机制级可逆性）
# --------------------------------------------------------------------- #
def test_associative_extinction_reversible(lp):
    loop = AssociativeLearningLoop(params=lp, eta=1e-2, seed=0)
    res = loop.run(n_test=4, t_test_ms=1500.0, t_train_ms=8000.0,
                   t_ext_ms=12000.0, seed_base=0)
    assert res["dw_ext"] < -0.01, (
        f"消退期 US 反号权重应回落（机制级可逆），实测 Δw_ext={res['dw_ext']:+.4f}")
    assert res["mean_ci_ext"] < res["mean_ci_post"], (
        f"消退后 CI_salt 应回落（趋势性；限制见 L23），实测 "
        f"CI_post={res['mean_ci_post']:+.3f} CI_ext={res['mean_ci_ext']:+.3f}")


# --------------------------------------------------------------------- #
# 8) 确定性重跑逐位一致 + 出图
# --------------------------------------------------------------------- #
def test_learning_determinism_and_plot(lp):
    loop1 = HabituationLoop(params=lp, substrate="reflex")
    r1 = loop1.run(n_stim=SMOKE_N_STIM, isi_ms=SMOKE_ISI_MS, seed=SMOKE_SEED,
                   stp_enabled=True)
    loop2 = HabituationLoop(params=lp, substrate="reflex")
    r2 = loop2.run(n_stim=SMOKE_N_STIM, isi_ms=SMOKE_ISI_MS, seed=SMOKE_SEED,
                   stp_enabled=True)
    assert np.array_equal(np.asarray(r1["r_seq"]), np.asarray(r2["r_seq"])), (
        "习惯化重跑应逐位一致（确定性 p=1/n=1）")
    assert r1["fit"]["tau_hab"] == r2["fit"]["tau_hab"]

    a1 = AssociativeLearningLoop(params=lp, eta=1e-2, seed=0)
    ra = a1.run(n_test=4, t_test_ms=1500.0, t_train_ms=8000.0,
                t_ext_ms=12000.0, seed_base=0)
    a2 = AssociativeLearningLoop(params=lp, eta=1e-2, seed=0)
    rb = a2.run(n_test=4, t_test_ms=1500.0, t_train_ms=8000.0,
                t_ext_ms=12000.0, seed_base=0)
    assert np.array_equal(np.asarray(ra["ci_pre"]), np.asarray(rb["ci_pre"]))
    assert np.array_equal(np.asarray(ra["ci_post"]), np.asarray(rb["ci_post"]))
    assert np.array_equal(np.asarray(ra["w_tr"]), np.asarray(rb["w_tr"]))

    # 出图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(REPORTS_NEURO, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    # (a) 习惯化 R(n)：STP 开 vs 关（reflex）
    ax = axes[0]
    for stp, lab, c in ((True, "STP on", "tab:red"),
                        (False, "STP off (ablation)", "tab:blue")):
        loop = HabituationLoop(params=lp, substrate="reflex")
        r = loop.run(n_stim=SMOKE_N_STIM, isi_ms=SMOKE_ISI_MS, seed=SMOKE_SEED,
                     stp_enabled=stp)
        ax.plot(np.arange(1, len(r["r_seq"]) + 1), r["r_seq"], marker="o",
                label=lab, color=c)
        if stp:
            n = np.arange(1, len(r["r_seq"]) + 1.0)
            A, tau, B = r["fit"]["A"], r["fit"]["tau_hab"], r["fit"]["B"]
            if np.isfinite(tau):
                ax.plot(n, A * np.exp(-n / tau) + B, ls="--", color=c,
                        label=f"exp fit τ_hab={tau:.1f} R²={r['fit']['r2']:.2f}")
    ax.axhline(0.3, color="gray", ls=":", lw=1)
    ax.set_xlabel("stimulus n"); ax.set_ylabel("R(n) = D_peak")
    ax.set_title("P3 habituation (reflex substrate)")
    ax.legend(fontsize=8)
    # (b) P4 CI_salt：基线/训练后/消退/η=0
    ax = axes[1]
    loop0 = AssociativeLearningLoop(params=lp, eta=1e-2, seed=0)
    res = loop0.run(n_test=4, t_test_ms=1500.0, t_train_ms=8000.0,
                    t_ext_ms=12000.0, seed_base=0)
    labels = ["CI_pre", "CI_post", "CI_ext", "CI_eta0=0"]
    means = [res["mean_ci_pre"], res["mean_ci_post"], res["mean_ci_ext"],
             res["mean_ci_eta0"]]
    colors = ["tab:blue", "tab:red", "tab:green", "tab:gray"]
    bars = ax.bar(labels, means, color=colors)
    for b, m in zip(bars, means):
        if np.isfinite(m):
            ax.text(b.get_x() + b.get_width() / 2, m + 0.01, f"{m:+.2f}",
                    ha="center", fontsize=8)
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_ylabel("CI_salt"); ax.set_title("P4 associative learning")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS_NEURO, "m6_learning_smoke.png"), dpi=120)
    plt.close(fig)
    assert os.path.exists(os.path.join(REPORTS_NEURO, "m6_learning_smoke.png"))
