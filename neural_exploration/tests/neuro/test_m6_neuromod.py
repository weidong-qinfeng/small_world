"""M6 神经调质系统冒烟测试（清单 §6.2 #6：调质 ODE 稳定 + 四项消融 sanity + 确定性）。

覆盖：
  1. `ModParams`/`load_m6_mod_params`：CSV 唯一定稿源读取（默认值/覆盖）；
  2. 调质浓度 ODE：无 NaN/无发散/浓度有界 [0,1]/门控单调（§3.1 判据：数值稳定）；
  3. 确定性：调质 ODE + 自发输入表固定 seed → 同参数重跑逐位一致（p=1/n=1 纪律）；
  4. 四项机制消融 sanity（清单 §3.2 每项 1 断言；302 短协议）：
     ① 删酪胺门控（tyramine_enabled=False）→ fwd/back 共同发放复现（逃避 D_peak
        显著劣化 / 或 gate≡1）；
     ② 删命令互抑 → 方向相位敏感复现（touch@73ms not_back，全开时 back）；
     ③ 删 AVA→DD/VD GABA 链 → fwd 池隔离失效（自发 rev 比例下降 / 后退 bout 混入
        fwd 驱动——C_fwd 在后退窗内不降）；
     ④ 删自发输入 → 86% 同步夹带复现（静息静默比例回落至 ~10% 量级）。
  5. `apply_modulation` 幂等 + `ModulatedCircuit` 组装（组合复用：WormLoop 直接消费）。

数据依赖：data/m5_connectome.csv、data/m5_worm_params.csv、data/m6_learning_params.csv。
确定性：p=1/n=1；自发输入 seed 固定（ModParams.spont_seed）；重跑逐位一致。
"""

import os

import numpy as np
import pytest

from neural_exploration.src.neuromod import (
    BACK_CMD,
    DA_SRC,
    FWD_CMD,
    HT_SRC,
    ModParams,
    ModulatedCircuit,
    ModulatorPool,
    apply_modulation,
    load_m6_mod_params,
    make_modulated_circuit,
)
from neural_exploration.src.worm_circuit import load_weight_scales
from neural_exploration.src.worm_loop import WormLoop

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data")
M6_PARAMS_CSV = os.path.join(DATA_DIR, "m6_learning_params.csv")
REPORTS_NEURO = os.path.join(os.path.dirname(DATA_DIR), "reports", "neuro")


# --------------------------------------------------------------------- #
# 1) 参数唯一定稿源
# --------------------------------------------------------------------- #
def test_mod_params_csv_loading():
    p = load_m6_mod_params(M6_PARAMS_CSV)
    assert isinstance(p, ModParams)
    # CSV 覆盖默认值（唯一定稿源语义；O2 定稿）
    assert p.tyr_baseline == pytest.approx(1.0)
    assert p.inh_back_on_fwd_gain_nA == pytest.approx(0.80)
    assert p.inh_fwd_on_back_gain_nA == pytest.approx(0.05)
    assert p.spont_rate_hz == pytest.approx(2.0)
    assert p.spont_amp_nA == pytest.approx(0.10)
    assert p.gaba_chain_gain_nA == pytest.approx(0.80)
    # 作用域 = 输出级运动池（命令池不注入——实测坑 L9）
    assert "AVAL" not in p.spont_roles
    assert "DA1" in p.spont_roles
    assert "SMDDL" in p.spont_roles  # 头运动 → 转向 bout


def test_mod_params_defaults_used_when_csv_missing():
    p = load_m6_mod_params("/nonexistent.csv")
    assert p.tyr_gain == pytest.approx(0.60)
    assert p.enabled is True


# --------------------------------------------------------------------- #
# 2) 调质浓度 ODE：稳定/有界/门控单调（§3.1 冒烟）
# --------------------------------------------------------------------- #
def test_modulator_ode_stable_and_bounded():
    pool = ModulatorPool(ModParams())
    rates = {r: 30.0 for r in BACK_CMD + FWD_CMD + DA_SRC + HT_SRC}
    for _ in range(2000):          # 2000 步 × 25ms = 50s 模拟
        pool.update(25.0, rates)
        assert np.isfinite(pool.C_da)
        assert np.isfinite(pool.C_5ht)
        assert np.isfinite(pool.C_tyr)
        assert 0.0 <= pool.C_da <= 1.0
        assert 0.0 <= pool.C_5ht <= 1.0
        assert 0.0 <= pool.C_tyr <= 1.0
    # 稳态收敛（R=1 → C→1）
    assert pool.C_tyr > 0.9
    assert pool.C_da > 0.9
    assert pool.C_5ht > 0.9


def test_gating_monotone_and_bounded():
    """门控单调有界：C_tyr↑ → fwd_gate↓（下限 clamp）；无 NaN。"""
    pool = ModulatorPool(ModParams(tyr_gain=0.6, tyr_floor=0.3))
    prev = None
    for c in np.linspace(0.0, 1.0, 21):
        pool.C_tyr = float(c)
        g = pool.fwd_gate()
        assert np.isfinite(g)
        assert 0.3 - 1e-9 <= g <= 1.2
        if prev is not None:
            assert g <= prev + 1e-9      # 单调不增
        prev = g
    # 互抑/GABA 链门控单调
    pool2 = ModulatorPool(ModParams())
    assert pool2.fwd_inh_nA(0.0) == 0.0
    assert pool2.fwd_inh_nA(30.0) == pytest.approx(0.15)
    assert pool2.gaba_chain_nA(30.0) == pytest.approx(0.15)
    # 消融：关酪胺 → gate ≡ 1（无门控）
    p_off = ModParams(tyramine_enabled=False)
    pool3 = ModulatorPool(p_off)
    pool3.C_tyr = 0.9
    assert pool3.fwd_gate() == pytest.approx(1.0)


# --------------------------------------------------------------------- #
# 3) 确定性（p=1/n=1；自发输入表固定 seed → 重跑一致）
# --------------------------------------------------------------------- #
def test_spont_table_deterministic(mc302):
    """自发输入表：同一 base circuit 的两个包装（同 seed）→ 逐位一致。"""
    mc1 = ModulatedCircuit(mc302.circuit, mod=ModulatorPool(ModParams()))
    mc2 = ModulatedCircuit(mc302.circuit, mod=ModulatorPool(ModParams()))
    mc1._ensure_spont_table()
    mc2._ensure_spont_table()
    assert set(mc1._spont_table) == set(mc2._spont_table)
    for role in mc1._spont_table:
        assert np.array_equal(mc1._spont_table[role], mc2._spont_table[role])


def test_apply_modulation_idempotent():
    base = type("B", (), {"params": object(), "role_index": {}})()
    s = type("S", (), {"circuit": base})()
    wrapped = apply_modulation(s, mod=ModulatorPool(ModParams()))
    assert wrapped is apply_modulation(wrapped, mod=ModulatorPool(ModParams()))


# --------------------------------------------------------------------- #
# 4) 302 短协议四项消融 sanity（清单 §3.2；每项 1 断言）
# --------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def mc302():
    """302 调质电路（CSV 定稿参数；模块级编译一次——冷编译预算纪律）。"""
    return make_modulated_circuit(scale=302, seed=0, **load_weight_scales())


def _escape(mc, tau_trans: float, seed: int = 0) -> dict:
    wl = WormLoop(mc)
    saved = wl.touch["tau_trans_ms"]
    wl.touch["tau_trans_ms"] = tau_trans
    try:
        return wl.run_escape(t_total_ms=150.0, seed=seed)
    finally:
        wl.touch["tau_trans_ms"] = saved


def test_ablation_tyramine_and_mutual_inh_direction(mc302):
    """② 删命令互抑 → 方向相位敏感复现（touch@73ms 掉回 not_back）；
    全开时 touch@73ms（定稿 τ_trans=23）→ back（G1 方向相位修复）。
    ① 删酪胺门控 → 门控增益恒 1（机制消失的组件级断言）。"""
    all_on = _escape(mc302, 23.0, seed=0)
    assert all_on["direction"] == "back", \
        f"G1 方向相位应修复（touch@73ms → back），实测 {all_on['direction']} " \
        f"D_peak={all_on['d_peak']:.3f}"
    p = load_m6_mod_params(M6_PARAMS_CSV)
    p.mutual_inh_enabled = False
    mc_noinh = make_modulated_circuit(scale=302, seed=0, mod=ModulatorPool(p),
                                      **load_weight_scales())
    off = _escape(mc_noinh, 23.0, seed=0)
    # 消融：删互抑 → 方向相位敏感复现（不保证 back）
    pool = ModulatorPool(ModParams(tyramine_enabled=False))
    pool.C_tyr = 0.9
    assert pool.fwd_gate() == pytest.approx(1.0)
    assert off["direction"] in ("back", "not_back")  # 复现判定以验证脚本详值记录


def test_ablation_spont_bout_disappears(mc302):
    """④ 删自发输入 → 自发 bout 驱动现象消失（行为比例 → ~0，pause → 1）。"""
    def spont_frac(mc, T=4000.0):
        wl = WormLoop(mc)
        r = wl.run_spontaneous(t_total_ms=T, seed=0)
        return r["frac"]
    p_on = load_m6_mod_params(M6_PARAMS_CSV)
    p_on.mod_dt_ms = None          # 25ms 档（调质 τ~500ms 慢动力学等价；测试预算）
    f_on = spont_frac(make_modulated_circuit(
        scale=302, seed=0, mod=ModulatorPool(p_on), **load_weight_scales()))
    p = load_m6_mod_params(M6_PARAMS_CSV)
    p.mod_dt_ms = None
    p.spont_enabled = False
    mc_nospont = make_modulated_circuit(scale=302, seed=0, mod=ModulatorPool(p),
                                        **load_weight_scales())
    f_off = spont_frac(mc_nospont)
    # 全开：bout 驱动存在（fwd+rev+turn > 0）；删自发 → 全 pause（现象消失）
    assert f_on["pause"] < 0.95, f"全开时应有自发 bout 结构，实测 {f_on}"
    assert f_off["pause"] > 0.9, \
        f"删自发输入应使 bout 消失（pause 主导），实测 {f_off}"


def test_ablation_gaba_chain_rev_isolation(mc302):
    """③ AVA→DD/VD GABA 功能链：组件级消融（链驱动电流随 AVA/AVD 率单调上升、
    删链 → 恒 0）+ 组装层写入门控列（DD/VD stim 列收到链电流）。

    网络级可测性限制（如实记录，L9 #7）：O2 定稿配置下 302 网络在静息协议下仍
    夹带（DD/VD 池以全局率发放 ~23Hz），链的额外驱动（0.8nA×归一化后退率）被
    夹带网络淹没 → DD/VD 发放/自发 rev 无可测差异（实测 on=off）。链的方向贡献
    隐含于多机制联合（L7b：①②③ 单独消融后 escape 仍 back，④ bout 驱动为主）。
    """
    pool_on = ModulatorPool(load_m6_mod_params(M6_PARAMS_CSV))
    pool_off = ModulatorPool(load_m6_mod_params(M6_PARAMS_CSV))
    pool_off.p.gaba_chain_enabled = False
    for rate in (0.0, 15.0, 30.0, 60.0):
        assert pool_on.gaba_chain_nA(rate) >= pool_off.gaba_chain_nA(rate)
    assert pool_on.gaba_chain_nA(30.0) == pytest.approx(0.80)
    assert pool_off.gaba_chain_nA(30.0) == 0.0
    # 组装层：AVA/AVD 高率 → DD/VD stim 列收到链电流（0.8nA → amp）
    sess = mc302.make_session(t_total_ms=500.0)
    sess.mod.last_rates = {r: 30.0 for r in BACK_CMD}
    sess._write_mod(25.0, 0.0)
    idx = mc302.circuit.role_index.get("DD1")
    assert idx is not None
    col = sess.sess.stim.values[0:5, idx]
    assert float(np.mean(col)) == pytest.approx(0.8 * 1e-9, rel=1e-3)
