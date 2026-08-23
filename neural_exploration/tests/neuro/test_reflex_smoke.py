"""M3 触觉反射弧冒烟测试（清单 §4.3 验收：≥4 断言绿 + reports/neuro/m3_smoke.png）。

覆盖（清单 §4.3 + 肌肉规格 §2.3）：
  1. 触刺激 → PLM/AVM/DA 依次发放（各 ≥1 spike）、C_back > 0；
  2. 无刺激（intensity=0）→ 触觉链静默（PLM/AVM/DA 无发放）、C_back = 0；
     同时 VB 张力注入维持前进基线（C_fwd ≈ 0.2）；
  3. 方向：C_back 峰值 > C_fwd 峰值，D_peak > 0.3（后退）；
  4. 确定性：同参数重跑结果逐位一致（ReflexResult.__eq__ 数值比较）；
  5. 链拓扑/极性：CSV 读入连接数与递质类型与规格一致（P2 前置断言）；
  6. 出图 reports/neuro/m3_smoke.png（链各级 V + 双肌肉收缩叠加）。

CSV 依赖：data/m3_reflex_params.csv（B1a 定稿）——若缺失则 wait_for_csv 轮询
（30s 间隔、最多 15min），期间其他测试已先跑完。
"""

import os

import numpy as np
import pytest

from neural_exploration.src.reflex_arc import (
    ReflexArc,
    ReflexResult,
    load_reflex_params,
    wait_for_csv,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data")
CSV_PATH = os.path.join(DATA_DIR, "m3_reflex_params.csv")


@pytest.fixture(scope="module")
def arc():
    """CSV 依赖就绪后构建 ReflexArc（module 级复用，确定性参数）。"""
    wait_for_csv(CSV_PATH, timeout_s=900.0, interval_s=30.0)
    return ReflexArc(csv_path=CSV_PATH)


def test_csv_chain_topology(arc):
    """拓扑/极性断言（P2 前置）：3 化学突触 + 2 肌肉驱动，递质类型正确。"""
    s = arc.chain_summary()
    assert s["roles"] == ["PLM", "AVM", "DA", "VB"]
    assert s["n_chemical"] == 3
    assert s["n_muscle_drives"] == 2
    assert s["synapse_types"] == {
        "PLM->AVM": "ampa", "AVM->DA": "ampa", "AVM->VB": "gaba"}
    assert s["tonic_uA_cm2"].get("VB", 0.0) > 0, "VB 需有张力注入"


def test_touch_evokes_chain(arc):
    """触刺激 → PLM/AVM/DA 依次发放（各 ≥1 spike）、C_back > 0。"""
    r = arc.run(intensity=1.0)
    assert len(r.spikes("PLM", "node3")) >= 1, "PLM 必须发放"
    assert len(r.spikes("AVM", "node3")) >= 1, "AVM 必须发放"
    assert len(r.spikes("DA", "node3")) >= 1, "DA 必须发放"
    assert r.c_back_peak > 0.0, "C_back 必须有收缩"


def test_chain_order(arc):
    """发放次序严格递增：t_PLM < t_AVM < t_DA（P2 前置断言）。"""
    r = arc.run(intensity=1.0)
    t_plm = r.spikes("PLM", "node3")[0]
    t_avm = r.spikes("AVM", "node3")[0]
    t_da = r.spikes("DA", "node3")[0]
    assert t_plm < t_avm < t_da, f"链发放次序错误: {t_plm} < {t_avm} < {t_da}"


def test_no_stimulus_chain_silent(arc):
    """无刺激（intensity=0）→ 触觉链静默、C_back = 0；VB 张力维持前进基线。"""
    r = arc.run(intensity=0.0)
    assert len(r.spikes("PLM", "node3")) == 0
    assert len(r.spikes("AVM", "node3")) == 0
    assert len(r.spikes("DA", "node3")) == 0
    assert r.c_back_peak == 0.0, "无刺激时 C_back 必须为 0"
    # 前进基线：VB 张力注入维持 C_fwd ≈ 0.2（清单 §2.1 静息态规格）
    assert 0.1 < r.c_fwd_peak < 0.4, f"C_fwd 基线异常: {r.c_fwd_peak}"
    assert np.all(np.isfinite(r.v_mv["plm_soma"]))


def test_direction_backward(arc):
    """方向：C_back 峰值 > C_fwd 峰值，D_peak > 0.3（后退，P1 判据前置）。"""
    r = arc.run(intensity=1.0)
    assert r.c_back_peak > r.c_fwd_peak, \
        f"方向错误：C_back_peak={r.c_back_peak} 应 > C_fwd_peak={r.c_fwd_peak}"
    assert r.d_peak > 0.3, f"D_peak={r.d_peak} 应 > 0.3"


def test_deterministic_rerun(arc):
    """确定性铁律：同参数重跑逐位一致（ReflexResult.__eq__ 数值比较）。"""
    r1 = arc.run(intensity=1.0)
    r2 = arc.run(intensity=1.0)
    assert isinstance(r1, ReflexResult) and isinstance(r2, ReflexResult)
    assert r1 == r2, "同参数重跑必须逐位一致"
    assert r1.d_peak == r2.d_peak


def test_plot_m3_smoke(arc):
    """出图 reports/neuro/m3_smoke.png（链各级 V + 双肌肉收缩叠加）。"""
    from neural_exploration.src.reflex_arc import plot_reflex

    r = arc.run(intensity=1.0)
    reports_dir = os.path.join(os.path.dirname(CSV_PATH), "..", "reports", "neuro")
    out_png = os.path.join(os.path.abspath(reports_dir), "m3_smoke.png")
    png = plot_reflex(r, out_png)
    assert os.path.exists(png) and os.path.getsize(png) > 10_000, f"图未生成: {png}"
