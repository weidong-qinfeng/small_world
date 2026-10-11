"""M11 小鼠皮层微环路内核冒烟测试（P3/G0 回归；CPU、O(N≤400)、秒级）。

覆盖（全部为**可证伪的硬断言**，任一失败即内核回归）：
  - E1b 四亚型槽：逐类边数 == 逐类槽内边数（编码路径前置断言）+ 丢 `syn_class` 必须被拦截；
  - E1b 关闭态等价：`n_ch=2 + REF_FROZEN` 与 M10 冻结内核**逐位一致**（含多延迟档）；
  - E2b 慢变量：开关**必须**使逐神经元计数逐位不同；`inc=0` 必须被拒绝；
  - E3b 层间延迟打包：到达步差 == 档差；多档**共享单一环槽**；延迟档 >4 必须被拒绝；
  - 分段可组合性 (e)：k=1/2/5 总脉冲与逐神经元计数逐位相同；
  - 双估计器：`spk_total` vs `counts.sum()`/`pop_trace.sum()` 相对差 ≤2%。
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from neural_exploration.src.larval_zf_engine import LarvalZFEngine             # noqa: E402
from neural_exploration.src.mouse_cortex_circuit import (                      # noqa: E402
    attach_to_engine, build_laminar_probe_column)
from neural_exploration.src.mouse_cortex_engine import (MouseCortexEngine,     # noqa: E402
                                                        REF_FROZEN)

N = 400


def _events(T=2000, n=N, density=2.0, seed=11):
    n_ev = int(density * T * 5e-5 * n)
    rng = np.random.default_rng(seed)
    return (rng.integers(0, T, n_ev), rng.integers(0, n, n_ev),
            rng.uniform(17.0, 23.0, n_ev).astype(np.float32))


def _net(seed=0, delay_bins=(40, 60, 80), tol_ok=True):
    return build_laminar_probe_column(N, indegree=20, seed=seed, delay_bins=delay_bins)


def test_subtype_slot_encoding_assertion():
    """四类槽逐类断言：装配方意图 == 内核实际槽；丢 syn_class 必须被拦截且断言失败。"""
    net = _net()
    eng = MouseCortexEngine(N, device="cpu", use_compile=False, verbose=False)
    eng.configure_subtypes(True)
    eng.configure_layers(net["layer_of"], delay_bins=net["meta"]["delay_bins"])
    attach_to_engine(eng, net)
    r = eng.assert_slot_encoding(net["meta"]["class_edge_counts"])
    assert r["ok"]
    assert eng.class_slot_counts() == net["meta"]["class_edge_counts"]
    assert eng.class_edge_counts() == net["meta"]["class_edge_counts"]

    with pytest.raises(ValueError):                       # M10 L21 静默失效的守卫
        bad = MouseCortexEngine(N, device="cpu", use_compile=False, verbose=False)
        bad.configure_subtypes(True)
        bad.finalize()
        bad.add_synapses(net["pre"], net["post"], net["gmax"], net["delay"],
                         inhibitory=net["inhibitory"])    # 故意丢 syn_class

    bad2 = MouseCortexEngine(N, device="cpu", use_compile=False, verbose=False,
                             strict_syn_class=False)
    bad2.configure_subtypes(True)
    bad2.finalize()
    bad2.add_synapses(net["pre"], net["post"], net["gmax"], net["delay"],
                      inhibitory=net["inhibitory"])
    bad2.finalize()
    with pytest.raises(AssertionError):                   # 断言必须**可证伪**
        bad2.assert_slot_encoding(net["meta"]["class_edge_counts"])


@pytest.mark.parametrize("delay_bins,use_ext", [((40,), True), ((40, 60, 80), False)])
def test_off_state_equivalent_to_m10(delay_bins, use_ext):
    """E1b/E2b/E3b 全关 + REF_FROZEN ⇒ 与 M10 冻结内核逐位一致。

    **多延迟档臂必须不带外部事件**：冻结件在 ≥2 延迟组下静默丢弃全部外部事件（E-M10-2），
    M11 修好后会正常交付 ⇒ 两者不可比（差异来自上游缺陷，不是内核回归）。
    """
    net = _net(delay_bins=delay_bins)
    cls = np.where(net["syn_class"] > 0, 1, 0).astype(np.int8)
    net2 = dict(net, syn_class=cls, inhibitory=(cls > 0))
    if use_ext:
        st, ne, gg = _events()
    else:
        st = ne = np.zeros(0, dtype=int)
        gg = np.zeros(0, dtype=np.float32)
    outs = []
    for eng in (LarvalZFEngine(N, device="cpu", use_compile=False, verbose=False),
                MouseCortexEngine(N, device="cpu", use_compile=False, verbose=False)):
        if isinstance(eng, MouseCortexEngine):
            eng.configure_refractory(REF_FROZEN)
        eng.add_synapses(net2["pre"], net2["post"], net2["gmax"], net2["delay"],
                         inhibitory=net2["inhibitory"], syn_class=net2["syn_class"])
        eng.finalize()
        eng.set_neuron_heterogeneity(net2["bias"])
        eng.set_external_events(st, ne, gg, n_steps=2000)
        eng.reset(seed=1)
        eng.reset_counts()
        eng.run(2000, delivery="step", record="counts")
        outs.append(eng.t_count.detach().cpu().numpy().astype(np.int64))
    assert np.array_equal(outs[0], outs[1]), "关闭态必须与 M10 逐位一致"


def test_slow_variable_is_alive_and_changes_dynamics():
    """E2b：慢变量必须使逐神经元计数逐位不同；`inc=0` 必须被拒绝（"参数是活的"门）。"""
    net = _net()
    st, ne, gg = _events()

    def _run(slow):
        eng = MouseCortexEngine(N, device="cpu", use_compile=False, verbose=False)
        eng.configure_subtypes(True)
        if slow:
            eng.configure_slow(True, tau_slow_ms=1000.0, inc_per_spike=6.0)
        eng.configure_layers(net["layer_of"], delay_bins=net["meta"]["delay_bins"])
        attach_to_engine(eng, net)
        eng.set_external_events(st, ne, gg, n_steps=2000)
        eng.reset(seed=1)
        eng.reset_counts()
        r = eng.run(2000, delivery="step", record="counts")
        return eng, r

    e_on, r_on = _run(True)
    _e_off, r_off = _run(False)
    assert not np.array_equal(r_on["spike_counts"], r_off["spike_counts"]), \
        "慢变量开关未改变轨迹 ⇒ MECHANISM_NOT_IMPLEMENTED"
    rb = e_on.slow_readback()
    assert rb["implemented"] and rb["alive"] and rb["n_nonzero"] > 0
    with pytest.raises(ValueError):
        MouseCortexEngine(4, device="cpu", use_compile=False, verbose=False).configure_slow(
            True, inc_per_spike=0.0)


def test_layer_delay_bins_and_shared_ring():
    """E3b：到达步差 == 档差；多延迟档**共享单一环槽**；>4 档必须被拒绝。"""
    def arrival(delay):
        def mk(with_edge):
            e = MouseCortexEngine(3, device="cpu", use_compile=False, verbose=False)
            e.configure_subtypes(True)
            e.finalize()
            if with_edge:
                e.add_synapses(np.array([1]), np.array([2]), np.array([60.0], dtype=np.float32),
                               np.array([delay], dtype=np.int32), inhibitory=True,
                               syn_class=np.array([1], dtype=np.int8))
                e.finalize()
            b = np.zeros(3, dtype=np.float32); b[1] = 6000.0
            e.set_neuron_heterogeneity(b)
            e.reset(seed=0)
            return e
        a, b = mk(False), mk(True)
        for s in range(1, 400):
            a.run(1, delivery="step", record="counts")
            b.run(1, delivery="step", record="counts")
            if not np.array_equal(a.t_v.detach().cpu().numpy(), b.t_v.detach().cpu().numpy()):
                return s - 1
        return None
    assert arrival(80) - arrival(40) == 40
    net = _net()
    eng = MouseCortexEngine(N, device="cpu", use_compile=False, verbose=False)
    eng.configure_subtypes(True)
    eng.configure_layers(net["layer_of"], delay_bins=(40, 60, 80))
    attach_to_engine(eng, net)
    assert len({id(r) for r in eng._rings.values()}) == 1, "多延迟档必须共享单一环槽（E3b 打包）"
    with pytest.raises(ValueError):
        MouseCortexEngine(8, device="cpu", use_compile=False, verbose=False).configure_layers(
            np.zeros(8, dtype=np.int64), delay_bins=(40, 45, 50, 55, 60))


def test_segment_composability_and_dual_estimator():
    """(e) k=1/2/5 逐位一致；双估计器 ≤2%；绝对步号外部事件索引（E-M10-1 修正）。"""
    net = _net()
    st, ne, gg = _events()
    base = None
    for k in (1, 2, 5):
        eng = MouseCortexEngine(N, device="cpu", use_compile=False, verbose=False)
        eng.configure_subtypes(True)
        eng.configure_slow(True, tau_slow_ms=1000.0, inc_per_spike=6.0)
        eng.configure_layers(net["layer_of"], delay_bins=net["meta"]["delay_bins"])
        attach_to_engine(eng, net)
        eng.set_external_events(st, ne, gg, n_steps=2000)
        eng.reset(seed=1)
        eng.reset_counts()
        for _ in range(k):
            r = eng.run(2000 // k, delivery="step", record="counts")
        cnt = eng.t_count.detach().cpu().numpy().astype(np.int64)
        if base is None:
            base = (int(r["spk_total"]), cnt)
        assert int(r["spk_total"]) == base[0], "分段总脉冲必须逐位相同"
        assert np.array_equal(cnt, base[1]), "分段逐神经元计数必须逐位相同"
    d = r["dual_estimators"]
    assert d["rel_diff_count_vs_total"] <= 0.02
    # 行内硬断言（引擎内置）：pop_trace 与 spk_total 一致
    eng2 = MouseCortexEngine(N, device="cpu", use_compile=False, verbose=False)
    eng2.configure_subtypes(True)
    eng2.configure_layers(net["layer_of"], delay_bins=net["meta"]["delay_bins"])
    attach_to_engine(eng2, net)
    eng2.set_external_events(st, ne, gg, n_steps=2000)
    eng2.reset(seed=1)
    eng2.reset_counts()
    r2 = eng2.run(2000, delivery="step", record="counts", pop_trace=True)
    assert r2["dual_estimators"]["rel_diff_total_vs_pop"] <= 0.02


def test_two_comp_is_explicitly_unavailable():
    """E4b 未实现 ⇒ 必须显式拒绝，不得静默降级为 point。"""
    eng = MouseCortexEngine(8, device="cpu", use_compile=False, verbose=False)
    eng.finalize()
    with pytest.raises(NotImplementedError):
        eng.run(10, fidelity="two_comp")


# ===========================================================================
# RF-M11-F 回归测试（已知缺陷必须成为**被测试的显式状态**）
# ===========================================================================
def _expected_delivered(edges, cls_of_edge, firing, step, n_slot, n, n_ch):
    """独立期望：逐边算 (arrival_slot, cls, post, amp) 后按目标格**合并求和**（index_add_ 语义）。"""
    merged = {}
    for k, (pre, post, g, d) in enumerate(edges):
        if pre in firing:
            cls = int(cls_of_edge[k])
            if n_ch == 2 and cls > 0:
                cls = 1
            key = ((step + int(d)) % n_slot, cls, int(post))
            merged[key] = merged.get(key, 0.0) + float(g)
    return sorted((k[0], k[1], k[2], v) for k, v in merged.items())


def _actual_delivered(engine, firing, step):
    """由**写入前/后环槽差**反解实际落槽（与期望算途径完全独立）。"""
    ncn = int(engine.n_ch) * engine.n
    before = engine._ring.clone()
    v = torch.zeros(engine.n, dtype=torch.bool, device=engine.device)
    v[list(firing)] = True
    engine._deliver_step(v, step)
    after = engine._ring - before
    out = []
    for flat in (after != 0).nonzero(as_tuple=False).flatten().cpu().numpy():
        slot, rem = divmod(int(flat), ncn)
        cls, post = divmod(rem, engine.n)
        out.append((slot, int(cls), int(post), float(after[flat].item())))
    return sorted(out)


def test_sync_firing_delivery_no_raise_and_per_class_slots():
    """(f) 回归：**同一步多源发放**不得抛错，且逐类落槽必须与独立期望逐项相等。

    构造含"单源 2 条同类边打到同一目标"的最小网络（M9 冻结公式在此越界，M10 E5-FIX 修好）
    + 三条同类不同延迟档的边 + E/PV/SST/VIP 四类标签 ⇒ 覆盖 M10 L17/L21 的全部失效语境。
    """
    edges = [(0, 3, 11.0, 40), (0, 3, 13.0, 40), (0, 3, 23.0, 40),
             (1, 3, 31.0, 40), (1, 4, 37.0, 60), (2, 5, 41.0, 80)]
    cls = [0, 0, 0, 1, 2, 3]
    firing = [0, 1, 2]                       # 3 个源神经元在**同一步**发放
    for classes in (4, 2):
        e = MouseCortexEngine(6, device="cpu", use_compile=False, verbose=False)
        if classes == 4:
            e.configure_subtypes(True)
        e.finalize()
        e.configure_layers(np.zeros(6, dtype=np.int64), delay_bins=(40, 60, 80))
        c = np.asarray(cls, dtype=np.int8) if classes == 4 else np.asarray(
            [0, 0, 0, 1, 1, 1], dtype=np.int8)
        e.add_synapses(np.asarray([x[0] for x in edges]), np.asarray([x[1] for x in edges]),
                       np.asarray([x[2] for x in edges], dtype=np.float32),
                       np.asarray([x[3] for x in edges], dtype=np.int32),
                       inhibitory=(c > 0), syn_class=c)
        e.finalize()
        e.reset(seed=0)
        exp = _expected_delivered(edges, c.tolist(), firing, 0, e.n_slot, e.n, e.n_ch)
        got = _actual_delivered(e, firing, 0)          # 不得抛错
        assert got == exp, "同步发放下的逐类落槽与独立期望不一致：%s vs %s" % (got, exp)
        if classes == 4:
            # 逐类边数 == 逐类槽内边数（编码路径断言，装配方意图 vs 内核实际）
            assert e.class_slot_counts() == e.class_edge_counts() == {
                "E": 3, "PV": 1, "SST": 1, "VIP": 1}


@pytest.mark.parametrize("delivery", ["step", "chunk"])
def test_step_segment_composability_k_scan(delivery):
    """(e) 回归：step/chunk 两路径的 k=1/2/3/5/10 分段必须**逐位一致**（0.0%）。

    含外部事件（正是 RF-M11-F 中 step 口径回归的触发条件：冻结件按局部步号索引外部事件）。
    """
    net = _net()
    st, ne, gg = _events(T=2000)
    base = None
    for k in (1, 2, 3, 5, 10):
        e = MouseCortexEngine(N, device="cpu", use_compile=False, verbose=False)
        e.configure_subtypes(True)
        e.configure_slow(True, tau_slow_ms=1000.0, inc_per_spike=6.0)
        e.configure_layers(net["layer_of"], delay_bins=net["meta"]["delay_bins"])
        attach_to_engine(e, net)
        e.set_external_events(st, ne, gg, n_steps=2000)
        e.reset(seed=1)
        e.reset_counts()
        done = 0
        lens = [2000 // k + (1 if i < 2000 % k else 0) for i in range(k)]
        for ln in lens:
            e.run(ln, delivery=delivery, record="counts")
            done += ln
        assert done == 2000
        cnt = e.t_count.detach().cpu().numpy().astype(np.int64)
        tot = int(cnt.sum())
        if base is None:
            base = (tot, cnt)
        assert tot == base[0] and np.array_equal(cnt, base[1]), (
            "(e) 回归失败：delivery=%s k=%d 总脉冲 %d vs k=1 %d、逐神经元计数差异 %d 个"
            % (delivery, k, tot, base[0], int((cnt != base[1]).sum())))


def test_compose_matrix_and_domain_guard():
    """RF-M11-G：合成矩阵（密度×延迟档×E1b×k）全过 + **分段适用域守卫**必须拦冻结档。

    矩阵结论（`tools/scan_m11_compose_matrix.py`，96 格）：失败**只**由 `ext_abs_index=False` 决定
    （step ∧ 外部事件非空 ∧ k>1），与密度/延迟档数/E1b 无关 ⇒ 冻结档在 step 下不可分段。
    """
    net3 = _net(delay_bins=(40, 60, 80))
    net1 = _net(delay_bins=(40,))
    st, ne, gg = _events(T=1500)
    for delay_bins, n_ in (((40,), net1), ((40, 60, 80), net3)):
        for e1b in (4, 2):
            base = None
            for k in (1, 2, 5, 10):
                e = MouseCortexEngine(N, device="cpu", use_compile=False, verbose=False)
                if e1b == 4:
                    e.configure_subtypes(True)
                e.configure_layers(n_["layer_of"], delay_bins=delay_bins)
                cls = (n_["syn_class"] if e1b == 4
                       else np.where(n_["syn_class"] > 0, 1, 0).astype(np.int8))
                nb = dict(n_, syn_class=cls, inhibitory=(cls > 0))
                attach_to_engine(e, nb)
                e.set_external_events(st, ne, gg, n_steps=1500)
                e.assert_composable_for_segmentation("step")          # 守卫：默认档必须放行
                e.reset(seed=1)
                e.reset_counts()
                for ln in [1500 // k + (1 if i < 1500 % k else 0) for i in range(k)]:
                    e.run(ln, delivery="step", record="counts")
                cnt = e.t_count.detach().cpu().numpy().astype(np.int64)
                if base is None:
                    base = cnt
                assert np.array_equal(cnt, base), (
                    "(e) 矩阵格失败：delay_bins=%s E1b=%s k=%d 差异神经元=%d"
                    % (delay_bins, e1b, k, int((cnt != base).sum())))
    # 守卫：冻结档 + step + 外部事件 ⇒ 必须拒绝
    e = MouseCortexEngine(N, device="cpu", use_compile=False, verbose=False,
                          ext_abs_index=False)
    e_frozen = e
    e_frozen.configure_subtypes(True)
    e_frozen.configure_layers(net1["layer_of"], delay_bins=(40,))
    attach_to_engine(e_frozen, net1)
    e_frozen.set_external_events(st, ne, gg, n_steps=1500)
    with pytest.raises(RuntimeError):                                  # 冻结档 step 不可分段
        e_frozen.assert_composable_for_segmentation("step")
    e_frozen.assert_composable_for_segmentation("chunk")               # chunk 路径不受影响
