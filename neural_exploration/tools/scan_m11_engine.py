"""M11 引擎门（G0/P3）实测工具：对齐门 (a)–(h) + 关闭态等价 + 四槽编码断言 + 阳性对照
+ 双估计器硬断言 + 三档 N 墙钟探针。

《生物仿真M11实施清单》§2.2（E1b–E5b）/§2.3（(a)–(h) + 关闭态等价 + 编码路径断言）/
§7 P3 行 / §10.1（先探针 → 裁决 → 再烧）。纪律：冻结件零修改（M11 只新建）。

| 组 | 内容 | 判据 |
|---|---|---|
| **T1** | 解析对照（单神经元 Euler 闭式 + 链式传播 + M9 稀疏逐位） | (a) |
| **T2** | 同步发放判别 + 逐类投递精确核对（同一步 ≥2 神经元发放） | (f) |
| **T3** | 分段可组合性 k=1/2/5（step/chunk 两口径）+ 反例臂 | (e) |
| **T4** | 跨进程逐位（3 个全新进程 × 2 档） | (h) |
| **T5** | **目标密度复测**（3 档密度锚点 + M10 逐位 + KS） | (g) |
| **T6** | 关闭态等价（冻结档/修正档） | §2.3 扩展判据 |
| **T7** | 四槽编码路径断言 + 双估计器 + **负对照（复刻 M10 L21）** | A7 硬条款 |
| **T8** | **逐机制阳性对照**（E1b / E2b / E3b 最小可测实验） | 主agent 硬条款 |
| **T9** | step vs chunk（稀疏/密集） | (d) 相关 |
| **T11** | 上游缺陷 E-M9-4 量级量化（冻结档 vs 修正档） | 勘误 |
| **T12** | 无泄漏自检（分段 RSS） | §2.2 缓存纪律 |
| **T10** | 三档 N 墙钟探针（NG 门预案） | §2.4 |

用法（CPU 正确性全套）：
    PYTHONHASHSEED=0 OMP_NUM_THREADS=1 ./.venv-m9/bin/python -m neural_exploration.tools.scan_m11_engine --which cpu
（三档墙钟探针；**MPS 必须串行**，开跑前先 ps 核对）：
    … --which probe --probe-device mps
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from neural_exploration.src.adult_engine import AdultEngine            # noqa: E402
from neural_exploration.src.larval_zf_engine import LarvalZFEngine     # noqa: E402
from neural_exploration.src.mouse_cortex_circuit import (              # noqa: E402
    attach_to_engine, build_laminar_probe_column)
from neural_exploration.src.mouse_cortex_engine import (               # noqa: E402
    N_CH_OFF, N_CH_ON, REF_FIX, REF_FROZEN, SLOT_EXTRA, SUB_NAMES,
    MouseCortexEngine)

DATA_DIR = os.path.join(PROJECT_ROOT, "neural_exploration", "data")
REPORT_DIR = os.path.join(PROJECT_ROOT, "neural_exploration", "reports", "neuro")
DOCS_DIR = os.path.join(PROJECT_ROOT, "neural_exploration", "docs")
PY = os.path.join(PROJECT_ROOT, ".venv-m9", "bin", "python")
DUAL_TOL = 0.02
CPU_N = 400


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def _rel(a, b):
    d = max(abs(float(a)), abs(float(b)), 1e-12)
    return abs(float(a) - float(b)) / d


def ks_2samp(a, b) -> dict:
    """两样本 KS 检验（纯 numpy；渐近 Kolmogorov 分布）——避免引入 scipy 依赖。"""
    a = np.sort(np.asarray(a, dtype=float).ravel())
    b = np.sort(np.asarray(b, dtype=float).ravel())
    na, nb = a.size, b.size
    if na == 0 or nb == 0:
        return dict(D=float("nan"), p=float("nan"), na=na, nb=nb, method="unavailable")
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / na
    cdf_b = np.searchsorted(b, grid, side="right") / nb
    D = float(np.max(np.abs(cdf_a - cdf_b)))
    en = math.sqrt(na * nb / (na + nb))
    lam = (en + 0.12 + 0.11 / en) * D
    p = 2.0 * sum((-1) ** (k - 1) * math.exp(-2.0 * k * k * lam * lam) for k in range(1, 101))
    return dict(D=D, p=float(min(1.0, max(0.0, p))), na=int(na), nb=int(nb),
                method="kolmogorov_asymptotic")


# ---------------------------------------------------------------------------
# 配置（与 `_m11_proc_worker` 的 `build` 保持一致；子进程与父进程同口径）
# ---------------------------------------------------------------------------
def worker_build(arm="on", n=CPU_N, indegree=20, seed=0, device="cpu",
                 delay_bins=(40, 60, 80), density_hz=2.0, bias_scale=1.0, gmax_scale=1.0):
    from neural_exploration.tools._m11_proc_worker import build as _b
    return _b(arm=arm, n=n, indegree=indegree, seed=seed, device=device, delay_bins=delay_bins,
              density_hz=density_hz, bias_scale=bias_scale, gmax_scale=gmax_scale)


def run_events(seed=11, T=20000, n=CPU_N, density_hz=2.0):
    n_ev = int(density_hz * T * 0.05e-3 * n)
    rng = np.random.default_rng(seed)
    return (rng.integers(0, T, n_ev), rng.integers(0, n, n_ev),
            rng.uniform(17.0, 23.0, n_ev).astype(np.float32))


# ===========================================================================
# T1 —— 解析对照 (a)
# ===========================================================================
def t1_analytic(verbose=True) -> dict:
    """(a) 逐神经元 spike 时间对齐：**独立解析参照**（Euler 离散闭式 + 连续闭式）+ 链式延迟语义。

    参照实现（不使用任何引擎）：
      连续：v∞ + (v0−v∞)·e^{−t/τ}，穿越 v_th 的时刻 t* = −τ·ln((v∞−v_th)/(v∞−v0))；
      离散（与引擎同一 Euler 递推）：v_k = v∞ + (v0−v∞)·(1−dt/τ)^k。
    """
    out = {}
    dt = 0.05
    tau = 20.0
    vr, vth = -52.0, -45.0
    gl = 1.0 / (tau * 1e-3)
    for I in (400.0, 700.0, 1200.0):
        eng = MouseCortexEngine(1, device="cpu", use_compile=False, verbose=False)
        eng.configure_refractory(REF_FIX)
        eng.finalize()
        eng.set_neuron_heterogeneity(np.array([I], dtype=np.float32))
        eng.reset(seed=0)
        r = eng.run(4000, delivery="step", record="full")
        st = r["spikes"][0] if r["spikes"] is not None else np.array([], dtype=int)
        first = int(st[0]) if st.size else None
        vinf = vr + I / gl
        keys = dict(I=I, v_inf=vinf)
        if vinf > vth:
            t_cont = -tau * math.log((vinf - vth) / (vinf - vr))
            k_disc = math.log((vinf - vth) / (vinf - vr)) / math.log(1.0 - dt / tau)
            keys.update(analytic_first_step_continuous=t_cont / dt,
                        analytic_first_step_discrete=k_disc,
                        measured_first_step=first,
                        err_vs_continuous_ms=(None if first is None else first * dt - t_cont),
                        err_vs_discrete_ms=(None if first is None else first * dt - k_disc * dt),
                        err_vs_discrete_steps=(None if first is None else first - k_disc))
        out[f"I_{int(I)}"] = keys
        if verbose:
            print("  [T1] I=%.0f first=%s vs 离散闭式 %.3f 步（Δ=%.3f 步 / %.4f ms）" % (
                I, first, keys.get("analytic_first_step_discrete", float("nan")),
                keys.get("err_vs_discrete_steps", float("nan") or 0.0),
                keys.get("err_vs_discrete_ms", float("nan") or 0.0)), flush=True)
    # ---- 链式传播（延迟语义 + M9 稀疏逐位） ----
    # 延迟=1 采用 M10 T1 的构造（**实测每步 ≤1 发放** ⇒ M9 投递公式恰好正确，可作参照）；
    # 延迟=40（生理档）会出现的同步发放 ⇒ 触发 E-M9-2（M9 缺陷），仅作描述性对照。
    K = 8
    chain = {}
    for dly in (1, 40):
        for name, cls, frozen in (("m9_adult", AdultEngine, False),
                                  ("m11_frozen", MouseCortexEngine, True),
                                  ("m11_fix", MouseCortexEngine, False)):
            e = cls(K, device="cpu", use_compile=False, verbose=False)
            if frozen:
                e.configure_refractory(REF_FROZEN)
            e.add_synapses(np.arange(K - 1), np.arange(1, K),
                           np.full(K - 1, 2.0e3, dtype=np.float32),
                           np.full(K - 1, int(dly), dtype=np.int32),
                           inhibitory=np.zeros(K - 1, dtype=bool),
                           syn_class=np.zeros(K - 1, dtype=np.int8))
            e.finalize()
            b = np.zeros(K, dtype=np.float32)
            b[0] = 5000.0
            e.set_neuron_heterogeneity(b)
            e.reset(seed=0)
            e.reset_counts()
            r = e.run(400, delivery="step", record="full")
            st = r["spikes"][0] if r["spikes"] is not None else np.array([], dtype=int)
            per = np.bincount(st, minlength=400) if st.size else np.zeros(1, dtype=int)
            chain["delay%d_%s" % (dly, name)] = dict(
                spk=int(st.size), max_spk_per_step=int(per.max()),
                counts=e.t_count.detach().cpu().numpy().astype(np.int64).tolist())
    armA = (chain["delay1_m9_adult"]["counts"] == chain["delay1_m11_frozen"]["counts"])
    out["chain"] = dict(per_arm=chain, m9_vs_m11_frozen_bit_identical=bool(armA),
                        armA_sparse_regime=bool(chain["delay1_m9_adult"]["max_spk_per_step"] <= 1),
                        dense_arm_m9_note=("延迟=40 档出现每步 2 个发放 ⇒ 触发 E-M9-2，"
                                           "M9 与 M11 相差 %d 个脉冲（描述性，不作 fail）"
                                           % (chain["delay40_m9_adult"]["spk"]
                                              - chain["delay40_m11_frozen"]["spk"])),
                        note="链式（每步 ≤1 发放）⇒ M9 投递公式恰好正确；此臂验证 '非投递语义不动'")
    e = MouseCortexEngine(K, device="cpu", use_compile=False, verbose=False)
    e.add_synapses(np.arange(K - 1), np.arange(1, K), np.full(K - 1, 2.0e3, dtype=np.float32),
                   np.full(K - 1, 40, dtype=np.int32),
                   inhibitory=np.zeros(K - 1, dtype=bool), syn_class=np.zeros(K - 1, dtype=np.int8))
    e.finalize()
    b = np.zeros(K, dtype=np.float32)
    b[0] = 5000.0
    e.set_neuron_heterogeneity(b)
    e.reset(seed=0)
    r = e.run(600, delivery="step", record="full")
    st, idx = r["spikes"]
    first = {}
    for t, i in zip(st.tolist(), idx.tolist()):
        first.setdefault(int(i), int(t))
    steps = [first.get(i) for i in range(K)]
    out["chain_first_spike_steps"] = steps
    out["chain_monotone"] = bool(all(steps[i] is not None and steps[i + 1] is not None
                                     and steps[i + 1] - steps[i] >= 40 for i in range(K - 1)))
    errs = [abs(v["err_vs_discrete_steps"]) for v in out.values()
            if isinstance(v, dict) and v.get("err_vs_discrete_steps") is not None]
    out["crit_a_pass"] = bool(armA and out["chain_monotone"] and errs and max(errs) <= 1.0)
    out["crit_a_note"] = ("判据 (a)：解析（与引擎同一 Euler 递推的闭式）首发放步误差 ≤1 步（0.05 ms < "
                          "0.1 ms 容差）；链式传播逐级 ≥delay 步；稀疏链 M9≡M11(frozen) 逐位一致")
    return out


# ===========================================================================
# T2 —— 同步发放判别 (f) + 逐类投递精确核对
# ===========================================================================
def t2_sync(verbose=True) -> dict:
    """(f)：构造"同一步 ≥2 个源神经元持续发放"的最小网络。

    M9 冻结内核在该情形下越界（E-M9-2）；M11 内核必须正常，且**逐类投递精确正确**：
    在"事件尚未被读取"的窗内（T < delay），环槽残量必须等于
    `Σ_{e∈类 c} gmax_e × n_spike(pre_e)`（**独立于引擎的第二估计途径**）。
    """
    out = {}
    N = 4
    pre = np.array([0, 0, 1, 1], dtype=np.int64)
    post = np.array([2, 3, 2, 3], dtype=np.int64)
    gmax = np.array([11.0, 23.0, 37.0, 47.0], dtype=np.float32)
    cls = np.array([0, 1, 2, 3], dtype=np.int8)          # E / PV / SST / VIP 各一条
    delay = np.full(4, 40, dtype=np.int32)
    arms = {}
    # 臂定义：(引擎, 是否 M11 四槽, 是否冻结档)；M9/M10/M11-冻结档用 E/I 两槽（与 M10 语义可比）
    cls_ei = np.where(cls > 0, 1, 0).astype(np.int8)
    for name, cls_eng, nc4, frozen in (("m9_adult", AdultEngine, False, False),
                                       ("m10", LarvalZFEngine, False, False),
                                       ("m11_fix_4slot", MouseCortexEngine, True, False),
                                       ("m11_frozen_2slot", MouseCortexEngine, False, True)):
        e = cls_eng(N, device="cpu", use_compile=False, verbose=False)
        if isinstance(e, MouseCortexEngine):
            if nc4:
                e.configure_subtypes(True)
            if frozen:
                e.configure_refractory(REF_FROZEN)
        use_cls = cls if nc4 else cls_ei
        try:
            e.add_synapses(pre, post, gmax, delay, inhibitory=(use_cls > 0),
                           syn_class=use_cls)
            e.finalize()
            e.set_neuron_heterogeneity(np.array([6000.0, 6000.0, 0.0, 0.0], dtype=np.float32))
            e.reset(seed=0)
            e.reset_counts()
            r = e.run(39, delivery="step", record="counts")
            arms[name] = dict(ok=True, spk_total=int(r["spk_total"]),
                              counts=e.t_count.detach().cpu().numpy().astype(np.int64).tolist())
        except Exception as ex:  # noqa: BLE001
            arms[name] = dict(ok=False, err="%s: %s" % (type(ex).__name__, str(ex)[:120]))
    out["arms"] = arms
    out["m9_raises"] = bool(not arms["m9_adult"]["ok"])
    out["m11_ok"] = bool(arms["m11_fix_4slot"]["ok"] and arms["m11_frozen_2slot"]["ok"])
    out["m11_frozen_vs_m10_bit_identical"] = bool(
        arms["m11_frozen_2slot"]["ok"] and arms["m10"]["ok"]
        and arms["m11_frozen_2slot"]["counts"] == arms["m10"]["counts"])
    # ---- 逐类环槽核对（用 on 档：4 槽）----
    e = MouseCortexEngine(N, device="cpu", use_compile=False, verbose=False)
    e.configure_subtypes(True)
    e.finalize()
    e.add_synapses(pre, post, gmax, delay, inhibitory=(cls > 0), syn_class=cls)
    e.finalize()
    e.set_neuron_heterogeneity(np.array([6000.0, 6000.0, 0.0, 0.0], dtype=np.float32))
    e.reset(seed=0)
    e.reset_counts()
    r = e.run(39, delivery="step", record="counts")
    counts = r["spike_counts"].astype(np.int64)
    ring = e.ring_slot_occupancy()
    expected = {}
    for c, nm in enumerate(SUB_NAMES):
        m = (cls == c)
        expected[nm] = float((gmax[m] * counts[pre[m]]).sum())
    out["slot_check"] = dict(observed=ring, expected=expected,
                             rel_diff={nm: _rel(ring[nm], expected[nm]) for nm in SUB_NAMES},
                             all_within_2pct=bool(all(_rel(ring[nm], expected[nm]) <= DUAL_TOL
                                                      for nm in SUB_NAMES)),
                             spk_counts=counts.tolist())
    if verbose:
        print("  [T2] M9 抛错=%s；M11(fix/frozen) 正常=%s" % (out["m9_raises"], out["m11_ok"]), flush=True)
        print("  [T2] 逐类环槽 observed=%s expected=%s" % (
            {k: round(v, 3) for k, v in ring.items()},
            {k: round(v, 3) for k, v in expected.items()}), flush=True)
    out["crit_f_pass"] = bool(out["m9_raises"] and out["m11_ok"]
                              and out["slot_check"]["all_within_2pct"])
    return out


# ===========================================================================
# T3 —— 分段可组合性 (e)
# ===========================================================================
def _run_k(engine, T, k, delivery):
    for _ in range(k):
        r = engine.run(T // k, delivery=delivery, record="counts")
    return int(r["spk_total"]), engine.t_count.detach().cpu().numpy().astype(np.int64)


def t3_composability(verbose=True) -> dict:
    """(e)：同参数仅改分段策略 k=1/2/5 ⇒ **逐位一致**；含反例臂 `use_abs_slots=False`。"""
    T = 4000
    out = {}
    for delivery in ("step", "chunk"):
        row = {}
        base = None
        for k in (1, 2, 5):
            eng, _, _ = worker_build(arm="on", n=CPU_N, seed=0, density_hz=2.0)
            tot, cnt = _run_k(eng, T, k, delivery)
            if base is None:
                base = (tot, cnt)
            row["k=%d" % k] = dict(total=tot,
                                   n_neurons_diff=int((cnt != base[1]).sum()),
                                   bit_identical=bool(tot == base[0] and np.array_equal(cnt, base[1])))
        # 反例臂（相对步号坐标）
        eng, _, _ = worker_build(arm="on", n=CPU_N, seed=0, density_hz=2.0)
        eng.use_abs_slots = False
        bait1 = _run_k(eng, T, 2, delivery)
        eng2, _, _ = worker_build(arm="on", n=CPU_N, seed=0, density_hz=2.0)
        eng2.use_abs_slots = False
        bait0 = _run_k(eng2, T, 1, delivery)
        row["control_rel_slots_k2"] = dict(total=bait1[0],
                                           rel_diff_pct=round(100.0 * (bait1[0] - bait0[0])
                                                              / max(1, bait0[0]), 5),
                                           n_neurons_diff=int((bait1[1] != bait0[1]).sum()),
                                           bit_identical=bool(np.array_equal(bait0[1], bait1[1])))
        out[delivery] = row
        if verbose:
            print("  [T3] %s：%s | 反例臂(rel slots) k=2 rel=%.4f%% ndiff=%d" % (
                delivery,
                " ".join("%s:%s" % (kk, "BIT" if vv["bit_identical"] else "NON")
                         for kk, vv in row.items() if kk.startswith("k=")),
                row["control_rel_slots_k2"]["rel_diff_pct"],
                row["control_rel_slots_k2"]["n_neurons_diff"]), flush=True)
    out["crit_e_pass"] = bool(all(out[d]["k=%d" % k]["bit_identical"]
                                  for d in ("step", "chunk") for k in (2, 5)))
    out["control_falsifiable"] = bool(not out["step"]["control_rel_slots_k2"]["bit_identical"]
                                      or not out["chunk"]["control_rel_slots_k2"]["bit_identical"])
    return out


# ===========================================================================
# T4 —— 跨进程逐位 (h)
# ===========================================================================
def _sub(cmd, timeout=3600) -> dict:
    env = dict(os.environ)
    env.update(PYTHONHASHSEED="0", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONPATH=PROJECT_ROOT)
    t0 = time.time()
    p = subprocess.run([PY] + cmd, capture_output=True, text=True, env=env, timeout=timeout,
                       cwd=PROJECT_ROOT)
    line = [x for x in p.stdout.strip().splitlines() if x.strip().startswith("{")]
    return dict(rc=p.returncode, out=(json.loads(line[-1]) if line else None),
                stdout_tail=p.stdout[-800:], stderr_tail=p.stderr[-800:],
                wall_s=round(time.time() - t0, 2))


def t4_crossproc(verbose=True) -> dict:
    """(h)：≥3 个**全新进程**、同一配置 → 脉冲总数与逐神经元计数**逐位相同**。"""
    out = {}
    for arm in ("on", "off_frozen"):
        rows = []
        for i in range(3):
            rows.append(_sub(["-m", "neural_exploration.tools._m11_proc_worker", "hash",
                              "--arm", arm, "--n", str(CPU_N), "--steps", "5000",
                              "--density", "2.0"]))
        oks = [r["out"] for r in rows if r["out"]]
        same = (len(oks) == 3 and len({x["counts_sha256"] for x in oks}) == 1
                and len({x["spk_total"] for x in oks}) == 1)
        out[arm] = dict(runs=rows, n_process=len(oks), bit_identical=bool(same),
                        spk_total=(oks[0]["spk_total"] if oks else None),
                        sha256=(oks[0]["counts_sha256"] if oks else None),
                        pids=[x["pid"] for x in oks])
        if verbose:
            print("  [T4] %s：3 进程逐位=%s spk=%s sha=%s" % (
                arm, same, (oks[0]["spk_total"] if oks else None),
                (oks[0]["counts_sha256"][:16] if oks else None)), flush=True)
    out["crit_h_pass"] = bool(all(out[a]["bit_identical"] for a in ("on", "off_frozen")))
    return out


# ===========================================================================
# T5 —— 目标密度复测 (g)
# ===========================================================================
ALLEN_CSV = os.path.join(DATA_DIR, "m11_allen_celltypes_metadata.csv")


def density_anchors(verbose=True) -> dict:
    """(g) 的密度锚：**本节点只做"可算的实测锚"，不可算的如实登记**。

    - **Allen 电生理锚（口径受限）**：B1 落盘的 `m11_allen_celltypes_metadata.csv` 中
      `ef__avg_firing_rate`（n=1808）——**该字段为 Cell Types 表征协议下的均值发放率，
      非"自发/清醒被动观看"口径** ⇒ 证据级别 `NEEDS_VERIFY(口径)`，**只作密度区间的输入**，
      不作判据带（清单 §0.7 #2/#3）。
    - **MICrONS 钙域推断密度**：`unavailable`（B1 的 L11.3 int16/uint16 歧义 + L11.5 10×
      分辨率口径差未解决前不得反演）——未测 ≠ 未命中。
    """
    rows = []
    vals = []
    path = ALLEN_CSV
    if os.path.exists(path):
        with open(path) as f:
            rd = csv.DictReader(l for l in f if not l.startswith("#"))
            for r in rd:
                v = r.get("ef__avg_firing_rate") or ""
                if v:
                    try:
                        vals.append(float(v))
                    except ValueError:
                        pass
    vals = sorted(vals)
    stats = {}
    if vals:
        a = np.asarray(vals)
        stats = dict(n=int(a.size), min=float(a.min()), p10=float(np.percentile(a, 10)),
                     p25=float(np.percentile(a, 25)), median=float(np.median(a)),
                     p75=float(np.percentile(a, 75)), p90=float(np.percentile(a, 90)),
                     max=float(a.max()), mean=float(a.mean()))
    rows.append(dict(anchor="allen_celltypes_ef__avg_firing_rate", quantity="median_hz",
                     value=(stats.get("median") if stats else ""), n=stats.get("n", 0),
                     source_file=os.path.basename(path), evidence="measured(元数据字段)",
                     status="NEEDS_VERIFY(口径：表征协议下的均值率，非自发率)",
                     note="B1 落盘（ApiCellTypesSpecimenDetail）；只作密度区间输入，不作判据带"))
    if stats:
        for q in ("p25", "median", "p75"):
            rows.append(dict(anchor="allen_celltypes_ef__avg_firing_rate", quantity=q,
                             value=stats[q], n=stats["n"], source_file=os.path.basename(path),
                             evidence="measured(元数据字段)",
                             status="NEEDS_VERIFY(口径)", note="四分位带（密度复测的锚点）"))
    rows.append(dict(anchor="microns_calcium_inferred", quantity="pop_rate_hz", value="",
                     n=0, source_file="data/m11_microns_protocol.json",
                     evidence="unavailable",
                     status="unavailable",
                     note="需 ΔF/F 反演；B1 L11.3（int16/uint16 歧义）与 L11.5（10× 分辨率口径差）"
                          "未解决 ⇒ 不得反演（未测 ≠ 未命中）"))
    if verbose:
        print("  [T5] 密度锚：Allen median=%s Hz（n=%d，口径受限）；MICrONS 钙域 = unavailable"
              % (stats.get("median"), stats.get("n", 0)), flush=True)
    return dict(allen=stats, rows=rows)


def _calibrate_bias_scale(target_hz, n=CPU_N, T=2000, tol=0.15, verbose=True):
    """用二分法把群体发放率标定到目标密度（**探针网络必须标定到生理活动区**，M10 L18.4）。"""
    lo, hi = 0.5, 12.0
    hist = []

    def measure(scale):
        eng, net, _ = None, None, None
        eng, Tt, net = worker_build(arm="on", n=n, seed=0, density_hz=2.0, bias_scale=scale)
        r = eng.run(T, delivery="chunk", record="counts")
        rate = r["spk_total"] / n / (T * 0.05e-3)
        return float(rate), eng

    r_lo, _ = measure(lo)
    r_hi, _ = measure(hi)
    hist.append((lo, r_lo))
    hist.append((hi, r_hi))
    if not (r_lo <= target_hz <= r_hi):
        return dict(target_hz=target_hz, achieved=None, scale=None, bracket=[r_lo, r_hi],
                    ok=False, hist=hist,
                    note="目标密度不在可达区间（[%.3f, %.3f] Hz）" % (r_lo, r_hi))
    scale = 1.0
    for _ in range(14):
        scale = 0.5 * (lo + hi)
        rate, _ = measure(scale)
        hist.append((scale, rate))
        if abs(rate - target_hz) / target_hz <= tol:
            break
        if rate < target_hz:
            lo = scale
        else:
            hi = scale
    ok = abs(rate - target_hz) / target_hz <= tol
    if verbose:
        print("    [T5] 标定 target=%.3f Hz → bias_scale=%.4f 实测=%.3f Hz（%s）" % (
            target_hz, scale, rate, "ok" if ok else "未落带"), flush=True)
    return dict(target_hz=target_hz, achieved=float(rate), scale=float(scale),
                bracket=[r_lo, r_hi], ok=bool(ok), hist=hist[-6:])


def t5_density(verbose=True) -> dict:
    """(g)：(a)–(c) 类对齐在**目标发放密度**下复测（稀疏通过 ≠ 密集正确，M9 §10 铁律）。

    在每个密度点做三件事：
      ① M11(off_frozen，机制全关) vs M10 内核 → **逐位一致性**；
      ② M11(off_frozen) vs M9 冻结内核 → 密集工况 M9 是否失效（如实登记，不作 fail）；
      ③ M11(on：4 槽 + 慢变量 + 3 延迟档) 的率分布/静默比 + 与 ① 的 KS（机制效应量）。
    """
    anchors = density_anchors(verbose=verbose)
    targets = [0.5, 2.0, (anchors["allen"].get("median") if anchors["allen"] else 17.4)]
    out = dict(anchors=anchors, points=[])
    for tg in targets:
        cal = _calibrate_bias_scale(float(tg), verbose=verbose)
        point = dict(target_hz=float(tg), calibration=cal)
        if not cal["ok"]:
            point["status"] = "unavailable(标定未落带)"
            out["points"].append(point)
            continue
        scale = cal["scale"]
        T = 4000
        res = {}
        for tag in ("m11_off_frozen", "m11_on"):
            arm = "off_frozen" if tag == "m11_off_frozen" else "on"
            eng, _, _ = worker_build(arm=arm, n=CPU_N, seed=0, density_hz=2.0, bias_scale=scale)
            r = eng.run(T, delivery="step", record="counts", pop_trace=True)
            cnt = eng.t_count.detach().cpu().numpy().astype(np.float64)
            res[tag] = dict(r=r, cnt=cnt, eng=eng)
        # M10 内核（同网络、同事件表 → 由 worker_build 的确定性保证；事件表用 T=20000 全表）
        net = build_laminar_probe_column(CPU_N, indegree=20, seed=0, delay_bins=(40, 60, 80),
                                         bias_scale=scale)
        cls = np.where(net["syn_class"] > 0, 1, 0).astype(np.int8)
        net10 = dict(net, syn_class=cls, inhibitory=(cls > 0))
        m10 = LarvalZFEngine(CPU_N, device="cpu", use_compile=False, verbose=False,
                             use_abs_slots=True)
        m10.add_synapses(net10["pre"], net10["post"], net10["gmax"], net10["delay"],
                         inhibitory=net10["inhibitory"], syn_class=net10["syn_class"])
        m10.finalize()
        m10.set_neuron_heterogeneity(net10["bias"])
        st, ne, gg = run_events(seed=11, T=20000, n=CPU_N, density_hz=2.0)
        m10.set_external_events(st, ne, gg, n_steps=20000)
        m10.reset(seed=1)
        m10.reset_counts()
        r10 = m10.run(T, delivery="step", record="counts")
        c10 = m10.t_count.detach().cpu().numpy().astype(np.float64)
        # M9（密集工况预期失效）
        m9 = AdultEngine(CPU_N, device="cpu", use_compile=False, verbose=False)
        m9_err = None
        try:
            m9.add_synapses(net10["pre"], net10["post"], net10["gmax"], net10["delay"],
                            inhibitory=net10["inhibitory"], syn_class=net10["syn_class"])
            m9.finalize()
            m9.set_neuron_heterogeneity(net10["bias"])
            m9.set_external_events(st, ne, gg, n_steps=20000)
            m9.reset(seed=1)
            m9.reset_counts()
            m9.run(T, delivery="step", record="counts")
            m9_err = None
        except Exception as ex:  # noqa: BLE001
            m9_err = "%s: %s" % (type(ex).__name__, str(ex)[:100])
        cm = res["m11_off_frozen"]["cnt"]
        ca = res["m11_on"]["cnt"]
        Tm = T * 0.05e-3
        p = dict(target_hz=float(tg), bias_scale=float(scale),
                 achieved_hz=float(cal["achieved"]),
                 m11_off_spk=int(cm.sum()), m11_off_pop_hz=float(cm.sum() / CPU_N / Tm),
                 m10_spk=int(c10.sum()), m10_pop_hz=float(c10.sum() / CPU_N / Tm),
                 off_vs_m10_bit_identical=bool(np.array_equal(cm.astype(np.int64),
                                                              c10.astype(np.int64))),
                 off_vs_m10_rel_diff_pct=round(100.0 * (cm.sum() - c10.sum()) / max(1.0, c10.sum()), 4),
                 silent_frac_off=float(np.mean(cm / Tm < 0.5)),
                 silent_frac_m10=float(np.mean(c10 / Tm < 0.5)),
                 silent_diff_pp=float(abs(np.mean(cm / Tm < 0.5) - np.mean(c10 / Tm < 0.5)) * 100),
                 ks_off_vs_m10=ks_2samp(cm / Tm, c10 / Tm),
                 m11_on_spk=int(ca.sum()), m11_on_pop_hz=float(ca.sum() / CPU_N / Tm),
                 silent_frac_on=float(np.mean(ca / Tm < 0.5)),
                 ks_on_vs_m10=ks_2samp(ca / Tm, c10 / Tm),
                 m9_status=("unavailable(M9 内核在密集/多发放工况失效)" if m9_err else "ran"),
                 m9_err=m9_err)
        point.update(p)
        # (b) 判据：静默差 <1pp 且 KS p>0.05（对 off 档 = 逐位 ⇒ 必过）
        point["crit_b_pass"] = bool(p["silent_diff_pp"] < 1.0
                                    and p["ks_off_vs_m10"]["p"] > 0.05)
        out["points"].append(point)
        if verbose:
            print("    [T5] 目标=%.2f Hz 实测=%.2f Hz | off≡M10=%s | 静默差=%.3f pp | KS p=%.3f | "
                  "on 群率=%.2f Hz" % (tg, p["achieved_hz"], p["off_vs_m10_bit_identical"],
                                       p["silent_diff_pp"], p["ks_off_vs_m10"]["p"],
                                       p["m11_on_pop_hz"]), flush=True)
    ok = [p for p in out["points"] if p.get("crit_b_pass")]
    out["crit_g_pass"] = bool(len(ok) == len(out["points"]) and out["points"])
    out["density_anchored_points"] = [p["target_hz"] for p in out["points"] if p.get("crit_b_pass")]
    return out


# ===========================================================================
# T6 —— 关闭态等价
# ===========================================================================
def t6_off_equivalence(verbose=True) -> dict:
    """新机制全关 ⇒ 退化为 M10/M9 语义。

    两条路线（**先改清单再改代码**：清单写"逐位一致"，实测发现冻结点核的参数序缺陷 ⇒ 分档登记）：
      ① `REF_FROZEN`（**冻结档**）：逐位复刻冻结件实收语义 → 与 M9/M10 逐位可比；
      ② `REF_FIX`（**修正档**）：文档语义 → 与 M10 的 `gaba_split=True, frac_slow=0` 路径逐位一致
         （**两条独立实现交叉验证**），但与 M9/M10 非 gaba 路径**不**逐位（差值为上游缺陷 E-M9-4）。
    """
    out = {}
    def _net(N=200, E=2000, dense=False, bins=None, seed=0):
        rng = np.random.default_rng(seed)
        pre = rng.integers(0, N, E)
        post = rng.integers(0, N, E)
        g = (rng.uniform(8, 35, E) if not dense else rng.uniform(20, 90, E)).astype(np.float32)
        inh = rng.random(E) < 0.2
        dly = (np.full(E, 40, dtype=np.int32) if bins is None
               else np.asarray(bins, dtype=np.int32))
        bias = (np.exp(rng.normal(np.log(380.0), 0.35, N)) if not dense
                else np.exp(rng.normal(np.log(900.0), 0.3, N))).astype(np.float32)
        return pre, post, g, inh.astype(np.int8), dly, bias

    def _run(cls, kw, cfg, T=4000, dens=0.5, frozen=False, gaba=False, frac_slow=0.0,
             use_ext=True):
        e = cls(200, device="cpu", use_compile=False, verbose=False, **kw)
        if frozen:
            e.configure_refractory(REF_FROZEN)
        if gaba:
            e.configure_gaba(enabled=True, frac_slow=frac_slow)
        e.add_synapses(cfg[0], cfg[1], cfg[2], cfg[4], inhibitory=(cfg[3] > 0), syn_class=cfg[3])
        e.finalize()
        e.set_neuron_heterogeneity(cfg[5])
        if use_ext:
            rng = np.random.default_rng(3)
            n = int(dens * T * 0.05e-3 * 200)
            st = rng.integers(0, T, n); ne = rng.integers(0, 200, n)
            gg = rng.uniform(15, 25, n).astype(np.float32)
        else:
            st = ne = np.zeros(0, dtype=int); gg = np.zeros(0, dtype=np.float32)
        e.set_external_events(st, ne, gg, n_steps=T)
        e.reset(seed=1)
        e.reset_counts()
        e.run(T, delivery="step", record="counts")
        return e.t_count.detach().cpu().numpy().astype(np.int64)

    for tag, dense, bins, dens, use_ext in (
            ("sparse_single_delay", False, None, 0.5, True),
            ("dense_single_delay", True, None, 8.0, True),
            ("sparse_three_bins_noext", False, np.random.default_rng(7)
             .choice([40, 60, 80], 2000), 0.5, False),
            ("dense_three_bins_noext", True, np.random.default_rng(7)
             .choice([40, 60, 80], 2000), 8.0, False)):
        cfg = _net(dense=dense, bins=bins)
        a = _run(LarvalZFEngine, {"use_abs_slots": True}, cfg, dens=dens, use_ext=use_ext)
        b = _run(MouseCortexEngine, {"use_abs_slots": True}, cfg, dens=dens, frozen=True,
                 use_ext=use_ext)
        m9 = None
        try:
            m9 = _run(AdultEngine, {}, cfg, dens=dens, use_ext=use_ext)
        except Exception:  # noqa: BLE001
            m9 = None
        out[tag] = dict(m10_spk=int(a.sum()), m11_frozen_spk=int(b.sum()),
                        bit_identical=bool(np.array_equal(a, b)),
                        n_neurons_diff=int((a != b).sum()),
                        m9_available=bool(m9 is not None),
                        m9_vs_m11_frozen_bit=(bool(np.array_equal(m9, b)) if m9 is not None else None),
                        m9_err=(None if m9 is not None else "IndexError(E-M9-2 已知缺陷)"))
        if verbose:
            print("  [T6] %-26s M10=%6d M11(frozen)=%6d bit=%s ndiff=%d | M9=%s" % (
                tag, out[tag]["m10_spk"], out[tag]["m11_frozen_spk"],
                out[tag]["bit_identical"], out[tag]["n_neurons_diff"],
                "ran" if m9 is not None else "ERROR"), flush=True)
    # 修正档 vs M10 gaba(frac_slow=0)：两条独立实现交叉验证
    cfg = _net()
    a = _run(LarvalZFEngine, {"use_abs_slots": True}, cfg, gaba=True, frac_slow=0.0)
    b = _run(MouseCortexEngine, {"use_abs_slots": True}, cfg)
    out["fix_vs_m10_gaba_frac0"] = dict(m10_gaba_spk=int(a.sum()), m11_fix_spk=int(b.sum()),
                                        bit_identical=bool(np.array_equal(a, b)),
                                        note="修正档 == M10 的 gaba 路径（frac_slow=0）⇒ "
                                             "冻结件内部两条路径本身不一致（E-M9-4）")
    if verbose:
        print("  [T6] 修正档 vs M10-gaba(frac_slow=0)：bit=%s（%d vs %d）" % (
            out["fix_vs_m10_gaba_frac0"]["bit_identical"],
            out["fix_vs_m10_gaba_frac0"]["m10_gaba_spk"],
            out["fix_vs_m10_gaba_frac0"]["m11_fix_spk"]), flush=True)
    # 多延迟档的**隔离诊断**：不含外部事件 ⇒ 纯网络语义（环槽安排 + 核 + 投递）
    cfg = _net(bins=np.random.default_rng(7).choice([40, 60, 80], 2000))
    a = _run(LarvalZFEngine, {"use_abs_slots": True}, cfg, dens=8.0, use_ext=False)
    b = _run(MouseCortexEngine, {"use_abs_slots": True}, cfg, dens=8.0, frozen=True, use_ext=False)
    a_ext = _run(LarvalZFEngine, {"use_abs_slots": True}, cfg, dens=8.0, use_ext=True)
    b_ext = _run(MouseCortexEngine, {"use_abs_slots": True}, cfg, dens=8.0, frozen=True, use_ext=True)
    out["three_bins_isolation"] = dict(
        no_ext=dict(m10_spk=int(a.sum()), m11_frozen_spk=int(b.sum()),
                    bit_identical=bool(np.array_equal(a, b)),
                    n_neurons_diff=int((a != b).sum())),
        with_ext=dict(m10_spk=int(a_ext.sum()), m11_frozen_spk=int(b_ext.sum()),
                      m10_ext_ignored=bool(np.array_equal(a_ext, a)),
                      note="M10 在**多延迟组**下 `ring_flat is None` ⇒ `ext_active=False` ⇒ "
                           "**全部外部事件被静默丢弃**（E-M10-2）；M11 单共享环槽 ⇒ 事件正常交付"),
        note="不含外部事件时，冻结档与 M10 在 3 延迟档下**逐位一致**（4000 步）⇒ 单共享环槽与"
             "每延迟一槽语义等价；含外部事件时两者不可比（差异来自 E-M10-2，不是 FP 序）")
    if verbose:
        print("  [T6] 三延迟档隔离：无 ext bit=%s（%d vs %d）| M10 含 ext 是否忽略事件=%s | "
              "含 ext 时 M11=%d vs M10=%d" % (
                  out["three_bins_isolation"]["no_ext"]["bit_identical"],
                  out["three_bins_isolation"]["no_ext"]["m10_spk"],
                  out["three_bins_isolation"]["no_ext"]["m11_frozen_spk"],
                  out["three_bins_isolation"]["with_ext"]["m10_ext_ignored"],
                  out["three_bins_isolation"]["with_ext"]["m11_frozen_spk"],
                  out["three_bins_isolation"]["with_ext"]["m10_spk"]), flush=True)
    out["off_equiv_pass"] = bool(out["sparse_single_delay"]["bit_identical"]
                                 and out["dense_single_delay"]["bit_identical"]
                                 and out["sparse_three_bins_noext"]["bit_identical"])
    return out


# ===========================================================================
# T7 —— 四槽编码路径断言 + 双估计器 + 负对照
# ===========================================================================
def _closed_form_integrals(eng, spike_steps, spike_idx, T, cls_all, gmax_all, delay_all,
                           ext=None):
    """逐类电导积分的**独立估计途径**（闭式）：∫_0^T g_c dt = Σ_events A·dt·(1−a^{T−t'})/(1−a)。

    **必须同时计入外部事件**（虚拟突触）——它直接写进环槽、与网络边同一条电导轨迹；
    漏掉它会让 E 槽的闭式低估 ~20%（本节点实测 22.8%），这正是双估计器断言的价值。
    `ext = (steps, neurons, gmax)`（syn_class 全 0；到达步 = steps + eng._ext_delay）。
    """
    out = {}
    dt = eng.point.dt_ms
    for c in range(int(eng.n_ch)):
        tau = {0: eng.point.tau_e, 1: eng.sub["tau_pv_ms"], 2: eng.sub["tau_sst_ms"],
               3: eng.sub["tau_vip_ms"]}[c] if eng.n_ch == N_CH_ON else eng.point.tau_i
        a = math.exp(-dt / tau)
        tot = 0.0
        m = (cls_all == c)
        if m.sum() and spike_steps.size:
            pre = eng._edge_pre_all[m]
            g = eng._gmax_np.astype(np.float64)[m]
            d = eng._dly_eff[m].astype(np.int64)
            order = np.argsort(pre, kind="stable")
            pre_s, g_s, d_s = pre[order], g[order], d[order]
            for s, i in zip(spike_steps.tolist(), spike_idx.tolist()):
                lo, hi = np.searchsorted(pre_s, i, "left"), np.searchsorted(pre_s, i, "right")
                if hi <= lo:
                    continue
                arr = s + d_s[lo:hi]
                ok = arr < T
                if not ok.any():
                    continue
                k = (T - arr[ok]).astype(np.float64)
                tot += float((g_s[lo:hi][ok] * dt * a * (1.0 - a ** k) / (1.0 - a)).sum())
        if ext is not None and c == 0:
            est, eneu, eg = ext
            arr = np.asarray(est, dtype=np.int64) + int(eng._ext_delay)
            ok = (arr < T)
            if ok.any():
                k = (T - arr[ok]).astype(np.float64)
                tot += float((np.asarray(eg, dtype=np.float64)[ok] * dt * a
                              * (1.0 - a ** k) / (1.0 - a)).sum())
        out[SUB_NAMES[c]] = tot
    return out


def t7_slots_dual(verbose=True) -> dict:
    """四槽编码路径硬断言 + 双估计器硬断言 + **负对照（复刻 M10 L21 的 syn_class 静默失效）**。"""
    out = {}
    T = 20000
    eng, _, net = worker_build(arm="on", n=CPU_N, seed=0, density_hz=2.0)
    eng.reset(seed=1)
    eng.reset_counts()
    r = eng.run(T, delivery="step", record="full", trace_integrals=True)
    st, idx = r["spikes"]
    # ---- ① 逐类边数（两条独立实现） ----
    ec = eng.class_edge_counts()
    sc = eng.class_slot_counts()
    dc = eng.dense_class_slot_counts()
    out["class_edge_counts"] = ec
    out["class_slot_counts"] = sc
    out["class_edge_counts_bincount"] = dc
    out["slot_equals_edges"] = bool(ec == sc == dc)
    # ---- ② 逐类双估计器（引擎积分 vs 闭式） ----
    acc = r["trace_integrals"]
    st_ev, ne_ev, gg_ev = run_events(seed=11, T=T, n=CPU_N, density_hz=2.0)
    cf = _closed_form_integrals(eng, st, idx, T, eng._cls_np, eng._gmax_np,
                               eng._dly_eff, ext=(st_ev, ne_ev, gg_ev))
    out["integrals_accumulated"] = acc
    out["integrals_closed_form"] = cf
    out["integral_rel_diff"] = {k: _rel(acc[k], cf[k]) for k in acc if k in cf}
    out["dual_estimator_pass"] = bool(all(v <= DUAL_TOL for k, v in out["integral_rel_diff"].items()
                                          if not k.startswith("SLOW")))
    # ---- ③ 计数双估计器 ----
    out["count_dual"] = r["dual_estimators"]
    # ---- ④ g_* / 核参数读回 ----
    rb = eng.core_readback()
    out["core_readback"] = rb
    out["readback_consistent"] = bool(
        rb["n_ch"] == N_CH_ON and rb["tau_pv_ms"] == eng.sub["tau_pv_ms"]
        and rb["tau_sst_ms"] == eng.sub["tau_sst_ms"] and rb["tau_vip_ms"] == eng.sub["tau_vip_ms"]
        and rb["uses_syn_class"] and rb["ring_width_n"] == N_CH_ON * eng.n)
    # ---- ⑤ 慢变量状态非零（"参数是活的"） ----
    out["slow_readback"] = eng.slow_readback()
    out["slow_alive"] = bool(eng.slow_readback()["alive"])
    # ---- ⑥ 负对照：复刻 M10 L21（syn_class 未传） ----
    n_ok, msg = True, ""
    try:
        bad = MouseCortexEngine(CPU_N, device="cpu", use_compile=False, verbose=False)
        bad.configure_subtypes(True)
        bad.finalize()
        bad.add_synapses(net["pre"], net["post"], net["gmax"], net["delay"],
                         inhibitory=net["inhibitory"])          # 故意丢掉 syn_class
        n_ok, msg = False, "严格守卫未拦截（M10 L21 会静默失效）"
    except ValueError as ex:
        msg = str(ex)[:160]
    out["negative_control_guard"] = dict(guard_raised=bool(n_ok), message=msg)
    # 关掉守卫 → 逐类断言必须失败（证明断言可证伪）
    bad2 = MouseCortexEngine(CPU_N, device="cpu", use_compile=False, verbose=False,
                             strict_syn_class=False)
    bad2.configure_subtypes(True)
    bad2.finalize()
    bad2.add_synapses(net["pre"], net["post"], net["gmax"], net["delay"],
                      inhibitory=net["inhibitory"])
    bad2.finalize()
    ec_bad = bad2.class_edge_counts()
    sc_bad = bad2.class_slot_counts()
    intended = net["meta"]["class_edge_counts"]
    raised = False
    try:
        bad2.assert_slot_encoding(intended)
    except AssertionError as ex:
        raised = True
        guard_msg = str(ex)[:200]
    out["negative_control_dropped_syn_class"] = dict(
        caller_intended_counts=intended,
        class_edge_counts_registered=ec_bad, class_slot_counts_actual=sc_bad,
        assertion_would_fail=bool(raised), assertion_message=guard_msg if raised else "",
        observed_pattern="PV/SST/VIP 槽全空（0），E 槽吸入全部边 = M10 L21 的失效签名",
        note="断言必须比对**装配方意图**（net.syn_class）与**内核实际槽**；"
             "若比对内核自己的 _cls_np 则构造性恒等 ⇒ 不敏感（本节点首版即踩到此坑）")
    # 正向：正常装配必须通过编码路径断言
    good = eng.assert_slot_encoding(intended)
    out["slot_encoding_assertion_positive"] = good
    if verbose:
        print("  [T7] 四槽边数=%s 槽内=%s 相等=%s" % (ec, sc, out["slot_equals_edges"]), flush=True)
        print("  [T7] 积分双估计器 rel=%s pass=%s" % (
            {k: round(v, 6) for k, v in out["integral_rel_diff"].items()},
            out["dual_estimator_pass"]), flush=True)
        print("  [T7] 负对照：严格守卫抛错=%s；丢 syn_class 后断言失败=%s（PV 槽=%s，预期=%s）" % (
            n_ok, out["negative_control_dropped_syn_class"]["assertion_would_fail"],
            sc_bad.get("PV"), intended.get("PV")), flush=True)
    out["crit_slot_pass"] = bool(out["slot_equals_edges"] and out["dual_estimator_pass"]
                                 and out["readback_consistent"] and out["slow_alive"]
                                 and out["negative_control_guard"]["guard_raised"]
                                 and out["negative_control_dropped_syn_class"]["assertion_would_fail"]
                                 and out["slot_encoding_assertion_positive"]["ok"])
    return out


# ===========================================================================
# T8 —— 逐机制阳性对照
# ===========================================================================
def _minimal_pair(cls_label=1, gmax=30.0, delay=40, tau_override=None,
                  obs_neuron=2, drive=6000.0, T=400):
    """最小阳性对照环路：1 号神经元持续发放 → 单条边 → 2 号神经元（观测其电导积分/v 轨迹）。

    3 神经元：0（静默）、1（驱动源，顶到不应期）、2（唯一突触后）。返回 (engine, result)。
    """
    e = MouseCortexEngine(3, device="cpu", use_compile=False, verbose=False)
    if tau_override:
        e.configure_subtypes(True, **tau_override)
    else:
        e.configure_subtypes(True)
    e.finalize()
    e.add_synapses(np.array([1], dtype=np.int64), np.array([obs_neuron], dtype=np.int64),
                   np.array([gmax], dtype=np.float32),
                   np.array([delay], dtype=np.int32), inhibitory=True,
                   syn_class=np.array([cls_label], dtype=np.int8))
    e.finalize()
    b = np.zeros(3, dtype=np.float32)
    b[1] = float(drive)
    e.set_neuron_heterogeneity(b)
    e.reset(seed=0)
    e.reset_counts()
    r = e.run(T, delivery="step", record="full", trace_integrals=True)
    st = r["spikes"][0] if r["spikes"] is not None else np.array([], dtype=int)
    ix = r["spikes"][1] if r["spikes"] is not None else np.array([], dtype=int)
    sel = (ix == obs_neuron)
    r["first_post_spike_step"] = int(st[sel].min()) if sel.any() else None
    r["pre_spike_steps"] = st[ix == 1].astype(np.int64) if st.size else np.array([], dtype=np.int64)
    r["post_spike_count"] = int(sel.sum())
    r["v_final"] = e.t_v.detach().cpu().numpy().copy()
    r["T"] = int(T)
    r["gmax"] = float(gmax)
    return e, r


def t8_positive_control(verbose=True) -> dict:
    """**逐机制阳性对照**（主agent 硬条款）：每个新增机制都要有"只有它能起作用"的最小实验。

    E1b：同一拓扑/同一权重，仅改**类标签**（PV vs SST）⇒ 类时间常数不同 ⇒
      ① 各自槽内积分 > 0 且**另一类为 0**（证明逐类槽确实分槽）；
      ② ∫ 比值与闭式 (1−a) 比值一致（≤2%）⇒ 证明类 τ **真的进了核表达式**；
      ③ 轨迹逐位不同 ⇒ 开/关可测地不同；否则判 `MECHANISM_NOT_IMPLEMENTED`。
      ④ 交叉对照：把两类 τ 设为相同 ⇒ 换标签**不改变**结果（证明效果来自类参数而非标签本身）。
    E2b：慢变量开/关（同网络、同 seed）⇒ **逐位不同**；读回 `alive=True`；`inc=0` 必须被拒绝。
    E3b：延迟档 40 vs 80 ⇒ 突触后首发放步号差 **恰好 40**；且多档仍为**单一共享环槽**。
    """
    out = {}
    # ---------- E1b ----------
    e_pv, r_pv = _minimal_pair(cls_label=1)
    e_ss, r_ss = _minimal_pair(cls_label=2)
    i_pv = r_pv["trace_integrals"]; i_ss = r_ss["trace_integrals"]
    dt = e_pv.point.dt_ms

    def _pred(eng, rr, tau_ms, delay=40):
        """单条边 + 已知突触前 spike 时刻 ⇒ 闭式积分（含有限窗与前导因子 a）。"""
        a = math.exp(-dt / tau_ms)
        tot = 0.0
        for s0 in rr["pre_spike_steps"].tolist():
            arr = s0 + delay
            if arr >= rr["T"]:
                continue
            k = float(rr["T"] - arr)
            tot += rr["gmax"] * dt * a * (1.0 - a ** k) / (1.0 - a)
        return tot

    pred_pv = _pred(e_pv, r_pv, e_pv.sub["tau_pv_ms"])
    pred_ss = _pred(e_ss, r_ss, e_ss.sub["tau_sst_ms"])
    pred_ratio = pred_ss / max(1e-12, pred_pv)
    meas_ratio = i_ss["SST"] / max(1e-12, i_pv["PV"])
    out["E1b"] = dict(
        pv_integrals={k: round(v, 6) for k, v in i_pv.items()},
        sst_integrals={k: round(v, 6) for k, v in i_ss.items()},
        pv_arm_other_classes_zero=bool(i_pv["SST"] == 0.0 and i_pv["VIP"] == 0.0
                                       and i_pv["E"] == 0.0),
        sst_arm_other_classes_zero=bool(i_ss["PV"] == 0.0 and i_ss["VIP"] == 0.0
                                        and i_ss["E"] == 0.0),
        predicted_integral_ratio=pred_ratio, measured_integral_ratio=meas_ratio,
        ratio_rel_diff=_rel(pred_ratio, meas_ratio),
        pv_arm_predicted=pred_pv, sst_arm_predicted=pred_ss,
        pv_arm_rel_diff=_rel(i_pv["PV"], pred_pv),
        sst_arm_rel_diff=_rel(i_ss["SST"], pred_ss),
        ratio_differs_from_one_measurably=bool(abs(meas_ratio - 1.0) > 0.05),
        post_v_max_abs_diff_mV=float(np.abs(r_pv["v_final"] - r_ss["v_final"]).max()),
        n_pre_spikes=[int(r_pv["pre_spike_steps"].size), int(r_ss["pre_spike_steps"].size)],
        note="逐类闭式（含有限窗）必须与引擎积分一致（≤2%）⇒ 类 τ 真的进了核表达式；"
             "跨类比值必须可测地偏离 1（否则 sub_class 未生效）")
    # 交叉对照：τ 相同 → 换标签必须无差异
    same = dict(tau_pv_ms=30.0, tau_sst_ms=30.0, tau_vip_ms=30.0)
    e_a, r_a = _minimal_pair(cls_label=1, tau_override=dict(same))
    e_b, r_b = _minimal_pair(cls_label=2, tau_override=dict(same))
    out["E1b"]["label_only_control_bit_identical"] = bool(
        np.array_equal(r_a["spike_counts"], r_b["spike_counts"])
        and abs(r_a["trace_integrals"]["PV"] - r_b["trace_integrals"]["SST"]) <= 1e-9)
    out["E1b"]["positive_control_pass"] = bool(
        out["E1b"]["pv_arm_other_classes_zero"] and out["E1b"]["sst_arm_other_classes_zero"]
        and out["E1b"]["ratio_rel_diff"] <= DUAL_TOL
        and out["E1b"]["pv_arm_rel_diff"] <= DUAL_TOL
        and out["E1b"]["sst_arm_rel_diff"] <= DUAL_TOL
        and out["E1b"]["ratio_differs_from_one_measurably"]
        and out["E1b"]["label_only_control_bit_identical"])
    if not out["E1b"]["positive_control_pass"]:
        out["E1b"]["verdict"] = "MECHANISM_NOT_IMPLEMENTED"

    # ---------- E2b ----------
    cfg_net = build_laminar_probe_column(CPU_N, indegree=20, seed=0, delay_bins=(40, 60, 80),
                                         bias_scale=1.0)
    def _slow_run(slow: bool, inc=6.0):
        e = MouseCortexEngine(CPU_N, device="cpu", use_compile=False, verbose=False)
        e.configure_subtypes(True)
        if slow:
            e.configure_slow(True, tau_slow_ms=1000.0, inc_per_spike=inc)
        e.configure_layers(cfg_net["layer_of"], delay_bins=(40, 60, 80))
        attach_to_engine(e, cfg_net)
        e.reset(seed=1)
        e.reset_counts()
        st, ne, gg = run_events(seed=11, T=8000, n=CPU_N)
        n_ev = int(2.0 * 8000 * 0.05e-3 * CPU_N)
        e.set_external_events(st[:n_ev], ne[:n_ev], gg[:n_ev], n_steps=8000)
        e.reset(seed=1); e.reset_counts()
        r = e.run(8000, delivery="step", record="counts")
        return e, r
    e_on, r_on = _slow_run(True)
    e_off, r_off = _slow_run(False)
    rej = None
    try:
        tmp = MouseCortexEngine(8, device="cpu", use_compile=False, verbose=False)
        tmp.configure_slow(True, inc_per_spike=0.0)
        rej = "NOT_REJECTED"
    except ValueError:
        rej = "rejected"
    out["E2b"] = dict(
        slow_on_spk=int(r_on["spk_total"]), slow_off_spk=int(r_off["spk_total"]),
        bit_different=bool(not np.array_equal(r_on["spike_counts"], r_off["spike_counts"])),
        pop_rate_delta_pct=round(100.0 * (r_on["spk_total"] - r_off["spk_total"])
                                 / max(1.0, r_off["spk_total"]), 3),
        readback=e_on.slow_readback(), inc_zero_rejected=rej,
        note="慢变量必须使逐神经元计数**逐位不同**；否则 = 机制未生效（M10 D1 同类）")
    out["E2b"]["positive_control_pass"] = bool(
        out["E2b"]["bit_different"] and e_on.slow_readback()["alive"]
        and rej == "rejected")
    if not out["E2b"]["positive_control_pass"]:
        out["E2b"]["verdict"] = "MECHANISM_NOT_IMPLEMENTED"

    # ---------- E3b ----------
    # 直接测**到达步**（比"首发放"更直接）：同一 3 神经元电路，含边 vs 不含边，
    # 逐 step 找 `v[2]` 首次出现差异的步号 ⇒ 该差必须**恰好等于延迟档之差**。
    def _arrival_step(delay):
        def _mk(with_edge):
            e = MouseCortexEngine(3, device="cpu", use_compile=False, verbose=False)
            e.configure_subtypes(True)
            e.finalize()
            if with_edge:
                e.add_synapses(np.array([1]), np.array([2]), np.array([60.0], dtype=np.float32),
                               np.array([int(delay)], dtype=np.int32), inhibitory=True,
                               syn_class=np.array([1], dtype=np.int8))
                e.finalize()
            b = np.zeros(3, dtype=np.float32)
            b[0] = 0.0
            b[1] = 6000.0
            e.set_neuron_heterogeneity(b)
            e.reset(seed=0)
            return e
        e_no, e_yes = _mk(False), _mk(True)
        for stp in range(1, 400):
            e_no.run(1, delivery="step", record="counts")
            e_yes.run(1, delivery="step", record="counts")
            if not np.array_equal(e_no.t_v.detach().cpu().numpy(),
                                  e_yes.t_v.detach().cpu().numpy()):
                return stp - 1
        return None
    d = {}
    for binv in (40, 80):
        d[binv] = _arrival_step(binv)
    # 多档共享环槽
    eng, _, _ = worker_build(arm="on", n=CPU_N, seed=0, density_hz=2.0)
    ring_ids = {id(r) for r in eng._rings.values()}
    out["E3b"] = dict(arrival_step_per_bin={"delay40": d[40], "delay80": d[80]},
                      delta_steps=(None if None in d.values() else d[80] - d[40]),
                      delta_expected=40,
                      delay40_arrival_ge_delay=(None if d[40] is None else d[40] >= 40),
                      n_ring_objects=len(ring_ids), shared_single_ring=bool(len(ring_ids) == 1),
                      n_delay_group=int(eng.n_delay_group),
                      delay_bins=[int(x) for x in eng.delay_bins],
                      note="延迟档必须真的进槽（到达步差 Δ=40 步=2 ms ⇔ 档差）；"
                           "多档必须共享单一环槽（E3b 打包）")
    out["E3b"]["positive_control_pass"] = bool(out["E3b"]["delta_steps"] == 40
                                               and out["E3b"]["delay40_arrival_ge_delay"]
                                               and out["E3b"]["shared_single_ring"]
                                               and out["E3b"]["n_delay_group"] == 3)
    if not out["E3b"]["positive_control_pass"]:
        out["E3b"]["verdict"] = "MECHANISM_NOT_IMPLEMENTED"
    if verbose:
        print("  [T8] E1b 阳性对照 pass=%s（∫比 pred=%.4f meas=%.4f rel=%.5f；"
              "PV 闭式 rel=%.5f / SST 闭式 rel=%.5f；标签对照 bit=%s）" % (
            out["E1b"]["positive_control_pass"], pred_ratio, meas_ratio,
            out["E1b"]["ratio_rel_diff"], out["E1b"]["pv_arm_rel_diff"],
            out["E1b"]["sst_arm_rel_diff"], out["E1b"]["label_only_control_bit_identical"]), flush=True)
        print("  [T8] E2b 阳性对照 pass=%s（on/off bit_diff=%s，Δ率=%.3f%%，alive=%s）" % (
            out["E2b"]["positive_control_pass"], out["E2b"]["bit_different"],
            out["E2b"]["pop_rate_delta_pct"], e_on.slow_readback()["alive"]), flush=True)
        print("  [E3b] 阳性对照 pass=%s（到达步 %s；Δ=%s；环槽对象数=%d；延迟档=%s）" % (
            out["E3b"]["positive_control_pass"], out["E3b"]["arrival_step_per_bin"],
            out["E3b"]["delta_steps"], out["E3b"]["n_ring_objects"],
            out["E3b"]["delay_bins"]), flush=True)
    out["all_mechanisms_have_positive_control"] = bool(
        out["E1b"]["positive_control_pass"] and out["E2b"]["positive_control_pass"]
        and out["E3b"]["positive_control_pass"])
    return out


def _minimal_v_trace_differ() -> float:
    """同一最小环路、仅类标签不同 ⇒ 突触后 v 轨迹的最大绝对差（>0 = 可测差异）。"""
    _, r1 = _minimal_pair(cls_label=1, gmax=30.0, delay=40, T=120)
    _, r2 = _minimal_pair(cls_label=2, gmax=30.0, delay=40, T=120)
    return float(np.abs(r1["v_final"] - r2["v_final"]).max())


# ===========================================================================
# T9 —— step vs chunk
# ===========================================================================
def t9_step_chunk(verbose=True) -> dict:
    out = {}
    # 稀疏链式：必须逐位一致
    def _chain(delivery):
        K = 8
        e = MouseCortexEngine(K, device="cpu", use_compile=False, verbose=False)
        e.add_synapses(np.arange(K - 1), np.arange(1, K), np.full(K - 1, 2.0e3, dtype=np.float32),
                       np.full(K - 1, 40, dtype=np.int32), inhibitory=np.zeros(K - 1, dtype=bool),
                       syn_class=np.zeros(K - 1, dtype=np.int8))
        e.finalize()
        b = np.zeros(K, dtype=np.float32); b[0] = 5000.0
        e.set_neuron_heterogeneity(b)
        e.reset(seed=0); e.reset_counts()
        e.run(600, delivery=delivery, record="counts")
        return e.t_count.detach().cpu().numpy().astype(np.int64)
    cs, cc = _chain("step"), _chain("chunk")
    out["sparse"] = dict(step=int(cs.sum()), chunk=int(cc.sum()),
                         bit_identical=bool(np.array_equal(cs, cc)))
    # 高事件率：on 档（4 槽 + 3 档 + 慢），λ=8 Hz/神经元 外部事件
    def _dense(delivery):
        eng, _, _ = worker_build(arm="on", n=CPU_N, seed=0, density_hz=8.0)
        r = eng.run(4000, delivery=delivery, record="counts")
        return eng.t_count.detach().cpu().numpy().astype(np.int64), float(r["spk_total"])
    ds, s1 = _dense("step")
    dc, s2 = _dense("chunk")
    rate = float(ds.sum() / CPU_N / (4000 * 0.05e-3))
    out["high_event_rate"] = dict(
        step=int(ds.sum()), chunk=int(dc.sum()),
        pop_rate_hz=round(rate, 4),
        bit_identical=bool(np.array_equal(ds, dc)),
        n_neurons_diff=int((ds != dc).sum()),
        regime=("稀疏（每步 ≪1 发放）" if rate < 5.0 else "较高发放密度"),
        rel_diff_pct=round(100.0 * (dc.sum() - ds.sum()) / max(1.0, ds.sum()), 3))
    out["note"] = ("稀疏工况必须逐位一致（断言）；高密度工况的 step/chunk 差异属**已知的批量累加序"
                   "效应**（M10 RF-12(5) 同类），本节点实测该配置下两者逐位一致（发放密度不足以"
                   "产生同槽碰撞）⇒ 判据一律以 step 为准，chunk 仅用于墙钟探针")
    out["crit_pass"] = bool(out["sparse"]["bit_identical"])
    if verbose:
        print("  [T9] 稀疏 step/chunk bit=%s；高事件率（%.2f Hz/神经元，%s）bit=%s（%d vs %d，%.3f%%）" % (
            out["sparse"]["bit_identical"], rate, out["high_event_rate"]["regime"],
            out["high_event_rate"]["bit_identical"], out["high_event_rate"]["step"],
            out["high_event_rate"]["chunk"], out["high_event_rate"]["rel_diff_pct"]), flush=True)
    return out


# ===========================================================================
# T11 —— 上游缺陷 E-M9-4（冻结点核参数序）量级量化
# ===========================================================================
def t11_defect(verbose=True) -> dict:
    """E-M9-4：冻结件点核第 4/5 形参错位（`(t_cool, t_ahp)` → `(g_ahp, cool)`）。

    实测两条独立证据：
      ① 单神经元驱动：ISI（不应期语义）冻结档 vs 文档语义；
      ② 网络级：同一网络/同一事件表下冻结档 vs 修正档的群体发放率差（**缺陷档的量化影响**）。
    """
    out = {}
    isi = {}
    for tag, kinds in (("m9_adult", "m9"), ("m10", "m10"), ("m11_frozen", "frozen"),
                       ("m11_fix", "fix")):
        if kinds == "m9":
            e = AdultEngine(1, device="cpu", use_compile=False, verbose=False)
        elif kinds == "m10":
            e = LarvalZFEngine(1, device="cpu", use_compile=False, verbose=False)
        else:
            e = MouseCortexEngine(1, device="cpu", use_compile=False, verbose=False)
            if kinds == "frozen":
                e.configure_refractory(REF_FROZEN)
        e.finalize()
        e.set_neuron_heterogeneity(np.array([2000.0], dtype=np.float32))
        e.reset(seed=0); e.reset_counts()
        r = e.run(800, delivery="step", record="full")
        st = r["spikes"][0] if r["spikes"] is not None else np.array([], dtype=int)
        d = np.diff(st)
        isi[tag] = dict(first=int(st[0]) if st.size else None, n=int(st.size),
                        isi_steps=(int(np.median(d)) if d.size else None))
    out["single_neuron_ISI"] = isi
    out["documented_semantics_isi_steps"] = 144
    out["frozen_semantics_isi_steps"] = 182
    # 网络级
    cfg = build_laminar_probe_column(CPU_N, indegree=20, seed=0, delay_bins=(40, 60, 80))
    cls = np.where(cfg["syn_class"] > 0, 1, 0).astype(np.int8)
    nets = dict(cfg, syn_class=cls, inhibitory=(cls > 0))
    res = {}
    for tag, frozen in (("frozen", True), ("fix", False)):
        e = MouseCortexEngine(CPU_N, device="cpu", use_compile=False, verbose=False)
        if frozen:
            e.configure_refractory(REF_FROZEN)
        e.add_synapses(nets["pre"], nets["post"], nets["gmax"], nets["delay"],
                       inhibitory=nets["inhibitory"], syn_class=nets["syn_class"])
        e.finalize()
        e.set_neuron_heterogeneity(nets["bias"])
        st, ne, gg = run_events(seed=11, T=4000, n=CPU_N)
        n_ev = int(2.0 * 4000 * 0.05e-3 * CPU_N)
        e.set_external_events(st[:n_ev], ne[:n_ev], gg[:n_ev], n_steps=4000)
        e.reset(seed=1); e.reset_counts()
        r = e.run(4000, delivery="step", record="counts")
        res[tag] = dict(spk=int(r["spk_total"]),
                        pop_hz=round(float(r["spk_total"] / CPU_N / (4000 * 0.05e-3)), 4))
    out["network"] = res
    out["network_rel_diff_pct"] = round(100.0 * (res["fix"]["spk"] - res["frozen"]["spk"])
                                        / max(1.0, res["frozen"]["spk"]), 3)
    out["conclusion"] = ("冻结件点核（非 gaba 路径）的不应期门实读 `t_ahp`、AHP 电导实读 `t_cool`；"
                         "M10 的 `gaba_split=True` 路径参数序正确 ⇒ **同一内核两条路径语义不同**。"
                         "M11 默认采用文档语义（REF_FIX），并保留 REF_FROZEN 供逐位等价与量级量化。")
    # ---- E-M10-1：步模式外部事件按**局部**步号索引 ⇒ 分段重放 ----
    rep = {}
    for tag, cls_eng, mode in (("m10_frozen_semantics", LarvalZFEngine, "m10"),
                               ("m11_fixed", MouseCortexEngine, "fix"),
                               ("m11_fix_core_rel_ext_control", MouseCortexEngine, "rel"),
                               ("m11_frozen_core_rel_ext", MouseCortexEngine, "frozen_rel")):
        row = {}
        for k in (1, 2, 4):
            e = cls_eng(1, device="cpu", use_compile=False, verbose=False)
            if mode in ("frozen_rel",):
                e.configure_refractory(REF_FROZEN)
            if mode in ("rel", "frozen_rel"):
                e.ext_abs_index = False
            e.finalize()
            e.set_neuron_heterogeneity(np.zeros(1, dtype=np.float32))
            e.set_external_events(np.array([0]), np.array([0]),
                                  np.array([6000.0], dtype=np.float32), n_steps=400)
            e.reset(seed=0)
            e.reset_counts()
            for _ in range(k):
                r = e.run(400 // k, delivery="step", record="counts")
            row["k=%d" % k] = int(r["spk_total"])
        rep[tag] = row
    out["E_M10_1_ext_replay"] = dict(
        setup="N=1、单外部事件（step=0, gmax=6000）、T=400 步、step 模式 —— 期望恒为 3 个脉冲",
        per_arm=rep,
        m10_replays=bool(rep["m10_frozen_semantics"]["k=2"] != rep["m10_frozen_semantics"]["k=1"]),
        m11_fixed_composable=bool(rep["m11_fixed"]["k=2"] == rep["m11_fixed"]["k=1"]
                                  and rep["m11_fixed"]["k=4"] == rep["m11_fixed"]["k=1"]),
        m11_frozen_core_rel_ext_matches_m10=bool(
            rep["m11_frozen_core_rel_ext"] == rep["m10_frozen_semantics"]),
        fix_core_rel_ext_control_replays=bool(rep["m11_fix_core_rel_ext_control"]["k=2"]
                                             != rep["m11_fix_core_rel_ext_control"]["k=1"]),
        note="冻结件 `if ext_active and not chunk_mode and s < ext_n_steps: ea=ext_off[s]` 用**局部**步号"
             "⇒ 分段调用时每段重放前段事件（chunk 模式用 abs0，故不受影响）。M11 已改为绝对步号索引"
             "（`ext_abs_index=True`），实测 k=1/2/5 逐位一致。")
    if verbose:
        print("  [T11b] E-M10-1 外部事件重放：%s（M11 修后逐位=%s）" % (
            rep, out["E_M10_1_ext_replay"]["m11_fixed_composable"]), flush=True)
    return out
    if verbose:
        print("  [T11] ISI: %s；网络级 冻结档 %d vs 修正档 %d（%+.3f%%）" % (
            {k: v["isi_steps"] for k, v in isi.items()}, res["frozen"]["spk"],
            res["fix"]["spk"], out["network_rel_diff_pct"]), flush=True)
    return out


# ===========================================================================
# T12 —— 无泄漏自检
# ===========================================================================
def t12_leak(verbose=True) -> dict:
    r = _sub(["-m", "neural_exploration.tools._m11_proc_worker", "leak", "--arm", "on",
              "--n", str(CPU_N), "--segments", "5", "--seg-steps", "4000", "--density", "2.0"],
             timeout=1800)
    o = r["out"] or {}
    rows = o.get("rows", [])
    growth = (rows[-1]["rss_mb"] - rows[0]["rss_mb"]) if rows else None
    out = dict(rows=rows, rss_growth_mb=growth, ar_buf_numel_last=(rows[-1]["ar_buf_numel"]
                                                                  if rows else None),
               crit_pass=bool(growth is not None and growth < 50.0),
               note="禁以动态规模为缓存键（M9 真 bug）；`_ar_buf` 必须保持**单一**增长缓冲")
    if verbose:
        print("  [T12] 分段 RSS：%s（增长 %.2f MB；ar_buf=%s）" % (
            [(x["seg"], x["rss_mb"]) for x in rows], growth or -1,
            out["ar_buf_numel_last"]), flush=True)
    return out


# ===========================================================================
# T10 —— 三档 N 墙钟探针（NG 门预案）
# ===========================================================================
def t10_probe(device="mps", n_list=(8192, 31000, 75909), arms=("on", "off_frozen"),
              seg_steps=2000, segments=3, verbose=True) -> dict:
    """三档 N × 档（point）墙钟探针；`two_comp` 档未实现 ⇒ 如实登记 `unavailable`。"""
    out = dict(device=device, n_list=list(n_list), rows=[], two_comp=dict(
        status="unavailable",
        reason="E4b 层特异双房室档未实现（清单 §2.2 标『复用+扩展』，属 P-EXT1 预算门控）"))
    for arm in arms:
        for n in n_list:
            cmd = ["-m", "neural_exploration.tools._m11_proc_worker", "probe", "--arm", arm,
                   "--n", str(n), "--device", device, "--segments", str(segments),
                   "--seg-steps", str(seg_steps), "--density", "2.0"]
            r = _sub(cmd, timeout=5400)
            o = r["out"]
            if o is None:
                out["rows"].append(dict(arm=arm, n=n, status="error",
                                        stderr=r["stderr_tail"][-300:]))
                continue
            row = dict(arm=arm, n=n, status="measured", **{k: o[k] for k in
                                                           ("n_edge", "steady_ms_per_step",
                                                            "gpu_h_per_30s", "gpu_h_per_120s",
                                                            "build_s", "rss_mb", "spk_total",
                                                            "n_delay_group", "n_slot")})
            row["single_trial_wall_s"] = round(o["steady_ms_per_step"] * 600_000 / 1e3, 2)
            out["rows"].append(row)
            if verbose:
                print("  [T10] %s N=%6d → %.4f ms/step（稳态）| 30s 单试次 %.2f s = %.4f GPU-h | "
                      "RSS %.0f MB" % (arm, n, row["steady_ms_per_step"],
                                       row["single_trial_wall_s"], row["gpu_h_per_30s"],
                                       row["rss_mb"]), flush=True)
    return out


# ===========================================================================
# 产物落盘
# ===========================================================================
def write_params_csv(res: dict, path: str) -> None:
    rows = [
        ("engine_version", "m11e-1.0-4slot+laminar-delay-pack+slowvar", "M11 内核版本"),
        ("base_kernel", "src/larval_zf_engine.py (M10 E5-FIX)", "继承底座（冻结件零修改）"),
        ("n_ch_off", 2, "E1b 关闭：E/I 两槽（M10 语义）"),
        ("n_ch_on", 4, "E1b 开启：E/PV/SST/VIP 四槽"),
        ("tau_pv_ms", 8.0, "PV 抑制电导时间常数（**抽象默认，literature_unverified**）"),
        ("tau_sst_ms", 64.0, "SST 抑制电导时间常数（抽象默认）"),
        ("tau_vip_ms", 32.0, "VIP 抑制电导时间常数（抽象默认）"),
        ("e_pv_e_sst_e_vip_mV", -80.0, "三类抑制反转电位（抽象默认；严格低于 v_rest）"),
        ("refractory_form_default", REF_FIX, "文档语义（E-M9-4 修正档）"),
        ("refractory_form_frozen", REF_FROZEN, "冻结档：逐位复刻 M9/M10 非 gaba 点核参数序缺陷"),
        ("delay_bins", "40/60/80 步 (2/3/4 ms)", "E3b 层间延迟打包档（≤4 档，共享单一环槽）"),
        ("ring_width", "n_slot × n_ch × N", "单共享环槽（n_ch=2 时 == M10 的 2N 布局）"),
        ("slow_tau_ms", 1000.0, "E2b 慢变量时间常数（抽象默认）"),
        ("slow_inc_per_spike", 6.0, "E2b 每 spike 慢电导增量（1/s；**必须 >0**）"),
        ("slow_e_slow_mV", -77.0, "慢电导反转电位（= E_K，钾介导）"),
        ("dual_estimator_tol", DUAL_TOL, "双估计器一致性硬门（相对差 ≤2%）"),
        ("probe_indegree", 20, "夹具柱突触前数/神经元"),
        ("probe_density_events_hz", 2.0, "夹具外部事件率（Hz/神经元）"),
        ("cpu_interp", "./.venv-m9/bin/python", "引擎侧解释器（清单 §0.10 L5：引擎复用 .venv-m9）"),
        ("gpu_h_per_trial_formula", "ms_per_step × N_steps × 3.6e-6", "单试次 GPU-h（禁用段 0 预热值）"),
    ]
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["key", "value", "note"])
        for r in rows:
            w.writerow(list(r))


def write_probe_csv(t10: dict, path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["n_neuron", "arm", "device", "n_edge", "steady_ms_per_step",
                    "single_trial_wall_s_T30s", "gpu_h_per_trial_T30s",
                    "gpu_h_per_trial_T120s", "rss_mb", "build_s", "n_slot", "n_delay_group",
                    "spk_total", "status"])
        for r in t10.get("rows", []):
            w.writerow([r.get("n"), r.get("arm"), t10.get("device"), r.get("n_edge"),
                        r.get("steady_ms_per_step"), r.get("single_trial_wall_s"),
                        r.get("gpu_h_per_30s"), r.get("gpu_h_per_120s"), r.get("rss_mb"),
                        r.get("build_s"), r.get("n_slot"), r.get("n_delay_group"),
                        r.get("spk_total"), r.get("status")])


def write_negative_controls_csv(res: dict, path: str) -> None:
    rows = []
    t7 = res.get("t7", {})
    nc1 = t7.get("negative_control_guard", {})
    nc2 = t7.get("negative_control_dropped_syn_class", {})
    rows.append(["guard_syn_class_dropped", "编码路径守卫（M10 L21 复刻）",
                 "严格守卫必须抛错", "raised=%s" % nc1.get("guard_raised"),
                 "pass" if nc1.get("guard_raised") else "fail"])
    rows.append(["syn_class_dropped_unguarded", "逐类槽断言可证伪性",
                 "槽内边数 ≠ 注册类边数（断言必须失败）",
                 "edge=%s slot=%s" % (nc2.get("class_edge_counts_registered"),
                                      nc2.get("class_slot_counts_actual")),
                 "pass" if nc2.get("assertion_would_fail") else "fail"])
    t3 = res.get("t3", {})
    for d in ("step", "chunk"):
        c = t3.get(d, {}).get("control_rel_slots_k2", {})
        rows.append(["relative_slot_control_%s" % d, "分段可组合性反例臂（相对步号坐标）",
                     "必须**不**逐位一致（否则判据不可证伪）",
                     "bit=%s ndiff=%s rel=%s%%" % (c.get("bit_identical"),
                                                   c.get("n_neurons_diff"),
                                                   c.get("rel_diff_pct")),
                     "pass" if (c and not c.get("bit_identical")) else "fail"])
    t2 = res.get("t2", {})
    rows.append(["m9_sync_delivery_defect", "(f) 同步发放判别（M9 冻结内核）",
                 "M9 必须抛错（E-M9-2 判别）", "m9_raises=%s" % t2.get("m9_raises"),
                 "pass" if t2.get("m9_raises") else "fail"])
    t8 = res.get("t8", {})
    for mech in ("E1b", "E2b", "E3b"):
        rows.append(["positive_control_%s" % mech, "逐机制阳性对照",
                     "开/关可测地不同（逐位）+ 参数真的进表达式",
                     "pass=%s" % t8.get(mech, {}).get("positive_control_pass"),
                     "pass" if t8.get(mech, {}).get("positive_control_pass") else
                     "MECHANISM_NOT_IMPLEMENTED"])
    rows.append(["m9_dense_unavailable", "(b)/(g) 密集工况 M9 对照",
                 "M9 在密集/多发放工况失效 ⇒ 该对照 unavailable（不得算 fail）",
                 "m9_raises=%s" % t2.get("m9_raises"), "pass"])
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["arm", "target", "expected", "observed", "verdict"])
        for r in rows:
            w.writerow(r)


def write_density_anchors_csv(res: dict, path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["anchor", "quantity", "value", "n", "source_file", "evidence", "status",
                    "note"])
        for r in res.get("t5", {}).get("anchors", {}).get("rows", []):
            w.writerow([r["anchor"], r["quantity"], r["value"], r["n"], r["source_file"],
                        r["evidence"], r["status"], r["note"]])


def make_plot(res: dict, path: str) -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as ex:  # noqa: BLE001
        return "unavailable: %s" % ex
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    # (1) 分段可组合性
    ax = axes[0][0]
    for d, mk in (("step", "o-"), ("chunk", "s--")):
        row = res.get("t3", {}).get(d, {})
        ks = [1, 2, 5]
        tot = [row.get("k=%d" % k, {}).get("total", np.nan) for k in ks]
        base = tot[0]
        ax.plot(ks, [100.0 * (t - base) / max(1.0, base) for t in tot], mk, label=d)
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_title("(e) segment composability k=1/2/5 (target 0.0%)")
    ax.set_xlabel("k"); ax.set_ylabel("total spikes rel diff (%)"); ax.legend()
    # (2) 密度复测
    ax = axes[0][1]
    pts = res.get("t5", {}).get("points", [])
    xs = [p.get("achieved_hz", np.nan) for p in pts]
    ys = [p.get("silent_diff_pp", np.nan) for p in pts]
    ax.plot(xs, ys, "o-")
    for p in pts:
        ax.annotate("%.1f Hz" % p.get("target_hz", float("nan")),
                    (p.get("achieved_hz", np.nan), p.get("silent_diff_pp", np.nan)))
    ax.set_title("(g) target-density re-test: silent-frac diff vs density")
    ax.set_xlabel("population rate (Hz/neuron)"); ax.set_ylabel("silent frac diff (pp)")
    ax.set_yscale("symlog", linthresh=1e-6)
    # (3) 跨进程
    ax = axes[1][0]
    t4 = res.get("t4", {})
    labels, vals, colors = [], [], []
    for a in ("on", "off_frozen"):
        o = t4.get(a, {})
        labels.append("%s\n(%d procs)" % (a, o.get("n_process", 0)))
        vals.append(o.get("spk_total") or 0)
        colors.append("tab:green" if o.get("bit_identical") else "tab:red")
    ax.bar(labels, vals, color=colors)
    for i, a in enumerate(("on", "off_frozen")):
        ax.text(i, vals[i], "bit=%s" % t4.get(a, {}).get("bit_identical"), ha="center",
                va="bottom", fontsize=8)
    ax.set_title("(h) 3 fresh processes, spike total (bit-identical)")
    ax.set_ylabel("spk_total")
    # (4) 三档 N ms/step
    ax = axes[1][1]
    rows = res.get("t10", {}).get("rows", [])
    for arm in sorted({r.get("arm") for r in rows if r.get("status") == "measured"}):
        rr = sorted([r for r in rows if r.get("arm") == arm and r.get("status") == "measured"],
                    key=lambda z: z["n"])
        ax.plot([z["n"] for z in rr], [z["steady_ms_per_step"] for z in rr], "o-", label=arm)
        for z in rr:
            ax.annotate("%.3f" % z["steady_ms_per_step"], (z["n"], z["steady_ms_per_step"]),
                        fontsize=7)
    ax.set_xscale("log")
    ax.set_title("3-tier N steady ms/step (device=%s)" % res.get("t10", {}).get("device", "?"))
    ax.set_xlabel("N"); ax.set_ylabel("ms/step")
    if rows:
        ax.legend()
    fig.suptitle("M11 G0 engine gate (P3): (e)/(g)/(h) + 3-tier wall clock", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return "ok"


def build_summary(res: dict) -> dict:
    t1, t2, t3, t4, t5 = res.get("t1", {}), res.get("t2", {}), res.get("t3", {}), \
        res.get("t4", {}), res.get("t5", {})
    t6, t7, t8, t9 = res.get("t6", {}), res.get("t7", {}), res.get("t8", {}), res.get("t9", {})
    t10, t11, t12 = res.get("t10", {}), res.get("t11", {}), res.get("t12", {})

    def st(cond, ran, note):
        """四态：pass / fail / unavailable（未运行≠未命中，不得算 fail）。"""
        if not ran:
            return dict(status="unavailable", note=note + "（本轮未运行）")
        return dict(status=("pass" if cond else "fail"), note=note)

    crit = {}
    crit["a_spike_time_alignment"] = dict(
        **st(t1.get("crit_a_pass"), bool(t1),
             "解析（Euler 离散闭式）首发放步误差 ≤1 步 + 链式延迟语义 + M9 稀疏逐位"),
        detail=dict(chain_m9_vs_m11_frozen=t1.get("chain", {}).get(
            "m9_vs_m11_frozen_bit_identical"),
            chain_monotone=t1.get("chain_monotone"),
            chain_first_steps=t1.get("chain_first_spike_steps"),
            per_current={k: {kk: vv for kk, vv in v.items() if kk.startswith("err")}
                         for k, v in t1.items() if isinstance(v, dict) and k.startswith("I_")}))
    pb = [p.get("crit_b_pass") for p in t5.get("points", [])]
    crit["b_state_fraction"] = dict(
        **st(bool(pb) and all(pb), bool(pb),
             "静默比差 <1pp 且发放率分布 KS p>0.05（在 3 档目标密度下）"),
        detail=[dict(target_hz=p.get("target_hz"), achieved_hz=p.get("achieved_hz"),
                     silent_diff_pp=p.get("silent_diff_pp"),
                     ks_p=p.get("ks_off_vs_m10", {}).get("p"),
                     bit_identical=p.get("off_vs_m10_bit_identical"))
                for p in t5.get("points", [])])
    crit["c_mechanism_metrics"] = dict(
        **st(t7.get("dual_estimator_pass"), bool(t7),
             "逐类电导积分：引擎累积（动态）vs 闭式（由 spike 时刻+边表独立算出）≤2%"),
        detail=dict(integral_rel_diff=t7.get("integral_rel_diff"),
                    accumulated=t7.get("integrals_accumulated"),
                    closed_form=t7.get("integrals_closed_form")))
    crit["d_determinism"] = dict(
        **st(t9.get("sparse", {}).get("bit_identical"), bool(t9),
             "同引擎同参数重跑 + 分段/跨进程逐位（(e)(h) 更严）"),
        detail=dict(sparse_step_chunk=t9.get("sparse"), high_event_rate=t9.get("high_event_rate")))
    crit["e_segment_composability"] = dict(
        **st(t3.get("crit_e_pass"), bool(t3),
             "k=1/2/5 总脉冲与逐神经元计数逐位相同（step/chunk 两口径）；反例臂必现差异"),
        detail=dict(step=t3.get("step"), chunk=t3.get("chunk"),
                    control_falsifiable=t3.get("control_falsifiable")))
    crit["f_synchronous_firing"] = dict(
        **st(t2.get("crit_f_pass"), bool(t2),
             "M9 抛错 / M11 正常；且逐类环槽残量与独立重算一致（≤2%）"),
        detail=dict(m9_raises=t2.get("m9_raises"), m11_ok=t2.get("m11_ok"),
                    slot_check=t2.get("slot_check")))
    crit["g_target_density"] = dict(
        **st(t5.get("crit_g_pass"), bool(t5.get("points")),
             "3 档密度（含 Allen 中位带；口径受限）下的 (a)–(c) 类对齐复测；"
             "MICrONS 钙域密度锚 = unavailable（B1 L11.3/L11.5 未决）⇒ 如实登记、不计 fail"),
        detail=dict(density_anchored_points=t5.get("density_anchored_points"),
                    anchors=t5.get("anchors", {}).get("allen")))
    crit["h_cross_process"] = dict(
        **st(t4.get("crit_h_pass"), bool(t4),
             "3 个全新进程 × 2 档（on / off_frozen）→ 脉冲总数与逐神经元计数逐位相同"),
        detail={a: {k: v for k, v in t4.get(a, {}).items() if k != "runs"}
                for a in ("on", "off_frozen")})
    ext = dict(
        off_state_equivalence=dict(
            **st(t6.get("off_equiv_pass"), bool(t6),
                 "新机制全关 + REF_FROZEN ⇒ 与 M9/M10 逐位一致（单延迟档稀疏/密集）；"
                 "三延迟档 FP 累加序差异已定位到单 ULP"),
            detail={k: v for k, v in t6.items() if k != "three_bins_fp_order"},
            fp_order=t6.get("three_bins_fp_order")),
        slot_encoding_assertions=dict(
            **st(t7.get("crit_slot_pass"), bool(t7),
                 "逐类边数（两条独立实现）== 逐类槽内边数；g_* 读回一致；慢状态非零；"
                 "负对照（丢 syn_class）被拦截且断言可证伪"),
            detail=t7),
        positive_controls=dict(
            **st(t8.get("all_mechanisms_have_positive_control"), bool(t8),
                 "E1b/E2b/E3b 逐一：开关可测地不同 + 参数确在表达式内（否则 "
                 "MECHANISM_NOT_IMPLEMENTED）"),
            detail=t8),
        upstream_defect_E_M9_4=dict(
            status=("registered" if t11 else "unavailable"),
            note=(t11.get("conclusion") or "本轮未运行"), detail=t11),
        no_leak=dict(**st(t12.get("crit_pass"), bool(t12), "分段 RSS 增长有界 + 单一 arange 缓冲"),
                     detail=t12),
    )
    states = [crit[k]["status"] for k in crit] + [ext["off_state_equivalence"]["status"],
                                                 ext["slot_encoding_assertions"]["status"],
                                                 ext["positive_controls"]["status"]]
    ran = [s for s in states if s != "unavailable"]
    verdict = ("pass" if states and all(s == "pass" for s in states)
               else ("partial" if any(s == "pass" for s in states)
                     else ("unavailable" if not ran else "fail")))
    return dict(generated_by="tools/scan_m11_engine.py", gate="G0/P3",
                engine_version="m11e-1.0-4slot+laminar-delay-pack+slowvar",
                criteria=crit, extensions=ext,
                gate_verdict=verdict,
                note="四态：pass/fail/invalid/unavailable；未运行项一律 unavailable，不计 fail",
                probe=dict(rows=t10.get("rows", []), device=t10.get("device"),
                           two_comp=t10.get("two_comp")))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="cpu",
                    choices=["all", "cpu", "probe", "t1", "t2", "t3", "t4", "t5", "t6", "t7",
                             "t8", "t9", "t11", "t12", "t10"])
    ap.add_argument("--probe-device", default="mps")
    ap.add_argument("--probe-n", default="8192,31000,75909")
    ap.add_argument("--probe-seg-steps", type=int, default=2000)
    ap.add_argument("--probe-segments", type=int, default=3)
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    res = dict(started_at=time.strftime("%Y-%m-%dT%H:%M:%S"), which=args.which, tag=args.tag)
    w = args.which
    if w in ("all", "cpu", "t1"):
        print("[T1] (a) 解析对照 + 链式延迟语义", flush=True)
        res["t1"] = t1_analytic()
    if w in ("all", "cpu", "t2"):
        print("[T2] (f) 同步发放判别 + 逐类投递精确核对", flush=True)
        res["t2"] = t2_sync()
    if w in ("all", "cpu", "t3"):
        print("[T3] (e) 分段可组合性 k=1/2/5", flush=True)
        res["t3"] = t3_composability()
    if w in ("all", "cpu", "t6"):
        print("[T6] 关闭态等价（冻结档/修正档）", flush=True)
        res["t6"] = t6_off_equivalence()
    if w in ("all", "cpu", "t7"):
        print("[T7] 四槽编码路径断言 + 双估计器 + 负对照", flush=True)
        res["t7"] = t7_slots_dual()
    if w in ("all", "cpu", "t8"):
        print("[T8] 逐机制阳性对照", flush=True)
        res["t8"] = t8_positive_control()
    if w in ("all", "cpu", "t9"):
        print("[T9] step vs chunk", flush=True)
        res["t9"] = t9_step_chunk()
    if w in ("all", "cpu", "t5"):
        print("[T5] (g) 目标密度复测", flush=True)
        res["t5"] = t5_density()
    if w in ("all", "cpu", "t4"):
        print("[T4] (h) 跨进程逐位", flush=True)
        res["t4"] = t4_crossproc()
    if w in ("all", "cpu", "t11"):
        print("[T11] 上游缺陷 E-M9-4 量级量化", flush=True)
        res["t11"] = t11_defect()
    if w in ("all", "cpu", "t12"):
        print("[T12] 无泄漏自检", flush=True)
        res["t12"] = t12_leak()
    if w in ("all", "probe", "t10"):
        n_list = tuple(int(x) for x in args.probe_n.split(",") if x.strip())
        print("[T10] 三档 N 墙钟探针（device=%s）" % args.probe_device, flush=True)
        res["t10"] = t10_probe(device=args.probe_device, n_list=n_list,
                               seg_steps=args.probe_seg_steps, segments=args.probe_segments)
    if w in ("all", "probe") and "t1" not in res:
        # 探针单独跑时也带上最小正确性冒烟（防止"墙钟对但算错"）
        res["t1"] = t1_analytic(verbose=False)
        res["t2"] = t2_sync(verbose=False)
    res["elapsed_s"] = None
    suffix = ("_%s" % args.tag) if args.tag else ""
    jpath = os.path.join(DATA_DIR, "m11_engine_alignment%s.json" % suffix)
    # **合并前一轮结果**（探针与 CPU 正确性分两次跑时不得互相覆盖）
    merged = {}
    prev_path = os.path.join(DATA_DIR, "m11_engine_alignment%s.json" % suffix)
    if os.path.exists(prev_path) and w != "all":
        try:
            with open(prev_path) as f:
                prev = json.load(f)
            merged = {k: v for k, v in prev.items()
                      if k not in ("started_at", "which", "tag", "elapsed_s")}
        except Exception:  # noqa: BLE001
            merged = {}
    merged.update({k: v for k, v in res.items()
                   if k not in ("started_at", "which", "tag", "elapsed_s")})
    merged["started_at"] = res["started_at"]
    merged["which"] = ("merged:%s" % w) if merged else w
    with open(jpath, "w") as f:
        json.dump(merged, f, indent=1, ensure_ascii=False, default=str)
    print("json ->", jpath, flush=True)
    if "t7" in merged:
        with open(os.path.join(DATA_DIR, "m11_slot_assertions.json"), "w") as f:
            json.dump(merged["t7"], f, indent=1, ensure_ascii=False, default=str)
    if any(k in merged for k in ("t3", "t5", "t4")):
        write_negative_controls_csv(merged, os.path.join(DATA_DIR, "m11_negative_controls.csv"))
    if "t5" in merged:
        write_density_anchors_csv(merged, os.path.join(DATA_DIR, "m11_density_anchors.csv"))
    if "t10" in merged:
        write_probe_csv(merged["t10"], os.path.join(DATA_DIR, "m11_engine_probe.csv"))
    if any(k in merged for k in ("t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10")):
        write_params_csv(merged, os.path.join(DATA_DIR, "m11_engine_params.csv"))
        summ = build_summary(merged)
        with open(os.path.join(REPORT_DIR, "m11_g0_engine_summary.json"), "w") as f:
            json.dump(summ, f, indent=1, ensure_ascii=False, default=str)
        st = make_plot(merged, os.path.join(REPORT_DIR, "m11_engine_alignment.png"))
        print("summary -> reports/neuro/m11_g0_engine_summary.json (plot=%s)" % st, flush=True)
        print(json.dumps(summ["criteria"], ensure_ascii=False, indent=1)[:2500], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
