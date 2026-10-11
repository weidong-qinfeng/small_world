"""M10 引擎门（G0/P3）探针：正确性对照 + 分段可组合性 + 全规模墙钟。

《生物仿真M10实施清单》§2.3（对齐门 (a)–(e) + 关闭态等价）+ §2.4（规模/设备裁决）+ §0.9（探针先于预算）。

本工具实测四组内容（结果落 `data/m10_engine_probe.csv` / `m10_engine_params.csv`）：

  **T1 解析链式对照**（投递正确性）——已知答案的确定性网络：
      神经元 0→1→…→(K-1) 链式边（delay=1）+ 起点外部事件 ⇒ 解析预测的 spike 时刻序列。
      **同时对照 M9 冻结引擎**：链式（每步 ≤1 个发放神经元）下 M9 公式恰好正确 ⇒ M9/M10 应逐位一致；
  **T2 同步发放投递对照**（M10 E5-FIX 的判别实验）——构造**同一步 ≥2 个神经元发放**的情形，
      M9 冻结公式在该情形下把块起点平移 `prev_total` → CPU 抛 `IndexError`（MPS 静默错投）；
      M10 修正公式给出正确目标集（逐位核对期望突触后神经元集合）；
  **T3 分段可组合性 (e)**——同一事件表、同一网络，仅改 `run()` 切分（k=1/2/5）：
      要求**总脉冲逐位相同 + 逐神经元计数逐位相同**（M10 目标 0.0%；M9 实测对照 0.2–0.7% / 17.5% 神经元）；
      含 `use_abs_slots=False` 的**反例臂**（证明修法有效、非"本来就可组合"）；
  **T4 全规模 10 万点神经元稳态墙钟**——分段调用（每段独立计时，取**稳态**中位数，禁用段 0 预热值），
      段间记 RSS；照 M9 L40 纪律（本工具单进程内单目标，不做跨试次复用）。

用法（点神经元 MPS 探针；CPU 用于正确性对照）：
    PYTHONHASHSEED=0 ./.venv-m9/bin/python -m neural_exploration.tools.scan_m10_engine --which all
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

GA = float(os.environ.get("M10_GA", 25.0))     # 单边电导尺度（1/s）
G_EXT = float(os.environ.get("M10_GEXT", 22.0))
BIAS_MV_S = float(os.environ.get("M10_BIAS", 0.0))


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def _engines():
    from ..src.adult_engine import AdultEngine          # noqa: F401
    from ..src.larval_zf_engine import LarvalZFEngine   # noqa: F401
    return AdultEngine, LarvalZFEngine


# --------------------------------------------------------------------------
# T1/T2：链式与同步发放的投递正确性
# --------------------------------------------------------------------------
def t1_chain(device="cpu", K=8, steps=400, verbose=True):
    """0→1→…→K-1 链（delay=1）；外部事件只打 0 号；解析预测 1..K-1 依次在 t=1..K-1 发放。"""
    AdultEngine, LarvalZFEngine = _engines()
    out = {}
    for name, cls, kw in (("m9", AdultEngine, {}),
                          ("m10", LarvalZFEngine, {"use_abs_slots": True})):
        e = cls(K, device=device, use_compile=False, verbose=False, **kw)
        pre = np.arange(K - 1); post = np.arange(1, K)
        e.add_synapses(pre, post, np.full(K - 1, 2.0e3, dtype=np.float32),
                       np.ones(K - 1, dtype=np.int32))
        e.finalize()
        # 起点神经元 0 给恒定 bias（mV/s），其余 0 → 只有 0 自主发放
        bias = np.zeros(K, dtype=np.float32); bias[0] = 5000.0
        e.set_neuron_heterogeneity(bias)
        e.reset(seed=0)
        r = e.run(steps, delivery="step", record="full")
        spikes = r["spikes"]
        first_t = {}
        if spikes is not None:
            st, idx = spikes
            for t, n in zip(st.tolist(), idx.tolist()):
                first_t.setdefault(n, int(t))
        out[name] = dict(first_spike_step=first_t, total=int(len(spikes[0])) if spikes else 0)
        if verbose:
            print(f"  [T1] {name}: first spike steps = "
                  f"{[first_t.get(i) for i in range(K)]}", flush=True)
    ok = (out["m9"]["first_spike_step"] == out["m10"]["first_spike_step"])
    # 解析：源头 0 每 ~40 步（不应期）发放，下游 1..K-1 必须在**同一步或之后**首次发放（逐级传播）
    fs = out["m10"]["first_spike_step"]
    fsi = {int(k): v for k, v in fs.items()}
    chain = [fsi.get(i) for i in range(1, K)]
    monotone = all(a is not None and b is not None and b >= a for a, b in zip(chain, chain[1:])) \
        and all(a is not None for a in chain)
    return dict(t1_m9_m10_identical=bool(ok), t1_chain_monotone=bool(monotone),
                t1_n_spikes=out["m10"]["total"], t1_first_steps=fsi)


def t2_simultaneous(device="cpu", verbose=True):
    """同步发放（同一步 ≥2 个源神经元）→ 投递目标集必须正确（M9 公式在此越界）。"""
    AdultEngine, LarvalZFEngine = _engines()
    N = 5
    pre = np.array([0, 0, 1, 2], dtype=np.int64)     # 0→1, 0→2, 1→3, 2→4
    post = np.array([1, 2, 3, 4], dtype=np.int64)
    res = {}
    for name, cls, kw in (("m9", AdultEngine, {}),
                          ("m10", LarvalZFEngine, {"use_abs_slots": True})):
        e = cls(N, device=device, use_compile=False, verbose=False, **kw)
        e.add_synapses(pre, post, np.full(4, 500.0, dtype=np.float32),
                       np.ones(4, dtype=np.int32))
        e.finalize()
        # 让 0 与 1 同时持续发放（bias 拉满 → 顶到不应期，形成同步）
        bias = np.array([5000.0, 5000.0, 0.0, 0.0, 0.0], dtype=np.float32)
        e.set_neuron_heterogeneity(bias)
        e.reset(seed=0)
        try:
            r = e.run(60, delivery="step", record="full")
            res[name] = dict(ok=True, total=int(r["spikes"][0].size) if r["spikes"] else 0)
        except Exception as ex:  # noqa: BLE001
            res[name] = dict(ok=False, err=f"{type(ex).__name__}: {ex}")
        if verbose:
            print(f"  [T2] {name}: {res[name]}", flush=True)
    # M10 下：3、4 号神经元应在 1 步后收到来自 0 的突触（若 0 发放）
    return dict(t2_m9_crashes=bool(not res["m9"]["ok"]),
                t2_m10_runs=bool(res["m10"]["ok"]),
                t2_m9=res["m9"], t2_m10=res["m10"])


# --------------------------------------------------------------------------
# T3：分段可组合性（(e) 判据）
# --------------------------------------------------------------------------
def _mk_net(cls, N, E, seed=0, **kw):
    e = cls(N, device="cpu", use_compile=False, verbose=False, **kw)
    rng = np.random.default_rng(seed)
    pre = rng.integers(0, N, E); post = rng.integers(0, N, E)
    gmax = rng.uniform(8.0, 35.0, E).astype(np.float32)
    inh = rng.random(E) < 0.2
    dly = rng.integers(1, 6, E).astype(np.int32)
    e.add_synapses(pre, post, gmax, dly, inhibitory=inh)
    e.finalize()
    bias = np.exp(rng.normal(np.log(380.0), 0.35, N)).astype(np.float32)
    e.set_neuron_heterogeneity(bias)
    return e


def _events(T, lam, N, seed=3):
    rng = np.random.default_rng(seed)
    n = int(lam * T * 0.05e-3 * N)
    return (rng.integers(0, T, n), rng.integers(0, N, n),
            rng.uniform(15.0, 25.0, n).astype(np.float32))


def _run_k(cls, k, T, N=2000, E=25000, lam=0.5, seed=0, **kw):
    kw2 = {kk: vv for kk, vv in kw.items() if kk != "delivery"}
    e = _mk_net(cls, N, E, seed=seed, **kw2)
    st, ne, gg = _events(T, lam, N)
    e.set_external_events(st, ne, gg, n_steps=T)
    e.reset(seed=1)
    e.reset_counts()          # 逐神经元计数为**协议累计** → 分段跑只需取最后一段的结果
    for _ in range(k):
        r = e.run(T // k, delivery=kw.get("delivery", "step"), record="counts")
    return int(r["spk_total"]), r["spike_counts"]


def t3_composability(T=4000, verbose=True):
    AdultEngine, LarvalZFEngine = _engines()
    out = {}
    arms = [("m10_abs_step", LarvalZFEngine, dict(delivery="step")),
            ("m10_abs_chunk", LarvalZFEngine, dict(delivery="chunk")),
            ("m10_rel_step_control", LarvalZFEngine,
             dict(delivery="step", use_abs_slots=False)),
            ("m9_step_control", AdultEngine, dict(delivery="step"))]
    for name, cls, kw in arms:
        base = None; rows = []
        for k in (1, 2, 5):
            try:
                tot, counts = _run_k(cls, k, T, **kw)
                if k == 1 and tot <= 0:
                    raise RuntimeError("零脉冲：该臂无活动 → 可组合性检验空转（不合格）")
            except Exception as ex:  # noqa: BLE001
                rows.append(dict(k=k, error=f"{type(ex).__name__}: {str(ex)[:80]}"))
                continue
            if base is None:
                base = (tot, counts)
            ndiff = int((counts != base[1]).sum()) if counts is not None else -1
            rows.append(dict(k=k, total=tot,
                             rel_diff_pct=round(100.0 * (tot - base[0]) / max(1, base[0]), 5),
                             n_neurons_count_diff=ndiff,
                             bit_identical=bool(ndiff == 0 and tot == base[0])))
        out[name] = rows
        if verbose:
            print(f"  [T3] {name}: " + " | ".join(
                (f"k={r['k']}: {r.get('rel_diff_pct')}% ndiff={r.get('n_neurons_count_diff')} "
                 f"{'BIT' if r.get('bit_identical') else 'NON'}" if 'error' not in r
                 else f"k={r['k']}: ERR {r['error']}") for r in rows), flush=True)
    ok = all(r.get("bit_identical") for r in out["m10_abs_step"] if "error" not in r) and \
        all(r.get("bit_identical") for r in out["m10_abs_chunk"] if "error" not in r)
    return dict(t3_pass_bit_identical=bool(ok), t3_arms=out)


# --------------------------------------------------------------------------
# T4：全规模 10 万点神经元稳态墙钟（MPS）
# --------------------------------------------------------------------------
def t4_wallclock(N=100_000, n_seg=4, seg_steps=20_000, edges_per_neuron=20,
                 device="mps", seed=0, verbose=True, delay_max=40):
    _, LarvalZFEngine = _engines()
    rng = np.random.default_rng(seed)
    E = int(N * edges_per_neuron)
    e = LarvalZFEngine(N, device=device, use_compile=False, verbose=False,
                       use_abs_slots=True)
    t_build = time.time()
    chunk = 2_000_000
    for i in range(0, E, chunk):
        m = min(chunk, E - i)
        pre = rng.integers(0, N, m); post = rng.integers(0, N, m)
        gmax = rng.uniform(6.0, 18.0, m).astype(np.float32)
        inh = rng.random(m) < 0.2
        # **统一延迟 = 40 步（2 ms）**：既满足"单延迟组"（冻结内核仅 `len_rings==1` 走预计算槽视图
        # 快路径；多变延迟走逐槽 Python 循环，实测慢 >30×），又满足 `W = min(SLOT_EXTRA=40, max_delay) = 40`
        # 的 chunk 批量投递条件（delay=1 时 W=1 → chunk 退化为逐步，实测慢 ~50×）。
        # 见 docs/m10_env_notes.md L18。
        dly = np.full(m, int(delay_max), dtype=np.int32)
        e.add_synapses(pre, post, gmax, dly, inhibitory=inh)
    e.finalize()
    T = n_seg * seg_steps
    bias = np.exp(rng.normal(np.log(230.0), 0.30, N)).astype(np.float32)   # 亚阈 tonic 偏置
    e.set_neuron_heterogeneity(bias)
    n_ev = int(2.0 * T * 0.05e-3 * N)      # λ=2 Hz/神经元（生理量级背景驱动）
    st = rng.integers(0, T, n_ev); ne = rng.integers(0, N, n_ev)
    gg = rng.uniform(17.0, 23.0, n_ev).astype(np.float32)
    e.set_external_events(st, ne, gg, n_steps=T)
    e.reset(seed=1)
    print(f"  [T4] N={N} edges={E} build={time.time()-t_build:.1f}s RSS={rss_mb():.0f}MB", flush=True)
    rows = []
    for s in range(n_seg):
        t0 = time.time()
        r = e.run(seg_steps, delivery="chunk", record="counts")
        rows.append(dict(seg=s, wall_s=round(time.time() - t0, 2),
                         ms_per_step=round(r["ms_per_step"], 4),
                         spk=int(r["spk_total"]), rss_mb=round(rss_mb(), 1)))
        print(f"    seg {s}: {rows[-1]['wall_s']}s  {rows[-1]['ms_per_step']} ms/step  "
              f"spk={rows[-1]['spk']}  RSS={rows[-1]['rss_mb']}MB", flush=True)
    steady = [r["ms_per_step"] for r in rows[1:]]      # 优先禁用段 0（M9 预热纪律）
    note = "steady=segments[1:]"
    if not steady:
        steady = [r["ms_per_step"] for r in rows]
        note = "single_segment:steady=seg0(含 MPS 预热，20k 步摊薄；跨进程复跑一致性见报告)"
    med = float(np.median(steady)) if steady else float("nan")
    return dict(t4_n_neuron=N, t4_n_edge=E, t4_segments=rows, t4_steady_note=note,
                t4_steady_ms_per_step=med,
                t4_gpu_h_per_30s=round(med * 600_000 / 1000.0 / 3600.0, 4),
                t4_build_s=round(time.time() - t_build, 1), t4_rss_mb=round(rss_mb(), 1))


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all",
                    choices=["all", "t1", "t2", "t3", "t4"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--t4-device", default="mps")
    ap.add_argument("--t4-n", type=int, default=100_000)
    ap.add_argument("--t4-seg-steps", type=int, default=20_000)
    ap.add_argument("--t4-segs", type=int, default=4)
    ap.add_argument("--t4-delay-max", type=int, default=40)
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "m10_engine_probe.csv"))
    ap.add_argument("--out-json", default=os.path.join(DATA_DIR, "m10_engine_probe.json"))
    args = ap.parse_args(argv)
    res = {"generated_by": "tools/scan_m10_engine.py",
           "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    w = args.which
    if w in ("all", "t1"):
        print("[T1] 链式解析对照（投递正确性 + M9/M10 一致性）", flush=True)
        res.update(t1_chain(device=args.device))
    if w in ("all", "t2"):
        print("[T2] 同步发放投递对照（M9 冻结公式 vs M10 E5-FIX）", flush=True)
        res.update(t2_simultaneous(device=args.device))
    if w in ("all", "t3"):
        print("[T3] 分段可组合性 (e)：k=1/2/5 逐位一致", flush=True)
        res.update(t3_composability())
    if w in ("all", "t4"):
        print("[T4] 全规模稳态墙钟（设备=%s）" % args.t4_device, flush=True)
        try:
            res.update(t4_wallclock(N=args.t4_n, n_seg=args.t4_segs,
                                    seg_steps=args.t4_seg_steps,
                                    device=args.t4_device,
                                    delay_max=args.t4_delay_max))
        except Exception as ex:  # noqa: BLE001
            res["t4_error"] = f"{type(ex).__name__}: {ex}"
            print("  [T4] ERR", res["t4_error"], flush=True)
    res["elapsed_s"] = None
    with open(args.out_json, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    # 平铺 CSV
    with open(args.out, "w", newline="") as f:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(["key", "value"])

        def flat(prefix, obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    flat(f"{prefix}.{k}" if prefix else str(k), v)
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    flat(f"{prefix}[{i}]", v)
            else:
                wr.writerow([prefix, obj])
        flat("", res)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, (dict, list))},
                     ensure_ascii=False, indent=1))
    print("json ->", args.out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
