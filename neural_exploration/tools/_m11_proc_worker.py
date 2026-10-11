"""M11 引擎门的**逐目标全新子进程**工作体（(h) 跨进程逐位 + 三档 N 墙钟探针）。

纪律（清单 §0.7 #13/#14）：**逐目标/逐试次全新子进程 + 显式 venv 解释器（禁 `sys.executable`）**；
本文件只被 `tools/scan_m11_engine.py` 以 `subprocess` 调用，自身不 import 主工具。

子命令：
  `hash`   —— 跑同一配置 → 打印 `sha256(counts.tobytes())` 与脉冲总数（(h) 用）
  `probe`  —— 三档 N 稳态墙钟（分段 ≥3，取**稳态**中位数，禁用段 0）

用法（解释器必须显式给 venv）：
    PYTHONHASHSEED=0 OMP_NUM_THREADS=1 ./.venv-m9/bin/python -m neural_exploration.tools._m11_proc_worker hash --arm on --n 400
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from neural_exploration.src.mouse_cortex_circuit import (attach_to_engine,  # noqa: E402
                                                         build_laminar_probe_column)
from neural_exploration.src.mouse_cortex_engine import (MouseCortexEngine,  # noqa: E402
                                                        REF_FIX, REF_FROZEN)


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def build(arm: str = "on", n: int = 400, indegree: int = 20, seed: int = 0, device: str = "cpu",
          delay_bins=(40, 60, 80), density_hz: float = 2.0, bias_scale: float = 1.0,
          gmax_scale: float = 1.0):
    """装配确定性探针网络（**同一配置跨进程可复现**：全部由 seed 决定）。"""
    eng = MouseCortexEngine(n, device=device, use_compile=False, verbose=False,
                            use_abs_slots=True)
    if arm in ("on", "on_frozen"):
        eng.configure_subtypes(True)
    if arm in ("on", "on_frozen"):
        eng.configure_slow(True, tau_slow_ms=1000.0, inc_per_spike=6.0)
    if arm == "on_frozen":
        raise ValueError("REF_FROZEN 只在 E1b 关闭（n_ch=2）下有意义")
    if arm == "off_frozen":
        eng.configure_refractory(REF_FROZEN)
    elif arm == "off_fix":
        eng.configure_refractory(REF_FIX)
    net = build_laminar_probe_column(n, indegree=indegree, seed=seed, delay_bins=delay_bins,
                                     bias_scale=bias_scale, gmax_scale=gmax_scale)
    if arm in ("off_frozen", "off_fix"):
        # 关闭态：把四类标签归并为 E/I 两槽（PV/SST/VIP → I），与 M10 语义可比
        cls = np.where(net["syn_class"] > 0, 1, 0).astype(np.int8)
        net = dict(net, syn_class=cls, inhibitory=(cls > 0))
    eng.configure_layers(net["layer_of"], delay_bins=net["meta"]["delay_bins"])
    attach_to_engine(eng, net)
    T = int(20000)
    n_ev = int(density_hz * T * 0.05e-3 * n)
    rng = np.random.default_rng(11)
    st = rng.integers(0, T, n_ev)
    ne = rng.integers(0, n, n_ev)
    gg = rng.uniform(17.0, 23.0, n_ev).astype(np.float32)
    eng.set_external_events(st, ne, gg, n_steps=T)
    eng.reset(seed=1)
    eng.reset_counts()
    return eng, T, net


def cmd_hash(args) -> int:
    eng, T, net = build(arm=args.arm, n=args.n, indegree=args.indegree, seed=args.seed,
                        device=args.device, density_hz=args.density,
                        bias_scale=args.bias_scale, gmax_scale=args.gmax_scale)
    done = 0
    w = max(1, int(args.steps))
    while done < T:
        ww = min(w, T - done)
        eng.run(ww, delivery=args.delivery, record="counts")
        done += ww
    counts = eng.t_count.detach().cpu().numpy().astype(np.int64)
    out = dict(arm=args.arm, n=args.n, steps=T, delivery=args.delivery,
               spk_total=int(counts.sum()),
               counts_sha256=hashlib.sha256(counts.tobytes()).hexdigest(),
               n_nonzero=int((counts > 0).sum()), pid=os.getpid(),
               n_ch=int(eng.n_ch), core=eng.core_readback()["core_key"],
               slot_counts=eng.class_slot_counts(), edge_counts=eng.class_edge_counts())
    print(json.dumps(out, ensure_ascii=False))
    return 0


def cmd_probe(args) -> int:
    t_build0 = time.perf_counter()
    eng, T, net = build(arm=args.arm, n=args.n, indegree=args.indegree, seed=args.seed,
                        device=args.device, density_hz=args.density,
                        bias_scale=args.bias_scale, gmax_scale=args.gmax_scale)
    build_s = time.perf_counter() - t_build0
    rows = []
    total_steps = 0
    for s in range(int(args.segments)):
        t0 = time.perf_counter()
        r = eng.run(int(args.seg_steps), delivery="chunk", record="counts")
        wall = time.perf_counter() - t0
        total_steps += int(args.seg_steps)
        rows.append(dict(seg=s, wall_s=round(wall, 3),
                         ms_per_step=round(wall / int(args.seg_steps) * 1e3, 4),
                         spk=int(r["spk_total"]), rss_mb=round(rss_mb(), 1)))
    steady = [x["ms_per_step"] for x in rows[1:]] or [x["ms_per_step"] for x in rows]
    med = float(np.median(steady))
    out = dict(arm=args.arm, n=args.n, n_edge=int(net["meta"]["n_edge"]),
               device=args.device, segments=rows, total_steps=total_steps,
               steady_ms_per_step=round(med, 4),
               steady_note=("segments[1:]" if len(rows) > 1 else "seg0_only"),
               gpu_h_per_30s=round(med * 600_000 / 1e3 / 3600.0, 4),
               gpu_h_per_120s=round(med * 2_400_000 / 1e3 / 3600.0, 4),
               build_s=round(build_s, 2), rss_mb=round(rss_mb(), 1), pid=os.getpid(),
               n_ch=int(eng.n_ch), n_slot=int(eng.n_slot),
               n_delay_group=int(eng.n_delay_group),
               ring_shared=True, spk_total=int(sum(x["spk"] for x in rows)))
    print(json.dumps(out, ensure_ascii=False))
    return 0


def cmd_leak(args) -> int:
    """无泄漏自检：连续多段的 RSS 与对象数（清单 §2.2 缓存纪律：禁以动态规模为缓存键）。"""
    eng, T, net = build(arm=args.arm, n=args.n, indegree=args.indegree, seed=args.seed,
                        device=args.device, density_hz=args.density)
    rows = []
    for s in range(int(args.segments)):
        eng.run(int(args.seg_steps), delivery="chunk", record="counts")
        rows.append(dict(seg=s, rss_mb=round(rss_mb(), 2),
                         ar_buf_numel=int(getattr(eng._ar_buf, "numel", lambda: 0)())))
    print(json.dumps(dict(arm=args.arm, n=args.n, rows=rows,
                          rss_growth_mb=round(rows[-1]["rss_mb"] - rows[0]["rss_mb"], 2)),
                     ensure_ascii=False))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["hash", "probe", "leak"])
    ap.add_argument("--arm", default="on", choices=["on", "off_fix", "off_frozen"])
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--indegree", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--density", type=float, default=2.0)
    ap.add_argument("--bias-scale", type=float, default=1.0)
    ap.add_argument("--gmax-scale", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--delivery", default="step")
    ap.add_argument("--segments", type=int, default=3)
    ap.add_argument("--seg-steps", type=int, default=2000)
    args = ap.parse_args(argv)
    return {"hash": cmd_hash, "probe": cmd_probe, "leak": cmd_leak}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
