"""**P5 生产者每臂固定开销的分段诊断**（主agent 紧急任务；**不改生产者**，只 import 其函数）。

背景：主agent 按"驱动器启动 → 产物落盘"端到端时间戳量到**每臂 ≈82–88 min 非仿真开销**
（自报 `wall_s` 只覆盖仿真循环 ⇒ 定义域窄，不是错报）。本工具把该开销**逐段实测**：

| 段 | 内容 |
|---|---|
| **A** 进程/解释器启动 | = 端到端总墙钟 − 脚本内各段之和（父进程计） |
| **B** `import torch` / `import numpy` | 解释器内计时 |
| **C1** 夹具建网（`build_laminar_probe_column`）| N=31k 时含 **O(N·indegree) 的 Python 级 RNG 调用** |
| **C2** `add_synapses` + `finalize`（CSR/环槽/张量分配）| |
| **C3** 层标签装配 + 异质性 + `reset` | |
| **D** 事件表构造（`make_events`）| λ×N×T ⇒ N=31k/T=10 s 时 **≈2.6M 事件** |
| **E** `set_external_events`（排序/前缀和/设备张量）| |
| **F** settle 仿真 | 仍在 `wall_s` 域内（对照用） |
| **G** 测量仿真 | 同上 |
| **H** 产物落盘（`np.save` + 逐行 CSV）| N=31k ⇒ 31k 行 Python 级 `writerow` |
| **I** 收尾/进程退出 | = A 的镜像（父进程计） |

**纪律**：一律 **`--device cpu`**（MPS 由主agent 独占）。输出：段名/秒/占总墙钟% + 每段 RSS。
用法：
    ./.venv-m9/bin/python -m neural_exploration.tools.profile_m11_p5_setup --n 8192 --T-ms 1000
    ./.venv-m9/bin/python -m neural_exploration.tools.profile_m11_p5_setup --n 31000 --T-ms 10000 --segments-only
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

T0 = time.perf_counter()


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


class Seg:
    def __init__(self):
        self.rows = []

    def __call__(self, name, t_start):
        dt = time.perf_counter() - t_start
        self.rows.append(dict(seg=name, seconds=round(dt, 3), rss_mb=round(rss_mb(), 1)))
        print("    %-34s %8.3f s   RSS=%.0f MB" % (name, dt, rss_mb()), flush=True)
        return time.perf_counter()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--T-ms", type=float, default=1000.0)
    ap.add_argument("--settle-ms", type=float, default=500.0)
    ap.add_argument("--dose", type=float, default=1.0)
    ap.add_argument("--drive-rate-hz", type=float, default=8.0)
    ap.add_argument("--indegree", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--segments-only", action="store_true", help="跳过仿真段（只量固定开销）")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args(argv)
    seg = Seg()
    t = time.perf_counter()
    print("[profile] N=%d T=%sms dose=%s device=%s ↓ 分段计时（脚本内起点=import 前）" % (
        args.n, args.T_ms, args.dose, args.device), flush=True)

    import importlib
    t = time.perf_counter()
    np_ = importlib.import_module("numpy")
    t = seg("B1 import numpy", t)
    import torch
    t = seg("B2 import torch", t)
    from neural_exploration.src.mouse_cortex_circuit import (attach_to_engine,
                                                             build_laminar_probe_column)
    from neural_exploration.src.mouse_cortex_engine import MouseCortexEngine
    from neural_exploration.tools.run_m11_p5_arms import DT_MS, make_events
    t = seg("B3 import 项目模块（含引擎）", t)

    n, T_ms, settle_ms = args.n, args.T_ms, args.settle_ms
    n_meas = int(round(T_ms / DT_MS))
    n_settle = int(round(settle_ms / DT_MS))
    T_table = n_settle + n_meas
    bins = (40, 60, 80)
    spec = dict(indegree=args.indegree, gmax_scale=1.0, bias_scale=1.0,
                bias_log_mean=230.0, bias_log_sigma=0.30, net_seed=args.seed,
                delay_bins=bins, subtypes=True, slow=True)

    eng = MouseCortexEngine(n, device=args.device, use_compile=False, verbose=False)
    eng.configure_subtypes(True)
    eng.configure_slow(True, tau_slow_ms=1000.0, inc_per_spike=6.0)
    t = seg("C0 引擎对象 + 机制配置", t)
    net = build_laminar_probe_column(n, indegree=args.indegree, seed=args.seed, delay_bins=bins,
                                     bias_scale=1.0, gmax_scale=1.0)
    t = seg("C1 夹具建网（indegree=%d，n_edge=%d）" % (args.indegree, net["meta"]["n_edge"]), t)
    eng.configure_layers(net["layer_of"], delay_bins=bins)
    attach_to_engine(eng, net)
    t = seg("C2 add_synapses + finalize（CSR/环槽/张量）", t)

    st, ne, gg = make_events(n, T_table, args.drive_rate_hz * args.dose, 11)
    t = seg("D make_events（n_events=%d）" % int(st.size), t)
    eng.set_external_events(st, ne, gg, n_steps=T_table)
    t = seg("E set_external_events（排序/前缀和/设备张量）", t)
    eng.reset(seed=args.seed)
    t = seg("F reset", t)

    if not args.segments_only:
        t = seg("G settle 仿真（%d 步）" % n_settle, t)
        t = time.perf_counter()
        eng.run(n_settle, delivery="step", record="counts")
        t = seg("G settle 仿真（%d 步）" % n_settle, t)
        eng.reset_counts()
        t = time.perf_counter()
        res = eng.run(n_meas, delivery="step", record="counts", pop_trace=True, ei_integrals=True)
        t = seg("H 测量仿真（%d 步）" % n_meas, t)
        pop = np.asarray(res["pop"], dtype=np.float32)
        counts = eng.t_count.detach().cpu().numpy().astype(np.int64)
        out_dir = "/tmp/p5_profile"
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "prof_poptrace.npy"), pop)
        with open(os.path.join(out_dir, "prof_counts.csv"), "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["neuron_id", "count", "cell_type"])
            names = {0: "E", 1: "PV", 2: "SST", 3: "VIP"}
            for i in range(n):
                w.writerow([i, int(counts[i]), names.get(int(net["type_of"][i]), "E")])
        t = seg("I 产物落盘（npy + %d 行 CSV）" % n, t)

    in_script = time.perf_counter() - T0
    total_before_report = time.perf_counter()
    print("[profile] 脚本内累计（含 import 前起点） = %.3f s" % in_script, flush=True)
    out = dict(n=n, T_ms=T_ms, settle_ms=settle_ms, dose=args.dose,
               drive_rate_hz=args.drive_rate_hz, device=args.device,
               n_steps_meas=n_meas, n_steps_settle=n_settle, n_events=int(st.size),
               n_edge=int(net["meta"]["n_edge"]), segments=seg.rows,
               in_script_s=round(in_script, 3), n_edge_per_neuron=args.indegree,
               python=sys.version.split()[0], torch=torch.__version__,
               numpy=np_.__version__, executable=sys.executable)
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
    print("[profile] 固定开销合计（脚本内，不含 A/I）= %.1f s ⇒ 供与端到端时间戳差对比" % in_script)
    print("[profile] 提示：**进程启动/退出（A/I）** 需在父进程用 `time` 或 subprocess 计：")
    print("    /usr/bin/time -p ./.venv-m9/bin/python -m neural_exploration.tools.profile_m11_p5_setup "
          "--n %d --T-ms %d --segments-only" % (n, int(T_ms)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
