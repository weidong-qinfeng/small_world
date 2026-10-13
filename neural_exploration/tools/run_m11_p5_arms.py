"""M11 **P5/G1 跑臂生产者**（引擎侧；把 B3 的评估器缺的"生产者"补上）。

> **为什么存在**：`tools/validate_p11_resting.py`（M11-B3）是**纯离线评估器**（读 protocol JSON + 每臂
> 的 `pop_trace.npy`/`counts.csv`/`ei`/`agg`，**不产生任何仿真**）⇒ G1 缺"能跑出臂的入口"。
> 本工具就是那个入口（主agent RF/关键路径指派）。

**接口（逐字段对齐评估器，不得自创）**
- 臂字段（`Arm` 读取）：`arm_id / config_id / T_ms / settle_ms / role / mechanism / drive_mult / seed /
  replica_of / n_neuron / pop_trace / counts / agg{spk_total} / ei{i_exc_integral,i_inh_integral} /
  wall_s / ms_per_step / device`；另按 R34 §12.4 追加 `host / python / numpy / torch /
  ms_per_step_steady / event_seed / spikes_total / sha256_of_trace`。
- protocol JSON（`Evaluator` 读取）：`dt_ms / settle_ms / bin_ms / smooth_frames / estimator / n_neuron /
  arms[]`（本工具另写 `protocol_id / fixture_kind / engine{...} / plan_id`）。
- **硬要求**：E/I 电流积分必须**由引擎注入实测**（`MouseCortexEngine.run(ei_integrals=True)`）；
  每臂**同时**给 `pop_trace` 与 `counts`（缺一 ⇒ 评估器硬门③不可校验）。

**产物（B3 §10.3 规范）**：`<run_id>.json`（**完成即写**）/ `.done`（mtime=完成时刻）/ `.fail`（含 stderr 尾部；
不计入分母、四态 `invalid`）/ `<run_id>_poptrace.npy`（float32 逐步群体脉冲数）/ `<run_id>_counts.csv`
（`neuron_id,count,cell_type`）；JSON 内含 `sha256_of_trace + ms_per_step_steady + wall_s + pid/host/device`。
**run_id 规范**：`{stage}_{N}_{T}ms_d{dose}_s{seed}`（如 `p0_8192_T10000ms_d4.0_s0`）。

**纪律**：MPS **串行**（plan 模式逐臂 spawn，绝不并发）；`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONHASHSEED=0`；**逐臂新进程**（`subprocess` spawn，显式 `.venv-m9` 解释器）；每进程结束即写 `.done`。

用法：
  # ① 生成 plan（P0 剂量探针 / P2 时长阶梯）
  ./.venv-m9/bin/python -m neural_exploration.tools.run_m11_p5_arms --make-plan p0 --n 31000 \\
      --T-ms 10000 --doses 0,1,4 --seed 0 --out-json data/m11_p5_plan_p0.json
  # ② 跑 plan（逐臂新进程；串行）
  ./.venv-m9/bin/python -m neural_exploration.tools.run_m11_p5_arms --plan data/m11_p5_plan_p0.json \\
      --out-dir neural_exploration/data/m11_p5_runs
  # ③ 单臂模式（主agent 逐档放行）
  ./.venv-m9/bin/python -m neural_exploration.tools.run_m11_p5_arms --arm <arm_spec.json> \\
      --out-dir neural_exploration/data/m11_p5_runs
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import resource
import socket
import subprocess
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "neural_exploration", "data")
DEFAULT_OUT_DIR = os.path.join(DATA_DIR, "m11_p5_runs")
PY_ENGINE = os.path.join(PROJECT_ROOT, ".venv-m9", "bin", "python")   # 引擎侧（R34 §12.2，唯一有 torch）
DT_MS = 0.05
# **R93 第 43 条（复刻轴）**：`seed` 只驱动初态 ⇒ 必须显式声明三源。
# 派生策略（未显式给 net_seed/event_seed 时）：
#   net_seed   = seed            （网络拓扑/权重/偏置）
#   event_seed = EVENT_SEED_BASE + seed   （外部事件表；BASE=11 ⇒ **seed=0 臂与既有产物逐位一致**）
#   reset_seed = seed            （初态）
EVENT_SEED_BASE = 11

# ---- P5 基准网络（**我的引擎配置**；B3 的 L0–L3 杠杆通过参数暴露，不改代码）----
NET_DEFAULTS = dict(indegree=20, gmax_scale=1.0, bias_scale=1.0, delay_bins=(40, 60, 80),
                    bias_log_mean=230.0, bias_log_sigma=0.30, indeg_fixture=20)


ENGINE_HINT = ("./.venv-m9/bin/python（R34 §12.2：**唯一**具备 torch 2.8.0+MPS 的解释器；"
                ".venv-m11 只有 h5py，无 torch）")


def interpreter_guard(role: str = "engine") -> dict:
    """**解释器口径守卫（R34）**：引擎侧必须能 import torch；否则**显式报错**（不静默、不降级）。

    返回并留痕解释器三元组（python/numpy/torch），供产物与 protocol JSON 记录：
      - 引擎侧（本工具） ⇒ 必须 `./.venv-m9/bin/python`；
      - 评估器侧（`validate_p11_resting.py`） ⇒ `./.venv-m11/bin/python`（只需 numpy，**不得**跑仿真）。
    """
    import sys as _sys
    tri = dict(executable=_sys.executable, python=_sys.version.split()[0], role=role)
    try:
        import numpy as _np
        tri["numpy"] = _np.__version__
    except Exception:  # noqa: BLE001
        tri["numpy"] = "MISSING"
    try:
        import torch as _t
        tri["torch"] = _t.__version__
    except Exception as ex:  # noqa: BLE001
        tri["torch"] = "MISSING"
        if role == "engine":
            raise SystemExit(
                "[解释器口径错误 R34] 引擎侧必须用 %s 运行本工具（当前 sys.executable=%s，"
                "import torch 失败：%s）⇒ 严禁用 .venv-m11 跑任何 import torch 的脚本。"
                % (ENGINE_HINT, _sys.executable, ex))
    return tri


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def _ver(mod):
    try:
        return getattr(mod, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        return "unknown"


def build_engine(n: int, spec: dict, device: str = "mps"):
    """按 spec 装配 M11 内核（四亚型槽 + 层流 + 慢变量可关）。"""
    import torch
    from neural_exploration.src.mouse_cortex_circuit import (attach_to_engine,
                                                             build_laminar_probe_column)
    from neural_exploration.src.mouse_cortex_engine import REF_FROZEN, MouseCortexEngine
    net_kw = {k: v for k, v in spec.items()
              if k in ("indegree", "gmax_scale", "bias_scale", "bias_log_mean", "bias_log_sigma")}
    bins = tuple(spec.get("delay_bins") or NET_DEFAULTS["delay_bins"])
    eng = MouseCortexEngine(n, device=device, use_compile=False, verbose=False)
    eng.configure_subtypes(bool(spec.get("subtypes", True)))
    if spec.get("slow", True):
        eng.configure_slow(True, tau_slow_ms=float(spec.get("tau_slow_ms", 1000.0)),
                           inc_per_spike=float(spec.get("slow_inc", 6.0)))
    if spec.get("refractory_form") == "m10_frozen":
        # 冻结档只在 n_ch=2（E1b 关）下有意义；P5 正式臂一律 REF_FIX（默认）
        raise SystemExit("P5 正式臂禁用 REF_FROZEN（缺陷档）")
    net = build_laminar_probe_column(n, seed=int(spec.get("net_seed", 0)), delay_bins=bins, **net_kw)
    if not spec.get("subtypes", True):
        cls = np.where(net["syn_class"] > 0, 1, 0).astype(np.int8)
        net = dict(net, syn_class=cls, inhibitory=(cls > 0))
    eng.configure_layers(net["layer_of"], delay_bins=bins)
    attach_to_engine(eng, net)
    return eng, net, torch


def make_events(n: int, T_table_steps: int, rate_hz: float, seed: int):
    """外部虚拟突触事件（Poisson 近似：整数计数）；λ = rate_hz 事件/神经元/秒。"""
    n_ev = int(rate_hz * T_table_steps * DT_MS * 1e-3 * n)
    rng = np.random.default_rng(int(seed))
    return (rng.integers(0, T_table_steps, n_ev), rng.integers(0, n, n_ev),
            rng.uniform(17.0, 23.0, n_ev).astype(np.float32))



def effective_seeds(spec: dict) -> dict:
    """**复刻轴三源解析**（R93 第 43 条）：显式优先，否则由 `seed` 派生；并给三元组供断言。

    ⚠️ 历史缺陷（B3 实测 + 本次修复）：初版只有 `reset(seed=spec["seed"])` 吃调用方的 `seed`，
    而 `net_seed`（默认 0）与 `event_seed`（默认 11）**全程常量** ⇒ 同一 (config,T,dose) 下
    三个 seed 的 `pop_trace` 逐位相同（15/15 组）⇒ **复刻轴失效**（(c) 的 CI 由计数噪声撑出）。
    """
    seed = int(spec.get("seed", 0))
    ns = spec.get("net_seed", None)
    es = spec.get("event_seed", None)
    net_seed = seed if ns is None else int(ns)
    event_seed = (EVENT_SEED_BASE + seed) if es is None else int(es)
    return dict(reset_seed=seed, net_seed=net_seed, event_seed=event_seed,
                derived_net_seed=(ns is None), derived_event_seed=(es is None),
                policy=("derive(net_seed=seed, event_seed=%d+seed)" % EVENT_SEED_BASE
                        if (ns is None and es is None) else "explicit/partial"),
                triple="net%d|ev%d|reset%d" % (net_seed, event_seed, seed))

def _net_echo(spec: dict, net: dict, seeds: dict) -> dict:
    """产物里的网络档回显。**`bias_log_sigma` 只在 spec 显式带上时才写入** ⇒ 历史上不带该键的臂
    产物 JSON **字节不变**（J5 正在跑的 L0 臂属此类；不得让同一作业内出现版本偏斜）。"""
    d = dict(indegree=int(spec.get("indegree", NET_DEFAULTS["indegree"])),
             delay_bins=list(spec.get("delay_bins") or NET_DEFAULTS["delay_bins"]),
             gmax_scale=float(spec.get("gmax_scale", 1.0)),
             bias_scale=float(spec.get("bias_scale", 1.0)),
             net_seed=int(seeds["net_seed"]), n_edge=int(net["meta"]["n_edge"]))
    if "bias_log_sigma" in spec:
        d["bias_log_sigma"] = float(spec["bias_log_sigma"])
    return d


def run_one(spec: dict, out_dir: str) -> dict:
    """**单臂**：settle（丢弃）→ reset_counts → 测量窗（pop_trace + counts + E/I 积分）。"""
    n = int(spec["n_neuron"])
    T_ms = float(spec["T_ms"])
    settle_ms = float(spec["settle_ms"])
    n_meas = int(round(T_ms / DT_MS))
    n_settle = int(round(settle_ms / DT_MS))
    T_table = n_settle + n_meas
    dose = float(spec.get("drive_mult", 1.0))
    rate0 = float(spec.get("drive_rate_hz", 8.0))
    _seeds = effective_seeds(spec)
    event_seed = _seeds["event_seed"]
    device = str(spec.get("device", "mps"))

    spec = dict(spec, net_seed=_seeds["net_seed"])      # ← 派生后的网络种子显式传下去
    eng, net, torch = build_engine(n, spec, device=device)
    st, ne, gg = make_events(n, T_table, rate0 * dose, event_seed)
    eng.set_external_events(st, ne, gg, n_steps=T_table)
    eng.reset(seed=_seeds["reset_seed"])
    t0 = time.perf_counter()
    if n_settle > 0:
        eng.run(n_settle, delivery="step", record="counts")          # settle：丢弃
    eng.reset_counts()
    delivery = str(spec.get("delivery", "step"))
    res = eng.run(n_meas, delivery=delivery, record="counts", pop_trace=True, ei_integrals=True)
    wall_s = time.perf_counter() - t0
    pop = np.asarray(res["pop"], dtype=np.float32)
    counts = eng.t_count.detach().cpu().numpy().astype(np.int64)
    spk_total = int(res["spk_total"])
    assert pop.size == n_meas, "群体轨迹长度必须 == round(T_ms/dt_ms)"
    assert counts.size == n, "逐神经元计数长度必须 == n_neuron"
    assert int(counts.sum()) == spk_total, "双估计器（counts vs 轨迹）不一致"
    assert abs(float(pop.sum()) - spk_total) <= 0.02 * max(1.0, spk_total), \
        "双估计器（pop_trace vs counts）不一致"
    assert res["ei_integrals"]["i_exc_integral"] > 0, "E/I 积分必须由引擎实测且非零"

    run_id = str(spec["run_id"])
    trace_path = os.path.join(out_dir, f"{run_id}_poptrace.npy")
    counts_path = os.path.join(out_dir, f"{run_id}_counts.csv")
    np.save(trace_path, pop)
    type_names = {0: "E", 1: "PV", 2: "SST", 3: "VIP"}
    with open(counts_path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["neuron_id", "count", "cell_type"])
        for i in range(n):
            w.writerow([i, int(counts[i]), type_names.get(int(net["type_of"][i]), "E")])
    sha = hashlib.sha256(pop.tobytes()).hexdigest()
    arm = dict(
        arm_id=run_id, config_id=str(spec.get("config_id", "p5")),
        T_ms=T_ms, settle_ms=settle_ms, role=str(spec.get("role", "dose")),
        mechanism=str(spec.get("mechanism", "none")), drive_mult=dose,
        seed=int(spec.get("seed", 0)), replica_of=spec.get("replica_of"),
        n_neuron=n,
        pop_trace=os.path.relpath(trace_path, PROJECT_ROOT),
        counts=os.path.relpath(counts_path, PROJECT_ROOT),
        agg=dict(spk_total=spk_total),
        ei=dict(i_exc_integral=float(res["ei_integrals"]["i_exc_integral"]),
                i_inh_integral=float(res["ei_integrals"]["i_inh_integral"]),
                ei_ratio=res["ei_integrals"]["ei_ratio"],
                by_class=res["ei_integrals"]["by_class"],
                sign_convention=res["ei_integrals"]["sign_convention"],
                i_inh_integral_signed_raw=res["ei_integrals"]["i_inh_integral_signed_raw"]),
        wall_s=wall_s, ms_per_step=float(res["ms_per_step"]),
        device=str(eng.device),
        # ---- R34 §12.4 追加字段 ----
        host=socket.gethostname(), python=sys.version.split()[0],
        numpy=_ver(np), torch=_ver(torch),
        ms_per_step_steady=float(res["ms_per_step"]),
        event_seed=event_seed, spikes_total=spk_total,
        seeds=_seeds,                 # **复刻轴三元组（R93）**：net/event/reset 三源分别留痕
        sha256_of_trace=sha, n_steps_meas=n_meas, n_steps_settle=n_settle,
        delivery=delivery,
        engine_version=str(eng.version), n_ch=int(eng.n_ch),
        interpreter=spec.get("interpreter"),
        drive_rate_hz=rate0, drive_rate_hz_eff=rate0 * dose,
        n_events=int(st.size), rss_mb=round(rss_mb(), 1),
        pid=os.getpid(), finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        net=_net_echo(spec, net, _seeds),
    )
    jpath = os.path.join(out_dir, f"{run_id}.json")
    with open(jpath, "w") as f:
        json.dump(arm, f, indent=1, ensure_ascii=False)
    return arm


def arm_spec_from_cli(args, run_id: str) -> dict:
    """CLI ⇒ 单臂 spec。

    ⚠️ **杠杆键只在显式给出时才进 spec**（`bias_log_sigma`）：缺键 ⇒ `net_kw` 过滤器不提该参数
    ⇒ 夹具用其签名默认值（`mouse_cortex_circuit.build_laminar_probe_column(bias_log_sigma=0.30)`）
    ⇒ **既有臂（含 seed=0 逐位锚点）的行为与产物字节不变**（已由
    `tools/verify_m11_producer_inert.py` 与 `--self-test-sigma` 的 C3 断言实测）。
    """
    sp = dict(run_id=run_id, stage=args.stage, n_neuron=args.n, T_ms=args.T_ms,
              delivery=args.delivery,
              settle_ms=args.settle_ms, drive_mult=args.dose, drive_rate_hz=args.drive_rate_hz,
              seed=args.seed, event_seed=args.event_seed, config_id=args.config_id,
              role=args.role, mechanism=args.mechanism, device=args.device,
              indegree=args.indegree, gmax_scale=args.gmax_scale, bias_scale=args.bias_scale,
              net_seed=args.net_seed, delay_bins=tuple(int(x) for x in args.delay_bins.split(",")),
              subtypes=not args.no_subtypes, slow=not args.no_slow)
    if getattr(args, "bias_log_sigma", None) is not None:      # L1 杠杆：仅显式时出现
        sp["bias_log_sigma"] = float(args.bias_log_sigma)
    return sp


def run_arm_main(args) -> int:
    interpreter_guard("engine")          # R34：引擎侧必须能 import torch
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    if args.arm:
        with open(args.arm) as f:
            spec = json.load(f)
    else:
        rid = args.run_id or "%s_%s_T%dms_d%s_s%s" % (args.stage, args.n, int(args.T_ms),
                                                      args.dose, args.seed)
        spec = arm_spec_from_cli(args, rid)
    spec["interpreter"] = interpreter_guard("engine")
    run_id = str(spec["run_id"])
    done = os.path.join(out_dir, f"{run_id}.done")
    fail = os.path.join(out_dir, f"{run_id}.fail")
    for p in (done, fail):
        if os.path.exists(p):
            os.remove(p)
    try:
        arm = run_one(spec, out_dir)
    except Exception as ex:  # noqa: BLE001
        import traceback
        with open(fail, "w") as f:
            f.write("run_id=%s\n%s\n%s\n" % (run_id, ex, traceback.format_exc()[-2000:]))
        print("[FAIL] %s: %s" % (run_id, ex), flush=True)
        return 1
    with open(done, "w") as f:
        f.write("%s %s\n" % (run_id, arm["finished_at"]))
    print("[DONE] %s  spk=%d  ei_ratio=%.4f  ms/step=%.4f  wall=%.2fs  RSS=%.0fMB" % (
        run_id, arm["agg"]["spk_total"], arm["ei"]["ei_ratio"], arm["ms_per_step_steady"],
        arm["wall_s"], arm["rss_mb"]), flush=True)
    return 0


def assert_plan_replication(plan: dict) -> dict:
    """**复刻轴有效性预检**（R93）：不同 `seed` 必须给出不同的 (net_seed,event_seed,reset_seed)。

    规则：按 `(config_id, T_ms, drive_mult, mechanism)` 分组；
      ① 组内 `seed` 不同 ⇒ **三元组必须互不相同**（否则"复刻"退化为"重复"）；
      ② `role="replica"` 臂 ⇒ 三元组必须与 `replica_of` 指向的臂**相同**（(h) 要的是同参数重复）。
    失败 ⇒ `SystemExit(3)` 并打印三元组表。
    """
    by = {}
    for a in plan["arms"]:
        sp = dict(a)
        for k in ("settle_ms", "drive_rate_hz", "device", "net_seed", "event_seed"):
            if k not in sp and k in plan:
                sp[k] = plan[k]
        if sp.get("net_seed", None) is None:
            sp.pop("net_seed", None)
        if sp.get("event_seed", None) is None:
            sp.pop("event_seed", None)
        sp["seed"] = int(a.get("seed", 0))
        e = effective_seeds(sp)
        by.setdefault((a["config_id"], float(a["T_ms"]), float(a["drive_mult"]),
                       a.get("mechanism", "none")), []).append((a, e))
    fails, rows = [], []
    for key, lst in sorted(by.items()):
        # ⚠️ **关键：判据必须"非构造性"**。若只比 (net,event,reset) **三元组**，因 `reset_seed ≡ seed`
        # 恒不同 ⇒ 该判据**永远通过**（构造性恒真，M10 D2/L23 同族失效）。本节点实测确认：
        # 计划级恒定 net_seed=0/event_seed=11 的"坏计划"用三元组判**照样通过**。
        # ⇒ 主判据改为 **(net_seed, event_seed) 对**在组内必须随 seed 变化（且与 seed 数一一对应）。
        pairs, tris = {}, {}
        seed_set = set()
        for a, e in lst:
            rows.append((key[0], key[1], key[2], a["seed"], e["triple"], a["role"]))
            if a["role"] == "replica":
                continue
            seed_set.add(int(a["seed"]))
            pairs.setdefault((e["net_seed"], e["event_seed"]), []).append(a["seed"])
            tris.setdefault(e["triple"], []).append(a["seed"])
        if len(seed_set) > 1:
            n_pair = len(pairs)
            n_seed = len(seed_set)
            if n_pair < n_seed:
                fails.append("**复刻轴失效** %s T=%s dose=%s: %d 个 seed 只对应 %d 组 (net_seed,event_seed)"
                             " %r ⇒ '复刻'退化为'重复'（(c) 的 CI 由计数噪声撑出）"
                             % (key[0], key[1], key[2], n_seed, n_pair,
                                {k: v for k, v in pairs.items()}))
            # 附带的构造性检查：reset_seed 必须 == 该臂声明的 seed（防止 reset 也被常量固定）
            for a, e in lst:
                if a["role"] != "replica" and e["reset_seed"] != int(a["seed"]):
                    fails.append("%s 的 reset_seed(%s) != seed(%s)" % (a["run_id"],
                                                                      e["reset_seed"], a["seed"]))
    for r in rows:
        print("  [复刻轴] %-8s T=%-8.0f dose=%-4s seed=%d  %-22s role=%s" % r)
    if fails:
        print("**复刻轴预检失败（不同 seed 实为同一次实现）**：")
        for x in fails:
            print("   -", x)
        raise SystemExit(3)
    print("复刻轴预检通过：%d 组 (config,T,dose) 内不同 seed 的三元组互不相同" % len(by))
    return dict(n_groups=len(by), n_arms=len(plan["arms"]))


def write_protocol(plan: dict, arms: list, out_dir: str, plan_id: str) -> str:
    proto = dict(
        protocol_id="m11_p5_%s" % plan_id,
        fixture_kind="M11 引擎侧生产者产物（%s）" % "tools/run_m11_p5_arms.py",
        plan_id=plan_id, dt_ms=float(plan.get("dt_ms", DT_MS)),
        settle_ms=float(plan["settle_ms"]), bin_ms=float(plan.get("bin_ms", 100.0)),
        smooth_frames=plan.get("smooth_frames", 5),
        estimator=str(plan.get("estimator", "per_neuron")),
        n_neuron=int(plan["n_neuron"]),
        drive_rate_hz=float(plan.get("drive_rate_hz", 8.0)),
        engine_interpreters=dict(
            engine=dict(executable=PY_ENGINE,
                        python=(arms[0].get("python") if arms else None),
                        numpy=(arms[0].get("numpy") if arms else None),
                        torch=(arms[0].get("torch") if arms else None)),
            evaluator=dict(executable="./.venv-m11/bin/python", role="只评估、不仿真"),
            numpy_skew_discipline=("numpy 偏斜（.venv-m9 = 1.26.4 vs .venv-m11 = 2.0.2）⇒ "
                                   "**评估器侧数值不得用于支持或推翻任何「逐位/BIT」结论**；"
                                   "逐位类断言只在「同解释器 + 同 numpy」内可比（R34 §12.3）")),
        engine=dict(python=PY_ENGINE, engine_version=arms[0]["engine_version"] if arms else None,
                    torch=arms[0]["torch"] if arms else None,
                    numpy=arms[0]["numpy"] if arms else None, device=arms[0]["device"] if arms else None),
        arms=arms)
    path = os.path.join(DATA_DIR, "m11_p5_protocol_%s.json" % plan_id)
    with open(path, "w") as f:
        json.dump(proto, f, indent=1, ensure_ascii=False)
    print("protocol ->", path, flush=True)
    return path


def run_plan_main(args) -> int:
    with open(args.plan) as f:
        plan = json.load(f)
    plan_id = str(plan.get("plan_id", "plan"))
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    env = dict(os.environ)
    env.update(PYTHONHASHSEED="0", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONPATH=PROJECT_ROOT)
    # ---- **复刻轴预检（R93；非可选）**：同一 (config,T,dose,mechanism) 下，
    #      seed 不同的臂其三源三元组必须互不相同；replica 臂必须与其目标**三元组相同**。----
    assert_plan_replication(plan)
    arms, n_fail = [], 0
    for spec in plan["arms"]:
        rid = str(spec["run_id"])
        done = os.path.join(out_dir, f"{rid}.done")
        if os.path.exists(done) and not args.force:
            with open(os.path.join(out_dir, f"{rid}.json")) as f:
                arms.append(json.load(f))
            print("[SKIP] %s（已有 .done）" % rid, flush=True)
            continue
        spec = dict(spec)
        spec.setdefault("settle_ms", plan["settle_ms"])
        spec.setdefault("drive_rate_hz", plan.get("drive_rate_hz", 8.0))
        spec.setdefault("device", plan.get("device", "mps"))
        for k in ("indegree", "gmax_scale", "bias_scale", "bias_log_sigma", "bias_log_mean",
                  "net_seed", "delay_bins", "subtypes", "slow", "tau_slow_ms", "slow_inc"):
            if k in plan and k not in spec:
                spec[k] = plan[k]
        spec["n_neuron"] = int(spec.get("n_neuron", plan["n_neuron"]))
        spath = os.path.join(out_dir, f"{rid}.spec.json")
        with open(spath, "w") as f:
            json.dump(spec, f, indent=1)
        t0 = time.time()
        # **逐臂新进程**（绝不在同进程循环；MPS 串行 ⇒ 严格顺序 spawn）
        p = subprocess.run([PY_ENGINE, "-m", "neural_exploration.tools.run_m11_p5_arms",
                            "--arm", spath, "--out-dir", out_dir],
                           env=env, cwd=PROJECT_ROOT, capture_output=True, text=True)
        sys.stdout.write(p.stdout)
        if p.returncode != 0:
            n_fail += 1
            sys.stderr.write(p.stderr[-800:])
            print("[FAIL] %s（%.1fs）" % (rid, time.time() - t0), flush=True)
            if n_fail >= 2:
                print("**停点**：连续 ≥2 臂失败 ⇒ 停并汇报（不自行重跑）", flush=True)
                break
            continue
        with open(os.path.join(out_dir, f"{rid}.json")) as f:
            arms.append(json.load(f))
    if arms:
        write_protocol(plan, arms, out_dir, plan_id)
    print("完成：%d 臂（失败 %d）｜out_dir=%s" % (len(arms), n_fail, out_dir), flush=True)
    return 0 if not n_fail else 1


def replication_micro_test(args) -> int:
    """**复刻轴有效性微测试（R93；非可选）**：N=1000 / T=100 ms / 同 dose / seed∈{0,1,2}
    ⇒ 断言三臂 `sha256_of_trace` **互不相同**；相同 ⇒ 非零退出并打印三元组。"""
    import tempfile
    tmp = tempfile.mkdtemp(prefix="m11_repl_")
    three, shas = [], []
    for sd in (0, 1, 2):
        sp = dict(run_id="repl_1000_T100ms_d1.0_s%d" % sd, n_neuron=1000, T_ms=100.0,
                  settle_ms=0.0, drive_mult=1.0, drive_rate_hz=8.0, seed=sd, device="cpu",
                  config_id="repl_test", role="dose", mechanism="none", delivery="chunk",
                  indegree=20, gmax_scale=1.0, bias_scale=1.0, delay_bins=(40, 60, 80),
                  subtypes=True, slow=True,
                  **({"bias_log_sigma": float(args.bias_log_sigma)}
                     if getattr(args, "bias_log_sigma", None) is not None else {}))
        a = run_one(sp, tmp)
        three.append(a["seeds"])
        shas.append(a["sha256_of_trace"])
        print("  [微测试] seed=%d 三元组 %s  sha256=%s" % (sd, a["seeds"]["triple"],
                                                       a["sha256_of_trace"][:24]))
    ok = len(set(shas)) == 3 and len({t["triple"] for t in three}) == 3
    print("复刻轴微测试：三臂 sha256 互不相同 = %s（唯一数 %d/3）" % (ok, len(set(shas))))
    if not ok:
        print("**复刻轴微测试失败（复刻轴失效 ⇒ 计划不可用）**")
        return 4
    return 0


def _counts_of(arm: dict, out_dir: str) -> np.ndarray:
    """从臂产物的 `_counts.csv` 读逐神经元计数（与 `pop_trace` 的双估计器一致性已由 run_one 断言）。"""
    import csv as _csv
    p = os.path.join(out_dir, str(arm["arm_id"]) + "_counts.csv")
    out = []
    with open(p) as f:
        for row in _csv.DictReader(f):
            out.append(int(row["count"]))
    return np.asarray(out, dtype=np.int64)


def ks_statistic(a, b) -> float:
    """两样本 KS 统计量 D = sup|F_a − F_b|（**精确排序算法，不给 p 值**）。

    M10 教训：KS 的**渐近 p 值**在 λ→0 时会振荡（曾把"完全相同的分布"报成 p=0.0）⇒ 本工具只取
    与分布无关的 D，阈值由**同杠杆不同 seed** 的实测噪声底标定，不依赖渐近近似。
    """
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    if a.size == 0 or b.size == 0:
        return float("nan")
    data = np.concatenate([a, b])
    ca = np.searchsorted(a, data, side="right") / float(a.size)
    cb = np.searchsorted(b, data, side="right") / float(b.size)
    return float(np.max(np.abs(ca - cb)))


def rate_dist_metrics(counts, T_ms: float) -> dict:
    """**率的分布指标（量纲自检：rate_hz = counts / (T_ms/1000)）**。

    算例自检（脚本打印）：counts=8 @ T_ms=1000 ⇒ rate=8 Hz。指标都是"率"的统计量（单位 Hz），
    `silent_frac` 是无量纲比例 ⇒ 与单位无关。
    """
    r = np.asarray(counts, dtype=np.float64) / (float(T_ms) / 1000.0)
    p10, p50, p90 = (float(np.percentile(r, q)) for q in (10, 50, 90))
    return dict(n=int(r.size), rate_mean=float(r.mean()), rate_median=p50,
                rate_p10=p10, rate_p50=p50, rate_p90=p90,
                spread_p10_p90=float(p90 - p10),
                spread_ratio_p90_over_p10=(float(p90 / p10) if p10 > 0 else float("inf")),
                silent_frac=float((r == 0.0).mean()), rate_max=float(r.max()))


# ---- 阳性对照阈值（**跑前声明**，见 --self-test-sigma；反例自证见 C2/C3）----
SIGMA_EFFECT_MIN_D = 0.10          # KS 距离下限（分布形状明显不同）
SIGMA_EFFECT_NOISE_MULT = 3.0      # 且必须 > 3 × 同杠杆不同 seed 的噪声底
SIGMA_EFFECT_MIN_SPREAD_REL = 0.25  # 且 p90−p10 极差相对变化 ≥ 25%


def sigma_lever_micro_test(args) -> int:
    """**新杠杆启用前的阳性对照（R84 第 37 条：未证伪过的断言不构成证据）**——以 `bias_log_sigma` 为例。

    臂集（同进程、CPU、同 bias/gmax/seed，只改 σ；**N/T 由 `--n/--T-ms` 给**）：
      ① 对照 σ=σ_ctl  ② 高 σ=σ_hi  ③ 高 σ 的**重跑**（同规格 ⇒ 必须逐位相同）④ **不带键**（默认）
      ⑤ 对照 σ 换 seed（**噪声底**：同杠杆不同实现）
    断言（任一不成立 ⇒ 非零退出，绝不静默通过）：
      C1 **杠杆生效**：D(①,②) ≥ max(%.2f, %.1f×D(①,⑤)) 且 |(p90−p10)比 − 1| ≥ %.0f%%；
         否则打印 `no_effect` 并退出 **9**（⇒ 该杠杆不值得投 GPU-h）
      C2 **反例自证（检测器不假阳性）**：D(②,③)=0 且两臂 `sha256_of_trace` 相同；否则退出 **10**
      C3 **默认语义保持**：④（不带键）与 ①（显式 σ=默认 0.30）逐位相同；否则退出 **11**
    另：`--sigma-negative-only` ⇒ 只跑"两臂同 σ"并**按声明有论断处理** ⇒ 打印 `no_effect` 并退出 **12**
    （这就是"故意让两臂 σ 相同"的反例自证）。
    """ % (SIGMA_EFFECT_MIN_D, SIGMA_EFFECT_NOISE_MULT, 100 * SIGMA_EFFECT_MIN_SPREAD_REL)
    import tempfile
    tmp = tempfile.mkdtemp(prefix="m11_sigma_")
    n = int(args.n)
    T = float(args.T_ms)
    base = dict(n_neuron=n, T_ms=T, settle_ms=float(args.settle_ms), drive_mult=float(args.dose),
                drive_rate_hz=float(args.drive_rate_hz), device="cpu", config_id="sigma_probe",
                role="baseline", mechanism="none", delivery=args.delivery, indegree=args.indegree,
                gmax_scale=float(args.gmax_scale), bias_scale=float(args.bias_scale),
                delay_bins=(40, 60, 80), subtypes=True, slow=True)
    lo, hi = float(args.sigma_control), float(args.sigma_high)
    print("== σ 杠杆阳性对照（CPU / N=%d / T=%.0f ms / settle=%.0f ms / bias=%.2f / gmax=%.2f / "
          "dose=%.1f / %s / 默认=0.30）=="
          % (n, T, base["settle_ms"], base["bias_scale"], base["gmax_scale"], base["drive_mult"],
             base["delivery"]))

    def go(tag, sigma, seed, with_key=True):
        sp = dict(base, run_id="sig_%s" % tag, seed=int(seed))
        if with_key:
            sp["bias_log_sigma"] = float(sigma)
        a = run_one(sp, tmp)
        a["_rate"] = rate_dist_metrics(_counts_of(a, tmp), T)
        print("  [臂] %-22s σ=%-6s seed=%d  key=%s  spk=%-7d 中位率=%.4f Hz 静默=%.4f "
              "p10=%.3f p90=%.3f 极差=%.3f  sha=%s"
              % (tag, ("%.2f" % sigma) if with_key else "缺省", seed, with_key,
                 a["agg"]["spk_total"], a["_rate"]["rate_median"], a["_rate"]["silent_frac"],
                 a["_rate"]["rate_p10"], a["_rate"]["rate_p90"], a["_rate"]["spread_p10_p90"],
                 a["sha256_of_trace"][:16]), flush=True)
        return a

    if args.sigma_negative_only:
        a = go("hi_first", hi, 0)
        b = go("hi_second", hi, 0)
        D = ks_statistic(_counts_of(a, tmp), _counts_of(b, tmp))
        same = (a["sha256_of_trace"] == b["sha256_of_trace"])
        print("  [反例自证] 两臂 σ 相同（%.2f）⇒ D=%.4f、逐位相同=%s ⇒ 判定 **no_effect**"
              % (hi, D, same))
        print("  若把这对臂当作『σ 改变了形状』的证据，本命令**拒绝**（退出码 12）")
        return 12

    a_ctl = go("ctl", lo, 0)
    a_hi = go("hi", hi, 0)
    a_dup = go("hi_dup", hi, 0)
    a_def = go("default", lo, 0, with_key=False)
    a_ctl2 = go("ctl_otherseed", lo, 1)
    c_ctl, c_hi = _counts_of(a_ctl, tmp), _counts_of(a_hi, tmp)
    D_effect = ks_statistic(c_ctl, c_hi)
    D_noise = ks_statistic(c_ctl, _counts_of(a_ctl2, tmp))
    D_null = ks_statistic(c_hi, _counts_of(a_dup, tmp))
    sp_ratio = (a_hi["_rate"]["spread_p10_p90"] / a_ctl["_rate"]["spread_p10_p90"]
                if a_ctl["_rate"]["spread_p10_p90"] > 0 else float("inf"))
    thr = max(SIGMA_EFFECT_MIN_D, SIGMA_EFFECT_NOISE_MULT * D_noise)
    eff = (D_effect >= thr) and (abs(sp_ratio - 1.0) >= SIGMA_EFFECT_MIN_SPREAD_REL)
    rep = (D_null == 0.0) and (a_hi["sha256_of_trace"] == a_dup["sha256_of_trace"])
    dflt = (a_def["sha256_of_trace"] == a_ctl["sha256_of_trace"])
    # 量纲自检算例（R76/R80）
    ex_counts, ex_T = 8, 1000.0
    print("  [量纲自检] counts=8 @ T=1000 ms ⇒ rate=%g Hz（rate_hz = counts/(T_ms/1e3)）；"
          "silent_frac 无量纲" % (ex_counts / (ex_T / 1000.0)))
    print("  [C1 杠杆生效] D(σ%.2f,σ%.2f)=%.4f ≥ max(%.2f, %.1f×D_noise=%.4f)=%.4f ；"
          "极差比 %.4f（需偏离 1 至少 %.0f%%）⇒ %s"
          % (lo, hi, D_effect, SIGMA_EFFECT_MIN_D, SIGMA_EFFECT_NOISE_MULT, D_noise, thr,
             sp_ratio, 100 * SIGMA_EFFECT_MIN_SPREAD_REL, "**EFFECT**" if eff else "**no_effect**"))
    print("  [C2 反例自证] 同 σ 同 seed 重跑：D=%.4f、逐位相同=%s ⇒ %s"
          % (D_null, a_hi["sha256_of_trace"] == a_dup["sha256_of_trace"],
             "检测器不假阳性 ✓" if rep else "**检测器假阳性 ✗**"))
    print("  [C3 默认语义] 不带键 vs 显式 σ=%.2f：逐位相同=%s ⇒ %s"
          % (lo, dflt, "历史臂语义保持 ✓" if dflt else "**语义被改变 ✗**"))
    print("  [噪声底] 同 σ=%.2f 换 seed ⇒ D=%.4f（检测器的抽样噪声底，非 0）" % (lo, D_noise))
    rep_json = dict(tool="run_m11_p5_arms.sigma_lever_micro_test",
                    generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"), lever="bias_log_sigma",
                    conditions=dict(n_neuron=n, T_ms=T, settle_ms=base["settle_ms"],
                                    bias_scale=base["bias_scale"], gmax_scale=base["gmax_scale"],
                                    drive_mult=base["drive_mult"], delivery=base["delivery"],
                                    device="cpu", sigma_control=lo, sigma_high=hi),
                    thresholds=dict(min_D=SIGMA_EFFECT_MIN_D, noise_mult=SIGMA_EFFECT_NOISE_MULT,
                                    min_spread_rel=SIGMA_EFFECT_MIN_SPREAD_REL),
                    arms={k: dict(sha256_of_trace=v["sha256_of_trace"], spk_total=v["agg"]["spk_total"],
                                  **v["_rate"]) for k, v in
                          (("control", a_ctl), ("high", a_hi), ("high_dup", a_dup),
                           ("default_no_key", a_def), ("control_other_seed", a_ctl2))},
                    verdict=dict(C1_lever_effective=eff, C1_threshold=thr, D_effect=D_effect,
                                 D_noise_floor=D_noise, spread_p10_p90_ratio=sp_ratio,
                                 C2_detector_no_false_positive=rep, C2_D_null=D_null,
                                 C3_default_semantics_preserved=dflt,
                                 one_liner="if you claim an effect and C1 is false, this command "
                                           "exits non-zero (9) — no silent pass"))
    p = os.path.join(DATA_DIR, "m11_lever_sigma_selftest.json")
    with open(p, "w") as f:
        json.dump(rep_json, f, indent=1, ensure_ascii=False)
    print("  报告 -> %s" % p)
    if not eff:
        print("**σ 杠杆判定 no_effect（形状没动）⇒ 退出码 9，L1 全量不值当投 GPU-h**")
        return 9
    if not rep:
        print("**反例自证失败（检测器对同规格臂报出差异）⇒ 退出码 10**")
        return 10
    if not dflt:
        print("**默认语义断言失败（不带键 ≠ 显式 0.30）⇒ 退出码 11**")
        return 11
    print("σ 杠杆阳性对照通过：C1 生效 + C2 反例自证 + C3 默认语义保持（三者齐备才算证据）")
    return 0


def make_plan(args) -> int:
    n = args.n
    doses = [float(x) for x in args.doses.split(",")]
    stage = args.make_plan
    arms = []
    if stage == "p0":
        for d in doses:
            rid = "%s_%s_T%dms_d%s_s%s" % (stage, n, int(args.T_ms), d, args.seed)
            a = dict(run_id=rid, stage=stage, config_id="p5_%s" % stage, T_ms=float(args.T_ms),
                     drive_mult=d, seed=int(args.seed),
                     role=("zero_drive" if d == 0 else "dose"), mechanism="none")
            if args.bias_log_sigma is not None:            # L1 杠杆：仅显式时进臂 key 组
                a["bias_log_sigma"] = float(args.bias_log_sigma)
            arms.append(a)
    elif stage == "p2":
        for ratio in (1, 2, 4):
            T = float(args.T_ms) * ratio
            for d in doses:
                rid = "%s_%s_T%dms_d%s_s%s" % (stage, n, int(T), d, args.seed)
                a = dict(run_id=rid, stage=stage, config_id="p5_%s" % stage, T_ms=T,
                         drive_mult=d, seed=int(args.seed),
                         role=("zero_drive" if d == 0 else "dose"), mechanism="none")
                if args.bias_log_sigma is not None:
                    a["bias_log_sigma"] = float(args.bias_log_sigma)
                arms.append(a)
    else:
        raise SystemExit("未知模板：%s" % stage)
    plan = dict(plan_id="%s_n%s_T%d_s%s" % (stage, n, int(args.T_ms), args.seed),
                stage=stage, n_neuron=n, settle_ms=float(args.settle_ms),
                dt_ms=DT_MS, bin_ms=100.0, smooth_frames=5, estimator="per_neuron",
                drive_rate_hz=float(args.drive_rate_hz), device=args.device,
                indegree=args.indegree, gmax_scale=args.gmax_scale, bias_scale=args.bias_scale,
                net_seed=args.net_seed,
                delay_bins=[int(x) for x in args.delay_bins.split(",")],
                arms=arms)
    if args.bias_log_sigma is not None:                    # 计划级同键（仅显式时）
        plan["bias_log_sigma"] = float(args.bias_log_sigma)
    with open(args.out_json, "w") as f:
        json.dump(plan, f, indent=1, ensure_ascii=False)
    print("plan(%d arms) -> %s" % (len(arms), args.out_json), flush=True)
    for a in arms:
        print("   ", a["run_id"], a["role"], "dose=%s" % a["drive_mult"], flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=None, help="plan JSON（驱动器模式；逐臂 spawn 新进程）")
    ap.add_argument("--arm", default=None, help="单臂 spec JSON（单臂模式，供逐档放行）")
    ap.add_argument("--make-plan", default=None, choices=[None, "p0", "p2"],
                    help="生成 plan 模板（p0 剂量探针 / p2 时长阶梯）")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--out-json", default=os.path.join(DATA_DIR, "m11_p5_plan.json"))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--stage", default="p0")
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--T-ms", type=float, default=10000.0)
    ap.add_argument("--settle-ms", type=float, default=500.0)
    ap.add_argument("--doses", default="0,1,4")
    ap.add_argument("--dose", type=float, default=1.0)
    ap.add_argument("--drive-rate-hz", type=float, default=8.0, help="dose=1 时的事件率（Hz/神经元）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--event-seed", type=int, default=None,
                    help="不显式给 ⇒ 由 seed 派生（=11+seed）；R93 复刻轴纪律")
    ap.add_argument("--config-id", default="p5")
    ap.add_argument("--role", default="dose",
                    choices=("zero_drive", "baseline", "dose", "replica"))
    ap.add_argument("--mechanism", default="none")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--delivery", default="step", choices=("step", "chunk"),
                    help="step=判据口径（默认）；chunk 仅用于墙钟对照（须登记！）")
    ap.add_argument("--indegree", type=int, default=NET_DEFAULTS["indegree"])
    ap.add_argument("--gmax-scale", type=float, default=1.0)
    ap.add_argument("--bias-scale", type=float, default=1.0)
    ap.add_argument("--bias-log-sigma", type=float, default=None,
                    help="L1 阈值异质性杠杆（对数正态宽度，无量纲）。**不显式给 ⇒ 沿用夹具默认 0.30**"
                         "（保持既有臂（含 seed=0 逐位锚点）的行为与产物字节不变）")
    ap.add_argument("--self-test-sigma", action="store_true",
                    help="**新杠杆启用前的阳性对照**（R84 第 37 条）：对照/高/重跑/缺键/换 seed 五臂 ⇒ "
                         "C1 杠杆生效 + C2 反例自证 + C3 默认语义保持；任一不成立即非零退出")
    ap.add_argument("--sigma-control", type=float, default=0.30, help="self-test-sigma：对照组 σ")
    ap.add_argument("--sigma-high", type=float, default=0.90, help="self-test-sigma：高组 σ")
    ap.add_argument("--sigma-negative-only", action="store_true",
                    help="self-test-sigma：只跑『两臂同 σ』⇒ 必须判 no_effect 且退出 12（反例自证）")
    ap.add_argument("--net-seed", type=int, default=None,
                    help="不显式给 ⇒ 由 seed 派生（=seed）；R93 复刻轴纪律")
    ap.add_argument("--delay-bins", default="40,60,80")
    ap.add_argument("--no-subtypes", action="store_true")
    ap.add_argument("--no-slow", action="store_true")
    ap.add_argument("--force", action="store_true", help="plan 模式：忽略已有 .done 重跑")
    ap.add_argument("--self-test-replication", action="store_true",
                    help="复刻轴微测试（R93）：N=1000/T=100ms/seed∈{0,1,2} ⇒ 三 sha256 必须互不相同")
    args = ap.parse_args(argv)
    if args.self_test_sigma:
        return sigma_lever_micro_test(args)
    if args.self_test_replication:
        return replication_micro_test(args)
    if args.make_plan:
        return make_plan(args)
    if args.plan:
        return run_plan_main(args)
    return run_arm_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
