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

# ---- P5 基准网络（**我的引擎配置**；B3 的 L0–L3 杠杆通过参数暴露，不改代码）----
NET_DEFAULTS = dict(indegree=20, gmax_scale=1.0, bias_scale=1.0, delay_bins=(40, 60, 80),
                    bias_log_mean=230.0, bias_log_sigma=0.30, indeg_fixture=20)


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
    event_seed = int(spec.get("event_seed", 11))
    device = str(spec.get("device", "mps"))

    eng, net, torch = build_engine(n, spec, device=device)
    st, ne, gg = make_events(n, T_table, rate0 * dose, event_seed)
    eng.set_external_events(st, ne, gg, n_steps=T_table)
    eng.reset(seed=int(spec.get("seed", 0)))
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
        sha256_of_trace=sha, n_steps_meas=n_meas, n_steps_settle=n_settle,
        delivery=delivery,
        engine_version=str(eng.version), n_ch=int(eng.n_ch),
        drive_rate_hz=rate0, drive_rate_hz_eff=rate0 * dose,
        n_events=int(st.size), rss_mb=round(rss_mb(), 1),
        pid=os.getpid(), finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        net=dict(indegree=int(spec.get("indegree", NET_DEFAULTS["indegree"])),
                 delay_bins=list(spec.get("delay_bins") or NET_DEFAULTS["delay_bins"]),
                 gmax_scale=float(spec.get("gmax_scale", 1.0)),
                 bias_scale=float(spec.get("bias_scale", 1.0)),
                 net_seed=int(spec.get("net_seed", 0)), n_edge=int(net["meta"]["n_edge"])),
    )
    jpath = os.path.join(out_dir, f"{run_id}.json")
    with open(jpath, "w") as f:
        json.dump(arm, f, indent=1, ensure_ascii=False)
    return arm


def arm_spec_from_cli(args, run_id: str) -> dict:
    return dict(run_id=run_id, stage=args.stage, n_neuron=args.n, T_ms=args.T_ms,
                delivery=args.delivery,
                settle_ms=args.settle_ms, drive_mult=args.dose, drive_rate_hz=args.drive_rate_hz,
                seed=args.seed, event_seed=args.event_seed, config_id=args.config_id,
                role=args.role, mechanism=args.mechanism, device=args.device,
                indegree=args.indegree, gmax_scale=args.gmax_scale, bias_scale=args.bias_scale,
                net_seed=args.net_seed, delay_bins=tuple(int(x) for x in args.delay_bins.split(",")),
                subtypes=not args.no_subtypes, slow=not args.no_slow)


def run_arm_main(args) -> int:
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    if args.arm:
        with open(args.arm) as f:
            spec = json.load(f)
    else:
        rid = args.run_id or "%s_%s_T%dms_d%s_s%s" % (args.stage, args.n, int(args.T_ms),
                                                      args.dose, args.seed)
        spec = arm_spec_from_cli(args, rid)
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
        for k in ("indegree", "gmax_scale", "bias_scale", "net_seed", "delay_bins",
                  "subtypes", "slow", "tau_slow_ms", "slow_inc"):
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


def make_plan(args) -> int:
    n = args.n
    doses = [float(x) for x in args.doses.split(",")]
    stage = args.make_plan
    arms = []
    if stage == "p0":
        for d in doses:
            rid = "%s_%s_T%dms_d%s_s%s" % (stage, n, int(args.T_ms), d, args.seed)
            arms.append(dict(run_id=rid, stage=stage, config_id="p5_%s" % stage, T_ms=float(args.T_ms),
                             drive_mult=d, seed=int(args.seed),
                             role=("zero_drive" if d == 0 else "dose"),
                             mechanism="none"))
    elif stage == "p2":
        for ratio in (1, 2, 4):
            T = float(args.T_ms) * ratio
            for d in doses:
                rid = "%s_%s_T%dms_d%s_s%s" % (stage, n, int(T), d, args.seed)
                arms.append(dict(run_id=rid, stage=stage, config_id="p5_%s" % stage, T_ms=T,
                                 drive_mult=d, seed=int(args.seed),
                                 role=("zero_drive" if d == 0 else "dose"),
                                 mechanism="none"))
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
    ap.add_argument("--event-seed", type=int, default=11)
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
    ap.add_argument("--net-seed", type=int, default=0)
    ap.add_argument("--delay-bins", default="40,60,80")
    ap.add_argument("--no-subtypes", action="store_true")
    ap.add_argument("--no-slow", action="store_true")
    ap.add_argument("--force", action="store_true", help="plan 模式：忽略已有 .done 重跑")
    args = ap.parse_args(argv)
    if args.make_plan:
        return make_plan(args)
    if args.plan:
        return run_plan_main(args)
    return run_arm_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
