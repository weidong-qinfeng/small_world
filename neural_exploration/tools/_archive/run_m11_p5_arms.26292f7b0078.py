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
  **唯一例外（R113 §⑤，本次修复）**：**零驱动臂**（`role == "zero_drive"` ∨ `drive_mult == 0`）
  的 **0 脉冲是核心观测而非故障** ⇒ **不抛错、正常落盘**，并写显式状态
  `zero_activity=true / spikes_total=0 / ei.ei_ratio=null / ei.i_exc_integral=0.0`；
  **非零驱动臂**的 0 响应仍是硬失败（`EXIT=1`，错误码**不再与正常观测混用**）。

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
  # ④ L4 机制键（R153 §⑤ D2：N2 → C-8 → N1 → N3）
  ./.venv-m9/bin/python -m neural_exploration.tools.run_m11_p5_arms --device cpu --n 1000 \\
      --mod --tau-mod-ms 2000 --gain-lo 1.3 --gain-hi 0.8 --rate-ref-hz 5.0        # C-8 慢增益
  ./.venv-m9/bin/python -m neural_exploration.tools.run_m11_p5_arms --device cpu --n 1000 \\
      --inh-gain 2.5                                                               # N1 类级抑制
  ./.venv-m9/bin/python -m neural_exploration.tools.run_m11_p5_arms --device cpu --n 1000 \\
      --stp --u0 0.5 --tau-fac-ms 5000 --tau-rec-ms 500 --stp-mode depress         # N3 STP

**L4 机制键（本段为接口契约；三族各自独立，互不影响）**
| 族 | spec 键 | 产物新键 | 跑前闸门 |
|---|---|---|---|
| **C-8 慢增益**（N2 之后第 2 优先）| `mod` / `mod_ablation` / `tau_mod_ms` / `gain_lo` / `gain_hi` / `rate_ref_hz` | `mod_enabled` / `mod_readback` / `mod_trace_stats` / `mod_rate_lag_corr` / `mod_params_ignored` / `mod_semantics` | **参数已给而 `mod` 未启用 ⇒ 拒跑 EXIT=101**（M10 M-4：静态增益 = 机制未实施；清单 L63）|
| **N1 类级抑制** | `inh_gain`（标量，乘 PV/SST/VIP） / `gmax_band_gain`（dict） | `net.inh_gain` / `net.gmax_band_gain` / `gmax_band_readback` / `gmax_band_applied` / `gmax_semantics` | 逐类乘子非法 ⇒ 拒跑 EXIT=103；读回不达带 ⇒ 断言失败 |
| **N3 STP depress** | `stp` / `u0` / `tau_fac_ms` / `tau_rec_ms` / `stp_mode` | `stp_enabled` / `stp_readback` / `stp_mode_asserted` / `stp_verdict` / `stp_params_ignored` / `stp_semantics` | **`mode != "depress"` 且 `stp` 开 ⇒ 拒跑 EXIT=102**（预注册 F-3：该臂 invalid）|

🔴 **字节不变契约**：三族的键**一律只在 spec 显式给出时才进入代码路径**（不调 `configure_mod`/
`configure_stp`、不给 `net_kw` 加键、不写新 JSON 键）⇒ **未给新键的既有臂产物逐字节不变**。
🔴 **拒跑语义**：`ArmRefused`（101/102/103）写 `.fail`（本文件规范：不计入分母、四态 `invalid`）
并以该码退出 ⇒ "**跑了也不算数**"与"**跑失败**"在盘上**可区分**。
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


def build_engine(n: int, spec: dict, device: str = "mps", mech: dict = None):
    """按 spec 装配 M11 内核（四亚型槽 + 层流 + 慢变量可关 + **L4 的 C-8/N1/N3**）。

    `mech`：由 `_mechanism_preflight(spec)` 得到的解析结果（`run_one` 传入；缺省则内部现算）。
    **本函数在 `mech` 的三族键全不出现时与接线前逐字等价**（不调 `configure_mod`/
    `configure_stp`、不给 `net_kw` 加键、`_net_echo` 也不多写键 ⇒ 产物字节不变）。
    """
    import torch
    from neural_exploration.src.mouse_cortex_circuit import (attach_to_engine,
                                                             build_laminar_probe_column)
    from neural_exploration.src.mouse_cortex_engine import REF_FROZEN, MouseCortexEngine
    if mech is None:
        mech = _mechanism_preflight(spec)
    net_kw = {k: v for k, v in spec.items()
              if k in ("indegree", "gmax_scale", "bias_scale", "bias_log_mean", "bias_log_sigma")}
    # ---- N1：逐类乘子 ⇒ 整表。**None ⇒ 不传 `gmax_band`** ⇒ 夹具走 DEFAULT_GMAX_BAND
    #      （签名默认），与接线前逐位一致 ----
    band, band_echo = _resolve_gmax_band(spec)
    if band is not None:
        net_kw["gmax_band"] = band
    bins = tuple(spec.get("delay_bins") or NET_DEFAULTS["delay_bins"])
    eng = MouseCortexEngine(n, device=device, use_compile=False, verbose=False)
    eng.configure_subtypes(bool(spec.get("subtypes", True)))
    if spec.get("slow", True):
        eng.configure_slow(True, tau_slow_ms=float(spec.get("tau_slow_ms", 1000.0)),
                           inc_per_spike=float(spec.get("slow_inc", 6.0)))
    # ---- C-8（慢增益变量 / E3）：**显式 slow=True**（M10 M-4 的防线；清单 L63）----
    if mech["mod_on"]:
        mp = mech["mod_params"]
        rb_mod = dict(eng.configure_mod(g_mod=1.0, slow=True,
                                        tau_mod_ms=mp["tau_mod_ms"], gain_hi=mp["gain_hi"],
                                        gain_lo=mp["gain_lo"], rate_ref_hz=mp["rate_ref_hz"]))
        # **读回断言**：引擎自带断言只覆盖 g_mod / mod_slow / tau_mod_ms
        # （larval_zf_engine.py **L100–102**）⇒ gain_hi / gain_lo / rate_ref_hz / slow 四项由本工具补：
        assert rb_mod["g_mod"] == 1.0, "mod 读回：g_mod=%r ≠ 1.0" % rb_mod["g_mod"]
        assert rb_mod["slow"] is True, ("mod 读回：slow=%r ⇒ 慢变量未接线（退化为静态增益 = "
                                        "机制未实施；M10 M-4 同族）" % rb_mod["slow"])
        assert rb_mod["tau_mod_ms"] == mp["tau_mod_ms"], "mod 读回：tau_mod_ms 不一致"
        assert rb_mod["gain_hi"] == mp["gain_hi"], "mod 读回：gain_hi 不一致"
        assert rb_mod["gain_lo"] == mp["gain_lo"], "mod 读回：gain_lo 不一致"
        assert rb_mod["rate_ref_hz"] == mp["rate_ref_hz"], "mod 读回：rate_ref_hz 不一致"
        assert bool(getattr(eng, "mod_slow", False)) is True, "mod 读回：引擎实例 mod_slow != True"
        assert getattr(eng, "t_mmod", None) is not None or not getattr(eng, "_built", False), \
            "mod 读回：慢变量张量 t_mmod 未分配（finalize 前允许为空，finalize 后必须存在）"
        mech["readback"]["mod"] = dict(rb_mod, explicit_slow_passed=True,
                                       call="configure_mod(g_mod=1.0, slow=True, tau_mod_ms, gain_hi, "
                                            "gain_lo, rate_ref_hz)")
    # ---- N3（STP depress）：接线 + **补 mode 回显/断言** ----
    if mech["stp_on"]:
        sp = mech["stp_params"]
        rb_stp = dict(eng.configure_stp(True, u0=sp["u0"], tau_fac_ms=sp["tau_fac_ms"],
                                        tau_rec_ms=sp["tau_rec_ms"], mode=str(mech["stp_mode"])))
        # 🔴 **补断言**：引擎 `configure_stp` 的读回断言**只覆盖 u0 与 tau_rec_ms**
        #    （larval_zf_engine.py **L78**）⇒ `mode` / `tau_fac_ms` / `enabled` **不在其中**
        #    ⇒ 若不同时回显，则"facilitate 已被显式禁止"在审计上不可核验（R153 §§预注册 §2.2）。
        assert rb_stp.get("mode") == str(mech["stp_mode"]), \
            "stp 读回：mode=%r ≠ 写入值 %r（引擎 L78 的断言不覆盖 mode ⇒ 本工具补）" % (
                rb_stp.get("mode"), str(mech["stp_mode"]))
        assert rb_stp.get("tau_fac_ms") == sp["tau_fac_ms"], \
            "stp 读回：tau_fac_ms=%r ≠ 写入值 %r（L78 亦不覆盖）" % (
                rb_stp.get("tau_fac_ms"), sp["tau_fac_ms"])
        assert rb_stp.get("u0") == sp["u0"], "stp 读回：u0 不一致"
        assert rb_stp.get("tau_rec_ms") == sp["tau_rec_ms"], "stp 读回：tau_rec_ms 不一致"
        assert rb_stp.get("enabled") is True, "stp 读回：enabled != True"
        assert str(eng.stp.get("mode")) == "depress", \
            ("stp 引擎**实装状态** mode=%r != 'depress' ⇒ invalid（预注册 F-3；"
             "跑前闸门只管 spec，这一条管引擎真实状态）" % (eng.stp.get("mode"),))
        mech["readback"]["stp"] = rb_stp
    if spec.get("refractory_form") == "m10_frozen":
        # 冻结档只在 n_ch=2（E1b 关）下有意义；P5 正式臂一律 REF_FIX（默认）
        raise SystemExit("P5 正式臂禁用 REF_FROZEN（缺陷档）")
    net = build_laminar_probe_column(n, seed=int(spec.get("net_seed", 0)), delay_bins=bins, **net_kw)
    if not spec.get("subtypes", True):
        cls = np.where(net["syn_class"] > 0, 1, 0).astype(np.int8)
        net = dict(net, syn_class=cls, inhibitory=(cls > 0))
    eng.configure_layers(net["layer_of"], delay_bins=bins)
    attach_to_engine(eng, net)          # ← 内含 finalize()：mod 慢变量张量在此被分配
    if mech["mod_on"]:
        # **finalize 之后的第二道读回**（装前那道只能证明"我们传了 slow=True"，这道证明
        # "引擎真的分配了慢变量状态"：larval_zf_engine.py **L125–126** ⇒ `t_mmod is not None`）
        assert getattr(eng, "t_mmod", None) is not None, \
            ("mod 已启用但 finalize 后 t_mmod 仍为 None ⇒ 慢变量状态未分配 ⇒ "
             "`MouseCortexEngine.run` 的 `mod_slow = bool(self.mod_slow) and "
             "getattr(self,'t_mmod',None) is not None`（L629）会退化为 False ⇒ "
             "静态增益 = 机制未实施（M10 M-4 同族）")
        assert bool(getattr(eng, "mod_slow", False)) is True, "finalize 后 mod_slow != True"
    # ---- N1 读回断言（必须在 net 装配之后：需要实装权重向量 net["gmax"]）----
    if band is not None:
        rb_band = _gmax_band_readback(net, band_echo, spec)
        _assert_gmax_band_readback(rb_band, spec)
        mech["readback"]["gmax_band"] = rb_band
        mech["readback"]["gmax_band_echo"] = band_echo
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
    # ---- L4 / N1：逐类乘子的**回显**（同样**只在 spec 显式带上时才写入** ⇒ 既有臂 JSON 逐字节不变）----
    if "inh_gain" in spec:
        d["inh_gain"] = float(spec["inh_gain"])
    if "gmax_band_gain" in spec:
        d["gmax_band_gain"] = {str(k).upper(): float(v) for k, v in
                               dict(spec["gmax_band_gain"]).items()}
    return d


# ======================================================================
# L4 机制接线（R153 §⑤ **D2 批准的顺序 = N2 → C-8 → N1 → N3**；N2 已接线，本段新增后三者）
# 依据：`docs/m11_l4_mechanism_plan.md` §13.3（C-8 判定）/§3 N1/§3 N3/§14（F-5/F-6）、
#       预注册 `data/m11_p5_prereg_amend_l4.json` 的 `candidate_dispositions`/`falsifiers_addendum`。
# ======================================================================
# 🔴 **三条共同纪律（每条都由一次既有失效逼出来）**：
#   ① **只在 spec 显式给了该机制的键时才进入代码路径** ⇒ 未给新键的既有臂**永不进入**任何新分支
#      ⇒ 产物与接线前**逐字节相同**（同 R116 §③ 的"逻辑证明 + 字节比对"验收法；同 `_net_echo`
#      对 `bias_log_sigma` 的既有约定）。
#   ② **"机制未实施"必须是可区分的显式状态，不得静默**（M10 D1 家族）：调质参数已给而机制未启用
#      ⇒ **硬拒绝**（`ArmRefused` 101）—— 绝不退化成"静态增益"（清单 **L63** 的 M-4 原样重演）。
#   ③ 落盘的读回**只写统计量 + 单位 + 来源**，不写整条轨迹（同 `_slow_trace_stats`）。
MOD_PARAM_KEYS = ("tau_mod_ms", "gain_lo", "gain_hi", "rate_ref_hz")
MOD_FAMILY_KEYS = ("mod", "mod_ablation") + MOD_PARAM_KEYS
GMAX_BAND_KEYS = ("inh_gain", "gmax_band_gain")
STP_PARAM_KEYS = ("u0", "tau_fac_ms", "tau_rec_ms", "stp_mode")
STP_FAMILY_KEYS = ("stp",) + STP_PARAM_KEYS
MOD_SAMPLE_MS = 50.0            # 引擎侧 `mod_trace` 采样周期（来源见 `_mod_sample_every`）


class ArmRefused(RuntimeError):
    """**拒绝产出该臂**（≠"跑失败"）：用于两类"跑了也不算数"的臂。

      - **code 101**：**调质（C-8）参数已给但机制未启用**（`mod`/`slow` 未开）⇒ 只会得到
        **静态增益**，即 M10 M-4 的"**慢变量根本没接线 ⇒ 机制未实施（而非机制无效）**"
        （清单 **L63**；R153 §② 的强制条款）；
      - **code 102**：`stp_mode != "depress"`（预注册 **F-3** 强制条款："若出现 `mode != depress`
        的臂 ⇒ 整批判 `invalid`，不进判定"）；
      - **code 103**：逐类乘子非法（非有限/非正）。

    由 `run_arm_main` 写成 `.fail`（本文件产物规范：**不计入分母、四态 `invalid`**）并以本码退出
    ⇒ "跑了也不算数"与"跑失败"在盘上**可区分**。
    """

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = int(code)


def _mod_sample_every(dt_ms: float) -> int:
    """引擎侧 `mod_trace` 的采样周期（步）。

    ⚠️ **来源说明（不得当成"读回值"）**：引擎**没有**把该量暴露成属性 —— `mod_sample_every`
    在 `mouse_cortex_engine.py` **L636** 与 `larval_zf_engine.py` **L311** 都是 `run()` 内的
    **局部变量** `max(1, int(round(50.0 / dt_ms)))`（对比 `slow_sample_every` 是实例属性，
    `mouse_cortex_engine.py` **L238/L352**）⇒ 本工具**重算同一个式子**，并在产物里标注
    `sample_every_source`。
    """
    return max(1, int(round(MOD_SAMPLE_MS / max(float(dt_ms), 1e-9))))


def _mechanism_preflight(spec: dict) -> dict:
    """L4 机制的**跑前口径闸门**（纯 spec 判定，不碰引擎 ⇒ 对既有臂零影响）。

    做三件事：① 执行"参数已给而机制未启用 ⇒ 拒绝"（M10 M-4 的防线）；② 执行
    "`stp_mode != depress` ⇒ invalid"（预注册 F-3）；③ 把"该臂是否声明了各机制的键"解析出来，
    供 `run_one` 决定是否写新 JSON 键（**门控 = 字节不变的关键**）。
    """
    mod_keys = [k for k in MOD_PARAM_KEYS if k in spec]
    mod_on = bool(spec.get("mod", False))
    mod_ablation = bool(spec.get("mod_ablation", False))
    if mod_keys and not mod_on and not mod_ablation:
        raise ArmRefused(101, (
            "调质（C-8）参数 %s 已提供，但机制未启用（spec['mod']=%r）⇒ 只会得到**静态增益 g_mod**，"
            "即 M10 M-4 的『慢变量根本没接线 ⇒ 机制未实施』而非『机制无效』（清单 L63 / R153 §②）"
            "⇒ 拒绝该臂。修法三选一：① 加 --mod（显式 slow=True）；② 把调质参数从该臂移除；"
            "③ 若确为**消融臂**，显式写 mod_ablation=true（会落盘 mod_params_ignored，仍不接线）。"
            % (mod_keys, spec.get("mod", None))))

    stp_keys = [k for k in STP_PARAM_KEYS if k in spec]
    stp_on = bool(spec.get("stp", False))
    stp_mode = str(spec.get("stp_mode", "depress"))
    if stp_on and stp_mode != "depress":
        raise ArmRefused(102, (
            "stp_mode=%r != 'depress'（预注册 F-3 强制条款：`mode != \"depress\"` 的臂 ⇒ **invalid**，"
            "不进判定；`facilitate` 方向使 `u` 向 u0 松弛 ⇒ 有效电导更高 ⇒ `(c)` 只会更差，"
            "larval_zf_engine.py **L488–490**）⇒ 拒绝该臂（EXIT=102）。"
            % (stp_mode,)))

    return dict(
        mod_family=any(k in spec for k in MOD_FAMILY_KEYS),
        mod_on=mod_on, mod_ablation=mod_ablation,
        mod_params_ignored=(list(mod_keys) if not mod_on else []),
        mod_params=dict(tau_mod_ms=float(spec.get("tau_mod_ms", 2000.0)),
                        gain_lo=float(spec.get("gain_lo", 0.8)),
                        gain_hi=float(spec.get("gain_hi", 1.3)),
                        rate_ref_hz=float(spec.get("rate_ref_hz", 5.0))),
        gmax_band_family=any(k in spec for k in GMAX_BAND_KEYS),
        stp_family=any(k in spec for k in STP_FAMILY_KEYS),
        stp_on=stp_on, stp_mode=stp_mode,
        stp_params_ignored=(list(stp_keys) if not stp_on else []),
        stp_params=dict(u0=float(spec.get("u0", 0.5)),
                        tau_fac_ms=float(spec.get("tau_fac_ms", 50.0)),
                        tau_rec_ms=float(spec.get("tau_rec_ms", 500.0))),
        readback=dict(mod=None, stp=None, gmax_band=None))


def _resolve_gmax_band(spec: dict) -> "tuple[dict, dict] | tuple[None, None]":
    """**逐类乘子 ⇒ 整表**（N1；接口要求：调用方给**乘子**，整表在内部构造）。

    🔴 **为什么只接受乘子、不接受整表**：`gmax_band` 在夹具里是**整表替换**
    （`mouse_cortex_circuit.py` **L111**：`dict(DEFAULT_GMAX_BAND if gmax_band is None else gmax_band)`）
    ⇒ 要求调用方手写整表会让**默认带一改就静默漂移**。故整表由 `DEFAULT_GMAX_BAND`
    （`circuit.py` **L30**）在内部构造后乘上逐类乘子。

    **两个旋钮的语义差异（必须随产物落盘，见 `gmax_semantics`）**：
      - `gmax_scale`（`circuit.py` **L43/L114**）：**统一**乘在**全部类**的权重上、**最外层**
        ⇒ **E/I 比逐位不变**（§2.5：它触及不到"E/I 相对强度"这一维）；
      - 逐类乘子（本函数）：只乘**指定的类** ⇒ E 段不动 ⇒ **改变 E/I 构成**（N1 的全部价值）。

    键：`inh_gain`（标量，乘 PV/SST/VIP 三类，E 段恒 1.0）与 `gmax_band_gain`
    （dict，任意子集，逐类优先于 `inh_gain`）。
    """
    has_scalar = "inh_gain" in spec
    has_dict = "gmax_band_gain" in spec
    if not (has_scalar or has_dict):
        return None, None
    from neural_exploration.src.mouse_cortex_circuit import DEFAULT_GMAX_BAND
    names = ("E", "PV", "SST", "VIP")
    mult = {nm: 1.0 for nm in names}
    if has_scalar:
        g = float(spec["inh_gain"])
        for nm in ("PV", "SST", "VIP"):
            mult[nm] = g
    if has_dict:
        for k, v in dict(spec["gmax_band_gain"]).items():
            k = str(k).upper()
            if k not in mult:
                raise ArmRefused(103, "gmax_band_gain 的类名 %r 不在 %r 中" % (k, names))
            mult[k] = float(v)
    for nm in names:
        if not np.isfinite(mult[nm]) or mult[nm] <= 0.0:
            raise ArmRefused(103, "逐类乘子必须有限且 >0：%s=%r" % (nm, mult[nm]))
    band = {nm: (float(DEFAULT_GMAX_BAND[nm][0]) * mult[nm],
                 float(DEFAULT_GMAX_BAND[nm][1]) * mult[nm]) for nm in names}
    echo = dict(multipliers={nm: float(mult[nm]) for nm in names},
                requested_band={nm: [float(band[nm][0]), float(band[nm][1])] for nm in names},
                source="DEFAULT_GMAX_BAND(circuit.py L30) × 逐类乘子（本工具内部构造整表）",
                inh_gain=(float(spec["inh_gain"]) if has_scalar else None),
                gmax_band_gain=({str(k).upper(): float(v) for k, v in dict(spec["gmax_band_gain"]).items()}
                                if has_dict else None))
    return band, echo


def _gmax_band_readback(net: dict, resolved: dict, spec: dict, min_edges: int = 200) -> dict:
    """**N1 的读回断言**：写入值 == 夹具实装值（逐类）。

    🔴 **必须先登记的接口事实（否则会把"没做"说成"做了"）**：夹具的 `meta`
    （`mouse_cortex_circuit.py` **L121–131**）**不回显 `gmax_band`** —— 全仓
    `grep -n gmax_band` 只有 **2 处命中**（**L42** 函数签名 / **L111** 整表替换），`meta` 里只有
    `gmax_scale` / `class_edge_counts` / `type_counts` 等。⇒ **任务书要求的"写入值 == 夹具 meta
    回显值"这一字面比对在 `gmax_band` 上不可得**（见未确认项 U-N1-1）。
    **替代读回（可核且更强）**：`meta` 回显的 `gmax_scale` 逐位比对 **＋** 用**实装权重向量**
    `net["gmax"]` 按 `syn_class` 分组取 min/max，断言其落在**请求带** `[lo,hi]` 的
    ±`tol_rel` 之内（`gmax = (lo + (hi−lo)·u)·gmax_scale`，`u~U[0,1)` ⇒ min→lo、max→hi）。
    边数 < `min_edges` 时（抽样覆盖不足，min/max 达不到带端点）**不断言**，只落 `insufficient_edges`。
    """
    g = np.asarray(net["gmax"], dtype=np.float64)
    cls = np.asarray(net["syn_class"], dtype=np.int64)
    names = ("E", "PV", "SST", "VIP")
    tol_rel = 0.02
    want_scale = float(spec.get("gmax_scale", 1.0))
    out = dict(requested_band=resolved["requested_band"], multipliers=resolved["multipliers"],
               applied_via="build_laminar_probe_column(gmax_band=<内部构造的整表>)（circuit.py L42/L111）",
               meta_echo_gmax_scale=float(net["meta"]["gmax_scale"]),
               meta_echo_gmax_scale_expected=want_scale,
               meta_echo_matches=bool(float(net["meta"]["gmax_scale"]) == want_scale),
               tol_rel=tol_rel, observed={}, n_edge_per_class={},
               meta_has_gmax_band=("gmax_band" in net["meta"]))
    for c, nm in enumerate(names):
        m = (cls == c)
        out["n_edge_per_class"][nm] = int(m.sum())
        if not m.any():
            out["observed"][nm] = None
            continue
        lo, hi = float(resolved["requested_band"][nm][0]), float(resolved["requested_band"][nm][1])
        o_min, o_max = float(g[m].min()), float(g[m].max())
        tol = tol_rel * (hi - lo) + 1e-6
        rec = dict(min=o_min, max=o_max, lo_expected=lo, hi_expected=hi,
                   min_ok=bool(abs(o_min - lo) <= tol), max_ok=bool(abs(o_max - hi) <= tol))
        if int(m.sum()) < int(min_edges):
            rec["insufficient_edges"] = ("边数 %d < %d ⇒ min/max 达不到带端点，本类**不断言**"
                                         "（抽样覆盖不足，非机制未实施）" % (int(m.sum()), int(min_edges)))
        out["observed"][nm] = rec
    out["all_classes_in_band"] = all(
        (v is None) or bool(v.get("insufficient_edges")) or (v["min_ok"] and v["max_ok"])
        for v in out["observed"].values())
    return out


def _assert_gmax_band_readback(rb: dict, spec: dict) -> None:
    assert rb["meta_echo_matches"], (
        "N1 读回断言失败：夹具 meta.gmax_scale=%r != 写入值 %r（写入未到达夹具）"
        % (rb["meta_echo_gmax_scale"], rb["meta_echo_gmax_scale_expected"]))
    assert rb["all_classes_in_band"], (
        "N1 读回断言失败：实装权重带（net['gmax'] 按 syn_class 分组的 min/max）不在请求带内 "
        "⇒ 逐类乘子未生效（这是『机制未实施』，不得静默通过）：%r"
        % {k: v for k, v in rb["observed"].items() if v and not v.get("insufficient_edges")
           and not (v["min_ok"] and v["max_ok"])})
    if rb["multipliers"]["E"] != 1.0:
        pass          # E 段乘子允许显式给出（通用接口），默认 1.0（N1 要求 E 段不动）


def _mod_trace_stats(eng, spec: dict, res: dict, mech: dict, n_settle: int, n_meas: int) -> dict:
    """**慢增益变量（C-8/E3）机制读回**（PC-3 硬前置；M10 M-4 的防线）。

    🔴 **结构性事实（本节点读码 + 实测确认，必须登记）**：任务书说的 `res["mod_trace"]`
    （`larval_zf_engine.py` **L447–449**）在**本工具实际使用的 `MouseCortexEngine` 上不存在** ——
    该类的 `run()`（`mouse_cortex_engine.py` **L551–867**）**整体覆写**父类 `run()`，只写
    `res["slow_trace"]`（**L858–864**），**没有** `res["mod_trace"]` 这一段。
    ⇒ 本工具改用**引擎实例属性** `eng.mod_trace`（`mouse_cortex_engine.py` **L744** 每 50 ms
    `self.mod_trace.append(...)`）作为采样序列，并把 `res_mod_trace_key_present` 一并落盘
    ⇒ **"缺的那个键"本身成为可核对的证据**，而不是被静默绕过。
    """
    raw = list(getattr(eng, "mod_trace", []) or [])
    dt = float(eng.point.dt_ms)
    every = _mod_sample_every(dt)
    mod_dict = dict(getattr(eng, "mod", {}) or {})
    rb = dict(g_mod=float(getattr(eng, "g_mod", 1.0)),
              mod_slow=bool(getattr(eng, "mod_slow", False)),
              t_mmod_present=bool(getattr(eng, "t_mmod", None) is not None),
              tau_mod_ms=mod_dict.get("tau_mod_ms"),
              gain_hi=mod_dict.get("gain_hi"), gain_lo=mod_dict.get("gain_lo"),
              rate_ref_hz=mod_dict.get("rate_ref_hz"))
    _ghi = rb.get("gain_hi")
    _glo = rb.get("gain_lo")
    out = dict(enabled=bool(mech["mod_on"]), n=len(raw), readback=rb,
               n_settle_samples=int(n_settle) // every,
               n_meas_samples=max(0, len(raw) - int(n_settle) // every),
               sample_every_steps=every,
               sample_every_source=("本工具重算 max(1, round(50/dt_ms))（引擎把它放在 run() 局部变量里，"
                                    "未暴露为属性：mouse_cortex_engine.py L636 / larval_zf_engine.py L311）"),
               res_mod_trace_key_present=bool(isinstance(res, dict) and "mod_trace" in res),
               res_mod_trace_gap=("MouseCortexEngine.run() 覆写父类且不写 res['mod_trace']"
                                  "（mouse_cortex_engine.py L551–867；只写 slow_trace L858–864）"
                                  "⇒ 采样序列改由引擎实例属性 eng.mod_trace 读出（L744）"),
               alive_criterion="std > 1e-3（与引擎 res['mod_trace'].alive 同阈值，larval_zf_engine.py L455）")
    if raw:
        arr = np.asarray(raw, dtype=float)
        k = int(np.argmax(arr))
        out.update(mean=float(arr.mean()), std=float(arr.std()), min=float(arr.min()),
                   max=float(arr.max()), dynamic_range=float(arr.max() - arr.min()),
                   argmax_sample_index=k, argmax_step=int(k * every),
                   argmax_t_ms=float(k * every * dt),
                   alive=bool(float(arr.std()) > 1e-3))
        if (_ghi is not None) and (_glo is not None):
            out["gain_range"] = [float(_glo + (_ghi - _glo) * float(arr.min())),
                                 float(_glo + (_ghi - _glo) * float(arr.max()))]
        _tmm = getattr(eng, "t_mmod", None)
        if _tmm is not None:
            out["final_m"] = float(_tmm.detach().cpu().reshape(-1)[0].item())
            out["final_gain"] = (float(_glo + (_ghi - _glo) * out["final_m"])
                                 if (_ghi is not None and _glo is not None) else None)
    else:
        out.update(mean=None, std=None, min=None, max=None, dynamic_range=None,
                   argmax_sample_index=None, argmax_step=None, argmax_t_ms=None, alive=False,
                   empty_reason=("引擎未产生采样序列：mod 未启用（enabled=False）"
                                 if not mech["mod_on"] else
                                 "mod 已启用但表步数 < sample_every(%d) ⇒ 未触发采样（极短诊断臂）" % every))
    out["units"] = dict(argmax_step="engine step（原点 = 本次 run() 调用的表起点）",
                        argmax_t_ms="millisecond", sample_every_steps="step",
                        mean_std_min_max="无量纲（m 是 target=rate/rate_ref_hz 的低通 ⇒ O(1)）",
                        gain_range="增益倍数（gain_lo + (gain_hi−gain_lo)·m）")
    return out


def _mod_rate_lag_corr(eng, res: dict, mech: dict, n_settle: int, n_meas: int,
                       lag_max: int = 3) -> dict:
    """**`mod_trace` 与群体率的 lag 相关**（F-5(i) 的第三项硬前置；R153 §② 强制）。

    **定义（逐字；含单位、原点与"为什么这样切分"）**
      - `m_j` = 引擎侧慢增益样本序列 `eng.mod_trace`（**不是** `res["mod_trace"]` —— 该键在
        `MouseCortexEngine` 上不存在，见 `_mod_trace_stats` 的登记）。采样周期
        `every = max(1, round(50 ms / dt_ms))` **步**；第 j 个样本在处理完**局部步号
        `j·every − 1`** 之后追加（`mouse_cortex_engine.py` **L737–744**）⇒ 它反映的正是**该
        50 ms 箱内**的活动。
      - 🔴 **切分（关键，不是启发式对齐）**：`mod_trace` 跨**两次 `run()` 调用**累积 —— settle 段
        （`n_settle` 步，其 `pop_trace` **被丢弃**）贡献 `n_settle // every` 个样本，测量段贡献
        `n_meas // every` 个；且每次 `run()` 内部的 `_mstep` 都从 0 起算（`mouse_cortex_engine.py`
        L637）⇒ **按 `n_settle // every` 精确丢弃前缀**即得"与测量窗逐箱对齐"的序列。
      - `r_b` = **测量窗内**第 b 个 50 ms 箱的**群体发放率**（Hz）= `pop[(b)·every : (b+1)·every]`
        的均值 / `(every · dt_ms · 1e-3)`；`pop` = `res["pop"]`（**逐步群体脉冲数**，float32，
        原点 = 测量窗起点）。
      - **lag 相关**：`ρ(k) = Pearson(m_b, r_{b+k})`，配对 b 需两侧都有定义；`k` 的**单位 = 样本
        = `every·dt_ms` ms**（落盘 `lag_ms`），`k>0` ⇒ **率滞后于 m**（`k<0` ⇒ m 滞后于率）。
        落盘 `ρ(0)`、`ρ(k)` 全表、`|ρ|` 最大者及其 lag。
      - 样本量门：`K = n_meas // every`；`K < 2·(lag_max+1)` ⇒ **判"不可得"并写明原因**，
        不得用极短序列自造相关。
    """
    dt = float(eng.point.dt_ms)
    every = _mod_sample_every(dt)
    pop = res.get("pop")
    base = dict(n_settle=n_settle, n_meas=n_meas, sample_every_steps=every,
                sample_every_ms=float(every) * dt, lag_unit="sample(=%g ms)" % (every * dt),
                definition=("rho(k)=Pearson(m_b, r_{b+k})；m=eng.mod_trace 的测量段样本（每 %d 步一个），"
                            "r=测量窗每 %d 步箱的群体率(Hz)=箱内 pop 计数均值/(%d·dt_s)；"
                            "k>0 ⇒ 率滞后于 m" % (every, every, every)),
                source="eng.mod_trace（引擎实例属性，mouse_cortex_engine.py L744）+ res['pop']")
    if not bool(mech["mod_on"]):
        base.update(computable=False, reason="mod 未启用（enabled=False）⇒ 无 m 序列")
        return base
    if pop is None:
        base.update(computable=False, reason="res['pop'] 为 None（未请求 pop_trace）")
        return base
    raw = list(getattr(eng, "mod_trace", []) or [])
    pop = np.asarray(pop, dtype=float)
    K = int(n_meas) // every
    base["n_bins_meas"] = K
    if K < 2 * (int(lag_max) + 1):
        base.update(computable=False,
                    reason=("测量窗箱数 K=%d < 2·(lag_max+1)=%d ⇒ 样本量不足以给 ±%d 样本的 lag 相关"
                            "（不得自造）" % (K, 2 * (int(lag_max) + 1), int(lag_max))))
        return base
    drop = int(n_settle) // every
    m = np.asarray(raw[drop:drop + K], dtype=float)
    if m.size != K:
        base.update(computable=False,
                    reason=("切分失败：eng.mod_trace 长度 %d，按 drop=%d 取测量段只得 %d ≠ K=%d "
                            "⇒ 拒绝给出相关（宁可'不可得'，不得自造对齐）" % (len(raw), drop, m.size, K)))
        return base
    rate = pop[:K * every].reshape(K, every).mean(axis=1) / (every * dt * 1e-3)
    base.update(m_mean=float(m.mean()), m_std=float(m.std()),
                rate_mean_hz=float(rate.mean()), rate_std_hz=float(rate.std()))
    if float(m.std()) == 0.0 or float(rate.std()) == 0.0:
        base.update(computable=False,
                    reason=("序列常值（m.std=%r, rate.std=%r）⇒ Pearson 无定义"
                            % (float(m.std()), float(rate.std()))))
        return base
    lags, rho, n_pairs = [], [], []
    for k in range(-int(lag_max), int(lag_max) + 1):
        b0, b1 = max(0, -k), min(K, K - k)
        if b1 - b0 < 3:
            lags.append(k); rho.append(None); n_pairs.append(int(b1 - b0)); continue
        a, c = m[b0:b1], rate[(b0 + k):(b1 + k)]
        if float(a.std()) == 0.0 or float(c.std()) == 0.0:
            lags.append(k); rho.append(None); n_pairs.append(int(b1 - b0)); continue
        lags.append(k)
        rho.append(float(np.corrcoef(a, c)[0, 1]))
        n_pairs.append(int(b1 - b0))
    fin = [(abs(v), i) for i, v in enumerate(rho) if v is not None]
    best_i = max(fin)[1] if fin else None
    base.update(computable=True, lags=lags, rho=rho, n_pairs=n_pairs,
                lag_ms=[int(k * every * dt) for k in lags],
                rho_lag0=(rho[lags.index(0)] if 0 in lags else None),
                best_lag=(lags[best_i] if best_i is not None else None),
                best_rho=(rho[best_i] if best_i is not None else None),
                best_abs_rho=(abs(rho[best_i]) if best_i is not None else None),
                n_meas_samples_used=int(m.size))
    return base


def _stp_state_stats(eng, mech: dict) -> dict:
    """**STP 投递因子 `u·x` 的直接读回**（引擎侧 `t_u`/`t_x` 的终态统计）。

    > 计划 §3 N3 的读回量原文说"`u`/`x` 的统计量**引擎当前未导出** ⇒ 本计划不要求新增该导出"。
    > **本工具在生产者侧补上这个读回**（不改引擎）：`t_u` / `t_x` 是**实例属性**
    > （`larval_zf_engine.py` **L132–136**）⇒ `run_one` 在 `run()` 之后可直接读。

    ⚠️ **口径（不得混读）**：读的是**测量窗结束时刻**的值 ⇒ 对**最后一批发放的边**它是
    **投递后**（post-update）的值 ⇒ 它是投递因子（引擎取**更新前**的 `u·x`，
    `mouse_cortex_engine.py` **L943–944**）的**后验均值场**，不是逐时刻精确前值。
    🔴 **dtype = float32**（`self.dtype`）⇒ `u` 的几何衰减（mode='depress'：`u ← u·exp(−dt/τ_fac)`，
    `larval_zf_engine.py` **L492**）在长时间静默后**下溢为精确 0.0** ⇒
    "投递因子测到 0" 是**真读数**而非缺失（这正是 F-3/P1 说的"递归隔离消融"）。
    """
    out = dict(enabled=bool(mech["stp_on"]), dtype=str(getattr(eng, "dtype", "")))
    if not mech["stp_on"]:
        out["note"] = "stp 未启用 ⇒ 无 u/x 状态（显式缺席，非键丢失）"
        return out
    tu, tx = getattr(eng, "t_u", None), getattr(eng, "t_x", None)
    if tu is None or tx is None:
        out["note"] = "引擎未分配 t_u/t_x（stp_enabled=True 但 _init_stp_state 未执行）"
        out["missing_state"] = True
        return out
    u = tu.detach().cpu().numpy().astype(np.float64)
    x = tx.detach().cpu().numpy().astype(np.float64)
    fx = u * x
    eps = 1e-6
    out.update(n_edge=int(u.size),
               u_mean=float(u.mean()), u_min=float(u.min()), u_max=float(u.max()),
               x_mean=float(x.mean()), x_min=float(x.min()), x_max=float(x.max()),
               delivery_factor_mean_u_times_x=float(fx.mean()),
               delivery_factor_median=float(np.median(fx)),
               delivery_factor_max=float(fx.max()),
               frac_edges_factor_below_eps=float((fx < eps).mean()),
               frac_edges_u_exactly_zero=float((u == 0.0).mean()),
               eps=eps,
               units=dict(u="无量纲（利用度）", x="无量纲（恢复池，0..1）",
                          delivery_factor="无量纲（投递电导的乘子，1 = 无 STP）"),
               caliber=("测量窗**结束时刻**的逐边状态（投递后）；引擎投递取**更新前** u·x "
                        "⇒ 这是后验均值场，不是逐时刻精确前值"),
               tau_fac_ms=float(eng.stp.get("tau_fac_ms", float("nan"))),
               tau_rec_ms=float(eng.stp.get("tau_rec_ms", float("nan"))),
               u0=float(eng.stp.get("u0", float("nan"))),
               mode=str(eng.stp.get("mode")),
               readback_note=("t_u/t_x 未参与 res 字典 ⇒ 本工具从引擎实例属性直读"
                              "（larval_zf_engine.py L132–136 分配）"))
    return out


def _slow_trace_stats(eng, spec: dict, res: dict):
    """**慢变量机制读回（L2 硬前置；M10 D1「静默失效」家族的防线）**。

    返回 `slow_trace_stats` 统计量（**不写整条轨迹**，避免产物膨胀）或 **`None`**。

    ⚠️ **"缺席"必须是可区分的显式状态**：`slow=False` 时返回 `None`，且臂 JSON 同时写
    `slow_enabled=False` ⇒ 读回断言看到的是"显式未启用"，而不是"键丢了"。

    **单位与原点（R80 §⑥：跨接口传递的量必须在接口处写明单位）**：
      - `argmax_step` = 采样序号 argmax × `sample_every_steps`，**单位 = 引擎步（原点含 settle 的表起点）**；
      - `argmax_t_ms` = `argmax_step × dt_ms`（ms）；
      - `mean/std/min/max` = **采样序列**（每 `sample_every_steps` 步取一次 `t_slow` 的神经元均值）上的统计量；
      - `alive` 取自引擎公开读出 `slow_readback()`（判**逐神经元状态**是否真在动），另给 `trace_range`
        （采样序列极差）供交叉核对；`engine_trace_*_matches` 为与引擎内部统计量的一致性核对。
    """
    if not bool(spec.get("slow", True)):
        return None
    try:
        rb = dict(eng.slow_readback())
    except Exception as ex:  # noqa: BLE001
        rb = {"readback_error": str(ex)}
    tr = res.get("slow_trace") or {}
    raw = list(getattr(eng, "slow_trace", []) or [])
    every = int(getattr(eng, "slow_sample_every", 2000) or 2000)
    dt = float(eng.point.dt_ms)
    if raw:
        arr = np.asarray(raw, dtype=float)
        k = int(np.argmax(arr))
        stats = dict(n=int(arr.size), alive=rb.get("alive"),
                     mean=float(arr.mean()), std=float(arr.std()),
                     min=float(arr.min()), max=float(arr.max()),
                     argmax_step=int(k * every), argmax_sample_index=k,
                     sample_every_steps=every, argmax_t_ms=float(k * every * dt))
        stats["engine_trace_n_matches"] = bool(int(tr.get("n", -1)) == int(arr.size))
        stats["engine_trace_mean_matches"] = bool(tr and abs(float(tr.get("mean", 1e30))
                                                            - float(arr.mean())) <= 1e-9)
    else:
        stats = dict(n=0, alive=rb.get("alive"), mean=None, std=None, min=None, max=None,
                     argmax_step=None, argmax_sample_index=None, sample_every_steps=every,
                     argmax_t_ms=None,
                     empty_reason="引擎未产生采样序列（slow_on ∧ 采样未触发）")
    stats["readback"] = {k: rb.get(k) for k in
                         ("implemented", "n", "mean", "std", "min", "max", "dynamic_range",
                          "n_nonzero", "frac_nonzero", "trace_n", "trace_std", "trace_range",
                          "alive", "inc_per_spike", "tau_slow_ms")}
    stats["units"] = dict(argmax_step="engine step（原点=含 settle 的表起点）",
                          argmax_t_ms="millisecond", sample_every_steps="step",
                          mean_std_min_max="慢电导内部单位（与 t_slow 同）")
    return stats


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
    # ---- L4 机制跑前闸门（**先于任何产物写入**）：调质参数未启用 ⇒ 拒跑（101）；stp mode 非
    #      depress ⇒ invalid（102）。对不含新键的既有臂**纯只读**（三族键全不出现 ⇒ 全 False）----
    mech = _mechanism_preflight(spec)

    spec = dict(spec, net_seed=_seeds["net_seed"])      # ← 派生后的网络种子显式传下去
    eng, net, torch = build_engine(n, spec, device=device, mech=mech)
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
    # ---- **零驱动臂的"零活动"是显式状态，不是引擎故障**（M11 R113 §⑤；M10 D1「缺席必须是可区分的
    # 显式状态」家族的又一次复现）----
    # ⚠️ **缺陷（本次修复）**：下面那行原先是**无条件**硬断言 ⇒ `i_exc_integral == 0` 时
    #   ①"零驱动臂 0 脉冲"（L3b 的**核心观测**：自持比 R→0，**正常且有价值**）
    #   与 ②"引擎坏了"（真错误）
    #   **共用同一个错误码**（`EXIT=1` + 产物**根本不落盘**）⇒ **无法区分**。
    #   实测：`role=zero_drive, bias=1.00, σ=0.05, T=1 s` ⇒ 0 脉冲 ⇒ 原断言触发 ⇒ 无 JSON 产物。
    # **判据（语义轴 ∨ 数值轴，两条都覆盖）**：
    #   ① `role == "zero_drive"`（语义轴）：`make_plan()` 对 `dose==0` 自动打的角色；
    #   ② `drive_mult == 0`（数值轴）：外部事件率 `drive_rate_hz * dose == 0` ⇒ `make_events()`
    #      的 `n_ev = int(rate*T_table*dt*1e-3*n) == 0` ⇒ 引擎收到**空事件表**（与①同一物理条件）。
    # ⇒ **必须取或**：既有产物里两种写法都出现过（模板给 `role=zero_drive, dose=0.0`；手写 spec 只给
    #   `drive_mult=0` 而 role 仍写作 `dose`/`baseline`）⇒ 只认一条必漏另一条。
    # **绝非放松**：`role != zero_drive` **且** `drive_mult != 0` ⇒ **原断言逐字保留**
    #   （非零驱动下零响应 = 网络真坏；反例探针实测仍 `EXIT=1`，见验证记录）。
    # **新分支只落在"原本必然抛错"的那条路径上** ⇒ 对 `i_exc_integral > 0` 的既有臂
    #   ⇒ 永不进入 ⇒ 产物**逐字节不变**（改前/改后 sha256 实测一致）。
    _ei = res["ei_integrals"]
    _zero_drive_expected = (str(spec.get("role", "dose")) == "zero_drive") or (dose == 0.0)
    ei_ratio_out = _ei["ei_ratio"]       # 引擎 `ei_readback()` 在 `exc == 0` 时**已经**给 `None`（非 0！）
    zero_activity = None                 # None ⇒ 未进入新分支（正常路径）；bool ⇒ 本分支的**显式状态**
    if not _zero_drive_expected:
        # **非零驱动臂：原硬断言逐字保留（同一判据、同一文本）**
        assert _ei["i_exc_integral"] > 0, "E/I 积分必须由引擎实测且非零"
    elif not bool(_ei["i_exc_integral"] > 0):
        # **零驱动臂 × 未实测到非零 E 电流积分** ⇒ 允许零活动，**正常落盘**
        #   `zero_activity=True`（脉冲数 0）/ `False`（有脉冲但 E 电流积分为 0 ⇒ 同样不得静默）
        zero_activity = (spk_total == 0)
        try:                             # 只在本分支规范化：`ei_ratio` 必须是 JSON `null` 或有限数
            if ei_ratio_out is None or not bool(np.isfinite(float(ei_ratio_out))):
                ei_ratio_out = None      # **"缺席"（null）与"测得 0"（0.0）必须可区分**
        except (TypeError, ValueError):
            ei_ratio_out = None

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
                ei_ratio=ei_ratio_out,        # 正常路径 == 引擎原值（逐字节不变）；零活动路径 ⇒ null
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
        # ---- L2 硬前置：慢变量机制读回（slow=False ⇒ 显式 null + slow_enabled=False）----
        slow_enabled=bool(spec.get("slow", True)),
        slow_trace_stats=_slow_trace_stats(eng, spec, res),
    )
    if zero_activity is not None:
        # **显式状态字段（只在本任务新开的零活动路径上出现 ⇒ 既有臂 JSON 逐字节不变）**：
        #   `zero_activity=True, spikes_total=0, ei.ei_ratio=None(null), ei.i_exc_integral=0.0,
        #    ei.i_inh_integral=0.0` —— 读回方据此**区分**"零活动（正常观测）"与"引擎坏了（真错误）"。
        arm["zero_activity"] = bool(zero_activity)
        arm["zero_activity_criterion"] = (
            "role=%r, drive_mult=%r ⇒ zero_drive_expected=%s；i_exc_integral=%r（引擎实测，"
            "未达 >0 判据）⇒ %s"
            % (str(spec.get("role", "dose")), dose, _zero_drive_expected,
               float(_ei["i_exc_integral"]),
               ("允许零活动（=L3b 核心观测，非错误）" if zero_activity
                else "E 电流积分为 0 但脉冲数 >0（如实记录，不静默）")))
    # **读回断言前置自检**（R102/§① 精神：生产者先自己证明"机制是活的"，不把问题留给下游）
    # 只在"步数足以采样"时才硬断言（极短诊断臂不因采样未触发而失败，但会显式登记原因）
    if arm["slow_enabled"]:
        _st = arm["slow_trace_stats"] or {}
        _every = int(getattr(eng, "slow_sample_every", 2000) or 2000)
        if (n_settle + n_meas) >= _every:
            assert (_st.get("n") or 0) > 0, ("slow=True 且步数足以采样，但 slow_trace_stats.n == 0 "
                                             "⇒ 机制静默失效（不得静默通过）")
            assert _st.get("alive") is not None, "slow_trace_stats.alive 必须由引擎读出（不得为 None）"
            assert _st.get("argmax_step") is not None, "slow_trace_stats.argmax_step 缺失"
        else:
            _st["insufficient_steps_to_sample"] = True
            _st["note"] = ("表步数 %d < sample_every(%d) ⇒ 未采样（**极短诊断臂**的显式状态，"
                           "不是字段丢失）" % (n_settle + n_meas, _every))
    # ==================================================================
    # L4 机制的新键（**三族各自门控：spec 不含该族任何键 ⇒ 一个键都不写**）
    #   ⇒ 既有臂（无新键）的 JSON 与 trace **逐字节不变**（R116 §③ 的逻辑证明同族）。
    #   反向保证（"机制未实施"不可能静默）：只要 spec **声明过**该族的键，
    #   ① 若机制启用 ⇒ `*_enabled=True` + 完整读回 + 硬断言；
    #   ② 若机制未启用 ⇒ 调质族**硬拒跑**（101），STP 族写 `*_enabled=False` +
    #      `*_params_ignored`（显式列出被忽略的键，不静默）。
    # ==================================================================
    if mech["mod_family"]:
        arm["mod_enabled"] = bool(mech["mod_on"])
        arm["mod_readback"] = mech["readback"]["mod"]        # None ⇒ 显式"未启用"（非"键丢了"）
        arm["mod_params_ignored"] = list(mech["mod_params_ignored"])
        arm["mod_ablation"] = bool(mech["mod_ablation"])
        arm["mod_trace_stats"] = _mod_trace_stats(eng, spec, res, mech, n_settle, n_meas)
        arm["mod_rate_lag_corr"] = _mod_rate_lag_corr(eng, res, mech, n_settle, n_meas)
        arm["mod_semantics"] = (
            "C-8（稳定/稳态可塑性，抽象为慢增益变量；PC-6/清单 L141）。更新方程 "
            "m ← m + (dt/τ)·(target − m)、target = 瞬时群体发放率/rate_ref_hz（活动依赖）；"
            "增益 = gain_lo + (gain_hi−gain_lo)·m；作用点 = 突触电导（mouse_cortex_engine.py "
            "L938–940）**与外部事件幅度**（L759/L780）—— 不是只调背景率。"
            "gain_hi<gain_lo ⇒ 负反馈（服务 (c)）；gain_hi>gain_lo ⇒ 正反馈（服务 dwell）。")
    if mech["gmax_band_family"]:
        arm["gmax_band_readback"] = mech["readback"]["gmax_band"]
        arm["gmax_band_applied"] = mech["readback"]["gmax_band_echo"]   # 乘子 + 解析后的整表 + 来源
        arm["gmax_semantics"] = (
            "**两个旋钮语义不同，不得混读**：`gmax_scale`（net.gmax_scale，circuit.py L43/L114）"
            "**统一**乘在**全部类**的权重上、最外层 ⇒ **E/I 比逐位不变**；逐类乘子"
            "（`inh_gain` / `gmax_band_gain`，本工具内部把它乘到 DEFAULT_GMAX_BAND（circuit.py L30）"
            "上构造整表后传给夹具）**只乘指定的类** ⇒ E 段不动 ⇒ **改变 E/I 构成**（N1 的全部价值）。"
            "⚠️ 归因限制（F-2(iv)）：`syn_class` 是**逐突触随机标签**（circuit.py L99–105），"
            "与突触前/后细胞类型均无关 ⇒ **只许报『该随机类槽权重 ↑』，不得写成『PV 细胞介导』**。")
    if mech["stp_family"]:
        arm["stp_enabled"] = bool(mech["stp_on"])
        arm["stp_readback"] = mech["readback"]["stp"]        # configure_stp 返回 dict **逐字**落盘
        arm["stp_params_ignored"] = list(mech["stp_params_ignored"])
        arm["stp_mode_asserted"] = str(mech["stp_mode"])
        arm["stp_state_stats"] = _stp_state_stats(eng, mech)
        arm["stp_verdict"] = ("pass_structure" if mech["stp_on"] else "disabled")
        arm["stp_semantics"] = (
            "N3 = C2 STP depress（Tsodyks–Markram）。投递因子取**更新前**的 u·x"
            "（mouse_cortex_engine.py L943–944），`u` 在 mode='depress' 下按 tau_fac 指数衰减到 0"
            "（larval_zf_engine.py L492）⇒ **逐边 ISI ≫ tau_fac 时投递因子 → 0 = 递归隔离消融**"
            "（§2.7；F-3 的 P1）。⇒ 控制旋钮是 **tau_fac_ms 而非 tau_rec_ms**；"
            "`mode != 'depress'` ⇒ 该臂 invalid（F-3，跑前即拒，EXIT=102）。"
            "⚠️ 启用 STP 会**强制放弃 chunk 快路径**（mouse_cortex_engine.py L896–900）⇒ 单价必高于 L3。")
        if mech["stp_on"]:
            _st = arm["stp_readback"] or {}
            assert _st.get("mode") == "depress", \
                ("stp_readback.mode=%r ⇒ 该臂 invalid（预注册 F-3：mode != 'depress' 的臂整批 invalid）"
                 % (_st.get("mode"),))
    if mech["mod_family"] and mech["mod_on"]:
        # **读回断言（强制条款 (c)，R153 §②）**：`mod_trace.n > 0` 且 `std > 0`
        _mt = arm["mod_trace_stats"] or {}
        _every = int(_mt.get("sample_every_steps") or _mod_sample_every(float(eng.point.dt_ms)))
        if (n_settle + n_meas) >= 2 * _every:
            assert (_mt.get("n") or 0) > 0, (
                "mod=True 且步数足以采样（%d 步 ≥ 2×sample_every=%d），但 mod_trace_stats.n == 0 "
                "⇒ 慢增益变量**静默失效**（M10 M-4 同族）⇒ 不得静默通过"
                % (n_settle + n_meas, 2 * _every))
            assert float(_mt.get("std") or 0.0) > 0.0, (
                "mod=True 但 mod_trace.std == 0（m 恒定）⇒ 增益是**静态**的 ⇒ 机制未实施"
                "（这正是 M10 M-4 的形态）⇒ 不得静默通过；stats=%r" % (_mt,))
        else:
            _mt["insufficient_steps_to_sample"] = True
            _mt["note"] = ("表步数 %d < 2×sample_every(%d) ⇒ 采样不足以给 std>0 断言"
                           "（**极短诊断臂**的显式状态，不是字段丢失）" % (n_settle + n_meas, 2 * _every))
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
    if getattr(args, "tau_slow_ms", None) is not None:         # L2 杠杆：仅显式时出现（缺键 ⇒ 默认 1000 ms）
        sp["tau_slow_ms"] = float(args.tau_slow_ms)
    if getattr(args, "slow_inc", None) is not None:            # L2 杠杆：仅显式时出现（缺键 ⇒ 默认 6.0）
        sp["slow_inc"] = float(args.slow_inc)
    # ---- L4 机制键（C-8 / N1 / N3）：**一律只在显式给出时才进 spec**（缺键 ⇒ 走原路径）----
    if getattr(args, "mod", None) is not None:                 # C-8：--mod / --no-mod
        sp["mod"] = bool(args.mod)
    if getattr(args, "mod_ablation", False):                   # C-8 消融臂显式标记（免被 101 拒跑）
        sp["mod_ablation"] = True
    for _k in ("tau_mod_ms", "gain_lo", "gain_hi", "rate_ref_hz"):
        if getattr(args, _k, None) is not None:
            sp[_k] = float(getattr(args, _k))
    if getattr(args, "inh_gain", None) is not None:            # N1：逐类乘子（标量版，PV/SST/VIP）
        sp["inh_gain"] = float(args.inh_gain)
    if getattr(args, "gmax_band_gain", None):                  # N1：逐类乘子（dict 版）
        sp["gmax_band_gain"] = _parse_band_gain(args.gmax_band_gain)
    if getattr(args, "stp", None) is not None:                 # N3：--stp / --no-stp
        sp["stp"] = bool(args.stp)
    for _k in ("u0", "tau_fac_ms", "tau_rec_ms"):
        if getattr(args, _k, None) is not None:
            sp[_k] = float(getattr(args, _k))
    if getattr(args, "stp_mode", None) is not None:
        sp["stp_mode"] = str(args.stp_mode)
    return sp


def _parse_band_gain(txt) -> dict:
    """解析 `--gmax-band-gain`：接受 JSON dict 或 `PV=2.5,SST=2.5,VIP=2.5` 形式。"""
    if isinstance(txt, dict):
        return {str(k).upper(): float(v) for k, v in txt.items()}
    s = str(txt).strip()
    if s.startswith("{"):
        return {str(k).upper(): float(v) for k, v in json.loads(s).items()}
    out = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit("--gmax-band-gain 需形如 PV=2.5,SST=2.5,VIP=2.5 或 JSON dict：%r" % txt)
        k, v = part.split("=", 1)
        out[k.strip().upper()] = float(v)
    return out


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
    except ArmRefused as ex:
        # **"跑了也不算数"**：`.fail` 是本文件产物规范里"不计入分母、四态 invalid"的载体
        # ⇒ 拒跑臂与"跑失败"在盘上**可区分**（前者带 code=101/102/103 一行）。
        import traceback
        with open(fail, "w") as f:
            f.write("run_id=%s\nREFUSED code=%d\n%s\n%s\n"
                    % (run_id, ex.code, ex, traceback.format_exc()[-2000:]))
        print("[INVALID] %s  REFUSED code=%d: %s" % (run_id, ex.code, ex), flush=True)
        return int(ex.code)
    except Exception as ex:  # noqa: BLE001
        import traceback
        with open(fail, "w") as f:
            f.write("run_id=%s\n%s\n%s\n" % (run_id, ex, traceback.format_exc()[-2000:]))
        print("[FAIL] %s: %s" % (run_id, ex), flush=True)
        return 1
    with open(done, "w") as f:
        f.write("%s %s\n" % (run_id, arm["finished_at"]))
    # **零活动臂必须在 stdout 里显式可见**（否则主agent 只能从产物 JSON 才能区分，日志侧仍"静默"）
    # ⚠️ 非零活动臂的该行**逐字节不变**（`_ratio_txt` 对有限数就是原来的 `%.4f`；`ZERO_ACTIVITY` 后缀为空）
    try:
        _ratio_txt = "%.4f" % float(arm["ei"]["ei_ratio"])
    except (TypeError, ValueError):
        _ratio_txt = "null"          # `ei_ratio=None`（显式缺席）⇒ 不得让 `%.4f` 抛错
    _zero_txt = "  ZERO_ACTIVITY=1" if arm.get("zero_activity") is True else ""
    print("[DONE] %s  spk=%d  ei_ratio=%s  ms/step=%.4f  wall=%.2fs  RSS=%.0fMB%s" % (
        run_id, arm["agg"]["spk_total"], _ratio_txt, arm["ms_per_step_steady"],
        arm["wall_s"], arm["rss_mb"], _zero_txt), flush=True)
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
                  "net_seed", "delay_bins", "subtypes", "slow", "tau_slow_ms", "slow_inc",
                  # ---- L4 机制键（计划级继承；`if k in plan and k not in spec` ⇒ 计划不含则零影响）----
                  "mod", "mod_ablation", "tau_mod_ms", "gain_lo", "gain_hi", "rate_ref_hz",
                  "inh_gain", "gmax_band_gain",
                  "stp", "u0", "tau_fac_ms", "tau_rec_ms", "stp_mode"):
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
    ap.add_argument("--tau-slow-ms", type=float, default=None,
                    help="L2 慢变量时间常数（ms）。**不显式给 ⇒ 沿用既有默认 1000.0**（既有 G1/L0/L1 "
                         "结果全部在 slow 默认档下取得）")
    ap.add_argument("--slow-inc", type=float, default=None,
                    help="L2 每 spike 的慢电导增量。**不显式给 ⇒ 沿用既有默认 6.0**")
    ap.add_argument("--force", action="store_true", help="plan 模式：忽略已有 .done 重跑")
    # ---------- L4 机制键（R153 §⑤ D2 批准：C-8 → N1 → N3；全部**不显式给就不进 spec**）----------
    ap.add_argument("--mod", dest="mod", action="store_true", default=None,
                    help="C-8 慢增益变量**启用**（显式传 `slow=True`，M10 M-4 的防线）。"
                         "**不给则该键不进 spec**（既有臂字节不变）")
    ap.add_argument("--no-mod", dest="mod", action="store_false", default=None,
                    help="C-8 显式关闭（消融臂 S-11）；给参数而关机制会被拒跑（EXIT=101）")
    ap.add_argument("--mod-ablation", action="store_true",
                    help="显式承认该臂是 C-8 **消融臂**（允许同时携带调质参数而不接线；"
                         "落盘 mod_params_ignored）")
    ap.add_argument("--tau-mod-ms", type=float, default=None, help="C-8 慢增益时间常数（ms）")
    ap.add_argument("--gain-lo", type=float, default=None, help="C-8 m=0 端增益；gain_hi<gain_lo ⇒ 负反馈")
    ap.add_argument("--gain-hi", type=float, default=None, help="C-8 m=1 端增益；gain_hi>gain_lo ⇒ 正反馈")
    ap.add_argument("--rate-ref-hz", type=float, default=None,
                    help="C-8 归一化参考率（Hz）：target = 瞬时群体率 / rate_ref_hz")
    ap.add_argument("--inh-gain", type=float, default=None,
                    help="N1 类级抑制增益（逐类乘子标量版：乘 PV/SST/VIP，E 段不动）")
    ap.add_argument("--gmax-band-gain", default=None,
                    help="N1 逐类乘子（dict 版）：`PV=2.5,SST=2.5,VIP=2.5` 或 JSON dict")
    ap.add_argument("--stp", dest="stp", action="store_true", default=None,
                    help="N3 STP 启用（**只有 mode='depress' 被接受**，否则拒跑 EXIT=102）")
    ap.add_argument("--no-stp", dest="stp", action="store_false", default=None,
                    help="N3 显式关闭（消融臂；会落盘 stp_enabled=false + stp_params_ignored）")
    ap.add_argument("--u0", type=float, default=None, help="N3 STP 初始利用度（引擎默认 0.5）")
    ap.add_argument("--tau-fac-ms", type=float, default=None,
                    help="N3 STP 易化时常数（ms）。🔴 **不许下调**：默认 50 ms 在 M11 低率区间"
                         "预测 ≈ 递归隔离消融（F-3/P1）⇒ 只许上调到 {2000, 5000} 档")
    ap.add_argument("--tau-rec-ms", type=float, default=None, help="N3 STP 恢复时常数（ms，默认 500）")
    ap.add_argument("--stp-mode", default=None, choices=(None, "depress", "facilitate"),
                    help="N3 STP 工作模式。**预注册 F-3：mode != 'depress' ⇒ invalid**（拒跑 EXIT=102）")
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
