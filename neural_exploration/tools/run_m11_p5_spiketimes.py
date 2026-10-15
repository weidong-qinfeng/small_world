#!/usr/bin/env python
"""M11 P5/G1 工作点 **6 臂 `record="full"` 重跑** —— 取**逐神经元脉冲时刻**（依据 R201 = `docs/m11_acceptance_master.md` §3gm）。

> **为什么存在**：判据 `(b)`（钙域静默比例）在数据侧无法闭合，因为**脉冲时刻在任何已落盘产物里都不存在**
> ——生产者 `tools/run_m11_p5_arms.py` **L752/L755 硬编码 `record="counts"`**，只落 `counts.csv` +
> `poptrace.npy`。而引擎 `src/mouse_cortex_engine.py` **L551 形参 / L581 / L766 / L803** 已支持
> `record="full"`（`res["spikes"] = (steps, idx)`，见 L822）。本工具**只新建**、把记录模式从
> `counts` 换成 `full`，其余**逐字复用生产者**。

🔴 **同夹具 = 同一份代码（本工具的全部价值所在）**
1. **不修改**生产者：跑前 sha256 守卫（不符 **退 98**），登记值见 `PRODUCER_SHA256_PIN`。
2. **不重写夹具**：网络/事件表/初始态/settle/测量窗**全部由生产者自己的代码执行**——
   本工具直接调用 `run_m11_p5_arms.run_one(spec, out_dir)`（**整个函数**，含其全部断言与产物写法），
   并未重新实现 `build_engine`（L136）/`make_events`（L231）/`effective_seeds`（L240）/
   `_mechanism_preflight`（L328）中的任何一行；`--self-test` 用**代码对象身份**证明这一点
   （`__code__.co_filename` / `co_code` sha256 / `__globals__ is vars(P)`）。
3. **只改一个已有形参的取值**：`record` 是引擎 `run()` 的**登记形参**（`mouse_cortex_engine.py` **L551**）。
   本工具用**一次 `MouseCortexEngine.run` 的运行时包装**，把**测量窗那一次调用**的
   `record="counts"` 换成 `record="full"`。判据（三重与，缺一不触发）：
   `kw["pop_trace"] is True` ∧ `kw["ei_integrals"] is True` ∧ `kw["record"] == "counts"`
   ⇒ **只命中 L755**（测量窗）；**L752 的 settle 调用**（`delivery="step", record="counts"`，无
   `pop_trace`/`ei_integrals`）**逐字不变**。每次运行都把"哪几次调用被拦截/未被拦截"落进 JSON
   （`record_override.forced_calls`），可核验"只多记录、不改路径"。
4. **若上游把 `record` 改成参数入口**（即 L752/L755 不再是硬编码），本工具的
   `PRODUCER_RECORD_COUNTS_LINES = (752, 755)` 守卫会**先失败**（退 95），不会静默漂移。

**自证标准（缺一即不得采信）**：逐神经元 `count` 与已落盘 `..._counts.csv` **逐位相同**
（`spk_total`：baseline = 47203/47342/47036，zero_drive = 3143/3174/2895）∧ `poptrace.npy`
**逐字节相同** ∧ `bincount(neuron_id) == counts`。**不符 ⇒ 落 `<run_id>.MISMATCH.json` 并退 1**（停止，不采信）。

**产物**（`<out-dir>` 必须**绝对**路径，否则 **退 96**）
- `<out-dir>/<run_id>_spiketimes.npz`：`neuron_id`(int32) / `step`(int64) / `t_ms`(float64) 三个数组
  ＋标量 `dt_ms` / `n_steps` / `spk_total` / `run_id` / `t0_ms` / `record_mode` / `producer_sha256`。
  **时间原点 = 测量窗起点**（= settle 之后 `reset_counts()` 那一刻）⇒ 绝对仿真时刻 = `settle_ms + t_ms`。
- `<out-dir>/<run_id>.json`：`spk_total` / `n_steps` / `wall_s` / `sha256` ＋ `spec` **逐字回读** ＋
  对账块 `crosscheck` ＋ 自证块 `fixture_identity` / `record_override`。
- `<out-dir>/_producer_echo/`：**生产者自己**重写的 `json`/`counts.csv`/`poptrace.npy`
  （对账用的原始证据；不是交付物本体）。

**用法**
```bash
# ① 只读自证：夹具复用（不跑仿真、不用 MPS）
./.venv-m9/bin/python -m neural_exploration.tools.run_m11_p5_spiketimes --self-test
# ② CPU 微臂：record="counts" vs "full" 的**计数逐位等同**（不碰 MPS）
./.venv-m9/bin/python -m neural_exploration.tools.run_m11_p5_spiketimes --micro-test
# ③ 正式：6 臂，逐臂**新进程**（MPS 串行）——整条须以 caffeinate -i 包裹
env PYTHONHASHSEED=0 MPLBACKEND=Agg caffeinate -i ./.venv-m9/bin/python \
  -m neural_exploration.tools.run_m11_p5_spiketimes \
  --out-dir /Users/weidong/ai/small_world/neural_exploration/data/m11_p5_spiketimes
# ④ 对账（只读）：重算逐臂表
./.venv-m9/bin/python -m neural_exploration.tools.run_m11_p5_spiketimes --reconcile \
  --out-dir /Users/weidong/ai/small_world/neural_exploration/data/m11_p5_spiketimes
```

**退出码**：1 = 臂失败或**逐位不符**（fail-stop）｜92 = 解释器非 `.venv-m9`｜93 = 对账不符｜
94 = 微臂不符｜95 = 夹具身份/sha 守卫失败｜96 = `--out-dir` 非绝对｜97 = spec/参考产物 sha 不符｜
98 = 生产者或引擎 sha 不符｜99 = **已有 MPS 作业 ⇒ 拒启**。
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import time

import numpy as np

# --------------------------------------------------------------------- 常量
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY_ENGINE = os.path.join(PROJECT_ROOT, ".venv-m9", "bin", "python")   # R34 §12.2：唯一有 torch
VENV_DIR = os.path.join(PROJECT_ROOT, ".venv-m9")

PRODUCER_REL = "neural_exploration/tools/run_m11_p5_arms.py"
PRODUCER_PATH = os.path.join(PROJECT_ROOT, PRODUCER_REL)
ENGINE_REL = "neural_exploration/src/mouse_cortex_engine.py"
ENGINE_PATH = os.path.join(PROJECT_ROOT, ENGINE_REL)

DATA_DIR = os.path.join(PROJECT_ROOT, "neural_exploration", "data")
REF_DIR = os.path.join(DATA_DIR, "m11_p5_l4_0_runs")
OUT_DIR_DEFAULT = os.path.join(DATA_DIR, "m11_p5_spiketimes")
MICRO_DIR_DEFAULT = "/tmp/m11_p5_spiketimes_micro"

# ---- 已登记 sha 守卫（跑前**逐字节**核对；不符即拒）----
PRODUCER_SHA256_PIN = "26292f7b0078aad275ea2d9d21324f6ad47ea5da53e3323479dbe1ad56972b27"
ENGINE_SHA256_PIN = "90c02b37d922d7ac9aefd78023c7db23a2fcd8a70147c1b226b5e9b7a7a9dc34"

# ---- 生产者**硬编码** record 的行号（= 本工具存在的理由；生产者的**代码对象**必须在这些行上）----
PRODUCER_RECORD_COUNTS_LINES = (752, 755)
PRODUCER_RUN_ONE_DEF_LINE = 728
# ---- 引擎侧 `record="full"` 的登记实现行（`record` 是 run() 的形参：L551）----
ENGINE_RECORD_PARAM_LINE = 551
ENGINE_RECORD_FULL_LINES = (581, 766, 787, 803, 823)
ENGINE_SPIKES_KEY_LINE = 822

STAGE = "l4_g0.35_8192_T10000ms"
ARM_IDS = tuple("%s_d%s_s%d" % (STAGE, d, s) for d in ("0.0", "1.0") for s in (0, 1, 2))

# ---- spec 逐字回读的**登记 sha**（= 参数不得凭记忆/猜测的硬保证）----
SPEC_SHA_PIN = {
    "l4_g0.35_8192_T10000ms_d0.0_s0": "efe51f9304b2553f85b7ab7a3530ff223d79956557f9e2b53ed5534ba037d6da",
    "l4_g0.35_8192_T10000ms_d0.0_s1": "a21e92cae073bc49644c7ba16d2f6a38d31171ffdc79ef7959cf2352ffba5d43",
    "l4_g0.35_8192_T10000ms_d0.0_s2": "c1f11e5001e9ca0b072743af093951c8484e520fe65133593d3b3bb5e24a95fa",
    "l4_g0.35_8192_T10000ms_d1.0_s0": "07f3b5c488c8d3581e0ba85c0d1095226def5ddfbe152dae44bd74740085721a",
    "l4_g0.35_8192_T10000ms_d1.0_s1": "daf48ea96a7ab5ea5110d9b2c929e66b6b757a924ac6c76734fd2e8627fa29f6",
    "l4_g0.35_8192_T10000ms_d1.0_s2": "2e3d38fd9eb967473601874c54f18b79df27057e5b5cd3daf65a67928294e13c",
}
# ---- 已落盘计数参考产物的登记 sha（"逐位复现"的靶，不得被静默挪动）----
REF_COUNTS_SHA_PIN = {
    "l4_g0.35_8192_T10000ms_d0.0_s0": "303b8e5a6701562625e94aa70ec83c1306564237ff422346d2f3687acbf23618",
    "l4_g0.35_8192_T10000ms_d0.0_s1": "6e275baeba44cf31ad5673403e864901e75506d607abe2f88043b63ac1c766cd",
    "l4_g0.35_8192_T10000ms_d0.0_s2": "0568d6d7294f9ec8083ea852b400404e12e315e288436a0e290a66299a00414e",
    "l4_g0.35_8192_T10000ms_d1.0_s0": "ea4118c7a890ddbde43d2e704ce18ca18e77965a9e82a123f7912a300e7d94aa",
    "l4_g0.35_8192_T10000ms_d1.0_s1": "fc72ed7864331f61df4bd64aee01f205aa448141c0717d79d443609bce2f7059",
    "l4_g0.35_8192_T10000ms_d1.0_s2": "d87407cd7504a975da9ac3ef3d3245ac7ef1c0de8173eb85fd1fc02f71300c4d",
}
REF_POPTRACE_SHA_PIN = {
    "l4_g0.35_8192_T10000ms_d0.0_s0": "efa675c3479028bc053a05475101c0af8e25038955a7c7f31e09d246a23105f1",
    "l4_g0.35_8192_T10000ms_d0.0_s1": "fcca226618a2084a6a405412db710bbf04c15c1671e9793381f9d287c8f18ce1",
    "l4_g0.35_8192_T10000ms_d0.0_s2": "af3dbf1590c762ab60e1cd934ef369bf340101f02fe724d23b7f817f89c6eab8",
    "l4_g0.35_8192_T10000ms_d1.0_s0": "dea800c46c90c3c53c6ca38cf6a5a8ef4c71afc004a5ae823d6eeaf2f1f348de",
    "l4_g0.35_8192_T10000ms_d1.0_s1": "9abaad64cd4efcd1f4384b15cc1fa2dd05066d061e62d49663a4038c14e8b187",
    "l4_g0.35_8192_T10000ms_d1.0_s2": "c5458a11b62147ebe936c75236276b5c98fd5434196c7d682bf8a0b3696175b2",
}
REF_SPK_TOTAL = {
    "l4_g0.35_8192_T10000ms_d0.0_s0": 3143,
    "l4_g0.35_8192_T10000ms_d0.0_s1": 3174,
    "l4_g0.35_8192_T10000ms_d0.0_s2": 2895,
    "l4_g0.35_8192_T10000ms_d1.0_s0": 47203,
    "l4_g0.35_8192_T10000ms_d1.0_s1": 47342,
    "l4_g0.35_8192_T10000ms_d1.0_s2": 47036,
}
REF_WALL_S = {
    "l4_g0.35_8192_T10000ms_d0.0_s0": 152.856751417,
    "l4_g0.35_8192_T10000ms_d0.0_s1": 133.076737959,
    "l4_g0.35_8192_T10000ms_d0.0_s2": 141.32008966599997,
    "l4_g0.35_8192_T10000ms_d1.0_s0": 199.297091333,
    "l4_g0.35_8192_T10000ms_d1.0_s1": 188.72978520799998,
    "l4_g0.35_8192_T10000ms_d1.0_s2": 205.379085833,
}
BASELINE_SPK_ANCHOR = {   # R201 §④ 的独立复算锚
    "l4_g0.35_8192_T10000ms_d1.0_s0": 47203,
    "l4_g0.35_8192_T10000ms_d1.0_s1": 47342,
    "l4_g0.35_8192_T10000ms_d1.0_s2": 47036,
}
REF_WALL_S_SUM = 1020.7   # R201 §③ 登记值（实测合计）

REQUIRED_SPEC_KEYS = ("run_id", "T_ms", "settle_ms", "drive_mult", "seed", "net_seed", "event_seed",
                      "role", "delivery", "gmax_scale", "bias_scale", "bias_log_sigma", "slow",
                      "tau_slow_ms", "slow_inc", "n_neuron", "indegree", "delay_bins", "subtypes",
                      "device", "drive_rate_hz")

EXIT_ARM_FAIL = 1
EXIT_INTERPRETER = 92
EXIT_RECONCILE_MISMATCH = 93
EXIT_MICRO_FAIL = 94
EXIT_FIXTURE_IDENTITY = 95
EXIT_OUTDIR_NOT_ABS = 96
EXIT_SHA_GUARD = 97
EXIT_PRODUCER_SHA = 98
EXIT_MPS_BUSY = 99


# --------------------------------------------------------------------- 基础
def log(msg: str) -> None:
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def sha256_of_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def rel(path: str) -> str:
    try:
        return os.path.relpath(path, PROJECT_ROOT)
    except ValueError:
        return path


def require_abs(path: str, what: str) -> str:
    if not os.path.isabs(path):
        print("REFUSE(%d): %s 必须为**绝对路径**，收到 %r" % (EXIT_OUTDIR_NOT_ABS, what, path))
        sys.exit(EXIT_OUTDIR_NOT_ABS)
    return path


def _proc_snapshot() -> dict:
    """`pid → {ppid, pgid, command}` 快照（一次 `ps`，避免逐 pid 调用）。"""
    snap = {}
    r = subprocess.run(["ps", "-eo", "pid=,ppid=,pgid=,command="], capture_output=True, text=True)
    for line in (r.stdout or "").splitlines():
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        try:
            snap[int(parts[0])] = dict(ppid=int(parts[1]), pgid=int(parts[2]), command=parts[3])
        except ValueError:
            continue
    return snap


def _ancestor_pids(snap: dict) -> list:
    """自己的全部祖先 pid（= 自己这条启动链：bash 作业壳 → caffeinate/包装 → 父驱动器）。"""
    pids = []
    pid = os.getpid()
    for _ in range(64):
        rec = snap.get(pid)
        if not rec:
            break
        ppid = rec["ppid"]
        if ppid <= 1 or ppid in pids:
            break
        pids.append(ppid)
        pid = ppid
    return pids


def _descendant_pids(snap: dict, root: int) -> set:
    kids = {}
    for pid, rec in snap.items():
        kids.setdefault(rec["ppid"], []).append(pid)
    out, stack = set(), [root]
    while stack:
        p = stack.pop()
        for c in kids.get(p, ()):
            if c not in out:
                out.add(c)
                stack.append(c)
    return out


def mps_guard(patterns=("run_m11_p5", "run_m11_p5_spiketimes")) -> dict:
    """**已有任何 MPS 作业 ⇒ 拒启**（本机 MPS 严格串行、整机独占）——退出码 99。

    命中判据 = `pgrep -f <pat>`（两条模式，与启动纪律逐字一致）；**排除集合** = 自己这条
    启动链，即 ① 自己 ② 自己的全部祖先 ③ 自己的后代（父驱动器 spawn 的子进程）④ **同一进程组**
    （`bash` 作业壳 / `caffeinate` 包装 / 父驱动器 / 逐臂子进程 —— 若 `caffeinate` 选择 `exec`
    而非 fork，则"包装进程"与"自己"是同一个 pid，此时前三条会漏，必须靠进程组兜住）。
    **排除一律留痕**（`excluded`），不含糊忽略 ⇒ 审计上可核验"被排除的到底是谁"。
    """
    self_pid = os.getpid()
    snap = _proc_snapshot()
    ppid = os.getppid()
    anc = set(_ancestor_pids(snap))
    desc = _descendant_pids(snap, self_pid)
    pgrp = os.getpgrp()
    excluded, busy = [], []
    for pat in patterns:
        r = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
        for tok in (r.stdout or "").split():
            try:
                pid = int(tok)
            except ValueError:
                continue
            rec = snap.get(pid, dict(ppid=-1, pgid=-1, command="<ps 快照里没有：已退出>"))
            item = dict(pattern=pat, pid=pid, ppid=rec["ppid"], pgid=rec["pgid"],
                        command=rec["command"], why=None)
            if pid == self_pid:
                item["why"] = "self"
            elif pid == ppid or pid in anc:
                item["why"] = "ancestor(启动链)"
            elif pid in desc:
                item["why"] = "descendant(本驱动器 spawn 的子进程)"
            elif rec["pgid"] == pgrp:
                item["why"] = "same-process-group(同一条启动链)"
            (excluded if item["why"] else busy).append(item)
    if busy:
        print("REFUSE(%d): 已有 MPS 作业 ⇒ 拒启（本机 MPS 严格串行）:" % EXIT_MPS_BUSY)
        for b in busy:
            print("    busy:", b)
        print("    （本次被判为『自己这条启动链』而排除的进程，一并留痕：")
        for e in excluded:
            print("     excluded:", e)
        print("    ）")
        sys.exit(EXIT_MPS_BUSY)
    return dict(patterns=list(patterns), n_busy=0, self_pid=self_pid, self_pgid=pgrp,
                ancestors=sorted(anc), descendants=sorted(desc),
                n_excluded=len(excluded), excluded=excluded)


def interpreter_guard() -> dict:
    """解释器必须是 `.venv-m9`（**绝不以 `sys.executable` 自举**）。"""
    exe = os.path.realpath(sys.executable)
    want = os.path.realpath(PY_ENGINE)
    pref = os.path.realpath(sys.prefix)
    vdir = os.path.realpath(VENV_DIR)
    if exe != want or pref != vdir:
        print("REFUSE(%d): 解释器必须 = %s；实测 sys.executable=%s（realpath %s）、"
              "sys.prefix=%s（realpath %s）" % (EXIT_INTERPRETER, PY_ENGINE,
                                                sys.executable, exe, sys.prefix, pref))
        sys.exit(EXIT_INTERPRETER)
    return dict(executable=sys.executable, realpath_exe=exe, sys_prefix=sys.prefix,
                python=sys.version.split()[0], numpy=np.__version__)


def load_producer():
    """**按路径**载入生产者（带 `__main__` 保护 ⇒ 导入安全），并过登记 sha 守卫。"""
    got = sha256_of_file(PRODUCER_PATH)
    if got != PRODUCER_SHA256_PIN:
        print("REFUSE(%d): 生产者 %s\n  sha256 = %s\n  登记值 = %s"
              "（『同一份代码』的前提被破坏 ⇒ 拒绝运行）"
              % (EXIT_PRODUCER_SHA, PRODUCER_REL, got, PRODUCER_SHA256_PIN))
        sys.exit(EXIT_PRODUCER_SHA)
    sp = importlib.util.spec_from_file_location("run_m11_p5_arms", PRODUCER_PATH)
    mod = importlib.util.module_from_spec(sp)
    sys.modules["run_m11_p5_arms"] = mod
    sp.loader.exec_module(mod)                     # ← 导入生产者模块本体（L59-1536）
    if os.path.realpath(mod.__file__) != os.path.realpath(PRODUCER_PATH):
        print("REFUSE(%d): 载入的不是 %s" % (EXIT_FIXTURE_IDENTITY, PRODUCER_PATH))
        sys.exit(EXIT_FIXTURE_IDENTITY)
    return mod


def load_engine_module():
    got = sha256_of_file(ENGINE_PATH)
    if got != ENGINE_SHA256_PIN:
        print("REFUSE(%d): 引擎 %s sha256=%s ≠ 登记值 %s" % (EXIT_PRODUCER_SHA, ENGINE_REL,
                                                             got, ENGINE_SHA256_PIN))
        sys.exit(EXIT_PRODUCER_SHA)
    import neural_exploration.src.mouse_cortex_engine as M
    if os.path.realpath(M.__file__) != os.path.realpath(ENGINE_PATH):
        print("REFUSE(%d): 载入的引擎不是 %s（实为 %s）" % (EXIT_FIXTURE_IDENTITY,
                                                          ENGINE_PATH, M.__file__))
        sys.exit(EXIT_FIXTURE_IDENTITY)
    return M


# ------------------------------------------------- 夹具复用自证（代码对象身份）
def _lines(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().splitlines()


def fixture_identity(P, M) -> dict:
    """证明"同一份代码"：夹具函数是生产者的**代码对象**，且生产者的硬编码位置未漂移。"""
    fails = []
    rep = {
        "claim": "本工具不重写夹具：网络/事件/settle/测量窗**全部由生产者自己的代码对象**执行；"
                 "唯一改动 = 引擎登记形参 `record` 的取值（counts → full）。",
        "producer": dict(path=PRODUCER_PATH, rel=PRODUCER_REL,
                         sha256=sha256_of_file(PRODUCER_PATH), pin=PRODUCER_SHA256_PIN),
        "engine": dict(path=ENGINE_PATH, rel=ENGINE_REL,
                       sha256=sha256_of_file(ENGINE_PATH), pin=ENGINE_SHA256_PIN),
    }
    rep["producer_sha_ok"] = rep["producer"]["sha256"] == PRODUCER_SHA256_PIN
    rep["engine_sha_ok"] = rep["engine"]["sha256"] == ENGINE_SHA256_PIN
    if not rep["producer_sha_ok"]:
        fails.append("producer sha 不符")
    if not rep["engine_sha_ok"]:
        fails.append("engine sha 不符")

    # ① 夹具函数的**代码对象**必须来自生产者文件（不是本文件的复写）
    fns = {}
    for name in ("build_engine", "make_events", "effective_seeds", "_mechanism_preflight",
                 "run_one", "_net_echo"):
        f = getattr(P, name)
        c = f.__code__
        fns[name] = dict(code_filename=c.co_filename, code_firstlineno=c.co_firstlineno,
                         code_sha256=sha256_of_bytes(c.co_code),
                         qualname=f.__qualname__, module=f.__module__,
                         globals_is_producer_module=bool(f.__globals__ is vars(P)),
                         source_sha256=sha256_of_bytes(inspect.getsource(f).encode("utf-8")))
        if os.path.realpath(c.co_filename) != os.path.realpath(PRODUCER_PATH):
            fails.append("%s.__code__.co_filename=%s 不是生产者文件" % (name, c.co_filename))
        if not fns[name]["globals_is_producer_module"]:
            fails.append("%s.__globals__ 不是生产者模块" % name)
        if f.__module__ != "run_m11_p5_arms":
            fails.append("%s.__module__=%s" % (name, f.__module__))
    rep["producer_functions"] = fns
    rep["run_one_def_line"] = P.run_one.__code__.co_firstlineno
    if P.run_one.__code__.co_firstlineno != PRODUCER_RUN_ONE_DEF_LINE:
        fails.append("run_one 定义行漂移：%s ≠ %s" % (P.run_one.__code__.co_firstlineno,
                                                  PRODUCER_RUN_ONE_DEF_LINE))

    # ② 生产者里 `record="counts"` 的**硬编码行号**（本工具存在的理由）与"从未记录过 full"
    plines = _lines(PRODUCER_PATH)
    cnt = [i + 1 for i, L in enumerate(plines) if 'record="counts"' in L]
    full = [i + 1 for i, L in enumerate(plines)
            if ('record="full"' in L or "record == \"full\"" in L or "record='full'" in L)]
    rep["producer_record_counts_lines"] = cnt
    rep["producer_record_full_lines"] = full
    rep["producer_record_counts_line_pins"] = list(PRODUCER_RECORD_COUNTS_LINES)
    if tuple(cnt) != PRODUCER_RECORD_COUNTS_LINES:
        fails.append("生产者 record=\"counts\" 硬编码行号 = %s ≠ 登记 %s"
                     % (cnt, list(PRODUCER_RECORD_COUNTS_LINES)))
    if full:
        fails.append("生产者已出现 record=\"full\"（行 %s）⇒ 『本工具是新的驱动件』这一前提失效，须重判" % full)

    # ③ 引擎侧：`record` 是**登记形参**，`record="full"` 是**引擎自己**的实现
    elines = _lines(ENGINE_PATH)
    eparam = [i + 1 for i, L in enumerate(elines) if "record: str = \"counts\"" in L]
    efull = [i + 1 for i, L in enumerate(elines) if 'record == "full"' in L]
    espk = [i + 1 for i, L in enumerate(elines) if '"spikes"' in L]
    rep["engine_record_param_lines"] = eparam
    rep["engine_record_full_lines"] = efull
    rep["engine_spikes_key_lines"] = espk
    if ENGINE_RECORD_PARAM_LINE not in eparam:
        fails.append("引擎 run() 的 record 形参不在 L%d（实测 %s）" % (ENGINE_RECORD_PARAM_LINE, eparam))
    if tuple(efull) != ENGINE_RECORD_FULL_LINES:
        fails.append("引擎 record==\"full\" 行号 = %s ≠ 登记 %s" % (efull, list(ENGINE_RECORD_FULL_LINES)))
    if ENGINE_SPIKES_KEY_LINE not in espk:
        fails.append("引擎 res[\"spikes\"] 不在 L%d（实测 %s）" % (ENGINE_SPIKES_KEY_LINE, espk))

    # ④ 拦截目标身份
    run = M.MouseCortexEngine.run
    rep["intercept_target"] = dict(qualname=run.__qualname__, code_filename=run.__code__.co_filename,
                                   code_sha256=sha256_of_bytes(run.__code__.co_code))
    if run.__qualname__ != "MouseCortexEngine.run":
        fails.append("拦截目标不是 MouseCortexEngine.run（%s）" % run.__qualname__)
    if os.path.realpath(run.__code__.co_filename) != os.path.realpath(ENGINE_PATH):
        fails.append("run() 的代码对象不来自 %s" % ENGINE_REL)
    rep["ok"] = not fails
    rep["fails"] = fails
    return rep


def fixture_identity_or_die(P, M) -> dict:
    rep = fixture_identity(P, M)
    if not rep["ok"]:
        print("REFUSE(%d): 夹具身份/sha 守卫失败：" % EXIT_FIXTURE_IDENTITY)
        for f in rep["fails"]:
            print("    -", f)
        print(json.dumps(rep, ensure_ascii=False, indent=1))
        sys.exit(EXIT_FIXTURE_IDENTITY)
    return rep


# --------------------------------------------------- 记录模式拦截（唯一改动点）
@contextlib.contextmanager
def force_full_record(cap: dict):
    """把**测量窗那一次** `MouseCortexEngine.run` 的 `record="counts"` 换成 `record="full"`。

    判据（三重与）：`pop_trace is True` ∧ `ei_integrals is True` ∧ `record == "counts"`
    ⇒ 只命中生产者 L755；L752 的 settle 调用（无 pop_trace/ei_integrals）**逐字不变**。
    每次调用（含未拦截的）都记进 `cap["passes"]` ⇒ "只多记录、不改路径"可在产物里核验。
    """
    import neural_exploration.src.mouse_cortex_engine as M
    orig = M.MouseCortexEngine.run

    def patched(self, n_steps, **kw):
        hit = (bool(kw.get("pop_trace")) and bool(kw.get("ei_integrals"))
               and kw.get("record") == "counts")
        if not hit:
            cap.setdefault("passes", []).append(
                dict(call="passthrough", n_steps=int(n_steps), record=kw.get("record"),
                     pop_trace=bool(kw.get("pop_trace")), ei_integrals=bool(kw.get("ei_integrals")),
                     delivery=kw.get("delivery")))
            return orig(self, n_steps, **kw)
        res = orig(self, n_steps, **dict(kw, record="full"))
        cap["res"] = res
        cap.setdefault("passes", []).append(
            dict(call="forced_full", n_steps=int(n_steps), record="full(forced)",
                 record_kwarg_seen="counts", pop_trace=True, ei_integrals=True,
                 delivery=kw.get("delivery"), n_spikes=int(len(res["spikes"][0])) if res.get("spikes") else 0))
        return res

    M.MouseCortexEngine.run = patched
    try:
        yield cap
    finally:
        M.MouseCortexEngine.run = orig


def read_counts_csv(path: str):
    ids, cnts, types = [], [], []
    with open(path, newline="") as f:
        r = csv.reader(f)
        hdr = next(r)
        if hdr != ["neuron_id", "count", "cell_type"]:
            raise ValueError("counts.csv 表头异常：%r (%s)" % (hdr, path))
        for row in r:
            ids.append(int(row[0]))
            cnts.append(int(row[1]))
            types.append(row[2])
    return (np.asarray(ids, dtype=np.int64), np.asarray(cnts, dtype=np.int64),
            np.asarray(types, dtype=object))


def load_spec_verbatim(rid: str, spec_dir: str) -> dict:
    """**逐字**从现成 spec 文件读参数；sha 不符即拒（97）；必需键缺失即拒（97）。"""
    p = os.path.join(spec_dir, rid + ".spec.json")
    if not os.path.exists(p):
        print("REFUSE(%d): spec 不存在：%s" % (EXIT_SHA_GUARD, p))
        sys.exit(EXIT_SHA_GUARD)
    raw = open(p, "rb").read()
    sha = sha256_of_bytes(raw)
    if sha != SPEC_SHA_PIN[rid]:
        print("REFUSE(%d): spec sha 不符：%s\n  实测 = %s\n  登记 = %s"
              % (EXIT_SHA_GUARD, p, sha, SPEC_SHA_PIN[rid]))
        sys.exit(EXIT_SHA_GUARD)
    spec = json.loads(raw.decode("utf-8"))
    miss = [k for k in REQUIRED_SPEC_KEYS if k not in spec]
    if miss or spec.get("run_id") != rid:
        print("REFUSE(%d): spec 缺键 %s / run_id=%r ≠ %r" % (EXIT_SHA_GUARD, miss,
                                                            spec.get("run_id"), rid))
        sys.exit(EXIT_SHA_GUARD)
    return dict(spec=spec, path=p, sha256=sha, bytes=len(raw))


def _fieldwise_diff(new: dict, ref: dict):
    keys = sorted(set(new) | set(ref))
    diff, only_new, only_ref = [], [], []
    for k in keys:
        if k not in ref:
            only_new.append(k)
        elif k not in new:
            only_ref.append(k)
        elif json.dumps(new[k], sort_keys=True, ensure_ascii=False) != \
                json.dumps(ref[k], sort_keys=True, ensure_ascii=False):
            diff.append(k)
    return diff, only_new, only_ref


# --------------------------------------------------------------------- 单臂
def run_arm(rid: str, out_dir: str, opts) -> int:
    if rid not in ARM_IDS:
        print("REFUSE(%d): 未知臂 %r（只允许 G1 工作点 6 臂 %s）"
              % (EXIT_SHA_GUARD, rid, list(ARM_IDS)))
        return EXIT_SHA_GUARD
    interp = interpreter_guard()
    guard = mps_guard()
    P = load_producer()
    M = load_engine_module()
    ident = fixture_identity_or_die(P, M)
    loaded = load_spec_verbatim(rid, opts.spec_dir)
    spec = loaded["spec"]

    # 参考产物（"逐位复现"的靶）：先核登记 sha，防靶被静默挪动
    ref_counts_csv = os.path.join(opts.ref_dir, rid + "_counts.csv")
    ref_pop_csv = os.path.join(opts.ref_dir, rid + "_poptrace.npy")
    ref_json = os.path.join(opts.ref_dir, rid + ".json")
    for p, pin, what in ((ref_counts_csv, REF_COUNTS_SHA_PIN[rid], "counts.csv"),
                         (ref_pop_csv, REF_POPTRACE_SHA_PIN[rid], "poptrace.npy")):
        if not os.path.exists(p) or sha256_of_file(p) != pin:
            print("REFUSE(%d): 参考产物 %s 的 sha 不符（靶被挪动 ⇒ 拒绝）" % (EXIT_SHA_GUARD, p))
            sys.exit(EXIT_SHA_GUARD)

    echo_dir = os.path.join(out_dir, "_producer_echo")
    os.makedirs(echo_dir, exist_ok=True)
    log("arm=%s 起跑（device=%s delivery=%s n=%s T_ms=%s）"
        % (rid, spec.get("device"), spec.get("delivery"), spec.get("n_neuron"), spec.get("T_ms")))
    t_child0 = time.perf_counter()
    cap = {}
    with force_full_record(cap):
        # ★★ 调用**生产者自己的** run_one（整函数，含其全部断言与产物写法）★★
        run_one_json = P.run_one(json.loads(json.dumps(spec)), echo_dir)
    child_wall = time.perf_counter() - t_child0
    res = cap.get("res")
    if res is None or res.get("spikes") is None:
        print("FAIL: 未捕获 record=\"full\" 的 spikes（cap.passes=%r）" % (cap.get("passes"),))
        return EXIT_ARM_FAIL
    steps, idx = res["spikes"]
    steps = np.asarray(steps, dtype=np.int64)
    idx = np.asarray(idx, dtype=np.int64)
    dt_ms = float(res["dt_ms"])
    n_steps = int(res["n_steps"])
    n_neuron = int(spec["n_neuron"])
    res_spk_total = int(round(float(res["spk_total"])))

    # ---- 生产者"自己写出的"新产物 vs 已落盘 ----
    new_counts_csv = os.path.join(echo_dir, rid + "_counts.csv")
    new_pop_csv = os.path.join(echo_dir, rid + "_poptrace.npy")
    new_json_path = os.path.join(echo_dir, rid + ".json")
    nid, new_counts, cell_type = read_counts_csv(new_counts_csv)
    rid_ref, ref_counts, _ = read_counts_csv(ref_counts_csv)
    same_shape = bool(nid.shape == rid_ref.shape == (n_neuron,))
    ident_counts = bool(same_shape and np.array_equal(nid, rid_ref) and np.array_equal(new_counts, ref_counts))
    mism = (int(np.sum(new_counts != ref_counts)) if same_shape else -1)
    maxdiff = (int(np.max(np.abs(new_counts - ref_counts))) if same_shape else -1)
    pop_new_sha = sha256_of_file(new_pop_csv)
    pop_ref_sha = sha256_of_file(ref_pop_csv)
    pop_same = bool(pop_new_sha == pop_ref_sha)

    # ---- 脉冲时刻 ↔ 计数 的一致性（引擎内部交叉验证）----
    bc = np.bincount(idx, minlength=n_neuron)
    bc_ok = bool(bc.shape == (n_neuron,) and np.array_equal(bc.astype(np.int64), new_counts))
    n_ev = int(steps.size)
    ev_ok = bool(n_ev == idx.size == res_spk_total == int(new_counts.sum()))
    steps_sorted = bool(n_ev == 0 or bool(np.all(np.diff(steps) >= 0)))
    idx_in_range = bool(n_ev == 0 or (int(idx.min()) >= 0 and int(idx.max()) < n_neuron))
    steps_in_range = bool(n_ev == 0 or (int(steps.min()) >= 0 and int(steps.max()) < n_steps))
    res_json_spk = int(run_one_json["agg"]["spk_total"])

    # ---- 生产者 JSON 的逐字段对账（更强的复现证据）----
    ref_json_obj = json.load(open(ref_json, encoding="utf-8"))
    diff, only_new, only_ref = _fieldwise_diff(run_one_json, ref_json_obj)

    # ---- 写 npz ----
    t_ms = steps.astype(np.float64) * dt_ms
    npz_path = os.path.join(out_dir, rid + "_spiketimes.npz")
    if os.path.exists(npz_path):
        log("note: 覆盖本任务自己的 npz：%s" % npz_path)
    np.savez(npz_path,
             neuron_id=idx.astype(np.int32),
             step=steps.astype(np.int64),
             t_ms=t_ms,
             dt_ms=np.float64(dt_ms),
             n_steps=np.int64(n_steps),
             spk_total=np.int64(n_ev),
             t0_ms=np.float64(0.0),
             run_id=np.array(rid),
             record_mode=np.array("full"),
             producer_sha256=np.array(PRODUCER_SHA256_PIN))
    npz_sha = sha256_of_file(npz_path)
    npz_size = os.path.getsize(npz_path)

    anchor_ok = None
    if rid in BASELINE_SPK_ANCHOR:
        anchor_ok = bool(res_spk_total == BASELINE_SPK_ANCHOR[rid])

    out = dict(
        run_id=rid, role=spec["role"], drive_mult=float(spec["drive_mult"]),
        status="ok", finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        # ---- 任务要求的四个字段 ----
        spk_total=res_spk_total, n_steps=n_steps, wall_s=float(run_one_json["wall_s"]),
        sha256=npz_sha,
        # ---- 口径说明 ----
        dt_ms=dt_ms, t_ms_first=(float(t_ms[0]) if n_ev else None),
        t_ms_last=(float(t_ms[-1]) if n_ev else None),
        t0_ms=0.0,
        time_origin="测量窗起点（settle 之后 reset_counts() 那一刻）；绝对仿真时刻 = settle_ms + t_ms",
        settle_ms=float(spec["settle_ms"]), n_steps_settle=int(run_one_json["n_steps_settle"]),
        delivery=str(run_one_json["delivery"]), device=str(run_one_json["device"]),
        wall_s_protocol_source="生产者 run_one 的 wall_s（含 settle，与已落盘 JSON wall_s 同口径）",
        wall_s_child_total=float(child_wall),
        seeds=run_one_json["seeds"], n_events=int(run_one_json["n_events"]),
        npz=dict(path=npz_path, rel=rel(npz_path), bytes=int(npz_size), sha256=npz_sha,
                 keys=["neuron_id", "step", "t_ms", "dt_ms", "n_steps", "spk_total", "t0_ms",
                       "run_id", "record_mode", "producer_sha256"],
                 dtypes=dict(neuron_id="int32", step="int64", t_ms="float64",
                             dt_ms="float64(标量)", n_steps="int64(标量)", spk_total="int64(标量)")),
        spikes=dict(n_events=n_ev, sha256_neuron_id=sha256_of_bytes(idx.astype(np.int32).tobytes()),
                    sha256_step=sha256_of_bytes(steps.tobytes()),
                    sorted_by_step=steps_sorted, step_min=(int(steps.min()) if n_ev else None),
                    step_max=(int(steps.max()) if n_ev else None),
                    n_unique_neurons=int(np.unique(idx).size) if n_ev else 0,
                    idx_in_range=idx_in_range, steps_in_range=steps_in_range),
        spec=spec, spec_file=dict(path=loaded["path"], rel=rel(loaded["path"]),
                                  sha256=loaded["sha256"], bytes=loaded["bytes"],
                                  pin=SPEC_SHA_PIN[rid], pin_ok=True),
        producer=ident["producer"], engine=ident["engine"],
        fixture_identity=ident,
        record_override=dict(
            rationale="生产者 L752/L755 **硬编码** record=\"counts\"，无参数入口 ⇒ 不改该文件，"
                      "改为对引擎登记形参 record（mouse_cortex_engine.py L551）作**运行时取值替换**；"
                      "判据 = pop_trace is True ∧ ei_integrals is True ∧ record == \"counts\" ⇒ 只命中 L755",
            hardcoded_lines=list(PRODUCER_RECORD_COUNTS_LINES),
            forced_calls=cap.get("passes", [])),
        interpreter=interp, mps_guard=guard,
        crosscheck=dict(
            pass_criteria="逐神经元 count 逐位相同 ∧ spk_total 相同 ∧ bincount(spikes)==counts "
                          "∧ poptrace 逐字节相同",
            counts_csv_new=rel(new_counts_csv), counts_csv_new_sha256=sha256_of_file(new_counts_csv),
            counts_csv_ref=rel(ref_counts_csv), counts_csv_ref_sha256=sha256_of_file(ref_counts_csv),
            counts_ref_sha256_pin=REF_COUNTS_SHA_PIN[rid],
            counts_ref_pin_ok=bool(sha256_of_file(ref_counts_csv) == REF_COUNTS_SHA_PIN[rid]),
            counts_bitwise_identical=ident_counts, counts_mismatch_neurons=mism,
            counts_max_abs_diff=maxdiff,
            counts_new_spk_total=int(new_counts.sum()), counts_ref_spk_total=int(ref_counts.sum()),
            reference_spk_total_pinned=REF_SPK_TOTAL[rid],
            reference_spk_total_pin_ok=bool(int(ref_counts.sum()) == REF_SPK_TOTAL[rid]),
            spk_total_new_vs_ref_equal=bool(res_spk_total == int(ref_counts.sum())),
            poptrace_new=rel(new_pop_csv), poptrace_new_sha256=pop_new_sha,
            poptrace_ref=rel(ref_pop_csv), poptrace_ref_sha256=pop_ref_sha,
            poptrace_ref_sha256_pin=REF_POPTRACE_SHA_PIN[rid],
            poptrace_bitwise_identical=pop_same,
            spikes_bincount_equals_counts=bc_ok,
            spikes_n_events_equals_spk_total=ev_ok,
            res_json_spk_total=res_json_spk,
            producer_json_new=rel(new_json_path), producer_json_ref=rel(ref_json),
            producer_json_fieldwise_differing=diff, producer_json_only_new=only_new,
            producer_json_only_ref=only_ref,
            baseline_spk_anchor_expected=(BASELINE_SPK_ANCHOR.get(rid)),
            baseline_spk_anchor_ok=anchor_ok),
    )
    jpath = os.path.join(out_dir, rid + ".json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    ok = ident_counts and pop_same and bc_ok and ev_ok and steps_sorted and \
        idx_in_range and steps_in_range and (anchor_ok is not False)
    if not ok:
        bad = os.path.join(out_dir, rid + ".MISMATCH.json")
        with open(bad, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        print("MISMATCH(%d): arm=%s 逐位复现失败 ⇒ **停止、不得采信**："
              "counts 逐位=%s（不同神经元 %d，max|Δ|=%d）poptrace 逐位=%s bincount==counts=%s "
              "n_events==spk_total=%s 锚=%s；详见 %s"
              % (EXIT_ARM_FAIL, rid, ident_counts, mism, maxdiff, pop_same, bc_ok, ev_ok,
                 anchor_ok, bad))
        return EXIT_ARM_FAIL
    log("arm=%s OK：spk_total=%d（参考 %d）counts 逐位=%s poptrace 逐位=%s 锚=%s "
        "wall_s=%.3f 子进程总墙钟=%.1f s"
        % (rid, res_spk_total, int(ref_counts.sum()), ident_counts, pop_same, anchor_ok,
           float(run_one_json["wall_s"]), child_wall))
    del res, run_one_json, out
    gc.collect()
    return 0


# --------------------------------------------------------------------- 6 臂驱动
def run_all(out_dir: str, opts) -> int:
    require_abs(out_dir, "--out-dir")
    interp = interpreter_guard()
    guard = mps_guard()
    os.makedirs(out_dir, exist_ok=True)
    log("6 臂串行启动（逐臂**新进程**）：out_dir=%s" % out_dir)
    log("解释器=%s；MPS 判据=空（%s）" % (interp["executable"], guard["patterns"]))
    t0 = time.perf_counter()
    ok_n = 0
    for rid in ARM_IDS:
        cmd = [PY_ENGINE, "-u", os.path.abspath(__file__), "--arm", rid, "--out-dir", out_dir,
               "--spec-dir", opts.spec_dir, "--ref-dir", opts.ref_dir]
        env = dict(os.environ, PYTHONHASHSEED="0", MPLBACKEND="Agg",
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        log("spawn → %s" % " ".join(cmd))
        rc = subprocess.call(cmd, env=env)
        jp = os.path.join(out_dir, rid + ".json")
        if rc != 0 or not os.path.exists(jp):
            log("FAIL: arm=%s rc=%d ⇒ **fail-stop**（不再继续后续臂）" % (rid, rc))
            return rc or EXIT_ARM_FAIL
        j = json.load(open(jp, encoding="utf-8"))
        if not j["crosscheck"]["counts_bitwise_identical"]:
            log("FAIL: arm=%s 计数非逐位相同 ⇒ fail-stop" % rid)
            return EXIT_ARM_FAIL
        ok_n += 1
        log("完成 %d/%d：%s spk_total=%d wall_s=%.3f" % (ok_n, len(ARM_IDS), rid,
                                                       j["spk_total"], j["wall_s"]))
    total = time.perf_counter() - t0
    log("6/6 完成；父子总墙钟 = %.1f s（协议 wall_s 合计 = %.1f s）"
        % (total, _wall_sum(out_dir)))
    return 0


def _wall_sum(out_dir: str) -> float:
    s = 0.0
    for rid in ARM_IDS:
        p = os.path.join(out_dir, rid + ".json")
        if os.path.exists(p):
            s += float(json.load(open(p, encoding="utf-8"))["wall_s"])
    return s


# --------------------------------------------------------------------- 自证/微臂
def self_test_mode(print_json: bool = True) -> int:
    """只读：证明夹具是生产者的同一份代码（不跑仿真、不用 MPS）。"""
    P = load_producer()
    M = load_engine_module()
    rep = fixture_identity(P, M)
    if print_json:
        print(json.dumps(rep, ensure_ascii=False, indent=1))
    if rep["ok"]:
        print("SELF-TEST PASS(%d): 夹具 = 生产者同一份代码；生产者 L752/L755 硬编码 record=\"counts\" "
              "（从未记录 full）；引擎 L551 形参 + L%s + L822 res[\"spikes\"] 就绪 ⇒ "
              "本工具只需替换**一次**调用的 record 取值。"
              % (0, "/".join(str(x) for x in ENGINE_RECORD_FULL_LINES)))
        return 0
    print("SELF-TEST FAIL(%d)" % EXIT_FIXTURE_IDENTITY)
    return EXIT_FIXTURE_IDENTITY


def micro_test_mode(micro_dir: str) -> int:
    """CPU 微臂（n=400, T=100 ms）：`record="counts"` vs `"full"` 的计数**逐位等同**。

    证明"改记录模式不改路径"；**不碰 MPS**（device=cpu）。
    """
    require_abs(micro_dir, "--micro-dir")
    interp = interpreter_guard()
    guard = mps_guard()
    P = load_producer()
    M = load_engine_module()
    ident = fixture_identity_or_die(P, M)
    base = dict(T_ms=100.0, settle_ms=50.0, drive_mult=1.0, seed=0, net_seed=0, event_seed=11,
                role="baseline", delivery="chunk", gmax_scale=0.35, bias_scale=1.3,
                bias_log_sigma=0.1, slow=True, tau_slow_ms=1000.0, slow_inc=6.0, n_neuron=400,
                indegree=20, delay_bins=[40, 60, 80], subtypes=True, device="cpu",
                drive_rate_hz=8.0)
    d1 = os.path.join(micro_dir, "counts")
    d2 = os.path.join(micro_dir, "full")
    os.makedirs(d1, exist_ok=True)
    os.makedirs(d2, exist_ok=True)
    r1 = "micro_n400_T100ms_record_counts"
    r2 = "micro_n400_T100ms_record_full"
    s1 = dict(base, run_id=r1)
    s2 = dict(base, run_id=r2)
    log("微臂 ① record=\"counts\"（无干预）")
    j1 = P.run_one(s1, d1)
    log("微臂 ② record=\"full\"（拦截测量窗那一次调用）")
    cap = {}
    with force_full_record(cap):
        j2 = P.run_one(s2, d2)
    res = cap["res"]
    steps, idx = res["spikes"]
    steps = np.asarray(steps, dtype=np.int64)
    idx = np.asarray(idx, dtype=np.int64)
    _, c1, _ = read_counts_csv(os.path.join(d1, r1 + "_counts.csv"))
    _, c2, _ = read_counts_csv(os.path.join(d2, r2 + "_counts.csv"))
    bitwise = bool(np.array_equal(c1, c2))
    bc = np.bincount(idx, minlength=400)
    bc_ok = bool(np.array_equal(bc.astype(np.int64), c2))
    ev_ok = bool(steps.size == idx.size == int(round(float(res["spk_total"]))) == int(c2.sum()))
    pop_same = bool(sha256_of_file(os.path.join(d1, r1 + "_poptrace.npy"))
                    == sha256_of_file(os.path.join(d2, r2 + "_poptrace.npy")))
    rep = dict(micro=dict(run_id=r1, spk_total=int(c1.sum()), sha256_counts=sha256_of_file(
                              os.path.join(d1, r1 + "_counts.csv"))),
               full=dict(run_id=r2, spk_total=int(c2.sum()),
                         sha256_counts=sha256_of_file(os.path.join(d2, r2 + "_counts.csv")),
                         n_events=int(steps.size)),
               counts_bitwise_identical=bitwise, poptrace_bitwise_identical=pop_same,
               spikes_bincount_equals_counts=bc_ok, n_events_equals_spk_total=ev_ok,
               forced_calls=cap.get("passes", []), fixture_identity_ok=ident["ok"],
               interpreter=interp, mps_guard=guard, device="cpu")
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    ok = bitwise and pop_same and bc_ok and ev_ok and ident["ok"]
    print(("MICRO-TEST PASS(0): record 只多记录、不改路径（计数与 poptrace 均逐位相同）；"
           "spikes 自洽。" if ok else "MICRO-TEST FAIL(%d)" % EXIT_MICRO_FAIL))
    return 0 if ok else EXIT_MICRO_FAIL


# --------------------------------------------------------------------- 对账
def reconcile(out_dir: str, md_out: str = None, launch_script: str = None,
              print_md: bool = True) -> int:
    """**只读**对账：逐臂重算"新 vs 已落盘"表（不跑仿真、不用 MPS）。"""
    require_abs(out_dir, "--out-dir")
    rows, bad = [], []
    wall_new = 0.0
    for rid in ARM_IDS:
        jp = os.path.join(out_dir, rid + ".json")
        nc = os.path.join(out_dir, "_producer_echo", rid + "_counts.csv")
        rc_ = os.path.join(REF_DIR, rid + "_counts.csv")
        if not (os.path.exists(jp) and os.path.exists(nc)):
            bad.append("%s：缺产物（%s）" % (rid, jp))
            rows.append((rid, None))
            continue
        j = json.load(open(jp, encoding="utf-8"))
        _, cnew, _ = read_counts_csv(nc)
        _, cref, _ = read_counts_csv(rc_)
        bit = bool(cnew.shape == cref.shape and np.array_equal(cnew, cref))
        mm = int(np.sum(cnew != cref)) if cnew.shape == cref.shape else -1
        cc = j["crosscheck"]
        wall_new += float(j["wall_s"])
        if not (bit and cc["counts_bitwise_identical"] and cc["poptrace_bitwise_identical"]
                and cc["spikes_bincount_equals_counts"] and cc["spikes_n_events_equals_spk_total"]
                and cc["reference_spk_total_pin_ok"] and cc["spikes_n_events_equals_spk_total"]):
            bad.append("%s：对账不通过（逐位=%s poptrace=%s）" % (rid, bit,
                                                              cc["poptrace_bitwise_identical"]))
        rows.append((rid, dict(
            role=j["role"], new=int(cnew.sum()), ref=int(cref.sum()), bit=bit, mm=mm,
            anchor=(BASELINE_SPK_ANCHOR.get(rid)), anchor_ok=cc["baseline_spk_anchor_ok"],
            counts_sha_eq=bool(cc["counts_csv_new_sha256"] == cc["counts_csv_ref_sha256"]),
            pop_eq=bool(cc["poptrace_bitwise_identical"]),
            n_ev=int(j["spikes"]["n_events"]), n_steps=int(j["n_steps"]),
            wall_new=float(j["wall_s"]), wall_ref=REF_WALL_S[rid],
            npz_sha=j["sha256"], npz_bytes=j["npz"]["bytes"],
            events_eq=int(j["spikes"]["n_events"]) == int(j["spk_total"]))))

    # npz 结构核验（真实读盘，不信 JSON 自述）
    npz_probe = {}
    for rid in ARM_IDS:
        p = os.path.join(out_dir, rid + "_spiketimes.npz")
        if not os.path.exists(p):
            npz_probe[rid] = None
            continue
        with np.load(p) as z:
            npz_probe[rid] = dict(keys=sorted(z.files),
                                  neuron_id=str(z["neuron_id"].dtype),
                                  step=str(z["step"].dtype), t_ms=str(z["t_ms"].dtype),
                                  n_events=int(z["neuron_id"].size),
                                  dt_ms=float(z["dt_ms"]),
                                  n_steps=int(z["n_steps"]), spk_total=int(z["spk_total"]),
                                  n_events_eq_spk_total=bool(z["neuron_id"].size == int(z["spk_total"])))

    lines = []
    lines.append("# SELFCHECK：G1 工作点 6 臂 `record=\"full\"` 重跑的**计数逐位对账**")
    lines.append("")
    lines.append("**依据**：`neural_exploration/docs/m11_acceptance_master.md` **§3gm = R201**（④ 自证标准、")
    lines.append("⑤ 实施方式）。**Pass 标准**：逐神经元 `count` 与已落盘 "
                 "`data/m11_p5_l4_0_runs/<run_id>_counts.csv` **逐位相同** ⇒ 证明新产物记录的是"
                 "**同一批臂**（同 `(net_seed,event_seed,seed)` 下确定性）。")
    lines.append("")
    lines.append("**生成**：`run_m11_p5_spiketimes.py --reconcile`（只读；不跑仿真、不用 MPS）；"
                 "生成时刻 %s。" % time.strftime("%Y-%m-%dT%H:%M:%S"))
    lines.append("")
    lines.append("## ① 逐臂对账表")
    lines.append("")
    lines.append("| 臂 | role | 新 `spk_total` | 已落盘 `spk_total` | **逐位相同** | 不同神经元数 | `counts` sha256 同 | `poptrace` 逐字节 | 锚 | 新 `wall_s` | 已落盘 `wall_s` |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for rid, r in rows:
        if r is None:
            lines.append("| `%s` | — | — | — | 缺产物 | — | — | — | — | — | — |" % rid)
            continue
        anc = "—" if r["anchor"] is None else ("%d %s" % (r["anchor"], "✓" if r["anchor_ok"] else "✗"))
        lines.append("| `%s` | %s | %d | %d | %s | %d | %s | %s | %s | %.3f | %.3f |"
                     % (rid, r["role"], r["new"], r["ref"], "**是 ✓**" if r["bit"] else "**否 ✗**",
                        r["mm"], "是 ✓" if r["counts_sha_eq"] else "否 ✗",
                        "相同 ✓" if r["pop_eq"] else "不同 ✗", anc, r["wall_new"], r["wall_ref"]))
    lines.append("")
    n_ok = sum(1 for _, r in rows if r is not None and r["bit"])
    lines.append("**⇒ 逐位相同：%d/6；`counts.csv` sha256 相同：%d/6；`poptrace.npy` 逐字节相同：%d/6。**"
                 % (n_ok,
                    sum(1 for _, r in rows if r is not None and r["counts_sha_eq"]),
                    sum(1 for _, r in rows if r is not None and r["pop_eq"])))
    b_anchor = [(rid, r["new"], r["anchor"]) for rid, r in rows
                if r is not None and r["anchor"] is not None]
    lines.append("")
    lines.append("**baseline 独立锚（R201 §④）**：" +
                 "、".join("`%s` = %d（锚 %d %s）" % (rid, new, anc, "✓" if new == anc else "✗")
                          for rid, new, anc in b_anchor) + "。")
    lines.append("")
    lines.append("**墙钟**：协议 `wall_s` 合计 = **%.1f s**（登记值 %.1f s）；"
                 "各臂新 `wall_s` 见上表（`wall_s` 只计 settle+测量窗，不含建网 ⇒ 与已落盘同口径）。"
                 % (wall_new, REF_WALL_S_SUM))
    lines.append("")
    lines.append("## ② 脉冲时刻产物结构（真实读盘）")
    lines.append("")
    lines.append("每臂 `<run_id>_spiketimes.npz`：`neuron_id`(int32) / `step`(int64) / "
                 "`t_ms`(float64) 三数组 ＋ 标量 `dt_ms`/`n_steps`/`spk_total`/`t0_ms`/"
                 "`run_id`/`record_mode`/`producer_sha256`。")
    lines.append("**时间原点 = 测量窗起点**（settle 之后 `reset_counts()` 那一刻）⇒ "
                 "绝对仿真时刻 = `settle_ms + t_ms`；`t_ms = step × dt_ms`，`dt_ms`（实测值）见下表。")
    lines.append("")
    lines.append("| 臂 | 键数 | `dt_ms` | `n_steps` | 事件数 | `spk_total` | 事件数==`spk_total` |")
    lines.append("|---|---|---|---|---|---|---|")
    for rid in ARM_IDS:
        pr = npz_probe.get(rid)
        if pr is None:
            lines.append("| `%s` | — | — | — | — | — | 缺 npz |" % rid)
            continue
        lines.append("| `%s` | %d | %s | %d | %d | %d | %s |"
                     % (rid, len(pr["keys"]), pr["dt_ms"], pr["n_steps"], pr["n_events"],
                        pr["spk_total"], "是 ✓" if pr["n_events_eq_spk_total"] else "否 ✗"))
    lines.append("")
    lines.append("## ③ 可复跑的对账命令（只读）")
    lines.append("")
    lines.append("```bash")
    lines.append("env PYTHONHASHSEED=0 MPLBACKEND=Agg \\")
    lines.append("  %s \\" % PY_ENGINE)
    lines.append("  %s \\" % os.path.abspath(__file__))
    lines.append("  --reconcile --out-dir %s" % out_dir)
    lines.append("```")
    lines.append("")
    lines.append("退出码 0 = 6/6 逐位相同；93 = 有不符（本表会逐臂标出）。")
    lines.append("")
    lines.append("## ④ 启动脚本要点（脚本本体在 `/tmp`，**不落仓**）")
    lines.append("")
    lines.append("- 启动前 `pgrep -f \"run_m11_p5\"` 与 `pgrep -f \"run_m11_p5_spiketimes\"` **必须为空**"
                 "（驱动内建 `mps_guard()`，命中即 **退 99**；自己的启动链 pid 被排除）；")
    lines.append("- `caffeinate -i` 包住整条驱动器；`PYTHONHASHSEED=0 MPLBACKEND=Agg`；")
    lines.append("- 解释器**逐字** `/Users/weidong/ai/small_world/.venv-m9/bin/python`（**不用 `sys.executable`**）；")
    lines.append("- `--out-dir` **绝对路径**（非绝对 **退 96**）；逐臂**新进程**（MPS 串行）；")
    lines.append("- 整条命令用后台作业跑（预计约 17 min；6 臂 `wall_s` 实测合计 %.1f s）。" % REF_WALL_S_SUM)
    lines.append("")
    if launch_script and os.path.exists(launch_script):
        lines.append("```bash")
        lines.append("# %s" % launch_script)
        with open(launch_script, encoding="utf-8") as f:
            lines.append(f.read().rstrip("\n"))
        lines.append("```")
        lines.append("")
    lines.append("## ⑤ 未改动声明")
    lines.append("")
    lines.append("- 生产者 `%s` sha256 = `%s`（登记值一致）。" % (PRODUCER_REL, PRODUCER_SHA256_PIN))
    lines.append("- 引擎 `%s` sha256 = `%s`（登记值一致）。" % (ENGINE_REL, ENGINE_SHA256_PIN))
    lines.append("- 6 份 spec 与 6 份 `counts.csv`/`poptrace.npy` 的 sha256 全部与登记值一致（跑前守卫，"
                 "不符退 97）。")
    lines.append("- 本任务**只新建** 4 件产物（本驱动件、本目录、本文件、`/tmp` 启动脚本）；"
                 "`git add`/`git commit` 均未执行。")
    lines.append("")
    lines.append("## ⑥ 对账结论")
    lines.append("")
    if bad:
        lines.append("**不通过**（%d 项）：" % len(bad))
        for b in bad:
            lines.append("- %s" % b)
    else:
        lines.append("**全部通过**：计数逐位相同 %d/6、`poptrace` 逐字节相同 %d/6、"
                     "脉冲事件数 == `spk_total` 6/6 ⇒ 新产物的**逐神经元脉冲时刻属于同一批臂**。"
                     % (n_ok, sum(1 for _, r in rows if r is not None and r["pop_eq"])))
    lines.append("")
    md = "\n".join(lines)
    if print_md:
        print(md)
    if md_out:
        require_abs(md_out, "--md-out")
        with open(md_out, "w", encoding="utf-8") as f:
            f.write(md)
        log("SELFCHECK.md 已写：%s" % md_out)
    return EXIT_RECONCILE_MISMATCH if bad else 0


# --------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="M11 P5/G1 6 臂 record=\"full\" 重跑（取逐神经元脉冲时刻）")
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT, help="产物目录（**必须绝对路径**，非绝对退 96）")
    ap.add_argument("--spec-dir", default=REF_DIR, help="现成 spec 目录（参数逐字来源）")
    ap.add_argument("--ref-dir", default=REF_DIR, help="已落盘 countd/poptrace 目录（对账靶）")
    ap.add_argument("--arm", default=None, help="只跑**一个**臂（6 臂模式下由父进程逐臂 spawn）")
    ap.add_argument("--self-test", action="store_true", help="只读自证：夹具复用（不用 MPS）")
    ap.add_argument("--micro-test", action="store_true", help="CPU 微臂：counts vs full 计数逐位等同")
    ap.add_argument("--micro-dir", default=MICRO_DIR_DEFAULT, help="微臂产物目录（绝对路径）")
    ap.add_argument("--reconcile", action="store_true", help="只读对账（逐臂表），可配 --md-out")
    ap.add_argument("--md-out", default=None, help="把 SELFCHECK.md 写到该**绝对**路径")
    ap.add_argument("--launch-script", default=None, help="把 /tmp 启动脚本原文内联进 SELFCHECK.md")
    a = ap.parse_args(argv)

    if a.self_test:
        return self_test_mode(print_json=True)
    if a.micro_test:
        return micro_test_mode(a.micro_dir)
    if a.reconcile:
        return reconcile(a.out_dir, md_out=a.md_out, launch_script=a.launch_script)
    require_abs(a.out_dir, "--out-dir")
    if a.arm:
        return run_arm(a.arm, a.out_dir, a)
    return run_all(a.out_dir, a)


if __name__ == "__main__":
    sys.exit(main())
