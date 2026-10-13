"""**生产者改动惰性验证**：证明"给 `run_m11_p5_arms.py` 加 `--bias-log-sigma`"对**不带该键的臂**
是**逐位惰性**的（⇒ 正在跑的 J5 L0 作业与历史 seed=0 锚点都不受影响）。

**为什么必须做**：J5（L0 全量）此刻正在用 `--plan` 驱动器跑，而驱动器**逐臂 spawn 新进程**读取
**磁盘上当时的**生产者 ⇒ 我在作业中途改工具，若改动不是惰性的，同一作业会前后臂语义不一致
（且我会同时破坏"冻结工具 sha256"的记录）。

**做法（三层，缺一层都不算证明）**
1. 从 `git show HEAD:` 取出**改动前**的生产者（sha256 必须 == 记录值 `24617d90dd4a6eb9…`）；
2. 同一份**不带 `bias_log_sigma` 键**的 spec，分别用旧/新模块跑（CPU、同解释器、同 numpy）
   ⇒ 比 `sha256_of_trace` / `spk_total` / `ei_ratio` / 产物 JSON 键集：必须**完全相同**；
3. **默认等价**：新模块下"不带键" vs "显式 `bias_log_sigma=0.30`（夹具默认值）" ⇒ 必须逐位相同
   （⇒ 不显式给时沿用默认，语义保持）。

用法：
    OMP_NUM_THREADS=1 ./.venv-m9/bin/python -m neural_exploration.tools.verify_m11_producer_inert
产物：`data/m11_producer_inert_check.json`（并打印逐项对照）
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "neural_exploration", "data")
REL = "neural_exploration/tools/run_m11_p5_arms.py"
PRE_CHANGE_SHA = "24617d90dd4a6eb9f518244808029ac3"
PY = "./.venv-m9/bin/python"

BASE = dict(n_neuron=1024, T_ms=200.0, settle_ms=0.0, drive_mult=1.0, drive_rate_hz=8.0,
            seed=0, device="cpu", config_id="inert_check", role="baseline", mechanism="none",
            delivery="chunk", indegree=20, gmax_scale=1.0, bias_scale=0.70,
            delay_bins=[40, 60, 80], subtypes=True, slow=True)
DYNAMIC_FIELDS = ("sha256_of_trace", "agg.spk_total", "ei.ei_ratio", "spikes_total")


def dig(a: dict, path: str):
    """按 `a.b` 取嵌套字段（产物 JSON 里 `spk_total` 在 `agg` 下、`ei_ratio` 在 `ei` 下）。"""
    cur = a
    for k in path.split("."):
        cur = cur[k]
    return cur


def sha_of_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def git_head_text() -> bytes:
    p = subprocess.run(["git", "show", "HEAD:%s" % REL], cwd=PROJECT_ROOT,
                       capture_output=True, check=True)
    return p.stdout


def run_arm(module: str, spec_path: str, out_dir: str) -> dict:
    """跑一臂（新模块走 `-m`，旧模块走文件路径）。"""
    env = dict(os.environ, PYTHONPATH=PROJECT_ROOT, PYTHONHASHSEED="0",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if module == "new":
        cmd = [PY, "-m", "neural_exploration.tools.run_m11_p5_arms",
               "--arm", spec_path, "--out-dir", out_dir]
    else:
        cmd = [PY, module, "--arm", spec_path, "--out-dir", out_dir]
    p = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit("臂失败（%s）：\n%s\n%s" % (module, p.stdout[-500:], p.stderr[-1500:]))
    run_id = json.load(open(spec_path))["run_id"]
    with open(os.path.join(out_dir, run_id + ".json")) as f:
        return json.load(f)


def main() -> int:
    t0 = time.time()
    tmp = tempfile.mkdtemp(prefix="m11_inert_")
    old_path = os.path.join(tmp, "old_producer.py")
    head = git_head_text()
    with open(old_path, "wb") as f:
        f.write(head)
    head_sha = sha_of_bytes(head)
    print("== 生产者惰性验证 ==")
    print("  改动前（git HEAD）sha256 = %s（期望前缀 %s）%s"
          % (head_sha[:32], PRE_CHANGE_SHA, "✓" if head_sha.startswith(PRE_CHANGE_SHA) else "✗"))
    with open(os.path.join(PROJECT_ROOT, REL), "rb") as f:
        new_sha = sha_of_bytes(f.read())
    print("  改动后（工作树）sha256   = %s" % new_sha[:32])
    if not head_sha.startswith(PRE_CHANGE_SHA):
        print("**git HEAD 的版本不是记录的冻结版本 ⇒ 惰性对照对象不对，停**")
        return 13

    specs = {}
    for tag, sigma in (("nokey", None), ("default_030", 0.30)):
        sp = dict(BASE, run_id="inert_1024_T200ms_d1.0_s0_%s" % tag)
        if sigma is not None:
            sp["bias_log_sigma"] = sigma
        p = os.path.join(tmp, "spec_%s.json" % tag)
        with open(p, "w") as f:
            json.dump(sp, f, indent=1)
        specs[tag] = p

    print("  ① 旧模块 × 不带键 …"); a = run_arm(old_path, specs["nokey"], os.path.join(tmp, "o1"))
    print("  ② 新模块 × 不带键 …"); b = run_arm("new", specs["nokey"], os.path.join(tmp, "o2"))
    print("  ③ 新模块 × 显式 σ=0.30 …"); c = run_arm("new", specs["default_030"], os.path.join(tmp, "o3"))

    rows, fails = [], []
    for name, x, y in (("旧(无键) vs 新(无键)", a, b), ("新(无键) vs 新(σ=0.30 显式)", b, c)):
        d = {k: (dig(x, k), dig(y, k), dig(x, k) == dig(y, k)) for k in DYNAMIC_FIELDS}
        same_keys = set(x.keys()) == set(y.keys())
        rows.append(dict(pair=name, fields=d, same_product_key_set=same_keys))
        print("  %-26s " % name + " | ".join("%s %s" % (k, "✓" if v[2] else "✗") for k, v in d.items())
              + " | 键集相同 %s" % ("✓" if same_keys else "✗"))
        for k, (vx, vy, ok) in d.items():
            if not ok:
                fails.append("%s: %s 不同（%r vs %r）" % (name, k, vx, vy))
        if not same_keys:
            fails.append("%s: 产物 JSON 键集不同（%r vs %r）"
                         % (name, sorted(set(x) - set(y)), sorted(set(y) - set(x))))
    out = dict(tool="tools/verify_m11_producer_inert.py", generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
               pre_change_sha256=head_sha, post_change_sha256=new_sha,
               spec_base=BASE, dynamic_fields=list(DYNAMIC_FIELDS), rows=rows,
               verdict=("惰性成立：加 --bias-log-sigma 对不带该键的臂逐位无影响，且"
                        "不显式给 ⇒ 等价于显式默认 0.30（既有臂/seed=0 锚点安全）") if not fails
                       else "**惰性失败**",
               failures=fails, elapsed_s=round(time.time() - t0, 2))
    p = os.path.join(DATA_DIR, "m11_producer_inert_check.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("产物 -> %s" % p)
    if fails:
        print("**惰性验证失败（%d 条）**：" % len(fails))
        for x in fails[:10]:
            print("   -", x)
        return 14
    print("惰性验证通过：①=② 逐位相同、②=③ 逐位相同 ⇒ J5 正在跑的 L0 臂与历史 seed=0 锚点均不受影响")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
