"""**生产者改动惰性验证**：证明"给 `run_m11_p5_arms.py` 加 `--bias-log-sigma`"对**不带该键的臂**
是**逐位惰性**的（⇒ 正在跑的 J5 L0 作业与历史 seed=0 锚点都不受影响）。

**为什么必须做**：J5（L0 全量）此刻正在用 `--plan` 驱动器跑，而驱动器**逐臂 spawn 新进程**读取
**磁盘上当时的**生产者 ⇒ 我在作业中途改工具，若改动不是惰性的，同一作业会前后臂语义不一致
（且我会同时破坏"冻结工具 sha256"的记录）。

**做法（四对，缺一对都不算证明）**
1. 从 `git show HEAD:` 取出**改动前**的生产者（sha256 必须 == 记录值，见 `--expect-old`）；
2. **惰性**：同一份**不带任何新键**的 spec，旧/新模块各跑一次 ⇒ 除**新增字段**外，
   **所有共同字段（非易变）逐位相同**、且键集差异**恰好等于**声明的新增键集（**只增不改**）；
3. **默认等价**：新模块"不带键" vs "显式默认值（`bias_log_sigma=0.30`/`tau_slow_ms=1000`/`slow_inc=6.0`）"
   ⇒ **键集 + 全部非易变字段逐位相同**（⇒ 不显式给时沿用默认，既有臂语义保持）；
4. **显式缺席**：`slow=False` 臂 ⇒ `slow_trace_stats is None` ∧ `slow_enabled is False`
   （**"未启用"是可区分的显式状态，不是键缺失**）；
5. **杠杆有效（可证伪的正对照）**：`tau_slow_ms=2000 / slow_inc=12` 与默认档 ⇒ 轨迹**必须不同**
   （否则新 CLI 键"被接受但没生效"）。

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
# **旧版本锚点**：取 `data/m11_p5_plan_l1.json` 的 `frozen_tool.sha256_full`（= 本次改动的"改动前"版本，
# 也是 L1 计划跑前必须匹配的版本）⇒ 不硬编码，随计划记录走（可用 --expect-old 覆盖）。
L1_PLAN = os.path.join(DATA_DIR, "m11_p5_plan_l1.json")
L0_PLAN = os.path.join(DATA_DIR, "m11_p5_plan_l0.json")


def _plan_frozen_sha(path: str) -> str:
    try:
        return json.load(open(path))["frozen_tool"]["sha256_full"]
    except Exception:  # noqa: BLE001
        return "unavailable"
PY = "./.venv-m9/bin/python"

BASE = dict(n_neuron=1024, T_ms=200.0, settle_ms=0.0, drive_mult=1.0, drive_rate_hz=8.0,
            seed=0, device="cpu", config_id="inert_check", role="baseline", mechanism="none",
            delivery="chunk", indegree=20, gmax_scale=1.0, bias_scale=0.70,
            delay_bins=[40, 60, 80], subtypes=True, slow=True)


def flat(d, pre=""):
    """把产物 JSON 展平成 `路径→标量`（嵌套 dict/list 也要逐个比）。"""
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, pre + ("/" if pre else "") + str(k)))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(flat(v, pre + "/%d" % i))
    else:
        out[pre] = d
    return out


def same_except_volatile(x: dict, y: dict):
    """共同字段（排除易变字段）逐位相同？返回 (ok, 逐条差异)。"""
    fx, fy = flat(x), flat(y)
    bad = []
    for k in sorted(set(fx) & set(fy)):
        top = k.split("/")[0]
        if top in VOLATILE:
            continue
        if fx[k] != fy[k]:
            bad.append("%s: %r vs %r" % (k, fx[k], fy[k]))
    return (not bad), bad
DYNAMIC_FIELDS = ("sha256_of_trace", "agg.spk_total", "ei.ei_ratio", "spikes_total")
# 逐次运行必然不同的字段（不能用来判"惰性"）；其余共同字段必须逐位相同
VOLATILE = {"wall_s", "ms_per_step", "ms_per_step_steady", "rss_mb", "pid", "finished_at",
            # **存储路径字符串**（随 out-dir 与模块所在位置变化，不是数值/语义）：逐位比较无意义
            "counts", "pop_trace"}
# 本次改动**允许新增**的产物键（"只增不改"契约；键集差异必须恰好等于它）
EXPECTED_NEW_KEYS = {"slow_enabled", "slow_trace_stats"}
EXPECT_SLOW_KEYS = {"n", "alive", "mean", "std", "min", "max", "argmax_step"}


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


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-old", default=None,
                    help="期望的『改动前』sha256 前缀（默认取 L1 计划 frozen_tool 记录）")
    args = ap.parse_args(argv)
    t0 = time.time()
    tmp = tempfile.mkdtemp(prefix="m11_inert_")
    old_path = os.path.join(tmp, "old_producer.py")
    head = git_head_text()
    with open(old_path, "wb") as f:
        f.write(head)
    head_sha = sha_of_bytes(head)
    expect_old = args.expect_old or _plan_frozen_sha(L1_PLAN)
    print("== 生产者惰性验证（五对；CPU）==")
    print("  改动前（git HEAD）sha256 = %s" % head_sha[:32])
    print("  期望的『改动前』版本（L1 计划 frozen_tool 记录）= %s ⇒ %s"
          % (str(expect_old)[:32], "✓" if head_sha.startswith(str(expect_old)[:32]) else "✗"))
    print("  [链] L0 计划记录 %s → (--bias-log-sigma) → 本次改动前 %s → 本次改动后 %s"
          % (str(_plan_frozen_sha(L0_PLAN))[:16], str(expect_old)[:16], sha_of_bytes(
              open(os.path.join(PROJECT_ROOT, REL), "rb").read())[:16]))
    with open(os.path.join(PROJECT_ROOT, REL), "rb") as f:
        new_sha = sha_of_bytes(f.read())
    print("  改动后（工作树）sha256   = %s" % new_sha[:32])
    if not head_sha.startswith(str(expect_old)[:32]):
        print("**git HEAD 的版本不是期望的『改动前』版本 ⇒ 惰性对照对象不对，停**"
              "（若刚提交过改动，请用 --expect-old 指定）")
        return 13

    specs = {}
    for tag, extra in (("nokey", None),
                       ("explicit_defaults", {"bias_log_sigma": 0.30, "tau_slow_ms": 1000.0,
                                              "slow_inc": 6.0}),
                       ("slow_off", None),
                       ("slow_amp", {"tau_slow_ms": 2000.0, "slow_inc": 12.0})):
        # 默认等价对故意复用同一 run_id：**两臂规格的唯一差异就是新增的键**（arm_id 不应成为差异来源）
        rid_tag = "nokey" if tag in ("nokey", "explicit_defaults") else tag
        sp = dict(BASE, run_id="inert_1024_T200ms_d1.0_s0_%s" % rid_tag)
        if tag == "slow_off":
            sp["slow"] = False
        if extra:
            sp.update(extra)
        p = os.path.join(tmp, "spec_%s.json" % tag)
        with open(p, "w") as f:
            json.dump(sp, f, indent=1)
        specs[tag] = p

    a = run_arm(old_path, specs["nokey"], os.path.join(tmp, "o1"))          # 旧 × 无键
    b = run_arm("new", specs["nokey"], os.path.join(tmp, "o2"))              # 新 × 无键
    c = run_arm("new", specs["explicit_defaults"], os.path.join(tmp, "o3"))  # 新 × 显式默认
    d = run_arm("new", specs["slow_off"], os.path.join(tmp, "o4"))           # 新 × slow=False
    e = run_arm("new", specs["slow_amp"], os.path.join(tmp, "o5"))           # 新 × 放大档
    rows, fails = [], []

    # ---- 第 2 对：惰性（只增不改）----
    ok_same, bad_same = same_except_volatile(a, b)
    key_diff = (set(b.keys()) - set(a.keys()), set(a.keys()) - set(b.keys()))
    dyn = {k: (dig(a, k), dig(b, k), dig(a, k) == dig(b, k)) for k in DYNAMIC_FIELDS}
    print("  比较口径：易变字段 %r 与存储路径 %r 不参与逐位比较（其余共同字段必须全同）"
          % (sorted(VOLATILE - {"counts", "pop_trace"}), ["counts", "pop_trace"]))
    print("  [惰性] 旧(无键) vs 新(无键)：动态字段 %s｜共同字段(非易变)逐位相同 %s｜键集差异 +%r -%r"
          % (" ".join("%s%s" % (k, "✓" if v[2] else "✗") for k, v in dyn.items()),
             "✓" if ok_same else "✗", sorted(key_diff[0]), sorted(key_diff[1])))
    rows.append(dict(pair="旧 vs 新（无新键 spec）", dynamic=dyn, common_fields_identical=ok_same,
                     keys_added=sorted(key_diff[0]), keys_removed=sorted(key_diff[1])))
    if not ok_same:
        fails.append("惰性失败：共同字段被改动 ⇒ " + "；".join(bad_same[:5]))
    if set(key_diff[0]) != EXPECTED_NEW_KEYS or key_diff[1]:
        fails.append("键集差异不是『只增不改』：新增 %r（期望 %r）、删除 %r"
                     % (sorted(key_diff[0]), sorted(EXPECTED_NEW_KEYS), sorted(key_diff[1])))

    # ---- 第 3 对：默认等价 ----
    ok_def, bad_def = same_except_volatile(b, c)
    same_keys_def = set(b.keys()) == set(c.keys())
    dyn_def = {k: (dig(b, k), dig(c, k), dig(b, k) == dig(c, k)) for k in DYNAMIC_FIELDS}
    print("  [默认等价] 无键 vs 显式(σ=0.30, τ=1000, inc=6.0)：键集相同 %s｜共同字段逐位相同 %s｜%s"
          % ("✓" if same_keys_def else "✗", "✓" if ok_def else "✗",
             " ".join("%s%s" % (k, "✓" if v[2] else "✗") for k, v in dyn_def.items())))
    rows.append(dict(pair="无键 vs 显式默认", common_fields_identical=ok_def,
                     same_product_key_set=same_keys_def, dynamic=dyn_def))
    if not (ok_def and same_keys_def):
        fails.append("默认等价失败：%s" % ("；".join(bad_def[:5]) or "键集不同"))

    # ---- 第 4 对：显式缺席（slow=False ⇒ null + false）----
    st = a.get("slow_trace_stats", "<缺>")
    print("  [显式缺席] slow=True 臂：slow_enabled=%r、slow_trace_stats 键集 ⊇ %s ⇒ %s"
          % (b.get("slow_enabled"), "✓" if (isinstance(b.get("slow_trace_stats"), dict)
                                           and EXPECT_SLOW_KEYS <= set(b["slow_trace_stats"]))
             else "✗", {k: b["slow_trace_stats"][k] for k in
                        ("n", "alive", "mean", "std", "min", "max", "argmax_step")}))
    print("             slow=False 臂：slow_enabled=%r、slow_trace_stats=%r（**显式 null，不是键缺失**）"
          % (d.get("slow_enabled"), d.get("slow_trace_stats")))
    rows.append(dict(pair="slow=False 显式状态", slow_enabled=d.get("slow_enabled"),
                     slow_trace_stats_is_null=(d.get("slow_trace_stats") is None),
                     slow_on_stats_keys=(sorted(b.get("slow_trace_stats", {}).keys())
                                         if isinstance(b.get("slow_trace_stats"), dict) else None)))
    if d.get("slow_enabled") is not False or d.get("slow_trace_stats") is not None:
        fails.append("slow=False 臂未写显式 null/False：enabled=%r stats=%r"
                     % (d.get("slow_enabled"), d.get("slow_trace_stats")))
    if not (isinstance(b.get("slow_trace_stats"), dict)
            and EXPECT_SLOW_KEYS <= set(b["slow_trace_stats"])):
        fails.append("slow=True 臂的 slow_trace_stats 缺必需键：%r"
                     % sorted(EXPECT_SLOW_KEYS - set(b.get("slow_trace_stats") or {})))

    # ---- 第 5 对：新键**有效**（可证伪正对照）----
    ok_amp = (b["sha256_of_trace"] != e["sha256_of_trace"])
    print("  [新键有效] 默认档 vs τ=2000/inc=12：trace 逐位不同 = %s（不同才说明键真进了引擎）"
          % ("✓" if ok_amp else "✗"))
    rows.append(dict(pair="新键有效（正对照）", trace_differs=ok_amp,
                     sha_default=b["sha256_of_trace"], sha_amp=e["sha256_of_trace"]))
    if not ok_amp:
        fails.append("新 CLI 键**被接受但没生效**（τ/inc 改了而轨迹逐位相同）⇒ 静默失效")
    out = dict(tool="tools/verify_m11_producer_inert.py", generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
               pre_change_sha256=head_sha, post_change_sha256=new_sha,
               spec_base=BASE, dynamic_fields=list(DYNAMIC_FIELDS), rows=rows,
               verdict=("五对全过：①共同字段只增不改（新增 %r）②不显式给 ≡ 显式默认 0.30/1000/6.0 "
                        "③slow=False ⇒ 显式 null+False ④新键改 τ/inc 真的改轨迹 ⇒ 既有臂/J6 数值安全"
                        % sorted(EXPECTED_NEW_KEYS)) if not fails else "**惰性/等价/显式性检查失败**",
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
