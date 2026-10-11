"""M9 P9：类型级锚扰动预测验证（清单 §1.4/§4.5；全规模 + 分段调用）。

判据（**只读** `data/m9_behavior_reference.csv` 的 perturbation 段，运行前定稿，不事后调）：
  - **命中率 ≥70%**（有锚子集）：预注册因果方向预测 vs 实测后果类一致率；
  - **sham 零臂自检**：同代码路径 factor=1.0 → **逐神经元脉冲计数逐位相同**；
  - **确定性逐位**：同配置重跑逐位一致；
  - 操作有效性前置（非科学判据）：激活 → 目标池自身率↑；
  - 特异性对照（描述性）：同规模随机目标集施加同一操作的效应量对照。

协议：T=1000ms / settle=300ms / 分段 2000 步 / 定稿物理档（e_inh=−80、ek=−77、ahp_g=25、
v_floor=−85，**构建后读回断言**）/ MPS。

**工程结构（M9-B3 实测教训，见 docs/m9_env_notes.md L40）**：单进程连续多试次会触发
**MPS 分配器退化**（同一进程内试次墙钟 35s → 133s → 201s → 248s → >1000s）→ 本工具改为
**每个目标池一个全新子进程**（各自 build + 后台事件 + 运行），**基线只算一次**并把逐神经元
计数落盘复用到全部池 → 既规避退化，也保证各池口径逐位一致（同 seed 同配置）。父进程只读盘
判定，不占 MPS。

**前置诚实声明**：P7 R3 已证明"活动自主自持、输入增益过低"→ 类型级扰动可能不显著。
本工具**不得为凑命中率改带或选择性报告**；不显著/方向不符**如实记录为 M9 的正当反证**。

输出：`data/m9_p9_perturbation.csv` + `data/m9_p9_target_mapping.csv`（**运行前写盘的预测**）
      + `data/m9_p9_runs/*`（逐运行计数/掩码留档）+ `reports/neuro/m9_perturbation_hitrate.png`
用法：PYTHONHASHSEED=0 MPLBACKEND=Agg ./.venv-m9/bin/python -u -m neural_exploration.tools.validate_p9_perturbation
"""

from __future__ import annotations

import csv as _csv
import json
import os
import subprocess
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # = neural_exploration/
REPO_ROOT = os.path.dirname(PROJECT_ROOT)   # = 仓库根（子进程 -m 需要它在 sys.path 上）
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports", "neuro")
REF_CSV = os.path.join(DATA_DIR, "m9_behavior_reference.csv")
PLAN_CSV = os.path.join(DATA_DIR, "m9_perturbation_plan.csv")
RUNS_DIR = os.path.join(DATA_DIR, "m9_p9_runs")
OUT_CSV = os.path.join(DATA_DIR, "m9_p9_perturbation.csv")
OUT_MAP = os.path.join(DATA_DIR, "m9_p9_target_mapping.csv")
OUT_DIAG = os.path.join(DATA_DIR, "m9_p9_pool_diag.json")
OUT_PNG = os.path.join(REPORT_DIR, "m9_perturbation_hitrate.png")
OUT_LOG = os.path.join(REPORT_DIR, "m9_p9_progress.log")

T_MS = float(os.environ.get("M9_P9_T_MS", "1000"))
SETTLE_MS = float(os.environ.get("M9_P9_SETTLE_MS", "300"))
SEG_STEPS = int(os.environ.get("M9_P9_SEG", "2000"))
ABORT_S = float(os.environ.get("M9_P9_ABORT_S", "180"))
DEVICE = os.environ.get("M9_P9_DEVICE", "mps")
POOLS = os.environ.get("M9_P9_POOLS",
                       "MBON,MOD_MB,MOD_ALL,CX_EB,CX_ALL,KC_TO_MBON").split(",")
SEED = 0


def log(msg: str) -> None:
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(OUT_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_band(role: str) -> dict:
    out = {}
    with open(REF_CSV, encoding="utf-8") as f:
        hdr = None
        for row in _csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            if hdr is None:
                hdr = row
                continue
            d = dict(zip(hdr, row))
            if d.get("role") != role:
                continue
            out[d["metric"]] = {"lo": float(d["lo"]), "hi": float(d["hi"]),
                                "unit": d.get("unit", ""),
                                "provenance": d.get("provenance", "")}
    return out


def load_anchors() -> list:
    rows = []
    with open(PLAN_CSV, encoding="utf-8") as f:
        hdr = None
        for row in _csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            if hdr is None:
                hdr = row
                continue
            rows.append(dict(zip(hdr, row)))
    return rows


def save_run(task: str, r, self_mask, partner_mask) -> None:
    os.makedirs(RUNS_DIR, exist_ok=True)
    np.save(os.path.join(RUNS_DIR, task + "_counts.npy"), r.counts.astype(np.float64))
    np.save(os.path.join(RUNS_DIR, task + "_self.npy"), np.asarray(self_mask, bool))
    np.save(os.path.join(RUNS_DIR, task + "_partner.npy"), np.asarray(partner_mask, bool))
    meta = r.brief()
    meta["task"] = task
    with open(os.path.join(RUNS_DIR, task + ".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)


def load_run(task: str):
    return {"counts": np.load(os.path.join(RUNS_DIR, task + "_counts.npy")),
            "self": np.load(os.path.join(RUNS_DIR, task + "_self.npy")),
            "partner": np.load(os.path.join(RUNS_DIR, task + "_partner.npy")),
            "meta": json.load(open(os.path.join(RUNS_DIR, task + ".json"), encoding="utf-8"))}


def run_tasks(key: str) -> list:
    """该目标需要落盘的运行任务名。"""
    from neural_exploration.src.adult_perturb import POOL_SPECS
    if key in POOL_SPECS:
        return [key + "|silence", key + "|activation", key + "|random_silence",
                key + "|random_activation"]
    return [key + "|edge_silence", key + "|random_edge_silence"]


# ---------------------------------------------------------------------------
# 子进程任务模式（每池全新 MPS 上下文）
# ---------------------------------------------------------------------------
def build_circuit():
    from neural_exploration.src.adult_circuit import AdultCircuit, CircuitParams
    from neural_exploration.tools.validate_p9_resting import PARAMS, BG
    c = AdultCircuit(device=DEVICE, params=CircuitParams(**PARAMS), use_compile=False)
    c.build()
    pt = c.engine.point
    rb = {"e_inh": float(pt.e_inh), "ek": float(pt.ek), "ahp_form": str(pt.ahp_form),
          "ahp_g_inc": float(pt.ahp_g_inc), "v_floor": float(pt.v_floor)}
    assert rb["e_inh"] == -80.0 and rb["ek"] == -77.0, rb
    assert rb["ahp_form"] == "conductance" and rb["ahp_g_inc"] == 25.0, rb
    assert rb["v_floor"] == -85.0, rb
    g_ext = BG["epsp_mv"] / 0.104
    n_steps = int(round((SETTLE_MS + T_MS) / PARAMS["dt_ms"])) + 8
    c.build_background(rate_hz=BG["rate_hz"], g_ext=g_ext, n_steps=n_steps, seed=BG["seed"])
    c._bg_steps, c._bg_args = n_steps, (BG["rate_hz"], g_ext)
    return c, rb


def _run(p, tag):
    return p.run(T_ms=T_MS, settle_ms=SETTLE_MS, seed=SEED, segment_steps=SEG_STEPS,
                 abort_s=ABORT_S, tag=tag)


def task_baseline() -> int:
    from neural_exploration.src.adult_perturb import (ANCHOR_MAP, CLASS_NAMES, EDGE_SPECS,
                                                     POOL_SPECS, AdultPerturb)
    c, rb = build_circuit()
    p = AdultPerturb(c)
    log("任务 baseline：参数读回 ✓ %s" % rb)
    z = np.zeros(c.n_neurons, bool)
    p.restore()
    r = _run(p, "baseline")
    save_run("baseline", r, z, z)
    log("  运行 baseline：群体 %.4f Hz，%.0fs（%.2f ms/step）"
        % (r.pop_rate_hz, r.wall_s, r.ms_per_step))
    p.prepare_silence(pool="MBON", factor=1.0)
    s = _run(p, "sham_noop")
    save_run("sham_noop", s, p.pool_mask("MBON"), p.partner_mask("MBON"))
    log("  运行 sham_noop（factor=1.0 同一代码路径）：群体 %.4f Hz" % s.pop_rate_hz)
    p.restore()
    d = _run(p, "baseline_rep")
    save_run("baseline_rep", d, z, z)
    log("  运行 baseline_rep（确定性复跑）：群体 %.4f Hz" % d.pop_rate_hz)
    import dataclasses as _dc
    write_pool_diag(c, rb)
    return 0


def write_pool_diag(c, rb=None) -> None:
    """池结构诊断（**无仿真**：仅 build + 掩码/递质统计）→ OUT_DIAG。"""
    from neural_exploration.src.adult_perturb import (ANCHOR_MAP, CLASS_NAMES, EDGE_SPECS,
                                                     POOL_SPECS, AdultPerturb)
    import dataclasses as _dc
    p = AdultPerturb(c, verbose=False)
    if rb is None:
        pt = c.engine.point
        rb = {"e_inh": float(pt.e_inh), "ek": float(pt.ek), "ahp_form": str(pt.ahp_form),
              "ahp_g_inc": float(pt.ahp_g_inc), "v_floor": float(pt.v_floor)}
    diag = {"readback": rb, "params": {k: v for k, v in _dc.asdict(c.params).items()},
            "n_neurons": int(c.n_neurons), "n_edges": int(c.pre.size), "pools": {}}
    for key in sorted({ANCHOR_MAP[a]["target"] for a in ANCHOR_MAP if ANCHOR_MAP[a]["target"]}):
        if key in POOL_SPECS:
            m = p.pool_mask(key)
            em = m[c.pre]
            post = p.partner_mask(key)
            n_tgt = int(m.sum())
            n_tgt_e = int(em.sum())
        else:
            em = p.edge_mask(key)
            post = p.pool_mask(EDGE_SPECS[key]["to_pool"])
            n_tgt = int(np.unique(c.pre[em]).size)
            n_tgt_e = int(em.sum())
        nt = p.dominant_nt_class(em)
        frac = np.bincount(c.nt_code[em].astype(np.int64), minlength=3) / max(n_tgt_e, 1)
        diag["pools"][key] = {
            "kind": "pool" if key in POOL_SPECS else "edge",
            "n_target": n_tgt, "n_target_edges": n_tgt_e,
            "n_partner": int(post.sum()), "nt_class": nt,
            "nt_name": CLASS_NAMES.get(nt, "?"), "nt_frac": [float(x) for x in frac]}
        log("  池诊断 %s：%s" % (key, {k: v for k, v in diag["pools"][key].items()
                                       if k != "nt_frac"}))
        log("    代理边递质类 exc %.3f / inh %.3f / mod %.3f"
            % tuple(diag["pools"][key]["nt_frac"]))
    with open(OUT_DIAG, "w", encoding="utf-8") as f:
        json.dump(diag, f, ensure_ascii=False, indent=1)
    log("→ %s（池诊断落盘；无仿真）" % OUT_DIAG)


def task_diag() -> int:
    """仅结构诊断（无仿真）：供预算/环境受限时先落盘映射与预测。"""
    c, rb = build_circuit()
    write_pool_diag(c, rb)
    log("diag 任务完成（无仿真）：池诊断 + 预测可落盘")
    return 0


def task_pool(key: str) -> int:
    from neural_exploration.src.adult_perturb import POOL_SPECS, AdultPerturb
    c, _ = build_circuit()
    p = AdultPerturb(c)
    log("任务 %s（全新进程）" % key)
    if key in POOL_SPECS:
        p.prepare_silence(pool=key, factor=0.0)
        save_run(key + "|silence", _run(p, key + "|silence"),
                 p.pool_mask(key), p.partner_mask(key))
        p.prepare_activation(pool=key, mult=1.0)
        save_run(key + "|activation", _run(p, key + "|activation"),
                 p.pool_mask(key), p.partner_mask(key))
        rnd = p.random_matched_mask(key)
        p.prepare_silence(mask=rnd, factor=0.0, label=key + "|random")
        save_run(key + "|random_silence", _run(p, key + "|random_silence"),
                 rnd, p._partner_of(rnd))
        p.prepare_activation(mask=rnd, mult=1.0, label=key + "|random")
        save_run(key + "|random_activation", _run(p, key + "|random_activation"),
                 rnd, p._partner_of(rnd))
    else:
        em = p.edge_mask(key)
        pre_m = np.zeros(c.n_neurons, bool); pre_m[c.pre[em]] = True
        post_m = np.zeros(c.n_neurons, bool); post_m[c.post[em]] = True
        p.prepare_edge_silence(key, factor=0.0)
        save_run(key + "|edge_silence", _run(p, key + "|edge_silence"), pre_m, post_m)
        rem = p.random_matched_edge_mask(key)
        pr = np.zeros(c.n_neurons, bool); pr[c.pre[rem]] = True
        po = np.zeros(c.n_neurons, bool); po[c.post[rem]] = True
        p.prepare_edge_silence(edge_mask=rem, factor=0.0, label=key + "|random")
        save_run(key + "|random_edge_silence", _run(p, key + "|random_edge_silence"), pr, po)
    return 0


# ---------------------------------------------------------------------------
# 父进程编排
# ---------------------------------------------------------------------------
def venv_python() -> str:
    """子进程必须用**项目 venv** 的解释器（`sys.executable` 在 `python -m` 下解析为
    venv 的基底解释器 → 丢失 venv 包与项目根 sys.path，实测 ModuleNotFoundError）。"""
    cand = os.path.join(REPO_ROOT, ".venv-m9", "bin", "python")
    return cand if os.path.exists(cand) else sys.executable


def run_child(task: str) -> int:
    env = dict(os.environ)
    env["M9_P9_TASK"] = task
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [venv_python(), "-u", "-m",
           "neural_exploration.tools.validate_p9_perturbation"]
    t0 = time.perf_counter()
    log(">>> 子进程任务 %s 启动（解释器 %s）" % (task, venv_python()))
    to = float(os.environ.get("M9_P9_CHILD_TIMEOUT", "900"))
    try:
        r = subprocess.run(cmd, env=env, cwd=REPO_ROOT, capture_output=True, text=True,
                           timeout=to)
    except subprocess.TimeoutExpired as e:
        with open(OUT_LOG, "a", encoding="utf-8") as f:
            f.write("[%s] !!! 子进程 %s 超时 %.0fs → 主动终止（**不静默**：该目标登记为 "
                    "unavailable，预算/环境限制）\n" % (time.strftime("%H:%M:%S"), task, to))
            if e.stdout:
                f.write("[child-timeout] %s\n" % e.stdout[-2000:])
        log("!!! 子进程 %s 超时 %.0fs 被终止 → 登记 unavailable" % (task, to))
        return 124
    with open(OUT_LOG, "a", encoding="utf-8") as f:
        for ln in (r.stdout or "").splitlines():
            f.write(ln if ln.startswith("[") else "[child] %s" % ln)
            f.write("\n")
        if r.stderr:
            f.write("!!! 子进程 %s stderr（末 1200 字符）：\n%s\n" % (task, r.stderr[-1200:]))
    for ln in (r.stdout or "").splitlines():
        if "运行 " in ln or "池诊断" in ln or "失败" in ln:
            print(ln, flush=True)
    if r.stderr:
        print("!!! %s stderr: %s" % (task, r.stderr.strip()[-300:]), flush=True)
    log("<<< 子进程 %s 结束 rc=%d（墙钟 %.0fs）" % (task, r.returncode,
                                                  time.perf_counter() - t0))
    return r.returncode


def main() -> int:
    task = os.environ.get("M9_P9_TASK", "")
    if task == "baseline":
        return task_baseline()
    if task == "diag":
        return task_diag()
    if task:
        return task_pool(task)

    from neural_exploration.src.adult_perturb import (ANCHOR_MAP, POOL_SPECS, classify,
                                                     dir_to_class, map_predictions, rel_delta)
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)
    open(OUT_LOG, "w", encoding="utf-8").close()
    band = read_band("perturbation")
    thr = band["effect_threshold_rel"]["lo"]
    hit_min = band["hit_rate_min"]["lo"]
    log("=== M9 P9 扰动预测（§1.4/§4.5）===")
    log("判据带（只读 CSV perturbation 段）：命中率 ≥%.2f / 阈值 %.2f / sham=1 / det=1"
        % (hit_min, thr))
    log("协议：设备 %s、T=%.0fms、settle=%.0fms、%d 步/段；**逐池全新子进程**（规避 MPS "
        "分配器跨试次退化，L40）" % (DEVICE, T_MS, SETTLE_MS, SEG_STEPS))
    t_all = time.perf_counter()

    if not os.path.exists(OUT_DIAG):
        log("先执行 baseline 子进程（池结构诊断 + 零臂自检 + 确定性复跑）")
        rc = run_child("baseline")
        if rc != 0:
            log("baseline 子进程失败 rc=%d → 终止（不静默）" % rc)
            return rc
    diag = json.load(open(OUT_DIAG, encoding="utf-8"))
    pools_nt = {k: v["nt_class"] for k, v in diag["pools"].items()}
    anchors = load_anchors()
    preds = map_predictions([{"anchor_id": a["anchor_id"],
                              "kind": ANCHOR_MAP[a["anchor_id"]]["kind"],
                              "target": ANCHOR_MAP[a["anchor_id"]]["target"],
                              "lit": ANCHOR_MAP[a["anchor_id"]]["lit_direction"],
                              "reason": ANCHOR_MAP[a["anchor_id"]]["map_reason"],
                              "pred_act": ANCHOR_MAP[a["anchor_id"]]["pred_act"],
                              "pred_source": ANCHOR_MAP[a["anchor_id"]]["pred_source"]}
                             for a in anchors if a["anchor_id"] in ANCHOR_MAP], pools_nt)
    n_mapped = len({d["anchor_id"] for d in preds if d["predicted"] != "no-experiment"})
    n_unavail = len({d["anchor_id"] for d in preds if d["predicted"] == "no-experiment"})
    n_slots = sum(1 for d in preds if d["predicted"] != "no-experiment")
    log("锚映射：可测锚 %d（≥%.0f 下限）/ 不可映射 %d（no-experiment）/ 预测槽 %d"
        % (n_mapped, band["n_anchor_min"]["lo"], n_unavail, n_slots))
    with open(OUT_MAP, "w", encoding="utf-8", newline="") as f:
        f.write("# M9 P9 类型级锚 → 代理目标 + **预注册因果方向预测**（运行前写盘；不做事后调）\n")
        f.write("# 预测来源：锚自身的**文献机制/已登记抽象**（pred_source 列；**不由实测反推**）\n")
        f.write("# target_nt_class_diag = 代理目标输出边主导递质类（**仅诊断**；混类 = 限制 R7 直接证据）\n")
        w = _csv.writer(f, lineterminator="\n")
        w.writerow(["anchor_id", "kind", "target", "target_nt_class_diag", "slot",
                    "readout", "predicted", "pred_source", "lit_direction", "map_reason"])
        for d in preds:
            w.writerow([d["anchor_id"], d["kind"], d["target"] or "",
                        diag["pools"].get(d["target"], {}).get("nt_name", "-")
                        if d["target"] else "-",
                        d["slot"], d["readout"], d["predicted"], d.get("pred_source", ""),
                        d.get("lit", ""), d.get("reason", "")])
    log("→ %s（预测先于本会话运行落盘）" % OUT_MAP)

    judge_only = os.environ.get("M9_P9_JUDGE_ONLY", "0") == "1"
    # 本会话**尝试运行**的目标子集（判据集合 POOLS 不变；未尝试者如实登记 unavailable）
    run_pools = os.environ.get("M9_P9_RUN_POOLS", ",".join(POOLS)).split(",")
    for key in POOLS:
        need = run_tasks(key)
        if all(os.path.exists(os.path.join(RUNS_DIR, t + ".json")) for t in need):
            log("目标 %s 结果已在盘（跳过运行）" % key)
            continue
        if judge_only or key not in run_pools:
            log("目标 %s **本会话未测**（judge-only/预算与环境限制；登记 unavailable，不静默："
                "未测 ≠ 未命中）" % key)
            continue
        rc = run_child(key)
        if rc != 0:
            log("目标 %s 子进程失败 rc=%d（不静默：该池判定为 unavailable）" % (key, rc))

    base = load_run("baseline")
    sham = load_run("sham_noop")
    rep = load_run("baseline_rep")
    sham_ok = bool(np.array_equal(base["counts"], sham["counts"]))
    det_ok = bool(np.array_equal(base["counts"], rep["counts"]))
    log("sham 零臂自检（factor=1.0 经同一路径）：逐位相同 = %s" % sham_ok)
    log("确定性复跑（同 seed）：逐位相同 = %s" % det_ok)

    def rate(counts, mask):
        if not mask.any():
            return float("nan")
        return float(counts[mask].sum() / mask.sum() / (T_MS / 1000.0))

    meas = {}
    for key in POOLS:
        need = run_tasks(key)
        # **部分臂可判**：静默臂存在即判静默槽；激活臂存在即判激活槽；缺失臂逐槽登记 unavailable
        if not os.path.exists(os.path.join(RUNS_DIR, need[0] + ".json")):
            log("  %s 静默臂结果缺失 → 本会话不参与判定（不静默）" % key)
            continue
        s = load_run(need[0])
        tgt, partner = s["self"], s["partner"]
        b_self, b_part = rate(base["counts"], tgt), rate(base["counts"], partner)
        row = {"self_base": b_self, "partner_base": b_part,
               "n_target": int(tgt.sum()), "n_partner": int(partner.sum()),
               "nt_class": diag["pools"][key]["nt_class"],
               "nt_name": diag["pools"][key]["nt_name"],
               "nt_frac": diag["pools"][key]["nt_frac"],
               "n_target_edges": diag["pools"][key].get("n_target_edges", 0)}
        row["silence_self"] = rate(s["counts"], tgt)
        row["silence_partner"] = rate(s["counts"], partner)
        row["d_silence_self"] = rel_delta(row["silence_self"], b_self)
        row["d_silence_partner"] = rel_delta(row["silence_partner"], b_part)
        row["class_silence_self"] = classify(row["d_silence_self"], thr)
        row["class_silence_partner"] = classify(row["d_silence_partner"], thr)
        def _maybe(task):
            f = os.path.join(RUNS_DIR, task + ".json")
            return load_run(task) if os.path.exists(f) else None

        if key in POOL_SPECS:
            a = _maybe(key + "|activation")
            if a is not None:
                row["activation_self"] = rate(a["counts"], tgt)
                row["activation_partner"] = rate(a["counts"], partner)
                row["d_activation_self"] = rel_delta(row["activation_self"], b_self)
                row["d_activation_partner"] = rel_delta(row["activation_partner"], b_part)
                row["class_activation_partner"] = classify(row["d_activation_partner"], thr)
            else:
                row["class_activation_partner"] = "unavailable"
                row["d_activation_partner"] = float("nan")
                row["d_activation_self"] = float("nan")
            rs = _maybe(key + "|random_silence")
            row["random_silence_partner"] = (rate(rs["counts"], partner) if rs else float("nan"))
            row["d_random_silence_partner"] = rel_delta(row["random_silence_partner"], b_part)
            ra = _maybe(key + "|random_activation")
            row["random_activation_partner"] = (rate(ra["counts"], partner) if ra
                                                else float("nan"))
            row["d_random_activation_partner"] = rel_delta(row["random_activation_partner"],
                                                           b_part)
        else:
            rs = _maybe(key + "|random_edge_silence")
            row["random_silence_partner"] = (rate(rs["counts"], partner) if rs else float("nan"))
            row["d_random_silence_partner"] = rel_delta(row["random_silence_partner"], b_part)
            row["class_activation_partner"] = "n/a(edge)"
            row["d_activation_partner"] = float("nan")
            row["d_activation_self"] = float("nan")
        meas[key] = row
        log("  %s：基线 自身 %.4f / 伙伴 %.4f Hz｜静默 Δ伙伴 %+.4f Δ自身 %+.4f｜"
            "激活 Δ伙伴 %s Δ自身 %s｜随机对照 Δ伙伴 %+.4f"
            % (key, row["self_base"], row["partner_base"], row["d_silence_partner"],
               row["d_silence_self"], row.get("d_activation_partner"),
               row.get("d_activation_self"), row["d_random_silence_partner"]))

    slot_rows = []
    for d in preds:
        if d["predicted"] == "no-experiment":
            slot_rows.append({**d, "measured": "no-experiment", "hit": ""})
            continue
        if d["target"] not in meas:
            slot_rows.append({**d, "measured": "unavailable", "hit": ""})
            continue
        m = meas[d["target"]]
        got = {"S1_silence_partner": m["class_silence_partner"],
               "S2_silence_self": m["class_silence_self"],
               "A1_activation_partner": m["class_activation_partner"],
               "S1_edge_silence_post": m["class_silence_partner"],
               "S2_edge_silence_pre": m["class_silence_self"]}.get(d["slot"], "?")
        slot_rows.append({**d, "measured": got, "hit": bool(got == d["predicted"])})
    testable = [r for r in slot_rows if r["predicted"] != "no-experiment"
                and r["measured"] not in ("unavailable",)]
    n_hit = sum(1 for r in testable if r["hit"])
    hit_rate = n_hit / max(len(testable), 1)
    pool_slots = {}
    for r in testable:
        pool_slots.setdefault((r["target"], r["slot"]), []).append(r)
    pool_hit = sum(1 for v in pool_slots.values() if v[0]["hit"])
    pool_rate = pool_hit / max(len(pool_slots), 1)

    sign_hit = sign_tot = nt_hit = nt_tot = 0
    for r in testable:
        m = meas[r["target"]]
        d_meas = {"S1_silence_partner": m["d_silence_partner"],
                  "S2_silence_self": m["d_silence_self"],
                  "A1_activation_partner": m.get("d_activation_partner", float("nan")),
                  "S1_edge_silence_post": m["d_silence_partner"],
                  "S2_edge_silence_pre": m["d_silence_self"]}.get(r["slot"], float("nan"))
        if np.isfinite(d_meas):
            sign_tot += 1
            if r["predicted_dir"] == 0:
                if abs(d_meas) < thr:
                    sign_hit += 1
            elif (d_meas > 0) == (r["predicted_dir"] > 0):
                sign_hit += 1
        nt = m["nt_class"]
        if nt >= 0 and r["slot"] in ("S1_silence_partner", "A1_activation_partner",
                                     "S1_edge_silence_post") and np.isfinite(d_meas):
            nt_dir = -1 if nt == 1 else +1
            if r["slot"] in ("S1_silence_partner", "S1_edge_silence_post"):
                nt_dir = -nt_dir
            nt_tot += 1
            if classify(d_meas, thr) == dir_to_class(nt_dir):
                nt_hit += 1
    eff = [(k, meas[k].get("d_activation_self", float("nan")))
           for k in POOLS if k in POOL_SPECS and k in meas]
    n_eff = sum(1 for _, v in eff if np.isfinite(v) and v > 0)
    eff_frac = n_eff / max(len(eff), 1) if eff else float("nan")
    crit = {
        "hit_rate": bool(hit_rate >= hit_min) if len(testable) else "not_measured",
        "sham_noop_identical": bool(sham_ok),
        "determinism_bitwise": bool(det_ok),
        "efficacy_activation_self_up": bool(len(eff) == 0 or n_eff / len(eff) >= 0.8),
        "anchor_min_20": bool(n_mapped >= 20),
    }
    # **未测 ≠ 未命中**：若无任何已测预测槽 → 判 PARTIAL（部分完成），不判 FAIL
    if len(testable) == 0:
        verdict = "PARTIAL(未测：无已运行目标池)"
    else:
        verdict = "PASS" if all(crit.values()) else "FAIL"
    log("命中率：锚级 %.4f（%d/%d）｜池级 %.4f（%d/%d）｜判据 ≥%.2f → %s"
        % (hit_rate, n_hit, len(testable), pool_rate, pool_hit, len(pool_slots), hit_min,
           "PASS" if crit["hit_rate"] else "FAIL"))
    log("描述性（**事后**、非判据）：方向符号一致率 %.4f（%d/%d）；代理递质类规则命中率 %.4f（%d/%d）"
        % (sign_hit / max(sign_tot, 1), sign_hit, sign_tot,
           nt_hit / max(nt_tot, 1), nt_hit, nt_tot))
    log("操作有效性（激活自身↑）：%d/%d 池" % (n_eff, len(eff)))
    log("判定：%s（%s）" % (verdict, crit))
    log("总墙钟 %.0fs" % (time.perf_counter() - t_all))

    rows = [["section", "key", "value", "note"],
            ["protocol", "device", DEVICE, ""],
            ["protocol", "T_ms", "%.0f" % T_MS, "测量窗；率判据与 T 无关"],
            ["protocol", "settle_ms", "%.0f" % SETTLE_MS, ""],
            ["protocol", "segment_steps", SEG_STEPS, "全分段（规避长单次调用退化）"],
            ["protocol", "execution", "per-pool fresh subprocess",
             "规避 MPS 分配器跨试次退化（L40）"],
            ["protocol", "seed", SEED, "确定性（基线计数落盘跨池复用，口径逐位一致）"],
            ["protocol", "params_readback", str(diag.get("readback")), "构建后读回断言 ✓"],
            ["protocol", "effect_threshold_rel", thr, "后果类判定阈值（运行前定稿）"],
            ["anchor", "n_anchors_plan", len(anchors), "锚库类型级锚（含 unavailable 行）"],
            ["anchor", "n_anchors_mapped", n_mapped, "可映射到代理池（≥20 下限）"],
            ["anchor", "n_anchors_unavailable", n_unavail, "no-experiment（不入分母）"],
            ["anchor", "n_prediction_slots", n_slots, "池锚 3 槽 / 边级锚 2 槽"],
            ["anchor", "pool_collapse_log",
             json.dumps({k: sum(1 for d in preds if d["target"] == k
                                and d["predicted"] != "no-experiment") for k in POOLS},
                        ensure_ascii=False),
             "同池多锚共享目标集（无 cell_type → 区间不可分；池级为统计诚实单位）"]]
    for k in POOLS:
        if k not in meas:
            rows.append(["pool", k + "_status", "unavailable", "子进程失败/结果缺失（不静默）"])
            continue
        m = meas[k]
        rows.append(["pool", k + "_n_target", m["n_target"], ""])
        rows.append(["pool", k + "_n_partner", m["n_partner"], ""])
        rows.append(["pool", k + "_n_target_edges", m["n_target_edges"], ""])
        rows.append(["pool", k + "_nt_class_diag", m["nt_name"], "代理边主导递质类（诊断列）"])
        rows.append(["pool", k + "_nt_frac_exc_inh_mod", "%.3f/%.3f/%.3f" % tuple(m["nt_frac"]),
                     "代理混类程度（限制 R7 直接证据）"])
        rows.append(["pool", k + "_self_base_hz", "%.4f" % m["self_base"], "基线测量窗"])
        rows.append(["pool", k + "_partner_base_hz", "%.4f" % m["partner_base"], ""])
        for tag in ("silence_partner", "silence_self", "activation_partner",
                    "activation_self", "random_silence_partner",
                    "random_activation_partner"):
            if ("d_" + tag) in m:
                rows.append(["measure", "%s_%s_rel_delta" % (k, tag), "%.4f" % m["d_" + tag],
                             "相对 sham 基线（同 seed 配对）"])
        for tag in ("silence_partner", "silence_self", "activation_partner"):
            if ("class_" + tag) in m:
                rows.append(["measure", "%s_class_%s" % (k, tag), m["class_" + tag], ""])
    for r in slot_rows:
        rows.append(["slot", "%s|%s" % (r["anchor_id"], r["slot"] or "-"),
                     "pred=%s meas=%s hit=%s" % (r["predicted"], r["measured"], r["hit"]),
                     "target=%s slot_dir=%+d" % (r["target"], r["predicted_dir"])])
    n_unavail_slots = sum(1 for r in slot_rows if r["measured"] == "unavailable")
    rows += [["agg", "slots_unavailable", n_unavail_slots,
              "**未测槽（预算/环境限制；不入分母，未测 ≠ 未命中）**"],
             ["agg", "hit_rate_anchor_level", "%.4f" % hit_rate,
              "%d/%d 槽" % (n_hit, len(testable))],
             ["agg", "hit_rate_pool_level", "%.4f" % pool_rate,
              "%d/%d 池×槽" % (pool_hit, len(pool_slots))],
             ["agg", "sign_agreement_rate_descriptive", "%.4f" % (sign_hit / max(sign_tot, 1)),
              "**事后描述性**（非判据）：方向符号一致率 %d/%d" % (sign_hit, sign_tot)],
             ["agg", "nt_class_rule_hit_rate_descriptive", "%.4f" % (nt_hit / max(nt_tot, 1)),
              "**事后描述性**：数据驱动递质类规则命中率 %d/%d（诊断代理混类影响）"
              % (nt_hit, nt_tot)],
             ["agg", "efficacy_activation_self_up_frac", "%.4f" % eff_frac,
              "%d/%d 池自身率↑" % (n_eff, len(eff))],
             ["agg", "sham_noop_bitwise", str(sham_ok), "零臂自检"],
             ["agg", "determinism_bitwise", str(det_ok), ""],
             ["agg", "specificity_target_gt_control_pools",
              "%d/%d（可判定池数，对照组缺失者不计）" % (
                  sum(1 for k in meas
                      if np.isfinite(meas[k].get("d_random_silence_partner", np.nan))
                      and abs(meas[k]["d_silence_partner"])
                      > abs(meas[k]["d_random_silence_partner"])),
                  sum(1 for k in meas
                      if np.isfinite(meas[k].get("d_random_silence_partner", np.nan)))),
              "描述性（目标效应量 > 随机对照；对照组缺失登记为不可判定）"],
             ["agg", "wall_s", "%.0f" % (time.perf_counter() - t_all), ""]]
    for k, v in crit.items():
        rows.append(["crit", k, str(bool(v)), "P9 判据"])
    rows.append(["crit", "verdict", verdict,
                 {"PASS": "PASS",
                  "PARTIAL(未测：无已运行目标池)":
                  "**PARTIAL：本会话无已运行目标池 → 命中率未测（未测 ≠ 未命中；分母为 0）**"}.get(
                     verdict,
                     "**FAIL → 反证记录（方向不符/不显著；与 R3「活动自主自持、输入增益过低」一致）**")])
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        f.write("# M9 P9 扰动预测（§4.5；判据带只读 m9_behavior_reference.csv perturbation 段）\n")
        f.write("# 槽级预测与实测逐条留档（含未命中项）；不显著/不符方向如实记录，未做选择性报告\n")
        f.write("# 运行结构：逐池全新子进程（规避 MPS 分配器跨试次退化）；基线计数落盘跨池复用\n")
        w = _csv.writer(f, lineterminator="\n")
        w.writerows(rows)
    log("→ %s" % OUT_CSV)
    make_plot(meas, hit_rate, pool_rate, hit_min, eff, crit, POOLS, sign_hit, sign_tot)
    log("→ %s" % OUT_PNG)
    log("完成标记 M9_P9_DONE verdict=%s hit_rate=%.4f pool_hit=%.4f"
        % (verdict, hit_rate, pool_rate))
    return 0 if verdict == "PASS" else 2


def make_plot(meas, hit_rate, pool_rate, hit_min, eff, crit, pools, sign_hit, sign_tot):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    ax = axes[0][0]
    ax.bar(["anchor-level", "pool-level", "sign-agree\n(descriptive)"],
           [hit_rate, pool_rate, sign_hit / max(sign_tot, 1)],
           color=["#2ca02c" if hit_rate >= hit_min else "#d62728", "#1f77b4", "#7f7f7f"])
    ax.axhline(hit_min, color="red", ls="--", lw=1, label="pre-registered ≥%.2f" % hit_min)
    ax.set_ylim(0, 1.05); ax.set_ylabel("hit rate")
    ax.set_title("(a) prediction hit rate (mechanism-based, pre-registered)")
    ax.legend(fontsize=8)
    ax = axes[0][1]
    keys = [k for k in pools if k in meas]
    x = np.arange(len(keys))
    ax.bar(x - 0.25, [meas[k]["d_silence_partner"] for k in keys], 0.25,
           label="silence: partner Δ/base", color="#d62728")
    ax.bar(x, [meas[k]["d_random_silence_partner"] for k in keys], 0.25,
           label="silence: random control", color="#7f7f7f")
    ax.bar(x + 0.25, [meas[k].get("d_activation_partner", float("nan")) for k in keys], 0.25,
           label="activation: partner Δ/base", color="#1f77b4")
    ax.axhline(0.05, color="k", ls=":", lw=1); ax.axhline(-0.05, color="k", ls=":", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(keys, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("relative Δ")
    ax.set_title("(b) downstream (partner-pool) consequence + controls")
    ax.legend(fontsize=8)
    ax = axes[1][0]
    if eff:
        ax.bar([k for k, _ in eff], [v for _, v in eff], color="#2ca02c")
    ax.axhline(0, color="k", lw=1)
    ax.set_ylabel("relative Δ of target pool rate")
    ax.set_title("(c) manipulation efficacy: activation → target pool ↑")
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    ax = axes[1][1]
    ks = list(crit.keys())
    ax.barh(ks, [1 if crit[k] else 0 for k in ks],
            color=["#2ca02c" if crit[k] else "#d62728" for k in ks])
    ax.set_xlim(0, 1.2); ax.set_title("(d) P9 criteria (pass=1)")
    ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("M9 P9 type-level perturbation prediction (FlyWire whole-brain; per-pool fresh "
                 "processes; segmented calls)")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=110)


if __name__ == "__main__":
    sys.exit(main())
