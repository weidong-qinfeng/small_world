"""主agent 独立验收探针（归因判别）：M9 记录值 4.6783 Hz 到底能不能复现？

**背景**：M10-B1 结算声称"M9 冻结引擎同网络/同参数/同协议重跑得 6.0032 Hz vs 记录 4.6783 Hz（+28%）
⇒ M9 全规模绝对数值不可复现，机制=MPS 静默读越界内存、值依赖分配历史"。

**主agent 复核发现两处口径疑点**（读码确认，非推测）：
  1. `tools/reeval_m10_m9_rebuttals.py::run_arm` 的群率口径**包含 settle 窗**：
     `counts` 覆盖 `nn = (SETTLE+T)/dt + 8 = 26,008` 步，却除以 `t_meas_s = T_MS*1e-3 = 1.0 s`
     ⇒ 报告值被放大 **26,008/20,000 = 1.3004 倍**（settle 300.4 ms 计入了分子）。
     折算：冻结引擎 6.0032/1.3004 = **4.6163 Hz**，与记录 4.6783 只差 **−1.3%**。
  2. M9 记录值出自 `AdultCircuit.run_resting`（settle 后 `reset_counts()` 再测 20,000 步，**口径正确**），
     且 `v0 = v_rest + 2*N(0,1)`（`rng(1)`）由 `run_resting` 内部设定；重跑工具用 `engine.reset(seed=0)`
     **未传 v0** ⇒ 初值语义也不同。

**本探针做判别（三臂，全部经 M9 自己的 `run_resting` 入口）**：
  * `A_frozen`  = 冻结引擎原样（应复现 M9 记录值 ⇒ 证明"不可复现"是重跑工具口径问题）
  * `B_patched` = **仅**把 CSR 投递公式改正确（实例级 monkey-patch，**不修改冻结文件**），其余全同
                 ⇒ 给出"修复本身"对群率的真实影响（与 B1 的 m10 引擎对照）
  * `C_oscount` = 统计每步发放神经元数分布（量化缺陷触发频率：≥2 个神经元同一步发放的比例）

只读 M9 冻结件；产物写 `data/_mainagent_m9_path_attribution.{json,csv}`。

用法：
    PYTHONHASHSEED=0 MPLBACKEND=Agg ./.venv-m9/bin/python \
        -m neural_exploration.tools.verify_m9_path_attribution --arm A_frozen
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
import types

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUT_JSON = os.path.join(DATA_DIR, "_mainagent_m9_path_attribution.json")
OUT_CSV = os.path.join(DATA_DIR, "_mainagent_m9_path_attribution.csv")

T_MS = 1000.0
SETTLE_MS = 300.0
SEGMENT = 2000                      # diag_p7_drive_sensitivity.SEGMENT（与 M9 原脚本一致）
M9_RECORDED_LAM0_0 = 4.6783         # M9 P7 rhythm0 参考值（m9_p7_sleep.csv）


def patched_deliver_step(self, spk, s: int) -> int:
    """冻结件 `AdultEngine._deliver_step` 的**唯一**改动：CSR 块起点公式（M10 E5-FIX）。"""
    total = 0
    for grp in self._groups:
        idx = spk.nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        c = grp["t_counts"][idx]
        tot = int(c.sum().item())
        if tot == 0:
            continue
        total += tot
        off = torch.repeat_interleave(grp["t_row"][idx] - (torch.cumsum(c, 0) - c), c)
        base = self._arange(tot)[:tot] + off
        slot = ((s + grp["delay"]) % self.n_slot) * (2 * self.n)
        self._rings[grp["delay"]].index_add_(
            0, grp["t_post_off"][base] + slot, grp["t_gmax"][base])
    return total


def patched_deliver_chunk(self, spk_ring, w: int, step0: int, t_count) -> int:
    """同 `_deliver_chunk`，唯一改动 = packed 的 CSR 块起点公式。"""
    if w <= 0:
        return 0
    rows = spk_ring[:w]
    t_count += rows.sum(dim=0)
    flat = rows.reshape(-1).nonzero(as_tuple=False).flatten()
    if flat.numel() == 0:
        return 0
    n_spk = int(flat.numel())
    from ..src.adult_engine import MAX_EDGES_PER_FLUSH as _MAXF
    if n_spk * (self.n_edge / max(self.n, 1)) > _MAXF:
        tot_all = 0
        for k in range(w):
            tot_all += self._deliver_step(rows[k], step0 + k)
        return tot_all
    n = self.n
    kk = torch.div(flat, n, rounding_mode="floor")
    nn = flat - kk * n
    total = 0
    for grp in self._groups:
        c = grp["t_counts"][nn]
        tot = int(c.sum().item())
        if tot == 0:
            continue
        total += tot
        slot_k = ((step0 + kk + grp["delay"]) % self.n_slot) * (2 * n)
        packed = torch.stack([grp["t_row"][nn] - (torch.cumsum(c, 0) - c), slot_k], dim=1)
        rep = torch.repeat_interleave(packed, c, dim=0)
        base = self._arange(tot)[:tot] + rep[:, 0]
        dst = grp["t_post_off"][base] + rep[:, 1]
        self._rings[grp["delay"]].index_add_(0, dst, grp["t_gmax"][base])
    return total


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="A_frozen",
                    choices=["A_frozen", "B_patched"])
    ap.add_argument("--lam0", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    from ..src.adult_circuit import AdultCircuit, CircuitParams
    from .validate_p9_resting import PARAMS, BG

    P = dict(PARAMS)
    c = AdultCircuit(device=os.environ.get("M9_ATTR_DEVICE", "mps"),
                     params=CircuitParams(**P), use_compile=False)
    c.build()
    dt = P["dt_ms"]
    g_ext = BG["epsp_mv"] / 0.104
    nn = int(round((SETTLE_MS + T_MS) / dt)) + 8

    if args.arm == "B_patched":
        c.engine._deliver_step = types.MethodType(patched_deliver_step, c.engine)
        c.engine._deliver_chunk = types.MethodType(patched_deliver_chunk, c.engine)
        print("[attr] 已实例级打补丁：仅 CSR 块起点公式（冻结文件零修改）", flush=True)

    # M9 原脚本（diag_p7_drive_sensitivity.run_lam）逐行复刻
    rng = np.random.default_rng(args.seed + 4242)
    n_ev = rng.poisson(max(args.lam0, 0.0) * nn * dt * 1e-3, size=c.n_neurons)
    total = int(n_ev.sum())
    if total == 0:
        c.engine.set_external_events(np.zeros(0, np.int64), np.zeros(0, np.int64),
                                     np.zeros(0, np.float32), nn)
    else:
        neurons = np.repeat(np.arange(c.n_neurons, dtype=np.int64), n_ev)
        steps = rng.integers(0, nn, size=total).astype(np.int64)
        c.engine.set_external_events(steps, neurons,
                                     np.full(total, g_ext, dtype=np.float32), nn)
    c._bg_steps, c._bg_args = nn, (BG["rate_hz"], g_ext)

    t0 = time.time()
    st = c.run_resting(T_ms=T_MS, settle_ms=SETTLE_MS, seed=args.seed, pop_trace=True,
                       segment_steps=SEGMENT)
    pop = np.asarray(st["pop"], dtype=np.float64)
    wall = time.time() - t0

    # 缺陷触发频率：每步发放神经元数（pop 为每步 spike 数）
    step_spk = pop
    frac_multi = float(np.mean(step_spk >= 2)) if step_spk.size else float("nan")
    mean_spk_per_step = float(step_spk.mean()) if step_spk.size else float("nan")
    p99 = float(np.quantile(step_spk, 0.99)) if step_spk.size else float("nan")

    rec = dict(arm=args.arm, lam0=args.lam0, seed=args.seed, device=str(c.engine.device),
               n_neurons=int(c.n_neurons), n_edge=int(c.pre.size),
               protocol=dict(T_MS=T_MS, SETTLE_MS=SETTLE_MS, SEGMENT=SEGMENT,
                             g_ext=g_ext, entry="AdultCircuit.run_resting"),
               pop_rate_hz=float(st["pop_rate_hz"]), n_spikes=int(st["n_spikes"]),
               rate_mean_hz=float(st["rate_mean_hz"]),
               rate_median_hz=float(st["rate_median_hz"]),
               silent_frac=float(st["silent_frac"]),
               ms_per_step=float(st["ms_per_step"]),
               m9_recorded_hz=M9_RECORDED_LAM0_0,
               rel_dev_vs_recorded=float((st["pop_rate_hz"] - M9_RECORDED_LAM0_0)
                                         / M9_RECORDED_LAM0_0),
               bug_trigger=dict(frac_steps_ge2_spikes=frac_multi,
                                mean_spikes_per_step=mean_spk_per_step,
                                p99_spikes_per_step=p99),
               wall_s=round(wall, 1), frozen_file_modified=False)

    rows = []
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f:
            rows = json.load(f)
    rows = [r for r in rows if not (r["arm"] == rec["arm"] and r["lam0"] == rec["lam0"])]
    rows.append(rec)
    with open(OUT_JSON, "w") as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["arm", "lam0", "n_neurons", "n_edge", "pop_rate_hz", "n_spikes",
                    "rate_median_hz", "silent_frac", "ms_per_step", "m9_recorded_hz",
                    "rel_dev_vs_recorded", "frac_steps_ge2_spikes", "mean_spikes_per_step",
                    "wall_s"])
        for r in rows:
            w.writerow([r["arm"], r["lam0"], r["n_neurons"], r["n_edge"],
                        round(r["pop_rate_hz"], 6), r["n_spikes"],
                        round(r["rate_median_hz"], 6), round(r["silent_frac"], 6),
                        round(r["ms_per_step"], 4), r["m9_recorded_hz"],
                        round(r["rel_dev_vs_recorded"], 6),
                        round(r["bug_trigger"]["frac_steps_ge2_spikes"], 6),
                        round(r["bug_trigger"]["mean_spikes_per_step"], 4), r["wall_s"]])
    print("MAINAGENT_ATTR " + json.dumps(rec, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
