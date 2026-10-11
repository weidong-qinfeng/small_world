"""M11 P5（G1 生理工作点门）**离线评估器** —— 只做分析，不做仿真。

《生物仿真M11实施清单》§4.5（P5 判据 (a)–(i)）+ §0.7（判据可测性预注册）+ §7 P5 行
+ 《docs/m11_list_review_master.md》RF-M11-3 + 《docs/m11_acceptance_master.md》A4/A8/A12。

**为什么需要这个工具（主agent 对 M10 的裁定，本工具的三类堵漏对象）**
  1. **门 1（自持比）不是统计可判**：M10 同配置 3 seed = 0.0987/0.1056/0.1005、合并 R=0.1015、
     CI95 [0.0945,0.1091] ⇒ **跨门限**；且零驱动臂只有 **60 个脉冲**（相对不确定度 13%）。
  2. **门 1/2 对机制阶梯无判别性**：M10 基线臂自身即过（+579.6%）⇒「通过」不构成任何证据。
  3. **判据有时长依赖性**：M10 同一配置 T=2 s→0.0980、T=60 s→0.1082、T=120 s→**0.1720**（+76%）；
     λ×4 中位率随 T 由 0→1.0 Hz ⇒ **判据带必须与其协议时长绑定**。
另：M10 的四次「口径事故」（×1.3004 / ×1.15 / 43× / 无断言）与 D2/D3「构造性判据」教训。

**本工具的六项硬门（任一不满足即按四态处置，绝不静默）**
  ① **样本量门**：零驱动臂与基准臂各 ≥3 seed（去重）+ T 使零驱动臂**总脉冲 ≥300**；不满足 ⇒
     该子判据 `unavailable`（**不得判 pass，也不得判 fail**，点估计仅作描述性输出）。
  ② **自持比必须带 CI**：主法 = 泊松对数正态近似（`se(log R)=sqrt(1/S0+1/S1)`，与 M10 实现一致），
     次法 = seed 重采样 + 泊松扰动的参数自助；判据 =「点估计 ≤ 上限 **且** CI 上界 ≤ 上限×1.2」；
     两法 CI 上界相对差 > 阈值 ⇒ `invalid`（口径不一致，先对齐再判）。
  ③ **双估计器一致性硬断言**：`counts` 求和 vs `pop_trace` 求和相对差 ≤2%，否则**整轮 invalid**；
     并做三条非平凡量纲自检：轨迹长度 ⟺ (T_ms, dt_ms)、计数长度 ⟺ N、率×T=计数。
  ④ **时长依赖性实测**：同一配置至少在 {T_min, 2×T_min, 4×T_min} 三档报**同一判据值**，
     把「判据随 T 漂移」做成**被测量的量**；跨 T 判定翻转 ⇒ 该子判据 `invalid`。
  ⑤ **判别性断言**：必须给出「**基线臂（无机制）**」的同一判据值；缺基线臂 ⇒ 整体 `unavailable`；
     基线自身即达标 ⇒ 该判据标 `discriminative=false`，**不得**作为机制有效性证据
     （`m11_p5_targets.csv::criterion_d_as_mechanism_evidence` 中已登记的事实会被机械执行）。
  ⑥ **口径标注**：每条输出行都带 `protocol`（T/settle/seed/bin 宽/平滑窗）与 `estimator`
     （逐神经元 vs 群体总率）字段；非主口径的判定只作**描述性**交叉核对。

**判据带纪律（机械强制）**：带值全部读自 `data/m11_p5_targets.csv`（唯一定稿表，只允许新增行）。
若某子判据依赖的带行 `final=0` 或 `evidence_level=unavailable` ⇒ 该子判据 **status=unavailable**
（本工具拒绝用未定稿的带判 pass/fail）。**本工具不写带、不改带。**

**四态**：`pass` / `fail` / `invalid`（口径或编码路径缺陷 ⇒ 该轮结论无效）/ `unavailable`
（样本量/数据/带未定稿 ⇒ 未测，**不得**计为 pass 或 fail）。（清单 §0.7 #6）

--------------------------------------------------------------------------------
用法
--------------------------------------------------------------------------------
单臂模式（快速核对某个轨迹/计数）::

    PYTHONHASHSEED=0 OMP_NUM_THREADS=1 ./.venv-m11/bin/python \\
      -m neural_exploration.tools.validate_p11_resting \\
      --pop-trace neural_exploration/data/m10_p6_poptraces/poptrace_M0_lamx0.0_n250.npy \\
      --dt-ms 0.05 --T-ms 120000 --n-neuron 250 --protocol-json <protocol.json>

多臂模式（G1 正式评估；协议元数据 + 臂清单都在 protocol-json 内）::

    ... --protocol-json neural_exploration/data/m11_p5_smoke_protocol.json \\
        --out-csv  neural_exploration/data/m11_p5_resting_eval.csv \\
        --out-json neural_exploration/data/m11_p5_resting_eval.json

自检 ::

    ... --check-targets          # CSV ⟺ JSON 判据表互查 + 行级完整性（provenance 不得为空）
    ... --smoke-emit-counts OUT  # 生成**合成**计数夹具（仅用于测双估计器断言；文件内标注 SYNTHETIC）

退出码：**0 = G1 pass；1 = G1 fail；2 = invalid（硬失败）；3 = unavailable（未测/带未定稿）**

--------------------------------------------------------------------------------
protocol-json schema（简版；缺字段即报错，不做默认值静默回退）
--------------------------------------------------------------------------------
{
  "protocol_id": "m11_p5_smoke",          // 必填
  "dt_ms": 0.05, "settle_ms": 500.0,      // 必填（口径）
  "bin_ms": 100.0, "smooth_frames": 5,    // 可选（口径登记）
  "n_neuron": 250,                        // 可选（臂未给时取其值）
  "estimator": "per_neuron",              // 可选，主口径（默认取判据表 estimator_primary）
  "arms": [                               // 必填，≥1
    {"arm_id": "M0_zero_seed0",
     "config_id": "M0_T120",              // 同一配置的臂共享 config_id
     "T_ms": 120000.0, "settle_ms": 500.0,
     "role": "zero_drive",                 // zero_drive|baseline|dose|replica
     "mechanism": "none",                  // none|C-1..C-8|...
     "drive_mult": 0.0, "seed": 0, "replica_of": null,
     "n_neuron": 250,                      // 可选
     "pop_trace": "data/.../*.npy",         // 可选（逐 step 群体脉冲数）
     "counts":    "data/.../*.csv",         // 可选（逐神经元计数：neuron_id,count[,cell_type]）
     "agg": {"spk_total": 1154.0},          // 可选（既有软件的聚合读数，须给 spk_total）
     "ei": {"i_inh_integral": null, "i_exc_integral": null},   // 可选（(e) 的唯一来源）
     "wall_s": 12.3, "ms_per_step": 0.068, "device": "cpu"     // 可选（(g)/L46 检查）
    }
  ]
}
相对路径按**协议文件所在目录的父目录**（项目根）解析，也接受绝对路径。

作者：M11 执行节点 B3（工作包 G1 判据/协议设计与工具骨架）。**只新建；不碰任何冻结件。**
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_TARGETS = os.path.join(PROJECT_ROOT, "neural_exploration", "data", "m11_p5_targets.csv")
DEFAULT_TARGETS_JSON = os.path.join(PROJECT_ROOT, "neural_exploration", "data", "m11_p5_targets.json")
DEFAULT_OUT_CSV = os.path.join(PROJECT_ROOT, "neural_exploration", "data", "m11_p5_resting_eval.csv")
DEFAULT_OUT_JSON = os.path.join(PROJECT_ROOT, "neural_exploration", "data", "m11_p5_resting_eval.json")

STATES = ("pass", "fail", "invalid", "unavailable")
EXIT_PASS, EXIT_FAIL, EXIT_INVALID, EXIT_UNAVAILABLE = 0, 1, 2, 3
Z95 = 1.959963984540054

# 判据 id（清单 §4.5 的 (a)–(i)）
CRIT = {
    "a": "p5_a_rate_median",
    "b": "p5_b_silent_frac",
    "c": "p5_c_selfsustain",
    "d": "p5_d_drive_elasticity",
    "e": "p5_e_ei_balance",
    "f": "p5_f_bounded",
    "g": "p5_g_wallclock",
    "h": "p5_h_determinism",
    "i": "p5_i_tradeoff",
}
GATE_CRITERIA = (CRIT["c"], CRIT["d"], CRIT["e"])


# ===========================================================================
# 判据带表（唯一定稿件）——不写、不改、缺失即报错
# ===========================================================================
class TargetsTable:
    """读 `data/m11_p5_targets.csv`；机械强制「未定稿带不得判定」。"""

    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"判据带表不存在（不做默认值静默回退）：{path}")
        self.path = path
        self.rows: Dict[str, Dict[str, str]] = {}
        order: List[str] = []
        with open(path, encoding="utf-8") as f:
            rdr = csv.DictReader(ln for ln in f if not ln.startswith("#"))
            need = {"key", "value", "value_kind", "unit", "evidence_level", "provenance", "note", "final"}
            if not need.issubset(set(rdr.fieldnames or [])):
                raise ValueError(f"判据表缺列：{need - set(rdr.fieldnames or [])}")
            for row in rdr:
                k = (row["key"] or "").strip()
                if not k:
                    continue
                if k in self.rows:
                    raise ValueError(f"判据表键重复（定稿纪律：只允许新增行）：{k}")
                if not (row.get("provenance") or "").strip():
                    raise ValueError(f"判据行缺 provenance（不得凭印象）：{k}")
                self.rows[k] = row
                order.append(k)
        self.order = order

    # -- 读取 ---------------------------------------------------------------
    def row(self, key: str) -> Dict[str, str]:
        if key not in self.rows:
            raise KeyError(f"判据表缺键：{key}（不得静默回退）")
        return self.rows[key]

    def raw(self, key: str) -> str:
        return (self.row(key)["value"] or "").strip()

    def num(self, key: str) -> Optional[float]:
        v = self.raw(key)
        if v == "" or v.lower() in ("unavailable", "none", "null", "nan"):
            return None
        return float(v)

    def as_list(self, key: str, cast=str) -> List[Any]:
        v = self.raw(key)
        if v == "" or v.lower() == "unavailable":
            return []
        return [cast(x) for x in v.split("|") if x != ""]

    def finalized(self, key: str) -> Tuple[bool, str]:
        """返回 (可用?, 不可用原因)。可用 = final=1 且证据级别非 unavailable。"""
        r = self.row(key)
        ev = r["evidence_level"].strip()
        fin = (r.get("final") or "").strip()
        if ev == "unavailable":
            return False, f"band_unavailable(evidence_level=unavailable): {key}"
        if fin != "1":
            return False, f"band_not_final(final={fin or '?'}): {key}"
        return True, ""

    def band(self, lo_key: str, hi_key: str) -> Tuple[Optional[float], Optional[float], str, List[str]]:
        ok_lo, r_lo = self.finalized(lo_key)
        ok_hi, r_hi = self.finalized(hi_key)
        reasons = [x for x in (r_lo if not ok_lo else "", r_hi if not ok_hi else "") if x]
        lo = self.num(lo_key) if ok_lo else None
        hi = self.num(hi_key) if ok_hi else None
        status = "ok" if (ok_lo and ok_hi) else "unavailable"
        return lo, hi, status, reasons


# ===========================================================================
# 臂（arm）与输入加载
# ===========================================================================
class Arm:
    __slots__ = ("arm_id", "config_id", "T_ms", "settle_ms", "role", "mechanism", "drive_mult",
                 "seed", "replica_of", "n_neuron", "pop_trace_path", "counts_path", "agg", "ei",
                 "wall_s", "ms_per_step", "device", "raw")

    def __init__(self, d: Dict[str, Any], base_dir: str, default_n: Optional[int]):
        self.raw = d
        self.arm_id = str(d.get("arm_id") or f"arm{id(d)}")
        self.config_id = str(d.get("config_id") or "default")
        if "T_ms" not in d:
            raise ValueError(f"臂 {self.arm_id} 缺 T_ms")
        self.T_ms = float(d["T_ms"])
        self.settle_ms = d.get("settle_ms", None)
        self.role = str(d.get("role") or "baseline")
        if self.role not in ("zero_drive", "baseline", "dose", "replica"):
            raise ValueError(f"臂 {self.arm_id} 的 role 非法：{self.role}")
        self.mechanism = str(d.get("mechanism") or "none")
        self.drive_mult = float(d.get("drive_mult", 0.0 if self.role == "zero_drive" else 1.0))
        self.seed = d.get("seed", None)
        self.replica_of = d.get("replica_of", None)
        self.n_neuron = int(d["n_neuron"]) if d.get("n_neuron") else default_n
        self.pop_trace_path = _resolve(d.get("pop_trace"), base_dir)
        self.counts_path = _resolve(d.get("counts"), base_dir)
        self.agg = d.get("agg") or {}
        self.ei = d.get("ei") or {}
        self.wall_s = d.get("wall_s", None)
        self.ms_per_step = d.get("ms_per_step", None)
        self.device = d.get("device", None)
        # 载入
        self.trace: Optional[np.ndarray] = None
        self.counts: Optional[np.ndarray] = None
        self.cell_type: Optional[np.ndarray] = None
        if self.pop_trace_path:
            self.trace = np.load(self.pop_trace_path).astype(np.float64).ravel()
        if self.counts_path:
            self.counts, self.cell_type = load_counts(self.counts_path)
        if self.n_neuron is None and self.counts is not None:
            self.n_neuron = int(self.counts.size)


def _resolve(p: Optional[str], base_dir: str) -> Optional[str]:
    if not p:
        return None
    if os.path.isabs(p):
        return p
    cand = os.path.join(base_dir, p)
    if os.path.exists(cand):
        return cand
    cand2 = os.path.join(PROJECT_ROOT, p)
    return cand2 if os.path.exists(cand2) else cand


COUNT_KEYS = ("count", "spike_count", "n_spikes", "spikes", "counts", "n")
TYPE_KEYS = ("cell_type", "type", "class", "cell_class")


def load_counts(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """逐神经元计数 CSV：必须含 count 列（别名见 COUNT_KEYS），可选 cell_type 列。"""
    with open(path, encoding="utf-8") as f:
        rdr = csv.reader(ln for ln in f if not ln.startswith("#"))
        head = next(rdr, None)
        if head is None:
            raise ValueError(f"counts 文件为空：{path}")
        head = [h.strip() for h in head]
        low = [h.lower() for h in head]
        ci = next((low.index(k) for k in COUNT_KEYS if k in low), None)
        if ci is None:
            # 单列数值文件（无表头）
            try:
                v = np.array([float(head[0])] + [float(r[0]) for r in rdr if r])
                return v, None
            except ValueError:
                raise ValueError(f"counts 文件缺计数列 {COUNT_KEYS}：{path}（表头={head}）")
        ti = next((low.index(k) for k in TYPE_KEYS if k in low), None)
        vals, types = [], []
        for r in rdr:
            if not r:
                continue
            vals.append(float(r[ci]))
            if ti is not None:
                types.append(r[ti].strip())
        if len(vals) < 2:  # 只有一行数据的情形也已覆盖
            raise ValueError(f"counts 文件数据行过少：{path}")
        return (np.asarray(vals, dtype=np.float64),
                np.asarray(types, dtype=object) if ti is not None else None)


# ===========================================================================
# 单臂指标（两个独立估计器并列；每个量都注明 estimator 与 protocol）
# ===========================================================================
def arm_metrics(arm: Arm, dt_ms: float, bin_ms: float, silent_thr_hz: float) -> Dict[str, Any]:
    m: Dict[str, Any] = {"arm_id": arm.arm_id, "checks": [], "hard_fail": []}
    T_s = arm.T_ms / 1000.0
    m["T_s"] = T_s
    steps_expected = int(round(arm.T_ms / dt_ms))
    m["steps_expected"] = steps_expected
    m["n_neuron"] = arm.n_neuron
    m["seed"] = arm.seed

    # ---- 估计器 1：群体轨迹（逐 step 计数）----
    if arm.trace is not None:
        tr = arm.trace
        m["trace_len"] = int(tr.size)
        m["trace_sum"] = float(tr.sum())
        m["trace_finite"] = bool(np.all(np.isfinite(tr)))
        # 非平凡量纲自检：轨迹长度 ⟺ (T_ms, dt_ms)
        if tr.size != steps_expected:
            m["hard_fail"].append(
                f"trajectory_len_mismatch: trace.size={tr.size} != round(T_ms/dt_ms)={steps_expected} "
                f"(T_ms={arm.T_ms}, dt_ms={dt_ms}) —— 测量窗/时长口径不符（M10 ×1.3004 同族缺陷）")
        else:
            m["checks"].append("trace_len == round(T_ms/dt_ms) OK")
        if not m["trace_finite"]:
            m["hard_fail"].append("trajectory_non_finite: 群体轨迹含 NaN/Inf")
        if arm.n_neuron:
            m["trace_mean_rate_hz"] = float(tr.mean() / (dt_ms * 1e-3) / arm.n_neuron)
        # 箱宽口径下的群体统计（描述性）
        b = max(int(round(bin_ms / dt_ms)), 1)
        nb = tr.size // b
        if nb >= 4:
            m_ = tr[:nb * b].reshape(nb, b).sum(axis=1)
            m["bin_spikes"] = m_
            if arm.n_neuron:
                m["bin_rate_hz"] = m_ / (b * dt_ms * 1e-3) / arm.n_neuron
                m["trace_bin_median_rate_hz"] = float(np.median(m["bin_rate_hz"]))
            m["silent_bin_frac"] = float(np.mean(m_ == 0))
            m["n_bins"] = int(nb)

    # ---- 估计器 2：逐神经元计数 ----
    if arm.counts is not None:
        c = arm.counts
        m["counts_n"] = int(c.size)
        m["counts_sum"] = float(c.sum())
        m["counts_finite"] = bool(np.all(np.isfinite(c)))
        if arm.n_neuron and c.size != arm.n_neuron:
            m["hard_fail"].append(
                f"counts_len_mismatch: counts.size={c.size} != n_neuron={arm.n_neuron}（N 口径不符）")
        else:
            m["checks"].append("counts_len == n_neuron OK")
        if not m["counts_finite"]:
            m["hard_fail"].append("counts_non_finite: 逐神经元计数含 NaN/Inf")
        elif float(c.min()) < 0.0:
            m["hard_fail"].append("counts_negative: 逐神经元计数出现负值（不合理 ⇒ 记录损坏）")
        rate = c / T_s
        m["per_neuron_rate_hz"] = rate
        m["rate_median_hz"] = float(np.median(rate))
        m["rate_mean_hz"] = float(rate.mean())
        m["rate_p95_hz"] = float(np.percentile(rate, 95))
        m["rate_p5_hz"] = float(np.percentile(rate, 5))
        m["silent_frac_zero_count"] = float(np.mean(c == 0.0))
        m["silent_frac_lt_thr"] = float(np.mean(rate < silent_thr_hz))
        m["n_active_neurons"] = int(np.sum(c > 0.0))
        if arm.cell_type is not None:
            byc: Dict[str, Dict[str, float]] = {}
            for t in sorted(set(str(x) for x in arm.cell_type)):
                sel = np.array([str(x) == t for x in arm.cell_type])
                if sel.sum() == 0:
                    continue
                byc[t] = {"n": int(sel.sum()), "rate_median_hz": float(np.median(rate[sel])),
                          "silent_frac_zero_count": float(np.mean(c[sel] == 0.0))}
            m["by_cell_type"] = byc

    # ---- 估计器 3：外部聚合读数 ----
    if "spk_total" in arm.agg:
        m["agg_spk_total"] = float(arm.agg["spk_total"])

    # ---- 硬门③：双估计器一致性（同一物理量：总脉冲 / 率）----
    est = {k: v for k, v in (("pop_trace_sum", m.get("trace_sum")),
                             ("counts_sum", m.get("counts_sum")),
                             ("agg_spk_total", m.get("agg_spk_total"))) if v is not None}
    m["estimators_available"] = sorted(est)
    m["estimator_rel_diffs"] = {}
    if len(est) >= 2:
        keys = sorted(est)
        base = max(est[keys[0]], 1e-12)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b_ = est[keys[i]], est[keys[j]]
                den = max(abs(a), abs(b_), 1e-12)
                m["estimator_rel_diffs"][f"{keys[i]}__vs__{keys[j]}"] = float(abs(a - b_) / den)
    else:
        m["hard_fail"].append(
            "double_estimator_unavailable: 仅有 %d 个独立估计器（%s）⇒ 无法做硬门③"
            % (len(est), ",".join(m["estimators_available"]) or "none"))

    # ---- 量纲自检（A10）：率×T=计数（对每个可用估计器）----
    if arm.n_neuron:
        if m.get("counts_sum") is not None:
            m["dim_check_counts"] = float(abs(m.get("counts_sum", 0.0) - (m.get("counts_sum", 0.0) / (arm.n_neuron * T_s)) * arm.n_neuron * T_s))
        if m.get("trace_sum") is not None:
            w = m["trace_sum"] / (arm.n_neuron * T_s)
            m["dim_check_trace"] = float(abs(w * arm.n_neuron * T_s - m["trace_sum"]))
    # ---- (g)/L46：ms_per_step × 步数 vs 墙钟量级（QoS 饥饿检测，非 hard fail）----
    if arm.wall_s and arm.ms_per_step and steps_expected:
        pred_s = float(arm.ms_per_step) * steps_expected / 1000.0
        m["wall_pred_s"] = pred_s
        m["wall_s"] = float(arm.wall_s)
        m["wall_over_pred"] = float(arm.wall_s / pred_s) if pred_s > 0 else None

    # ---- (e)：E/I（唯一来源是臂内注入值；工具不从发放推电流）----
    ie, ii = arm.ei.get("i_exc_integral"), arm.ei.get("i_inh_integral")
    if ie is not None and ii is not None:
        m["ei_i_exc_integral"] = float(ie)
        m["ei_i_inh_integral"] = float(ii)
        m["ei_ratio"] = float(ii) / float(ie) if float(ie) != 0.0 else None
    return m


# ===========================================================================
# 统计：自持比 + CI（主法泊松对数正态 / 次法自助）
# ===========================================================================
def selfsustain_ci(zero_counts: Sequence[float], base_counts: Sequence[float],
                   n_boot: int = 20000, seed: int = 12345) -> Dict[str, Any]:
    z = np.asarray([x for x in zero_counts], dtype=np.float64)
    b = np.asarray([x for x in base_counts], dtype=np.float64)
    out: Dict[str, Any] = {"n_seed_zero": int(z.size), "n_seed_base": int(b.size),
                           "zero_spikes": float(z.sum()), "base_spikes": float(b.sum()),
                           "per_seed_ratio": [float(a / c) if c > 0 else None for a, c in zip(z, b)]}
    if b.sum() <= 0:
        out["error"] = "base_arm_zero_spikes"
        return out
    R = float(z.sum() / b.sum())
    out["ratio_pooled"] = R
    # 主法：泊松对数正态近似（与 data/m10_p5_gate1_ci.json 的实现一致）
    if z.sum() > 0:
        se = math.sqrt(1.0 / z.sum() + 1.0 / b.sum())
        out["se_log"] = float(se)
        out["ci95_lognormal"] = [float(R * math.exp(-Z95 * se)), float(R * math.exp(Z95 * se))]
    else:
        out["se_log"] = None
        out["ci95_lognormal"] = [0.0, 0.0]
        out["note_ci"] = "零驱动臂 0 脉冲 ⇒ R=0 为下界，对数正态近似退化"
    # 次法：seed 重采样 + 泊松扰动的参数自助
    rng = np.random.default_rng(seed)
    Rs = np.empty(n_boot, dtype=np.float64)
    k = 0
    for _ in range(n_boot):
        zs = z[rng.integers(0, z.size, z.size)] if z.size else z
        bs = b[rng.integers(0, b.size, b.size)] if b.size else b
        zh = float(rng.poisson(np.clip(zs, 0, None)).sum()) if zs.size else 0.0
        bh = float(rng.poisson(np.clip(bs, 0, None)).sum()) if bs.size else 0.0
        if bh <= 0:
            continue
        Rs[k] = zh / bh
        k += 1
    Rs = Rs[:k]
    if k >= 100:
        out["n_boot_effective"] = int(k)
        out["ci95_bootstrap"] = [float(np.percentile(Rs, 2.5)), float(np.percentile(Rs, 97.5))]
    else:
        out["n_boot_effective"] = int(k)
        out["ci95_bootstrap"] = None
    # 两法交叉核对（M10 L40 三律之③）
    cl, cb = out.get("ci95_lognormal"), out.get("ci95_bootstrap")
    if cl and cb and cl[1] > 0:
        out["ci_upper_rel_diff"] = float(abs(cb[1] - cl[1]) / max(abs(cl[1]), 1e-12))
        out["ci_upper_conservative"] = max(cl[1], cb[1])
    return out


def _spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """无 scipy 依赖的 Spearman ρ（秩相关；并列取平均秩）。"""
    if len(x) < 3 or len(x) != len(y):
        return None
    def rank(v):
        v = np.asarray(v, dtype=np.float64)
        order = np.argsort(v, kind="mergesort")
        r = np.empty(v.size, dtype=np.float64)
        r[order] = np.arange(v.size, dtype=np.float64)
        # 平均秩处理并列
        sv = v[order]
        i = 0
        while i < sv.size:
            j = i
            while j + 1 < sv.size and sv[j + 1] == sv[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


# ===========================================================================
# 评估主体
# ===========================================================================
class Evaluator:
    def __init__(self, tt: TargetsTable, proto: Dict[str, Any], proto_path: str, args):
        self.tt = tt
        self.proto = proto
        self.proto_path = proto_path
        self.args = args
        self.dt_ms = float(args.dt_ms if args.dt_ms is not None else proto.get("dt_ms"))
        self.settle_ms = float(proto.get("settle_ms"))
        self.bin_ms = float(proto.get("bin_ms", tt.num("bin_ms_binwise") or 100.0))
        self.smooth_frames = proto.get("smooth_frames", tt.num("smooth_frames_state"))
        self.estimator_primary = str(proto.get("estimator") or tt.raw("estimator_primary"))
        self.silent_thr_hz = float(tt.num("silent_def_m10_compat_hz") or 0.5)
        base_dir = os.path.dirname(os.path.abspath(proto_path))
        default_n = proto.get("n_neuron")
        self.arms: List[Arm] = []
        for d in proto.get("arms") or []:
            self.arms.append(Arm(d, base_dir, default_n))
        self.rows: List[Dict[str, Any]] = []
        self.hard_failures: List[str] = []
        self.arm_metrics: Dict[str, Dict[str, Any]] = {}
        self.duration_ladder: Dict[str, Any] = {}
        self.tension: Dict[str, Any] = {}

    # ---------------- 协议前缀（口径字段，逐行携带） ----------------
    def _proto_fields(self, arm: Optional[Arm]) -> Dict[str, Any]:
        return {
            "protocol_id": self.proto.get("protocol_id", "?"),
            "protocol_T_ms": None if arm is None else arm.T_ms,
            "protocol_settle_ms": None if arm is None else (arm.settle_ms if arm.settle_ms is not None else self.settle_ms),
            "protocol_seed": None if arm is None else arm.seed,
            "protocol_dt_ms": self.dt_ms,
            "protocol_bin_ms": self.bin_ms,
            "protocol_smooth_frames": self.smooth_frames,
        }

    def _emit(self, *, config_id: str, T_ms: float, mechanism: str, criterion: str, estimator: str,
              value: Any, unit: str, status: str, reason: str, band_lo: Any = None, band_hi: Any = None,
              band_keys: str = "", ci_lo: Any = None, ci_hi: Any = None, n_seeds: Any = None,
              zero_spikes: Any = None, n_neuron: Any = None, evidence_class: str = "working_point_precondition",
              discriminative: Any = None, baseline_value: Any = None, baseline_meets: Any = None,
              is_primary: int = 1, note: str = "", arm: Optional[Arm] = None) -> None:
        assert status in STATES, status
        row = {"config_id": config_id, "T_ms": T_ms, "mechanism": mechanism, "criterion": criterion,
               "estimator": estimator, "value": value, "unit": unit, "band_lo": band_lo, "band_hi": band_hi,
               "band_keys": band_keys, "ci_lo": ci_lo, "ci_hi": ci_hi, "status": status, "reason": reason,
               "evidence_class": evidence_class, "discriminative": discriminative,
               "baseline_value": baseline_value, "baseline_meets": baseline_meets,
               "is_primary_estimator": is_primary, "n_seeds": n_seeds, "zero_drive_spikes": zero_spikes,
               "n_neuron": n_neuron, "note": note}
        row.update(self._proto_fields(arm))
        self.rows.append(row)

    # ---------------- 主流程 ----------------
    def run(self) -> Dict[str, Any]:
        for a in self.arms:
            self.arm_metrics[a.arm_id] = arm_metrics(a, self.dt_ms, self.bin_ms, self.silent_thr_hz)
            for hf in self.arm_metrics[a.arm_id]["hard_fail"]:
                self.hard_failures.append(f"{a.arm_id}: {hf}")
        units = sorted({(a.config_id, a.T_ms, a.mechanism) for a in self.arms},
                       key=lambda t: (t[0], t[1], t[2]))
        for (cfg, T_ms, mech) in units:
            self._eval_unit(cfg, T_ms, mech)
        self._duration_ladder()
        self._tension()
        self._apply_duration_flip()
        return self._verdict()

    def _unit_arms(self, cfg: str, T_ms: float, mech: str, role: Optional[str] = None) -> List[Arm]:
        out = [a for a in self.arms if a.config_id == cfg and abs(a.T_ms - T_ms) < 1e-9 and a.mechanism == mech]
        if role is not None:
            out = [a for a in out if a.role == role]
        return out

    # ---- 基线臂（无机制）解析（硬门⑤）----
    def _baseline_arm_value(self, cfg: str, T_ms: float, criterion: str, estimator: str) -> Dict[str, Any]:
        """找「无机制基线臂」在同一 (config,T) 下的同一判据值；找不到即 unavailable。"""
        cand = [a for a in self.arms if a.config_id == cfg and abs(a.T_ms - T_ms) < 1e-9
                and a.mechanism == "none" and a.role in ("baseline", "dose", "zero_drive")]
        cand = [a for a in cand if abs(a.drive_mult - 1.0) < 1e-9]
        if not cand:
            allnone = [a for a in self.arms if a.config_id == cfg and abs(a.T_ms - T_ms) < 1e-9
                       and a.mechanism == "none"]
            if not allnone:
                return {"available": False, "reason": "no_baseline_arm(mechanism=none 且 drive_mult=1 的臂缺失)"}
            cand = allnone
        m = self.arm_metrics[cand[0].arm_id]
        if criterion == CRIT["a"]:
            v = m.get("rate_median_hz") if estimator == "per_neuron" else m.get("trace_bin_median_rate_hz")
        elif criterion == CRIT["b"]:
            v = m.get("silent_frac_zero_count") if estimator == "per_neuron" else m.get("silent_bin_frac")
        elif criterion == CRIT["e"]:
            v = m.get("ei_ratio")
        else:
            v = None
        return {"available": v is not None, "value": v, "arm_id": cand[0].arm_id,
                "reason": "" if v is not None else "baseline_arm_metric_unavailable"}

    def _eval_unit(self, cfg: str, T_ms: float, mech: str) -> None:
        T_s = T_ms / 1000.0
        base_arms = self._unit_arms(cfg, T_ms, mech, "baseline")
        zero_arms = self._unit_arms(cfg, T_ms, mech, "zero_drive")
        dose_arms = self._unit_arms(cfg, T_ms, mech, "dose")
        n_neuron = next((a.n_neuron for a in (base_arms + zero_arms) if a.n_neuron), None)

        # ---------------- (a) 中位率 / (b) 静默比例 ----------------
        est_list = [(self.estimator_primary, 1)]
        if self.estimator_primary != "population_trace":
            est_list.append(("population_trace", 0))
        for est, primary in est_list:
            if est == "per_neuron":
                vals = [self.arm_metrics[a.arm_id].get("rate_median_hz") for a in base_arms]
                vals = [v for v in vals if v is not None]
                stat, unit, estimator = (float(np.median(vals)) if vals else None), "Hz", "per_neuron"
            elif est == "population_trace":
                vals = [self.arm_metrics[a.arm_id].get("trace_bin_median_rate_hz") for a in base_arms]
                vals = [v for v in vals if v is not None]
                stat, unit, estimator = (float(np.median(vals)) if vals else None), "Hz", "population_trace(bin_median)"
            else:
                stat, unit, estimator = None, "Hz", str(est)
            lo, hi, bstat, breasons = self.tt.band("rate_median_hz_lo", "rate_median_hz_hi")
            bl = self._baseline_arm_value(cfg, T_ms, CRIT["a"], "per_neuron" if est == "per_neuron" else "population_trace")
            st, rs = self._judge(stat, lo, hi, bstat, breasons, arm_present=bool(base_arms))
            self._emit(config_id=cfg, T_ms=T_ms, mechanism=mech, criterion=CRIT["a"], estimator=estimator,
                       value=stat, unit=unit, status=st, reason=rs, band_lo=lo, band_hi=hi,
                       band_keys="rate_median_hz_lo|rate_median_hz_hi", is_primary=primary,
                       discriminative=(None if not bl["available"] else (not _meets(bl["value"], lo, hi))),
                       baseline_value=bl.get("value"), baseline_meets=(None if not bl["available"] else _meets(bl["value"], lo, hi)),
                       n_seeds=len({a.seed for a in base_arms if a.seed is not None}) or None,
                       n_neuron=n_neuron, note=("主口径：逐神经元计数/T 的中位数（清单 §4.5(a)）" if primary
                                                else "并列口径：群体轨迹逐箱率的中位数（描述性交叉核对）"),
                       arm=(base_arms[0] if base_arms else None))

            s_est = "per_neuron" if est == "per_neuron" else "population_trace"
            if s_est == "per_neuron":
                svals = [self.arm_metrics[a.arm_id].get("silent_frac_zero_count") for a in base_arms]
                svals = [v for v in svals if v is not None]
                sstat, sunit = (float(np.median(svals)) if svals else None), ""
            else:
                svals = [self.arm_metrics[a.arm_id].get("silent_bin_frac") for a in base_arms]
                svals = [v for v in svals if v is not None]
                sstat, sunit = (float(np.median(svals)) if svals else None), ""
            slo, shi, sbstat, sbreasons = self.tt.band("silent_frac_lo", "silent_frac_hi")
            sbl = self._baseline_arm_value(cfg, T_ms, CRIT["b"], s_est)
            st2, rs2 = self._judge(sstat, slo, shi, sbstat, sbreasons, arm_present=bool(base_arms),
                                  extra=(f"口径={self.tt.raw('silent_def_primary')}(主)/<{self.silent_thr_hz} Hz(M10 兼容)"
                                         if est == "per_neuron" else "口径=零脉冲箱比例（群体轨迹）"))
            self._emit(config_id=cfg, T_ms=T_ms, mechanism=mech, criterion=CRIT["b"],
                       estimator=("per_neuron(zero_count)" if est == "per_neuron" else "population_trace(silent_bin)"),
                       value=sstat, unit=sunit, status=st2, reason=rs2, band_lo=slo, band_hi=shi,
                       band_keys="silent_frac_lo|silent_frac_hi", is_primary=primary,
                       discriminative=(None if not sbl["available"] else (not _meets(sbl["value"], slo, shi))),
                       baseline_value=sbl.get("value"), baseline_meets=(None if not sbl["available"] else _meets(sbl["value"], slo, shi)),
                       n_seeds=len({a.seed for a in base_arms if a.seed is not None}) or None,
                       n_neuron=n_neuron,
                       note=("M11 主静默口径 = 计数==0 的神经元比例；与 M10 的 rate<0.5 Hz 口径不同（见判据表 silent_def_*）"
                             if est == "per_neuron" else "并列口径（描述性）"),
                       arm=(base_arms[0] if base_arms else None))

        # ---------------- (c) 自持性（硬门①②④） ----------------
        self._eval_selfsustain(cfg, T_ms, mech, zero_arms, base_arms, n_neuron)

        # ---------------- (d) 驱动弹性 ----------------
        self._eval_elasticity(cfg, T_ms, mech, base_arms, zero_arms, dose_arms, n_neuron)

        # ---------------- (e) E/I 平衡 ----------------
        self._eval_ei(cfg, T_ms, mech, base_arms + zero_arms + dose_arms, n_neuron)

        # ---------------- (f)(g)(h)(i) ----------------
        self._eval_engineering(cfg, T_ms, mech, base_arms + zero_arms + dose_arms, n_neuron)

    # ---- 统一判定：band 未定稿/无臂 ⇒ unavailable；无量 ⇒ unavailable ----
    @staticmethod
    def _judge(value: Optional[float], lo: Optional[float], hi: Optional[float], band_status: str,
               band_reasons: Sequence[str], arm_present: bool, extra: str = "") -> Tuple[str, str]:
        if not arm_present:
            return "unavailable", "arm_missing(该 role 的臂未提供)"
        if band_status != "ok" or lo is None or hi is None:
            return "unavailable", "band_not_finalized: " + "; ".join(band_reasons)
        if value is None:
            return "unavailable", "value_unavailable(所需输入缺失，例如缺逐神经元 counts)"
        ok = (lo <= value <= hi)
        return ("pass" if ok else "fail"), (f"band=[{lo},{hi}] " + (extra or ""))

    def _eval_selfsustain(self, cfg: str, T_ms: float, mech: str, zero_arms: List[Arm],
                          base_arms: List[Arm], n_neuron: Optional[int]) -> None:
        lim = self.tt.num("selfsustain_ratio_max")
        factor = self.tt.num("selfsustain_ci_upper_factor")
        min_seeds = int(self.tt.num("protocol_seeds_min_per_arm") or 3)
        min_spk = float(self.tt.num("selfsustain_zero_drive_min_spikes") or 300)
        tol_ci = float(self.tt.num("selfsustain_ci_crosscheck_rel_tol") or 0.25)
        Z = [self.arm_metrics[a.arm_id].get("counts_sum") if self.arm_metrics[a.arm_id].get("counts_sum") is not None
             else self.arm_metrics[a.arm_id].get("trace_sum") for a in zero_arms]
        B = [self.arm_metrics[a.arm_id].get("counts_sum") if self.arm_metrics[a.arm_id].get("counts_sum") is not None
             else self.arm_metrics[a.arm_id].get("trace_sum") for a in base_arms]
        Z = [float(x) for x in Z if x is not None]
        B = [float(x) for x in B if x is not None]
        nz = len({a.seed for a in zero_arms if a.seed is not None})
        nb = len({a.seed for a in base_arms if a.seed is not None})
        ci = selfsustain_ci(Z, B) if (Z and B) else {}
        R = ci.get("ratio_pooled")
        r = (f"点估计 R={R:.6g}（上限 {lim}）；CI95={ci.get('ci95_lognormal')}；"
             f"零驱动脉冲 {ci.get('zero_spikes')}；seed 数 零={nz}/基准={nb}") if R is not None else "臂缺失或基准臂 0 脉冲"
        # ---- 硬门① 样本量门 ----
        gate_fail = []
        if not zero_arms or not base_arms:
            gate_fail.append("zero_drive/baseline 臂缺失")
        if nz < min_seeds:
            gate_fail.append(f"零驱动臂 seed 数 {nz} < {min_seeds}")
        if nb < min_seeds:
            gate_fail.append(f"基准臂 seed 数 {nb} < {min_seeds}")
        if ci.get("zero_spikes", 0.0) < min_spk:
            gate_fail.append(f"零驱动臂总脉冲 {ci.get('zero_spikes')} < {min_spk}"
                             f"（相对不确定度 ≈{100/math.sqrt(max(ci.get('zero_spikes',1),1)):.1f}%）")
        # ---- 硬门② 两 CI 法一致性 ----
        ci_disagree = (ci.get("ci_upper_rel_diff") is not None and ci["ci_upper_rel_diff"] > tol_ci)
        if gate_fail:
            status = "unavailable"
            reason = "sample_size_gate_unmet: " + "; ".join(gate_fail) + f"；点估计仅作描述性（{r}）"
        elif ci.get("ci95_lognormal") is None:
            status, reason = "unavailable", "ci_unavailable"
        elif ci_disagree:
            status = "invalid"
            reason = (f"two_ci_methods_disagree: 上界相对差 {ci['ci_upper_rel_diff']:.3f} > {tol_ci}"
                      f"（对数正态 {ci['ci95_lognormal'][1]:.6g} vs 自助 {ci['ci95_bootstrap'][1]:.6g}）⇒ 先对齐口径")
        else:
            up = ci.get("ci_upper_conservative", ci["ci95_lognormal"][1])
            ok_pt = (lim is not None) and (R <= lim)
            ok_ci = (lim is not None) and (factor is not None) and (up <= lim * factor)
            if ok_pt and ok_ci:
                status, reason = "pass", r + f"；判定通过（保守 CI 上界 {up:.6g} ≤ {lim * factor:.6g}）"
            else:
                tag = "统计未分辨（点估计未过，但 CI 上界未跨越 ×%.1f 带）" % (factor or 0) if (not ok_pt and ok_ci) else ""
                status, reason = "fail", r + f"；判定未过（点估计≤上限={ok_pt}、CI 上界≤{lim * factor if lim else None}={ok_ci}）{tag}"
        bl = self._baseline_arm_value(cfg, T_ms, CRIT["c"], "per_neuron")
        self._emit(config_id=cfg, T_ms=T_ms, mechanism=mech, criterion=CRIT["c"],
                   estimator="per_neuron(pooled_counts_ratio)",
                   value=R, unit="", status=status, reason=reason,
                   band_lo=lim, band_hi=None, band_keys="selfsustain_ratio_max",
                   ci_lo=(ci.get("ci95_lognormal") or [None, None])[0],
                   ci_hi=(ci.get("ci95_lognormal") or [None, None])[1],
                   n_seeds=nz, zero_spikes=ci.get("zero_spikes"), n_neuron=n_neuron,
                   note=("硬门①（≥3 seed/臂、零驱动 ≥300 脉冲）与硬门②（点估计 ≤ 上限 且 CI 上界 ≤ 上限×%.1f）"
                         "已机械执行；两 CI 法（泊松对数正态 / seed 自助）并列报告" % (factor or 0)),
                   baseline_value=None, baseline_meets=None,
                   arm=(zero_arms[0] if zero_arms else None))
        self.rows[-1]["ci_lo_bootstrap"] = (ci.get("ci95_bootstrap") or [None, None])[0]
        self.rows[-1]["ci_hi_bootstrap"] = (ci.get("ci95_bootstrap") or [None, None])[1]
        self.rows[-1]["per_seed_ratio"] = ci.get("per_seed_ratio")
        self.rows[-1]["selfsustain_detail"] = {k: v for k, v in ci.items() if k != "per_seed_ratio"}

    def _eval_elasticity(self, cfg: str, T_ms: float, mech: str, base_arms: List[Arm], zero_arms: List[Arm],
                         dose_arms: List[Arm], n_neuron: Optional[int]) -> None:
        grid = self.tt.as_list("protocol_drive_mult_grid", float)
        lo_lim = self.tt.num("drive_elasticity_min")
        reg = self.tt.num("criterion_d_as_mechanism_evidence")
        allarms = base_arms + dose_arms + zero_arms
        def rate_of(a: Arm) -> Optional[float]:
            m = self.arm_metrics[a.arm_id]
            if m.get("counts_sum") is not None and a.n_neuron:
                return m["counts_sum"] / a.n_neuron / (a.T_ms / 1000.0)
            if m.get("trace_sum") is not None and a.n_neuron:
                return m["trace_sum"] / a.n_neuron / (a.T_ms / 1000.0)
            return None
        pts = sorted({(a.drive_mult, rate_of(a)) for a in allarms if rate_of(a) is not None},
                     key=lambda t: t[0])
        d = dict(pts)
        r1, r4 = d.get(1.0), d.get(4.0)
        monotone = all(b >= a - 1e-12 for (_, a), (_, b) in zip(pts, pts[1:])) if len(pts) >= 2 else None
        if r1 is None or r4 is None or r1 <= 0:
            status, reason, val = "unavailable", "dose_arm_missing(需 drive_mult∈{1,4} 且基准群率>0)", None
        elif lo_lim is None:
            status, reason, val = "unavailable", "band_not_finalized: drive_elasticity_min", None
        else:
            val = (r4 - r1) / r1
            ok = (val >= lo_lim) and bool(monotone)
            status = "pass" if ok else "fail"
            reason = (f"4× 增幅={val:+.4f}（下限 {lo_lim}）、单调={monotone}、剂量点={[p[0] for p in pts]}"
                      + ("；【登记条款】该判据通过**不得**作为机制有效性证据"
                         if (reg == 0) else ""))
        self._emit(config_id=cfg, T_ms=T_ms, mechanism=mech, criterion=CRIT["d"],
                   estimator="per_neuron(pop_mean_rate)",
                   value=val, unit="", status=status, reason=reason, band_lo=lo_lim, band_hi=None,
                   band_keys="drive_elasticity_min", n_seeds=len({a.seed for a in allarms if a.seed is not None}) or None,
                   n_neuron=n_neuron, evidence_class="sanity_precondition(non_discriminative)",
                   discriminative=(False if reg == 0 else None),
                   baseline_value=None, baseline_meets=None,
                   note=(f"剂量-响应 {grid}；判据表登记 criterion_d_as_mechanism_evidence={reg} ⇒ "
                         "基线自身即过该判据（M10 实测 +579.6%），故它在本工具中被标为非判别性 sanity 前置"),
                   arm=(base_arms[0] if base_arms else None))

    def _eval_ei(self, cfg: str, T_ms: float, mech: str, arms: List[Arm], n_neuron: Optional[int]) -> None:
        lo, hi, bstat, breasons = self.tt.band("ei_ratio_lo", "ei_ratio_hi")
        vals = [self.arm_metrics[a.arm_id].get("ei_ratio") for a in arms]
        vals = [v for v in vals if v is not None]
        val = float(np.median(vals)) if vals else None
        bl = self._baseline_arm_value(cfg, T_ms, CRIT["e"], "per_neuron")
        if not arms:
            status, reason = "unavailable", "arm_missing"
        elif not vals:
            status, reason = "unavailable", ("ei_input_unavailable: 臂内未注入 i_exc_integral/i_inh_integral"
                                            "（本工具不从发放率反推电流；(e) 的唯一来源是仿真侧注入）")
        else:
            status, reason = self._judge(val, lo, hi, bstat, breasons, True,
                                        extra="定义=i_inh_integral/i_exc_integral（判据表 ei_ratio_definition）")
        self._emit(config_id=cfg, T_ms=T_ms, mechanism=mech, criterion=CRIT["e"],
                   estimator="synaptic_current_integral_ratio",
                   value=val, unit="", status=status, reason=reason, band_lo=lo, band_hi=hi,
                   band_keys="ei_ratio_lo|ei_ratio_hi", n_neuron=n_neuron,
                   discriminative=(None if not bl["available"] else (not _meets(bl["value"], lo, hi))),
                   baseline_value=bl.get("value"), baseline_meets=(None if not bl["available"] else _meets(bl["value"], lo, hi)),
                   note="数值端点为预注册设计选择（derived）；四篇文献仅支撑方向与量级（判据表 ei_ratio_* 行）",
                   arm=(arms[0] if arms else None))

    def _eval_engineering(self, cfg: str, T_ms: float, mech: str, arms: List[Arm],
                          n_neuron: Optional[int]) -> None:
        # (f) 有界性
        p95max = self.tt.num("rate_p95_max_hz")
        nonfinite = []
        p95s = []
        for a in arms:
            m = self.arm_metrics[a.arm_id]
            if m.get("trace_finite") is False or m.get("counts_finite") is False:
                nonfinite.append(a.arm_id)
            if m.get("rate_p95_hz") is not None:
                p95s.append(m["rate_p95_hz"])
        val = float(np.max(p95s)) if p95s else None
        if not arms:
            st, rs = "unavailable", "arm_missing"
        elif nonfinite:
            st, rs = "invalid", f"non_finite_values: {nonfinite}"
        elif val is None:
            st, rs = "unavailable", "rate_p95_unavailable（缺逐神经元 counts）"
        elif val > p95max:
            st, rs = "fail", f"per-neuron rate p95 {val:.4g} > sanity 上界 {p95max} Hz"
        else:
            st, rs = "pass", f"无非有限值；率 p95 {val:.4g} ≤ {p95max} Hz"
        self._emit(config_id=cfg, T_ms=T_ms, mechanism=mech, criterion=CRIT["f"], estimator="per_neuron",
                   value=val, unit="Hz", status=st, reason=rs, band_lo=None, band_hi=p95max,
                   band_keys="rate_p95_max_hz", n_neuron=n_neuron, evidence_class="engineering_sanity",
                   arm=(arms[0] if arms else None))

        # (g) 单试次墙钟
        wmax = self.tt.num("single_trial_wall_s_max")
        gwmax = self.tt.num("single_trial_gpu_h_max")
        ws = [self.arm_metrics[a.arm_id].get("wall_s") for a in arms]
        ws = [w for w in ws if w is not None]
        val = float(np.max(ws)) if ws else None
        if not ws:
            st, rs = "unavailable", "wall_s 未注入（(g) 依赖仿真侧时钟读数）"
        else:
            ok_final, why = self.tt.finalized("single_trial_wall_s_max")
            if not ok_final:
                st = "unavailable"; rs = f"band_not_finalized: {why}；实测墙钟上界 {val:.4g} s（描述性，需 G0 探针定稿）"
            elif val <= wmax:
                st, rs = "pass", f"实测 {val:.4g} s ≤ {wmax} s"
            else:
                st, rs = "fail", f"实测 {val:.4g} s > {wmax} s"
        self._emit(config_id=cfg, T_ms=T_ms, mechanism=mech, criterion=CRIT["g"], estimator="wall_clock",
                   value=val, unit="s", status=st, reason=rs, band_lo=None, band_hi=wmax,
                   band_keys="single_trial_wall_s_max|single_trial_gpu_h_max", n_neuron=n_neuron,
                   evidence_class="budget", note=f"GPU-h 上限 {gwmax}（草案 final=0，探针后定稿）",
                   arm=(arms[0] if arms else None))

        # (h) 确定性
        reps = [a for a in self.arms if a.replica_of]
        detail = []
        for a in reps:
            tgt = next((x for x in self.arms if x.arm_id == a.replica_of), None)
            if tgt is None:
                detail.append({"arm": a.arm_id, "status": "invalid", "reason": "replica_of 指向不存在的臂"})
                continue
            sa = self.arm_metrics[a.arm_id].get("counts_sum", self.arm_metrics[a.arm_id].get("trace_sum"))
            sb = self.arm_metrics[tgt.arm_id].get("counts_sum", self.arm_metrics[tgt.arm_id].get("trace_sum"))
            detail.append({"arm": a.arm_id, "vs": tgt.arm_id, "spk_a": sa, "spk_b": sb,
                           "identical": bool(sa == sb) if (sa is not None and sb is not None) else None})
        if not reps:
            st, rs = "unavailable", "no_replica_arm（(h) 要求同参数重复臂；本工具只比较，不生成）"
        elif any(d.get("identical") is False for d in detail):
            st = "fail"
            rs = f"同参数重跑脉冲数不一致：{['%s(%s vs %s)' % (d['arm'], d.get('spk_a'), d.get('spk_b')) for d in detail if d.get('identical') is False]}"
        elif all(d.get("identical") is True for d in detail):
            st, rs = "pass", "同参数重复臂逐值一致（统计级）"
        else:
            st, rs = "unavailable", f"replica 比较不可完成：{detail}"
        self._emit(config_id=cfg, T_ms=T_ms, mechanism=mech, criterion=CRIT["h"], estimator="spk_total_identity",
                   value=(int(all(d.get("identical") is True for d in detail)) if detail and reps else None),
                   unit="bool", status=st, reason=rs, band_keys="determinism_mode", n_neuron=n_neuron,
                   evidence_class="engineering_sanity", arm=(arms[0] if arms else None))

        # (i) 标定权衡：(a) ∧ (b)
        a_rows = [r for r in self.rows if r["config_id"] == cfg and abs(r["T_ms"] - T_ms) < 1e-9
                  and r["mechanism"] == mech and r["criterion"] == CRIT["a"] and r["is_primary_estimator"] == 1]
        b_rows = [r for r in self.rows if r["config_id"] == cfg and abs(r["T_ms"] - T_ms) < 1e-9
                  and r["mechanism"] == mech and r["criterion"] == CRIT["b"] and r["is_primary_estimator"] == 1]
        sa = a_rows[-1]["status"] if a_rows else "unavailable"
        sb = b_rows[-1]["status"] if b_rows else "unavailable"
        if sa == "pass" and sb == "pass":
            st, rs = "pass", "(a)∧(b) 同时成立"
        elif "unavailable" in (sa, sb):
            st = "unavailable"
            rs = f"(a)={sa}、(b)={sb}；{'带未定稿或量缺失' if True else ''} ⇒ 标定权衡不可判（不得静默放宽，清单 §4.5(i)）"
        else:
            st = "fail"
            rs = f"(a)={sa}、(b)={sb} ⇒ **标定权衡**：登记为 P6 的输入约束（不得静默放宽带）"
        self._emit(config_id=cfg, T_ms=T_ms, mechanism=mech, criterion=CRIT["i"], estimator="per_neuron",
                   value=None, unit="bool", status=st, reason=rs,
                   band_keys="calibration_tradeoff_requires_a_and_b|tension_detect_required",
                   n_neuron=n_neuron, evidence_class="precondition_conjunction",
                   note=f"子判据状态：(a)={sa}、(b)={sb}", arm=(arms[0] if arms else None))

    # ---------------- 硬门④ 时长依赖性 ----------------
    def _duration_ladder(self) -> None:
        ratios = self.tt.as_list("protocol_T_ladder_ratios", float)
        for cfg in sorted({a.config_id for a in self.arms}):
            Ts = sorted({a.T_ms for a in self.arms if a.config_id == cfg})
            if not Ts:
                continue
            Tmin = Ts[0]
            need = [Tmin * r for r in ratios] if ratios else [Tmin]
            present = [t for t in need if any(abs(t - x) < 1e-9 for x in Ts)]
            per: Dict[str, Dict[str, Any]] = {}
            for t in Ts:
                for crit in (CRIT["a"], CRIT["b"], CRIT["c"], CRIT["d"], CRIT["i"]):
                    rs = [r for r in self.rows if r["config_id"] == cfg and abs(r["T_ms"] - t) < 1e-9
                          and r["criterion"] == crit and (r["is_primary_estimator"] == 1)]
                    if rs:
                        per.setdefault(crit, {})[f"T={t:g}ms"] = {"value": rs[-1]["value"], "status": rs[-1]["status"]}
            drift: Dict[str, Any] = {}
            for crit, d in per.items():
                xs = sorted(d.keys(), key=lambda k: float(k[2:-2]))
                if len(xs) < 2:
                    continue
                v0, v1 = d[xs[0]]["value"], d[xs[-1]]["value"]
                tv = [float(k[2:-2]) for k in xs]
                vv = [d[k]["value"] for k in xs]
                drift[crit] = {
                    "values_by_T": d,
                    "rel_change_first_to_last": (None if not (isinstance(v0, (int, float)) and isinstance(v1, (int, float)) and v0)
                                                 else float((v1 - v0) / v0)),
                    "spearman_vs_T": _spearman(tv, [x if isinstance(x, (int, float)) else float("nan") for x in vv]),
                    "status_flip": len({d[k]["status"] for k in xs}) > 1,
                    "statuses": {k: d[k]["status"] for k in xs},
                }
            complete = len(present) == len(need) and len(need) >= 3
            # ★ 截断构造守卫：跨度阶梯必须来自**逐档独立重跑**；若短轨迹是长轨迹的逐位前缀
            #   ⇒ 该阶梯是「同一次运行的截断」，测的是窗口长度而非协议时长 ⇒ invalid
            trunc = []
            for t1 in Ts:
                for t2 in Ts:
                    if t2 <= t1:
                        continue
                    a1s = [a for a in self.arms if a.config_id == cfg and abs(a.T_ms - t1) < 1e-9 and a.trace is not None]
                    a2s = [a for a in self.arms if a.config_id == cfg and abs(a.T_ms - t2) < 1e-9 and a.trace is not None]
                    for x in a1s:
                        for y in a2s:
                            n = x.trace.size
                            if y.trace.size >= n and n > 0 and np.array_equal(x.trace, y.trace[:n]):
                                trunc.append({"short": x.arm_id, "long": y.arm_id,
                                              "T_short_ms": t1, "T_long_ms": t2})
            if trunc:
                status = "invalid"
                reason = ("ladder_arms_look_truncated: 短时长臂的轨迹是长时长臂的**逐位前缀** %s ⇒ "
                          "阶梯由同一次运行的截断构造，测的是窗口长度而非协议时长（settle/初值/RNG 语义不满足）"
                          "⇒ 硬门④ 无效；必须逐档独立重跑" % trunc)
            elif complete:
                status, reason = "pass", ""
            else:
                status, reason = "unavailable", (
                    f"硬门④：需至少 {ratios}×T_min={need} 三档同一配置的臂；实有 T={Ts} ⇒ 时长依赖性**未测**"
                    "（判据不得就此定稿）")
            self.duration_ladder[cfg] = {
                "T_levels_present_ms": Ts, "ladder_required_ms": need, "ladder_present_ms": present,
                "ladder_complete": bool(complete), "per_criterion": per, "drift": drift,
                "truncation_suspicion": trunc, "status": status, "reason": reason,
            }


    def _apply_duration_flip(self) -> None:
        """跨 T 判定翻转 ⇒ 该子判据 invalid（协议时长绑定失败，清单 §0.7 #8 + M10 A12）。"""
        for cfg, blk in self.duration_ladder.items():
            for crit, d in blk["drift"].items():
                if d.get("status_flip"):
                    for r in self.rows:
                        if r["config_id"] == cfg and r["criterion"] == crit and r["status"] in ("pass", "fail"):
                            r["status"] = "invalid"
                            r["reason"] = (f"duration_dependent_verdict_flip: 同一配置跨 T 的判定翻转 "
                                           f"{d['statuses']}（硬门④；M10 A12 实测自持比 T=2/60/120 s = "
                                           f"0.0980/0.1082/0.1720）⇒ 该子判据随协议时长漂移，不构成可判判据") + \
                                          "；" + str(r["reason"])
                            self.hard_failures.append(f"{cfg}/{crit}: duration_flip")

    # ---------------- 张力识别（中位率 vs 静默） ----------------
    def _tension(self) -> None:
        # 用剂量-响应臂（或同机制的 baseline+dose）构造 (驱动, 中位率, 静默) 三元组
        rho_max = self.tt.num("tension_spearman_max")
        for cfg in sorted({a.config_id for a in self.arms}):
            for mech in sorted({a.mechanism for a in self.arms if a.config_id == cfg}):
                cand = [a for a in self.arms if a.config_id == cfg and a.mechanism == mech
                        and a.role in ("baseline", "dose", "zero_drive")]
                Ts = sorted({a.T_ms for a in cand})
                for T in Ts:
                    sub = [a for a in cand if abs(a.T_ms - T) < 1e-9]
                    pts = []
                    for a in sorted(sub, key=lambda x: x.drive_mult):
                        m = self.arm_metrics[a.arm_id]
                        med = m.get("rate_median_hz")
                        sil = m.get("silent_frac_zero_count")
                        if med is None or sil is None:
                            continue
                        pts.append({"drive_mult": a.drive_mult, "rate_median_hz": med, "silent_frac": sil,
                                    "arm_id": a.arm_id})
                    key = f"{cfg}@{T:g}ms|mech={mech}"
                    if len(pts) < 3:
                        self.tension[key] = {"status": "unavailable",
                                             "reason": "需 ≥3 个剂量点（含 zero_drive）才能识别张力；"
                                                       f"现 {len(pts)} 点（另需逐神经元 counts 才有中位率/静默）"}
                        continue
                    lo_a, hi_a, sa_, _ = self.tt.band("rate_median_hz_lo", "rate_median_hz_hi")
                    lo_b, hi_b, sb_, _ = self.tt.band("silent_frac_lo", "silent_frac_hi")
                    n_both = sum(1 for p in pts if (lo_a is not None and lo_a <= p["rate_median_hz"] <= hi_a)
                                 and (lo_b is not None and lo_b <= p["silent_frac"] <= hi_b))
                    rho = _spearman([p["drive_mult"] for p in pts], [p["rate_median_hz"] for p in pts])
                    rho_s = _spearman([p["drive_mult"] for p in pts], [p["silent_frac"] for p in pts])
                    pos = [p for p in pts if p["rate_median_hz"] > 0.0]
                    first_pos = pos[0] if pos else None
                    rec = {"status": "ok" if (sa_ == "ok" and sb_ == "ok") else "unavailable",
                           "reason": "" if (sa_ == "ok" and sb_ == "ok") else
                                     "(a) 或 (b) 的带未定稿 ⇒ 张力只能定性",
                           "points": pts, "n_points": len(pts),
                           "n_both_in_band": int(n_both),
                           "spearman_drive_vs_rate": rho, "spearman_drive_vs_silent": rho_s,
                           "first_drive_with_median_rate_gt0": first_pos,
                           "structural_tension": bool(
                               n_both == 0 and first_pos is not None and rho is not None and rho_s is not None
                               and rho >= 0.8 and (rho_max is not None and rho_s <= rho_max)),
                           }
                    if rec["structural_tension"]:
                        rec["consequence"] = ("存在的剂量点各自能落带、但**无一剂量同时满足 (a)∧(b)**，"
                                             "且驱动↑ 使中位率↑、静默↓ 单调对立 ⇒ 判为**结构性标定张力**："
                                             "须换杠杆（E/I 比例、异质阈值/适应、慢变量分级响应），"
                                             "**不得**静默放宽带（清单 §4.5(i)）")
                    self.tension[key] = rec

    # ---------------- 总判定 ----------------
    def _verdict(self) -> Dict[str, Any]:
        by_crit: Dict[str, List[Dict[str, Any]]] = {}
        for r in self.rows:
            by_crit.setdefault(r["criterion"], []).append(r)
        crit_status: Dict[str, str] = {}
        for c, rs in by_crit.items():
            prim = [r for r in rs if r["is_primary_estimator"] == 1] or rs
            sts = {r["status"] for r in prim}
            if "invalid" in sts:
                crit_status[c] = "invalid"
            elif sts == {"pass"}:
                crit_status[c] = "pass"
            elif "fail" in sts:
                crit_status[c] = "fail"
            else:
                crit_status[c] = "unavailable"
        gate = {}
        for c in GATE_CRITERIA:
            gate[c] = crit_status.get(c, "unavailable")
        # 判别性摘要（硬门⑤）
        non_disc = sorted({r["criterion"] for r in self.rows
                           if r["discriminative"] is False or r["evidence_class"].startswith("sanity_precondition")})
        baseline_ok = all(self._baseline_arm_value(r["config_id"], r["T_ms"], r["criterion"],
                                                  "per_neuron" if r["estimator"].startswith("per_neuron") else "population_trace")["available"]
                          for r in self.rows if r["criterion"] in (CRIT["a"], CRIT["b"]))
        # 总态
        ladder_bad = [cfg for cfg, blk in self.duration_ladder.items() if blk["status"] == "invalid"]
        invalid = bool(self.hard_failures) or "invalid" in gate.values() or bool(ladder_bad)
        if invalid:
            g1, why = "invalid", ("存在硬失败/口径缺陷（见 hard_failures、duration_flip 与 ladder_arms_look_truncated）"
                                  "⇒ 该轮结论无效")
            if ladder_bad:
                self.hard_failures.append(f"duration_ladder_invalid: {ladder_bad}")
        elif all(v == "pass" for v in gate.values()):
            g1, why = "pass", "G1 门条件 (c)∧(d)∧(e) 全过"
        elif any(v == "fail" for v in gate.values()):
            g1, why = "fail", f"G1 门条件未过：{[f'{k}={v}' for k, v in gate.items()]}"
        else:
            g1, why = "unavailable", (f"G1 门条件中存在未测项：{[f'{k}={v}' for k, v in gate.items()]}"
                                     " ⇒ 不得判 pass（未测 ≠ 未命中）")
        if not baseline_ok and g1 == "pass":
            g1, why = "unavailable", "硬门⑤：缺「无机制基线臂」的同一判据值 ⇒ 判别性未登记，整体不得判 pass"
        counts = {s: sum(1 for r in self.rows if r["status"] == s) for s in STATES}
        return {"verdict_G1": g1, "verdict_reason": why, "gate_conditions": gate,
                "criterion_status": crit_status, "state_counts": counts,
                "hard_failures": self.hard_failures,
                "non_discriminative_criteria": non_disc,
                "baseline_arm_registered": bool(baseline_ok),
                "duration_ladder": self.duration_ladder, "tension": self.tension,
                "exit_code": {"pass": EXIT_PASS, "fail": EXIT_FAIL, "invalid": EXIT_INVALID,
                              "unavailable": EXIT_UNAVAILABLE}[g1]}


def _meets(v: Optional[float], lo: Optional[float], hi: Optional[float]) -> bool:
    if v is None or lo is None or hi is None:
        return False
    return bool(lo <= v <= hi)


# ===========================================================================
# 输出
# ===========================================================================
CSV_COLS = ["config_id", "T_ms", "mechanism", "criterion", "estimator", "value", "unit", "band_lo", "band_hi",
            "band_keys", "ci_lo", "ci_hi", "status", "reason", "evidence_class", "discriminative",
            "baseline_value", "baseline_meets", "is_primary_estimator", "n_seeds", "zero_drive_spikes",
            "n_neuron", "protocol_id", "protocol_T_ms", "protocol_settle_ms", "protocol_seed",
            "protocol_dt_ms", "protocol_bin_ms", "protocol_smooth_frames", "note"]


def write_rows_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    extra = [k for r in rows for k in r if k not in CSV_COLS]
    cols = CSV_COLS + sorted(set(extra))
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write("# M11 P5/G1 评估输出（generated_by=tools/validate_p11_resting.py）\n")
        f.write("# 四态：pass/fail/invalid/unavailable；每行带 protocol(T/settle/seed/bin/smooth) 与 estimator 字段\n")
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v)
                        for k, v in r.items()})


def print_summary(out: Dict[str, Any], rows: List[Dict[str, Any]], arm_m: Dict[str, Dict[str, Any]],
                  ladder: Dict[str, Any]) -> None:
    print("=" * 100)
    print(f"M11 P5 / G1 评估：**{out['verdict_G1']}**  exit={out['exit_code']}")
    print(f"  reason: {out['verdict_reason']}")
    print(f"  门条件: {out['gate_conditions']}")
    print(f"  四态计数: {out['state_counts']}")
    if out["hard_failures"]:
        print("  硬失败：")
        for h in out["hard_failures"]:
            print(f"    - {h}")
    print(f"  基线臂登记: {out['baseline_arm_registered']}；非判别性判据: {out['non_discriminative_criteria']}")
    print("-" * 100)
    print(f"{'criterion':<24}{'estimator':<34}{'value':>14}{'status':>12}  reason")
    for r in rows:
        v = r["value"]
        vs = "None" if v is None else (f"{v:.6g}" if isinstance(v, (int, float)) else str(v))
        print(f"{r['criterion']:<24}{str(r['estimator'])[:33]:<34}{vs:>14}{r['status']:>12}  {str(r['reason'])[:80]}")
    print("-" * 100)
    print("臂指标（三估计器）：")
    for aid, m in arm_m.items():
        print(f"  {aid:<34} trace_sum={m.get('trace_sum')} counts_sum={m.get('counts_sum')} "
              f"agg={m.get('agg_spk_total')} est_avail={m.get('estimators_available')} "
              f"rel_diff={m.get('estimator_rel_diffs')} rate_med={m.get('rate_median_hz')} "
              f"silent={m.get('silent_frac_zero_count')}")
    print("-" * 100)
    for cfg, blk in ladder.items():
        print(f"时长阶梯 {cfg}: status={blk['status']} 需={blk['ladder_required_ms']} 有={blk['T_levels_present_ms']}"
              f" complete={blk['ladder_complete']}")
        for crit, d in blk["drift"].items():
            print(f"    {crit}: rel_change={d['rel_change_first_to_last']} spearman_vs_T={d['spearman_vs_T']} "
                  f"flip={d['status_flip']}")
    print("=" * 100)


# ===========================================================================
# 自检 / 夹具
# ===========================================================================
def check_targets(csv_path: str, json_path: str) -> int:
    tt = TargetsTable(csv_path)
    rc = 0
    print(f"判据表：{csv_path}  行数={len(tt.rows)}")
    levels: Dict[str, int] = {}
    finals: Dict[str, int] = {}
    for k in tt.order:
        r = tt.rows[k]
        levels[r["evidence_level"]] = levels.get(r["evidence_level"], 0) + 1
        finals[r["final"]] = finals.get(r["final"], 0) + 1
    print(f"  证据级别分布：{levels}")
    print(f"  final 分布：{finals}")
    proxy = [k for k in tt.order if k.startswith("proxy_")]
    print(f"  跨制备/跨口径代理行：{len(proxy)} → {proxy}")
    noconc = [k for k in tt.order if k.startswith("proxy_") and not (tt.rows[k].get("wrong_usage_consequence") or "").strip()]
    if noconc:
        print(f"  ⚠ 代理行缺 wrong_usage_consequence：{noconc}")
        rc = 1
    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            jd = json.load(f)
        jrows = {r["key"]: r for r in jd.get("rows", [])}
        miss = [k for k in tt.order if k not in jrows]
        extra = [k for k in jrows if k not in tt.rows]
        diff = [k for k in tt.order if k in jrows and str(jrows[k].get("value")) != str(tt.rows[k]["value"])]
        print(f"  CSV ⟺ JSON 互查：csv={len(tt.order)} json={len(jrows)} 缺={miss} 多={extra} value 不一致={diff}")
        if miss or extra or diff:
            rc = 1
    else:
        print(f"  ⚠ JSON 判据表不存在（{json_path}）⇒ 无法互查")
        rc = 1
    # 关键键存在性（G1 门条件依赖）
    for k in ("selfsustain_ratio_max", "selfsustain_ci_upper_factor", "selfsustain_zero_drive_min_spikes",
              "protocol_seeds_min_per_arm", "protocol_T_ladder_ratios", "double_estimator_rel_tol",
              "silent_def_primary", "estimator_primary", "ei_ratio_lo", "ei_ratio_hi", "gate_g1_conditions"):
        try:
            tt.row(k)
        except KeyError:
            print(f"  ✗ 缺关键键：{k}")
            rc = 1
    print("自检结论：" + ("OK" if rc == 0 else "发现问题（见上）"))
    return rc


def smoke_emit_counts(trace_path: str, out_csv: str, n_neuron: int, dt_ms: float, T_ms: float,
                      seed: int = 7, scale: float = 1.0) -> int:
    """生成**合成**逐神经元计数夹具（仅用于测硬门③；文件内标注 SYNTHETIC；**不是**任何测量数据）。

    规则：按轨迹总脉冲总数 S，用确定性多项式分配（seed 固定）摊到 n_neuron 个神经元上，
    并允许 scale 人为扰动总量（用于验证硬门③会不会真的硬失败）。
    """
    tr = np.load(trace_path).astype(np.float64).ravel()
    S = float(tr.sum())
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.full(n_neuron, 0.35))
    cnt = np.floor(w * S * scale)
    shortfall = int(round(S * scale - cnt.sum()))
    if shortfall > 0:
        idx = rng.choice(n_neuron, size=min(shortfall, n_neuron), replace=False)
        cnt[idx] += 1
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        f.write(f"# SYNTHETIC FIXTURE（仅用于测试硬门③双估计器断言；**不是**任何仿真/实测数据；"
                f"trace={os.path.basename(trace_path)} seed={seed} scale={scale}）\n")
        w_ = csv.writer(f, lineterminator="\n")
        w_.writerow(["neuron_id", "count", "cell_type"])
        for i, c in enumerate(cnt):
            w_.writerow([i, int(c), "exc" if i % 5 else "inh"])
    print(f"[smoke] 合成计数夹具 → {out_csv}  n={n_neuron} sum={int(cnt.sum())} "
          f"(trace_sum={int(S)}, scale={scale}, T_ms={T_ms}, dt_ms={dt_ms})")
    return 0


# ===========================================================================
# CLI
# ===========================================================================
def build_protocol_from_cli(args) -> Dict[str, Any]:
    proto: Dict[str, Any] = {"protocol_id": args.protocol_id or "cli_single_arm",
                             "dt_ms": float(args.dt_ms), "settle_ms": float(args.settle_ms),
                             "n_neuron": args.n_neuron, "arms": []}
    if args.pop_trace or args.counts:
        proto["arms"].append({"arm_id": "cli_arm", "config_id": args.config_id or "cli",
                              "T_ms": float(args.T_ms), "settle_ms": float(args.settle_ms),
                              "role": args.role, "mechanism": args.mechanism, "seed": args.seed,
                              "n_neuron": args.n_neuron, "pop_trace": args.pop_trace,
                              "counts": args.counts})
    return proto


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="validate_p11_resting", description="M11 P5/G1 离线评估器（只做分析，不做仿真）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol-json", default=None, help="协议元数据 + 臂清单（见模块 docstring）")
    ap.add_argument("--pop-trace", default=None, help="单臂模式：逐 step 群体脉冲数 .npy")
    ap.add_argument("--counts", default=None, help="单臂模式：逐神经元计数 .csv")
    ap.add_argument("--dt-ms", type=float, default=None, help="步长（默认取 protocol-json.dt_ms）")
    ap.add_argument("--T-ms", type=float, default=None, help="单臂模式时长（ms）")
    ap.add_argument("--settle-ms", type=float, default=500.0)
    ap.add_argument("--n-neuron", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--role", default="baseline", choices=("zero_drive", "baseline", "dose", "replica"))
    ap.add_argument("--mechanism", default="none")
    ap.add_argument("--config-id", default="cli")
    ap.add_argument("--protocol-id", default=None)
    ap.add_argument("--targets-csv", default=DEFAULT_TARGETS)
    ap.add_argument("--out-csv", default=DEFAULT_OUT_CSV)
    ap.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    ap.add_argument("--n-boot", type=int, default=20000, help="自助 CI 次数")
    ap.add_argument("--check-targets", action="store_true", help="判据表 CSV⟺JSON 自检后退出")
    ap.add_argument("--smoke-emit-counts", metavar="OUT_CSV", default=None,
                    help="生成 SYNTHETIC 计数夹具（需 --pop-trace/--T-ms/--n-neuron）")
    ap.add_argument("--smoke-scale", type=float, default=1.0, help="夹具总量扰动因子（测硬门③）")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--save-json-dir", default=None, help="（兼容位，未使用）")
    args = ap.parse_args(argv)

    if args.check_targets:
        return check_targets(args.targets_csv, DEFAULT_TARGETS_JSON)

    if args.smoke_emit_counts:
        if not (args.pop_trace and args.T_ms and args.n_neuron):
            print("--smoke-emit-counts 需要 --pop-trace --T-ms --n-neuron", file=sys.stderr)
            return 2
        return smoke_emit_counts(args.pop_trace, args.smoke_emit_counts, args.n_neuron,
                                 args.dt_ms or 0.05, args.T_ms, seed=7, scale=args.smoke_scale)

    tt = TargetsTable(args.targets_csv)
    if args.protocol_json:
        with open(args.protocol_json, encoding="utf-8") as f:
            proto = json.load(f)
        proto_path = args.protocol_json
    else:
        if not (args.pop_trace or args.counts) or args.T_ms is None:
            print("需 --protocol-json 或 (--pop-trace/--counts 且 --T-ms)", file=sys.stderr)
            return 2
        proto = build_protocol_from_cli(args)
        if args.dt_ms is None:
            args.dt_ms = float(tt.raw("protocol_dt_ms"))
        proto["dt_ms"] = args.dt_ms
        proto_path = os.path.join(PROJECT_ROOT, "_cli_protocol.json")
    if args.dt_ms is not None:
        proto["dt_ms"] = args.dt_ms
    if args.n_neuron is not None:
        proto["n_neuron"] = args.n_neuron
    for k, dflt in (("dt_ms", None), ("settle_ms", None)):
        if proto.get(k) is None:
            print(f"protocol 缺 {k}（不做默认值静默回退）", file=sys.stderr)
            return 2

    ev = Evaluator(tt, proto, proto_path, args)
    if not ev.arms:
        print("protocol.arms 为空", file=sys.stderr)
        return 2
    out = ev.run()
    out.update({
        "generated_by": "tools/validate_p11_resting.py",
        "generated_at": _dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "role": "M11 P5/G1 离线评估（只做分析，不做仿真；不写判据带）",
        "protocol_path": os.path.abspath(proto_path),
        "protocol": {k: v for k, v in proto.items() if k != "arms"},
        "targets_csv": os.path.abspath(args.targets_csv),
        "n_arms": len(ev.arms),
        "hard_gates": {
            "①_sample_size": "零驱动/基准臂各 ≥3 seed 且零驱动总脉冲 ≥300（否则 unavailable）",
            "②_selfsustain_ci": "点估计 ≤ 上限 且 CI 上界 ≤ 上限×1.2；两 CI 法并列 + 一致性核对",
            "③_double_estimator": "counts/pop_trace/agg 三估计器两两相对差 ≤2%，并含轨迹长度⟺(T,dt) 量纲自检",
            "④_duration_ladder": "同一配置 {T_min,2×T_min,4×T_min} 三档；跨 T 判定翻转 ⇒ invalid",
            "⑤_discriminativeness": "必须给出无机制基线臂的同一判据值；不得用非判别性判据作机制证据",
            "⑥_caliber_annotation": "逐行带 protocol(T/settle/seed/bin/smooth) 与 estimator",
        },
        "arms": [{"arm_id": a.arm_id, "config_id": a.config_id, "T_ms": a.T_ms, "role": a.role,
                  "mechanism": a.mechanism, "drive_mult": a.drive_mult, "seed": a.seed,
                  "n_neuron": a.n_neuron, "pop_trace": a.pop_trace_path, "counts": a.counts_path,
                  "n_steps_present": (None if a.trace is None else int(a.trace.size))} for a in ev.arms],
        "arm_metrics": ev.arm_metrics,
        "rows": ev.rows,
        "elapsed_s": None,
    })
    one = os.path.dirname(os.path.abspath(args.out_json))
    one2 = os.path.dirname(os.path.abspath(args.out_csv))
    for d in (one, one2):
        if d:
            os.makedirs(d, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    write_rows_csv(args.out_csv, ev.rows)
    if not args.quiet:
        print_summary(out, ev.rows, ev.arm_metrics, ev.duration_ladder)
        print(f"→ {args.out_csv}\n→ {args.out_json}")
    return int(out["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
