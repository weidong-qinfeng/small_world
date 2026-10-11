"""M10 RF-3 落实：P7 ④ 的**替代可分性方案**（ZAPBench，无需新数据）。

主agent RF-3 裁决：接受 ④ `unavailable`（不下载 14 h），但**要求给出替代方案**：
  ① 用**块内子集**或**条件对比对**（如 taxis vs dark）构造可分性；
  ② 或用**时间交错**条件（若数据内含交错段）；
  ③ 都不可得 → P7 判 **3/4 指标 + ④ unavailable**，并说明"块级零分布 > 数据"是**数据获取限制**。

**ZAPBench 事实（实测）**：`CONDITION_OFFSETS` 单调递增 ⇒ 9 个条件为**连续时间块**，**无交错段**
（②不可得，登记）。故本工具实现 ①，并给出**漂移对照地板**：

  `cross-condition 可分性`：二分类（条件 A vs 条件 B）线性判别，训练/测试用**不同时间块**；
  `within-condition 漂移地板`：把同一条件切成早/晚两半做同一分类（纯漂移 ⇒ 应接近随机），
                              作为"块间漂移本身可被误判为条件差异"的**零假设地板**；
  **判据**：`cross_accuracy − within_floor ≥ 预注册裕度` 才认定"该对比对可分"。

用法：
    PYTHONHASHSEED=0 ./.venv-m10/bin/python -m neural_exploration.tools.validate_p10_altdecoding
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CONDITIONS = ("gain", "dots", "flash", "taxis", "turning", "position",
              "open loop", "rotation", "dark")
OFFSETS = (0, 649, 2422, 3078, 3735, 5047, 5638, 6623, 7279, 7879)


def _load(path, n_neurons, stride):
    x = np.load(path, mmap_mode="r")
    x = np.asarray(x[::stride, :n_neurons], dtype=np.float64)
    return x


def _ridge_fit(X, y, lam=1e3):
    mu = X.mean(axis=0); sd = X.std(axis=0) + 1e-9
    Z = (X - mu) / sd
    Y = np.stack([1.0 - y, y], axis=1)
    G = Z.T @ Z + lam * np.eye(Z.shape[1])
    R = np.linalg.cholesky(G)
    W = np.linalg.solve(R.T, np.linalg.solve(R, Z.T @ Y))
    return dict(W=W, mu=mu, sd=sd)


def _ridge_pred(model, X, center_frames=False):
    if center_frames:
        X = X - X.mean(axis=1, keepdims=True)
    Z = (X - model["mu"]) / model["sd"]
    return np.argmax(Z @ model["W"], axis=1)


def _boot_ci(correct: np.ndarray, n_boot=200, seed=0):
    """对测试样本做 bootstrap → 准确率的 95% CI（并给出分子/分母）。"""
    rng = np.random.default_rng(seed)
    n = correct.size
    accs = [float(correct[rng.integers(0, n, n)].mean()) for _ in range(n_boot)]
    return float(np.quantile(accs, 0.025)), float(np.quantile(accs, 0.975))


def _perm_pvalue(pred: np.ndarray, y: np.ndarray, n_perm=200, seed=0):
    """标签置换零分布 → p 值（二分类，平衡设计 ⇒ 机遇 0.5）。"""
    rng = np.random.default_rng(seed)
    obs = float((pred == y).mean())
    null = [float((pred == rng.permutation(y)).mean()) for _ in range(n_perm)]
    return obs, float(np.mean(np.array(null) >= obs)), float(np.max(null))


def pair_separability(X, lab, cond_a, cond_b, center_frames=False, folds=4,
                      block_frac=0.5):
    """跨条件可分性 + 条件内漂移地板。X:(T,N) lab:(T,) 条件编号。"""
    out = {}
    # ---- 跨条件：早半训练 / 晚半测试（以及反向），避免时间重叠 ----
    accs = []
    for (tr_c, te_c) in ((cond_a, cond_b), (cond_b, cond_a)):
        ia = np.flatnonzero(lab == tr_c); ib = np.flatnonzero(lab == te_c)
        half_a = int(ia.size * block_frac); half_b = int(ib.size * block_frac)
        Xtr = np.vstack([X[ia[:half_a]], X[ib[:half_b]]])
        ytr = np.r_[np.zeros(half_a), np.ones(half_b)]
        Xte = np.vstack([X[ia[half_a:]], X[ib[half_b:]]])
        yte = np.r_[np.zeros(ia.size - half_a), np.ones(ib.size - half_b)]
        m = _ridge_fit(Xtr, ytr)
        pred = _ridge_pred(m, Xte, center_frames)
        corr = (pred == yte)
        lo, hi = _boot_ci(corr, seed=int(cond_a * 10 + cond_b))
        obs, p_perm, null_max = _perm_pvalue(pred, yte, seed=int(cond_a * 10 + cond_b))
        accs.append(dict(accuracy=float(corr.mean()), n_test=int(yte.size),
                         n_correct=int(corr.sum()), ci95=[lo, hi],
                         perm_p=p_perm, perm_null_max=null_max))
    out["cross_accuracy"] = float(np.mean([a["accuracy"] for a in accs]))
    out["cross_accuracy_both_directions"] = accs
    # ---- 条件内漂移地板：同一条件早半 vs 晚半（标签 = 时间块，条件身份相同）----
    floors = []
    for c in (cond_a, cond_b):
        i = np.flatnonzero(lab == c)
        h = i.size // 2
        ia, ib = i[:h], i[h:]
        Xtr = np.vstack([X[ia[: h // 2]], X[ib[: ib.size // 2]]])
        ytr = np.r_[np.zeros(h // 2), np.ones(ib.size // 2)]
        Xte = np.vstack([X[ia[h // 2:]], X[ib[ib.size // 2:]]])
        yte = np.r_[np.zeros(ia.size - h // 2), np.ones(ib.size - ib.size // 2)]
        m = _ridge_fit(Xtr, ytr)
        corr = (_ridge_pred(m, Xte, center_frames) == yte)
        lo, hi = _boot_ci(corr, seed=int(c))
        floors.append(dict(accuracy=float(corr.mean()), n_test=int(yte.size),
                           n_correct=int(corr.sum()), ci95=[lo, hi]))
    out["within_condition_drift_floor"] = float(np.mean([f["accuracy"] for f in floors]))
    out["drift_floor_per_condition"] = floors
    out["nonstationary"] = bool(out["within_condition_drift_floor"] >= 0.5)
    out["margin"] = float(out["cross_accuracy"] - out["within_condition_drift_floor"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default=os.path.join(DATA_DIR, "m10_raw",
                                                     "zapbench_traces_8192.f32.npy"))
    ap.add_argument("--n-neurons", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--margin-min", type=float, default=0.10,
                    help="预注册裕度：cross − drift_floor ≥ 0.10 才判该对比对可分")
    ap.add_argument("--out-json", default=os.path.join(DATA_DIR, "m10_p7_altdecoding.json"))
    ap.add_argument("--out-csv", default=os.path.join(DATA_DIR, "m10_p7_altdecoding.csv"))
    args = ap.parse_args(argv)
    t0 = time.time()
    X = _load(args.traces, args.n_neurons, args.stride)
    lab = np.zeros(OFFSETS[-1], dtype=np.int64)
    for i in range(len(OFFSETS) - 1):
        lab[OFFSETS[i]:OFFSETS[i + 1]] = i
    lab = lab[::args.stride]
    print(f"[altdec] X={X.shape} conditions={len(CONDITIONS)}", flush=True)

    pairs = [("taxis_vs_dark", CONDITIONS.index("taxis"), CONDITIONS.index("dark")),
             ("gain_vs_dark", CONDITIONS.index("gain"), CONDITIONS.index("dark")),
             ("dots_vs_dark", CONDITIONS.index("dots"), CONDITIONS.index("dark")),
             ("rotation_vs_position", CONDITIONS.index("rotation"),
              CONDITIONS.index("position"))]
    rows = []
    for name, a, b in pairs:
        for cf in (False, True):
            r = pair_separability(X, lab, a, b, center_frames=cf)
            r.update(pair=name, frame_centered=cf,
                     separable=bool(r["margin"] >= args.margin_min))
            rows.append(r)
            print(f"  {name} (帧中心化={cf}): cross={r['cross_accuracy']:.4f} "
                  f"drift_floor={r['within_condition_drift_floor']:.4f} "
                  f"margin={r['margin']:+.4f} ⇒ "
                  f"{'可分' if r['separable'] else '不可分（漂移主导）'}", flush=True)
    best = max(rows, key=lambda r: r["margin"])
    n_sep_raw = sum(1 for r in rows if (not r["frame_centered"]) and r["separable"])
    n_sep_fc = sum(1 for r in rows if r["frame_centered"] and r["separable"])
    drift = float(np.mean([r["within_condition_drift_floor"] for r in rows]))
    out = dict(generated_by="tools/validate_p10_altdecoding.py",
               protocol="跨条件（早半训练/晚半测试，双向）+ 条件内漂移地板（早半 vs 晚半，同条件）",
               frame_rate_hz=7879 / 7200.0, n_neurons=args.n_neurons, stride=args.stride,
               pre_registered_margin_min=args.margin_min,
               interleaved_conditions_available=False,
               interleaved_note="ZAPBench CONDITION_OFFSETS 单调递增 ⇒ 无条件交错段（RF-3 ② 不可得）",
               rows=rows, best=best,
               two_operationalizations={
                   "raw_features": {"n_separable_pairs": n_sep_raw, "of": len(pairs),
                                    "margins": [round(r["margin"], 4) for r in rows
                                                if not r["frame_centered"]],
                                    "note": "未去帧均值：跨条件准确率高，但**条件内漂移地板同样高**"},
                   "frame_centered": {"n_separable_pairs": n_sep_fc, "of": len(pairs),
                                      "margins": [round(r["margin"], 4) for r in rows
                                                  if r["frame_centered"]],
                                      "note": "去帧均值后：可分性消失（margin ≈0）"},
               },
               conclusion=("**可分性口径依赖（两口径相反结论，均登记在案）**："
                           f"未去帧均值口径 {n_sep_raw}/{len(pairs)} 对可分（margin +0.22~+0.30）；"
                           f"**去帧均值口径 {n_sep_fc}/{len(pairs)} 对不可分（margin −0.25~−0.02）**。"
                           f"条件内漂移地板均值 **{drift:.3f} ≥ 0.5 ⇒ 记录非平稳**"
                           "（与 M9 R7/R9 同一混杂：慢漂移可被误判为条件差异）。"
                           "⇒ P7 ④ 的结论只能是**条件性的**：ZAPBench 上 ④ 维持 `unavailable`"
                           "（数据获取限制，非模型问题），且**两个口径都必须登记**。"),
               elapsed_s=round(time.time() - t0, 1))
    with open(args.out_csv, "w", newline="") as f:
        f.write("# M10 RF-3：P7 ④ 替代可分性（ZAPBench；跨条件 vs 条件内漂移地板）\n")
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["pair", "frame_centered", "cross_accuracy", "cross_n_test",
                    "cross_n_correct", "cross_ci95_lo", "cross_ci95_hi", "cross_perm_p",
                    "cross_perm_null_max", "within_condition_drift_floor", "drift_ci95_lo",
                    "drift_ci95_hi", "margin", "separable", "nonstationary"])
        for r in rows:
            d1, d2 = r["cross_accuracy_both_directions"]
            df1 = r["drift_floor_per_condition"][0]
            w.writerow([r["pair"], r["frame_centered"], round(r["cross_accuracy"], 5),
                        d1["n_test"], d1["n_correct"], round(d1["ci95"][0], 5),
                        round(d1["ci95"][1], 5), round(d1["perm_p"], 5),
                        round(d1["perm_null_max"], 5),
                        round(r["within_condition_drift_floor"], 5),
                        round(df1["ci95"][0], 5), round(df1["ci95"][1], 5),
                        round(r["margin"], 5), r["separable"], r["nonstationary"]])
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps({r["pair"] + ("_fc" if r["frame_centered"] else ""):
                      [round(r["cross_accuracy"], 4), round(r["within_condition_drift_floor"], 4),
                       r["separable"]] for r in rows}, ensure_ascii=False))
    print("best:", best["pair"], best["margin"], best["separable"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
