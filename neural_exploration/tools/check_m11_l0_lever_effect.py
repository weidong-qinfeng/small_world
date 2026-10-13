"""**L0 杠杆有效性核对（不跑仿真）**：证明 `gmax_scale/bias_scale` 真的作用到夹具网络上。

**动机（M10 L21 同族缺陷）**：计划里写 `bias_scale=0.70` ≠ 网络真的变了。若生产者把 kwarg 名写错、
或 `build_laminar_probe_column` 把它丢掉，产物里的 `net.bias_scale` 仍会**回显 0.7**（自证），
判据会以为"杠杆没效果"而不是"杠杆没生效"。故须有一条**独立于产物回显**的效果证据：

- 直接装配 5 个档位的网络（同一 `net_seed=0`），比对 `net["bias"]` / `net["gmax"]` 张量；
- **量纲恒等式自检**：`sum(bias | bias_scale=x) / sum(bias | bias_scale=1) == x`（无量纲倍数），
  `gmax` 同理——比值是**无量纲**的，故断言与单位无关；
- 逐位可复现：同参数两次装配的两个张量 sha256 必须相同（网络是 `net_seed` 的确定函数）。

用法：
    OMP_NUM_THREADS=1 ./.venv-m9/bin/python -m neural_exploration.tools.check_m11_l0_lever_effect
产物：`data/m11_l0_lever_effect.json`
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np

from neural_exploration.src.mouse_cortex_circuit import build_laminar_probe_column

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "neural_exploration", "data")
N = 8192
NET_SEED = 0
INDEGREE = 20
DELAY_BINS = (40, 60, 80)
REL_TOL = 1e-5          # float32 张量按比例缩放后的相对误差上限（理论 ~1e-7）
LEVELS = (
    ("p5_L0_g1.00_b1.00", 1.00, 1.00),
    ("p5_L0_g1.00_b0.70", 1.00, 0.70),
    ("p5_L0_g1.00_b0.50", 1.00, 0.50),
    ("p5_L0_g2.00_b1.00", 2.00, 1.00),
    ("p5_L0_g4.00_b1.00", 4.00, 1.00),
)


def sha32(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:32]


def build(gs: float, bs: float):
    return build_laminar_probe_column(N, seed=NET_SEED, indegree=INDEGREE, delay_bins=DELAY_BINS,
                                      gmax_scale=gs, bias_scale=bs)


def main() -> int:
    t0 = time.time()
    print("== L0 杠杆有效性核对（装配级，不跑仿真；N=%d net_seed=%d）==" % (N, NET_SEED))
    rows, nets = [], {}
    for cfg, gs, bs in LEVELS:
        net = build(gs, bs)
        nets[cfg] = net
        rows.append(dict(config_id=cfg, gmax_scale=gs, bias_scale=bs,
                         bias_sum=float(net["bias"].astype(np.float64).sum()),
                         bias_mean=float(net["bias"].astype(np.float64).mean()),
                         gmax_sum=float(net["gmax"].astype(np.float64).sum()),
                         gmax_mean=float(net["gmax"].astype(np.float64).mean()),
                         bias_sha256=sha32(net["bias"]), gmax_sha256=sha32(net["gmax"]),
                         n_edge=int(net["gmax"].size)))
    ref = nets["p5_L0_g1.00_b1.00"]
    ref_bias = float(ref["bias"].astype(np.float64).sum())
    ref_gmax = float(ref["gmax"].astype(np.float64).sum())
    fails = []
    # ---- 量纲恒等式（逐档）----
    print("  %-20s %-8s %-8s %14s %14s %8s %8s" % ("config_id", "gmax", "bias",
                                                    "bias_sum比率", "gmax_sum比率", "biasΔ%", "gmaxΔ%"))
    for r, (cfg, gs, bs) in zip(rows, LEVELS):
        rb, rg = r["bias_sum"] / ref_bias, r["gmax_sum"] / ref_gmax
        r["bias_ratio_to_reference"] = rb
        r["gmax_ratio_to_reference"] = rg
        r["bias_ratio_minus_declared"] = rb - bs
        r["gmax_ratio_minus_declared"] = rg - gs
        r["bias_rel_diff_to_reference"] = abs(rb - 1.0)
        r["gmax_rel_diff_to_reference"] = abs(rg - 1.0)
        print("  %-20s %-8.2f %-8.2f %14.9f %14.9f %8.4f %8.4f"
              % (cfg, gs, bs, rb, rg, 100 * (rb - 1), 100 * (rg - 1)))
        if abs(rb - bs) > REL_TOL * max(bs, 1e-9):
            fails.append("%s: bias 张量比 %r ≠ 声明 bias_scale %r（杠杆被静默忽略/放大错误）"
                         % (cfg, rb, bs))
        if abs(rg - gs) > REL_TOL * max(gs, 1e-9):
            fails.append("%s: gmax 张量比 %r ≠ 声明 gmax_scale %r" % (cfg, rg, gs))
    # 非平凡性：非参考档必须**真的**与参考档不同（否则"档位不同"是假的）
    for r, (cfg, gs, bs) in zip(rows, LEVELS):
        if (gs, bs) == (1.0, 1.0):
            continue
        d_bias = float(np.abs(nets[cfg]["bias"].astype(np.float64) - ref["bias"].astype(np.float64)).sum())
        d_gmax = float(np.abs(nets[cfg]["gmax"].astype(np.float64) - ref["gmax"].astype(np.float64)).sum())
        r["l1_diff_bias_to_reference"] = d_bias
        r["l1_diff_gmax_to_reference"] = d_gmax
        same_b = (r["bias_sha256"] == rows[0]["bias_sha256"])
        same_g = (r["gmax_sha256"] == rows[0]["gmax_sha256"])
        if bs != 1.0 and (same_b or d_bias <= 0.0):
            fails.append("%s: bias 张量与参考档无差异（该档实为同一网络）" % cfg)
        if gs != 1.0 and (same_g or d_gmax <= 0.0):
            fails.append("%s: gmax 张量与参考档无差异（该档实为同一网络）" % cfg)
        print("     非平凡性 %-18s L1(bias)=%.1f L1(gmax)=%.1f（仅该杠杆对应张量应为非零）"
              % (cfg, d_bias, d_gmax))
    # ---- 装配确定性（同一参数两次 ⇒ 逐位相同）----
    net2 = build(1.00, 0.70)
    refrow = {r["config_id"]: r for r in rows}["p5_L0_g1.00_b0.70"]
    det = (sha32(net2["bias"]) == refrow["bias_sha256"]
           and sha32(net2["gmax"]) == refrow["gmax_sha256"])
    print("  装配确定性（同参数两次逐位相同）= %s" % det)
    if not det:
        fails.append("同参数两次装配的张量不同 ⇒ 夹具网络不确定（复刻轴不可比）")
    # ---- 量纲自检的"算例"（R76：中间值由脚本算出并落盘）----
    ex = dict(bias_scale_declared=0.70,
              bias_scale_observed=rows[1]["bias_ratio_to_reference"],
              identity="sum(bias|scale=x)/sum(bias|scale=1) == x  （比值无量纲，与偏置单位无关）",
              tolerance=REL_TOL, gmax_scale_declared=2.00,
              gmax_scale_observed=rows[3]["gmax_ratio_to_reference"])
    print("  量纲自检算例：bias 声明 0.70 / 实测比率 %.9f；gmax 声明 2.00 / 实测比率 %.9f"
          % (ex["bias_scale_observed"], ex["gmax_scale_observed"]))
    out = dict(tool="tools/check_m11_l0_lever_effect.py",
               generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
               n_neuron=N, net_seed=NET_SEED, indegree=INDEGREE, delay_bins=list(DELAY_BINS),
               units=dict(bias="引擎内亚阈 tonic 偏置单位（`set_neuron_heterogeneity` 同一张量）",
                          gmax="引擎内突触权重上界单位", ratio="无量纲倍数"),
               identity_self_check=ex, rows=rows,
               knob_effect_verified=(not fails),
               falsifier="若某人把 `bias_scale` 从 `build_laminar_probe_column` 里删掉，"
                         "本核对的比率会变成 1.0 ≠ 0.70 ⇒ 退出码 7（本次实测见 rows）",
               n_failures=len(fails), failures=fails, elapsed_s=round(time.time() - t0, 2))
    p = os.path.join(DATA_DIR, "m11_l0_lever_effect.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("产物 -> %s" % p)
    if fails:
        print("**杠杆有效性核对失败（%d 条）**：" % len(fails))
        for x in fails[:10]:
            print("   -", x)
        return 7
    print("杠杆有效性核对通过：5 档的 bias/gmax 张量比率 == 声明倍数（相对误差 ≤ %g）；"
          "非参考档均与参考档有实质差异；装配可复现" % REL_TOL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
