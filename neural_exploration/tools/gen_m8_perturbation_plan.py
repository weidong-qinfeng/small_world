#!/usr/bin/env python
"""P7 扰动计划生成器：data/m8_perturbation_plan.csv（确定性 top-N=50，预注册）。

《生物仿真M8实施清单》§1 D3 / §9.1 / §0.7 #4：
- top-N=50 选择规则（确定性算法，固定 seed=0，名单落盘后不事后改）：
  ① 中枢命令样/下行神经元（celltype ∈ {pre-DN-VNC, DN-VNC, pre-DN-SEZ, DN-SEZ}
     或 CMD/INH/COMMAND 前缀或 brain→vnc 下行边 pre）——行为控制候选；
  ② 连接度枢纽（子集内 chem_all 入+出度 top 分位）；
  ③ 实验可及性（有 skid/CATMAID id、有 celltype 标注；MD class IV 伤害感受器
     驱动线成熟 → access=1.0，文献锚明确）；
  total = 1.0·command + 0.5·hub_pct + 0.3·access；平局按 role_index（确定性）；
- 行为后果类集 {无变化/前进↑/前进↓/转弯↑/停驻↑/蜷缩↑/后退↑} + 阈值预注册；
- 实验锚映射表（来源/效应类）：MD → 蜷缩↑/后退↑（class IV 伤害感受器光遗传
  激活 → 蜷缩/滚动/后退逃避：Hwang et al. 2007 PNAS；Robertson et al. 2013
  Curr Biol；PMC5555049 图 1）；其余无逐神经元驱动线锚 → "no-experiment"
  （不入命中率分母，§0.7 #4 不静默剔除）。

运行：MPLBACKEND=Agg ./.venv-neuro/bin/python tools/gen_m8_perturbation_plan.py
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.src.larva_circuit import (  # noqa: E402
    load_connectome,
    scale_names,
)

DATA_DIR = os.path.join(ROOT, "neural_exploration", "data")
CONNECTOME_CSV = os.path.join(DATA_DIR, "m8_larva_connectome.csv")
PLAN_CSV = os.path.join(DATA_DIR, "m8_perturbation_plan.csv")

SCALE = 300
TOP_N = 50
SEED = 0

COMMAND_CELLTYPES = {"pre-dn-vnc", "dn-vnc", "pre-dn-sez", "dn-sez"}
COMMAND_PREFIXES = ("CMD", "INH", "COMMAND")
#: 行为后果类阈值（预注册，§9.1；与 validate_p8_perturbation.py 一致）
THRESH = dict(fwd=0.05, back=0.05, turn=0.05, pause=0.05, curl=0.05)
#: 激活注入参数（预注册）
ACTIVATION_nA = 0.5
ACTIVATION_DUR_MS = 2000.0
SILENCE_SEMANTICS = "出边 gmax→0（M6 L15：整体重建数组）"


def main() -> int:
    spec = load_connectome(CONNECTOME_CSV)
    names = scale_names(spec, SCALE)
    sub = spec.subset(names)
    role_index = {r: k for k, r in enumerate(names)}

    # 子集内 chem_all 度（连接组事实；双向）
    deg = {n: 0 for n in names}
    for r in sub.chem_all:
        if r.pre in deg:
            deg[r.pre] += 1
        if r.post in deg:
            deg[r.post] += 1
    # 连接度枢纽 = 度分位（在子集内排序中的百分位，[0,1]；规则②语义）
    deg_sorted = sorted(deg.values())
    n_deg = len(deg_sorted)

    def _deg_pct(d: int) -> float:
        return float(np.searchsorted(deg_sorted, d, side="right") / n_deg)

    # brain→vnc 下行 pre 集合（命令层语义）
    down_pre = set()
    for r in sub.chem_all:
        if r.pre in deg and r.post in deg:
            pre_region = sub.neurons.get(r.pre, {}).get("region", "")
            post_region = sub.neurons.get(r.post, {}).get("region", "")
            if pre_region == "brain" and post_region == "vnc":
                down_pre.add(r.pre)

    rows = []
    for n in names:
        meta = sub.neurons.get(n, {})
        ct = (meta.get("celltype") or "").strip().lower()
        command = 1.0 if (ct in COMMAND_CELLTYPES
                          or n.startswith(COMMAND_PREFIXES)
                          or n in down_pre) else 0.0
        hub_pct = _deg_pct(deg.get(n, 0))
        skid = (meta.get("skid") or "").strip()
        has_ct = 1.0 if ct else 0.0
        is_md = n.startswith("MD")
        access = 1.0 if is_md else (0.6 if skid else 0.0) + 0.3 * has_ct
        total = 1.0 * command + 0.5 * hub_pct + 0.3 * access
        rows.append(dict(role=n, neuron_class=meta.get("neuron_class", ""),
                         celltype=meta.get("celltype", ""),
                         region=meta.get("region", ""),
                         skid=skid, degree=deg.get(n, 0),
                         command=command, hub_pct=hub_pct, access=access,
                         total=total, idx=role_index[n]))

    # 确定性排序（total desc，平局 role_index asc；无随机性——强于固定 seed 抽样）
    rows.sort(key=lambda r: (-r["total"], r["idx"]))
    top = rows[:TOP_N]

    with open(PLAN_CSV, "w", newline="", encoding="utf-8") as f:
        wtr = csv.writer(f)
        wtr.writerow([
            "# M8 P7 扰动预测计划（预注册，确定性生成；seed=%d，名单定稿不事后改）"
            % SEED])
        wtr.writerow([
            "# 选择规则：①命令样/下行（command=1.0）②连接度枢纽（hub_pct=度/max）"
            "③实验可及性（access：MD=1.0/skid=0.6/celltype=0.3）"])
        wtr.writerow([
            "# total = 1.0*command + 0.5*hub_pct + 0.3*access；"
            "平局按 role_index（确定性算法，无随机性）"])
        wtr.writerow([
            "# 行为后果类阈值（预注册）：Δfrac 阈值 fwd/back/turn/pause/curl=0.05；"
            "curl 通道在 provisional 肌肉映射不存在 → 蜷缩↑ 结构性不可达（记录）"])
        wtr.writerow([
            "# 激活：tonic %.1f nA × %s ms；沉默：%s" % (
                ACTIVATION_nA, ACTIVATION_DUR_MS, SILENCE_SEMANTICS)])
        wtr.writerow([
            "rank", "role", "neuron_class", "celltype", "region", "skid",
            "degree", "command", "hub_pct", "access", "total",
            "anchor_status", "anchor_effect_class", "anchor_source",
            "silence_protocol", "activation_protocol"])
        for k, r in enumerate(top):
            is_md = r["role"].startswith("MD")
            if is_md:
                anchor = "anchored"
                effect = "蜷缩↑/后退↑"
                src = ("class IV 伤害感受器光遗传激活→蜷缩/滚动/后退逃避："
                       "Hwang et al. 2007 PNAS 104:11388; Robertson et al. 2013 "
                       "Curr Biol 23:79; PMC5555049（nociceptor activation "
                       "→ curling/rolling）")
            else:
                anchor = "no-experiment"
                effect = ""
                src = "无逐神经元驱动线锚（B1d 网络受限，预注册回退 §0.7 #4）"
            wtr.writerow([
                k + 1, r["role"], r["neuron_class"], r["celltype"],
                r["region"], r["skid"], r["degree"], f"{r['command']:.2f}",
                f"{r['hub_pct']:.4f}", f"{r['access']:.2f}",
                f"{r['total']:.4f}", anchor, effect, src,
                SILENCE_SEMANTICS,
                f"tonic {ACTIVATION_nA} nA × {ACTIVATION_DUR_MS} ms"])

    n_anchored = sum(1 for r in top if r["role"].startswith("MD"))
    print(f"plan CSV: {PLAN_CSV}")
    print(f"top-{TOP_N} 生成（确定性）；命令样 {sum(1 for r in top if r['command']>0)}"
          f"，有锚 {n_anchored}（MD，文献锚）")
    print("top-10 名单：")
    for k, r in enumerate(top[:10]):
        print(f"  {k+1:2d} {r['role'][:46]:48s} cls={r['neuron_class']:8s} "
              f"deg={r['degree']:4d} cmd={r['command']:.1f} "
              f"hub={r['hub_pct']:.2f} acc={r['access']:.2f}")
    if n_anchored < 20:
        print(f"⚠ 预注册 §0.7 #4：有锚子集 {n_anchored}/50 < 20 下限 → 命中率分母"
              f"缩小为有锚子集 + 记录测量限制（B1d 网络受限回退）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
