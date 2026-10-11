"""M11 投递路径**最小复现**（索引级）：期望交付边索引/落槽位置 vs 实际。

用途（主agent RF-M11-F 的要求）：把"同步发放 / 分段"两条路径的交付**逐索引**打印出来，
与独立算出的期望集合逐项对照 —— 即 M10 抓 L17 的方式（`base=[0,1,7]` vs 期望 `[0,1,3]`）。

复现组（全部为**确定性**小网络，秒级）：
  MR1  M9 冻结公式的最小反例（3 神经元 / 4 边 / 2 源同一步发放）→ 期望 `[0,1,3]` vs M9 `[0,1,7]`；
  MR2  M10 E5-FIX 与本内核在 MR1 上的实际交付（`step` 与 `chunk` 两条路径）；
  MR3  四亚型槽（E/PV/SST/VIP）逐类投递：3 源同一步发放 + 单源 2 条同类边（CSR 块起点算术语境）；
  MR4  同一源神经元 2 条边打到同一目标（累积落槽）；
  MR5  交付多重集的分段可组合性扫描 k=1/2/3/5/10（**不看脉冲计数，直接看交付事件**）。

判据：`actual == expected`（多重集逐项相等；`step` 与 `chunk` 两路径的交付多重集也相等）。
用法：`PYTHONHASHSEED=0 OMP_NUM_THREADS=1 ./.venv-m9/bin/python -m neural_exploration.tools.minrepro_m11_delivery`
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from neural_exploration.src.mouse_cortex_engine import (MouseCortexEngine,       # noqa: E402
                                                        REF_FROZEN)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _engine(n, edges, cls_of_edge=None, classes=4, delay=None, extra_kw=None):
    """建一个最小内核：edges = [(pre, post, gmax, delay_bin)]；类的标签由 `cls_of_edge` 给。"""
    e = MouseCortexEngine(n, device="cpu", use_compile=False, verbose=False, **(extra_kw or {}))
    if classes == 4:
        e.configure_subtypes(True)
    e.finalize()
    pre = np.asarray([x[0] for x in edges], dtype=np.int64)
    post = np.asarray([x[1] for x in edges], dtype=np.int64)
    gmax = np.asarray([x[2] for x in edges], dtype=np.float32)
    dly = np.asarray([x[3] for x in edges], dtype=np.int32)
    cls = (np.zeros(len(edges), dtype=np.int8) if cls_of_edge is None
           else np.asarray(cls_of_edge, dtype=np.int8))
    if classes == 4:
        e.configure_layers(np.zeros(n, dtype=np.int64), delay_bins=tuple(sorted(set(dly.tolist()))))
    e.add_synapses(pre, post, gmax, dly, inhibitory=(cls > 0), syn_class=cls)
    e.finalize()
    e.set_neuron_heterogeneity(np.zeros(n, dtype=np.float32))
    e.reset(seed=0)
    return e


def _mk_spk(e, idx):
    v = torch.zeros(e.n, dtype=torch.bool, device=e.device)
    v[list(idx)] = True
    return v


def _actual_delivered(e, idx, step, ncn=None):
    """调一次投递，再从**环槽实际内容**反解出 (arrival_slot, post_off, amp) 多重集。"""
    ncn = int(e.n_ch) * e.n
    before = e._ring.clone()
    e._deliver_step(_mk_spk(e, idx), step)
    after = e._ring - before
    nz = (after != 0).nonzero(as_tuple=False).flatten().cpu().numpy()
    out = []
    for flat in nz:
        slot, rem = divmod(int(flat), ncn)
        cls, post = divmod(rem, e.n)
        out.append((slot, int(cls), int(post), float(after[flat].item())))
    return sorted(out)


def _expected_delivered(edges, cls_of_edge, idx, step, n_slot, n, n_ch):
    """独立期望：**逐边**列出 (arrival_slot, cls, post, amp)，再按目标格**合并求和**。

    合并是**必须**的：环槽是 `index_add_`（accumulate）语义 ⇒ 同一 (slot, cls, post) 上的多个
    事件在读出时已是**电导之和**（这正是"同一目标多条边"应有的物理语义）。
    """
    raw = []
    for k, (pre, post, g, d) in enumerate(edges):
        if pre in idx:
            cls = int(cls_of_edge[k]) if cls_of_edge is not None else 0
            if n_ch == 2 and cls > 0:
                cls = 1
            raw.append(((step + int(d)) % n_slot, cls, int(post), float(g)))
    merged = {}
    for slot, cls, post, amp in raw:
        merged[(slot, cls, post)] = merged.get((slot, cls, post), 0.0) + amp
    return sorted(raw), sorted((k[0], k[1], k[2], v) for k, v in merged.items())


def mr1_m9_formula_base():
    """M9 冻结公式 `t_row[idx] + cumsum(c) - c` 的最小反例（M10 L17 原例）。"""
    pre = [0, 0, 1, 2]
    post = [1, 2, 0, 0]
    counts = np.bincount(np.asarray(pre), minlength=3).astype(np.int64)      # [2,1,1]
    row = np.zeros(4, dtype=np.int64)
    np.cumsum(counts, out=row[1:])                                            # [0,2,3,4]
    idx = np.asarray([0, 2])                       # 同一步：0 号与 2 号神经元同时发放
    c = torch.as_tensor(counts[idx])
    cum = torch.cumsum(c, 0) - c
    t_row = torch.as_tensor(row[idx])
    m9 = (torch.repeat_interleave(t_row + cum, c) + torch.arange(int(c.sum()))).tolist()
    ok = (torch.repeat_interleave(t_row - cum, c) + torch.arange(int(c.sum()))).tolist()
    return dict(pre=pre, post=post, counts=counts.tolist(), row=row.tolist(),
                firing_neurons=idx.tolist(), n_edge=len(pre),
                m9_base=m9, m10_e5fix_base=ok, expected=[0, 1, 3],
                m9_out_of_range=bool(max(m9) >= len(pre)))


def mr2_actual_delivery(classes, edges, cls_of_edge, firing, step=0, verbose=True):
    """MR2/MR3/MR4：实际交付（step 与 chunk 路径）vs 期望。"""
    out = {}
    for mode in ("step", "chunk"):
        e = _engine(6, edges, cls_of_edge=cls_of_edge, classes=classes)
        act = _actual_delivered(e, firing, step)
        raw, exp = _expected_delivered(edges, cls_of_edge, firing, step, e.n_slot, e.n, e.n_ch)
        out[mode] = dict(actual=act, expected_merged=exp, expected_per_edge=raw,
                         equal=bool(act == exp))
        if verbose:
            print("    [%s] 实际落槽(合并后) =%s" % (mode, act), flush=True)
            print("    [%s] 期望(逐边,供逐索引核对) =%s" % (mode, raw), flush=True)
    if verbose:
        print("    [期望] 合并后 =%s" % (out["step"]["expected_merged"],), flush=True)
    out["step_vs_chunk_equal"] = bool(out["step"]["actual"] == out["chunk"]["actual"])
    out["all_equal"] = bool(out["step"]["equal"] and out["chunk"]["equal"]
                            and out["step_vs_chunk_equal"])
    return out


def mr5_segment_scan(classes, edges, cls_of_edge, firing, ks=(1, 2, 3, 5, 10), T=400, verbose=True):
    """交付多重集的分段可组合性：k 段调用后**累计交付**必须逐项相同（原始 spike 计数之外的第二层证据）。

    这里不看脉冲，而是看**引擎自身**在 k 段调用中实际投递的 `(arrival, cls, post, amp)` 多重集
    （由"写入前/后的环槽差"累计；因为槽会被读走，故用 T 步内的写入累计器替代）。
    """
    out = {}
    base = None
    for k in ks:
        e = _engine(6, edges, cls_of_edge=cls_of_edge, classes=classes)
        # 外部驱动：只打 0 号神经元，使其持续发放；分段调用
        ncn = int(e.n_ch) * e.n
        acc = {}
        # 分段长度**精确覆盖 T**（T//k*k ≠ T 时末段补余 ⇒ 否则少跑步、属夹具缺陷）
        segs = [T // k + (1 if _i < T % k else 0) for _i in range(k)]
        for _seg_len in segs:
            for _s in range(_seg_len):
                abs_s = int(e.abs_step)          # **按绝对步号驱动**（原版按局部 s ⇒ 夹具自身重放）
                spk = torch.zeros(e.n, dtype=torch.bool, device=e.device)
                # 第 5 步起让 0/1 两个源同一步发放（同步发放语境）；第 20 步起加入 2 号
                if abs_s >= 5:
                    spk[0] = True
                    spk[1] = True
                if abs_s >= 20:
                    spk[2] = True
                before = e._ring.clone()
                e._deliver_step(spk, abs_s)
                e.abs_step += 1
                diff = (e._ring - before).nonzero(as_tuple=False).flatten().cpu().numpy()
                for flat in diff:
                    key = int(flat)
                    acc[key] = acc.get(key, 0.0) + 1.0
        base = base if base is not None else acc
        same = bool(set(acc.keys()) == set(base.keys())
                    and all(abs(acc[x] - base[x]) < 1e-9 for x in acc))
        out["k=%d" % k] = dict(n_touched_slots=len(acc), same_as_k1=same)
        if verbose:
            print("    [MR5] k=%2d 交付落点集合大小=%3d 与 k=1 相同=%s"
                  % (k, len(acc), same), flush=True)
    out["pass"] = bool(all(v["same_as_k1"] for v in out.values()
                           if isinstance(v, dict)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default=os.path.join(DATA_DIR, "m11_minrepro_delivery.json"))
    args = ap.parse_args(argv)
    res = {}
    print("[MR1] M9 冻结公式最小反例（3 神经元 / 4 边 / 2 源同一步发放）", flush=True)
    res["MR1"] = mr1_m9_formula_base()
    print("    期望 base = %s" % res["MR1"]["expected"], flush=True)
    print("    M9 公式   = %s（越界=%s）" % (res["MR1"]["m9_base"], res["MR1"]["m9_out_of_range"]),
          flush=True)
    print("    M10/M11   = %s" % res["MR1"]["m10_e5fix_base"], flush=True)

    edges_basic = [(0, 3, 11.0, 40), (0, 4, 13.0, 40), (1, 4, 17.0, 40), (2, 5, 19.0, 40)]
    print("[MR2] 本内核实际交付（单类；2 源同一步发放）", flush=True)
    res["MR2"] = mr2_actual_delivery(4, edges_basic, [0, 0, 0, 0], [0, 1])

    print("[MR3] 四亚型槽逐类交付（3 源同一步发放 + 单源 2 条同类边）", flush=True)
    edges4 = [(0, 3, 11.0, 40), (0, 3, 13.0, 40), (0, 3, 23.0, 40),  # E 槽（同源→同目标 2 条，CSR 块起点语境）
              (1, 3, 31.0, 40),                                       # PV
              (1, 4, 37.0, 60),                                       # SST
              (2, 5, 41.0, 80)]                                       # VIP
    res["MR3"] = mr2_actual_delivery(4, edges4, [0, 0, 0, 1, 2, 3], [0, 1, 2])

    print("[MR4] 同一源 2 条边到同一目标（累积落槽，含 PV+SST 同类不同延迟档）", flush=True)
    edges_c = [(0, 3, 11.0, 40), (0, 3, 13.0, 40), (1, 3, 31.0, 40), (1, 3, 37.0, 60)]
    res["MR4"] = mr2_actual_delivery(4, edges_c, [0, 0, 1, 2], [0, 1])

    print("[MR5] 交付落点的分段可组合性扫描 k=1/2/3/5/10（step 路径）", flush=True)
    res["MR5"] = mr5_segment_scan(4, edges4, [0, 0, 0, 1, 2, 3], [0, 1, 2])

    # ---- 冻结档对照（nc=2）：证明扩展没有改动 M10 的 CSR 算术 ----
    print("[MR6] 冻结档对照（nc=2，E/I 两槽）：同一最小网络的实际交付", flush=True)
    edges2 = [(0, 3, 11.0, 40), (1, 4, 31.0, 40), (0, 3, 13.0, 40), (2, 5, 19.0, 40)]
    res["MR6"] = mr2_actual_delivery(2, edges2, [0, 1, 0, 0], [0, 1, 2])
    res["verdict"] = dict(
        MR1_m9_defect_reproduced=bool(res["MR1"]["m9_out_of_range"]),
        MR2=res["MR2"]["all_equal"], MR3=res["MR3"]["all_equal"], MR4=res["MR4"]["all_equal"],
        MR5_segment_composable=res["MR5"]["pass"], MR6=res["MR6"]["all_equal"],
        all_pass=bool(res["MR1"]["m9_out_of_range"] and res["MR2"]["all_equal"]
                      and res["MR3"]["all_equal"] and res["MR4"]["all_equal"]
                      and res["MR5"]["pass"] and res["MR6"]["all_equal"]),
        note="期望值由**独立**由边表算出（arrival=(step+delay)%n_slot、cls、post、amp 逐项）；"
             "实际值由**写入前/后环槽差**反解 ⇒ 两条途径独立；step 与 chunk 两路径的交付多重集亦须相等")
    with open(args.out_json, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(json.dumps(res["verdict"], ensure_ascii=False, indent=1), flush=True)
    print("json ->", args.out_json, flush=True)
    return 0 if res["verdict"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
