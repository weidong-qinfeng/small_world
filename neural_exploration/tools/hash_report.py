#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""跨节点哈希传值器：**一次打印三种基准**，杜绝"同值不同哈希"被误判为数据不一致。

纪律（RF-31，见 docs/m10_env_notes.md L52）：
    任何在两个节点之间传递的 sha256 **必须标注基准**：
      `file_bytes` —— 文件**原始字节**（.npy 含 128 字节 header）⇒ `sha256(open(p,'rb').read())`
      `payload`    —— **数组载荷**（不含 header）        ⇒ `sha256(np.load(p).tobytes())`
      `semantic`   —— **语义等价判据**（如"逐位相同 + sum 相同"），用于 header/元数据必然不同的场合

背景（真实事故）：同一 `poptrace_M0_lamx0.0_n250.npy` 在两节点报出
`6b794438e487bddc`（file_bytes）与 `6294245cc83d148e`（payload），一度被疑为"重复作业覆盖"；
实测系**基准差异**，两口径结论一致（M0/M-1/M-4 逐位相同、sum 均 1154）。

用法：
    # 默认：对 data/m10_p6_poptraces/*.npy 全量打印三口径
    PYTHONHASHSEED=0 ./.venv-m9/bin/python neural_exploration/tools/hash_report.py

    # 指定文件（可混用 .npy 与普通文件；非 npy 的 payload 列显示 n/a）
    ... tools/hash_report.py f1.npy f2.bin --json out.json

    # 只比较两节点报来的值是否属于同一文件的某个基准
    ... tools/hash_report.py f.npy --explain 6b794438e487bddc 6294245cc83d148e
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # neural_exploration/
DEFAULT_GLOB = os.path.join(ROOT, "data", "m10_p6_poptraces", "*.npy")


def hashes(path: str) -> dict:
    raw = open(path, "rb").read()
    out = {"path": path, "n_bytes": len(raw),
           "file_bytes": hashlib.sha256(raw).hexdigest(),
           "payload": None, "semantic": None, "n_elements": None, "sum": None}
    if path.endswith(".npy"):
        a = np.load(path)
        out["payload"] = hashlib.sha256(a.tobytes()).hexdigest()
        out["n_elements"] = int(a.size)
        out["sum"] = float(a.sum())
        # semantic 基准：把"逐位相同"的判据也固定下来（同 shape/dtype 时按载荷 + 总和双条件）
        out["semantic"] = "bitwise-identical+sum-equal (shape=%s dtype=%s)" % (a.shape, a.dtype)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="待哈希文件（默认为 M10 P6 全部 pop trace）")
    ap.add_argument("--json", default=None, help="把结果写成 JSON")
    ap.add_argument("--explain", nargs="*", default=None,
                    help="给定若干哈希（可带前缀），说明它们各属于哪个文件的哪个基准")
    a = ap.parse_args(argv)

    files = a.files or sorted(glob.glob(DEFAULT_GLOB))
    if not files:
        print("[FATAL] 无文件可哈希：%s" % DEFAULT_GLOB, file=sys.stderr)
        return 2

    print("基准词表：file_bytes（原始字节，.npy 含 128 B header）/ payload（arr.tobytes()）/ semantic（语义等价）")
    print("%-38s %-17s %-17s %-9s %s" % ("file", "file_bytes(16)", "payload(16)", "size", "sum"))
    rows = []
    for f in files:
        r = hashes(f)
        rows.append(r)
        print("%-38s %-17s %-17s %-9s %s"
              % (os.path.basename(f), r["file_bytes"][:16],
                 (r["payload"] or "n/a")[:16], r["n_elements"] if r["n_elements"] is not None else r["n_bytes"],
                 r["sum"]))

    # 自检：同一基准下，载荷相同 ⇒ payload 必相同；此处对 λ×0 组做断言
    lam0 = [r for r in rows if "lamx0.0" in r["path"]]
    if len(lam0) >= 3:
        distinct = {r["payload"] for r in lam0}
        print("\n[自检] λ×0 组 %d 个文件：file_bytes 去重 %d 个、payload 去重 %d 个 ⇒ %s"
              % (len(lam0), len({r["file_bytes"] for r in lam0}), len(distinct),
                 "同一基准内一致（差异来自 arm 而非基准）" if len(distinct) <= 2 else "存在多值，需人工核对"))

    if a.explain:
        print("\n[explain] 给定哈希归属：")
        for h in a.explain:
            hit = []
            for r in rows:
                for basis in ("file_bytes", "payload"):
                    if r[basis] and r[basis].startswith(h):
                        hit.append("%s / %s" % (os.path.basename(r["path"]), basis))
            print("  %-20s ⇒ %s" % (h, "; ".join(hit) if hit else "**未匹配任何文件任何基准**"))

    if a.json:
        json.dump(dict(basis_vocab=["file_bytes", "payload", "semantic"],
                       discipline="跨节点 sha256 必须标注基准", rows=rows),
                  open(a.json, "w"), ensure_ascii=False, indent=2)
        print("\n[OK] 落盘 %s" % a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
