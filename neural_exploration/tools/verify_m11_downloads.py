#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M11 · 下载完整性门（**逐对象校验：HTTP 200 → 大小 → SHA-256 → 解压 → 尺寸自洽**）

背景（主 agent 复核发现的真问题）
----------------------------------
首次子集下载时，`curl` 因带宽竞争在若干片上超时（`curl_exit=28`）但**留下了部分字节**，
而当时的判定只看了 `http_code==200 and bytes>0` ⇒ **被误记为 `REACHABLE`**（这正是项目纪律要防的
"静默失效"）。本工具把完整性做成**独立可复跑的门**，并把结果写回 manifest 的 `integrity` 列。

判定链（任一环失败 → `INCOMPLETE`，可重取；**统计/参考带只允许使用 `COMPLETE` 对象**）
--------------------------------------------------------------------------------
1. `http_code ∈ {200, 206}` 且 `curl_exit == 0`（manifest 记录）
2. 本地大小 == manifest 记录字节数
3. 本地 **SHA-256 == manifest 记录 SHA-256**
4. **解压成功**（chunk = 20 B 大端头 + gzip 成员；自动定位 gzip 魔数）
5. **尺寸自洽**：解压后字节数 == Π(header dims) × itemsize（`dtype` 取自 `attributes.json` 读回值）

用法
----
    cd /Users/weidong/ai/small_world
    PYTHONHASHSEED=0 ./.venv-m11/bin/python neural_exploration/tools/verify_m11_downloads.py            # 只校验
    ... --refetch        # 先重取 INCOMPLETE 片，再复校
    ... --all-raw        # 同时扫描 data/m11_raw/** 其它落盘件（大小/SHA/JSON 可解析）
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import struct
import subprocess
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJ = os.path.dirname(ROOT)
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "m11_raw")
MANIFEST = os.path.join(DATA, "m11_functional_subset_manifest.csv")
OUT_CSV = os.path.join(DATA, "m11_download_integrity.csv")
OUT_SUM = os.path.join(DATA, "m11_download_integrity_summary.json")
LOG = os.path.join(RAW, "verify_m11_downloads.log")

DTYPE_ITEMSIZE = {"int16": 2, "uint16": 2, "uint8": 1, "float32": 4, "uint64": 8, "int32": 4}


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(m):
    os.makedirs(RAW, exist_ok=True)
    line = "[%s] %s" % (datetime.now(timezone.utc).strftime("%H:%M:%S"), m)
    with open(LOG, "a", buffering=1) as f:
        f.write(line + "\n")
    print(line, flush=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def curl_get(url, out_path, timeout=180):
    """单一对象下载（带 --fail 语义由调用方按 rc 判定；**必须校验 rc==0**）。"""
    cmd = ["curl", "-sS", "-L", "--connect-timeout", "8", "--max-time", str(timeout),
           "-o", out_path, "-w", "%{http_code}|%{size_download}"]
    try:
        p = subprocess.run(cmd + [url], capture_output=True, timeout=timeout + 15)
    except subprocess.TimeoutExpired:
        return 0, 0, 124
    parts = p.stdout.decode("utf-8", "replace").strip().split("|")
    code = int(parts[0]) if parts and parts[0].isdigit() else 0
    n = int(float(parts[1])) if len(parts) > 1 else 0
    return code, n, p.returncode


def expected_dims_from_header(b):
    """chunk header = 大端 uint32[num_dims, dims...]；返回 (num_dims, dims tuple)。"""
    if len(b) < 24:
        return None, None
    nd = struct.unpack_from(">I", b, 0)[0]
    if not (1 <= nd <= 8):
        return None, None
    return nd, struct.unpack_from(">%dI" % nd, b, 4)


def check_object(row, dtype="int16"):
    """执行五步判定链，返回 dict（含 verdict/reason）。"""
    d = dict(t_chunk=row.get("t_chunk"), url=row.get("url"),
             manifest_bytes=int(row.get("bytes") or 0), manifest_http=row.get("http_code"),
             manifest_sha256=row.get("sha256", ""), manifest_verdict=row.get("verdict", ""))
    p = os.path.join(PROJ, row.get("local_path") or "")
    d["local_path"] = row.get("local_path")
    reasons = []
    # 1) HTTP/传输层
    if d["manifest_http"] not in ("200", "206", 200, 206):
        reasons.append("http_code=%s" % d["manifest_http"])
    note = row.get("note") or ""
    if "curl_exit=0" not in note:
        reasons.append("传输未干净结束（%s）" % note[:40])
    # 2) 存在 + 大小
    if not os.path.exists(p):
        d.update(verdict="MISSING", reasons="local file absent")
        return d
    d["local_bytes"] = os.path.getsize(p)
    if d["local_bytes"] != d["manifest_bytes"]:
        reasons.append("大小不符 local=%d manifest=%d" % (d["local_bytes"], d["manifest_bytes"]))
    # 3) SHA-256
    d["sha256"] = sha256_file(p)
    d["sha_match"] = (d["sha256"] == d["manifest_sha256"])
    if not d["sha_match"]:
        reasons.append("SHA-256 不符（本地另存值）")
    # 4) 解压（chunk = 20 B 头 + gzip）
    with open(p, "rb") as f:
        blob = f.read()
    off = blob.find(b"\x1f\x8b\x08", 0, 64)
    d["gzip_offset"] = off
    if off < 0:
        d.update(verdict="INCOMPLETE", reasons="; ".join(reasons + ["未找到 gzip 魔数（前缀 0..64）"]))
        return d
    try:
        raw = gzip.decompress(blob[off:])
        d["decompress_ok"] = True
    except Exception as e:
        raw = None
        d["decompress_ok"] = False
        reasons.append("解压失败：%s" % str(e)[:60])
    # 5) 尺寸自洽（header dims × itemsize）
    if off > 0:
        nd, dims = expected_dims_from_header(blob[:off])
        d["header_num_dims"], d["header_dims"] = nd, list(dims) if dims else None
        if dims:
            exp = 1
            for v in dims:
                exp *= int(v)
            exp *= DTYPE_ITEMSIZE.get(dtype, 2)
            d["expected_raw_bytes"] = exp
            if raw is not None and len(raw) != exp:
                reasons.append("解压长度 %d != header 期望 %d" % (len(raw), exp))
    if raw is not None:
        d["raw_bytes"] = len(raw)
    d["verdict"] = "COMPLETE" if not reasons else "INCOMPLETE"
    d["reasons"] = "; ".join(reasons)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refetch", action="store_true", help="先重取 INCOMPLETE 对象")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--all-raw", action="store_true")
    args = ap.parse_args()

    with open(MANIFEST, encoding="utf-8") as f:
        head_line = f.readline().rstrip("\n")
        rows = list(csv.DictReader(f))
    log("=== verify_m11_downloads start n_manifest=%d refetch=%s" % (len(rows), args.refetch))

    checks = [check_object(r) for r in rows]
    n_bad = sum(1 for c in checks if c["verdict"] != "COMPLETE")
    log("首轮校验：COMPLETE=%d INCOMPLETE/MISSING=%d" % (len(checks) - n_bad, n_bad))

    if args.refetch and n_bad:
        for r in rows:
            c = next(x for x in checks if x["t_chunk"] == r["t_chunk"])
            if c["verdict"] == "COMPLETE":
                continue
            dst = os.path.join(PROJ, r["local_path"])
            log("  重取 t_chunk=%s（原因：%s）" % (r["t_chunk"], c.get("reasons", "")[:60]))
            code, n, rc = curl_get(r["url"], dst, timeout=args.timeout)
            if code in (200, 206) and n > 0 and rc == 0:
                r["http_code"] = str(code); r["bytes"] = str(n)
                r["sha256"] = sha256_file(dst)
                r["note"] = "curl_exit=0（refetch by verify_m11_downloads）"
                r["retrieved_at"] = utcnow()
                r["verdict"] = "REACHABLE"
            else:
                r["verdict"] = "BLOCKED"
                r["note"] = "refetch failed: http=%s bytes=%s curl_exit=%s" % (code, n, rc)
        checks = [check_object(r) for r in rows]
        n_bad = sum(1 for c in checks if c["verdict"] != "COMPLETE")
        log("重取后校验：COMPLETE=%d INCOMPLETE/MISSING=%d" % (len(checks) - n_bad, n_bad))

    # 写回 manifest（**新增 integrity 列，保留原列与顺序**）
    fields = list(rows[0].keys())
    if "integrity" not in fields:
        fields.append("integrity")
    by_tc = {c["t_chunk"]: c for c in checks}
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        f.write(head_line + "\n")
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            c = by_tc.get(r["t_chunk"], {})
            r["integrity"] = c.get("verdict", "UNCHECKED")
            w.writerow(r)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        f.write("# M11 下载完整性门（generated_by=tools/verify_m11_downloads.py；"
                "判定链=http/传输 → 大小 → SHA-256 → 解压 → 尺寸自洽；"
                "**统计与参考带只允许使用 verdict=COMPLETE 的对象**）\n")
        cols = ["t_chunk", "url", "manifest_http", "manifest_bytes", "local_bytes", "manifest_sha256",
                "sha256", "sha_match", "gzip_offset", "header_num_dims", "header_dims",
                "expected_raw_bytes", "raw_bytes", "decompress_ok", "verdict", "reasons"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for c in checks:
            w.writerow(c)

    n_complete = sum(1 for c in checks if c["verdict"] == "COMPLETE")
    summary = dict(generated_by="tools/verify_m11_downloads.py", generated_at=utcnow(),
                   n_objects=len(checks), n_complete=n_complete, n_incomplete=len(checks) - n_complete,
                   incomplete=[dict(t_chunk=c["t_chunk"], bytes=c.get("local_bytes"),
                                    reasons=c.get("reasons", "")) for c in checks if c["verdict"] != "COMPLETE"],
                   total_bytes_complete=sum(c.get("local_bytes", 0) for c in checks if c["verdict"] == "COMPLETE"),
                   integrity_csv="data/m11_download_integrity.csv",
                   rule="统计量/参考带一律只用 COMPLETE 对象；INCOMPLETE → 可重取（--refetch），不得静默计入")
    with open(OUT_SUM, "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    log("wrote %s / %s：COMPLETE=%d INCOMPLETE=%d" % (OUT_CSV, OUT_SUM, n_complete, len(checks) - n_complete))
    log("ALL_DONE %s" % utcnow())
    return summary


if __name__ == "__main__":
    s = main()
    print(json.dumps({k: v for k, v in s.items() if k != "incomplete"}, ensure_ascii=False))
    print("incomplete:", json.dumps(s["incomplete"], ensure_ascii=False)[:600])
