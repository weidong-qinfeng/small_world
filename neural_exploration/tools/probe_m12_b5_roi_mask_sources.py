#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M12 · B-5 取数探针：**第二个 field 的真实 ROI 掩膜**（复现 field，非主口径）

目的
----
J2 验收条件 b) 要求**(b) 的待定稿带须给"跨 field 与跨时间窗离散度（≥2 个 field）"**。
主 field（`session_4_scan_7_field_4`）的掩膜 8 个 chunk 已在盘（M11 落盘件，本节点复核
sha256 与线上逐字节一致）。本工具只新增**第二个 field** 的掩膜：
`field_seg_masks/session_5_scan_3_field_1/2500_2500_10000/<x0>-<x1>_<y0>-<y1>_0-1`
（与 `functional_subset_f2/` 的 100×100 电影同源同 field）。

纪律（M12 env_notes L2/L4/L5/L9/L20）
-------------------------------------
① **先测吞吐再取数**，逐件登记 (TTFB, 传输段, 对象大小) **三元组**；
② 完整性门：`size_download == Content-Length`（**不是 206 续传响应的那个**，见 L20）
   + sha256 + 复算幂等（重跑同 URL 应得同一 sha256）；
③ 落**新目录**，不覆盖任何既有件；
④ 父进程硬回收在 `curl` 层用 `--connect-timeout` + `--max-time` 兜底。

用法
----
    PYTHONHASHSEED=0 OMP_NUM_THREADS=1 ./.venv-m11/bin/python \
        neural_exploration/tools/probe_m12_b5_roi_mask_sources.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, "data", "m12_b5_roi_mask")
F2DIR = os.path.join(OUTDIR, "field2_masks")

S3 = "https://bossdb-open-data.s3.amazonaws.com"
FUNC = "iarpa_microns/minnie/functional_data"
FIELD2 = "session_5_scan_3_field_1"
FIELD4 = "session_4_scan_7_field_4"
SCALE = "2500_2500_10000"
# 与 field4 逐值相同的网格（两 field 的 info.json sha256 相同 → 几何相同）
XS, YS = (0, 128, 248), (0, 128, 256, 384, 440)


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def curl_get(url, dest, max_time=120):
    """取一件并返回 (TTFB, 传输段, 大小, code, content_length, curl_exit)。"""
    cmd = ["curl", "-sS", "-L", "--connect-timeout", "10", "--max-time", str(max_time),
           "-D", dest + ".hdr", "-o", dest,
           "-w", "%{time_starttransfer}|%{time_total}|%{size_download}|%{http_code}|%{speed_download}"]
    p = subprocess.run(cmd + [url], capture_output=True, text=True, timeout=max_time + 30)
    if p.returncode != 0:
        return dict(curl_exit=p.returncode, stderr=p.stderr.strip()[:400], url=url)
    ttfb, total, size, code, speed = p.stdout.strip().split("|")
    clen = None
    for line in open(dest + ".hdr", "rb").read().decode("latin1").split("\r\n"):
        if line.lower().startswith("content-length:"):
            clen = int(line.split(":", 1)[1].strip())
    return dict(url=url, http_code=int(code), ttfb_s=round(float(ttfb), 4),
                total_s=round(float(total), 4),
                transfer_s=round(float(total) - float(ttfb), 4),
                bytes=int(size), content_length=clen, bytes_per_s=round(float(speed), 2),
                curl_exit=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-existing", action="store_true", default=True)
    args = ap.parse_args()
    os.makedirs(F2DIR, exist_ok=True)
    recs = []
    out = dict(generated_by="neural_exploration/tools/probe_m12_b5_roi_mask_sources.py",
               generated_at=utcnow(), note="只新增；不覆盖既有件", field=FIELD2, objects=[])

    # (0) 两个 field 的 info（几何/编码声明）
    infos = {}
    for f in (FIELD4, FIELD2):
        url = "%s/%s/field_seg_masks/%s/info" % (S3, FUNC, f)
        dest = os.path.join(OUTDIR, "info_%s.json" % f)
        r = curl_get(url, dest)
        if r.get("curl_exit") == 0:
            r["sha256"] = hashlib.sha256(open(dest, "rb").read()).hexdigest()
            r["local"] = os.path.relpath(dest, os.path.dirname(ROOT))
            infos[f] = json.load(open(dest))
        out["objects"].append(dict(kind="info", field=f, **r))
        print("info %-24s http=%s bytes=%s clen=%s ttfb=%.3f transfer=%.3f sha=%s"
              % (f, r.get("http_code"), r.get("bytes"), r.get("content_length"),
                 r.get("ttfb_s", -1), r.get("transfer_s", -1), r.get("sha256", "-")[:16]))
    out["info_sha256_identical"] = (len(infos) == 2 and
                                    hashlib.sha256(json.dumps(infos[FIELD4], sort_keys=True).encode()).hexdigest()
                                    == hashlib.sha256(json.dumps(infos[FIELD2], sort_keys=True).encode()).hexdigest())

    # (1) 8 个 chunk
    for i in range(len(XS) - 1):
        for j in range(len(YS) - 1):
            key = "%d-%d_%d-%d_0-1" % (XS[i], XS[i + 1], YS[j], YS[j + 1])
            url = "%s/%s/field_seg_masks/%s/%s/%s" % (S3, FUNC, FIELD2, SCALE, key)
            dest = os.path.join(F2DIR, key + ".bin")
            if args.skip_existing and os.path.exists(dest) and os.path.getsize(dest) > 0:
                recs.append(dict(kind="chunk", key=key, url=url, verdict="SKIPPED_EXISTS",
                                 local=os.path.relpath(dest, os.path.dirname(ROOT)),
                                 bytes=os.path.getsize(dest)))
                continue
            r = curl_get(url, dest)
            if r.get("curl_exit") == 0:
                r["sha256"] = hashlib.sha256(open(dest, "rb").read()).hexdigest()
                r["local"] = os.path.relpath(dest, os.path.dirname(ROOT))
                r["integrity"] = ("COMPLETE" if r.get("content_length") is not None
                                  and r["bytes"] == r["content_length"] else "INCOMPLETE")
                r["verdict"] = "REACHABLE"
                out["objects"].append(dict(kind="chunk", key=key, **r))
                print("chunk %-24s http=%s bytes=%-6s clen=%-6s ttfb=%.3f transfer=%.3f %s"
                      % (key, r["http_code"], r["bytes"], r["content_length"], r["ttfb_s"],
                         r["transfer_s"], r["integrity"]))
            else:
                out["objects"].append(dict(kind="chunk", key=key, **r))
                print("chunk %-24s FAILED curl_exit=%s %s" % (key, r.get("curl_exit"), r.get("stderr")))

    ok = [o for o in out["objects"] if o.get("kind") == "chunk" and o.get("verdict") == "REACHABLE"
          and o.get("integrity") == "COMPLETE"]
    skipped = [o for o in out["objects"] if o.get("kind") == "chunk" and o.get("verdict") == "SKIPPED_EXISTS"]
    total_bytes = sum(o["bytes"] for o in out["objects"])
    tt = [o["ttfb_s"] for o in out["objects"] if o.get("ttfb_s") is not None]
    out["summary"] = dict(
        n_chunk_complete=len(ok), n_chunk_skipped_existing=len(skipped),
        n_objects=len(out["objects"]), total_bytes=total_bytes,
        ttfb_s_range=[min(tt), max(tt)] if tt else None,
        etas="全部小对象（≈2–6 KB/件）⇒ 端到端 s 级；**不作为链路吞吐**（L5）")
    print(json.dumps(out["summary"], ensure_ascii=False))
    with open(os.path.join(OUTDIR, "field2_fetch_manifest.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("wrote", os.path.relpath(os.path.join(OUTDIR, "field2_fetch_manifest.json"), os.path.dirname(ROOT)))
    return out


if __name__ == "__main__":
    main()
