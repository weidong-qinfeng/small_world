#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M11 数据门（G2/P1/P2）· Allen 侧数据获取器（Cell Types 元数据 + per-cell NWB + BO Neuropixels 目录）

交付（清单 §1.2 / §7 P1）：
- `data/m11_raw/allen/celltypes_mouse_metadata.json`  —— Cell Types API 原始返回（小鼠，含 total_rows）
- `data/m11_allen_celltypes_metadata.csv`             —— 逐标本一行的元数据表（含 `ef__*` 电生理特征）
- `data/m11_allen_sources.json`                        —— 各端点的 HTTP/字节/SHA + per-cell NWB 的 HEAD/Range 证据
- per-cell NWB：HEAD（content-length / content-disposition）+ HTTP Range 部分读取 + **HDF5/NWB 头解析**
  （用"稀疏文件"技巧：把 Range 到的前缀放进声明为全长的稀疏文件，使 HDF5 超级块校验通过 →
  可读 header；**数据区未取**，逐项标注 `unavailable`）
- Brain Observatory Neuropixels NWB（S3）：有界目录列举（对象数/总字节）

纪律：curl 子进程；单请求硬超时；**不下载** 477 GB 级 DANDI 资产；增量日志 + `ALL_DONE`。
用法：
    PYTHONHASHSEED=0 ./.venv-m11/bin/python neural_exploration/tools/fetch_m11_allen.py --all
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "m11_raw", "allen")
LOG = os.path.join(RAW, "fetch_m11_allen.log")
API = "https://api.brain-map.org/api/v2"
WKF_ID = "664222041"          # per-cell ephys NWB（清单 §1.2 实测对象）
NWB_EXPECT_BYTES = 91560414   # 清单/本轮 HEAD 实测
BO_S3 = "https://allen-brain-observatory.s3.amazonaws.com"
CYTO_SPECIES = "Mus musculus"


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(m):
    os.makedirs(RAW, exist_ok=True)
    line = "[%s] %s" % (datetime.now(timezone.utc).strftime("%H:%M:%S"), m)
    with open(LOG, "a", buffering=1) as f:
        f.write(line + "\n")
    print(line, flush=True)


def curl(args, timeout=60):
    cmd = ["curl", "-sS", "-L", "--connect-timeout", "8", "--max-time", str(timeout),
           "-w", "%{http_code}|%{size_download}|%{content_type}|%{time_total}"] + args
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=timeout + 15)
    except subprocess.TimeoutExpired:
        return 0, 0, "", 124, 0.0, b""
    parts = p.stdout.decode("utf-8", "replace").strip().split("|")
    code, n, ct, t = 0, 0, "", 0.0
    if len(parts) >= 4:
        try:
            code = int(parts[0]) if parts[0] else 0
            n = int(float(parts[1]))
            t = float(parts[3])
        except ValueError:
            pass
        ct = parts[2]
    # 响应体写到 stdout? 不 —— 调用方用 -o；若用 -o 则 p.stdout 只有 -w 行
    body = b""
    return code, n, ct, p.returncode, t, body


def curl_head(url, timeout=30):
    """HEAD：返回 (code, headers dict)。"""
    try:
        p = subprocess.run(["curl", "-sS", "-I", "-L", "--connect-timeout", "8", "--max-time", str(timeout), url],
                           capture_output=True, timeout=timeout + 10)
    except subprocess.TimeoutExpired:
        return 0, {}
    out = p.stdout.decode("utf-8", "replace")
    code = 0
    m = re.search(r"HTTP/[\d.]+\s+(\d+)", out)
    if m:
        code = int(m.group(1))
    hdr = {}
    for line in out.splitlines():
        if ":" in line and not line.startswith("HTTP/"):
            k, v = line.split(":", 1)
            hdr[k.strip().lower()] = v.strip()
    return code, hdr


def curl_body(url, timeout=90, byte_range=None):
    tmp = os.path.join(RAW, "_tmp_allen_%d.bin" % os.getpid())
    args = ["-o", tmp]
    if byte_range is not None:
        args = ["-r", "%d-%d" % byte_range] + args
    code, n, ct, rc, t, _ = curl(args + [url], timeout=timeout)
    b = b""
    if os.path.exists(tmp):
        with open(tmp, "rb") as f:
            b = f.read()
        os.remove(tmp)
    return code, n, ct, rc, t, b


def list_s3(bucket_url, prefix, page_cap=5, max_keys=1000, timeout=30):
    tok, n_obj, n_by, pages, keys, sizes, truncated = None, 0, 0, 0, [], [], False
    while pages < page_cap:
        u = "%s/?list-type=2&max-keys=%d" % (bucket_url, max_keys)
        if prefix:
            u += "&prefix=%s" % prefix
        if tok:
            u += "&continuation-token=%s" % tok
        code, n, ct, rc, t, b = curl_body(u, timeout=timeout)
        if code != 200 or not b:
            return dict(error="http=%s" % code, n_objects=n_obj, n_bytes=n_by, pages=pages, truncated=True)
        x = b.decode("utf-8", "replace")
        ks = re.findall(r"<Key>([^<]*)</Key>", x)
        ss = [int(v) for v in re.findall(r"<Size>(\d+)</Size>", x)]
        n_obj += len(ks)
        n_by += sum(ss)
        keys += ks
        sizes += ss
        pages += 1
        m = re.search(r"<IsTruncated>(true|false)</IsTruncated>", x)
        m2 = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", x)
        if m and m.group(1) == "false":
            truncated = False
            break
        if not m2:
            truncated = False
            break
        tok = m2.group(1)
        truncated = True
    return dict(n_objects=n_obj, n_bytes=n_by, pages=pages, truncated=truncated,
                keys_sample=keys[:10], size_min=min(sizes) if sizes else None,
                size_max=max(sizes) if sizes else None, keys_all=keys)


# ---------------------------------------------------------------------------
def _ct_query(opts, timeout=0):
    """opts: 选项字符串列表，每个选项**各占一个方括号**（RMA 语法）。"""
    import urllib.parse
    crit = ("model::ApiCellTypesSpecimenDetail,rma::criteria[donor__species$eq'%s'],rma::options%s"
            % (CYTO_SPECIES, "".join("[%s]" % o for o in opts)))
    return "%s/data/query.json?criteria=%s" % (API, urllib.parse.quote(crit, safe=""))


def fetch_celltypes(args):
    """分页取小鼠 Cell Types 元数据（大响应会超时 → 必须分页 + 校验 curl_exit）。"""
    out = dict(species=CYTO_SPECIES, retrieved_at=utcnow(), pages=[])
    # (1) 计数查询（小响应）
    q0 = _ct_query(["num_rows$eq1", "count$eqtrue"])
    code, n, ct, rc, t, b = curl_body(q0, timeout=args.timeout)
    out.update(count_url=q0, count_http_code=code, count_bytes=n, count_curl_exit=rc)
    if not b or rc != 0:
        out["verdict"] = "BLOCKED"
        out["note"] = "计数查询失败（curl_exit=%s）" % rc
        log("celltypes count FAILED http=%s curl_exit=%s" % (code, rc))
        return out, []
    js0 = json.loads(b.decode("utf-8"))
    total = int(js0.get("total_rows") or 0)
    out["total_rows"] = total
    out["n_fields"] = len((js0.get("msg") or [{}])[0].keys())
    out["fields_sample"] = sorted((js0.get("msg") or [{}])[0].keys())[:60]
    out["n_ef_fields"] = sum(1 for k in (js0.get("msg") or [{}])[0].keys() if k.startswith("ef__"))
    out["n_wkf_download_paths"] = sum(1 for v in ((js0.get("msg") or [{}])[0]).values()
                                      if isinstance(v, str) and "/well_known_file_download/" in v)
    log("celltypes count: total_rows=%s n_fields=%s n_ef=%s"
        % (total, out["n_fields"], out["n_ef_fields"]))
    # (2) 分页取行
    rows, start, page_size = [], 0, max(1, args.page_size)
    while start < total and len(out["pages"]) < args.max_pages:
        q = _ct_query(["start_row$eq%d" % start, "num_rows$eq%d" % page_size, "count$eqtrue"])
        code, n, ct, rc, t, b = curl_body(q, timeout=args.timeout)
        ok = (code == 200 and rc == 0 and b)
        nrows = 0
        if ok:
            try:
                js = json.loads(b.decode("utf-8"))
                rr = js.get("msg")
                if not isinstance(rr, list):
                    ok = False
                    out.setdefault("parse_errors", []).append(
                        "start=%d: msg 非列表（API 报错？msg=%s）" % (start, str(rr)[:120]))
                    rr, nrows = [], 0
                else:
                    nrows = len(rr)
                    rows += rr
            except Exception as e:
                ok = False
                out.setdefault("parse_errors", []).append("start=%d: %s" % (start, str(e)[:80]))
        out["pages"].append(dict(start_row=start, num_rows=page_size, http_code=code, bytes=n,
                                 curl_exit=rc, time_total_s=round(t, 3), rows_returned=nrows, ok=bool(ok)))
        log("celltypes page start=%d http=%s rc=%s bytes=%s rows=%s" % (start, code, rc, n, nrows))
        if not ok or nrows == 0:
            break
        start += nrows
    out["num_rows_returned"] = len(rows)
    out["download_complete"] = (len(rows) == total)
    out["verdict"] = "REACHABLE" if rows else "REACHABLE_NO_DATA"
    if not out["download_complete"]:
        out["note"] = ("**未取全**：total_rows=%s，仅落 %d 行（分页上限/超时）→ 该表按 partial 使用，"
                       "禁止把未取行写成 0/未命中" % (total, len(rows)))
    return out, rows


def write_celltypes_csv(rows, path):
    if not rows:
        return 0
    cols = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write("# Allen Cell Types API 元数据表（小鼠；generated_by=tools/fetch_m11_allen.py；"
                "来源 api.brain-map.org model::ApiCellTypesSpecimenDetail；逐标本一行）\n")
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})
    return len(rows)


def fetch_nwb(args):
    url = "%s/well_known_file_download/%s" % (API, WKF_ID)
    ent = dict(url=url, wkf_id=WKF_ID, retrieved_at=utcnow())
    code, hdr = curl_head(url, timeout=args.timeout)
    ent["head_http_code"] = code
    ent["head_content_length"] = int(hdr.get("content-length", 0) or 0)
    ent["head_content_disposition"] = hdr.get("content-disposition", "")
    ent["head_content_type"] = hdr.get("content-type", "")
    c2, n2, ct2, rc2, t2, b2 = curl_body(url, timeout=args.timeout, byte_range=(0, 99))
    ent["range_0_99_http_code"] = c2
    ent["range_0_99_bytes"] = n2
    ent["nwb_expected_bytes"] = NWB_EXPECT_BYTES
    ent["size_match_expect"] = bool(ent["head_content_length"] == NWB_EXPECT_BYTES)
    log("nwb HEAD: %s len=%s disp=%s" % (code, ent["head_content_length"], ent["head_content_disposition"]))

    # 局部/全量文件（若已由外部下载作业落盘则直接用）
    full = os.path.join(RAW, "486940961_ephys.nwb")
    part = os.path.join(RAW, "486940961_ephys_head8MB.bin")
    ent["local_full_bytes"] = os.path.getsize(full) if os.path.exists(full) else 0
    ent["local_partial_bytes"] = os.path.getsize(part) if os.path.exists(part) else 0
    hdr_source = None
    if ent["local_full_bytes"] == NWB_EXPECT_BYTES:
        hdr_source = full
        ent["header_source"] = "full(complete file)"
    else:
        # 取"可见前缀"最长者（部分下载文件 vs 8 MB Range 前缀）
        cands = [(os.path.getsize(x), x) for x in (full, part) if os.path.exists(x)]
        if cands:
            npre, src = max(cands)
            ent["prefix_bytes_used"] = npre
            # 稀疏文件技巧：仅用于读 header（其余为洞；**数据区未取**）
            sparse = os.path.join(RAW, "head_sparse.nwb")
            with open(src, "rb") as fi, open(sparse, "wb") as fo:
                fo.write(fi.read())
            with open(sparse, "r+b") as f:
                f.truncate(NWB_EXPECT_BYTES)
            hdr_source = sparse
            ent["header_source"] = "sparse(prefix %d B + holes)" % npre
    ent["nwb_header"] = parse_nwb_header(hdr_source) if hdr_source else dict(
        status="unavailable", reason="本地无 NWB 前缀/全量文件（仅 HEAD+Range 证据）")
    return ent


def parse_nwb_header(path):
    """尽力解析 NWB(HDF5) 头：根组属性 + 组/数据集清单。逐项 try/except，失败如实登记。"""
    out = dict(path=os.path.relpath(path, os.path.dirname(ROOT)), status="ok", errors=[],
               groups=[], datasets=[], n_groups=0, n_datasets=0)
    try:
        import h5py
    except Exception as e:
        return dict(status="unavailable", reason="h5py 不可用（%s）" % e)
    try:
        f = h5py.File(path, "r")
    except Exception as e:
        return dict(status="unavailable", reason="HDF5 打开失败：%s: %s" % (type(e).__name__, str(e)[:160]))
    with f:
        try:
            out["root_attrs"] = {k: (str(v)[:120]) for k, v in list(f.attrs.items())}
        except Exception as e:
            out["errors"].append("root attrs: %s" % str(e)[:100])
        try:
            out["root_keys"] = list(f.keys())
        except Exception as e:
            out["errors"].append("root keys: %s" % str(e)[:100])
            out["root_keys"] = []
        # 逐 root 对象自省（前缀模式下部分对象元数据在洞内 ⇒ 失败者如实登记，不猜）
        out["root_objects"] = {}
        for k in out["root_keys"]:
            try:
                o = f[k]
                if isinstance(o, h5py.Dataset):
                    d = dict(kind="dataset", shape=list(o.shape), dtype=str(o.dtype))
                    if o.size <= 4:
                        try:
                            d["value"] = repr(o[()])[:120]
                        except Exception as e2:
                            d["value_error"] = str(e2)[:80]
                    out["root_objects"][k] = d
                else:
                    out["root_objects"][k] = dict(kind="group", n_keys=len(o.keys()),
                                                  keys=list(o.keys())[:10])
            except Exception as e2:
                out["root_objects"][k] = dict(kind="error", error=str(e2)[:100])
        def walk(name, obj):
            try:
                if isinstance(obj, h5py.Dataset):
                    out["datasets"].append(dict(name="/" + name, shape=list(obj.shape), dtype=str(obj.dtype)))
                else:
                    out["groups"].append("/" + name)
            except Exception as e:
                out["errors"].append("%s: %s" % (name, str(e)[:80]))

        try:
            f.visititems(walk)
            out["visit_complete"] = True
        except Exception as e:
            out["visit_complete"] = False
            out["errors"].append("visititems 提前终止：%s" % str(e)[:120])
        # 已知 NWB 结构点：acquisition/timeseries 的 sweep 语义
        # 注意：前缀模式下 `ts.keys()` 可能失败（symbol table 在洞内）⇒ 用 visititems 已发现的组名兜底
        ts_names = [g.strip("/") for g in out["groups"] if g.startswith("/acquisition/timeseries/")]
        try:
            ts = f["acquisition/timeseries"]
            out["timeseries_keys"] = list(ts.keys())
        except Exception as e2:
            out["errors"].append("timeseries.keys(): %s" % str(e2)[:100])
            out["timeseries_keys"] = ts_names
        for nm in (ts_names[:1] or out["timeseries_keys"][:1]):
            try:
                sw = f["acquisition/timeseries/%s" % nm.split("/")[-1]]
                out["first_sweep"] = dict(name=nm,
                                          attrs={k: str(v)[:140] for k, v in list(sw.attrs.items())[:14]})
                break
            except Exception as e2:
                out["errors"].append("sweep %s: %s" % (nm, str(e2)[:100]))

    out["n_groups"] = len(out["groups"])
    out["n_datasets"] = len(out["datasets"])
    out["status"] = "ok" if out.get("visit_complete") else "partial"
    return out


def fetch_bo_neuropixels(args):
    r = list_s3(BO_S3, "visual-coding-neuropixels/ecephys-cache", page_cap=args.bo_pages, timeout=args.timeout)
    r["retrieved_at"] = utcnow()
    r["prefix"] = "visual-coding-neuropixels/ecephys-cache"
    log("BO neuropixels ecephys-cache: objects=%s bytes=%s truncated=%s"
        % (r.get("n_objects"), r.get("n_bytes"), r.get("truncated")))
    # 对象级验证：取第一个 nwb 的 HEAD
    first = next((k for k in r.get("keys_all", []) if k.endswith(".nwb")), None)
    if first:
        u = "%s/%s" % (BO_S3, first)
        c, hdr = curl_head(u, timeout=args.timeout)
        r["object_probe"] = dict(url=u, http_code=c, content_length=int(hdr.get("content-length", 0) or 0),
                                 retrieved_at=utcnow(),
                                 verdict="REACHABLE" if c in (200, 206) else "BLOCKED")
        log("BO object probe: %s %s len=%s" % (c, first[-40:], r["object_probe"]["content_length"]))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--celltypes", action="store_true")
    ap.add_argument("--nwb", action="store_true")
    ap.add_argument("--bo", action="store_true")
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--num-rows", type=int, default=2000)
    ap.add_argument("--bo-pages", type=int, default=5)
    ap.add_argument("--page-size", type=int, default=400)
    ap.add_argument("--max-pages", type=int, default=20)
    args = ap.parse_args()
    os.makedirs(RAW, exist_ok=True)
    log("=== fetch_m11_allen start argv=%s" % " ".join(sys.argv[1:]))
    # **合并**写入（避免只跑 --nwb 时丢掉 celltypes 等其它 section —— 本轮踩到，已修）
    prev_path = os.path.join(DATA, "m11_allen_sources.json")
    out = {}
    if os.path.exists(prev_path):
        try:
            with open(prev_path) as f:
                out = json.load(f)
        except Exception:
            out = {}
    out.update(generated_by="tools/fetch_m11_allen.py", generated_at=utcnow(),
               last_run_argv=" ".join(sys.argv[1:]))
    if args.all or args.celltypes:
        ent, rows = fetch_celltypes(args)
        prev = out.get("celltypes_api") or {}
        # **瞬态失败不得降级既有产物**（本轮踩到：一次 curl 超时把 1920 行覆盖成 0 行）
        if not rows and prev.get("num_rows_returned"):
            prev = dict(prev)
            prev.setdefault("attempts", []).append({k: ent.get(k) for k in
                                                    ("retrieved_at", "verdict", "count_http_code", "note")})
            prev["attempts"][-1]["failed_run"] = True
            out["celltypes_api"] = prev
            log("⚠️ celltypes 本次失败（rows=0）→ **保留既有 %s 行结果**，失败尝试已登记"
                % prev.get("num_rows_returned"))
        else:
            out["celltypes_api"] = ent
            if rows:
                n = write_celltypes_csv(rows, os.path.join(DATA, "m11_allen_celltypes_metadata.csv"))
                out["celltypes_api"]["csv_rows"] = n
                log("wrote data/m11_allen_celltypes_metadata.csv (%d rows)" % n)
    if args.all or args.nwb:
        out["per_cell_nwb"] = fetch_nwb(args)
        log("nwb header status=%s n_datasets=%s" % (out["per_cell_nwb"]["nwb_header"].get("status"),
                                                    out["per_cell_nwb"]["nwb_header"].get("n_datasets")))
    if args.all or args.bo:
        out["bo_neuropixels"] = fetch_bo_neuropixels(args)
    # CSV 落盘核查（**独立于本次 API 运行**）：行数 + 物种列全为小鼠
    csvp = os.path.join(DATA, "m11_allen_celltypes_metadata.csv")
    if os.path.exists(csvp):
        with open(csvp, encoding="utf-8") as f:
            rr = list(csv.DictReader([ln for ln in f if not ln.startswith("#")]))
        out.setdefault("celltypes_api", {})
        out["celltypes_api"]["csv_rows_on_disk"] = len(rr)
        out["celltypes_api"]["csv_verified_species_all_mouse"] = all(
            (r.get("donor__species") or "") == CYTO_SPECIES for r in rr) if rr else False
        log("celltypes CSV on disk: %d rows, species_all_mouse=%s"
            % (len(rr), out["celltypes_api"]["csv_verified_species_all_mouse"]))
    with open(os.path.join(DATA, "m11_allen_sources.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=False, ensure_ascii=False)
    log("wrote data/m11_allen_sources.json")
    log("ALL_DONE %s" % utcnow())


if __name__ == "__main__":
    main()
