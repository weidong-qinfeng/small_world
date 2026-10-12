#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M12 数据门 G2-M · B1（提前备料）节点探针：**S2 CCFv3 本体 / S7 Allen ecephys 汇总表 / S5 IBL 行为白名单**

用途
----
对《生物仿真M12实施清单.md》§2 的 **S2 / S7 / S5** 三个源，各走一次完整闭环：

    T1 列举/计数 → T2 对象级取证 → T3 **当场吞吐实测（30 s 探针）** → T4 ETA 裁决
    → T5 内容级取证（三级完整性门）→ T6 装配可行性 → 四态判定

**三条硬约束（本工具的立身之处）**
1. **先测吞吐、再决定取数**：取数决策行必须能回溯到**当场**实测 `speed_Bps`，**不得**按锚外推
   （M11-R40 / M11-R56：实测同对象吞吐可在数小时内摆动 **3,600×**）。
2. **父进程硬回收**：所有网络调用 = `Popen` + `communicate(timeout)` + `kill()`
   （`curl --max-time` 实测失效：一次挂死 **925.9 s**，M11 L9.3a）。
3. **完整性门三级**：① `size == Content-Length`；② S3 多段对象 **ETag 复算**
   （`md5(concat(md5(part_i))) + "-N"`，part_size 由候选集试出）；③ sha256（标注基准
   `file_bytes`/`payload`/`semantic`）。**未完成下载 ⇒ `INCOMPLETE`，不得进统计**（M11-R63）。

输出（**只增不改 / append-only**；每行带 utc）
----
- `data/m12_throughput_measured.csv`  `utc,source,http_code,bytes,seconds,bytes_per_s,probe_or_fetch`
- `data/m12_probe_b1_integrity.csv`   完整性门逐件
- `data/m12_probe_b1_reachability.csv` 源级四态（M11 词表 + M12 新增 `TOO_SLOW`/`UNLOADABLE`）
- `data/m12_ibl_sessions.csv`         S5 预注册 session 白名单 + A/B/C/D 归属
- `data/m12_probe_b1_summary.json`    断言/计数元数据（供报告引用，避免引陈旧数字）
- 落盘对象：`data/m12_raw/{s2,s7,s5}/…`
- 增量日志：`data/m12_raw/probe/probe_m12_b1.log`（末行 `ALL_DONE`）

判定词表：`REACHABLE` / `REACHABLE_NO_DATA` / `NEEDS_TOKEN` / `NEEDS_VERIFY` / `BLOCKED` /
`NOT_EXIST` / **`TOO_SLOW`（T4 ETA 超限）** / **`UNLOADABLE`（T6 装配不可行）**。

环境口径（**不得凭直觉选解释器**，M11-R34）
----
- 网络 + 解析本工具**只用标准库**；`--parse-pqt` 分支需要 `pyarrow` ⇒ 用
  `./.venv-m10/bin/python`（numpy 2.0.2 / pandas 2.3.3 / **pyarrow 21.0.0**）。
- `.venv-m11`（numpy 2.0.2 / h5py 3.14.0）**无 pandas/pyarrow**。
- 本工具**不 import torch、不 import numpy、不起任何仿真、不触碰 MPS/GPU**（派发硬约束）。

用法
----
    cd /Users/weidong/ai/small_world
    PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
      ./.venv-m10/bin/python neural_exploration/tools/probe_m12_sources.py --only S2
    … --only S7 / --only S5 / --only all
    … --probe-only          # 只跑 30 s 吞吐探针，不取数
    … --max-sessions 40     # S5 白名单规模（默认 40，**预注册**）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from urllib.parse import quote

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../neural_exploration
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "m12_raw")
LOG_DIR = os.path.join(RAW, "probe")
LOG_PATH = os.path.join(LOG_DIR, "probe_m12_b1.log")

CSV_THROUGHPUT = os.path.join(DATA, "m12_throughput_measured.csv")
CSV_INTEGRITY = os.path.join(DATA, "m12_probe_b1_integrity.csv")
CSV_REACH = os.path.join(DATA, "m12_probe_b1_reachability.csv")
CSV_IBL = os.path.join(DATA, "m12_ibl_sessions.csv")
CSV_REACH2 = os.path.join(DATA, "m12_source_reachability.csv")
JSON_SUMMARY = os.path.join(DATA, "m12_probe_b1_summary.json")

THROUGHPUT_HEADER = "utc,source,http_code,bytes,seconds,bytes_per_s,probe_or_fetch"
INTEGRITY_HEADER = (
    "utc,t,source,url,http_code,content_length,local_bytes,size_match,etag,etag_n,part_size_tried,"
    "etag_recomputed,etag_match,sha256_file_bytes,sha256_payload,sha256_semantic,sha_basis,"
    "parse_ok,rows_parsed,rows_expected,curl_exit,verdict,reasons"
)
REACH_HEADER = (
    "utc,source_id,source,category,kind,url,http_code,http_code_pass2,bytes,content_type,etag,"
    "accept_ranges,time_total_s,speed_Bps,probe_1MiB_Bps,probe_30s_Bps,eta_s,verdict,"
    "verdict_scope,missing_dimension,note"
)
REACH2_HEADER = (
    "utc,source_id,source,category,kind,url,http_code,http_code_pass2,bytes,content_type,speed_Bps,"
    "verdict,verdict_scope,curl_exit,evidence_level,sha_basis,sha256_prefix,provenance,source_url,"
    "retrieved_at,wrong_usage_consequence,missing_dimension,note"
)
IBL_HEADER = (
    "rank,lab,subject,date,session_no,eid_key,alf_prefix,trials_table_key,trials_table_bytes,"
    "partition,assigned_before_stats,enum_rule,note"
)

# ---------------------------------------------------------------------------
# 端点（**不臆造**：全部由本轮列举/计数实测确认）
# ---------------------------------------------------------------------------
ALLEN_API = "https://api.brain-map.org/api/v2/data/query.json"
S2_COUNT_URL = ALLEN_API + "?criteria=model::Structure,rma::options[count$eqtrue]"
S2_FULL_URL = ALLEN_API + "?criteria=model::Structure,rma::options[num_rows$eqall]"
S2_PAGE_URL = ALLEN_API + "?criteria=model::Structure,rma::options[num_rows$eq50]"
S2_GRAPH_TMPL = ALLEN_API + "?criteria=model::Structure,rma::criteria[graph_id$eq{0}],rma::options[count$eqtrue]"

ALLEN_BO = "https://allen-brain-observatory.s3.amazonaws.com"
S7_URLS = [
    ALLEN_BO + "/visual-coding-neuropixels/ecephys-cache/sessions.csv",
    ALLEN_BO + "/visual-coding-neuropixels/ecephys-cache/probes.csv",
]
# M11 已实测落盘件的 sha256（独立交叉核对用；非本工具产物）
S7_M11_SHA = {
    "sessions.csv": "cdd12b8becac84d284f222718a2a94f4155893fb9f26f1a9c725f541acc738f8",
    "probes.csv": "f555ce63317d5b6a83bfafd9f7e6a7b7393c0d7416952bafae0fe67412bfb6bf",
    "sessions.csv.m11_bytes": 7849,
    "probes.csv.m11_bytes": 26954,
}

IBL_BUCKET = "https://ibl-brain-wide-map-public.s3.amazonaws.com"


def ibl_url(key):
    """IBL 桶的对象 URL。

    ★ **实测陷阱**：IBL 的 revision 子目录名形如 `alf/#2024-07-15#/`，其中的 `#` 会被 curl
    当作 **URL fragment 分隔符** ⇒ 请求实际打到 `.../alf/` ⇒ **404**（本轮 3 号 session 实测）。
    ⇒ 必须把 `#` 编码为 `%23`（`quote(key, safe="/")`）。
    """
    return "%s/%s" % (IBL_BUCKET, quote(key, safe="/"))
IBL_AGG_TRIALS = IBL_BUCKET + "/aggregates/2026_Q2_IBL_et_al_BWM/trials.pqt"  # 无 `#`，无需编码
S5FULL_DIR = os.path.join(RAW, "s5_aggregate")          # **独立子目录**（派发要求 ⑤）
CSV_AGG_SESSIONS = os.path.join(DATA, "m12_ibl_trials_aggregate.csv")
AGG_SESS_HEADER = "rank,session,n_trials,share_of_total"
# R72 批准的取数上限（**>50 MB 才须先问；本件 24,212,510 B**）
S5FULL_EXPECT_BYTES = 24212510  # 24,212,510 B（本轮实测）
IBL_PUBLIC_TREE = IBL_BUCKET + "/data/public_tree.txt"

# 探针口径（预注册，**事中不得择口径**）
PROBE_1MIB_MAXTIME = 30
PROBE_ESCALATE_RANGE = 16777216 - 1  # 16 MiB-1：短探针跑得太快（链路快）时升级到本档
PROBE_ESCALATE_FAST_S = 1.0          # 1 MiB 在 <1.0 s 内完成 ⇒ 判定"短探针被握手主导"

CANDIDATE_PART_SIZES = [
    5 * 1024 * 1024, 8 * 1024 * 1024, 16 * 1024 * 1024,
    32 * 1024 * 1024, 64 * 1024 * 1024, 100 * 1024 * 1024,
]

_THROUGHPUT_ROWS: list[str] = []
_INTEGRITY_ROWS: list[str] = []
_REACH_ROWS: list[str] = []
_REACH2_ROWS: list[str] = []


def load_json_safe(path):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return None


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    line = "%s  %s" % (utc_now(), msg)
    print(line, flush=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def csv_escape(v) -> str:
    s = "" if v is None else str(v)
    if any(c in s for c in ',"\n'):
        return '"' + s.replace('"', '""') + '"'
    return s


def ensure_header(path: str, header: str, comment: str = "") -> None:
    """仅当文件不存在或为空时写表头（**旧行为逐字保留**，append-only）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    with open(path, "w", encoding="utf-8") as f:
        if comment:
            f.write("# " + comment + "\n")
        f.write(header + "\n")


def record_throughput(source, http_code, nbytes, seconds, probe_or_fetch):
    bps = (nbytes / seconds) if seconds and seconds > 0 else 0.0
    row = [utc_now(), source, http_code, nbytes, "%.3f" % seconds, "%.1f" % bps, probe_or_fetch]
    _THROUGHPUT_ROWS.append(",".join(csv_escape(x) for x in row))
    log("THROUGHPUT  %s  code=%s bytes=%s %.3fs  %.1f B/s  [%s]"
        % (source, http_code, nbytes, seconds, bps, probe_or_fetch))
    return bps


def flush_throughput():
    if not _THROUGHPUT_ROWS:
        return
    ensure_header(
        CSV_THROUGHPUT, THROUGHPUT_HEADER,
        "M12 数据门 S2/S7/S5 的**当场实测吞吐**依据（generated_by=tools/probe_m12_sources.py；"
        "口径：取数决策必须有**当场**实测吞吐支撑，禁止按锚/按早前实测外推 —— M11-R40/R56）")
    with open(CSV_THROUGHPUT, "a", encoding="utf-8") as f:
        f.write("\n".join(_THROUGHPUT_ROWS) + "\n")
    _THROUGHPUT_ROWS.clear()


def flush_integrity():
    if not _INTEGRITY_ROWS:
        return
    ensure_header(
        CSV_INTEGRITY, INTEGRITY_HEADER,
        "M12 S2/S7/S5 完整性门（generated_by=tools/probe_m12_sources.py；三级=①size==Content-Length "
        "②S3 多段 ETag 复算 md5(concat(md5(part_i)))+'-N' ③sha256(标注基准)；"
        "verdict=COMPLETE 才允许进统计，INCOMPLETE 必须重取）")
    with open(CSV_INTEGRITY, "a", encoding="utf-8") as f:
        f.write("\n".join(_INTEGRITY_ROWS) + "\n")
    _INTEGRITY_ROWS.clear()


def flush_reach2():
    if not _REACH2_ROWS:
        return
    ensure_header(
        CSV_REACH2, REACH2_HEADER,
        "M12 数据门 §2 数据源可达性实测（generated_by=tools/probe_m12_sources.py --only REACH；"
        "**清单 §2 的 S1–S12 全部源 + 子端点**；每行必带 utc/provenance/source_url/retrieved_at/"
        "evidence_level/sha_basis/wrong_usage_consequence，**空值为 0**；"
        "verdict 词表=REACHABLE/REACHABLE_NO_DATA/NEEDS_TOKEN/NEEDS_VERIFY/BLOCKED/BLOCKED_DNS/"
        "NOT_EXIST/TOO_SLOW/UNLOADABLE/NOT_APPLICABLE）")
    with open(CSV_REACH2, "a", encoding="utf-8") as f:
        f.write("\n".join(_REACH2_ROWS) + "\n")
    _REACH2_ROWS.clear()


def flush_reach():
    if not _REACH_ROWS:
        return
    ensure_header(
        CSV_REACH, REACH_HEADER,
        "M12 S2/S7/S5 源级可达性（generated_by=tools/probe_m12_sources.py；"
        "**本表只是 §2.5 要求的 ≥40 源 `m12_source_reachability.csv` 的 3 源前缀，非其替代**；"
        "verdict 词表=REACHABLE/REACHABLE_NO_DATA/NEEDS_TOKEN/NEEDS_VERIFY/BLOCKED/NOT_EXIST/TOO_SLOW/UNLOADABLE）")
    with open(CSV_REACH, "a", encoding="utf-8") as f:
        f.write("\n".join(_REACH_ROWS) + "\n")
    _REACH_ROWS.clear()


# ---------------------------------------------------------------------------
# curl 包装（**父进程硬回收**是强制项）
# ---------------------------------------------------------------------------
def _hdr_dict(hdr_text: str) -> dict:
    d = {}
    for line in hdr_text.splitlines():
        if ":" in line and not line.startswith("HTTP/"):
            k, _, v = line.partition(":")
            d[k.strip().lower()] = v.strip()
    return d


def run_curl(url, timeout=30, byte_range=None, out_path=None, hdr_path=None,
             extra=None, max_filesize=None, resume=False, fail_on_error=True):
    """单次 curl。返回 dict(code, nbytes, seconds, speed, exit, err, headers, out_path)。"""
    cmd = ["curl", "-sS", "-g", "-L", "--connect-timeout", "8", "--max-time", str(timeout),
           "-w", "%{http_code}|%{size_download}|%{content_type}|%{time_total}|%{speed_download}|"
                 "%{num_redirects}|%{time_starttransfer}|%{time_connect}"]
    # ★ `--fail` 只用于**会落盘成对象**的取数（防错误页被当对象；M12-M-04）。
    #   **探针**必须允许错误体进来，否则无法区分 NEEDS_TOKEN / AccessDenied / 普通 403
    #   （本轮实测：OpenAlyx 的 403 正文含 "Authentication credentials were not provided."，
    #    被 --fail 吞掉后会误判为 BLOCKED）。
    if fail_on_error:
        cmd += ["--fail"]
    if resume:
        cmd += ["-C", "-"]
    if out_path:
        cmd += ["-o", out_path]
    else:
        cmd += ["-o", os.devnull]
    if hdr_path:
        cmd += ["-D", hdr_path]
    if byte_range is not None:
        cmd += ["-r", "%d-%d" % byte_range]
    if max_filesize is not None:
        cmd += ["--max-filesize", str(max_filesize)]
    if extra:
        cmd += list(extra)
    cmd.append(url)

    rec = dict(code=0, nbytes=0, seconds=0.0, speed=0.0, exit=0, err="", ttfb=0.0,
               connect=0.0, headers={}, out_path=out_path, cmd=" ".join(cmd))
    t0 = time.time()
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            so, se = proc.communicate(timeout=timeout + 10)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                so, se = proc.communicate(timeout=10)
            except Exception:
                so, se = b"", b""
            rec["exit"] = 124
            rec["seconds"] = time.time() - t0
            rec["err"] = "hard kill by parent (curl --max-time %ss 未生效)" % timeout
            rec["code"] = 0
            if hdr_path and os.path.exists(hdr_path):
                rec["headers"] = _hdr_dict(open(hdr_path, encoding="utf-8", errors="replace").read())
            return rec
    except Exception as e:  # pragma: no cover
        rec["exit"] = 125
        rec["err"] = "spawn failed: %s" % str(e)[:160]
        return rec

    rec["exit"] = proc.returncode
    txt = (so or b"").decode("utf-8", "replace").strip()
    parts = txt.split("|")
    if len(parts) >= 5:
        rec["code"] = parts[0]
        try:
            rec["nbytes"] = int(float(parts[1]))
        except ValueError:
            rec["nbytes"] = 0
        try:
            rec["seconds"] = float(parts[3])
        except ValueError:
            rec["seconds"] = time.time() - t0
        try:
            rec["speed"] = float(parts[4])
        except ValueError:
            rec["speed"] = 0.0
        for idx, key in ((6, "ttfb"), (7, "connect")):
            if len(parts) > idx:
                try:
                    rec[key] = float(parts[idx])
                except ValueError:
                    rec[key] = 0.0
    if se:
        rec["err"] = se.decode("utf-8", "replace")[:200]
    if hdr_path and os.path.exists(hdr_path):
        rec["headers"] = _hdr_dict(open(hdr_path, encoding="utf-8", errors="replace").read())
    return rec


def probe_plain_get(source, url, label, max_time=PROBE_1MIB_MAXTIME):
    """对**不支持 Range** 的端点（实测：`api.brain-map.org` 的动态 JSON 查询端点 ⇒ `-r` 挂死 30 s
    零字节）改用 **时间上限 GET** 测持续吞吐。

    口径：curl `--max-time = max_time`；被父进程硬回收时以**落盘文件字节数 / 实际墙钟**计
    （`-w` 不会输出）⇒ 该数只可能**低估**（含握手 + 尾部未 flush）。

    ★ **零字节探针 != 端点慢**（本轮实测教训）：本端点 **TTFB≈31 s**（服务端生成 13,709 行 SQL 结果），
    30 s 窗口内 curl 收到 **0 字节**。此时 `B/s=0` **不是**链路吞吐，而是"探针窗短于服务端生成时延"。
    => 判定为 `probe_window_too_short`，**必须**在登记 TTFB 后转入"整件取数即测量"路径。
    返回 (bps, note, window_too_short)。
    """
    fd = os.path.join(LOG_DIR, "_probe_plain.bin")
    if os.path.exists(fd):
        os.remove(fd)
    t0 = time.time()
    rec = run_curl(url, timeout=max_time, out_path=fd)
    wall = time.time() - t0
    nbytes = os.path.getsize(fd) if os.path.exists(fd) else 0
    if os.path.exists(fd):
        os.remove(fd)
    ttfb = rec.get("ttfb") or wall
    if nbytes == 0:
        tag = "probe_window_too_short(ttfb=%.1fs)" % ttfb
        bps = record_throughput(
            source + " " + label + " [probe 时间上限 GET %ss 窗口内 0 字节]" % max_time,
            rec["code"] or ("rc=%d" % rec["exit"]), 0, max(wall, 1e-6), tag)
        note = ("**0 字节**：TTFB=%.1f s >= 探针窗 %ss => **探针窗短于服务端生成时延**；"
                "`B/s=0` **不是**链路吞吐（**不得据此判 TOO_SLOW**）；转'整件取数即测量'路径"
                % (ttfb, max_time))
        return bps, note, True
    tag = "probe_30s_plainGET" if rec["exit"] == 0 else "probe_plainGET_hardkill"
    bps = record_throughput(source + " " + label + " [probe 时间上限 GET %ss]" % max_time,
                            rec["code"] or ("rc=%d" % rec["exit"]), nbytes, max(wall, 1e-6), tag)
    note = ("时间上限 GET %ss 实测 %.1f B/s（%.1f s 收 %d B，TTFB=%.1f s，exit=%s：端点不支持 Range => "
            "以时间上限 GET 替代 1 MiB Range 档；该数只可能低估）"
            % (max_time, bps, wall, nbytes, ttfb, rec["exit"]))
    return bps, note, False


def probe_throughput(source, url, object_bytes=None, max_time=PROBE_1MIB_MAXTIME, label="",
                     ref_url=None, ref_label="", allow_plain_fallback=True):
    """T3 **当场吞吐实测**。

    预注册口径（**事中不得择口径**）：
      P-a `probe_1MiB`        ：`-r 0-1048575 --max-time 30`（清单 §2.3 T3 模板口径）。
      P-b 若 P-a 收满 1 MiB 且 <1.0 s 完成（链路快、握手持平）⇒ 升级 `probe_30s_16MiB`（16 MiB 档）。
      P-c 若 P-a 收到 **<1 MiB** 且**不是**被 `--max-time` 截断 ⇒ 目标对象本身 <1 MiB
          （服务端对越界 Range 返回全件）⇒ 记 `probe_whole_object`，**该数只反映"单件端到端"，
          不反映链路持续吞吐**；此时若提供同通道大对象 `ref_url`，另跑 `probe_ref_1MiB` 作为
          **同通道持续吞吐**（两者并列，不得混为一个数）。
      P-d 若 `-r` 完全不被支持（**零字节 + 超时**）⇒ 转 `probe_plain_get`（时间上限 GET）。
    返回 (bps_1mib, bps_main, note)。
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    hr = os.path.join(LOG_DIR, "_hdr_probe.txt")

    r1 = run_curl(url, timeout=max_time, byte_range=(0, 1048575), hdr_path=hr)
    if r1["nbytes"] == 0 and allow_plain_fallback:
        log("PROBE  -r 档零字节（code=%s exit=%s）=> 端点不支持/挂死 Range，转时间上限 GET"
            % (r1["code"], r1["exit"]))
        code1 = r1["code"] or ("rc=%d" % r1["exit"])
        record_throughput(source + " " + label + " [probe 1 MiB · Range 不支持，登记为失败]",
                          code1, 0, max(r1["seconds"], 1e-6), "probe_1MiB_RANGE_UNSUPPORTED")
        bps, note, too_short = probe_plain_get(source, url, label, max_time)
        return 0.0, bps, note, too_short

    code1 = r1["code"] or ("rc=%d" % r1["exit"])
    truncated_by_time = (max_time - r1["seconds"]) < 1.5
    whole_object = (r1["nbytes"] < 1048576) and not truncated_by_time
    tag1 = "probe_whole_object" if whole_object else "probe_1MiB"
    bps1 = record_throughput(source + " " + label + " [probe %s]" % ("whole-object <1 MiB" if whole_object else "1 MiB"),
                             code1, r1["nbytes"], max(r1["seconds"], 1e-6), tag1)
    if r1["nbytes"] == 0:
        return bps1, 0.0, ("1 MiB 探针零字节（code=%s exit=%d）：%s"
                           % (code1, r1["exit"], r1["err"][:120])), False

    bps_main, mode, note = bps1, tag1, ""
    if not whole_object and r1["seconds"] < PROBE_ESCALATE_FAST_S:
        r2 = run_curl(url, timeout=max_time, byte_range=(0, PROBE_ESCALATE_RANGE), hdr_path=hr)
        bps2 = record_throughput(source + " " + label + " [probe 16 MiB cap]",
                                 r2["code"] or ("rc=%d" % r2["exit"]), r2["nbytes"],
                                 max(r2["seconds"], 1e-6), "probe_30s_16MiB")
        if r2["nbytes"] > 0:
            bps_main, mode = bps2, "probe_30s_16MiB"
    if whole_object and ref_url:
        rr = run_curl(ref_url, timeout=max_time, byte_range=(0, 1048575), hdr_path=hr)
        bps_ref = record_throughput(source + " " + ref_label + " [probe 1 MiB · 同通道大对象]",
                                    rr["code"] or ("rc=%d" % rr["exit"]), rr["nbytes"],
                                    max(rr["seconds"], 1e-6), "probe_ref_1MiB")
        if rr["nbytes"] > 0:
            bps_main, mode = bps_ref, "probe_ref_1MiB"
        note = ("目标对象 %d B <1 MiB ⇒ `probe_whole_object` %.1f B/s（**单件端到端，含握手，不得当链路吞吐**）；"
                "同通道大对象 `probe_ref_1MiB` %.1f B/s ⇒ **主档取后者**" % (r1["nbytes"], bps1, bps_main))
    if not note:
        note = ("1 MiB 探针 %.1f B/s（%.3f s）→ 主档 %s %.1f B/s"
                % (bps1, r1["seconds"], mode, bps_main))
    return bps1, bps_main, note, False


# ---------------------------------------------------------------------------
# T5 完整性门
# ---------------------------------------------------------------------------
def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def etag_multipart_check(path, etag, size):
    """S3 多段对象 ETag 复算：`md5(concat(md5(part_i))) + "-N"`。

    先由 size 与 N 试出 part_size（候选集），再逐段复算。返回 (ok, part_size, computed, note)。
    """
    if not etag:
        return None, None, "", "无 ETag（非 S3 对象或无响应头）"
    m = re.match(r'^"?([0-9a-fA-F]{32})-(\d+)"?$', etag.strip())
    if not m:
        # 单段：ETag 应为整个对象的 md5
        plain = etag.strip().strip('"')
        if re.match(r"^[0-9a-fA-F]{32}$", plain):
            h = hashlib.md5()
            with open(path, "rb") as f:
                for blk in iter(lambda: f.read(1 << 20), b""):
                    h.update(blk)
            return (h.hexdigest() == plain.lower()), None, h.hexdigest(), "单段 ETag = 全对象 md5"
        return None, None, "", "ETag 形态未知：%s" % etag
    n_parts = int(m.group(2))
    tried = []
    for ps in CANDIDATE_PART_SIZES:
        if size <= ps * (n_parts - 1) or size > ps * n_parts:
            continue
        tried.append(ps)
        digests = []
        with open(path, "rb") as f:
            for _ in range(n_parts - 1):
                h = hashlib.md5()
                got = 0
                while got < ps:
                    blk = f.read(min(1 << 20, ps - got))
                    if not blk:
                        break
                    h.update(blk)
                    got += len(blk)
                if got != ps:
                    digests = None
                    break
                digests.append(h.digest())
            if digests is not None:
                h = hashlib.md5()
                for blk in iter(lambda: f.read(1 << 20), b""):
                    h.update(blk)
                digests.append(h.digest())
        if digests is None:
            continue
        cat = hashlib.md5(b"".join(digests)).hexdigest()
        if cat == m.group(1).lower():
            return True, ps, cat, "part_size=%d（%d MiB）、N=%d 复算通过" % (ps, ps // (1 << 20), n_parts)
    return False, (tried[-1] if tried else None), "", (
        "复算**失败**：ETag 前缀 %s / N=%d / size=%d；候选 part_size 试过 %s（**不得**按'大小对得上'放行）"
        % (m.group(1), n_parts, size, [t // (1 << 20) for t in tried]))


def canonical_json_sha256(path):
    """semantic 基准：解析 JSON 后按键排序重新序列化（去 whitespace 差异）。"""
    try:
        obj = json.load(open(path, encoding="utf-8"))
    except Exception:
        return ""
    if isinstance(obj, dict) and "msg" in obj:
        obj = {k: obj[k] for k in sorted(obj) if k != "msg"}
        obj["msg"] = json.load(open(path, encoding="utf-8"))["msg"]
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def integrity_gate(t, source, url, path, clen, etag, curl_exit, parse_fn=None,
                   rows_expected=None, http_code=""):
    """三级完整性门。parse_fn(path) -> (ok, rows_parsed, note)；返回 verdict 与行。"""
    size = os.path.getsize(path) if os.path.exists(path) else 0
    clen = int(clen) if (clen is not None and str(clen).isdigit()) else None
    reasons = []
    if clen:
        size_match = "True" if size == clen else "False"
        if size != clen:
            reasons.append("size %d != Content-Length %d" % (size, clen))
    else:
        size_match = "NA(no content-length)" if size > 0 else "False"
        reasons.append("服务端未给 Content-Length（chunked）⇒ 门 ① 不适用，改以语义锚 + 两次独立取数 sha256 一致性替代")
    if curl_exit != 0:
        reasons.append("curl_exit=%d（传输未正常结束）" % curl_exit)

    etag_ok, part_size, etag_computed, etag_note = etag_multipart_check(path, etag, size)
    reasons.append(etag_note)

    sha_bytes = sha256_file(path) if size else ""
    sha_payload = sha_bytes  # 本工具落盘即原始响应体 ⇒ payload 与 file_bytes 同基准（已登记）
    sha_sem = canonical_json_sha256(path) if path.endswith(".json") else ""

    parse_ok, rows_parsed, parse_note = (None, None, "")
    if parse_fn:
        parse_ok, rows_parsed, parse_note = parse_fn(path)
        reasons.append(parse_note)
        if not parse_ok:
            reasons.append("解析失败 ⇒ 判 INCOMPLETE（不得进统计）")
    if rows_expected is not None and rows_parsed is not None:
        if rows_parsed != rows_expected:
            reasons.append("行数自洽失败：parsed=%s expected=%s" % (rows_parsed, rows_expected))
        else:
            reasons.append("行数自洽：parsed=%s == expected=%s" % (rows_parsed, rows_expected))

    hard_ok = (size > 0) and (curl_exit == 0) and (size_match in ("True", "NA(no content-length)"))
    if etag_ok is False:
        hard_ok = False
    if parse_ok is False:
        hard_ok = False
    if rows_expected is not None and rows_parsed is not None and rows_parsed != rows_expected:
        hard_ok = False
    verdict = "COMPLETE" if hard_ok else "INCOMPLETE"

    row = [utc_now(), t, source, url, http_code, clen or "", size, size_match, etag or "", "",
           part_size or "", etag_computed, etag_ok, sha_bytes, sha_payload, sha_sem,
           "file_bytes(=payload)", parse_ok, rows_parsed, rows_expected, curl_exit,
           verdict, " | ".join(r for r in reasons if r)]
    _INTEGRITY_ROWS.append(",".join(csv_escape(x) for x in row))
    log("INTEGRITY  %s  verdict=%s  size=%d clen=%s etag_ok=%s sha256=%s"
        % (source, verdict, size, clen, etag_ok, sha_bytes[:16]))
    return dict(verdict=verdict, size=size, sha256=sha_bytes, etag_ok=etag_ok,
                part_size=part_size, reasons=[r for r in reasons if r], rows_parsed=rows_parsed)


def fetch_full(url, path, expect_bytes=None, source="", label="", max_time=600, attempts=2,
               initial_resume=False):
    """T5 取数（**带 `--max-time` + `curl -C -` 续传**）。返回 (rec, clen, bps, resume_used)。

    `initial_resume=True`：目标文件已存在且是不完整的部分件时，**首轮就用 `-C -`**，
    避免把已有进度覆盖掉（派发硬要求②；M11-R63「大件必须带 --max-time + curl -C - 续传」）。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    hr = os.path.join(LOG_DIR, "_hdr_fetch.txt")
    resume_used = False
    if initial_resume:
        resume_used = True
        log("FETCH  %s 已有部分件 %d B ⇒ **首轮即 `curl -C -` 续传**（不覆盖已有进度）"
            % (label, os.path.getsize(path) if os.path.exists(path) else 0))
        rec = run_curl(url, timeout=max_time, out_path=path, hdr_path=hr, resume=True)
    else:
        rec = run_curl(url, timeout=max_time, out_path=path, hdr_path=hr)
    http_ok = rec["code"] in ("200", "206")
    if not http_ok:
        log("FETCH  %s **HTTP %s**（exit=%d）⇒ 该对象**未取到**（判 INCOMPLETE/NOT_EXIST，不进统计；"
            "不尝试 --fail 下的错误页续传）" % (label, rec["code"], rec["exit"]))
    # **有界重试环**：每次 `--max-time` 到点（exit=28）后 `curl -C -` 续传，直到完成或次数用尽。
    # ★ 实测必要性（本轮）：本机出现 **15–36 min 的整机冻结**，socket 仍"活着"但**零字节增长**
    #   （M11-R39：进程存在 ≠ 在跑）；此时**唯一能自救的动作**就是 `--max-time` 主动中断 + `-C -` 续传
    #   —— 冻结的既存连接**不会自己恢复**（实测 13:55→14:32 冻结 36 min 后由续传恢复）。
    for k in range(1, max(1, attempts)):
        size_now = os.path.getsize(path) if os.path.exists(path) else 0
        if expect_bytes and size_now >= expect_bytes:
            break
        if not (rec["code"] in ("200", "206")):
            break
        resume_used = True
        log("FETCH  %s 第 %d 轮未完成（http=%s exit=%s, %d/%s B）⇒ `curl -C -` 续传"
            % (label, k, rec["code"], rec["exit"], size_now, expect_bytes))
        rec = run_curl(url, timeout=max_time, out_path=path, hdr_path=hr, resume=True)
        if rec["code"] not in ("200", "206"):
            log("FETCH  %s 续传遇 HTTP %s ⇒ 停止重试（不覆盖对象）" % (label, rec["code"]))
            break
    size = os.path.getsize(path) if os.path.exists(path) else 0
    bps = record_throughput(source + " " + label + " [fetch]", rec["code"] or "rc=%d" % rec["exit"],
                            size, max(rec["seconds"], 1e-6),
                            "fetch(resume)" if resume_used else "fetch")
    hdr = rec.get("headers", {})
    clen = hdr.get("content-length")
    clen = int(clen) if (clen and clen.isdigit()) else None
    return rec, clen, bps, resume_used


# ---------------------------------------------------------------------------
# S2 · CCFv3 结构本体
# ---------------------------------------------------------------------------
def s2_parse_structures(path):
    try:
        j = json.load(open(path, encoding="utf-8"))
    except Exception as e:
        return False, None, "JSON 解析异常：%s" % str(e)[:120]
    if not j.get("success"):
        return False, None, "success != true"
    msgs = j.get("msg") or []
    need = ("id", "acronym", "name", "parent_structure_id", "depth", "graph_id",
            "structure_id_path", "graph_order")
    miss = [k for k in need if k not in (msgs[0] if msgs else {})]
    if miss:
        return False, len(msgs), "字段缺失（%s）⇒ 结构口径不可声明" % miss
    return True, len(msgs), "msg 行数=%d；必需字段齐（id/acronym/name/parent_structure_id/depth/graph_id/structure_id_path）" % len(msgs)


def run_s2(args):
    log("=== S2 CCFv3 结构本体 ===")
    hr = os.path.join(LOG_DIR, "_hdr_s2.txt")
    os.makedirs(os.path.join(RAW, "s2"), exist_ok=True)

    # ---- T1 计数（成本极小，**先拿"数"再决定取不取**）----
    total_rows = None
    tmp = os.path.join(RAW, "s2", "_count.json")
    run_curl(S2_COUNT_URL, timeout=30, out_path=tmp, hdr_path=hr)
    try:
        total_rows = json.load(open(tmp, encoding="utf-8")).get("total_rows")
    except Exception as e:
        log("S2 计数解析失败：%s" % str(e)[:120])
    log("S2 T1 total_rows(model::Structure)=%s" % total_rows)

    # graph_id 分布（为"子图口径"提供依据，不得凭印象）
    graph_counts = {}
    for g in (1, 2, 17):
        gtmp = os.path.join(RAW, "s2", "_count_graph%d.json" % g)
        run_curl(S2_GRAPH_TMPL.format(g), timeout=30, out_path=gtmp, hdr_path=hr)
        try:
            graph_counts[g] = json.load(open(gtmp, encoding="utf-8")).get("total_rows")
        except Exception:
            graph_counts[g] = None
    log("S2 graph_id 计数：%s" % graph_counts)

    # ---- T1b 单页取数 ⇒ 每行字节 ⇒ **零成本体积估计**（无需下全表即可算 ETA）----
    page_rows = 50
    page = os.path.join(RAW, "s2", "_page50.json")
    pr = run_curl(S2_PAGE_URL, timeout=30, out_path=page, hdr_path=hr)
    bytes_per_row = None
    if os.path.exists(page) and os.path.getsize(page) > 0:
        bytes_per_row = os.path.getsize(page) / float(page_rows)
    est_total = int(total_rows * bytes_per_row) if (total_rows and bytes_per_row) else None
    log("S2 T1b 单页 %d 行 = %s B ⇒ 每行 %.1f B ⇒ 全表估计 = %s B（估计口径，用于 T4 预判）"
        % (page_rows, os.path.getsize(page) if os.path.exists(page) else 0,
           bytes_per_row or -1, est_total))

    # ---- T2 对象级取证 ----
    # 实测（本轮）：`api.brain-map.org` 的**动态 JSON 查询端点不支持 Range**（`-r 0-1023`
    # 30 s 零字节、curl exit=28）⇒ 对象级取证改用**同端点同 host 的计数 GET**（200 + 非空 JSON），
    # 并在 note 中显式登记"全表端点 chunked（无 Content-Length）+ 无 Accept-Ranges + 无 Range"。
    head = run_curl(S2_COUNT_URL, timeout=30, hdr_path=hr)
    hdr = head.get("headers", {})
    log("S2 T2 计数端点 GET code=%s ct=%s clen=%s accept-ranges=%s"
        % (head["code"], hdr.get("content-type"), hdr.get("content-length"), hdr.get("accept-ranges")))

    # ---- T3 **当场吞吐实测**（先测吞吐、再决定取数；Range 不支持 ⇒ 自动转时间上限 GET）----
    bps_1m, bps_main, probe_note, win_too_short = probe_throughput(
        "S2:api.brain-map.org model::Structure", S2_FULL_URL, label="(full table)")
    if win_too_short:
        log("S2 T3 == 探针窗过短（TTFB > 窗）=> 依预注册规则转'整件取数即测量'："
            "**不得**由 0 B/s 判 TOO_SLOW")

    # ---- T4 ETA 裁决（用 T1b 的**估计**体积预判；取回后用**实测**体积复算并登记两值）----
    eta_est = (est_total / bps_main) if (est_total and bps_main) else None
    if win_too_short:
        # 探针窗过短：0 B/s 不可用 => 以"整件取数"的完成时间作为测量（预注册规则 3）
        eta_ok = True
        log("S2 T4 预判：探测窗过短 => 走整件取数即测量（对象 %s B，估计体积 %s B）；"
            "实测 ETA 由完成时间给出" % (est_total, est_total))
    else:
        eta_ok = (eta_est is not None and eta_est <= 1800.0)
        log("S2 T4 预判 ETA=%.1f s（估计体积 %s B @ %.1f B/s；门=1800 s/对象）=> %s"
            % (eta_est or -1, est_total, bps_main, "PASS" if eta_ok else "TOO_SLOW/需托管"))

    verdict, scope, missing = None, "", ""
    integ = None
    etag_val = ""
    eta_act = None
    if args.probe_only:
        verdict = "NEEDS_VERIFY"
        scope = "仅吞吐探针（--probe-only，未取数）"
    elif eta_est is not None and not eta_ok:
        verdict = "TOO_SLOW"
        scope = "T4 判 ETA 超 30 min 门 ⇒ unavailable（吞吐限制）"
        missing = "吞吐维度：须以**当场**探针复核（≤24 h）后重判；禁止按锚外推"
    else:
        p = os.path.join(RAW, "s2", "model_Structure_all.json")
        rec, clen, bps_f, resumed = fetch_full(S2_FULL_URL, p, expect_bytes=None,
                                               source="S2:api.brain-map.org model::Structure",
                                               label="(full table)", max_time=240)
        actual_bytes = os.path.getsize(p) if os.path.exists(p) else 0
        eta_act = (actual_bytes / bps_f) if bps_f else None
        log("S2 T4 实测复算：实际体积 %d B（估计 %s B，偏差 %s）@ %.1f B/s ⇒ 实测 ETA=%.1f s"
            % (actual_bytes, est_total,
               ("%+.1f%%" % (100.0 * (actual_bytes - est_total) / est_total)) if est_total else "NA",
               bps_f, eta_act or -1))
        integ = integrity_gate(2, "S2:model::Structure 全表", S2_FULL_URL, p, clen,
                               rec.get("headers", {}).get("etag"), rec["exit"],
                               parse_fn=s2_parse_structures, rows_expected=total_rows,
                               http_code=rec["code"])
        etag_val = rec.get("headers", {}).get("etag", "")
        # 门 ① 替代锚（chunked 无 Content-Length）：
        #   ①-a **服务端 ETag 单段 md5 复算通过** ⇒ 他证（锚在服务器侧），强于自洽；
        #   ①-b 仅当 ETag 缺失/不复算时，才额外做"两次独立取数 sha256 一致"（自洽）。
        if integ["etag_ok"] is True:
            repeat_match = "NOT_NEEDED(服务端 ETag=全对象 md5 复算通过 => 逐字节等同服务器对象)"
            log("S2 门 ① 替代锚：服务端 ETag 复算通过 => 跳过冗余的第二次全量取数（省 35 s / 7.2 MB）")
        else:
            p2 = os.path.join(RAW, "s2", "_repeat2.json")
            rec2, clen2, _, _ = fetch_full(S2_FULL_URL, p2,
                                           source="S2:api.brain-map.org model::Structure",
                                           label="(full table · 第二次独立取数)", max_time=240)
            sha2 = sha256_file(p2) if os.path.exists(p2) else ""
            repeat_match = (sha2 == integ["sha256"])
            log("S2 两次独立取数 sha256 一致 = %s" % repeat_match)
            if os.path.exists(p2):
                os.remove(p2)
        integ["repeat_sha256_match"] = repeat_match
        if integ["verdict"] == "COMPLETE" and repeat_match is not False:
            verdict = "REACHABLE"
            scope = ("对象级 200 + 完整性门（①无 CL ⇒ 以服务端 ETag md5 复算 + 语义行数自洽替代；"
                     "②服务端 ETag 单段 md5 复算通过；③sha256 file_bytes/payload/semantic）+ 语义锚"
                     "（msg 行数 == T1 total_rows）")
        else:
            verdict = "REACHABLE_NO_DATA"
            scope = "完整性门未过"
            missing = ("; ".join(integ["reasons"])
                       + ("；两次取数 sha256 不一致" if repeat_match is False else ""))[:320]
    note = ("T1 total_rows=%s；graph_id 计数=%s；T1b 每行 %.1f B ⇒ 估计全表 %s B；"
            "T4 预判 ETA=%.1f s（探针窗过短=%s）；实测 %s B ⇒ 实测 ETA=%.1f s；"
            "端点无 Range/无 CL/无 Accept-Ranges；%s"
            % (total_rows, graph_counts, bytes_per_row or -1, est_total, eta_est or -1,
               win_too_short, (integ["size"] if integ else "NA"), eta_act or -1, probe_note))
    if len(note) > 1200:
        note = note[:1200]
    # ---- ★ 更正行（append-only）：本端点**不适用 Range**，故 1 MiB 档只能给出 0 B/s
    #      （探针窗短于 TTFB，**不是**吞吐慢）⇒ 另立一行承载"整件取数即测量"的**可用**吞吐与 ETA。
    if integ is not None and eta_act:
        bps_measured = integ["size"] / eta_act
        _REACH_ROWS.append(",".join(csv_escape(x) for x in [
            utc_now(), "S2(更正行·整件取数即测量)", "CCFv3 结构本体 —— **可用吞吐与 ETA 的更正行**",
            "allen_ontology", "api_json", S2_FULL_URL, "200", "200", integ["size"],
            "application/json; charset=utf-8", etag_val, "", "%.3f" % eta_act,
            "%.1f" % bps_measured, "%.1f" % bps_measured, "%.1f" % bps_measured,
            "%.1f" % eta_act, "REACHABLE",
            "T4 以**整件取数完成时间**为测量（预注册规则：探针窗 < TTFB ⇒ 转整件取数即测量）", "",
            "**为什么需要本行**：本端点不支持 Range 且 TTFB≈31 s ⇒ 1 MiB / 30 s 探针档只能给出 0 B/s；"
            "若把该 0 B/s 当吞吐就会误判 unavailable。本行给出**当场实测可用值**："
            "%d B / %.1f s = %.1f B/s（端到端含 TTFB）⇒ 门=1800 s/对象，通过 %.0f×。"
            "**判决时刻=%s**；须以当场探针复核（≤24 h，M11-R56）"
            % (integ["size"], eta_act, bps_measured, 1800.0 / eta_act, utc_now())]))
    _REACH_ROWS.append(",".join(csv_escape(x) for x in [
        utc_now(), "S2", "CCFv3 结构本体（api.brain-map.org model::Structure）", "allen_ontology", "api_json",
        S2_FULL_URL, head["code"], head["code"], (integ["size"] if integ else ""),
        hdr.get("content-type", ""), etag_val,
        hdr.get("accept-ranges", ""),
        "%.3f" % head["seconds"], "%.1f" % bps_main, "%.1f" % bps_1m, "%.1f" % bps_main,
        ("%.1f" % eta_est if eta_est else ""), verdict, scope, missing, note]))
    return dict(source="S2", verdict=verdict, probe_1mib_Bps=bps_1m, probe_main_Bps=bps_main,
                probe_note=probe_note, t1_total_rows=total_rows, graph_counts=graph_counts,
                bytes_per_row=bytes_per_row, est_total_bytes=est_total, eta_est_s=eta_est,
                eta_actual_s=eta_act, full_url=S2_FULL_URL,
                integrity=(integ or {}), missing_dimension=missing)


# ---------------------------------------------------------------------------
# S7 · Allen ecephys 汇总表
# ---------------------------------------------------------------------------
def _csv_parse(path):
    try:
        import csv as _csv
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            rows = list(_csv.reader(f))
        if not rows:
            return False, 0, "空表"
        return True, len(rows) - 1, "CSV 列数=%d，数据行=%d" % (len(rows[0]), len(rows) - 1)
    except Exception as e:
        return False, None, "CSV 解析异常：%s" % str(e)[:120]


def run_s7(args):
    log("=== S7 Allen Brain Observatory ecephys 汇总表 ===")
    hr = os.path.join(LOG_DIR, "_hdr_s7.txt")
    out = []
    for url in S7_URLS:
        name = url.rsplit("/", 1)[-1]
        p = os.path.join(RAW, "s7", name)
        head = run_curl(url, timeout=30, byte_range=(0, 1023), hdr_path=hr)
        hdr = head.get("headers", {})
        bps_1m, bps_main, probe_note, _ = probe_throughput(
            "S7:allen-bo ecephys-cache %s" % name, url, label="(%s)" % name,
            ref_url=ALLEN_BO + "/visual-coding-neuropixels/ecephys-cache/channels.csv",
            ref_label="渠道参照 channels.csv(6.6 MB)")
        rec, clen, bps_f, resumed = fetch_full(url, p, source="S7:allen-bo ecephys-cache %s" % name,
                                               label="(%s)" % name, max_time=120)
        clen = clen or (S7_M11_SHA.get(name + ".m11_bytes"))
        integ = integrity_gate("S7.%s" % name, "S7:ecephys-cache/%s" % name, url, p,
                               rec.get("headers", {}).get("content-length"),
                               rec.get("headers", {}).get("etag"), rec["exit"], parse_fn=_csv_parse)
        # 与 M11 已落盘件**独立交叉核对**（同路径同日复测）
        m11_sha = S7_M11_SHA.get(name)
        same_as_m11 = (integ["sha256"] == m11_sha)
        verdict = "REACHABLE" if integ["verdict"] == "COMPLETE" else "REACHABLE_NO_DATA"
        note = ("%s；M11 已落盘件 sha256=%s ⇒ 同日复测逐字节一致=%s；行数=%s"
                % (probe_note, (m11_sha or "")[:16], same_as_m11, integ["rows_parsed"]))
        _REACH_ROWS.append(",".join(csv_escape(x) for x in [
            utc_now(), "S7.%s" % name, "Allen BO ecephys 汇总表 %s" % name, "allen_ecephys", "s3_object", url,
            head["code"], head["code"], integ["size"], hdr.get("content-type", ""),
            rec.get("headers", {}).get("etag", ""), hdr.get("accept-ranges", ""),
            "%.3f" % head["seconds"], "%.1f" % bps_main, "%.1f" % bps_1m, "%.1f" % bps_main,
            "%.1f" % (integ["size"] / bps_main if bps_main else 0), verdict,
            "对象级 200 + 完整性门三级（含 ETag 单段 md5 复算）", "",
            note + "；本条 T4 ETA=%.1f s ⇒ **远低于 30 min 门**" % (integ["size"] / bps_main if bps_main else 0)]))
        out.append(dict(name=name, url=url, verdict=verdict, bytes=integ["size"],
                        sha256=integ["sha256"], m11_sha256_match=same_as_m11,
                        rows=integ["rows_parsed"], etag_ok=integ["etag_ok"],
                        probe_main_Bps=bps_main, probe_1mib_Bps=bps_1m))
    return dict(source="S7", items=out)


# ---------------------------------------------------------------------------
# S5 · IBL 行为白名单
# ---------------------------------------------------------------------------
def _list_prefix(prefix, maxkeys=1000, delimiter="/"):
    url = "%s/?list-type=2&max-keys=%d&prefix=%s" % (IBL_BUCKET, maxkeys, prefix)
    if delimiter:
        url += "&delimiter=" + delimiter
    hr = os.path.join(LOG_DIR, "_hdr_list.txt")
    body_path = os.path.join(LOG_DIR, "_list.xml")
    rec = run_curl(url, timeout=30, out_path=body_path, hdr_path=hr)
    txt = open(body_path, encoding="utf-8", errors="replace").read() if os.path.exists(body_path) else ""
    keys = re.findall(r"<Key>([^<]+)</Key>", txt)
    sizes = [int(x) for x in re.findall(r"<Size>(\d+)</Size>", txt)]
    prefixes = re.findall(r"<Prefix>([^<]+)</Prefix>", txt)[1:]
    kc = re.search(r"<KeyCount>(\d+)</KeyCount>", txt)
    trunc = re.search(r"<IsTruncated>(\w+)</IsTruncated>", txt)
    return dict(code=rec["code"], keys=keys, sizes=sizes, prefixes=prefixes,
                keycount=int(kc.group(1)) if kc else None,
                truncated=(trunc.group(1) == "true") if trunc else None, raw_len=len(txt))


def build_ibl_whitelist(max_sessions=40):
    """**预注册**白名单：轮转（round-robin）跨 lab 取 session，确定性排序。

    枚举规则（**在看任何统计量之前定死**）：
      1. `data/` 一级前缀（lab）按字母序；
      2. 每 lab 内 subject 按字母序、date 按字母序、session 号按数值序；
      3. 以"轮"为单位：第 r 轮各 lab 各取第 r 个 (subject,date,session)；
      4. 取到 `max_sessions` 停；缺 `_ibl_trials.table.*.pqt` 的 session **显式登记原因并跳过**（不得静默跳过）；
      5. 归属：第 1 轮 ⇒ A（拟合集）、第 2 轮 ⇒ B（held-out）、第 3 轮 ⇒ C、第 4 轮 ⇒ D。
    """
    log("=== S5 IBL 行为：白名单枚举（round-robin，预注册规则）===")
    top = _list_prefix("data/", delimiter="/")
    labs = sorted(p[len("data/"):].rstrip("/") for p in top["prefixes"] if not p.endswith("resources/"))
    log("S5 labs(%d)=%s" % (len(labs), labs))

    per_lab_subjects = {}
    for lab in labs:
        r = _list_prefix("data/%s/Subjects/" % lab, delimiter="/")
        subs = sorted(p.rsplit("/", 2)[-2] for p in r["prefixes"])
        per_lab_subjects[lab] = subs
    log("S5 subjects/lab = %s" % {k: len(v) for k, v in per_lab_subjects.items()})

    out_rows = []
    skipped = []
    cursor = {lab: 0 for lab in labs}
    round_no = 0
    partition_of_round = {1: "A", 2: "B", 3: "C", 4: "D"}
    while len(out_rows) < max_sessions and round_no < 12:
        round_no += 1
        progressed = False
        for lab in labs:
            if len(out_rows) >= max_sessions:
                break
            subs = per_lab_subjects.get(lab) or []
            if cursor[lab] >= len(subs):
                continue
            subj = subs[cursor[lab]]
            cursor[lab] += 1
            progressed = True
            rd = _list_prefix("data/%s/Subjects/%s/" % (lab, subj), delimiter="/")
            dates = sorted(p.rsplit("/", 2)[-2] for p in rd["prefixes"])
            if not dates:
                skipped.append("%s/%s: 无 date 前缀" % (lab, subj))
                continue
            date = dates[0]
            rs = _list_prefix("data/%s/Subjects/%s/%s/" % (lab, subj, date), delimiter="/")
            nums = sorted((p.rsplit("/", 2)[-2] for p in rs["prefixes"]),
                          key=lambda x: (len(x), x))
            if not nums:
                skipped.append("%s/%s/%s: 无 session 前缀" % (lab, subj, date))
                continue
            num = nums[0]
            alf_prefix = "data/%s/Subjects/%s/%s/%s/alf/" % (lab, subj, date, num)
            ra = _list_prefix(alf_prefix, delimiter="")
            cands = [(k, s) for k, s in zip(ra["keys"], ra["sizes"])
                     if re.search(r"_ibl_trials\.table[^/]*\.pqt$", k)]
            if not cands:
                skipped.append("%s/%s/%s/%s: alf 内无 `_ibl_trials.table.*.pqt`（%d 个对象）"
                               % (lab, subj, date, num, len(ra["keys"])))
                continue
            # **口径预注册**：优先"当前版"（`alf/_ibl_trials.table.*.pqt`，无 revision 子目录），
            # 否则取 revision 目录里**键序最后**（= 最新 revision，目录名形如 `#YYYY-MM-DD#`）。
            # 两者**不得混用**；选中哪一档必须登记（`revision_used`）。
            cur = [(k, s) for k, s in cands if "/" not in k[len(alf_prefix):]]
            if cur:
                key, size = cur[0]
                rev_used = "current(alf/_ibl_trials.table.*.pqt)"
            else:
                key, size = sorted(cands, key=lambda x: x[0])[-1]
                rev_used = "revision:%s" % key[len(alf_prefix):].split("/", 1)[0]
            rank = len(out_rows) + 1
            eid = "%s/%s/%s/%s" % (lab, subj, date, num)
            out_rows.append([rank, lab, subj, date, num, eid, alf_prefix, key, size,
                             partition_of_round.get(round_no, "R%d" % round_no), "True",
                             "round-robin(字母序 lab/subject/date + 数值序 session)；"
                             "归属按最终序号三分（A=1..13 / B=14..26 / C=27..39 / D=40）",
                             "%s | %s" % ("本行在**任何统计量计算之前**预注册", rev_used)])
            log("S5 [%02d] %s  partition=%s  bytes=%d" % (rank, eid, partition_of_round.get(round_no, "?"), size))
        if not progressed:
            break
    # 归属重算：因 round-robin ⇒ 按**最终序号三分**，使 A/B/C 各自横跨同一组 lab
    # （**预注册**：A=1..13 / B=14..26 / C=27..39 / D=40；在算任何统计量之前定死）
    for i, r in enumerate(out_rows):
        rk = i + 1
        r[9] = "A" if rk <= 13 else ("B" if rk <= 26 else ("C" if rk <= 39 else "D"))
        r[10] = "True"
    if skipped:
        log("S5 **显式跳过 %d 个 session**（不得静默跳过）：%s" % (len(skipped), skipped[:10]))
    ensure_header(CSV_IBL, IBL_HEADER,
                  "M12 S5 IBL 行为预注册 session 白名单（generated_by=tools/probe_m12_sources.py；"
                  "枚举规则+归属在执行前定死，见 note；**只增不改**）")
    with open(CSV_IBL, "w", encoding="utf-8") as f:
        f.write("# M12 S5 IBL 行为预注册 session 白名单（generated_by=tools/probe_m12_sources.py；"
                "枚举规则+归属在执行前定死，见 note；**只增不改**）\n")
        f.write(IBL_HEADER + "\n")
        for r in out_rows:
            f.write(",".join(csv_escape(x) for x in r) + "\n")
    return out_rows, skipped, labs


def verify_trials_aggregate(path):
    """T5 内容级：pyarrow 解析 `trials.pqt`，验证它**真是"逐 trial 合并表"而非别的东西**。

    检查项：
      ① 行数 / 列数 / 列名（与 P8/P9 所需列对照）；
      ② **会话列存在**且 **逐 session 计数之和 == 表行数**（内部自洽锚）；
      ③ 报每 session 行数分布（n_sessions / min / p10 / median / p90 / max / 总和）。
    返回 (ok, n_rows, note)。**副作用**：写 `data/m12_ibl_trials_aggregate.csv` 与 `.json`。
    """
    try:
        import pyarrow.parquet as pq
    except Exception as e:
        return False, None, "pyarrow 不可用（应用 ./.venv-m10/bin/python 运行）：%s" % str(e)[:120]
    try:
        pf = pq.ParquetFile(path)
        n_rows = pf.metadata.num_rows
        cols = list(pf.schema_arrow.names)
    except Exception as e:
        return False, None, "parquet 头解析失败：%s" % str(e)[:160]

    sess_col = None
    for c in cols:
        if c.lower() in ("session", "eid", "session_id", "session_uuid", "sessions"):
            sess_col = c
            break
    want = [c for c in ("choice", "feedback", "response_times", "contrastLeft", "contrastRight",
                        "probabilityLeft", "rewardVolume", "firstMovement_times", "stimOn_times",
                        "goCue_times", "feedback_times", "intervals") if c in cols]
    if sess_col is None:
        return False, n_rows, ("**无会话列**（列名=%s）⇒ 无法按 session 归属 ⇒ 不得当逐 trial 合并表使用"
                               % cols[:12])

    from collections import Counter
    col = pq.read_table(path, columns=[sess_col]).column(sess_col).to_pylist()
    cnt = Counter(col)
    total = sum(cnt.values())
    vals = sorted(cnt.values())
    dist = dict(
        n_sessions=len(cnt), total_trials=total, sum_equals_num_rows=(total == n_rows),
        min=vals[0] if vals else 0,
        p10=vals[int(0.10 * (len(vals) - 1))] if vals else 0,
        median=(vals[len(vals) // 2] if len(vals) % 2 else
                0.5 * (vals[len(vals) // 2 - 1] + vals[len(vals) // 2])) if vals else 0,
        p90=vals[int(0.90 * (len(vals) - 1))] if vals else 0,
        max=vals[-1] if vals else 0,
    )
    ensure_header(CSV_AGG_SESSIONS, AGG_SESS_HEADER,
                  "M12 S5 全量聚合行为表 trials.pqt 的**逐 session 试次数**"
                  "（generated_by=tools/probe_m12_sources.py --only S5FULL；"
                  "**行数之和 == 表行数** 为内部自洽锚；只增不改）")
    with open(CSV_AGG_SESSIONS, "w", encoding="utf-8") as f:
        f.write("# M12 S5 全量聚合行为表 trials.pqt 的**逐 session 试次数**"
                "（generated_by=tools/probe_m12_sources.py --only S5FULL；sum == num_rows 为自洽锚）\n")
        f.write(AGG_SESS_HEADER + "\n")
        for i, (k, v) in enumerate(sorted(cnt.items(), key=lambda x: (-x[1], str(x[0]))), 1):
            f.write("%d,%s,%d,%.8f\n" % (i, k, v, (v / total if total else 0.0)))
    agg = dict(generated_by="tools/probe_m12_sources.py --only S5FULL", generated_at=utc_now(),
               path=os.path.relpath(path, os.path.dirname(ROOT)), n_rows=n_rows,
               n_cols=len(cols), columns=cols, session_column=sess_col,
               behavior_columns_hit=want, distribution=dist)
    with open(os.path.join(DATA, "m12_ibl_trials_aggregate.json"), "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=1)

    ok = (n_rows > 0) and dist["sum_equals_num_rows"] and len(cnt) > 1
    note = ("parquet 行数=%d 列数=%d 会话列=%s；**逐 session 计数之和=%d == 行数 ⇒ %s**；"
            "n_sessions=%d（min %d / p10 %d / median %s / p90 %d / max %d）；"
            "P8/P9 行为列命中=%s"
            % (n_rows, len(cols), sess_col, total, dist["sum_equals_num_rows"], dist["n_sessions"],
               dist["min"], dist["p10"], dist["median"], dist["p90"], dist["max"], want))
    return ok, n_rows, note


def run_s5full(args):
    """R72 批准的 S5 全量聚合行为表取数（24,212,510 B）——**先测吞吐、再取数、再三级完整性门**。"""
    log("=== S5FULL · IBL 全量聚合行为表 trials.pqt（R72 批准）===")
    os.makedirs(S5FULL_DIR, exist_ok=True)
    hr = os.path.join(LOG_DIR, "_hdr_s5full.txt")

    # ---- T2 对象级取证（HEAD 拿 content-length / ETag / accept-ranges）----
    head = run_curl(IBL_AGG_TRIALS, timeout=30, byte_range=(0, 1023), hdr_path=hr)
    hh = head.get("headers", {})
    clen_hdr = int(hh["content-length"]) if (hh.get("content-length") or "").isdigit() else None
    log("S5FULL T2 Range code=%s content-range=%s clen=%s etag=%s accept-ranges=%s"
        % (head["code"], hh.get("content-range"), clen_hdr, hh.get("etag"), hh.get("accept-ranges")))
    obj_bytes = None
    mcr = re.search(r"/(\d+)\s*$", hh.get("content-range", ""))
    if mcr:
        obj_bytes = int(mcr.group(1))
    # 无 Range 时（本桶支持 Range，应不会走到）用 S3 列举的声明字节
    if obj_bytes is None:
        agg = _list_prefix("aggregates/2026_Q2_IBL_et_al_BWM/", delimiter="")
        for k, sz in zip(agg["keys"], agg["sizes"]):
            if k.endswith("/trials.pqt"):
                obj_bytes = sz
    log("S5FULL 对象总长 = %s B（S3 声明）" % obj_bytes)

    # ---- T3 **当场 30 s 吞吐探针**（以最近一次为准，R56/R63/R72）----
    bps_1m, bps_main, probe_note, win_short = probe_throughput(
        "S5FULL:ibl-public aggregates/2026_Q2_IBL_et_al_BWM/trials.pqt", IBL_AGG_TRIALS,
        label="(24.2 MB 全量聚合行为表)")
    if win_short:
        log("S5FULL ⚠ 探针窗过短（TTFB ≥ 窗）⇒ 按 M12-M-01 转整件取数即测量，**不得判 TOO_SLOW**")

    # ---- T4 ETA 裁决 ----
    eta = (obj_bytes / bps_main) if (obj_bytes and bps_main) else None
    eta_ok = win_short or (eta is not None and eta <= 1800.0)
    log("S5FULL T4 当场 ETA = %s s @ %.1f B/s（门 1800 s/对象）⇒ %s"
        % (("%.1f" % eta) if eta else "NA", bps_main, "PASS" if eta_ok else "TOO_SLOW"))

    path = os.path.join(S5FULL_DIR, "trials.pqt")
    integ = None
    verdict, missing = None, ""
    if not eta_ok:
        verdict = "TOO_SLOW"
        missing = ("吞吐维度：当场实测 ETA=%.1f s > 1800 s 门；**须以当场探针复核（≤24 h）**；"
                   "判决时刻与当时吞吐见 throughput CSV" % eta)
    else:
        # ---- T5 取数（`--max-time` + `curl -C -` 续传）----
        pre = os.path.getsize(path) if os.path.exists(path) else 0
        partial = bool(obj_bytes and 0 < pre < obj_bytes)
        if pre == obj_bytes:
            log("S5FULL 已有**完整**件 %d B ⇒ 跳过传输，直接过完整性门" % pre)
        if partial:
            log("S5FULL 检测到部分件 %d / %s B（%.2f%%）⇒ 续传"
                % (pre, obj_bytes, 100.0 * pre / obj_bytes))
        rec, clen, bps_f, resumed = fetch_full(
            IBL_AGG_TRIALS, path, expect_bytes=obj_bytes,
            source="S5FULL:ibl-public aggregates trials.pqt", label="(24.2 MB)",
            max_time=args.max_time, initial_resume=partial, attempts=args.attempts)
        size = os.path.getsize(path) if os.path.exists(path) else 0
        log("S5FULL 取数完成：%d B（期望 %s）resume_used=%s http=%s exit=%s"
            % (size, obj_bytes, resumed, rec["code"], rec["exit"]))
        # ★ **门① 的基准 = 对象总长**（S3 声明 / HEAD 的 Content-Length）。
        # 206 续传响应的 `Content-Length` 是**剩余段长度**，不是对象长度 ⇒ 若拿它当基准，
        # 会得到 **假的 size_match=False**（本轮实测踩到：clen=11439993 而对象应为 24212510）。
        gate_clen = obj_bytes or clen_hdr or clen
        log("门① 基准 = 对象总长 %s B（**不用** 206 续传响应的 Content-Length=%s，那是剩余段长度）"
            % (gate_clen, clen))
        integ = integrity_gate("S5FULL", "S5FULL:aggregates/2026_Q2_IBL_et_al_BWM/trials.pqt",
                               IBL_AGG_TRIALS, path, gate_clen,
                               rec.get("headers", {}).get("etag", hh.get("etag")), rec["exit"],
                               parse_fn=verify_trials_aggregate, rows_expected=None,
                               http_code=rec["code"])
        if integ["verdict"] == "COMPLETE":
            verdict = "REACHABLE"
        else:
            verdict = "REACHABLE_NO_DATA"
            missing = "; ".join(integ["reasons"])[:320]

    aggjson = load_json_safe(os.path.join(DATA, "m12_ibl_trials_aggregate.json"))
    dist = (aggjson or {}).get("distribution", {})
    note = ("%s；对象总长 %s B；当场 ETA=%s s；%s；逐 session：n=%s, min=%s, median=%s, max=%s, "
            "sum==num_rows=%s"
            % (probe_note, obj_bytes, ("%.1f" % eta) if eta else "NA",
               (integ["reasons"][0] if integ and integ.get("reasons") else ""),
               dist.get("n_sessions"), dist.get("min"), dist.get("median"), dist.get("max"),
               dist.get("sum_equals_num_rows")))
    _REACH_ROWS.append(",".join(csv_escape(x) for x in [
        utc_now(), "S5FULL", "IBL 全量聚合行为表 trials.pqt（R72 批准，解 22-session 未命中缺口）",
        "ibl_behavior", "s3_object", IBL_AGG_TRIALS, head["code"], rec["code"] if integ else "",
        (integ["size"] if integ else ""), "application/octet-stream",
        (hh.get("etag") or ""), (hh.get("accept-ranges") or ""),
        ("%.3f" % head["seconds"]), "%.1f" % bps_main, "%.1f" % bps_1m, "%.1f" % bps_main,
        ("%.1f" % eta if eta else ""), verdict,
        "对象级 200/206 + 完整性门三级（size==Content-Length ∧ **ETag 多段复算** ∧ sha256 三基准）+ "
        "pyarrow 语义锚（逐 session 计数之和 == 表行数）", missing, note[:1200]]))
    return dict(source="S5FULL", verdict=verdict, object_bytes=obj_bytes, eta_s=eta,
                probe_1mib_Bps=bps_1m, probe_main_Bps=bps_main, probe_note=probe_note,
                window_too_short=win_short, integrity=(integ or {}),
                aggregate_json=(aggjson or {}), missing_dimension=missing)


def run_s5(args):
    log("=== S5 IBL 行为（白名单 ≤%d session）===" % args.max_sessions)
    hr = os.path.join(LOG_DIR, "_hdr_s5.txt")

    # T1：桶/聚合对象级列举（先计数，再决定取不取）
    agg = _list_prefix("aggregates/2026_Q2_IBL_et_al_BWM/", delimiter="")
    log("S5 T1 aggregates/2026_Q2_IBL_et_al_BWM KeyCount=%s 对象=%s"
        % (agg["keycount"], list(zip([k.rsplit('/',1)[-1] for k in agg["keys"]], agg["sizes"]))))
    tree_head = run_curl(IBL_PUBLIC_TREE, timeout=30, byte_range=(0, 1023), hdr_path=hr)
    tree_hdr = tree_head.get("headers", {})

    # 参考路径的**当场吞吐实测**（决策探针：全量聚合表 trials.pqt 24.2 MB 若取需要多少时间）
    trials_bytes = None
    for k, s in zip(agg["keys"], agg["sizes"]):
        if k.endswith("/trials.pqt"):
            trials_bytes = s
    bps_t_1m, bps_t_main, probe_trials_note, _ = probe_throughput(
        "S5:ibl-public aggregates/.../trials.pqt(全量行为表, 探针路径)", IBL_AGG_TRIALS,
        label="(24.2 MB 全量聚合表)")
    eta_trials = (trials_bytes / bps_t_main) if (trials_bytes and bps_t_main) else None
    log("S5 全量聚合 trials.pqt bytes=%s ⇒ ETA=%.1f s @ %.1f B/s"
        % (trials_bytes, eta_trials or -1, bps_t_main))

    # 白名单 + 逐 session 取行为试次表
    rows, skipped, labs = build_ibl_whitelist(args.max_sessions)
    if not rows:
        _REACH_ROWS.append(",".join(csv_escape(x) for x in [
            utc_now(), "S5", "IBL 行为（桶 data/<lab>/Subjects/.../alf/_ibl_trials.table.*.pqt）", "ibl_behavior",
            "s3_object", IBL_BUCKET + "/data/", agg["code"], agg["code"], "", "application/xml", "",
            "", "", "%.1f" % bps_t_main, "%.1f" % bps_t_1m, "%.1f" % bps_t_main, "",
            "REACHABLE_NO_DATA", "列举可达但白名单内无对象级证实", "",
            "白名单枚举 0 命中；跳过登记=%s" % skipped[:3]]))
        return dict(source="S5", verdict="REACHABLE_NO_DATA", rows=[], skipped=skipped,
                    trials_bytes=trials_bytes, eta_trials=eta_trials, probe_trials_note=probe_trials_note)

    # 代表性路径的**当场吞吐探针**（取第 1 个白名单 session 行为表；白名单内其余件以 fetch 行记录实测吞吐）
    first = rows[0]
    first_url = ibl_url(first[7])
    bps_s_1m, bps_s_main, probe_s5_note, _ = probe_throughput(
        "S5:ibl-public 白名单首个 session 行为表(%d B)" % first[8], first_url, label="(代表路径)",
        ref_url=IBL_AGG_TRIALS, ref_label="渠道参照 aggregates trials.pqt(24.2 MB)")

    t = 0
    integ_rows = []
    for r in rows:
        t += 1
        key, size_expect = r[7], r[8]
        url = ibl_url(key)
        p = os.path.join(RAW, "s5", key.replace("/", "__"))
        rec, clen, bps_f, resumed = fetch_full(url, p, expect_bytes=size_expect,
                                               source="S5:ibl %s" % r[5], label="(trials table)", max_time=120)
        clen = clen or size_expect

        def _pqt_parse(path):
            try:
                import pyarrow.parquet as pq
                pf = pq.ParquetFile(path)
                n = pf.metadata.num_rows
                cols = list(pf.schema_arrow.names)
                need = [c for c in ("choice", "feedback", "response_times", "contrastLeft", "contrastRight") if c in cols]
                return True, n, "parquet 行数=%d，列=%d，P8/P9 所需列命中=%s" % (n, len(cols), need)
            except Exception as e:
                return False, None, "parquet 解析失败：%s" % str(e)[:160]

        integ = integrity_gate("S5.%03d" % t, "S5:ibl %s" % r[5], url, p, clen,
                               rec.get("headers", {}).get("etag"), rec["exit"],
                               parse_fn=_pqt_parse, rows_expected=None)
        integ_rows.append(dict(eid=r[5], partition=r[9], bytes=integ["size"],
                               sha256=integ["sha256"], verdict=integ["verdict"],
                               rows=integ["rows_parsed"], etag_ok=integ["etag_ok"]))

    n_complete = sum(1 for x in integ_rows if x["verdict"] == "COMPLETE")
    total_bytes = sum(x["bytes"] for x in integ_rows if x["verdict"] == "COMPLETE")
    verdict = "REACHABLE" if n_complete == len(integ_rows) else "REACHABLE_NO_DATA"
    _REACH_ROWS.append(",".join(csv_escape(x) for x in [
        utc_now(), "S5", "IBL 行为白名单（≤%d session，_ibl_trials.table.*.pqt）" % args.max_sessions,
        "ibl_behavior", "s3_object", IBL_BUCKET + "/data/<lab>/Subjects/<subj>/<date>/<num>/alf/",
        "200", "200", total_bytes, "application/octet-stream", "", "bytes",
        "", "%.1f" % bps_s_main, "%.1f" % bps_s_1m, "%.1f" % bps_s_main,
        "%.1f" % (total_bytes / bps_s_main if bps_s_main else 0), verdict,
        "对象级 200 + 完整性门三级；白名单 %d/%d COMPLETE；分区=%s" % (n_complete, len(integ_rows),
                                                                     sorted(set(x["partition"] for x in integ_rows))),
        "",
        "%s；白名单枚举轮次=%d；显式跳过 %d 个 session（未纳入，**不得**计作'不可用'）；"
        "全量聚合 trials.pqt bytes=%s ETA=%.1f s @ %.1f B/s ⇒ **>5 MB 预算、<50 MB 硬阈值 ⇒ 未取，报主agent 裁决**"
        % (probe_s5_note, 1, len(skipped), trials_bytes, eta_trials or -1, bps_t_main)]))
    return dict(source="S5", verdict=verdict, n_complete=n_complete, n_total=len(integ_rows),
                total_bytes_complete=total_bytes, skipped=skipped, labs=labs,
                items=integ_rows, trials_bytes=trials_bytes, eta_trials=eta_trials,
                probe_trials_note=probe_trials_note, probe_whitelist_note=probe_s5_note,
                probe_repr_Bps=bps_s_main, probe_repr_1MiB_Bps=bps_s_1m,
                public_tree_bytes=int(tree_hdr.get("content-length") or 0))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all", help="S2 / S7 / S5 / all")
    ap.add_argument("--probe-only", action="store_true", help="只跑吞吐探针，不取数")
    ap.add_argument("--max-sessions", type=int, default=40, help="S5 白名单规模（预注册，默认 40）")
    ap.add_argument("--max-time", type=int, default=300, help="单次传输上限（S5FULL 用；父进程硬回收 +10 s）")
    ap.add_argument("--timeout", type=int, default=15, help="REACH 模式单次探测硬超时（默认 15 s）")
    ap.add_argument("--sids", default="", help="REACH 定向复核：只跑指定 sid（逗号分隔）")
    ap.add_argument("--attempts", type=int, default=8, help="取数最大轮数（每轮 --max-time 到点后 curl -C - 续传）")
    args = ap.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    log("=== probe_m12_sources 启动：only=%s probe_only=%s max_sessions=%d ==="
        % (args.only, args.probe_only, args.max_sessions))
    log("解释器=%s %s" % (sys.executable, sys.version.split()[0]))
    log("GPU/torch：**本工具不 import torch、不起任何仿真、不使用 MPS/GPU**（派发硬约束）")

    summary = dict(generated_by="tools/probe_m12_sources.py", generated_at=utc_now(),
                   interpreter=sys.executable, python=sys.version.split()[0],
                   only=args.only, probe_only=args.probe_only, max_sessions=args.max_sessions,
                   sources={})

    which = ["S2", "S7", "S5"] if args.only == "all" else [x.strip().upper() for x in args.only.split(",")]
    if "S2" in which:
        summary["sources"]["S2"] = run_s2(args)
        flush_throughput(); flush_integrity(); flush_reach()
    if "S7" in which:
        summary["sources"]["S7"] = run_s7(args)
        flush_throughput(); flush_integrity(); flush_reach()
    if "S5" in which:
        summary["sources"]["S5"] = run_s5(args)
        flush_throughput(); flush_integrity(); flush_reach()
    if "S5FULL" in which:
        summary["sources"]["S5FULL"] = run_s5full(args)
        flush_throughput(); flush_integrity(); flush_reach()
    if "REACH" in which:
        summary["sources"]["REACH"] = run_reach_all(args)
        flush_reach2()

    with open(JSON_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    log("summary → %s" % JSON_SUMMARY)
    log("ALL_DONE")
    return 0




# ---------------------------------------------------------------------------
# §2.5 要求的 **≥40 源** 可达性表（清单 §2 的 S1–S12 全部源 + 子端点）
# ---------------------------------------------------------------------------
def _spec_probe_once(sp, hr, probe_timeout=15):
    """单次探测一个规格。返回 dict。**父进程硬回收**由 run_curl 保证。"""
    mode, url = sp["mode"], sp["url"]
    o = dict(http="", nbytes=0, ct="", etag="", ar="", sec=0.0, speed=0.0, exit=0,
             sha="", text="", body_path=None)
    if mode == "derived":
        o.update(http="NA(no-network)", ct="not_applicable", sha="not_applicable")
        return o
    if mode == "local":
        rel_p = url.split("（")[0].split("(")[0].strip()
        full = os.path.join(os.path.dirname(ROOT), rel_p)
        if os.path.exists(full):
            o.update(http="LOCAL", nbytes=os.path.getsize(full), ct="not_applicable",
                     sha=sha256_file(full)[:16])
        else:
            o.update(http="LOCAL_MISSING", ct="not_applicable", sha="not_applicable")
        return o

    body = os.path.join(LOG_DIR, "_reach_body.bin")
    if os.path.exists(body):
        os.remove(body)
    if mode == "head":
        cmd = ["curl", "-sS", "-g", "-L", "-I", "--connect-timeout", "8",
               "--max-time", str(probe_timeout), "-o", os.devnull, "-D", hr,
               "-w", "%{http_code}|%{size_download}|%{content_type}|%{time_total}|%{speed_download}|"
                     "%{num_redirects}|%{time_starttransfer}"]
        cmd.append(url)
        t0 = time.time()
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                so, se = proc.communicate(timeout=probe_timeout + 10)
            except subprocess.TimeoutExpired:
                proc.kill()
                so, se = proc.communicate(timeout=10)
                o.update(exit=124, sec=time.time() - t0)
                if os.path.exists(hr):
                    o["hdr"] = _hdr_dict(open(hr, encoding="utf-8", errors="replace").read())
                return o
        except Exception as e:
            o.update(exit=125, text="spawn failed: %s" % str(e)[:100])
            return o
        o["exit"] = proc.returncode
        p = (so or b"").decode("utf-8", "replace").strip().split("|")
        if len(p) >= 5:
            o["http"] = p[0]
            try:
                o["sec"] = float(p[3])
                o["speed"] = float(p[4])
            except ValueError:
                pass
        if os.path.exists(hr):
            h = _hdr_dict(open(hr, encoding="utf-8", errors="replace").read())
            o["hdr"] = h
            o["ct"] = h.get("content-type", "")
            o["etag"] = h.get("etag", "")
            o["ar"] = h.get("accept-ranges", "")
            cl = h.get("content-length")
            o["nbytes"] = int(cl) if (cl or "").isdigit() else 0
        o["sha"] = "head_only(无响应体)"
        return o

    br = (0, 1048575) if mode == "range" else None
    extra = ["--max-filesize", "5000000"] if mode == "fetch" else None
    rec = run_curl(url, timeout=probe_timeout, out_path=body, hdr_path=hr,
                   byte_range=br, extra=extra, fail_on_error=False)
    o["exit"] = rec["exit"]
    o["http"] = rec["code"] or ""
    o["sec"] = rec["seconds"]
    o["speed"] = rec["speed"]
    o["nbytes"] = os.path.getsize(body) if os.path.exists(body) else rec["nbytes"]
    o["ct"] = rec.get("headers", {}).get("content-type", "")
    o["etag"] = rec.get("headers", {}).get("etag", "")
    o["ar"] = rec.get("headers", {}).get("accept-ranges", "")
    o["hdr"] = rec.get("headers", {})
    if os.path.exists(body) and o["nbytes"]:
        o["body_path"] = body
        raw = open(body, "rb").read()
        o["text"] = raw[:400000].decode("utf-8", "replace")
        if o["http"] in ("200", "206"):
            o["sha"] = hashlib.sha256(raw).hexdigest()[:16]
        else:
            o["sha"] = "NA(http=%s)" % o["http"]
    else:
        o["sha"] = "NA(no-body)"
    return o


def _spec_parse(sp, o1, o2):
    """由探测结果给出 (verdict, note, extra_bits)。**不猜**：只依据 http/内容。"""
    mode = sp["mode"]
    code = o1.get("http", "")
    bits = []
    # ---- 两次探测码 ----
    if o2.get("http") and o2["http"] != code:
        bits.append("**两次探测码不一致**（pass1=%s / pass2=%s）⇒ 按多数裁决需第三次；本条如实登记不稳定"
                    % (code, o2["http"]))
    # ---- 模式专属解析 ----
    if mode == "derived":
        return "NOT_APPLICABLE", "（无网络探测；纯派生/抽象登记）", bits
    if mode == "local":
        if code == "LOCAL":
            return "REACHABLE", "本机件存在（%d B，sha256 前缀 %s）⇒ **只读核对，未改动**" % (o1["nbytes"], o1["sha"]), bits
        return "NOT_EXIST", "本机件不存在（**这是本机落盘状态，不代表上游不存在**）", bits
    if code in ("", "000") or o1.get("exit") == 124:
        return "BLOCKED", "超时/无响应（父进程硬回收；curl exit=%s）——**未测 ≠ 未命中**，" % o1.get("exit", "?"), bits
    if code == "403":
        t = o1.get("text", "")[:300]
        if "Authentication credentials were not provided" in t or "NEEDS_TOKEN" in t:
            return "NEEDS_TOKEN", "403 且响应明示需要凭据 ⇒ `NEEDS_TOKEN`（**不得写成数据不存在**）", bits
        if "AccessDenied" in t:
            return "NEEDS_VERIFY", "403 AccessDenied（对象级不可读；**桶/端点是否存在未证实**）⇒ `NEEDS_VERIFY`", bits
        return "BLOCKED", "403（无凭据语义的普通拒绝）", bits
    if code == "404":
        return "NOT_EXIST", "404（该路径/对象不存在；**换变体前不得当数据可得**）", bits
    if code == "401":
        return "NEEDS_TOKEN", "401 需凭据", bits
    if code.startswith("5"):
        return "BLOCKED", "5xx 服务端错误（可重试；**不是不可用**）", bits
    if code not in ("200", "206"):
        return "BLOCKED", "非 2xx（http=%s，curl exit=%s）" % (code, o1.get("exit")), bits

    # ---- 2xx：按模式解析内容 ----
    t = o1.get("text", "")
    if mode == "count":
        try:
            j = json.loads(t)
            tr = j.get("total_rows")
            if tr is None:
                return "REACHABLE_NO_DATA", "JSON 可解析但**无 total_rows 字段**（keys=%s）" % list(j)[:6], bits
            bits.append("T1 count$eqtrue total_rows = **%s**" % tr)
            return "REACHABLE", "计数查询成功（成本极小）", bits
        except Exception as e:
            return "REACHABLE_NO_DATA", "响应非可解析 JSON（%s）" % str(e)[:80], bits
    if mode == "page":
        try:
            j = json.loads(t)
            msgs = j.get("msg") or []
            if not msgs:
                return "REACHABLE_NO_DATA", "msg 为空（**0 行 ⇒ 该过滤条件下无数据**，不是端点不可用）", bits
            bits.append("字段（%d 个）= %s" % (len(msgs[0]), sorted(msgs[0].keys())))
            return "REACHABLE", "小页返回 %d 行，字段清单已读回" % len(msgs), bits
        except Exception as e:
            return "REACHABLE_NO_DATA", "响应非可解析 JSON（%s）" % str(e)[:80], bits
    if mode == "list":
        kc = re.search(r"<KeyCount>(\d+)</KeyCount>", t)
        tr = re.search(r"<IsTruncated>(\w+)</IsTruncated>", t)
        pref = re.findall(r"<Prefix>([^<]+)</Prefix>", t)[1:]
        keys = re.findall(r"<Key>([^<]+)</Key>", t)
        sizes = [int(x) for x in re.findall(r"<Size>(\d+)</Size>", t)]
        bits.append("KeyCount=%s IsTruncated=%s 子前缀=%d 对象=%d；对象字节合计=%d"
                    % (kc.group(1) if kc else "?", tr.group(1) if tr else "?", len(pref), len(keys),
                       sum(sizes)))
        if pref:
            bits.append("子前缀（前 8）= %s" % pref[:8])
        big = sorted(zip(sizes, keys), reverse=True)[:3]
        if big:
            bits.append("最大 3 件 = %s" % [(k.rsplit("/", 1)[-1], s) for s, k in big])
        if (kc and kc.group(1) == "0") and not pref:
            return "REACHABLE_NO_DATA", "列举成功但 **KeyCount=0**（L9.2 #4 陷阱：**不是**数据可得）", bits
        return "REACHABLE", "列举成功", bits
    if mode == "range":
        if o1["nbytes"] <= 0:
            return "BLOCKED", "1 MiB Range 零字节（Range 可能不被支持）", bits
        bits.append("1 MiB Range 当场吞吐 = **%.1f B/s**（%.3f s）" % (o1["speed"], o1["sec"]))
        return "REACHABLE", "对象级 206 + 吞吐实测", bits
    if mode == "fetch":
        if o1["nbytes"] <= 0:
            return "REACHABLE_NO_DATA", "2xx 但 0 字节（REACHABLE_NO_DATA 陷阱）", bits
        bits.append("实收 %d B；sha256(prefix)=%s" % (o1["nbytes"], o1["sha"]))
        return "REACHABLE", "全量取回（限 5 MB；小件）", bits
    if mode == "epmc":
        try:
            j = json.loads(t)
            hc = (j.get("hitCount") if isinstance(j, dict) else None)
            if hc is None:
                return "REACHABLE_NO_DATA", "无法解析 hitCount", bits
            bits.append("**hitCount = %s**" % hc)
            if str(hc) == "0":
                return "REACHABLE_NO_DATA", "题录**未命中**（hitCount=0）——「未命中 ≠ 不存在」，须保留查询式", bits
            return "REACHABLE", "题录命中", bits
        except Exception as e:
            return "REACHABLE_NO_DATA", "响应非 JSON（%s）" % str(e)[:80], bits
    if mode in ("head", "http_dir", "supplement", "publisher"):
        bits.append("http=%s ct=%s clen=%s etag=%s accept-ranges=%s"
                    % (code, o1.get("ct"), o1.get("nbytes"), o1.get("etag") or "NA", o1.get("ar") or "NA"))
        return "REACHABLE", "对象级取证成功（HEAD/GET）", bits
    return "NEEDS_VERIFY", "模式 %s 无专属判定 ⇒ 保守记 NEEDS_VERIFY（**不得当 pass**）" % mode, bits


def run_reach_all(args):
    """清单 §2 的 S1–S12 全部源 + 子端点（≥40 行）逐条实测 → `data/m12_source_reachability.csv`。"""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import _m12_reach_specs as specs_mod
    specs = specs_mod.SPECS
    if args.sids:
        want = {x.strip() for x in args.sids.split(",") if x.strip()}
        specs = [x for x in specs if x["sid"] in want]
        log("REACH 定向复核：只跑 %d 条（%s）⇒ 结果作为 **append-only 更正行** 追加"
            % (len(specs), sorted(want)))
    log("=== REACH：清单 §2 的 S1–S12 全源可达性（%d 条规格）===" % len(specs))
    hr = os.path.join(LOG_DIR, "_hdr_reach.txt")
    t0 = time.time()
    counts = {}
    for i, sp in enumerate(specs, 1):
        # **spec 级 timeout 覆盖**（预注册）：对实测已知"服务端生成慢"的端点给更长窗口，
        # 否则会得到"探针窗短于 TTFB"造成的**假 BLOCKED**（判据行 M12-M-01 的同类）
        tmo = int(sp.get("timeout") or args.timeout)
        o1 = _spec_probe_once(sp, hr, tmo)
        need2 = sp["mode"] in ("count", "page", "list", "range", "fetch", "epmc", "head")
        o2 = _spec_probe_once(sp, hr, tmo) if need2 else dict(http=o1.get("http", ""))
        verdict, note, bits = _spec_parse(sp, o1, o2)
        counts[verdict] = counts.get(verdict, 0) + 1
        scope = {"REACHABLE": "对象级/内容级实测通过；**本条的作用域仅限本行列出的这一维**",
                 "REACHABLE_NO_DATA": "端点可达但**对象级未证实**或**该口径下 0 命中**",
                 "NEEDS_TOKEN": "需凭据 ⇒ **不是**数据不存在",
                 "NEEDS_VERIFY": "可达性未证实 ⇒ 保守判定，**不得当 pass**",
                 "BLOCKED": "网络/权限受阻 ⇒ **未测 ≠ 未命中**，须当场复核",
                 "NOT_EXIST": "该路径/对象不存在（本机件则为落盘状态）",
                 "NOT_APPLICABLE": "无网络语义（派生/抽象登记）"}.get(verdict, "")
        missing = ("无（本行所声明的作用域内证据完整：对象级/内容级均已实测）"
                   if verdict == "REACHABLE" else
                   ("不适用（无网络语义；本条为派生/抽象登记，不是数据源）"
                    if verdict == "NOT_APPLICABLE" else ""))
        if verdict in ("REACHABLE_NO_DATA", "NEEDS_TOKEN", "NEEDS_VERIFY", "BLOCKED", "NOT_EXIST"):
            tm = ("；★ **本行是超时/零字节**（curl exit=%s）：按 M12-M-01，其根因**可能是探针窗短于"
                  "服务端生成时延（TTFB）**而非不可达 ⇒ **须以更长窗当场复核后方可判 unavailable**"
                  % o1.get("exit")) if (o1.get("nbytes") in (0, None) and o1.get("exit") in (28, 124)) else ""
            missing = ("缺的维度：**该行的对象级内容未取得**（http=%s）；已覆盖维度=端点可达性/响应码%s%s。"
                       "⇒ 降级路径见该源在清单 §2.2 的「降级路径」条目"
                       % (o1.get("http") or "NA",
                          ("/字节数" if o1.get("nbytes") else ""), tm))
        prov = ("本轮当场探测（父进程硬回收，两次口径）｜模式=%s｜依据 HTTP %s + 内容解析：%s"
                % (sp["mode"], o1.get("http") or "NA", "; ".join(bits) or note))
        _REACH2_ROWS.append(",".join(csv_escape(x) for x in [
            utc_now(), sp["sid"], sp["source"], sp["category"], sp["kind"], sp["url"],
            o1.get("http") or "NA(no-network)" if sp["mode"] != "derived" else "NA(no-network)",
            (o2.get("http") if o2.get("http") else "NA(仅一次探测)"),
            (o1.get("nbytes") if o1.get("nbytes") else ("0" if sp["mode"] != "derived" else "NA")),
            o1.get("ct") or "NA", ("%.1f" % o1["speed"]) if (o1.get("speed") or 0) > 0 else "NA",
            verdict, scope, str(o1.get("exit", "NA")),
            sp["evidence_level"], specs_mod.__dict__.get("SHA_BASIS_HINT", "payload")
            if False else ("file_bytes" if sp["mode"] == "local" else
                           ("head_only" if sp["mode"] == "head" else
                            ("not_applicable" if sp["mode"] == "derived" else "payload"))),
            o1.get("sha") or "NA", prov, sp["url"], utc_now(), sp["wrong"], missing,
            ("%s｜%s" % (sp["note"], note))[:1200]]))
        log("REACH [%02d/%d] %-7s %-6s %s  %s" % (i, len(specs), sp["sid"], verdict,
                                                  (o1.get("http") or "-"), sp["source"][:52]))
        if i % 10 == 0:
            flush_reach2()
            log("REACH 进度：%d/%d，已用 %.1f s" % (i, len(specs), time.time() - t0))
    flush_reach2()
    log("REACH 完成：%d 行，用时 %.1f s；四态分布=%s" % (len(specs), time.time() - t0, counts))
    return dict(n_specs=len(specs), by_verdict=counts, seconds=round(time.time() - t0, 1),
                csv=os.path.relpath(CSV_REACH2, os.path.dirname(ROOT)))


if __name__ == "__main__":
    sys.exit(main())
