#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M11 数据门（G2/P1/P2）· 数据源可达性探针（**可复跑、确定性、逐源实测**）

用途
----
对《生物仿真M11实施清单.md》§1.6 的 **51 个预注册数据源**（source_id 1..51）逐源实测，
并追加 M11 特有的**补测源**（source_id F1..Fxx，清单 §1.5/§1.7 标注的"G2 首要补测项"）。
输出 `data/m11_source_reachability.csv`（逐行 provenance）。

判定词表（与清单 §1.6 一致；**不得把 NEEDS_VERIFY 写成 EXISTS，不得把
REACHABLE_NO_DATA 写成有数据**）
--------------------------------------------------------------------
- `REACHABLE`          对象级 200/206 实测，且返回体非空、非 parked page
- `REACHABLE_NO_DATA`  端点/桶可达但**未证实对象数据**（0 字节、parked page、空列举、路径未确认的 404）
- `NEEDS_TOKEN`        401/403 且响应明示需要凭据（认证凭据未提供）
- `NEEDS_VERIFY`       桶存在但**对象级可读性未证实**（403 AccessDenied 而非 NoSuchBucket；可 list 但未取对象）
- `BLOCKED`            403/连接重置/超时/DNS 失败/TLS 失败/5xx
- `NOT_EXIST`          明确不存在（`NoSuchBucket` / `NoSuchKey`）

方法（纪律）
------------
- 传输后端 = **curl 子进程**（M10 L11：Python urllib 对 GCS 的 TLS 握手会超时）
- 每源**两次探测**（记录 http_code_pass1 / http_code_pass2；判定取两次一致的语义）
- 单源硬超时 ≤ `--timeout`（默认 15 s）；大对象一律 `--range 0-N` 或 `--max-filesize`
  （**禁止**为了探针下载大档）
- `PYTHONHASHSEED=0`；无随机；输出按 source_id 固定顺序（确定性）
- 增量落盘 + 结束时打 `ALL_DONE`（项目教训：长任务误杀）
- 稳定字段：verdict / http_code / bytes_range / content_type；易变字段：time_total / speed（不参与判定）

用法
----
    cd /Users/weidong/ai/small_world
    PYTHONHASHSEED=0 ./.venv-m11/bin/python neural_exploration/tools/probe_m11_sources.py --all
    # 只跑清单 51 源 / 只跑补测源 / 只跑单个：
    ... --only 1-51 / --only extras / --only 1,17,32,F8
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../neural_exploration
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "m11_raw")
PROBE_RAW = os.path.join(RAW, "probe")
CSV_OUT = os.path.join(DATA, "m11_source_reachability.csv")
LOG_OUT = os.path.join(PROBE_RAW, "probe_m11_sources.log")

GCS_MICRONS = "https://storage.googleapis.com/iarpa_microns"
S3_BOSS = "https://bossdb-open-data.s3.amazonaws.com"

# ---------------------------------------------------------------------------
# 源清单（清单 §1.6 表；source_id 1..51 **严格按清单行号**）
# kind: info(元数据 JSON) / obj(具体对象) / list(前缀/桶列举) / portal(HTML 门户)
#       / api(JSON API) / bucket_head(桶根)
# ---------------------------------------------------------------------------
SOURCES = [
    dict(sid="1", name="MICrONS EM (GCS)", category="microns_structure", kind="info",
         url=f"{GCS_MICRONS}/minnie/minnie65/em/info"),
    dict(sid="2", name="MICrONS EM (S3 镜像)", category="microns_structure", kind="info",
         url=f"{S3_BOSS}/iarpa_microns/minnie/minnie65/em/info"),
    dict(sid="3", name="MICrONS seg (GCS)", category="microns_structure", kind="info",
         url=f"{GCS_MICRONS}/minnie/minnie65/seg/info"),
    dict(sid="4", name="MICrONS seg_m1300 (GCS)", category="microns_structure", kind="info",
         url=f"{GCS_MICRONS}/minnie/minnie65/seg_m1300/info"),
    dict(sid="5", name="MICrONS GCS 桶列举", category="microns_structure", kind="list",
         url=f"{GCS_MICRONS}/?list-type=2&max-keys=6"),
    dict(sid="6", name="BossDB 开放桶列举", category="microns_structure", kind="list",
         url=f"{S3_BOSS}/?list-type=2&delimiter=/&prefix=iarpa_microns/"),
    dict(sid="7", name="MICrONS 功能数据根", category="microns_functional", kind="list",
         url=f"{S3_BOSS}/?list-type=2&delimiter=/&max-keys=25&prefix=iarpa_microns/minnie/functional_data/"),
    dict(sid="8", name="MICrONS coregistration", category="microns_coreg", kind="list",
         url=f"{S3_BOSS}/?list-type=2&delimiter=/&max-keys=25&prefix=iarpa_microns/minnie/functional_coregistration/"),
    dict(sid="9", name="MICrONS 钙电影分片 (functional_series chunk)", category="microns_functional", kind="obj",
         per_timeout=90,
         url=f"{S3_BOSS}/iarpa_microns/minnie/functional_data/functional_series/session4_scan7_field4/0/0/0/0"),
    dict(sid="10", name="MICrONS 功能连接组学属性前缀", category="microns_functional", kind="list",
         url=f"{S3_BOSS}/?list-type=2&delimiter=/&max-keys=25&prefix=iarpa_microns/minnie/functional_data/functional_connectomics/"),
    dict(sid="11", name="MICrONS 刺激电影前缀", category="microns_functional", kind="list",
         url=f"{S3_BOSS}/?list-type=2&delimiter=/&max-keys=25&prefix=iarpa_microns/minnie/functional_data/platinum_stimulus_movies/"),
    dict(sid="12", name="MICrONS 技术文档 v3 (PDF)", category="microns_functional", kind="obj",
         url=f"{S3_BOSS}/iarpa_microns/minnie/functional_data/functional_data_technical_documentation_v3.pdf"),
    dict(sid="13", name="MICrONS 官方教程 (CloudVolume)", category="microns_doc", kind="portal",
         url="https://tutorial.microns-explorer.org/quickstart_notebooks/06-cloudvolume-download-mesh.html"),
    dict(sid="14", name="MICrONS Explorer 门户", category="microns_doc", kind="portal",
         url="https://www.microns-explorer.org/"),
    dict(sid="15", name="MICrONS Neuroglancer 前端", category="microns_doc", kind="portal",
         url="https://ngl.microns-explorer.org/"),
    dict(sid="16", name="minnie CAVE UI", category="microns_cave", kind="portal",
         url="https://minnie.microns-daf.com/"),
    dict(sid="17", name="minnie CAVE 表 API (datastacks)", category="microns_cave", kind="api",
         url="https://minnie.microns-daf.com/info/api/v2/datastacks"),
    dict(sid="18", name="新版 CAVE global.daf-apis", category="microns_cave", kind="api",
         url="https://global.daf-apis.com/info/api/v2/datastacks"),
    dict(sid="19", name="GCS microns_public_datasets", category="microns_other", kind="list",
         url="https://storage.googleapis.com/microns_public_datasets/?list-type=2&max-keys=5"),
    dict(sid="20", name="GCS microns-seunglab", category="microns_other", kind="list",
         url="https://storage.googleapis.com/microns-seunglab/?list-type=2&max-keys=5"),
    dict(sid="21", name="Allen Cell Types 门户", category="allen", kind="portal",
         url="https://celltypes.brain-map.org/"),
    dict(sid="22", name="Allen Cell Types API (小鼠 count)", category="allen", kind="api",
         url=("https://api.brain-map.org/api/v2/data/query.json?criteria="
              "model%3A%3AApiCellTypesSpecimenDetail%2Crma%3A%3Acriteria%5Bdonor__species%24eq%27Mus%20musculus%27%5D"
              "%2Crma%3A%3Aoptions%5Bnum_rows%24eq1%5D%5Bcount%24eqtrue%5D")),
    dict(sid="23", name="Allen per-cell NWB (range 0-99)", category="allen", kind="obj", byte_range=(0, 99),
         url="https://api.brain-map.org/api/v2/well_known_file_download/664222041"),
    dict(sid="24", name="Allen Cell Types S3 (假设桶)", category="allen", kind="bucket_head",
         url="https://allen-cell-types.s3.amazonaws.com/"),
    dict(sid="25", name="Allen Brain Observatory 门户", category="allen", kind="portal",
         url="https://observatory.brain-map.org/visualcoding"),
    dict(sid="26", name="Allen BO API", category="allen", kind="api",
         url=("https://api.brain-map.org/api/v2/data/query.json?criteria="
              "model%3A%3AApiBrainObservatoryData%2Crma%3A%3Aoptions%5Bnum_rows%24eq1%5D")),
    dict(sid="27", name="Allen BO Neuropixels NWB (S3 列举)", category="allen", kind="list",
         url="https://allen-brain-observatory.s3.amazonaws.com/?list-type=2&max-keys=4&prefix=visual-coding-neuropixels/ecephys-cache"),
    dict(sid="28", name="Allen Visual Behavior Neuropixels (S3)", category="allen", kind="list",
         url="https://visual-behavior-neuropixels-data.s3.us-west-2.amazonaws.com/?list-type=2&max-keys=4"),
    dict(sid="29", name="Allen via DANDI 000021", category="allen", kind="api",
         url="https://api.dandiarchive.org/api/dandisets/000021/versions/"),
    dict(sid="30", name="allensdk 文档", category="allen", kind="portal",
         url="https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html"),
    dict(sid="31", name="Allen synphys (AWS Open Data 元数据)", category="allen_synphys", kind="portal",
         url="https://registry.opendata.aws/allen-synphys/"),
    dict(sid="32", name="Allen synphys 桶列举", category="allen_synphys", kind="list",
         url="https://allen-synphys.s3.amazonaws.com/?list-type=2&max-keys=3"),
    dict(sid="33", name="Allen synphys 门户", category="allen_synphys", kind="portal",
         url="https://portal.brain-map.org/explore/connectivity/synaptic-physiology", insecure=False),
    dict(sid="34", name="synapse.brain-map.org (DNS)", category="allen_synphys", kind="portal",
         url="https://synapse.brain-map.org/"),
    dict(sid="35", name="IBL OpenAlyx API", category="ibl", kind="api",
         url="https://openalyx.internationalbrainlab.org/datasets?format=json"),
    dict(sid="36", name="IBL 公开桶 (S3)", category="ibl", kind="list",
         url="https://ibl-brain-wide-map-public.s3.amazonaws.com/?list-type=2&max-keys=8&delimiter=/"),
    dict(sid="37", name="Europe PMC REST", category="literature", kind="api",
         url=("https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=TITLE%3A%22Functional%20connectomics%20"
              "spanning%20multiple%20areas%20of%20mouse%20visual%20cortex%22&format=json&pageSize=5")),
    dict(sid="38", name="nature.com MICrONS 论文", category="literature", kind="portal",
         url="https://www.nature.com/articles/s41586-025-08790-w", max_filesize=1048576),
    dict(sid="39", name="elifesciences.org", category="literature", kind="portal",
         url="https://elifesciences.org/", max_filesize=524288),
    dict(sid="40", name="DANDI API (总)", category="archive", kind="api",
         url="https://api.dandiarchive.org/api/dandisets/?page_size=1"),
    dict(sid="41", name="NMC 门户", category="nmc", kind="portal",
         url="https://bbp.epfl.ch/nmc-portal/"),
    dict(sid="42", name="NMC 下载路径", category="nmc", kind="portal",
         url="https://bbp.epfl.ch/nmc-portal/download"),
    dict(sid="43", name="EBRAINS KG 搜索", category="nmc", kind="portal",
         url="https://search.kg.ebrains.eu/"),
    dict(sid="44", name="GitHub API", category="code", kind="api", url="https://api.github.com/"),
    dict(sid="45", name="raw.githubusercontent.com", category="code", kind="obj",
         url="https://raw.githubusercontent.com/AllenInstitute/allenSDK/master/README.md"),
    dict(sid="46", name="Zenodo API", category="archive", kind="api",
         url="https://zenodo.org/api/records/1009488"),
    dict(sid="47", name="figshare API", category="archive", kind="api",
         url="https://api.figshare.com/v2/articles?page_size=1"),
    dict(sid="48", name="GIN", category="archive", kind="portal", url="https://gin.g-node.org/"),
    dict(sid="49", name="OpenNeuro", category="archive", kind="portal", url="https://openneuro.org/"),
    dict(sid="50", name="DataLad", category="code", kind="portal", url="https://www.datalad.org/"),
    dict(sid="51", name="brain-map.org 总站", category="allen", kind="portal",
         url="https://brain-map.org/", max_filesize=524288),
]

# ---------------------------------------------------------------------------
# 补测源（M11 特有；清单 §1.5「采集参数读回」/§1.7「G2 首要补测项」）
# ---------------------------------------------------------------------------
EXTRA = [
    dict(sid="F1", name="[补测] minnie CAVE materialize version", category="microns_cave", kind="api",
         url="https://minnie.microns-daf.com/materialize/api/v2/datastack/minnie65_public/version"),
    dict(sid="F2", name="[补测] GCS iarpa_microns 正确前缀 (minnie/)", category="microns_structure", kind="list",
         url=f"{GCS_MICRONS}/?list-type=2&delimiter=/&max-keys=10&prefix=minnie/minnie65/"),
    dict(sid="F3", name="[补测] GCS em/8x8x40 shard 清单", category="microns_structure", kind="list",
         url=f"{GCS_MICRONS}/?list-type=2&max-keys=1000&prefix=minnie/minnie65/em/8x8x40/"),
    dict(sid="F4", name="[补测] GCS seg/8x8x40 shard 清单", category="microns_structure", kind="list",
         url=f"{GCS_MICRONS}/?list-type=2&max-keys=1000&prefix=minnie/minnie65/seg/8x8x40/"),
    dict(sid="F5", name="[补测] GCS seg_m1300/8.0x8.0x40.0 shard 清单", category="microns_structure", kind="list",
         url=f"{GCS_MICRONS}/?list-type=2&max-keys=1000&prefix=minnie/minnie65/seg_m1300/8.0x8.0x40.0/"),
    dict(sid="F6", name="[补测] seg_m1300 S3 镜像 info", category="microns_structure", kind="info",
         url=f"{S3_BOSS}/iarpa_microns/minnie/minnie65/seg_m1300/info"),
    dict(sid="F7", name="[补测] functional_series attributes.json (采集参数)", category="microns_functional", kind="info",
         url=f"{S3_BOSS}/iarpa_microns/minnie/functional_data/functional_series/session4_scan7_field4/attributes.json"),
    dict(sid="F8", name="[补测] field_summaries attributes.json", category="microns_functional", kind="info",
         url=f"{S3_BOSS}/iarpa_microns/minnie/functional_data/field_summaries/session_4_scan_7_field_4/attributes.json"),
    dict(sid="F9", name="[补测] field_seg_masks info (神经域掩膜格)", category="microns_functional", kind="info",
         url=f"{S3_BOSS}/iarpa_microns/minnie/functional_data/field_seg_masks/session_4_scan_7_field_4/info"),
    dict(sid="F10", name="[补测] field_summaries 全字段前缀列举", category="microns_functional", kind="list",
         url=f"{S3_BOSS}/?list-type=2&delimiter=/&max-keys=1000&prefix=iarpa_microns/minnie/functional_data/field_summaries/"),
    dict(sid="F11", name="[补测] two_photon_functional_scans 前缀列举", category="microns_functional", kind="list",
         url=f"{S3_BOSS}/?list-type=2&delimiter=/&max-keys=25&prefix=iarpa_microns/minnie/functional_data/two_photon_functional_scans/"),
    dict(sid="F12", name="[补测] 原始 BigTIFF 头 (range 0-65535)", category="microns_functional", kind="obj",
         byte_range=(0, 65535),
         url=(f"{S3_BOSS}/iarpa_microns/minnie/functional_data/two_photon_functional_scans/"
              "functional_scan_17797_4_10_v1.tif")),
    dict(sid="F13", name="[补测] 双光子处理/元数据 v8 技术文档 (PDF)", category="microns_functional", kind="obj",
         url=(f"{S3_BOSS}/iarpa_microns/minnie/functional_data/two_photon_processed_data_and_metadata/"
              "database_v8/two_photon_processed_data_and_metadata_technical_documentation_v8.pdf")),
    dict(sid="F14", name="[补测] access_v8 README (处理管线语义)", category="microns_functional", kind="obj",
         url=(f"{S3_BOSS}/iarpa_microns/minnie/functional_data/two_photon_processed_data_and_metadata/"
              "access_v8/README.md")),
    dict(sid="F15", name="[补测] IBL 公开桶对象级 (aggregates 前缀)", category="ibl", kind="list",
         url=("https://ibl-brain-wide-map-public.s3.amazonaws.com/?list-type=2&max-keys=10&delimiter=/"
              "&prefix=aggregates/")),
    dict(sid="F16", name="[补测] Allen synphys 桶对象级 HEAD", category="allen_synphys", kind="bucket_head",
         url="https://allen-synphys.s3.amazonaws.com/"),
    dict(sid="F17", name="[补测] node_and_edge_properties/v1 前缀", category="microns_functional", kind="list",
         url=(f"{S3_BOSS}/?list-type=2&delimiter=/&max-keys=25&prefix=iarpa_microns/minnie/functional_data/"
              "functional_connectomics/node_and_edge_properties/v1/")),
]

VERDICT_VOCAB = ["REACHABLE", "REACHABLE_NO_DATA", "NEEDS_TOKEN", "NEEDS_VERIFY", "BLOCKED", "NOT_EXIST"]

CSV_FIELDS = ["source_id", "source", "category", "kind", "url", "http_code", "http_code_pass2",
              "bytes", "content_type", "time_total_s", "speed_Bps", "verdict", "note"]


# ---------------------------------------------------------------------------
_TMPSEQ = [0]


def _run_curl(url, timeout=15, byte_range=None, max_filesize=None, insecure=False, user_agent=None):
    """单次探测。返回 dict(code, bytes, ct, exit, t, speed, body_path, body_head, err)。

    每次调用使用**独立临时文件**（避免两次探测互相覆盖 body，保证 sha256 provenance 诚实）。
    """
    _TMPSEQ[0] += 1
    fd = os.path.join(PROBE_RAW, "_tmp_probe_%03d.bin" % _TMPSEQ[0])
    if os.path.exists(fd):
        os.remove(fd)
    cmd = ["curl", "-sS", "-L", "--connect-timeout", "8", "--max-time", str(timeout),
           "-o", fd, "-w", "%{http_code}|%{size_download}|%{content_type}|%{num_redirects}|%{time_total}|%{speed_download}"]
    if byte_range is not None:
        cmd += ["-r", "%d-%d" % byte_range]
    if max_filesize is not None:
        cmd += ["--max-filesize", str(max_filesize)]
    if insecure:
        cmd += ["-k"]
    if user_agent:
        cmd += ["-A", user_agent]
    cmd.append(url)
    rec = dict(code=0, nbytes=0, ct="", exit=0, t=0.0, speed=0.0, body_path=None, body_head="", err="")
    # **硬超时**：Popen + communicate(timeout) + kill（某些站点上 curl --max-time 会失效，
    # 实测出现过 925 s 挂死 → 必须由父进程强制回收）
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
            rec["err"] = "hard kill by parent (curl --max-time %s 未生效)" % timeout
            if os.path.exists(fd):
                rec["body_path"] = fd
            return rec
    except Exception as e:
        rec["exit"] = 125
        rec["err"] = "spawn failed: %s" % str(e)[:120]
        return rec
    rec["exit"] = proc.returncode
    class _P:
        pass
    p = _P()
    p.returncode = proc.returncode
    p.stdout = so
    p.stderr = se
    out = p.stdout.decode("utf-8", "replace").strip()
    parts = out.split("|")
    if len(parts) >= 6:
        try:
            rec["code"] = int(parts[0]) if parts[0] else 0
        except ValueError:
            rec["code"] = 0
        try:
            rec["nbytes"] = int(float(parts[1]))
        except ValueError:
            rec["nbytes"] = 0
        rec["ct"] = parts[2]
        try:
            rec["t"] = float(parts[4])
            rec["speed"] = float(parts[5])
        except ValueError:
            pass
    if os.path.exists(fd):
        rec["body_path"] = fd
        with open(fd, "rb") as f:
            head = f.read(65536)
        rec["body_head"] = head.decode("utf-8", "replace")
    if p.returncode != 0:
        rec["err"] = p.stderr.decode("utf-8", "replace").strip()[:300]
    return rec


PARKED_MARKERS = ["domain for sale", "buy this domain", "future home of", "this domain is parked",
                  "sedoparking", "parkingcrew", "hugedomains", "namecheap parking"]


def classify(src, rec):
    """确定性判定：仅依赖 (curl_exit, http_code, bytes, content_type, body)。"""
    code, n, ct, head = rec["code"], rec["nbytes"], (rec["ct"] or ""), rec["body_head"]
    low = head[:20000].lower()
    kind = src["kind"]
    if rec["exit"] == 124 or code == 0:
        return "BLOCKED", "无响应（curl_exit=%s: %s）" % (rec["exit"], (rec["err"] or "timeout")[:160])
    if rec["exit"] == 6:
        return "BLOCKED", "DNS 解析失败（curl 6: %s）" % (rec["err"] or "")[:120]
    if rec["exit"] in (35, 56):
        return "BLOCKED", "连接/取数失败（curl %d: %s）" % (rec["exit"], (rec["err"] or "")[:140])
    if rec["exit"] == 60:
        return "BLOCKED", "TLS 证书失败（curl 60，自签证书；未用 -k）"
    if rec["exit"] == 63:
        return "REACHABLE", "超过 --max-filesize 截断（HTTP %d，已收 %d B）" % (code, n)
    if rec["exit"] == 28:
        return "BLOCKED", "超时（curl 28，HTTP %s，收 %d B）" % (code, n)
    if rec["exit"] not in (0,):
        # 其它 curl 错误：仍按 HTTP 码兜底，但如实记录
        pass
    if code in (401,):
        if "authentication credentials" in low or "unauthorized" in low or "token" in low:
            return "NEEDS_TOKEN", "HTTP 401，需凭据（%s）" % head[:120].replace("\n", " ")
        return "NEEDS_VERIFY", "HTTP 401，未证实对象级可读（%s）" % head[:120].replace("\n", " ")
    if code == 403:
        if "nosuchbucket" in low:
            return "NOT_EXIST", "HTTP 403 但含 NoSuchBucket"
        if "authentication credentials" in low or "not authenticated" in low or "credentials were not provided" in low:
            return "NEEDS_TOKEN", "HTTP 403，需凭据（%s）" % head[:120].replace("\n", " ")
        if "accessdenied" in low or "access denied" in low:
            if kind == "list":
                return "NEEDS_VERIFY", "HTTP 403 AccessDenied（非 NoSuchBucket ⇒ 桶存在、不可 list）；对象级未证实"
            return "BLOCKED", "HTTP 403 AccessDenied（对象级不可读）"
        if "self signed" in low or "certificate" in low:
            return "BLOCKED", "HTTP 403（TLS 自签证书站点，-k 后仍 403）"
        return "BLOCKED", "HTTP 403（%s）" % head[:120].replace("\n", " ")
    if code == 404:
        if "nosuchbucket" in low:
            return "NOT_EXIST", "HTTP 404 + NoSuchBucket（桶不存在，XML 明示）"
        if "nosuchkey" in low:
            return "NOT_EXIST", "HTTP 404 + NoSuchKey（对象不存在）"
        if kind == "list":
            return "REACHABLE_NO_DATA", "HTTP 404（列举路径未确认，但未见 NoSuchBucket）"
        return "REACHABLE_NO_DATA", "HTTP 404（路径未确认；非 NoSuchBucket/Key）"
    if code in (500, 502, 503, 504):
        return "BLOCKED", "HTTP %d（服务端错误）" % code
    if code in (200, 206):
        br = src.get("byte_range")
        if br is not None and code == 206 and n == (br[1] - br[0] + 1):
            return "REACHABLE", "HTTP 206 Range %s-%s 精确命中（%d B）" % (br[0], br[1], n)
        if n == 0:
            return "REACHABLE_NO_DATA", "HTTP %d 但 0 字节（无对象数据）" % code
        for mk in PARKED_MARKERS:
            if mk in low:
                return "REACHABLE_NO_DATA", "HTTP %d / %d B：parked page（%s）" % (code, n, mk)
        if kind == "list":
            m = re.search(r"<KeyCount>(\d+)</KeyCount>", head)
            kc = int(m.group(1)) if m else None
            npfx = len(re.findall(r"<CommonPrefixes>", head))
            if kc == 0 and npfx == 0:
                return "REACHABLE_NO_DATA", "HTTP %d 列举成功但 KeyCount=0（无内容）" % code
            return "REACHABLE", "HTTP %d 列举成功：KeyCount=%s，CommonPrefixes=%d" % (code, kc, npfx)
        if n < 200:
            return "REACHABLE_NO_DATA", "HTTP %d 但仅 %d B（非数据体；小体积 HTML/占位）" % (code, n)
        return "REACHABLE", "HTTP %d / %d B / %s" % (code, n, ct or "?")
    if 300 <= code < 400:
        return "NEEDS_VERIFY", "HTTP %d（重定向未落地，-L 后仍为 3xx）" % code
    return "NEEDS_VERIFY", "HTTP %s（未归类；如实登记）" % code


def parse_only(spec, all_sources):
    """--only: '1-51' / 'extras' / 'all' / '1,17,F8'"""
    if spec in (None, "", "all", "ALL"):
        return all_sources
    keep = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.lower() in ("extra", "extras", "f", "F"):
            keep += [s for s in all_sources if s["sid"].startswith("F")]
        elif "-" in tok and not tok.startswith("F"):
            a, b = tok.split("-")
            keep += [s for s in all_sources if s["sid"].isdigit() and int(a) <= int(s["sid"]) <= int(b)]
        else:
            keep += [s for s in all_sources if s["sid"] == tok]
    seen, out = set(), []
    for s in all_sources:
        if s in keep and s["sid"] not in seen:
            out.append(s)
            seen.add(s["sid"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="跑全部（51 清单源 + 补测源）")
    ap.add_argument("--only", default=None, help="1-51 / extras / 1,17,F8")
    ap.add_argument("--timeout", type=float, default=15.0, help="单源硬超时（s，默认 15）")
    ap.add_argument("--passes", type=int, default=2, help="每源探测次数（默认 2，用于 HTTP 码稳定性）")
    ap.add_argument("--out", default=CSV_OUT)
    args = ap.parse_args()

    os.makedirs(PROBE_RAW, exist_ok=True)
    if args.only is not None and os.path.abspath(args.out) == os.path.abspath(CSV_OUT):
        print("拒绝执行：--only 会覆盖全量 m11_source_reachability.csv；请显式指定 --out（如 /tmp/x.csv）",
              file=sys.stderr)
        return 2
    if args.only is None:
        todo = (SOURCES + EXTRA) if args.all else SOURCES
    else:
        todo = parse_only(args.only, SOURCES + EXTRA)

    logf = open(LOG_OUT, "a", buffering=1)
    logf.write("\n=== probe run %s  n_sources=%d timeout=%.1f passes=%d PYTHONHASHSEED=%s\n"
               % (datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), len(todo), args.timeout,
                  args.passes, os.environ.get("PYTHONHASHSEED", "(unset)")))

    rows = []
    # 表头（含 provenance 注释行，沿 M10 m10_source_reachability.csv 惯例）
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        f.write("# M11 数据源可达性实测（generated_by=tools/probe_m11_sources.py；"
                "method=curl 子进程 -L --connect-timeout 8 --max-time %.0f；两次探测；"
                "verdict 词表=%s）\n" % (args.timeout, "/".join(VERDICT_VOCAB)))
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        f.flush()
        for i, src in enumerate(todo):
            st = src.get("per_timeout", args.timeout)
            recs = []
            for _ in range(max(1, args.passes)):
                recs.append(_run_curl(src["url"], timeout=st,
                                      byte_range=src.get("byte_range"),
                                      max_filesize=src.get("max_filesize"),
                                      insecure=src.get("insecure", False)))
            r0 = recs[0]
            r1 = recs[-1]
            verdict, note = classify(src, r0)
            v1, n1 = classify(src, r1)
            if v1 != verdict:
                # 两次不一致 → 补第三次（多数裁决；三次码全部记录）
                r2 = _run_curl(src["url"], timeout=st, byte_range=src.get("byte_range"),
                               max_filesize=src.get("max_filesize"), insecure=src.get("insecure", False))
                v2, n2 = classify(src, r2)
                votes = [verdict, v1, v2]
                maj = max(set(votes), key=votes.count)
                note += "；⚠️ 三次探测不一致：%s/%s/%s（码 %s/%s/%s）→ 按多数裁决 %s" % (
                    verdict, v1, v2, r0["code"], r1["code"], r2["code"], maj)
                verdict = maj
                recs.append(r2)
            r1 = recs[-1]
            sha = ""
            if r0["body_path"] and os.path.exists(r0["body_path"]):
                with open(r0["body_path"], "rb") as fh:
                    sha = hashlib.sha256(fh.read()).hexdigest()
            if sha and src.get("byte_range") is None and r0["exit"] == 0:
                note += "；sha256=%s" % sha[:16]
            note += "；pass1_code=%s pass2_code=%s" % (r0["code"], r1["code"])
            row = dict(source_id=src["sid"], source=src["name"], category=src["category"], kind=src["kind"],
                       url=src["url"], http_code=r0["code"], http_code_pass2=r1["code"],
                       bytes=r0["nbytes"], content_type=r0["ct"],
                       time_total_s=round(r0["t"], 3), speed_Bps=int(r0["speed"]),
                       verdict=verdict, note=note)
            rows.append(row)
            w.writerow(row)
            f.flush()
            logf.write("[%02d/%02d] %-4s %-46s %-18s HTTP %-4s %9d B  %.2fs  %s\n"
                       % (i + 1, len(todo), src["sid"], src["name"][:46], verdict, r0["code"],
                          r0["nbytes"], r0["t"], note[:110]))
            if r0["body_path"] and os.path.exists(r0["body_path"]):
                os.remove(r0["body_path"])
            if r1["body_path"] and os.path.exists(r1["body_path"]):
                os.remove(r1["body_path"])

    n_by = {}
    for r in rows:
        n_by[r["verdict"]] = n_by.get(r["verdict"], 0) + 1
    logf.write("SUMMARY n=%d %s\n" % (len(rows), json.dumps(n_by, sort_keys=True)))
    logf.write("ALL_DONE %s\n" % datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    logf.close()
    print("wrote %s (%d rows)" % (args.out, len(rows)))
    print(json.dumps(n_by, sort_keys=True))


if __name__ == "__main__":
    sys.exit(main())
