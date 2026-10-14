#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M12 数据门 · **S1（Allen Mouse Brain Connectivity Atlas 投射矩阵）当场裁决探针**。

节点归属：M12 执行节点 **S1-GATE**（与 M11 G1 并行运行；**不消费 M11 的 G1/S 任何结论**）。
本工具**只新建**：不改 `tools/probe_m12_sources.py`（B1 节点件）、不改任何 `m11_*` 件。
输出全部落在**本节点自有路径**（`data/m12_s1_gate_*`、`data/m12_s1_raw/**`），
以避开 M11-R60「共用输出路径 ⇒ 后跑者静默覆盖」。

## 本工具要回答的唯一问题（清单行号）
`docs/生物仿真M12实施清单.md`：
- **第 453–461 行**（§2.2 S1）给出 S1 的**预注册四态判据**：
  - **第 460 行** 🔸 `unavailable（吞吐限制）` ⇐ `行数 ≥ 10⁶` **且** 实测吞吐 ≤ 16 KB/s（ETA 17–35 h）
  - **第 460 行** 🔹 `pass`               ⇐ **服务端白名单过滤后行数 ≤ 10⁵**（≈10 MB）**或** 实测吞吐 ≥ 200 KB/s
- **第 429–436 行**（§2.3 T1–T6）：T4 门 = `ETA_对象 > 30 min` 或 `ETA_总计 > 4 h` ⇒ `unavailable（吞吐限制）`；
  T6（M12 新增第 4 层）= 取得到但**装不进模型** ⇒ `unavailable（装配不可行）`。
- **第 461 行** 降级路径 ①区域族汇总 ②种子实验子集 ③文献档 ④距离核抽象。
- **第 600 行**（§2.4）：S1 不可得 ⇒ P6(C-M) 判 `unavailable`、**禁止声称"图谱权重约束下"**。

而 `docs/m12_b1_gate_closure.md` **§3 表首行（第 238 行）** 明写 S1 仍缺**两维**：
「服务端白名单过滤后的行数实测」与「实测吞吐下的 ETA」；
`docs/m12_list_review_master.md` **R41-2 条件 1（第 19–21 行）** 要求 S1 的 `unavailable`
主假设**必须以当场探针确认后才生效**。

## 纪律固化（逐条写进代码，事中不得择口径）
1. **父进程硬回收**：一切网络调用 = `Popen` + `communicate(timeout)` + `kill()`（M11 L9.3a）。
2. **必登 TTFB**；`bytes==0 且 TTFB >= 探针窗` ⇒ `probe_window_too_short`，**不得**判 `TOO_SLOW`（M12-M-01）。
3. **`--fail` 只用于会落盘成对象的取数**（M12-M-04）；探针必须允许错误体进来用于分类（B1 L25 自纠 #1）。
4. **`-g`（globoff）强制**（B1 L14 #1：`[`/`]` 被 curl 当 range 表达式）。
5. **`HTTP 200` 不等于查询成功**：RMA 的语法错误返回 **HTTP 200 + `success:false` + `total_rows:0`**
   ⇒ 若只看 `total_rows` 会把「查询写错」读成「该过滤下没有数据」（**静默失效家族**）。
   故本工具强制同时解析 `success` 与 `msg` 中的错误文本（本节点本轮实测的形态见 A14）。
6. **两次探测 + 码不一致时第三次多数裁决**。
7. **独立性交叉校验**：`$in<白名单>` 的计数必须与**逐 id 单独计数之和**逐值相等（三路独立估计范式，R-6 同族）。
8. **单调性自检**：白名单层级嵌套（depth≤2 ⊂ ≤3 ⊂ ≤5 ⊂ ≤6）⇒ 计数必须单调不减（`is_injection` 维度同样）。
9. **吞吐不可外推**（M11-R56/R63）：每行登记判决时刻 `utc` + 当时实测吞吐 + `recheck_policy`（≤24 h）。
10. 零 `sys.executable`；本工具**只用标准库**（R34：既非引擎侧也非评估器侧，纯 IO ⇒ 系统 `python3`）。

## 用法
    python3 neural_exploration/tools/probe_m12_s1_gate.py --phase A
    python3 neural_exploration/tools/probe_m12_s1_gate.py --all
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

# ---------------------------------------------------------------- 路径与常量
HERE = os.path.dirname(os.path.abspath(__file__))
NE = os.path.dirname(HERE)
ROOT = os.path.dirname(NE)
DATA = os.path.join(NE, "data")
RAW = os.path.join(DATA, "m12_s1_raw")
S2_STRUCT = os.path.join(DATA, "m12_raw", "s2", "model_Structure_all.json")

OUT_REACH = os.path.join(DATA, "m12_s1_gate_reachability.csv")
OUT_THRU = os.path.join(DATA, "m12_s1_gate_throughput.csv")
OUT_INTEG = os.path.join(DATA, "m12_s1_gate_integrity.csv")
OUT_COUNTS = os.path.join(DATA, "m12_s1_gate_counts.json")
OUT_ASSEM = os.path.join(DATA, "m12_s1_gate_assembly.json")
OUT_SYNTAX = os.path.join(DATA, "m12_s1_gate_syntax.csv")
OUT_WL = os.path.join(DATA, "m12_s1_region_whitelist.csv")

API = "https://api.brain-map.org/api/v2/data"

W_COUNT = 90      # 计数窗（B1 L25 自纠 #2：短窗对服务端生成型端点会造假 BLOCKED）
W_RANGE = 90
W_THRU = 60
W_FETCH = 180

RECHECK = "须以当场探针复核（≤24 h；M11-R56/R63：吞吐不可外推）"
PROV = "M12 执行节点 S1-GATE 当场探测（父进程硬回收 Popen+communicate(timeout)+kill）"
WL_SCORE = "若把 HTTP 200 + total_rows=0 读成「该过滤下无数据」而实际是 RMA 语法错误" \
           "（success:false）⇒ 白名单过滤结论完全反向（把「可达且可过滤」误判成「无数据」）"

BYID = {}


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- 传输层
def _hdr_dict(path):
    d = {}
    if not path or not os.path.exists(path):
        return d
    for line in open(path, encoding="utf-8", errors="replace"):
        if ":" in line and not line.startswith("HTTP/"):
            k, _, v = line.partition(":")
            d[k.strip().lower()] = v.strip()
    return d


def run_curl(url, timeout=W_COUNT, byte_range=None, out_path=None, hdr_path=None,
             fail_on_error=False, connect_timeout=8):
    """单次 curl；**父进程硬回收**。返回 dict(code, nbytes, seconds, speed, ttfb, exit, ...)。"""
    cmd = ["curl", "-sS", "-g", "-L", "--connect-timeout", str(connect_timeout),
           "--max-time", str(timeout),
           "-w", "%{http_code}|%{size_download}|%{content_type}|%{time_total}|%{speed_download}|"
                 "%{num_redirects}|%{time_starttransfer}|%{time_connect}"]
    if fail_on_error:
        cmd += ["--fail"]
    cmd += ["-o", out_path if out_path else os.devnull]
    if hdr_path:
        cmd += ["-D", hdr_path]
    if byte_range is not None:
        cmd += ["-r", "%d-%d" % byte_range]
    cmd.append(url)

    rec = dict(code="000", nbytes=0, seconds=0.0, speed=0.0, ttfb=0.0, connect=0.0,
               exit=0, err="", headers={}, hard_killed=False, wall=0.0, cmd=" ".join(cmd))
    t0 = time.time()
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        rec["exit"] = 125
        rec["err"] = "spawn failed: %s" % str(e)[:160]
        rec["wall"] = time.time() - t0
        return rec
    try:
        so, se = proc.communicate(timeout=timeout + 10)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            so, se = proc.communicate(timeout=10)
        except Exception:
            so, se = b"", b""
        rec["exit"] = 124
        rec["hard_killed"] = True
        rec["err"] = "hard kill by parent（--max-time 未生效，M11-L9.3a）"
        rec["wall"] = time.time() - t0
        rec["headers"] = _hdr_dict(hdr_path)
        return rec
    rec["wall"] = time.time() - t0
    rec["exit"] = proc.returncode
    parts = (so or b"").decode("utf-8", "replace").strip().split("|")
    if len(parts) >= 5:
        rec["code"] = parts[0]
        for idx, key in ((1, "nbytes"), (3, "seconds"), (4, "speed"),
                         (6, "ttfb"), (7, "connect")):
            try:
                val = float(parts[idx])
            except ValueError:
                val = 0.0
            rec[key] = int(val) if key == "nbytes" else val
    if se:
        rec["err"] = se.decode("utf-8", "replace")[:200]
    rec["headers"] = _hdr_dict(hdr_path)
    return rec


def _hdr(rec, key):
    return rec.get("headers", {}).get(key, "")


def curl_body(url, timeout=W_COUNT, out_path=None):
    """取一次**响应体**（探针口径：不用 `--fail`，错误体也要能分类）。"""
    cmd = ["curl", "-sS", "-g", "-L", "--connect-timeout", "8", "--max-time", str(timeout)]
    if out_path:
        cmd += ["-o", out_path]
    cmd.append(url)
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
            return b"", 124, "hard kill by parent"
        return so, proc.returncode, se.decode("utf-8", "replace")[:200]
    except Exception as e:
        return b"", 125, "spawn failed: %s" % str(e)[:160]


# ---------------------------------------------------------------- 输出器
class Sink(object):
    """主表**每 spec 一行（最新一次实测 ⇒ 重跑确定性）**；
    逐次探测的全量行同时追加到 `<path>.history.csv`（append-only 审计链，不覆盖历史）。"""

    def __init__(self, path, cols, title):
        self.path, self.cols, self.title = path, cols, title
        self.rows, self.history = [], []
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self.rows = [r for r in csv.DictReader(l for l in f if not l.startswith("#"))]

    def add(self, row):
        base = {c: row.get(c, "") for c in self.cols}
        self.history.append(dict(base))
        for i, r in enumerate(self.rows):
            if r.get("spec") == base.get("spec"):
                self.rows[i] = base
                return
        self.rows.append(base)

    def flush(self):
        with open(self.path, "w", encoding="utf-8", newline="") as f:
            f.write("# %s\n" % self.title)
            f.write("# 口径：**每 spec 一行 = 最新一次当场实测**（重跑幂等，同一 spec 覆盖）；"
                    "逐次探测的全量行见同名 `.history.csv`（append-only）；吞吐判决类行必带 "
                    "utc + bytes + seconds + bytes_per_s + recheck_policy（M11-R56/R63）\n")
            w = csv.DictWriter(f, fieldnames=self.cols)
            w.writeheader()
            for r in self.rows:
                w.writerow(r)
        if self.history:
            new = not os.path.exists(self.path + ".history.csv")
            with open(self.path + ".history.csv", "a", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.cols)
                if new:
                    f.write("# M12 S1-GATE 逐次探测全量审计链（append-only；与主表同列）\n")
                    w.writeheader()
                for r in self.history:
                    w.writerow(r)
        return len(self.rows)


REACH_COLS = ["utc", "spec", "label", "kind", "url", "http_code", "http_code_pass2",
              "curl_exit", "rma_success", "bytes", "content_type", "ttfb_s", "seconds",
              "speed_Bps", "total_rows", "verdict", "verdict_scope", "sha256_prefix",
              "evidence_level", "provenance", "recheck_policy", "wrong_usage_consequence",
              "missing_dimension", "note"]
THRU_COLS = ["utc", "spec", "object", "http_code", "bytes", "seconds", "bytes_per_s",
             "ttfb_s", "probe_or_fetch", "round", "median_of_rounds", "provenance",
             "recheck_policy", "note"]
INTEG_COLS = ["utc", "spec", "object_path", "bytes", "content_length", "size_match",
              "etag", "etag_md5_recomputed", "etag_match", "sha256", "sha_basis",
              "semantic_anchor", "semantic_ok", "verdict", "note"]
SYNTAX_COLS = ["utc", "spec", "form", "url", "http_code", "rma_success", "rma_msg",
               "total_rows", "bytes", "verdict", "note"]


# ---------------------------------------------------------------- 白名单（预注册确定性规则）
def load_graph1():
    with open(S2_STRUCT, encoding="utf-8") as f:
        d = json.load(f)
    rows = [x for x in d["msg"] if x.get("graph_id") == 1]
    rows.sort(key=lambda x: x["id"])
    return rows, d.get("total_rows")


def whitelist_rules(rows):
    """预注册规则：只依赖 S2 已落盘件 + `graph_id=1` + CCFv3 depth 阈值（**出数前定死**）。"""
    return {"W_DEPTH_LE%d" % md: [x["id"] for x in rows if x["depth"] <= md]
            for md in (2, 3, 5, 6)}


def write_whitelist_csv(rules, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("# M12 S1-GATE 预注册区域白名单（来源=data/m12_raw/s2/model_Structure_all.json，"
                "graph_id=1；规则=CCFv3 depth 阈值；**出数前定死**；由 "
                "tools/probe_m12_s1_gate.py 确定性重生成 ⇒ 重跑 sha256 应一致）\n")
        w = csv.writer(f)
        w.writerow(["whitelist_id", "structure_id", "acronym", "name", "depth",
                    "st_level", "parent_structure_id", "graph_id", "structure_id_path"])
        for name in sorted(rules):
            for i in rules[name]:
                x = BYID[i]
                w.writerow([name, x["id"], x["acronym"], x["name"], x["depth"],
                            x["st_level"], x["parent_structure_id"], x["graph_id"],
                            x["structure_id_path"]])
    return path


# ---------------------------------------------------------------- RMA 计数（含语法候选与 success 检查）
def count_url(model, criteria_forms, options="rma::options[count$eqtrue]"):
    return "%s/query.json?criteria=model::%s%s,%s" % (
        API, model, "".join("," + c if not c.startswith("rma::") else "," + c
                            for c in criteria_forms), options)


def rma_query(model, criteria_forms, options="rma::options[count$eqtrue]",
              spec="", timeout=W_COUNT, save=None, hdr_name=None):
    """一次 RMA 查询（**单次 curl 同时落盘响应体 + 状态码**，省一半往返）：
    解析 `success` / `total_rows` / `msg` 错误文本。**HTTP 200 ≠ 查询成功**。"""
    url = count_url(model, criteria_forms, options)
    hdr = os.path.join(RAW, hdr_name or "_hdr_%s.txt" % (spec or "x"))
    p = save or os.path.join(RAW, "%s_body.json" % (spec or "x"))
    if os.path.exists(p):
        os.remove(p)
    rec = run_curl(url, timeout=timeout, out_path=p, hdr_path=hdr)
    out = dict(spec=spec, url=url, code=rec["code"], exit=rec["exit"],
               nbytes=rec["nbytes"], ttfb=rec["ttfb"], seconds=rec["seconds"],
               speed=rec["speed"], content_type=_hdr(rec, "content-type"),
               success=None, total_rows=None, msg="", raw_bytes=None,
               hard_killed=rec["hard_killed"], rec=rec)
    if rec["code"] == "200" and os.path.exists(p):
        body = open(p, "rb").read()
        out["raw_bytes"] = len(body)
        try:
            j = json.loads(body.decode("utf-8", "replace"))
            out["success"] = j.get("success")
            out["total_rows"] = j.get("total_rows")
            if isinstance(j.get("msg"), str):
                out["msg"] = j["msg"]
        except Exception as e:
            out["msg"] = "PARSE_FAIL: %s" % str(e)[:100]
    return out


def probe_count_twice(spec, model, criteria_forms, save=None, timeout=W_COUNT):
    """两次探测 + 码不一致时第三次多数裁决；返回 (chosen, code2, extra_note)。"""
    a = rma_query(model, criteria_forms, spec=spec, timeout=timeout, save=save)
    b = rma_query(model, criteria_forms, spec=spec, timeout=timeout, save=save)
    note = ""
    chosen = a
    if a["code"] != b["code"]:
        c = rma_query(model, criteria_forms, spec=spec, timeout=timeout, save=save)
        from collections import Counter
        maj = Counter([a["code"], b["code"], c["code"]]).most_common(1)[0][0]
        for x in (a, b, c):
            if x["code"] == maj:
                chosen = x
                break
        note = "两次探测码不一致（%s/%s）⇒ 第三次多数裁决 ⇒ %s" % (a["code"], b["code"], maj)
    else:
        note = "两次探测码一致（%s）" % a["code"]
    if chosen["success"] is False:
        note += "｜🔴 **HTTP %s 但 rma success=false**（RMA 语法/过滤错误）msg=%s" % (
            chosen["code"], chosen["msg"][:120])
    elif chosen["success"] is True:
        note += "｜rma success=true, total_rows=%s" % chosen["total_rows"]
    return chosen, b["code"], note


def reach_row(spec, label, kind, o, code2, note, verdict, scope, rows, wrong, missing,
              evidence="measured"):
    return dict(utc=utcnow(), spec=spec, label=label, kind=kind, url=o["url"],
                http_code=o["code"], http_code_pass2=code2, curl_exit=o["exit"],
                rma_success=o["success"], bytes=o["nbytes"],
                content_type=o["content_type"], ttfb_s="%.3f" % o["ttfb"],
                seconds="%.3f" % o["seconds"], speed_Bps="%.1f" % o["speed"],
                total_rows=rows if rows is not None else "",
                verdict=verdict, verdict_scope=scope,
                sha256_prefix=(hashlib.sha256(open(
                    os.path.join(RAW, "%s_body.json" % spec), "rb").read()).hexdigest()[:16]
                    if os.path.exists(os.path.join(RAW, "%s_body.json" % spec)) else ""),
                evidence_level=evidence, provenance=PROV, recheck_policy=RECHECK,
                wrong_usage_consequence=wrong, missing_dimension=missing, note=note)


# ---------------------------------------------------------------- Phase A
def phase_A(reach, syntax, counts):
    out = {}
    # A0 单页：字段名读回 + 取一个真实 section_data_set_id
    o, code2, note = probe_count_twice(
        "A0", "ProjectionStructureUnionize", [], save=os.path.join(RAW, "A0_body.json"))
    fields, sds = [], None
    p0 = os.path.join(RAW, "A0_body.json")
    if o["code"] == "200" and os.path.exists(p0):
        try:
            j = json.load(open(p0, encoding="utf-8"))
            if j.get("msg") and isinstance(j["msg"], list):
                fields = sorted(j["msg"][0].keys())
                sds = str(j["msg"][0]["section_data_set_id"])
        except Exception as e:
            note += "｜parse_fail:%s" % str(e)[:80]
    counts["_sample_sds"] = sds or "0"
    out["A0_fields"] = fields
    out["A0_sample_sds"] = sds
    reach.add(reach_row(
        "A0", "PSU 单页读回（字段名以实测为准 + 取一个真实 section_data_set_id 供 A8）",
        "api_page", o, code2, note, "REACHABLE" if fields else "BLOCKED",
        "本条作用域仅限本行列出的这一维（字段清单 + 一个真实 sds id）",
        o["total_rows"],
        "按常识臆造字段名（如 `injection_structure`）⇒ 静默取不到列"
        "（B1 §1 实测：IBL 的 `feedback`/`intervals` 真名是 `feedbackType`/`intervals_0/1`）",
        "无", ))

    # A1 S2 锚
    o, code2, note = probe_count_twice("A1", "Structure", [], save=os.path.join(RAW, "A1_body.json"))
    out["A1_structure_count"] = o["total_rows"]
    reach.add(reach_row("A1", "S2 结构表计数锚（应 == 已落盘件 total_rows=13,709）", "api_count",
                        o, code2, note, "REACHABLE" if o["success"] else "NEEDS_VERIFY",
                        "本条作用域仅限本行列出的这一维（Structure 全表行数）", o["total_rows"],
                        "S2 口径错误 ⇒ 区域 ID 串台（判据行 M12-S2-02）", "无"))

    # A2 PSU 全表计数
    o, code2, note = probe_count_twice("A2", "ProjectionStructureUnionize", [],
                                       save=os.path.join(RAW, "A2_body.json"))
    out["A2_psu_full_count"] = o["total_rows"]
    reach.add(reach_row(
        "A2", "PSU 全表行数（T1）⇒ §2.2 第 460 行主假设 `行数 ≥ 10⁶` 的第一半", "api_count",
        o, code2, note, "REACHABLE" if o["success"] else "NEEDS_VERIFY",
        "本条作用域仅限本行列出的这一维（PSU 全表行数）", o["total_rows"],
        "把它当「区域×区域矩阵行数」会错几个量级：该表是「实验 × 结构」长表；按行数直接建 "
        "500×500 矩阵 ⇒ 维度与语义双错", "无"))

    # A3..A6 目标结构白名单过滤计数（`$in` 无括号语法）+ 单调性
    rules = counts["_rules"]
    wl = {}
    for i, name in enumerate(sorted(rules), start=3):
        ids = rules[name]
        crit = ["rma::criteria[structure_id$in%s]" % ",".join(str(x) for x in ids)]
        o, code2, note = probe_count_twice("A%d" % i, "ProjectionStructureUnionize", crit,
                                           save=os.path.join(RAW, "A%d_body.json" % i))
        wl[name] = o["total_rows"]
        reach.add(reach_row(
            "A%d" % i, "PSU 服务端白名单过滤计数 · %s（n_ids=%d）⇒ §2.2 第 460 行"
            "「白名单过滤后行数 ≤ 10⁵」判据的输入" % (name, len(ids)), "api_count",
            o, code2, note, "REACHABLE" if o["success"] else "NEEDS_VERIFY",
            "该白名单形态下的行数（**不含** is_injection 语义限定）", o["total_rows"], WL_SCORE,
            "缺「与逐 id 计数之和的交叉校验」（见 A7，仅最小白名单做了）"))
        time.sleep(0.3)
    out["A_whitelist_target_counts"] = wl

    # A7 逐 id 独立交叉校验（最小白名单）
    base = "W_DEPTH_LE2"
    per_id = {}
    for sid in rules[base]:
        o, _, _ = probe_count_twice("A7_%d" % sid, "ProjectionStructureUnionize",
                                    ["rma::criteria[structure_id$eq%d]" % sid],
                                    save=os.path.join(RAW, "A7_%d_body.json" % sid))
        per_id[str(sid)] = o["total_rows"]
        time.sleep(0.2)
    ssum = None if any(v is None for v in per_id.values()) else sum(per_id.values())
    out["A7_per_id_counts"], out["A7_sum"] = per_id, ssum
    out["A7_in_count"] = wl.get(base)
    ok = (ssum is not None and ssum == wl.get(base))
    reach.add(dict(utc=utcnow(), spec="A7",
                   label="PSU 逐 id 计数之和 vs `$in<W_DEPTH_LE2>` 服务端计数（白名单过滤语义的"
                         "独立性交叉校验；n_ids=%d）" % len(rules[base]),
                   kind="crosscheck",
                   url="(24 次 structure_id$eq<sid> 单独计数；明细见 "
                       "data/m12_s1_gate_counts.json::A7_per_id_counts)",
                   http_code="200", http_code_pass2="200", curl_exit=0, rma_success=True,
                   bytes="", content_type="application/json", ttfb_s="", seconds="",
                   speed_Bps="", total_rows="Σ=%s vs $in=%s" % (ssum, wl.get(base)),
                   verdict="REACHABLE" if ok else "NEEDS_VERIFY",
                   verdict_scope="白名单过滤语义是否与逐 id 计数等价（本行作用域）",
                   sha256_prefix="", evidence_level="measured", provenance=PROV,
                   recheck_policy=RECHECK,
                   wrong_usage_consequence="若 Σ(逐 id) != count($in) ⇒ `$in` 过滤语义不成立 ⇒ "
                   "所有白名单过滤计数与其 ETA 全部作废（invalid，不是 pass/fail）",
                   missing_dimension="无", note="discipline 7（三路独立估计，R-6 同族）"))

    # A8 行/实验（实验维）
    o, code2, note = probe_count_twice(
        "A8", "ProjectionStructureUnionize",
        ["rma::criteria[section_data_set_id$eq%s]" % counts["_sample_sds"]],
        save=os.path.join(RAW, "A8_body.json"))
    out["A8_rows_per_experiment"] = o["total_rows"]
    reach.add(reach_row("A8", "PSU 抽样实验行数（section_data_set_id$eq%s）⇒ 行/实验 维"
                        % counts["_sample_sds"], "api_count", o, code2, note,
                        "REACHABLE" if o["success"] else "NEEDS_VERIFY",
                        "本条作用域仅限本行列出的这一维（单个实验的行数；**不得**当全表均值）",
                        o["total_rows"],
                        "把「单实验行数 × 实验数」当全表行数 ⇒ 装配时间/内存估算错量级（T6 输入）",
                        "未做多实验抽样 ⇒ 「行/实验」的离散度未测"))

    # A9 SectionDataSet 全表
    o, code2, note = probe_count_twice("A9", "SectionDataSet", [],
                                       save=os.path.join(RAW, "A9_body.json"))
    out["A9_sectiondataset_count"] = o["total_rows"]
    reach.add(reach_row("A9", "SectionDataSet 计数（实验池锚，清单第 457 行的变体兜底路径）",
                        "api_count", o, code2, note,
                        "REACHABLE" if o["success"] else "NEEDS_VERIFY",
                        "本条作用域仅限本行列出的这一维（全表；含非投射产品）",
                        o["total_rows"],
                        "把 SectionDataSet 全表当『投射实验数』⇒ 分母虚高（未测 ≠ 未命中 的计数版）",
                        "缺「按产品/product 过滤后的投射实验数」一维"))

    # A10 is_injection 维度（源的实验数锚）
    o, code2, note = probe_count_twice("A10", "ProjectionStructureUnionize",
                                       ["rma::criteria[is_injection$eqtrue]"],
                                       save=os.path.join(RAW, "A10_body.json"))
    out["A10_is_injection_true"] = o["total_rows"]
    reach.add(reach_row("A10", "PSU `is_injection$eqtrue` 计数 ⇒ **源（注射）维**的实验数锚"
                         "（R×R 装配的行维输入）", "api_count", o, code2, note,
                         "REACHABLE" if o["success"] else "NEEDS_VERIFY",
                         "本条作用域仅限本行列出的这一维（源侧行数）", o["total_rows"],
                         "不区分 is_injection ⇒ 把目标行当源行 ⇒ 源/目标维互换，装配出的矩阵转置",
                         "无（本行作用域内）"))

    # A11/A12 源×白名单 / 目标×白名单（多 facet 语法候选）
    multi = [("A11", ["rma::criteria[is_injection$eqfalse]",
                      "rma::criteria[structure_id$in%s]" % ",".join(
                          str(x) for x in rules["W_DEPTH_LE6"])],
              "目标行 ∧ 白名单(depth≤6, n=%d)" % len(rules["W_DEPTH_LE6"])),
             ("A12", ["rma::criteria[is_injection$eqtrue]",
                      "rma::criteria[structure_id$in%s]" % ",".join(
                          str(x) for x in rules["W_DEPTH_LE3"])],
              "源行 ∧ 族白名单(depth≤3, n=%d)" % len(rules["W_DEPTH_LE3"]))]
    for spec, crit, desc in multi:
        cands = [(spec, crit),
                 (spec + "b", ["rma::criteria[%s]" % ",".join(
                     [c[len("rma::criteria["):-1] for c in crit])]),
                 (spec + "c", ["rma::criteria[%s]" % ",".join(
                     [c[len("rma::criteria["):-1].replace("$eqfalse", "$ne1")
                      .replace("$eqtrue", "$eq1") for c in crit])])]
        chosen = None
        for cs, cc in cands:
            o, code2, note = probe_count_twice(cs, "ProjectionStructureUnionize", cc,
                                               save=os.path.join(RAW, "%s_body.json" % cs))
            syntax.add(dict(utc=utcnow(), spec=cs, form=desc + "｜form=%s" % (
                "多 facet 分离" if cs == spec else ("同 facet 逗号连接" if cs == spec + "b"
                                                    else "$eq1 写法")),
                url=o["url"], http_code=o["code"], rma_success=o["success"],
                rma_msg=str(o["msg"])[:160], total_rows=o["total_rows"], bytes=o["nbytes"],
                verdict="OK" if o["success"] else "SYNTAX_ERROR",
                note=note))
            if o["success"] and chosen is None:
                chosen = (o, code2, note)
            time.sleep(0.2)
        if chosen:
            o, code2, note = chosen
            out["A_%s" % spec] = o["total_rows"]
            reach.add(reach_row(spec, "PSU %s ⇒ R×R 装配的行/列维输入" % desc, "api_count",
                                o, code2, note, "REACHABLE", "本条作用域仅限本行列出的这一维",
                                o["total_rows"], WL_SCORE, "无"))
        else:
            reach.add(dict(utc=utcnow(), spec=spec,
                           label="PSU %s（**全部语法候选均 success=false**）" % desc,
                           kind="api_count", url="见 data/m12_s1_gate_syntax.csv",
                           http_code="200", http_code_pass2="", curl_exit=0, rma_success=False,
                           bytes="", content_type="", ttfb_s="", seconds="", speed_Bps="",
                           total_rows="", verdict="NEEDS_VERIFY",
                           verdict_scope="本条作用域仅限本行列出的这一维",
                           sha256_prefix="", evidence_level="measured", provenance=PROV,
                           recheck_policy=RECHECK, wrong_usage_consequence=WL_SCORE,
                           missing_dimension="多 facet 过滤语法未实测成功 ⇒ 该维（源白名单过滤）"
                           "**未测**（≠ 无数据）", note="见 m12_s1_gate_syntax.csv"))

    # A14 语法变体对照（$in 括号 vs 无括号）+ B1 S1-03 空信封复现
    for spec, form, crit in (("A14", "$in[...] 带方括号（**清单第 457 行模板的自然写法**）",
                              ["rma::criteria[structure_id$in[16997,1,2]]"]),
                             ("A15", "$in... 逗号列表（无括号）",
                              ["rma::criteria[structure_id$in16997,1,2]"]),
                             ("A16", "-- 复现 B1 表 `m12_source_reachability.csv::S1-03` 的原 URL "
                              "（B1 记 bytes=85 / verdict=REACHABLE）",
                              ["rma::criteria[structure_id$eq16997]"])):
        o, code2, note = probe_count_twice(spec, "ProjectionStructureUnionize", crit,
                                           save=os.path.join(RAW, "%s_body.json" % spec))
        syntax.add(dict(utc=utcnow(), spec=spec, form=form, url=o["url"],
                        http_code=o["code"], rma_success=o["success"],
                        rma_msg=str(o["msg"])[:160], total_rows=o["total_rows"],
                        bytes=o["nbytes"],
                        verdict="OK" if o["success"] else "SYNTAX_ERROR", note=note))
        counts["_syntax_%s" % spec] = dict(code=o["code"], success=o["success"],
                                           total_rows=o["total_rows"], bytes=o["nbytes"],
                                           msg=str(o["msg"])[:160])
    return out


# ---------------------------------------------------------------- Phase B（T2 对象级 + Range 升级规则）
def object_probe(spec, label, criteria_forms, reach, counts, allow_range=True,
                 window=W_RANGE, wrong=""):
    """对一个 RMA 查询的 **CSV 对象**做对象级取证：`-r 0-1023` 拿 Content-Range 分母（对象总长）。

    ★ **Range 升级规则（预注册，事中不得择口径；沿 B1 L3/L5 的 P-a…P-d 范式）**：
      R-b：若 `-r 0-1023` 返回 **206 + content-range** ⇒ 得**对象总长**（零成本，不下载整件）；
      R-c：若返回 **200**（服务端忽略 Range）⇒ 只接受 `content-length` 作为对象总长
           （且此时 `-r` 探针**已经**下了 1 KB 便结束，安全）；
      R-d：若**无 CL 且无 content-range**（chunked 生成型）⇒ 判 `REACHABLE_NO_DATA` 的
           "对象总长未取到"，**不得**据此判 TOO_SLOW（M12-M-01），改走"整件取数即测量"。
    """
    url = ("%s/query.csv?criteria=model::ProjectionStructureUnionize%s,"
           "rma::options[num_rows$eqall]" % (API, "".join("," + c for c in criteria_forms)))
    hdr = os.path.join(RAW, "_hdr_%s.txt" % spec)
    p = os.path.join(RAW, "%s_head.bin" % spec)
    rec = run_curl(url, timeout=window, byte_range=(0, 1023) if allow_range else None,
                   hdr_path=hdr, out_path=p)
    cl = _hdr(rec, "content-length")
    cr = _hdr(rec, "content-range")
    ct = _hdr(rec, "content-type")
    total, src = None, "NA"
    if cr and "/" in cr:
        try:
            total, src = int(cr.rsplit("/", 1)[1]), "content-range分母(206)"
        except ValueError:
            total = None
    if total is None and cl and rec["code"] == "200":
        try:
            total, src = int(cl), "content-length(200)"
        except ValueError:
            total = None
    if total is None and allow_range and rec["code"] == "200" and rec["nbytes"] > 0:
        # ★ 本轮实测：该端点**忽略 Range**（返回 200、无 accept-ranges）⇒ 请求 `-r 0-1023`
        #   实际拿到的是**完整对象** ⇒ nbytes 即对象总长（**「整件取数即测量」路径**，
        #   与 M12-M-01 / B1 L3 的 P-d 规则同族）。该判定的**独立性佐证**见断言 Z9
        #   （与本节点另一次无 Range 整件取数的 sha256 逐字节一致）。
        total, src = rec["nbytes"], "nbytes(200; Range 被服务端忽略 ⇒ 整件已落盘)"
    out = dict(code=rec["code"], content_length=cl, content_range=cr, content_type=ct,
               total_bytes=total, total_src=src, ttfb=rec["ttfb"], nbytes=rec["nbytes"],
               accept_ranges=_hdr(rec, "accept-ranges"), etag=_hdr(rec, "etag"),
               seconds=rec["seconds"], speed=rec["speed"], hard_killed=rec["hard_killed"],
               err=rec["err"], wall=rec["wall"])
    counts["_obj_%s" % spec] = out
    verdict = "REACHABLE" if total else "REACHABLE_NO_DATA"
    note = ("Range 规则 %s｜content-length=%s｜content-range=%s｜accept-ranges=%s｜"
            "curl_exit=%s%s") % ("R-b(206)" if rec["code"] == "206" else
                                 ("R-c(200)" if rec["code"] == "200" else "R-d"),
                                 cl, cr, out["accept_ranges"], rec["exit"],
                                 ("｜**父进程硬回收**" if rec["hard_killed"] else ""))
    if total is None:
        note += "｜🔴 对象总长**未取到** ⇒ 按 R-d 判 REACHABLE_NO_DATA，**不得**判 TOO_SLOW"
    reach.add(dict(utc=utcnow(), spec=spec, label=label, kind="api_range_object", url=url,
                   http_code=rec["code"], http_code_pass2="", curl_exit=rec["exit"],
                   rma_success="", bytes=rec["nbytes"], content_type=ct,
                   ttfb_s="%.3f" % rec["ttfb"], seconds="%.3f" % rec["seconds"],
                   speed_Bps="%.1f" % rec["speed"],
                   total_rows="（行数见同名 A 行）", verdict=verdict,
                   verdict_scope="本条作用域仅限本行列出的这一维（该过滤下的 CSV 对象总长）",
                   sha256_prefix=sha256_file(p)[:16] if os.path.exists(p) else "",
                   evidence_level="measured", provenance=PROV, recheck_policy=RECHECK,
                   wrong_usage_consequence=wrong or
                   "**门① 基准绝不能用 206 续传响应的 Content-Length**（那是剩余段长度；"
                   "B1 L20 实测的假失败）⇒ 必须用 Content-Range 的分母 / 对象总长",
                   missing_dimension="无" if total else
                   "对象总长（chunked 生成型 ⇒ 须走整件取数即测量）", note=note))
    return out


def phase_B(reach, counts):
    out = {}
    # 把上一轮"整件请求挂死"的记录显式改名为 `_obj_B3_prior_freeze`（不留在当前口径的键名下，
    # 以免被误读成本轮的有效实测；历史行仍完整保留在主表的 `.history.csv` 里）
    if counts.get("_obj_B3", {}).get("code") == "000":
        counts["_obj_B3_prior_freeze"] = counts.pop("_obj_B3")
    b1 = object_probe("B1", "S1 单实验整件 CSV（section_data_set_id$eq%s ⇒ 与 A8 的 2,574 行为同一"
                      "语义锚；用来验 Range 支持 + 得到 CSV 对象的**真实** bytes/row）"
                      % counts["_sample_sds"],
                      ["rma::criteria[section_data_set_id$eq%s]" % counts["_sample_sds"]],
                      reach, counts,
                      wrong="用 JSON 端点的 bytes/row 外推 CSV 对象 ⇒ 高估 2–3×（CSV 无键名）")
    rng_ok = (b1["code"] == "206")
    counts["B1_range_supported"] = rng_ok
    b2 = object_probe("B2", "S1 源行子集整件 CSV（is_injection$eqtrue；342,475 行）⇒ 整件请求的"
                      "**服务端生成上限**实测", ["rma::criteria[is_injection$eqtrue]"],
                      reach, counts, allow_range=rng_ok,
                      wrong="把整件 504 读成「无数据」或「链路慢」⇒ 假 unavailable（504 的真实语义是"
                            "**服务端在 60 s 生成上限内产不出该对象**）")
    # ★ B3（396,424 行，≈54.7 MB 整件）：**本轮不重试**，登记为"未测 + 已有更强证据"。
    #   依据：① B2（更小的对象，342,475 行）已实测 HTTP 504 @ TTFB 60.60 s ⇒ 整件路径**已证不行**；
    #        ② 上一轮 B3 的整件请求**挂死 936.2 s / 0 字节 / curl exit 28**（父进程 communicate
    #           硬回收也未在 190 s 触发）⇒ 与 B1 L19/L24 记录的**整机冻结**同型；
    #        ③ 分页路径（Phase F）已实测可用 ⇒ 整件请求对本节点**已无必要**。
    prior = counts.get("_obj_B3_prior_freeze") or {}
    reach.add(dict(utc=utcnow(), spec="B3",
                   label="S1 最小族白名单整件 CSV 对象总长（396,424 行）——**登记为未测**，"
                         "不重试（B2 已实测整件 504；且上一轮该请求实测 936.2 s 冻结型挂死）",
                   kind="api_range_object",
                   url="%s/query.csv?criteria=model::ProjectionStructureUnionize,"
                       "rma::criteria[structure_id$in<W_DEPTH_LE2>],"
                       "rma::options[num_rows$eqall]" % API,
                   http_code="NA", http_code_pass2="", curl_exit="", rma_success="",
                   bytes="", content_type="", ttfb_s="", seconds="", speed_Bps="",
                   total_rows="396,424（A3 实测）", verdict="REACHABLE_NO_DATA",
                   verdict_scope="本条作用域仅限本行列出的这一维",
                   sha256_prefix="", evidence_level="measured", provenance=PROV,
                   recheck_policy=RECHECK,
                   wrong_usage_consequence="把「未测」写成「已测不可行」⇒ 违『未测 ≠ 未命中』",
                   missing_dimension="**整件对象总长：未测**（整件路径已被 B2 的 504 与分页路径取代；"
                   "总字节可由 396,424 行 × 137.895 B/row **估计**，估计不得当实测）",
                   note="不重试理由见工具 `phase_B` 注释与 "
                        "`data/m12_s1_gate_reachability.csv.history.csv` 中上一轮 B3 行"
                        "（实测 %s s / %s 字节 / exit=%s）"
                   % (prior.get("seconds"), prior.get("nbytes"), prior.get("rec", {}).get("exit"))))
    # B4 全表对象总长求值的**替代口径**（分页）（不发起整件请求）
    reach.add(dict(utc=utcnow(), spec="B4",
                   label="S1 全表整件 CSV 对象总长（16,357,801 行）——**未测**（整件路径已证不行；"
                         "改由分页路径取证）", kind="api_range_object",
                   url="(不发起；依据 B2 的 504 与 Phase F 的分页实测)",
                   http_code="NA", http_code_pass2="", curl_exit="", rma_success="",
                   bytes="", content_type="", ttfb_s="", seconds="", speed_Bps="",
                   total_rows="16,357,801（A2 实测）", verdict="REACHABLE_NO_DATA",
                   verdict_scope="本条作用域仅限本行列出的这一维",
                   sha256_prefix="", evidence_level="measured", provenance=PROV,
                   recheck_policy=RECHECK,
                   wrong_usage_consequence="把「未测」写成「已测不可行」⇒ 违『未测 ≠ 未命中』；"
                   "另：把全表对象当 M12 实际取数体量 ⇒ 高估（M12 只需白名单/实验子集）",
                   missing_dimension="**全表 CSV 对象总长：未测**（按 16,357,801 行 × 137.895 B/row "
                   "≈ 2.26 GB 可**估计**，但估计不得当实测）",
                   note="替代口径 = Phase F 的服务端分页（已实测可用）"))
    counts["B_objects"] = {k: v for k, v in counts.items() if k.startswith("_obj_")}
    counts["B_bytes_per_row_csv"] = (
        b1["total_bytes"] / float(counts["A8_rows_per_experiment"])
        if (b1.get("total_bytes") and counts.get("A8_rows_per_experiment")) else None)
    counts["B_estimate"] = dict(
        bytes_per_row_csv=counts["B_bytes_per_row_csv"],
        full_table_csv_bytes_estimated=(
            int(counts["B_bytes_per_row_csv"] * counts["A2_psu_full_count"])
            if counts.get("B_bytes_per_row_csv") else None),
        W_DEPTH_LE2_csv_bytes_estimated=(
            int(counts["B_bytes_per_row_csv"] * counts["A_whitelist_target_counts"]["W_DEPTH_LE2"])
            if counts.get("B_bytes_per_row_csv") else None),
        note="**估计值**（单价 × 行数），不得当实测；实测口径为 B1 的整件字节与其行数之比")
    return out
    # bytes/row（用 B1 实测对象总长 / A8 实测行数）
    br = None
    if b1["total_bytes"] and counts.get("A8_rows_per_experiment"):
        br = b1["total_bytes"] / float(counts["A8_rows_per_experiment"])
    counts["B_bytes_per_row_csv"] = br
    return out


# ---------------------------------------------------------------- Phase C（T3 吞吐：多轮取中位）
def phase_C(thru, counts, rounds=3):
    """3 轮吞吐（中位）。

    ★ **口径修正（本节点当场实测后修正，并留痕）**：清单第 579–582 行的 T2/T3 模板用
    `-r 0-1048575`（1 MiB Range）。实测该端点 **忽略 Range**（返回 200 而非 206，
    无 `accept-ranges`）⇒ 1 MiB Range 探针在 S1 上**口径不适用**；且大结果集整件请求会
    **HTTP 504（服务端 60 s 生成超时）**。⇒ 正确的吞吐量 = **服务端分页对象**（
    `num_rows$eqN][start_row$eqM]`）上的端到端 `bytes/秒`，并按"页数 × 单页墙钟"给 ETA。
    """
    ids = counts["_rules"]["W_DEPTH_LE2"]
    crit = "rma::criteria[structure_id$in%s]" % ",".join(str(x) for x in ids)
    speeds = []
    for i in range(1, rounds + 1):
        pr = 10000
        url = ("%s/query.csv?criteria=model::ProjectionStructureUnionize,%s,"
               "rma::options[num_rows$eq%d][start_row$eq%d]" % (API, crit, pr, (i - 1) * pr))
        p = os.path.join(RAW, "C_round%d.bin" % i)
        rec = run_curl(url, timeout=W_THRU, out_path=p)
        rows, cols, rej = read_psu_csv(p)
        bps = rec["speed"]
        if not rows:
            tag = "口径不适用/无效页（%s）" % (rej or "空页")
            note = ("本页 **未取得有效数据**（%s）⇒ 其 `bytes_per_s` **不得**用作吞吐量"
                    "（M12-M-01 同族：0 或极小 B/s 的根因不是链路）" % (rej or rec["err"][:60]))
        else:
            tag = "fetch_page(num_rows$eq%d)" % pr
            note = ("服务端分页对象（%d 行；http=%s）⇒ 该端点的**唯一**有效吞吐口径"
                    "（Range 被忽略、整件 504）" % (len(rows), rec["code"]))
            speeds.append(bps)
        thru.add(dict(utc=utcnow(), spec="C%d" % i,
                      object="PSU CSV 服务端分页（structure_id$in<W_DEPTH_LE2>，"
                             "num_rows$eq%d][start_row$eq%d）" % (pr, (i - 1) * pr),
                      http_code=rec["code"], bytes=rec["nbytes"],
                      seconds="%.3f" % rec["seconds"], bytes_per_s="%.1f" % bps,
                      ttfb_s="%.3f" % rec["ttfb"], probe_or_fetch=tag, round=i,
                      median_of_rounds="", provenance=PROV, recheck_policy=RECHECK, note=note))
        time.sleep(0.5)
    med = sorted(speeds)[len(speeds) // 2] if speeds else None
    counts["C_speeds_Bps"] = speeds
    counts["C_median_Bps"] = med
    counts["C_median"] = med
    thru.add(dict(utc=utcnow(), spec="C-median", object="PSU CSV 服务端分页（同 C1–C3）",
                  http_code="200", bytes="", seconds="",
                  bytes_per_s=("%.1f" % med) if med else "", ttfb_s="",
                  probe_or_fetch="fetch_page", round="median-of-%d" % len(speeds),
                  median_of_rounds=("%.1f" % med) if med else "", provenance=PROV,
                  recheck_policy=RECHECK,
                  note="有效轮 %d 轮中位 = %s B/s（逐轮 %s）⇒ §2.2 第 460 行「实测吞吐 ≥ 200 KB/s」"
                       "判据的输入" % (len(speeds), ("%.1f" % med) if med else "NA",
                                       ["%.0f" % s for s in speeds])))
    return med


# ---------------------------------------------------------------- Phase D（T4 ETA + 四态裁决）
def phase_D(counts, reach):
    """把清单第 434 行（T4 门）与第 460 行（四态条件）逐条**数值化**，给出可判定的裁决。"""
    sp = counts.get("C_median") or 0.0
    br = counts.get("B_bytes_per_row_csv")
    full_rows = counts.get("A2_psu_full_count")
    wl_rows = counts.get("A_whitelist_target_counts", {})
    obj = counts.get("_obj_B3", {}).get("total_bytes")
    obj_b1 = counts.get("_obj_B1", {}).get("total_bytes")
    rows_b1 = counts.get("A8_rows_per_experiment")
    est = {}
    if br and full_rows:
        est["full_table_csv_bytes"] = int(br * full_rows)
    for k, v in wl_rows.items():
        if br and v:
            est["%s_csv_bytes" % k] = int(br * v)
    d = dict(
        utc=utcnow(),
        speed_Bps_median=sp,
        bytes_per_row_csv=br,
        obj_B1_bytes=obj_b1, obj_B1_rows=rows_b1,
        obj_B3_bytes=obj,
        rows=dict(full=full_rows, **{k: v for k, v in wl_rows.items()},
                  source_rows=counts.get("A10_is_injection_true"),
                  rows_per_experiment=rows_b1),
        estimated_bytes=est,
    )
    if sp and obj:
        d["ETA_B3_s"] = obj / sp
        d["ETA_B3_min"] = obj / sp / 60.0
    if sp and br and full_rows:
        d["ETA_full_s"] = br * full_rows / sp
        d["ETA_full_h"] = br * full_rows / sp / 3600.0
    if sp and br and rows_b1:
        d["ETA_1experiment_s"] = br * rows_b1 / sp
    # T4 门（第 434 行）：ETA_对象 > 30 min 或 ETA_总计 > 4 h ⇒ unavailable（吞吐限制）
    # ★ 口径：S1 的端点忽略 Range 且大结果集 504 ⇒ 「对象字节 / 链路吞吐」**不适用**；
    #    正确口径 = **页数 × 实测单页墙钟**（Phase F 实测）。下面两式并列，以分页口径为准。
    maxpr = counts.get("F_max_page_rows") or None
    pw = counts.get("F_page_wall_s")
    pw_max = max(pw) if pw else None
    paged = {}
    if maxpr and pw_max:
        for k, v in sorted(wl_rows.items()):
            pages = -(-v // maxpr)
            paged[k] = dict(rows=v, pages=pages, page_wall_s=pw_max,
                            ETA_s=pages * pw_max, ETA_min=pages * pw_max / 60.0,
                            T4_pass=pages * pw_max / 60.0 <= 30.0,
                            needs_managed_job=(30.0 < pages * pw_max / 60.0 <= 240.0))
    d["paged_ETA"] = paged
    d["paged_ETA_basis"] = counts.get("F_ETA_basis")
    gate = []
    if "W_DEPTH_LE2" in paged:
        gate.append(dict(
            rule="§2.3 T4 行 434：ETA_对象 > 30 min ⇒ unavailable（吞吐限制）",
            obj="最小族白名单对象（W_DEPTH_LE2，396,424 行；**分页口径**）",
            ETA_min=paged["W_DEPTH_LE2"]["ETA_min"], limit_min=30.0,
            pass_=paged["W_DEPTH_LE2"]["T4_pass"]))
    gate.append(dict(
        rule="§2.3 T4 行 434：ETA_总计 > 4 h ⇒ unavailable（吞吐限制）",
        obj="全表（**估计** 2.26 GB @ 实测中位吞吐）", ETA_h=d.get("ETA_full_h"),
        limit_h=4.0, pass_=None if "ETA_full_h" not in d else d["ETA_full_h"] <= 4.0))
    if "W_DEPTH_LE6" in paged:
        gate.append(dict(
            rule="§2.3 T4 行 434 的**档位依赖**：586 区白名单对象（8,162,198 行）分页 ETA",
            obj="W_DEPTH_LE6（586 区）", ETA_min=paged["W_DEPTH_LE6"]["ETA_min"],
            limit_min=30.0, pass_=paged["W_DEPTH_LE6"]["T4_pass"],
            note=">30 min ⇒ 按第 434 行需**主 agent 托管作业 + 断点续传**（不是 unavailable）"))
    # 第 460 行：pass ⇐ 白名单过滤后行数 ≤ 1e5 或 实测吞吐 ≥ 200 KB/s
    c_rows = [("白名单过滤后行数 ≤ 10^5", (wl_rows.get("W_DEPTH_LE2") or 0) <= 1e5,
               wl_rows.get("W_DEPTH_LE2")),
              ("实测吞吐 ≥ 200 KB/s", sp >= 204800.0, sp)]
    d["clause_460"] = [dict(cond=a, satisfied=bool(b), measured=c) for a, b, c in c_rows]
    # 第 460 行 unavailable ⇐ 行数 ≥ 10^6 且 吞吐 ≤ 16 KB/s
    unav = [(full_rows or 0) >= 1e6, sp <= 16384.0]
    d["clause_460_unavailable"] = dict(rows_ge_1e6=bool(unav[0]),
                                       speed_le_16KBps=bool(unav[1]),
                                       both=bool(unav[0] and unav[1]),
                                       measured_rows=full_rows, measured_speed=sp)
    d["T4_gate"] = gate
    # 四态裁决（预注册：两个分支都写，取触发者）
    if d["clause_460_unavailable"]["both"]:
        verdict, why = "unavailable（吞吐限制）", "第 460 行 unavailable 条件双满足"
    elif any(x["satisfied"] for x in d["clause_460"]):
        verdict, why = "pass", "第 460 行 pass 条件至少一支满足：" + \
            "; ".join("%s(实测 %s)" % (x["cond"], x["measured"])
                      for x in d["clause_460"] if x["satisfied"])
    elif any(g["pass_"] is False for g in gate):
        verdict, why = "unavailable（吞吐限制）", "§2.3 T4 门超限（行 434）"
    else:
        verdict, why = "NEEDS_VERIFY", "预注册的 pass / unavailable 条件**均未满足** ⇒ 不得自选；上报主 agent"
    d["four_state"] = dict(verdict=verdict, trigger=why)
    # 降级路径可行性（逐条，用实测数字）
    d["degradation_paths"] = dict(
        p1_region_family=dict(rows_W_DEPTH_LE2=wl_rows.get("W_DEPTH_LE2"),
                              rows_W_DEPTH_LE3=wl_rows.get("W_DEPTH_LE3"),
                              note="族白名单（depth≤2 的 24 个 / depth≤3 的 85 个）行数为实测"),
        p2_seed_experiments=dict(experiments_measurable=True,
                                 rows_per_experiment=rows_b1,
                                 bytes_per_experiment=int(br * rows_b1) if br else None,
                                 note="单实验整件 CSV 的字节数与行数为实测（B1/A8）"),
        p3_literature=None,
        p4_distance_kernel=None)
    with open(OUT_ASSEM, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2, sort_keys=True)
    reach.add(dict(utc=utcnow(), spec="D1",
                   label="★ S1 四态裁决（按 §2.2 第 460 行 + §2.3 第 434 行预注册条件逐条数值化）",
                   kind="verdict", url="见 data/m12_s1_gate_assembly.json",
                   http_code="", http_code_pass2="", curl_exit="", rma_success="", bytes="",
                   content_type="", ttfb_s="", seconds="",
                   speed_Bps="%.1f" % sp, total_rows="",
                   verdict=("REACHABLE" if verdict == "pass" else
                            ("TOO_SLOW" if verdict.startswith("unavailable") else "NEEDS_VERIFY")),
                   verdict_scope="S1（ProjectionStructureUnionize）作为 M12 区域间权重来源的取数可行性",
                   sha256_prefix="", evidence_level="measured", provenance=PROV,
                   recheck_policy=RECHECK,
                   wrong_usage_consequence="不自选四态：条件均不满足时判 NEEDS_VERIFY 上报，"
                   "而不是判 pass 或 unavailable",
                   missing_dimension="p3 文献档（Oh 2014 补充材料可得性）与 S3 体数据未测",
                   note="四态=%s｜触发=%s｜中位吞吐=%.1f B/s｜bytes/row=%.2f" % (
                       verdict, why, sp, br or -1)))
    return d


# ---------------------------------------------------------------- Phase E（T5/T6：真实取数 + 解析 + 装配）
def fetch_csv(url, out_name, timeout=W_FETCH):
    """**会落盘成对象的取数** ⇒ `--fail=True`（防错误页落盘成假件，M12-M-04）。"""
    p = os.path.join(RAW, out_name)
    if os.path.exists(p):
        os.remove(p)
    hdr = os.path.join(RAW, "_hdr_%s.txt" % out_name.replace(".", "_"))
    rec = run_curl(url, timeout=timeout, out_path=p, hdr_path=hdr, fail_on_error=True)
    return p, rec, _hdr(rec, "content-length"), _hdr(rec, "etag")


def phase_E(reach, integ, counts):
    """T5 内容级 + T6 装配可行性：**在真实数据上**把取数→解析→装配走通一遍（有界，≤10 MB）。"""
    out = {}
    sds = counts["_sample_sds"]
    # E1 单实验整件 CSV
    url = ("%s/query.csv?criteria=model::ProjectionStructureUnionize,"
           "rma::criteria[section_data_set_id$eq%s],rma::options[num_rows$eqall]" % (API, sds))
    p, rec, cl, etag = fetch_csv(url, "E1_experiment_%s.csv" % sds)
    rows, cols, err = read_psu_csv(p)
    size = os.path.getsize(p) if os.path.exists(p) else 0
    counts["_E1_rows"] = len(rows)
    counts["_E1_cols"] = cols
    counts["_E1_bytes"] = size
    integrity_row("E1", p, size, cl, etag, rec, integ,
                  semantic="解析数据行数 == A8 实测 total_rows（%s）" % counts.get(
                      "A8_rows_per_experiment"),
                  ok=(len(rows) == counts.get("A8_rows_per_experiment")),
                  note="列数=%d；curl_exit=%s；err=%s" % (len(cols), rec["exit"], err))
    reach.add(reach_row(
        "E1", "T5 内容级：单个投射实验的整件 CSV 取回并解析（语义锚 = 解析行数 == 计数查询行数）",
        "api_object_fetch", dict(url=url, code=rec["code"], exit=rec["exit"],
                                 nbytes=size, ttfb=rec["ttfb"], seconds=rec["seconds"],
                                 speed=rec["speed"],
                                 content_type=_hdr(rec, "content-type"), success=True),
        "", "列数=%d｜解析行数=%d" % (len(cols), len(rows)),
        "REACHABLE" if (rec["code"] == "200" and rows) else "BLOCKED",
        "本条作用域仅限本行列出的这一维（该实验的行能否取回并解析）",
        len(rows),
        "不取整件而按『行数 × 单价』估算 ⇒ 体量与装配时间错量级",
        "无（本行作用域内：整件已取回并解析）"))

    # E2 重跑确定性（清单 §2.5 通过条件 ④）
    p2, rec2, _, _ = fetch_csv(url, "E2_experiment_%s_rerun.csv" % sds)
    s1 = sha256_file(p) if os.path.exists(p) else "NA"
    s2 = sha256_file(p2) if os.path.exists(p2) else "NA"
    counts["_E2_determinism"] = dict(sha_first=s1, sha_second=s2, equal=(s1 == s2))
    reach.add(reach_row(
        "E2", "重跑确定性：同一 URL 二次取回（清单 §2.5 通过条件 ④）", "determinism",
        dict(url=url, code=rec2["code"], exit=rec2["exit"], nbytes=os.path.getsize(p2)
             if os.path.exists(p2) else 0, ttfb=rec2["ttfb"], seconds=rec2["seconds"],
             speed=rec2["speed"], content_type=_hdr(rec2, "content-type"), success=True),
        "", "sha256 一次=%s… 二次=%s… ⇒ %s" % (s1[:16], s2[:16], "一致" if s1 == s2 else "不一致"),
        "REACHABLE" if (s1 == s2 and s1 != "NA") else "NEEDS_VERIFY", "本条作用域仅限本行",
        "与 E1 同", "两次取数不一致 ⇒ 该源非确定性 ⇒ 全部解析结论 invalid", "无"))

    # E3 该实验的 is_injection=true 行（源区提取路径）
    urls = ("%s/query.csv?criteria=model::ProjectionStructureUnionize,"
            "rma::criteria[section_data_set_id$eq%s],rma::criteria[is_injection$eqtrue],"
            "rma::options[num_rows$eqall]" % (API, sds))
    p3, rec3, cl3, etag3 = fetch_csv(urls, "E3_experiment_%s_injection.csv" % sds)
    inj, _, _ = read_psu_csv(p3)
    counts["_E3_injection_rows"] = len(inj)
    counts["_E3_injection_structure_ids"] = sorted({r.get("structure_id") for r in inj})
    integrity_row("E3", p3, os.path.getsize(p3) if os.path.exists(p3) else 0, cl3, etag3,
                  rec3, integ, semantic="is_injection=true 行数 > 0", ok=len(inj) > 0,
                  note="源结构 id 集合=%s" % sorted({r.get("structure_id") for r in inj})[:8])
    reach.add(reach_row(
        "E3", "源（注射）维提取路径实测：该实验 `is_injection$eqtrue` 的行数与其 structure_id 集合"
              "（19 字段中**无** `injection_structure_id` ⇒ 源维只能由该布尔列反推）",
        "api_object_fetch", dict(url=urls, code=rec3["code"], exit=rec3["exit"],
                                 nbytes=os.path.getsize(p3) if os.path.exists(p3) else 0,
                                 ttfb=rec3["ttfb"], seconds=rec3["seconds"],
                                 speed=rec3["speed"],
                                 content_type=_hdr(rec3, "content-type"), success=True),
        "", "is_injection=true 行数=%d｜structure_id 集合=%s" % (
            len(inj), sorted({r.get("structure_id") for r in inj})[:8]),
        "REACHABLE" if len(inj) > 0 else "NEEDS_VERIFY",
        "本条作用域仅限本行列出的这一维（单实验的源行可否反推）", len(inj),
        "把 is_injection=true 行当『该实验的源区唯一标识』而不做粒度核对 ⇒ 源维可能是"
        "『注射区及其祖先集合』而非单一区域 ⇒ 区域对矩阵的行维错",
        "**is_injection 的粒度语义（单区域 vs 注射区+祖先集合）未确认**——卡在 "
        "`data/m12_s1_raw/A0_body.json` 的 19 字段中**无** `injection_structure_id`，"
        "只能用 A10/A8 的比值与 E3 的实测行数间接判断"))

    # E4 T6 装配可行性（稀疏度 + 内存 + 逐实验对齐）
    asm = assembly_report(rows, inj, counts)
    counts["_E4_assembly"] = asm
    reach.add(reach_row(
        "E4", "T6 装配可行性（M12 新增第 4 层）：把实测 CSV 透视成『源区 → 目标区 × 强度』"
              "并算稀疏度/内存/逐实验参数数", "assembly",
        dict(url="(本地解析 E1/E3 落盘件)", code="200", exit=0, nbytes=0, ttfb=0.0,
             seconds=0.0, speed=0.0, content_type="text/csv", success=True), "",
        asm["note"], "REACHABLE" if asm["ok"] else "UNLOADABLE",
        "本条作用域仅限本行列出的这一维（装配可行性；**不含**全脑 586 区域装配）",
        asm.get("n_targets"), "取得到但装不进 ⇒ 必须判 unavailable（装配不可行）（第 436 行 T6）",
        "全脑 586×586 的**真实装配**未做（本节点只做单实验透视）⇒ 内存/时间上界为**推算**，"
        "不是实测"))
    return out


def integrity_row(spec, p, size, cl, etag, rec, integ, semantic, ok, note=""):
    md5 = ""
    try:
        if os.path.exists(p) and etag and "-" not in etag:
            md5 = hashlib.md5(open(p, "rb").read()).hexdigest()
    except Exception:
        md5 = ""
    size_match = (str(size) == str(cl)) if cl else "NA(无CL)"
    etag_match = (md5 == etag.strip('"')) if (etag and md5) else "NA"
    integ.add(dict(utc=utcnow(), spec=spec, object_path=os.path.relpath(p, ROOT),
                   bytes=size, content_length=cl or "NA", size_match=size_match,
                   etag=etag or "NA", etag_md5_recomputed=md5 or "NA",
                   etag_match=etag_match,
                   sha256=sha256_file(p) if os.path.exists(p) else "NA",
                   sha_basis="file_bytes(=payload)",
                   semantic_anchor=semantic, semantic_ok=ok,
                   verdict=("COMPLETE" if (rows_ok(size_match) and ok) else "INCOMPLETE"),
                   note=note + "｜T5 五步= http→大小→SHA256→解析→尺寸自洽"))


def rows_ok(x):
    return x is True or x == "NA(无CL)"


# ---------------------------------------------------------------- Phase X（跨节点核对）
def phase_X(reach, counts):
    """把本节点对 **B1 表 `m12_source_reachability.csv::S1-03`** 的复现结果写成更正行（append-only）。"""
    b1_bytes = None
    p = os.path.join(DATA, "m12_source_reachability.csv")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(l for l in f if not l.startswith("#")):
                if r.get("source_id") == "S1-03":
                    b1_bytes = r.get("bytes")
                    break
    mine = counts.get("_syntax_A16", {})
    same = str(b1_bytes) == str(mine.get("bytes"))
    reach.add(dict(utc=utcnow(), spec="X1",
                   label="**跨节点核对**：B1 表 `S1-03`（`structure_id$eq16997`）记 "
                         "verdict=REACHABLE / missing_dimension=无；本节点同 URL 复现",
                   kind="crossnode_correction",
                   url=mine.get("code") and ("见 data/m12_s1_gate_syntax.csv::A16") or "",
                   http_code=mine.get("code", ""), http_code_pass2="", curl_exit=0,
                   rma_success=mine.get("success"), bytes=mine.get("bytes"),
                   content_type="application/json", ttfb_s="", seconds="", speed_Bps="",
                   total_rows=mine.get("total_rows"),
                   verdict="NEEDS_VERIFY", verdict_scope="仅 B1 表 `S1-03` 这一行的口径",
                   sha256_prefix="", evidence_level="measured", provenance=PROV,
                   recheck_policy=RECHECK,
                   wrong_usage_consequence="把「85 字节的空结果信封」读成「过滤成功且可达」⇒ "
                   "白名单试算结论为空却被当成通过（静默失效家族第 6 例）",
                   missing_dimension="B1 表 `m12_source_reachability.csv` **无 `total_rows` 列** ⇒ "
                   "「服务端白名单过滤后行数」在该表里**没有落点**（这正是 §3 表首行列的缺口）",
                   note="本节点实测：bytes=%s（B1 记 %s，**逐字节相同**）、rma success=%s、"
                        "total_rows=%s ⇒ 该 85 字节响应的语义是**空结果信封**"
                        "（`{\"success\": true, \"num_rows\": 0, \"total_rows\": 0, \"msg\": []}`），"
                        "而**不是**「白名单试算命中」；B1 自注也写「若返回 4xx/空 ⇒ 记 "
                        "REACHABLE_NO_DATA」，与其 verdict 列（REACHABLE）不一致"
                        % (mine.get("bytes"), b1_bytes, mine.get("success"),
                           mine.get("total_rows"))))
    counts["_X1_b1_bytes"] = b1_bytes
    counts["_X1_same_bytes"] = same


def _is_true(v):
    return str(v).strip().lower() in ("t", "true", "1", "yes")


def assembly_report(rows, inj, counts):
    """把单实验实测行透视 + 给出全脑装配的**推算上界**（明确标为推算）。"""
    if not rows:
        return dict(ok=False, note="无解析行 ⇒ 装配不可判（不是『装不进』）")
    n = len(rows)
    targets = {r["structure_id"] for r in rows if not _is_true(r.get("is_injection"))}
    src = sorted({r["structure_id"] for r in inj})
    nz_vol = sum(1 for r in rows if float(r.get("projection_volume") or 0) > 0)
    nz_norm = sum(1 for r in rows if float(r.get("normalized_projection_volume") or 0) > 0)
    nz_dens = sum(1 for r in rows if float(r.get("projection_density") or 0) > 0)
    hem = sorted({r.get("hemisphere_id") for r in rows})
    wl = counts["_rules"]["W_DEPTH_LE6"]
    R = len(wl)
    dense = nz_vol / float(max(1, n - len(inj)))
    return dict(ok=(n > 0 and len(targets) > 0),
                n_rows_experiment=n, n_source_rows=len(inj), n_target_rows=n - len(inj),
                n_distinct_targets=len(targets), source_structure_ids=src,
                hemispheres=hem,
                nonzero_projection_volume=nz_vol, nonzero_normalized_projection_volume=nz_norm,
                nonzero_projection_density=nz_dens,
                nonzero_fraction_of_all_rows=round(nz_vol / float(n), 6),
                implied_nonzero_pairs_per_experiment=int(round(dense * len(targets))),
                R_whitelist=R, R_times_R_pairs=R * R,
                memory_RxR_float64_MB=round(R * R * 8 / 1e6, 3),
                note=("单实验实测：%d 行（is_injection=true %d 行 / 目标行 %d），"
                      "distinct 目标结构 **%d**，hemisphere=%s，projection_volume>0 %d 行（%.1f%%）；"
                      "R=%d ⇒ R×R matrix 内存 %.3f MB（float64）⇒ **装得进内存**；"
                      "但 R×R 的真实装配（跨实验拼接）**本节点未做**"
                      % (n, len(inj), n - len(inj), len(targets), hem, nz_vol,
                         100.0 * dense, R, R * R * 8 / 1e6)))


# ---------------------------------------------------------------- Phase F（分页可行性 + 页上限）
EXPECTED_COLS = ["id", "section_data_set_id", "structure_id", "hemisphere_id", "is_injection",
                 "sum_pixels", "sum_projection_pixels", "sum_pixel_intensity",
                 "sum_projection_pixel_intensity", "projection_density", "projection_intensity",
                 "projection_energy", "volume", "projection_volume",
                 "normalized_projection_volume", "max_voxel_density", "max_voxel_x",
                 "max_voxel_y", "max_voxel_z"]


def read_psu_csv(path):
    """★ 解析 PSU CSV，**列名不匹配即判 invalid**。

    **本节点自查拦下的缺陷（值得留档的第 7 类静默失效）**：首版 `page_fetch` 用朴素的
    `csv.DictReader` 统计行数 ⇒ RMA 的**语法错误页**（HTTP 200 + 两行文本：
    错误描述 + 回显的查询串）被读成 **2 条"数据行"**，于是 `parsed == 2`、
    `bytes ≈ 495`，而工具**一度据此认为"分页返回 2 行、页大小参数生效"**。
    根因与 B1 L6/L25（错误体落盘成假件、`--fail` 吞掉 403 正文）**同族**：
    **HTTP 200 与"拿到数据"是两件事**。
    ⇒ 固化：任何 PSU CSV 必须**先核列名 == 19 字段 schema**，否则判 `invalid` 并不入统计（第 13 条纪律）。
    """
    if not path or not os.path.exists(path):
        return [], [], "file missing"
    try:
        with open(path, encoding="utf-8", newline="") as f:
            first = f.readline().strip()
            cols = [c.strip() for c in first.split(",")]
            if cols != EXPECTED_COLS:
                head = open(path, encoding="utf-8", errors="replace").read(240)
                return [], cols, ("非 PSU CSV ⇒ **判 invalid，不得入统计**（实际首行列数=%d：%s；"
                                  "原文前 120 字=%s）" % (len(cols), cols[:4],
                                                        head[:120].replace("\n", " / ")))
            rows = list(csv.DictReader(open(path, encoding="utf-8", newline="")))
    except Exception as e:
        return [], [], "parse_exc:%s" % str(e)[:100]
    return rows, cols, ""


def page_fetch(url, out_name, timeout=W_FETCH):
    p, rec, cl, etag = fetch_csv(url, out_name, timeout=timeout)
    rows, cols, rej = read_psu_csv(p)
    return p, rec, cl, etag, len(rows), rej


def phase_F(reach, integ, counts):
    """★ 决定性测量：**服务端 60 s 生成上限 + Range 被忽略** 下，能否用
    `rma::options[num_rows$eqN][start_row$eqM]`（**每项一个方括号组**；逗号连接会解析失败）
    把大结果集取全？——这一维才是"实测吞吐下的 ETA"（`m12_b1_gate_closure.md` §3 表首行）
    在该端点上的**正确口径**。
    """
    out = {}
    ids = counts["_rules"]["W_DEPTH_LE2"]
    crit = "rma::criteria[structure_id$in%s]" % ",".join(str(x) for x in ids)
    total = counts["A_whitelist_target_counts"]["W_DEPTH_LE2"]

    def csv_page(spec, page_rows, start_row, label, wrong, expect=None):
        u = ("%s/query.csv?criteria=model::ProjectionStructureUnionize,%s,"
             "rma::options[num_rows$eq%d][start_row$eq%d]" % (API, crit, page_rows, start_row))
        p, rec, cl, etag, n, rej = page_fetch(u, "%s_page.csv" % spec, timeout=W_FETCH)
        size = os.path.getsize(p) if os.path.exists(p) else 0
        ok = (rec["code"] == "200" and n > 0 and not rej and
              (expect is None or n == expect) and (n == page_rows or expect is not None))
        integrity_row(spec, p, size, cl, etag, rec, integ,
                      semantic=("解析行数 == 请求页大小(%d)" % page_rows) if expect is None
                      else ("解析行数 == 期望 %d（末尾页锚）" % expect),
                      ok=ok, note="start_row=%d｜解析=%d｜rej=%s" % (start_row, n, rej))
        reach.add(reach_row(spec, label, "api_page_fetch",
                            dict(url=u, code=rec["code"], exit=rec["exit"], nbytes=size,
                                 ttfb=rec["ttfb"], seconds=rec["seconds"], speed=rec["speed"],
                                 content_type=_hdr(rec, "content-type"), success=None),
                            "", "http=%s｜解析行数=%d｜字节=%d｜TTFB=%.2fs｜墙钟=%.2fs%s"
                            % (rec["code"], n, size, rec["ttfb"], rec["wall"],
                               ("｜🔴 %s" % rej) if rej else ""),
                            "REACHABLE" if ok else ("REACHABLE_NO_DATA" if rec["code"] == "504"
                                                    else "INVALID"),
                            "本条作用域仅限本行列出的这一维（该页大小/起始行下的可取性）",
                            n, wrong,
                            "无" if ok else "该页参数下的可取性（504 ⇒ 服务端生成上限；"
                            "非 PSU CSV ⇒ 语法/口径错误）"))
        return dict(spec=spec, code=rec["code"], start_row=start_row, requested=page_rows,
                    parsed=n, rej=rej, bytes=size, ttfb=rec["ttfb"], wall=rec["wall"],
                    bytes_per_s=rec["speed"], etag=etag, url=u)

    # F1 页大小扫描：找服务端 60 s 生成上限内的**最大可取页**
    sweep = []
    for pr in (2000, 10000, 50000, 200000):
        sweep.append(csv_page("F1_n%d" % pr, pr, 0,
                              "★ 分页可行性 · 页大小扫描 num_rows$eq%d][start_row$eq0"
                              "（找服务端 60 s 生成上限下的**最大可取页**）" % pr,
                              "把「整件 504」读成「数据不可得」而不用分页 ⇒ 假 unavailable"
                              "（504 是**服务端生成超时**，不是链路慢、也不是无数据）"))
        time.sleep(0.4)
    counts["_F1_sweep"] = sweep
    ok_pages = [s for s in sweep if s["code"] == "200" and s["parsed"] == s["requested"]
                and not s["rej"]]
    maxpr = max([s["requested"] for s in ok_pages]) if ok_pages else 2000
    counts["F_max_page_rows"] = maxpr
    counts["F_max_page_measured"] = bool(ok_pages)

    # F2 末尾页锚：start_row = total − 500 ⇒ **应恰好返回 500 行**（偏移语义 + 计数自洽的强锚）
    f2 = csv_page("F2_tail500", 2000, total - 500,
                  "★ 分页可行性 · **末尾页锚** num_rows$eq2000][start_row$eq%d（= 总行数 %d − 500）"
                  "⇒ 期望**恰好 500 行**（若不等于 500 ⇒ start_row 语义或计数不自洽）"
                  % (total - 500, total),
                  "只做整件取数 ⇒ 无法断点续传（B1 L19：冻结下 `-C -` 是必需件）", expect=500)
    # F3 中段页随机访问
    f3 = csv_page("F3_mid", 2000, 200000,
                  "★ 分页可行性 · 中段页 start_row$eq200000（证明可任意定位，非只有首页可取）",
                  "只有首页可取 ⇒ 分页不成立")
    counts["_F2_tail500"], counts["_F3_mid"] = f2, f3
    npages = -(-total // maxpr)
    wall = max([s["wall"] for s in ok_pages]) if ok_pages else None
    counts["F_n_pages"] = npages
    counts["F_page_wall_s"] = [round(s["wall"], 3) for s in sweep]
    counts["F_ETA_paged_s"] = (wall * npages) if wall else None
    counts["F_ETA_paged_min"] = (wall * npages / 60.0) if wall else None
    counts["F_ETA_basis"] = ("页数(%d = ceil(%d/%d)) × 实测最大单页墙钟(%s s) —— 因为该端点"
                             "**忽略 Range** 且大结果集 **504**，「总字节 / 链路吞吐」在 S1 上"
                             "**不适用**" % (npages, total, maxpr,
                                            ("%.2f" % wall) if wall else "NA"))

    # F4 is_injection=true 行的归属（源维语义）
    u = ("%s/query.csv?criteria=model::ProjectionStructureUnionize,"
         "rma::criteria[is_injection$eqtrue],rma::options[num_rows$eq10]" % API)
    p, rec, cl, etag, n, rej = page_fetch(u, "F4_injection_rows.csv", timeout=W_COUNT)
    rows4, _, _ = read_psu_csv(p)
    sds_set = sorted({r["section_data_set_id"] for r in rows4})
    structs = sorted(int(r["structure_id"]) for r in rows4)
    counts["_F4_injection_sample"] = dict(n=n, sds=sds_set, structs=structs,
                                          hemi=sorted({r["hemisphere_id"] for r in rows4}),
                                          rej=rej)
    # F5：同一实验下的 is_injection=true 计数（**必须显式核 success**）
    o = rma_query("ProjectionStructureUnionize",
                  ["rma::criteria[section_data_set_id$eq%s]" % counts["_sample_sds"],
                   "rma::criteria[is_injection$eqtrue]"], spec="F5",
                  save=os.path.join(RAW, "F5_body.json"))
    counts["_F5_same_exp_injection"] = dict(code=o["code"], success=o["success"],
                                            total_rows=o["total_rows"], msg=str(o["msg"])[:120])
    # 祖先链假说的**本地证伪**（用 S2 已落盘件，零网络成本）
    chain = {}
    for sid in structs:
        x = BYID.get(sid)
        if x:
            chain[sid] = dict(depth=x["depth"], acronym=x["acronym"], name=x["name"],
                              structure_id_path=x["structure_id_path"])
    counts["_F4_ancestor_check"] = chain
    req = sorted(chain)
    common = None
    for sid in req:
        pth = chain[sid]["structure_id_path"]
        common = pth if common is None else os.path.commonprefix([common, pth])
    chain_like = all(chain[b]["structure_id_path"].startswith(
        chain[a]["structure_id_path"] + "/") or
        chain[a]["structure_id_path"].startswith(chain[b]["structure_id_path"] + "/")
        for i, a in enumerate(req) for b in req[i + 1:])
    counts["_F4_is_single_chain"] = bool(chain_like)
    reach.add(reach_row(
        "F4", "源维语义实测：`is_injection$eqtrue` 抽样行的 sds / structure_id / hemisphere 归属"
              "（+ 同实验下该过滤计数 + **祖先链假说的本地证伪**）", "api_page_fetch",
        dict(url=u, code=rec["code"], exit=rec["exit"],
             nbytes=os.path.getsize(p) if os.path.exists(p) else 0, ttfb=rec["ttfb"],
             seconds=rec["seconds"], speed=rec["speed"],
             content_type=_hdr(rec, "content-type"), success=True), "",
        "抽样 %d 行同属 sds=%s｜structure_id=%s｜hemi=%s｜同一实验(%s)下该过滤 count=%s"
        "(success=%s)｜十个 structure_id 是否构成**单一祖先链**=%s（本地按 structure_id_path 核）"
        % (n, sds_set, structs, sorted({r["hemisphere_id"] for r in rows4}),
           counts["_sample_sds"], counts["_F5_same_exp_injection"]["total_rows"],
           counts["_F5_same_exp_injection"]["success"], chain_like),
        "REACHABLE" if (n > 0 and not rej) else "NEEDS_VERIFY",
        "本条作用域仅限本行列出的这一维（源维可否由布尔列反推）", n,
        "把 is_injection=true 行当『该实验的单一注射源区』而不核粒度/半球/祖先关系 ⇒ 区域对矩阵"
        "的**行维**多算或错配（P6a 的 ρ 会算在一个语义错误的矩阵上）",
        "**is_injection 的粒度语义未确认**——卡在两处：① `A0_body.json` 的 19 字段中**没有** "
        "`injection_structure_id`；② 抽样行显示同一实验的 is_injection=true 覆盖**多个不相关"
        "分支**的区域（非单一祖先链）⇒ 「按 sds 取 is_injection=true 反推源区」**不成立**，"
        "源维须改走 `SectionDataSet` 的注射元数据（**该 join 本节点未测**）"))
    # F6：整件 504 的临界（记录 B2/B3 的 504 与 F1 的对比）
    out["F_summary"] = dict(total_rows_W_DEPTH_LE2=total, max_page_rows=maxpr,
                            n_pages=npages, page_wall_s=wall,
                            ETA_paged_s=counts["F_ETA_paged_s"],
                            ETA_paged_min=counts["F_ETA_paged_min"],
                            whole_object_http=counts.get("_obj_B3", {}).get("code"),
                            whole_object_ttfb=counts.get("_obj_B3", {}).get("ttfb"))
    return out


# ---------------------------------------------------------------- Phase Z（机验断言表）
OUT_ASSERT = os.path.join(DATA, "m12_s1_gate_assertions.json")


def phase_Z(counts, rules):
    """**逐条机验断言**（每条：断言 → 实测值 → pass/fail）。报告只许引这里，不许引记忆。"""
    A = []

    def add(aid, claim, measured, ok, note=""):
        A.append(dict(id=aid, claim=claim, measured=measured, pass_=bool(ok), note=note))

    def jload(p):
        return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}

    wl = counts.get("A_whitelist_target_counts", {})
    per_id = counts.get("A7_per_id_counts", {})
    ssum = counts.get("A7_sum")
    s2disk = counts.get("s2_total_rows_on_disk")
    f1 = counts.get("_F1_sweep", [])
    okpages = [s for s in f1 if s["code"] == "200" and s["parsed"] == s["requested"] and not s["rej"]]

    add("Z1", "S2 结构表计数（本轮实测）== S2 已落盘件 total_rows（跨节点一致性）",
        "A1=%s vs disk=%s" % (counts.get("A1_structure_count"), s2disk),
        counts.get("A1_structure_count") == s2disk and s2disk is not None)
    add("Z2", "PSU 全表行数 ≥ 10⁶（§2.2 第 460 行 unavailable 条件的第一半）",
        "A2=%s" % counts.get("A2_psu_full_count"),
        (counts.get("A2_psu_full_count") or 0) >= 10 ** 6)
    add("Z3", "**白名单过滤语义成立**：Σ(逐 id 单独计数) == count(`$in<同一白名单>`)",
        "Σ=%s vs $in=%s（n_ids=%d）" % (ssum, wl.get("W_DEPTH_LE2"), len(per_id)),
        (ssum is not None and ssum == wl.get("W_DEPTH_LE2")),
        "不成立 ⇒ 所有白名单过滤计数与由其推出的 ETA 全部 invalid")
    mono = [wl.get("W_DEPTH_LE2"), wl.get("W_DEPTH_LE3"), wl.get("W_DEPTH_LE5"), wl.get("W_DEPTH_LE6")]
    add("Z4", "白名单层级嵌套 ⇒ 过滤计数单调不减（depth≤2 ⊂ ≤3 ⊂ ≤5 ⊂ ≤6）",
        str(mono), all(x is not None for x in mono) and all(
            mono[i] < mono[i + 1] for i in range(len(mono) - 1)))
    s14 = counts.get("_syntax_A14", {})
    s15 = counts.get("_syntax_A15", {})
    add("Z5", "`$in[...]`（带方括号）**语法错误**且 `$in...`（逗号列表）成功 —— HTTP 均 200，"
              "只有 `success` 字段能区分 ⇒ 「HTTP 200 ≠ 查询成功」",
        "A14(success=%s, msg=%s) / A15(success=%s, total_rows=%s)"
        % (s14.get("success"), str(s14.get("msg"))[:60], s15.get("success"),
           s15.get("total_rows")),
        s14.get("success") is False and s15.get("success") is True)
    add("Z6", "复现 B1 表 `S1-03` 的同 URL 响应：bytes == B1 记录值 且 total_rows == 0 "
              "⇒ 该行是**空结果信封**（B1 却记 verdict=REACHABLE / missing_dimension=无）",
        "本轮 bytes=%s vs B1 表 bytes=%s｜total_rows=%s｜success=%s"
        % (counts.get("_syntax_A16", {}).get("bytes"), counts.get("_X1_b1_bytes"),
           counts.get("_syntax_A16", {}).get("total_rows"),
           counts.get("_syntax_A16", {}).get("success")),
        counts.get("_X1_same_bytes") is True
        and counts.get("_syntax_A16", {}).get("total_rows") == 0,
        "B1 表**无 total_rows 列** ⇒ 「白名单过滤后行数」在该表里没有落点（§3 缺口）")
    e1r, e1b = counts.get("_E1_rows"), counts.get("_E1_bytes")
    add("Z7", "T5 语义锚：单实验整件 CSV 解析行数 == 计数查询行数（%s）"
        % counts.get("A8_rows_per_experiment"),
        "解析=%s vs 计数=%s" % (e1r, counts.get("A8_rows_per_experiment")),
        e1r == counts.get("A8_rows_per_experiment") and e1r is not None)
    add("Z8", "重跑确定性（清单 §2.5 条件④）：同一 URL 两次取数 sha256 一致",
        str(counts.get("_E2_determinism", {}).get("equal")) + "｜"
        + str(counts.get("_E2_determinism", {}).get("sha_first", ""))[:16],
        counts.get("_E2_determinism", {}).get("equal") is True)
    add("Z9", "该端点**忽略 Range**：带 `-r 0-1023` 的请求实际返回**完整对象**"
              "（B1 落盘字节 == E1 无 Range 整件字节，sha256 逐字节一致）",
        "B1 nbytes=%s vs E1 bytes=%s｜sha256 一致=%s" % (
            counts.get("_obj_B1", {}).get("nbytes"), e1b,
            _same_sha(counts)),
        counts.get("_obj_B1", {}).get("nbytes") == e1b and _same_sha(counts),
        "⇒ 清单第 579–582 行的 1 MiB Range 吞吐模板在 S1 上**口径不适用**")
    add("Z10", "大结果集**整件**请求：B2（is_injection=true, 342,475 行）⇒ **HTTP 504 @ TTFB "
               "60.60 s**（服务端生成上限）；B3/B4 登记为**未测**（不重试：上一轮 B3 整件请求实测 "
               "936.2 s / 0 字节 = 本机**冻结型挂死**，按 M12-M-01/L24 排除出统计，"
               "**不得**读成「无数据」或「链路慢」）",
        "B2=%s(ttfb=%.1fs,bytes=%s)｜B2/B3 504 判定｜上一轮 B3: 936.2 s/0 B/exit=28"
        % (counts.get("_obj_B2", {}).get("code"), counts.get("_obj_B2", {}).get("ttfb") or -1,
           counts.get("_obj_B2", {}).get("nbytes")),
        counts.get("_obj_B2", {}).get("code") == "504"
        and counts.get("_obj_B2", {}).get("nbytes") == 132,
        "整件路径已证不行 ⇒ 必须走分页（Z11）")
    f2 = counts.get("_F2_tail500", {})
    add("Z11", "**服务端分页语法** `rma::options[num_rows$eqN][start_row$eqM]`（每项一个方括号组）"
               "可用，且末尾页锚 `start_row = 总行数−500` **恰好返回 500 行**"
               "（start_row 语义 == 有序结果的**行偏移**，且与计数自洽）",
        "最大可取页=%s（实测=%s）｜末尾页 parsed=%s（期望 500）｜中段页=%s"
        % (counts.get("F_max_page_rows"), counts.get("F_max_page_measured"),
           f2.get("parsed"), counts.get("_F3_mid", {}).get("parsed")),
        counts.get("F_max_page_measured") is True and f2.get("parsed") == 500,
        "页数 × 实测单页墙钟 = 该端点上唯一有效的 ETA 口径")
    add("Z12", "`is_injection=true` 抽样行**同属一个 sds**，但其 structure_id **不构成单一祖先链**"
               "⇒ 「按 sds 取 is_injection=true 反推单一注射源区」**被证伪**",
        "sds=%s｜structs=%s｜单链=%s" % (counts.get("_F4_injection_sample", {}).get("sds"),
                                        counts.get("_F4_injection_sample", {}).get("structs"),
                                        counts.get("_F4_is_single_chain")),
        counts.get("_F4_is_single_chain") is False
        and len(counts.get("_F4_injection_sample", {}).get("structs", [])) > 3)
    add("Z13", "抽样实验的 `is_injection=true` 行数 == 0（且 `success=true`，为**真 0** 而非语法错）"
               "⇒ 该布尔列**不是**逐实验常量 ⇒ 源（行）维**未建立**",
        "same-exp count=%s(success=%s)｜全表 is_injection=true=%s"
        % (counts.get("_F5_same_exp_injection", {}).get("total_rows"),
           counts.get("_F5_same_exp_injection", {}).get("success"),
           counts.get("A10_is_injection_true")),
        counts.get("_F5_same_exp_injection", {}).get("total_rows") == 0
        and counts.get("_F5_same_exp_injection", {}).get("success") is True)
    sp = counts.get("C_median")
    add("Z14", "实测吞吐（**有效页**上的多轮中位）可得；`bytes/row` 由整件实测得出",
        "中位=%s B/s｜有效轮=%s｜bytes/row=%s" % (("%.1f" % sp) if sp else "NA",
                                                counts.get("C_speeds_Bps"),
                                                _fmt(counts.get("B_bytes_per_row_csv"))),
        sp is not None and sp > 0 and counts.get("B_bytes_per_row_csv"))
    add("Z15", "本工具**只用标准库**（无 torch / 无 MPS / 无 GPU；R34：既非引擎侧也非评估器侧）、"
               "且**零** `sys.executable` 用法（AST 级检查，非字符串匹配）",
        "imports=%s｜sys.executable 出现次数=%d" % (_imports_of_self(), _sys_exec_count()),
        "torch" not in _imports_of_self() and _sys_exec_count() == 0,
        "MPS 由主 agent 的 M11 作业独占（本节点的硬约束）")
    add("Z16", "白名单预注册件**确定性可重生成**（重跑 sha256 一致）",
        "sha=%s" % (sha256_file(OUT_WL)[:16] if os.path.exists(OUT_WL) else "NA"),
        _wl_deterministic(rules))
    add("Z17", "产物表非空且 spec 无重复（主表每 spec 一行）",
        "reach=%d｜syntax=%d｜thru=%d｜integ=%d" % (
            counts.get("_reach_rows", 0), counts.get("_syntax_rows", 0),
            counts.get("_thru_rows", 0), counts.get("_integ_rows", 0)),
        counts.get("_reach_rows", 0) >= 15 and counts.get("_thru_rows", 0) >= 3)
    asm = jload(OUT_ASSEM)
    add("Z18", "四态裁决由**预注册条件**推出（非自选），且触发条件可回读",
        "%s ← %s" % (asm.get("four_state", {}).get("verdict"),
                     asm.get("four_state", {}).get("trigger")),
        bool(asm.get("four_state", {}).get("verdict")))

    res = dict(utc=utcnow(), n=len(A), n_pass=sum(1 for x in A if x["pass_"]),
               assertions=A)
    with open(OUT_ASSERT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, sort_keys=True)
    print("ASSERTIONS %d/%d pass" % (res["n_pass"], res["n"]))
    for x in A:
        print("  %-5s %s  %s" % (x["id"], "PASS" if x["pass_"] else "FAIL",
                                 x["claim"][:80]))
    return res


def _sys_exec_count():
    """AST 级计数：真正使用 `sys.executable` 的次数（**不是**字符串匹配——docstring 里提到它不算）。"""
    import ast as _ast
    tree = _ast.parse(open(os.path.abspath(__file__), encoding="utf-8").read())
    n = 0
    for x in _ast.walk(tree):
        if (isinstance(x, _ast.Attribute) and x.attr == "executable"
                and isinstance(x.value, _ast.Name) and x.value.id == "sys"):
            n += 1
    return n


def _fmt(x):
    return "%.3f" % x if isinstance(x, float) else str(x)


def _imports_of_self():
    import ast as _ast
    tree = _ast.parse(open(os.path.abspath(__file__), encoding="utf-8").read())
    mods = set()
    for n in _ast.walk(tree):
        if isinstance(n, _ast.Import):
            mods.update(a.name.split(".")[0] for a in n.names)
        elif isinstance(n, _ast.ImportFrom) and n.module:
            mods.add(n.module.split(".")[0])
    return sorted(mods)


def _same_sha(counts):
    try:
        a = sha256_file(os.path.join(RAW, "B1_head.bin"))
        b = sha256_file(os.path.join(RAW, "E1_experiment_%s.csv" % counts["_sample_sds"]))
        return a == b
    except Exception:
        return False


def _wl_deterministic(rules):
    try:
        tmp = os.path.join(RAW, "_wl_regen.csv")
        write_whitelist_csv(rules, tmp)
        return sha256_file(tmp) == sha256_file(OUT_WL)
    except Exception:
        return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="A")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    os.makedirs(RAW, exist_ok=True)
    ROWS, S2_TOTAL = load_graph1()
    BYID.update({x["id"]: x for x in ROWS})
    rules = whitelist_rules(ROWS)
    write_whitelist_csv(rules, OUT_WL)

    reach = Sink(OUT_REACH, REACH_COLS,
                 "M12 S1-GATE 逐 spec 可达性/计数/对象级实测（generated_by="
                 "tools/probe_m12_s1_gate.py；**每 spec 一行 = 最新实测**）")
    thru = Sink(OUT_THRU, THRU_COLS,
                "M12 S1-GATE 吞吐实测（多轮，取中位；M11-R56/R63 吞吐不可外推）")
    integ = Sink(OUT_INTEG, INTEG_COLS, "M12 S1-GATE 完整性门（三级）")
    syntax = Sink(OUT_SYNTAX, SYNTAX_COLS,
                  "M12 S1-GATE RMA 查询**语法**实测（success 字段；HTTP 200 ≠ 成功）")

    counts = {"node": "S1-GATE", "utc_start": utcnow(), "_rules": rules,
              "s2_total_rows_on_disk": S2_TOTAL,
              "whitelist_sizes": {k: len(v) for k, v in rules.items()},
              "api": API, "python": sys.version.split()[0]}
    if os.path.exists(OUT_COUNTS):
        try:
            counts.update(json.load(open(OUT_COUNTS, encoding="utf-8")))
            counts["_rules"] = rules
        except Exception:
            pass
    phases = ["A", "B", "C", "D", "E", "F", "X", "Z"] if args.all else [args.phase]
    for p in phases:
        if p == "A":
            counts.update(phase_A(reach, syntax, counts))
        elif p == "B":
            counts.update(phase_B(reach, counts))
        elif p == "C":
            phase_C(thru, counts)
        elif p == "D":
            phase_D(counts, reach)
        elif p == "E":
            counts.update(phase_E(reach, integ, counts))
        elif p == "F":
            counts.update(phase_F(reach, integ, counts))
        elif p == "X":
            phase_X(reach, counts)
        elif p == "Z":
            phase_Z(counts, rules)

    # 清掉被本轮修正口径**取代**的历史键（首次（错误）分页语法产生的 _F2_lastpage/_F3_midpage
    # 与 F_first_page_wall_s：其"parsed=2"是 RMA 错误页被读成数据行，见 read_psu_csv 的 docstring）
    for k in ("_F2_lastpage", "_F3_midpage", "F_first_page_wall_s"):
        counts.pop(k, None)
    counts["_reach_rows"] = reach.flush()
    counts["_syntax_rows"] = syntax.flush()
    counts["_thru_rows"] = thru.flush()
    counts["_integ_rows"] = integ.flush()
    with open(OUT_COUNTS, "w", encoding="utf-8") as f:
        json.dump(counts, f, ensure_ascii=False, indent=2, sort_keys=True)
    print("phases=%s reach_rows=%d syntax_rows=%d thru_rows=%d integ_rows=%d"
          % ("/".join(phases), counts["_reach_rows"], counts["_syntax_rows"],
             counts["_thru_rows"], counts["_integ_rows"]))
    for k in sorted(counts):
        if k.startswith(("A1_", "A2_", "A7_", "A8_", "A9_", "A10", "A_", "_syntax_", "_obj_",
                         "B_", "C_", "_E1", "_E2", "_E3", "_E4", "_X1")):
            print("  %s = %s" % (k, counts[k]))
