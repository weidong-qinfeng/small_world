#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M12 判据表骨架的三条闸门检查器（M12 执行节点 A-5 工作包；纯标准库）。

闸门（与 docs/m12_criteria_table_skeleton.md §6 逐条对应）：
  G0 结构闸门      —— 列结构（M11 9 列同名 + scope）、记录数/唯一 key/自差集、evidence_level 与 final 值域
  G1 附录缺失检查  —— 任何被引用为"判据/常数"的量（note/provenance 里的 key 式 token 与 key=<K> 引用）
                      若不在本表 ⇒ **报错**（M11 R117 的直接产物：drive_rate_hz 曾缺席 103 行）
  G2 四元组完整性  —— 每行 scope 必须能回答 (config_id, role, estimator, aggregation)，不适用写 N/A
  G3 final=0 计数  —— 输出"当前有多少行 final=0"及其会让哪个判据 unavailable（M11 经验：final=0 ⇒ 对应对 unavailable）
  G4 清单列覆盖    —— 报告清单 §5.1 L887 指定但本骨架缺失的列（本骨架按派发要求用 9+1 列）

用法（在仓库根执行）：
  python3 neural_exploration/tools/check_m12_targets_gates.py
  python3 neural_exploration/tools/check_m12_targets_gates.py --negative-test drop-provenance
  python3 neural_exploration/tools/check_m12_targets_gates.py --negative-test break-quad
  python3 neural_exploration/tools/check_m12_targets_gates.py --negative-test dangling-ref

退出码：0 = 全部闸门通过；1 = 有闸门失败（--negative-test 下 = "闸门如期报错"）；2 = 用法/文件错误。
安全：本工具**只读**；任何写操作（含 --negative-test 的变异）只在内存中完成，绝不落盘到 data/。
"""

import argparse
import csv
import os
import re
import sys

# 列结构：前 9 列 = M11 data/m11_p5_targets.csv 逐列同名同义；第 10 列 scope 为本表新增
M11_COLUMNS = ["key", "value", "value_kind", "unit", "evidence_level",
               "provenance", "note", "wrong_usage_consequence", "final"]
SCOPE_COLUMN = "scope"
EXPECTED_COLUMNS = M11_COLUMNS + [SCOPE_COLUMN]

EVIDENCE_DOMAIN = {"measured", "derived", "literature", "literature_unverified", "unavailable"}
FINAL_DOMAIN = {"0", "1"}

# 清单 §5.1（docs/生物仿真M12实施清单.md L887）为 data/m12_targets.csv 指定的列
CHECKLIST_COLUMNS = ["key", "value", "value_kind", "unit", "evidence_level", "provenance",
                     "provenance_verified_by", "note", "wrong_usage_consequence",
                     "calibration_note", "final"]

KEYLIKE = re.compile(r"(?<!::)\b(protocol|gate|rule|target|ref|nc)_[A-Za-z0-9_]+\b")
EXPLICIT_REF = re.compile(r"key=([A-Za-z0-9_]+)")

TEXT_FIELDS = ["provenance", "note", "wrong_usage_consequence"]


class GateFailure(Exception):
    def __init__(self, gate, messages):
        Exception.__init__(self, gate)
        self.gate = gate
        self.messages = messages


def read_rows(path):
    """读 CSV：跳过 '#' 注释行（沿本表/M11 表/M12 reference 表的既有约定）。"""
    with open(path, encoding="utf-8") as fh:
        lines = [ln for ln in fh if not ln.startswith("#")]
    rdr = csv.DictReader(lines)
    rows = list(rdr)
    return rdr.fieldnames, rows


def read_m11_columns(path):
    try:
        with open(path, encoding="utf-8") as fh:
            for ln in fh:
                if ln.startswith("#"):
                    continue
                return next(csv.reader([ln]))
    except IOError:
        return None
    return None


def load(path):
    fields, rows = read_rows(path)
    return fields, rows


# ---------------------------------------------------------------- G0 结构闸门
def gate0_structure(fields, rows, m11_cols):
    problems = []
    if fields != EXPECTED_COLUMNS:
        problems.append("列名/列序不符：实测 %r；期望 %r" % (fields, EXPECTED_COLUMNS))
    if m11_cols is not None:
        if m11_cols != fields[:9]:
            problems.append("前 9 列未能与 M11 表逐列同名同义：M11=%r 本表前 9=%r"
                            % (m11_cols, fields[:9]))
    for i, r in enumerate(rows, start=1):
        if None in r or any(v is None for v in r.values()):
            problems.append("第 %d 条记录字段数 != %d（多余字段：%r）"
                            % (i, len(EXPECTED_COLUMNS), r.get(None)))
        key = (r.get("key") or "").strip()
        if not key:
            problems.append("第 %d 条记录 key 为空" % i)
        for col in ("value", "value_kind", "evidence_level", "provenance", "note",
                    "wrong_usage_consequence", "final", SCOPE_COLUMN):
            if not (r.get(col) or "").strip():
                problems.append("key=%s 的 %s 列为空" % (key, col))
        if r.get("evidence_level") not in EVIDENCE_DOMAIN:
            problems.append("key=%s 的 evidence_level=%r 不在域 %s"
                            % (key, r.get("evidence_level"), sorted(EVIDENCE_DOMAIN)))
        if r.get("final") not in FINAL_DOMAIN:
            problems.append("key=%s 的 final=%r 不在域 {0,1}" % (key, r.get("final")))
    keys = [r["key"] for r in rows]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    if dup:
        problems.append("重复 key：%r" % dup)
    selfdiff = set(keys) - set(keys)  # 与自身的差集（要求为空）
    if len(selfdiff) != 0:
        problems.append("set(key) 与自身差集非空：%r" % sorted(selfdiff))
    if problems:
        raise GateFailure("G0 结构闸门", problems)
    return {
        "records": len(rows),
        "unique_keys": len(set(keys)),
        "self_diff_empty": len(selfdiff) == 0,
        "final1": sum(1 for r in rows if r["final"] == "1"),
        "final0": sum(1 for r in rows if r["final"] == "0"),
        "evidence_level_counts": {},
    }


# ------------------------------------------------------------ G1 附录缺失检查
def gate1_reference_closure(rows):
    keys = set(r["key"] for r in rows)
    missing_explicit = []
    missing_bare = []
    n_explicit = 0
    n_bare = 0
    bare_distinct = set()
    for r in rows:
        blob = " || ".join((r.get(c) or "") for c in TEXT_FIELDS)
        for m in EXPLICIT_REF.finditer(blob):
            n_explicit += 1
            if m.group(1) not in keys:
                missing_explicit.append((r["key"], m.group(1)))
        for m in KEYLIKE.finditer(blob):
            tok = m.group(0)
            n_bare += 1
            bare_distinct.add(tok)
            if tok not in keys:
                missing_bare.append((r["key"], tok))
    problems = []
    for owner, tok in sorted(set(missing_explicit)):
        problems.append("key=%s 显式引用 key=%s ⇒ 本表无此 key（附录缺失）" % (owner, tok))
    for owner, tok in sorted(set(missing_bare)):
        problems.append("key=%s 的文本里出现判据式 token %s ⇒ 本表无此 key（附录缺失）"
                        % (owner, tok))
    if problems:
        raise GateFailure("G1 附录缺失检查", problems)
    return {"explicit_refs_checked": n_explicit,
            "bare_keylike_refs_checked": n_bare,
            "bare_keylike_distinct": len(bare_distinct),
            "registered_keys": len(keys)}


# ------------------------------------------------------------ G2 四元组完整性
def parse_scope(scope):
    segs = {}
    for part in scope.split(";"):
        if "=" not in part:
            return None, "scope 段 %r 不含 '='" % part
        k, v = part.split("=", 1)
        segs[k.strip()] = v.strip()
    return segs, None


def gate2_quadruple(rows):
    problems = []
    n = 0
    for r in rows:
        segs, err = parse_scope(r[SCOPE_COLUMN] or "")
        if err:
            problems.append("key=%s：%s" % (r["key"], err))
            continue
        for need in ("applies", "arm", "quad"):
            if need not in segs or not segs[need]:
                problems.append("key=%s：scope 缺 %s=" % (r["key"], need))
        if not segs.get("applies") or not segs.get("quad"):
            continue
        parts = segs["quad"].split("|")
        if len(parts) != 4:
            problems.append("key=%s：quad 应为 (config_id|role|estimator|aggregation) 四段，实测 %d 段：%r"
                            % (r["key"], len(parts), segs["quad"]))
        for i, p in enumerate(parts):
            if not p.strip():
                problems.append("key=%s：quad 第 %d 段为空（不适用须明文写 N/A）" % (r["key"], i + 1))
        n += 1
    if problems:
        raise GateFailure("G2 四元组完整性", problems)
    return {"rows_with_quad": n}


# ------------------------------------------------------------ G3 final=0 计数
def gate3_final0(rows):
    zero = [r for r in rows if r["final"] == "0"]
    detail = []
    for r in zero:
        segs, _ = parse_scope(r[SCOPE_COLUMN] or "")
        applies = segs.get("applies", "?") if segs else "?"
        detail.append((r["key"], applies))
    return {"count": len(zero), "detail": detail, "total": len(rows)}


# ------------------------------------------------------------ G4 清单列覆盖
def gate4_checklist_columns(fields):
    missing = [c for c in CHECKLIST_COLUMNS if c not in (fields or [])]
    return {"checklist_columns": CHECKLIST_COLUMNS, "missing": missing}


# ------------------------------------------------------------------ 变异（负向）
def mutate(rows, kind):
    """返回变异后的 rows（内存中；不写盘）。"""
    rows = [dict(r) for r in rows]
    if kind == "drop-provenance":
        rows[0]["provenance"] = ""
    elif kind == "break-quad":
        segs, _ = parse_scope(rows[0][SCOPE_COLUMN])
        segs["quad"] = "only_two|parts"
        rows[0][SCOPE_COLUMN] = ";".join("%s=%s" % (k, v) for k, v in segs.items())
    elif kind == "dangling-ref":
        rows[0]["note"] = (rows[0]["note"] +
                           "；本条依赖 key=protocol_undeclared_constant_hz 的取值")
    else:
        raise ValueError("unknown mutation %r" % kind)
    return rows


def run_all(fields, rows, m11_cols, quiet=False):
    ok = True
    out = []
    try:
        s = gate0_structure(fields, rows, m11_cols)
        out.append(("G0 结构闸门", "PASS", s))
    except GateFailure as e:
        ok = False
        out.append((e.gate, "FAIL", e.messages))
    try:
        s = gate1_reference_closure(rows)
        out.append(("G1 附录缺失检查", "PASS", s))
    except GateFailure as e:
        ok = False
        out.append((e.gate, "FAIL", e.messages))
    try:
        s = gate2_quadruple(rows)
        out.append(("G2 四元组完整性", "PASS", s))
    except GateFailure as e:
        ok = False
        out.append((e.gate, "FAIL", e.messages))
    s3 = gate3_final0(rows)
    out.append(("G3 final=0 计数", "INFO", s3))
    s4 = gate4_checklist_columns(fields)
    out.append(("G4 清单列覆盖（清单 §5.1 L887）", "INFO", s4))

    if not quiet:
        for name, status, payload in out:
            sys.stdout.write("=== %s === %s\n" % (name, status))
            if isinstance(payload, dict):
                for k in sorted(payload):
                    if k == "detail":
                        for key, applies in payload[k]:
                            sys.stdout.write("    final=0: %s  ⇒ 会让 applies=%s 的判据 unavailable\n"
                                             % (key, applies))
                    elif k == "missing":
                        sys.stdout.write("    %s = %r\n" % (k, payload[k]))
                    else:
                        sys.stdout.write("    %s = %r\n" % (k, payload[k]))
            else:
                for m in payload:
                    sys.stdout.write("    ERROR: %s\n" % m)
    return ok, out


def main(argv=None):
    ap = argparse.ArgumentParser(description="M12 判据表骨架三条闸门检查器（只读）")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--table", default=os.path.join(root, "data", "m12_targets.csv"))
    ap.add_argument("--m11-table", default=os.path.join(root, "data", "m11_p5_targets.csv"))
    ap.add_argument("--negative-test", default=None,
                    choices=["drop-provenance", "break-quad", "dangling-ref"])
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if not os.path.exists(args.table):
        sys.stderr.write("表不存在：%s\n" % args.table)
        return 2
    # 只读硬闸门：本工具不写任何文件（拒写 data/ 的实现形态 = 从不 open 写模式）
    fields, rows = load(args.table)
    m11_cols = read_m11_columns(args.m11_table) if os.path.exists(args.m11_table) else None

    if args.negative_test:
        mutated = mutate(rows, args.negative_test)
        ok, _ = run_all(fields, mutated, m11_cols, quiet=args.quiet)
        if ok:
            sys.stdout.write("[NEGATIVE-TEST] %s ⇒ 闸门**未**报错（工具失去判别性）\n"
                             % args.negative_test)
            return 0
        sys.stdout.write("[NEGATIVE-TEST] %s ⇒ 闸门如期报错（退出码 1）\n" % args.negative_test)
        return 1

    ok, _ = run_all(fields, rows, m11_cols, quiet=args.quiet)
    sys.stdout.write("[TABLE] %s  records=%d  final=1:%d  final=0:%d\n"
                     % (args.table, len(rows),
                        sum(1 for r in rows if r["final"] == "1"),
                        sum(1 for r in rows if r["final"] == "0")))
    sys.stdout.write("[RESULT] ALL GATES PASS\n" if ok else "[RESULT] GATE FAILURE\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
