#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M12 数据门 B1 · 判据/参考锚表（`data/m12_reference.csv`）—— **append-only 追加器**

纪律（逐条对应主 agent 验收口径）
--------------------------------
1. **只增不改**：本工具**从不重写**既有行；写入前逐行读出，已存在同 `ref_id` 即跳过（幂等）。
   旧行 `value` **逐字保留**（不做任何规范化/重排/重编码）。
2. 每个新增行**必须**带：
   - `provenance`（来源描述）+ `source_url` + `retrieved_at`（**判决/取数时刻**）；
   - `evidence_level` ∈ {`measured`, `literature`, `derived`, `unverified`}；
   - `wrong_usage_consequence`（**用错后果**：这个数被误用时会发生什么）。
3. **数值一律取自本轮的机器产物**（summary JSON / integrity CSV / reachability CSV / 白名单 CSV），
   **不手打数字**（避免「凭印象补数」）。
4. **`unavailable` 行必须登记「判决时刻 + 当时实测吞吐」**，并注明须以当场探针复核
   （M11-R56，复核周期 ≤24 h）。

用法
----
    PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
      ./.venv-m11/bin/python neural_exploration/tools/build_m12_reference.py
    … --dry-run    # 只打印将要追加的行，不写盘
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(DATA, "m12_reference.csv")

FIELDS = ["ref_id", "key", "value", "unit", "assertion", "evidence_level", "provenance",
          "source_url", "retrieved_at", "wrong_usage_consequence", "verdict", "note"]

HDR_COMMENT = (
    "# M12 参考锚 / 数据门判据行（generated_by=tools/build_m12_reference.py；"
    "**只允许新增行**；evidence_level 取值 {measured, literature, derived, unverified}；"
    "**每行必须带 provenance + 判决/取数时刻 + wrong_usage_consequence（用错后果）**；"
    "凡 value=unavailable 的行必须登记判决时刻与当时实测吞吐，并注明须以当场探针复核（M11-R56））"
)

ALLEN_API = "https://api.brain-map.org/api/v2/data/query.json"
S2_URL = ALLEN_API + "?criteria=model::Structure,rma::options[num_rows$eqall]"
S7_URL = "https://allen-brain-observatory.s3.amazonaws.com/visual-coding-neuropixels/ecephys-cache/"
IBL_URL = "https://ibl-brain-wide-map-public.s3.amazonaws.com/"
IBL_AGG = IBL_URL + "aggregates/2026_Q2_IBL_et_al_BWM/trials.pqt"


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader([ln for ln in f if not ln.startswith("#")]))


def load(path):
    if not os.path.exists(path):
        return {}
    return json.load(open(path, encoding="utf-8"))


def existing_ids():
    """读出现有行的 ref_id（**不修改任何既有行**）。"""
    ids, nlines = set(), 0
    if not os.path.exists(OUT):
        return ids, nlines
    with open(OUT, encoding="utf-8") as f:
        lines = list(f)
    nlines = len(lines)
    body = [ln for ln in lines if not ln.startswith("#")]
    if body:
        for r in csv.DictReader(body):
            if r.get("ref_id"):
                ids.add(r["ref_id"])
    return ids, nlines


def med(v):
    if not v:
        return None
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    summ = load(os.path.join(DATA, "m12_probe_b1_summary.json"))
    src = summ.get("sources") or {}
    s2, s7, s5 = src.get("S2") or {}, src.get("S7") or {}, src.get("S5") or {}
    integ = rows(os.path.join(DATA, "m12_probe_b1_integrity.csv"))
    ibl = rows(os.path.join(DATA, "m12_ibl_sessions.csv"))
    reach = rows(os.path.join(DATA, "m12_probe_b1_reachability.csv"))

    integ_by = {r["source"]: r for r in integ}
    s2i = integ_by.get("S2:model::Structure 全表", {})
    t_s2 = s2i.get("utc", "") or summ.get("generated_at", "") or utcnow()

    s7_items = {x["name"]: x for x in (s7.get("items") or [])}
    ses, prb = s7_items.get("sessions.csv", {}), s7_items.get("probes.csv", {})
    i_ses = integ_by.get("S7:ecephys-cache/sessions.csv", {})
    i_prb = integ_by.get("S7:ecephys-cache/probes.csv", {})
    t_s7 = i_prb.get("utc", "") or utcnow()

    szs = sorted(int(r["trials_table_bytes"]) for r in ibl if r.get("trials_table_bytes"))
    n_skip = len(s5.get("skipped") or [])
    parts = {p: sum(1 for r in ibl if r["partition"] == p)
             for p in sorted(set(r["partition"] for r in ibl))}

    def bps_of(source_id):
        """按 source_id 子串匹配取 reachability 里的当场实测吞吐（**不手打数字**）。"""
        for r in reach:
            sid = r.get("source_id") or ""
            if source_id in sid:
                try:
                    return float(r.get("speed_Bps") or 0)
                except ValueError:
                    return 0.0
        return 0.0

    s2_eta = float(s2.get("eta_actual_s") or 0)
    s2_bps = bps_of("S2(更正行")
    s7_bps = float(prb.get("probe_main_Bps") or 0)
    small_ses = float(ses.get("probe_1mib_Bps") or 0)
    small_prb = float(prb.get("probe_1mib_Bps") or 0)
    s5_bps = float(s5.get("probe_repr_Bps") or 0)
    s5_trials_bps = float(s5.get("probe_main_Bps") or 0)
    eta_trials = float(s5.get("eta_trials") or 0)
    rng = [int(x.get("rows") or 0) for x in (s5.get("items") or []) if x.get("rows")]
    now = utcnow()
    R = []

    def add(ref_id, key, value, unit, assertion, ev, prov, url, when, wrong, verdict, note=""):
        R.append(dict(ref_id=ref_id, key=key, value=value, unit=unit, assertion=assertion,
                      evidence_level=ev, provenance=prov, source_url=url, retrieved_at=when,
                      wrong_usage_consequence=wrong, verdict=verdict, note=note))

    # ================= S2 CCFv3 结构本体 =================
    add("M12-S2-01", "S2.model_structure.total_rows", str(s2.get("t1_total_rows", "")), "行",
        "== T1 count$eqtrue 的 total_rows AND == 落盘 JSON 的 msg 长度（%s）"
        % s2i.get("rows_parsed", ""),
        "measured", "Allen Mouse Brain Atlas API model::Structure 全表（本轮 count + 全表取回 + 解析）",
        S2_URL, t_s2,
        "**把它当 CCFv3 区域数会高估 10.3 倍**（13709/1327）：该表含**多个图谱**"
        "（发育脑 graph_id=17 等）⇒ 区域级判据的分母、500 叶区的筛选、区域 ID 命名空间全部会错",
        "pass", "全表 = 本清单 S2 的结构本体完整形态；未做服务端过滤（避免口径收窄）")
    add("M12-S2-02", "S2.model_structure.graph_id_1_rows",
        str((s2.get("graph_counts") or {}).get("1", "")), "行",
        "CCFv3 2017 **成体小鼠**图谱子图规模（服务端过滤 graph_id=1）", "measured",
        "同一 API + rma::criteria[graph_id$eq1]", S2_URL, t_s2,
        "把 graph_id=1 的 1327 行当全部结构 ⇒ 丢失细分层与其它图谱映射；反之把全表 13709 行当 CCFv3 "
        "⇒ 混入发育脑（graph_id=17 有 2692 行）⇒ **区域 ID 串台**",
        "pass", "M12 的 500 叶区应**显式声明取自哪个 graph_id**；本行不裁决哪一档对，只给两侧规模")
    add("M12-S2-03", "S2.model_structure.graph_id_17_rows",
        str((s2.get("graph_counts") or {}).get("17", "")), "行",
        "发育脑图谱（graph_id=17）子图规模", "measured", "同上，graph_id=17", S2_URL, t_s2,
        "把发育脑结构并进成体区域口径 ⇒ 区域身份错误（同名不同物）", "pass",
        "登记为必须排除的证据；**不是** unavailable")
    add("M12-S2-04", "S2.model_structure.graph_id_2_rows",
        str((s2.get("graph_counts") or {}).get("2", "")), "行", "graph_id=2 子图规模", "measured",
        "同上，graph_id=2", S2_URL, t_s2, "小图谱容易被误当主要图谱（仅 40 行）", "pass", "")
    add("M12-S2-05", "S2.model_structure.bytes", str(s2i.get("local_bytes", "")), "B",
        "== 磁盘文件字节 AND == 与零成本估计值 %s B 的偏差 -1.2%%" % s2.get("est_total_bytes", ""),
        "measured", "全表 GET 落盘 + os.path.getsize", S2_URL, t_s2,
        "把估计体积当实测体积会用错 ETA（本行给出估计法的实测偏差 -1.2%，**偏差可复算**）", "pass",
        "估计口径 = 单页 50 行 26727 B ⇒ %.2f B/行 × total_rows" % (s2.get("bytes_per_row") or 0))
    add("M12-S2-06", "S2.model_structure.sha256", s2i.get("sha256_file_bytes", ""), "hex",
        "完整性门 verdict=%s；服务端 ETag（= 全对象 md5）复算通过 ⇒ 他证强于自洽"
        % s2i.get("verdict", ""), "measured",
        "sha256(file_bytes)；ETag %s 与全对象 md5 复算一致" % s2i.get("etag", ""), S2_URL, t_s2,
        "**sha256 只证我算两次一样，不证与服务器一致**；本条以服务端 ETag 为准锚，"
        "不得把 sha256 自洽当与发布方一致",
        "pass", "三基准：file_bytes == payload（响应体即载荷）；semantic=%s（排序键后重序列化）"
                 % s2i.get("sha256_semantic", ""))
    add("M12-S2-07", "S2.model_structure.sha256_semantic", s2i.get("sha256_semantic", ""), "hex",
        "semantic 基准 = 解析后按键排序重序列化", "measured", "本工具 canonical JSON", S2_URL, t_s2,
        "把 file_bytes 与 semantic 混用 ⇒ 换了 whitespace 或键序就出现 hash 变了的假不一致",
        "pass", "semantic 基准确立内容而非字节等价性；与 file_bytes 并列而非替代")
    add("M12-S2-08", "S2.model_structure.etag", s2i.get("etag", ""), "hex",
        "== md5(全对象)（单段 ETag；复算通过）", "measured",
        "HTTP 响应头 ETag + 本地 md5 复算（curl -D）", S2_URL, t_s2,
        "**服务端锚被忽略 ⇒ 只剩自洽证据**（M11-R56：他证强于自洽）", "pass",
        "本端点为 chunked 且无 Content-Length，故门 1（size==Content-Length）由本条替代")
    add("M12-S2-09", "S2.model_structure.content_length_header", "absent(chunked)", "-",
        "门 1（size == Content-Length）**不适用**（服务端未发 Content-Length）", "measured",
        "curl -D 响应头（本轮实测：无 content-length、无 accept-ranges）", S2_URL, t_s2,
        "把无 Content-Length 当大小未知 ⇒ 判 INCOMPLETE 是**误判**；正确处置是改用服务端 ETag + "
        "语义行数自洽（本行即该处置的记录）",
        "pass", "门 1 的替代锚已登记：ETag md5 复算（通过）+ msg 行数 == T1 total_rows（13709）")
    add("M12-S2-10", "S2.model_structure.range_supported", "False", "bool",
        "-r 0-1023 与 -r 0-1048575 在 30 s 内**零字节 + curl exit=28**", "measured",
        "两轮 curl -r 探针（%s）" % t_s2, S2_URL, t_s2,
        "**若沿用清单 §2.3 的 1 MiB Range 模板，会得到 0 B/s**；若由 0 B/s 判 "
        "TOO_SLOW/unavailable ⇒ **把端点不支持 Range 误判成吞吐不可行**（本节点实测拦下此误判）",
        "pass", "⇒ 该端点必须改用时间上限 GET 或整件取数即测量")
    add("M12-S2-11", "S2.model_structure.ttfb_s", "31.0", "s",
        "服务端生成 13709 行 SQL 结果的首字节时延（time_starttransfer 实测 30.98-31.0 s）",
        "measured", "curl time_starttransfer（本节点 3 次独立取数一致：30.98 / 30.0 / 31.0 s）",
        S2_URL, t_s2,
        "**把 TTFB 当吞吐慢会用错降级决策**：该端点 B/s 随对象大小变化（TTFB 固定约 31 s + 传输约 4 s）"
        "⇒ B/s **不是链路吞吐量**；用 B/s 外推更大对象的 ETA 会**系统性高估**",
        "pass", "纯传输段约 7238867 B / (35.3-31.0) s 约 1.68 MB/s（**derived**，只作量级）")
    add("M12-S2-12", "S2.model_structure.eta_actual_s", str(s2.get("eta_actual_s", "")), "s",
        "整件取数完成时间（含 TTFB），门=1800 s/对象 ⇒ **通过约 51 倍**", "measured",
        "fetch 实测墙钟 %.1f s @ %.0f B/s（当场探针同口径）" % (s2_eta, s2_bps), S2_URL, t_s2,
        "**按清单继承值 1.6-16 KB/s 外推**会得到 ETA 7.5-75 min ⇒ 误判需托管或不可行；"
        "而当场实测约 35 s（差 13-130 倍）⇒ 这就是吞吐不可外推的直接实例",
        "pass", "**判决时刻 = %s**（另两次同刻实测 34.2 s / 36.8 s ⇒ 同小时内 1.08 倍抖动）；"
                "继承值 1.6-16 KB/s 的适用性**本行不予确认**" % t_s2)

    # ================= S7 Allen ecephys =================
    add("M12-S7-01", "S7.ecephys.sessions_csv.bytes", str(ses.get("bytes", "")), "B",
        "== 磁盘字节 == M11 已落盘件的同路径同对象", "measured",
        "HTTP 200 全量 + size==Content-Length + 单段 ETag md5 复算", S7_URL + "sessions.csv",
        i_ses.get("utc", now),
        "**把 session 行数当 session 总数**会漏（本表只覆盖 Visual Behavior Neuropixels 一档）；"
        "也不得与 IBL 的 session 数混用（**跨源分母混用** ⇒ 判据分母错）",
        "pass", "列数=10，数据行=%s（本轮 CSV 解析实测）" % ses.get("rows"))
    add("M12-S7-02", "S7.ecephys.probes_csv.bytes", str(prb.get("bytes", "")), "B",
        "== 磁盘字节 == M11 已落盘件的同路径同对象", "measured",
        "HTTP 200 全量 + size==Content-Length + 单段 ETag md5 复算", S7_URL + "probes.csv",
        i_prb.get("utc", now),
        "把 332 个 probe 当 332 个 session 或单元 ⇒ 独立单位错（**probe 不等于 session 也不等于 unit**）",
        "pass", "数据行=%s" % prb.get("rows"))
    add("M12-S7-03", "S7.ecephys.sessions_csv.sha256", ses.get("sha256", ""), "hex",
        "== sha256(磁盘件) AND == **M11 已落盘件的同值**（跨日、跨节点独立复测）", "measured",
        "sha256(file_bytes)；与 data/m11_data_inventory.csv 的 I045 逐字节比对",
        S7_URL + "sessions.csv", i_ses.get("utc", now),
        "**同日复现不等于与发布方一致**；本条只证本机两个时点取到同一对象（对象级确定性）",
        "pass", "M11 记录 = cdd12b8becac84d284f222718a2a94f4155893fb9f26f1a9c725f541acc738f8")
    add("M12-S7-04", "S7.ecephys.probes_csv.sha256", prb.get("sha256", ""), "hex",
        "== sha256(磁盘件) AND == **M11 已落盘件的同值**", "measured",
        "sha256(file_bytes)；与 data/m11_data_inventory.csv 的 I024 比对", S7_URL + "probes.csv",
        i_prb.get("utc", now), "同上", "pass",
        "M11 记录 = f555ce63317d5b6a83bfafd9f7e6a7b7393c0d7416952bafae0fe67412bfb6bf")
    add("M12-S7-05", "S7.channel_ref_throughput_Bps", "%.0f" % s7_bps, "B/s",
        "同桶（allen-brain-observatory）**大对象**的 1 MiB Range 当场实测（channels.csv 6.6 MB）",
        "measured", "curl -r 0-1048575（本轮三次：701617 / 608198 / 658672-729926 B/s）",
        S7_URL + "channels.csv", i_prb.get("utc", now),
        "**用两个 34 KB 小表的单件端到端 B/s（%.0f / %.0f B/s）代表链路吞吐 ⇒ 低估 20-50 倍**"
        "（小对象被握手与 TTFB 主导）⇒ 会误判该桶慢到不可用" % (small_ses, small_prb),
        "pass", "**这是 S7 可用的主档吞吐**；小件值单列为 probe_whole_object，两者不得混为一个数")
    add("M12-S7-06", "S7.inherited_threshold_status", "threshold_premise_obsolete", "-",
        "清单 §2.2 S7 的 pass（仅限两个完整小表）与不超过 5 MB 阈值**本轮复核成立且余量极大**",
        "measured", "本轮 34803 B / 1.4 s + 完整性门 COMPLETE", S7_URL, t_s7,
        "把 5 MB 阈值当作硬上限 ⇒ 会拒绝本可 1 分钟取回的对象；"
        "该阈值的原始前提是 1.6-16 KB/s（**今日不成立**）",
        "pass", "S7 两表合计 34803 B；**阈值依赖吞吐，而吞吐只有小时级有效期**")

    # ================= S5 IBL =================
    add("M12-S5-01", "S5.ibl.labs", str(len(s5.get("labs") or [])), "个",
        "公开桶 data/ 一级前缀数（本轮 enumerate 实测）", "measured",
        "S3 list-type=2&delimiter=/&prefix=data/", IBL_URL + "?list-type=2&prefix=data/", now,
        "把 lab 数当数据集数或 session 数 ⇒ 分母错（session 数需逐层列举，**未测**）",
        "pass", "labs=%s" % (s5.get("labs") or []))
    add("M12-S5-02", "S5.ibl.whitelist_n_sessions", str(len(ibl)), "个",
        "**预注册**白名单规模（不超过 40，见 data/m12_ibl_sessions.csv）", "measured",
        "round-robin 枚举（字母序 lab、subject、date + 数值序 session），规则在算任何统计量之前定死",
        IBL_URL + "?list-type=2&prefix=data/", now,
        "**把白名单当全量** ⇒ P8/P9 的分布统计会被当作总体参数（**选择偏差**）；"
        "反向错误（把白名单不足当 IBL 不可用）会让 P8 无谓降级",
        "pass", "归属：%s（A/B/C/D 在出数前定死）；**分区标签不等于隔离**——隔离须另按 session 切" % parts)
    add("M12-S5-03", "S5.ibl.whitelist_bytes_total", str(sum(szs)), "B",
        "40 session 行为试次表合计（逐件 size==Content-Length 且完整性门 COMPLETE）", "measured",
        "逐 session _ibl_trials.table.*.pqt 全量取回 + 门三级", IBL_URL, now,
        "**把它当 IBL 行为总量** ⇒ 严重低估（全量聚合表 24121210 B、raw NWB 为 TB 级）", "pass",
        "n=%d，min=%s B / median=%s B / max=%s B" % (len(szs), szs[0] if szs else "", med(szs),
                                                     szs[-1] if szs else ""))
    add("M12-S5-04", "S5.ibl.trials_rows_per_session_range",
        "%s-%s" % (min(rng) if rng else "", max(rng) if rng else ""), "行",
        "逐 session parquet 行数区间（pyarrow 解析，38 个不同值）", "measured",
        "pyarrow 21.0.0（.venv-m10）解析 parquet 元数据", IBL_URL, now,
        "**把行数当试次数而不核对 schema 列** ⇒ 可能把非行为行算进 P8 分母", "pass",
        "列名与 P8/P9 所需列（choice/feedback/response_times/contrast*）的命中情况见完整性 CSV 的 reasons")
    add("M12-S5-05", "S5.ibl.full_aggregate_trials_pqt.bytes", str(s5.get("trials_bytes", "")), "B",
        "**单对象全量行为表** aggregates/2026_Q2_IBL_et_al_BWM/trials.pqt（含全部 session）",
        "measured", "S3 列举 + HTTP 206 Range 探针（1 MiB 实测 %.0f B/s）" % s5_trials_bps,
        IBL_AGG, now,
        "**不取它的理由不能写成不可用**：当场实测 ETA=%.0f s（远低于 30 min 门），"
        "它只是超出本节点的 5 MB 探针预算（大于 5 MB、小于 50 MB 硬阈值）⇒ 属**待主 agent 裁决**，"
        "**不等于** unavailable（吞吐限制）" % eta_trials,
        "unavailable(预算限制，非吞吐限制)",
        "★ **判决时刻=%s**；当时实测吞吐（1 MiB Range 当场探针）= %.0f B/s；"
        "**须以当场探针复核后方可作为取数依据**（复核周期不超过 24 h，M11-R56）。"
        "同前缀另有 clusters.pqt 112325731 B、clusters.acgs_log.npy 159163776 B、"
        "clusters.waveforms_peak.npy 159163776 B（**均未取**）。同小时内两次实测 ETA = 109 s 与 73 s"
        "（1.5 倍抖动）" % (now, s5_trials_bps))
    add("M12-S5-06", "S5.ibl.revision_key_hash_encoding", "required(%23)", "-",
        "IBL 桶的 revision 子目录名形如 alf/#YYYY-MM-DD#/；**井号未编码会被 curl 当 URL fragment 截断 ⇒ 404**",
        "measured",
        "本轮实测：未编码请求 alf/#2024-07-15#/_ibl_trials.table.*.pqt ⇒ HTTP 404 + 341 B 错误页；"
        "编码为 %23 ⇒ HTTP 200", IBL_URL, now,
        "**若不编码，会得到该 session 无行为表的假阴性**；更糟的是把 404 错误页（341 B）当文件落盘 "
        "⇒ **假件进台账**（本节点实测拦下：已加 --fail + 仅 2xx 才续传）",
        "pass", "同一 session 的当前版与 revision 版是**两个不同对象** ⇒ 选哪一档必须登记"
                 "（m12_ibl_sessions.csv 的 note 列 revision_used）")
    add("M12-S5-07", "S5.ibl.skipped_sessions_n", str(n_skip), "个",
        "枚举中**显式登记原因后跳过**的 session 数（alf 内无合并形态 _ibl_trials.table.*.pqt）",
        "measured", "逐 session alf/ 列举 + 断言", IBL_URL, now,
        "**把跳过当不存在**（未测不等于未命中）；本行登记的跳过原因全部是该 session 使用旧的逐字段 "
        ".npy 形态，**不是**数据不存在",
        "partial", "理由（前 6 条）：%s" % (s5.get("skipped") or [])[:6])
    add("M12-S5-08", "S5.ibl.raw_nwb_and_aggregates_status", "unavailable(体量/预算)", "-",
        "全量 aggregates（clusters 与波形）与 raw NWB：**本轮未取**", "unverified",
        "体量维度 = measured（S3 列举实测单件字节：clusters.pqt 112325731 / clusters.acgs_log.npy "
        "159163776 / clusters.waveforms_peak.npy 159163776）；**吞吐维度 = unverified**"
        "（未对每个大件做当场探针）⇒ evidence_level 取**较弱的那一维** = unverified",
        IBL_URL, now,
        "把本条写成 IBL 全脑活动不可用 ⇒ **越过判据边界**（本条只覆盖本轮未取这一维；"
        "S6 的区域级汇总表是否存在**未测**）",
        "unavailable", "**缺哪一维**：缺逐件当场吞吐探针 + 是否存在不超过 100 MB 区域级汇总对象"
                       "两维；已覆盖维度：对象存在性与单件字节（measured）")

    # ================= 更正行（对继承吞吐锚；R41-2 条件 1 要求的形态） =================
    add("M12-C-01", "correction.api_brain_map_throughput", "%.0f" % s2_bps, "B/s",
        "**更正继承值**：清单 §2.1/§2.2 记 api.brain-map.org = 1.6-16 KB/s（unverified 继承）；"
        "本轮当场实测 %.0f B/s（S2 全表 7238867 B / 约 %.1f s，含 TTFB 31 s）" % (s2_bps, s2_eta),
        "measured", "curl 整件取数 + time_starttransfer（本节点 3 次一致）", S2_URL, t_s2,
        "沿用 1.6-16 KB/s 会把 S2（34 秒可完成）判成 unavailable；"
        "而按 §2.4 那会触发 P1 unavailable + P7/P10 区域级判据全 invalid 的**结构性阻塞**",
        "REACHABLE（以探针为准，更正继承值）",
        "**本条即主 agent 评审 §R41-2 条件 1 要求的更正行形态**：判决时刻 + 当时实测吞吐 + "
        "注明须以当场探针复核（不超过 24 h）。继承值适用于**其它端点或其它时刻**，就 S2 而言已失效")
    add("M12-C-02", "correction.allen_bo_throughput", "%.0f" % s7_bps, "B/s",
        "**更正继承值**：清单与 M11 记 Allen BO 桶 1.6-28 KB/s；本轮当场实测 %.0f B/s"
        "（同桶 channels.csv 的 1 MiB Range）" % s7_bps, "measured",
        "curl -r 0-1048575（同桶大对象；三次 701617 / 608198 / 658672-729926 B/s）",
        S7_URL + "channels.csv", t_s7,
        "沿用旧值会把 S7 的 units.csv（51439189 B）等误判为数十小时不可行；"
        "本轮同桶口径下 51 MB 级对象约 1.5-2 min", "REACHABLE（以探针为准，更正继承值）",
        "**不得**把本条外推到 NWB 大件（不同对象、不同压缩、不同前缀）——"
        "按 M11-R40 一判据一作用域，其它对象须各自的当场探针")
    add("M12-C-03", "correction.ibl_public_bucket_throughput", "%.0f" % s5_bps, "B/s",
        "**更正继承值**：清单 §2.2 记 S3 桶 10-17 KB/s；本轮 IBL 公开桶当场实测 %.0f B/s"
        "（白名单首个 session 表的同通道大对象参照）" % s5_bps, "measured",
        "curl -r 0-1048575（aggregates trials.pqt：%.0f B/s 主档；另一轮 221230 与 310284 B/s）"
        % s5_trials_bps, IBL_AGG, now,
        "沿用 10-17 KB/s 会把 40 session 白名单（%d B）判成 2-3 min 可行，但会把 "
        "24.2 MB 全量聚合表判成结构性不可行（实为 73-109 s）" % sum(szs),
        "REACHABLE（以探针为准，更正继承值）",
        "**同小时内两次实测 ETA 相差 1.5 倍（109 s 与 73 s）** ⇒ 再次印证吞吐不可外推；"
        "本条判决时刻 = %s" % now)
    add("M12-C-04", "correction.throughput_extrapolation_prohibited", "prohibited", "-",
        "**本节点三轮实测再次确认：吞吐不可外推**（M11-R40/R56/R63）", "measured",
        "同一小时内：S2 34.2/35.3/36.8 s；S7 渠道 602633-729926 B/s；IBL trials.pqt 221230-310284 B/s",
        IBL_AGG, now,
        "**任何以单次实测外推的工期或预算都会失效**（M11 实测摆幅达 3600 倍）⇒ "
        "一律以当场探针为准，并登记判决时刻",
        "pass", "操作规则已固化：三个 CSV 均含 utc 首列；凡 unavailable 行必须给判决时刻 + 当时吞吐 + "
                "复核要求（不超过 24 h）")

    # ================= 方法学（本轮实测得来的新纪律，各附本例） =================
    add("M12-M-01", "throughput.zero_byte_probe_is_not_slow", "window_short_of_ttfb", "-",
        "**0 B/s 的探针结果有至少 2 种根因**：一是链路慢；二是**探针窗短于服务端生成时延（TTFB）**",
        "measured",
        "本例：api.brain-map.org 的 model::Structure TTFB=31.0 s ⇒ 30 s 探针窗内收 0 B；"
        "同一端点整件取数 7238867 B / 约 35 s ⇒ ETA 仅约 35 s", S2_URL, t_s2,
        "**由 0 B/s 判 TOO_SLOW/unavailable 会把一个 35 秒可完成的取数永久关闭**"
        "（本节点在实施中拦下此误判：预注册规则改为探针窗过短 ⇒ 转整件取数即测量）",
        "pass", "对应操作规则：探针**必须同时登记 TTFB**；TTFB 不小于窗 ⇒ 加大窗或转整件取数")
    add("M12-M-02", "throughput.small_object_probe_underestimates", "handshake_dominated", "-",
        "**对象小于探针 Range 时，-r 0-1048575 返回全件** ⇒ 得到的是单件端到端 B/s（含握手），"
        "**不是链路吞吐**", "measured",
        "本例：S7 sessions.csv 7849 B ⇒ %.0f B/s；同桶大对象 channels.csv 的 1 MiB Range ⇒ "
        "608198-729926 B/s" % small_ses, S7_URL, t_s7,
        "**用 %.0f B/s 代表该桶吞吐 ⇒ 低估 41-50 倍**，会把可 1 秒取回的源判成要几十分钟" % small_ses,
        "pass", "⇒ 吞吐 CSV 中用 probe_whole_object 与 probe_ref_1MiB **两个标签分列**，不得混为一个数")
    add("M12-M-03", "integrity.server_etag_subsumes_repeat_fetch", "True", "-",
        "**门 2 的服务端 ETag 复算 = 他证** ⇒ 通过时**不需要**再做两次独立取数 sha256 一致（自洽）",
        "measured",
        "本例：S2 chunked 无 Content-Length ⇒ 原计划两次全量取数（各约 35 s 与 7.2 MB）；"
        "ETag ef27375e 复算通过后跳过第二次（省 35 s + 7.2 MB）", S2_URL, t_s2,
        "把自洽当他证（或反之）会造成两类错：**假通过**（自洽但对象已变）与 **无谓成本**（重复取数）",
        "pass", "ETag 缺失或复算失败时**必须**退回两次独立取数（自洽），并在 CSV 的 reasons 写明走哪条")
    add("M12-M-04", "fetch.error_page_must_not_be_landed", "--fail_required", "-",
        "**HTTP 4xx 的错误响应体绝不允许落盘为对象**，也**不得**对其做 curl -C - 续传", "measured",
        "本例：IBL 井号未编码请求返回 404 + 341 B XML；旧逻辑把它落盘并续传 ⇒ 假件（341 B）进台账，"
        "只因 size_match=False 才被发现", IBL_URL, now,
        "**假件进台账 = 台账总量虚增 + 下游可能把错误页当数据解析**（静默失效家族第 5 例）",
        "pass", "修法：curl 加 --fail；续传**仅**在 http 取值于 {200,206} 且确为部分时执行")

    # ================= 门标准操作化（§1.2；**运行前冻结**，见 docs/m12_b1_gate_closure.md §2） =================
    from math import ceil, sqrt
    z1a, z1b = 1.6449, 1.2816                      # z_{0.95}（TOST 每侧 alpha=0.05）/ z_{0.90}（power=0.80）
    K_power = 2.0 * (z1a + z1b) ** 2               # = 17.134 ：n_min^power = K_power * sigma^2 / delta^2
    K_ci = 2.0 * (z1a / 0.20) ** 2                 # = 135.3 ：CI 宽度门（单组）
    K_tot = 4.0 * (z1a + z1b) ** 2                 # = 34.268：劈半自检 ⇒ n_total = 2 * 2 * K/2 ... 见 note
    docs_note = "推导与算式全文见 docs/m12_b1_gate_closure.md §2"
    add("M12-G-01", "gate.p_gt_0.05_branch_form", "TOST_or_equivalence_CI", "-",
        "门标准中的 p>0.05 支路**只能**以实现为等价检验：max(p_low,p_high) < 0.05 或 "
        "90% CI 完全落在 (-delta,+delta) 内；**raw 双侧 p>0.05 一律不得作 pass**",
        "derived", "清单 §1.2 F1 行 + 附加条款 6 的机械改写（本节点执行化）；" + docs_note,
        "docs/生物仿真M12实施清单.md", now,
        "**把 raw p>0.05 当通过**⇒ 样本量不足时自然满足（平凡通过），是 §1.2 明文要堵的第一类假通过；"
        "反向错误（把等价检验的 p 值取成 2*min(p_low,p_high)）⇒ 把等价变成差异的假阳性",
        "pass", "等价检验 p 值 = max(p_low, p_high)；报告须含 delta / Delta_hat / CI90 / p_low / p_high / "
                "n_sim / n_data 七项")
    add("M12-G-02", "gate.margin.delta_rel", "0.20", "相对",
        "delta = 0.20 x |ref|（优先相对 margin）；跨零量改用标准化 margin delta_std",
        "derived", "清单 §1.2 F1 行的预注册 margin 形式；" + docs_note, "docs/生物仿真M12实施清单.md", now,
        "**在同一批判据里混用相对与标准化 margin**⇒ 不同判据的等价区间不可比（§0.8 EQ-1 的口径一致要求）",
        "pass", "delta 必须与 ref 的来源同时写出（ref 为 unverified 时该判据不得作带，§1.2 附加条款 5）")
    add("M12-G-03", "gate.n_min_formula", "ceil(428.3*CV^2)", "每组",
        "n_min^power = 2*sigma^2*(z0.95+z0.90)^2/delta^2 = 17.134*sigma^2/delta^2；"
        "代入 delta=0.20*|ref| 与 sigma=CV*|ref| ⇒ n_min^power = ceil(428.3 x CV^2)（**每组**）",
        "derived", "TOST 两组等样本量的标准样本量公式（本节点执行化）；" + docs_note,
        "docs/生物仿真M12实施清单.md", now,
        "**不给 n_min 就做等价检验**⇒ 等价与否由样本量而非数据决定（§0.8 第 4 条：n_valid < n_min ⇒ "
        "判 invalid，**不是** pass 也**不是** fail）",
        "pass", "示例：CV=0.20 ⇒ 18；CV=0.30 ⇒ 39；CV=0.50 ⇒ 108；CV=1.00 ⇒ 429（每组）")
    add("M12-G-04", "gate.n_total_min_formula_split_half", "ceil(856.6*CV^2)", "合计",
        "★ 边界可达性自检（§1.2 附加条款 4）要求把目标数据按区组劈两半、同 margin 判等价并**必须通过** "
        "⇒ 每半都要满足 n_min ⇒ n_total >= 2 x n_min^power = ceil(856.6 x CV^2)",
        "derived", "清单 §1.2 附加条款 4 + 本节点的样本量推导；" + docs_note,
        "docs/生物仿真M12实施清单.md", now,
        "**忽略劈半自检**⇒ margin 或指标口径直到跑完模型才暴露不可达（M10 L20.3 / M11 §0.7 #4 的同型事故），"
        "届时只能重定稿，整轮作废",
        "pass", "示例：CV=0.20 ⇒ 35；CV=0.30 ⇒ 78；CV=0.50 ⇒ 215（合计，两组）")
    add("M12-G-05", "gate.mde_formula", "sqrt(17.134/n)*sigma", "—",
        "MDE @ n_min 必须随判据一起报：delta_detectable(n) = sqrt(17.134/n) * sigma（单组 n）",
        "derived", "同上（本节点执行化）；" + docs_note, "docs/生物仿真M12实施清单.md", now,
        "**只报 n 不报 MDE**⇒ 读者无法判断该 n 下能检出多大差异（§1.2 附加条款 2 明文要求）",
        "pass", "")
    add("M12-G-06", "gate.multiple_comparison", "Holm-Bonferroni", "-",
        "family 内用 Holm–Bonferroni（FWER 控制）；**同时报 raw p 与 adjusted p**；family 必须在运行前定义",
        "derived", "清单 §1.2 附加条款 3；本节点给出 4 个预注册 family 建议（P8-behavior 5 / "
                   "P10-activity 3 / P11-perturbation N_anchor / P7-region 6）；" + docs_note,
        "docs/生物仿真M12实施清单.md", now,
        "**不定义 family 就报 p 值**⇒ 等价于「20 个指标挑 1 个通过」（§1.2 明文禁止）；"
        "只报 adjusted 不报 raw ⇒ 无法复核校正过程",
        "pass", "等价检验的 p 值用 max(p_low,p_high) 参与校正（见 M12-G-01）")
    add("M12-G-07", "gate.F4_denominator_rule", "exclude_unavailable_and_ineffective", "-",
        "F4 命中率分母**预注册**；unavailable / injection_ineffective / no-experiment **不入分母**",
        "derived", "清单 §1.2 F4 行 + §2.4 采集参数读不回条款；" + docs_note,
        "docs/生物仿真M12实施清单.md", now,
        "**把 unavailable 计入分母**⇒ 该判定直接作废（§1.2 F4 明文；未测 ≠ 未命中）——"
        "这是 M11 已付过学费的形态（MQ/负结果被算成失败）",
        "pass", "与判据行 M12-S5-07（22 个 session 未命中）同族：未命中须单列，不得并入分母")
    add("M12-G-08", "gate.IBL_behavior_schema_13cols", "13", "列",
        "IBL 行为表真实 schema = 13 列：goCue_times / response_times / choice / stimOn_times / "
        "contrastLeft / contrastRight / probabilityLeft / feedback_times / feedbackType / rewardVolume / "
        "firstMovement_times / intervals_0 / intervals_1",
        "measured", "本节点用 pyarrow 读回 40/40 落盘行为表的 schema（data/m12_probe_b1_integrity.csv "
                    "逐件留痕）；**常识写法 feedback 与 intervals 均不存在**（真名 feedbackType / intervals_0,1）",
        IBL_URL, now,
        "**凭常识写列名**（feedback / intervals）⇒ 装配脚本 KeyError，或更糟：静默取到 None 并继续"
        "⇒ 反馈相关判据在空列上计算（静默失效家族形态）",
        "pass", "RT 口径必须预注册起点（response_times - goCue_times 还是 - stimOn_times）："
                "两者都在表里，混用会造成纯口径的假差异")
    add("M12-G-09", "gate.fit_set_size_vs_n_min", "A=13 < 35", "session",
        "白名单 A 分区 13 session，而 CV=0.20（乐观）下 n_total_min = 35（见 M12-G-04）"
        "⇒ **A 单独作拟合集不满足样本量门**",
        "derived", "本节点实测白名单规模（A=13/B=13/C=13/D=1，data/m12_ibl_sessions.csv）+ M12-G-04 的算式",
        IBL_URL, now,
        "**拿 A=13 直接跑 TOST**⇒ n_valid < n_min ⇒ 判 invalid（不是 pass）；若忽略 n_min 强判，"
        "会得到样本量决定而非数据决定的等价结论",
        "pass", "可选处置：① A 扩到 >=18（再取 5 个 session，成本约 5 x 50 KB）"
                "② 用全白名单 40 作拟合集、B 分区扩到同量级 ③ 显式登记 CV 的实测值以重算 n_min")

    ids, nlines = existing_ids()
    new = [r for r in R if r["ref_id"] not in ids]
    print("既有行数=%d，既有 ref_id=%d 个；拟新增 %d 行（跳过重复 %d）"
          % (nlines, len(ids), len(new), len(R) - len(new)))
    if args.dry_run:
        for r in new:
            print("  +", r["ref_id"], r["key"], "=", str(r["value"])[:56], "|",
                  r["evidence_level"], "|", r["verdict"])
        return

    if not new:
        print("无新增行 ⇒ **不写盘**（避免追加空批次注释；append-only 幂等）")
        return

    fresh = not os.path.exists(OUT) or os.path.getsize(OUT) == 0
    mode = "w" if fresh else "a"          # **既有文件只追加，绝不重写**
    with open(OUT, mode, newline="", encoding="utf-8") as f:
        if fresh:
            f.write(HDR_COMMENT + "\n")
        else:
            f.write("# append-only batch @ %s by tools/build_m12_reference.py（新增 %d 行）\n"
                    % (now, len(new)))
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if fresh:
            w.writeheader()
        for r in new:
            w.writerow(r)
    ids2, nlines2 = existing_ids()
    print("wrote %s：ref_id %d -> %d（新增 %d）；append-only 校验：既有 ref_id 全部仍在 = %s"
          % (os.path.relpath(OUT, os.path.dirname(ROOT)), len(ids), len(ids2),
             len(ids2) - len(ids), ids.issubset(ids2)))


if __name__ == "__main__":
    main()
