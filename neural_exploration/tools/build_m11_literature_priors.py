#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M11 数据门（G2/P1/P2）· 文献 provenance 化器

把《生物仿真M11实施清单.md》**§1.3（文献先验）与 §4.1（C-1..C-8 候选依据）**里的每一条
可核查标识（PMID/DOI）**逐条**送到 Europe PMC REST 复核，并输出 `data/m11_literature_priors.csv`。

纪律（清单 §1.3 写作纪律 + §4.1 "依据诚实性说明"）
--------------------------------------------------
1. **命中状态只有两档**：`literature`（本轮 Europe PMC 实测命中，题录可核查）
   / `literature_unverified`（**未命中** —— 必须写明查询式与 hitCount，**禁止凭印象补**）。
2. `literature` **不等于**"全文/参数值已核实"：列 `evidence_level` 区分
   `题录+摘要可得`（Europe PMC 返回 abstractText）与 `题录only`；参数值是否在摘要文本中出现
   由 `value_in_abstract` 列如实标注（`yes` / `no` / `not_checked`）。
3. **值或范围**列：来自清单原文（不改写）；若摘要中未出现，note 明确写"值未在摘要中出现（全文未读）"。
4. `literature_unverified` 的行 **不得**作为参数范围唯一依据（清单 §4.1 明示）。

用法
----
    cd /Users/weidong/ai/small_world
    PYTHONHASHSEED=0 ./.venv-m11/bin/python neural_exploration/tools/build_m11_literature_priors.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "m11_raw", "literature")
OUT_CSV = os.path.join(DATA, "m11_literature_priors.csv")
OUT_JSON = os.path.join(DATA, "m11_literature_priors.json")
LOG = os.path.join(RAW, "build_m11_literature_priors.log")
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

# ---------------------------------------------------------------------------
# 行清单：clause=清单条款 / mechanism=机制或参数 / value=清单原文的值或范围
# ident=清单声明的标识（PMID/DOI，空=清单未给标识）/ query=复核用查询串
# value_regex=用于判断"值是否出现在摘要中"的宽松正则（可空）
# ---------------------------------------------------------------------------
PRIORS = [
    # ---------- §1.3 文献先验 ----------
    dict(pid="P-1.3-01", expect=["neocortical", "local", "circuitry"], clause="§1.3 行1", mechanism="皮层局部微环路连接概率（层间/类间）",
         value="连接概率先验（论文表格）", ident="PMID:18982117",
         query='AUTH:"Thomson AM" AND "neocortical" AND "local circuitry"',
         note_claim="清单记为命中（hitCount=3）"),
    dict(pid="P-1.3-02", expect=["connectivity", "morphologically", "defined", "cell", "types"], clause="§1.3 行2 / §4.1 C-4", mechanism="形态定义细胞类型间连接原理（PV/SST/VIP 类间概率与方向）",
         value="类间连接概率与方向", ident="PMID:26612957",
         query='TITLE:"Principles of connectivity among morphologically defined cell types in adult neocortex"',
         note_claim="清单记 TITLE 查询 hitCount=4（含 Comment PMID 27609882 / Response 27609883）"),
    dict(pid="P-1.3-03", expect=["nonrandom", "synaptic", "connectivity", "local", "cortical"], clause="§1.3 行3", mechanism="局部皮层电路高非随机连接（聚类）",
         value="连接非随机性（聚类）先验", ident="PMID:15737062",
         query='TITLE:"Highly nonrandom features of synaptic connectivity in local cortical circuits"',
         note_claim=""),
    dict(pid="P-1.3-04", expect=["reconstruction", "simulation", "neocortical", "microcircuitry"], clause="§1.3 行4 / §4.1 C-1", mechanism="NMC 数字重建（31k / ~8M 突触）层流模板 + 适应参数",
         value="层流模板 + 适应参数 + 突触动力学先验", ident="PMID:26451489",
         query='TITLE:"Reconstruction and Simulation of Neocortical Microcircuitry"',
         note_claim="清单记 hitCount=1"),
    dict(pid="P-1.3-05", expect=["local", "connectivity", "synaptic", "dynamics", "mouse", "human"], clause="§1.3 行5 / §4.1 C-2/C-3", mechanism="Allen synphys：mouse/human 皮层连接与突触动力学",
         value="突触恢复/动力学参数先验（τ_rec 可达数秒）", ident="PMID:35271334",
         query='TITLE:"Local connectivity and synaptic dynamics in mouse and human neocortex"',
         note_claim="Allen synphys 桶对象级不可 list（见可达性表 #32/F16）→ 论文档为回退路径"),
    dict(pid="P-1.3-06", expect=["inhibition", "logic", "connections", "interneurons"], clause="§1.3 行6 / §4.1 C-4", mechanism="抑制性微回路去抑制逻辑（VIP→SST）",
         value="去抑制回路方向与强度", ident="PMID:23817549",
         query='TITLE:"Inhibition of inhibition in visual cortex: the logic of connections between molecularly distinct interneurons"',
         note_claim=""),
    dict(pid="P-1.3-07", expect=["disinhibitory", "circuit", "motor", "integration"], clause="§1.3 行7 / §4.1 C-4", mechanism="去抑制回路介导运动整合（SST/VIP）",
         value="运动→VIP→SST→E 因果链", ident="PMID:24097044",
         query='TITLE:"A disinhibitory circuit mediates motor integration in the somatosensory cortex"',
         note_claim=""),
    dict(pid="P-1.3-08", expect=["rhythmic", "recurrent", "activity", "neocortex"], clause="§1.3 行8 / §4.1 C-5 / §1.5 C3", mechanism="皮层节律性递归活动的细胞/网络机制（慢振荡 up/down dwell）",
         value="**文献档（非判据带）**：up/down dwell 0.3–3 s 量级",
         ident="PMID:11017176",
         query='TITLE:"Cellular and network mechanisms of rhythmic recurrent activity in neocortex"',
         value_regex=r"0\.3|3 s|\b\d(\.\d+)?\s*(s|sec)", note_claim=""),
    dict(pid="P-1.3-09", expect=["wakefulness", "local", "cortical", "networks"], clause="§1.3 行9 / §4.1 C-7", mechanism="觉醒状态对局部皮层网络的效应",
         value="状态依赖慢调制（分钟级觉醒态）", ident="PMID:21435553",
         query='TITLE:"Effects and mechanisms of wakefulness on local cortical networks"', note_claim=""),
    dict(pid="P-1.3-10", expect=["neuromodulation", "brain", "state", "behavior"], clause="§1.3 行10 / §4.1 C-7", mechanism="神经调质与脑状态",
         value="慢调质状态量（τ 数秒–数十秒）", ident="PMID:32250724",
         query='TITLE:"Neuromodulation of brain state and behavior"', note_claim=""),
    dict(pid="P-1.3-11", expect=["inhibition", "membrane", "potential"], clause="§1.3 行11 / §4.5(e)", mechanism="抑制决定膜电位动力学并控制发放（清醒/睡眠猫皮层）",
         value="抑制主导方向（E/I 平衡带 [0.5,3.0] 的方向性依据）", ident="PMID:17507551",
         query='TITLE:"Inhibition dominates sensory responses in the awake cortex"', note_claim=""),
    dict(pid="P-1.3-12", expect=["fast-spiking", "gamma", "sensory", "responses"], clause="§1.3 行12 / §0.6 H3 / P10", mechanism="γ 振荡的 PV 机制归属（驱动 FS 细胞诱导 γ）",
         value="PV 驱动 → γ 峰（消融归因锚）", ident="PMID:19396156",
         query='TITLE:"Driving fast-spiking cells induces gamma rhythm and controls sensory responses"', note_claim=""),
    dict(pid="P-1.3-13", expect=["interneuron"], clause="§1.3 行13", mechanism="γ/synphys 检索集（PV+SST+VIP+visual cortex）",
         value="P8 类型带（具体条目待 provenance 化）", ident="",
         query='("parvalbumin" AND "somatostatin" AND "VIP" AND "visual cortex") AND SRC:MED', note_claim="清单记 HTTP 200 / 2,068 B"),
    dict(pid="P-1.3-14", clause="§1.3 行14 / §1.2", mechanism="Allen Visual Coding Neuropixels 数据集论文题录",
         value="P7/P8/P14 电生理带", ident="",
         query='TITLE:"Survey of spiking in the mouse visual cortex"',
         note_claim="清单记该 TITLE 查询 hitCount=0（题录未命中 ≠ 不存在）→ 本轮复核 + 换查询式"),
    dict(pid="P-1.3-15", clause="§1.2 / §1.3", mechanism="Allen Visual Coding Neuropixels 论文（替代查询式）",
         value="P7/P8/P14 电生理带 provenance", ident="",
         query='AUTH:"Siegle" AND TITLE:"functional hierarchy" AND SRC:MED', note_claim="本轮新增的替代查询式（不臆造 PMID）"),
    dict(pid="P-1.3-16", clause="§1.4", mechanism="IBL 行为协议论文（决策任务标准化）",
         value="P12 行为协议 provenance（正确率/RT/psychometric）", ident="",
         query='TITLE:"Standardized and reproducible measurement of decision-making in mice" AND SRC:MED AND NOT TITLE:"Correction"',
         note_claim="清单记 HTTP 200 / 2,867 B，**具体 PMID 待 G2 落盘** → 本轮必须落 PMID"),
    # ---------- §4.1 C-1..C-8 依据 ----------
    dict(pid="C-1-a", expect=["reconstruction", "simulation", "neocortical", "microcircuitry"], clause="§4.1 C-1", mechanism="多时间尺度发放适应（I_AHP/I_M）作为慢变量",
         value="τ_ad ∈ G-fast [0.2,3] s；g_ad ∈ [0,0.2] mS/cm²；≥2 个 τ",
         ident="PMID:26451489", query='TITLE:"Reconstruction and Simulation of Neocortical Microcircuitry"',
         note_claim="NMC 适应参数来源"),
    dict(pid="C-1-b", expect=["rhythmic", "recurrent", "activity", "neocortex"], clause="§4.1 C-1", mechanism="适应电流与 up/down 态时间尺度",
         value="G-fast 内可达；单靠 C-1 到不了 G-slow（清单标为预注册假设，未实测）",
         ident="PMID:11017176", query='TITLE:"Cellular and network mechanisms of rhythmic recurrent activity in neocortex"',
         note_claim=""),
    dict(pid="C-2-a", expect=["local", "connectivity", "synaptic", "dynamics", "mouse", "human"], clause="§4.1 C-2", mechanism="突触抑制恢复时间常数（逐类 PV/SST/VIP，上界扩到 10 s）",
         value="τ_rec^PV ∈[0.1,3] s；τ_rec^SST/VIP ∈[0.1,10] s",
         ident="PMID:35271334",
         query='TITLE:"Local connectivity and synaptic dynamics in mouse and human neocortex"',
         note_claim="清单明确：该上界扩展依据 = Campagnola 2022；**10 s 上界是否在文献内 → 必须核对摘要/表**"),
    dict(pid="C-2-b", expect=["inhibitory", "synap"], clause="§4.1 C-2", mechanism="抑制性突触恢复时间常数量级（独立检索，防单源依赖）",
         value="τ_rec 量级复核", ident="",
         query='("recovery time constant" OR "recovery from depression") AND "inhibitory synapse" AND ("neocortex" OR "cortex")',
         note_claim="本轮新增（清单未给该项的独立第二源）"),
    dict(pid="C-3-a", expect=["local", "connectivity", "synaptic", "dynamics", "mouse", "human"], clause="§4.1 C-3", mechanism="GABA_B 样慢抑制组分（受体级）",
         value="τ_slow ∈ [0.1,3] s；权重占比 ∈ [0.05,0.5]", ident="PMID:35271334",
         query='TITLE:"Local connectivity and synaptic dynamics in mouse and human neocortex"',
         note_claim="清单的主要依据是 **M10 自身实测**（GABA_B 只改水平不产生双态）→ 文献为辅助"),
    dict(pid="C-4-a", expect=["inhibition", "logic", "connections", "interneurons"], clause="§4.1 C-4", mechanism="VIP→SST 去抑制慢门控",
         value="τ_gate ∈ [0.2,10] s；g_vip→sst ∈ [0,1]；SST→E τ_inh ∈ [5,25] ms",
         ident="PMID:23817549",
         query='TITLE:"Inhibition of inhibition in visual cortex: the logic of connections between molecularly distinct interneurons"',
         note_claim="**关键**：τ_gate∈[0.2,10] s 与 τ_inh∈[5,25] ms 的文献支持需核对"),
    dict(pid="C-4-b", expect=["disinhibitory", "circuit", "motor", "integration"], clause="§4.1 C-4", mechanism="去抑制回路介导运动整合",
         value="三阶因果链（VIP 群发放→SST 群抑制→E 群发放）", ident="PMID:24097044",
         query='TITLE:"A disinhibitory circuit mediates motor integration in the somatosensory cortex"', note_claim=""),
    dict(pid="C-4-c", expect=["connectivity", "morphologically", "defined", "cell", "types"], clause="§4.1 C-4", mechanism="类间连接（VIP→SST）方向与强度",
         value="连接概率/方向", ident="PMID:26612957",
         query='TITLE:"Principles of connectivity among morphologically defined cell types in adult neocortex"', note_claim=""),
    dict(pid="C-5-a", expect=["rhythmic", "recurrent", "activity", "neocortex"], clause="§4.1 C-5", mechanism="丘脑-皮层样慢涨落（频率 f ∈ [0.1,1.5] Hz ⇒ dwell 0.7–10 s）",
         value="f ∈ [0.1,1.5] Hz；幅度 ∈ [0,1]；nRT 反馈 ∈ [0,1]", ident="PMID:11017176",
         query='TITLE:"Cellular and network mechanisms of rhythmic recurrent activity in neocortex"', note_claim=""),
    dict(pid="C-5-b", expect=["thalamocortical", "oscillat"], clause="§4.1 C-5", mechanism="Steriade 类丘脑-皮层慢振荡工作（清单标 literature_unverified）",
         value="TC/nRT 慢抑制动力学", ident="",
         query='AUTH:"Steriade M" AND ("thalamocortical" OR "slow oscillation")',
         note_claim="清单标 `literature_unverified` → **本轮必须复核**；命中则升级"),
    dict(pid="C-6-a", expect=["adenosine", "synaptic"], clause="§4.1 C-6", mechanism="代谢/腺苷慢反馈（A1 受体介导突触抑制，τ_meta ∈ [5,60] s）",
         value="τ_meta ∈ G-slow [5,60] s；反馈增益 ∈ [0,1]", ident="",
         query='TITLE:"adenosine" AND TITLE:"synaptic" AND SRC:MED',
         note_claim="清单标 `literature_unverified`（本轮未做实测命中）→ **G2 必须 provenance 化**"),
    dict(pid="C-6-b", expect=["adenosine"], clause="§4.1 C-6", mechanism="腺苷与睡眠压力/突触稳态",
         value="慢时间尺度（数秒–数十秒）依据", ident="",
         query='(TITLE:"adenosine" AND (TITLE:"sleep" OR TITLE:"wakefulness" OR TITLE:"cortex")) AND SRC:MED',
         note_claim="本轮新增第二查询式"),
    dict(pid="C-7-a", expect=["wakefulness", "local", "cortical", "networks"], clause="§4.1 C-7", mechanism="慢调质状态量（ACh/NE → 增益 + E/I + 阈值三联动）",
         value="τ_mod ∈ [2,60] s；g_mod ∈ [0.5,2.0]", ident="PMID:21435553",
         query='TITLE:"Effects and mechanisms of wakefulness on local cortical networks"', note_claim=""),
    dict(pid="C-7-b", expect=["neuromodulation", "brain", "state", "behavior"], clause="§4.1 C-7", mechanism="神经调质与脑状态（综述）",
         value="调质慢状态时间尺度", ident="PMID:32250724",
         query='TITLE:"Neuromodulation of brain state and behavior"', note_claim=""),
    dict(pid="C-8-a", expect=["homeostatic", "plasticity"], clause="§4.1 C-8", mechanism="稳态可塑性慢变量（突触缩放/内稳态阈值，τ_homeo ∈ [10,300] s）",
         value="τ_homeo ∈ [10,300] s（压缩语义）；目标率 r* 带；缩放幅度 ∈ [0,1]", ident="",
         query='AUTH:"Turrigiano GG" AND ("synaptic scaling" OR "homeostatic plasticity" OR "quantal amplitude")',
         note_claim="清单标 `literature_unverified`（Turrigiano 类工作本轮未实测命中）→ **G2 必须 provenance 化**"),
    dict(pid="C-8-b", expect=["scaling", "synaptic"], clause="§4.1 C-8", mechanism="稳态可塑性时间尺度（分钟–小时）",
         value="真实 τ 为 min–h（M11 压缩到 [10,300] s 需登记抽象）", ident="",
         query='("synaptic scaling" AND ("time course" OR "hours" OR "homeostatic")) AND SRC:MED',
         note_claim="本轮新增：用于支撑「压缩语义」的抽象登记"),
    # ---------- §4.5 / §1.5 判据带输入（G2 需提供的实测/文献输入） ----------
    dict(pid="D-1", expect=["visual", "cortex"], clause="§4.5(a)", mechanism="小鼠 V1 在体自发放电带（P5(a) 目标带输入；**不得用斑马鱼带**）",
         value="中位率带（运行前定稿）", ident="",
         query='("mouse" AND "visual cortex" AND "spontaneous activity" AND "firing rate" AND "awake")',
         note_claim="清单要求「运行前由 §1.5 C2 同口径定稿」；本轮给文献档（非判据带）"),
    dict(pid="D-2", expect=["visual", "cortex"], clause="§4.5(b)", mechanism="小鼠皮层静默/低活动比例（P5(b) 目标带输入）",
         value="静默比例带", ident="",
         query='("mouse" AND "visual cortex" AND "spontaneous" AND ("silent neurons" OR "low firing rate"))',
         note_claim=""),
    dict(pid="D-3", expect=["orientation", "selectivity"], clause="§1.5 C1 / P7", mechanism="小鼠 V1 取向选择性指数（OSI）分布参考",
         value="OSI 分布/半宽（与 MICrONS 同口径对照）", ident="",
         query='("orientation selectivity" AND "mouse" AND "V1" AND ("distribution" OR "index")) AND SRC:MED',
         note_claim=""),
    dict(pid="D-4", expect=["functional", "connectomics", "multiple", "areas", "visual", "cortex"], clause="§1.5 C2 / P14", mechanism="MICrONS 论文（结构-功能同源，权威锚）",
         value="~75,000 神经元钙成像 + >200,000 细胞 / 0.5B 突触 EM", ident="DOI:10.1038/s41586-025-08790-w",
         query='TITLE:"Functional connectomics spanning multiple areas of mouse visual cortex"',
         note_claim="清单记 nature.com 200 / 693,531 B（本轮复核）"),
    dict(pid="D-5", clause="§1.5 C3 / RF-M11-2", mechanism="**目标 dwell 带**（M11 必须自建；禁用 M10/ZAPBench 26.5 s）",
         value="由 MICrONS 功能数据同口径重建（本轮给出采集口径：帧率/时长）", ident="",
         query="（数据侧，非文献：见 data/m11_microns_protocol.json）", note_claim="文献档仅作量级参考：0.3–3 s（PMID 11017176）"),
]

CSV_FIELDS = ["prior_id", "clause", "mechanism_or_param", "value_or_range", "ident_claim",
              "ident_basis", "title_verified", "epmc_source", "query_string",
              "hit_count_query", "pmid", "pmcid", "doi", "title", "journal", "year",
              "status", "evidence_level", "value_in_abstract", "abstract_snippet", "note", "retrieved_at"]


def curl_json(url, timeout=25):
    tmp = os.path.join(RAW, "_tmp_epmc.json")
    cmd = ["curl", "-sS", "-L", "--connect-timeout", "8", "--max-time", str(timeout), "-o", tmp,
           "-w", "%{http_code}|%{size_download}"]
    try:
        p = subprocess.run(cmd + [url], capture_output=True, timeout=timeout + 10)
    except subprocess.TimeoutExpired:
        return None, dict(http_code=0, curl_exit=124)
    parts = p.stdout.decode().strip().split("|")
    code = int(parts[0]) if parts and parts[0].isdigit() else 0
    try:
        with open(tmp, "rb") as f:
            b = f.read()
        os.remove(tmp)
        return json.loads(b.decode("utf-8")), dict(http_code=code, nbytes=len(b))
    except Exception as e:
        return None, dict(http_code=code, error=str(e)[:120])


def epmc_search(query, page_size=3, timeout=25, sort_cited=False):
    import urllib.parse
    u = "%s?query=%s&format=json&pageSize=%d&resultType=core" % (EPMC, urllib.parse.quote(query), page_size)
    if sort_cited:
        u += "&sort=CITED%20desc"
    return curl_json(u, timeout=timeout)


def find_by_ext_id(pmid, timeout=25):
    import urllib.parse
    u = "%s?query=%s&format=json&pageSize=1&resultType=core" % (EPMC, urllib.parse.quote("EXT_ID:%s AND SRC:MED" % pmid))
    return curl_json(u, timeout=timeout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=25.0)
    ap.add_argument("--out", default=OUT_CSV)
    args = ap.parse_args()
    os.makedirs(RAW, exist_ok=True)
    logf = open(LOG, "a", buffering=1)

    def log(m):
        s = "[%s] %s" % (datetime.now(timezone.utc).strftime("%H:%M:%S"), m)
        logf.write(s + "\n")
        print(s, flush=True)

    log("=== build_m11_literature_priors start n_rows=%d PYTHONHASHSEED=%s"
        % (len(PRIORS), os.environ.get("PYTHONHASHSEED", "(unset)")))
    rows, raw = [], {}
    for pr in PRIORS:
        rec = {k: "" for k in CSV_FIELDS}
        rec.update(prior_id=pr["pid"], clause=pr["clause"], mechanism_or_param=pr["mechanism"],
                   value_or_range=pr["value"], ident_claim=pr["ident"], query_string=pr["query"],
                   retrieved_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        notes = []
        if pr.get("note_claim"):
            notes.append("清单原注：" + pr["note_claim"])
        # (a) 若清单给了 PMID，先按 EXT_ID 精确复核
        core = None
        ident_basis = "清单声明标识（EXT_ID 精确复核）" if pr["ident"] else ""
        if pr["ident"].startswith("PMID:"):
            pmid = pr["ident"].split(":", 1)[1]
            js, meta = find_by_ext_id(pmid, timeout=args.timeout)
            raw["%s_extid" % pr["pid"]] = dict(query="EXT_ID:%s" % pmid, http=meta,
                                               hit=(js or {}).get("hitCount"))
            res = ((js or {}).get("resultList") or {}).get("result") or []
            if res:
                core = res[0]
                rec["hit_count_query"] = (js or {}).get("hitCount")
            else:
                notes.append("⚠️ EXT_ID:%s 在 Europe PMC **未命中** → 该标识不可核查" % pmid)
        elif pr["ident"].startswith("DOI:"):
            doi = pr["ident"].split(":", 1)[1]
            js, meta = epmc_search('DOI:"%s"' % doi, page_size=1, timeout=args.timeout)
            raw["%s_doi" % pr["pid"]] = dict(query='DOI:"%s"' % doi, http=meta, hit=(js or {}).get("hitCount"))
            res = ((js or {}).get("resultList") or {}).get("result") or []
            if res:
                core = res[0]
                rec["hit_count_query"] = (js or {}).get("hitCount")
            else:
                notes.append("⚠️ DOI:%s 在 Europe PMC 未命中" % doi)
        # (b) 运行清单给的（或本轮替代的）查询式
        # 清单未给标识的行 = "检索集/发现式"查询：要求 MED 库 + 按被引排序（可复现的发现式协议）
        discover = (pr["ident"] == "")
        q = pr["query"]
        if discover and "SRC:MED" not in q:
            q = q + " AND SRC:MED"
        js2, meta2 = epmc_search(q, page_size=3, timeout=args.timeout, sort_cited=discover)
        pr = dict(pr, query_used=q)
        rec["query_string"] = q
        n_hit = (js2 or {}).get("hitCount")
        raw["%s_query" % pr["pid"]] = dict(query=pr["query"], http=meta2, hit=n_hit,
                                           top=[dict(pmid=r.get("pmid"), doi=r.get("doi"), title=r.get("title"))
                                                for r in (((js2 or {}).get("resultList") or {}).get("result") or [])])
        if core is None:
            res2 = ((js2 or {}).get("resultList") or {}).get("result") or []
            if res2:
                core = res2[0]
                ident_basis = "本轮发现式查询首条（清单未给标识；**未经清单核对**）"
                notes.append("清单未给标识 → 取「SRC:MED + 按被引排序」发现式查询首条作为 provenance 候选；"
                             "**该条与清单声明无对应关系**，参数范围**不得**以本行为唯一依据")
        if rec["hit_count_query"] == "" or rec["hit_count_query"] is None:
            rec["hit_count_query"] = n_hit
        ab = ""
        if core:
            rec["pmid"] = core.get("pmid", "") or ""
            rec["pmcid"] = core.get("pmcid", "") or ""
            rec["doi"] = core.get("doi", "") or ""
            rec["title"] = (core.get("title", "") or "").strip()
            rec["epmc_source"] = core.get("source", "") or ""
            rec["ident_basis"] = ident_basis
            if pr.get("expect"):
                tl = re.sub(r"[^a-z0-9]+", " ", rec["title"].lower())
                miss = [t for t in pr["expect"]
                        if re.sub(r"[^a-z0-9]+", " ", t.lower()).strip() not in tl]
                rec["title_verified"] = "yes" if not miss else "no"
                if miss:
                    if pr["ident"]:
                        notes.append("⚠️ **题录核对不通过**：返回题录缺少关键词 %s → 清单该标识可能指错文献" % miss)
                    else:
                        notes.append("⚠️ **题录核对不通过**：返回题录缺少主题关键词 %s（本行无清单声明的标识）" % miss)
            rec["journal"] = ((core.get("journalInfo") or {}).get("journal") or {}).get("title", "") or ""
            rec["year"] = core.get("pubYear", "") or ""
            ab = core.get("abstractText", "") or ""
            ab = re.sub(r"<[^>]+>", "", ab)
            ab = re.sub(r"\s+", " ", ab).strip()
            rec["abstract_snippet"] = ab[:400]
            rec["evidence_level"] = "题录+摘要可得" if ab else "题录only"
            if pr.get("value_regex") and ab:
                rec["value_in_abstract"] = "yes" if re.search(pr["value_regex"], ab, re.I) else "no"
                if rec["value_in_abstract"] == "no":
                    notes.append("值/范围未在摘要文本中出现（全文未读 ⇒ 不得声称已核实）")
            elif ab:
                rec["value_in_abstract"] = "not_checked"
            rec["status"] = "literature"
            # 发现式行（清单未给标识）：题录必须与主题关键词相符，否则**不得**记为 literature
            if pr["ident"] == "" and rec["title_verified"] == "no":
                rec["status"] = "literature_unverified"
                rec["evidence_level"] = "检索集有命中但**未找到可直接支撑该参数的条目**"
                notes.append("**未命中（参数级）**：检索集 hitCount=%s > 0（检索集本身可得），但被引排序首条的题录"
                             "与本行主题不符 ⇒ 本行参数范围**不得**以该条为依据（改由同口径数据或另找命中文献）；"
                             "关键词核对缺：%s" % (n_hit, rec.get("title_verified")))
        else:
            rec["ident_basis"] = ident_basis or "无"
            rec["status"] = "literature_unverified"
            rec["evidence_level"] = "未命中"
            rec["value_in_abstract"] = "no_abstract"
            notes.append("**未命中**：查询式 `%s` → hitCount=%s；不得凭印象补 PMID" % (pr["query"], n_hit))
        rec["note"] = "；".join(notes)
        rows.append(rec)
        log("%-9s %-6s pmid=%-9s %-52s %s" % (pr["pid"], rec["status"], rec["pmid"] or "-",
                                              (rec["title"] or "")[:52], ("hit=%s" % n_hit)))
        time.sleep(0.3)  # 温和限速

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        f.write("# M11 文献 provenance 表（generated_by=tools/build_m11_literature_priors.py；"
                "source=Europe PMC REST；status ∈ {literature, literature_unverified}；"
                "literature=题录经 Europe PMC 实测命中（**不等于全文/参数值已核实**））\n")
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    n_lit = sum(1 for r in rows if r["status"] == "literature")
    n_unv = sum(1 for r in rows if r["status"] == "literature_unverified")
    summary = dict(generated_by="tools/build_m11_literature_priors.py",
                   generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                   n_rows=len(rows), n_literature=n_lit, n_literature_unverified=n_unv,
                   status_vocab=["literature", "literature_unverified"],
                   caveat="`literature` = Europe PMC 题录实测命中；参数值仅在摘要出现时标 value_in_abstract=yes，"
                          "否则不得声称已核实（全文未读）",
                   raw_queries=raw)
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    log("wrote %s (n=%d literature=%d unverified=%d)" % (OUT_CSV, len(rows), n_lit, n_unv))
    log("ALL_DONE")
    logf.close()
    return summary


if __name__ == "__main__":
    s = main()
    print(json.dumps({k: v for k, v in s.items() if k != "raw_queries"}, ensure_ascii=False, sort_keys=True))
    sys.exit(0)
