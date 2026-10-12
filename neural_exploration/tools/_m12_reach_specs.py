#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M12 数据门 · 清单 §2 的 S1–S12 全部源 + 子端点 可达性规格表（≥40 行）

**本模块只声明规格，不写任何产物**；实际探测与 CSV 写入由 `tools/probe_m12_sources.py`
的 `--only REACH` 模式完成 ⇒ **单一 owner**（M11-R60：两个工具不得共用产物路径）。

每一行必须给出（主 agent 收尾派发的要求，**空值必须为 0**）
--------------------------------------------------------
- `provenance`      来源/依据（查到什么、依据什么判定）
- `source_url`      真实 URL
- `retrieved_at`    **判决时刻**（由工具在探测时打 UTC 戳）
- `evidence_level`  ∈ {measured, literature, derived, unverified}
- `wrong_usage_consequence`  **用错后果**
- `sha_basis`       哈希基准：payload / file_bytes / head_only / not_applicable

`mode` 说明
-----------
- `count`   Allen API count$eqtrue（先拿「数」再决定取不取）
- `page`    Allen API 小页（读 schema / 字段）
- `list`    S3 list-type=2（KeyCount / IsTruncated / CommonPrefixes）
- `range`   -r 0-1048575 当场吞吐探针（**不取全件**）
- `head`    仅 -I（对象级取证，不下载）
- `fetch`   全量 GET（**限 --max-filesize**，只用于小件）
- `epmc`    Europe PMC REST 题录查询（读 hitCount）
- `local`   本机既有件（M11 复用件）——**只读不写**
- `derived` 由其它实测行派生（无网络）

注意：本文件的散文一律用 「」 作引号，**避免** ASCII 双引号与 Python 字符串定界符冲突。
"""

from __future__ import annotations

ALLEN_API = "https://api.brain-map.org/api/v2/data/query.json"
ALLEN_BO = "https://allen-brain-observatory.s3.amazonaws.com"
VBN = "https://visual-behavior-neuropixels-data.s3.us-west-2.amazonaws.com"
IBL = "https://ibl-brain-wide-map-public.s3.amazonaws.com"
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
CCF_DL = "https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf"

WHITELIST_CSV = "data/m12_ibl_sessions.csv"

SPECS = [
    # ================= S1 Allen Mouse Brain Connectivity Atlas =================
    dict(sid="S1-01", source="S1 Allen 投射矩阵 · ProjectionStructureUnionize 全表行数",
         category="allen_connectivity", kind="api_count", mode="count", timeout=90,
         url=ALLEN_API + "?criteria=model::ProjectionStructureUnionize,rma::options[count$eqtrue]",
         evidence_level="measured",
         wrong="把它当「区域对矩阵行数」会错几个量级：该表是「实验 × 结构」的长表，不是区域×区域矩阵；"
               "若按行数直接建 500×500 矩阵（应先在服务端聚合/透视）⇒ 矩阵维度与语义双错",
         note="§2.2 S1 的 T1 步骤：先拿行数（成本极小）再决定取不取"),
    dict(sid="S1-02", source="S1 投射矩阵 · 单页 schema（字段名读回）",
         category="allen_connectivity", kind="api_page", mode="page",
         url=ALLEN_API + "?criteria=model::ProjectionStructureUnionize,rma::options[num_rows$eq5]",
         evidence_level="measured",
         wrong="凭印象写字段名（如假定一定有 projection_volume / normalized_projection_volume）⇒ "
               "装配脚本会 KeyError 或静默取到 None；本行给出磁盘读回的字段清单",
         note="字段清单由实测响应体给出，不臆造"),
    dict(sid="S1-03", source="S1 投射矩阵 · 服务端按单结构过滤后的行数（白名单口径试算）",
         category="allen_connectivity", kind="api_count", mode="count",
         url=ALLEN_API + "?criteria=model::ProjectionStructureUnionize,rma::criteria[structure_id$eq16997],"
                         "rma::options[count$eqtrue]",
         evidence_level="measured",
         wrong="用未过滤的全表行数外推「白名单过滤后可取」⇒ 会错判可行性"
               "（§2.2 S1 的 pass 条件正是「服务端白名单过滤后行数 ≤ 10^5」）；"
               "本条给出过滤确实生效的证据形态",
         note="若返回 4xx/空 ⇒ 记 REACHABLE_NO_DATA 并换 StructureUnionize 变体（§2.2 S1 明文要求）"),
    dict(sid="S1-04", source="S1 变体 · SectionDataSet 行数（实验数锚）",
         category="allen_connectivity", kind="api_count", mode="count",
         url=ALLEN_API + "?criteria=model::SectionDataSet,rma::options[count$eqtrue]",
         evidence_level="measured",
         wrong="把 SectionDataSet 行数当「注射实验数」可能高估（含非投射实验与重复切片集）⇒ "
               "Oh 2014 的「千级实验」锚与它对不上时要如实登记，不得挑好看的数",
         note="§2.2 S1：命名以 G2 实测为准（400/404 ⇒ REACHABLE_NO_DATA 并换变体）"),
    dict(sid="S1-05", source="S1 变体 · StructureUnionize 行数",
         category="allen_connectivity", kind="api_count", mode="count", timeout=90,
         url=ALLEN_API + "?criteria=model::StructureUnionize,rma::options[count$eqtrue]",
         evidence_level="measured",
         wrong="把 StructureUnionize 与 ProjectionStructureUnionize 混用 ⇒ 两个不同的表被当同一个"
               "「投射数据」（列语义与行粒度都不同）",
         note="S1 的降级变体之一（§2.2 S1 明文列举）"),
    dict(sid="S1-06", source="S1 文献锚 · Oh 2014（PMID 24695228 / DOI 10.1038/nature13186）",
         category="allen_connectivity", kind="literature", mode="epmc",
         url=EPMC + "?query=DOI:%2210.1038/nature13186%22&format=json&pageSize=1",
         evidence_level="literature",
         wrong="把题录命中当「数据可得」：文献可查 ≠ 其补充材料可下载（S1 降级路径③要求逐条实测"
               "其可得性）；也不得把题录里的数字当 M12 判据带（§0.8 EQ-2）",
         note="§2.2 S1 明写「权威出处 Oh 2014（本轮实测命中）」⇒ 本条复核命中状态"),
    dict(sid="S1-07", source="S1 降级路径③ · Oh 2014 论文页面可得性（nature.com）",
         category="allen_connectivity", kind="supplement", mode="head",
         url="https://www.nature.com/articles/nature13186",
         evidence_level="measured",
         wrong="把「论文页面可达」当成「补充材料矩阵可得」⇒ 会承诺一个拿不到的文献档；"
               "本条只证页面层，补充材料需另行逐件实测（未测 ≠ 未命中）",
         note="S1 降级路径③ 的可得性首步；未对补充材料逐件探测（缺这一维，如实登记）"),

    # ================= S2 CCFv3 本体 =================
    dict(sid="S2-01", source="S2 CCFv3 结构本体 · model::Structure 全表（13,709 行）",
         category="allen_ontology", kind="api_json", mode="range", timeout=90,
         url=ALLEN_API + "?criteria=model::Structure,rma::options[num_rows$eqall]",
         evidence_level="measured",
         wrong="把 13,709 当 CCFv3 区域数会高估 10.3 倍（成体图谱 graph_id=1 只有 1,327 行）；"
               "混用多个 graph_id 会造成区域 ID 串台（判据行 M12-S2-01/02/03）",
         note="本节点已取回并过完整性门（7,238,867 B / ETag md5 复算通过 / msg 行数 == 13,709）；"
              "详见 data/m12_raw/s2/model_Structure_all.json 与判据行 M12-S2-01..12"),
    dict(sid="S2-02", source="S2 结构本体 · 全表行数（count$eqtrue）",
         category="allen_ontology", kind="api_count", mode="count",
         url=ALLEN_API + "?criteria=model::Structure,rma::options[count$eqtrue]",
         evidence_level="measured",
         wrong="只用 count 不取表 ⇒ 拿不到 parent_structure_id / structure_id_path ⇒ "
               "区域层级与 500 叶区无法声明",
         note="与 S2-01 构成「数 ↔ 表」双证据"),
    dict(sid="S2-03", source="S2 子图 · CCFv3 2017 成体图谱（graph_id=1）行数",
         category="allen_ontology", kind="api_count", mode="count",
         url=ALLEN_API + "?criteria=model::Structure,rma::criteria[graph_id$eq1],"
                         "rma::options[count$eqtrue]",
         evidence_level="measured",
         wrong="不声明取自哪个 graph_id ⇒ 区域映射口径不可复现（§0.8 第 1 条要求逐字写明区域映射口径）",
         note="M12 的 500 叶区必须显式绑定 graph_id（判据行 M12-S2-02）"),
    dict(sid="S2-04", source="S2 备选通道 · 单结构读数（api Structure/1.json）",
         category="allen_ontology", kind="api_json", mode="fetch",
         url="https://api.brain-map.org/api/v2/data/Structure/1.json",
         evidence_level="measured",
         wrong="把它当 allensdk 的 structure_graph.json 等价物是错的：两者字段与序号口径不同"
               "（allensdk 另有本地缓存的 graph 结构）⇒ 混用会出现同一区域两个 id",
         note="§2.2 S2 提到「或 allensdk structure_graph.json 对应件」⇒ 本条实测该备选通道的可达性，"
              "不对其与全表的一致性下结论（那需另做逐行比对）"),

    # ================= S3 CCFv3 标注体数据 =================
    dict(sid="S3-01", source="S3 标注体 · 100 µm 档目录（annotation/ccf_2017）",
         category="allen_annotation", kind="http_dir", mode="head",
         url=CCF_DL + "/annotation/ccf_2017/",
         evidence_level="measured",
         wrong="凭印象写文件名（§2.2 S3 明写「文件名以 G2 实测为准，不得臆造」）⇒ 取数脚本会 404 或取错档",
         note="S3 的 T1 步骤；只探目录可达性，未下载任何体数据"),
    dict(sid="S3-02", source="S3 标注体 · 25 µm 档对象（annotation_25.nrrd）",
         category="allen_annotation", kind="http_dir", mode="head",
         url=CCF_DL + "/annotation/ccf_2017/annotation_25.nrrd",
         evidence_level="measured",
         wrong="把 25 µm 档当「和 100 µm 一样小」⇒ 体量差两个数量级（§2.2 S3 估计）；"
               "本行只证对象级可达性，不给该档的可用性结论",
         note="§2.2 S3 预备判定：100 µm 档若实测 ≥1 MB/s 才可能 pass ⇒ 本条不做吞吐结论"),
    dict(sid="S3-03", source="S3 降级路径① · Structure 表是否含体积/中心坐标（质心来源）",
         category="allen_annotation", kind="api_page", mode="page",
         url=ALLEN_API + "?criteria=model::Structure,rma::options[num_rows$eq3]",
         evidence_level="measured",
         wrong="把质心欧氏距离当「体素接触面」⇒ 区域间距离语义错（§2.2 S3 降级路径① 明确要求"
               "口径写明「质心距离，非体素接触面」）",
         note="S3 降级路径①（只取区域质心/体积表）⇒ 与 S2 同通道同量级；本条实测该表是否真带体积/中心字段"),
    dict(sid="S3-04", source="S3 标注体 · 参考体积清单页（第二证据）",
         category="allen_annotation", kind="http_dir", mode="head",
         url=CCF_DL + "/annotation/",
         evidence_level="measured",
         wrong="把「参考体积清单页可达」当「标注体可取」⇒ 把元数据可得性当选数据可得性"
               "（L9.2 #4 REACHABLE_NO_DATA 陷阱）",
         note="10²–10³ MB 级体数据本节点不取（超 5 MB 预算，未达 50 MB 硬阈值）"),

    # ================= S4 调质系统全脑投射图谱 =================
    dict(sid="S4-01", source="S4 调质投射 · Cre 系过滤的 ProjectionStructureUnionize 行数",
         category="allen_modulatory", kind="api_count", mode="count",
         url=ALLEN_API + "?criteria=model::ProjectionStructureUnionize,"
                         "rma::criteria[transgenic_line$eqSlc6a3-Cre],rma::options[count$eqtrue]",
         evidence_level="measured",
         wrong="凭印象写转基因系名（如把 DAT-Cre 写成 dat-cre）⇒ 过滤命中 0 行而静默产出空矩阵；"
               "本行给出该过滤的有效性证据（含 0 命中的可能）",
         note="§2.2 S4：同一 Atlas 通道 + Cre 系过滤；命中 0 不得写成数据不存在"),
    dict(sid="S4-02", source="S4 文献锚 · 单胺能全脑投射题录查询（hitCount 登记）",
         category="allen_modulatory", kind="literature", mode="epmc",
         url=EPMC + "?query=TITLE:%22monoaminergic%22%20AND%20TITLE:%22projections%22%20AND%20"
                    "TITLE:%22whole-brain%22&format=json&pageSize=1",
         evidence_level="literature",
         wrong="把 hitCount=0 写成「该文献不存在」⇒ §2.2 S4 明确要求登记为题录未命中（≠ 不存在）+ 查询式；"
               "把 0 当反证会误杀 T2 的文献依据",
         note="§2.2 S4 明文登记的未命中项；本行原样复核查询式与 hitCount"),
    dict(sid="S4-03", source="S4 降级路径② · 全局标量调制（无网络，纯抽象登记）",
         category="allen_modulatory", kind="abstraction", mode="derived",
         url="（无 URL；抽象登记，不取数）",
         evidence_level="derived",
         wrong="把「全局标量增益」当「投射拓扑」⇒ 越界声称（§2.2 S4 降级路径② 明确要求登记抽象、"
               "不声称投射拓扑，并下调 P6/H4 的证据等级）",
         note="若 S4 不可得，T2 退化为 DA/5HT/NE/ACh 各 1 标量 × 区域增益表；本条不含网络实测"),
    dict(sid="S4-04", source="S4 文献锚备选 · 中脑多巴胺投射题录",
         category="allen_modulatory", kind="literature", mode="epmc",
         url=EPMC + "?query=%22mesolimbic%22%20AND%20%22dopamine%22%20AND%20%22projections%22&"
                    "format=json&pageSize=1",
         evidence_level="literature",
         wrong="用一条泛题录充当「区域投射强度」⇒ 幅度量缺失而方向量被当幅度用"
               "（§0.8 EQ-2：文献档只能作 direction，除非逐条 value_verified）",
         note="S4 降级路径①（少量预注册种子实验）的文献侧伴证"),

    # ================= S5 IBL 行为 =================
    dict(sid="S5-01", source="S5 IBL 公开桶 · 根列举（一级前缀）",
         category="ibl_behavior", kind="s3_list", mode="list",
         url=IBL + "/?list-type=2&max-keys=20&delimiter=/",
         evidence_level="measured",
         wrong="把「桶可 list」当「对象可取」（NEEDS_VERIFY 陷阱）⇒ §2.3 T1 明写列举成功不得当作数据可得",
         note="S5 的入口通道（§2.2 S5 的公开桶）"),
    dict(sid="S5-02", source="S5 IBL 公开桶 · data/ 一级 lab 前缀列举",
         category="ibl_behavior", kind="s3_list", mode="list",
         url=IBL + "/?list-type=2&max-keys=200&delimiter=/&prefix=data/",
         evidence_level="measured",
         wrong="把 lab 数当 session 数（判据行 M12-S5-01 的用错后果）⇒ 判据分母错",
         note="本节点实测 13 个 lab（含 registration 这个 0 subject 前缀）"),
    dict(sid="S5-03", source="S5 IBL 公开桶 · aggregates/ 前缀列举",
         category="ibl_behavior", kind="s3_list", mode="list",
         url=IBL + "/?list-type=2&max-keys=50&delimiter=/&prefix=aggregates/",
         evidence_level="measured",
         wrong="把 aggregates/ 当「只有活动数据」⇒ 会漏掉行为侧的合并表（trials.pqt 就在其中）⇒ "
               "P8/P9 被迫走 40 次逐 session 列举（本节点已实测后者可行但低效）",
         note="本轮新增的关键发现：aggregates/ 同时含行为（trials.pqt）与活动（clusters*）两类对象"),
    dict(sid="S5-04", source="S5 IBL 全量聚合行为表 · trials.pqt（R72 批准取数）",
         category="ibl_behavior", kind="s3_object", mode="head",
         url=IBL + "/aggregates/2026_Q2_IBL_et_al_BWM/trials.pqt",
         evidence_level="measured",
         wrong="把 24,212,510 B 的单对象全量行为表当「某个 session 的表」⇒ 行粒度错（它是逐 trial 长表，"
               "必须按会话列聚合/切分后才能进 P8 的分母）",
         note="本节点已按 R72 批准取回并过三级完整性门 + pyarrow 解析；明细见 "
              "data/m12_ibl_trials_aggregate.json 与 reachability 的 S5FULL 行"),
    dict(sid="S5-05", source="S5 OpenAlyx API（服务端按查询过滤的正规通道）",
         category="ibl_behavior", kind="api_json", mode="fetch",
         url="https://openalyx.internationalbrainlab.org/datasets?format=json",
         evidence_level="measured",
         wrong="把 403 NEEDS_TOKEN 写成「数据不存在」⇒ §2.2 S5 降级路径③ 明确不得把 NEEDS_TOKEN 或吞吐"
               "不可行写成数据不存在；且 OpenAlyx 可按查询过滤，拿 token 后应优先于桶",
         note="M11 已实测 403（继承）；本条当场复核并登记"),
    dict(sid="S5-06", source="S5 预注册白名单 · 40 个 session 的 _ibl_trials.table.*.pqt",
         category="ibl_behavior", kind="s3_object_set", mode="local",
         url="neural_exploration/data/m12_ibl_sessions.csv（40 行白名单；逐件 HTTP 200 已过门）",
         evidence_level="measured",
         wrong="把白名单当全量（选择偏差）；也不得把 22 个「该口径未命中」的 session 算作「无行为数据」"
               "（判据行 M12-S5-02 / M12-S5-07）",
         note="本节点已取回 40/40 COMPLETE（合计 1,798,898 B）；清单见 " + WHITELIST_CSV +
              "；A/B/C/D 归属已在出数前预注册"),
    dict(sid="S5-07", source="S5 IBL 公开桶 · 示例 session 树 public_tree.txt",
         category="ibl_behavior", kind="s3_object", mode="head",
         url=IBL + "/data/public_tree.txt",
         evidence_level="measured",
         wrong="把它当全量 session 索引 ⇒ 严重漏计：本轮实测该文件（105,625 B / 507 行）只覆盖 "
               "churchlandlab 一支，不是全库索引",
         note="本节点已落盘并逐行核对（台账 note 已写明该限制）"),

    # ================= S6 IBL 全脑活动 =================
    dict(sid="S6-01", source="S6 IBL 活动 · 2026_Q2 clusters.pqt（逐单元表）",
         category="ibl_activity", kind="s3_object", mode="head",
         url=IBL + "/aggregates/2026_Q2_IBL_et_al_BWM/clusters.pqt",
         evidence_level="measured",
         wrong="把它当「逐 session 活动表」⇒ 行粒度错（这是逐单元表）；且它 112,325,731 B，"
               "若按小表处理会撞 50 MB 阈值",
         note="本节点只做对象级取证（HEAD），不取（超 50 MB 硬阈值 ⇒ 须主 agent 批准）"),
    dict(sid="S6-02", source="S6 IBL 活动 · clusters.acgs_log.npy（派生矩阵）",
         category="ibl_activity", kind="s3_object", mode="head",
         url=IBL + "/aggregates/2026_Q2_IBL_et_al_BWM/clusters.acgs_log.npy",
         evidence_level="measured",
         wrong="把 .npy 的字节数当元素数 ⇒ 形状错（须按 dtype 与 shape 头反解）",
         note="159,163,776 B；未取（超 50 MB）"),
    dict(sid="S6-03", source="S6 IBL 活动 · clusters.waveforms_peak.npy",
         category="ibl_activity", kind="s3_object", mode="head",
         url=IBL + "/aggregates/2026_Q2_IBL_et_al_BWM/clusters.waveforms_peak.npy",
         evidence_level="measured",
         wrong="同上；另不得把波形矩阵当「活动时间序列」（它是逐单元峰值波形，没有时间轴）",
         note="159,163,776 B；未取"),
    dict(sid="S6-04", source="S6 IBL 活动 · spikesorting/ 前缀列举（原始分选产物）",
         category="ibl_activity", kind="s3_list", mode="list",
         url=IBL + "/?list-type=2&max-keys=20&delimiter=/&prefix=spikesorting/",
         evidence_level="measured",
         wrong="把原始分选前缀当「已处理的区域活动」⇒ 从 raw spike 到区域率之间缺整条装配管线"
               "（T6 装配可行性：M12 特有的第四种「可达 ≠ 可行」）",
         note="S6 的 T1 步骤；不取任何 raw spike"),
    dict(sid="S6-05", source="S6 IBL 活动 · paper_reproducible_ephys/ 前缀列举",
         category="ibl_activity", kind="s3_list", mode="list",
         url=IBL + "/?list-type=2&max-keys=20&delimiter=/&prefix=paper_reproducible_ephys/",
         evidence_level="measured",
         wrong="把论文复现集当「全库活动」⇒ 分母错；但也不得因它小就当「区域级汇总表」——子集 ≠ 汇总",
         note="§2.2 S6 降级路径①（区域级汇总表优先）的候选之一 ⇒ 须逐件看 schema 才能定（本节点未取）"),
    dict(sid="S6-06", source="S6 IBL 活动 · aggregates/np1_np2_kim_recordings 前缀列举",
         category="ibl_activity", kind="s3_list", mode="list",
         url=IBL + "/?list-type=2&max-keys=20&delimiter=/&prefix=aggregates/np1_np2_kim_recordings/",
         evidence_level="measured",
         wrong="把双探针记录当「区域级汇总」（两者是不同量级与不同口径）",
         note="§2.2 S6 的区域级汇总候选；未取"),

    # ================= S7 Allen BO / VBN =================
    dict(sid="S7-01", source="S7 Allen BO ecephys 汇总表 · sessions.csv",
         category="allen_ecephys", kind="s3_object", mode="fetch",
         url=ALLEN_BO + "/visual-coding-neuropixels/ecephys-cache/sessions.csv",
         evidence_level="measured",
         wrong="把 session 行数当 session 总数（本表只覆盖 VBN 一档）；也不得与 IBL 的 session 数混用"
               "（跨源分母混用 ⇒ 判据分母错；判据行 M12-S7-01）",
         note="本节点已取回：7,849 B / 58 数据行 / sha256 与 M11 已落盘件逐字节一致"),
    dict(sid="S7-02", source="S7 Allen BO ecephys 汇总表 · probes.csv",
         category="allen_ecephys", kind="s3_object", mode="fetch",
         url=ALLEN_BO + "/visual-coding-neuropixels/ecephys-cache/probes.csv",
         evidence_level="measured",
         wrong="把 332 个 probe 当 session 或单元（probe ≠ session ≠ unit；判据行 M12-S7-02）",
         note="本节点已取回：26,954 B / 332 数据行 / sha256 与 M11 一致"),
    dict(sid="S7-03", source="S7 Allen BO · channels.csv（1 MiB Range 探针，不取全件）",
         category="allen_ecephys", kind="s3_object", mode="range",
         url=ALLEN_BO + "/visual-coding-neuropixels/ecephys-cache/channels.csv",
         evidence_level="measured",
         wrong="用小表的单件端到端 B/s 代表该桶吞吐 ⇒ 低估 20–50 倍（判据行 M12-M-02）；"
               "本行给出同桶大对象的当场吞吐档，是 S7 唯一可用的链路吞吐依据",
         note="6,613,329 B；本节点只做 1 MiB Range（未取全件）"),
    dict(sid="S7-04", source="S7 Allen BO · units.csv（51,439,189 B ⇒ 仅 HEAD）",
         category="allen_ecephys", kind="s3_object", mode="head",
         url=ALLEN_BO + "/visual-coding-neuropixels/ecephys-cache/units.csv",
         evidence_level="measured",
         wrong="把「1 MB 前缀够用」当真：M11 实测前缀覆盖率 0%（M11-R56）⇒ 前缀能读 schema 但取不到"
               "任何被引用的 unit id；也不得把它当自发率（口径含刺激期）",
         note="51,439,189 B；本节点不取（M11 已完成件，重复取无收益）；M11 实测 ETag 尾码 -7（part=8 MiB × 7）复算通过"),
    dict(sid="S7-05", source="S7 Allen BO · 单 probe LFP NWB（2,062,950,784 B > 2 GB）",
         category="allen_ecephys", kind="s3_object", mode="head",
         url=ALLEN_BO + "/visual-coding-neuropixels/ecephys-cache/session_715093703/"
                        "probe_810755797_lfp.nwb",
         evidence_level="measured",
         wrong="把它当可取 ⇒ 撞 >2 GB 单件上限（§2.2 S7 明文「合规不下载」）；若误取会占用数小时带宽并"
               "污染同期任何吞吐测量",
         note="§2.2 S7 登记的合规不下载件；本条只做对象级取证"),
    dict(sid="S7-06", source="S7 参照通道 · Visual Behavior Neuropixels 桶列举",
         category="allen_ecephys", kind="s3_list", mode="list",
         url=VBN + "/?list-type=2&max-keys=20&delimiter=/",
         evidence_level="measured",
         wrong="把该桶与 allen-brain-observatory 桶混用 ⇒ key 前缀不同（M11 L9.2 #2 同型陷阱：跨桶混用 = 口径错）",
         note="§2.2 S7 的第二通道（M11 实测可 list）"),
    dict(sid="S7-07", source="S7 参照通道 · DANDI 路由（api.dandiarchive.org）",
         category="allen_ecephys", kind="api_json", mode="head",
         url="https://api.dandiarchive.org/api/dandisets/000021/",
         evidence_level="measured",
         wrong="把早前 7,237 B/s 的外推当现状 ⇒ 会永久关闭一条本可解封的通道（本节点实测三条通道今日"
               "吞吐均比继承值高 13–456 倍）——以当场探针为准（R56/R63/R72）",
         note="§2.2 S7 明写「不尝试 DANDI 全量」；本条只探路由可达性，不取 NWB"),

    # ================= S8 药理/损伤扰动锚 =================
    dict(sid="S8-01", source="S8 文献锚 · Europe PMC REST 主通道（查询延迟复核）",
         category="literature_anchor", kind="literature", mode="epmc",
         url=EPMC + "?query=%22prefrontal%22%20AND%20%22lesion%22%20AND%20%22attention%22&"
                    "format=json&pageSize=1",
         evidence_level="measured",
         wrong="把 Europe PMC 的每次 1–2 s 当「文献档无穷快」⇒ 逐篇全文（PDF/补充材料）不走这条通道，"
               "两者成本差 2–3 个数量级；混用会把 ETA 估错",
         note="§2.2 S8 实测可用；本行复核延迟与返回码"),
    dict(sid="S8-02", source="S8 全文可得性 · PMC 全文 XML（开放获取部分）",
         category="literature_anchor", kind="literature", mode="head",
         url="https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6138437/fullTextXML",
         evidence_level="measured",
         wrong="把「题录命中」当「全文可得」⇒ §2.4 明确：文献锚不可得时该锚须降为 literature_unverified "
               "并排除出 P11 分母（未测 ≠ 未命中）",
         note="S8 降级路径①（付费墙 ⇒ 用 PMC 全文/补充材料）的第一跳"),
    dict(sid="S8-03", source="S8 全文可得性 · nature.com",
         category="literature_anchor", kind="publisher", mode="head",
         url="https://www.nature.com/articles/nature13186",
         evidence_level="measured",
         wrong="把「出版社页面可达」当「参数幅度值可得」⇒ 幅度量必须 value_verified 逐篇实测，"
               "否则不得作 M12 判据带（§0.8 EQ-2）",
         note="§2.2 S8 继承记录「nature.com 可达」⇒ 本条当场复核"),
    dict(sid="S8-04", source="S8 全文可得性 · science.org（继承记录：曾 403）",
         category="literature_anchor", kind="publisher", mode="head",
         url="https://www.science.org/doi/10.1126/science.1077081",
         evidence_level="measured",
         wrong="把 403 当「该文献不存在」（未测 ≠ 未命中）⇒ 应改走 PMC/补充材料，并在判据行如实登记缺这一维",
         note="§2.2 S8 继承记录；本条当场复核 403 是否仍在"),
    dict(sid="S8-05", source="S8 全文可得性 · cell.com（继承记录：曾超时）",
         category="literature_anchor", kind="publisher", mode="head",
         url="https://www.cell.com/cell/fulltext/S0092-8674(20)30342-8",
         evidence_level="measured",
         wrong="同上；且超时 ≠ 不存在——须以当场探针复核（本节点按硬回收口径给 15 s 窗），"
               "不得把一次超时写成永久不可得（M11-R56 的对偶）",
         note="§2.2 S8 继承记录；本条当场复核"),
    dict(sid="S8-06", source="S8 锚 · 预注册可用锚之一（PMID 37644210）题录",
         category="literature_anchor", kind="literature", mode="epmc",
         url=EPMC + "?query=EXT_ID:37644210&format=json&pageSize=1",
         evidence_level="literature",
         wrong="把题录当「参数已核实」⇒ P11 的幅度值必须逐条 value_verified，否则该锚不得进 P11 的分子",
         note="§2.2 S8 降级路径③ 列出的 5 条本轮已命中锚之一，逐条复核命中"),

    # ================= S9 MICrONS（复用 M11 落盘件） =================
    dict(sid="S9-01", source="S9 复用件 · M11 已落盘 MICrONS 结构/协议/参考件（零新增下载）",
         category="microns_reuse", kind="local_artifact", mode="local",
         url="neural_exploration/data/m11_microns_structure.json（+ m11_reference.csv 等）",
         evidence_level="measured",
         wrong="把「M11 已落盘」当零成本本身没错，但不得把 M11 的绝对数值移植为 M12 判据带"
               "（§0.8 EQ-2：跨尺度不可移植，M11 §10.4 明示 M11 的 S 门结论在 M12 必须重测）",
         note="S9 的获取方式 = 零新增下载；本条只核对本机件存在性（只读）"),
    dict(sid="S9-02", source="S9 永久不可得 · MICrONS 逐细胞 ΔF/F 数据库（103.8 GB docker tar）",
         category="microns_reuse", kind="permanent_limit", mode="derived",
         url="（无 URL；M11 §3p/§3r 裁决的永久 unavailable）",
         evidence_level="measured（继承 M11 裁决）",
         wrong="把「永久不可得」与「本轮未取」混为一谈 ⇒ 前者不得当借口，后者须随时可解封"
               "（§R41-2 条件 3 的同族要求）",
         note="103.8 GB ⇒ 结构性不可行；本节点不取"),
    dict(sid="S9-03", source="S9 永久不可得 · MICrONS 功能连接组学两表（496 MB）",
         category="microns_reuse", kind="permanent_limit", mode="derived",
         url="iarpa_microns/minnie/functional_data/functional_connectomics/node_and_edge_properties/v1/",
         evidence_level="measured（继承 M11 统计）",
         wrong="同上；另不得把 readme 的 schema 当数据（M11 只落了 readme 5,594 B）",
         note="node 294,635,032 B + edge 201,511,086 B = 496 MB（M11 实测）；本节点不取"),

    # ================= S10 / S11 / S12 =================
    dict(sid="S10-01", source="S10 区域几何/距离矩阵（由 S2 的 Structure 表派生）",
         category="derived_geometry", kind="derived", mode="derived",
         url="（无 URL；由 S2-01 的 model::Structure 全表派生）",
         evidence_level="derived",
         wrong="把「质心欧氏距离」当「体素接触面/纤维长度」⇒ 距离核的物理语义错；且 500×500 距离矩阵"
               "必须与区域映射口径同步（换 graph_id 就换矩阵）",
         note="随 S2 通过而通过（§2.2 S10）；本节点未实际构建矩阵（属 B3 装配件，此处只登记输入已可得）"),
    dict(sid="S11-01", source="S11 区域功能锚 · mPFC 持续注意（PMID 23073640）题录",
         category="literature_anchor", kind="literature", mode="epmc",
         url=EPMC + "?query=EXT_ID:23073640&format=json&pageSize=1",
         evidence_level="literature",
         wrong="把设计文档的举例（PFC → 工作记忆）当锚 ⇒ §1.3 反向纪律明写「以文献实测为准」，该锚实测"
               "支持「损伤持续注意而保留工作记忆」⇒ 判据须偏向持续注意/lapse，不写工作记忆缺陷",
         note="§1.3 表中 mPFC 行的依据；本条复核其题录命中状态"),
    dict(sid="S11-02", source="S11 区域功能锚 · 未命中登记（题录未命中 ≠ 不存在）",
         category="literature_anchor", kind="literature", mode="epmc",
         url=EPMC + "?query=TITLE:%22whole-brain%22%20AND%20TITLE:%22monoaminergic%22&"
                    "format=json&pageSize=1",
         evidence_level="literature",
         wrong="把 hitCount=0 写成「该区域角色无文献依据」⇒ §2.2 S11 明确：未命中的区域角色该行须 "
               "literature_unverified 且 final=0，该区域判据不入分母（未测 ≠ 未命中）",
         note="与 S4-02 同族；本行存在的意义就是留下「这是未命中，不是不存在」的记录"),
    dict(sid="S12-01", source="S12 任务/装置参照 · IBL 桶 atlas/ 前缀",
         category="task_reference", kind="s3_list", mode="list",
         url=IBL + "/?list-type=2&max-keys=20&delimiter=/&prefix=atlas/",
         evidence_level="measured",
         wrong="把 atlas/ 里的成像图谱件当「电生理区域标注」⇒ 两套坐标/口径不同"
               "（§0.8 EQ-1：区域映射口径必须逐字写明）",
         note="§2.2 S12 的候选通道（装置与刺激参数）"),
    dict(sid="S12-02", source="S12 任务/装置参照 · IBL 桶 resources/ 前缀",
         category="task_reference", kind="s3_list", mode="list",
         url=IBL + "/?list-type=2&max-keys=20&delimiter=/&prefix=resources/",
         evidence_level="measured",
         wrong="按「resources = 小文件」取数 ⇒ 本轮实测该前缀下含 1,744,400,771 B（约 1.74 GB）的 "
               "regressors_charles_1.5.0.tar.gz 等大件（§2.2 S12 估计「≤10 MB」已被实测否证）",
         note="★ 本条是对 §2.2 S12「预期小（≤10 MB）」的实测更正线索：resources/ 下有 GB 级对象"),
    dict(sid="S12-03", source="S12 任务/装置参照 · IBL 论文（PMID 34011433）题录",
         category="task_reference", kind="literature", mode="epmc",
         url=EPMC + "?query=EXT_ID:34011433&format=json&pageSize=1",
         evidence_level="literature",
         wrong="不注勘误（PMID 36301084）⇒ §2.2 S12 明写勘误指针；引用未勘误版会造成装置参数引用错版",
         note="§2.2 S12 的论文档主通道；本条复核命中并保留勘误指针"),
]
