"""事件状态机（EventFSM） - M3 里程碑

输入：SRL 组块列表（每个语义事件块一组）
输出：问题 DAG（直接由原子操作节点构建，不再走 PatternMatcher 滑窗）

事件类型与 DAG 生成：
    POSSESSION    拥有陈述（小明有4本故事书）→ write_memory(owner, theme, count)
    ACQUISITION   获取事件（妈妈又给他买了3本）→ search_context(recipient)
                  → adding(collect theme) → write
    LOSS          失去事件（弟弟又借走了1本）→ read 最近拥有者 → subtracting → write
    QUERY_TOTAL   总量提问（现在小明总共有几本？）→ read_memory(subject, theme) → return
    （无提问块时，返回最近一次写入值）

设计要点：
    1. 语序无关：参数全部从 SRL 角色组块提取，不依赖 token 顺序
    2. 零指代：THEME/主语缺失时回退"最近 write_memory 的 entity/attr"（语义栈顶等价物）
    3. 货币量词：元/块钱/元钱 → 隐含宾语"钱"
    4. 任一事件块无法识别 → 返回 None，由调用方回退 PatternMatcher
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from digital_brain.src.core.pattern.semantic_labeler import (
    ADV_AGAIN,
    AGENT,
    DATIVE,
    LOCATION,
    MOD_TOTAL,
    NOISE,
    Q_MARKER,
    QUANTITY,
    THEME,
    VERB_ACQUIRE,
    VERB_LOSE,
    VERB_POSS,
    Chunk,
    SemanticLabeler,
)
from digital_brain.src.core.pattern.event_segmenter import EventSegmenter
from digital_brain.src.core.workspace.dag import (
    DAGGraph,
    DAGNode,
    OP_CALL_ALGORITHM,
    OP_READ_MEMORY,
    OP_RETURN_VALUE,
    OP_SEARCH_CONTEXT,
    OP_WRITE_MEMORY,
)

# 货币量词 → 隐含宾语"钱"
MONEY_CLASSIFIERS = {"元", "块钱", "元钱"}

# 事件类型常量
EV_POSSESSION = "POSSESSION"
EV_ACQUISITION = "ACQUISITION"
EV_LOSS = "LOSS"
EV_QUERY_TOTAL = "QUERY_TOTAL"


class EventFSM:
    """事件状态机：SRL 组块 → 事件 → DAG"""

    def __init__(
        self,
        declarative: Optional[Any] = None,
        working_memory: Optional[Any] = None,
    ) -> None:
        self.declarative = declarative
        self.working_memory = working_memory
        self.labeler = SemanticLabeler(declarative)
        self.segmenter = EventSegmenter(declarative)
        self._counter = 0

    # ---------- 主入口 ----------
    def build_dag(self, tokens: List[str], raw_text: Optional[str] = None) -> Optional[DAGGraph]:
        """tokens → 事件块 → DAG。任一事件块无法识别则返回 None（回退滑窗）。

        raw_text 提供时：按原始文本中的标点切分语段（tokenizer 会丢弃中文逗号，
        只有 raw_text 才能还原真实的子句边界），再对每个语段做 SRL 事件识别。
        """
        if raw_text is not None:
            blocks = self._clauses_from_raw(raw_text, tokens)
        else:
            blocks = self.segmenter.segment(tokens)
        if not blocks:
            return None

        dag = DAGGraph()
        write_ids: List[str] = []                 # 所有 write_memory 节点 id
        last_write: Optional[Dict[str, Any]] = None  # 最近一次写参数 {entity, attr}
        has_query = False
        query_ret_id: Optional[str] = None

        for block in blocks:
            chunks = self.labeler.label(block)
            if not chunks:
                continue  # 纯噪声块（请问/嗯/那个…）跳过
            etype = self._classify_event(chunks)
            if etype is None:
                return None  # 无法识别 → 整体回退 PatternMatcher
            params = self._extract_params(etype, chunks)
            if etype == EV_POSSESSION:
                nid = self._build_possession(dag, params, write_ids)
                if nid is None:
                    return None
                last_write = {"entity": params["owner"], "attr": params["theme"]}
            elif etype == EV_ACQUISITION:
                ok = self._build_acquisition(dag, params, write_ids, last_write)
                if not ok:
                    return None
                last_write = {"entity": f"$FIRST:{params['search_id']}", "attr": params["theme"]}
            elif etype == EV_LOSS:
                ok = self._build_loss(dag, params, write_ids, last_write)
                if not ok:
                    return None
                last_write = {"entity": params["target_entity"], "attr": params["target_attr"]}
            elif etype == EV_QUERY_TOTAL:
                ret = self._build_query_total(dag, params, write_ids, last_write)
                if ret is None:
                    return None
                has_query = True
                query_ret_id = ret

        # 无提问块：返回最近一次写入值（如 M3-a "…妈妈又给他买了3本" → 7）
        if not has_query:
            if last_write is None:
                return None
            ret = self._build_return_last_write(dag, last_write, write_ids)
            if ret is None:
                return None

        if dag.node_count == 0:
            return None
        return dag

    # ---------- 原始文本子句切分 ----------
    _PUNCT = set("，。；！？、!?,;")

    def _clauses_from_raw(self, raw_text: str, tokens: List[str]) -> List[List[str]]:
        """用原始文本中的标点还原子句边界，把 tokens 归入各子句。

        原理：tokenizer 丢弃了中文逗号，但 token 在原始文本中仍是连续子串；
        双指针同步扫描 raw_text 与 tokens，遇标点即切子句。
        """
        clauses: List[List[str]] = []
        cur: List[str] = []
        ti = 0
        i = 0
        n = len(raw_text)
        while i < n:
            ch = raw_text[i]
            if ch in self._PUNCT:
                if cur:
                    clauses.append(cur)
                    cur = []
                i += 1
                continue
            # 尝试在 i 处匹配下一个 token
            if ti < len(tokens):
                tok = tokens[ti]
                if raw_text[i:i + len(tok)] == tok:
                    cur.append(tok)
                    i += len(tok)
                    ti += 1
                    continue
            # 单个字符无法匹配当前 token：可能是空白或 token 内部差异，前进一个字符
            i += 1
        if cur:
            clauses.append(cur)
        return clauses

    # ---------- 事件分类 ----------
    def _classify_event(self, chunks: List[Chunk]) -> Optional[str]:
        roles = {c.role for c in chunks}
        has_q = Q_MARKER in roles
        has_poss = VERB_POSS in roles
        has_acq = VERB_ACQUIRE in roles
        has_lose = VERB_LOSE in roles
        has_total = MOD_TOTAL in roles
        has_adv = ADV_AGAIN in roles
        has_dative = DATIVE in roles
        has_quantity = any(c.role == QUANTITY for c in chunks)

        # 提问：疑问词 + (拥有动词 或 总量标记)
        if has_q and (has_poss or has_total):
            return EV_QUERY_TOTAL
        # 失去：失去动词 + 数量
        if has_lose and has_quantity:
            return EV_LOSS
        # 获取：获取动词 或 (又/给 + 数量)
        if has_quantity and (has_acq or has_dative or has_adv):
            return EV_ACQUISITION
        # 拥有：拥有动词 + 数量
        if has_poss and has_quantity:
            return EV_POSSESSION
        return None

    def _extract_params(self, etype: str, chunks: List[Chunk]) -> Dict[str, Any]:
        """按事件类型提取结构化参数（全部来自 SRL 角色组块，与词序无关）"""
        agents = self._agents(chunks)
        if etype == EV_POSSESSION:
            return {
                "owner": agents[0].first() if agents else None,
                "theme": self._theme_of(chunks),
                "count": self._quantity_of(chunks),
            }
        if etype == EV_ACQUISITION:
            return {
                "recipient": self._recipient_of(chunks),
                "giver": self._giver_of(chunks),
                "theme": self._theme_of(chunks),
                "count": self._quantity_of(chunks),
            }
        if etype == EV_LOSS:
            return {"count": self._quantity_of(chunks)}
        if etype == EV_QUERY_TOTAL:
            return {
                "subject": agents[0].first() if agents else None,
                "theme": self._theme_of(chunks),
            }
        return {}

    # ---------- 参数提取 ----------
    def _agents(self, chunks: List[Chunk]) -> List[Chunk]:
        return [c for c in chunks if c.role == AGENT]

    def _quantity_of(self, chunks: List[Chunk]) -> Optional[int]:
        for c in chunks:
            if c.role == QUANTITY and c.value is not None:
                return c.value
        return None

    def _theme_of(self, chunks: List[Chunk]) -> Optional[str]:
        """提取主题：
        1. 货币量词（元/块钱/元钱）→ "钱"（隐含宾语）
        2. 数量组块之后的 THEME（"4本故事书"→故事书；"50元红包"→红包）
           若数量之后无 THEME，取最后一个 THEME（"书包里有4本"→书包）
        """
        money = self._money_theme(chunks)
        if money:
            return money
        themes = [c for c in chunks if c.role == THEME]
        if not themes:
            return None
        # 数量组块之后的第一个 THEME 优先
        qty_idx = None
        for i, c in enumerate(chunks):
            if c.role == QUANTITY:
                qty_idx = i
                break
        if qty_idx is not None:
            after = [c for c in themes if chunks.index(c) > qty_idx]
            if after:
                return after[0].first()
        return themes[-1].first()

    def _money_theme(self, chunks: List[Chunk]) -> Optional[str]:
        for c in chunks:
            if c.role == QUANTITY and c.classifier in MONEY_CLASSIFIERS:
                return "钱"
        return None

    def _recipient_of(self, chunks: List[Chunk]) -> Optional[str]:
        """获取事件的接受者：
        1. DATIVE(给) 之后的 AGENT
        2. 否则代词 AGENT（他/她/它）
        3. 否则唯一 AGENT
        """
        agents = self._agents(chunks)
        if not agents:
            return None
        # DATIVE 之后第一个 AGENT
        dative_idx = None
        for i, c in enumerate(chunks):
            if c.role == DATIVE:
                dative_idx = i
                break
        if dative_idx is not None:
            for c in chunks[dative_idx + 1:]:
                if c.role == AGENT:
                    return c.first()
        # 代词优先（他/她/它 = 接受者；妈妈 = 给予者）
        pron = {"他", "她", "它", "他们", "她们", "它们"}
        for c in agents:
            if c.first() in pron:
                return c.first()
        if len(agents) == 1:
            # 唯一 AGENT：若它是"给"的施动者（有 DATIVE），则接受者被省略
            # （"妈妈又给3本" → 给谁？→ 语用默认：当前主题拥有者）
            if dative_idx is not None:
                return None
            return agents[0].first()
        # 多个非代词：取最后一个（妈妈给弟弟 → 弟弟是接受者）
        return agents[-1].first()

    def _giver_of(self, chunks: List[Chunk]) -> Optional[str]:
        recipient = self._recipient_of(chunks)
        for c in self._agents(chunks):
            if c.first() != recipient:
                return c.first()
        return None

    # ---------- DAG 构建 ----------
    def _new_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}{self._counter}"

    def _build_possession(
        self, dag: DAGGraph, params: Dict[str, Any],
        write_ids: List[str],
    ) -> Optional[str]:
        """拥有陈述 → write_memory(owner, theme, count)"""
        owner = params["owner"]
        theme = params["theme"]
        count = params["count"]
        nid = self._new_id("srl_w")
        dag.add_node(DAGNode(
            id=nid,
            action=OP_WRITE_MEMORY,
            params={"entity": owner, "attr": theme, "value": count},
            description=f"SRL拥有陈述：{owner}.{theme} = {count}",
        ))
        write_ids.append(nid)
        return nid

    def _build_acquisition(
        self, dag: DAGGraph, params: Dict[str, Any],
        write_ids: List[str], last_write: Optional[Dict[str, Any]],
    ) -> bool:
        """获取事件 → search(recipient) → adding(collect theme, count) → write

        零指代回退（语用默认）：
          - theme 缺失（"妈妈又给他买了3本"）→ 回退最近写入的 attr
          - recipient 缺失（"妈妈又给3本" → 给谁？）→ 回退最近写入的 entity
            （当前主题拥有者 = 小明），而不是把"妈妈"当接受者
        """
        recipient = params["recipient"]
        theme = params["theme"]
        if theme is None and last_write is not None:
            theme = last_write["attr"]
        if recipient is None and last_write is not None:
            recipient = last_write["entity"]
        count = params["count"]
        if recipient is None or theme is None or count is None:
            return False

        sid = self._new_id("srl_s")
        dag.add_node(DAGNode(
            id=sid,
            action=OP_SEARCH_CONTEXT,
            params={"keyword": recipient},
            depends_on=list(write_ids),
            description=f"SRL代词消解：接受者 '{recipient}' → 上下文实体",
        ))
        cid = self._new_id("srl_c")
        dag.add_node(DAGNode(
            id=cid,
            action=OP_CALL_ALGORITHM,
            params={
                "key": "adding",
                "collect_from": {"node": sid, "attr": theme},
                "args": [count],
            },
            depends_on=[sid],
            description=f"SRL获取：adding(已有{theme}, {count})",
        ))
        wid = self._new_id("srl_w")
        dag.add_node(DAGNode(
            id=wid,
            action=OP_WRITE_MEMORY,
            params={
                "entity": f"$FIRST:{sid}",
                "attr": theme,
                "value": f"${cid}",
            },
            depends_on=[cid],
            description=f"SRL更新：$FIRST:{sid}.{theme} = ${cid}",
        ))
        params["search_id"] = sid
        params["theme"] = theme
        write_ids.append(wid)
        return True

    def _build_loss(
        self, dag: DAGGraph, params: Dict[str, Any],
        write_ids: List[str], last_write: Optional[Dict[str, Any]],
    ) -> bool:
        """失去事件 → read 最近拥有者 → subtracting → write"""
        count = params["count"]
        if last_write is None:
            return False
        entity = last_write["entity"]
        attr = last_write["attr"]

        rid = self._new_id("srl_r")
        dag.add_node(DAGNode(
            id=rid,
            action=OP_READ_MEMORY,
            params={"entity": entity, "attr": attr},
            depends_on=list(write_ids),
            description=f"SRL读取：{entity}.{attr}",
        ))
        cid = self._new_id("srl_c")
        dag.add_node(DAGNode(
            id=cid,
            action=OP_CALL_ALGORITHM,
            params={"key": "subtracting", "args": [f"${rid}", count]},
            depends_on=[rid],
            description=f"SRL失去：subtracting(${rid}, {count})",
        ))
        wid = self._new_id("srl_w")
        dag.add_node(DAGNode(
            id=wid,
            action=OP_WRITE_MEMORY,
            params={"entity": entity, "attr": attr, "value": f"${cid}"},
            depends_on=[cid],
            description=f"SRL更新：{entity}.{attr} = ${cid}",
        ))
        params["target_entity"] = entity
        params["target_attr"] = attr
        write_ids.append(wid)
        return True

    def _build_query_total(
        self, dag: DAGGraph, params: Dict[str, Any],
        write_ids: List[str], last_write: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        """总量提问 → read(subject, theme) → return"""
        subject = params.get("subject")
        theme = params.get("theme")
        if subject is None and last_write is not None:
            subject = last_write["entity"]
        if theme is None and last_write is not None:
            theme = last_write["attr"]
        if subject is None or theme is None:
            return None

        rid = self._new_id("srl_r")
        dag.add_node(DAGNode(
            id=rid,
            action=OP_READ_MEMORY,
            params={"entity": subject, "attr": theme},
            depends_on=list(write_ids),
            description=f"SRL读取：{subject}.{theme}",
        ))
        ret = self._new_id("srl_ret")
        dag.add_node(DAGNode(
            id=ret,
            action=OP_RETURN_VALUE,
            params={"value": f"${rid}"},
            depends_on=[rid],
            description="SRL返回最终答案",
        ))
        return ret

    def _build_return_last_write(
        self, dag: DAGGraph, last_write: Dict[str, Any],
        write_ids: List[str],
    ) -> Optional[str]:
        """无提问块：读取最近写入并返回（M3-a 场景）"""
        rid = self._new_id("srl_r")
        dag.add_node(DAGNode(
            id=rid,
            action=OP_READ_MEMORY,
            params={"entity": last_write["entity"], "attr": last_write["attr"]},
            depends_on=list(write_ids),
            description=f"SRL读取：{last_write['entity']}.{last_write['attr']}",
        ))
        ret = self._new_id("srl_ret")
        dag.add_node(DAGNode(
            id=ret,
            action=OP_RETURN_VALUE,
            params={"value": f"${rid}"},
            depends_on=[rid],
            description="SRL返回最终答案",
        ))
        return ret
