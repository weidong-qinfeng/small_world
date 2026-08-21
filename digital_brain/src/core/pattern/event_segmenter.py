"""事件切分器（EventSegmenter） - M3 里程碑

把一句话切分为独立的语义事件块（语段）：
    陈述句（拥有）→ POSSESSION
    增量句（获取）→ ACQUISITION
    减量句（失去）→ LOSS
    提问句（问总量）→ QUERY_TOTAL

切分规则（与词序无关）：
    1. 标点边界：，。；！？ 等 → 直接切块
    2. 新施动者边界：当前块已完成一个"拥有陈述"（有 语义动词 + 数量）后，
       若再出现新的人名/代词，说明开始了下一个分句（如"小明有4本故事书
       妈妈又给他3本"这种无标点连写），在新的人名处切块。
    3. 噪声前缀（请问/嗯/那个…）留在当前块内，由 SRL 组块时丢弃。

输出：List[List[str]]，每个子列表是一个事件块的 token 序列。
"""
from __future__ import annotations

from typing import Any, List, Optional

# 标点断句符（全角+半角）
_PUNCT_BREAKS = set("，。；！？、!?,;")

# 施动者词性：人名 / 代词 → 可能是新分句主语
_AGENT_POS = {"person_name", "pron_person", "pron_person_plural"}

# 拥有动词：看到它说明是"拥有陈述"，配合数量构成完整事件
_POSS_VERB_POS = {"verb_possess"}


class EventSegmenter:
    """事件切分器：tokens → 事件块列表（每块一个语义事件）"""

    def __init__(self, declarative: Optional[Any] = None) -> None:
        self.declarative = declarative

    def _pos_of(self, token: str) -> str:
        if self.declarative is None:
            return ""
        ents = self.declarative.find_entity_by_name(token)
        for e in ents:
            pos = (getattr(e, "attributes", None) or {}).get("pos")
            if pos:
                return pos
        return ""

    def _is_number(self, token: str) -> bool:
        if self.declarative is None:
            return token.isdigit()
        ents = self.declarative.find_entity_by_name(token)
        for e in ents:
            if (getattr(e, "attributes", None) or {}).get("kind") == "number":
                return True
        return len(token) > 1 and token.isdigit()

    def segment(self, tokens: List[str]) -> List[List[str]]:
        """按标点 + 新施动者边界切分为事件块。"""
        blocks: List[List[str]] = []
        cur: List[str] = []
        has_poss_verb = False    # 当前块是否已出现"拥有动词"
        has_quantity = False     # 当前块是否已出现数量

        def flush() -> None:
            nonlocal cur, has_poss_verb, has_quantity
            if cur:
                blocks.append(cur)
            cur = []
            has_poss_verb = False
            has_quantity = False

        for tok in tokens:
            if tok in _PUNCT_BREAKS:
                flush()
                continue
            pos = self._pos_of(tok)
            if self._is_number(tok):
                has_quantity = True
            elif pos in _POSS_VERB_POS:
                has_poss_verb = True
            # 新施动者边界：当前块已完成拥有陈述（有动词+有数量），
            # 又出现新的人名/代词 → 开始新分句
            if (pos in _AGENT_POS and cur and has_poss_verb and has_quantity):
                flush()
            cur.append(tok)
        flush()
        return blocks
