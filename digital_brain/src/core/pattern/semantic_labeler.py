"""语义角色标注器（SRL） - M3 里程碑

设计哲学（对应 M3 方案）：
    Layer 1 的 slot 滑窗本质是"第 i 个 token 是什么类型"的词序模板，
    语序自由 / 成分省略 / 噪声插入时就会爆。
    Layer 2 把模式匹配从 词序（syntactic）升级到 语义角色（semantic）：
    不再按 token 顺序硬套槽位，而是先给每个 token 打语义标签（与顺序无关），
    再组块成语义组块（chunk），最后交给事件状态机做推理。

角色体系（9 类 + 辅助角色）：
    AGENT        施动者/主语（人、代词）
    THEME        主题/宾语（被操作的物品）
    LOCATION     位置/容器（书包里 / 在书包里）
    VERB_POSS    所属动词（有/拥有）
    VERB_ACQUIRE 获取动词（买/给/收到）
    VERB_LOSE    失去动词（借走/用掉/吃了）
    QUANTITY     数量组块（4本 / 100元 / 3支）
    MOD_TOTAL    总量标记（一共/总共/合计）
    MOD_TIME     时间标记（现在）
    Q_MARKER     疑问词（几/多少/?）
    ADV_AGAIN    累加副词（又）
    DATIVE       与格介词（给）
    NOISE        噪声/未知（请问/呀/嗯 等口语词）

标注依据：declarative memory 中实体的 pos / kind，与 token 出现顺序无关。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ============================================================
# 语义角色常量
# ============================================================
AGENT = "AGENT"
THEME = "THEME"
LOCATION = "LOCATION"
VERB_POSS = "VERB_POSS"
VERB_ACQUIRE = "VERB_ACQUIRE"
VERB_LOSE = "VERB_LOSE"
QUANTITY = "QUANTITY"
MOD_TOTAL = "MOD_TOTAL"
MOD_TIME = "MOD_TIME"
Q_MARKER = "Q_MARKER"
ADV_AGAIN = "ADV_AGAIN"
DATIVE = "DATIVE"
NOISE = "NOISE"


@dataclass
class Chunk:
    """语义组块：role + 覆盖的 token 序列 + 结构化信息"""

    role: str
    tokens: List[str] = field(default_factory=list)
    value: Optional[int] = None     # QUANTITY 的数值
    classifier: str = ""            # QUANTITY 的量词
    text: str = ""                  # 组块文本（tokens 拼接，调试用）

    def __post_init__(self) -> None:
        if not self.text and self.tokens:
            self.text = "".join(self.tokens)

    def first(self) -> str:
        return self.tokens[0] if self.tokens else ""


class SemanticLabeler:
    """语义角色标注器：tokens → 语义组块列表（与词序无关）"""

    # pos → 角色
    POS_TO_ROLE: Dict[str, str] = {
        "person_name": AGENT,
        "pron_person": AGENT,
        "pron_person_plural": AGENT,
        "noun": THEME,
        "verb_possess": VERB_POSS,
        "verb_acquire": VERB_ACQUIRE,
        "verb_residual": VERB_LOSE,
        "prep_dative": DATIVE,
        "adv_total": MOD_TOTAL,
        "adv_temporal": MOD_TIME,
        "adv_accumulate": ADV_AGAIN,
    }
    # 功能词性 → NOISE（不参与语义）
    NOISE_POS = {
        "classifier", "part_aspect", "prep_locative", "question_marker",
        "discourse_marker",
    }

    def __init__(self, declarative: Optional[Any] = None) -> None:
        self.declarative = declarative

    # ---------- 单 token 角色 ----------
    def role_of(self, token: str) -> str:
        if self.declarative is None:
            return NOISE
        ents = self.declarative.find_entity_by_name(token)
        if not ents:
            # 未学词：单字 CJK 视为噪声（宁可丢，不可错）
            return NOISE
        ent = ents[0]
        attrs = getattr(ent, "attributes", None) or {}
        kind = attrs.get("kind")
        if kind == "number":
            return "N"                      # 数字原始角色（组块阶段与量词合并）
        if kind == "marker":
            if attrs.get("marker_kind") == "question":
                return Q_MARKER
            return NOISE
        if kind == "operator":
            return NOISE
        pos = attrs.get("pos")
        if pos in self.POS_TO_ROLE:
            return self.POS_TO_ROLE[pos]
        if pos in self.NOISE_POS:
            return NOISE
        if kind in ("word", "morpheme"):
            return NOISE                     # 普通未标角色词 → 噪声
        return NOISE

    def is_number(self, token: str) -> bool:
        if self.declarative is None:
            return False
        ents = self.declarative.find_entity_by_name(token)
        for e in ents:
            if (getattr(e, "attributes", None) or {}).get("kind") == "number":
                return True
        # 多位数字串（digit_merge 规则学来的 morpheme）
        if len(token) > 1 and token.isdigit():
            return True
        return False

    def number_value(self, token: str) -> Optional[int]:
        if self.declarative is None:
            return None
        ents = self.declarative.find_entity_by_name(token)
        for e in ents:
            if (getattr(e, "attributes", None) or {}).get("kind") == "number":
                return e.attributes.get("value")
        if len(token) > 1 and token.isdigit():
            try:
                return int(token)
            except (ValueError, TypeError):
                return None
        return None

    def is_classifier(self, token: str) -> bool:
        if self.declarative is None:
            return False
        ents = self.declarative.find_entity_by_name(token)
        for e in ents:
            if (getattr(e, "attributes", None) or {}).get("pos") == "classifier":
                return True
        return False

    # ---------- 组块化 ----------
    def label(self, tokens: List[str]) -> List[Chunk]:
        """tokens → 语义组块列表。

        组块规则（顺序无关，只做相邻合并）：
          1. N + 量词 → QUANTITY(value, classifier)
          2. 其余角色各自成块；NOISE 丢弃（方位词里/中/在 等不参与推理，
             主题选择交给事件状态机按"数量之后"定位）
        """
        # 第一遍：逐 token 角色
        roles: List[str] = []
        for tok in tokens:
            if self.is_number(tok):
                roles.append("N")
            elif self.is_classifier(tok):
                roles.append("CLS")
            else:
                roles.append(self.role_of(tok))

        chunks: List[Chunk] = []
        i = 0
        n = len(tokens)
        while i < n:
            tok = tokens[i]
            role = roles[i]

            # 1) 数量组块：N [CLS]
            if role == "N":
                value = self.number_value(tok)
                cls = ""
                j = i + 1
                if j < n and roles[j] == "CLS":
                    cls = tokens[j]
                    j += 1
                chunks.append(Chunk(
                    role=QUANTITY, tokens=tokens[i:j],
                    value=value, classifier=cls,
                ))
                i = j
                continue

            # 2) 普通角色（单个 token 成块）
            if role != NOISE and role != "CLS":
                chunks.append(Chunk(role=role, tokens=[tok]))
            i += 1
        return chunks


# ============================================================
# 便捷：给定 tokens 输出 (role, token) 明细（调试/测试用）
# ============================================================
def label_flat(labeler: SemanticLabeler, tokens: List[str]) -> List[tuple]:
    """返回 [(role, token), ...] 扁平标注，不含 NOISE"""
    out: List[tuple] = []
    for tok in tokens:
        if labeler.is_number(tok):
            out.append(("N", tok))
        elif labeler.is_classifier(tok):
            out.append(("CLS", tok))
        else:
            r = labeler.role_of(tok)
            if r != NOISE:
                out.append((r, tok))
    return out
