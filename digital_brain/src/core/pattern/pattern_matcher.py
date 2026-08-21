"""模式匹配器 - v2

设计哲学（对应详细设计/模式匹配设计.md）：
    模式作为实体存于知识图谱，PatternMatcher 从图谱加载。
    词性分类纯查图谱（已学实体才知道是什么词性）。
    不再硬编码模式定义。

向后兼容：
    若 declarative_memory=None（孤立测试场景），退化为 v1 fallback 模式 + fallback 词性分类。
    生产环境（绑定图谱后）严格走图谱查询。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from digital_brain.src.core.memory.declarative_memory import DeclarativeMemory


class PatternMatchResult(BaseModel):
    """模式匹配结果"""
    pattern_name: str
    match_score: float = 0.0
    matched_tokens: List[str] = Field(default_factory=list)
    captured: Dict[str, Any] = Field(default_factory=dict)
    span: Tuple[int, int] = (0, 0)  # [start, end) 索引范围
    action: str = ""                # v2: 模式匹配后触发的动作（如 build_dag:binary_op）


class MathPattern(BaseModel):
    """内置的数学表达式模式定义（v1 兼容）"""
    name: str
    tokens_template: List[str] = Field(default_factory=list)
    description: str = ""


# v1 fallback 模式（仅 declarative=None 时使用）
_FALLBACK_PATTERNS: Dict[str, dict] = {
    "binary_op_question": {
        "slots": [
            {"type": "N", "capture": "left"},
            {"type": "OP", "capture": "op"},
            {"type": "N", "capture": "right"},
            {"type": "=", "capture": None},
            {"type": "?", "capture": None},
        ],
        "action": "compute_binary",
        "priority": 1.0,
        "description": "二元运算提问式：A + B = ?",
    },
    "binary_op": {
        "slots": [
            {"type": "N", "capture": "left"},
            {"type": "OP", "capture": "op"},
            {"type": "N", "capture": "right"},
        ],
        "action": "compute_binary",
        "priority": 0.8,
        "description": "二元运算：A + B",
    },
    "compare": {
        "slots": [
            {"type": "N", "capture": "left"},
            {"type": "CMP", "capture": None},
            {"type": "N", "capture": "right"},
        ],
        "action": "compare",
        "priority": 0.7,
        "description": "比较：A > B",
    },
}


class PatternMatcher:
    """模式匹配器

    v2 行为：
    - 启动时从图谱加载所有 kind="pattern" 的实体
    - 词性分类纯查图谱（已学实体）
    - 匹配时按 pattern.slots 滑动匹配 token 序列
    - 支持字面值约束（literal）和词性约束（type）

    v1 兼容：
    - 若 declarative=None，退化为 fallback 模式 + fallback 词性集合
    """

    # ---- 仅作兼容 fallback：仅当 declarative_memory 为 None 时启用 ----
    NUMBER_TOKENS_FALLBACK = set("0123456789零一二三四五六七八九十百千万")
    OP_TOKENS_FALLBACK = {"+", "-", "*", "/", "加", "减", "乘", "除", "＋", "－", "×", "÷"}
    EQ_TOKENS_FALLBACK = {"=", "＝", "等于", "是", "得"}
    Q_TOKENS_FALLBACK = {"?", "？", "多少", "几", "什么"}
    CMP_TOKENS_FALLBACK = {">", "<", "≥", "≤", "等于", "大于", "小于"}

    def __init__(
        self,
        declarative_memory: Optional["DeclarativeMemory"] = None,
    ) -> None:
        self.declarative = declarative_memory
        # v1 兼容：保留 _patterns 供老代码读取（但 v2 主要走图谱）
        self._patterns: Dict[str, MathPattern] = {}
        self._register_v1_defaults()

    def _register_v1_defaults(self) -> None:
        """v1 兼容：保留内置 MathPattern（仅供 _try_match_binary_op_question 等老方法使用）"""
        self._patterns["binary_op_question"] = MathPattern(
            name="binary_op_question",
            tokens_template=["N", "OP", "N", "=", "?"],
            description="二元运算提问式：A + B = ?",
        )
        self._patterns["binary_op"] = MathPattern(
            name="binary_op",
            tokens_template=["N", "OP", "N"],
            description="二元运算：A + B",
        )
        self._patterns["compare"] = MathPattern(
            name="compare",
            tokens_template=["N", "CMP", "N"],
            description="比较：A > B",
        )

    # ---------- M1-R2 匹配后清洗 ----------
    SUBJECT_BAD_POS = {
        "adv_temporal", "adv_total", "adv_accumulate",
        "part_aspect", "prep_locative", "prep_dative",
        "question_marker", "classifier",
    }

    OBJECT_BAD_POS = {
        "person_name",
        "adv_temporal", "adv_total", "adv_accumulate",
        "part_aspect", "prep_locative", "prep_dative",
        "question_marker", "classifier",
    }

    def _get_pos_of_word(self, word: str) -> Optional[str]:
        if self.declarative is None or not word:
            return None
        ents = self.declarative.find_entity_by_name(word)
        if not ents:
            return None
        attr = getattr(ents[0], "attributes", None) or {}
        return attr.get("pos")

    def _sanitize_match(self, match: Any) -> Optional[Any]:
        try:
            captured = match.captured if match.captured is not None else {}
            pname = match.pattern_name

            if pname == "total_question":
                subj = captured.get("subject")
                if subj:
                    sp = self._get_pos_of_word(subj)
                    if sp in self.SUBJECT_BAD_POS:
                        return None
                    if subj in ("?", "。", "，", ",", "！", "!", ".") or len(subj) == 1 and not ('\u4e00' <= subj <= '\u9fff' or subj.isalpha() or subj.isdigit()):
                        return None

            if pname in ("total_question", "acquire_event"):
                obj = captured.get("object")
                bad = False
                if obj is None:
                    bad = False
                elif obj in ("?", "。", "，", ",", "！", "!", ".") or (len(obj) == 1 and not ('\u4e00' <= obj <= '\u9fff' or obj.isalpha())):
                    bad = True
                else:
                    op = self._get_pos_of_word(obj)
                    if op in self.OBJECT_BAD_POS:
                        bad = True
                if bad:
                    captured["object"] = None
                    match.captured = captured
        except Exception:
            pass
        return match

    # ---------- 模式加载：从图谱读取 ----------
    def _load_patterns_from_memory(self) -> Dict[str, dict]:
        """从图谱加载所有 kind=pattern 实体。图谱为空或无模式时返回 v1 fallback。"""
        if self.declarative is None:
            return dict(_FALLBACK_PATTERNS)
        patterns: Dict[str, dict] = {}
        for e in self.declarative.find_entities_by_attr("kind", "pattern"):
            patterns[e.name] = {
                "slots": e.attributes.get("slots", []),
                "action": e.attributes.get("action", ""),
                "priority": e.attributes.get("priority", 0.5),
                "description": e.attributes.get("description", ""),
            }
        # 若图谱中无任何模式，退化为 fallback（保证基础可用）
        if not patterns:
            return dict(_FALLBACK_PATTERNS)
        return patterns

    # ---------- 词性归类（核心：以 declarative memory 为准）----------
    def classify_token(self, token: str) -> str:
        # 1) 优先查记忆系统：学过才知道是什么词
        if self.declarative is not None:
            entities = self.declarative.find_entity_by_name(token)
            if entities:
                ent = entities[0]
                kind = ent.attributes.get("kind")
                if kind == "number":
                    return "N"
                if kind == "operator":
                    return "OP"
                if kind == "marker":
                    mk = ent.attributes.get("marker_kind")
                    if mk == "equality":
                        return "="
                    if mk == "question":
                        return "?"
                    if mk in ("compare_gt", "compare_lt", "compare_eq", "compare"):
                        return "CMP"
                # 其他已学实体（word/morpheme）归类为 word
                if kind in ("word", "morpheme"):
                    # 特例：morpheme 是纯数字串时归为 N（如 tokenize 泛化产生的 "31"）
                    if kind == "morpheme" and self._is_number_str(token):
                        return "N"
                    return "word"
                # 未识别 kind 也归类为 word（兼容）
                return "word"
            # 记忆中查不到 -> 没学过 -> X
            return "X"

        # 2) 兼容 fallback：declarative=None 时用内置字符集（老测试场景）
        if token in self.OP_TOKENS_FALLBACK:
            return "OP"
        if token in self.EQ_TOKENS_FALLBACK:
            return "="
        if token in self.Q_TOKENS_FALLBACK:
            return "?"
        if token in self.CMP_TOKENS_FALLBACK:
            return "CMP"
        if token in self.NUMBER_TOKENS_FALLBACK or self._is_number_str(token):
            return "N"
        return "X"  # 其他

    @staticmethod
    def _is_number_str(token: str) -> bool:
        try:
            float(token)
            return True
        except (ValueError, TypeError):
            return False

    def classify_token_pos(self, token: str) -> str:
        """Phase 5d.2: 返回细粒度词性（person_name / noun / verb_possess / prep_locative / ...）。
        优先查 declarative_memory 中实体的 attributes.pos。
        未找到 pos 属性时，返回空字符串表示"未标注"。
        """
        if self.declarative is None:
            return ""
        entities = self.declarative.find_entity_by_name(token)
        for ent in entities:
            pos = ent.attributes.get("pos")
            if pos:
                return pos
        return ""

    # ---------- 主匹配 ----------
    def match(self, tokens: List[str]) -> List[PatternMatchResult]:
        """对 tokens 应用所有 pattern 做匹配，返回按得分排序的结果列表

        v2: 同一 pattern 在不同 span 命中时保留全部（用于多句应用题），
            仅去除完全相同的 (pattern_name, span) 重复。
        """
        patterns = self._load_patterns_from_memory()
        results: List[PatternMatchResult] = []
        for name, pat in patterns.items():
            for r in self._try_match_pattern(tokens, name, pat):
                # 填入 action 字段
                r.action = pat.get("action", "")
                results.append(r)
        # 仅去除完全相同的 (pattern_name, span_start) 重复
        seen: Dict[Tuple[str, int], PatternMatchResult] = {}
        for r in results:
            key = (r.pattern_name, r.span[0])
            cur = seen.get(key)
            if cur is None or r.match_score > cur.match_score:
                seen[key] = r
        results = list(seen.values())

        # M1-R4 重叠抑制：同一 pattern 在重叠 span 上多次命中时（如 possessive_state
        # 从 holder 滑到容器上重复匹配），只保留最完整（token 数最多、起始最早）的一条。
        # 非重叠的多句命中（如两条独立拥有陈述）不受影响。
        by_pattern: Dict[str, List[PatternMatchResult]] = {}
        for r in results:
            by_pattern.setdefault(r.pattern_name, []).append(r)
        suppressed: List[PatternMatchResult] = []
        for pname, group in by_pattern.items():
            group.sort(key=lambda r: (-len(r.matched_tokens), r.span[0]))
            kept: List[PatternMatchResult] = []
            for r in group:
                overlap = any(r.span[0] < k.span[1] and k.span[0] < r.span[1] for k in kept)
                if not overlap:
                    kept.append(r)
            suppressed.extend(kept)
        results = suppressed

        # M1-R2 清洗：丢弃语义错位匹配 + object bad pos → None
        cleaned = []
        for m in results:
            sm = self._sanitize_match(m)
            if sm is not None:
                cleaned.append(sm)
        results = cleaned

        results.sort(key=lambda r: -r.match_score)
        return results

    def _try_match_pattern(
        self, tokens: List[str], name: str, pat: dict
    ) -> List[PatternMatchResult]:
        """对 token 序列滑动匹配 pattern 的 slots"""
        slots = pat.get("slots", [])
        if not slots:
            return []
        n = len(tokens)
        L = len(slots)
        # 检查最少必填 slot 数
        min_required = sum(1 for s in slots if not s.get("optional", False))
        if n < min_required:
            return []

        results: List[PatternMatchResult] = []
        # 滑动起始位置
        max_start = n - min_required
        for start in range(max_start + 1):
            captured: Dict[str, Any] = {}
            matched_tokens: List[str] = []
            ok = True
            i = start
            j = 0
            while j < L:
                slot = slots[j]
                optional = slot.get("optional", False)
                if i >= n:
                    # token 用完：若剩余都是可选 slot 则跳过，否则失败
                    if optional:
                        cap = slot.get("capture")
                        if cap:
                            captured[cap] = None
                        j += 1
                        continue
                    ok = False
                    break
                tok = tokens[i]
                if self._slot_matches(tok, slot):
                    matched_tokens.append(tok)
                    cap = slot.get("capture")
                    if cap:
                        captured[cap] = tok
                    i += 1
                    j += 1
                elif optional:
                    # 可选 slot 不匹配则跳过该 slot（不消耗 token）
                    cap = slot.get("capture")
                    if cap:
                        captured[cap] = None
                    j += 1
                else:
                    ok = False
                    break
            if ok and j >= L:
                results.append(PatternMatchResult(
                    pattern_name=name,
                    match_score=float(pat.get("priority", 0.5)),
                    matched_tokens=matched_tokens,
                    captured=captured,
                    span=(start, start + len(matched_tokens)),
                ))
        return results

    def _slot_matches(self, token: str, slot: dict) -> bool:
        """检查 token 是否匹配 slot 约束"""
        # 1) 字面值约束（优先）
        literal = slot.get("literal")
        if literal is not None:
            return token == literal
        # 2) 词性约束
        type_ = slot.get("type")
        if type_ is None:
            return True
        if isinstance(type_, str) and type_.startswith("pos:"):
            expected_pos = type_[len("pos:"):]
            actual_pos = self.classify_token_pos(token)
            return actual_pos == expected_pos
        if type_ == "word":
            if self.declarative is None:
                return bool(token)
            ents = self.declarative.find_entity_by_name(token)
            if ents:
                pos = None
                for m in ents:
                    p = (getattr(m, "attributes", None) or {}).get("pos")
                    if p:
                        pos = p
                        break
                WORD_BLACKLIST = {
                    "classifier", "part_aspect",
                    "prep_locative", "prep_dative",
                    "adv_total", "adv_temporal", "adv_accumulate",
                    "verb_possess", "verb_acquire", "verb_residual",
                    "discourse_marker", "question_marker",
                }
                if pos in WORD_BLACKLIST:
                    return False
                return True
            if len(token) == 1:
                if '\u4e00' <= token <= '\u9fff' or '\u3400' <= token <= '\u4dbf':
                    return True
            return False
        # N/OP/=/等：通过 classify_token
        return self.classify_token(token) == type_

    def best_match(self, tokens: List[str]) -> Optional[PatternMatchResult]:
        results = self.match(tokens)
        return results[0] if results else None

    # ---------- v1 兼容老方法（保留供老代码调用）----------
    def _try_match_binary_op_question(
        self, tokens: List[str], tags: List[str]
    ) -> Optional[PatternMatchResult]:
        n = len(tokens)
        if n < 3:
            return None
        for i in range(n - 2):
            if tags[i] == "N" and tags[i + 1] == "OP" and tags[i + 2] == "N":
                j = i + 3
                has_eq = False
                has_q = False
                while j < n:
                    if tags[j] == "=":
                        has_eq = True
                    elif tags[j] == "?":
                        has_q = True
                    j += 1
                score = 0.6
                captured = {
                    "left": tokens[i],
                    "op": tokens[i + 1],
                    "right": tokens[i + 2],
                }
                span_end = min(n, i + 5)
                matched = tokens[i:span_end]
                if has_eq:
                    score += 0.2
                if has_q:
                    score += 0.2
                return PatternMatchResult(
                    pattern_name="binary_op_question",
                    match_score=round(score, 3),
                    matched_tokens=matched,
                    captured=captured,
                    span=(i, span_end),
                )
        return None

    def _try_match_binary_op(
        self, tokens: List[str], tags: List[str]
    ) -> Optional[PatternMatchResult]:
        n = len(tokens)
        for i in range(n - 2):
            if tags[i] == "N" and tags[i + 1] == "OP" and tags[i + 2] == "N":
                captured = {
                    "left": tokens[i],
                    "op": tokens[i + 1],
                    "right": tokens[i + 2],
                }
                return PatternMatchResult(
                    pattern_name="binary_op",
                    match_score=0.8,
                    matched_tokens=tokens[i: i + 3],
                    captured=captured,
                    span=(i, i + 3),
                )
        return None
