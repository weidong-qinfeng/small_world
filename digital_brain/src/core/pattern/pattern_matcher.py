"""模式匹配器 - 识别输入序列中的模式结构"""
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


class MathPattern(BaseModel):
    """内置的数学表达式模式定义"""
    name: str
    tokens_template: List[str] = Field(default_factory=list)   # 如 ["N", "+", "N", "=", "?"]
    description: str = ""


class PatternMatcher:
    """模式匹配器

    核心原则（"没有初始能力"）：分类 token 时必须先查 declarative memory 中"是否学过该词的含义"。
    没学过的词一律分类为 X（未知）。只有在传入 declarative_memory=None 的兼容旧模式下，
    才退化为用内置字符集硬编码分类（用于老测试 / KnowledgeBuilder 兼容）。
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
        self._patterns: Dict[str, MathPattern] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
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

    # ---------- 词素归类（核心：以 declarative memory 为准）----------
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
                # 其他已学实体也不归类为 OP/N/=/？，保持 X
                return "X"
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

    # ---------- 主匹配 ----------
    def match(self, tokens: List[str]) -> List[PatternMatchResult]:
        results: List[PatternMatchResult] = []
        tags = [self.classify_token(t) for t in tokens]
        # 1) 完整匹配 binary_op_question [N, OP, N, =, ?]
        result = self._try_match_binary_op_question(tokens, tags)
        if result:
            results.append(result)
        # 2) 匹配 binary_op [N, OP, N]（允许尾部有 =? 或其他）
        result = self._try_match_binary_op(tokens, tags)
        if result:
            results.append(result)
        # 排序：得分高的在前
        results.sort(key=lambda r: -r.match_score)
        return results

    def _try_match_binary_op_question(
        self, tokens: List[str], tags: List[str]
    ) -> Optional[PatternMatchResult]:
        """匹配 [N, OP, N, =, ?] 结构，长度5+"""
        n = len(tokens)
        if n < 3:
            return None
        # 在 tags 里滑动找序列 N-OP-N-(=?可选)-(?可选)
        for i in range(n - 2):
            if tags[i] == "N" and tags[i + 1] == "OP" and tags[i + 2] == "N":
                # 找后面的 = 和 ?
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
        """匹配 [N, OP, N]"""
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
                    matched_tokens=tokens[i : i + 3],
                    captured=captured,
                    span=(i, i + 3),
                )
        return None

    def best_match(self, tokens: List[str]) -> Optional[PatternMatchResult]:
        results = self.match(tokens)
        return results[0] if results else None
