"""意图识别器 - 从模式匹配和激活记忆中抽取用户意图"""
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, Field

from digital_brain.src.core.pattern.pattern_matcher import PatternMatchResult

if TYPE_CHECKING:
    from digital_brain.src.core.memory.declarative_memory import DeclarativeMemory


class IntentType(str, Enum):
    """意图类型枚举"""
    COMPUTE_BINARY = "compute_binary"       # 二元计算：A + B
    COMPUTE_COUNT = "compute_count"         # 计数
    COMPARE = "compare"                     # 比较
    UNKNOWN = "unknown"                     # 未知


class Intent(BaseModel):
    """识别出的意图"""
    intent_type: IntentType = IntentType.UNKNOWN
    confidence: float = 0.0
    slots: Dict[str, Any] = Field(default_factory=dict)
    description: str = ""


class IntentRecognizer:
    """意图识别器

    核心原则（"没有初始能力"）：
      - 操作符归一化（"+" -> "add"）必须通过查 declarative memory 实体 attributes.op_type
      - 中文/阿拉伯数字数值必须通过查 declarative memory 实体 attributes.numeric_value
      - 查不到实体的 token，一律不认为是已知数字/操作符

    只有当 declarative_memory=None（兼容旧模式）时，才退化为使用内置映射表。
    """

    # ---- 兼容 fallback：仅当 declarative_memory=None 时启用 ----
    OP_NORMALIZE_FALLBACK = {
        "+": "add", "＋": "add", "加": "add", "加上": "add",
        "-": "sub", "－": "sub", "减": "sub", "减去": "sub",
        "*": "mul", "×": "mul", "乘": "mul", "乘以": "mul",
        "/": "div", "÷": "div", "除": "div", "除以": "div",
    }
    CN_MAP_FALLBACK = {
        "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
        "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    }

    def __init__(
        self,
        declarative_memory: Optional["DeclarativeMemory"] = None,
    ) -> None:
        self.declarative = declarative_memory

    # ---------- 语义解析（以记忆系统为准）----------
    def _resolve_operator(self, op_token: str) -> Optional[str]:
        """给定一个 token，解析为 op_type(add/sub/mul/div)。没学过返回 None。"""
        if self.declarative is not None:
            entities = self.declarative.find_entity_by_name(op_token)
            if entities:
                ent = entities[0]
                if ent.attributes.get("kind") == "operator":
                    return ent.attributes.get("op_type")
            return None
        # fallback
        return self.OP_NORMALIZE_FALLBACK.get(op_token)

    def _resolve_number(self, token: Any) -> Optional[float]:
        """给定一个 token（字符串或数字），解析为数值。没学过返回 None。"""
        if token is None:
            return None
        if isinstance(token, (int, float)):
            # 传进来已经是数字：declarative 模式下仍然需要通过查实体确认
            # （否则 3 在没有教过"3 是数字 3"时也被信任了）
            if self.declarative is not None:
                # 如果是整数，尝试转字符串查实体
                s = str(int(token)) if float(token).is_integer() else str(token)
                entities = self.declarative.find_entity_by_name(s)
                if entities and entities[0].attributes.get("kind") == "number":
                    v = entities[0].attributes.get("numeric_value", entities[0].attributes.get("value"))
                    if v is not None:
                        return float(v)
                return None
            return float(token)
        if isinstance(token, str):
            token_s = token.strip()
            if self.declarative is not None:
                # 1) 先按 symbol 查实体（比如 "一" 是别名，也会被 find_entity_by_name 命中）
                entities = self.declarative.find_entity_by_name(token_s)
                if entities and entities[0].attributes.get("kind") == "number":
                    v = entities[0].attributes.get("numeric_value", entities[0].attributes.get("value"))
                    if v is not None:
                        return float(v)
                return None
            # fallback
            if token_s in self.CN_MAP_FALLBACK:
                return float(self.CN_MAP_FALLBACK[token_s])
            try:
                return float(token_s)
            except (ValueError, TypeError):
                return None
        return None

    def recognize(
        self,
        pattern_results: List[PatternMatchResult],
        tokens: List[str],
    ) -> Intent:
        if not pattern_results:
            return self._try_fallback(tokens)
        best = pattern_results[0]
        if best.pattern_name in ("binary_op_question", "binary_op"):
            return self._intent_from_binary_op(best)
        if best.pattern_name == "compare":
            return Intent(
                intent_type=IntentType.COMPARE,
                confidence=best.match_score,
                slots=best.captured,
                description=f"比较意图: {best.captured}",
            )
        return self._try_fallback(tokens)

    def _intent_from_binary_op(self, pr: PatternMatchResult) -> Intent:
        op_raw = pr.captured.get("op", "")
        op = self._resolve_operator(op_raw) if op_raw else None
        left = self._resolve_number(pr.captured.get("left"))
        right = self._resolve_number(pr.captured.get("right"))
        if not op or left is None or right is None:
            return Intent(
                intent_type=IntentType.UNKNOWN,
                confidence=0.2,
                slots=pr.captured,
                description="无法完整解析二元运算（有词未学过）",
            )
        intent_map = {
            "add": IntentType.COMPUTE_BINARY,
            "sub": IntentType.COMPUTE_BINARY,
            "mul": IntentType.COMPUTE_BINARY,
            "div": IntentType.COMPUTE_BINARY,
        }
        return Intent(
            intent_type=intent_map.get(op, IntentType.COMPUTE_BINARY),
            confidence=pr.match_score,
            slots={
                "operation": op,
                "operand_left": left,
                "operand_right": right,
                "op_raw": op_raw,
            },
            description=f"二元运算: {left} {op} {right}",
        )

    def _try_fallback(self, tokens: List[str]) -> Intent:
        """兜底：扫描 tokens 里有没有已学过的数字和操作符"""
        found_ops: List[str] = []
        found_nums: List[float] = []
        for t in tokens:
            n = self._resolve_number(t)
            if n is not None:
                found_nums.append(n)
                continue
            op = self._resolve_operator(t)
            if op:
                found_ops.append(op)
        if len(found_ops) == 1 and len(found_nums) >= 2:
            return Intent(
                intent_type=IntentType.COMPUTE_BINARY,
                confidence=0.5,
                slots={
                    "operation": found_ops[0],
                    "operand_left": found_nums[0],
                    "operand_right": found_nums[1],
                },
                description=f"兜底二元运算识别: {found_nums[0]} {found_ops[0]} {found_nums[1]}",
            )
        return Intent(intent_type=IntentType.UNKNOWN, confidence=0.0, description="无法识别意图")
