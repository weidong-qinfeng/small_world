"""词素拆分器 - 无状态执行引擎（v2）

设计哲学（对应详细设计/分词引擎设计.md）：
    Tokenizer 是无状态的"执行引擎"，不是知识存储。
    词素知识唯一来源是知识图谱（陈述性记忆）。
    Tokenizer 不学不会，也不存任何领域知识。

实现机制：
1. **引导层（先天基础）**：只做无关字符过滤（空格、标点跳过）。
   不包含任何"操作符/数字合并"等领域知识。
2. **查询层**：
   - tokenize 时从知识图谱查询所有已知词素（实体的 name + aliases）
   - 从知识图谱查询"数字合并"规则实体
   - 用已知词素做最长匹配（2-char 优先于单字）
   - 退化兜底：单字切分（保证系统在知识不足时也能继续运行）
3. **学习层**：
   - learn_from_example(input, expected_tokens) 把样本中的词素写入图谱
   - 若样本含 >=2 位数字串，写入"digit_merge"规则实体
   - 学习行为是"写入图谱"，不是"存到内部 set"

向后兼容：
    - LearnableTokenizer 保留旧 API 名称（learn_from_example / learn_from_dataset），
      但内部行为已改为"写入图谱"。
    - known_morphemes / merge_digit_sequences 改为只读 property，从图谱派生。
    - 旧测试代码中 `t.known_morphemes.add(...)` / `t.merge_digit_sequences = True`
      等修改操作不再被支持（违反"知识唯一来源是图谱"原则）。
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from digital_brain.src.core.models import Entity, EntityType


_DIGIT_RE = re.compile(r"\d+(?:\.\d+)?")

# 引导层：唯一"先天"的东西 —— 视觉/听觉上的无意义边界符号
# 这属于"物理引导"（§6.2 中"物理引导的边界学习"的先天基础），
# 不是领域知识。
_DEFAULT_SKIP_CHARS = set(" \t\n\r,，。!！；;:：\"'()（）[]【】{}<>《》、/\\")

# 词素实体在知识图谱中的 attributes.kind 标记
_MORPHEME_KINDS = {"number", "operator", "marker", "word", "morpheme"}
# 规则实体的 attributes.kind 标记
_RULE_KIND = "rule"
# "数字合并"规则实体 attributes.rule_type
_DIGIT_MERGE_RULE = "digit_merge"


class BootstrapTokenizer:
    """最小引导分词器：只过滤无关字符，其余按单字拆分。

    用于：
    - 学习前的初始退化状态；
    - LearnableTokenizer 兜底处理（图谱中查不到时按单字拆）。
    """

    def __init__(self, skip_chars: Optional[Iterable[str]] = None) -> None:
        self.skip_chars: set = set(skip_chars) if skip_chars else set(_DEFAULT_SKIP_CHARS)

    def tokenize_raw(self, text: str) -> Tuple[List[str], List[int]]:
        """逐字拆分（跳过 skip chars），同时返回每个 token 的起始索引"""
        tokens: List[str] = []
        indices: List[int] = []
        for i, c in enumerate(text):
            if c in self.skip_chars:
                continue
            tokens.append(c)
            indices.append(i)
        return tokens, indices


class LearnableTokenizer:
    """可学习词素拆分器 —— 无状态执行引擎（v2）

    所有词素知识来源于绑定的 declarative_memory（知识图谱）。
    没有绑定图谱时退化为单字拆分（仅用于无 brain 的孤立测试场景）。
    """

    DEFAULT_DATASET_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "data", "learning", "tokenization_examples.json",
    )

    def __init__(
        self,
        skip_chars: Optional[Iterable[str]] = None,
        auto_learn: bool = False,
        dataset_path: Optional[str] = None,
        declarative_memory: Optional[Any] = None,
        *,
        merge_numbers: Optional[bool] = None,  # 向后兼容：True 时在图谱中写入 digit_merge 规则
    ) -> None:
        self._bootstrap = BootstrapTokenizer(skip_chars=skip_chars)
        self.declarative = declarative_memory  # 知识图谱引用（可空）
        # 学习统计（这是"机制状态"而非"知识"，允许保留）
        self.learned_examples: int = 0
        self.conflicts: List[dict] = []  # 学习冲突（调试用）
        # 兼容旧参数 merge_numbers：若为 True 且绑定了图谱，则写入 digit_merge 规则
        if merge_numbers and self.declarative is not None:
            self._ensure_digit_merge_rule()
        # 兼容旧参数 auto_learn=True：从默认数据集加载样本
        # （会通过 learn_from_example 写入图谱；若未绑定图谱则跳过）
        if auto_learn:
            self.learn_from_default_dataset(path=dataset_path)

    # =========================================================
    # 只读属性：从图谱派生（兼容旧 API）
    # =========================================================
    @property
    def known_morphemes(self) -> set:
        """已知词素集合（从图谱派生，只读）"""
        if self.declarative is None:
            return set()
        result: set = set()
        for e in self.declarative._entities.values():  # type: ignore[attr-defined]
            if e.attributes.get("kind") in _MORPHEME_KINDS:
                result.add(e.name)
                for a in e.aliases:
                    result.add(a)
        return result

    @property
    def merge_digit_sequences(self) -> bool:
        """是否学过"数字合并"规则（从图谱派生，只读）"""
        if self.declarative is None:
            return False
        return self.declarative.has_entity_with_attr("kind", _RULE_KIND) and \
            bool(self.declarative.find_entities_by_attr("rule_type", _DIGIT_MERGE_RULE))

    @property
    def _morphemes_by_len(self) -> Dict[int, set]:
        """按长度分组的词素索引（从图谱派生）"""
        d: Dict[int, set] = {}
        for mor in self.known_morphemes:
            L = len(mor)
            if L <= 1:
                continue
            d.setdefault(L, set()).add(mor)
        return d

    # =========================================================
    # 学习接口：把样本中的词素写入图谱
    # =========================================================
    def learn_from_example(self, input_text: str, expected_tokens: List[str]) -> dict:
        """从单个 (input, expected_tokens) 样本学习

        若绑定了图谱：把每个 expected_token 作为词素实体写入图谱
                     （已存在则跳过，避免重复）；
                     若样本中含 >=2 位数字串，写入 digit_merge 规则实体。
        若未绑定图谱：仅记录到 learned_examples / conflicts（兼容孤立测试）。

        Returns:
            stats dict: {n_added, merge_digit_learned, ok}
        """
        stats: Dict[str, Any] = {
            "n_added": 0, "merge_digit_learned": False, "ok": True, "example": input_text,
        }

        # 1) 把每个 expected token 写入图谱
        for tok in expected_tokens:
            if not tok:
                continue
            if self.declarative is not None:
                added = self._ensure_morpheme_entity(tok)
                if added:
                    stats["n_added"] += 1
            # 检测数字合并规则
            if len(tok) >= 2 and tok.isdigit() and not stats["merge_digit_learned"]:
                if self.declarative is not None:
                    if self._ensure_digit_merge_rule():
                        stats["merge_digit_learned"] = True
                else:
                    stats["merge_digit_learned"] = True  # 兼容孤立测试

        # 2) 一致性检查：拼回去应该能还原文本的非跳过部分
        normalized_expected = "".join(expected_tokens)
        normalized_input = "".join(
            ch for ch in input_text if ch not in self._bootstrap.skip_chars
        )
        if normalized_expected != normalized_input:
            stats["ok"] = False
            self.conflicts.append({
                "input": input_text,
                "expected": expected_tokens,
                "normalized_input": normalized_input,
                "normalized_expected": normalized_expected,
            })

        # 3) 学习计数
        self.learned_examples += 1
        return stats

    def learn_from_dataset(self, examples: Iterable[dict]) -> dict:
        """批量学习。examples 元素格式: {"input": ..., "expected_tokens": [...]}"""
        total_added = 0
        total = 0
        merge_learned = False
        for ex in examples:
            if "input" not in ex or "expected_tokens" not in ex:
                continue
            st = self.learn_from_example(ex["input"], ex["expected_tokens"])
            total_added += st["n_added"]
            merge_learned = merge_learned or st["merge_digit_learned"]
            total += 1
        return {
            "examples": total,
            "morphemes_added": total_added,
            "total_morphemes_now": len(self.known_morphemes),
            "merge_digit_learned": merge_learned,
            "conflicts": len(self.conflicts),
        }

    def learn_from_default_dataset(self, path: Optional[str] = None) -> dict:
        p = path or self.DEFAULT_DATASET_PATH
        if not os.path.exists(p):
            return {"examples": 0, "morphemes_added": 0}
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return self.learn_from_dataset(data)

    # =========================================================
    # 推理接口：分词
    # =========================================================
    def tokenize(self, text: str) -> List[str]:
        # 未绑定图谱 → 退化兜底：仅按单字拆分（跳过 skip chars）
        if self.declarative is None:
            tokens, _ = self._bootstrap.tokenize_raw(text)
            return tokens

        # 已绑定图谱 → 从图谱查词素 + 规则，做最长匹配
        morphemes_by_len = self._morphemes_by_len
        merge_digits = self.merge_digit_sequences

        tokens: List[str] = []
        i = 0
        n = len(text)
        while i < n:
            c = text[i]
            if c in self._bootstrap.skip_chars:
                i += 1
                continue
            # 1) 若学过数字合并：贪婪吞掉整个数字串
            if merge_digits and c.isdigit():
                m = _DIGIT_RE.match(text, i)
                if m:
                    num = m.group(0)
                    tokens.append(num)
                    # 若数字串本身不是已知词素，作为新词素写入图谱（泛化学习）
                    self._ensure_morpheme_entity(num)
                    i = m.end()
                    continue
            # 2) 已知词素最长匹配
            matched = self._longest_morpheme_match(text, i, morphemes_by_len)
            if matched:
                tokens.append(matched)
                i += len(matched)
                continue
            # 3) 兜底：单字符 token
            tokens.append(c)
            i += 1
        return tokens

    # =========================================================
    # 内部：把词素/规则写入图谱
    # =========================================================
    def _ensure_morpheme_entity(self, morpheme: str) -> bool:
        """把 morpheme 作为词素实体写入图谱。已存在则跳过。

        返回 True 表示新增，False 表示已存在。
        """
        if self.declarative is None:
            return False
        # 已有同名实体（name 或 alias）→ 跳过
        existing = self.declarative.find_entity_by_name(morpheme)
        if existing:
            return False
        # 创建 kind="morpheme" 实体
        eid = f"morph_{abs(hash(morpheme)) % 100000:05d}_{morpheme[:8]}"
        entity = Entity(
            id=eid,
            name=morpheme,
            aliases=[],
            entity_type=EntityType.ABSTRACT,
            attributes={"kind": "morpheme"},
        )
        try:
            self.declarative.add_entity(entity)
            return True
        except ValueError:
            return False

    def _ensure_digit_merge_rule(self) -> bool:
        """把 digit_merge 规则写入图谱。已存在则跳过。"""
        if self.declarative is None:
            return False
        existing = self.declarative.find_entities_by_attr("rule_type", _DIGIT_MERGE_RULE)
        if existing:
            return False
        entity = Entity(
            id="rule_digit_merge",
            name="digit_merge_rule",
            aliases=[],
            entity_type=EntityType.ABSTRACT,
            attributes={"kind": _RULE_KIND, "rule_type": _DIGIT_MERGE_RULE,
                        "description": "数字串要合并为一个 token"},
        )
        try:
            self.declarative.add_entity(entity)
            return True
        except ValueError:
            return False

    def _longest_morpheme_match(
        self, text: str, start: int, morphemes_by_len: Dict[int, set]
    ) -> Optional[str]:
        if not morphemes_by_len:
            return None
        max_len = max(morphemes_by_len.keys())
        for L in range(max(max_len, 1), 1, -1):
            if L not in morphemes_by_len:
                continue
            piece = text[start: start + L]
            if piece in morphemes_by_len[L]:
                return piece
        return None


# ------------------------------------------------------------------
# 向后兼容：外部代码里 `from utils.tokenizer import Tokenizer` 不需改动
# ------------------------------------------------------------------
Tokenizer = LearnableTokenizer  # type: ignore[assignment,misc]
