"""词素拆分器 - 可学习版本

设计哲学（对应开发计划 §1.1 + §6.2）：
    所有能力通过学习获得，不用硬编码领域规则。

实现机制：
1. **引导层（最小启动）**：只做无关字符过滤（空格、标点跳过）。
   不包含任何"操作符/数字合并"等领域知识。
2. **学习层**：
   - `learn_from_example(input, expected_tokens)` 从单个样本归纳：
       a) 词素词典（morpheme lexicon）：记录每个出现过的 token。
       b) 结构规则：若样本中出现 >=2 位的连续数字作为一个 token，
          则学会"数字串要合并"。
   - `learn_from_dataset([...])`：批量学习。
3. **推理层 tokenize**：
   - 忽略无关字符（skip set）；
   - 若学会了数字合并规则，优先贪婪匹配整段数字；
   - 用"已知词素词典"做最长匹配（2-char 优先于单字）；
   - 退化兜底：单字切分（保证系统在学习不足时也能继续运行、后续再学）。
"""
from __future__ import annotations

import json
import os
import re
from typing import Iterable, List, Optional, Tuple


_DIGIT_RE = re.compile(r"\d+(?:\.\d+)?")

# 引导层：唯一"先天"的东西 —— 视觉/听觉上的无意义边界符号
# 这属于"物理引导"（§6.2 中"物理引导的边界学习"的先天基础），
# 不是领域知识。
_DEFAULT_SKIP_CHARS = set(" \t\n\r,，。.!！；;:：\"'()（）[]【】{}<>《》、/\\")


class BootstrapTokenizer:
    """最小引导分词器：只过滤无关字符，其余按单字拆分。

    用于：
    - 学习前的初始退化状态；
    - LearnableTokenizer 学到的词典处理不到时的最后兜底。
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
    """可学习词素拆分器 —— 所有领域词素通过学习获得"""

    DEFAULT_DATASET_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "data", "learning", "tokenization_examples.json",
    )

    def __init__(
        self,
        skip_chars: Optional[Iterable[str]] = None,
        auto_learn: bool = True,
        dataset_path: Optional[str] = None,
        *,
        merge_numbers: Optional[bool] = None,  # 向后兼容：等价于手动预设数字合并规则
    ) -> None:
        self._bootstrap = BootstrapTokenizer(skip_chars=skip_chars)
        # ---- 学到的知识 ----
        self.known_morphemes: set = set()                 # 已知词素集合
        self._morphemes_by_len: dict = {}                 # 按长度分组 -> list 方便最长匹配
        self.merge_digit_sequences: bool = False          # 学到的"数字合并"模式
        # 学习统计
        self.learned_examples: int = 0
        self.conflicts: List[dict] = []                   # 学习冲突（调试用）
        # ---- 兼容旧参数 merge_numbers ----
        # 旧测试 `Tokenizer(merge_numbers=True)` 等同于：预设已学会"数字要合并"
        if merge_numbers is not None:
            self.merge_digit_sequences = bool(merge_numbers)
        if auto_learn:
            self.learn_from_default_dataset(path=dataset_path)

    # =========================================================
    # 学习接口
    # =========================================================
    def learn_from_example(self, input_text: str, expected_tokens: List[str]) -> dict:
        """从单个 (input, expected_tokens) 样本学习

        Returns:
            stats dict: {n_added, merge_digit_learned, ok}
        """
        stats = {"n_added": 0, "merge_digit_learned": False, "ok": True, "example": input_text}
        # 1) 把每一个 expected token 记入词素词典
        for tok in expected_tokens:
            if not tok:
                continue
            if tok not in self.known_morphemes:
                self.known_morphemes.add(tok)
                stats["n_added"] += 1
            # 检测结构规则：>=2 位的数字串作为一个 token，说明"数字要合并"
            if len(tok) >= 2 and tok.isdigit() and not self.merge_digit_sequences:
                self.merge_digit_sequences = True
                stats["merge_digit_learned"] = True
            if len(tok) >= 2 and _DIGIT_RE.fullmatch(tok) and "." in tok and not self.merge_digit_sequences:
                # 带小数点的也学
                self.merge_digit_sequences = True
                stats["merge_digit_learned"] = True
        # 2) （可选的一致性检查）把 expected_tokens 拼回去应该能还原文本的非跳过部分
        normalized_expected = "".join(expected_tokens)
        normalized_input = "".join(
            ch for ch in input_text if ch not in self._bootstrap.skip_chars
        )
        if normalized_expected != normalized_input:
            stats["ok"] = False
            self.conflicts.append(
                {
                    "input": input_text,
                    "expected": expected_tokens,
                    "normalized_input": normalized_input,
                    "normalized_expected": normalized_expected,
                }
            )
        # 3) 重建 morphemes 分组
        self._rebuild_morpheme_index()
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
            # 允许路径不存在（空状态启动），返回空统计
            return {"examples": 0, "morphemes_added": 0}
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return self.learn_from_dataset(data)

    # =========================================================
    # 推理接口：分词
    # =========================================================
    def tokenize(self, text: str) -> List[str]:
        tokens: List[str] = []
        i = 0
        n = len(text)
        while i < n:
            c = text[i]
            if c in self._bootstrap.skip_chars:
                i += 1
                continue
            # 1) 若已学会数字合并：贪婪吞掉整个数字串
            if self.merge_digit_sequences and c.isdigit():
                m = _DIGIT_RE.match(text, i)
                if m:
                    num = m.group(0)
                    # 若数字串本身就是一个已知词素（如 "12"）直接收；
                    # 即便不是已知的，也按学到的"数字合并"结构规则产出（泛化）
                    tokens.append(num)
                    self.known_morphemes.add(num)
                    i = m.end()
                    continue
            # 2) 已知词素最长匹配：优先匹配更长的词素
            matched = self._longest_morpheme_match(text, i)
            if matched:
                tokens.append(matched)
                i += len(matched)
                continue
            # 3) 兜底：单字符 token
            tokens.append(c)
            # 自动把单字符记为已知词素（"独立符号识别"——§6.2）
            self.known_morphemes.add(c)
            i += 1
        # 每次分词后顺带更新索引（若单字符有新增）
        self._rebuild_morpheme_index()
        return tokens

    # =========================================================
    # 内部：最长匹配 & 索引
    # =========================================================
    def _rebuild_morpheme_index(self) -> None:
        d: dict = {}
        for mor in self.known_morphemes:
            L = len(mor)
            if L <= 1:
                # 单字符不走最长匹配分支，直接兜底处理
                continue
            d.setdefault(L, []).append(mor)
        self._morphemes_by_len = {L: set(mors) for L, mors in d.items()}

    def _longest_morpheme_match(self, text: str, start: int) -> Optional[str]:
        if not self._morphemes_by_len:
            return None
        max_len = max(self._morphemes_by_len.keys())
        # 从最长尝试到 2
        for L in range(max(max_len, 1), 1, -1):
            if L not in self._morphemes_by_len:
                continue
            piece = text[start : start + L]
            if piece in self._morphemes_by_len[L]:
                return piece
        return None


# ------------------------------------------------------------------
# 向后兼容：外部代码里 `from utils.tokenizer import Tokenizer` 不需改动
# 新默认 = 可学习版本
# ------------------------------------------------------------------
Tokenizer = LearnableTokenizer  # type: ignore[assignment,misc]
