"""词素拆分学习单元测试（v2 - 无状态查图谱）

验证：
- BootstrapTokenizer 仅做单字拆分（先天基础）
- LearnableTokenizer 未绑定图谱时退化为单字拆分（兼容孤立场景）
- 通过 SymbolicInterface 学习后分词正确（知识唯一来源是图谱）
- 数字合并规则存图谱，泛化到未见过的数字
- 多字词最长匹配优先（"等于"优先于"等"+"于"）
- 学习新多字词后能正确分词
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pytest

from digital_brain.src.core.memory.declarative_memory import DeclarativeMemory
from digital_brain.src.utils.tokenizer import (
    BootstrapTokenizer,
    LearnableTokenizer,
    Tokenizer,
)


# ==============================================================
# 引导层（Bootstrap）：最小启动能力
# ==============================================================

class TestBootstrap:
    def test_bootstrap_only_skips_and_chars(self):
        bt = BootstrapTokenizer()
        tokens, _ = bt.tokenize_raw("1+1=? 你好")
        # 空格被跳过，其他一切按单字
        assert tokens == ["1", "+", "1", "=", "?", "你", "好"]


# ==============================================================
# 无状态 Tokenizer：未绑定图谱时退化为单字
# ==============================================================

class TestStatelessFallback:
    def test_no_memory_degrades_to_single_char(self):
        """未绑定图谱 → 没有任何词素知识 → 全部按单字拆"""
        t = LearnableTokenizer()  # 默认 declarative=None
        assert t.merge_digit_sequences is False
        assert t.known_morphemes == set()
        # 没学时 999 被拆成 ['9','9','9']
        assert t.tokenize("999") == ["9", "9", "9"]
        # 一加一等于多少 全部按单字
        assert t.tokenize("一加一等于多少") == ["一", "加", "一", "等", "于", "多", "少"]


# ==============================================================
# 通过 SymbolicInterface 学习后分词正确（知识唯一来源是图谱）
# ==============================================================

class TestLearnThroughBrain:
    def test_brain_can_tokenize_after_learning_default_curriculum(self):
        """学习标准课程后能正确分词"""
        from digital_brain.src.interfaces.symbolic_interface import SymbolicInterface
        brain = SymbolicInterface(auto_build=True)
        # 学过 0-10 数字、+ - = ? 等词，能正确分词
        assert brain.tokenizer.tokenize("1+1=?") == ["1", "+", "1", "=", "?"]
        # 学过"等于""多少"作为别名（marker 的 aliases），能分多字词
        assert brain.tokenizer.tokenize("一加一等于多少") == ["一", "加", "一", "等于", "多少"]

    def test_digit_merge_learned_through_brain(self):
        """学过多位数后图谱中有 digit_merge 规则，能泛化到未见数字"""
        from digital_brain.src.interfaces.symbolic_interface import SymbolicInterface
        brain = SymbolicInterface(auto_build=True)
        # 学过数字 10（两位数）→ 图谱中有 digit_merge 规则
        assert brain.tokenizer.merge_digit_sequences is True
        # 泛化到未见过的 9876
        assert brain.tokenizer.tokenize("9876") == ["9876"]
        # 泛化到 77+88
        assert brain.tokenizer.tokenize("77+88") == ["77", "+", "88"]

    def test_learn_new_multichar_word_then_tokenize(self):
        """主动学习新的多字词（如"请问"）后能正确分词"""
        from digital_brain.src.interfaces.symbolic_interface import SymbolicInterface
        brain = SymbolicInterface(auto_build=True)
        # 学习前 "请问" 拆成单字（因为图谱里没有这个词）
        before = brain.tokenizer.tokenize("请问一加一")
        assert "请问" not in before
        # 通过 learn_tokenizer_example 学习
        brain.learn_tokenizer_example(
            "请问三加二等于几？",
            ["请问", "三", "加", "二", "等于", "几", "？"],
        )
        # 学习后能正确分词
        after = brain.tokenizer.tokenize("请问一加一")
        assert "请问" in after

    def test_longest_match_priority(self):
        """最长匹配优先：若学过"等于"则不拆成"等"+"于" """
        from digital_brain.src.interfaces.symbolic_interface import SymbolicInterface
        brain = SymbolicInterface(auto_build=True)
        # "等于" 作为 marker "=" 的别名已经在图谱中
        toks = brain.tokenizer.tokenize("等于")
        assert "等于" in toks


# ==============================================================
# 通过 DeclarativeMemory 直接测试 Tokenizer 无状态行为
# ==============================================================

class TestTokenizerWithMemory:
    def _make_brain_with_memory(self):
        """构造一个绑定了图谱的 Tokenizer（不通过 SymbolicInterface）"""
        dm = DeclarativeMemory()
        t = LearnableTokenizer(declarative_memory=dm)
        return t, dm

    def test_learn_example_writes_to_memory(self):
        """learn_from_example 把词素写入图谱"""
        t, dm = self._make_brain_with_memory()
        # 学习前图谱为空
        assert t.known_morphemes == set()
        # 学习一个样本
        stats = t.learn_from_example("一加一等于多少", ["一", "加", "一", "等于", "多少"])
        assert stats["n_added"] == 4   # 4 个新词素（"一"重复，只算 1 次）
        assert stats["ok"] is True
        assert t.learned_examples == 1
        # 学习后图谱中有这些词素
        assert "等于" in t.known_morphemes
        assert "多少" in t.known_morphemes
        # 能正确分词
        assert t.tokenize("一加一等于多少") == ["一", "加", "一", "等于", "多少"]

    def test_learn_digit_merge_and_generalize(self):
        """学过含 2 位数字的样本后，digit_merge 规则写入图谱，能泛化"""
        t, dm = self._make_brain_with_memory()
        # 学习前不会数字合并
        assert t.merge_digit_sequences is False
        # 学习"12+34"样本
        stats = t.learn_from_example("12+34", ["12", "+", "34"])
        assert stats["merge_digit_learned"] is True
        # 学习后规则在图谱中
        assert t.merge_digit_sequences is True
        # 泛化到未见过的 9876
        assert t.tokenize("9876") == ["9876"]
        # 泛化：77+88 正确
        assert t.tokenize("77+88") == ["77", "+", "88"]

    def test_learn_dataset_batch(self):
        """批量学习"""
        t, dm = self._make_brain_with_memory()
        dataset = [
            {"input": "1+1=?", "expected_tokens": ["1", "+", "1", "=", "?"]},
            {"input": "一加一等于多少", "expected_tokens": ["一", "加", "一", "等于", "多少"]},
            {"input": "12+34", "expected_tokens": ["12", "+", "34"]},
        ]
        stats = t.learn_from_dataset(dataset)
        assert stats["examples"] == 3
        assert stats["merge_digit_learned"] is True
        assert stats["conflicts"] == 0
        # 学完就能泛化
        assert t.tokenize("56+78") == ["56", "+", "78"]
        assert "等于" in t.tokenize("五减去二等于多少")

    def test_conflict_detection(self):
        """标注不一致时被记为冲突（不抛异常）"""
        t, dm = self._make_brain_with_memory()
        stats = t.learn_from_example("abc", ["ab"])  # "abc" != "ab" 拼接
        assert stats["ok"] is False
        assert len(t.conflicts) == 1


# ==============================================================
# 向后兼容：Tokenizer 类名别名
# ==============================================================

class TestBackwardCompatibility:
    def test_tokenizer_class_name_alias(self):
        # 默认 Tokenizer 就是 LearnableTokenizer
        assert Tokenizer is LearnableTokenizer
        t = Tokenizer()
        assert isinstance(t, LearnableTokenizer)


# ==============================================================
# 完整 E2E：通过 SymbolicInterface 学新词 → solve
# ==============================================================

class TestEndToEndLearningNewMorphemes:
    def test_teach_then_solve_chinese_phrase(self):
        """学习新的中文多字词素后，能正确拆分并解题。"""
        from digital_brain.src.interfaces.symbolic_interface import SymbolicInterface
        brain = SymbolicInterface(auto_build=True)
        question = "请问三加二等于几？"
        # 先教一遍"请问""等于""几"该怎么拆（多字词）
        brain.learn_tokenizer_example(
            "请问三加二等于几？",
            ["请问", "三", "加", "二", "等于", "几", "？"],
        )
        r = brain.solve(question)
        tokens = r.tokens
        # 应该学到了多字词"请问/等于"
        assert "请问" in tokens or "等于" in tokens
        # 加法答案：3 + 2 = 5
        assert r.answer == 5
