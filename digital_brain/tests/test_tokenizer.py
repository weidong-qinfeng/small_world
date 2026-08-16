"""词素拆分学习单元测试（开发计划 §6.2）

验证：
- 学习前 vs 学习后 的分词行为差异（证明能力不是先天的）
- 从单个样本归纳多字词素
- 从数字合并样本学到"数字串合并"规则并泛化到未见过的数字
- 批量学习（load_from_dataset）
- 独立符号识别（未知字符首次出现后被记住）
- 上下文关联：长词优先匹配（如"等于"优先于"等"/"于"）
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pytest

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

    def test_bootstrap_knows_no_digit_merge(self):
        """先天没有"数字合并"概念"""
        t = LearnableTokenizer(auto_learn=False)  # 完全空白
        assert t.merge_digit_sequences is False
        assert t.tokenize("1234") == ["1", "2", "3", "4"]


# ==============================================================
# 单样本学习能力
# ==============================================================

class TestLearnFromExample:
    def test_before_after_one_example(self):
        """学习前："等于/多少" 拆成单字 → 学 1 个样本后：合并"""
        t = LearnableTokenizer(auto_learn=False)
        text = "一加一等于多少"
        # ---- 学习前 ----
        assert t.tokenize(text) == ["一", "加", "一", "等", "于", "多", "少"]
        # ---- 学习 ----
        stats = t.learn_from_example(text, ["一", "加", "一", "等于", "多少"])
        assert stats["n_added"] == 2   # 等于、多少 是新词素
        assert stats["merge_digit_learned"] is False
        assert stats["ok"] is True
        assert t.learned_examples == 1
        # ---- 学习后 ----
        assert t.tokenize(text) == ["一", "加", "一", "等于", "多少"]

    def test_learn_digit_merge_and_generalize(self):
        t = LearnableTokenizer(auto_learn=False)
        # 没学时 999 被拆成 ['9','9','9']
        assert "".join(t.tokenize("999")) == "999"
        assert len(t.tokenize("999")) == 3
        # 学一个"12+34"样本 —— 里面有 12、34 两个 >=2 位数字串
        stats = t.learn_from_example("12+34", ["12", "+", "34"])
        assert stats["merge_digit_learned"] is True
        # 泛化：从未见过的 9876 正确合并成 1 个 token
        assert t.tokenize("9876") == ["9876"]
        # 泛化：77+88 正确
        assert t.tokenize("77+88") == ["77", "+", "88"]

    def test_longest_match_priority(self):
        """最长匹配优先：若学会"等于"则不拆成"等"+"于" """
        t = LearnableTokenizer(auto_learn=False)
        t.learn_from_example("X", ["等于", "于"])  # 同时学会 2-char 和 1-char
        # 上下文：出现"等于"时 2-char 优先
        assert "等于" in t.tokenize("等于a")

    def test_conflict_detection(self):
        """标注不一致时被记为冲突（不抛异常）"""
        t = LearnableTokenizer(auto_learn=False)
        stats = t.learn_from_example("abc", ["ab"])  # "abc" != "ab" 拼接
        assert stats["ok"] is False
        assert len(t.conflicts) == 1


# ==============================================================
# 批量学习 + 默认数据集
# ==============================================================

class TestLearnFromDataset:
    def test_learn_from_list(self):
        t = LearnableTokenizer(auto_learn=False)
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

    def test_auto_learn_from_default_dataset_on_init(self):
        """构造时默认加载 tokenization_examples.json"""
        t = LearnableTokenizer(auto_learn=True)
        assert t.learned_examples >= 4   # 预设 4 个样本
        assert t.merge_digit_sequences is True
        assert "等于" in t.known_morphemes
        assert "多少" in t.known_morphemes

    def test_default_dataset_cover_all_basic_cases(self):
        """加载默认数据集后，预设 4 个样本都能正确分词"""
        t = LearnableTokenizer(auto_learn=True)
        cases = [
            ("1+1=?", ["1", "+", "1", "=", "?"]),
            ("1 + 1 = ？", ["1", "+", "1", "=", "？"]),
            ("一加一等于多少", ["一", "加", "一", "等于", "多少"]),
            ("12+34", ["12", "+", "34"]),
        ]
        for inp, expected in cases:
            assert t.tokenize(inp) == expected, f"Failed on {inp}"


# ==============================================================
# 独立符号识别（§6.2）
# ==============================================================

class TestIndependentSymbolRecognition:
    def test_unknown_symbol_learned_after_first_encounter(self):
        t = LearnableTokenizer(auto_learn=False)
        assert "@" not in t.known_morphemes
        t.tokenize("x@y")        # 首次遇到 @，被当单字兜底添加
        assert "@" in t.known_morphemes  # 下次就是"熟人"


# ==============================================================
# 与 SymbolicInterface 集成
# ==============================================================

class TestTokenizerThroughBrain:
    def test_brain_exposes_learning_api(self):
        from digital_brain.src.interfaces.symbolic_interface import SymbolicInterface
        brain = SymbolicInterface(auto_build=True)
        # 有学习统计
        stats = brain.tokenizer_stats()
        assert stats["learned_examples"] >= 4
        assert stats["merge_digit_sequences_learned"] is True
        # 可主动学习新样本
        before = stats["known_morphemes_count"]
        brain.learn_tokenizer_example("请问 99 乘以 88 等于多少？",
                                       ["请问", "99", "乘以", "88", "等于", "多少", "？"])
        after = brain.tokenizer_stats()["known_morphemes_count"]
        assert after > before  # 新增了"请问""乘以""？"等
        # 再对同一输入做 solve，词素拆分正确包含"请问"
        r = brain.solve("请问 99 乘以 88 等于多少？")
        assert "请问" in r.tokens or "乘以" in r.tokens


# ==============================================================
# 向后兼容：旧代码 Tokenizer() 仍然可用
# ==============================================================

class TestBackwardCompatibility:
    def test_tokenizer_class_name_alias(self):
        # 默认 Tokenizer 就是 LearnableTokenizer
        assert Tokenizer is LearnableTokenizer
        t = Tokenizer()
        assert isinstance(t, LearnableTokenizer)

    def test_merge_numbers_kwarg_still_works(self):
        # merge_numbers=True 立即得到数字合并能力
        t = Tokenizer(auto_learn=False, merge_numbers=True)
        assert t.merge_digit_sequences is True
        assert t.tokenize("42") == ["42"]
        # merge_numbers=False 不合并
        t = Tokenizer(auto_learn=False, merge_numbers=False)
        assert t.merge_digit_sequences is False
        assert t.tokenize("42") == ["4", "2"]


# ==============================================================
# 完整 E2E：没学过 token 的全新词 → 学习后 solve 正确
# ==============================================================

class TestEndToEndLearningNewMorphemes:
    def test_teach_then_solve_chinese_multiplication_phrase(self):
        """学习新的中文多字词素后，能正确拆分并解加法题。
        注意：遵循「没有初始能力」原则，加减之外的运算（乘/除）必须单独 teach 对应程序才会解。
        这里用加法保证在 teach_default_curriculum 范围内也能验证分词学习效果。
        """
        from digital_brain.src.interfaces.symbolic_interface import SymbolicInterface
        brain = SymbolicInterface(auto_build=True)
        question = "请问三加二等于几？"
        # 先教一遍"请问""等于""几"该怎么拆（两字词）
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
