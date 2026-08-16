"""工具模块测试：Tokenizer 词素拆分 + KnowledgeBuilder 初始知识构建

v2 调整：
    - Tokenizer 已改为无状态引擎，未绑定图谱时退化为单字拆分。
    - 需要测试多字词/数字合并的场景，通过绑定 DeclarativeMemory 实现。
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pytest

from digital_brain.src.core.memory.declarative_memory import DeclarativeMemory
from digital_brain.src.core.memory.procedural_memory import ProceduralMemory
from digital_brain.src.utils.knowledge_builder import KnowledgeBuilder
from digital_brain.src.utils.tokenizer import Tokenizer


class TestTokenizer:
    def test_basic_fallback_single_char(self):
        """未绑定图谱时按单字拆分"""
        t = Tokenizer()
        assert t.tokenize("1+1=?") == ["1", "+", "1", "=", "?"]

    def test_whitespace_ignored(self):
        t = Tokenizer()
        assert t.tokenize("  3 + 4  =  ?  ") == ["3", "+", "4", "=", "?"]

    def test_merge_numbers_with_memory(self):
        """绑定图谱并写入 digit_merge 规则后能合并数字"""
        dm = DeclarativeMemory()
        t = Tokenizer(declarative_memory=dm, merge_numbers=True)
        # merge_numbers=True 触发写入 digit_merge 规则到图谱
        assert t.merge_digit_sequences is True
        assert t.tokenize("12+34") == ["12", "+", "34"]

    def test_chinese_ops_with_memory(self):
        """绑定图谱并通过样本学习后能分多字词"""
        dm = DeclarativeMemory()
        t = Tokenizer(declarative_memory=dm)
        t.learn_from_example("一加一等于多少", ["一", "加", "一", "等于", "多少"])
        toks = t.tokenize("一加一等于多少")
        assert "一" in toks and "加" in toks
        assert "等于" in toks
        assert "多少" in toks

    def test_chinese_question_mark(self):
        t = Tokenizer()
        assert "？" in t.tokenize("1+1=？")


class TestKnowledgeBuilder:
    def test_build_declarative(self):
        mem = DeclarativeMemory()
        kb = KnowledgeBuilder(max_number=10)
        ents = kb.build_declarative(mem)
        # 0-10 共 11 个数字 + 6 个运算符 = 17
        assert mem.entity_count == 17
        # 后继关系：0->1, 1->2, ..., 9->10 共 10 条
        assert mem.relation_count == 10
        # 实体类型
        abstracts = mem.find_entities_by_type("abstract")
        assert len(abstracts) == 17  # 全部是 abstract
        # 中文别名可用
        two = mem.find_entity_by_name("二")
        assert len(two) >= 1
        two_alias = mem.find_entity_by_name("两")
        assert len(two_alias) >= 1

    def test_build_procedural(self):
        pm = ProceduralMemory()
        kb = KnowledgeBuilder()
        procs = kb.build_procedural(pm)
        assert pm.procedure_count == 5
        names = {p.name for p in procs}
        assert names == {"counting", "mapping", "merging", "adding", "subtracting"}
        adding = pm.find_by_name("adding")[0]
        assert adding.steps  # 有步骤
        # 触发条件可被匹配
        assert "+" in adding.trigger.pattern_tokens or "加" in adding.trigger.pattern_tokens

    def test_build_all(self):
        dm = DeclarativeMemory()
        pm = ProceduralMemory()
        kb = KnowledgeBuilder(max_number=5)
        stats = kb.build_all(dm, pm)
        assert stats["number_entities"] == 6      # 0-5
        assert stats["procedures"] == 5
        assert stats["declarative_total"] == 12   # 6 num + 6 op
        assert stats["relations_total"] == 5      # 0->1..4->5
