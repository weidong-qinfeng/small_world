"""Phase 1 集成测试 - 验证 MVP 完整流程

核心验证项（对应开发计划 §4.2 任务）：
  * 测试 "1+1=？" 完整流程：
    - 输入接收 ✓
    - 词素拆分 ✓
    - 记忆激活 ✓
    - 模式匹配 ✓
    - 意图识别 ✓
    - 算法执行 ✓
    - 结果输出 ✓
  * 验证输出正确性
  * 验证推理链完整性
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pytest

from digital_brain.src.core.workspace.output_buffer import OutputBuffer
from digital_brain.src.core.workspace.reasoning_area import ReasoningArea
from digital_brain.src.interfaces.symbolic_interface import SymbolicInterface, BrainResult


@pytest.fixture(scope="module")
def brain():
    """模块级共享一个大脑实例 —— 通过标准课程注入最小知识（兼容旧测试）。
    新的"空白脑 + 教学接口"模式请另行构造 auto_build=False 实例并手动 teach。
    """
    return SymbolicInterface(use_embodied=True, auto_build=True)


# ============== MVP 核心验证 ==============

class TestMVPOnePlusOne:
    """MVP 验收：1+1=？完整流程"""

    def test_solve_returns_result(self, brain):
        result = brain.solve("1+1=?")
        assert isinstance(result, BrainResult)

    def test_input_tokens(self, brain):
        result = brain.solve("1+1=?")
        # 词素拆分正确
        assert result.tokens == ["1", "+", "1", "=", "?"]
        assert result.raw_text == "1+1=?"

    def test_activation_happened(self, brain):
        result = brain.solve("1+1=?")
        # 至少激活了 2 个实体 + 1 个程序
        assert result.activated_count >= 3

    def test_answer_correct(self, brain):
        result = brain.solve("1+1=?")
        # MVP 核心：答案必须是 2
        assert result.answer == 2

    def test_confidence_high(self, brain):
        result = brain.solve("1+1=?")
        assert result.confidence >= 0.9

    def test_reasoning_chain_complete(self, brain):
        result = brain.solve("1+1=?")
        chain = result.reasoning_chain
        actions = [s["action"] for s in chain]
        # 推理链中必须包含以下步骤
        assert "pattern_match" in actions
        assert "intent_recognize" in actions
        assert "execute_algorithm" in actions
        # 必须有具身子步骤：mapping / merging / counting sub_steps
        sub_descs = " ".join(s.get("description", "") for s in chain)
        assert "映射" in sub_descs or "mapping" in sub_descs or "[mapping]" in sub_descs
        assert "合并" in sub_descs or "[merge]" in sub_descs
        assert "计数" in sub_descs or "[count]" in sub_descs
        # 最后一步答案写入
        execute_step = next(s for s in chain if s["action"] == "execute_algorithm")
        assert execute_step["outputs"]["result"] == 2

    def test_format_output_readable(self, brain):
        result = brain.solve("1+1=?")
        text = result.format()
        assert "1+1=?" in text
        assert "最终答案: 2" in text


# ============== 10 以内加减法完整验证 ==============

class TestMathWithinTen:
    """支持 10 以内加减法精确计算"""

    @pytest.mark.parametrize("question,expected", [
        ("0+0=?", 0), ("0+1=?", 1), ("1+2=?", 3),
        ("3+4=?", 7), ("5+5=?", 10), ("9+1=?", 10),
        ("2+3", 5),
    ])
    def test_addition_within_10(self, brain, question, expected):
        r = brain.solve(question)
        assert r.answer == expected, f"Failed {question}: got {r.answer}"
        assert r.confidence >= 0.8

    @pytest.mark.parametrize("question,expected", [
        ("5-2=?", 3), ("10-5=?", 5), ("3-0=?", 3),
        ("9-4=?", 5), ("1-1=?", 0), ("7-3=?", 4),
    ])
    def test_subtraction_within_10(self, brain, question, expected):
        r = brain.solve(question)
        assert r.answer == expected, f"Failed {question}: got {r.answer}"
        assert r.confidence >= 0.8


class TestChineseInput:
    """中文问题支持"""

    def test_chinese_add(self, brain):
        r = brain.solve("一加一等于多少")
        assert r.answer == 2
        assert r.confidence >= 0.8

    def test_chinese_sub(self, brain):
        r = brain.solve("五减去二等于几")
        assert r.answer == 3


class TestWorkspaceEndToEnd:
    """通过 workspace 驱动的底层 E2E 测试"""

    def test_reasoning_area_runs(self):
        brain = SymbolicInterface(use_embodied=True, auto_build=True)
        ws = brain.workspace
        ws.receive_input("2+3=?", tokens=brain.tokenizer.tokenize("2+3=?"))
        ws.activate()
        out = brain.reasoning.run(ws)
        assert isinstance(out, OutputBuffer)
        assert out.is_ready()
        assert out.final_answer == 5

    def test_solve_is_idempotent(self, brain):
        """多次求解同一个问题结果一致"""
        q = "3+2=?"
        a = brain.solve(q).answer
        b = brain.solve(q).answer
        assert a == b == 5


# ============== 验收清单汇总 ==============

def test_phase1_acceptance_entity_crud(brain):
    """能存储和检索陈述性记忆"""
    from digital_brain.src.core.models import Entity, EntityType
    dm = brain.declarative
    before = dm.entity_count
    eid = dm.add_entity(Entity(id="tmp_test", name="测试", entity_type=EntityType.ABSTRACT))
    assert dm.get_entity(eid).name == "测试"
    dm.delete_entity(eid)
    assert dm.entity_count == before


def test_phase1_acceptance_procedural_crud(brain):
    """能存储和检索程序性记忆"""
    from digital_brain.src.core.models import Procedure
    pm = brain.procedural
    assert pm.procedure_count >= 5
    assert len(pm.find_by_name("adding")) == 1
    assert len(pm.find_by_tokens(["+"])) >= 1


def test_phase1_acceptance_workspace_lifecycle(brain):
    """工作区能正常管理生命周期"""
    ws = brain.workspace
    ws.clear()
    assert ws.phase == "idle"
    ws.receive_input("hello", tokens=["h"])
    assert ws.phase == "input"
    ws.activate()
    assert ws.phase == "activation"
    ws.mark_reasoning()
    assert ws.phase == "reasoning"
    ws.mark_done()
    ws.clear()
    assert ws.phase == "idle"


def test_phase1_acceptance_pattern_and_intent(brain):
    """模式匹配和意图识别功能正常"""
    pm = brain.reasoning.pattern_matcher
    ir = brain.reasoning.intent_recognizer
    toks = pm.classify_token
    assert toks("1") == "N"
    assert toks("+") == "OP"
    assert toks("=") == "="
    assert toks("?") == "?"
    tokens = ["9", "+", "8", "=", "?"]
    best = pm.best_match(tokens)
    assert best is not None
    intent = ir.recognize([best], tokens)
    assert intent.intent_type.value == "compute_binary"
    assert intent.slots["operand_left"] == 9


def test_phase1_acceptance_algorithms_run(brain):
    """基础算法（计数、映射、合并、加法）可正常执行"""
    algs = brain.algorithm_registry
    assert algs.counting.execute(["a", "b"])["count"] == 2
    assert algs.mapping.number_to_set(4)["size"] == 4
    assert algs.merging.execute([1, 2], [3])["size"] == 3
    assert algs.adding.execute(2, 2)["result"] == 4
    assert algs.subtracting.execute(8, 5)["result"] == 3
