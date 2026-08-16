"""工作区、记忆激活、模式匹配、意图识别单元测试"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pytest

from digital_brain.src.core.memory.declarative_memory import DeclarativeMemory
from digital_brain.src.core.memory.memory_activation import MemoryActivation
from digital_brain.src.core.memory.procedural_memory import ProceduralMemory
from digital_brain.src.core.models import (
    Entity, EntityType, Procedure, Relation, RelationType, TriggerCondition,
)
from digital_brain.src.core.pattern.intent_recognizer import IntentRecognizer, IntentType
from digital_brain.src.core.pattern.pattern_matcher import PatternMatcher
from digital_brain.src.core.workspace.workspace import Workspace
from digital_brain.src.utils.knowledge_builder import KnowledgeBuilder


@pytest.fixture
def seeded_memory():
    dm = DeclarativeMemory()
    pm = ProceduralMemory()
    KnowledgeBuilder(max_number=5).build_all(dm, pm)
    return dm, pm


def test_activation_by_tokens(seeded_memory):
    dm, pm = seeded_memory
    ma = MemoryActivation(dm, pm)
    acts = ma.activate(["1", "+", "1"])
    assert len(acts) > 0
    # 必须激活了数字实体
    entities = [a.entity for a in acts if a.entity]
    assert any(e.name == "1" for e in entities)
    # 必须激活了 adding 程序
    procs = [a.procedure for a in acts if a.procedure]
    assert any(p.name == "adding" for p in procs)
    # resolve_conflicts 截断
    top = MemoryActivation.resolve_conflicts(acts, top_k=3)
    assert len(top) <= 3
    # 按强度降序
    strengths = [a.activation_strength for a in top]
    assert strengths == sorted(strengths, reverse=True)


def test_pattern_matcher_binary_op():
    pm = PatternMatcher()
    tokens = ["1", "+", "1", "=", "?"]
    best = pm.best_match(tokens)
    assert best is not None
    assert best.pattern_name == "binary_op_question"
    assert best.captured["left"] == "1"
    assert best.captured["op"] == "+"
    assert best.captured["right"] == "1"
    assert best.match_score >= 0.8


def test_pattern_matcher_no_eq():
    pm = PatternMatcher()
    tokens = ["3", "+", "2"]
    best = pm.best_match(tokens)
    assert best is not None
    assert best.pattern_name in ("binary_op", "binary_op_question")
    assert best.captured["op"] == "+"


def test_intent_recognize_add():
    ir = IntentRecognizer()
    pm = PatternMatcher()
    tokens = ["5", "+", "3", "=", "?"]
    pr = pm.match(tokens)
    intent = ir.recognize(pr, tokens)
    assert intent.intent_type == IntentType.COMPUTE_BINARY
    assert intent.slots["operation"] == "add"
    assert intent.slots["operand_left"] == 5
    assert intent.slots["operand_right"] == 3


def test_intent_recognize_sub_chinese():
    ir = IntentRecognizer()
    pm = PatternMatcher()
    tokens = ["五", "减", "二"]
    pr = pm.match(tokens)
    intent = ir.recognize(pr, tokens)
    # 兜底 fallback 能识别
    assert intent.intent_type == IntentType.COMPUTE_BINARY
    assert intent.slots["operation"] == "sub"


def test_workspace_lifecycle(seeded_memory):
    dm, pm = seeded_memory
    ws = Workspace(dm, pm)
    assert ws.phase == "idle"
    ws.receive_input("1+1=?", tokens=["1", "+", "1", "=", "?"])
    assert ws.phase == "input"
    assert ws.input_buffer.tokens == ["1", "+", "1", "=", "?"]
    acts = ws.activate()
    assert ws.phase == "activation"
    assert len(ws.activation_area) > 0
    ws.mark_reasoning()
    assert ws.phase == "reasoning"
    ws.output_buffer.set_answer(2, confidence=1.0)
    ws.mark_done()
    assert ws.output_buffer.final_answer == 2
    ws.clear()
    assert ws.phase == "idle"
    assert len(ws.activation_area) == 0


def test_procedural_memory_crud(seeded_memory):
    _, pm = seeded_memory
    assert pm.procedure_count == 5  # counting, mapping, merging, adding, subtracting
    adding = pm.find_by_name("adding")
    assert len(adding) == 1
    assert len(adding[0].steps) == 4  # map->map->merge->count
    # token 匹配
    matches = pm.find_by_tokens(["+", "加"])
    assert any(p.name == "adding" for p in matches)
    # children/dependencies
    assert pm.get_dependencies(adding[0].id)  # adding has dependencies
