"""基础算法单元测试"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pytest

from digital_brain.src.algorithms.adding import AddingAlgorithm
from digital_brain.src.algorithms.counting import CountingAlgorithm
from digital_brain.src.algorithms.mapping import MappingAlgorithm
from digital_brain.src.algorithms.merging import MergingAlgorithm
from digital_brain.src.algorithms.subtracting import SubtractingAlgorithm


def test_counting_int():
    c = CountingAlgorithm()
    r = c.execute(5)
    assert r["count"] == 5
    assert len(r["steps"]) >= 1


def test_counting_list():
    c = CountingAlgorithm()
    r = c.execute(["a", "b", "c", "d"])
    assert r["count"] == 4
    assert any("第 4 个" in s for s in r["steps"])


def test_counting_empty():
    c = CountingAlgorithm()
    assert c.execute([])["count"] == 0


def test_mapping_round_trip():
    m = MappingAlgorithm()
    r = m.number_to_set(3)
    assert r["size"] == 3 and len(r["set"]) == 3
    r2 = m.set_to_number(r["set"])
    assert r2["number"] == 3


def test_mapping_zero():
    m = MappingAlgorithm()
    assert m.number_to_set(0)["size"] == 0


def test_merging_basic():
    mg = MergingAlgorithm()
    r = mg.execute(["a", "b"], ["c", "d", "e"])
    assert r["size"] == 5
    assert r["merged"] == ["a", "b", "c", "d", "e"]


def test_adding_embodied():
    a = AddingAlgorithm()
    r = a.execute(1, 1, use_embodied=True)
    assert r["result"] == 2
    assert r["method"] == "embodied"
    assert r["intermediate"]["set_a"] == ["•"]
    assert r["intermediate"]["merged"] == ["•", "•"]
    # 推理链含关键字（中文描述或英文 tag 任一）
    trace = r["trace"]
    assert ("映射" in trace or "[mapping]" in trace)
    assert ("合并" in trace or "[merge]" in trace)
    assert ("数第" in trace or "[count]" in trace)
    assert "1 + 1 = 2" in trace


def test_adding_larger():
    a = AddingAlgorithm()
    assert a.execute(3, 4, use_embodied=True)["result"] == 7
    assert a.execute(0, 5)["result"] == 5
    assert a.execute(10, 0)["result"] == 10
    assert a.execute(9, 1)["result"] == 10


def test_adding_direct():
    a = AddingAlgorithm()
    r = a.execute(2, 3, use_embodied=False)
    assert r["result"] == 5
    assert r["method"] == "direct"


def test_subtracting_embodied():
    s = SubtractingAlgorithm()
    r = s.execute(5, 2, use_embodied=True)
    assert r["result"] == 3
    assert r["method"] == "embodied"
    assert len(r["intermediate"]["remaining"]) == 3


def test_subtracting_bounds():
    s = SubtractingAlgorithm()
    assert s.execute(0, 0)["result"] == 0
    # MVP: 负数返回 0
    r = s.execute(1, 3)
    assert r["result"] == 0
    assert r["method"] == "warn"
