"""Phase 3 DAG 推理测试

验证：
  - WorkingMemory 读写
  - DAG 数据结构与拓扑排序
  - DAGBuilder 构建（4种模板 + 三道闸门）
  - DAGExecutor 执行（原子操作集）
  - 端到端：空白脑→learn→DAG推理解应用题
  - "不理解"验证：未学知识→DAG构建失败→提示缺什么
  - 持久化验证：learn→退出→重启→DAG推理仍可用
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pytest

from digital_brain.src.core.workspace.dag import DAGGraph, DAGNode, DAGBuildResult
from digital_brain.src.core.workspace.working_memory import WorkingMemory
from digital_brain.src.core.workspace.reasoning_area import (
    DAGBuilder, DAGExecutor, AlgorithmRegistry,
)
from digital_brain.src.interfaces.symbolic_interface import SymbolicInterface


# ============================================================
# WorkingMemory 单元测试
# ============================================================

class TestWorkingMemory:
    def test_write_and_read_attr(self):
        wm = WorkingMemory()
        wm.write_attr("爸爸", "年龄", 31)
        assert wm.read_attr("爸爸", "年龄") == 31

    def test_read_attr_or_none(self):
        wm = WorkingMemory()
        wm.write_attr("爸爸", "年龄", 31)
        assert wm.read_attr_or_none("爸爸", "年龄") == 31
        assert wm.read_attr_or_none("爸爸", "高度") is None

    def test_read_attr_unbound_raises(self):
        wm = WorkingMemory()
        with pytest.raises(KeyError):
            wm.read_attr("爸爸", "年龄")

    def test_overwrite_attr(self):
        wm = WorkingMemory()
        wm.write_attr("爸爸", "年龄", 31)
        wm.write_attr("爸爸", "年龄", 35)
        assert wm.read_attr("爸爸", "年龄") == 35

    def test_resolve_pronoun_plural(self):
        wm = WorkingMemory()
        wm.write_attr("爸爸", "年龄", 31)
        wm.write_attr("妈妈", "年龄", 25)
        result = wm.resolve_pronoun("他们")
        assert "爸爸" in result
        assert "妈妈" in result

    def test_resolve_pronoun_singular(self):
        wm = WorkingMemory()
        wm.write_attr("爸爸", "年龄", 31)
        wm.write_attr("妈妈", "年龄", 25)
        result = wm.resolve_pronoun("他")
        assert result == ["妈妈"]  # 最后写入的

    def test_node_output(self):
        wm = WorkingMemory()
        wm.put_node_output("n1", 56)
        assert wm.get_node_output("n1") == 56
        assert wm.get_node_output_or_none("n1") == 56
        assert wm.get_node_output_or_none("n2") is None

    def test_clear(self):
        wm = WorkingMemory()
        wm.write_attr("爸爸", "年龄", 31)
        wm.put_node_output("n1", 56)
        wm.resolve_pronoun("他们")
        wm.clear()
        assert wm.read_attr_or_none("爸爸", "年龄") is None
        assert wm.get_node_output_or_none("n1") is None
        assert wm.get_resolved("他们") == []


# ============================================================
# DAG 数据结构测试
# ============================================================

class TestDAGGraph:
    def test_topological_sort_simple(self):
        dag = DAGGraph()
        dag.add_node(DAGNode(id="a", action="write_memory", params={}))
        dag.add_node(DAGNode(id="b", action="call_algorithm", params={}, depends_on=["a"]))
        dag.add_node(DAGNode(id="c", action="return_value", params={}, depends_on=["b"]))
        seq, ok = dag.topological_sort()
        assert ok
        assert seq[0] == "a"
        assert seq[1] == "b"
        assert seq[2] == "c"

    def test_topological_sort_parallel(self):
        dag = DAGGraph()
        dag.add_node(DAGNode(id="a", action="write_memory", params={}))
        dag.add_node(DAGNode(id="b", action="write_memory", params={}))
        dag.add_node(DAGNode(id="c", action="call_algorithm", params={}, depends_on=["a", "b"]))
        seq, ok = dag.topological_sort()
        assert ok
        assert seq[-1] == "c"
        assert "a" in seq[:2] and "b" in seq[:2]

    def test_cycle_detection(self):
        dag = DAGGraph()
        dag.add_node(DAGNode(id="a", action="x", depends_on=["c"]))
        dag.add_node(DAGNode(id="b", action="x", depends_on=["a"]))
        dag.add_node(DAGNode(id="c", action="x", depends_on=["b"]))
        seq, ok = dag.topological_sort()
        assert not ok
        assert dag.has_cycle()


# ============================================================
# DAG Builder 测试
# ============================================================

class TestDAGBuilder:
    @pytest.fixture
    def learned_brain(self):
        b = SymbolicInterface(auto_build=False, auto_learn_tokenizer=False)
        b.learn_from_package("base_curriculum")
        return b

    def test_build_binary_op(self, learned_brain):
        from digital_brain.src.core.pattern.pattern_matcher import PatternMatcher
        pm = learned_brain.pattern_matcher
        tokens = learned_brain.tokenizer.tokenize("1+1=?")
        matches = pm.match(tokens)
        builder = DAGBuilder(learned_brain.declarative, learned_brain.procedural)
        result = builder.build(matches)
        assert result.success
        assert result.dag is not None
        assert result.dag.node_count == 2  # call_algorithm + return_value

    def test_build_app_problem(self, learned_brain):
        from digital_brain.src.core.pattern.pattern_matcher import PatternMatcher
        pm = learned_brain.pattern_matcher
        tokens = learned_brain.tokenizer.tokenize("爸爸的年龄31，妈妈的年龄25，他们的年龄之和是多少？")
        matches = pm.match(tokens)
        builder = DAGBuilder(learned_brain.declarative, learned_brain.procedural)
        result = builder.build(matches)
        assert result.success
        assert result.dag is not None
        # 2 write + 1 search + 1 call + 1 return = 5
        assert result.dag.node_count == 5

    def test_build_fail_unlearned_number(self, learned_brain):
        """删掉一个数字实体后，DAG构建应失败并提示缺什么"""
        from digital_brain.src.core.pattern.pattern_matcher import PatternMatcher
        # 删掉数字5
        for e in learned_brain.declarative.find_entity_by_name("5"):
            learned_brain.declarative.delete_entity(e.id)
        pm = learned_brain.pattern_matcher
        tokens = learned_brain.tokenizer.tokenize("5+3=?")
        matches = pm.match(tokens)
        builder = DAGBuilder(learned_brain.declarative, learned_brain.procedural)
        result = builder.build(matches)
        assert not result.success
        assert result.failure_type == "missing_dependency"
        assert any("5" in m for m in result.missing)


# ============================================================
# 端到端 DAG 推理测试
# ============================================================

class TestDAGEndToEnd:
    @pytest.fixture(scope="class")
    def brain(self):
        b = SymbolicInterface(auto_build=False, auto_learn_tokenizer=False)
        b.learn_from_package("base_curriculum")
        return b

    def test_blank_brain_cannot_answer(self):
        b = SymbolicInterface(auto_build=False, auto_learn_tokenizer=False)
        r = b.solve("1+1=?")
        assert r.answer is None

    def test_binary_op_addition(self, brain):
        r = brain.solve("1+1=?")
        assert r.answer == 2
        assert r.confidence >= 0.9

    def test_binary_op_subtraction(self, brain):
        r = brain.solve("10-3=?")
        assert r.answer == 7

    def test_chinese_addition(self, brain):
        r = brain.solve("一加一等于多少")
        assert r.answer == 2

    def test_chinese_subtraction(self, brain):
        r = brain.solve("十减三等于几")
        assert r.answer == 7

    def test_app_problem_sum(self, brain):
        r = brain.solve("爸爸的年龄31，妈妈的年龄25，他们的年龄之和是多少？")
        assert r.answer == 56
        # 验证推理链包含 DAG 步骤
        actions = [s["action"] for s in r.reasoning_chain]
        assert "dag_build" in actions
        assert "write_memory" in actions
        assert "search_context" in actions
        assert "call_algorithm" in actions
        assert "return_value" in actions

    def test_app_problem_diff(self, brain):
        r = brain.solve("爸爸的年龄31，妈妈的年龄25，他们的年龄之差是多少？")
        assert r.answer == 6

    # ============================================================
    # M1 表层鲁棒性回归（含同义词/省略/总量提问）
    # ============================================================
    def test_m1_regression_simplified_bag(self, brain):
        """M1 回归：'包'代替'书包' + 宾语省略 + 总/共拆分，答案应为 7"""
        r = brain.solve("小明包里有4本故事书,妈妈又给他了3本,现在小明总共有几本?")
        assert r.answer == 7

    def test_m1_regression_full_version(self, brain):
        """M1 回归：完整版（书包/买了/一共），答案应为 7"""
        r = brain.solve("小明书包里有4本故事书,妈妈又给他买了3本,现在小明一共有几本?")
        assert r.answer == 7

    def test_m1_regression_pencil(self, brain):
        """M1 回归：哥哥铅笔题（显式接受者+对象），答案应为 7"""
        r = brain.solve("哥哥有5支铅笔,妈妈又给哥哥买了2支铅笔,现在哥哥一共有几支铅笔?")
        assert r.answer == 7

    def test_no_duplicate_possessive_match(self, brain):
        """M1-R4 回归：'小明包里有4本' 只应建立 小明.故事书=4，不应再写 包.故事书=4"""
        from digital_brain.src.core.pattern.pattern_matcher import PatternMatcher
        pm = brain.pattern_matcher
        tokens = brain.tokenizer.tokenize("小明包里有4本故事书,妈妈又给他了3本,现在小明总共有几本?")
        matches = pm.match(tokens)
        poss = [m for m in matches if m.pattern_name == "possessive_state"]
        assert len(poss) == 1, f"possessive_state 应只命中1条，实际 {len(poss)}"
        assert poss[0].captured.get("holder") == "小明"

    # ============================================================
    # M2 语义栈验证集
    # ============================================================
    def test_m2_loss_event_borrow(self, brain):
        """M2-a：借走（减法事件），宾语故事书贯穿 3 句 → 6"""
        r = brain.solve("小明有4本故事书，妈妈又给他了3本，弟弟又借走了1本，小明现在总共有几本？")
        assert r.answer == 6

    def test_m2_theme_not_mixed(self, brain):
        """M2-b：两物交替（橡皮≠铅笔），弟弟.铅笔 = 2（不可把橡皮混入铅笔）"""
        r = brain.solve("哥哥有5支铅笔，弟弟有3支橡皮，妈妈又给弟弟2支铅笔，弟弟现在有几支铅笔？")
        assert r.answer == 2

    def test_m2_gender_resolution(self, brain):
        """M2-c：她→小红（性别过滤），货币量词 元→钱 → 150"""
        r = brain.solve("小红钱包里有100元，她又收到了50元红包，小红现在一共有多少钱？")
        assert r.answer == 150

    def test_m2_concrete_name_not_resolved_by_recency(self, brain):
        """M2 回归：search_context('哥哥') 必须返回哥哥本身，不能按最近提及解析成弟弟"""
        r = brain.solve("哥哥有5支铅笔，弟弟有2支铅笔，妈妈又给哥哥3支铅笔，哥哥现在有几支铅笔？")
        assert r.answer == 8

    def test_not_understood(self, brain):
        r = brain.solve("光速是多少？")
        assert r.answer is None

    def test_idempotent(self, brain):
        a = brain.solve("3+5=?").answer
        b = brain.solve("3+5=?").answer
        assert a == b == 8


# ============================================================
# 持久化 + DAG 推理验证
# ============================================================

class TestDAGPersistence:
    def test_learn_consolidate_restore_solve(self, tmp_path):
        b1 = SymbolicInterface(auto_build=False, auto_learn_tokenizer=False)
        b1.learn_from_package("base_curriculum")
        storage = str(tmp_path)
        b1.consolidate(storage)

        b2 = SymbolicInterface(storage_dir=storage, auto_restore=True)
        assert b2.declarative.entity_count == b1.declarative.entity_count
        assert b2.procedural.procedure_count == b1.procedural.procedure_count

        r = b2.solve("1+1=?")
        assert r.answer == 2

        r = b2.solve("爸爸的年龄31，妈妈的年龄25，他们的年龄之和是多少？")
        assert r.answer == 56
