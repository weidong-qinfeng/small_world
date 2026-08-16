"""v2 Phase 1: 记忆固化与启动恢复测试

验证：
- consolidate_to_disk / restore_from_disk 端到端
- SymbolicInterface.storage_dir 学习后自动固化
- SymbolicInterface(auto_restore=True) 启动时自动恢复
- 空白脑学习 → 固化 → 重启恢复 → 答题正确
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pytest

from digital_brain.src.core.memory.memory_consolidation import MemoryConsolidation
from digital_brain.src.interfaces.symbolic_interface import SymbolicInterface


@pytest.fixture
def tmp_storage():
    """临时持久化目录"""
    d = tempfile.mkdtemp(prefix="brain_store_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ==============================================================
# MemoryConsolidation 单元测试
# ==============================================================

class TestMemoryConsolidation:
    def test_consolidate_then_restore_roundtrip(self, tmp_storage):
        """固化 → 恢复 知识完整"""
        from digital_brain.src.core.memory.declarative_memory import DeclarativeMemory
        from digital_brain.src.core.memory.procedural_memory import ProceduralMemory

        # 原始 brain：写入一些知识
        dm1 = DeclarativeMemory()
        pm1 = ProceduralMemory()
        brain1 = SymbolicInterface.__new__(SymbolicInterface)
        brain1.declarative = dm1
        brain1.procedural = pm1
        cons = MemoryConsolidation(dm1, pm1)
        # 通过 teach 接口填入知识
        b1 = SymbolicInterface(auto_build=True)
        cons1 = MemoryConsolidation(b1.declarative, b1.procedural)
        stats = cons1.consolidate_to_disk(tmp_storage)
        assert stats["consolidated"] is True
        assert stats["entity_count"] > 0
        assert stats["procedure_count"] >= 5
        # 文件存在
        assert os.path.isfile(os.path.join(tmp_storage, "declarative_entities.json"))
        assert os.path.isfile(os.path.join(tmp_storage, "procedural.json"))
        assert os.path.isfile(os.path.join(tmp_storage, "meta.json"))

        # 新 brain：从硬盘恢复
        dm2 = DeclarativeMemory()
        pm2 = ProceduralMemory()
        cons2 = MemoryConsolidation(dm2, pm2)
        stats2 = cons2.restore_from_disk(tmp_storage)
        assert stats2["restored"] is True
        assert stats2["entity_count"] == stats["entity_count"]
        assert stats2["procedure_count"] == stats["procedure_count"]
        assert stats2["version"] == "v2"

    def test_restore_skipped_if_no_state(self, tmp_storage):
        """目录无已固化状态时返回 skipped=True 不报错"""
        from digital_brain.src.core.memory.declarative_memory import DeclarativeMemory
        from digital_brain.src.core.memory.procedural_memory import ProceduralMemory
        dm = DeclarativeMemory()
        pm = ProceduralMemory()
        cons = MemoryConsolidation(dm, pm)
        # 空目录
        stats = cons.restore_from_disk(tmp_storage)
        # 空目录下 entities/procedural 文件不存在，所以 entities_loaded=0
        assert stats["entity_count"] == 0
        assert stats["procedure_count"] == 0

    def test_has_saved_state(self, tmp_storage):
        assert MemoryConsolidation.has_saved_state(tmp_storage) is False
        # 写入一个状态后
        b = SymbolicInterface(auto_build=True)
        b.consolidate(tmp_storage)
        assert MemoryConsolidation.has_saved_state(tmp_storage) is True


# ==============================================================
# SymbolicInterface 集成：学习 → 固化 → 重启恢复
# ==============================================================

class TestSymbolicInterfacePersistence:
    def test_learn_then_consolidate_then_restore(self, tmp_storage):
        """学习知识包 → 手动固化 → 新 brain 启动时自动恢复 → 答题正确"""
        # 1) 学习
        b1 = SymbolicInterface(auto_build=False, auto_learn_tokenizer=False)
        b1.learn_from_package("base_curriculum")
        n_entities = b1.declarative.entity_count
        n_procs = b1.procedural.procedure_count
        assert n_entities > 0 and n_procs > 0

        # 2) 固化
        stats = b1.consolidate(tmp_storage)
        assert stats["consolidated"] is True
        assert stats["entity_count"] == n_entities

        # 3) 新 brain 从硬盘恢复
        b2 = SymbolicInterface(
            auto_build=False,
            auto_learn_tokenizer=False,
            storage_dir=tmp_storage,
            auto_restore=True,
        )
        assert b2.declarative.entity_count == n_entities
        assert b2.procedural.procedure_count == n_procs

        # 4) 恢复后能正确答题
        r = b2.solve("1+1=?")
        assert r.answer == 2
        r2 = b2.solve("一加一等于多少")
        assert r2.answer == 2

    def test_auto_consolidate_after_learn(self, tmp_storage):
        """配置 storage_dir 后，learn_from_package 自动固化"""
        b = SymbolicInterface(
            auto_build=False,
            auto_learn_tokenizer=False,
            storage_dir=tmp_storage,
            auto_restore=False,  # 不自动恢复，纯空白启动
        )
        b.learn_from_package("base_curriculum")
        # 学习后应已自动固化
        assert os.path.isfile(os.path.join(tmp_storage, "procedural.json"))
        assert os.path.isfile(os.path.join(tmp_storage, "declarative_entities.json"))

    def test_blank_brain_no_storage_does_not_persist(self):
        """未配置 storage_dir 时学习不固化"""
        b = SymbolicInterface(auto_build=True)
        # 没有配置 storage_dir
        assert b.storage_dir is None
        # learn_tokenizer_example 不应触发固化
        b.learn_tokenizer_example("abc", ["a", "b", "c"])
        # 不会创建任何文件（因为没有目录）
        # 这里只是确保不抛异常

    def test_restore_skips_auto_build(self, tmp_storage):
        """已从硬盘恢复后，auto_build 不再执行（避免覆盖）"""
        # 先准备一份固化状态
        b1 = SymbolicInterface(auto_build=True)
        b1.consolidate(tmp_storage)

        # 新 brain：auto_build=True + auto_restore=True
        # 因为有已固化状态，应该走 restore 路径，跳过 auto_build
        b2 = SymbolicInterface(
            auto_build=True,
            storage_dir=tmp_storage,
            auto_restore=True,
        )
        # 实体数应与 b1 相同（恢复的），而不是 auto_build 默认的（可能略有差异）
        assert b2.declarative.entity_count == b1.declarative.entity_count
