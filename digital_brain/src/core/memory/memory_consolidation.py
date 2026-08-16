"""记忆巩固机制 - Phase 1 占位，Phase 2+ 扩展"""
from __future__ import annotations

from digital_brain.src.core.memory.declarative_memory import DeclarativeMemory
from digital_brain.src.core.memory.procedural_memory import ProceduralMemory


class MemoryConsolidation:
    """记忆巩固（简化实现）

    MVP 阶段：将高频激活的知识保留，低频的淘汰。
    Phase 1 主要提供占位接口。
    """

    def __init__(
        self,
        declarative_memory: DeclarativeMemory,
        procedural_memory: ProceduralMemory,
        threshold: float = 0.2,
    ) -> None:
        self.declarative = declarative_memory
        self.procedural = procedural_memory
        self.threshold = threshold
        self._activation_counts = {}   # id -> count

    def record_activation(self, entity_or_proc_id: str) -> None:
        self._activation_counts[entity_or_proc_id] = self._activation_counts.get(entity_or_proc_id, 0) + 1

    def consolidate(self) -> dict:
        """执行巩固，返回统计信息"""
        # Phase 1: 仅统计，不真正删除
        return {
            "total_activation_records": len(self._activation_counts),
            "declarative_count": self.declarative.entity_count,
            "procedural_count": self.procedural.procedure_count,
            "threshold": self.threshold,
        }
