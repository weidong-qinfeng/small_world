"""激活记忆区 - 保存被激活进入工作区的知识"""
from __future__ import annotations

from typing import Dict, List, Optional

from digital_brain.src.core.models import ActivatedKnowledge, Entity, Procedure


class ActivationArea:
    """工作区的激活记忆区

    存储激活的实体和程序性记忆，支持：
    - 按激活强度排序
    - 只保留前 N 个（容量管理）
    - 按类型检索（实体 / 程序）
    """

    def __init__(self, capacity: int = 20) -> None:
        self.capacity = capacity
        self._items: List[ActivatedKnowledge] = []

    def load(self, activations: List[ActivatedKnowledge]) -> None:
        """加载激活列表（会截断到容量），并按强度排序"""
        sorted_items = sorted(activations, key=lambda x: -x.activation_strength)
        self._items = sorted_items[: self.capacity]

    def add(self, item: ActivatedKnowledge) -> None:
        self._items.append(item)
        self._items.sort(key=lambda x: -x.activation_strength)
        if len(self._items) > self.capacity:
            self._items = self._items[: self.capacity]

    def all(self) -> List[ActivatedKnowledge]:
        return list(self._items)

    def entities(self) -> List[Entity]:
        result = []
        for a in self._items:
            if a.entity is not None:
                result.append(a.entity)
        return result

    def procedures(self) -> List[Procedure]:
        result = []
        for a in self._items:
            if a.procedure is not None:
                result.append(a.procedure)
        return result

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        for a in self._items:
            if a.entity and a.entity.id == entity_id:
                return a.entity
        return None

    def get_procedure(self, procedure_id: str) -> Optional[Procedure]:
        for a in self._items:
            if a.procedure and a.procedure.id == procedure_id:
                return a.procedure
        return None

    def top_entity_by_name(self, name: str) -> Optional[Entity]:
        for a in self._items:
            if a.entity and (a.entity.name == name or name in a.entity.aliases):
                return a.entity
        return None

    def clear(self) -> None:
        self._items = []

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        parts = [f"{i.activation_strength:.2f}:" for i in self._items]
        return f"ActivationArea([{', '.join(parts)}])"
