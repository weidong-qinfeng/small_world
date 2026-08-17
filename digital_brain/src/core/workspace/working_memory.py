"""工作记忆 - 任务级临时状态，DAG节点间数据传递的唯一通道

对应详细设计/工作区设计.md §3：
    - context_entities: 上下文实体引用
    - attribute_bindings: 临时属性绑定（entity.attr = value）
    - resolved_references: 代词消解结果（pronoun -> [entities]）
    - intermediate_results: DAG节点中间结果（node_id.out -> value）

生命周期：任务开始创建，任务结束清空，不污染长期记忆。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from digital_brain.src.core.models import Entity
from digital_brain.src.core.workspace.ready_queue import ReadyQueue


class WorkingMemory:
    """工作记忆 - DAG节点间数据传递的唯一通道"""

    def __init__(self) -> None:
        self.context_entities: Dict[str, Entity] = {}
        self.attribute_bindings: Dict[str, Any] = {}            # key: "entity.attr"
        self.resolved_references: Dict[str, List[str]] = {}     # key: pronoun -> entity names
        self.intermediate_results: Dict[str, Any] = {}          # key: "node_id.out"
        self._entity_write_order: List[str] = []                # 实体写入顺序（代词消解用）
        self.ready_queue = ReadyQueue()        # Phase 5c: 就绪操作队列（5d 启用）
        self.recent_entities: List[str] = []  # Phase 5d.5: 跨句实体追踪，最近提及的实体名（按出现序）

    # ---- context_entities ----
    def put_context(self, name: str, entity: Entity) -> None:
        if name not in self.context_entities:
            self._entity_write_order.append(name)
        self.context_entities[name] = entity

    def get_context(self, name: str) -> Optional[Entity]:
        return self.context_entities.get(name)

    def has_context(self, name: str) -> bool:
        return name in self.context_entities

    def all_context_entity_names(self) -> List[str]:
        """按写入顺序返回所有上下文实体名"""
        return list(self._entity_write_order)

    # ---- attribute_bindings ----
    @staticmethod
    def _attr_key(entity: str, attr: str) -> str:
        return f"{entity}.{attr}"

    def write_attr(self, entity: str, attr: str, value: Any) -> None:
        """写入临时属性绑定。同属性多次写入，后写覆盖先写。"""
        key = self._attr_key(entity, attr)
        self.attribute_bindings[key] = value
        # 同时登记上下文实体（用于代词消解）
        if entity not in self.context_entities:
            self._entity_write_order.append(entity)
            # 创建一个轻量实体占位（DAG写入时实体已在陈述记忆中）
            self.context_entities[entity] = Entity(
                id=f"wm_{entity}", name=entity, attributes={"kind": "word"}
            )

    def read_attr(self, entity: str, attr: str) -> Any:
        """读取临时属性绑定。未绑定则抛 KeyError。"""
        key = self._attr_key(entity, attr)
        if key not in self.attribute_bindings:
            raise KeyError(f"属性未绑定: {key}")
        return self.attribute_bindings[key]

    def read_attr_or_none(self, entity: str, attr: str) -> Any:
        return self.attribute_bindings.get(self._attr_key(entity, attr))

    # ---- resolved_references ----
    def resolve(self, pronoun: str, entity_names: List[str]) -> None:
        """记录代词消解结果。"""
        self.resolved_references[pronoun] = entity_names

    def get_resolved(self, pronoun: str) -> List[str]:
        """获取代词消解结果。未消解则返回空列表。"""
        return self.resolved_references.get(pronoun, [])

    # ---- intermediate_results ----
    def put_node_output(self, node_id: str, value: Any) -> None:
        self.intermediate_results[f"{node_id}.out"] = value

    def get_node_output(self, node_id: str) -> Any:
        key = f"{node_id}.out"
        if key not in self.intermediate_results:
            raise KeyError(f"节点输出未就绪: {key}")
        return self.intermediate_results[key]

    def get_node_output_or_none(self, node_id: str) -> Any:
        return self.intermediate_results.get(f"{node_id}.out")

    # ---- 代词消解辅助 ----
    def resolve_pronoun(self, pronoun: str) -> List[str]:
        """依据上下文把代词解析为实体名列表。

        MVP规则：
        - 复数代词（他们/她们/它们/大家）→ 全部上下文实体
        - 单数代词（他/她/它/这/那）→ 最后写入的实体
        - 具体名词 → 该名词本身
        """
        # 已消解过则直接返回缓存
        cached = self.resolved_references.get(pronoun)
        if cached is not None:
            return cached
        plural = {"他们", "她们", "它们", "大家", "这些", "那些"}
        singular = {"他", "她", "它", "这", "那", "其"}
        if pronoun in plural:
            result = self.all_context_entity_names()
        elif pronoun in singular:
            names = self.all_context_entity_names()
            result = names[-1:] if names else []
        else:
            # 具体名词：该名词本身（若在上下文中）
            result = [pronoun] if self.has_context(pronoun) else [pronoun]
        self.resolved_references[pronoun] = result
        return result

    # ---- lifecycle ----
    def clear(self) -> None:
        self.context_entities.clear()
        self.attribute_bindings.clear()
        self.resolved_references.clear()
        self.intermediate_results.clear()
        self._entity_write_order.clear()
        self.ready_queue.clear()
        self.recent_entities.clear()

    # ---- 跨句实体追踪辅助（Phase 5d.5）----
    def register_entity_mention(self, entity_name: str) -> None:
        """登记最近提及实体（从 tokens 或 属性绑定中的 entity 值提取）"""
        if entity_name and entity_name not in self.recent_entities:
            self.recent_entities.append(entity_name)
        # 最多保留最近 20 个，避免无限增长
        if len(self.recent_entities) > 20:
            self.recent_entities = self.recent_entities[-20:]

    def recent_person_names(self, declarative_memory: Optional[Any] = None) -> List[str]:
        """筛选 recent_entities 中 pos=person_name 的实体名。
        Phase 5d.5: resolve_pronoun 扩展依据。
        当前阶段：若 declarative_memory 为 None，直接返回 recent_entities。
        """
        if declarative_memory is None:
            return list(self.recent_entities)
        result: List[str] = []
        for name in self.recent_entities:
            matches = declarative_memory.find_entity_by_name(name)
            for m in matches:
                if m.attributes.get("pos") == "person_name":
                    result.append(name)
                    break
            else:
                # 未标注 pos 时默认保留（兼容旧知识包，未标注 pos 的词）
                result.append(name)
        return result

    def __repr__(self) -> str:
        return (
            f"WorkingMemory(entities={len(self.context_entities)}, "
            f"bindings={len(self.attribute_bindings)}, "
            f"resolved={len(self.resolved_references)}, "
            f"results={len(self.intermediate_results)})"
        )
