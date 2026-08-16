"""陈述性记忆 - 知道'是什么'的记忆，包括事实、概念、关系

使用 NetworkX 作为图存储后端，存储 Entity 节点和 Relation 边。
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

import networkx as nx

from digital_brain.src.core.models import (
    Entity,
    EntityType,
    Relation,
    RelationType,
)


class DeclarativeMemory:
    """陈述性记忆存储类"""

    def __init__(self) -> None:
        self._graph = nx.DiGraph()
        self._entities: Dict[str, Entity] = {}
        self._relations: Dict[str, Relation] = {}
        self._name_index: Dict[str, List[str]] = {}       # name -> [entity_ids]
        self._alias_index: Dict[str, List[str]] = {}      # alias -> [entity_ids]

    # ---------- Entity CRUD ----------
    def add_entity(self, entity: Entity) -> str:
        if entity.id in self._entities:
            raise ValueError(f"Entity id '{entity.id}' already exists")
        self._entities[entity.id] = entity
        self._graph.add_node(entity.id, entity=entity)
        self._add_to_name_index(entity)
        return entity.id

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self._entities.get(entity_id)

    def find_entity_by_name(self, name: str) -> List[Entity]:
        ids = set(self._name_index.get(name, []))
        ids.update(self._alias_index.get(name, []))
        return [self._entities[eid] for eid in ids if eid in self._entities]

    def find_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        return [e for e in self._entities.values() if e.entity_type == entity_type]

    def update_entity(self, entity: Entity) -> None:
        if entity.id not in self._entities:
            raise ValueError(f"Entity id '{entity.id}' does not exist")
        old = self._entities[entity.id]
        self._remove_from_name_index(old)
        self._entities[entity.id] = entity
        self._graph.nodes[entity.id]["entity"] = entity
        self._add_to_name_index(entity)

    def delete_entity(self, entity_id: str) -> bool:
        if entity_id not in self._entities:
            return False
        e = self._entities.pop(entity_id)
        self._remove_from_name_index(e)
        # 删除相关关系
        rel_ids = list(self._graph.edges(entity_id, data="relation_id"))
        for src, tgt, rid in list(self._graph.in_edges(entity_id, data="relation_id")):
            if rid and rid in self._relations:
                del self._relations[rid]
        for src, tgt, rid in rel_ids:
            if rid and rid in self._relations:
                del self._relations[rid]
        self._graph.remove_node(entity_id)
        return True

    # ---------- Relation CRUD ----------
    def add_relation(self, relation: Relation) -> str:
        if relation.id in self._relations:
            raise ValueError(f"Relation id '{relation.id}' already exists")
        if relation.source_id not in self._entities:
            raise ValueError(f"Source entity '{relation.source_id}' not found")
        if relation.target_id not in self._entities:
            raise ValueError(f"Target entity '{relation.target_id}' not found")
        self._relations[relation.id] = relation
        self._graph.add_edge(
            relation.source_id,
            relation.target_id,
            relation_id=relation.id,
            relation_type=relation.relation_type,
            weight=relation.weight,
        )
        return relation.id

    def get_relation(self, relation_id: str) -> Optional[Relation]:
        return self._relations.get(relation_id)

    def find_relations_by_type(self, relation_type: RelationType) -> List[Relation]:
        return [r for r in self._relations.values() if r.relation_type == relation_type]

    def find_relations_of(self, entity_id: str, direction: str = "both") -> List[Relation]:
        results = []
        if direction in ("out", "both"):
            for src, tgt, data in self._graph.out_edges(entity_id, data=True):
                rid = data.get("relation_id")
                if rid and rid in self._relations:
                    results.append(self._relations[rid])
        if direction in ("in", "both"):
            for src, tgt, data in self._graph.in_edges(entity_id, data=True):
                rid = data.get("relation_id")
                if rid and rid in self._relations:
                    results.append(self._relations[rid])
        return results

    def get_neighbors(self, entity_id: str, hops: int = 1) -> Dict[str, List[str]]:
        """获取 N 跳邻居，返回 {entity_id: path_weight} 列表形式的 entity_ids 按跳数"""
        if entity_id not in self._entities:
            return {}
        result: Dict[str, List[str]] = {h: [] for h in range(1, hops + 1)}
        visited = {entity_id}
        current = {entity_id}
        for hop in range(1, hops + 1):
            next_set = set()
            for n in current:
                for nb in self._graph.neighbors(n):
                    if nb not in visited:
                        visited.add(nb)
                        next_set.add(nb)
                        result[hop].append(nb)
                for nb in self._graph.predecessors(n):
                    if nb not in visited:
                        visited.add(nb)
                        next_set.add(nb)
                        result[hop].append(nb)
            current = next_set
            if not current:
                break
        return result

    def delete_relation(self, relation_id: str) -> bool:
        if relation_id not in self._relations:
            return False
        rel = self._relations.pop(relation_id)
        self._graph.remove_edge(rel.source_id, rel.target_id)
        return True

    # ---------- 持久化 ----------
    def save_json(self, entity_path: str, relation_path: str) -> None:
        with open(entity_path, "w", encoding="utf-8") as f:
            json.dump([e.dict() for e in self._entities.values()], f, ensure_ascii=False, indent=2)
        with open(relation_path, "w", encoding="utf-8") as f:
            json.dump([r.dict() for r in self._relations.values()], f, ensure_ascii=False, indent=2)

    def load_json(self, entity_path: str, relation_path: str) -> None:
        with open(entity_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for ed in data:
                self.add_entity(Entity(**ed))
        with open(relation_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for rd in data:
                self.add_relation(Relation(**rd))

    # ---------- 内部方法 ----------
    def _add_to_name_index(self, entity: Entity) -> None:
        self._name_index.setdefault(entity.name, []).append(entity.id)
        for alias in entity.aliases:
            self._alias_index.setdefault(alias, []).append(entity.id)

    def _remove_from_name_index(self, entity: Entity) -> None:
        if entity.name in self._name_index and entity.id in self._name_index[entity.name]:
            self._name_index[entity.name].remove(entity.id)
            if not self._name_index[entity.name]:
                del self._name_index[entity.name]
        for alias in entity.aliases:
            if alias in self._alias_index and entity.id in self._alias_index[alias]:
                self._alias_index[alias].remove(entity.id)
                if not self._alias_index[alias]:
                    del self._alias_index[alias]

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph

    @property
    def entity_count(self) -> int:
        return len(self._entities)

    @property
    def relation_count(self) -> int:
        return len(self._relations)


class WorldModel:
    """简化世界模型 - 封装陈述性记忆，提供更高层的世界视图"""

    def __init__(self, declarative_memory: Optional[DeclarativeMemory] = None) -> None:
        self.memory = declarative_memory or DeclarativeMemory()

    def describe_entity(self, entity_id: str) -> str:
        e = self.memory.get_entity(entity_id)
        if not e:
            return f"[未知实体 {entity_id}]"
        parts = [f"({e.entity_type}) {e.name}"]
        if e.aliases:
            parts.append(f"别名: {','.join(e.aliases)}")
        rels = self.memory.find_relations_of(entity_id)
        if rels:
            rel_strs = []
            for r in rels:
                src = self.memory.get_entity(r.source_id)
                tgt = self.memory.get_entity(r.target_id)
                if src and tgt:
                    rel_strs.append(f"{src.name} -{r.relation_type}-> {tgt.name}")
            parts.append("关系: " + "; ".join(rel_strs))
        return " | ".join(parts)
