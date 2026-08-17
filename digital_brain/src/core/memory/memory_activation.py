"""记忆激活机制 - 根据输入词素激活相关记忆并扩散到 1-3 跳邻居"""
from __future__ import annotations

from typing import Dict, List, Optional

from digital_brain.src.core.memory.declarative_memory import DeclarativeMemory
from digital_brain.src.core.memory.procedural_memory import ProceduralMemory
from digital_brain.src.core.models import (
    ActivatedKnowledge,
    DigitalBrainError,
    Entity,
    Procedure,
)


class MemoryActivation:
    """记忆激活器

    工作流程：
    1. 根据词素 token 匹配实体名称/别名，直接激活
    2. 根据词素序列匹配程序性记忆触发条件
    3. 对已激活实体进行扩散激活（1-3 跳邻居）
    4. 计算激活强度，冲突解决（保留强度最高者）
    """

    def __init__(
        self,
        declarative_memory: DeclarativeMemory,
        procedural_memory: ProceduralMemory,
        max_hops: int = 2,
        decay_factor: float = 0.6,
    ) -> None:
        self.declarative = declarative_memory
        self.procedural = procedural_memory
        self.max_hops = max_hops
        self.decay_factor = decay_factor

    # ---------- 主入口 ----------
    def activate(self, tokens: List[str]) -> List[ActivatedKnowledge]:
        """根据词素激活所有相关记忆，按激活强度降序返回"""
        activations: Dict[str, ActivatedKnowledge] = {}

        # Step 1: 实体激活
        entity_acts = self._activate_entities(tokens)
        for a in entity_acts:
            key = f"e:{a.entity.id}" if a.entity else id(a)
            if key not in activations or activations[key].activation_strength < a.activation_strength:
                activations[key] = a

        # Step 2: 扩散激活
        spread = self._spread_activation(entity_acts)
        for a in spread:
            key = f"e:{a.entity.id}" if a.entity else id(a)
            if key not in activations or activations[key].activation_strength < a.activation_strength:
                activations[key] = a

        # Step 3: 程序性记忆激活
        proc_acts = self._activate_procedures(tokens)
        for a in proc_acts:
            key = f"p:{a.procedure.id}" if a.procedure else id(a)
            if key not in activations or activations[key].activation_strength < a.activation_strength:
                activations[key] = a

        # 排序返回
        result = sorted(activations.values(), key=lambda x: -x.activation_strength)
        return result

    # ---------- 内部方法 ----------
    def _activate_entities(self, tokens: List[str]) -> List[ActivatedKnowledge]:
        """根据词素找到直接匹配的实体"""
        acts: List[ActivatedKnowledge] = []
        seen: set = set()
        for token in tokens:
            matches = self.declarative.find_entity_by_name(token)
            for e in matches:
                if e.id in seen:
                    continue
                seen.add(e.id)
                strength = 1.0
                acts.append(
                    ActivatedKnowledge(
                        entity=e,
                        activation_strength=strength,
                        source=f"direct_match:{token}",
                    )
                )
        return acts

    def _spread_activation(self, seeds: List[ActivatedKnowledge]) -> List[ActivatedKnowledge]:
        """从种子实体扩散到邻居（BFS，按 hop 衰减）"""
        result: List[ActivatedKnowledge] = []
        strength_by_id: Dict[str, float] = {}
        for seed in seeds:
            if not seed.entity:
                continue
            eid = seed.entity.id
            neighbors = self.declarative.get_neighbors(eid, hops=self.max_hops)
            for hop, ids in neighbors.items():
                for nb_id in ids:
                    hop_strength = seed.activation_strength * (self.decay_factor ** hop)
                    if nb_id not in strength_by_id or strength_by_id[nb_id] < hop_strength:
                        strength_by_id[nb_id] = hop_strength
        for nb_id, strength in strength_by_id.items():
            entity = self.declarative.get_entity(nb_id)
            if entity and strength > 0.1:
                result.append(
                    ActivatedKnowledge(
                        entity=entity,
                        activation_strength=strength,
                        source="spread",
                    )
                )
        return result

    def _activate_procedures(self, tokens: List[str]) -> List[ActivatedKnowledge]:
        """根据词素序列匹配程序性记忆触发条件"""
        matches = self.procedural.find_by_tokens(tokens)
        acts: List[ActivatedKnowledge] = []
        for proc in matches:
            total = max(1, len(proc.trigger.pattern_tokens))
            matched = sum(1 for t in proc.trigger.pattern_tokens if t in tokens)
            strength = matched / total
            if strength > 0:
                acts.append(
                    ActivatedKnowledge(
                        procedure=proc,
                        activation_strength=round(strength, 3),
                        source="trigger_match",
                    )
                )
        return acts

    # ---------- 冲突解决 ----------
    @staticmethod
    def resolve_conflicts(activations: List[ActivatedKnowledge], top_k: int = 10) -> List[ActivatedKnowledge]:
        """只保留激活强度最高的 top_k 个"""
        return sorted(activations, key=lambda x: -x.activation_strength)[:top_k]

    # ---------- 激活条件判断辅助 ----------
    def check_required_entities(self, proc: Procedure, available_entities: List[Entity]) -> bool:
        required = set(proc.trigger.required_entities)
        available_ids = set(e.id for e in available_entities)
        available_names = set(e.name for e in available_entities)
        if not required:
            return True
        # 按 id 或 name 匹配
        for r in required:
            if r not in available_ids and r not in available_names:
                return False
        return True

    # ---------- Phase 5c: pattern 神经元激活（5d.3 真正并入主流程）----------
    def activate_pattern(self, pattern_name: str) -> Optional[ActivatedKnowledge]:
        """激活指定 pattern 的神经元。Phase 5c 仅记录激活，不沿边扩散。"""
        if self.declarative is None:
            return None
        eid = f"pattern_{pattern_name}"
        entity = self.declarative.get_entity(eid)
        if entity is None:
            matches = self.declarative.find_entity_by_name(pattern_name)
            for m in matches:
                if m.attributes.get("kind") == "pattern":
                    entity = m
                    break
        if entity is None:
            return None
        return ActivatedKnowledge(
            entity=entity,
            activation_strength=0.9,
            source=f"pattern_hit:{pattern_name}",
        )
