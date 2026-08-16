"""知识构建工具 - 为陈述性/程序性记忆填充初始数学知识

⚠️  已废弃（DEPRECATED）
-----------------------------------------------------------
本文件属于"硬编码初始知识"的旧方案，违反「没有初始能力，所有能力通过学习获得」
的核心设计哲学。请改用 `digital_brain.src.interfaces.symbolic_interface.SymbolicInterface`
中的教学接口：

  * `brain = SymbolicInterface(auto_build=False, auto_learn_tokenizer=False)`  # 空白脑
  * `brain.learn_number_word(...)`         学数字词
  * `brain.learn_operator_word(...)`       学操作符词
  * `brain.learn_marker_word(...)`         学标记（= ? 等）
  * `brain.learn_relation(...)`            学关系（后继等）
  * `brain.learn_procedure(...)`           学程序性记忆（算法）
  * `brain.teach_default_curriculum(10)`   标准入门课：一次性教完 MVP 全部

保持本文件仅用于：
  * 兼容旧代码路径（`auto_build=True` 在内部已改为调用 teach_*，不再使用本类）
  * 历史参考
-----------------------------------------------------------

MVP 目标：支持 10 以内加减法
填充内容：
- 实体：数字 0-10，操作符 (+ -)，符号 = ?
- 关系：数字的后继关系（1 -> 2 -> 3...）
- 程序性记忆：counting, mapping, merging, adding, subtracting
"""
from __future__ import annotations

import warnings
from typing import List, Tuple

from digital_brain.src.core.memory.declarative_memory import DeclarativeMemory
from digital_brain.src.core.memory.procedural_memory import ProceduralMemory
from digital_brain.src.core.models import (
    Entity,
    EntityType,
    OperationStep,
    Procedure,
    Relation,
    RelationType,
    TriggerCondition,
)


class KnowledgeBuilder:
    """初始知识构建器"""

    CN_NUMBERS = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]

    def __init__(self, max_number: int = 10) -> None:
        self.max_number = max_number
        self._eid_counter = 0
        self._rid_counter = 0
        self._pid_counter = 0

    # ---------- ID helpers ----------
    def _eid(self, prefix: str = "ent") -> str:
        self._eid_counter += 1
        return f"{prefix}_{self._eid_counter:04d}"

    def _rid(self) -> str:
        self._rid_counter += 1
        return f"rel_{self._rid_counter:04d}"

    def _pid(self) -> str:
        self._pid_counter += 1
        return f"proc_{self._pid_counter:04d}"

    # ---------- 构建陈述性记忆 ----------
    def build_declarative(self, mem: DeclarativeMemory) -> List[Entity]:
        """填充数字 0..max_number 和操作符实体，以及后继关系"""
        number_entities: List[Entity] = []
        # 1. 数字实体：name 为阿拉伯数字，aliases 含中文
        for value in range(0, self.max_number + 1):
            eid = self._eid("num")
            aliases = []
            if 0 <= value < len(self.CN_NUMBERS):
                aliases.append(self.CN_NUMBERS[value])
            # 特殊：2 的别名 "两"
            if value == 2:
                aliases.append("两")
            entity = Entity(
                id=eid,
                name=str(value),
                aliases=aliases,
                entity_type=EntityType.ABSTRACT,
                attributes={"kind": "number", "value": value, "numeric_value": value},
                embodied_mapping=[f"finger_{value}"],
            )
            mem.add_entity(entity)
            number_entities.append(entity)

        # 2. 操作符实体
        ops = [
            ("+", ["加", "加上", "＋"], "addition"),
            ("-", ["减", "减去", "－"], "subtraction"),
            ("*", ["乘", "乘以", "×"], "multiplication"),
            ("/", ["除", "除以", "÷"], "division"),
            ("=", ["等于", "＝"], "equality"),
            ("?", ["？", "多少", "几", "什么"], "question"),
        ]
        for name, aliases, kind in ops:
            eid = self._eid("op")
            entity = Entity(
                id=eid,
                name=name,
                aliases=aliases,
                entity_type=EntityType.ABSTRACT,
                attributes={"kind": "operator", "op_type": kind},
            )
            mem.add_entity(entity)

        # 3. 后继关系：0->1, 1->2 ...
        for i in range(len(number_entities) - 1):
            src = number_entities[i]
            tgt = number_entities[i + 1]
            rel = Relation(
                id=self._rid(),
                source_id=src.id,
                target_id=tgt.id,
                relation_type=RelationType.SUCCESSOR,
                weight=1.0,
                attributes={"delta": 1},
            )
            mem.add_relation(rel)

        return number_entities

    # ---------- 构建程序性记忆 ----------
    def build_procedural(self, pm: ProceduralMemory) -> List[Procedure]:
        """注册基础算法作为程序性记忆"""
        procedures: List[Procedure] = []
        templates: List[Tuple[str, str, List[str], List[str]]] = [
            # (name, desc, trigger_tokens, category)
            ("counting", "计数算法：对集合或数字计数", ["多少", "几", "数", "count"], "algorithm"),
            ("mapping", "具身映射：数字 <-> 集合", ["映射", "手指", "具身", "map"], "algorithm"),
            ("merging", "合并算法：集合并在一起", ["合并", "放一起", "merge"], "algorithm"),
            ("adding", "加法算法：a + b = count(merge(map(a),map(b)))", ["+", "加", "加上", "add"], "algorithm"),
            ("subtracting", "减法算法：a - b = 拿走b个后计数", ["-", "减", "减去", "sub"], "algorithm"),
        ]
        # 创建 adding/subtracting 作为组合型 procedure，其他为基础
        counting_id = None
        mapping_id = None
        merging_id = None
        for name, desc, triggers, category in templates:
            pid = self._pid()
            steps = []
            if name == "adding":
                steps = [
                    OperationStep(order=1, action="mapping", parameters={"operand": "a"}, description="映射 a 为集合 A"),
                    OperationStep(order=2, action="mapping", parameters={"operand": "b"}, description="映射 b 为集合 B"),
                    OperationStep(order=3, action="merging", parameters={"sets": ["A", "B"]}, description="合并 A、B"),
                    OperationStep(order=4, action="counting", parameters={"set": "merged"}, description="计数合并后集合"),
                ]
            elif name == "subtracting":
                steps = [
                    OperationStep(order=1, action="mapping", parameters={"operand": "a"}, description="映射 a 为集合 A"),
                    OperationStep(order=2, action="remove", parameters={"count": "b"}, description=f"从 A 中移除 b 个元素"),
                    OperationStep(order=3, action="counting", parameters={"set": "remaining"}, description="计数剩余集合"),
                ]
            deps = []
            if name == "adding":
                if counting_id and mapping_id and merging_id:
                    deps = [mapping_id, mapping_id, merging_id, counting_id]
            if name == "subtracting":
                if counting_id and mapping_id:
                    deps = [mapping_id, counting_id]
            proc = Procedure(
                id=pid,
                name=name,
                description=desc,
                trigger=TriggerCondition(
                    pattern_tokens=triggers,
                    required_entities=[],
                ),
                steps=steps,
                dependencies=deps,
                parent_id=None,
                category=category,
            )
            pm.add_procedure(proc)
            procedures.append(proc)
            if name == "counting":
                counting_id = pid
            elif name == "mapping":
                mapping_id = pid
            elif name == "merging":
                merging_id = pid
        return procedures

    def build_all(
        self, declarative: DeclarativeMemory, procedural: ProceduralMemory
    ) -> dict:
        ents = self.build_declarative(declarative)
        procs = self.build_procedural(procedural)
        return {
            "number_entities": len(ents),
            "procedures": len(procs),
            "declarative_total": declarative.entity_count,
            "relations_total": declarative.relation_count,
            "procedural_total": procedural.procedure_count,
        }
