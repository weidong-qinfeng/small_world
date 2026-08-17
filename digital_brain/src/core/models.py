"""数字大脑核心数据模型 - Pydantic 定义"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DigitalBrainError(Exception):
    """数字大脑基础异常"""
    pass


class MemoryActivationError(DigitalBrainError):
    """记忆激活异常"""
    pass


class EntityType(str, Enum):
    """实体分类体系"""
    PHYSICAL = "physical"          # 物理实体：手指、苹果
    ABSTRACT = "abstract"          # 抽象概念：数字1、加法
    RELATION = "relation"          # 关系类型
    PROCESS = "process"            # 过程/动作


class RelationType(str, Enum):
    """关系类型体系"""
    MAPPING = "mapping"            # 映射：如 <-> 具身映射
    COMPOSITION = "composition"    # 组成：部分-整体
    CAUSAL = "causal"              # 因果：导致
    EQUIVALENCE = "equivalence"    # 等价：相等
    CONTAINMENT = "containment"    # 包含：属于
    DEPENDENCY = "dependency"      # 依赖：需要
    SUCCESSOR = "successor"        # 后继：1 的后继是 2
    TRIGGERS = "triggers"          # 激活触发：pattern → operation
    PASSES_PARAM = "passes_param"  # 参数传递：词实体 → operation
    FEEDS = "feeds"                # 数据流：operation A → operation B
    REQUIRES = "requires"          # 依赖：operation B → operation A（B 需 A 先完成）
    MAPS_TO = "maps_to"            # 实体映射：概念 → operation（如"有"→write_memory）
    RESOLVES_TO = "resolves_to"    # 指代消解：代词 → 上下文实体


class Entity(BaseModel):
    """实体数据模型 - 陈述性记忆的基本节点"""
    id: str
    name: str
    aliases: List[str] = Field(default_factory=list)
    entity_type: EntityType = EntityType.ABSTRACT
    attributes: Dict[str, Any] = Field(default_factory=dict)
    embodied_mapping: List[str] = Field(default_factory=list)

    class Config:
        use_enum_values = True


class Relation(BaseModel):
    """关系数据模型 - 连接实体的边"""
    id: str
    source_id: str
    target_id: str
    relation_type: RelationType
    weight: float = 1.0
    attributes: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class OperationStep(BaseModel):
    """操作步骤 - 程序性记忆中的一步"""
    order: int
    action: str                              # 动作名：如 "count", "map", "merge", "add"
    parameters: Dict[str, Any] = Field(default_factory=list if False else dict)
    description: str = ""


class TriggerCondition(BaseModel):
    """触发条件 - 程序性记忆何时被激活"""
    pattern_tokens: List[str] = Field(default_factory=list)     # 匹配的词素序列
    required_entities: List[str] = Field(default_factory=list)  # 需要的实体
    context: Dict[str, Any] = Field(default_factory=dict)       # 上下文条件


class Procedure(BaseModel):
    """程序性记忆数据模型 - 知道怎么做"""
    id: str
    name: str
    description: str = ""
    trigger: TriggerCondition = Field(default_factory=TriggerCondition)
    steps: List[OperationStep] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)       # 依赖的其他 procedure id
    parent_id: Optional[str] = None                              # 父 procedure（层次结构）
    category: str = "algorithm"                                  # algorithm / skill / strategy
    attributes: Dict[str, Any] = Field(default_factory=dict)


class ActivatedKnowledge(BaseModel):
    """被激活的知识单元 - 进入工作区"""
    entity: Optional[Entity] = None
    procedure: Optional[Procedure] = None
    relation: Optional[Relation] = None
    activation_strength: float = 0.0
    source: str = ""                                              # 激活来源
