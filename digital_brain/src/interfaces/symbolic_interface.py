"""符号输入输出接口 - 数字大脑的主对外接口

封装完整的处理管线：
    文本输入 → 词素拆分 → 工作区接收 → 记忆激活 → 模式匹配 → 意图识别
    → 推理/算法执行 → 输出缓冲 → 结构化结果/答案

核心设计哲学（用户要求）：
    系统「初始没有任何领域能力」，只具备「学习单词及含义」这一种能力。
    所有数字 / 操作符 / 算法 / 分词 全部通过 `learn_*` 接口从零教给大脑。
    提供 `teach_default_curriculum()` 作为"标准入门课程"（0-10 数字、加减乘除、5 条算法），
    但它本身不是先天知识，只是一次性调用学习接口的便利函数。
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from digital_brain.src.core.memory.declarative_memory import DeclarativeMemory
from digital_brain.src.core.memory.memory_activation import MemoryActivation
from digital_brain.src.core.memory.memory_consolidation import MemoryConsolidation
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
from digital_brain.src.core.workspace.reasoning_area import AlgorithmRegistry, ReasoningArea, _V2_ALGORITHM_KEYS
from digital_brain.src.core.workspace.workspace import Workspace
from digital_brain.src.core.pattern.pattern_matcher import PatternMatcher
from digital_brain.src.core.pattern.intent_recognizer import IntentRecognizer
from digital_brain.src.interfaces.physical_interface import PhysicalInterface
from digital_brain.src.utils.tokenizer import LearnableTokenizer, Tokenizer


# ============================================================
# 结果数据类
# ============================================================

@dataclass
class BrainResult:
    """数字大脑的求解结果"""
    answer: Any
    confidence: float
    reasoning_chain: List[Dict[str, Any]] = field(default_factory=list)
    tokens: List[str] = field(default_factory=list)
    activated_count: int = 0
    raw_text: str = ""
    dag_text: str = ""   # 问题理解 DAG 的人类可读描述（无 DAG 路径时为空）

    def format(self) -> str:
        lines = [f"输入: {self.raw_text}"]
        lines.append(f"词素: {self.tokens}")
        lines.append(f"激活知识数: {self.activated_count}")
        if self.dag_text:
            lines.append("--- 问题理解 DAG ---")
            lines.append(self.dag_text)
        lines.append("--- 推理链 ---")
        for i, step in enumerate(self.reasoning_chain, 1):
            desc = step.get("description", "")
            action = step.get("action", "")
            line = f"  {i}. [{action}] {desc}"
            outs = step.get("outputs")
            if outs and ("result" in outs or "answer" in outs):
                line += f" => {outs.get('result', outs.get('answer'))}"
            lines.append(line)
        lines.append(f"--- 最终答案: {self.answer} (置信度 {self.confidence:.2f}) ---")
        return "\n".join(lines)


# ============================================================
# 主接口
# ============================================================

class SymbolicInterface:
    """数字大脑对外符号接口

    默认启动：**空白大脑**（`auto_learn_tokenizer=False`, `auto_build=False`），
    任何数字/操作符/算法都不会被先天知道。
    """

    DEFAULT_CONFIG_PATH = "digital_brain/config/config.yaml"

    def __init__(
        self,
        config_path: Optional[str] = None,
        use_embodied: bool = True,
        *,
        auto_build: bool = False,           # 默认：不自动构建（遵循"初始没有能力"）
        auto_learn_tokenizer: bool = False, # 默认：分词器也空白（不加载 tokenization_examples.json）
        storage_dir: Optional[str] = None,  # v2: 记忆持久化目录，None=不持久化
        auto_restore: bool = False,         # v2: 启动时自动从 storage_dir 恢复
        innate: "Optional[InnateInterface]" = None,  # M7 P-A2: 先天机制层（感知/运动底座）注入；None=无机制层
    ) -> None:
        # 1. 加载配置
        self.config = self._load_config(config_path or self.DEFAULT_CONFIG_PATH)
        # 2. 记忆系统（真正空白）
        self.declarative = DeclarativeMemory()
        self.procedural = ProceduralMemory()
        self.consolidation = MemoryConsolidation(self.declarative, self.procedural)
        # v2: 持久化目录
        self.storage_dir = storage_dir
        self._auto_consolidate = storage_dir is not None  # 学习后自动固化
        # v2: 启动时自动恢复（优先级高于 auto_build）
        restored = False
        if auto_restore and storage_dir and MemoryConsolidation.has_saved_state(storage_dir):
            stats = self.consolidation.restore_from_disk(storage_dir)
            restored = stats.get("restored", False)
        # v2: 先天 operation 神经元入图谱（L0 硬件指令集）—— 恢复后再补，
        #     保证：① 持久化数据中已存在时不冲突（upsert），② 旧数据缺失时补齐
        self._ensure_operation_neurons()
        # 3. 激活引擎
        act_cfg = self.config.get("memory", {}).get("activation", {})
        self.memory_activation = MemoryActivation(
            self.declarative,
            self.procedural,
            max_hops=act_cfg.get("max_hops", 2),
            decay_factor=act_cfg.get("decay_factor", 0.6),
        )
        # 4. 工作区
        ws_cfg = self.config.get("workspace", {})
        self.workspace = Workspace(
            self.declarative,
            self.procedural,
            memory_activation=self.memory_activation,
            input_capacity=ws_cfg.get("input_capacity", 100),
            activation_capacity=ws_cfg.get("activation_capacity", 20),
        )
        # 5. 推理区（算法执行器 = "执行硬件"，Phase 1 手工搭建；不是知识，保留为底层机制）
        #    但 PatternMatcher / IntentRecognizer 必须绑定 declarative memory：
        #    没学过的词一律不认（"没有初始能力"）
        self.algorithm_registry = AlgorithmRegistry()
        self.pattern_matcher = PatternMatcher(declarative_memory=self.declarative)
        self.intent_recognizer = IntentRecognizer(declarative_memory=self.declarative)
        self.reasoning = ReasoningArea(
            algorithm_registry=self.algorithm_registry,
            procedural_memory=self.procedural,
            pattern_matcher=self.pattern_matcher,
            intent_recognizer=self.intent_recognizer,
            use_embodied=use_embodied,
            declarative_memory=self.declarative,
        )
        # 6. 工具：Tokenizer 默认空白（不会自动加载 JSON 样本 —— 那等于先天有能力）
        #    但 LearnableTokenizer 本身是"学习机制"，允许存在。
        #    v2: Tokenizer 绑定 declarative_memory，词素知识唯一来源是图谱
        self.tokenizer: Tokenizer = Tokenizer(
            auto_learn=auto_learn_tokenizer,
            declarative_memory=self.declarative,
        )
        self.physical = PhysicalInterface()
        # M7 P-A2：先天机制层（感知/运动底座）注入——可选参数，默认 None（117 基线
        # 行为不变，组合复用纪律）；非 None 时仅被 M7 场景测试消费，solve 语义零修改
        # （InnateInterface 定义见 digital_brain/src/interfaces/innate_interface.py）。
        self.innate = innate
        # 7. 向后兼容：auto_build=True 时走标准课程（内部也是调用 teach_*，模拟教学）
        #    v2: 若已从硬盘恢复，跳过 auto_build（避免覆盖已学知识）
        if auto_build and not restored:
            self.teach_default_curriculum(
                max_number=self.config.get("knowledge", {}).get("max_initial_number", 10)
            )

    # ============================================================
    # 教学接口 1. 「学习单词（及含义）」—— 符合用户提出的初始唯一能力
    # ============================================================

    def learn_number_word(
        self,
        symbol: str,
        value: int,
        aliases: Optional[Sequence[str]] = None,
        embodied_tag: Optional[str] = None,
    ) -> str:
        """教大脑一个数字单词：`symbol` 表示的数字含义是 `value`。

        例如：
            brain.learn_number_word("1", 1, aliases=["一"])
            brain.learn_number_word("两", 2)
        """
        aliases = list(aliases or [])
        # v2: 词素知识写入图谱（实体本身即词素），无需单独教 tokenizer
        # 若 value 是多位数，在图谱中写入 digit_merge 规则
        if len(str(abs(int(value)))) >= 2:
            self.tokenizer._ensure_digit_merge_rule()

        attrs: Dict[str, Any] = {"kind": "number", "value": int(value), "numeric_value": int(value)}
        embodied = [embodied_tag] if embodied_tag else [f"finger_{value}"]
        entity = Entity(
            id=self._new_entity_id(f"num_{value}"),
            name=symbol,
            aliases=aliases,
            entity_type=EntityType.ABSTRACT,
            attributes=attrs,
            embodied_mapping=embodied,
        )
        # 若 symbol/aliases 已有同名实体，走 update，避免冲突
        existing = self.declarative.find_entity_by_name(symbol)
        if existing:
            old = existing[0]
            entity.id = old.id
            self.declarative.update_entity(entity)
            return old.id
        return self.declarative.add_entity(entity)

    def learn_operator_word(
        self,
        symbol: str,
        op_type: str,
        aliases: Optional[Sequence[str]] = None,
    ) -> str:
        """教大脑一个操作符单词：`symbol` 对应 `add/sub/mul/div/...` 运算

        例如：
            brain.learn_operator_word("+", "add", aliases=["加", "加上", "＋"])
        """
        aliases = list(aliases or [])
        # v2: 词素知识写入图谱（实体本身即词素）
        entity = Entity(
            id=self._new_entity_id(f"op_{op_type}"),
            name=symbol,
            aliases=aliases,
            entity_type=EntityType.ABSTRACT,
            attributes={"kind": "operator", "op_type": op_type},
        )
        existing = self.declarative.find_entity_by_name(symbol)
        if existing:
            old = existing[0]
            entity.id = old.id
            self.declarative.update_entity(entity)
            return old.id
        return self.declarative.add_entity(entity)

    def learn_marker_word(
        self,
        symbol: str,
        marker_kind: str,
        aliases: Optional[Sequence[str]] = None,
    ) -> str:
        """教大脑一个标记符号：等号、问号、比较号等。

        marker_kind 示例："equality"(等号), "question"(疑问号), "compare_gt"/"compare_lt"
        """
        aliases = list(aliases or [])
        # v2: 词素知识写入图谱（实体本身即词素）
        entity = Entity(
            id=self._new_entity_id(f"marker_{marker_kind}"),
            name=symbol,
            aliases=aliases,
            entity_type=EntityType.ABSTRACT,
            attributes={"kind": "marker", "marker_kind": marker_kind},
        )
        existing = self.declarative.find_entity_by_name(symbol)
        if existing:
            old = existing[0]
            entity.id = old.id
            self.declarative.update_entity(entity)
            return old.id
        return self.declarative.add_entity(entity)

    def learn_word(
        self,
        symbol: str,
        meaning: str = "",
        pinyin: str = "",
        word_type: str = "",
        aliases: Optional[Sequence[str]] = None,
        pos: str = "",
        gender: str = "",
    ) -> str:
        """教大脑一个普通字词及其含义。

        例如：
            brain.learn_word("天", meaning="天空", pinyin="tiān", word_type="名词")
            brain.learn_word("大", meaning="size大", pinyin="dà", word_type="形容词")
            brain.learn_word("爸爸", meaning="父亲", pinyin="bà ba", word_type="名词", aliases=["爸"])
        """
        aliases = list(aliases or [])
        # v2: 词素知识写入图谱（实体本身即词素）
        attrs: Dict[str, Any] = {
            "kind": "word",
            "meaning": meaning,
            "pinyin": pinyin,
            "word_type": word_type,
        }
        if pos:
            attrs["pos"] = pos
        if gender:
            attrs["gender"] = gender
        entity = Entity(
            id=self._new_entity_id(f"word_{symbol}"),
            name=symbol,
            aliases=aliases,
            entity_type=EntityType.ABSTRACT,
            attributes=attrs,
        )
        existing = self.declarative.find_entity_by_name(symbol)
        if existing:
            old = existing[0]
            entity.id = old.id
            self.declarative.update_entity(entity)
            return old.id
        return self.declarative.add_entity(entity)

    def learn_relation(
        self,
        source_symbol_or_id: str,
        target_symbol_or_id: str,
        relation_type: RelationType,
        weight: float = 1.0,
        attrs: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """教大脑一条关系。源/目标可以是符号名，会自动查找。"""
        src = self._resolve_entity(source_symbol_or_id)
        tgt = self._resolve_entity(target_symbol_or_id)
        if src is None or tgt is None:
            return None
        rel = Relation(
            id=self._new_relation_id(),
            source_id=src.id,
            target_id=tgt.id,
            relation_type=relation_type,
            weight=weight,
            attributes=dict(attrs or {}),
        )
        return self.declarative.add_relation(rel)

    # ============================================================
    # 教学接口 2. 「学习程序性记忆（怎么做）」
    # ============================================================

    def learn_procedure(
        self,
        name: str,
        algorithm_key: str,
        trigger_words: Sequence[str],
        *,
        steps: Optional[List[Dict[str, Any]]] = None,
        dependencies: Optional[List[str]] = None,
        description: str = "",
        category: str = "algorithm",
    ) -> str:
        """教大脑一条程序性记忆（知道怎么做）。

        - `name`: 如 "adding"
        - `algorithm_key`: 对应 algorithm_registry 中的键（counting/mapping/merging/adding/subtracting）
        - `trigger_words`: 看到哪些词会触发这条程序（用于记忆激活 + 倒排索引）

        调用后：该 procedure 被写入 procedural memory，工作区激活时可被检索到。
        """
        if algorithm_key not in self.algorithm_registry.all_algorithms() and \
           algorithm_key not in _V2_ALGORITHM_KEYS:
            raise ValueError(
                f"algorithm_key='{algorithm_key}' 不在注册表里。"
                f"可用: {list(self.algorithm_registry.all_algorithms().keys())} + "
                f"v2: {sorted(_V2_ALGORITHM_KEYS)}"
            )
        # v2: trigger_words 作为词素存入图谱
        for w in trigger_words:
            if w:
                self.tokenizer._ensure_morpheme_entity(w)
        # 步骤：用户有传就用；否则按 algorithm_key 生成标准步骤
        op_steps: List[OperationStep] = []
        if steps:
            for i, s in enumerate(steps, start=1):
                op_steps.append(OperationStep(order=i, **s))
        else:
            op_steps = self._default_steps_for_algorithm(algorithm_key)
        # 依赖：用户有传 + 根据步骤自动推导
        deps_ids: List[str] = []
        if dependencies:
            for dep_name in dependencies:
                existing = self.procedural.find_by_name(dep_name)
                if existing:
                    deps_ids.append(existing[0].id)
        proc = Procedure(
            id=self._new_procedure_id(name),
            name=name,
            description=description or f"程序性记忆：{name}（{algorithm_key}）",
            trigger=TriggerCondition(
                pattern_tokens=list(trigger_words),
                required_entities=[],
            ),
            steps=op_steps,
            dependencies=deps_ids,
            parent_id=None,
            category=category,
            attributes={"algorithm_key": algorithm_key},
        )
        proc_id = self.procedural.add_procedure(proc)
        # v2: procedure 的陈述性镜像入图谱
        try:
            eid = f"ent_proc_{name}"
            proc_entity = Entity(
                id=eid,
                name=name,
                aliases=[],
                entity_type=EntityType.ABSTRACT,
                attributes={
                    "kind": "procedure",
                    "procedure_id": proc.id,
                    "algorithm_key": algorithm_key,
                    "steps": [s.dict() for s in op_steps],
                },
            )
            existing = self.declarative.get_entity(eid)
            if existing:
                self.declarative.update_entity(proc_entity)
            else:
                try:
                    self.declarative.add_entity(proc_entity)
                except ValueError:
                    self.declarative.update_entity(proc_entity)
        except Exception:
            pass
        return proc_id

    # ============================================================
    # 分词教学接口（§6.2）—— 复用之前能力
    # ============================================================

    def learn_tokenizer_example(self, text: str, expected_tokens: List[str]) -> dict:
        assert isinstance(self.tokenizer, Tokenizer)
        return self.tokenizer.learn_from_example(text, expected_tokens)

    def learn_tokenizer_dataset(self, examples: List[dict]) -> dict:
        assert isinstance(self.tokenizer, Tokenizer)
        return self.tokenizer.learn_from_dataset(examples)

    def tokenizer_stats(self) -> dict:
        t = self.tokenizer
        if isinstance(t, LearnableTokenizer):
            return {
                "learned_examples": t.learned_examples,
                "known_morphemes_count": len(t.known_morphemes),
                "merge_digit_sequences_learned": t.merge_digit_sequences,
                "conflicts": len(t.conflicts),
            }
        return {"type": "unknown_tokenizer"}

    # ============================================================
    # v2: 模式学习接口
    # ============================================================

    def learn_pattern(
        self,
        name: str,
        slots: List[Dict[str, Any]],
        action: str = "",
        priority: float = 0.5,
        description: str = "",
        dag_template: Optional[Dict[str, Any]] = None,
    ) -> str:
        """教大脑一个模式：把模式作为实体写入知识图谱。

        Args:
            name: 模式名（如 "binary_op_question"）
            slots: 模式的 slot 定义列表，每个 slot 含 type/literal/capture/optional 字段
            action: 匹配后触发的动作（如 "build_dag:binary_op"）
            priority: 优先级（用于多模式同时命中时排序）
            description: 模式描述
            dag_template: DAG 模板定义，若提供则 DAGBuilder 优先从该模板构建节点

        学习后：PatternMatcher._load_patterns_from_memory() 会自动读到该模式。
        同时，slots 中的字面值（如"的""之和"）也会作为词素写入图谱。
        """
        # 把 slots 中的字面值作为词素写入图谱
        for slot in slots:
            literal = slot.get("literal")
            if literal:
                self.tokenizer._ensure_morpheme_entity(literal)
        # 创建或更新 pattern 实体
        eid = f"pattern_{name}"
        existing = self.declarative.get_entity(eid)
        attrs = {
            "kind": "pattern",
            "slots": slots,
            "action": action,
            "priority": priority,
            "description": description,
        }
        if dag_template is not None:
            attrs["dag_template"] = dag_template
        entity = Entity(
            id=eid,
            name=name,
            aliases=[],
            entity_type=EntityType.ABSTRACT,
            attributes=attrs,
        )
        if existing:
            self.declarative.update_entity(entity)
            # Phase 5c.4: pattern → operation triggers 边（框架铺垫，5d 启用）
            try:
                if dag_template:
                    for node_tpl in dag_template.get("nodes", []):
                        action_name = node_tpl.get("action", "")
                        if not action_name:
                            continue
                        # 找同 name 的 operation 神经元
                        op_entities = self.declarative.find_entity_by_name(action_name)
                        for op_ent in op_entities:
                            if op_ent.attributes.get("kind") != "operation":
                                continue
                            # 已有同名边则跳过
                            rel_exists = False
                            for r in self.declarative.find_relations_of(entity.id, "out"):
                                if r.target_id == op_ent.id and r.relation_type == RelationType.TRIGGERS:
                                    rel_exists = True
                                    break
                            if not rel_exists:
                                self.declarative.add_relation(Relation(
                                    id=self._new_relation_id(),
                                    source_id=entity.id,
                                    target_id=op_ent.id,
                                    relation_type=RelationType.TRIGGERS,
                                    weight=1.0,
                                    attributes={"from_template": True},
                                ))
            except Exception:
                pass  # 触发器边创建失败不影响学习主流程
            return eid
        try:
            self.declarative.add_entity(entity)
            # Phase 5c.4: pattern → operation triggers 边（框架铺垫，5d 启用）
            try:
                if dag_template:
                    for node_tpl in dag_template.get("nodes", []):
                        action_name = node_tpl.get("action", "")
                        if not action_name:
                            continue
                        # 找同 name 的 operation 神经元
                        op_entities = self.declarative.find_entity_by_name(action_name)
                        for op_ent in op_entities:
                            if op_ent.attributes.get("kind") != "operation":
                                continue
                            # 已有同名边则跳过
                            rel_exists = False
                            for r in self.declarative.find_relations_of(entity.id, "out"):
                                if r.target_id == op_ent.id and r.relation_type == RelationType.TRIGGERS:
                                    rel_exists = True
                                    break
                            if not rel_exists:
                                self.declarative.add_relation(Relation(
                                    id=self._new_relation_id(),
                                    source_id=entity.id,
                                    target_id=op_ent.id,
                                    relation_type=RelationType.TRIGGERS,
                                    weight=1.0,
                                    attributes={"from_template": True},
                                ))
            except Exception:
                pass  # 触发器边创建失败不影响学习主流程
            return eid
        except ValueError:
            # 已存在（按 id），走 update
            self.declarative.update_entity(entity)
            # Phase 5c.4: pattern → operation triggers 边（框架铺垫，5d 启用）
            try:
                if dag_template:
                    for node_tpl in dag_template.get("nodes", []):
                        action_name = node_tpl.get("action", "")
                        if not action_name:
                            continue
                        # 找同 name 的 operation 神经元
                        op_entities = self.declarative.find_entity_by_name(action_name)
                        for op_ent in op_entities:
                            if op_ent.attributes.get("kind") != "operation":
                                continue
                            # 已有同名边则跳过
                            rel_exists = False
                            for r in self.declarative.find_relations_of(entity.id, "out"):
                                if r.target_id == op_ent.id and r.relation_type == RelationType.TRIGGERS:
                                    rel_exists = True
                                    break
                            if not rel_exists:
                                self.declarative.add_relation(Relation(
                                    id=self._new_relation_id(),
                                    source_id=entity.id,
                                    target_id=op_ent.id,
                                    relation_type=RelationType.TRIGGERS,
                                    weight=1.0,
                                    attributes={"from_template": True},
                                ))
            except Exception:
                pass  # 触发器边创建失败不影响学习主流程
            return eid

    def learn_pattern_sample(
        self,
        input_text: str,
        tokens: List[str],
        pattern: str,
        captured: Dict[str, Any],
    ) -> str:
        """教大脑一个模式样本：把样本作为实体写入图谱。

        用于学习后自检：检查模式匹配是否正确。
        """
        import uuid as _uuid
        eid = f"psample_{_uuid.uuid4().hex[:8]}"
        entity = Entity(
            id=eid,
            name=f"sample_{pattern}",
            aliases=[],
            entity_type=EntityType.ABSTRACT,
            attributes={
                "kind": "pattern_sample",
                "input": input_text,
                "tokens": tokens,
                "pattern": pattern,
                "captured": captured,
            },
        )
        self.declarative.add_entity(entity)
        return eid

    def learn_semantic_link(
        self,
        word_symbol: str,
        op_type: str,
        param_name: str,
        link_type: str = "MAPS_TO",
    ) -> Optional[str]:
        """教大脑一条"词到操作神经元"的语义连接（maps_to 边）。

        Phase 5d.6: 建立 word_symbol 的实体 → operation 神经元 的 MAPS_TO 关系，
        attributes 带 param_name 表示"这个词对应操作的哪个参数来源"。
        例如：learn_semantic_link("有", "write_memory", "entity_value_trigger")
        表示：看到"有"这个词，触发 write_memory 操作。

        当前阶段（5c/5d框架铺垫）：仅存边，不参与激活扩散。

        link_type 支持：MAPS_TO / maps_to / RESOLVES_TO / resolves_to
        """
        lt_upper = link_type.upper()
        if lt_upper not in ("MAPS_TO", "RESOLVES_TO"):
            raise ValueError(
                f"link_type 必须是 MAPS_TO 或 RESOLVES_TO，收到: {link_type}"
            )
        relation_type = (
            RelationType.MAPS_TO if lt_upper == "MAPS_TO" else RelationType.RESOLVES_TO
        )

        word_matches = self.declarative.find_entity_by_name(word_symbol)
        if not word_matches:
            return None
        word_ent = word_matches[0]

        op_ent = None
        op_entities = self.declarative.find_entity_by_name(op_type)
        for e in op_entities:
            if e.attributes.get("kind") == "operation":
                op_ent = e
                break
        if op_ent is None:
            return None

        for r in self.declarative.find_relations_of(word_ent.id, "out"):
            if r.target_id == op_ent.id and r.relation_type == relation_type:
                r.attributes["param_name"] = param_name
                return r.id

        rel = Relation(
            id=self._new_relation_id(),
            source_id=word_ent.id,
            target_id=op_ent.id,
            relation_type=relation_type,
            weight=1.0,
            attributes={"param_name": param_name},
        )
        return self.declarative.add_relation(rel)

    # ============================================================
    # 知识包学习：从可读文件加载知识（人工触发）
    # ============================================================

    PACKAGES_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "data", "knowledge_packages",
    )

    @classmethod
    def list_packages(cls, packages_dir: Optional[str] = None) -> List[str]:
        """列出所有可用知识包名"""
        d = packages_dir or cls.PACKAGES_DIR
        if not os.path.isdir(d):
            return []
        return sorted([
            name for name in os.listdir(d)
            if os.path.isfile(os.path.join(d, name, "package.yaml"))
        ])

    def learn_from_package(self, package_name_or_path: str) -> Dict[str, Any]:
        """从知识包文件学习（人工触发）。

        Args:
            package_name_or_path: 知识包名（如 "1plus1"）或 package.yaml 的完整路径

        知识包格式为 YAML，人类可读，包含：
            numbers:      数字单词及含义
            operators:    操作符单词及含义
            markers:      标记符号（等号、问号等）
            procedures:    程序性记忆（算法）
            tokenizer_examples: 分词样本

        大脑逐条读取并调用 learn_* 接口学习，和老师讲课一样。
        """
        self._ensure_operation_neurons()
        # 解析路径
        if os.path.isfile(package_name_or_path):
            pkg_path = package_name_or_path
        else:
            pkg_path = os.path.join(self.PACKAGES_DIR, package_name_or_path, "package.yaml")
        if not os.path.isfile(pkg_path):
            raise FileNotFoundError(f"知识包不存在: {pkg_path}")

        with open(pkg_path, "r", encoding="utf-8") as f:
            pkg = yaml.safe_load(f) or {}

        stats = {
            "package": pkg.get("name", package_name_or_path),
            "numbers": 0,
            "operators": 0,
            "markers": 0,
            "words": 0,
            "procedures": 0,
            "tokenizer_samples": 0,
            "patterns": 0,        # v2: 模式
            "pattern_samples": 0, # v2: 模式样本
            "semantic_links": 0,  # v2: 词到操作神经元的 maps_to 语义连接
        }

        # 1) 数字单词
        for item in pkg.get("numbers", []):
            self.learn_number_word(
                symbol=item["symbol"],
                value=item["value"],
                aliases=item.get("aliases"),
            )
            stats["numbers"] += 1

        # 2) 操作符单词
        for item in pkg.get("operators", []):
            self.learn_operator_word(
                symbol=item["symbol"],
                op_type=item["op_type"],
                aliases=item.get("aliases"),
            )
            stats["operators"] += 1

        # 3) 标记符号
        for item in pkg.get("markers", []):
            self.learn_marker_word(
                symbol=item["symbol"],
                marker_kind=item["marker_kind"],
                aliases=item.get("aliases"),
            )
            stats["markers"] += 1

        # 4) 普通字词（语文等）
        for item in pkg.get("words", []):
            self.learn_word(
                symbol=item["symbol"],
                meaning=item.get("meaning", ""),
                pinyin=item.get("pinyin", ""),
                word_type=item.get("word_type", ""),
                aliases=item.get("aliases"),
                pos=item.get("pos", ""),
                gender=item.get("gender", ""),
            )
            stats["words"] += 1

        # 5) 程序性记忆
        for item in pkg.get("procedures", []):
            self.learn_procedure(
                name=item["name"],
                algorithm_key=item["algorithm_key"],
                trigger_words=item["trigger_words"],
                steps=item.get("steps"),
                dependencies=item.get("dependencies"),
                description=item.get("description", ""),
            )
            stats["procedures"] += 1

        # 5b) v2 程序性记忆（removing/bind_attribute/resolve_pronoun/retrieve_attr）
        for item in pkg.get("procedures_v2", []):
            self.learn_procedure(
                name=item["name"],
                algorithm_key=item["algorithm_key"],
                trigger_words=item["trigger_words"],
                steps=item.get("steps"),
                dependencies=item.get("dependencies"),
                description=item.get("description", ""),
            )
            stats["procedures"] += 1

        # 6) 分词样本
        examples = pkg.get("tokenizer_examples", [])
        if examples:
            ts = self.learn_tokenizer_dataset(examples)
            stats["tokenizer_samples"] = ts.get("examples", 0)

        # 7) v2: 模式定义
        for item in pkg.get("patterns", []):
            self.learn_pattern(
                name=item["name"],
                slots=item.get("slots", []),
                action=item.get("action", ""),
                priority=item.get("priority", 0.5),
                description=item.get("description", ""),
                dag_template=item.get("dag_template"),
            )
            stats["patterns"] += 1

        # 8) v2: 模式样本
        for item in pkg.get("pattern_samples", []):
            self.learn_pattern_sample(
                input_text=item["input"],
                tokens=item.get("tokens", []),
                pattern=item["pattern"],
                captured=item.get("captured", {}),
            )
            stats["pattern_samples"] += 1

        # 9) v2: 词到操作神经元的 maps_to 语义连接
        for item in pkg.get("semantic_links", []):
            self.learn_semantic_link(
                word_symbol=item["word"],
                op_type=item["op_type"],
                param_name=item.get("param_name", ""),
            )
            stats["semantic_links"] += 1

        self.knowledge_stats = {
            "learned_from_package": stats,
            "declarative_total": self.declarative.entity_count,
            "procedural_total": self.procedural.procedure_count,
        }
        # v2: 学习后自动固化（若启用了持久化目录）
        if self._auto_consolidate and self.storage_dir:
            self.consolidation.consolidate_to_disk(self.storage_dir)
        return self.knowledge_stats

    # ============================================================
    # v2: 记忆固化与启动恢复
    # ============================================================

    def consolidate(self, storage_dir: Optional[str] = None) -> Dict[str, Any]:
        """固化当前知识到硬盘。

        Args:
            storage_dir: 固化目录；None 则用 self.storage_dir
        """
        target = storage_dir or self.storage_dir
        if not target:
            return {"consolidated": False, "reason": "no storage_dir configured"}
        if storage_dir:
            self.storage_dir = storage_dir
            self._auto_consolidate = True
        return self.consolidation.consolidate_to_disk(target)

    def restore(self, storage_dir: Optional[str] = None) -> Dict[str, Any]:
        """从硬盘恢复知识到当前 brain（覆盖现有内存）。

        Args:
            storage_dir: 恢复目录；None 则用 self.storage_dir
        """
        target = storage_dir or self.storage_dir
        if not target:
            return {"restored": False, "reason": "no storage_dir configured"}
        if storage_dir:
            self.storage_dir = storage_dir
            self._auto_consolidate = True
        return self.consolidation.restore_from_disk(target)

    def has_persisted_state(self, storage_dir: Optional[str] = None) -> bool:
        """检查是否已有持久化状态"""
        target = storage_dir or self.storage_dir
        if not target:
            return False
        return MemoryConsolidation.has_saved_state(target)

    # ============================================================
    # 标准入门课程：从零教完 10 以内加减法所需最小知识集合
    # ============================================================

    def teach_default_curriculum(self, max_number: int = 10) -> Dict[str, Any]:
        """标准入门课程（模拟老师一堂一堂讲课，而不是先天注入）。

        授课顺序：
          1. 数字 0 ~ max_number（含中文数字别名、2 的别名"两"）
          2. 后继关系 0→1→2→...→(max_number)
          3. 操作符 + - * /
          4. 等号 = 、问号 ?
          5. 程序性记忆：counting, mapping, merging, adding, subtracting
          6. 几个分词样本（让 tokenizer 学会"等于/多少"这种多字词 + 数字合并）

        Returns:
            授课统计：{numbers: int, relations: int, operators: int, markers: int,
                        procedures: int, tokenizer_samples: int}
        """
        self._ensure_operation_neurons()
        stats = {
            "numbers": 0,
            "relations": 0,
            "operators": 0,
            "markers": 0,
            "procedures": 0,
            "tokenizer_samples": 0,
        }
        cn = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]
        prev_symbol: Optional[str] = None

        # 1) 数字
        for v in range(0, max_number + 1):
            aliases = []
            if 0 <= v < len(cn):
                aliases.append(cn[v])
            if v == 2 and "两" not in aliases:
                aliases.append("两")
            self.learn_number_word(str(v), v, aliases=aliases)
            stats["numbers"] += 1
            # 2) 后继关系（讲完 n+1 之后，"n 的后面是 n+1"）
            if prev_symbol is not None:
                self.learn_relation(prev_symbol, str(v), RelationType.SUCCESSOR, attrs={"delta": 1})
                stats["relations"] += 1
            prev_symbol = str(v)

        # 3) 操作符
        operators: List[Tuple[str, str, List[str]]] = [
            ("+", "add", ["加", "加上", "＋"]),
            ("-", "sub", ["减", "减去", "－"]),
            ("*", "mul", ["乘", "乘以", "×"]),
            ("/", "div", ["除", "除以", "÷"]),
        ]
        for sym, op_type, aliases in operators:
            self.learn_operator_word(sym, op_type, aliases=aliases)
            stats["operators"] += 1

        # 4) 等号、问号
        self.learn_marker_word("=", "equality", aliases=["等于", "＝", "是", "得"])
        stats["markers"] += 1
        self.learn_marker_word("?", "question", aliases=["？", "多少", "几", "什么"])
        stats["markers"] += 1

        # 5) 程序性记忆
        counting_proc_id = self.learn_procedure(
            "counting", "counting",
            trigger_words=["多少", "几", "数", "count"],
            description="计数算法：对集合或数字逐个数",
        )
        mapping_proc_id = self.learn_procedure(
            "mapping", "mapping",
            trigger_words=["映射", "手指", "具身", "map"],
            description="数字 <-> 集合的具身映射",
        )
        merging_proc_id = self.learn_procedure(
            "merging", "merging",
            trigger_words=["合并", "放一起", "merge"],
            description="集合 A + 集合 B 合成一个集合",
        )
        stats["procedures"] += 3
        self.learn_procedure(
            "adding", "adding",
            trigger_words=["+", "加", "加上", "add", "＋"],
            steps=[
                {"action": "mapping", "parameters": {"operand": "a"}, "description": "把 a 映射为集合 A"},
                {"action": "mapping", "parameters": {"operand": "b"}, "description": "把 b 映射为集合 B"},
                {"action": "merging", "parameters": {"sets": ["A", "B"]}, "description": "把 A、B 合并成一个集合"},
                {"action": "counting", "parameters": {"set": "merged"}, "description": "数合并后集合的个数 = a+b"},
            ],
            dependencies=["mapping", "mapping", "merging", "counting"],
            description="a+b = count( merge( map(a), map(b) ) )",
        )
        self.learn_procedure(
            "subtracting", "subtracting",
            trigger_words=["-", "减", "减去", "sub", "－"],
            steps=[
                {"action": "mapping", "parameters": {"operand": "a"}, "description": "把 a 映射为集合 A"},
                {"action": "remove", "parameters": {"count": "b"}, "description": "从 A 中拿开 b 个元素"},
                {"action": "counting", "parameters": {"set": "remaining"}, "description": "数剩余集合 = a-b"},
            ],
            dependencies=["mapping", "counting"],
            description="a-b = 从 a 的集合里拿走 b 个后计数",
        )
        stats["procedures"] += 2

        # 6) 分词句子样本：教 tokenizer 认识"等于/多少"这种两字词，以及"数字合并"的结构
        token_samples = [
            {"input": "1+1=?",        "expected_tokens": ["1", "+", "1", "=", "?"]},
            {"input": "一加一等于多少", "expected_tokens": ["一", "加", "一", "等于", "多少"]},
            {"input": "12+34",        "expected_tokens": ["12", "+", "34"]},
        ]
        ts = self.learn_tokenizer_dataset(token_samples)
        stats["tokenizer_samples"] = ts.get("examples", 0)

        self.knowledge_stats = {
            "taught_by_curriculum": stats,
            "declarative_total": self.declarative.entity_count,
            "relations_total": self.declarative.relation_count,
            "procedural_total": self.procedural.procedure_count,
        }
        return self.knowledge_stats

    # ============================================================
    # 主求解
    # ============================================================

    def solve(self, text: str) -> BrainResult:
        tokens = self.tokenizer.tokenize(text)
        self.workspace.receive_input(text, tokens=tokens)
        activated = self.workspace.activate()
        for a in activated:
            if a.entity:
                self.consolidation.record_activation(a.entity.id)
            if a.procedure:
                self.consolidation.record_activation(a.procedure.id)
        out = self.reasoning.run(self.workspace)
        # 提取问题理解 DAG 文本（从 dag_build 步骤的 outputs.dag_text 中取）
        dag_text = ""
        for s in out.reasoning_chain:
            if s.action == "dag_build" and s.outputs.get("dag_text"):
                dag_text = s.outputs["dag_text"]
                break
        return BrainResult(
            answer=out.final_answer,
            confidence=out.confidence,
            reasoning_chain=[s.dict() for s in out.reasoning_chain],
            tokens=tokens,
            activated_count=len(activated),
            raw_text=text,
            dag_text=dag_text,
        )

    def quick_ask(self, text: str, show_chain: bool = True) -> Any:
        result = self.solve(text)
        if show_chain:
            print(result.format())
        else:
            print(f"Q: {text}\nA: {result.answer}")
        return result.answer

    # ============================================================
    # 持久化
    # ============================================================

    def save_knowledge(self, data_dir: Optional[str] = None) -> None:
        d = data_dir or self.config.get("knowledge", {}).get("persist_path", "digital_brain/data/knowledge/")
        os.makedirs(d, exist_ok=True)
        self.declarative.save_json(
            os.path.join(d, "entities.json"),
            os.path.join(d, "relations.json"),
        )
        self.procedural.save_json(os.path.join(d, "procedures.json"))

    def load_knowledge(self, data_dir: Optional[str] = None) -> None:
        d = data_dir or self.config.get("knowledge", {}).get("persist_path", "digital_brain/data/knowledge/")
        self.declarative = DeclarativeMemory()
        self.procedural = ProceduralMemory()
        self.declarative.load_json(
            os.path.join(d, "entities.json"),
            os.path.join(d, "relations.json"),
        )
        self.procedural.load_json(os.path.join(d, "procedures.json"))

    # ============================================================
    # 内部工具
    # ============================================================

    @staticmethod
    def _load_config(path: str) -> dict:
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}

    # ---- ID 生成（教学过程中确定性）----
    def _new_entity_id(self, tag: str) -> str:
        return f"ent_{tag}_{uuid.uuid4().hex[:6]}"

    def _new_relation_id(self) -> str:
        return f"rel_{uuid.uuid4().hex[:8]}"

    def _new_procedure_id(self, name: str) -> str:
        return f"proc_{name}_{uuid.uuid4().hex[:6]}"

    # ---- entity 解析：可以是 id 也可以是 symbol ----
    def _resolve_entity(self, symbol_or_id: str) -> Optional[Entity]:
        e = self.declarative.get_entity(symbol_or_id)
        if e is not None:
            return e
        matches = self.declarative.find_entity_by_name(symbol_or_id)
        return matches[0] if matches else None

    # ---- 先天 operation 神经元入图谱（L0 硬件指令集）----
    def _ensure_operation_neurons(self) -> None:
        operations = [
            ("ent_op_write_memory", "write_memory"),
            ("ent_op_read_memory", "read_memory"),
            ("ent_op_search_context", "search_context"),
            ("ent_op_call_algorithm", "call_algorithm"),
            ("ent_op_return_value", "return_value"),
        ]
        try:
            for eid, op_type in operations:
                if self.declarative.get_entity(eid) is None:
                    entity = Entity(
                        id=eid,
                        name=op_type,
                        aliases=[],
                        entity_type=EntityType.ABSTRACT,
                        attributes={"kind": "operation", "op_type": op_type},
                    )
                    try:
                        self.declarative.add_entity(entity)
                    except ValueError:
                        pass
        except Exception:
            pass

    # ---- 程序性步骤默认模板 ----
    @staticmethod
    def _default_steps_for_algorithm(algorithm_key: str) -> List[OperationStep]:
        if algorithm_key == "adding":
            return [
                OperationStep(order=1, action="mapping", parameters={"operand": "a"}, description="a -> set A"),
                OperationStep(order=2, action="mapping", parameters={"operand": "b"}, description="b -> set B"),
                OperationStep(order=3, action="merging", parameters={"sets": ["A", "B"]}, description="A + B"),
                OperationStep(order=4, action="counting", parameters={"set": "merged"}, description="count result"),
            ]
        if algorithm_key == "subtracting":
            return [
                OperationStep(order=1, action="mapping", parameters={"operand": "a"}, description="a -> set A"),
                OperationStep(order=2, action="remove",  parameters={"count": "b"}, description="remove b from A"),
                OperationStep(order=3, action="counting", parameters={"set": "remaining"}, description="count result"),
            ]
        if algorithm_key == "counting":
            return [OperationStep(order=1, action="count", description="iterate & count")]
        if algorithm_key == "mapping":
            return [OperationStep(order=1, action="map", description="number <-> set")]
        if algorithm_key == "merging":
            return [OperationStep(order=1, action="merge", description="concat sets")]
        return []
