"""工作记忆 - 任务级临时状态，DAG节点间数据传递的唯一通道

对应详细设计/工作区设计.md §3：
    - context_entities: 上下文实体引用
    - attribute_bindings: 临时属性绑定（entity.attr = value）
    - resolved_references: 代词消解结果（pronoun -> [entities]）
    - intermediate_results: DAG节点中间结果（node_id.out -> value）

生命周期：任务开始创建，任务结束清空，不污染长期记忆。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

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
        # Phase M2: 语义栈（跨句零指代宾语/主语/位置回退）
        self.agent_stack: List[str] = []       # 最近主语（施动者）栈，LIFO，top=最近
        self.theme_stack: List[str] = []       # 最近宾语（被操作物）栈
        self.location_stack: List[str] = []    # 最近位置/容器栈
        # Phase M2: 代词→候选实体得分排名缓存（供 search_context 输出有序结果）
        self._pronoun_candidate_scores: Dict[str, List[Tuple[str, float]]] = {}

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

    def write_attr(self, entity: str, attr: str, value: Any,
                   declarative_memory: Optional[Any] = None) -> None:
        """写入临时属性绑定。同属性多次写入，后写覆盖先写。

        Phase M2 新增：同步更新 agent/theme/location 语义栈
            - entity 若 pos == person_name → 入 agent_stack（去重置顶）
            - attr   若 pos == noun 或 未标 pos 但非功能词性 → 入 theme_stack（去重置顶）
            - 显式传 location 时入 location_stack（当前先留空，M3 SRL 再启用）
        declarative_memory 用来查 pos 属性，若为 None 则用兜底规则（person_name 类实体仍然准确，
        因为通常 entity 本身就是 person_name）。
        """
        key = self._attr_key(entity, attr)
        self.attribute_bindings[key] = value
        # 登记上下文实体（兼容旧行为）
        if entity not in self.context_entities:
            self._entity_write_order.append(entity)
            # 创建一个轻量实体占位（DAG写入时实体已在陈述记忆中）
            self.context_entities[entity] = Entity(
                id=f"wm_{entity}", name=entity, attributes={"kind": "word"}
            )
        # 跨句追踪登记
        self.register_entity_mention(entity)
        self.register_entity_mention(attr)

        # ---------- Phase M2: 语义栈同步 ----------
        def _pos_of(name: str) -> Optional[str]:
            if declarative_memory is None:
                return None
            matches = declarative_memory.find_entity_by_name(name)
            for m in matches:
                attr = getattr(m, "attributes", None) or {}
                if "pos" in attr:
                    return attr["pos"]
            return None

        # agent_stack：entity 是 person_name 类 → 去重置顶（出现新的就提到栈顶）
        e_pos = _pos_of(entity)
        # 兼容：未标 pos 但看起来像人名的词（M1 加的 person_name 词汇都会有 pos）
        is_agent_like = (e_pos == "person_name") or (
            e_pos is None and 1 < len(entity) <= 3 and entity not in ("故事书", "铅笔", "书包")
        )
        if is_agent_like:
            self._push_stack_top(self.agent_stack, entity)

        # theme_stack：attr 是 noun 类（故事书/铅笔/年龄等）或未标 pos 的业务属性
        a_pos = _pos_of(attr)
        NON_THEME_POS = {
            "adv_temporal", "adv_total", "adv_accumulate",
            "part_aspect", "prep_locative", "prep_dative",
            "question_marker", "classifier", "discourse_marker",
        }
        if a_pos is None and attr:
            # 未标 pos：兜底认为非功能词性即 noun
            is_noun_like = True
        else:
            is_noun_like = a_pos == "noun"
        if a_pos not in NON_THEME_POS and attr:
            self._push_stack_top(self.theme_stack, attr)

    def read_attr(self, entity: str, attr: str) -> Any:
        """读取临时属性绑定。未绑定则抛 KeyError。"""
        key = self._attr_key(entity, attr)
        if key not in self.attribute_bindings:
            raise KeyError(f"属性未绑定: {key}")
        return self.attribute_bindings[key]

    def read_attr_or_none(self, entity: str, attr: str) -> Any:
        return self.attribute_bindings.get(self._attr_key(entity, attr))

    # ---- M2 语义栈辅助 ----
    def _push_stack_top(self, stack: List[str], value: str) -> None:
        """把 value 压到语义栈的顶部（最近位置）；已存在则先移除原来的位置再置顶。"""
        if not value:
            return
        if value in stack:
            stack.remove(value)
        stack.append(value)
        # 最多保留最近 10 条，避免无限增长
        if len(stack) > 10:
            del stack[:-10]

    def top_agent(self, default: Optional[str] = None) -> Optional[str]:
        return self.agent_stack[-1] if self.agent_stack else default

    def top_theme(self, default: Optional[str] = None) -> Optional[str]:
        return self.theme_stack[-1] if self.theme_stack else default

    def top_location(self, default: Optional[str] = None) -> Optional[str]:
        return self.location_stack[-1] if self.location_stack else default

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
    def resolve_pronoun(self, pronoun: str,
                        declarative_memory: Optional[Any] = None,
                        require_gender: Optional[str] = None) -> List[str]:
        """依据上下文把代词解析为实体名列表。

        Phase M2 升级：
            1. 分类代词（他=男她=女它=物/复数）→ require_gender 过滤
            2. 距离衰减：recent_entities 越近分越高（索引越大分越高）
            3. 候选排序：按 (gender_match? * 大权重 + 距离分) 排序
            4. 结果缓存到 resolved_references，下次直接命中
            5. 每对 (pronoun, candidate) 记一次得分，供 debug。

        Args:
            pronoun: 代词（"他/她/它/他们/..."）或具体名词
            declarative_memory: 陈述记忆，用来查 pos/gender
            require_gender: 外部强制性别，None 则从 pronoun 本身推导 ∈ {M, F, N, PLURAL, None}
        """
        # 缓存命中
        cached = self.resolved_references.get(pronoun)
        if cached is not None:
            return cached

        # ---- M2: 代词 -> 性别/数 推导 ----
        pron_to_gender = {
            "他": "M", "他们": "PLURAL_MALE",
            "她": "F", "她们": "PLURAL_FEMALE",
            "它": "N", "它们": "PLURAL_NEUTER",
            "她们": "PLURAL_FEMALE",
            "大家": "PLURAL", "这些": "PLURAL", "那些": "PLURAL",
            "这": "NEAR", "那": "FAR", "其": "NEAR",
        }
        plural_prons = {"他们", "她们", "它们", "大家", "这些", "那些"}
        singular_prons = {"他", "她", "它", "这", "那", "其"}
        pronoun_vocab = set(pron_to_gender) | plural_prons | singular_prons

        # 具体名词/人名（哥哥/弟弟/小明/妈妈…）不是代词：直接返回其本身。
        # （M2 回归：不可用"最近提及"排序，否则 search_context("哥哥") 会错解成"弟弟"）
        if pronoun not in pronoun_vocab:
            result = [pronoun] if self.has_context(pronoun) else [pronoun]
            self.resolved_references[pronoun] = result
            return result

        if require_gender is None:
            require_gender = pron_to_gender.get(pronoun)

        is_plural = pronoun in plural_prons or (
            isinstance(require_gender, str) and require_gender.startswith("PLURAL")
        )

        # ---- M2: 候选池 = recent_entities（优先）+ 所有 context entity 名 ----
        recent_names = list(self.recent_entities)
        all_names = self.all_context_entity_names()
        # 合并：recent 优先，去重保留顺序；同时过滤：只有在 context_entities 中的才算"真正实体"
        # （排除属性名等被 register_entity_mention(attr) 登记但不是实体的条目）
        merged: List[str] = []
        seen = set()
        for n in recent_names + all_names:
            if n in seen:
                continue
            if n not in self.context_entities:
                continue
            seen.add(n)
            merged.append(n)

        # ---- M2: 性别 + 距离打分 ----
        def _gender_of(name: str) -> Optional[str]:
            if declarative_memory is None:
                return None
            ents = declarative_memory.find_entity_by_name(name)
            for m in ents:
                g = (getattr(m, "attributes", None) or {}).get("gender")
                if g:
                    return g
            return None

        scored: List[Tuple[str, float]] = []
        for idx, name in enumerate(merged):
            # 距离分：越近（idx 越大）分越高，0~1 归一
            dist_score = idx / max(1, len(merged) - 1) if len(merged) > 1 else 1.0
            # 性别匹配分
            gender_score = 0.0
            if require_gender == "M":
                # 单数"他"：F 直接排除（妈妈不可能是"他"）
                g = _gender_of(name)
                if g == "M":
                    gender_score = 2.0
                elif g == "F":
                    gender_score = -10.0
                else:
                    gender_score = -0.2
            elif require_gender == "PLURAL_MALE":
                # 复数"他们"：可指代混合群体，M 优先但 F 不排除
                g = _gender_of(name)
                if g == "M":
                    gender_score = 2.0
                elif g == "F":
                    gender_score = -0.2
                else:
                    gender_score = -0.2
            elif require_gender == "F":
                # 单数"她"：M 直接排除
                g = _gender_of(name)
                if g == "F":
                    gender_score = 2.0
                elif g == "M":
                    gender_score = -10.0
                else:
                    gender_score = -0.2
            elif require_gender == "PLURAL_FEMALE":
                # 复数"她们"：F 优先但 M 不排除（极少数混合情况）
                g = _gender_of(name)
                if g == "F":
                    gender_score = 2.0
                elif g == "M":
                    gender_score = -0.2
                else:
                    gender_score = -0.2
            elif require_gender == "N":
                g = _gender_of(name)
                if g == "N":
                    gender_score = 2.0
                elif g in ("M", "F"):
                    gender_score = -10.0
            total = dist_score + gender_score
            if total < -5:  # 明显性别不匹配，踢掉
                continue
            scored.append((name, total))

        # 按总分降序，同分按 dist_score 降序
        scored.sort(key=lambda x: x[1], reverse=True)
        # 缓存候选得分表（调试用）
        self._pronoun_candidate_scores[pronoun] = scored

        if not scored:
            # fallback：如果没有候选，按原来的复数/单数规则
            if is_plural:
                result = all_names
            elif pronoun in singular_prons:
                result = all_names[-1:] if all_names else []
            else:
                result = [pronoun] if self.has_context(pronoun) else [pronoun]
            self.resolved_references[pronoun] = result
            return result

        names = [n for (n, _) in scored]
        if is_plural:
            # 复数：保持原始写入顺序（all_names 中的先后），不按距离衰减排序
            # 加减法等对参数顺序敏感，需与原有 MVP 行为一致
            scored_name_set = set(names)
            ordered_names = [n for n in all_names if n in scored_name_set]
            # 补进 filtered 后在 all_names 中不存在的（理论上不会发生）
            for n in names:
                if n not in ordered_names:
                    ordered_names.append(n)
            result = ordered_names
        else:
            result = names[:1]  # 单数只返回 top-1

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
        self.agent_stack.clear()
        self.theme_stack.clear()
        self.location_stack.clear()
        self._pronoun_candidate_scores.clear()

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
