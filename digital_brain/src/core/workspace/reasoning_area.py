"""推理操作区 - 协调模式匹配、意图识别和程序性记忆/算法的执行

v2 变更（Phase 3）：
    - 新增 DAG 推理路径：问题→DAG构建（理解）→DAG执行（求解）
    - 新增原子操作集：read_memory/write_memory/search_context/call_algorithm/return_value
    - 新增 DAG 构建器：模式匹配结果→问题DAG
    - 新增 DAG 执行引擎：拓扑排序→逐节点执行→工作记忆传数据
    - 保留旧单步推理路径作为回退（无 build_dag 模式命中时使用）

将推理链写入 OutputBuffer。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from digital_brain.src.algorithms.adding import AddingAlgorithm
from digital_brain.src.algorithms.counting import CountingAlgorithm
from digital_brain.src.algorithms.mapping import MappingAlgorithm
from digital_brain.src.algorithms.merging import MergingAlgorithm
from digital_brain.src.algorithms.subtracting import SubtractingAlgorithm
from digital_brain.src.core.memory.procedural_memory import ProceduralMemory
from digital_brain.src.core.models import Procedure
from digital_brain.src.core.pattern.intent_recognizer import Intent, IntentRecognizer, IntentType
from digital_brain.src.core.pattern.pattern_matcher import PatternMatchResult, PatternMatcher
from digital_brain.src.core.workspace.dag import (
    ATOMIC_OPS,
    DAGBuildResult,
    DAGGraph,
    DAGNode,
    OP_CALL_ALGORITHM,
    OP_READ_MEMORY,
    OP_RETURN_VALUE,
    OP_SEARCH_CONTEXT,
    OP_WRITE_MEMORY,
)
from digital_brain.src.core.workspace.output_buffer import OutputBuffer
from digital_brain.src.core.workspace.workspace import Workspace


class AlgorithmRegistry:
    """算法注册表 - 将 procedure.name/algorithm name 映射到可执行对象"""

    def __init__(self) -> None:
        self.counting = CountingAlgorithm()
        self.mapping = MappingAlgorithm()
        self.merging = MergingAlgorithm()
        self.adding = AddingAlgorithm(self.mapping, self.merging, self.counting)
        self.subtracting = SubtractingAlgorithm(self.mapping, self.counting)

    def all_algorithms(self) -> Dict[str, Any]:
        return {
            "counting": self.counting,
            "mapping": self.mapping,
            "merging": self.merging,
            "adding": self.adding,
            "subtracting": self.subtracting,
        }


# ============================================================
# DAG 构建器
# ============================================================

# op_type → algorithm_key 映射（引擎命名约定，非领域知识）
_OP_TYPE_TO_ALGORITHM: Dict[str, str] = {
    "add": "adding",
    "sub": "subtracting",
    "mul": "multiplying",
    "div": "dividing",
}

# v2 算法 key 集合（原子操作的包装，允许 learn_procedure 注册）
_V2_ALGORITHM_KEYS = frozenset({
    "removing", "bind_attribute", "resolve_pronoun", "retrieve_attr",
})


class _TemplateResolveError(Exception):
    """模板解析失败时内部抛出，用于中断实例化"""
    pass


class DAGBuilder:
    """DAG 构建器 - 把模式匹配结果翻译为问题DAG

    构建成功 = 系统理解了问题。
    构建失败 = 不理解，返回结构化缺失清单。

    模板由模式实体的 action 字段指定（如 build_dag:binary_op），
    构建器按 action 类型实例化对应的 DAG 节点骨架。
    """

    def __init__(
        self,
        declarative_memory: Optional[Any] = None,
        procedural_memory: Optional[ProceduralMemory] = None,
        working_memory: Optional[Any] = None,
    ) -> None:
        self.declarative = declarative_memory
        self.procedural = procedural_memory
        self.working_memory = working_memory
        self._node_counter = 0

    def _new_node_id(self, prefix: str = "n") -> str:
        self._node_counter += 1
        return f"{prefix}{self._node_counter}"

    def build(self, matches: List[PatternMatchResult]) -> DAGBuildResult:
        """根据模式匹配结果构建问题DAG。

        三道闸门：
        1. 节点完备：每个 action 的输入都有来源
        2. 无环：图是真正的 DAG
        3. 算法已学：call_algorithm 引用的 key 已注册
        """
        self._node_counter = 0
        dag = DAGGraph()
        missing: List[str] = []

        # 按 span 排序，确保 write_memory 在 search_context 之前
        sorted_matches = sorted(matches, key=lambda m: (m.span[0], -m.match_score))

        write_node_ids: List[str] = []  # 所有 write_memory 节点（供 search_context 依赖）

        for match in sorted_matches:
            action = match.action or ""
            if not action.startswith("build_dag:"):
                continue
            template = action.split(":", 1)[1]
            self._build_template(template, match, dag, missing, write_node_ids)

        # 闸门1：节点完备
        if missing:
            return DAGBuildResult(
                success=False, dag=None,
                failure_type="missing_dependency",
                failure_reason="缺少依赖：" + "、".join(missing),
                missing=missing,
                node_count=dag.node_count,
                pattern_count=len(sorted_matches),
            )

        # 闸门2：无环
        if dag.has_cycle():
            return DAGBuildResult(
                success=False, dag=None,
                failure_type="cycle",
                failure_reason="问题自相矛盾，DAG存在环",
                node_count=dag.node_count,
                pattern_count=len(sorted_matches),
            )

        # 闸门3：算法已学
        unlearned = self._check_algorithms_learned(dag)
        if unlearned:
            return DAGBuildResult(
                success=False, dag=None,
                failure_type="algorithm_not_learned",
                failure_reason="未学算法：" + "、".join(unlearned),
                missing=unlearned,
                node_count=dag.node_count,
                pattern_count=len(sorted_matches),
            )

        if dag.node_count == 0:
            return DAGBuildResult(
                success=False, dag=None,
                failure_type="no_pattern",
                failure_reason="没有命中可构建DAG的模式",
                node_count=0,
                pattern_count=len(sorted_matches),
            )

        return DAGBuildResult(
            success=True, dag=dag,
            node_count=dag.node_count,
            pattern_count=len(sorted_matches),
        )

    # ---- 模板构建 ----

    def _build_template(
        self,
        template: str,
        match: PatternMatchResult,
        dag: DAGGraph,
        missing: List[str],
        write_node_ids: List[str],
    ) -> None:
        captured = match.captured

        # Phase 5b: 尝试从 pattern 实体的 dag_template 属性加载
        dag_tpl = self._lookup_dag_template(match.pattern_name)
        if dag_tpl is not None:
            self._instantiate_from_template(dag_tpl, captured, dag, missing, write_node_ids)
            return

        # Fallback: 旧的 elif 链（保持完整向后兼容）
        if template == "binary_op":
            self._build_binary_op(captured, dag, missing)
        elif template == "attr_value":
            self._build_attr_value(captured, dag, write_node_ids)
        elif template == "pronoun_sum":
            self._build_pronoun_aggregate(captured, dag, write_node_ids, "adding", "之和")
        elif template == "pronoun_diff":
            self._build_pronoun_aggregate(captured, dag, write_node_ids, "subtracting", "之差")
        # 未知模板忽略

    def _lookup_dag_template(self, pattern_name: str) -> Optional[Dict[str, Any]]:
        """从 declarative memory 查 pattern 实体的 dag_template 属性"""
        if self.declarative is None:
            return None
        # pattern 实体 id 约定为 f"pattern_{pattern_name}"（见 learn_pattern 实现）
        eid = f"pattern_{pattern_name}"
        entity = self.declarative.get_entity(eid)
        if entity is None:
            # fallback: 按 name 查询
            matches = self.declarative.find_entity_by_name(pattern_name)
            for m in matches:
                if m.attributes.get("kind") == "pattern":
                    entity = m
                    break
        if entity is None:
            return None
        return entity.attributes.get("dag_template")

    def _instantiate_from_template(
        self,
        dag_tpl: Dict[str, Any],
        captured: Dict[str, Any],
        dag: DAGGraph,
        missing: List[str],
        write_node_ids: List[str],
    ) -> None:
        """从 dag_template 实例化 DAG 节点

        dag_template 结构：
        {
          "nodes": [
            {
              "tpl_id": "call",
              "action": "call_algorithm",
              "params_spec": { ... },
              "description_tpl": "...",
              "depends_on_all_write_memories": False,
              "depends_on_tpl": [...],
            },
            ...
          ]
        }
        """
        nodes_tpl = dag_tpl.get("nodes", [])
        if not nodes_tpl:
            return

        # Step 1: 先扫描一遍，解析所有 params 中的 captured 引用和 resolve，确定值
        # 保存 tpl_id -> (resolved_params_dict, has_error)
        resolved_params: Dict[str, Dict[str, Any]] = {}
        tpl_to_node_ids: Dict[str, str] = {}

        # First pass: resolve all params that reference captured values (not node_out yet)
        for node_tpl in nodes_tpl:
            tpl_id = node_tpl.get("tpl_id")
            if not tpl_id:
                continue
            params_spec = node_tpl.get("params_spec", {})
            try:
                resolved = self._resolve_params_spec(params_spec, captured, missing, dag=dag)
            except _TemplateResolveError:
                return
            resolved_params[tpl_id] = resolved
            tpl_to_node_ids[tpl_id] = self._new_node_id()

        # Step 2: 第二次扫描，处理 node_out 引用（此时所有 tpl_id->actual_id 映射已就绪）
        # 然后创建 DAGNode
        for node_tpl in nodes_tpl:
            tpl_id = node_tpl.get("tpl_id")
            if not tpl_id or tpl_id not in tpl_to_node_ids:
                continue

            actual_id = tpl_to_node_ids[tpl_id]
            action = node_tpl.get("action", "")
            params = resolved_params.get(tpl_id, {})

            # 二次替换 params 中的 node_out 引用
            params = self._replace_node_out_refs(params, tpl_to_node_ids)

            # 构建 depends_on
            depends_on: List[str] = []
            depends_on_tpl = node_tpl.get("depends_on_tpl", [])
            for dep_tpl in depends_on_tpl:
                if dep_tpl in tpl_to_node_ids:
                    depends_on.append(tpl_to_node_ids[dep_tpl])
            if node_tpl.get("depends_on_all_write_memories", False):
                depends_on.extend(list(write_node_ids))

            # 构建 description
            description_tpl = node_tpl.get("description_tpl")
            if description_tpl:
                try:
                    description = description_tpl.format(**params)
                except Exception:
                    description = description_tpl
            else:
                # 简化描述
                param_strs = [f"{k}={v}" for k, v in params.items()]
                description = f"{action}(" + ", ".join(param_strs) + ")"

            # 创建节点
            dag.add_node(DAGNode(
                id=actual_id,
                action=action,
                params=params,
                depends_on=depends_on,
                description=description,
            ))

            # 若是 write_memory，登记到 write_node_ids
            if action == OP_WRITE_MEMORY:
                write_node_ids.append(actual_id)

    # ---------- M1-R3 object 清洗常量 ----------
    OBJECT_BAD_POS = {
        "adv_temporal", "adv_total", "adv_accumulate",
        "part_aspect", "prep_locative", "prep_dative",
        "question_marker", "classifier",
    }

    def _get_word_pos(self, word: str) -> Optional[str]:
        dm = getattr(self, "declarative", None)
        if dm is None:
            wm = getattr(self, "workspace", None)
            dm = getattr(wm, "declarative", None) if wm else None
        if dm is None or not word:
            return None
        ents = dm.find_entity_by_name(word)
        if not ents:
            return None
        attr = getattr(ents[0], "attributes", None) or {}
        return attr.get("pos")

    def _is_invalid_object_candidate(self, raw: Any) -> bool:
        if raw is None:
            return True
        if not isinstance(raw, str):
            return False
        if raw in ("?", "。", "，", ",", "！", "!", "."):
            return True
        if len(raw) == 1:
            c = raw
            is_cjk = ('\u4e00' <= c <= '\u9fff') or ('\u3400' <= c <= '\u4dbf')
            if not (c.isalpha() or c.isdigit() or is_cjk):
                return True
        pos = self._get_word_pos(raw)
        if pos in self.OBJECT_BAD_POS:
            return True
        return False

    def _find_last_write_attr(self, dag: Any) -> Optional[str]:
        """返回当前 DAG 中最后一个 write_memory 节点的 attr 参数（零指代宾语回退用）。"""
        try:
            write_op = OP_WRITE_MEMORY
        except Exception:
            write_op = "write_memory"
        last_attr = None
        nodes_dict = getattr(dag, "nodes", {})
        nodes_list = list(nodes_dict.values()) if isinstance(nodes_dict, dict) else list(nodes_dict)
        for node in nodes_list:
            action = getattr(node, "action", None)
            params = getattr(node, "params", None) or {}
            if action == write_op and isinstance(params, dict) and "attr" in params:
                last_attr = params["attr"]
        return last_attr

    def _resolve_params_spec(
        self,
        params_spec: Dict[str, Any],
        captured: Dict[str, Any],
        missing: List[str],
        dag: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """递归解析 params_spec，处理 resolve / from_captured / node_out 等规则

        node_out 引用此时先保留为占位符，之后二次扫描再替换。
        """
        result: Dict[str, Any] = {}
        for key, spec in params_spec.items():
            result[key] = self._resolve_single_spec(spec, captured, missing, dag=dag)
        return result

    def _resolve_single_spec(
        self,
        spec: Any,
        captured: Dict[str, Any],
        missing: List[str],
        dag: Optional[Any] = None,
    ) -> Any:
        """解析单个参数规范值

        支持：
        - dict with "resolve" + "from_captured": 使用解析器
        - dict with only "from_captured": 直接取 captured 值
        - dict with "node_out": 保留为占位符 (标记 __NODE_OUT__:{tpl_id})
        - list/dict: 递归处理元素
        - 其他: 字面量直接返回
        """
        if isinstance(spec, dict):
            if "node_out" in spec:
                # 保留占位符，二次扫描时替换
                return f"__NODE_OUT__:{spec['node_out']}"
            if "node_ref" in spec:
                # 引用模板中的另一个节点 tpl_id（用于 collect_from.node 等）
                # 保留占位符，二次扫描时替换为实际 id（非 $ 前缀，直接是 id 字符串）
                return f"__NODE_REF__:{spec['node_ref']}"
            if "node_ref_first" in spec:
                # 取目标节点输出列表的第一个元素（如果是列表）；否则直接取输出值本身
                # 先存占位符，在 instantiate_from_template 中 DAG 节点创建之后再处理
                # 这里是第一阶段（params_spec 解析），实际节点输出还不存在。
                # 所以把它和 node_out 一样先占位，之后在第二阶段和 DAG 执行前做替换。
                return f"__NODE_REF_FIRST__:{spec['node_ref_first']}"
            if "literal" in spec:
                # 字面量直接返回其值（{"literal": "adding"} → "adding"）
                return spec["literal"]
            if "from_captured" in spec:
                from_cap = spec["from_captured"]
                raw = captured.get(from_cap)
                if from_cap == "object" and self._is_invalid_object_candidate(raw):
                    top_theme = None
                    if self.working_memory is not None:
                        top_theme = getattr(self.working_memory, "top_theme", lambda d=None: d)()
                    if top_theme:
                        raw = top_theme
                    elif dag is not None:
                        last_attr = self._find_last_write_attr(dag)
                        if last_attr is not None:
                            raw = last_attr
                if raw is None:
                    missing.append(f"模式捕获字段'{from_cap}'缺失（且无可用回退）")
                    raise _TemplateResolveError()
                resolve = spec.get("resolve")
                if resolve == "number_value":
                    val = self._resolve_number(str(raw))
                    if val is None:
                        missing.append(f"数字'{raw}'未学")
                        raise _TemplateResolveError()
                    return val
                if resolve == "op_algorithm":
                    val = self._resolve_op_algorithm(str(raw))
                    if val is None:
                        missing.append(f"操作符'{raw}'未学")
                        raise _TemplateResolveError()
                    return val
                # 无 resolve: 直接返回 captured 值
                return raw
            # 普通 dict，递归处理每个 value
            return {k: self._resolve_single_spec(v, captured, missing, dag=dag) for k, v in spec.items()}
        if isinstance(spec, list):
            return [self._resolve_single_spec(item, captured, missing, dag=dag) for item in spec]
        # 字面量
        return spec

    def _replace_node_out_refs(
        self,
        params: Any,
        tpl_to_node_ids: Dict[str, str],
    ) -> Any:
        """替换 params 中的占位符：
        - __NODE_OUT__:{tpl_id} → $actual_node_id（用于 return_value.value 等输出引用）
        - __NODE_REF__:{tpl_id} → actual_node_id（用于 collect_from.node 等 ID 引用）
        """
        if isinstance(params, str):
            if params.startswith("__NODE_OUT__:"):
                tpl_id = params[len("__NODE_OUT__:"):]
                actual_id = tpl_to_node_ids.get(tpl_id, tpl_id)
                return f"${actual_id}"
            if params.startswith("__NODE_REF__:"):
                tpl_id = params[len("__NODE_REF__:"):]
                return tpl_to_node_ids.get(tpl_id, tpl_id)
            if params.startswith("__NODE_REF_FIRST__:"):
                # 这个值在 DAG 节点创建时尚无法确定（因为依赖上游节点的运行时输出），
                # 所以把它保留为一个特殊标记 $FIRST:{actual_node_id}，在 DAGExecutor._resolve_arg 中再处理
                tpl_id = params[len("__NODE_REF_FIRST__:"):]
                actual_id = tpl_to_node_ids.get(tpl_id, tpl_id)
                return f"$FIRST:{actual_id}"
        if isinstance(params, dict):
            return {k: self._replace_node_out_refs(v, tpl_to_node_ids) for k, v in params.items()}
        if isinstance(params, list):
            return [self._replace_node_out_refs(item, tpl_to_node_ids) for item in params]
        return params

    def _build_binary_op(
        self,
        captured: Dict[str, Any],
        dag: DAGGraph,
        missing: List[str],
    ) -> None:
        """二元运算：A op B = ? → call_algorithm + return_value"""
        left_tok = str(captured.get("left", ""))
        op_tok = str(captured.get("op", ""))
        right_tok = str(captured.get("right", ""))

        left_val = self._resolve_number(left_tok)
        right_val = self._resolve_number(right_tok)
        algo_key = self._resolve_op_algorithm(op_tok)

        if left_val is None:
            missing.append(f"数字'{left_tok}'未学")
        if right_val is None:
            missing.append(f"数字'{right_tok}'未学")
        if algo_key is None:
            missing.append(f"操作符'{op_tok}'未学")

        if left_val is None or right_val is None or algo_key is None:
            return

        call_id = self._new_node_id()
        dag.add_node(DAGNode(
            id=call_id,
            action=OP_CALL_ALGORITHM,
            params={"key": algo_key, "args": [left_val, right_val]},
            description=f"调用算法 {algo_key}({left_val}, {right_val})",
        ))

        ret_id = self._new_node_id()
        dag.add_node(DAGNode(
            id=ret_id,
            action=OP_RETURN_VALUE,
            params={"value": f"${call_id}"},
            depends_on=[call_id],
            description="返回最终结果",
        ))

    def _build_attr_value(
        self,
        captured: Dict[str, Any],
        dag: DAGGraph,
        write_node_ids: List[str],
    ) -> None:
        """属性绑定：entity 的 attr value → write_memory"""
        entity = str(captured.get("entity", ""))
        attr = str(captured.get("attr", ""))
        value_tok = str(captured.get("value", ""))

        value = self._resolve_number(value_tok)
        if value is None:
            value = value_tok  # 保留原始 token

        node_id = self._new_node_id("w")
        dag.add_node(DAGNode(
            id=node_id,
            action=OP_WRITE_MEMORY,
            params={"entity": entity, "attr": attr, "value": value},
            description=f"写入工作记忆：{entity}.{attr} = {value}",
        ))
        write_node_ids.append(node_id)

    def _build_pronoun_aggregate(
        self,
        captured: Dict[str, Any],
        dag: DAGGraph,
        write_node_ids: List[str],
        algo_key: str,
        op_label: str,
    ) -> None:
        """指代聚合：pronoun 的 attr 之和/之差 → search_context + call_algorithm + return_value"""
        pronoun = str(captured.get("pronoun", ""))
        attr = str(captured.get("attr", ""))

        # search_context 节点：依赖所有 write_memory 节点
        search_id = self._new_node_id("s")
        dag.add_node(DAGNode(
            id=search_id,
            action=OP_SEARCH_CONTEXT,
            params={"keyword": pronoun},
            depends_on=list(write_node_ids),
            description=f"代词消解：'{pronoun}' → 实体列表",
        ))

        # call_algorithm 节点：从 search 结果收集 attr 值，调用算法
        call_id = self._new_node_id()
        dag.add_node(DAGNode(
            id=call_id,
            action=OP_CALL_ALGORITHM,
            params={
                "key": algo_key,
                "collect_from": {"node": search_id, "attr": attr},
            },
            depends_on=[search_id],
            description=f"调用算法 {algo_key}（收集'{attr}'属性值）",
        ))

        # return_value 节点
        ret_id = self._new_node_id()
        dag.add_node(DAGNode(
            id=ret_id,
            action=OP_RETURN_VALUE,
            params={"value": f"${call_id}"},
            depends_on=[call_id],
            description="返回最终结果",
        ))

    # ---- 值解析（查询陈述性记忆）----

    def _resolve_number(self, token: str) -> Optional[int]:
        """从陈述性记忆解析数字token的数值。

        单字符数字必须通过学习获得（查知识图谱）；
        多位数字串（如"31"）可解析（因为 digit_merge 规则是学来的）。
        """
        if self.declarative is not None:
            entities = self.declarative.find_entity_by_name(token)
            for e in entities:
                if e.attributes.get("kind") == "number":
                    return e.attributes.get("value")
        # fallback: 多位数字串可解析（digit_merge 规则是学来的）
        if len(token) > 1 and token.isdigit():
            try:
                return int(token)
            except (ValueError, TypeError):
                return None
        return None

    def _resolve_op_algorithm(self, token: str) -> Optional[str]:
        """从陈述性记忆解析操作符token的算法key"""
        if self.declarative is not None:
            entities = self.declarative.find_entity_by_name(token)
            for e in entities:
                if e.attributes.get("kind") == "operator":
                    op_type = e.attributes.get("op_type")
                    return _OP_TYPE_TO_ALGORITHM.get(op_type)
        return None

    def _check_algorithms_learned(self, dag: DAGGraph) -> List[str]:
        """闸门3：检查 call_algorithm 引用的 key 是否已学习"""
        if self.procedural is None:
            return []
        learned_keys = set()
        for proc in self.procedural.list_procedures():
            key = proc.attributes.get("algorithm_key") if proc.attributes else None
            if key:
                learned_keys.add(key)
        unlearned: List[str] = []
        for node in dag.nodes.values():
            if node.action == OP_CALL_ALGORITHM:
                key = node.params.get("key", "")
                if key and key not in learned_keys:
                    unlearned.append(key)
        return unlearned


# ============================================================
# DAG 执行引擎
# ============================================================

class DAGExecutor:
    """DAG 执行引擎 - 通用虚拟机

    输入一张 DAG，按拓扑序逐节点执行原子操作。
    节点之间通过工作记忆传数据。引擎不认识任何业务语义。
    """

    def __init__(
        self,
        algorithm_registry: AlgorithmRegistry,
        procedural_memory: Optional[ProceduralMemory] = None,
        use_embodied: bool = True,
        declarative_memory: Optional[Any] = None,
    ) -> None:
        self.algorithms = algorithm_registry
        self.procedural = procedural_memory
        self.use_embodied = use_embodied
        self.declarative = declarative_memory

    def execute(
        self,
        dag: DAGGraph,
        working_memory: Any,
        output_buffer: OutputBuffer,
    ) -> Tuple[Optional[Any], float, List[str]]:
        """执行 DAG，返回 (答案, 置信度, 节点执行序列描述)。

        将每步执行结果写入 output_buffer 推理链。
        """
        seq, ok = dag.topological_sort()
        if not ok:
            output_buffer.add_step(
                action="dag_execute",
                description="拓扑排序失败：DAG存在环",
                confidence=0.0,
            )
            return None, 0.0, []

        answer: Optional[Any] = None
        confidence = 1.0
        node_descs: List[str] = []

        for node_id in seq:
            node = dag.get_node(node_id)
            if node is None:
                continue
            try:
                result = self._execute_node(node, working_memory)
                working_memory.put_node_output(node_id, result)
                desc = self._describe_node(node, result)
                node_descs.append(desc)
                output_buffer.add_step(
                    action=node.action,
                    description=desc,
                    inputs=dict(node.params),
                    outputs={"result": result},
                )
                if node.action == OP_RETURN_VALUE:
                    answer = result
            except Exception as exc:
                output_buffer.add_step(
                    action=node.action,
                    description=f"节点 {node_id} 执行失败：{exc}",
                    inputs=dict(node.params),
                    confidence=0.0,
                )
                return None, 0.0, node_descs

        return answer, round(confidence, 3), node_descs

    def _lookup_op_type(self, action: str) -> Optional[str]:
        if self.declarative is not None:
            entities = self.declarative.find_entity_by_name(action)
            for e in entities:
                if e.attributes.get("kind") == "operation":
                    return e.attributes.get("op_type")
        return None

    def _execute_node(self, node: DAGNode, wm: Any) -> Any:
        """执行单个节点，分发到对应的原子操作处理器"""
        action = node.action
        params = node.params

        op_type = self._lookup_op_type(action)
        if op_type is not None:
            if op_type == "write_memory":
                return self._op_write_memory(wm, params)
            if op_type == "read_memory":
                return self._op_read_memory(wm, params)
            if op_type == "search_context":
                return self._op_search_context(wm, params)
            if op_type == "call_algorithm":
                return self._op_call_algorithm(wm, params)
            if op_type == "return_value":
                return self._op_return_value(wm, params)
            raise ValueError(f"未知 op_type：{op_type}")

        if action == OP_WRITE_MEMORY:
            return self._op_write_memory(wm, params)
        if action == OP_READ_MEMORY:
            return self._op_read_memory(wm, params)
        if action == OP_SEARCH_CONTEXT:
            return self._op_search_context(wm, params)
        if action == OP_CALL_ALGORITHM:
            return self._op_call_algorithm(wm, params)
        if action == OP_RETURN_VALUE:
            return self._op_return_value(wm, params)
        raise ValueError(f"未知原子操作：{action}")

    # ---- 原子操作实现（L0 硬件指令集）----

    def _op_write_memory(self, wm: Any, params: Dict[str, Any]) -> Any:
        """write_memory(entity, attr, value) → ok

        所有参数支持 $ / $FIRST: 节点输出引用（运行时resolve）。
        这用于 acquire_event 更新"代词解析后的第一个实体"的属性。
        """
        entity = self._resolve_arg(params.get("entity", ""), wm)
        attr = self._resolve_arg(params.get("attr", ""), wm)
        value = self._resolve_arg(params.get("value"), wm)
        wm.write_attr(entity, attr, value, declarative_memory=self.declarative)
        return value

    def _op_read_memory(self, wm: Any, params: Dict[str, Any]) -> Any:
        """read_memory(entity, attr) → value（参数支持节点输出引用）"""
        entity = self._resolve_arg(params.get("entity", ""), wm)
        attr = self._resolve_arg(params.get("attr", ""), wm)
        return wm.read_attr(entity, attr)

    def _op_search_context(self, wm: Any, params: Dict[str, Any]) -> Any:
        """search_context(keyword) → entities（参数支持节点输出引用）

        Phase M2: 传递 declarative_memory 以启用性别过滤 + 距离衰减排序
        """
        keyword = self._resolve_arg(params.get("keyword", ""), wm)
        return wm.resolve_pronoun(keyword, declarative_memory=self.declarative)

    def _op_call_algorithm(self, wm: Any, params: Dict[str, Any]) -> Any:
        """call_algorithm(key, args/collect_from) → result

        支持同时有 collect_from 和 args：最终 args = collect 收集到的列表 + args 参数列表。
        这用于"获取事件句"等需要"把上下文已有的值 + 新捕获的值 一起传给算法"的场景。
        """
        key = params.get("key", "")
        collected_args: List[Any] = []
        direct_args: List[Any] = []

        # 1) collect_from：从搜索结果（上下文实体列表）中逐个读属性，收集参数
        if "collect_from" in params:
            cf = params["collect_from"]
            ref_node = cf.get("node", "")
            attr = cf.get("attr", "")
            entity_names = wm.get_node_output(ref_node)
            if not isinstance(entity_names, list):
                entity_names = [entity_names] if entity_names else []
            for name in entity_names:
                val = wm.read_attr_or_none(name, attr)
                if val is not None:
                    collected_args.append(val)

        # 2) direct args：字面量 args 列表（支持 $ 节点输出引用）
        raw_args = params.get("args", [])
        direct_args = [self._resolve_arg(a, wm) for a in raw_args]

        # 3) 合并：collected 在前，direct 在后（语义对应 adding(已有值, 新增加值)）
        args = collected_args + direct_args

        # 门控：算法 key 必须已注册
        algs = self.algorithms.all_algorithms()
        if key not in algs:
            raise ValueError(f"算法 '{key}' 未注册")

        alg = algs[key]
        if not hasattr(alg, "execute"):
            raise ValueError(f"算法 '{key}' 无 execute 方法")

        # adding/subtracting 接受两个位置参数
        if key in ("adding", "subtracting"):
            if len(args) < 2:
                raise ValueError(f"算法 '{key}' 需要至少2个参数，得到 {args}")
            result = alg.execute(args[0], args[1], use_embodied=self.use_embodied)
        else:
            result = alg.execute(*args)
        if isinstance(result, dict):
            return result.get("result")
        return result

    def _op_return_value(self, wm: Any, params: Dict[str, Any]) -> Any:
        """return_value(value) → value（终止推理并输出答案）"""
        value = params.get("value")
        return self._resolve_arg(value, wm)

    # ---- 参数解析 ----

    def _resolve_arg(self, arg: Any, wm: Any) -> Any:
        if isinstance(arg, str):
            if arg.startswith("$FIRST:"):
                ref_id = arg[len("$FIRST:"):]
                raw = wm.get_node_output(ref_id)
                if isinstance(raw, list):
                    return raw[0] if raw else None
                return raw
            if arg.startswith("$"):
                ref_id = arg[1:]
                return wm.get_node_output(ref_id)
        return arg

    def _describe_node(self, node: DAGNode, result: Any) -> str:
        """生成节点执行的人类可读描述"""
        action = node.action
        params = node.params
        if action == OP_WRITE_MEMORY:
            return f"写入工作记忆：{params.get('entity')}.{params.get('attr')} = {params.get('value')}"
        if action == OP_READ_MEMORY:
            return f"读取工作记忆：{params.get('entity')}.{params.get('attr')} → {result}"
        if action == OP_SEARCH_CONTEXT:
            return f"代词消解：'{params.get('keyword')}' → {result}"
        if action == OP_CALL_ALGORITHM:
            return f"调用算法 {params.get('key')} → {result}"
        if action == OP_RETURN_VALUE:
            return f"返回最终答案：{result}"
        return f"{action}: {params}"


# ============================================================
# 推理操作区
# ============================================================

class ReasoningArea:
    """推理操作区"""

    # op_type -> 对应的 algorithm_key（兼容旧单步路径）
    _OP_TO_ALGORITHM_KEY = {
        "add": "adding",
        "sub": "subtracting",
        "mul": "multiplying",
        "div": "dividing",
    }

    def __init__(
        self,
        algorithm_registry: Optional[AlgorithmRegistry] = None,
        procedural_memory: Optional[ProceduralMemory] = None,
        pattern_matcher: Optional[PatternMatcher] = None,
        intent_recognizer: Optional[IntentRecognizer] = None,
        use_embodied: bool = True,
        declarative_memory: Optional[Any] = None,
    ) -> None:
        self.algorithms = algorithm_registry or AlgorithmRegistry()
        self.procedural = procedural_memory
        self.pattern_matcher = pattern_matcher or PatternMatcher()
        self.intent_recognizer = intent_recognizer or IntentRecognizer()
        self.use_embodied = use_embodied
        self.declarative = declarative_memory

    # ---------- 主入口 ----------
    def run(self, workspace: Workspace) -> OutputBuffer:
        """对 workspace 执行完整推理流程，写回 output_buffer 并返回"""
        workspace.mark_reasoning()
        out = workspace.output_buffer

        # 1. 模式匹配
        tokens = workspace.input_buffer.tokens
        patterns = self.pattern_matcher.match(tokens)
        out.add_step(
            action="pattern_match",
            description=f"对 tokens={tokens} 执行模式匹配",
            outputs={
                "matched_count": len(patterns),
                "best": patterns[0].dict() if patterns else None,
            },
        )

        # 2. 检查是否有 build_dag 模式命中 → DAG 推理路径
        dag_matches = [p for p in patterns if p.action and p.action.startswith("build_dag:")]
        if dag_matches:
            return self._run_dag(workspace, patterns)

        # 3. 旧单步推理路径（无 build_dag 模式时回退）
        return self._run_single_step(workspace, patterns)

    # ---------- DAG 推理路径 ----------
    def _run_dag(self, workspace: Workspace, patterns: List[PatternMatchResult]) -> OutputBuffer:
        out = workspace.output_buffer
        wm = workspace.working_memory
        wm.clear()

        # DAG 构建（= 理解）
        builder = DAGBuilder(self.declarative, self.procedural, working_memory=wm)
        build_result = builder.build(patterns)

        if not build_result.success:
            out.add_step(
                action="dag_build_failed",
                description=f"DAG构建失败：{build_result.failure_reason}",
                outputs={
                    "failure_type": build_result.failure_type,
                    "missing": build_result.missing,
                },
                confidence=0.0,
            )
            out.set_answer(None, confidence=0.0)
            workspace.mark_done()
            return out

        out.add_step(
            action="dag_build",
            description=f"DAG构建成功：{build_result.node_count}个节点，{build_result.pattern_count}条模式命中",
            outputs={
                "node_count": build_result.node_count,
                "dag_text": build_result.dag.to_text() if build_result.dag else "",
            },
        )

        # DAG 执行（= 求解）
        executor = DAGExecutor(self.algorithms, self.procedural, self.use_embodied, declarative_memory=self.declarative)
        answer, confidence, _ = executor.execute(build_result.dag, wm, out)

        out.set_answer(answer, confidence=confidence)
        workspace.mark_done()
        return out

    # ---------- 旧单步推理路径（回退）----------
    def _run_single_step(self, workspace: Workspace, patterns: List[PatternMatchResult]) -> OutputBuffer:
        out = workspace.output_buffer

        # 意图识别
        tokens = workspace.input_buffer.tokens
        intent = self.intent_recognizer.recognize(patterns, tokens)
        out.add_step(
            action="intent_recognize",
            description=intent.description or f"识别为 {intent.intent_type}",
            outputs=intent.dict(),
            confidence=intent.confidence,
        )

        # 根据意图选择算法并执行
        answer: Optional[Any] = None
        final_confidence = intent.confidence

        if intent.intent_type == IntentType.COMPUTE_BINARY:
            execution = self._execute_binary(intent.slots)
            out.add_step(
                action="execute_algorithm",
                description="执行二元运算算法",
                inputs=intent.slots,
                outputs={"result": execution.get("result"), "method": execution.get("method")},
                procedure_id=intent.slots.get("operation"),
            )
            for sub in execution.get("steps", [])[1:-1] if execution.get("method") == "embodied" else []:
                out.add_step(action="sub_step", description=sub)
            answer = execution.get("result")
            final_confidence = min(1.0, intent.confidence + 0.1)

        if answer is None:
            answer = self._try_run_procedures(workspace)
            if answer is not None:
                out.add_step(
                    action="execute_procedure",
                    description="从激活的程序性记忆执行成功",
                    outputs={"answer": answer},
                )
                final_confidence = max(final_confidence, 0.7)

        if answer is None:
            out.add_step(
                action="no_result",
                description="未能求解",
                confidence=0.0,
            )
            final_confidence = 0.0

        out.set_answer(answer, confidence=round(final_confidence, 3))
        workspace.mark_done()
        return out

    # ---------- 内部执行（旧路径）----------
    def _is_algorithm_learned(self, algorithm_key: str) -> bool:
        if self.procedural is None:
            return True
        for proc in self.procedural.list_procedures():
            if proc.attributes and proc.attributes.get("algorithm_key") == algorithm_key:
                return True
        return False

    def _execute_binary(self, slots: Dict[str, Any]) -> Dict[str, Any]:
        op = slots.get("operation")
        a = slots.get("operand_left")
        b = slots.get("operand_right")
        algorithm_key = self._OP_TO_ALGORITHM_KEY.get(op)

        if not algorithm_key or not self._is_algorithm_learned(algorithm_key):
            return {
                "result": None,
                "method": "not_learned",
                "steps": [f"[gate] 未学过操作 '{op}' 的算法程序（algorithm_key={algorithm_key}）"],
                "trace": "",
            }

        if algorithm_key == "adding":
            return self.algorithms.adding.execute(a, b, use_embodied=self.use_embodied)
        if algorithm_key == "subtracting":
            return self.algorithms.subtracting.execute(a, b, use_embodied=self.use_embodied)
        algs = self.algorithms.all_algorithms()
        if algorithm_key in algs:
            alg = algs[algorithm_key]
            if hasattr(alg, "execute"):
                return alg.execute(a, b, use_embodied=self.use_embodied)
            return {"result": None, "method": "no_execute", "steps": [], "trace": ""}
        return {
            "result": None,
            "method": "algorithm_unavailable",
            "steps": [f"[gate] algorithm_key='{algorithm_key}' 未注册到算法硬件"],
            "trace": "",
        }

    def _try_run_procedures(self, workspace: Workspace) -> Optional[Any]:
        if self.procedural is None:
            return None
        procs = workspace.activation_area.procedures()
        for proc in procs:
            ans = self._execute_procedure(proc, workspace)
            if ans is not None:
                return ans
        return None

    def _execute_procedure(self, proc: Procedure, workspace: Workspace) -> Optional[Any]:
        algs = self.algorithms.all_algorithms()
        algorithm_key = proc.attributes.get("algorithm_key") if proc.attributes else None
        if not algorithm_key or algorithm_key not in algs:
            return None
        alg = algs[algorithm_key]
        if algorithm_key == "counting":
            entities = workspace.activation_area.entities()
            nums = [
                e.attributes.get("value")
                for e in entities
                if e.attributes.get("kind") == "number" and e.attributes.get("value") is not None
            ]
            if nums:
                execution = alg.execute(nums)
                return execution.get("count")
            return None
        return None
