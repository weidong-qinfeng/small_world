"""推理操作区 - 协调模式匹配、意图识别和程序性记忆/算法的执行

将推理链写入 OutputBuffer。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from digital_brain.src.algorithms.adding import AddingAlgorithm
from digital_brain.src.algorithms.counting import CountingAlgorithm
from digital_brain.src.algorithms.mapping import MappingAlgorithm
from digital_brain.src.algorithms.merging import MergingAlgorithm
from digital_brain.src.algorithms.subtracting import SubtractingAlgorithm
from digital_brain.src.core.memory.procedural_memory import ProceduralMemory
from digital_brain.src.core.models import Procedure
from digital_brain.src.core.pattern.intent_recognizer import Intent, IntentRecognizer, IntentType
from digital_brain.src.core.pattern.pattern_matcher import PatternMatchResult, PatternMatcher
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


class ReasoningArea:
    """推理操作区"""

    def __init__(
        self,
        algorithm_registry: Optional[AlgorithmRegistry] = None,
        procedural_memory: Optional[ProceduralMemory] = None,
        pattern_matcher: Optional[PatternMatcher] = None,
        intent_recognizer: Optional[IntentRecognizer] = None,
        use_embodied: bool = True,
    ) -> None:
        self.algorithms = algorithm_registry or AlgorithmRegistry()
        self.procedural = procedural_memory
        self.pattern_matcher = pattern_matcher or PatternMatcher()
        self.intent_recognizer = intent_recognizer or IntentRecognizer()
        self.use_embodied = use_embodied

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

        # 2. 意图识别
        intent = self.intent_recognizer.recognize(patterns, tokens)
        out.add_step(
            action="intent_recognize",
            description=intent.description or f"识别为 {intent.intent_type}",
            outputs=intent.dict(),
            confidence=intent.confidence,
        )

        # 3. 根据意图选择算法并执行
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
            # 将具身子步骤也写进推理链（若有）
            for sub in execution.get("steps", [])[1:-1] if execution.get("method") == "embodied" else []:
                out.add_step(action="sub_step", description=sub)
            answer = execution.get("result")
            final_confidence = min(1.0, intent.confidence + 0.1)

        if answer is None:
            # 兜底：尝试从激活的 procedural 中找可执行的
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

    # ---------- 内部执行 ----------
    # op_type -> 对应的 algorithm_key（必须同时在 procedural memory 中学过对应 procedure）
    _OP_TO_ALGORITHM_KEY = {
        "add": "adding",
        "sub": "subtracting",
        "mul": "multiplying",   # 预留：需对应 procedural 里学过 algorithm_key="multiplying"
        "div": "dividing",      # 预留：需对应 procedural 里学过 algorithm_key="dividing"
    }

    def _is_algorithm_learned(self, algorithm_key: str) -> bool:
        """是否在 procedural memory 里学过指向该 algorithm_key 的程序性记忆。
        procedural_memory=None（兼容旧模式）时视为已学过，用于老测试 / KnowledgeBuilder 场景。
        """
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

        # --- 门控：没学过对应程序性记忆 → 拒绝执行 ---
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
        # mul / dividing 等需要 registry 里有 + procedural 里学过，这里统一防护
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
        """根据 procedure.attributes["algorithm_key"] 路由到注册算法执行。
        禁止用"取激活实体的 value"这种绕过方式，必须走真正的算法硬件。
        """
        algs = self.algorithms.all_algorithms()
        algorithm_key = proc.attributes.get("algorithm_key") if proc.attributes else None
        if not algorithm_key or algorithm_key not in algs:
            return None
        alg = algs[algorithm_key]
        # counting: 对工作区激活的数字实体做具身映射后计数（简化：取第一个有 embodied_mapping 的实体数）
        if algorithm_key == "counting":
            entities = workspace.activation_area.entities()
            nums = [
                e.attributes.get("value")
                for e in entities
                if e.attributes.get("kind") == "number" and e.attributes.get("value") is not None
            ]
            if nums:
                # counting 算法在 MVP 中被 adding/subtracting 内部流程驱动，
                # 这里顶层单独触发时只返回"集合大小 = 激活数字个数"
                execution = alg.execute(nums)
                return execution.get("count")
            return None
        # mapping / merging 顶层独立触发的场景：MVP 中暂不使用（总是被 adding 内部步骤调用）
        return None
