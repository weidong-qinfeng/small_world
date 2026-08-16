"""输出缓冲区 - 保存推理结果和推理链"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ReasoningStep(BaseModel):
    """推理链中的一步"""
    order: int
    action: str = ""                            # 动作名
    description: str = ""                       # 人类可读描述
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    procedure_id: Optional[str] = None
    confidence: float = 1.0


class OutputBuffer:
    """工作区的输出缓冲区

    存储：
    - final_answer: 最终答案（any 类型，通常是 str 或 int）
    - reasoning_chain: 推理步骤序列
    - confidence: 总体置信度
    """

    def __init__(self) -> None:
        self.final_answer: Optional[Any] = None
        self.reasoning_chain: List[ReasoningStep] = []
        self.confidence: float = 0.0
        self._step_counter = 0

    def add_step(
        self,
        action: str,
        description: str = "",
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        procedure_id: Optional[str] = None,
        confidence: float = 1.0,
    ) -> ReasoningStep:
        self._step_counter += 1
        step = ReasoningStep(
            order=self._step_counter,
            action=action,
            description=description,
            inputs=inputs or {},
            outputs=outputs or {},
            procedure_id=procedure_id,
            confidence=confidence,
        )
        self.reasoning_chain.append(step)
        return step

    def set_answer(self, answer: Any, confidence: float = 1.0) -> None:
        self.final_answer = answer
        self.confidence = confidence

    def format_chain(self) -> str:
        lines = ["=== 推理链 ==="]
        for s in self.reasoning_chain:
            line = f"  Step {s.order}: [{s.action}] {s.description}"
            if s.outputs:
                line += f" -> {s.outputs}"
            lines.append(line)
        lines.append(f"=== 最终答案: {self.final_answer} (置信度={self.confidence:.2f}) ===")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.final_answer,
            "confidence": self.confidence,
            "chain": [s.dict() for s in self.reasoning_chain],
        }

    def clear(self) -> None:
        self.final_answer = None
        self.reasoning_chain = []
        self.confidence = 0.0
        self._step_counter = 0

    def is_ready(self) -> bool:
        return self.final_answer is not None
