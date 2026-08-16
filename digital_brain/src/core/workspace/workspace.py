"""工作区主类 - 协调输入缓冲区、激活记忆区、推理操作区、输出缓冲区

工作区生命周期：接收 → 激活 → 推理 → 输出 → 清空
"""
from __future__ import annotations

from typing import List, Optional

from digital_brain.src.core.memory.declarative_memory import DeclarativeMemory
from digital_brain.src.core.memory.memory_activation import MemoryActivation
from digital_brain.src.core.memory.procedural_memory import ProceduralMemory
from digital_brain.src.core.models import ActivatedKnowledge
from digital_brain.src.core.workspace.activation_area import ActivationArea
from digital_brain.src.core.workspace.input_buffer import InputBuffer
from digital_brain.src.core.workspace.output_buffer import OutputBuffer


class Workspace:
    """工作区"""

    def __init__(
        self,
        declarative_memory: DeclarativeMemory,
        procedural_memory: ProceduralMemory,
        memory_activation: Optional[MemoryActivation] = None,
        input_capacity: int = 100,
        activation_capacity: int = 20,
    ) -> None:
        self.declarative = declarative_memory
        self.procedural = procedural_memory
        self.activation_engine = memory_activation or MemoryActivation(declarative_memory, procedural_memory)
        self.input_buffer = InputBuffer(capacity=input_capacity)
        self.activation_area = ActivationArea(capacity=activation_capacity)
        self.output_buffer = OutputBuffer()
        # 生命周期跟踪
        self._current_phase = "idle"  # idle / input / activation / reasoning / done

    # ---------- 生命周期 ----------
    def receive_input(self, text: str, tokens: Optional[List[str]] = None) -> None:
        """Phase 1: 接收输入"""
        self.clear()
        self._current_phase = "input"
        self.input_buffer.receive(text, tokens=tokens)

    def activate(self) -> List[ActivatedKnowledge]:
        """Phase 2: 激活记忆"""
        if self._current_phase not in ("input", "activation"):
            raise RuntimeError("Must receive_input before activate")
        self._current_phase = "activation"
        acts = self.activation_engine.activate(self.input_buffer.tokens)
        acts_top = MemoryActivation.resolve_conflicts(acts, top_k=self.activation_area.capacity)
        self.activation_area.load(acts_top)
        return acts_top

    def mark_reasoning(self) -> None:
        """标记进入推理阶段"""
        self._current_phase = "reasoning"

    def mark_done(self) -> None:
        """标记完成"""
        self._current_phase = "done"

    def clear(self) -> None:
        """清空工作区，准备下一次循环"""
        self.input_buffer.clear()
        self.activation_area.clear()
        self.output_buffer.clear()
        self._current_phase = "idle"

    @property
    def phase(self) -> str:
        return self._current_phase

    def __repr__(self) -> str:
        return (
            f"Workspace(phase={self._current_phase}, "
            f"input={len(self.input_buffer)} tokens, "
            f"activated={len(self.activation_area)}, "
            f"answer={self.output_buffer.final_answer})"
        )
