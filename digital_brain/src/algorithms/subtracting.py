"""减法算法 - 基于逆操作的组合实现

人类学习：3-1 -> 有3个苹果，拿走1个，还剩2个
算法实现：map(a) 取前 b 个（或 a-b 的直接算数）
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from digital_brain.src.algorithms.counting import CountingAlgorithm
from digital_brain.src.algorithms.mapping import MappingAlgorithm


class SubtractingAlgorithm:
    """减法算法"""

    name = "subtracting"
    description = "计算 a - b，用具身集合模拟'拿走'，或直接算术"

    def __init__(
        self,
        mapping: Optional[MappingAlgorithm] = None,
        counting: Optional[CountingAlgorithm] = None,
    ) -> None:
        self.mapping = mapping or MappingAlgorithm()
        self.counting = counting or CountingAlgorithm()

    def execute(self, a: int, b: int, use_embodied: bool = True) -> Dict[str, Any]:
        a, b = int(a), int(b)
        if b > a:
            # MVP: 不处理负数，返回 0 + 警告
            steps = [f"[sub-warn] {a} - {b} 为负数，MVP 返回 0"]
            return {"result": 0, "steps": steps, "method": "warn", "trace": "; ".join(steps)}
        if use_embodied:
            return self._embodied_sub(a, b)
        result = a - b
        steps = [f"[sub-direct] {a} - {b} = {result}"]
        return {"result": result, "steps": steps, "method": "direct", "trace": "; ".join(steps)}

    def _embodied_sub(self, a: int, b: int) -> Dict[str, Any]:
        all_steps: List[str] = []
        ma = self.mapping.number_to_set(a)
        all_steps.extend(ma["steps"])
        set_a = ma["set"]
        all_steps.append(f"[sub] 从 {set_a} 中拿走前 {b} 个")
        remaining = set_a[b:]
        all_steps.append(f"[sub] 剩余集合: {remaining}")
        ct = self.counting.execute(remaining)
        all_steps.extend(ct["steps"])
        result = ct["count"]
        all_steps.append(f"[sub-embodied] 结论: {a} - {b} = {result}")
        return {
            "result": result,
            "steps": all_steps,
            "method": "embodied",
            "trace": "; ".join(all_steps),
            "intermediate": {"set_a": set_a, "remaining": remaining},
        }
