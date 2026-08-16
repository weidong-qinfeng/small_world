"""加法算法 - 组合：映射 + 合并 + 计数

人类学习过程：1+1 -> 左手1个手指，右手1个手指，放到一起，数一数 -> 2
算法实现：
    add(a, b) = count( merge( map(a), map(b) ) )
也提供直接算术加法作为确定性 fallback。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from digital_brain.src.algorithms.counting import CountingAlgorithm
from digital_brain.src.algorithms.mapping import MappingAlgorithm
from digital_brain.src.algorithms.merging import MergingAlgorithm


class AddingAlgorithm:
    """加法算法（组合版）"""

    name = "adding"
    description = "计算 a + b，通过 map->merge->count 组合实现，也支持直接算术"

    def __init__(
        self,
        mapping: Optional[MappingAlgorithm] = None,
        merging: Optional[MergingAlgorithm] = None,
        counting: Optional[CountingAlgorithm] = None,
    ) -> None:
        self.mapping = mapping or MappingAlgorithm()
        self.merging = merging or MergingAlgorithm()
        self.counting = counting or CountingAlgorithm()

    def execute(self, a: int, b: int, use_embodied: bool = True) -> Dict[str, Any]:
        """执行 a + b

        Args:
            a: 操作数1
            b: 操作数2
            use_embodied: 是否用具身组合路径（map->merge->count）。True 展示推理链，False 直接算数

        Returns:
            {"result": int, "steps": [...], "method": "embodied"|"direct", "trace": str}
        """
        a, b = int(a), int(b)
        if use_embodied:
            return self._embodied_add(a, b)
        result = a + b
        steps = [f"[add-direct] {a} + {b} = {result}"]
        return {
            "result": result,
            "steps": steps,
            "method": "direct",
            "trace": "; ".join(steps),
        }

    def _embodied_add(self, a: int, b: int) -> Dict[str, Any]:
        all_steps: List[str] = []
        # 1. 映射 a -> set_a
        ma = self.mapping.number_to_set(a)
        all_steps.extend(ma["steps"])
        set_a = ma["set"]
        # 2. 映射 b -> set_b
        mb = self.mapping.number_to_set(b)
        all_steps.extend(mb["steps"])
        set_b = mb["set"]
        # 3. 合并
        mg = self.merging.execute(set_a, set_b)
        all_steps.extend(mg["steps"])
        merged = mg["merged"]
        # 4. 计数
        ct = self.counting.execute(merged)
        all_steps.extend(ct["steps"])
        result = ct["count"]
        all_steps.append(f"[add-embodied] 结论: {a} + {b} = {result}")
        return {
            "result": result,
            "steps": all_steps,
            "method": "embodied",
            "trace": "; ".join(all_steps),
            "intermediate": {
                "set_a": set_a,
                "set_b": set_b,
                "merged": merged,
            },
        }
