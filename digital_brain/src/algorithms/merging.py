"""合并算法 - 把两个集合合并成一个新集合

人类学习过程：左手3个苹果，右手2个苹果，放到一起 -> 一堆苹果
算法实现：list A + list B
"""
from __future__ import annotations

from typing import Any, Dict, List


class MergingAlgorithm:
    """合并算法"""

    name = "merging"
    description = "将两个集合 A、B 合并为 A∪B（保持顺序的拼接，允许重复元素）"

    def execute(self, set_a: List[Any], set_b: List[Any]) -> Dict[str, Any]:
        set_a = list(set_a) if set_a is not None else []
        set_b = list(set_b) if set_b is not None else []
        merged = set_a + set_b
        steps = [
            f"[merge] 集合 A (size={len(set_a)}): {set_a}",
            f"[merge] 集合 B (size={len(set_b)}): {set_b}",
            f"[merge] 合并后 (size={len(merged)}): {merged}",
        ]
        return {
            "merged": merged,
            "size": len(merged),
            "steps": steps,
            "trace": "; ".join(steps),
        }
