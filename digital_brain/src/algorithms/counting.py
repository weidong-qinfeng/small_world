"""计数算法 - 最基础的数学算法：数出一个集合里有多少个元素

人类学习过程：对着手指/苹果一个一个念 1、2、3...
算法实现：遍历可迭代对象计数，或直接按整数值返回。
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Union


class CountingAlgorithm:
    """计数算法"""

    name = "counting"
    description = "对集合或数字进行计数，返回元素个数"

    def execute(self, items: Union[int, Iterable[Any]]) -> Dict[str, Any]:
        """执行计数

        Args:
            items: 可以是数字 int（直接作为计数值），也可以是一个集合
        Returns:
            {"count": int, "steps": List[str], "trace": str}
        """
        steps: List[str] = []
        if isinstance(items, int):
            # 数字本身即为计数结果（具身映射：数字 N <-> 数到 N）
            result = items
            steps.append(f"[count] 数字 {items} 直接对应计数结果 {items}")
        elif isinstance(items, Iterable):
            items_list = list(items)
            result = 0
            for idx, it in enumerate(items_list, start=1):
                result = idx
                steps.append(f"[count] 数第 {idx} 个元素 {it!r} -> 当前={idx}")
            if result == 0:
                steps.append("[count] 空集合 -> 计数结果 0")
        else:
            raise TypeError(f"Cannot count items of type {type(items).__name__}")
        trace = "; ".join(steps)
        return {"count": result, "steps": steps, "trace": trace}
