"""具身映射算法 - 将抽象符号映射到物理/集合表示

人类学习过程：看到数字"3"，在脑中映射出3个手指头/3个苹果
算法实现：把数字 N 转换成长度为 N 的集合表示 (list of units)，或反向把集合压缩回数字。
"""
from __future__ import annotations

from typing import Any, Dict, List, Union


class MappingAlgorithm:
    """具身映射算法：数字 <-> 集合"""

    name = "mapping"
    description = "数字 N 映射为 N 个单元的集合，或反向压缩"

    DEFAULT_UNIT = "•"  # 用小圆点表示具身单元（类似手指）

    def number_to_set(self, n: int, unit: Any = None) -> Dict[str, Any]:
        """数字 -> 集合"""
        if n < 0:
            raise ValueError("Cannot map negative number to set")
        unit = unit if unit is not None else self.DEFAULT_UNIT
        result_set: List[Any] = [unit for _ in range(n)]
        steps = [f"[mapping] 数字 {n} 映射为 {n} 个 {unit!r} 的集合"]
        return {
            "set": result_set,
            "size": n,
            "steps": steps,
            "trace": "; ".join(steps),
        }

    def set_to_number(self, s: Union[List[Any], int]) -> Dict[str, Any]:
        """集合 -> 数字"""
        if isinstance(s, int):
            size = s
        else:
            size = len(s)
        steps = [f"[mapping] 大小为 {size} 的集合压缩为数字 {size}"]
        return {
            "number": size,
            "size": size,
            "steps": steps,
            "trace": "; ".join(steps),
        }

    def execute(self, value: Union[int, List[Any]], direction: str = "to_set") -> Dict[str, Any]:
        if direction == "to_set":
            return self.number_to_set(int(value))
        elif direction == "to_number":
            return self.set_to_number(value)
        else:
            raise ValueError(f"Unknown direction: {direction}")
