"""物理世界交互接口 - Phase 1 占位，Phase 3 扩展"""
from __future__ import annotations

from typing import Any, Dict, List


class PhysicalInterface:
    """与物理世界/虚拟物理环境交互的接口

    MVP 阶段：仅提供模拟接口（虚拟对象），不真正操作硬件。
    - 可生成"手指""苹果"等物理对象集合
    - 可执行视觉观察、手指抬起等模拟动作
    """

    def __init__(self) -> None:
        self._objects: Dict[str, List[Any]] = {}
        self._reset_scene()

    def _reset_scene(self) -> None:
        self._objects = {
            "fingers_left": [f"finger_L{i}" for i in range(1, 6)],
            "fingers_right": [f"finger_R{i}" for i in range(1, 6)],
            "apples": [],
        }

    def show_fingers(self, count: int, hand: str = "right") -> List[str]:
        """显示指定数量的手指 -> 返回选中的对象列表"""
        key = f"fingers_{hand}"
        fingers = self._objects.get(key, [])
        count = min(count, len(fingers))
        shown = fingers[:count]
        return shown

    def place_apples(self, count: int) -> List[str]:
        """放置 N 个苹果到场景"""
        apples = [f"apple_{i+1}" for i in range(count)]
        self._objects["apples"].extend(apples)
        return apples

    def observe(self, obj_group: str = "apples") -> List[Any]:
        """观察场景，返回观察到的对象列表"""
        return list(self._objects.get(obj_group, []))

    def clear_scene(self) -> None:
        self._reset_scene()
