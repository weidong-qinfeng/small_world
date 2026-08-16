"""教学策略库 - Phase 2 占位实现"""
from __future__ import annotations

from typing import Any, Dict, List


class TeachingStrategies:
    """教学策略库

    - 示范教学 (demonstration)
    - 引导发现 (guided_discovery)
    - 组合教学 (compositional)
    - 类比教学 (analogy)

    Phase 1：仅提供接口，不真正生成教案。
    """

    def __init__(self) -> None:
        self.progress: Dict[str, float] = {}   # topic -> 0..1
        self.library: Dict[str, List[Any]] = {
            "demonstration": [],
            "guided_discovery": [],
            "compositional": [],
            "analogy": [],
        }

    def add_material(self, strategy: str, material: Any) -> None:
        if strategy in self.library:
            self.library[strategy].append(material)

    def pick_strategy(self, topic: str) -> str:
        return "demonstration"

    def record_progress(self, topic: str, mastery: float) -> None:
        self.progress[topic] = max(self.progress.get(topic, 0.0), mastery)

    def get_progress(self, topic: str) -> float:
        return self.progress.get(topic, 0.0)
