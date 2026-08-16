"""记忆巩固机制 - v2 实现记忆固化与启动恢复

对应详细设计/记忆系统设计.md 中的"记忆固化机制"章节。

核心职责：
1. consolidate_to_disk(dir)：把 declarative + procedural 序列化到硬盘
2. restore_from_disk(dir)：启动时从硬盘恢复知识
3. record_activation：记录激活次数（用于将来淘汰低频知识，当前保留）

存储格式：
- {dir}/declarative_entities.json
- {dir}/declarative_relations.json
- {dir}/procedural.json
- {dir}/meta.json   (固化时间、版本等元信息)
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

from digital_brain.src.core.memory.declarative_memory import DeclarativeMemory
from digital_brain.src.core.memory.procedural_memory import ProceduralMemory


# 存储文件名常量
ENTITIES_FILE = "declarative_entities.json"
RELATIONS_FILE = "declarative_relations.json"
PROCEDURAL_FILE = "procedural.json"
META_FILE = "meta.json"

# 当前存储格式版本
STORAGE_VERSION = "v2"


class MemoryConsolidation:
    """记忆巩固机制

    Phase 1 阶段：保留占位接口。
    v2 阶段：实现真正的固化/恢复。
    """

    def __init__(
        self,
        declarative_memory: DeclarativeMemory,
        procedural_memory: ProceduralMemory,
        threshold: float = 0.2,
    ) -> None:
        self.declarative = declarative_memory
        self.procedural = procedural_memory
        self.threshold = threshold
        self._activation_counts: Dict[str, int] = {}  # id -> count

    # ---------- 激活记录（用于将来淘汰） ----------
    def record_activation(self, entity_or_proc_id: str) -> None:
        self._activation_counts[entity_or_proc_id] = \
            self._activation_counts.get(entity_or_proc_id, 0) + 1

    # ---------- 固化到硬盘 ----------
    def consolidate_to_disk(self, storage_dir: str) -> Dict[str, Any]:
        """把当前陈述性 + 程序性记忆固化到 storage_dir。

        会覆盖 storage_dir 中已有的内容。
        返回固化统计信息。
        """
        os.makedirs(storage_dir, exist_ok=True)
        entities_path = os.path.join(storage_dir, ENTITIES_FILE)
        relations_path = os.path.join(storage_dir, RELATIONS_FILE)
        procedural_path = os.path.join(storage_dir, PROCEDURAL_FILE)
        meta_path = os.path.join(storage_dir, META_FILE)

        # 固化陈述性记忆
        self.declarative.save_json(entities_path, relations_path)
        # 固化程序性记忆
        self.procedural.save_json(procedural_path)
        # 写元信息
        meta = {
            "version": STORAGE_VERSION,
            "consolidated_at": datetime.now().isoformat(),
            "entity_count": self.declarative.entity_count,
            "relation_count": self.declarative.relation_count,
            "procedure_count": self.procedural.procedure_count,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        return {
            "consolidated": True,
            "storage_dir": storage_dir,
            "entity_count": self.declarative.entity_count,
            "relation_count": self.declarative.relation_count,
            "procedure_count": self.procedural.procedure_count,
            "consolidated_at": meta["consolidated_at"],
        }

    # ---------- 从硬盘恢复 ----------
    def restore_from_disk(self, storage_dir: str) -> Dict[str, Any]:
        """从 storage_dir 加载陈述性 + 程序性记忆到当前 brain。

        若 storage_dir 不存在或为空，返回 skipped=True 不报错。
        返回恢复统计信息。
        """
        if not os.path.isdir(storage_dir):
            return {"restored": False, "skipped": True, "reason": "storage_dir not found"}

        entities_path = os.path.join(storage_dir, ENTITIES_FILE)
        relations_path = os.path.join(storage_dir, RELATIONS_FILE)
        procedural_path = os.path.join(storage_dir, PROCEDURAL_FILE)
        meta_path = os.path.join(storage_dir, META_FILE)

        # 元信息检查
        meta: Dict[str, Any] = {}
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

        # 恢复陈述性记忆
        entities_loaded = 0
        relations_loaded = 0
        if os.path.isfile(entities_path) and os.path.isfile(relations_path):
            self.declarative.load_json(entities_path, relations_path)
            entities_loaded = self.declarative.entity_count
            relations_loaded = self.declarative.relation_count

        # 恢复程序性记忆
        procedures_loaded = 0
        if os.path.isfile(procedural_path):
            self.procedural.load_json(procedural_path)
            procedures_loaded = self.procedural.procedure_count

        return {
            "restored": True,
            "skipped": False,
            "storage_dir": storage_dir,
            "version": meta.get("version", "unknown"),
            "consolidated_at": meta.get("consolidated_at", ""),
            "entity_count": entities_loaded,
            "relation_count": relations_loaded,
            "procedure_count": procedures_loaded,
        }

    @staticmethod
    def has_saved_state(storage_dir: str) -> bool:
        """检查 storage_dir 是否存在已固化的状态文件"""
        if not os.path.isdir(storage_dir):
            return False
        return (
            os.path.isfile(os.path.join(storage_dir, ENTITIES_FILE))
            and os.path.isfile(os.path.join(storage_dir, PROCEDURAL_FILE))
        )

    # ---------- 兼容旧 API（v1 占位） ----------
    def consolidate(self) -> dict:
        """v1 兼容：仅返回统计信息，不真正固化"""
        return {
            "total_activation_records": len(self._activation_counts),
            "declarative_count": self.declarative.entity_count,
            "procedural_count": self.procedural.procedure_count,
            "threshold": self.threshold,
        }
