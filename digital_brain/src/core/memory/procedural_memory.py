"""程序性记忆 - 知道'怎么做'的记忆，包括算法、技能、策略"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from digital_brain.src.core.models import Procedure


class ProceduralMemory:
    """程序性记忆存储类

    以层次结构存储 Procedure：父 procedure 可以组合子 procedure。
    支持按触发条件检索。
    """

    def __init__(self) -> None:
        self._procedures: Dict[str, Procedure] = {}
        self._name_index: Dict[str, List[str]] = {}
        self._children: Dict[str, List[str]] = {}       # parent_id -> [child_ids]
        self._roots: List[str] = []
        # 倒排索引：token -> [procedure_ids]
        self._token_index: Dict[str, List[str]] = {}

    # ---------- CRUD ----------
    def add_procedure(self, procedure: Procedure) -> str:
        if procedure.id in self._procedures:
            raise ValueError(f"Procedure id '{procedure.id}' already exists")
        self._procedures[procedure.id] = procedure
        self._name_index.setdefault(procedure.name, []).append(procedure.id)
        # 层次关系
        if procedure.parent_id:
            if procedure.parent_id not in self._procedures:
                raise ValueError(f"Parent procedure '{procedure.parent_id}' not found")
            self._children.setdefault(procedure.parent_id, []).append(procedure.id)
        else:
            self._roots.append(procedure.id)
        # 触发词索引
        for token in procedure.trigger.pattern_tokens:
            self._token_index.setdefault(token, []).append(procedure.id)
        return procedure.id

    def get_procedure(self, procedure_id: str) -> Optional[Procedure]:
        return self._procedures.get(procedure_id)

    def find_by_name(self, name: str) -> List[Procedure]:
        ids = self._name_index.get(name, [])
        return [self._procedures[pid] for pid in ids if pid in self._procedures]

    def find_by_tokens(self, tokens: List[str]) -> List[Procedure]:
        """根据词素序列查找可能匹配的 procedure（重叠匹配度高的排前）"""
        score: Dict[str, int] = {}
        for token in tokens:
            for pid in self._token_index.get(token, []):
                score[pid] = score.get(pid, 0) + 1
        # 按触发词匹配比例排序
        def _ratio(pid: str) -> float:
            proc = self._procedures[pid]
            total = max(1, len(proc.trigger.pattern_tokens))
            return score[pid] / total
        sorted_ids = sorted(score.keys(), key=lambda pid: (-score[pid], -_ratio(pid)))
        return [self._procedures[pid] for pid in sorted_ids]

    def find_by_category(self, category: str) -> List[Procedure]:
        return [p for p in self._procedures.values() if p.category == category]

    def get_children(self, procedure_id: str) -> List[Procedure]:
        return [
            self._procedures[cid]
            for cid in self._children.get(procedure_id, [])
            if cid in self._procedures
        ]

    def get_dependencies(self, procedure_id: str) -> List[Procedure]:
        proc = self.get_procedure(procedure_id)
        if not proc:
            return []
        return [
            self._procedures[did]
            for did in proc.dependencies
            if did in self._procedures
        ]

    def update_procedure(self, procedure: Procedure) -> None:
        if procedure.id not in self._procedures:
            raise ValueError(f"Procedure id '{procedure.id}' does not exist")
        self.delete_procedure(procedure.id)
        self.add_procedure(procedure)

    def delete_procedure(self, procedure_id: str) -> bool:
        if procedure_id not in self._procedures:
            return False
        proc = self._procedures.pop(procedure_id)
        # name index
        if proc.name in self._name_index and procedure_id in self._name_index[proc.name]:
            self._name_index[proc.name].remove(procedure_id)
            if not self._name_index[proc.name]:
                del self._name_index[proc.name]
        # children index
        if proc.parent_id and proc.parent_id in self._children:
            if procedure_id in self._children[proc.parent_id]:
                self._children[proc.parent_id].remove(procedure_id)
        if procedure_id in self._children:
            del self._children[procedure_id]
        # roots
        if procedure_id in self._roots:
            self._roots.remove(procedure_id)
        # token index
        for token, pids in list(self._token_index.items()):
            if procedure_id in pids:
                pids.remove(procedure_id)
                if not pids:
                    del self._token_index[token]
        return True

    # ---------- 持久化 ----------
    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump([p.dict() for p in self._procedures.values()], f, ensure_ascii=False, indent=2)

    def load_json(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # 先按 parent_id 排序：无 parent 的先加载
            sorted_data = sorted(data, key=lambda d: (bool(d.get("parent_id")), d.get("parent_id") or ""))
            for pd in sorted_data:
                self.add_procedure(Procedure(**pd))

    @property
    def procedure_count(self) -> int:
        return len(self._procedures)

    def list_procedures(self) -> List[Procedure]:
        """返回所有程序性记忆的副本（按插入顺序）"""
        return list(self._procedures.values())

    @property
    def root_procedures(self) -> List[Procedure]:
        return [self._procedures[rid] for rid in self._roots if rid in self._procedures]
