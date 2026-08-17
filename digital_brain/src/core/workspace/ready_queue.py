"""就绪操作队列 - 维护参数齐备且依赖满足的操作，按拓扑序执行

Phase 5c 新增：工作记忆的子组件，用于 Phase 5d 替代 DAGExecutor。
当前阶段（5c）仅实现数据结构，不参与主执行路径。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class ReadyOperation:
    """一个就绪的操作（对应 operation 神经元激活到可执行状态）"""

    def __init__(
        self,
        op_id: str,                     # operation 神经元实体 id
        op_type: str,                   # write_memory / read_memory / search_context / call_algorithm / return_value
        params: Dict[str, Any],         # 参数齐备的 dict
        depends_on: Optional[List[str]] = None,  # 上游操作 op_id
        description: str = "",
        source_pattern: str = "",       # 激活来源的 pattern 名（调试用）
    ) -> None:
        self.op_id = op_id
        self.op_type = op_type
        self.params = params
        self.depends_on = list(depends_on or [])
        self.description = description
        self.source_pattern = source_pattern
        self.completed: bool = False
        self.output: Any = None

    def __repr__(self) -> str:
        return (
            f"ReadyOp({self.op_type}, params={self.params}, "
            f"done={self.completed}, out={self.output})"
        )


class ReadyQueue:
    """操作神经元的就绪队列

    Phase 5d 时：替代 DAGExecutor，按 requires 拓扑执行。
    Phase 5c 时：仅实现 CRUD + 基本调度，不在主求解路径中启用。
    """

    def __init__(self) -> None:
        self._ops: Dict[str, ReadyOperation] = {}
        self._order: List[str] = []   # 登记顺序（用于 tie-break）

    # ---- CRUD ----
    def enqueue(self, op: ReadyOperation) -> None:
        if op.op_id in self._ops:
            return  # 幂等
        self._ops[op.op_id] = op
        self._order.append(op.op_id)

    def get(self, op_id: str) -> Optional[ReadyOperation]:
        return self._ops.get(op_id)

    def mark_completed(self, op_id: str, output: Any = None) -> None:
        op = self._ops.get(op_id)
        if op is not None:
            op.completed = True
            op.output = output

    # ---- 调度：返回依赖全部满足的、未完成的 op_id 列表（按登记序）----
    def next_ready(self) -> List[str]:
        result: List[str] = []
        for op_id in self._order:
            op = self._ops[op_id]
            if op.completed:
                continue
            deps_ok = all(
                self._ops.get(dep) and self._ops[dep].completed
                for dep in op.depends_on
            )
            if deps_ok:
                result.append(op_id)
        return result

    def all_completed(self) -> bool:
        return all(op.completed for op in self._ops.values())

    def op_count(self) -> int:
        return len(self._ops)

    def completed_count(self) -> int:
        return sum(1 for op in self._ops.values() if op.completed)

    def clear(self) -> None:
        self._ops.clear()
        self._order.clear()

    def __len__(self) -> int:
        return len(self._ops)

    def __repr__(self) -> str:
        return (
            f"ReadyQueue(total={self.op_count()}, "
            f"completed={self.completed_count()})"
        )
