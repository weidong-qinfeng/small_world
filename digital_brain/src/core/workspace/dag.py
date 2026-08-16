"""DAG 数据结构 - 问题理解的结构化表示

对应详细设计/推理执行设计.md §2-3：
    - DAGNode: 一个原子操作节点
    - DAGGraph: 有向无环图
    - DAGBuildResult: 构建结果（成功/失败+缺失清单）
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


# 原子操作类型常量（L0 硬件指令集）
OP_READ_MEMORY = "read_memory"
OP_WRITE_MEMORY = "write_memory"
OP_SEARCH_CONTEXT = "search_context"
OP_CALL_ALGORITHM = "call_algorithm"
OP_RETURN_VALUE = "return_value"

ATOMIC_OPS = frozenset({
    OP_READ_MEMORY, OP_WRITE_MEMORY, OP_SEARCH_CONTEXT,
    OP_CALL_ALGORITHM, OP_RETURN_VALUE,
})


class DAGNode(BaseModel):
    """DAG节点 - 一个原子操作"""
    id: str
    action: str               # read_memory/write_memory/search_context/call_algorithm/return_value
    params: Dict[str, Any] = Field(default_factory=dict)
    depends_on: List[str] = Field(default_factory=list)  # 上游节点id列表
    description: str = ""

    class Config:
        use_enum_values = True


class DAGGraph:
    """有向无环图"""

    def __init__(self) -> None:
        self.nodes: Dict[str, DAGNode] = {}
        self._order: List[str] = []  # 插入顺序（拓扑序 tie-break 用）

    def add_node(self, node: DAGNode) -> None:
        self.nodes[node.id] = node
        self._order.append(node.id)

    def get_node(self, node_id: str) -> Optional[DAGNode]:
        return self.nodes.get(node_id)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    def topological_sort(self) -> Tuple[List[str], bool]:
        """Kahn拓扑排序。返回(序列, 是否成功)。

        同层就绪节点按插入序 tie-break，保证确定性。
        """
        in_degree: Dict[str, int] = {nid: 0 for nid in self.nodes}
        for node in self.nodes.values():
            for dep in node.depends_on:
                if dep in in_degree:
                    in_degree[node.id] = in_degree.get(node.id, 0) + 1
        # 入度为0的节点入队（按插入序保证确定性）
        queue: List[str] = [nid for nid in self._order if in_degree.get(nid, 0) == 0]
        result: List[str] = []
        while queue:
            nid = queue.pop(0)
            result.append(nid)
            # 找所有依赖nid的节点，入度减1
            for node in self.nodes.values():
                if nid in node.depends_on and node.id not in result:
                    in_degree[node.id] -= 1
                    if in_degree[node.id] == 0 and node.id not in queue:
                        queue.append(node.id)
        success = len(result) == len(self.nodes)
        return result, success

    def has_cycle(self) -> bool:
        _, ok = self.topological_sort()
        return not ok

    def __repr__(self) -> str:
        return f"DAGGraph(nodes={self.node_count})"


class DAGBuildResult(BaseModel):
    """DAG构建结果"""
    success: bool
    dag: Optional[Any] = None       # DAGGraph 实例
    failure_reason: str = ""
    failure_type: str = ""          # missing_dependency / cycle / algorithm_not_learned / no_pattern
    missing: List[str] = Field(default_factory=list)  # 缺失的实体/属性/算法
    node_count: int = 0
    pattern_count: int = 0          # 命中的模式数

    class Config:
        arbitrary_types_allowed = True
