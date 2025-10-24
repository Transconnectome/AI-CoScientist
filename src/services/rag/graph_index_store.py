"""In-memory knowledge graph index for multi-agent RAG.

이 모듈은 우선순위 1 구현 단계에서 사용할 경량 그래프 계층을 제공한다.
노드/엣지 관리와 간단한 서브그래프 추출 기능을 포함하며, 추후 실제
데이터베이스나 GNN 백엔드로 대체/확장될 수 있다.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class GraphNode:
    """단일 그래프 노드 표현."""

    node_id: str
    node_type: str
    content: str
    metadata: Dict[str, object]


@dataclass(frozen=True)
class GraphEdge:
    """노드 간 관계."""

    source_id: str
    target_id: str
    relation: str
    weight: float = 1.0


@dataclass
class GraphSubgraph:
    """서브그래프 결과."""

    nodes: Dict[str, GraphNode]
    edges: List[GraphEdge]


class GraphIndexStore:
    """간단한 로컬 그래프 인덱스."""

    def __init__(self) -> None:
        self._nodes: Dict[str, GraphNode] = {}
        self._adjacency: Dict[str, List[GraphEdge]] = {}

    # ------------------------------------------------------------------
    # 노드/엣지 관리
    # ------------------------------------------------------------------
    def add_node(
        self,
        node_id: str,
        node_type: str,
        content: str,
        metadata: Optional[Dict[str, object]] = None,
    ) -> GraphNode:
        """노드를 추가하거나 업데이트한다."""

        node = GraphNode(
            node_id=node_id,
            node_type=node_type,
            content=content,
            metadata=metadata or {},
        )
        self._nodes[node_id] = node
        self._adjacency.setdefault(node_id, [])
        return node

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        weight: float = 1.0,
    ) -> GraphEdge:
        """단방향 엣지를 추가한다."""

        if source_id not in self._nodes:
            raise KeyError(f"Source node '{source_id}' not found")
        if target_id not in self._nodes:
            raise KeyError(f"Target node '{target_id}' not found")

        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            weight=weight,
        )
        self._adjacency[source_id].append(edge)
        return edge

    # ------------------------------------------------------------------
    # 조회
    # ------------------------------------------------------------------
    def get_node(self, node_id: str) -> GraphNode:
        """노드를 조회한다."""

        if node_id not in self._nodes:
            raise KeyError(f"Node '{node_id}' not found")
        return self._nodes[node_id]

    def get_neighbors(self, node_id: str) -> List[GraphEdge]:
        """이웃 엣지를 반환한다."""

        return list(self._adjacency.get(node_id, []))

    def iter_nodes(self) -> Iterable[GraphNode]:
        """그래프의 모든 노드를 순회한다."""

        return self._nodes.values()

    # ------------------------------------------------------------------
    # 서브그래프 추출
    # ------------------------------------------------------------------
    def extract_subgraph(
        self,
        seed_node_ids: Iterable[str],
        max_depth: int = 1,
        max_nodes: int = 50,
    ) -> GraphSubgraph:
        """시드 노드로부터 BFS 방식으로 서브그래프를 수집한다."""

        visited: Dict[str, GraphNode] = {}
        collected_edges: List[GraphEdge] = []

        queue: deque[tuple[str, int]] = deque()
        for node_id in seed_node_ids:
            if node_id in self._nodes:
                queue.append((node_id, 0))

        while queue and len(visited) < max_nodes:
            current_id, depth = queue.popleft()

            if current_id in visited:
                continue

            if current_id not in self._nodes:
                continue

            visited[current_id] = self._nodes[current_id]

            if depth >= max_depth:
                continue

            for edge in self._adjacency.get(current_id, []):
                collected_edges.append(edge)
                if edge.target_id not in visited and len(visited) < max_nodes:
                    queue.append((edge.target_id, depth + 1))

        # 엣지 정제: 방문하지 않은 노드를 향하는 엣지는 제외
        filtered_edges = [
            edge for edge in collected_edges if edge.source_id in visited and edge.target_id in visited
        ]

        return GraphSubgraph(nodes=visited, edges=filtered_edges)


__all__ = [
    "GraphIndexStore",
    "GraphNode",
    "GraphEdge",
    "GraphSubgraph",
]
