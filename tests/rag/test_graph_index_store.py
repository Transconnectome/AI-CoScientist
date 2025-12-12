"""GraphIndexStore 테스트 (TDD - RED 단계).

우선순위 1인 멀티에이전트/그래프 기반 RAG를 위해 그래프 인덱스 계층이
필요하다. 본 테스트는 최소한의 그래프 기능(노드/엣지 추가, 이웃 조회,
서브그래프 추출)을 검증한다.
"""

import pytest

from src.services.rag.graph_index_store import (
    GraphEdge,
    GraphIndexStore,
    GraphNode,
    GraphSubgraph,
)


@pytest.fixture
def graph_store() -> GraphIndexStore:
    """그래프 인덱스 스토어 초기화."""

    return GraphIndexStore()


def test_add_node_and_get_node(graph_store: GraphIndexStore) -> None:
    """노드를 추가하면 동일한 ID로 조회할 수 있어야 한다."""

    graph_store.add_node(
        node_id="doc-1",
        node_type="document",
        content="Deep learning improves accuracy.",
        metadata={"title": "Sample Paper", "year": 2025},
    )

    node: GraphNode = graph_store.get_node("doc-1")

    assert node.node_id == "doc-1"
    assert node.node_type == "document"
    assert node.metadata["title"] == "Sample Paper"
    assert "Deep learning" in node.content


def test_add_edge_and_neighbors(graph_store: GraphIndexStore) -> None:
    """노드 간 엣지 추가 후 이웃 조회 검증."""

    graph_store.add_node("doc-1", "document", "Doc text", {})
    graph_store.add_node("concept-ner", "concept", "NER", {})

    graph_store.add_edge(
        source_id="doc-1",
        target_id="concept-ner",
        relation="mentions",
        weight=0.8,
    )

    neighbors = graph_store.get_neighbors("doc-1")

    assert len(neighbors) == 1
    edge: GraphEdge = neighbors[0]
    assert edge.source_id == "doc-1"
    assert edge.target_id == "concept-ner"
    assert pytest.approx(edge.weight, 0.01) == 0.8


def test_extract_subgraph_limits_depth_and_size(graph_store: GraphIndexStore) -> None:
    """서브그래프 추출 시 깊이/노드 수 제한을 준수해야 한다."""

    graph_store.add_node("doc-1", "document", "Doc", {})
    graph_store.add_node("concept-ner", "concept", "NER", {})
    graph_store.add_node("paper-2", "document", "Paper", {})

    graph_store.add_edge("doc-1", "concept-ner", "mentions")
    graph_store.add_edge("concept-ner", "paper-2", "supported_by")

    subgraph: GraphSubgraph = graph_store.extract_subgraph(
        seed_node_ids=["doc-1"],
        max_depth=1,
        max_nodes=2,
    )

    assert len(subgraph.nodes) == 2
    assert "doc-1" in subgraph.nodes
    assert "concept-ner" in subgraph.nodes
    assert "paper-2" not in subgraph.nodes

    assert len(subgraph.edges) == 1
    assert subgraph.edges[0].target_id == "concept-ner"

