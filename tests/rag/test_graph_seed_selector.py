"""GraphSeedSelector TDD 테스트."""

import pytest

from src.services.rag.graph_index_store import GraphIndexStore
from src.services.rag.graph_seed_selector import GraphSeedSelector


@pytest.fixture
def populated_graph() -> GraphIndexStore:
    graph = GraphIndexStore()
    graph.add_node(
        "paper-1",
        "document",
        "Graph neural networks improve document retrieval.",
        {"title": "GNN for Retrieval", "keywords": ["graph", "retrieval"]},
    )
    graph.add_node(
        "paper-2",
        "document",
        "Transformers excel at sequence modeling tasks.",
        {"title": "Transformers Overview"},
    )
    graph.add_node(
        "concept-rag",
        "concept",
        "Retrieval-augmented generation combines search and generation.",
        {"aliases": ["RAG"]},
    )
    graph.add_edge("paper-1", "concept-rag", "refers_to")
    graph.add_edge("paper-2", "concept-rag", "refers_to")
    return graph


def test_graph_seed_selector_ranks_by_token_overlap(populated_graph: GraphIndexStore) -> None:
    selector = GraphSeedSelector(populated_graph)

    seeds = selector.suggest_seeds(
        query="How do graph-based retrieval methods enhance RAG pipelines?",
        limit=2,
    )

    assert seeds == ["paper-1", "concept-rag"]


def test_graph_seed_selector_handles_no_matches(populated_graph: GraphIndexStore) -> None:
    selector = GraphSeedSelector(populated_graph)

    seeds = selector.suggest_seeds(
        query="Quantum annealing for optimization",
        limit=2,
    )

    # Should gracefully fall back to the most informative nodes (by degree/content length)
    assert len(seeds) == 2
    assert "paper-1" in seeds or "paper-2" in seeds

