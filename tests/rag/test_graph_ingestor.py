"""GraphDocumentIngestor TDD tests."""

import pytest

from src.services.rag import GraphDocumentIngestor, GraphIndexStore, GraphSeedSelector


@pytest.fixture
def ingest_setup():
    store = GraphIndexStore()
    ingestor = GraphDocumentIngestor(store)
    return store, ingestor


def test_ingestor_creates_document_sections_and_edges(ingest_setup):
    store, ingestor = ingest_setup

    ingestor.ingest_paper(
        paper_id="paper-42",
        title="Graph retrieval improves reasoning",
        abstract="We explore graph-based retrieval for large language models.",
        sections=[
            {"name": "Introduction", "content": "Graphs capture structure."},
            {"name": "Method", "content": "We build a retrieval graph."},
        ],
        keywords=["graph retrieval", "reasoning"],
        citations=["arXiv:2501.00001"],
    )

    doc_node = store.get_node("paper-42")
    assert doc_node.node_type == "document"
    assert "Graph retrieval" in doc_node.content
    assert "graph retrieval" in doc_node.metadata["keywords"]

    section_nodes = [node for node in store.iter_nodes() if node.node_type == "section"]
    assert len(section_nodes) == 2
    assert any("Introduction" in node.metadata["name"] for node in section_nodes)

    edges = store.get_neighbors("paper-42")
    assert any(edge.relation == "has_section" for edge in edges)
    assert any(edge.relation == "cites" for edge in edges)


def test_ingestor_reuses_concept_nodes(ingest_setup):
    store, ingestor = ingest_setup

    ingestor.ingest_paper(
        paper_id="paper-a",
        title="Graph retrieval",
        abstract="",
        sections=[],
        keywords=["Graph Retrieval"],
        citations=[],
    )

    ingestor.ingest_paper(
        paper_id="paper-b",
        title="Another study",
        abstract="",
        sections=[],
        keywords=["graph retrieval"],
        citations=[],
    )

    concept_nodes = [node for node in store.iter_nodes() if node.node_type == "concept"]
    assert len(concept_nodes) == 1


def test_seed_selector_prefers_ingested_document(ingest_setup):
    store, ingestor = ingest_setup

    ingestor.ingest_paper(
        paper_id="paper-gnn",
        title="Graph retrieval improves rag",
        abstract="We propose a graph-based rag architecture.",
        sections=[],
        keywords=["graph-based rag"],
        citations=[],
    )

    selector = GraphSeedSelector(store)
    seeds = selector.suggest_seeds("graph based rag improvements", limit=1)
    assert seeds == ["paper-gnn"]

