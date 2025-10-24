"""RAGManager integration with GraphRAGPipeline (TDD)."""

import pytest

from src.services.rag import (
    AgentDefinition,
    AgentRunResult,
    GraphIndexStore,
    GraphRAGPipelineResult,
    GraphSeedSelector,
    MultiAgentRunResult,
    RAGManager,
)


class DummyEmbeddingService:
    async def embed_text(self, text: str):  # pragma: no cover - unused in tests
        return [0.0]

    def get_model_info(self):  # pragma: no cover - unused
        return {"provider": "dummy"}


class StubOrchestrator:
    def __init__(self, result: MultiAgentRunResult) -> None:
        self.result = result
        self.calls = []

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


@pytest.mark.asyncio
async def test_rag_manager_runs_graph_pipeline_when_configured() -> None:
    graph_store = GraphIndexStore()
    graph_store.add_node(
        "paper-gnn",
        "document",
        "Graph neural networks boost retrieval accuracy.",
        {"title": "Graph Retrieval"},
    )
    graph_store.add_node(
        "paper-transformers",
        "document",
        "Transformers excel at sequence modeling tasks.",
        {"title": "Transformers"},
    )
    graph_store.add_edge("paper-gnn", "paper-transformers", "cites")

    seed_selector = GraphSeedSelector(graph_store)

    orchestrator_result = MultiAgentRunResult(
        query="How do graphs improve RAG?",
        graph={"nodes": {}, "edges": []},
        agent_results=[
            AgentRunResult(
                name="retriever",
                role="retrieval",
                output="context",
                prompt_tokens=50,
                completion_tokens=10,
                latency_ms=25.0,
            )
        ],
        final_answer="Graphs add structure",
        shared_state={},
    )
    orchestrator = StubOrchestrator(orchestrator_result)

    manager = RAGManager(
        embedding_service=DummyEmbeddingService(),
        chromadb_mode="disabled",
        graph_store=graph_store,
        graph_seed_selector=seed_selector,
        graph_orchestrator=orchestrator,
    )

    async def dummy_handler(state):  # pragma: no cover
        return {"content": "", "prompt_tokens": 0, "completion_tokens": 0}

    agents = [AgentDefinition(name="retriever", role="retrieval", handler=dummy_handler)]

    result = await manager.run_graph_rag(
        query="Graph retrieval for RAG",
        agents=agents,
        seed_limit=2,
        max_depth=1,
        max_nodes=3,
    )

    assert isinstance(result, GraphRAGPipelineResult)
    assert result.final_answer == "Graphs add structure"
    assert result.seeds[:1] == ["paper-gnn"]

    assert len(orchestrator.calls) == 1
    call_kwargs = orchestrator.calls[0]
    assert call_kwargs["seed_node_ids"] == result.seeds
    assert call_kwargs["agents"] == agents


@pytest.mark.asyncio
async def test_rag_manager_graph_pipeline_missing_raises() -> None:
    manager = RAGManager(
        embedding_service=DummyEmbeddingService(),
        chromadb_mode="disabled",
    )

    with pytest.raises(RuntimeError):
        await manager.run_graph_rag(query="test", agents=[])

