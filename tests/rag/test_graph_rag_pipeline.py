"""GraphRAGPipeline TDD 테스트."""

import pytest

from src.services.rag import (
    AgentDefinition,
    GraphIndexStore,
    GraphRAGPipeline,
    GraphSeedSelector,
    MultiAgentRunResult,
    AgentRunResult,
)


class StubOrchestrator:
    def __init__(self, result: MultiAgentRunResult) -> None:
        self.result = result
        self.calls = []

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


@pytest.mark.asyncio
async def test_pipeline_selects_seeds_and_calls_orchestrator() -> None:
    graph = GraphIndexStore()
    graph.add_node(
        "paper-1",
        "document",
        "Graph retrieval enhances RAG.",
        {"title": "Graph RAG"},
    )
    graph.add_node(
        "paper-2",
        "document",
        "Transformers for sequence modeling.",
        {"title": "Transformers"},
    )
    graph.add_edge("paper-1", "paper-2", "cites")

    selector = GraphSeedSelector(graph)

    orchestrator_result = MultiAgentRunResult(
        query="How do graphs help RAG?",
        graph={"nodes": {}, "edges": []},
        agent_results=[
            AgentRunResult(
                name="retriever",
                role="retrieval",
                output="retrieved context",
                prompt_tokens=100,
                completion_tokens=10,
                latency_ms=50.0,
            )
        ],
        final_answer="graphs add structure",
        shared_state={"agent_outputs": {}},
    )

    orchestrator = StubOrchestrator(orchestrator_result)

    pipeline = GraphRAGPipeline(
        graph_store=graph,
        seed_selector=selector,
        orchestrator=orchestrator,
    )

    async def dummy_handler(state):  # pragma: no cover - orchestrator stub bypasses
        return {"content": "", "prompt_tokens": 0, "completion_tokens": 0}

    agents = [AgentDefinition(name="retriever", role="retrieval", handler=dummy_handler)]

    result = await pipeline.run(
        query="How do graph retrieval methods enhance RAG?",
        agents=agents,
        seed_limit=2,
        max_depth=1,
        max_nodes=5,
    )

    assert result.final_answer == "graphs add structure"
    assert result.seeds == ["paper-1", "paper-2"]
    assert len(orchestrator.calls) == 1
    call_kwargs = orchestrator.calls[0]
    assert call_kwargs["seed_node_ids"] == ["paper-1", "paper-2"]
    assert call_kwargs["agents"] == agents


@pytest.mark.asyncio
async def test_pipeline_handles_empty_graph() -> None:
    graph = GraphIndexStore()
    selector = GraphSeedSelector(graph)

    orchestrator_result = MultiAgentRunResult(
        query="Any",
        graph={"nodes": {}, "edges": []},
        agent_results=[],
        final_answer="",
        shared_state={},
    )

    orchestrator = StubOrchestrator(orchestrator_result)
    pipeline = GraphRAGPipeline(graph, selector, orchestrator)

    result = await pipeline.run(query="test", agents=[])

    assert result.seeds == []
    assert orchestrator.calls[0]["seed_node_ids"] == []
