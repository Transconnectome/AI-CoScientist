"""Empirical benchmarking of graph pipeline (TDD)."""

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
from src.services.rag.benchmarking import EmpiricalBenchmarkResult, benchmark_graph_pipeline


class DummyEmbeddingService:
    async def embed_text(self, text: str):  # pragma: no cover
        return [0.0]

    def get_model_info(self):  # pragma: no cover
        return {}


class StubOrchestrator:
    def __init__(self, latencies, prompt_tokens, completion_tokens):
        self.latencies = latencies
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.calls = 0

    async def run(self, **kwargs):
        idx = self.calls % len(self.latencies)
        self.calls += 1
        agent_result = AgentRunResult(
            name="retriever",
            role="retrieval",
            output="context",
            prompt_tokens=self.prompt_tokens[idx],
            completion_tokens=self.completion_tokens[idx],
            latency_ms=self.latencies[idx],
        )
        return MultiAgentRunResult(
            query=kwargs["query"],
            graph={"nodes": {}, "edges": []},
            agent_results=[agent_result],
            final_answer="answer",
            shared_state={},
        )


@pytest.mark.asyncio
async def test_benchmark_collects_latency_and_tokens():
    graph_store = GraphIndexStore()
    graph_store.add_node("paper-1", "document", "Graph retrieval", {"keywords": ["graph"]})
    seed_selector = GraphSeedSelector(graph_store)

    orchestrator = StubOrchestrator(latencies=[40.0, 60.0], prompt_tokens=[100, 120], completion_tokens=[20, 30])

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

    result = await benchmark_graph_pipeline(
        runner=manager,
        queries=["graph retrieval", "graph reasoning"],
        agents=agents,
        simulation_latency_ms=55.0,
    )

    assert isinstance(result, EmpiricalBenchmarkResult)
    assert result.query_count == 2
    assert result.avg_agent_latency_ms == pytest.approx(50.0, rel=1e-3)
    assert result.total_prompt_tokens == 220
    assert result.total_completion_tokens == 50
    assert result.latency_gap_ms == pytest.approx(-5.0, rel=1e-3)


@pytest.mark.asyncio
async def test_benchmark_handles_no_agent_results():
    graph_store = GraphIndexStore()
    seed_selector = GraphSeedSelector(graph_store)

    class EmptyOrchestrator:
        async def run(self, **kwargs):
            return MultiAgentRunResult(
                query=kwargs["query"],
                graph={"nodes": {}, "edges": []},
                agent_results=[],
                final_answer="",
                shared_state={},
            )

    manager = RAGManager(
        embedding_service=DummyEmbeddingService(),
        chromadb_mode="disabled",
        graph_store=graph_store,
        graph_seed_selector=seed_selector,
        graph_orchestrator=EmptyOrchestrator(),
    )

    result = await benchmark_graph_pipeline(runner=manager, queries=["test"], agents=[])

    assert result.avg_agent_latency_ms == 0.0
    assert result.total_prompt_tokens == 0
    assert result.total_completion_tokens == 0
