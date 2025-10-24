"""Empirical benchmarking helpers for graph RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol

from src.services.rag.multi_agent_orchestrator import AgentDefinition


@dataclass
class EmpiricalBenchmarkResult:
    query_count: int
    avg_agent_latency_ms: float
    avg_total_tokens: float
    total_prompt_tokens: int
    total_completion_tokens: int
    avg_seed_count: float
    simulation_latency_ms: Optional[float]
    latency_gap_ms: Optional[float]


class GraphRAGRunner(Protocol):
    async def run_graph_rag(
        self,
        query: str,
        agents: List[AgentDefinition],
        seed_limit: int = 5,
        max_depth: int = 2,
        max_nodes: int = 50,
    ):
        ...


async def benchmark_graph_pipeline(
    runner: GraphRAGRunner,
    queries: List[str],
    agents: List[AgentDefinition],
    seed_limit: int = 5,
    max_depth: int = 2,
    max_nodes: int = 50,
    simulation_latency_ms: Optional[float] = None,
) -> EmpiricalBenchmarkResult:
    """Run graph pipeline across queries and aggregate metrics."""

    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_seed_count = 0

    for query in queries:
        result = await runner.run_graph_rag(
            query=query,
            agents=agents,
            seed_limit=seed_limit,
            max_depth=max_depth,
            max_nodes=max_nodes,
        )

        total_seed_count += len(result.seeds)

        agent_latency = sum(agent.latency_ms for agent in result.orchestrator_result.agent_results)
        total_latency += agent_latency

        total_prompt_tokens += sum(agent.prompt_tokens for agent in result.orchestrator_result.agent_results)
        total_completion_tokens += sum(agent.completion_tokens for agent in result.orchestrator_result.agent_results)

    query_count = len(queries)

    avg_agent_latency_ms = total_latency / query_count if query_count else 0.0
    avg_total_tokens = (
        (total_prompt_tokens + total_completion_tokens) / query_count if query_count else 0.0
    )
    avg_seed_count = total_seed_count / query_count if query_count else 0.0

    latency_gap = (
        avg_agent_latency_ms - simulation_latency_ms if simulation_latency_ms is not None else None
    )

    return EmpiricalBenchmarkResult(
        query_count=query_count,
        avg_agent_latency_ms=avg_agent_latency_ms,
        avg_total_tokens=avg_total_tokens,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        avg_seed_count=avg_seed_count,
        simulation_latency_ms=simulation_latency_ms,
        latency_gap_ms=latency_gap,
    )


__all__ = ["EmpiricalBenchmarkResult", "benchmark_graph_pipeline"]

