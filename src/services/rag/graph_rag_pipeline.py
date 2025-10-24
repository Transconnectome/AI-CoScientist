"""Graph-aware multi-agent RAG pipeline orchestration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.services.rag.graph_index_store import GraphIndexStore
from src.services.rag.graph_seed_selector import GraphSeedSelector
from src.services.rag.multi_agent_orchestrator import (
    AgentDefinition,
    MultiAgentOrchestrator,
    MultiAgentRunResult,
)


@dataclass
class GraphRAGPipelineResult:
    query: str
    seeds: List[str]
    orchestrator_result: MultiAgentRunResult

    @property
    def final_answer(self) -> str:
        return self.orchestrator_result.final_answer


class GraphRAGPipeline:
    """Convenience wrapper combining seed selection and multi-agent execution."""

    def __init__(
        self,
        graph_store: GraphIndexStore,
        seed_selector: GraphSeedSelector,
        orchestrator: MultiAgentOrchestrator,
    ) -> None:
        self.graph_store = graph_store
        self.seed_selector = seed_selector
        self.orchestrator = orchestrator

    async def run(
        self,
        query: str,
        agents: List[AgentDefinition],
        seed_limit: int = 5,
        max_depth: int = 2,
        max_nodes: int = 50,
    ) -> GraphRAGPipelineResult:
        seeds = self.seed_selector.suggest_seeds(query, limit=seed_limit)

        result = await self.orchestrator.run(
            query=query,
            seed_node_ids=seeds,
            agents=agents,
            max_depth=max_depth,
            max_nodes=max_nodes,
        )

        return GraphRAGPipelineResult(
            query=query,
            seeds=seeds,
            orchestrator_result=result,
        )


__all__ = ["GraphRAGPipeline", "GraphRAGPipelineResult"]

