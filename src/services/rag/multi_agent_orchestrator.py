"""Multi-agent orchestration for graph-aware RAG.

TDD 단계에서 기본 기능(에이전트 실행 순서, 공유 상태, 성능 추적)을
구현하며 추후 실제 LLM 연동 및 고급 정책으로 확장한다.
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

from src.services.rag.graph_index_store import GraphIndexStore, GraphSubgraph
from src.services.rag.performance_tracker import PerformanceTracker


HandlerType = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]


@dataclass
class AgentDefinition:
    """에이전트 실행 정의."""

    name: str
    role: str
    handler: HandlerType
    model: str = "gpt-5"


@dataclass
class AgentRunResult:
    """단일 에이전트 실행 결과."""

    name: str
    role: str
    output: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


@dataclass
class MultiAgentRunResult:
    """멀티에이전트 실행 최종 결과."""

    query: str
    graph: Dict[str, Any]
    agent_results: List[AgentRunResult]
    final_answer: str
    shared_state: Dict[str, Any]


class MultiAgentOrchestrator:
    """멀티에이전트/그래프 기반 RAG 오케스트레이터."""

    def __init__(
        self,
        graph_store: GraphIndexStore,
        tracker: Optional[PerformanceTracker] = None,
    ) -> None:
        self.graph_store = graph_store
        self.tracker = tracker

    async def run(
        self,
        query: str,
        seed_node_ids: List[str],
        agents: List[AgentDefinition],
        max_depth: int = 2,
        max_nodes: int = 50,
    ) -> MultiAgentRunResult:
        """시드 노드와 에이전트 구성을 받아 멀티에이전트 파이프라인을 실행한다."""

        subgraph: GraphSubgraph = self.graph_store.extract_subgraph(
            seed_node_ids=seed_node_ids,
            max_depth=max_depth,
            max_nodes=max_nodes,
        )

        shared_state: Dict[str, Any] = {
            "query": query,
            "graph": {"nodes": subgraph.nodes, "edges": subgraph.edges},
            "agent_outputs": {},
        }

        agent_results: List[AgentRunResult] = []
        final_answer = ""

        for agent in agents:
            tracker_cm = (
                self.tracker.track_latency(f"agent:{agent.name}")
                if self.tracker
                else nullcontext()
            )

            start_time = time.perf_counter()
            with tracker_cm:  # type: ignore[arg-type]
                result_payload = await agent.handler(shared_state)
            latency_ms = (time.perf_counter() - start_time) * 1000

            output_text = result_payload.get("content", "")
            prompt_tokens = int(result_payload.get("prompt_tokens", 0))
            completion_tokens = int(result_payload.get("completion_tokens", 0))

            shared_state.setdefault("agent_outputs", {})[agent.name] = result_payload
            final_answer = output_text or final_answer

            if self.tracker and (prompt_tokens or completion_tokens):
                self.tracker.track_tokens(
                    model=agent.model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

            agent_results.append(
                AgentRunResult(
                    name=agent.name,
                    role=agent.role,
                    output=output_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=latency_ms,
                )
            )

        return MultiAgentRunResult(
            query=query,
            graph={"nodes": subgraph.nodes, "edges": subgraph.edges},
            agent_results=agent_results,
            final_answer=final_answer,
            shared_state=shared_state,
        )


__all__ = [
    "AgentDefinition",
    "AgentRunResult",
    "MultiAgentRunResult",
    "MultiAgentOrchestrator",
]
