"""MultiAgentOrchestrator TDD 테스트 (RED 단계).

그래프 기반 멀티에이전트 RAG 오케스트레이터가 공유 상태를 전달하고
성능 트래커를 갱신하는지 검증한다.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Dict

import pytest

from src.services.rag.graph_index_store import GraphIndexStore
from src.services.rag.multi_agent_orchestrator import (
    AgentDefinition,
    MultiAgentOrchestrator,
)
from src.services.rag.performance_tracker import PerformanceTracker


@dataclass
class DummyAgent:
    """간단한 더미 에이전트."""

    name: str
    role: str
    response: str

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        calls = state.setdefault("calls", [])
        calls.append(self.name)
        return {
            "content": f"{self.response} | context={len(state['graph']['nodes'])}",
            "prompt_tokens": 120,
            "completion_tokens": 40,
        }


@pytest.mark.asyncio
async def test_orchestrator_runs_agents_and_tracks_latency() -> None:
    """에이전트를 순서대로 실행하고 성능 메트릭을 남겨야 한다."""

    graph_store = GraphIndexStore()
    graph_store.add_node("doc-1", "document", "Some content", {})
    graph_store.add_node("concept-1", "concept", "Named Entity", {})
    graph_store.add_edge("doc-1", "concept-1", "mentions")

    tracker = PerformanceTracker()
    orchestrator = MultiAgentOrchestrator(graph_store=graph_store, tracker=tracker)

    agents = [
        AgentDefinition(
            name="retriever",
            role="retrieval",
            handler=DummyAgent("retriever", "retrieval", "retrieved").run,
            model="gpt-4",
        ),
        AgentDefinition(
            name="reader",
            role="analysis",
            handler=DummyAgent("reader", "analysis", "analyzed").run,
            model="gpt-4",
        ),
    ]

    result = await orchestrator.run(
        query="Explain the key concept",
        seed_node_ids=["doc-1"],
        agents=agents,
        max_depth=1,
        max_nodes=3,
    )

    assert result.query == "Explain the key concept"
    assert len(result.graph["nodes"]) == 2
    assert [entry.name for entry in result.agent_results] == ["retriever", "reader"]
    assert result.agent_results[-1].output.startswith("analyzed")

    metrics = tracker.get_metrics()
    assert "agent:retriever" in metrics["latency"]
    assert "agent:reader" in metrics["latency"]


@pytest.mark.asyncio
async def test_orchestrator_enriches_shared_state_between_agents() -> None:
    """첫 번째 에이전트의 결과가 두 번째 에이전트에 공유된다."""

    graph_store = GraphIndexStore()
    graph_store.add_node("doc-1", "document", "Doc", {})
    graph_store.add_node("concept-1", "concept", "Concept", {})
    graph_store.add_edge("doc-1", "concept-1", "mentions")

    tracker = PerformanceTracker()
    orchestrator = MultiAgentOrchestrator(graph_store=graph_store, tracker=tracker)

    class RecordingAgent(DummyAgent):
        async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
            state.setdefault("records", {})[self.name] = list(state.get("calls", []))
            return await super().run(state)

    agents = [
        AgentDefinition(
            name="retriever",
            role="retrieval",
            handler=RecordingAgent("retriever", "retrieval", "retrieved").run,
            model="gpt-4",
        ),
        AgentDefinition(
            name="critic",
            role="critique",
            handler=RecordingAgent("critic", "critique", "critique").run,
            model="gpt-4",
        ),
    ]

    result = await orchestrator.run(
        query="Summarize",
        seed_node_ids=["doc-1"],
        agents=agents,
        max_depth=1,
        max_nodes=3,
    )

    # critic는 retriever가 먼저 호출된 사실을 state.records에 남긴다.
    critic_record = result.shared_state["records"]["critic"]
    assert critic_record == ["retriever"]
    assert result.agent_results[0].output.startswith("retrieved")
    assert result.agent_results[1].output.startswith("critique")
