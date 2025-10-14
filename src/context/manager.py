# src/context/manager.py
from typing import Dict, List, Optional
from src.context.types import Insight, ResearchSession
from uuid import uuid4

class ResearchContextManager:
    """Manages research context and insights"""

    def __init__(self, vector_store, graph_db):
        self.vector_store = vector_store
        self.graph_db = graph_db

        # In-memory storage for Tier 1 (will add vector/graph in Tier 2)
        self.insights: Dict[str, Insight] = {}
        self.current_session = ResearchSession()

    async def store_insight(
        self,
        insight: Insight,
        source_agent: str,
        task_id: str,
        metadata: Dict
    ) -> str:
        """Store insight with provenance"""

        node_id = str(uuid4())

        # Store in memory
        self.insights[node_id] = insight
        self.current_session.insights.append(insight)

        # TODO Tier 2: Store in vector DB and graph DB
        # await self.vector_store.add(...)
        # await self.graph_db.create_node(...)

        return node_id

    async def get_relevant(
        self,
        agent_id: str,
        task_type: str,
        max_tokens: int = 4000
    ) -> Dict:
        """Get relevant context for agent within token budget"""

        # For Tier 1: Simple filtering
        relevant_insights = [
            insight for insight in self.insights.values()
            if self._is_relevant(insight, agent_id, task_type)
        ]

        # Sort by score
        relevant_insights.sort(key=lambda x: x.score, reverse=True)

        # Apply token budget
        selected = self._select_within_budget(relevant_insights, max_tokens)

        return {
            "insights": selected,
            "relationships": [],  # TODO Tier 2
            "provenance": []      # TODO Tier 2
        }

    def _is_relevant(
        self,
        insight: Insight,
        agent_id: str,
        task_type: str
    ) -> bool:
        """Simple relevance check for Tier 1"""
        # Always include high-scoring insights
        if insight.score > 0.85:
            return True

        # Include insights from last hour
        from datetime import datetime, timedelta
        if datetime.utcnow() - insight.timestamp < timedelta(hours=1):
            return True

        return False

    def _select_within_budget(
        self,
        insights: List[Insight],
        max_tokens: int
    ) -> List[Insight]:
        """Select insights within token budget"""

        selected = []
        estimated_tokens = 0

        for insight in insights:
            # Rough estimate: 1 token ≈ 4 characters
            insight_tokens = len(insight.content) // 4

            if estimated_tokens + insight_tokens <= max_tokens:
                selected.append(insight)
                estimated_tokens += insight_tokens
            else:
                break

        return selected
