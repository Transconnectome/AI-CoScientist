"""Seed selection utilities for graph-aware RAG."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List

from src.services.rag.graph_index_store import GraphIndexStore, GraphNode


def _tokenize(text: str) -> List[str]:
    return [token for token in text.lower().replace("/", " ").split() if token]


class GraphSeedSelector:
    """Selects seed nodes for multi-agent graph traversal."""

    def __init__(self, graph_store: GraphIndexStore) -> None:
        self.graph_store = graph_store

    def suggest_seeds(self, query: str, limit: int = 5) -> List[str]:
        if limit <= 0:
            return []

        query_tokens = set(_tokenize(query))

        scored: List[tuple[float, float, str]] = []

        for node in self.graph_store.iter_nodes():
            node_score, fallback = self._score_node(node, query_tokens)
            scored.append((node_score, fallback, node.node_id))

        if not scored:
            return []

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)

        top = [node_id for _, _, node_id in scored[:limit] if _ > 0]

        if len(top) < limit:
            # fallback to high fallback score nodes not already selected
            remaining = [node_id for _, fallback, node_id in scored if node_id not in top]
            top.extend(remaining[: max(0, limit - len(top))])

        return top[:limit]

    def _score_node(self, node: GraphNode, query_tokens: Iterable[str]) -> tuple[float, float]:
        content_tokens = _tokenize(node.content)

        metadata_tokens: List[str] = []
        for value in node.metadata.values():
            if isinstance(value, str):
                metadata_tokens.extend(_tokenize(value))
            elif isinstance(value, (list, tuple, set)):
                for item in value:
                    metadata_tokens.extend(_tokenize(str(item)))
            else:
                metadata_tokens.extend(_tokenize(str(value)))

        combined = content_tokens + metadata_tokens
        counter = Counter(combined)
        overlap = sum(counter[token] for token in query_tokens)

        degree = len(self.graph_store.get_neighbors(node.node_id))
        fallback = degree + len(content_tokens) / 50.0

        return float(overlap), float(fallback)


__all__ = ["GraphSeedSelector"]

