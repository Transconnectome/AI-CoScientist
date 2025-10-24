"""Utilities for ingesting paper content into the graph index."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional

from src.services.rag.graph_index_store import GraphIndexStore


class GraphDocumentIngestor:
    """Creates graph nodes/edges from structured paper data."""

    def __init__(self, graph_store: GraphIndexStore) -> None:
        self.graph_store = graph_store

    def ingest_paper(
        self,
        paper_id: str,
        title: str,
        abstract: str,
        sections: Iterable[Dict[str, str]],
        keywords: Optional[Iterable[str]] = None,
        citations: Optional[Iterable[str]] = None,
    ) -> Dict[str, List[str]]:
        """Insert a paper and related nodes into the graph."""

        keywords = list(keywords or [])
        citations = list(citations or [])

        created: Dict[str, List[str]] = {
            "document": [],
            "sections": [],
            "concepts": [],
            "citations": [],
        }

        doc_content = f"{title.strip()}\n\n{abstract.strip()}".strip()
        doc_metadata = {
            "title": title.strip(),
            "keywords": [kw.lower() for kw in keywords],
            "citation_count": len(citations),
        }

        self.graph_store.add_node(
            node_id=paper_id,
            node_type="document",
            content=doc_content,
            metadata=doc_metadata,
        )
        created["document"].append(paper_id)

        # Sections
        for index, section in enumerate(sections):
            name = section.get("name", f"Section {index+1}")
            section_id = self._slugify(f"{paper_id}::section::{name}")
            metadata = {"name": name, "order": index}
            self.graph_store.add_node(
                node_id=section_id,
                node_type="section",
                content=section.get("content", ""),
                metadata=metadata,
            )
            self.graph_store.add_edge(paper_id, section_id, "has_section")
            created["sections"].append(section_id)

        # Keywords -> concept nodes
        for keyword in keywords:
            concept_id = self._slugify(f"concept::{keyword}")
            self.graph_store.add_node(
                node_id=concept_id,
                node_type="concept",
                content=keyword.lower(),
                metadata={"label": keyword.lower()},
            )
            self.graph_store.add_edge(paper_id, concept_id, "mentions")
            if concept_id not in created["concepts"]:
                created["concepts"].append(concept_id)

        # Citations
        for citation in citations:
            citation_id = self._slugify(f"citation::{citation}")
            self.graph_store.add_node(
                node_id=citation_id,
                node_type="citation",
                content=citation,
                metadata={"reference": citation},
            )
            self.graph_store.add_edge(paper_id, citation_id, "cites")
            created["citations"].append(citation_id)

        return created

    @staticmethod
    def _slugify(value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
        return slug


__all__ = ["GraphDocumentIngestor"]

