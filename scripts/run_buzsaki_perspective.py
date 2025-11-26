#!/usr/bin/env python3
"""Generate Buzsáki-style annotations using RAG-augmented context."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from textwrap import dedent, shorten
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.config import Settings as ChromaSettings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services.llm.prompt_manager import PromptManager
from src.services.knowledge_base.embedding import EmbeddingService


CHROMADB_PATH = PROJECT_ROOT / "chromadb_data"
COLLECTION_NAME = "action_documents"
TEMPLATE_NAME = "buzsaki_inside_out_perspective"
PROMPT_OUTPUT = PROJECT_ROOT / "output" / "buzsaki_inside_out_prompt.txt"
DRAFT_OUTPUT = PROJECT_ROOT / "output" / "buzsaki_inside_out_draft.md"


THEMATIC_QUERIES: Dict[str, str] = {
    "Inside-out inversion of perception": "inside out perception action phi phenomenon corollary discharge",
    "Cell assembly trajectories": "cell assembly trajectory replay play exploratory sequences",
    "Predictive hierarchies": "good enough brain predictive hierarchy fast slow action",
    "Mirror-action coupling": "mirror neuron action simulation internalization speech",
    "Hippocampal replay dynamics": "sharp wave ripple replay planning future past",
    "Educational corollaries": "exploratory learning environment educator scaffolding inside out"
}


def get_collection():
    if not CHROMADB_PATH.exists():
        raise FileNotFoundError(
            "ChromaDB path not found. Ensure action documents have been ingested."
        )

    client = chromadb.PersistentClient(
        path=str(CHROMADB_PATH),
        settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
    )
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' not available. Run the ingestion pipeline first."
        ) from exc


def build_rag_block(collection, embedder: EmbeddingService) -> str:
    lines: List[str] = []

    for theme, query in THEMATIC_QUERIES.items():
        query_vector = embedder.encode(query)
        results = collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=4
        )

        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        if not ids:
            lines.append(f"### {theme}\n- No matches for '{query}'")
            continue

        snippet_lines = []
        for idx, doc_id in enumerate(ids):
            raw = docs[idx] if idx < len(docs) else ""
            cleaned = " ".join(raw.split())
            snippet = shorten(cleaned, width=280, placeholder="…")

            metadata = metas[idx] if idx < len(metas) else {}
            source_file = metadata.get("source_file", "unknown.pdf")
            chunk_index = metadata.get("chunk_index")
            location = f"{source_file}#{chunk_index}" if chunk_index is not None else source_file

            distance = dists[idx] if idx < len(dists) else 1.0
            score = 1 - float(distance)

            snippet_lines.append(
                f"- [{score:.2f}] {location}: {snippet}"
            )

        lines.append(f"### {theme}\n" + "\n".join(snippet_lines))

    return "\n\n".join(lines)


def build_context(rag_block: str) -> Dict[str, str]:
    thesis_title = "Action as the Brain’s Primordial Learning Signal: Embodied Inference from Development to Cognition"
    thesis_statement = (
        "Volitional action is the brain's primordial learning signal: behaviour launches predictions, "
        "compares them against sensory consequences, and reshapes neural assemblies—so acting is a form "
        "of delayed, embodied thinking."
    )

    outline_highlights = dedent(
        """
        - The perspective argues that behaviour precedes perception and sculpts neural assemblies through Hebbian dynamics.
        - Play and exploratory actions seed predictive coding hierarchies that replay during offline states.
        - Hippocampal sharp-wave ripples consolidate action-derived models, enabling transfer and abstraction.
        - Educational and translational designs should engineer exploratory niches that respect inside-out learning principles.
        """
    ).strip()

    return {
        "thesis_title": thesis_title,
        "thesis_statement": thesis_statement,
        "outline_highlights": outline_highlights,
        "rag_evidence": rag_block,
    }


def render_prompt(context: Dict[str, str]) -> str:
    manager = PromptManager(templates_dir="prompts")
    return manager.render_prompt(TEMPLATE_NAME, context)


def call_model(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing. Load the .env file.")

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model="gpt-4.1-2025-04-14",
        input=prompt,
        max_output_tokens=3500,
        temperature=0.55,
    )
    return response.output_text


def main():
    load_dotenv(PROJECT_ROOT / ".env")

    collection = get_collection()
    embedder = EmbeddingService()
    rag_block = build_rag_block(collection, embedder)
    context = build_context(rag_block)

    prompt = render_prompt(context)
    PROMPT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    PROMPT_OUTPUT.write_text(prompt)

    content = call_model(prompt)
    DRAFT_OUTPUT.write_text(content)

    print("✅ Generated Buzsáki-style augmentation")
    print(f"   Prompt saved to: {PROMPT_OUTPUT}")
    print(f"   Draft saved to: {DRAFT_OUTPUT}")
    print(f"   Draft length: {len(content)} characters")


if __name__ == "__main__":
    main()
