#!/usr/bin/env python3
"""Generate a Nature-style perspective on action-driven learning."""

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
RAG_COLLECTION_NAME = "action_documents"


def get_chroma_collection():
    """Return ChromaDB collection for action documents if available."""
    if not CHROMADB_PATH.exists():
        return None

    client = chromadb.PersistentClient(
        path=str(CHROMADB_PATH),
        settings=ChromaSettings(
            anonymized_telemetry=False,
            allow_reset=True,
        ),
    )

    try:
        return client.get_collection(RAG_COLLECTION_NAME)
    except Exception:
        return None


EMBEDDING_SERVICE = EmbeddingService()
CHROMA_COLLECTION = get_chroma_collection()


def retrieve_rag_evidence() -> str:
    """Build a formatted evidence block from the action_documents collection."""

    if CHROMA_COLLECTION is None:
        return (
            "- Collection `action_documents` not found in ChromaDB. "
            "Run scripts/ingest_action_documents.py first if additional grounding is required."
        )

    thematic_queries: Dict[str, str] = {
        "Action precedes perception": "infant action perception predictive coding corollary discharge",
        "Play-driven cell assemblies": "play behavior hebbian cell assembly neural syntax",
        "Predictive coding as embodied": "predictive coding action loop motor inference",
        "Replay and consolidation": "hippocampal sharp wave ripple replay learning",
        "Learning environments": "exploratory learning environment educator scaffolding"
    }

    lines: List[str] = []

    for theme, query in thematic_queries.items():
        try:
            query_vector = EMBEDDING_SERVICE.encode(query)
        except Exception as exc:
            lines.append(f"### {theme}\n- Embedding failure: {exc}")
            continue

        results = CHROMA_COLLECTION.query(
            query_embeddings=[query_vector.tolist()],
            n_results=3
        )

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not ids:
            lines.append(f"### {theme}\n- No matches found for query: {query}")
            continue

        snippet_lines = []
        for idx, doc_id in enumerate(ids):
            raw_text = documents[idx] if idx < len(documents) else ""
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            distance = distances[idx] if idx < len(distances) else 1.0
            score = 1 - float(distance)

            source_file = metadata.get("source_file", "unknown.pdf")
            chunk_index = metadata.get("chunk_index")
            location_tag = f"{source_file}#{chunk_index}" if chunk_index is not None else source_file

            cleaned = " ".join(raw_text.split())
            snippet = shorten(cleaned, width=260, placeholder="…")

            snippet_lines.append(
                f"- [{score:.2f}] {location_tag}: {snippet}"
            )

        block = "\n".join(snippet_lines)
        lines.append(f"### {theme}\n{block}")

    return "\n\n".join(lines)


def build_context() -> dict:
    thesis = (
        "Volitional action is the brain's primordial learning signal: behaviour launches predictions, "
        "compares them against sensory consequences, and reshapes neural assemblies—so 'acting' is "
        "delayed thinking and the substrate of adaptive cognition."
    )

    slide_highlights = dedent(
        """
        - Slides 1–8 argue that the brain is a self-organizing complex system whose intrinsic dynamics drive exploration; learning is not passive input processing but emergent behaviour.
        - Slides 9–14 position action ahead of perception: infants build internal models by moving first, with corollary discharge and the Troxler effect illustrating that sensory meaning collapses without action-contingent updating.
        - Slides 17–20 describe play as the sculptor of cell assemblies via Hebbian co-activation, providing neural syntax for later abstraction and creativity.
        - Slides 21–24 emphasise rest- and replay-driven consolidation (hippocampal sharp wave ripples) as the offline optimisation step linking action episodes to long-term models.
        - Slides 26–38 reframe educators as facilitators of exploratory niches: trust, unstructured play, externalisation of thought, and collective scaffolding convert individual action into shared intelligence.
        """
    ).strip()

    book_evidence = dedent(
        """
        - Chapter 3 (“Perception from Action”) demonstrates the action–perception cycle via phi phenomena and fixation fade, concluding that movement is prerequisite for stable perception [Buzsáki 2019, Ch.3].
        - Chapter 5 (“Internalization of Experience”) details multi-loop cortico-hippocampal circuits that decouple from immediate input to simulate futures, grounding cognition in prior action repertoires [Buzsáki 2019, Ch.5].
        - Chapter 8 (“Internally Organized Activity During Offline Brain States”) characterises sharp-wave ripples as synchronous replay events that reweight synapses and discover novel associations during rest [Buzsáki 2019, Ch.8].
        - Chapter 13 (“The Brain’s Best Guess”) formalises the “good-enough” predictive brain: fast action-centric networks provide satisficing inference that is later refined by slower deliberative circuits, echoing predictive coding debates [Buzsáki 2019, Ch.13].
        """
    ).strip()

    return {
        "thesis": thesis,
        "slide_highlights": slide_highlights,
        "book_evidence": book_evidence,
        "rag_evidence": retrieve_rag_evidence(),
    }


def render_prompt(context: dict, template_name: str = "nature_perspective_action_learning") -> str:
    manager = PromptManager(templates_dir="prompts")
    return manager.render_prompt(template_name, context)


def call_model(prompt: str, output_path: Path) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Did you load the .env file?")

    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model="gpt-4.1-2025-04-14",
        input=prompt,
        max_output_tokens=4000,
        temperature=0.6,
    )

    content = response.output_text

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content)

    return content


def main():
    load_dotenv(PROJECT_ROOT / ".env")

    context = build_context()
    prompt = render_prompt(context)

    prompts_dir = PROJECT_ROOT / "output"
    prompts_dir.mkdir(exist_ok=True)

    prompt_path = prompts_dir / "nature_perspective_action_learning_prompt.txt"
    prompt_path.write_text(prompt)

    draft_path = prompts_dir / "nature_perspective_action_learning_draft.md"
    content = call_model(prompt, draft_path)

    print("✅ Generated perspective draft")
    print(f"   Prompt saved to: {prompt_path}")
    print(f"   Draft saved to: {draft_path}")
    print(f"   Draft length: {len(content)} characters")


if __name__ == "__main__":
    main()
