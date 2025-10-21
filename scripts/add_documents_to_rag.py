#!/usr/bin/env python3
"""Add new papers and books to RAG system."""

import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.embeddings import EmbeddingService
from src.services.rag import RAGManager
from src.services.paper.ensemble_scorer import EnsembleScorer


async def add_document_to_rag(
    file_path: str,
    document_id: str,
    document_type: str = "paper",  # paper or book
    field: str = "neuroscience",
    sections: List[str] = None
):
    """Add a document to RAG system.

    Args:
        file_path: Path to the document file
        document_id: Unique identifier for the document
        document_type: Type of document (paper or book)
        field: Academic field
        sections: List of sections to process (default: Auto-detect)
    """
    print(f"\n{'='*70}")
    print(f"📄 Adding {document_type}: {document_id}")
    print(f"{'='*70}\n")

    # Read document
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    print(f"📏 Document length: {len(content):,} characters")

    # Initialize services
    print("\n🔧 Initializing services...")
    embedding_service = EmbeddingService(provider="openai")
    rag_manager = RAGManager(
        embedding_service=embedding_service,
        chromadb_mode="auto"
    )

    if not rag_manager.is_enabled():
        print("❌ RAG system not available")
        return

    print("✅ RAG system ready")

    # Auto-detect sections if not provided
    if sections is None:
        sections = detect_sections(content, document_type)

    print(f"\n📁 Processing sections: {', '.join(sections)}")

    # Split document by sections
    section_contents = split_by_sections(content, sections)

    # Evaluate if it's a paper
    scorer = None
    baseline_scores = {}

    if document_type == "paper":
        print("\n📊 Evaluating paper quality...")
        scorer = EnsembleScorer()
        baseline_scores = await scorer.evaluate(content)
        print(f"   Overall score: {baseline_scores.get('overall', 0):.2f}/10")
        print(f"   Clarity: {baseline_scores.get('clarity', 0):.2f}")
        print(f"   Novelty: {baseline_scores.get('novelty', 0):.2f}")

    # Store each section
    print("\n💾 Storing sections in RAG...")
    for section, section_content in section_contents.items():
        if not section_content.strip():
            continue

        # Create metadata
        metadata = {
            "document_id": document_id,
            "document_type": document_type,
            "section": section,
            "field": field,
            "content_length": len(section_content)
        }

        # Add scores if available
        if baseline_scores:
            for key, value in baseline_scores.items():
                metadata[f"baseline_{key}"] = value

        # Store in RAG
        await rag_manager.store_document(
            document_id=f"{document_id}_{section}",
            content=section_content,
            metadata=metadata
        )

        print(f"   ✅ Stored: {section} ({len(section_content):,} chars)")

    print(f"\n✅ Successfully added {document_id} to RAG system")
    print(f"{'='*70}\n")


def detect_sections(content: str, document_type: str) -> List[str]:
    """Auto-detect sections in document.

    Args:
        content: Document content
        document_type: Type of document

    Returns:
        List of section names
    """
    if document_type == "paper":
        # Common paper sections
        possible_sections = [
            "Abstract", "Introduction", "Related Work", "Methods",
            "Results", "Discussion", "Conclusion", "References"
        ]
    else:  # book
        # Common book sections
        possible_sections = [
            "Preface", "Introduction", "Chapter", "Conclusion",
            "References", "Index"
        ]

    detected = []
    content_lower = content.lower()

    for section in possible_sections:
        if section.lower() in content_lower:
            detected.append(section)

    # Default sections if none detected
    if not detected:
        if document_type == "paper":
            detected = ["Abstract", "Introduction", "Methods", "Results", "Discussion"]
        else:
            detected = ["Introduction", "Main Content", "Conclusion"]

    return detected


def split_by_sections(content: str, sections: List[str]) -> Dict[str, str]:
    """Split content by sections.

    Args:
        content: Document content
        sections: List of section names

    Returns:
        Dictionary mapping section names to content
    """
    section_contents = {}

    # Simple heuristic: split by section headers
    lines = content.split('\n')
    current_section = None
    current_content = []

    for line in lines:
        line_lower = line.lower().strip()

        # Check if this line is a section header
        found_section = None
        for section in sections:
            if section.lower() in line_lower:
                found_section = section
                break

        if found_section:
            # Save previous section
            if current_section:
                section_contents[current_section] = '\n'.join(current_content)

            # Start new section
            current_section = found_section
            current_content = []
        elif current_section:
            current_content.append(line)

    # Save last section
    if current_section:
        section_contents[current_section] = '\n'.join(current_content)

    # If no sections found, treat entire content as one section
    if not section_contents:
        section_contents["Full Content"] = content

    return section_contents


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Add documents to RAG system")
    parser.add_argument("file_path", help="Path to document file")
    parser.add_argument("--id", required=True, help="Document ID")
    parser.add_argument("--type", default="paper", choices=["paper", "book"],
                       help="Document type")
    parser.add_argument("--field", default="neuroscience", help="Academic field")
    parser.add_argument("--sections", nargs="+", help="Sections to process")

    args = parser.parse_args()

    await add_document_to_rag(
        file_path=args.file_path,
        document_id=args.id,
        document_type=args.type,
        field=args.field,
        sections=args.sections
    )


if __name__ == "__main__":
    asyncio.run(main())
