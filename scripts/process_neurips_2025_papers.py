#!/usr/bin/env python3
"""
Process NeurIPS 2025 papers through RAPTOR pipeline.

This script processes the 14 downloaded NeurIPS 2025 papers from both priority1 and priority2
directories, creating a comprehensive knowledge base with hierarchical RAPTOR indexing.

Usage:
    poetry run python scripts/process_neurips_2025_papers.py
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from enum import Enum
import PyPDF2
from datetime import datetime
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import numpy as np
from tqdm import tqdm

# Import LLM providers
import anthropic
import openai
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


# ============================================================================
# Configuration
# ============================================================================

class LLMProvider(str, Enum):
    """Available LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"


# Latest best models from each provider
PROVIDER_MODELS = {
    LLMProvider.OPENAI: "gpt-4o",
    LLMProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
    LLMProvider.GEMINI: "gemini-2.5-pro",
    LLMProvider.DEEPSEEK: "deepseek-chat"
}

# Provider priority order (fallback chain)
PROVIDER_PRIORITY = [
    LLMProvider.GEMINI,     # User requested priority
    LLMProvider.ANTHROPIC,  # Best for reasoning and code
    LLMProvider.OPENAI,     # Excellent all-around
    LLMProvider.DEEPSEEK    # Fast and cost-effective
]


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class Section:
    """Paper section with content and metadata."""
    name: str
    content: str
    order: int
    word_count: int = 0

    def __post_init__(self):
        self.word_count = len(self.content.split())


@dataclass
class Chunk:
    """Text chunk with metadata."""
    chunk_id: str
    content: str
    section: str
    chunk_index: int
    total_chunks: int
    embedding: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class RAPTORNode:
    """Node in RAPTOR hierarchical tree."""
    node_id: str
    content: str
    level: int  # 0=chunk, 1=section_summary, 2=paper_summary
    embedding: Optional[np.ndarray] = None
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class ProcessedPaper:
    """Complete paper data for storage."""
    paper_id: str
    filename: str
    title: str
    journal: str
    year: int
    sections: List[Section] = field(default_factory=list)
    level0_chunks: List[Chunk] = field(default_factory=list)
    level1_summaries: List[RAPTORNode] = field(default_factory=list)
    level2_summary: Optional[RAPTORNode] = None
    abstract: str = ""
    keywords: List[str] = field(default_factory=list)
    total_words: int = 0
    full_text: str = ""


# ============================================================================
# Import reusable components from existing script
# ============================================================================

# Import from existing script
from scripts.ingest_golden_references_advanced import (
    PDFExtractor,
    BaseLLMProvider,
    AnthropicProvider,
    OpenAIProvider,
    GeminiProvider,
    DeepSeekProvider,
    MultiProviderLLM,
    SectionParser,
    SectionAwareChunker,
    RAPTORBuilder
)


# ============================================================================
# NeurIPS 2025 Paper Processor
# ============================================================================

class NeurIPS2025Processor:
    """Process NeurIPS 2025 papers through RAPTOR pipeline."""

    def __init__(self, output_dir: str = "data/reference_papers/neurips_2025_processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.pdf_extractor = PDFExtractor()
        self.llm = MultiProviderLLM()
        self.section_parser = SectionParser(self.llm)
        self.chunker = SectionAwareChunker(chunk_size=512, overlap=50)
        self.raptor_builder = RAPTORBuilder(self.section_parser)

        # Initialize embedding model
        print("Loading SciBERT embedding model...")
        self.embedding_model = SentenceTransformer('allenai/scibert_scivocab_uncased')
        print("✓ SciBERT loaded\n")

    def extract_neurips_metadata(self, filename: str, text: str) -> Dict:
        """Extract metadata from NeurIPS 2025 paper."""
        # All papers are from NeurIPS 2025
        year = 2025
        journal = "NeurIPS 2025"

        # Extract title from first lines of text
        lines = [l.strip() for l in text.split('\n') if l.strip()]

        # Try to find a good title (usually first few lines)
        title = None
        for i, line in enumerate(lines[:10]):
            if len(line) > 20 and len(line) < 200 and not line.lower().startswith('abstract'):
                title = line
                break

        if not title:
            # Fallback to filename
            title = filename.replace('.pdf', '').replace('_', ' ')

        return {
            'title': title[:200],  # Limit title length
            'journal': journal,
            'year': year
        }

    async def process_paper(self, pdf_path: Path) -> Optional[Dict]:
        """Process a single paper through the complete pipeline."""

        print("=" * 70)
        print(f"Processing: {pdf_path.name}")
        print("=" * 70)

        # 1. Extract PDF text
        print("1. Extracting PDF text...")
        full_text = self.pdf_extractor.extract_text(pdf_path)

        if not full_text or len(full_text) < 500:
            print(f"  ✗ No text extracted from {pdf_path.name}")
            return None

        print(f"  ✓ Extracted {len(full_text):,} characters")

        # Get metadata
        metadata = self.extract_neurips_metadata(pdf_path.name, full_text)
        print(f"  Title: {metadata['title']}")
        print(f"  Journal: {metadata['journal']} ({metadata['year']})")

        paper_id = pdf_path.stem

        # 2. Parse sections with LLM
        print("2. Parsing sections with LLM...")
        sections = await self.section_parser.parse_sections(full_text, metadata['title'])
        print(f"  ✓ Found {len(sections)} sections:")
        for section in sections[:10]:  # Show first 10
            print(f"    - {section.name}: {section.word_count} words")
        if len(sections) > 10:
            print(f"    ... and {len(sections) - 10} more sections")

        # 3. Chunk sections
        print("3. Chunking sections...")
        all_chunks = []
        for section in sections:
            chunks = self.chunker.chunk_section(section, paper_id)
            all_chunks.extend(chunks)

        print(f"  ✓ Total chunks: {len(all_chunks)}")

        # 4. Build RAPTOR hierarchy
        print("4. Building RAPTOR hierarchy...")
        level1_summaries, level2_summary = await self.raptor_builder.build_hierarchy(
            all_chunks, sections, metadata['title'], paper_id
        )

        # 5. Generate embeddings
        print("5. Generating embeddings...")

        # L0 embeddings
        l0_texts = [chunk.content for chunk in all_chunks]
        l0_embeddings = self.embedding_model.encode(l0_texts, show_progress_bar=False)
        for chunk, embedding in zip(all_chunks, l0_embeddings):
            chunk.embedding = embedding

        # L1 embeddings
        l1_texts = [node.content for node in level1_summaries]
        if l1_texts:
            l1_embeddings = self.embedding_model.encode(l1_texts, show_progress_bar=False)
            for node, embedding in zip(level1_summaries, l1_embeddings):
                node.embedding = embedding

        # L2 embedding
        if level2_summary:
            l2_embedding = self.embedding_model.encode([level2_summary.content], show_progress_bar=False)[0]
            level2_summary.embedding = l2_embedding

        print("  ✓ Generated embeddings for all levels")

        # 6. Serialize to JSON
        print("6. Saving to JSON...")

        # Convert to serializable format
        paper_data = {
            'paper_id': paper_id,
            'filename': pdf_path.name,
            'title': metadata['title'],
            'journal': metadata['journal'],
            'year': metadata['year'],
            'sections': [
                {
                    'name': s.name,
                    'content': s.content,
                    'order': s.order,
                    'word_count': s.word_count
                }
                for s in sections
            ],
            'level0_chunks': [
                {
                    'chunk_id': c.chunk_id,
                    'content': c.content,
                    'section': c.section,
                    'chunk_index': c.chunk_index,
                    'total_chunks': c.total_chunks,
                    'embedding': c.embedding.tolist(),
                    'metadata': c.metadata
                }
                for c in all_chunks
            ],
            'level1_summaries': [
                {
                    'node_id': n.node_id,
                    'content': n.content,
                    'level': n.level,
                    'embedding': n.embedding.tolist(),
                    'parent_id': n.parent_id,
                    'children_ids': n.children_ids,
                    'metadata': n.metadata
                }
                for n in level1_summaries
            ],
            'level2_summary': {
                'node_id': level2_summary.node_id,
                'content': level2_summary.content,
                'level': level2_summary.level,
                'embedding': level2_summary.embedding.tolist(),
                'parent_id': level2_summary.parent_id,
                'children_ids': level2_summary.children_ids,
                'metadata': level2_summary.metadata
            } if level2_summary else None,
            'full_text': full_text,
            'total_words': len(full_text.split())
        }

        # Save to JSON
        output_path = self.output_dir / f"{paper_id}.json"
        with open(output_path, 'w') as f:
            json.dump(paper_data, f, indent=2)

        print(f"  ✓ Saved to {output_path}")
        print(f"\n✅ Successfully processed: {metadata['title'][:100]}\n")

        return paper_data

    async def process_all(self, base_dir: Path):
        """Process all NeurIPS 2025 papers from priority1 and priority2 directories."""

        # Collect all PDFs from both directories
        priority1_dir = base_dir / "priority1"
        priority2_dir = base_dir / "priority2"

        pdfs = []
        if priority1_dir.exists():
            pdfs.extend(sorted(priority1_dir.glob("*.pdf")))
        if priority2_dir.exists():
            pdfs.extend(sorted(priority2_dir.glob("*.pdf")))

        print("=" * 70)
        print("NEURIPS 2025 PAPERS - RAPTOR PROCESSING")
        print("=" * 70)
        print(f"Total PDFs: {len(pdfs)}")
        print(f"  Priority 1: {len(list(priority1_dir.glob('*.pdf'))) if priority1_dir.exists() else 0}")
        print(f"  Priority 2: {len(list(priority2_dir.glob('*.pdf'))) if priority2_dir.exists() else 0}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 70)
        print("\n")

        results = {
            'success': [],
            'failed': [],
            'total_chunks_l0': 0,
            'total_chunks_l1': 0,
            'total_chunks_l2': 0,
            'papers': []
        }

        for idx, pdf_path in enumerate(pdfs, 1):
            print(f"[{idx}/{len(pdfs)}]\n")

            try:
                paper_data = await self.process_paper(pdf_path)

                if paper_data:
                    results['success'].append(pdf_path.name)
                    results['total_chunks_l0'] += len(paper_data['level0_chunks'])
                    results['total_chunks_l1'] += len(paper_data['level1_summaries'])
                    results['total_chunks_l2'] += (1 if paper_data['level2_summary'] else 0)
                    results['papers'].append({
                        'filename': pdf_path.name,
                        'title': paper_data['title'],
                        'chunks_l0': len(paper_data['level0_chunks']),
                        'chunks_l1': len(paper_data['level1_summaries']),
                        'total_words': paper_data['total_words']
                    })
                else:
                    results['failed'].append(pdf_path.name)

            except Exception as e:
                print(f"✗ Error processing {pdf_path.name}: {e}\n")
                import traceback
                traceback.print_exc()
                results['failed'].append(pdf_path.name)

            print(f"Progress: {len(results['success'])}/{len(pdfs)} completed\n")

        # Print summary
        print("=" * 70)
        print("PROCESSING COMPLETE")
        print("=" * 70)
        print(f"✅ Success: {len(results['success'])}/{len(pdfs)}")
        print(f"✗ Failed: {len(results['failed'])}/{len(pdfs)}")
        print(f"\nChunks created:")
        print(f"  Level 0 (chunks): {results['total_chunks_l0']}")
        print(f"  Level 1 (sections): {results['total_chunks_l1']}")
        print(f"  Level 2 (papers): {results['total_chunks_l2']}")
        print(f"  Total: {results['total_chunks_l0'] + results['total_chunks_l1'] + results['total_chunks_l2']}")

        # Save detailed results
        results_path = self.output_dir / "processing_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to: {results_path}")
        print("=" * 70)

        return results


# ============================================================================
# Main
# ============================================================================

async def main():
    """Main execution."""

    # Path to NeurIPS 2025 papers
    neurips_dir = Path("data/발달장애/neurips_2025_papers")

    if not neurips_dir.exists():
        print(f"Error: NeurIPS papers directory not found: {neurips_dir}")
        return

    # Initialize processor
    processor = NeurIPS2025Processor()

    # Process all papers
    results = await processor.process_all(neurips_dir)

    print("\n🎉 All done! Papers ready for ChromaDB loading.")


if __name__ == "__main__":
    asyncio.run(main())
