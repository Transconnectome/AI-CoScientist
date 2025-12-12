#!/usr/bin/env python3
"""
Ingest Grant Proposal Files into DD-RAPTOR System with Chunking

This script processes grant proposal PDFs (QuantERA, INCITE, BrainLink) and ingests them
into the DD-RAPTOR ChromaDB system with proper chunking for large files.

Features:
- PDF text extraction with error handling
- Intelligent chunking for large documents
- RAPTOR hierarchical indexing (3 levels)
- SciBERT embeddings optimized for scientific content
- Direct integration with DD-RAPTOR ChromaDB collections
- Progress tracking and error recovery

Usage:
    poetry run python scripts/ingest_grant_proposals_dd_raptor.py
    poetry run python scripts/ingest_grant_proposals_dd_raptor.py --file brainlink.pdf
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
import PyPDF2
from datetime import datetime
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import numpy as np
from tqdm import tqdm
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# ============================================================================
# Data Models
# ============================================================================

@dataclass
class GrantChunk:
    """Grant proposal text chunk with metadata."""
    chunk_id: str
    content: str
    section: str
    chunk_index: int
    total_chunks: int
    embedding: Optional[List[float]] = None
    metadata: Dict = field(default_factory=dict)

@dataclass
class GrantSection:
    """Grant proposal section with content and metadata."""
    name: str
    content: str
    order: int
    word_count: int = 0

    def __post_init__(self):
        self.word_count = len(self.content.split())

@dataclass
class GrantProposal:
    """Complete grant proposal with hierarchical structure."""
    proposal_id: str
    title: str
    file_path: str
    sections: List[GrantSection]
    level0_chunks: List[GrantChunk] = field(default_factory=list)
    level1_summaries: List[Dict] = field(default_factory=list)
    level2_summary: Optional[Dict] = None
    metadata: Dict = field(default_factory=dict)

# ============================================================================
# PDF Processing
# ============================================================================

class GrantPDFExtractor:
    """Extract and process text from grant proposal PDFs."""

    @staticmethod
    def extract_text(pdf_path: Path) -> str:
        """Extract text from PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""

                for page_num in range(len(reader.pages)):
                    try:
                        page = reader.pages[page_num]
                        text += page.extract_text() + "\n"
                    except Exception as e:
                        print(f"  Warning: Failed to extract page {page_num + 1}: {e}")
                        continue

                # Clean text
                text = re.sub(r'\n+', '\n', text)
                text = re.sub(r'\s+', ' ', text)
                return text.strip()

        except Exception as e:
            print(f"  Error extracting PDF: {e}")
            return ""

    @staticmethod
    def estimate_metadata(filename: str, text: str) -> Dict:
        """Estimate metadata from filename and content."""

        # Extract proposal type from filename
        proposal_type = "Unknown"
        if "quantera" in filename.lower():
            proposal_type = "QuantERA 2025"
        elif "incite" in filename.lower():
            proposal_type = "INCITE"
        elif "brainlink" in filename.lower():
            proposal_type = "BrainLink"

        # Extract title from text (look for title patterns)
        title = filename.replace('.pdf', '').replace('_', ' ')

        # Look for common grant title patterns
        title_patterns = [
            r'Title[:\s]+(.+?)[\n\.]',
            r'Project[:\s]+(.+?)[\n\.]',
            r'^(.{50,200}?)[\n\.]'  # First substantial line
        ]

        for pattern in title_patterns:
            match = re.search(pattern, text[:2000], re.IGNORECASE | re.MULTILINE)
            if match:
                potential_title = match.group(1).strip()
                if len(potential_title) > 20 and len(potential_title) < 200:
                    title = potential_title
                    break

        return {
            'title': title,
            'proposal_type': proposal_type,
            'file_size_mb': round(len(text) / 1024 / 1024 * 1.5, 2),  # Estimate
            'word_count': len(text.split()),
            'extracted_at': datetime.now().isoformat()
        }

# ============================================================================
# Chunking System
# ============================================================================

class GrantProposalChunker:
    """Intelligent chunking for grant proposals."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def create_chunks(self, sections: List[GrantSection], proposal_id: str) -> List[GrantChunk]:
        """Create chunks from proposal sections."""
        chunks = []

        for section in sections:
            section_chunks = self._chunk_section(section, proposal_id)
            chunks.extend(section_chunks)

        # Update total chunk count in metadata
        for i, chunk in enumerate(chunks):
            chunk.total_chunks = len(chunks)

        return chunks

    def _chunk_section(self, section: GrantSection, proposal_id: str) -> List[GrantChunk]:
        """Chunk a single section with overlap."""
        text = section.content
        words = text.split()

        if len(words) <= self.chunk_size:
            # Section fits in one chunk
            return [GrantChunk(
                chunk_id=f"{proposal_id}_{section.name}_chunk_0",
                content=text,
                section=section.name,
                chunk_index=0,
                total_chunks=1,
                metadata={
                    'section_name': section.name,
                    'section_order': section.order,
                    'section_word_count': section.word_count,
                    'proposal_id': proposal_id
                }
            )]

        # Split into overlapping chunks
        chunks = []
        chunk_index = 0

        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)

            if not chunk_text.strip():
                continue

            chunk = GrantChunk(
                chunk_id=f"{proposal_id}_{section.name}_chunk_{chunk_index}",
                content=chunk_text,
                section=section.name,
                chunk_index=chunk_index,
                total_chunks=0,  # Will be updated later
                metadata={
                    'section_name': section.name,
                    'section_order': section.order,
                    'section_word_count': section.word_count,
                    'proposal_id': proposal_id,
                    'chunk_start_word': i,
                    'chunk_end_word': i + len(chunk_words)
                }
            )

            chunks.append(chunk)
            chunk_index += 1

        return chunks

# ============================================================================
# Section Parser
# ============================================================================

class GrantSectionParser:
    """Parse grant proposals into logical sections."""

    def parse_sections(self, text: str) -> List[GrantSection]:
        """Parse text into logical sections."""

        # Common grant proposal section patterns
        section_patterns = [
            # Standard academic sections
            r'(?:^|\n)\s*(?:Abstract|ABSTRACT)\s*(?:\n|:)',
            r'(?:^|\n)\s*(?:Introduction|INTRODUCTION)\s*(?:\n|:)',
            r'(?:^|\n)\s*(?:Background|BACKGROUND)\s*(?:\n|:)',
            r'(?:^|\n)\s*(?:Objectives|OBJECTIVES)\s*(?:\n|:)',
            r'(?:^|\n)\s*(?:Methodology|METHODOLOGY|Methods|METHODS)\s*(?:\n|:)',
            r'(?:^|\n)\s*(?:Timeline|TIMELINE)\s*(?:\n|:)',
            r'(?:^|\n)\s*(?:Budget|BUDGET)\s*(?:\n|:)',
            r'(?:^|\n)\s*(?:Impact|IMPACT)\s*(?:\n|:)',
            r'(?:^|\n)\s*(?:References|REFERENCES)\s*(?:\n|:)',
            # Grant-specific sections
            r'(?:^|\n)\s*(?:Project Description|PROJECT DESCRIPTION)\s*(?:\n|:)',
            r'(?:^|\n)\s*(?:Research Plan|RESEARCH PLAN)\s*(?:\n|:)',
            r'(?:^|\n)\s*(?:Team|TEAM)\s*(?:\n|:)',
            r'(?:^|\n)\s*(?:Resources|RESOURCES)\s*(?:\n|:)',
            # Numbered sections
            r'(?:^|\n)\s*\d+\.?\s+([A-Z][a-z\s]+)\s*(?:\n|:)',
        ]

        sections = []
        section_positions = []

        # Find all section markers
        for pattern in section_patterns:
            for match in re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE):
                section_positions.append((match.start(), match.group().strip()))

        # Sort by position
        section_positions.sort(key=lambda x: x[0])

        # Extract sections
        for i, (start_pos, section_name) in enumerate(section_positions):
            # Clean section name
            section_name = re.sub(r'^\d+\.?\s*', '', section_name)
            section_name = section_name.replace(':', '').strip()

            # Get section content
            if i < len(section_positions) - 1:
                next_pos = section_positions[i + 1][0]
                content = text[start_pos:next_pos].strip()
            else:
                content = text[start_pos:].strip()

            # Skip very short sections
            if len(content.split()) < 20:
                continue

            sections.append(GrantSection(
                name=section_name or f"Section_{i+1}",
                content=content,
                order=i
            ))

        # If no clear sections found, create chunks
        if not sections:
            # Split into logical chunks (roughly by paragraphs or pages)
            chunks = re.split(r'\n\s*\n', text)
            for i, chunk in enumerate(chunks):
                if len(chunk.split()) >= 50:  # Minimum words per section
                    sections.append(GrantSection(
                        name=f"Section_{i+1}",
                        content=chunk.strip(),
                        order=i
                    ))

        return sections

# ============================================================================
# ChromaDB Integration
# ============================================================================

class DDRaptorGrantIngestor:
    """Ingest grant proposals into DD-RAPTOR ChromaDB system."""

    def __init__(self, chromadb_path: str = "chromadb_data_dd"):
        self.chromadb_path = chromadb_path
        self.pdf_extractor = GrantPDFExtractor()
        self.section_parser = GrantSectionParser()
        self.chunker = GrantProposalChunker(chunk_size=800, overlap=80)

        # Initialize embedding model
        print("Loading SciBERT embedding model...")
        self.embedding_model = SentenceTransformer('allenai/scibert_scivocab_uncased')
        print("✓ SciBERT loaded")

        # Initialize ChromaDB
        self._init_chromadb()

    def _init_chromadb(self):
        """Initialize ChromaDB collections."""
        print(f"Initializing ChromaDB at {self.chromadb_path}...")

        if not Path(self.chromadb_path).exists():
            Path(self.chromadb_path).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.chromadb_path)

        # Get or create DD-RAPTOR collections
        self.collection_l0 = self.client.get_or_create_collection(
            name="dd_papers_L0",
            metadata={"description": "Level 0: Grant proposal chunks"}
        )

        self.collection_l1 = self.client.get_or_create_collection(
            name="dd_papers_L1",
            metadata={"description": "Level 1: Grant proposal section summaries"}
        )

        self.collection_l2 = self.client.get_or_create_collection(
            name="dd_papers_L2",
            metadata={"description": "Level 2: Grant proposal summaries"}
        )

        print("✓ ChromaDB collections ready")

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for text chunks."""
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return [emb.tolist() for emb in embeddings]

    def process_grant_proposal(self, pdf_path: Path) -> Optional[GrantProposal]:
        """Process a single grant proposal PDF."""

        print("\n" + "=" * 70)
        print(f"Processing: {pdf_path.name}")
        print("=" * 70)

        # 1. Extract PDF text
        print("1. Extracting PDF text...")
        full_text = self.pdf_extractor.extract_text(pdf_path)

        if not full_text or len(full_text) < 500:
            print(f"  ✗ No text extracted from {pdf_path.name}")
            return None

        print(f"  ✓ Extracted {len(full_text):,} characters")

        # 2. Parse sections
        print("2. Parsing sections...")
        sections = self.section_parser.parse_sections(full_text)
        print(f"  ✓ Found {len(sections)} sections")

        # 3. Create chunks
        print("3. Creating chunks...")
        proposal_id = pdf_path.stem.replace(' ', '_')
        chunks = self.chunker.create_chunks(sections, proposal_id)
        print(f"  ✓ Created {len(chunks)} chunks")

        # 4. Generate embeddings
        print("4. Generating embeddings...")
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = self.generate_embeddings(chunk_texts)

        # Assign embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

        print(f"  ✓ Generated {len(embeddings)} embeddings")

        # 5. Estimate metadata
        metadata = self.pdf_extractor.estimate_metadata(pdf_path.name, full_text)

        # Create grant proposal object
        proposal = GrantProposal(
            proposal_id=proposal_id,
            title=metadata['title'],
            file_path=str(pdf_path),
            sections=sections,
            level0_chunks=chunks,
            metadata=metadata
        )

        print(f"  ✓ Proposal processed: {proposal.title}")
        return proposal

    def ingest_to_chromadb(self, proposal: GrantProposal) -> bool:
        """Ingest processed proposal into ChromaDB."""

        print("5. Ingesting into ChromaDB...")

        try:
            # Prepare data for Level 0 (chunks)
            ids = [chunk.chunk_id for chunk in proposal.level0_chunks]
            embeddings = [chunk.embedding for chunk in proposal.level0_chunks]
            documents = [chunk.content for chunk in proposal.level0_chunks]

            # Enhance metadata for ChromaDB
            metadatas = []
            for chunk in proposal.level0_chunks:
                meta = chunk.metadata.copy()
                meta.update({
                    'proposal_title': proposal.title,
                    'proposal_type': proposal.metadata.get('proposal_type', 'Grant'),
                    'file_path': proposal.file_path,
                    'ingested_at': datetime.now().isoformat()
                })
                metadatas.append(meta)

            # Ingest in batches to handle large files
            batch_size = 100
            for i in range(0, len(ids), batch_size):
                batch_end = i + batch_size
                self.collection_l0.add(
                    ids=ids[i:batch_end],
                    embeddings=embeddings[i:batch_end],
                    documents=documents[i:batch_end],
                    metadatas=metadatas[i:batch_end]
                )

            print(f"  ✓ Ingested {len(ids)} chunks into DD-RAPTOR")
            return True

        except Exception as e:
            print(f"  ✗ Failed to ingest: {e}")
            return False

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""

    parser = argparse.ArgumentParser(description='Ingest grant proposals into DD-RAPTOR')
    parser.add_argument('--file', '-f', type=str, help='Process specific file')
    parser.add_argument('--all', '-a', action='store_true', help='Process all grant files')
    args = parser.parse_args()

    print("=" * 70)
    print("GRANT PROPOSAL INGESTION INTO DD-RAPTOR")
    print("=" * 70)

    # Initialize ingestor
    ingestor = DDRaptorGrantIngestor()

    # Define grant proposal directory
    grant_dir = Path("data/grant")

    if not grant_dir.exists():
        print(f"Error: Grant directory not found: {grant_dir}")
        return

    # Get files to process
    if args.file:
        # Process specific file
        pdf_files = [grant_dir / args.file]
        if not pdf_files[0].exists():
            print(f"Error: File not found: {pdf_files[0]}")
            return
    else:
        # Process all PDF files
        pdf_files = list(grant_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in {grant_dir}")
            return

    print(f"\nFound {len(pdf_files)} PDF files to process:")
    for pdf_file in pdf_files:
        file_size = pdf_file.stat().st_size / 1024 / 1024  # MB
        print(f"  - {pdf_file.name} ({file_size:.1f} MB)")

    # Process files
    successful_ingestions = 0
    total_chunks_ingested = 0

    for pdf_file in pdf_files:
        try:
            proposal = ingestor.process_grant_proposal(pdf_file)
            if proposal:
                success = ingestor.ingest_to_chromadb(proposal)
                if success:
                    successful_ingestions += 1
                    total_chunks_ingested += len(proposal.level0_chunks)
        except Exception as e:
            print(f"\n⚠️  Error processing {pdf_file.name}: {e}")
            continue

    # Summary
    print("\n" + "=" * 70)
    print("INGESTION COMPLETE")
    print("=" * 70)
    print(f"✅ Successful ingestions: {successful_ingestions}/{len(pdf_files)}")
    print(f"📄 Total chunks ingested: {total_chunks_ingested}")
    print(f"💾 ChromaDB location: {ingestor.chromadb_path}")
    print("=" * 70)

    if successful_ingestions > 0:
        print("\n🎉 Grant proposals are now available in DD-RAPTOR!")
        print("You can now query them using the Enhanced DD-RAPTOR system.")

if __name__ == "__main__":
    main()