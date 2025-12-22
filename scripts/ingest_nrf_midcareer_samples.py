#!/usr/bin/env python3
"""
Ingest NRF Mid-Career Proposal Samples into RAG System

Processes Korean NRF grant proposal PDFs with:
- Large file handling (up to 50MB)
- Korean/English mixed text support
- RAPTOR 3-level hierarchical indexing
- Dedicated ChromaDB collection for NRF proposals

Usage:
    poetry run python scripts/ingest_nrf_midcareer_samples.py --test    # Test mode (1 file)
    poetry run python scripts/ingest_nrf_midcareer_samples.py --all     # All files
    poetry run python scripts/ingest_nrf_midcareer_samples.py --file "샘플-incite.pdf"
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import argparse
import hashlib

# PDF processing
try:
    import PyPDF2
except ImportError:
    print("Installing PyPDF2...")
    os.system("pip install PyPDF2")
    import PyPDF2

# Embeddings
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Installing sentence-transformers...")
    os.system("pip install sentence-transformers")
    from sentence_transformers import SentenceTransformer

# Vector store
try:
    import chromadb
except ImportError:
    print("Installing chromadb...")
    os.system("pip install chromadb")
    import chromadb

import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# Add project root
sys.path.append(str(Path(__file__).parent.parent))


# ============================================================================
# Configuration
# ============================================================================

# Collection names
COLLECTION_L0 = "nrf_midcareer_samples_L0"  # Chunks
COLLECTION_L1 = "nrf_midcareer_samples_L1"  # Section summaries
COLLECTION_L2 = "nrf_midcareer_samples_L2"  # Document summaries

# Chunking parameters (optimized for large Korean PDFs)
CHUNK_SIZE = 500  # tokens (Korean ~2 chars/token)
CHUNK_OVERLAP = 50

# Source directory
SOURCE_DIR = Path("data/중견")
CHROMADB_PATH = Path("chromadb_data")


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class NRFChunk:
    """NRF proposal chunk with metadata."""
    chunk_id: str
    content: str
    section: str
    chunk_index: int
    total_chunks: int
    page_number: int
    embedding: Optional[List[float]] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class NRFSection:
    """NRF proposal section."""
    name: str
    content: str
    order: int
    start_page: int
    word_count: int = 0

    def __post_init__(self):
        self.word_count = len(self.content.split())


@dataclass
class NRFProposal:
    """Complete NRF proposal document."""
    proposal_id: str
    filename: str
    title: str
    proposal_type: str  # incite, brainlink, samsung, quantERA, etc.
    sections: List[NRFSection] = field(default_factory=list)
    level0_chunks: List[NRFChunk] = field(default_factory=list)
    level1_summaries: List[Dict] = field(default_factory=list)
    level2_summary: Optional[Dict] = None
    metadata: Dict = field(default_factory=dict)


# ============================================================================
# PDF Extraction (Optimized for Large Files)
# ============================================================================

class NRFPDFExtractor:
    """Extract text from large NRF proposal PDFs."""

    @staticmethod
    def extract_text_by_page(pdf_path: Path) -> List[Tuple[int, str]]:
        """
        Extract text page by page for large file handling.

        Returns:
            List of (page_number, text) tuples
        """
        pages = []

        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                total_pages = len(reader.pages)

                print(f"  Extracting {total_pages} pages...")

                for page_num in tqdm(range(total_pages), desc="  Pages", leave=False):
                    try:
                        page = reader.pages[page_num]
                        text = page.extract_text() or ""

                        # Clean text
                        text = NRFPDFExtractor._clean_text(text)

                        if text.strip():
                            pages.append((page_num + 1, text))

                    except Exception as e:
                        print(f"  Warning: Page {page_num + 1} extraction failed: {e}")
                        continue

        except Exception as e:
            print(f"  Error reading PDF: {e}")

        return pages

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        # Remove common PDF artifacts
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
        return text.strip()

    @staticmethod
    def detect_proposal_type(filename: str) -> str:
        """Detect proposal type from filename."""
        filename_lower = filename.lower()

        if 'incite' in filename_lower:
            return 'INCITE'
        elif 'brainlink' in filename_lower:
            return 'BrainLink'
        elif '삼성' in filename_lower or 'samsung' in filename_lower:
            return 'Samsung'
        elif 'quantera' in filename_lower:
            return 'QuantERA'
        elif '발달' in filename_lower:
            return 'Developmental'
        elif '양식' in filename_lower:
            return 'NRF_Template'
        else:
            return 'NRF_General'

    @staticmethod
    def estimate_metadata(filename: str, pages: List[Tuple[int, str]]) -> Dict:
        """Extract metadata from document."""
        full_text = '\n'.join([text for _, text in pages[:5]])  # First 5 pages

        # Try to extract title from first page
        first_page_text = pages[0][1] if pages else ""
        lines = [l.strip() for l in first_page_text.split('\n') if l.strip()]
        title = lines[0][:150] if lines else filename.replace('.pdf', '')

        return {
            'title': title,
            'proposal_type': NRFPDFExtractor.detect_proposal_type(filename),
            'total_pages': len(pages),
            'total_words': sum(len(text.split()) for _, text in pages),
            'extracted_at': datetime.now().isoformat()
        }


# ============================================================================
# Section Detection (Korean NRF Format)
# ============================================================================

class NRFSectionDetector:
    """Detect sections in Korean NRF proposals."""

    # Common Korean NRF proposal section patterns
    SECTION_PATTERNS = [
        # Korean patterns
        (r'연구\s*과제의?\s*필요성', '연구과제의 필요성'),
        (r'연구\s*목표', '연구 목표'),
        (r'연구\s*내용', '연구 내용'),
        (r'연구\s*방법', '연구 방법'),
        (r'추진\s*전략', '추진전략'),
        (r'추진\s*체계', '추진체계'),
        (r'기대\s*효과', '기대효과'),
        (r'연구\s*업적', '연구업적'),
        (r'참고\s*문헌', '참고문헌'),

        # English patterns (for INCITE, QuantERA)
        (r'(?i)abstract', 'Abstract'),
        (r'(?i)introduction', 'Introduction'),
        (r'(?i)background', 'Background'),
        (r'(?i)methodology|methods', 'Methods'),
        (r'(?i)research\s*plan', 'Research Plan'),
        (r'(?i)expected\s*outcomes?', 'Expected Outcomes'),
        (r'(?i)timeline|schedule', 'Timeline'),
        (r'(?i)references?', 'References'),
        (r'(?i)budget', 'Budget'),

        # Aim patterns
        (r'(?i)aim\s*1', 'Aim 1'),
        (r'(?i)aim\s*2', 'Aim 2'),
        (r'(?i)aim\s*3', 'Aim 3'),
    ]

    @classmethod
    def detect_sections(cls, pages: List[Tuple[int, str]]) -> List[NRFSection]:
        """Detect sections from page content."""
        full_text = '\n'.join([text for _, text in pages])

        # Find all section positions
        section_positions = []

        for pattern, name in cls.SECTION_PATTERNS:
            for match in re.finditer(pattern, full_text):
                section_positions.append((match.start(), name))

        # Sort by position
        section_positions.sort(key=lambda x: x[0])

        # Remove duplicates (keep first occurrence)
        seen_names = set()
        unique_sections = []
        for pos, name in section_positions:
            if name not in seen_names:
                unique_sections.append((pos, name))
                seen_names.add(name)

        # Create sections
        sections = []
        for i, (pos, name) in enumerate(unique_sections):
            start = pos
            end = unique_sections[i + 1][0] if i < len(unique_sections) - 1 else len(full_text)
            content = full_text[start:end].strip()

            # Find start page
            start_page = 1
            char_count = 0
            for page_num, text in pages:
                char_count += len(text)
                if char_count >= start:
                    start_page = page_num
                    break

            if len(content) > 100:  # Minimum section length
                sections.append(NRFSection(
                    name=name,
                    content=content,
                    order=i,
                    start_page=start_page
                ))

        # If no sections detected, create single section
        if not sections:
            full_content = '\n'.join([text for _, text in pages])
            sections.append(NRFSection(
                name='full_document',
                content=full_content,
                order=0,
                start_page=1
            ))

        return sections


# ============================================================================
# Chunking System
# ============================================================================

class NRFChunker:
    """Chunk NRF proposals with Korean text support."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_sections(self, sections: List[NRFSection], proposal_id: str) -> List[NRFChunk]:
        """Create chunks from all sections."""
        all_chunks = []
        global_index = 0

        for section in sections:
            section_chunks = self._chunk_section(section, proposal_id, global_index)
            all_chunks.extend(section_chunks)
            global_index += len(section_chunks)

        # Update total count
        for chunk in all_chunks:
            chunk.total_chunks = len(all_chunks)

        return all_chunks

    def _chunk_section(self, section: NRFSection, proposal_id: str, start_index: int) -> List[NRFChunk]:
        """Chunk a single section with overlap."""
        text = section.content

        # Split by sentences (Korean + English)
        sentences = re.split(r'(?<=[.!?。])\s+', text)

        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_index = start_index

        for sentence in sentences:
            if not sentence.strip():
                continue

            # Estimate tokens (Korean: ~2 chars/token, English: ~4 chars/token)
            sentence_tokens = len(sentence) // 3  # Average

            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                chunk_id = f"{proposal_id}_{section.name}_{chunk_index}"

                chunks.append(NRFChunk(
                    chunk_id=chunk_id,
                    content=chunk_text,
                    section=section.name,
                    chunk_index=chunk_index,
                    total_chunks=0,
                    page_number=section.start_page,
                    metadata={
                        'section': section.name,
                        'section_order': section.order,
                        'proposal_id': proposal_id,
                        'token_estimate': current_tokens
                    }
                ))

                chunk_index += 1

                # Keep overlap
                overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk[-1:]
                current_chunk = list(overlap_sentences) + [sentence]
                current_tokens = sum(len(s) // 3 for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens

        # Final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_id = f"{proposal_id}_{section.name}_{chunk_index}"

            chunks.append(NRFChunk(
                chunk_id=chunk_id,
                content=chunk_text,
                section=section.name,
                chunk_index=chunk_index,
                total_chunks=0,
                page_number=section.start_page,
                metadata={
                    'section': section.name,
                    'section_order': section.order,
                    'proposal_id': proposal_id,
                    'token_estimate': current_tokens
                }
            ))

        return chunks


# ============================================================================
# Summary Generator (Simple, no LLM required)
# ============================================================================

class SimpleSummaryGenerator:
    """Generate summaries without LLM (extractive)."""

    @staticmethod
    def generate_section_summary(section: NRFSection) -> str:
        """Extract first 2-3 sentences as summary."""
        sentences = re.split(r'(?<=[.!?。])\s+', section.content)
        summary_sentences = sentences[:3]
        return ' '.join(summary_sentences)[:500]

    @staticmethod
    def generate_document_summary(sections: List[NRFSection], title: str) -> str:
        """Generate document summary from section summaries."""
        summaries = []
        for section in sections[:5]:  # Top 5 sections
            section_summary = SimpleSummaryGenerator.generate_section_summary(section)
            summaries.append(f"{section.name}: {section_summary[:150]}")

        return f"Title: {title}\n\n" + "\n".join(summaries)


# ============================================================================
# ChromaDB Storage
# ============================================================================

class NRFChromaDBStore:
    """Store NRF proposals in ChromaDB."""

    def __init__(self, chromadb_path: str = str(CHROMADB_PATH)):
        print(f"Initializing ChromaDB at {chromadb_path}...")
        self.client = chromadb.PersistentClient(path=chromadb_path)

        # Create collections for each level
        self.collection_l0 = self.client.get_or_create_collection(
            name=COLLECTION_L0,
            metadata={"description": "NRF Mid-Career Proposal Chunks (Level 0)"}
        )
        self.collection_l1 = self.client.get_or_create_collection(
            name=COLLECTION_L1,
            metadata={"description": "NRF Mid-Career Section Summaries (Level 1)"}
        )
        self.collection_l2 = self.client.get_or_create_collection(
            name=COLLECTION_L2,
            metadata={"description": "NRF Mid-Career Document Summaries (Level 2)"}
        )

        print(f"  Collections initialized:")
        print(f"    - {COLLECTION_L0}: {self.collection_l0.count()} existing")
        print(f"    - {COLLECTION_L1}: {self.collection_l1.count()} existing")
        print(f"    - {COLLECTION_L2}: {self.collection_l2.count()} existing")

    def store_chunks(self, chunks: List[NRFChunk]):
        """Store Level 0 chunks."""
        if not chunks:
            return

        self.collection_l0.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=[c.embedding for c in chunks],
            documents=[c.content for c in chunks],
            metadatas=[c.metadata for c in chunks]
        )
        print(f"    Stored {len(chunks)} chunks to L0")

    def store_section_summaries(self, summaries: List[Dict], embeddings: List[List[float]]):
        """Store Level 1 section summaries."""
        if not summaries:
            return

        self.collection_l1.add(
            ids=[s['id'] for s in summaries],
            embeddings=embeddings,
            documents=[s['content'] for s in summaries],
            metadatas=[s['metadata'] for s in summaries]
        )
        print(f"    Stored {len(summaries)} section summaries to L1")

    def store_document_summary(self, summary: Dict, embedding: List[float]):
        """Store Level 2 document summary."""
        self.collection_l2.add(
            ids=[summary['id']],
            embeddings=[embedding],
            documents=[summary['content']],
            metadatas=[summary['metadata']]
        )
        print(f"    Stored document summary to L2")

    def get_stats(self) -> Dict:
        """Get collection statistics."""
        return {
            'l0_chunks': self.collection_l0.count(),
            'l1_sections': self.collection_l1.count(),
            'l2_documents': self.collection_l2.count()
        }


# ============================================================================
# Main Ingestion Pipeline
# ============================================================================

class NRFProposalIngestor:
    """Main ingestion pipeline for NRF proposals."""

    def __init__(self):
        print("\n" + "=" * 70)
        print("NRF MID-CAREER PROPOSAL INGESTION SYSTEM")
        print("=" * 70)

        # Initialize components
        self.pdf_extractor = NRFPDFExtractor()
        self.section_detector = NRFSectionDetector()
        self.chunker = NRFChunker()
        self.summary_gen = SimpleSummaryGenerator()
        self.store = NRFChromaDBStore()

        # Initialize embedding model
        print("\nLoading embedding model (SciBERT)...")
        self.embedding_model = SentenceTransformer('allenai/scibert_scivocab_uncased')
        print("  Model loaded successfully")

    def _generate_proposal_id(self, filename: str) -> str:
        """Generate unique proposal ID."""
        base = filename.replace('.pdf', '').replace(' ', '_')
        # Add hash for uniqueness
        hash_suffix = hashlib.md5(filename.encode()).hexdigest()[:6]
        return f"nrf_{base}_{hash_suffix}"

    async def process_proposal(self, pdf_path: Path) -> Optional[NRFProposal]:
        """Process a single NRF proposal PDF."""
        print("\n" + "-" * 70)
        print(f"Processing: {pdf_path.name}")
        print("-" * 70)

        # 1. Extract text by page
        print("\n1. Extracting PDF text...")
        pages = self.pdf_extractor.extract_text_by_page(pdf_path)

        if not pages:
            print("  ERROR: No text extracted")
            return None

        total_chars = sum(len(text) for _, text in pages)
        print(f"  Extracted {len(pages)} pages, {total_chars:,} characters")

        # 2. Get metadata
        metadata = self.pdf_extractor.estimate_metadata(pdf_path.name, pages)
        proposal_id = self._generate_proposal_id(pdf_path.name)
        print(f"  Title: {metadata['title'][:80]}...")
        print(f"  Type: {metadata['proposal_type']}")
        print(f"  ID: {proposal_id}")

        # 3. Detect sections
        print("\n2. Detecting sections...")
        sections = self.section_detector.detect_sections(pages)
        print(f"  Found {len(sections)} sections:")
        for section in sections:
            print(f"    - {section.name}: {section.word_count} words")

        # 4. Chunk sections
        print("\n3. Chunking document...")
        chunks = self.chunker.chunk_sections(sections, proposal_id)
        print(f"  Created {len(chunks)} chunks")

        # 5. Generate embeddings for chunks
        print("\n4. Generating embeddings...")
        chunk_texts = [c.content for c in chunks]
        chunk_embeddings = self.embedding_model.encode(
            chunk_texts,
            show_progress_bar=True,
            batch_size=32
        )

        for chunk, embedding in zip(chunks, chunk_embeddings):
            chunk.embedding = embedding.tolist()

        # 6. Generate section summaries
        print("\n5. Generating section summaries...")
        section_summaries = []
        for section in sections:
            summary_text = self.summary_gen.generate_section_summary(section)
            section_summaries.append({
                'id': f"{proposal_id}_L1_{section.name}",
                'content': summary_text,
                'metadata': {
                    'section': section.name,
                    'section_order': section.order,
                    'proposal_id': proposal_id,
                    'proposal_type': metadata['proposal_type']
                }
            })

        summary_texts = [s['content'] for s in section_summaries]
        summary_embeddings = self.embedding_model.encode(summary_texts)

        # 7. Generate document summary
        print("\n6. Generating document summary...")
        doc_summary_text = self.summary_gen.generate_document_summary(sections, metadata['title'])
        doc_summary = {
            'id': f"{proposal_id}_L2_document",
            'content': doc_summary_text,
            'metadata': {
                'proposal_id': proposal_id,
                'title': metadata['title'],
                'proposal_type': metadata['proposal_type'],
                'total_pages': metadata['total_pages'],
                'total_sections': len(sections),
                'total_chunks': len(chunks)
            }
        }
        doc_embedding = self.embedding_model.encode([doc_summary_text])[0]

        # 8. Store in ChromaDB
        print("\n7. Storing in ChromaDB...")
        self.store.store_chunks(chunks)
        self.store.store_section_summaries(section_summaries, summary_embeddings.tolist())
        self.store.store_document_summary(doc_summary, doc_embedding.tolist())

        # Create proposal object
        proposal = NRFProposal(
            proposal_id=proposal_id,
            filename=pdf_path.name,
            title=metadata['title'],
            proposal_type=metadata['proposal_type'],
            sections=sections,
            level0_chunks=chunks,
            level1_summaries=section_summaries,
            level2_summary=doc_summary,
            metadata=metadata
        )

        print(f"\n  SUCCESS: {pdf_path.name}")
        return proposal

    async def ingest_all(self, source_dir: Path, limit: Optional[int] = None,
                         specific_file: Optional[str] = None):
        """Ingest all PDFs from source directory."""

        # Find PDFs
        if specific_file:
            pdfs = [source_dir / specific_file]
            if not pdfs[0].exists():
                print(f"ERROR: File not found: {specific_file}")
                return
        else:
            pdfs = sorted(source_dir.glob("*.pdf"))

        if limit:
            pdfs = pdfs[:limit]

        print(f"\nFound {len(pdfs)} PDF files to process")

        # Process each
        results = {
            'success': [],
            'failed': [],
            'total_chunks': 0
        }

        for idx, pdf_path in enumerate(pdfs, 1):
            print(f"\n[{idx}/{len(pdfs)}]")

            try:
                proposal = await self.process_proposal(pdf_path)

                if proposal:
                    results['success'].append(pdf_path.name)
                    results['total_chunks'] += len(proposal.level0_chunks)
                else:
                    results['failed'].append(pdf_path.name)

            except Exception as e:
                print(f"  ERROR: {e}")
                results['failed'].append(pdf_path.name)

        # Print summary
        print("\n" + "=" * 70)
        print("INGESTION COMPLETE")
        print("=" * 70)
        print(f"Success: {len(results['success'])}/{len(pdfs)}")
        print(f"Failed: {len(results['failed'])}/{len(pdfs)}")
        print(f"Total chunks created: {results['total_chunks']}")

        stats = self.store.get_stats()
        print(f"\nChromaDB Statistics:")
        print(f"  L0 (Chunks): {stats['l0_chunks']}")
        print(f"  L1 (Sections): {stats['l1_sections']}")
        print(f"  L2 (Documents): {stats['l2_documents']}")

        # Save results
        results_path = source_dir / "ingestion_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {results_path}")


# ============================================================================
# Main
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Ingest NRF Mid-Career Proposal Samples")
    parser.add_argument("--test", action="store_true", help="Test mode: process 1 file only")
    parser.add_argument("--all", action="store_true", help="Process all PDF files")
    parser.add_argument("--file", type=str, help="Process specific file")
    args = parser.parse_args()

    # Default to test mode
    if not args.test and not args.all and not args.file:
        args.test = True

    ingestor = NRFProposalIngestor()

    if args.file:
        await ingestor.ingest_all(SOURCE_DIR, specific_file=args.file)
    elif args.test:
        await ingestor.ingest_all(SOURCE_DIR, limit=1)
    else:
        await ingestor.ingest_all(SOURCE_DIR, limit=None)


if __name__ == "__main__":
    asyncio.run(main())
