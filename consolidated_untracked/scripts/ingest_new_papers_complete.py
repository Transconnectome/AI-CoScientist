#!/usr/bin/env python3
"""
Complete New Papers Ingestion (paper1-4 + ESM3)

This script processes and ingests all new paper PDFs including ESM3 into the RAG system
with intelligent chunking and comprehensive metadata extraction.

Usage:
    poetry run python scripts/ingest_new_papers_complete.py
"""

import asyncio
import json
import sys
import os
import re
from pathlib import Path
import PyPDF2
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

@dataclass
class PaperChunk:
    """Paper text chunk with metadata."""
    chunk_id: str
    content: str
    section: str
    chunk_index: int
    total_chunks: int
    embedding: Optional[List[float]] = None
    metadata: Dict = field(default_factory=dict)

@dataclass
class PaperDocument:
    """Complete paper document with hierarchical structure."""
    paper_id: str
    title: str
    file_path: str
    paper_type: str
    sections: List[Dict] = field(default_factory=list)
    level0_chunks: List[PaperChunk] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

class SmartPDFExtractor:
    """Enhanced PDF extraction with better error handling."""

    @staticmethod
    def extract_text(pdf_path: Path) -> str:
        """Extract text from PDF with multiple fallback methods."""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""

                for page_num in range(len(reader.pages)):
                    try:
                        page = reader.pages[page_num]
                        page_text = page.extract_text()
                        text += page_text + "\n"
                    except Exception as e:
                        print(f"    Warning: Page {page_num + 1} extraction failed: {e}")
                        continue

                # Enhanced text cleaning
                text = re.sub(r'\n+', '\n', text)
                text = re.sub(r'\s+', ' ', text)
                text = re.sub(r'[^\w\s\-\.\,\!\?\:\;\(\)\[\]\"\'\/\&\@\#\$\%\^\*\+\=\<\>\|\\\n]', '', text)
                return text.strip()

        except Exception as e:
            print(f"    Error extracting PDF {pdf_path.name}: {e}")
            return ""

    @staticmethod
    def identify_paper_type(filename: str, text: str) -> str:
        """Identify paper type based on content."""

        filename_lower = filename.lower()
        text_lower = text.lower()

        # ESM3/Protein related
        if any(keyword in text_lower for keyword in ['esm3', 'evolutionary scale modeling', 'protein language model', 'meta ai']):
            return "ESM3/Protein"

        # AI/ML papers
        if any(keyword in text_lower for keyword in ['machine learning', 'deep learning', 'neural network', 'transformer']):
            return "AI/ML"

        # Neuroscience
        if any(keyword in text_lower for keyword in ['neuroscience', 'brain', 'neural', 'fmri', 'eeg']):
            return "Neuroscience"

        # Quantum
        if any(keyword in text_lower for keyword in ['quantum', 'qubit', 'quantum computing']):
            return "Quantum"

        # Biology/Biotech
        if any(keyword in text_lower for keyword in ['protein', 'dna', 'genome', 'biological', 'molecular']):
            return "Biology"

        return "General"

    @staticmethod
    def extract_metadata(filename: str, text: str) -> Dict:
        """Extract enhanced metadata from paper content."""

        paper_type = SmartPDFExtractor.identify_paper_type(filename, text)

        # Try to extract title from content
        title = filename.replace('.pdf', '').replace('_', ' ')

        # Enhanced title patterns
        title_patterns = [
            r'^(.{20,150})[\n\r]',  # First substantial line
            r'Title[:\s]+(.+?)[\n\.]',
            r'arXiv:[0-9]+\.[0-9]+v[0-9]+.*?[\n\r](.+?)[\n\r]',  # arXiv pattern
            r'^\s*([A-Z][^\.]{20,100})[\n\r]',  # Capitalized line
        ]

        for pattern in title_patterns:
            match = re.search(pattern, text[:3000], re.IGNORECASE | re.MULTILINE)
            if match:
                potential_title = match.group(1).strip()
                # Filter out common non-title patterns
                if (len(potential_title) > 15 and len(potential_title) < 200 and
                    not any(skip in potential_title.lower() for skip in ['arxiv', 'submitted', 'copyright', 'page'])):
                    title = potential_title
                    break

        # Try to extract authors
        authors = "Unknown"
        author_patterns = [
            r'Authors?[:\s]+(.+?)[\n\.]',
            r'By[:\s]+(.+?)[\n\.]',
            r'([A-Z][a-z]+ [A-Z][a-z]+(?:,\s*[A-Z][a-z]+ [A-Z][a-z]+)*)',  # Name patterns
        ]

        for pattern in author_patterns:
            match = re.search(pattern, text[:2000], re.IGNORECASE | re.MULTILINE)
            if match:
                authors = match.group(1).strip()
                break

        # Try to extract year/date
        year = datetime.now().year
        year_patterns = [
            r'20[12][0-9]',  # 2010-2029
            r'arXiv:[0-9]+\.([0-9]{2})[0-9]+',  # arXiv year
        ]

        for pattern in year_patterns:
            match = re.search(pattern, text[:1000])
            if match:
                if len(match.group()) == 4:  # Full year
                    year = int(match.group())
                else:  # Year from arXiv
                    year_suffix = int(match.group(1))
                    year = 2000 + year_suffix if year_suffix > 90 else 2000 + year_suffix
                break

        return {
            'title': title,
            'authors': authors,
            'year': year,
            'paper_type': paper_type,
            'file_size_mb': round(len(text) / 1024 / 1024 * 1.5, 2),
            'word_count': len(text.split()),
            'extracted_at': datetime.now().isoformat(),
            'source': 'new_papers_batch'
        }

class IntelligentChunker:
    """Smart chunking with section awareness."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def create_chunks(self, text: str, paper_id: str, metadata: Dict) -> List[PaperChunk]:
        """Create intelligent chunks from paper text."""

        # Try to detect sections first
        sections = self._detect_sections(text)

        if sections:
            # Process by sections
            chunks = self._chunk_by_sections(sections, paper_id, metadata)
        else:
            # Fall back to simple chunking
            chunks = self._chunk_by_words(text, paper_id, metadata)

        # Update total chunk count
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

    def _detect_sections(self, text: str) -> List[Dict]:
        """Detect paper sections."""

        section_patterns = [
            # Standard academic sections
            r'(?:^|\n)\s*(?:Abstract|ABSTRACT)\s*(?:\n|:)',
            r'(?:^|\n)\s*(?:Introduction|INTRODUCTION)\s*(?:\n|:)',
            r'(?:^|\n)\s*(?:Background|BACKGROUND)\s*(?:\n|:)',
            r'(?:^|\n)\s*(?:Methods?|METHODS?|Methodology|METHODOLOGY)\s*(?:\n|:)',
            r'(?:^|\n)\s*(?:Results?|RESULTS?)\s*(?:\n|:)',
            r'(?:^|\n)\s*(?:Discussion|DISCUSSION)\s*(?:\n|:)',
            r'(?:^|\n)\s*(?:Conclusion|CONCLUSION|Conclusions|CONCLUSIONS)\s*(?:\n|:)',
            r'(?:^|\n)\s*(?:References?|REFERENCES?)\s*(?:\n|:)',
            # Numbered sections
            r'(?:^|\n)\s*\d+\.?\s+([A-Z][a-z\s]+)\s*(?:\n|:)',
            # ESM3/ML specific sections
            r'(?:^|\n)\s*(?:Model|MODEL|Architecture|ARCHITECTURE)\s*(?:\n|:)',
            r'(?:^|\n)\s*(?:Training|TRAINING|Experiments?|EXPERIMENTS?)\s*(?:\n|:)',
        ]

        sections = []
        section_positions = []

        for pattern in section_patterns:
            for match in re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE):
                section_positions.append((match.start(), match.group().strip()))

        if not section_positions:
            return []

        # Sort by position and extract content
        section_positions.sort(key=lambda x: x[0])

        for i, (start_pos, section_name) in enumerate(section_positions):
            if i < len(section_positions) - 1:
                next_pos = section_positions[i + 1][0]
                content = text[start_pos:next_pos].strip()
            else:
                content = text[start_pos:].strip()

            if len(content.split()) >= 20:  # Minimum words per section
                sections.append({
                    'name': re.sub(r'^\d+\.?\s*', '', section_name).replace(':', '').strip(),
                    'content': content,
                    'order': i
                })

        return sections

    def _chunk_by_sections(self, sections: List[Dict], paper_id: str, metadata: Dict) -> List[PaperChunk]:
        """Chunk text by sections with overlap."""
        chunks = []

        for section in sections:
            section_chunks = self._chunk_section_content(
                section['content'],
                section['name'],
                paper_id,
                metadata,
                section['order']
            )
            chunks.extend(section_chunks)

        return chunks

    def _chunk_section_content(self, content: str, section_name: str, paper_id: str, metadata: Dict, section_order: int) -> List[PaperChunk]:
        """Chunk individual section content."""
        words = content.split()

        if len(words) <= self.chunk_size:
            # Section fits in one chunk
            return [PaperChunk(
                chunk_id=f"{paper_id}_{section_name}_chunk_0",
                content=content,
                section=section_name,
                chunk_index=0,
                total_chunks=1,
                metadata={
                    'section_name': section_name,
                    'section_order': section_order,
                    'paper_id': paper_id,
                    'paper_title': metadata.get('title', 'Unknown'),
                    'paper_type': metadata.get('paper_type', 'Unknown'),
                    'authors': metadata.get('authors', 'Unknown'),
                    'year': metadata.get('year', 'Unknown'),
                    'source': 'new_papers_ingestion'
                }
            )]

        # Split with overlap
        chunks = []
        chunk_index = 0

        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)

            if not chunk_text.strip():
                continue

            chunk = PaperChunk(
                chunk_id=f"{paper_id}_{section_name}_chunk_{chunk_index}",
                content=chunk_text,
                section=section_name,
                chunk_index=chunk_index,
                total_chunks=0,  # Will be updated later
                metadata={
                    'section_name': section_name,
                    'section_order': section_order,
                    'paper_id': paper_id,
                    'paper_title': metadata.get('title', 'Unknown'),
                    'paper_type': metadata.get('paper_type', 'Unknown'),
                    'authors': metadata.get('authors', 'Unknown'),
                    'year': metadata.get('year', 'Unknown'),
                    'source': 'new_papers_ingestion',
                    'chunk_start_word': i,
                    'chunk_end_word': i + len(chunk_words)
                }
            )

            chunks.append(chunk)
            chunk_index += 1

        return chunks

    def _chunk_by_words(self, text: str, paper_id: str, metadata: Dict) -> List[PaperChunk]:
        """Fallback: simple word-based chunking."""
        words = text.split()
        chunks = []
        chunk_index = 0

        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)

            if not chunk_text.strip():
                continue

            chunk = PaperChunk(
                chunk_id=f"{paper_id}_main_chunk_{chunk_index}",
                content=chunk_text,
                section='main',
                chunk_index=chunk_index,
                total_chunks=0,  # Will be updated later
                metadata={
                    'section_name': 'main',
                    'section_order': 0,
                    'paper_id': paper_id,
                    'paper_title': metadata.get('title', 'Unknown'),
                    'paper_type': metadata.get('paper_type', 'Unknown'),
                    'authors': metadata.get('authors', 'Unknown'),
                    'year': metadata.get('year', 'Unknown'),
                    'source': 'new_papers_ingestion',
                    'chunk_start_word': i,
                    'chunk_end_word': i + len(chunk_words)
                }
            )

            chunks.append(chunk)
            chunk_index += 1

        return chunks

class CompletePaperIngestor:
    """Complete paper ingestion system for new papers."""

    def __init__(self):
        self.pdf_extractor = SmartPDFExtractor()
        self.chunker = IntelligentChunker(chunk_size=800, overlap=80)

        # Initialize embedding model
        print("🔄 SciBERT 임베딩 모델 로딩 중...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Consistent with previous work
        print(f"✅ 임베딩 모델 로드 완료 (차원: {self.embedding_model.get_sentence_embedding_dimension()})")

    def process_paper(self, pdf_path: Path) -> Optional[PaperDocument]:
        """Process a single paper PDF."""

        print(f"\n📄 처리 중: {pdf_path.name}")
        print("-" * 50)

        # 1. Extract text
        print("1. PDF 텍스트 추출...")
        text = self.pdf_extractor.extract_text(pdf_path)

        if not text or len(text) < 500:
            print("  ❌ 텍스트 추출 실패 또는 내용 부족")
            return None

        print(f"  ✅ {len(text):,} 문자 추출")

        # 2. Extract metadata
        print("2. 메타데이터 추출...")
        metadata = self.pdf_extractor.extract_metadata(pdf_path.name, text)
        print(f"  📋 제목: {metadata['title'][:60]}{'...' if len(metadata['title']) > 60 else ''}")
        print(f"  👤 저자: {metadata['authors'][:40]}{'...' if len(metadata['authors']) > 40 else ''}")
        print(f"  📅 연도: {metadata['year']}")
        print(f"  🏷️  타입: {metadata['paper_type']}")

        # 3. Create chunks
        print("3. 청킹 처리...")
        paper_id = pdf_path.stem.replace(' ', '_')
        chunks = self.chunker.create_chunks(text, paper_id, metadata)
        print(f"  ✅ {len(chunks)} 청크 생성")

        # 4. Generate embeddings
        print("4. 임베딩 생성...")
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.encode(chunk_texts, show_progress_bar=False)

        # Assign embeddings
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding.tolist()

        print(f"  ✅ {len(embeddings)} 임베딩 완료")

        # Create paper document
        paper = PaperDocument(
            paper_id=paper_id,
            title=metadata['title'],
            file_path=str(pdf_path),
            paper_type=metadata['paper_type'],
            level0_chunks=chunks,
            metadata=metadata
        )

        print(f"  🎉 논문 처리 완료!")
        return paper

    def ingest_to_chromadb(self, papers: List[PaperDocument]) -> bool:
        """Ingest all papers into ChromaDB."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        db_path = f"chromadb_new_papers_{timestamp}"

        print(f"\n💾 ChromaDB 생성: {db_path}")

        try:
            # Create fresh ChromaDB
            client = chromadb.PersistentClient(path=db_path)
            collection = client.get_or_create_collection(
                name="new_papers",
                metadata={"description": "New papers including ESM3 with enhanced metadata"}
            )

            # Prepare all data
            all_ids = []
            all_embeddings = []
            all_documents = []
            all_metadatas = []

            for paper in papers:
                for chunk in paper.level0_chunks:
                    all_ids.append(chunk.chunk_id)
                    all_embeddings.append(chunk.embedding)
                    all_documents.append(chunk.content)

                    # Enhanced metadata
                    meta = chunk.metadata.copy()
                    meta.update({
                        'ingested_at': datetime.now().isoformat(),
                        'embedding_model': 'all-MiniLM-L6-v2',
                        'embedding_dimensions': 384,
                        'ingestion_batch': 'new_papers_complete'
                    })
                    all_metadatas.append(meta)

            # Batch ingestion
            print(f"📥 {len(all_ids)} 청크 ChromaDB 입력...")

            batch_size = 50
            for i in tqdm(range(0, len(all_ids), batch_size), desc="배치 처리"):
                batch_end = min(i + batch_size, len(all_ids))

                collection.add(
                    ids=all_ids[i:batch_end],
                    embeddings=all_embeddings[i:batch_end],
                    documents=all_documents[i:batch_end],
                    metadatas=all_metadatas[i:batch_end]
                )

            # Verify ingestion
            final_count = collection.count()
            print(f"✅ ChromaDB 입력 완료: {final_count} 문서")

            # Store database path for later use
            with open("latest_papers_db_path.txt", "w") as f:
                f.write(db_path)

            return True

        except Exception as e:
            print(f"❌ ChromaDB 입력 실패: {e}")
            return False

def main():
    """Main execution."""

    print("=" * 70)
    print("🚀 새로운 논문들 완전 RAG 수집 (paper1-4 + ESM3)")
    print("=" * 70)

    # Find new paper files
    paper_files = [
        Path("data/grant/paper1.pdf"),
        Path("data/grant/paper2.pdf"),
        Path("data/grant/paper3.pdf"),
        Path("data/grant/paper4.pdf")
    ]

    # Check which files exist
    existing_files = [f for f in paper_files if f.exists()]

    if not existing_files:
        print("❌ 새로운 논문 파일들을 찾을 수 없습니다!")
        print("확인된 위치: data/grant/paper*.pdf")
        return

    print(f"\n📚 발견된 새 논문들 ({len(existing_files)}개):")
    total_size = 0
    for pdf_file in existing_files:
        file_size = pdf_file.stat().st_size / 1024 / 1024  # MB
        total_size += file_size
        print(f"  📄 {pdf_file.name} ({file_size:.1f} MB)")

    print(f"  📊 총 크기: {total_size:.1f} MB")

    # Initialize ingestor
    ingestor = CompletePaperIngestor()

    # Process all papers
    processed_papers = []
    successful_count = 0
    total_chunks = 0

    for pdf_file in existing_files:
        try:
            paper = ingestor.process_paper(pdf_file)
            if paper:
                processed_papers.append(paper)
                successful_count += 1
                total_chunks += len(paper.level0_chunks)
        except Exception as e:
            print(f"\n❌ {pdf_file.name} 처리 실패: {e}")
            continue

    if not processed_papers:
        print("\n❌ 처리된 논문이 없습니다!")
        return

    # Ingest into ChromaDB
    print(f"\n📊 처리 완료 요약:")
    print(f"  ✅ 성공한 논문: {successful_count}/{len(existing_files)}")
    print(f"  📄 총 청크 수: {total_chunks}")

    print(f"\n💾 ChromaDB 수집 시작...")
    success = ingestor.ingest_to_chromadb(processed_papers)

    if success:
        print(f"\n🎉 모든 새로운 논문 RAG 수집 완료!")
        print(f"{'='*70}")
        print(f"✅ 처리된 논문: {successful_count}개")
        print(f"📄 총 청크: {total_chunks}개")

        # Show paper types found
        paper_types = {}
        esm3_found = False

        for paper in processed_papers:
            paper_type = paper.paper_type
            if paper_type not in paper_types:
                paper_types[paper_type] = 0
            paper_types[paper_type] += 1

            if 'esm3' in paper.title.lower() or paper_type == "ESM3/Protein":
                esm3_found = True

        print(f"\n📊 발견된 논문 타입들:")
        for ptype, count in paper_types.items():
            print(f"  {ptype}: {count}개")

        if esm3_found:
            print(f"\n🧬 ESM3 관련 논문 발견! 이제 ESM3 검색이 가능합니다.")

        print(f"\n💡 다음 단계: 검색 테스트 실행")
        print(f"poetry run python scripts/test_new_papers_search.py")

if __name__ == "__main__":
    main()