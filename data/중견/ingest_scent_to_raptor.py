#!/usr/bin/env python3
"""
Ingest Scent Paper (Main + Appendix) into DD-RAPTOR System (Enhanced)

This script processes scent_main.pdf and scent_appendix.pdf with enhanced section parsing,
smart chunking, and high-performance embedding model.

Usage:
    poetry run python data/중견/ingest_scent_to_raptor.py
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Optional
import PyPDF2
import re
from datetime import datetime
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm
from dataclasses import dataclass, field
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

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
    level: int = 1  # 1 for main sections, 2 for subsections
    word_count: int = 0

    def __post_init__(self):
        self.word_count = len(self.content.split())

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
                # 1. Remove multiple newlines
                text = re.sub(r'\n{3,}', '\n\n', text)
                # 2. Fix hyphenation at line breaks (e.g., "algo-\nrithm" -> "algorithm")
                text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
                # 3. Replace non-breaking spaces
                text = text.replace('\xa0', ' ')
                
                return text.strip()

        except Exception as e:
            print(f"  Error extracting PDF: {e}")
            return ""

class EnhancedSectionParser:
    """Advanced parser for scientific papers with hierarchical structure."""

    def parse_sections(self, text: str) -> List[GrantSection]:
        """Parse text into logical sections using multiple heuristic patterns."""

        # Define hierarchical patterns for scientific papers
        section_patterns = [
            # Level 1: "1. Introduction", "2. Methods"
            (1, r'(?:^|\n)\s*(\d+)\.\s+([A-Z][a-zA-Z\s\-\(\)]+)(?:\n|$)'),
            # Level 1: "Abstract", "References", "Appendix" (No number)
            (1, r'(?:^|\n)\s*(Abstract|ABSTRACT|Introduction|INTRODUCTION|Related Work|RELATED WORK|Method|METHOD|Experiments|EXPERIMENTS|Conclusion|CONCLUSION|References|REFERENCES|Appendix|APPENDIX)(?:\n|$)'),
            # Level 1: "A. Appendix Title"
            (1, r'(?:^|\n)\s*([A-Z])\.\s+([A-Z][a-zA-Z\s\-\(\)]+)(?:\n|$)'),
            # Level 2: "2.1. Model Architecture"
            (2, r'(?:^|\n)\s*(\d+\.\d+)\.?\s+([A-Z][a-zA-Z\s\-\(\)]+)(?:\n|$)'),
            # Level 2: "A.1. Proofs"
            (2, r'(?:^|\n)\s*([A-Z]\.\d+)\.?\s+([A-Z][a-zA-Z\s\-\(\)]+)(?:\n|$)'),
        ]

        # Find all potential section headers
        matches = []
        for level, pattern in section_patterns:
            for match in re.finditer(pattern, text, re.MULTILINE):
                # Get the full matched header text
                header = match.group().strip()
                # Clean up the header name (remove numbers)
                if len(match.groups()) >= 2 and match.group(2):
                    name = match.group(2).strip()
                    number = match.group(1).strip()
                    full_name = f"{number} {name}"
                else:
                    full_name = match.group(1).strip()
                
                matches.append({
                    'start': match.start(),
                    'end': match.end(),
                    'name': full_name,
                    'level': level,
                    'raw_header': header
                })

        # Sort matches by position
        matches.sort(key=lambda x: x['start'])

        # Filter overlapping matches (prefer longer/more specific ones or earlier ones)
        # This is a simple greedy filtering
        filtered_matches = []
        if matches:
            current_match = matches[0]
            for next_match in matches[1:]:
                # If next match starts after current match ends, keep current and move to next
                if next_match['start'] >= current_match['end']:
                    filtered_matches.append(current_match)
                    current_match = next_match
                # Else: overlapping. Logic to choose better one could be added here.
                # For now, we stick with the first one found (greedy)
            filtered_matches.append(current_match)

        sections = []
        
        # If no sections found, treat whole text as one
        if not filtered_matches:
            sections.append(GrantSection(
                name="Main Content",
                content=text,
                order=0,
                level=1
            ))
            return sections

        # Create sections from matches
        for i, match in enumerate(filtered_matches):
            start_pos = match['end'] # Content starts after header
            
            if i < len(filtered_matches) - 1:
                end_pos = filtered_matches[i+1]['start']
            else:
                end_pos = len(text)
            
            content = text[start_pos:end_pos].strip()
            
            # Skip empty sections
            if not content:
                continue
                
            sections.append(GrantSection(
                name=match['name'],
                content=content,
                order=i,
                level=match['level']
            ))
            
        return sections

class SmartChunker:
    """Sentence-aware chunking with overlap."""

    def __init__(self, chunk_size_words: int = 300, overlap_words: int = 50):
        self.chunk_size = chunk_size_words
        self.overlap = overlap_words

    def create_chunks(self, sections: List[GrantSection], proposal_id: str) -> List[GrantChunk]:
        """Create chunks from sections respecting sentence boundaries."""
        chunks = []
        global_chunk_index = 0

        for section in sections:
            section_chunks = self._chunk_section(section, proposal_id, global_chunk_index)
            chunks.extend(section_chunks)
            global_chunk_index += len(section_chunks)

        # Update total chunk count
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total_chunks

        return chunks

    def _chunk_section(self, section: GrantSection, proposal_id: str, start_index: int) -> List[GrantChunk]:
        """Chunk a single section."""
        text = section.content
        
        # Split into sentences (simple approximation)
        # Regex for sentence splitting: period/question/exclamation followed by space and uppercase
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        if not sentences:
            return []

        chunks = []
        current_chunk_words = []
        current_chunk_word_count = 0
        current_sentences = []
        
        # Sliding window over sentences
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_words = sentence.split()
            word_count = len(sentence_words)
            
            # If a single sentence is huge, split it by words
            if word_count > self.chunk_size:
                # Force split big sentence
                # (For simplicity, just add it for now, optimization possible later)
                current_sentences.append(sentence)
                current_chunk_word_count += word_count
            else:
                current_sentences.append(sentence)
                current_chunk_word_count += word_count
            
            # Check if chunk is full
            if current_chunk_word_count >= self.chunk_size:
                # Create chunk
                chunk_text = ' '.join(current_sentences)
                chunk_id = f"{proposal_id}_{section.order}_{len(chunks)}"
                
                chunks.append(GrantChunk(
                    chunk_id=chunk_id,
                    content=chunk_text,
                    section=section.name,
                    chunk_index=start_index + len(chunks),
                    total_chunks=0,
                    metadata={
                        'section_name': section.name,
                        'section_level': section.level,
                        'section_order': section.order,
                        'proposal_id': proposal_id
                    }
                ))
                
                # Handle overlap for next chunk
                # Keep last N sentences that sum up to overlap size
                overlap_buffer_words = 0
                overlap_sentences = []
                for s in reversed(current_sentences):
                    s_len = len(s.split())
                    if overlap_buffer_words + s_len <= self.overlap:
                        overlap_sentences.insert(0, s)
                        overlap_buffer_words += s_len
                    else:
                        break
                
                current_sentences = overlap_sentences
                current_chunk_word_count = overlap_buffer_words
            
            i += 1
            
        # Add remaining text as last chunk
        if current_sentences:
            chunk_text = ' '.join(current_sentences)
            if len(chunk_text.strip()) > 50: # Skip very small fragments
                chunks.append(GrantChunk(
                    chunk_id=f"{proposal_id}_{section.order}_{len(chunks)}",
                    content=chunk_text,
                    section=section.name,
                    chunk_index=start_index + len(chunks),
                    total_chunks=0,
                    metadata={
                        'section_name': section.name,
                        'section_level': section.level,
                        'section_order': section.order,
                        'proposal_id': proposal_id
                    }
                ))

        return chunks

def combine_scent_papers(main_path: Path, appendix_path: Path) -> str:
    """Combine main paper and appendix."""
    print("=" * 70)
    print("COMBINING SCENT PAPER FILES")
    print("=" * 70)
    
    extractor = GrantPDFExtractor()
    
    print(f"\n1. Extracting from main paper: {main_path.name}")
    main_text = extractor.extract_text(main_path)
    print(f"   ✓ Extracted {len(main_text):,} characters")
    
    print(f"\n2. Extracting from appendix: {appendix_path.name}")
    appendix_text = extractor.extract_text(appendix_path)
    print(f"   ✓ Extracted {len(appendix_text):,} characters")
    
    combined_text = f"{main_text}\n\n{'='*80}\nAPPENDIX START\n{'='*80}\n\n{appendix_text}"
    return combined_text

def ingest_scent_papers():
    """Main function with improved logic."""
    
    print("=" * 70)
    print("SCENT PAPER INGESTION (ENHANCED)")
    print("=" * 70)
    
    # Configuration
    EMBEDDING_MODEL_NAME = 'all-mpnet-base-v2'  # Better performance than MiniLM
    CHUNK_SIZE = 300  # Smaller chunks (approx 1000 chars)
    OVERLAP = 50      # Overlap to maintain context
    
    # File paths
    data_dir = Path(__file__).parent
    main_pdf = data_dir / "scent_main.pdf"
    appendix_pdf = data_dir / "scent_appendix.pdf"
    
    # Combine papers
    combined_text = combine_scent_papers(main_pdf, appendix_pdf)
    
    if not combined_text:
        print("❌ Error: Failed to extract text")
        return
    
    # Parse sections
    print("\n3. Parsing sections (Enhanced)...")
    section_parser = EnhancedSectionParser()
    sections = section_parser.parse_sections(combined_text)
    print(f"   ✓ Found {len(sections)} sections")
    for i, s in enumerate(sections[:5]): # Show first 5 sections
        print(f"     - {s.name} ({s.word_count} words)")
    if len(sections) > 5:
        print(f"     ... and {len(sections)-5} more")
    
    # Create chunks
    print(f"\n4. Creating chunks (Size={CHUNK_SIZE}, Overlap={OVERLAP})...")
    proposal_id = "scent_paper_combined"
    chunker = SmartChunker(chunk_size_words=CHUNK_SIZE, overlap_words=OVERLAP)
    chunks = chunker.create_chunks(sections, proposal_id)
    print(f"   ✓ Created {len(chunks)} chunks")
    
    # Generate embeddings
    print(f"\n5. Generating embeddings with {EMBEDDING_MODEL_NAME}...")
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = embedding_model.encode(chunk_texts, show_progress_bar=True)
        embeddings_list = [emb.tolist() for emb in embeddings]
        
        for chunk, embedding in zip(chunks, embeddings_list):
            chunk.embedding = embedding
            
        print(f"   ✓ Generated {len(embeddings_list)} embeddings")
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        return

    # Metadata
    metadata = {
        'title': 'SCENT Paper (Main + Appendix)',
        'proposal_type': 'Research Paper',
        'file_size_mb': round((main_pdf.stat().st_size + appendix_pdf.stat().st_size) / 1024 / 1024, 2),
        'word_count': len(combined_text.split()),
        'chunking_strategy': f'SmartChunker(size={CHUNK_SIZE}, overlap={OVERLAP})',
        'embedding_model': EMBEDDING_MODEL_NAME,
        'extracted_at': datetime.now().isoformat()
    }

    # Save to JSON
    print("\n6. Saving to JSON...")
    output_file = data_dir / "scent_paper_chunks_enhanced.json"
    output_data = {
        'chunks': [
            {
                'chunk_id': chunk.chunk_id,
                'content': chunk.content,
                'section': chunk.section,
                'metadata': {
                    **chunk.metadata,
                    'proposal_title': metadata['title'],
                    'embedding_model': metadata['embedding_model']
                },
                'embedding': chunk.embedding
            }
            for chunk in chunks
        ],
        'metadata': metadata
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"   ✓ Saved enhanced chunks to {output_file}")
    
    print("\n" + "=" * 70)
    print("✅ ENHANCED INGESTION COMPLETE")
    print("=" * 70)
    print(f"📄 Model: {EMBEDDING_MODEL_NAME}")
    print(f"📊 Total chunks: {len(chunks)}")
    print(f"📝 Total sections: {len(sections)}")
    print(f"💾 File: {output_file.name}")
    print("=" * 70)

if __name__ == "__main__":
    ingest_scent_papers()
