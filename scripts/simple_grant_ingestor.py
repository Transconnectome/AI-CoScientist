#!/usr/bin/env python3
"""
Simple Grant Proposal Ingestor for DD-RAPTOR

A simplified approach to ingest grant proposal PDFs into the existing DD-RAPTOR system
without causing ChromaDB conflicts.
"""

import asyncio
import json
import sys
from pathlib import Path
import PyPDF2
import re
from typing import List, Dict, Optional
from datetime import datetime
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def extract_pdf_text(pdf_path: Path) -> str:
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

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = ' '.join(chunk_words)

        if chunk_text.strip():
            chunks.append(chunk_text)

    return chunks

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
        'file_size_mb': round(len(text) / 1024 / 1024 * 1.5, 2),
        'word_count': len(text.split()),
        'extracted_at': datetime.now().isoformat(),
        'source': 'grant_proposal'
    }

def save_processed_data(pdf_path: Path, text: str, chunks: List[str], metadata: Dict) -> Path:
    """Save processed data as JSON for later ChromaDB ingestion."""

    # Create output directory
    output_dir = Path("data/processed_grants")
    output_dir.mkdir(exist_ok=True)

    # Prepare data structure compatible with existing DD-RAPTOR loader
    proposal_id = pdf_path.stem.replace(' ', '_')

    # Create chunks with metadata
    level0_chunks = []
    for i, chunk_text in enumerate(chunks):
        chunk_data = {
            'chunk_id': f"{proposal_id}_chunk_{i}",
            'content': chunk_text,
            'embedding': None,  # Will be generated during ChromaDB loading
            'metadata': {
                'proposal_id': proposal_id,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'section': 'main',
                'proposal_title': metadata['title'],
                'proposal_type': metadata['proposal_type'],
                'file_path': str(pdf_path),
                'source': 'grant_proposal'
            }
        }
        level0_chunks.append(chunk_data)

    # Create paper structure compatible with existing system
    paper_data = {
        'paper_id': proposal_id,
        'title': metadata['title'],
        'level0_chunks': level0_chunks,
        'level1_summaries': [],  # Not implemented for now
        'level2_summary': None,  # Not implemented for now
        'metadata': metadata
    }

    # Save to JSON file
    output_file = output_dir / f"{proposal_id}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(paper_data, f, indent=2, ensure_ascii=False)

    return output_file

def main():
    """Main execution."""

    print("=" * 70)
    print("SIMPLE GRANT PROPOSAL PROCESSOR")
    print("=" * 70)

    # Define grant proposal directory
    grant_dir = Path("data/grant")

    if not grant_dir.exists():
        print(f"Error: Grant directory not found: {grant_dir}")
        return

    # Get PDF files
    pdf_files = list(grant_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {grant_dir}")
        return

    print(f"\nFound {len(pdf_files)} PDF files to process:")
    for pdf_file in pdf_files:
        file_size = pdf_file.stat().st_size / 1024 / 1024  # MB
        print(f"  - {pdf_file.name} ({file_size:.1f} MB)")

    processed_files = []

    for pdf_file in pdf_files:
        print(f"\n" + "-" * 50)
        print(f"Processing: {pdf_file.name}")
        print("-" * 50)

        try:
            # 1. Extract text
            print("1. Extracting text...")
            text = extract_pdf_text(pdf_file)

            if not text or len(text) < 500:
                print(f"  ✗ No usable text extracted")
                continue

            print(f"  ✓ Extracted {len(text):,} characters")

            # 2. Estimate metadata
            metadata = estimate_metadata(pdf_file.name, text)
            print(f"  ✓ Title: {metadata['title']}")
            print(f"  ✓ Type: {metadata['proposal_type']}")

            # 3. Create chunks
            print("2. Creating chunks...")
            chunks = chunk_text(text, chunk_size=800, overlap=80)
            print(f"  ✓ Created {len(chunks)} chunks")

            # 4. Save processed data
            print("3. Saving processed data...")
            output_file = save_processed_data(pdf_file, text, chunks, metadata)
            print(f"  ✓ Saved to: {output_file}")

            processed_files.append(output_file)

        except Exception as e:
            print(f"  ✗ Error processing {pdf_file.name}: {e}")
            continue

    # Summary
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"✅ Processed: {len(processed_files)}/{len(pdf_files)} files")
    print(f"📁 Output directory: data/processed_grants/")

    if processed_files:
        print("\n📋 Next steps:")
        print("1. The processed JSON files are ready for ChromaDB ingestion")
        print("2. Use the existing DD-RAPTOR loader to ingest into ChromaDB:")
        print("   poetry run python scripts/load_json_to_chromadb_dd.py")
        print("\nGenerated files:")
        for output_file in processed_files:
            print(f"  - {output_file.name}")

    print("=" * 70)

if __name__ == "__main__":
    main()