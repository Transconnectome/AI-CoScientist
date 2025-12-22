#!/usr/bin/env python3
"""
Direct processor for the compressed NRF proposal files
"""

import PyPDF2
import json
from pathlib import Path
import re

def extract_pdf_text(pdf_path: str) -> str:
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

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> list:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())

    return chunks

def process_files():
    """Process the compressed PDF files"""
    files = [
        "../../data/샘플-발달연구_compressed.pdf",
        "../../data/샘플-brainlink_compressed.pdf"
    ]

    processed_data = {}

    for file_path in files:
        if not Path(file_path).exists():
            print(f"File not found: {file_path}")
            continue

        print(f"Processing: {file_path}")

        # Extract text
        text = extract_pdf_text(file_path)
        if not text:
            print(f"  ✗ No text extracted from {file_path}")
            continue

        print(f"  ✓ Extracted {len(text)} characters")

        # Create chunks
        chunks = chunk_text(text)
        print(f"  ✓ Created {len(chunks)} chunks")

        # Store processed data
        file_name = Path(file_path).stem
        processed_data[file_name] = {
            "title": file_name,
            "type": "NRF_MidCareer",
            "text": text,
            "chunks": chunks,
            "metadata": {
                "total_chars": len(text),
                "total_chunks": len(chunks),
                "source_file": file_path
            }
        }

        # Save individual JSON
        output_path = f"../../data/processed_grants/{file_name}.json"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data[file_name], f, ensure_ascii=False, indent=2)

        print(f"  ✓ Saved to: {output_path}")

    return processed_data

if __name__ == "__main__":
    print("=" * 50)
    print("PROCESSING COMPRESSED NRF PROPOSAL FILES")
    print("=" * 50)

    results = process_files()

    print("\n" + "=" * 50)
    print("PROCESSING COMPLETE")
    print("=" * 50)

    for name, data in results.items():
        print(f"✅ {name}: {data['metadata']['total_chars']} chars, {data['metadata']['total_chunks']} chunks")