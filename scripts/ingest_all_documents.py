#!/usr/bin/env python3
"""Ingest all unique documents from papers_collection into RAG system."""

import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Any
import hashlib
from datetime import datetime
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.embeddings import EmbeddingService


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF."""
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
            return '\n\n'.join(text)
    except Exception as e:
        print(f"      ⚠️  PDF extraction failed: {e}")
        return ""


def extract_text_from_docx(docx_path: Path) -> str:
    """Extract text from DOCX."""
    try:
        from docx import Document
        doc = Document(docx_path)
        text = []
        for paragraph in doc.paragraphs:
            if paragraph.text:
                text.append(paragraph.text)
        return '\n\n'.join(text)
    except ImportError:
        print("      ⚠️  python-docx not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "python-docx"])
        return extract_text_from_docx(docx_path)
    except Exception as e:
        print(f"      ⚠️  DOCX extraction failed: {e}")
        return ""


def extract_text_from_doc(doc_path: Path) -> str:
    """Extract text from DOC (old format)."""
    # DOC files are binary, need special handling
    # For now, skip or try converting to DOCX
    print(f"      ⚠️  DOC format not supported, skipping")
    return ""


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)

            if break_point > chunk_size * 0.5:
                chunk = chunk[:break_point + 1]
                end = start + break_point + 1

        chunks.append(chunk.strip())
        start = end - overlap

    return chunks


async def ingest_document(
    file_path: Path,
    embedding_service: EmbeddingService,
    chroma_client,
    doc_index: int,
    total_docs: int
) -> int:
    """Ingest a single document into RAG.

    Returns:
        Number of chunks added (0 if failed)
    """
    print(f"\n[{doc_index}/{total_docs}] {file_path.name}")
    print(f"   Size: {file_path.stat().st_size / 1024:.1f} KB")

    # Extract text based on file type
    suffix = file_path.suffix.lower()

    if suffix == '.pdf':
        text = extract_text_from_pdf(file_path)
    elif suffix == '.docx':
        text = extract_text_from_docx(file_path)
    elif suffix == '.doc':
        text = extract_text_from_doc(file_path)
    else:
        print(f"   ⚠️  Unsupported file type: {suffix}")
        return 0

    if not text or len(text) < 100:
        print(f"   ⚠️  No text extracted or too short ({len(text)} chars)")
        return 0

    print(f"   ✅ Extracted {len(text):,} characters")

    # Chunk the text
    chunks = chunk_text(text, chunk_size=1500, overlap=200)
    print(f"   📦 Created {len(chunks)} chunks")

    # Generate embeddings (batch processing for efficiency)
    print(f"   🧮 Generating embeddings...")
    embeddings = []

    for i, chunk in enumerate(chunks):
        try:
            emb = await embedding_service.embed_text(chunk)
            embeddings.append(emb)

            if (i + 1) % 20 == 0:
                print(f"      Progress: {i+1}/{len(chunks)}")
        except Exception as e:
            print(f"      ⚠️  Embedding error at chunk {i}: {e}")
            # Use zero embedding as fallback
            embeddings.append([0.0] * 1536)

    # Prepare metadata
    doc_id = file_path.stem[:80]  # Limit length
    metadatas = []
    ids = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}_{i}_{hashlib.md5(chunk.encode()).hexdigest()[:8]}"
        ids.append(chunk_id)

        metadata = {
            "document_id": doc_id,
            "document_type": "research",
            "source_file": file_path.name,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "timestamp": datetime.now().isoformat(),
            "content_length": len(chunk),
            "file_type": suffix[1:]  # Remove dot
        }
        metadatas.append(metadata)

    # Add to vector store
    print(f"   💾 Storing in ChromaDB...")
    try:
        collection = chroma_client.get_or_create_collection(
            name="research_documents",
            metadata={"hnsw:space": "cosine"}
        )

        # Add in batches
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            end = min(i + batch_size, len(chunks))
            collection.add(
                documents=chunks[i:end],
                embeddings=embeddings[i:end],
                metadatas=metadatas[i:end],
                ids=ids[i:end]
            )

        print(f"   ✅ Added {len(chunks)} chunks")
        return len(chunks)
    except Exception as e:
        print(f"   ❌ Storage error: {e}")
        return 0


async def main():
    """Main entry point."""
    # Load filtered documents list (version-filtered)
    filtered_docs_file = Path("/Users/jiookcha/Documents/git/AI-CoScientist/filtered_documents.txt")

    if not filtered_docs_file.exists():
        print("❌ filtered_documents.txt not found. Run filter_latest_versions.py first.")
        return

    # Read filtered document paths
    with open(filtered_docs_file, 'r', encoding='utf-8') as f:
        file_paths = [Path(line.strip()) for line in f if line.strip()]

    # Filter only supported formats
    supported_files = [
        f for f in file_paths
        if f.suffix.lower() in ['.pdf', '.docx']  # Skip .doc for now
    ]

    print(f"\n{'='*70}")
    print(f"📚 Ingesting Documents into RAG System")
    print(f"{'='*70}")
    print(f"Total unique documents: {len(file_paths)}")
    print(f"Supported formats (PDF, DOCX): {len(supported_files)}")
    print(f"Will process: {len(supported_files)} documents\n")

    # Initialize services
    print("🔧 Initializing services...")

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set. Please set it to use embeddings.")
        return

    embedding_service = EmbeddingService(provider="openai")

    # Use PersistentClient instead of HttpClient
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    chroma_client = chromadb.PersistentClient(
        path="./chromadb_data",
        settings=ChromaSettings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )

    print("✅ Services ready\n")

    # Process each document
    total_chunks = 0
    successful = 0
    failed = 0

    for i, file_path in enumerate(supported_files, 1):
        try:
            if not file_path.exists():
                print(f"\n[{i}/{len(supported_files)}] ⚠️  File not found: {file_path.name}")
                failed += 1
                continue

            chunks_added = await ingest_document(
                file_path=file_path,
                embedding_service=embedding_service,
                chroma_client=chroma_client,
                doc_index=i,
                total_docs=len(supported_files)
            )

            if chunks_added > 0:
                total_chunks += chunks_added
                successful += 1
            else:
                failed += 1

            # Progress update every 10 docs
            if i % 10 == 0:
                print(f"\n{'─'*70}")
                print(f"Progress: {i}/{len(supported_files)} documents")
                print(f"Success: {successful} | Failed: {failed} | Total chunks: {total_chunks}")
                print(f"{'─'*70}")

        except Exception as e:
            print(f"\n[{i}/{len(supported_files)}] ❌ Error: {e}")
            failed += 1

    # Summary
    print(f"\n{'='*70}")
    print(f"✅ INGESTION COMPLETE")
    print(f"{'='*70}")
    print(f"Documents processed: {successful + failed}/{len(supported_files)}")
    print(f"  ✅ Successful: {successful}")
    print(f"  ❌ Failed: {failed}")
    print(f"Total chunks added: {total_chunks}")
    print(f"Collection: research_documents")
    print(f"{'='*70}\n")

    # Verify
    collection = chroma_client.get_collection("research_documents")
    doc_count = collection.count()
    print(f"📊 ChromaDB verification:")
    print(f"   Total documents in collection: {doc_count}")


if __name__ == "__main__":
    asyncio.run(main())
