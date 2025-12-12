#!/usr/bin/env python3
"""Extract text from PDFs and add to RAG system."""

import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Any
import hashlib
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.embeddings import EmbeddingService
from src.services.knowledge_base.vector_store_optimized import (
    OptimizedVectorStore,
    VectorStoreConnectionPool
)


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF file.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Extracted text
    """
    try:
        import PyPDF2

        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = []

            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)

            return '\n\n'.join(text)
    except ImportError:
        print("⚠️  PyPDF2 not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "PyPDF2"])
        return extract_text_from_pdf(pdf_path)


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks.

    Args:
        text: Text to split
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks

    Returns:
        List of text chunks
    """
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

            if break_point > chunk_size * 0.5:  # Only break if reasonable
                chunk = chunk[:break_point + 1]
                end = start + break_point + 1

        chunks.append(chunk.strip())
        start = end - overlap

    return chunks


async def add_pdf_to_rag(
    pdf_path: Path,
    embedding_service: EmbeddingService,
    vector_store: OptimizedVectorStore,
    field: str = "psychology",
    document_type: str = "paper"
) -> int:
    """Add PDF to RAG system.

    Args:
        pdf_path: Path to PDF file
        embedding_service: Embedding service
        vector_store: Vector store
        field: Academic field
        document_type: Type of document (paper or book)

    Returns:
        Number of chunks added
    """
    print(f"\n📄 Processing: {pdf_path.name}")
    print(f"   Size: {pdf_path.stat().st_size / 1024:.1f} KB")

    # Extract text
    print("   📝 Extracting text...")
    text = extract_text_from_pdf(pdf_path)

    if not text or len(text) < 100:
        print("   ❌ No text extracted or file too short")
        return 0

    print(f"   ✅ Extracted {len(text):,} characters")

    # Create document ID from filename
    doc_id = pdf_path.stem[:50]  # Limit length

    # Detect document type from filename/content
    if "mindset" in pdf_path.name.lower() or "search for meaning" in pdf_path.name.lower():
        document_type = "book"
    else:
        document_type = "paper"

    # Chunk the text
    print("   ✂️  Chunking text...")
    chunks = chunk_text(text, chunk_size=1500, overlap=200)
    print(f"   ✅ Created {len(chunks)} chunks")

    # Generate embeddings
    print("   🧮 Generating embeddings...")
    embeddings = []
    for i, chunk in enumerate(chunks):
        if i % 10 == 0:
            print(f"      Progress: {i}/{len(chunks)}")
        emb = await embedding_service.embed_text(chunk)
        embeddings.append(emb)

    # Prepare metadata
    metadatas = []
    ids = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}_chunk_{i}_{hashlib.md5(chunk.encode()).hexdigest()[:8]}"
        ids.append(chunk_id)

        metadata = {
            "document_id": doc_id,
            "document_type": document_type,
            "field": field,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "source_file": pdf_path.name,
            "timestamp": datetime.now().isoformat(),
            "content_length": len(chunk)
        }
        metadatas.append(metadata)

    # Add to vector store
    print("   💾 Storing in ChromaDB...")
    await vector_store.add_documents(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
        collection_name="research_documents"
    )

    print(f"   ✅ Added {len(chunks)} chunks to RAG")
    return len(chunks)


async def main():
    """Main entry point."""
    # PDF directory
    pdf_dir = Path("/Users/jiookcha/Documents/git/Mindset/docs/references")

    if not pdf_dir.exists():
        print(f"❌ Directory not found: {pdf_dir}")
        return

    # Get all PDFs
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    print(f"\n{'='*70}")
    print(f"📚 Adding PDFs to RAG System")
    print(f"{'='*70}")
    print(f"Found {len(pdf_files)} PDF files\n")

    # Initialize services
    print("🔧 Initializing services...")
    embedding_service = EmbeddingService(provider="openai")
    connection_pool = VectorStoreConnectionPool(max_connections=5)
    vector_store = OptimizedVectorStore(
        connection_pool=connection_pool,
        collection_name="research_documents"
    )

    print("✅ Services ready\n")

    # Process each PDF
    total_chunks = 0
    successful = 0

    for pdf_file in pdf_files:
        try:
            chunks_added = await add_pdf_to_rag(
                pdf_path=pdf_file,
                embedding_service=embedding_service,
                vector_store=vector_store,
                field="psychology"
            )
            total_chunks += chunks_added
            successful += 1
        except Exception as e:
            print(f"   ❌ Error: {e}")

    # Summary
    print(f"\n{'='*70}")
    print(f"✅ COMPLETE")
    print(f"{'='*70}")
    print(f"Files processed: {successful}/{len(pdf_files)}")
    print(f"Total chunks added: {total_chunks}")
    print(f"Collection: research_documents")
    print(f"{'='*70}\n")

    # Clean up
    await vector_store.close()


if __name__ == "__main__":
    asyncio.run(main())
