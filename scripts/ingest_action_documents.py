#!/usr/bin/env python3
"""Ingest documents from data/action folder into RAG system."""

import asyncio
import sys
from pathlib import Path
from typing import List
import hashlib
from datetime import datetime
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.knowledge_base.embedding import EmbeddingService


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

    # Extract text
    text = extract_text_from_pdf(file_path)

    if not text or len(text) < 100:
        print(f"   ⚠️  No text extracted or too short ({len(text)} chars)")
        return 0

    print(f"   ✅ Extracted {len(text):,} characters")

    # Chunk the text
    chunks = chunk_text(text, chunk_size=1500, overlap=200)
    print(f"   📦 Created {len(chunks)} chunks")

    # Generate embeddings using local model (batch processing for efficiency)
    print(f"   🧮 Generating embeddings with local model...")
    try:
        # Use batch encoding for better performance
        embeddings_array = await embedding_service.encode_async(chunks)
        embeddings = embeddings_array.tolist()  # Convert numpy array to list
        print(f"   ✅ Generated {len(embeddings)} embeddings")
    except Exception as e:
        print(f"   ❌ Embedding generation failed: {e}")
        return 0

    # Prepare metadata
    doc_id = file_path.stem[:80]
    metadatas = []
    ids = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}_{i}_{hashlib.md5(chunk.encode()).hexdigest()[:8]}"
        ids.append(chunk_id)

        metadata = {
            "document_id": doc_id,
            "document_type": "action",  # Special type for action documents
            "source_file": file_path.name,
            "source_folder": "data/action",
            "chunk_index": i,
            "total_chunks": len(chunks),
            "timestamp": datetime.now().isoformat(),
            "content_length": len(chunk),
            "file_type": "pdf"
        }
        metadatas.append(metadata)

    # Add to vector store
    print(f"   💾 Storing in ChromaDB...")
    try:
        # Use separate collection for action documents (different embedding dimension)
        collection = chroma_client.get_or_create_collection(
            name="action_documents",
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
    # Target folder
    action_folder = Path("/Users/jiookcha/Documents/git/AI-CoScientist/data/action")

    if not action_folder.exists():
        print(f"❌ Folder not found: {action_folder}")
        return

    # Get all PDF files
    pdf_files = sorted(action_folder.glob("*.pdf"))

    if not pdf_files:
        print(f"❌ No PDF files found in {action_folder}")
        return

    print(f"\n{'='*70}")
    print(f"📚 Ingesting Documents from data/action")
    print(f"{'='*70}")
    print(f"Source folder: {action_folder}")
    print(f"PDF files found: {len(pdf_files)}")
    print(f"Target collection: action_documents (separate from research papers)")
    print(f"Embedding model: SciBERT (768-dim, local)")
    print()

    # List files to be ingested
    print("Files to ingest:")
    for i, pdf_file in enumerate(pdf_files, 1):
        size_kb = pdf_file.stat().st_size / 1024
        print(f"  {i:2d}. {pdf_file.name} ({size_kb:.1f} KB)")
    print()

    # Initialize services
    print("🔧 Initializing services...")
    print("   Using local sentence-transformers (no API key needed)")

    # Use local sentence-transformers embedding (no API quota issues)
    embedding_service = EmbeddingService()

    # Use PersistentClient
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

    # Check existing collection
    try:
        collection = chroma_client.get_collection("action_documents")
        existing_count = collection.count()
        print(f"📊 Current collection status:")
        print(f"   Existing documents in action_documents: {existing_count}")
        print()
    except Exception:
        print("📊 Collection 'action_documents' will be created\n")

    # Process each document
    total_chunks = 0
    successful = 0
    failed = 0

    for i, pdf_file in enumerate(pdf_files, 1):
        try:
            chunks_added = await ingest_document(
                file_path=pdf_file,
                embedding_service=embedding_service,
                chroma_client=chroma_client,
                doc_index=i,
                total_docs=len(pdf_files)
            )

            if chunks_added > 0:
                total_chunks += chunks_added
                successful += 1
            else:
                failed += 1

        except Exception as e:
            print(f"\n[{i}/{len(pdf_files)}] ❌ Error: {e}")
            failed += 1

    # Summary
    print(f"\n{'='*70}")
    print(f"✅ INGESTION COMPLETE")
    print(f"{'='*70}")
    print(f"Documents processed: {successful + failed}/{len(pdf_files)}")
    print(f"  ✅ Successful: {successful}")
    print(f"  ❌ Failed: {failed}")
    print(f"Total chunks added: {total_chunks}")
    print(f"Collection: action_documents")
    print(f"{'='*70}\n")

    # Verify
    try:
        collection = chroma_client.get_collection("action_documents")
        final_count = collection.count()
        print(f"📊 ChromaDB verification:")
        print(f"   Total documents in action_documents collection: {final_count}")
        print(f"   Book chapters ingested: {successful}/{len(pdf_files)}")
    except Exception as e:
        print(f"⚠️  Verification error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
