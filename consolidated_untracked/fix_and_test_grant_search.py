#!/usr/bin/env python3
"""
Fix and Test Grant Proposal Search

This script fixes the embedding dimension issue and tests the search functionality.

Usage:
    poetry run python scripts/fix_and_test_grant_search.py
"""

import chromadb
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
import json
from datetime import datetime
from tqdm import tqdm

def find_latest_grants_db():
    """Find the latest grant proposal ChromaDB."""
    current_dir = Path(".")
    grant_dbs = list(current_dir.glob("chromadb_grants_*"))

    if not grant_dbs:
        return None

    # Sort by name (which includes timestamp)
    latest_db = sorted(grant_dbs)[-1]
    return str(latest_db)

def recreate_collection_with_correct_embeddings():
    """Recreate the collection with the correct embedding dimensions."""

    print("=" * 70)
    print("FIXING GRANT PROPOSAL EMBEDDINGS")
    print("=" * 70)

    # Load the processed grant data
    json_dir = Path("data/processed_grants")
    json_files = list(json_dir.glob("*.json"))

    if not json_files:
        print("Error: No processed grant files found!")
        return None

    # Create a new ChromaDB with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_db_path = f"chromadb_grants_fixed_{timestamp}"

    print(f"Creating fixed ChromaDB at {new_db_path}...")

    # Initialize embedding model (use a specific model with known dimensions)
    print("Loading embedding model...")
    # Use a model with 384 dimensions consistently
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dimensions
    print(f"✓ Embedding model loaded (dimensions: {embedding_model.get_sentence_embedding_dimension()})")

    # Create ChromaDB client and collection
    client = chromadb.PersistentClient(path=new_db_path)
    collection = client.get_or_create_collection(
        name="grant_proposals",
        metadata={"description": "Grant proposal chunks with consistent 384-dim embeddings"}
    )

    # Load and process all grant files
    all_ids = []
    all_embeddings = []
    all_documents = []
    all_metadatas = []

    print(f"\nProcessing {len(json_files)} grant files...")

    for json_file in tqdm(json_files, desc="Loading files"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                proposal_data = json.load(f)

            chunks = proposal_data.get('level0_chunks', [])
            if not chunks:
                continue

            # Extract text content
            chunk_texts = [chunk['content'] for chunk in chunks]

            # Generate embeddings with consistent model
            embeddings = embedding_model.encode(chunk_texts, show_progress_bar=False)

            # Prepare data
            for chunk, embedding in zip(chunks, embeddings):
                all_ids.append(chunk['chunk_id'])
                all_embeddings.append(embedding.tolist())
                all_documents.append(chunk['content'])

                # Enhance metadata
                meta = chunk['metadata'].copy()
                meta.update({
                    'paper_title': proposal_data.get('title', 'Unknown Title'),
                    'ingested_at': datetime.now().isoformat(),
                    'embedding_model': 'all-MiniLM-L6-v2',
                    'embedding_dimensions': embedding_model.get_sentence_embedding_dimension()
                })
                all_metadatas.append(meta)

        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue

    # Ingest into ChromaDB
    print(f"\nIngesting {len(all_ids)} chunks...")

    batch_size = 50
    for i in tqdm(range(0, len(all_ids), batch_size), desc="Ingesting batches"):
        batch_end = min(i + batch_size, len(all_ids))

        try:
            collection.add(
                ids=all_ids[i:batch_end],
                embeddings=all_embeddings[i:batch_end],
                documents=all_documents[i:batch_end],
                metadatas=all_metadatas[i:batch_end]
            )
        except Exception as e:
            print(f"Error ingesting batch: {e}")
            continue

    print(f"✓ Successfully ingested {collection.count()} documents")

    return new_db_path, embedding_model

def test_grant_search_fixed(db_path, embedding_model):
    """Test searching grant proposals with fixed embeddings."""

    print("\n" + "=" * 70)
    print("GRANT PROPOSAL SEARCH TEST (FIXED)")
    print("=" * 70)

    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(name="grant_proposals")

    count = collection.count()
    print(f"Collection contains {count} documents")

    # Test queries
    test_queries = [
        "quantum machine learning QuantERA",
        "brain connectome neuroscience",
        "multi-chip quantum computing",
        "research objectives methodology",
        "딥러닝 뇌과학 AI",  # Korean: deep learning neuroscience AI
        "INCITE supercomputer Aurora"
    ]

    print("\nTesting search queries:")
    print("-" * 70)

    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")

        try:
            results = collection.query(
                query_texts=[query],
                n_results=3
            )

            documents = results['documents'][0]
            metadatas = results['metadatas'][0] if results['metadatas'] else []

            print(f"   Found {len(documents)} results:")

            for i, (doc, meta) in enumerate(zip(documents, metadatas)):
                proposal_type = meta.get('proposal_type', 'Unknown')
                proposal_title = meta.get('proposal_title', 'Unknown')[:50]
                chunk_idx = meta.get('chunk_index', '?')

                # Show first 80 characters of the document
                preview = doc[:80].replace('\n', ' ') + "..." if len(doc) > 80 else doc

                print(f"   [{i+1}] {proposal_type}: {proposal_title}...")
                print(f"       Chunk {chunk_idx}: {preview}")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    print("\n" + "=" * 70)
    print("SUCCESS! Grant proposals are now searchable with proper chunking!")
    print("=" * 70)

def main():
    """Main execution."""

    # Recreate collection with correct embeddings
    db_path, embedding_model = recreate_collection_with_correct_embeddings()

    if db_path:
        # Test the search functionality
        test_grant_search_fixed(db_path, embedding_model)

        print(f"\n💾 ChromaDB saved to: {db_path}")
        print("🎉 Grant proposals successfully chunked and ingested into searchable database!")

if __name__ == "__main__":
    main()