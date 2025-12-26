#!/usr/bin/env python3
"""
Test search functionality on NeurIPS 2025 ChromaDB collection.

This script demonstrates basic search queries across the hierarchical RAPTOR structure.

Usage:
    poetry run python scripts/test_neurips_2025_search.py
"""

import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer


def test_search():
    """Test search functionality on NeurIPS 2025 collections."""

    print("=" * 70)
    print("NEURIPS 2025 CHROMADB - SEARCH FUNCTIONALITY TEST")
    print("=" * 70)
    print()

    # Load embedding model (same as used for indexing)
    print("Loading SciBERT embedding model...")
    embedding_model = SentenceTransformer('allenai/scibert_scivocab_uncased')
    print("✓ SciBERT loaded")
    print()

    # Connect to ChromaDB
    db_path = "chromadb_data_neurips2025"
    print(f"Connecting to ChromaDB at: {db_path}")
    client = chromadb.PersistentClient(path=db_path)

    # Get collections
    collection_l0 = client.get_collection("neurips_2025_L0")
    collection_l1 = client.get_collection("neurips_2025_L1")
    collection_l2 = client.get_collection("neurips_2025_L2")

    print(f"✓ Connected to 3 collections")
    print(f"  - L0 (chunks): {collection_l0.count()} documents")
    print(f"  - L1 (sections): {collection_l1.count()} documents")
    print(f"  - L2 (papers): {collection_l2.count()} documents")
    print()

    # Test queries
    test_queries = [
        ("brain foundation models", "Level 2 - Paper Summaries"),
        ("transformer architecture for EEG", "Level 0 - Detailed Chunks"),
        ("multimodal learning", "Level 1 - Section Summaries"),
        ("scientific reasoning with LLMs", "Level 2 - Paper Summaries"),
        ("neural interfaces", "Level 0 - Detailed Chunks")
    ]

    for query_text, level_desc in test_queries:
        print("-" * 70)
        print(f"QUERY: '{query_text}'")
        print(f"LEVEL: {level_desc}")
        print("-" * 70)

        # Determine which collection to search
        if "Level 2" in level_desc:
            collection = collection_l2
            n_results = 3
        elif "Level 1" in level_desc:
            collection = collection_l1
            n_results = 5
        else:
            collection = collection_l0
            n_results = 5

        # Generate query embedding
        query_embedding = embedding_model.encode([query_text])[0].tolist()

        # Perform search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        # Display results
        if results['documents'][0]:
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ), 1):
                print(f"\n{i}. Distance: {distance:.4f}")
                print(f"   Paper: {metadata.get('paper_title', 'Unknown')[:80]}")

                if 'section' in metadata:
                    print(f"   Section: {metadata['section']}")

                # Show snippet
                snippet = doc[:200] + "..." if len(doc) > 200 else doc
                print(f"   Content: {snippet}")

        else:
            print("No results found")

        print()

    print("=" * 70)
    print("SEARCH TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    test_search()
