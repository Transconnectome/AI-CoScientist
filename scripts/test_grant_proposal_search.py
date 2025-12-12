#!/usr/bin/env python3
"""
Test Grant Proposal Search in ChromaDB

This script tests the ingested grant proposals by running sample searches.

Usage:
    poetry run python scripts/test_grant_proposal_search.py
"""

import chromadb
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

def find_latest_grants_db():
    """Find the latest grant proposal ChromaDB."""
    current_dir = Path(".")
    grant_dbs = list(current_dir.glob("chromadb_grants_*"))

    if not grant_dbs:
        return None

    # Sort by name (which includes timestamp)
    latest_db = sorted(grant_dbs)[-1]
    return str(latest_db)

def test_grant_search():
    """Test searching grant proposals."""

    print("=" * 70)
    print("GRANT PROPOSAL SEARCH TEST")
    print("=" * 70)

    # Find latest grants database
    db_path = find_latest_grants_db()
    if not db_path:
        print("Error: No grant proposal ChromaDB found!")
        print("Please run 'poetry run python scripts/load_grants_fresh_chromadb.py' first.")
        return

    print(f"Using ChromaDB: {db_path}")

    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(name="grant_proposals")

    # Get collection info
    count = collection.count()
    print(f"Collection contains {count} documents")

    # Test queries
    test_queries = [
        "quantum machine learning QuantERA",
        "brain connectome neuroscience",
        "research objectives methodology",
        "multi-chip quantum computing",
        "딥러닝 뇌과학",  # Korean: deep learning neuroscience
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
                proposal_title = meta.get('proposal_title', 'Unknown')
                chunk_idx = meta.get('chunk_index', '?')

                # Show first 100 characters of the document
                preview = doc[:100] + "..." if len(doc) > 100 else doc

                print(f"   [{i+1}] {proposal_type}: {proposal_title}")
                print(f"       Chunk {chunk_idx}: {preview}")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    print("\n" + "=" * 70)
    print("SEARCH TEST COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    test_grant_search()