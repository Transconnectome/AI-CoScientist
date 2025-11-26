#!/usr/bin/env python3
"""
Quick test script to verify ChromaDB storage and search quality
for Option C RAPTOR golden reference ingestion.
"""

import chromadb
from sentence_transformers import SentenceTransformer

def test_chromadb_storage():
    """Test ChromaDB collections and content."""

    print("=" * 70)
    print("CHROMADB STORAGE VERIFICATION")
    print("=" * 70)

    # Connect to ChromaDB
    client = chromadb.PersistentClient(path="chromadb_data")

    # Get collections
    print("\n1. Checking Collections:")
    collections = client.list_collections()
    print(f"   Total collections: {len(collections)}")

    for collection in collections:
        print(f"\n   Collection: {collection.name}")
        count = collection.count()
        print(f"   - Document count: {count}")

        if count > 0:
            # Get sample document
            sample = collection.peek(limit=1)
            if sample['ids']:
                print(f"   - Sample ID: {sample['ids'][0]}")
                if sample['embeddings'] is not None and len(sample['embeddings']) > 0:
                    print(f"   - Has embedding: {len(sample['embeddings'][0])} dims")
                if sample['metadatas']:
                    print(f"   - Metadata keys: {list(sample['metadatas'][0].keys())}")

    print("\n" + "=" * 70)
    print("SEARCH QUALITY TEST")
    print("=" * 70)

    # Test search on L0 collection
    try:
        l0_collection = client.get_collection("golden_references_advanced_L0")

        # Load embedding model
        print("\n2. Loading SciBERT embedding model...")
        model = SentenceTransformer('allenai/scibert_scivocab_uncased')
        print("   ✓ Model loaded")

        # Test query
        query = "foundation models for medical imaging"
        print(f"\n3. Test Query: '{query}'")

        query_embedding = model.encode(query)

        results = l0_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=3
        )

        print(f"\n   Top 3 Results:")
        for i, (doc_id, distance, metadata) in enumerate(zip(
            results['ids'][0],
            results['distances'][0],
            results['metadatas'][0]
        ), 1):
            print(f"\n   [{i}] ID: {doc_id}")
            print(f"       Distance: {distance:.4f}")
            print(f"       Paper: {metadata.get('paper_id', 'N/A')}")
            print(f"       Section: {metadata.get('section', 'N/A')}")

    except Exception as e:
        print(f"\n   ⚠️  Search test failed: {e}")

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    test_chromadb_storage()
