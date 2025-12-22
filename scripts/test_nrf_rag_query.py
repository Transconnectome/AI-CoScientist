#!/usr/bin/env python3
"""
Test RAG retrieval for NRF Mid-Career Proposal samples.

Usage:
    poetry run python scripts/test_nrf_rag_query.py
"""

import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

# Configuration
CHROMADB_PATH = "chromadb_data"
COLLECTION_L0 = "nrf_midcareer_samples_L0"
COLLECTION_L1 = "nrf_midcareer_samples_L1"
COLLECTION_L2 = "nrf_midcareer_samples_L2"


def main():
    print("\n" + "=" * 70)
    print("NRF MID-CAREER PROPOSAL RAG RETRIEVAL TEST")
    print("=" * 70)

    # Initialize
    print("\n1. Initializing ChromaDB and embedding model...")
    client = chromadb.PersistentClient(path=CHROMADB_PATH)
    model = SentenceTransformer('allenai/scibert_scivocab_uncased')

    # Get collections
    collection_l0 = client.get_collection(COLLECTION_L0)
    collection_l1 = client.get_collection(COLLECTION_L1)
    collection_l2 = client.get_collection(COLLECTION_L2)

    print(f"  L0 (Chunks): {collection_l0.count()}")
    print(f"  L1 (Sections): {collection_l1.count()}")
    print(f"  L2 (Documents): {collection_l2.count()}")

    # Test queries
    test_queries = [
        # Korean queries (NRF proposal related)
        "연구과제의 창의성과 도전성을 어떻게 작성해야 하나요?",
        "뇌영상 데이터 분석 방법론",
        "발달장애 진단을 위한 AI 모델",

        # English queries (INCITE related)
        "NeuroX-Fusion foundation model architecture",
        "4D Swin Transformer for brain imaging",
        "federated learning for brain data",
    ]

    print("\n" + "=" * 70)
    print("2. Running Test Queries")
    print("=" * 70)

    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Query {i}: {query} ---")

        # Generate query embedding
        query_embedding = model.encode([query])[0].tolist()

        # Query L0 (chunks)
        results = collection_l0.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=['documents', 'metadatas', 'distances']
        )

        print("\nTop 3 Results (L0 Chunks):")
        for j, (doc, meta, dist) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            print(f"\n  [{j+1}] Score: {1-dist:.4f}")
            print(f"      Section: {meta.get('section', 'N/A')}")
            print(f"      Proposal: {meta.get('proposal_id', 'N/A')}")
            print(f"      Content: {doc[:200]}...")

    # Query L1 (section summaries)
    print("\n" + "=" * 70)
    print("3. Testing L1 (Section-level) Retrieval")
    print("=" * 70)

    query = "연구 방법론과 추진 전략"
    query_embedding = model.encode([query])[0].tolist()

    results = collection_l1.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=['documents', 'metadatas', 'distances']
    )

    print(f"\nQuery: {query}")
    print("\nTop 5 Section Summaries:")
    for j, (doc, meta, dist) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        print(f"\n  [{j+1}] Score: {1-dist:.4f}")
        print(f"      Section: {meta.get('section', 'N/A')}")
        print(f"      Type: {meta.get('proposal_type', 'N/A')}")
        print(f"      Summary: {doc[:150]}...")

    # Query L2 (document summaries)
    print("\n" + "=" * 70)
    print("4. Testing L2 (Document-level) Retrieval")
    print("=" * 70)

    query = "뇌 파운데이션 모델 연구"
    query_embedding = model.encode([query])[0].tolist()

    results = collection_l2.query(
        query_embeddings=[query_embedding],
        n_results=4,
        include=['documents', 'metadatas', 'distances']
    )

    print(f"\nQuery: {query}")
    print("\nAll Document Summaries:")
    for j, (doc, meta, dist) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        print(f"\n  [{j+1}] Score: {1-dist:.4f}")
        print(f"      Title: {meta.get('title', 'N/A')[:60]}...")
        print(f"      Type: {meta.get('proposal_type', 'N/A')}")
        print(f"      Chunks: {meta.get('total_chunks', 'N/A')}")

    print("\n" + "=" * 70)
    print("RAG RETRIEVAL TEST COMPLETE")
    print("=" * 70)
    print("\nNRF Mid-Career proposal samples are now available for RAG queries!")


if __name__ == "__main__":
    main()
