#!/usr/bin/env python3
"""
CLI script to query the Developmental Disability RAG system.
Useful for integration with Cursor or terminal usage.

Usage:
    poetry run python scripts/query_dd_rag.py "your query here"
"""

import argparse
import chromadb
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer, CrossEncoder

def main():
    parser = argparse.ArgumentParser(description="Query the Developmental Disability RAG system")
    parser.add_argument("query", type=str, help="The search query")
    parser.add_argument("-n", "--n-results", type=int, default=5, help="Number of results to return")
    parser.add_argument("--db-path", type=str, default="chromadb_data_dd", help="Path to ChromaDB")
    parser.add_argument("--no-rerank", action="store_true", help="Disable re-ranking (faster but less accurate)")
    
    args = parser.parse_args()
    
    # Check if DB exists
    if not Path(args.db_path).exists():
        print(f"Error: Database not found at {args.db_path}")
        print("Please run 'poetry run python scripts/load_json_to_chromadb_dd.py' first.")
        sys.exit(1)
        
    try:
        # Load embedding model
        print("Loading SciBERT model...", file=sys.stderr)
        embedding_model = SentenceTransformer('allenai/scibert_scivocab_uncased')
        query_embedding = embedding_model.encode([args.query])[0].tolist()
        
        client = chromadb.PersistentClient(path=args.db_path)
        collection = client.get_collection(name="dd_papers_L0")
        
        # 1. Retrieval (Fetch more candidates for re-ranking)
        initial_k = 50 if not args.no_rerank else args.n_results
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=initial_k
        )
        
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        final_results = []
        
        # 2. Re-ranking
        if not args.no_rerank and documents:
            print("Loading Cross-Encoder for re-ranking...", file=sys.stderr)
            cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            
            # Pair query with each document
            pairs = [[args.query, doc] for doc in documents]
            scores = cross_encoder.predict(pairs)
            
            # Sort by score (descending)
            scored_results = []
            for i, score in enumerate(scores):
                scored_results.append({
                    'document': documents[i],
                    'metadata': metadatas[i],
                    'score': score
                })
            
            scored_results.sort(key=lambda x: x['score'], reverse=True)
            
            # Take top N
            final_results = scored_results[:args.n_results]
        else:
            # No re-ranking, just take top N
            for i in range(min(len(documents), args.n_results)):
                final_results.append({
                    'document': documents[i],
                    'metadata': metadatas[i],
                    'score': 0.0 # No score
                })
        
        # Output
        print(f"\n🔍 Search Results for: '{args.query}'")
        print("=" * 60)
        
        for i, res in enumerate(final_results):
            title = res['metadata'].get('paper_title', 'Unknown Title')
            section = res['metadata'].get('section', 'Unknown Section')
            content = res['document']
            score_str = f" (Score: {res['score']:.4f})" if not args.no_rerank else ""
            
            print(f"\n📄 Result {i+1}{score_str}")
            print(f"Title:   {title}")
            print(f"Section: {section}")
            print("-" * 60)
            print(content.strip())
            print("=" * 60)
            
    except Exception as e:
        print(f"Error querying database: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
