#!/usr/bin/env python3
"""
Test RAG System with SCENT Paper - GINR Query Test

This script tests if the RAG system can understand GINR (Generalized Implicit Neural Representation)
from the ingested SCENT paper.
"""

import json
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def load_scent_chunks(json_path: Path) -> List[Dict]:
    """Load scent paper chunks from JSON."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['chunks']

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def search_chunks(query: str, chunks: List[Dict], embedding_model, top_k: int = 3) -> List[Tuple[Dict, float]]:
    """Search chunks using semantic similarity."""
    # Generate query embedding
    query_embedding = embedding_model.encode(query, convert_to_numpy=True).tolist()
    
    # Calculate similarities
    results = []
    for chunk in chunks:
        chunk_embedding = chunk['embedding']
        similarity = cosine_similarity(query_embedding, chunk_embedding)
        results.append((chunk, similarity))
    
    # Sort by similarity (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results[:top_k]

def test_ginr_understanding():
    """Test if RAG system understands INR/GINR."""
    
    print("=" * 80)
    print("SCENT PAPER RAG SYSTEM TEST - INR/GINR Understanding")
    print("=" * 80)
    
    # Load chunks
    json_path = Path(__file__).parent / "scent_paper_chunks_enhanced.json"
    if not json_path.exists():
        print(f"❌ Error: {json_path} not found")
        return
    
    print(f"\n1. Loading chunks from {json_path.name}...")
    chunks = load_scent_chunks(json_path)
    print(f"   ✓ Loaded {len(chunks)} chunks")
    
    # Load embedding model
    model_name = chunks[0]['metadata'].get('embedding_model', 'all-MiniLM-L6-v2')
    print(f"\n2. Loading embedding model ({model_name})...")
    embedding_model = SentenceTransformer(model_name)
    print("   ✓ Model loaded")
    
    # Test queries related to INR (Implicit Neural Representations)
    test_queries = [
        "What are Implicit Neural Representations (INRs)?",
        "How do INRs model continuous signals?",
        "What are the limitations of traditional INRs regarding generalization?",
        "How does SCENT utilize INRs for spatiotemporal learning?",
        "What is the advantage of INRs over grid-based representations?",
        "How do Neural Radiance Fields (NeRF) relate to INRs?",
        "Why is retraining required for traditional INRs for new data?",
        "How does the paper propose to make INRs generalizable?"
    ]
    
    print("\n" + "=" * 80)
    print("TESTING INR UNDERSTANDING")
    print("=" * 80)
    
    all_results = {}
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'─' * 80}")
        print(f"Query {i}: {query}")
        print('─' * 80)
        
        # Search
        results = search_chunks(query, chunks, embedding_model, top_k=3)
        
        if not results:
            print("   ❌ No results found")
            continue
        
        # Display top results
        for rank, (chunk, similarity) in enumerate(results, 1):
            print(f"\n   [{rank}] Similarity: {similarity:.4f}")
            print(f"   Section: {chunk['metadata'].get('section_name', 'Unknown')}")
            
            # Extract relevant snippet
            content = chunk['content']
            query_words = query.lower().split()
            
            # Find sentences containing query keywords
            sentences = content.split('.')
            relevant_sentences = []
            for sentence in sentences:
                sentence_lower = sentence.lower()
                # Simple keyword matching for snippet extraction
                keywords = ['inr', 'implicit', 'neural', 'representation', 'generaliz', 'scent', 'grid', 'nerf']
                if any(k in sentence_lower for k in keywords):
                     relevant_sentences.append(sentence.strip())
            
            if relevant_sentences:
                # Get up to 2 most relevant sentences
                snippet = '. '.join(relevant_sentences[:3]) + '.'
                print(f"   Snippet: {snippet[:300]}...")
            else:
                print(f"   Content preview: {content[:200]}...")
        
        all_results[query] = results
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    avg_similarities = []
    for query, results in all_results.items():
        if results:
            avg_sim = sum(sim for _, sim in results) / len(results)
            avg_similarities.append(avg_sim)
            print(f"✓ {query[:60]:<60} Avg Sim: {avg_sim:.4f}")
    
    if avg_similarities:
        overall_avg = sum(avg_similarities) / len(avg_similarities)
        print(f"\nOverall Average Similarity: {overall_avg:.4f}")
        
        if overall_avg > 0.5:
            print("✅ RAG system shows GOOD understanding of INR")
        elif overall_avg > 0.3:
            print("⚠️  RAG system shows MODERATE understanding of INR")
        else:
            print("❌ RAG system shows POOR understanding of INR")
    
    # Check if GINR content exists in chunks
    print("\n" + "=" * 80)
    print("CONTENT VERIFICATION")
    print("=" * 80)
    
    ginr_keywords = ['GINR', 'Generalized Implicit Neural Representation', 'generalizable', 'INR']
    ginr_mentions = []
    
    for chunk in chunks:
        content_lower = chunk['content'].lower()
        for keyword in ginr_keywords:
            if keyword.lower() in content_lower:
                ginr_mentions.append({
                    'chunk_id': chunk['chunk_id'],
                    'keyword': keyword,
                    'section': chunk['metadata'].get('section_name', 'Unknown')
                })
                break
    
    print(f"Found {len(ginr_mentions)} chunks mentioning GINR-related terms:")
    for mention in ginr_mentions[:5]:
        print(f"  - {mention['section']}: {mention['keyword']}")
    
    if ginr_mentions:
        print("✅ GINR content is present in ingested data")
    else:
        print("⚠️  GINR content may not be properly ingested")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    test_ginr_understanding()

