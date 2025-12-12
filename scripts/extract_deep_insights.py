#!/usr/bin/env python3
"""
Extract deep scientific insights from specific high-value queries.
Focus on extracting concrete findings, effect sizes, and methodological details.
"""

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import json


def query_and_extract(collection, embedding_model, cross_encoder, query, n_results=10):
    """Query and extract detailed information"""

    # Generate embedding
    query_embedding = embedding_model.encode([query])[0].tolist()

    # Retrieve candidates
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=50
    )

    documents = results['documents'][0]
    metadatas = results['metadatas'][0]

    if not documents:
        return []

    # Re-rank
    pairs = [[query, doc] for doc in documents]
    scores = cross_encoder.predict(pairs)

    # Sort by score
    scored_results = []
    for i, score in enumerate(scores):
        scored_results.append({
            'document': documents[i],
            'metadata': metadatas[i],
            'score': float(score)
        })

    scored_results.sort(key=lambda x: x['score'], reverse=True)

    return scored_results[:n_results]


def main():
    # Initialize
    print("Loading models...")
    embedding_model = SentenceTransformer('allenai/scibert_scivocab_uncased')
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    client = chromadb.PersistentClient(path="chromadb_data_dd")
    collection = client.get_collection(name="dd_papers_L0")

    # Deep dive queries
    deep_queries = {
        "Specific Biomarkers": [
            "early childhood eye tracking biomarkers autism prediction accuracy AUC",
            "brain connectivity patterns autism default mode network functional connectivity",
            "genetic variants autism GWAS polygenic risk scores",
            "EEG biomarkers autism event-related potentials ERP",
        ],
        "Advanced ML Methods": [
            "convolutional neural networks CNN fMRI autism classification accuracy",
            "transformer attention mechanism brain imaging diagnosis",
            "transfer learning pretrained models neuroimaging small datasets",
            "ensemble methods random forest SVM autism diagnosis performance",
        ],
        "Longitudinal Studies": [
            "infant sibling studies autism risk prediction sensitivity specificity",
            "developmental trajectories brain volume cortical thickness autism",
            "early intervention outcomes autism longitudinal follow-up",
        ],
        "Technical Challenges": [
            "heterogeneity autism spectrum variability classification challenges",
            "small sample size overfitting cross-validation replication",
            "multi-site neuroimaging harmonization batch effects",
            "interpretability explainability black box models clinical translation",
        ]
    }

    all_insights = {}

    for category, queries in deep_queries.items():
        print(f"\n{'='*80}")
        print(f"CATEGORY: {category}")
        print(f"{'='*80}\n")

        category_insights = []

        for query in queries:
            print(f"Query: {query}")
            results = query_and_extract(collection, embedding_model, cross_encoder, query, n_results=3)

            for i, res in enumerate(results):
                if res['score'] > -2.0:  # Only include reasonably relevant results
                    print(f"  [{i+1}] Score: {res['score']:.3f} | {res['metadata'].get('paper_title', 'Unknown')[:60]}")
                    print(f"      {res['document'][:200]}...")

                    category_insights.append({
                        'query': query,
                        'score': res['score'],
                        'paper': res['metadata'].get('paper_title', 'Unknown'),
                        'section': res['metadata'].get('section', 'Unknown'),
                        'content': res['document']
                    })

            print()

        all_insights[category] = category_insights

    # Save insights
    with open('dd_deep_insights.json', 'w') as f:
        json.dump(all_insights, f, indent=2)

    print(f"\n✓ Deep insights saved to dd_deep_insights.json")


if __name__ == "__main__":
    main()
