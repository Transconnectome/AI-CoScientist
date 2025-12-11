#!/usr/bin/env python3
"""
Load Grant Proposals into Fresh ChromaDB

This script creates a fresh ChromaDB instance specifically for grant proposals
to avoid conflicts with the existing DD-RAPTOR database.

Usage:
    poetry run python scripts/load_grants_fresh_chromadb.py
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
import numpy as np
import chromadb
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def load_grant_json_files(json_dir: Path) -> List[Dict]:
    """Load all grant proposal JSON files from directory."""
    json_files = sorted(json_dir.glob("*.json"))
    proposals = []

    print(f"Found {len(json_files)} grant proposal JSON files")

    for json_file in tqdm(json_files, desc="Loading grant proposal JSON files"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                proposal_data = json.load(f)
                proposals.append(proposal_data)
        except Exception as e:
            print(f"Error loading {json_file.name}: {e}")

    return proposals

def main():
    """Main execution."""

    print("=" * 70)
    print("GRANT PROPOSALS - FRESH CHROMADB INGESTION")
    print("=" * 70)

    # Use a fresh ChromaDB path to avoid conflicts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fresh_db_path = f"chromadb_grants_{timestamp}"

    # Paths
    json_dir = Path("data/processed_grants")

    if not json_dir.exists():
        print(f"Error: Grant JSON directory not found: {json_dir}")
        print("Please run 'poetry run python scripts/simple_grant_ingestor.py' first.")
        return

    # Load grant proposal JSON files
    proposals = load_grant_json_files(json_dir)

    if not proposals:
        print("No grant proposals loaded. Exiting.")
        return

    print(f"\nLoaded {len(proposals)} grant proposals:")
    total_chunks = 0
    for proposal in proposals:
        chunks_count = len(proposal.get('level0_chunks', []))
        total_chunks += chunks_count
        print(f"  - {proposal['title'][:60]}{'...' if len(proposal['title']) > 60 else ''} ({chunks_count} chunks)")

    print(f"\nTotal chunks to process: {total_chunks}")

    # Initialize embedding model
    print("\nLoading SciBERT embedding model...")
    embedding_model = SentenceTransformer('allenai/scibert_scivocab_uncased')
    print("✓ SciBERT loaded")

    # Create fresh ChromaDB instance
    print(f"\nCreating fresh ChromaDB at {fresh_db_path}...")
    client = chromadb.PersistentClient(path=fresh_db_path)

    # Create collection
    collection = client.get_or_create_collection(
        name="grant_proposals",
        metadata={"description": "Grant proposal chunks with embeddings"}
    )

    print("✓ Fresh ChromaDB created")

    # Process all chunks
    all_ids = []
    all_embeddings = []
    all_documents = []
    all_metadatas = []

    print("\nProcessing grant proposals...")
    for proposal in tqdm(proposals, desc="Processing proposals"):
        try:
            chunks = proposal.get('level0_chunks', [])
            if not chunks:
                continue

            # Extract text content for embedding generation
            chunk_texts = [chunk['content'] for chunk in chunks]

            # Generate embeddings
            print(f"  Generating embeddings for {len(chunks)} chunks...")
            embeddings = embedding_model.encode(chunk_texts, show_progress_bar=False)

            # Prepare data
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                all_ids.append(chunk['chunk_id'])
                all_embeddings.append(embedding.tolist())
                all_documents.append(chunk['content'])

                # Enhance metadata
                meta = chunk['metadata'].copy()
                meta.update({
                    'paper_title': proposal.get('title', 'Unknown Title'),
                    'ingested_at': datetime.now().isoformat()
                })
                all_metadatas.append(meta)

        except Exception as e:
            print(f"\n⚠️  Error processing {proposal.get('title', 'Unknown')}: {e}")
            continue

    # Ingest all chunks into ChromaDB
    print(f"\nIngesting {len(all_ids)} chunks into ChromaDB...")

    # Ingest in batches to avoid memory issues
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
            print(f"\n⚠️  Error ingesting batch {i//batch_size + 1}: {e}")
            continue

    # Test the collection
    print("\nTesting collection...")
    try:
        collection_info = collection.count()
        print(f"✓ Collection contains {collection_info} documents")

        # Test query
        test_results = collection.query(
            query_texts=["quantum machine learning"],
            n_results=3
        )
        print(f"✓ Test query returned {len(test_results['documents'][0])} results")

    except Exception as e:
        print(f"⚠️  Error testing collection: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("GRANT PROPOSAL INGESTION COMPLETE")
    print("=" * 70)
    print(f"✅ Grant proposals loaded: {len(proposals)}")
    print(f"📄 Total chunks ingested: {len(all_ids)}")
    print(f"💾 ChromaDB location: {fresh_db_path}")
    print("\n🎉 Grant proposals are now available for querying!")

    # Show sample queries
    print("\nSample queries you can try:")
    print("- 'quantum machine learning QuantERA'")
    print("- 'brain connectome neuroscience'")
    print("- 'INCITE supercomputer resources'")
    print("- 'research objectives methodology'")

    print("=" * 70)

if __name__ == "__main__":
    main()