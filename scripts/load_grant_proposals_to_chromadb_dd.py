#!/usr/bin/env python3
"""
Load Processed Grant Proposal JSON Files into DD-RAPTOR ChromaDB

This script loads the processed grant proposal JSON files into the DD-RAPTOR
ChromaDB system with embeddings generation.

Based on load_json_to_chromadb_dd.py but adapted for grant proposals.

Usage:
    poetry run python scripts/load_grant_proposals_to_chromadb_dd.py
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
import numpy as np
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

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

def generate_embeddings(chunks: List[str], embedding_model) -> List[List[float]]:
    """Generate embeddings for text chunks."""
    if not chunks:
        return []

    print(f"  Generating embeddings for {len(chunks)} chunks...")
    embeddings = embedding_model.encode(chunks, show_progress_bar=False)
    return [emb.tolist() for emb in embeddings]

def get_or_create_chromadb_collections(db_path: str = "chromadb_data_dd"):
    """Get or create ChromaDB collections for grant proposals."""

    print(f"Connecting to ChromaDB at {db_path}...")

    # Check if directory exists
    if not Path(db_path).exists():
        print(f"Creating ChromaDB directory: {db_path}")
        Path(db_path).mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=db_path)

    # Get or create collections for each RAPTOR level
    collection_l0 = client.get_or_create_collection(
        name="dd_papers_L0",
        metadata={"description": "Level 0: Grant proposal chunks + DD papers"}
    )

    collection_l1 = client.get_or_create_collection(
        name="dd_papers_L1",
        metadata={"description": "Level 1: Section summaries + DD papers"}
    )

    collection_l2 = client.get_or_create_collection(
        name="dd_papers_L2",
        metadata={"description": "Level 2: Paper summaries + DD papers"}
    )

    print("✓ Collections ready")

    return client, collection_l0, collection_l1, collection_l2

def load_grant_proposals_to_chromadb(
    proposals: List[Dict],
    collection_l0,
    embedding_model
):
    """Load grant proposals into ChromaDB collections."""

    total_l0 = 0

    for proposal in tqdm(proposals, desc="Loading grant proposals to ChromaDB"):
        proposal_id = proposal['paper_id']

        try:
            # Load L0 chunks (only level implemented for grant proposals)
            if proposal['level0_chunks']:
                chunks = proposal['level0_chunks']

                # Extract text content for embedding generation
                chunk_texts = [chunk['content'] for chunk in chunks]

                # Generate embeddings
                embeddings = generate_embeddings(chunk_texts, embedding_model)

                # Prepare data for ChromaDB
                ids = [chunk['chunk_id'] for chunk in chunks]
                documents = [chunk['content'] for chunk in chunks]

                # Enhance metadata with embeddings
                metadatas = []
                for i, chunk in enumerate(chunks):
                    meta = chunk['metadata'].copy()
                    # Ensure paper_title is present for compatibility
                    if 'paper_title' not in meta:
                        meta['paper_title'] = proposal.get('title', 'Unknown Title')
                    metadatas.append(meta)

                # Add in batches to avoid memory issues
                batch_size = 100
                for i in range(0, len(ids), batch_size):
                    batch_end = min(i + batch_size, len(ids))
                    collection_l0.add(
                        ids=ids[i:batch_end],
                        embeddings=embeddings[i:batch_end],
                        documents=documents[i:batch_end],
                        metadatas=metadatas[i:batch_end]
                    )

                total_l0 += len(chunks)
                print(f"  ✓ Loaded {len(chunks)} chunks from {proposal['title']}")

        except Exception as e:
            print(f"\n⚠️  Error loading {proposal_id}: {e}")
            continue

    return total_l0

def main():
    """Main execution."""

    print("=" * 70)
    print("GRANT PROPOSALS - JSON TO DD-RAPTOR CHROMADB")
    print("=" * 70)

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
    for proposal in proposals:
        chunks_count = len(proposal.get('level0_chunks', []))
        print(f"  - {proposal['title']} ({chunks_count} chunks)")

    # Initialize embedding model
    print("\nLoading SciBERT embedding model...")
    embedding_model = SentenceTransformer('allenai/scibert_scivocab_uncased')
    print("✓ SciBERT loaded")

    # Create ChromaDB collections
    client, collection_l0, collection_l1, collection_l2 = get_or_create_chromadb_collections()

    # Load grant proposals
    print("\nLoading grant proposals to DD-RAPTOR ChromaDB...")
    total_l0 = load_grant_proposals_to_chromadb(
        proposals,
        collection_l0,
        embedding_model
    )

    # Summary
    print("\n" + "=" * 70)
    print("GRANT PROPOSAL LOADING COMPLETE")
    print("=" * 70)
    print(f"✅ Grant proposals loaded: {len(proposals)}")
    print(f"\nChunks loaded:")
    print(f"  Level 0 (chunks): {total_l0}")
    print(f"  Total: {total_l0}")
    print(f"\n✓ ChromaDB saved to: chromadb_data_dd/")
    print("\n🎉 Grant proposals are now available in DD-RAPTOR!")
    print("You can now query them using the Enhanced DD-RAPTOR system.")
    print("=" * 70)

if __name__ == "__main__":
    main()