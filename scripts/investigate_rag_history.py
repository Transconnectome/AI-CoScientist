#!/usr/bin/env python3
"""Investigate RAG system history and current state."""

import chromadb
from chromadb.config import Settings
from collections import defaultdict
from datetime import datetime

def main():
    # Connect to ChromaDB
    chromadb_path = "./chromadb_data"
    client = chromadb.PersistentClient(
        path=chromadb_path,
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )

    print("=" * 70)
    print("🔍 RAG System Investigation")
    print("=" * 70)

    # List all collections
    collections = client.list_collections()
    print(f"\n📁 Collections found: {len(collections)}")

    for coll in collections:
        print(f"\n{'─'*70}")
        print(f"Collection: {coll.name}")
        print(f"{'─'*70}")

        count = coll.count()
        print(f"Total documents: {count}")

        if count > 0:
            # Get all data
            all_data = coll.get()

            # Analyze metadata
            if all_data['metadatas']:
                # Group by source file
                sources = defaultdict(int)
                doc_ids = defaultdict(int)
                timestamps = []
                doc_types = defaultdict(int)

                for meta in all_data['metadatas']:
                    if 'source_file' in meta:
                        sources[meta['source_file']] += 1
                    if 'document_id' in meta:
                        doc_ids[meta['document_id']] += 1
                    if 'timestamp' in meta:
                        timestamps.append(meta['timestamp'])
                    if 'document_type' in meta:
                        doc_types[meta['document_type']] += 1
                    elif 'section' in meta:
                        doc_types['improvement_pattern'] += 1

                print(f"\n📊 Document Statistics:")
                print(f"  Unique document IDs: {len(doc_ids)}")
                print(f"  Unique source files: {len(sources)}")

                print(f"\n📝 Document Types:")
                for dtype, cnt in sorted(doc_types.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {dtype}: {cnt} chunks")

                if timestamps:
                    timestamps_sorted = sorted(timestamps)
                    print(f"\n⏰ Timeline:")
                    print(f"  Oldest: {timestamps_sorted[0]}")
                    print(f"  Newest: {timestamps_sorted[-1]}")

                print(f"\n📄 Top 20 Source Files:")
                for source, cnt in sorted(sources.items(), key=lambda x: x[1], reverse=True)[:20]:
                    print(f"  {source}: {cnt} chunks")

                # Check for papers_collection references
                papers_collection_count = sum(
                    1 for s in sources.keys()
                    if 'papers_collection' in s or any(
                        keyword in s.lower() for keyword in [
                            'arxiv', 'pubmed', 'paper', 'research', 'study'
                        ]
                    )
                )

                print(f"\n🔍 Papers Collection Analysis:")
                print(f"  Documents likely from papers_collection: {papers_collection_count}")

                # Sample metadata
                print(f"\n📋 Sample Metadata (First Document):")
                sample = all_data['metadatas'][0]
                for key, value in sorted(sample.items()):
                    print(f"  {key}: {value}")

    print(f"\n{'='*70}")
    print(f"🎯 ROOT CAUSE ANALYSIS")
    print(f"{'='*70}")

    # Check for research_documents collection specifically
    research_coll_exists = any(c.name == 'research_documents' for c in collections)

    if not research_coll_exists:
        print("\n❌ FINDING: 'research_documents' collection does NOT exist!")
        print("   This suggests documents from papers_collection were never ingested.")
        print("   Only 'improvement_patterns' collection exists (for paper improvements).")
    else:
        research_coll = client.get_collection('research_documents')
        research_count = research_coll.count()
        print(f"\n✅ 'research_documents' collection exists with {research_count} documents")

        if research_count == 0:
            print("   ❌ But it's EMPTY! Documents were never added.")
        else:
            print("   ✅ Contains documents, checking if they're from papers_collection...")

    print(f"\n{'='*70}")

if __name__ == "__main__":
    main()
