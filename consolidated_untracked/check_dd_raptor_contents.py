#!/usr/bin/env python3
"""
Check DD-RAPTOR Contents for ESM3 Papers

This script searches the DD-RAPTOR ChromaDB system to see what papers are available,
specifically checking for ESM3-related content.

Usage:
    poetry run python scripts/check_dd_raptor_contents.py
"""

import chromadb
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def check_dd_raptor_esm3():
    """Check if DD-RAPTOR contains ESM3 papers."""

    print("=" * 70)
    print("DD-RAPTOR 내용 확인 - ESM3 논문 검색")
    print("=" * 70)

    # Check different possible ChromaDB locations
    db_paths = [
        "chromadb_data_dd",           # Original DD database
        "chromadb_data",              # Main ChromaDB
    ]

    # Find grant databases too
    current_dir = Path(".")
    grant_dbs = list(current_dir.glob("chromadb_grants_*"))
    db_paths.extend([str(db) for db in grant_dbs])

    results_found = False

    for db_path in db_paths:
        if not Path(db_path).exists():
            continue

        print(f"\n🔍 검색 중: {db_path}")

        try:
            client = chromadb.PersistentClient(path=db_path)
            collections = client.list_collections()

            print(f"  컬렉션 수: {len(collections)}")

            for collection_info in collections:
                collection_name = collection_info.name
                collection = client.get_collection(name=collection_name)
                count = collection.count()

                print(f"  📂 {collection_name}: {count} 문서")

                # Search for ESM3 related content
                esm3_queries = [
                    "ESM3",
                    "Evolutionary Scale Modeling",
                    "protein language model",
                    "protein structure prediction",
                    "protein folding",
                    "Meta AI protein"
                ]

                for query in esm3_queries:
                    try:
                        results = collection.query(
                            query_texts=[query],
                            n_results=3
                        )

                        if results['documents'][0]:  # If any results found
                            print(f"    ✅ '{query}' 검색 결과: {len(results['documents'][0])} 개")
                            results_found = True

                            # Show first result preview
                            for i, (doc, meta) in enumerate(zip(results['documents'][0][:1],
                                                               results['metadatas'][0][:1] if results['metadatas'] else [{}])):
                                title = meta.get('paper_title', meta.get('title', 'Unknown'))
                                preview = doc[:100].replace('\n', ' ') + "..." if len(doc) > 100 else doc
                                print(f"      [{i+1}] {title}")
                                print(f"          {preview}")

                    except Exception as e:
                        # Skip query errors (dimension mismatch, etc.)
                        continue

        except Exception as e:
            print(f"  ❌ 연결 실패: {e}")
            continue

    # Check for specific paper files that might contain ESM3
    print(f"\n📁 로컬 파일 검색:")

    # Check data directories for ESM3 content
    search_dirs = [
        Path("data/reference_papers"),
        Path("data/QuantERA"),
        Path("data/grant"),
        Path("data/processed_grants"),
        Path("data/reference_papers/processed_json") if Path("data/reference_papers/processed_json").exists() else None
    ]

    for search_dir in search_dirs:
        if search_dir and search_dir.exists():
            print(f"  🔍 검색 중: {search_dir}")

            # Search for ESM3 in filenames
            esm3_files = list(search_dir.glob("*esm3*")) + list(search_dir.glob("*ESM3*"))
            protein_files = list(search_dir.glob("*protein*")) + list(search_dir.glob("*folding*"))

            if esm3_files:
                print(f"    ✅ ESM3 파일 발견: {len(esm3_files)} 개")
                for file in esm3_files[:3]:  # Show first 3
                    print(f"      - {file.name}")
                results_found = True

            if protein_files:
                print(f"    📄 단백질 관련 파일: {len(protein_files)} 개")
                for file in protein_files[:3]:  # Show first 3
                    print(f"      - {file.name}")

    print("\n" + "=" * 70)
    if results_found:
        print("✅ ESM3 또는 관련 내용이 발견되었습니다!")
    else:
        print("❌ ESM3 관련 내용이 DD-RAPTOR에서 발견되지 않았습니다.")
        print("\n💡 ESM3 논문을 추가하려면:")
        print("1. ESM3 논문 PDF를 data/reference_papers/ 디렉토리에 추가")
        print("2. scripts/ingest_golden_references_advanced.py 실행하여 처리")
        print("3. scripts/load_json_to_chromadb_dd.py 실행하여 DD-RAPTOR에 추가")

    print("=" * 70)

def search_specific_papers():
    """Search for specific paper content across all databases."""

    print("\n📊 전체 데이터베이스 논문 목록 (샘플):")

    db_paths = ["chromadb_data_dd"]

    # Add grant databases
    current_dir = Path(".")
    grant_dbs = list(current_dir.glob("chromadb_grants_*"))
    db_paths.extend([str(db) for db in grant_dbs])

    all_papers = set()

    for db_path in db_paths:
        if not Path(db_path).exists():
            continue

        try:
            client = chromadb.PersistentClient(path=db_path)
            collections = client.list_collections()

            for collection_info in collections:
                collection = client.get_collection(name=collection_info.name)

                # Get a sample of documents to see paper titles
                try:
                    sample_results = collection.get(limit=10)

                    for meta in sample_results['metadatas']:
                        if meta:
                            title = meta.get('paper_title', meta.get('title', 'Unknown'))
                            if title != 'Unknown':
                                all_papers.add(title)
                except:
                    continue

        except:
            continue

    if all_papers:
        print(f"\n발견된 논문들 (총 {len(all_papers)}개):")
        for i, paper in enumerate(sorted(list(all_papers))[:10]):  # Show first 10
            print(f"  {i+1}. {paper[:80]}{'...' if len(paper) > 80 else ''}")

        if len(all_papers) > 10:
            print(f"  ... 및 {len(all_papers) - 10}개 더")

    # Check specifically for protein/biological papers
    biological_keywords = ["protein", "biological", "molecular", "genetic", "evolution", "amino", "folding"]
    bio_papers = [p for p in all_papers if any(kw.lower() in p.lower() for kw in biological_keywords)]

    if bio_papers:
        print(f"\n🧬 생물학/단백질 관련 논문들 ({len(bio_papers)}개):")
        for paper in bio_papers[:5]:
            print(f"  - {paper}")

def main():
    """Main execution."""
    check_dd_raptor_esm3()
    search_specific_papers()

if __name__ == "__main__":
    main()