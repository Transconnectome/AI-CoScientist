#!/usr/bin/env python3
"""
Count Total Papers Across All RAG Systems

This script counts all papers across different ChromaDB instances and RAG systems.

Usage:
    poetry run python scripts/count_total_papers_in_rag.py
"""

import chromadb
import sys
from pathlib import Path
from typing import Dict, Set, Any
from collections import defaultdict

def get_all_chromadb_paths():
    """Find all ChromaDB database paths."""
    current_dir = Path(".")

    db_paths = []

    # Known database paths
    known_dbs = [
        "chromadb_data_dd",           # DD-RAPTOR database
        "chromadb_data",              # Main ChromaDB
    ]

    for db_path in known_dbs:
        if Path(db_path).exists():
            db_paths.append(db_path)

    # Find grant proposal databases
    grant_dbs = list(current_dir.glob("chromadb_grants_*"))
    db_paths.extend([str(db) for db in grant_dbs])

    # Find new papers databases
    new_paper_dbs = list(current_dir.glob("chromadb_new_papers_*"))
    db_paths.extend([str(db) for db in new_paper_dbs])

    # Find any other ChromaDB directories
    other_dbs = list(current_dir.glob("chromadb_*"))
    for db in other_dbs:
        if str(db) not in db_paths and db.is_dir():
            db_paths.append(str(db))

    return db_paths

def analyze_database(db_path: str) -> Dict[str, Any]:
    """Analyze a single ChromaDB database."""
    try:
        client = chromadb.PersistentClient(path=db_path)
        collections = client.list_collections()

        total_documents = 0
        unique_papers = set()
        paper_types = defaultdict(int)

        collection_info = []

        for collection_info_obj in collections:
            collection_name = collection_info_obj.name
            collection = client.get_collection(name=collection_name)

            doc_count = collection.count()
            total_documents += doc_count

            # Sample metadata to identify papers
            try:
                sample_results = collection.get(limit=min(doc_count, 50))

                for meta in sample_results['metadatas']:
                    if meta:
                        paper_title = meta.get('paper_title', meta.get('title', None))
                        paper_type = meta.get('paper_type', meta.get('proposal_type', 'Unknown'))

                        if paper_title and paper_title != 'Unknown Title':
                            unique_papers.add(paper_title)
                            paper_types[paper_type] += 1

            except Exception as e:
                pass  # Skip metadata analysis errors

            collection_info.append({
                'name': collection_name,
                'documents': doc_count
            })

        return {
            'path': db_path,
            'status': 'success',
            'total_documents': total_documents,
            'unique_papers': list(unique_papers),
            'paper_types': dict(paper_types),
            'collections': collection_info
        }

    except Exception as e:
        return {
            'path': db_path,
            'status': 'error',
            'error': str(e)
        }

def main():
    """Main execution."""

    print("=" * 70)
    print("📊 RAG 시스템 전체 논문 수 집계")
    print("=" * 70)

    # Find all databases
    db_paths = get_all_chromadb_paths()

    if not db_paths:
        print("❌ ChromaDB 데이터베이스를 찾을 수 없습니다!")
        return

    print(f"🔍 발견된 데이터베이스: {len(db_paths)}개")

    # Analyze each database
    all_unique_papers = set()
    total_documents_across_all = 0
    paper_type_summary = defaultdict(int)
    successful_dbs = 0

    database_results = []

    for db_path in db_paths:
        print(f"\n📂 분석 중: {db_path}")
        print("-" * 50)

        result = analyze_database(db_path)
        database_results.append(result)

        if result['status'] == 'success':
            successful_dbs += 1
            total_documents_across_all += result['total_documents']

            print(f"  ✅ 총 문서: {result['total_documents']:,}개")
            print(f"  📄 고유 논문: {len(result['unique_papers'])}개")

            # Show collections
            for collection in result['collections']:
                print(f"    📁 {collection['name']}: {collection['documents']} 문서")

            # Add to global counts
            for paper in result['unique_papers']:
                all_unique_papers.add(paper)

            for paper_type, count in result['paper_types'].items():
                paper_type_summary[paper_type] += count

            # Show some sample papers
            if result['unique_papers']:
                print(f"  📚 논문 샘플:")
                for i, paper in enumerate(result['unique_papers'][:3]):
                    print(f"    {i+1}. {paper[:60]}{'...' if len(paper) > 60 else ''}")
                if len(result['unique_papers']) > 3:
                    print(f"    ... 및 {len(result['unique_papers'])-3}개 더")

        else:
            print(f"  ❌ 분석 실패: {result['error']}")

    # Summary
    print(f"\n" + "=" * 70)
    print(f"📈 전체 RAG 시스템 요약")
    print("=" * 70)
    print(f"✅ 성공적으로 분석된 데이터베이스: {successful_dbs}/{len(db_paths)}")
    print(f"📄 전체 문서 수: {total_documents_across_all:,}개")
    print(f"📚 전체 고유 논문 수: {len(all_unique_papers)}개")

    if paper_type_summary:
        print(f"\n📊 논문 유형별 분포:")
        for paper_type, count in sorted(paper_type_summary.items(), key=lambda x: x[1], reverse=True):
            print(f"  📋 {paper_type}: {count}개")

    # Show detailed breakdown by database
    print(f"\n🔍 데이터베이스별 상세 분석:")
    print("-" * 50)

    for result in database_results:
        if result['status'] == 'success':
            db_name = result['path']
            unique_papers_count = len(result['unique_papers'])
            total_docs = result['total_documents']

            # Identify database type
            if 'dd' in db_name.lower():
                db_type = "DD-RAPTOR (발달장애)"
            elif 'grant' in db_name.lower():
                db_type = "Grant Proposals (제안서)"
            elif 'new_papers' in db_name.lower():
                db_type = "New Papers (ESM3 포함)"
            else:
                db_type = "일반"

            print(f"📂 {db_name} ({db_type})")
            print(f"   📄 논문: {unique_papers_count}개 | 문서: {total_docs}개")

    # Special ESM3 detection
    esm3_papers = [paper for paper in all_unique_papers
                   if any(keyword in paper.lower() for keyword in ['esm3', 'protein', 'evolutionary scale'])]

    if esm3_papers:
        print(f"\n🧬 ESM3/단백질 관련 논문: {len(esm3_papers)}개 발견!")
        for paper in esm3_papers[:3]:
            print(f"  • {paper[:80]}{'...' if len(paper) > 80 else ''}")

    print("\n" + "=" * 70)
    print(f"🎉 분석 완료! 총 {len(all_unique_papers)}개의 논문이 RAG 시스템에 수집되어 있습니다.")
    print("=" * 70)

if __name__ == "__main__":
    main()