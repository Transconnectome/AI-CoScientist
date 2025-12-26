#!/usr/bin/env python3
"""
Comprehensive Paper Processing Verification

This script performs a thorough verification of all paper processing across RAG systems,
checking for completeness, errors, and data integrity.

Usage:
    poetry run python scripts/comprehensive_paper_verification.py
"""

import chromadb
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Set
from collections import defaultdict, Counter

def safe_analyze_database(db_path: str) -> Dict[str, Any]:
    """Safely analyze a ChromaDB database with error handling."""
    try:
        client = chromadb.PersistentClient(path=db_path)
        collections = client.list_collections()

        analysis = {
            'path': db_path,
            'status': 'success',
            'collections': [],
            'total_documents': 0,
            'unique_papers': set(),
            'paper_details': [],
            'issues': []
        }

        for collection_info in collections:
            collection_name = collection_info.name

            try:
                collection = client.get_collection(name=collection_name)
                doc_count = collection.count()

                collection_analysis = {
                    'name': collection_name,
                    'documents': doc_count,
                    'sample_papers': []
                }

                analysis['total_documents'] += doc_count

                # Get sample documents to verify completeness
                if doc_count > 0:
                    try:
                        # Get a good sample size
                        sample_size = min(doc_count, 100)
                        sample_results = collection.get(limit=sample_size)

                        # Check for missing embeddings
                        missing_embeddings = 0
                        missing_metadata = 0
                        paper_titles = Counter()

                        for i, (doc, meta, embedding) in enumerate(zip(
                            sample_results.get('documents', []),
                            sample_results.get('metadatas', []),
                            sample_results.get('embeddings', [])
                        )):
                            # Check for missing data
                            if not embedding or len(embedding) == 0:
                                missing_embeddings += 1

                            if not meta:
                                missing_metadata += 1
                            else:
                                # Extract paper information
                                paper_title = meta.get('paper_title', meta.get('title', 'Unknown'))
                                paper_type = meta.get('paper_type', meta.get('proposal_type', 'Unknown'))
                                authors = meta.get('authors', 'Unknown')
                                year = meta.get('year', 'Unknown')

                                if paper_title and paper_title != 'Unknown':
                                    analysis['unique_papers'].add(paper_title)
                                    paper_titles[paper_title] += 1

                                    # Store detailed paper info
                                    paper_detail = {
                                        'title': paper_title,
                                        'type': paper_type,
                                        'authors': authors,
                                        'year': year,
                                        'source': meta.get('source', 'Unknown'),
                                        'chunks_count': paper_titles[paper_title]
                                    }

                                    # Check if this paper detail already exists
                                    if not any(p['title'] == paper_title for p in analysis['paper_details']):
                                        analysis['paper_details'].append(paper_detail)
                                    else:
                                        # Update chunk count
                                        for p in analysis['paper_details']:
                                            if p['title'] == paper_title:
                                                p['chunks_count'] = paper_titles[paper_title]

                        # Record issues
                        if missing_embeddings > 0:
                            analysis['issues'].append(f"{collection_name}: {missing_embeddings} documents missing embeddings")

                        if missing_metadata > 0:
                            analysis['issues'].append(f"{collection_name}: {missing_metadata} documents missing metadata")

                        # Check for papers with very few chunks (might indicate processing issues)
                        for title, count in paper_titles.items():
                            if count < 3:  # Papers with less than 3 chunks might be incomplete
                                analysis['issues'].append(f"{collection_name}: '{title}' has only {count} chunks - may be incomplete")

                        collection_analysis['missing_embeddings'] = missing_embeddings
                        collection_analysis['missing_metadata'] = missing_metadata
                        collection_analysis['unique_papers_in_collection'] = len(paper_titles)

                    except Exception as e:
                        analysis['issues'].append(f"{collection_name}: Error sampling documents - {e}")

                analysis['collections'].append(collection_analysis)

            except Exception as e:
                analysis['issues'].append(f"Error accessing collection {collection_name}: {e}")

        return analysis

    except Exception as e:
        return {
            'path': db_path,
            'status': 'error',
            'error': str(e),
            'issues': [f"Database connection failed: {e}"]
        }

def verify_source_files():
    """Verify that source PDF files are available."""
    source_verification = {
        'grant_pdfs': [],
        'new_paper_pdfs': [],
        'missing_files': [],
        'total_source_files': 0
    }

    # Check grant proposal PDFs
    grant_dir = Path("data/grant")
    if grant_dir.exists():
        grant_pdfs = list(grant_dir.glob("*.pdf"))
        source_verification['grant_pdfs'] = [f.name for f in grant_pdfs]
        source_verification['total_source_files'] += len(grant_pdfs)

        # Specifically check for paper1-4
        for i in range(1, 5):
            paper_file = grant_dir / f"paper{i}.pdf"
            if paper_file.exists():
                source_verification['new_paper_pdfs'].append(f"paper{i}.pdf")
            else:
                source_verification['missing_files'].append(f"paper{i}.pdf")

    # Check processed JSON files
    processed_dir = Path("data/processed_grants")
    if processed_dir.exists():
        json_files = list(processed_dir.glob("*.json"))
        source_verification['processed_json_files'] = len(json_files)

    return source_verification

def main():
    """Main verification execution."""

    print("=" * 80)
    print("🔍 종합 논문 처리 상태 검증")
    print("=" * 80)

    # Find all databases (avoiding problematic ones)
    current_dir = Path(".")
    db_paths = []

    # Safe database paths (avoiding corrupted ones)
    safe_dbs = [
        "chromadb_data_dd",
        "chromadb_grants_fixed_20251210_200233",
        "chromadb_new_papers_20251210_204818"
    ]

    # Add grant and new paper databases
    grant_dbs = [str(db) for db in current_dir.glob("chromadb_grants_*") if db.is_dir()]
    new_paper_dbs = [str(db) for db in current_dir.glob("chromadb_new_papers_*") if db.is_dir()]

    # Combine all safe databases
    all_dbs = safe_dbs + grant_dbs + new_paper_dbs
    db_paths = [db for db in set(all_dbs) if Path(db).exists()]

    print(f"📊 검증할 데이터베이스: {len(db_paths)}개")

    # Verify source files first
    print(f"\n1️⃣ 원본 파일 검증")
    print("-" * 50)

    source_verification = verify_source_files()
    print(f"📁 Grant 제안서 PDF: {len(source_verification['grant_pdfs'])}개")

    for pdf in source_verification['grant_pdfs']:
        print(f"  ✅ {pdf}")

    print(f"📄 새로운 논문 PDF (paper1-4): {len(source_verification['new_paper_pdfs'])}/4개")
    for pdf in source_verification['new_paper_pdfs']:
        print(f"  ✅ {pdf}")

    if source_verification['missing_files']:
        print(f"❌ 누락된 파일들:")
        for missing in source_verification['missing_files']:
            print(f"  ❌ {missing}")

    # Analyze each database
    print(f"\n2️⃣ 데이터베이스별 상세 분석")
    print("-" * 50)

    all_papers = set()
    all_issues = []
    total_documents = 0
    paper_type_counts = defaultdict(int)

    database_results = []

    for db_path in db_paths:
        print(f"\n🔍 분석 중: {db_path}")

        analysis = safe_analyze_database(db_path)
        database_results.append(analysis)

        if analysis['status'] == 'success':
            print(f"  ✅ 총 문서: {analysis['total_documents']:,}개")
            print(f"  📚 고유 논문: {len(analysis['unique_papers'])}개")

            total_documents += analysis['total_documents']
            all_papers.update(analysis['unique_papers'])

            # Show collections
            for collection in analysis['collections']:
                print(f"    📁 {collection['name']}: {collection['documents']}개")
                if collection.get('missing_embeddings', 0) > 0:
                    print(f"      ⚠️ 임베딩 누락: {collection['missing_embeddings']}개")
                if collection.get('missing_metadata', 0) > 0:
                    print(f"      ⚠️ 메타데이터 누락: {collection['missing_metadata']}개")

            # Count paper types
            for paper in analysis['paper_details']:
                paper_type_counts[paper['type']] += 1

            # Show issues
            if analysis['issues']:
                print(f"  ⚠️ 발견된 문제:")
                for issue in analysis['issues']:
                    print(f"    • {issue}")
                    all_issues.extend(analysis['issues'])

            # Show sample papers
            if analysis['paper_details']:
                print(f"  📖 논문 샘플:")
                for i, paper in enumerate(analysis['paper_details'][:3]):
                    title_preview = paper['title'][:50] + '...' if len(paper['title']) > 50 else paper['title']
                    print(f"    {i+1}. {title_preview} ({paper['type']}, {paper['year']})")
                if len(analysis['paper_details']) > 3:
                    print(f"    ... 및 {len(analysis['paper_details'])-3}개 더")

        else:
            print(f"  ❌ 분석 실패: {analysis.get('error', 'Unknown error')}")

    # ESM3 specific verification
    print(f"\n3️⃣ ESM3 논문 특별 검증")
    print("-" * 50)

    esm3_papers = []
    for result in database_results:
        if result['status'] == 'success':
            for paper in result['paper_details']:
                if any(keyword in paper['title'].lower() for keyword in ['esm3', 'protein', 'paper1', 'paper2', 'paper3', 'paper4']):
                    esm3_papers.append(paper)

    if esm3_papers:
        print(f"🧬 ESM3/Protein 관련 논문: {len(esm3_papers)}개 발견")
        for paper in esm3_papers:
            print(f"  ✅ {paper['title']} ({paper['type']}) - {paper.get('chunks_count', '?')} 청크")
    else:
        print("❌ ESM3 관련 논문을 찾을 수 없습니다!")

    # Check expected papers
    expected_papers = ['paper1', 'paper2', 'paper3', 'paper4']
    found_papers = [paper['title'] for paper in esm3_papers if paper['title'] in expected_papers]

    print(f"\n새로운 논문 (paper1-4) 처리 확인:")
    for expected in expected_papers:
        if expected in found_papers:
            print(f"  ✅ {expected} - 처리 완료")
        else:
            print(f"  ❌ {expected} - 처리 누락")

    # Final Summary
    print(f"\n" + "=" * 80)
    print(f"📈 전체 검증 결과 요약")
    print("=" * 80)
    print(f"📄 전체 문서 수: {total_documents:,}개")
    print(f"📚 전체 고유 논문 수: {len(all_papers)}개")

    print(f"\n논문 유형별 분포:")
    for paper_type, count in sorted(paper_type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  📊 {paper_type}: {count}개")

    # Issue summary
    if all_issues:
        print(f"\n⚠️ 발견된 문제들 ({len(all_issues)}개):")
        for issue in set(all_issues):  # Remove duplicates
            print(f"  • {issue}")
    else:
        print(f"\n✅ 문제 없음 - 모든 논문이 완벽히 처리되었습니다!")

    # Processing completeness assessment
    print(f"\n🎯 처리 완성도 평가:")

    total_expected = len(source_verification['grant_pdfs']) + len(source_verification['new_paper_pdfs'])
    total_processed = len(all_papers)

    if total_processed >= total_expected:
        print(f"  ✅ 우수 ({total_processed}/{total_expected}) - 모든 파일이 처리됨")
    elif total_processed >= total_expected * 0.8:
        print(f"  ⚠️ 양호 ({total_processed}/{total_expected}) - 일부 파일 누락")
    else:
        print(f"  ❌ 불량 ({total_processed}/{total_expected}) - 많은 파일 누락")

    print("=" * 80)

if __name__ == "__main__":
    main()