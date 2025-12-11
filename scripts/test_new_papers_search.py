#!/usr/bin/env python3
"""
Test New Papers Search Including ESM3

Test search functionality for the newly ingested papers including ESM3.

Usage:
    poetry run python scripts/test_new_papers_search.py
"""

import chromadb
import sys
from pathlib import Path

def find_latest_papers_db():
    """Find the latest new papers ChromaDB."""
    current_dir = Path(".")
    paper_dbs = list(current_dir.glob("chromadb_new_papers_*"))

    if not paper_dbs:
        # Check for database path file
        path_file = Path("latest_papers_db_path.txt")
        if path_file.exists():
            with open(path_file, 'r') as f:
                db_path = f.read().strip()
                if Path(db_path).exists():
                    return db_path

        return None

    # Sort by name (which includes timestamp)
    latest_db = sorted(paper_dbs)[-1]
    return str(latest_db)

def test_new_papers_search():
    """Test searching newly ingested papers."""

    print("=" * 70)
    print("🧬 새 논문 검색 테스트 (ESM3 포함)")
    print("=" * 70)

    # Find latest papers database
    db_path = find_latest_papers_db()
    if not db_path:
        print("❌ 새로 수집된 논문 데이터베이스를 찾을 수 없습니다!")
        print("먼저 다음을 실행하세요:")
        print("poetry run python scripts/ingest_new_papers_complete.py")
        return

    print(f"🔍 데이터베이스: {db_path}")

    # Connect to ChromaDB
    try:
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(name="new_papers")

        # Get collection info
        count = collection.count()
        print(f"📄 총 문서 수: {count}")

        # Get sample metadata to see what papers we have
        sample_results = collection.get(limit=5)
        if sample_results['metadatas']:
            print(f"\n📚 수집된 논문들:")
            unique_papers = set()
            for meta in sample_results['metadatas']:
                if meta:
                    paper_title = meta.get('paper_title', 'Unknown')[:60]
                    paper_type = meta.get('paper_type', 'Unknown')
                    authors = meta.get('authors', 'Unknown')[:30]
                    year = meta.get('year', 'Unknown')

                    paper_info = f"  📄 {paper_title}... ({paper_type}, {year})"
                    if paper_info not in unique_papers:
                        unique_papers.add(paper_info)
                        print(paper_info)
                        print(f"      저자: {authors}...")

        # Test queries specifically for ESM3 and protein research
        test_queries = [
            # ESM3 specific
            "ESM3 evolutionary scale modeling",
            "protein language model Meta AI",
            "protein structure prediction evolution",
            "amino acid sequence generation",
            "evolutionary scale modeling 500 million years",

            # General protein research
            "protein folding artificial intelligence",
            "multimodal biomedical AI",
            "machine learning protein sequences",

            # AI/ML general
            "deep learning neural networks",
            "transformer architecture",
            "artificial intelligence biomedical"
        ]

        print(f"\n🔍 검색 테스트:")
        print("-" * 70)

        for query in test_queries:
            print(f"\n📝 검색어: '{query}'")

            try:
                results = collection.query(
                    query_texts=[query],
                    n_results=3
                )

                documents = results['documents'][0]
                metadatas = results['metadatas'][0] if results['metadatas'] else []

                if documents:
                    print(f"   ✅ {len(documents)}개 결과:")

                    for i, (doc, meta) in enumerate(zip(documents, metadatas)):
                        paper_type = meta.get('paper_type', 'Unknown')
                        paper_title = meta.get('paper_title', 'Unknown')[:40]
                        section = meta.get('section_name', 'Unknown')
                        chunk_idx = meta.get('chunk_index', '?')

                        # Show preview
                        preview = doc[:100].replace('\n', ' ') + "..."

                        print(f"   [{i+1}] {paper_type} | {paper_title}... | {section} #{chunk_idx}")
                        print(f"       {preview}")
                else:
                    print("   ❌ 검색 결과 없음")

            except Exception as e:
                print(f"   ⚠️ 검색 오류: {e}")

        # Special ESM3 deep search
        print(f"\n" + "=" * 50)
        print(f"🧬 ESM3 전용 깊이 검색")
        print("=" * 50)

        esm3_specific_queries = [
            "ESM3",
            "evolutionary scale modeling",
            "protein language model",
            "Meta AI protein"
        ]

        esm3_found = False

        for query in esm3_specific_queries:
            try:
                results = collection.query(
                    query_texts=[query],
                    n_results=5
                )

                if results['documents'][0]:
                    esm3_found = True
                    print(f"\n🎯 '{query}' - {len(results['documents'][0])} 결과:")

                    for i, (doc, meta) in enumerate(zip(results['documents'][0][:2], results['metadatas'][0][:2])):
                        paper_title = meta.get('paper_title', 'Unknown')
                        paper_type = meta.get('paper_type', 'Unknown')

                        # Look for ESM3 specific content
                        esm3_indicators = ['esm3', 'evolutionary scale', 'meta ai', 'protein language']
                        doc_lower = doc.lower()

                        found_indicators = [ind for ind in esm3_indicators if ind in doc_lower]

                        if found_indicators:
                            print(f"      [{i+1}] {paper_title} ({paper_type})")
                            print(f"          ESM3 지표: {', '.join(found_indicators)}")
                            print(f"          내용: {doc[:120]}...")

            except Exception as e:
                continue

        print(f"\n" + "=" * 70)
        if esm3_found:
            print(f"🎉 ESM3 관련 논문이 성공적으로 수집되었습니다!")
            print(f"✅ 이제 ESM3 단백질 연구를 RAG로 검색할 수 있습니다.")
        else:
            print(f"⚠️ ESM3 관련 내용을 찾지 못했습니다.")
            print(f"다시 확인해보세요: 논문 내용에 ESM3가 포함되어 있는지 확인")

        print(f"\n💡 고급 RAG 시스템:")
        print(f"이제 Unified RAG Orchestrator로 더 향상된 검색이 가능합니다!")
        print(f"poetry run python scripts/test_unified_rag_search.py")

        print("=" * 70)

    except Exception as e:
        print(f"❌ 데이터베이스 연결 실패: {e}")

if __name__ == "__main__":
    test_new_papers_search()