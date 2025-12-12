#!/usr/bin/env python3
"""
Add ESM3 Papers to DD-RAPTOR System

This script helps download and ingest ESM3 papers into the DD-RAPTOR system.

Usage:
    poetry run python scripts/add_esm3_to_dd_raptor.py
"""

import sys
import requests
from pathlib import Path
from typing import List, Dict
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def download_esm3_papers():
    """Guide for downloading ESM3 papers."""

    print("=" * 70)
    print("ESM3 논문 DD-RAPTOR 추가 가이드")
    print("=" * 70)

    esm3_papers = [
        {
            "title": "ESM3: Simulating 500 million years of evolution with a language model",
            "authors": "Hayes et al. (Meta AI)",
            "arxiv": "2411.12143",
            "url": "https://arxiv.org/abs/2411.12143",
            "description": "ESM3 main paper - 500M year evolution simulation"
        },
        {
            "title": "Evolutionary Scale Modeling",
            "description": "Original ESM paper series",
            "note": "ESM1, ESM2 관련 논문들"
        },
        {
            "title": "Language models of protein sequences at the scale of evolution enable accurate structure prediction",
            "description": "ESM-1v, ESM-1b papers for protein structure prediction",
            "note": "구조 예측 관련 핵심 논문"
        }
    ]

    print("📚 추가 권장 ESM3 관련 논문들:")
    print("-" * 50)

    for i, paper in enumerate(esm3_papers, 1):
        print(f"\n{i}. {paper['title']}")
        if 'authors' in paper:
            print(f"   저자: {paper['authors']}")
        if 'arxiv' in paper:
            print(f"   arXiv: {paper['arxiv']}")
            print(f"   URL: {paper['url']}")
        if 'description' in paper:
            print(f"   설명: {paper['description']}")
        if 'note' in paper:
            print(f"   참고: {paper['note']}")

    print("\n" + "=" * 70)
    print("📥 ESM3 논문 추가 단계")
    print("=" * 70)

    steps = [
        "1. 논문 다운로드:",
        "   • ESM3 논문 PDF를 data/reference_papers/ 디렉토리에 저장",
        "   • 파일명 예시: esm3_main_2024.pdf, esm2_structure_prediction.pdf",
        "",
        "2. DD-RAPTOR 처리:",
        "   poetry run python scripts/ingest_golden_references_advanced.py --test",
        "",
        "3. ChromaDB 추가:",
        "   poetry run python scripts/load_json_to_chromadb_dd.py",
        "",
        "4. 검색 테스트:",
        "   poetry run python scripts/test_esm3_search.py"
    ]

    for step in steps:
        print(step)

    print("\n" + "=" * 70)
    print("🔍 자동 다운로드 옵션")
    print("=" * 70)

    print("자동으로 ESM3 논문을 다운로드하시겠습니까? (y/n): ", end="")

def create_esm3_search_test():
    """Create a test script for ESM3 search functionality."""

    test_script = '''#!/usr/bin/env python3
"""
Test ESM3 Search in DD-RAPTOR

After adding ESM3 papers, use this script to test search functionality.
"""

import chromadb
import sys
from pathlib import Path

def test_esm3_search():
    """Test ESM3-related searches in DD-RAPTOR."""

    print("=" * 60)
    print("ESM3 DD-RAPTOR 검색 테스트")
    print("=" * 60)

    # Connect to DD-RAPTOR
    try:
        client = chromadb.PersistentClient(path="chromadb_data_dd")
        collection = client.get_collection(name="dd_papers_L0")

        print(f"DD-RAPTOR 연결 성공 - {collection.count()} 문서")

        # ESM3 specific queries
        esm3_queries = [
            "ESM3 language model evolution",
            "protein structure prediction Meta AI",
            "500 million years evolution simulation",
            "evolutionary scale modeling",
            "protein folding language model",
            "amino acid sequence generation"
        ]

        print("\\n🔍 ESM3 관련 검색 테스트:")
        print("-" * 40)

        for query in esm3_queries:
            try:
                results = collection.query(
                    query_texts=[query],
                    n_results=2
                )

                docs = results['documents'][0]
                metas = results['metadatas'][0] if results['metadatas'] else []

                print(f"\\n📝 '{query}':")
                if docs:
                    for i, (doc, meta) in enumerate(zip(docs, metas)):
                        title = meta.get('paper_title', 'Unknown')[:50]
                        preview = doc[:80].replace('\\n', ' ') + "..."
                        print(f"   [{i+1}] {title}")
                        print(f"       {preview}")
                else:
                    print("   ❌ 결과 없음")

            except Exception as e:
                print(f"   ⚠️ 검색 오류: {e}")

    except Exception as e:
        print(f"❌ DD-RAPTOR 연결 실패: {e}")
        print("\\n💡 해결 방법:")
        print("1. DD-RAPTOR 데이터베이스가 초기화되었는지 확인")
        print("2. ESM3 논문이 추가되었는지 확인")

    print("\\n" + "=" * 60)

if __name__ == "__main__":
    test_esm3_search()
'''

    # Save the test script
    test_script_path = Path("scripts/test_esm3_search.py")
    with open(test_script_path, 'w', encoding='utf-8') as f:
        f.write(test_script)

    print(f"✅ ESM3 검색 테스트 스크립트 생성: {test_script_path}")

def main():
    """Main execution."""
    download_esm3_papers()

    # Get user input
    try:
        choice = input().strip().lower()

        if choice in ['y', 'yes', '예', 'ㅇ']:
            print("\n🚀 ESM3 자동 다운로드 구현 중...")
            print("현재는 수동 다운로드를 권장합니다.")
            print("\n📖 권장 다운로드 사이트:")
            print("• arXiv: https://arxiv.org/abs/2411.12143 (ESM3)")
            print("• Papers with Code: https://paperswithcode.com/paper/esm")
            print("• Meta AI Research: https://ai.meta.com/research/")

        else:
            print("\n📝 수동 다운로드를 선택하셨습니다.")

        # Create test script anyway
        print("\n" + "=" * 50)
        create_esm3_search_test()

    except KeyboardInterrupt:
        print("\n\n👋 종료되었습니다.")

if __name__ == "__main__":
    main()