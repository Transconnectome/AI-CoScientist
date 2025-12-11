#!/usr/bin/env python3
"""
Simple Database Check

Direct check of ChromaDB databases to diagnose the issue.
"""

import chromadb
from pathlib import Path

def simple_check():
    """Simple direct check of new papers database."""

    print("🔍 간단한 데이터베이스 직접 검사")
    print("=" * 50)

    # Check the newest papers database
    db_path = "chromadb_new_papers_20251210_204818"

    if not Path(db_path).exists():
        print(f"❌ 데이터베이스 없음: {db_path}")
        return

    print(f"📂 데이터베이스: {db_path}")

    try:
        # Connect
        client = chromadb.PersistentClient(path=db_path)
        collections = client.list_collections()

        print(f"📁 컬렉션 수: {len(collections)}")

        for collection_info in collections:
            collection_name = collection_info.name
            collection = client.get_collection(name=collection_name)

            count = collection.count()
            print(f"📄 {collection_name}: {count} 문서")

            if count > 0:
                # Try different approaches to get data
                print("\n🔍 데이터 접근 테스트:")

                # Method 1: Get with limit
                try:
                    results = collection.get(limit=2)
                    print(f"  ✅ Get 방법: {len(results.get('documents', []))} 문서 반환")

                    # Check what's actually returned
                    for key in results.keys():
                        data = results[key]
                        if data and len(data) > 0:
                            print(f"    📋 {key}: {len(data)} 항목")
                            if key == 'metadatas' and data[0]:
                                meta = data[0]
                                title = meta.get('paper_title', meta.get('title', 'Unknown'))
                                paper_type = meta.get('paper_type', 'Unknown')
                                print(f"      예시: {title} ({paper_type})")
                        else:
                            print(f"    ❌ {key}: None 또는 빈 데이터")

                except Exception as e:
                    print(f"  ❌ Get 실패: {e}")

                # Method 2: Query
                try:
                    query_results = collection.query(
                        query_texts=["test"],
                        n_results=1
                    )
                    print(f"  ✅ Query 방법: {len(query_results.get('documents', [[]])[0])} 문서 반환")

                except Exception as e:
                    print(f"  ❌ Query 실패: {e}")

                # Method 3: Peek
                try:
                    peek_results = collection.peek(limit=2)
                    print(f"  ✅ Peek 방법: {len(peek_results.get('documents', []))} 문서 반환")

                    if peek_results.get('metadatas'):
                        for i, meta in enumerate(peek_results['metadatas'][:1]):
                            if meta:
                                title = meta.get('paper_title', meta.get('title', 'Unknown'))
                                paper_type = meta.get('paper_type', 'Unknown')
                                print(f"    📋 논문 {i+1}: {title} ({paper_type})")

                except Exception as e:
                    print(f"  ❌ Peek 실패: {e}")

    except Exception as e:
        print(f"❌ 전체 연결 실패: {e}")

if __name__ == "__main__":
    simple_check()