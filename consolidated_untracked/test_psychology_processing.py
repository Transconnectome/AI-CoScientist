#!/usr/bin/env python3
"""
Psychology RAG System 샘플 테스트
소규모 테스트를 통한 시스템 검증

Usage:
    python scripts/test_psychology_processing.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Safe imports
try:
    from src.services.psychology.psychology_vector_store import (
        PsychologyVectorStore, PaperMetadata
    )
    VECTOR_STORE_AVAILABLE = True
except Exception as e:
    print(f"❌ Vector Store not available: {e}")
    VECTOR_STORE_AVAILABLE = False

try:
    from src.services.psychology.korean_nlp_processor import KoreanNLPPipeline
    NLP_AVAILABLE = True
except Exception as e:
    print(f"❌ Korean NLP not available: {e}")
    NLP_AVAILABLE = False


async def test_mock_paper_processing():
    """Mock 논문 데이터로 시스템 테스트"""
    print("🧪 Mock Paper Processing 테스트")
    print("=" * 50)

    # Mock 논문 데이터
    mock_papers = [
        {
            "title": "ADHD 아동의 실행기능 향상을 위한 tDCS 뇌자극 연구",
            "content": """
            본 연구는 ADHD 아동 30명을 대상으로 tDCS(transcranial Direct Current Stimulation)
            뇌자극이 실행기능에 미치는 효과를 검증하였다. 연구 참가자들은 무작위로 실험군과
            대조군으로 배정되어 20분간 전전두엽 영역에 tDCS를 적용받았다.

            실험 결과, 작업기억(working memory) 과제에서 실험군의 정확도가 대조군에 비해
            유의미하게 향상되었다(p < 0.05). 특히 인지유연성과 주의집중 능력에서 두드러진
            개선이 관찰되었다.

            이러한 결과는 ADHD 아동의 실행기능 장애가 tDCS를 통한 비침습적 뇌자극으로
            개선될 수 있음을 시사한다. 향후 장기적인 효과와 최적의 자극 프로토콜에 대한
            추가 연구가 필요하다.
            """,
            "metadata": {
                "authors": ["김민수", "이지영"],
                "year": 2023,
                "journal": "Korean Journal of Psychology",
                "keywords": ["ADHD", "실행기능", "tDCS", "뇌자극", "아동"],
                "folder": "박주용"
            }
        },
        {
            "title": "청소년 우울증의 인지행동치료 효과에 관한 메타분석",
            "content": """
            본 메타분석은 청소년 우울증에 대한 인지행동치료(CBT)의 효과를 종합적으로
            검토하였다. 2015년부터 2023년까지 발표된 28편의 무작위 대조시험(RCT)을
            포함하여 총 1,847명의 청소년을 대상으로 분석하였다.

            분석 결과, CBT는 청소년 우울증상 감소에 중간 크기의 효과크기(Cohen's d = 0.65)를
            보였다. 특히 부정적 사고 패턴의 변화와 행동 활성화 기법이 효과적인 것으로
            나타났다.

            치료 기간별로는 12-16주 프로그램이 가장 효과적이었으며, 집단치료보다
            개인치료에서 더 큰 효과크기를 보였다. 치료 종료 후 6개월 추적 조사에서도
            치료 효과가 유지되는 것으로 확인되었다.
            """,
            "metadata": {
                "authors": ["정수아", "박영진"],
                "year": 2023,
                "journal": "Clinical Psychology Review",
                "keywords": ["우울증", "인지행동치료", "청소년", "메타분석"],
                "folder": "이수현"
            }
        },
        {
            "title": "fMRI를 이용한 전전두엽 기능과 의사결정 과정 분석",
            "content": """
            본 연구는 기능적 자기공명영상(fMRI)을 사용하여 건강한 성인의 의사결정
            과정에서 전전두엽의 활성화 패턴을 조사하였다. 24명의 참가자가 경제적
            의사결정 과제를 수행하는 동안 뇌 활성화를 측정하였다.

            결과적으로, 복잡한 의사결정 상황에서 배외측 전전두엽(dlPFC)과 복내측
            전전두엽(vmPFC)에서 증가된 활성화가 관찰되었다. 특히 위험 회피 성향이
            높은 개인들에서 전전두엽 활성화가 더욱 두드러졌다.

            또한 의사결정의 정확도와 전전두엽 활성화 간에 정적 상관관계가 발견되어
            (r = 0.72, p < 0.001), 전전두엽이 효율적인 의사결정에 중요한 역할을
            함을 확인하였다.
            """,
            "metadata": {
                "authors": ["최영호", "한지민"],
                "year": 2024,
                "journal": "Cognitive Neuroscience",
                "keywords": ["fMRI", "전전두엽", "의사결정", "신경영상"],
                "folder": "안우영"
            }
        }
    ]

    # 1. Korean NLP 테스트
    if NLP_AVAILABLE:
        print("🧠 Korean NLP 분석 테스트")
        nlp_pipeline = KoreanNLPPipeline()

        for i, paper in enumerate(mock_papers, 1):
            try:
                result = await nlp_pipeline.analyze_text(paper["content"][:500])
                print(f"   📄 논문 {i}: {paper['title'][:30]}...")
                print(f"      심리학 용어: {[term.korean for term in result.psychology_terms][:3]}")
                print(f"      감정: {result.sentiment['label']} ({result.sentiment['confidence']:.2f})")
                print(f"      영어 매핑: {list(result.english_mappings.items())[:2]}")
            except Exception as e:
                print(f"      ❌ NLP 분석 오류: {e}")
        print()
    else:
        print("⏭️ Korean NLP 건너뛰기 (사용 불가)")

    # 2. Vector Store 테스트
    if VECTOR_STORE_AVAILABLE:
        print("🗄️ Vector Store 테스트")
        try:
            vector_store = PsychologyVectorStore()

            # 논문들 추가
            print("   📝 Mock 논문 추가 중...")
            for i, paper in enumerate(mock_papers, 1):
                metadata = PaperMetadata(
                    title=paper["title"],
                    authors=paper["metadata"]["authors"],
                    year=paper["metadata"]["year"],
                    journal=paper["metadata"]["journal"],
                    keywords=paper["metadata"]["keywords"]
                )

                success = await vector_store.add_paper(
                    title=paper["title"],
                    content=paper["content"],
                    metadata=metadata
                )

                status = "✅" if success else "❌"
                print(f"      {status} 논문 {i}: {paper['title'][:30]}...")

            print()

            # 검색 테스트
            print("   🔍 검색 테스트...")
            test_queries = [
                "ADHD 실행기능",
                "우울증 치료",
                "fMRI 뇌영상",
                "전전두엽 기능"
            ]

            for query in test_queries:
                try:
                    results = await vector_store.search_papers(query, limit=2)
                    print(f"      쿼리 '{query}': {len(results)}개 결과")

                    for result in results[:1]:  # 첫 번째 결과만
                        print(f"         📄 {result.title[:40]}... (유사도: {result.similarity_score:.3f})")

                except Exception as e:
                    print(f"      쿼리 '{query}': 검색 오류 - {e}")

            print()

        except Exception as e:
            print(f"   ❌ Vector Store 오류: {e}")
    else:
        print("⏭️ Vector Store 건너뛰기 (사용 불가)")

    print("✅ Mock Paper Processing 테스트 완료!")


async def test_real_file_discovery():
    """실제 논문 파일 발견 테스트"""
    print("📁 실제 파일 발견 테스트")
    print("=" * 50)

    psychology_dir = Path("data/심리학과")

    if not psychology_dir.exists():
        print(f"❌ 심리학과 폴더 없음: {psychology_dir}")
        return

    # PDF 파일들 찾기
    pdf_files = list(psychology_dir.rglob("*.pdf"))
    print(f"📄 발견된 PDF 파일: {len(pdf_files)}개")

    # 폴더별 분류
    folder_counts = {}
    for pdf_file in pdf_files:
        folder = pdf_file.parent.name
        folder_counts[folder] = folder_counts.get(folder, 0) + 1

    print("📁 폴더별 분포:")
    for folder, count in sorted(folder_counts.items()):
        print(f"   {folder}: {count}개")

    # 샘플 파일들
    if pdf_files:
        print(f"\n📝 샘플 파일들 (처음 3개):")
        for i, pdf_file in enumerate(pdf_files[:3], 1):
            rel_path = pdf_file.relative_to(psychology_dir)
            size_mb = pdf_file.stat().st_size / (1024 * 1024)
            print(f"   {i}. {rel_path} ({size_mb:.1f}MB)")

    print()


async def main():
    """메인 테스트 함수"""
    print("🎉 Psychology RAG System 샘플 테스트")
    print("=" * 60)
    print("Foundation Model 기반 심리학 RAG 시스템 검증")
    print("Seoul National University Psychology Department")
    print("=" * 60)
    print()

    # 컴포넌트 가용성 확인
    print("🔧 시스템 컴포넌트 확인:")
    print(f"   Vector Store: {'✅' if VECTOR_STORE_AVAILABLE else '❌'}")
    print(f"   Korean NLP: {'✅' if NLP_AVAILABLE else '❌'}")
    print()

    # 테스트 실행
    test_functions = [
        test_mock_paper_processing,
        test_real_file_discovery
    ]

    for test_func in test_functions:
        try:
            await test_func()
        except Exception as e:
            print(f"❌ {test_func.__name__} 오류: {e}")
            import traceback
            traceback.print_exc()

        print("-" * 60)
        print()

    print("🎊 샘플 테스트 완료!")
    print("\n📝 다음 단계:")
    print("   🚀 Docker 서비스 완료 대기")
    print("   🗄️ Production ChromaDB 연동")
    print("   📦 66편 논문 전체 배치 처리")
    print("   🧠 Foundation Models 통합 테스트")


if __name__ == "__main__":
    asyncio.run(main())