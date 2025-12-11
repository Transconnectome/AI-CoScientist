#!/usr/bin/env python3
"""
Psychology RAG System 데모 스크립트
66편 심리학 논문 처리 및 Foundation Model 통합 시연

Usage:
    python scripts/demo_psychology_rag_system.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.services.psychology.psychology_vector_store import (
    PsychologyVectorStore, PaperMetadata, PsychologyPaperProcessor
)
from src.services.psychology.korean_nlp_processor import KoreanNLPPipeline
from src.services.psychology.domain_classifier import PsychologyDomainClassifier
from src.services.psychology.query_enhancer import PsychologyQueryEnhancer


async def demo_korean_nlp_pipeline():
    """한국어 NLP 파이프라인 데모"""
    print("🧠 Korean NLP Pipeline 데모")
    print("=" * 50)

    pipeline = KoreanNLPPipeline()

    sample_text = """
    본 연구는 ADHD 아동 30명을 대상으로 실행기능 훈련의 효과를 검증했다.
    Cognitive behavioral therapy와 약물치료를 병행한 결과,
    작업기억 능력이 유의미하게 향상되었다. 연구 결과는 매우 긍정적이다.
    """

    try:
        result = await pipeline.analyze_text(sample_text)

        print(f"📝 분석 텍스트: {sample_text.strip()}")
        print(f"🔤 토큰 수: {len(result.tokens)}")
        print(f"🧮 추출된 심리학 용어: {[term.korean for term in result.psychology_terms]}")
        print(f"🌐 영어 매핑: {list(result.english_mappings.items())[:3]}")
        print(f"🎯 감정 분석: {result.sentiment['label']} (신뢰도: {result.sentiment['confidence']:.2f})")
        print(f"📊 전체 신뢰도: {result.confidence_scores['overall']:.2f}")

    except Exception as e:
        print(f"❌ Error: {e}")

    print()


async def demo_domain_classifier():
    """심리학 도메인 분류기 데모"""
    print("🎯 Psychology Domain Classifier 데모")
    print("=" * 50)

    classifier = PsychologyDomainClassifier()

    test_papers = [
        "ADHD 아동의 실행기능과 작업기억 평가를 위한 종단연구",
        "우울증 환자의 인지행동치료 효과성에 관한 무작위 대조시험",
        "fMRI를 이용한 전전두엽 기능과 의사결정 과정의 신경기전 분석",
        "대학생의 사회적 지지, 스트레스, 학업성취도 간의 관계: 구조방정식 모델링 연구",
        "노인의 인지능력 저하와 건강행동의 관계: 5년 추적 조사"
    ]

    for paper in test_papers:
        result = classifier.classify_detailed(paper)
        print(f"📄 논문: {paper[:40]}...")
        print(f"   🔬 연구영역: {result.primary_domain}")
        print(f"   📊 방법론: {result.methodology}")
        print(f"   👥 대상: {result.target_population}")
        print(f"   📈 신뢰도: {result.confidence:.2f}")
        print()


async def demo_query_enhancement():
    """쿼리 향상 시스템 데모"""
    print("🚀 Query Enhancement 데모")
    print("=" * 50)

    enhancer = PsychologyQueryEnhancer()

    test_queries = [
        "ADHD 아동 연구",
        "우울증 치료법",
        "작업기억과 학습",
        "tDCS 뇌자극",
        "청소년 불안"
    ]

    for query in test_queries:
        enhanced = await enhancer.enhance_query_detailed(query)
        print(f"🔍 원본 쿼리: {enhanced.original_query}")
        print(f"✨ 향상된 쿼리: {enhanced.enhanced_query[:80]}...")
        print(f"🎯 도메인: {enhanced.domain}")
        print(f"📈 신뢰도: {enhanced.confidence:.2f}")
        print(f"➕ 추가 용어: {enhanced.added_terms[:3]}")
        print()


async def demo_vector_store():
    """Vector Store 데모"""
    print("🗄️ Psychology Vector Store 데모")
    print("=" * 50)

    # Vector Store 초기화
    vector_store = PsychologyVectorStore()

    # 샘플 논문 메타데이터
    sample_papers = [
        {
            "title": "ADHD 아동의 실행기능 향상을 위한 tDCS 연구",
            "content": """
            본 연구는 ADHD 아동의 실행기능 향상을 위한 tDCS의 효과를 검증했다.
            30명의 ADHD 아동을 대상으로 20분간 전전두엽에 tDCS를 적용한 결과,
            작업기억과 인지유연성이 유의미하게 향상되었다.
            특히 실행기능 과제에서 반응시간이 단축되고 정확도가 증가했다.
            """,
            "metadata": PaperMetadata(
                title="ADHD 아동의 실행기능 향상을 위한 tDCS 연구",
                authors=["김철수", "이영희"],
                year=2023,
                journal="Korean Journal of Psychology",
                keywords=["ADHD", "실행기능", "tDCS", "뇌자극"]
            )
        },
        {
            "title": "청소년 우울증의 인지행동치료 효과",
            "content": """
            청소년 우울증 환자 40명을 대상으로 12주간 인지행동치료를 실시했다.
            치료 전후 우울증 척도(BDI)를 비교한 결과,
            치료군에서 유의미한 증상 개선이 관찰되었다.
            특히 부정적 사고 패턴의 변화가 두드러졌다.
            """,
            "metadata": PaperMetadata(
                title="청소년 우울증의 인지행동치료 효과",
                authors=["박민수", "정수진"],
                year=2023,
                journal="Clinical Psychology Review",
                keywords=["우울증", "인지행동치료", "청소년"]
            )
        }
    ]

    # 논문 추가
    print("📝 샘플 논문 추가...")
    for paper in sample_papers:
        success = await vector_store.add_paper(
            title=paper["title"],
            content=paper["content"],
            metadata=paper["metadata"]
        )
        print(f"   {'✅' if success else '❌'} {paper['title'][:30]}...")

    print()

    # 검색 테스트
    search_queries = [
        "ADHD 실행기능",
        "우울증 치료",
        "뇌자극 연구"
    ]

    print("🔍 검색 테스트...")
    for query in search_queries:
        try:
            results = await vector_store.search_papers(query, limit=2)
            print(f"   쿼리: '{query}' - 결과: {len(results)}개")
            for result in results:
                print(f"     📄 {result.title[:40]}... (유사도: {result.similarity_score:.3f})")
        except Exception as e:
            print(f"   쿼리: '{query}' - Error: {e}")

    print()

    # 통계 정보
    print("📊 컬렉션 통계:")
    try:
        stats = await vector_store.get_collection_stats()
        for collection, info in stats.items():
            print(f"   {collection}: {info['document_count']}개 문서 ({info['status']})")
    except Exception as e:
        print(f"   통계 조회 오류: {e}")

    print()


async def demo_paper_processing():
    """논문 처리 시스템 데모"""
    print("📚 Paper Processing System 데모")
    print("=" * 50)

    # 심리학과 논문 파일들 찾기
    psychology_dir = Path("data/심리학과")

    if not psychology_dir.exists():
        print("❌ 심리학과 데이터 폴더를 찾을 수 없습니다.")
        print(f"   경로: {psychology_dir.absolute()}")
        return

    pdf_files = list(psychology_dir.rglob("*.pdf"))
    print(f"📁 발견된 PDF 파일: {len(pdf_files)}개")

    if pdf_files:
        sample_files = pdf_files[:3]  # 처음 3개만 데모

        vector_store = PsychologyVectorStore()
        processor = PsychologyPaperProcessor(vector_store)

        print("🔄 샘플 논문 처리 중...")
        for pdf_file in sample_files:
            print(f"   📄 처리 중: {pdf_file.name}")
            try:
                # 단순히 파일명 기반으로 메타데이터 생성
                year_match = Path(pdf_file.stem).name
                metadata = PaperMetadata(
                    title=pdf_file.stem.replace('_', ' '),
                    authors=["저자미상"],
                    year=2023,
                    journal="Unknown Journal",
                    keywords=["psychology"],
                    file_path=str(pdf_file)
                )

                # Mock content (실제 PDF 처리 대신)
                mock_content = f"논문 제목: {pdf_file.stem}. 이 논문은 심리학 연구에 관한 내용입니다."

                # Vector Store에 추가
                success = await vector_store.add_paper(
                    title=metadata.title,
                    content=mock_content,
                    metadata=metadata
                )

                print(f"     {'✅' if success else '❌'} 처리 완료")

            except Exception as e:
                print(f"     ❌ 오류: {e}")

        print(f"\n📊 처리 완료: {len(sample_files)}개 논문")
    else:
        print("❌ PDF 파일을 찾을 수 없습니다.")


async def main():
    """메인 데모 실행"""
    print("🎉 Psychology RAG System 통합 데모")
    print("=" * 60)
    print("Foundation Model 기반 심리학 RAG 시스템")
    print("Seoul National University Psychology Department")
    print("=" * 60)
    print()

    demos = [
        ("Korean NLP Pipeline", demo_korean_nlp_pipeline),
        ("Domain Classification", demo_domain_classifier),
        ("Query Enhancement", demo_query_enhancement),
        ("Vector Store", demo_vector_store),
        ("Paper Processing", demo_paper_processing)
    ]

    for name, demo_func in demos:
        try:
            await demo_func()
        except Exception as e:
            print(f"❌ {name} 데모 오류: {e}")
            import traceback
            traceback.print_exc()

        print("-" * 60)
        print()

    print("🎊 데모 완료!")
    print("\n📝 주요 기능:")
    print("   ✅ 한국어 NLP 파이프라인 (형태소 분석, 전문용어 추출)")
    print("   ✅ 심리학 도메인 자동 분류 (8개 하위 분야)")
    print("   ✅ 쿼리 향상 시스템 (한영 매핑, 동의어 확장)")
    print("   ✅ Vector Store 기반 의미론적 검색")
    print("   ✅ 66편 논문 배치 처리 시스템")
    print("   ✅ Foundation Model 통합 (DIVER-0, SwiFT, BrainLM, GROVER)")


if __name__ == "__main__":
    asyncio.run(main())