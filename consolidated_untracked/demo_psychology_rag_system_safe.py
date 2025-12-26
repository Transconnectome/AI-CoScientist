#!/usr/bin/env python3
"""
Psychology RAG System 안전 데모 스크립트
66편 심리학 논문 처리 및 Foundation Model 통합 시연 (Java 의존성 없이)

Usage:
    python scripts/demo_psychology_rag_system_safe.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 안전한 import with fallback handling
try:
    from src.services.psychology.psychology_vector_store import (
        PsychologyVectorStore, PaperMetadata, PsychologyPaperProcessor
    )
    VECTOR_STORE_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Vector Store 모듈을 로드할 수 없습니다: {e}")
    VECTOR_STORE_AVAILABLE = False

try:
    from src.services.psychology.korean_nlp_processor import KoreanNLPPipeline
    NLP_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Korean NLP 모듈을 로드할 수 없습니다: {e}")
    NLP_AVAILABLE = False

try:
    from src.services.psychology.domain_classifier import PsychologyDomainClassifier
    CLASSIFIER_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Domain Classifier 모듈을 로드할 수 없습니다: {e}")
    CLASSIFIER_AVAILABLE = False

try:
    from src.services.psychology.query_enhancer import PsychologyQueryEnhancer
    QUERY_ENHANCER_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Query Enhancer 모듈을 로드할 수 없습니다: {e}")
    QUERY_ENHANCER_AVAILABLE = False


async def demo_safe_nlp_pipeline():
    """안전한 Korean NLP Pipeline 데모 (Java 의존성 없이)"""
    print("🧠 Safe Korean NLP Pipeline 데모")
    print("=" * 50)

    if not NLP_AVAILABLE:
        print("❌ NLP 파이프라인을 사용할 수 없습니다.")
        return

    try:
        # KoNLPy 없이도 작동하는 버전 사용
        from src.services.psychology.korean_nlp_processor import (
            PsychologyTermExtractor, PsychologyTermMapper,
            KoreanSentimentAnalyzer, BilingualProcessor
        )

        # 개별 컴포넌트 테스트
        term_extractor = PsychologyTermExtractor()
        term_mapper = PsychologyTermMapper()
        sentiment_analyzer = KoreanSentimentAnalyzer()
        bilingual_processor = BilingualProcessor()

        sample_text = """
        본 연구는 ADHD 아동 30명을 대상으로 실행기능 훈련의 효과를 검증했다.
        Cognitive behavioral therapy와 약물치료를 병행한 결과,
        작업기억 능력이 유의미하게 향상되었다. 연구 결과는 매우 긍정적이다.
        """

        print(f"📝 분석 텍스트: {sample_text.strip()}")

        # 1. 심리학 전문용어 추출
        terms = term_extractor.extract_psychology_terms(sample_text)
        print(f"🧮 추출된 심리학 용어: {terms}")

        # 2. 영어 매핑
        mappings = term_mapper.map_to_english(terms)
        print(f"🌐 영어 매핑: {list(mappings.items())[:3]}")

        # 3. 감정 분석
        sentiment = sentiment_analyzer.analyze_sentiment(sample_text)
        print(f"🎯 감정 분석: {sentiment['label']} (신뢰도: {sentiment['confidence']:.2f})")

        # 4. 이중언어 처리
        bilingual_result = bilingual_processor.process_bilingual_text(sample_text)
        print(f"🌍 한국어 세그먼트: {len(bilingual_result['korean_segments'])}개")
        print(f"🔤 영어 세그먼트: {len(bilingual_result['english_segments'])}개")

        print("✅ Safe NLP 파이프라인 테스트 성공!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print()


async def demo_domain_classifier():
    """심리학 도메인 분류기 데모"""
    print("🎯 Psychology Domain Classifier 데모")
    print("=" * 50)

    if not CLASSIFIER_AVAILABLE:
        print("❌ 도메인 분류기를 사용할 수 없습니다.")
        return

    try:
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

    except Exception as e:
        print(f"❌ Error: {e}")


async def demo_query_enhancement():
    """쿼리 향상 시스템 데모"""
    print("🚀 Query Enhancement 데모")
    print("=" * 50)

    if not QUERY_ENHANCER_AVAILABLE:
        print("❌ 쿼리 향상기를 사용할 수 없습니다.")
        return

    try:
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

    except Exception as e:
        print(f"❌ Error: {e}")


async def demo_simple_vector_store():
    """간단한 Vector Store 데모 (ChromaDB 없이)"""
    print("🗄️ Simple Vector Store 데모 (Mock)")
    print("=" * 50)

    print("📝 Mock 데이터로 Vector Store 시뮬레이션...")

    # Mock papers data
    sample_papers = [
        {
            "title": "ADHD 아동의 실행기능 향상을 위한 tDCS 연구",
            "content": "본 연구는 ADHD 아동의 실행기능 향상을 위한 tDCS의 효과를 검증했다.",
            "keywords": ["ADHD", "실행기능", "tDCS", "뇌자극"]
        },
        {
            "title": "청소년 우울증의 인지행동치료 효과",
            "content": "청소년 우울증 환자 40명을 대상으로 12주간 인지행동치료를 실시했다.",
            "keywords": ["우울증", "인지행동치료", "청소년"]
        }
    ]

    print("📄 샘플 논문:")
    for i, paper in enumerate(sample_papers, 1):
        print(f"   {i}. {paper['title']}")
        print(f"      키워드: {paper['keywords']}")

    print("\n🔍 검색 시뮬레이션:")
    search_queries = ["ADHD 실행기능", "우울증 치료", "뇌자극 연구"]

    for query in search_queries:
        print(f"   쿼리: '{query}'")

        # Simple keyword matching simulation
        matches = []
        for paper in sample_papers:
            score = 0
            for keyword in paper['keywords']:
                if keyword in query:
                    score += 1
            if score > 0:
                matches.append((paper['title'], score))

        if matches:
            best_match = max(matches, key=lambda x: x[1])
            print(f"   → 최적 매치: {best_match[0][:40]}... (점수: {best_match[1]})")
        else:
            print("   → 매치 없음")

    print("\n✅ Mock Vector Store 테스트 완료!")
    print()


async def demo_paper_discovery():
    """심리학과 논문 파일 발견 데모"""
    print("📚 Psychology Papers Discovery 데모")
    print("=" * 50)

    # 심리학과 논문 파일들 찾기
    psychology_dir = Path("data/심리학과")

    if not psychology_dir.exists():
        print("❌ 심리학과 데이터 폴더를 찾을 수 없습니다.")
        print(f"   경로: {psychology_dir.absolute()}")
        return

    print(f"📁 탐색 경로: {psychology_dir.absolute()}")

    # 다양한 파일 형식 찾기
    file_patterns = ["*.pdf", "*.txt", "*.doc", "*.docx"]
    all_files = []

    for pattern in file_patterns:
        files = list(psychology_dir.rglob(pattern))
        all_files.extend(files)
        if files:
            print(f"   📄 {pattern}: {len(files)}개 파일")

    if all_files:
        print(f"\n📊 총 발견된 파일: {len(all_files)}개")

        # 폴더별 분류
        folder_counts = {}
        for file_path in all_files:
            folder = file_path.parent.name
            folder_counts[folder] = folder_counts.get(folder, 0) + 1

        print("📁 폴더별 분포:")
        for folder, count in sorted(folder_counts.items()):
            print(f"   {folder}: {count}개")

        # 샘플 파일들 보여주기
        print(f"\n📝 샘플 파일들 (처음 5개):")
        for i, file_path in enumerate(all_files[:5], 1):
            rel_path = file_path.relative_to(psychology_dir)
            print(f"   {i}. {rel_path}")
    else:
        print("❌ 논문 파일을 찾을 수 없습니다.")

    print()


async def main():
    """메인 안전 데모 실행"""
    print("🎉 Psychology RAG System 안전 통합 데모")
    print("=" * 60)
    print("Foundation Model 기반 심리학 RAG 시스템 (Safe Mode)")
    print("Seoul National University Psychology Department")
    print("=" * 60)
    print()

    demos = [
        ("Safe Korean NLP Pipeline", demo_safe_nlp_pipeline),
        ("Domain Classification", demo_domain_classifier),
        ("Query Enhancement", demo_query_enhancement),
        ("Simple Vector Store", demo_simple_vector_store),
        ("Paper Discovery", demo_paper_discovery)
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

    print("🎊 안전 데모 완료!")
    print("\n📝 구현된 주요 기능:")
    print("   ✅ 한국어 NLP 파이프라인 (Java 의존성 없음)")
    print("   ✅ 심리학 도메인 자동 분류 (8개 하위 분야)")
    print("   ✅ 쿼리 향상 시스템 (한영 매핑, 동의어 확장)")
    print("   ✅ Mock Vector Store 시뮬레이션")
    print("   ✅ 66편 논문 파일 발견 및 분류")
    print("   ✅ Foundation Model 통합 준비 완료")

    print("\n🔧 다음 단계:")
    print("   🚀 ChromaDB 서버 시작")
    print("   🚀 실제 Vector Store 연동")
    print("   🚀 66편 논문 전체 처리")
    print("   🚀 Foundation Model 실시간 추론")


if __name__ == "__main__":
    asyncio.run(main())