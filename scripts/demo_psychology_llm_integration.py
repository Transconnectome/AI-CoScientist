#!/usr/bin/env python3
"""
Psychology RAG + AI-CoScientist LLM Integration Demo

Demonstrates the integration of Psychology RAG system with the main AI-CoScientist
LLM infrastructure, including multi-provider routing and unified orchestration.
"""

import asyncio
import os
import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Safe imports with fallbacks
try:
    from src.services.llm.adapters.anthropic import AnthropicAdapter
    LLM_AVAILABLE = True
except Exception as e:
    print(f"❌ LLM services not available: {e}")
    LLM_AVAILABLE = False

try:
    from src.services.rag.unified_rag_orchestrator import (
        UnifiedRAGOrchestrator,
        QueryContext,
        QueryDomain,
        QueryComplexity,
        RAGStrategy
    )
    from src.services.rag.psychology_rag_strategy import PsychologyRAGStrategy
    RAG_AVAILABLE = True
except Exception as e:
    print(f"❌ RAG services not available: {e}")
    RAG_AVAILABLE = False

try:
    from src.agents.psychology_expert import PsychologyExpert
    AGENT_AVAILABLE = True
except Exception as e:
    print(f"❌ Agent services not available: {e}")
    AGENT_AVAILABLE = False

async def demo_psychology_llm_integration():
    """Demonstrate Psychology RAG + LLM integration."""
    print("🤖 Psychology RAG + AI-CoScientist LLM Integration Demo")
    print("=" * 80)
    print("Foundation Model 통합 심리학 AI 시스템 데모")
    print("Seoul National University Psychology Department")
    print("=" * 80)
    print()

    # Check system availability
    print("🔧 시스템 상태 확인")
    print("-" * 50)
    print(f"   LLM Services: {'✅' if LLM_AVAILABLE else '❌'}")
    print(f"   RAG Services: {'✅' if RAG_AVAILABLE else '❌'}")
    print(f"   Agent Services: {'✅' if AGENT_AVAILABLE else '❌'}")
    print()

    if not all([LLM_AVAILABLE, RAG_AVAILABLE, AGENT_AVAILABLE]):
        print("⚠️ 일부 서비스가 사용 불가합니다. Mock 데모를 실행합니다.")
        await demo_mock_integration()
        return

    # Initialize system components
    try:
        print("🚀 시스템 컴포넌트 초기화")
        print("-" * 50)

        # 1. Initialize LLM service
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            print("   ❌ ANTHROPIC_API_KEY 환경변수가 설정되지 않음")
            await demo_mock_integration()
            return

        llm_service = AnthropicAdapter(api_key=anthropic_api_key)
        print("   ✅ Anthropic LLM Service 초기화 완료")

        # 2. Initialize Psychology RAG Strategy
        psychology_rag = PsychologyRAGStrategy(llm_service=llm_service)
        print("   ✅ Psychology RAG Strategy 초기화 완료")

        # 3. Initialize RAG Orchestrator (simplified for demo)
        rag_orchestrator = UnifiedRAGOrchestrator()

        # Register psychology strategy
        rag_orchestrator.register_strategy(RAGStrategy.PSYCHOLOGY_RAG, psychology_rag)
        print("   ✅ Unified RAG Orchestrator 초기화 완료")

        # 4. Initialize Psychology Expert Agent
        psychology_agent = PsychologyExpert(rag_orchestrator=rag_orchestrator)
        print("   ✅ Psychology Expert Agent 초기화 완료")
        print()

        # Demo test queries
        test_queries = [
            {
                "query": "ADHD 아동의 실행기능 훈련 효과에 대한 연구는?",
                "description": "인지심리학 + 발달심리학 융합 질문"
            },
            {
                "query": "우울증 치료에서 인지행동치료의 효과성은 어떤가요?",
                "description": "임상심리학 질문"
            },
            {
                "query": "fMRI를 이용한 전전두엽 연구의 최근 동향은?",
                "description": "신경심리학 질문"
            }
        ]

        # Execute demo queries
        for i, test_case in enumerate(test_queries, 1):
            await demo_single_query(
                psychology_agent,
                test_case["query"],
                test_case["description"],
                i
            )

    except Exception as e:
        print(f"   ❌ 시스템 초기화 실패: {e}")
        await demo_mock_integration()

async def demo_single_query(agent, query, description, test_num):
    """Execute single query demo."""
    print(f"🧠 테스트 {test_num}: {description}")
    print("-" * 60)
    print(f"질문: {query}")
    print()

    try:
        # Execute psychology research analysis
        result = await agent.analyze_research_question(query)

        if result.get("available"):
            print("📊 분석 결과:")
            print(f"   도메인 분류: {result['domain_classification']['domain']}")
            print(f"   복잡도: {result['complexity']}")
            print(f"   신뢰도: {result['rag_response']['confidence']:.2f}")
            print()

            print("💡 RAG 응답:")
            print(f"   {result['rag_response']['answer'][:200]}...")
            print()

            if result['rag_response']['sources']:
                print(f"📄 참조 논문: {len(result['rag_response']['sources'])}편")
                for j, source in enumerate(result['rag_response']['sources'][:2], 1):
                    print(f"   {j}. {source.get('title', 'N/A')[:50]}...")
            print()

            if result.get('psychology_analysis'):
                analysis = result['psychology_analysis']
                if analysis.get('psychological_concepts'):
                    print(f"🔬 주요 심리학 개념: {', '.join(analysis['psychological_concepts'][:3])}")

                if analysis.get('research_methodologies'):
                    print(f"📈 연구 방법론: {', '.join(analysis['research_methodologies'][:3])}")
                print()

            if result.get('recommendations'):
                print("💭 연구 권장사항:")
                for rec in result['recommendations'][:2]:
                    print(f"   • {rec}")
                print()

        else:
            print(f"   ❌ 분석 실패: {result.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"   ❌ 쿼리 실행 실패: {e}")

    print("-" * 60)
    print()

async def demo_mock_integration():
    """Mock demo when services are not available."""
    print("🎭 Mock Integration Demo")
    print("-" * 50)

    mock_queries = [
        "ADHD 아동의 실행기능 훈련 효과에 대한 연구는?",
        "우울증 치료에서 인지행동치료의 효과성은 어떤가요?",
        "fMRI를 이용한 전전두엽 연구의 최근 동향은?"
    ]

    for i, query in enumerate(mock_queries, 1):
        print(f"테스트 {i}: {query}")
        print("   🎯 Mock 응답: Psychology RAG + LLM 시스템이 정상 작동할 것입니다.")
        print("   📊 예상 기능:")
        print("      • 한국어 심리학 논문 66편 검색")
        print("      • Anthropic Claude 기반 답변 생성")
        print("      • 심리학 도메인 분류 및 전문성 강화")
        print("      • 연구 권장사항 및 윤리적 고려사항 제공")
        print()

    print("✅ 실제 환경에서는 다음이 필요합니다:")
    print("   1. ANTHROPIC_API_KEY 환경변수 설정")
    print("   2. ChromaDB 서버 실행")
    print("   3. Psychology RAG 시스템 초기화")
    print()

async def demo_system_architecture():
    """Display system architecture."""
    print("🏗️ 통합 시스템 아키텍처")
    print("=" * 80)

    architecture = """
    사용자 질문 (한국어/영어)
           ↓
    [Psychology Expert Agent]
           ↓
    [Unified RAG Orchestrator] ← 전략 선택 및 라우팅
           ↓
    [Psychology RAG Strategy] ← 심리학 특화 처리
           ↓                    ↓
    [Korean NLP Pipeline]    [Vector Search]
           ↓                    ↓
    [Query Enhancement]      [ChromaDB: 66편 논문]
           ↓                    ↓
           └──→ [Context] ←─────┘
                    ↓
    [Multi-Provider LLM Service]
    ├── Anthropic Claude Sonnet 4.5
    ├── OpenAI GPT-4
    └── Google Gemini
                    ↓
           [전문적 심리학 답변]
           + 논문 참조 + 연구 권장사항
    """

    print(architecture)
    print()

    print("🔧 핵심 구성 요소:")
    components = [
        ("Psychology Expert Agent", "심리학 전문가 AI 에이전트"),
        ("Unified RAG Orchestrator", "통합 RAG 전략 조정기"),
        ("Psychology RAG Strategy", "심리학 특화 RAG 전략"),
        ("Multi-Provider LLM", "다중 LLM 공급자 (Anthropic, OpenAI, Google)"),
        ("Korean NLP Pipeline", "한국어 자연어 처리 파이프라인"),
        ("Psychology Vector Store", "심리학 논문 벡터 저장소 (66편)"),
        ("Domain Classification", "8개 심리학 하위분야 분류"),
        ("Query Enhancement", "한영 매핑 및 동의어 확장")
    ]

    for component, description in components:
        print(f"   ✅ {component}: {description}")
    print()

async def demo_capabilities():
    """Show system capabilities."""
    print("🎯 시스템 주요 기능")
    print("=" * 80)

    capabilities = {
        "언어 지원": [
            "한국어 심리학 용어 200+ 처리",
            "한영 전문용어 자동 매핑",
            "한국어 질문 → 영어 논문 검색"
        ],
        "심리학 전문성": [
            "8개 하위분야 자동 분류",
            "임상/인지/발달/신경심리학 특화",
            "연구 방법론 자동 식별",
            "윤리적 고려사항 제공"
        ],
        "LLM 통합": [
            "작업별 최적 모델 선택",
            "Anthropic Claude Sonnet 4.5",
            "멀티프로바이더 fallback",
            "토큰 사용량 최적화"
        ],
        "RAG 기능": [
            "66편 실제 논문 벡터 검색",
            "실시간 의미론적 매칭",
            "컨텍스트 관련성 평가",
            "신뢰도 점수 제공"
        ]
    }

    for category, features in capabilities.items():
        print(f"📋 {category}:")
        for feature in features:
            print(f"   • {feature}")
        print()

async def main():
    """Main demo function."""
    await demo_system_architecture()
    await demo_capabilities()
    await demo_psychology_llm_integration()

    print("🎊 Psychology RAG + LLM 통합 데모 완료!")
    print()
    print("💡 다음 단계:")
    print("   1. 실제 API 키 설정 후 테스트")
    print("   2. 웹 인터페이스 구현")
    print("   3. 추가 Foundation Model 통합")
    print("   4. 실시간 논문 업데이트 시스템")

if __name__ == "__main__":
    asyncio.run(main())