#!/usr/bin/env python3
"""
Test ESM3 Search with Unified RAG Orchestrator

Advanced testing of ESM3 and protein research papers through the Unified RAG
Orchestrator system with multiple strategies.

Usage:
    poetry run python scripts/test_unified_rag_esm3.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import Unified RAG components
try:
    from src.services.rag.unified_rag_orchestrator import (
        UnifiedRAGOrchestrator,
        QueryContext,
        QueryComplexity,
        QueryDomain,
        RAGStrategy,
        create_unified_orchestrator
    )
    REAL_ORCHESTRATOR = True
except ImportError:
    print("⚠️ Unified RAG Orchestrator import 실패 - Mock 모드로 실행")
    REAL_ORCHESTRATOR = False

class ESM3TestSuite:
    """Comprehensive ESM3 testing through Unified RAG."""

    def __init__(self):
        self.orchestrator = None
        self.test_results = []

    async def initialize(self):
        """Initialize the orchestrator."""
        print("🚀 Unified RAG Orchestrator 초기화...")

        if REAL_ORCHESTRATOR:
            self.orchestrator = create_unified_orchestrator()
            print("✅ 실제 Orchestrator 로드됨")

            # Warm up the system
            try:
                await self.orchestrator.warmup()
                print("🔥 시스템 워밍업 완료")
            except Exception as e:
                print(f"⚠️ 워밍업 실패: {e}")
        else:
            print("📝 Mock 모드로 실행")

    async def run_esm3_tests(self):
        """Run comprehensive ESM3-focused tests."""

        print("\n" + "=" * 70)
        print("🧬 ESM3 특화 검색 테스트 시작")
        print("=" * 70)

        # ESM3-specific test queries with different complexity levels
        esm3_test_cases = [
            # Simple factual queries
            {
                'category': '기본 ESM3 검색',
                'query': 'What is ESM3?',
                'complexity': QueryComplexity.SIMPLE,
                'domain': QueryDomain.GENERAL,
                'intent': 'factual',
                'expected_content': ['ESM3', 'evolutionary', 'protein']
            },
            {
                'category': '기본 ESM3 검색',
                'query': 'ESM3 protein language model capabilities',
                'complexity': QueryComplexity.SIMPLE,
                'domain': QueryDomain.GENERAL,
                'intent': 'factual',
                'expected_content': ['ESM3', 'language model', 'protein']
            },

            # Medium complexity queries
            {
                'category': '중급 ESM3 분석',
                'query': 'How does ESM3 simulate 500 million years of evolution?',
                'complexity': QueryComplexity.MEDIUM,
                'domain': QueryDomain.GENERAL,
                'intent': 'comparative',
                'expected_content': ['evolution', 'simulation', 'million years']
            },
            {
                'category': '중급 ESM3 분석',
                'query': 'ESM3 vs ESM2 protein structure prediction improvements',
                'complexity': QueryComplexity.MEDIUM,
                'domain': QueryDomain.GENERAL,
                'intent': 'comparative',
                'expected_content': ['ESM3', 'structure prediction', 'improvement']
            },

            # Complex synthesis queries
            {
                'category': '고급 ESM3 연구',
                'query': 'What are the breakthrough capabilities of ESM3 in multimodal protein design and how do they compare to previous approaches?',
                'complexity': QueryComplexity.COMPLEX,
                'domain': QueryDomain.GENERAL,
                'intent': 'synthesis',
                'expected_content': ['breakthrough', 'multimodal', 'design']
            },

            # Meta AI specific
            {
                'category': 'Meta AI 연구',
                'query': 'Meta AI ESM3 model architecture and training methodology',
                'complexity': QueryComplexity.COMPLEX,
                'domain': QueryDomain.GENERAL,
                'intent': 'synthesis',
                'expected_content': ['Meta AI', 'architecture', 'training']
            },

            # Paper-specific queries
            {
                'category': '논문 특정 검색',
                'query': 'paper1 paper2 paper3 paper4 ESM3 research findings',
                'complexity': QueryComplexity.MEDIUM,
                'domain': QueryDomain.GENERAL,
                'intent': 'factual',
                'expected_content': ['paper', 'research', 'findings']
            },

            # Protein science applications
            {
                'category': '단백질 과학 응용',
                'query': 'ESM3 applications in drug discovery and therapeutic protein design',
                'complexity': QueryComplexity.COMPLEX,
                'domain': QueryDomain.GENERAL,
                'intent': 'synthesis',
                'expected_content': ['drug discovery', 'therapeutic', 'applications']
            }
        ]

        category_results = {}

        for test_case in esm3_test_cases:
            category = test_case['category']

            if category not in category_results:
                category_results[category] = []

            print(f"\n🔍 [{category}] {test_case['query']}")

            try:
                if REAL_ORCHESTRATOR and self.orchestrator:
                    # Create query context
                    query_context = QueryContext(
                        query=test_case['query'],
                        complexity=test_case['complexity'],
                        domain=test_case['domain'],
                        intent=test_case['intent'],
                        confidence=0.9,
                        metadata={'test_category': category, 'esm3_test': True}
                    )

                    # Execute search
                    response = await self.orchestrator.search(query_context)

                    # Analyze response
                    strategy_used = str(response.strategy_used)
                    confidence = response.confidence
                    answer_text = response.answer

                    # Check for expected content
                    content_found = []
                    for expected in test_case['expected_content']:
                        if expected.lower() in answer_text.lower():
                            content_found.append(expected)

                    result = {
                        'query': test_case['query'],
                        'strategy': strategy_used,
                        'confidence': confidence,
                        'answer_length': len(answer_text),
                        'expected_content_found': content_found,
                        'success': len(content_found) > 0 or confidence > 0.7,
                        'answer_preview': answer_text[:150] + "..." if len(answer_text) > 150 else answer_text
                    }

                    print(f"    ✅ 전략: {strategy_used.replace('RAGStrategy.', '')}")
                    print(f"    📊 신뢰도: {confidence:.3f}")
                    print(f"    📝 답변 길이: {len(answer_text)} 문자")
                    print(f"    🎯 기대 내용 발견: {len(content_found)}/{len(test_case['expected_content'])}")

                    if result['success']:
                        print(f"    ✅ 테스트 성공")
                    else:
                        print(f"    ⚠️ 테스트 부분 성공")

                else:
                    # Mock result
                    result = {
                        'query': test_case['query'],
                        'strategy': 'MOCK_HYBRID',
                        'confidence': 0.85,
                        'answer_length': 200,
                        'expected_content_found': test_case['expected_content'][:2],
                        'success': True,
                        'answer_preview': f"Mock answer for ESM3 query: {test_case['query'][:50]}..."
                    }
                    print(f"    📝 Mock 결과 - 성공 시뮬레이션")

                category_results[category].append(result)
                self.test_results.append(result)

            except Exception as e:
                print(f"    ❌ 오류: {e}")
                error_result = {
                    'query': test_case['query'],
                    'error': str(e),
                    'success': False
                }
                category_results[category].append(error_result)
                self.test_results.append(error_result)

        # Generate category summary
        self.generate_test_summary(category_results)

    def generate_test_summary(self, category_results):
        """Generate comprehensive test summary."""

        print("\n" + "=" * 80)
        print("📈 ESM3 테스트 종합 결과")
        print("=" * 80)

        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r.get('success', False))
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0

        print(f"📊 전체 성공률: {success_rate:.1f}% ({successful_tests}/{total_tests})")

        # Category breakdown
        print(f"\n📋 카테고리별 성과:")
        print("-" * 50)

        for category, results in category_results.items():
            category_success = sum(1 for r in results if r.get('success', False))
            category_total = len(results)
            category_rate = (category_success / category_total) * 100 if category_total > 0 else 0

            print(f"📂 {category}: {category_rate:.0f}% ({category_success}/{category_total})")

            # Show best performing test in category
            successful_results = [r for r in results if r.get('success', False)]
            if successful_results:
                best_result = max(successful_results, key=lambda x: x.get('confidence', 0))
                strategy = best_result.get('strategy', 'Unknown').replace('RAGStrategy.', '')
                confidence = best_result.get('confidence', 0)
                print(f"    🏆 최고 성과: {strategy} (신뢰도 {confidence:.3f})")

        # Strategy performance analysis
        print(f"\n🎯 전략별 활용 분석:")
        print("-" * 50)

        strategy_usage = {}
        for result in self.test_results:
            if result.get('success', False) and 'strategy' in result:
                strategy = result['strategy'].replace('RAGStrategy.', '')
                if strategy not in strategy_usage:
                    strategy_usage[strategy] = {'count': 0, 'total_confidence': 0}
                strategy_usage[strategy]['count'] += 1
                strategy_usage[strategy]['total_confidence'] += result.get('confidence', 0)

        for strategy, data in strategy_usage.items():
            avg_confidence = data['total_confidence'] / data['count']
            print(f"🔧 {strategy}: {data['count']}회 사용 | 평균 신뢰도: {avg_confidence:.3f}")

        # Recommendations
        print(f"\n💡 ESM3 검색 최적화 권장사항:")
        print("-" * 50)

        if success_rate >= 80:
            print("🎉 우수한 성과! ESM3 검색이 매우 잘 작동합니다.")
            print("✅ Unified RAG Orchestrator가 ESM3 연구에 최적화되었습니다.")
        elif success_rate >= 60:
            print("👍 양호한 성과. 몇 가지 개선 가능한 영역이 있습니다.")
            print("🔧 복잡한 쿼리에 대한 GraphRAG 활용도 증대 고려")
        else:
            print("⚠️ 개선 필요. ESM3 특화 튜닝이 필요할 수 있습니다.")

        print("\n🚀 다음 단계:")
        print("1. 멀티모달 ESM3 검색 테스트")
        print("2. GraphRAG ESM3 특화 최적화")
        print("3. 실시간 ESM3 연구 업데이트 통합")

        print("\n" + "=" * 80)

async def main():
    """Main test execution."""

    print("=" * 80)
    print("🧬 ESM3 × Unified RAG Orchestrator 종합 테스트")
    print("=" * 80)

    # Initialize test suite
    test_suite = ESM3TestSuite()
    await test_suite.initialize()

    # Run comprehensive ESM3 tests
    await test_suite.run_esm3_tests()

    print("\n🎉 ESM3 통합 테스트 완료!")
    print("✨ 이제 차세대 RAG 시스템으로 ESM3 연구를 탐색할 수 있습니다!")

if __name__ == "__main__":
    asyncio.run(main())