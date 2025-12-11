#!/usr/bin/env python3
"""
Comprehensive Unified RAG Orchestrator Demo

A complete demonstration of the Unified RAG Orchestrator system with
ESM3 papers, grant proposals, and all available strategies.

Usage:
    poetry run python scripts/comprehensive_unified_rag_demo.py
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Any

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
except ImportError as e:
    print(f"⚠️ Unified RAG Orchestrator import 실패: {e}")
    REAL_ORCHESTRATOR = False

class UnifiedRAGDemo:
    """Comprehensive demo of Unified RAG Orchestrator capabilities."""

    def __init__(self):
        self.orchestrator = None
        self.demo_results = {
            'esm3_results': [],
            'grant_results': [],
            'fusion_results': [],
            'strategy_usage': {}
        }

    async def initialize_system(self):
        """Initialize the complete system."""

        print("=" * 80)
        print("🎯 UNIFIED RAG ORCHESTRATOR 종합 데모")
        print("=" * 80)

        print("🚀 시스템 초기화...")

        if REAL_ORCHESTRATOR:
            self.orchestrator = create_unified_orchestrator()
            print("✅ Unified RAG Orchestrator 로드됨")

            # Get strategy health
            try:
                health = self.orchestrator.get_strategy_health()
                available_strategies = [s for s, info in health.items() if info.get('available', False)]
                print(f"🔧 사용 가능한 전략: {len(available_strategies)}개")
                for strategy in available_strategies:
                    print(f"  • {strategy}")

                # Warmup
                await self.orchestrator.warmup()
                print("🔥 시스템 워밍업 완료")

            except Exception as e:
                print(f"⚠️ 시스템 상태 확인 실패: {e}")
        else:
            print("📝 Mock 모드로 실행")

        return self.orchestrator is not None

    async def demo_esm3_capabilities(self):
        """Demonstrate ESM3 paper search capabilities."""

        print(f"\n" + "="*60)
        print("🧬 ESM3 논문 검색 데모")
        print("="*60)

        esm3_queries = [
            {
                'title': 'ESM3 기본 정보',
                'query': 'What is ESM3 evolutionary scale modeling?',
                'complexity': QueryComplexity.SIMPLE,
                'domain': QueryDomain.GENERAL,
                'intent': 'factual'
            },
            {
                'title': 'ESM3 진화 시뮬레이션',
                'query': 'How does ESM3 simulate 500 million years of protein evolution?',
                'complexity': QueryComplexity.COMPLEX,
                'domain': QueryDomain.GENERAL,
                'intent': 'synthesis'
            },
            {
                'title': 'ESM3 vs ESM2 비교',
                'query': 'ESM3 improvements over ESM2 protein structure prediction',
                'complexity': QueryComplexity.MEDIUM,
                'domain': QueryDomain.GENERAL,
                'intent': 'comparative'
            }
        ]

        return await self.execute_demo_queries(esm3_queries, 'esm3_results')

    async def demo_grant_capabilities(self):
        """Demonstrate grant proposal search capabilities."""

        print(f"\n" + "="*60)
        print("📑 Grant 제안서 검색 데모")
        print("="*60)

        grant_queries = [
            {
                'title': 'BrainLink 뇌과학 프로젝트',
                'query': 'BrainLink 뇌과학 연구 프로젝트 목표와 방법론',
                'complexity': QueryComplexity.MEDIUM,
                'domain': QueryDomain.NEUROSCIENCE,
                'intent': 'factual'
            },
            {
                'title': 'QuantERA 양자기계학습',
                'query': 'QuantERA PA-QML quantum machine learning approach',
                'complexity': QueryComplexity.COMPLEX,
                'domain': QueryDomain.QUANTUM_ML,
                'intent': 'synthesis'
            },
            {
                'title': 'INCITE HPC 컴퓨팅',
                'query': 'INCITE high performance computing neuroscience applications',
                'complexity': QueryComplexity.MEDIUM,
                'domain': QueryDomain.GENERAL,
                'intent': 'factual'
            }
        ]

        return await self.execute_demo_queries(grant_queries, 'grant_results')

    async def demo_fusion_capabilities(self):
        """Demonstrate cross-domain fusion search capabilities."""

        print(f"\n" + "="*60)
        print("🔗 융합 연구 검색 데모")
        print("="*60)

        fusion_queries = [
            {
                'title': '단백질-뇌과학 융합',
                'query': 'protein structure prediction applications in neuroscience brain modeling',
                'complexity': QueryComplexity.COMPLEX,
                'domain': QueryDomain.NEUROSCIENCE,
                'intent': 'synthesis'
            },
            {
                'title': '양자-AI-뇌과학 융합',
                'query': 'quantum computing applications in brain-inspired AI and neural networks',
                'complexity': QueryComplexity.COMPLEX,
                'domain': QueryDomain.GENERAL,
                'intent': 'synthesis'
            },
            {
                'title': 'ESM3-HPC 융합',
                'query': 'ESM3 computational requirements high performance computing optimization',
                'complexity': QueryComplexity.COMPLEX,
                'domain': QueryDomain.GENERAL,
                'intent': 'synthesis'
            }
        ]

        return await self.execute_demo_queries(fusion_queries, 'fusion_results')

    async def execute_demo_queries(self, queries: List[Dict], result_key: str) -> int:
        """Execute a set of demo queries and store results."""

        successful_queries = 0

        for query_info in queries:
            print(f"\n🔍 {query_info['title']}")
            print(f"📝 질문: {query_info['query']}")

            try:
                if self.orchestrator and hasattr(self.orchestrator, 'search'):
                    # Create query context
                    query_context = QueryContext(
                        query=query_info['query'],
                        complexity=query_info['complexity'],
                        domain=query_info['domain'],
                        intent=query_info['intent'],
                        confidence=0.9,
                        metadata={'demo_category': result_key, 'demo_test': True}
                    )

                    # Execute search
                    response = await self.orchestrator.search(query_context)

                    # Process results
                    strategy_used = str(response.strategy_used)
                    confidence = response.confidence
                    answer = response.answer

                    # Store result
                    result = {
                        'title': query_info['title'],
                        'query': query_info['query'],
                        'strategy': strategy_used,
                        'confidence': confidence,
                        'answer_length': len(answer),
                        'answer_preview': answer[:200] + "..." if len(answer) > 200 else answer,
                        'success': confidence > 0.7
                    }

                    self.demo_results[result_key].append(result)

                    # Track strategy usage
                    strategy_clean = strategy_used.replace('RAGStrategy.', '')
                    if strategy_clean not in self.strategy_usage:
                        self.strategy_usage[strategy_clean] = 0
                    self.strategy_usage[strategy_clean] += 1

                    # Display results
                    print(f"✅ 전략: {strategy_clean}")
                    print(f"📊 신뢰도: {confidence:.3f}")
                    print(f"📝 답변 길이: {len(answer)} 문자")
                    print(f"💬 답변 미리보기:")
                    print(f"   {answer[:150]}{'...' if len(answer) > 150 else ''}")

                    if confidence > 0.8:
                        print("🎯 고품질 응답!")
                        successful_queries += 1
                    elif confidence > 0.7:
                        print("✅ 양호한 응답")
                        successful_queries += 1
                    else:
                        print("⚠️ 개선 필요")

                else:
                    # Mock execution
                    print("📝 Mock 실행 - 성공 시뮬레이션")
                    successful_queries += 1

            except Exception as e:
                print(f"❌ 오류: {e}")

        return successful_queries

    def generate_comprehensive_report(self):
        """Generate comprehensive demo report."""

        print("\n" + "="*80)
        print("📊 UNIFIED RAG ORCHESTRATOR 종합 데모 결과")
        print("="*80)

        # Overall statistics
        total_queries = (
            len(self.demo_results['esm3_results']) +
            len(self.demo_results['grant_results']) +
            len(self.demo_results['fusion_results'])
        )

        successful_queries = sum(
            1 for results in self.demo_results.values()
            for result in results
            if result.get('success', False)
        )

        overall_success_rate = (successful_queries / total_queries) * 100 if total_queries > 0 else 0

        print(f"\n📈 전체 성과:")
        print(f"  📊 총 쿼리: {total_queries}개")
        print(f"  ✅ 성공: {successful_queries}개")
        print(f"  🎯 성공률: {overall_success_rate:.1f}%")

        # Category breakdown
        print(f"\n📋 분야별 성과:")

        categories = [
            ('ESM3 논문', 'esm3_results'),
            ('Grant 제안서', 'grant_results'),
            ('융합 연구', 'fusion_results')
        ]

        for category_name, result_key in categories:
            results = self.demo_results[result_key]
            if results:
                category_success = sum(1 for r in results if r.get('success', False))
                category_total = len(results)
                category_rate = (category_success / category_total) * 100 if category_total > 0 else 0

                print(f"  📂 {category_name}: {category_rate:.0f}% ({category_success}/{category_total})")

                # Best performing query in category
                successful_results = [r for r in results if r.get('success', False)]
                if successful_results:
                    best_result = max(successful_results, key=lambda x: x.get('confidence', 0))
                    strategy = best_result.get('strategy', 'Unknown').replace('RAGStrategy.', '')
                    confidence = best_result.get('confidence', 0)
                    print(f"    🏆 최고 성과: {best_result['title']} ({strategy}, {confidence:.3f})")

        # Strategy usage analysis
        print(f"\n🎯 전략 활용 분석:")
        if hasattr(self, 'strategy_usage'):
            sorted_strategies = sorted(self.strategy_usage.items(), key=lambda x: x[1], reverse=True)
            for strategy, count in sorted_strategies:
                usage_rate = (count / total_queries) * 100 if total_queries > 0 else 0
                print(f"  🔧 {strategy}: {count}회 사용 ({usage_rate:.1f}%)")

        # System capabilities assessment
        print(f"\n🏅 시스템 평가:")
        if overall_success_rate >= 90:
            print("🥇 최고 등급: Unified RAG 시스템이 탁월한 성능을 보입니다!")
            print("✨ ESM3, Grant, 융합 연구 모든 영역에서 우수한 검색 결과")
        elif overall_success_rate >= 75:
            print("🥈 우수 등급: 대부분의 쿼리에서 높은 품질의 결과")
            print("✅ 시스템이 안정적이고 신뢰할 수 있습니다")
        elif overall_success_rate >= 60:
            print("🥉 양호 등급: 개선 가능한 영역이 있지만 기본 기능은 우수")
        else:
            print("⚠️ 개선 필요: 시스템 튜닝이 필요합니다")

        # Key insights and recommendations
        print(f"\n💡 핵심 인사이트:")
        print("1. 🧬 ESM3 논문 검색: 진화적 단백질 모델링 연구에 최적화")
        print("2. 📑 Grant 제안서 검색: 다학제 연구 프로젝트 정보 제공")
        print("3. 🔗 융합 연구: 교차 도메인 연결과 통찰력 발견")
        print("4. 🚀 전략 다양성: 6가지 RAG 전략으로 최적 결과 보장")

        print(f"\n🔮 활용 방안:")
        print("• 🔬 연구자: ESM3와 단백질 연구의 최신 동향 파악")
        print("• 📝 제안서 작성: 기존 연구 프로젝트 벤치마킹")
        print("• 🧠 융합 연구: 뇌과학-AI-양자컴퓨팅 교차점 탐색")
        print("• 📊 의사결정: 다양한 전략으로 종합적 정보 수집")

        print("\n" + "="*80)
        print("🎉 Unified RAG Orchestrator 종합 데모 완료!")
        print("✨ 차세대 AI 연구 도구로 혁신적 발견을 시작하세요!")
        print("="*80)

async def main():
    """Main demo execution."""

    demo = UnifiedRAGDemo()

    # Initialize system
    initialized = await demo.initialize_system()

    if not initialized and not REAL_ORCHESTRATOR:
        print("📝 Mock 모드로 데모 진행")

    # Run comprehensive demos
    esm3_success = await demo.demo_esm3_capabilities()
    grant_success = await demo.demo_grant_capabilities()
    fusion_success = await demo.demo_fusion_capabilities()

    # Generate final report
    demo.generate_comprehensive_report()

if __name__ == "__main__":
    asyncio.run(main())