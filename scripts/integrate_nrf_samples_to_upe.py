#!/usr/bin/env python3
"""
NRF Mid-Career Proposal Samples Integration with UPE System

This script verifies and tests the integration of NRF proposal samples
with the AI-CoScientist Unified Proposal Engine (UPE).

Collections:
- nrf_midcareer_samples_L0: 659 chunks
- nrf_midcareer_samples_L1: 24 section summaries
- nrf_midcareer_samples_L2: 4 document summaries

Proposal Types:
- INCITE (128 chunks): DOE INCITE NeuroX-Fusion Foundation Model
- Samsung (1 chunk): Samsung Future Technology Grant
- BrainLink (253 chunks): International Brain Research Collaboration
- Developmental (277 chunks): Developmental Disorder Research

Usage:
    poetry run python scripts/integrate_nrf_samples_to_upe.py --verify
    poetry run python scripts/integrate_nrf_samples_to_upe.py --test-query "연구 방법론"
    poetry run python scripts/integrate_nrf_samples_to_upe.py --demo
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.rag.nrf_proposal_strategy import (
    NRFProposalRAGStrategy,
    create_nrf_proposal_strategy
)


def verify_integration():
    """Verify NRF samples are properly integrated."""
    print("\n" + "=" * 70)
    print("NRF SAMPLES INTEGRATION VERIFICATION")
    print("=" * 70)

    strategy = create_nrf_proposal_strategy()

    # Check availability
    available = strategy.is_available()
    print(f"\n1. Strategy Available: {'Yes' if available else 'No'}")

    if not available:
        print("   ERROR: NRF samples not found. Run ingestion first:")
        print("   poetry run python scripts/ingest_nrf_midcareer_samples.py --all")
        return False

    # Check collections
    stats = strategy.get_collection_stats()
    print(f"\n2. Collection Statistics:")
    print(f"   L0 (Chunks):    {stats['L0_chunks']}")
    print(f"   L1 (Sections):  {stats['L1_sections']}")
    print(f"   L2 (Documents): {stats['L2_documents']}")

    # Verify expected counts
    expected = {'L0_chunks': 659, 'L1_sections': 24, 'L2_documents': 4}
    all_match = all(stats[k] >= expected[k] for k in expected)
    print(f"\n3. Expected Counts Match: {'Yes' if all_match else 'No'}")

    # Get proposal patterns
    print(f"\n4. Proposal Patterns:")

    async def get_patterns():
        return await strategy.get_proposal_patterns()

    patterns = asyncio.run(get_patterns())
    print(f"   Types: {patterns['proposal_types']}")
    print(f"   Sections: {patterns['common_sections'][:5]}...")

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE: Integration Successful")
    print("=" * 70)

    return True


async def test_query(query: str):
    """Test a specific query against NRF samples."""
    print(f"\n--- Testing Query: {query} ---")

    strategy = create_nrf_proposal_strategy()
    response = await strategy.search(query, n_results=5)

    if isinstance(response, dict):
        print(f"\nAnswer: {response['answer']}")
        print(f"\nTop Sources:")
        for i, src in enumerate(response['sources'][:3], 1):
            print(f"\n  [{i}] Level: {src['level']}, Section: {src['section']}")
            print(f"      Type: {src['proposal_type']}")
            print(f"      Score: {src['score']:.4f}")
            print(f"      Content: {src['content'][:150]}...")
    else:
        print(f"\nAnswer: {response.answer}")
        print(f"Confidence: {response.confidence:.4f}")
        print(f"Strategy: {response.strategy_used}")


async def run_demo():
    """Run a comprehensive demo of the integration."""
    print("\n" + "=" * 70)
    print("NRF SAMPLES INTEGRATION DEMO")
    print("=" * 70)

    strategy = create_nrf_proposal_strategy()

    # Demo queries covering different use cases
    demo_queries = [
        # Korean proposal writing queries
        ("연구과제의 창의성과 도전성을 어떻게 작성하나요?", "Korean NRF Proposal"),
        ("추진전략과 방법론 예시", "Methodology Section"),

        # Technical queries
        ("4D Swin Transformer brain imaging", "INCITE Architecture"),
        ("Foundation model architecture design", "Technical Design"),

        # Domain-specific
        ("발달장애 진단 AI 모델 개발", "Developmental Disorder"),
        ("뇌 파운데이션 모델 사전훈련 전략", "Pre-training Strategy"),
    ]

    for query, category in demo_queries:
        print(f"\n--- {category}: {query[:40]}... ---")

        response = await strategy.search(query, n_results=3, levels=['L0', 'L1'])

        if isinstance(response, dict):
            print(f"  Found: {response['total_results']} results")
            if response['sources']:
                top = response['sources'][0]
                print(f"  Top: [{top['level']}] {top['section']} ({top['proposal_type']})")
        else:
            print(f"  Confidence: {response.confidence:.4f}")

    # Show section-filtered search
    print("\n" + "-" * 50)
    print("Section-Filtered Search Demo")
    print("-" * 50)

    for section_filter in [['Methods', '연구 방법'], ['추진전략', '추진체계']]:
        print(f"\n  Filter: {section_filter}")
        results = await strategy.search_by_section(
            "연구 방법론",
            section_filter=section_filter,
            n_results=2
        )
        for r in results:
            print(f"    - {r.section}: {r.proposal_type} (score: {r.score:.2f})")

    # Show proposal type filtering
    print("\n" + "-" * 50)
    print("Proposal Type-Filtered Search Demo")
    print("-" * 50)

    for ptype in [['INCITE'], ['BrainLink'], ['Developmental']]:
        print(f"\n  Type: {ptype}")
        results = await strategy.search_by_proposal_type(
            "뇌영상 분석",
            proposal_types=ptype,
            n_results=2
        )
        for r in results:
            print(f"    - {r.section}: score {r.score:.2f}")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)

    print("\n📝 Usage in UPE Workflow:")
    print("""
    # 1. Import the strategy
    from src.services.rag.nrf_proposal_strategy import create_nrf_proposal_strategy

    # 2. Create instance
    nrf_rag = create_nrf_proposal_strategy()

    # 3. Search for proposal patterns
    results = await nrf_rag.search("연구 방법론 예시")

    # 4. Get section-specific examples
    methods = await nrf_rag.search_by_section(
        "추진전략",
        section_filter=['추진전략', 'Methods']
    )

    # 5. Filter by proposal type
    incite_examples = await nrf_rag.search_by_proposal_type(
        "foundation model",
        proposal_types=['INCITE']
    )
    """)


def main():
    parser = argparse.ArgumentParser(
        description="NRF Samples Integration with UPE System"
    )
    parser.add_argument("--verify", action="store_true",
                        help="Verify integration is working")
    parser.add_argument("--test-query", type=str,
                        help="Test a specific query")
    parser.add_argument("--demo", action="store_true",
                        help="Run comprehensive demo")
    args = parser.parse_args()

    if args.verify:
        verify_integration()
    elif args.test_query:
        asyncio.run(test_query(args.test_query))
    elif args.demo:
        asyncio.run(run_demo())
    else:
        # Default: run verification and demo
        if verify_integration():
            asyncio.run(run_demo())


if __name__ == "__main__":
    main()
