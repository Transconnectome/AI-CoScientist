#!/usr/bin/env python3
"""
Advanced Unified RAG Query Tool
===============================

enhanced_dd_query.py에 대한 통합 RAG 인터페이스 래퍼

실제 구현: scripts/enhanced_dd_query.py

Usage:
    # 기본 검색
    poetry run python scripts/advanced_unified_query.py \
        --query "연구 주제의 핵심 미해결 문제" \
        --strategies "GRAPH_RAG,HYBRID"

    # Systematic Review 모드
    poetry run python scripts/advanced_unified_query.py \
        --input "proposal.md" \
        --mode systematic_review \
        --strategies "MULTIMODAL_RAG,GRAPH_RAG" \
        --output "literature_review.json"
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    parser = argparse.ArgumentParser(
        description="Advanced Unified RAG Query (wrapper for enhanced_dd_query.py)"
    )
    parser.add_argument("--query", "-q", help="Query string")
    parser.add_argument("--input", "-i", help="Input file for context")
    parser.add_argument("--output", "-o", help="Output file")
    parser.add_argument("--strategies", default="HYBRID,GRAPH_RAG",
                       help="RAG strategies (comma-separated)")
    parser.add_argument("--mode", choices=["search", "systematic_review", "gap_analysis"],
                       default="search", help="Query mode")
    parser.add_argument("--n_results", type=int, default=10,
                       help="Number of results")

    args = parser.parse_args()

    print("=" * 60)
    print("Advanced Unified RAG Query System")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Strategies: {args.strategies}")
    print()

    # Get query from input file if not provided directly
    query = args.query
    if not query and args.input:
        input_path = Path(args.input)
        if input_path.exists():
            query = input_path.read_text(encoding='utf-8')[:500]  # First 500 chars as context

    if not query:
        print("Error: Provide --query or --input")
        sys.exit(1)

    print(f"Query: {query[:100]}...")
    print()

    try:
        from scripts.enhanced_dd_query import EnhancedDDQuery

        querier = EnhancedDDQuery()

        if args.mode == "systematic_review":
            results = querier.systematic_review(query, n_results=args.n_results)
        else:
            results = querier.search(query, n_results=args.n_results)

        print("-" * 40)
        print(f"Found {len(results)} results")
        print("-" * 40)

        for i, result in enumerate(results[:5], 1):
            print(f"\n{i}. {result.paper_title}")
            print(f"   Section: {result.section}")
            print(f"   Score: {result.relevance_score:.3f}")
            print(f"   Snippet: {result.content[:100]}...")

        if args.output:
            import json
            from dataclasses import asdict
            output_data = {
                "query": query,
                "mode": args.mode,
                "strategies": args.strategies.split(","),
                "results": [asdict(r) for r in results]
            }
            Path(args.output).write_text(json.dumps(output_data, ensure_ascii=False, indent=2))
            print(f"\nResults saved to: {args.output}")

    except ImportError as e:
        print(f"Note: Full query requires ChromaDB setup. Error: {e}")
        print("Install dependencies: poetry install")

if __name__ == "__main__":
    main()
