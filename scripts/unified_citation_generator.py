#!/usr/bin/env python3
"""
Unified Citation Generator
==========================

automated_citation_generator.py에 대한 통합 인터페이스 래퍼

실제 구현: scripts/automated_citation_generator.py

Usage:
    poetry run python scripts/unified_citation_generator.py \
        --input "proposal.md" \
        --output "cited_proposal.md" \
        --cross-domain-refs \
        --format "NRF"
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    parser = argparse.ArgumentParser(
        description="Unified Citation Generator (wrapper for automated_citation_generator.py)"
    )
    parser.add_argument("--input", "-i", required=True, help="Input proposal file")
    parser.add_argument("--output", "-o", help="Output file with citations")
    parser.add_argument("--cross-domain-refs", action="store_true",
                       help="Enable cross-domain reference generation")
    parser.add_argument("--format", choices=["NRF", "APA", "IEEE", "Vancouver"],
                       default="NRF", help="Citation format")
    parser.add_argument("--mode", choices=["auto_cite", "interactive", "generate_references"],
                       default="auto_cite", help="Citation mode")

    args = parser.parse_args()

    print("=" * 60)
    print("Unified Citation Generator")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Format: {args.format}")
    print(f"Mode: {args.mode}")
    print(f"Cross-domain: {args.cross_domain_refs}")
    print()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    try:
        from scripts.automated_citation_generator import AutomatedCitationGenerator

        generator = AutomatedCitationGenerator()

        proposal_text = input_path.read_text(encoding='utf-8')

        print("Generating citations...")

        if args.mode == "auto_cite":
            cited_text, citations = generator.auto_cite(proposal_text)
        elif args.mode == "generate_references":
            cited_text = proposal_text
            citations = generator.generate_references(proposal_text)
        else:
            cited_text, citations = generator.interactive_cite(proposal_text)

        print("-" * 40)
        print(f"Generated {len(citations)} citations")
        print("-" * 40)

        # Show sample citations
        for i, citation in enumerate(citations[:5], 1):
            print(f"{i}. {citation.paper_title}")
            print(f"   -> {citation.formatted_citation[:80]}...")

        if args.output:
            output_path = Path(args.output)

            # Add references section
            output_text = cited_text
            if citations:
                output_text += "\n\n---\n\n## 참고문헌\n\n"
                for i, citation in enumerate(citations, 1):
                    output_text += f"{i}. {citation.formatted_citation}\n"

            output_path.write_text(output_text, encoding='utf-8')
            print(f"\nOutput saved to: {args.output}")

    except ImportError as e:
        print(f"Note: Full citation requires ChromaDB setup. Error: {e}")
        print("Running basic citation generation...")

        # Basic mode without ChromaDB
        proposal_text = input_path.read_text(encoding='utf-8')
        if args.output:
            Path(args.output).write_text(proposal_text + "\n\n[Citations to be added]\n", encoding='utf-8')
            print(f"Basic output saved to: {args.output}")

if __name__ == "__main__":
    main()
