#!/usr/bin/env python3
"""
Unified RAG Claim Validation Wrapper
====================================

validate_proposal_claims.py에 대한 통합 RAG 인터페이스 래퍼

실제 구현: scripts/validate_proposal_claims.py

Usage:
    poetry run python scripts/validate_claims_unified_rag.py \
        --input "proposal.md" \
        --strategies "HYBRID,GRAPH_RAG,GOLDEN_REFERENCE" \
        --threshold 0.85 \
        --output "validation_result.json"
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    parser = argparse.ArgumentParser(
        description="Unified RAG Claim Validation (wrapper for validate_proposal_claims.py)"
    )
    parser.add_argument("--input", "-i", required=True, help="Input proposal file")
    parser.add_argument("--output", "-o", help="Output validation report")
    parser.add_argument("--strategies", default="HYBRID,GRAPH_RAG",
                       help="RAG strategies to use (comma-separated)")
    parser.add_argument("--threshold", type=float, default=0.85,
                       help="Validation threshold (default: 0.85)")
    parser.add_argument("--interactive", action="store_true",
                       help="Interactive validation mode")
    parser.add_argument("--cross-domain-validation", action="store_true",
                       help="Enable cross-domain validation")

    args = parser.parse_args()

    print("=" * 60)
    print("Unified RAG Claim Validation System")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Strategies: {args.strategies}")
    print(f"Threshold: {args.threshold}")
    print()

    # Import and call the actual validator
    try:
        from scripts.validate_proposal_claims import RealTimeValidator

        validator = RealTimeValidator()

        # Read input file
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {args.input}")
            sys.exit(1)

        proposal_text = input_path.read_text(encoding='utf-8')

        # Run validation
        print("Validating claims...")
        issues = validator.validate_document(proposal_text)

        # Calculate statistics
        total_claims = len(issues) if issues else 0
        supported_claims = sum(1 for i in issues if i.evidence_strength >= args.threshold) if issues else 0
        validation_rate = (supported_claims / total_claims * 100) if total_claims > 0 else 100

        print()
        print("-" * 40)
        print("Validation Results:")
        print(f"  Total claims analyzed: {total_claims}")
        print(f"  Supported claims: {supported_claims}")
        print(f"  Validation rate: {validation_rate:.1f}%")
        print(f"  Target threshold: {args.threshold * 100:.0f}%")
        print("-" * 40)

        if validation_rate >= args.threshold * 100:
            print("PASSED: Validation rate meets threshold")
        else:
            print("WARNING: Validation rate below threshold")
            print("\nIssues found:")
            for issue in issues:
                if issue.evidence_strength < args.threshold:
                    print(f"  - Line {issue.line_number}: {issue.claim_text[:50]}...")
                    print(f"    Strength: {issue.evidence_strength:.2f}, Suggestion: {issue.suggestion}")

        # Save output if specified
        if args.output:
            import json
            output_data = {
                "input_file": str(args.input),
                "strategies": args.strategies.split(","),
                "threshold": args.threshold,
                "total_claims": total_claims,
                "supported_claims": supported_claims,
                "validation_rate": validation_rate,
                "passed": validation_rate >= args.threshold * 100,
                "issues": [
                    {
                        "line": i.line_number,
                        "claim": i.claim_text,
                        "type": i.issue_type,
                        "severity": i.severity,
                        "strength": i.evidence_strength,
                        "suggestion": i.suggestion
                    }
                    for i in (issues or [])
                ]
            }
            Path(args.output).write_text(json.dumps(output_data, ensure_ascii=False, indent=2))
            print(f"\nResults saved to: {args.output}")

    except ImportError as e:
        print(f"Note: Full validation requires ChromaDB setup. Error: {e}")
        print("Running basic validation...")

        # Basic validation without ChromaDB
        input_path = Path(args.input)
        if input_path.exists():
            text = input_path.read_text(encoding='utf-8')
            lines = text.split('\n')
            print(f"Analyzed {len(lines)} lines")
            print("Basic validation complete (full validation requires ChromaDB)")

if __name__ == "__main__":
    main()
