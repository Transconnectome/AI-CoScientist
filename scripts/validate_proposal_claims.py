#!/usr/bin/env python3
"""
Real-time Proposal Claim Validation System
==========================================

제안서 작성 중 실시간으로 과학적 주장을 검증하는 시스템

Features:
- Real-time claim validation during writing
- Interactive claim fixing with suggestions
- Automatic citation generation
- Evidence strength monitoring

Usage:
    # Validate entire proposal
    poetry run python scripts/validate_proposal_claims.py \
        --input "proposal.md" \
        --threshold 0.8

    # Interactive validation mode
    poetry run python scripts/validate_proposal_claims.py \
        --interactive \
        --input "proposal.md"
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from dataclasses import dataclass
import sys
from datetime import datetime

@dataclass
class ValidationIssue:
    """Validation issue found in proposal"""
    line_number: int
    claim_text: str
    issue_type: str  # "weak_evidence", "unsupported", "contradiction", "unclear"
    severity: str  # "critical", "major", "minor"
    evidence_strength: float
    suggestion: str
    auto_fix: Optional[str] = None

class RealTimeValidator:
    """Real-time claim validation engine"""

    def __init__(self, db_path: str = "chromadb_data_dd", threshold: float = 0.7):
        self.db_path = db_path
        self.threshold = threshold

        print("🔧 Initializing Real-time Validator...")
        print("   Loading SciBERT model...")
        self.embedding_model = SentenceTransformer('allenai/scibert_scivocab_uncased')
        print("   Loading Cross-Encoder...")
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        print("   Connecting to DD-RAPTOR...")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection("dd_papers_L0")

        # Validation patterns
        self.critical_patterns = [
            r'세계\s*최초',
            r'혁신적',
            r'획기적',
            r'\d+%\s*향상',
            r'AUC\s*[>]\s*0\.\d+',
            r'민감도\s*[>]\s*\d+%'
        ]

        self.citation_patterns = [
            r'\[[\d\w\-,\s]+\]',  # [1,2,3]
            r'\([\w\s]+et\s+al\.?,?\s*\d{4}\)',  # (Author et al., 2023)
            r'(Nature|Science|Cell|ICLR|NeurIPS|ICML)\s*\d{4}'
        ]

        print("✅ Validator ready!\n")

    def validate_text(self, text: str, context: str = "") -> List[ValidationIssue]:
        """Validate text and return issues"""
        issues = []
        lines = text.split('\n')

        for line_num, line in enumerate(lines, 1):
            line_issues = self._validate_line(line, line_num, context)
            issues.extend(line_issues)

        return issues

    def _validate_line(self, line: str, line_num: int, context: str) -> List[ValidationIssue]:
        """Validate a single line"""
        issues = []

        # Skip headers and empty lines
        if line.strip().startswith('#') or not line.strip():
            return issues

        # Check for critical claims without citations
        for pattern in self.critical_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                if not self._has_citation(line):
                    # Validate claim against DD-RAPTOR
                    evidence_strength = self._check_evidence(line)

                    if evidence_strength < self.threshold:
                        issue = ValidationIssue(
                            line_number=line_num,
                            claim_text=line.strip(),
                            issue_type="weak_evidence" if evidence_strength > 0.3 else "unsupported",
                            severity="critical" if evidence_strength < 0.3 else "major",
                            evidence_strength=evidence_strength,
                            suggestion=self._generate_suggestion(line, evidence_strength),
                            auto_fix=self._generate_auto_fix(line, evidence_strength)
                        )
                        issues.append(issue)

        # Check for unsupported technical claims
        if self._is_technical_claim(line) and not self._has_citation(line):
            evidence_strength = self._check_evidence(line)

            if evidence_strength < 0.5:
                issue = ValidationIssue(
                    line_number=line_num,
                    claim_text=line.strip(),
                    issue_type="unsupported",
                    severity="major",
                    evidence_strength=evidence_strength,
                    suggestion=f"Add citation or evidence for technical claim",
                    auto_fix=self._suggest_citation(line)
                )
                issues.append(issue)

        return issues

    def _has_citation(self, line: str) -> bool:
        """Check if line has proper citation"""
        for pattern in self.citation_patterns:
            if re.search(pattern, line):
                return True
        return False

    def _is_technical_claim(self, line: str) -> bool:
        """Check if line contains technical claim"""
        technical_keywords = [
            'foundation model', 'transformer', 'deep learning',
            'neural network', 'algorithm', 'architecture',
            '딥러닝', '신경망', '알고리즘', '모델', '아키텍처'
        ]

        line_lower = line.lower()
        return any(keyword in line_lower for keyword in technical_keywords)

    def _check_evidence(self, claim: str) -> float:
        """Check evidence strength for claim in DD-RAPTOR"""
        try:
            # Encode claim
            query_embedding = self.embedding_model.encode([claim])[0].tolist()

            # Search DD-RAPTOR
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=5,
                include=["documents", "metadatas"]
            )

            if not results['documents'][0]:
                return 0.0

            # Re-rank with cross-encoder
            documents = results['documents'][0]
            pairs = [[claim, doc] for doc in documents]
            scores = self.cross_encoder.predict(pairs)

            # Return best score
            return float(max(scores)) if scores else 0.0

        except Exception as e:
            print(f"   Warning: Evidence check failed for '{claim[:50]}...': {e}")
            return 0.0

    def _generate_suggestion(self, claim: str, evidence_strength: float) -> str:
        """Generate improvement suggestion"""
        if evidence_strength < 0.2:
            return f"Consider removing unsupported claim or find strong evidence"
        elif evidence_strength < 0.5:
            return f"Weaken claim language or add qualifying statements"
        else:
            return f"Add citation to strengthen claim (evidence available)"

    def _generate_auto_fix(self, claim: str, evidence_strength: float) -> Optional[str]:
        """Generate automatic fix suggestion"""
        if evidence_strength < 0.2:
            # Suggest removal or qualification
            return claim.replace('혁신적', '효과적').replace('세계 최초', '새로운')
        elif evidence_strength < 0.5:
            # Add qualifying language
            if '달성' in claim:
                return claim.replace('달성', '달성 예상')
            elif '%' in claim:
                return claim.replace('%', '% 목표')
        return None

    def _suggest_citation(self, claim: str) -> str:
        """Suggest citation for claim"""
        # Find relevant papers from DD-RAPTOR
        try:
            query_embedding = self.embedding_model.encode([claim])[0].tolist()
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=1,
                include=["metadatas"]
            )

            if results['metadatas'][0]:
                paper_title = results['metadatas'][0][0].get('paper_title', 'Unknown')
                return f"{claim} [{paper_title}]"

        except Exception:
            pass

        return f"{claim} [Citation needed]"

class InteractiveValidator:
    """Interactive validation interface"""

    def __init__(self, validator: RealTimeValidator):
        self.validator = validator

    def run_interactive_session(self, input_file: str):
        """Run interactive validation session"""
        print("🔍 INTERACTIVE PROPOSAL VALIDATION")
        print("=" * 50)
        print("Commands:")
        print("  'fix' - Apply suggested fix")
        print("  'skip' - Skip this issue")
        print("  'quit' - Exit session")
        print("=" * 50)

        # Read proposal
        with open(input_file, 'r', encoding='utf-8') as f:
            original_text = f.read()

        # Validate
        issues = self.validator.validate_text(original_text)

        if not issues:
            print("✅ No validation issues found!")
            return

        # Sort by severity and evidence strength
        issues.sort(key=lambda x: (
            0 if x.severity == 'critical' else 1 if x.severity == 'major' else 2,
            x.evidence_strength
        ))

        modified_text = original_text
        lines = original_text.split('\n')

        for i, issue in enumerate(issues):
            print(f"\n{'='*50}")
            print(f"Issue {i+1}/{len(issues)} - {issue.severity.upper()}")
            print(f"{'='*50}")
            print(f"📍 Line {issue.line_number}: {issue.issue_type}")
            print(f"🎯 Evidence Strength: {issue.evidence_strength:.3f}")
            print(f"\n📝 Original:")
            print(f"   {issue.claim_text}")
            print(f"\n💡 Issue: {issue.suggestion}")

            if issue.auto_fix:
                print(f"\n🔧 Suggested Fix:")
                print(f"   {issue.auto_fix}")

            # User input
            while True:
                choice = input(f"\nAction [fix/skip/quit]: ").strip().lower()

                if choice == 'quit':
                    print("👋 Exiting interactive session")
                    return
                elif choice == 'skip':
                    break
                elif choice == 'fix' and issue.auto_fix:
                    # Apply fix
                    lines[issue.line_number - 1] = issue.auto_fix
                    modified_text = '\n'.join(lines)
                    print("✅ Fix applied!")
                    break
                elif choice == 'fix':
                    print("❌ No automatic fix available")
                else:
                    print("Invalid choice. Use 'fix', 'skip', or 'quit'")

        # Save modified proposal
        if modified_text != original_text:
            backup_file = input_file.replace('.md', '_backup.md')
            with open(backup_file, 'w', encoding='utf-8') as f:
                f.write(original_text)

            with open(input_file, 'w', encoding='utf-8') as f:
                f.write(modified_text)

            print(f"\n💾 Original backed up to: {backup_file}")
            print(f"💾 Modified proposal saved to: {input_file}")

class ValidationReporter:
    """Generate validation reports"""

    def __init__(self):
        pass

    def generate_report(self, issues: List[ValidationIssue],
                       total_lines: int, filename: str) -> Dict[str, Any]:
        """Generate comprehensive validation report"""

        # Statistics
        critical_issues = [i for i in issues if i.severity == 'critical']
        major_issues = [i for i in issues if i.severity == 'major']
        minor_issues = [i for i in issues if i.severity == 'minor']

        avg_evidence_strength = sum(i.evidence_strength for i in issues) / len(issues) if issues else 1.0

        # Calculate validation score
        validation_score = self._calculate_validation_score(issues, total_lines)

        # Group by issue type
        issue_types = {}
        for issue in issues:
            issue_types[issue.issue_type] = issue_types.get(issue.issue_type, 0) + 1

        # Priority fixes (worst issues first)
        priority_fixes = sorted(issues, key=lambda x: (
            0 if x.severity == 'critical' else 1 if x.severity == 'major' else 2,
            x.evidence_strength
        ))[:10]

        report = {
            "metadata": {
                "file": filename,
                "timestamp": datetime.now().isoformat(),
                "total_lines": total_lines,
                "total_issues": len(issues)
            },

            "summary": {
                "validation_score": round(validation_score, 1),
                "average_evidence_strength": round(avg_evidence_strength, 3),
                "issues_by_severity": {
                    "critical": len(critical_issues),
                    "major": len(major_issues),
                    "minor": len(minor_issues)
                },
                "issues_by_type": issue_types
            },

            "priority_fixes": [
                {
                    "line": issue.line_number,
                    "severity": issue.severity,
                    "type": issue.issue_type,
                    "claim": issue.claim_text[:100] + "..." if len(issue.claim_text) > 100 else issue.claim_text,
                    "evidence_strength": round(issue.evidence_strength, 3),
                    "suggestion": issue.suggestion,
                    "auto_fix_available": issue.auto_fix is not None
                }
                for issue in priority_fixes
            ],

            "recommendations": self._generate_recommendations(issues, validation_score)
        }

        return report

    def _calculate_validation_score(self, issues: List[ValidationIssue], total_lines: int) -> float:
        """Calculate validation score (0-100)"""
        if not issues:
            return 100.0

        # Penalty per issue type
        severity_penalties = {
            'critical': 10.0,
            'major': 5.0,
            'minor': 2.0
        }

        total_penalty = sum(severity_penalties.get(issue.severity, 0) for issue in issues)

        # Base score minus penalties, with line count normalization
        base_score = 100.0
        normalized_penalty = total_penalty * (total_lines / 100)  # Normalize by document length

        score = max(0.0, base_score - normalized_penalty)

        return score

    def _generate_recommendations(self, issues: List[ValidationIssue],
                                validation_score: float) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Score-based recommendations
        if validation_score < 60:
            recommendations.append("🔴 Critical: Multiple serious validation issues. Major revision needed.")
        elif validation_score < 80:
            recommendations.append("🟡 Moderate: Several issues found. Address priority fixes.")
        else:
            recommendations.append("✅ Good: Minor issues only. Quick fixes recommended.")

        # Issue-specific recommendations
        critical_count = len([i for i in issues if i.severity == 'critical'])
        if critical_count > 0:
            recommendations.append(f"⚡ Fix {critical_count} critical claims lacking evidence")

        unsupported_count = len([i for i in issues if i.issue_type == 'unsupported'])
        if unsupported_count > 3:
            recommendations.append(f"📚 Add citations for {unsupported_count} unsupported claims")

        # DD-RAPTOR usage recommendation
        recommendations.append("🔍 Use DD-RAPTOR database for evidence-based claims")

        return recommendations

def main():
    parser = argparse.ArgumentParser(
        description="Validate proposal claims in real-time"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input proposal file"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Evidence strength threshold (0.0-1.0)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive validation mode"
    )
    parser.add_argument(
        "--output",
        help="Output report file (JSON)"
    )
    parser.add_argument(
        "--db-path",
        default="chromadb_data_dd",
        help="Path to DD-RAPTOR ChromaDB"
    )

    args = parser.parse_args()

    # Check files
    if not Path(args.input).exists():
        print(f"❌ Input file not found: {args.input}")
        return

    if not Path(args.db_path).exists():
        print(f"❌ DD-RAPTOR database not found: {args.db_path}")
        print("Run: poetry run python scripts/load_json_to_chromadb_dd.py")
        return

    try:
        # Initialize validator
        validator = RealTimeValidator(args.db_path, args.threshold)

        # Read input file
        with open(args.input, 'r', encoding='utf-8') as f:
            proposal_text = f.read()

        total_lines = len(proposal_text.split('\n'))

        if args.interactive:
            # Interactive mode
            interactive = InteractiveValidator(validator)
            interactive.run_interactive_session(args.input)
        else:
            # Batch validation mode
            print("🔍 REAL-TIME PROPOSAL VALIDATION")
            print("=" * 50)
            print(f"📄 File: {args.input}")
            print(f"🎯 Threshold: {args.threshold}")
            print("=" * 50)

            # Validate
            issues = validator.validate_text(proposal_text)

            # Generate report
            reporter = ValidationReporter()
            report = reporter.generate_report(issues, total_lines, args.input)

            # Print summary
            print(f"\n📊 VALIDATION SUMMARY")
            print("=" * 50)
            print(f"📝 Total Lines: {total_lines}")
            print(f"❗ Issues Found: {len(issues)}")
            print(f"⭐ Validation Score: {report['summary']['validation_score']}/100")

            severity_counts = report['summary']['issues_by_severity']
            print(f"\n🚨 Issues by Severity:")
            print(f"   🔴 Critical: {severity_counts['critical']}")
            print(f"   🟡 Major: {severity_counts['major']}")
            print(f"   🟢 Minor: {severity_counts['minor']}")

            # Show priority fixes
            if report['priority_fixes']:
                print(f"\n🎯 TOP PRIORITY FIXES:")
                for i, fix in enumerate(report['priority_fixes'][:5], 1):
                    print(f"   {i}. Line {fix['line']}: {fix['severity']} - {fix['suggestion']}")

            # Recommendations
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in report['recommendations']:
                print(f"   • {rec}")

            # Save report
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                print(f"\n💾 Report saved: {args.output}")

            print("=" * 50)

    except Exception as e:
        print(f"❌ Validation error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()