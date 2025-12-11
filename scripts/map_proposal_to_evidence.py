#!/usr/bin/env python3
"""
DD-RAPTOR Evidence Mapping Tool
===============================

제안서의 모든 주장을 DD-RAPTOR 논문 DB에서 검증하고
evidence mapping을 생성하는 도구

Usage:
    poetry run python scripts/map_proposal_to_evidence.py \
        --proposal "data/발달장애/과학적_엄밀성_기반_제안서_수정계획_FINAL_2025.md" \
        --output "evidence_mapping_report.json"
"""

import argparse
import json
import re
import chromadb
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

@dataclass
class Claim:
    """Scientific claim extracted from proposal"""
    text: str
    section: str
    confidence: float
    line_number: int
    claim_type: str  # "technical", "statistical", "literature", "innovation"

@dataclass
class Evidence:
    """Supporting evidence from DD-RAPTOR"""
    paper_title: str
    section: str
    text: str
    relevance_score: float
    citation_ready: str

@dataclass
class ValidationResult:
    """Claim validation result"""
    claim: Claim
    evidence_strength: float
    supporting_evidence: List[Evidence]
    validation_status: str  # "strong", "moderate", "weak", "unsupported"
    suggested_revision: Optional[str] = None

class ClaimExtractor:
    """Extract scientific claims from proposal text"""

    def __init__(self):
        # 과학적 주장을 나타내는 패턴들
        self.claim_patterns = [
            # Technical claims
            r'(\d+%\s*(?:향상|개선|증가|달성))',
            r'(세계\s*(?:최초|최고|선도))',
            r'(\d+(?:배|번째|개|명|례)\s*(?:이상|규모|달성))',

            # Statistical claims
            r'(AUC\s*[>]\s*0\.\d+)',
            r'(민감도\s*[>]\s*\d+%)',
            r'(정확도\s*\d+%)',

            # Innovation claims
            r'((?:혁신적|획기적|새로운|차별화된)\s*[\w\s]+)',
            r'(\w+\s*foundation\s*model)',

            # Literature claims
            r'(Nature|Science|Cell|ICLR|NeurIPS|ICML)(?:\s*\d{4})?',
            r'(\w+\s*et\s*al\.?,?\s*\d{4})'
        ]

        self.section_headers = [
            "연구의 필요성", "연구 목표", "기술적 접근", "혁신성",
            "실현가능성", "기대효과", "예산", "일정"
        ]

    def extract_claims(self, text: str) -> List[Claim]:
        """Extract scientific claims from text"""
        claims = []
        lines = text.split('\n')
        current_section = "unknown"

        for line_num, line in enumerate(lines):
            # Update current section
            for header in self.section_headers:
                if header in line and line.startswith('#'):
                    current_section = header
                    break

            # Extract claims from line
            line_claims = self._extract_claims_from_line(
                line, current_section, line_num
            )
            claims.extend(line_claims)

        return claims

    def _extract_claims_from_line(self, line: str, section: str, line_num: int) -> List[Claim]:
        """Extract claims from a single line"""
        claims = []

        # Skip markdown headers and empty lines
        if line.strip().startswith('#') or not line.strip():
            return claims

        # Check each pattern
        for pattern in self.claim_patterns:
            matches = re.findall(pattern, line, re.IGNORECASE)
            for match in matches:
                claim_text = match if isinstance(match, str) else match[0]

                claim = Claim(
                    text=claim_text,
                    section=section,
                    confidence=self._estimate_confidence(claim_text),
                    line_number=line_num + 1,
                    claim_type=self._classify_claim(claim_text)
                )
                claims.append(claim)

        # Extract high-confidence sentences
        sentences = self._split_into_sentences(line)
        for sentence in sentences:
            if self._is_scientific_claim(sentence):
                claim = Claim(
                    text=sentence,
                    section=section,
                    confidence=0.7,
                    line_number=line_num + 1,
                    claim_type="general"
                )
                claims.append(claim)

        return claims

    def _estimate_confidence(self, text: str) -> float:
        """Estimate confidence of claim"""
        confidence_indicators = {
            'high': ['세계 최초', '획기적', '혁신적', '99%', '95%'],
            'medium': ['효과적', '향상', '개선', '80%', '90%'],
            'low': ['가능성', '예상', '기대', '약간']
        }

        text_lower = text.lower()

        for indicator in confidence_indicators['high']:
            if indicator.lower() in text_lower:
                return 0.9

        for indicator in confidence_indicators['medium']:
            if indicator.lower() in text_lower:
                return 0.7

        return 0.5

    def _classify_claim(self, text: str) -> str:
        """Classify claim type"""
        if any(word in text.lower() for word in ['model', 'algorithm', 'architecture']):
            return "technical"
        elif any(char in text for char in ['%', 'AUC', '정확도', '민감도']):
            return "statistical"
        elif any(word in text for word in ['Nature', 'Science', 'NeurIPS', 'ICLR']):
            return "literature"
        elif any(word in text.lower() for word in ['혁신', '최초', '획기적']):
            return "innovation"
        else:
            return "general"

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _is_scientific_claim(self, sentence: str) -> bool:
        """Check if sentence contains scientific claim"""
        scientific_keywords = [
            '파운데이션 모델', 'foundation model', 'transformer',
            '뇌영상', '유전체', '발달장애', 'autism', 'ADHD',
            '딥러닝', 'deep learning', '인공지능', 'AI',
            '임상시험', 'clinical trial', '검증', 'validation'
        ]

        sentence_lower = sentence.lower()
        return any(keyword.lower() in sentence_lower for keyword in scientific_keywords)

class DDRAPTORValidator:
    """Validate claims against DD-RAPTOR database"""

    def __init__(self, db_path: str = "chromadb_data_dd"):
        self.db_path = db_path
        self.embedding_model = SentenceTransformer('allenai/scibert_scivocab_uncased')
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        # Connect to ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection("dd_papers_L0")

    def validate_claim(self, claim: Claim, n_results: int = 10) -> ValidationResult:
        """Validate a single claim against DD-RAPTOR"""

        # 1. Vector search for relevant evidence
        query_embedding = self.embedding_model.encode([claim.text])[0].tolist()

        search_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas"]
        )

        # 2. Re-rank with cross-encoder
        documents = search_results['documents'][0]
        metadatas = search_results['metadatas'][0]

        pairs = [[claim.text, doc] for doc in documents]
        relevance_scores = self.cross_encoder.predict(pairs)

        # 3. Create evidence objects
        evidence_list = []
        for doc, meta, score in zip(documents, metadatas, relevance_scores):
            evidence = Evidence(
                paper_title=meta.get('paper_title', 'Unknown'),
                section=meta.get('section', 'Unknown'),
                text=doc[:200] + "..." if len(doc) > 200 else doc,
                relevance_score=float(score),
                citation_ready=f"{meta.get('paper_title', 'Unknown')} ({meta.get('section', 'Unknown')})"
            )
            evidence_list.append(evidence)

        # 4. Sort by relevance score
        evidence_list.sort(key=lambda x: x.relevance_score, reverse=True)

        # 5. Determine validation status
        top_score = evidence_list[0].relevance_score if evidence_list else 0
        evidence_strength = self._calculate_evidence_strength(evidence_list)

        validation_status, suggested_revision = self._determine_status(
            claim, evidence_strength, top_score, evidence_list
        )

        return ValidationResult(
            claim=claim,
            evidence_strength=evidence_strength,
            supporting_evidence=evidence_list[:5],  # Top 5 evidence
            validation_status=validation_status,
            suggested_revision=suggested_revision
        )

    def _calculate_evidence_strength(self, evidence_list: List[Evidence]) -> float:
        """Calculate overall evidence strength"""
        if not evidence_list:
            return 0.0

        # Weighted average of top evidence scores
        weights = np.exp(np.linspace(0, -2, len(evidence_list[:5])))  # Exponential decay
        scores = [e.relevance_score for e in evidence_list[:5]]

        if len(scores) < len(weights):
            weights = weights[:len(scores)]

        weighted_score = np.average(scores, weights=weights)
        return float(weighted_score)

    def _determine_status(self, claim: Claim, evidence_strength: float,
                         top_score: float, evidence_list: List[Evidence]) -> Tuple[str, Optional[str]]:
        """Determine validation status and suggest revision if needed"""

        if evidence_strength >= 0.8:
            return "strong", None
        elif evidence_strength >= 0.6:
            return "moderate", None
        elif evidence_strength >= 0.4:
            return "weak", self._suggest_weak_revision(claim, evidence_list)
        else:
            return "unsupported", self._suggest_unsupported_revision(claim, evidence_list)

    def _suggest_weak_revision(self, claim: Claim, evidence_list: List[Evidence]) -> str:
        """Suggest revision for weak claims"""
        if evidence_list:
            top_evidence = evidence_list[0]
            return f"Consider revising based on evidence from {top_evidence.paper_title}: {top_evidence.text[:100]}..."
        return "Consider providing more specific evidence or qualifying the claim"

    def _suggest_unsupported_revision(self, claim: Claim, evidence_list: List[Evidence]) -> str:
        """Suggest revision for unsupported claims"""
        return f"Claim '{claim.text}' lacks supporting evidence. Consider removing or finding alternative support."

class ProposalEvidenceMapper:
    """Main class for mapping proposal to evidence"""

    def __init__(self, db_path: str = "chromadb_data_dd"):
        self.claim_extractor = ClaimExtractor()
        self.validator = DDRAPTORValidator(db_path)

    def map_proposal(self, proposal_file: str) -> Dict[str, Any]:
        """Map entire proposal to evidence"""

        print(f"📖 Reading proposal: {proposal_file}")
        with open(proposal_file, 'r', encoding='utf-8') as f:
            proposal_text = f.read()

        print("🔍 Extracting claims...")
        claims = self.claim_extractor.extract_claims(proposal_text)
        print(f"   Found {len(claims)} claims")

        print("🧪 Validating claims against DD-RAPTOR...")
        validation_results = []

        for claim in tqdm(claims, desc="Validating claims"):
            try:
                result = self.validator.validate_claim(claim)
                validation_results.append(result)
            except Exception as e:
                print(f"   Error validating claim '{claim.text[:50]}...': {e}")

        # Generate report
        report = self._generate_report(claims, validation_results)

        return report

    def _generate_report(self, claims: List[Claim],
                        validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate comprehensive evidence mapping report"""

        # Statistics
        total_claims = len(claims)
        strong_claims = len([r for r in validation_results if r.validation_status == "strong"])
        moderate_claims = len([r for r in validation_results if r.validation_status == "moderate"])
        weak_claims = len([r for r in validation_results if r.validation_status == "weak"])
        unsupported_claims = len([r for r in validation_results if r.validation_status == "unsupported"])

        # Evidence coverage
        evidence_coverage = (strong_claims + moderate_claims) / total_claims * 100 if total_claims > 0 else 0

        # Claims by type
        claim_types = {}
        for claim in claims:
            claim_types[claim.claim_type] = claim_types.get(claim.claim_type, 0) + 1

        # Section analysis
        section_analysis = {}
        for result in validation_results:
            section = result.claim.section
            if section not in section_analysis:
                section_analysis[section] = {
                    "total_claims": 0,
                    "strong": 0,
                    "moderate": 0,
                    "weak": 0,
                    "unsupported": 0
                }

            section_analysis[section]["total_claims"] += 1
            section_analysis[section][result.validation_status] += 1

        # Top issues
        weak_and_unsupported = [r for r in validation_results
                               if r.validation_status in ["weak", "unsupported"]]

        report = {
            "summary": {
                "total_claims": total_claims,
                "evidence_coverage_percent": round(evidence_coverage, 1),
                "validation_breakdown": {
                    "strong": strong_claims,
                    "moderate": moderate_claims,
                    "weak": weak_claims,
                    "unsupported": unsupported_claims
                },
                "scientific_rigor_score": round(self._calculate_rigor_score(validation_results), 1)
            },

            "claim_analysis": {
                "by_type": claim_types,
                "by_section": section_analysis
            },

            "priority_issues": [
                {
                    "claim": r.claim.text,
                    "section": r.claim.section,
                    "line": r.claim.line_number,
                    "status": r.validation_status,
                    "evidence_strength": round(r.evidence_strength, 3),
                    "suggested_revision": r.suggested_revision,
                    "supporting_evidence": [
                        {
                            "paper": e.paper_title,
                            "relevance": round(e.relevance_score, 3),
                            "text": e.text
                        }
                        for e in r.supporting_evidence[:2]  # Top 2 evidence
                    ]
                }
                for r in weak_and_unsupported[:10]  # Top 10 issues
            ],

            "strong_claims": [
                {
                    "claim": r.claim.text,
                    "section": r.claim.section,
                    "evidence_strength": round(r.evidence_strength, 3),
                    "top_evidence": r.supporting_evidence[0].citation_ready if r.supporting_evidence else None
                }
                for r in validation_results if r.validation_status == "strong"
            ][:10],  # Top 10 strong claims

            "recommendations": self._generate_recommendations(validation_results, evidence_coverage)
        }

        return report

    def _calculate_rigor_score(self, validation_results: List[ValidationResult]) -> float:
        """Calculate scientific rigor score (0-100)"""
        if not validation_results:
            return 0

        status_weights = {
            "strong": 1.0,
            "moderate": 0.7,
            "weak": 0.3,
            "unsupported": 0.0
        }

        total_weight = sum(status_weights[r.validation_status] for r in validation_results)
        max_possible = len(validation_results) * 1.0

        return (total_weight / max_possible) * 100

    def _generate_recommendations(self, validation_results: List[ValidationResult],
                                evidence_coverage: float) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Coverage recommendations
        if evidence_coverage < 70:
            recommendations.append("🔴 Critical: Evidence coverage below 70%. Add citations and supporting data.")
        elif evidence_coverage < 85:
            recommendations.append("🟡 Moderate: Evidence coverage could be improved. Consider additional citations.")
        else:
            recommendations.append("✅ Good: Evidence coverage above 85%.")

        # Specific improvements
        weak_claims = [r for r in validation_results if r.validation_status == "weak"]
        if len(weak_claims) > 5:
            recommendations.append(f"🔧 Fix {len(weak_claims)} weak claims with specific evidence")

        unsupported_claims = [r for r in validation_results if r.validation_status == "unsupported"]
        if len(unsupported_claims) > 0:
            recommendations.append(f"⚠️ Remove or support {len(unsupported_claims)} unsupported claims")

        # DD-RAPTOR usage recommendations
        recommendations.append("📚 Consider citing papers from DD-RAPTOR database for stronger evidence")
        recommendations.append("🔄 Use real-time validation during writing for better quality")

        return recommendations

def main():
    parser = argparse.ArgumentParser(
        description="Map proposal claims to DD-RAPTOR evidence"
    )
    parser.add_argument(
        "--proposal",
        required=True,
        help="Path to proposal markdown file"
    )
    parser.add_argument(
        "--output",
        default="evidence_mapping_report.json",
        help="Output JSON file for evidence mapping report"
    )
    parser.add_argument(
        "--db-path",
        default="chromadb_data_dd",
        help="Path to DD-RAPTOR ChromaDB database"
    )

    args = parser.parse_args()

    # Verify files exist
    if not Path(args.proposal).exists():
        print(f"❌ Proposal file not found: {args.proposal}")
        return

    if not Path(args.db_path).exists():
        print(f"❌ DD-RAPTOR database not found: {args.db_path}")
        print("Run: poetry run python scripts/load_json_to_chromadb_dd.py")
        return

    try:
        # Initialize mapper and run analysis
        mapper = ProposalEvidenceMapper(args.db_path)

        print("=" * 60)
        print("🧬 DD-RAPTOR EVIDENCE MAPPING")
        print("=" * 60)

        report = mapper.map_proposal(args.proposal)

        # Save report
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Print summary
        print("\n" + "=" * 60)
        print("📊 ANALYSIS SUMMARY")
        print("=" * 60)

        summary = report["summary"]
        print(f"📝 Total Claims: {summary['total_claims']}")
        print(f"🎯 Evidence Coverage: {summary['evidence_coverage_percent']}%")
        print(f"⭐ Scientific Rigor Score: {summary['scientific_rigor_score']}/100")

        print(f"\n📊 Validation Breakdown:")
        breakdown = summary["validation_breakdown"]
        print(f"   ✅ Strong: {breakdown['strong']}")
        print(f"   🟡 Moderate: {breakdown['moderate']}")
        print(f"   🟠 Weak: {breakdown['weak']}")
        print(f"   ❌ Unsupported: {breakdown['unsupported']}")

        print(f"\n💡 Key Recommendations:")
        for rec in report["recommendations"][:3]:
            print(f"   • {rec}")

        print(f"\n💾 Full report saved to: {args.output}")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()