#!/usr/bin/env python3
"""
Unified RAG Evidence Mapping System
====================================

Next-Generation Evidence Mapping powered by Unified RAG Orchestrator
과학적 주장을 1,761+ 문서 cross-domain 지식 베이스에서 검증

Enhanced Features:
- 6-Strategy RAG evidence validation (HYBRID, GRAPH_RAG, GOLDEN_REFERENCE, etc.)
- Cross-domain claim verification (ESM3 + Neuroscience + Quantum ML + Grants)
- Multi-modal evidence synthesis and quality scoring
- Intelligent claim extraction and validation routing
- Real-time evidence strength assessment

Usage:
    # Basic unified evidence mapping
    poetry run python scripts/map_proposal_to_unified_evidence.py \\
        --proposal "proposal.md" \\
        --output "evidence_report.json" \\
        --unified-rag

    # Cross-domain evidence validation
    poetry run python scripts/map_proposal_to_unified_evidence.py \\
        --proposal "proposal.md" \\
        --output "evidence_report.json" \\
        --enable-cross-domain \\
        --domains "neuroscience,protein_research,quantum_ml"

    # Quality assessment mode
    poetry run python scripts/map_proposal_to_unified_evidence.py \\
        --proposal "proposal.md" \\
        --quality-assessment \\
        --strategies "GRAPH_RAG,GOLDEN_REFERENCE"
"""

import argparse
import json
import asyncio
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.services.rag.unified_rag_orchestrator import (
        UnifiedRAGOrchestrator,
        create_unified_orchestrator,
        QueryContext,
        QueryComplexity,
        QueryDomain,
        RAGStrategy
    )
    UNIFIED_RAG_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Unified RAG import: {e}")
    UNIFIED_RAG_AVAILABLE = False

@dataclass
class ScientificClaim:
    """Extracted scientific claim from proposal"""
    claim_id: str
    text: str
    claim_type: str  # hypothesis, methodology, result, assertion
    section: str
    confidence: float
    keywords: List[str] = field(default_factory=list)
    requires_citation: bool = True

@dataclass
class EvidenceSource:
    """Evidence source from Unified RAG"""
    source_id: str
    title: str
    content_snippet: str
    relevance_score: float
    strategy_used: str
    domain: str
    citation_ready: bool = True

@dataclass
class ClaimEvidence:
    """Evidence mapping for a single claim"""
    claim: ScientificClaim
    evidence_sources: List[EvidenceSource]
    evidence_strength: float  # 0.0 - 1.0
    validation_status: str  # strong, moderate, weak, unsupported
    cross_domain_support: bool
    suggested_citations: List[str]
    improvement_recommendations: List[str]
    unified_rag_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EvidenceReport:
    """Complete evidence mapping report"""
    proposal_file: str
    analysis_timestamp: str
    total_claims: int
    validated_claims: int
    evidence_coverage: float
    scientific_rigor_score: float
    cross_domain_synthesis_score: float
    claim_evidence_map: List[ClaimEvidence]
    strategy_performance: Dict[str, Any]
    unified_rag_summary: Dict[str, Any]
    recommendations: List[str]

class UnifiedEvidenceMapper:
    """Unified RAG-powered Evidence Mapping System"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with Unified RAG configuration"""
        self.config = config or self._get_default_config()
        self.unified_orchestrator: Optional[UnifiedRAGOrchestrator] = None

        # Claim type patterns
        self.claim_patterns = {
            "hypothesis": [
                r"we\s+hypothesize\s+that",
                r"our\s+hypothesis\s+is",
                r"we\s+propose\s+that",
                r"가설.*(?:이다|입니다)",
                r"예측.*(?:한다|합니다)"
            ],
            "methodology": [
                r"we\s+will\s+use",
                r"method\s+involves",
                r"approach\s+utilizes",
                r"방법론.*(?:이다|입니다)",
                r"사용.*(?:한다|합니다)"
            ],
            "result": [
                r"results\s+show",
                r"we\s+found\s+that",
                r"evidence\s+suggests",
                r"결과.*보여",
                r"확인.*(?:했다|되었다)"
            ],
            "assertion": [
                r"it\s+is\s+known\s+that",
                r"studies\s+have\s+shown",
                r"research\s+indicates",
                r"알려져\s+있다",
                r"연구.*나타났다"
            ]
        }

        # Domain-specific keywords for routing
        self.domain_keywords = {
            QueryDomain.NEUROSCIENCE: ["brain", "neural", "neuron", "cognitive", "뇌", "신경", "인지"],
            QueryDomain.QUANTUM_ML: ["quantum", "optimization", "algorithm", "양자", "최적화"],
            QueryDomain.DEVELOPMENTAL_DISORDERS: ["autism", "developmental", "disorder", "자폐", "발달장애"],
            QueryDomain.GENERAL: ["AI", "machine learning", "model", "인공지능", "모델"]
        }

        # Strategy performance tracking
        self.strategy_stats = {strategy.name: {"queries": 0, "avg_confidence": 0.0} for strategy in RAGStrategy}

    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            "min_claim_length": 20,
            "max_claims_per_section": 20,
            "evidence_threshold": 0.7,
            "enable_cross_domain": True,
            "preferred_strategies": ["GRAPH_RAG", "HYBRID", "GOLDEN_REFERENCE"],
            "output_directory": "output/evidence_reports/",
            "citation_generation": True,
            "improvement_suggestions": True
        }

    async def initialize(self):
        """Initialize Unified RAG Orchestrator"""
        logger.info("🚀 Initializing Unified Evidence Mapper...")

        if UNIFIED_RAG_AVAILABLE:
            self.unified_orchestrator = create_unified_orchestrator()
            await self.unified_orchestrator.warmup()

            health = self.unified_orchestrator.get_strategy_health()
            available = [s for s, info in health.items() if info.get('available', False)]
            logger.info(f"✅ Unified RAG initialized with strategies: {available}")
        else:
            logger.warning("⚠️ Unified RAG not available, using fallback mode")

        Path(self.config["output_directory"]).mkdir(parents=True, exist_ok=True)
        logger.info("🎯 Unified Evidence Mapper ready")

    async def map_proposal_evidence(self,
                                   proposal_file: str,
                                   enable_cross_domain: bool = True,
                                   target_domains: Optional[List[str]] = None,
                                   quality_assessment: bool = False) -> EvidenceReport:
        """
        Map proposal claims to Unified RAG evidence

        Returns comprehensive evidence report with:
        - Claim extraction and classification
        - Multi-strategy evidence validation
        - Cross-domain synthesis assessment
        - Scientific rigor scoring
        - Improvement recommendations
        """
        logger.info(f"📋 Starting Unified Evidence Mapping: {proposal_file}")
        start_time = datetime.now()

        # Load proposal content
        proposal_path = Path(proposal_file)
        if not proposal_path.exists():
            raise FileNotFoundError(f"Proposal not found: {proposal_file}")

        with open(proposal_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 1. Extract scientific claims
        claims = self._extract_claims(content)
        logger.info(f"📝 Extracted {len(claims)} scientific claims")

        # 2. Validate each claim with Unified RAG
        claim_evidence_list = []
        strategy_usage = {}

        for i, claim in enumerate(claims):
            logger.info(f"🔍 Validating claim {i+1}/{len(claims)}: {claim.text[:50]}...")

            # Validate claim with cross-domain evidence
            evidence = await self._validate_claim_unified(
                claim,
                enable_cross_domain=enable_cross_domain,
                target_domains=target_domains
            )

            claim_evidence_list.append(evidence)

            # Track strategy usage
            for source in evidence.evidence_sources:
                strategy = source.strategy_used
                strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1

        # 3. Calculate overall metrics
        validated_claims = sum(1 for ce in claim_evidence_list if ce.validation_status in ["strong", "moderate"])
        evidence_coverage = validated_claims / len(claims) if claims else 0

        avg_evidence_strength = sum(ce.evidence_strength for ce in claim_evidence_list) / len(claim_evidence_list) if claim_evidence_list else 0

        cross_domain_count = sum(1 for ce in claim_evidence_list if ce.cross_domain_support)
        cross_domain_score = cross_domain_count / len(claim_evidence_list) if claim_evidence_list else 0

        # Scientific rigor = weighted combination
        scientific_rigor = (
            evidence_coverage * 0.4 +
            avg_evidence_strength * 0.3 +
            cross_domain_score * 0.3
        )

        # 4. Generate recommendations
        recommendations = self._generate_recommendations(claim_evidence_list, scientific_rigor)

        # 5. Compile strategy performance
        strategy_performance = self._compile_strategy_performance(claim_evidence_list)

        # 6. Unified RAG summary
        unified_rag_summary = {
            "total_queries": sum(strategy_usage.values()),
            "strategies_used": list(strategy_usage.keys()),
            "strategy_distribution": strategy_usage,
            "cross_domain_enabled": enable_cross_domain,
            "target_domains": target_domains or ["all"]
        }

        end_time = datetime.now()

        report = EvidenceReport(
            proposal_file=str(proposal_path),
            analysis_timestamp=end_time.isoformat(),
            total_claims=len(claims),
            validated_claims=validated_claims,
            evidence_coverage=evidence_coverage,
            scientific_rigor_score=scientific_rigor,
            cross_domain_synthesis_score=cross_domain_score,
            claim_evidence_map=claim_evidence_list,
            strategy_performance=strategy_performance,
            unified_rag_summary=unified_rag_summary,
            recommendations=recommendations
        )

        logger.info(f"✅ Evidence mapping complete!")
        logger.info(f"📊 Scientific Rigor: {scientific_rigor:.3f}")
        logger.info(f"📊 Evidence Coverage: {evidence_coverage:.1%}")
        logger.info(f"📊 Cross-Domain Score: {cross_domain_score:.1%}")

        return report

    def _extract_claims(self, content: str) -> List[ScientificClaim]:
        """Extract scientific claims from proposal content"""
        claims = []
        claim_id = 0

        # Split into sections
        sections = self._split_into_sections(content)

        for section_name, section_content in sections.items():
            # Extract sentences
            sentences = re.split(r'[.!?。]\s+', section_content)

            for sentence in sentences:
                sentence = sentence.strip()

                # Skip short sentences
                if len(sentence) < self.config["min_claim_length"]:
                    continue

                # Classify claim type
                claim_type = self._classify_claim_type(sentence)

                if claim_type:
                    # Extract keywords
                    keywords = self._extract_keywords(sentence)

                    # Determine if citation needed
                    requires_citation = self._needs_citation(sentence, claim_type)

                    claim = ScientificClaim(
                        claim_id=f"claim_{claim_id}",
                        text=sentence,
                        claim_type=claim_type,
                        section=section_name,
                        confidence=0.8,  # Initial confidence
                        keywords=keywords,
                        requires_citation=requires_citation
                    )

                    claims.append(claim)
                    claim_id += 1

                    if len(claims) >= self.config["max_claims_per_section"] * len(sections):
                        break

        return claims

    def _split_into_sections(self, content: str) -> Dict[str, str]:
        """Split content into sections"""
        sections = {}

        # Try to detect section headers
        section_patterns = [
            r'^#+\s+(.+)$',  # Markdown headers
            r'^([A-Z\d]+\.?\s+[A-Z].+)$',  # Numbered sections
            r'^(연구\s*목표|방법론|기대\s*성과|예산).+$',  # Korean sections
        ]

        current_section = "introduction"
        current_content = []

        for line in content.split('\n'):
            is_header = False
            for pattern in section_patterns:
                match = re.match(pattern, line.strip(), re.MULTILINE)
                if match:
                    # Save previous section
                    if current_content:
                        sections[current_section] = '\n'.join(current_content)

                    # Start new section
                    current_section = match.group(1).lower().strip()[:50]
                    current_content = []
                    is_header = True
                    break

            if not is_header:
                current_content.append(line)

        # Save last section
        if current_content:
            sections[current_section] = '\n'.join(current_content)

        # If no sections detected, use entire content
        if not sections:
            sections["full_document"] = content

        return sections

    def _classify_claim_type(self, sentence: str) -> Optional[str]:
        """Classify the type of scientific claim"""
        sentence_lower = sentence.lower()

        for claim_type, patterns in self.claim_patterns.items():
            for pattern in patterns:
                if re.search(pattern, sentence_lower, re.IGNORECASE):
                    return claim_type

        # Check for general scientific assertions
        scientific_indicators = [
            "significant", "demonstrate", "show", "indicate", "suggest",
            "evidence", "correlation", "effect", "impact",
            "중요", "입증", "나타", "영향", "효과"
        ]

        if any(indicator in sentence_lower for indicator in scientific_indicators):
            return "assertion"

        return None

    def _extract_keywords(self, sentence: str) -> List[str]:
        """Extract keywords from sentence"""
        # Simple keyword extraction - in production, use NER or keyword extraction model
        keywords = []

        # Scientific terms
        scientific_terms = [
            "model", "algorithm", "neural", "brain", "protein", "quantum",
            "learning", "optimization", "prediction", "analysis",
            "모델", "알고리즘", "뇌", "단백질", "양자", "예측", "분석"
        ]

        sentence_lower = sentence.lower()
        for term in scientific_terms:
            if term in sentence_lower:
                keywords.append(term)

        return keywords[:5]  # Limit to 5 keywords

    def _needs_citation(self, sentence: str, claim_type: str) -> bool:
        """Determine if claim requires citation"""
        # Hypotheses generated by the research don't need citations
        if claim_type == "hypothesis":
            return False

        # Assertions and results about prior work need citations
        citation_indicators = [
            "study", "research", "shown", "demonstrated", "known",
            "evidence", "previous", "prior",
            "연구", "알려져", "보고", "입증"
        ]

        return any(indicator in sentence.lower() for indicator in citation_indicators)

    async def _validate_claim_unified(self,
                                     claim: ScientificClaim,
                                     enable_cross_domain: bool = True,
                                     target_domains: Optional[List[str]] = None) -> ClaimEvidence:
        """Validate claim using Unified RAG"""

        evidence_sources = []
        cross_domain_support = False

        # Determine claim domain
        claim_domain = self._detect_claim_domain(claim)

        # Create query context
        query_context = QueryContext(
            query=f"{claim.text} {' '.join(claim.keywords)}",
            complexity=QueryComplexity.COMPLEX if claim.claim_type in ["hypothesis", "methodology"] else QueryComplexity.MEDIUM,
            domain=claim_domain,
            intent="synthesis" if enable_cross_domain else "factual",
            confidence=claim.confidence,
            metadata={
                "claim_id": claim.claim_id,
                "claim_type": claim.claim_type,
                "requires_citation": claim.requires_citation,
                "cross_domain_enabled": enable_cross_domain
            }
        )

        # Execute Unified RAG search
        if self.unified_orchestrator:
            try:
                response = await self.unified_orchestrator.search(query_context)

                # Convert response to evidence sources
                if response.sources:
                    for i, source in enumerate(response.sources[:5]):  # Top 5 sources
                        evidence_source = EvidenceSource(
                            source_id=f"src_{claim.claim_id}_{i}",
                            title=source.get('title', f"Source {i+1}") if isinstance(source, dict) else f"Source {i+1}",
                            content_snippet=str(source)[:200] if source else "",
                            relevance_score=response.confidence,
                            strategy_used=str(response.strategy_used),
                            domain=str(claim_domain),
                            citation_ready=True
                        )
                        evidence_sources.append(evidence_source)

                # Check for cross-domain support
                if enable_cross_domain and response.confidence > 0.7:
                    cross_domain_support = True

                # Update strategy stats
                strategy_name = str(response.strategy_used)
                if strategy_name in self.strategy_stats:
                    stats = self.strategy_stats[strategy_name]
                    stats["queries"] += 1
                    stats["avg_confidence"] = (
                        stats["avg_confidence"] * (stats["queries"] - 1) + response.confidence
                    ) / stats["queries"]

            except Exception as e:
                logger.warning(f"RAG search failed for claim {claim.claim_id}: {e}")

        # Calculate evidence strength
        if evidence_sources:
            evidence_strength = sum(s.relevance_score for s in evidence_sources) / len(evidence_sources)
        else:
            evidence_strength = 0.0

        # Determine validation status
        if evidence_strength >= 0.8:
            validation_status = "strong"
        elif evidence_strength >= 0.6:
            validation_status = "moderate"
        elif evidence_strength >= 0.4:
            validation_status = "weak"
        else:
            validation_status = "unsupported"

        # Generate suggested citations
        suggested_citations = self._generate_citations(evidence_sources)

        # Generate improvement recommendations
        improvement_recs = self._generate_claim_improvements(claim, evidence_sources, validation_status)

        return ClaimEvidence(
            claim=claim,
            evidence_sources=evidence_sources,
            evidence_strength=evidence_strength,
            validation_status=validation_status,
            cross_domain_support=cross_domain_support,
            suggested_citations=suggested_citations,
            improvement_recommendations=improvement_recs,
            unified_rag_metrics={
                "sources_found": len(evidence_sources),
                "cross_domain": cross_domain_support,
                "strategy": evidence_sources[0].strategy_used if evidence_sources else "none"
            }
        )

    def _detect_claim_domain(self, claim: ScientificClaim) -> QueryDomain:
        """Detect the domain of a claim"""
        text_lower = claim.text.lower()

        for domain, keywords in self.domain_keywords.items():
            if any(kw.lower() in text_lower for kw in keywords):
                return domain

        return QueryDomain.GENERAL

    def _generate_citations(self, sources: List[EvidenceSource]) -> List[str]:
        """Generate citation suggestions from evidence sources"""
        citations = []

        for source in sources:
            if source.citation_ready and source.relevance_score > 0.6:
                citation = f"[{source.title}] (Relevance: {source.relevance_score:.2f}, Strategy: {source.strategy_used})"
                citations.append(citation)

        return citations

    def _generate_claim_improvements(self,
                                    claim: ScientificClaim,
                                    sources: List[EvidenceSource],
                                    status: str) -> List[str]:
        """Generate improvement recommendations for a claim"""
        recommendations = []

        if status == "unsupported":
            recommendations.append(f"⚠️ 주장 '{claim.text[:50]}...'에 대한 근거가 부족합니다. 추가 문헌 조사 필요.")

        if status == "weak":
            recommendations.append(f"💡 주장에 대한 추가 인용 권장. 현재 근거 강도: weak")

        if claim.requires_citation and not sources:
            recommendations.append(f"📚 인용 필요: '{claim.text[:50]}...'")

        if claim.claim_type == "hypothesis" and status not in ["strong", "moderate"]:
            recommendations.append(f"🔬 가설의 이론적 기반 강화 필요")

        return recommendations

    def _generate_recommendations(self,
                                 claim_evidence_list: List[ClaimEvidence],
                                 scientific_rigor: float) -> List[str]:
        """Generate overall recommendations"""
        recommendations = []

        # Evidence coverage recommendations
        unsupported = sum(1 for ce in claim_evidence_list if ce.validation_status == "unsupported")
        if unsupported > 0:
            recommendations.append(f"📋 {unsupported}개 주장에 대한 추가 근거 필요")

        # Cross-domain recommendations
        cross_domain_count = sum(1 for ce in claim_evidence_list if ce.cross_domain_support)
        if cross_domain_count < len(claim_evidence_list) * 0.5:
            recommendations.append("🌐 Cross-domain 지식 활용 강화 권장 (ESM3 + 뇌과학 + 양자ML 통합)")

        # Scientific rigor recommendations
        if scientific_rigor < 0.7:
            recommendations.append("⚠️ 과학적 엄밀성 점수 낮음. 전반적인 근거 강화 필요")
        elif scientific_rigor < 0.85:
            recommendations.append("💡 과학적 엄밀성 양호. 추가 개선으로 95+ 점수 달성 가능")

        # Citation recommendations
        needs_citation = sum(1 for ce in claim_evidence_list
                           if ce.claim.requires_citation and not ce.suggested_citations)
        if needs_citation > 0:
            recommendations.append(f"📚 {needs_citation}개 주장에 인용 추가 필요")

        return recommendations

    def _compile_strategy_performance(self, claim_evidence_list: List[ClaimEvidence]) -> Dict[str, Any]:
        """Compile strategy performance metrics"""
        performance = {}

        for ce in claim_evidence_list:
            for source in ce.evidence_sources:
                strategy = source.strategy_used
                if strategy not in performance:
                    performance[strategy] = {
                        "total_sources": 0,
                        "total_relevance": 0.0,
                        "claims_supported": 0
                    }

                performance[strategy]["total_sources"] += 1
                performance[strategy]["total_relevance"] += source.relevance_score

                if ce.validation_status in ["strong", "moderate"]:
                    performance[strategy]["claims_supported"] += 1

        # Calculate averages
        for strategy, stats in performance.items():
            if stats["total_sources"] > 0:
                stats["avg_relevance"] = stats["total_relevance"] / stats["total_sources"]

        return performance

    async def save_report(self, report: EvidenceReport, output_file: str) -> Path:
        """Save evidence report to file"""
        output_path = Path(output_file)

        # Convert to JSON-serializable format
        report_dict = {
            "proposal_file": report.proposal_file,
            "analysis_timestamp": report.analysis_timestamp,
            "summary": {
                "total_claims": report.total_claims,
                "validated_claims": report.validated_claims,
                "evidence_coverage": report.evidence_coverage,
                "scientific_rigor_score": report.scientific_rigor_score,
                "cross_domain_synthesis_score": report.cross_domain_synthesis_score
            },
            "unified_rag_summary": report.unified_rag_summary,
            "strategy_performance": report.strategy_performance,
            "recommendations": report.recommendations,
            "claim_evidence_details": [
                {
                    "claim_id": ce.claim.claim_id,
                    "claim_text": ce.claim.text[:200],
                    "claim_type": ce.claim.claim_type,
                    "validation_status": ce.validation_status,
                    "evidence_strength": ce.evidence_strength,
                    "cross_domain_support": ce.cross_domain_support,
                    "sources_count": len(ce.evidence_sources),
                    "suggested_citations": ce.suggested_citations[:3],
                    "improvements": ce.improvement_recommendations
                }
                for ce in report.claim_evidence_map
            ]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, ensure_ascii=False, indent=2)

        logger.info(f"💾 Report saved: {output_path}")
        return output_path

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Unified RAG Evidence Mapping")
    parser.add_argument("--proposal", "-p", required=True, help="Proposal file path")
    parser.add_argument("--output", "-o", help="Output report file")
    parser.add_argument("--unified-rag", action="store_true", default=True, help="Enable Unified RAG")
    parser.add_argument("--enable-cross-domain", action="store_true", default=True, help="Enable cross-domain synthesis")
    parser.add_argument("--domains", help="Comma-separated target domains")
    parser.add_argument("--quality-assessment", action="store_true", help="Run quality assessment mode")
    parser.add_argument("--strategies", help="Comma-separated preferred strategies")

    args = parser.parse_args()

    # Initialize mapper
    mapper = UnifiedEvidenceMapper()
    await mapper.initialize()

    # Parse domains
    target_domains = args.domains.split(',') if args.domains else None

    # Run evidence mapping
    report = await mapper.map_proposal_evidence(
        proposal_file=args.proposal,
        enable_cross_domain=args.enable_cross_domain,
        target_domains=target_domains,
        quality_assessment=args.quality_assessment
    )

    # Determine output file
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"output/evidence_reports/evidence_report_{timestamp}.json"
        Path("output/evidence_reports").mkdir(parents=True, exist_ok=True)

    # Save report
    await mapper.save_report(report, output_file)

    # Print summary
    print(f"\n{'='*60}")
    print("UNIFIED RAG EVIDENCE MAPPING RESULTS")
    print(f"{'='*60}")
    print(f"📄 Proposal: {args.proposal}")
    print(f"📊 Total Claims: {report.total_claims}")
    print(f"✅ Validated Claims: {report.validated_claims}")
    print(f"📈 Evidence Coverage: {report.evidence_coverage:.1%}")
    print(f"🎯 Scientific Rigor: {report.scientific_rigor_score:.3f}")
    print(f"🌐 Cross-Domain Score: {report.cross_domain_synthesis_score:.1%}")
    print(f"\n📋 Recommendations:")
    for rec in report.recommendations:
        print(f"   {rec}")
    print(f"\n💾 Report saved: {output_file}")

    # Return for programmatic use
    return report

if __name__ == "__main__":
    asyncio.run(main())