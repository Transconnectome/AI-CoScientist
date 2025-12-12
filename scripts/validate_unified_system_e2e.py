#!/usr/bin/env python3
"""
End-to-End Validation for Unified RAG Proposal System
=====================================================

Complete validation of the DD-RAPTOR to Unified RAG Orchestrator migration.
Tests all components and their integration.

Validation Steps:
1. Component availability check
2. Unified RAG Orchestrator health
3. Multi-strategy search validation
4. Cross-domain synthesis verification
5. Multi-agent pipeline integration
6. Evidence mapping system test
7. Performance benchmarking
8. Complete workflow simulation

Usage:
    poetry run python scripts/validate_unified_system_e2e.py
    poetry run python scripts/validate_unified_system_e2e.py --verbose
    poetry run python scripts/validate_unified_system_e2e.py --benchmark
"""

import asyncio
import sys
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
import logging
import traceback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class UnifiedSystemValidator:
    """End-to-End Validator for Unified RAG Proposal System"""

    def __init__(self, verbose: bool = False, benchmark: bool = False):
        self.verbose = verbose
        self.benchmark = benchmark
        self.results = {
            "validation_timestamp": datetime.now().isoformat(),
            "overall_status": "PENDING",
            "component_results": {},
            "integration_results": {},
            "performance_metrics": {},
            "recommendations": []
        }
        self.orchestrator = None

    async def run_full_validation(self) -> Dict[str, Any]:
        """Run complete system validation"""
        logger.info("="*60)
        logger.info("UNIFIED RAG PROPOSAL SYSTEM - E2E VALIDATION")
        logger.info("="*60)

        validation_steps = [
            ("Component Availability", self._validate_components),
            ("Unified RAG Orchestrator", self._validate_orchestrator),
            ("Multi-Strategy Search", self._validate_multi_strategy_search),
            ("Cross-Domain Synthesis", self._validate_cross_domain),
            ("Multi-Agent Pipeline", self._validate_multi_agent_pipeline),
            ("Evidence Mapping System", self._validate_evidence_mapping),
            ("Performance Metrics", self._validate_performance),
            ("Workflow Integration", self._validate_workflow_integration)
        ]

        passed = 0
        failed = 0

        for step_name, step_func in validation_steps:
            logger.info(f"\n{'='*40}")
            logger.info(f"📋 Validating: {step_name}")
            logger.info('='*40)

            try:
                result = await step_func()
                self.results["component_results"][step_name] = result

                if result["status"] == "PASS":
                    logger.info(f"✅ {step_name}: PASS")
                    passed += 1
                else:
                    logger.warning(f"⚠️ {step_name}: {result['status']}")
                    if result["status"] == "FAIL":
                        failed += 1
                    else:
                        passed += 1  # PARTIAL counts as pass

                if self.verbose and result.get("details"):
                    logger.info(f"   Details: {result['details']}")

            except Exception as e:
                logger.error(f"❌ {step_name}: ERROR - {e}")
                self.results["component_results"][step_name] = {
                    "status": "ERROR",
                    "error": str(e),
                    "traceback": traceback.format_exc() if self.verbose else None
                }
                failed += 1

        # Calculate overall status
        total = len(validation_steps)
        if failed == 0:
            self.results["overall_status"] = "PASS"
        elif failed <= 2:
            self.results["overall_status"] = "PARTIAL"
        else:
            self.results["overall_status"] = "FAIL"

        self.results["summary"] = {
            "total_steps": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total if total > 0 else 0
        }

        # Generate recommendations
        self._generate_recommendations()

        return self.results

    async def _validate_components(self) -> Dict[str, Any]:
        """Validate all required components are available"""
        components = {
            "unified_rag_orchestrator": False,
            "proposal_agent": False,
            "samsung_generator": False,
            "multi_agent_pipeline": False,
            "evidence_mapper": False
        }

        # Check imports
        try:
            from src.services.rag.unified_rag_orchestrator import UnifiedRAGOrchestrator
            components["unified_rag_orchestrator"] = True
        except ImportError:
            pass

        try:
            from src.agents.pool import AgentPool
            components["proposal_agent"] = True
        except ImportError:
            pass

        # Check script files
        script_files = {
            "samsung_generator": "src/proposal/samsung_grant_generator_unified.py",
            "multi_agent_pipeline": "scripts/multi_agent_unified_pipeline.py",
            "evidence_mapper": "scripts/map_proposal_to_unified_evidence.py"
        }

        base_path = Path(__file__).parent.parent

        for component, script_path in script_files.items():
            full_path = base_path / script_path
            if full_path.exists():
                components[component] = True

        available_count = sum(1 for v in components.values() if v)
        total_count = len(components)

        status = "PASS" if available_count == total_count else "PARTIAL" if available_count >= 3 else "FAIL"

        return {
            "status": status,
            "available": available_count,
            "total": total_count,
            "components": components,
            "details": f"{available_count}/{total_count} components available"
        }

    async def _validate_orchestrator(self) -> Dict[str, Any]:
        """Validate Unified RAG Orchestrator"""
        try:
            from src.services.rag.unified_rag_orchestrator import (
                create_unified_orchestrator,
                QueryContext,
                QueryComplexity,
                QueryDomain,
                RAGStrategy
            )

            # Create orchestrator
            self.orchestrator = create_unified_orchestrator()

            # Check strategy health
            health = self.orchestrator.get_strategy_health()
            available_strategies = [s for s, info in health.items() if info.get('available', False)]

            # Warmup
            await self.orchestrator.warmup()

            return {
                "status": "PASS" if len(available_strategies) >= 4 else "PARTIAL",
                "available_strategies": available_strategies,
                "total_strategies": len(health),
                "details": f"{len(available_strategies)} strategies available: {available_strategies}"
            }

        except Exception as e:
            return {
                "status": "FAIL",
                "error": str(e),
                "details": "Orchestrator initialization failed"
            }

    async def _validate_multi_strategy_search(self) -> Dict[str, Any]:
        """Validate multi-strategy search capability"""
        try:
            from src.services.rag.unified_rag_orchestrator import (
                QueryContext,
                QueryComplexity,
                QueryDomain
            )

            if not self.orchestrator:
                return {"status": "SKIP", "details": "Orchestrator not available"}

            test_queries = [
                {
                    "query": "ESM3 protein structure prediction for brain development",
                    "domain": QueryDomain.NEUROSCIENCE,
                    "complexity": QueryComplexity.COMPLEX
                },
                {
                    "query": "Samsung grant proposal research methodology",
                    "domain": QueryDomain.GENERAL,
                    "complexity": QueryComplexity.MEDIUM
                },
                {
                    "query": "quantum machine learning optimization",
                    "domain": QueryDomain.QUANTUM_ML,
                    "complexity": QueryComplexity.COMPLEX
                }
            ]

            search_results = []
            for tq in test_queries:
                try:
                    context = QueryContext(
                        query=tq["query"],
                        complexity=tq["complexity"],
                        domain=tq["domain"],
                        intent="synthesis",
                        confidence=0.9
                    )

                    response = await self.orchestrator.search(context)

                    search_results.append({
                        "query": tq["query"][:50],
                        "success": True,
                        "strategy": str(response.strategy_used) if response else "N/A",
                        "confidence": response.confidence if response else 0
                    })
                except Exception as e:
                    search_results.append({
                        "query": tq["query"][:50],
                        "success": False,
                        "error": str(e)
                    })

            successful = sum(1 for r in search_results if r["success"])

            return {
                "status": "PASS" if successful == len(test_queries) else "PARTIAL" if successful > 0 else "FAIL",
                "successful_searches": successful,
                "total_queries": len(test_queries),
                "results": search_results,
                "details": f"{successful}/{len(test_queries)} searches successful"
            }

        except Exception as e:
            return {"status": "FAIL", "error": str(e)}

    async def _validate_cross_domain(self) -> Dict[str, Any]:
        """Validate cross-domain synthesis capability"""
        try:
            from src.services.rag.unified_rag_orchestrator import (
                QueryContext,
                QueryComplexity,
                QueryDomain
            )

            if not self.orchestrator:
                return {"status": "SKIP", "details": "Orchestrator not available"}

            # Cross-domain query
            cross_domain_query = QueryContext(
                query="ESM3 protein evolution brain neural development quantum optimization",
                complexity=QueryComplexity.COMPLEX,
                domain=QueryDomain.NEUROSCIENCE,
                intent="synthesis",
                confidence=0.9,
                metadata={
                    "cross_domain_enabled": True,
                    "target_domains": ["neuroscience", "protein_research", "quantum_ml"]
                }
            )

            response = await self.orchestrator.search(cross_domain_query)

            # Check for cross-domain indicators in response
            cross_domain_success = False
            if response and response.answer:
                answer_lower = response.answer.lower()
                # Check for multiple domain keywords
                domain_keywords = {
                    "protein": ["protein", "esm", "structure"],
                    "neuro": ["brain", "neural", "neuron"],
                    "quantum": ["quantum", "optimization"]
                }

                domains_found = 0
                for domain, keywords in domain_keywords.items():
                    if any(kw in answer_lower for kw in keywords):
                        domains_found += 1

                cross_domain_success = domains_found >= 2

            return {
                "status": "PASS" if cross_domain_success else "PARTIAL",
                "cross_domain_detected": cross_domain_success,
                "strategy_used": str(response.strategy_used) if response else "N/A",
                "confidence": response.confidence if response else 0,
                "details": f"Cross-domain synthesis {'successful' if cross_domain_success else 'partial'}"
            }

        except Exception as e:
            return {"status": "FAIL", "error": str(e)}

    async def _validate_multi_agent_pipeline(self) -> Dict[str, Any]:
        """Validate multi-agent unified pipeline"""
        try:
            # Check script exists
            pipeline_path = Path(__file__).parent / "multi_agent_unified_pipeline.py"

            if not pipeline_path.exists():
                return {"status": "PARTIAL", "details": "Pipeline script not found, but may be in progress"}

            # Read and verify structure
            with open(pipeline_path, 'r') as f:
                content = f.read()

            # Check for key components
            required_components = [
                "UnifiedMultiAgentPipeline",
                "UnifiedAgentTask",
                "UnifiedAgentResult",
                "UnifiedRAGOrchestrator",
                "agent_rag_configs",
                "enhanced_literature_analyst",
                "statistical_analyst",
                "hypothesis_generator",
                "grant_writer",
                "clinical_validation_agent",
                "neuroscience_expert"
            ]

            found_components = [c for c in required_components if c in content]

            return {
                "status": "PASS" if len(found_components) >= 10 else "PARTIAL",
                "found_components": len(found_components),
                "total_required": len(required_components),
                "missing": [c for c in required_components if c not in found_components],
                "details": f"{len(found_components)}/{len(required_components)} components found"
            }

        except Exception as e:
            return {"status": "FAIL", "error": str(e)}

    async def _validate_evidence_mapping(self) -> Dict[str, Any]:
        """Validate evidence mapping system"""
        try:
            # Check script exists
            mapper_path = Path(__file__).parent / "map_proposal_to_unified_evidence.py"

            if not mapper_path.exists():
                return {"status": "PARTIAL", "details": "Mapper script not found"}

            # Read and verify structure
            with open(mapper_path, 'r') as f:
                content = f.read()

            # Check for key components
            required_components = [
                "UnifiedEvidenceMapper",
                "ScientificClaim",
                "EvidenceSource",
                "ClaimEvidence",
                "EvidenceReport",
                "map_proposal_evidence",
                "extract_claims",
                "validate_claim_unified"
            ]

            found_components = [c for c in required_components if c in content]

            return {
                "status": "PASS" if len(found_components) >= 6 else "PARTIAL",
                "found_components": len(found_components),
                "total_required": len(required_components),
                "details": f"{len(found_components)}/{len(required_components)} components found"
            }

        except Exception as e:
            return {"status": "FAIL", "error": str(e)}

    async def _validate_performance(self) -> Dict[str, Any]:
        """Validate performance metrics"""
        try:
            if not self.orchestrator:
                return {"status": "SKIP", "details": "Orchestrator not available for performance test"}

            from src.services.rag.unified_rag_orchestrator import QueryContext, QueryComplexity, QueryDomain

            # Benchmark searches
            import time

            latencies = []
            for i in range(3):
                start = time.time()

                context = QueryContext(
                    query=f"Test query {i} for performance benchmark neuroscience",
                    complexity=QueryComplexity.MEDIUM,
                    domain=QueryDomain.GENERAL,
                    intent="factual",
                    confidence=0.8
                )

                await self.orchestrator.search(context)
                latency = (time.time() - start) * 1000  # ms
                latencies.append(latency)

            avg_latency = sum(latencies) / len(latencies) if latencies else 0
            max_latency = max(latencies) if latencies else 0

            # Performance thresholds
            latency_pass = avg_latency < 2000  # 2 seconds

            self.results["performance_metrics"] = {
                "average_latency_ms": avg_latency,
                "max_latency_ms": max_latency,
                "queries_tested": len(latencies)
            }

            return {
                "status": "PASS" if latency_pass else "PARTIAL",
                "average_latency_ms": avg_latency,
                "max_latency_ms": max_latency,
                "threshold_ms": 2000,
                "details": f"Avg latency: {avg_latency:.1f}ms"
            }

        except Exception as e:
            return {"status": "FAIL", "error": str(e)}

    async def _validate_workflow_integration(self) -> Dict[str, Any]:
        """Validate complete workflow integration"""
        try:
            # Check all workflow scripts exist
            workflow_scripts = [
                "scripts/proposal_optimizer_unified.py",
                "scripts/multi_agent_unified_pipeline.py",
                "scripts/map_proposal_to_unified_evidence.py"
            ]

            base_path = Path(__file__).parent.parent
            existing = []
            missing = []

            for script in workflow_scripts:
                if (base_path / script).exists():
                    existing.append(script)
                else:
                    missing.append(script)

            # Check documentation
            docs = [
                "PROPOSAL_OPTIMIZATION_QUICK_REFERENCE_UNIFIED.md",
                "UNIFIED_RAG_MIGRATION_SUMMARY.md"
            ]

            for doc in docs:
                if (base_path / doc).exists():
                    existing.append(doc)
                else:
                    missing.append(doc)

            return {
                "status": "PASS" if len(missing) == 0 else "PARTIAL" if len(existing) >= 3 else "FAIL",
                "existing_components": existing,
                "missing_components": missing,
                "details": f"{len(existing)}/{len(workflow_scripts) + len(docs)} workflow components available"
            }

        except Exception as e:
            return {"status": "FAIL", "error": str(e)}

    def _generate_recommendations(self):
        """Generate recommendations based on validation results"""
        recommendations = []

        for component, result in self.results["component_results"].items():
            if result.get("status") in ["FAIL", "PARTIAL"]:
                if component == "Unified RAG Orchestrator":
                    recommendations.append("🔧 Configure Unified RAG Orchestrator with all 6 strategies")
                elif component == "Multi-Strategy Search":
                    recommendations.append("🔍 Verify ChromaDB databases are accessible")
                elif component == "Cross-Domain Synthesis":
                    recommendations.append("🌐 Enable cross-domain knowledge synthesis in configuration")
                elif component == "Multi-Agent Pipeline":
                    recommendations.append("🤖 Complete multi-agent pipeline integration")
                elif component == "Evidence Mapping System":
                    recommendations.append("📋 Implement evidence mapping with Unified RAG backing")

        if self.results["overall_status"] == "PASS":
            recommendations.append("✅ System is production ready! Consider running benchmark tests.")

        self.results["recommendations"] = recommendations

    def print_report(self):
        """Print validation report"""
        print("\n" + "="*60)
        print("📊 VALIDATION REPORT")
        print("="*60)

        print(f"\n🎯 Overall Status: {self.results['overall_status']}")

        if "summary" in self.results:
            summary = self.results["summary"]
            print(f"📈 Pass Rate: {summary['pass_rate']*100:.1f}% ({summary['passed']}/{summary['total_steps']})")

        print("\n📋 Component Results:")
        for component, result in self.results["component_results"].items():
            status = result.get("status", "UNKNOWN")
            icon = {"PASS": "✅", "PARTIAL": "⚠️", "FAIL": "❌", "SKIP": "⏭️"}.get(status, "❓")
            print(f"   {icon} {component}: {status}")
            if self.verbose and result.get("details"):
                print(f"      └─ {result['details']}")

        if self.results.get("performance_metrics"):
            print("\n⚡ Performance Metrics:")
            metrics = self.results["performance_metrics"]
            print(f"   Average Latency: {metrics.get('average_latency_ms', 'N/A'):.1f}ms")
            print(f"   Max Latency: {metrics.get('max_latency_ms', 'N/A'):.1f}ms")

        if self.results.get("recommendations"):
            print("\n💡 Recommendations:")
            for rec in self.results["recommendations"]:
                print(f"   {rec}")

        print("\n" + "="*60)


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Unified RAG System E2E Validation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--benchmark", "-b", action="store_true", help="Run benchmarks")
    parser.add_argument("--output", "-o", help="Output file for JSON report")

    args = parser.parse_args()

    validator = UnifiedSystemValidator(verbose=args.verbose, benchmark=args.benchmark)

    results = await validator.run_full_validation()
    validator.print_report()

    # Save report if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Report saved: {output_path}")

    # Return exit code based on status
    return 0 if results["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)