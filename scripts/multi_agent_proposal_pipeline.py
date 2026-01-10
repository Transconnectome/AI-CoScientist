#!/usr/bin/env python3
"""
Multi-Agent Proposal Pipeline
============================

AI-CoScientist의 6개 전문 에이전트를 체계적으로 활용하여
과학적 엄밀성 90+ 점수의 제안서를 생성하는 파이프라인

Agents:
- Enhanced Literature Analyst: DD-RAPTOR 기반 문헌 검토
- Statistical Analyst: 샘플 사이즈, 검정력 계산 검증
- Hypothesis Generator: 혁신적 연구 가설 생성
- Grant Writer: 삼성 양식에 맞는 제안서 작성
- Clinical Validation Agent: 임상 적용 가능성 검증
- Neuroscience Expert: 뇌과학 전문성 검토

Usage:
    # Full pipeline
    poetry run python scripts/multi_agent_proposal_pipeline.py \
        --mode full_pipeline \
        --input "proposal_draft.md" \
        --output "enhanced_proposal.md"

    # Agent-specific processing
    poetry run python scripts/multi_agent_proposal_pipeline.py \
        --mode agent_specific \
        --agent statistical_analyst \
        --input "proposal.md"
"""

import argparse
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.agents.pool import AgentPool
    from src.services.hybrid_rag_service import HybridRAGService
    from src.core.config import Settings
    from src.services.llm.interface import LLMService
    from src.services.knowledge_base.context_manager import ContextManager
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the AI-CoScientist directory and all dependencies are installed")
    sys.exit(1)

@dataclass
class AgentTask:
    """Task for a specific agent"""
    agent_name: str
    task_type: str
    input_data: str
    context: Dict[str, Any]
    priority: int
    dependencies: List[str]  # Other agents this depends on

@dataclass
class AgentResult:
    """Result from agent processing"""
    agent_name: str
    task_type: str
    status: str  # "success", "error", "warning"
    output: str
    metadata: Dict[str, Any]
    processing_time: float
    confidence_score: float

@dataclass
class PipelineReport:
    """Final pipeline processing report"""
    input_file: str
    total_agents_used: int
    processing_time: float
    quality_improvement: Dict[str, float]
    agent_results: List[AgentResult]
    recommendations: List[str]
    final_score: float

class MultiAgentProposalPipeline:
    """Multi-agent proposal enhancement pipeline"""

    def __init__(self):
        print("🤖 Initializing Multi-Agent Proposal Pipeline...")

        # Initialize config and services
        self.config = Settings()

        print("   Loading LLM service...")
        self.llm_service = LLMService(self.config)

        print("   Loading context manager...")
        self.context_manager = ContextManager()

        print("   Loading agent pool...")
        self.agent_pool = AgentPool(self.llm_service, self.context_manager)

        print("   Loading hybrid RAG service...")
        self.rag_service = HybridRAGService(
            llm_service=self.llm_service,
            config=self.config
        )

        # Agent pipeline configuration
        self.pipeline_config = {
            "phase_1_evidence": {
                "agent": "enhanced_literature_analyst",
                "priority": 1,
                "dependencies": []
            },
            "phase_2_statistics": {
                "agent": "statistical_analyst",
                "priority": 2,
                "dependencies": ["phase_1_evidence"]
            },
            "phase_3_hypothesis": {
                "agent": "hypothesis_generator",
                "priority": 3,
                "dependencies": ["phase_1_evidence"]
            },
            "phase_4_writing": {
                "agent": "grant_writer",
                "priority": 4,
                "dependencies": ["phase_1_evidence", "phase_2_statistics", "phase_3_hypothesis"]
            },
            "phase_5_validation": {
                "agent": "clinical_validation_agent",
                "priority": 5,
                "dependencies": ["phase_4_writing"]
            },
            "phase_6_review": {
                "agent": "neuroscience_expert",
                "priority": 6,
                "dependencies": ["phase_4_writing", "phase_5_validation"]
            }
        }

        print("   ✅ Pipeline ready with 6 specialist agents\n")

    async def run_full_pipeline(self, input_file: str, output_file: str) -> PipelineReport:
        """Run complete multi-agent pipeline"""

        start_time = datetime.now()
        print("🚀 STARTING MULTI-AGENT PROPOSAL PIPELINE")
        print("=" * 60)
        print(f"📄 Input: {input_file}")
        print(f"📝 Output: {output_file}")
        print("=" * 60)

        # Load input proposal
        with open(input_file, 'r', encoding='utf-8') as f:
            original_text = f.read()

        print(f"📊 Original proposal: {len(original_text)} characters")

        # Calculate initial quality score
        initial_score = await self._calculate_quality_score(original_text)
        print(f"📈 Initial quality score: {initial_score:.1f}/100")

        # Create agent tasks based on pipeline configuration
        tasks = self._create_agent_tasks(original_text)

        # Execute agents in dependency order
        agent_results = []
        enhanced_content = original_text

        for phase, config in self.pipeline_config.items():
            agent_name = config["agent"]
            dependencies = config["dependencies"]

            print(f"\n🤖 Phase {phase.split('_')[1]}: {agent_name}")
            print("-" * 40)

            # Check dependencies completed
            completed_phases = [r.task_type for r in agent_results]
            if not all(dep in completed_phases for dep in dependencies):
                print(f"   ⚠️  Skipping due to missing dependencies: {dependencies}")
                continue

            # Prepare context from previous phases
            context = self._prepare_agent_context(agent_results, enhanced_content)

            # Execute agent
            try:
                result = await self._execute_agent(
                    agent_name,
                    enhanced_content,
                    context,
                    phase
                )

                agent_results.append(result)

                if result.status == "success":
                    # Update enhanced content with agent output
                    enhanced_content = self._integrate_agent_output(
                        enhanced_content,
                        result,
                        phase
                    )
                    print(f"   ✅ {agent_name} completed successfully")
                    print(f"   📊 Confidence: {result.confidence_score:.3f}")
                    print(f"   ⏱️  Time: {result.processing_time:.1f}s")
                else:
                    print(f"   ❌ {agent_name} failed: {result.output}")

            except Exception as e:
                print(f"   ❌ Error executing {agent_name}: {e}")
                # Create error result
                error_result = AgentResult(
                    agent_name=agent_name,
                    task_type=phase,
                    status="error",
                    output=str(e),
                    metadata={},
                    processing_time=0.0,
                    confidence_score=0.0
                )
                agent_results.append(error_result)

        # Calculate final quality score
        final_score = await self._calculate_quality_score(enhanced_content)
        quality_improvement = {
            "initial": initial_score,
            "final": final_score,
            "improvement": final_score - initial_score
        }

        # Save enhanced proposal
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(enhanced_content)

        # Generate recommendations
        recommendations = self._generate_recommendations(agent_results, quality_improvement)

        # Create final report
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        report = PipelineReport(
            input_file=input_file,
            total_agents_used=len([r for r in agent_results if r.status == "success"]),
            processing_time=processing_time,
            quality_improvement=quality_improvement,
            agent_results=agent_results,
            recommendations=recommendations,
            final_score=final_score
        )

        # Print final summary
        print("\n" + "=" * 60)
        print("📊 PIPELINE COMPLETION SUMMARY")
        print("=" * 60)
        print(f"📈 Quality improvement: {initial_score:.1f} → {final_score:.1f} (+{final_score-initial_score:.1f})")
        print(f"🤖 Agents used: {report.total_agents_used}/6")
        print(f"⏱️  Total time: {processing_time:.1f}s")
        print(f"💾 Enhanced proposal saved: {output_file}")

        return report

    async def run_agent_specific(self, agent_name: str, input_file: str,
                               context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """Run specific agent on proposal"""

        print(f"🎯 AGENT-SPECIFIC PROCESSING")
        print("=" * 50)
        print(f"🤖 Agent: {agent_name}")
        print(f"📄 Input: {input_file}")

        # Load input
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Execute agent
        result = await self._execute_agent(
            agent_name,
            content,
            context or {},
            f"standalone_{agent_name}"
        )

        # Print result
        print(f"\n📊 RESULT:")
        print(f"Status: {result.status}")
        print(f"Confidence: {result.confidence_score:.3f}")
        print(f"Processing time: {result.processing_time:.1f}s")
        print(f"\nOutput:\n{result.output}")

        return result

    def _create_agent_tasks(self, content: str) -> List[AgentTask]:
        """Create agent tasks from pipeline configuration"""
        tasks = []

        for phase, config in self.pipeline_config.items():
            task = AgentTask(
                agent_name=config["agent"],
                task_type=phase,
                input_data=content,
                context={},
                priority=config["priority"],
                dependencies=config["dependencies"]
            )
            tasks.append(task)

        return tasks

    async def _execute_agent(self, agent_name: str, content: str,
                           context: Dict[str, Any], task_type: str) -> AgentResult:
        """Execute specific agent"""

        start_time = datetime.now()

        try:
            # Get agent from pool
            if agent_name not in self.agent_pool.agents:
                available_agents = list(self.agent_pool.agents.keys())
                raise ValueError(f"Agent '{agent_name}' not found. Available: {available_agents}")

            agent = self.agent_pool.agents[agent_name]

            # Prepare agent-specific prompt based on task type
            prompt = self._create_agent_prompt(agent_name, content, context, task_type)

            # Execute agent
            agent_result = await agent.process_request(prompt, context)

            # Calculate processing time
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            # Extract output and confidence
            output = agent_result.get('response', 'No response')
            confidence = agent_result.get('confidence', 0.7)

            return AgentResult(
                agent_name=agent_name,
                task_type=task_type,
                status="success",
                output=output,
                metadata=agent_result,
                processing_time=processing_time,
                confidence_score=confidence
            )

        except Exception as e:
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            return AgentResult(
                agent_name=agent_name,
                task_type=task_type,
                status="error",
                output=str(e),
                metadata={"error": str(e)},
                processing_time=processing_time,
                confidence_score=0.0
            )

    def _create_agent_prompt(self, agent_name: str, content: str,
                           context: Dict[str, Any], task_type: str) -> str:
        """Create agent-specific prompt"""

        base_context = f"""
당신은 삼성미래기술육성사업 제안서를 개선하는 {agent_name} 전문가입니다.

제안서 내용:
{content}

이전 단계 결과:
{json.dumps(context, indent=2, ensure_ascii=False)}
"""

        if agent_name == "enhanced_literature_analyst":
            return f"""{base_context}

임무: DD-RAPTOR 데이터베이스의 26개 논문을 활용하여 제안서의 문헌적 근거를 강화하세요.

구체적 작업:
1. 제안서의 주요 주장에 대한 문헌적 증거 검토
2. 누락된 중요 논문 식별 (SwiFT, BrainLM 등)
3. Citation 추가 및 강화 방안 제시
4. 경쟁 연구와의 차별점 명확화

출력: 문헌 검토 결과와 개선 제안사항을 제공하세요."""

        elif agent_name == "statistical_analyst":
            return f"""{base_context}

임무: 제안서의 통계적 타당성을 검증하고 개선방안을 제시하세요.

구체적 작업:
1. 샘플 사이즈 적절성 검토 (현재 n=3,000)
2. 검정력 분석 (Power Analysis)
3. 통계적 가설 설정의 적절성
4. WES vs SNP array 선택의 통계적 근거
5. Effect size 추정의 현실성

출력: 통계적 검토 결과와 구체적 개선안을 제공하세요."""

        elif agent_name == "hypothesis_generator":
            return f"""{base_context}

임무: 혁신적이면서 검증 가능한 연구 가설을 생성하세요.

구체적 작업:
1. 현재 가설의 혁신성 평가
2. Brain-genomics connection 구체화
3. 검증 가능한 세부 가설 생성
4. 국제 경쟁력 확보 방안

출력: 개선된 연구 가설과 검증 전략을 제공하세요."""

        elif agent_name == "grant_writer":
            return f"""{base_context}

임무: 삼성미래기술육성사업 양식에 맞는 제안서를 작성하세요.

구체적 작업:
1. 삼성 평가 기준에 최적화
2. 혁신성, 실현가능성, 파급효과 강조
3. 예산 2.5억원 제약 내 최적화
4. 한국 상황에 특화된 내용 강화

출력: 삼성 양식에 맞는 완성된 제안서를 제공하세요."""

        elif agent_name == "clinical_validation_agent":
            return f"""{base_context}

임무: 임상 적용 가능성과 규제 승인 경로를 검증하세요.

구체적 작업:
1. FDA/KFDA 의료기기 승인 경로 분석
2. 임상시험 설계의 현실성
3. IRB 승인 가능성
4. 임상 파트너십 전략

출력: 임상 검증 전략과 현실적 구현 방안을 제공하세요."""

        elif agent_name == "neuroscience_expert":
            return f"""{base_context}

임무: 뇌과학 전문가로서 최종 검토와 개선사항을 제시하세요.

구체적 작업:
1. 뇌과학적 타당성 검토
2. 최신 뇌영상 기술 동향 반영
3. 국제 연구 경쟁력 평가
4. 전체적 과학적 엄밀성 평가

출력: 뇌과학 전문가 관점의 최종 검토 의견을 제공하세요."""

        else:
            return f"""{base_context}

임무: 제안서 개선을 위한 전문적 검토를 수행하세요.
출력: 구체적 개선사항과 권장사항을 제공하세요."""

    def _prepare_agent_context(self, previous_results: List[AgentResult],
                             current_content: str) -> Dict[str, Any]:
        """Prepare context from previous agent results"""

        context = {
            "current_content_length": len(current_content),
            "previous_agents": []
        }

        for result in previous_results:
            if result.status == "success":
                context["previous_agents"].append({
                    "agent": result.agent_name,
                    "confidence": result.confidence_score,
                    "key_output": result.output[:500] + "..." if len(result.output) > 500 else result.output
                })

        return context

    def _integrate_agent_output(self, current_content: str, result: AgentResult,
                              phase: str) -> str:
        """Integrate agent output into proposal"""

        # For now, append agent output as comments
        # In a full implementation, this would be more sophisticated
        integration_note = f"""

<!-- {result.agent_name.upper()} OUTPUT ({phase}) -->
<!-- Confidence: {result.confidence_score:.3f} -->
{result.output}
<!-- END {result.agent_name.upper()} OUTPUT -->

"""

        return current_content + integration_note

    async def _calculate_quality_score(self, content: str) -> float:
        """Calculate proposal quality score"""

        # Simple heuristic scoring (in full implementation, use ML model)
        score = 50.0  # Base score

        # Length and structure
        if len(content) > 10000:
            score += 10
        elif len(content) > 5000:
            score += 5

        # Citations
        citation_count = len([line for line in content.split('\n') if '[' in line and ']' in line])
        score += min(citation_count * 2, 20)

        # Technical terms
        technical_terms = [
            'foundation model', 'transformer', 'fMRI', 'DTI',
            'genomics', 'GWAS', 'deep learning', 'neural network'
        ]
        content_lower = content.lower()
        technical_score = sum(5 for term in technical_terms if term in content_lower)
        score += min(technical_score, 20)

        return min(score, 100.0)

    def _generate_recommendations(self, agent_results: List[AgentResult],
                                quality_improvement: Dict[str, float]) -> List[str]:
        """Generate final recommendations"""

        recommendations = []

        # Quality improvement assessment
        improvement = quality_improvement["improvement"]
        if improvement >= 20:
            recommendations.append("✅ Excellent quality improvement achieved")
        elif improvement >= 10:
            recommendations.append("🟡 Good quality improvement, consider additional refinements")
        else:
            recommendations.append("🔴 Limited improvement, major revision needed")

        # Agent-specific recommendations
        successful_agents = [r for r in agent_results if r.status == "success"]
        failed_agents = [r for r in agent_results if r.status != "success"]

        if len(successful_agents) >= 5:
            recommendations.append("🤖 Multi-agent pipeline executed successfully")
        else:
            recommendations.append(f"⚠️ Only {len(successful_agents)}/6 agents completed successfully")

        if failed_agents:
            failed_names = [r.agent_name for r in failed_agents]
            recommendations.append(f"🔧 Retry failed agents: {', '.join(failed_names)}")

        # Confidence assessment
        avg_confidence = sum(r.confidence_score for r in successful_agents) / len(successful_agents) if successful_agents else 0
        if avg_confidence >= 0.8:
            recommendations.append("🎯 High confidence results achieved")
        elif avg_confidence >= 0.6:
            recommendations.append("🟡 Moderate confidence, validate key claims")
        else:
            recommendations.append("⚠️ Low confidence, major revision recommended")

        return recommendations

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Proposal Enhancement Pipeline"
    )

    parser.add_argument(
        "--mode",
        choices=["full_pipeline", "agent_specific"],
        required=True,
        help="Processing mode"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input proposal file"
    )

    parser.add_argument(
        "--output",
        help="Output file (required for full_pipeline)"
    )

    parser.add_argument(
        "--agent",
        choices=[
            "enhanced_literature_analyst",
            "statistical_analyst",
            "hypothesis_generator",
            "grant_writer",
            "clinical_validation_agent",
            "neuroscience_expert"
        ],
        help="Specific agent to run (for agent_specific mode)"
    )

    parser.add_argument(
        "--report",
        help="Save pipeline report (JSON)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.mode == "full_pipeline" and not args.output:
        print("❌ Full pipeline mode requires --output")
        return

    if args.mode == "agent_specific" and not args.agent:
        print("❌ Agent-specific mode requires --agent")
        return

    if not Path(args.input).exists():
        print(f"❌ Input file not found: {args.input}")
        return

    async def run_pipeline():
        try:
            # Initialize pipeline
            pipeline = MultiAgentProposalPipeline()

            if args.mode == "full_pipeline":
                # Run full pipeline
                report = await pipeline.run_full_pipeline(args.input, args.output)

                # Save report if requested
                if args.report:
                    report_dict = asdict(report)
                    with open(args.report, 'w', encoding='utf-8') as f:
                        json.dump(report_dict, f, indent=2, ensure_ascii=False)
                    print(f"📊 Pipeline report saved: {args.report}")

            else:
                # Run specific agent
                result = await pipeline.run_agent_specific(args.agent, args.input)

                # Save result if report requested
                if args.report:
                    result_dict = asdict(result)
                    with open(args.report, 'w', encoding='utf-8') as f:
                        json.dump(result_dict, f, indent=2, ensure_ascii=False)
                    print(f"📊 Agent result saved: {args.report}")

        except Exception as e:
            print(f"❌ Pipeline error: {e}")
            import traceback
            traceback.print_exc()

    # Run async pipeline
    asyncio.run(run_pipeline())

if __name__ == "__main__":
    main()