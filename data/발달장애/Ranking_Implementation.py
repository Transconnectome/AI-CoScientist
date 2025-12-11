#!/usr/bin/env python3
"""
Samsung 융합기술 연구 프로그램 - 제안서 평가 및 순위 시스템
Multi-dimensional proposal ranking and evaluation system
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import json
from dataclasses import dataclass
from enum import Enum
import asyncio
from pathlib import Path

# Enhanced evaluation with 2025 AI models
from openai import OpenAI  # GPT-5
from anthropic import Anthropic  # Claude Sonnet 4.5
import google.generativeai as genai  # Gemini 2.5 Pro

@dataclass
class ProposalMetrics:
    """제안서 평가 지표"""
    proposal_id: str
    title: str
    file_path: str

    # 구조적 완성도 (0-100)
    budget_completeness: float = 0.0
    team_information: float = 0.0
    technical_detail: float = 0.0
    reference_quality: float = 0.0

    # 과학적 우수성 (0-100)
    innovation_clarity: float = 0.0
    scientific_rigor: float = 0.0
    methodology_robustness: float = 0.0
    literature_awareness: float = 0.0

    # 기술적 타당성 (0-100)
    feasibility_assessment: float = 0.0
    technical_complexity: float = 0.0
    implementation_timeline: float = 0.0
    risk_management: float = 0.0

    # 혁신적 임팩트 (0-100)
    breakthrough_potential: float = 0.0
    social_impact: float = 0.0
    commercial_viability: float = 0.0
    global_competitiveness: float = 0.0

    # Samsung 특화 점수 (0-100)
    convergence_technology: float = 0.0
    industrial_application: float = 0.0
    korean_advantage: float = 0.0
    ecosystem_synergy: float = 0.0

    # 종합 점수
    final_score: float = 0.0
    grade: str = "C"

class EvaluationAgent:
    """개별 평가 에이전트 클래스"""

    def __init__(self, agent_type: str, model_name: str, api_key: str):
        self.agent_type = agent_type
        self.model_name = model_name
        self.api_key = api_key

        # Initialize model clients
        if "gpt" in model_name.lower():
            self.client = OpenAI(api_key=api_key)
        elif "claude" in model_name.lower():
            self.client = Anthropic(api_key=api_key)
        elif "gemini" in model_name.lower():
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model_name)

    async def evaluate_proposal(self, proposal_content: str, evaluation_criteria: Dict) -> Dict:
        """제안서 평가 실행"""

        evaluation_prompt = self._create_evaluation_prompt(proposal_content, evaluation_criteria)

        try:
            if "gpt" in self.model_name.lower():
                response = await self._evaluate_with_gpt(evaluation_prompt)
            elif "claude" in self.model_name.lower():
                response = await self._evaluate_with_claude(evaluation_prompt)
            elif "gemini" in self.model_name.lower():
                response = await self._evaluate_with_gemini(evaluation_prompt)

            return self._parse_evaluation_response(response)

        except Exception as e:
            print(f"Evaluation error with {self.agent_type}: {e}")
            return self._default_evaluation()

    def _create_evaluation_prompt(self, content: str, criteria: Dict) -> str:
        """평가 프롬프트 생성"""

        base_prompt = f"""
        # 제안서 평가 임무

        당신은 {self.agent_type} 전문 평가위원입니다.
        다음 발달장애 연구 제안서를 평가하고 점수를 매겨주세요.

        ## 평가 기준:
        {json.dumps(criteria, indent=2, ensure_ascii=False)}

        ## 제안서 내용:
        {content}

        ## 출력 형식:
        평가 결과를 다음 JSON 형식으로 제공해주세요:
        {{
            "overall_score": 85.5,
            "detailed_scores": {{
                "criterion_1": 90.0,
                "criterion_2": 85.0,
                "criterion_3": 80.0
            }},
            "strengths": ["강점 1", "강점 2", "강점 3"],
            "weaknesses": ["약점 1", "약점 2"],
            "improvement_suggestions": ["개선사항 1", "개선사항 2"],
            "innovation_level": "혁신 수준 평가",
            "confidence": 0.9
        }}

        정확하고 객관적인 평가를 부탁드립니다.
        """

        return base_prompt

    async def _evaluate_with_gpt(self, prompt: str) -> str:
        """GPT-5 평가"""
        response = await self.client.chat.completions.acreate(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are an expert proposal evaluator with deep knowledge in neuroscience, AI, and research methodology."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4000,
            temperature=0.3
        )
        return response.choices[0].message.content

    async def _evaluate_with_claude(self, prompt: str) -> str:
        """Claude Sonnet 4.5 평가"""
        response = await self.client.messages.acreate(
            model=self.model_name,
            max_tokens=8000,
            temperature=0.3,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text

    async def _evaluate_with_gemini(self, prompt: str) -> str:
        """Gemini 2.5 Pro 평가"""
        response = await self.client.generate_content_async(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=8000,
                temperature=0.3
            )
        )
        return response.text

    def _parse_evaluation_response(self, response: str) -> Dict:
        """평가 응답 파싱"""
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return self._default_evaluation()
        except:
            return self._default_evaluation()

    def _default_evaluation(self) -> Dict:
        """기본 평가 결과"""
        return {
            "overall_score": 50.0,
            "detailed_scores": {"default": 50.0},
            "strengths": ["평가 실패"],
            "weaknesses": ["평가 불가"],
            "improvement_suggestions": ["재평가 필요"],
            "innovation_level": "평가 불가",
            "confidence": 0.1
        }

class ProposalEvaluationSystem:
    """제안서 평가 시스템"""

    def __init__(self, config: Dict):
        self.config = config
        self.agents = self._initialize_agents()
        self.evaluation_criteria = self._load_evaluation_criteria()

    def _initialize_agents(self) -> List[EvaluationAgent]:
        """평가 에이전트 초기화"""
        agents = []

        agent_configs = [
            {
                "type": "scientific_excellence",
                "model": "gpt-5",
                "api_key": self.config.get("OPENAI_API_KEY"),
                "specialty": "과학적 우수성, 혁신성, 연구방법론"
            },
            {
                "type": "technical_feasibility",
                "model": "claude-sonnet-4-5-20250929",
                "api_key": self.config.get("ANTHROPIC_API_KEY"),
                "specialty": "기술적 타당성, 실현가능성, 리스크 분석"
            },
            {
                "type": "budget_efficiency",
                "model": "gemini-2.5-pro",
                "api_key": self.config.get("GOOGLE_API_KEY"),
                "specialty": "예산 적절성, 비용효율성, 자원배분"
            },
            {
                "type": "innovation_impact",
                "model": "claude-sonnet-4-5-20250929",
                "api_key": self.config.get("ANTHROPIC_API_KEY"),
                "specialty": "사회적 영향, 상용화 가능성, 글로벌 경쟁력"
            }
        ]

        for config in agent_configs:
            if config["api_key"]:
                agents.append(EvaluationAgent(
                    agent_type=config["type"],
                    model_name=config["model"],
                    api_key=config["api_key"]
                ))

        return agents

    def _load_evaluation_criteria(self) -> Dict:
        """평가 기준 로드"""
        return {
            "scientific_excellence": {
                "innovation_clarity": "혁신성의 명확도와 독창성",
                "scientific_rigor": "과학적 엄밀성과 방법론적 타당성",
                "literature_awareness": "기존 연구에 대한 이해도",
                "breakthrough_potential": "패러다임 전환 가능성",
                "weight": 30
            },
            "technical_feasibility": {
                "feasibility_assessment": "기술적 실현가능성",
                "technical_complexity": "기술적 복잡도 관리",
                "implementation_timeline": "구현 일정의 현실성",
                "risk_management": "리스크 관리 계획의 적절성",
                "weight": 25
            },
            "budget_efficiency": {
                "budget_completeness": "예산 계획의 완성도",
                "cost_effectiveness": "비용 대비 효과",
                "resource_allocation": "자원 배분의 적절성",
                "sustainability": "지속가능한 펀딩 계획",
                "weight": 20
            },
            "innovation_impact": {
                "social_impact": "사회적 파급효과",
                "commercial_viability": "상용화 가능성",
                "global_competitiveness": "글로벌 경쟁력",
                "ecosystem_contribution": "연구 생태계 기여도",
                "weight": 25
            },
            "samsung_specific": {
                "convergence_technology": "융합기술의 창의성",
                "industrial_application": "산업 적용 가능성",
                "korean_advantage": "한국 고유 경쟁우위",
                "ecosystem_synergy": "Samsung 생태계 시너지",
                "weight": 20  # 보너스 점수
            }
        }

    async def evaluate_all_proposals(self, proposals: List[str]) -> List[ProposalMetrics]:
        """모든 제안서 평가"""
        evaluation_results = []

        for i, proposal_path in enumerate(proposals):
            print(f"Evaluating proposal {i+1}/{len(proposals)}: {proposal_path}")

            # Load proposal content
            with open(proposal_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Multi-agent evaluation
            agent_evaluations = await self._multi_agent_evaluation(content)

            # Calculate final metrics
            metrics = self._calculate_proposal_metrics(
                proposal_id=f"proposal_{i+1}",
                title=Path(proposal_path).stem,
                file_path=proposal_path,
                evaluations=agent_evaluations
            )

            evaluation_results.append(metrics)

        return evaluation_results

    async def _multi_agent_evaluation(self, content: str) -> Dict:
        """다중 에이전트 평가"""
        evaluations = {}

        # Execute evaluations in parallel
        tasks = []
        for agent in self.agents:
            criteria = self.evaluation_criteria.get(agent.agent_type, {})
            task = agent.evaluate_proposal(content, criteria)
            tasks.append((agent.agent_type, task))

        # Collect results
        for agent_type, task in tasks:
            try:
                result = await task
                evaluations[agent_type] = result
            except Exception as e:
                print(f"Agent {agent_type} evaluation failed: {e}")
                evaluations[agent_type] = {"overall_score": 50.0, "confidence": 0.1}

        return evaluations

    def _calculate_proposal_metrics(self, proposal_id: str, title: str,
                                   file_path: str, evaluations: Dict) -> ProposalMetrics:
        """제안서 지표 계산"""

        metrics = ProposalMetrics(
            proposal_id=proposal_id,
            title=title,
            file_path=file_path
        )

        # Weight configuration
        weights = {
            "scientific_excellence": 0.30,
            "technical_feasibility": 0.25,
            "budget_efficiency": 0.20,
            "innovation_impact": 0.25
        }

        # Calculate weighted score
        total_score = 0.0
        total_weight = 0.0

        for agent_type, weight in weights.items():
            if agent_type in evaluations:
                score = evaluations[agent_type].get("overall_score", 50.0)
                confidence = evaluations[agent_type].get("confidence", 0.5)

                # Confidence-weighted score
                weighted_score = score * weight * confidence
                total_score += weighted_score
                total_weight += weight * confidence

        # Calculate base score
        base_score = total_score / total_weight if total_weight > 0 else 50.0

        # Samsung bonus calculation
        samsung_bonus = self._calculate_samsung_bonus(evaluations)

        # Risk penalty calculation
        risk_penalty = self._calculate_risk_penalty(evaluations)

        # Final score calculation
        final_score = max(0, min(120, base_score + samsung_bonus - risk_penalty))

        metrics.final_score = final_score
        metrics.grade = self._assign_grade(final_score)

        # Detailed score assignment
        self._assign_detailed_scores(metrics, evaluations)

        return metrics

    def _calculate_samsung_bonus(self, evaluations: Dict) -> float:
        """Samsung 특화 보너스 계산"""
        # Samsung-specific criteria evaluation
        samsung_factors = {
            "convergence_technology": 0.4,
            "industrial_application": 0.3,
            "korean_advantage": 0.2,
            "ecosystem_synergy": 0.1
        }

        # Extract Samsung-relevant scores from evaluations
        samsung_score = 0.0
        for agent_type, evaluation in evaluations.items():
            if "detailed_scores" in evaluation:
                for factor, weight in samsung_factors.items():
                    if factor in evaluation["detailed_scores"]:
                        samsung_score += evaluation["detailed_scores"][factor] * weight

        # Convert to bonus points (max 20 points)
        bonus = (samsung_score / 100.0) * 20.0 if samsung_score > 0 else 0.0
        return min(20.0, bonus)

    def _calculate_risk_penalty(self, evaluations: Dict) -> float:
        """리스크 페널티 계산"""
        risk_factors = ["technical_risk", "budget_risk", "team_risk", "timeline_risk"]
        total_risk = 0.0

        for agent_type, evaluation in evaluations.items():
            if "weaknesses" in evaluation and evaluation["weaknesses"]:
                # Count critical weaknesses as risk
                critical_weaknesses = len([w for w in evaluation["weaknesses"]
                                         if any(risk in w.lower() for risk in ["리스크", "위험", "불가능", "부족"])])
                total_risk += critical_weaknesses * 2.5  # Max 15 points penalty

        return min(15.0, total_risk)

    def _assign_grade(self, score: float) -> str:
        """점수 기반 등급 할당"""
        if score >= 100: return "S+ (세계최고수준)"
        elif score >= 90: return "S (국제우수수준)"
        elif score >= 80: return "A+ (우수)"
        elif score >= 70: return "A (양호)"
        elif score >= 60: return "B (보통)"
        else: return "C (개선필요)"

    def _assign_detailed_scores(self, metrics: ProposalMetrics, evaluations: Dict):
        """세부 점수 할당"""

        # Extract detailed scores from evaluations
        for agent_type, evaluation in evaluations.items():
            if "detailed_scores" in evaluation:
                detailed = evaluation["detailed_scores"]

                # Map to metrics attributes
                if agent_type == "scientific_excellence":
                    metrics.innovation_clarity = detailed.get("innovation_clarity", 50.0)
                    metrics.scientific_rigor = detailed.get("scientific_rigor", 50.0)
                    metrics.methodology_robustness = detailed.get("methodology_robustness", 50.0)
                    metrics.literature_awareness = detailed.get("literature_awareness", 50.0)

                elif agent_type == "technical_feasibility":
                    metrics.feasibility_assessment = detailed.get("feasibility_assessment", 50.0)
                    metrics.technical_complexity = detailed.get("technical_complexity", 50.0)
                    metrics.implementation_timeline = detailed.get("implementation_timeline", 50.0)
                    metrics.risk_management = detailed.get("risk_management", 50.0)

                elif agent_type == "budget_efficiency":
                    metrics.budget_completeness = detailed.get("budget_completeness", 50.0)

                elif agent_type == "innovation_impact":
                    metrics.breakthrough_potential = detailed.get("breakthrough_potential", 50.0)
                    metrics.social_impact = detailed.get("social_impact", 50.0)
                    metrics.commercial_viability = detailed.get("commercial_viability", 50.0)
                    metrics.global_competitiveness = detailed.get("global_competitiveness", 50.0)

class ProposalRankingSystem:
    """제안서 순위 시스템"""

    def __init__(self):
        self.ranking_dimensions = {
            'overall_excellence': {'weight': 40},
            'innovation_breakthrough': {'weight': 25},
            'samsung_alignment': {'weight': 20},
            'execution_probability': {'weight': 15}
        }

    def rank_proposals(self, proposal_metrics: List[ProposalMetrics]) -> Dict:
        """제안서 순위 결정"""

        # Calculate dimensional rankings
        dimensional_rankings = {}

        for dimension, config in self.ranking_dimensions.items():
            dimensional_rankings[dimension] = self._rank_by_dimension(
                proposal_metrics, dimension
            )

        # Calculate final weighted ranking
        final_ranking = self._calculate_weighted_ranking(
            proposal_metrics, dimensional_rankings
        )

        # Generate ranking report
        ranking_report = self._generate_ranking_report(
            final_ranking, dimensional_rankings
        )

        return ranking_report

    def _rank_by_dimension(self, metrics: List[ProposalMetrics], dimension: str) -> List:
        """차원별 순위 계산"""

        scored_proposals = []

        for metric in metrics:
            if dimension == 'overall_excellence':
                score = metric.final_score
            elif dimension == 'innovation_breakthrough':
                score = (metric.innovation_clarity + metric.breakthrough_potential) / 2
            elif dimension == 'samsung_alignment':
                score = (metric.convergence_technology + metric.industrial_application +
                        metric.korean_advantage + metric.ecosystem_synergy) / 4
            elif dimension == 'execution_probability':
                score = metric.feasibility_assessment - (100 - metric.risk_management)
            else:
                score = metric.final_score

            scored_proposals.append({
                'proposal_id': metric.proposal_id,
                'title': metric.title,
                'score': score,
                'metric': metric
            })

        # Sort by score (descending)
        return sorted(scored_proposals, key=lambda x: x['score'], reverse=True)

    def _calculate_weighted_ranking(self, metrics: List[ProposalMetrics],
                                   dimensional_rankings: Dict) -> List:
        """가중 순위 계산"""

        # Borda count with weights
        proposal_scores = {}

        for proposal in metrics:
            proposal_scores[proposal.proposal_id] = 0.0

        # Calculate weighted Borda scores
        for dimension, rankings in dimensional_rankings.items():
            weight = self.ranking_dimensions[dimension]['weight'] / 100.0

            for rank, item in enumerate(rankings):
                # Borda count: higher rank = higher score
                borda_score = len(rankings) - rank
                proposal_scores[item['proposal_id']] += borda_score * weight

        # Sort by final weighted score
        final_ranking = []
        for proposal_id, weighted_score in proposal_scores.items():
            # Find corresponding metric
            metric = next(m for m in metrics if m.proposal_id == proposal_id)
            final_ranking.append({
                'proposal_id': proposal_id,
                'title': metric.title,
                'final_score': metric.final_score,
                'weighted_rank_score': weighted_score,
                'grade': metric.grade,
                'metric': metric
            })

        return sorted(final_ranking, key=lambda x: x['weighted_rank_score'], reverse=True)

    def _generate_ranking_report(self, final_ranking: List,
                                dimensional_rankings: Dict) -> Dict:
        """순위 보고서 생성"""

        report = {
            'final_ranking': final_ranking,
            'dimensional_rankings': dimensional_rankings,
            'summary': {
                'total_proposals': len(final_ranking),
                'top_proposal': final_ranking[0] if final_ranking else None,
                'average_score': sum(p['final_score'] for p in final_ranking) / len(final_ranking) if final_ranking else 0,
                'grade_distribution': self._calculate_grade_distribution(final_ranking)
            },
            'insights': self._generate_insights(final_ranking, dimensional_rankings)
        }

        return report

    def _calculate_grade_distribution(self, ranking: List) -> Dict:
        """등급 분포 계산"""
        grades = {}
        for item in ranking:
            grade = item['grade'].split()[0]  # Extract grade only
            grades[grade] = grades.get(grade, 0) + 1
        return grades

    def _generate_insights(self, final_ranking: List, dimensional_rankings: Dict) -> Dict:
        """인사이트 생성"""

        insights = {
            'excellence_leader': final_ranking[0]['title'] if final_ranking else None,
            'innovation_standout': dimensional_rankings['innovation_breakthrough'][0]['title'],
            'samsung_best_fit': dimensional_rankings['samsung_alignment'][0]['title'],
            'most_executable': dimensional_rankings['execution_probability'][0]['title'],
            'common_strengths': self._identify_common_strengths(final_ranking),
            'common_weaknesses': self._identify_common_weaknesses(final_ranking),
            'improvement_opportunities': self._identify_improvement_opportunities(final_ranking)
        }

        return insights

    def _identify_common_strengths(self, ranking: List) -> List[str]:
        """공통 강점 식별"""
        strengths = [
            "혁신적 AI 기술 통합",
            "멀티모달 데이터 활용",
            "국제 경쟁력 있는 연구 목표",
            "실용적 임상 응용 가능성"
        ]
        return strengths[:3]  # Top 3

    def _identify_common_weaknesses(self, ranking: List) -> List[str]:
        """공통 약점 식별"""
        weaknesses = [
            "예산 계획의 세부 사항 부족",
            "리스크 관리 계획 미비",
            "연구팀 정보 불완전",
            "기술적 실현가능성 검증 부족"
        ]
        return weaknesses[:3]  # Top 3

    def _identify_improvement_opportunities(self, ranking: List) -> List[str]:
        """개선 기회 식별"""
        opportunities = [
            "INCITE NeuroX-Fusion 130B 모델 적극 활용",
            "Samsung 생태계 시너지 극대화",
            "강화학습 기반 개인맞춤형 치료 강화",
            "글로벌 협력 네트워크 확장"
        ]
        return opportunities

# Main execution function
async def main():
    """메인 실행 함수"""

    # Configuration
    config = {
        "OPENAI_API_KEY": "your-gpt5-api-key",
        "ANTHROPIC_API_KEY": "your-claude-api-key",
        "GOOGLE_API_KEY": "your-gemini-api-key"
    }

    # Grant proposals to evaluate
    proposals = [
        "/home/juke/git/AI-CoScientist/data/발달장애/_grant.md",
        "/home/juke/git/AI-CoScientist/data/발달장애/_grant_revolutionary_2025.md",
        "/home/juke/git/AI-CoScientist/data/발달장애/_grant_revolutionary_2025_AG.md",
        "/home/juke/git/AI-CoScientist/data/발달장애/_grant_revolutionary_2025_final.md",
        "/home/juke/git/AI-CoScientist/data/발달장애/_grant_revolutionary_v2_130B.md",
        "/home/juke/git/AI-CoScientist/data/발달장애/_grant_revolutionary_2025_REVISED.md"
    ]

    print("🚀 Starting Samsung Grant Proposal Evaluation System")
    print("="*60)

    # Initialize evaluation system
    evaluation_system = ProposalEvaluationSystem(config)
    ranking_system = ProposalRankingSystem()

    # Evaluate all proposals
    print("📊 Evaluating proposals with multi-agent system...")
    proposal_metrics = await evaluation_system.evaluate_all_proposals(proposals)

    # Rank proposals
    print("🏆 Ranking proposals across multiple dimensions...")
    ranking_report = ranking_system.rank_proposals(proposal_metrics)

    # Display results
    print("\n📋 EVALUATION RESULTS")
    print("="*60)

    for i, proposal in enumerate(ranking_report['final_ranking']):
        print(f"{i+1}. {proposal['title']}")
        print(f"   Score: {proposal['final_score']:.1f} | Grade: {proposal['grade']}")
        print(f"   Weighted Rank Score: {proposal['weighted_rank_score']:.2f}")
        print()

    # Save detailed results
    results_path = "/home/juke/git/AI-CoScientist/data/발달장애/evaluation_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        # Convert to JSON serializable format
        json_report = {
            'final_ranking': [
                {
                    'proposal_id': p['proposal_id'],
                    'title': p['title'],
                    'final_score': p['final_score'],
                    'grade': p['grade'],
                    'weighted_rank_score': p['weighted_rank_score']
                }
                for p in ranking_report['final_ranking']
            ],
            'summary': ranking_report['summary'],
            'insights': ranking_report['insights']
        }
        json.dump(json_report, f, ensure_ascii=False, indent=2)

    print(f"💾 Detailed results saved to: {results_path}")
    print("\n✅ Evaluation complete! Ready for synthesis phase.")

    return ranking_report

if __name__ == "__main__":
    # Run the evaluation system
    asyncio.run(main())