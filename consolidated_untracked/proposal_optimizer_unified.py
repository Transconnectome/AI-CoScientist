#!/usr/bin/env python3
"""
🚀 Unified RAG Proposal Optimizer - Next-Generation Workflow Automation
================================================================

Advanced 5-step optimization workflow powered by Unified RAG Orchestrator
Replacing DD-RAPTOR with 6-strategy intelligent RAG orchestration

Enhanced Features:
- Cross-domain knowledge synthesis (ESM3, Neuroscience, Quantum ML, Grant proposals)
- 6-strategy RAG orchestration (HYBRID, GRAPH_RAG, ENHANCED_DD_RAPTOR, etc.)
- Intelligent query classification and strategy routing
- Real-time multi-modal evidence validation
- Advanced quality assessment with cross-domain insights

Workflow Steps:
1. 🔍 Unified Evidence Mapping & Cross-Domain Analysis
2. ⚡ Real-time Multi-Strategy Claim Validation
3. 📚 Advanced RAG Query & Multi-Modal Literature Review
4. 🤖 Multi-Agent Enhancement with RAG Integration
5. ✅ Intelligent Citation & Quality Finalization

Usage Examples:
    # Full unified optimization (recommended)
    poetry run python scripts/proposal_optimizer_unified.py optimize \\
        --input "proposal.md" --mode full --enable-cross-domain

    # Quick unified improvement (ESM3 + Grant knowledge)
    poetry run python scripts/proposal_optimizer_unified.py optimize \\
        --input "proposal.md" --mode quick --strategies "HYBRID,GRAPH_RAG"

    # Advanced multi-domain synthesis
    poetry run python scripts/proposal_optimizer_unified.py optimize \\
        --input "proposal.md" --mode research --domains "neuroscience,protein_research,quantum_ml"

    # Interactive unified wizard
    poetry run python scripts/proposal_optimizer_unified.py wizard --unified-rag
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Enhanced imports for Unified RAG support
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UnifiedProposalOptimizer:
    """Unified RAG 기반 제안서 최적화 시스템"""

    def __init__(self):
        """초기화"""
        self.base_dir = Path(__file__).parent.parent
        self.scripts_dir = Path(__file__).parent
        self.output_dir = Path("output/optimized_proposals_unified")
        self.output_dir.mkdir(exist_ok=True)

        # Unified RAG 워크플로우 단계 정의 (DD-RAPTOR 단계 업그레이드)
        self.steps = {
            1: {
                "name": "Unified Evidence Mapping & Cross-Domain Analysis",
                "emoji": "🔍",
                "script": "map_proposal_to_unified_evidence.py",  # Updated script
                "description": "다중 전략 RAG를 통한 과학적 주장 분석 및 cross-domain 근거 강도 평가",
                "estimated_time": "3-5분",
                "strategies": ["HYBRID", "GRAPH_RAG", "ENHANCED_DD_RAPTOR"],
                "domains": ["neuroscience", "quantum_ml", "protein_research", "general"]
            },
            2: {
                "name": "Real-time Multi-Strategy Claim Validation",
                "emoji": "⚡",
                "script": "validate_claims_unified_rag.py",  # Enhanced validation
                "description": "Unified RAG 기반 실시간 주장 검증 및 지능적 자동 수정",
                "estimated_time": "10-30분",
                "strategies": ["GOLDEN_REFERENCE", "HYBRID", "SIMPLE_RAG"],
                "cross_modal": True
            },
            3: {
                "name": "Advanced RAG Query & Multi-Modal Literature Review",
                "emoji": "📚",
                "script": "advanced_unified_query.py",  # Replaced enhanced_dd_query.py
                "description": "6-strategy RAG orchestration을 통한 체계적 문헌 검토 및 ESM3/Grant 근거 수집",
                "estimated_time": "5-15분",
                "strategies": ["GRAPH_RAG", "MULTIMODAL_RAG", "PSYCHOLOGY_RAG"],
                "knowledge_synthesis": True
            },
            4: {
                "name": "Multi-Agent Enhancement with Unified RAG",
                "emoji": "🤖",
                "script": "multi_agent_unified_pipeline.py",  # Enhanced multi-agent
                "description": "6개 전문 AI 에이전트 + Unified RAG 협업 개선",
                "estimated_time": "10-20분",
                "agent_rag_integration": True
            },
            5: {
                "name": "Intelligent Citation & Unified Quality Finalization",
                "emoji": "✅",
                "script": "unified_citation_generator.py",  # Enhanced citation
                "description": "Unified RAG 기반 자동 인용 생성 및 multi-domain 품질 최종 검증",
                "estimated_time": "5-10분",
                "final_validation": True
            }
        }

        # Unified RAG 특화 모드
        self.modes = {
            "full": {
                "steps": [1, 2, 3, 4, 5],
                "description": "전체 Unified RAG 최적화 (모든 전략 활용)",
                "estimated_time": "30-70분",
                "strategies": "ALL",
                "cross_domain_synthesis": True
            },
            "quick": {
                "steps": [1, 3, 5],
                "description": "빠른 Unified RAG 개선 (핵심 전략만)",
                "estimated_time": "15-30분",
                "strategies": ["HYBRID", "GRAPH_RAG"],
                "focus": "efficiency"
            },
            "research": {
                "steps": [1, 2, 3, 4],
                "description": "연구 집중 모드 (문헌 강화, 에이전트 협업)",
                "estimated_time": "25-50분",
                "strategies": ["GRAPH_RAG", "ENHANCED_DD_RAPTOR", "GOLDEN_REFERENCE"],
                "deep_research": True
            },
            "validation": {
                "steps": [1, 2, 5],
                "description": "검증 집중 모드 (주장 검증, 품질 확보)",
                "estimated_time": "20-40분",
                "strategies": ["GOLDEN_REFERENCE", "HYBRID"],
                "validation_focus": True
            },
            "cross_domain": {
                "steps": [1, 3, 4, 5],
                "description": "교차 도메인 합성 (ESM3 + 뇌과학 + 양자ML)",
                "estimated_time": "30-60분",
                "strategies": ["GRAPH_RAG", "MULTIMODAL_RAG", "HYBRID"],
                "domains": ["neuroscience", "protein_research", "quantum_ml"]
            }
        }

        # Unified RAG 성능 추적
        self.optimization_stats = {
            "total_optimizations": 0,
            "strategy_performance": {},
            "cross_domain_successes": 0,
            "quality_improvements": []
        }

    def optimize_proposal(self,
                         input_file: str,
                         mode: str = "full",
                         enable_cross_domain: bool = True,
                         preferred_strategies: Optional[List[str]] = None,
                         target_domains: Optional[List[str]] = None) -> Dict[str, Any]:
        """제안서 최적화 메인 함수 - Unified RAG 기반"""

        logger.info(f"🚀 Starting Unified RAG Proposal Optimization")
        logger.info(f"📄 Input: {input_file}")
        logger.info(f"🎯 Mode: {mode}")
        logger.info(f"🌐 Cross-domain: {enable_cross_domain}")

        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # 최적화 세션 준비
        session_id = self._create_session_id()
        session_dir = self.output_dir / session_id
        session_dir.mkdir(exist_ok=True)

        # 초기 파일 품질 평가 (Unified RAG)
        initial_score = self._calculate_unified_quality_score(str(input_path))

        execution_log = {
            "session_id": session_id,
            "input_file": str(input_path),
            "mode": mode,
            "initial_score": initial_score,
            "unified_rag_config": {
                "cross_domain_enabled": enable_cross_domain,
                "preferred_strategies": preferred_strategies,
                "target_domains": target_domains
            },
            "steps_completed": [],
            "strategy_performance": {},
            "started_at": datetime.now().isoformat(),
            "completed_at": None,
            "total_time": 0
        }

        print(f"\n" + "="*70)
        print("🎯 UNIFIED RAG PROPOSAL OPTIMIZATION")
        print("="*70)
        print(f"📄 파일: {input_path.name}")
        print(f"🔍 모드: {self.modes[mode]['description']}")
        print(f"📊 초기 품질 점수: {initial_score:.3f}")
        print(f"⏱️  예상 소요 시간: {self.modes[mode]['estimated_time']}")

        # 사용자 확인
        if not self._confirm_execution(mode):
            print("❌ 사용자에 의해 취소됨")
            return {"status": "cancelled"}

        # 단계별 실행
        mode_config = self.modes[mode]
        steps_to_run = mode_config["steps"]
        current_file = str(input_path)

        for step_num in steps_to_run:
            step = self.steps[step_num]

            print(f"\n{step['emoji']} Step {step_num}: {step['name']}")
            print(f"   📝 {step['description']}")
            print(f"   ⏰ 예상 시간: {step['estimated_time']}")

            if 'strategies' in step:
                print(f"   🔧 RAG 전략: {', '.join(step['strategies'])}")

            start_time = datetime.now()

            try:
                # Unified RAG 단계 실행
                step_result = self._execute_unified_step(
                    step_num,
                    current_file,
                    session_dir,
                    enable_cross_domain=enable_cross_domain,
                    preferred_strategies=preferred_strategies,
                    target_domains=target_domains
                )

                end_time = datetime.now()
                step_duration = (end_time - start_time).total_seconds()

                # 실행 로그 업데이트
                execution_log["steps_completed"].append({
                    "step": step_num,
                    "name": step["name"],
                    "duration": step_duration,
                    "result": step_result,
                    "unified_rag_metrics": step_result.get("rag_metrics", {}),
                    "completed_at": end_time.isoformat()
                })

                # 출력 파일 업데이트
                if step_result.get("output_file"):
                    current_file = step_result["output_file"]

                print(f"   ✅ 완료! ({step_duration:.1f}초)")

                # Strategy performance tracking
                if "strategy_used" in step_result:
                    strategy = step_result["strategy_used"]
                    if strategy not in execution_log["strategy_performance"]:
                        execution_log["strategy_performance"][strategy] = []
                    execution_log["strategy_performance"][strategy].append({
                        "step": step_num,
                        "confidence": step_result.get("confidence", 0),
                        "quality_improvement": step_result.get("quality_delta", 0)
                    })

            except Exception as e:
                print(f"   ❌ 실패: {str(e)}")
                logger.error(f"Step {step_num} failed: {e}")

                execution_log["steps_completed"].append({
                    "step": step_num,
                    "name": step["name"],
                    "error": str(e),
                    "failed_at": datetime.now().isoformat()
                })

                # Continue with next step despite error
                continue

        # 최종 품질 평가
        final_score = self._calculate_unified_quality_score(current_file)
        improvement = final_score - initial_score

        execution_log.update({
            "final_score": final_score,
            "improvement": improvement,
            "completed_at": datetime.now().isoformat(),
            "final_output_file": current_file
        })

        # 최종 파일 복사
        final_output = session_dir / f"optimized_{input_path.name}"
        shutil.copy2(current_file, final_output)

        # 실행 로그 저장
        log_file = session_dir / "optimization_log.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(execution_log, f, ensure_ascii=False, indent=2)

        # 통계 업데이트
        self._update_optimization_stats(execution_log)

        # 결과 요약 출력
        self._print_unified_summary(execution_log, final_output)

        return execution_log

    def _execute_unified_step(self,
                             step_num: int,
                             input_file: str,
                             session_dir: Path,
                             enable_cross_domain: bool = True,
                             preferred_strategies: Optional[List[str]] = None,
                             target_domains: Optional[List[str]] = None) -> Dict[str, Any]:
        """Unified RAG 단계 실행"""

        step = self.steps[step_num]
        script_name = step["script"]

        # 단계별 출력 파일명
        output_file = session_dir / f"step_{step_num}_{input_file.split('/')[-1]}"

        # Unified RAG 기반 명령어 구성
        cmd = [
            "poetry", "run", "python", str(self.scripts_dir / script_name),
            "--input", input_file,
            "--output", str(output_file),
            "--unified-rag",  # Enable Unified RAG mode
        ]

        # Cross-domain synthesis 옵션
        if enable_cross_domain:
            cmd.extend(["--enable-cross-domain"])

        # Preferred strategies
        if preferred_strategies:
            cmd.extend(["--strategies", ",".join(preferred_strategies)])

        # Target domains
        if target_domains:
            cmd.extend(["--domains", ",".join(target_domains)])

        # 단계별 특화 옵션
        if "strategies" in step:
            cmd.extend(["--step-strategies", ",".join(step["strategies"])])

        if step.get("cross_modal"):
            cmd.extend(["--enable-multimodal"])

        if step.get("knowledge_synthesis"):
            cmd.extend(["--enable-synthesis"])

        if step.get("agent_rag_integration"):
            cmd.extend(["--enable-agent-rag"])

        # 명령어 실행
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30min timeout

            if result.returncode == 0:
                step_result = {
                    "status": "success",
                    "output_file": str(output_file),
                    "command": " ".join(cmd)
                }

                # 스크립트가 JSON 결과를 반환하는 경우 파싱
                try:
                    if result.stdout.strip():
                        script_output = json.loads(result.stdout.strip())
                        step_result.update(script_output)
                except json.JSONDecodeError:
                    # Text output인 경우
                    step_result["output"] = result.stdout

                return step_result
            else:
                raise Exception(f"Command failed with code {result.returncode}: {result.stderr}")

        except subprocess.TimeoutExpired:
            raise Exception(f"Step timed out after 30 minutes")
        except Exception as e:
            raise Exception(f"Step execution failed: {e}")

    def _calculate_unified_quality_score(self, file_path: str) -> float:
        """Unified RAG 기반 품질 점수 계산"""
        # Unified evidence mapping으로 품질 점수 계산
        temp_output = Path("temp_unified_score.json")

        cmd = [
            "poetry", "run", "python", str(self.scripts_dir / "map_proposal_to_unified_evidence.py"),
            "--proposal", file_path,
            "--output", str(temp_output),
            "--unified-rag",
            "--quality-assessment"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0 and temp_output.exists():
                with open(temp_output, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Unified RAG 기반 점수 (다중 전략 평균)
                    score = data.get("unified_quality_metrics", {}).get("overall_score", 0.0)

                temp_output.unlink()  # 임시 파일 삭제
                return score
            else:
                logger.warning(f"Quality assessment failed: {result.stderr}")
                return 0.5  # Default fallback score

        except Exception as e:
            logger.warning(f"Quality assessment error: {e}")
            return 0.5

    def _create_session_id(self) -> str:
        """세션 ID 생성"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"unified_optimization_{timestamp}"

    def _confirm_execution(self, mode: str) -> bool:
        """실행 확인"""
        mode_info = self.modes[mode]

        print(f"\n📋 실행 계획:")
        print(f"   🎯 모드: {mode_info['description']}")
        print(f"   📝 단계: {len(mode_info['steps'])}개")
        print(f"   ⏰ 예상 시간: {mode_info['estimated_time']}")

        if 'strategies' in mode_info:
            strategies = mode_info['strategies']
            if strategies == "ALL":
                print(f"   🔧 RAG 전략: 모든 6개 전략 활용")
            else:
                print(f"   🔧 RAG 전략: {', '.join(strategies)}")

        # Auto-confirm if AUTO_CONFIRM env var is set
        if os.environ.get('UPE_AUTO_CONFIRM', '').lower() in ['1', 'true', 'yes']:
            print("\n✅ 자동 확인 (UPE_AUTO_CONFIRM=1)")
            return True
        response = input(f"\n계속 진행하시겠습니까? (y/n): ").strip().lower()
        return response in ['y', 'yes', '예']

    def _print_unified_summary(self, execution_log: Dict[str, Any], final_output: Path):
        """Unified RAG 최적화 결과 요약 출력"""
        print(f"\n" + "="*70)
        print("📊 UNIFIED RAG OPTIMIZATION SUMMARY")
        print("="*70)

        # 실행 정보
        print(f"📄 입력 파일: {execution_log['input_file']}")
        print(f"💾 최종 출력: {final_output}")
        print(f"🎯 모드: {execution_log['mode']}")
        print(f"⏰ 세션 ID: {execution_log['session_id']}")

        # 품질 향상
        initial = execution_log['initial_score']
        final = execution_log['final_score']
        improvement = execution_log['improvement']

        print(f"\n📈 품질 향상:")
        print(f"   초기 점수: {initial:.3f}")
        print(f"   최종 점수: {final:.3f}")
        print(f"   향상도: {improvement:+.3f} ({improvement/initial*100:+.1f}%)" if initial > 0 else "")

        # Unified RAG 전략 성능
        if execution_log.get('strategy_performance'):
            print(f"\n🔧 RAG 전략 성능:")
            for strategy, performances in execution_log['strategy_performance'].items():
                avg_confidence = sum(p['confidence'] for p in performances) / len(performances)
                avg_improvement = sum(p['quality_improvement'] for p in performances) / len(performances)
                print(f"   {strategy}: 평균 신뢰도 {avg_confidence:.3f}, 평균 향상 {avg_improvement:+.3f}")

        # 단계별 결과
        print(f"\n📝 단계별 실행 결과:")
        for step_log in execution_log['steps_completed']:
            if 'error' in step_log:
                print(f"   ❌ Step {step_log['step']}: {step_log['name']} - 실패")
            else:
                duration = step_log.get('duration', 0)
                print(f"   ✅ Step {step_log['step']}: {step_log['name']} ({duration:.1f}초)")

                # RAG 메트릭 표시
                rag_metrics = step_log.get('unified_rag_metrics', {})
                if rag_metrics:
                    strategy = rag_metrics.get('primary_strategy', 'Unknown')
                    confidence = rag_metrics.get('confidence', 0)
                    print(f"      🔧 전략: {strategy}, 신뢰도: {confidence:.3f}")

        # Cross-domain insights
        cross_domain_count = sum(1 for step in execution_log['steps_completed']
                               if step.get('unified_rag_metrics', {}).get('cross_domain_insights'))
        if cross_domain_count > 0:
            print(f"\n🌐 Cross-domain 통찰: {cross_domain_count}개 단계에서 발견")

        print(f"\n🎉 최적화 완료!")
        print(f"📁 결과 파일: {final_output}")
        print("="*70)

    def _update_optimization_stats(self, execution_log: Dict[str, Any]):
        """최적화 통계 업데이트"""
        self.optimization_stats["total_optimizations"] += 1
        self.optimization_stats["quality_improvements"].append(execution_log.get("improvement", 0))

        # Strategy performance tracking
        for strategy, performances in execution_log.get("strategy_performance", {}).items():
            if strategy not in self.optimization_stats["strategy_performance"]:
                self.optimization_stats["strategy_performance"][strategy] = []
            self.optimization_stats["strategy_performance"][strategy].extend(performances)

        # Cross-domain success tracking
        if any(step.get('unified_rag_metrics', {}).get('cross_domain_insights')
               for step in execution_log.get('steps_completed', [])):
            self.optimization_stats["cross_domain_successes"] += 1

    def run_specific_steps(self,
                          input_file: str,
                          steps: List[int],
                          unified_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """특정 단계만 실행"""

        logger.info(f"🎯 Running specific Unified RAG steps: {steps}")

        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        session_id = self._create_session_id() + "_custom"
        session_dir = self.output_dir / session_id
        session_dir.mkdir(exist_ok=True)

        unified_config = unified_config or {}

        print(f"\n🔧 사용자 정의 Unified RAG 실행")
        print(f"📄 파일: {input_path.name}")
        print(f"📝 실행 단계: {steps}")

        current_file = str(input_path)
        execution_results = []

        for step_num in steps:
            if step_num not in self.steps:
                print(f"⚠️ 잘못된 단계 번호: {step_num}")
                continue

            step = self.steps[step_num]
            print(f"\n{step['emoji']} Step {step_num}: {step['name']}")

            try:
                step_result = self._execute_unified_step(
                    step_num,
                    current_file,
                    session_dir,
                    **unified_config
                )

                execution_results.append({
                    "step": step_num,
                    "result": step_result,
                    "success": True
                })

                if step_result.get("output_file"):
                    current_file = step_result["output_file"]

                print(f"   ✅ 완료!")

            except Exception as e:
                print(f"   ❌ 실패: {str(e)}")
                execution_results.append({
                    "step": step_num,
                    "error": str(e),
                    "success": False
                })

        return {
            "session_id": session_id,
            "results": execution_results,
            "final_file": current_file
        }

    def interactive_wizard(self):
        """대화형 Unified RAG 최적화 마법사"""
        print("\n" + "="*70)
        print("🧙 UNIFIED RAG OPTIMIZATION WIZARD")
        print("="*70)

        # 파일 선택
        input_file = input("📄 제안서 파일 경로를 입력하세요: ").strip()
        if not Path(input_file).exists():
            print("❌ 파일을 찾을 수 없습니다.")
            return

        # 모드 선택
        print(f"\n🎯 최적화 모드를 선택하세요:")
        for i, (mode_key, mode_info) in enumerate(self.modes.items(), 1):
            print(f"   {i}. {mode_key}: {mode_info['description']}")
            print(f"      ⏰ {mode_info['estimated_time']}")

        try:
            mode_choice = int(input("모드 번호를 선택하세요 (1-5): "))
            mode_key = list(self.modes.keys())[mode_choice - 1]
        except (ValueError, IndexError):
            print("❌ 잘못된 선택입니다.")
            return

        # Cross-domain 옵션
        cross_domain = input("🌐 Cross-domain 합성을 활성화하시겠습니까? (y/n): ").strip().lower() in ['y', 'yes']

        # 전략 선택 (선택사항)
        print(f"\n🔧 선호하는 RAG 전략을 선택하세요 (Enter로 자동 선택):")
        available_strategies = ["HYBRID", "GRAPH_RAG", "ENHANCED_DD_RAPTOR", "GOLDEN_REFERENCE", "MULTIMODAL_RAG", "PSYCHOLOGY_RAG"]
        for i, strategy in enumerate(available_strategies, 1):
            print(f"   {i}. {strategy}")

        strategy_input = input("전략 번호들을 쉼표로 구분하여 입력 (예: 1,2,3): ").strip()
        preferred_strategies = None

        if strategy_input:
            try:
                strategy_indices = [int(x.strip()) - 1 for x in strategy_input.split(',')]
                preferred_strategies = [available_strategies[i] for i in strategy_indices if 0 <= i < len(available_strategies)]
            except ValueError:
                print("⚠️ 전략 선택 형식이 잘못되었습니다. 자동 선택으로 진행합니다.")

        # 도메인 선택 (선택사항)
        if cross_domain:
            print(f"\n🌐 대상 도메인을 선택하세요:")
            available_domains = ["neuroscience", "quantum_ml", "protein_research", "general"]
            for i, domain in enumerate(available_domains, 1):
                print(f"   {i}. {domain}")

            domain_input = input("도메인 번호들을 쉼표로 구분하여 입력 (Enter로 모든 도메인): ").strip()
            target_domains = None

            if domain_input:
                try:
                    domain_indices = [int(x.strip()) - 1 for x in domain_input.split(',')]
                    target_domains = [available_domains[i] for i in domain_indices if 0 <= i < len(available_domains)]
                except ValueError:
                    print("⚠️ 도메인 선택 형식이 잘못되었습니다. 모든 도메인을 대상으로 합니다.")
        else:
            target_domains = None

        # 최적화 실행
        print(f"\n🚀 Unified RAG 최적화를 시작합니다...")
        result = self.optimize_proposal(
            input_file=input_file,
            mode=mode_key,
            enable_cross_domain=cross_domain,
            preferred_strategies=preferred_strategies,
            target_domains=target_domains
        )

        print(f"🎉 마법사 완료!")
        return result

    def get_optimization_stats(self) -> Dict[str, Any]:
        """최적화 통계 반환"""
        stats = self.optimization_stats.copy()

        # 계산된 메트릭 추가
        if stats["quality_improvements"]:
            stats["average_improvement"] = sum(stats["quality_improvements"]) / len(stats["quality_improvements"])
            stats["success_rate"] = sum(1 for imp in stats["quality_improvements"] if imp > 0) / len(stats["quality_improvements"])

        if stats["total_optimizations"] > 0:
            stats["cross_domain_success_rate"] = stats["cross_domain_successes"] / stats["total_optimizations"]

        return stats

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="Unified RAG Proposal Optimizer")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Run optimization workflow")
    optimize_parser.add_argument("--input", "-i", required=True, help="Input proposal file")
    optimize_parser.add_argument("--mode", "-m", default="full",
                               choices=["full", "quick", "research", "validation", "cross_domain"],
                               help="Optimization mode")
    optimize_parser.add_argument("--enable-cross-domain", action="store_true",
                               help="Enable cross-domain synthesis")
    optimize_parser.add_argument("--strategies", help="Comma-separated preferred strategies")
    optimize_parser.add_argument("--domains", help="Comma-separated target domains")

    # Run specific steps
    run_parser = subparsers.add_parser("run", help="Run specific steps")
    run_parser.add_argument("--input", "-i", required=True, help="Input proposal file")
    run_parser.add_argument("--steps", required=True, help="Comma-separated step numbers")
    run_parser.add_argument("--enable-cross-domain", action="store_true")
    run_parser.add_argument("--strategies", help="Comma-separated preferred strategies")

    # Interactive wizard
    wizard_parser = subparsers.add_parser("wizard", help="Interactive optimization wizard")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show optimization statistics")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    optimizer = UnifiedProposalOptimizer()

    try:
        if args.command == "optimize":
            preferred_strategies = args.strategies.split(',') if args.strategies else None
            target_domains = args.domains.split(',') if args.domains else None

            result = optimizer.optimize_proposal(
                input_file=args.input,
                mode=args.mode,
                enable_cross_domain=args.enable_cross_domain,
                preferred_strategies=preferred_strategies,
                target_domains=target_domains
            )

        elif args.command == "run":
            steps = [int(x.strip()) for x in args.steps.split(',')]
            preferred_strategies = args.strategies.split(',') if args.strategies else None

            unified_config = {
                "enable_cross_domain": args.enable_cross_domain,
                "preferred_strategies": preferred_strategies
            }

            result = optimizer.run_specific_steps(
                input_file=args.input,
                steps=steps,
                unified_config=unified_config
            )

        elif args.command == "wizard":
            result = optimizer.interactive_wizard()

        elif args.command == "stats":
            stats = optimizer.get_optimization_stats()
            print(json.dumps(stats, indent=2))

    except KeyboardInterrupt:
        print("\n❌ 사용자에 의해 중단됨")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        logger.exception("Optimization failed")
        sys.exit(1)

if __name__ == "__main__":
    main()