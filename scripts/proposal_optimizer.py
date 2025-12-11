#!/usr/bin/env python3
"""
🚀 Proposal Optimizer - 통합 워크플로우 자동화 시스템
================================================================

5단계 최적화 워크플로우를 one-click으로 실행하는 통합 시스템

Features:
- 전체 파이프라인 자동 실행
- 단계별 선택 실행
- 진행 상황 실시간 모니터링
- 품질 점수 추적
- 자동 백업 및 버전 관리

Usage Examples:
    # 전체 최적화 (추천)
    poetry run python scripts/proposal_optimizer.py optimize \
        --input "proposal.md" --mode full

    # 빠른 개선 (2시간)
    poetry run python scripts/proposal_optimizer.py optimize \
        --input "proposal.md" --mode quick

    # 단계별 실행
    poetry run python scripts/proposal_optimizer.py run \
        --steps "1,3,5" --input "proposal.md"

    # 대화형 모드
    poetry run python scripts/proposal_optimizer.py wizard
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import time

class ProposalOptimizer:
    """통합 제안서 최적화 시스템"""

    def __init__(self):
        self.base_dir = Path.cwd()
        self.scripts_dir = self.base_dir / "scripts"
        self.output_dir = self.base_dir / "optimization_output"
        self.output_dir.mkdir(exist_ok=True)

        # 워크플로우 단계 정의
        self.steps = {
            1: {
                "name": "Evidence Mapping & Diagnosis",
                "emoji": "🔍",
                "script": "map_proposal_to_evidence.py",
                "description": "과학적 주장 분석 및 근거 강도 평가",
                "estimated_time": "2-3분"
            },
            2: {
                "name": "Real-time Claim Validation",
                "emoji": "⚡",
                "script": "validate_proposal_claims.py",
                "description": "실시간 주장 검증 및 자동 수정",
                "estimated_time": "10-30분"
            },
            3: {
                "name": "Enhanced DD Query & Literature Review",
                "emoji": "📚",
                "script": "enhanced_dd_query.py",
                "description": "체계적 문헌 검토 및 추가 근거 수집",
                "estimated_time": "5-10분"
            },
            4: {
                "name": "Multi-Agent Enhancement",
                "emoji": "🤖",
                "script": "multi_agent_proposal_pipeline.py",
                "description": "6개 전문 AI 에이전트 협업 개선",
                "estimated_time": "15-45분"
            },
            5: {
                "name": "Automated Citation & Finalization",
                "emoji": "📖",
                "script": "automated_citation_generator.py",
                "description": "자동 citation 및 참고문헌 생성",
                "estimated_time": "3-5분"
            }
        }

        # 실행 모드 정의
        self.modes = {
            "full": {
                "name": "완전 최적화",
                "description": "모든 5단계 실행 (최고 품질)",
                "steps": [1, 2, 3, 4, 5],
                "estimated_time": "35-95분"
            },
            "quick": {
                "name": "빠른 개선",
                "description": "핵심 3단계 실행 (시간 절약)",
                "steps": [1, 2, 5],
                "estimated_time": "15-40분"
            },
            "research": {
                "name": "연구 중심",
                "description": "문헌 검토 및 근거 강화",
                "steps": [1, 3, 4, 5],
                "estimated_time": "25-65분"
            },
            "validation": {
                "name": "검증 중심",
                "description": "주장 검증 및 자동 수정",
                "steps": [1, 2, 4],
                "estimated_time": "20-50분"
            }
        }

        print("🚀 Proposal Optimizer 초기화 완료")
        print(f"📁 출력 디렉토리: {self.output_dir}")

    def display_workflow_menu(self):
        """워크플로우 메뉴 표시"""
        print("\n" + "="*60)
        print("🎯 PROPOSAL OPTIMIZATION WORKFLOW")
        print("="*60)

        print("\n📋 사용 가능한 실행 모드:")
        for mode_key, mode_info in self.modes.items():
            steps_str = " → ".join([f"{self.steps[s]['emoji']}{s}" for s in mode_info['steps']])
            print(f"   {mode_key:12} | {mode_info['name']:12} | {steps_str}")
            print(f"   {'':12} | {mode_info['description']:12} | ⏱️  {mode_info['estimated_time']}")
            print()

        print("📝 개별 단계:")
        for step_num, step_info in self.steps.items():
            print(f"   Step {step_num}: {step_info['emoji']} {step_info['name']}")
            print(f"          {step_info['description']} (⏱️ {step_info['estimated_time']})")

    def optimize_proposal(self, input_file: str, mode: str = "full",
                        interactive: bool = False,
                        output_prefix: Optional[str] = None) -> Dict[str, Any]:
        """제안서 최적화 실행"""

        if mode not in self.modes:
            raise ValueError(f"Unknown mode: {mode}. Available: {list(self.modes.keys())}")

        if not Path(input_file).exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # 출력 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_prefix:
            session_name = f"{output_prefix}_{timestamp}"
        else:
            session_name = f"optimization_{timestamp}"

        session_dir = self.output_dir / session_name
        session_dir.mkdir(exist_ok=True)

        # 입력 파일 백업
        input_backup = session_dir / f"original_{Path(input_file).name}"
        shutil.copy2(input_file, input_backup)

        # 실행 정보
        mode_info = self.modes[mode]
        steps_to_run = mode_info["steps"]

        print(f"\n🚀 시작: {mode_info['name']} 모드")
        print(f"📄 입력: {input_file}")
        print(f"📁 세션: {session_name}")
        print(f"📋 단계: {' → '.join([str(s) for s in steps_to_run])}")
        print(f"⏱️  예상 시간: {mode_info['estimated_time']}")

        if interactive:
            response = input("\n계속 진행하시겠습니까? [y/N]: ").strip().lower()
            if response != 'y':
                print("❌ 사용자에 의해 취소됨")
                return {"status": "cancelled"}

        # 실행 로그
        execution_log = {
            "session_name": session_name,
            "mode": mode,
            "input_file": str(input_file),
            "start_time": datetime.now().isoformat(),
            "steps": [],
            "final_score": None,
            "improvement": None
        }

        current_file = str(input_backup)  # 단계간 파일 전달

        # 단계별 실행
        for step_num in steps_to_run:
            step_info = self.steps[step_num]

            print(f"\n{step_info['emoji']} Step {step_num}: {step_info['name']}")
            print("-" * 50)
            print(f"📝 {step_info['description']}")
            print(f"⏱️  예상 시간: {step_info['estimated_time']}")

            step_start = time.time()

            try:
                # 단계별 실행
                step_result = self._run_step(
                    step_num,
                    current_file,
                    session_dir,
                    interactive
                )

                step_duration = time.time() - step_start

                step_log = {
                    "step": step_num,
                    "name": step_info['name'],
                    "status": "success",
                    "duration": round(step_duration, 2),
                    "output_file": step_result.get("output_file"),
                    "metadata": step_result.get("metadata", {})
                }

                execution_log["steps"].append(step_log)

                # 다음 단계를 위한 파일 업데이트
                if step_result.get("output_file"):
                    current_file = step_result["output_file"]

                print(f"   ✅ 완료 (⏱️ {step_duration:.1f}초)")

            except Exception as e:
                step_duration = time.time() - step_start
                print(f"   ❌ 오류: {e}")

                step_log = {
                    "step": step_num,
                    "name": step_info['name'],
                    "status": "error",
                    "duration": round(step_duration, 2),
                    "error": str(e)
                }

                execution_log["steps"].append(step_log)

                # 오류 시 계속 진행할지 결정
                if interactive:
                    response = input(f"\nStep {step_num} 오류 발생. 계속 진행하시겠습니까? [y/N]: ").strip().lower()
                    if response != 'y':
                        break
                else:
                    print(f"⚠️  Step {step_num} 건너뛰고 계속 진행")

        # 최종 품질 평가
        try:
            final_score = self._calculate_final_score(current_file)
            execution_log["final_score"] = final_score

            # 개선도 계산 (첫 번째 단계 결과와 비교)
            if execution_log["steps"] and execution_log["steps"][0]["metadata"].get("initial_score"):
                initial_score = execution_log["steps"][0]["metadata"]["initial_score"]
                execution_log["improvement"] = final_score - initial_score

        except Exception as e:
            print(f"⚠️  최종 평가 오류: {e}")

        # 실행 로그 저장
        execution_log["end_time"] = datetime.now().isoformat()
        log_file = session_dir / "execution_log.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(execution_log, f, indent=2, ensure_ascii=False)

        # 최종 파일 복사
        final_output = session_dir / f"optimized_{Path(input_file).name}"
        if Path(current_file).exists() and current_file != str(input_backup):
            shutil.copy2(current_file, final_output)

        # 결과 요약
        self._print_summary(execution_log, final_output)

        return execution_log

    def _run_step(self, step_num: int, input_file: str, session_dir: Path,
                 interactive: bool = False) -> Dict[str, Any]:
        """개별 단계 실행"""

        step_info = self.steps[step_num]
        script_name = step_info["script"]
        script_path = self.scripts_dir / script_name

        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")

        # 단계별 출력 파일
        step_output_dir = session_dir / f"step_{step_num}_{step_info['name'].replace(' ', '_').lower()}"
        step_output_dir.mkdir(exist_ok=True)

        result = {"metadata": {}}

        if step_num == 1:  # Evidence Mapping
            output_file = step_output_dir / "evidence_mapping.json"
            cmd = [
                "poetry", "run", "python", str(script_path),
                "--proposal", input_file,
                "--output", str(output_file)
            ]

            self._run_command(cmd)

            # 결과 파싱하여 초기 점수 추출
            if output_file.exists():
                with open(output_file, 'r', encoding='utf-8') as f:
                    evidence_data = json.load(f)
                    result["metadata"]["initial_score"] = evidence_data["summary"]["scientific_rigor_score"]

            result["output_file"] = input_file  # 다음 단계로 원본 전달

        elif step_num == 2:  # Real-time Validation
            output_file = step_output_dir / f"validated_{Path(input_file).name}"

            if interactive:
                # 대화형 모드
                cmd = [
                    "poetry", "run", "python", str(script_path),
                    "--input", input_file,
                    "--interactive"
                ]
            else:
                # 자동 모드
                cmd = [
                    "poetry", "run", "python", str(script_path),
                    "--input", input_file,
                    "--output", str(step_output_dir / "validation_report.json"),
                    "--threshold", "0.7"
                ]

            self._run_command(cmd)
            result["output_file"] = str(output_file) if output_file.exists() else input_file

        elif step_num == 3:  # Enhanced DD Query
            output_file = step_output_dir / "literature_review.json"
            cmd = [
                "poetry", "run", "python", str(script_path),
                "--mode", "systematic_review",
                "--topic", "korean developmental disorder foundation model genomics",
                "--n_results", "15",
                "--export", str(output_file)
            ]

            self._run_command(cmd)
            result["output_file"] = input_file  # 문헌검토는 원본 파일 유지

        elif step_num == 4:  # Multi-Agent Enhancement
            output_file = step_output_dir / f"enhanced_{Path(input_file).name}"
            report_file = step_output_dir / "agent_report.json"

            cmd = [
                "poetry", "run", "python", str(script_path),
                "--mode", "full_pipeline",
                "--input", input_file,
                "--output", str(output_file),
                "--report", str(report_file)
            ]

            self._run_command(cmd)
            result["output_file"] = str(output_file)

        elif step_num == 5:  # Automated Citation
            output_file = step_output_dir / f"cited_{Path(input_file).name}"

            if interactive:
                # 대화형 citation
                cmd = [
                    "poetry", "run", "python", str(script_path),
                    "--input", input_file,
                    "--mode", "interactive",
                    "--output", str(output_file)
                ]
            else:
                # 자동 citation
                cmd = [
                    "poetry", "run", "python", str(script_path),
                    "--input", input_file,
                    "--mode", "auto_cite",
                    "--output", str(output_file),
                    "--threshold", "0.75"
                ]

            self._run_command(cmd)
            result["output_file"] = str(output_file)

        return result

    def _run_command(self, cmd: List[str], capture_output: bool = False) -> Optional[str]:
        """명령어 실행"""
        print(f"   🔧 실행: {' '.join(cmd)}")

        try:
            if capture_output:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                return result.stdout
            else:
                subprocess.run(cmd, check=True)
                return None
        except subprocess.CalledProcessError as e:
            print(f"   ❌ 명령어 실행 실패: {e}")
            raise

    def _calculate_final_score(self, file_path: str) -> float:
        """최종 품질 점수 계산"""
        # Evidence mapping으로 최종 점수 계산
        temp_output = Path("temp_final_score.json")

        cmd = [
            "poetry", "run", "python", str(self.scripts_dir / "map_proposal_to_evidence.py"),
            "--proposal", file_path,
            "--output", str(temp_output)
        ]

        self._run_command(cmd)

        if temp_output.exists():
            with open(temp_output, 'r', encoding='utf-8') as f:
                data = json.load(f)
                score = data["summary"]["scientific_rigor_score"]

            temp_output.unlink()  # 임시 파일 삭제
            return score

        return 0.0

    def _print_summary(self, execution_log: Dict[str, Any], final_output: Path):
        """실행 결과 요약 출력"""
        print(f"\n" + "="*60)
        print("📊 OPTIMIZATION SUMMARY")
        print("="*60)

        # 실행 정보
        print(f"🎯 모드: {execution_log['mode']} ({self.modes[execution_log['mode']]['name']})")
        print(f"📄 입력: {execution_log['input_file']}")
        print(f"💾 최종 출력: {final_output}")

        # 단계별 결과
        successful_steps = [s for s in execution_log['steps'] if s['status'] == 'success']
        failed_steps = [s for s in execution_log['steps'] if s['status'] == 'error']

        print(f"\n📋 실행 결과:")
        print(f"   ✅ 성공: {len(successful_steps)}/{len(execution_log['steps'])} 단계")
        if failed_steps:
            print(f"   ❌ 실패: {len(failed_steps)} 단계")
            for step in failed_steps:
                print(f"      - Step {step['step']}: {step['name']}")

        # 품질 개선
        if execution_log.get('final_score') is not None:
            print(f"\n📈 품질 점수:")
            final_score = execution_log['final_score']
            print(f"   🎯 최종 점수: {final_score:.1f}/100")

            if execution_log.get('improvement') is not None:
                improvement = execution_log['improvement']
                if improvement > 0:
                    print(f"   📈 개선: +{improvement:.1f}점")
                else:
                    print(f"   📉 변화: {improvement:.1f}점")

            # 목표 달성도
            if final_score >= 90:
                print(f"   🏆 삼성 1등급 달성 가능! (90+ 점)")
            elif final_score >= 80:
                print(f"   🥈 양호한 수준 (80+ 점)")
            elif final_score >= 70:
                print(f"   🥉 개선 여지 있음 (70+ 점)")
            else:
                print(f"   ⚠️  추가 개선 필요 (<70 점)")

        # 총 소요 시간
        if execution_log.get('start_time') and execution_log.get('end_time'):
            start_dt = datetime.fromisoformat(execution_log['start_time'])
            end_dt = datetime.fromisoformat(execution_log['end_time'])
            total_duration = (end_dt - start_dt).total_seconds() / 60
            print(f"\n⏱️  총 소요 시간: {total_duration:.1f}분")

        print(f"\n📁 세션 디렉토리: {execution_log['session_name']}")
        print("="*60)

    def run_wizard(self):
        """대화형 워크플로우 마법사"""
        print("\n🧙‍♂️ PROPOSAL OPTIMIZATION WIZARD")
        print("="*50)

        # 1. 입력 파일 선택
        while True:
            input_file = input("📄 제안서 파일 경로를 입력하세요: ").strip()
            if input_file and Path(input_file).exists():
                break
            else:
                print("❌ 파일을 찾을 수 없습니다. 다시 입력하세요.")

        # 2. 모드 선택
        self.display_workflow_menu()
        while True:
            mode = input(f"\n🎯 실행 모드를 선택하세요 {list(self.modes.keys())}: ").strip()
            if mode in self.modes:
                break
            else:
                print("❌ 올바른 모드를 선택하세요.")

        # 3. 대화형 여부
        interactive = input("\n💬 대화형 모드를 사용하시겠습니까? [y/N]: ").strip().lower() == 'y'

        # 4. 출력 접두사
        output_prefix = input("📝 출력 파일 접두사 (선택사항): ").strip() or None

        # 5. 실행 확인
        mode_info = self.modes[mode]
        print(f"\n📋 설정 확인:")
        print(f"   📄 파일: {input_file}")
        print(f"   🎯 모드: {mode_info['name']}")
        print(f"   ⏱️  예상 시간: {mode_info['estimated_time']}")
        print(f"   💬 대화형: {'예' if interactive else '아니오'}")

        if input("\n🚀 실행하시겠습니까? [Y/n]: ").strip().lower() not in ['n', 'no']:
            return self.optimize_proposal(
                input_file=input_file,
                mode=mode,
                interactive=interactive,
                output_prefix=output_prefix
            )
        else:
            print("❌ 실행이 취소되었습니다.")
            return {"status": "cancelled"}

def main():
    parser = argparse.ArgumentParser(
        description="🚀 Proposal Optimizer - 통합 워크플로우 자동화",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
    # 전체 최적화
    poetry run python scripts/proposal_optimizer.py optimize --input proposal.md --mode full

    # 빠른 개선
    poetry run python scripts/proposal_optimizer.py optimize --input proposal.md --mode quick

    # 대화형 마법사
    poetry run python scripts/proposal_optimizer.py wizard

    # 도움말
    poetry run python scripts/proposal_optimizer.py menu
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='사용 가능한 명령')

    # optimize 명령
    optimize_parser = subparsers.add_parser('optimize', help='제안서 최적화 실행')
    optimize_parser.add_argument('--input', '-i', required=True, help='입력 제안서 파일')
    optimize_parser.add_argument('--mode', '-m', choices=['full', 'quick', 'research', 'validation'],
                                default='full', help='실행 모드')
    optimize_parser.add_argument('--interactive', action='store_true', help='대화형 모드')
    optimize_parser.add_argument('--output-prefix', help='출력 파일 접두사')

    # wizard 명령
    subparsers.add_parser('wizard', help='대화형 워크플로우 마법사')

    # menu 명령
    subparsers.add_parser('menu', help='워크플로우 메뉴 표시')

    # run 명령 (개별 단계 실행)
    run_parser = subparsers.add_parser('run', help='개별 단계 실행')
    run_parser.add_argument('--steps', required=True, help='실행할 단계 (예: 1,3,5)')
    run_parser.add_argument('--input', '-i', required=True, help='입력 파일')
    run_parser.add_argument('--interactive', action='store_true', help='대화형 모드')

    args = parser.parse_args()

    try:
        optimizer = ProposalOptimizer()

        if args.command == 'optimize':
            result = optimizer.optimize_proposal(
                input_file=args.input,
                mode=args.mode,
                interactive=args.interactive,
                output_prefix=args.output_prefix
            )

            if result.get('status') != 'cancelled':
                print(f"\n✅ 최적화 완료!")

        elif args.command == 'wizard':
            optimizer.run_wizard()

        elif args.command == 'menu':
            optimizer.display_workflow_menu()

        elif args.command == 'run':
            steps = [int(s.strip()) for s in args.steps.split(',')]
            # 개별 단계 실행 로직 (간단화)
            print(f"🎯 단계 {steps} 실행: {args.input}")
            for step in steps:
                print(f"Step {step}: {optimizer.steps[step]['name']}")

        elif not args.command:
            optimizer.display_workflow_menu()
            parser.print_help()

    except Exception as e:
        print(f"❌ 오류: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()