#!/usr/bin/env python3
"""
🧙‍♂️ Proposal Optimization Wizard
===================================

초보자도 쉽게 사용할 수 있는 대화형 제안서 최적화 마법사

Features:
- 단계별 안내 및 설명
- 파일 자동 감지
- 맞춤형 최적화 전략 추천
- 실시간 진행 상황 표시
- 결과 분석 및 해석

Usage:
    poetry run python scripts/proposal_wizard.py
"""

import os
import json
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

class ProposalWizard:
    """대화형 제안서 최적화 마법사"""

    def __init__(self):
        self.base_dir = Path.cwd()
        self.scripts_dir = self.base_dir / "scripts"

        # 지원하는 파일 확장자
        self.supported_extensions = ['.md', '.txt', '.docx', '.pdf']

        # 색깔 코드
        self.colors = {
            'header': '\033[95m',
            'blue': '\033[94m',
            'cyan': '\033[96m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'red': '\033[91m',
            'bold': '\033[1m',
            'underline': '\033[4m',
            'end': '\033[0m'
        }

        self.session_data = {}

    def print_colored(self, text: str, color: str = 'end'):
        """색깔 있는 텍스트 출력"""
        print(f"{self.colors.get(color, '')}{text}{self.colors['end']}")

    def print_header(self, text: str):
        """헤더 출력"""
        self.print_colored("="*60, 'blue')
        self.print_colored(f"🧙‍♂️ {text}", 'header')
        self.print_colored("="*60, 'blue')

    def print_step(self, step: int, title: str):
        """단계 헤더 출력"""
        self.print_colored(f"\n📋 Step {step}: {title}", 'cyan')
        self.print_colored("-"*40, 'cyan')

    def get_user_input(self, prompt: str, default: Optional[str] = None,
                      options: Optional[List[str]] = None) -> str:
        """사용자 입력 받기"""
        if options:
            option_str = "/".join(options)
            if default:
                full_prompt = f"{prompt} [{option_str}] (기본값: {default}): "
            else:
                full_prompt = f"{prompt} [{option_str}]: "
        else:
            if default:
                full_prompt = f"{prompt} (기본값: {default}): "
            else:
                full_prompt = f"{prompt}: "

        user_input = input(full_prompt).strip()

        if not user_input and default:
            return default

        if options and user_input.lower() in [opt.lower() for opt in options]:
            return user_input.lower()
        elif options:
            self.print_colored(f"❌ 올바른 옵션을 선택하세요: {options}", 'red')
            return self.get_user_input(prompt, default, options)

        return user_input

    def show_welcome(self):
        """환영 메시지"""
        self.print_header("제안서 최적화 마법사에 오신 것을 환영합니다!")
        print("""
🎯 이 마법사가 도와드릴 것:
   • 제안서 파일 자동 탐지
   • 현재 상태 진단
   • 최적 최적화 전략 추천
   • 단계별 실행 및 진행 상황 모니터링
   • 결과 분석 및 개선 방향 제시

📝 현재 지원하는 파일: .md, .txt, .docx, .pdf
⏱️  전체 과정: 약 30-90분
🎯 목표: 과학적 엄밀성 90+ 점 달성
        """)

    def detect_proposal_files(self) -> List[Path]:
        """제안서 파일 자동 감지"""
        self.print_step(1, "제안서 파일 탐지")

        proposal_files = []

        # 현재 디렉토리와 하위 디렉토리 검색
        search_dirs = [
            self.base_dir,
            self.base_dir / "data" / "발달장애",
            self.base_dir / "docs",
            self.base_dir / "proposals"
        ]

        for search_dir in search_dirs:
            if search_dir.exists():
                for ext in self.supported_extensions:
                    files = list(search_dir.rglob(f"*{ext}"))
                    # 제안서 키워드가 포함된 파일 우선
                    proposal_keywords = ['제안서', '계획서', 'proposal', 'grant', '삼성', 'samsung']
                    for file in files:
                        if any(keyword in file.name.lower() for keyword in proposal_keywords):
                            proposal_files.append(file)

        # 중복 제거 및 정렬
        proposal_files = list(set(proposal_files))
        proposal_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)  # 최신순

        return proposal_files

    def select_proposal_file(self) -> Path:
        """제안서 파일 선택"""
        detected_files = self.detect_proposal_files()

        if detected_files:
            print(f"📁 {len(detected_files)}개의 제안서 파일을 발견했습니다:")

            for i, file in enumerate(detected_files[:10], 1):  # 최대 10개 표시
                relative_path = file.relative_to(self.base_dir)
                file_size = file.stat().st_size / 1024  # KB
                mod_time = datetime.fromtimestamp(file.stat().st_mtime).strftime("%m-%d %H:%M")
                print(f"   {i}. {relative_path} ({file_size:.1f}KB, 수정: {mod_time})")

            print(f"   {len(detected_files) + 1}. 직접 입력")

            while True:
                choice = self.get_user_input("파일을 선택하세요", "1")

                try:
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(detected_files):
                        selected_file = detected_files[choice_num - 1]
                        self.print_colored(f"✅ 선택됨: {selected_file.relative_to(self.base_dir)}", 'green')
                        return selected_file
                    elif choice_num == len(detected_files) + 1:
                        break  # 직접 입력으로 이동
                    else:
                        self.print_colored("❌ 올바른 번호를 선택하세요", 'red')
                except ValueError:
                    self.print_colored("❌ 숫자를 입력하세요", 'red')

        # 직접 입력
        while True:
            file_path = self.get_user_input("제안서 파일 경로를 입력하세요")
            if file_path and Path(file_path).exists():
                return Path(file_path)
            else:
                self.print_colored("❌ 파일을 찾을 수 없습니다", 'red')

    def analyze_current_status(self, proposal_file: Path) -> Dict:
        """현재 상태 분석"""
        self.print_step(2, "현재 상태 진단")

        print("🔍 제안서를 분석하고 있습니다...")

        # Evidence mapping 실행
        temp_output = self.base_dir / "temp_diagnosis.json"

        cmd = [
            "poetry", "run", "python", str(self.scripts_dir / "map_proposal_to_evidence.py"),
            "--proposal", str(proposal_file),
            "--output", str(temp_output)
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)

            if temp_output.exists():
                with open(temp_output, 'r', encoding='utf-8') as f:
                    diagnosis = json.load(f)

                temp_output.unlink()  # 임시 파일 삭제

                # 결과 출력
                summary = diagnosis['summary']

                print("\n📊 진단 결과:")
                print(f"   📝 총 주장 수: {summary['total_claims']}")
                print(f"   🎯 현재 점수: {summary['scientific_rigor_score']:.1f}/100")
                print(f"   📚 근거 커버리지: {summary['evidence_coverage_percent']:.1f}%")

                breakdown = summary['validation_breakdown']
                print(f"\n📋 주장 분류:")
                print(f"   ✅ 강한 근거: {breakdown['strong']}")
                print(f"   🟡 중간 근거: {breakdown['moderate']}")
                print(f"   🟠 약한 근거: {breakdown['weak']}")
                print(f"   ❌ 근거 없음: {breakdown['unsupported']}")

                return diagnosis

        except subprocess.CalledProcessError as e:
            self.print_colored(f"❌ 진단 실패: {e}", 'red')
            return {}

    def recommend_strategy(self, diagnosis: Dict) -> str:
        """최적화 전략 추천"""
        self.print_step(3, "최적화 전략 추천")

        if not diagnosis:
            return "full"

        summary = diagnosis['summary']
        score = summary['scientific_rigor_score']
        coverage = summary['evidence_coverage_percent']
        unsupported = summary['validation_breakdown']['unsupported']

        print("🧠 AI가 분석한 최적 전략:")

        if score < 30 and unsupported > 80:
            strategy = "validation"
            self.print_colored("🔍 검증 중심 모드 추천", 'yellow')
            print("   이유: 대부분의 주장이 근거 없음. 검증 집중 필요")

        elif score < 50 and coverage < 30:
            strategy = "research"
            self.print_colored("📚 연구 중심 모드 추천", 'cyan')
            print("   이유: 문헌적 근거 부족. 체계적 검토 필요")

        elif score >= 70:
            strategy = "quick"
            self.print_colored("⚡ 빠른 개선 모드 추천", 'green')
            print("   이유: 기본 품질 양호. 빠른 마무리 가능")

        else:
            strategy = "full"
            self.print_colored("🏆 완전 최적화 모드 추천", 'blue')
            print("   이유: 종합적 개선으로 최고 품질 달성")

        print(f"\n📋 예상 결과:")

        expected_scores = {
            "validation": min(score + 30, 85),
            "research": min(score + 35, 90),
            "quick": min(score + 20, 80),
            "full": min(score + 50, 95)
        }

        expected_score = expected_scores[strategy]
        print(f"   📈 예상 점수: {score:.1f} → {expected_score:.1f} (+{expected_score - score:.1f})")

        if expected_score >= 90:
            print(f"   🏆 삼성 1등급 달성 가능!")
        elif expected_score >= 80:
            print(f"   🥈 우수 등급 예상")

        # 사용자 확인
        modes = {
            "1": "validation",
            "2": "research",
            "3": "quick",
            "4": "full"
        }

        print(f"\n🎯 모드 선택:")
        print(f"   1. 검증 중심 (20-50분)")
        print(f"   2. 연구 중심 (25-65분)")
        print(f"   3. 빠른 개선 (15-40분)")
        print(f"   4. 완전 최적화 (35-95분)")
        print(f"   5. 추천 모드 사용 ({strategy})")

        choice = self.get_user_input("선택하세요", "5", ["1", "2", "3", "4", "5"])

        if choice == "5":
            return strategy
        else:
            return modes.get(choice, strategy)

    def confirm_execution(self, proposal_file: Path, mode: str) -> bool:
        """실행 확인"""
        self.print_step(4, "실행 확인")

        mode_info = {
            "validation": ("검증 중심", "20-50분", "주장 검증 및 자동 수정"),
            "research": ("연구 중심", "25-65분", "문헌 검토 및 근거 강화"),
            "quick": ("빠른 개선", "15-40분", "핵심 단계만 실행"),
            "full": ("완전 최적화", "35-95분", "모든 단계 실행, 최고 품질")
        }

        name, time_est, desc = mode_info.get(mode, ("알 수 없음", "?", ""))

        print("📋 실행 계획 요약:")
        print(f"   📄 파일: {proposal_file.name}")
        print(f"   🎯 모드: {name}")
        print(f"   📝 설명: {desc}")
        print(f"   ⏱️  예상 시간: {time_est}")
        print(f"   🎯 목표: 과학적 엄밀성 90+ 점")

        print(f"\n⚠️  주의사항:")
        print(f"   • 원본 파일은 자동으로 백업됩니다")
        print(f"   • 인터넷 연결이 필요합니다")
        print(f"   • 중간에 대화형 입력이 필요할 수 있습니다")

        choice = self.get_user_input("🚀 최적화를 시작하시겠습니까?", "y", ["y", "n"])
        return choice == "y"

    def execute_optimization(self, proposal_file: Path, mode: str) -> Dict:
        """최적화 실행"""
        self.print_step(5, "최적화 실행")

        self.print_colored("🚀 최적화를 시작합니다...", 'green')

        start_time = time.time()

        cmd = [
            "poetry", "run", "python", str(self.scripts_dir / "proposal_optimizer.py"),
            "optimize",
            "--input", str(proposal_file),
            "--mode", mode,
            "--interactive"
        ]

        print(f"⚙️  명령어: {' '.join(cmd)}")
        print(f"📊 진행 상황을 모니터링하겠습니다...\n")

        try:
            # 실행 및 실시간 출력
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            output_lines = []
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
                    output_lines.append(output.strip())

            return_code = process.poll()
            execution_time = time.time() - start_time

            if return_code == 0:
                self.print_colored(f"✅ 최적화 완료! (⏱️ {execution_time/60:.1f}분)", 'green')

                # 결과 파일 찾기
                output_dir = self.base_dir / "optimization_output"
                if output_dir.exists():
                    sessions = list(output_dir.glob("optimization_*"))
                    if sessions:
                        latest_session = max(sessions, key=lambda x: x.stat().st_mtime)
                        log_file = latest_session / "execution_log.json"

                        if log_file.exists():
                            with open(log_file, 'r', encoding='utf-8') as f:
                                return json.load(f)

                return {"status": "success", "execution_time": execution_time}

            else:
                self.print_colored(f"❌ 최적화 실패 (코드: {return_code})", 'red')
                return {"status": "failed", "return_code": return_code}

        except Exception as e:
            self.print_colored(f"❌ 실행 오류: {e}", 'red')
            return {"status": "error", "error": str(e)}

    def analyze_results(self, execution_result: Dict):
        """결과 분석 및 해석"""
        self.print_step(6, "결과 분석")

        if execution_result.get("status") != "success":
            self.print_colored("❌ 최적화가 완료되지 않아 결과 분석을 생략합니다", 'red')
            return

        if "final_score" in execution_result:
            final_score = execution_result["final_score"]
            improvement = execution_result.get("improvement", 0)

            print("📊 최종 결과:")
            print(f"   🎯 최종 점수: {final_score:.1f}/100")

            if improvement > 0:
                print(f"   📈 점수 개선: +{improvement:.1f}점")
                self.print_colored("🎉 성공적으로 개선되었습니다!", 'green')
            else:
                print(f"   📉 점수 변화: {improvement:.1f}점")

            # 등급 평가
            if final_score >= 90:
                self.print_colored("🏆 삼성미래기술육성사업 1등급 달성 가능!", 'green')
                print("   축하합니다! 제출하셔도 좋을 품질입니다.")

            elif final_score >= 80:
                self.print_colored("🥈 우수한 품질입니다", 'cyan')
                print("   추가 개선으로 1등급 도전해보세요.")

            elif final_score >= 70:
                self.print_colored("🥉 양호한 수준입니다", 'yellow')
                print("   한 번 더 최적화를 권장합니다.")

            else:
                self.print_colored("⚠️ 추가 개선이 필요합니다", 'red')
                print("   다른 모드로 재실행을 고려해보세요.")

        # 단계별 결과
        if "steps" in execution_result:
            successful_steps = [s for s in execution_result["steps"] if s["status"] == "success"]
            print(f"\n📋 실행 단계:")
            print(f"   ✅ 성공: {len(successful_steps)}/{len(execution_result['steps'])} 단계")

        # 추천 사항
        self.provide_recommendations(execution_result)

    def provide_recommendations(self, execution_result: Dict):
        """추천 사항 제공"""
        print(f"\n💡 다음 단계 추천:")

        final_score = execution_result.get("final_score", 0)

        if final_score >= 90:
            print("   🎯 제출 준비: 최종 검토 후 제출하세요")
            print("   📝 마지막 체크: 삼성 제출 요구사항 확인")

        elif final_score >= 80:
            print("   🔄 한 번 더 최적화: 'full' 모드로 재실행")
            print("   ✍️  수동 편집: AI 결과를 바탕으로 직접 정제")

        else:
            print("   🔍 문제 분석: validation 모드로 문제점 파악")
            print("   📚 추가 연구: research 모드로 근거 보강")

        # 결과 파일 위치 안내
        if "session_name" in execution_result:
            output_dir = self.base_dir / "optimization_output" / execution_result["session_name"]
            print(f"\n📁 결과 파일 위치:")
            print(f"   {output_dir}")

    def run_wizard(self):
        """마법사 실행"""
        try:
            # 환영 메시지
            self.show_welcome()

            if not self.get_user_input("계속 진행하시겠습니까?", "y", ["y", "n"]) == "y":
                print("👋 마법사를 종료합니다.")
                return

            # 1. 파일 선택
            proposal_file = self.select_proposal_file()
            self.session_data['proposal_file'] = str(proposal_file)

            # 2. 현재 상태 분석
            diagnosis = self.analyze_current_status(proposal_file)
            self.session_data['initial_diagnosis'] = diagnosis

            # 3. 전략 추천
            strategy = self.recommend_strategy(diagnosis)
            self.session_data['strategy'] = strategy

            # 4. 실행 확인
            if not self.confirm_execution(proposal_file, strategy):
                print("👋 실행이 취소되었습니다.")
                return

            # 5. 최적화 실행
            execution_result = self.execute_optimization(proposal_file, strategy)
            self.session_data['execution_result'] = execution_result

            # 6. 결과 분석
            self.analyze_results(execution_result)

            # 세션 데이터 저장
            session_file = self.base_dir / f"wizard_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(self.session_data, f, indent=2, ensure_ascii=False)

            self.print_colored(f"\n💾 세션 데이터 저장: {session_file}", 'cyan')
            self.print_colored("🎉 마법사 완료! 좋은 결과 있으시길 바랍니다!", 'green')

        except KeyboardInterrupt:
            self.print_colored("\n⚠️ 사용자에 의해 중단되었습니다", 'yellow')
        except Exception as e:
            self.print_colored(f"\n❌ 예기치 않은 오류: {e}", 'red')

def main():
    wizard = ProposalWizard()
    wizard.run_wizard()

if __name__ == "__main__":
    main()