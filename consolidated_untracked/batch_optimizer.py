#!/usr/bin/env python3
"""
⚙️ Batch Proposal Optimizer
============================

YAML 설정 파일 기반 배치 최적화 실행기

Features:
- YAML 설정 기반 워크플로우 정의
- 다중 파일 배치 처리
- 조건부 실행 및 반복 루프
- 결과 집계 및 리포팅
- 스케줄링 및 모니터링

Usage:
    # 설정 파일로 실행
    poetry run python scripts/batch_optimizer.py \
        --config batch_config.yaml

    # 다중 파일 배치 처리
    poetry run python scripts/batch_optimizer.py \
        --config multi_file_config.yaml

    # 샘플 설정 생성
    poetry run python scripts/batch_optimizer.py \
        --create-sample-config sample.yaml
"""

import argparse
import yaml
import json
import subprocess
import time
import asyncio
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

class BatchOptimizer:
    """배치 최적화 실행기"""

    def __init__(self):
        self.base_dir = Path.cwd()
        self.scripts_dir = self.base_dir / "scripts"
        self.output_dir = self.base_dir / "batch_output"
        self.output_dir.mkdir(exist_ok=True)

    def create_sample_config(self, output_file: str):
        """샘플 설정 파일 생성"""

        sample_config = {
            "batch_info": {
                "name": "제안서 최적화 배치",
                "description": "여러 제안서를 자동으로 최적화하는 배치 작업",
                "version": "1.0"
            },

            "global_settings": {
                "parallel": False,
                "max_workers": 2,
                "continue_on_error": True,
                "backup_originals": True,
                "output_prefix": "batch_optimized"
            },

            "workflows": [
                {
                    "name": "단일 파일 완전 최적화",
                    "description": "하나의 제안서를 완전 최적화",
                    "enabled": True,
                    "input": {
                        "file": "data/발달장애/제안서.md",
                        "type": "single"
                    },
                    "optimization": {
                        "mode": "full",
                        "interactive": False,
                        "threshold": 0.7,
                        "target_score": 90
                    },
                    "output": {
                        "prefix": "fully_optimized",
                        "include_reports": True
                    }
                },

                {
                    "name": "다중 파일 빠른 개선",
                    "description": "여러 제안서를 빠르게 개선",
                    "enabled": False,
                    "input": {
                        "pattern": "proposals/*.md",
                        "type": "glob",
                        "max_files": 5
                    },
                    "optimization": {
                        "mode": "quick",
                        "interactive": False,
                        "threshold": 0.6
                    },
                    "output": {
                        "directory": "quick_optimized",
                        "format": "individual"
                    }
                },

                {
                    "name": "조건부 재최적화",
                    "description": "점수가 낮은 파일만 재최적화",
                    "enabled": False,
                    "input": {
                        "file": "optimized_proposal.md",
                        "type": "single"
                    },
                    "conditions": [
                        {
                            "type": "score_threshold",
                            "operator": "less_than",
                            "value": 80,
                            "action": "reoptimize"
                        }
                    ],
                    "optimization": {
                        "mode": "research",
                        "max_iterations": 3
                    }
                }
            ],

            "post_processing": {
                "generate_summary": True,
                "compare_scores": True,
                "create_backup": True,
                "notification": {
                    "enabled": False,
                    "method": "file",
                    "target": "completion_notification.txt"
                }
            }
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(sample_config, f, default_flow_style=False, allow_unicode=True)

        print(f"📄 샘플 설정 파일 생성: {output_file}")
        print("✏️  파일을 편집한 후 --config 옵션으로 실행하세요.")

    def load_config(self, config_file: str) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # 설정 검증
            self._validate_config(config)

            return config

        except FileNotFoundError:
            raise FileNotFoundError(f"설정 파일을 찾을 수 없음: {config_file}")
        except yaml.YAMLError as e:
            raise ValueError(f"YAML 파싱 오류: {e}")

    def _validate_config(self, config: Dict[str, Any]):
        """설정 검증"""
        required_sections = ["batch_info", "workflows"]

        for section in required_sections:
            if section not in config:
                raise ValueError(f"필수 섹션 누락: {section}")

        if not config["workflows"]:
            raise ValueError("최소 하나의 워크플로우가 필요합니다")

    def resolve_input_files(self, input_config: Dict[str, Any]) -> List[Path]:
        """입력 파일 해석"""
        input_type = input_config.get("type", "single")
        files = []

        if input_type == "single":
            file_path = Path(input_config["file"])
            if file_path.exists():
                files.append(file_path)
            else:
                print(f"⚠️ 파일 없음: {file_path}")

        elif input_type == "glob":
            pattern = input_config["pattern"]
            matched_files = list(self.base_dir.glob(pattern))
            max_files = input_config.get("max_files")

            if max_files:
                matched_files = matched_files[:max_files]

            files.extend(matched_files)

        elif input_type == "list":
            for file_path in input_config.get("files", []):
                path = Path(file_path)
                if path.exists():
                    files.append(path)
                else:
                    print(f"⚠️ 파일 없음: {path}")

        return files

    def check_conditions(self, file_path: Path, conditions: List[Dict]) -> bool:
        """조건 검사"""
        for condition in conditions:
            condition_type = condition["type"]
            operator = condition["operator"]
            value = condition["value"]

            if condition_type == "score_threshold":
                # 파일의 현재 점수 확인
                current_score = self._get_file_score(file_path)

                if operator == "less_than" and current_score >= value:
                    return False
                elif operator == "greater_than" and current_score <= value:
                    return False

            elif condition_type == "file_size":
                file_size = file_path.stat().st_size

                if operator == "less_than" and file_size >= value:
                    return False
                elif operator == "greater_than" and file_size <= value:
                    return False

        return True

    def _get_file_score(self, file_path: Path) -> float:
        """파일의 현재 점수 확인"""
        try:
            temp_output = self.base_dir / f"temp_score_{file_path.stem}.json"

            cmd = [
                "poetry", "run", "python", str(self.scripts_dir / "map_proposal_to_evidence.py"),
                "--proposal", str(file_path),
                "--output", str(temp_output)
            ]

            subprocess.run(cmd, capture_output=True, check=True)

            if temp_output.exists():
                with open(temp_output, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    score = data["summary"]["scientific_rigor_score"]

                temp_output.unlink()
                return score

        except Exception as e:
            print(f"⚠️ 점수 확인 실패 {file_path}: {e}")

        return 0.0

    def execute_workflow(self, workflow: Dict[str, Any],
                        global_settings: Dict[str, Any]) -> Dict[str, Any]:
        """워크플로우 실행"""

        workflow_name = workflow["name"]
        print(f"\n🚀 워크플로우 실행: {workflow_name}")
        print("-" * 50)

        if not workflow.get("enabled", True):
            print("⏸️ 비활성화된 워크플로우 - 건너뜀")
            return {"status": "skipped", "reason": "disabled"}

        # 입력 파일 해석
        input_files = self.resolve_input_files(workflow["input"])

        if not input_files:
            print("❌ 입력 파일 없음")
            return {"status": "failed", "reason": "no_input_files"}

        print(f"📁 입력 파일 {len(input_files)}개:")
        for file in input_files:
            print(f"   • {file.relative_to(self.base_dir)}")

        # 조건 검사
        if "conditions" in workflow:
            filtered_files = []
            for file in input_files:
                if self.check_conditions(file, workflow["conditions"]):
                    filtered_files.append(file)
                else:
                    print(f"⏸️ 조건 불만족으로 건너뜀: {file.name}")

            input_files = filtered_files

        if not input_files:
            print("❌ 조건을 만족하는 파일 없음")
            return {"status": "failed", "reason": "no_files_match_conditions"}

        # 워크플로우 실행
        results = []
        optimization_config = workflow["optimization"]

        # 병렬 실행 여부 결정
        parallel = global_settings.get("parallel", False) and len(input_files) > 1

        if parallel:
            results = self._execute_parallel(input_files, optimization_config, workflow)
        else:
            results = self._execute_sequential(input_files, optimization_config, workflow)

        # 결과 집계
        successful = [r for r in results if r.get("status") == "success"]
        failed = [r for r in results if r.get("status") == "failed"]

        workflow_result = {
            "status": "completed",
            "workflow_name": workflow_name,
            "total_files": len(input_files),
            "successful": len(successful),
            "failed": len(failed),
            "results": results
        }

        print(f"\n📊 워크플로우 완료:")
        print(f"   ✅ 성공: {len(successful)}/{len(input_files)}")
        print(f"   ❌ 실패: {len(failed)}/{len(input_files)}")

        return workflow_result

    def _execute_sequential(self, files: List[Path],
                          optimization_config: Dict[str, Any],
                          workflow: Dict[str, Any]) -> List[Dict[str, Any]]:
        """순차 실행"""
        results = []

        for i, file_path in enumerate(files, 1):
            print(f"\n📄 파일 {i}/{len(files)}: {file_path.name}")

            try:
                result = self._optimize_single_file(file_path, optimization_config, workflow)
                results.append(result)

                if result.get("status") == "success":
                    print(f"   ✅ 성공")
                else:
                    print(f"   ❌ 실패: {result.get('error', '알 수 없는 오류')}")

            except Exception as e:
                print(f"   ❌ 예외 발생: {e}")
                results.append({
                    "status": "failed",
                    "file": str(file_path),
                    "error": str(e)
                })

        return results

    def _execute_parallel(self, files: List[Path],
                         optimization_config: Dict[str, Any],
                         workflow: Dict[str, Any]) -> List[Dict[str, Any]]:
        """병렬 실행"""
        max_workers = min(len(files), 4)  # 최대 4개 동시 실행
        results = []

        print(f"⚡ 병렬 실행 (worker: {max_workers})")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 작업 제출
            future_to_file = {
                executor.submit(self._optimize_single_file, file, optimization_config, workflow): file
                for file in files
            }

            # 결과 수집
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"   ✅ 완료: {file_path.name}")
                except Exception as e:
                    print(f"   ❌ 실패: {file_path.name} - {e}")
                    results.append({
                        "status": "failed",
                        "file": str(file_path),
                        "error": str(e)
                    })

        return results

    def _optimize_single_file(self, file_path: Path,
                            optimization_config: Dict[str, Any],
                            workflow: Dict[str, Any]) -> Dict[str, Any]:
        """단일 파일 최적화"""

        mode = optimization_config.get("mode", "full")
        interactive = optimization_config.get("interactive", False)
        max_iterations = optimization_config.get("max_iterations", 1)
        target_score = optimization_config.get("target_score")

        result = {
            "status": "success",
            "file": str(file_path),
            "iterations": 0,
            "final_score": 0,
            "improvement": 0
        }

        current_file = file_path
        initial_score = self._get_file_score(file_path)

        for iteration in range(max_iterations):
            result["iterations"] = iteration + 1

            # 최적화 실행
            cmd = [
                "poetry", "run", "python", str(self.scripts_dir / "proposal_optimizer.py"),
                "optimize",
                "--input", str(current_file),
                "--mode", mode,
                "--output-prefix", f"batch_{file_path.stem}_iter{iteration + 1}"
            ]

            if not interactive:
                # Non-interactive mode (배치에서는 기본적으로 비대화형)
                pass

            try:
                subprocess.run(cmd, capture_output=True, check=True)

                # 결과 파일 찾기
                output_dir = self.base_dir / "optimization_output"
                sessions = list(output_dir.glob(f"*batch_{file_path.stem}_iter{iteration + 1}*"))

                if sessions:
                    latest_session = max(sessions, key=lambda x: x.stat().st_mtime)
                    optimized_files = list(latest_session.glob("optimized_*"))

                    if optimized_files:
                        current_file = optimized_files[0]

                # 현재 점수 확인
                current_score = self._get_file_score(current_file)
                result["final_score"] = current_score
                result["improvement"] = current_score - initial_score

                # 목표 점수 달성 시 조기 종료
                if target_score and current_score >= target_score:
                    print(f"   🎯 목표 점수 달성: {current_score:.1f} >= {target_score}")
                    break

            except subprocess.CalledProcessError as e:
                result["status"] = "failed"
                result["error"] = f"Optimization failed: {e}"
                break

        return result

    def generate_summary_report(self, batch_results: List[Dict[str, Any]],
                              config: Dict[str, Any]) -> Dict[str, Any]:
        """요약 보고서 생성"""

        summary = {
            "batch_info": config["batch_info"],
            "execution_time": datetime.now().isoformat(),
            "total_workflows": len(batch_results),
            "workflow_results": batch_results,
            "statistics": {
                "total_files_processed": 0,
                "total_successful": 0,
                "total_failed": 0,
                "average_improvement": 0
            }
        }

        # 통계 계산
        total_files = 0
        total_successful = 0
        total_failed = 0
        total_improvement = 0
        improvement_count = 0

        for workflow_result in batch_results:
            if workflow_result.get("status") == "completed":
                total_files += workflow_result.get("total_files", 0)
                total_successful += workflow_result.get("successful", 0)
                total_failed += workflow_result.get("failed", 0)

                for result in workflow_result.get("results", []):
                    if result.get("improvement"):
                        total_improvement += result["improvement"]
                        improvement_count += 1

        summary["statistics"]["total_files_processed"] = total_files
        summary["statistics"]["total_successful"] = total_successful
        summary["statistics"]["total_failed"] = total_failed

        if improvement_count > 0:
            summary["statistics"]["average_improvement"] = total_improvement / improvement_count

        return summary

    def run_batch(self, config_file: str) -> Dict[str, Any]:
        """배치 실행"""
        print("⚙️ 배치 최적화 시작")
        print("="*50)

        start_time = time.time()

        # 설정 로드
        config = self.load_config(config_file)

        batch_info = config["batch_info"]
        print(f"📋 배치: {batch_info['name']}")
        print(f"📝 설명: {batch_info['description']}")

        global_settings = config.get("global_settings", {})
        workflows = config["workflows"]

        # 워크플로우 실행
        batch_results = []

        for workflow in workflows:
            try:
                result = self.execute_workflow(workflow, global_settings)
                batch_results.append(result)

                # 오류 시 중단 여부
                if (result.get("status") == "failed" and
                    not global_settings.get("continue_on_error", True)):
                    print("❌ 오류로 인해 배치 중단")
                    break

            except Exception as e:
                print(f"❌ 워크플로우 실행 오류: {e}")

                if not global_settings.get("continue_on_error", True):
                    break

        # 요약 보고서 생성
        summary = self.generate_summary_report(batch_results, config)

        # 후처리
        post_processing = config.get("post_processing", {})
        if post_processing.get("generate_summary", True):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = self.output_dir / f"batch_summary_{timestamp}.json"

            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            print(f"\n📊 요약 보고서: {summary_file}")

        # 실행 시간 계산
        execution_time = time.time() - start_time
        summary["execution_time_seconds"] = execution_time

        print(f"\n🎉 배치 완료!")
        print(f"   ⏱️ 총 소요 시간: {execution_time/60:.1f}분")
        print(f"   📊 처리된 파일: {summary['statistics']['total_files_processed']}")
        print(f"   ✅ 성공: {summary['statistics']['total_successful']}")
        print(f"   ❌ 실패: {summary['statistics']['total_failed']}")

        if summary['statistics']['average_improvement'] > 0:
            print(f"   📈 평균 개선: +{summary['statistics']['average_improvement']:.1f}점")

        return summary

def main():
    parser = argparse.ArgumentParser(
        description="⚙️ 배치 제안서 최적화 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
    # 설정 파일로 배치 실행
    poetry run python scripts/batch_optimizer.py --config batch_config.yaml

    # 샘플 설정 생성
    poetry run python scripts/batch_optimizer.py --create-sample-config sample.yaml
        """
    )

    parser.add_argument("--config", help="배치 설정 파일 (YAML)")
    parser.add_argument("--create-sample-config", help="샘플 설정 파일 생성")

    args = parser.parse_args()

    try:
        optimizer = BatchOptimizer()

        if args.create_sample_config:
            optimizer.create_sample_config(args.create_sample_config)

        elif args.config:
            if not Path(args.config).exists():
                print(f"❌ 설정 파일 없음: {args.config}")
                return

            summary = optimizer.run_batch(args.config)

        else:
            parser.print_help()

    except Exception as e:
        print(f"❌ 오류: {e}")

if __name__ == "__main__":
    main()