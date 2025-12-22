#!/usr/bin/env python3
"""
NRF 중견연구 제출 체크리스트 자동 검증 시스템
===========================================

제출 전 필수 요건을 자동으로 검증합니다:
- 페이지 수 (10페이지 이내)
- 필수 섹션 존재 여부
- 필수 그림 4종 포함 여부
- 서식 요건 준수

Usage:
    # 기본 검증
    poetry run python scripts/validate_submission_checklist.py \
        --input "proposal.md"

    # 상세 보고서 출력
    poetry run python scripts/validate_submission_checklist.py \
        --input "proposal.md" \
        --output "checklist_report.json" \
        --verbose

    # NRF 중견 전용 체크리스트
    poetry run python scripts/validate_submission_checklist.py \
        --input "proposal.md" \
        --checklist NRF_midcareer
"""

import argparse
import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from enum import Enum

class CheckStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    SKIPPED = "SKIPPED"

@dataclass
class CheckResult:
    """Individual check result"""
    name: str
    status: CheckStatus
    message: str
    details: Optional[str] = None
    auto_fix_available: bool = False

@dataclass
class ValidationReport:
    """Complete validation report"""
    input_file: str
    checklist_type: str
    timestamp: str
    overall_status: CheckStatus
    total_checks: int
    passed: int
    failed: int
    warnings: int
    checks: List[CheckResult] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

class NRFMidcareerChecklist:
    """NRF 중견연구 제출 체크리스트"""

    # 필수 섹션 (공식 양식 기준)
    REQUIRED_SECTIONS = [
        ("1. 연구과제의 필요성", ["연구과제의 필요성", "필요성", "1.", "1 ."]),
        ("2. 연구과제의 목표 및 내용", ["연구과제의 목표", "목표 및 내용", "2.", "2 ."]),
        ("3. 추진전략·방법 및 추진체계", ["추진전략", "추진체계", "3.", "3 ."]),
        ("4. 연구자의 연구 수행역량", ["수행역량", "연구역량", "4.", "4 ."]),
        ("5. 활용방안 및 기대효과", ["활용방안", "기대효과", "5.", "5 ."]),
    ]

    # 필수 그림
    REQUIRED_FIGURES = [
        ("Fig 1: 문제-갭-가설-기여", ["fig", "figure", "그림", "문제", "갭", "가설"]),
        ("Fig 2: 방법 파이프라인", ["fig", "figure", "그림", "방법", "파이프라인", "워크플로"]),
        ("Fig 3: Aim별 실험 흐름", ["fig", "figure", "그림", "aim", "실험", "흐름"]),
        ("Fig 4: Gantt 차트", ["fig", "figure", "그림", "gantt", "간트", "일정"]),
    ]

    # 페이지 제한
    MAX_PAGES = 10
    CHARS_PER_PAGE = 2000  # 대략적인 추정 (한글 기준)

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.checks: List[CheckResult] = []

    def validate(self, content: str, figures_dir: Optional[Path] = None) -> ValidationReport:
        """전체 체크리스트 검증 실행"""
        self.checks = []

        # 1. 페이지 수 검사
        self._check_page_count(content)

        # 2. 필수 섹션 검사
        self._check_required_sections(content)

        # 3. 필수 그림 검사
        self._check_required_figures(content, figures_dir)

        # 4. 서식 검사
        self._check_formatting(content)

        # 5. 콘텐츠 품질 검사
        self._check_content_quality(content)

        # 6. 제출 시간 검사
        self._check_submission_time()

        # 리포트 생성
        return self._generate_report()

    def _check_page_count(self, content: str):
        """페이지 수 검사 (10페이지 이내)"""
        # 대략적인 페이지 수 추정
        char_count = len(content)
        estimated_pages = char_count / self.CHARS_PER_PAGE

        # 줄 수 기반 추정 (한 페이지 약 40줄)
        line_count = len(content.split('\n'))
        line_based_pages = line_count / 40

        # 두 추정치의 평균
        avg_pages = (estimated_pages + line_based_pages) / 2

        if avg_pages <= self.MAX_PAGES:
            status = CheckStatus.PASS
            message = f"페이지 수 적정 (약 {avg_pages:.1f}페이지 / 최대 {self.MAX_PAGES}페이지)"
        elif avg_pages <= self.MAX_PAGES * 1.1:  # 10% 초과
            status = CheckStatus.WARNING
            message = f"페이지 수 경계선 (약 {avg_pages:.1f}페이지) - 줄이기 권장"
        else:
            status = CheckStatus.FAIL
            message = f"페이지 수 초과! (약 {avg_pages:.1f}페이지 > {self.MAX_PAGES}페이지)"

        self.checks.append(CheckResult(
            name="페이지 수 검사",
            status=status,
            message=message,
            details=f"문자 수: {char_count:,}, 줄 수: {line_count:,}"
        ))

    def _check_required_sections(self, content: str):
        """필수 섹션 존재 여부 검사"""
        content_lower = content.lower()

        for section_name, keywords in self.REQUIRED_SECTIONS:
            found = any(kw.lower() in content_lower for kw in keywords)

            if found:
                status = CheckStatus.PASS
                message = f"'{section_name}' 섹션 발견됨"
            else:
                status = CheckStatus.FAIL
                message = f"'{section_name}' 섹션 누락!"

            self.checks.append(CheckResult(
                name=f"섹션: {section_name}",
                status=status,
                message=message,
                auto_fix_available=not found
            ))

    def _check_required_figures(self, content: str, figures_dir: Optional[Path] = None):
        """필수 그림 4종 검사"""
        content_lower = content.lower()

        for fig_name, keywords in self.REQUIRED_FIGURES:
            # 텍스트 내 참조 확인
            text_ref = sum(1 for kw in keywords if kw.lower() in content_lower)
            has_text_ref = text_ref >= 2  # 최소 2개 키워드 매칭

            # 실제 파일 존재 확인
            has_file = False
            if figures_dir and figures_dir.exists():
                for ext in ['.png', '.jpg', '.jpeg', '.pdf', '.svg']:
                    fig_files = list(figures_dir.glob(f"*{ext}"))
                    if fig_files:
                        has_file = True
                        break

            if has_text_ref or has_file:
                status = CheckStatus.PASS
                message = f"'{fig_name}' 포함됨"
                if has_file:
                    message += " (파일 확인됨)"
            else:
                status = CheckStatus.WARNING
                message = f"'{fig_name}' 참조 불명확 - 확인 필요"

            self.checks.append(CheckResult(
                name=f"그림: {fig_name}",
                status=status,
                message=message
            ))

    def _check_formatting(self, content: str):
        """서식 요건 검사"""
        issues = []

        # 이모지 검사 (제안서에 이모지 사용 금지)
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
        emojis = emoji_pattern.findall(content)
        if emojis:
            issues.append(f"이모지 {len(emojis)}개 발견 - 제거 필요")

        # 작성요령 텍스트 잔존 검사
        guide_patterns = ["작성요령", "※ 삭제", "본 항목은", "양식에 따라"]
        found_guides = [p for p in guide_patterns if p in content]
        if found_guides:
            issues.append(f"작성요령 텍스트 잔존: {found_guides}")

        if not issues:
            status = CheckStatus.PASS
            message = "서식 요건 준수"
        else:
            status = CheckStatus.FAIL
            message = "; ".join(issues)

        self.checks.append(CheckResult(
            name="서식 검사",
            status=status,
            message=message,
            auto_fix_available=bool(emojis)
        ))

    def _check_content_quality(self, content: str):
        """콘텐츠 품질 검사"""
        issues = []

        # 참고문헌 섹션 확인
        if "참고문헌" not in content and "references" not in content.lower():
            issues.append("참고문헌 섹션 누락")

        # 최소 내용량 확인
        if len(content) < 5000:
            issues.append("내용이 너무 짧음 (최소 5000자 권장)")

        # 정량지표 존재 확인
        quantitative_patterns = [r'\d+%', r'>\s*\d+', r'<\s*\d+', r'N\s*[>=<]\s*\d+', r'AUC', r'MAE', r'R\^?2']
        has_metrics = any(re.search(p, content) for p in quantitative_patterns)
        if not has_metrics:
            issues.append("정량적 성공기준(지표) 부족")

        if not issues:
            status = CheckStatus.PASS
            message = "콘텐츠 품질 양호"
        else:
            status = CheckStatus.WARNING
            message = "; ".join(issues)

        self.checks.append(CheckResult(
            name="콘텐츠 품질",
            status=status,
            message=message
        ))

    def _check_submission_time(self):
        """제출 시간 검사 (18:00 마감 알림)"""
        now = datetime.now()
        hour = now.hour

        if hour >= 17:
            status = CheckStatus.WARNING
            message = f"현재 {now.strftime('%H:%M')} - 18:00 마감 임박! 즉시 제출 권장"
        elif hour >= 15:
            status = CheckStatus.WARNING
            message = f"현재 {now.strftime('%H:%M')} - 제출 마감 3시간 이내"
        else:
            status = CheckStatus.PASS
            message = f"현재 {now.strftime('%H:%M')} - 제출 시간 여유 있음"

        self.checks.append(CheckResult(
            name="제출 시간",
            status=status,
            message=message,
            details="마감: 18:00:00 (시스템 자동 차단)"
        ))

    def _generate_report(self) -> ValidationReport:
        """검증 리포트 생성"""
        passed = sum(1 for c in self.checks if c.status == CheckStatus.PASS)
        failed = sum(1 for c in self.checks if c.status == CheckStatus.FAIL)
        warnings = sum(1 for c in self.checks if c.status == CheckStatus.WARNING)

        # 전체 상태 결정
        if failed > 0:
            overall = CheckStatus.FAIL
        elif warnings > 0:
            overall = CheckStatus.WARNING
        else:
            overall = CheckStatus.PASS

        # 권장사항 생성
        recommendations = []
        for check in self.checks:
            if check.status == CheckStatus.FAIL:
                recommendations.append(f"[필수] {check.name}: {check.message}")
            elif check.status == CheckStatus.WARNING:
                recommendations.append(f"[권장] {check.name}: {check.message}")

        return ValidationReport(
            input_file="",  # Will be set by caller
            checklist_type="NRF_midcareer",
            timestamp=datetime.now().isoformat(),
            overall_status=overall,
            total_checks=len(self.checks),
            passed=passed,
            failed=failed,
            warnings=warnings,
            checks=self.checks,
            recommendations=recommendations
        )


def print_report(report: ValidationReport):
    """리포트 출력"""
    print()
    print("=" * 70)
    print("  NRF 중견연구 제출 체크리스트 검증 결과")
    print("=" * 70)
    print(f"  파일: {report.input_file}")
    print(f"  검증 시간: {report.timestamp}")
    print()

    # 요약
    status_emoji = {
        CheckStatus.PASS: "[PASS]",
        CheckStatus.FAIL: "[FAIL]",
        CheckStatus.WARNING: "[WARN]"
    }

    print(f"  전체 상태: {status_emoji.get(report.overall_status, '?')} {report.overall_status.value}")
    print(f"  총 {report.total_checks}개 항목: {report.passed} 통과 / {report.failed} 실패 / {report.warnings} 경고")
    print()

    # 상세 결과
    print("-" * 70)
    print("  상세 검증 결과:")
    print("-" * 70)

    for check in report.checks:
        emoji = status_emoji.get(check.status, "?")
        print(f"  {emoji} {check.name}")
        print(f"      -> {check.message}")
        if check.details:
            print(f"      ({check.details})")

    # 권장사항
    if report.recommendations:
        print()
        print("-" * 70)
        print("  조치 필요 사항:")
        print("-" * 70)
        for rec in report.recommendations:
            print(f"  - {rec}")

    print()
    print("=" * 70)

    # 최종 판정
    if report.overall_status == CheckStatus.PASS:
        print("  제출 준비 완료")
    elif report.overall_status == CheckStatus.WARNING:
        print("  주의: 일부 항목 확인 필요 (제출 가능)")
    else:
        print("  제출 불가: 필수 항목 수정 필요!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="NRF 중견연구 제출 체크리스트 자동 검증"
    )
    parser.add_argument("--input", "-i", required=True,
                       help="검증할 제안서 파일 (md/txt)")
    parser.add_argument("--output", "-o",
                       help="검증 결과 저장 파일 (JSON)")
    parser.add_argument("--checklist", choices=["NRF_midcareer"],
                       default="NRF_midcareer",
                       help="체크리스트 유형")
    parser.add_argument("--figures-dir",
                       help="그림 파일 디렉토리")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="상세 출력")

    args = parser.parse_args()

    # 입력 파일 확인
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: 파일을 찾을 수 없습니다: {args.input}")
        sys.exit(1)

    # 콘텐츠 읽기
    content = input_path.read_text(encoding='utf-8')

    # 그림 디렉토리
    figures_dir = Path(args.figures_dir) if args.figures_dir else None

    # 체크리스트 실행
    checker = NRFMidcareerChecklist(verbose=args.verbose)
    report = checker.validate(content, figures_dir)
    report.input_file = str(args.input)

    # 결과 출력
    print_report(report)

    # JSON 저장
    if args.output:
        output_data = asdict(report)
        # Enum을 문자열로 변환
        output_data['overall_status'] = report.overall_status.value
        output_data['checks'] = [
            {**asdict(c), 'status': c.status.value}
            for c in report.checks
        ]
        Path(args.output).write_text(
            json.dumps(output_data, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
        print(f"\n결과 저장: {args.output}")

    # 종료 코드
    if report.overall_status == CheckStatus.FAIL:
        sys.exit(1)
    elif report.overall_status == CheckStatus.WARNING:
        sys.exit(0)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
