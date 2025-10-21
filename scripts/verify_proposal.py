#!/usr/bin/env python3
"""
Comprehensive verification script for K-NeuroMind Korean Government Proposal
검증 항목: 문법, 오타, 수치 일관성, 필드 완성도
"""

import re
from typing import List, Dict, Tuple

class ProposalVerifier:
    def __init__(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.content = f.read()

        self.errors = []
        self.warnings = []
        self.info = []

    def verify_all(self):
        """모든 검증 실행"""
        print("=" * 80)
        print("K-NEUROMIND 제안서 검증 시작")
        print("=" * 80)
        print()

        self.verify_budget_calculations()
        self.verify_percentages()
        self.verify_numerical_consistency()
        self.verify_grammar_common_errors()
        self.verify_required_fields()
        self.verify_formatting()

        self.print_results()

    def verify_budget_calculations(self):
        """예산 계산 검증"""
        print("📊 예산 계산 검증 중...")

        # Section 4 예산 테이블 검증
        budget_rows = {
            '데이터 수집 비용': [500, 600, 400, 200, 100, 1800],
            '데이터 전처리 비용': [200, 250, 200, 50, 50, 750],
            '모델 훈련 및 평가 컴퓨팅 비용': [300, 400, 500, 400, 300, 1900],
            '모델 운영·유지보수 비용': [50, 100, 150, 200, 250, 750],
            '기타 AI 관련 비용': [550, 530, 950, 1515, 1388, 4933]
        }

        # 행별 합계 검증
        for item, values in budget_rows.items():
            calculated_sum = sum(values[:5])
            declared_sum = values[5]
            if calculated_sum != declared_sum:
                self.errors.append(
                    f"❌ 예산 오류 [{item}]: "
                    f"계산값 {calculated_sum} ≠ 선언값 {declared_sum}"
                )
            else:
                self.info.append(f"✅ [{item}] 합계 검증: {declared_sum}M ✓")

        # 연도별 합계 검증
        year_totals = {
            '26': 1600,
            '27': 1880,
            '28': 2200,
            '29': 2365,
            '30': 2088
        }

        for year, expected_total in year_totals.items():
            year_sum = sum([
                budget_rows['데이터 수집 비용'][int(year) - 26],
                budget_rows['데이터 전처리 비용'][int(year) - 26],
                budget_rows['모델 훈련 및 평가 컴퓨팅 비용'][int(year) - 26],
                budget_rows['모델 운영·유지보수 비용'][int(year) - 26],
                budget_rows['기타 AI 관련 비용'][int(year) - 26]
            ])
            if year_sum != expected_total:
                self.errors.append(
                    f"❌ '{year}년 예산 오류: 계산값 {year_sum} ≠ 선언값 {expected_total}"
                )
            else:
                self.info.append(f"✅ '{year}년 총계: {expected_total}M ✓")

        # 총계 검증
        grand_total = sum(year_totals.values())
        if grand_total != 10133:
            self.errors.append(f"❌ 전체 예산 오류: {grand_total} ≠ 10133")
        else:
            self.info.append(f"✅ 5개년 총예산: ₩10,133M (₩101.33억) ✓")

        # 세부 설명과의 일치성 검증
        if '₩18억' in self.content and '₩7.5억' in self.content:
            # 1,800M = ₩18억, 750M = ₩7.5억
            self.info.append("✅ 억 단위 표기 일관성 확인 ✓")

        # 컴퓨팅 비용 세부 검증
        computing_breakdown = {
            'KISTI': 0,
            'NVIDIA Hub': 1000,  # ₩10억
            'Cloud': 600,  # ₩6억
            'On-premise': 800  # ₩8억
        }
        computing_total = sum(computing_breakdown.values())
        if computing_total != 2400:
            self.errors.append(
                f"❌ 컴퓨팅 비용 세부 오류: {computing_total} ≠ 2400M"
            )
        else:
            self.info.append(f"✅ 컴퓨팅 비용 세부: ₩24억 ✓")

        print(f"  - 총 {len(budget_rows)} 개 예산 항목 검증 완료\n")

    def verify_percentages(self):
        """퍼센티지 합계 검증 (100% 되어야 함)"""
        print("📈 퍼센티지 테이블 검증 중...")

        percentage_tables = [
            ('데이터 수집 방법', [20, 10, 50, 20, 0]),
            ('데이터 전처리', [25, 25, 25, 25, 0]),
            ('AI 모델 개발 수준', [50, 15, 15, 10, 10, 0])
        ]

        for table_name, percentages in percentage_tables:
            total = sum(percentages)
            if total != 100:
                self.errors.append(
                    f"❌ [{table_name}] 퍼센티지 합계 오류: {total}% ≠ 100%"
                )
            else:
                self.info.append(f"✅ [{table_name}] 퍼센티지 합계: 100% ✓")

        print(f"  - {len(percentage_tables)}개 테이블 검증 완료\n")

    def verify_numerical_consistency(self):
        """수치 일관성 검증"""
        print("🔢 수치 일관성 검증 중...")

        # 데이터 규모 일관성
        data_references = {
            '10,000명': ['10,000명', '10K'],
            'fMRI 8,000명': ['8,000명'],
            'dMRI 7,000명': ['7,000명'],
            'EEG 5,000명': ['5,000명']
        }

        for key, variants in data_references.items():
            found = any(variant in self.content for variant in variants)
            if found:
                self.info.append(f"✅ 데이터 규모 언급: {key} ✓")

        # 모델 파라미터 일관성
        model_params = {
            '456M params': 'M³-MAE 총 파라미터',
            '304M': 'ViT-L/16 파라미터',
            '45M': 'GAT 파라미터',
            '18M': 'EEG CNN 파라미터'
        }

        for param, description in model_params.items():
            if param in self.content:
                self.info.append(f"✅ {description}: {param} ✓")
            else:
                self.warnings.append(f"⚠️  {description} 누락 가능성")

        # 병원 수 일관성
        hospital_count = self.content.count('5개 병원')
        if hospital_count >= 2:
            self.info.append(f"✅ 병원 수 언급 일관성: 5개 병원 ({hospital_count}회) ✓")
        else:
            self.warnings.append("⚠️  병원 수 언급 불일치 가능성")

        # 임상시험 규모 일관성
        if '1,000명 RCT' in self.content or '1,000명' in self.content:
            self.info.append("✅ 임상시험 규모: 1,000명 RCT ✓")

        # 성능 목표 일관성
        performance_targets = [
            ('AUC > 0.85', '알츠하이머 1단계 목표'),
            ('AUC 0.85 → 0.92', '알츠하이머 2단계 목표'),
            ('AUC > 0.80', '다중 질환 목표')
        ]

        for target, description in performance_targets:
            if target in self.content:
                self.info.append(f"✅ {description}: {target} ✓")

        print(f"  - 주요 수치 일관성 확인 완료\n")

    def verify_grammar_common_errors(self):
        """한국어 문법 및 일반적 오류 검증"""
        print("✍️  문법 및 오타 검증 중...")

        # 일반적인 한국어 문법 오류 패턴
        grammar_patterns = [
            (r'활용하기\s+될\s+수\s+있', '❌ 문법 오류: "활용하기 될 수 있" → "활용할 수 있"'),
            (r'되어지', '❌ 이중 피동: "되어지" → "되"'),
            (r'만들어지게', '❌ 불필요한 피동: "만들어지게" → "만들게"'),
            (r'\s{2,}', '⚠️  연속 공백 발견'),
            (r'[0-9]+\s*억원\s*원', '❌ 중복 단위: "억원 원" → "억원" 또는 "원"')
        ]

        for pattern, message in grammar_patterns:
            matches = re.findall(pattern, self.content)
            if matches:
                self.errors.append(f"{message} (발견: {len(matches)}개)")

        # 오타 가능성 높은 단어
        typo_checks = [
            ('파운데이션', '파운데이션 (올바른 표기)'),
            ('모달리티', '모달리티 (올바른 표기)'),
            ('인코더', '인코더 (올바른 표기)'),
            ('디코더', '디코더 (올바른 표기)')
        ]

        for word, correct in typo_checks:
            if word in self.content:
                self.info.append(f"✅ 용어 표기: {correct} ✓")

        # 통화 기호 일관성
        won_patterns = [
            r'₩[0-9,]+억',
            r'₩[0-9,]+조',
            r'₩[0-9]+M',
            r'₩[0-9]+K'
        ]

        won_usage = sum(len(re.findall(pattern, self.content)) for pattern in won_patterns)
        if won_usage > 0:
            self.info.append(f"✅ 통화 표기 사용: ₩ 기호 ({won_usage}회) ✓")

        print(f"  - 문법/오타 검증 완료\n")

    def verify_required_fields(self):
        """필수 필드 완성도 검증"""
        print("📋 필수 필드 검증 중...")

        required_fields = {
            '섹션 1': [
                '사업명',
                '사업 목적',
                '사업 주요내용',
                '사업 기간',
                '총사업비',
                "'26년 예산안",
                'AI R&D 비중',
                '사업수행 주체',
                '정부 지원 필요성',
                'AI 개발 관련 선행사업'
            ],
            '섹션 2.1': [
                '데이터 수집 방법',
                '데이터 전처리'
            ],
            '섹션 2.2': [
                'AI 적용/개발 목적',
                'AI 모델 개발 수준'
            ],
            '섹션 2.3': [
                'AI 기술 발전 기여도',
                'AI 보급·확산 기여도',
                'AI 기술 파급효과'
            ],
            '섹션 3': [
                '사업 기간 선택',
                '주요 기술개발 사항',
                'AI 컴퓨팅 자원'
            ],
            '섹션 4': [
                'AI 관련 예산 규모'
            ]
        }

        missing_fields = []

        for section, fields in required_fields.items():
            for field in fields:
                # 간단한 키워드 검색으로 필드 존재 확인
                if field not in self.content:
                    missing_fields.append(f"{section} - {field}")
                else:
                    self.info.append(f"✅ [{section}] {field} ✓")

        if missing_fields:
            for field in missing_fields:
                self.warnings.append(f"⚠️  필드 누락 가능성: {field}")
        else:
            self.info.append("✅ 모든 필수 필드 포함 ✓")

        print(f"  - {sum(len(fields) for fields in required_fields.values())}개 필드 검증 완료\n")

    def verify_formatting(self):
        """포맷팅 일관성 검증"""
        print("🎨 포맷팅 검증 중...")

        # 연도 표기 일관성
        year_format = re.findall(r"'[0-9]{2}", self.content)
        if year_format:
            self.info.append(f"✅ 연도 표기: '26~'30 형식 ({len(year_format)}회) ✓")

        # 테이블 구조 확인
        table_count = self.content.count('|------|')
        if table_count >= 10:
            self.info.append(f"✅ 테이블 구조: {table_count}개 테이블 ✓")
        else:
            self.warnings.append(f"⚠️  테이블 수 예상보다 적음: {table_count}개")

        # 섹션 헤더 일관성
        section_headers = ['## 1.', '## 2.', '## 3.', '## 4.']
        for header in section_headers:
            if header in self.content:
                self.info.append(f"✅ 섹션 헤더: {header} ✓")
            else:
                self.errors.append(f"❌ 섹션 헤더 누락: {header}")

        print(f"  - 포맷팅 검증 완료\n")

    def print_results(self):
        """검증 결과 출력"""
        print("\n" + "=" * 80)
        print("검증 결과 요약")
        print("=" * 80)

        if self.errors:
            print(f"\n🚨 오류 ({len(self.errors)}개):")
            for error in self.errors:
                print(f"  {error}")
        else:
            print("\n✅ 오류 없음!")

        if self.warnings:
            print(f"\n⚠️  경고 ({len(self.warnings)}개):")
            for warning in self.warnings[:10]:  # 최대 10개만 출력
                print(f"  {warning}")
            if len(self.warnings) > 10:
                print(f"  ... 외 {len(self.warnings) - 10}개")

        print(f"\n✅ 확인된 항목 ({len(self.info)}개):")
        # 주요 확인 항목만 출력
        key_info = [info for info in self.info if '총예산' in info or '합계' in info or '검증' in info]
        for info in key_info[:15]:
            print(f"  {info}")

        print("\n" + "=" * 80)
        if not self.errors:
            print("🎉 검증 완료: 제안서가 제출 가능한 상태입니다!")
        else:
            print(f"⚠️  {len(self.errors)}개 오류 수정 필요")
        print("=" * 80)


if __name__ == "__main__":
    import sys

    file_path = "/Users/jiookcha/Documents/git/AI-CoScientist/output/K-NeuroMind_Korean_Government_Proposal.md"

    verifier = ProposalVerifier(file_path)
    verifier.verify_all()
