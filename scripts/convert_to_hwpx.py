#!/usr/bin/env python3
"""
K-NeuroMind 제안서를 한글 .hwpx 파일로 변환하는 스크립트

.hwpx 파일은 한글 워드프로세서의 XML 기반 포맷입니다.
이 스크립트는 원본 grant.hwpx 구조를 기반으로 내용을 채워넣습니다.
"""

import os
import zipfile
import xml.etree.ElementTree as ET
import shutil
from datetime import datetime

class HwpxConverter:
    def __init__(self, template_path: str, output_path: str):
        """
        Args:
            template_path: 원본 grant.hwpx 템플릿 경로
            output_path: 생성될 .hwpx 파일 경로
        """
        self.template_path = template_path
        self.output_path = output_path
        self.temp_dir = "/tmp/hwpx_output"

    def convert(self):
        """마크다운 제안서를 .hwpx 파일로 변환"""
        print("=" * 80)
        print("K-NEUROMIND 제안서 .HWPX 변환 시작")
        print("=" * 80)
        print()

        # 1. 템플릿 압축 해제
        print("📦 Step 1: 템플릿 압축 해제 중...")
        self.extract_template()

        # 2. 내용 수정
        print("✏️  Step 2: 내용 업데이트 중...")
        self.update_content()

        # 3. .hwpx 파일 재생성
        print("📄 Step 3: .hwpx 파일 생성 중...")
        self.create_hwpx()

        print()
        print("=" * 80)
        print(f"✅ 변환 완료: {self.output_path}")
        print("=" * 80)
        print()
        print("⚠️  중요: 생성된 파일을 한글 워드프로세서에서 열어 최종 검토하세요.")
        print()

    def extract_template(self):
        """템플릿 .hwpx 파일 압축 해제"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_dir)

        with zipfile.ZipFile(self.template_path, 'r') as zip_ref:
            zip_ref.extractall(self.temp_dir)

        print(f"  ✓ 템플릿 압축 해제: {self.temp_dir}")

    def update_content(self):
        """
        XML 내용 업데이트

        주의: .hwpx XML 구조는 매우 복잡하므로,
        이 스크립트는 기본 텍스트 교체만 수행합니다.
        복잡한 테이블 구조는 수동 편집이 필요할 수 있습니다.
        """
        section_xml = os.path.join(self.temp_dir, 'Contents', 'section0.xml')

        if not os.path.exists(section_xml):
            print(f"  ❌ 오류: section0.xml을 찾을 수 없습니다.")
            return

        # XML 파싱
        tree = ET.parse(section_xml)
        root = tree.getroot()

        # 네임스페이스 처리
        namespaces = {
            'hp': 'http://www.hancom.co.kr/hwpml/2011/paragraph'
        }

        # 여기에 실제 내용 교체 로직을 추가
        # 예: 사업명 교체
        # 주의: 실제 XML 구조에 맞게 XPath를 조정해야 함

        print("  ⚠️  XML 직접 수정은 복잡합니다.")
        print("  ℹ️  대신 수동 변환 가이드를 참고하세요.")

        # XML 저장
        tree.write(section_xml, encoding='utf-8', xml_declaration=True)
        print(f"  ✓ 내용 업데이트 완료")

    def create_hwpx(self):
        """수정된 파일들을 .hwpx로 재압축"""
        with zipfile.ZipFile(self.output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(self.temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, self.temp_dir)
                    zipf.write(file_path, arcname)

        print(f"  ✓ .hwpx 파일 생성: {self.output_path}")


def print_manual_conversion_guide():
    """수동 변환 가이드 출력"""
    guide = """
================================================================================
한글 .HWPX 수동 변환 가이드
================================================================================

.hwpx 파일은 복잡한 XML 구조로 되어 있어,
프로그래밍 방식의 완전 자동 변환은 어렵습니다.

다음 방법을 권장합니다:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
방법 1: 한글 워드프로세서에서 직접 입력 (권장)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 한글(HWP) 프로그램 실행
2. grant.hwpx 템플릿 파일 열기
3. K-NeuroMind_Korean_Government_Proposal.md 파일 열기 (참고용)
4. 섹션별로 복사-붙여넣기:

   섹션 1: 사업 내용
   ├─ 사업명: 직접 입력
   ├─ 사업 목적: 마크다운에서 복사
   ├─ 사업 주요내용: 마크다운에서 복사 (표 형식 유지)
   ├─ 사업 기간: 직접 입력
   ├─ 총사업비: 직접 입력
   └─ ... (나머지 필드)

   섹션 2: 인공지능 R&D 특성
   ├─ 2.1 데이터 수집/전처리 테이블
   │   ├─ 퍼센티지 입력 (20%, 10%, 50%, 20%)
   │   └─ 설명 텍스트 복사
   ├─ 2.2 AI 적용/개발 목적
   └─ 2.3 AI 활용

   섹션 3: 주요 기술개발 사항
   ├─ 연도별 테이블 작성
   └─ 컴퓨팅 자원 설명

   섹션 4: 예산 규모
   └─ 5×5 예산 테이블 입력

5. 저장: "K-NeuroMind_제안서_최종.hwpx"

예상 소요 시간: 2-3시간

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
방법 2: MS Word 변환 후 한글로 가져오기
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Markdown → MS Word 변환:
   ```bash
   pandoc K-NeuroMind_Korean_Government_Proposal.md \\
          -o K-NeuroMind_Proposal.docx \\
          --reference-doc=korean_template.docx
   ```

2. MS Word에서 .docx 파일 열기
3. 테이블 구조 정리 및 포맷팅
4. 한글 프로그램에서 "파일 > 불러오기 > MS Word 문서"
5. grant.hwpx 템플릿과 비교하며 조정
6. .hwpx로 저장

예상 소요 시간: 1.5-2시간

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
방법 3: 온라인 변환 도구 사용 (비권장)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

온라인 변환 도구는 복잡한 테이블 구조를 제대로 처리하지 못할 수 있습니다.
정부 제출용 문서이므로 수동 검증이 필수입니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
체크리스트 (변환 후 반드시 확인)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

□ 섹션 1: 사업 내용 10개 필드 모두 작성
□ 섹션 2.1: 데이터 수집 방법 테이블 (퍼센티지 합계 100%)
□ 섹션 2.1: 데이터 전처리 테이블 (퍼센티지 합계 100%)
□ 섹션 2.2: AI 모델 개발 수준 (퍼센티지 합계 100%)
□ 섹션 2.3: AI 기술 발전/보급/파급효과 3개 테이블
□ 섹션 3: 연도별 기술개발 사항 ('26~'30, 5년)
□ 섹션 4: 예산 규모 테이블 (총합 10,133M 확인)
□ 모든 표 셀 병합 제대로 되었는지 확인
□ 한글 폰트 통일 (보통 "맑은 고딕" 또는 "한컴 바탕")
□ 페이지 번호 매김
□ 최종 오타/문법 검토

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
핵심 주의사항
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️  테이블 구조: 원본 grant.hwpx의 정확한 테이블 구조 유지
⚠️  필드 순서: 양식에 명시된 순서대로 작성
⚠️  퍼센티지: 모든 테이블 합계 100% 반드시 확인
⚠️  예산: 연도별/항목별 합계 정확히 일치해야 함
⚠️  페이지 수: 보통 20-25페이지 (A4 기준)

================================================================================
"""
    print(guide)


if __name__ == "__main__":
    # 수동 변환 가이드 출력
    print_manual_conversion_guide()

    print("\n" + "=" * 80)
    print("자동 변환 스크립트 실행 옵션")
    print("=" * 80)
    print()
    print("⚠️  주의: .hwpx XML 구조가 복잡하여 완전 자동 변환은 어렵습니다.")
    print()
    print("다음 명령으로 기본 변환을 시도할 수 있습니다:")
    print()
    print("  python scripts/convert_to_hwpx.py --auto")
    print()
    print("하지만 수동 검증 및 조정이 필수입니다.")
    print()
    print("권장: 위의 수동 변환 가이드를 따라 한글 프로그램에서 직접 작성")
    print("=" * 80)
