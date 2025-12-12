#!/usr/bin/env python3
"""
grant.hwpx의 실제 텍스트 내용을 추출하는 스크립트
기존 내용을 파악하여 확장/보강 방향 결정
"""

import xml.etree.ElementTree as ET
import re

def extract_all_text(element):
    """XML 요소에서 모든 텍스트 추출"""
    text = []
    if element.text:
        text.append(element.text.strip())
    for child in element:
        text.extend(extract_all_text(child))
        if child.tail:
            text.append(child.tail.strip())
    return [t for t in text if t]

def extract_content_from_hwpx():
    """grant.hwpx에서 실제 내용 추출"""

    section_xml = '/tmp/hwpx_extract/Contents/section0.xml'

    # XML 파싱
    tree = ET.parse(section_xml)
    root = tree.getroot()

    # 모든 텍스트 추출
    all_texts = extract_all_text(root)

    # 중복 제거 및 정리
    unique_texts = []
    seen = set()
    for text in all_texts:
        if text and len(text) > 1 and text not in seen:
            unique_texts.append(text)
            seen.add(text)

    # 섹션별로 그룹핑 시도
    current_section = "문서 시작"
    sections = {current_section: []}

    section_patterns = [
        r'^1\.',
        r'^2\.',
        r'^3\.',
        r'^4\.',
        r'^섹션',
        r'^Section',
        r'사업 내용',
        r'인공지능 R&D',
        r'기술개발',
        r'예산'
    ]

    for text in unique_texts:
        # 섹션 시작 감지
        is_section = False
        for pattern in section_patterns:
            if re.search(pattern, text):
                current_section = text[:50]  # 섹션명은 처음 50자
                if current_section not in sections:
                    sections[current_section] = []
                is_section = True
                break

        if not is_section:
            sections[current_section].append(text)

    # 결과 출력
    print("=" * 80)
    print("GRANT.HWPX 기존 내용 추출 결과")
    print("=" * 80)
    print()

    for section_name, contents in sections.items():
        print(f"\n{'='*60}")
        print(f"📌 {section_name}")
        print(f"{'='*60}")

        for i, content in enumerate(contents, 1):
            if len(content) > 10:  # 의미있는 내용만
                print(f"{i}. {content[:200]}")
                if len(content) > 200:
                    print("   [...]")

    # 파일로도 저장
    output_file = '/Users/jiookcha/Documents/git/AI-CoScientist/output/grant_hwpx_기존내용.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("GRANT.HWPX 기존 내용 전체 추출\n")
        f.write("=" * 80 + "\n\n")

        for section_name, contents in sections.items():
            f.write(f"\n{'='*60}\n")
            f.write(f"📌 {section_name}\n")
            f.write(f"{'='*60}\n\n")

            for content in contents:
                if len(content) > 10:
                    f.write(f"{content}\n\n")

    print(f"\n\n✅ 전체 내용 저장: {output_file}")
    print(f"총 {len(unique_texts)}개 텍스트 항목 추출")
    print(f"총 {len(sections)}개 섹션 발견")

    # 통계
    total_chars = sum(len(text) for texts in sections.values() for text in texts)
    print(f"총 문자 수: {total_chars:,}자")

if __name__ == "__main__":
    extract_content_from_hwpx()
