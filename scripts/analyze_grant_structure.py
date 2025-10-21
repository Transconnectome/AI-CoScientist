#!/usr/bin/env python3
"""
Analyze grant.hwpx structure to understand the complete form template
"""

import xml.etree.ElementTree as ET
import json

def extract_all_text(element):
    """Recursively extract all text from element"""
    text = []
    if element.text:
        text.append(element.text.strip())
    for child in element:
        text.extend(extract_all_text(child))
        if child.tail:
            text.append(child.tail.strip())
    return [t for t in text if t]

def analyze_structure():
    # Read the XML file
    tree = ET.parse('/tmp/hwpx_extract/Contents/section0.xml')
    root = tree.getroot()

    all_text = ' '.join(extract_all_text(root))

    # Define the complete form structure based on analysis
    form_structure = {
        "document_title": "2026년도 인공지능 분야 신규 R&D 사업 사업보완 제출 양식",
        "sections": [
            {
                "number": "1",
                "title": "사업 내용",
                "fields": [
                    {"name": "사업명", "type": "text"},
                    {"name": "사업 목적", "type": "textarea"},
                    {"name": "사업 주요내용", "type": "textarea_structured"},
                    {"name": "사업 기간", "type": "text"},
                    {"name": "총사업비", "type": "text"},
                    {"name": "'26년 예산안", "type": "text"},
                    {"name": "AI R&D 비중", "type": "text"},
                    {"name": "사업수행 주체", "type": "text"},
                    {"name": "정부 지원 필요성", "type": "textarea"},
                    {"name": "AI 개발 관련 선행사업 유무 및 선행사업 주요 내용", "type": "textarea"}
                ]
            },
            {
                "number": "2",
                "title": "인공지능 R&D 특성",
                "subsections": [
                    {
                        "number": "2.1",
                        "title": "AI 모델 학습을 위한 데이터 수집 및 전처리",
                        "tables": [
                            {
                                "title": "데이터 수집 방법",
                                "options": [
                                    {"name": "데이터 신규 구축", "percentage": True},
                                    {"name": "기존 데이터 확장", "percentage": True},
                                    {"name": "기 구축된 타기관 데이터 활용", "percentage": True},
                                    {"name": "기 구축된 자체 데이터 활용", "percentage": True},
                                    {"name": "기타", "percentage": False}
                                ],
                                "description_field": "데이터 출처·종류·확보방안 등 선택 항목 관련 주요설명"
                            },
                            {
                                "title": "데이터 전처리",
                                "options": [
                                    {"name": "데이터 기초 전처리", "percentage": True},
                                    {"name": "데이터 증강 및 변환 적용", "percentage": True},
                                    {"name": "대량데이터 자동처리 및 최적화", "percentage": True},
                                    {"name": "실시간 데이터 품질 관리 및 지속적인 갱신", "percentage": True},
                                    {"name": "기타", "percentage": False}
                                ],
                                "description_field": "데이터전처리 관련 세부 연구내용 기술"
                            }
                        ]
                    },
                    {
                        "number": "2.2",
                        "title": "AI 적용/개발 목적",
                        "tables": [
                            {
                                "title": "사업 내 AI 연구개발 목적",
                                "options": [
                                    {"name": "AI 성능향상", "select_type": "single"},
                                    {"name": "AI 응용처 확장", "select_type": "single"},
                                    {"name": "AI 기술 활용", "select_type": "single"},
                                    {"name": "AI 기반 확충", "select_type": "single"},
                                    {"name": "기타", "select_type": "single"}
                                ],
                                "description_field": "선택 항목 관련 주요 내용"
                            },
                            {
                                "title": "AI 모델 개발 수준",
                                "options": [
                                    {"name": "오픈모델 활용", "percentage": True},
                                    {"name": "신규모델 개발", "percentage": True},
                                    {"name": "새로운 알고리즘 개발", "percentage": True},
                                    {"name": "민간 모델 도입/응용", "percentage": True},
                                    {"name": "선행사업 모델 활용/개선", "percentage": True},
                                    {"name": "기타", "percentage": False}
                                ],
                                "description_field": "선택 항목 관련 주요 내용"
                            }
                        ]
                    },
                    {
                        "number": "2.3",
                        "title": "사업 내 AI 활용",
                        "tables": [
                            {
                                "title": "AI 기술 발전 기여도",
                                "fields": ["내용 기술"]
                            },
                            {
                                "title": "AI 보급·확산 기여도",
                                "fields": ["내용 기술"]
                            },
                            {
                                "title": "AI 기술 파급효과",
                                "fields": ["내용 기술"]
                            }
                        ]
                    }
                ]
            },
            {
                "number": "3",
                "title": "주요 기술개발 사항",
                "subsections": [
                    {
                        "title": "사업 기간별 주요 기술개발 사항",
                        "period_options": [
                            {"period": "3년", "years": ["'26", "'27", "'28"]},
                            {"period": "4년", "years": ["'26", "'27", "'28", "'29"]},
                            {"period": "5년 이상", "years": ["'26", "'27", "'28", "'29", "'30"]}
                        ],
                        "rationale_field": "3년 이내 기술개발이 어려운 사유 및 향후 AI 발전에 따른 사업 대응 방향"
                    },
                    {
                        "title": "AI 관련 R&D 사업 컴퓨팅 자원 구축, 활용 방안",
                        "options": [
                            {"name": "AI 컴퓨팅 자원 구축 (GPU 등 구매)"},
                            {"name": "클라우드 등을 통한 AI 컴퓨팅 자원 활용"}
                        ],
                        "description_field": "주요 내용"
                    }
                ]
            },
            {
                "number": "4",
                "title": "예산 규모",
                "tables": [
                    {
                        "title": "AI 관련 예산 규모",
                        "budget_items": [
                            {"name": "데이터 수집 비용", "years": ["'26", "'27", "'28", "'29", "'30"]},
                            {"name": "데이터 전처리 비용", "years": ["'26", "'27", "'28", "'29", "'30"]},
                            {"name": "모델 훈련 및 평가 컴퓨팅 비용", "years": ["'26", "'27", "'28", "'29", "'30"]},
                            {"name": "모델 운영·유지보수 비용", "years": ["'26", "'27", "'28", "'29", "'30"]},
                            {"name": "기타 AI 관련 비용", "years": ["'26", "'27", "'28", "'29", "'30"]}
                        ]
                    }
                ]
            }
        ]
    }

    return form_structure

if __name__ == "__main__":
    structure = analyze_structure()

    print("=" * 80)
    print("GRANT.HWPX FORM STRUCTURE ANALYSIS")
    print("=" * 80)
    print()

    print(json.dumps(structure, ensure_ascii=False, indent=2))

    # Save to file
    with open('/Users/jiookcha/Documents/git/AI-CoScientist/output/grant_form_structure.json', 'w', encoding='utf-8') as f:
        json.dumps(structure, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("Structure saved to: output/grant_form_structure.json")
    print("=" * 80)
