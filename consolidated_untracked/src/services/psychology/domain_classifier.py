"""
Psychology Domain Classifier
심리학 연구 영역 자동 분류 시스템

Features:
1. 다양한 심리학 하위 분야 분류
2. 한국어/영어 키워드 기반 분류
3. 연구 방법론 식별
4. 대상 인구 분류
5. 신뢰도 점수 제공
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class DomainClassification:
    """도메인 분류 결과"""
    primary_domain: str
    confidence: float
    secondary_domains: List[Tuple[str, float]]
    methodology: str = ""
    target_population: str = ""
    keywords_found: List[str] = None

    def __post_init__(self):
        if self.keywords_found is None:
            self.keywords_found = []


class PsychologyDomainClassifier:
    """심리학 도메인 분류기"""

    def __init__(self):
        # 심리학 하위 분야별 키워드 사전
        self.domain_keywords = {
            'developmental_psychology': {
                'korean': [
                    '아동', '청소년', '발달', '성장', '언어발달', '인지발달',
                    '사회성발달', '정서발달', '애착', '유아', '아기', '성인기',
                    '노년기', '발달장애', '학습장애'
                ],
                'english': [
                    'child', 'children', 'adolescent', 'development', 'developmental',
                    'attachment', 'infant', 'toddler', 'cognitive development',
                    'language development', 'social development', 'aging'
                ],
                'weight': 1.0
            },

            'cognitive_psychology': {
                'korean': [
                    '인지', '기억', '주의', '학습', '사고', '판단', '의사결정',
                    '문제해결', '언어', '지각', '인지편향', '작업기억', '실행기능',
                    '인지부하', '정보처리', '인지능력'
                ],
                'english': [
                    'cognitive', 'cognition', 'memory', 'attention', 'learning',
                    'thinking', 'decision making', 'problem solving', 'language',
                    'perception', 'executive function', 'working memory',
                    'cognitive load', 'information processing'
                ],
                'weight': 1.0
            },

            'clinical_psychology': {
                'korean': [
                    '우울증', '불안장애', '조현병', '양극성장애', '강박장애',
                    '외상후스트레스장애', '치료', '상담', '심리치료', '인지행동치료',
                    '정신건강', '임상', '진단', '평가', 'DSM', 'ADHD'
                ],
                'english': [
                    'depression', 'anxiety', 'schizophrenia', 'bipolar', 'OCD',
                    'PTSD', 'therapy', 'treatment', 'counseling', 'psychotherapy',
                    'CBT', 'mental health', 'clinical', 'diagnosis', 'DSM',
                    'disorder', 'psychopathology'
                ],
                'weight': 1.0
            },

            'social_psychology': {
                'korean': [
                    '사회', '집단', '대인관계', '사회인지', '태도', '편견',
                    '사회적영향', '동조', '리더십', '집단역학', '사회정체성',
                    '친사회행동', '공격성', '협력'
                ],
                'english': [
                    'social', 'group', 'interpersonal', 'social cognition',
                    'attitude', 'prejudice', 'social influence', 'conformity',
                    'leadership', 'group dynamics', 'social identity',
                    'prosocial', 'aggression', 'cooperation'
                ],
                'weight': 1.0
            },

            'neuroscience': {
                'korean': [
                    '뇌', '신경', '뇌파', '뇌영상', '신경망', '신경전달물질',
                    '전전두엽', '해마', '편도체', '기저핵', '뇌자극', 'tDCS',
                    'fMRI', 'EEG', '신경과학', '인지신경과학'
                ],
                'english': [
                    'brain', 'neural', 'neuron', 'neuroscience', 'neuroimaging',
                    'EEG', 'fMRI', 'TMS', 'tDCS', 'prefrontal', 'hippocampus',
                    'amygdala', 'basal ganglia', 'neurotransmitter',
                    'cognitive neuroscience', 'brain stimulation'
                ],
                'weight': 1.2  # 신경과학은 가중치 높임
            },

            'personality_psychology': {
                'korean': [
                    '성격', '개인차', '특질', '성격검사', '빅파이브', '성격장애',
                    '기질', '성격유형', '개성', '성향', '성격이론'
                ],
                'english': [
                    'personality', 'individual differences', 'trait', 'traits',
                    'big five', 'personality disorder', 'temperament',
                    'personality type', 'personality theory', 'MMPI'
                ],
                'weight': 1.0
            },

            'educational_psychology': {
                'korean': [
                    '교육', '학습', '교수법', '학습자', '교육과정', '평가',
                    '동기', '학습동기', '학업성취', '교실', '교사', '학생',
                    '교육심리학', '학습효과'
                ],
                'english': [
                    'education', 'educational', 'learning', 'teaching',
                    'instruction', 'motivation', 'achievement', 'classroom',
                    'student', 'teacher', 'curriculum', 'assessment',
                    'educational psychology'
                ],
                'weight': 1.0
            },

            'health_psychology': {
                'korean': [
                    '건강', '스트레스', '건강행동', '의료', '질병', '웰빙',
                    '생활습관', '건강증진', '예방', '재활', '건강심리학',
                    '심신의학'
                ],
                'english': [
                    'health', 'stress', 'wellness', 'medical', 'disease',
                    'wellbeing', 'health behavior', 'health promotion',
                    'prevention', 'rehabilitation', 'health psychology',
                    'psychosomatic'
                ],
                'weight': 1.0
            }
        }

        # 연구 방법론 키워드
        self.methodology_keywords = {
            'experimental': [
                '실험', '실험설계', '조작', '통제', '실험군', '대조군',
                'experiment', 'experimental', 'manipulation', 'control',
                'randomized', 'RCT'
            ],
            'survey': [
                '설문', '조사', '설문조사', '척도', '질문지',
                'survey', 'questionnaire', 'scale', 'inventory'
            ],
            'observational': [
                '관찰', '행동관찰', '자연관찰', '참여관찰',
                'observation', 'observational', 'naturalistic'
            ],
            'neuroimaging': [
                '뇌영상', 'fMRI', 'PET', 'EEG', 'MEG', 'neuroimaging'
            ],
            'longitudinal': [
                '종단', '추적', '장기', 'longitudinal', 'follow-up'
            ],
            'cross_sectional': [
                '횡단', '단면', 'cross-sectional'
            ],
            'meta_analysis': [
                '메타분석', '체계적리뷰', 'meta-analysis', 'systematic review'
            ],
            'case_study': [
                '사례연구', '사례', 'case study', 'case report'
            ]
        }

        # 대상 인구 키워드
        self.population_keywords = {
            'children': ['아동', '어린이', '유아', 'children', 'child', 'infant'],
            'adolescents': ['청소년', '중고생', 'adolescent', 'teenager', 'youth'],
            'adults': ['성인', '대학생', 'adult', 'university student'],
            'elderly': ['노인', '고령자', 'elderly', 'older adult', 'senior'],
            'clinical': ['환자', '임상', 'patient', 'clinical', 'disorder'],
            'healthy': ['정상', '건강한', 'healthy', 'normal', 'control']
        }

    def classify_research_domain(self, text: str) -> str:
        """연구 영역 분류 (단일 결과)"""
        classification = self.classify_detailed(text)
        return classification.primary_domain

    def classify_detailed(self, text: str) -> DomainClassification:
        """상세한 도메인 분류"""
        # 텍스트 전처리
        text_lower = text.lower()

        # 각 도메인별 점수 계산
        domain_scores = {}
        all_found_keywords = []

        for domain, keywords_dict in self.domain_keywords.items():
            score = 0
            found_keywords = []

            # 한국어 키워드 검사
            for keyword in keywords_dict['korean']:
                if keyword in text:
                    score += 1
                    found_keywords.append(keyword)

            # 영어 키워드 검사
            for keyword in keywords_dict['english']:
                if keyword.lower() in text_lower:
                    score += 1
                    found_keywords.append(keyword)

            # 가중치 적용
            weighted_score = score * keywords_dict.get('weight', 1.0)
            domain_scores[domain] = weighted_score

            all_found_keywords.extend(found_keywords)

        # 가장 높은 점수의 도메인 선택
        if not domain_scores or max(domain_scores.values()) == 0:
            primary_domain = 'general_psychology'
            confidence = 0.1
            secondary_domains = []
        else:
            sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
            primary_domain = sorted_domains[0][0]

            # 신뢰도 계산
            max_score = sorted_domains[0][1]
            total_keywords = sum(len(kw['korean']) + len(kw['english'])
                               for kw in self.domain_keywords.values())
            confidence = min(1.0, max_score / 10)  # 최대 10개 키워드 기준

            # 상위 2-3개 도메인
            secondary_domains = [(domain, score) for domain, score in sorted_domains[1:3] if score > 0]

        # 연구 방법론 식별
        methodology = self._identify_methodology(text)

        # 대상 인구 식별
        target_population = self._identify_target_population(text)

        return DomainClassification(
            primary_domain=primary_domain,
            confidence=confidence,
            secondary_domains=secondary_domains,
            methodology=methodology,
            target_population=target_population,
            keywords_found=list(set(all_found_keywords))
        )

    def _identify_methodology(self, text: str) -> str:
        """연구 방법론 식별"""
        text_lower = text.lower()
        methodology_scores = {}

        for method, keywords in self.methodology_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    score += 1
            methodology_scores[method] = score

        if not methodology_scores or max(methodology_scores.values()) == 0:
            return 'not_specified'

        return max(methodology_scores.items(), key=lambda x: x[1])[0]

    def _identify_target_population(self, text: str) -> str:
        """대상 인구 식별"""
        text_lower = text.lower()
        population_scores = {}

        for population, keywords in self.population_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    score += 1
            population_scores[population] = score

        if not population_scores or max(population_scores.values()) == 0:
            return 'not_specified'

        return max(population_scores.items(), key=lambda x: x[1])[0]

    def get_domain_statistics(self, texts: List[str]) -> Dict[str, Any]:
        """여러 텍스트에 대한 도메인 분포 통계"""
        domain_counts = {}
        methodology_counts = {}
        population_counts = {}
        confidence_scores = []

        for text in texts:
            classification = self.classify_detailed(text)

            # 도메인 카운트
            domain_counts[classification.primary_domain] = domain_counts.get(
                classification.primary_domain, 0) + 1

            # 방법론 카운트
            methodology_counts[classification.methodology] = methodology_counts.get(
                classification.methodology, 0) + 1

            # 인구 카운트
            population_counts[classification.target_population] = population_counts.get(
                classification.target_population, 0) + 1

            # 신뢰도 점수
            confidence_scores.append(classification.confidence)

        return {
            'domain_distribution': domain_counts,
            'methodology_distribution': methodology_counts,
            'population_distribution': population_counts,
            'average_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'total_texts': len(texts)
        }

    def suggest_related_domains(self, domain: str) -> List[str]:
        """관련 도메인 제안"""
        related_domains = {
            'cognitive_psychology': [
                'neuroscience', 'developmental_psychology', 'educational_psychology'
            ],
            'clinical_psychology': [
                'health_psychology', 'personality_psychology', 'social_psychology'
            ],
            'developmental_psychology': [
                'cognitive_psychology', 'educational_psychology', 'social_psychology'
            ],
            'neuroscience': [
                'cognitive_psychology', 'clinical_psychology', 'health_psychology'
            ],
            'social_psychology': [
                'personality_psychology', 'developmental_psychology'
            ]
        }

        return related_domains.get(domain, [])


# 사용 예시
if __name__ == "__main__":
    classifier = PsychologyDomainClassifier()

    # 테스트 텍스트들
    test_texts = [
        "ADHD 아동의 실행기능과 작업기억 능력을 평가하기 위한 실험 연구",
        "우울증 환자의 인지행동치료 효과성에 관한 메타분석",
        "대학생의 사회적 지지와 스트레스의 관계: 설문조사 연구",
        "fMRI를 이용한 전전두엽 기능 분석과 의사결정 과정",
        "노인의 인지능력 저하와 건강행동의 종단연구"
    ]

    for text in test_texts:
        result = classifier.classify_detailed(text)
        print(f"Text: {text[:50]}...")
        print(f"Domain: {result.primary_domain} (confidence: {result.confidence:.2f})")
        print(f"Methodology: {result.methodology}")
        print(f"Population: {result.target_population}")
        print(f"Keywords: {result.keywords_found[:5]}")
        print("-" * 50)