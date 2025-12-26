"""
Psychology Query Enhancer
심리학 쿼리 향상 및 확장 시스템

Features:
1. 한국어 쿼리를 영어로 확장
2. 유의어 및 관련 용어 추가
3. 컨텍스트 기반 쿼리 보완
4. 심리학 도메인 특화 확장
5. 검색 정확도 향상
"""

import asyncio
import re
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
import logging

from .korean_nlp_processor import PsychologyTermMapper, PsychologyTermExtractor
from .domain_classifier import PsychologyDomainClassifier

logger = logging.getLogger(__name__)


@dataclass
class EnhancedQuery:
    """향상된 쿼리 결과"""
    original_query: str
    enhanced_query: str
    added_terms: List[str]
    synonyms: Dict[str, List[str]]
    domain: str
    confidence: float


class PsychologyQueryEnhancer:
    """심리학 쿼리 향상기"""

    def __init__(self):
        self.term_mapper = PsychologyTermMapper()
        self.term_extractor = PsychologyTermExtractor()
        self.domain_classifier = PsychologyDomainClassifier()

        # 심리학 특화 동의어/유의어 사전
        self.synonym_dict = {
            'ADHD': [
                'attention deficit hyperactivity disorder',
                'attention deficit',
                'hyperactivity',
                'impulsivity',
                '주의력결핍과잉행동장애',
                '주의력 결핍',
                '과잉행동'
            ],

            '실행기능': [
                'executive function',
                'executive control',
                'cognitive control',
                'attention control',
                'working memory control',
                '인지조절',
                '주의조절',
                '실행통제'
            ],

            '작업기억': [
                'working memory',
                'short term memory',
                'memory span',
                'cognitive load',
                'memory capacity',
                '단기기억',
                '기억용량',
                '인지부하'
            ],

            '인지편향': [
                'cognitive bias',
                'cognitive distortion',
                'thinking error',
                'bias',
                'cognitive error',
                '인지왜곡',
                '사고오류',
                '편향'
            ],

            '우울증': [
                'depression',
                'major depression',
                'depressive disorder',
                'mood disorder',
                'melancholia',
                '우울장애',
                '기분장애',
                '우울감'
            ],

            '불안장애': [
                'anxiety disorder',
                'anxiety',
                'generalized anxiety',
                'panic disorder',
                'phobia',
                '불안',
                '공포',
                '공황장애'
            ],

            '인지행동치료': [
                'cognitive behavioral therapy',
                'CBT',
                'cognitive therapy',
                'behavioral therapy',
                'cognitive behavioral intervention',
                '인지치료',
                '행동치료',
                '인지행동개입'
            ],

            '뇌파': [
                'EEG',
                'electroencephalography',
                'brain waves',
                'neural oscillations',
                'brainwave',
                '뇌전도',
                '뇌신호'
            ],

            '뇌자극': [
                'brain stimulation',
                'tDCS',
                'TMS',
                'transcranial stimulation',
                'neuromodulation',
                '경두개자극',
                '신경조절'
            ]
        }

        # 도메인별 확장 키워드
        self.domain_expansions = {
            'cognitive_psychology': [
                'cognitive processes', 'information processing', 'mental processes',
                '인지과정', '정보처리', '정신과정'
            ],
            'clinical_psychology': [
                'psychopathology', 'mental health', 'psychological disorder',
                '정신병리', '정신건강', '심리장애'
            ],
            'developmental_psychology': [
                'child development', 'developmental milestones', 'maturation',
                '아동발달', '발달과정', '성숙'
            ],
            'neuroscience': [
                'neural mechanisms', 'brain function', 'neural networks',
                '신경기전', '뇌기능', '신경망'
            ],
            'social_psychology': [
                'social behavior', 'interpersonal relationships', 'group dynamics',
                '사회행동', '대인관계', '집단역학'
            ]
        }

        # 연구 방법론 키워드 확장
        self.methodology_expansions = {
            'experimental': ['randomized controlled trial', 'RCT', 'experiment design'],
            'neuroimaging': ['brain imaging', 'functional MRI', 'neuroimaging study'],
            'survey': ['questionnaire study', 'survey research', 'self-report'],
            'longitudinal': ['follow-up study', 'prospective study', 'long-term'],
            'meta_analysis': ['systematic review', 'meta-analytic study']
        }

    async def enhance_query(self, query: str) -> str:
        """쿼리 향상 (간단한 문자열 반환)"""
        enhanced = await self.enhance_query_detailed(query)
        return enhanced.enhanced_query

    async def enhance_query_detailed(self, query: str) -> EnhancedQuery:
        """상세한 쿼리 향상"""
        # 1. 원본 쿼리 분석
        psychology_terms = self.term_extractor.extract_psychology_terms(query)
        domain = self.domain_classifier.classify_research_domain(query)

        # 2. 한영 매핑
        english_mappings = self.term_mapper.map_to_english(psychology_terms)

        # 3. 동의어 확장
        synonyms_dict = {}
        expanded_terms = []

        for term in psychology_terms:
            # 직접 동의어
            direct_synonyms = self.synonym_dict.get(term, [])
            if direct_synonyms:
                synonyms_dict[term] = direct_synonyms
                expanded_terms.extend(direct_synonyms[:3])  # 상위 3개만

            # 영어 매핑된 용어의 동의어
            english_term = english_mappings.get(term)
            if english_term and english_term in self.synonym_dict:
                english_synonyms = self.synonym_dict[english_term]
                synonyms_dict[english_term] = english_synonyms
                expanded_terms.extend(english_synonyms[:3])

        # 4. 도메인별 확장
        domain_terms = self.domain_expansions.get(domain, [])
        expanded_terms.extend(domain_terms[:2])

        # 5. 향상된 쿼리 생성
        enhanced_query = self._build_enhanced_query(
            query, english_mappings, expanded_terms, psychology_terms
        )

        # 6. 신뢰도 계산
        confidence = self._calculate_enhancement_confidence(
            psychology_terms, expanded_terms, domain
        )

        return EnhancedQuery(
            original_query=query,
            enhanced_query=enhanced_query,
            added_terms=list(set(expanded_terms)),
            synonyms=synonyms_dict,
            domain=domain,
            confidence=confidence
        )

    def _build_enhanced_query(self, original_query: str, english_mappings: Dict[str, str],
                            expanded_terms: List[str], psychology_terms: List[str]) -> str:
        """향상된 쿼리 문자열 구성"""
        # 원본 쿼리로 시작
        enhanced_parts = [original_query]

        # 영어 매핑 추가
        for korean_term, english_term in english_mappings.items():
            if english_term != korean_term:  # 실제로 매핑된 경우만
                enhanced_parts.append(english_term)

        # 중요한 확장 용어들 추가
        important_expansions = []
        for term in expanded_terms:
            if len(term) > 3 and term.lower() not in original_query.lower():
                important_expansions.append(term)

        # 중복 제거하고 상위 5개만 추가
        unique_expansions = list(dict.fromkeys(important_expansions))[:5]
        enhanced_parts.extend(unique_expansions)

        # 최종 쿼리 생성
        enhanced_query = " ".join(enhanced_parts)

        return enhanced_query

    def _calculate_enhancement_confidence(self, psychology_terms: List[str],
                                       expanded_terms: List[str], domain: str) -> float:
        """향상 신뢰도 계산"""
        # 기본 점수
        base_score = 0.5

        # 심리학 용어가 있으면 점수 상승
        if psychology_terms:
            base_score += 0.2 * min(len(psychology_terms), 3)

        # 확장 용어가 있으면 점수 상승
        if expanded_terms:
            base_score += 0.1 * min(len(expanded_terms), 5)

        # 특정 도메인이 식별되면 점수 상승
        if domain != 'general_psychology':
            base_score += 0.2

        return min(1.0, base_score)

    async def suggest_related_queries(self, original_query: str) -> List[str]:
        """관련 쿼리 제안"""
        # 도메인 분류
        domain = self.domain_classifier.classify_research_domain(original_query)

        # 심리학 용어 추출
        psychology_terms = self.term_extractor.extract_psychology_terms(original_query)

        suggestions = []

        # 1. 동의어 기반 변형
        for term in psychology_terms:
            synonyms = self.synonym_dict.get(term, [])
            if synonyms:
                # 원본에서 용어만 바꾼 버전
                for synonym in synonyms[:2]:
                    suggested_query = original_query.replace(term, synonym)
                    if suggested_query != original_query:
                        suggestions.append(suggested_query)

        # 2. 도메인 확장 기반
        domain_terms = self.domain_expansions.get(domain, [])
        for domain_term in domain_terms[:2]:
            suggestions.append(f"{original_query} {domain_term}")

        # 3. 연구 방법론 기반
        if '연구' in original_query or 'study' in original_query.lower():
            methodology_suggestions = [
                f"{original_query} experimental study",
                f"{original_query} longitudinal research",
                f"{original_query} meta-analysis"
            ]
            suggestions.extend(methodology_suggestions)

        # 중복 제거 및 정리
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion not in unique_suggestions and len(suggestion) < 200:
                unique_suggestions.append(suggestion)

        return unique_suggestions[:5]  # 최대 5개

    def expand_query_with_context(self, query: str, context: str = "") -> str:
        """컨텍스트를 고려한 쿼리 확장"""
        if not context:
            return query

        # 컨텍스트에서 추가 키워드 추출
        context_terms = self.term_extractor.extract_psychology_terms(context)

        # 쿼리와 관련 있는 컨텍스트 용어들 선별
        relevant_terms = []
        query_lower = query.lower()

        for term in context_terms:
            # 유사한 도메인이거나 관련성이 있는 용어
            if any(word in query_lower for word in term.lower().split()):
                relevant_terms.append(term)

        # 상위 3개 관련 용어만 추가
        if relevant_terms:
            expanded_query = f"{query} {' '.join(relevant_terms[:3])}"
            return expanded_query

        return query

    def get_search_filters_from_query(self, query: str) -> Dict[str, Any]:
        """쿼리에서 검색 필터 추출"""
        filters = {}

        # 연도 필터
        year_matches = re.findall(r'20\d{2}', query)
        if year_matches:
            filters['year_range'] = [int(year_matches[0]) - 2, int(year_matches[0]) + 2]

        # 도메인 필터
        domain = self.domain_classifier.classify_research_domain(query)
        if domain != 'general_psychology':
            filters['research_domain'] = domain

        # 방법론 필터
        if '실험' in query or 'experiment' in query.lower():
            filters['methodology'] = 'experimental'
        elif '설문' in query or 'survey' in query.lower():
            filters['methodology'] = 'survey'
        elif 'fMRI' in query or '뇌영상' in query:
            filters['methodology'] = 'neuroimaging'

        # 대상 인구 필터
        if '아동' in query or 'child' in query.lower():
            filters['target_population'] = 'children'
        elif '청소년' in query or 'adolescent' in query.lower():
            filters['target_population'] = 'adolescents'
        elif '성인' in query or 'adult' in query.lower():
            filters['target_population'] = 'adults'

        return filters


# 사용 예시
if __name__ == "__main__":
    async def main():
        enhancer = PsychologyQueryEnhancer()

        # 테스트 쿼리들
        test_queries = [
            "ADHD 아동의 실행기능 연구",
            "우울증 치료 효과",
            "작업기억과 학습능력",
            "tDCS를 이용한 인지 향상",
            "청소년 불안장애 상담"
        ]

        for query in test_queries:
            enhanced = await enhancer.enhance_query_detailed(query)
            print(f"Original: {enhanced.original_query}")
            print(f"Enhanced: {enhanced.enhanced_query}")
            print(f"Domain: {enhanced.domain}")
            print(f"Confidence: {enhanced.confidence:.2f}")
            print(f"Added terms: {enhanced.added_terms[:5]}")
            print("-" * 50)

    # asyncio.run(main())