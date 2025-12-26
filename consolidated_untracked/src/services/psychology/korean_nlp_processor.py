"""
Korean NLP Processor for Psychology Domain
심리학 특화 한국어 자연어처리 파이프라인

Features:
1. 한국어 형태소 분석 및 토큰화
2. 심리학 전문용어 추출 및 매핑
3. 한영 이중언어 처리
4. 개체명 인식 (심리학 특화)
5. 감정 분석 및 임상 키워드 감지
"""

import re
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

# Korean NLP Dependencies
try:
    import konlpy
    from konlpy.tag import Okt, Komoran
    KOREAN_NLP_AVAILABLE = True
except ImportError:
    KOREAN_NLP_AVAILABLE = False
    logging.warning("KoNLPy not available. Korean NLP features will be limited.")

import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

logger = logging.getLogger(__name__)


@dataclass
class PsychologyTerm:
    """심리학 전문용어 구조"""
    korean: str
    english: str
    category: str  # disorder, cognitive, clinical, research
    confidence: float = 1.0


@dataclass
class AnalysisResult:
    """NLP 분석 결과"""
    tokens: List[str]
    entities: List[Dict[str, Any]]
    psychology_terms: List[PsychologyTerm]
    sentiment: Dict[str, Any]
    english_mappings: Dict[str, str]
    confidence_scores: Dict[str, float]


class KoreanTextProcessor:
    """한국어 텍스트 전처리"""

    def __init__(self):
        self.special_chars_pattern = re.compile(r'[^가-힣a-zA-Z0-9\s\-_]')

    def normalize_korean_text(self, text: str) -> str:
        """한국어 텍스트 정규화"""
        # 공백 정리
        text = re.sub(r'\s+', ' ', text.strip())

        # 특수문자 제거 (일부 보존)
        # text = self.special_chars_pattern.sub('', text)

        # 영어 대소문자 정규화
        text = re.sub(r'\b([A-Z]+)\b', lambda m: m.group(1).upper(), text)

        return text

    def morphological_analysis(self, text: str) -> List[Tuple[str, str]]:
        """형태소 분석"""
        if not KOREAN_NLP_AVAILABLE:
            # Fallback: 간단한 공백 기반 토큰화
            tokens = text.split()
            return [(token, 'UNK') for token in tokens]

        try:
            okt = Okt()
            morphemes = okt.pos(text, norm=True, stem=True)
            return morphemes
        except Exception as e:
            logger.warning(f"Morphological analysis failed: {e}")
            tokens = text.split()
            return [(token, 'UNK') for token in tokens]


class PsychologyTermExtractor:
    """심리학 전문용어 추출기"""

    def __init__(self):
        # 심리학 전문용어 사전 구축
        self.psychology_terms = {
            # 장애/질환
            'ADHD': 'disorder',
            '주의력결핍과잉행동장애': 'disorder',
            '자폐스펙트럼장애': 'disorder',
            'ASD': 'disorder',
            '우울증': 'disorder',
            '불안장애': 'disorder',
            '조현병': 'disorder',
            '양극성장애': 'disorder',

            # 인지기능
            '실행기능': 'cognitive',
            '작업기억': 'cognitive',
            '인지편향': 'cognitive',
            '주의집중': 'cognitive',
            '언어발달': 'cognitive',
            '인지능력': 'cognitive',
            '기억력': 'cognitive',
            '학습능력': 'cognitive',
            '문제해결능력': 'cognitive',

            # 임상/치료
            '인지행동치료': 'clinical',
            'CBT': 'clinical',
            '약물치료': 'clinical',
            '심리치료': 'clinical',
            '상담': 'clinical',
            '진단': 'clinical',
            '평가': 'clinical',
            '개입': 'clinical',

            # 연구방법
            '뇌파': 'research',
            'EEG': 'research',
            'fMRI': 'research',
            'tDCS': 'research',
            '뇌자극': 'research',
            '행동실험': 'research',
            '설문조사': 'research',
            '임상시험': 'research'
        }

        # 정규표현식 패턴들
        self.disorder_pattern = re.compile(
            r'(ADHD|ASD|우울증|불안장애|조현병|자폐|양극성|강박|공포|외상|스트레스장애)'
        )
        self.cognitive_pattern = re.compile(
            r'(실행기능|작업기억|인지편향|주의집중|언어발달|기억|학습|인지)'
        )

    def extract_psychology_terms(self, text: str) -> List[str]:
        """심리학 전문용어 추출"""
        found_terms = []

        # 사전 기반 추출
        for term in self.psychology_terms.keys():
            if term in text:
                found_terms.append(term)

        # 패턴 기반 추출
        disorder_matches = self.disorder_pattern.findall(text)
        cognitive_matches = self.cognitive_pattern.findall(text)

        found_terms.extend(disorder_matches)
        found_terms.extend(cognitive_matches)

        return list(set(found_terms))  # 중복 제거

    def categorize_terms(self, terms: List[str]) -> List[PsychologyTerm]:
        """용어 카테고리 분류"""
        categorized_terms = []

        for term in terms:
            category = self.psychology_terms.get(term, 'general')
            psychology_term = PsychologyTerm(
                korean=term,
                english="",  # 매핑에서 채움
                category=category,
                confidence=1.0
            )
            categorized_terms.append(psychology_term)

        return categorized_terms


class PsychologyTermMapper:
    """한영 심리학 용어 매핑"""

    def __init__(self):
        self.korean_to_english = {
            # 장애
            'ADHD': 'attention deficit hyperactivity disorder',
            '주의력결핍과잉행동장애': 'attention deficit hyperactivity disorder',
            'ASD': 'autism spectrum disorder',
            '자폐스펙트럼장애': 'autism spectrum disorder',
            '우울증': 'depression',
            '불안장애': 'anxiety disorder',
            '조현병': 'schizophrenia',
            '양극성장애': 'bipolar disorder',

            # 인지기능
            '실행기능': 'executive function',
            '작업기억': 'working memory',
            '인지편향': 'cognitive bias',
            '주의집중': 'attention',
            '언어발달': 'language development',
            '인지능력': 'cognitive ability',
            '기억력': 'memory',
            '학습능력': 'learning ability',
            '문제해결능력': 'problem solving',

            # 임상
            '인지행동치료': 'cognitive behavioral therapy',
            '심리치료': 'psychotherapy',
            '약물치료': 'pharmacotherapy',
            '상담': 'counseling',
            '진단': 'diagnosis',
            '평가': 'assessment',
            '개입': 'intervention',

            # 연구
            '뇌파': 'electroencephalography',
            '뇌자극': 'brain stimulation',
            '행동실험': 'behavioral experiment',
            '설문조사': 'survey',
            '임상시험': 'clinical trial'
        }

    def map_to_english(self, korean_terms: List[str]) -> Dict[str, str]:
        """한국어 용어를 영어로 매핑"""
        mappings = {}

        for term in korean_terms:
            english_term = self.korean_to_english.get(term, term)
            mappings[term] = english_term

        return mappings

    def get_synonym_terms(self, term: str) -> List[str]:
        """동의어/유의어 확장"""
        synonym_dict = {
            'ADHD': ['attention deficit', 'hyperactivity', 'impulsivity'],
            '실행기능': ['executive control', 'cognitive control', 'attention control'],
            '작업기억': ['short term memory', 'cognitive load', 'memory span'],
            '인지편향': ['cognitive distortion', 'thinking error', 'bias']
        }

        return synonym_dict.get(term, [])


class KoreanTokenizer:
    """한국어 토큰화"""

    def __init__(self):
        if KOREAN_NLP_AVAILABLE:
            try:
                self.tokenizer = Okt()
            except Exception as e:
                logger.warning(f"KoNLPy initialization failed: {e}. Using fallback tokenizer.")
                self.tokenizer = None
        else:
            self.tokenizer = None

    def tokenize(self, text: str) -> List[str]:
        """텍스트 토큰화"""
        if self.tokenizer is None:
            # Fallback tokenization
            return text.split()

        try:
            # 명사, 형용사, 동사, 영어 등 주요 품사만 추출
            tokens = self.tokenizer.morphs(text, norm=True, stem=True)
            return [token for token in tokens if len(token) > 1]
        except Exception as e:
            logger.warning(f"Tokenization failed: {e}")
            return text.split()


class PsychologyNER:
    """심리학 도메인 개체명 인식"""

    def __init__(self):
        # 심리학 특화 개체명 패턴들
        self.entity_patterns = {
            'DISORDER': re.compile(r'(ADHD|ASD|우울증|불안장애|조현병|자폐|양극성장애)'),
            'ASSESSMENT': re.compile(r'(WAIS|WISC|K-ABC|TMT|Stroop|Wisconsin)'),
            'BRAIN_REGION': re.compile(r'(전전두엽|해마|편도체|기저핵|소뇌|측두엽)'),
            'NEUROTRANSMITTER': re.compile(r'(도파민|세로토닌|노르에피네프린|GABA|아세틸콜린)'),
            'MEDICATION': re.compile(r'(메틸페니데이트|아토목세틴|리스페리돈|할로페리돌)')
        }

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """개체명 추출"""
        entities = []

        for entity_type, pattern in self.entity_patterns.items():
            matches = pattern.finditer(text)
            for match in matches:
                entity = {
                    'text': match.group(),
                    'label': entity_type,
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.9
                }
                entities.append(entity)

        return entities


class KoreanSentimentAnalyzer:
    """한국어 감정 분석"""

    def __init__(self):
        # 감정 키워드 사전
        self.positive_words = {
            '효과적', '성공적', '향상', '개선', '긍정적', '유의미', '만족',
            '도움', '회복', '치료', '완화', '감소', '증가'
        }

        self.negative_words = {
            '악화', '부족', '문제', '어려움', '심각', '부정적', '실패',
            '저하', '결함', '손상', '장애', '증상', '고통'
        }

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """감정 분석"""
        positive_count = sum(1 for word in self.positive_words if word in text)
        negative_count = sum(1 for word in self.negative_words if word in text)

        if positive_count > negative_count:
            label = 'positive'
            confidence = positive_count / (positive_count + negative_count + 1)
        elif negative_count > positive_count:
            label = 'negative'
            confidence = negative_count / (positive_count + negative_count + 1)
        else:
            label = 'neutral'
            confidence = 0.5

        return {
            'label': label,
            'confidence': confidence,
            'positive_count': positive_count,
            'negative_count': negative_count
        }


class BilingualProcessor:
    """한영 이중언어 처리"""

    def __init__(self):
        # 한글 패턴
        self.korean_pattern = re.compile(r'[가-힣]+')
        # 영어 패턴
        self.english_pattern = re.compile(r'[a-zA-Z]+')

    def process_bilingual_text(self, text: str) -> Dict[str, Any]:
        """이중언어 텍스트 처리"""
        # 한국어 세그먼트 추출
        korean_segments = self.korean_pattern.findall(text)

        # 영어 세그먼트 추출
        english_segments = self.english_pattern.findall(text)

        # 통합 용어 추출
        unified_terms = self._extract_unified_terms(text)

        return {
            'korean_segments': korean_segments,
            'english_segments': english_segments,
            'unified_terms': unified_terms,
            'language_ratio': len(korean_segments) / (len(korean_segments) + len(english_segments) + 1)
        }

    def _extract_unified_terms(self, text: str) -> List[str]:
        """통합 용어 추출"""
        # 전문용어 패턴 (한영 혼재)
        term_patterns = [
            r'ADHD\s*아동',
            r'executive\s*function',
            r'working\s*memory',
            r'cognitive\s*bias'
        ]

        unified_terms = []
        for pattern in term_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            unified_terms.extend(matches)

        return unified_terms


class KoreanNLPPipeline:
    """한국어 NLP 통합 파이프라인"""

    def __init__(self):
        self.text_processor = KoreanTextProcessor()
        self.term_extractor = PsychologyTermExtractor()
        self.term_mapper = PsychologyTermMapper()
        self.tokenizer = KoreanTokenizer()
        self.ner = PsychologyNER()
        self.sentiment_analyzer = KoreanSentimentAnalyzer()
        self.bilingual_processor = BilingualProcessor()

    async def analyze_text(self, text: str) -> AnalysisResult:
        """전체 텍스트 분석"""
        # 1. 텍스트 전처리
        normalized_text = self.text_processor.normalize_korean_text(text)

        # 2. 토큰화
        tokens = self.tokenizer.tokenize(normalized_text)

        # 3. 심리학 전문용어 추출
        psychology_terms_raw = self.term_extractor.extract_psychology_terms(text)
        psychology_terms = self.term_extractor.categorize_terms(psychology_terms_raw)

        # 4. 영어 매핑
        english_mappings = self.term_mapper.map_to_english(psychology_terms_raw)

        # 5. 개체명 인식
        entities = self.ner.extract_entities(text)

        # 6. 감정 분석
        sentiment = self.sentiment_analyzer.analyze_sentiment(text)

        # 7. 신뢰도 점수 계산
        confidence_scores = self._calculate_confidence_scores(
            tokens, psychology_terms, entities
        )

        return AnalysisResult(
            tokens=tokens,
            entities=entities,
            psychology_terms=psychology_terms,
            sentiment=sentiment,
            english_mappings=english_mappings,
            confidence_scores=confidence_scores
        )

    def _calculate_confidence_scores(self, tokens, psychology_terms, entities) -> Dict[str, float]:
        """신뢰도 점수 계산"""
        total_tokens = len(tokens)
        psych_term_ratio = len(psychology_terms) / (total_tokens + 1)
        entity_ratio = len(entities) / (total_tokens + 1)

        overall_confidence = min(1.0, (psych_term_ratio + entity_ratio) * 2)

        return {
            'overall': overall_confidence,
            'term_extraction': psych_term_ratio,
            'entity_recognition': entity_ratio,
            'domain_relevance': psych_term_ratio * 2  # 심리학 관련성
        }


# 예시 사용법
if __name__ == "__main__":
    async def main():
        pipeline = KoreanNLPPipeline()

        sample_text = """
        본 연구는 ADHD 아동 30명을 대상으로 실행기능 훈련의 효과를 검증했다.
        Cognitive behavioral therapy와 약물치료를 병행한 결과,
        작업기억 능력이 유의미하게 향상되었다.
        """

        result = await pipeline.analyze_text(sample_text)
        print(f"Analysis Result: {result}")

    # asyncio.run(main())