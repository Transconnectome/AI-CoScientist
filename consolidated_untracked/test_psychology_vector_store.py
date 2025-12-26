"""
Test suite for Psychology Papers Vector Store and Korean NLP Pipeline
TDD 구현: 심리학 논문 처리 및 Korean NLP 통합 테스트
"""

import pytest
import asyncio
import tempfile
import os
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock, patch
import numpy as np


class TestPsychologyVectorStore:
    """Psychology 전용 Vector Store 테스트"""

    def test_psychology_collection_initialization(self):
        """심리학 컬렉션 초기화 테스트"""
        from src.services.psychology.psychology_vector_store import PsychologyVectorStore

        vector_store = PsychologyVectorStore()

        # 심리학 전용 컬렉션 이름들 확인
        assert 'psychology_papers' in vector_store.collections
        assert 'psychology_terms' in vector_store.collections
        assert 'clinical_cases' in vector_store.collections
        assert 'korean_abstracts' in vector_store.collections

    def test_paper_metadata_structure(self):
        """논문 메타데이터 구조 테스트"""
        from src.services.psychology.psychology_vector_store import PaperMetadata

        metadata = PaperMetadata(
            title="tDCS 기반 인지 향상 연구",
            authors=["김철수", "이영희"],
            year=2023,
            journal="Korean Journal of Psychology",
            keywords=["tDCS", "인지", "향상", "뇌자극"],
            abstract_ko="한국어 초록",
            abstract_en="English abstract",
            research_domain="cognitive_neuroscience",
            methodology="experimental",
            subject_population="healthy_adults",
            file_path="data/심리학과/안우영/kim2023_tdcs.pdf"
        )

        assert metadata.title == "tDCS 기반 인지 향상 연구"
        assert len(metadata.keywords) == 4
        assert metadata.research_domain == "cognitive_neuroscience"

    def test_korean_text_preprocessing(self):
        """한국어 텍스트 전처리 테스트"""
        from src.services.psychology.korean_nlp_processor import KoreanTextProcessor

        processor = KoreanTextProcessor()

        # 한국어 텍스트 정규화
        korean_text = "tDCS를 이용한 인지기능 향상에 대한 연구입니다."
        normalized = processor.normalize_korean_text(korean_text)

        assert isinstance(normalized, str)
        assert len(normalized) > 0

        # 형태소 분석
        morphemes = processor.morphological_analysis(korean_text)
        assert isinstance(morphemes, list)
        assert len(morphemes) > 0

    def test_psychology_term_extraction(self):
        """심리학 전문용어 추출 테스트"""
        from src.services.psychology.korean_nlp_processor import PsychologyTermExtractor

        extractor = PsychologyTermExtractor()

        korean_text = """
        본 연구는 ADHD 아동의 실행기능 결함과 작업기억 능력을 측정하였다.
        인지편향과 주의집중 문제를 분석하여 언어발달 지연을 확인했다.
        """

        # 전문용어 추출
        terms = extractor.extract_psychology_terms(korean_text)

        assert "ADHD" in terms
        assert "실행기능" in terms
        assert "작업기억" in terms
        assert "인지편향" in terms

    def test_english_term_mapping(self):
        """한영 심리학 용어 매핑 테스트"""
        from src.services.psychology.korean_nlp_processor import PsychologyTermMapper

        mapper = PsychologyTermMapper()

        # 한국어 -> 영어 매핑
        korean_terms = ["실행기능", "작업기억", "인지편향", "주의집중"]
        english_mappings = mapper.map_to_english(korean_terms)

        assert english_mappings["실행기능"] == "executive function"
        assert english_mappings["작업기억"] == "working memory"
        assert english_mappings["인지편향"] == "cognitive bias"
        assert english_mappings["주의집중"] == "attention"

    @pytest.mark.asyncio
    async def test_pdf_text_extraction(self):
        """PDF 텍스트 추출 테스트"""
        from src.services.psychology.pdf_processor import PsychologyPDFProcessor

        processor = PsychologyPDFProcessor()

        # Mock PDF 처리
        with patch('PyPDF2.PdfReader') as mock_reader:
            mock_page = Mock()
            mock_page.extract_text.return_value = "Sample PDF text with 인지기능 keywords"
            mock_reader.return_value.pages = [mock_page]

            # 임시 PDF 파일 생성
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_path = temp_file.name

            try:
                extracted_text = await processor.extract_text_from_pdf(temp_path)
                assert isinstance(extracted_text, str)
                assert "인지기능" in extracted_text
            finally:
                os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_paper_embedding_generation(self):
        """논문 임베딩 생성 테스트"""
        from src.services.psychology.psychology_vector_store import PsychologyEmbeddingGenerator

        generator = PsychologyEmbeddingGenerator()

        # 테스트 텍스트
        paper_text = """
        본 연구는 ADHD 아동의 실행기능을 분석했다.
        This study analyzes executive function in ADHD children.
        """

        # 임베딩 생성
        embeddings = await generator.generate_paper_embeddings(paper_text)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] > 0  # 벡터 차원 확인
        assert not np.isnan(embeddings).any()  # NaN 값 없는지 확인

    @pytest.mark.asyncio
    async def test_semantic_search(self):
        """의미론적 검색 테스트"""
        from src.services.psychology.psychology_vector_store import PsychologyVectorStore

        vector_store = PsychologyVectorStore()

        # Mock 데이터로 검색 테스트
        query = "ADHD 아동의 실행기능 연구"

        with patch.object(vector_store, 'search_papers') as mock_search:
            mock_search.return_value = [
                {
                    'title': 'ADHD 아동의 실행기능 결함 연구',
                    'similarity_score': 0.92,
                    'research_domain': 'developmental_psychology'
                }
            ]

            results = await vector_store.search_papers(query, limit=5)

            assert len(results) > 0
            assert results[0]['similarity_score'] > 0.8
            assert 'ADHD' in results[0]['title']

    def test_research_domain_classification(self):
        """연구 영역 자동 분류 테스트"""
        from src.services.psychology.domain_classifier import PsychologyDomainClassifier

        classifier = PsychologyDomainClassifier()

        # 다양한 심리학 텍스트 분류
        texts = [
            "ADHD 아동의 주의집중 능력 연구",  # developmental
            "인지편향과 의사결정 과정 분석",    # cognitive
            "우울증 환자의 치료 효과성 연구",   # clinical
            "사회적 인지와 타인 이해 능력",     # social
            "뇌파 측정을 통한 인지 과정 분석"   # neuroscience
        ]

        for text in texts:
            domain = classifier.classify_research_domain(text)
            assert domain in [
                'developmental_psychology',
                'cognitive_psychology',
                'clinical_psychology',
                'social_psychology',
                'neuroscience'
            ]

    @pytest.mark.asyncio
    async def test_paper_batch_processing(self):
        """논문 배치 처리 테스트"""
        from src.services.psychology.psychology_vector_store import PsychologyPaperProcessor

        processor = PsychologyPaperProcessor()

        # Mock 논문 파일들
        paper_files = [
            "data/심리학과/안우영/kim2023_tdcs.pdf",
            "data/심리학과/안우영/lee2024_igd.pdf"
        ]

        with patch.object(processor, 'process_single_paper') as mock_process:
            mock_process.return_value = {
                'title': 'Mock Paper',
                'embeddings': np.random.rand(384),
                'metadata': {'domain': 'cognitive_psychology'}
            }

            # 배치 처리
            results = await processor.process_papers_batch(paper_files)

            assert len(results) == 2
            assert all('embeddings' in result for result in results)

    @pytest.mark.asyncio
    async def test_chromadb_integration(self):
        """ChromaDB 통합 테스트"""
        from src.services.psychology.psychology_vector_store import PsychologyVectorStore

        vector_store = PsychologyVectorStore()

        # Mock ChromaDB client
        with patch.object(vector_store, 'chroma_client') as mock_client:
            mock_collection = Mock()
            mock_client.get_or_create_collection.return_value = mock_collection

            # 문서 추가 테스트
            await vector_store.add_paper(
                title="테스트 논문",
                content="테스트 내용 with 인지기능",
                metadata={'domain': 'cognitive_psychology'}
            )

            # ChromaDB add 메서드 호출 확인
            mock_collection.add.assert_called_once()

    def test_korean_english_bilingual_processing(self):
        """한영 이중언어 처리 테스트"""
        from src.services.psychology.korean_nlp_processor import BilingualProcessor

        processor = BilingualProcessor()

        # 한영 혼재 텍스트
        bilingual_text = """
        본 연구는 ADHD children의 executive function을 분석했다.
        Results show significant deficits in 작업기억 capacity.
        """

        # 언어 분리 처리
        processed = processor.process_bilingual_text(bilingual_text)

        assert 'korean_segments' in processed
        assert 'english_segments' in processed
        assert 'unified_terms' in processed

    @pytest.mark.asyncio
    async def test_query_enhancement(self):
        """쿼리 향상 시스템 테스트"""
        from src.services.psychology.query_enhancer import PsychologyQueryEnhancer

        enhancer = PsychologyQueryEnhancer()

        # 한국어 쿼리 향상
        original_query = "ADHD 아이들의 집중력 문제"
        enhanced_query = await enhancer.enhance_query(original_query)

        assert len(enhanced_query) > len(original_query)
        assert "attention deficit" in enhanced_query.lower()
        assert "executive function" in enhanced_query.lower()


class TestKoreanNLPPipeline:
    """Korean NLP Pipeline 통합 테스트"""

    def test_korean_tokenization(self):
        """한국어 토큰화 테스트"""
        from src.services.psychology.korean_nlp_processor import KoreanTokenizer

        tokenizer = KoreanTokenizer()

        text = "ADHD 아동의 실행기능과 작업기억 능력을 평가했다."
        tokens = tokenizer.tokenize(text)

        assert isinstance(tokens, list)
        assert "ADHD" in tokens
        assert "아동" in tokens
        assert "실행기능" in tokens

    def test_named_entity_recognition(self):
        """개체명 인식 테스트"""
        from src.services.psychology.korean_nlp_processor import PsychologyNER

        ner = PsychologyNER()

        text = "서울대학교 심리학과에서 진행한 ADHD 연구"
        entities = ner.extract_entities(text)

        assert any(entity['text'] == 'ADHD' for entity in entities)
        assert any(entity['label'] == 'DISORDER' for entity in entities)

    def test_sentiment_analysis(self):
        """감정 분석 테스트"""
        from src.services.psychology.korean_nlp_processor import KoreanSentimentAnalyzer

        analyzer = KoreanSentimentAnalyzer()

        positive_text = "이 치료법은 매우 효과적이고 만족스럽다."
        negative_text = "치료 효과가 부족하고 개선이 필요하다."

        pos_sentiment = analyzer.analyze_sentiment(positive_text)
        neg_sentiment = analyzer.analyze_sentiment(negative_text)

        assert pos_sentiment['label'] == 'positive'
        assert neg_sentiment['label'] == 'negative'

    @pytest.mark.asyncio
    async def test_full_pipeline_integration(self):
        """전체 파이프라인 통합 테스트"""
        from src.services.psychology.korean_nlp_processor import KoreanNLPPipeline

        pipeline = KoreanNLPPipeline()

        text = """
        본 연구는 ADHD 아동 30명을 대상으로 실행기능 훈련의 효과를 검증했다.
        결과는 매우 긍정적이며 작업기억 능력이 유의미하게 향상되었다.
        """

        # 전체 분석
        analysis_result = await pipeline.analyze_text(text)

        assert 'tokens' in analysis_result
        assert 'entities' in analysis_result
        assert 'psychology_terms' in analysis_result
        assert 'sentiment' in analysis_result
        assert 'english_mappings' in analysis_result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])