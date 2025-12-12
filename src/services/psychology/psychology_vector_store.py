"""
Psychology Vector Store with ChromaDB Integration
심리학 논문 전용 벡터 스토어 및 검색 시스템

Features:
1. 심리학 논문 전용 컬렉션 관리
2. 한국어 지원 임베딩 생성
3. 의미론적 검색 및 필터링
4. 메타데이터 기반 분류
5. 실시간 논문 추가/업데이트
"""

import asyncio
import logging
import re
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import os
from pathlib import Path

import numpy as np
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logging.warning("ChromaDB not available")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("Sentence transformers not available")

import torch

# Psychology specific modules
from .korean_nlp_processor import KoreanNLPPipeline, AnalysisResult
from .pdf_processor import PsychologyPDFProcessor
from .domain_classifier import PsychologyDomainClassifier
from .query_enhancer import PsychologyQueryEnhancer

# AI-CoScientist integration
from src.core.config import get_settings
from src.monitoring.rag_metrics import RAGMetrics

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class PaperMetadata:
    """논문 메타데이터 구조"""
    title: str
    authors: List[str]
    year: int
    journal: str
    keywords: List[str]
    abstract_ko: str = ""
    abstract_en: str = ""
    research_domain: str = "general_psychology"
    methodology: str = "theoretical"
    subject_population: str = "not_specified"
    file_path: str = ""
    processing_date: str = ""
    language: str = "mixed"  # ko, en, mixed

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return asdict(self)


@dataclass
class SearchResult:
    """검색 결과 구조"""
    title: str
    content_snippet: str
    similarity_score: float
    metadata: PaperMetadata
    highlights: List[str] = None
    reasoning: str = ""


class PsychologyEmbeddingGenerator:
    """심리학 특화 임베딩 생성기"""

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        다국어 지원 모델로 한국어와 영어 동시 처리
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("Sentence transformers not available, using mock embeddings")
            self.model = None
            self.dimension = 384  # Default dimension
            return

        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Embedding model loaded: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load {model_name}, using fallback")
            # Fallback to a simpler model
            try:
                self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            except Exception as e2:
                logger.error(f"All embedding models failed: {e2}")
                self.model = None

        if self.model:
            self.dimension = self.model.get_sentence_embedding_dimension()
        else:
            self.dimension = 384

    async def generate_paper_embeddings(self, text: str, metadata: Optional[PaperMetadata] = None) -> np.ndarray:
        """논문 텍스트의 임베딩 생성"""
        try:
            # 텍스트 전처리
            processed_text = await self._preprocess_text(text, metadata)

            # 모델이 없는 경우 mock 임베딩 생성
            if self.model is None:
                # 텍스트 기반 간단한 해시 임베딩 (테스트용)
                hash_value = hash(processed_text) % (2**31)
                mock_embedding = np.random.RandomState(hash_value).randn(self.dimension)
                return mock_embedding.astype(np.float32)

            # 임베딩 생성
            embeddings = self.model.encode(processed_text, normalize_embeddings=True)

            return embeddings.astype(np.float32)

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            # 기본 임베딩 반환
            return np.zeros(self.dimension, dtype=np.float32)

    async def _preprocess_text(self, text: str, metadata: Optional[PaperMetadata] = None) -> str:
        """텍스트 전처리 및 메타데이터 통합"""
        # 메타데이터 정보 추가
        if metadata:
            enhanced_text = f"""
            Title: {metadata.title}
            Keywords: {' '.join(metadata.keywords)}
            Domain: {metadata.research_domain}
            Content: {text[:2000]}  # 처음 2000자만 사용
            """
        else:
            enhanced_text = text[:2000]

        return enhanced_text.strip()

    def generate_query_embedding(self, query: str) -> np.ndarray:
        """쿼리 임베딩 생성"""
        try:
            # 모델이 없는 경우 mock 임베딩 생성
            if self.model is None:
                hash_value = hash(query) % (2**31)
                mock_embedding = np.random.RandomState(hash_value).randn(self.dimension)
                return mock_embedding.astype(np.float32)

            embeddings = self.model.encode(query, normalize_embeddings=True)
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            return np.zeros(self.dimension, dtype=np.float32)


class PsychologyVectorStore:
    """심리학 전용 Vector Store"""

    def __init__(self, chroma_host: str = None, chroma_port: int = None):
        """ChromaDB 연결 초기화"""
        self.chroma_host = chroma_host or settings.chromadb_host
        self.chroma_port = chroma_port or settings.chromadb_port

        # ChromaDB 클라이언트 초기화
        if not CHROMADB_AVAILABLE:
            logger.warning("ChromaDB not available, using mock vector store")
            self.chroma_client = None
        else:
            try:
                self.chroma_client = chromadb.HttpClient(
                    host=self.chroma_host,
                    port=self.chroma_port
                )
                logger.info(f"Connected to ChromaDB at {self.chroma_host}:{self.chroma_port}")
            except Exception as e:
                logger.warning(f"ChromaDB connection failed: {e}. Using in-memory client.")
                try:
                    self.chroma_client = chromadb.Client()
                except Exception as e2:
                    logger.error(f"In-memory ChromaDB also failed: {e2}")
                    self.chroma_client = None

        # 컬렉션들 초기화
        self.collections = self._initialize_collections()

        # 다른 컴포넌트들
        self.embedding_generator = PsychologyEmbeddingGenerator()
        self.nlp_pipeline = KoreanNLPPipeline()
        self.domain_classifier = PsychologyDomainClassifier()

    def _initialize_collections(self) -> Dict[str, Any]:
        """심리학 관련 컬렉션들 초기화"""
        collections = {}

        if self.chroma_client is None:
            logger.warning("ChromaDB client not available, using mock collections")
            return {
                'psychology_papers': None,
                'psychology_terms': None,
                'clinical_cases': None,
                'korean_abstracts': None
            }

        collection_configs = {
            'psychology_papers': {
                'metadata': {"hnsw:space": "cosine"},
                'description': "심리학 논문 본문 및 메타데이터"
            },
            'psychology_terms': {
                'metadata': {"hnsw:space": "cosine"},
                'description': "심리학 전문용어 및 정의"
            },
            'clinical_cases': {
                'metadata': {"hnsw:space": "cosine"},
                'description': "임상 사례 및 치료 사례"
            },
            'korean_abstracts': {
                'metadata': {"hnsw:space": "cosine"},
                'description': "한국어 초록 전용 컬렉션"
            }
        }

        for name, config in collection_configs.items():
            try:
                collection = self.chroma_client.get_or_create_collection(
                    name=name,
                    metadata=config['metadata']
                )
                collections[name] = collection
                logger.info(f"Collection '{name}' initialized")
            except Exception as e:
                logger.error(f"Failed to initialize collection '{name}': {e}")
                collections[name] = None

        return collections

    async def add_paper(self, title: str, content: str, metadata: PaperMetadata) -> bool:
        """논문을 벡터 스토어에 추가"""
        try:
            # 1. NLP 분석
            analysis_result = await self.nlp_pipeline.analyze_text(content)

            # 2. 도메인 분류
            domain = self.domain_classifier.classify_research_domain(content)
            metadata.research_domain = domain

            # 3. 임베딩 생성
            embeddings = await self.embedding_generator.generate_paper_embeddings(
                content, metadata
            )

            # 4. 문서 ID 생성
            doc_id = self._generate_document_id(title, metadata)

            # 5. ChromaDB에 저장
            collection = self.collections['psychology_papers']

            # 메타데이터에 분석 결과 추가
            enhanced_metadata = metadata.to_dict()
            enhanced_metadata.update({
                'psychology_terms': [term.korean for term in analysis_result.psychology_terms],
                'entities': [entity['text'] for entity in analysis_result.entities],
                'sentiment_label': analysis_result.sentiment['label'],
                'confidence_scores': analysis_result.confidence_scores,
                'processing_date': datetime.now().isoformat()
            })

            collection.add(
                documents=[content[:1000]],  # 처음 1000자를 문서로 저장
                embeddings=[embeddings.tolist()],
                metadatas=[enhanced_metadata],
                ids=[doc_id]
            )

            logger.info(f"Paper added successfully: {title}")
            return True

        except Exception as e:
            logger.error(f"Failed to add paper: {e}")
            return False

    async def search_papers(self, query: str, limit: int = 10, filters: Dict[str, Any] = None) -> List[SearchResult]:
        """논문 검색"""
        try:
            # 1. 쿼리 향상
            query_enhancer = PsychologyQueryEnhancer()
            enhanced_query = await query_enhancer.enhance_query(query)

            # 2. 쿼리 임베딩 생성
            query_embedding = self.embedding_generator.generate_query_embedding(enhanced_query)

            # 3. ChromaDB에서 검색
            collection = self.collections['psychology_papers']

            # 필터 변환
            where_filter = self._convert_filters(filters) if filters else None

            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=limit,
                where=where_filter
            )

            # 4. 결과 변환
            search_results = []
            if results['documents'] and len(results['documents']) > 0:
                for i in range(len(results['documents'][0])):
                    metadata_dict = results['metadatas'][0][i]
                    paper_metadata = self._dict_to_paper_metadata(metadata_dict)

                    search_result = SearchResult(
                        title=paper_metadata.title,
                        content_snippet=results['documents'][0][i],
                        similarity_score=1 - results['distances'][0][i],  # ChromaDB는 거리를 반환
                        metadata=paper_metadata,
                        highlights=self._extract_highlights(results['documents'][0][i], query),
                        reasoning=f"Domain: {paper_metadata.research_domain}, Relevance: {1 - results['distances'][0][i]:.3f}"
                    )
                    search_results.append(search_result)

            return search_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def _generate_document_id(self, title: str, metadata: PaperMetadata) -> str:
        """문서 ID 생성"""
        # 파일 경로에서 파일명 추출
        if metadata.file_path:
            filename = Path(metadata.file_path).stem
            return f"{filename}_{metadata.year}"
        else:
            # 제목 기반 ID 생성
            clean_title = "".join(c for c in title if c.isalnum() or c in (' ', '_')).rstrip()
            return f"{clean_title[:30]}_{metadata.year}".replace(' ', '_')

    def _convert_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """필터를 ChromaDB 형식으로 변환"""
        where_filter = {}

        for key, value in filters.items():
            if key == "research_domain":
                where_filter["research_domain"] = {"$eq": value}
            elif key == "year_range":
                where_filter["year"] = {"$gte": value[0], "$lte": value[1]}
            elif key == "authors":
                # 배열 내 포함 검사
                where_filter["authors"] = {"$in": value}

        return where_filter

    def _dict_to_paper_metadata(self, metadata_dict: Dict[str, Any]) -> PaperMetadata:
        """딕셔너리를 PaperMetadata로 변환"""
        return PaperMetadata(
            title=metadata_dict.get('title', ''),
            authors=metadata_dict.get('authors', []),
            year=metadata_dict.get('year', 0),
            journal=metadata_dict.get('journal', ''),
            keywords=metadata_dict.get('keywords', []),
            abstract_ko=metadata_dict.get('abstract_ko', ''),
            abstract_en=metadata_dict.get('abstract_en', ''),
            research_domain=metadata_dict.get('research_domain', 'general_psychology'),
            methodology=metadata_dict.get('methodology', 'theoretical'),
            subject_population=metadata_dict.get('subject_population', 'not_specified'),
            file_path=metadata_dict.get('file_path', ''),
            processing_date=metadata_dict.get('processing_date', ''),
            language=metadata_dict.get('language', 'mixed')
        )

    def _extract_highlights(self, content: str, query: str) -> List[str]:
        """쿼리와 관련된 하이라이트 추출"""
        # 간단한 키워드 매칭
        query_terms = query.lower().split()
        sentences = content.split('.')

        highlights = []
        for sentence in sentences[:3]:  # 처음 3문장만
            if any(term in sentence.lower() for term in query_terms):
                highlights.append(sentence.strip())

        return highlights

    async def get_collection_stats(self) -> Dict[str, Any]:
        """컬렉션 통계 정보"""
        stats = {}

        for name, collection in self.collections.items():
            try:
                count = collection.count()
                stats[name] = {
                    'document_count': count,
                    'status': 'active'
                }
            except Exception as e:
                stats[name] = {
                    'document_count': 0,
                    'status': f'error: {e}'
                }

        return stats


class PsychologyPaperProcessor:
    """논문 배치 처리기"""

    def __init__(self, vector_store: PsychologyVectorStore):
        self.vector_store = vector_store
        self.pdf_processor = PsychologyPDFProcessor()
        self.domain_classifier = PsychologyDomainClassifier()

    async def process_papers_batch(self, paper_files: List[str]) -> List[Dict[str, Any]]:
        """논문 파일들을 배치로 처리"""
        results = []

        for paper_file in paper_files:
            try:
                result = await self.process_single_paper(paper_file)
                results.append(result)
                logger.info(f"Processed: {paper_file}")
            except Exception as e:
                logger.error(f"Failed to process {paper_file}: {e}")
                results.append({
                    'file_path': paper_file,
                    'status': 'failed',
                    'error': str(e)
                })

        return results

    async def process_single_paper(self, paper_file: str) -> Dict[str, Any]:
        """단일 논문 처리"""
        # PDF에서 텍스트 추출
        content = await self.pdf_processor.extract_text_from_pdf(paper_file)

        # 메타데이터 추출
        metadata = await self._extract_metadata_from_filename(paper_file, content)

        # 벡터 스토어에 추가
        success = await self.vector_store.add_paper(
            title=metadata.title,
            content=content,
            metadata=metadata
        )

        return {
            'file_path': paper_file,
            'title': metadata.title,
            'status': 'success' if success else 'failed',
            'metadata': metadata.to_dict()
        }

    async def _extract_metadata_from_filename(self, file_path: str, content: str) -> PaperMetadata:
        """파일명과 내용에서 메타데이터 추출"""
        filename = Path(file_path).stem

        # 파일명에서 연도 추출 (예: kim2023_tdcs.pdf -> 2023)
        year_match = re.search(r'20\d{2}', filename)
        year = int(year_match.group()) if year_match else 2023

        # 연구 도메인 분류
        domain = self.domain_classifier.classify_research_domain(content)

        # 기본 메타데이터 생성
        metadata = PaperMetadata(
            title=filename.replace('_', ' ').title(),
            authors=["추출된 저자명"],  # TODO: 실제 PDF에서 추출
            year=year,
            journal="추출된 저널명",     # TODO: 실제 PDF에서 추출
            keywords=["추출된", "키워드"],  # TODO: NLP로 추출
            research_domain=domain,
            file_path=file_path,
            processing_date=datetime.now().isoformat()
        )

        return metadata


# 사용 예시
if __name__ == "__main__":
    async def main():
        # Vector Store 초기화
        vector_store = PsychologyVectorStore()

        # 샘플 논문 추가
        sample_metadata = PaperMetadata(
            title="ADHD 아동의 실행기능 향상을 위한 tDCS 연구",
            authors=["김철수", "이영희"],
            year=2023,
            journal="Korean Journal of Psychology",
            keywords=["ADHD", "실행기능", "tDCS", "뇌자극"]
        )

        sample_content = """
        본 연구는 ADHD 아동의 실행기능 향상을 위한 tDCS의 효과를 검증했다.
        30명의 ADHD 아동을 대상으로 20분간 전전두엽에 tDCS를 적용한 결과,
        작업기억과 인지유연성이 유의미하게 향상되었다.
        """

        # 논문 추가
        success = await vector_store.add_paper(
            title=sample_metadata.title,
            content=sample_content,
            metadata=sample_metadata
        )

        print(f"Paper added: {success}")

        # 검색 테스트
        results = await vector_store.search_papers("ADHD 실행기능")
        print(f"Search results: {len(results)}")

    # asyncio.run(main())