#!/usr/bin/env python3
"""
Safe Psychology Papers Batch Processing Script
66편 심리학 논문 Java 의존성 없이 안전하게 처리 및 Vector Store 구축

Usage:
    python scripts/process_psychology_papers_safe.py
"""

import asyncio
import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Safe imports with graceful fallbacks
try:
    from src.services.psychology.psychology_vector_store import (
        PsychologyVectorStore, PaperMetadata
    )
    VECTOR_STORE_AVAILABLE = True
except Exception as e:
    logger.warning(f"Vector Store 모듈을 로드할 수 없습니다: {e}")
    VECTOR_STORE_AVAILABLE = False

try:
    from src.services.psychology.domain_classifier import PsychologyDomainClassifier
    CLASSIFIER_AVAILABLE = True
except Exception as e:
    logger.warning(f"Domain Classifier 모듈을 로드할 수 없습니다: {e}")
    CLASSIFIER_AVAILABLE = False

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    logger.warning("PyPDF2를 사용할 수 없습니다. pip install PyPDF2로 설치하세요.")
    PDF_AVAILABLE = False

# Safe Korean NLP imports (avoiding KoNLPy)
try:
    from src.services.psychology.korean_nlp_processor import (
        PsychologyTermExtractor, PsychologyTermMapper,
        KoreanSentimentAnalyzer
    )
    SAFE_NLP_AVAILABLE = True
except Exception as e:
    logger.warning(f"Safe NLP 모듈을 로드할 수 없습니다: {e}")
    SAFE_NLP_AVAILABLE = False


class SafePsychologyPapersProcessor:
    """Java 의존성 없이 66편 심리학 논문 안전 배치 처리기"""

    def __init__(self):
        self.psychology_dir = Path("data/심리학과")
        self.output_dir = Path("data/processed_papers")
        self.output_dir.mkdir(exist_ok=True)

        # 안전한 컴포넌트만 초기화
        self.vector_store = None
        self.domain_classifier = None
        self.term_extractor = None
        self.term_mapper = None
        self.sentiment_analyzer = None

        # 처리 결과 저장
        self.processing_results = {
            "total_papers": 0,
            "processed_successfully": 0,
            "failed_papers": [],
            "domain_distribution": {},
            "author_distribution": {},
            "papers_metadata": [],
            "system_info": {
                "java_safe_mode": True,
                "vector_store_available": VECTOR_STORE_AVAILABLE,
                "classifier_available": CLASSIFIER_AVAILABLE,
                "pdf_available": PDF_AVAILABLE,
                "safe_nlp_available": SAFE_NLP_AVAILABLE
            }
        }

    async def initialize_components(self):
        """안전한 시스템 컴포넌트 초기화 (Java 의존성 없이)"""
        logger.info("🚀 Safe Psychology RAG 시스템 컴포넌트 초기화 중...")

        # Vector Store 초기화 (ChromaDB in-memory fallback)
        if VECTOR_STORE_AVAILABLE:
            try:
                self.vector_store = PsychologyVectorStore()
                logger.info("✅ Vector Store 초기화 완료 (in-memory mode)")
            except Exception as e:
                logger.warning(f"❌ Vector Store 초기화 실패: {e}")
                self.vector_store = None

        # Domain Classifier 초기화
        if CLASSIFIER_AVAILABLE:
            try:
                self.domain_classifier = PsychologyDomainClassifier()
                logger.info("✅ Domain Classifier 초기화 완료")
            except Exception as e:
                logger.warning(f"❌ Domain Classifier 초기화 실패: {e}")
                self.domain_classifier = None

        # Safe NLP 컴포넌트 초기화 (KoNLPy 없이)
        if SAFE_NLP_AVAILABLE:
            try:
                self.term_extractor = PsychologyTermExtractor()
                self.term_mapper = PsychologyTermMapper()
                self.sentiment_analyzer = KoreanSentimentAnalyzer()
                logger.info("✅ Safe Korean NLP 컴포넌트 초기화 완료")
            except Exception as e:
                logger.warning(f"❌ Safe NLP 컴포넌트 초기화 실패: {e}")

    def discover_papers(self) -> List[Path]:
        """심리학 논문 파일 발견"""
        logger.info(f"📁 논문 파일 탐색: {self.psychology_dir}")

        if not self.psychology_dir.exists():
            logger.error(f"❌ 심리학과 데이터 폴더가 존재하지 않습니다: {self.psychology_dir}")
            return []

        # PDF 파일들 찾기
        pdf_files = list(self.psychology_dir.rglob("*.pdf"))
        logger.info(f"📄 발견된 PDF 파일: {len(pdf_files)}개")

        # 폴더별 분류
        folder_counts = {}
        for pdf_file in pdf_files:
            folder = pdf_file.parent.name
            folder_counts[folder] = folder_counts.get(folder, 0) + 1

        logger.info("📁 폴더별 분포:")
        for folder, count in sorted(folder_counts.items()):
            logger.info(f"   {folder}: {count}개")

        self.processing_results["total_papers"] = len(pdf_files)
        return pdf_files

    def extract_pdf_text_safe(self, pdf_path: Path) -> str:
        """안전한 PDF 텍스트 추출"""
        if not PDF_AVAILABLE:
            return f"PDF 처리 미지원. 파일: {pdf_path.name}"

        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""

                # 처음 3페이지만 추출 (안전성과 성능을 위해)
                max_pages = min(3, len(reader.pages))
                for page_num in range(max_pages):
                    try:
                        page = reader.pages[page_num]
                        page_text = page.extract_text()
                        text += page_text + "\n"
                    except Exception as page_e:
                        logger.warning(f"페이지 {page_num} 추출 실패: {page_e}")
                        continue

                # 텍스트 길이 제한 (메모리 안전성)
                if len(text) > 5000:
                    text = text[:5000] + "... [텍스트 자르기]"

                return text.strip()

        except Exception as e:
            logger.warning(f"PDF 텍스트 추출 실패 {pdf_path.name}: {e}")
            return f"텍스트 추출 실패: {pdf_path.name}"

    def extract_metadata_from_path(self, pdf_path: Path) -> Dict[str, Any]:
        """파일 경로에서 안전하게 메타데이터 추출"""
        filename = pdf_path.stem
        folder = pdf_path.parent.name

        # 기본 메타데이터
        metadata = {
            "title": filename.replace('_', ' ').replace('-', ' '),
            "authors": [folder],  # 폴더명을 연구자 이름으로 사용
            "year": 2023,  # 기본값
            "journal": "Korean Psychology Journal",
            "keywords": [],
            "file_path": str(pdf_path),
            "folder": folder,
            "file_size_mb": round(pdf_path.stat().st_size / (1024 * 1024), 2)
        }

        # 파일명에서 연도 추출 시도
        import re
        year_match = re.search(r'20\d{2}', filename)
        if year_match:
            metadata["year"] = int(year_match.group())

        return metadata

    async def process_single_paper(self, pdf_path: Path) -> Dict[str, Any]:
        """단일 논문 안전 처리"""
        relative_path = pdf_path.relative_to(self.psychology_dir)
        logger.info(f"📄 처리 중: {relative_path}")

        try:
            # 1. PDF 텍스트 추출 (안전 모드)
            text_content = self.extract_pdf_text_safe(pdf_path)

            # 2. 메타데이터 생성
            metadata_dict = self.extract_metadata_from_path(pdf_path)

            # 3. 안전한 NLP 분석 (KoNLPy 없이)
            if self.term_extractor and len(text_content) > 100:
                try:
                    # 심리학 용어 추출 (KoNLPy 없는 버전)
                    psychology_terms = self.term_extractor.extract_psychology_terms(
                        f"{metadata_dict['title']} {text_content[:1000]}"
                    )
                    metadata_dict["psychology_terms"] = psychology_terms[:10]  # 최대 10개만

                    # 한영 매핑
                    if self.term_mapper:
                        english_mappings = self.term_mapper.map_to_english(psychology_terms[:5])
                        metadata_dict["english_mappings"] = english_mappings

                    # 감정 분석
                    if self.sentiment_analyzer:
                        sentiment = self.sentiment_analyzer.analyze_sentiment(text_content[:500])
                        metadata_dict["sentiment"] = sentiment

                except Exception as e:
                    logger.warning(f"Safe NLP 분석 실패: {e}")
                    metadata_dict["psychology_terms"] = []
                    metadata_dict["sentiment"] = {"label": "neutral", "confidence": 0.5}

            # 4. 도메인 분류 (안전)
            if self.domain_classifier:
                try:
                    domain_result = self.domain_classifier.classify_detailed(
                        f"{metadata_dict['title']} {text_content[:500]}"
                    )
                    metadata_dict["research_domain"] = domain_result.primary_domain
                    metadata_dict["methodology"] = domain_result.methodology
                    metadata_dict["target_population"] = domain_result.target_population
                    metadata_dict["domain_confidence"] = domain_result.confidence
                except Exception as e:
                    logger.warning(f"도메인 분류 실패: {e}")
                    metadata_dict["research_domain"] = "general_psychology"
                    metadata_dict["methodology"] = "not_specified"
                    metadata_dict["target_population"] = "not_specified"
                    metadata_dict["domain_confidence"] = 0.1

            # 5. Vector Store에 안전하게 추가
            if self.vector_store:
                try:
                    paper_metadata = PaperMetadata(
                        title=metadata_dict["title"],
                        authors=metadata_dict["authors"],
                        year=metadata_dict["year"],
                        journal=metadata_dict["journal"],
                        keywords=metadata_dict.get("psychology_terms", [])[:5],  # 최대 5개
                        file_path=metadata_dict["file_path"]
                    )

                    # 텍스트 내용 길이 제한 (Vector Store 안전성)
                    safe_content = text_content[:2000] if len(text_content) > 2000 else text_content

                    success = await self.vector_store.add_paper(
                        title=metadata_dict["title"],
                        content=safe_content,
                        metadata=paper_metadata
                    )

                    if success:
                        logger.info(f"✅ Vector Store 저장 성공: {pdf_path.name}")
                        metadata_dict["vector_store_added"] = True
                    else:
                        logger.warning(f"❌ Vector Store 저장 실패: {pdf_path.name}")
                        metadata_dict["vector_store_added"] = False

                except Exception as e:
                    logger.warning(f"Vector Store 저장 중 오류: {e}")
                    metadata_dict["vector_store_added"] = False

            # 처리 결과 업데이트
            self.processing_results["processed_successfully"] += 1

            # 도메인 분포 업데이트
            domain = metadata_dict.get("research_domain", "unknown")
            self.processing_results["domain_distribution"][domain] = \
                self.processing_results["domain_distribution"].get(domain, 0) + 1

            # 저자 분포 업데이트
            author = metadata_dict.get("folder", "unknown")
            self.processing_results["author_distribution"][author] = \
                self.processing_results["author_distribution"].get(author, 0) + 1

            # 메타데이터 저장
            self.processing_results["papers_metadata"].append(metadata_dict)

            return {
                "status": "success",
                "file": str(relative_path),
                "metadata": metadata_dict
            }

        except Exception as e:
            logger.error(f"❌ 논문 처리 실패 {pdf_path.name}: {e}")
            self.processing_results["failed_papers"].append({
                "file": str(relative_path),
                "error": str(e)
            })

            return {
                "status": "error",
                "file": str(relative_path),
                "error": str(e)
            }

    async def process_papers_batch_safe(self, pdf_files: List[Path], batch_size: int = 3):
        """안전한 배치 처리 (작은 배치 크기로)"""
        logger.info(f"📦 안전 배치 처리 시작: {len(pdf_files)}개 파일, 배치 크기: {batch_size}")

        results = []

        for i in range(0, len(pdf_files), batch_size):
            batch = pdf_files[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(pdf_files) + batch_size - 1) // batch_size

            logger.info(f"📦 배치 {batch_num}/{total_batches} 처리 중 ({len(batch)}개 파일)")

            # 배치 내 파일들을 순차적으로 처리 (안전성 우선)
            for pdf_path in batch:
                try:
                    result = await self.process_single_paper(pdf_path)
                    results.append(result)
                except Exception as e:
                    logger.error(f"논문 처리 중 예외 발생: {e}")
                    results.append({
                        "status": "error",
                        "file": str(pdf_path.relative_to(self.psychology_dir)),
                        "error": str(e)
                    })

                # 논문 간 짧은 대기 (시스템 안정성)
                await asyncio.sleep(0.5)

            # 배치 간 대기
            if i + batch_size < len(pdf_files):
                await asyncio.sleep(2)

        return results

    async def save_processing_results(self):
        """처리 결과를 JSON 파일로 저장"""
        results_file = self.output_dir / "safe_processing_results.json"

        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.processing_results, f, ensure_ascii=False, indent=2)

            logger.info(f"💾 안전 처리 결과 저장: {results_file}")

        except Exception as e:
            logger.error(f"❌ 처리 결과 저장 실패: {e}")

    def print_processing_summary(self):
        """처리 요약 출력"""
        results = self.processing_results

        print("\n" + "="*60)
        print("📊 Safe Psychology Papers Processing Summary")
        print("="*60)
        print(f"📄 총 논문 수: {results['total_papers']}")
        print(f"✅ 성공 처리: {results['processed_successfully']}")
        print(f"❌ 실패: {len(results['failed_papers'])}")

        if results['total_papers'] > 0:
            success_rate = (results['processed_successfully'] / results['total_papers'] * 100)
            print(f"📈 성공률: {success_rate:.1f}%")

        print(f"\n🔬 연구 영역 분포:")
        for domain, count in sorted(results['domain_distribution'].items()):
            print(f"   {domain}: {count}개")

        print(f"\n👥 연구자(폴더) 분포:")
        for author, count in sorted(results['author_distribution'].items()):
            print(f"   {author}: {count}개")

        print(f"\n🛡️ 시스템 안전성 정보:")
        system_info = results['system_info']
        print(f"   Java Safe Mode: {'✅' if system_info['java_safe_mode'] else '❌'}")
        print(f"   Vector Store: {'✅' if system_info['vector_store_available'] else '❌'}")
        print(f"   Domain Classifier: {'✅' if system_info['classifier_available'] else '❌'}")
        print(f"   PDF Processing: {'✅' if system_info['pdf_available'] else '❌'}")
        print(f"   Safe NLP: {'✅' if system_info['safe_nlp_available'] else '❌'}")

        if results['failed_papers']:
            print(f"\n❌ 실패한 파일들 (처음 3개):")
            for failed in results['failed_papers'][:3]:
                print(f"   {failed['file']}: {failed['error']}")
            if len(results['failed_papers']) > 3:
                print(f"   ... (총 {len(results['failed_papers'])}개)")

        print("\n" + "="*60)


async def main():
    """메인 안전 처리 함수"""
    print("🎉 Safe Psychology Papers Batch Processing 시작")
    print("=" * 60)
    print("66편 심리학 논문 안전 데이터 처리 및 Vector Store 구축")
    print("Seoul National University Psychology Department")
    print("Java 의존성 없는 안전 모드")
    print("=" * 60)

    # 안전 처리기 초기화
    processor = SafePsychologyPapersProcessor()

    try:
        # 1. 안전한 시스템 컴포넌트 초기화
        await processor.initialize_components()

        # 2. 논문 파일들 발견
        pdf_files = processor.discover_papers()

        if not pdf_files:
            print("❌ 처리할 PDF 파일이 없습니다.")
            return

        # 3. 안전한 배치 처리 실행
        print(f"\n🚀 {len(pdf_files)}개 논문 안전 배치 처리 시작...")
        results = await processor.process_papers_batch_safe(pdf_files, batch_size=2)

        # 4. 결과 저장
        await processor.save_processing_results()

        # 5. 요약 출력
        processor.print_processing_summary()

        print("\n🎊 Safe Psychology Papers Processing 완료!")
        print("\n📂 처리 결과:")
        print("   💾 data/processed_papers/safe_processing_results.json")
        print("\n🔧 다음 단계:")
        print("   🔍 Vector Store에서 검색 테스트")
        print("   🧠 Foundation Models 통합 테스트")
        print("   📊 RAG 시스템 성능 평가")
        print("   🌐 Production ChromaDB 연동")

    except Exception as e:
        logger.error(f"메인 처리 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 환경변수 설정 (Java 의존성 회피)
    os.environ["JAVA_TOOL_OPTIONS"] = ""

    # 안전 모드 실행
    asyncio.run(main())