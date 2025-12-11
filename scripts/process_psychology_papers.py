#!/usr/bin/env python3
"""
Psychology Papers Batch Processing Script
66편 심리학 논문 실제 데이터 처리 및 Vector Store 구축

Usage:
    python scripts/process_psychology_papers.py
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

# Safe imports with fallbacks
try:
    from src.services.psychology.psychology_vector_store import (
        PsychologyVectorStore, PaperMetadata, PsychologyPaperProcessor
    )
    VECTOR_STORE_AVAILABLE = True
except Exception as e:
    logger.warning(f"Vector Store 모듈을 로드할 수 없습니다: {e}")
    VECTOR_STORE_AVAILABLE = False

try:
    from src.services.psychology.korean_nlp_processor import KoreanNLPPipeline
    NLP_AVAILABLE = True
except Exception as e:
    logger.warning(f"Korean NLP 모듈을 로드할 수 없습니다: {e}")
    NLP_AVAILABLE = False

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


class PsychologyPapersProcessor:
    """66편 심리학 논문 배치 처리기"""

    def __init__(self):
        self.psychology_dir = Path("data/심리학과")
        self.output_dir = Path("data/processed_papers")
        self.output_dir.mkdir(exist_ok=True)

        # 컴포넌트 초기화
        self.vector_store = None
        self.nlp_pipeline = None
        self.domain_classifier = None

        # 처리 결과 저장
        self.processing_results = {
            "total_papers": 0,
            "processed_successfully": 0,
            "failed_papers": [],
            "domain_distribution": {},
            "author_distribution": {},
            "papers_metadata": []
        }

    async def initialize_components(self):
        """시스템 컴포넌트 초기화"""
        logger.info("🚀 Psychology RAG 시스템 컴포넌트 초기화 중...")

        # Vector Store 초기화
        if VECTOR_STORE_AVAILABLE:
            try:
                self.vector_store = PsychologyVectorStore()
                logger.info("✅ Vector Store 초기화 완료")
            except Exception as e:
                logger.warning(f"❌ Vector Store 초기화 실패: {e}")
                self.vector_store = None

        # NLP Pipeline 초기화
        if NLP_AVAILABLE:
            try:
                self.nlp_pipeline = KoreanNLPPipeline()
                logger.info("✅ Korean NLP Pipeline 초기화 완료")
            except Exception as e:
                logger.warning(f"❌ NLP Pipeline 초기화 실패: {e}")
                self.nlp_pipeline = None

        # Domain Classifier 초기화
        if CLASSIFIER_AVAILABLE:
            try:
                self.domain_classifier = PsychologyDomainClassifier()
                logger.info("✅ Domain Classifier 초기화 완료")
            except Exception as e:
                logger.warning(f"❌ Domain Classifier 초기화 실패: {e}")
                self.domain_classifier = None

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

    def extract_pdf_text(self, pdf_path: Path) -> str:
        """PDF에서 텍스트 추출"""
        if not PDF_AVAILABLE:
            return f"PDF 텍스트 추출을 사용할 수 없습니다. 파일: {pdf_path.name}"

        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""

                # 모든 페이지에서 텍스트 추출
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text += page.extract_text() + "\n"

                return text.strip()

        except Exception as e:
            logger.warning(f"PDF 텍스트 추출 실패 {pdf_path.name}: {e}")
            return f"텍스트 추출 실패: {pdf_path.name}"

    def extract_metadata_from_filename(self, pdf_path: Path) -> Dict[str, Any]:
        """파일명에서 메타데이터 추출"""
        filename = pdf_path.stem
        folder = pdf_path.parent.name

        # 기본 메타데이터
        metadata = {
            "title": filename.replace('_', ' '),
            "authors": [folder],  # 폴더명을 저자로 가정
            "year": 2023,  # 기본값
            "journal": "Unknown Journal",
            "keywords": [],
            "file_path": str(pdf_path),
            "folder": folder
        }

        # 파일명에서 연도 추출 시도
        import re
        year_match = re.search(r'20\d{2}', filename)
        if year_match:
            metadata["year"] = int(year_match.group())

        return metadata

    async def process_single_paper(self, pdf_path: Path) -> Dict[str, Any]:
        """단일 논문 처리"""
        logger.info(f"📄 처리 중: {pdf_path.relative_to(self.psychology_dir)}")

        try:
            # 1. PDF 텍스트 추출
            text_content = self.extract_pdf_text(pdf_path)

            # 2. 메타데이터 생성
            metadata_dict = self.extract_metadata_from_filename(pdf_path)

            # 3. NLP 분석
            if self.nlp_pipeline and len(text_content) > 100:
                try:
                    nlp_result = await self.nlp_pipeline.analyze_text(text_content[:1000])  # 첫 1000자만
                    metadata_dict["psychology_terms"] = [term.korean for term in nlp_result.psychology_terms]
                    metadata_dict["sentiment"] = nlp_result.sentiment
                except Exception as e:
                    logger.warning(f"NLP 분석 실패: {e}")
                    metadata_dict["psychology_terms"] = []
                    metadata_dict["sentiment"] = {"label": "neutral", "confidence": 0.5}

            # 4. 도메인 분류
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

            # 5. Vector Store에 추가
            if self.vector_store:
                try:
                    paper_metadata = PaperMetadata(
                        title=metadata_dict["title"],
                        authors=metadata_dict["authors"],
                        year=metadata_dict["year"],
                        journal=metadata_dict["journal"],
                        keywords=metadata_dict["psychology_terms"],
                        file_path=metadata_dict["file_path"]
                    )

                    success = await self.vector_store.add_paper(
                        title=metadata_dict["title"],
                        content=text_content[:2000],  # 첫 2000자만 저장
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
                "file": str(pdf_path.relative_to(self.psychology_dir)),
                "metadata": metadata_dict
            }

        except Exception as e:
            logger.error(f"❌ 논문 처리 실패 {pdf_path.name}: {e}")
            self.processing_results["failed_papers"].append({
                "file": str(pdf_path.relative_to(self.psychology_dir)),
                "error": str(e)
            })

            return {
                "status": "error",
                "file": str(pdf_path.relative_to(self.psychology_dir)),
                "error": str(e)
            }

    async def process_papers_batch(self, pdf_files: List[Path], batch_size: int = 5):
        """배치로 논문들 처리"""
        logger.info(f"📦 배치 처리 시작: {len(pdf_files)}개 파일, 배치 크기: {batch_size}")

        results = []

        for i in range(0, len(pdf_files), batch_size):
            batch = pdf_files[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(pdf_files) + batch_size - 1) // batch_size

            logger.info(f"📦 배치 {batch_num}/{total_batches} 처리 중 ({len(batch)}개 파일)")

            # 배치 내 파일들을 병렬로 처리
            batch_tasks = [self.process_single_paper(pdf_path) for pdf_path in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"배치 처리 중 예외 발생: {result}")
                else:
                    results.append(result)

            # 배치 간 짧은 대기 (시스템 부하 방지)
            if i + batch_size < len(pdf_files):
                await asyncio.sleep(1)

        return results

    async def save_processing_results(self):
        """처리 결과를 파일로 저장"""
        results_file = self.output_dir / "processing_results.json"

        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.processing_results, f, ensure_ascii=False, indent=2)

            logger.info(f"💾 처리 결과 저장: {results_file}")

        except Exception as e:
            logger.error(f"❌ 처리 결과 저장 실패: {e}")

    def print_processing_summary(self):
        """처리 요약 출력"""
        results = self.processing_results

        print("\n" + "="*60)
        print("📊 Psychology Papers Processing Summary")
        print("="*60)
        print(f"📄 총 논문 수: {results['total_papers']}")
        print(f"✅ 성공 처리: {results['processed_successfully']}")
        print(f"❌ 실패: {len(results['failed_papers'])}")
        print(f"📈 성공률: {(results['processed_successfully']/results['total_papers']*100):.1f}%")

        print(f"\n🔬 연구 영역 분포:")
        for domain, count in sorted(results['domain_distribution'].items()):
            print(f"   {domain}: {count}개")

        print(f"\n👥 저자(폴더) 분포:")
        for author, count in sorted(results['author_distribution'].items()):
            print(f"   {author}: {count}개")

        if results['failed_papers']:
            print(f"\n❌ 실패한 파일들:")
            for failed in results['failed_papers'][:5]:  # 첫 5개만
                print(f"   {failed['file']}: {failed['error']}")
            if len(results['failed_papers']) > 5:
                print(f"   ... (총 {len(results['failed_papers'])}개)")

        print("\n" + "="*60)


async def main():
    """메인 처리 함수"""
    print("🎉 Psychology Papers Batch Processing 시작")
    print("=" * 60)
    print("66편 심리학 논문 실제 데이터 처리 및 Vector Store 구축")
    print("Seoul National University Psychology Department")
    print("=" * 60)

    # 처리기 초기화
    processor = PsychologyPapersProcessor()

    try:
        # 1. 시스템 컴포넌트 초기화
        await processor.initialize_components()

        # 2. 논문 파일들 발견
        pdf_files = processor.discover_papers()

        if not pdf_files:
            print("❌ 처리할 PDF 파일이 없습니다.")
            return

        # 3. 배치 처리 실행
        print(f"\n🚀 {len(pdf_files)}개 논문 배치 처리 시작...")
        results = await processor.process_papers_batch(pdf_files, batch_size=3)

        # 4. 결과 저장
        await processor.save_processing_results()

        # 5. 요약 출력
        processor.print_processing_summary()

        print("\n🎊 Psychology Papers Processing 완료!")
        print("\n📂 다음 단계:")
        print("   🔍 Vector Store에서 검색 테스트")
        print("   🧠 Foundation Models 연동")
        print("   📊 RAG 시스템 성능 평가")

    except Exception as e:
        logger.error(f"메인 처리 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())