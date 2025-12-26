"""
Psychology PDF Processor
심리학 논문 PDF 텍스트 추출 및 처리

Features:
1. PDF 텍스트 추출
2. 메타데이터 추출 (제목, 저자, 연도)
3. 섹션별 분리 (초록, 본문, 결론)
4. 한국어/영어 이중언어 처리
5. 과학 논문 구조 인식
"""

import asyncio
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# PDF processing
try:
    import PyPDF2
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logging.warning("PDF libraries not available. Install PyPDF2 and PyMuPDF")

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExtractedContent:
    """추출된 PDF 내용"""
    full_text: str
    title: str = ""
    authors: List[str] = None
    abstract_ko: str = ""
    abstract_en: str = ""
    sections: Dict[str, str] = None
    references: List[str] = None
    language: str = "mixed"  # ko, en, mixed

    def __post_init__(self):
        if self.authors is None:
            self.authors = []
        if self.sections is None:
            self.sections = {}
        if self.references is None:
            self.references = []


class PsychologyPDFProcessor:
    """심리학 논문 PDF 처리기"""

    def __init__(self):
        self.section_patterns = {
            'abstract_ko': re.compile(r'(초록|요약|개요)\s*\n(.+?)(?=\n\s*[A-Z]|$)', re.DOTALL | re.IGNORECASE),
            'abstract_en': re.compile(r'(abstract|summary)\s*\n(.+?)(?=\n\s*[가-힣]|키워드|keywords|$)', re.DOTALL | re.IGNORECASE),
            'introduction': re.compile(r'(서론|introduction|서 론)\s*\n(.+?)(?=\n\s*(방법|method|재료|material)|$)', re.DOTALL | re.IGNORECASE),
            'method': re.compile(r'(방법|method|재료 및 방법|materials and methods)\s*\n(.+?)(?=\n\s*(결과|result)|$)', re.DOTALL | re.IGNORECASE),
            'results': re.compile(r'(결과|results?)\s*\n(.+?)(?=\n\s*(논의|discussion|결론)|$)', re.DOTALL | re.IGNORECASE),
            'discussion': re.compile(r'(논의|discussion|고찰)\s*\n(.+?)(?=\n\s*(결론|conclusion|참고문헌)|$)', re.DOTALL | re.IGNORECASE),
            'conclusion': re.compile(r'(결론|conclusion)\s*\n(.+?)(?=\n\s*(참고문헌|reference)|$)', re.DOTALL | re.IGNORECASE)
        }

        self.title_patterns = [
            re.compile(r'^(.+?)\n\s*[가-힣\s]*대학교', re.MULTILINE),  # 한국 논문
            re.compile(r'^(.+?)\n\s*Abstract', re.MULTILINE | re.IGNORECASE),  # 영어 논문
            re.compile(r'^(.{10,100}?)\n', re.MULTILINE)  # 첫 줄이 제목인 경우
        ]

        self.author_patterns = [
            re.compile(r'([가-힣]{2,4})\s*[,·]\s*([가-힣]{2,4})'),  # 한국 저자
            re.compile(r'([A-Z][a-z]+\s+[A-Z][a-z]+)'),  # 영어 저자
        ]

    async def extract_text_from_pdf(self, pdf_path: str) -> str:
        """PDF에서 텍스트 추출"""
        if not PDF_AVAILABLE:
            logger.warning("PDF libraries not available")
            return "PDF processing not available"

        try:
            # PyMuPDF 사용 (한국어 처리 더 좋음)
            return await self._extract_with_pymupdf(pdf_path)
        except Exception as e:
            logger.warning(f"PyMuPDF failed: {e}, trying PyPDF2")
            try:
                return await self._extract_with_pypdf2(pdf_path)
            except Exception as e2:
                logger.error(f"Both PDF extraction methods failed: {e2}")
                return ""

    async def _extract_with_pymupdf(self, pdf_path: str) -> str:
        """PyMuPDF로 텍스트 추출"""
        doc = fitz.open(pdf_path)
        text_content = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()

            # 텍스트 정제
            cleaned_text = self._clean_extracted_text(text)
            if cleaned_text.strip():
                text_content.append(cleaned_text)

        doc.close()
        return '\n\n'.join(text_content)

    async def _extract_with_pypdf2(self, pdf_path: str) -> str:
        """PyPDF2로 텍스트 추출"""
        text_content = []

        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)

            for page in pdf_reader.pages:
                text = page.extract_text()
                cleaned_text = self._clean_extracted_text(text)
                if cleaned_text.strip():
                    text_content.append(cleaned_text)

        return '\n\n'.join(text_content)

    def _clean_extracted_text(self, text: str) -> str:
        """추출된 텍스트 정제"""
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)

        # 페이지 번호 제거
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

        # 불필요한 줄바꿈 정리
        text = re.sub(r'\n+', '\n', text)

        return text.strip()

    async def extract_structured_content(self, pdf_path: str) -> ExtractedContent:
        """구조화된 내용 추출"""
        # 전체 텍스트 추출
        full_text = await self.extract_text_from_pdf(pdf_path)

        if not full_text:
            return ExtractedContent(full_text="")

        # 구조화된 내용 생성
        content = ExtractedContent(full_text=full_text)

        # 제목 추출
        content.title = self._extract_title(full_text)

        # 저자 추출
        content.authors = self._extract_authors(full_text)

        # 초록 추출
        content.abstract_ko, content.abstract_en = self._extract_abstracts(full_text)

        # 섹션별 내용 추출
        content.sections = self._extract_sections(full_text)

        # 언어 감지
        content.language = self._detect_language(full_text)

        return content

    def _extract_title(self, text: str) -> str:
        """제목 추출"""
        lines = text.split('\n')[:5]  # 처음 5줄에서 찾기

        for pattern in self.title_patterns:
            for line in lines:
                match = pattern.search(line)
                if match:
                    title = match.group(1).strip()
                    if 10 <= len(title) <= 200:  # 적절한 길이의 제목
                        return title

        # 제목을 찾지 못한 경우 첫 줄 반환
        first_line = lines[0] if lines else "제목 없음"
        return first_line[:100] if len(first_line) > 100 else first_line

    def _extract_authors(self, text: str) -> List[str]:
        """저자명 추출"""
        authors = []
        lines = text.split('\n')[:10]  # 처음 10줄에서 찾기

        for line in lines:
            # 한국 저자명 패턴
            korean_authors = re.findall(r'[가-힣]{2,4}(?:\s*[,·]\s*[가-힣]{2,4})*', line)
            if korean_authors:
                for author_group in korean_authors:
                    individual_authors = re.split(r'[,·]', author_group)
                    authors.extend([name.strip() for name in individual_authors if name.strip()])

            # 영어 저자명 패턴
            english_authors = re.findall(r'[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s*,\s*[A-Z][a-z]+\s+[A-Z][a-z]+)*', line)
            if english_authors:
                for author_group in english_authors:
                    individual_authors = author_group.split(',')
                    authors.extend([name.strip() for name in individual_authors if name.strip()])

        return list(set(authors))[:5]  # 최대 5명, 중복 제거

    def _extract_abstracts(self, text: str) -> Tuple[str, str]:
        """초록 추출 (한국어/영어)"""
        abstract_ko = ""
        abstract_en = ""

        # 한국어 초록
        ko_match = self.section_patterns['abstract_ko'].search(text)
        if ko_match:
            abstract_ko = ko_match.group(2).strip()[:500]  # 최대 500자

        # 영어 초록
        en_match = self.section_patterns['abstract_en'].search(text)
        if en_match:
            abstract_en = en_match.group(2).strip()[:500]  # 최대 500자

        return abstract_ko, abstract_en

    def _extract_sections(self, text: str) -> Dict[str, str]:
        """섹션별 내용 추출"""
        sections = {}

        for section_name, pattern in self.section_patterns.items():
            if section_name.startswith('abstract_'):
                continue  # 초록은 별도로 처리

            match = pattern.search(text)
            if match:
                content = match.group(2).strip()[:1000]  # 최대 1000자
                sections[section_name] = content

        return sections

    def _detect_language(self, text: str) -> str:
        """언어 감지"""
        korean_chars = len(re.findall(r'[가-힣]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = korean_chars + english_chars

        if total_chars == 0:
            return "unknown"

        korean_ratio = korean_chars / total_chars

        if korean_ratio > 0.7:
            return "ko"
        elif korean_ratio < 0.3:
            return "en"
        else:
            return "mixed"

    async def extract_metadata_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """PDF에서 메타데이터 추출"""
        content = await self.extract_structured_content(pdf_path)

        # 연도 추출 (파일명이나 내용에서)
        year = self._extract_year(pdf_path, content.full_text)

        # 키워드 추출 (간단한 방법)
        keywords = self._extract_keywords(content.full_text)

        metadata = {
            'title': content.title,
            'authors': content.authors,
            'year': year,
            'abstract_ko': content.abstract_ko,
            'abstract_en': content.abstract_en,
            'language': content.language,
            'keywords': keywords,
            'sections': content.sections,
            'file_path': pdf_path
        }

        return metadata

    def _extract_year(self, pdf_path: str, text: str) -> int:
        """연도 추출"""
        # 파일명에서 연도 추출
        filename = Path(pdf_path).stem
        year_match = re.search(r'20\d{2}', filename)
        if year_match:
            return int(year_match.group())

        # 텍스트에서 연도 추출
        year_matches = re.findall(r'20\d{2}', text[:1000])  # 처음 1000자에서 찾기
        if year_matches:
            return int(year_matches[0])

        # 기본값
        return 2023

    def _extract_keywords(self, text: str) -> List[str]:
        """키워드 추출 (간단한 방법)"""
        # 키워드 섹션 찾기
        keyword_patterns = [
            re.compile(r'키워드[:\s]*(.+?)(?=\n|$)', re.IGNORECASE),
            re.compile(r'keywords[:\s]*(.+?)(?=\n|$)', re.IGNORECASE),
            re.compile(r'key words[:\s]*(.+?)(?=\n|$)', re.IGNORECASE)
        ]

        for pattern in keyword_patterns:
            match = pattern.search(text)
            if match:
                keyword_text = match.group(1)
                # 콤마, 세미콜론, 한국어 쉼표로 분리
                keywords = re.split(r'[,;，、]', keyword_text)
                return [kw.strip() for kw in keywords if kw.strip()][:10]  # 최대 10개

        return []


# 사용 예시
if __name__ == "__main__":
    async def main():
        processor = PsychologyPDFProcessor()

        # 테스트 파일 (실제 경로로 변경)
        test_file = "data/심리학과/안우영/kim2023_tdcs.pdf"

        if Path(test_file).exists():
            content = await processor.extract_structured_content(test_file)
            print(f"Title: {content.title}")
            print(f"Authors: {content.authors}")
            print(f"Language: {content.language}")
            print(f"Abstract (KO): {content.abstract_ko[:100]}...")
        else:
            print("Test file not found")

    # asyncio.run(main())