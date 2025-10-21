"""
RAG evaluation dataset creation and management.

This module implements test case generation for RAG system evaluation
following the TDD methodology.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, validator


class TestCaseMetadata(BaseModel):
    """테스트 케이스 메타데이터 스키마"""
    section: str
    improvement_type: str
    created_at: str


class TestCase(BaseModel):
    """테스트 케이스 스키마"""
    id: str
    query: str
    content: str
    expected_answer: str
    relevant_patterns: list[str] = Field(default_factory=list)
    metadata: TestCaseMetadata

    @validator('query')
    def query_not_empty(cls, v: str) -> str:
        """쿼리가 비어있지 않은지 검증"""
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v

    @validator('relevant_patterns')
    def patterns_not_empty(cls, v: list[str]) -> list[str]:
        """관련 패턴이 최소 1개 이상인지 검증"""
        if len(v) == 0:
            raise ValueError('At least one relevant pattern required')
        return v


class EvaluationDataset:
    """RAG 평가를 위한 테스트 데이터셋 생성"""

    def __init__(self, storage_path: str = "tests/fixtures/rag_test_cases.json"):
        self.storage_path = Path(storage_path)
        self.test_cases: list[dict[str, Any]] = []

    def create_test_cases(self, count: int = 100) -> list[dict[str, Any]]:
        """테스트 케이스 생성

        Args:
            count: 생성할 테스트 케이스 수

        Returns:
            생성된 테스트 케이스 리스트 (Dict 형태로 반환하여 하위 호환성 유지)
        """
        sections = ['Abstract', 'Introduction', 'Methods', 'Results', 'Discussion']
        improvement_types = ['CLARITY', 'QUANTITATIVE', 'NOVELTY', 'METHODOLOGY']

        test_cases = []
        for i in range(count):
            section = sections[i % len(sections)]
            imp_type = improvement_types[i % len(improvement_types)]

            # Pydantic 모델로 생성하여 검증
            test_case_model = TestCase(
                id=f'test_{i+1}',
                query=f'Improve {section} {imp_type.lower()}',
                content=self._generate_sample_content(section),
                expected_answer=f'Improved {section} with {imp_type} focus',
                relevant_patterns=[f'pattern_{section}_{imp_type}'],
                metadata=TestCaseMetadata(
                    section=section,
                    improvement_type=imp_type,
                    created_at=datetime.now().isoformat()
                )
            )

            # Dict로 변환 (기존 테스트 호환성)
            test_cases.append(test_case_model.model_dump())

        self.test_cases = test_cases
        return test_cases

    def label_ground_truth(self, test_case: dict[str, Any]) -> dict[str, Any]:
        """Ground truth 레이블 추가

        Args:
            test_case: 레이블링할 테스트 케이스

        Returns:
            레이블이 추가된 테스트 케이스
        """
        # 간단한 규칙 기반 레이블링 (실제로는 수동 레이블링 또는 고품질 LLM 사용)
        if 'expected_answer' not in test_case:
            test_case['expected_answer'] = f"Improved {test_case['content'][:50]}..."

        if 'relevant_patterns' not in test_case:
            section = test_case.get('metadata', {}).get('section', 'Unknown')
            imp_type = test_case.get('metadata', {}).get('improvement_type', 'GENERAL')
            test_case['relevant_patterns'] = [f"pattern_{section}_{imp_type}"]

        return test_case

    def save_to_file(self) -> None:
        """데이터셋 파일로 저장"""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump(self.test_cases, f, indent=2)

    def load_from_file(self) -> list[dict[str, Any]]:
        """저장된 데이터셋 로드

        Returns:
            로드된 테스트 케이스 리스트
        """
        if self.storage_path.exists():
            with open(self.storage_path) as f:
                self.test_cases = json.load(f)
        return self.test_cases

    def _generate_sample_content(self, section: str) -> str:
        """샘플 콘텐츠 생성

        Args:
            section: 논문 섹션 이름

        Returns:
            생성된 샘플 텍스트
        """
        templates = {
            'Abstract': 'This study investigates the relationship between neural activity and cognitive performance in healthy adults.',
            'Introduction': 'Recent advances in neuroimaging have enabled researchers to examine brain function with unprecedented spatial and temporal resolution.',
            'Methods': 'Participants were recruited from the local community and underwent functional MRI scanning while performing cognitive tasks.',
            'Results': 'Statistical analysis revealed significant correlations between brain activity patterns and task performance across multiple cognitive domains.',
            'Discussion': 'Our findings suggest that individual differences in neural processing contribute to variability in cognitive outcomes.'
        }
        return templates.get(section, 'Sample content for evaluation')
