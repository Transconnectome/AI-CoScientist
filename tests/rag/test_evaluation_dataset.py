"""
Tests for RAG evaluation dataset creation.

Following TDD methodology (RED → GREEN → REFACTOR):
1. RED: Write failing tests first
2. GREEN: Implement minimal code to pass
3. REFACTOR: Improve code quality
"""

import pytest
from pathlib import Path
from src.services.rag.evaluation_dataset import EvaluationDataset


class TestEvaluationDataset:
    """평가 데이터셋 테스트"""

    def test_dataset_creation(self):
        """테스트 케이스 생성 검증"""
        # RED: 아직 구현 안됨, 실패해야 함
        dataset = EvaluationDataset()
        test_cases = dataset.create_test_cases(count=10)

        assert len(test_cases) == 10
        assert all('query' in case for case in test_cases)
        assert all('expected_answer' in case for case in test_cases)
        assert all('relevant_patterns' in case for case in test_cases)

    def test_ground_truth_labeling(self):
        """Ground truth 레이블 검증"""
        dataset = EvaluationDataset()
        test_case = {
            'id': 'test_1',
            'query': 'Improve Abstract clarity',
            'content': 'Abstract text...'
        }

        labeled = dataset.label_ground_truth(test_case)

        assert 'expected_answer' in labeled
        assert 'relevant_patterns' in labeled
        assert isinstance(labeled['relevant_patterns'], list)
        assert len(labeled['relevant_patterns']) > 0

    def test_dataset_validation(self):
        """데이터셋 품질 검증"""
        dataset = EvaluationDataset()
        test_cases = dataset.create_test_cases(count=100)

        # 모든 섹션이 커버되는지 확인
        sections = [case['metadata']['section'] for case in test_cases]
        assert 'Abstract' in sections
        assert 'Introduction' in sections
        assert 'Methods' in sections
        assert 'Results' in sections

        # 다양한 개선 타입 확인
        types = [case['metadata']['improvement_type'] for case in test_cases]
        assert 'CLARITY' in types
        assert 'QUANTITATIVE' in types
        assert 'NOVELTY' in types

    def test_save_and_load(self):
        """데이터셋 저장 및 로드 테스트"""
        dataset = EvaluationDataset(storage_path="tests/fixtures/test_rag_cases.json")

        # 생성 및 저장
        original_cases = dataset.create_test_cases(count=10)
        dataset.save_to_file()

        # 로드
        loaded_dataset = EvaluationDataset(storage_path="tests/fixtures/test_rag_cases.json")
        loaded_cases = loaded_dataset.load_from_file()

        assert len(loaded_cases) == len(original_cases)
        assert loaded_cases[0]['id'] == original_cases[0]['id']

        # 정리
        storage_path = Path("tests/fixtures/test_rag_cases.json")
        if storage_path.exists():
            storage_path.unlink()

    def test_dataset_diversity(self):
        """데이터셋 다양성 검증"""
        dataset = EvaluationDataset()
        test_cases = dataset.create_test_cases(count=100)

        # 섹션 분포 확인
        section_counts = {}
        for case in test_cases:
            section = case['metadata']['section']
            section_counts[section] = section_counts.get(section, 0) + 1

        # 모든 섹션이 균등하게 분포되어야 함
        assert len(section_counts) >= 4  # 최소 4개 섹션

        # 각 섹션이 최소 10% 이상 차지
        for count in section_counts.values():
            assert count >= 10
