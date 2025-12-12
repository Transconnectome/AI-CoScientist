"""
Tests for RAGAS evaluator integration.

Following TDD methodology (RED → GREEN → REFACTOR):
1. RED: Write failing tests first
2. GREEN: Implement minimal code to pass
3. REFACTOR: Improve code quality
"""

import pytest
from src.services.rag.ragas_evaluator import RAGASEvaluator


class TestRAGASEvaluator:
    """RAGAS 평가 메트릭 테스트"""

    @pytest.fixture
    def evaluator(self):
        """평가자 인스턴스 생성"""
        return RAGASEvaluator()

    @pytest.fixture
    def sample_data(self):
        """샘플 평가 데이터"""
        return {
            'question': 'What are the key findings of this study?',
            'answer': 'The study found significant correlations between neural activity and cognitive performance.',
            'contexts': [
                'Statistical analysis revealed significant correlations between brain activity patterns and task performance.',
                'Our findings suggest that individual differences in neural processing contribute to variability in cognitive outcomes.'
            ],
            'ground_truth': 'The study demonstrated significant correlations between neural activity patterns and cognitive performance across multiple domains.'
        }

    def test_evaluator_initialization(self, evaluator):
        """평가자 초기화 검증"""
        assert evaluator is not None
        assert hasattr(evaluator, 'evaluate_faithfulness')
        assert hasattr(evaluator, 'evaluate_answer_relevancy')
        assert hasattr(evaluator, 'evaluate_context_precision')
        assert hasattr(evaluator, 'evaluate_context_recall')

    def test_faithfulness_metric(self, evaluator, sample_data):
        """Faithfulness 메트릭 검증"""
        score = evaluator.evaluate_faithfulness(
            question=sample_data['question'],
            answer=sample_data['answer'],
            contexts=sample_data['contexts']
        )

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_answer_relevancy_metric(self, evaluator, sample_data):
        """Answer Relevancy 메트릭 검증"""
        score = evaluator.evaluate_answer_relevancy(
            question=sample_data['question'],
            answer=sample_data['answer']
        )

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_context_precision_metric(self, evaluator, sample_data):
        """Context Precision 메트릭 검증"""
        score = evaluator.evaluate_context_precision(
            question=sample_data['question'],
            contexts=sample_data['contexts'],
            ground_truth=sample_data['ground_truth']
        )

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_context_recall_metric(self, evaluator, sample_data):
        """Context Recall 메트릭 검증"""
        score = evaluator.evaluate_context_recall(
            contexts=sample_data['contexts'],
            ground_truth=sample_data['ground_truth'],
            question=sample_data['question']
        )

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_evaluate_pipeline(self, evaluator, sample_data):
        """전체 파이프라인 평가 검증"""
        results = evaluator.evaluate_pipeline(
            question=sample_data['question'],
            answer=sample_data['answer'],
            contexts=sample_data['contexts'],
            ground_truth=sample_data['ground_truth']
        )

        # 모든 메트릭이 결과에 포함되어야 함
        assert 'faithfulness' in results
        assert 'answer_relevancy' in results
        assert 'context_precision' in results
        assert 'context_recall' in results

        # 모든 점수가 유효한 범위 내에 있어야 함
        for metric, score in results.items():
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_batch_evaluation(self, evaluator):
        """배치 평가 검증"""
        batch_data = [
            {
                'question': 'What is the methodology?',
                'answer': 'Participants underwent fMRI scanning.',
                'contexts': ['Participants were recruited and underwent functional MRI scanning.'],
                'ground_truth': 'The methodology involved fMRI scanning of participants.'
            },
            {
                'question': 'What are the results?',
                'answer': 'Significant correlations were found.',
                'contexts': ['Statistical analysis revealed significant correlations.'],
                'ground_truth': 'The results showed significant correlations between variables.'
            }
        ]

        results = evaluator.evaluate_batch(batch_data)

        assert len(results) == 2
        assert all('faithfulness' in r for r in results)
        assert all('answer_relevancy' in r for r in results)

    def test_invalid_input_handling(self, evaluator):
        """잘못된 입력 처리 검증"""
        with pytest.raises(ValueError):
            evaluator.evaluate_faithfulness(
                question='',
                answer='Some answer',
                contexts=['context']
            )

        with pytest.raises(ValueError):
            evaluator.evaluate_context_recall(
                contexts=[],
                ground_truth='ground truth'
            )

    def test_metric_scoring_consistency(self, evaluator, sample_data):
        """메트릭 점수 일관성 검증"""
        # 같은 입력에 대해 같은 점수가 나와야 함
        try:
            score1 = evaluator.evaluate_faithfulness(
                question=sample_data['question'],
                answer=sample_data['answer'],
                contexts=sample_data['contexts']
            )
            score2 = evaluator.evaluate_faithfulness(
                question=sample_data['question'],
                answer=sample_data['answer'],
                contexts=sample_data['contexts']
            )
            assert score1 == score2
        except ValueError as e:
            # If we get ValueError due to NaN (rate limiting), that's acceptable
            if "NaN" in str(e):
                pytest.skip("Skipping due to API rate limiting")
            raise
