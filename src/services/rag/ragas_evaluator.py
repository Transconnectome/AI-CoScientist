"""
RAGAS-based RAG evaluation system.

This module provides a wrapper around RAGAS metrics for evaluating
RAG system quality following the TDD methodology.
"""

import logging
import math
from typing import Any

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

logger = logging.getLogger(__name__)


class RAGASEvaluator:
    """RAGAS 메트릭을 사용한 RAG 평가"""

    def __init__(self) -> None:
        """평가자 초기화"""
        self.metrics = {
            'faithfulness': faithfulness,
            'answer_relevancy': answer_relevancy,
            'context_precision': context_precision,
            'context_recall': context_recall
        }

    def _extract_score(self, score: Any, metric_name: str = "") -> float:
        """점수를 추출하고 검증

        Args:
            score: RAGAS에서 반환된 점수 (pandas Series or float)
            metric_name: 메트릭 이름 (로깅용)

        Returns:
            0.0-1.0 범위의 float 점수

        Raises:
            ValueError: NaN 또는 유효하지 않은 점수인 경우
        """
        # Extract from pandas Series if needed
        if hasattr(score, '__iter__') and not isinstance(score, str):
            score = float(score[0])
        else:
            score = float(score)

        # Validate score
        if math.isnan(score):
            error_msg = f"Metric {metric_name} returned NaN (likely due to API rate limiting or connection error)"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if not (0.0 <= score <= 1.0):
            logger.warning(f"Metric {metric_name} returned score {score} outside [0,1] range")

        return score

    def evaluate_faithfulness(
        self,
        question: str,
        answer: str,
        contexts: list[str]
    ) -> float:
        """Faithfulness 메트릭 평가

        Args:
            question: 질문 텍스트
            answer: 생성된 답변
            contexts: 검색된 컨텍스트 리스트

        Returns:
            0.0-1.0 범위의 faithfulness 점수
        """
        if not question.strip():
            raise ValueError("Question cannot be empty")
        if not contexts:
            raise ValueError("Contexts cannot be empty")

        data = {
            'question': [question],
            'answer': [answer],
            'contexts': [contexts]
        }
        dataset = Dataset.from_dict(data)

        result = evaluate(dataset, metrics=[faithfulness])
        return self._extract_score(result['faithfulness'], 'faithfulness')

    def evaluate_answer_relevancy(
        self,
        question: str,
        answer: str
    ) -> float:
        """Answer Relevancy 메트릭 평가

        Args:
            question: 질문 텍스트
            answer: 생성된 답변

        Returns:
            0.0-1.0 범위의 answer relevancy 점수
        """
        if not question.strip():
            raise ValueError("Question cannot be empty")

        data = {
            'question': [question],
            'answer': [answer]
        }
        dataset = Dataset.from_dict(data)

        result = evaluate(dataset, metrics=[answer_relevancy])
        return self._extract_score(result['answer_relevancy'], 'answer_relevancy')

    def evaluate_context_precision(
        self,
        question: str,
        contexts: list[str],
        ground_truth: str
    ) -> float:
        """Context Precision 메트릭 평가

        Args:
            question: 질문 텍스트
            contexts: 검색된 컨텍스트 리스트
            ground_truth: 정답 텍스트

        Returns:
            0.0-1.0 범위의 context precision 점수
        """
        if not question.strip():
            raise ValueError("Question cannot be empty")
        if not contexts:
            raise ValueError("Contexts cannot be empty")

        data = {
            'question': [question],
            'contexts': [contexts],
            'ground_truth': [ground_truth]
        }
        dataset = Dataset.from_dict(data)

        result = evaluate(dataset, metrics=[context_precision])
        return self._extract_score(result['context_precision'], 'context_precision')

    def evaluate_context_recall(
        self,
        contexts: list[str],
        ground_truth: str,
        question: str = ""
    ) -> float:
        """Context Recall 메트릭 평가

        Args:
            contexts: 검색된 컨텍스트 리스트
            ground_truth: 정답 텍스트
            question: 사용자 질문 (RAGAS context_recall requires 'user_input')

        Returns:
            0.0-1.0 범위의 context recall 점수
        """
        if not contexts:
            raise ValueError("Contexts cannot be empty")

        data = {
            'user_input': [question],
            'contexts': [contexts],
            'ground_truth': [ground_truth]
        }
        dataset = Dataset.from_dict(data)

        result = evaluate(dataset, metrics=[context_recall])
        return self._extract_score(result['context_recall'], 'context_recall')

    def evaluate_pipeline(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str
    ) -> dict[str, float]:
        """전체 RAG 파이프라인 평가

        Args:
            question: 질문 텍스트
            answer: 생성된 답변
            contexts: 검색된 컨텍스트 리스트
            ground_truth: 정답 텍스트

        Returns:
            모든 메트릭 점수를 포함하는 딕셔너리
        """
        data = {
            'question': [question],
            'answer': [answer],
            'contexts': [contexts],
            'ground_truth': [ground_truth]
        }
        dataset = Dataset.from_dict(data)

        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ]
        )

        return {
            'faithfulness': self._extract_score(result['faithfulness'], 'faithfulness'),
            'answer_relevancy': self._extract_score(result['answer_relevancy'], 'answer_relevancy'),
            'context_precision': self._extract_score(result['context_precision'], 'context_precision'),
            'context_recall': self._extract_score(result['context_recall'], 'context_recall')
        }

    def evaluate_batch(
        self,
        batch_data: list[dict[str, Any]]
    ) -> list[dict[str, float]]:
        """배치 데이터 평가

        Args:
            batch_data: 평가할 데이터 리스트
                각 항목은 question, answer, contexts, ground_truth 포함

        Returns:
            각 항목의 평가 결과 리스트
        """
        results = []
        for item in batch_data:
            result = self.evaluate_pipeline(
                question=item['question'],
                answer=item['answer'],
                contexts=item['contexts'],
                ground_truth=item['ground_truth']
            )
            results.append(result)

        return results
