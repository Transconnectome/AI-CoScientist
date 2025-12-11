#!/usr/bin/env python3
"""
Data Quality Assessor for DD-RAPTOR System
2025 Research: Data-Centric AI with In-Run Data Shapley
"""

import logging
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DataQualityMetrics:
    """데이터 품질 메트릭"""
    completeness: float  # 완성도 (0-1)
    consistency: float   # 일관성 (0-1)
    accuracy: float      # 정확도 (0-1)
    timeliness: float    # 시의성 (0-1)
    overall_score: float # 전체 점수 (0-1)
    issues: List[str]    # 발견된 문제들

class DataQualityAssessor:
    """데이터 품질 평가기 (2025 Data-Centric AI 패턴)"""

    def __init__(self):
        self.quality_thresholds = {
            "completeness": 0.85,
            "consistency": 0.90,
            "accuracy": 0.80,
            "timeliness": 0.75
        }

    async def assess_dataset(self, dataset: Dict) -> DataQualityMetrics:
        """데이터셋 품질 평가"""
        logger.info("Assessing dataset quality...")

        # 1. 완성도 평가
        completeness = self._assess_completeness(dataset)

        # 2. 일관성 평가
        consistency = self._assess_consistency(dataset)

        # 3. 정확도 평가
        accuracy = self._assess_accuracy(dataset)

        # 4. 시의성 평가
        timeliness = self._assess_timeliness(dataset)

        # 5. 전체 점수 계산 (가중평균)
        weights = {"completeness": 0.3, "consistency": 0.3, "accuracy": 0.3, "timeliness": 0.1}
        overall_score = (
            completeness * weights["completeness"] +
            consistency * weights["consistency"] +
            accuracy * weights["accuracy"] +
            timeliness * weights["timeliness"]
        )

        # 6. 문제 식별
        issues = self._identify_issues(
            completeness, consistency, accuracy, timeliness
        )

        return DataQualityMetrics(
            completeness=completeness,
            consistency=consistency,
            accuracy=accuracy,
            timeliness=timeliness,
            overall_score=overall_score,
            issues=issues
        )

    def _assess_completeness(self, dataset: Dict) -> float:
        """완성도 평가: 누락 데이터 비율"""
        if not dataset:
            return 0.0

        total_fields = 0
        missing_fields = 0

        for record in dataset.get("records", []):
            for field, value in record.items():
                total_fields += 1
                if value is None or value == "" or value == []:
                    missing_fields += 1

        if total_fields == 0:
            return 0.0

        completeness = 1 - (missing_fields / total_fields)
        return max(0.0, min(1.0, completeness))

    def _assess_consistency(self, dataset: Dict) -> float:
        """일관성 평가: 데이터 형식 및 범위 일관성"""
        if not dataset.get("records"):
            return 0.0

        records = dataset["records"]
        if len(records) < 2:
            return 1.0  # 레코드가 하나뿐이면 일관성 문제 없음

        # 필드별 데이터 타입 일관성 검사
        field_types = {}
        inconsistent_fields = 0
        total_field_checks = 0

        for record in records:
            for field, value in record.items():
                value_type = type(value).__name__
                total_field_checks += 1

                if field not in field_types:
                    field_types[field] = value_type
                elif field_types[field] != value_type and value is not None:
                    inconsistent_fields += 1

        if total_field_checks == 0:
            return 1.0

        consistency = 1 - (inconsistent_fields / total_field_checks)
        return max(0.0, min(1.0, consistency))

    def _assess_accuracy(self, dataset: Dict) -> float:
        """정확도 평가: 데이터 값의 유효성"""
        if not dataset.get("records"):
            return 0.0

        records = dataset["records"]
        total_values = 0
        invalid_values = 0

        for record in records:
            for field, value in record.items():
                total_values += 1

                # 기본적인 유효성 검사
                if self._is_invalid_value(field, value):
                    invalid_values += 1

        if total_values == 0:
            return 1.0

        accuracy = 1 - (invalid_values / total_values)
        return max(0.0, min(1.0, accuracy))

    def _assess_timeliness(self, dataset: Dict) -> float:
        """시의성 평가: 데이터의 최신성"""
        # 간단한 구현: 메타데이터의 타임스탬프 확인
        if "metadata" in dataset and "last_updated" in dataset["metadata"]:
            # 실제로는 현재 시간과 비교하여 계산
            return 0.85  # 임시 값
        return 0.75  # 기본값

    def _is_invalid_value(self, field: str, value: Any) -> bool:
        """값의 유효성 검사"""
        if value is None:
            return False  # None은 누락 데이터로 별도 처리

        # 나이 필드 검사
        if "age" in field.lower():
            if isinstance(value, (int, float)):
                return value < 0 or value > 150  # 비현실적인 나이

        # 점수나 비율 필드 검사
        if any(keyword in field.lower() for keyword in ["score", "ratio", "percentage"]):
            if isinstance(value, (int, float)):
                return value < 0 or value > 1

        # 빈 문자열이나 공백만 있는 문자열
        if isinstance(value, str):
            return len(value.strip()) == 0

        # 빈 리스트
        if isinstance(value, list):
            return len(value) == 0

        return False

    def _identify_issues(self, completeness: float, consistency: float,
                        accuracy: float, timeliness: float) -> List[str]:
        """품질 문제 식별"""
        issues = []

        if completeness < self.quality_thresholds["completeness"]:
            issues.append(f"Low completeness: {completeness:.2%} < {self.quality_thresholds['completeness']:.2%}")

        if consistency < self.quality_thresholds["consistency"]:
            issues.append(f"Low consistency: {consistency:.2%} < {self.quality_thresholds['consistency']:.2%}")

        if accuracy < self.quality_thresholds["accuracy"]:
            issues.append(f"Low accuracy: {accuracy:.2%} < {self.quality_thresholds['accuracy']:.2%}")

        if timeliness < self.quality_thresholds["timeliness"]:
            issues.append(f"Low timeliness: {timeliness:.2%} < {self.quality_thresholds['timeliness']:.2%}")

        return issues

    async def calculate_data_shapley(self, training_data: List[Dict],
                                   model_performance: float) -> Dict[str, float]:
        """In-Run Data Shapley 기여도 계산 (2025 연구)"""
        logger.info("Calculating Data Shapley contributions...")

        # 간단한 Shapley 값 근사 (실제로는 더 복잡한 알고리즘 사용)
        contributions = {}

        for i, data_point in enumerate(training_data):
            # 각 데이터 포인트의 기여도 계산
            # 실제로는 모델을 다시 훈련하여 성능 변화 측정
            contribution = self._approximate_shapley_value(data_point, model_performance)
            contributions[f"data_point_{i}"] = contribution

        return contributions

    def _approximate_shapley_value(self, data_point: Dict, baseline_performance: float) -> float:
        """Shapley 값 근사 계산"""
        # 임시 구현: 실제로는 복잡한 순열 기반 계산
        # 데이터 품질이 좋을수록 높은 기여도
        quality_score = self._estimate_single_point_quality(data_point)
        return quality_score * 0.1  # 정규화된 기여도

    def _estimate_single_point_quality(self, data_point: Dict) -> float:
        """단일 데이터 포인트 품질 추정"""
        # 간단한 품질 추정: 완성도와 일치성
        total_fields = len(data_point)
        complete_fields = sum(1 for v in data_point.values() if v is not None and v != "")

        if total_fields == 0:
            return 0.0

        return complete_fields / total_fields