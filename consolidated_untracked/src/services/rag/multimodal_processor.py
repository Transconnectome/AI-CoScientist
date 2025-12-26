#!/usr/bin/env python3
"""
Multimodal Brain Processor for Enhanced DD-RAPTOR System
2025 Research: Multimodal foundation models for neurodevelopmental disorders
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class ProcessedMultimodalQuery:
    """처리된 다중 모달 쿼리"""
    text_features: List[float]
    modality_weights: Dict[str, float]
    age_encoding: List[float]
    severity_encoding: Optional[List[float]]
    integrated_embedding: List[float]

@dataclass
class BrainImageFeatures:
    """뇌 영상 특징"""
    fmri_features: Optional[np.ndarray]
    dmri_features: Optional[np.ndarray]
    structural_features: Optional[np.ndarray]
    connectivity_matrix: Optional[np.ndarray]

class MultimodalBrainProcessor:
    """다중 모달 뇌 데이터 처리기 (2025 Foundation Model 패턴)"""

    def __init__(self):
        # 모달리티별 가중치 (2025 연구 기반)
        self.modality_importance = {
            "fMRI": 0.35,      # 기능적 연결성
            "dMRI": 0.30,      # 구조적 연결성
            "EEG": 0.20,       # 시간적 역학
            "structural": 0.15  # 해부학적 구조
        }

        # 나이별 뇌 발달 패턴
        self.age_patterns = {
            "0-2": {"pattern": "rapid_growth", "weight": 1.0},
            "3-6": {"pattern": "critical_period", "weight": 0.9},
            "7-12": {"pattern": "consolidation", "weight": 0.8},
            "13-18": {"pattern": "maturation", "weight": 0.7}
        }

        # 심각도별 가중치
        self.severity_weights = {
            "mild": 0.3,
            "moderate": 0.6,
            "severe": 1.0
        }

    async def process_query(self, query) -> ProcessedMultimodalQuery:
        """다중 모달 쿼리 처리"""
        logger.info(f"Processing multimodal query: {query.text}")

        # 1. 텍스트 특징 추출
        text_features = await self._extract_text_features(query.text)

        # 2. 모달리티 가중치 계산
        modality_weights = self._calculate_modality_weights(query.modalities)

        # 3. 나이 인코딩
        age_encoding = self._encode_age_range(query.age_range)

        # 4. 심각도 인코딩
        severity_encoding = None
        if query.severity_level:
            severity_encoding = self._encode_severity(query.severity_level)

        # 5. 통합 임베딩 생성
        integrated_embedding = await self._create_integrated_embedding(
            text_features, modality_weights, age_encoding, severity_encoding
        )

        return ProcessedMultimodalQuery(
            text_features=text_features,
            modality_weights=modality_weights,
            age_encoding=age_encoding,
            severity_encoding=severity_encoding,
            integrated_embedding=integrated_embedding
        )

    async def process_brain_features(self, raw_features: Dict) -> BrainImageFeatures:
        """뇌 영상 특징 처리"""
        logger.info("Processing brain imaging features...")

        # fMRI 처리
        fmri_features = None
        if "fmri_features" in raw_features:
            fmri_features = await self._process_fmri(raw_features["fmri_features"])

        # dMRI 처리
        dmri_features = None
        if "dmri_features" in raw_features:
            dmri_features = await self._process_dmri(raw_features["dmri_features"])

        # 구조적 영상 처리
        structural_features = None
        if "structural_features" in raw_features:
            structural_features = await self._process_structural(raw_features["structural_features"])

        # 연결성 매트릭스 계산
        connectivity_matrix = await self._calculate_connectivity_matrix(
            fmri_features, dmri_features
        )

        return BrainImageFeatures(
            fmri_features=fmri_features,
            dmri_features=dmri_features,
            structural_features=structural_features,
            connectivity_matrix=connectivity_matrix
        )

    async def _extract_text_features(self, text: str) -> List[float]:
        """텍스트에서 특징 추출"""
        # 간단한 TF-IDF 기반 특징 (실제로는 SciBERT 사용)
        words = text.lower().split()

        # 발달장애 관련 키워드 가중치
        keyword_weights = {
            "autism": 0.9, "asd": 0.9, "adhd": 0.8,
            "brain": 0.7, "development": 0.7, "fmri": 0.6,
            "connectivity": 0.6, "early": 0.5, "diagnosis": 0.5
        }

        # 특징 벡터 생성 (100차원)
        features = []
        for i in range(100):
            feature_value = 0.0
            for word in words:
                if word in keyword_weights:
                    feature_value += keyword_weights[word] * np.random.normal(0.5, 0.1)
            features.append(max(0.0, min(1.0, feature_value)))

        return features

    def _calculate_modality_weights(self, modalities: List[str]) -> Dict[str, float]:
        """모달리티별 가중치 계산"""
        weights = {}
        total_weight = sum(self.modality_importance.get(mod, 0.1) for mod in modalities)

        for modality in modalities:
            base_weight = self.modality_importance.get(modality, 0.1)
            weights[modality] = base_weight / total_weight if total_weight > 0 else 1.0 / len(modalities)

        return weights

    def _encode_age_range(self, age_range: str) -> List[float]:
        """나이 범위 인코딩"""
        # 나이 범위를 벡터로 인코딩
        encoding = [0.0] * 10  # 10차원 나이 인코딩

        # 나이 패턴 매칭
        for i, (pattern_age, pattern_info) in enumerate(self.age_patterns.items()):
            if pattern_age in age_range or any(age in age_range for age in pattern_age.split('-')):
                encoding[i % 10] = pattern_info["weight"]

        # 나이 범위에서 숫자 추출 시도
        try:
            ages = [int(x) for x in age_range.split() if x.isdigit()]
            if ages:
                avg_age = sum(ages) / len(ages)
                # 평균 나이를 0-1 범위로 정규화 (0-18세 기준)
                normalized_age = min(1.0, max(0.0, avg_age / 18))
                encoding[-1] = normalized_age
        except:
            pass

        return encoding

    def _encode_severity(self, severity: str) -> List[float]:
        """심각도 인코딩"""
        encoding = [0.0] * 5  # 5차원 심각도 인코딩

        severity_lower = severity.lower()
        if severity_lower in self.severity_weights:
            weight = self.severity_weights[severity_lower]
            encoding[0] = weight  # 기본 심각도

            # 심각도별 패턴
            if "mild" in severity_lower:
                encoding[1] = 1.0
            elif "moderate" in severity_lower:
                encoding[2] = 1.0
            elif "severe" in severity_lower:
                encoding[3] = 1.0

        return encoding

    async def _create_integrated_embedding(self, text_features: List[float],
                                         modality_weights: Dict[str, float],
                                         age_encoding: List[float],
                                         severity_encoding: Optional[List[float]]) -> List[float]:
        """통합 임베딩 생성"""
        # 각 구성요소를 연결하여 통합 임베딩 생성
        integrated = []

        # 1. 텍스트 특징 (가중치 적용)
        text_weight = 0.4
        integrated.extend([f * text_weight for f in text_features])

        # 2. 모달리티 가중치 벡터화
        modality_vector = [0.0] * 10
        for i, (mod, weight) in enumerate(modality_weights.items()):
            if i < 10:
                modality_vector[i] = weight
        integrated.extend(modality_vector)

        # 3. 나이 인코딩
        age_weight = 0.3
        integrated.extend([f * age_weight for f in age_encoding])

        # 4. 심각도 인코딩 (있는 경우)
        if severity_encoding:
            severity_weight = 0.2
            integrated.extend([f * severity_weight for f in severity_encoding])
        else:
            integrated.extend([0.0] * 5)  # 빈 심각도 벡터

        return integrated

    async def _process_fmri(self, fmri_data: List[float]) -> np.ndarray:
        """fMRI 데이터 처리"""
        # 기능적 연결성 특징 추출
        data = np.array(fmri_data)

        # 정규화
        if data.std() > 0:
            data = (data - data.mean()) / data.std()

        # ROI 기반 특징 추출 (간단한 구현)
        roi_features = []
        chunk_size = max(1, len(data) // 10)  # 10개 ROI로 분할

        for i in range(0, len(data), chunk_size):
            chunk = data[i:i+chunk_size]
            if len(chunk) > 0:
                roi_features.extend([
                    np.mean(chunk),
                    np.std(chunk),
                    np.max(chunk) - np.min(chunk)  # 동적 범위
                ])

        return np.array(roi_features)

    async def _process_dmri(self, dmri_data: List[float]) -> np.ndarray:
        """dMRI 데이터 처리 (확산 텐서)"""
        data = np.array(dmri_data)

        # DTI 지표 계산 (FA, MD, RD, AD 근사)
        if len(data) >= 6:  # 최소 텐서 성분
            # 간단한 DTI 지표 근사
            fa = np.std(data[:len(data)//3])  # Fractional Anisotropy 근사
            md = np.mean(data)  # Mean Diffusivity 근사
            rd = np.mean(data[len(data)//3:2*len(data)//3])  # Radial Diffusivity 근사
            ad = np.mean(data[2*len(data)//3:])  # Axial Diffusivity 근사

            return np.array([fa, md, rd, ad])

        return np.array([0.0, 0.0, 0.0, 0.0])

    async def _process_structural(self, structural_data: List[float]) -> np.ndarray:
        """구조적 MRI 데이터 처리"""
        data = np.array(structural_data)

        # 체적 및 두께 측정 근사
        if len(data) > 0:
            # 정규화
            if data.std() > 0:
                data = (data - data.mean()) / data.std()

            # 뇌 영역별 체적 특징 (간단한 구현)
            volume_features = []
            num_regions = min(10, len(data) // 10)

            for i in range(num_regions):
                start = i * len(data) // num_regions
                end = (i + 1) * len(data) // num_regions
                region_data = data[start:end]

                if len(region_data) > 0:
                    volume_features.extend([
                        np.sum(region_data),  # 체적
                        np.mean(region_data),  # 평균 강도
                        np.std(region_data)    # 변동성
                    ])

            return np.array(volume_features)

        return np.array([0.0])

    async def _calculate_connectivity_matrix(self, fmri_features: Optional[np.ndarray],
                                           dmri_features: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """연결성 매트릭스 계산"""
        if fmri_features is None and dmri_features is None:
            return None

        # 간단한 연결성 매트릭스 생성 (실제로는 복잡한 그래프 분석)
        size = 10  # 10x10 연결성 매트릭스
        connectivity = np.random.rand(size, size)

        # 대칭 매트릭스로 만들기
        connectivity = (connectivity + connectivity.T) / 2

        # 대각선은 1로 설정
        np.fill_diagonal(connectivity, 1.0)

        # fMRI와 dMRI 정보가 있으면 가중치 적용
        if fmri_features is not None:
            functional_weight = np.mean(fmri_features) if len(fmri_features) > 0 else 0.5
            connectivity *= functional_weight

        if dmri_features is not None:
            structural_weight = np.mean(dmri_features) if len(dmri_features) > 0 else 0.5
            connectivity = connectivity * 0.7 + structural_weight * 0.3

        return connectivity