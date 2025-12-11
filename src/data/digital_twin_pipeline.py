#!/usr/bin/env python3
"""
Digital Twin Brain Data Pipeline Implementation
2025 Data-Centric AI: 발달장애 Digital Twin Brain 시스템

Features:
- 20-year longitudinal data processing (3,000+ pediatric cohort)
- Multimodal alignment (fMRI, dMRI, EEG, genetics, behavior)
- In-Run Data Shapley quality assessment
- Federated learning support for multi-site data
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import json
import numpy as np
from datetime import datetime, timedelta
import pandas as pd

# Project imports
from ..services.rag.data_quality_assessor import DataQualityAssessor, DataQualityMetrics

logger = logging.getLogger(__name__)

@dataclass
class PatientTimestamp:
    """환자 시간점 데이터"""
    patient_id: str
    age_months: int
    visit_date: datetime
    data_types: List[str]  # ["fMRI", "dMRI", "EEG", "genetics", "behavioral"]

@dataclass
class PatientTrajectory:
    """환자 발달 궤적"""
    patient_id: str
    timestamps: List[PatientTimestamp]
    trajectory_vector: np.ndarray  # 시간에 따른 발달 벡터
    diagnosis_progression: List[str]  # 진단 변화 추적
    biomarkers: Dict[str, List[float]]  # 시간별 바이오마커
    quality_score: float
    completeness_score: float

@dataclass
class MultimodalDataPoint:
    """다중 모달 데이터 포인트"""
    patient_id: str
    timestamp: PatientTimestamp
    fmri_data: Optional[Dict] = None
    dmri_data: Optional[Dict] = None
    eeg_data: Optional[Dict] = None
    genetic_data: Optional[Dict] = None
    behavioral_data: Optional[Dict] = None
    aligned_features: Optional[np.ndarray] = None

class DataIntegrityError(Exception):
    """데이터 무결성 오류"""
    pass

class DigitalTwinDataPipeline:
    """Digital Twin Brain 데이터 처리 파이프라인"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.quality_assessor = DataQualityAssessor()

        # 데이터 저장소
        self.data_storage = {}
        self.processed_trajectories = {}
        self.quality_reports = {}

        # 다중 모달 정렬 파라미터
        self.alignment_params = {
            "temporal_window_hours": 24,  # 24시간 내 데이터는 동일 시점으로 처리
            "missing_data_threshold": 0.3,  # 30% 이상 누락 시 해당 시점 제외
            "quality_threshold": 0.8
        }

    def _default_config(self) -> Dict:
        """기본 설정"""
        return {
            "data_root": "./data/digital_twin",
            "processed_data_path": "./processed/trajectories",
            "quality_reports_path": "./reports/quality",
            "batch_size": 100,
            "max_trajectory_length": 240,  # 20년 = 240개월
            "min_trajectory_points": 3,  # 최소 3회 방문
            "federated_sites": ["seoul_national", "yonsei", "samsung", "ajou", "konkuk"],
            "data_retention_months": 300  # 25년 보관
        }

    async def initialize_pipeline(self):
        """파이프라인 초기화"""
        logger.info("Initializing Digital Twin Brain Data Pipeline...")

        # 디렉토리 생성
        for path_key in ["processed_data_path", "quality_reports_path"]:
            Path(self.config[path_key]).mkdir(parents=True, exist_ok=True)

        # 연합학습 사이트별 설정
        for site in self.config["federated_sites"]:
            site_path = Path(self.config["data_root"]) / site
            site_path.mkdir(parents=True, exist_ok=True)

        logger.info("Digital Twin Pipeline initialized successfully")

    async def process_patient_trajectory(self, patient_id: str) -> PatientTrajectory:
        """환자의 발달 궤적 처리 (20년 종단 데이터)"""
        logger.info(f"Processing trajectory for patient {patient_id}")

        try:
            # 1. 환자 원시 데이터 로드
            raw_data = await self._load_patient_data(patient_id)

            # 2. 데이터 품질 검증
            quality_metrics = await self.quality_assessor.assess_dataset(raw_data)
            if quality_metrics.overall_score < self.alignment_params["quality_threshold"]:
                raise DataIntegrityError(
                    f"Patient {patient_id} data quality too low: {quality_metrics.overall_score:.3f}"
                )

            # 3. 시간점별 데이터 정리
            timestamps = await self._organize_temporal_data(raw_data)

            # 4. 다중 모달 정렬
            aligned_data = await self._align_multimodal_data(timestamps)

            # 5. 종단 궤적 구성
            trajectory = await self._build_longitudinal_trajectory(aligned_data, quality_metrics)

            # 6. 바이오마커 추출
            biomarkers = await self._extract_developmental_biomarkers(trajectory)

            # 7. 결과 저장
            await self._save_trajectory(patient_id, trajectory)

            return trajectory

        except Exception as e:
            logger.error(f"Failed to process patient {patient_id}: {e}")
            raise

    async def process_cohort_batch(self, patient_ids: List[str]) -> Dict[str, PatientTrajectory]:
        """코호트 배치 처리 (3,000명 대규모 처리)"""
        logger.info(f"Processing cohort batch of {len(patient_ids)} patients")

        # 배치별 처리 (메모리 관리)
        batch_size = self.config["batch_size"]
        results = {}

        for i in range(0, len(patient_ids), batch_size):
            batch = patient_ids[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(patient_ids)-1)//batch_size + 1}")

            # 배치 내 병렬 처리
            batch_tasks = [
                self.process_patient_trajectory(patient_id)
                for patient_id in batch
            ]

            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # 결과 정리
            for patient_id, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to process {patient_id}: {result}")
                else:
                    results[patient_id] = result

            # 메모리 정리
            await asyncio.sleep(0.1)  # GC 시간 허용

        logger.info(f"Batch processing completed: {len(results)} successful")
        return results

    async def _load_patient_data(self, patient_id: str) -> Dict:
        """환자 원시 데이터 로드"""
        # Mock 데이터 생성 (실제로는 데이터베이스/파일에서 로드)
        mock_data = {
            "patient_id": patient_id,
            "records": [],
            "metadata": {
                "last_updated": datetime.now().isoformat(),
                "data_sources": ["fMRI", "dMRI", "EEG", "genetics"],
                "total_visits": np.random.randint(5, 25)
            }
        }

        # 시간점별 데이터 생성 (20년 종단)
        num_visits = mock_data["metadata"]["total_visits"]
        start_age_months = np.random.randint(6, 24)  # 6-24개월 시작

        for visit_num in range(num_visits):
            age_months = start_age_months + visit_num * np.random.randint(3, 12)  # 3-12개월 간격
            if age_months > 240:  # 20세 초과 시 중단
                break

            visit_data = {
                "visit_id": f"{patient_id}_visit_{visit_num}",
                "age_months": age_months,
                "visit_date": datetime.now() - timedelta(days=(240-age_months)*30),
                "fmri_data": np.random.rand(100).tolist() if np.random.rand() > 0.1 else None,
                "dmri_data": np.random.rand(50).tolist() if np.random.rand() > 0.15 else None,
                "eeg_data": np.random.rand(200).tolist() if np.random.rand() > 0.2 else None,
                "genetic_data": {
                    "snp_variants": np.random.randint(0, 2, 100).tolist(),
                    "expression_levels": np.random.rand(50).tolist()
                } if visit_num == 0 else None,  # 유전자 데이터는 첫 방문에만
                "behavioral_scores": {
                    "ados_score": np.random.randint(1, 20),
                    "vineland_score": np.random.randint(20, 120),
                    "iq_score": np.random.randint(70, 130)
                },
                "diagnosis": self._generate_mock_diagnosis(age_months)
            }

            mock_data["records"].append(visit_data)

        return mock_data

    def _generate_mock_diagnosis(self, age_months: int) -> str:
        """나이에 따른 모의 진단 생성"""
        if age_months < 18:
            return "under_observation"
        elif age_months < 36:
            return np.random.choice(["TD", "ASD_suspected", "delayed"])
        else:
            return np.random.choice(["TD", "ASD", "ADHD", "other_DD"], p=[0.7, 0.15, 0.1, 0.05])

    async def _organize_temporal_data(self, raw_data: Dict) -> List[PatientTimestamp]:
        """시간점별 데이터 정리"""
        timestamps = []

        for record in raw_data.get("records", []):
            # 사용 가능한 데이터 타입 확인
            data_types = []
            for data_type in ["fmri_data", "dmri_data", "eeg_data", "genetic_data", "behavioral_scores"]:
                if record.get(data_type) is not None:
                    data_types.append(data_type.replace("_data", "").replace("_scores", ""))

            timestamp = PatientTimestamp(
                patient_id=raw_data["patient_id"],
                age_months=record["age_months"],
                visit_date=record["visit_date"] if isinstance(record["visit_date"], datetime)
                          else datetime.fromisoformat(record["visit_date"]),
                data_types=data_types
            )

            timestamps.append(timestamp)

        # 나이순으로 정렬
        timestamps.sort(key=lambda x: x.age_months)
        return timestamps

    async def _align_multimodal_data(self, timestamps: List[PatientTimestamp]) -> List[MultimodalDataPoint]:
        """다중 모달 데이터 정렬"""
        aligned_data = []

        for timestamp in timestamps:
            # 각 시간점에서 다중 모달 데이터 통합
            data_point = MultimodalDataPoint(
                patient_id=timestamp.patient_id,
                timestamp=timestamp
            )

            # 모달리티별 데이터 로드 (실제로는 원시 데이터에서)
            if "fmri" in timestamp.data_types:
                data_point.fmri_data = await self._load_fmri_timepoint(
                    timestamp.patient_id, timestamp.age_months
                )

            if "dmri" in timestamp.data_types:
                data_point.dmri_data = await self._load_dmri_timepoint(
                    timestamp.patient_id, timestamp.age_months
                )

            if "eeg" in timestamp.data_types:
                data_point.eeg_data = await self._load_eeg_timepoint(
                    timestamp.patient_id, timestamp.age_months
                )

            if "genetic" in timestamp.data_types:
                data_point.genetic_data = await self._load_genetic_timepoint(
                    timestamp.patient_id
                )

            if "behavioral" in timestamp.data_types:
                data_point.behavioral_data = await self._load_behavioral_timepoint(
                    timestamp.patient_id, timestamp.age_months
                )

            # 정렬된 특징 벡터 생성
            data_point.aligned_features = await self._create_aligned_features(data_point)

            aligned_data.append(data_point)

        return aligned_data

    async def _build_longitudinal_trajectory(self, aligned_data: List[MultimodalDataPoint],
                                          quality_metrics: DataQualityMetrics) -> PatientTrajectory:
        """종단 궤적 구성"""
        if not aligned_data:
            raise DataIntegrityError("No aligned data available for trajectory")

        patient_id = aligned_data[0].patient_id

        # 궤적 벡터 생성 (시간에 따른 발달 패턴)
        trajectory_vectors = []
        diagnosis_progression = []
        biomarkers = {"connectivity": [], "volume": [], "behavior": []}

        for data_point in aligned_data:
            if data_point.aligned_features is not None:
                trajectory_vectors.append(data_point.aligned_features)

                # 진단 진행 추적
                diagnosis = self._extract_diagnosis_from_behavioral(data_point.behavioral_data)
                diagnosis_progression.append(diagnosis)

                # 바이오마커 추출
                if data_point.fmri_data:
                    biomarkers["connectivity"].append(
                        np.mean(data_point.fmri_data.get("connectivity_matrix", [0.5]))
                    )
                if data_point.dmri_data:
                    biomarkers["volume"].append(
                        np.sum(data_point.dmri_data.get("volume_measures", [1.0]))
                    )
                if data_point.behavioral_data:
                    biomarkers["behavior"].append(
                        data_point.behavioral_data.get("composite_score", 100)
                    )

        # 궤적 벡터 결합
        if trajectory_vectors:
            trajectory_matrix = np.array(trajectory_vectors)
            # PCA 또는 다른 차원 축소 기법 적용
            trajectory_vector = np.mean(trajectory_matrix, axis=0)
        else:
            trajectory_vector = np.zeros(100)  # 기본 크기

        return PatientTrajectory(
            patient_id=patient_id,
            timestamps=[dp.timestamp for dp in aligned_data],
            trajectory_vector=trajectory_vector,
            diagnosis_progression=diagnosis_progression,
            biomarkers=biomarkers,
            quality_score=quality_metrics.overall_score,
            completeness_score=quality_metrics.completeness
        )

    async def _extract_developmental_biomarkers(self, trajectory: PatientTrajectory) -> Dict[str, Any]:
        """발달 바이오마커 추출"""
        biomarkers = {
            "trajectory_slope": self._calculate_trajectory_slope(trajectory),
            "critical_periods": self._identify_critical_periods(trajectory),
            "deviation_score": self._calculate_deviation_from_normative(trajectory),
            "prediction_markers": self._extract_prediction_markers(trajectory)
        }

        return biomarkers

    def _calculate_trajectory_slope(self, trajectory: PatientTrajectory) -> float:
        """궤적 기울기 계산 (발달 속도)"""
        if len(trajectory.timestamps) < 2:
            return 0.0

        ages = [ts.age_months for ts in trajectory.timestamps]
        # 간단한 선형 회귀로 기울기 계산
        if len(trajectory.biomarkers.get("behavior", [])) >= 2:
            behavior_scores = trajectory.biomarkers["behavior"]
            if len(behavior_scores) == len(ages):
                # numpy polyfit으로 기울기 계산
                slope, _ = np.polyfit(ages, behavior_scores, 1)
                return float(slope)

        return 0.0

    def _identify_critical_periods(self, trajectory: PatientTrajectory) -> List[int]:
        """중요 발달 시기 식별"""
        critical_periods = []

        for i, timestamp in enumerate(trajectory.timestamps):
            age_months = timestamp.age_months

            # 알려진 중요 발달 시기
            if 18 <= age_months <= 24:  # 언어 발달 임계기
                critical_periods.append(age_months)
            elif 36 <= age_months <= 48:  # 사회성 발달 임계기
                critical_periods.append(age_months)
            elif 60 <= age_months <= 84:  # 인지 발달 임계기
                critical_periods.append(age_months)

        return critical_periods

    def _calculate_deviation_from_normative(self, trajectory: PatientTrajectory) -> float:
        """정상 발달에서의 편차 계산"""
        # 정상 발달 곡선과의 차이 계산 (간단한 구현)
        if not trajectory.biomarkers.get("behavior"):
            return 0.0

        behavior_scores = trajectory.biomarkers["behavior"]
        ages = [ts.age_months for ts in trajectory.timestamps]

        # 정상 발달 곡선 (간단한 모델)
        expected_scores = [100 + age * 0.1 for age in ages]  # 나이에 따라 조금씩 증가

        if len(behavior_scores) == len(expected_scores):
            deviations = [abs(actual - expected)
                         for actual, expected in zip(behavior_scores, expected_scores)]
            return float(np.mean(deviations))

        return 0.0

    def _extract_prediction_markers(self, trajectory: PatientTrajectory) -> Dict[str, float]:
        """예측 마커 추출"""
        markers = {
            "early_connectivity": 0.0,
            "volume_trajectory": 0.0,
            "behavioral_consistency": 0.0
        }

        # 초기 연결성 패턴
        if trajectory.biomarkers.get("connectivity"):
            connectivity_values = trajectory.biomarkers["connectivity"]
            if connectivity_values:
                markers["early_connectivity"] = float(connectivity_values[0])

        # 체적 변화 궤적
        if trajectory.biomarkers.get("volume"):
            volume_values = trajectory.biomarkers["volume"]
            if len(volume_values) >= 2:
                volume_change = volume_values[-1] - volume_values[0]
                markers["volume_trajectory"] = float(volume_change)

        # 행동 점수 일관성
        if trajectory.biomarkers.get("behavior"):
            behavior_values = trajectory.biomarkers["behavior"]
            if len(behavior_values) >= 2:
                markers["behavioral_consistency"] = float(1.0 - np.std(behavior_values) / 100.0)

        return markers

    async def _load_fmri_timepoint(self, patient_id: str, age_months: int) -> Dict:
        """fMRI 시간점 데이터 로드"""
        # Mock fMRI 데이터
        return {
            "connectivity_matrix": np.random.rand(10, 10).tolist(),
            "roi_activations": np.random.rand(50).tolist(),
            "network_measures": {
                "default_mode": np.random.rand(),
                "executive": np.random.rand(),
                "salience": np.random.rand()
            }
        }

    async def _load_dmri_timepoint(self, patient_id: str, age_months: int) -> Dict:
        """dMRI 시간점 데이터 로드"""
        # Mock dMRI 데이터
        return {
            "fa_values": np.random.rand(20).tolist(),
            "md_values": np.random.rand(20).tolist(),
            "volume_measures": np.random.rand(10).tolist(),
            "tract_integrity": np.random.rand()
        }

    async def _load_eeg_timepoint(self, patient_id: str, age_months: int) -> Dict:
        """EEG 시간점 데이터 로드"""
        # Mock EEG 데이터
        return {
            "power_spectrum": np.random.rand(50).tolist(),
            "coherence_measures": np.random.rand(20).tolist(),
            "event_related_potentials": np.random.rand(30).tolist()
        }

    async def _load_genetic_timepoint(self, patient_id: str) -> Dict:
        """유전자 데이터 로드 (시간 불변)"""
        # Mock 유전자 데이터
        return {
            "risk_snps": np.random.randint(0, 2, 100).tolist(),
            "polygenic_risk_score": np.random.rand(),
            "expression_profile": np.random.rand(50).tolist()
        }

    async def _load_behavioral_timepoint(self, patient_id: str, age_months: int) -> Dict:
        """행동 평가 시간점 데이터 로드"""
        # Mock 행동 데이터
        return {
            "ados_score": np.random.randint(1, 20),
            "vineland_adaptive": np.random.randint(20, 120),
            "iq_score": np.random.randint(70, 130),
            "composite_score": np.random.randint(80, 120)
        }

    async def _create_aligned_features(self, data_point: MultimodalDataPoint) -> np.ndarray:
        """정렬된 특징 벡터 생성"""
        features = []

        # fMRI 특징 (20차원)
        if data_point.fmri_data:
            fmri_features = data_point.fmri_data.get("roi_activations", [0] * 20)[:20]
            features.extend(fmri_features)
        else:
            features.extend([0.0] * 20)

        # dMRI 특징 (20차원)
        if data_point.dmri_data:
            dmri_features = data_point.dmri_data.get("fa_values", [0] * 20)[:20]
            features.extend(dmri_features)
        else:
            features.extend([0.0] * 20)

        # EEG 특징 (30차원)
        if data_point.eeg_data:
            eeg_features = data_point.eeg_data.get("power_spectrum", [0] * 30)[:30]
            features.extend(eeg_features)
        else:
            features.extend([0.0] * 30)

        # 유전자 특징 (20차원)
        if data_point.genetic_data:
            genetic_features = data_point.genetic_data.get("expression_profile", [0] * 20)[:20]
            features.extend(genetic_features)
        else:
            features.extend([0.0] * 20)

        # 행동 특징 (10차원)
        if data_point.behavioral_data:
            behavioral_features = [
                data_point.behavioral_data.get("ados_score", 0) / 20.0,  # 정규화
                data_point.behavioral_data.get("vineland_adaptive", 0) / 120.0,
                data_point.behavioral_data.get("iq_score", 0) / 130.0,
                data_point.behavioral_data.get("composite_score", 0) / 120.0
            ]
            behavioral_features.extend([0.0] * 6)  # 10차원 맞추기
            features.extend(behavioral_features)
        else:
            features.extend([0.0] * 10)

        return np.array(features)

    def _extract_diagnosis_from_behavioral(self, behavioral_data: Optional[Dict]) -> str:
        """행동 데이터에서 진단 추출"""
        if not behavioral_data:
            return "unknown"

        ados_score = behavioral_data.get("ados_score", 10)

        if ados_score >= 15:
            return "ASD"
        elif ados_score >= 10:
            return "ASD_suspected"
        else:
            return "TD"

    async def _save_trajectory(self, patient_id: str, trajectory: PatientTrajectory):
        """궤적 데이터 저장"""
        output_path = Path(self.config["processed_data_path"]) / f"{patient_id}_trajectory.json"

        # 궤적 데이터를 JSON 직렬화 가능한 형태로 변환
        trajectory_dict = asdict(trajectory)

        # numpy 배열 변환
        trajectory_dict["trajectory_vector"] = trajectory.trajectory_vector.tolist()

        # datetime 객체 변환
        for i, ts in enumerate(trajectory_dict["timestamps"]):
            ts["visit_date"] = ts["visit_date"].isoformat()

        with open(output_path, 'w') as f:
            json.dump(trajectory_dict, f, indent=2)

        logger.info(f"Trajectory saved: {output_path}")


# Factory function
async def create_digital_twin_pipeline(config: Optional[Dict] = None) -> DigitalTwinDataPipeline:
    """Digital Twin 파이프라인 생성 및 초기화"""
    pipeline = DigitalTwinDataPipeline(config)
    await pipeline.initialize_pipeline()
    return pipeline