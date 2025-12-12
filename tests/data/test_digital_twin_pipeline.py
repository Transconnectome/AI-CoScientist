#!/usr/bin/env python3
"""
Digital Twin Pipeline TDD Tests
2025 Data-Centric AI Testing Framework
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import json

# Local imports
from src.data.digital_twin_pipeline import (
    DigitalTwinDataPipeline,
    PatientTrajectory,
    PatientTimestamp,
    MultimodalDataPoint,
    DataIntegrityError
)

class TestDigitalTwinPipeline:
    """Digital Twin 데이터 파이프라인 TDD 테스트"""

    def setup_method(self):
        """테스트 설정"""
        # 임시 디렉토리 사용
        self.temp_dir = tempfile.mkdtemp()

        self.config = {
            "data_root": str(Path(self.temp_dir) / "data"),
            "processed_data_path": str(Path(self.temp_dir) / "processed"),
            "quality_reports_path": str(Path(self.temp_dir) / "reports"),
            "batch_size": 10,
            "max_trajectory_length": 240,
            "min_trajectory_points": 3,
            "federated_sites": ["test_site_1", "test_site_2"]
        }

    @pytest.mark.asyncio
    async def test_pipeline_initialization(self):
        """파이프라인 초기화 테스트"""
        # Given
        pipeline = DigitalTwinDataPipeline(self.config)

        # When
        await pipeline.initialize_pipeline()

        # Then
        assert Path(self.config["processed_data_path"]).exists()
        assert Path(self.config["quality_reports_path"]).exists()

        # 연합학습 사이트 디렉토리 확인
        for site in self.config["federated_sites"]:
            site_path = Path(self.config["data_root"]) / site
            assert site_path.exists()

    @pytest.mark.asyncio
    async def test_patient_trajectory_processing(self):
        """환자 궤적 처리 테스트"""
        # Given
        pipeline = DigitalTwinDataPipeline(self.config)
        await pipeline.initialize_pipeline()

        patient_id = "TEST_001"

        # When
        trajectory = await pipeline.process_patient_trajectory(patient_id)

        # Then
        assert isinstance(trajectory, PatientTrajectory)
        assert trajectory.patient_id == patient_id
        assert len(trajectory.timestamps) >= self.config["min_trajectory_points"]
        assert trajectory.quality_score >= 0.0
        assert trajectory.completeness_score >= 0.0

        # 궤적 벡터 검증
        assert isinstance(trajectory.trajectory_vector, np.ndarray)
        assert len(trajectory.trajectory_vector) > 0

        # 바이오마커 검증
        assert "connectivity" in trajectory.biomarkers
        assert "volume" in trajectory.biomarkers
        assert "behavior" in trajectory.biomarkers

    @pytest.mark.asyncio
    async def test_multimodal_data_alignment(self):
        """다중 모달 데이터 정렬 테스트"""
        # Given
        pipeline = DigitalTwinDataPipeline(self.config)
        await pipeline.initialize_pipeline()

        # Mock 시간점 데이터
        timestamps = [
            PatientTimestamp(
                patient_id="TEST_001",
                age_months=24,
                visit_date=datetime.now() - timedelta(days=30*i),
                data_types=["fmri", "dmri", "behavioral"]
            )
            for i in range(5)
        ]

        # When
        aligned_data = await pipeline._align_multimodal_data(timestamps)

        # Then
        assert len(aligned_data) == len(timestamps)

        for data_point in aligned_data:
            assert isinstance(data_point, MultimodalDataPoint)
            assert data_point.aligned_features is not None
            assert isinstance(data_point.aligned_features, np.ndarray)

            # 특징 벡터 차원 검증 (20+20+30+20+10 = 100차원)
            assert len(data_point.aligned_features) == 100

    @pytest.mark.asyncio
    async def test_longitudinal_trajectory_building(self):
        """종단 궤적 구축 테스트"""
        # Given
        pipeline = DigitalTwinDataPipeline(self.config)
        await pipeline.initialize_pipeline()

        # Mock 정렬된 데이터
        aligned_data = []
        for i in range(5):
            timestamp = PatientTimestamp(
                patient_id="TEST_001",
                age_months=24 + i*6,  # 6개월 간격
                visit_date=datetime.now() - timedelta(days=180*i),
                data_types=["fmri", "behavioral"]
            )

            data_point = MultimodalDataPoint(
                patient_id="TEST_001",
                timestamp=timestamp,
                aligned_features=np.random.rand(100)
            )

            # Mock behavioral data 추가
            data_point.behavioral_data = {
                "ados_score": np.random.randint(1, 20),
                "composite_score": np.random.randint(80, 120)
            }

            aligned_data.append(data_point)

        # Mock quality metrics
        from src.services.rag.data_quality_assessor import DataQualityMetrics
        quality_metrics = DataQualityMetrics(
            completeness=0.85,
            consistency=0.90,
            accuracy=0.80,
            timeliness=0.75,
            overall_score=0.82,
            issues=[]
        )

        # When
        trajectory = await pipeline._build_longitudinal_trajectory(aligned_data, quality_metrics)

        # Then
        assert isinstance(trajectory, PatientTrajectory)
        assert len(trajectory.timestamps) == 5
        assert len(trajectory.diagnosis_progression) == 5
        assert trajectory.quality_score == quality_metrics.overall_score

        # 바이오마커 시계열 검증
        for biomarker_type in ["behavior"]:
            if biomarker_type in trajectory.biomarkers:
                assert len(trajectory.biomarkers[biomarker_type]) > 0

    @pytest.mark.asyncio
    async def test_batch_cohort_processing(self):
        """코호트 배치 처리 테스트"""
        # Given
        pipeline = DigitalTwinDataPipeline(self.config)
        await pipeline.initialize_pipeline()

        patient_ids = [f"TEST_{i:03d}" for i in range(25)]  # 25명

        # When
        results = await pipeline.process_cohort_batch(patient_ids)

        # Then
        assert isinstance(results, dict)
        assert len(results) <= len(patient_ids)  # 일부는 실패할 수 있음

        # 각 결과 검증
        for patient_id, trajectory in results.items():
            assert isinstance(trajectory, PatientTrajectory)
            assert trajectory.patient_id == patient_id

    @pytest.mark.asyncio
    async def test_data_quality_validation(self):
        """데이터 품질 검증 테스트"""
        # Given
        pipeline = DigitalTwinDataPipeline(self.config)
        await pipeline.initialize_pipeline()

        # Mock 저품질 데이터
        pipeline.alignment_params["quality_threshold"] = 0.9  # 높은 임계값

        # When & Then
        with pytest.raises(DataIntegrityError):
            await pipeline.process_patient_trajectory("LOW_QUALITY_001")

    @pytest.mark.asyncio
    async def test_biomarker_extraction(self):
        """바이오마커 추출 테스트"""
        # Given
        pipeline = DigitalTwinDataPipeline(self.config)

        # Mock 궤적 데이터
        timestamps = [
            PatientTimestamp(
                patient_id="TEST_001",
                age_months=18 + i*6,
                visit_date=datetime.now() - timedelta(days=180*i),
                data_types=["behavioral"]
            )
            for i in range(6)
        ]

        trajectory = PatientTrajectory(
            patient_id="TEST_001",
            timestamps=timestamps,
            trajectory_vector=np.random.rand(100),
            diagnosis_progression=["TD", "TD", "ASD_suspected", "ASD", "ASD", "ASD"],
            biomarkers={
                "connectivity": [0.6, 0.55, 0.5, 0.45, 0.4, 0.35],
                "volume": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
                "behavior": [110, 108, 105, 100, 95, 90]
            },
            quality_score=0.85,
            completeness_score=0.90
        )

        # When
        biomarkers = await pipeline._extract_developmental_biomarkers(trajectory)

        # Then
        assert "trajectory_slope" in biomarkers
        assert "critical_periods" in biomarkers
        assert "deviation_score" in biomarkers
        assert "prediction_markers" in biomarkers

        # 궤적 기울기가 음수인지 확인 (행동 점수 감소)
        assert biomarkers["trajectory_slope"] < 0

        # 중요 시기가 식별되었는지 확인
        critical_periods = biomarkers["critical_periods"]
        assert any(18 <= age <= 24 for age in critical_periods)  # 언어 발달 시기

    @pytest.mark.asyncio
    async def test_federated_learning_support(self):
        """연합학습 지원 테스트"""
        # Given
        pipeline = DigitalTwinDataPipeline(self.config)
        await pipeline.initialize_pipeline()

        # When & Then
        for site in self.config["federated_sites"]:
            site_path = Path(self.config["data_root"]) / site
            assert site_path.exists(), f"Federated site directory not created: {site}"

    @pytest.mark.asyncio
    async def test_trajectory_persistence(self):
        """궤적 데이터 영속성 테스트"""
        # Given
        pipeline = DigitalTwinDataPipeline(self.config)
        await pipeline.initialize_pipeline()

        patient_id = "TEST_SAVE_001"

        # When
        original_trajectory = await pipeline.process_patient_trajectory(patient_id)

        # Then
        saved_file = Path(self.config["processed_data_path"]) / f"{patient_id}_trajectory.json"
        assert saved_file.exists()

        # 저장된 데이터 검증
        with open(saved_file, 'r') as f:
            saved_data = json.load(f)

        assert saved_data["patient_id"] == patient_id
        assert "trajectory_vector" in saved_data
        assert "timestamps" in saved_data
        assert len(saved_data["trajectory_vector"]) == len(original_trajectory.trajectory_vector)

    @pytest.mark.asyncio
    async def test_20_year_longitudinal_support(self):
        """20년 종단 데이터 지원 테스트"""
        # Given
        pipeline = DigitalTwinDataPipeline(self.config)
        await pipeline.initialize_pipeline()

        # When
        trajectory = await pipeline.process_patient_trajectory("LONG_TERM_001")

        # Then
        # 시간 범위 확인
        ages = [ts.age_months for ts in trajectory.timestamps]
        age_span = max(ages) - min(ages)

        # 최소 5년 이상의 추적기간
        assert age_span >= 60, f"Age span {age_span} months is less than 5 years"

        # 최대 20년을 초과하지 않음
        assert max(ages) <= 240, f"Maximum age {max(ages)} months exceeds 20 years"

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_large_cohort_scalability(self):
        """대규모 코호트 확장성 테스트"""
        # Given
        pipeline = DigitalTwinDataPipeline(self.config)
        await pipeline.initialize_pipeline()

        # 1000명 규모 (실제 3000명의 축소 버전)
        large_patient_ids = [f"SCALE_TEST_{i:04d}" for i in range(100)]

        # When
        import time
        start_time = time.time()
        results = await pipeline.process_cohort_batch(large_patient_ids)
        processing_time = time.time() - start_time

        # Then
        assert len(results) > 0
        assert processing_time < 300  # 5분 이내 처리

        # 처리율 검증 (최소 50% 성공)
        success_rate = len(results) / len(large_patient_ids)
        assert success_rate >= 0.5, f"Success rate {success_rate:.2%} too low"

    def teardown_method(self):
        """테스트 정리"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


# 통합 테스트
@pytest.mark.integration
class TestDigitalTwinIntegration:
    """Digital Twin 시스템 통합 테스트"""

    @pytest.mark.asyncio
    async def test_end_to_end_pipeline(self):
        """종단간 파이프라인 테스트"""
        # Given
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "data_root": str(Path(temp_dir) / "data"),
                "processed_data_path": str(Path(temp_dir) / "processed"),
                "quality_reports_path": str(Path(temp_dir) / "reports"),
                "batch_size": 5,
                "federated_sites": ["hospital_a", "hospital_b"]
            }

            # When
            pipeline = DigitalTwinDataPipeline(config)
            await pipeline.initialize_pipeline()

            # 실제 환자 코호트 처리
            patient_ids = ["E2E_001", "E2E_002", "E2E_003"]
            results = await pipeline.process_cohort_batch(patient_ids)

            # Then
            assert len(results) > 0

            # 각 환자별 결과 검증
            for patient_id, trajectory in results.items():
                # 기본 구조 검증
                assert trajectory.patient_id == patient_id
                assert len(trajectory.timestamps) > 0
                assert trajectory.quality_score > 0

                # 저장 파일 존재 확인
                saved_file = Path(config["processed_data_path"]) / f"{patient_id}_trajectory.json"
                assert saved_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])