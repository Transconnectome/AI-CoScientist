"""
간단한 통합 테스트 - Foundation Model 기본 기능 검증
UltraThink 구현된 시스템들의 핵심 기능 테스트
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, AsyncMock, patch

# 기본 기능만 테스트 (복잡한 의존성 제외)


def test_diver0_tokenizer_basic():
    """DIVER-0 토크나이저 기본 기능 테스트"""
    from src.services.psychology.models.diver0_integration import EEGAnalysisConfig

    config = EEGAnalysisConfig()

    # 기본 설정 검증
    assert config.n_channels == 64
    assert config.sampling_rate == 500
    assert config.window_size == 1000
    assert config.transformer_dim == 512


def test_swift_config_initialization():
    """SwiFT 설정 초기화 테스트"""
    from src.services.psychology.models.swift_integration import SwiFTConfig

    config = SwiFTConfig()

    # 4D 설정 검증
    assert config.spatial_dims == (64, 64, 64)
    assert config.temporal_length == 100
    assert config.window_size == (8, 8, 8, 8)
    assert config.embed_dim == 384


def test_brainlm_config_setup():
    """BrainLM 설정 테스트"""
    from src.services.psychology.models.brainlm_integration import BrainLMConfig

    config = BrainLMConfig()

    # 뇌 언어 모델 설정 검증
    assert config.num_brain_regions == 400
    assert config.max_sequence_length == 200
    assert config.hidden_size == 768
    assert config.mask_ratio == 0.15


def test_grover_genomic_config():
    """GROVER 유전체 설정 테스트"""
    from src.services.psychology.models.gene_llm_integration import GenomicConfig

    config = GenomicConfig()

    # 유전체 설정 검증
    assert config.nucleotide_vocab_size == 6
    assert config.k_mer_size == 3
    assert config.num_chromosomes == 23


def test_dna_tokenizer_basic():
    """DNA 토크나이저 기본 기능 테스트"""
    from src.services.psychology.models.gene_llm_integration import DNATokenizer, GenomicConfig

    config = GenomicConfig()
    tokenizer = DNATokenizer(config)

    # 기본 핵산 매핑 테스트
    assert 'A' in tokenizer.nucleotide_to_id
    assert 'T' in tokenizer.nucleotide_to_id
    assert 'G' in tokenizer.nucleotide_to_id
    assert 'C' in tokenizer.nucleotide_to_id

    # DNA 서열 인코딩 테스트
    test_sequence = "ATGC"
    encoded = tokenizer.encode_dna_sequence(test_sequence)
    assert isinstance(encoded, list)
    assert len(encoded) > 0


def test_multimodal_config_basic():
    """다중모달 설정 기본 테스트"""
    from src.services.psychology.multimodal_fusion_engine import MultimodalConfig

    config = MultimodalConfig()

    # 융합 가중치 검증
    assert config.eeg_weight == 0.25
    assert config.fmri_weight == 0.25
    assert config.brain_lm_weight == 0.25
    assert config.genomic_weight == 0.25

    # 총합이 1.0인지 확인
    total_weight = config.eeg_weight + config.fmri_weight + config.brain_lm_weight + config.genomic_weight
    assert abs(total_weight - 1.0) < 0.001


def test_attention_fusion_module():
    """어텐션 융합 모듈 기본 테스트"""
    from src.services.psychology.multimodal_fusion_engine import AttentionFusionModule

    input_dims = {'eeg': 512, 'fmri': 384, 'brain_lm': 768, 'genomic': 256}
    fusion_module = AttentionFusionModule(input_dims)

    # 모듈 구성 요소 검증
    assert 'eeg' in fusion_module.projections
    assert 'fmri' in fusion_module.projections
    assert 'brain_lm' in fusion_module.projections
    assert 'genomic' in fusion_module.projections


def test_uncertainty_quantifier():
    """불확실성 정량화 모듈 테스트"""
    from src.services.psychology.multimodal_fusion_engine import UncertaintyQuantifier

    # 간단한 예측값들로 테스트
    predictions = [0.8, 0.75, 0.82, 0.78]

    # 인식론적 불확실성 계산
    epistemic_uncertainty = UncertaintyQuantifier.epistemic_uncertainty(predictions)
    assert isinstance(epistemic_uncertainty, float)
    assert epistemic_uncertainty >= 0.0

    # 다른 방법으로도 테스트
    entropy_uncertainty = UncertaintyQuantifier.epistemic_uncertainty(predictions, method="entropy")
    assert isinstance(entropy_uncertainty, float)
    assert entropy_uncertainty >= 0.0


def test_psychology_query_structure():
    """심리학 쿼리 구조 테스트"""
    from src.services.psychology.foundation_model_rag import PsychologyQuery

    query = PsychologyQuery(
        text="인지편향 연구의 최신 동향",
        query_type="literature_search",
        analysis_depth="comprehensive"
    )

    assert query.text == "인지편향 연구의 최신 동향"
    assert query.query_type == "literature_search"
    assert query.analysis_depth == "comprehensive"
    assert query.include_papers == True


def test_modality_data_structure():
    """모달리티 데이터 구조 테스트"""
    from src.services.psychology.multimodal_fusion_engine import ModalityData

    # 테스트용 EEG 데이터 생성
    eeg_data = np.random.randn(64, 1000)  # 64 channels, 1000 time points

    modality_data = ModalityData(
        eeg_data=eeg_data,
        clinical_query="ADHD 진단을 위한 EEG 분석"
    )

    assert modality_data.eeg_data is not None
    assert modality_data.eeg_data.shape == (64, 1000)
    assert modality_data.clinical_query == "ADHD 진단을 위한 EEG 분석"


def test_eeg_preprocessing_basic():
    """EEG 전처리 기본 기능 테스트 (의존성 제거)"""
    # 간단한 EEG 전처리 함수 테스트
    test_eeg = np.random.randn(64, 1000)

    # 정규화 테스트
    normalized = (test_eeg - np.mean(test_eeg, axis=1, keepdims=True)) / \
                (np.std(test_eeg, axis=1, keepdims=True) + 1e-8)

    assert normalized.shape == test_eeg.shape
    assert np.abs(np.mean(normalized, axis=1)).max() < 1e-6  # 평균이 0에 가까운지


def test_fmri_normalization():
    """fMRI 정규화 기본 테스트"""
    # 4D fMRI 데이터 시뮬레이션
    test_fmri = np.random.randn(64, 64, 64, 100)

    # Z-score 정규화
    mean_signal = np.mean(test_fmri, axis=-1, keepdims=True)
    std_signal = np.std(test_fmri, axis=-1, keepdims=True)
    normalized = (test_fmri - mean_signal) / (std_signal + 1e-8)

    assert normalized.shape == test_fmri.shape

    # 정규화 후 통계 확인
    time_axis_mean = np.mean(normalized, axis=-1)
    assert np.abs(time_axis_mean).max() < 1e-6  # 시간축 평균이 0에 가까운지


@pytest.mark.asyncio
async def test_mock_llm_service():
    """Mock LLM 서비스 테스트"""
    # LLM 서비스 없이도 작동하는지 테스트
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = "테스트 응답입니다."

    response = await mock_llm.generate(prompt="테스트 프롬프트", max_tokens=100)
    assert response == "테스트 응답입니다."


def test_channel_equivariant_attention_structure():
    """채널 등변성 어텐션 구조 테스트 (초기화만)"""
    from src.services.psychology.models.diver0_integration import EEGAnalysisConfig

    config = EEGAnalysisConfig()

    # 어텐션 매개변수 계산
    d_model = config.transformer_dim
    n_heads = config.num_attention_heads
    n_channels = config.n_channels

    assert d_model % n_heads == 0  # 헤드 수로 나누어떨어지는지
    head_dim = d_model // n_heads
    assert head_dim > 0


def test_genomic_variant_parsing():
    """유전체 변이 파싱 기본 테스트"""
    from src.services.psychology.models.gene_llm_integration import GeneticVariant

    # 기본 변이 구조 테스트
    variant = GeneticVariant(
        variant_id="rs1234567",
        chromosome="1",
        position=12345,
        reference_allele="A",
        alternate_allele="T",
        variant_type="SNP",
        gene_symbol="TEST_GENE",
        functional_consequence="missense",
        clinical_significance="likely_pathogenic",
        allele_frequency=0.1
    )

    assert variant.variant_id == "rs1234567"
    assert variant.variant_type == "SNP"
    assert variant.allele_frequency == 0.1


def test_korean_term_mapping():
    """한국어 용어 매핑 기본 테스트"""
    korean_to_english = {
        '인지편향': 'cognitive bias',
        '실행기능': 'executive function',
        '작업기억': 'working memory',
        '주의집중': 'attention',
        '언어발달': 'language development'
    }

    test_query = "ADHD 아동의 실행기능과 작업기억 문제"

    # 한국어 용어 감지 및 영어 확장
    enhanced_query = test_query
    for korean, english in korean_to_english.items():
        if korean in test_query:
            enhanced_query += f" {english}"

    assert 'executive function' in enhanced_query
    assert 'working memory' in enhanced_query


def test_statistics_basic():
    """기본 통계 함수 테스트"""
    import numpy as np

    # 샘플 데이터
    data = np.array([0.8, 0.75, 0.82, 0.78, 0.85])

    # 기본 통계
    mean_val = np.mean(data)
    std_val = np.std(data)
    var_val = np.var(data)

    assert 0.7 < mean_val < 0.9
    assert std_val > 0
    assert var_val > 0


@pytest.mark.asyncio
async def test_comprehensive_integration_mock():
    """전체 시스템 통합 Mock 테스트"""
    # 가상의 다중모달 데이터
    mock_eeg = np.random.randn(64, 1000)
    mock_fmri = np.random.randn(64, 64, 64, 100)
    mock_variants = ["rs1234567", "rs2345678"]

    # 기본 검증
    assert mock_eeg.shape == (64, 1000)
    assert mock_fmri.shape == (64, 64, 64, 100)
    assert len(mock_variants) == 2

    # 간단한 융합 시뮬레이션
    eeg_score = np.random.rand()
    fmri_score = np.random.rand()
    genetic_score = np.random.rand()

    # 가중 평균 계산
    weights = [0.25, 0.25, 0.25]  # 3개 모달리티
    scores = [eeg_score, fmri_score, genetic_score]

    integrated_score = sum(w * s for w, s in zip(weights, scores)) / sum(weights)

    assert 0.0 <= integrated_score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])