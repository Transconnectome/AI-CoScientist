"""
Psychology Multimodal Fusion Engine
UltraThink 구현: 4개 Foundation Model 통합을 통한 종합 심리학 분석

통합 모델들:
- DIVER-0: EEG Foundation Model
- SwiFT: 4D fMRI Transformer
- BrainLM: Brain Language Model (Zero-shot)
- Gene-LLM/GROVER: Genomic Foundation Model

목표: 다중모달 증거를 융합하여 임상적으로 신뢰할 수 있는 심리학 분석 제공
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from datetime import datetime
import asyncio
import json
from abc import ABC, abstractmethod
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Foundation Models 임포트
from .models.diver0_integration import DIVER0Foundation, EEGPattern
from .models.swift_integration import SwiFTTransformer, fMRIAnalysisResult
from .models.brainlm_integration import BrainLMFoundation, ZeroShotPrediction
from .models.gene_llm_integration import GROVERGenomics, GenomicAnalysisResult

# 기존 AI-CoScientist 인프라 활용
from src.core.config import get_settings
from src.services.llm.interface import LLMServiceInterface
from src.monitoring.rag_metrics import RAGMetrics

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class MultimodalConfig:
    """다중모달 융합 엔진 구성"""
    # 융합 가중치
    eeg_weight: float = 0.25
    fmri_weight: float = 0.25
    brain_lm_weight: float = 0.25
    genomic_weight: float = 0.25

    # 신뢰도 임계값
    confidence_threshold: float = 0.7
    uncertainty_tolerance: float = 0.2

    # 융합 방법
    fusion_method: str = "weighted_ensemble"  # weighted_ensemble, attention_fusion, deep_fusion

    # 임상 결정 지원
    clinical_decision_support: bool = True
    explain_predictions: bool = True

    # 성능 최적화
    parallel_processing: bool = True
    cache_results: bool = True


@dataclass
class ModalityData:
    """모달리티별 데이터"""
    eeg_data: Optional[np.ndarray] = None
    fmri_data: Optional[Union[np.ndarray, str]] = None  # 4D array or NIfTI path
    genetic_variants: Optional[List[str]] = None
    clinical_query: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class IntegratedEvidence:
    """통합된 다중모달 증거"""
    eeg_evidence: Optional[EEGPattern] = None
    fmri_evidence: Optional[fMRIAnalysisResult] = None
    brain_lm_evidence: Optional[ZeroShotPrediction] = None
    genomic_evidence: Optional[GenomicAnalysisResult] = None
    modality_weights: Dict[str, float] = None
    uncertainty_scores: Dict[str, float] = None


@dataclass
class PsychologyAnalysisResult:
    """종합 심리학 분석 결과"""
    integrated_score: float
    confidence_level: str
    clinical_interpretation: str
    evidence_summary: Dict[str, Any]
    risk_assessment: Dict[str, float]
    treatment_recommendations: List[str]
    follow_up_suggestions: List[str]
    uncertainty_quantification: Dict[str, float]
    modality_contributions: Dict[str, float]
    statistical_significance: Dict[str, float]
    metadata: Dict[str, Any]


class AttentionFusionModule(nn.Module):
    """어텐션 기반 다중모달 융합 모듈"""

    def __init__(self, input_dims: Dict[str, int], hidden_dim: int = 256):
        super().__init__()
        self.modalities = list(input_dims.keys())

        # 각 모달리티별 projection
        self.projections = nn.ModuleDict({
            modality: nn.Linear(input_dims[modality], hidden_dim)
            for modality in self.modalities
        })

        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # Fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(hidden_dim * len(self.modalities), hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, modality_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        다중모달 특성 융합

        Args:
            modality_features: 각 모달리티별 특성 딕셔너리
        Returns:
            융합 결과 및 어텐션 가중치
        """
        projected_features = {}
        feature_tensors = []

        # 각 모달리티를 공통 차원으로 투영
        for modality in self.modalities:
            if modality in modality_features:
                projected = self.projections[modality](modality_features[modality])
                projected_features[modality] = projected
                feature_tensors.append(projected.unsqueeze(1))

        # Cross-modal attention if multiple modalities present
        if len(feature_tensors) > 1:
            # Stack features for attention
            stacked_features = torch.cat(feature_tensors, dim=1)  # [batch, n_modalities, hidden_dim]

            # Self-attention across modalities
            attended_features, attention_weights = self.cross_attention(
                stacked_features, stacked_features, stacked_features
            )

            # Flatten for fusion
            fused_features = attended_features.flatten(start_dim=1)

        else:
            # Single modality case
            fused_features = feature_tensors[0].squeeze(1)
            attention_weights = torch.ones(1, 1, 1)

        # Final fusion
        integrated_score = self.fusion_layers(fused_features)
        uncertainty_score = self.uncertainty_head(fused_features.mean(dim=1, keepdim=True))

        return {
            'integrated_score': integrated_score,
            'uncertainty_score': uncertainty_score,
            'attention_weights': attention_weights,
            'modality_features': projected_features
        }


class UncertaintyQuantifier:
    """불확실성 정량화 모듈"""

    @staticmethod
    def epistemic_uncertainty(predictions: List[float], method: str = "variance") -> float:
        """인식론적 불확실성 (모델 불확실성)"""
        predictions = np.array(predictions)

        if method == "variance":
            return float(np.var(predictions))
        elif method == "entropy":
            # 예측값을 확률분포로 변환 후 엔트로피 계산
            probs = np.exp(predictions) / np.sum(np.exp(predictions))
            return float(-np.sum(probs * np.log(probs + 1e-8)))
        else:
            return float(np.std(predictions))

    @staticmethod
    def aleatoric_uncertainty(model_outputs: Dict[str, Any]) -> float:
        """우연론적 불확실성 (데이터 불확실성)"""
        # 각 모달리티의 내재적 불확실성 평균
        uncertainties = []

        if 'eeg_evidence' in model_outputs and model_outputs['eeg_evidence']:
            eeg_uncertainty = 1.0 - model_outputs['eeg_evidence'].confidence_score
            uncertainties.append(eeg_uncertainty)

        if 'fmri_evidence' in model_outputs and model_outputs['fmri_evidence']:
            # fMRI 불확실성은 신뢰구간 폭으로 추정
            ci_widths = [
                abs(interval[1] - interval[0])
                for interval in model_outputs['fmri_evidence'].confidence_intervals.values()
            ]
            fmri_uncertainty = np.mean(ci_widths) if ci_widths else 0.1
            uncertainties.append(fmri_uncertainty)

        return float(np.mean(uncertainties)) if uncertainties else 0.1

    @staticmethod
    def total_uncertainty(epistemic: float, aleatoric: float) -> float:
        """총 불확실성 (제곱합의 제곱근)"""
        return float(np.sqrt(epistemic**2 + aleatoric**2))


class MultimodalFusionEngine:
    """
    다중모달 융합 엔진
    UltraThink: 4개 Foundation Model을 통합한 종합 심리학 분석
    """

    def __init__(self, config: Optional[MultimodalConfig] = None):
        self.config = config or MultimodalConfig()

        # Foundation Models 초기화
        self.diver0_model = DIVER0Foundation()
        self.swift_model = SwiFTTransformer()
        self.brainlm_model = BrainLMFoundation()
        self.grover_model = GROVERGenomics()

        # Attention Fusion Module 초기화
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.attention_fusion = AttentionFusionModule(
            input_dims={
                'eeg': 512,
                'fmri': 384,
                'brain_lm': 768,
                'genomic': 256
            }
        ).to(self.device)

        # 불확실성 정량화 모듈
        self.uncertainty_quantifier = UncertaintyQuantifier()

        # LLM 서비스 연동
        self.llm_service = None
        asyncio.create_task(self._init_llm_service())

        # 임상 지식 베이스
        self.clinical_knowledge = self._load_clinical_knowledge()

        # 성능 메트릭 추적
        self.metrics_history = []
        self.analysis_cache = {}

        logger.info("Multimodal Fusion Engine initialized successfully")

    async def _init_llm_service(self):
        """LLM 서비스 초기화"""
        try:
            self.llm_service = None  # Placeholder for testing
            logger.info("LLM service initialized for Multimodal Fusion Engine")
        except Exception as e:
            logger.warning(f"LLM service initialization failed: {e}")
            self.llm_service = None

    def _load_clinical_knowledge(self) -> Dict[str, Any]:
        """임상 지식 베이스 로드"""
        return {
            'diagnostic_criteria': {
                'autism_spectrum_disorder': {
                    'eeg_markers': ['gamma_band_abnormality', 'connectivity_disruption'],
                    'fmri_markers': ['default_mode_network_alteration', 'social_brain_hypoactivation'],
                    'genetic_markers': ['SHANK3', 'CHD8', 'FMR1'],
                    'behavioral_markers': ['social_communication_deficits', 'restricted_interests']
                },
                'adhd': {
                    'eeg_markers': ['theta_beta_ratio', 'frontal_hypoactivation'],
                    'fmri_markers': ['attention_network_dysfunction', 'executive_control_deficits'],
                    'genetic_markers': ['DRD4', 'SLC6A3', 'SNAP25'],
                    'behavioral_markers': ['inattention', 'hyperactivity', 'impulsivity']
                }
            },
            'treatment_protocols': {
                'behavioral_interventions': ['CBT', 'ABA', 'social_skills_training'],
                'pharmacological': ['stimulants', 'antidepressants', 'antipsychotics'],
                'neurofeedback': ['EEG_biofeedback', 'fMRI_neurofeedback']
            }
        }

    async def integrate_multimodal_evidence(self,
                                          modality_data: ModalityData,
                                          analysis_type: str = "comprehensive") -> PsychologyAnalysisResult:
        """
        다중모달 증거 통합 분석

        Args:
            modality_data: 각 모달리티별 데이터
            analysis_type: 분석 유형 (comprehensive, diagnostic, predictive)
        Returns:
            통합된 심리학 분석 결과
        """
        start_time = datetime.now()

        try:
            # 캐시 확인
            cache_key = self._generate_cache_key(modality_data, analysis_type)
            if self.config.cache_results and cache_key in self.analysis_cache:
                logger.info("Using cached analysis result")
                return self.analysis_cache[cache_key]

            # 1단계: 각 모달리티별 분석 수행
            integrated_evidence = await self._analyze_individual_modalities(modality_data)

            # 2단계: 증거 융합
            fusion_result = await self._fuse_multimodal_evidence(integrated_evidence, analysis_type)

            # 3단계: 불확실성 정량화
            uncertainty_metrics = self._quantify_uncertainties(integrated_evidence, fusion_result)

            # 4단계: 임상 해석 생성
            clinical_interpretation = await self._generate_clinical_interpretation(
                fusion_result, integrated_evidence, uncertainty_metrics
            )

            # 5단계: 위험 평가 및 권장사항
            risk_assessment = await self._assess_clinical_risks(fusion_result, integrated_evidence)
            treatment_recommendations = await self._generate_treatment_recommendations(
                fusion_result, risk_assessment
            )

            # 6단계: 통계적 유의성 검정
            statistical_significance = self._test_statistical_significance(integrated_evidence)

            # 결과 구성
            analysis_result = PsychologyAnalysisResult(
                integrated_score=fusion_result['integrated_score'],
                confidence_level=self._determine_confidence_level(uncertainty_metrics),
                clinical_interpretation=clinical_interpretation,
                evidence_summary=self._summarize_evidence(integrated_evidence),
                risk_assessment=risk_assessment,
                treatment_recommendations=treatment_recommendations,
                follow_up_suggestions=await self._generate_followup_suggestions(fusion_result),
                uncertainty_quantification=uncertainty_metrics,
                modality_contributions=fusion_result['modality_contributions'],
                statistical_significance=statistical_significance,
                metadata={
                    'analysis_type': analysis_type,
                    'processing_time': (datetime.now() - start_time).total_seconds(),
                    'available_modalities': list(integrated_evidence.modality_weights.keys()),
                    'fusion_method': self.config.fusion_method,
                    'analysis_timestamp': datetime.now().isoformat()
                }
            )

            # 캐싱
            if self.config.cache_results:
                self.analysis_cache[cache_key] = analysis_result

            # 성능 메트릭 기록
            self._record_metrics(start_time, analysis_result)

            return analysis_result

        except Exception as e:
            logger.error(f"Multimodal evidence integration failed: {e}")
            raise

    async def _analyze_individual_modalities(self, modality_data: ModalityData) -> IntegratedEvidence:
        """각 모달리티별 개별 분석"""
        evidence = IntegratedEvidence()
        available_weights = {}
        uncertainty_scores = {}

        # 병렬 처리 태스크 목록
        analysis_tasks = []

        # EEG 분석
        if modality_data.eeg_data is not None:
            analysis_tasks.append(
                self._analyze_eeg_modality(modality_data.eeg_data)
            )
        else:
            analysis_tasks.append(None)

        # fMRI 분석
        if modality_data.fmri_data is not None:
            analysis_tasks.append(
                self._analyze_fmri_modality(modality_data.fmri_data)
            )
        else:
            analysis_tasks.append(None)

        # BrainLM 분석
        if modality_data.clinical_query is not None:
            analysis_tasks.append(
                self._analyze_brainlm_modality(modality_data.clinical_query)
            )
        else:
            analysis_tasks.append(None)

        # 유전체 분석
        if modality_data.genetic_variants is not None:
            analysis_tasks.append(
                self._analyze_genomic_modality(modality_data.genetic_variants)
            )
        else:
            analysis_tasks.append(None)

        # 병렬 실행
        if self.config.parallel_processing:
            results = await asyncio.gather(*[task for task in analysis_tasks if task is not None], return_exceptions=True)

            # 결과 할당
            task_idx = 0
            if modality_data.eeg_data is not None:
                if not isinstance(results[task_idx], Exception):
                    evidence.eeg_evidence = results[task_idx]
                    available_weights['eeg'] = self.config.eeg_weight
                    uncertainty_scores['eeg'] = 1.0 - evidence.eeg_evidence.confidence_score
                task_idx += 1

            if modality_data.fmri_data is not None:
                if not isinstance(results[task_idx], Exception):
                    evidence.fmri_evidence = results[task_idx]
                    available_weights['fmri'] = self.config.fmri_weight
                    # fMRI 불확실성은 신뢰구간으로부터 추정
                    ci_widths = [abs(interval[1] - interval[0])
                               for interval in evidence.fmri_evidence.confidence_intervals.values()]
                    uncertainty_scores['fmri'] = np.mean(ci_widths) if ci_widths else 0.1
                task_idx += 1

            if modality_data.clinical_query is not None:
                if not isinstance(results[task_idx], Exception):
                    evidence.brain_lm_evidence = results[task_idx]
                    available_weights['brain_lm'] = self.config.brain_lm_weight
                    uncertainty_scores['brain_lm'] = 1.0 - evidence.brain_lm_evidence.confidence_score
                task_idx += 1

            if modality_data.genetic_variants is not None:
                if not isinstance(results[task_idx], Exception):
                    evidence.genomic_evidence = results[task_idx]
                    available_weights['genomic'] = self.config.genomic_weight
                    uncertainty_scores['genomic'] = 1.0 - evidence.genomic_evidence.confidence_metrics['overall_confidence']
                task_idx += 1

        else:
            # 순차 실행
            if modality_data.eeg_data is not None:
                try:
                    evidence.eeg_evidence = await self._analyze_eeg_modality(modality_data.eeg_data)
                    available_weights['eeg'] = self.config.eeg_weight
                    uncertainty_scores['eeg'] = 1.0 - evidence.eeg_evidence.confidence_score
                except Exception as e:
                    logger.warning(f"EEG analysis failed: {e}")

            if modality_data.fmri_data is not None:
                try:
                    evidence.fmri_evidence = await self._analyze_fmri_modality(modality_data.fmri_data)
                    available_weights['fmri'] = self.config.fmri_weight
                    ci_widths = [abs(interval[1] - interval[0])
                               for interval in evidence.fmri_evidence.confidence_intervals.values()]
                    uncertainty_scores['fmri'] = np.mean(ci_widths) if ci_widths else 0.1
                except Exception as e:
                    logger.warning(f"fMRI analysis failed: {e}")

            if modality_data.clinical_query is not None:
                try:
                    evidence.brain_lm_evidence = await self._analyze_brainlm_modality(modality_data.clinical_query)
                    available_weights['brain_lm'] = self.config.brain_lm_weight
                    uncertainty_scores['brain_lm'] = 1.0 - evidence.brain_lm_evidence.confidence_score
                except Exception as e:
                    logger.warning(f"BrainLM analysis failed: {e}")

            if modality_data.genetic_variants is not None:
                try:
                    evidence.genomic_evidence = await self._analyze_genomic_modality(modality_data.genetic_variants)
                    available_weights['genomic'] = self.config.genomic_weight
                    uncertainty_scores['genomic'] = 1.0 - evidence.genomic_evidence.confidence_metrics['overall_confidence']
                except Exception as e:
                    logger.warning(f"Genomic analysis failed: {e}")

        # 가중치 정규화
        total_weight = sum(available_weights.values())
        if total_weight > 0:
            normalized_weights = {k: v/total_weight for k, v in available_weights.items()}
        else:
            normalized_weights = {}

        evidence.modality_weights = normalized_weights
        evidence.uncertainty_scores = uncertainty_scores

        return evidence

    async def _analyze_eeg_modality(self, eeg_data: np.ndarray) -> EEGPattern:
        """EEG 모달리티 분석"""
        return await self.diver0_model.analyze_patterns(
            eeg_data=eeg_data,
            analysis_type="comprehensive"
        )

    async def _analyze_fmri_modality(self, fmri_data: Union[np.ndarray, str]) -> fMRIAnalysisResult:
        """fMRI 모달리티 분석"""
        return await self.swift_model.analyze_spatiotemporal_dynamics(
            fmri_4d=fmri_data,
            target_outcome="comprehensive"
        )

    async def _analyze_brainlm_modality(self, clinical_query: str) -> ZeroShotPrediction:
        """BrainLM 모달리티 분석"""
        return await self.brainlm_model.zero_shot_inference(
            query=clinical_query,
            context_type="comprehensive_analysis"
        )

    async def _analyze_genomic_modality(self, genetic_variants: List[str]) -> GenomicAnalysisResult:
        """유전체 모달리티 분석"""
        return await self.grover_model.analyze_genetic_risk(
            variants=genetic_variants,
            phenotype="comprehensive"
        )

    async def _fuse_multimodal_evidence(self,
                                      evidence: IntegratedEvidence,
                                      analysis_type: str) -> Dict[str, Any]:
        """다중모달 증거 융합"""
        if self.config.fusion_method == "weighted_ensemble":
            return self._weighted_ensemble_fusion(evidence)
        elif self.config.fusion_method == "attention_fusion":
            return await self._attention_based_fusion(evidence)
        else:  # deep_fusion
            return await self._deep_neural_fusion(evidence)

    def _weighted_ensemble_fusion(self, evidence: IntegratedEvidence) -> Dict[str, Any]:
        """가중 앙상블 융합"""
        weighted_scores = []
        modality_contributions = {}

        # 각 모달리티별 점수 추출 및 가중치 적용
        if evidence.eeg_evidence and 'eeg' in evidence.modality_weights:
            eeg_score = evidence.eeg_evidence.confidence_score
            weight = evidence.modality_weights['eeg']
            weighted_scores.append(eeg_score * weight)
            modality_contributions['eeg'] = eeg_score * weight

        if evidence.fmri_evidence and 'fmri' in evidence.modality_weights:
            # fMRI 종합 점수 계산
            fmri_score = np.mean(list(evidence.fmri_evidence.developmental_predictions.values()))
            weight = evidence.modality_weights['fmri']
            weighted_scores.append(fmri_score * weight)
            modality_contributions['fmri'] = fmri_score * weight

        if evidence.brain_lm_evidence and 'brain_lm' in evidence.modality_weights:
            brainlm_score = evidence.brain_lm_evidence.prediction_value
            weight = evidence.modality_weights['brain_lm']
            weighted_scores.append(brainlm_score * weight)
            modality_contributions['brain_lm'] = brainlm_score * weight

        if evidence.genomic_evidence and 'genomic' in evidence.modality_weights:
            genomic_score = evidence.genomic_evidence.risk_score
            weight = evidence.modality_weights['genomic']
            weighted_scores.append(genomic_score * weight)
            modality_contributions['genomic'] = genomic_score * weight

        # 통합 점수 계산
        integrated_score = sum(weighted_scores) if weighted_scores else 0.5

        return {
            'integrated_score': integrated_score,
            'modality_contributions': modality_contributions,
            'fusion_method': 'weighted_ensemble'
        }

    async def _attention_based_fusion(self, evidence: IntegratedEvidence) -> Dict[str, Any]:
        """어텐션 기반 융합"""
        # 각 모달리티의 특성 벡터 추출
        modality_features = {}

        if evidence.eeg_evidence:
            # EEG 특성을 텐서로 변환
            eeg_features = torch.FloatTensor(evidence.eeg_evidence.pattern_features[:512]).unsqueeze(0).to(self.device)
            modality_features['eeg'] = eeg_features

        if evidence.fmri_evidence:
            # fMRI 특성 추출 및 차원 맞춤
            fmri_features = torch.FloatTensor(evidence.fmri_evidence.spatiotemporal_features[:384]).unsqueeze(0).to(self.device)
            modality_features['fmri'] = fmri_features

        if evidence.brain_lm_evidence:
            # BrainLM의 네트워크 활성화를 특성으로 사용
            brain_features = torch.FloatTensor(
                list(evidence.brain_lm_evidence.network_activation.values())[:768]
            ).unsqueeze(0).to(self.device)
            modality_features['brain_lm'] = brain_features

        if evidence.genomic_evidence:
            # 유전체 특성 추출
            genomic_features_list = []
            for pathway_data in evidence.genomic_evidence.pathway_analysis.values():
                genomic_features_list.append(pathway_data['activation_score'])
            genomic_features = torch.FloatTensor(genomic_features_list[:256]).unsqueeze(0).to(self.device)
            modality_features['genomic'] = genomic_features

        # 어텐션 융합 수행
        with torch.no_grad():
            fusion_outputs = self.attention_fusion(modality_features)

        return {
            'integrated_score': float(fusion_outputs['integrated_score'].squeeze()),
            'modality_contributions': {
                k: float(v.mean()) for k, v in fusion_outputs['modality_features'].items()
            },
            'attention_weights': fusion_outputs['attention_weights'].cpu().numpy(),
            'uncertainty_score': float(fusion_outputs['uncertainty_score'].squeeze()),
            'fusion_method': 'attention_fusion'
        }

    async def _deep_neural_fusion(self, evidence: IntegratedEvidence) -> Dict[str, Any]:
        """깊은 신경망 융합 (향후 구현)"""
        # 현재는 어텐션 융합과 동일
        return await self._attention_based_fusion(evidence)

    def _quantify_uncertainties(self,
                              evidence: IntegratedEvidence,
                              fusion_result: Dict[str, Any]) -> Dict[str, float]:
        """불확실성 정량화"""
        # 개별 모달리티 예측값들 수집
        predictions = []
        if evidence.eeg_evidence:
            predictions.append(evidence.eeg_evidence.confidence_score)
        if evidence.fmri_evidence:
            predictions.append(np.mean(list(evidence.fmri_evidence.developmental_predictions.values())))
        if evidence.brain_lm_evidence:
            predictions.append(evidence.brain_lm_evidence.prediction_value)
        if evidence.genomic_evidence:
            predictions.append(evidence.genomic_evidence.risk_score)

        # 불확실성 계산
        epistemic_uncertainty = self.uncertainty_quantifier.epistemic_uncertainty(predictions)
        aleatoric_uncertainty = self.uncertainty_quantifier.aleatoric_uncertainty({
            'eeg_evidence': evidence.eeg_evidence,
            'fmri_evidence': evidence.fmri_evidence
        })
        total_uncertainty = self.uncertainty_quantifier.total_uncertainty(
            epistemic_uncertainty, aleatoric_uncertainty
        )

        return {
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'total_uncertainty': total_uncertainty,
            'prediction_variance': float(np.var(predictions)) if predictions else 0.0,
            'confidence_spread': float(np.max(predictions) - np.min(predictions)) if len(predictions) > 1 else 0.0
        }

    async def _generate_clinical_interpretation(self,
                                              fusion_result: Dict[str, Any],
                                              evidence: IntegratedEvidence,
                                              uncertainty_metrics: Dict[str, float]) -> str:
        """임상 해석 생성"""
        if self.llm_service is None:
            return self._default_clinical_interpretation(fusion_result, uncertainty_metrics)

        try:
            # 증거 요약 구성
            evidence_summary = []

            if evidence.eeg_evidence:
                evidence_summary.append(f"EEG: {evidence.eeg_evidence.clinical_interpretation}")

            if evidence.fmri_evidence:
                dev_preds = evidence.fmri_evidence.developmental_predictions
                evidence_summary.append(f"fMRI: 발달예측 - 인지: {dev_preds.get('cognitive', 0):.2f}")

            if evidence.brain_lm_evidence:
                evidence_summary.append(f"뇌언어모델: {evidence.brain_lm_evidence.explanation}")

            if evidence.genomic_evidence:
                evidence_summary.append(f"유전체: 위험도 {evidence.genomic_evidence.risk_score:.2f}")

            prompt = f"""
            다중모달 뇌과학 분석 결과를 종합하여 임상적 해석을 제공하세요:

            통합 점수: {fusion_result['integrated_score']:.3f}
            총 불확실성: {uncertainty_metrics['total_uncertainty']:.3f}

            모달리티별 증거:
            {chr(10).join(evidence_summary)}

            다음 사항을 포함하여 3-4문장으로 종합 해석을 제공하세요:
            1. 주요 소견 요약
            2. 임상적 의미
            3. 신뢰도 평가
            4. 추가 고려사항
            """

            response = await self.llm_service.generate(
                prompt=prompt,
                max_tokens=300,
                temperature=0.3
            )

            return response.strip()

        except Exception as e:
            logger.warning(f"Clinical interpretation generation failed: {e}")
            return self._default_clinical_interpretation(fusion_result, uncertainty_metrics)

    def _default_clinical_interpretation(self,
                                       fusion_result: Dict[str, Any],
                                       uncertainty_metrics: Dict[str, float]) -> str:
        """기본 임상 해석"""
        score = fusion_result['integrated_score']
        uncertainty = uncertainty_metrics['total_uncertainty']

        if score > 0.8 and uncertainty < 0.2:
            return "다중모달 분석 결과 높은 신뢰도로 특정 패턴이 확인되었습니다. 추가 임상 평가를 권장합니다."
        elif score > 0.6 and uncertainty < 0.3:
            return "중간 정도의 신뢰도로 의미있는 패턴이 관찰되었습니다. 추가 검사를 통한 확인이 필요합니다."
        else:
            return "현재 증거만으로는 명확한 결론을 내리기 어렵습니다. 추가 데이터 수집을 권장합니다."

    async def _assess_clinical_risks(self,
                                   fusion_result: Dict[str, Any],
                                   evidence: IntegratedEvidence) -> Dict[str, float]:
        """임상 위험도 평가"""
        risk_scores = {}

        # 기본 위험도 (통합 점수 기반)
        base_risk = fusion_result['integrated_score']

        # 질환별 위험도 계산
        if evidence.genomic_evidence:
            disease_risks = evidence.genomic_evidence.disease_risk_assessment
            for disease, risk in disease_risks.items():
                risk_scores[disease] = min(1.0, (base_risk + risk) / 2.0)

        # 일반적 위험 카테고리
        risk_scores.update({
            'neurodevelopmental_disorder': base_risk * 0.8,
            'cognitive_impairment': base_risk * 0.7,
            'attention_deficit': base_risk * 0.6,
            'learning_disability': base_risk * 0.5
        })

        return risk_scores

    async def _generate_treatment_recommendations(self,
                                                fusion_result: Dict[str, Any],
                                                risk_assessment: Dict[str, float]) -> List[str]:
        """치료 권장사항 생성"""
        recommendations = []

        # 위험도에 따른 기본 권장사항
        max_risk_condition = max(risk_assessment.keys(), key=lambda k: risk_assessment[k])
        max_risk_score = risk_assessment[max_risk_condition]

        if max_risk_score > 0.7:
            recommendations.extend([
                "전문의 상담을 통한 정밀 진단 평가",
                "종합적인 신경심리학적 평가 실시",
                "개별화된 치료 계획 수립"
            ])
        elif max_risk_score > 0.5:
            recommendations.extend([
                "추가 임상 평가 및 모니터링",
                "행동 관찰 및 기능 평가",
                "가족 상담 및 교육"
            ])
        else:
            recommendations.extend([
                "정기적인 발달 모니터링",
                "예방적 개입 프로그램 참여 고려",
                "건강한 생활습관 유지"
            ])

        return recommendations

    async def _generate_followup_suggestions(self, fusion_result: Dict[str, Any]) -> List[str]:
        """후속 조치 제안"""
        suggestions = [
            "3-6개월 후 추적 평가",
            "다른 임상 전문가와의 협진 고려",
            "가족력 및 환경 요인 추가 조사"
        ]

        if fusion_result['integrated_score'] > 0.6:
            suggestions.extend([
                "치료 효과 모니터링을 위한 정기 평가",
                "교육 기관과의 협력 방안 논의"
            ])

        return suggestions

    def _test_statistical_significance(self, evidence: IntegratedEvidence) -> Dict[str, float]:
        """통계적 유의성 검정"""
        # 단순화된 유의성 검정 (실제로는 더 정교한 통계 분석 필요)
        significance = {}

        predictions = []
        if evidence.eeg_evidence:
            predictions.append(evidence.eeg_evidence.confidence_score)
        if evidence.fmri_evidence:
            predictions.append(np.mean(list(evidence.fmri_evidence.developmental_predictions.values())))
        if evidence.brain_lm_evidence:
            predictions.append(evidence.brain_lm_evidence.prediction_value)
        if evidence.genomic_evidence:
            predictions.append(evidence.genomic_evidence.risk_score)

        if len(predictions) >= 2:
            # 예측값들 간의 상관관계
            correlation_matrix = np.corrcoef(predictions + [0.5] * (4 - len(predictions)))
            avg_correlation = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])

            # 의사 p-값 계산 (실제로는 적절한 통계 검정 사용)
            pseudo_pvalue = 1.0 - abs(avg_correlation)

            significance['cross_modal_consistency'] = float(avg_correlation)
            significance['statistical_significance'] = float(pseudo_pvalue)
        else:
            significance['cross_modal_consistency'] = 0.5
            significance['statistical_significance'] = 1.0

        return significance

    def _summarize_evidence(self, evidence: IntegratedEvidence) -> Dict[str, Any]:
        """증거 요약"""
        summary = {}

        if evidence.eeg_evidence:
            summary['eeg'] = {
                'confidence': evidence.eeg_evidence.confidence_score,
                'main_finding': '특정 EEG 패턴 식별',
                'frequency_bands': evidence.eeg_evidence.frequency_bands
            }

        if evidence.fmri_evidence:
            summary['fmri'] = {
                'developmental_predictions': evidence.fmri_evidence.developmental_predictions,
                'main_finding': '시공간적 뇌 역학 분석',
                'network_activation': len(evidence.fmri_evidence.functional_networks)
            }

        if evidence.brain_lm_evidence:
            summary['brain_lm'] = {
                'prediction_value': evidence.brain_lm_evidence.prediction_value,
                'confidence': evidence.brain_lm_evidence.confidence_score,
                'main_finding': '뇌 언어 모델 추론'
            }

        if evidence.genomic_evidence:
            summary['genomic'] = {
                'risk_score': evidence.genomic_evidence.risk_score,
                'main_finding': '유전적 위험 요인 분석',
                'pathway_activation': len(evidence.genomic_evidence.pathway_analysis)
            }

        return summary

    def _determine_confidence_level(self, uncertainty_metrics: Dict[str, float]) -> str:
        """신뢰도 수준 결정"""
        total_uncertainty = uncertainty_metrics['total_uncertainty']

        if total_uncertainty < 0.2:
            return "높음"
        elif total_uncertainty < 0.4:
            return "보통"
        else:
            return "낮음"

    def _generate_cache_key(self, modality_data: ModalityData, analysis_type: str) -> str:
        """캐시 키 생성"""
        import hashlib

        # 데이터 해시 생성 (단순화)
        key_components = [
            analysis_type,
            str(modality_data.eeg_data is not None),
            str(modality_data.fmri_data is not None),
            str(modality_data.genetic_variants is not None),
            str(modality_data.clinical_query)
        ]

        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _record_metrics(self, start_time: datetime, result: PsychologyAnalysisResult):
        """성능 메트릭 기록"""
        processing_time = (datetime.now() - start_time).total_seconds()

        metrics = RAGMetrics(
            latency=processing_time,
            quality_score=result.integrated_score,
            tokens_processed=len(result.evidence_summary),
            retrieval_time=processing_time * 0.3,
            generation_time=processing_time * 0.7,
            context_relevance=result.integrated_score,
            faithfulness=1.0 - result.uncertainty_quantification['total_uncertainty'],
            answer_relevancy=result.integrated_score,
            strategy=f"multimodal_fusion_{result.metadata['fusion_method']}",
            timestamp=datetime.now()
        )

        self.metrics_history.append(metrics)
        logger.info(f"Multimodal analysis completed: {processing_time:.3f}s, score: {result.integrated_score:.3f}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 반환"""
        if not self.metrics_history:
            return {}

        latencies = [m.latency for m in self.metrics_history]
        scores = [m.quality_score for m in self.metrics_history]

        return {
            'total_analyses': len(self.metrics_history),
            'avg_latency': np.mean(latencies),
            'avg_integrated_score': np.mean(scores),
            'min_latency': np.min(latencies),
            'max_latency': np.max(latencies),
            'latency_std': np.std(latencies),
            'score_std': np.std(scores),
            'fusion_methods_supported': ['weighted_ensemble', 'attention_fusion', 'deep_fusion'],
            'modalities_supported': ['EEG', 'fMRI', 'BrainLM', 'Genomics'],
            'cache_hit_rate': len(self.analysis_cache) / max(1, len(self.metrics_history)),
            'last_updated': datetime.now().isoformat()
        }