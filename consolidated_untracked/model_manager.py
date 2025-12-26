#!/usr/bin/env python3
"""
Model Manager for Enhanced DD-RAPTOR System
2025 Best Practice: 모델 로딩 최적화 및 싱글톤 패턴 적용
"""

import logging
import asyncio
from typing import Dict, Optional
from threading import Lock

from sentence_transformers import SentenceTransformer, CrossEncoder
import torch

logger = logging.getLogger(__name__)

class ModelManager:
    """모델 로딩 및 관리 (싱글톤 패턴)"""

    _instance = None
    _lock = Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.embedding_model = None
            self.cross_encoder = None
            self.small_lm = None
            self._config = None
            ModelManager._initialized = True

    async def load_models(self, config: Dict):
        """모델 비동기 로딩"""
        self._config = config

        logger.info("Loading models...")

        # GPU 사용 가능 여부 확인
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        # 임베딩 모델 로딩
        if self.embedding_model is None:
            logger.info(f"Loading embedding model: {config['embedding_model']}")
            self.embedding_model = SentenceTransformer(
                config['embedding_model'],
                device=device
            )

        # Cross-encoder 로딩
        if self.cross_encoder is None:
            logger.info(f"Loading cross-encoder: {config['cross_encoder_model']}")
            self.cross_encoder = CrossEncoder(
                config['cross_encoder_model'],
                device=device
            )

        logger.info("All models loaded successfully")

    async def get_embedding_model(self) -> SentenceTransformer:
        """임베딩 모델 반환"""
        if self.embedding_model is None:
            raise RuntimeError("Embedding model not loaded. Call load_models() first.")
        return self.embedding_model

    async def get_cross_encoder(self) -> CrossEncoder:
        """Cross-encoder 모델 반환"""
        if self.cross_encoder is None:
            raise RuntimeError("Cross-encoder not loaded. Call load_models() first.")
        return self.cross_encoder

    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        return {
            "embedding_model_loaded": self.embedding_model is not None,
            "cross_encoder_loaded": self.cross_encoder is not None,
            "config": self._config
        }