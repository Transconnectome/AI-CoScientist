"""Hybrid paper quality scorer with ordinal regression loss.

Improves upon standard hybrid model by:
1. Adding ordinal regression head for ranking consistency
2. Using HybridOrdinalLoss (MSE + Ordinal) for better QWK
3. Stronger regularization (dropout 0.4) to prevent overfitting
4. Expected QWK improvement: 0.000 → 0.15-0.30+
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np


class HybridPaperScorerOrdinal(nn.Module):
    """Hybrid model with ordinal regression for improved ranking.

    Architecture:
    - RoBERTa embeddings: 768-dim
    - Linguistic features: 20-dim
    - Fusion network: 788 → 512 → 256
    - Dual outputs:
      * Score head: Direct quality prediction (1-10)
      * Ordinal head: 9 binary classifiers for ordinal ranking
    """

    def __init__(
        self,
        roberta_model: str = "roberta-base",
        dropout: float = 0.4,  # Increased from 0.2 for better generalization
        device: Optional[torch.device] = None
    ):
        """Initialize ordinal hybrid scorer.

        Args:
            roberta_model: HuggingFace model identifier
            dropout: Dropout rate (increased to 0.4 for regularization)
            device: Computing device
        """
        super().__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.roberta_model_name = roberta_model

        # Lazy loading
        self._tokenizer = None
        self._roberta = None
        self._linguistic_extractor = None

        # Shared fusion network: 788 → 512 → 256
        self.fusion = nn.Sequential(
            nn.Linear(788, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(512),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(256)
        ).to(self.device)

        # Dual output heads
        # 1. Score head: Direct prediction (1-10 scale)
        self.score_head = nn.Linear(256, 1).to(self.device)

        # 2. Ordinal head: 9 binary classifiers (is score > 1?, > 2?, ..., > 9?)
        self.ordinal_head = nn.Linear(256, 9).to(self.device)

        self.is_trained = False

    def _ensure_loaded(self):
        """Lazy load RoBERTa and linguistic feature extractor."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer, AutoModel
            try:
                from src.services.paper.linguistic_features import LinguisticFeatureExtractor
            except ModuleNotFoundError:
                from linguistic_features import LinguisticFeatureExtractor

            self._tokenizer = AutoTokenizer.from_pretrained(self.roberta_model_name)
            self._roberta = AutoModel.from_pretrained(self.roberta_model_name).to(self.device)
            self._roberta.eval()

            self._linguistic_extractor = LinguisticFeatureExtractor()

    def extract_embeddings(self, text: str, max_length: int = 512) -> torch.Tensor:
        """Extract RoBERTa embeddings."""
        self._ensure_loaded()

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self._roberta(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :]

        return embedding

    def extract_linguistic_features(self, text: str) -> torch.Tensor:
        """Extract linguistic features."""
        self._ensure_loaded()
        features = self._linguistic_extractor.extract(text)
        return features.unsqueeze(0).to(self.device)

    def forward(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with dual outputs.

        Args:
            text: Paper content

        Returns:
            Tuple of (score_output, ordinal_logits)
            - score_output: (1, 1) direct score prediction
            - ordinal_logits: (1, 9) binary classification logits
        """
        # Extract representations
        roberta_embedding = self.extract_embeddings(text)
        linguistic_features = self.extract_linguistic_features(text)

        # Concatenate: (1, 788)
        combined = torch.cat([roberta_embedding, linguistic_features], dim=1)

        # Shared fusion: (1, 788) → (1, 256)
        fused = self.fusion(combined)

        # Dual outputs
        score_logits = self.score_head(fused)  # (1, 1)
        ordinal_logits = self.ordinal_head(fused)  # (1, 9)

        # Convert score to 1-10 range
        score_output = torch.sigmoid(score_logits) * 9 + 1

        return score_output, ordinal_logits

    def predict_from_ordinal(self, ordinal_logits: torch.Tensor) -> torch.Tensor:
        """Convert ordinal logits to score prediction.

        Args:
            ordinal_logits: (batch, 9) binary classification logits

        Returns:
            (batch,) predicted scores (1-10 scale)
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(ordinal_logits)

        # Count how many thresholds exceeded (sum of probs > 0.5)
        predictions = torch.sum(probs > 0.5, dim=1).float() + 1.0

        # Clamp to valid range
        predictions = torch.clamp(predictions, 1.0, 10.0)

        return predictions

    async def score_paper(self, text: str, use_ordinal: bool = False) -> Dict[str, float]:
        """Score paper quality.

        Args:
            text: Paper content
            use_ordinal: Use ordinal prediction instead of direct score

        Returns:
            Quality score dictionary
        """
        self.eval()

        with torch.no_grad():
            score_output, ordinal_logits = self.forward(text)

            if use_ordinal:
                # Use ordinal prediction
                score_value = self.predict_from_ordinal(ordinal_logits)[0].item()
            else:
                # Use direct score prediction
                score_value = score_output[0, 0].item()

        return {
            "overall_quality": round(score_value, 2),
            "model_type": "hybrid_ordinal",
            "trained": self.is_trained,
            "prediction_method": "ordinal" if use_ordinal else "direct"
        }

    def train_mode(self):
        """Set to training mode."""
        self.train()
        if self._roberta is not None:
            self._roberta.eval()  # Keep RoBERTa frozen

    def save_weights(self, path: str):
        """Save model weights."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'fusion_state_dict': self.fusion.state_dict(),
            'score_head_state_dict': self.score_head.state_dict(),
            'ordinal_head_state_dict': self.ordinal_head.state_dict(),
            'is_trained': self.is_trained,
            'roberta_model': self.roberta_model_name
        }, path)

    def load_weights(self, path: str):
        """Load pre-trained weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.fusion.load_state_dict(checkpoint['fusion_state_dict'])
        self.score_head.load_state_dict(checkpoint['score_head_state_dict'])
        self.ordinal_head.load_state_dict(checkpoint['ordinal_head_state_dict'])
        self.is_trained = checkpoint.get('is_trained', True)
        self.eval()


class HybridTrainerOrdinal:
    """Training pipeline for ordinal hybrid model.

    Uses HybridOrdinalLoss combining MSE and ordinal regression.
    """

    def __init__(
        self,
        model: HybridPaperScorerOrdinal,
        learning_rate: float = 5e-5,  # Reduced for finer optimization
        mse_weight: float = 0.3,
        ordinal_weight: float = 0.7,
        device: Optional[torch.device] = None
    ):
        """Initialize ordinal trainer.

        Args:
            model: HybridPaperScorerOrdinal instance
            learning_rate: Learning rate (reduced for stability)
            mse_weight: Weight for MSE loss (0.3)
            ordinal_weight: Weight for ordinal loss (0.7)
            device: Computing device
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Optimizer: train fusion + both heads
        self.optimizer = torch.optim.AdamW(
            list(self.model.fusion.parameters()) +
            list(self.model.score_head.parameters()) +
            list(self.model.ordinal_head.parameters()),
            lr=learning_rate,
            weight_decay=0.01
        )

        # Hybrid ordinal loss
        try:
            from src.services.paper.ordinal_loss import HybridOrdinalLoss
        except ModuleNotFoundError:
            from ordinal_loss import HybridOrdinalLoss

        self.criterion = HybridOrdinalLoss(
            num_classes=10,
            mse_weight=mse_weight,
            ordinal_weight=ordinal_weight,
            use_corn=False  # Use simple ordinal loss
        )

        # Scheduler with more aggressive patience
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )

    def train_epoch(
        self,
        train_data: list[Tuple[str, float]],
        batch_size: int = 4
    ) -> float:
        """Train for one epoch.

        Args:
            train_data: List of (text, score) tuples
            batch_size: Batch size (note: processing is sequential per batch)

        Returns:
            Average training loss
        """
        self.model.train_mode()
        total_loss = 0.0
        num_batches = 0

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]

            batch_loss = 0.0
            self.optimizer.zero_grad()

            for text, target_score in batch:
                # Forward pass
                score_output, ordinal_logits = self.model.forward(text)

                # Prepare targets
                target = torch.tensor([target_score], device=self.device)

                # Compute hybrid loss
                loss = self.criterion(
                    score_output=score_output.squeeze(),
                    ordinal_logits=ordinal_logits,
                    targets=target
                )

                batch_loss += loss

            # Average loss over batch
            batch_loss = batch_loss / len(batch)
            batch_loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(
                list(self.model.fusion.parameters()) +
                list(self.model.score_head.parameters()) +
                list(self.model.ordinal_head.parameters()),
                max_norm=1.0
            )

            self.optimizer.step()

            total_loss += batch_loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def validate(self, val_data: list[Tuple[str, float]]) -> Dict[str, float]:
        """Validate model.

        Args:
            val_data: Validation data

        Returns:
            Validation metrics
        """
        self.model.eval()

        total_loss = 0.0
        predictions = []
        targets = []

        with torch.no_grad():
            for text, target_score in val_data:
                score_output, ordinal_logits = self.model.forward(text)

                target = torch.tensor([target_score], device=self.device)

                loss = self.criterion(
                    score_output=score_output.squeeze(),
                    ordinal_logits=ordinal_logits,
                    targets=target
                )

                total_loss += loss.item()

                # Use ordinal prediction for validation
                pred_score = self.model.predict_from_ordinal(ordinal_logits)[0].item()
                predictions.append(pred_score)
                targets.append(target_score)

        # Compute metrics
        predictions = np.array(predictions)
        targets = np.array(targets)

        mae = np.mean(np.abs(predictions - targets))
        correlation = np.corrcoef(predictions, targets)[0, 1] if len(predictions) > 1 else 0.0

        return {
            'loss': total_loss / len(val_data),
            'mae': float(mae),
            'correlation': float(correlation)
        }

    def train(
        self,
        train_data: list[Tuple[str, float]],
        val_data: list[Tuple[str, float]],
        epochs: int = 30,  # Increased epochs for ordinal learning
        batch_size: int = 4,
        checkpoint_dir: str = "models/hybrid_ordinal"
    ) -> Dict:
        """Full training loop.

        Args:
            train_data: Training data
            val_data: Validation data
            epochs: Number of epochs
            batch_size: Batch size
            checkpoint_dir: Directory to save checkpoints

        Returns:
            Training history
        """
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        best_val_loss = float('inf')
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'val_correlation': []
        }

        print("=" * 80)
        print("HYBRID ORDINAL MODEL TRAINING")
        print("=" * 80)
        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        print(f"Epochs: {epochs}")
        print(f"Device: {self.device}")
        print(f"Loss: HybridOrdinalLoss (MSE 30% + Ordinal 70%)")
        print()

        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_data, batch_size)

            # Validate
            val_metrics = self.validate(val_data)

            # Update scheduler
            self.scheduler.step(val_metrics['loss'])

            # Save history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_metrics['loss'])
            history['val_mae'].append(val_metrics['mae'])
            history['val_correlation'].append(val_metrics['correlation'])

            # Print progress
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val MAE: {val_metrics['mae']:.4f}")
            print(f"  Val Correlation: {val_metrics['correlation']:.4f}")

            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_path = checkpoint_path / "best_model.pt"
                self.model.save_weights(str(best_path))
                print(f"  ✅ Saved best model (val_loss: {best_val_loss:.4f})")

            print()

        print("=" * 80)
        print("TRAINING COMPLETE")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print("=" * 80)

        self.model.is_trained = True
        return history
