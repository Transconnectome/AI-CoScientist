"""Multi-task model with ordinal regression for all dimensions.

Improves upon standard multi-task model by:
1. Ordinal regression for all 5 quality dimensions
2. Better ranking consistency across all dimensions
3. Stronger regularization (dropout 0.4)
4. Expected QWK improvement: 0.000 → 0.20-0.40+
"""

from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np


class MultiTaskPaperScorerOrdinal(nn.Module):
    """Multi-task model with ordinal regression for all dimensions.

    Architecture:
    - Shared encoder: RoBERTa (768) + Linguistic (20) → 512 → 256
    - Per-dimension outputs (5 dimensions):
      * Score head: Direct prediction (1-10)
      * Ordinal head: 9 binary classifiers
    """

    def __init__(
        self,
        roberta_model: str = "roberta-base",
        dropout: float = 0.4,  # Increased regularization
        device: Optional[torch.device] = None
    ):
        """Initialize ordinal multi-task scorer.

        Args:
            roberta_model: HuggingFace model identifier
            dropout: Dropout rate (increased to 0.4)
            device: Computing device
        """
        super().__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.roberta_model_name = roberta_model

        # Lazy loading
        self._tokenizer = None
        self._roberta = None
        self._linguistic_extractor = None

        # Shared encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(788, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(512),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(256)
        ).to(self.device)

        # Dimension names
        self.dimension_names = ["overall", "novelty", "methodology", "clarity", "significance"]

        # Dual heads for each dimension
        self.score_heads = nn.ModuleDict({
            dim: nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1)
            ).to(self.device)
            for dim in self.dimension_names
        })

        self.ordinal_heads = nn.ModuleDict({
            dim: nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 9)  # 9 binary classifiers
            ).to(self.device)
            for dim in self.dimension_names
        })

        self.is_trained = False

    def _ensure_loaded(self):
        """Lazy load RoBERTa and linguistic extractor."""
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

    def forward(self, text: str) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with dual outputs per dimension.

        Args:
            text: Paper content

        Returns:
            Dict mapping dimension → (score_output, ordinal_logits)
        """
        # Extract representations
        roberta_embedding = self.extract_embeddings(text)
        linguistic_features = self.extract_linguistic_features(text)

        # Concatenate and encode
        combined = torch.cat([roberta_embedding, linguistic_features], dim=1)
        shared_repr = self.shared_encoder(combined)

        # Generate outputs for each dimension
        outputs = {}
        for dim in self.dimension_names:
            score_logits = self.score_heads[dim](shared_repr)
            ordinal_logits = self.ordinal_heads[dim](shared_repr)

            # Convert to 1-10 scale
            score_output = torch.sigmoid(score_logits) * 9 + 1

            outputs[dim] = (score_output, ordinal_logits)

        return outputs

    def predict_from_ordinal(self, ordinal_logits: torch.Tensor) -> torch.Tensor:
        """Convert ordinal logits to score.

        Args:
            ordinal_logits: (batch, 9) binary logits

        Returns:
            (batch,) predicted scores
        """
        probs = torch.sigmoid(ordinal_logits)
        predictions = torch.sum(probs > 0.5, dim=1).float() + 1.0
        predictions = torch.clamp(predictions, 1.0, 10.0)
        return predictions

    async def score_paper(self, text: str, use_ordinal: bool = False) -> Dict[str, float]:
        """Score paper across all dimensions.

        Args:
            text: Paper content
            use_ordinal: Use ordinal predictions

        Returns:
            Quality scores for all dimensions
        """
        self.eval()

        with torch.no_grad():
            outputs = self.forward(text)

            scores = {}
            for dim in self.dimension_names:
                score_output, ordinal_logits = outputs[dim]

                if use_ordinal:
                    score_value = self.predict_from_ordinal(ordinal_logits)[0].item()
                else:
                    score_value = score_output[0, 0].item()

                scores[f"{dim}_quality"] = round(score_value, 2)

            scores["model_type"] = "multitask_ordinal"
            scores["trained"] = self.is_trained
            scores["prediction_method"] = "ordinal" if use_ordinal else "direct"

        return scores

    def train_mode(self):
        """Set to training mode."""
        self.train()
        if self._roberta is not None:
            self._roberta.eval()

    def save_weights(self, path: str):
        """Save model weights."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Collect all state dicts
        state_dict = {
            'shared_encoder_state_dict': self.shared_encoder.state_dict(),
            'is_trained': self.is_trained,
            'roberta_model': self.roberta_model_name
        }

        for dim in self.dimension_names:
            state_dict[f'score_head_{dim}_state_dict'] = self.score_heads[dim].state_dict()
            state_dict[f'ordinal_head_{dim}_state_dict'] = self.ordinal_heads[dim].state_dict()

        torch.save(state_dict, path)

    def load_weights(self, path: str):
        """Load pre-trained weights."""
        checkpoint = torch.load(path, map_location=self.device)

        self.shared_encoder.load_state_dict(checkpoint['shared_encoder_state_dict'])

        for dim in self.dimension_names:
            self.score_heads[dim].load_state_dict(checkpoint[f'score_head_{dim}_state_dict'])
            self.ordinal_heads[dim].load_state_dict(checkpoint[f'ordinal_head_{dim}_state_dict'])

        self.is_trained = checkpoint.get('is_trained', True)
        self.eval()


class MultiTaskTrainerOrdinal:
    """Training pipeline for ordinal multi-task model."""

    def __init__(
        self,
        model: MultiTaskPaperScorerOrdinal,
        learning_rate: float = 5e-5,
        task_weights: Optional[Dict[str, float]] = None,
        mse_weight: float = 0.3,
        ordinal_weight: float = 0.7,
        device: Optional[torch.device] = None
    ):
        """Initialize ordinal multi-task trainer.

        Args:
            model: MultiTaskPaperScorerOrdinal instance
            learning_rate: Learning rate
            task_weights: Weights for each dimension
            mse_weight: Weight for MSE component
            ordinal_weight: Weight for ordinal component
            device: Computing device
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Default task weights
        if task_weights is None:
            task_weights = {
                "overall": 2.0,
                "novelty": 1.0,
                "methodology": 1.5,
                "clarity": 1.0,
                "significance": 1.5
            }
        self.task_weights = task_weights

        # Collect all parameters
        params = list(self.model.shared_encoder.parameters())
        for dim in self.model.dimension_names:
            params.extend(self.model.score_heads[dim].parameters())
            params.extend(self.model.ordinal_heads[dim].parameters())

        self.optimizer = torch.optim.AdamW(
            params,
            lr=learning_rate,
            weight_decay=0.01
        )

        # Hybrid ordinal loss for each dimension
        try:
            from src.services.paper.ordinal_loss import HybridOrdinalLoss
        except ModuleNotFoundError:
            from ordinal_loss import HybridOrdinalLoss

        self.criterion = HybridOrdinalLoss(
            num_classes=10,
            mse_weight=mse_weight,
            ordinal_weight=ordinal_weight
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )

    def train_epoch(
        self,
        train_data: List[Tuple[str, Dict[str, float]]],
        batch_size: int = 4
    ) -> float:
        """Train for one epoch.

        Args:
            train_data: List of (text, scores_dict) tuples
            batch_size: Batch size

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

            for text, target_scores in batch:
                # Forward pass
                outputs = self.model.forward(text)

                # Compute loss for each dimension
                for dim in self.model.dimension_names:
                    score_output, ordinal_logits = outputs[dim]
                    target = torch.tensor([target_scores[dim]], device=self.device)

                    loss = self.criterion(
                        score_output=score_output.squeeze(),
                        ordinal_logits=ordinal_logits,
                        targets=target
                    )

                    # Apply task weight
                    weighted_loss = loss * self.task_weights.get(dim, 1.0)
                    batch_loss += weighted_loss

            # Average and backprop
            batch_loss = batch_loss / len(batch)
            batch_loss.backward()

            # Gradient clipping
            params = list(self.model.shared_encoder.parameters())
            for dim in self.model.dimension_names:
                params.extend(self.model.score_heads[dim].parameters())
                params.extend(self.model.ordinal_heads[dim].parameters())

            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

            self.optimizer.step()

            total_loss += batch_loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def validate(
        self,
        val_data: List[Tuple[str, Dict[str, float]]]
    ) -> Dict[str, float]:
        """Validate model.

        Args:
            val_data: Validation data

        Returns:
            Validation metrics per dimension
        """
        self.model.eval()

        total_loss = 0.0
        predictions = {dim: [] for dim in self.model.dimension_names}
        targets = {dim: [] for dim in self.model.dimension_names}

        with torch.no_grad():
            for text, target_scores in val_data:
                outputs = self.model.forward(text)

                for dim in self.model.dimension_names:
                    score_output, ordinal_logits = outputs[dim]
                    target = torch.tensor([target_scores[dim]], device=self.device)

                    loss = self.criterion(
                        score_output=score_output.squeeze(),
                        ordinal_logits=ordinal_logits,
                        targets=target
                    )

                    weighted_loss = loss * self.task_weights.get(dim, 1.0)
                    total_loss += weighted_loss.item()

                    # Use ordinal prediction
                    pred_score = self.model.predict_from_ordinal(ordinal_logits)[0].item()
                    predictions[dim].append(pred_score)
                    targets[dim].append(target_scores[dim])

        # Compute metrics
        metrics = {'loss': total_loss / len(val_data)}

        for dim in self.model.dimension_names:
            preds = np.array(predictions[dim])
            targs = np.array(targets[dim])

            mae = np.mean(np.abs(preds - targs))
            corr = np.corrcoef(preds, targs)[0, 1] if len(preds) > 1 else 0.0

            metrics[f"{dim}_mae"] = float(mae)
            metrics[f"{dim}_correlation"] = float(corr)

        return metrics

    def train(
        self,
        train_data: List[Tuple[str, Dict[str, float]]],
        val_data: List[Tuple[str, Dict[str, float]]],
        epochs: int = 30,
        batch_size: int = 4,
        checkpoint_dir: str = "models/multitask_ordinal"
    ) -> Dict:
        """Full training loop.

        Args:
            train_data: Training data
            val_data: Validation data
            epochs: Number of epochs
            batch_size: Batch size
            checkpoint_dir: Checkpoint directory

        Returns:
            Training history
        """
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': []}

        print("=" * 80)
        print("MULTI-TASK ORDINAL MODEL TRAINING")
        print("=" * 80)
        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        print(f"Quality dimensions: {', '.join(self.model.dimension_names)}")
        print(f"Task weights: {self.task_weights}")
        print(f"Epochs: {epochs}")
        print(f"Device: {self.device}")
        print(f"Loss: HybridOrdinalLoss per dimension")
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

            # Print progress
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")

            for dim in self.model.dimension_names:
                mae = val_metrics[f"{dim}_mae"]
                corr = val_metrics[f"{dim}_correlation"]
                print(f"  {dim.capitalize():15s}: MAE={mae:.4f}, Corr={corr:.4f}")

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
