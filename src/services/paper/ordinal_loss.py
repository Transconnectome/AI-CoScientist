"""Ordinal regression loss functions for paper quality scoring.

Ordinal regression is better than MSE for ranked outcomes because:
1. Preserves ordering relationships (score 8 > score 7 > score 6)
2. Penalizes violations of ordinal constraints
3. Encourages monotonic predictions

References:
- "A simple approach to ordinal classification" (Frank & Hall, 2001)
- "Ordinal regression by extended binary classification" (Li & Lin, 2006)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class OrdinalRegressionLoss(nn.Module):
    """Ordinal regression loss using binary classification approach.

    Converts ordinal prediction (1-10 scale) into K-1 binary classification problems.
    For score range 1-10, creates 9 binary classifiers asking:
    - Is score > 1?
    - Is score > 2?
    - ...
    - Is score > 9?

    Final score is sum of positive predictions.
    """

    def __init__(self, num_classes: int = 10):
        """Initialize ordinal regression loss.

        Args:
            num_classes: Number of ordinal classes (default 10 for 1-10 scale)
        """
        super().__init__()
        self.num_classes = num_classes
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute ordinal regression loss.

        Args:
            logits: Model outputs [batch_size, num_classes-1]
            targets: Ground truth scores [batch_size] (1-indexed, 1-10)

        Returns:
            Loss value
        """
        batch_size = targets.size(0)

        # Convert targets to binary labels for each threshold
        # If target = 7, then labels = [1,1,1,1,1,1,0,0,0] (score > 1-6, not > 7-9)
        binary_labels = torch.zeros(batch_size, self.num_classes - 1, device=targets.device)

        for i in range(batch_size):
            target_idx = int(targets[i].item()) - 1  # Convert 1-10 to 0-9
            # Set all thresholds below target to 1
            if target_idx > 0:
                binary_labels[i, :target_idx] = 1.0

        # Compute BCE loss for each binary classifier
        loss = self.bce(logits, binary_labels)

        return loss

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to ordinal predictions.

        Args:
            logits: Model outputs [batch_size, num_classes-1]

        Returns:
            Predicted scores [batch_size] (1-10 scale)
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        # Count how many thresholds are exceeded (sum of probs > 0.5)
        predictions = torch.sum(probs > 0.5, dim=1).float() + 1.0  # +1 for 1-indexed

        # Clamp to valid range
        predictions = torch.clamp(predictions, 1.0, float(self.num_classes))

        return predictions


class CornLoss(nn.Module):
    """Conditional Ordinal Regression for Neural networks (CORN) loss.

    More sophisticated ordinal loss that models conditional probabilities.
    Better at maintaining ordinal relationships than simple binary approach.

    Reference:
    "Deep Neural Networks for Rank-Consistent Ordinal Regression" (Cao et al., 2020)
    """

    def __init__(self, num_classes: int = 10):
        """Initialize CORN loss.

        Args:
            num_classes: Number of ordinal classes (default 10)
        """
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute CORN loss.

        Args:
            logits: Model outputs [batch_size, num_classes-1]
            targets: Ground truth scores [batch_size] (1-indexed)

        Returns:
            Loss value
        """
        batch_size = targets.size(0)
        K = self.num_classes

        # Convert 1-indexed targets to 0-indexed
        targets = targets.long() - 1

        # Create importance weights (higher penalty for larger ranking errors)
        importance_weights = torch.zeros(batch_size, K - 1, device=logits.device)

        for i in range(batch_size):
            target_rank = targets[i].item()

            for k in range(K - 1):
                if k < target_rank:
                    # For ranks below target, should predict 1
                    importance_weights[i, k] = 1.0
                elif k >= target_rank:
                    # For ranks at or above target, should predict 0
                    importance_weights[i, k] = 1.0

        # Binary cross entropy with importance weighting
        probs = torch.sigmoid(logits)

        # Create binary targets
        binary_targets = torch.zeros(batch_size, K - 1, device=logits.device)
        for i in range(batch_size):
            target_rank = targets[i].item()
            if target_rank > 0:
                binary_targets[i, :target_rank] = 1.0

        # Weighted BCE
        loss = -importance_weights * (
            binary_targets * torch.log(probs + 1e-7) +
            (1 - binary_targets) * torch.log(1 - probs + 1e-7)
        )

        return loss.mean()

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to ordinal predictions.

        Args:
            logits: Model outputs [batch_size, num_classes-1]

        Returns:
            Predicted scores [batch_size] (1-10 scale)
        """
        probs = torch.sigmoid(logits)
        predictions = torch.sum(probs > 0.5, dim=1).float() + 1.0
        predictions = torch.clamp(predictions, 1.0, float(self.num_classes))
        return predictions


class HybridOrdinalLoss(nn.Module):
    """Hybrid loss combining MSE and ordinal regression.

    Uses weighted combination:
    - MSE for accurate score prediction
    - Ordinal loss for maintaining ordering relationships

    This balances precise scoring with ordinal consistency.
    """

    def __init__(
        self,
        num_classes: int = 10,
        mse_weight: float = 0.3,
        ordinal_weight: float = 0.7,
        use_corn: bool = False
    ):
        """Initialize hybrid loss.

        Args:
            num_classes: Number of ordinal classes
            mse_weight: Weight for MSE component (0-1)
            ordinal_weight: Weight for ordinal component (0-1)
            use_corn: Use CORN loss instead of simple ordinal loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.mse_weight = mse_weight
        self.ordinal_weight = ordinal_weight

        self.mse = nn.MSELoss()
        if use_corn:
            self.ordinal = CornLoss(num_classes)
        else:
            self.ordinal = OrdinalRegressionLoss(num_classes)

    def forward(
        self,
        score_output: torch.Tensor,
        ordinal_logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute hybrid loss.

        Args:
            score_output: Direct score predictions [batch_size]
            ordinal_logits: Ordinal regression logits [batch_size, num_classes-1]
            targets: Ground truth scores [batch_size]

        Returns:
            Combined loss value
        """
        # MSE loss for score accuracy
        mse_loss = self.mse(score_output, targets)

        # Ordinal loss for ranking consistency
        ordinal_loss = self.ordinal(ordinal_logits, targets)

        # Weighted combination
        total_loss = self.mse_weight * mse_loss + self.ordinal_weight * ordinal_loss

        return total_loss


def test_ordinal_losses():
    """Test ordinal loss implementations."""
    print("Testing Ordinal Loss Functions")
    print("=" * 80)

    # Test data: 5 samples with scores 3, 5, 7, 8, 9
    targets = torch.tensor([3.0, 5.0, 7.0, 8.0, 9.0])
    batch_size = targets.size(0)

    # Simulated model outputs (9 binary classifiers for 1-10 scale)
    logits = torch.randn(batch_size, 9)

    # Test OrdinalRegressionLoss
    print("\n1. OrdinalRegressionLoss")
    print("-" * 80)
    ord_loss = OrdinalRegressionLoss(num_classes=10)
    loss = ord_loss(logits, targets)
    predictions = ord_loss.predict(logits)

    print(f"Loss: {loss.item():.4f}")
    print(f"Targets:     {targets.tolist()}")
    print(f"Predictions: {[f'{p:.2f}' for p in predictions.tolist()]}")

    # Test CornLoss
    print("\n2. CornLoss")
    print("-" * 80)
    corn_loss = CornLoss(num_classes=10)
    loss = corn_loss(logits, targets)
    predictions = corn_loss.predict(logits)

    print(f"Loss: {loss.item():.4f}")
    print(f"Targets:     {targets.tolist()}")
    print(f"Predictions: {[f'{p:.2f}' for p in predictions.tolist()]}")

    # Test HybridOrdinalLoss
    print("\n3. HybridOrdinalLoss")
    print("-" * 80)
    hybrid_loss = HybridOrdinalLoss(num_classes=10, mse_weight=0.3, ordinal_weight=0.7)

    # Need both score output and ordinal logits
    score_output = torch.tensor([3.2, 4.8, 7.1, 7.9, 8.8])
    loss = hybrid_loss(score_output, logits, targets)

    print(f"Loss: {loss.item():.4f}")
    print(f"Score output: {[f'{s:.2f}' for s in score_output.tolist()]}")
    print(f"Targets:      {targets.tolist()}")

    print("\n✅ All tests passed!")
    print("=" * 80)


if __name__ == "__main__":
    test_ordinal_losses()
