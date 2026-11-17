"""PyTorch models: a compact EEGNet-inspired 1D CNN for raw clips.

Input: clips of shape (batch, n_channels, n_samples) = (B, 64, 320).
The CNN treats the 64 electrodes as input features and convolves over time.

Per-channel normalisation (mean/std per electrode) is NOT part of the model;
it is applied to the data beforehand by eegauth.train.PerChannelNormalizer,
which is fitted on TRAIN clips only.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["EEGCNNSmall", "get_device"]


def get_device() -> torch.device:
    """Pick the best available device: mps > cuda > cpu."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class EEGCNNSmall(nn.Module):
    """Compact 1D CNN: two temporal conv blocks + global average pooling.

    Architecture (n_filters=32 default):
        Conv1d(64 -> 32, k=32, 'same') -> BatchNorm -> ELU -> MaxPool(4)   # 320 -> 80
        Conv1d(32 -> 64, k=16, 'same') -> BatchNorm -> ELU -> MaxPool(4)   # 80  -> 20
        AdaptiveAvgPool1d(1) -> Flatten -> Dropout -> Linear(64 -> n_classes)
    ~99k parameters for 20 classes — deliberately small to keep training fast
    and overfitting visible.
    """

    def __init__(
        self,
        n_channels: int = 64,
        n_samples: int = 320,
        n_classes: int = 20,
        n_filters: int = 32,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.block1 = nn.Sequential(
            nn.Conv1d(n_channels, n_filters, kernel_size=32, padding="same"),
            nn.BatchNorm1d(n_filters),
            nn.ELU(),
            nn.MaxPool1d(4),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(n_filters, 2 * n_filters, kernel_size=16, padding="same"),
            nn.BatchNorm1d(2 * n_filters),
            nn.ELU(),
            nn.MaxPool1d(4),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(2 * n_filters, n_classes),
        )

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Embedding: pooled conv features, (B, 2*n_filters), before the head."""
        x = self.block1(x)
        x = self.block2(x)
        return self.head[:2](x)  # AdaptiveAvgPool1d + Flatten

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T) float32 -> (B, n_classes) logits."""
        return self.head[2:](self.features(x))  # Dropout + Linear
