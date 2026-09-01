"""
Dual-Stream Spatial and Frequency Multi-Modal Deepfake Detector.
Fuses RGB spatial texture features with 2D FFT spectral signatures.
"""

from typing import Optional
import torch
import torch.nn as nn
from models.spatial_detector import SpatialDetector
from models.frequency_detector import FrequencyDetector


class DualStreamDetector(nn.Module):
    """
    Two-stream deepfake detector integrating spatial and frequency domain representations.
    """

    def __init__(
        self,
        spatial_backbone: str = "efficientnet_b0",
        frequency_backbone: str = "resnet34",
        pretrained: bool = True,
        num_classes: int = 1,
        dropout: float = 0.3,
        high_pass_cutoff: float = 0.1,
    ):
        super().__init__()
        self.spatial_stream = SpatialDetector(
            backbone_name=spatial_backbone,
            pretrained=pretrained,
            num_classes=num_classes,
            dropout=dropout,
        )
        self.frequency_stream = FrequencyDetector(
            backbone_name=frequency_backbone,
            pretrained=pretrained,
            num_classes=num_classes,
            dropout=dropout,
            high_pass_cutoff=high_pass_cutoff,
        )

        spatial_dim = self.spatial_stream.backbone.num_features
        freq_dim = self.frequency_stream.backbone.num_features
        fused_dim = spatial_dim + freq_dim

        # Gated Multi-Modal Fusion Head
        self.gate = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.Sigmoid(),
        )

        self.fusion_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(fused_dim, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(p=dropout / 2.0),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input RGB tensor (B, 3, H, W).
        Returns:
            Fused binary classification logits (B, 1).
        """
        spatial_feat = self.spatial_stream.extract_features(x)
        freq_feat = self.frequency_stream.extract_features(x)

        fused = torch.cat([spatial_feat, freq_feat], dim=1)
        gated_fused = fused * self.gate(fused)
        logits = self.fusion_head(gated_fused)
        return logits
