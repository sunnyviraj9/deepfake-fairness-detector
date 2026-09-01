"""
Spatial Deepfake Detector using timm Backbones (EfficientNet, Xception, ResNet, ConvNeXt).
"""

from typing import Optional, Tuple
import timm
import torch
import torch.nn as nn


class SpatialDetector(nn.Module):
    """
    Spatial Domain Deepfake Detection Model.
    Utilizes a fine-tuned CNN/Transformer backbone via timm with a customized classification head.
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        num_classes: int = 1,
        dropout: float = 0.3,
        in_channels: int = 3,
        freeze_backbone_stages: int = 0,
    ):
        """
        Args:
            backbone_name: Name of timm backbone (e.g. 'efficientnet_b0', 'efficientnet_b4', 'legacy_xception', 'resnet50').
            pretrained: Whether to load ImageNet pretrained weights.
            num_classes: Output dimension (1 for binary logits).
            dropout: Dropout probability in classification head.
            in_channels: Input channels (default 3 for RGB).
            freeze_backbone_stages: Number of initial layers to freeze (for transfer learning).
        """
        super().__init__()
        self.backbone_name = backbone_name

        # Initialize backbone with num_classes=0 to get raw pooled features
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=in_channels,
            drop_rate=dropout,
        )

        in_features = self.backbone.num_features

        # Custom classification head
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(p=dropout / 2.0),
            nn.Linear(256, num_classes),
        )

        self._init_head_weights()

        if freeze_backbone_stages > 0:
            self._freeze_stages(freeze_backbone_stages)

    def _init_head_weights(self):
        """Xavier initialization for classifier head."""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _freeze_stages(self, num_stages: int):
        """Freezes first N parameter blocks of the backbone."""
        params = list(self.backbone.parameters())
        freeze_count = min(num_stages * 10, len(params) // 2)
        for p in params[:freeze_count]:
            p.requires_grad = False

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts pooled embedding representation of shape (B, feature_dim)."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input RGB tensor of shape (B, 3, H, W).
        Returns:
            Logits of shape (B, 1).
        """
        features = self.backbone(x)
        logits = self.head(features)
        return logits
