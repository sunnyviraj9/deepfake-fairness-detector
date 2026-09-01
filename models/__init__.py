"""
Model architectures for Deepfake Detection.
"""

from typing import Optional
import torch.nn as nn
from models.spatial_detector import SpatialDetector
from models.frequency_detector import FrequencyDetector
from models.dual_stream_detector import DualStreamDetector


def build_model(
    model_type: str = "spatial",
    backbone: str = "efficientnet_b0",
    pretrained: bool = True,
    dropout: float = 0.3,
    num_classes: int = 1,
    **kwargs,
) -> nn.Module:
    """
    Factory function to instantiate models.

    Args:
        model_type: One of ['spatial', 'frequency', 'dual_stream'].
        backbone: timm backbone name.
        pretrained: Whether to load pretrained weights.
        dropout: Classifier dropout.
        num_classes: Output logits count (default 1 for binary classification).
    """
    model_type = model_type.lower()
    if model_type == "spatial":
        return SpatialDetector(
            backbone_name=backbone,
            pretrained=pretrained,
            num_classes=num_classes,
            dropout=dropout,
            **kwargs,
        )
    elif model_type == "frequency":
        return FrequencyDetector(
            backbone_name=backbone if backbone != "efficientnet_b0" else "resnet34",
            pretrained=pretrained,
            num_classes=num_classes,
            dropout=dropout,
            **kwargs,
        )
    elif model_type in ["dual_stream", "hybrid", "dual"]:
        return DualStreamDetector(
            spatial_backbone=backbone,
            frequency_backbone=kwargs.get("frequency_backbone", "resnet34"),
            pretrained=pretrained,
            num_classes=num_classes,
            dropout=dropout,
            high_pass_cutoff=kwargs.get("high_pass_cutoff", 0.1),
        )
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Choose from 'spatial', 'frequency', 'dual_stream'."
        )


__all__ = [
    "SpatialDetector",
    "FrequencyDetector",
    "DualStreamDetector",
    "build_model",
]
