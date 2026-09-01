import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from models import build_model
from models.spatial_detector import SpatialDetector
from models.frequency_detector import FrequencyDetector
from models.dual_stream_detector import DualStreamDetector


def test_spatial_detector():
    model = SpatialDetector(backbone_name="resnet18", pretrained=False, num_classes=1)
    dummy_input = torch.randn(2, 3, 224, 224)
    logits = model(dummy_input)
    assert logits.shape == (2, 1)
    features = model.extract_features(dummy_input)
    assert features.ndim == 2 and features.shape[0] == 2


def test_frequency_detector():
    model = FrequencyDetector(backbone_name="resnet18", pretrained=False, num_classes=1)
    dummy_input = torch.randn(2, 3, 224, 224)
    logits = model(dummy_input)
    assert logits.shape == (2, 1)


def test_dual_stream_detector():
    model = DualStreamDetector(
        spatial_backbone="resnet18",
        frequency_backbone="resnet18",
        pretrained=False,
        num_classes=1,
    )
    dummy_input = torch.randn(2, 3, 224, 224)
    logits = model(dummy_input)
    assert logits.shape == (2, 1)


def test_model_factory():
    m1 = build_model("spatial", backbone="resnet18", pretrained=False)
    assert isinstance(m1, SpatialDetector)

    m2 = build_model("frequency", backbone="resnet18", pretrained=False)
    assert isinstance(m2, FrequencyDetector)

    m3 = build_model("dual_stream", backbone="resnet18", pretrained=False)
    assert isinstance(m3, DualStreamDetector)


if __name__ == "__main__":
    test_spatial_detector()
    test_frequency_detector()
    test_dual_stream_detector()
    test_model_factory()
    print("All model architecture unit tests passed successfully!")
