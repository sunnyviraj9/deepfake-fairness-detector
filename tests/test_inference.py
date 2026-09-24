"""
Integration test: verifies the trained checkpoint loads and produces
valid predictions on a synthetic image without needing real dataset files.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from models import build_model


CHECKPOINT = "checkpoints/best_model.pt"


def test_checkpoint_exists():
    assert os.path.exists(CHECKPOINT), f"Checkpoint not found: {CHECKPOINT}"


def test_checkpoint_loads():
    device = torch.device("cpu")
    ckpt = torch.load(CHECKPOINT, map_location=device)
    assert "model_state_dict" in ckpt, "Missing model_state_dict in checkpoint"
    assert "model_type" in ckpt
    assert "backbone" in ckpt
    assert "val_auc" in ckpt
    print(f"  model_type : {ckpt['model_type']}")
    print(f"  backbone   : {ckpt['backbone']}")
    print(f"  epoch      : {ckpt.get('epoch')}")
    print(f"  val_auc    : {ckpt.get('val_auc'):.4f}")


def test_inference_on_synthetic_image():
    """Model should produce a probability in [0, 1] for a random input."""
    device = torch.device("cpu")
    ckpt = torch.load(CHECKPOINT, map_location=device)

    model = build_model(
        model_type=ckpt["model_type"],
        backbone=ckpt["backbone"],
        pretrained=False,
        num_classes=1,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Synthetic batch of 4 images (224x224)
    dummy = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        logits = model(dummy).view(-1)
        probs = torch.sigmoid(logits).numpy()

    assert probs.shape == (4,), f"Unexpected output shape: {probs.shape}"
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0), "Probabilities out of [0,1]"
    print(f"  Sample probs: {probs.round(4)}")


def test_binary_classification_threshold():
    """Predictions should be deterministic and threshold at 0.5."""
    device = torch.device("cpu")
    ckpt = torch.load(CHECKPOINT, map_location=device)

    model = build_model(
        model_type=ckpt["model_type"],
        backbone=ckpt["backbone"],
        pretrained=False,
        num_classes=1,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    torch.manual_seed(0)
    dummy = torch.randn(8, 3, 224, 224)

    with torch.no_grad():
        probs1 = torch.sigmoid(model(dummy).view(-1)).numpy()
        probs2 = torch.sigmoid(model(dummy).view(-1)).numpy()

    np.testing.assert_array_equal(probs1, probs2, err_msg="Model is not deterministic in eval mode")
    preds = (probs1 >= 0.5).astype(int)
    assert set(preds).issubset({0, 1}), "Predictions should be binary (0 or 1)"


if __name__ == "__main__":
    test_checkpoint_exists()
    test_checkpoint_loads()
    test_inference_on_synthetic_image()
    test_binary_classification_threshold()
    print("\nAll inference integration tests passed!")
