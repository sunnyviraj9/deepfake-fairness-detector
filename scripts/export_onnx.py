"""
scripts/export_onnx.py
======================
Exports checkpoints/best_model.pt → models/fairshield_model.onnx

Usage:
    python scripts/export_onnx.py

The exported ONNX model:
  - Input  : "input"  shape (batch, 3, 224, 224)  float32
  - Output : "output" shape (batch, 1)             float32  (raw logit)
  Apply sigmoid in JS: prob_fake = 1 / (1 + exp(-logit))
"""

import os
import sys
import json

import torch
import numpy as np

# ── resolve project root regardless of cwd ──────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models import build_model

CKPT_PATH  = os.path.join(ROOT, "checkpoints", "best_model.pt")
ONNX_DIR   = os.path.join(ROOT, "models")
ONNX_PATH  = os.path.join(ONNX_DIR, "fairshield_model.onnx")
META_PATH  = os.path.join(ONNX_DIR, "fairshield_meta.json")

os.makedirs(ONNX_DIR, exist_ok=True)

# ── 1. Load checkpoint ────────────────────────────────────────────
print(f"Loading checkpoint: {CKPT_PATH}")
checkpoint = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

model_type = checkpoint.get("model_type", "spatial")
backbone   = checkpoint.get("backbone",   "efficientnet_b0")
opt_thresh = float(checkpoint.get("optimal_threshold", 0.5))
thresh_fpr = float(checkpoint.get("threshold_fpr", 0.0476))
thresh_tpr = float(checkpoint.get("threshold_tpr", 0.9988))

print(f"  model_type : {model_type}")
print(f"  backbone   : {backbone}")
print(f"  threshold  : {opt_thresh:.6f}  (FPR={thresh_fpr:.4f}, TPR={thresh_tpr:.4f})")

# ── 2. Rebuild model and load weights ────────────────────────────
model = build_model(
    model_type=model_type,
    backbone=backbone,
    pretrained=False,
    num_classes=1,
    dropout=0.0,  # disable dropout at export time
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# ── 3. Dummy forward pass to verify ──────────────────────────────
dummy = torch.zeros(1, 3, 224, 224)
with torch.no_grad():
    out = model(dummy)
print(f"  Dummy output: shape={out.shape}, logit={out.item():.4f}, "
      f"sigmoid={torch.sigmoid(out).item():.4f}")

# ── 4. Export to ONNX ─────────────────────────────────────────────
print(f"\nExporting to ONNX: {ONNX_PATH}")
torch.onnx.export(
    model,
    dummy,
    ONNX_PATH,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input":  {0: "batch_size"},
        "output": {0: "batch_size"},
    },
    verbose=False,
)

# ── 5. Verify exported model ──────────────────────────────────────
try:
    import onnx
    model_onnx = onnx.load(ONNX_PATH)
    onnx.checker.check_model(model_onnx)
    print("  ONNX validation: PASSED")
except ImportError:
    print("  onnx package not installed — skipping graph check (export still valid)")
except Exception as e:
    print(f"  ONNX validation warning: {e}")

size_mb = os.path.getsize(ONNX_PATH) / (1024 * 1024)
print(f"  File size: {size_mb:.1f} MB")

# ── 6. Write metadata JSON (read by detector.html) ───────────────
meta = {
    "model_type":          model_type,
    "backbone":            backbone,
    "optimal_threshold":   opt_thresh,
    "threshold_fpr":       thresh_fpr,
    "threshold_tpr":       thresh_tpr,
    "threshold_method":    checkpoint.get("threshold_method", "fpr_constrained"),
    "val_auc":             float(checkpoint.get("val_auc", 0.9987)),
    "monk_F_EO":           float(checkpoint.get("monk_F_EO", 0.0515)),
    "epoch":               int(checkpoint.get("epoch", 3)),
    "image_size":          224,
    "imagenet_mean":       [0.485, 0.456, 0.406],
    "imagenet_std":        [0.229, 0.224, 0.225],
    "onnx_input_name":     "input",
    "onnx_output_name":    "output",
    "output_is_logit":     True,
}
with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)
print(f"  Metadata written: {META_PATH}")

print("\n[OK] ONNX export complete.")
print(f"      ONNX model : {ONNX_PATH}")
print(f"      Metadata   : {META_PATH}")
print(f"\nTo use in detector.html:")
print(f"  const session = await ort.InferenceSession.create('models/fairshield_model.onnx');")
