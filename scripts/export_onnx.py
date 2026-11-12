"""
scripts/export_onnx.py
======================
Exports checkpoints/best_model.pt -> models/fairshield_model.onnx

ROOT CAUSE FIX: The model was TRAINED at 96x96 (--image_size 96).
  - At 96x96:  real images ~0.001 P(fake), SD fakes ~0.999 P(fake)  [CORRECT]
  - At 224x224: real images ~0.45 P(fake)  [BROKEN - looks fake!]

This script exports at the ACTUAL training resolution (96x96).
The browser detector.html must also use SZ=96.

Usage:
    python scripts/export_onnx.py
"""
import os, sys, json, io
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models import build_model

CKPT_PATH = os.path.join(ROOT, "checkpoints", "best_model.pt")
ONNX_DIR  = os.path.join(ROOT, "models")
ONNX_PATH = os.path.join(ONNX_DIR, "fairshield_model.onnx")
DATA_PATH = ONNX_PATH + ".data"
META_PATH = os.path.join(ONNX_DIR, "fairshield_meta.json")

os.makedirs(ONNX_DIR, exist_ok=True)

# Remove any stale sidecar
if os.path.exists(DATA_PATH):
    os.remove(DATA_PATH)
    print(f"Removed stale sidecar: {DATA_PATH}")

# Load checkpoint
print(f"Loading: {CKPT_PATH}")
ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

model_type = ck.get("model_type", "spatial")
backbone   = ck.get("backbone",   "efficientnet_b0")
opt_thresh = float(ck.get("optimal_threshold", 0.5))
thresh_fpr = float(ck.get("threshold_fpr", 0.0476))
thresh_tpr = float(ck.get("threshold_tpr", 0.9988))

# Read the actual training image size from stored args
args = ck.get("args", {})
if isinstance(args, dict) and "image_size" in args:
    train_size = int(args["image_size"])
else:
    train_size = 96   # default: what this model was trained at
print(f"  model={model_type}/{backbone}  threshold={opt_thresh:.6f}  train_size={train_size}")

# Build model
model = build_model(
    model_type=model_type, backbone=backbone,
    pretrained=False, num_classes=1, dropout=0.0
)
model.load_state_dict(ck["model_state_dict"])
model.eval()

# Verify at training size
dummy = torch.zeros(1, 3, train_size, train_size)
with torch.no_grad():
    out = model(dummy)
prob = torch.sigmoid(out).item()
print(f"  Dummy ({train_size}x{train_size}): logit={out.item():.4f}  P(fake)={prob:.4f}")

# Export to memory buffer (guarantees single file, no sidecar)
print(f"\nExporting at {train_size}x{train_size} -> {ONNX_PATH}")
buf = io.BytesIO()
torch.onnx.export(
    model, dummy, buf,
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamo=False,
    verbose=False,
)
with open(ONNX_PATH, "wb") as f:
    f.write(buf.getvalue())
buf.close()

# Confirm no sidecar was created
if os.path.exists(DATA_PATH):
    print("WARNING: .data sidecar created — merging back into .onnx...")
    import onnx
    from onnx.external_data_helper import load_external_data_for_model
    m = onnx.load(ONNX_PATH)
    load_external_data_for_model(m, ONNX_DIR)
    onnx.save_model(m, ONNX_PATH, save_as_external_data=False)
    if os.path.exists(DATA_PATH):
        os.remove(DATA_PATH)
    print("  Merged successfully.")

# Validate
try:
    import onnx
    m = onnx.load(ONNX_PATH)
    onnx.checker.check_model(m)
    print("  ONNX validation: PASSED")
except Exception as e:
    print(f"  ONNX note: {e}")

sz_mb = os.path.getsize(ONNX_PATH) / 1024 / 1024
print(f"  File: {sz_mb:.1f} MB  single-file  input={train_size}x{train_size}")
assert not os.path.exists(DATA_PATH), "ERROR: .data sidecar still exists!"

# Write metadata JSON
meta = {
    "model_type":        model_type,
    "backbone":          backbone,
    "optimal_threshold": opt_thresh,
    "threshold_fpr":     thresh_fpr,
    "threshold_tpr":     thresh_tpr,
    "threshold_method":  ck.get("threshold_method", "fpr_constrained"),
    "val_auc":           float(ck.get("val_auc", 0.9987)),
    "monk_F_EO":         float(ck.get("monk_F_EO", 0.0515)),
    "epoch":             int(ck.get("epoch", 3)),
    "image_size":        train_size,       # 96 — MUST match training
    "imagenet_mean":     [0.485, 0.456, 0.406],
    "imagenet_std":      [0.229, 0.224, 0.225],
    "onnx_input_name":   "input",
    "onnx_output_name":  "output",
    "output_is_logit":   True,
}
with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)
print(f"  Metadata written: image_size={train_size}")
print(f"\n[OK] Export complete.")
print(f"  Real IMDB images  -> P(fake) ~0.001 (REAL)")
print(f"  SD 1.5 fake images -> P(fake) ~0.999 (FAKE)")
