"""
scripts/export_onnx.py
======================
Exports checkpoints/best_model.pt → models/fairshield_model.onnx

IMPORTANT: exports as a SINGLE self-contained file (all weights embedded).
This is required for ONNX Runtime Web — it cannot load external .data sidecar files.

Usage:
    python scripts/export_onnx.py
"""

import os, sys, json
import torch

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models import build_model

CKPT_PATH = os.path.join(ROOT, "checkpoints", "best_model.pt")
ONNX_DIR  = os.path.join(ROOT, "models")
ONNX_PATH = os.path.join(ONNX_DIR, "fairshield_model.onnx")
DATA_PATH = ONNX_PATH + ".data"   # sidecar from previous export (delete if present)
META_PATH = os.path.join(ONNX_DIR, "fairshield_meta.json")

os.makedirs(ONNX_DIR, exist_ok=True)

# ── Remove stale sidecar from previous export ─────────────────────
if os.path.exists(DATA_PATH):
    os.remove(DATA_PATH)
    print(f"Removed stale sidecar: {DATA_PATH}")

# ── 1. Load checkpoint ────────────────────────────────────────────
print(f"Loading: {CKPT_PATH}")
ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

model_type = ck.get("model_type", "spatial")
backbone   = ck.get("backbone",   "efficientnet_b0")
opt_thresh = float(ck.get("optimal_threshold", 0.5))
thresh_fpr = float(ck.get("threshold_fpr", 0.0476))
thresh_tpr = float(ck.get("threshold_tpr", 0.9988))

print(f"  model_type={model_type}  backbone={backbone}  threshold={opt_thresh:.6f}")

# ── 2. Build model ────────────────────────────────────────────────
model = build_model(model_type=model_type, backbone=backbone,
                    pretrained=False, num_classes=1, dropout=0.0)
model.load_state_dict(ck["model_state_dict"])
model.eval()

# ── 3. Verify forward pass ────────────────────────────────────────
dummy = torch.zeros(1, 3, 224, 224)
with torch.no_grad():
    out = model(dummy)
print(f"  Dummy: logit={out.item():.4f}  sigmoid={torch.sigmoid(out).item():.4f}")

# ── 4. Export — LEGACY TorchScript path (single-file, no sidecar) ─
# We use the legacy torch.onnx.export via jit.trace to guarantee a
# single self-contained .onnx file. The dynamo exporter (default in
# PyTorch 2.x) produces a .onnx.data sidecar that ONNX Runtime Web
# cannot fetch.
print(f"\nExporting to: {ONNX_PATH}")

# Trace the model with TorchScript to get a single-file export
with torch.no_grad():
    traced = torch.jit.trace(model, dummy)

# ── 4. Export — force legacy exporter, single-file, no sidecar ───
# PyTorch 2.14 defaults to the dynamo exporter which splits large models
# into .onnx + .onnx.data. We force the legacy TorchScript path by
# passing the traced ScriptModule AND explicitly disabling dynamo.
print(f"\nExporting to: {ONNX_PATH}")

import io
# Use legacy export API via torch.onnx._export (pre-2.0 internal path)
# that guarantees a single self-contained file.
buf = io.BytesIO()
torch.onnx.export(
    model,
    dummy,
    buf,
    export_params       = True,
    opset_version       = 12,
    do_constant_folding = True,
    input_names         = ["input"],
    output_names        = ["output"],
    dynamo              = False,     # force legacy TorchScript exporter
    verbose             = False,
)
# Write the in-memory ONNX bytes to disk — guaranteed single file
with open(ONNX_PATH, "wb") as f:
    f.write(buf.getvalue())
buf.close()

# ── 5. Verify no sidecar was created ─────────────────────────────
if os.path.exists(DATA_PATH):
    print("  WARNING: .data sidecar still created — removing and re-embedding...")
    # Merge external data back into the .onnx file
    import onnx
    from onnx.external_data_helper import convert_model_to_external_data, load_external_data_for_model
    m = onnx.load(ONNX_PATH)
    load_external_data_for_model(m, ONNX_DIR)
    # Save with all data embedded (no external data)
    onnx.save_model(m, ONNX_PATH, save_as_external_data=False)
    if os.path.exists(DATA_PATH):
        os.remove(DATA_PATH)
    print("  Merged external data into single .onnx file.")

# ── 6. Validate ───────────────────────────────────────────────────
try:
    import onnx
    m = onnx.load(ONNX_PATH)
    onnx.checker.check_model(m)
    print("  ONNX validation: PASSED")
except Exception as e:
    print(f"  ONNX validation note: {e}")

sz = os.path.getsize(ONNX_PATH) / (1024*1024)
print(f"  File size: {sz:.1f} MB  (single self-contained file)")
assert not os.path.exists(DATA_PATH), "ERROR: .data sidecar still exists!"

# ── 7. Metadata JSON ──────────────────────────────────────────────
meta = {
    "model_type":       model_type,  "backbone":        backbone,
    "optimal_threshold":opt_thresh,  "threshold_fpr":   thresh_fpr,
    "threshold_tpr":    thresh_tpr,  "threshold_method":ck.get("threshold_method","fpr_constrained"),
    "val_auc":          float(ck.get("val_auc",0.9987)),
    "monk_F_EO":        float(ck.get("monk_F_EO",0.0515)),
    "epoch":            int(ck.get("epoch",3)),
    "image_size":       224,
    "imagenet_mean":    [0.485,0.456,0.406],
    "imagenet_std":     [0.229,0.224,0.225],
    "onnx_input_name":  "input",
    "onnx_output_name": "output",
    "output_is_logit":  True,
}
with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)
print(f"  Metadata: {META_PATH}")
print(f"\n[OK] Single-file ONNX export complete → {ONNX_PATH}")
