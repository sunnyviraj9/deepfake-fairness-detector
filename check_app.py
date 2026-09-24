import sys
try:
    import torch
    print("torch OK")
    import torch.nn as nn
    import torch.nn.functional as F
    print("torch.nn OK")
    from torchvision import transforms
    print("torchvision OK")
    from PIL import Image
    print("PIL OK")
    import numpy as np
    print("numpy OK")
    import cv2
    print("cv2 OK")
    from models.spatial_detector import SpatialDetector
    print("SpatialDetector OK")
except Exception as e:
    print(f"IMPORT ERROR: {e}")
    sys.exit(1)

try:
    model = SpatialDetector()
    ckpt = torch.load('checkpoints/best_model.pt', map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print("Model load OK")
    print(f"  backbone: {ckpt.get('backbone')}")
    print(f"  image_size: {ckpt.get('args', {}).get('image_size')}")
    print(f"  val_auc: {ckpt.get('val_auc')}")
except Exception as e:
    print(f"MODEL ERROR: {e}")
    sys.exit(1)

# Test inference
import torch
dummy = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    out = model(dummy)
    prob = torch.sigmoid(out).item()
print(f"Inference OK — prob={prob:.4f}")
print("ALL CHECKS PASSED")
