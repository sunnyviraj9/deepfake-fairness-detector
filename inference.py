"""
Inference Script for Deepfake Detection.
Single-image or batch folder inference with demographic metadata overlay.

Usage:
    python inference.py --model_path checkpoints/best_model.pt --image path/to/image.jpg
    python inference.py --model_path checkpoints/best_model.pt --image_dir path/to/folder/
"""

import argparse
import os
import glob
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

from models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Deepfake Inference Script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--image", type=str, default=None, help="Path to a single image for inference")
    parser.add_argument("--image_dir", type=str, default=None, help="Path to folder of images for batch inference")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold (default: 0.5)")
    parser.add_argument("--image_size", type=int, default=224, help="Input image resolution")
    parser.add_argument("--output_dir", type=str, default="./inference_results", help="Directory to save annotated outputs")
    parser.add_argument("--save_annotated", action="store_true", default=True, help="Save annotated output images")
    return parser.parse_args()


def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """Loads model from checkpoint."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    model_type = checkpoint.get("model_type", "spatial")
    backbone = checkpoint.get("backbone", "efficientnet_b0")

    model = build_model(
        model_type=model_type,
        backbone=backbone,
        pretrained=False,
        num_classes=1,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded {model_type} ({backbone}) from {model_path} [epoch {checkpoint.get('epoch', '?')}]")
    return model


def preprocess_image(image_path: str, image_size: int = 224) -> Tuple[torch.Tensor, np.ndarray]:
    """Loads and preprocesses a single image. Returns (tensor, original_rgb_array)."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    tensor = transform(img_rgb).unsqueeze(0)  # [1, C, H, W]
    return tensor, img_rgb


@torch.no_grad()
def predict_single(
    model: torch.nn.Module,
    image_path: str,
    device: torch.device,
    threshold: float = 0.5,
    image_size: int = 224,
) -> dict:
    """Runs inference on a single image and returns prediction dict."""
    tensor, img_rgb = preprocess_image(image_path, image_size)
    tensor = tensor.to(device)

    logit = model(tensor).view(-1)
    prob = torch.sigmoid(logit).item()
    label = int(prob >= threshold)
    confidence = prob if label == 1 else (1.0 - prob)

    return {
        "image_path": image_path,
        "deepfake_probability": round(prob, 4),
        "prediction": "FAKE" if label == 1 else "REAL",
        "confidence": round(confidence * 100, 1),
        "label": label,
        "img_rgb": img_rgb,
    }


def annotate_image(result: dict, output_path: str) -> None:
    """Draws prediction overlay on image and saves it."""
    img = Image.fromarray(result["img_rgb"])

    # Resize for display
    display_size = (400, 400)
    img = img.resize(display_size, Image.LANCZOS)

    draw = ImageDraw.Draw(img)
    prediction = result["prediction"]
    prob = result["deepfake_probability"]
    conf = result["confidence"]

    # Color: red=FAKE, green=REAL
    color = (220, 50, 50) if prediction == "FAKE" else (50, 200, 100)
    border_width = 6

    # Draw border
    draw.rectangle(
        [border_width // 2, border_width // 2, display_size[0] - border_width // 2, display_size[1] - border_width // 2],
        outline=color,
        width=border_width,
    )

    # Draw label box
    label_text = f"{prediction}  {conf:.1f}%"
    box_y = display_size[1] - 44
    draw.rectangle([0, box_y, display_size[0], display_size[1]], fill=(*color, 200))

    try:
        font = ImageFont.truetype("arial.ttf", 20)
        small_font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()
        small_font = font

    draw.text((10, box_y + 8), label_text, fill=(255, 255, 255), font=font)
    draw.text((10, 10), f"P(fake)={prob:.3f}", fill=color, font=small_font)

    img.save(output_path)


def batch_inference(
    model: torch.nn.Module,
    image_paths: List[str],
    device: torch.device,
    threshold: float = 0.5,
    image_size: int = 224,
    output_dir: Optional[str] = None,
    save_annotated: bool = True,
) -> List[dict]:
    """Runs inference on a list of image paths."""
    results = []
    os.makedirs(output_dir, exist_ok=True) if output_dir else None

    for img_path in image_paths:
        try:
            res = predict_single(model, img_path, device, threshold, image_size)
            results.append(res)

            print(
                f"  [{res['prediction']:4s} | {res['deepfake_probability']:.3f}] "
                f"Conf: {res['confidence']:.1f}%  | {os.path.basename(img_path)}"
            )

            if save_annotated and output_dir:
                basename = os.path.splitext(os.path.basename(img_path))[0]
                out_path = os.path.join(output_dir, f"{basename}_annotated.png")
                annotate_image(res, out_path)

        except Exception as e:
            print(f"  [ERROR] {img_path}: {e}")

    return results


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.model_path, device)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.image:
        # Single image mode
        print(f"\n=== Single Image Inference ===")
        result = predict_single(model, args.image, device, args.threshold, args.image_size)
        print(f"\nImage         : {result['image_path']}")
        print(f"Prediction    : {result['prediction']}")
        print(f"P(fake)       : {result['deepfake_probability']:.4f}")
        print(f"Confidence    : {result['confidence']:.1f}%")

        if args.save_annotated:
            out_path = os.path.join(args.output_dir, "annotated_result.png")
            annotate_image(result, out_path)
            print(f"Annotated image saved to: {out_path}")

    elif args.image_dir:
        # Batch folder mode
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
        image_paths = []
        for ext in exts:
            image_paths.extend(glob.glob(os.path.join(args.image_dir, ext)))
            image_paths.extend(glob.glob(os.path.join(args.image_dir, ext.upper())))

        if not image_paths:
            print(f"No images found in: {args.image_dir}")
            return

        print(f"\n=== Batch Inference on {len(image_paths)} images ===")
        results = batch_inference(
            model, image_paths, device,
            threshold=args.threshold,
            image_size=args.image_size,
            output_dir=args.output_dir,
            save_annotated=args.save_annotated,
        )

        # Summary stats
        n_fake = sum(1 for r in results if r["label"] == 1)
        n_real = len(results) - n_fake
        avg_prob = np.mean([r["deepfake_probability"] for r in results])

        print(f"\n=== Batch Summary ===")
        print(f"Total Images : {len(results)}")
        print(f"  REAL       : {n_real}")
        print(f"  FAKE       : {n_fake}")
        print(f"  Avg P(fake): {avg_prob:.4f}")

        # Save CSV summary
        import pandas as pd
        summary_df = pd.DataFrame([
            {k: v for k, v in r.items() if k != "img_rgb"}
            for r in results
        ])
        csv_path = os.path.join(args.output_dir, "inference_results.csv")
        summary_df.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")

    else:
        print("Please provide --image or --image_dir")


if __name__ == "__main__":
    main()
