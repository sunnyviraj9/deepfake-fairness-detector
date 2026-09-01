"""
Synthetic Dataset Generator for Deepfake Detection and Fairness Benchmarking.
Generates face-like structured test images with diverse Monk Skin Tones (1-10) and manipulation artifacts.
"""

import argparse
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# Monk Skin Tone RGB Base Approximations (Monk Scale 1-10)
MST_PALETTES = {
    1: (246, 237, 228),
    2: (243, 231, 219),
    3: (247, 219, 196),
    4: (234, 192, 158),
    5: (219, 160, 126),
    6: (174, 124, 95),
    7: (140, 94, 69),
    8: (92, 60, 48),
    9: (59, 39, 34),
    10: (38, 25, 23),
}

GENDERS = ["Male", "Female"]


def generate_face_canvas(
    skin_tone: int,
    is_fake: bool,
    image_size: int = 224,
) -> np.ndarray:
    """
    Generates a synthetic portrait canvas with skin tone base and subtle frequency/spatial artifacts if fake.
    """
    canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    # Background gradient
    bg_color = np.random.randint(40, 80, size=3)
    canvas[:, :] = bg_color

    # Face base oval
    base_rgb = MST_PALETTES.get(skin_tone, (180, 130, 100))
    # Add minor random tone jitter
    tone_jitter = np.random.randint(-10, 11, size=3)
    face_color = np.clip(np.array(base_rgb) + tone_jitter, 0, 255).astype(int).tolist()

    # Draw face oval
    center = (image_size // 2, int(image_size * 0.52))
    axes = (int(image_size * 0.32), int(image_size * 0.42))
    cv2.ellipse(canvas, center, axes, 0, 0, 360, face_color, -1)

    # Smooth face boundary
    canvas = cv2.GaussianBlur(canvas, (9, 9), 3)

    # Facial features (Eyes, Mouth)
    eye_color = (30, 20, 20)
    left_eye = (int(image_size * 0.38), int(image_size * 0.45))
    right_eye = (int(image_size * 0.62), int(image_size * 0.45))
    cv2.circle(canvas, left_eye, int(image_size * 0.04), eye_color, -1)
    cv2.circle(canvas, right_eye, int(image_size * 0.04), eye_color, -1)

    mouth_center = (image_size // 2, int(image_size * 0.72))
    cv2.ellipse(canvas, mouth_center, (int(image_size * 0.12), int(image_size * 0.04)), 0, 0, 180, (150, 50, 50), -1)

    # If deepfake (is_fake = True), inject synthetic boundary and spectral artifacts
    if is_fake:
        # Spatial artifact: subtle boundary blending mismatch or eye warping
        if np.random.rand() > 0.5:
            cv2.ellipse(canvas, center, (axes[0] - 8, axes[1] - 8), 0, 0, 360, (200, 200, 200), 1)
        
        # High-frequency artifact: periodic checkerboard grid noise (simulating GAN upsampling)
        y_grid, x_grid = np.ogrid[:image_size, :image_size]
        pattern = np.sin(x_grid * 0.8) * np.sin(y_grid * 0.8) * 12.0
        pattern_3ch = np.repeat(pattern[:, :, np.newaxis], 3, axis=2)
        canvas = np.clip(canvas.astype(np.float32) + pattern_3ch, 0, 255).astype(np.uint8)

    return canvas


def create_synthetic_dataset(
    output_dir: str = "./data/mock_dataset",
    num_samples: int = 200,
    image_size: int = 224,
    seed: int = 42,
) -> str:
    """
    Creates a synthetic deepfake dataset with balanced Monk Skin Tone and gender metadata.
    """
    np.random.seed(seed)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    records = []
    print(f"Generating {num_samples} synthetic face samples in '{output_dir}'...")

    for i in tqdm(range(num_samples)):
        # Stratified attribute assignment
        skin_tone = (i % 10) + 1  # Cycle 1 to 10
        gender = GENDERS[(i // 10) % len(GENDERS)]
        target = 1 if (i % 2 == 1) else 0  # Balanced 50% real / 50% fake

        img_filename = f"sample_{i:05d}_mst{skin_tone}_{gender.lower()}_t{target}.jpg"
        img_rel_path = os.path.join("images", img_filename)
        img_abs_path = os.path.join(images_dir, img_filename)

        canvas_rgb = generate_face_canvas(skin_tone, is_fake=(target == 1), image_size=image_size)
        canvas_bgr = cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_abs_path, canvas_bgr)

        records.append({
            "image_path": img_rel_path,
            "target": target,
            "gender": gender,
            "skin_tone": skin_tone,
        })

    annotations_csv_path = os.path.join(output_dir, "annotations.csv")
    df = pd.DataFrame(records)
    df.to_csv(annotations_csv_path, index=False)
    print(f"Dataset generated successfully! Annotations saved to: {annotations_csv_path}")
    print(f"Distribution summary:\n{df.groupby(['gender', 'skin_tone', 'target']).size().head(10)}")

    return annotations_csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic deepfake dataset with fairness annotations.")
    parser.add_argument("--output_dir", type=str, default="./data/mock_dataset", help="Output directory path")
    parser.add_argument("--num_samples", type=int, default=200, help="Total number of images to generate")
    parser.add_argument("--image_size", type=int, default=224, help="Target image resolution")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    create_synthetic_dataset(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        image_size=args.image_size,
        seed=args.seed,
    )
