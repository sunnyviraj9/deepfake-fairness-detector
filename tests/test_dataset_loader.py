import os
import sys
import shutil
import tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import pandas as pd
import torch
from utils.dataset_loader import DeepfakeFairnessDataset, create_dataloaders


def test_dataset_loading():
    temp_dir = tempfile.mkdtemp()
    try:
        # Create 10 dummy images and annotations
        img_paths = []
        for i in range(10):
            img_path = os.path.join(temp_dir, f"img_{i}.jpg")
            img = np.full((100, 100, 3), i * 20, dtype=np.uint8)
            cv2.imwrite(img_path, img)
            img_paths.append(img_path)

        df = pd.DataFrame({
            "image_path": img_paths,
            "target": [0, 1] * 5,
            "gender": ["Male", "Female"] * 5,
            "skin_tone": list(range(1, 11)),
        })
        csv_path = os.path.join(temp_dir, "annotations.csv")
        df.to_csv(csv_path, index=False)

        dataset = DeepfakeFairnessDataset(csv_path)
        assert len(dataset) == 10

        image, target, meta = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert target.item() == 0
        assert meta["skin_tone"] == 1
        assert meta["gender"] == "Male"
        assert meta["skin_tone_group"] == "Light (1-3)"

        # Test dataloader creation
        train_l, val_l, test_l, _ = create_dataloaders(
            annotations_path=csv_path,
            batch_size=2,
            val_split=0.2,
            test_split=0.2,
            num_workers=0,
        )
        assert len(train_l.dataset) == 6
        assert len(val_l.dataset) == 2
        assert len(test_l.dataset) == 2

    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_dataset_loading()
    print("All dataset loader unit tests passed successfully!")
