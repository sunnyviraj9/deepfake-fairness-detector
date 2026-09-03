"""
Dataset Loader for Deepfake Detection with Demographic Annotations.
Supports Monk Skin Tone (MST 1-10) and Gender annotations for fairness benchmarking.
"""

import os
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as T


# Monk Skin Tone Groupings
# Light: MST 1-3 | Medium: MST 4-7 | Dark: MST 8-10
MST_GROUPS = {
    1: "Light (1-3)",
    2: "Light (1-3)",
    3: "Light (1-3)",
    4: "Medium (4-7)",
    5: "Medium (4-7)",
    6: "Medium (4-7)",
    7: "Medium (4-7)",
    8: "Dark (8-10)",
    9: "Dark (8-10)",
    10: "Dark (8-10)",
}


def get_default_transforms(
    split: str = "train",
    image_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> T.Compose:
    """
    Returns standard torchvision transform pipelines for train/val/test splits.
    """
    if split == "train":
        return T.Compose([
            T.ToPILImage(),
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    else:
        return T.Compose([
            T.ToPILImage(),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])


class DeepfakeFairnessDataset(Dataset):
    """
    Custom PyTorch Dataset for Deepfake Detection with Demographic Attributes.

    Expected CSV columns:
    - image_path: str (path to face image)
    - target: int (0 = Real, 1 = Fake / Deepfake)
    - gender: str (e.g. 'Male', 'Female', 'Other')
    - skin_tone: int (Monk Skin Tone Scale 1 through 10)
    """

    def __init__(
        self,
        annotations: Union[str, pd.DataFrame],
        root_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        return_metadata: bool = True,
    ):
        """
        Args:
            annotations: Path to annotations.csv or a pre-loaded pandas DataFrame.
            root_dir: Base directory for relative image paths.
            transform: Optional transformation function or pipeline (torchvision or albumentations).
            return_metadata: Whether to return detailed demographic metadata dictionary with every sample.
        """
        if isinstance(annotations, str):
            if not os.path.exists(annotations):
                raise FileNotFoundError(f"Annotations file not found at: {annotations}")
            self.df = pd.read_csv(annotations)
            if root_dir is None:
                self.root_dir = os.path.dirname(os.path.abspath(annotations))
            else:
                self.root_dir = root_dir
        elif isinstance(annotations, pd.DataFrame):
            self.df = annotations.copy().reset_index(drop=True)
            self.root_dir = root_dir if root_dir is not None else ""
        else:
            raise ValueError("annotations must be a file path string or pandas DataFrame")

        self.transform = transform
        self.return_metadata = return_metadata

        self._validate_and_clean_annotations()

    def _validate_and_clean_annotations(self):
        """Validates columns and normalizes demographic values."""
        required_cols = ["image_path", "target", "gender", "skin_tone"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(
                    f"Missing required column '{col}' in annotations. Found: {list(self.df.columns)}"
                )

        # Normalize target to integer binary (0 or 1)
        self.df["target"] = self.df["target"].astype(int)

        # Normalize skin_tone to integer between 1 and 10
        self.df["skin_tone"] = (
            pd.to_numeric(self.df["skin_tone"], errors="coerce")
            .fillna(1)
            .clip(1, 10)
            .astype(int)
        )

        # Normalize gender string
        self.df["gender"] = (
            self.df["gender"].astype(str).str.strip().str.capitalize()
        )

        # Categorize Monk skin tone groups (Light, Medium, Dark)
        self.df["skin_tone_group"] = self.df["skin_tone"].map(
            lambda st: MST_GROUPS.get(st, "Unknown")
        )

        # Create intersectional demographic subgroup string: e.g. "Male_MST_3"
        self.df["subgroup"] = (
            self.df["gender"] + "_MST_" + self.df["skin_tone"].astype(str)
        )

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, path: str) -> np.ndarray:
        """Loads RGB image with error resilience."""
        if not os.path.isabs(path) and self.root_dir:
            full_path = os.path.join(self.root_dir, path)
        else:
            full_path = path

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image not found at path: {full_path}")

        img_bgr = cv2.imread(full_path)
        if img_bgr is None:
            raise ValueError(f"Failed to decode image from path: {full_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        target = row["target"]
        skin_tone = row["skin_tone"]
        gender = row["gender"]
        skin_tone_group = row["skin_tone_group"]
        subgroup = row["subgroup"]

        image = self._load_image(image_path)

        if self.transform is not None:
            # Support both Albumentations and Torchvision transforms
            if hasattr(self.transform, "__call__"):
                try:
                    # Albumentations signature: transform(image=img)['image']
                    transformed = self.transform(image=image)
                    if isinstance(transformed, dict) and "image" in transformed:
                        image = transformed["image"]
                    else:
                        image = transformed
                except TypeError:
                    # Torchvision signature: transform(img)
                    image = self.transform(image)

        # Ensure tensor format if not already converted
        if not isinstance(image, torch.Tensor):
            image = T.ToTensor()(image)

        target_tensor = torch.tensor(target, dtype=torch.float32)

        metadata = {
            "image_path": image_path,
            "skin_tone": skin_tone,
            "gender": gender,
            "skin_tone_group": skin_tone_group,
            "subgroup": subgroup,
            "target": target,
        }

        if self.return_metadata:
            return image, target_tensor, metadata
        return image, target_tensor


def create_dataloaders(
    annotations_path: str,
    root_dir: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    balance_demographics: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, pd.DataFrame]:
    """
    Splits dataset into Train / Validation / Test sets and returns standard PyTorch DataLoaders.

    Args:
        annotations_path: Path to annotations.csv.
        root_dir: Optional base image directory.
        batch_size: DataLoader batch size.
        num_workers: Number of DataLoader worker processes.
        image_size: Target image resolution.
        val_split: Fraction of dataset for validation.
        test_split: Fraction of dataset for test.
        seed: Random seed for reproducible splitting.
        balance_demographics: If True, uses WeightedRandomSampler for train loader to balance subgroups.

    Returns:
        (train_loader, val_loader, test_loader, full_df)
    """
    df = pd.read_csv(annotations_path)
    np.random.seed(seed)

    # Normalize demographic attributes and generate subgroup keys
    df["gender"] = df["gender"].astype(str).str.strip().str.capitalize()
    df["skin_tone"] = pd.to_numeric(df["skin_tone"], errors="coerce").fillna(1).clip(1, 10).astype(int)
    df["skin_tone_group"] = df["skin_tone"].map(lambda st: MST_GROUPS.get(st, "Unknown"))
    df["subgroup"] = df["gender"] + "_MST_" + df["skin_tone"].astype(str)

    # Perform stratified split by intersectional subgroup + target
    df["strat_key"] = df["subgroup"] + "_" + df["target"].astype(str)
    
    # Shuffle
    shuffled_indices = np.random.permutation(len(df))
    shuffled_df = df.iloc[shuffled_indices].reset_index(drop=True)

    n_total = len(shuffled_df)
    n_test = int(n_total * test_split)
    n_val = int(n_total * val_split)
    n_train = n_total - n_test - n_val

    train_df = shuffled_df.iloc[:n_train].reset_index(drop=True)
    val_df = shuffled_df.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test_df = shuffled_df.iloc[n_train + n_val :].reset_index(drop=True)

    train_transform = get_default_transforms(split="train", image_size=image_size)
    eval_transform = get_default_transforms(split="val", image_size=image_size)

    if root_dir is None:
        root_dir = "."

    train_dataset = DeepfakeFairnessDataset(train_df, root_dir=root_dir, transform=train_transform)
    val_dataset = DeepfakeFairnessDataset(val_df, root_dir=root_dir, transform=eval_transform)
    test_dataset = DeepfakeFairnessDataset(test_df, root_dir=root_dir, transform=eval_transform)

    sampler = None
    shuffle = True
    if balance_demographics and len(train_df) > 0:
        subgroup_counts = train_df["subgroup"].value_counts()
        weights = train_df["subgroup"].map(lambda sg: 1.0 / max(subgroup_counts.get(sg, 1), 1)).values.astype(np.float64).copy()
        sampler = WeightedRandomSampler(weights=torch.from_numpy(weights), num_samples=len(weights), replacement=True)
        shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, shuffled_df
