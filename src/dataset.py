"""
Data Pipeline for CropGuardNet
===============================
Handles loading, preprocessing, augmentation, and splitting
for multi-crop leaf disease datasets.
"""

import os
import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit
from PIL import Image


# ──────────────────────────────────────────
# Preprocessing (CLAHE, Gaussian, High-pass)
# ──────────────────────────────────────────

class CLAHETransform:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Enhances local contrast to make disease symptoms more visible
    under variable lighting conditions (as used in MangoXNet).
    """
    def __init__(self, clip_limit: float = 2.0, grid_size: int = 8):
        self.clip_limit = clip_limit
        self.grid_size = grid_size

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 2:
            clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit,
                tileGridSize=(self.grid_size, self.grid_size)
            )
            return clahe.apply(img)
        # For color images, apply to L channel in LAB space
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=(self.grid_size, self.grid_size)
        )
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


class GaussianSmooth:
    """Apply Gaussian smoothing to reduce sensor noise."""
    def __init__(self, kernel_size: int = 3, sigma: float = 0.5):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(
            img, (self.kernel_size, self.kernel_size), self.sigma
        )


class HighPassFilter:
    """Apply high-pass filtering to enhance lesion boundaries.

    Blends high-frequency details with the original image.
    """
    def __init__(self, kernel_size: int = 5, alpha: float = 0.3):
        self.kernel_size = kernel_size
        self.alpha = alpha

    def __call__(self, img: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(
            img, (self.kernel_size, self.kernel_size), 0
        )
        high_freq = cv2.subtract(img, blurred)
        enhanced = cv2.addWeighted(img, 1.0, high_freq, self.alpha, 0)
        return np.clip(enhanced, 0, 255).astype(np.uint8)


def get_preprocessing_pipeline(
    use_clahe: bool = True,
    use_gaussian: bool = True,
    use_highpass: bool = True
) -> list:
    """Build preprocessing pipeline as list of numpy transforms."""
    pipeline = []
    if use_clahe:
        pipeline.append(CLAHETransform(clip_limit=2.0, grid_size=8))
    if use_gaussian:
        pipeline.append(GaussianSmooth(kernel_size=3, sigma=0.5))
    if use_highpass:
        pipeline.append(HighPassFilter(kernel_size=5, alpha=0.3))
    return pipeline


# ──────────────────────────────────────────
# Augmentation
# ──────────────────────────────────────────

def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Training augmentations (applied ONLY to training set)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=20),
        transforms.RandomAffine(degrees=0, scale=(0.85, 1.15)),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """Validation/test transforms (deterministic, no augmentation)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


# ──────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────

class CropDiseaseDataset(Dataset):
    """Multi-crop leaf disease dataset.

    Loads images from a directory structure:
        root/
          class_name_1/
            img001.jpg
            ...
          class_name_2/
            ...

    Each image is associated with:
        - A disease class label (int)
        - A crop type label (int) for crop-aware conditioning
    """
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        crop_ids: List[int],
        transform: transforms.Compose = None,
        preprocessing: list = None,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.crop_ids = crop_ids
        self.transform = transform
        self.preprocessing = preprocessing or []

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        crop_id = self.crop_ids[idx]

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply preprocessing (CLAHE, Gaussian, high-pass)
        for preprocess in self.preprocessing:
            img = preprocess(img)

        # Convert to PIL for torchvision transforms
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, label, crop_id


def load_dataset_from_directory(
    root: str,
    crop_type: str,
    crop_label: int,
    class_names: List[str] = None,
    global_label_offset: int = 0,
) -> Tuple[List[str], List[int], List[int], List[str]]:
    """Load image paths and labels from a directory.

    Args:
        root: Root directory containing class subdirectories
        crop_type: Crop type name
        crop_label: Integer crop type label
        class_names: Expected class subdirectory names (auto-detected if None)
        global_label_offset: Offset to add to local class labels for unified labeling

    Returns:
        paths, labels, crop_ids, discovered_class_names
    """
    paths, labels, crop_ids = [], [], []

    if class_names is None:
        class_names = sorted([
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ])

    found_classes = []
    for local_label, cls_name in enumerate(class_names):
        cls_dir = os.path.join(root, cls_name)
        if not os.path.isdir(cls_dir):
            # Try common naming variations
            for variation in [cls_name.replace("_", " "),
                            cls_name.replace(" ", "_"),
                            cls_name.lower(),
                            cls_name.title()]:
                alt_dir = os.path.join(root, variation)
                if os.path.isdir(alt_dir):
                    cls_dir = alt_dir
                    break
            else:
                print(f"  [WARN] Class directory not found: {cls_dir}")
                continue

        found_classes.append(cls_name)
        img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        for fname in os.listdir(cls_dir):
            if Path(fname).suffix.lower() in img_exts:
                paths.append(os.path.join(cls_dir, fname))
                labels.append(local_label + global_label_offset)
                crop_ids.append(crop_label)

    print(f"  [{crop_type}] Loaded {len(paths)} images across "
          f"{len(found_classes)} classes from {root}")
    return paths, labels, crop_ids, found_classes


def create_merged_dataset(
    dataset_configs: list,
    crop_labels: dict,
    preprocessing_config: dict = None,
    image_size: int = 224,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """Create merged multi-crop dataset with train/val/test splits.

    Args:
        dataset_configs: List of DatasetConfig objects
        crop_labels: Dict mapping crop_type → int label
        preprocessing_config: Dict with use_clahe, use_gaussian, use_highpass
        image_size: Target image size
        train/val/test_ratio: Split ratios
        seed: Random seed

    Returns:
        train_loader, val_loader, test_loader, metadata
    """
    all_paths, all_labels, all_crop_ids = [], [], []
    class_map = {}
    global_offset = 0

    print("\n" + "=" * 60)
    print("Loading and Merging Datasets")
    print("=" * 60)

    for cfg in dataset_configs:
        if not os.path.exists(cfg.root):
            print(f"  [SKIP] {cfg.name}: directory not found at {cfg.root}")
            continue

        crop_label = crop_labels.get(cfg.crop_type, 0)
        paths, labels, cids, found_classes = load_dataset_from_directory(
            root=cfg.root,
            crop_type=cfg.crop_type,
            crop_label=crop_label,
            class_names=cfg.class_names,
            global_label_offset=global_offset,
        )
        all_paths.extend(paths)
        all_labels.extend(labels)
        all_crop_ids.extend(cids)

        for i, cls in enumerate(found_classes):
            unified_name = f"{cfg.crop_type}/{cls}"
            class_map[global_offset + i] = unified_name

        global_offset += len(found_classes)

    if len(all_paths) == 0:
        raise RuntimeError("No images found! Check dataset paths in config.py")

    total = len(all_paths)
    num_classes = len(class_map)
    print(f"\nTotal: {total} images, {num_classes} classes")

    # Stratified split
    labels_arr = np.array(all_labels)

    splitter1 = StratifiedShuffleSplit(
        n_splits=1, test_size=(val_ratio + test_ratio), random_state=seed
    )
    train_idx, temp_idx = next(splitter1.split(all_paths, labels_arr))

    val_frac = val_ratio / (val_ratio + test_ratio)
    splitter2 = StratifiedShuffleSplit(
        n_splits=1, test_size=(1 - val_frac), random_state=seed
    )
    temp_labels = labels_arr[temp_idx]
    val_sub_idx, test_sub_idx = next(splitter2.split(temp_idx, temp_labels))
    val_idx = temp_idx[val_sub_idx]
    test_idx = temp_idx[test_sub_idx]

    print(f"Split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # Preprocessing
    pp = preprocessing_config or {}
    preprocessing = get_preprocessing_pipeline(
        use_clahe=pp.get('use_clahe', True),
        use_gaussian=pp.get('use_gaussian', True),
        use_highpass=pp.get('use_highpass', True),
    )

    # Create datasets
    def make_subset(indices, transform):
        return CropDiseaseDataset(
            image_paths=[all_paths[i] for i in indices],
            labels=[all_labels[i] for i in indices],
            crop_ids=[all_crop_ids[i] for i in indices],
            transform=transform,
            preprocessing=preprocessing,
        )

    train_ds = make_subset(train_idx, get_train_transforms(image_size))
    val_ds   = make_subset(val_idx,   get_val_transforms(image_size))
    test_ds  = make_subset(test_idx,  get_val_transforms(image_size))

    metadata = {
        "num_classes": num_classes,
        "class_map": class_map,
        "total_images": total,
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_idx),
    }

    return train_ds, val_ds, test_ds, metadata
