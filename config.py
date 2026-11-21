"""
CropGuardNet Configuration
==========================
Central configuration for all experiments.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

# ──────────────────────────────────────────
# Paths
# ──────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_RAW     = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROC    = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_DIR    = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
XAI_DIR      = os.path.join(PROJECT_ROOT, "xai_outputs")


@dataclass
class TrainingConfig:
    """Hyperparameters for training."""
    # Data
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Optimizer
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = "cosine"       # "cosine" | "step" | "plateau"
    warmup_epochs: int = 5

    # Training
    epochs: int = 100
    patience: int = 15              # Early stopping patience
    seed: int = 42
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])

    # Model
    model_name: str = "cropguardnet"
    backbone: str = "mobilenetv3_small_050"  # Lightest: 477K params, 1.82 MB
    feature_dim: int = 32                   # Internal feature dimension
    pretrained: bool = True
    num_classes: int = 15            # Updated per dataset
    dropout: float = 0.3

    # CropGuardNet-specific
    use_multi_scale: bool = True
    use_se_attention: bool = True
    use_transformer: bool = True
    use_crop_conditioning: bool = True
    transformer_heads: int = 2
    transformer_layers: int = 1
    se_reduction: int = 4

    # Augmentation
    use_clahe: bool = True
    use_gaussian_smooth: bool = True
    use_highpass_filter: bool = True

    # Device
    device: str = "auto"  # "auto" | "cuda" | "cpu"

    # Logging
    log_interval: int = 10
    save_best: bool = True
    experiment_name: str = "cropguardnet_v1"


@dataclass
class DatasetConfig:
    """Per-dataset configuration."""
    name: str
    root: str
    num_classes: int
    class_names: List[str]
    crop_type: str  # For crop-aware conditioning


# ──────────────────────────────────────────
# Dataset Definitions
# ──────────────────────────────────────────
DATASETS = {
    "plantvillage_tomato": DatasetConfig(
        name="PlantVillage-Tomato",
        root=os.path.join(DATA_RAW, "plantvillage", "Tomato"),
        num_classes=10,
        class_names=[
            "Bacterial_spot", "Early_blight", "Late_blight",
            "Leaf_Mold", "Septoria_leaf_spot",
            "Spider_mites", "Target_Spot",
            "Yellow_Leaf_Curl_Virus", "Mosaic_virus", "healthy"
        ],
        crop_type="tomato"
    ),
    "plantvillage_grape": DatasetConfig(
        name="PlantVillage-Grape",
        root=os.path.join(DATA_RAW, "plantvillage", "Grape"),
        num_classes=4,
        class_names=[
            "Black_rot", "Esca", "Leaf_blight", "healthy"
        ],
        crop_type="grape"
    ),
    "plantvillage_apple": DatasetConfig(
        name="PlantVillage-Apple",
        root=os.path.join(DATA_RAW, "plantvillage", "Apple"),
        num_classes=4,
        class_names=[
            "Apple_scab", "Black_rot", "Cedar_apple_rust", "healthy"
        ],
        crop_type="apple"
    ),
    "plantvillage_corn": DatasetConfig(
        name="PlantVillage-Corn",
        root=os.path.join(DATA_RAW, "plantvillage", "Corn"),
        num_classes=4,
        class_names=[
            "Cercospora_leaf_spot", "Common_rust",
            "Northern_Leaf_Blight", "healthy"
        ],
        crop_type="corn"
    ),
    "plantvillage_potato": DatasetConfig(
        name="PlantVillage-Potato",
        root=os.path.join(DATA_RAW, "plantvillage", "Potato"),
        num_classes=3,
        class_names=["Early_blight", "Late_blight", "healthy"],
        crop_type="potato"
    ),
    "mangoleafbd": DatasetConfig(
        name="MangoLeafBD",
        root=os.path.join(DATA_RAW, "mangoleafbd"),
        num_classes=8,
        class_names=[
            "Anthracnose", "Bacterial Canker", "Cutting Weevil",
            "Die Back", "Gall Midge", "Healthy",
            "Powdery Mildew", "Sooty Mould"
        ],
        crop_type="mango"
    ),
    "rice_leaf": DatasetConfig(
        name="RiceLeafDisease",
        root=os.path.join(DATA_RAW, "rice_leaf", "rice_leaf_diseases"),
        num_classes=3,
        class_names=["Bacterial leaf blight", "Brown spot", "Leaf smut"],
        crop_type="rice"
    ),
    "corn_maize": DatasetConfig(
        name="CornMaizeDisease",
        root=os.path.join(DATA_RAW, "corn_maize", "data"),
        num_classes=4,
        class_names=["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"],
        crop_type="corn"
    ),
    "cassava": DatasetConfig(
        name="CassavaLeafDisease",
        root=os.path.join(DATA_RAW, "cassava"),
        num_classes=5,
        class_names=[
            "Cassava_Bacterial_Blight", "Cassava_Brown_Streak",
            "Cassava_Green_Mottle", "Cassava_Mosaic", "Healthy"
        ],
        crop_type="cassava"
    ),
}

# Crop type → integer label for crop-aware conditioning
CROP_LABELS = {
    "tomato": 0, "grape": 1, "apple": 2, "corn": 3,
    "potato": 4, "mango": 5, "rice": 6, "cassava": 7,
}
NUM_CROP_TYPES = len(CROP_LABELS)
