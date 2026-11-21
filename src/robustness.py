"""
Robustness Testing Module for CropGuardNet
=============================================
Tests model resilience under realistic deployment conditions:
  1. Gaussian noise (sensor noise)
  2. Gaussian blur (focus issues)
  3. Brightness shifts (lighting variation)
  4. Occlusion (partial leaf visibility)
  5. JPEG compression artifacts
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ══════════════════════════════════════════
# Perturbation Functions
# ══════════════════════════════════════════

def add_gaussian_noise(images: torch.Tensor, std: float = 0.1) -> torch.Tensor:
    """Add Gaussian noise to image batch.

    Args:
        images: (B, C, H, W) tensor in [0, 1]
        std: Noise standard deviation

    Returns:
        Noisy images (clamped to [0, 1])
    """
    noise = torch.randn_like(images) * std
    return torch.clamp(images + noise, 0, 1)


def add_gaussian_blur(images: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """Apply Gaussian blur to image batch.

    Args:
        images: (B, C, H, W) tensor
        kernel_size: Blur kernel size (odd number)

    Returns:
        Blurred images
    """
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Create Gaussian kernel
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    kernel_1d = gauss / gauss.sum()
    kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)

    # Apply separably for each channel
    kernel = kernel_2d.unsqueeze(0).unsqueeze(0)
    kernel = kernel.expand(images.shape[1], 1, -1, -1).to(images.device)
    pad = kernel_size // 2

    return F.conv2d(images, kernel, padding=pad, groups=images.shape[1])


def adjust_brightness(images: torch.Tensor, factor: float = 1.5) -> torch.Tensor:
    """Adjust image brightness.

    Args:
        images: (B, C, H, W) tensor in [0, 1]
        factor: Brightness factor (1.0 = unchanged, >1 brighter, <1 darker)

    Returns:
        Brightness-adjusted images
    """
    return torch.clamp(images * factor, 0, 1)


def add_occlusion(images: torch.Tensor, patch_ratio: float = 0.2) -> torch.Tensor:
    """Add random rectangular occlusion (cutout).

    Args:
        images: (B, C, H, W) tensor
        patch_ratio: Size of occluded patch relative to image

    Returns:
        Occluded images
    """
    B, C, H, W = images.shape
    result = images.clone()
    patch_h = int(H * patch_ratio)
    patch_w = int(W * patch_ratio)

    for i in range(B):
        top = np.random.randint(0, H - patch_h)
        left = np.random.randint(0, W - patch_w)
        result[i, :, top:top+patch_h, left:left+patch_w] = 0  # Black occlusion

    return result


def add_jpeg_compression(images: torch.Tensor, quality: int = 30) -> torch.Tensor:
    """Simulate JPEG compression artifacts.

    Args:
        images: (B, C, H, W) tensor in [0, 1]
        quality: JPEG quality (1-100, lower = more artifacts)

    Returns:
        Compressed images
    """
    B, C, H, W = images.shape
    result = torch.zeros_like(images)

    for i in range(B):
        # Convert to numpy uint8 for cv2
        img_np = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        # Encode and decode JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', img_np, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        result[i] = torch.from_numpy(decoded).permute(2, 0, 1).float() / 255.0

    return result.to(images.device)


# ══════════════════════════════════════════
# Robustness Evaluation
# ══════════════════════════════════════════

PERTURBATION_CONFIGS = {
    "gaussian_noise": {
        "fn": add_gaussian_noise,
        "levels": [
            {"std": 0.05, "label": "σ=0.05"},
            {"std": 0.10, "label": "σ=0.10"},
            {"std": 0.15, "label": "σ=0.15"},
            {"std": 0.20, "label": "σ=0.20"},
        ]
    },
    "gaussian_blur": {
        "fn": add_gaussian_blur,
        "levels": [
            {"kernel_size": 3, "label": "k=3"},
            {"kernel_size": 5, "label": "k=5"},
            {"kernel_size": 7, "label": "k=7"},
            {"kernel_size": 9, "label": "k=9"},
        ]
    },
    "brightness_increase": {
        "fn": adjust_brightness,
        "levels": [
            {"factor": 1.2, "label": "×1.2"},
            {"factor": 1.5, "label": "×1.5"},
            {"factor": 1.8, "label": "×1.8"},
            {"factor": 2.0, "label": "×2.0"},
        ]
    },
    "brightness_decrease": {
        "fn": adjust_brightness,
        "levels": [
            {"factor": 0.8, "label": "×0.8"},
            {"factor": 0.6, "label": "×0.6"},
            {"factor": 0.4, "label": "×0.4"},
            {"factor": 0.2, "label": "×0.2"},
        ]
    },
    "occlusion": {
        "fn": add_occlusion,
        "levels": [
            {"patch_ratio": 0.10, "label": "10%"},
            {"patch_ratio": 0.20, "label": "20%"},
            {"patch_ratio": 0.30, "label": "30%"},
            {"patch_ratio": 0.40, "label": "40%"},
        ]
    },
    "jpeg_compression": {
        "fn": add_jpeg_compression,
        "levels": [
            {"quality": 75, "label": "Q=75"},
            {"quality": 50, "label": "Q=50"},
            {"quality": 25, "label": "Q=25"},
            {"quality": 10, "label": "Q=10"},
        ]
    },
}


def evaluate_robustness(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    perturbations: List[str] = None,
    has_crop_ids: bool = True,
) -> Dict:
    """Evaluate model under various perturbations.

    Args:
        model: Trained model
        dataloader: Test data loader
        device: Torch device
        perturbations: List of perturbation names (None = all)
        has_crop_ids: Whether the dataloader returns crop IDs

    Returns:
        Dict mapping perturbation name → {level_label → accuracy}
    """
    model.eval()

    if perturbations is None:
        perturbations = list(PERTURBATION_CONFIGS.keys())

    # First, get clean accuracy
    clean_correct, clean_total = 0, 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Clean eval"):
            if has_crop_ids:
                images, labels, crop_ids = batch
                crop_ids = crop_ids.to(device)
            else:
                images, labels = batch
                crop_ids = None

            images = images.to(device)
            labels = labels.to(device)

            if crop_ids is not None:
                outputs = model(images, crop_ids)
            else:
                outputs = model(images)

            preds = outputs.argmax(dim=1)
            clean_correct += (preds == labels).sum().item()
            clean_total += labels.size(0)

    clean_acc = clean_correct / max(clean_total, 1)
    results = {"clean": {"accuracy": clean_acc}}

    # Evaluate each perturbation
    for pert_name in perturbations:
        if pert_name not in PERTURBATION_CONFIGS:
            print(f"  Unknown perturbation: {pert_name}, skipping")
            continue

        config = PERTURBATION_CONFIGS[pert_name]
        fn = config["fn"]
        results[pert_name] = {}

        for level in config["levels"]:
            label = level["label"]
            params = {k: v for k, v in level.items() if k != "label"}

            correct, total = 0, 0
            with torch.no_grad():
                for batch in dataloader:
                    if has_crop_ids:
                        images, labels, crop_ids = batch
                        crop_ids = crop_ids.to(device)
                    else:
                        images, labels = batch
                        crop_ids = None

                    images = images.to(device)
                    labels = labels.to(device)

                    # Apply perturbation
                    perturbed = fn(images, **params)

                    if crop_ids is not None:
                        outputs = model(perturbed, crop_ids)
                    else:
                        outputs = model(perturbed)

                    preds = outputs.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            acc = correct / max(total, 1)
            results[pert_name][label] = acc
            print(f"  {pert_name} [{label}]: {acc:.4f} "
                  f"(Δ = {(acc - clean_acc)*100:+.2f}%)")

    return results


def plot_robustness_results(
    results: Dict,
    save_path: str,
    title: str = "Robustness Analysis"
):
    """Create comprehensive robustness plots.

    Args:
        results: Output from evaluate_robustness()
        save_path: Path to save figure
        title: Plot title
    """
    perturbations = [k for k in results if k != "clean"]
    n_perts = len(perturbations)

    if n_perts == 0:
        return

    cols = min(3, n_perts)
    rows = (n_perts + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if n_perts == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    clean_acc = results["clean"]["accuracy"] * 100

    for idx, pert_name in enumerate(perturbations):
        ax = axes[idx]
        pert_data = results[pert_name]
        labels = list(pert_data.keys())
        accs = [v * 100 for v in pert_data.values()]

        # Add clean accuracy as reference
        all_labels = ["Clean"] + labels
        all_accs = [clean_acc] + accs

        colors = ['#4CAF50'] + ['#2196F3' if a >= clean_acc * 0.9
                                 else '#FF9800' if a >= clean_acc * 0.7
                                 else '#F44336' for a in accs]

        bars = ax.bar(all_labels, all_accs, color=colors, alpha=0.8)
        ax.set_title(pert_name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=10)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

        # Reference line at clean accuracy
        ax.axhline(y=clean_acc, color='green', linestyle='--', alpha=0.5)

    # Hide unused subplots
    for idx in range(n_perts, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f"{title}\nClean Accuracy: {clean_acc:.2f}%",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Robustness plot saved: {save_path}")


def full_robustness_analysis(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: str,
    model_name: str = "CropGuardNet",
    has_crop_ids: bool = True,
    perturbations: List[str] = None,
) -> Dict:
    """Run full robustness analysis pipeline.

    Args:
        model: Trained model
        dataloader: Test data loader
        device: Torch device
        output_dir: Output directory
        model_name: Model name for titles
        has_crop_ids: Whether the dataloader returns crop IDs
        perturbations: List of perturbation names (None = all)

    Returns:
        Robustness results dictionary
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Robustness Analysis: {model_name}")
    print(f"{'='*60}")

    # Run evaluation
    results = evaluate_robustness(
        model, dataloader, device, perturbations, has_crop_ids
    )

    # Plot
    plot_robustness_results(
        results,
        save_path=os.path.join(output_dir, f"{model_name}_robustness.png"),
        title=f"{model_name} — Robustness Analysis"
    )

    # Save JSON
    import json
    with open(os.path.join(output_dir, f"{model_name}_robustness.json"), 'w') as f:
        json.dump(results, f, indent=2)

    return results


def visualize_perturbations(
    image: torch.Tensor,
    save_path: str,
    title: str = "Perturbation Examples"
):
    """Create a visual grid showing all perturbation types on one image.

    Args:
        image: Single image tensor (C, H, W) or (1, C, H, W)
        save_path: Path to save figure
        title: Plot title
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    def tensor_to_img(t):
        return t[0].permute(1, 2, 0).cpu().numpy().clip(0, 1)

    # Row 1: Original + Noise + Blur + Brightness
    axes[0, 0].imshow(tensor_to_img(image))
    axes[0, 0].set_title("Original", fontsize=12)

    axes[0, 1].imshow(tensor_to_img(add_gaussian_noise(image, std=0.15)))
    axes[0, 1].set_title("Noise (σ=0.15)", fontsize=12)

    axes[0, 2].imshow(tensor_to_img(add_gaussian_blur(image, kernel_size=7)))
    axes[0, 2].set_title("Blur (k=7)", fontsize=12)

    axes[0, 3].imshow(tensor_to_img(adjust_brightness(image, factor=1.8)))
    axes[0, 3].set_title("Bright (×1.8)", fontsize=12)

    # Row 2: Dim + Occlusion + JPEG + Combined
    axes[1, 0].imshow(tensor_to_img(adjust_brightness(image, factor=0.4)))
    axes[1, 0].set_title("Dim (×0.4)", fontsize=12)

    axes[1, 1].imshow(tensor_to_img(add_occlusion(image, patch_ratio=0.25)))
    axes[1, 1].set_title("Occlusion (25%)", fontsize=12)

    axes[1, 2].imshow(tensor_to_img(add_jpeg_compression(image, quality=15)))
    axes[1, 2].set_title("JPEG (Q=15)", fontsize=12)

    # Combined: noise + blur + dim
    combined = add_gaussian_noise(
        add_gaussian_blur(
            adjust_brightness(image, factor=0.6),
            kernel_size=5
        ),
        std=0.1
    )
    axes[1, 3].imshow(tensor_to_img(combined))
    axes[1, 3].set_title("Combined", fontsize=12)

    for ax in axes.flatten():
        ax.axis('off')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Perturbation examples saved: {save_path}")
