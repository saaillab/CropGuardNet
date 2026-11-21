"""
Explainable AI (XAI) Module for CropGuardNet
==============================================
Implements 4 XAI methods for model interpretability:
  1. Grad-CAM++ — Class-discriminative localization with pixel-level gradients
  2. LayerCAM — Layer-wise class activation mapping
  3. LIME — Local Interpretable Model-agnostic Explanations
  4. SHAP — SHapley Additive exPlanations (GradientExplainer)
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from typing import Optional, Tuple, List
import cv2


# ══════════════════════════════════════════
# 1. Grad-CAM++
# ══════════════════════════════════════════

class GradCAMPlusPlus:
    """Grad-CAM++ for generating class-discriminative localization maps.

    Extends Grad-CAM by incorporating pixel-level gradient information
    with higher-order derivatives for more accurate class-specific
    activation maps (Chattopadhay et al., 2018).

    Usage:
        gcam = GradCAMPlusPlus(model, target_layer)
        heatmap = gcam(input_image, class_idx)
        gcam.visualize(input_image, heatmap, save_path)
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, x: torch.Tensor, class_idx: int = None,
                 crop_ids: torch.Tensor = None) -> np.ndarray:
        """Generate Grad-CAM++ heatmap.

        Args:
            x: Input image tensor (1, C, H, W)
            class_idx: Target class index (None = predicted class)
            crop_ids: Crop type IDs for crop-aware models

        Returns:
            Heatmap as numpy array (H, W) normalized to [0, 1]
        """
        self.model.eval()
        x.requires_grad_(True)

        # Forward pass
        if crop_ids is not None:
            output = self.model(x, crop_ids)
        else:
            output = self.model(x)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward(retain_graph=True)

        # Grad-CAM++ computation
        gradients = self.gradients[0]     # (C, H, W)
        activations = self.activations[0]  # (C, H, W)

        # Second and third order gradients for Grad-CAM++
        grad_2 = gradients ** 2
        grad_3 = gradients ** 3

        # Alpha coefficients
        denom = 2 * grad_2 + activations * grad_3
        denom = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha = grad_2 / denom

        # Weights with ReLU on gradients
        weights = (alpha * F.relu(gradients)).sum(dim=(1, 2))

        # Weighted combination
        cam = torch.zeros(activations.shape[1:], device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam = cam.cpu().numpy()

        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    @staticmethod
    def visualize(image: np.ndarray, heatmap: np.ndarray,
                  title: str = "Grad-CAM++",
                  save_path: str = None, alpha: float = 0.4) -> np.ndarray:
        """Overlay heatmap on original image.

        Args:
            image: Original image (H, W, 3) in [0, 255]
            heatmap: CAM heatmap (h, w) in [0, 1]
            title: Plot title
            save_path: Path to save figure
            alpha: Overlay transparency

        Returns:
            Overlay image (H, W, 3)
        """
        H, W = image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (W, H))
        heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

        overlay = cv2.addWeighted(
            image.astype(np.uint8), 1 - alpha,
            heatmap_colored, alpha, 0
        )

        if save_path:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(image.astype(np.uint8))
            axes[0].set_title("Original")
            axes[0].axis('off')

            axes[1].imshow(heatmap_resized, cmap='jet')
            axes[1].set_title(title + " Heatmap")
            axes[1].axis('off')

            axes[2].imshow(overlay)
            axes[2].set_title(title + " Overlay")
            axes[2].axis('off')

            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

        return overlay


# ══════════════════════════════════════════
# 2. LayerCAM
# ══════════════════════════════════════════

class LayerCAM:
    """LayerCAM for layer-wise class activation mapping.

    Generates more fine-grained activation maps by considering
    the positive contributions of each spatial location
    (Jiang et al., 2021).
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, x: torch.Tensor, class_idx: int = None,
                 crop_ids: torch.Tensor = None) -> np.ndarray:
        self.model.eval()
        x.requires_grad_(True)

        if crop_ids is not None:
            output = self.model(x, crop_ids)
        else:
            output = self.model(x)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward(retain_graph=True)

        gradients = self.gradients[0]      # (C, H, W)
        activations = self.activations[0]  # (C, H, W)

        # LayerCAM: element-wise positive gradient * activation
        cam = F.relu(gradients * activations).sum(dim=0)
        cam = cam.cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    visualize = staticmethod(GradCAMPlusPlus.visualize)  # Reuse visualize


# ══════════════════════════════════════════
# 3. LIME Wrapper
# ══════════════════════════════════════════

class LIMEExplainer:
    """LIME (Local Interpretable Model-agnostic Explanations).

    Explains individual predictions by learning a local linear model
    around the input, highlighting which superpixel regions most
    influenced the prediction (Ribeiro et al., 2016).
    """
    def __init__(self, model: torch.nn.Module, device: torch.device,
                 preprocess_fn=None):
        self.model = model
        self.device = device
        self.preprocess_fn = preprocess_fn

    def _predict_fn(self, images: np.ndarray) -> np.ndarray:
        """Batch prediction function for LIME."""
        self.model.eval()
        batch = []
        for img in images:
            if self.preprocess_fn:
                tensor = self.preprocess_fn(Image.fromarray(img.astype(np.uint8)))
            else:
                tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            batch.append(tensor)

        batch = torch.stack(batch).to(self.device)
        with torch.no_grad():
            outputs = self.model(batch)
            probs = F.softmax(outputs, dim=1)
        return probs.cpu().numpy()

    def explain(self, image: np.ndarray, num_samples: int = 500,
                num_features: int = 10, top_labels: int = 3,
                save_path: str = None) -> dict:
        """Generate LIME explanation.

        Args:
            image: Input image (H, W, 3) in [0, 255]
            num_samples: Number of perturbed samples
            num_features: Number of superpixels to highlight
            top_labels: Number of top classes to explain
            save_path: Path to save visualization

        Returns:
            Dict with explanation data
        """
        try:
            from lime import lime_image
        except ImportError:
            print("[WARN] lime not installed. Run: pip install lime")
            return {}

        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            image.astype(np.double),
            self._predict_fn,
            top_labels=top_labels,
            hide_color=0,
            num_samples=num_samples
        )

        if save_path:
            fig, axes = plt.subplots(1, top_labels + 1, figsize=(5 * (top_labels + 1), 5))
            axes[0].imshow(image.astype(np.uint8))
            axes[0].set_title("Original")
            axes[0].axis('off')

            for i, label in enumerate(explanation.top_labels[:top_labels]):
                temp, mask = explanation.get_image_and_mask(
                    label, positive_only=True,
                    num_features=num_features, hide_rest=False
                )
                axes[i+1].imshow(temp.astype(np.uint8))
                axes[i+1].set_title(f"LIME (class {label})")
                axes[i+1].axis('off')

            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

        return {"explanation": explanation, "top_labels": explanation.top_labels}


# ══════════════════════════════════════════
# 4. SHAP Wrapper
# ══════════════════════════════════════════

class SHAPExplainer:
    """SHAP (SHapley Additive exPlanations) using GradientExplainer.

    Computes feature importance using Shapley values approximated
    via integrated gradients (Lundberg & Lee, 2017).
    """
    def __init__(self, model: torch.nn.Module, background_data: torch.Tensor):
        """
        Args:
            model: PyTorch model
            background_data: Background dataset for SHAP (subset of training data)
        """
        self.model = model
        self.background = background_data

    def explain(self, x: torch.Tensor, save_path: str = None,
                class_names: list = None) -> np.ndarray:
        """Generate SHAP explanation.

        Args:
            x: Input tensor (1, C, H, W)
            save_path: Path to save visualization
            class_names: List of class names

        Returns:
            SHAP values array
        """
        try:
            import shap
        except ImportError:
            print("[WARN] shap not installed. Run: pip install shap")
            return None

        self.model.eval()
        explainer = shap.GradientExplainer(self.model, self.background)
        shap_values = explainer.shap_values(x)

        if save_path and shap_values is not None:
            # Convert for visualization
            if isinstance(shap_values, list):
                # Multi-class: take predicted class
                with torch.no_grad():
                    pred = self.model(x).argmax(dim=1).item()
                sv = shap_values[pred][0]  # (C, H, W)
            else:
                sv = shap_values[0]

            # Aggregate across channels
            importance = np.abs(sv).sum(axis=0)  # (H, W)
            if importance.max() > 0:
                importance = importance / importance.max()

            img_np = x[0].permute(1, 2, 0).cpu().numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(img_np)
            axes[0].set_title("Original")
            axes[0].axis('off')

            axes[1].imshow(importance, cmap='hot')
            axes[1].set_title("SHAP Importance")
            axes[1].axis('off')

            overlay = cv2.addWeighted(
                (img_np * 255).astype(np.uint8), 0.6,
                (cm.hot(importance)[:, :, :3] * 255).astype(np.uint8), 0.4, 0
            )
            axes[2].imshow(overlay)
            axes[2].set_title("SHAP Overlay")
            axes[2].axis('off')

            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

        return shap_values


# ══════════════════════════════════════════
# Multi-XAI Analysis Runner
# ══════════════════════════════════════════

def run_xai_analysis(
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    images: List[torch.Tensor],
    labels: List[int],
    original_images: List[np.ndarray],
    class_names: List[str],
    output_dir: str,
    device: torch.device,
    crop_ids: List[int] = None,
    preprocess_fn=None,
    background_data: torch.Tensor = None,
    num_samples: int = 5,
):
    """Run comprehensive XAI analysis on sample images.

    Generates Grad-CAM++, LayerCAM, LIME, and SHAP visualizations
    for a set of sample images.

    Args:
        model: Trained model
        target_layer: Target layer for gradient-based methods
        images: List of preprocessed input tensors
        labels: Ground truth labels
        original_images: Original images (for overlay)
        class_names: Class name list
        output_dir: Directory to save outputs
        device: Torch device
        crop_ids: Crop type IDs
        preprocess_fn: Preprocessing function for LIME
        background_data: Background data for SHAP
        num_samples: Number of sample images to analyze
    """
    os.makedirs(output_dir, exist_ok=True)

    # Initialize explainers
    gradcam = GradCAMPlusPlus(model, target_layer)
    layercam = LayerCAM(model, target_layer)
    lime_exp = LIMEExplainer(model, device, preprocess_fn)

    shap_exp = None
    if background_data is not None:
        shap_exp = SHAPExplainer(model, background_data)

    model.eval()
    num_samples = min(num_samples, len(images))

    for i in range(num_samples):
        img_tensor = images[i].unsqueeze(0).to(device)
        true_label = labels[i]
        orig_img = original_images[i]
        class_name = class_names[true_label] if true_label < len(class_names) else f"class_{true_label}"

        crop_id = None
        if crop_ids is not None:
            crop_id = torch.tensor([crop_ids[i]]).to(device)

        # Get prediction
        with torch.no_grad():
            if crop_id is not None:
                pred = model(img_tensor, crop_id)
            else:
                pred = model(img_tensor)
            pred_class = pred.argmax(dim=1).item()
            confidence = F.softmax(pred, dim=1)[0, pred_class].item()

        pred_name = class_names[pred_class] if pred_class < len(class_names) else f"class_{pred_class}"

        print(f"\n[Sample {i+1}/{num_samples}] "
              f"True: {class_name} | Pred: {pred_name} ({confidence:.2%})")

        prefix = f"sample_{i+1}_{class_name}"

        # 1. Grad-CAM++
        try:
            heatmap = gradcam(img_tensor, pred_class, crop_id)
            GradCAMPlusPlus.visualize(
                orig_img, heatmap,
                title=f"Grad-CAM++ ({pred_name})",
                save_path=os.path.join(output_dir, f"{prefix}_gradcam++.png")
            )
            print(f"  ✓ Grad-CAM++ saved")
        except Exception as e:
            print(f"  ✗ Grad-CAM++ failed: {e}")

        # 2. LayerCAM
        try:
            heatmap = layercam(img_tensor, pred_class, crop_id)
            GradCAMPlusPlus.visualize(
                orig_img, heatmap,
                title=f"LayerCAM ({pred_name})",
                save_path=os.path.join(output_dir, f"{prefix}_layercam.png")
            )
            print(f"  ✓ LayerCAM saved")
        except Exception as e:
            print(f"  ✗ LayerCAM failed: {e}")

        # 3. LIME
        try:
            lime_exp.explain(
                orig_img, num_samples=300, num_features=8, top_labels=3,
                save_path=os.path.join(output_dir, f"{prefix}_lime.png")
            )
            print(f"  ✓ LIME saved")
        except Exception as e:
            print(f"  ✗ LIME failed: {e}")

        # 4. SHAP
        if shap_exp:
            try:
                shap_exp.explain(
                    img_tensor,
                    save_path=os.path.join(output_dir, f"{prefix}_shap.png"),
                    class_names=class_names
                )
                print(f"  ✓ SHAP saved")
            except Exception as e:
                print(f"  ✗ SHAP failed: {e}")

    # Combined comparison figure
    _create_comparison_grid(output_dir, num_samples)
    print(f"\nXAI outputs saved to: {output_dir}")


def _create_comparison_grid(output_dir: str, num_samples: int):
    """Create a comparison grid of all XAI methods for the paper."""
    methods = ['gradcam++', 'layercam', 'lime', 'shap']

    fig_paths = []
    for i in range(1, num_samples + 1):
        row_images = []
        for method in methods:
            pattern = f"sample_{i}_*_{method}.png"
            import glob
            matches = glob.glob(os.path.join(output_dir, pattern))
            if matches:
                row_images.append(matches[0])
        fig_paths.append(row_images)

    if not any(fig_paths):
        return

    # Create grid
    num_methods = len(methods)
    fig, axes = plt.subplots(
        num_samples, num_methods,
        figsize=(5 * num_methods, 5 * num_samples)
    )
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    for i, row in enumerate(fig_paths):
        for j, path in enumerate(row):
            if os.path.exists(path):
                img = plt.imread(path)
                axes[i, j].imshow(img)
            axes[i, j].axis('off')
            if i == 0:
                axes[0, j].set_title(methods[j].upper(), fontsize=14)

    plt.suptitle("XAI Comparison: CropGuardNet", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "xai_comparison_grid.png"),
        dpi=150, bbox_inches='tight'
    )
    plt.close()


def get_target_layer(model, model_name: str = "cropguardnet"):
    """Get the appropriate target layer for XAI visualization."""
    if model_name == "cropguardnet":
        return model.dsr  # After depthwise separable refinement
    elif model_name == "leafxnet":
        return model.features[-2]  # Last conv before pooling
    elif model_name == "mangoxnet":
        return model.dsr
    else:
        # Try to find last conv layer
        target = None
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                target = module
        return target
