"""
Evaluation Module for CropGuardNet
====================================
Comprehensive evaluation metrics, confusion matrices, classification
reports, and per-class performance analysis for the paper.
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, cohen_kappa_score,
    matthews_corrcoef, roc_auc_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: List[str],
    has_crop_ids: bool = True,
) -> Dict:
    """Run comprehensive evaluation on a dataset.

    Args:
        model: Trained model
        dataloader: Evaluation data loader
        device: Torch device
        class_names: List of class names
        has_crop_ids: Whether dataset returns crop IDs

    Returns:
        Dictionary with all metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
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

            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Overall metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    kappa = cohen_kappa_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)

    # Per-class metrics
    per_class_precision, per_class_recall, per_class_f1, per_class_support = \
        precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )

    # AUC-ROC (one-vs-rest)
    try:
        if all_probs.shape[1] == 2:
            auc_roc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            auc_roc = roc_auc_score(
                all_labels, all_probs, multi_class='ovr', average='macro'
            )
    except Exception:
        auc_roc = None

    # Classification report
    report = classification_report(
        all_labels, all_preds, target_names=class_names, zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Top-5 accuracy (if applicable)
    top5_acc = None
    if all_probs.shape[1] > 5:
        top5_preds = np.argsort(all_probs, axis=1)[:, -5:]
        top5_correct = sum(
            all_labels[i] in top5_preds[i] for i in range(len(all_labels))
        )
        top5_acc = top5_correct / len(all_labels)

    results = {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "precision_weighted": precision_w,
        "recall_weighted": recall_w,
        "f1_weighted": f1_w,
        "cohen_kappa": kappa,
        "matthews_corrcoef": mcc,
        "auc_roc": auc_roc,
        "top5_accuracy": top5_acc,
        "per_class": {
            class_names[i]: {
                "precision": float(per_class_precision[i]),
                "recall": float(per_class_recall[i]),
                "f1": float(per_class_f1[i]),
                "support": int(per_class_support[i]),
            }
            for i in range(len(class_names))
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
    }

    return results


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str = "Confusion Matrix",
    normalize: bool = True,
    figsize: Tuple[int, int] = None,
):
    """Create publication-quality confusion matrix plot.

    Args:
        cm: Confusion matrix array
        class_names: Class labels
        save_path: Path to save figure
        title: Plot title
        normalize: Whether to show percentages
        figsize: Figure size override
    """
    if figsize is None:
        n = len(class_names)
        figsize = (max(8, n * 0.8), max(6, n * 0.7))

    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)  # Handle zero-support classes
    else:
        cm_norm = cm

    fig, ax = plt.subplots(figsize=figsize)

    # Heatmap
    sns.heatmap(
        cm_norm, annot=True,
        fmt='.1%' if normalize else 'd',
        cmap='Blues', xticklabels=class_names, yticklabels=class_names,
        ax=ax, vmin=0, vmax=1 if normalize else None,
        linewidths=0.5, linecolor='white',
        annot_kws={'size': 8 if len(class_names) > 10 else 10}
    )

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrix saved: {save_path}")


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str,
    title: str = "Training History"
):
    """Plot training and validation loss/accuracy curves.

    Args:
        history: Dict with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save figure
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss curves
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Curves', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Accuracy curves
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Accuracy Curves', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Training curves saved: {save_path}")


def plot_per_class_metrics(
    results: Dict,
    save_path: str,
    title: str = "Per-Class Performance"
):
    """Bar chart of per-class precision, recall, F1.

    Args:
        results: Output from evaluate_model()
        save_path: Path to save figure
        title: Plot title
    """
    per_class = results['per_class']
    classes = list(per_class.keys())
    precision = [per_class[c]['precision'] for c in classes]
    recall = [per_class[c]['recall'] for c in classes]
    f1 = [per_class[c]['f1'] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(classes) * 0.8), 6))
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#2196F3', alpha=0.8)
    bars2 = ax.bar(x, recall, width, label='Recall', color='#4CAF50', alpha=0.8)
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#FF9800', alpha=0.8)

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.05:
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Per-class metrics saved: {save_path}")


def plot_ablation_results(
    ablation_data: List[Dict],
    save_path: str,
    title: str = "Ablation Study"
):
    """Create ablation study bar chart.

    Args:
        ablation_data: List of dicts with 'name', 'accuracy', 'f1', 'params'
        save_path: Path to save figure
        title: Plot title
    """
    names = [d['name'] for d in ablation_data]
    accs = [d['accuracy'] * 100 for d in ablation_data]
    f1s = [d['f1'] * 100 for d in ablation_data]
    params = [d['params'] / 1000 for d in ablation_data]  # in K

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Accuracy & F1
    x = np.arange(len(names))
    width = 0.35

    bars1 = ax1.bar(x - width/2, accs, width, label='Accuracy (%)', color='#2196F3', alpha=0.8)
    bars2 = ax1.bar(x + width/2, f1s, width, label='F1-Score (%)', color='#4CAF50', alpha=0.8)

    ax1.set_ylabel('Score (%)', fontsize=12)
    ax1.set_title(f'{title} — Performance', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    # Parameters
    colors = ['#FF5722' if p > 500 else '#4CAF50' for p in params]
    bars3 = ax2.bar(x, params, color=colors, alpha=0.8)
    ax2.axhline(y=500, color='red', linestyle='--', alpha=0.5, label='500K threshold')
    ax2.set_xlabel('Configuration', fontsize=12)
    ax2.set_ylabel('Parameters (K)', fontsize=12)
    ax2.set_title(f'{title} — Model Size', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')

    for bar in bars3:
        height = bar.get_height()
        ax2.annotate(f'{height:.0f}K',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Ablation results saved: {save_path}")


def plot_model_comparison(
    models_data: List[Dict],
    save_path: str,
    title: str = "Model Comparison"
):
    """Radar/spider chart comparing multiple models.

    Args:
        models_data: List of dicts with 'name', 'accuracy', 'f1', 'params', 'size_mb'
        save_path: Path to save figure
        title: Plot title
    """
    metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
    N = len(metrics)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(models_data))
    width = 0.8 / len(metrics)

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']

    for i, metric in enumerate(metrics):
        key = metric.lower().replace('-', '_')
        values = [d.get(key, d.get(f'{key}_macro', 0)) * 100 for d in models_data]
        bars = ax.bar(x + i * width, values, width, label=metric,
                     color=colors[i % len(colors)], alpha=0.8)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels([d['name'] for d in models_data], fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Model comparison saved: {save_path}")


def save_results(results: Dict, save_path: str):
    """Save evaluation results to JSON.

    Args:
        results: Results dict from evaluate_model()
        save_path: JSON output path
    """
    # Remove non-serializable items
    save_dict = {k: v for k, v in results.items()
                 if k not in ['predictions', 'labels', 'probabilities']}

    with open(save_path, 'w') as f:
        json.dump(save_dict, f, indent=2, default=str)

    print(f"  Results saved: {save_path}")


def print_results_table(results: Dict, model_name: str = "Model"):
    """Print formatted results table to console.

    Args:
        results: Results dict from evaluate_model()
        model_name: Model name for header
    """
    print(f"\n{'='*60}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'='*60}")
    print(f"  Accuracy:           {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"  Precision (macro):  {results['precision_macro']:.4f}")
    print(f"  Recall (macro):     {results['recall_macro']:.4f}")
    print(f"  F1-Score (macro):   {results['f1_macro']:.4f}")
    print(f"  F1-Score (weighted):{results['f1_weighted']:.4f}")
    print(f"  Cohen's Kappa:      {results['cohen_kappa']:.4f}")
    print(f"  MCC:                {results['matthews_corrcoef']:.4f}")
    if results['auc_roc'] is not None:
        print(f"  AUC-ROC:            {results['auc_roc']:.4f}")
    if results['top5_accuracy'] is not None:
        print(f"  Top-5 Accuracy:     {results['top5_accuracy']:.4f}")
    print(f"{'='*60}")
    print(f"\n{results['classification_report']}")


def full_evaluation(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: List[str],
    output_dir: str,
    model_name: str = "CropGuardNet",
    has_crop_ids: bool = True,
) -> Dict:
    """Run full evaluation pipeline with all plots.

    Args:
        model: Trained model
        dataloader: Test data loader
        device: Torch device
        class_names: Class name list
        output_dir: Output directory for plots and metrics
        model_name: Model name for titles
        has_crop_ids: Whether the dataloader returns crop IDs

    Returns:
        Complete results dictionary
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Full Evaluation: {model_name}")
    print(f"{'='*60}")

    # 1. Evaluate
    results = evaluate_model(
        model, dataloader, device, class_names, has_crop_ids
    )

    # 2. Print results
    print_results_table(results, model_name)

    # 3. Confusion matrix
    cm = np.array(results['confusion_matrix'])
    plot_confusion_matrix(
        cm, class_names,
        save_path=os.path.join(output_dir, f"{model_name}_confusion_matrix.png"),
        title=f"{model_name} — Confusion Matrix"
    )

    # 4. Per-class metrics
    plot_per_class_metrics(
        results,
        save_path=os.path.join(output_dir, f"{model_name}_per_class.png"),
        title=f"{model_name} — Per-Class Performance"
    )

    # 5. Save JSON
    save_results(
        results,
        save_path=os.path.join(output_dir, f"{model_name}_results.json")
    )

    return results
