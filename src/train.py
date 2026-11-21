"""
Training Engine for CropGuardNet
=================================
Handles training loops, evaluation, logging, and checkpointing.
"""

import os
import time
import json
import copy
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, matthews_corrcoef
)
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TrainingConfig, MODEL_DIR, RESULTS_DIR


class Trainer:
    """Training engine for CropGuardNet and baselines.

    Handles:
        - Training loop with mixed precision
        - Validation and early stopping
        - Learning rate scheduling (cosine / plateau)
        - Checkpointing (best model + periodic)
        - Comprehensive metrics logging
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: TrainingConfig,
        class_map: dict = None,
    ):
        self.config = config
        self.class_map = class_map or {}

        # Device
        if config.device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(config.device)
        print(f"[Trainer] Using device: {self.device}")

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Scheduler
        if config.scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=config.epochs, eta_min=1e-6
            )
        elif config.scheduler == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5,
                patience=5, min_lr=1e-6
            )
        else:
            self.scheduler = None

        # Tracking
        self.best_val_acc = 0.0
        self.best_model_state = None
        self.patience_counter = 0
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'lr': []
        }

        # Directories
        self.save_dir = os.path.join(MODEL_DIR, config.experiment_name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.results_dir = os.path.join(RESULTS_DIR, config.experiment_name)
        os.makedirs(self.results_dir, exist_ok=True)

    def train_one_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]",
                     leave=False)
        for batch_idx, (images, labels, crop_ids) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            crop_ids = crop_ids.to(self.device)

            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images, crop_ids)
            loss = self.criterion(outputs, labels)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Track
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate(self, loader: DataLoader = None,
                 desc: str = "Val") -> Tuple[float, float, dict]:
        """Evaluate on validation or test set."""
        if loader is None:
            loader = self.val_loader

        self.model.eval()
        running_loss = 0.0
        all_preds, all_labels = [], []

        for images, labels, crop_ids in tqdm(loader, desc=desc, leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)
            crop_ids = crop_ids.to(self.device)

            outputs = self.model(images, crop_ids)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(loader.dataset)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision_macro': precision_score(
                all_labels, all_preds, average='macro', zero_division=0),
            'recall_macro': recall_score(
                all_labels, all_preds, average='macro', zero_division=0),
            'f1_macro': f1_score(
                all_labels, all_preds, average='macro', zero_division=0),
            'precision_weighted': precision_score(
                all_labels, all_preds, average='weighted', zero_division=0),
            'recall_weighted': recall_score(
                all_labels, all_preds, average='weighted', zero_division=0),
            'f1_weighted': f1_score(
                all_labels, all_preds, average='weighted', zero_division=0),
            'mcc': matthews_corrcoef(all_labels, all_preds),
            'loss': epoch_loss,
        }

        return epoch_loss, metrics['accuracy'], metrics

    def train(self) -> dict:
        """Full training loop with early stopping."""
        print(f"\n{'='*60}")
        print(f"Training: {self.config.experiment_name}")
        print(f"Model: {self.config.model_name}")
        print(f"Epochs: {self.config.epochs}, Batch: {self.config.batch_size}")
        print(f"LR: {self.config.learning_rate}, Device: {self.device}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(self.config.epochs):
            # Train
            train_loss, train_acc = self.train_one_epoch(epoch)

            # Validate
            val_loss, val_acc, val_metrics = self.validate()

            # Learning rate scheduling
            current_lr = self.optimizer.param_groups[0]['lr']
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            elif self.scheduler is not None:
                self.scheduler.step()

            # History
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)

            # Print
            print(f"Epoch {epoch+1:3d}/{self.config.epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                  f"LR: {current_lr:.2e}")

            # Best model checkpoint
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.patience_counter = 0
                if self.config.save_best:
                    self._save_checkpoint(epoch, is_best=True)
                print(f"  ★ New best validation accuracy: {val_acc:.4f}")
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= self.config.patience:
                print(f"\nEarly stopping at epoch {epoch+1} "
                      f"(patience={self.config.patience})")
                break

        elapsed = time.time() - start_time
        print(f"\nTraining completed in {elapsed/60:.1f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")

        # Restore best weights and evaluate on test set
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        print(f"\n{'='*60}")
        print("Final Test Evaluation")
        print(f"{'='*60}")
        test_loss, test_acc, test_metrics = self.validate(
            self.test_loader, desc="Test"
        )
        self._print_metrics(test_metrics, "Test")

        # Save results
        results = {
            'config': {
                'model_name': self.config.model_name,
                'backbone': self.config.backbone,
                'epochs_trained': len(self.history['train_loss']),
                'seed': self.config.seed,
                'image_size': self.config.image_size,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
            },
            'best_val_acc': self.best_val_acc,
            'test_metrics': test_metrics,
            'history': self.history,
            'training_time_minutes': elapsed / 60,
            'timestamp': datetime.now().isoformat(),
        }
        self._save_results(results)

        return results

    def _print_metrics(self, metrics: dict, split: str = "Test"):
        """Pretty-print evaluation metrics."""
        print(f"\n{split} Results:")
        print(f"  Accuracy:           {metrics['accuracy']:.4f}")
        print(f"  Precision (macro):  {metrics['precision_macro']:.4f}")
        print(f"  Recall (macro):     {metrics['recall_macro']:.4f}")
        print(f"  F1-Score (macro):   {metrics['f1_macro']:.4f}")
        print(f"  F1-Score (weighted):{metrics['f1_weighted']:.4f}")
        print(f"  MCC:                {metrics['mcc']:.4f}")
        print(f"  Loss:               {metrics['loss']:.4f}")

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        fname = "best_model.pth" if is_best else f"checkpoint_epoch{epoch+1}.pth"
        path = os.path.join(self.save_dir, fname)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': self.best_val_acc,
            'config': self.config.__dict__,
        }, path)

    def _save_results(self, results: dict):
        """Save results to JSON."""
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        path = os.path.join(self.results_dir, "results.json")
        with open(path, 'w') as f:
            json.dump(results, f, indent=2, default=convert)
        print(f"\nResults saved to {path}")


def run_multi_seed_experiment(
    model_fn,
    train_ds, val_ds, test_ds,
    config: TrainingConfig,
    class_map: dict = None,
) -> dict:
    """Run experiment with multiple seeds for robustness analysis.

    Args:
        model_fn: Callable that returns a new model instance
        train/val/test_ds: Dataset objects
        config: Training configuration
        class_map: Label → class name mapping

    Returns:
        Summary dict with mean ± std for all metrics
    """
    all_metrics = []

    for i, seed in enumerate(config.seeds):
        print(f"\n{'#'*60}")
        print(f"Seed {i+1}/{len(config.seeds)}: {seed}")
        print(f"{'#'*60}")

        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        cfg = copy.deepcopy(config)
        cfg.seed = seed
        cfg.experiment_name = f"{config.experiment_name}_seed{seed}"

        model = model_fn()

        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size,
            shuffle=True, num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size,
            shuffle=False, num_workers=cfg.num_workers
        )
        test_loader = DataLoader(
            test_ds, batch_size=cfg.batch_size,
            shuffle=False, num_workers=cfg.num_workers
        )

        trainer = Trainer(
            model, train_loader, val_loader, test_loader,
            cfg, class_map
        )
        results = trainer.train()
        all_metrics.append(results['test_metrics'])

    # Aggregate
    summary = {}
    keys = all_metrics[0].keys()
    for key in keys:
        values = [m[key] for m in all_metrics]
        summary[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'values': values
        }

    print(f"\n{'='*60}")
    print("Multi-Seed Summary (mean ± std)")
    print(f"{'='*60}")
    for key in ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'mcc']:
        m = summary[key]['mean']
        s = summary[key]['std']
        print(f"  {key:20s}: {m:.4f} ± {s:.4f}")

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, f"{config.experiment_name}_multi_seed.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    return summary
