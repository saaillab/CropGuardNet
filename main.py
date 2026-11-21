"""
CropGuardNet — Main Entry Point
=================================
Run training, evaluation, and ablation experiments.

Usage:
    python main.py --mode train --model cropguardnet
    python main.py --mode train --model leafxnet
    python main.py --mode train --model mangoxnet
    python main.py --mode ablation
    python main.py --mode multi_seed --model cropguardnet
    python main.py --mode test --checkpoint models/cropguardnet_v1/best_model.pth
"""

import os
import sys
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    TrainingConfig, DATASETS, CROP_LABELS, NUM_CROP_TYPES,
    MODEL_DIR, RESULTS_DIR
)
from src.models import CropGuardNet, LeafXNet, MangoXNet, build_model
from src.dataset import create_merged_dataset
from src.train import Trainer, run_multi_seed_experiment


def setup_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_dataloaders(config: TrainingConfig):
    """Load and prepare all datasets."""
    # Select which datasets to use
    active_datasets = []
    for key, cfg in DATASETS.items():
        if os.path.exists(cfg.root):
            active_datasets.append(cfg)

    if not active_datasets:
        print("\n[ERROR] No datasets found!")
        print("Please download datasets first:")
        print("  python download_datasets.py")
        print(f"\nExpected data directory: {config.image_size}")
        print("Or manually place datasets in data/raw/")

        # Try to find any image directory
        from config import DATA_RAW
        if os.path.exists(DATA_RAW):
            print(f"\nContents of {DATA_RAW}:")
            for item in os.listdir(DATA_RAW):
                print(f"  {item}")
        sys.exit(1)

    train_ds, val_ds, test_ds, metadata = create_merged_dataset(
        dataset_configs=active_datasets,
        crop_labels=CROP_LABELS,
        preprocessing_config={
            'use_clahe': config.use_clahe,
            'use_gaussian': config.use_gaussian_smooth,
            'use_highpass': config.use_highpass_filter,
        },
        image_size=config.image_size,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
    )

    config.num_classes = metadata['num_classes']
    print(f"\nDataset ready: {metadata['num_classes']} classes, "
          f"{metadata['total_images']} images")

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers
    )

    return train_loader, val_loader, test_loader, metadata


def train_single(config: TrainingConfig):
    """Train a single model."""
    setup_seed(config.seed)
    train_loader, val_loader, test_loader, metadata = get_dataloaders(config)

    # Build model
    model = build_model(
        config.model_name,
        num_classes=config.num_classes,
        num_crops=NUM_CROP_TYPES,
        backbone=config.backbone,
        pretrained=config.pretrained,
        use_multi_scale=config.use_multi_scale,
        use_se=config.use_se_attention,
        use_transformer=config.use_transformer,
        use_crop_conditioning=config.use_crop_conditioning,
        transformer_heads=config.transformer_heads,
        transformer_layers=config.transformer_layers,
        se_reduction=config.se_reduction,
        dropout=config.dropout,
    )

    # Print model info
    if hasattr(model, 'count_parameters'):
        params = model.count_parameters()
        print(f"\nModel: {config.model_name}")
        print(f"  Parameters: {params['total']:,}")
        print(f"  Size: {params['size_mb']:.2f} MB")
    else:
        total = sum(p.numel() for p in model.parameters())
        print(f"\nModel: {config.model_name}")
        print(f"  Parameters: {total:,}")

    # Train
    trainer = Trainer(
        model, train_loader, val_loader, test_loader,
        config, metadata.get('class_map')
    )
    results = trainer.train()
    return results


def run_ablation(config: TrainingConfig):
    """Run ablation study: remove components one at a time."""
    ablation_configs = {
        "full_model": {
            "use_multi_scale": True,
            "use_se_attention": True,
            "use_transformer": True,
            "use_crop_conditioning": True,
        },
        "no_multi_scale": {
            "use_multi_scale": False,
            "use_se_attention": True,
            "use_transformer": True,
            "use_crop_conditioning": True,
        },
        "no_se": {
            "use_multi_scale": True,
            "use_se_attention": False,
            "use_transformer": True,
            "use_crop_conditioning": True,
        },
        "no_transformer": {
            "use_multi_scale": True,
            "use_se_attention": True,
            "use_transformer": False,
            "use_crop_conditioning": True,
        },
        "no_crop_conditioning": {
            "use_multi_scale": True,
            "use_se_attention": True,
            "use_transformer": True,
            "use_crop_conditioning": False,
        },
        "backbone_only": {
            "use_multi_scale": False,
            "use_se_attention": False,
            "use_transformer": False,
            "use_crop_conditioning": False,
        },
    }

    # Preprocessing ablation
    preprocessing_configs = {
        "full_preprocessing": {
            "use_clahe": True,
            "use_gaussian_smooth": True,
            "use_highpass_filter": True,
        },
        "no_clahe": {
            "use_clahe": False,
            "use_gaussian_smooth": True,
            "use_highpass_filter": True,
        },
        "no_gaussian": {
            "use_clahe": True,
            "use_gaussian_smooth": False,
            "use_highpass_filter": True,
        },
        "no_highpass": {
            "use_clahe": True,
            "use_gaussian_smooth": True,
            "use_highpass_filter": False,
        },
        "no_preprocessing": {
            "use_clahe": False,
            "use_gaussian_smooth": False,
            "use_highpass_filter": False,
        },
    }

    results = {}

    print("\n" + "=" * 60)
    print("ARCHITECTURAL ABLATION STUDY")
    print("=" * 60)

    for name, overrides in ablation_configs.items():
        print(f"\n{'━'*50}")
        print(f"Ablation: {name}")
        print(f"{'━'*50}")

        cfg = TrainingConfig()
        cfg.experiment_name = f"ablation_{name}"
        cfg.model_name = "cropguardnet"
        for key, val in overrides.items():
            setattr(cfg, key, val)

        try:
            res = train_single(cfg)
            results[name] = res['test_metrics']
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            results[name] = {"error": str(e)}

    print("\n" + "=" * 60)
    print("PREPROCESSING ABLATION STUDY")
    print("=" * 60)

    for name, overrides in preprocessing_configs.items():
        print(f"\n{'━'*50}")
        print(f"Preprocessing Ablation: {name}")
        print(f"{'━'*50}")

        cfg = TrainingConfig()
        cfg.experiment_name = f"preproc_ablation_{name}"
        cfg.model_name = "cropguardnet"
        for key, val in overrides.items():
            setattr(cfg, key, val)

        try:
            res = train_single(cfg)
            results[f"preproc_{name}"] = res['test_metrics']
        except Exception as e:
            results[f"preproc_{name}"] = {"error": str(e)}

    # Print summary
    print("\n" + "=" * 60)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Configuration':<30s} {'Accuracy':>10s} {'F1-Macro':>10s} {'MCC':>10s}")
    print("─" * 62)
    for name, metrics in results.items():
        if 'error' in metrics:
            print(f"{name:<30s} {'ERROR':>10s}")
        else:
            print(f"{name:<30s} "
                  f"{metrics['accuracy']:>10.4f} "
                  f"{metrics['f1_macro']:>10.4f} "
                  f"{metrics['mcc']:>10.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="CropGuardNet Experiments")
    parser.add_argument(
        '--mode', type=str, default='train',
        choices=['train', 'ablation', 'multi_seed', 'test', 'info'],
        help="Experiment mode"
    )
    parser.add_argument('--model', type=str, default='cropguardnet',
                        choices=['cropguardnet', 'leafxnet', 'mangoxnet'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--backbone', type=str, default='efficientnet_lite0')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--experiment_name', type=str, default=None)
    args = parser.parse_args()

    config = TrainingConfig()
    config.model_name = args.model
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.seed = args.seed
    config.backbone = args.backbone
    config.experiment_name = (
        args.experiment_name or f"{args.model}_{args.backbone}"
    )

    if args.mode == 'info':
        # Just print model architecture info
        print("Building model for inspection...")
        model = build_model(args.model, num_classes=15, num_crops=8,
                          backbone=args.backbone)
        if hasattr(model, 'count_parameters'):
            p = model.count_parameters()
            print(f"Total parameters: {p['total']:,}")
            print(f"Trainable:        {p['trainable']:,}")
            print(f"Size:             {p['size_mb']:.2f} MB")
        print(f"\nArchitecture:\n{model}")

    elif args.mode == 'train':
        train_single(config)

    elif args.mode == 'ablation':
        run_ablation(config)

    elif args.mode == 'multi_seed':
        setup_seed(config.seed)
        train_loader, val_loader, test_loader, metadata = get_dataloaders(config)
        config.num_classes = metadata['num_classes']

        def model_fn():
            return build_model(
                config.model_name,
                num_classes=config.num_classes,
                num_crops=NUM_CROP_TYPES,
                backbone=config.backbone,
            )

        run_multi_seed_experiment(
            model_fn,
            train_loader.dataset, val_loader.dataset, test_loader.dataset,
            config, metadata.get('class_map')
        )

    elif args.mode == 'test':
        if args.checkpoint is None:
            print("Please provide --checkpoint path")
            sys.exit(1)
        # Load and evaluate
        _, _, test_loader, metadata = get_dataloaders(config)
        config.num_classes = metadata['num_classes']
        model = build_model(config.model_name, num_classes=config.num_classes)
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        trainer = Trainer(model, None, None, test_loader, config)
        _, _, metrics = trainer.validate(test_loader, desc="Test")
        trainer._print_metrics(metrics, "Test")


if __name__ == "__main__":
    main()
