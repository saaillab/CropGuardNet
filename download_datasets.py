"""
Dataset Download Script
========================
Downloads all required datasets for CropGuardNet experiments.

Datasets:
  1. PlantVillage (New Plant Diseases Dataset) — Kaggle
  2. MangoLeafBD — Kaggle
  3. Rice Leaf Disease — Kaggle
  4. Cassava Leaf Disease — Kaggle

Usage:
  python download_datasets.py
  python download_datasets.py --dataset plantvillage
"""

import os
import sys
import argparse
import zipfile
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_RAW

# Kaggle dataset identifiers
KAGGLE_DATASETS = {
    "plantvillage": {
        "kaggle_id": "vipoooool/new-plant-diseases-dataset",
        "extract_to": os.path.join(DATA_RAW, "plantvillage_raw"),
        "description": "New Plant Diseases Dataset (87K images, 38 classes)"
    },
    "mangoleafbd": {
        "kaggle_id": "aryashah2k/mango-leaf-disease-dataset",
        "extract_to": os.path.join(DATA_RAW, "mangoleafbd"),
        "description": "MangoLeafBD Dataset (4K images, 8 classes)"
    },
    "rice": {
        "kaggle_id": "vbookshelf/rice-leaf-diseases",
        "extract_to": os.path.join(DATA_RAW, "rice_leaf"),
        "description": "Rice Leaf Diseases (5.9K images, 4 classes)"
    },
    "cassava": {
        "kaggle_id": "marquis03/bean-leaf-lesions-classification",
        "extract_to": os.path.join(DATA_RAW, "cassava"),
        "description": "Cassava Leaf Disease Classification (21K images, 5 classes)",
        "alt_kaggle_id": "c/cassava-leaf-disease-classification",
        "is_competition": True,
    },
    "corn": {
        "kaggle_id": "smaranjitghose/corn-or-maize-leaf-disease-dataset",
        "extract_to": os.path.join(DATA_RAW, "corn_maize"),
        "description": "Corn/Maize Leaf Disease Dataset (4K images, 4 classes)"
    }
}


def download_kaggle_dataset(name: str, kaggle_id: str, extract_to: str,
                            is_competition: bool = False):
    """Download a dataset from Kaggle."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        print(f"  [ERROR] Kaggle API authentication failed: {e}")
        print(f"  Please ensure ~/.kaggle/kaggle.json exists with your credentials.")
        print(f"  Get your API token from: https://www.kaggle.com/settings")
        return False

    os.makedirs(extract_to, exist_ok=True)
    print(f"  Downloading {name} ({kaggle_id})...")

    try:
        if is_competition:
            api.competition_download_files(kaggle_id, path=extract_to)
        else:
            api.dataset_download_files(kaggle_id, path=extract_to, unzip=True)
        print(f"  ✓ Downloaded to {extract_to}")
        return True
    except Exception as e:
        print(f"  [ERROR] Download failed: {e}")
        return False


def organize_plantvillage(raw_dir: str, target_dir: str):
    """Organize PlantVillage into per-crop subdirectories.

    The raw dataset has class names like 'Tomato___Early_blight'.
    We reorganize into: target_dir/Tomato/Early_blight/
    """
    print("\n  Organizing PlantVillage into per-crop structure...")

    # Find the actual image directories
    search_dirs = [raw_dir]
    for root, dirs, files in os.walk(raw_dir):
        if any(f.endswith(('.jpg', '.JPG', '.jpeg', '.png')) for f in files):
            parent = os.path.dirname(root)
            if parent not in search_dirs:
                search_dirs.append(parent)
            break

    source_dir = None
    for sd in search_dirs:
        subdirs = [d for d in os.listdir(sd)
                   if os.path.isdir(os.path.join(sd, d)) and '___' in d]
        if subdirs:
            source_dir = sd
            break

    if source_dir is None:
        # Try deeper
        for root, dirs, files in os.walk(raw_dir):
            subdirs = [d for d in dirs if '___' in d]
            if subdirs:
                source_dir = root
                break

    if source_dir is None:
        print(f"  [WARN] Could not find PlantVillage class directories in {raw_dir}")
        return

    for class_dir in sorted(os.listdir(source_dir)):
        full_path = os.path.join(source_dir, class_dir)
        if not os.path.isdir(full_path) or '___' not in class_dir:
            continue

        # Parse 'Crop___Disease' format
        parts = class_dir.split('___')
        if len(parts) == 2:
            crop = parts[0].strip()
            disease = parts[1].strip().replace(' ', '_')
        else:
            crop = class_dir
            disease = "unknown"

        dest = os.path.join(target_dir, crop, disease)
        os.makedirs(dest, exist_ok=True)

        count = 0
        for f in os.listdir(full_path):
            src = os.path.join(full_path, f)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(dest, f))
                count += 1

        print(f"    {crop}/{disease}: {count} images")


def main():
    parser = argparse.ArgumentParser(description="Download CropGuardNet datasets")
    parser.add_argument(
        '--dataset', type=str, default='all',
        choices=['all'] + list(KAGGLE_DATASETS.keys()),
        help="Which dataset to download (default: all)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("CropGuardNet Dataset Downloader")
    print("=" * 60)

    os.makedirs(DATA_RAW, exist_ok=True)

    datasets_to_download = (
        KAGGLE_DATASETS.keys() if args.dataset == 'all'
        else [args.dataset]
    )

    for name in datasets_to_download:
        info = KAGGLE_DATASETS[name]
        print(f"\n{'─'*40}")
        print(f"Dataset: {info['description']}")
        print(f"{'─'*40}")

        if os.path.exists(info['extract_to']) and os.listdir(info['extract_to']):
            print(f"  [SKIP] Already exists at {info['extract_to']}")
            continue

        success = download_kaggle_dataset(
            name, info['kaggle_id'], info['extract_to'],
            is_competition=info.get('is_competition', False)
        )

        # Post-processing for PlantVillage
        if success and name == "plantvillage":
            organize_plantvillage(
                info['extract_to'],
                os.path.join(DATA_RAW, "plantvillage")
            )

    print(f"\n{'='*60}")
    print("Download complete!")
    print(f"Data directory: {DATA_RAW}")
    print(f"{'='*60}")

    # Summary
    print("\nDataset status:")
    for name, info in KAGGLE_DATASETS.items():
        path = info['extract_to']
        if os.path.exists(path) and os.listdir(path):
            count = sum(1 for _ in Path(path).rglob('*.jpg'))
            count += sum(1 for _ in Path(path).rglob('*.JPG'))
            count += sum(1 for _ in Path(path).rglob('*.png'))
            print(f"  ✓ {name:15s}: {count:,} images")
        else:
            print(f"  ✗ {name:15s}: not downloaded")


if __name__ == "__main__":
    main()
