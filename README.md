# CropGuardNet

> **SAAIL Lab · Precision Agriculture & Crop Pathology** | [Lab Website](https://saaillab.github.io/) · [All Publications](https://saaillab.github.io/publications/)

[![Project Page](https://img.shields.io/badge/Project%20Page-Live-14532d)](https://saaillab.github.io/CropGuardNet/)
[![SAAIL Lab](https://img.shields.io/badge/SAAIL%20Lab-Papers%20Hub-0f172a)](https://saaillab.github.io/papers/cropguardnet/)
[![Status](https://img.shields.io/badge/Status-Under%20Peer%20Review-b45309)](https://saaillab.github.io/papers/cropguardnet/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official PyTorch and ONNX Runtime implementation of **CropGuardNet** (*Manuscript Under Peer Review*).

![CropGuardNet Architecture](figures/architecture.png)

## Key Highlights

- **Leakage-Free Data Curation:** Purges **80,238** duplicate leaf images via specimen UUID grouping and 64-bit DCT perceptual-hash (pHash) auditing across 8 crops and 38 classes, establishing a clean benchmark of **35,420 unique leaf images** (`n = 5,314` held-out test set).
- **Compact Hybrid Backbone:** **481,574 trainable parameters** (**1.84 MB** FP32, **63.8M FLOPs**), combining a half-width MobileNetV3-Small (`alpha = 0.50`) stem, a 3-branch dilated Multi-Scale Feature Pyramid (`r in {1, 2, 4}`), depthwise separable refinement, squeeze-and-excitation attention, a 1-layer 2-head Compact ViT encoder (`10,176` params), dual-path GAP fusion, and Crop-Aware Conditioning.
- **Cross-Crop Accuracy & Edge Latency:** Reaches **99.17%** accuracy, **0.9874** F1-macro, **0.9914** MCC, and **5.00 ms** (`200.0` FPS) single-thread CPU ONNX Runtime latency, eliminating cross-crop false positives (`99.79%` automatic crop routing accuracy under Mode A1-Auto).
- **Robustness, Cross-Crop Probing & XAI Faithfulness:** Leads 20-regime corruption evaluation (`83.61%` mean accuracy), 8-fold Leave-One-Crop-Out probing (`81.22%`, `0.8879` AUC-ROC), PlantDoc external field transfer (`63.66%` Raw RGB after 20-epoch adaptation), and four-family XAI faithfulness (`LayerCAM` Deletion AUC `0.7262`, `GradientSHAP` Insertion AUC `0.9259`).

## Repository Structure

```text
CropGuardNet/
├── config.py                 # Model, dataset, preprocessing, and training configuration
├── main.py                   # CLI entry point (train / eval / ablation / multi-seed)
├── download_datasets.py      # Dataset download helper (original terms apply)
├── requirements.txt
├── src/
│   ├── models.py             # CropGuardNet, MSFP, SE, CompactTransformerEncoder, CropAwareConditioning
│   ├── dataset.py              # Dataset loading, stratified splits, augmentation pipeline
│   ├── train.py              # Training loop & multi-seed orchestration
│   ├── evaluate.py           # Classification, calibration, efficiency, and ONNX benchmarks
│   ├── xai.py                # Grad-CAM++, LayerCAM, LIME, GradientSHAP & faithfulness metrics
│   └── robustness.py         # Synthetic weather/sensor degradations & PlantDoc evaluation
├── results/                  # Verified dataset summary & test classification report
├── figures/                  # Submitted TikZ architecture diagram, XAI gallery, and Pareto plots
└── docs/                     # Standalone GitHub Pages project website
```

## Quick Start

```bash
git clone https://github.com/saaillab/CropGuardNet.git
cd CropGuardNet
pip install -r requirements.txt

# 1. Fetch official source datasets into data/raw/
python download_datasets.py

# 2. Train CropGuardNet (seed 42)
python main.py --mode train --model cropguardnet
```

> **Note on Weights & Raw Datasets:** Heavy comparison baseline checkpoints (`VGG-16`, `ResNet-50`) and the raw 35,420 third-party `.jpg` images are excluded from git history to respect upstream licenses and repository size limits.

## Citation

```bibtex
@misc{nyalala2026cropguardnet,
  title={CropGuardNet: A Unified Lightweight CNN-Transformer Framework with Multi-Scale Attention and Explainable AI for Cross-Crop Leaf Disease Detection},
  author={Nyalala, Innocent},
  howpublished={Manuscript Under Peer Review (SAAIL Lab)},
  year={2026},
  url={https://saaillab.github.io/papers/cropguardnet/}
}
```
