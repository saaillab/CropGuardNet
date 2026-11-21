# CropGuardNet

> **CropGuard Research Series (Papers 1–6)** | [Umbrella Hub](https://saaillab.github.io/cropguard/)
> **[P1: CropGuardNet](https://github.com/saaillab/CropGuardNet)** · **[P2: CropKD-Net](https://github.com/saaillab/CropKD-Net)** · P3: CropMambaNet · P4: CropDomainNet · P5: CropContinual · P6: CropGuard-Severity

[![Project Page](https://img.shields.io/badge/Project%20Page-Live-2563eb)](https://saaillab.github.io/CropGuardNet/)
[![SAAIL Lab](https://img.shields.io/badge/SAAIL%20Lab-Papers%20Hub-0f172a)](https://saaillab.github.io/papers/cropguardnet/)
[![Status](https://img.shields.io/badge/Status-Under%20Review%20(AIIA)-059669)](https://saaillab.github.io/papers/cropguardnet/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official PyTorch implementation of **CropGuardNet** (**Paper 1** of the CropGuard Research Ecosystem, currently under review at *Artificial Intelligence in Agriculture*, Elsevier).

![CropGuardNet Architecture](figures/architecture.png)

## Key Highlights

- **Leakage-Free Data Curation:** Purges **80,238** exact SHA/UUID and perceptual-hash (pHash) duplicates across 8 crops and 38 classes, leaving **35,420 unique leaf images**.
- **Compact Hybrid Backbone:** **481,574 parameters** (**1.84 MB** FP32, **63.8M FLOPs**), combining MobileNetV3-Small stages, a Multi-Scale Feature Pyramid (MSFP), Squeeze-and-Excitation recalibration, a 2-layer Compact Vision Transformer encoder, and Crop-Aware Conditioning.
- **Cross-Crop Accuracy & Edge Speed:** Achieves **99.17%** accuracy (**0.9874** F1-macro across 38 classes; **0.075** ECE; **5.00 ms** CPU ONNX inference).
- **Four-Method XAI Verification:** Benchmarks Grad-CAM++, LayerCAM, LIME, and GradientSHAP with quantitative Deletion/Insertion AUC and Pointing Game faithfulness evaluation.

## Repository Structure

```text
CropGuardNet/
├── config.py                 # Model, dataset, preprocessing, and training configuration
├── main.py                   # CLI entry point (train / eval / ablation / multi-seed)
├── download_datasets.py      # Dataset download helper (original terms apply)
├── requirements.txt
├── src/
│   ├── models.py             # CropGuardNet, MSFP, SE, CompactTransformerEncoder, CropAwareConditioning
│   ├── dataset.py            # Dataset loading, stratified splits, augmentation pipeline
│   ├── train.py              # Training loop & multi-seed orchestration
│   ├── evaluate.py           # Classification, calibration, efficiency, and ONNX benchmarks
│   ├── xai.py                # Grad-CAM++, LayerCAM, LIME, GradientSHAP & faithfulness metrics
│   └── robustness.py         # Synthetic weather/sensor degradations & PlantDoc evaluation
├── results/                  # Verified dataset summary & test classification report
├── figures/                  # Architecture diagram, XAI gallery, and Pareto plots
└── docs/                     # GitHub Pages project website
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

# 3. Inspect checkpoint weights
# Pretrained CropGuardNet FP32 checkpoint (2.06 MB) is attached to Release v1.0.0.
```

> **Note on Weights & Raw Datasets:** Heavy comparison baseline checkpoints (`VGG-16`, `ResNet-50`) and the raw 35,420 third-party `.jpg` images are excluded from git history to respect upstream licenses and repository size limits.

## Citation

```bibtex
@article{nyalala2026cropguardnet,
  title={CropGuardNet: A Unified Lightweight CNN-Transformer Framework with Multi-Scale Attention and Explainable AI for Cross-Crop Leaf Disease Detection},
  author={Nyalala, Innocent and Makame, Twamaa and Khamis, Khamis and Gudla, Surabhi and others},
  journal={Artificial Intelligence in Agriculture},
  year={2026},
  note={Under Review}
}
```
