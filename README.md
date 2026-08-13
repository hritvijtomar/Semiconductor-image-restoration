# AI-Based Restoration of Degraded Images for Semiconductor Inspection

## Overview

This project develops a deep learning pipeline for restoring degraded grayscale inspection images using convolutional super-resolution and restoration networks. The objective is to improve image quality for semiconductor inspection applications while preserving fine structural details.

The project was developed for the **KLA Semiconductor Image Restoration Challenge**.

## Problem statement

Semiconductor inspection images often suffer from:

* noise
* blur
* low resolution
* acquisition artifacts

These degradations reduce the visibility of critical micro-scale features. Our approach learns a mapping from degraded low-quality images to clean high-quality images using supervised deep learning.

## Methodology

The restoration pipeline consists of:

1. **Low-quality image input**
2. **Residual feature extraction**
3. **Residual Channel Attention Network (RCAN) restoration**
4. **Reconstructed high-quality output**

## Model architecture

### EDSR baseline

We first implemented **Enhanced Deep Super-Resolution (EDSR)** as a baseline restoration network.

### RCAN

The final model uses **Residual Channel Attention Network (RCAN)** with:

* 5 residual groups
* 5 residual blocks per group
* 64 feature channels
* channel attention mechanism
* long and short residual connections

RCAN selectively emphasizes informative feature channels and preserves fine image structures more effectively than the baseline model.

## Loss function

Training uses **Charbonnier Loss**, a robust differentiable variant of L1 loss that is well suited for image restoration tasks.

## Data processing

Training includes:

* horizontal flip
* vertical flip
* 90° rotation augmentation
* paired LR-HR transformation consistency

A **90/10 train-validation split** is used with a fixed random seed to ensure reproducibility.

## Training configuration

| Parameter     |             Value |
| ------------- | ----------------: |
| Epochs        |                30 |
| Batch size    |                 8 |
| Optimizer     |              Adam |
| Learning rate |              1e-4 |
| Scheduler     | CosineAnnealingLR |
| Loss          |       Charbonnier |

## Evaluation metrics

We evaluate restoration quality using:

* **PSNR (Peak Signal-to-Noise Ratio)**
* **SSIM (Structural Similarity Index)**
* **LPIPS (Learned Perceptual Image Patch Similarity)**

## Benchmark results

Validation set: **320 images**

| Model       | PSNR (dB) |       SSIM |      LPIPS |    Inference |
| ----------- | --------: | ---------: | ---------: | -----------: |
| EDSR        |     27.55 |     0.7330 |     0.3253 |     16.16 ms |
| **RCAN v2** | **27.86** | **0.7498** | **0.3137** | **39.50 ms** |

RCAN improves reconstruction quality across all evaluation metrics.

## Repository structure

```text
Semiconductor-image-restoration/
├── src/
│   ├── train.py
│   ├── inference.py
│   ├── benchmark.py
│   ├── plot_training.py
│   ├── visualize_results.py
│   ├── visualize_inference.py
│   ├── models/
│   └── utils/
├── reports/
├── results/
├── notebooks/
├── README.md
└── requirements.txt
```

## Installation

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Training

Update the dataset paths in `src/train.py` and run:

```bash
python src/train.py
```

The script automatically:

* trains RCAN
* evaluates on the validation set
* saves the best checkpoint
* records training history

## Benchmarking

Run:

```bash
python src/benchmark.py
```

This reports:

* PSNR
* SSIM
* LPIPS
* inference time
* GPU information

## Inference

Restore images from an input directory:

```bash
python src/inference.py --input test_input --output test_output
```

The script loads the trained RCAN checkpoint and restores all supported images automatically.

## Training curves

Training and validation curves are generated using:

```bash
python src/plot_training.py
```

Outputs are stored in `results/training/`.

## Failure-case analysis

The repository includes examples where RCAN:

* successfully removes heavy noise
* preserves edge structures
* struggles with extremely low-information or highly textured regions

A detailed discussion is provided in `reports/failure_analysis.md`.

## Current status

* EDSR baseline implemented
* RCAN implemented
* augmentation pipeline completed
* optimized training completed
* PSNR/SSIM/LPIPS evaluation completed
* inference pipeline completed
* benchmarking completed
* failure analysis completed
