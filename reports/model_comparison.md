# Phase 3: model comparison report

## Objective

This phase evaluates two super-resolution architectures, **EDSR** and **RCAN**, on the KLA image restoration dataset. The goal is to compare reconstruction quality and inference speed in order to select the most suitable model for the V1 restoration pipeline.

## Dataset

* Low-resolution input: 128 × 128 grayscale images
* Ground truth: 256 × 256 grayscale images
* Total image pairs: 3200
* Training split: 2880 images (90%)
* Validation split: 320 images (10%)

## Training configuration

| Parameter     | Value                              |
| ------------- | ---------------------------------- |
| Batch size    | 8                                  |
| Epochs        | 10                                 |
| Learning rate | 1e-4                               |
| Optimizer     | Adam                               |
| Loss function | Charbonnier loss                   |
| GPU           | NVIDIA GeForce RTX 3050 Laptop GPU |

## Models evaluated

### EDSR

* 16 residual blocks
* 64 feature channels
* Residual scaling: 0.1
* PixelShuffle upsampling

### RCAN

* 5 residual groups
* 5 residual blocks per group
* 64 feature channels
* Channel attention reduction: 16
* PixelShuffle upsampling

## Evaluation metrics

The models were evaluated on the validation set using:

* **PSNR (Peak Signal-to-Noise Ratio)** for reconstruction quality
* **Average inference time per image** for deployment efficiency

## Benchmark results

| Model | Average PSNR |     Inference time |
| ----- | -----------: | -----------------: |
| EDSR  | **27.55 dB** | **20.93 ms/image** |
| RCAN  | **27.66 dB** | **37.68 ms/image** |

## Visual comparison

Qualitative inspection of the validation outputs shows that both models significantly improve image quality compared to the noisy low-resolution input.

Observed differences:

* RCAN produces slightly cleaner edges.
* RCAN reduces noise more effectively in textured regions.
* Smooth surfaces contain fewer reconstruction artifacts.
* EDSR produces visually similar results while remaining substantially faster.

Example comparisons are available in:

`results/benchmark/comparison_00.png`

through

`results/benchmark/comparison_09.png`

## Analysis

RCAN achieved a **0.11 dB PSNR improvement** over EDSR, indicating a modest but consistent quality gain. However, RCAN required **approximately 1.8× longer inference time**.

This creates a clear trade-off:

* **EDSR** is the better choice for real-time or resource-constrained deployment.
* **RCAN** is the better choice when maximum restoration quality is the primary objective.

## Model selection for V1

For the V1 restoration pipeline, **RCAN is selected as the primary restoration model** because image restoration quality is prioritized over inference speed.

EDSR will be retained as the baseline model for future optimization experiments.

## Conclusion

Phase 3 establishes a reproducible benchmark between EDSR and RCAN. The comparison demonstrates that RCAN provides slightly superior reconstruction quality, while EDSR offers significantly faster inference. These benchmark results will serve as the reference point for the optimization work in Phase 4.
