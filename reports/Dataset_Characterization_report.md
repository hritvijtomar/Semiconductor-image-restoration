Project: KLA – AI-based restoration of degraded semiconductor inspection images

Phase: Phase 0 – dataset understanding

Status: Completed

Date: 7 August 2026

Dataset overview
Training pairs: 3,200
Test images: 400
Image format: .npy
Image type: grayscale
Supervision: paired (NoisyLR → GT)
Resolution
Ground truth resolution: 256 × 256
Input resolution: 128 × 128
Super-resolution scale factor: 2×
Data format

Ground truth images:

dtype: float32
range: [0, 1]

NoisyLR images:

dtype: float32
observed range across 200 sampled images:
minimum: -0.2201
maximum: 2.1580

The degraded images are not clipped to the GT range and contain values outside [0,1].

Statistical observations

Across 200 sampled image pairs:

Average |mean(GT) − mean(LR)|: 0.000595
Maximum |mean(GT) − mean(LR)|: 0.002568

This indicates that the degradation process preserves global brightness while corrupting local image structure.

Visual degradation characteristics

Visual inspection across multiple samples shows:

strong texture degradation,
noticeable grain/noise,
softened edges,
loss of fine high-frequency details,
larger errors in textured regions than in smooth regions.

The degradation appears to combine 2× downsampling with noise corruption.

Bicubic baseline

Using bicubic upsampling as the restoration baseline:

Average PSNR: 22.920 dB
Median PSNR: 22.468 dB
Average SSIM: 0.5485
Median SSIM: 0.5880

PSNR ranges from 10.39 dB to 33.50 dB, indicating a broad range of restoration difficulty.

Distribution analysis

The PSNR distribution is centered around 21–24 dB with a small tail of very difficult images.

The SSIM distribution shows a substantial number of structurally challenging samples, suggesting that texture reconstruction is a major component of the task.

Engineering implications

The model must perform:

denoising,
edge restoration,
texture reconstruction,
2× super-resolution,

simultaneously.

Input images should initially be kept in their original float32 intensity range (including out-of-range values), and bicubic performance establishes a baseline that future models must significantly exceed.