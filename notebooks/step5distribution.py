from pathlib import Path
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# =====================================================
# Locate project root
# =====================================================
project_root = Path.cwd()
if project_root.name == "notebooks":
    project_root = project_root.parent

gt_dir = project_root / "train" / "train" / "GT"
lr_dir = project_root / "train" / "train" / "NoisyLR"

# =====================================================
# Randomly sample 200 image pairs
# =====================================================
files = sorted([f.name for f in gt_dir.glob("*.npy")])
random.seed(42)
sample_files = random.sample(files, 200)

psnr_scores = []
ssim_scores = []

# =====================================================
# Compute metrics
# =====================================================
for fname in sample_files:
    gt = np.load(gt_dir / fname)
    lr = np.load(lr_dir / fname)

    lr_up = np.array(
        Image.fromarray(lr).resize((256, 256), Image.BICUBIC),
        dtype=np.float32
    )

    psnr_scores.append(
        peak_signal_noise_ratio(gt, lr_up, data_range=1.0)
    )

    ssim_scores.append(
        structural_similarity(gt, lr_up, data_range=1.0)
    )

# =====================================================
# Print summary
# =====================================================
print("=" * 40)
print("BICUBIC BASELINE (200 IMAGE PAIRS)")
print("=" * 40)

print(f"Average PSNR : {np.mean(psnr_scores):.3f} dB")
print(f"Median PSNR  : {np.median(psnr_scores):.3f} dB")
print(f"Min PSNR     : {np.min(psnr_scores):.3f} dB")
print(f"Max PSNR     : {np.max(psnr_scores):.3f} dB")

print()

print(f"Average SSIM : {np.mean(ssim_scores):.4f}")
print(f"Median SSIM  : {np.median(ssim_scores):.4f}")
print(f"Min SSIM     : {np.min(ssim_scores):.4f}")
print(f"Max SSIM     : {np.max(ssim_scores):.4f}")

# =====================================================
# PSNR Histogram
# =====================================================
plt.figure(figsize=(7,4))
plt.hist(psnr_scores, bins=20)
plt.title("Distribution of PSNR (Bicubic Baseline)")
plt.xlabel("PSNR (dB)")
plt.ylabel("Number of images")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# =====================================================
# SSIM Histogram
# =====================================================
plt.figure(figsize=(7,4))
plt.hist(ssim_scores, bins=20)
plt.title("Distribution of SSIM (Bicubic Baseline)")
plt.xlabel("SSIM")
plt.ylabel("Number of images")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()