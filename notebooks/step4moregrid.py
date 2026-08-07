from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

# ---------- Locate project root ----------
project_root = Path.cwd()
if project_root.name == "notebooks":
    project_root = project_root.parent

gt_dir = project_root / "train" / "train" / "GT"
lr_dir = project_root / "train" / "train" / "NoisyLR"

# ---------- Select 4 random samples ----------
files = sorted([f.name for f in gt_dir.glob("*.npy")])
random.seed(7)
sample_files = random.sample(files, 4)

fig, axes = plt.subplots(4, 3, figsize=(12, 16))

for i, fname in enumerate(sample_files):
    gt = np.load(gt_dir / fname)
    lr = np.load(lr_dir / fname)

    lr_up = np.array(Image.fromarray(lr).resize((256, 256), Image.BICUBIC))
    diff = np.abs(gt - lr_up)

    axes[i, 0].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axes[i, 0].set_title(f"GT\\n{fname}")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(lr_up, cmap="gray", vmin=0, vmax=1)
    axes[i, 1].set_title("Bicubic LR")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(diff, cmap="hot")
    axes[i, 2].set_title("Difference")
    axes[i, 2].axis("off")

plt.tight_layout()
plt.show()