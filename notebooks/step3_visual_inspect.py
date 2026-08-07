from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ---------- Locate project root ----------
project_root = Path.cwd()
if project_root.name == "notebooks":
    project_root = project_root.parent

# ---------- Load one image pair ----------
gt = np.load(project_root / "train" / "train" / "GT" / "000000.npy")
lr = np.load(project_root / "train" / "train" / "NoisyLR" / "000000.npy")

# ---------- Bicubic upsample ----------
lr_img = Image.fromarray(lr)
lr_up = np.array(lr_img.resize((256, 256), Image.BICUBIC))

# ---------- Difference map ----------
diff = np.abs(gt - lr_up)

# ---------- Display ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(gt, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Ground Truth (256×256)")
axes[0].axis("off")

axes[1].imshow(lr_up, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Bicubic Upsampled NoisyLR")
axes[1].axis("off")

im = axes[2].imshow(diff, cmap="hot")
axes[2].set_title("Absolute Difference")
axes[2].axis("off")

plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()