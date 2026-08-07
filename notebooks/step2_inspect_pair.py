from pathlib import Path
import numpy as np

# ---------- Locate project root ----------
project_root = Path.cwd()
if project_root.name == "notebooks":
    project_root = project_root.parent

# ---------- Load one image pair ----------
gt_path = project_root / "train" / "train" / "GT" / "000000.npy"
lr_path = project_root / "train" / "train" / "NoisyLR" / "000000.npy"

gt = np.load(gt_path)
lr = np.load(lr_path)

# ---------- Print detailed statistics ----------
def image_stats(img, name):
    print(f"===== {name} =====")
    print(f"Shape              : {img.shape}")
    print(f"Dtype              : {img.dtype}")
    print(f"Height × Width     : {img.shape[0]} × {img.shape[1]}")
    print(f"Min                : {img.min():.6f}")
    print(f"Max                : {img.max():.6f}")
    print(f"Mean               : {img.mean():.6f}")
    print(f"Std                : {img.std():.6f}")
    print(f"Dynamic Range      : {img.max() - img.min():.6f}")
    print()

image_stats(gt, "Ground Truth (GT)")
image_stats(lr, "Noisy Low Resolution (NoisyLR)")

# ---------- Resolution scale factor ----------
scale_h = gt.shape[0] / lr.shape[0]
scale_w = gt.shape[1] / lr.shape[1]

print(f"Super-resolution scale factor: {scale_h:.1f}× (height), {scale_w:.1f}× (width)")