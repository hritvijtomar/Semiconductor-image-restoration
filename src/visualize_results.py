import sys
from pathlib import Path

# Add model and utility folders
sys.path.append(str(Path(__file__).parent / "models"))
sys.path.append(str(Path(__file__).parent / "utils"))

import torch
import matplotlib.pyplot as plt
from torch.utils.data import random_split

from edsr import EDSR
from dataset import SuperResolutionDataset
from metrics import calculate_psnr

# =========================
# Configuration
# =========================

PROJECT_ROOT = Path(__file__).parent.parent

LR_DIR = r"C:\Users\abhis\Desktop\KLA\train\train\NoisyLR"
HR_DIR = r"C:\Users\abhis\Desktop\KLA\train\train\GT"

CHECKPOINT = PROJECT_ROOT / "best_edsr_v1.pth"

NUM_IMAGES = 5
TRAIN_SPLIT = 0.9

# =========================
# Device
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# =========================
# Dataset
# =========================

full_dataset = SuperResolutionDataset(
    lr_dir=LR_DIR,
    hr_dir=HR_DIR
)

train_size = int(TRAIN_SPLIT * len(full_dataset))
val_size = len(full_dataset) - train_size

_, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# =========================
# Model
# =========================

model = EDSR(
    channels=64,
    num_blocks=16,
    res_scale=0.1
).to(device)

model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()

# =========================
# Output directory
# =========================

output_dir = PROJECT_ROOT / "results"
output_dir.mkdir(exist_ok=True)

# =========================
# Visualization
# =========================

with torch.no_grad():

    for i in range(NUM_IMAGES):

        lr, hr = val_dataset[i]

        lr_batch = lr.unsqueeze(0).to(device)
        hr_batch = hr.unsqueeze(0).to(device)

        pred = model(lr_batch)

        psnr = calculate_psnr(pred, hr_batch)

        lr_img = lr.squeeze().cpu().numpy()
        pred_img = pred.squeeze().cpu().numpy()
        hr_img = hr.squeeze().cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(lr_img, cmap="gray")
        axes[0].set_title("Noisy LR (128×128)")
        axes[0].axis("off")

        axes[1].imshow(pred_img, cmap="gray")
        axes[1].set_title(f"EDSR Output\\nPSNR: {psnr:.2f} dB")
        axes[1].axis("off")

        axes[2].imshow(hr_img, cmap="gray")
        axes[2].set_title("Ground Truth (256×256)")
        axes[2].axis("off")

        plt.tight_layout()

        save_path = output_dir / f"comparison_{i:02d}.png"
        plt.savefig(save_path, dpi=200)
        plt.close()

        print(f"Saved {save_path}")

print(f"Visualization complete. Results saved to: {output_dir}")