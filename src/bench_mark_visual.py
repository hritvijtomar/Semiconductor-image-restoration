import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "models"))
sys.path.append(str(Path(__file__).parent / "utils"))

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from edsr import EDSR
from rcan import RCAN
from dataset import SuperResolutionDataset
from metrics import calculate_psnr

# =========================
# Configuration
# =========================

PROJECT_ROOT = Path(__file__).parent.parent

LR_DIR = r"C:\Users\abhis\Desktop\KLA\train\train\NoisyLR"
HR_DIR = r"C:\Users\abhis\Desktop\KLA\train\train\GT"

NUM_IMAGES = 10
TRAIN_SPLIT = 0.9

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# =========================
# Dataset
# =========================

dataset = SuperResolutionDataset(LR_DIR, HR_DIR)

train_size = int(TRAIN_SPLIT * len(dataset))
val_size = len(dataset) - train_size

_, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False
)

# =========================
# Models
# =========================

edsr = EDSR(
    channels=64,
    num_blocks=16,
    res_scale=0.1
).to(device)

rcan = RCAN(
    channels=64,
    num_groups=5,
    num_blocks=5,
    reduction=16
).to(device)

edsr.load_state_dict(
    torch.load(
        PROJECT_ROOT / "best_edsr_v1.pth",
        map_location=device,
        weights_only=True
    )
)

rcan.load_state_dict(
    torch.load(
        PROJECT_ROOT / "best_rcan_v1.pth",
        map_location=device,
        weights_only=True
    )
)

edsr.eval()
rcan.eval()

# =========================
# Output directory
# =========================

out_dir = PROJECT_ROOT / "results" / "benchmark"
out_dir.mkdir(parents=True, exist_ok=True)

# =========================
# Generate comparisons
# =========================

with torch.no_grad():

    for idx, (lr, hr) in enumerate(loader):

        if idx >= NUM_IMAGES:
            break

        lr = lr.to(device)
        hr = hr.to(device)

        edsr_out = edsr(lr)
        rcan_out = rcan(lr)

        edsr_psnr = calculate_psnr(edsr_out, hr)
        rcan_psnr = calculate_psnr(rcan_out, hr)

        lr_img = lr.squeeze().cpu().numpy()
        edsr_img = edsr_out.squeeze().cpu().numpy()
        rcan_img = rcan_out.squeeze().cpu().numpy()
        hr_img = hr.squeeze().cpu().numpy()

        fig, axes = plt.subplots(1, 4, figsize=(12, 3))

        axes[0].imshow(lr_img, cmap="gray")
        axes[0].set_title("LR")

        axes[1].imshow(edsr_img, cmap="gray")
        axes[1].set_title(f"EDSR\\n{edsr_psnr:.2f} dB")

        axes[2].imshow(rcan_img, cmap="gray")
        axes[2].set_title(f"RCAN\\n{rcan_psnr:.2f} dB")

        axes[3].imshow(hr_img, cmap="gray")
        axes[3].set_title("Ground truth")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()

        save_path = out_dir / f"comparison_{idx:02d}.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"Saved {save_path.name}")

print(f"Done. Results saved to {out_dir}")