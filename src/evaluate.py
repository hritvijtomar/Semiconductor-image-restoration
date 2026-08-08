import sys
from pathlib import Path

# Add model and utility folders to Python path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "models"))
sys.path.insert(0, str(BASE_DIR / "utils"))

import torch
from torch.utils.data import DataLoader, Subset

from edsr import EDSR
from dataset import SuperResolutionDataset
from metrics import calculate_psnr


# -----------------------
# Dataset
# -----------------------
dataset = SuperResolutionDataset(
    lr_dir=r"C:\Users\abhis\Desktop\KLA\train\train\NoisyLR",
    hr_dir=r"C:\Users\abhis\Desktop\KLA\train\train\GT"
)

# Evaluate on 50 images
dataset = Subset(dataset, range(50))

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False
)


# -----------------------
# Device
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# -----------------------
# Model
# -----------------------
model = EDSR(
    channels=32,
    num_blocks=4
).to(device)

model.load_state_dict(
    torch.load("edsr_v1_debug.pth", map_location=device)
)

model.eval()


# -----------------------
# Evaluation
# -----------------------
psnr_values = []

with torch.no_grad():

    for lr, hr in loader:

        lr = lr.to(device)
        hr = hr.to(device)

        pred = model(lr)

        pred = torch.clamp(pred, 0.0, 1.0)

        psnr = calculate_psnr(pred, hr)

        psnr_values.append(psnr)

avg_psnr = sum(psnr_values) / len(psnr_values)

print(f"Average PSNR: {avg_psnr:.2f} dB")