import sys
import time
from pathlib import Path

# Add model and utility folders to Python path
sys.path.append(str(Path(__file__).parent / "models"))
sys.path.append(str(Path(__file__).parent / "utils"))

import torch
from torch.utils.data import DataLoader, Subset

from rcan import RCAN
from dataset import SuperResolutionDataset
from metrics import calculate_psnr
from losses import CharbonnierLoss

# =========================
# Configuration
# =========================

PROJECT_ROOT = Path(__file__).parent.parent

LR_DIR = r"C:\Users\abhis\Desktop\KLA\train\train\NoisyLR"
HR_DIR = r"C:\Users\abhis\Desktop\KLA\train\train\GT"

BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 1e-4
TRAIN_SPLIT = 0.9
RANDOM_SEED = 42

# =========================
# Device
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# =========================
# Dataset
# =========================

base_dataset = SuperResolutionDataset(
    lr_dir=LR_DIR,
    hr_dir=HR_DIR,
    augment=False
)

total_size = len(base_dataset)
train_size = int(TRAIN_SPLIT * total_size)

generator = torch.Generator().manual_seed(RANDOM_SEED)
indices = torch.randperm(total_size, generator=generator).tolist()

train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_dataset = Subset(
    SuperResolutionDataset(
        lr_dir=LR_DIR,
        hr_dir=HR_DIR,
        augment=True,
        crop_size=64
    ),
    train_indices
)

val_dataset = Subset(
    SuperResolutionDataset(
        lr_dir=LR_DIR,
        hr_dir=HR_DIR,
        augment=False,
        crop_size=None
    ),
    val_indices
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=device.type == "cuda"
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=device.type == "cuda"
)

print(f"Train images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")

# =========================
# Model
# =========================

model = RCAN(
    channels=64,
    num_groups=5,
    num_blocks=5,
    reduction=16
).to(device)

# =========================
# Loss and optimizer
# =========================

criterion = CharbonnierLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-6
)

best_psnr = 0.0
history = {
    "train_loss": [],
    "val_loss": [],
    "val_psnr": []
}

start_time = time.time()

# =========================
# Training loop
# =========================

for epoch in range(EPOCHS):

    model.train()
    train_loss = 0.0

    for lr, hr in train_loader:

        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)

        optimizer.zero_grad()

        pred = model(lr)

        loss = criterion(pred, hr)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()

    val_loss = 0.0
    val_psnr = 0.0

    with torch.no_grad():

        for lr, hr in val_loader:

            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)

            pred = model(lr)

            loss = criterion(pred, hr)

            val_loss += loss.item()

            val_psnr += calculate_psnr(pred, hr)

    val_loss /= len(val_loader)
    val_psnr /= len(val_loader)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_psnr"].append(val_psnr)

    scheduler.step()

    if val_psnr > best_psnr:

        best_psnr = val_psnr

        torch.save(
            model.state_dict(),
            PROJECT_ROOT / "best_rcan_v2.pth"
        )

        best_flag = " (best model saved)"
    else:
        best_flag = ""

    torch.save(
        model.state_dict(),
        PROJECT_ROOT / "last_rcan_v2.pth"
    )

    current_lr = scheduler.get_last_lr()[0]

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {val_loss:.6f} | "
        f"Val PSNR: {val_psnr:.2f} dB | "
        f"LR: {current_lr:.6f}"
        f"{best_flag}"
    )

training_time = time.time() - start_time

torch.save(
    history,
    PROJECT_ROOT / "training_history_v2.pth"
)

print(f"Training complete. Best validation PSNR: {best_psnr:.2f} dB")
print(f"Total training time: {training_time / 60:.2f} minutes")