import sys
from pathlib import Path

# Add model and utility folders to Python path
sys.path.append(str(Path(__file__).parent / "models"))
sys.path.append(str(Path(__file__).parent / "utils"))

import torch
from torch.utils.data import DataLoader, random_split

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
EPOCHS = 10
LEARNING_RATE = 1e-4
TRAIN_SPLIT = 0.9

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

full_dataset = SuperResolutionDataset(
    lr_dir=LR_DIR,
    hr_dir=HR_DIR
)

train_size = int(TRAIN_SPLIT * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
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

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=20,
    gamma=0.5
)

best_psnr = 0.0

# =========================
# Training loop
# =========================

for epoch in range(EPOCHS):

    # ---------------------
    # Training
    # ---------------------

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

    # ---------------------
    # Validation
    # ---------------------

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

    # ---------------------
    # Learning rate update
    # ---------------------

    scheduler.step()

    # ---------------------
    # Checkpointing
    # ---------------------

    if val_psnr > best_psnr:

        best_psnr = val_psnr

        torch.save(
            model.state_dict(),
            PROJECT_ROOT / "best_rcan_v1.pth"
        )

        best_flag = " (best model saved)"
    else:
        best_flag = ""

    torch.save(
        model.state_dict(),
        PROJECT_ROOT / "last_rcan_v1.pth"
    )

    # ---------------------
    # Logging
    # ---------------------

    current_lr = scheduler.get_last_lr()[0]

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {val_loss:.6f} | "
        f"Val PSNR: {val_psnr:.2f} dB | "
        f"LR: {current_lr:.6f}"
        f"{best_flag}"
    )

print(f"Training complete. Best validation PSNR: {best_psnr:.2f} dB")