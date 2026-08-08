import sys
from pathlib import Path

# Add model and utility folders to Python path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "models"))
sys.path.insert(0, str(BASE_DIR / "utils"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from edsr import EDSR
from dataset import SuperResolutionDataset


# -----------------------
# Dataset
# -----------------------
dataset = SuperResolutionDataset(
    lr_dir=r"C:\Users\abhis\Desktop\KLA\train\train\NoisyLR",
    hr_dir=r"C:\Users\abhis\Desktop\KLA\train\train\GT"
)

# Use only 200 samples for fast debugging
dataset = Subset(dataset, range(200))

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)


# -----------------------
# Device
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# -----------------------
# Model
# -----------------------
model = EDSR(
    channels=32,
    num_blocks=4
).to(device)


# -----------------------
# Loss and optimizer
# -----------------------
criterion = nn.L1Loss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)


# -----------------------
# Training loop
# -----------------------
epochs = 1

for epoch in range(epochs):

    model.train()
    epoch_loss = 0.0

    for batch_idx, (lr, hr) in enumerate(loader):

        lr = lr.to(device)
        hr = hr.to(device)

        optimizer.zero_grad()

        pred = model(lr)

        loss = criterion(pred, hr)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

        print(
            f"Batch {batch_idx + 1}/{len(loader)} | "
            f"Loss: {loss.item():.6f}"
        )

    avg_loss = epoch_loss / len(loader)

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Average Loss: {avg_loss:.6f}"
    )


# -----------------------
# Save model
# -----------------------
torch.save(model.state_dict(), "edsr_v1_debug.pth")

print("Model saved as edsr_v1_debug.pth")