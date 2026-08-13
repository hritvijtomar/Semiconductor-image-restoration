from pathlib import Path

import matplotlib.pyplot as plt
import torch

# =========================
# Paths
# =========================

PROJECT_ROOT = Path(__file__).parent.parent

HISTORY_FILE = PROJECT_ROOT / "training_history_v2.pth"
OUTPUT_DIR = PROJECT_ROOT / "results" / "training"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Load history
# =========================

history = torch.load(HISTORY_FILE, map_location="cpu")

train_loss = history["train_loss"]
val_loss = history["val_loss"]
val_psnr = history["val_psnr"]

epochs = list(range(1, len(train_loss) + 1))

# =========================
# Train loss plot
# =========================

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, linewidth=2)
plt.title("RCAN training loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "train_loss.png", dpi=300)
plt.close()

# =========================
# Validation loss plot
# =========================

plt.figure(figsize=(8, 5))
plt.plot(epochs, val_loss, linewidth=2)
plt.title("RCAN validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "val_loss.png", dpi=300)
plt.close()

# =========================
# Validation PSNR plot
# =========================

plt.figure(figsize=(8, 5))
plt.plot(epochs, val_psnr, linewidth=2)
plt.title("RCAN validation PSNR")
plt.xlabel("Epoch")
plt.ylabel("PSNR (dB)")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "val_psnr.png", dpi=300)
plt.close()

# =========================
# Print summary
# =========================

best_epoch = val_psnr.index(max(val_psnr)) + 1
best_psnr = max(val_psnr)

print("Training plots saved to:", OUTPUT_DIR)
print(f"Best epoch: {best_epoch}")
print(f"Best validation PSNR: {best_psnr:.2f} dB")