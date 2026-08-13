import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "models"))
sys.path.append(str(Path(__file__).parent / "utils"))

import torch
from torch.utils.data import DataLoader, random_split

from edsr import EDSR
from rcan import RCAN
from dataset import SuperResolutionDataset
from metrics import calculate_psnr, calculate_ssim, calculate_lpips

# =========================
# Configuration
# =========================

PROJECT_ROOT = Path(__file__).parent.parent

LR_DIR = r"C:\Users\abhis\Desktop\KLA\train\train\NoisyLR"
HR_DIR = r"C:\Users\abhis\Desktop\KLA\train\train\GT"

BATCH_SIZE = 1
TRAIN_SPLIT = 0.9

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
if device.type == "cuda":
    GPU_NAME = torch.cuda.get_device_name(0)
    print(f"GPU: {GPU_NAME}")
else:
    GPU_NAME = "CPU"

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

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"Validation images: {len(val_dataset)}")

# =========================
# Benchmark function
# =========================

def benchmark_model(model, checkpoint_path, model_name):

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True
        )
    )

    model.to(device)
    model.eval()

    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    total_time = 0.0

    with torch.no_grad():

        for lr, hr in val_loader:

            lr = lr.to(device)
            hr = hr.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            pred = model(lr)

            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start

            total_time += elapsed
            total_psnr += calculate_psnr(pred, hr)
            total_ssim += calculate_ssim(pred, hr)
            total_lpips += calculate_lpips(pred, hr)

    avg_psnr = total_psnr / len(val_loader)
    avg_ssim = total_ssim / len(val_loader)
    avg_lpips = total_lpips / len(val_loader)
    avg_time = total_time / len(val_loader)

    print()
    print(f"{model_name}:")
    print(f"  Average PSNR : {avg_psnr:.2f} dB")
    print(f"  Average SSIM : {avg_ssim:.4f}")
    print(f"  Average LPIPS: {avg_lpips:.4f}")
    print(f"  Avg inference: {avg_time * 1000:.2f} ms/image")

    return avg_psnr, avg_ssim, avg_lpips, avg_time

# =========================
# Models
# =========================

edsr = EDSR(
    channels=64,
    num_blocks=16,
    res_scale=0.1
)

rcan = RCAN(
    channels=64,
    num_groups=5,
    num_blocks=5,
    reduction=16
)

# =========================
# Run benchmark
# =========================

edsr_psnr, edsr_ssim, edsr_lpips, edsr_time = benchmark_model(
    edsr,
    PROJECT_ROOT / "best_edsr_v1.pth",
    "EDSR"
)

rcan_psnr, rcan_ssim, rcan_lpips, rcan_time = benchmark_model(
    rcan,
    PROJECT_ROOT / "best_rcan_v2.pth",
    "RCAN v2"
)

# =========================
# Save report
# =========================

report_dir = PROJECT_ROOT / "reports"
report_dir.mkdir(exist_ok=True)

report_path = report_dir / "benchmark_results.txt"

with open(report_path, "w") as f:

    f.write("Phase 5 Benchmark Results\n")
    f.write("========================\n\n")

    f.write(f"Device: {device}\n")
    f.write(f"GPU: {GPU_NAME}\n")
    f.write(f"Batch size: {BATCH_SIZE}\n")
    f.write(f"Validation images: {len(val_dataset)}\n\n")

    f.write("EDSR\n")
    f.write(f"PSNR: {edsr_psnr:.2f} dB\n")
    f.write(f"SSIM: {edsr_ssim:.4f}\n")
    f.write(f"LPIPS: {edsr_lpips:.4f}\n")
    f.write(f"Inference: {edsr_time * 1000:.2f} ms/image\n\n")

    f.write("RCAN v2\n")
    f.write(f"PSNR: {rcan_psnr:.2f} dB\n")
    f.write(f"SSIM: {rcan_ssim:.4f}\n")
    f.write(f"LPIPS: {rcan_lpips:.4f}\n")
    f.write(f"Inference: {rcan_time * 1000:.2f} ms/image\n")

print(f"\nBenchmark report saved to: {report_path}")