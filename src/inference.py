import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Add model folder
sys.path.append(str(Path(__file__).parent / "models"))

from rcan import RCAN

# =========================
# Argument parser
# =========================

parser = argparse.ArgumentParser(description="RCAN image restoration inference")

parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Input directory containing .npy images"
)

parser.add_argument(
    "--output",
    type=str,
    required=True,
    help="Output directory for restored images"
)

parser.add_argument(
    "--checkpoint",
    type=str,
    default=str(Path(__file__).parent.parent / "best_rcan_v2.pth"),
    help="Path to trained RCAN checkpoint"
)

args = parser.parse_args()

# =========================
# Device
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# =========================
# Load model
# =========================

model = RCAN(
    channels=64,
    num_groups=5,
    num_blocks=5,
    reduction=16
)

model.load_state_dict(
    torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=True
    )
)

model.to(device)
model.eval()

# =========================
# Directories
# =========================

input_dir = Path(args.input)
output_dir = Path(args.output)

output_dir.mkdir(parents=True, exist_ok=True)

input_files = sorted(input_dir.glob("*.npy"))

print(f"Found {len(input_files)} images")

# =========================
# Inference
# =========================

with torch.no_grad():

    for file in input_files:

        lr = np.load(file).astype(np.float32)

        tensor = (
            torch.from_numpy(lr)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

        pred = model(tensor)

        restored = (
            pred.squeeze()
            .cpu()
            .numpy()
        )

        output_path = output_dir / file.name

        np.save(output_path, restored)

        print(f"Saved: {output_path.name}")

print("Inference complete.")