import sys
from pathlib import Path

import numpy as np
import torch


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "src" / "models"
CHECKPOINT = ROOT / "models" / "best_rcan_v2.pth"

# rcan.py imports upsampling.py directly, so add src/models
# to Python's import path.
sys.path.insert(0, str(MODEL_DIR))

from rcan import RCAN


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Load model
# ============================================================

def load_model():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT}"
        )

    model = RCAN(
        channels=64,
        num_groups=5,
        num_blocks=5,
        reduction=16,
        scale=2,
    )

    state_dict = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# Restore one image
# ============================================================

def restore_image(model, input_path, output_path):
    image = np.load(input_path).astype(np.float32)

    # Accept (H, W) or (H, W, 1)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[:, :, 0]

    if image.ndim != 2:
        raise ValueError(
            f"{input_path.name}: expected grayscale array with "
            f"shape (H, W) or (H, W, 1), got {image.shape}"
        )

    if not np.isfinite(image).all():
        raise ValueError(
            f"{input_path.name}: input contains NaN or Inf values"
        )

    tensor = (
        torch.from_numpy(np.ascontiguousarray(image))
        .unsqueeze(0)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():
        prediction = model(tensor)

    restored = prediction.squeeze().detach().cpu().numpy()

    # Required by the competition:
    # - grayscale
    # - finite
    # - values in [0, 1]
    restored = np.nan_to_num(
        restored,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    restored = np.clip(restored, 0.0, 1.0).astype(np.float32)

    # Ensure final output is exactly H x W.
    if restored.ndim != 2:
        raise ValueError(
            f"{input_path.name}: model produced unexpected shape "
            f"{restored.shape}"
        )

    np.save(output_path, restored)


# ============================================================
# Main
# ============================================================

def main():

    if len(sys.argv) != 3:
        print(
            "Usage: python run.py <input-dir> <output-dir>",
            file=sys.stderr,
        )
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    if not input_dir.exists():
        print(
            f"Input directory does not exist: {input_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not input_dir.is_dir():
        print(
            f"Input path is not a directory: {input_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob("*.npy"))

    print("=" * 60)
    print("KLA Semiconductor Image Restoration")
    print("=" * 60)
    print(f"Device:     {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Input:      {input_dir}")
    print(f"Output:     {output_dir}")
    print(f"Images:     {len(input_files)}")
    print("=" * 60)

    if not input_files:
        print("No .npy files found.")
        return

    model = load_model()

    for input_path in input_files:

        output_path = output_dir / input_path.name

        try:
            restore_image(
                model,
                input_path,
                output_path,
            )

            print(f"[OK] {input_path.name} -> {output_path.name}")

        except Exception as exc:
            print(
                f"[ERROR] {input_path.name}: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)

    print("=" * 60)
    print("Inference complete.")
    print(f"Generated {len(input_files)} output files.")
    print("=" * 60)


if __name__ == "__main__":
    main()