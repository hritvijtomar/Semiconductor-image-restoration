from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# =========================
# Configuration
# =========================

PROJECT_ROOT = Path(__file__).parent.parent

INPUT_DIR = PROJECT_ROOT / "test_output"
OUTPUT_DIR = PROJECT_ROOT / "results" / "inference"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(INPUT_DIR.glob("*.npy"))

print(f"Found {len(files)} restored images")

# =========================
# Convert to PNG
# =========================

for file in files:

    image = np.load(file)

    plt.figure(figsize=(4, 4))
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    output_path = OUTPUT_DIR / f"{file.stem}.png"

    plt.savefig(
        output_path,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()

    print(f"Saved: {output_path.name}")

print(f"Done. PNG images saved to {OUTPUT_DIR}")