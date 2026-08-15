"""
image_utils.py

Image I/O, validation, and visualization helpers for the Streamlit frontend.

All functions operate on plain ``numpy.ndarray`` objects so this module has
no dependency on Streamlit and can be unit tested on its own.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image

try:  # matplotlib is used only for the histogram / difference-map colormap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    _MPL_AVAILABLE = True
except Exception:  # pragma: no cover - matplotlib ships with requirements
    _MPL_AVAILABLE = False


class InvalidImageError(ValueError):
    """Raised when an uploaded/loaded .npy file cannot be used as input."""


@dataclass
class ImageInfo:
    filename: str
    shape: Tuple[int, ...]
    dtype: str
    is_grayscale: bool
    value_range: Tuple[float, float]


# ---------------------------------------------------------------------------
# Loading & validation
# ---------------------------------------------------------------------------

def load_npy_from_bytes(data: bytes, filename: str = "uploaded.npy") -> np.ndarray:
    """
    Parse raw bytes from an uploaded .npy file into a numpy array.

    Raises:
        InvalidImageError: if the bytes are not a valid / readable .npy file.
    """
    try:
        buffer = io.BytesIO(data)
        array = np.load(buffer, allow_pickle=False)
    except Exception as exc:
        raise InvalidImageError(
            f"'{filename}' could not be read as a .npy file. It may be "
            "corrupted or saved in an unsupported format."
        ) from exc

    return array


def validate_image_array(array: np.ndarray, filename: str = "image") -> None:
    """
    Validate that an array is a usable single-channel (grayscale) image.

    Raises:
        InvalidImageError: with a human-readable explanation of the problem.
    """
    if not isinstance(array, np.ndarray):
        raise InvalidImageError(f"'{filename}' is not a valid numpy array.")

    squeezable = array.squeeze()

    if squeezable.ndim != 2:
        raise InvalidImageError(
            f"'{filename}' has shape {array.shape}, but a 2D grayscale "
            "image (H, W) is required."
        )

    if squeezable.size == 0:
        raise InvalidImageError(f"'{filename}' is empty.")

    if not np.isfinite(squeezable).all():
        raise InvalidImageError(
            f"'{filename}' contains NaN or infinite values and cannot be "
            "processed."
        )

    h, w = squeezable.shape
    if h < 8 or w < 8:
        raise InvalidImageError(
            f"'{filename}' is too small ({h}x{w}). Expected an image of at "
            "least 8x8 pixels."
        )


def get_image_info(array: np.ndarray, filename: str) -> ImageInfo:
    squeezed = array.squeeze()
    finite = squeezed[np.isfinite(squeezed)]
    value_range = (
        (float(finite.min()), float(finite.max())) if finite.size else (0.0, 0.0)
    )
    return ImageInfo(
        filename=filename,
        shape=tuple(squeezed.shape),
        dtype=str(array.dtype),
        is_grayscale=(squeezed.ndim == 2),
        value_range=value_range,
    )


def prepare_model_input(array: np.ndarray) -> np.ndarray:
    """
    Normalize an arbitrary grayscale array into float32 values in [0, 1],
    ready to be fed to the RCAN model.
    """
    arr = array.squeeze().astype(np.float32)

    lo, hi = float(arr.min()), float(arr.max())

    if hi <= 1.0 and lo >= 0.0:
        return arr  # already normalized

    if hi - lo < 1e-8:
        return np.zeros_like(arr)

    return (arr - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Format conversions
# ---------------------------------------------------------------------------

def to_uint8(array: np.ndarray) -> np.ndarray:
    """Convert a float/other array into a display-ready uint8 image."""
    arr = array.squeeze().astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())

    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)

    normalized = (arr - lo) / (hi - lo)
    return (normalized * 255.0).clip(0, 255).astype(np.uint8)


def array_to_pil(array: np.ndarray) -> Image.Image:
    return Image.fromarray(to_uint8(array), mode="L")


def array_to_png_bytes(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    array_to_pil(array).save(buf, format="PNG")
    return buf.getvalue()


def array_to_npy_bytes(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, array.astype(np.float32))
    return buf.getvalue()


def array_to_base64_png(array: np.ndarray) -> str:
    return base64.b64encode(array_to_png_bytes(array)).decode("utf-8")


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------

def plot_intensity_histogram(array: np.ndarray, label: str = "Restored", color: str = "#3b82f6"):
    """
    Build a dark-themed matplotlib figure of the grayscale intensity
    histogram. Returns None if matplotlib is unavailable.
    """
    if not _MPL_AVAILABLE:
        return None

    arr = to_uint8(array).ravel()

    fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
    fig.patch.set_facecolor("#0b1220")
    ax.set_facecolor("#0b1220")

    ax.hist(arr, bins=64, range=(0, 255), color=color, alpha=0.85, edgecolor="none")

    ax.set_title(f"{label} intensity histogram", color="#e5e7eb", fontsize=11)
    ax.set_xlabel("Pixel intensity", color="#9ca3af", fontsize=9)
    ax.set_ylabel("Frequency", color="#9ca3af", fontsize=9)
    ax.tick_params(colors="#9ca3af", labelsize=8)

    for spine in ax.spines.values():
        spine.set_color("#1f2937")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Difference map (demo validation mode only)
# ---------------------------------------------------------------------------

def compute_difference_map(restored: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    """
    Compute a normalized absolute-error map between a restored image and its
    ground truth. Both inputs are normalized to [0, 1] before differencing so
    the result is comparable regardless of the original value ranges.
    """
    r = prepare_model_input(restored)
    g = prepare_model_input(ground_truth)

    if r.shape != g.shape:
        raise InvalidImageError(
            f"Restored image shape {r.shape} does not match ground truth "
            f"shape {g.shape}; cannot compute a difference map."
        )

    return np.abs(r - g)


def difference_map_to_heatmap(diff: np.ndarray, cmap_name: str = "inferno") -> np.ndarray:
    """Render an absolute-error map as an RGB heatmap (uint8, HxWx3)."""
    lo, hi = float(diff.min()), float(diff.max())
    normalized = (diff - lo) / (hi - lo) if hi - lo > 1e-8 else np.zeros_like(diff)

    if _MPL_AVAILABLE:
        from matplotlib import colormaps
        colormap = colormaps[cmap_name]
        rgba = colormap(normalized)
        return (rgba[:, :, :3] * 255).astype(np.uint8)

    # Fallback: simple red-intensity heatmap without matplotlib
    heat = np.zeros((*normalized.shape, 3), dtype=np.uint8)
    heat[:, :, 0] = (normalized * 255).astype(np.uint8)
    return heat


# ---------------------------------------------------------------------------
# Comparison views: overlay blend & swipe widget markup
# ---------------------------------------------------------------------------

def blend_images(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    """Alpha-blend two grayscale arrays (0 = fully `a`, 1 = fully `b`)."""
    a_u8 = to_uint8(a).astype(np.float32)
    b_u8 = to_uint8(b).astype(np.float32)

    if a_u8.shape != b_u8.shape:
        b_img = Image.fromarray(b_u8.astype(np.uint8)).resize(
            (a_u8.shape[1], a_u8.shape[0])
        )
        b_u8 = np.array(b_img).astype(np.float32)

    blended = a_u8 * (1 - alpha) + b_u8 * alpha
    return blended.clip(0, 255).astype(np.uint8)


def build_swipe_widget_html(
    left_array: np.ndarray,
    right_array: np.ndarray,
    left_label: str = "Before",
    right_label: str = "After",
    height: int = 380,
) -> str:
    """
    Build a self-contained HTML/CSS/JS snippet implementing a before/after
    swipe comparison slider, for use with st.components.v1.html.
    """
    left_b64 = array_to_base64_png(left_array)
    right_b64 = array_to_base64_png(right_array)

    return f"""
    <div class="swipe-wrap">
      <style>
        .swipe-wrap {{
          position: relative;
          width: 100%;
          max-width: 640px;
          height: {height}px;
          margin: 0 auto;
          border-radius: 12px;
          overflow: hidden;
          border: 1px solid #1e293b;
          background: #0b1220;
          font-family: 'Inter', 'Segoe UI', sans-serif;
        }}
        .swipe-wrap img {{
          position: absolute;
          top: 0; left: 0;
          width: 100%; height: 100%;
          object-fit: contain;
          user-select: none;
          -webkit-user-drag: none;
          background: #0b1220;
        }}
        .swipe-wrap .swipe-right {{
          clip-path: inset(0 0 0 50%);
        }}
        .swipe-wrap input[type=range] {{
          position: absolute;
          bottom: 12px;
          left: 8%;
          width: 84%;
          accent-color: #3b82f6;
          z-index: 5;
        }}
        .swipe-wrap .tag {{
          position: absolute;
          top: 10px;
          padding: 3px 10px;
          font-size: 12px;
          font-weight: 600;
          border-radius: 999px;
          background: rgba(11, 18, 32, 0.85);
          color: #e5e7eb;
          border: 1px solid #1e293b;
          z-index: 4;
        }}
        .swipe-wrap .tag-left {{ left: 10px; }}
        .swipe-wrap .tag-right {{ right: 10px; }}
        .swipe-wrap .divider {{
          position: absolute;
          top: 0; bottom: 0;
          width: 2px;
          background: #3b82f6;
          left: 50%;
          z-index: 3;
          pointer-events: none;
        }}
      </style>
      <img class="swipe-left" src="data:image/png;base64,{left_b64}" />
      <img class="swipe-right" id="swipe-right-img" src="data:image/png;base64,{right_b64}" />
      <div class="divider" id="swipe-divider"></div>
      <div class="tag tag-left">{left_label}</div>
      <div class="tag tag-right">{right_label}</div>
      <input type="range" min="0" max="100" value="50" id="swipe-slider" />
      <script>
        const slider = document.getElementById('swipe-slider');
        const rightImg = document.getElementById('swipe-right-img');
        const divider = document.getElementById('swipe-divider');
        slider.addEventListener('input', (e) => {{
          const val = e.target.value;
          rightImg.style.clipPath = `inset(0 0 0 ${{val}}%)`;
          divider.style.left = `${{val}}%`;
        }});
      </script>
    </div>
    """
