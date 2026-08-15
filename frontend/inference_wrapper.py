"""
inference_wrapper.py

Bridges the Streamlit frontend to the existing restoration pipeline in
``src/`` (model definition + metric implementations) without modifying any
of that code. Model weights and the LPIPS network are cached with
``st.cache_resource`` so they are only loaded once per server session.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st
import torch

# ---------------------------------------------------------------------------
# Repository paths (no hardcoded absolute paths - resolved relative to this
# file so the app works regardless of where the repo is cloned)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
MODELS_DIR = SRC_DIR / "models"
UTILS_DIR = SRC_DIR / "utils"
DEFAULT_CHECKPOINT = REPO_ROOT / "best_rcan_v2.pth"

for _path in (MODELS_DIR, UTILS_DIR):
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


class CheckpointNotFoundError(FileNotFoundError):
    """Raised when the requested .pth checkpoint file does not exist."""


class ModelLoadError(RuntimeError):
    """Raised when a checkpoint exists but cannot be loaded into RCAN."""


@dataclass
class InferenceResult:
    restored: np.ndarray
    elapsed_ms: float
    device_label: str


@dataclass
class MetricResult:
    psnr: Optional[float]
    ssim: Optional[float]
    lpips: Optional[float]
    lpips_available: bool


# ---------------------------------------------------------------------------
# Device handling
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Select CUDA if available, otherwise fall back to CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_label(device: torch.device) -> str:
    if device.type == "cuda":
        try:
            return f"CUDA ({torch.cuda.get_device_name(device)})"
        except Exception:
            return "CUDA"
    return "CPU"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_rcan_model(
    checkpoint_path: str,
    channels: int = 64,
    num_groups: int = 5,
    num_blocks: int = 5,
    reduction: int = 16,
    scale: int = 2,
):
    """
    Load the trained RCAN model from a checkpoint. Cached across reruns for
    the current session so the (potentially large) checkpoint is only read
    from disk once.

    Returns:
        (model, device)

    Raises:
        CheckpointNotFoundError: if the checkpoint file does not exist.
        ModelLoadError: if the checkpoint exists but cannot be loaded, e.g.
            because it is corrupted or incompatible with the architecture.
    """
    ckpt_path = Path(checkpoint_path)

    if not ckpt_path.exists():
        raise CheckpointNotFoundError(
            f"Checkpoint not found at '{ckpt_path}'. Make sure "
            f"'{ckpt_path.name}' is present in the repository root."
        )

    try:
        from rcan import RCAN  # src/models/rcan.py
    except ImportError as exc:
        raise ModelLoadError(
            "Could not import the RCAN architecture from "
            f"'src/models/rcan.py'. Details: {exc}"
        ) from exc

    device = get_device()

    model = RCAN(
        channels=channels,
        num_groups=num_groups,
        num_blocks=num_blocks,
        reduction=reduction,
        scale=scale,
    )

    try:
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception as exc:
        raise ModelLoadError(
            f"'{ckpt_path.name}' could not be read. The file may be "
            f"corrupted or was not saved with torch.save(). Details: {exc}"
        ) from exc

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise ModelLoadError(
            f"'{ckpt_path.name}' is not compatible with the RCAN "
            f"architecture defined in src/models/rcan.py. Details: {exc}"
        ) from exc

    model.to(device)
    model.eval()

    return model, device


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, device: torch.device, lr_image: np.ndarray) -> InferenceResult:
    """Run a single grayscale image through the model, timing the forward pass."""
    tensor = (
        torch.from_numpy(np.ascontiguousarray(lr_image, dtype=np.float32))
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        prediction = model(tensor)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    restored = prediction.squeeze().detach().cpu().numpy()

    return InferenceResult(restored=restored, elapsed_ms=elapsed_ms, device_label=device_label(device))


# ---------------------------------------------------------------------------
# Metrics (demo validation mode only - requires ground truth)
# ---------------------------------------------------------------------------

def _fallback_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(1.0 / np.sqrt(mse)))


def _fallback_ssim(pred: np.ndarray, target: np.ndarray) -> Optional[float]:
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        return None
    return float(ssim(target, pred, data_range=1.0))


@st.cache_resource(show_spinner=False)
def _get_lpips_model():
    """
    Lazily load the LPIPS perceptual-similarity network used by
    src/utils/metrics.py. This performs a one-time download of pretrained
    AlexNet weights, so it is wrapped defensively: if it fails (e.g. no
    internet access in the deployment environment) LPIPS is simply reported
    as unavailable while PSNR/SSIM continue to work normally.
    """
    import lpips
    return lpips.LPIPS(net="alex")


def compute_metrics(
    restored: np.ndarray,
    ground_truth: np.ndarray,
) -> MetricResult:
    """
    Compute PSNR, SSIM and LPIPS between a restored image and its ground
    truth, preferring the repository's own implementations
    (src/utils/metrics.py) and falling back to local equivalents if that
    module cannot be imported.
    """
    restored_n = np.clip(restored.astype(np.float32), 0.0, 1.0)
    gt_n = np.clip(ground_truth.astype(np.float32), 0.0, 1.0)

    pred_t = torch.from_numpy(restored_n).unsqueeze(0).unsqueeze(0)
    target_t = torch.from_numpy(gt_n).unsqueeze(0).unsqueeze(0)

    psnr_val: Optional[float] = None
    ssim_val: Optional[float] = None
    lpips_val: Optional[float] = None
    lpips_ok = False

    try:
        from metrics import calculate_psnr, calculate_ssim  # src/utils/metrics.py
        psnr_val = calculate_psnr(pred_t, target_t)
        ssim_val = calculate_ssim(pred_t, target_t)
    except Exception:
        psnr_val = _fallback_psnr(restored_n, gt_n)
        ssim_val = _fallback_ssim(restored_n, gt_n)

    try:
        lpips_model = _get_lpips_model()
        pred3 = pred_t.repeat(1, 3, 1, 1) * 2 - 1
        target3 = target_t.repeat(1, 3, 1, 1) * 2 - 1
        with torch.no_grad():
            lpips_val = float(lpips_model(pred3, target3).mean().item())
        lpips_ok = True
    except Exception:
        lpips_val = None
        lpips_ok = False

    return MetricResult(psnr=psnr_val, ssim=ssim_val, lpips=lpips_val, lpips_available=lpips_ok)
