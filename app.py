"""
app.py

Streamlit frontend for the Semiconductor Image Restoration project.

Run from the repository root with:

    streamlit run app.py

The app wraps the trained RCAN checkpoint (best_rcan_v2.pth) and the
existing pipeline in src/ with a KLA-style inspection dashboard supporting
two modes:

- Inference Mode: restore an arbitrary uploaded .npy image (no ground
  truth -> no PSNR/SSIM/LPIPS).
- Demo Validation Mode: restore a predefined validation pair
  (demo_samples/sample_input.npy + sample_gt.npy) and compute real metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st
from PIL import Image

from frontend import image_utils, ui
from frontend.inference_wrapper import (
    CheckpointNotFoundError,
    ModelLoadError,
    DEFAULT_CHECKPOINT,
    compute_metrics,
    load_rcan_model,
    run_inference,
)

# ---------------------------------------------------------------------------
# Constants (no hardcoded absolute paths - everything relative to this file)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
DEMO_DIR = REPO_ROOT / "demo_samples"
DEMO_INPUT_PATH = DEMO_DIR / "sample_input.npy"
DEMO_GT_PATH = DEMO_DIR / "sample_gt.npy"
ASSETS_DIR = REPO_ROOT / "assets"

REPO_URL = "https://github.com/hritvijtomar/Semiconductor-image-restoration"

MODEL_INFO = {
    "architecture": "RCAN",
    "checkpoint": DEFAULT_CHECKPOINT.name,
    "epochs": 30,
    "loss": "Charbonnier",
    "scheduler": "CosineAnnealingLR",
}

BENCHMARK = {
    "PSNR": "27.86 dB",
    "SSIM": "0.7498",
    "LPIPS": "0.3137",
    "Inference": "39.5 ms/image",
}

INFERENCE_MODE = "Inference Mode"
DEMO_MODE = "Demo Validation Mode"


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Semiconductor Image Restoration",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

ui.inject_theme()

if "result" not in st.session_state:
    st.session_state.result = None  # holds the most recent restoration output
if "uploaded_array" not in st.session_state:
    st.session_state.uploaded_array = None
if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

ui.render_sidebar(
    architecture=MODEL_INFO["architecture"],
    checkpoint_name=MODEL_INFO["checkpoint"],
    epochs=MODEL_INFO["epochs"],
    loss=MODEL_INFO["loss"],
    scheduler=MODEL_INFO["scheduler"],
    benchmark=BENCHMARK,
    repo_url=REPO_URL,
)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

ui.render_header(model_badge="RCAN v2")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_demo_pair() -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Load the bundled demo validation pair, or None with an on-screen error."""
    if not DEMO_INPUT_PATH.exists() or not DEMO_GT_PATH.exists():
        ui.render_error_card(
            "Demo samples missing",
            f"Expected '{DEMO_INPUT_PATH.name}' and '{DEMO_GT_PATH.name}' in "
            f"'{DEMO_DIR.relative_to(REPO_ROOT)}/'. Generate or add a validation "
            "pair to use Demo Validation Mode.",
        )
        return None

    try:
        lr = np.load(DEMO_INPUT_PATH)
        hr = np.load(DEMO_GT_PATH)
    except Exception as exc:
        ui.render_error_card("Could not read demo samples", str(exc))
        return None

    return lr, hr


def _zoomed(array: np.ndarray, zoom: float) -> np.ndarray:
    """Return a center-cropped, upsampled version of `array` for zoomed viewing."""
    if zoom <= 1.0:
        return array

    h, w = array.shape[-2:]
    crop_h, crop_w = int(h / zoom), int(w / zoom)
    top = (h - crop_h) // 2
    left = (w - crop_w) // 2
    cropped = array[top:top + crop_h, left:left + crop_w]

    img = Image.fromarray(image_utils.to_uint8(cropped)).resize((w, h), Image.LANCZOS)
    return np.array(img)


def _do_restore(mode: str) -> None:
    """Load the model, run inference (and metrics in demo mode), store results."""
    with st.spinner(""):
        ui.run_progress_stages()

    try:
        model, device = load_rcan_model(str(DEFAULT_CHECKPOINT))
    except CheckpointNotFoundError as exc:
        ui.render_error_card("Checkpoint not found", str(exc))
        return
    except ModelLoadError as exc:
        ui.render_error_card("Model failed to load", str(exc))
        return
    except Exception as exc:  # pragma: no cover - defensive catch-all
        ui.render_error_card("Unexpected error while loading the model", str(exc))
        return

    if mode == INFERENCE_MODE:
        raw = st.session_state.uploaded_array
        if raw is None:
            ui.render_error_card("No image uploaded", "Upload a .npy image before restoring.")
            return

        lr = image_utils.prepare_model_input(raw)
        inf = run_inference(model, device, lr)

        st.session_state.result = {
            "mode": INFERENCE_MODE,
            "input": lr,
            "restored": inf.restored,
            "ground_truth": None,
            "elapsed_ms": inf.elapsed_ms,
            "device": inf.device_label,
        }

    else:  # DEMO_MODE
        pair = _load_demo_pair()
        if pair is None:
            return
        lr_raw, gt_raw = pair

        try:
            image_utils.validate_image_array(lr_raw, DEMO_INPUT_PATH.name)
            image_utils.validate_image_array(gt_raw, DEMO_GT_PATH.name)
        except image_utils.InvalidImageError as exc:
            ui.render_error_card("Invalid demo sample", str(exc))
            return

        lr = image_utils.prepare_model_input(lr_raw)
        gt = image_utils.prepare_model_input(gt_raw)

        inf = run_inference(model, device, lr)
        metrics = compute_metrics(inf.restored, gt)

        st.session_state.result = {
            "mode": DEMO_MODE,
            "input": lr,
            "restored": inf.restored,
            "ground_truth": gt,
            "elapsed_ms": inf.elapsed_ms,
            "device": inf.device_label,
            "metrics": metrics,
        }


# ---------------------------------------------------------------------------
# Main layout: left (controls) / right (results)
# ---------------------------------------------------------------------------

left_col, right_col = st.columns([0.38, 0.62], gap="large")

with left_col:
    st.markdown("#### Upload section")

    uploaded_file = st.file_uploader(
        "Upload .npy inspection image",
        type=["npy"],
        help="Grayscale semiconductor inspection image saved as a NumPy .npy file.",
    )

    if uploaded_file is not None:
        try:
            array = image_utils.load_npy_from_bytes(uploaded_file.getvalue(), uploaded_file.name)
            image_utils.validate_image_array(array, uploaded_file.name)
        except image_utils.InvalidImageError as exc:
            ui.render_error_card("Invalid file", str(exc))
            st.session_state.uploaded_array = None
            st.session_state.uploaded_name = None
        else:
            st.session_state.uploaded_array = array
            st.session_state.uploaded_name = uploaded_file.name
            info = image_utils.get_image_info(array, uploaded_file.name)
            ui.render_file_info_card(
                filename=info.filename,
                shape=info.shape,
                dtype=info.dtype,
                is_grayscale=info.is_grayscale,
            )

    st.markdown("#### Mode")
    mode = st.radio(
        "Restoration mode",
        options=[INFERENCE_MODE, DEMO_MODE],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )

    if mode == DEMO_MODE:
        st.caption(
            "Loads a predefined validation pair from `demo_samples/` and computes "
            "PSNR / SSIM / LPIPS against the ground truth."
        )

    st.markdown("&nbsp;", unsafe_allow_html=True)
    restore_clicked = st.button("🚀  Restore Image", type="primary", use_container_width=True)

    if restore_clicked:
        if mode == INFERENCE_MODE and st.session_state.uploaded_array is None:
            ui.render_error_card(
                "No image uploaded", "Please upload a .npy image before running restoration."
            )
        else:
            _do_restore(mode)

with right_col:
    st.markdown("#### Restoration preview")
    result = st.session_state.result

    if result is None:
        st.markdown(
            f"""
            <div style="border:1px dashed {ui.COLORS['border']}; border-radius:14px;
                        padding:2.4rem 1rem; text-align:center; color:{ui.COLORS['text_faint']};
                        margin-top:0.5rem;">
              Upload an image and click <b style="color:{ui.COLORS['blue']};">Restore Image</b>
              to see results here.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        result_mode = result["mode"]

        # ---- Comparison view ----
        if result_mode == INFERENCE_MODE:
            zoom = st.slider("🔍 Zoom", min_value=1.0, max_value=4.0, value=1.0, step=0.5)

            # FIX: RCAN output can fall slightly outside [0, 1] (e.g. from the
            # Charbonnier-trained head overshooting on noisy inputs). Streamlit's
            # st.image() raises "Data is outside [0.0, 1.0] and clamp is not set"
            # for such arrays, so clip once here and reuse everywhere below.
            restored_display = np.clip(result["restored"], 0.0, 1.0)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Input (Noisy LR)**")
                st.image(_zoomed(result["input"], zoom), use_container_width=True)
            with c2:
                st.markdown(f"**RCAN Restored**")
                st.image(_zoomed(restored_display, zoom), use_container_width=True)

        else:  # DEMO_MODE
            # FIX: same clipping applied here so Side-by-side / Overlay / Swipe
            # all render from the same, valid-range restored array. Metrics
            # below are computed separately (in _do_restore) from the raw,
            # unclipped model output, so this clip is display-only and does
            # not affect PSNR / SSIM / LPIPS.
            restored_display = np.clip(result["restored"], 0.0, 1.0)

            view = st.radio(
                "Comparison view",
                options=["Side-by-side", "Overlay", "Swipe"],
                index=0,
                horizontal=True,
                label_visibility="collapsed",
            )

            if view == "Side-by-side":
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**Input**")
                    st.image(result["input"], use_container_width=True)
                with c2:
                    st.markdown("**Ground Truth**")
                    st.image(result["ground_truth"], use_container_width=True)
                with c3:
                    st.markdown("**Restored**")
                    st.image(restored_display, use_container_width=True)

            elif view == "Overlay":
                alpha = st.slider(
                    "Blend: Ground Truth ↔ Restored", min_value=0.0, max_value=1.0,
                    value=0.5, step=0.05,
                )
                blended = image_utils.blend_images(result["ground_truth"], restored_display, alpha)
                st.image(np.clip(blended, 0.0, 1.0), use_container_width=True, caption="Ground Truth ↔ Restored overlay")

            else:  # Swipe
                import streamlit.components.v1 as components

                html = image_utils.build_swipe_widget_html(
                    result["ground_truth"], restored_display,
                    left_label="Ground Truth", right_label="Restored",
                )
                components.html(html, height=420)

        st.markdown("&nbsp;", unsafe_allow_html=True)
        st.markdown("#### Quality metrics")

        if result_mode == INFERENCE_MODE:
            ui.render_metric_cards([
                {"label": "PSNR", "value": "N/A", "available": False},
                {"label": "SSIM", "value": "N/A", "available": False},
                {"label": "LPIPS", "value": "N/A", "available": False},
            ])
            st.caption("Ground truth not available.")
        else:
            m = result["metrics"]
            psnr_str = f"{m.psnr:.2f} dB" if m.psnr is not None else "N/A"
            ssim_str = f"{m.ssim:.4f}" if m.ssim is not None else "N/A"
            lpips_str = f"{m.lpips:.4f}" if (m.lpips is not None and m.lpips_available) else "Unavailable"

            ui.render_metric_cards([
                {"label": "PSNR", "value": psnr_str, "available": m.psnr is not None},
                {"label": "SSIM", "value": ssim_str, "available": m.ssim is not None},
                {"label": "LPIPS", "value": lpips_str, "available": m.lpips_available},
            ])
            if not m.lpips_available:
                st.caption(
                    "LPIPS could not be computed (the pretrained perceptual network "
                    "could not be downloaded in this environment)."
                )

        info_cols = st.columns(3)
        info_cols[0].metric("Inference Time", f"{result['elapsed_ms']:.1f} ms")
        info_cols[1].metric("Device", result["device"])
        info_cols[2].metric("Model", "RCAN")

        if result_mode == INFERENCE_MODE:
            st.caption("Ground truth: Not available")

        ui.render_status_banner("success", "Completed successfully")

        # ---- Additional features ----
        st.markdown("&nbsp;", unsafe_allow_html=True)
        feat_cols = st.columns(2 if result_mode == INFERENCE_MODE else 2)

        with feat_cols[0]:
            with st.expander("📊 Grayscale intensity histogram"):
                fig = image_utils.plot_intensity_histogram(result["restored"], label="Restored")
                if fig is not None:
                    st.pyplot(fig, use_container_width=True)
                else:
                    st.caption("matplotlib is not available in this environment.")

        if result_mode == DEMO_MODE:
            with feat_cols[1]:
                with st.expander("🌡️ Absolute error heatmap"):
                    # FIX: use the clipped restored array here too so the heatmap
                    # image never receives out-of-range values.
                    diff = image_utils.compute_difference_map(restored_display, result["ground_truth"])
                    heatmap = image_utils.difference_map_to_heatmap(diff)
                    st.image(heatmap, use_container_width=True, caption="|Restored − Ground Truth|")

        # ---- Downloads ----
        st.markdown("&nbsp;", unsafe_allow_html=True)
        dl_cols = st.columns(2)
        with dl_cols[0]:
            st.download_button(
                "⬇️  Download restored .npy",
                data=image_utils.array_to_npy_bytes(result["restored"]),
                file_name="restored.npy",
                mime="application/octet-stream",
                use_container_width=True,
            )
        with dl_cols[1]:
            st.download_button(
                "⬇️  Download restored .png",
                # FIX: PNG export uses the clipped display array so the
                # downloaded image visually matches what's shown in the app.
                data=image_utils.array_to_png_bytes(restored_display),
                file_name="restored.png",
                mime="image/png",
                use_container_width=True,
            )