"""
Frontend package for the Semiconductor Image Restoration dashboard.

This package contains everything the Streamlit app (``app.py``) needs to
render the UI and talk to the existing training/inference codebase in
``src/`` without modifying it:

- ``image_utils``       -> .npy loading/validation, conversions, histograms,
                            difference maps, swipe/overlay comparison markup.
- ``inference_wrapper``  -> cached RCAN model loading, inference timing, and
                            PSNR/SSIM/LPIPS metric computation.
- ``ui``                 -> theming (CSS) and reusable UI components
                            (header, sidebar, metric cards, progress stages).
"""

__all__ = ["image_utils", "inference_wrapper", "ui"]
