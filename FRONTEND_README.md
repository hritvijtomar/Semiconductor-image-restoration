# Frontend — Semiconductor Image Restoration Dashboard

A Streamlit dashboard for the RCAN restoration pipeline in this repository.

## Files in this bundle

```
app.py                      # entry point — streamlit run app.py
frontend/
  __init__.py
  ui.py                      # theme (CSS) + reusable UI components
  image_utils.py             # .npy I/O, validation, histogram, diff map, swipe/overlay
  inference_wrapper.py       # cached model loading, inference timing, metrics
demo_samples/
  sample_input.npy           # synthetic 128x128 demo LR image (placeholder)
  sample_gt.npy               # synthetic 256x256 demo HR/ground-truth image (placeholder)
assets/
  logo.svg, icon.svg          # placeholder branding assets — swap for your own
requirements.txt              # your original deps + streamlit, pillow
src/models/blocks.py          # only needed if your repo doesn't already have it
src/models/upsampling.py      # only needed if your repo doesn't already have it
```

## Setup

1. Copy `app.py`, `frontend/`, `demo_samples/`, and `assets/` into the root of
   `Semiconductor-image-restoration/`.
2. Merge `requirements.txt` with your existing one (or just install the two
   new lines: `streamlit>=1.32`, `pillow`).
3. **Only if** `src/models/blocks.py` or `src/models/upsampling.py` don't
   already exist in your repo (they're imported by `rcan.py`/`edsr.py` but
   weren't in the files you shared), copy the two files under `src/models/`
   from this bundle. They were reconstructed and verified to load
   `best_rcan_v2.pth` and `best_edsr_v1.pth` with `strict=True` — every key
   matched.
4. Make sure `best_rcan_v2.pth` is at the repository root (next to `app.py`).
5. Run:

   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

## About `demo_samples/`

The bundled `sample_input.npy` / `sample_gt.npy` are **synthetic placeholder
images** (a generated chip-like pattern), included so Demo Validation Mode
works out of the box. They are not real inspection data and the metrics they
produce are not your reported benchmark numbers. Replace both files with a
real paired validation LR/HR sample from your dataset to get meaningful
PSNR/SSIM/LPIPS in that mode.

## Notes

- The app never hardcodes absolute paths — everything resolves relative to
  `app.py` via `pathlib`.
- The model and the LPIPS network are loaded once per session
  (`st.cache_resource`) so repeated clicks of "Restore Image" are fast.
- If `lpips`'s pretrained weights can't be downloaded (no internet in your
  deployment environment), the app degrades gracefully: PSNR/SSIM still
  compute, LPIPS shows "Unavailable" instead of crashing.
- Inference Mode never computes/display PSNR/SSIM/LPIPS, per your spec —
  it always shows "Ground truth not available."
