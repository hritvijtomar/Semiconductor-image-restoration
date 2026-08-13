import torch
import lpips
from skimage.metrics import structural_similarity as ssim

# Initialize LPIPS model once
_lpips_model = lpips.LPIPS(net="alex")


def calculate_psnr(pred, target):
    """
    Compute PSNR between two tensors in the range [0, 1].

    pred:   [B, 1, H, W]
    target: [B, 1, H, W]
    """

    mse = torch.mean((pred - target) ** 2)

    if mse == 0:
        return float("inf")

    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))

    return psnr.item()


def calculate_ssim(pred, target):
    """
    Compute average SSIM over a batch.

    pred:   [B, 1, H, W]
    target: [B, 1, H, W]
    """

    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    values = []

    for p, t in zip(pred, target):
        p = p.squeeze()
        t = t.squeeze()

        values.append(
            ssim(
                t,
                p,
                data_range=1.0
            )
        )

    return float(sum(values) / len(values))


def calculate_lpips(pred, target):
    """
    Compute average LPIPS over a batch.

    pred:   [B, 1, H, W]
    target: [B, 1, H, W]

    Returns:
        Average LPIPS (lower is better).
    """

    # Move LPIPS model to the same device as the tensors
    model = _lpips_model.to(pred.device)

    # LPIPS expects 3-channel images in [-1, 1]
    pred = pred.repeat(1, 3, 1, 1)
    target = target.repeat(1, 3, 1, 1)

    pred = pred * 2 - 1
    target = target * 2 - 1

    with torch.no_grad():
        value = model(pred, target)

    return value.mean().item()