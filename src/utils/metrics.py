import torch


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