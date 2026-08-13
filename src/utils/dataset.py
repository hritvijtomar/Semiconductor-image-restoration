from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class SuperResolutionDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, augment=False, crop_size=64):
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)

        self.lr_files = sorted(self.lr_dir.glob("*.npy"))
        self.hr_files = sorted(self.hr_dir.glob("*.npy"))

        assert len(self.lr_files) == len(self.hr_files), \
            "LR and HR image counts do not match."

        self.augment = augment
        self.crop_size = crop_size

    def __len__(self):
        return len(self.lr_files)

    def augment_pair(self, lr, hr):
        """
        Apply identical geometric augmentations to LR and HR images.
        LR: 128x128
        HR: 256x256
        Scale factor: 2
        """

        # -------------------------
        # Random crop
        # -------------------------
        if self.crop_size is not None:
            h, w = lr.shape
            cs = self.crop_size

            top = random.randint(0, h - cs)
            left = random.randint(0, w - cs)

            # LR crop
            lr = lr[top:top + cs, left:left + cs]

            # Corresponding HR crop
            hr_top = top * 2
            hr_left = left * 2
            hr = hr[
                hr_top:hr_top + cs * 2,
                hr_left:hr_left + cs * 2
            ]

        # -------------------------
        # Horizontal flip
        # -------------------------
        if random.random() < 0.5:
            lr = np.fliplr(lr)
            hr = np.fliplr(hr)

        # -------------------------
        # Vertical flip
        # -------------------------
        if random.random() < 0.5:
            lr = np.flipud(lr)
            hr = np.flipud(hr)

        # -------------------------
        # Random rotation
        # -------------------------
        k = random.randint(0, 3)
        lr = np.rot90(lr, k)
        hr = np.rot90(hr, k)

        # -------------------------
        # Mild intensity augmentation
        # -------------------------
        if random.random() < 0.3:
            scale = random.uniform(0.95, 1.05)
            bias = random.uniform(-0.02, 0.02)

            lr = np.clip(lr * scale + bias, 0.0, 1.0)

        return lr.copy(), hr.copy()

    def __getitem__(self, idx):
        # Load noisy image
        lr = np.load(self.lr_files[idx]).astype(np.float32)

        # Load clean image
        hr = np.load(self.hr_files[idx]).astype(np.float32)

        # Apply augmentation
        if self.augment:
            lr, hr = self.augment_pair(lr, hr)

        # Add channel dimension
        lr = torch.from_numpy(lr).unsqueeze(0)
        hr = torch.from_numpy(hr).unsqueeze(0)

        return lr, hr