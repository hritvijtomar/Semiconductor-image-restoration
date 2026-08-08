from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SuperResolutionDataset(Dataset):
    def __init__(self, lr_dir, hr_dir):
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)

        self.lr_files = sorted(self.lr_dir.glob("*.npy"))
        self.hr_files = sorted(self.hr_dir.glob("*.npy"))

        assert len(self.lr_files) == len(self.hr_files), \
            "LR and HR image counts do not match."

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx):
        # Load noisy image
        lr = np.load(self.lr_files[idx]).astype(np.float32)

        # Load clean image
        hr = np.load(self.hr_files[idx]).astype(np.float32)

        # Add channel dimension: [128,128] -> [1,128,128]
        lr = torch.from_numpy(lr).unsqueeze(0)
        hr = torch.from_numpy(hr).unsqueeze(0)

        return lr, hr