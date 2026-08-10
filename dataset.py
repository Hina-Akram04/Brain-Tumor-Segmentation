import os
import h5py
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset


class BraTSDataset(Dataset):
    """Loads BraTS h5 slices, normalizes, resizes, and builds a 4-class label map."""

    def __init__(self, metadata_df, slices_dir, img_size=128):
        self.df = metadata_df.reset_index(drop=True)
        self.slices_dir = slices_dir
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fname = os.path.basename(self.df.loc[idx, 'slice_path'])
        with h5py.File(os.path.join(self.slices_dir, fname), 'r') as f:
            image = f['image'][:].astype(np.float32)   # (240,240,4)
            mask = f['mask'][:].astype(np.uint8)        # (240,240,3)

        # normalize each MRI modality channel independently
        for c in range(image.shape[-1]):
            ch = image[..., c]
            mean, std = ch.mean(), ch.std()
            image[..., c] = (ch - mean) / (std + 1e-8)

        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # collapse 3 binary sub-region channels into one class-index map
        label = np.zeros((self.img_size, self.img_size), dtype=np.int64)
        label[mask[..., 0] == 1] = 1  # NCR/NET
        label[mask[..., 1] == 1] = 2  # ED
        label[mask[..., 2] == 1] = 3  # ET

        image = torch.from_numpy(image).permute(2, 0, 1).float()  # (4,H,W)
        label = torch.from_numpy(label).long()                     # (H,W)

        return image, label