import numpy as np
import torch
from torch.utils.data import Dataset


def trfl_normalize(x):
    x = x.astype(np.float32).copy()

    for b in range(x.shape[2]):
        band_max = np.max(x[:, :, b])

        if abs(band_max) > 1e-8:
            x[:, :, b] /= band_max

    return x


def extract_patches(
    T1,
    T2,
    GT,
    tiles,
    patch_size=32,
    stride=16,
    tile_size=64
):
    x1_list = []
    x2_list = []
    y_list = []

    for tile_r, tile_c in tiles:

        r0 = tile_r * tile_size
        c0 = tile_c * tile_size

        t1_tile = T1[r0:r0+tile_size, c0:c0+tile_size]
        t2_tile = T2[r0:r0+tile_size, c0:c0+tile_size]
        gt_tile = GT[r0:r0+tile_size, c0:c0+tile_size]

        for r in range(0, tile_size-patch_size+1, stride):
            for c in range(0, tile_size-patch_size+1, stride):

                x1_list.append(
                    t1_tile[r:r+patch_size, c:c+patch_size]
                )

                x2_list.append(
                    t2_tile[r:r+patch_size, c:c+patch_size]
                )

                y_list.append(
                    gt_tile[r:r+patch_size, c:c+patch_size]
                )

    return (
        np.stack(x1_list),
        np.stack(x2_list),
        np.stack(y_list)
    )


class MangroveChangeDataset(Dataset):

    def __init__(self, x1, x2, y):

        self.x1 = torch.tensor(
            x1.transpose(0, 3, 1, 2),
            dtype=torch.float32
        )

        self.x2 = torch.tensor(
            x2.transpose(0, 3, 1, 2),
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.long
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.x1[idx],
            self.x2[idx],
            self.y[idx]
        )
