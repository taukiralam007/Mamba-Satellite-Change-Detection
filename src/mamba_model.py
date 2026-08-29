import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class RealMambaBlock(nn.Module):
    def __init__(self, dim=64):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.mamba = Mamba(
            d_model=dim,
            d_state=16,
            d_conv=4,
            expand=2
        )

    def forward(self, x):
        return x + self.mamba(self.norm(x))


class MambaChangeDetector(nn.Module):
    """
    Lightweight Mamba-based multispectral change detector.

    Input:
        T1: [B, 12, H, W]
        T2: [B, 12, H, W]

    Output:
        logits: [B, 4, H, W]
    """

    def __init__(self, in_channels=12, num_classes=4, dim=64):
        super().__init__()

        # Shared feature encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels,
                32,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                dim,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        # Fuse T1, T2 and signed temporal difference
        self.fusion = nn.Sequential(
            nn.Conv2d(
                dim * 3,
                dim,
                kernel_size=1
            ),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        # Genuine Mamba blocks
        self.mamba1 = RealMambaBlock(dim)
        self.mamba2 = RealMambaBlock(dim)

        # Spatial refinement
        self.refine = nn.Sequential(
            nn.Conv2d(
                dim,
                dim,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                dim,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Conv2d(
            32,
            num_classes,
            kernel_size=1
        )

    def forward(self, t1, t2):

        # Shared bi-temporal feature extraction
        f1 = self.encoder(t1)
        f2 = self.encoder(t2)

        # Signed temporal difference preserves
        # temporal change direction
        diff = f2 - f1

        x = torch.cat(
            [f1, f2, diff],
            dim=1
        )

        x = self.fusion(x)

        B, C, H, W = x.shape

        # Convert image features into spatial sequence
        x = x.flatten(2).transpose(1, 2)

        # Mamba sequence modeling
        x = self.mamba1(x)
        x = self.mamba2(x)

        # Sequence -> spatial feature map
        x = x.transpose(1, 2).reshape(
            B,
            C,
            H,
            W
        )

        x = self.refine(x)

        # Restore original image resolution
        x = F.interpolate(
            x,
            size=t1.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        return self.classifier(x)
