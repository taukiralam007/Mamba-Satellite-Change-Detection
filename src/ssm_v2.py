import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ssm_v1 import SelectiveSSMBlock


class SSMChangeDetectorV2(nn.Module):
    """
    Selective SSM-CD V2.
    32x32 input -> 16x16 latent representation.
    """

    def __init__(
        self,
        in_channels=12,
        num_classes=4,
        dim=64
    ):
        super().__init__()

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

        self.fusion = nn.Sequential(
            nn.Conv2d(
                dim * 3,
                dim,
                kernel_size=1
            ),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        self.ssm1 = SelectiveSSMBlock(dim)
        self.ssm2 = SelectiveSSMBlock(dim)

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

        f1 = self.encoder(t1)
        f2 = self.encoder(t2)

        diff = f2 - f1

        x = torch.cat(
            [f1, f2, diff],
            dim=1
        )

        x = self.fusion(x)

        B, C, H, W = x.shape

        x = x.flatten(2).transpose(1, 2)

        x = self.ssm1(x)
        x = self.ssm2(x)

        x = x.transpose(1, 2).reshape(
            B, C, H, W
        )

        x = self.refine(x)

        x = F.interpolate(
            x,
            size=t1.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        return self.classifier(x)
