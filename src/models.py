import torch
import torch.nn as nn


class SharedEncoder(nn.Module):
    def __init__(self, in_channels=12):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x


class CNNChangeDetector(nn.Module):
    def __init__(self, in_channels=12, num_classes=4):
        super().__init__()

        self.encoder = SharedEncoder(in_channels)

        self.fusion = nn.Sequential(
            nn.Conv2d(64 * 3, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),

            nn.Conv2d(96, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Conv2d(
            64,
            num_classes,
            kernel_size=1
        )

    def forward(self, x1, x2):
        f1 = self.encoder(x1)
        f2 = self.encoder(x2)

        diff = torch.abs(f1 - f2)

        fused = torch.cat(
            [f1, f2, diff],
            dim=1
        )

        fused = self.fusion(fused)

        return self.classifier(fused)
