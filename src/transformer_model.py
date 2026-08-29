import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerChangeDetector(nn.Module):
    """
    Transformer-based bi-temporal change detector.

    Input:
        T1, T2: [B, 12, 32, 32]

    Output:
        logits: [B, 4, 32, 32]
    """

    def __init__(
        self,
        in_channels=12,
        num_classes=4,
        dim=64,
        num_heads=4,
        num_layers=2
    ):
        super().__init__()

        # Shared CNN encoder
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

        # Temporal feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(
                dim * 3,
                dim,
                kernel_size=1
            ),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=128,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

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

        f1 = self.encoder(t1)
        f2 = self.encoder(t2)

        # Signed temporal difference
        diff = f2 - f1

        x = torch.cat(
            [f1, f2, diff],
            dim=1
        )

        x = self.fusion(x)

        B, C, H, W = x.shape

        # Spatial tokens
        x = x.flatten(2).transpose(1, 2)

        # Transformer feature learning
        x = self.transformer(x)

        # Tokens back to image
        x = x.transpose(1, 2).reshape(
            B,
            C,
            H,
            W
        )

        x = self.refine(x)

        x = F.interpolate(
            x,
            size=t1.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        return self.classifier(x)
