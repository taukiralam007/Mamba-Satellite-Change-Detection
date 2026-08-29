import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSMBlock(nn.Module):
    """
    Lightweight Mamba-inspired selective state-space block.
    Pure PyTorch, no external CUDA extension.
    """

    def __init__(self, dim, expansion=2, kernel_size=3):
        super().__init__()

        hidden = dim * expansion

        self.norm = nn.LayerNorm(dim)

        self.in_proj = nn.Linear(
            dim,
            hidden * 2
        )

        self.dwconv = nn.Conv1d(
            hidden,
            hidden,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden
        )

        self.delta_proj = nn.Linear(
            hidden,
            hidden
        )

        self.state_proj = nn.Linear(
            hidden,
            hidden
        )

        self.out_proj = nn.Linear(
            hidden,
            dim
        )

    def forward(self, x):
        residual = x

        x = self.norm(x)

        x, gate = self.in_proj(x).chunk(
            2,
            dim=-1
        )

        x = x.transpose(1, 2)
        x = self.dwconv(x)
        x = x.transpose(1, 2)

        x = F.silu(x)

        delta = torch.sigmoid(
            self.delta_proj(x)
        )

        candidate = torch.tanh(
            self.state_proj(x)
        )

        state = torch.zeros_like(
            candidate[:, 0]
        )

        outputs = []

        for t in range(candidate.shape[1]):
            state = (
                (1.0 - delta[:, t]) * state
                +
                delta[:, t] * candidate[:, t]
            )

            outputs.append(state)

        y = torch.stack(
            outputs,
            dim=1
        )

        y = y * F.silu(gate)

        y = self.out_proj(y)

        return residual + y


class SSMChangeDetectorV1(nn.Module):
    """
    Selective SSM-CD V1.
    32x32 input -> 8x8 latent representation.
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
                stride=2,
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
