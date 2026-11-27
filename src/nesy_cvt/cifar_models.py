"""CIFAR-specific reference architectures used in research comparisons."""

from __future__ import annotations


def wide_resnet28x10_cifar(num_classes: int = 100, dropout: float = 0.3):
    """Build a CIFAR Wide-ResNet-28-10 with the standard 3-stage layout.

    The implementation uses four residual blocks per stage, channel widths
    160/320/640, and no ImageNet stem or pretrained weights.
    """
    import torch
    from torch import nn
    from torch.nn import functional as F

    class WideBasicBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, stride: int):
            super().__init__()
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
            )
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.dropout = nn.Dropout(p=dropout)
            self.conv2 = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.shortcut = (
                nn.Identity()
                if stride == 1 and in_channels == out_channels
                else nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            )

        def forward(self, x):
            activated = F.relu(self.bn1(x), inplace=True)
            residual = self.shortcut(activated) if not isinstance(self.shortcut, nn.Identity) else x
            out = self.conv1(activated)
            out = self.conv2(self.dropout(F.relu(self.bn2(out), inplace=True)))
            return residual + out

    class WideResNetCIFAR(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
            channels = (160, 320, 640)
            stages = []
            current = 16
            for stage_index, width in enumerate(channels):
                blocks = []
                for block_index in range(4):
                    stride = 2 if stage_index > 0 and block_index == 0 else 1
                    blocks.append(WideBasicBlock(current, width, stride))
                    current = width
                stages.append(nn.Sequential(*blocks))
            self.stage1, self.stage2, self.stage3 = stages
            self.bn = nn.BatchNorm2d(current)
            self.fc = nn.Linear(current, num_classes)
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Linear):
                    nn.init.zeros_(module.bias)

        def forward(self, x):
            x = self.conv1(x)
            x = self.stage1(x)
            x = self.stage2(x)
            x = self.stage3(x)
            x = F.relu(self.bn(x), inplace=True)
            x = F.adaptive_avg_pool2d(x, output_size=1).flatten(1)
            return self.fc(x)

    return WideResNetCIFAR()
