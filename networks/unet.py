# networks/unet.py
#
# NEW FILE. train.py referenced `UNet(n_classes=num_classes)` in its
# `elif args.backbone == "unet":` branch, but no UNet class existed
# anywhere in the repository (only the TransUNet/R50-ViT files were
# included), so that branch was dead code that would raise a
# NameError the moment it was reached. This is a standard, self
# contained UNet (Ronneberger et al., 2015 - the same architecture
# used as the "Unet" baseline in Table 1 of the paper) so there is at
# least one second working backbone to plug CEM/IEM into out of the
# box, alongside TransUNet.
#
# The other 5 baselines compared against in the paper (UNeXt, UNet++,
# AttentionUNet, UCTransNet, VM-UNet, MedSAM) are not reproduced here -
# see FIXES.md for pointers to their official repos.

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # pad in case of odd input sizes
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        x = nn.functional.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard UNet with skip connections.

    num_classes: number of independent (sigmoid, multi-label) output
        channels - one per rib.
    in_channels: 3 to match the 3-channel (grayscale-repeated-to-RGB)
        preprocessing used by utils/dataloader.py.
    base_channels: width of the first encoder stage; doubles at every
        downsampling step, as usual.
    """

    def __init__(self, num_classes, in_channels=3, base_channels=64):
        super().__init__()
        c = base_channels
        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, c * 16)

        self.up1 = Up(c * 16, c * 8)
        self.up2 = Up(c * 8, c * 4)
        self.up3 = Up(c * 4, c * 2)
        self.up4 = Up(c * 2, c)

        self.out_conv = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.out_conv(x)
