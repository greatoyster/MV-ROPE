""" Full assembly of the parts to form the complete network """

from .unet_parts import *

try:
    import ipdb as pdb
except:
    import pdb


class UNet_encoder(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_encoder, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)

        self.up_out = self.up = nn.Upsample(
            scale_factor=4, mode="bilinear", align_corners=True
        )

    def forward(self, x):  # x: 4, 3, 480, 640
        x = self.inc(x)  # 4, 64, 480, 640
        x = self.down1(x)  # 4, 128, 240, 320
        x3 = self.down2(x)  # 4, 256, 120, 160
        x4 = self.down3(x3)  # 4, 512, 60, 80
        x5 = self.down4(x4)  # 4, 1024, 30, 40
        x = self.up1(x5, x4)  # 4, 512, 60, 80
        x = self.up2(x, x3)  # 4, 256, 120, 160
        x = self.up_out(x)  # 4, 256, 480, 640

        return x
