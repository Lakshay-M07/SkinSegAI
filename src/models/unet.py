
import torch
import torch.nn as nn


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self._layers = nn.Sequential(

            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, padding=1, bias=False
            ),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, padding=1, bias=False
            ),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self._layers(x)


class EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self._conv = DoubleConv(
            in_channels, out_channels
        )

        self._pool = nn.MaxPool2d(2)

    def forward(self, x):

        skip = self._conv(x)
        pooled = self._pool(skip)

        return skip, pooled


class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self._upsample = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        self._conv = DoubleConv(
            out_channels * 2,
            out_channels
        )

    def forward(self, x, skip):

        x = self._upsample(x)

        x = torch.cat((x, skip), dim=1)

        return self._conv(x)


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1):

        super().__init__()

        self._encoder1 = EncoderBlock(in_channels, 16)
        self._encoder2 = EncoderBlock(16, 32)
        self._encoder3 = EncoderBlock(32, 64)
        self._encoder4 = EncoderBlock(64, 128)

        self._bottleneck = DoubleConv(128, 256)

        self._decoder4 = DecoderBlock(256, 128)
        self._decoder3 = DecoderBlock(128, 64)
        self._decoder2 = DecoderBlock(64, 32)
        self._decoder1 = DecoderBlock(32, 16)

        self._output = nn.Conv2d(
            16, out_channels, kernel_size=1
        )

    def forward(self, x):

        s1, x = self._encoder1(x)
        s2, x = self._encoder2(x)
        s3, x = self._encoder3(x)
        s4, x = self._encoder4(x)

        x = self._bottleneck(x)

        x = self._decoder4(x, s4)
        x = self._decoder3(x, s3)
        x = self._decoder2(x, s2)
        x = self._decoder1(x, s1)

        return self._output(x)
