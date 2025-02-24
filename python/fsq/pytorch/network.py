import torch
import torch.nn.functional as F
from torch import nn

from fsq import FSQ


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=3,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.deconv3 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=3,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = self.deconv3(x)
        return x


class VQVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.quantizer = FSQ(levels=[3, 3, 3])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape  # (64, 3, 96, 96)
        encoded = self.encoder(x)  # (64, 3, 12, 12)
        _, _, zh, zw = encoded.shape
        encoded = encoded.permute(0, 2, 3, 1).contiguous()
        encoded = encoded.view(b, zh * zw, c)
        quantized = self.quantizer.quantize(encoded)
        quantized = quantized.view(b, zh, zw, c)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        decoded = self.decoder(quantized)
        return decoded
