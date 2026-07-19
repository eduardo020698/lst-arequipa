"""U-Net para downscaling de LST (GOES 2km + SZA + DEM -> LST 1km).

Arquitectura fiel al espiritu de Kurchaba & Meyer (2026), con el DEM como
canal extra (mejora para terreno andino). Entrada (B,3,H,W) -> salida (B,1,H,W).
"""
import torch
import torch.nn as nn


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, cin=3, base=32):
        super().__init__()
        b = base
        self.e1 = block(cin, b)
        self.e2 = block(b, b * 2)
        self.e3 = block(b * 2, b * 4)
        self.e4 = block(b * 4, b * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.d3 = block(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.d2 = block(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.d1 = block(b * 2, b)
        self.head = nn.Conv2d(b, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        d3 = self.d3(torch.cat([self.up3(e4), e3], 1))
        d2 = self.d2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], 1))
        # residual: la red aprende la CORRECCION sobre el campo GOES (canal 0)
        return self.head(d1) + x[:, :1]


def masked_mse(pred, target, mask):
    """MSE solo sobre pixeles validos (en unidades normalizadas)."""
    diff = (pred.squeeze(1) - target) ** 2
    return (diff * mask).sum() / mask.sum().clamp(min=1)
