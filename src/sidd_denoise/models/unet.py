"""Plain residual U-Net: the ablation baseline for NAFNet.

Same 4-level encoder/decoder topology and residual output as NAFNet, but built from
conventional conv-ReLU double-conv blocks (no LayerNorm, SimpleGate or channel attention).
With base width 32 its compute is close to NAFNet-w32's (~15 GMACs at 256x256), so the
comparison isolates the NAFBlock design rather than model size.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def double_conv(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_ch: int = 3, base: int = 32, depth: int = 4):
        super().__init__()
        chs = [base * 2**i for i in range(depth + 1)]
        self.inc = double_conv(in_ch, chs[0])
        self.downs = nn.ModuleList(nn.Conv2d(chs[i], chs[i + 1], 2, 2) for i in range(depth))
        self.enc = nn.ModuleList(double_conv(chs[i + 1], chs[i + 1]) for i in range(depth))
        self.ups = nn.ModuleList(nn.ConvTranspose2d(chs[i + 1], chs[i], 2, 2) for i in reversed(range(depth)))
        self.dec = nn.ModuleList(double_conv(2 * chs[i], chs[i]) for i in reversed(range(depth)))
        self.out = nn.Conv2d(chs[0], in_ch, 3, padding=1)
        self.pad_to = 2**depth

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        _, _, h, w = inp.shape
        ph, pw = (-h) % self.pad_to, (-w) % self.pad_to
        x0 = F.pad(inp, (0, pw, 0, ph), mode="reflect") if ph or pw else inp
        x = self.inc(x0)
        skips = [x]
        for down, enc in zip(self.downs, self.enc):
            x = enc(down(x))
            skips.append(x)
        skips.pop()
        for up, dec in zip(self.ups, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], dim=1))
        return (x0 + self.out(x))[:, :, :h, :w]
