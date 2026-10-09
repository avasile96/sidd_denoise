"""DRUNet (Zhang et al., TPAMI 2021, "Plug-and-Play Image Restoration with Deep Denoiser Prior").

Port of KAIR's UNetRes (MIT license). Parameter names match `drunet_color.pth`.
DRUNet was trained on synthetic AWGN and takes a 4th input channel holding a constant
noise-level map sigma/255. Real sRGB noise has no single sigma, so `DRUNetDenoiser`
turns that map into one learnable scalar (log-parameterised, initialised from the
best zero-shot sigma on the holdout split).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(c, c, 3, 1, 1, bias=False)
        )

    def forward(self, x):
        return x + self.res(x)


class UNetRes(nn.Module):
    def __init__(self, in_nc=4, out_nc=3, nc=(64, 128, 256, 512), nb=4):
        super().__init__()
        self.m_head = nn.Conv2d(in_nc, nc[0], 3, 1, 1, bias=False)
        self.m_down1 = nn.Sequential(*[ResBlock(nc[0]) for _ in range(nb)], nn.Conv2d(nc[0], nc[1], 2, 2, 0, bias=False))
        self.m_down2 = nn.Sequential(*[ResBlock(nc[1]) for _ in range(nb)], nn.Conv2d(nc[1], nc[2], 2, 2, 0, bias=False))
        self.m_down3 = nn.Sequential(*[ResBlock(nc[2]) for _ in range(nb)], nn.Conv2d(nc[2], nc[3], 2, 2, 0, bias=False))
        self.m_body = nn.Sequential(*[ResBlock(nc[3]) for _ in range(nb)])
        self.m_up3 = nn.Sequential(nn.ConvTranspose2d(nc[3], nc[2], 2, 2, 0, bias=False), *[ResBlock(nc[2]) for _ in range(nb)])
        self.m_up2 = nn.Sequential(nn.ConvTranspose2d(nc[2], nc[1], 2, 2, 0, bias=False), *[ResBlock(nc[1]) for _ in range(nb)])
        self.m_up1 = nn.Sequential(nn.ConvTranspose2d(nc[1], nc[0], 2, 2, 0, bias=False), *[ResBlock(nc[0]) for _ in range(nb)])
        self.m_tail = nn.Conv2d(nc[0], out_nc, 3, 1, 1, bias=False)

    def forward(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        return self.m_tail(x + x1)


class DRUNetDenoiser(nn.Module):
    """RGB-in / RGB-out wrapper around UNetRes with a learnable global noise level."""

    def __init__(self, sigma: float = 15.0, learn_sigma: bool = True):
        super().__init__()
        self.net = UNetRes()
        log_sigma = torch.tensor(math.log(sigma))
        if learn_sigma:
            self.log_sigma = nn.Parameter(log_sigma)
        else:
            self.register_buffer("log_sigma", log_sigma)

    @property
    def sigma(self) -> float:
        return float(self.log_sigma.exp())

    def forward(self, x):
        _, _, h, w = x.shape
        ph, pw = (-h) % 8, (-w) % 8
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        level = (self.log_sigma.exp() / 255.0).to(x.dtype)
        sigma_map = level.expand(x.shape[0], 1, x.shape[2], x.shape[3])
        return self.net(torch.cat([x, sigma_map], dim=1))[..., :h, :w]


def load_official(model: DRUNetDenoiser, path: str) -> DRUNetDenoiser:
    sd = torch.load(path, map_location="cpu", weights_only=True)
    model.net.load_state_dict(sd, strict=True)
    return model
