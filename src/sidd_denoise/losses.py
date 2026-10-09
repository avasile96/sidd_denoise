"""Pixel losses used by SIDD state-of-the-art models."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class PSNRLoss(nn.Module):
    """Negative PSNR, as in NAFNet: 10/ln(10) * log(MSE + eps) per image, averaged over the batch.

    Computed in fp32 even under autocast, because log(MSE) at ~40 dB is ~1e-4 and
    underflows easily in fp16.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.scale = 10.0 / math.log(10.0)
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred, target = pred.float(), target.float()
        mse = (pred - target).pow(2).mean(dim=(1, 2, 3))
        return self.scale * torch.log(mse + self.eps).mean()


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((pred.float() - target.float()).pow(2) + self.eps2).mean()


class L1Loss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (pred.float() - target.float()).abs().mean()


def build_loss(name: str) -> nn.Module:
    return {"psnr": PSNRLoss, "charbonnier": CharbonnierLoss, "l1": L1Loss}[name.lower()]()
