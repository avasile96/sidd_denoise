"""Exponential moving average of model weights (evaluated instead of the raw weights)."""
from __future__ import annotations

import copy

import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999, warmup: int = 100):
        self.decay = decay
        self.warmup = warmup
        self.num_updates = 0
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        # Ramp the decay up so early EMA weights are not dominated by the random init.
        d = min(self.decay, (1 + self.num_updates) / (self.warmup + self.num_updates))
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])

    def state_dict(self) -> dict:
        return {"module": self.module.state_dict(), "num_updates": self.num_updates}

    def load_state_dict(self, sd: dict) -> None:
        self.module.load_state_dict(sd["module"])
        self.num_updates = sd["num_updates"]
