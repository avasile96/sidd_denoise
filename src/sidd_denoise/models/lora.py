"""Minimal LoRA for Linear and Conv2d layers (Hu et al., 2021; conv variant as in CoLoRA / LoRA-IR).

    W x  ->  W x + (alpha / r) * B(A(x)),   A: random init, B: zero init

so a freshly wrapped model computes exactly the same function as the base model.
`merge_lora` folds the update back into the base weights for zero-overhead inference.
Written in-house to avoid pulling in `peft` + `transformers` for ~80 lines of logic.
"""
from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 8.0):
        super().__init__()
        self.base = base
        self.r, self.scale = r, alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale

    def delta_weight(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scale


class LoRAConv2d(nn.Module):
    """Low-rank update for a (non-grouped) Conv2d: a k x k conv to r channels, then a 1x1 conv back."""

    def __init__(self, base: nn.Conv2d, r: int = 8, alpha: float = 8.0):
        super().__init__()
        assert base.groups == 1, "LoRA on grouped/depthwise convs is not supported"
        self.base = base
        self.r, self.scale = r, alpha / r
        kh, kw = base.kernel_size
        self.lora_A = nn.Parameter(torch.empty(r, base.in_channels, kh, kw))
        self.lora_B = nn.Parameter(torch.zeros(base.out_channels, r, 1, 1))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        b = self.base
        low = F.conv2d(x, self.lora_A, None, b.stride, b.padding, b.dilation)
        return b(x) + F.conv2d(low, self.lora_B) * self.scale

    def delta_weight(self) -> torch.Tensor:
        # (out, r, 1, 1) x (r, in, kh, kw) -> (out, in, kh, kw)
        return torch.einsum("or,rikl->oikl", self.lora_B[:, :, 0, 0], self.lora_A) * self.scale


def apply_lora(
    model: nn.Module,
    r: int = 8,
    alpha: float | None = None,
    target: Callable[[str, nn.Module], bool] | None = None,
) -> nn.Module:
    """Freeze `model` and wrap every targeted Linear / non-grouped Conv2d with LoRA, in place."""
    alpha = float(alpha if alpha is not None else r)
    for p in model.parameters():
        p.requires_grad_(False)
    target = target or (lambda name, m: True)
    replacements = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and target(name, m):
            replacements.append((name, LoRALinear(m, r, alpha)))
        elif isinstance(m, nn.Conv2d) and m.groups == 1 and target(name, m):
            replacements.append((name, LoRAConv2d(m, r, alpha)))
    for name, new in replacements:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, new)
    return model


@torch.no_grad()
def merge_lora(model: nn.Module) -> nn.Module:
    """Fold every LoRA update into its base layer and unwrap it, in place."""
    for name, m in list(model.named_modules()):
        if isinstance(m, (LoRALinear, LoRAConv2d)):
            m.base.weight.add_(m.delta_weight().to(m.base.weight.dtype))
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child, m.base)
    return model


def count_params(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
