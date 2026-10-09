"""Model registry: build a model (optionally pretrained, optionally partially frozen) from a config dict.

    model:
      name: scunet            # nafnet | unet | scunet | drunet | identity
      weights: weights/scunet_color_real_psnr.pth   # optional pretrained checkpoint
      finetune: lora          # full | lora | head_tail | none
      lora_r: 8
      kwargs: {}              # forwarded to the architecture constructor
"""
from __future__ import annotations

from pathlib import Path

import torch.nn as nn

from . import drunet, nafnet, scunet, unet
from .lora import apply_lora, count_params


class Identity(nn.Module):
    """'Model' that returns its input: the noisy-input baseline."""

    def forward(self, x):
        return x


# Parameter-name prefixes that make up each architecture's input head and output tail.
HEAD_TAIL = {
    "nafnet": ("intro", "ending"),
    "unet": ("inc", "out"),
    "scunet": ("m_head", "m_tail"),
    "drunet": ("net.m_head", "net.m_tail", "log_sigma"),
}


def build_model(cfg: dict) -> nn.Module:
    name = cfg["name"].lower()
    kwargs = dict(cfg.get("kwargs") or {})
    weights = cfg.get("weights")
    if name == "identity":
        return Identity()
    if name == "nafnet":
        model = nafnet.NAFNet(**kwargs)
        loader = nafnet.load_official
    elif name == "unet":
        model = unet.UNet(**kwargs)
        loader = None
    elif name == "scunet":
        model = scunet.SCUNet(**kwargs)
        loader = scunet.load_official
    elif name == "drunet":
        model = drunet.DRUNetDenoiser(**kwargs)
        loader = drunet.load_official
    else:
        raise ValueError(f"unknown model {name!r}")

    if weights:
        if loader is None:
            raise ValueError(f"{name} has no official pretrained weights")
        if not Path(weights).exists():
            raise FileNotFoundError(f"{weights} (run scripts/download_weights.py)")
        loader(model, weights)

    mode = cfg.get("finetune", "full")
    if mode == "lora":
        apply_lora(model, r=int(cfg.get("lora_r", 8)), alpha=cfg.get("lora_alpha"))
        # Keep the learnable noise level of DRUNet trainable alongside the adapters.
        for n, p in model.named_parameters():
            if n == "log_sigma":
                p.requires_grad_(True)
    elif mode == "head_tail":
        prefixes = HEAD_TAIL[name]
        for n, p in model.named_parameters():
            p.requires_grad_(n.startswith(prefixes))
    elif mode == "none":
        for p in model.parameters():
            p.requires_grad_(False)
    elif mode != "full":
        raise ValueError(f"unknown finetune mode {mode!r}")
    return model


__all__ = ["build_model", "count_params", "Identity"]
