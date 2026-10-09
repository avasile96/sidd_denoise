"""Evaluate a model or classical baseline on the 1280 official SIDD sRGB validation blocks."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch

from .data import SIDDBlocks, get_device
from .metrics import (
    DISTSMetric, LPIPSMetric, bootstrap_ci, per_image_means, psnr, quantize, ssim2d, ssim_batch,
)
from .models import build_model, count_params

# Fixed blocks used for the visual comparison grids (spread over images and cameras).
VISUAL_BLOCKS = [5, 97, 262, 391, 523, 650, 811, 1004, 1150, 1262]


def tta_x8(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Geometric self-ensemble: average the 8 dihedral transforms."""
    outs = []
    for flip in (False, True):
        xf = x.flip(-1) if flip else x
        for k in range(4):
            y = model(torch.rot90(xf, k, dims=(-2, -1)))
            y = torch.rot90(y, -k, dims=(-2, -1))
            outs.append(y.flip(-1) if flip else y)
    return torch.stack(outs).mean(0)


def load_trained(ckpt: str | Path) -> tuple[torch.nn.Module, dict]:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    mcfg = dict(cfg["model"])
    mcfg.pop("weights", None)  # the checkpoint already holds every weight
    model = build_model(mcfg)
    model.load_state_dict(ck["model"])
    return model, cfg


def flops_per_block(model: torch.nn.Module) -> float:
    from torch.utils.flop_counter import FlopCounterMode

    model = model.cpu().eval()
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        model(torch.zeros(1, 3, 256, 256))
    return fc.get_total_flops() / 2  # MACs


@torch.no_grad()
def predict_model(model, ds: SIDDBlocks, device, batch_size=16, tta=False) -> tuple[np.ndarray, float]:
    model = model.to(device).eval()
    preds, t_total = [], 0.0
    for b in range(0, len(ds), batch_size):
        x = torch.stack([ds[i][0] for i in range(b, min(b + batch_size, len(ds)))]).to(device)
        t = time.time()
        y = tta_x8(model, x) if tta else model(x)
        y = y.float().cpu()  # .cpu() synchronises the device, so the timing is honest
        t_total += time.time() - t
        preds.append(quantize(y))
    return np.concatenate(preds), 1000 * t_total / len(ds)


def score(
    name: str,
    preds: np.ndarray,
    ds: SIDDBlocks,
    out_dir: str | Path = "results",
    extra: dict | None = None,
    lpips: bool = True,
    dists: bool = False,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gt = ds.gt
    per_block = {
        "psnr": np.array([psnr(p, g) for p, g in zip(preds, gt)]),
        "ssim": np.concatenate([ssim_batch(preds[i : i + 64], gt[i : i + 64]) for i in range(0, len(gt), 64)]),
        "ssim2d": np.array([ssim2d(p, g) for p, g in zip(preds, gt)]),
    }
    if lpips:
        m = LPIPSMetric()
        per_block["lpips"] = np.concatenate([m(preds[i : i + 32], gt[i : i + 32]) for i in range(0, len(gt), 32)])
    if dists:
        try:
            m = DISTSMetric()
            per_block["dists"] = np.concatenate([m(preds[i : i + 32], gt[i : i + 32]) for i in range(0, len(gt), 32)])
        except Exception as e:  # pyiqa is optional
            print(f"  (skipping DISTS: {e})")

    res = {"name": name, "n_blocks": int(len(gt))}
    for k, v in per_block.items():
        img = per_image_means(v, ds.image_ids)
        res[k] = float(v.mean())
        res[f"{k}_std"] = float(v.std())
        res[f"{k}_ci95_images"] = bootstrap_ci(img)
    res.update(extra or {})
    (out_dir / f"{name}.json").write_text(json.dumps(res, indent=2))
    np.savez_compressed(out_dir / f"{name}_per_block.npz", image_ids=ds.image_ids, **per_block)

    vis = out_dir / "visuals" / name
    vis.mkdir(parents=True, exist_ok=True)
    for i in VISUAL_BLOCKS:
        if i < len(preds):
            cv2.imwrite(str(vis / f"block_{i:04d}.png"), cv2.cvtColor(preds[i], cv2.COLOR_RGB2BGR))
    return res


def evaluate_model(
    name: str,
    model: torch.nn.Module,
    raw_dir="data/raw",
    out_dir="results",
    device=None,
    tta=False,
    limit=None,
    extra=None,
    dists=False,
) -> dict:
    device = get_device(device)
    ds = SIDDBlocks(raw_dir, limit=limit)
    info = {"params": count_params(model), "gmacs_256": flops_per_block(model) / 1e9, "tta": tta}
    preds, ms = predict_model(model, ds, device, tta=tta)
    info.update({"ms_per_block": ms, "device": str(device)})
    info.update(extra or {})
    return score(name, preds, ds, out_dir, info, dists=dists)


def evaluate_fn(name: str, fn: Callable, raw_dir="data/raw", out_dir="results", limit=None, workers=1, extra=None, dists=False):
    from .baselines import run_batch

    ds = SIDDBlocks(raw_dir, limit=limit)
    t = time.time()
    preds = run_batch(fn, ds.noisy, workers=workers)
    info = {"ms_per_block": 1000 * (time.time() - t) / len(ds), "device": f"cpu x{workers}", "params": 0}
    info.update(extra or {})
    return score(name, preds, ds, out_dir, info, dists=dists)
