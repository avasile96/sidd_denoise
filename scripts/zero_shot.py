"""Evaluate pretrained models *without* SIDD training (zero-shot), plus the reference ceiling.

  * SCUNet real-PSNR / real-GAN: trained only on SCUNet's synthetic realistic-noise pipeline.
  * DRUNet: trained on AWGN; its noise-level input sigma is tuned on the SIDD-Small holdout.
  * NAFNet-SIDD w32 / w64: official checkpoints trained on SIDD-Medium, which *contains*
    SIDD-Small (our training set). Not a fair competitor: an upper bound only.

    python scripts/zero_shot.py
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_baselines import holdout_crops  # noqa: E402
from sidd_denoise.data import get_device, to_tensor  # noqa: E402
from sidd_denoise.evaluate import evaluate_model  # noqa: E402
from sidd_denoise.metrics import psnr, quantize  # noqa: E402
from sidd_denoise.models import build_model  # noqa: E402

MEDIUM_NOTE = "Official checkpoint trained on SIDD-Medium (superset of our SIDD-Small train set): reference ceiling, not comparable."


@torch.no_grad()
def tune_drunet_sigma(noisy, gt, device, grid=(10, 20, 30, 40, 50, 65, 80, 100)) -> tuple[float, dict]:
    x = torch.stack([to_tensor(a) for a in noisy])
    scores = {}
    for s in grid:
        m = build_model({"name": "drunet", "weights": "weights/drunet_color.pth", "kwargs": {"sigma": s}}).to(device).eval()
        out = torch.cat([m(x[i : i + 8].to(device)).cpu() for i in range(0, len(x), 8)])
        scores[s] = float(np.mean([psnr(p, g) for p, g in zip(quantize(out), gt)]))
        print(f"  DRUNet sigma={s:<3} holdout PSNR {scores[s]:.3f}", flush=True)
        del m
    return max(scores, key=scores.get), scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patches", default="data/patches")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="results")
    ap.add_argument("--device")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    device = get_device(args.device)

    noisy, gt = holdout_crops(args.patches, 96)
    best_sigma, scores = tune_drunet_sigma(noisy, gt, device)
    Path(args.out).mkdir(exist_ok=True)
    (Path(args.out) / "drunet_sigma_tuning.json").write_text(json.dumps({"best_sigma": best_sigma, "holdout_psnr": scores}, indent=2))

    runs = [
        ("scunet_real_psnr_zeroshot", {"name": "scunet", "weights": "weights/scunet_color_real_psnr.pth"}, {}),
        ("scunet_real_gan_zeroshot", {"name": "scunet", "weights": "weights/scunet_color_real_gan.pth"}, {}),
        (f"drunet_zeroshot", {"name": "drunet", "weights": "weights/drunet_color.pth", "kwargs": {"sigma": best_sigma}},
         {"sigma": best_sigma}),
        ("ref_nafnet_sidd_w32", {"name": "nafnet", "weights": "weights/NAFNet-SIDD-width32.pth"}, {"note": MEDIUM_NOTE}),
        ("ref_nafnet_sidd_w64", {"name": "nafnet", "weights": "weights/NAFNet-SIDD-width64.pth",
                                 "kwargs": {"width": 64}}, {"note": MEDIUM_NOTE}),
    ]
    for name, cfg, extra in runs:
        model = build_model(cfg)
        res = evaluate_model(name, model, args.raw, args.out, device, limit=args.limit, extra=extra, dists=True)
        print(f"{name}: PSNR {res['psnr']:.3f}  SSIM {res['ssim']:.4f}  LPIPS {res.get('lpips', float('nan')):.4f}", flush=True)
        del model
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
