"""Classical baselines: tune sigma on the SIDD-Small holdout split, then score on the test blocks.

    python scripts/run_baselines.py                   # identity, Gaussian, BM3D (estimated + tuned sigma)
    python scripts/run_baselines.py --skip-bm3d       # fast: identity + Gaussian only

The holdout tiles are 512x512; for tuning we score the central 256x256 crop of an
evenly spaced subset of them (--tune-tiles), which keeps the BM3D sweep affordable.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sidd_denoise.baselines import bm3d_denoise, gaussian_denoise, run_batch  # noqa: E402
from sidd_denoise.data import TileDataset, read_rgb  # noqa: E402
from sidd_denoise.evaluate import evaluate_fn  # noqa: E402
from sidd_denoise.metrics import psnr  # noqa: E402


def holdout_crops(root: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    ds = TileDataset(root, "holdout", 256, random_crop=False)
    tiles = ds.tiles[:: max(1, math.ceil(len(ds.tiles) / n))]
    noisy, gt = [], []
    for t in tiles:
        a, b = read_rgb(Path(root) / t["noisy"]), read_rgb(Path(root) / t["gt"])
        y, x = (a.shape[0] - 256) // 2, (a.shape[1] - 256) // 2
        noisy.append(a[y : y + 256, x : x + 256])
        gt.append(b[y : y + 256, x : x + 256])
    return np.stack(noisy), np.stack(gt)


def sweep(fn, grid, noisy, gt, workers) -> tuple[float, dict]:
    scores = {}
    for s in grid:
        out = run_batch(partial(fn, sigma=s), noisy, workers)
        scores[s] = float(np.mean([psnr(p, g) for p, g in zip(out, gt)]))
        print(f"    sigma={s:<6} holdout PSNR {scores[s]:.3f}", flush=True)
    best = max(scores, key=scores.get)
    return best, scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patches", default="data/patches")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="results")
    ap.add_argument("--tune-tiles", type=int, default=96)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--skip-bm3d", action="store_true")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    noisy, gt = holdout_crops(args.patches, args.tune_tiles)
    print(f"tuning on {len(noisy)} holdout crops")
    tuning = {}

    r = evaluate_fn("noisy_input", lambda x: x, args.raw, args.out, args.limit)
    print(f"noisy_input: PSNR {r['psnr']:.3f}  SSIM {r['ssim']:.4f}")

    print("Gaussian filter sweep:")
    g_best, g_scores = sweep(gaussian_denoise, [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0], noisy, gt, 1)
    tuning["gaussian"] = {"best_sigma": g_best, "holdout_psnr": g_scores}
    r = evaluate_fn(f"gaussian", partial(gaussian_denoise, sigma=g_best), args.raw, args.out, args.limit,
                    extra={"sigma": g_best})
    print(f"gaussian (sigma={g_best}): PSNR {r['psnr']:.3f}  SSIM {r['ssim']:.4f}")

    if not args.skip_bm3d:
        r = evaluate_fn("bm3d_est", partial(bm3d_denoise, sigma=None), args.raw, args.out, args.limit, args.workers,
                        extra={"sigma": "estimated per image (wavelet MAD)"})
        print(f"bm3d_est: PSNR {r['psnr']:.3f}  SSIM {r['ssim']:.4f}")
        print("BM3D sweep:")
        b_best, b_scores = sweep(bm3d_denoise, [10, 20, 30, 40, 50, 65, 80, 100], noisy, gt, args.workers)
        tuning["bm3d"] = {"best_sigma": b_best, "holdout_psnr": b_scores}
        r = evaluate_fn("bm3d_tuned", partial(bm3d_denoise, sigma=b_best), args.raw, args.out, args.limit, args.workers,
                        extra={"sigma": b_best})
        print(f"bm3d_tuned (sigma={b_best}): PSNR {r['psnr']:.3f}  SSIM {r['ssim']:.4f}")

    Path(args.out).mkdir(exist_ok=True)
    (Path(args.out) / "baseline_tuning.json").write_text(json.dumps(tuning, indent=2))


if __name__ == "__main__":
    main()
