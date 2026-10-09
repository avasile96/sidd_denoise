"""Classical baselines: Gaussian filter and CBM3D.

Each denoiser maps a uint8 HWC image to a uint8 HWC image, with a single
hyper-parameter sigma. Sigma is tuned on the *holdout split of SIDD-Small*, never on
the test blocks.

BM3D is reported two ways:
  * ``bm3d_est``: sigma estimated per image from the noisy input (skimage's
    wavelet-MAD estimator), the protocol behind the weak ~25.7 dB number in the
    original SIDD paper;
  * ``bm3d_tuned``: one global sigma chosen on holdout, a much stronger and fairer
    classical baseline for spatially correlated sRGB noise.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from functools import partial

import cv2
import numpy as np


def gaussian_denoise(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    return cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)


def bm3d_denoise(img: np.ndarray, sigma: float | None) -> np.ndarray:
    """CBM3D (bm3d package, YCbCr/opponent colour transform). sigma is on the 0-255 scale;
    None means estimate it from the image."""
    import bm3d

    x = img.astype(np.float32) / 255.0
    if sigma is None:
        from skimage.restoration import estimate_sigma

        s = float(np.mean(estimate_sigma(x, channel_axis=-1)))
    else:
        s = sigma / 255.0
    y = bm3d.bm3d_rgb(x, sigma_psd=max(s, 1e-4))
    return np.clip(np.round(y * 255.0), 0, 255).astype(np.uint8)


def _apply(fn, img):
    return fn(img)


def run_batch(fn, images: np.ndarray, workers: int = 1) -> np.ndarray:
    """Apply a per-image denoiser to an (N, H, W, 3) uint8 array, optionally across processes."""
    if workers <= 1:
        return np.stack([fn(im) for im in images])
    with ProcessPoolExecutor(workers) as ex:
        return np.stack(list(ex.map(partial(_apply, fn), images, chunksize=4)))


BASELINES = {
    "gaussian": gaussian_denoise,
    "bm3d": bm3d_denoise,
}
