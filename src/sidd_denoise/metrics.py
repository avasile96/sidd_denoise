"""Evaluation metrics following the SIDD / NAFNet / Restormer convention.

* Outputs are clamped to [0, 1] and quantized to uint8 *before* scoring.
* PSNR and SSIM are computed in RGB (no Y channel, no border crop) per 256x256
  block and averaged over all blocks.
* Two SSIM variants are reported:
  - ``ssim``: the SIDD-literature convention. MATLAB's ``ssim`` treats an HxWx3
    image as a 3-D volume, and NAFNet/BasicSR's ``calculate_ssim`` mirrors that with
    an 11x11x11 Gaussian (sigma 1.5) and replicate padding. This is the number
    comparable to published SIDD tables (NAFNet-w32 reports 0.960).
  - ``ssim2d``: per-channel 2-D SSIM (skimage with Wang et al. 2004 settings:
    11x11 Gaussian with sigma 1.5 and population covariance), averaged over RGB. This
    is stricter; it gives about 0.92 for the same NAFNet model.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from scipy.ndimage import correlate1d
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def quantize(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """float image(s) in [0,1] (CHW / NCHW tensor or HWC array) -> uint8 HWC / NHWC."""
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu()
        x = x.permute(0, 2, 3, 1) if x.ndim == 4 else x.permute(1, 2, 0)
        x = x.numpy()
    return np.clip(np.round(np.clip(x, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    """PSNR (dB) between two uint8 HWC images, data_range=255."""
    if np.array_equal(pred, gt):
        return float("inf")
    return float(peak_signal_noise_ratio(gt, pred, data_range=255))


_G1D = cv2.getGaussianKernel(11, 1.5)[:, 0]


def _blur3d(t: np.ndarray) -> np.ndarray:
    """11x11x11 Gaussian (sigma 1.5) over the H, W and channel axes of an NHWC float64 array,
    with replicate borders. The 3-D kernel is separable, so three 1-D passes give exactly
    BasicSR's replicate-padded conv3d at a fraction of the cost."""
    for axis in (1, 2, 3):
        t = correlate1d(t, _G1D, axis=axis, mode="nearest")
    return t


def ssim_batch(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """3-D Gaussian SSIM (SIDD / MATLAB / BasicSR convention) for uint8 NHWC batches -> (N,)."""
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    x, y = pred.astype(np.float64), gt.astype(np.float64)
    mx, my = _blur3d(x), _blur3d(y)
    sxx, syy, sxy = _blur3d(x * x) - mx**2, _blur3d(y * y) - my**2, _blur3d(x * y) - mx * my
    smap = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx**2 + my**2 + c1) * (sxx + syy + c2))
    return smap.mean(axis=(1, 2, 3))


def ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    """3-D SSIM of one uint8 HWC image pair (the number comparable to SIDD papers)."""
    return float(ssim_batch(pred[None], gt[None])[0])


def ssim2d(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(
        structural_similarity(
            gt,
            pred,
            data_range=255,
            channel_axis=-1,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
        )
    )


def _metric_device(device):
    if device is not None:
        return torch.device(device)
    from .data import get_device

    return get_device()


class LPIPSMetric:
    """LPIPS (AlexNet). Inputs are uint8 NHWC. MPS gives the same values as CPU (to ~1e-6), ~8x faster."""

    def __init__(self, device: str | None = None):
        import lpips

        self.device = _metric_device(device)
        self.net = lpips.LPIPS(net="alex", verbose=False).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        def t(a):
            return torch.from_numpy(a).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(self.device)

        return self.net(t(pred), t(gt)).flatten().cpu().numpy()


class DISTSMetric:
    """DISTS via pyiqa (optional dependency). Inputs are uint8 NHWC. ~60x faster on MPS than CPU."""

    def __init__(self, device: str | None = None):
        import pyiqa

        self.device = _metric_device(device)
        self.net = pyiqa.create_metric("dists", device=self.device)

    @torch.no_grad()
    def __call__(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        def t(a):
            return torch.from_numpy(a).permute(0, 3, 1, 2).float().div(255).to(self.device)

        return self.net(t(pred), t(gt)).flatten().cpu().numpy()


def bootstrap_ci(values: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI of the mean."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> dict:
    """Paired bootstrap over images for mean(a - b). Returns the diff, 95% CI and a
    one-sided p-value for H0: mean(a) <= mean(b)."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return {
        "mean_diff": float(d.mean()),
        "ci95": (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))),
        "p_value": float((means <= 0).mean()),
    }


def per_image_means(values: np.ndarray, image_ids: np.ndarray) -> np.ndarray:
    ids = np.unique(image_ids)
    return np.array([values[image_ids == i].mean() for i in ids])
