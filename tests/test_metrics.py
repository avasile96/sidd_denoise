import math

import numpy as np
import torch

from sidd_denoise.metrics import (
    bootstrap_ci, paired_bootstrap, per_image_means, psnr, quantize, ssim, ssim2d, ssim_batch,
)


def test_psnr_identical_and_known_value():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    assert psnr(a, a) == float("inf")
    b = a.astype(np.int16)
    b[..., 0] = np.clip(b[..., 0] + np.where(b[..., 0] < 128, 1, -1), 0, 255)  # every R pixel off by exactly 1
    mse = 1.0 / 3.0
    assert math.isclose(psnr(b.astype(np.uint8), a), 10 * math.log10(255**2 / mse), rel_tol=1e-9)


def test_ssim_identical_is_one_and_noise_lowers_it():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    assert math.isclose(ssim(a, a), 1.0, abs_tol=1e-12)
    assert math.isclose(ssim2d(a, a), 1.0, abs_tol=1e-12)
    noisy = np.clip(a + rng.normal(0, 25, a.shape), 0, 255).astype(np.uint8)
    assert ssim(noisy, a) < 0.99 and ssim2d(noisy, a) < 0.99


def test_ssim3d_matches_basicsr_conv3d_reference():
    """Our separable implementation == BasicSR's replicate-padded 11x11x11 conv3d SSIM."""
    import cv2

    rng = np.random.default_rng(2)
    a = rng.integers(0, 256, (2, 40, 48, 3), dtype=np.uint8)
    b = np.clip(a.astype(int) + rng.integers(-30, 30, a.shape), 0, 255).astype(np.uint8)
    g = cv2.getGaussianKernel(11, 1.5)
    k = torch.tensor(np.stack([np.outer(g, g.T) * kk for kk in g[:, 0]], 0))
    conv = torch.nn.Conv3d(1, 1, 11, padding=5, bias=False, padding_mode="replicate").double()
    conv.weight.data[0, 0] = k

    def ref(x, y):
        x, y = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
        f = lambda t: conv(t[None, None])[0, 0]  # noqa: E731
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        mx, my = f(x), f(y)
        sxx, syy, sxy = f(x * x) - mx**2, f(y * y) - my**2, f(x * y) - mx * my
        return float((((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx**2 + my**2 + c1) * (sxx + syy + c2))).mean())

    with torch.no_grad():
        expected = [ref(a[i], b[i]) for i in range(2)]
    np.testing.assert_allclose(ssim_batch(a, b), expected, rtol=1e-10)


def test_quantize_rounds_and_clamps():
    t = torch.tensor([[[-0.1, 0.5 / 255, 1.49 / 255, 1.2]]]).expand(3, 1, 4)  # CHW
    q = quantize(t)
    assert q.dtype == np.uint8 and q.shape == (1, 4, 3)
    assert q[0, :, 0].tolist() == [0, 0, 1, 255]  # numpy rounds half to even: 0.5 -> 0


def test_bootstrap_helpers():
    v = np.arange(40, dtype=float)
    lo, hi = bootstrap_ci(v)
    assert lo < v.mean() < hi
    r = paired_bootstrap(v + 1.0, v)
    assert math.isclose(r["mean_diff"], 1.0) and r["p_value"] == 0.0
    ids = np.repeat(np.arange(4), 3)
    np.testing.assert_allclose(per_image_means(np.arange(12, dtype=float), ids), [1, 4, 7, 10])
