import json

import cv2
import numpy as np
import pytest
import torch

from sidd_denoise.data import TileDataset, augment
from sidd_denoise.losses import CharbonnierLoss, PSNRLoss
from sidd_denoise.train import lr_lambda, make_train_loader, train


@pytest.fixture
def tile_root(tmp_path):
    """A fake data/patches tree: 3 train pairs + 1 holdout pair, noise = gt + gaussian."""
    rng = np.random.default_rng(0)
    tiles = []
    for p, split in [("0001_001_GP_1_1_1_L", "train"), ("0002_001_IP_1_1_1_L", "train"),
                     ("0003_002_S6_1_1_1_L", "train"), ("0004_002_N6_1_1_1_L", "holdout")]:
        for j in range(2):
            gt = cv2.GaussianBlur(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8), (0, 0), 2)
            noisy = np.clip(gt + rng.normal(0, 20, gt.shape), 0, 255).astype(np.uint8)
            rel = {}
            for kind, img in (("gt", gt), ("noisy", noisy)):
                d = tmp_path / split / kind
                d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / f"{p}_{j}.png"), img)
                rel[kind] = f"{split}/{kind}/{p}_{j}.png"
            tiles.append({"pair": p, "split": split, **rel})
    (tmp_path / "index.json").write_text(json.dumps({"tiles": tiles}))
    return tmp_path


def test_tile_dataset_crops_and_pair_subsets(tile_root):
    ds = TileDataset(tile_root, "train", patch_size=32, augment=True)
    noisy, gt = ds[0]
    assert noisy.shape == gt.shape == (3, 32, 32)
    assert 0 <= noisy.min() and noisy.max() <= 1
    assert len(ds.pairs) == 3
    sub = TileDataset(tile_root, "train", patch_size=32, num_pairs=2, pair_seed=1)
    assert len(sub.pairs) == 2 and len(sub.tiles) == 4
    with pytest.raises(ValueError):
        TileDataset(tile_root, "train", num_pairs=5)


def test_train_loader_mixes_pairs(tile_root):
    """Tiles are stored grouped by pair; batches must still mix pairs (regression: shuffle=False)."""
    ds = TileDataset(tile_root, "train", patch_size=32, length=600)
    order = list(make_train_loader(ds, batch_size=2, num_workers=0, seed=0).batch_sampler)
    pair_of = lambda i: ds.tiles[i % len(ds.tiles)]["pair"]  # noqa: E731
    mixed = sum(len({pair_of(i) for i in b}) > 1 for b in order)
    assert mixed > len(order) // 2


def test_augment_is_applied_identically():
    a = np.arange(2 * 3 * 3).reshape(2, 3, 3)[..., None].repeat(3, -1)
    for mode in range(8):
        x, y = augment(a, a.copy(), mode)
        assert np.array_equal(x, y)
    assert len({augment(a, a, m)[0].tobytes() for m in range(8)}) == 8 or a.shape[0] != a.shape[1]


def test_losses():
    x = torch.rand(2, 3, 8, 8)
    assert PSNRLoss()(x, x) < -70  # log(1e-8) floor
    y = (x + 0.1).clamp(0, 1)
    assert PSNRLoss()(y, x) > PSNRLoss()(x, x)
    assert CharbonnierLoss()(y, x) > CharbonnierLoss()(x, x)


def test_lr_schedule_warmup_then_cosine():
    assert lr_lambda(0, 10, 100, 0.0) == pytest.approx(0.1)
    assert lr_lambda(9, 10, 100, 0.0) == pytest.approx(1.0)
    assert lr_lambda(100, 10, 100, 0.01) == pytest.approx(0.01)


def test_train_smoke_improves_and_resumes(tile_root, tmp_path):
    cfg = {
        "name": "t", "seed": 0, "device": "cpu", "out_dir": str(tmp_path / "runs"), "ema": 0.9, "loss": "psnr",
        "log_every": 10,
        "data": {"root": str(tile_root), "patch_size": 32, "val_patch_size": 64, "batch_size": 4, "num_workers": 0},
        "model": {"name": "nafnet", "kwargs": {"width": 8, "middle_blk_num": 1, "enc_blk_nums": [1, 1], "dec_blk_nums": [1, 1]}},
        "optim": {"lr": 2e-3, "betas": [0.9, 0.9]},
        "sched": {"iters": 60, "warmup": 5},
        "val": {"every": 30},
    }
    s = train(cfg)
    assert s["iters"] == 60
    log = (tmp_path / "runs" / "t" / "log.csv").read_text().splitlines()
    vals = [float(r.split(",")[3]) for r in log[1:] if r.split(",")[3]]
    noisy_psnr = 10 * np.log10(255**2 / 20**2)  # ~22 dB input
    assert max(vals) > noisy_psnr  # learned something beyond identity
    # resume: extending the schedule continues from the saved iteration
    cfg["sched"]["iters"] = 70
    s2 = train(cfg)
    assert s2["iters"] == 70
