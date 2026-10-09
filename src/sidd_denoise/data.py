"""Datasets: pre-cropped SIDD-Small training tiles and the official validation blocks."""
from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def get_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_mat_blocks(path: str | Path) -> np.ndarray:
    """Load a SIDD *Blocks*Srgb.mat file as a (40, 32, 256, 256, 3) uint8 array."""
    path = Path(path)
    try:
        from scipy.io import loadmat

        mat = loadmat(path)
        key = next(k for k in mat if not k.startswith("__"))
        return np.ascontiguousarray(mat[key])
    except NotImplementedError:  # MATLAB v7.3 (HDF5) fallback
        import h5py

        with h5py.File(path, "r") as f:
            key = next(iter(f.keys()))
            arr = np.array(f[key])
        return np.ascontiguousarray(arr.transpose(tuple(reversed(range(arr.ndim)))))


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """HWC uint8 -> CHW float32 in [0, 1]."""
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div_(255.0)


def read_rgb(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def augment(noisy: np.ndarray, gt: np.ndarray, mode: int) -> tuple[np.ndarray, np.ndarray]:
    """One of the 8 dihedral transforms (mode in 0..7), applied identically to both images."""
    if mode >= 4:
        noisy, gt = noisy[:, ::-1], gt[:, ::-1]
    k = mode % 4
    return np.rot90(noisy, k), np.rot90(gt, k)


class TileDataset(Dataset):
    """Random crops from the pre-extracted tiles written by scripts/prepare_patches.py.

    ``index.json`` lists every tile with the SIDD pair it came from, so a training
    subset can be drawn at the *pair* level (``num_pairs``) for the data-efficiency study.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        patch_size: int = 256,
        augment: bool = False,
        num_pairs: int | None = None,
        pair_seed: int = 0,
        length: int | None = None,
        random_crop: bool = True,
    ):
        self.root = Path(root)
        index = json.loads((self.root / "index.json").read_text())
        tiles = [t for t in index["tiles"] if t["split"] == split]
        if num_pairs is not None:
            pairs = sorted({t["pair"] for t in tiles})
            if num_pairs > len(pairs):
                raise ValueError(f"asked for {num_pairs} pairs, only {len(pairs)} in {split}")
            keep = set(random.Random(pair_seed).sample(pairs, num_pairs))
            tiles = [t for t in tiles if t["pair"] in keep]
        if not tiles:
            raise RuntimeError(f"no tiles for split={split!r} in {self.root}")
        self.tiles = tiles
        self.patch_size = patch_size
        self.augment = augment
        self.random_crop = random_crop
        # With random crops an "epoch" is arbitrary; `length` lets the sampler draw more.
        self.length = length or len(tiles)

    @property
    def pairs(self) -> list[str]:
        return sorted({t["pair"] for t in self.tiles})

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int):
        t = self.tiles[i % len(self.tiles)]
        noisy = read_rgb(self.root / t["noisy"])
        gt = read_rgb(self.root / t["gt"])
        p = self.patch_size
        h, w = gt.shape[:2]
        if self.random_crop:
            y, x = random.randint(0, h - p), random.randint(0, w - p)
        else:
            y, x = (h - p) // 2, (w - p) // 2
        noisy, gt = noisy[y : y + p, x : x + p], gt[y : y + p, x : x + p]
        if self.augment:
            noisy, gt = augment(noisy, gt, random.randint(0, 7))
        return to_tensor(noisy), to_tensor(gt)


class SIDDBlocks(Dataset):
    """The official 1280 SIDD sRGB validation blocks (40 images x 32 blocks of 256x256).

    Used as the held-out *test* set: never for training or model selection.
    """

    def __init__(self, raw_dir: str | Path = "data/raw", limit: int | None = None):
        raw_dir = Path(raw_dir)
        noisy = load_mat_blocks(raw_dir / "ValidationNoisyBlocksSrgb.mat")
        gt = load_mat_blocks(raw_dir / "ValidationGtBlocksSrgb.mat")
        assert noisy.shape == gt.shape == (40, 32, 256, 256, 3), noisy.shape
        self.noisy = noisy.reshape(-1, 256, 256, 3)
        self.gt = gt.reshape(-1, 256, 256, 3)
        self.image_ids = np.repeat(np.arange(40), 32)
        if limit is not None:
            self.noisy, self.gt, self.image_ids = self.noisy[:limit], self.gt[:limit], self.image_ids[:limit]

    def __len__(self) -> int:
        return len(self.gt)

    def __getitem__(self, i: int):
        return to_tensor(self.noisy[i]), to_tensor(self.gt[i])
