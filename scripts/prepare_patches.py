"""Cut SIDD-Small sRGB pairs into 512x512 tiles and write a train/holdout split.

Training takes random 256x256 crops from these tiles, which is much faster than
decoding 12 MP PNGs every step (the NAFNet recipe does the same, with LMDB).

Split: 16 of the 160 pairs are held out for early stopping and baseline tuning,
stratified by smartphone. SIDD has only 10 scenes, so a scene-disjoint split would
throw away most of the data. The test set is always the official validation blocks.

    python scripts/prepare_patches.py --src data/raw/SIDD_Small_sRGB_Only --out data/patches
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2


def parse_name(name: str) -> dict:
    inst, scene, cam, iso, shutter, temp, bright = name.split("_")
    return {"pair": name, "instance": inst, "scene": scene, "camera": cam, "iso": int(iso),
            "shutter": int(shutter), "temperature": int(temp), "brightness": bright}


def holdout_split(names: list[str], n_holdout: int, seed: int) -> set[str]:
    by_cam = defaultdict(list)
    for n in sorted(names):
        by_cam[parse_name(n)["camera"]].append(n)
    rng = random.Random(seed)
    total = len(names)
    # Proportional allocation per camera (largest remainder), at least one each.
    quota = {c: max(1, round(n_holdout * len(v) / total)) for c, v in by_cam.items()}
    while sum(quota.values()) > n_holdout:
        quota[max(quota, key=quota.get)] -= 1
    while sum(quota.values()) < n_holdout:
        quota[max(by_cam, key=lambda c: len(by_cam[c]) / quota[c])] += 1
    return {n for c, v in by_cam.items() for n in rng.sample(v, quota[c])}


def tile_pair(args) -> list[dict]:
    pair_dir, out_root, split, tile, stride = args
    gt = cv2.imread(str(next(pair_dir.glob("GT_SRGB_*.PNG"))), cv2.IMREAD_COLOR)
    noisy = cv2.imread(str(next(pair_dir.glob("NOISY_SRGB_*.PNG"))), cv2.IMREAD_COLOR)
    assert gt.shape == noisy.shape, pair_dir
    h, w = gt.shape[:2]
    ys = list(range(0, h - tile + 1, stride))
    xs = list(range(0, w - tile + 1, stride))
    # Make sure the bottom/right borders are covered.
    if ys[-1] != h - tile:
        ys.append(h - tile)
    if xs[-1] != w - tile:
        xs.append(w - tile)
    meta = parse_name(pair_dir.name)
    out = []
    for kind, img in (("gt", gt), ("noisy", noisy)):
        (out_root / split / kind).mkdir(parents=True, exist_ok=True)
    for y in ys:
        for x in xs:
            stem = f"{pair_dir.name}_{y:04d}_{x:04d}.png"
            rel = {}
            for kind, img in (("gt", gt), ("noisy", noisy)):
                path = out_root / split / kind / stem
                if not path.exists():
                    cv2.imwrite(str(path), img[y : y + tile, x : x + tile], [cv2.IMWRITE_PNG_COMPRESSION, 1])
                rel[kind] = str(path.relative_to(out_root))
            out.append({**meta, "split": split, "y": y, "x": x, **rel})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path("data/raw/SIDD_Small_sRGB_Only"))
    ap.add_argument("--out", type=Path, default=Path("data/patches"))
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--stride", type=int, default=384)
    ap.add_argument("--n-holdout", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    pair_dirs = sorted(p for p in (args.src / "Data").iterdir() if p.is_dir())
    assert len(pair_dirs) == 160, f"expected 160 SIDD-Small pairs, found {len(pair_dirs)}"
    holdout = holdout_split([p.name for p in pair_dirs], args.n_holdout, args.seed)
    jobs = [(p, args.out, "holdout" if p.name in holdout else "train", args.tile, args.stride) for p in pair_dirs]

    tiles = []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, res in enumerate(ex.map(tile_pair, jobs), 1):
            tiles.extend(res)
            if i % 20 == 0:
                print(f"  {i}/{len(jobs)} pairs, {len(tiles)} tiles", flush=True)

    index = {"tile": args.tile, "stride": args.stride, "seed": args.seed,
             "holdout_pairs": sorted(holdout), "tiles": tiles}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "index.json").write_text(json.dumps(index))
    n_tr = sum(t["split"] == "train" for t in tiles)
    cams = sorted({parse_name(p)["camera"] for p in holdout})
    print(f"done: {n_tr} train tiles, {len(tiles) - n_tr} holdout tiles; holdout cameras {cams}")


if __name__ == "__main__":
    main()
