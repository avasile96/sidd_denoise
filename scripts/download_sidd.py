"""Download SIDD-Small sRGB and the official sRGB validation blocks.

Files come from the official SIDD server (York University). A single
connection gets ~1.5 MB/s, so large files are fetched as parallel HTTP range
requests. Each chunk is resumable: re-running the script only fetches what is
missing.

    python scripts/download_sidd.py                # everything
    python scripts/download_sidd.py --only val     # just the 1280 validation blocks

If the server is down, see the manual instructions printed at the end (Kaggle /
CodaLab mirrors).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

BASE = "http://130.63.97.225/share"
FILES = {
    "val": [
        ("SIDD_Blocks/ValidationNoisyBlocksSrgb.mat", 229_724_517),
        ("SIDD_Blocks/ValidationGtBlocksSrgb.mat", 232_055_617),
    ],
    "train": [
        ("SIDD_Small_sRGB_Only.zip", 6_615_978_508),
    ],
}
CHUNK = 64 * 1024 * 1024
MANUAL = """
Manual download (if the York server is unreachable):
  * https://abdokamel.github.io/sidd/  -> SIDD-Small (sRGB only) and "SIDD Benchmark" -> Validation blocks
  * CodaLab mirrors are linked from http://130.63.97.225/share/download_benchmark.html
  * Kaggle: kaggle datasets download -d rajat95gupta/smartphone-image-denoising-dataset
Place ValidationNoisyBlocksSrgb.mat, ValidationGtBlocksSrgb.mat and the extracted
SIDD_Small_sRGB_Only/ folder in data/raw/.
"""


def _fetch_range(url: str, part: Path, start: int, end: int, retries: int = 8) -> int:
    want = end - start + 1
    for attempt in range(retries):
        have = part.stat().st_size if part.exists() else 0
        if have >= want:
            return want
        req = urllib.request.Request(url, headers={"Range": f"bytes={start + have}-{end}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab") as f:
                while buf := r.read(1 << 20):
                    f.write(buf)
        except Exception as e:  # network hiccup -> resume from what we have
            print(f"  retry {attempt + 1} {part.name}: {e}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))
    have = part.stat().st_size if part.exists() else 0
    if have != want:
        raise RuntimeError(f"{part.name}: got {have} of {want} bytes")
    return have


def download(rel: str, size: int, out_dir: Path, workers: int) -> Path:
    dst = out_dir / Path(rel).name
    if dst.exists() and dst.stat().st_size == size:
        print(f"[ok] {dst.name} already complete")
        return dst
    url = f"{BASE}/{rel}"
    parts_dir = out_dir / f".{dst.name}.parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    ranges = [(i, s, min(s + CHUNK, size) - 1) for i, s in enumerate(range(0, size, CHUNK))]
    print(f"[get] {dst.name}: {size / 1e9:.2f} GB in {len(ranges)} chunks x {workers} workers")
    t0 = time.time()
    done = 0
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(_fetch_range, url, parts_dir / f"{i:05d}", s, e) for i, s, e in ranges]
        for fut in cf.as_completed(futs):
            done += fut.result()
            mbps = done / 1e6 / max(time.time() - t0, 1e-6)
            print(f"  {dst.name}: {done / size:6.1%}  {mbps:6.1f} MB/s", flush=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with open(tmp, "wb") as out:
        for i, _, _ in ranges:
            with open(parts_dir / f"{i:05d}", "rb") as f:
                while buf := f.read(1 << 24):
                    out.write(buf)
    assert tmp.stat().st_size == size, "size mismatch after merge"
    os.replace(tmp, dst)
    for p in parts_dir.iterdir():
        p.unlink()
    parts_dir.rmdir()
    print(f"[done] {dst.name} in {time.time() - t0:.0f}s")
    return dst


def check_val_blocks(out_dir: Path) -> None:
    from sidd_denoise.data import load_mat_blocks

    for name in ("ValidationNoisyBlocksSrgb.mat", "ValidationGtBlocksSrgb.mat"):
        arr = load_mat_blocks(out_dir / name)
        assert arr.shape == (40, 32, 256, 256, 3) and arr.dtype.name == "uint8", arr.shape
        print(f"[check] {name}: {arr.shape} {arr.dtype}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    ap.add_argument("--only", choices=["val", "train"], default=None)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--no-extract", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    groups = [args.only] if args.only else ["val", "train"]
    try:
        for g in groups:
            for rel, size in FILES[g]:
                path = download(rel, size, args.out, args.workers)
                if path.suffix == ".zip" and not args.no_extract:
                    target = args.out / "SIDD_Small_sRGB_Only"
                    if not target.exists():
                        print(f"[unzip] {path.name} -> {args.out}")
                        with zipfile.ZipFile(path) as z:
                            z.extractall(args.out)
        if "val" in groups:
            check_val_blocks(args.out)
    except Exception as e:
        print(f"\nDownload failed: {e}\n{MANUAL}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    main()
