"""Download pretrained checkpoints into weights/.

* SCUNet real-noise (PSNR + GAN) and DRUNet color: official KAIR GitHub release (cszn).
* NAFNet-SIDD w32/w64 (reference ceiling only): the official links are Google Drive
  (github.com/megvii-research/NAFNet); for scripted download we use the Hugging Face
  mirror nyanko7/nafnet-models. Its w32 file reproduces the paper (40.0 dB on SIDD val),
  which verifies the mirror's integrity.

    python scripts/download_weights.py
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

KAIR = "https://github.com/cszn/KAIR/releases/download/v1.0"
HF = "https://huggingface.co/nyanko7/nafnet-models/resolve/main"
FILES = {
    "scunet_color_real_psnr.pth": (f"{KAIR}/scunet_color_real_psnr.pth", 71_982_841),
    "scunet_color_real_gan.pth": (f"{KAIR}/scunet_color_real_gan.pth", 71_982_835),
    "drunet_color.pth": (f"{KAIR}/drunet_color.pth", 130_579_305),
    "NAFNet-SIDD-width32.pth": (f"{HF}/NAFNet-SIDD-width32.pth", 116_861_841),
    "NAFNet-SIDD-width64.pth": (f"{HF}/NAFNet-SIDD-width64.pth", 464_154_961),
}


def main(out: Path = Path("weights")) -> None:
    out.mkdir(exist_ok=True)
    for name, (url, size) in FILES.items():
        dst = out / name
        if dst.exists() and dst.stat().st_size == size:
            print(f"[ok] {name}")
            continue
        print(f"[get] {name} ({size / 1e6:.0f} MB)")
        tmp = dst.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        if tmp.stat().st_size != size:
            sys.exit(f"size mismatch for {name}: got {tmp.stat().st_size}, expected {size}")
        tmp.replace(dst)


if __name__ == "__main__":
    main()
