"""Train a denoiser from a YAML config.

    python scripts/train.py --config configs/nafnet_w32_scratch.yaml
    python scripts/train.py --config configs/scunet_ft_full.yaml --set sched.iters=2000 data.num_pairs=16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sidd_denoise.config import load_config  # noqa: E402
from sidd_denoise.train import train  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted overrides, e.g. optim.lr=5e-4")
    ap.add_argument("--name", help="override the run name")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    if args.name:
        cfg["name"] = args.name
    print(train(cfg))


if __name__ == "__main__":
    main()
