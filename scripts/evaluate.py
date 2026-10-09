"""Evaluate on the 1280 official SIDD sRGB validation blocks and write results/<name>.json.

    # a trained checkpoint
    python scripts/evaluate.py --ckpt runs/nafnet_w32_scratch/best.pt --name nafnet_w32_scratch
    # a pretrained model without training (zero-shot / reference)
    python scripts/evaluate.py --model scunet --weights weights/scunet_color_real_psnr.pth --name scunet_real_psnr_zeroshot
    # x8 self-ensemble
    python scripts/evaluate.py --ckpt runs/nafnet_w32_scratch/best.pt --name nafnet_w32_scratch_tta --tta
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sidd_denoise.evaluate import evaluate_model, load_trained  # noqa: E402
from sidd_denoise.models import build_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", help="best.pt from scripts/train.py")
    ap.add_argument("--model", help="architecture name for an untrained / pretrained-only model")
    ap.add_argument("--weights", help="official pretrained checkpoint for --model")
    ap.add_argument("--kwargs", default="{}", help='JSON constructor kwargs, e.g. \'{"sigma": 25}\'')
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--dists", action="store_true", help="also compute DISTS (needs pyiqa)")
    ap.add_argument("--device")
    ap.add_argument("--limit", type=int, help="only the first N blocks (debugging)")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="results")
    ap.add_argument("--note", default="", help="free-text note stored in the JSON (e.g. data caveats)")
    args = ap.parse_args()

    extra = {"note": args.note} if args.note else {}
    if args.ckpt:
        model, cfg = load_trained(args.ckpt)
        extra.update({"ckpt": args.ckpt, "train_cfg": cfg["name"]})
        summary = Path(args.ckpt).parent / "summary.json"
        if summary.exists():
            extra["train"] = json.loads(summary.read_text())
    elif args.model:
        model = build_model({"name": args.model, "weights": args.weights, "kwargs": json.loads(args.kwargs)})
        extra.update({"weights": args.weights})
    else:
        ap.error("give --ckpt or --model")

    res = evaluate_model(args.name, model, args.raw, args.out, args.device, args.tta, args.limit, extra, args.dists)
    keys = [k for k in ("psnr", "ssim", "ssim2d", "lpips", "dists") if k in res]
    print(f"{args.name}: " + "  ".join(f"{k}={res[k]:.4f}" for k in keys) + f"  ({res['ms_per_block']:.1f} ms/block)")


if __name__ == "__main__":
    main()
