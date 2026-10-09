"""Build the results tables and figures from results/*.json (no model inference needed)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .evaluate import VISUAL_BLOCKS
from .metrics import paired_bootstrap, per_image_means

# Display order, label and group of every result that may exist.
ROWS = [
    ("noisy_input", "Noisy input", "baseline"),
    ("gaussian", "Gaussian filter (σ tuned)", "baseline"),
    ("bm3d_est", "CBM3D (σ estimated)", "baseline"),
    ("bm3d_tuned", "CBM3D (σ tuned)", "baseline"),
    ("drunet_zeroshot", "DRUNet zero-shot (AWGN prior)", "zero-shot"),
    ("scunet_real_psnr_zeroshot", "SCUNet-real zero-shot", "zero-shot"),
    ("scunet_real_gan_zeroshot", "SCUNet-real-GAN zero-shot", "zero-shot"),
    ("unet_scratch", "Plain U-Net, from scratch", "trained"),
    ("nafnet_w32_scratch", "NAFNet-w32, from scratch", "trained"),
    ("nafnet_w32_scratch_tta", "NAFNet-w32, from scratch + x8 TTA", "trained"),
    ("drunet_ft_full", "DRUNet fine-tuned (full)", "transfer"),
    ("scunet_ft_headtail", "SCUNet fine-tuned (head+tail only)", "transfer"),
    ("scunet_ft_lora", "SCUNet fine-tuned (LoRA r=8)", "transfer"),
    ("scunet_ft_full", "SCUNet fine-tuned (full)", "transfer"),
    ("scunet_ft_full_tta", "SCUNet fine-tuned (full) + x8 TTA", "transfer"),
    ("ref_nafnet_sidd_w32", "NAFNet-w32 official SIDD ckpt †", "reference"),
    ("ref_nafnet_sidd_w64", "NAFNet-w64 official SIDD ckpt †", "reference"),
]
SWEEP = {  # method -> {n_pairs: result name}
    "NAFNet from scratch": {16: "sweep_nafnet_p16", 144: "sweep_nafnet_p144"},
    "SCUNet full fine-tune": {16: "scunet_ft_full_p16", 144: "scunet_ft_full"},
    "SCUNet LoRA": {16: "scunet_ft_lora_p16", 144: "scunet_ft_lora"},
}


def load_results(results_dir: str | Path = "results") -> dict[str, dict]:
    out = {}
    for p in Path(results_dir).glob("*.json"):
        d = json.loads(p.read_text())
        if isinstance(d, dict) and "psnr" in d:
            out[p.stem] = d
    return out


def per_block(name: str, results_dir="results") -> dict[str, np.ndarray]:
    return dict(np.load(Path(results_dir) / f"{name}_per_block.npz"))


def results_table(res: dict, results_dir="results") -> list[dict]:
    rows = []
    for key, label, group in ROWS:
        if key not in res:
            continue
        r = res[key]
        tr = r.get("train") or {}
        rows.append({
            "key": key, "method": label, "group": group,
            "psnr": r["psnr"], "psnr_ci": r["psnr_ci95_images"], "ssim": r["ssim"], "ssim2d": r["ssim2d"],
            "lpips": r.get("lpips"), "dists": r.get("dists"),
            "params_m": (r.get("params") or 0) / 1e6,
            "trainable_m": (tr.get("trainable_params") or 0) / 1e6 if tr else None,
            "gmacs": r.get("gmacs_256"), "ms_block": r.get("ms_per_block"),
            "train_h": tr.get("train_hours"), "iters": tr.get("iters"),
        })
    return rows


def markdown_table(rows: list[dict]) -> str:
    def f(v, fmt):
        return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else format(v, fmt)

    head = "| Method | PSNR ↑ (95% CI) | SSIM ↑ | SSIM-2D ↑ | LPIPS ↓ | DISTS ↓ | Params (trainable) | GMACs | Train h |\n"
    head += "|---|---|---|---|---|---|---|---|---|\n"
    lines = []
    for r in rows:
        ci = f"{r['psnr']:.2f} ({r['psnr_ci'][0]:.2f}–{r['psnr_ci'][1]:.2f})"
        params = "–" if not r["params_m"] else f"{r['params_m']:.1f}M"
        if r["trainable_m"] and abs(r["trainable_m"] - r["params_m"]) > 1e-3:
            t = r["trainable_m"]
            params += f" ({t * 1e3:.1f}K)" if t < 0.01 else f" ({t:.2f}M)"
        lines.append(
            f"| {r['method']} | {ci} | {f(r['ssim'], '.4f')} | {f(r['ssim2d'], '.4f')} | {f(r['lpips'], '.4f')} | "
            f"{f(r['dists'], '.4f')} | {params} | {f(r['gmacs'], '.1f')} | {f(r['train_h'], '.2f')} |"
        )
    return head + "\n".join(lines)


def significance(res: dict, pairs: list[tuple[str, str]], results_dir="results") -> list[dict]:
    """Paired bootstrap over the 40 test images (per-image mean PSNR), a better than b."""
    out = []
    for a, b in pairs:
        if a not in res or b not in res:
            continue
        pa, pb = per_block(a, results_dir), per_block(b, results_dir)
        ia, ib = per_image_means(pa["psnr"], pa["image_ids"]), per_image_means(pb["psnr"], pb["image_ids"])
        r = paired_bootstrap(ia, ib)
        out.append({"a": a, "b": b, **r, "wins": int((ia > ib).sum()), "n": len(ia)})
    return out


def plot_bars(rows, path=None):
    colors = {"baseline": "#9aa0a6", "zero-shot": "#e8a33d", "trained": "#4c78a8", "transfer": "#54a24b", "reference": "#bbbbbb"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.42 * len(rows) + 1.2), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, key, title in [(axes[0], "psnr", "PSNR (dB) ↑"), (axes[1], "ssim", "SSIM ↑")]:
        vals = [r[key] for r in rows]
        ax.barh(y, vals, color=[colors[r["group"]] for r in rows])
        for yi, v in zip(y, vals):
            ax.text(v, yi, f" {v:.2f}" if key == "psnr" else f" {v:.3f}", va="center", fontsize=8)
        ax.set_title(title)
        ax.set_xlim(min(vals) * 0.95, max(vals) * 1.04)
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(y, [r["method"] for r in rows], fontsize=9)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_perception_distortion(rows, path=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in rows:
        if r["lpips"] is None or r["group"] == "baseline" and r["key"] == "noisy_input":
            continue
        ax.scatter(r["psnr"], r["lpips"], s=40)
        ax.annotate(r["method"].replace(" fine-tuned", " FT"), (r["psnr"], r["lpips"]), fontsize=7,
                    xytext=(4, 2), textcoords="offset points")
    ax.set_xlabel("PSNR (dB) →  better fidelity")
    ax.set_ylabel("LPIPS ↓  better perceptual similarity")
    ax.set_title("Perception–distortion view of the test set")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_data_efficiency(res, path=None):
    fig, ax = plt.subplots(figsize=(7, 4.8))
    top = {}  # at each x, the highest point gets its label above, the others below
    for pts in SWEEP.values():
        for n, k in pts.items():
            if k in res:
                top[n] = max(top.get(n, -1.0), res[k]["psnr"])
    for method, pts in SWEEP.items():
        xs = [n for n, k in sorted(pts.items()) if k in res]
        ys = [res[pts[n]]["psnr"] for n in xs]
        if xs:
            ax.plot(xs, ys, "o-", label=method)
            for x_, y_ in zip(xs, ys):
                ax.annotate(f"{y_:.2f}", (x_, y_), fontsize=8, xytext=(6, 6) if y_ >= top[x_] else (6, -12),
                            textcoords="offset points")
    refs = [("nafnet_w32_scratch", "-.", "NAFNet from scratch, full budget (12k its, 256², 4.4 h)"),
            ("scunet_real_psnr_zeroshot", ":", "SCUNet zero-shot (0 SIDD pairs)"),
            ("bm3d_tuned", "--", "CBM3D (tuned)")]
    for key, ls, lab in refs:
        if key in res:
            ax.axhline(res[key]["psnr"], ls=ls, color="gray", lw=1)
            ax.text(150, res[key]["psnr"] - 0.08, f"{lab}: {res[key]['psnr']:.2f}", fontsize=7.5, color="dimgray",
                    ha="right", va="top")
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks([16, 144], ["16", "144 (all)"])
    ax.set_xlim(12, 200)
    ax.set_xlabel("SIDD-Small training pairs")
    ax.set_ylabel("Test PSNR (dB)")
    ax.set_title("Data efficiency at equal compute (3k iters, 128² patches)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_training_curves(run_names, runs_dir="runs", path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name in run_names:
        p = Path(runs_dir) / name / "log.csv"
        if not p.exists():
            continue
        rows = list(csv.DictReader(p.open()))
        tr = [(int(r["iter"]), float(r["loss"])) for r in rows if r["loss"]]
        va = [(int(r["iter"]), float(r["val_psnr"])) for r in rows if r["val_psnr"]]
        if tr:
            it, lo = zip(*tr)
            k = max(1, len(lo) // 50)
            smooth = np.convolve(lo, np.ones(k) / k, mode="valid")
            axes[0].plot(it[k - 1:], smooth, label=name)
        if va:
            axes[1].plot(*zip(*va), "o-", ms=3, label=name)
    axes[0].set_title("Training loss (per-batch, lightly smoothed; PSNR loss = −PSNR)")
    axes[1].set_title("Holdout PSNR (EMA weights)")
    for ax in axes:
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_visual_grid(methods: list[tuple[str, str]], raw_dir="data/raw", results_dir="results",
                     blocks=None, crop=96, path=None):
    """Rows = test blocks, columns = noisy / methods / GT; shows a zoomed centre crop."""
    from .data import SIDDBlocks

    ds = SIDDBlocks(raw_dir)
    blocks = blocks or VISUAL_BLOCKS[:5]
    cols = [("Noisy", None)] + methods + [("Ground truth", "__gt__")]
    fig, axes = plt.subplots(len(blocks), len(cols), figsize=(2.1 * len(cols), 2.2 * len(blocks)))
    s = (256 - crop) // 2
    for i, b in enumerate(blocks):
        gt = ds.gt[b]
        for j, (label, key) in enumerate(cols):
            if key is None:
                img = ds.noisy[b]
            elif key == "__gt__":
                img = gt
            else:
                p = Path(results_dir) / "visuals" / key / f"block_{b:04d}.png"
                img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) if p.exists() else np.zeros_like(gt)
            ax = axes[i, j]
            ax.imshow(img[s : s + crop, s : s + crop], interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            if key != "__gt__":
                d = 10 * np.log10(255**2 / max(np.mean((img.astype(float) - gt) ** 2), 1e-10))
                ax.set_xlabel(f"{d:.2f} dB", fontsize=8)
            if i == 0:
                ax.set_title(label, fontsize=9)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_gain_vs_noise(res, keys, results_dir="results", path=None):
    """Per test image: PSNR gain over the noisy input vs how noisy the input is."""
    base = per_block("noisy_input", results_dir)
    b_img = per_image_means(base["psnr"], base["image_ids"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k in keys:
        if k not in res:
            continue
        pb = per_block(k, results_dir)
        g = per_image_means(pb["psnr"], pb["image_ids"]) - b_img
        ax.scatter(b_img, g, s=18, label=dict((r[0], r[1]) for r in ROWS).get(k, k))
    ax.set_xlabel("Noisy-input PSNR of the test image (dB)  ← noisier")
    ax.set_ylabel("PSNR gain over noisy input (dB)")
    ax.set_title("Where the gains come from (40 test images)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
