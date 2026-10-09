"""Training loop: AdamW + warmup/cosine, EMA, AMP on CUDA, holdout validation, resumable checkpoints."""
from __future__ import annotations

import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import TileDataset, get_device
from .ema import EMA
from .losses import build_loss
from .metrics import psnr, quantize
from .models import build_model, count_params


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def lr_lambda(it: int, warmup: int, total: int, eta_min_ratio: float) -> float:
    if it < warmup:
        return (it + 1) / warmup
    t = (it - warmup) / max(1, total - warmup)
    return eta_min_ratio + (1 - eta_min_ratio) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    scores = []
    for noisy, gt in loader:
        pred = model(noisy.to(device)).float().cpu()
        for p, g in zip(quantize(pred), quantize(gt)):
            scores.append(psnr(p, g))
    return float(np.mean(scores))


def _worker_init(worker_id: int) -> None:
    s = torch.initial_seed() % 2**32
    random.seed(s)
    np.random.seed(s)


def make_train_loader(ds: TileDataset, batch_size: int, num_workers: int, seed: int) -> DataLoader:
    """Shuffled loader. index.json groups each pair's tiles together, so a sequential
    sampler would fill every batch from a single image."""
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True, drop_last=True, generator=torch.Generator().manual_seed(seed),
        num_workers=num_workers, persistent_workers=num_workers > 0, worker_init_fn=_worker_init,
    )


def train(cfg: dict) -> dict:
    seed_everything(cfg.get("seed", 0))
    device = get_device(cfg.get("device"))
    out = Path(cfg.get("out_dir", "runs")) / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    d, o, s = cfg["data"], cfg["optim"], cfg["sched"]
    iters, bs = int(s["iters"]), int(d["batch_size"])
    train_ds = TileDataset(
        d["root"], "train", d["patch_size"], augment=d.get("augment", False),
        num_pairs=d.get("num_pairs"), pair_seed=d.get("pair_seed", 0), length=iters * bs,
    )
    val_ds = TileDataset(d["root"], "holdout", d.get("val_patch_size", 512), random_crop=False)
    max_val = cfg.get("val", {}).get("max_tiles")
    if max_val and len(val_ds.tiles) > max_val:  # fixed, evenly spaced subset -> comparable across runs
        val_ds.tiles = val_ds.tiles[:: math.ceil(len(val_ds.tiles) / max_val)]
        val_ds.length = len(val_ds.tiles)
    nw = int(d.get("num_workers", 4))
    train_loader = make_train_loader(train_ds, bs, nw, cfg.get("seed", 0))
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=min(nw, 2))

    model = build_model(cfg["model"]).to(device)
    n_total, n_train = count_params(model), count_params(model, trainable_only=True)
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(
        params, lr=float(o["lr"]), betas=tuple(o.get("betas", (0.9, 0.999))), weight_decay=float(o.get("weight_decay", 0.0))
    )
    eta_min_ratio = float(s.get("eta_min", 1e-7)) / float(o["lr"])
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda it: lr_lambda(it, int(s.get("warmup", 1000)), iters, eta_min_ratio))
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = EMA(model, decay=float(cfg.get("ema", 0.999))) if cfg.get("ema", 0.999) else None
    loss_fn = build_loss(cfg.get("loss", "psnr"))
    clip = float(o.get("grad_clip", 1.0))

    start_it, best, elapsed = 0, -1.0, 0.0
    last = out / "last.pt"
    if last.exists() and cfg.get("resume", True):
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optim.load_state_dict(ck["optim"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        if ema and ck.get("ema"):
            ema.load_state_dict(ck["ema"])
        start_it, best, elapsed = ck["iter"], ck["best"], ck.get("elapsed", 0.0)
        # Draw only the remaining iterations' samples. The shuffle order is re-drawn on
        # resume, so a resumed run is not bit-identical to an uninterrupted one.
        train_ds.length = (iters - start_it) * bs
        print(f"resumed from {last} at iter {start_it} (best {best:.2f} dB)")

    print(f"[{cfg['name']}] device={device} amp={use_amp} params={n_total / 1e6:.2f}M trainable={n_train / 1e6:.3f}M "
          f"train_pairs={len(train_ds.pairs)} train_tiles={len(train_ds.tiles)} val_tiles={len(val_ds.tiles)}")
    log_path = out / "log.csv"
    new_log = not log_path.exists() or start_it == 0
    log_f = open(log_path, "w" if new_log else "a", newline="")
    log = csv.writer(log_f)
    if new_log:
        log.writerow(["iter", "loss", "lr", "val_psnr", "elapsed_s"])

    val_every = int(cfg.get("val", {}).get("every", 2000))
    log_every = int(cfg.get("log_every", 100))
    eval_model = lambda: ema.module if ema else model  # noqa: E731

    def save(it: int) -> None:
        torch.save({
            "model": model.state_dict(), "optim": optim.state_dict(), "sched": sched.state_dict(),
            "scaler": scaler.state_dict(), "ema": ema.state_dict() if ema else None,
            "iter": it, "best": best, "elapsed": elapsed, "cfg": cfg,
        }, out / "last.tmp")
        (out / "last.tmp").replace(last)

    it, t0 = start_it, time.time()
    model.train()
    for noisy, gt in train_loader:
        if it >= iters:
            break
        noisy, gt = noisy.to(device, non_blocking=True), gt.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            pred = model(noisy)
        loss = loss_fn(pred, gt)
        optim.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(params, clip)
        scaler.step(optim)
        scaler.update()
        sched.step()
        if ema:
            ema.update(model)
        it += 1

        if it % log_every == 0:
            now = time.time()
            elapsed += now - t0
            t0 = now
            lr = optim.param_groups[0]["lr"]
            print(f"  it {it:6d}/{iters}  loss {loss.item():8.4f}  lr {lr:.2e}  {elapsed / max(it - start_it, 1):.3f}s/it", flush=True)
            log.writerow([it, f"{loss.item():.5f}", f"{lr:.3e}", "", f"{elapsed:.1f}"])
        if it % val_every == 0 or it == iters:
            v = validate(eval_model(), val_loader, device)
            model.train()
            if v > best:
                best = v
                torch.save({"model": eval_model().state_dict(), "cfg": cfg, "iter": it, "val_psnr": v}, out / "best.pt")
            print(f"  [val] it {it}  holdout PSNR {v:.3f} dB  (best {best:.3f})", flush=True)
            log.writerow([it, "", "", f"{v:.4f}", f"{elapsed:.1f}"])
            log_f.flush()
            save(it)
    elapsed += time.time() - t0
    log_f.close()
    summary = {
        "name": cfg["name"], "device": str(device), "iters": it, "best_val_psnr": best,
        "train_hours": elapsed / 3600, "params": n_total, "trainable_params": n_train,
        "train_pairs": len(train_ds.pairs),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
