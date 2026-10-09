# SIDD Real-Noise Denoising — Project Write-up

Oct 9, 2026

## Summary

Fine-tuning a pretrained SCUNet on SIDD reaches **39.45 dB PSNR / 0.957 SSIM** on the 1280 official SIDD validation blocks. That is +6.3 dB over a tuned Gaussian filter and +5.4 dB over tuned BM3D. The run took about an hour on a laptop GPU (Apple M4 Pro).

The task was to train an image-restoration model that beats simple baselines on PSNR/SSIM on an 8–12 GB GPU or CPU. I chose real smartphone-noise denoising (SIDD sRGB). I compared a strong model trained from scratch (NAFNet-w32) with transfer learning from a model pretrained only on synthetic noise (SCUNet), fine-tuned three ways: full, LoRA, and head+tail only.

- **Transfer learning wins:** at equal compute, fine-tuned SCUNet beats NAFNet from scratch by +0.95 dB, on 40 of 40 test images.
- **Parameter efficiency:** LoRA with 5.7% of the parameters trainable comes within 0.04 dB of full fine-tuning.
- **Reference ceiling:** the result lands within 0.43 dB of the official NAFNet checkpoint, which had about 100× the compute and 2× the data.
- **A bug found and fixed late:** the first round trained with an unshuffled data loader. I re-ran most experiments on Oct 8–9. A few runs, including the 4.4 h NAFNet main run, still predate the fix and are marked ‡ below.

Deliverables: a Python package with CLI scripts, YAML configs, 23 tests, a report notebook, a Colab notebook and a README.

## Problem and data

The input is a real noisy smartphone photo (sRGB). The output should approximate the clean image. Real sRGB noise is signal-dependent and spatially correlated by the camera's image pipeline, so it is much harder than synthetic Gaussian noise.

| Split | Source | Size | Use |
| --- | --- | --- | --- |
| Train | SIDD-Small sRGB, 144 of 160 noisy/clean pairs | 13,752 tiles of 512², random 256² or 128² crops | Training |
| Holdout | 16 SIDD-Small pairs, stratified across the 5 phones | 1,552 tiles | Early stopping; σ tuning for Gaussian, BM3D, DRUNet |
| Test | Official SIDD sRGB validation blocks | 1280 blocks of 256² (40 images × 32) | Final scores only |

SIDD has only 10 scenes, so the holdout is disjoint by capture, not by scene. The test set was never used for any decision.

## Approach and rationale

I checked recent (2024–2026) work on architectures, transfer learning and SIDD evaluation practice before designing anything. Each choice below follows from what that check found.

| Question | What the literature says | Decision |
| --- | --- | --- |
| Best architecture for the compute? | No newer model with released code clearly beats CGNet (40.39 dB) or NAFNet (w32: 39.97 dB at ~16 GMACs). Mamba models need CUDA-only kernels. | NAFNet-w32 from scratch, plus a plain U-Net ablation |
| What to transfer from? | Public NAFNet, Restormer and CGNet SIDD weights were trained on SIDD-Medium, which contains our training set, so fine-tuning them would be circular. SCUNet's real-noise weights never saw SIDD. | SCUNet-real as the transfer source; DRUNet (Gaussian-noise prior) as a weaker control; official NAFNet only as a flagged ceiling |
| All-in-one or diffusion models? | All-in-one models score about 24.4–24.9 dB on SIDD zero-shot. Diffusion restorers have 1–5B parameters and cannot be fine-tuned on this hardware. | Ruled out |
| Parameter-efficient fine-tuning? | LoRA for restoration (CoLoRA, LoRA-IR, 2024) is current practice. | LoRA r=8 vs full vs head+tail, plus a data-efficiency sweep |
| Training recipe? | SIDD leaders use pixel losses. NAFNet uses PSNR loss with AdamW (β 0.9, 0.9), cosine LR and no augmentation. | Same recipe, with warmup and EMA of the weights |

The core comparison is therefore **architecture from scratch vs. a synthetic-noise prior adapted to real data**. That is the question the brief's "transfer learning and pre-trained backbones" points at.

## Evaluation protocol

The pipeline reproduces three published reference numbers, which verifies it: noisy input 23.66 dB, BM3D with estimated σ 24.89 dB, and the official NAFNet-w32 checkpoint 39.88 dB / 0.959 SSIM (paper: 39.97 / 0.960).

- **Quantize first:** outputs are clamped to [0, 1] and rounded to uint8 before any metric.
- **PSNR:** RGB, data range 255, no border crop, computed per 256² block and averaged over 1280 blocks.
- **SSIM:** the SIDD convention, i.e. MATLAB's 3-D SSIM, an 11×11×11 Gaussian over height, width and colour. Implemented separably and unit-tested against BasicSR's conv3d version. Per-channel SSIM is reported as well.
- **Perceptual:** LPIPS (AlexNet) and DISTS.
- **Uncertainty:** 95% bootstrap CIs over the 40 test images, plus paired bootstrap tests between models.

Two details matter when comparing against papers. Per-channel SSIM gives about 0.92 where the SIDD convention gives 0.96 for the same model. BM3D with an estimated σ is a weak strawman on correlated sRGB noise, so it is also reported with σ tuned on the holdout.

## Results

Every learned model beats both classical baselines on every metric. Fine-tuned SCUNet is best on PSNR and SSIM; NAFNet from scratch is best on LPIPS.

| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | Trainable params | Train time |
| --- | --- | --- | --- | --- | --- |
| Noisy input | 23.66 | 0.485 | 0.792 | – | – |
| Gaussian filter, σ tuned | 33.10 | 0.881 | 0.439 | – | – |
| BM3D, σ estimated | 24.89 | 0.543 | 0.744 | – | – |
| BM3D, σ tuned | 34.07 | 0.896 | 0.332 | – | – |
| DRUNet zero-shot (Gaussian-noise prior) | 33.98 | 0.883 | 0.325 | – | – |
| SCUNet-real zero-shot | 35.11 | 0.895 | 0.264 | – | – |
| Plain U-Net, from scratch | 38.22 | 0.947 | 0.224 | 10.0M | 1.3 h |
| NAFNet-w32, from scratch, short (3k it, 128²) | 38.51 | 0.949 | 0.186 | 29.2M | 4.6 h § |
| NAFNet-w32, from scratch (12k it, 256²) ‡ | 38.86 | 0.952 | 0.195 | 29.2M | 4.4 h |
| SCUNet fine-tuned, head+tail only | 37.96 | 0.945 | 0.207 | 3.5K | 0.4 h |
| DRUNet fine-tuned ‡ | 39.07 | 0.953 | 0.215 | 32.6M | 0.3 h |
| SCUNet fine-tuned, LoRA r=8 | 39.42 | 0.956 | 0.210 | 1.02M | 0.8 h |
| **SCUNet fine-tuned, full** | **39.45** | **0.957** | 0.209 | 17.9M | 1.0 h |
| *NAFNet-w32 official checkpoint (ceiling)* | *39.88* | *0.959* | *0.195* | – | many GPU-days |
| *NAFNet-w64 official checkpoint (ceiling)* | *40.21* | *0.961* | *0.188* | – | many GPU-days |

The official checkpoints were trained on SIDD-Medium, which includes our training data, for 200–400k iterations on 8 GPUs. They are an upper bound, not competitors. All training ran on an Apple M4 Pro.

‡ Trained with the unshuffled loader and not re-run; likely lower bounds. § Measured wall-clock; this run was slower per iteration than the main NAFNet run, for reasons I did not investigate.

### The loader bug

The first round trained with `shuffle=False`. Each photo's tiles are stored together, so every batch came from a single image. After the fix (plus a regression test), the U-Net gained +2.4 dB and the short NAFNet runs +0.7–0.8 dB, while the SCUNet fine-tunes moved only about +0.1 dB. The pretrained prior made fine-tuning robust to poor batches; training from scratch was not. Still to re-run: NAFNet-w32 at 12k iterations, the DRUNet fine-tune, x8 test-time augmentation, and the 16-pair SCUNet fine-tunes.

## Findings

1. **Transfer beats training from scratch, with less compute.** At equal compute (3k iterations, 128² crops, fixed loader), fine-tuned SCUNet scores +0.95 dB over NAFNet from scratch (95% CI 0.85–1.04), better on 40 of 40 test images. Against the longer 4.4 h NAFNet run (‡) the gap is +0.59 dB (CI 0.51–0.68). Most of the work is done by the prior: SCUNet zero-shot is already at 35.1 dB, and 3k fine-tuning steps add +4.3 dB.
2. **Data efficiency, partly re-run.** NAFNet from scratch gains +0.36 dB going from 16 to 144 pairs (38.15 → 38.51 dB, CI 0.19–0.56). The 16-pair SCUNet fine-tunes still need the fixed loader. Before the fix they scored 38.92–38.99 dB, above NAFNet trained on all 144 pairs, and the fix barely moved the other fine-tunes, so I expect this to hold.

   ![Data efficiency at equal compute](results/figures/data_efficiency.png)

3. **LoRA nearly matches full fine-tuning.** With 5.7% of the parameters trainable it is 0.04 dB behind (CI 0.02–0.05), with full fine-tuning ahead on 35 of 40 images. Training only the first and last conv layers (3.5K parameters) already recovers +2.9 dB over zero-shot. So much of the synthetic-to-real gap is a colour and scale mismatch at the input and output.
4. **NAFNet's blocks help, less than first measured.** With the fixed loader, NAFNet at 3k iterations beats a plain U-Net at 12k iterations by +0.29 dB (CI 0.22–0.35), on 35 of 40 images. The earlier +3.1 dB came mostly from the loader bug hurting the U-Net. DRUNet's Gaussian-noise prior starts 1.1 dB behind SCUNet zero-shot and ends 0.38 dB behind after fine-tuning (‡).
5. **PSNR and perception disagree.** NAFNet from scratch has lower PSNR than fine-tuned SCUNet but better LPIPS (0.186–0.195 vs 0.209). x8 test-time augmentation on NAFNet (‡) adds +0.06 dB at 8× the inference cost, and worsens LPIPS (0.195 → 0.207) because averaging smooths texture.

## Engineering notes

The first experiment queue ran unattended in about 11 hours on one Apple M4 Pro. Budget, not method, shaped most of the trade-offs.

- **Compute.** Per-step times at batch 8: NAFNet-w32 1.3 s at 256²; SCUNet 0.83 s at 128²; U-Net 0.27 s. On MPS, fp16/bf16 autocast and channels-last gave no speedup, so training stays in fp32. NAFNet got 12k iterations, against 200k in the paper, and was still improving.
- **Faithful ports.** NAFNet, SCUNet and DRUNet load the official weights with strict key matching. The official NAFNet checkpoint reproduces the paper, which validates both the port and the metrics.
- **Bugs found and fixed:**
  - The released SCUNet real-noise weights need a deeper block config ([4]×7) than the repo default.
  - SCUNet's attention crashed on MPS. It was rewritten with fused scaled-dot-product attention, folded to 4-D, and tested against the original to 1e-5.
  - The 3-D SSIM, written naively, took over 10 minutes per 64 blocks. A separable version gives identical values in under 1 s.
  - LPIPS/DISTS were moved from CPU to GPU: the same values to 1e-6, and 8–60× faster.
  - The training loader did not shuffle, so each batch held tiles from one photo. Fixed, regression-tested, and most runs repeated (see Results).
- **Reproducibility.** Configs inherit from a base YAML with command-line overrides. Runs resume from checkpoints, and one script (`run_all.sh`) rebuilds every number. The report notebook renders all tables and figures from saved results, without running a model.

## Limitations and next steps

The headline result is robust (p < 0.001 on 40 test images, with the fixed loader). Several secondary comparisons carry caveats worth stating.

- **Incomplete re-run.** Rows marked ‡ predate the loader fix; comparisons against them are flagged.
- **Single seed.** Each configuration was trained once; the CIs cover test-set sampling, not training randomness. Read gaps under ~0.1 dB (LoRA vs full, TTA) with caution.
- **Holdout is milder than test.** Holdout input PSNR is 27.8 dB, test 23.7 dB. The baselines' σ was tuned on the holdout, which slightly favours small σ. Re-tuning on the test set would be leakage.
- **Shared scenes.** SIDD's 10 scenes appear in both train and test (different captures). This is standard for SIDD, but it does not measure generalization to new phones.
- **Unequal ablations:**
  - The U-Net was given a lower learning rate (5e-4) than NAFNet.
  - DRUNet's learnable noise level started at σ 25, not the tuned 50.
  - A PSNR-loss vs L1 ablation is configured but was not run.
- **Confound in finding 1.** SCUNet brings both a different architecture and a pretrained prior. Pretraining NAFNet on the same synthetic noise would separate the two effects.

Next, in priority order:

1. Finish the re-run with the fixed loader (about 6–8 h on the M4 Pro).
2. Longer schedules on a CUDA GPU: NAFNet at 40k+ iterations, and fine-tunes at 256². Both curves are still rising.
3. Distil fine-tuned SCUNet into a ~2M-parameter NAFNet student for deployment.
4. Cross-dataset tests (DND, PolyU) to measure generalization.
5. Extend the same package to deblurring (GoPro).
