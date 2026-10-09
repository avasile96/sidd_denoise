# Computer Vision & Deep Learning Concepts — A Refresher

Oct 1, 2026

A working glossary for the SIDD denoising project and for interviews: each term is defined in plain words and, where it applies, tied to where it shows up in the project. Every acronym is spelled out where it is used; the full list is in [ACRONYMS.md](ACRONYMS.md).

## At a glance

Training loops data → model → loss → optimizer; evaluation scores it once.

```
┌────────────────────┐   ┌────────────────────────┐   ┌──────────────────┐   ┌──────────────┐
│ Training data      │──▶│ Network                │──▶│ Loss             │──▶│ Optimizer    │
│ noisy + clean      │   │ NAFNet, SCUNet (U-Nets)│   │ output vs clean  │   │ AdamW updates│
│ SIDD-Small, 256²   │   │ + optional pretrained  │   │ PSNR loss        │   │ the weights  │
└────────────────────┘   └────────────────────────┘   └──────────────────┘   └──────┬───────┘
                                │          ▲                                        │
                after training  │          └──── backpropagation: repeat for ───────┘
                                ▼                thousands of iterations
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│ Evaluation: run the trained model on the untouched test set; score PSNR, SSIM, LPIPS      │
│ against baselines                                                                         │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

Every supervised restoration project has this shape. The sections below define the terms behind each box, in roughly that order.

## 1. Images as data

To a network, an image is a 3-D array of numbers: height × width × channels.

| Term | What it means | In this project |
|---|---|---|
| Pixel | One sample of the image grid. Each pixel holds one value per channel. | A SIDD photo is 3000×5328 pixels (~16 megapixels). |
| Channel | One plane of the array: R, G and B for colour, 1 for grayscale. Inside a network, "channels" also means the feature maps a layer outputs. | Images have 3 channels; NAFNet-w32's first layer outputs 32. |
| Tensor layout (NCHW / NHWC) | The order of axes: batch N, channels C, height H, width W. PyTorch uses NCHW; image libraries use HWC. | Data is loaded as HWC uint8 and converted to NCHW float. |
| Bit depth / uint8 (unsigned 8-bit integer) | How many levels each value can take. 8-bit is 0–255; RAW sensors give 10–14 bits. | Outputs are rounded back to uint8 before scoring. |
| Normalization to [0, 1] | Dividing by 255 so the network sees small floats. | Every tensor in training. |
| RAW vs sRGB (standard red-green-blue) | RAW is the sensor's linear, unprocessed data. sRGB is the final display-ready image after the camera's processing. | We denoise sRGB, the harder and more practical case. |
| ISP (image signal processor) | The camera pipeline that turns RAW into sRGB: demosaicing, white balance, colour correction, gamma, tone mapping, sharpening, compression. | It is why sRGB noise is spatially correlated and colourful. |
| Demosaicing | Rebuilding full RGB from a Bayer sensor, where each pixel records only one colour. | Spreads noise across neighbouring pixels. |
| Gamma / tone curve | A non-linear brightness curve applied for display. | Makes noise signal-dependent in sRGB. |
| Colour space (RGB = red, green, blue; YCbCr = luma Y + blue-difference Cb + red-difference Cr; Lab = lightness L + colour axes a and b) | Different ways to encode colour. YCbCr splits brightness (Y) from colour. | Metrics are in RGB; some papers report Y-only PSNR, which inflates numbers. |
| Patch / crop / tile | A small sub-image cut from a larger one, used to fit memory and add variety. | 512² tiles stored on disk; random 256² or 128² crops fed to training. |
| Block | SIDD's fixed 256² test crops. | 1280 test blocks (40 images × 32). |
| Ground truth (GT) | The clean target image. In SIDD it is estimated by averaging many shots, so it is not perfect. | Explains why PSNR saturates near 40 dB. |

## 2. Degradations and restoration tasks

Image restoration means undoing a degradation: recovering x from y = degrade(x) + noise. It is an inverse problem, usually ill-posed, because many clean images could explain the same corrupted one.

| Term | What it means | In this project |
|---|---|---|
| Noise | Random variation added to the true signal. | The thing being removed. |
| AWGN (additive white Gaussian noise) | Independent Gaussian noise with one fixed σ at every pixel. Easy to simulate, unrealistic. | DRUNet was trained on it. |
| σ (sigma, noise level) | The standard deviation of the noise, often on the 0–255 scale. | Tuned for BM3D (50) and DRUNet. |
| Shot (Poisson) noise | Photon-counting noise that grows with brightness: signal-dependent. | Dominant in bright regions of real photos. |
| Read noise | Electronic noise from the sensor readout, roughly constant. | Dominant in dark regions. |
| Signal-dependent noise | Noise whose strength depends on pixel intensity. | Why real sRGB noise is not AWGN. |
| Spatially correlated noise | Noise that is smeared across neighbours by demosaicing and processing. It looks blotchy, not grainy. | Why classical filters struggle on SIDD. |
| ISO | Sensor gain; the name comes from the ISO (International Organization for Standardization) film-speed scale. Higher ISO means brighter images but more noise. | SIDD covers ISO 50–10,000. |
| Synthetic vs real noise | Simulated noise vs noise captured by a real camera. The difference between them is the "domain gap". | SCUNet was trained on carefully simulated realistic noise. |
| Blur / blur kernel (PSF = point-spread function) | Smearing of the image, modelled as convolution with a kernel, the point-spread function. | Not used; relevant to deblurring. |
| Motion blur / defocus blur | Blur from camera or object motion, or from the lens being out of focus. | The GoPro deblurring alternative. |
| Deconvolution | Inverting a known blur kernel; blind deconvolution means the kernel is unknown too. | Classical approach to deblurring. |
| Task family | Denoising, deblurring, super-resolution (upscaling), JPEG (Joint Photographic Experts Group) artifact removal, deraining, dehazing, low-light enhancement, inpainting (filling holes). | We do denoising; the same networks transfer across tasks. |
| Blind vs non-blind | Blind means the degradation's parameters (σ, kernel) are unknown at test time. | SIDD denoising is blind. |
| All-in-one restoration | A single model trained for many degradations at once. | Ruled out: weak on real noise. |

## 3. Classical (non-learned) methods

Before deep learning, denoisers used hand-designed assumptions about natural images, called priors. They still make good baselines: no training, one or two knobs.

| Term | What it means | In this project |
|---|---|---|
| Prior | An assumption about what clean images look like: smooth, sparse, self-similar. A pretrained network is a learned prior. | The "prior" in transfer learning. |
| Gaussian filter | Replaces each pixel with a weighted average of its neighbours. Removes noise but blurs edges. | Baseline: 33.10 dB at σ = 3 px. |
| Median filter | Takes the median of each neighbourhood. Good for salt-and-pepper noise. | Not used. |
| Bilateral filter | A Gaussian blur that also weights by colour similarity, so edges are preserved. | Not used. |
| Non-local means (NLM) | Averages pixels whose surrounding patches look alike, even if far apart. | Not used; the idea behind BM3D. |
| BM3D (block-matching and 3-D filtering) / CBM3D (colour BM3D) | Block-matching 3-D filtering. Groups similar patches, denoises them jointly in a transform domain, then aggregates. CBM3D is the colour version. The classical gold standard. | Baseline: 24.89 dB with estimated σ, 34.07 dB with tuned σ. |
| Wiener filter | The optimal linear filter, given signal and noise spectra. | Not used. |
| Noise-level estimation | Estimating σ from the noisy image itself, e.g. via wavelet median absolute deviation. | Fails on correlated sRGB noise, which is why BM3D-estimated scores poorly. |
| Hyperparameter tuning on holdout | Choosing knobs like σ on held-out training data, never on the test set. | Done for every baseline. |

## 4. Neural network building blocks

Modern vision networks are stacks of a few reusable layers. Knowing these lets you read any architecture diagram.

| Term | What it means | In this project |
|---|---|---|
| Convolution (conv) | Slides a small learned filter (kernel) over the image. Each output pixel is a weighted sum of its neighbourhood. Shares weights across positions. | The workhorse of NAFNet, U-Net, DRUNet. |
| Kernel size, stride, padding | Filter footprint (3×3), step size (stride 2 halves resolution), and border handling (padding keeps size). | 3×3 convs; 2×2 stride-2 convs to downsample. |
| 1×1 (pointwise) conv | Mixes channels at each pixel without looking at neighbours. | Used heavily in NAFNet blocks. |
| Depthwise conv | One filter per channel; looks at space but not across channels. Much cheaper than a full conv. | NAFNet's 3×3 spatial mixing. |
| Feature map | The intermediate "image" of learned features a layer outputs: H × W × C. | Width 32 → 64 → 128 → 256 → 512 channels in NAFNet. |
| Receptive field | How large a region of the input can influence one output pixel. It grows with depth and downsampling. | Why U-shaped networks see wide context. |
| Activation (ReLU = rectified linear unit; GELU = Gaussian error linear unit) | A non-linearity between layers. ReLU = max(0, x); GELU is a smooth variant. Without activations, a deep net collapses into one linear map. | ReLU in U-Net and DRUNet, GELU in SCUNet. |
| SimpleGate | NAFNet's activation replacement: split channels in two halves and multiply them. | The "NAF" (nonlinear-activation-free) in NAFNet. |
| Normalization (BatchNorm, LayerNorm) | Rescales activations to zero mean and unit variance for stable training. BatchNorm uses batch statistics; LayerNorm uses per-sample statistics, better for small batches. | NAFNet and SCUNet use LayerNorm; the plain U-Net has none. |
| Residual / skip connection | Adds a block's input to its output (x + f(x)), so the block only learns a correction. Makes deep nets trainable. | Everywhere; the whole denoiser often predicts the noise residual. |
| Encoder skip connection | In U-Nets, high-resolution encoder features are passed straight to the decoder. | Restores fine detail after downsampling. |
| Downsampling / upsampling | Reducing or increasing resolution: strided conv or pooling down; transposed conv, PixelShuffle or interpolation up. | NAFNet uses PixelShuffle; SCUNet uses transposed convs. |
| PixelShuffle | Rearranges channels into space (C·4 × H × W → C × 2H × 2W). Cheap, artifact-free upsampling. | NAFNet decoder. |
| Attention / self-attention | Each position computes a weighted average over other positions, with weights from query–key similarity. Captures long-range dependencies. | SCUNet's transformer half. |
| Window attention (Swin = shifted windows) | Attention restricted to local windows (8×8), with windows shifted between layers so information crosses borders. Keeps cost linear in image size. | SCUNet's W and SW blocks. |
| Channel attention | Reweights channels using a global summary of the image. | NAFNet's "simplified channel attention". |
| Scaled dot-product attention (SDPA) | The core attention formula, softmax(QKᵀ/√d)·V; PyTorch has a fused fast kernel for it. | Rewrote SCUNet to use it on MPS. |
| Dropout / DropPath | Randomly zeroing activations or whole residual branches during training, as regularization. | Off in our runs. |

## 5. Architectures

For restoration the dominant shape is the U-Net: an encoder that shrinks the image while widening the features, a bottleneck, and a decoder that grows it back with skip connections. Most named models differ in the block they repeat inside that shape.

| Term | What it means | In this project |
|---|---|---|
| CNN | Convolutional neural network: built mostly from convolutions. | U-Net, DRUNet, NAFNet are CNNs. |
| Encoder–decoder | Compress to a compact representation, then expand back to the output. | The general shape of every model here. |
| U-Net (2015; named for its U shape) | Encoder–decoder with skip connections at every scale. Originally for medical segmentation, now standard for restoration and diffusion. | Our plain ablation model: 35.78 dB. |
| ResNet | Residual network: a network built from residual blocks; the idea that made very deep nets trainable. | DRUNet is a U-Net of ResNet blocks. |
| DnCNN | Denoising CNN: an early (2017) residual CNN denoiser that predicts the noise. | Historical baseline. |
| DRUNet (2021; read as deep residual U-Net) | A residual U-Net denoiser that takes a noise-level map as an extra input. | Pretrained on AWGN; 39.07 dB after fine-tuning. |
| NAFNet (2022; nonlinear activation free network) | A U-Net of simplified blocks: LayerNorm, depthwise conv, SimpleGate, channel attention, no activations. Strong accuracy per FLOP. | Our from-scratch model: 38.86 dB. |
| Transformer | An architecture built on self-attention and MLPs (multi-layer perceptrons), from NLP (natural language processing). | Inside SCUNet and Restormer. |
| ViT (vision transformer) | Cuts an image into patches (e.g. 16×16) and treats them as tokens. | Too coarse for pixel-level restoration as-is. |
| Swin Transformer | Hierarchical transformer with shifted window attention (Swin = shifted windows). | SCUNet's attention design. |
| Restormer (restoration transformer) / SwinIR (Swin for image restoration) / Uformer (U-shaped transformer) | Transformer-based restoration networks from 2021–2022. | Surveyed; not trained. |
| SCUNet (2023; Swin-Conv U-Net) | Swin-Conv U-Net: each block splits channels into a conv path and a window-attention path. Released with weights trained on a realistic synthetic-noise pipeline. | Our transfer source: 35.11 zero-shot, 39.36 dB fine-tuned. |
| CGNet (CascadedGaze network) / KBNet (kernel basis network) | Recent CNNs at the top of the SIDD leaderboard (~40.4 dB). | Cited as the state of the art. |
| Mamba / state-space models (SSM) | Sequence models with linear-time scanning, an attention alternative. MambaIR (Mamba for image restoration) applies them to images. | Ruled out: CUDA-only kernels, no SIDD gain. |
| GAN (generative adversarial network) | A generator trained against a discriminator that tries to tell real images from outputs. Produces sharp, realistic texture but can hallucinate. | SCUNet-real-GAN checkpoint: better LPIPS, lower PSNR. |
| Diffusion model | Generates images by learning to reverse a gradual noising process. DiffBIR (diffusion blind image restoration) and SUPIR (scaling-up image restoration) use them for restoration. | Ruled out: 1–5B params, perceptual rather than fidelity focus. |
| Foundation model | A large model pretrained on broad data and adapted to many tasks (CLIP = contrastive language–image pretraining; DINOv2 = self-distillation with no labels, version 2; SAM = Segment Anything Model). | Discussed; little PSNR gain for denoising. |
| Parameters vs width vs depth | Width is the number of channels; depth the number of blocks; parameters the learned weights, which grow with both. | NAFNet-w32 = width 32, 29.2M params. |

## 6. Training

Training repeats four steps: run a batch forward, measure error with a loss, compute gradients by backpropagation, and let the optimizer nudge the weights.

| Term | What it means | In this project |
|---|---|---|
| Forward pass / backward pass | Computing outputs from inputs, then gradients of the loss with respect to every weight (backpropagation). | Every training step. |
| Loss function | A single number measuring how wrong the output is. | PSNR loss. |
| L2 / MSE (mean squared error) loss | Mean squared error. Optimizes PSNR directly but tends to blur. | PSNR loss is a log of MSE. |
| L1 / Charbonnier loss | Mean absolute error; Charbonnier is a smooth L1. Sharper results than L2. | Available in configs; not ablated. |
| PSNR loss | Minus the PSNR of each image, averaged over the batch. Balances easy and hard images. | NAFNet's loss, used for all runs. |
| Perceptual (VGG/LPIPS) loss (VGG = Oxford's Visual Geometry Group network; LPIPS = learned perceptual image patch similarity) | Compares deep features instead of pixels, which favours natural texture. | Not used. |
| Adversarial loss | The GAN discriminator's penalty. | Used to train SCUNet-real-GAN, not by us. |
| Optimizer (SGD = stochastic gradient descent; Adam = adaptive moment estimation; AdamW = Adam with decoupled weight decay) | The rule for updating weights. Adam adapts the step size per weight; AdamW decouples weight decay. | AdamW everywhere. |
| Learning rate (LR) | The step size of each update. Too high diverges, too low crawls. | 1e-3 scratch, 5e-5 full fine-tune, 5e-4 LoRA. |
| β (betas) | Adam's momentum coefficients for gradient averages. | NAFNet uses (0.9, 0.9). |
| Weight decay | A penalty pulling weights toward zero, as regularization. | 0, as in NAFNet. |
| Warmup | Ramping the LR up from near zero over the first steps, to avoid early instability. | 500 iterations scratch, 100 fine-tune. |
| Cosine schedule | Decays the LR along a half cosine to near zero by the end. | All runs. |
| Batch / batch size | The number of samples per update step. | 8 patches. |
| Iteration vs epoch | An iteration is one update; an epoch is one full pass over the data. With random crops, iterations are the clearer unit. | 12k iterations scratch, 3k fine-tune. |
| Gradient clipping | Capping the gradient norm to prevent explosive updates. | Max-norm 1.0. |
| EMA (exponential moving average) of weights | Keeping an exponential moving average of the weights and evaluating that. A smoother, usually better model. | Decay 0.999. |
| Overfitting / regularization | Memorizing training data instead of generalizing; anything that counters it. | Watched via holdout PSNR. |
| Early stopping / best checkpoint | Keeping the weights that scored best on validation. | best.pt per run. |
| Data augmentation | Random flips, rotations or colour changes to enlarge the data. | Off, as in NAFNet; real noise is not rotation-invariant. |
| Mixed precision (fp16 = 16-bit floating point; bf16 = brain floating point, 16-bit; AMP = automatic mixed precision) | Computing in 16-bit floats for speed and memory, with fp32 where precision matters. GradScaler prevents fp16 gradient underflow. | On for CUDA; MPS gained nothing, so fp32. |
| Checkpoint / resume | Saving model + optimizer state so training can restart. | last.pt every 1k iterations. |
| Seed | The random-number starting point. Different seeds give slightly different models. | One seed per run, a stated limitation. |
| Hyperparameters | Settings chosen by people rather than learned: LR, batch size, iterations, rank. | Stored in YAML configs. |

## 7. Transfer learning

Transfer learning reuses what a model learned on one task or data source to do better on another, usually with less data and compute. It was the central question of this project.

| Term | What it means | In this project |
|---|---|---|
| Pretraining | Training on a large or easy-to-get dataset first. | SCUNet on synthetic realistic noise; DRUNet on AWGN. |
| Pretrained weights / checkpoint | The saved parameters of a trained model. | Downloaded from the KAIR and NAFNet releases. |
| Backbone | The main feature-extracting body of a network, reused across tasks. | SCUNet acts as the backbone being adapted. |
| Fine-tuning | Continuing training of a pretrained model on the target data, usually with a small LR. | Full fine-tune: 39.36 dB in 0.75 h. |
| Zero-shot | Using a model on a task or domain it was never trained on, without any updates. | SCUNet-real zero-shot: 35.11 dB. |
| Freezing layers | Keeping some weights fixed during fine-tuning. | Head+tail: everything but 3.5K weights frozen. |
| Head / tail | The first (input) and last (output) layers of a network. | Training only these recovers +2.7 dB. |
| PEFT (parameter-efficient fine-tuning) | Adapting a model by training a small number of extra or selected parameters. | LoRA and head+tail runs. |
| LoRA (low-rank adaptation) | Adds a trainable low-rank update B·A next to each frozen weight. B starts at zero, so the model is initially unchanged. It can be merged back into the weights afterwards at no cost. | r = 8, 5.7% of params, within 0.02 dB of full. |
| Rank (r) and alpha | LoRA's rank is the size of the bottleneck; alpha scales the update. | r = 8, alpha = 8. |
| Adapter | A small trainable module inserted into a frozen network; LoRA is one kind. | — |
| Domain gap / domain shift | Systematic difference between training and deployment data. | Synthetic noise vs real SIDD noise. |
| Domain adaptation | Methods for closing a domain gap. | Fine-tuning is the simplest form. |
| Data efficiency | How much performance you get per labelled example. | 16 pairs fine-tuned ≈ 144 pairs from scratch. |
| Catastrophic forgetting | Fine-tuning erasing what the model knew before. | Not measured; low LR limits it. |
| Knowledge distillation | Training a small "student" to mimic a large "teacher". | Proposed next step for deployment. |
| Data leakage via pretraining | When pretraining data overlaps the test or training data, which inflates or confounds results. | Why official SIDD checkpoints are only a ceiling. |

## 8. Evaluation

Restoration quality is measured against a clean reference with fidelity metrics (PSNR, SSIM) and perceptual metrics (LPIPS, DISTS). The two families can disagree.

| Term | What it means | In this project |
|---|---|---|
| PSNR (peak signal-to-noise ratio) | 10·log₁₀(255² / MSE), in dB. Higher is better; +1 dB is about 21% less squared error. | Primary metric: 23.66 noisy → 39.36 best. |
| SSIM (structural similarity index measure) | Compares local means, contrast and correlation; 0–1, higher is better. Closer to human judgment than PSNR. | 3-D SIDD convention; per-channel also reported. |
| 2-D vs 3-D SSIM | Per-channel SSIM averaged over RGB vs one SSIM computed jointly over height, width and colour (what MATLAB and SIDD papers report). | 0.915 vs 0.956 for the same model. |
| Full-reference vs no-reference | Full-reference metrics need the clean image; no-reference metrics (NIQE = natural image quality evaluator; MUSIQ = multi-scale image quality transformer; CLIP-IQA = CLIP-based image quality assessment) judge quality alone. | We use full-reference only. |
| LPIPS | Learned perceptual image patch similarity: distance between deep-network features of two images; lower is better. Correlates with human perception. | NAFNet scratch best at 0.195. |
| DISTS | Deep image structure and texture similarity: a perceptual metric combining both. | Reported in results. |
| Perception–distortion trade-off | You cannot generally maximize fidelity (PSNR) and perceptual realism together. | GAN checkpoint: higher realism, lower PSNR. |
| Train / validation / test split | Data to learn from, data to make decisions on, and data scored only once at the end. | SIDD-Small train, holdout, official blocks. |
| Holdout set | A held-out validation split carved from training data. | 16 pairs, stratified by phone. |
| Data leakage | Information from the test set influencing training or model choices. | Avoided: all tuning on holdout. |
| Quantization before scoring | Rounding to 8-bit before measuring, as a real saved image would be. | Explains a 0.09 dB gap to the paper. |
| Baseline | A simple reference method that the model must beat. | Noisy input, Gaussian filter, BM3D. |
| Ablation | Removing or swapping one component to measure its contribution. | U-Net vs NAFNet; LoRA vs full vs head+tail. |
| Bootstrap confidence interval | Resampling the test images with replacement many times to estimate uncertainty. | 95% CIs over 40 images. |
| Paired bootstrap / p-value | Comparing two models on the same images, so per-image difficulty cancels; p is the probability of the gap under no real difference. | p < 0.001 for the headline. |
| Test-time augmentation (TTA) / self-ensemble | Average predictions over flipped and rotated inputs (x8). | +0.03 to +0.06 dB at 8× cost. |
| TLC (test-time local converter) | Fixes the mismatch between training on patches and testing on full images. | Irrelevant: test blocks equal training size. |
| Leaderboard / benchmark | A shared test set with ranked published results. | SIDD validation and benchmark. |

## 9. Compute and hardware

Model size and speed are measured separately: parameters say how much memory the weights take, FLOPs say how much arithmetic one forward pass costs.

| Term | What it means | In this project |
|---|---|---|
| GPU | Graphics processing unit: a massively parallel processor that makes deep learning practical. | Apple M4 Pro GPU. |
| CUDA | Compute Unified Device Architecture, NVIDIA's GPU platform; the default for PyTorch. | Colab notebook path. |
| MPS (Metal Performance Shaders) | PyTorch's backend for Apple-silicon GPUs. Works, but slower than CUDA and with op gaps. | All training. |
| VRAM (video random-access memory) / unified memory | GPU memory that holds weights, activations and gradients. Apple shares one pool between CPU and GPU. | 48 GB unified. |
| Activations memory | Intermediate outputs stored for backprop; usually dominates training memory at high resolution. | Why patches are 256² or 128². |
| Parameters | The number of learned weights. | 1M (LoRA) to 116M (NAFNet-w64). |
| FLOPs / MACs | FLOPs = floating-point operations per forward pass; one multiply-accumulate (MAC) ≈ 2 FLOPs. Reported per 256² input. | NAFNet-w32 16 GMACs; SCUNet 75 GMACs. |
| Throughput vs latency | Samples per second in a batch vs time for one input. | 35 ms/block NAFNet, 79 ms/block SCUNet. |
| Inference | Running a trained model on new data. | Evaluation on 1280 blocks. |
| fp32 / fp16 / bf16 | fp = floating point. 32-bit float; 16-bit float (more precision, less range); 16-bit brain float (fp32's range, less precision). | fp32 on MPS. |
| Fused kernel | Several operations combined into one GPU call, for speed and memory savings. | SDPA in SCUNet. |
| torch.compile | PyTorch's graph compiler for speedups. | Not used; immature on MPS. |
| Colab / Kaggle / Paperspace | Hosted notebooks with free or cheap GPUs (NVIDIA T4 with 16 GB, P100). | Colab notebook shipped. |

## 10. Video analysis

Video adds a time axis: a clip is a stack of frames, and the extra job is to use neighbouring frames and keep results consistent over time. Everything above still applies per frame.

| Term | What it means | Why it matters |
|---|---|---|
| Frame / fps | One image in a sequence; frames per second sets the time between them. | Higher fps gives more redundancy to exploit. |
| Clip / temporal window | A short run of consecutive frames processed together. | Typical models use 5–16 frames. |
| Temporal consistency | Outputs should not flicker or shimmer between frames. | Per-frame denoising often flickers. |
| Optical flow | Per-pixel motion vectors between two frames (e.g. RAFT = recurrent all-pairs field transforms). | Used to align neighbours before fusing. |
| Motion estimation / compensation | Estimating motion, then warping frames to line up. | The core of classical and deep video restoration. |
| Frame alignment (explicit vs implicit) | Warping with flow vs letting deformable convs or attention learn alignment. | BasicVSR (basic video super-resolution) uses flow; EDVR (enhanced deformable video restoration) and VRT (video restoration transformer) use learned alignment. |
| Burst denoising | Merging several quick shots of the same scene, as phone night modes do. | Far stronger than single-image denoising. |
| Recurrent models | Carry a hidden state from frame to frame (BasicVSR++). | Efficient for long videos. |
| 3-D convolution | Convolution over time as well as space (C3D = convolutional 3-D network; I3D = inflated 3-D ConvNet). | Classic video-recognition building block. |
| Video transformers | Attention across space and time (TimeSformer = time-space transformer; ViViT = video vision transformer; VideoMAE = video masked autoencoder; VRT). | Current state of the art in many video tasks. |
| Video codecs and compression artifacts | H.264 (AVC, advanced video coding), H.265 (HEVC, high-efficiency video coding) and AV1 (AOMedia Video 1) compress across frames; artifacts include blocking and ringing. | Restoration often targets compressed video. |
| Keyframe / I-frame (intra-coded frame) | A frame stored whole; others are stored as differences from it. | Where codec artifacts reset. |
| Video super-resolution / interpolation | Upscaling video; synthesizing in-between frames (slow motion). | Common restoration tasks. |
| Object tracking | Following objects across frames (SORT = simple online and realtime tracking; ByteTrack). | Core of surveillance and sports analytics. |
| Action recognition | Classifying what is happening in a clip (Kinetics benchmark). | Main video-classification task. |
| Temporal action localization | Finding when an action starts and ends in a long video. | Video search and moderation. |

## 11. The wider computer vision landscape

Restoration is "low-level" vision: pixels in, pixels out. Most industry CV is "high-level": understanding what is in the image. The vocabulary below comes up in almost any CV interview.

| Term | What it means | Typical metric or model |
|---|---|---|
| Image classification | One label per image. | Top-1/top-5 accuracy on ImageNet; ResNet, ViT, ConvNeXt. |
| Object detection | Boxes plus labels for every object. | mAP; YOLO (you only look once), DETR (detection transformer), Faster R-CNN (region-based CNN). |
| Bounding box / IoU | A rectangle around an object; intersection-over-union measures box overlap. | IoU ≥ 0.5 often counts as a hit. |
| mAP (mean average precision) | Precision–recall area averaged over classes and IoU thresholds. | The standard detection metric on COCO (Common Objects in Context). |
| Non-maximum suppression (NMS) | Removing duplicate overlapping detections. | Post-processing in most detectors. |
| Semantic segmentation | A class label for every pixel. | mIoU (mean IoU); U-Net, DeepLab, SegFormer (segmentation transformer). |
| Instance / panoptic segmentation | Separate masks per object; panoptic combines objects and background. | Mask R-CNN (region-based CNN with masks), Mask2Former. |
| Keypoints / pose estimation | Locating landmarks such as body joints. | OpenPose, ViTPose. |
| Depth estimation | Predicting distance per pixel from one or two images. | Depth Anything, MiDaS. |
| Feature / embedding | A vector summarizing an image or patch, used for search and similarity. | CLIP, DINOv2 embeddings. |
| Self-supervised learning | Learning features without labels, e.g. by matching augmented views or reconstructing masked patches. | DINO (self-distillation with no labels), MAE (masked autoencoder), SimCLR (simple framework for contrastive learning of visual representations). |
| Contrastive learning | Pulling matching pairs together and pushing others apart in embedding space. | CLIP (contrastive language–image pretraining), SimCLR. |
| Vision-language model (VLM) | A model that jointly understands images and text. | CLIP, LLaVA (large language and vision assistant), GPT-4V-class models (GPT = generative pre-trained transformer; V = vision). |
| Segment Anything (SAM = Segment Anything Model) | A promptable segmentation foundation model. | Zero-shot masks from points or boxes. |
| Generative models | Models that create images: GANs, VAEs (variational autoencoders), diffusion. | FID metric; Stable Diffusion. |
| FID | Fréchet Inception Distance (FID): compares feature statistics of real and generated images. | Lower is better; for generation, not restoration. |
| Class imbalance / long tail | A few classes dominate the data. | Mitigated with reweighting or sampling. |
| Precision / recall / F1 | Share of predictions that are correct; share of true items found; their harmonic mean. | Detection and classification metrics. |

## Acronym index

Every acronym used in this document, with its expansion and the section that explains it, is listed in [ACRONYMS.md](ACRONYMS.md).
