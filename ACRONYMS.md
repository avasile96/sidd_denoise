# Computer Vision & Deep Learning Acronyms

Oct 1, 2026

About 100 acronyms from the SIDD denoising project and the wider computer vision field, alphabetically, each with its expansion and a one-line meaning.

## Acronyms A–Z

| Acronym | Stands for | What it is |
|---|---|---|
| Adam / AdamW | Adaptive moment estimation / Adam with decoupled weight decay | The default deep-learning optimizer; AdamW applies weight decay correctly. |
| AMP | Automatic mixed precision | Runs parts of training in 16-bit floats for speed. |
| AV1 | AOMedia Video 1 | Royalty-free video codec from the Alliance for Open Media. |
| AVC (H.264) | Advanced Video Coding | The most widespread video codec. |
| AWGN | Additive white Gaussian noise | Simple synthetic noise: independent, one fixed σ. |
| BasicVSR | Basic video super-resolution | Recurrent video restoration model that aligns frames with optical flow. |
| bf16 | Brain floating point, 16-bit | 16-bit format with fp32's range; from Google Brain. |
| BM3D / CBM3D | Block-matching and 3-D filtering / colour BM3D | The classical gold-standard denoiser. |
| C3D | Convolutional 3-D network | Early video network using 3-D convolutions. |
| CGNet | CascadedGaze network | CNN at the top of the SIDD leaderboard (~40.4 dB). |
| CI | Confidence interval | Range likely to contain the true value. |
| CLIP | Contrastive language–image pretraining | Model that embeds images and text in one space. |
| CLIP-IQA | CLIP-based image quality assessment | No-reference quality metric built on CLIP. |
| CNN | Convolutional neural network | Network built mostly from convolutions. |
| COCO | Common Objects in Context | Standard dataset and benchmark for detection and segmentation. |
| CUDA | Compute Unified Device Architecture | NVIDIA's GPU programming platform. |
| CV | Computer vision | Field of making computers understand images and video. |
| dB | Decibel | Log-scale unit used for PSNR. |
| DETR | Detection transformer | Transformer-based object detector. |
| DiffBIR | Diffusion blind image restoration | Diffusion-based restoration model. |
| DINO / DINOv2 | Self-distillation with no labels (version 2) | Self-supervised vision backbone. |
| DISTS | Deep image structure and texture similarity | Perceptual full-reference metric; lower is better. |
| DnCNN | Denoising convolutional neural network | Early (2017) residual CNN denoiser. |
| DRUNet | Deep residual U-Net (common reading; not spelled out in the paper) | Residual U-Net denoiser with a noise-level input. |
| EDVR | Enhanced deformable video restoration | Video restoration model with learned alignment. |
| EMA | Exponential moving average | Smoothed copy of the weights used for evaluation. |
| F1 | (score name, not an abbreviation) | Harmonic mean of precision and recall. |
| FID | Fréchet Inception Distance | Distance between real and generated image statistics. |
| FLOPs | Floating-point operations | Arithmetic cost of one forward pass. |
| fp16 / fp32 | 16-bit / 32-bit floating point | Number precisions for training and inference. |
| fps | Frames per second | Video frame rate. |
| GAN | Generative adversarial network | Generator trained against a discriminator. |
| GB | Gigabyte | 10⁹ bytes; GPU memory unit. |
| GELU | Gaussian error linear unit | Smooth activation function used in transformers. |
| GMACs | Giga multiply-accumulate operations | Billions of MACs; a model-cost unit. |
| GPT | Generative pre-trained transformer | Family of large language models. |
| GPU | Graphics processing unit | Parallel processor that runs deep learning. |
| GT | Ground truth | The clean target image. |
| HEVC (H.265) | High Efficiency Video Coding | Successor to H.264; about half the bitrate. |
| I3D | Inflated 3-D ConvNet | Video network made by inflating 2-D filters to 3-D. |
| I-frame | Intra-coded frame | Video frame stored without reference to others. |
| IoU | Intersection over union | Overlap between two boxes or masks. |
| ISO | International Organization for Standardization | Its film-speed scale names camera sensor gain. |
| ISP | Image signal processor | Camera pipeline that turns RAW into sRGB. |
| JPEG | Joint Photographic Experts Group | Lossy image format, and the committee that made it. |
| KBNet | Kernel basis network | Top SIDD CNN. |
| L1 / L2 | (norm names, not abbreviations) | Absolute-error / squared-error distance. |
| Lab (CIELAB) | Lightness L plus colour axes a and b | Perceptually uniform colour space. |
| LLaVA | Large language and vision assistant | Open vision-language model. |
| LoRA | Low-rank adaptation | Fine-tunes a small low-rank update next to frozen weights. |
| LPIPS | Learned perceptual image patch similarity | Perceptual metric from deep features; lower is better. |
| LR | Learning rate | Optimizer step size. |
| MAC | Multiply-accumulate operation | One multiply plus one add, about 2 FLOPs. |
| MAE | Masked autoencoder | Self-supervised pretraining by reconstructing hidden patches. |
| MambaIR | Mamba for image restoration | State-space-model restoration network. |
| mAP | Mean average precision | Standard object-detection metric. |
| mIoU | Mean intersection over union | Standard segmentation metric. |
| MLP | Multi-layer perceptron | Stack of fully connected layers. |
| MP | Megapixel | One million pixels. |
| MPS | Metal Performance Shaders | PyTorch's backend for Apple-silicon GPUs. |
| MSE | Mean squared error | Average squared pixel difference; basis of PSNR. |
| MUSIQ | Multi-scale image quality transformer | No-reference quality metric. |
| NAFNet | Nonlinear activation free network | Efficient U-Net restoration model. |
| NCHW / NHWC | Batch N, channels C, height H, width W | Tensor axis orders. |
| NIQE | Natural image quality evaluator | Classical no-reference quality metric. |
| NLM | Non-local means | Denoiser averaging similar patches. |
| NLP | Natural language processing | Text-focused AI; where transformers began. |
| NMS | Non-maximum suppression | Removes duplicate detections. |
| PEFT | Parameter-efficient fine-tuning | Adapting a model by training few parameters. |
| PSF | Point-spread function | The blur kernel of an imaging system. |
| PSNR | Peak signal-to-noise ratio | Primary restoration fidelity metric, in dB. |
| R-CNN | Region-based convolutional neural network | Detector family (Faster R-CNN, Mask R-CNN). |
| RAFT | Recurrent all-pairs field transforms | Strong optical-flow network. |
| RAM / VRAM | Random-access memory / video RAM | System memory / GPU memory. |
| ReLU | Rectified linear unit | Activation max(0, x). |
| ResNet | Residual network | Network of residual blocks. |
| RGB / sRGB | Red, green, blue / standard RGB | Colour channels / the standard display colour space. |
| SAM | Segment Anything Model | Promptable segmentation foundation model. |
| SCUNet | Swin-Conv U-Net | Conv + window-attention U-Net denoiser. |
| SDPA | Scaled dot-product attention | The core attention computation. |
| SegFormer | Segmentation transformer | Transformer for semantic segmentation. |
| SGD | Stochastic gradient descent | The basic optimizer. |
| SIDD | Smartphone Image Denoising Dataset | Real-noise benchmark used in the project. |
| SimCLR | Simple framework for contrastive learning of visual representations | Self-supervised contrastive method. |
| SORT | Simple online and realtime tracking | Classic multi-object tracker. |
| SR | Super-resolution | Upscaling an image while adding detail. |
| SSIM | Structural similarity index measure | Structure-aware fidelity metric, 0–1. |
| SSM | State-space model | Sequence model behind Mamba. |
| SUPIR | Scaling-up image restoration | Large diffusion-based restoration model. |
| Swin / SwinIR | Shifted windows / Swin for image restoration | Window-attention transformer / its restoration variant. |
| TimeSformer | Time-space transformer | Video transformer. |
| TLC | Test-time local converter | Fixes patch-vs-full-image mismatch at inference. |
| TTA | Test-time augmentation | Averaging predictions over flipped and rotated inputs. |
| uint8 | Unsigned 8-bit integer | Pixel values 0–255. |
| Uformer | U-shaped transformer | Transformer restoration U-Net. |
| VAE | Variational autoencoder | Probabilistic encoder–decoder generative model. |
| VGG | Visual Geometry Group (Oxford) | Classic CNN, used for perceptual losses. |
| VideoMAE | Video masked autoencoder | Self-supervised video pretraining. |
| ViT | Vision transformer | Transformer on image patches. |
| ViViT | Video vision transformer | ViT extended to video. |
| VLM | Vision-language model | Model that jointly understands images and text. |
| VRT | Video restoration transformer | Transformer for video restoration. |
| YCbCr | Luma Y, blue-difference Cb, red-difference Cr | Colour space separating brightness from colour. |
| YOLO | You only look once | Fast single-stage object detector. |
