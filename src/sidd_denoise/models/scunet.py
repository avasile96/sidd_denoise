"""SCUNet (Zhang et al., MIR 2023, "Practical Blind Image Denoising via Swin-Conv-UNet
and Data Synthesis").

Port of cszn/SCUNet (Apache-2.0), models/network_scunet.py. Parameter names are
unchanged, so `scunet_color_real_psnr.pth` / `scunet_color_real_gan.pth` load strictly.
Those checkpoints were trained *only on synthetic realistic noise* (SCUNet's degradation
pipeline), never on SIDD, which makes them a clean transfer-learning source.

Changes from upstream, none of which affect the math:
  * the relative-position index and the shifted-window mask are cached as
    non-persistent buffers instead of being rebuilt with numpy every forward pass
    (numpy-built index tensors lived on CPU and broke on MPS);
  * attention uses F.scaled_dot_product_attention (same math, fused kernel);
  * no timm / thop dependency and no debug prints.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(1 - self.p)
        return x * keep / (1 - self.p)


class WMSA(nn.Module):
    """(Shifted) window multi-head self-attention from Swin Transformer."""

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super().__init__()
        self.input_dim = input_dim
        self.head_dim = head_dim
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.type = type
        self.embedding_layer = nn.Linear(input_dim, 3 * input_dim, bias=True)
        p = torch.zeros((2 * window_size - 1) * (2 * window_size - 1), self.n_heads)
        nn.init.trunc_normal_(p, std=0.02)
        self.relative_position_params = nn.Parameter(
            p.view(2 * window_size - 1, 2 * window_size - 1, self.n_heads).transpose(1, 2).transpose(0, 1)
        )
        self.linear = nn.Linear(input_dim, output_dim)

        ws = window_size
        cord = torch.stack(torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing="ij"), -1).view(-1, 2)
        relation = cord[:, None, :] - cord[None, :, :] + ws - 1
        self.register_buffer("rel_idx0", relation[..., 0].long(), persistent=False)
        self.register_buffer("rel_idx1", relation[..., 1].long(), persistent=False)
        self._mask_cache: dict[tuple, torch.Tensor] = {}

    def relative_embedding(self):
        return self.relative_position_params[:, self.rel_idx0, self.rel_idx1]

    def generate_mask(self, h, w, p, shift, device):
        key = (h, w, p, shift, str(device))
        if key not in self._mask_cache:
            m = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=device)
            s = p - shift
            m[-1, :, :s, :, s:, :] = True
            m[-1, :, s:, :, :s, :] = True
            m[:, -1, :, :s, :, s:] = True
            m[:, -1, :, s:, :, :s] = True
            self._mask_cache[key] = rearrange(m, "w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)")
        return self._mask_cache[key]

    def forward(self, x):
        ws = self.window_size
        if self.type != "W":
            x = torch.roll(x, shifts=(-(ws // 2), -(ws // 2)), dims=(1, 2))
        x = rearrange(x, "b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c", p1=ws, p2=ws)
        hw, ww = x.size(1), x.size(2)
        x = rearrange(x, "b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c")
        # Same channel layout as upstream ("(three h c)"), but fused attention via SDPA
        # instead of explicit einsums + softmax (identical output to ~1e-6).
        qkv = rearrange(self.embedding_layer(x), "b nw np (three h c) -> three b h nw np c", three=3, c=self.head_dim)
        bias = rearrange(self.relative_embedding(), "h p q -> 1 h 1 p q")
        if self.type != "W":
            bias = bias.masked_fill(self.generate_mask(hw, ww, ws, ws // 2, x.device), float("-inf"))
        # SDPA wants 4-D inputs (the MPS kernel rejects 5-D), so fold heads and windows together.
        b, h, nw, n, c = qkv.shape[1:]
        q, k, v = (t.reshape(b, h * nw, n, c) for t in qkv)
        mask = bias.to(q.dtype).expand(1, h, nw, n, n).reshape(1, h * nw, n, n)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = rearrange(out, "b (h nw) np c -> b nw np (h c)", h=h)
        out = self.linear(out)
        out = rearrange(out, "b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c", w1=hw, p1=ws)
        if self.type != "W":
            out = torch.roll(out, shifts=(ws // 2, ws // 2), dims=(1, 2))
        return out


class Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type="W", input_resolution=None):
        super().__init__()
        self.type = "W" if input_resolution <= window_size else type
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(nn.Linear(input_dim, 4 * input_dim), nn.GELU(), nn.Linear(4 * input_dim, output_dim))

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        return x + self.drop_path(self.mlp(self.ln2(x)))


class ConvTransBlock(nn.Module):
    def __init__(self, conv_dim, trans_dim, head_dim, window_size, drop_path, type="W", input_resolution=None):
        super().__init__()
        self.conv_dim, self.trans_dim = conv_dim, trans_dim
        if input_resolution <= window_size:
            type = "W"
        self.trans_block = Block(trans_dim, trans_dim, head_dim, window_size, drop_path, type, input_resolution)
        self.conv1_1 = nn.Conv2d(conv_dim + trans_dim, conv_dim + trans_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(conv_dim + trans_dim, conv_dim + trans_dim, 1, 1, 0, bias=True)
        self.conv_block = nn.Sequential(
            nn.Conv2d(conv_dim, conv_dim, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(conv_dim, conv_dim, 3, 1, 1, bias=False),
        )

    def forward(self, x):
        conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
        conv_x = self.conv_block(conv_x) + conv_x
        trans_x = self.trans_block(trans_x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x + self.conv1_2(torch.cat((conv_x, trans_x), dim=1))


class SCUNet(nn.Module):
    def __init__(self, in_nc=3, config=(4, 4, 4, 4, 4, 4, 4), dim=64, drop_path_rate=0.0, input_resolution=256):
        super().__init__()
        hd, ws = 32, 8
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]

        def stage(n, d, begin, res):
            return [
                ConvTransBlock(d // 2, d // 2, hd, ws, dpr[i + begin], "W" if not i % 2 else "SW", res) for i in range(n)
            ]

        b = [sum(config[:i]) for i in range(7)]
        r = input_resolution
        self.m_head = nn.Sequential(nn.Conv2d(in_nc, dim, 3, 1, 1, bias=False))
        self.m_down1 = nn.Sequential(*stage(config[0], dim, b[0], r), nn.Conv2d(dim, 2 * dim, 2, 2, 0, bias=False))
        self.m_down2 = nn.Sequential(*stage(config[1], 2 * dim, b[1], r // 2), nn.Conv2d(2 * dim, 4 * dim, 2, 2, 0, bias=False))
        self.m_down3 = nn.Sequential(*stage(config[2], 4 * dim, b[2], r // 4), nn.Conv2d(4 * dim, 8 * dim, 2, 2, 0, bias=False))
        self.m_body = nn.Sequential(*stage(config[3], 8 * dim, b[3], r // 8))
        self.m_up3 = nn.Sequential(nn.ConvTranspose2d(8 * dim, 4 * dim, 2, 2, 0, bias=False), *stage(config[4], 4 * dim, b[4], r // 4))
        self.m_up2 = nn.Sequential(nn.ConvTranspose2d(4 * dim, 2 * dim, 2, 2, 0, bias=False), *stage(config[5], 2 * dim, b[5], r // 2))
        self.m_up1 = nn.Sequential(nn.ConvTranspose2d(2 * dim, dim, 2, 2, 0, bias=False), *stage(config[6], dim, b[6], r))
        self.m_tail = nn.Sequential(nn.Conv2d(dim, in_nc, 3, 1, 1, bias=False))

    def forward(self, x0):
        h, w = x0.shape[-2:]
        pb, pr = int(math.ceil(h / 64) * 64 - h), int(math.ceil(w / 64) * 64 - w)
        x0 = F.pad(x0, (0, pr, 0, pb), mode="replicate")
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)
        return x[..., :h, :w]


def load_official(model: SCUNet, path: str) -> SCUNet:
    sd = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    return model
