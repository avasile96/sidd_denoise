import pytest
import torch

from sidd_denoise.models import build_model, count_params
from sidd_denoise.models.lora import LoRAConv2d, LoRALinear, apply_lora, merge_lora

SMALL = {
    "nafnet": {"width": 8, "middle_blk_num": 1, "enc_blk_nums": [1, 1], "dec_blk_nums": [1, 1]},
    "unet": {"base": 8, "depth": 2},
    "scunet": {"config": [1] * 7, "dim": 64},
    "drunet": {"sigma": 10.0},
}


@pytest.mark.parametrize("name", ["nafnet", "unet", "scunet", "drunet"])
@pytest.mark.parametrize("hw", [(64, 64), (70, 90)])  # includes sizes that need padding
def test_forward_shape(name, hw):
    m = build_model({"name": name, "kwargs": SMALL[name]}).eval()
    x = torch.rand(2, 3, *hw)
    with torch.no_grad():
        y = m(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_lora_zero_init_preserves_function_and_merges():
    torch.manual_seed(0)
    base = build_model({"name": "scunet", "kwargs": SMALL["scunet"]}).eval()
    ref = build_model({"name": "scunet", "kwargs": SMALL["scunet"]}).eval()
    ref.load_state_dict(base.state_dict())
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        y0 = ref(x)
    apply_lora(base, r=4)
    assert any(isinstance(m, (LoRALinear, LoRAConv2d)) for m in base.modules())
    with torch.no_grad():
        assert torch.allclose(base(x), y0, atol=1e-5)  # B = 0 -> unchanged
        for m in base.modules():  # perturb the adapters
            if isinstance(m, (LoRALinear, LoRAConv2d)):
                m.lora_B.normal_(std=0.02)
        y_lora = base(x)
        assert not torch.allclose(y_lora, y0, atol=1e-4)
        merge_lora(base)
        assert not any(isinstance(m, (LoRALinear, LoRAConv2d)) for m in base.modules())
        assert torch.allclose(base(x), y_lora, atol=1e-4)  # merged == unmerged


def test_finetune_modes_trainable_counts():
    full = build_model({"name": "scunet", "kwargs": SMALL["scunet"], "finetune": "full"})
    lora = build_model({"name": "scunet", "kwargs": SMALL["scunet"], "finetune": "lora", "lora_r": 4})
    ht = build_model({"name": "scunet", "kwargs": SMALL["scunet"], "finetune": "head_tail"})
    n_full = count_params(full, trainable_only=True)
    assert 0 < count_params(ht, trainable_only=True) < count_params(lora, trainable_only=True) < n_full
    trainable = [n for n, p in ht.named_parameters() if p.requires_grad]
    assert trainable and all(n.startswith(("m_head", "m_tail")) for n in trainable)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")
def test_scunet_runs_on_mps_train_and_eval():
    m = build_model({"name": "scunet", "kwargs": SMALL["scunet"]}).to("mps")
    x = torch.rand(2, 3, 64, 64, device="mps")
    m.train()
    m(x).mean().backward()
    m.eval()
    with torch.no_grad():
        assert m(x).shape == x.shape


def test_scunet_sdpa_attention_matches_upstream_einsum():
    """The fused-SDPA WMSA must equal upstream's explicit einsum/softmax formulation."""
    from einops import rearrange

    from sidd_denoise.models.scunet import WMSA

    torch.manual_seed(0)
    for typ in ("W", "SW"):
        blk = WMSA(64, 64, 32, 8, typ).eval()
        x = torch.rand(2, 32, 24, 64)
        ws, hw, ww = 8, 4, 3
        with torch.no_grad():
            y = torch.roll(x, (-4, -4), (1, 2)) if typ == "SW" else x
            y = rearrange(y, "b (w1 p1) (w2 p2) c -> b (w1 w2) (p1 p2) c", p1=ws, p2=ws)
            q, k, v = rearrange(blk.embedding_layer(y), "b nw np (threeh c) -> threeh b nw np c", c=32).chunk(3, 0)
            sim = torch.einsum("hbwpc,hbwqc->hbwpq", q, k) * 32**-0.5
            sim = sim + rearrange(blk.relative_embedding(), "h p q -> h 1 1 p q")
            if typ == "SW":
                sim = sim.masked_fill(blk.generate_mask(hw, ww, ws, 4, x.device), float("-inf"))
            o = rearrange(torch.einsum("hbwij,hbwjc->hbwic", sim.softmax(-1), v), "h b w p c -> b w p (h c)")
            o = rearrange(blk.linear(o), "b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c", w1=hw, p1=ws)
            if typ == "SW":
                o = torch.roll(o, (4, 4), (1, 2))
            assert torch.allclose(blk(x), o, atol=1e-5)
