import torch
from models.config import ModelArgs
from models.mamba2 import Mamba, MambaBlock

if __name__ == "__main__":
    # Test
    params = ModelArgs(
        dim=256,
        n_heads=8,
        n_layers=3,
        vocab_size=64_000,
        max_seq_length=128,
    )
    head_dim = params.dim // params.n_heads
    block = MambaBlock(params)
    x = torch.randn(2, params.max_seq_length, params.dim)
    cos = torch.randn(1, params.max_seq_length, 1, head_dim)
    sin = torch.randn(1, params.max_seq_length, 1, head_dim)

    y_ssd, state_ssd = block(x, cos, sin)
    print(f"{y_ssd.shape=}, {state_ssd.shape=}")
    assert not torch.isnan(y_ssd).any(), f"{y_ssd=}"
    assert not torch.isnan(state_ssd).any(), f"{state_ssd=}"

    y_mat, state_mat = block.simple_matrix(x, cos, sin)
    print(f"{y_mat.shape=}, {state_mat.shape=}")
    assert not torch.isnan(y_mat).any(), f"{y_mat=}"
    assert torch.allclose(y_ssd, y_mat, atol=0.0001), f"{torch.abs(y_ssd - y_mat).max().item()=}"
    assert torch.allclose(state_ssd, state_mat, atol=0.0001), (
        f"{torch.abs(state_ssd - state_mat).max().item()=}"
    )

    y_rec, state_rec = block.simple_recurrsive(x, cos, sin)
    print(f"{y_rec.shape=}, {state_rec.shape=}")
    assert not torch.isnan(y_rec).any(), f"{y_rec=}"
    assert torch.allclose(y_ssd, y_rec, atol=0.0001), f"{torch.abs(y_ssd - y_rec).max().item()=}"
    assert torch.allclose(state_ssd, state_rec, atol=0.0001), (
        f"{torch.abs(state_ssd - state_rec).max().item()=}"
    )

    print("Tests pass.")

    mamba = Mamba(params)
    tokens = torch.randint(0, params.vocab_size, (2, 128))
    y = mamba(tokens)
    print(f"{y.shape=}")
