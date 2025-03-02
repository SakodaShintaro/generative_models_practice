import torch
from models.config import ModelArgs
from models.mamba2 import MambaBlock

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
    y = block(x, cos, sin)

    y_m = block.simple_matrix(x, cos, sin)
    print(f"{y.shape=}, {y_m.shape=}")
    assert torch.allclose(y, y_m, atol=0.0001), f"{torch.abs(y - y_m).max().item()=}"

    y_r = block.simple_recurrsive(x, cos, sin)
    print(f"{y.shape=}, {y_r.shape=}")
    assert torch.allclose(y, y_r)
    print("Tests pass.")
