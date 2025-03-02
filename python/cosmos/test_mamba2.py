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
    print(y.shape)
