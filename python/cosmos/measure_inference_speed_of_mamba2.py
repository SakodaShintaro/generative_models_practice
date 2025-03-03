import time

import torch
from models.config import ModelArgs
from models.mamba2 import MambaBlock

if __name__ == "__main__":
    NUM = 100
    print("seq_len,ssd,matrix,recursive")
    for seq_len in [128, 256, 512, 1024, 2048]:
        print(f"{seq_len}", end=",")
        params = ModelArgs(
            dim=256,
            n_heads=8,
            n_layers=3,
            vocab_size=64_000,
            max_seq_length=seq_len,
        )
        head_dim = params.dim // params.n_heads
        block = MambaBlock(params)
        x = torch.randn(2, params.max_seq_length, params.dim)
        cos = torch.randn(1, params.max_seq_length, 1, head_dim)
        sin = torch.randn(1, params.max_seq_length, 1, head_dim)

        with torch.no_grad():
            start = time.time()
            for _ in range(NUM):
                y_ssd, state_ssd = block(x, cos, sin)
            ave = 1e3 * (time.time() - start) / NUM
            print(f"{ave:.2f}", end=",")

            start = time.time()
            for _ in range(100):
                y_mat, state_mat = block.simple_matrix(x, cos, sin)
            ave = 1e3 * (time.time() - start) / 100
            print(f"{ave:.2f}", end=",")

            start = time.time()
            for _ in range(100):
                y_rec, state_rec = block.simple_recurrsive(x, cos, sin)
            ave = 1e3 * (time.time() - start) / 100
            print(f"{ave:.2f}")
