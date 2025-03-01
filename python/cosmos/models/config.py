from dataclasses import dataclass


@dataclass
class ModelArgs:
    dim: int
    n_layers: int
    n_heads: int
    vocab_size: int
    max_seq_length: int
