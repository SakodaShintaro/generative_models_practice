from dataclasses import dataclass


@dataclass
class ModelArgs:
    dim: int = 256
    n_layers: int = 32
    n_heads: int = 3
    vocab_size: int = 32000
    max_seq_length: int = 2048


DEBUG_CONFIG = ModelArgs(
    dim=32,
    n_layers=10,
    n_heads=4,
    vocab_size=32000,
)
LLAMA_7B_CONFIG = ModelArgs(
    dim=4096,
    n_layers=32,
    n_heads=32,
    vocab_size=32000,
)
LLAMA_13B_CONFIG = ModelArgs(
    dim=5120,
    n_layers=40,
    n_heads=40,
    vocab_size=32000,
)

LLAMA_CONFIG_DICT = {
    "7B": LLAMA_7B_CONFIG,
    "13B": LLAMA_13B_CONFIG,
    "debug": DEBUG_CONFIG,
}
