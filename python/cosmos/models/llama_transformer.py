# ref. https://github.com/zphang/minimal-llama/blob/main/minimal_llama/model.py

import math

import torch
import torch.nn.functional as F
from config import ModelArgs
from torch import nn

MULTIPLE_OF = 256


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.eps = 1e-6
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ) -> None:
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=MULTIPLE_OF,
        )
        self.attention_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        h = x + self.attention.forward(self.attention_norm(x), cos, sin, mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Attention(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        self.n_local_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        keys = xk[:, :seqlen]
        values = xv[:, :seqlen]

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple:
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def precompute_cos_sin(
    seq_len: int,
    dim: int,
    dtype: torch.dtype,
    device: torch.device,
    base: int = 10000,
) -> tuple:
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1).to(device)
    cos_cached = emb.cos()[None, :, None, :]
    sin_cached = emb.sin()[None, :, None, :]
    return cos_cached, sin_cached


class LlamaTransformer(nn.Module):
    def __init__(self, params: ModelArgs) -> None:
        super().__init__()
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()
        for _ in range(params.n_layers):
            self.layers.append(TransformerBlock(params))

        self.norm = RMSNorm(params.dim)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        self.cos_cached, self.sin_cached = precompute_cos_sin(
            params.max_seq_length,
            params.dim // params.n_heads,
            dtype=self.tok_embeddings.weight.dtype,
            device=self.tok_embeddings.weight.device,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, seq_len = tokens.shape
        h = self.tok_embeddings(tokens)
        cos = self.cos_cached[:, :seq_len].to(h.dtype)
        sin = self.sin_cached[:, :seq_len].to(h.dtype)

        mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=tokens.device)
        mask = torch.triu(mask, diagonal=1).type_as(h)

        for layer in self.layers:
            h = layer(h, cos, sin, mask)
        h = self.norm(h)
        output = self.output(h)
        return output.float()
