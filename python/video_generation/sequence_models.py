"""Switchable temporal sequence models (temporal mixers).

Each mixer compresses the past along the time axis so that the *last* timestep
carries a fixed-size summary of everything seen so far. This is the object under
test: how much of the surrounding scene can each recurrence remember?

Interface (all mixers are causal):
    forward(x) : (B, T, D) -> (B, T, D)
where ``x`` is already layer-normed by the calling block, and the returned tensor
is the mixer's contribution (a residual is added by the caller).

Implemented mixers:
    - "attention"      : causal multi-head self-attention (full O(T^2) memory)
    - "gru"            : gated recurrent unit (nn.GRU)
    - "gated_deltanet" : Gated DeltaNet, a linear-attention SSM with the delta rule
    - "ttt"            : Test-Time Training with a 2-layer MLP as the fast weight
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class CausalSelfAttention(nn.Module):
    """Causal multi-head self-attention over the time axis, with rotary PE (RoPE).

    Attention is permutation-equivariant, so it needs its own positional signal.
    RoPE is applied to q/k *inside* this module (over the time positions), which
    keeps the positional encoding local to the attention mixer: the recurrent
    mixers (GRU / GatedDeltaNet / TTT) receive no explicit temporal PE, since they
    already encode order through their recurrence. RoPE is relative and causal-safe.
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim % 2 == 0, "RoPE requires an even head dimension"
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def _apply_rope(self, x: torch.Tensor, t: int) -> torch.Tensor:
        # x: (B, H, T, hd); rotate by the time position of each token
        pos = torch.arange(t, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(pos, self.inv_freq)  # (T, hd/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (T, hd)
        cos, sin = emb.cos()[None, None], emb.sin()[None, None]  # (1, 1, T, hd)
        return (x * cos + self._rotate_half(x) * sin).to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, hd)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self._apply_rope(q, t)
        k = self._apply_rope(k, t)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(b, t, d)
        return self.proj(out)


class GRUTemporal(nn.Module):
    """A GRU applied along time. Causal by construction."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        del num_heads  # unused; kept for a uniform factory signature
        self.gru = nn.GRU(dim, dim, num_layers=1, batch_first=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.proj(out)


class GatedDeltaNet(nn.Module):
    """Gated DeltaNet (Yang et al., 2024): linear attention with the delta rule.

    Per head, an associative memory ``S`` (d_k x d_v) is updated at every step:

        S_t = a_t * S_{t-1} (I - b_t k_t k_t^T) + b_t k_t v_t^T
            = a_t * S_{t-1} + b_t k_t (v_t - a_t S_{t-1}^T k_t)^T

    with a scalar decay gate ``a_t`` in (0, 1) and a scalar write strength
    ``b_t`` in (0, 1). Output is ``o_t = S_t^T q_t``. Keys/queries are L2-normed.
    A short recurrent loop over T is fine here (T is small in this testbed).
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        # scalar decay + write gates per head
        self.gate = nn.Linear(dim, 2 * num_heads, bias=True)
        # bias decay towards ~0.9 retention at init so memory persists early on
        nn.init.constant_(self.gate.bias[:num_heads], 2.0)
        self.norm = nn.GroupNorm(num_heads, dim)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h, hd = self.num_heads, self.head_dim
        qkv = self.qkv(x).reshape(b, t, 3, h, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, T, hd)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        gates = torch.sigmoid(self.gate(x)).reshape(b, t, 2, h).permute(0, 3, 1, 2)
        alpha, beta = gates[..., 0], gates[..., 1]  # (B, H, T)

        state = x.new_zeros(b, h, hd, hd)  # keys -> values memory
        outs = []
        for i in range(t):
            k_i = k[:, :, i]  # (B, H, hd)
            v_i = v[:, :, i]
            q_i = q[:, :, i]
            a_i = alpha[:, :, i].unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
            b_i = beta[:, :, i].unsqueeze(-1)  # (B, H, 1)
            state = a_i * state
            # current prediction for this key, then delta-rule correction
            pred = torch.einsum("bhkv,bhk->bhv", state, k_i)  # (B, H, hd)
            delta = b_i * (v_i - pred)  # (B, H, hd)
            state = state + torch.einsum("bhk,bhv->bhkv", k_i, delta)
            o_i = torch.einsum("bhkv,bhk->bhv", state, q_i)  # (B, H, hd)
            outs.append(o_i)
        out = torch.stack(outs, dim=2).reshape(b, t, d)
        out = self.norm(out.reshape(b * t, d)).reshape(b, t, d)
        return self.proj(out)


class TTTMLP(nn.Module):
    """Test-Time Training with a 2-layer MLP fast weight, following RoboTTT.

    RoboTTT (Jiang et al., 2026; arXiv:2607.15275) integrates the TTT mechanism
    of Sun et al. (2024) into a DiT action head. The recurrent state is the
    weights ``W`` of a small MLP (the *fast weights*); the projection matrices
    theta_{Q,K,V} and the initial weights ``W_0`` are the *slow weights* learned
    by the outer loss. This module mirrors that design:

      - fast model  f_W(x) = W2 @ gelu(W1 @ x)  (a two-layer MLP, Sec. 3.4)
      - update  (Eq. 1):  W_t = W_{t-1} - eta * grad_W || f_W(K_t) - V_t ||^2
      - apply   (Eq. 2):  O_t = f_{W_t}(Q_t)           ("update then apply")
      - learnable learning rate ``eta`` (per head), meta-learned init ``W_0``
      - tanh gating (Eq. 3):  out = tanh(alpha) * O_TTT, alpha init ~= 0.001,
        so the TTT contribution starts near zero and is *added* to the attention
        output by the surrounding block's residual connection.

    The inner gradient is written in closed form so the whole recurrence stays a
    single differentiable graph, letting the outer optimizer backprop through the
    fast-weight updates (gradients of gradients) to shape theta and ``W_0``.
    Fast weights are reset to ``W_0`` at the start of every sequence and carried
    across all its timesteps.
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hd = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.norm = nn.GroupNorm(num_heads, dim)
        self.proj = nn.Linear(dim, dim, bias=True)
        # meta-learned fast-weight initialisation W_0 (slow weights)
        self.w1_init = nn.Parameter(torch.randn(num_heads, hd, hd) * (1.0 / math.sqrt(hd)))
        self.w2_init = nn.Parameter(torch.randn(num_heads, hd, hd) * (1.0 / math.sqrt(hd)))
        # learnable inner learning rate eta (per head), softplus-positive; init ~0.1
        # so the compounding inner updates stay stable early in training.
        self.log_lr = nn.Parameter(torch.full((num_heads,), -2.2))
        # tanh gate alpha (Eq. 3), initialized near zero
        self.gate_alpha = nn.Parameter(torch.full((dim,), 0.001))

    @staticmethod
    def _gelu_grad(x: torch.Tensor) -> torch.Tensor:
        # derivative of the (erf) gelu used by F.gelu
        cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        pdf = torch.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
        return cdf + x * pdf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h, hd = self.num_heads, self.head_dim
        qkv = self.qkv(x).reshape(b, t, 3, h, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, T, hd)
        # L2-normalize the key/query views so the fast model's pre-activations stay
        # O(1); without this the compounding inner updates diverge.
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        lr = F.softplus(self.log_lr).view(1, h, 1, 1)  # (1, H, 1, 1)

        w1 = self.w1_init.unsqueeze(0).expand(b, -1, -1, -1)  # (B, H, hd, hd)
        w2 = self.w2_init.unsqueeze(0).expand(b, -1, -1, -1)
        outs = []
        for i in range(t):
            k_i = k[:, :, i]  # (B, H, hd)
            v_i = v[:, :, i]
            q_i = q[:, :, i]
            # inner forward on the key view: f_W(K_t)
            pre = torch.einsum("bhoi,bhi->bho", w1, k_i)  # (B, H, hd)
            hid = F.gelu(pre)
            out_k = torch.einsum("bhoi,bhi->bho", w2, hid)  # (B, H, hd)
            err = out_k - v_i  # (B, H, hd), grad of ||.||^2 up to factor 2
            # closed-form gradient of the fast-weight loss w.r.t. W1, W2
            g_w2 = 2.0 * torch.einsum("bho,bhi->bhoi", err, hid)
            g_hid = 2.0 * torch.einsum("bhoi,bho->bhi", w2, err)
            g_pre = g_hid * self._gelu_grad(pre)
            g_w1 = torch.einsum("bho,bhi->bhoi", g_pre, k_i)
            # one SGD step on the fast weights (Eq. 1)
            w1 = w1 - lr * g_w1
            w2 = w2 - lr * g_w2
            # apply the updated weights to the query view (Eq. 2)
            pre_q = torch.einsum("bhoi,bhi->bho", w1, q_i)
            out_q = torch.einsum("bhoi,bhi->bho", w2, F.gelu(pre_q))
            outs.append(out_q)
        out = torch.stack(outs, dim=2).reshape(b, t, d)
        out = self.norm(out.reshape(b * t, d)).reshape(b, t, d)
        out = self.proj(out)
        # tanh gating (Eq. 3): keep the TTT contribution small at init
        return torch.tanh(self.gate_alpha) * out


TEMPORAL_MIXERS = {
    "attention": CausalSelfAttention,
    "gru": GRUTemporal,
    "gated_deltanet": GatedDeltaNet,
    "ttt": TTTMLP,
}


def build_temporal_mixer(name: str, dim: int, num_heads: int) -> nn.Module:
    if name not in TEMPORAL_MIXERS:
        msg = f"unknown temporal mixer '{name}'; choose from {list(TEMPORAL_MIXERS)}"
        raise ValueError(msg)
    return TEMPORAL_MIXERS[name](dim, num_heads=num_heads)
