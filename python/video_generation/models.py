"""Spatio-temporal world model for the video-generation testbed.

Pipeline (see instruction.md):

    context frames  (B, T, 3, H, W)
        -> frozen TAESD VAE            -> latents (B, T, C, h, w)      [done in train.py]
        -> STEncoder                   -> state   (B, S, D)           (last timestep)
    state + future actions (B, N, A)
        -> FlowPredictor (flow matching) -> future latents (B, N, C, h, w)
        -> frozen TAESD VAE decode       -> future frames  (B, N, 3, H, W)

Spatial mixing is attention (within a frame); temporal mixing is one of the
switchable sequence models in ``sequence_models.py``. The predictor is a DiT that
denoises all N future frames jointly, conditioned on the state (as prefix tokens)
and a per-frame action embedding.
"""

from __future__ import annotations

import numpy as np
import torch
from sequence_models import build_temporal_mixer
from torch import nn


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def sincos_pos_param(dim: int, grid: int) -> nn.Parameter:
    """A frozen 2D sin-cos positional embedding as an (1, grid*grid, dim) parameter."""
    pos = get_2d_sincos_pos_embed(dim, grid)
    return nn.Parameter(torch.from_numpy(pos).float().unsqueeze(0), requires_grad=False)


#################################################################################
#                          Embedding / helper modules                           #
#################################################################################


class TimestepEmbedder(nn.Module):
    """Embeds a scalar diffusion timestep into a vector."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -np.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32) / half,
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class Mlp(nn.Module):
    """Two-layer MLP with GELU, as used in transformer blocks."""

    def __init__(self, dim: int, mlp_ratio: float):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden, bias=True)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class SpatialAttention(nn.Module):
    """Bidirectional multi-head self-attention over the spatial tokens of a frame."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        qkv = self.qkv(x).reshape(b, s, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, s, d)
        return self.proj(out)


#################################################################################
#                          Spatio-temporal encoder                              #
#################################################################################


class STEncoderBlock(nn.Module):
    """One spatio-temporal block: spatial attention, then a temporal mixer, then MLP.

    Operates on features of shape (B, T, S, D): spatial attention mixes the S tokens
    within each frame, the temporal mixer mixes the T timesteps at each spatial
    position causally, and the MLP mixes channels. All are pre-norm residuals.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, temporal: str):
        super().__init__()
        self.norm_s = nn.LayerNorm(dim)
        self.spatial = SpatialAttention(dim, num_heads)
        self.norm_t = nn.LayerNorm(dim)
        self.temporal = build_temporal_mixer(temporal, dim, num_heads)
        self.norm_m = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, s, d = x.shape
        # spatial attention within each frame
        xs = self.norm_s(x).reshape(b * t, s, d)
        x = x + self.spatial(xs).reshape(b, t, s, d)
        # temporal mixing across frames at each spatial position (causal)
        xt = self.norm_t(x).permute(0, 2, 1, 3).reshape(b * s, t, d)
        x = x + self.temporal(xt).reshape(b, s, t, d).permute(0, 2, 1, 3)
        # channel MLP
        x = x + self.mlp(self.norm_m(x))
        return x


class STEncoder(nn.Module):
    """Encodes a sequence of latent frames into a fixed-size state (last timestep).

    Input latents (B, T, C, h, w) are patch-embedded into S = (h/p)(w/p) tokens per
    frame, run through ``depth`` spatio-temporal blocks, and the final timestep's
    S tokens are returned as the state. The state size is independent of T, so it is
    a genuine fixed-length compression of the past.
    """

    def __init__(
        self,
        latent_channels: int,
        latent_size: int,
        patch_size: int,
        hidden_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        temporal: str,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.grid = latent_size // patch_size
        self.num_tokens = self.grid * self.grid
        self.patch_embed = nn.Conv2d(
            latent_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )
        self.spatial_pos = sincos_pos_param(hidden_size, self.grid)
        # No temporal positional embedding here: the causal-attention mixer applies its
        # own RoPE internally, and the recurrent mixers encode order via their recurrence.
        self.blocks = nn.ModuleList(
            [STEncoderBlock(hidden_size, num_heads, mlp_ratio, temporal) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        b, t = latents.shape[0], latents.shape[1]
        x = latents.flatten(0, 1)  # (B*T, C, h, w)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B*T, S, D)
        x = x + self.spatial_pos
        x = x.reshape(b, t, self.num_tokens, -1)
        for block in self.blocks:
            x = block(x)
        state = self.norm(x[:, -1])  # (B, S, D): last timestep is the compressed state
        return state


#################################################################################
#                     Flow-matching future-frame predictor                      #
#################################################################################


class DiTBlock(nn.Module):
    """DiT block with adaLN-Zero conditioning (bidirectional attention)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = SpatialAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(dim, mlp_ratio)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c
        ).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """Final adaLN layer projecting tokens back to patch pixels."""

    def __init__(self, dim: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class FlowPredictor(nn.Module):
    """Flow-matching DiT that denoises N future latent frames jointly.

    The noised future latents are patch-embedded into N x S tokens. Conditioning:
      - the state (B, S, D) is prepended as S context tokens (attended to, never
        denoised) so spatial memory of the scene is available everywhere;
      - a per-frame action embedding is added to that frame's tokens;
      - spatial + temporal (frame-index) positional embeddings;
      - the diffusion time t drives adaLN-Zero modulation.
    The network outputs the flow-matching velocity for each future latent.
    """

    def __init__(
        self,
        latent_channels: int,
        latent_size: int,
        patch_size: int,
        hidden_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        horizon: int,
        action_dim: int,
        freq_embedding_size: int,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.patch_size = patch_size
        self.grid = latent_size // patch_size
        self.num_tokens = self.grid * self.grid
        self.horizon = horizon

        self.x_embed = nn.Conv2d(
            latent_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )
        self.t_embed = TimestepEmbedder(hidden_size, freq_embedding_size)
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.spatial_pos = sincos_pos_param(hidden_size, self.grid)
        self.future_pos = nn.Parameter(torch.zeros(1, horizon, 1, hidden_size))
        self.state_pos = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.normal_(self.future_pos, std=0.02)
        nn.init.normal_(self.state_pos, std=0.02)

        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, latent_channels)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        w = self.x_embed.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.constant_(self.x_embed.bias, 0)
        nn.init.normal_(self.t_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embed.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*N, S, p*p*C) -> (B*N, C, h, w)
        c, p, g = self.latent_channels, self.patch_size, self.grid
        x = x.reshape(x.shape[0], g, g, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, g * p, g * p)

    def forward(
        self, noised: torch.Tensor, t: torch.Tensor, state: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """noised: (B, N, C, h, w); t: (B, N) per-frame flow time; state: (B, S, D);
        actions: (B, N, A). Per-frame times implement RoboTTT's sequence forcing:
        each future frame carries its own independent noise level.
        """
        b, n = noised.shape[0], noised.shape[1]
        s = self.num_tokens
        x = self.x_embed(noised.flatten(0, 1)).flatten(2).transpose(1, 2)  # (B*N, S, D)
        x = x.reshape(b, n, s, -1) + self.spatial_pos.unsqueeze(1) + self.future_pos[:, :n]
        # per-frame time and action conditioning, added to every spatial token of that frame
        te = self.t_embed(t.reshape(-1)).reshape(b, n, -1)  # (B, N, D)
        x = x + self.action_embed(actions).unsqueeze(2) + te.unsqueeze(2)
        x = x.reshape(b, n * s, -1)
        # prepend state tokens as read-only context
        ctx = state + self.spatial_pos + self.state_pos
        x = torch.cat([ctx, x], dim=1)  # (B, S + N*S, D)

        c_global = te.mean(dim=1)  # (B, D): global adaLN modulation
        for block in self.blocks:
            x = block(x, c_global)
        x = x[:, s:]  # drop the state context tokens
        x = self.final_layer(x.reshape(b * n, s, -1), te.reshape(b * n, -1))
        side = self.grid * self.patch_size
        return self.unpatchify(x).reshape(b, n, self.latent_channels, side, side)


#################################################################################
#                                World model                                    #
#################################################################################


class WorldModel(nn.Module):
    """Bundles the spatio-temporal encoder and the flow-matching predictor."""

    def __init__(
        self,
        latent_channels: int,
        latent_size: int,
        patch_size: int,
        hidden_size: int,
        encoder_depth: int,
        predictor_depth: int,
        num_heads: int,
        mlp_ratio: float,
        horizon: int,
        action_dim: int,
        freq_embedding_size: int,
        temporal: str,
    ):
        super().__init__()
        self.encoder = STEncoder(
            latent_channels=latent_channels,
            latent_size=latent_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            depth=encoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            temporal=temporal,
        )
        self.predictor = FlowPredictor(
            latent_channels=latent_channels,
            latent_size=latent_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            depth=predictor_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            horizon=horizon,
            action_dim=action_dim,
            freq_embedding_size=freq_embedding_size,
        )

    def encode_state(self, context_latents: torch.Tensor) -> torch.Tensor:
        return self.encoder(context_latents)

    def forward(
        self,
        noised: torch.Tensor,
        t: torch.Tensor,
        context_latents: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        state = self.encoder(context_latents)
        return self.predictor(noised, t, state, actions)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    return get_2d_sincos_pos_embed_from_grid(embed_dim, grid)


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)
