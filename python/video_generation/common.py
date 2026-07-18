"""Shared helpers for the video-generation testbed: VAE, model build, sampler."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from dataset import ACTION_DIM
from diffusers import AutoencoderTiny
from models import WorldModel

if TYPE_CHECKING:
    import argparse

# TAESD latents have std ~0.56 on this data; scale to roughly unit variance so the
# flow-matching noise schedule is well matched.
LATENT_SCALE = 1.8
LATENT_CHANNELS = 4
VAE_DOWNSCALE = 8


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--context_frames", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--frame_stride", type=int, default=3)
    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=384)
    parser.add_argument("--encoder_depth", type=int, default=4)
    parser.add_argument("--predictor_depth", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument(
        "--temporal", choices=["attention", "gru", "gated_deltanet", "ttt"], required=True
    )


def build_model(args: argparse.Namespace) -> WorldModel:
    latent_size = args.image_size // VAE_DOWNSCALE
    return WorldModel(
        latent_channels=LATENT_CHANNELS,
        latent_size=latent_size,
        patch_size=args.patch_size,
        hidden_size=args.hidden_size,
        encoder_depth=args.encoder_depth,
        predictor_depth=args.predictor_depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        context_frames=args.context_frames,
        horizon=args.horizon,
        action_dim=ACTION_DIM,
        freq_embedding_size=256,
        temporal=args.temporal,
    )


def load_vae(device: torch.device) -> AutoencoderTiny:
    vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False
    return vae


@torch.no_grad()
def encode_frames(vae: AutoencoderTiny, frames: torch.Tensor) -> torch.Tensor:
    """frames (B, T, 3, H, W) in [0, 1] -> scaled latents (B, T, C, h, w)."""
    b, t = frames.shape[0], frames.shape[1]
    lat = vae.encode(frames.flatten(0, 1)).latents
    lat = lat.reshape(b, t, *lat.shape[1:])
    return lat * LATENT_SCALE


@torch.no_grad()
def decode_latents(vae: AutoencoderTiny, latents: torch.Tensor) -> torch.Tensor:
    """scaled latents (B, N, C, h, w) -> frames (B, N, 3, H, W) in [0, 1]."""
    b, n = latents.shape[0], latents.shape[1]
    frames = vae.decode(latents.flatten(0, 1) / LATENT_SCALE).sample
    frames = frames.reshape(b, n, *frames.shape[1:])
    return frames.clamp(0, 1)


@torch.no_grad()
def generate(
    model: WorldModel,
    context_latents: torch.Tensor,
    actions: torch.Tensor,
    nfe: int,
) -> torch.Tensor:
    """Sample future latents by integrating the flow ODE from t=1 (noise) to t=0.

    All future frames share the same descending schedule at inference (a special
    case of the per-frame times used in training). Returns (B, N, C, h, w).
    """
    device = context_latents.device
    b, n = actions.shape[0], actions.shape[1]
    c, hw = context_latents.shape[2], context_latents.shape[-1]
    state = model.encode_state(context_latents)
    z = torch.randn(b, n, c, hw, hw, device=device)
    dt = 1.0 / nfe
    for i in range(nfe):
        num_t = 1.0 - i * dt
        t = torch.full((b, n), num_t, device=device)
        v = model.predictor(z, t, state, actions)
        z = z - v * dt
    return z
