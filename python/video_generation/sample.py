"""Sample from a trained world model.

Two modes:
  - "clip"   : one-shot prediction of the next ``horizon`` frames from a context clip.
  - "rollout": autoregressive rollout beyond the horizon. The predicted frames are
               fed back in as new context (sliding window), so the fixed-size state
               is repeatedly re-summarised -- this is the memory stress test.

Outputs a grid PNG (rows = ground truth, prediction) and, for rollout, per-frame PNGs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from common import add_model_args, build_model, decode_latents, encode_frames, generate, load_vae
from dataset import Bench2DriveDataset
from torchvision.utils import save_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/media/sakoda/samsung_4t/bench2drive")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("results/samples"))
    parser.add_argument("--mode", choices=["clip", "rollout"], default="clip")
    parser.add_argument("--nfe", type=int, default=50)
    parser.add_argument("--rollout_steps", type=int, default=6)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--clip_stride", type=int, default=50)
    parser.add_argument("--max_routes", type=int, default=20)
    parser.add_argument("--use_ema", action="store_true")
    add_model_args(parser)
    return parser.parse_args()


@torch.no_grad()
def rollout(
    model: torch.nn.Module, vae: torch.nn.Module, ctx_latents: torch.Tensor,
    actions_seq: torch.Tensor, horizon: int, nfe: int, steps: int
) -> torch.Tensor:
    """Autoregressively predict ``steps * horizon`` future latents.

    actions_seq: (B, steps * horizon, A). After each block of ``horizon`` predictions,
    the context window slides forward to include them. Returns decoded frames.
    """
    window = ctx_latents.clone()
    all_frames = []
    for step in range(steps):
        acts = actions_seq[:, step * horizon : (step + 1) * horizon]
        pred_latents = generate(model, window, acts, nfe)  # (B, N, C, h, w)
        all_frames.append(decode_latents(vae, pred_latents))
        window = torch.cat([window[:, horizon:], pred_latents], dim=1)  # slide the window
    return torch.cat(all_frames, dim=1)  # (B, steps * horizon, 3, H, W)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    vae = load_vae(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = build_model(args, args.temporal).to(device).eval()
    model.load_state_dict(ckpt["ema" if args.use_ema else "model"])

    total_future = args.horizon * (args.rollout_steps if args.mode == "rollout" else 1)
    dataset = Bench2DriveDataset(
        data_root=args.data_root,
        image_size=args.image_size,
        context_frames=args.context_frames,
        horizon=total_future,
        frame_stride=args.frame_stride,
        clip_stride=args.clip_stride,
        max_routes=args.max_routes,
    )
    idxs = torch.linspace(0, len(dataset) - 1, args.num_samples).long().tolist()
    keys = ("context", "future", "actions")
    batch = {k: torch.stack([dataset[j][k] for j in idxs]) for k in keys}
    context = batch["context"].to(device)
    future = batch["future"].to(device)
    actions = batch["actions"].to(device)
    ctx_latents = encode_frames(vae, context)

    if args.mode == "clip":
        pred = decode_latents(vae, generate(model, ctx_latents, actions, args.nfe))
    else:
        pred = rollout(model, vae, ctx_latents, actions, args.horizon, args.nfe, args.rollout_steps)

    rows = []
    for i in range(context.shape[0]):
        rows.append(torch.cat([context[i], future[i]], dim=0))
        rows.append(torch.cat([context[i], pred[i]], dim=0))
    grid = torch.cat(rows, dim=0)
    out_path = args.out_dir / f"{args.mode}_{args.temporal}.png"
    save_image(grid, out_path, nrow=args.context_frames + total_future)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
