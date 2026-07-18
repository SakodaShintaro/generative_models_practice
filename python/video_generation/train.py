"""Train the spatio-temporal world model on Bench2Drive with flow matching.

The frozen TAESD VAE encodes context/future frames to latents. The spatio-temporal
encoder compresses the context into a fixed-size state (its temporal mixer is one of
attention / GRU / GatedDeltaNet / TTT). A flow-matching DiT then predicts the N future
latent frames from the state and the future action sequence, trained with RoboTTT-style
sequence forcing (independent per-frame noise levels).
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path
from time import time

import torch
from common import add_model_args, build_model, decode_latents, encode_frames, generate, load_vae
from dataset import Bench2DriveDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/media/sakoda/samsung_4t/bench2drive")
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--clip_stride", type=int, default=5)
    parser.add_argument("--max_routes", type=int, default=200)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--sample_nfe", type=int, default=25)
    parser.add_argument("--sample_every", type=int, default=5)
    parser.add_argument("--ckpt", type=Path, default=None)
    add_model_args(parser)
    return parser.parse_args()


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    ema_params = dict(ema_model.named_parameters())
    for name, param in model.named_parameters():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def train_step(
    model: torch.nn.Module,
    ema: torch.nn.Module,
    opt: torch.optim.Optimizer,
    vae: torch.nn.Module,
    batch: dict,
    beta: torch.distributions.Beta,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    """One flow-matching optimisation step with sequence forcing; returns the loss."""
    context = batch["context"].to(device, non_blocking=True)
    future = batch["future"].to(device, non_blocking=True)
    actions = batch["actions"].to(device, non_blocking=True)

    ctx_latents = encode_frames(vae, context)
    x0 = encode_frames(vae, future)  # (B, N, C, h, w)

    noise = torch.randn_like(x0)
    # independent per-frame flow time tau = 0.999 * (1 - u), u ~ Beta(1.5, 1)
    u = beta.sample((x0.shape[0], x0.shape[1])).to(device)
    tau = 0.999 * (1.0 - u)  # (B, N)
    tau_b = tau.view(tau.shape[0], tau.shape[1], 1, 1, 1)
    perturbed = (1 - tau_b) * x0 + tau_b * noise
    target = noise - x0

    pred = model(perturbed, tau, ctx_latents, actions)
    loss = (pred - target).pow(2).mean()

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    update_ema(ema, model, args.ema_decay)
    return loss.item()


@torch.no_grad()
def save_samples(
    ema: torch.nn.Module, vae: torch.nn.Module, batch: dict, args: argparse.Namespace, path: Path
) -> None:
    """Grid: for each sample, top row is context+ground-truth-future, bottom row is
    context+predicted-future (same context)."""
    device = next(ema.parameters()).device
    context = batch["context"].to(device)
    future = batch["future"].to(device)
    actions = batch["actions"].to(device)
    ctx_latents = encode_frames(vae, context)
    pred_latents = generate(ema, ctx_latents, actions, args.sample_nfe)
    pred = decode_latents(vae, pred_latents)

    rows = []
    for i in range(context.shape[0]):
        rows.append(torch.cat([context[i], future[i]], dim=0))  # (T+N, 3, H, W)
        rows.append(torch.cat([context[i], pred[i]], dim=0))
    grid = torch.cat(rows, dim=0)
    save_image(grid, path, nrow=args.context_frames + args.horizon)


def main() -> None:
    assert torch.cuda.is_available(), "Training requires a GPU."
    args = parse_args()
    device = torch.device("cuda")
    torch.manual_seed(0)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "samples").mkdir(exist_ok=True)
    (args.results_dir / "checkpoints").mkdir(exist_ok=True)

    vae = load_vae(device)
    model = build_model(args).to(device)
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
    ema = deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad = False
    print(f"[{args.temporal}] parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0
    )

    dataset = Bench2DriveDataset(
        data_root=args.data_root,
        image_size=args.image_size,
        context_frames=args.context_frames,
        horizon=args.horizon,
        frame_stride=args.frame_stride,
        clip_stride=args.clip_stride,
        max_routes=args.max_routes,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    print(f"dataset: {len(dataset):,} clips from {len(dataset.routes)} routes")
    fixed_batch = next(iter(loader))
    fixed_batch = {k: v[: min(4, args.batch_size)] for k, v in fixed_batch.items()}

    metrics_file = (args.results_dir / "metrics.csv").open("w", newline="")
    metrics_writer = csv.writer(metrics_file)
    metrics_writer.writerow(["epoch", "train_loss", "elapsed_sec"])
    metrics_file.flush()

    # Beta(1.5, 1) noise-level sampler for sequence forcing (RoboTTT Eq. 5).
    beta = torch.distributions.Beta(1.5, 1.0)
    start_time = time()
    save_samples(ema, vae, fixed_batch, args, args.results_dir / "samples" / "init.png")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        steps = 0
        for batch in tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}"):
            running_loss += train_step(model, ema, opt, vae, batch, beta, args, device)
            steps += 1

        avg_loss = running_loss / steps
        elapsed = int(time() - start_time)
        mm, ss = elapsed // 60, elapsed % 60
        print(f"(epoch {epoch + 1}) loss {avg_loss:.4f}  elapsed {mm:d}m{ss:02d}s")
        metrics_writer.writerow([epoch + 1, f"{avg_loss:.6f}", elapsed])
        metrics_file.flush()

        if (epoch + 1) % args.sample_every == 0 or epoch + 1 == args.epochs:
            save_samples(
                ema, vae, fixed_batch, args, args.results_dir / "samples" / f"{epoch + 1:04d}.png"
            )
            torch.save(
                {"model": model.state_dict(), "ema": ema.state_dict(), "args": args},
                args.results_dir / "checkpoints" / "latest.pt",
            )

    metrics_file.close()


if __name__ == "__main__":
    main()
