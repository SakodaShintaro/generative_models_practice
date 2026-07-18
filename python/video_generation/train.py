"""Round-robin training of all temporal models on Bench2Drive with flow matching.

The four temporal sequence models (ttt / gated_deltanet / attention / gru) are trained
side by side, one epoch each per round:

    ttt e1, gated_deltanet e1, attention e1, gru e1, ttt e2, gated_deltanet e2, ...

so their learning curves stay comparable at every point. Every epoch overwrites a
fixed-name checkpoint (``checkpoints/latest.pt``) and writes a fresh visualization for
that method.

The frozen TAESD VAE encodes context/future frames to latents. Each method's
spatio-temporal encoder compresses the context into a fixed-size state; a flow-matching
DiT then predicts the N future latent frames from the state and the future action
sequence, trained with RoboTTT-style sequence forcing (independent per-frame noise).
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path
from time import time

import torch
from common import (
    METHODS,
    add_shared_model_args,
    build_model,
    decode_latents,
    encode_frames,
    generate,
    load_vae,
)
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
    add_shared_model_args(parser)
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


def run_epoch(
    method: str,
    state: dict,
    vae: torch.nn.Module,
    loader: DataLoader,
    beta: torch.distributions.Beta,
    args: argparse.Namespace,
    device: torch.device,
    epoch: int,
) -> float:
    """Train one method for a single epoch; returns the average loss."""
    model, ema, opt = state["model"], state["ema"], state["opt"]
    model.train()
    running_loss = 0.0
    steps = 0
    for batch in tqdm(loader, desc=f"epoch {epoch}: {method}"):
        running_loss += train_step(model, ema, opt, vae, batch, beta, args, device)
        steps += 1
    return running_loss / steps


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
    pred = decode_latents(vae, generate(ema, ctx_latents, actions, args.sample_nfe))

    rows = []
    for i in range(context.shape[0]):
        rows.append(torch.cat([context[i], future[i]], dim=0))  # (T+N, 3, H, W)
        rows.append(torch.cat([context[i], pred[i]], dim=0))
    grid = torch.cat(rows, dim=0)
    save_image(grid, path, nrow=args.context_frames + args.horizon)


def make_method_state(method: str, args: argparse.Namespace, device: torch.device) -> dict:
    """Build the model / EMA / optimiser / output dirs / metrics writer for one method."""
    model = build_model(args, method).to(device)
    ema = deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0
    )
    run_dir = args.results_dir / method
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    metrics_file = (run_dir / "metrics.csv").open("w", newline="")
    writer = csv.writer(metrics_file)
    writer.writerow(["epoch", "train_loss", "elapsed_sec"])
    metrics_file.flush()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{method}] parameters: {n_params:,}")
    return {
        "model": model,
        "ema": ema,
        "opt": opt,
        "run_dir": run_dir,
        "metrics_file": metrics_file,
        "writer": writer,
    }


def main() -> None:
    assert torch.cuda.is_available(), "Training requires a GPU."
    args = parse_args()
    device = torch.device("cuda")
    torch.manual_seed(0)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    vae = load_vae(device)

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

    states = {method: make_method_state(method, args, device) for method in METHODS}

    # Beta(1.5, 1) noise-level sampler for sequence forcing (RoboTTT Eq. 5).
    beta = torch.distributions.Beta(1.5, 1.0)
    start_time = time()

    # Round-robin: one epoch per method per round.
    for epoch in range(1, args.epochs + 1):
        for method in METHODS:
            state = states[method]
            avg_loss = run_epoch(method, state, vae, loader, beta, args, device, epoch)
            elapsed = int(time() - start_time)
            mm, ss = elapsed // 60, elapsed % 60
            print(f"(epoch {epoch}) [{method}] loss {avg_loss:.4f}  elapsed {mm}m{ss:02d}s")
            state["writer"].writerow([epoch, f"{avg_loss:.6f}", elapsed])
            state["metrics_file"].flush()

            # every epoch: overwrite the fixed-name checkpoint + write a visualization
            save_samples(
                state["ema"], vae, fixed_batch, args,
                state["run_dir"] / "samples" / f"epoch_{epoch:04d}.png",
            )
            torch.save(
                {"model": state["model"].state_dict(), "ema": state["ema"].state_dict(),
                 "args": args, "temporal": method, "epoch": epoch},
                state["run_dir"] / "checkpoints" / "latest.pt",
            )

    for state in states.values():
        state["metrics_file"].close()


if __name__ == "__main__":
    main()
