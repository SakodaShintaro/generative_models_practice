# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A minimal training script for DiT."""

import argparse
import logging
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from time import time

import torch
from diffusers.models import AutoencoderKL
from models import DiT, MiT
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import STL10
from torchvision.utils import save_image
from tqdm import tqdm

# the first flag below was False when we tested this script but True makes training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
IMAGE_SIZE = 64
# MODEL_TYPE = "flow_matching"
MODEL_TYPE = "mean_flow"

#################################################################################
#                             Training Helper Functions                         #
#################################################################################


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--results_dir", type=Path, default="results")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--nfe", type=int, default=20, help="Number of Function Evaluations")
    return parser.parse_args()


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float = 0.9999) -> None:
    """Step the EMA model towards the current model."""
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model: torch.nn.Module, flag: bool) -> None:
    """Set requires_grad flag for all parameters in a model."""
    for p in model.parameters():
        p.requires_grad = flag


@torch.no_grad()
def sample_images(
    model: torch.nn.Module,
    vae: AutoencoderKL,
    args: argparse.Namespace,
) -> torch.Tensor:
    latent_size = IMAGE_SIZE // 8
    num_classes = args.num_classes
    device = model.parameters().__next__().device

    # Labels to condition the model with (feel free to change):
    class_labels = [num_classes for _ in range(20)]

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    if MODEL_TYPE == "flow_matching":
        sample_n = args.nfe
        dt = 1.0 / sample_n
        for i in range(sample_n):
            num_t = 1 - (i / sample_n * (1 - eps) + eps)
            t = torch.ones(n, device=device) * num_t
            pred = model.forward(z, t, y)
            z = z.detach().clone() - pred * dt

    elif MODEL_TYPE == "mean_flow":
        t = torch.ones(n, device=device)
        r = torch.zeros_like(t)
        pred = model.forward(z, t, r, y)
        z = z.detach().clone() - pred

    z = torch.split(z, n, dim=0)[0]
    return vae.decode(z / 0.18215).sample


def save_ckpt(
    model: torch.nn.Module,
    ema: torch.nn.Module,
    opt: torch.optim.Optimizer,
    args: argparse.Namespace,
    train_steps: int,
) -> None:
    results_dir = args.results_dir
    checkpoint_dir = results_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = f"{checkpoint_dir}/latest.pt"
    checkpoint = {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "opt": opt.state_dict(),
        "args": args,
    }
    torch.save(checkpoint, checkpoint_path)
    model.eval()
    samples = sample_images(model, vae, args)
    sample_dir = results_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    save_image(
        samples,
        sample_dir / f"{train_steps:08d}.png",
        nrow=4,
        normalize=True,
        value_range=(-1, 1),
    )


#################################################################################
#                                  Training Loop                                #
#################################################################################


if __name__ == "__main__":
    """Trains a new DiT model."""
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    args = parse_args()

    device = 0
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting seed={seed}.")

    # Setup an experiment folder:
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{results_dir}/log.txt")],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Experiment directory created at {results_dir}")

    # Create model:
    assert IMAGE_SIZE % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = IMAGE_SIZE // 8
    ckpt = torch.load(args.ckpt) if args.ckpt is not None else None
    model_class = DiT if MODEL_TYPE == "flow_matching" else MiT
    model = model_class(
        depth=12,
        hidden_size=384,
        patch_size=2,
        num_heads=6,
        input_size=latent_size,
        num_classes=args.num_classes,
        learn_sigma=False,
    )
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    if ckpt is not None:
        ema.load_state_dict(ckpt["ema"])
    requires_grad(ema, flag=False)
    model = model.to(device)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    logger.info(f"{MODEL_TYPE} Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0,
    )
    if ckpt is not None:
        opt.load_state_dict(ckpt["opt"])

    # Setup data:
    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ],
    )

    dataset = STL10(args.data_path, split="unlabeled", transform=transform, download=True)

    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    start_time = time()

    eps = 0.001
    save_ckpt(model, ema, opt, args, 0)

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        log_steps = 0
        running_loss = 0
        running_fm_mse = 0
        running_du_dt = 0
        for x, y in tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            x = x.to(device)
            y = y.to(device)
            # -1であるラベルはclass_numに変換する
            y = torch.where(y == -1, torch.tensor(args.num_classes, device=device), y)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            noise = torch.randn_like(x)
            # Logit-normal time sampling (MeanFlow paper recommendation): biases t toward
            # smaller values where modelling is harder. μ=-0.4, σ=1.0 from the reference.
            normal_samples = torch.randn(x.shape[0], 2, device=device) * 1.0 + (-0.4)
            time_samples = torch.sigmoid(normal_samples)
            r, t = time_samples.min(dim=1).values, time_samples.max(dim=1).values
            # r=t for 25% of samples (paper-recommended; was 50%).
            fm_mask = torch.rand(x.shape[0], device=device) < 0.25
            r = torch.where(fm_mask, t, r)
            t = t.view(-1, 1, 1, 1)
            r = r.view(-1, 1, 1, 1)
            perturbed_data = (1 - t) * x + t * noise
            t = t.squeeze()
            r = r.squeeze()
            v = noise - x

            if MODEL_TYPE == "flow_matching":
                out = model(perturbed_data, t, y)
                loss = (out - v).pow(2).mean()
                fm_mse = loss.detach()
                du_dt_mag = torch.tensor(0.0, device=device)
            elif MODEL_TYPE == "mean_flow":

                def f(x, t_, r=r, y=y):
                    return model(x, t_, r, y)

                with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                    u, du_dt = torch.func.jvp(f, (perturbed_data, t), (v, torch.ones_like(t)))
                t = t.view(-1, 1, 1, 1)
                r = r.view(-1, 1, 1, 1)
                u_target = v - (t - r) * du_dt
                # Adaptive loss weighting from MeanFlow paper:
                # without it (t-r)·du/dt self-amplifies and training diverges.
                delta_sq = (u - u_target.detach()).pow(2)
                w = 1.0 / (delta_sq.detach().mean(dim=(1, 2, 3), keepdim=True) + 1e-3)
                loss = (w * delta_sq).mean()
                # Tracking-only: restrict to the r=t subset, where u_target = v exactly
                # (pure flow-matching regime). For r<t the optimum u_MF ≠ v, so the
                # full-batch (u-v)² has no reason to decrease.
                per_sample_uv_sq = (u - v).detach().pow(2).mean(dim=(1, 2, 3))
                fm_mse = per_sample_uv_sq[fm_mask].mean()
                du_dt_mag = du_dt.detach().abs().mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            update_ema(ema, model)

            # Log loss values:
            running_loss += loss.item()
            running_fm_mse += fm_mse.item()
            running_du_dt += du_dt_mag.item()
            log_steps += 1

        # Measure training speed:
        torch.cuda.synchronize()
        end_time = time()
        elapsed_sec = int(end_time - start_time)
        elapsed_min = elapsed_sec // 60
        elapsed_sec = elapsed_sec % 60
        # Reduce loss history over all processes:
        avg_loss = running_loss / log_steps
        avg_fm_mse = running_fm_mse / log_steps
        avg_du_dt = running_du_dt / log_steps
        logger.info(
            f"(epoch={epoch + 1:07d}) "
            f"Train Loss: {avg_loss:.4f}, "
            f"FM MSE (u vs v): {avg_fm_mse:.4f}, "
            f"|du/dt|: {avg_du_dt:.4f}, "
            f"Elapsed Time: {elapsed_min:03d}:{elapsed_sec:02d}",
        )

        # Save DiT checkpoint:
        save_ckpt(model, ema, opt, args, epoch)
        model.train()
