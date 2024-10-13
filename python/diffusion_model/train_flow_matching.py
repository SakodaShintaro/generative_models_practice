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
from diffusion import create_diffusion
from models import DiT_models
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

# the first flag below was False when we tested this script but True makes training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

#################################################################################
#                             Training Helper Functions                         #
#################################################################################


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-S/2")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--results_dir", type=Path, default="results")
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--global_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=50_00)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--dataset", type=str, choices=["mnist", "cifar10", "stl10"])
    return parser.parse_args()


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float = 0.9999) -> None:
    """Step the EMA model towards the current model."""
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model: torch.nn.Module, flag: bool) -> None:  # noqa: FBT001
    """Set requires_grad flag for all parameters in a model."""
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir: str) -> logging.Logger:
    """Create a logger that writes to a log file and stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")],
    )
    return logging.getLogger(__name__)


def sample_images(
    model: torch.nn.Module,
    vae: AutoencoderKL,
    args: argparse.Namespace,
    sample_n: int = 10,
) -> torch.Tensor:
    latent_size = image_size // 8
    num_classes = args.num_classes
    device = model.parameters().__next__().device

    # Labels to condition the model with (feel free to change):
    class_labels = list(range(num_classes))

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    with torch.no_grad():
        dt = 1.0 / sample_n
        for i in range(sample_n):
            num_t = i / sample_n * (1 - eps) + eps
            t = torch.ones(n, device=device) * num_t
            pred = model(z, t * 999, y)

            z = z.detach().clone() + pred * dt

        return vae.decode(z / 0.18215).sample

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
    checkpoint_dir = results_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(results_dir)
    logger.info(f"Experiment directory created at {results_dir}")

    # Create model:
    image_size = args.image_size
    assert image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = image_size // 8
    ckpt = torch.load(args.ckpt) if args.ckpt is not None else None
    model = DiT_models[args.model](
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
    diffusion = create_diffusion(
        timestep_respacing="",
    )  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    if ckpt is not None:
        opt.load_state_dict(ckpt["opt"])

    # Setup data:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ],
    )
    dataset = None
    if args.dataset == "mnist":
        from torchvision.datasets import MNIST

        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 1chのMNISTを3chに変換
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ],
        )
        dataset = MNIST(args.data_path, train=True, transform=transform, download=True)
    elif args.dataset == "cifar10":
        from torchvision.datasets import CIFAR10

        dataset = CIFAR10(args.data_path, train=True, transform=transform, download=True)
    elif args.dataset == "stl10":
        from torchvision.datasets import STL10

        dataset = STL10(args.data_path, split="train", transform=transform, download=True)

    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size),
        shuffle=False,
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
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    eps = 0.001

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            # -1であるラベルはclass_numに変換する
            y = torch.where(y == -1, torch.tensor(args.num_classes, device=device), y)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            noise = torch.randn_like(x)
            t = torch.rand(x.shape[0], device=device) * (1 - eps) + eps
            t = t.view(-1, 1, 1, 1)
            perturbed_data = t * x + (1 - t) * noise
            t = t.squeeze()
            out = model(perturbed_data, t * 999, y)
            target = x - noise
            loss = torch.mean(torch.square(out - target))
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = avg_loss.item()
                logger.info(
                    f"(step={train_steps:07d}) "
                    f"Train Loss: {avg_loss:.4f}, "
                    f"Train Steps/Sec: {steps_per_sec:.2f}",
                )
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 or train_steps == 1:
                checkpoint = {
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args,
                }
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
                model.eval()
                samples = sample_images(model, vae, args)
                save_image(
                    samples,
                    results_dir / f"sample_{train_steps:07d}.png",
                    nrow=4,
                    normalize=True,
                    value_range=(-1, 1),
                )
                model.train()

    # Save final checkpoint:
    checkpoint = {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "opt": opt.state_dict(),
        "args": args,
    }
    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")

    model.eval()
    samples = sample_images(model, vae, args)
    save_image(
        samples,
        results_dir / "sample_last.png",
        nrow=4,
        normalize=True,
        value_range=(-1, 1),
    )

    logger.info("Done!")
