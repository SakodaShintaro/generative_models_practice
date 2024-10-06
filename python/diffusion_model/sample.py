# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Sample new images from a pre-trained DiT."""

import argparse
from pathlib import Path

import torch
from diffusers.models import AutoencoderKL
from diffusion import create_diffusion
from models import DiT_models
from torchvision.utils import save_image

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=Path, required=True)
    return parser.parse_args()


def sample_images(model: torch.nn.Module, vae: AutoencoderKL) -> torch.Tensor:
    diffusion = create_diffusion(str(250))
    latent_size = 96 // 8
    num_classes = 10
    device = model.parameters().__next__().device

    # Labels to condition the model with (feel free to change):
    class_labels = list(range(num_classes))

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([num_classes] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = {"y": y, "cfg_scale": 4.0}

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg,
        z.shape,
        z,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        progress=True,
        device=device,
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    return vae.decode(samples / 0.18215).sample


if __name__ == "__main__":
    args = parse_args()

    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model:
    latent_size = args.image_size // 8
    num_classes = args.num_classes
    model = DiT_models[args.model](input_size=latent_size, num_classes=num_classes).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    state_dict = torch.load(str(args.ckpt))
    state_dict = state_dict["ema"]
    model.load_state_dict(state_dict)
    model.eval()  # important!
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)

    samples = sample_images(model, vae)

    # Save and display images:
    save_dir = args.ckpt.parent.parent
    save_image(samples, save_dir / "sample.png", nrow=4, normalize=True, value_range=(-1, 1))
