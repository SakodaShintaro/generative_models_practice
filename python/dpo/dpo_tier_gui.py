"""Interactive Diffusion-DPO with LoRA on Qwen-Image-2.1 or MOSS-Image, ranked on a tier list.

A variant of `dpo_gui.py` for a single GPU. Instead of choosing between two images per round,
you rank images on a tier list (S > A > B > C > D) by dragging them around. A background thread
on the GPU repeats one step after another:

1. If images sit in at least two different tiers, pick one pair of images from different tiers
   uniformly at random and run one DPO step on it, the one in the higher tier being the winner.
2. Sample one new image from the current model and put it in the "unranked" row.

You can drag any image to any row at any time, including re-ranking older ones; each step reads
the tier list as it is then. Images in the "unranked" and "discarded" rows never enter the loss.
Sampling pauses while `--max_unranked` images are waiting to be ranked, so the GPU does not run
ahead of you. Right-click an image to see it larger.

Nothing is saved on its own: the button next to each tier saves that tier's images, named by the
tier and their position from the left, and the LoRA weights have a button of their own.

Diffusion-DPO (Wallace et al., 2023, <https://arxiv.org/abs/2311.12908>) for a preference
pair (winner y_w, loser y_l) sharing one prompt: add the *same* noise at the *same* sigma to
both and compare the errors of the model being trained against a frozen reference model:

    d      = ||v_hat_w - v||^2 - ||v_hat_l - v||^2
    d_ref  = ||v_ref_w - v||^2 - ||v_ref_l - v||^2
    loss   = -log sigmoid(-beta * (d - d_ref))

Both models are rectified flow models, so the noising is `x_t = (1 - sigma) * x_0 + sigma * eps`
and the target is the velocity `v = eps - x_0`.

`--pretrained_model` picks the model:

- Qwen/Qwen-Image-2.1 is sampled without classifier-free guidance. Optional `--image` reference
  images condition every sample, as in `qwen_generate.py`, e.g. to learn preferences about one
  person's poses. They are encoded once at startup, both by the Qwen3-VL text encoder (next to the
  prompt) and by the VAE, whose latent tokens are placed before the target image in every forward
  pass, for sampling and for the DPO steps alike. They are never noised and never enter the loss.
  The pipeline's `__call__` cannot take cached prompt embeddings together with condition images,
  so sampling runs its own denoising loop.
- OpenMOSS-Team/MOSS-Image-SFT and -Base (<https://github.com/OpenMOSS/MOSS-Image>) are
  text-to-image only. They are sampled with the recommended guidance: CFG 3 against an empty
  negative prompt whose branch skips block 22, i.e. two forward passes per step. The DPO loss is on
  the conditional prediction alone, as for any guided model.

With LoRA the reference model is free: it is the same transformer with the adapters disabled,
so only one set of weights is held in memory.
"""

import argparse
import functools
import gc
import itertools
import math
import queue
import threading
import time
import tkinter as tk
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers import DiffusionPipeline, QwenImage21Pipeline
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache
from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import (
    calculate_dimensions,
    calculate_shift,
    retrieve_timesteps,
)
from diffusers.training_utils import cast_training_params
from peft import LoraConfig, inject_adapter_in_model
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image, ImageTk
from safetensors.torch import load_file, save_file
from torchvision import transforms

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# The one GPU that both samples and trains.
DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16
NORMALIZE = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

QWEN_MAX_REFERENCE_IMAGES = 10

# MOSS-Image's recommended sampling (README "Sampling settings"; the pipeline's defaults).
MOSS_GUIDANCE_SCALE = 3.0
MOSS_UNCOND_SKIP_LAYERS = [22]
MOSS_NEGATIVE_PROMPT = ""

# The tiers from best to worst, with their row colors.
TIERS = [
    ("S", "#ff7f7f"),
    ("A", "#ffbf7f"),
    ("B", "#ffdf7f"),
    ("C", "#ffff7f"),
]
# Two more rows below the tiers that never enter the loss: new samples land in the first, and
# the second takes images you want out of the way.
ROWS = [*TIERS, ("未評価", "#d0d0d0"), ("除外", "#909090")]
UNRANKED_ROW = len(TIERS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument(
        "--pretrained_model", type=str, default="Qwen/Qwen-Image-2.1", choices=list(MODELS),
    )
    # Qwen-Image-2.1 only.
    parser.add_argument("--image", type=Path, nargs="*", default=[])
    parser.add_argument("--height", type=int, default=896)
    parser.add_argument("--width", type=int, default=512)
    # The longer side of the thumbnails and of the right-click preview; both keep the aspect ratio.
    parser.add_argument("--thumbnail_size", type=int, default=128)
    parser.add_argument("--display_size", type=int, default=512)
    parser.add_argument("--max_unranked", type=int, default=50)
    # MOSS-Image recommends at least 50.
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--beta_dpo", type=float, default=2500.0)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--base_seed", type=int, default=-1)
    parser.add_argument("--resume_lora", type=Path, default=None)
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    return parser.parse_args()


#################################################################################
#                                 Qwen-Image-2.1                                #
#################################################################################


class QwenImage:
    """Qwen-Image-2.1 with the prompt and the reference images encoded once."""

    # The transformer is single-stream, so these are all of its attention input projections.
    lora_target_modules: ClassVar[list[str]] = ["to_q", "to_k", "to_v"]

    def __init__(self, args: argparse.Namespace) -> None:
        # The target is (height / 16) x (width / 16) latent tokens, grouped 2x2 into
        # vision-language slots.
        assert args.height % 32 == 0, "--height must be a multiple of 32"
        assert args.width % 32 == 0, "--width must be a multiple of 32"
        assert len(args.image) <= QWEN_MAX_REFERENCE_IMAGES, (
            f"Qwen-Image-2.1 takes at most {QWEN_MAX_REFERENCE_IMAGES} reference images"
        )
        self.height = args.height
        self.width = args.width
        self.num_inference_steps = args.num_inference_steps
        references = [resize_reference(path, args.height * args.width) for path in args.image]
        # The Qwen3-VL text encoder (~17 GB) and the transformer (~14 GB) do not fit on one 24 GB
        # GPU together. Load on the CPU, run only the text encoder on the GPU for the one prompt
        # (and its reference images, which stay fixed for the session), drop it, and move the
        # transformer and the VAE over afterwards.
        self.pipeline = QwenImage21Pipeline.from_pretrained(args.pretrained_model, dtype=DTYPE)
        self.pipeline.text_encoder.to(DEVICE)
        with torch.no_grad():
            # encode_prompt takes None, not an empty list, for text-to-image.
            self.prompt_embeds, prompt_embeds_mask, self.image_pad_mask = (
                self.pipeline.encode_prompt(
                    prompt=args.prompt, image=references or None, device=DEVICE,
                )
            )
        # A single prompt has no padding, for which encode_prompt returns no mask at all.
        assert prompt_embeds_mask is None
        self.pipeline.text_encoder = None
        gc.collect()
        torch.cuda.empty_cache()
        self.pipeline.transformer.requires_grad_(False)
        self.pipeline.vae.requires_grad_(False)
        self.pipeline.to(DEVICE)
        self.transformer = self.pipeline.transformer
        self.vae = self.pipeline.vae
        self.scheduler = self.pipeline.scheduler
        # sample() caches the keys and values of the prompt and the reference images, which is only
        # valid because they are modulated with t = 0 at every step.
        assert self.transformer.config.causal_condition
        self.reference_latents, self.reference_shapes = self.encode_references(references)

    def enable_gradient_checkpointing(self) -> None:
        self.transformer.enable_gradient_checkpointing()

    @torch.no_grad()
    def sample(self, seed: int) -> Image.Image:
        """The pipeline's denoising loop without guidance, on the cached prompt and references."""
        height, width = self.latent_size()
        channels = self.transformer.config.in_channels
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        latents = torch.randn(
            (1, 1, channels, height, width), generator=generator, device=DEVICE, dtype=DTYPE,
        )
        latents = self.pipeline._pack_latents(latents, 1, channels, height, width)  # noqa: SLF001
        num_steps = self.num_inference_steps
        timesteps, _ = retrieve_timesteps(
            self.scheduler,
            num_steps,
            DEVICE,
            sigmas=np.linspace(1.0, 1 / num_steps, num_steps),
            mu=self.shift_mu(latents.shape[1]),
        )
        self.scheduler.set_begin_index(0)

        conditioning = self.conditioning(1, height, width)
        # The first step fills the cache with the prompt and the references; later steps only
        # recompute the target image's tokens.
        kv_cache = QwenImage21KVCache(len(self.transformer.transformer_blocks))
        for index, t in enumerate(timesteps):
            velocity = self.predict(
                latents,
                # Cast before scaling, as the pipeline does, so the timesteps round identically.
                t.expand(1).to(DTYPE) / 1000,
                conditioning,
                kv_cache,
                "extract" if index == 0 else "cached",
            )
            latents = self.scheduler.step(velocity, t, latents, return_dict=False)[0]
        return self.decode(latents, height, width)

    def latent_size(self) -> tuple[int, int]:
        """The target image's height and width in latent tokens."""
        scale = self.pipeline.vae_scale_factor
        return self.height // scale, self.width // scale

    def decode(self, latents: torch.Tensor, height: int, width: int) -> Image.Image:
        """Packed latents of one image back to a PIL image, as the pipeline does."""
        vae = self.vae
        latents = latents.transpose(1, 2).reshape(1, -1, 1, height, width).to(vae.dtype)
        stats_shape = (1, vae.config.z_dim, 1, 1, 1)
        latents_mean = torch.tensor(vae.config.latents_mean).view(stats_shape).to(latents)
        latents_std = torch.tensor(vae.config.latents_std).view(stats_shape).to(latents)
        image = vae.decode(latents * latents_std + latents_mean, return_dict=False)[0][:, :, 0]
        return self.pipeline.image_processor.postprocess(image, output_type="pil")[0]

    @torch.no_grad()
    def encode_references(
        self, references: list[Image.Image],
    ) -> tuple[torch.Tensor, list[tuple[int, int, int]]]:
        """Packed latents of all reference images, one after another, and their token shapes."""
        packed = [
            torch.zeros(1, 0, self.transformer.config.in_channels, device=DEVICE, dtype=DTYPE),
        ]
        shapes = []
        for reference in references:
            latents = self.encode(to_pixel_values([reference], "RGBA", DTYPE))
            _, channels, _, height, width = latents.shape
            packed.append(self.pipeline._pack_latents(latents, 1, channels, height, width))  # noqa: SLF001
            shapes.append((1, height, width))
        return torch.cat(packed, dim=1), shapes

    @torch.no_grad()
    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        """Packed latents of generated images, the ones DPO trains on."""
        # The VAE reads RGBA (and the pipeline returns RGBA).
        latents = self.encode(to_pixel_values(images, "RGBA", DTYPE))
        batch_size, channels, _, height, width = latents.shape
        return self.pipeline._pack_latents(latents, batch_size, channels, height, width)  # noqa: SLF001

    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # The VAE takes a length-1 frame axis.
        return self.pipeline._encode_vae_image(image=pixel_values.unsqueeze(2), generator=None)  # noqa: SLF001

    def conditioning(self, batch_size: int, height: int, width: int) -> dict:
        """Transformer inputs shared by `batch_size` targets of `height` x `width` latent tokens."""
        # As in the pipeline, the target image takes one vision-language slot per 2x2 group of
        # latent tokens, appended after the prompt (which holds the references' slots).
        img_mask = torch.cat(
            [self.image_pad_mask, self.image_pad_mask.new_ones(1, height * width // 4)], dim=1,
        )
        return {
            "encoder_hidden_states": self.prompt_embeds.repeat(batch_size, 1, 1),
            "img_shapes": [[*self.reference_shapes, (1, height, width)]] * batch_size,
            "img_mask": img_mask.repeat(batch_size, 1),
        }

    def shift_mu(self, image_seq_len: int) -> float:
        config = self.scheduler.config
        return calculate_shift(
            image_seq_len,
            config.base_image_seq_len,
            config.max_image_seq_len,
            config.base_shift,
            config.max_shift,
        )

    def shift_sigma(self, u: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """Shift a uniform-ish time in (0, 1) the same way inference shifts it."""
        return self.scheduler.time_shift(self.shift_mu(latents.shape[1]), 1.0, u)

    def velocity(self, noisy_latents: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """The trained transformer's prediction for a batch of noised target latents."""
        height, width = self.latent_size()
        conditioning = self.conditioning(noisy_latents.shape[0], height, width)
        timestep = sigma.expand(noisy_latents.shape[0])
        return self.predict(noisy_latents, timestep, conditioning, None, None)

    def predict(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        conditioning: dict,
        kv_cache: QwenImage21KVCache | None,
        kv_cache_mode: str | None,
    ) -> torch.Tensor:
        # The clean reference latents go before the target image.
        references = self.reference_latents.repeat(noisy_latents.shape[0], 1, 1)
        output = self.transformer(
            hidden_states=torch.cat([references, noisy_latents], dim=1),
            timestep=timestep,
            **conditioning,
            kv_cache=kv_cache,
            kv_cache_mode=kv_cache_mode,
            return_dict=False,
        )[0]
        # The output spans the whole joint text/image sequence; the target image is its tail.
        return output[:, -noisy_latents.shape[1]:]


def resize_reference(path: Path, area: int) -> Image.Image:
    """A reference image resized as the pipeline does: its aspect ratio, about `area` pixels.

    The size is a multiple of 32, so the text encoder (one slot per 32x32 pixels) and the VAE (one
    token per 16x16 pixels) agree on the 2x2 tokens per slot without resizing it again.
    """
    assert path.is_file(), f"{path} not found"
    image = Image.open(path).convert("RGBA")
    width, height, _ = calculate_dimensions(area, image.width / image.height)
    return image.resize((width, height), Image.LANCZOS)


#################################################################################
#                                    MOSS-Image                                 #
#################################################################################


class MossImage:
    """MOSS-Image with the prompt and the negative prompt encoded once.

    The transformer reads latents as [B, 128, H / 16, W / 16]: the FLUX.2 VAE's 32 channels at
    1/8 resolution with each 2x2 patch folded into the channels, normalized by the VAE's batch
    norm statistics. The text tokens go after the image tokens inside the transformer.
    """

    # The q/k/v projections of the 42 joint blocks; the two text-only blocks before them are left
    # alone (their fused `qkv` does not match).
    lora_target_modules = r"layers\.\d+\.proj\.[qkv]"

    def __init__(self, args: argparse.Namespace) -> None:
        assert not args.image, "MOSS-Image is text-to-image only; it takes no --image"
        assert args.height % 16 == 0, "--height must be a multiple of 16"
        assert args.width % 16 == 0, "--width must be a multiple of 16"
        self.height = args.height
        self.width = args.width
        self.num_inference_steps = args.num_inference_steps
        # The model code ships with the weights on the Hub, hence trust_remote_code.
        self.pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model,
            custom_pipeline="pipeline_moss_image",
            trust_remote_code=True,
            torch_dtype={"default": DTYPE, "vae": torch.float32},
        )
        # The Qwen3.5-9B text encoder (~18 GB) and the transformer (~10 GB) do not fit on one
        # 24 GB GPU together, so the prompts are encoded once and the encoder dropped, as for
        # Qwen-Image-2.1.
        self.pipeline.text_encoder.to(DEVICE)
        self.prompt_embeds, prompt_mask = self.pipeline.encode_prompt(args.prompt)
        self.negative_prompt_embeds, negative_mask = self.pipeline.encode_prompt(
            MOSS_NEGATIVE_PROMPT,
        )
        # One prompt per call is never padded, so the transformer can be given no mask.
        assert prompt_mask.all()
        assert negative_mask.all()
        self.pipeline.text_encoder = None
        gc.collect()
        torch.cuda.empty_cache()
        self.pipeline.transformer.requires_grad_(False)
        self.pipeline.vae.requires_grad_(False)
        self.pipeline.to(DEVICE)
        self.transformer = self.pipeline.transformer
        self.vae = self.pipeline.vae
        self.scheduler = self.pipeline.scheduler

    def enable_gradient_checkpointing(self) -> None:
        # The custom transformer does not implement diffusers' gradient checkpointing, so
        # checkpoint each joint block by hand. Without it the activations of the 42 blocks for a
        # winner/loser pair do not fit next to the weights.
        for block in self.transformer.layers:
            block.forward = functools.partial(
                torch.utils.checkpoint.checkpoint, block.forward, use_reentrant=False,
            )

    @torch.no_grad()
    def sample(self, seed: int) -> Image.Image:
        """The pipeline's denoising loop with its default guidance, on the cached prompts."""
        height, width = self.height // 16, self.width // 16
        channels = self.transformer.config.latent_channels
        # The pipeline samples the noise and integrates in float32.
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        latents = torch.randn(
            (1, channels, height, width),
            generator=generator,
            device=DEVICE,
            dtype=torch.float32,
        )
        self.scheduler.set_timesteps(self.num_inference_steps, device=DEVICE)
        for t in self.scheduler.timesteps:
            conditional = self.transformer(
                hidden_states=latents,
                timestep=t,
                encoder_hidden_states=self.prompt_embeds,
                return_dict=False,
            )[0].float()
            unconditional = self.transformer(
                hidden_states=latents,
                timestep=t,
                encoder_hidden_states=self.negative_prompt_embeds,
                skip_layers=MOSS_UNCOND_SKIP_LAYERS,
                return_dict=False,
            )[0].float()
            velocity = unconditional + MOSS_GUIDANCE_SCALE * (conditional - unconditional)
            latents = self.scheduler.step(velocity, t, latents, return_dict=False)[0]
        return self.decode(latents)

    def decode(self, latents: torch.Tensor) -> Image.Image:
        """Latents of one image back to a PIL image, as the pipeline does."""
        latents = latents.to(self.vae.dtype)
        mean, std = batch_norm_stats(self.vae)
        latents = Flux2Pipeline._unpatchify_latents(latents * std + mean)  # noqa: SLF001
        image = self.vae.decode(latents, return_dict=False)[0].clamp(-1, 1)
        return self.pipeline.image_processor.postprocess(image, output_type="pil")[0]

    @torch.no_grad()
    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        """Latents of generated images, the ones DPO trains on, as Flux2Pipeline encodes them."""
        pixel_values = to_pixel_values(images, "RGB", self.vae.dtype)
        latents = self.vae.encode(pixel_values).latent_dist.mode()
        mean, std = batch_norm_stats(self.vae)
        return (Flux2Pipeline._patchify_latents(latents) - mean) / std  # noqa: SLF001

    def shift_sigma(self, u: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """Shift a uniform-ish time in (0, 1) by the scheduler's fixed inference shift."""
        # Unlike Qwen-Image-2.1's, this shift does not depend on the image size.
        del latents
        shift = self.scheduler.config.shift
        return shift * u / (1 + (shift - 1) * u)

    def velocity(self, noisy_latents: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """The trained transformer's conditional prediction for a batch of noised latents."""
        batch_size = noisy_latents.shape[0]
        return self.transformer(
            hidden_states=noisy_latents,
            # The transformer takes the scheduler's timesteps, sigma * 1000.
            timestep=(sigma * self.scheduler.config.num_train_timesteps).expand(batch_size),
            encoder_hidden_states=self.prompt_embeds.repeat(batch_size, 1, 1),
            return_dict=False,
        )[0]


def batch_norm_stats(vae: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """The FLUX.2 VAE's latent mean and std, shaped to broadcast over [B, 128, H, W]."""
    mean = vae.bn.running_mean.view(1, -1, 1, 1)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
    return mean, std


MODELS = {
    "Qwen/Qwen-Image-2.1": QwenImage,
    "OpenMOSS-Team/MOSS-Image-SFT": MossImage,
    "OpenMOSS-Team/MOSS-Image-Base": MossImage,
}


#################################################################################
#                                       DPO                                     #
#################################################################################


@dataclass
class Sample:
    """One generated image."""

    index: int
    seed: int
    image: Image.Image


class TierDpo:
    """Holds the model, the LoRA optimizer, one sampler, and one DPO update step."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        # Everything this session saves (LoRA weights, images) goes into one directory per launch.
        self.session_dir = args.results_dir / timestamp()
        self.session_dir.mkdir(parents=True)
        self.model = MODELS[args.pretrained_model](args)
        self.transformer = self.model.transformer

        add_lora(self.transformer, args.lora_rank, self.model.lora_target_modules)
        if args.resume_lora is not None:
            self.load_lora(args.resume_lora)
        cast_training_params(self.transformer, dtype=torch.float32)
        self.model.enable_gradient_checkpointing()
        self.lora_params = [p for p in self.transformer.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            self.lora_params, lr=args.learning_rate, weight_decay=1e-2,
        )
        self.num_samples = 0
        self.num_steps = 0

    def seed(self, index: int) -> int:
        if self.args.base_seed == -1:
            return int(torch.randint(0, 2**31 - 1, (1,)))
        return self.args.base_seed + index

    def sample(self) -> Sample:
        """One sample of the current model."""
        index = self.num_samples
        seed = self.seed(index)
        start_time = time.time()
        image = self.model.sample(seed)
        elapsed = time.time() - start_time
        print(f"sample {index}: seed={seed}, {elapsed:.2f} seconds")
        self.num_samples += 1
        return Sample(index, seed, image)

    def sample_sigma(self, latents: torch.Tensor) -> torch.Tensor:
        """One sigma in (0, 1), logit-normal and shifted the same way inference shifts it."""
        u = torch.sigmoid(
            torch.randn(1, device=DEVICE) * self.args.logit_std + self.args.logit_mean,
        )
        return self.model.shift_sigma(u, latents)

    def dpo_step(self, latents: torch.Tensor) -> tuple[float, float]:
        # The winner and its loser must see identical noise and sigma.
        noise = torch.randn_like(latents[:1]).expand_as(latents)
        sigma = self.sample_sigma(latents).to(latents.dtype)
        noisy_latents = (1.0 - sigma) * latents + sigma * noise
        # Rectified flow: the model predicts the velocity from the noise towards the data.
        target = noise - latents

        model_pred = self.model.velocity(noisy_latents, sigma)
        model_diff = win_minus_lose_error(model_pred, target)
        with torch.no_grad(), reference_model(self.transformer):
            ref_pred = self.model.velocity(noisy_latents, sigma)
            ref_diff = win_minus_lose_error(ref_pred, target)

        logits = -self.args.beta_dpo * (model_diff - ref_diff)
        loss = -F.logsigmoid(logits).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.lora_params, 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.num_steps += 1
        return loss.item(), (logits > 0).float().mean().item()

    def learn_from(self, winner: Sample, loser: Sample) -> tuple[float, float]:
        """Run one DPO step on one pair of images from different tiers."""
        latents = self.model.encode_images([winner.image, loser.image])
        return self.dpo_step(latents)

    def load_lora(self, path: Path) -> None:
        """Restore the adapter of a previous session into the freshly added one."""
        assert path.is_file(), f"{path} not found"
        # save_lora prefixes every key with the name of the model it belongs to.
        state_dict = {
            key.removeprefix("transformer."): value for key, value in load_file(str(path)).items()
        }
        result = set_peft_model_state_dict(self.transformer, state_dict)
        assert not result.unexpected_keys, f"unexpected keys in {path}: {result.unexpected_keys}"
        print(f"resumed LoRA weights from {path}")

    def save_lora(self) -> Path:
        # The key layout of diffusers' save_lora_weights, which MOSS-Image's pipeline lacks.
        path = self.session_dir / f"{timestamp()}.safetensors"
        state_dict = get_peft_model_state_dict(self.transformer)
        save_file({f"transformer.{key}": value for key, value in state_dict.items()}, str(path))
        return path

    def save_tier(self, name: str, samples: list[Sample]) -> list[Path]:
        """Save one tier's images at their generated resolution, numbered from the left."""
        # One timestamp for the whole tier, so its images sort together.
        prefix = timestamp()
        paths = []
        for position, sample in enumerate(samples):
            path = self.session_dir / f"{prefix}_{name}_{position:03d}.png"
            sample.image.save(path)
            paths.append(path)
        return paths


def timestamp() -> str:
    return datetime.now(tz=UTC).astimezone().strftime("%Y%m%d_%H%M%S")


def add_lora(transformer: torch.nn.Module, rank: int, target_modules: list[str] | str) -> None:
    """Attach a trainable LoRA adapter. PEFT directly, since MOSS-Image's transformer lacks
    diffusers' `add_adapter`; the adapter is active as soon as it is injected."""
    config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    inject_adapter_in_model(config, transformer)


def to_pixel_values(images: list[Image.Image], mode: str, dtype: torch.dtype) -> torch.Tensor:
    """The images as one [-1, 1] batch on the GPU, with the channels of `mode`."""
    pixel_values = torch.stack([NORMALIZE(image.convert(mode)) for image in images])
    return pixel_values.to(DEVICE, dtype=dtype)


@contextmanager
def reference_model(transformer: torch.nn.Module):
    """Temporarily turn the transformer back into the frozen reference model."""
    lora_layers = [module for module in transformer.modules() if isinstance(module, BaseTunerLayer)]
    for layer in lora_layers:
        layer.enable_adapters(False)
    try:
        yield
    finally:
        for layer in lora_layers:
            layer.enable_adapters(True)


def win_minus_lose_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Difference of the flow-matching errors of the winner and the loser."""
    per_sample = F.mse_loss(pred.float(), target.float(), reduction="none").flatten(1).mean(dim=1)
    win, lose = per_sample.chunk(2)
    return win - lose


#################################################################################
#                                   Tier list                                   #
#################################################################################


class TierBoard:
    """The images in every row, shared by the GUI thread and the GPU thread.

    Every access holds `changed`, which also wakes the GPU thread when a row changes.
    """

    def __init__(self) -> None:
        self.changed = threading.Condition()
        self.rows: list[list[Sample]] = [[] for _ in ROWS]
        # Bumped on every change, so the GUI can tell when to redraw.
        self.version = 0

    def snapshot(self) -> tuple[int, list[list[Sample]]]:
        with self.changed:
            return self.version, [list(samples) for samples in self.rows]

    def add(self, sample: Sample) -> None:
        with self.changed:
            self.rows[UNRANKED_ROW].append(sample)
            self.bump()

    def move(self, sample: Sample, row: int, index: int) -> None:
        """Move `sample` to `row`, before the image at `index` as the row was shown."""
        with self.changed:
            old_row = next(r for r, samples in enumerate(self.rows) if sample in samples)
            old_index = self.rows[old_row].index(sample)
            self.rows[old_row].remove(sample)
            # Within one row, the images after the old position move one place forward.
            if old_row == row and old_index < index:
                index -= 1
            self.rows[row].insert(min(index, len(self.rows[row])), sample)
            self.bump()

    def bump(self) -> None:
        self.version += 1
        self.changed.notify_all()

    def num_unranked(self) -> int:
        with self.changed:
            return len(self.rows[UNRANKED_ROW])

    def num_pairs(self) -> int:
        with self.changed:
            return sum(
                len(self.rows[better]) * len(self.rows[worse])
                for better, worse in itertools.combinations(range(len(TIERS)), 2)
            )

    def random_pair(self) -> tuple[Sample, Sample] | None:
        """A (winner, loser) pair uniformly among all pairs of images in different tiers."""
        with self.changed:
            pairs = [
                pair
                for better, worse in itertools.combinations(range(len(TIERS)), 2)
                for pair in itertools.product(self.rows[better], self.rows[worse])
            ]
            if not pairs:
                return None
            return pairs[int(torch.randint(0, len(pairs), (1,)))]


class GpuWorker:
    """The GPU thread: one DPO step on a random pair, then one new sample, over and over."""

    def __init__(self, trainer: TierDpo, board: TierBoard, max_unranked: int) -> None:
        assert max_unranked >= 1, "--max_unranked must be at least 1"
        self.trainer = trainer
        self.board = board
        self.max_unranked = max_unranked
        # Status lines for the GUI, which polls them; tkinter must only be touched from its thread.
        self.outbox: queue.Queue[str] = queue.Queue()
        # Set by the GUI and served between steps, so saving never races a DPO step.
        self.save_requested = False
        self.last_step = "no DPO step yet"

    def request_save(self) -> None:
        with self.board.changed:
            self.save_requested = True
            self.board.changed.notify_all()

    def ready(self) -> bool:
        return self.save_requested or self.board.num_unranked() < self.max_unranked

    def run(self) -> None:
        try:
            while True:
                self.step()
        except BaseException as error:
            self.outbox.put(f"GPU thread stopped: {error!r}")
            raise

    def step(self) -> None:
        with self.board.changed:
            if not self.ready():
                self.outbox.put(
                    f"未評価の画像が {self.max_unranked} 枚あるので生成を止めています  "
                    f"({self.last_step})",
                )
            self.board.changed.wait_for(self.ready)
            save_requested = self.save_requested
            self.save_requested = False
            # The tier list as it is now; later drags only affect later steps.
            pair = self.board.random_pair()
        if save_requested:
            self.save()
            return
        if pair is not None:
            winner, loser = pair
            self.outbox.put(f"DPO step on #{winner.index} > #{loser.index}...")
            loss, accuracy = self.trainer.learn_from(winner, loser)
            self.last_step = (
                f"step {self.trainer.num_steps}: #{winner.index} > #{loser.index}, "
                f"loss={loss:.4f}, acc={accuracy:.2f}"
            )
            print(self.last_step)
        self.outbox.put(f"sampling #{self.trainer.num_samples}...  ({self.last_step})")
        sample = self.trainer.sample()
        self.board.add(sample)
        self.outbox.put(
            f"sampled #{sample.index}, {self.board.num_pairs()} pairs  ({self.last_step})",
        )

    def save(self) -> None:
        self.outbox.put("saving LoRA weights...")
        path = self.trainer.save_lora()
        print(f"saved LoRA weights to {path}")
        self.outbox.put(f"saved {path.name}  ({self.last_step})")


#################################################################################
#                                       GUI                                     #
#################################################################################

ROW_LABEL_WIDTH = 90
# Tall enough for a row's name, its count and the save button under them.
ROW_MIN_HEIGHT = 100
PADDING = 6
POLL_MILLISECONDS = 200


@dataclass
class Drag:
    """The image being dragged and where the pointer was last, in canvas coordinates."""

    item: int
    sample: Sample
    x: float
    y: float


class TierWindow:
    """Draws the tier list on one canvas and turns drag and drop into moves on the board."""

    def __init__(
        self, args: argparse.Namespace, board: TierBoard, worker: GpuWorker, trainer: TierDpo,
    ) -> None:
        self.args = args
        self.trainer = trainer
        self.board = board
        self.worker = worker
        thumbnail_width, thumbnail_height = fit_size(args.width, args.height, args.thumbnail_size)
        self.thumbnail_size = (thumbnail_width, thumbnail_height)
        self.cell_width = thumbnail_width + PADDING
        self.cell_height = thumbnail_height + PADDING
        self.thumbnails: dict[int, ImageTk.PhotoImage] = {}
        self.item_samples: dict[int, Sample] = {}
        # The (top, bottom) of every row and the images per line, as last drawn.
        self.bands: list[tuple[int, int]] = []
        self.per_line = 1
        self.drawn_version = -1
        self.drag: Drag | None = None
        self.preview_window: tk.Toplevel | None = None
        self.preview_label: tk.Label | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None

        self.root = tk.Tk()
        self.root.title(f"Tier Diffusion-DPO — {args.prompt}")
        self.root.geometry(f"{ROW_LABEL_WIDTH + 8 * self.cell_width + 40}x900")

        top_frame = tk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=10, pady=(10, 4))
        tk.Label(top_frame, text=args.prompt, font=("", 12)).pack(side=tk.LEFT)
        tk.Button(top_frame, text="LoRAを保存", width=14, command=self.on_save).pack(
            side=tk.RIGHT,
        )
        self.status = tk.Label(self.root, text="sampling...", font=("", 10), anchor=tk.W)
        self.status.pack(fill=tk.X, padx=10)

        canvas_frame = tk.Frame(self.root)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(4, 10))
        scrollbar = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas = tk.Canvas(
            canvas_frame, background="#303030", highlightthickness=0,
            yscrollcommand=scrollbar.set,
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.canvas.yview)
        # Made once and placed on the canvas at every redraw, which only deletes their placement.
        self.tier_save_buttons = [
            tk.Button(
                self.canvas,
                text="保存",
                width=6,
                command=functools.partial(self.on_save_tier, row),
            )
            for row in range(len(TIERS))
        ]

        self.canvas.tag_bind("sample", "<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.tag_bind("sample", "<ButtonPress-3>", self.on_preview)
        self.canvas.bind("<Configure>", lambda _: self.redraw())
        # X11 sends the wheel as buttons 4 and 5, other platforms as <MouseWheel>.
        self.canvas.bind_all("<Button-4>", lambda _: self.canvas.yview_scroll(-1, "units"))
        self.canvas.bind_all("<Button-5>", lambda _: self.canvas.yview_scroll(1, "units"))
        self.canvas.bind_all(
            "<MouseWheel>",
            lambda event: self.canvas.yview_scroll(-1 if event.delta > 0 else 1, "units"),
        )
        self.root.after(POLL_MILLISECONDS, self.poll)

    def poll(self) -> None:
        """Show the GPU thread's latest status and its new samples."""
        while not self.worker.outbox.empty():
            self.status.config(text=self.worker.outbox.get())
        # Redrawing mid-drag would drop the dragged image back into place.
        if self.drag is None and self.board.version != self.drawn_version:
            self.redraw()
        self.root.after(POLL_MILLISECONDS, self.poll)

    def thumbnail(self, sample: Sample) -> ImageTk.PhotoImage:
        if sample.index not in self.thumbnails:
            self.thumbnails[sample.index] = ImageTk.PhotoImage(
                sample.image.resize(self.thumbnail_size, Image.LANCZOS),
            )
        return self.thumbnails[sample.index]

    def redraw(self) -> None:
        if self.drag is not None:
            return
        self.drawn_version, rows = self.board.snapshot()
        self.canvas.delete("all")
        self.item_samples = {}
        self.bands = []
        width = max(self.canvas.winfo_width(), ROW_LABEL_WIDTH + PADDING + self.cell_width)
        self.per_line = max(1, (width - ROW_LABEL_WIDTH - PADDING) // self.cell_width)
        top = 0
        for row, ((name, color), samples) in enumerate(zip(ROWS, rows, strict=True)):
            num_lines = max(1, math.ceil(len(samples) / self.per_line))
            bottom = top + max(num_lines * self.cell_height + PADDING, ROW_MIN_HEIGHT)
            self.canvas.create_rectangle(
                0, top, ROW_LABEL_WIDTH, bottom, fill=color, outline="#202020",
            )
            self.canvas.create_text(
                ROW_LABEL_WIDTH / 2, top + PADDING,
                text=f"{name}\n({len(samples)})",
                font=("", 14, "bold"),
                justify=tk.CENTER,
                anchor=tk.N,
            )
            if row < len(TIERS):
                self.canvas.create_window(
                    ROW_LABEL_WIDTH / 2, bottom - PADDING,
                    window=self.tier_save_buttons[row],
                    anchor=tk.S,
                )
            self.canvas.create_rectangle(
                ROW_LABEL_WIDTH, top, width, bottom, fill="#404040", outline="#202020",
            )
            for position, sample in enumerate(samples):
                line, column = divmod(position, self.per_line)
                item = self.canvas.create_image(
                    ROW_LABEL_WIDTH + PADDING + column * self.cell_width,
                    top + PADDING + line * self.cell_height,
                    image=self.thumbnail(sample),
                    anchor=tk.NW,
                    tags=("sample",),
                )
                self.item_samples[item] = sample
            self.bands.append((top, bottom))
            top = bottom
        self.canvas.config(scrollregion=(0, 0, width, top))

    def on_press(self, event: tk.Event) -> None:
        item = self.canvas.find_withtag("current")[0]
        self.canvas.tag_raise(item)
        self.drag = Drag(
            item,
            self.item_samples[item],
            self.canvas.canvasx(event.x),
            self.canvas.canvasy(event.y),
        )

    def on_motion(self, event: tk.Event) -> None:
        if self.drag is None:
            return
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        self.canvas.move(self.drag.item, x - self.drag.x, y - self.drag.y)
        self.drag.x, self.drag.y = x, y

    def on_release(self, event: tk.Event) -> None:
        if self.drag is None:
            return
        sample = self.drag.sample
        self.drag = None
        row, index = self.drop_position(
            self.canvas.canvasx(event.x), self.canvas.canvasy(event.y),
        )
        self.board.move(sample, row, index)
        self.redraw()

    def drop_position(self, x: float, y: float) -> tuple[int, int]:
        """The row under the pointer, and the index in it that the pointer is in front of."""
        # Above the first row or below the last one counts as that row.
        row = len(self.bands) - 1
        for index, (_, bottom) in enumerate(self.bands):
            if y < bottom:
                row = index
                break
        top, _ = self.bands[row]
        line = max(0, int((y - top - PADDING) // self.cell_height))
        # The left half of an image drops in front of it, the right half behind it.
        column = int((x - ROW_LABEL_WIDTH - PADDING) / self.cell_width + 0.5)
        column = min(max(column, 0), self.per_line)
        return row, line * self.per_line + column

    def on_preview(self, event: tk.Event) -> None:
        del event
        sample = self.item_samples[self.canvas.find_withtag("current")[0]]
        if self.preview_window is None or not self.preview_window.winfo_exists():
            self.preview_window = tk.Toplevel(self.root)
            self.preview_label = tk.Label(self.preview_window)
            self.preview_label.pack()
        size = fit_size(self.args.width, self.args.height, self.args.display_size)
        self.preview_photo = ImageTk.PhotoImage(sample.image.resize(size, Image.LANCZOS))
        self.preview_label.config(image=self.preview_photo)
        self.preview_window.title(f"#{sample.index} (seed {sample.seed})")

    def on_save_tier(self, row: int) -> None:
        name, _ = TIERS[row]
        _, rows = self.board.snapshot()
        if not rows[row]:
            self.status.config(text=f"{name} has no images to save")
            return
        paths = self.trainer.save_tier(name, rows[row])
        print(f"saved {len(paths)} images of {name} to {self.trainer.session_dir}")
        self.status.config(text=f"saved {len(paths)} images of {name} as {paths[0].name}, ...")

    def on_save(self) -> None:
        # Saving does not end the session: keep ranking afterwards, and save again any time.
        self.worker.request_save()
        self.status.config(text="saving after the current step...")

    def run(self) -> None:
        self.root.mainloop()


def fit_size(width: int, height: int, longest: int) -> tuple[int, int]:
    """`width` x `height` scaled so that its longer side is `longest` pixels."""
    scale = longest / max(width, height)
    return round(width * scale), round(height * scale)


def main() -> None:
    args = parse_args()
    trainer = TierDpo(args)
    board = TierBoard()
    worker = GpuWorker(trainer, board, args.max_unranked)
    window = TierWindow(args, board, worker, trainer)
    # Daemon, so closing the window ends the program even in the middle of a step.
    threading.Thread(target=worker.run, daemon=True).start()
    window.run()


if __name__ == "__main__":
    main()
