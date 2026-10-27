"""Interactive Diffusion-DPO with LoRA on Qwen-Image-2.1 or MOSS-Image.

You are the preference dataset: each round the current model samples two images for one
prompt with two different seeds. A window shows them side by side; the button you press
decides the winner, and that single preference pair is immediately used for a few DPO
gradient steps on the LoRA adapters. The next round then samples from the updated model.

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

With two visible GPUs (e.g. no `CUDA_VISIBLE_DEVICES`), the two candidates of a round are sampled
in parallel, one per GPU. The second GPU holds a copy of the transformer and the VAE that only
samples; training stays on the first GPU, and the LoRA weights are copied over after each update.
`CUDA_VISIBLE_DEVICES=0` falls back to sampling both candidates one after the other on one GPU.
"""

import argparse
import copy
import functools
import gc
import time
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline
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

# The GPU that trains; it also samples, alone or next to a second GPU.
DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16
CANDIDATES_PER_ROUND = 2
NORMALIZE = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

QWEN_MAX_REFERENCE_IMAGES = 10

# MOSS-Image's recommended sampling (README "Sampling settings"; the pipeline's defaults).
MOSS_GUIDANCE_SCALE = 3.0
MOSS_UNCOND_SKIP_LAYERS = [22]
MOSS_NEGATIVE_PROMPT = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument(
        "--pretrained_model", type=str, default="Qwen/Qwen-Image-2.1", choices=list(MODELS),
    )
    # Qwen-Image-2.1 only.
    parser.add_argument("--image", type=Path, nargs="*", default=[])
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--display_size", type=int, default=512)
    # MOSS-Image recommends at least 50.
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--steps_per_pair", type=int, default=1)
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


@dataclass
class QwenReplica:
    """Everything one GPU needs to sample: the models and the cached prompt and references."""

    device: torch.device
    transformer: torch.nn.Module
    vae: torch.nn.Module
    # The scheduler keeps the current step index, so each concurrent sampler needs its own.
    scheduler: FlowMatchEulerDiscreteScheduler
    prompt_embeds: torch.Tensor
    image_pad_mask: torch.Tensor
    reference_latents: torch.Tensor


class QwenImage:
    """Qwen-Image-2.1 with the prompt and the reference images encoded once, one replica per GPU."""

    # The transformer is single-stream, so these are all of its attention input projections.
    lora_target_modules: ClassVar[list[str]] = ["to_q", "to_k", "to_v"]

    def __init__(self, args: argparse.Namespace, devices: list[torch.device]) -> None:
        # The target is (resolution / 16)^2 latent tokens, grouped 2x2 into vision-language slots.
        assert args.resolution % 32 == 0, "--resolution must be a multiple of 32"
        assert len(args.image) <= QWEN_MAX_REFERENCE_IMAGES, (
            f"Qwen-Image-2.1 takes at most {QWEN_MAX_REFERENCE_IMAGES} reference images"
        )
        self.resolution = args.resolution
        self.num_inference_steps = args.num_inference_steps
        references = [resize_reference(path, args.resolution) for path in args.image]
        # The Qwen3-VL text encoder (~17 GB) and the transformer (~14 GB) do not fit on one 24 GB
        # GPU together. Load on the CPU, run only the text encoder on the GPU for the one prompt
        # (and its reference images, which stay fixed for the session), drop it, and move the
        # transformer and the VAE over afterwards.
        self.pipeline = QwenImage21Pipeline.from_pretrained(args.pretrained_model, dtype=DTYPE)
        self.pipeline.text_encoder.to(DEVICE)
        with torch.no_grad():
            # encode_prompt takes None, not an empty list, for text-to-image.
            prompt_embeds, prompt_embeds_mask, image_pad_mask = self.pipeline.encode_prompt(
                prompt=args.prompt, image=references or None, device=DEVICE,
            )
        # A single prompt has no padding, for which encode_prompt returns no mask at all.
        assert prompt_embeds_mask is None
        self.pipeline.text_encoder = None
        gc.collect()
        torch.cuda.empty_cache()
        self.pipeline.transformer.requires_grad_(False)
        self.pipeline.vae.requires_grad_(False)
        # Two transformers do not fit on one GPU, so the copies for the other GPUs are made while
        # everything is still on the CPU.
        copies = [
            (
                copy.deepcopy(self.pipeline.transformer).to(device),
                copy.deepcopy(self.pipeline.vae).to(device),
            )
            for device in devices[1:]
        ]
        self.pipeline.to(DEVICE)
        self.scheduler = self.pipeline.scheduler
        # sample() caches the keys and values of the prompt and the reference images, which is only
        # valid because they are modulated with t = 0 at every step.
        assert self.pipeline.transformer.config.causal_condition
        reference_latents, self.reference_shapes = self.encode_references(references)

        self.replicas = [
            QwenReplica(
                DEVICE,
                self.pipeline.transformer,
                self.pipeline.vae,
                self.scheduler,
                prompt_embeds,
                image_pad_mask,
                reference_latents,
            ),
        ]
        for device, (transformer, vae) in zip(devices[1:], copies, strict=True):
            self.replicas.append(
                QwenReplica(
                    device,
                    transformer,
                    vae,
                    copy.deepcopy(self.scheduler),
                    prompt_embeds.to(device),
                    image_pad_mask.to(device),
                    reference_latents.to(device),
                ),
            )

    @property
    def transformers(self) -> list[torch.nn.Module]:
        return [replica.transformer for replica in self.replicas]

    def enable_gradient_checkpointing(self) -> None:
        self.replicas[0].transformer.enable_gradient_checkpointing()

    @torch.no_grad()
    def sample(self, replica_index: int, seed: int) -> Image.Image:
        """The pipeline's denoising loop without guidance, on the cached prompt and references."""
        replica = self.replicas[replica_index]
        size = self.resolution // self.pipeline.vae_scale_factor
        channels = replica.transformer.config.in_channels
        # The same seed gives the same noise on either GPU.
        generator = torch.Generator(device=replica.device).manual_seed(seed)
        latents = torch.randn(
            (1, 1, channels, size, size), generator=generator, device=replica.device, dtype=DTYPE,
        )
        latents = self.pipeline._pack_latents(latents, 1, channels, size, size)  # noqa: SLF001
        num_steps = self.num_inference_steps
        timesteps, _ = retrieve_timesteps(
            replica.scheduler,
            num_steps,
            replica.device,
            sigmas=np.linspace(1.0, 1 / num_steps, num_steps),
            mu=self.shift_mu(latents.shape[1]),
        )
        replica.scheduler.set_begin_index(0)

        conditioning = self.conditioning(replica, 1, size, size)
        # The first step fills the cache with the prompt and the references; later steps only
        # recompute the target image's tokens.
        kv_cache = QwenImage21KVCache(len(replica.transformer.transformer_blocks))
        for index, t in enumerate(timesteps):
            velocity = self.predict(
                replica,
                latents,
                # Cast before scaling, as the pipeline does, so the timesteps round identically.
                t.expand(1).to(DTYPE) / 1000,
                conditioning,
                kv_cache,
                "extract" if index == 0 else "cached",
            )
            latents = replica.scheduler.step(velocity, t, latents, return_dict=False)[0]
        return self.decode(replica, latents, size)

    def decode(self, replica: QwenReplica, latents: torch.Tensor, size: int) -> Image.Image:
        """Packed latents of one image back to a PIL image, as the pipeline does."""
        vae = replica.vae
        latents = latents.transpose(1, 2).reshape(1, -1, 1, size, size).to(vae.dtype)
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
            torch.zeros(
                1, 0, self.pipeline.transformer.config.in_channels, device=DEVICE, dtype=DTYPE,
            ),
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

    def conditioning(self, replica: QwenReplica, batch_size: int, height: int, width: int) -> dict:
        """Transformer inputs shared by `batch_size` targets of `height` x `width` latent tokens."""
        # As in the pipeline, the target image takes one vision-language slot per 2x2 group of
        # latent tokens, appended after the prompt (which holds the references' slots).
        img_mask = torch.cat(
            [replica.image_pad_mask, replica.image_pad_mask.new_ones(1, height * width // 4)],
            dim=1,
        )
        return {
            "encoder_hidden_states": replica.prompt_embeds.repeat(batch_size, 1, 1),
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
        size = self.resolution // self.pipeline.vae_scale_factor
        replica = self.replicas[0]
        conditioning = self.conditioning(replica, noisy_latents.shape[0], size, size)
        timestep = sigma.expand(noisy_latents.shape[0])
        return self.predict(replica, noisy_latents, timestep, conditioning, None, None)

    def predict(
        self,
        replica: QwenReplica,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        conditioning: dict,
        kv_cache: QwenImage21KVCache | None,
        kv_cache_mode: str | None,
    ) -> torch.Tensor:
        # The clean reference latents go before the target image.
        references = replica.reference_latents.repeat(noisy_latents.shape[0], 1, 1)
        output = replica.transformer(
            hidden_states=torch.cat([references, noisy_latents], dim=1),
            timestep=timestep,
            **conditioning,
            kv_cache=kv_cache,
            kv_cache_mode=kv_cache_mode,
            return_dict=False,
        )[0]
        # The output spans the whole joint text/image sequence; the target image is its tail.
        return output[:, -noisy_latents.shape[1]:]


def resize_reference(path: Path, resolution: int) -> Image.Image:
    """A reference image resized as the pipeline does: its aspect ratio, about resolution^2 pixels.

    The size is a multiple of 32, so the text encoder (one slot per 32x32 pixels) and the VAE (one
    token per 16x16 pixels) agree on the 2x2 tokens per slot without resizing it again.
    """
    assert path.is_file(), f"{path} not found"
    image = Image.open(path).convert("RGBA")
    width, height, _ = calculate_dimensions(resolution * resolution, image.width / image.height)
    return image.resize((width, height), Image.LANCZOS)


#################################################################################
#                                    MOSS-Image                                 #
#################################################################################


@dataclass
class MossReplica:
    """Everything one GPU needs to sample: the models and the cached prompt embeddings."""

    device: torch.device
    transformer: torch.nn.Module
    vae: torch.nn.Module
    # The scheduler keeps the current step index, so each concurrent sampler needs its own.
    scheduler: FlowMatchEulerDiscreteScheduler
    prompt_embeds: torch.Tensor
    negative_prompt_embeds: torch.Tensor


class MossImage:
    """MOSS-Image with the prompt and the negative prompt encoded once, one replica per GPU.

    The transformer reads latents as [B, 128, H / 16, W / 16]: the FLUX.2 VAE's 32 channels at
    1/8 resolution with each 2x2 patch folded into the channels, normalized by the VAE's batch
    norm statistics. The text tokens go after the image tokens inside the transformer.
    """

    # The q/k/v projections of the 42 joint blocks; the two text-only blocks before them are left
    # alone (their fused `qkv` does not match).
    lora_target_modules = r"layers\.\d+\.proj\.[qkv]"

    def __init__(self, args: argparse.Namespace, devices: list[torch.device]) -> None:
        assert not args.image, "MOSS-Image is text-to-image only; it takes no --image"
        assert args.resolution % 16 == 0, "--resolution must be a multiple of 16"
        self.resolution = args.resolution
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
        prompt_embeds, prompt_mask = self.pipeline.encode_prompt(args.prompt)
        negative_prompt_embeds, negative_mask = self.pipeline.encode_prompt(MOSS_NEGATIVE_PROMPT)
        # One prompt per call is never padded, so the transformer can be given no mask.
        assert prompt_mask.all()
        assert negative_mask.all()
        self.pipeline.text_encoder = None
        gc.collect()
        torch.cuda.empty_cache()
        self.pipeline.transformer.requires_grad_(False)
        self.pipeline.vae.requires_grad_(False)
        copies = [
            (
                copy.deepcopy(self.pipeline.transformer).to(device),
                copy.deepcopy(self.pipeline.vae).to(device),
            )
            for device in devices[1:]
        ]
        self.pipeline.to(DEVICE)
        self.scheduler = self.pipeline.scheduler

        self.replicas = [
            MossReplica(
                DEVICE,
                self.pipeline.transformer,
                self.pipeline.vae,
                self.scheduler,
                prompt_embeds,
                negative_prompt_embeds,
            ),
        ]
        for device, (transformer, vae) in zip(devices[1:], copies, strict=True):
            self.replicas.append(
                MossReplica(
                    device,
                    transformer,
                    vae,
                    copy.deepcopy(self.scheduler),
                    prompt_embeds.to(device),
                    negative_prompt_embeds.to(device),
                ),
            )

    @property
    def transformers(self) -> list[torch.nn.Module]:
        return [replica.transformer for replica in self.replicas]

    def enable_gradient_checkpointing(self) -> None:
        # The custom transformer does not implement diffusers' gradient checkpointing, so
        # checkpoint each joint block by hand. Without it the activations of the 42 blocks for a
        # winner/loser pair do not fit next to the weights.
        for block in self.replicas[0].transformer.layers:
            block.forward = functools.partial(
                torch.utils.checkpoint.checkpoint, block.forward, use_reentrant=False,
            )

    @torch.no_grad()
    def sample(self, replica_index: int, seed: int) -> Image.Image:
        """The pipeline's denoising loop with its default guidance, on the cached prompts."""
        replica = self.replicas[replica_index]
        size = self.resolution // 16
        channels = replica.transformer.config.latent_channels
        # The same seed gives the same noise on either GPU. The pipeline samples the noise and
        # integrates in float32.
        generator = torch.Generator(device=replica.device).manual_seed(seed)
        latents = torch.randn(
            (1, channels, size, size),
            generator=generator,
            device=replica.device,
            dtype=torch.float32,
        )
        replica.scheduler.set_timesteps(self.num_inference_steps, device=replica.device)
        for t in replica.scheduler.timesteps:
            conditional = replica.transformer(
                hidden_states=latents,
                timestep=t,
                encoder_hidden_states=replica.prompt_embeds,
                return_dict=False,
            )[0].float()
            unconditional = replica.transformer(
                hidden_states=latents,
                timestep=t,
                encoder_hidden_states=replica.negative_prompt_embeds,
                skip_layers=MOSS_UNCOND_SKIP_LAYERS,
                return_dict=False,
            )[0].float()
            velocity = unconditional + MOSS_GUIDANCE_SCALE * (conditional - unconditional)
            latents = replica.scheduler.step(velocity, t, latents, return_dict=False)[0]
        return self.decode(replica, latents)

    def decode(self, replica: MossReplica, latents: torch.Tensor) -> Image.Image:
        """Latents of one image back to a PIL image, as the pipeline does."""
        vae = replica.vae
        latents = latents.to(vae.dtype)
        mean, std = batch_norm_stats(vae)
        latents = Flux2Pipeline._unpatchify_latents(latents * std + mean)  # noqa: SLF001
        image = vae.decode(latents, return_dict=False)[0].clamp(-1, 1)
        return self.pipeline.image_processor.postprocess(image, output_type="pil")[0]

    @torch.no_grad()
    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        """Latents of generated images, the ones DPO trains on, as Flux2Pipeline encodes them."""
        vae = self.replicas[0].vae
        latents = vae.encode(to_pixel_values(images, "RGB", vae.dtype)).latent_dist.mode()
        mean, std = batch_norm_stats(vae)
        return (Flux2Pipeline._patchify_latents(latents) - mean) / std  # noqa: SLF001

    def shift_sigma(self, u: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """Shift a uniform-ish time in (0, 1) by the scheduler's fixed inference shift."""
        # Unlike Qwen-Image-2.1's, this shift does not depend on the image size.
        del latents
        shift = self.scheduler.config.shift
        return shift * u / (1 + (shift - 1) * u)

    def velocity(self, noisy_latents: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """The trained transformer's conditional prediction for a batch of noised latents."""
        replica = self.replicas[0]
        batch_size = noisy_latents.shape[0]
        return replica.transformer(
            hidden_states=noisy_latents,
            # The transformer takes the scheduler's timesteps, sigma * 1000.
            timestep=(sigma * replica.scheduler.config.num_train_timesteps).expand(batch_size),
            encoder_hidden_states=replica.prompt_embeds.repeat(batch_size, 1, 1),
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


class InteractiveDpo:
    """Holds the model, the LoRA optimizer, and one DPO update step."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        # Everything this session saves (LoRA weights, images) goes into one directory per launch.
        self.session_dir = args.results_dir / timestamp()
        self.session_dir.mkdir(parents=True)
        # One GPU per candidate at most.
        devices = [DEVICE] + [
            torch.device(f"cuda:{index}")
            for index in range(1, min(torch.cuda.device_count(), CANDIDATES_PER_ROUND))
        ]
        print(f"sampling on {[str(device) for device in devices]}")
        self.model = MODELS[args.pretrained_model](args, devices)
        # The first replica is the trained transformer itself; the others only sample.
        self.transformer = self.model.transformers[0]

        add_lora(self.transformer, args.lora_rank, self.model.lora_target_modules)
        if args.resume_lora is not None:
            self.load_lora(args.resume_lora)
        cast_training_params(self.transformer, dtype=torch.float32)
        self.model.enable_gradient_checkpointing()
        self.lora_params = [p for p in self.transformer.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            self.lora_params, lr=args.learning_rate, weight_decay=1e-2,
        )
        for transformer in self.model.transformers[1:]:
            # The same float32 adapters as the trained transformer, so both sample alike; their
            # weights are overwritten by sync_replicas.
            add_lora(transformer, args.lora_rank, self.model.lora_target_modules)
            cast_training_params(transformer, dtype=torch.float32)
            transformer.requires_grad_(False)
        self.sync_replicas()
        self.round_index = 0

    def sync_replicas(self) -> None:
        """Copy the trained LoRA weights to the sampling-only replicas on the other GPUs."""
        state_dict = get_peft_model_state_dict(self.transformer)
        for transformer in self.model.transformers[1:]:
            result = set_peft_model_state_dict(transformer, state_dict)
            assert not result.unexpected_keys, f"unexpected keys: {result.unexpected_keys}"

    def round_seeds(self) -> tuple[int, int]:
        """The two seeds of this round, one per candidate."""
        if self.args.base_seed == -1:
            random_seeds = torch.randint(0, 2**31 - 1, (2,))
            return int(random_seeds[0]), int(random_seeds[1])
        base = self.args.base_seed + 2 * self.round_index
        return base, base + 1

    @torch.no_grad()
    def generate_pair(self) -> tuple[Image.Image, Image.Image]:
        """Two samples of the current model for the same prompt, with different seeds."""
        seeds = self.round_seeds()
        print(f"round {self.round_index}: seeds={seeds}")
        start_time = time.time()
        # One thread per GPU: each candidate runs on its own GPU when there are two, and with one
        # GPU the single thread samples them one after the other.
        num_replicas = len(self.model.transformers)
        with ThreadPoolExecutor(max_workers=num_replicas) as executor:
            futures = [
                executor.submit(self.model.sample, index % num_replicas, seed)
                for index, seed in enumerate(seeds)
            ]
            images = [future.result() for future in futures]
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"round {self.round_index}: {elapsed:.2f} seconds")
        return images[0], images[1]

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
        return loss.item(), (logits > 0).float().mean().item()

    def learn_from(self, winner: Image.Image, loser: Image.Image) -> tuple[float, float]:
        """Run `steps_per_pair` DPO steps on one human-labeled pair."""
        latents = self.model.encode_images([winner, loser])
        losses, accuracies = [], []
        for _ in range(self.args.steps_per_pair):
            loss, accuracy = self.dpo_step(latents)
            losses.append(loss)
            accuracies.append(accuracy)
        self.sync_replicas()
        return sum(losses) / len(losses), sum(accuracies) / len(accuracies)

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

    def save_image(self, image: Image.Image, side: str) -> Path:
        """Save one candidate of the current round at its generated resolution."""
        path = self.session_dir / f"{timestamp()}_round{self.round_index:04d}_{side}.png"
        image.save(path)
        return path


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
#                                       GUI                                     #
#################################################################################


class DpoWindow:
    """Shows the two candidates and turns a button press into a DPO update."""

    def __init__(self, trainer: InteractiveDpo) -> None:
        self.trainer = trainer
        self.images: tuple[Image.Image, Image.Image] = None
        self.photos: list[ImageTk.PhotoImage] = []

        self.root = tk.Tk()
        self.root.title(f"Interactive Diffusion-DPO — {trainer.args.prompt}")

        self.prompt_label = tk.Label(self.root, text=trainer.args.prompt, font=("", 12))
        self.prompt_label.pack(pady=(10, 4))

        image_frame = tk.Frame(self.root)
        image_frame.pack(padx=10)
        self.canvases = [tk.Label(image_frame) for _ in range(2)]
        for index, canvas in enumerate(self.canvases):
            canvas.grid(row=0, column=index, padx=6)
        # Saves a candidate without choosing it, e.g. to keep the one that is not preferred.
        self.image_save_buttons = [
            tk.Button(
                image_frame, text="左の画像を保存", width=14, command=lambda: self.on_save_image(0),
            ),
            tk.Button(
                image_frame, text="右の画像を保存", width=14, command=lambda: self.on_save_image(1),
            ),
        ]
        for index, button in enumerate(self.image_save_buttons):
            button.grid(row=1, column=index, pady=(4, 0))

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=8)
        self.buttons = [
            tk.Button(
                button_frame,
                text="◀ 左が良い",
                width=14,
                command=lambda: self.on_choice(0),
            ),
            tk.Button(button_frame, text="引き分け / skip", width=14, command=self.on_skip),
            tk.Button(
                button_frame,
                text="右が良い ▶",
                width=14,
                command=lambda: self.on_choice(1),
            ),
            tk.Button(button_frame, text="LoRAを保存", width=14, command=self.on_save),
        ]
        for index, button in enumerate(self.buttons):
            button.grid(row=0, column=index, padx=4)
        self.buttons += self.image_save_buttons

        self.status = tk.Label(self.root, text="", font=("", 10))
        self.status.pack(pady=(0, 10))

        self.set_busy(text="loading model and sampling...")
        self.root.after(100, self.next_round)

    def set_busy(self, text: str) -> None:
        for button in self.buttons:
            button.config(state=tk.DISABLED)
        self.status.config(text=text)
        self.root.update()

    def set_ready(self, text: str) -> None:
        for button in self.buttons:
            button.config(state=tk.NORMAL)
        self.status.config(text=text)

    def next_round(self, extra: str = "") -> None:
        self.set_busy(f"round {self.trainer.round_index}: sampling... {extra}")
        self.images = self.trainer.generate_pair()
        # The full-resolution images are what gets trained on; only the display is shrunk.
        display_size = (self.trainer.args.display_size, self.trainer.args.display_size)
        self.photos = [
            ImageTk.PhotoImage(image.resize(display_size, Image.LANCZOS))
            for image in self.images
        ]
        for canvas, photo in zip(self.canvases, self.photos, strict=True):
            canvas.config(image=photo)
        self.set_ready(f"round {self.trainer.round_index}: どちらが好みですか?  {extra}")

    def on_choice(self, winner_index: int) -> None:
        self.set_busy("learning from your preference...")
        winner = self.images[winner_index]
        loser = self.images[1 - winner_index]
        path = self.trainer.save_image(winner, ("left", "right")[winner_index])
        print(f"saved the preferred image to {path}")
        loss, accuracy = self.trainer.learn_from(winner, loser)
        print(f"round {self.trainer.round_index}: loss={loss:.4f} acc={accuracy:.2f}")
        self.trainer.round_index += 1
        self.next_round(f"(last: loss={loss:.4f}, acc={accuracy:.2f})")

    def on_skip(self) -> None:
        self.trainer.round_index += 1
        self.next_round("(skipped)")

    def on_save(self) -> None:
        # Saving does not end the session: keep labeling afterwards, and save again any time.
        self.set_busy("saving LoRA weights...")
        path = self.trainer.save_lora()
        print(f"saved LoRA weights to {path}")
        self.set_ready(f"round {self.trainer.round_index}: saved {path.name}")

    def on_save_image(self, index: int) -> None:
        path = self.trainer.save_image(self.images[index], ("left", "right")[index])
        print(f"saved image to {path}")
        self.status.config(text=f"round {self.trainer.round_index}: saved {path.name}")

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    args = parse_args()
    DpoWindow(InteractiveDpo(args)).run()


if __name__ == "__main__":
    main()
