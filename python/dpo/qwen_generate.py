"""Super-resolution with Qwen-Image-2.1: re-render one image at a higher resolution.

The image is passed to the pipeline as its single condition image, with a fixed prompt asking for
the same picture with restored detail. Qwen-Image-2.1 reads it twice: the Qwen3-VL text encoder
sees its pixels next to the prompt, and the VAE turns it into latent tokens placed before the
noise. The output keeps the image's aspect ratio at about `--output_resolution`^2 pixels and is
saved next to the input with a suffix, e.g.

    uv run python dpo/qwen_generate.py --image photo.jpg --output_resolution 2048
    # -> photo_qwen2048_seed<seed>.png
"""

import argparse
from pathlib import Path

import torch
from diffusers import QwenImage21Pipeline
from PIL import Image

PROMPT = (
    "Upscale image 1 to a sharp, high-resolution version of the same picture. Keep the content, "
    "composition, colors, lighting, and every face and identity exactly the same. Only restore "
    "fine detail and crisp textures, and remove blur, noise, and compression artifacts."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--pretrained_model", type=str, default="Qwen/Qwen-Image-2.1")
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--base_seed", type=int, default=-1)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    # true_cfg_scale. Qwen-Image-2.1 is meant to be sampled without guidance (1.0).
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    # The model is native 2K, and 2048 is also the most two 24 GB GPUs hold: at 4096 the text
    # encoder alone runs out of memory. The input image is resized to this size too, before both
    # the text encoder and the VAE read it.
    parser.add_argument("--output_resolution", type=int, default=1024)
    # Cache the condition tokens' keys and values after the first step instead of recomputing
    # them every step. Faster, but at 2048 the cache alone (~10 GB) does not fit.
    parser.add_argument("--use_kv_cache", action="store_true")
    return parser.parse_args()


def image_seeds(base_seed: int, num_images: int) -> list[int]:
    if base_seed == -1:
        return [int(seed) for seed in torch.randint(0, 2**31 - 1, (num_images,))]
    return [base_seed + index for index in range(num_images)]


def main() -> None:
    args = parse_args()
    assert args.image.is_file(), f"{args.image} not found"
    reference = Image.open(args.image)

    # The Qwen3-VL text encoder (~17 GB) and the transformer (~14 GB) do not fit on one 24 GB
    # GPU together, so put them on two GPUs: the text encoder on one, the transformer and the
    # VAE on the other.
    pipeline = QwenImage21Pipeline.from_pretrained(
        args.pretrained_model, dtype=torch.bfloat16, device_map="balanced",
    )
    # The VAE shares its GPU with the transformer, and whole-image passes at 2048 do not fit
    # next to it, so encode and decode in tiles.
    pipeline.vae.enable_tiling()

    # The pipeline warns about a negative prompt it would ignore, so only pass it with CFG on.
    use_cfg = args.guidance_scale > 1.0
    negative_prompt = "" if use_cfg else None

    for seed in image_seeds(args.base_seed, args.num_images):
        image = pipeline(
            prompt=PROMPT,
            image=[reference],
            negative_prompt=negative_prompt,
            true_cfg_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            output_resolution=args.output_resolution,
            use_kv_cache=args.use_kv_cache,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).images[0]
        path = args.image.with_name(
            f"{args.image.stem}_qwen{args.output_resolution}_seed{seed}.png"
        )
        image.save(path)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
