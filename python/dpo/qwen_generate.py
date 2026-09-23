"""Image generation with Qwen-Image-2.1, from a prompt alone or from reference images.

Without `--image` this is plain text-to-image generation at `--output_resolution` square.

With `--image` it is reference-image editing, e.g. to vary the pose or situation of the same
person. The reference images are passed to the pipeline as condition images. Qwen-Image-2.1
reads them twice: the Qwen3-VL text encoder sees their pixels next to the prompt, and the VAE
turns them into latent tokens placed before the noise. The prompt then says what to change, e.g.

    uv run python dpo/qwen_generate.py --image person.png \
        --prompt "The person in image 1 is sitting on a park bench reading a book. \
Keep the face, hairstyle, and body proportions unchanged."

Several `--image` paths can be given (up to 10); refer to them as image 1, image 2, ... in the
prompt. The output size follows the aspect ratio of the last reference image, scaled to about
`--output_resolution`^2 pixels.
"""

import argparse
from datetime import UTC, datetime
from pathlib import Path

import torch
from diffusers import QwenImage21Pipeline
from PIL import Image

MAX_REFERENCE_IMAGES = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # Reference images; leave out for plain text-to-image generation.
    parser.add_argument("--image", type=Path, nargs="*", default=[])
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--pretrained_model", type=str, default="Qwen/Qwen-Image-2.1")
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--base_seed", type=int, default=-1)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    # true_cfg_scale. Qwen-Image-2.1 is meant to be sampled without guidance (1.0).
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    # The model is native 2K; 1024 is the pipeline's default and a lot cheaper.
    parser.add_argument("--output_resolution", type=int, default=1024)
    return parser.parse_args()


def image_seeds(base_seed: int, num_images: int) -> list[int]:
    if base_seed == -1:
        return [int(seed) for seed in torch.randint(0, 2**31 - 1, (num_images,))]
    return [base_seed + index for index in range(num_images)]


def main() -> None:
    args = parse_args()
    assert len(args.image) <= MAX_REFERENCE_IMAGES, (
        f"Qwen-Image-2.1 takes at most {MAX_REFERENCE_IMAGES} reference images"
    )
    for path in args.image:
        assert path.is_file(), f"{path} not found"
    # The pipeline takes None, not an empty list, for text-to-image.
    references = [Image.open(path) for path in args.image] if args.image else None

    pipeline = QwenImage21Pipeline.from_pretrained(args.pretrained_model, dtype=torch.bfloat16)
    # The Qwen3-VL text encoder (~17 GB) and the transformer (~14 GB) do not fit on one 24 GB
    # GPU together. With reference images the prompt embeddings cannot be cached and the encoder
    # dropped (as dpo_gui.py does), since the pipeline encodes the images alongside the prompt,
    # so let the offload hooks move each model onto the GPU only while it runs.
    pipeline.enable_model_cpu_offload()

    # The pipeline warns about a negative prompt it would ignore, so only pass it with CFG on.
    use_cfg = args.guidance_scale > 1.0
    negative_prompt = "" if use_cfg else None

    # One directory per run, named by its start time, so runs never overwrite each other.
    timestamp = datetime.now(tz=UTC).astimezone().strftime("%Y%m%d_%H%M%S")
    run_dir = args.results_dir / f"{timestamp}_qwen"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "prompt.txt").write_text(
        "\n".join([args.prompt, *[str(path) for path in args.image]]) + "\n",
    )
    for seed in image_seeds(args.base_seed, args.num_images):
        image = pipeline(
            prompt=args.prompt,
            image=references,
            negative_prompt=negative_prompt,
            true_cfg_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            output_resolution=args.output_resolution,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).images[0]
        path = run_dir / f"seed{seed}.png"
        image.save(path)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
