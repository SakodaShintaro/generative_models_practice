"""Interactive text-to-image with Qwen-Image-2.1: prompts read from stdin, one at a time.

The models are loaded once. Then, repeatedly, a prompt line is read, followed by a line with the
number of images to generate from it (empty means 1), each with its own seed. End with Ctrl-D
(EOF).

    uv run python dpo/qwen_interactive.py
    uv run python dpo/qwen_interactive.py --text_encoder_device cuda:1  # with a second GPU

The Qwen3-VL text encoder (~17 GB) and the transformer (~14 GB) do not fit on one 24 GB GPU
together, and unlike dpo_gui.py the prompt changes every time, so the encoder cannot be dropped
after one encoding. Instead it lives in a pipeline of its own on `--text_encoder_device` (the
CPU by default, or a second GPU), while the transformer and the VAE stay resident on `--device`.
Only the prompt embeddings cross between the two.
"""

import argparse
import readline  # noqa: F401  (gives input() line editing and history)
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from diffusers import QwenImage21Pipeline

DTYPE = torch.bfloat16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str, default="Qwen/Qwen-Image-2.1")
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--text_encoder_device", type=str, default="cpu")
    # The model is native 2K; 1024 is the pipeline's default and a lot cheaper.
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    # true_cfg_scale. Qwen-Image-2.1 is meant to be sampled without guidance (1.0).
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--base_seed", type=int, default=-1)
    return parser.parse_args()


class PromptEncoder:
    """The text-encoder half of the pipeline, kept apart from the transformer and the VAE."""

    def __init__(self, pretrained_model: str, device: str) -> None:
        self.device = torch.device(device)
        self.pipeline = QwenImage21Pipeline.from_pretrained(
            pretrained_model, transformer=None, vae=None, dtype=DTYPE,
        ).to(self.device)

    @torch.no_grad()
    def __call__(self, prompt: str, target_device: torch.device) -> torch.Tensor:
        prompt_embeds, prompt_embeds_mask, _ = self.pipeline.encode_prompt(
            prompt=prompt, device=self.device,
        )
        # A single prompt has no padding, for which encode_prompt returns no mask at all.
        assert prompt_embeds_mask is None
        return prompt_embeds.to(target_device)


def read_num_images() -> int | None:
    """The number of images for the current prompt: empty means 1, None means EOF."""
    while True:
        try:
            text = input("images [1]> ").strip()
        except EOFError:
            return None
        if not text:
            return 1
        if text.isdecimal() and int(text) > 0:
            return int(text)
        print(f"not a positive integer: {text!r}")


def next_seed(base_seed: int, index: int) -> int:
    if base_seed == -1:
        return int(torch.randint(0, 2**31 - 1, (1,)))
    return base_seed + index


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    encoder = PromptEncoder(args.pretrained_model, args.text_encoder_device)
    generator_pipeline = QwenImage21Pipeline.from_pretrained(
        args.pretrained_model, text_encoder=None, dtype=DTYPE,
    ).to(device)

    # The pipeline warns about a negative prompt it would ignore, so only pass it with CFG on.
    use_cfg = args.guidance_scale > 1.0
    negative_prompt_embeds = encoder("", device) if use_cfg else None

    # One directory per session, named by its start time, so sessions never overwrite each other.
    timestamp = datetime.now(tz=UTC).astimezone().strftime("%Y%m%d_%H%M%S")
    run_dir = args.results_dir / f"{timestamp}_qwen"
    run_dir.mkdir(parents=True, exist_ok=False)
    prompt_log = run_dir / "prompts.tsv"
    print(f"saving to {run_dir}; enter a prompt, then the number of images; Ctrl-D to quit")

    index = 0
    while True:
        try:
            prompt = input("prompt> ").strip()
        except EOFError:
            print()
            break
        if not prompt:
            continue
        num_images = read_num_images()
        if num_images is None:
            print()
            break

        start = time.perf_counter()
        prompt_embeds = encoder(prompt, device)
        print(f"encoded the prompt in {time.perf_counter() - start:.1f}s")
        for _ in range(num_images):
            seed = next_seed(args.base_seed, index)
            start = time.perf_counter()
            image = generator_pipeline(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                true_cfg_scale=args.guidance_scale,
                height=args.resolution,
                width=args.resolution,
                num_inference_steps=args.num_inference_steps,
                generator=torch.Generator(device="cpu").manual_seed(seed),
            ).images[0]

            path = run_dir / f"{index:04d}_seed{seed}.png"
            image.save(path)
            with prompt_log.open("a") as log:
                log.write(f"{path.name}\t{prompt}\n")
            print(f"saved {path} ({time.perf_counter() - start:.1f}s)")
            index += 1


if __name__ == "__main__":
    main()
