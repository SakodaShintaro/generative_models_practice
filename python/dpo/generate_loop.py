"""Sample images one after another and save each of them, without a GUI and without training.

The headless counterpart of `dpo_tier_gui.py --no_train 1`: the same models, prompt encoding and
sampling (imported from there, so both produce the same image for the same seed and reference),
with every sample written straight to one directory per launch. With `--image_dir`, every
sample is conditioned on one reference image picked from that directory uniformly at random, and
with `--resume_lora`, sampling uses the LoRA weights saved by `dpo_tier_gui.py`.

Runs until `--num_images` images are saved, or forever with -1 (stop it with Ctrl+C).
"""

import argparse
import shlex
import sys
import time
from pathlib import Path

import torch
from diffusers.training_utils import cast_training_params
from dpo_tier_gui import MODELS, MossImage, QwenImage, add_lora, timestamp
from peft.utils import set_peft_model_state_dict
from safetensors.torch import load_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="Qwen/Qwen-Image-2.1",
        choices=list(MODELS),
    )
    parser.add_argument("--image_dir", type=Path, default=None)
    # standard size
    parser.add_argument("--height", type=int, default=32 * 28)
    parser.add_argument("--width", type=int, default=32 * 21)

    # large size
    # parser.add_argument("--height", type=int, default=32 * 52)
    # parser.add_argument("--width", type=int, default=32 * 39)

    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--num_images", type=int, default=-1)
    parser.add_argument("--base_seed", type=int, default=-1)
    parser.add_argument("--resume_lora", type=Path, default=None)
    return parser.parse_args()


def load_lora(model: QwenImage | MossImage, path: Path) -> None:
    """Attach a LoRA adapter of the rank saved in `path` and load its weights for sampling."""
    assert path.is_file(), f"{path} not found"
    # dpo_tier_gui.py prefixes every key with the name of the model it belongs to.
    state_dict = {
        key.removeprefix("transformer."): value for key, value in load_file(str(path)).items()
    }
    # lora_A maps the input features down to the rank.
    rank = next(value.shape[0] for key, value in state_dict.items() if "lora_A" in key)
    add_lora(model.transformer, rank, model.lora_target_modules)
    result = set_peft_model_state_dict(model.transformer, state_dict)
    assert not result.unexpected_keys, f"unexpected keys in {path}: {result.unexpected_keys}"
    # The same float32 adapters as during training, so the samples match the GUI's.
    cast_training_params(model.transformer, dtype=torch.float32)
    model.transformer.requires_grad_(False)
    print(f"loaded LoRA weights of rank {rank} from {path}")


def save_command(args: argparse.Namespace, save_dir: Path) -> None:
    """The command line this run was launched with, and every argument it resolved to."""
    lines = [
        f"# cwd: {Path.cwd()}",
        shlex.join(sys.orig_argv),
        "",
        "# resolved arguments",
        *(f"{name}: {value}" for name, value in vars(args).items()),
    ]
    (save_dir / "command.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    assert args.num_images == -1 or args.num_images >= 1, "--num_images must be -1 or positive"
    model = MODELS[args.pretrained_model](args)
    if args.resume_lora is not None:
        load_lora(model, args.resume_lora)

    # Made only once the model has loaded, so a launch that fails early leaves nothing behind.
    save_dir = args.results_dir / timestamp()
    save_dir.mkdir(parents=True)
    save_command(args, save_dir)
    print(f"saving to {save_dir}")

    condition_names = model.condition_names
    index = 0
    while args.num_images == -1 or index < args.num_images:
        if args.base_seed == -1:
            seed = int(torch.randint(0, 2**31 - 1, (1,)))
        else:
            seed = args.base_seed + index
        # A new reference image, uniformly at random, for every sample.
        condition_index = int(torch.randint(0, len(condition_names), (1,)))
        start_time = time.time()
        image = model.sample(seed, condition_index)
        elapsed = time.time() - start_time
        name = f"{timestamp()}_{index:05d}_seed{seed}"
        if args.image_dir is not None:
            name += f"_{Path(condition_names[condition_index]).stem}"
        path = save_dir / f"{name}.png"
        image.save(path)
        print(f"sample {index}: {elapsed:.2f} seconds, saved {path.name}")
        index += 1


if __name__ == "__main__":
    main()
