"""Interactive Diffusion-DPO with LoRA on Qwen-Image-2.1.

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

Qwen-Image-2.1 is a rectified flow model, so the noising is `x_t = (1 - sigma) * x_0 + sigma * eps`
and the target is the velocity `v = eps - x_0`. It is sampled without classifier-free guidance.

With LoRA the reference model is free: it is the same transformer with the adapters disabled
(`transformer.disable_adapters()`), so only one set of weights is held in memory.
"""

import argparse
import gc
import tkinter as tk
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import QwenImage21Pipeline
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift
from diffusers.training_utils import cast_training_params
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image, ImageTk
from safetensors.torch import load_file
from torchvision import transforms

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
PRETRAINED_MODEL = "Qwen/Qwen-Image-2.1"
TO_TENSOR = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--display_size", type=int, default=256)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--steps_per_pair", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--beta_dpo", type=float, default=2500.0)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--base_seed", type=int, default=-1)
    parser.add_argument("--resume_lora", type=Path, default=None)
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    return parser.parse_args()


#################################################################################
#                                 Model / training                              #
#################################################################################


class InteractiveDpo:
    """Holds the pipeline, the LoRA optimizer, and one DPO update step."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        # The Qwen3-VL text encoder (~17 GB) and the transformer (~14 GB) do not fit on one 24 GB
        # GPU together. Load on the CPU, run only the text encoder on the GPU for the one prompt,
        # drop it, and move the transformer and the VAE over afterwards.
        self.pipeline = QwenImage21Pipeline.from_pretrained(PRETRAINED_MODEL, dtype=DTYPE)
        self.pipeline.text_encoder.to(DEVICE)
        with torch.no_grad():
            self.prompt_embeds, prompt_embeds_mask, self.image_pad_mask = (
                self.pipeline.encode_prompt(prompt=args.prompt, device=DEVICE)
            )
        # A single prompt has no padding, for which encode_prompt returns no mask at all.
        assert prompt_embeds_mask is None
        self.pipeline.text_encoder = None
        gc.collect()
        torch.cuda.empty_cache()
        self.pipeline.to(DEVICE)
        self.pipeline.set_progress_bar_config(disable=True)
        self.transformer = self.pipeline.transformer
        self.scheduler = self.pipeline.scheduler

        self.transformer.requires_grad_(False)
        self.pipeline.vae.requires_grad_(False)
        self.transformer.add_adapter(
            LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_rank,
                init_lora_weights="gaussian",
                # The transformer is single-stream, so these are all of its attention input
                # projections.
                target_modules=["to_q", "to_k", "to_v"],
            ),
        )
        if args.resume_lora is not None:
            self.load_lora(args.resume_lora)
        cast_training_params(self.transformer, dtype=torch.float32)
        if args.gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
        self.lora_params = [p for p in self.transformer.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            self.lora_params, lr=args.learning_rate, weight_decay=1e-2,
        )
        self.round_index = 0

    def round_seeds(self) -> tuple[int, int]:
        """The two seeds of this round, one per candidate."""
        if self.args.base_seed == -1:
            random_seeds = torch.randint(0, 2**31 - 1, (2,))
            return int(random_seeds[0]), int(random_seeds[1])
        base = self.args.base_seed + 2 * self.round_index
        return base, base + 1

    def sample(self, generator: torch.Generator) -> Image.Image:
        return self.pipeline(
            prompt_embeds=self.prompt_embeds,
            true_cfg_scale=1.0,
            height=self.args.resolution,
            width=self.args.resolution,
            num_inference_steps=self.args.num_inference_steps,
            generator=generator,
        ).images[0]

    @torch.no_grad()
    def generate_pair(self) -> tuple[Image.Image, Image.Image]:
        """Two samples of the current model for the same prompt, with different seeds."""
        seeds = self.round_seeds()
        print(f"round {self.round_index}: seeds={seeds}")
        images = [
            self.sample(torch.Generator(device=DEVICE).manual_seed(seed))
            for seed in seeds
        ]
        return images[0], images[1]

    def shift_mu(self, image_seq_len: int) -> float:
        config = self.pipeline.scheduler.config
        return calculate_shift(
            image_seq_len,
            config.base_image_seq_len,
            config.max_image_seq_len,
            config.base_shift,
            config.max_shift,
        )

    def sample_sigma(self, image_seq_len: int) -> torch.Tensor:
        """One sigma in (0, 1), logit-normal and shifted the same way inference shifts it."""
        u = torch.sigmoid(
            torch.randn(1, device=DEVICE) * self.args.logit_std + self.args.logit_mean,
        )
        return self.scheduler.time_shift(self.shift_mu(image_seq_len), 1.0, u)

    def predict(
        self, noisy_latents: torch.Tensor, timestep: torch.Tensor, conditioning: dict,
    ) -> torch.Tensor:
        output = self.pipeline.transformer(
            hidden_states=noisy_latents, timestep=timestep, **conditioning, return_dict=False,
        )[0]
        # The output spans the whole joint text/image sequence; the target image is its tail.
        return output[:, -noisy_latents.shape[1]:]

    def dpo_step(self, latents: torch.Tensor, conditioning: dict) -> tuple[float, float]:
        # The winner and its loser must see identical noise and sigma.
        noise = torch.randn_like(latents[:1]).repeat(2, 1, 1)
        sigma = self.sample_sigma(latents.shape[1]).to(DTYPE)
        noisy_latents = (1.0 - sigma) * latents + sigma * noise
        # Rectified flow: the model predicts the velocity from the noise towards the data.
        target = noise - latents
        timestep = sigma.repeat(2)

        model_pred = self.predict(noisy_latents, timestep, conditioning)
        model_diff = win_minus_lose_error(model_pred, target)
        with torch.no_grad(), reference_model(self.transformer):
            ref_pred = self.predict(noisy_latents, timestep, conditioning)
            ref_diff = win_minus_lose_error(ref_pred, target)

        logits = -self.args.beta_dpo * (model_diff - ref_diff)
        loss = -F.logsigmoid(logits).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.lora_params, 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return loss.item(), (logits > 0).float().mean().item()

    @torch.no_grad()
    def encode_pair(self, winner: Image.Image, loser: Image.Image) -> tuple[torch.Tensor, dict]:
        """Packed latents of [winner ; loser] and the conditioning shared by both."""
        # The VAE reads RGBA (and the pipeline returns RGBA), with a length-1 frame axis.
        pixel_values = to_pixel_values([winner, loser]).unsqueeze(2)
        latents = self.pipeline._encode_vae_image(image=pixel_values, generator=None)  # noqa: SLF001
        _, channels, _, height, width = latents.shape
        packed = self.pipeline._pack_latents(latents, 2, channels, height, width)  # noqa: SLF001

        # As in the pipeline, the target image takes one vision-language slot per 2x2 group of
        # latent tokens, appended after the prompt.
        img_mask = torch.cat(
            [self.image_pad_mask, self.image_pad_mask.new_ones(1, height * width // 4)], dim=1,
        )
        conditioning = {
            "encoder_hidden_states": self.prompt_embeds.repeat(2, 1, 1),
            "img_shapes": [[(1, height, width)]] * 2,
            "img_mask": img_mask.repeat(2, 1),
        }
        return packed, conditioning

    def learn_from(self, winner: Image.Image, loser: Image.Image) -> tuple[float, float]:
        """Run `steps_per_pair` DPO steps on one human-labeled pair."""
        latents, conditioning = self.encode_pair(winner, loser)
        losses, accuracies = [], []
        for _ in range(self.args.steps_per_pair):
            loss, accuracy = self.dpo_step(latents, conditioning)
            losses.append(loss)
            accuracies.append(accuracy)
        return sum(losses) / len(losses), sum(accuracies) / len(accuracies)

    def load_lora(self, path: Path) -> None:
        """Restore the adapter of a previous session into the freshly added one."""
        assert path.is_file(), f"{path} not found"
        # save_lora_weights prefixes every key with the name of the model it belongs to.
        state_dict = {
            key.removeprefix("transformer."): value for key, value in load_file(str(path)).items()
        }
        result = set_peft_model_state_dict(self.transformer, state_dict)
        assert not result.unexpected_keys, f"unexpected keys in {path}: {result.unexpected_keys}"
        print(f"resumed LoRA weights from {path}")

    def save_lora(self) -> Path:
        self.args.results_dir.mkdir(parents=True, exist_ok=True)
        weight_name = f"{datetime.now(tz=UTC).astimezone().strftime('%Y%m%d_%H%M%S')}.safetensors"
        type(self.pipeline).save_lora_weights(
            str(self.args.results_dir),
            transformer_lora_layers=get_peft_model_state_dict(self.transformer),
            weight_name=weight_name,
            safe_serialization=True,
        )
        return self.args.results_dir / weight_name


def to_pixel_values(images: list[Image.Image]) -> torch.Tensor:
    """The images as one RGBA [-1, 1] batch on the GPU."""
    pixel_values = torch.stack([TO_TENSOR(image.convert("RGBA")) for image in images])
    return pixel_values.to(DEVICE, dtype=DTYPE)


@contextmanager
def reference_model(transformer: torch.nn.Module):
    """Temporarily turn the transformer back into the frozen reference model."""
    transformer.disable_adapters()
    try:
        yield
    finally:
        transformer.enable_adapters()


def win_minus_lose_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Difference of the flow-matching errors of the winner and the loser."""
    per_sample = F.mse_loss(pred.float(), target.float(), reduction="none").mean(dim=(1, 2))
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
            tk.Button(button_frame, text="保存", width=14, command=self.on_save),
        ]
        for index, button in enumerate(self.buttons):
            button.grid(row=0, column=index, padx=4)

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

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    DpoWindow(InteractiveDpo(args)).run()


if __name__ == "__main__":
    main()
