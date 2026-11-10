import argparse
import math
import subprocess
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from queue import Empty, Queue

import gradio as gr
import torch
from diffusers import QwenImage21Pipeline
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str, default="Qwen/Qwen-Image-2.1")
    parser.add_argument("--port", type=int, default=7861)
    return parser.parse_args()


# Height (px) of the image boxes in the page
IMAGE_HEIGHT = 512
# How often the status line is refreshed while generating
STATUS_SECONDS = 0.5


@dataclass(frozen=True)
class Request:
    """One press of the generate button, with every setting already checked."""

    prompt: str
    # Condition images in order; empty for text-to-image.
    image_paths: list[Path]
    scale: float
    resolution: int
    seed: int
    num_inference_steps: int
    guidance_scale: float
    use_kv_cache: bool


@dataclass(frozen=True)
class Generated:
    """One generated image and the file name suggested when saving it."""

    image: Image.Image
    seed: int
    suggested_name: str


def resolve_seed(seed: int) -> int:
    """`seed` itself, or a random one for -1."""
    if seed == -1:
        return int(torch.randint(0, 2**31 - 1, (1,)))
    return seed


def load_pipeline(pretrained_model: str) -> QwenImage21Pipeline:
    # The Qwen3-VL text encoder (~17 GB) and the transformer (~14 GB) do not fit on one 24 GB GPU
    # together, so put them on two GPUs: the text encoder on one, the transformer and the VAE on
    # the other.
    pipeline = QwenImage21Pipeline.from_pretrained(
        pretrained_model, dtype=torch.bfloat16, device_map="balanced"
    )
    # The VAE shares its GPU with the transformer, and whole-image passes at 2048 do not fit next
    # to it, so encode and decode in tiles.
    pipeline.vae.enable_tiling()
    return pipeline


def check_request(
    prompt: str,
    gallery: list[tuple[str, str | None]] | None,
    scale: float,
    resolution: float,
    seed: float,
    num_inference_steps: float,
    guidance_scale: float,
    use_kv_cache: bool,
) -> Request:
    """The request in the form; raises gr.Error saying what is wrong with it."""
    # (is wrong, message), checked in order.
    checks = [
        (not prompt.strip(), "プロンプトが空です"),
        (scale <= 0, "scale は正の値にしてください"),
        (resolution <= 0, "resolution は正の整数にしてください"),
        (num_inference_steps <= 0, "steps は正の整数にしてください"),
        (seed < -1, "seed は -1 か 0 以上にしてください"),
    ]
    for is_wrong, message in checks:
        if is_wrong:
            raise gr.Error(message)
    return Request(
        prompt=prompt.strip(),
        image_paths=gallery_paths(gallery),
        scale=scale,
        resolution=int(resolution),
        seed=int(seed),
        num_inference_steps=int(num_inference_steps),
        guidance_scale=guidance_scale,
        use_kv_cache=use_kv_cache,
    )


def gallery_paths(gallery: list[tuple[str, str | None]] | None) -> list[Path]:
    """The image paths in a gr.Gallery value, in order (it is None when empty)."""
    if gallery is None:
        return []
    return [Path(path) for path, _caption in gallery]


def describe_output(
    gallery: list[tuple[str, str | None]] | None, scale: float, resolution: float
) -> str:
    """The output size for the current form, shown under the settings."""
    paths = gallery_paths(gallery)
    if not paths:
        return f"出力: {int(resolution)}x{int(resolution)} (t2i)"
    with Image.open(paths[-1]) as image:
        width, height = image.size
    # Approximate: the pipeline picks the exact size, keeping the last image's aspect ratio.
    return (
        f"出力: 約 {round(width * scale)}x{round(height * scale)} "
        f"(画像 {len(paths)} 枚, 最後の画像 {width}x{height})"
    )


def save_image(generated: Generated | None) -> str:
    """Ask where to save the latest image with a native dialog, then save it there.

    The dialog (zenity) opens on the machine running this script, which is also where the
    browser is when the page is opened locally.
    """
    if generated is None:
        message = "保存する画像がありません"
        raise gr.Error(message)
    result = subprocess.run(  # noqa: S603
        [  # noqa: S607
            "zenity",
            "--file-selection",
            "--save",
            "--confirm-overwrite",
            "--title=画像を保存",
            f"--filename={Path.cwd() / generated.suggested_name}",
            "--file-filter=画像 | *.png *.jpg *.jpeg *.webp",
            "--file-filter=すべて | *",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    # zenity exits with 1 when cancelled.
    if result.returncode == 1:
        return "保存をキャンセルしました"
    assert result.returncode == 0, result.stderr
    path = Path(result.stdout.strip())
    if path.suffix == "":
        path = path.with_suffix(".png")
    try:
        generated.image.save(path)
    except (OSError, ValueError) as error:
        message = f"保存に失敗しました: {error!r}"
        raise gr.Error(message) from error
    return f"保存しました: {path}"


class App:
    """The Gradio page, running each generation on `pipeline`."""

    def __init__(self, pipeline: QwenImage21Pipeline) -> None:
        self.pipeline = pipeline
        # The pipeline runs here, so the page handler can report its progress meanwhile.
        self.executor = ThreadPoolExecutor(max_workers=1)

    def run_pipeline(self, request: Request, seed: int, steps: Queue[int]) -> Image.Image:
        if not request.image_paths:
            images = None
            output_resolution = request.resolution
        else:
            images = [Image.open(path).convert("RGB") for path in request.image_paths]
            width, height = images[-1].size
            # The pipeline keeps the last image's aspect ratio and makes the output area
            # output_resolution^2, so a side of sqrt(width * height) * scale scales both sides of
            # the last image by `scale`.
            output_resolution = round(math.sqrt(width * height) * request.scale)

        # The pipeline warns about a negative prompt it would ignore, so only pass it with CFG on.
        use_cfg = request.guidance_scale > 1.0
        negative_prompt = "" if use_cfg else None

        def on_step_end(_pipeline: QwenImage21Pipeline, step: int, _t: int, _kwargs: dict) -> dict:
            steps.put(step + 1)
            # Nothing to replace in the denoising state.
            return {}

        return self.pipeline(
            prompt=request.prompt,
            image=images,
            negative_prompt=negative_prompt,
            true_cfg_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            output_resolution=output_resolution,
            use_kv_cache=request.use_kv_cache,
            generator=torch.Generator(device="cpu").manual_seed(seed),
            callback_on_step_end=on_step_end,
        ).images[0]

    def on_generate(
        self,
        prompt: str,
        gallery: list[tuple[str, str | None]] | None,
        scale: float,
        resolution: float,
        seed: float,
        num_inference_steps: float,
        guidance_scale: float,
        use_kv_cache: bool,
    ) -> Iterator[tuple]:
        """Yield (status, image, latest) updates while generating."""
        request = check_request(
            prompt,
            gallery,
            scale,
            resolution,
            seed,
            num_inference_steps,
            guidance_scale,
            use_kv_cache,
        )
        seed_value = resolve_seed(request.seed)
        if not request.image_paths:
            timestamp = datetime.now(tz=UTC).astimezone().strftime("%Y%m%d_%H%M%S")
            stem = f"qwen_{timestamp}"
        else:
            stem = f"{request.image_paths[-1].stem}_qwen{request.scale:g}x"

        steps: Queue[int] = Queue()
        start = time.perf_counter()
        future = self.executor.submit(self.run_pipeline, request, seed_value, steps)
        total = request.num_inference_steps
        phase = "エンコード中"
        while not future.done():
            try:
                step = steps.get(timeout=STATUS_SECONDS)
            except Empty:
                step = None
            if step is not None:
                phase = f"ステップ {step}/{total}" if step < total else "デコード中"
            elapsed = time.perf_counter() - start
            yield f"生成中… {phase} (seed {seed_value}, {elapsed:.0f} 秒)", gr.skip(), gr.skip()

        try:
            image = future.result()
        except Exception as error:  # e.g. an OOM; keep the page alive
            torch.cuda.empty_cache()
            message = f"生成に失敗しました: {error!r}"
            raise gr.Error(message) from error
        generated = Generated(image, seed_value, f"{stem}_seed{seed_value}.png")
        elapsed = time.perf_counter() - start
        status = f"完了 (seed {seed_value}, {image.width}x{image.height}, {elapsed:.1f} 秒)"
        yield status, image, generated

    def build(self) -> gr.Blocks:
        with gr.Blocks(title="Qwen-Image-2.1") as demo:
            gr.Markdown("## Qwen-Image-2.1")
            # The latest generated image, for the save button.
            latest = gr.State(None)
            with gr.Row():
                with gr.Column(scale=1):
                    images = gr.Gallery(
                        label="画像 (複数可、空なら t2i。並び順が image1, image2, …。"
                        "出力の縦横比と scale は最後の画像が基準)",
                        type="filepath",
                        interactive=True,
                        file_types=["image"],
                        columns=4,
                        height=IMAGE_HEIGHT // 2,
                    )
                    prompt = gr.Textbox(label="プロンプト", lines=8)
                    with gr.Row():
                        scale = gr.Slider(0.25, 4.0, value=1.0, step=0.05, label="scale (画像あり)")
                        resolution = gr.Slider(
                            256, 2048, value=1024, step=32, label="resolution (画像なし)"
                        )
                    output_info = gr.Markdown(describe_output(None, 1.0, 1024))
                    with gr.Accordion("詳細設定", open=False):
                        with gr.Row():
                            num_inference_steps = gr.Slider(1, 100, value=40, step=1, label="steps")
                            guidance_scale = gr.Slider(
                                1.0, 10.0, value=1.0, step=0.1, label="guidance scale (true CFG)"
                            )
                        with gr.Row():
                            seed = gr.Number(value=-1, precision=0, label="seed (-1 でランダム)")
                            use_kv_cache = gr.Checkbox(value=False, label="use_kv_cache")
                    generate_button = gr.Button("生成", variant="primary", size="lg")
                with gr.Column(scale=1):
                    output = gr.Image(
                        label="出力", type="pil", interactive=False, height=IMAGE_HEIGHT
                    )
                    status = gr.Markdown("")
                    save_button = gr.Button("保存…")

            inputs = [
                prompt,
                images,
                scale,
                resolution,
                seed,
                num_inference_steps,
                guidance_scale,
                use_kv_cache,
            ]
            generate_button.click(self.on_generate, inputs, [status, output, latest])
            save_button.click(save_image, latest, status)
            for component in (images, scale, resolution):
                component.change(describe_output, [images, scale, resolution], output_info)
        return demo


def main() -> None:
    args = parse_args()
    print("loading the model...")
    pipeline = load_pipeline(args.pretrained_model)
    print("ready")
    demo = App(pipeline).build()
    # One generation at a time: the pipeline uses both GPUs.
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        inbrowser=True,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
