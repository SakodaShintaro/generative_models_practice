"""Interactive image generation with Qwen-Image-2.1 in a small GUI.

The models are loaded once, in the background, and a window lets you set the prompt, an optional
condition image and the sampling settings, then generate as many times as you like.

- With an image, it is passed to the pipeline as its single condition image, e.g. for
  super-resolution with a prompt saying to render the same picture with restored detail.
  Qwen-Image-2.1 reads the image twice: the Qwen3-VL text encoder sees its pixels next to the
  prompt, and the VAE turns it into latent tokens placed before the noise. The output keeps the
  image's aspect ratio and is `scale` times the input on each side (rounded to a multiple of 32).
- Without an image it is plain text-to-image at `resolution` x `resolution`.

Nothing is saved on its own. Only the latest generated image is kept; save it with
"名前をつけて保存", which suggests a name like
`<stem>_qwen<scale>x_seed<seed>.png` (with an image) or `qwen_<time>_seed<seed>.png` (without).

    uv run python dpo/qwen_generate.py
"""

import argparse
import math
import queue
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tkinter import filedialog

import torch
from diffusers import QwenImage21Pipeline
from PIL import Image, ImageTk

POLL_MILLISECONDS = 100
PREVIEW_SIZE = 512
INPUT_THUMBNAIL_SIZE = 160


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str, default="Qwen/Qwen-Image-2.1")
    return parser.parse_args()


@dataclass(frozen=True)
class Request:
    """One press of the generate button, with every setting already parsed."""

    prompt: str
    image_path: Path | None
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


class Worker:
    """Owns the pipeline and runs requests one after another on its own thread."""

    def __init__(self, pretrained_model: str) -> None:
        self.pretrained_model = pretrained_model
        self.requests: queue.Queue[Request] = queue.Queue()
        # Status lines and finished images, for the GUI thread to pick up.
        self.statuses: queue.Queue[str] = queue.Queue()
        self.images: queue.Queue[Generated] = queue.Queue()
        self.busy = threading.Event()
        self.busy.set()

    def run(self) -> None:
        self.statuses.put("loading the model...")
        # The Qwen3-VL text encoder (~17 GB) and the transformer (~14 GB) do not fit on one 24 GB
        # GPU together, so put them on two GPUs: the text encoder on one, the transformer and the
        # VAE on the other.
        pipeline = QwenImage21Pipeline.from_pretrained(
            self.pretrained_model, dtype=torch.bfloat16, device_map="balanced",
        )
        # The VAE shares its GPU with the transformer, and whole-image passes at 2048 do not fit
        # next to it, so encode and decode in tiles.
        pipeline.vae.enable_tiling()
        self.statuses.put("ready")
        self.busy.clear()

        while True:
            request = self.requests.get()
            try:
                self.generate(pipeline, request)
            except Exception as error:  # noqa: BLE001  (keep the GUI alive, e.g. after an OOM)
                self.statuses.put(f"error: {error!r}")
                torch.cuda.empty_cache()
            self.busy.clear()

    def generate(self, pipeline: QwenImage21Pipeline, request: Request) -> None:
        if request.image_path is None:
            images = None
            output_resolution = request.resolution
            timestamp = datetime.now(tz=UTC).astimezone().strftime("%Y%m%d_%H%M%S")
            stem = f"qwen_{timestamp}"
        else:
            reference = Image.open(request.image_path).convert("RGB")
            width, height = reference.size
            images = [reference]
            # The pipeline keeps the input's aspect ratio and makes the output area
            # output_resolution^2, so a side of sqrt(width * height) * scale scales both sides
            # by `scale`.
            output_resolution = round(math.sqrt(width * height) * request.scale)
            stem = f"{request.image_path.stem}_qwen{request.scale:g}x"

        # The pipeline warns about a negative prompt it would ignore, so only pass it with CFG on.
        use_cfg = request.guidance_scale > 1.0
        negative_prompt = "" if use_cfg else None

        seed = resolve_seed(request.seed)
        self.statuses.put(
            f"generating (seed {seed}, output_resolution {output_resolution})...",
        )
        start = time.perf_counter()
        image = pipeline(
            prompt=request.prompt,
            image=images,
            negative_prompt=negative_prompt,
            true_cfg_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            output_resolution=output_resolution,
            use_kv_cache=request.use_kv_cache,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).images[0]
        self.images.put(Generated(image, seed, f"{stem}_seed{seed}.png"))
        self.statuses.put(
            f"generated (seed {seed}, {image.width}x{image.height}, "
            f"{time.perf_counter() - start:.1f}s)",
        )

class App:
    def __init__(self, worker: Worker) -> None:
        self.worker = worker
        self.input_photo: ImageTk.PhotoImage | None = None
        self.output_photo: ImageTk.PhotoImage | None = None
        self.latest: Generated | None = None
        # Where the save dialog opens; follows the last save, starting at the current directory.
        self.save_dir = Path.cwd()

        self.root = tk.Tk()
        self.root.title("Qwen-Image-2.1")

        form = tk.Frame(self.root)
        form.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        tk.Label(form, text="プロンプト").grid(row=0, column=0, sticky=tk.NW)
        self.prompt = tk.Text(form, width=50, height=8, wrap=tk.WORD, undo=True)
        self.prompt.grid(row=0, column=1, columnspan=2, sticky=tk.EW, pady=2)

        self.image_path = tk.StringVar(value="")
        tk.Label(form, text="画像 (空なら t2i)").grid(row=1, column=0, sticky=tk.W)
        image_entry = tk.Entry(form, textvariable=self.image_path, width=40)
        image_entry.grid(row=1, column=1, sticky=tk.EW, pady=2)
        image_buttons = tk.Frame(form)
        image_buttons.grid(row=1, column=2, sticky=tk.W)
        tk.Button(image_buttons, text="参照", command=self.on_browse_image).pack(side=tk.LEFT)
        tk.Button(image_buttons, text="クリア", command=self.on_clear_image).pack(side=tk.LEFT)
        self.image_path.trace_add("write", lambda *_: self.update_input_thumbnail())

        self.input_thumbnail = tk.Label(form, text="(画像なし)", relief=tk.SUNKEN)
        self.input_thumbnail.grid(row=2, column=1, sticky=tk.W, pady=2)

        # Kept as text and parsed when the generate button is pressed.
        self.scale = self.add_field(form, 3, "scale (画像あり)", "1.0")
        self.resolution = self.add_field(form, 4, "resolution (画像なし)", "1024")
        self.seed = self.add_field(form, 5, "seed (-1 でランダム)", "-1")
        self.num_inference_steps = self.add_field(form, 6, "steps", "40")
        self.guidance_scale = self.add_field(form, 7, "guidance scale (true CFG)", "1.0")
        self.use_kv_cache = tk.BooleanVar(value=False)
        tk.Checkbutton(form, text="use_kv_cache", variable=self.use_kv_cache).grid(
            row=8, column=1, sticky=tk.W, pady=2,
        )

        self.generate_button = tk.Button(
            form, text="生成", width=14, command=self.on_generate, state=tk.DISABLED,
        )
        self.generate_button.grid(row=9, column=1, sticky=tk.W, pady=(10, 2))

        self.status = tk.Label(
            form, text="", anchor=tk.W, justify=tk.LEFT, wraplength=520,
        )
        self.status.grid(row=10, column=0, columnspan=3, sticky=tk.EW, pady=(6, 0))

        self.build_output_panel()

        self.root.bind("<Control-Return>", lambda _: self.on_generate())
        self.root.bind("<Control-s>", lambda _: self.on_save())
        self.root.after(POLL_MILLISECONDS, self.poll)

    def build_output_panel(self) -> None:
        """The preview of the latest generated image, with a button to save it."""
        output_frame = tk.Frame(self.root)
        output_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.output = tk.Label(
            output_frame, text="(出力)", width=PREVIEW_SIZE // 8, height=PREVIEW_SIZE // 16,
            relief=tk.SUNKEN,
        )
        self.output.pack(side=tk.TOP)
        output_buttons = tk.Frame(output_frame)
        output_buttons.pack(side=tk.TOP, pady=(6, 0))
        self.output_info = tk.Label(output_buttons, text="", width=24)
        self.output_info.pack(side=tk.LEFT)
        self.save_button = tk.Button(
            output_buttons, text="名前をつけて保存", command=self.on_save, state=tk.DISABLED,
        )
        self.save_button.pack(side=tk.LEFT, padx=(12, 0))

    @staticmethod
    def add_field(form: tk.Frame, row: int, label: str, initial: str) -> tk.StringVar:
        variable = tk.StringVar(value=initial)
        tk.Label(form, text=label).grid(row=row, column=0, sticky=tk.W)
        tk.Entry(form, textvariable=variable, width=12).grid(
            row=row, column=1, sticky=tk.W, pady=2,
        )
        return variable

    def on_browse_image(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("images", "*.png *.jpg *.jpeg *.webp *.bmp"), ("all", "*")],
        )
        # The dialog returns an empty string (or an empty tuple) when cancelled.
        if path:
            self.image_path.set(path)

    def on_clear_image(self) -> None:
        self.image_path.set("")

    def update_input_thumbnail(self) -> None:
        text = self.image_path.get().strip()
        path = Path(text)
        if not text or not path.is_file():
            self.input_photo = None
            message = "(見つかりません)" if text else "(画像なし)"
            self.input_thumbnail.config(image="", text=message)
            return
        image = Image.open(path)
        image.thumbnail((INPUT_THUMBNAIL_SIZE, INPUT_THUMBNAIL_SIZE))
        self.input_photo = ImageTk.PhotoImage(image)
        self.input_thumbnail.config(image=self.input_photo, text="")

    def parse_request(self) -> Request | str:
        """The request in the form, or a message saying what is wrong with it."""
        prompt = self.prompt.get("1.0", tk.END).strip()
        image_text = self.image_path.get().strip()
        image_path = Path(image_text) if image_text else None
        try:
            request = Request(
                prompt=prompt,
                image_path=image_path,
                scale=float(self.scale.get()),
                resolution=int(self.resolution.get()),
                seed=int(self.seed.get()),
                num_inference_steps=int(self.num_inference_steps.get()),
                guidance_scale=float(self.guidance_scale.get()),
                use_kv_cache=self.use_kv_cache.get(),
            )
        except ValueError as error:
            return f"数値として読めない設定があります: {error}"
        # (is wrong, message), checked in order.
        checks = [
            (not prompt, "プロンプトが空です"),
            (
                image_path is not None and not image_path.is_file(),
                f"画像が見つかりません: {image_path}",
            ),
            (request.scale <= 0, "scale は正の値にしてください"),
            (request.resolution <= 0, "resolution は正の整数にしてください"),
            (request.num_inference_steps <= 0, "steps は正の整数にしてください"),
            (request.seed < -1, "seed は -1 か 0 以上にしてください"),
        ]
        for is_wrong, message in checks:
            if is_wrong:
                return message
        return request

    def on_generate(self) -> None:
        if self.worker.busy.is_set():
            return
        request = self.parse_request()
        if isinstance(request, str):
            self.status.config(text=request)
            return
        self.worker.busy.set()
        self.generate_button.config(state=tk.DISABLED)
        self.worker.requests.put(request)

    def show(self, generated: Generated) -> None:
        """Make `generated` the latest image and show it."""
        self.latest = generated
        image = generated.image.copy()
        image.thumbnail((PREVIEW_SIZE, PREVIEW_SIZE))
        self.output_photo = ImageTk.PhotoImage(image)
        self.output.config(image=self.output_photo, text="", width=0, height=0)
        self.output_info.config(text=f"seed {generated.seed}")
        self.save_button.config(state=tk.NORMAL)

    def on_save(self) -> None:
        if self.latest is None:
            return
        generated = self.latest
        path = filedialog.asksaveasfilename(
            initialdir=self.save_dir,
            initialfile=generated.suggested_name,
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg"), ("WebP", "*.webp")],
        )
        # The dialog returns an empty string (or an empty tuple) when cancelled.
        if not path:
            return
        path = Path(path)
        try:
            generated.image.save(path)
        except (OSError, ValueError) as error:
            self.status.config(text=f"保存に失敗しました: {error!r}")
            return
        self.save_dir = path.parent
        self.status.config(text=f"saved {path}")

    def poll(self) -> None:
        """Show the worker's latest status and image."""
        while not self.worker.statuses.empty():
            self.status.config(text=self.worker.statuses.get())
        while not self.worker.images.empty():
            self.show(self.worker.images.get())
        state = tk.DISABLED if self.worker.busy.is_set() else tk.NORMAL
        self.generate_button.config(state=state)
        self.root.after(POLL_MILLISECONDS, self.poll)


def main() -> None:
    args = parse_args()
    worker = Worker(args.pretrained_model)
    app = App(worker)
    threading.Thread(target=worker.run, daemon=True).start()
    app.root.mainloop()


if __name__ == "__main__":
    main()
