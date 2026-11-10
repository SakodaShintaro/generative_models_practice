import argparse
import json
import secrets
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import gradio as gr
from PIL import Image
from websockets.sync.client import connect


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--comfy_port", type=int, default=8189)
    return parser.parse_args()


SCRIPT_DIR = Path(__file__).resolve().parent
COMFY_DIR = SCRIPT_DIR / "ComfyUI"
RESULTS_DIR = SCRIPT_DIR / "results"

# MiniMaxH3ImageToVideo requires width/height to be multiples of 32
CANVAS_MULTIPLE = 32
FPS = 24

UNET_NAME = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
CLIP_NAME = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE_NAME = "minimax_h3_video_vae_int8_convrot.safetensors"
AUDIO_VAE_NAME = "minimax_h3_audio_vae_fp32.safetensors"
LORA_NAME = "minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors"

# beta: recommended by drbaph (the pruned LoRA's converter), simple: by larryvrh (its author)
SCHEDULERS = ["beta", "simple"]

COMFY_POLL_SECONDS = 1.0
# Height (px) of the output player in the page
VIDEO_HEIGHT = 480

# What each node of build_workflow's graph is doing, for the status line.
NODE_PHASES = {
    "unet": "モデル読込",
    "lora": "LoRA 適用",
    "clip": "テキストエンコーダ読込",
    "video_vae": "VAE 読込",
    "audio_vae": "VAE 読込",
    "image": "画像読込",
    "i2v": "プロンプト・画像エンコード",
    "noise": "サンプリング準備",
    "guider": "サンプリング準備",
    "sampler": "サンプリング準備",
    "sigmas": "サンプリング準備",
    "sample": "サンプリング",
    "decode_video": "動画デコード",
    "decode_audio": "音声デコード",
    "create_video": "動画作成",
    "save": "保存",
}
COMFY_STARTUP_TIMEOUT_SECONDS = 300.0


@dataclass(frozen=True)
class Request:
    """One press of the generate button, with every setting already checked."""

    prompt: str
    image_path: Path
    scale: float
    seconds: float
    steps: int
    scheduler: str
    lora_strength: float
    seed: int


def resolve_seed(seed: int) -> int:
    """`seed` itself, or a random one for -1."""
    if seed == -1:
        return secrets.randbelow(2**32)
    return seed


def round_to_multiple(value: float, multiple: int) -> int:
    return max(multiple, round(value / multiple) * multiple)


def snap_length(seconds: float) -> int:
    """Convert seconds at 24 fps to the 17k+5 frame grid (same formula as the official template)."""
    n = max(5, round(seconds * FPS))
    return n + (5 - (n % 17)) % 17


def output_size(image_path: Path, scale: float) -> tuple[int, int]:
    with Image.open(image_path) as image:
        image_width, image_height = image.size
    width = round_to_multiple(image_width * scale, CANVAS_MULTIPLE)
    height = round_to_multiple(image_height * scale, CANVAS_MULTIPLE)
    return width, height


def build_workflow(
    image_name: str,
    prompt: str,
    width: int,
    height: int,
    length: int,
    steps: int,
    scheduler: str,
    lora_strength: float,
    seed: int,
    output_prefix: str,
) -> dict:
    """The API-format ComfyUI graph: the official I2V template with the Turbo LoRA."""
    return {
        "unet": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": UNET_NAME, "weight_dtype": "default"},
        },
        "lora": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["unet", 0],
                "lora_name": LORA_NAME,
                "strength_model": lora_strength,
            },
        },
        "clip": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": CLIP_NAME, "type": "minimax", "device": "default"},
        },
        "video_vae": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE_NAME}},
        "audio_vae": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE_NAME}},
        "image": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "i2v": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {
                "clip": ["clip", 0],
                "vae": ["video_vae", 0],
                "prompt": prompt,
                "width": width,
                "height": height,
                "length": length,
                "first_frame": ["image", 0],
            },
        },
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "guider": {
            "class_type": "BasicGuider",
            "inputs": {"model": ["lora", 0], "conditioning": ["i2v", 0]},
        },
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "sigmas": {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": ["lora", 0],
                "scheduler": scheduler,
                "steps": steps,
                "denoise": 1.0,
            },
        },
        "sample": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["noise", 0],
                "guider": ["guider", 0],
                "sampler": ["sampler", 0],
                "sigmas": ["sigmas", 0],
                "latent_image": ["i2v", 1],
            },
        },
        "decode_video": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["sample", 0], "vae": ["video_vae", 0]},
        },
        "decode_audio": {
            "class_type": "VAEDecodeAudio",
            "inputs": {"samples": ["sample", 0], "vae": ["audio_vae", 0]},
        },
        "create_video": {
            "class_type": "CreateVideo",
            "inputs": {"images": ["decode_video", 0], "audio": ["decode_audio", 0], "fps": FPS},
        },
        "save": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["create_video", 0],
                # Unique per run, so SaveVideo runs again even when everything upstream is cached.
                "filename_prefix": output_prefix,
                "format": "auto",
                "codec": "auto",
            },
        },
    }


def http_json(url: str, payload: dict | None) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers)  # noqa: S310
    with urllib.request.urlopen(req) as res:  # noqa: S310 (localhost only)
        return json.loads(res.read())


def copy_to_comfy_input(path: Path, name: str) -> str:
    """Copy `path` into ComfyUI/input (where LoadImage reads) as `name` + its suffix."""
    input_dir = COMFY_DIR / "input"
    input_dir.mkdir(exist_ok=True)
    # LoadImage hashes the file, so overwriting the same name still invalidates its cache.
    file_name = f"{name}{path.suffix}"
    shutil.copy(path, input_dir / file_name)
    return file_name


def execution_error_message(status: dict) -> str:
    """The exception message in a failed ComfyUI history status, or the whole status."""
    for message_type, data in status["messages"]:
        if message_type == "execution_error":
            return f"{data['node_type']}: {data['exception_message']}"
    return json.dumps(status)


class ComfyServer:
    """A headless ComfyUI process and the requests sent to it."""

    def __init__(self, port: int, work_dir: Path) -> None:
        self.base_url = f"http://127.0.0.1:{port}"
        self.port = port
        # Where generated videos are kept until the app exits.
        self.work_dir = work_dir
        self.process: subprocess.Popen | None = None

    def start(self) -> None:
        cmd = [
            str(SCRIPT_DIR / ".venv/bin/python"),
            "main.py",
            "--listen",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--disable-auto-launch",
            "--use-sage-attention",
        ]
        # ComfyUI's stdout/stderr go straight to the terminal (no log file).
        self.process = subprocess.Popen(cmd, cwd=COMFY_DIR)  # noqa: S603
        start = time.perf_counter()
        while time.perf_counter() - start < COMFY_STARTUP_TIMEOUT_SECONDS:
            assert self.process.poll() is None, "ComfyUI exited during startup (see terminal)"
            try:
                http_json(f"{self.base_url}/system_stats", None)
            except (urllib.error.URLError, ConnectionError):
                time.sleep(COMFY_POLL_SECONDS)
                continue
            return
        msg = "ComfyUI did not become ready in time"
        raise TimeoutError(msg)

    def stop(self) -> None:
        if self.process is not None:
            self.process.terminate()
            self.process.wait()

    def run_prompt(
        self, workflow: dict, description: str, start: float
    ) -> Generator[str, None, dict]:
        """Run `workflow`, yielding status lines with the step progress; return its history."""
        assert self.process is not None
        client_id = str(uuid.uuid4())
        # ComfyUI reports the running node and its step progress only over its websocket.
        with connect(f"ws://127.0.0.1:{self.port}/ws?clientId={client_id}") as websocket:
            res = http_json(f"{self.base_url}/prompt", {"prompt": workflow, "client_id": client_id})
            assert "prompt_id" in res, res
            prompt_id = res["prompt_id"]
            node_phase = "待機中"
            phase = node_phase
            while True:
                elapsed = time.perf_counter() - start
                yield f"生成中… {phase} ({description}, {elapsed:.0f} 秒)"
                assert self.process.poll() is None, "ComfyUI exited during generation"
                try:
                    raw = websocket.recv(timeout=COMFY_POLL_SECONDS)
                except TimeoutError:
                    continue
                # Binary messages are latent previews; skip them.
                if not isinstance(raw, str):
                    continue
                message = json.loads(raw)
                if message["type"] not in ("executing", "progress"):
                    continue
                data = message["data"]
                if data["prompt_id"] != prompt_id:
                    continue
                if message["type"] == "executing":
                    # node None: the prompt has finished (successfully or not)
                    if data["node"] is None:
                        break
                    node_phase = NODE_PHASES[data["node"]]
                    phase = node_phase
                else:
                    # Progress belongs to the node last reported as executing.
                    phase = f"{node_phase} {data['value']}/{data['max']}"

        # ComfyUI writes the history before it reports the end.
        history = http_json(f"{self.base_url}/history/{prompt_id}", None)
        return history[prompt_id]

    def generate(self, request: Request) -> Iterator[str | Path]:
        """Yield status lines while generating, then the path of the generated video."""
        assert self.process is not None
        seed = resolve_seed(request.seed)
        width, height = output_size(request.image_path, request.scale)
        length = snap_length(request.seconds)

        timestamp = datetime.now(tz=UTC).astimezone().strftime("%Y%m%d_%H%M%S")
        stem = (
            f"{timestamp}_{request.image_path.stem}_{width}x{height}_"
            f"{request.seconds:g}s_{request.steps}step_{request.scheduler}_seed{seed}"
        )
        image_name = copy_to_comfy_input(request.image_path, "run_i2v_image")
        workflow = build_workflow(
            image_name=image_name,
            prompt=request.prompt,
            width=width,
            height=height,
            length=length,
            steps=request.steps,
            scheduler=request.scheduler,
            lora_strength=request.lora_strength,
            seed=seed,
            output_prefix=f"run_i2v/{stem}",
        )
        description = f"seed {seed}, {width}x{height}, {length} frames"
        start = time.perf_counter()
        entry = yield from self.run_prompt(workflow, description, start)
        status = entry["status"]
        assert status["status_str"] == "success", execution_error_message(status)

        saved = [
            COMFY_DIR / "output" / item["subfolder"] / item["filename"]
            for node_output in entry["outputs"].values()
            for item in node_output["images"]
        ]
        assert len(saved) == 1, saved
        video_path = self.work_dir / f"{stem}{saved[0].suffix}"
        shutil.move(saved[0], video_path)
        yield f"完了 ({description}, {time.perf_counter() - start:.1f} 秒)"
        yield video_path


def check_request(
    prompt: str,
    image_path: str | None,
    scale: float,
    seconds: float,
    steps: float,
    scheduler: str,
    lora_strength: float,
    seed: float,
) -> Request:
    """The request in the form; raises gr.Error saying what is wrong with it."""
    # (is wrong, message), checked in order.
    checks = [
        (not prompt.strip(), "プロンプトが空です"),
        (image_path is None, "画像を指定してください"),
        (scale <= 0, "scale は正の値にしてください"),
        (seconds <= 0, "秒数は正の値にしてください"),
        (steps <= 0, "steps は正の整数にしてください"),
        (seed < -1, "seed は -1 か 0 以上にしてください"),
    ]
    for is_wrong, message in checks:
        if is_wrong:
            raise gr.Error(message)
    assert image_path is not None
    return Request(
        prompt=prompt.strip(),
        image_path=Path(image_path),
        scale=scale,
        seconds=seconds,
        steps=int(steps),
        scheduler=scheduler,
        lora_strength=lora_strength,
        seed=int(seed),
    )


def describe_output(image_path: str | None, scale: float, seconds: float) -> str:
    """The output size for the current form, shown under the sliders."""
    if image_path is None or scale <= 0 or seconds <= 0:
        return ""
    width, height = output_size(Path(image_path), scale)
    return f"出力: {width}x{height}, {snap_length(seconds)} フレーム ({FPS} fps)"


def save_video(video_path: str | None) -> str:
    """Ask where to save the latest video with a native dialog, then copy it there.

    The dialog (zenity) opens on the machine running this script, which is also where the
    browser is when the page is opened locally.
    """
    if video_path is None:
        message = "保存する動画がありません"
        raise gr.Error(message)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suggested = RESULTS_DIR / Path(video_path).name
    result = subprocess.run(  # noqa: S603
        [  # noqa: S607
            "zenity",
            "--file-selection",
            "--save",
            "--confirm-overwrite",
            "--title=動画を保存",
            f"--filename={suggested}",
            "--file-filter=MP4 | *.mp4",
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
    destination = Path(result.stdout.strip())
    if destination.suffix == "":
        destination = destination.with_suffix(Path(video_path).suffix)
    shutil.copy(video_path, destination)
    return f"保存しました: {destination}"


class App:
    """The Gradio page, sending each generation to `server`."""

    def __init__(self, server: ComfyServer) -> None:
        self.server = server

    def on_generate(
        self,
        prompt: str,
        image_path: str | None,
        scale: float,
        seconds: float,
        steps: float,
        scheduler: str,
        lora_strength: float,
        seed: float,
    ) -> Iterator[tuple]:
        """Yield (status, video, latest video) updates while generating."""
        request = check_request(
            prompt, image_path, scale, seconds, steps, scheduler, lora_strength, seed
        )
        try:
            for item in self.server.generate(request):
                if isinstance(item, Path):
                    yield gr.skip(), str(item), str(item)
                else:
                    yield item, gr.skip(), gr.skip()
        except AssertionError as error:
            message = f"生成に失敗しました: {error}"
            raise gr.Error(message) from error

    def build(self) -> gr.Blocks:
        with gr.Blocks(title="MiniMax H3 image-to-video", delete_cache=(86400, 86400)) as demo:
            gr.Markdown("## MiniMax H3 image-to-video (Turbo LoRA v4, SageAttention)")
            # The path of the latest generated video, for the save button.
            latest_video = gr.State(None)
            with gr.Row():
                with gr.Column(scale=1):
                    image = gr.Image(label="画像 (最初のフレーム)", type="filepath", height=320)
                    prompt = gr.Textbox(
                        label="プロンプト",
                        lines=10,
                        placeholder="For the target video, at 0.00 seconds into the target "
                        "video, <Picture 1> is fully referenced. ...",
                    )
                    with gr.Row():
                        scale = gr.Slider(0.25, 1.5, value=0.75, step=0.05, label="scale")
                        seconds = gr.Slider(1, 15, value=5, step=0.5, label="秒数")
                    output_info = gr.Markdown("")
                    with gr.Accordion("詳細設定", open=False):
                        with gr.Row():
                            steps = gr.Slider(1, 20, value=8, step=1, label="steps")
                            scheduler = gr.Dropdown(
                                SCHEDULERS, value=SCHEDULERS[0], label="scheduler"
                            )
                        with gr.Row():
                            lora_strength = gr.Slider(
                                0.0, 2.0, value=1.0, step=0.05, label="LoRA strength"
                            )
                            seed = gr.Number(value=-1, precision=0, label="seed (-1 でランダム)")
                    generate_button = gr.Button("生成", variant="primary", size="lg")
                with gr.Column(scale=1):
                    # Fixed height: the player is fitted inside it instead of being scaled up
                    # to the column width (too tall for portrait videos).
                    video = gr.Video(
                        label="出力",
                        interactive=False,
                        autoplay=True,
                        loop=True,
                        height=VIDEO_HEIGHT,
                    )
                    status = gr.Markdown("")
                    save_button = gr.Button("保存…")

            inputs = [prompt, image, scale, seconds, steps, scheduler, lora_strength, seed]
            generate_button.click(self.on_generate, inputs, [status, video, latest_video])
            save_button.click(save_video, latest_video, status)
            for component in (image, scale, seconds):
                component.change(describe_output, [image, scale, seconds], output_info)
        return demo


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="run_i2v_") as work_dir:
        server = ComfyServer(args.comfy_port, Path(work_dir))
        print("starting ComfyUI...")
        try:
            server.start()
            print("ComfyUI is ready (the first generation also loads the models)")
            demo = App(server).build()
            # One generation at a time: ComfyUI runs them one by one anyway.
            demo.queue(default_concurrency_limit=1)
            demo.launch(
                server_name="127.0.0.1",
                server_port=args.port,
                inbrowser=True,
                theme=gr.themes.Soft(),
            )
        finally:
            server.stop()


if __name__ == "__main__":
    main()
