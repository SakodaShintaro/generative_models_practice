import argparse
import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--scheduler", type=str, default="beta")
    parser.add_argument("--lora_strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=8189)
    parser.add_argument("--out_dir", type=Path, default=SCRIPT_DIR / "results")
    return parser.parse_args()


SCRIPT_DIR = Path(__file__).resolve().parent
COMFY_DIR = SCRIPT_DIR / "ComfyUI"

# MiniMaxH3ImageToVideo requires width/height to be multiples of 32
CANVAS_MULTIPLE = 32

UNET_NAME = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
CLIP_NAME = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE_NAME = "minimax_h3_video_vae_int8_convrot.safetensors"
AUDIO_VAE_NAME = "minimax_h3_audio_vae_fp32.safetensors"
LORA_NAME = "minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors"


def round_to_multiple(value: float, multiple: int) -> int:
    return max(multiple, round(value / multiple) * multiple)


def snap_length(seconds: float) -> int:
    """Convert seconds at 24 fps to the 17k+5 frame grid (same formula as the official template)."""
    n = max(5, round(seconds * 24))
    return n + (5 - (n % 17)) % 17


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
            "inputs": {"images": ["decode_video", 0], "audio": ["decode_audio", 0], "fps": 24},
        },
        "save": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["create_video", 0],
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


def wait_server_ready(base_url: str, proc: subprocess.Popen, timeout_sec: float) -> None:
    start = time.time()
    while time.time() - start < timeout_sec:
        assert proc.poll() is None, "ComfyUI exited during startup (see output above)"
        try:
            http_json(f"{base_url}/system_stats", None)
        except (urllib.error.URLError, ConnectionError):
            time.sleep(2)
            continue
        return
    msg = "ComfyUI did not become ready in time"
    raise TimeoutError(msg)


def wait_prompt_done(base_url: str, prompt_id: str, proc: subprocess.Popen) -> dict:
    start = time.time()
    while True:
        assert proc.poll() is None, "ComfyUI exited during generation (see output above)"
        history = http_json(f"{base_url}/history/{prompt_id}", None)
        if prompt_id in history:
            entry = history[prompt_id]
            status = entry["status"]
            assert status["status_str"] == "success", json.dumps(status, indent=2)
            print(f"done in {time.time() - start:.1f} sec")
            return entry
        time.sleep(5)


def main() -> None:
    args = parse_args()

    assert args.image.is_file(), args.image
    assert args.scale > 0, args.scale

    with Image.open(args.image) as img:
        image_width, image_height = img.size
    width = round_to_multiple(image_width * args.scale, CANVAS_MULTIPLE)
    height = round_to_multiple(image_height * args.scale, CANVAS_MULTIPLE)
    print(f"input={image_width}x{image_height} scale={args.scale} -> output={width}x{height}")

    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005
    tag = (
        f"{start_time}_{args.image.stem}_{width}x{height}_{args.seconds:g}s_"
        f"{args.steps}step_{args.scheduler}_seed{args.seed}"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # LoadImage reads from ComfyUI/input
    input_dir = COMFY_DIR / "input"
    input_dir.mkdir(exist_ok=True)
    shutil.copy(args.image, input_dir / args.image.name)

    cmd = [
        str(SCRIPT_DIR / ".venv/bin/python"),
        "main.py",
        "--listen",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--disable-auto-launch",
        "--use-sage-attention",
    ]
    base_url = f"http://127.0.0.1:{args.port}"
    # ComfyUI's stdout/stderr go straight to this terminal (no log file)
    proc = subprocess.Popen(cmd, cwd=COMFY_DIR)  # noqa: S603
    try:
        wait_server_ready(base_url, proc, timeout_sec=300)
        length = snap_length(args.seconds)
        print(f"tag={tag} frames={length}")
        workflow = build_workflow(
            image_name=args.image.name,
            prompt=args.prompt,
            width=width,
            height=height,
            length=length,
            steps=args.steps,
            scheduler=args.scheduler,
            lora_strength=args.lora_strength,
            seed=args.seed,
            output_prefix=f"minimax_h3_i2v/{tag}",
        )
        res = http_json(f"{base_url}/prompt", {"prompt": workflow, "client_id": str(uuid.uuid4())})
        assert "prompt_id" in res, res
        entry = wait_prompt_done(base_url, res["prompt_id"], proc)
        for node_output in entry["outputs"].values():
            for item in node_output["images"]:
                src = COMFY_DIR / "output" / item["subfolder"] / item["filename"]
                dst = args.out_dir / f"{tag}{src.suffix}"
                shutil.move(src, dst)
                print(f"saved {dst}")
    finally:
        proc.terminate()
        proc.wait()


if __name__ == "__main__":
    main()
