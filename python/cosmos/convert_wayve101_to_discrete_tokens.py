"""A script to convert wayve101 dataset to discrete tokens using a pretrained Cosmos tokenizer."""

import argparse
from pathlib import Path

import numpy as np
import torch
from cosmos_tokenizer.image_lib import ImageTokenizer
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("ckpt_dir", type=Path)
    return parser.parse_args()


def add_text_to_image(image: Image.Image, text: str) -> Image.Image:
    """画像に文字を追加する関数."""
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    # 文字を描画（左上に配置）
    draw.text((5, 5), text, font=font, fill=(255, 255, 255))  # 白色で描画

    return image


if __name__ == "__main__":
    args = parse_args()
    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir

    result_dir = data_dir.parent / "wayve101_tokens"
    result_dir.mkdir(exist_ok=True, parents=True)
    print(f"{result_dir=}")

    COMPRESS_SCALE = 8
    model_name = f"Cosmos-Tokenizer-DI{COMPRESS_SCALE}x{COMPRESS_SCALE}"
    encoder = ImageTokenizer(checkpoint_enc=f"{ckpt_dir}/{model_name}/encoder.jit")
    decoder = ImageTokenizer(checkpoint_dec=f"{ckpt_dir}/{model_name}/decoder.jit")
    encoder.eval()
    decoder.eval()
    encoder.to("cuda")
    decoder.to("cuda")

    TARGET_SUBDIR = "front-forward"

    subdir_list = sorted(data_dir.glob("scene_*"))
    subdir_list = [
        subdir for subdir in subdir_list if (data_dir / subdir / "images" / TARGET_SUBDIR).exists()
    ]

    for subdir in tqdm(subdir_list):
        iamge_path_list = sorted((data_dir / subdir / "images" / TARGET_SUBDIR).glob("*.jpeg"))
        curr_tokens_result_dir = result_dir / subdir.name / "tokens" / TARGET_SUBDIR
        curr_tokens_result_dir.mkdir(exist_ok=True, parents=True)
        curr_reconstruct_result_dir = result_dir / subdir.name / "reconstruct" / TARGET_SUBDIR
        curr_reconstruct_result_dir.mkdir(exist_ok=True, parents=True)
        for image_path in iamge_path_list:
            original_image = Image.open(image_path).convert("RGB")
            ORIGINAL_H = 1080  # 16の倍数ではない
            ORIGINAL_W = 1920  # 16 * 120
            RESIZED_H = 128
            RESIZED_W = 256
            assert RESIZED_H % COMPRESS_SCALE == 0
            assert RESIZED_W % COMPRESS_SCALE == 0
            resized_image = original_image.resize((RESIZED_W, RESIZED_H), Image.BILINEAR)
            image = np.array(resized_image).transpose(2, 0, 1) / 255.0
            image = torch.tensor(image, dtype=torch.bfloat16).unsqueeze(0).to("cuda")
            with torch.no_grad():
                indices, codes = encoder.encode(image)
            indices_len = indices.shape[-2] * indices.shape[-1]
            with torch.no_grad():
                reconstructed_tensor = decoder.decode(indices)
            reconstructed_image = reconstructed_tensor.to(torch.float32).squeeze(0).cpu().numpy()
            reconstructed_image = np.clip(reconstructed_image, 0, 1)
            reconstructed_image = (reconstructed_image * 255).astype(np.uint8).transpose(1, 2, 0)
            reconstructed_image = Image.fromarray(reconstructed_image)
            resized_image = add_text_to_image(resized_image, "Original Image")
            reconstructed_image = add_text_to_image(reconstructed_image, "Reconstructed Image")
            comparison_image = Image.new("RGB", (resized_image.width * 2, resized_image.height))
            comparison_image.paste(resized_image, (0, 0))
            comparison_image.paste(reconstructed_image, (resized_image.width, 0))
            reconstruct_path = curr_reconstruct_result_dir / f"{image_path.stem}.jpeg"
            comparison_image.save(reconstruct_path)
            indices_path = curr_tokens_result_dir / f"{image_path.stem}.csv"
            indices = indices.cpu().numpy().flatten().reshape(1, -1)
            np.savetxt(indices_path, indices, delimiter=",", fmt="%d")
