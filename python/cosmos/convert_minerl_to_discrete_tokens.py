"""A script to convert minerl dataset to discrete tokens using a pretrained Cosmos tokenizer.

[Before]
$ tree ${DARA_ROOT}/0000 -L 1
${DATA_ROOT}/0000
├── action
├── inventory
├── obs
└── reward.csv

for 0000 ... 0099
`obs` has 18,000 images (360, 640)

[After]
$ tree ${DARA_ROOT}/0000 -L 1
${DATA_ROOT}/0000
├── action
├── inventory
├── obs
├── tokens
├── resized
├── reconstructed
├── comparison
└── reward.csv

"""

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

    COMPRESS_SCALE = 8
    model_name = f"Cosmos-Tokenizer-DI{COMPRESS_SCALE}x{COMPRESS_SCALE}"
    tokenizer = ImageTokenizer(
        checkpoint_enc=f"{ckpt_dir}/{model_name}/encoder.jit",
        checkpoint_dec=f"{ckpt_dir}/{model_name}/decoder.jit",
    )
    tokenizer.eval()
    tokenizer.to("cuda")

    subdir_list = sorted(data_dir.glob("*"))
    subdir_list = [subdir for subdir in subdir_list if (data_dir / subdir / "obs").exists()]

    ORIGINAL_H = 360  # 8 * 45
    ORIGINAL_W = 640  # 8 * 80
    RESIZED_H = 72  # 8 * 9
    RESIZED_W = 128  # 8 * 16
    assert RESIZED_H % COMPRESS_SCALE == 0
    assert RESIZED_W % COMPRESS_SCALE == 0

    BATCH_SIZE = 2 ** 8

    for subdir in tqdm(subdir_list):
        print(subdir)
        iamge_path_list = sorted((subdir / "obs").glob("*.png"))
        curr_tokens_result_dir = subdir / "tokens"
        curr_tokens_result_dir.mkdir(exist_ok=True, parents=True)
        curr_resized_result_dir = subdir / "resized"
        curr_resized_result_dir.mkdir(exist_ok=True, parents=True)
        curr_reconstructed_result_dir = subdir / "reconstructed"
        curr_reconstructed_result_dir.mkdir(exist_ok=True, parents=True)
        curr_comparison_result_dir = subdir / "comparison"
        curr_comparison_result_dir.mkdir(exist_ok=True, parents=True)
        for i in tqdm(range(0, len(iamge_path_list), BATCH_SIZE)):
            image_path_batch = iamge_path_list[i : i + BATCH_SIZE]
            original_image_batch = [
                Image.open(image_path).convert("RGB") for image_path in image_path_batch
            ]
            resized_image_batch = [
                original_image.resize((RESIZED_W, RESIZED_H), Image.BILINEAR)
                for original_image in original_image_batch
            ]
            image_batch = np.array(
                [
                    np.array(resized_image).transpose(2, 0, 1) / 255.0
                    for resized_image in resized_image_batch
                ]
            )
            image_batch = torch.tensor(image_batch, dtype=torch.bfloat16).to("cuda")
            with torch.no_grad():
                indices_batch, codes = tokenizer.encode(image_batch)
            with torch.no_grad():
                reconstructed_tensor = tokenizer.decode(indices_batch)
            reconstructed_image_batch = reconstructed_tensor.to(torch.float32).cpu().numpy()
            reconstructed_image_batch = np.clip(reconstructed_image_batch, 0, 1)
            reconstructed_image_batch = (reconstructed_image_batch * 255).astype(np.uint8)
            reconstructed_image_batch = [
                Image.fromarray(reconstructed_image.transpose(1, 2, 0))
                for reconstructed_image in reconstructed_image_batch
            ]

            for image_path, resized_image, reconstructed_image, indices in zip(
                image_path_batch, resized_image_batch, reconstructed_image_batch, indices_batch
            ):
                resized_path = curr_resized_result_dir / f"{image_path.stem}.jpeg"
                resized_image.save(resized_path)

                reconstructed_path = curr_reconstructed_result_dir / f"{image_path.stem}.jpeg"
                reconstructed_image.save(reconstructed_path)

                resized_image = add_text_to_image(resized_image, "Original Image")
                reconstructed_image = add_text_to_image(reconstructed_image, "Reconstructed Image")
                comparison_image = Image.new("RGB", (resized_image.width * 2, resized_image.height))
                comparison_image.paste(resized_image, (0, 0))
                comparison_image.paste(reconstructed_image, (resized_image.width, 0))
                comparison_path = curr_comparison_result_dir / f"{image_path.stem}.jpeg"
                comparison_image.save(comparison_path)

                indices_path = curr_tokens_result_dir / f"{image_path.stem}.csv"
                indices = indices.cpu().numpy().flatten()
                np.savetxt(indices_path, indices, delimiter=",", fmt="%d")
