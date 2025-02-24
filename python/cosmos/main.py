"""A script to analyze token usage in images using a pretrained Cosmos tokenizer."""  # noqa: INP001

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from shutil import rmtree

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from cosmos_tokenizer.image_lib import ImageTokenizer
from PIL import Image, ImageDraw, ImageFont, ImageOps


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

    result_dir = Path("result")

    output_dir = result_dir / "output"
    rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(exist_ok=True, parents=True)

    model_name = "Cosmos-Tokenizer-DI8x8"
    encoder = ImageTokenizer(checkpoint_enc=f"{ckpt_dir}/{model_name}/encoder.jit")
    decoder = ImageTokenizer(checkpoint_dec=f"{ckpt_dir}/{model_name}/decoder.jit")

    results = []
    image_paths = sorted(data_dir.glob("*.jpg"))

    all_set = set()

    count_map = defaultdict(int)

    for image_path in image_paths:
        original_image = Image.open(image_path).convert("RGB")
        # (H, W) = (144, 256) にリサイズ
        original_image = ImageOps.fit(
            original_image,
            (256, 144),
            method=0,
            bleed=0.0,
            centering=(0.5, 0.5),
        )
        image = np.array(original_image).transpose(2, 0, 1) / 255.0
        image = torch.tensor(image, dtype=torch.bfloat16).unsqueeze(0).to("cuda")
        indices, codes = encoder.encode(image)

        reconstructed_tensor = decoder.decode(indices)
        reconstructed_image = reconstructed_tensor.to(torch.float32).squeeze(0).cpu().numpy()
        reconstructed_image = np.clip(reconstructed_image, 0, 1)
        reconstructed_image = (reconstructed_image * 255).astype(np.uint8).transpose(1, 2, 0)
        reconstructed_image = Image.fromarray(reconstructed_image)

        print(indices.shape, indices.dtype)

        # ユニークじゃないIDを列挙
        curr_count_map = defaultdict(int)
        for token_id in indices.cpu().numpy().flatten():
            curr_count_map[token_id] += 1
        non_unique_tokens = [token_id for token_id, count in curr_count_map.items() if count > 1]
        color_list = [
            # 基本色（半透明）
            (255, 0, 0, 128),  # 赤
            (0, 255, 0, 128),  # 緑
            (0, 0, 255, 128),  # 青
            (255, 255, 0, 128),  # 黄
            (0, 255, 255, 128),  # シアン
            (255, 0, 255, 128),  # マゼンタ
            (255, 255, 255, 128),  # 白
            (0, 0, 0, 128),  # 黒
            # 中間色
            (255, 128, 0, 128),  # オレンジ
            (128, 255, 0, 128),  # 黄緑
            (0, 255, 128, 128),  # スプリンググリーン
            (0, 128, 255, 128),  # スカイブルー
            (128, 0, 255, 128),  # パープル
            (255, 0, 128, 128),  # ピンク
            # 暗め
            (128, 0, 0, 128),  # ダークレッド
            (0, 128, 0, 128),  # ダークグリーン
            (0, 0, 128, 128),  # ダークブルー
            (128, 128, 0, 128),  # オリーブ
            (0, 128, 128, 128),  # ティール
            (128, 0, 128, 128),  # パープル
            # 明るめ
            (255, 128, 128, 128),  # ライトレッド
            (128, 255, 128, 128),  # ライトグリーン
            (128, 128, 255, 128),  # ライトブルー
            (255, 255, 128, 128),  # ライトイエロー
            (128, 255, 255, 128),  # ライトシアン
            (255, 128, 255, 128),  # ライトマゼンタ
        ]
        for k, non_unique_token in enumerate(non_unique_tokens):
            print(f"Non-unique token: {non_unique_token}")
            # recon画像に色付けする
            for i in range(indices.shape[1]):
                for j in range(indices.shape[2]):
                    if indices[0, i, j] == non_unique_token:
                        # 半透明の赤い長方形をオーバーレイ
                        overlay = Image.new("RGBA", reconstructed_image.size, (0, 0, 0, 0))
                        draw = ImageDraw.Draw(overlay)
                        draw.rectangle(
                            (j * 8, i * 8, (j + 1) * 8, (i + 1) * 8),
                            fill=color_list[k % len(color_list)],
                        )
                        print(k, i, j)
                        # RGBAに変換して合成
                        reconstructed_image = reconstructed_image.convert("RGBA")
                        reconstructed_image = Image.alpha_composite(reconstructed_image, overlay)
        # 左上に文字追加
        original_image = add_text_to_image(original_image, "Original Image")
        reconstructed_image = add_text_to_image(reconstructed_image, "Reconstructed Image")

        # 元画像と再構築画像を左右に並べて保存
        comparison_image = Image.new("RGB", (original_image.width * 2, original_image.height))
        comparison_image.paste(original_image, (0, 0))
        comparison_image.paste(reconstructed_image, (original_image.width, 0))

        # 画像を保存
        output_path = output_dir / f"{image_path.stem}_comparison.jpg"
        comparison_image.save(output_path)

        token_ids = indices.cpu().numpy().flatten()

        all_set.update(token_ids)

        for token_id in token_ids:
            count_map[token_id] += 1

        # 使用率
        total_tokens = len(token_ids)
        unique_tokens = len(set(token_ids))
        unique_rate = unique_tokens / total_tokens

        print(f"{total_tokens=}, {unique_tokens=}, {unique_rate=}")

        stats = {
            "total_tokens": total_tokens,
            "unique_tokens": unique_tokens,
            "unique_rate": unique_rate,
        }

        results.append({"image": str(image_path), **stats})

    print(f"{len(all_set)=}")

    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv(result_dir / "token_usage_analysis.csv", index=False)
    plt.plot(results_df["unique_rate"])
    plt.xlabel("Image")
    plt.ylabel("Unique Rate(=unique_tokens/total_tokens)")
    plt.ylim(0, 1)
    plt.savefig(result_dir / "unique_rate.png", bbox_inches="tight", pad_inches=0.05)
    plt.close()

    sys.exit(0)

    # count_mapをソート
    count_map = sorted(count_map.items(), key=lambda x: x[1], reverse=True)
    plt.bar(range(len(count_map)), [x[1] for x in count_map])
    plt.savefig(result_dir / "histogram.png", bbox_inches="tight", pad_inches=0.05, dpi=300)
