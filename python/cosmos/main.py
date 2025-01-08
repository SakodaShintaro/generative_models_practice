"""A script to analyze token usage in images using a pretrained Cosmos tokenizer."""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from cosmos_tokenizer.image_lib import ImageTokenizer
from PIL import Image, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("ckpt_dir", type=Path)
    return parser.parse_args()


# トークン化と統計の計算
def analyze_image_tokens(token_ids):
    # token_ids(1, 112, 200), torch.int32

    token_ids = token_ids.cpu().numpy().flatten()

    # 使用率
    total_tokens = len(token_ids)
    unique_tokens = len(set(token_ids))
    usage_rate = unique_tokens / total_tokens

    print(f"{total_tokens=}, {unique_tokens=}, {usage_rate=}")

    return {
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,
        "usage_rate": usage_rate,
    }


if __name__ == "__main__":
    args = parse_args()
    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

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
            original_image, (256, 144), method=0, bleed=0.0, centering=(0.5, 0.5)
        )
        image = np.array(original_image).transpose(2, 0, 1) / 255.0
        image = torch.tensor(image, dtype=torch.bfloat16).unsqueeze(0).to("cuda")
        indices, codes = encoder.encode(image)

        reconstructed_tensor = decoder.decode(indices)
        reconstructed_image = reconstructed_tensor.to(torch.float32).squeeze(0).cpu().numpy()
        reconstructed_image = (reconstructed_image * 255).astype(np.uint8).transpose(1, 2, 0)
        reconstructed_image = Image.fromarray(reconstructed_image)

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

        stats = analyze_image_tokens(indices)
        results.append({"image": str(image_path), **stats})

    print(f"{len(all_set)=}")

    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv("token_usage_analysis.csv", index=False)
    plt.plot(results_df["usage_rate"])
    plt.xlabel("Image")
    plt.ylabel("Usage Rate(=unique_tokens/total_tokens)")
    plt.savefig("usage_rate.png", bbox_inches="tight", pad_inches=0.05)
    plt.close()

    # count_mapをソート
    count_map = sorted(count_map.items(), key=lambda x: x[1], reverse=True)
    print(count_map)
    plt.bar(range(len(count_map)), [x[1] for x in count_map])
    plt.savefig("histogram.png", bbox_inches="tight", pad_inches=0.05)
