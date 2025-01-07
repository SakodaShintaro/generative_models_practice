"""A script to analyze token usage in images using a pretrained Cosmos tokenizer."""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from cosmos_tokenizer.image_lib import ImageTokenizer
from PIL import Image


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

    model_name = "Cosmos-Tokenizer-DI8x8"
    input_tensor = torch.randn(1, 3, 512, 512).to("cuda").to(torch.bfloat16)  # [B, C, H, W]
    encoder = ImageTokenizer(checkpoint_enc=f"{ckpt_dir}/{model_name}/encoder.jit")
    (indices, codes) = encoder.encode(input_tensor)
    print(indices.shape, codes.shape)
    print(indices.dtype, codes.dtype)
    torch.testing.assert_close(indices.shape, (1, 64, 64))
    torch.testing.assert_close(codes.shape, (1, 6, 64, 64))

    # The input tensor can be reconstructed by the decoder as:
    decoder = ImageTokenizer(checkpoint_dec=f"{ckpt_dir}/{model_name}/decoder.jit")
    reconstructed_tensor = decoder.decode(indices)
    torch.testing.assert_close(reconstructed_tensor.shape, input_tensor.shape)

    results = []
    image_paths = list(data_dir.glob("*.jpg"))

    all_set = set()

    count_map = defaultdict(int)

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image = np.array(image).transpose(2, 0, 1) / 255.0
        image = torch.tensor(image, dtype=torch.bfloat16).unsqueeze(0).to("cuda")
        indices, codes = encoder.encode(image)
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
