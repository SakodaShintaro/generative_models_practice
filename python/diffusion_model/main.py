"""main."""

import argparse
from pathlib import Path

import torch
from data_loader import DataLoader
from models import DiT_S_4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--batch_size", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_dir = args.data_dir
    batch_size = args.batch_size

    data_loader = DataLoader(data_dir, batch_size)

    model = DiT_S_4(input_size=96, in_channels=3)
    print(model)

    for batch_images in data_loader:
        b, c, h, w = batch_images.shape
        x = torch.tensor(batch_images, dtype=torch.float32)
        t = torch.randn(batch_size)
        y = torch.randint(0, 10, (batch_size,))
        print(x.shape, t.shape, y.shape)
        out = model(x, t, y)
        print(out.shape)
        break
