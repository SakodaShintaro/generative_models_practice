import argparse
from pathlib import Path
from data_loader import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--batch_size", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_dir = args.data_dir
    batch_size = args.batch_size

    data_loader = DataLoader(data_dir, batch_size=4)
    for batch_images in data_loader:
        print(batch_images.shape)
