import argparse
import os

import cv2
import numpy as np
import torch
from data_loader import DataLoader
from network import VQVAE
from torch import nn, optim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    return parser.parse_args()


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    return img


def train_step(model, optimizer, batch):
    model.train()
    optimizer.zero_grad()
    reconstructions = model(batch)
    loss = nn.MSELoss()(reconstructions, batch)
    loss.backward()
    optimizer.step()
    return loss.item()


def test_step(model, batch):
    model.eval()
    with torch.no_grad():
        return model(batch)


if __name__ == "__main__":
    args = parse_args()
    train_data_dir = f"{args.data_dir}/train"
    test_data_dir = f"{args.data_dir}/test"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_data_dir, 64)
    test_loader = DataLoader(test_data_dir, 64)
    test_batch = next(iter(test_loader)).to(device)

    model = VQVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 100
    save_dir = "./result_test/"

    for epoch in range(num_epochs):
        sum_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            loss = train_step(model, optimizer, batch)
            sum_loss += loss * len(batch)
        loss = sum_loss / len(train_loader._image_path_list)
        print(f"Epoch {epoch}, Loss: {loss}")

        curr_save_dir = f"{save_dir}/{epoch:04d}"
        os.makedirs(curr_save_dir, exist_ok=True)
        count = 0

        batch = test_batch.to(device)
        reconstructions = test_step(model, batch)
        reconstructions = reconstructions.cpu().numpy()
        batch = batch.cpu().numpy()

        for original, reconstructed in zip(batch, reconstructions):
            # (C, H, W) -> (H, W, C)
            original_img = np.transpose(original, (1, 2, 0))
            reconstructed_img = np.transpose(reconstructed, (1, 2, 0))

            combined_image = np.hstack((original_img, reconstructed_img))
            combined_image_bgr = cv2.cvtColor(
                (combined_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
            )
            save_path = f"{curr_save_dir}/reconstruction_{count:08d}.png"
            cv2.imwrite(save_path, combined_image_bgr)
            count += 1
