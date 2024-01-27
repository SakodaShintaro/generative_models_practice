import argparse
import cv2
from network import VQVAE
import jax
import jax.numpy as jnp
from jax import random, jit
import optax
from flax.training import train_state
from data_loader import DataLoader
import os
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    return parser.parse_args()


def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    return img


def loss_fn(params, batch):
    reconstructions = model.apply(params, batch)
    loss = jnp.mean((reconstructions - batch) ** 2)
    return loss


@jit
def train_step(state, batch):
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params, batch)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jit
def test_step(state, batch):
    return model.apply(state.params, batch)


if __name__ == "__main__":
    args = parse_args()
    train_data_dir = f"{args.data_dir}/train"
    test_data_dir = f"{args.data_dir}/test"

    train_loader = DataLoader(train_data_dir, 64)
    test_loader = DataLoader(test_data_dir, 64)

    model = VQVAE()
    rng = random.PRNGKey(0)
    params = model.init(rng, jnp.ones((1, 96, 96, 3)))

    optimizer = optax.adam(learning_rate=1e-3)

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )

    num_epochs = 10

    save_dir = "./result_test/"

    for epoch in range(num_epochs):
        for batch in train_loader:
            state, loss = train_step(state, batch)
            print(loss)
        print(f'Epoch {epoch}, Loss: {loss}')

        curr_save_dir = f"{save_dir}/{epoch:04d}"
        os.makedirs(curr_save_dir, exist_ok=True)
        count = 0
        for batch in test_loader:
            reconstructions = test_step(state, batch)
            for (original, reconstructed) in zip(batch, reconstructions):
                save_path = f"{curr_save_dir}/reconstruction_{count:08d}.png"
                combined_image = np.hstack((original, reconstructed))
                combined_image_bgr = cv2.cvtColor(
                    combined_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, combined_image_bgr * 255)
                count += 1
