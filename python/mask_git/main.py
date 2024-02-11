import argparse
import cv2
from networks import VQVAE
import jax
import jax.numpy as jnp
from jax import random, jit
import optax
from flax.training import train_state, checkpoints
from data_loader import DataLoader
import os
import numpy as np
from datetime import datetime
from tensorboardX import SummaryWriter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("--epoch", type=int, default=1000)
    return parser.parse_args()


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

    batch_size = 100
    train_loader = DataLoader(train_data_dir, batch_size)
    test_loader = DataLoader(test_data_dir, batch_size, max_num=5)
    train_loader_for_test = DataLoader(train_data_dir, batch_size, max_num=5)

    model = VQVAE(train=True)
    rng = random.PRNGKey(0)
    params = model.init(rng, jnp.ones((1, 96, 96, 3)))

    num_epochs = args.epoch
    step_num_per_epoch = train_loader.step_num_per_epoch()
    num_steps = step_num_per_epoch * num_epochs
    print(f"{num_steps=}")

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=1e-3,
        warmup_steps=int(num_steps * 0.01),
        decay_steps=num_steps,
        end_value=0.0,
    )

    optimizer = optax.chain(
        optax.clip(1.0),
        optax.adamw(learning_rate=schedule, eps=1.5e-4, weight_decay=0.001),
    )

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )

    now = datetime.now()
    datetime_str = now.strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.abspath(f"./result_test_{datetime_str}/")
    print(f"{save_dir=}")
    writer = SummaryWriter(save_dir)
    global_step = 0

    for epoch in range(num_epochs):
        train_loader.shuffle()
        loss_sum = 0
        loss_num = 0
        for batch in train_loader:
            state, loss = train_step(state, batch)
            loss_sum += batch.shape[0] * loss
            loss_num += batch.shape[0]
            global_step += 1
            writer.add_scalar("train/loss", loss, global_step)
        print(f'Epoch {epoch}, Loss: {loss_sum / loss_num:.4f}')

        curr_save_dir = f"{save_dir}/{epoch:04d}"
        os.makedirs(curr_save_dir, exist_ok=True)
        for name, loader in zip(["train", "test"], [train_loader_for_test, test_loader]):
            for batch in loader:
                reconstructions = test_step(state, batch)
                for i, (original, reconstructed) in enumerate(zip(batch, reconstructions)):
                    save_path = f"{curr_save_dir}/reconstruction_{name}_{i:04d}.png"
                    combined_image = np.hstack((original, reconstructed))
                    combined_image_bgr = cv2.cvtColor(
                        combined_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(save_path, combined_image_bgr * 255)
                    combined_image = combined_image.transpose(2, 0, 1)
                    combined_image = combined_image * 255
                    combined_image = combined_image.astype(np.uint8)
                    writer.add_image(
                        f'test/reconstruction_{name}_{i:04d}', combined_image, global_step=epoch)

    checkpoints.save_checkpoint(ckpt_dir=save_dir, target=state, step=global_step)
