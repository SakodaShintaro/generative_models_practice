import jax
import jax.numpy as jnp

# ruff: noqa: ANN001, ANN201, ANN202, ANN204, ANN205, ERA001, N816


def f(S, w, z, b, v, k):
    S = S * w.mT - S @ z * b.mT + v * k.mT
    return S


if __name__ == "__main__":
    HEAD_NUM = 16
    HEAD_SIZE = 64
    TIMESTEP = 10

    rng = jax.random.PRNGKey(0)
    curr_S = jax.random.normal(rng, (HEAD_NUM, HEAD_SIZE, HEAD_SIZE))
    w = jax.random.normal(rng, (TIMESTEP, HEAD_NUM, HEAD_SIZE, 1))
    z = jax.random.normal(rng, (TIMESTEP, HEAD_NUM, HEAD_SIZE, 1))
    b = jax.random.normal(rng, (TIMESTEP, HEAD_NUM, HEAD_SIZE, 1))
    v = jax.random.normal(rng, (TIMESTEP, HEAD_NUM, HEAD_SIZE, 1))
    k = jax.random.normal(rng, (TIMESTEP, HEAD_NUM, HEAD_SIZE, 1))

    for i in range(TIMESTEP):
        curr_S = f(curr_S, w[i], z[i], b[i], v[i], k[i])
        print(f"{curr_S.shape=}")

    loss = jnp.mean(curr_S)
    print(f"{loss=}")
