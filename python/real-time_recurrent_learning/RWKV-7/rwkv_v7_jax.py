# Reference) https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v7_numpy.py

########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
# RWKV-7 in numpy, by https://github.com/johanwind

import jax
import jax.numpy as jnp
from torch import load as torch_load

# ruff: noqa: ANN001, ANN201, N802, ERA001


def layer_norm(x, w, b):
    return (x - x.mean()) / (x.var() + 1e-5) ** 0.5 * w + b


def group_norm(x, w, b):
    return (
        (x - x.mean(axis=1, keepdims=1)) / (x.var(axis=1, keepdims=1) + 64e-5) ** 0.5
    ).flatten() * w + b


def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


def time_mixing(x, v0, last_x, S, params):
    # m_{}は(N_EMBD,)
    # w_bias.shape=(1024,)
    # Ww1.shape=(1024, 64), Ww2.shape=(64, 1024)
    # Wa1.shape=(1024, 64), Wa2.shape=(64, 1024)
    # a_bias.shape=(1024,)
    # Wg1.shape=(1024, 128), Wg2.shape=(128, 1024)
    mr, mw, mk, mv, ma, mg, w_bias, r_k, Ww1, Ww2, Wa1, Wa2, a_bias, Wg1, Wg2 = params[:15]

    # k_k.shape=(1024,), k_a.shape=(1024,)
    # Wr.shape=(1024, 1024)
    # Wk.shape=(1024, 1024)
    # Wv.shape=(1024, 1024)
    # Wo.shape=(1024, 1024)
    # ln_w.shape=(1024,), ln_b.shape=(1024,)
    k_k, k_a, Wr, Wk, Wv, Wo, ln_w, ln_b = params[-8:]

    # (N_EMBD,).
    xr, xw, xk, xv, xa, xg = [x + m * (last_x - x) for m in [mr, mw, mk, mv, ma, mg]]

    r = Wr @ xr  # (1024,)
    w = jnp.exp(-sigmoid(jnp.tanh(xw @ Ww1) @ Ww2 + w_bias) / jnp.e**0.5)  # (1024,)
    k = Wk @ xk  # (1024,)
    v = Wv @ xv  # (1024,)
    if v0 is None:
        v0 = v
    else:
        # Wv2.shape=(32, 1024), Wv1.shape=(1024, 32), v_bias.shape=(1024,)
        Wv2, Wv1, v_bias = params[15:18]
        v += (v0 - v) * sigmoid(xv @ Wv1 @ Wv2 + v_bias)
    a = sigmoid(xa @ Wa1 @ Wa2 + a_bias)  # (1024,)
    g = sigmoid(xg @ Wg1) @ Wg2  # (1024,)
    kk = k * k_k
    k += k * (a - 1) * k_a

    r, w, k, v, kk, a, r_k = [i.reshape(N_HEAD, HEAD_SIZE, 1) for i in [r, w, k, v, kk, a, r_k]]
    kk /= jnp.maximum(jnp.linalg.norm(kk, axis=1, keepdims=1), 1e-12)

    S = S * w.mT - S @ kk * (kk * a).mT + v * k.mT  # S.shape=(N_HEAD, HEAD_SIZE, HEAD_SIZE)
    y = S @ r  # y.shape=(N_HEAD, HEAD_SIZE, 1)

    y = group_norm(y, ln_w, ln_b)
    y += ((r * k * r_k).sum(axis=1, keepdims=1) * v).flatten()
    return Wo @ (y * g), v0, x, S


def channel_mixing(x, last_x, mix, Wk, Wv):
    # x.shape=(1024,), last_x.shape=(1024,), mix.shape=(1024,)
    # Wk.shape=(4096, 1024), Wv.shape=(1024, 4096)
    k = Wk @ (x + mix * (last_x - x))
    v = Wv @ jnp.maximum(k, 0) ** 2
    return v, x


def RWKV7(params, token: int, state: tuple):
    """
    state = (
        jnp.zeros((N_LAYER, 2, N_EMBD), dtype=jnp.float32),
        jnp.zeros((N_LAYER, N_HEAD, HEAD_SIZE, HEAD_SIZE), dtype=jnp.float32),
    )
    """
    x = params("emb")[0][token]  # (N_EMBD,)
    x = layer_norm(x, *params("blocks.0.ln0"))  # (N_EMBD,)

    v0 = None
    state_0, state_1 = state
    # state_0: (N_LAYER, 2, N_EMBD).
    #   time_mixing, channel_mixingそれぞれのlast_x
    # state_1: (N_LAYER, N_HEAD, HEAD_SIZE, HEAD_SIZE).

    for i in range(N_LAYER):
        x_ = layer_norm(x, *params(f"blocks.{i}.ln1"))  # (N_EMBD,)
        dx, v0, new_state_0_i_0, new_state_1_i = time_mixing(
            x_, v0, state_0[i, 0], state_1[i], params(f"blocks.{i}.att")
        )
        state_0 = state_0.at[i, 0].set(new_state_0_i_0)
        state_1 = state_1.at[i].set(new_state_1_i)
        x = x + dx  # (N_EMBD,)

        x_ = layer_norm(x, *params(f"blocks.{i}.ln2"))  # (N_EMBD,)
        dx, new_state_0_i_1 = channel_mixing(x_, state_0[i, 1], *params(f"blocks.{i}.ffn"))
        state_0 = state_0.at[i, 1].set(new_state_0_i_1)
        x = x + dx  # (N_EMBD,)

    x = layer_norm(x, *params("ln_out"))
    logits = params("head")[0] @ x

    return logits, (state_0, state_1)


# Verification

# Available at https://huggingface.co/BlinkDL/rwkv-7-world/resolve/main/RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth
MODEL_FILE = "./RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth"
N_LAYER = 24
N_EMBD = 1024
HEAD_SIZE = 64
N_HEAD = N_EMBD // HEAD_SIZE

key = jax.random.PRNGKey(seed=42)  # 乱数生成のためのキーを作成
tokens = jax.random.randint(key, shape=(10,), minval=0, maxval=50277, dtype=jnp.int32)

weights = torch_load(MODEL_FILE, map_location="cpu", weights_only=True)
weights = {k: v.squeeze().float().numpy() for k, v in weights.items()}


def params(prefix):
    return [weights[key] for key in weights if key.startswith(prefix)]


state = (
    jnp.zeros((N_LAYER, 2, N_EMBD), dtype=jnp.float32),
    jnp.zeros((N_LAYER, N_HEAD, HEAD_SIZE, HEAD_SIZE), dtype=jnp.float32),
)
for token in tokens:
    minimal_logits, state = RWKV7(params, token, state)
